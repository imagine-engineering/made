use super::*;
use gpui::{TestAppContext, VisualTestContext};
use project::jvm_test_provider::JvmTestBuildTool;
use settings::WorktreeId;
use std::time::{Duration, Instant};

pub struct JvmWorkflowDebugContext {
    pub project: Entity<Project>,
    pub workspace: Entity<Workspace>,
    pub project_root: std::path::PathBuf,
    pub worktree_id: WorktreeId,
}

pub struct JvmWorkflowDebugTarget {
    pub session: Entity<project::debugger::session::Session>,
    pub process_id: u32,
}

pub struct JvmWorkflowDebugHooks {
    // Debug UI depends on Tasks UI, so the Debug owner supplies this phase to
    // the cached test fixture instead of introducing a reverse dependency.
    pub initialize: fn(&mut App),
    pub run: for<'a> fn(
        JvmWorkflowDebugContext,
        &'a mut VisualTestContext,
    ) -> futures::future::LocalBoxFuture<'a, JvmWorkflowDebugTarget>,
}

pub async fn run_cached_gradle_workflow(cx: &mut TestAppContext, hooks: JvmWorkflowDebugHooks) {
    selected_java_case_with_debug_hooks(
        cx,
        JvmTestBuildTool::Gradle,
        JavaTestOutcome::CombinedGradleWorkflow,
        false,
        Some(hooks),
    )
    .await;
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum JavaTestOutcome {
    Pass,
    FailAndRerun,
    Cancel,
    RestrictTrust,
    RestrictJavaProcess,
    CancelJavaProcess,
    DisconnectJavaProcess,
    MissingMavenCache,
    CombinedGradleWorkflow,
}

#[cfg(all(feature = "jvm-test-actions", target_os = "macos"))]
async fn selected_junit_ssh_project(
    root: &std::path::Path,
    app_state: Arc<workspace::AppState>,
    cx: &mut TestAppContext,
) -> gpui::Entity<Project> {
    use remote::{
        ConnectionIdentifier, RemoteClient, RemoteConnectionOptions, SshConnectionOptions,
    };

    cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
    let server =
        std::env::var("ZED_COPY_REMOTE_SERVER").expect("explicit current JVM server binary");
    assert!(std::path::Path::new(&server).is_absolute() && std::path::Path::new(&server).is_file());
    let key = std::env::var("GROOVY_SSH_KEY").expect("disposable loopback client key");
    let known_hosts = std::env::var("GROOVY_SSH_KNOWN_HOSTS").expect("verified fixture host key");
    assert!(std::path::Path::new(&key).is_file() && std::path::Path::new(&known_hosts).is_file());
    let port = std::env::var("GROOVY_SSH_PORT")
        .expect("disposable loopback port")
        .parse()
        .expect("numeric SSH port");
    let options = RemoteConnectionOptions::Ssh(SshConnectionOptions {
        host: "127.0.0.1".to_string().into(),
        username: Some(std::env::var("GROOVY_SSH_USER").expect("fixture user")),
        port: Some(port),
        password: None,
        args: Some(vec![
            "-i".into(),
            key,
            "-o".into(),
            format!("UserKnownHostsFile={known_hosts}"),
            "-o".into(),
            "StrictHostKeyChecking=yes".into(),
            "-o".into(),
            "BatchMode=yes".into(),
        ]),
        port_forwards: None,
        connection_timeout: Some(10),
        nickname: None,
        upload_binary_over_ssh: true,
    });
    let delegate = Arc::new(remote::MockDelegate);
    let connection = remote::connect(options, delegate.clone(), &mut cx.to_async())
        .await
        .expect("normal loopback SSH connection");
    let identifier = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("fixture clock")
        .as_micros()
        % 1_000_000_000;
    eprintln!("Groovy SSH fixture workspace identifier: {identifier}");
    let (_cancellation, cancelled) = futures::channel::oneshot::channel();
    let client = cx
        .update(|cx| {
            RemoteClient::new(
                ConnectionIdentifier::Workspace(identifier as i64),
                connection,
                cancelled,
                delegate,
                cx,
            )
        })
        .await
        .expect("SSH client startup")
        .expect("uncancelled fixture connection");
    let extensions = client
        .update(cx, |client, _| {
            client.proto_client().request(rpc::proto::SyncExtensions {
                extensions: vec![rpc::proto::Extension {
                    id: "java".into(),
                    version: "6.8.27".into(),
                    dev: false,
                }],
            })
        })
        .await
        .expect("normal remote extension sync");
    assert!(
        extensions.missing_extensions.is_empty(),
        "provision the pinned complete Java extension through the app before running this fixture"
    );
    let project = cx.update(|cx| {
        Project::remote(
            client,
            app_state.client.clone(),
            app_state.node_runtime.clone(),
            app_state.user_store.clone(),
            app_state.languages.clone(),
            app_state.fs.clone(),
            true,
            cx,
        )
    });
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(root, true, cx)
        })
        .await
        .expect("actual SSH JVM worktree");
    project
        .read_with(cx, |project, cx| {
            project.worktree_store().read(cx).wait_for_initial_scan()
        })
        .await;
    project
}

#[cfg(feature = "jvm-test-actions")]
pub async fn selected_java_case_through_tests_panel(
    cx: &mut TestAppContext,
    build_tool: JvmTestBuildTool,
    outcome: JavaTestOutcome,
) {
    selected_java_case_through_tests_panel_with_connection(cx, build_tool, outcome, false).await;
}

#[cfg(feature = "jvm-test-actions")]
pub async fn selected_java_case_through_tests_panel_with_connection(
    cx: &mut TestAppContext,
    build_tool: JvmTestBuildTool,
    outcome: JavaTestOutcome,
    use_ssh: bool,
) {
    selected_java_case_with_debug_hooks(cx, build_tool, outcome, use_ssh, None).await;
}

async fn selected_java_case_with_debug_hooks(
    cx: &mut TestAppContext,
    build_tool: JvmTestBuildTool,
    outcome: JavaTestOutcome,
    use_ssh: bool,
    debug_hooks: Option<JvmWorkflowDebugHooks>,
) {
    use std::{borrow::Cow, path::Path};

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt as _;

    use collections::HashSet;
    use language::{Language, LanguageConfig, LanguageMatcher, LanguageQueries};
    use project::trusted_worktrees::PathTrust;

    let use_gradle = build_tool == JvmTestBuildTool::Gradle;
    let with_real_providers = outcome == JavaTestOutcome::CombinedGradleWorkflow;
    let should_fail = outcome == JavaTestOutcome::FailAndRerun || with_real_providers;
    assert!(!with_real_providers || (use_gradle && !use_ssh));
    let passing_assertion = if use_gradle {
        "assertEquals(\"Hello from Kotlin\", new Greeting().message())"
    } else {
        "assertEquals(\"ready\", Main.message())"
    };
    let failing_assertion = if use_gradle {
        "assertEquals(\"wrong\", new Greeting().message())"
    } else {
        "assertEquals(\"wrong\", Main.message())"
    };
    let should_cancel = matches!(
        outcome,
        JavaTestOutcome::Cancel
            | JavaTestOutcome::RestrictTrust
            | JavaTestOutcome::RestrictJavaProcess
            | JavaTestOutcome::CancelJavaProcess
            | JavaTestOutcome::DisconnectJavaProcess
    );
    let real_java_process = matches!(
        outcome,
        JavaTestOutcome::RestrictJavaProcess
            | JavaTestOutcome::CancelJavaProcess
            | JavaTestOutcome::DisconnectJavaProcess
    );
    cx.executor().allow_parking();
    let app_state = cx.update(|cx| {
        let state = workspace::AppState::test(cx);
        crate::init(cx);
        editor::init(cx);
        project::task_store::TaskStore::init(None);
        state
    });
    #[cfg(not(target_os = "macos"))]
    drop(app_state);
    cx.update(|cx| terminal_view::init(cx));
    if let Some(hooks) = debug_hooks.as_ref() {
        cx.update(|cx| (hooks.initialize)(cx));
    }
    let (fixture_name, files, wrapper_name, report_path) = if use_gradle {
        (
            "groovy_jvm_gradle",
            &[
                "settings.gradle.kts",
                "build.gradle.kts",
                "gradlew",
                "gradlew.bat",
                "gradle/wrapper/gradle-wrapper.jar",
                "gradle/wrapper/gradle-wrapper.properties",
                "library/build.gradle.kts",
                "library/src/main/kotlin/evidence/Greeting.kt",
                "application/build.gradle.kts",
                "application/src/main/java/evidence/Main.java",
                "application/src/test/java/evidence/GreetingTest.java",
            ][..],
            if cfg!(windows) {
                "gradlew.bat"
            } else {
                "gradlew"
            },
            "application/build/test-results/test/TEST-evidence.GreetingTest.xml",
        )
    } else {
        (
            "groovy_jvm_maven",
            &[
                "pom.xml",
                "mvnw",
                "mvnw.cmd",
                ".mvn/wrapper/maven-wrapper.properties",
                "library/pom.xml",
                "library/src/main/kotlin/evidence/Greeting.kt",
                "application/pom.xml",
                "application/src/main/java/evidence/Main.java",
                "application/src/test/java/evidence/GreetingTest.java",
            ][..],
            if cfg!(windows) { "mvnw.cmd" } else { "mvnw" },
            "application/target/surefire-reports/TEST-evidence.GreetingTest.xml",
        )
    };
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../project/test_data")
        .join(fixture_name);
    let root = tempfile::tempdir().expect("temporary JVM worktree");
    let project_root = if with_real_providers {
        // Providers canonicalize macOS's /var symlink in workspace exports.
        root.path()
            .canonicalize()
            .expect("canonical JVM workflow root")
    } else {
        root.path().to_path_buf()
    };
    for &relative in files {
        let destination = root.path().join(relative);
        std::fs::create_dir_all(destination.parent().expect("fixture parent"))
            .expect("create fixture parent");
        std::fs::copy(fixture.join(relative), destination).expect("copy fixture file");
    }
    #[cfg(unix)]
    std::fs::set_permissions(
        root.path().join(wrapper_name),
        std::fs::Permissions::from_mode(0o755),
    )
    .expect("make project wrapper executable");
    let provider_state =
        with_real_providers.then(|| tempfile::tempdir().expect("isolated JVM provider state"));
    let java_settings = serde_json::json!({
        "import": {"gradle": {"enabled": true,
            "java": {"home": std::env::var("GROOVY_BUILD_JAVA_HOME").ok()},
            "offline": {"enabled": true}}},
        "jdt": {"ls": {"kotlinSupport": {"enabled": true}}}
    });
    if let Some(state) = &provider_state {
        let server_java_home =
            std::env::var("GROOVY_LSP_JAVA_HOME").expect("explicit provider JDK");
        let build_java_home = std::env::var("GROOVY_BUILD_JAVA_HOME").expect("explicit build JDK");
        for home in [&server_java_home, &build_java_home] {
            assert!(Path::new(home).is_absolute() && Path::new(home).join("bin/java").is_file());
        }
        std::fs::write(
            root.path().join("gradle.properties"),
            format!("org.gradle.java.home={build_java_home}\norg.gradle.daemon=false\n"),
        )
        .expect("explicit Gradle fixture JDK");
        let wrapper = root.path().join(wrapper_name);
        let original = std::fs::read_to_string(&wrapper).expect("Gradle wrapper");
        assert!(original.starts_with("#!/bin/sh"));
        std::fs::write(
            wrapper,
            original.replacen("#!/bin/sh", "#!/bin/sh\nset -- --offline \"$@\"", 1),
        )
        .expect("require cached Gradle fixture");
        let mut servers = serde_json::Map::new();
        for (name, variable) in [
            ("jdtls", "GROOVY_JDTLS_LAUNCHER"),
            ("kotlin-lsp", "GROOVY_KOTLIN_LSP"),
        ] {
            let launcher = std::env::var(variable).expect("explicit cached provider launcher");
            assert!(Path::new(&launcher).is_absolute() && Path::new(&launcher).is_file());
            let arguments = if name == "jdtls" {
                vec![
                    format!("--java-executable={server_java_home}/bin/java"),
                    "-data".to_string(),
                    state
                        .path()
                        .join("jdtls-data")
                        .to_string_lossy()
                        .into_owned(),
                ]
            } else {
                vec!["--stdio".to_string()]
            };
            let options = ["config", "system", "plugins", "log"]
                .into_iter()
                .map(|kind| {
                    format!(
                        "-Didea.{kind}.path={}",
                        state.path().join(name).join(kind).display()
                    )
                })
                .chain(std::iter::once(
                    "-Djava.net.preferIPv4Stack=true".to_string(),
                ))
                .collect::<Vec<_>>()
                .join(" ");
            servers.insert(
                name.to_string(),
                serde_json::json!({"binary": {
                    "path": launcher, "arguments": arguments,
                    "env": {"JAVA_HOME": server_java_home, "JAVA_TOOL_OPTIONS": options}
                }}),
            );
        }
        servers.get_mut("jdtls").expect("Java server settings")["settings"] =
            serde_json::json!({"java": java_settings});
        let settings = serde_json::json!({"lsp": servers});
        std::fs::create_dir_all(root.path().join(".zed")).expect("fixture settings");
        std::fs::write(root.path().join(".zed/settings.json"), settings.to_string())
            .expect("explicit provider overrides");
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(&settings.to_string(), cx)
                    .expect("provider settings");
            })
        });
    }
    #[cfg(windows)]
    {
        assert!(build_tool == JvmTestBuildTool::Maven && outcome == JavaTestOutcome::Pass);
        std::fs::write(
            root.path().join(wrapper_name),
            "@echo off\r\necho Groovy selected Java test started\r\necho.>> \"application\\target\\surefire-reports\\TEST-evidence.GreetingTest.xml\"\r\necho Tests run: 1\r\necho BUILD SUCCESS\r\nexit /b 0\r\n",
        )
        .expect("write selected-test command wrapper");
    }
    if should_cancel && !real_java_process {
        std::fs::write(
            root.path().join(wrapper_name),
            "#!/bin/sh\nprintf '%s\\n' \"$$\" > groovy-test-pid\nprintf 'started\\n' > groovy-test-started\nprintf 'Groovy test child started\\n'\nwhile [ ! -f groovy-test-release ]; do sleep 0.1; done\nprintf 'late\\n' > groovy-test-late\n",
        )
        .expect("write cancellable test wrapper");
    }
    if real_java_process {
        if use_gradle {
            let wrapper = root.path().join(wrapper_name);
            let original = std::fs::read_to_string(&wrapper).expect("Gradle wrapper");
            assert!(original.starts_with("#!/bin/sh"));
            std::fs::write(
                wrapper,
                original.replacen("#!/bin/sh", "#!/bin/sh\nset -- --offline \"$@\"", 1),
            )
            .expect("require cached Gradle dependencies");
        } else {
            std::fs::write(root.path().join(".mvn/maven.config"), "-o\n")
                .expect("require cached Maven dependencies");
        }
        let source = root
            .path()
            .join("application/src/test/java/evidence/GreetingTest.java");
        let original = std::fs::read_to_string(&source).expect("Java test source");
        let body = r#"void messageIsAvailable() throws Exception {
    final java.nio.file.Path control = java.nio.file.Path.of("..");
    java.nio.file.Files.writeString(control.resolve("groovy-test-pid"),
        Long.toString(ProcessHandle.current().pid()));
    java.nio.file.Files.writeString(control.resolve("groovy-test-java-version"),
        System.getProperty("java.runtime.version"));
    java.nio.file.Files.writeString(control.resolve("groovy-test-started"), "started");
    System.out.println("Groovy test child started");
    while (!java.nio.file.Files.exists(control.resolve("groovy-test-release"))) {
        Thread.sleep(100);
    }
    java.nio.file.Files.writeString(control.resolve("groovy-test-late"), "late");"#;
        assert!(original.contains("void messageIsAvailable() {"));
        std::fs::write(
            source,
            original.replace("void messageIsAvailable() {", body),
        )
        .expect("write cancellable Java test");
    }
    if outcome == JavaTestOutcome::MissingMavenCache {
        let repository = root.path().join("empty-maven-repository");
        std::fs::create_dir(&repository).expect("create empty Maven repository");
        std::fs::write(
            root.path().join(".mvn/maven.config"),
            format!("-o\n-Dmaven.repo.local={}\n", repository.display()),
        )
        .expect("require Maven offline from an empty repository");
    }
    if should_fail && !with_real_providers {
        let source = root
            .path()
            .join("application/src/test/java/evidence/GreetingTest.java");
        let original = std::fs::read_to_string(&source).expect("Java test source");
        assert!(original.contains(passing_assertion));
        std::fs::write(
            source,
            original.replace(passing_assertion, failing_assertion),
        )
        .expect("write failing Java assertion");
    }
    let report = root.path().join(report_path);
    std::fs::create_dir_all(report.parent().expect("report parent"))
        .expect("create cached report parent");
    std::fs::copy(
        fixture.join("reports/TEST-evidence.GreetingTest.xml"),
        &report,
    )
    .expect("copy cached JUnit report");
    let cached_report_modified = std::fs::metadata(&report)
        .expect("cached JUnit metadata")
        .modified()
        .expect("cached JUnit modification time");

    cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));

    if use_ssh {
        std::fs::create_dir_all(root.path().join(".zed")).expect("remote fixture settings");
        std::fs::write(
            root.path().join(".zed/settings.json"),
            r#"{"languages":{"Java":{"language_servers":[]},"Kotlin":{"language_servers":[]}}}"#,
        )
        .expect("isolate selected Task evidence from provider import");
    }
    let project = if use_ssh {
        #[cfg(target_os = "macos")]
        {
            selected_junit_ssh_project(root.path(), app_state, cx).await
        }
        #[cfg(not(target_os = "macos"))]
        {
            panic!("the opt-in loopback SSH fixture is currently macOS-only")
        }
    } else {
        Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [project_root.as_path()],
            cx,
        )
        .await
    };
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    let java = Language::new(
        LanguageConfig {
            name: "Java".into(),
            matcher: LanguageMatcher {
                path_suffixes: vec!["java".into()],
                ..Default::default()
            }
            .into(),
            ..Default::default()
        },
        Some(tree_sitter_java::LANGUAGE.into()),
    )
    .with_queries(LanguageQueries {
        runnables: Some(Cow::Borrowed(
            r#"(program
                (package_declaration [(identifier) (scoped_identifier)] @java_package_name)?
                (class_declaration
                  name: (identifier) @java_class_name
                  body: (class_body
                    (method_declaration
                      (modifiers (marker_annotation name: (identifier) @annotation_name))
                      name: (identifier) @run @java_method_name
                      (#match? @annotation_name "Test$")))) @_
                (#set! tag java-test-method))"#,
        )),
        ..Default::default()
    })
    .expect("pinned Java extension runnable contract");
    languages.add(Arc::new(java));

    let (worktree_store, worktree_id) = project.read_with(cx, |project, cx| {
        let worktree_store = project.worktree_store();
        let worktree_id = worktree_store
            .read(cx)
            .worktrees()
            .next()
            .expect("JVM worktree")
            .read(cx)
            .id();
        (worktree_store, worktree_id)
    });
    let trust = cx.read(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).expect("trust initialized")
    });
    trust.update(cx, |trust, cx| {
        trust.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        )
    });

    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("local Java test provider")
            .clone()
    });
    let snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("source-backed Java discovery");
    let projection = project_test_explorer(&[snapshot], &TestExplorerFilter::default());
    let case_label = if use_gradle {
        "messageIsAvailable()"
    } else {
        "messageIsAvailable"
    };
    let case = find_node(&projection.snapshot.roots, case_label).expect("real Java JUnit case");
    let class =
        find_node(&projection.snapshot.roots, "evidence.GreetingTest").expect("Java JUnit class");
    assert!(!projection.can_run(&class.id));
    let selection = projection
        .selection(&case.id)
        .expect("source-backed selector");
    assert!(projection.can_run(&case.id));
    assert_eq!(selection.provider_id.0, JVM_JUNIT_PROVIDER_ID);

    let (container, cx) = cx.add_window_view(|window, cx| {
        workspace::MultiWorkspace::test_new(project.clone(), window, cx)
    });
    let workspace = container.read_with(cx, |container, _| container.workspace().clone());
    let async_cx = cx.update(|window, cx| window.to_async(cx));
    let terminal_panel =
        terminal_view::terminal_panel::TerminalPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("production terminal panel");
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.add_panel(terminal_panel.clone(), window, cx);
    });
    let mut real_provider_servers = Vec::new();
    let mut dependencies_panel = None;
    if with_real_providers {
        use futures::FutureExt as _;
        use language::FakeLspAdapter;
        use language_tools::language_tool_tree::{ExpandAll, LanguageToolTreeStatus, SelectFirst};
        use project::jvm_project::KotlinWorkspaceDependency;
        use task::{TaskContext, TaskVariables, VariableName};

        let tasks = project
            .read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("shared Task inventory")
                    .read(cx)
                    .list_tasks(None, None, Some(worktree_id), cx)
            })
            .await;
        let (source, template) = tasks
            .into_iter()
            .find(|(source, template)| {
                source.groovy_jvm_build_worktree() == Some(worktree_id)
                    && template.label == "Build Gradle project"
            })
            .expect("provider-specific generated Gradle build Task");
        let host_root = worktree_store.read_with(cx, |store, cx| {
            store
                .worktree_for_id(worktree_id, cx)
                .expect("current build worktree")
                .read(cx)
                .abs_path()
                .to_path_buf()
        });
        let mut variables = TaskVariables::default();
        variables.insert(
            VariableName::WorktreeRoot,
            host_root.to_string_lossy().into_owned(),
        );
        let resolved = template
            .resolve_task(
                &source.to_id_base(),
                &TaskContext {
                    cwd: Some(host_root),
                    task_variables: variables,
                    ..Default::default()
                },
            )
            .expect("current generated build context");
        let (sender, receiver) = futures::channel::oneshot::channel();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source,
                resolved,
                false,
                move |result, _| sender.send(result).expect("build completion receiver"),
                window,
                cx,
            );
        });
        let completion = receiver.fuse();
        let timeout = cx
            .background_executor
            .timer(Duration::from_secs(180))
            .fuse();
        futures::pin_mut!(completion, timeout);
        let result = futures::select_biased! {
            result = completion => result.expect("product build Task outcome"),
            () = timeout => panic!("product Gradle build exceeded 180 seconds"),
        };
        assert_eq!(result, workspace::tasks::ScheduledTaskResult::Success);
        assert!(root.path().join("application/build/generated/sources/groovyFixture/java/evidence/BuildMessage.java").is_file());
        assert!(
            root.path()
                .join("library/build/classes/kotlin/main/evidence/Greeting.class")
                .is_file()
        );
        eprintln!(
            "Combined product Gradle build Task succeeded with fresh generated/Kotlin output"
        );

        let mut java_initialization = serde_json::json!({"settings":{"java":java_settings}});
        if debug_hooks.is_some() {
            let debug_jar =
                std::env::var("GROOVY_JAVA_DEBUG_JAR").expect("explicit cached Java Debug bundle");
            assert!(Path::new(&debug_jar).is_absolute() && Path::new(&debug_jar).is_file());
            java_initialization["bundles"] = serde_json::json!([debug_jar]);
        }
        languages.register_fake_lsp_adapter(
            "Java",
            FakeLspAdapter {
                name: "jdtls",
                initialization_options: Some(java_initialization),
                ..Default::default()
            },
        );
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Kotlin".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["kt".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        languages.register_fake_lsp_adapter(
            "Kotlin",
            FakeLspAdapter {
                name: "kotlin-lsp",
                ..Default::default()
            },
        );
        let java_path = project_root.join("application/src/main/java/evidence/Main.java");
        let java_source = std::fs::read_to_string(&java_path).expect("Java fixture source");
        let (java_buffer, _java_handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp(java_path, cx)
            })
            .await
            .expect("shared Java buffer");
        let (kotlin_buffer, _kotlin_handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp(
                    project_root.join("library/src/main/kotlin/evidence/Greeting.kt"),
                    cx,
                )
            })
            .await
            .expect("shared Kotlin buffer");
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let deadline = Instant::now() + Duration::from_secs(240);
        let mut refresh_at = Instant::now();
        loop {
            cx.run_until_parked();
            if lsp_store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(data, current)| current && data.projects.len() >= 2)
                    && store
                        .kotlin_workspace_snapshot(worktree_id)
                        .is_some_and(|(data, current)| current && data.projects.len() >= 2)
            }) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "current two-provider import did not complete: {:?}",
                lsp_store.read_with(cx, |store, _| (
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .map(|(data, current)| (data.projects.len(), current)),
                    store
                        .kotlin_workspace_snapshot(worktree_id)
                        .map(|(data, current)| (data.projects.len(), current)),
                    store
                        .language_server_statuses
                        .keys()
                        .copied()
                        .collect::<Vec<_>>()
                ))
            );
            if Instant::now() >= refresh_at {
                lsp_store.update(cx, |store, cx| {
                    store.refresh_jdtls_project(worktree_id, cx);
                    store.refresh_kotlin_workspace(worktree_id, cx);
                });
                refresh_at = Instant::now() + Duration::from_secs(5);
            }
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        lsp_store.read_with(cx, |store, _| {
            let java = store
                .jdtls_project_snapshot(worktree_id)
                .expect("Java import")
                .0;
            let kotlin = store
                .kotlin_workspace_snapshot(worktree_id)
                .expect("Kotlin import")
                .0;
            assert_ne!(java.server_id, kotlin.server_id);
            assert!(
                java.projects
                    .iter()
                    .any(|project| project.library_entries > 0)
            );
            assert!(
                java.projects
                    .iter()
                    .any(|project| project.source_roots.iter().any(|root| root
                        .path
                        .path
                        .as_unix_str()
                        .contains("generated/sources/groovyFixture")))
            );
            assert!(
                kotlin
                    .projects
                    .iter()
                    .any(
                        |project| project.dependencies.iter().any(|dependency| matches!(
                            dependency,
                            KotlinWorkspaceDependency::Artifact { .. }
                        ))
                    )
            );
            let local = store.as_local().expect("local provider host");
            for id in [java.server_id, kotlin.server_id] {
                let server = local
                    .running_language_server_for_id(id)
                    .expect("real provider")
                    .clone();
                let pid = server.process_id().expect("real provider PID");
                real_provider_servers.push((server, pid));
            }
            eprintln!(
                "Combined Tests Project: Java {} projects/Kotlin {} projects",
                java.projects.len(),
                kotlin.projects.len()
            );
        });
        for (needle, target) in [
            ("Greeting()", "class Greeting"),
            ("BuildMessage.value", "class BuildMessage"),
        ] {
            let definitions = project
                .update(cx, |project, cx| {
                    project.definitions(
                        &java_buffer,
                        java_source.find(needle).expect("source reference"),
                        cx,
                    )
                })
                .await
                .expect("Project definition request")
                .expect("definition result");
            assert!(definitions.iter().any(|definition| {
                definition
                    .target
                    .buffer
                    .read_with(cx, |buffer, _| buffer.text().contains(target))
            }));
        }
        let original_kotlin = kotlin_buffer.read_with(cx, |buffer, _| buffer.text());
        let probe = "\nfun combinedTestWorkflowProbe() = Greeting().message()\n";
        let offset = original_kotlin.len() + probe.find("message()").expect("Kotlin probe");
        kotlin_buffer.update(cx, |buffer, cx| {
            let end = buffer.len();
            buffer.edit([(end..end, probe)], None, cx);
        });
        project
            .update(cx, |project, cx| {
                project.save_buffer(kotlin_buffer.clone(), cx)
            })
            .await
            .expect("save Kotlin edit");
        let definitions = project
            .update(cx, |project, cx| {
                project.definitions(&kotlin_buffer, offset, cx)
            })
            .await
            .expect("edited Kotlin definition")
            .expect("Kotlin definition result");
        assert!(
            definitions
                .iter()
                .any(|definition| definition.target.buffer == kotlin_buffer)
        );
        kotlin_buffer.update(cx, |buffer, cx| {
            let end = buffer.len();
            buffer.edit([(0..end, original_kotlin)], None, cx);
        });
        project
            .update(cx, |project, cx| {
                project.save_buffer(kotlin_buffer.clone(), cx)
            })
            .await
            .expect("restore Kotlin source");

        cx.update(|_, cx| jvm_ui::init(cx));
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = jvm_ui::DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("existing Dependencies panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace.right_dock().update(cx, |dock, cx| {
                dock.set_open(true, window, cx);
            });
            workspace.toggle_panel_focus::<jvm_ui::DependenciesPanel>(window, cx);
        });
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            cx.run_until_parked();
            let ready = panel.read_with(cx, |panel, _| {
                let (status, rows, navigation) = panel.snapshot_for_test();
                matches!(
                    status,
                    LanguageToolTreeStatus::Current | LanguageToolTreeStatus::Partial(_)
                ) && rows
                    .iter()
                    .any(|row| row.accessibility_label.contains("JDTLS module"))
                    && rows
                        .iter()
                        .any(|row| row.accessibility_label.contains("Kotlin LSP module"))
                    && navigation
                        .values()
                        .any(|path| path.path.as_unix_str() == "application/build.gradle.kts")
            });
            if ready {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "provider-backed Dependencies did not become actionable"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        panel.update_in(cx, |panel, window, cx| {
            assert!(panel.focus_handle(cx).contains_focused(window, cx));
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(ExpandAll);
        cx.run_until_parked();
        let (rows, navigation) = panel.read_with(cx, |panel, _| {
            let (_, rows, navigation) = panel.snapshot_for_test();
            (rows, navigation)
        });
        assert!(
            rows.iter()
                .any(|row| row.accessibility_label.contains("JDTLS"))
        );
        assert!(
            rows.iter()
                .any(|row| row.accessibility_label.contains("Kotlin"))
        );
        let target = rows
            .iter()
            .position(|row| {
                navigation
                    .get(&row.id)
                    .is_some_and(|path| path.path.as_unix_str() == "application/build.gradle.kts")
            })
            .expect("current Gradle manifest navigation row");
        cx.dispatch_action(SelectFirst);
        for _ in 0..target {
            cx.simulate_keystrokes("down");
        }
        cx.simulate_keystrokes("enter");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            cx.run_until_parked();
            if workspace
                .read_with(cx, |workspace, cx| {
                    workspace.item_of_type::<editor::Editor>(cx)
                })
                .is_some()
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Dependencies keyboard activation did not open Editor"
            );
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        let editor = workspace
            .read_with(cx, |workspace, cx| {
                workspace.item_of_type::<editor::Editor>(cx)
            })
            .expect("manifest Editor");
        assert!(editor.read_with(cx, |editor, cx| {
            editor.text(cx).contains("generateBuildMessage")
        }));
        eprintln!(
            "Combined Dependencies shared-tree keyboard navigation opened the current Gradle manifest"
        );
        dependencies_panel = Some(panel);

        use editor::code_context_menus::CodeContextMenu;
        use editor::{MultiBufferOffset, SelectionEffects, actions};
        use language::BufferEvent;
        use std::{cell::Cell, rc::Rc};

        let probe_path = project_root.join("application/src/main/java/evidence/LanguageProbe.java");
        let original_java = "package evidence;\npublic final class LanguageProbe {\n    public static String message() { return \"ready\"; }\n    public static void main(String[] arguments) {\n        System.out.println(message());\n    }\n}\n";
        std::fs::write(&probe_path, original_java).expect("disposable Java Editor fixture");
        let (probe_buffer, _probe_handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp(probe_path.clone(), cx)
            })
            .await
            .expect("current Java Editor buffer");
        let probe_project_path = project.read_with(cx, |project, cx| {
            project
                .find_project_path(&probe_path, cx)
                .expect("Java Editor project path")
        });
        let item = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(probe_project_path, None, true, window, cx)
            })
            .await
            .expect("open Java through Workspace Editor");
        let java_editor = item.downcast::<editor::Editor>().expect("Java Editor item");
        java_editor.read_with(cx, |editor, cx| {
            assert_eq!(
                editor.buffer().read(cx).as_singleton(),
                Some(probe_buffer.clone())
            );
            assert_eq!(editor.text(cx), original_java);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        let java_server_id = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .expect("current Java provider")
                .0
                .server_id
        });
        let completion_offset = original_java
            .find("println")
            .expect("Java completion prefix")
            + 2;
        java_editor.update_in(cx, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx);
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([
                    MultiBufferOffset(completion_offset)..MultiBufferOffset(completion_offset)
                ]);
            });
        });
        cx.dispatch_action(actions::ShowCompletions);
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            cx.run_until_parked();
            let visible = java_editor.read_with(cx, |editor, _| {
                let menu = editor.context_menu().borrow();
                let Some(CodeContextMenu::Completions(menu)) = menu.as_ref() else {
                    return false;
                };
                let completions = menu.completions.borrow();
                menu.entries
                    .borrow()
                    .iter()
                    .filter_map(|entry| entry.as_match())
                    .any(|entry| {
                        completions
                            .get(entry.candidate_id)
                            .is_some_and(|completion| {
                                completion.label.text.starts_with("println")
                                    && matches!(&completion.source, project::CompletionSource::Lsp { server_id, .. } if *server_id == java_server_id)
                            })
                    })
            });
            if visible {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "real Java completion did not reach the Editor menu"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        cx.update(|window, cx| window.draw(cx).clear(cx));
        java_editor.update_in(cx, |editor, window, cx| {
            assert!(editor.context_menu_visible());
            editor.cancel(&actions::Cancel, window, cx);
        });
        let diagnostic_updates = Rc::new(Cell::new(0));
        let _diagnostic_subscription = cx.update(|_, cx| {
            cx.subscribe(&probe_buffer, {
                let diagnostic_updates = diagnostic_updates.clone();
                move |_, event, _| {
                    if matches!(event, BufferEvent::DiagnosticsUpdated) {
                        diagnostic_updates.set(diagnostic_updates.get() + 1);
                    }
                }
            })
        });
        let literal_offset = original_java
            .find("\"ready\"")
            .expect("Java return literal");
        java_editor.update_in(cx, |editor, window, cx| {
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([
                    MultiBufferOffset(literal_offset)..MultiBufferOffset(literal_offset + 7)
                ]);
            });
            editor.insert("missingMessage()", window, cx);
        });
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            cx.run_until_parked();
            let diagnosed = probe_buffer.read_with(cx, |buffer, _| {
                buffer
                    .snapshot()
                    .diagnostics_in_range::<_, usize>(literal_offset..literal_offset + 14, false)
                    .any(|entry| {
                        entry.diagnostic.severity == lsp::DiagnosticSeverity::ERROR
                            && entry.diagnostic.message.contains("missingMessage")
                    })
            });
            if diagnosed {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Java Editor invalid edit did not receive its current diagnostic"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        java_editor.update_in(cx, |editor, window, cx| {
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([
                    MultiBufferOffset(literal_offset)..MultiBufferOffset(literal_offset)
                ]);
            });
            editor.go_to_diagnostic(&actions::GoToDiagnostic::default(), window, cx);
            assert!(
                editor
                    .active_diagnostic_message()
                    .is_some_and(|message| message.contains("missingMessage"))
            );
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(actions::ToggleCodeActions::default());
        let deadline = Instant::now() + Duration::from_secs(60);
        let action_index = loop {
            cx.run_until_parked();
            let action_index = java_editor.read_with(cx, |editor, _| {
                let menu = editor.context_menu().borrow();
                let Some(CodeContextMenu::CodeActions(menu)) = menu.as_ref() else {
                    return None;
                };
                (0..)
                    .map_while(|index| menu.actions.get(index).map(|item| (index, item.label())))
                    .find(|(_, label)| {
                        label.starts_with("Create method") && label.contains("missingMessage")
                    })
                    .map(|(index, _)| index)
            });
            if let Some(index) = action_index {
                break index;
            }
            assert!(
                Instant::now() < deadline,
                "enabled Java quick fix did not reach the Editor action menu"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        };
        cx.update(|window, cx| window.draw(cx).clear(cx));
        let invalid_diagnostic_update = diagnostic_updates.get();
        java_editor
            .update_in(cx, |editor, window, cx| {
                assert!(editor.context_menu_visible());
                editor
                    .confirm_code_action(
                        &actions::ConfirmCodeAction {
                            item_ix: Some(action_index),
                        },
                        window,
                        cx,
                    )
                    .expect("Editor menu quick-fix task")
            })
            .await
            .expect("confirm real Java quick fix through Editor");
        assert!(java_editor.read_with(cx, |editor, cx| {
            editor
                .text(cx)
                .contains("private static String missingMessage()")
        }));
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            cx.run_until_parked();
            let recovered = probe_buffer.read_with(cx, |buffer, _| {
                !buffer
                    .snapshot()
                    .diagnostics_in_range::<_, usize>(0..buffer.len(), false)
                    .any(|entry| entry.diagnostic.severity == lsp::DiagnosticSeverity::ERROR)
            });
            if recovered && diagnostic_updates.get() > invalid_diagnostic_update {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Editor-confirmed Java quick fix needs fresh error-free diagnostics"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        let fixed_diagnostic_update = diagnostic_updates.get();
        java_editor.update_in(cx, |editor, window, cx| {
            let end = editor.text(cx).len();
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(0)..MultiBufferOffset(end)]);
            });
            editor.insert(original_java, window, cx);
            assert_eq!(editor.text(cx), original_java);
        });
        project
            .update(cx, |project, cx| {
                project.save_buffer(probe_buffer.clone(), cx)
            })
            .await
            .expect("save restored Java Editor source");
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            cx.run_until_parked();
            let recovered = probe_buffer.read_with(cx, |buffer, _| {
                !buffer
                    .snapshot()
                    .diagnostics_in_range::<_, usize>(0..buffer.len(), false)
                    .any(|entry| entry.diagnostic.severity == lsp::DiagnosticSeverity::ERROR)
            });
            if recovered && diagnostic_updates.get() > fixed_diagnostic_update {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "restored Java Editor source needs fresh error-free diagnostics"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        eprintln!(
            "Combined Java Editor completion menu, current diagnostic, confirmed quick fix and restored-source recovery passed"
        );

        let kotlin_project_path = project.read_with(cx, |project, cx| {
            project
                .find_project_path(
                    project_root.join("library/src/main/kotlin/evidence/Greeting.kt"),
                    cx,
                )
                .expect("Kotlin Editor project path")
        });
        let item = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(kotlin_project_path, None, true, window, cx)
            })
            .await
            .expect("open Kotlin through Workspace Editor");
        let kotlin_editor = item
            .downcast::<editor::Editor>()
            .expect("Kotlin Editor item");
        let original_kotlin = kotlin_editor.read_with(cx, |editor, cx| {
            assert_eq!(
                editor.buffer().read(cx).as_singleton(),
                Some(kotlin_buffer.clone())
            );
            editor.text(cx)
        });
        let kotlin_server_id = lsp_store.read_with(cx, |store, _| {
            store
                .kotlin_workspace_snapshot(worktree_id)
                .expect("current Kotlin provider")
                .0
                .server_id
        });
        let kotlin_diagnostic_updates = Rc::new(Cell::new(0));
        let _kotlin_diagnostic_subscription = cx.update(|_, cx| {
            cx.subscribe(&kotlin_buffer, {
                let diagnostic_updates = kotlin_diagnostic_updates.clone();
                move |_, event, _| {
                    if matches!(event, BufferEvent::DiagnosticsUpdated) {
                        diagnostic_updates.set(diagnostic_updates.get() + 1);
                    }
                }
            })
        });
        let probe = "\nfun editorWorkflowProbe() = Greeting().message()\n";
        let call_offset =
            original_kotlin.len() + probe.find("message()").expect("Kotlin completion call");
        kotlin_editor.update_in(cx, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx);
            let end = original_kotlin.len();
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(end)..MultiBufferOffset(end)]);
            });
            editor.insert(probe, window, cx);
        });
        project
            .update(cx, |project, cx| {
                project.save_buffer(kotlin_buffer.clone(), cx)
            })
            .await
            .expect("save valid Kotlin Editor probe");
        cx.update(|window, cx| window.draw(cx).clear(cx));
        kotlin_editor.update_in(cx, |editor, window, cx| {
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([
                    MultiBufferOffset(call_offset + 2)..MultiBufferOffset(call_offset + 2)
                ]);
            });
        });
        cx.dispatch_action(actions::ShowCompletions);
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            cx.run_until_parked();
            let visible = kotlin_editor.read_with(cx, |editor, _| {
                let menu = editor.context_menu().borrow();
                let Some(CodeContextMenu::Completions(menu)) = menu.as_ref() else { return false; };
                let completions = menu.completions.borrow();
                menu.entries.borrow().iter().filter_map(|entry| entry.as_match())
                    .any(|entry| completions.get(entry.candidate_id)
                        .is_some_and(|completion| completion.label.text.starts_with("message")
                            && matches!(&completion.source, project::CompletionSource::Lsp { server_id, .. } if *server_id == kotlin_server_id)))
            });
            if visible {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "official Kotlin completion did not reach the Editor menu"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        cx.update(|window, cx| window.draw(cx).clear(cx));
        kotlin_editor.update_in(cx, |editor, window, cx| {
            assert!(editor.context_menu_visible());
            editor.cancel(&actions::Cancel, window, cx);
            let end = editor.text(cx).len();
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(end)..MultiBufferOffset(end)]);
            });
            editor.insert(
                "\nfun invalidEditorWorkflowProbe() = missingKotlinWorkflow()\n",
                window,
                cx,
            );
        });
        let error_offset = kotlin_editor.read_with(cx, |editor, cx| {
            editor
                .text(cx)
                .find("missingKotlinWorkflow")
                .expect("Kotlin invalid call")
        });
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            cx.run_until_parked();
            let diagnosed = kotlin_buffer.read_with(cx, |buffer, _| {
                buffer
                    .snapshot()
                    .diagnostics_in_range::<_, usize>(error_offset..error_offset + 21, false)
                    .any(|entry| {
                        entry.diagnostic.severity == lsp::DiagnosticSeverity::ERROR
                            && entry.diagnostic.message.contains("missingKotlinWorkflow")
                    })
            });
            if diagnosed {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Kotlin Editor invalid edit did not receive its current diagnostic"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        kotlin_editor.update_in(cx, |editor, window, cx| {
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([
                    MultiBufferOffset(error_offset)..MultiBufferOffset(error_offset)
                ]);
            });
            editor.go_to_diagnostic(&actions::GoToDiagnostic::default(), window, cx);
            assert!(
                editor
                    .active_diagnostic_message()
                    .is_some_and(|message| message.contains("missingKotlinWorkflow"))
            );
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        let invalid_kotlin_diagnostic_update = kotlin_diagnostic_updates.get();
        kotlin_editor.update_in(cx, |editor, window, cx| {
            let end = editor.text(cx).len();
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(0)..MultiBufferOffset(end)]);
            });
            editor.insert(&original_kotlin, window, cx);
            assert_eq!(editor.text(cx), original_kotlin);
        });
        project
            .update(cx, |project, cx| {
                project.save_buffer(kotlin_buffer.clone(), cx)
            })
            .await
            .expect("save restored Kotlin Editor source");
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            cx.run_until_parked();
            let recovered = kotlin_buffer.read_with(cx, |buffer, _| {
                !buffer
                    .snapshot()
                    .diagnostics_in_range::<_, usize>(0..buffer.len(), false)
                    .any(|entry| entry.diagnostic.severity == lsp::DiagnosticSeverity::ERROR)
            });
            if recovered && kotlin_diagnostic_updates.get() > invalid_kotlin_diagnostic_update {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "restored Kotlin Editor source needs fresh error-free diagnostics"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        eprintln!(
            "Combined Kotlin Editor server completion menu, current diagnostic and restored-source recovery passed"
        );

        let (test_buffer, _handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp(
                    project_root.join("application/src/test/java/evidence/GreetingTest.java"),
                    cx,
                )
            })
            .await
            .expect("current selected-test source");
        test_buffer.update(cx, |buffer, cx| {
            let original = buffer.text();
            assert!(original.contains(passing_assertion));
            let edited = original.replace(passing_assertion, failing_assertion);
            let end = buffer.len();
            buffer.edit([(0..end, edited)], None, cx);
        });
        project
            .update(cx, |project, cx| {
                project.save_buffer(test_buffer.clone(), cx)
            })
            .await
            .expect("save controlled failing assertion");
    }
    let async_cx = cx.update(|window, cx| window.to_async(cx));
    let tests_panel = TestsPanel::load(workspace.downgrade(), async_cx)
        .await
        .expect("existing Tests panel");
    tests_panel.update(cx, |panel, cx| {
        panel.active = true;
        panel.refresh(cx);
    });
    cx.run_until_parked();
    if use_ssh {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while !tests_panel.read_with(cx, |panel, _| {
            find_node(&panel.projection.snapshot.roots, case_label).is_some()
        }) && std::time::Instant::now() < deadline
        {
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
            cx.run_until_parked();
        }
    }
    let selection = tests_panel.update_in(cx, |panel, window, cx| {
        let case = find_node(&panel.projection.snapshot.roots, case_label)
            .expect("selected Java case in Tests");
        let id = case.id.clone();
        let selection = panel
            .projection
            .selection(&id)
            .expect("selected Java case")
            .clone();
        assert!(panel.projection.can_run(&id));
        panel.host.expand_all();
        panel.host.select(id);
        assert!(panel.host.selected().is_some());
        panel.run_selected(window, cx);
        selection
    });
    let structured = project.read_with(cx, |project, _| {
        project.structured_execution_store().clone()
    });
    if should_cancel {
        let started = root.path().join("groovy-test-started");
        let startup_timeout = Duration::from_secs(if real_java_process { 50 } else { 5 });
        // GPUI timers advance the test clock while the child starts on the OS clock.
        let startup_deadline = std::time::Instant::now() + startup_timeout;
        let started_with_current_run = loop {
            cx.run_until_parked();
            if started.exists()
                && structured.read_with(cx, |store, _| {
                    store
                        .state()
                        .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                        .and_then(|provider| provider.current_run.as_ref())
                        .is_some_and(|run| {
                            run.phase == project::structured_execution::StructuredRunPhase::Running
                        })
                })
            {
                break true;
            }
            if std::time::Instant::now() >= startup_deadline {
                break false;
            }
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        };
        if !started_with_current_run {
            let state = structured.read_with(cx, |store, _| {
                format!(
                    "{:#?}",
                    store
                        .state()
                        .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                )
            });
            let notice = tests_panel.read_with(cx, |panel, _| panel.action_notice.clone());
            let terminal_output = terminal_panel.read_with(cx, |panel, cx| {
                panel.panes().into_iter().find_map(|pane| {
                    pane.read(cx).items().find_map(|item| {
                        item.downcast::<terminal_view::TerminalView>()
                            .map(|view| view.read(cx).terminal().read(cx).get_content())
                    })
                })
            });
            panic!(
                "selected test child did not start with a current Running test run: marker={}, notice={notice:?}, terminal={terminal_output:?}, state={state}",
                started.exists()
            );
        }
        if real_java_process {
            let runtime_version =
                std::fs::read_to_string(root.path().join("groovy-test-java-version"))
                    .expect("selected JUnit runtime version");
            assert!(!runtime_version.trim().is_empty());
            eprintln!("Selected JUnit runtime: {runtime_version}");
        }
        #[cfg(unix)]
        let child_pid = std::fs::read_to_string(root.path().join("groovy-test-pid"))
            .expect("selected wrapper process ID")
            .trim()
            .parse::<u32>()
            .expect("numeric selected wrapper process ID");
        #[cfg(unix)]
        assert!(
            util::command::new_command("/bin/kill")
                .args(["-0", &child_pid.to_string()])
                .output()
                .await
                .expect("check running selected wrapper")
                .status
                .success()
        );
        #[cfg(target_os = "macos")]
        if use_ssh {
            let mut ancestor = child_pid;
            let mut found_sshd = false;
            for _ in 0..16 {
                let output = util::command::new_command("/bin/ps")
                    .args(["-o", "ppid=,comm=", "-p", &ancestor.to_string()])
                    .output()
                    .await
                    .expect("host process ancestry");
                assert!(
                    output.status.success(),
                    "host ancestor disappeared before restriction"
                );
                let text = String::from_utf8(output.stdout).expect("process ancestry text");
                let mut fields = text.split_whitespace();
                ancestor = fields
                    .next()
                    .expect("parent PID")
                    .parse()
                    .expect("numeric parent PID");
                if fields.any(|field| field.contains("sshd")) {
                    found_sshd = true;
                    break;
                }
                if ancestor <= 1 {
                    break;
                }
            }
            assert!(
                found_sshd,
                "selected JUnit must descend from the loopback SSH daemon"
            );
            eprintln!("SSH selected JUnit PID={child_pid}: host sshd ancestry verified");
        }
        let connection_states = Arc::new(std::sync::Mutex::new(Vec::new()));
        let connection_observation = if outcome == JavaTestOutcome::DisconnectJavaProcess {
            let client = project.read_with(cx, |project, _| {
                project.remote_client().expect("SSH test client")
            });
            let observation = cx.update(|_, cx| {
                let connection_states = connection_states.clone();
                cx.observe(&client, move |client, cx| {
                    connection_states
                        .lock()
                        .expect("connection observations")
                        .push(client.read(cx).connection_state());
                })
            });
            client
                .update(cx, |client, cx| client.force_disconnect(cx))
                .await
                .expect("interrupt the actual SSH control connection");
            Some(observation)
        } else {
            None
        };
        if outcome == JavaTestOutcome::DisconnectJavaProcess {
            let client = project.read_with(cx, |project, _| {
                project.remote_client().expect("SSH test client")
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            while std::time::Instant::now() < deadline {
                cx.run_until_parked();
                let reconnected = {
                    let states = connection_states.lock().expect("connection observations");
                    states.contains(&remote::ConnectionState::Reconnecting)
                        && client.read_with(cx, |client, _| {
                            client.connection_state() == remote::ConnectionState::Connected
                        })
                };
                if reconnected {
                    break;
                }
                cx.background_executor
                    .timer(Duration::from_millis(100))
                    .await;
            }
            assert!(
                connection_states
                    .lock()
                    .expect("connection observations")
                    .contains(&remote::ConnectionState::Reconnecting),
                "the normal remote owner must observe the transport interruption"
            );
            assert_eq!(
                client.read_with(cx, |client, _| client.connection_state()),
                remote::ConnectionState::Connected,
                "the existing remote owner must reconnect"
            );
            let expired = provider
                .update(cx, |provider, cx| {
                    provider.plan_remote_action(
                        &selection.node_id,
                        selection.discovery_generation,
                        None,
                        cx,
                    )
                })
                .await;
            assert!(
                expired.is_err(),
                "reconnection must not revive old discovery"
            );
            eprintln!(
                "SSH reconnect states: {:?}",
                connection_states.lock().expect("connection observations")
            );
        } else if matches!(
            outcome,
            JavaTestOutcome::RestrictTrust | JavaTestOutcome::RestrictJavaProcess
        ) {
            trust.update(cx, |trust, cx| {
                trust.restrict(
                    worktree_store.downgrade(),
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                );
            });
        } else {
            tests_panel.update_in(cx, |panel, window, cx| {
                assert!(
                    panel
                        .selected()
                        .is_some_and(|selection| selection.run_id.is_some())
                );
                panel.cancel_selected(window, cx);
            });
        }
        drop(connection_observation);
        for _ in 0..100 {
            cx.run_until_parked();
            let stopped_run = structured.read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                    .and_then(|provider| provider.current_run.as_ref())
                    .is_some_and(|run| {
                        run.phase == project::structured_execution::StructuredRunPhase::Cancelled
                            || (outcome == JavaTestOutcome::DisconnectJavaProcess
                                && run.phase.is_terminal())
                    })
            });
            if stopped_run {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        let stopped_run = structured.read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                .and_then(|provider| provider.current_run.as_ref())
                .cloned()
                .expect("stopped Java run")
        });
        let task_exited_before_cancellation = outcome == JavaTestOutcome::DisconnectJavaProcess
            && stopped_run.phase == project::structured_execution::StructuredRunPhase::Completed;
        if task_exited_before_cancellation {
            assert_eq!(stopped_run.summary.passed, 0);
            assert!(stopped_run.events.is_empty());
            assert_eq!(
                stopped_run.diagnostic.as_deref(),
                Some("Structured run details remain on the project host."),
                "the shared remote snapshot must keep host diagnostics private"
            );
        } else {
            assert_eq!(
                stopped_run.phase,
                project::structured_execution::StructuredRunPhase::Cancelled
            );
            assert_eq!(stopped_run.summary.cancelled, 1);
        }
        let terminal = terminal_panel
            .read_with(cx, |panel, cx| {
                panel.panes().into_iter().find_map(|pane| {
                    pane.read(cx).items().find_map(|item| {
                        item.downcast::<terminal_view::TerminalView>()
                            .map(|view| view.read(cx).terminal().clone())
                    })
                })
            })
            .expect("stopped test terminal");
        for _ in 0..100 {
            cx.run_until_parked();
            let terminated = terminal.read_with(cx, |terminal, _| {
                terminal
                    .task()
                    .and_then(|task| task.structured_task.as_ref())
                    .is_some_and(|handle| {
                        matches!(
                            handle.state(),
                            task::StructuredTaskState::Cancelled {
                                termination_confirmed: true,
                                ..
                            }
                        ) || (task_exited_before_cancellation
                            && matches!(
                                handle.state(),
                                task::StructuredTaskState::Completed { success: false, .. }
                            ))
                    })
            });
            if terminated {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        let (task_id, task_state, terminal_output) = terminal.read_with(cx, |terminal, _| {
            let handle = terminal
                .task()
                .and_then(|task| task.structured_task.as_ref())
                .expect("selected Task handle");
            (handle.task_id(), handle.state(), terminal.get_content())
        });
        if task_exited_before_cancellation {
            let task::StructuredTaskState::Completed {
                success: false,
                exit_code,
                ..
            } = task_state
            else {
                panic!("interrupted result must match the failed Task: {task_state:?}");
            };
            assert_ne!(exit_code, Some(0));
            assert!(
                terminal_output.contains("finished with exit code:"),
                "{terminal_output}"
            );
        } else {
            assert!(matches!(
                task_state,
                task::StructuredTaskState::Cancelled {
                    termination_confirmed: true,
                    ..
                }
            ));
            assert!(
                terminal_output.contains("cancelled"),
                "Task {task_id:?}: {terminal_output}"
            );
        }
        if !use_gradle || !real_java_process {
            assert!(terminal_output.contains("Groovy test child started"));
        }
        #[cfg(unix)]
        if use_ssh {
            eprintln!(
                "SSH stopped run={:?}, Task={task_id:?}, PID={child_pid}, state={task_state:?}",
                stopped_run.id
            );
        }
        #[cfg(unix)]
        assert!(
            !util::command::new_command("/bin/kill")
                .args(["-0", &child_pid.to_string()])
                .output()
                .await
                .expect("check terminated selected wrapper")
                .status
                .success()
        );
        if matches!(
            outcome,
            JavaTestOutcome::RestrictTrust | JavaTestOutcome::RestrictJavaProcess
        ) {
            if !use_ssh {
                assert!(provider.read_with(cx, |provider, _| {
                    provider.selector(&selection.node_id).is_none()
                }));
            }
            trust.update(cx, |trust, cx| {
                trust.trust(
                    &worktree_store,
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                );
            });
            cx.run_until_parked();
            if use_ssh {
                let result = provider
                    .update(cx, |provider, cx| {
                        provider.plan_remote_action(
                            &selection.node_id,
                            selection.discovery_generation,
                            None,
                            cx,
                        )
                    })
                    .await;
                assert!(
                    result.is_err(),
                    "restored trust must not revive old discovery"
                );
            } else {
                assert!(provider.read_with(cx, |provider, cx| {
                    provider
                        .plan_selected_case(&selection.node_id, selection.discovery_generation, cx)
                        .is_err()
                }));
            }
        }
        std::fs::write(root.path().join("groovy-test-release"), "continue")
            .expect("release cancelled test child");
        cx.background_executor.timer(Duration::from_secs(1)).await;
        cx.run_until_parked();
        assert!(!root.path().join("groovy-test-late").exists());
        if let Ok(metadata) = std::fs::metadata(&report) {
            assert_eq!(
                metadata
                    .modified()
                    .expect("retained report modification time"),
                cached_report_modified
            );
        } else {
            assert!(!report.exists(), "cancelled report remains unreadable");
        }
        assert!(structured.read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                .and_then(|provider| provider.last_complete_run.as_ref())
                .is_none_or(|run| {
                    task_exited_before_cancellation
                        && run.id == stopped_run.id
                        && run.summary.passed == 0
                        && run.events.is_empty()
                })
        }));
        if outcome == JavaTestOutcome::DisconnectJavaProcess {
            tests_panel.update(cx, |panel, cx| panel.discover(cx));
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            while std::time::Instant::now() < deadline {
                cx.run_until_parked();
                let fresh = tests_panel.read_with(cx, |panel, _| {
                    find_node(&panel.projection.snapshot.roots, case_label)
                        .and_then(|case| panel.projection.selection(&case.id))
                        .is_some_and(|fresh| {
                            fresh.discovery_generation != selection.discovery_generation
                        })
                });
                if fresh {
                    break;
                }
                cx.background_executor
                    .timer(Duration::from_millis(100))
                    .await;
            }
            tests_panel.update_in(cx, |panel, window, cx| {
                let case = find_node(&panel.projection.snapshot.roots, case_label)
                    .expect("Java case after fresh remote discovery");
                let fresh = panel
                    .projection
                    .selection(&case.id)
                    .expect("fresh selection");
                assert_ne!(fresh.discovery_generation, selection.discovery_generation);
                assert_eq!(fresh.node_id, selection.node_id);
                let case_id = case.id.clone();
                assert!(panel.projection.can_run(&case_id));
                panel.host.expand_all();
                panel.host.select(case_id);
                panel.run_selected(window, cx);
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(40);
            let recovered = loop {
                cx.run_until_parked();
                let completed = structured.read_with(cx, |store, _| {
                    store
                        .state()
                        .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                        .and_then(|provider| provider.last_complete_run.clone())
                        .filter(|run| run.id != stopped_run.id)
                });
                if let Some(completed) = completed {
                    break completed;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "fresh SSH run did not complete"
                );
                cx.background_executor
                    .timer(Duration::from_millis(100))
                    .await;
            };
            assert_ne!(recovered.id, stopped_run.id);
            assert_eq!(recovered.summary.passed, 1);
            assert_eq!(recovered.events.len(), 1);
            assert_eq!(recovered.events[0].node_id, selection.node_id);
            assert_eq!(recovered.events[0].state, StructuredNodeState::Passed);
            eprintln!(
                "SSH fresh recovery run={:?}, summary={:?}",
                recovered.id, recovered.summary
            );
        }
        if use_ssh {
            let client = project.read_with(cx, |project, _| {
                project.remote_client().expect("SSH test client")
            });
            let shutdown = client
                .update(cx, |client, cx| {
                    client.shutdown_processes::<rpc::proto::ShutdownRemoteServer>(
                        Some(rpc::proto::ShutdownRemoteServer {}),
                        cx.background_executor().clone(),
                    )
                })
                .expect("connected fixture shutdown");
            shutdown.await;
        }
        return;
    }
    let mut completed = None;
    for _ in 0..160 {
        cx.run_until_parked();
        completed = structured.read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                .and_then(|provider| provider.last_complete_run.clone())
        });
        if completed.is_some() {
            break;
        }
        cx.background_executor
            .timer(Duration::from_millis(250))
            .await;
    }
    let completed = completed.unwrap_or_else(|| {
        let state = structured.read_with(cx, |store, _| {
            format!(
                "{:#?}",
                store
                    .state()
                    .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
            )
        });
        let notice = tests_panel.read_with(cx, |panel, _| panel.action_notice.clone());
        let terminal_output = terminal_panel.read_with(cx, |panel, cx| {
            panel.panes().into_iter().find_map(|pane| {
                pane.read(cx).items().find_map(|item| {
                    item.downcast::<terminal_view::TerminalView>()
                        .map(|view| view.read(cx).terminal().read(cx).get_content())
                })
            })
        });
        panic!(
            "real selected Java test did not complete: notice={notice:?}, terminal={terminal_output:?}, state={state}"
        )
    });
    let terminal_output = terminal_panel
        .read_with(cx, |panel, cx| {
            panel.panes().into_iter().find_map(|pane| {
                pane.read(cx).items().find_map(|item| {
                    item.downcast::<terminal_view::TerminalView>()
                        .and_then(|view| {
                            let terminal = view.read(cx).terminal().read(cx);
                            (!with_real_providers
                                || terminal
                                    .task()
                                    .is_some_and(|task| task.structured_task.is_some()))
                            .then(|| terminal.get_content())
                        })
                })
            })
        })
        .expect("selected Java Task terminal");
    if outcome == JavaTestOutcome::MissingMavenCache {
        assert_eq!(
            completed.phase,
            project::structured_execution::StructuredRunPhase::Completed
        );
        assert!(completed.events.is_empty());
        assert_eq!(completed.summary.passed, 0);
        assert!(completed.diagnostic.as_deref().is_some_and(|diagnostic| {
            diagnostic.contains("without a fresh readable JUnit result")
        }));
        assert!(terminal_output.contains("offline"), "{terminal_output}");
        assert!(terminal_output.contains("plugin"), "{terminal_output}");
        assert_eq!(
            std::fs::read_to_string(&report).expect("cached JUnit report"),
            std::fs::read_to_string(fixture.join("reports/TEST-evidence.GreetingTest.xml"))
                .expect("fixture JUnit report")
        );
        return;
    }
    assert_eq!(
        completed.phase,
        project::structured_execution::StructuredRunPhase::Completed
    );
    if should_fail {
        assert_eq!(completed.summary.failed, 1);
    } else {
        assert_eq!(
            completed.summary.passed, 1,
            "{completed:#?}; terminal={terminal_output}"
        );
    }
    assert_eq!(completed.events.len(), 1);
    assert_eq!(completed.events[0].node_id, selection.node_id);
    assert_eq!(
        completed.events[0].state,
        if should_fail {
            StructuredNodeState::Failed
        } else {
            StructuredNodeState::Passed
        }
    );
    let run_count = structured.read_with(cx, |store, _| {
        store
            .state()
            .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
            .expect("Java provider")
            .completed_runs
            .len()
    });
    assert_eq!(run_count, 1);
    if !use_gradle {
        assert!(terminal_output.contains("Tests run: 1"));
    }
    assert!(terminal_output.contains(if should_fail && use_gradle {
        "BUILD FAILED"
    } else if should_fail {
        "BUILD FAILURE"
    } else if use_gradle {
        "BUILD SUCCESSFUL"
    } else {
        "BUILD SUCCESS"
    }));
    let fresh_report = std::fs::read_to_string(&report).expect("fresh JUnit report");
    assert!(fresh_report.contains("messageIsAvailable"));
    if should_fail {
        assert!(fresh_report.contains("<failure"));
        let first_run_id = completed.id.clone();
        cx.run_until_parked();
        tests_panel.update_in(cx, |panel, window, cx| {
            assert_eq!(panel.projection.failed.len(), 1);
            panel.rerun_failed(window, cx);
        });
        let mut rerun = None;
        for _ in 0..120 {
            cx.run_until_parked();
            rerun = structured.read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                    .filter(|provider| provider.completed_runs.len() == 2)
                    .and_then(|provider| provider.last_complete_run.clone())
            });
            if rerun.is_some() {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(250))
                .await;
        }
        let rerun = rerun.expect("real failed-case rerun completed");
        assert_ne!(rerun.id, first_run_id);
        assert_eq!(rerun.summary.failed, 1);
        assert_eq!(rerun.events[0].node_id, selection.node_id);
        if with_real_providers {
            let (test_buffer, _handle) = project
                .update(cx, |project, cx| {
                    project.open_local_buffer_with_lsp(
                        project_root.join("application/src/test/java/evidence/GreetingTest.java"),
                        cx,
                    )
                })
                .await
                .expect("current Java test source");
            test_buffer.update(cx, |buffer, cx| {
                let original = buffer.text();
                assert!(original.contains(failing_assertion));
                let restored = original.replace(failing_assertion, passing_assertion);
                let end = buffer.len();
                buffer.edit([(0..end, restored)], None, cx);
            });
            project
                .update(cx, |project, cx| {
                    project.save_buffer(test_buffer.clone(), cx)
                })
                .await
                .expect("save repaired Java assertion");
            tests_panel.update(cx, |panel, cx| panel.discover(cx));
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                cx.run_until_parked();
                let fresh = tests_panel.read_with(cx, |panel, _| {
                    find_node(&panel.projection.snapshot.roots, case_label)
                        .and_then(|case| panel.projection.selection(&case.id))
                        .is_some_and(|fresh| {
                            fresh.node_id == selection.node_id
                                && fresh.discovery_generation != selection.discovery_generation
                        })
                });
                if fresh {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "repaired source needs fresh stable case discovery"
                );
                cx.background_executor
                    .timer(Duration::from_millis(100))
                    .await;
            }
            tests_panel.update_in(cx, |panel, window, cx| {
                let retained_failure_visible = !panel.projection.failed.is_empty();
                panel.rerun_failed(window, cx);
                assert_eq!(
                    panel.action_notice.as_deref(),
                    Some(if retained_failure_visible {
                        "The selected Java test is no longer a current failed case"
                    } else {
                        "No failed tests are available to rerun"
                    })
                );
                let case = find_node(&panel.projection.snapshot.roots, case_label)
                    .expect("fresh repaired test case");
                let fresh = panel
                    .projection
                    .selection(&case.id)
                    .expect("fresh selection");
                project
                    .read(cx)
                    .jvm_test_provider_store()
                    .expect("Java test provider")
                    .read_with(cx, |provider, cx| {
                        let error = provider
                            .plan_failed_case(
                                &fresh.node_id,
                                fresh.discovery_generation,
                                &rerun.id,
                                cx,
                            )
                            .expect_err("the retained failed result has expired");
                        assert_eq!(
                            error.to_string(),
                            "The selected Java test is no longer a current failed case"
                        );
                    });
                panel.host.expand_all();
                panel.host.select(case.id.clone());
                panel.run_selected(window, cx);
                assert!(panel.action_notice.is_none());
            });
            eprintln!(
                "Combined Tests: rediscovery rejected the retained failed generation; Run Selected uses the repaired current case"
            );
            let deadline = Instant::now() + Duration::from_secs(60);
            let recovered = loop {
                cx.run_until_parked();
                let recovered = structured.read_with(cx, |store, _| {
                    store
                        .state()
                        .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                        .filter(|provider| provider.completed_runs.len() == 3)
                        .and_then(|provider| provider.last_complete_run.clone())
                });
                if let Some(recovered) = recovered {
                    break recovered;
                }
                assert!(
                    Instant::now() < deadline,
                    "repaired source selected run did not finish"
                );
                cx.background_executor
                    .timer(Duration::from_millis(100))
                    .await;
            };
            assert_ne!(recovered.id, rerun.id);
            assert_eq!(recovered.summary.passed, 1);
            assert_eq!(recovered.summary.failed, 0);
            assert_eq!(recovered.events.len(), 1);
            assert_eq!(recovered.events[0].node_id, selection.node_id);
            assert_eq!(recovered.events[0].state, StructuredNodeState::Passed);
            let fresh_report = std::fs::read_to_string(&report).expect("repaired JUnit report");
            assert!(
                fresh_report.contains("messageIsAvailable") && !fresh_report.contains("<failure")
            );
            eprintln!(
                "Combined Tests: selected failure, distinct failed rerun and repaired pass share stable case {:?}",
                selection.node_id
            );
        }
    }
    if with_real_providers {
        let active_debug = if let Some(hooks) = debug_hooks {
            Some(
                (hooks.run)(
                    JvmWorkflowDebugContext {
                        project: project.clone(),
                        workspace: workspace.clone(),
                        project_root: project_root.clone(),
                        worktree_id,
                    },
                    cx,
                )
                .await,
            )
        } else {
            None
        };

        trust.update(cx, |trust, cx| {
            trust.restrict(
                worktree_store.downgrade(),
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        cx.run_until_parked();
        let panel = dependencies_panel.expect("combined Dependencies panel");
        panel.read_with(cx, |panel, _| {
            let (status, rows, navigation) = panel.snapshot_for_test();
            assert!(matches!(
                status,
                language_tools::language_tool_tree::LanguageToolTreeStatus::Restricted(_)
            ));
            assert!(rows.is_empty() && navigation.is_empty());
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let mut stopped = true;
            for (server, pid) in &real_provider_servers {
                let present = util::command::new_command("/bin/kill")
                    .args(["-0", &pid.to_string()])
                    .stderr(util::command::Stdio::null())
                    .status()
                    .await
                    .expect("probe restricted provider")
                    .success();
                stopped &= !present && server.process_id().is_none();
            }
            if let Some(target) = active_debug.as_ref() {
                let present = util::command::new_command("/bin/kill")
                    .args(["-0", &target.process_id.to_string()])
                    .stderr(util::command::Stdio::null())
                    .status()
                    .await
                    .expect("probe restricted Java debuggee")
                    .success();
                stopped &= !present
                    && target
                        .session
                        .read_with(cx, |session, _| session.is_terminated());
            }
            if stopped {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "combined providers or active Java Debug process did not exit after restriction"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
            cx.run_until_parked();
        }
        if let Some(target) = active_debug {
            assert!(project.read_with(cx, |project, cx| {
                project
                    .breakpoint_store()
                    .read(cx)
                    .active_position()
                    .is_none()
            }));
            eprintln!(
                "Combined Java Debug trust restriction terminated the paused session and actual process PID {}",
                target.process_id
            );
        }
        assert_eq!(real_provider_servers.len(), 2);
        for (_, pid) in real_provider_servers {
            eprintln!("Combined Tests provider PID {pid} stopped after restriction");
        }
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }
}

pub(super) fn find_node<'a>(
    nodes: &'a [LanguageToolNode],
    label: &str,
) -> Option<&'a LanguageToolNode> {
    for node in nodes {
        if node.label == label {
            return Some(node);
        }
        if let Some(found) = find_node(&node.children, label) {
            return Some(found);
        }
    }
    None
}
