use std::{
    borrow::Cow,
    cell::Cell,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

use collections::HashSet;
use futures::FutureExt as _;
use gpui::{BorrowAppContext as _, TestAppContext};
use language::{
    BufferEvent, FakeLspAdapter, Language, LanguageConfig, LanguageMatcher, LanguageQueries,
};
use lsp::LanguageServerSelector;
use project::{
    DEFAULT_COMPLETION_CONTEXT, LspAction, Project,
    jvm_project::KotlinWorkspaceDependency,
    structured_execution::StructuredNodeKind,
    trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees},
};
use serde_json::json;

fn copy_fixture(source: &Path, destination: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            // Prior fixture runs leave host-specific Gradle state and generated output.
            if entry.file_name() == ".gradle" || entry.file_name() == "build" {
                continue;
            }
            copy_fixture(&entry.path(), &destination.join(entry.file_name()))?;
        } else if file_type.is_file() {
            std::fs::copy(entry.path(), destination.join(entry.file_name()))?;
        } else {
            anyhow::bail!("unexpected fixture entry: {}", entry.path().display());
        }
    }
    Ok(())
}

#[gpui::test]
#[ignore = "requires a guarded Kotlin fallback 1.3.13 launcher, JDK, and isolated cached Maven or Gradle home"]
async fn cached_kotlin_fallback_navigates_unsaved_edits_across_restart(cx: &mut TestAppContext) {
    super::init_test(cx);
    cx.executor().allow_parking();
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));

    let fixture = std::env::var("GROOVY_KOTLIN_FALLBACK_FIXTURE").expect("fixture kind");
    assert!(matches!(fixture.as_str(), "maven" | "gradle"));
    let input = |key| {
        let path = PathBuf::from(std::env::var_os(key).unwrap_or_else(|| panic!("missing {key}")));
        assert!(path.is_absolute() && path.exists(), "invalid {key}");
        path
    };
    let launcher = input("GROOVY_KOTLIN_FALLBACK_LAUNCHER");
    let java_home = input("GROOVY_BUILD_JAVA_HOME");
    let cache = input("GROOVY_KOTLIN_FALLBACK_CACHE");
    let state = tempfile::tempdir().expect("isolated fallback state");
    let root = state.path().join("project");
    copy_fixture(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("test_data/groovy_jvm_{fixture}")),
        &root,
    )
    .expect("copy the checked-in mixed-language fixture");
    let root = root.canonicalize().expect("canonical project path");
    let kotlin = root.join("library/src/main/kotlin/evidence");
    let greeting = kotlin.join("Greeting.kt");
    let edited_greeting = kotlin.join("EditedGreetingTarget.kt");
    let source = kotlin.join("EditedDefinitionProbe.kt");
    let original = "package evidence\n\nfun editedDefinitionProbe() = Greeting().message()\n";
    std::fs::write(&source, original).expect("write the definition probe");
    std::fs::write(
        &edited_greeting,
        "package evidence\n\nclass EditedGreetingTarget { fun message(): String = \"edited\" }\n",
    )
    .expect("write a distinct definition target");
    for index in 0..500 {
        std::fs::write(
            kotlin.join(format!("GroovyScale{index:03}.kt")),
            format!("package evidence\n\nclass GroovyScale{index:03} {{ fun value() = Greeting().message() }}\n"),
        )
        .expect("write a fixture Kotlin source");
    }

    let home = state.path().join("home");
    std::fs::create_dir(&home).expect("isolated Kotlin JVM home");
    let mut java_options = format!(
        "-Duser.home={} -Djava.net.preferIPv4Stack=true",
        home.display()
    );
    let mut environment = json!({
        "JAVA_HOME": java_home,
        "KOTLIN_LANGUAGE_SERVER_OPTS": "-Xmx4g",
    });
    if fixture == "gradle" {
        environment["GRADLE_USER_HOME"] = json!(cache);
    } else {
        environment["MAVEN_REPOSITORY"] = json!(cache);
        environment["PATH"] = json!(std::env::var("PATH").expect("cached Maven launcher on PATH"));
        java_options.push_str(&format!(" -Dmaven.repo.local={}", cache.display()));
    }
    environment["JAVA_TOOL_OPTIONS"] = json!(java_options);
    let fallback = "kotlin-language-server";
    let settings = json!({
        "languages": {"Kotlin": {"language_servers": [fallback, "!kotlin-lsp", "..."]}},
        "lsp": {(fallback): {"binary": {"path": launcher, "env": environment}}},
    });
    cx.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store
                .set_user_settings(&settings.to_string(), cx)
                .expect("configure the installed fallback binary");
        })
    });
    let project = Project::test_with_worktree_trust(
        Arc::new(fs::RealFs::new(None, cx.executor())),
        [root.as_path()],
        cx,
    )
    .await;
    let worktrees = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktrees.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    cx.update(|cx| {
        let trusted = TrustedWorktrees::try_get_global(cx).expect("worktree trust store");
        trusted.update(cx, |trusted, cx| {
            trusted.trust(
                &worktrees,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
    });
    let languages = project.read_with(cx, |project, _| project.languages().clone());
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
            name: fallback,
            ..Default::default()
        },
    );
    let (buffer, _lsp_handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(&source, cx)
        })
        .await
        .expect("open the real fallback probe document");
    let start = original.rfind("Greeting").expect("original type reference");
    let definition = |cx: &mut TestAppContext| {
        project.update(cx, |project, cx| {
            project.definitions(&buffer, start + 1, cx)
        })
    };
    let target_paths = |definitions: Vec<project::LocationLink>, cx: &TestAppContext| {
        definitions
            .into_iter()
            .map(|definition| {
                definition.target.buffer.read_with(cx, |buffer, cx| {
                    buffer.file()?.as_local().map(|file| file.abs_path(cx))
                })
            })
            .collect::<Option<Vec<_>>>()
            .expect("only local source-file definitions")
    };
    let current_server = |cx: &TestAppContext| {
        project.read_with(cx, |project, cx| {
            project
                .language_server_statuses(cx)
                .find_map(|(id, status)| {
                    (status.name == lsp::LanguageServerName::new_static(fallback)
                        && status.server_version.as_deref() == Some("1.3.13"))
                    .then(|| status.process_id.map(|pid| (id, pid)))
                    .flatten()
                })
        })
    };
    let mut previous = None;
    for (original_type, original_target, edited_type, edited_target) in [
        (
            "Greeting",
            &greeting,
            "EditedGreetingTarget",
            &edited_greeting,
        ),
        (
            "EditedGreetingTarget",
            &edited_greeting,
            "Greeting",
            &greeting,
        ),
    ] {
        let deadline = Instant::now() + Duration::from_secs(180);
        let server = loop {
            cx.run_until_parked();
            if let Some(server) = current_server(cx)
                && previous.is_none_or(|(id, pid)| server.0 != id && server.1 != pid)
                && let Some(locations) = definition(cx)
                    .await
                    .expect("initial Kotlin definition request")
                && target_paths(locations, cx).as_slice() == std::slice::from_ref(original_target)
            {
                break server;
            }
            assert!(
                Instant::now() < deadline,
                "failed to establish current {fixture} definition"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        };
        buffer.update(cx, |buffer, cx| {
            let end = start + original_type.len();
            assert_eq!(
                buffer.text_for_range(start..end).collect::<String>(),
                original_type
            );
            buffer.edit([(start..end, edited_type)], None, cx);
        });
        let targets = target_paths(
            definition(cx)
                .await
                .expect("first definition after edit")
                .expect("Kotlin definition"),
            cx,
        );
        assert_eq!(targets.as_slice(), std::slice::from_ref(edited_target));
        assert_eq!(
            std::fs::read_to_string(&source).expect("read original fixture file"),
            original
        );
        println!(
            "{fixture} PID {} resolved the unsaved definition to {}",
            server.1,
            edited_target.display()
        );
        if previous.is_none() {
            assert!(buffer.read_with(cx, |buffer, _| buffer.is_dirty()));
            project.update(cx, |project, cx| {
                project.restart_language_servers_for_buffers(
                    vec![buffer.clone()],
                    HashSet::from_iter([LanguageServerSelector::Name(
                        lsp::LanguageServerName::new_static(fallback),
                    )]),
                    false,
                    cx,
                );
            });
        }
        previous = Some(server);
    }
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    lsp_store
        .update(cx, |store, cx| store.shutdown_all_language_servers(cx))
        .await;
}

#[gpui::test]
#[ignore = "requires GROOVY_JDTLS_LAUNCHER, GROOVY_KOTLIN_LSP, GROOVY_LSP_JAVA_HOME, GROOVY_BUILD_JAVA_HOME and cached Gradle dependencies"]
async fn cached_gradle_project_keeps_both_jvm_providers_and_test_identity(cx: &mut TestAppContext) {
    super::init_test(cx);
    cx.executor().allow_parking();
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let state = tempfile::tempdir().expect("isolated combined JVM workflow");
    let root = state.path().join("project");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/groovy_jvm_gradle");
    copy_fixture(&fixture, &root).expect("copy checked-in two-module Gradle fixture");
    assert!(!root.join(".gradle").exists());
    assert!(!root.join("application/build").exists());
    assert!(!root.join("library/build").exists());
    let root = root.canonicalize().expect("canonical JVM fixture");
    let server_java_home = std::env::var("GROOVY_LSP_JAVA_HOME").expect("explicit server JDK");
    let build_java_home =
        std::env::var("GROOVY_BUILD_JAVA_HOME").expect("explicit fixture build JDK");
    for home in [&server_java_home, &build_java_home] {
        assert!(Path::new(home).is_absolute() && Path::new(home).join("bin/java").is_file());
    }
    std::fs::write(
        root.join("gradle.properties"),
        format!(
            "org.gradle.java.home={}\norg.gradle.daemon=false\n",
            build_java_home
        ),
    )
    .expect("explicit cached-build Java home");
    let mut build = util::command::new_command("/bin/sh");
    // Match the loopback address admitted by the existing network-denial fixture.
    build
        .args(["gradlew", "--offline", "--no-daemon", ":application:build"])
        .current_dir(&root)
        .env("JAVA_HOME", &build_java_home)
        .env("JAVA_TOOL_OPTIONS", "-Djava.net.preferIPv4Stack=true")
        .kill_on_drop(true);
    let output = build.output().fuse();
    let timeout = cx
        .background_executor
        .timer(Duration::from_secs(180))
        .fuse();
    futures::pin_mut!(output, timeout);
    let output = futures::select_biased! {
        output = output => output.expect("prime controlled cached fixture"),
        () = timeout => panic!("cached fixture preparation exceeded 180 seconds"),
    };
    assert!(
        output.status.success(),
        "fixture preparation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        root.join(
            "application/build/generated/sources/groovyFixture/java/evidence/BuildMessage.java"
        )
        .is_file()
    );
    assert!(
        root.join("library/build/classes/kotlin/main/evidence/Greeting.class")
            .is_file()
    );

    let java_settings = json!({
        "import": {"gradle": {"enabled": true, "java": {"home": build_java_home}, "offline": {"enabled": true}}},
        "jdt": {"ls": {"kotlinSupport": {"enabled": true}}}
    });
    let mut server_settings = serde_json::Map::new();
    for (name, variable) in [
        ("jdtls", "GROOVY_JDTLS_LAUNCHER"),
        ("kotlin-lsp", "GROOVY_KOTLIN_LSP"),
    ] {
        let launcher = std::env::var(variable).expect("explicit provider launcher");
        assert!(Path::new(&launcher).is_absolute() && Path::new(&launcher).is_file());
        let mut arguments = vec![launcher];
        if name == "jdtls" {
            arguments.extend([
                format!("--java-executable={server_java_home}/bin/java"),
                "-data".to_string(),
                state
                    .path()
                    .join("jdtls-data")
                    .to_string_lossy()
                    .into_owned(),
            ]);
        } else {
            arguments.push("--stdio".to_string());
        }
        let tool_options = ["config", "system", "plugins", "log"]
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
        server_settings.insert(
            name.to_string(),
            json!({"binary": {
                "path": arguments.remove(0), "arguments": arguments,
                "env": {"JAVA_HOME": server_java_home, "JAVA_TOOL_OPTIONS": tool_options}
            }}),
        );
    }
    server_settings.get_mut("jdtls").expect("Java settings")["settings"] =
        json!({"java": java_settings});
    let settings = json!({"lsp": server_settings});
    std::fs::create_dir_all(root.join(".zed")).expect("project settings directory");
    std::fs::write(root.join(".zed/settings.json"), settings.to_string())
        .expect("provider overrides");
    cx.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store
                .set_user_settings(&settings.to_string(), cx)
                .expect("explicit provider overrides");
        })
    });
    let project = Project::test_with_worktree_trust(
        Arc::new(fs::RealFs::new(None, cx.executor())),
        [root.as_path()],
        cx,
    )
    .await;
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    let java = Language::new(LanguageConfig {
        name: "Java".into(), matcher: LanguageMatcher { path_suffixes: vec!["java".into()], ..Default::default() }.into(),
        ..Default::default()
    }, Some(tree_sitter_java::LANGUAGE.into())).with_queries(LanguageQueries {
        runnables: Some(Cow::Borrowed(r#"(program
            (package_declaration [(identifier) (scoped_identifier)] @java_package_name)?
            (class_declaration name: (identifier) @java_class_name body: (class_body
                (method_declaration (modifiers (marker_annotation name: (identifier) @annotation_name))
                    name: (identifier) @run @java_method_name (#match? @annotation_name "Test$")))) @_
            (#set! tag java-test-method))"#)), ..Default::default()
    }).expect("Java runnable query");
    languages.add(Arc::new(java));
    languages.register_fake_lsp_adapter(
        "Java",
        FakeLspAdapter {
            name: "jdtls",
            initialization_options: Some(json!({"settings": {"java": java_settings}})),
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
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("combined worktree")
            .read(cx)
            .id()
    });
    let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("shared trust owner"));
    trust.update(cx, |trust, cx| {
        trust.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        )
    });
    let java_path = root.join("application/src/main/java/evidence/Main.java");
    let java_source = std::fs::read_to_string(&java_path).expect("Java source");
    let (java_buffer, _java_handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(java_path, cx)
        })
        .await
        .expect("open Java buffer");
    let (kotlin_buffer, _kotlin_handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(
                root.join("library/src/main/kotlin/evidence/Greeting.kt"),
                cx,
            )
        })
        .await
        .expect("open Kotlin buffer");
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let deadline = Instant::now() + Duration::from_secs(240);
    let mut refresh_at = Instant::now();
    loop {
        cx.run_until_parked();
        let imported = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(snapshot, current)| current && snapshot.projects.len() >= 2)
                && store
                    .kotlin_workspace_snapshot(worktree_id)
                    .is_some_and(|(snapshot, current)| current && snapshot.projects.len() >= 2)
        });
        if imported {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "both provider project snapshots did not become current within 240 seconds"
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
        let (java, _) = store
            .jdtls_project_snapshot(worktree_id)
            .expect("Java project snapshot");
        let (kotlin, _) = store
            .kotlin_workspace_snapshot(worktree_id)
            .expect("Kotlin workspace snapshot");
        assert_ne!(java.server_id, kotlin.server_id);
        assert!(
            java.projects
                .iter()
                .any(|project| project.library_entries > 0)
        );
        assert!(
            kotlin
                .projects
                .iter()
                .any(|project| project.dependencies.iter().any(|dependency| {
                    matches!(dependency, KotlinWorkspaceDependency::Artifact { .. })
                }))
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
                .any(|project| project.source_roots.iter().any(|root| root
                    .path
                    .path
                    .as_unix_str()
                    .ends_with("library/src/main/kotlin")))
        );
        println!(
            "Combined provider snapshots: Java {} projects/Kotlin {} projects",
            java.projects.len(),
            kotlin.projects.len()
        );
    });
    let greeting_offset = java_source
        .find("Greeting()")
        .expect("Java Kotlin reference");
    let definitions = project
        .update(cx, |project, cx| {
            project.definitions(&java_buffer, greeting_offset, cx)
        })
        .await
        .expect("shared Java definition request")
        .expect("Java definition result");
    assert!(
        definitions.iter().any(|definition| definition
            .target
            .buffer
            .read_with(cx, |buffer, _| buffer.text().contains("class Greeting"))),
        "Java reference did not resolve to Kotlin source"
    );
    let generated_offset = java_source
        .find("BuildMessage.value")
        .expect("generated source reference");
    let definitions = project
        .update(cx, |project, cx| {
            project.definitions(&java_buffer, generated_offset, cx)
        })
        .await
        .expect("generated definition request")
        .expect("generated definition result");
    assert!(
        definitions.iter().any(|definition| definition
            .target
            .buffer
            .read_with(cx, |buffer, _| buffer.text().contains("class BuildMessage"))),
        "generated source was not navigable"
    );
    assert!(kotlin_buffer.read_with(cx, |buffer, _| buffer.text().contains("fun message")));
    let original_kotlin = kotlin_buffer.read_with(cx, |buffer, _| buffer.text());
    let probe = "\nfun workflowProbe() = Greeting().message()\n";
    let probe_offset = original_kotlin.len() + probe.find("message()").expect("Kotlin call");
    kotlin_buffer.update(cx, |buffer, cx| {
        let end = buffer.len();
        buffer.edit([(end..end, probe)], None, cx);
    });
    project
        .update(cx, |project, cx| {
            project.save_buffer(kotlin_buffer.clone(), cx)
        })
        .await
        .expect("save Kotlin editor change");
    let definitions = project
        .update(cx, |project, cx| {
            project.definitions(&kotlin_buffer, probe_offset, cx)
        })
        .await
        .expect("shared changed-Kotlin definition request")
        .expect("Kotlin definition result");
    assert!(
        definitions.iter().any(|definition| {
            definition.target.buffer == kotlin_buffer
                && definition.target.buffer.read_with(cx, |buffer, _| {
                    buffer
                        .text_for_range(definition.target.range.clone())
                        .collect::<String>()
                        .contains("message")
                })
        }),
        "edited Kotlin method call did not resolve to its declaration"
    );
    let completions = project
        .update(cx, |project, cx| {
            project.completions(
                &kotlin_buffer,
                probe_offset + 2,
                DEFAULT_COMPLETION_CONTEXT,
                cx,
            )
        })
        .await
        .expect("Kotlin completions through shared Project");
    assert!(completions.iter().any(|response| {
        response
            .completions
            .iter()
            .any(|completion| completion.label.text.starts_with("message"))
    }));
    let kotlin_diagnostic_updates = Rc::new(Cell::new(0));
    let _kotlin_diagnostic_subscription = cx.update(|cx| {
        cx.subscribe(&kotlin_buffer, {
            let diagnostic_updates = kotlin_diagnostic_updates.clone();
            move |_, event, _| {
                if matches!(event, BufferEvent::DiagnosticsUpdated) {
                    diagnostic_updates.set(diagnostic_updates.get() + 1);
                }
            }
        })
    });
    let invalid_kotlin_probe = "\nfun invalidWorkflowProbe() = missingKotlinWorkflow()\n";
    let kotlin_error_offset = kotlin_buffer.read_with(cx, |buffer, _| buffer.len())
        + invalid_kotlin_probe
            .find("missingKotlinWorkflow")
            .expect("invalid Kotlin call");
    kotlin_buffer.update(cx, |buffer, cx| {
        let end = buffer.len();
        buffer.edit([(end..end, invalid_kotlin_probe)], None, cx);
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        cx.run_until_parked();
        let diagnosed = kotlin_buffer.read_with(cx, |buffer, _| {
            buffer
                .snapshot()
                .diagnostics_in_range::<_, usize>(
                    kotlin_error_offset..kotlin_error_offset + 21,
                    false,
                )
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
            "current invalid Kotlin edit was not diagnosed"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
    let invalid_kotlin_diagnostic_update = kotlin_diagnostic_updates.get();
    kotlin_buffer.update(cx, |buffer, cx| {
        let end = buffer.len();
        buffer.edit([(0..end, original_kotlin)], None, cx);
    });
    project
        .update(cx, |project, cx| {
            project.save_buffer(kotlin_buffer.clone(), cx)
        })
        .await
        .expect("restore Kotlin fixture source");
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
            "restored Kotlin source has no fresh error-free diagnostics"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
    println!("Combined Kotlin completion, current error and fresh recovery passed");

    let probe_path = root.join("application/src/main/java/evidence/LanguageProbe.java");
    let original_java = "package evidence;\npublic final class LanguageProbe {\n    public static String message() { return \"ready\"; }\n    public static void main(String[] arguments) {\n        System.out.println(message());\n    }\n}\n";
    std::fs::write(&probe_path, original_java).expect("disposable Java language fixture");
    let (probe_buffer, _probe_handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(probe_path, cx)
        })
        .await
        .expect("open Java language fixture through shared Project");
    let diagnostic_updates = Rc::new(Cell::new(0));
    let _diagnostic_subscription = cx.update(|cx| {
        cx.subscribe(&probe_buffer, {
            let diagnostic_updates = diagnostic_updates.clone();
            move |_, event, _| {
                if matches!(event, BufferEvent::DiagnosticsUpdated) {
                    diagnostic_updates.set(diagnostic_updates.get() + 1);
                }
            }
        })
    });
    let completions = project
        .update(cx, |project, cx| {
            project.completions(
                &probe_buffer,
                original_java
                    .find("println")
                    .expect("Java completion prefix")
                    + 2,
                DEFAULT_COMPLETION_CONTEXT,
                cx,
            )
        })
        .await
        .expect("Java completions through shared Project");
    assert!(completions.iter().any(|response| {
        response
            .completions
            .iter()
            .any(|completion| completion.label.text.starts_with("println"))
    }));
    let invalid_java = original_java.replace("return \"ready\";", "return missingMessage();");
    let call_offset = invalid_java
        .find("missingMessage()")
        .expect("invalid Java call");
    probe_buffer.update(cx, |buffer, cx| {
        let end = buffer.len();
        buffer.edit([(0..end, invalid_java.as_str())], None, cx);
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        cx.run_until_parked();
        let diagnosed = probe_buffer.read_with(cx, |buffer, _| {
            buffer
                .snapshot()
                .diagnostics_in_range::<_, usize>(call_offset..call_offset + 14, false)
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
            "current invalid Java edit was not diagnosed"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
    let actions = project
        .update(cx, |project, cx| {
            project.code_actions(
                &probe_buffer,
                call_offset..call_offset + 14,
                Some(vec![lsp::CodeActionKind::QUICKFIX]),
                cx,
            )
        })
        .await
        .expect("current Java quick-fix request")
        .expect("Java quick-fix result");
    let action = actions
        .into_iter()
        .find(|action| {
            matches!(&action.lsp_action, LspAction::Action(action)
                if action.disabled.is_none()
                    && action.title.starts_with("Create method")
                    && action.title.contains("missingMessage"))
        })
        .expect("enabled missing-method creation action");
    let invalid_diagnostic_update = diagnostic_updates.get();
    let transaction = project
        .update(cx, |project, cx| {
            project.apply_code_action(probe_buffer.clone(), action, true, cx)
        })
        .await
        .expect("apply provider quick fix through shared Project");
    assert_eq!(transaction.0.len(), 1);
    assert!(transaction.0.contains_key(&probe_buffer));
    let repaired_java = probe_buffer.read_with(cx, |buffer, _| buffer.text());
    assert!(repaired_java.contains("private static String missingMessage()"));
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
            "Java quick fix did not clear the observed error"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
    let definitions = project
        .update(cx, |project, cx| {
            project.definitions(&probe_buffer, call_offset, cx)
        })
        .await
        .expect("quick-fix definition request")
        .expect("quick-fix definition result");
    assert!(definitions.iter().any(|definition| {
        definition.target.buffer == probe_buffer
            && definition.target.buffer.read_with(cx, |buffer, _| {
                buffer
                    .text_for_range(definition.target.range.clone())
                    .collect::<String>()
                    .contains("missingMessage")
            })
    }));
    let fixed_diagnostic_update = diagnostic_updates.get();
    probe_buffer.update(cx, |buffer, cx| {
        let end = buffer.len();
        buffer.edit([(0..end, original_java)], None, cx);
    });
    project
        .update(cx, |project, cx| {
            project.save_buffer(probe_buffer.clone(), cx)
        })
        .await
        .expect("restore Java language fixture");
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
            "restored Java source has no fresh error-free diagnostics"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
    println!("Combined Java completion, current error, applied quick fix and definition passed");
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("shared JUnit provider")
            .clone()
    });
    let first = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("current source-backed discovery");
    let case = first
        .nodes
        .iter()
        .find(|node| {
            node.kind == StructuredNodeKind::Case && node.label.starts_with("messageIsAvailable")
        })
        .expect("real Java JUnit method");
    provider
        .read_with(cx, |provider, cx| {
            provider.plan_selected_case(&case.id, first.discovery_generation, cx)
        })
        .expect("selected source-backed Task plan");
    let second = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("repeat discovery");
    assert!(second.nodes.iter().any(|node| node.id == case.id));
    assert_ne!(first.discovery_generation, second.discovery_generation);
    assert!(
        provider
            .read_with(cx, |provider, cx| provider.plan_selected_case(
                &case.id,
                first.discovery_generation,
                cx
            ))
            .is_err()
    );
    provider
        .read_with(cx, |provider, cx| {
            provider.plan_selected_case(&case.id, second.discovery_generation, cx)
        })
        .expect("fresh Task plan for stable identity");
    let servers = lsp_store.read_with(cx, |store, _| {
        let java = store
            .jdtls_project_snapshot(worktree_id)
            .expect("current Java")
            .0;
        let kotlin = store
            .kotlin_workspace_snapshot(worktree_id)
            .expect("current Kotlin")
            .0;
        let local = store.as_local().expect("local provider owner");
        [java.server_id, kotlin.server_id]
            .into_iter()
            .map(|id| {
                let server = local
                    .running_language_server_for_id(id)
                    .expect("running provider")
                    .clone();
                let process_id = server.process_id().expect("real provider process");
                (server, process_id)
            })
            .collect::<Vec<_>>()
    });
    trust.update(cx, |trust, cx| {
        trust.restrict(
            worktree_store.downgrade(),
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        )
    });
    cx.run_until_parked();
    assert!(lsp_store.read_with(
        cx,
        |store, _| store.jdtls_project_snapshot(worktree_id).is_none()
            && store.kotlin_workspace_snapshot(worktree_id).is_none()
    ));
    assert!(
        provider
            .read_with(cx, |provider, cx| provider.plan_selected_case(
                &case.id,
                second.discovery_generation,
                cx
            ))
            .is_err()
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    // Clearing the LSP process handle can precede the OS reporting PID absence.
    loop {
        let mut stopped = true;
        for (server, process_id) in &servers {
            let present = util::command::new_command("kill")
                .args(["-0", &process_id.to_string()])
                .stderr(util::command::Stdio::null())
                .status()
                .await
                .expect("probe restricted provider")
                .success();
            stopped &= !present && server.process_id().is_none();
        }
        if stopped {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "restricted providers did not exit within 30 seconds"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
        cx.run_until_parked();
    }
    for (server, process_id) in servers {
        assert!(
            server.process_id().is_none(),
            "restricted provider did not stop"
        );
        assert!(
            !util::command::new_command("kill")
                .args(["-0", &process_id.to_string()])
                .stderr(util::command::Stdio::null())
                .status()
                .await
                .expect("probe stopped provider")
                .success(),
            "restricted provider PID remains alive"
        );
        println!("Combined provider PID {process_id} stopped after restriction");
    }
    cx.run_until_parked();
}
