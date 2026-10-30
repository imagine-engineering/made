use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use collections::HashMap;
use fs::FakeFs;
#[cfg(feature = "jvm-tools")]
use futures::FutureExt as _;
use futures::StreamExt;
#[cfg(any(feature = "rust-tests", feature = "jvm-tools"))]
use gpui::BorrowAppContext as _;
use gpui::TestAppContext;
use language::{CodeLabel, FakeLspAdapter, HighlightId, rust_lang};
use lsp::{LanguageServerName, Uri};
use parking_lot::Mutex;
use project::{
    Project,
    lsp_store::{
        log_store::{TestRpcLogHeaderState, TestRpcRequestTracker},
        *,
    },
};
use serde_json::json;
use util::path;

use crate::init_test;

#[gpui::test]
async fn host_provider_capability_denial_status_preserves_copper_and_names_jvm_recovery(
    cx: &mut TestAppContext,
) {
    use language::{BinaryStatus, LspAdapterDelegate as _};

    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/provider-status"),
        json!({ "pom.xml": "<project/>" }),
    )
    .await;
    let project = Project::test(fs, [path!("/provider-status").as_ref()], cx).await;
    let (lsp_store, worktree, languages) = project.read_with(cx, |project, cx| {
        (
            project.lsp_store(),
            project
                .worktrees(cx)
                .next()
                .expect("status fixture worktree"),
            project.languages().clone(),
        )
    });
    let delegate = lsp_store.update(cx, |store, cx| {
        LocalLspAdapterDelegate::from_local_lsp(
            store.as_local().expect("local project LSP owner"),
            &worktree,
            cx,
        )
    });
    let mut updates = languages.language_server_binary_statuses();
    let denied_download = "capability for download_file https://private.invalid/server.zip?token=private-token is not granted by the extension host";
    let denied_java_probe = r#"Failed to build JDTLS launch arguments: Failed to determine Java version: Failed to execute '/private/host/java -version': capability for process:exec /private/host/java ["-version"] is not granted by the extension host"#;
    for (denied, download) in [(denied_download, true), (denied_java_probe, false)] {
        for server in [
            "jdtls",
            "kotlin-lsp",
            "kotlin-language-server",
            "gradle-language-server",
            "rust-analyzer",
        ] {
            let name = LanguageServerName::from(server);
            delegate.update_status(
                name.clone(),
                BinaryStatus::Failed {
                    error: denied.to_string(),
                },
            );
            let (updated_name, status) =
                updates.next().await.expect("shared binary failure update");
            assert_eq!(updated_name, name);
            let BinaryStatus::Failed { error } = status else {
                panic!("a denied provider capability must stay failed");
            };
            let mapped = cfg!(feature = "jvm-tools")
                && (server == "jdtls" || (download && server != "rust-analyzer"));
            if mapped {
                assert!(error.contains("project host"), "{error}");
                assert!(error.contains("application or remote server"), "{error}");
                assert!(!error.contains("private"), "{error}");
                if server.starts_with("kotlin") {
                    assert!(error.contains("selection remains unchanged"), "{error}");
                } else if !download {
                    assert!(
                        error.contains("process permissions for Java -version"),
                        "{error}"
                    );
                }
            } else {
                assert_eq!(error, denied);
            }
        }
    }
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jdtls_first_initialize_uses_current_saved_selection(cx: &mut TestAppContext) {
    use language::{BinaryStatus, Language, LanguageConfig, LanguageMatcher};
    init_test(cx);
    let settings_bytes = |selection| {
        json!({"jvm_builds":{"maven":{"selected_poms":selection}}})
            .to_string()
            .into_bytes()
    };
    for (name, settings, enabled) in [
        ("default", b"{}".to_vec(), Some(true)),
        ("empty", settings_bytes(json!([])), Some(false)),
        ("selected", settings_bytes(json!(["pom.xml"])), Some(false)),
        ("invalid", settings_bytes(json!([17])), None),
        ("unreadable", vec![0xff], None),
    ] {
        let fs = FakeFs::new(cx.executor());
        let root = std::path::Path::new(path!("/first-selection")).join(name);
        fs.insert_tree(
            &root,
            json!({
                ".zed": {},
                "Main.java": "class Main {}", "pom.xml": "<project/>"
            }),
        )
        .await;
        fs.insert_file(root.join(".zed/settings.json"), settings)
            .await;
        let project = Project::test(fs, [root.as_path()], cx).await;
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let initialized = Arc::new(Mutex::new(Vec::new()));
        let mut servers = languages.register_fake_lsp(
            "Java",
            FakeLspAdapter {
                name: "jdtls",
                initialization_options: Some(
                    json!({"settings":{"java":{"import":{"maven":{"enabled":true}}}}}),
                ),
                initializer: Some(Box::new({
                    let initialized = initialized.clone();
                    move |server| {
                        let initialized = initialized.clone();
                        server.set_request_handler::<lsp::request::Initialize, _, _>(
                            move |params, _| {
                                initialized.lock().push(params.initialization_options);
                                async { Ok(lsp::InitializeResult::default()) }
                            },
                        );
                    }
                })),
                ..Default::default()
            },
        );
        let mut statuses = languages.language_server_binary_statuses();
        let buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp(root.join("Main.java"), cx)
            })
            .await
            .expect("first Java buffer");
        for _ in 0..20 {
            cx.run_until_parked();
            cx.executor().advance_clock(Duration::from_millis(50));
        }
        let allocated =
            std::iter::from_fn(|| servers.next().now_or_never().flatten()).collect::<Vec<_>>();
        if let Some(enabled) = enabled {
            assert_eq!(allocated.len(), 1, "only current provider setup: {name}");
            let initialized = initialized.lock();
            assert_eq!(initialized.len(), 1, "only current Initialize: {name}");
            assert_eq!(
                initialized
                    .first()
                    .and_then(Option::as_ref)
                    .and_then(|options| options.pointer("/settings/java/import/maven/enabled")),
                Some(&json!(enabled)),
                "{name}"
            );
        } else {
            assert!(
                allocated.is_empty(),
                "invalid saved selection cannot set up JDTLS"
            );
            assert!(
                initialized.lock().is_empty(),
                "invalid saved selection cannot initialize JDTLS"
            );
            let updates =
                std::iter::from_fn(|| statuses.next().now_or_never().flatten()).collect::<Vec<_>>();
            assert!(
                updates
                    .iter()
                    .any(|(_, status)| matches!(status, BinaryStatus::Failed { .. }))
            );
            assert!(
                updates.iter().all(|(_, status)| matches!(
                    status,
                    BinaryStatus::None
                        | BinaryStatus::Starting
                        | BinaryStatus::Stopping
                        | BinaryStatus::Stopped
                        | BinaryStatus::Failed { .. }
                )),
                "invalid settings cannot begin download discovery"
            );
        }
        drop((allocated, buffer, project));
        cx.update(|_| {});
        cx.run_until_parked();
    }
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn kotlin_first_setup_honors_saved_fallback_selection(cx: &mut TestAppContext) {
    use language::{Language, LanguageConfig, LanguageMatcher};
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    let root = std::path::Path::new(path!("/first-kotlin-selection"));
    fs.insert_tree(root, json!({
        ".zed": {"settings.json": json!({"languages":{"Kotlin":{"language_servers":["kotlin-language-server"]}}}).to_string()},
        "Main.kt": "fun main() {}"
    })).await;
    let project = Project::test(fs, [root], cx).await;
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
    let mut preferred = languages.register_fake_lsp(
        "Kotlin",
        FakeLspAdapter {
            name: "kotlin-lsp",
            ..Default::default()
        },
    );
    let mut fallback = languages.register_fake_lsp(
        "Kotlin",
        FakeLspAdapter {
            name: "kotlin-language-server",
            ..Default::default()
        },
    );
    let _buffer = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(root.join("Main.kt"), cx)
        })
        .await
        .expect("first Kotlin buffer");
    for _ in 0..20 {
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_millis(50));
    }
    assert!(
        preferred.next().now_or_never().is_none(),
        "obsolete preferred selection cannot reach setup"
    );
    let _server = fallback
        .next()
        .now_or_never()
        .flatten()
        .expect("current saved fallback setup");
    assert!(
        fallback.next().now_or_never().is_none(),
        "one fallback setup"
    );
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jdtls_saved_import_controls_follow_shared_startup_restart_and_configuration(
    cx: &mut TestAppContext,
) {
    use language::{BinaryStatus, Language, LanguageConfig, LanguageMatcher};

    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    let selection = json!({"jvm_builds": {"maven": {"selected_poms": []}}});
    fs.insert_tree(
        path!("/startup-controls"),
        json!({
            ".zed": {"settings.json": selection.to_string()},
            "Main.java": "class Main {}"
        }),
    )
    .await;
    let project = Project::test(fs.clone(), [path!("/startup-controls").as_ref()], cx).await;
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(Arc::new(Language::new(
        LanguageConfig {
            name: "Java".into(),
            matcher: LanguageMatcher {
                path_suffixes: vec!["java".into()],
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
            name: "startup-configuration-contributor",
            additional_workspace_configuration: HashMap::from_iter([(
                LanguageServerName::new_static("jdtls"),
                json!({"java": {"import": {
                "maven": {"enabled": true}, "gradle": {"enabled": true}
            }}, "preserved": "shared-owner"}),
            )]),
            ..Default::default()
        },
    );
    let initialized_options = Arc::new(Mutex::new(Vec::new()));
    let mut servers = languages.register_fake_lsp(
        "Java",
        FakeLspAdapter {
            name: "jdtls",
            initialization_options: Some(
                json!({"bundles": ["fixture-bundle"], "settings": {"java": {"import": {
                    "maven": {"enabled": true}, "gradle": {"enabled": true}
                }}}}),
            ),
            initializer: Some(Box::new({
                let initialized_options = initialized_options.clone();
                move |server| {
                    let initialized_options = initialized_options.clone();
                    server.set_request_handler::<lsp::request::Initialize, _, _>(
                        move |params, _| {
                            initialized_options
                                .lock()
                                .push(params.initialization_options);
                            async move { Ok(lsp::InitializeResult::default()) }
                        },
                    );
                    server.set_request_handler::<lsp::request::ExecuteCommand, _, _>(
                        |_, _| async { Ok(None) },
                    );
                }
            })),
            ..Default::default()
        },
    );
    let mut statuses = languages.language_server_binary_statuses();
    let _buffer = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/startup-controls/Main.java"), cx)
        })
        .await
        .expect("Java buffer");
    let mut maven_controlled = servers.next().await.expect("controlled Maven startup");
    let configuration = maven_controlled
        .try_receive_notification::<lsp::notification::DidChangeConfiguration>()
        .await
        .expect("initial controlled startup publishes configuration")
        .settings;
    assert_eq!(
        configuration.pointer("/java/import/maven/enabled"),
        Some(&json!(false))
    );
    assert_eq!(
        configuration.pointer("/java/import/gradle/enabled"),
        Some(&json!(true))
    );
    assert_eq!(configuration["preserved"], json!("shared-owner"));
    let requested = maven_controlled
        .request::<lsp::request::WorkspaceConfiguration>(
            lsp::ConfigurationParams {
                items: vec![lsp::ConfigurationItem {
                    scope_uri: None,
                    section: None,
                }],
            },
            Duration::from_secs(5),
        )
        .await
        .into_response()
        .expect("requested configuration");
    assert_eq!(requested, vec![configuration]);

    fs.insert_file(
        path!("/startup-controls/.zed/settings.json"),
        json!({
            "jvm_builds": {"maven": {"selected_poms": []}},
            "lsp": {"jdtls": {"settings": {"java_home": "fixture-only-change"}}}
        })
        .to_string()
        .into_bytes(),
    )
    .await;
    cx.run_until_parked();
    let configuration = maven_controlled
        .try_receive_notification::<lsp::notification::DidChangeConfiguration>()
        .await
        .expect("dynamic settings publish configuration on the current server")
        .settings;
    assert_eq!(
        configuration.pointer("/java/import/maven/enabled"),
        Some(&json!(false))
    );
    assert_eq!(
        configuration.pointer("/java/import/gradle/enabled"),
        Some(&json!(true))
    );
    assert_eq!(initialized_options.lock().len(), 1);
    assert!(
        servers.next().now_or_never().is_none(),
        "dynamic settings preserve the current server"
    );

    for (settings, maven, gradle) in [
        (
            json!({"jvm_builds": {"gradle": {"selected_build_files": []}}}),
            true,
            false,
        ),
        (json!({}), true, true),
    ] {
        fs.insert_file(
            path!("/startup-controls/.zed/settings.json"),
            settings.to_string().into_bytes(),
        )
        .await;
        cx.run_until_parked();
        let mut restarted = servers
            .next()
            .await
            .expect("selection control removal restarts existing server");
        let configuration = restarted
            .try_receive_notification::<lsp::notification::DidChangeConfiguration>()
            .await
            .expect("selection removal replacement publishes configuration")
            .settings;
        assert_eq!(
            configuration.pointer("/java/import/maven/enabled"),
            Some(&json!(maven))
        );
        assert_eq!(
            configuration.pointer("/java/import/gradle/enabled"),
            Some(&json!(gradle))
        );
        assert_eq!(configuration["preserved"], json!("shared-owner"));
    }
    cx.run_until_parked();
    assert_eq!(initialized_options.lock().len(), 3);
    for (options, (maven, gradle)) in
        initialized_options
            .lock()
            .iter()
            .zip([(false, true), (true, false), (true, true)])
    {
        let options = options.as_ref().expect("initialization options");
        assert!(options.get("projectConfigurations").is_none());
        assert_eq!(options["bundles"], json!(["fixture-bundle"]));
        assert_eq!(
            options.pointer("/settings/java/import/maven/enabled"),
            Some(&json!(maven))
        );
        assert_eq!(
            options.pointer("/settings/java/import/gradle/enabled"),
            Some(&json!(gradle))
        );
    }
    fs.insert_file(path!("/startup-controls/.zed/settings.json"), json!({
        "jvm_builds": {"maven": {"selected_poms": []}},
        "lsp": {"jdtls": {"initialization_options": {"projectConfigurations": ["private-canary"]}}}
    }).to_string().into_bytes()).await;
    cx.run_until_parked();
    let failures = std::iter::from_fn(|| statuses.next().now_or_never().flatten())
        .filter_map(|(_, status)| match status {
            BinaryStatus::Failed { error } => Some(error),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(failures.iter().any(
        |error| error.contains("Saved JVM build selection conflicts")
            && error.contains("restart the language server")
            && !error.contains("private-canary")
    ));
    assert!(
        servers.next().now_or_never().is_none(),
        "invalid selection cannot start provider setup"
    );
    assert_eq!(initialized_options.lock().len(), 3);
    fs.insert_file(
        path!("/startup-controls/.zed/settings.json"),
        json!({
            "jvm_builds": {"maven": {"selected_poms": [17]}}
        })
        .to_string()
        .into_bytes(),
    )
    .await;
    cx.run_until_parked();
    let malformed_failures = std::iter::from_fn(|| statuses.next().now_or_never().flatten())
        .filter_map(|(_, status)| match status {
            BinaryStatus::Failed { error } => Some(error),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        malformed_failures
            .iter()
            .any(|error| error.contains("settings contain errors")
                && error.contains("repair user or project settings"))
    );
    assert_eq!(initialized_options.lock().len(), 3);
    assert!(
        servers.next().now_or_never().is_none(),
        "malformed saved settings cannot default to automatic import"
    );
    fs.insert_file(
        path!("/startup-controls/.zed/settings.json"),
        selection.to_string().into_bytes(),
    )
    .await;
    cx.run_until_parked();
    let mut recovered = servers
        .next()
        .await
        .expect("repaired settings recover existing launch owner");
    let configuration = recovered
        .try_receive_notification::<lsp::notification::DidChangeConfiguration>()
        .await
        .expect("repaired settings publish configuration")
        .settings;
    assert_eq!(
        configuration.pointer("/java/import/maven/enabled"),
        Some(&json!(false))
    );
    assert_eq!(
        configuration.pointer("/java/import/gradle/enabled"),
        Some(&json!(true))
    );
    assert_eq!(initialized_options.lock().len(), 4);
}

#[cfg(feature = "jvm-tools")]
#[derive(Debug)]
struct JdtlsProjectEvent;

#[cfg(feature = "jvm-tools")]
impl lsp::notification::Notification for JdtlsProjectEvent {
    type Params = serde_json::Value;
    const METHOD: &'static str = "language/eventNotification";
}

#[cfg(feature = "jvm-tools")]
#[test]
fn jdtls_compatibility_levels_keep_only_bounded_provider_numbers() {
    for (raw_level, expected_level, expected_partial) in [
        ("17", Some(17), false),
        ("1.8", Some(8), false),
        ("/private/host/jdk", None, true),
        ("100", None, true),
    ] {
        let settings = serde_json::from_value(json!({
            "org.eclipse.jdt.ls.core.sourcePaths": [],
            "org.eclipse.jdt.ls.core.classpathEntries": [],
            "org.eclipse.jdt.core.compiler.source": raw_level,
            "org.eclipse.jdt.core.compiler.codegen.targetPlatform": raw_level
        }))
        .expect("JDTLS project settings");
        let (project, partial) = project::jvm_project::project_from_settings(
            "file:///fixture/application",
            settings,
            Path::new("/fixture"),
            worktree::WorktreeId::from_usize(1),
            path::PathStyle::Unix,
        )
        .expect("bounded JDTLS project");
        assert_eq!(project.java_source_level, expected_level);
        assert_eq!(project.java_target_level, expected_level);
        assert_eq!(partial, expected_partial);
        assert!(
            !format!("{:?}", project::jvm_project::project_to_proto(&project))
                .contains("/private/host/jdk")
        );
    }
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jdtls_project_commands_populate_current_worktree_paths(cx: &mut TestAppContext) {
    use collections::HashSet;
    use language::{Language, LanguageConfig, LanguageMatcher};
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

    init_test(cx);
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let launch_jdk = tempfile::tempdir().expect("launch JDK fixture");
    std::fs::write(
        launch_jdk.path().join("release"),
        "JAVA_VERSION=\"25.0.1\"\n",
    )
    .expect("launch JDK release metadata");
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/jdtls-fixture"),
        json!({
            ".zed": { "settings.json": json!({
                "lsp": { "jdtls": { "binary": {
                    "path": launch_jdk.path().join("java-lsp-proxy"),
                    "arguments": [
                        launch_jdk.path(),
                        launch_jdk.path().join("bin/java"),
                        "-Declipse.application=org.eclipse.jdt.ls.core.id1"
                    ]
                } } }
            }).to_string() },
            "application": {
                "pom.xml": "<project/>",
                "src": { "main": { "java": { "Main.java": "class Main {}" } } },
                "build": { "generated": { "sources": { "java": {
                    "Generated.java": "class Generated {}"
                } } } }
            },
            "library": { "pom.xml": "<project/>" }
        }),
    )
    .await;
    let project =
        Project::test_with_worktree_trust(fs.clone(), [path!("/jdtls-fixture").as_ref()], cx).await;
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    let trusted_worktrees =
        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust global"));
    trusted_worktrees.update(cx, |trusted, cx| {
        trusted.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        );
    });
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(Arc::new(Language::new(
        LanguageConfig {
            name: "Java".into(),
            matcher: LanguageMatcher {
                path_suffixes: vec!["java".into()],
                ..Default::default()
            }
            .into(),
            ..Default::default()
        },
        None,
    )));
    let generated_sources_ready = Arc::new(Mutex::new(false));
    let library_project_ready = Arc::new(Mutex::new(true));
    let mut fake_servers = languages.register_fake_lsp(
        "Java",
        FakeLspAdapter {
            name: "jdtls",
            initializer: Some(Box::new({
                let generated_sources_ready = generated_sources_ready.clone();
                let library_project_ready = library_project_ready.clone();
                move |fake_server| {
                    fake_server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                        let generated_sources_ready = generated_sources_ready.clone();
                        let library_project_ready = library_project_ready.clone();
                        move |params, _| {
                            let generated_sources_ready = generated_sources_ready.clone();
                            let library_project_ready = library_project_ready.clone();
                            async move {
                                match params.command.as_str() {
                                    "java.project.getAll" => {
                                        let mut projects = vec![
                                            "file:///jdtls-fixture/application".to_string(),
                                        ];
                                        if *library_project_ready.lock() {
                                            projects.push("file:///jdtls-fixture/library".to_string());
                                        }
                                        Ok(Some(json!(projects)))
                                    }
                                    "java.project.getSettings" => {
                                        assert!(params.arguments.get(1).is_some_and(|keys| {
                                            keys.as_array().is_some_and(|keys| {
                                                ["org.eclipse.jdt.core.compiler.source", "org.eclipse.jdt.core.compiler.codegen.targetPlatform"].iter().all(|expected| keys.iter().any(|key| key.as_str() == Some(*expected)))
                                            })
                                        }));
                                        let module_uri = params
                                            .arguments
                                            .first()
                                            .and_then(|argument| argument.as_str())
                                            .expect("JDTLS settings module URI");
                                        let settings = if module_uri.ends_with("/application") {
                                            let mut source_paths = vec![
                                                "/jdtls-fixture/application/src/main/java",
                                                "/private/outside",
                                            ];
                                            if *generated_sources_ready.lock() {
                                                source_paths.push(
                                                    "/jdtls-fixture/application/build/generated/sources/java",
                                                );
                                            }
                                            json!({
                                                "org.eclipse.jdt.ls.core.sourcePaths": source_paths,
                                                "org.eclipse.jdt.ls.core.classpathEntries": [
                                                    {"kind": 2, "path": "/library"},
                                                    {"kind": 1, "path": "/private/cache/hidden.jar"}
                                                ],
                                                "org.eclipse.jdt.ls.core.vm.location": "/private/jdk",
                                                "org.eclipse.jdt.core.compiler.source": "17",
        "org.eclipse.jdt.core.compiler.codegen.targetPlatform": "17"
                                            })
                                        } else {
                                            json!({
                                                "org.eclipse.jdt.ls.core.sourcePaths": [],
                                                "org.eclipse.jdt.ls.core.classpathEntries": []
                                            })
                                        };
                                        Ok(Some(settings))
                                    }
                                    _ => anyhow::bail!("unexpected JDTLS project command"),
                                }
                            }
                        }
                    });
                }
            })),
            ..FakeLspAdapter::default()
        },
    );
    project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(
                path!("/jdtls-fixture/application/src/main/java/Main.java"),
                cx,
            )
        })
        .await
        .expect("open Java source");
    let fake_server = fake_servers
        .next()
        .await
        .expect("JDTLS-named provider launched");
    cx.executor().advance_clock(Duration::from_millis(300));
    cx.run_until_parked();

    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let initial_source_count = lsp_store.read_with(cx, |store, _| {
        store
            .jdtls_project_snapshot(worktree_id)
            .and_then(|(snapshot, _)| {
                snapshot
                    .projects
                    .iter()
                    .find(|project| project.module_root.path.as_unix_str() == "application")
                    .map(|project| project.source_roots.len())
            })
    });
    assert_eq!(initial_source_count, Some(1));
    *generated_sources_ready.lock() = true;
    fake_server.notify::<JdtlsProjectEvent>(json!({
        "eventType": 200,
        "data": ["file:///jdtls-fixture/application"]
    }));
    cx.executor().advance_clock(Duration::from_millis(300));
    cx.run_until_parked();

    let (snapshot, current) = lsp_store
        .read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .map(|(snapshot, current)| (snapshot.clone(), current))
        })
        .expect("JDTLS project snapshot");
    assert!(current);
    assert!(snapshot.partial);
    assert_eq!(snapshot.projects.len(), 2);
    let application = snapshot
        .projects
        .iter()
        .find(|project| project.module_root.path.as_unix_str() == "application")
        .expect("application module");
    assert_eq!(application.source_roots.len(), 2);
    assert_eq!(application.project_references, ["library"]);
    assert_eq!(application.library_entries, 1);
    assert_eq!(application.library_files[0].name, "hidden.jar");
    assert!(application.project_vm_known);
    assert_eq!(application.java_source_level, Some(17));
    assert_eq!(application.server_launch_java_major, Some(25));
    assert_eq!(application.project_vm_java_major, None);
    assert!(
        application
            .source_roots
            .iter()
            .all(|source| source.path.worktree_id == worktree_id)
    );
    let worktree_snapshot = worktree_store.read_with(cx, |store, cx| {
        store
            .worktree_for_id(worktree_id, cx)
            .expect("fixture worktree")
            .read(cx)
            .snapshot()
    });
    let shared_settings = serde_json::from_value(json!({
        "org.eclipse.jdt.ls.core.sourcePaths": [
            "/jdtls-fixture/application/src/main/java",
            "/jdtls-fixture/application/build/generated/sources/java",
            "/jdtls-fixture/application/not-yet-visible"
        ],
        "org.eclipse.jdt.ls.core.classpathEntries": [
            {"kind": 2, "path": "/library"},
            {"kind": 2, "path": "/privateModule"},
            {
                "kind": 1,
                "path": "/private/cache/customer.jar",
                "attributes": {
                    "maven.groupId": "org.example",
                    "maven.artifactId": "customer",
                    "maven.version": "1.2.3",
                    "maven.scope": "compile"
                }
            }
        ],
        "org.eclipse.jdt.core.compiler.source": "17",
        "org.eclipse.jdt.core.compiler.codegen.targetPlatform": "17"
    }))
    .expect("host JDTLS settings");
    let (shared_projects, shared_partial) = project::jvm_project::projects_from_provider(
        vec![
            (
                "file:///jdtls-fixture/application".to_string(),
                shared_settings,
            ),
            (
                "file:///jdtls-fixture/library".to_string(),
                serde_json::from_value(json!({
                    "org.eclipse.jdt.ls.core.sourcePaths": [],
                    "org.eclipse.jdt.ls.core.classpathEntries": []
                }))
                .expect("library settings"),
            ),
        ],
        Path::new(path!("/jdtls-fixture")),
        worktree_id,
        path::PathStyle::Unix,
        &worktree_snapshot,
        true,
    );
    assert!(shared_partial);
    let shared_application = shared_projects
        .iter()
        .find(|project| project.module_root.path.as_unix_str() == "application")
        .expect("shared application");
    assert_eq!(shared_application.source_roots.len(), 2);
    assert_eq!(shared_application.project_references, ["library"]);
    assert!(shared_application.library_files.is_empty());
    assert_eq!(shared_application.library_artifacts.len(), 1);
    assert_eq!(shared_application.java_source_level, Some(17));
    assert_eq!(shared_application.java_target_level, Some(17));
    assert_eq!(
        shared_application.library_artifacts[0].coordinate,
        "org.example:customer:1.2.3"
    );
    assert_eq!(
        shared_application.library_artifacts[0].scope.as_deref(),
        Some("compile")
    );
    let wire_projects = shared_projects
        .iter()
        .map(project::jvm_project::project_to_proto)
        .collect::<Vec<_>>();
    assert!(!format!("{wire_projects:?}").contains("/private"));
    assert_eq!(
        project::jvm_project::projects_from_proto(
            wire_projects.clone(),
            worktree_id,
            &worktree_snapshot
        )
        .expect("permission-filtered remote response"),
        shared_projects
    );
    let mut invalid_level = wire_projects.clone();
    invalid_level
        .first_mut()
        .expect("shared application")
        .java_source_level = 100;
    assert!(
        project::jvm_project::projects_from_proto(invalid_level, worktree_id, &worktree_snapshot)
            .is_err()
    );
    let mut invalid_target = wire_projects.clone();
    invalid_target
        .first_mut()
        .expect("shared application")
        .java_target_level = 100;
    assert!(
        project::jvm_project::projects_from_proto(invalid_target, worktree_id, &worktree_snapshot)
            .is_err()
    );
    let mut invalid_launch = wire_projects.clone();
    invalid_launch
        .first_mut()
        .expect("shared application")
        .server_launch_java_major = 100;
    assert!(
        project::jvm_project::projects_from_proto(invalid_launch, worktree_id, &worktree_snapshot)
            .is_err()
    );
    let mut older_host = wire_projects.clone();
    older_host
        .first_mut()
        .expect("shared application")
        .java_target_level = 0;
    let older_projects =
        project::jvm_project::projects_from_proto(older_host, worktree_id, &worktree_snapshot)
            .expect("older host target level remains unknown");
    assert_eq!(older_projects[0].java_target_level, None);
    assert_eq!(older_projects[0].java_source_level, Some(17));
    assert_eq!(older_projects[0].server_launch_java_major, None);
    let mut provider_jdk = wire_projects.clone();
    let application = provider_jdk.first_mut().expect("shared application");
    application.project_vm_known = true;
    application.project_vm_java_major = 17;
    application.server_launch_java_major = 25;
    let decoded = project::jvm_project::projects_from_proto(
        provider_jdk.clone(),
        worktree_id,
        &worktree_snapshot,
    )
    .expect("host-reduced project VM Java version");
    assert_eq!(decoded[0].project_vm_java_major, Some(17));
    assert_eq!(decoded[0].server_launch_java_major, Some(25));
    let mut older_launch = provider_jdk.clone();
    older_launch
        .first_mut()
        .expect("shared application")
        .server_launch_java_major = 0;
    let decoded =
        project::jvm_project::projects_from_proto(older_launch, worktree_id, &worktree_snapshot)
            .expect("older host launch Java remains unknown");
    assert_eq!(decoded[0].server_launch_java_major, None);
    assert_eq!(decoded[0].project_vm_java_major, Some(17));
    provider_jdk
        .first_mut()
        .expect("shared application")
        .project_vm_java_major = 100;
    assert!(
        project::jvm_project::projects_from_proto(provider_jdk, worktree_id, &worktree_snapshot)
            .is_err()
    );
    let mut invalid_library = wire_projects.clone();
    invalid_library
        .first_mut()
        .expect("shared application")
        .library_file_names
        .push("/private/cache/hidden.jar".to_string());
    assert!(
        project::jvm_project::projects_from_proto(invalid_library, worktree_id, &worktree_snapshot)
            .is_err()
    );
    let mut invalid_artifact = wire_projects.clone();
    invalid_artifact
        .first_mut()
        .expect("shared application")
        .library_artifacts
        .first_mut()
        .expect("Maven artifact")
        .coordinate = "../private:customer:1.2.3".to_string();
    assert!(
        project::jvm_project::projects_from_proto(
            invalid_artifact,
            worktree_id,
            &worktree_snapshot
        )
        .is_err()
    );
    let mut invalid_gradle_usage = wire_projects.clone();
    let application = invalid_gradle_usage
        .first_mut()
        .expect("shared application");
    application.library_entries += 1;
    application.library_file_names.push("safe.jar".to_string());
    application
        .library_source_sets
        .push(client::proto::JdtlsLibrarySourceSetInfo {
            file_index: 0,
            source_sets: vec!["test\n/private".to_string()],
        });
    assert!(
        project::jvm_project::projects_from_proto(
            invalid_gradle_usage,
            worktree_id,
            &worktree_snapshot
        )
        .is_err()
    );
    let mut hidden_path = wire_projects;
    hidden_path
        .first_mut()
        .expect("shared application")
        .source_roots
        .push(client::proto::JdtlsSourceInfo {
            path: "application/not-yet-visible".to_string(),
            navigable: true,
        });
    assert!(
        project::jvm_project::projects_from_proto(hidden_path, worktree_id, &worktree_snapshot)
            .is_err()
    );
    *generated_sources_ready.lock() = false;
    fake_server.notify::<JdtlsProjectEvent>(json!({
        "eventType": 100,
        "data": ["file:///jdtls-fixture/application"]
    }));
    cx.executor().advance_clock(Duration::from_millis(300));
    cx.run_until_parked();
    let source_count_after_classpath_change = lsp_store.read_with(cx, |store, _| {
        store
            .jdtls_project_snapshot(worktree_id)
            .and_then(|(snapshot, current)| current.then_some(snapshot))
            .and_then(|snapshot| {
                snapshot
                    .projects
                    .iter()
                    .find(|project| project.module_root.path.as_unix_str() == "application")
                    .map(|project| project.source_roots.len())
            })
    });
    assert_eq!(source_count_after_classpath_change, Some(1));
    *library_project_ready.lock() = false;
    fake_server.notify::<JdtlsProjectEvent>(json!({
        "eventType": 210,
        "data": ["file:///jdtls-fixture/library"]
    }));
    cx.executor().advance_clock(Duration::from_millis(300));
    cx.run_until_parked();
    assert!(lsp_store.read_with(cx, |store, _| {
        store
            .jdtls_project_snapshot(worktree_id)
            .is_some_and(|(snapshot, current)| {
                current
                    && snapshot.projects.len() == 1
                    && snapshot.projects[0].module_root.path.as_unix_str() == "application"
            })
    }));
    fs.insert_file(
        Path::new(path!("/jdtls-fixture/application/pom.xml")),
        b"<project><version>2</version></project>".to_vec(),
    )
    .await;
    cx.run_until_parked();
    assert!(lsp_store.read_with(cx, |store, _| {
        store
            .jdtls_project_snapshot(worktree_id)
            .is_some_and(|(_, current)| !current)
    }));
    trusted_worktrees.update(cx, |trusted, cx| {
        trusted.clear_trusted_paths();
        assert!(!trusted.can_trust(&worktree_store, worktree_id, cx));
    });
    cx.run_until_parked();
    assert!(lsp_store.read_with(cx, |store, _| {
        store.jdtls_project_snapshot(worktree_id).is_none()
    }));
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jdtls_query_failure_is_visible_only_for_the_current_server(cx: &mut TestAppContext) {
    use language::{Language, LanguageConfig, LanguageMatcher};

    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/jdtls-query-error"),
        json!({
            "pom.xml": "<project/>",
            "Main.java": "class Main {}"
        }),
    )
    .await;
    let project = Project::test(fs, [path!("/jdtls-query-error").as_ref()], cx).await;
    let worktree_id = project.read_with(cx, |project, cx| {
        project
            .worktree_store()
            .read(cx)
            .worktrees()
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(Arc::new(Language::new(
        LanguageConfig {
            name: "Java".into(),
            matcher: LanguageMatcher {
                path_suffixes: vec!["java".into()],
                ..Default::default()
            }
            .into(),
            ..Default::default()
        },
        None,
    )));
    let mut fake_servers = languages.register_fake_lsp(
        "Java",
        FakeLspAdapter {
            name: "jdtls",
            initializer: Some(Box::new(|fake_server| {
                fake_server.set_request_handler::<lsp::request::ExecuteCommand, _, _>(
                    |_, _| async { anyhow::bail!("JDTLS project commands unavailable") },
                );
            })),
            ..FakeLspAdapter::default()
        },
    );
    project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/jdtls-query-error/Main.java"), cx)
        })
        .await
        .expect("open Java source");
    let _fake_server = fake_servers.next().await.expect("JDTLS-named server");
    cx.executor().advance_clock(Duration::from_millis(300));
    cx.run_until_parked();
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    assert!(lsp_store.read_with(cx, |store, _| {
        store.jdtls_project_query_failed(worktree_id)
            && store.jdtls_project_snapshot(worktree_id).is_none()
    }));
    project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
    cx.run_until_parked();
    assert!(!lsp_store.read_with(cx, |store, _| {
        store.jdtls_project_query_failed(worktree_id)
    }));
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jdtls_held_import_reply_cannot_replace_a_newer_generation(cx: &mut TestAppContext) {
    use futures::channel::oneshot;
    use language::{Language, LanguageConfig, LanguageMatcher};

    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/jdtls-generation"),
        json!({
            "application": {
                "pom.xml": "<project/>",
                "src": { "main": { "java": { "Main.java": "class Main {}" } } },
                "build": { "generated": { "sources": { "java": {
                    "Generated.java": "class Generated {}"
                } } } }
            }
        }),
    )
    .await;
    let project = Project::test(fs, [path!("/jdtls-generation").as_ref()], cx).await;
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(Arc::new(Language::new(
        LanguageConfig {
            name: "Java".into(),
            matcher: LanguageMatcher {
                path_suffixes: vec!["java".into()],
                ..Default::default()
            }
            .into(),
            ..Default::default()
        },
        None,
    )));
    let (started_sender, started_receiver) = oneshot::channel();
    let (release_sender, release_receiver) = oneshot::channel();
    let first_reply = Arc::new(Mutex::new(Some((started_sender, release_receiver))));
    let (replacement_started_sender, replacement_started_receiver) = oneshot::channel();
    let (replacement_release_sender, replacement_release_receiver) = oneshot::channel();
    let replacement_reply: Arc<Mutex<Option<(oneshot::Sender<()>, oneshot::Receiver<()>)>>> =
        Arc::new(Mutex::new(None));
    let replacement_reply_for_server = replacement_reply.clone();
    let mut fake_servers = languages.register_fake_lsp(
        "Java",
        FakeLspAdapter {
            name: "jdtls",
            capabilities: lsp::ServerCapabilities {
                execute_command_provider: Some(lsp::ExecuteCommandOptions {
                    commands: vec![
                        "java.project.getAll".to_string(),
                        "java.project.getSettings".to_string(),
                    ],
                    ..Default::default()
                }),
                ..Default::default()
            },
            initializer: Some(Box::new(move |fake_server| {
                fake_server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                    let first_reply = first_reply.clone();
                    let replacement_reply = replacement_reply_for_server.clone();
                    move |params, _| {
                        let first_reply = first_reply.clone();
                        let replacement_reply = replacement_reply.clone();
                        async move {
                            match params.command.as_str() {
                                "java.project.getAll" => {
                                    Ok(Some(json!(["file:///jdtls-generation/application"])))
                                }
                                "java.project.getSettings" => {
                                    let held_reply = {
                                        let mut first_reply = first_reply.lock();
                                        first_reply.take()
                                    };
                                    if let Some((started_sender, release_receiver)) = held_reply {
                                        started_sender.send(()).map_err(|_| {
                                            anyhow::anyhow!("start receiver closed")
                                        })?;
                                        release_receiver.await.map_err(|_| {
                                            anyhow::anyhow!("release sender closed")
                                        })?;
                                        Ok(Some(json!({
                                            "org.eclipse.jdt.ls.core.sourcePaths": [],
                                            "org.eclipse.jdt.ls.core.classpathEntries": []
                                        })))
                                    } else {
                                        let held_replacement = replacement_reply.lock().take();
                                        if let Some((started_sender, release_receiver)) = held_replacement {
                                            started_sender.send(()).map_err(|_| {
                                                anyhow::anyhow!("replacement start receiver closed")
                                            })?;
                                            release_receiver.await.map_err(|_| {
                                                anyhow::anyhow!("replacement release sender closed")
                                            })?;
                                        }
                                        Ok(Some(json!({
                                            "org.eclipse.jdt.ls.core.sourcePaths": [
                                                "/jdtls-generation/application/build/generated/sources/java"
                                            ],
                                            "org.eclipse.jdt.ls.core.classpathEntries": []
                                        })))
                                    }
                                }
                                _ => anyhow::bail!("unexpected JDTLS project command"),
                            }
                        }
                    }
                });
            })),
            ..FakeLspAdapter::default()
        },
    );
    project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(
                path!("/jdtls-generation/application/src/main/java/Main.java"),
                cx,
            )
        })
        .await
        .expect("open Java source");
    let fake_server = fake_servers.next().await.expect("JDTLS-named provider");
    cx.executor().advance_clock(Duration::from_millis(300));
    started_receiver
        .await
        .expect("first settings query started");

    fake_server
        .start_progress_with(
            "jdtls-import",
            lsp::WorkDoneProgressBegin {
                title: "Importing Maven project(s)".to_string(),
                ..Default::default()
            },
            lsp::DEFAULT_LSP_REQUEST_TIMEOUT,
        )
        .await;
    fake_server.end_progress("jdtls-import");
    cx.run_until_parked();
    cx.executor().advance_clock(Duration::from_millis(300));
    cx.run_until_parked();

    let worktree_id = project.read_with(cx, |project, cx| {
        project
            .worktree_store()
            .read(cx)
            .worktrees()
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let snapshot = lsp_store
        .read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .map(|(snapshot, current)| (snapshot.clone(), current))
        })
        .expect("newer JDTLS snapshot");
    assert!(snapshot.1);
    assert_eq!(
        snapshot
            .0
            .projects
            .first()
            .map(|project| project.source_roots.len()),
        Some(1)
    );
    assert_eq!(
        snapshot
            .0
            .projects
            .first()
            .and_then(|project| project.source_roots.first())
            .map(|source| source.path.path.as_unix_str()),
        Some("application/build/generated/sources/java")
    );
    if release_sender.send(()).is_err() {
        assert!(
            snapshot.1,
            "cancelled stale request must leave current data"
        );
    }
    cx.run_until_parked();
    let after_late_reply = lsp_store
        .read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .map(|(snapshot, current)| (snapshot.clone(), current))
        })
        .expect("current snapshot after stale reply");
    assert_eq!(after_late_reply, snapshot);

    *replacement_reply.lock() = Some((replacement_started_sender, replacement_release_receiver));
    fake_server
        .start_progress_with(
            "jdtls-replacement",
            lsp::WorkDoneProgressBegin {
                title: "Importing Maven project(s)".to_string(),
                ..Default::default()
            },
            lsp::DEFAULT_LSP_REQUEST_TIMEOUT,
        )
        .await;
    fake_server.end_progress("jdtls-replacement");
    cx.run_until_parked();
    cx.executor().advance_clock(Duration::from_millis(300));
    replacement_started_receiver
        .await
        .expect("old-worktree settings query started");
    project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
    cx.run_until_parked();
    assert!(lsp_store.read_with(cx, |store, _| {
        store.jdtls_project_snapshot(worktree_id).is_none()
    }));
    let replacement = project
        .update(cx, |project, cx| {
            project.create_worktree(Path::new(path!("/jdtls-generation")), true, cx)
        })
        .await
        .expect("replacement fixture worktree");
    let replacement_id = replacement.read_with(cx, |worktree, _| worktree.id());
    assert_ne!(replacement_id, worktree_id);
    if replacement_release_sender.send(()).is_err() {
        assert!(lsp_store.read_with(cx, |store, _| {
            store.jdtls_project_snapshot(worktree_id).is_none()
        }));
    }
    cx.run_until_parked();
    assert!(lsp_store.read_with(cx, |store, _| {
        store.jdtls_project_snapshot(worktree_id).is_none()
            && store.jdtls_project_snapshot(replacement_id).is_none()
    }));
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn kotlin_workspace_export_projects_only_visible_maven_facts(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/kotlin-export-fixture"),
        json!({
            "application": {
                "pom.xml": "<project/>",
                "src": { "main": { "java": { "Main.java": "class Main {}" } } }
            },
            "library": {
                "pom.xml": "<project/>",
                "src": { "main": { "kotlin": { "Greeting.kt": "class Greeting" } } }
            }
        }),
    )
    .await;
    let project =
        Project::test_with_worktree_trust(fs, [path!("/kotlin-export-fixture").as_ref()], cx).await;
    let (worktree_id, snapshot) = project.read_with(cx, |project, cx| {
        let worktree = project
            .worktree_store()
            .read(cx)
            .worktrees()
            .next()
            .expect("fixture worktree")
            .read(cx);
        (worktree.id(), worktree.snapshot())
    });
    let export = json!({
        "externalSystem": "MAVEN",
        "libraries": [
            {"name": "Maven: org.junit.jupiter:junit-jupiter:5.11.4"},
            {"name": "Maven: /private/host/path"}
        ],
        "modules": [
            {
                "name": "application",
                "externalProjectPath": "/kotlin-export-fixture/application",
                "contentRoots": [{"sourceRoots": [
                    {"path": "/kotlin-export-fixture/application/src/main/java", "generated": true},
                    {"path": "/private/host/source"}
                ]}],
                "dependencies": [
                    {"type": "module", "name": "library", "scope": "compile"},
                    {"type": "library", "name": "Maven: org.junit.jupiter:junit-jupiter:5.11.4", "scope": "test"},
                    {"type": "library", "name": "Maven: /private/host/path", "scope": "compile"},
                    {"type": "module", "name": "outside", "scope": "compile"}
                ]
            },
            {
                "name": "library",
                "externalProjectPath": "/kotlin-export-fixture/library",
                "contentRoots": [{"sourceRoots": [
                    {"path": "/kotlin-export-fixture/library/src/main/kotlin"}
                ]}],
                "dependencies": []
            },
            {
                "name": "outside",
                "externalProjectPath": "/private/host/outside",
                "dependencies": []
            }
        ]
    });
    let bytes = serde_json::to_vec(&export).expect("workspace export fixture");
    let (projects, partial) = project::jvm_project::projects_from_kotlin_workspace(
        &bytes,
        Path::new("/kotlin-export-fixture"),
        worktree_id,
        path::PathStyle::Unix,
        &snapshot,
        true,
    )
    .expect("bounded Kotlin workspace export");
    assert!(partial);
    assert_eq!(projects.len(), 2);
    let application = projects
        .iter()
        .find(|module| module.name == "application")
        .expect("application module");
    assert_eq!(
        application.build_system,
        project::jvm_project::KotlinBuildSystem::Maven
    );
    assert_eq!(application.module_root.path.as_unix_str(), "application");
    assert_eq!(application.source_roots.len(), 1);
    assert_eq!(
        application.source_roots[0].path.path.as_unix_str(),
        "application/src/main/java"
    );
    assert!(application.source_roots[0].generated);
    assert_eq!(application.unidentified_libraries, 1);
    assert_eq!(
        application.dependencies,
        [
            project::jvm_project::KotlinWorkspaceDependency::Module {
                name: "library".to_string(),
                scope: Some("compile".to_string()),
            },
            project::jvm_project::KotlinWorkspaceDependency::Artifact {
                coordinate: "org.junit.jupiter:junit-jupiter:5.11.4".to_string(),
                scope: Some("test".to_string()),
            },
        ]
    );
    let wire_projects = projects
        .iter()
        .map(project::jvm_project::kotlin_project_to_proto)
        .collect::<Vec<_>>();
    assert!(
        wire_projects
            .iter()
            .all(|project| project.build_system == "MAVEN")
    );
    let decoded = project::jvm_project::kotlin_projects_from_proto(
        wire_projects.clone(),
        worktree_id,
        &snapshot,
    )
    .expect("permission-filtered Kotlin wire snapshot");
    assert_eq!(decoded, projects);
    let mut older_wire_projects = wire_projects.clone();
    older_wire_projects[0].build_system.clear();
    let older_decoded = project::jvm_project::kotlin_projects_from_proto(
        older_wire_projects,
        worktree_id,
        &snapshot,
    )
    .expect("older host has no build-system field");
    assert_eq!(
        older_decoded[0].build_system,
        project::jvm_project::KotlinBuildSystem::Unknown
    );
    let mut invalid_build_system = wire_projects.clone();
    invalid_build_system[0].build_system = "CARGO".to_string();
    assert!(
        project::jvm_project::kotlin_projects_from_proto(
            invalid_build_system,
            worktree_id,
            &snapshot,
        )
        .is_err()
    );
    let mut path_leak = wire_projects.clone();
    path_leak[0].module_root = "/private/host/path".to_string();
    assert!(
        project::jvm_project::kotlin_projects_from_proto(path_leak, worktree_id, &snapshot)
            .is_err()
    );
    let mut coordinate_leak = wire_projects;
    coordinate_leak[0]
        .dependencies
        .push(client::proto::KotlinWorkspaceDependencyInfo {
            is_module: false,
            name: "/private/host/library.jar".to_string(),
            scope: Some("compile".to_string()),
        });
    assert!(
        project::jvm_project::kotlin_projects_from_proto(coordinate_leak, worktree_id, &snapshot,)
            .is_err()
    );
    let no_build =
        serde_json::to_vec(&json!({"externalSystem": "JSON"})).expect("unsupported build fixture");
    assert!(
        project::jvm_project::projects_from_kotlin_workspace(
            &no_build,
            Path::new("/kotlin-export-fixture"),
            worktree_id,
            path::PathStyle::Unix,
            &snapshot,
            true,
        )
        .is_err()
    );
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn kotlin_named_fake_server_waits_for_worktree_trust_before_launch(cx: &mut TestAppContext) {
    use collections::HashSet;
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

    init_test(cx);
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(path!("/jvm-fixture"), json!({ "main.rs": "fn main() {}" }))
        .await;
    let project = Project::test_with_worktree_trust(fs, [path!("/jvm-fixture").as_ref()], cx).await;
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(rust_lang());
    let mut fake_servers = languages.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "kotlin-lsp",
            ..FakeLspAdapter::default()
        },
    );
    let (_buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/jvm-fixture/main.rs"), cx)
        })
        .await
        .expect("open fixture source");
    cx.run_until_parked();
    assert!(
        fake_servers.next().now_or_never().is_none(),
        "Kotlin-named fake server started before the worktree was trusted"
    );

    let trusted_worktrees =
        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust global"));
    trusted_worktrees.update(cx, |store, cx| {
        store.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        );
    });
    fake_servers
        .next()
        .await
        .expect("Kotlin-named fake server starts after trust");
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jvm_trust_restriction_stops_only_affected_language_servers(cx: &mut TestAppContext) {
    use collections::HashSet;
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

    init_test(cx);
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let fs = FakeFs::new(cx.executor());
    for root in [path!("/restricted-jvm"), path!("/other-jvm")] {
        fs.insert_tree(root, json!({ "main.rs": "fn main() {}" }))
            .await;
    }
    let project =
        Project::test_with_worktree_trust(fs.clone(), [path!("/restricted-jvm").as_ref()], cx)
            .await;
    let other_project =
        Project::test_with_worktree_trust(fs, [path!("/other-jvm").as_ref()], cx).await;
    let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("shared trust owner"));
    for project in [&project, &other_project] {
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktree_store.read_with(cx, |store, cx| {
            store
                .worktrees()
                .next()
                .expect("fixture root")
                .read(cx)
                .id()
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
    }
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(rust_lang());
    let mut server_streams = Vec::new();
    for name in [
        "jdtls",
        "kotlin-lsp",
        "kotlin-language-server",
        "gradle-language-server",
        "rust-analyzer",
    ] {
        server_streams.push(languages.register_fake_lsp(
            "Rust",
            FakeLspAdapter {
                name,
                ..FakeLspAdapter::default()
            },
        ));
    }
    let other_languages = other_project.read_with(cx, |project, _| project.languages().clone());
    other_languages.add(rust_lang());
    let mut other_servers = other_languages.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "kotlin-lsp",
            ..FakeLspAdapter::default()
        },
    );
    let (_buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/restricted-jvm/main.rs"), cx)
        })
        .await
        .expect("open restricted fixture");
    let (_other_buffer, _other_handle) = other_project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/other-jvm/main.rs"), cx)
        })
        .await
        .expect("open unrelated fixture");
    let mut servers = Vec::new();
    for stream in &mut server_streams {
        servers.push(stream.next().await.expect("running fixture server"));
    }
    let other_server = other_servers
        .next()
        .await
        .expect("unrelated running server");
    cx.run_until_parked();
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let other_lsp_store = other_project.read_with(cx, |project, _| project.lsp_store());
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("fixture root")
            .read(cx)
            .id()
    });
    trust.update(cx, |trust, cx| {
        trust.restrict(
            worktree_store.downgrade(),
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        );
    });
    cx.run_until_parked();
    for server in servers.iter().take(4) {
        assert!(
            lsp_store.read_with(cx, |store, _| store
                .language_server_for_id(server.server.server_id())
                .is_none()),
            "JVM server survived project trust restriction"
        );
    }
    let copper_server = servers.last().expect("Copper server");
    assert!(lsp_store.read_with(cx, |store, _| {
        store
            .language_server_for_id(copper_server.server.server_id())
            .is_some()
    }));
    assert!(other_lsp_store.read_with(cx, |store, _| {
        store
            .language_server_for_id(other_server.server.server_id())
            .is_some()
    }));
    for stream in server_streams.iter_mut().take(4) {
        assert!(
            stream.next().now_or_never().is_none(),
            "replacement started before renewed trust"
        );
    }
    trust.update(cx, |trust, cx| {
        trust.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        );
    });
    for (stream, old_server) in server_streams.iter_mut().take(4).zip(servers.iter()) {
        let replacement = stream
            .next()
            .await
            .expect("fresh server after renewed trust");
        assert_ne!(
            replacement.server.server_id(),
            old_server.server.server_id()
        );
        cx.run_until_parked();
        assert!(lsp_store.read_with(cx, |store, _| {
            store
                .language_server_for_id(replacement.server.server_id())
                .is_some()
        }));
        assert!(lsp_store.read_with(cx, |store, _| {
            store
                .language_server_for_id(old_server.server.server_id())
                .is_none()
        }));
    }
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jvm_trust_restriction_cancels_pending_initialization(cx: &mut TestAppContext) {
    use collections::HashSet;
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

    init_test(cx);
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(path!("/pending-jvm"), json!({ "main.rs": "fn main() {}" }))
        .await;
    let project = Project::test_with_worktree_trust(fs, [path!("/pending-jvm").as_ref()], cx).await;
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("fixture root")
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
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(rust_lang());
    let (release, reply) = futures::channel::oneshot::channel();
    let pending_reply = Arc::new(Mutex::new(Some(reply)));
    let mut fake_servers = languages.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "kotlin-lsp",
            initializer: Some(Box::new(move |server| {
                let pending_reply = pending_reply.clone();
                server.set_request_handler::<lsp::request::Initialize, _, _>(move |_, _| {
                    let pending_reply = pending_reply.lock().take();
                    async move {
                        if let Some(reply) = pending_reply {
                            reply.await.expect("release old initialization");
                        }
                        Ok(lsp::InitializeResult::default())
                    }
                });
            })),
            ..FakeLspAdapter::default()
        },
    );
    let (_buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/pending-jvm/main.rs"), cx)
        })
        .await
        .expect("open pending fixture");
    let old_server = fake_servers.next().await.expect("pending server");
    cx.run_until_parked();
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    trust.update(cx, |trust, cx| {
        trust.restrict(
            worktree_store.downgrade(),
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        )
    });
    cx.run_until_parked();
    assert!(
        release.send(()).is_err(),
        "restricted startup kept its initialization handler alive"
    );
    cx.run_until_parked();
    assert!(
        lsp_store.read_with(cx, |store, _| store
            .language_server_for_id(old_server.server.server_id())
            .is_none()),
        "late initialization revived restricted server"
    );
    assert!(fake_servers.next().now_or_never().is_none());
    trust.update(cx, |trust, cx| {
        trust.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        )
    });
    let replacement = fake_servers.next().await.expect("fresh trusted generation");
    cx.run_until_parked();
    assert_ne!(
        replacement.server.server_id(),
        old_server.server.server_id()
    );
    assert!(lsp_store.read_with(cx, |store, _| {
        store
            .language_server_for_id(replacement.server.server_id())
            .is_some()
    }));
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jvm_trust_restriction_preserves_another_worktree_in_the_same_project(
    cx: &mut TestAppContext,
) {
    use collections::HashSet;
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

    init_test(cx);
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let fs = FakeFs::new(cx.executor());
    for root in [path!("/first-jvm"), path!("/second-jvm")] {
        fs.insert_tree(root, json!({ "main.rs": "fn main() {}" }))
            .await;
    }
    let project = Project::test_with_worktree_trust(
        fs,
        [path!("/first-jvm").as_ref(), path!("/second-jvm").as_ref()],
        cx,
    )
    .await;
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_ids = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .map(|worktree| worktree.read(cx).id())
            .collect::<Vec<_>>()
    });
    let first_id = *worktree_ids.first().expect("first root");
    let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("shared trust owner"));
    trust.update(cx, |trust, cx| {
        trust.trust(
            &worktree_store,
            worktree_ids.into_iter().map(PathTrust::Worktree).collect(),
            cx,
        )
    });
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(rust_lang());
    let mut fake_servers = languages.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "kotlin-lsp",
            ..FakeLspAdapter::default()
        },
    );
    let (_first_buffer, _first_handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/first-jvm/main.rs"), cx)
        })
        .await
        .expect("first source");
    let first_server = fake_servers.next().await.expect("first server");
    let (_second_buffer, _second_handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/second-jvm/main.rs"), cx)
        })
        .await
        .expect("second source");
    let second_server = fake_servers.next().await.expect("second server");
    cx.run_until_parked();
    trust.update(cx, |trust, cx| {
        trust.restrict(
            worktree_store.downgrade(),
            HashSet::from_iter([PathTrust::Worktree(first_id)]),
            cx,
        )
    });
    cx.run_until_parked();
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    assert!(lsp_store.read_with(cx, |store, _| {
        store
            .language_server_for_id(first_server.server.server_id())
            .is_none()
    }));
    assert!(lsp_store.read_with(cx, |store, _| {
        store
            .language_server_for_id(second_server.server.server_id())
            .is_some()
    }));
    assert!(fake_servers.next().now_or_never().is_none());
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn kotlin_project_override_launches_only_the_selected_server(cx: &mut TestAppContext) {
    use language::{Language, LanguageConfig, LanguageMatcher};
    use settings::{LocalSettingsKind, LocalSettingsPath, SettingsStore};

    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/kotlin-fixture"),
        json!({ "Main.kt": "fun main() {}" }),
    )
    .await;
    let project = Project::test(fs, [path!("/kotlin-fixture").as_ref()], cx).await;
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    cx.update(|cx| {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store
                .set_user_settings(
                    r#"{"languages":{"Kotlin":{"language_servers":["kotlin-lsp"]}}}"#,
                    cx,
                )
                .expect("preferred Kotlin server setting");
            store
                .set_local_settings(
                    worktree_id,
                    LocalSettingsPath::InWorktree(util::rel_path::rel_path("").into()),
                    LocalSettingsKind::Settings,
                    Some(
                        r#"{"languages":{"Kotlin":{"language_servers":["!kotlin-lsp","kotlin-language-server"]}}}"#,
                    ),
                    cx,
                )
                .expect("project Kotlin fallback setting");
        });
    });
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(Arc::new(Language::new(
        LanguageConfig {
            name: "Kotlin".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["kt".into()],
                ..LanguageMatcher::default()
            })
            .into(),
            ..LanguageConfig::default()
        },
        None,
    )));
    let mut preferred_servers = languages.register_fake_lsp(
        "Kotlin",
        FakeLspAdapter {
            name: "kotlin-lsp",
            ..FakeLspAdapter::default()
        },
    );
    let mut fallback_servers = languages.register_fake_lsp(
        "Kotlin",
        FakeLspAdapter {
            name: "kotlin-language-server",
            ..FakeLspAdapter::default()
        },
    );
    let (_buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/kotlin-fixture/Main.kt"), cx)
        })
        .await
        .expect("open Kotlin source");
    fallback_servers
        .next()
        .await
        .expect("project-selected fallback launches");
    cx.run_until_parked();
    assert!(
        preferred_servers.next().now_or_never().is_none(),
        "project override must not launch the preferred Kotlin server"
    );
    assert!(
        fallback_servers.next().now_or_never().is_none(),
        "a second fallback Kotlin server must not launch"
    );
}

#[gpui::test]
async fn stopped_copper_server_cancels_pending_initialization(cx: &mut TestAppContext) {
    use collections::HashSet;

    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/copper-late-init"),
        json!({ "main.rs": "fn main() {}" }),
    )
    .await;
    let project = Project::test(fs, [path!("/copper-late-init").as_ref()], cx).await;
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(rust_lang());
    let (release, reply) = futures::channel::oneshot::channel();
    let pending_reply = Arc::new(Mutex::new(Some(reply)));
    let mut fake_servers = languages.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "rust-analyzer",
            initializer: Some(Box::new(move |server| {
                let pending_reply = pending_reply.clone();
                server.set_request_handler::<lsp::request::Initialize, _, _>(move |_, _| {
                    let pending_reply = pending_reply.lock().take();
                    async move {
                        if let Some(reply) = pending_reply {
                            reply.await.expect("release Copper initialization");
                        }
                        Ok(lsp::InitializeResult::default())
                    }
                });
            })),
            ..FakeLspAdapter::default()
        },
    );
    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/copper-late-init/main.rs"), cx)
        })
        .await
        .expect("open Copper fixture");
    let old_server = fake_servers.next().await.expect("pending Copper server");
    cx.run_until_parked();
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let stopped = lsp_store.update(cx, |store, cx| {
        store.stop_language_servers_for_buffers(vec![buffer], HashSet::default(), cx)
    });
    release.send(()).expect("release stopped server response");
    stopped.await.expect("existing shared stop path");
    cx.run_until_parked();
    assert!(
        lsp_store.read_with(cx, |store, _| store
            .language_server_for_id(old_server.server.server_id())
            .is_none()),
        "stopped Copper server was reinserted after late initialization"
    );
}

#[cfg(all(feature = "jvm-tools", unix))]
#[gpui::test]
#[ignore = "requires GROOVY_JDTLS_LAUNCHER and GROOVY_LSP_JAVA_HOME with cached Maven dependencies"]
async fn real_jdtls_process_obeys_worktree_trust(cx: &mut TestAppContext) {
    real_jvm_process_obeys_worktree_trust("jdtls", cx).await;
}

#[cfg(all(feature = "jvm-tools", unix))]
#[gpui::test]
#[ignore = "requires GROOVY_KOTLIN_LSP and GROOVY_LSP_JAVA_HOME with cached Maven dependencies"]
async fn real_kotlin_lsp_process_obeys_worktree_trust(cx: &mut TestAppContext) {
    real_jvm_process_obeys_worktree_trust("kotlin-lsp", cx).await;
}

#[cfg(all(feature = "jvm-tools", unix))]
async fn real_jvm_process_obeys_worktree_trust(name: &'static str, cx: &mut TestAppContext) {
    use collections::HashSet;
    use language::{Language, LanguageConfig, LanguageMatcher};
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

    init_test(cx);
    cx.executor().allow_parking();
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let state = tempfile::tempdir().expect("isolated JVM state");
    let root = state.path().join("project");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/groovy_jvm_maven");
    for relative in [
        "pom.xml",
        "library/pom.xml",
        "application/pom.xml",
        "library/src/main/kotlin/evidence/Greeting.kt",
        "application/src/main/java/evidence/Main.java",
    ] {
        let destination = root.join(relative);
        std::fs::create_dir_all(destination.parent().expect("fixture parent"))
            .expect("create fixture parent");
        std::fs::copy(fixture.join(relative), destination).expect("copy Maven source fixture");
    }
    let root = root.canonicalize().expect("canonical fixture root");
    std::fs::create_dir_all(root.join(".mvn")).expect("Maven configuration directory");
    std::fs::write(root.join(".mvn/maven.config"), "-o\n").expect("cached Maven only");
    let java_home = std::env::var("GROOVY_LSP_JAVA_HOME").expect("explicit server JDK");
    assert!(Path::new(&java_home).join("bin/java").is_file());
    let (language_name, suffix, source, launcher_variable) = if name == "jdtls" {
        (
            "Java",
            "java",
            "application/src/main/java/evidence/Main.java",
            "GROOVY_JDTLS_LAUNCHER",
        )
    } else {
        (
            "Kotlin",
            "kt",
            "library/src/main/kotlin/evidence/Greeting.kt",
            "GROOVY_KOTLIN_LSP",
        )
    };
    let launcher = std::env::var(launcher_variable).expect("explicit pinned JVM server launcher");
    assert!(Path::new(&launcher).is_absolute() && Path::new(&launcher).is_file());
    if name == "jdtls" {
        assert!(
            Path::new(&launcher)
                .parent()
                .expect("JDTLS launcher directory")
                .join("jdtls.py")
                .is_file(),
            "JDTLS candidate is incomplete: missing launcher companion"
        );
    }
    let trace = state.path().join("launcher-pids");
    let mut arguments = vec![
        "-c".to_string(),
        "printf '%s\\n' \"$$\" >> \"$1\"; shift; exec \"$@\"".to_string(),
        "groovy-lsp-trace".to_string(),
        trace.to_string_lossy().into_owned(),
        launcher,
    ];
    if name == "jdtls" {
        arguments.extend([
            format!("--java-executable={java_home}/bin/java"),
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
        .map(|kind| format!("-Didea.{kind}.path={}", state.path().join(kind).display()))
        .collect::<Vec<_>>()
        .join(" ");
    let settings = json!({"lsp": {name: {"binary": {
        "path": "/bin/sh", "arguments": arguments,
        "env": {"JAVA_HOME": java_home, "JAVA_TOOL_OPTIONS": tool_options}
    }}}});
    std::fs::create_dir_all(root.join(".zed")).expect("settings directory");
    std::fs::write(root.join(".zed/settings.json"), settings.to_string())
        .expect("explicit binary override");
    cx.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store
                .set_user_settings(&settings.to_string(), cx)
                .expect("configured override before LSP admission");
        });
    });
    let project = Project::test_with_worktree_trust(
        Arc::new(fs::RealFs::new(None, cx.executor())),
        [root.as_path()],
        cx,
    )
    .await;
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(Arc::new(Language::new(
        LanguageConfig {
            name: language_name.into(),
            matcher: LanguageMatcher {
                path_suffixes: vec![suffix.to_string()],
                ..Default::default()
            }
            .into(),
            ..Default::default()
        },
        None,
    )));
    languages.register_fake_lsp_adapter(
        language_name,
        FakeLspAdapter {
            name,
            ..FakeLspAdapter::default()
        },
    );
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("fixture root")
            .read(cx)
            .id()
    });
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let (_buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(root.join(source), cx)
        })
        .await
        .expect("open real JVM source");
    cx.run_until_parked();
    assert!(!trace.exists(), "provider launcher ran before trust");
    assert!(
        lsp_store.read_with(cx, |store, _| store.as_local().is_some_and(|local| local
            .language_servers
            .values()
            .any(|state| matches!(state, LanguageServerState::Starting { .. })))),
        "configured provider is not waiting at the shared trust gate"
    );
    assert!(lsp_store.read_with(cx, |store, _| {
        store.language_server_statuses().next().is_none()
    }));
    let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("shared trust owner"));
    let mut previous_id = None;
    for generation in 1..=2 {
        let (started, response) = futures::channel::oneshot::channel();
        let mut started = Some(started);
        let subscription = lsp_store.update(cx, |_, cx| {
            cx.subscribe(&lsp_store, move |_, _, event, _| {
                if let LspStoreEvent::LanguageServerAdded(id, _, _) = event
                    && let Some(started) = started.take()
                {
                    started.send(*id).expect("server admission observer");
                }
            })
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let initialization = response.fuse();
        let timeout = cx
            .background_executor
            .timer(Duration::from_secs(180))
            .fuse();
        futures::pin_mut!(initialization, timeout);
        let id = futures::select_biased! {
            response = initialization => response.expect("real provider initialization"),
            () = timeout => panic!("{name} did not initialize within the bounded fixture timeout"),
        };
        drop(subscription);
        assert_ne!(Some(id), previous_id);
        let server = lsp_store.read_with(cx, |store, _| {
            store
                .language_server_for_id(id)
                .expect("running real provider")
        });
        let pid = server.process_id().expect("real host process ID");
        let trace_text = std::fs::read_to_string(&trace).expect("launcher trace");
        assert_eq!(trace_text.lines().count(), generation);
        assert_eq!(
            trace_text
                .lines()
                .last()
                .expect("current launcher PID")
                .parse::<u32>()
                .expect("numeric PID"),
            pid
        );
        assert!(
            util::command::new_command("kill")
                .args(["-0", &pid.to_string()])
                .status()
                .await
                .expect("probe admitted process")
                .success()
        );
        println!(
            "{name}: generation={generation}, server={id}, pid={pid}, version={:?}, admitted after trust",
            server.version()
        );
        trust.update(cx, |trust, cx| {
            trust.restrict(
                worktree_store.downgrade(),
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        while server.process_id().is_some() && Instant::now() < deadline {
            cx.run_until_parked();
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        assert!(
            server.process_id().is_none(),
            "restricted provider process did not shut down"
        );
        assert!(
            !util::command::new_command("kill")
                .args(["-0", &pid.to_string()])
                .stderr(util::command::Stdio::null())
                .status()
                .await
                .expect("probe stopped process")
                .success(),
            "restricted provider PID remains alive"
        );
        assert!(lsp_store.read_with(cx, |store, _| store.language_server_for_id(id).is_none()));
        assert_eq!(
            std::fs::read_to_string(&trace)
                .expect("retained trace")
                .lines()
                .count(),
            generation
        );
        println!("{name}: generation={generation}, pid={pid}, stopped after restriction");
        previous_id = Some(id);
    }
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn kotlin_early_import_success_refreshes_after_server_registration(cx: &mut TestAppContext) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use collections::HashSet;
    use language::{Language, LanguageConfig, LanguageMatcher};
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

    struct KotlinWorkspaceImportState;
    impl lsp::notification::Notification for KotlinWorkspaceImportState {
        type Params = serde_json::Value;
        const METHOD: &'static str = "intellij/workspaceImportState";
    }

    init_test(cx);
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/kotlin-early-import"),
        json!({
            "pom.xml": "<project/>",
            "Main.kt": "fun main() {}"
        }),
    )
    .await;
    let project =
        Project::test_with_worktree_trust(fs, [path!("/kotlin-early-import").as_ref()], cx).await;
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("Kotlin worktree")
            .read(cx)
            .id()
    });
    let trusted_worktrees =
        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust global"));
    trusted_worktrees.update(cx, |trusted, cx| {
        trusted.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        );
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
    let exports = Arc::new(AtomicUsize::new(0));
    let mut fake_servers = languages.register_fake_lsp(
        "Kotlin",
        FakeLspAdapter {
            name: "kotlin-lsp",
            capabilities: lsp::ServerCapabilities {
                execute_command_provider: Some(lsp::ExecuteCommandOptions {
                    commands: vec!["exportWorkspace".to_string()],
                    ..Default::default()
                }),
                ..Default::default()
            },
            initializer: Some(Box::new({
                let exports = exports.clone();
                move |fake_server| {
                    fake_server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                        let exports = exports.clone();
                        move |params, _| {
                            let exports = exports.clone();
                            async move {
                                anyhow::ensure!(params.command == "exportWorkspace");
                                let directory = params
                                    .arguments
                                    .first()
                                    .and_then(serde_json::Value::as_str)
                                    .ok_or_else(|| anyhow::anyhow!("missing export directory"))?;
                                std::fs::write(
                                    Path::new(directory).join("workspace.json"),
                                    serde_json::to_vec(&json!({
                                        "externalSystem": "MAVEN",
                                        "modules": [{
                                            "name": "application",
                                            "externalProjectPath": "/kotlin-early-import"
                                        }]
                                    }))?,
                                )?;
                                exports.fetch_add(1, Ordering::SeqCst);
                                Ok(None)
                            }
                        }
                    });
                }
            })),
            ..Default::default()
        },
    );
    project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/kotlin-early-import/Main.kt"), cx)
        })
        .await
        .expect("open Kotlin source");
    let fake_server = fake_servers.next().await.expect("Kotlin provider");
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    assert!(
        lsp_store.read_with(cx, |store, _| store
            .language_server_statuses()
            .next()
            .is_none()),
        "the fixture must report import before the display-status entry exists"
    );
    fake_server.notify::<KotlinWorkspaceImportState>(json!({
        "phase": "FINISHED",
        "folders": [{"tool": "maven", "status": "SUCCESS"}]
    }));
    cx.run_until_parked();
    let snapshot = lsp_store.read_with(cx, |store, _| {
        store
            .kotlin_workspace_snapshot(worktree_id)
            .map(|(snapshot, current)| (snapshot.projects.len(), current))
    });
    assert_eq!(
        snapshot,
        Some((1, true)),
        "exports: {}, pending: {}, failed: {}, servers: {:?}",
        exports.load(Ordering::SeqCst),
        lsp_store.read_with(cx, |store, _| store
            .kotlin_workspace_query_pending(worktree_id)),
        lsp_store.read_with(cx, |store, _| store
            .kotlin_workspace_query_failed(worktree_id)),
        lsp_store.read_with(cx, |store, _| format!(
            "{:?}",
            store.language_server_statuses().collect::<Vec<_>>()
        ))
    );
    assert_eq!(exports.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn jdtls_import_failure_logs_reach_shared_health_without_host_paths(cx: &mut TestAppContext) {
    use language::{Language, LanguageConfig, LanguageMatcher};
    use rpc::proto;

    struct JdtlsLog;
    impl lsp::notification::Notification for JdtlsLog {
        type Params = serde_json::Value;
        const METHOD: &'static str = "window/logMessage";
    }

    struct JdtlsStatus;
    impl lsp::notification::Notification for JdtlsStatus {
        type Params = serde_json::Value;
        const METHOD: &'static str = "language/status";
    }

    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/jdtls-offline"),
        json!({ "Main.java": "class Main {}" }),
    )
    .await;
    let project = Project::test(fs, [path!("/jdtls-offline").as_ref()], cx).await;
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(Arc::new(Language::new(
        LanguageConfig {
            name: "Java".into(),
            matcher: LanguageMatcher {
                path_suffixes: vec!["java".into()],
                ..Default::default()
            }
            .into(),
            ..Default::default()
        },
        None,
    )));
    let mut fake_servers = languages.register_fake_lsp(
        "Java",
        FakeLspAdapter {
            name: "jdtls",
            ..Default::default()
        },
    );
    let (_buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/jdtls-offline/Main.java"), cx)
        })
        .await
        .expect("open Java source");
    let fake_server = fake_servers.next().await.expect("JDTLS provider");
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let health_events = Arc::new(Mutex::new(Vec::<(i32, String)>::new()));
    let server_logs = Arc::new(Mutex::new(Vec::<String>::new()));
    cx.update({
        let health_events = health_events.clone();
        let server_logs = server_logs.clone();
        |cx| {
            cx.subscribe(&lsp_store, move |_, event, _| match event {
                LspStoreEvent::LanguageServerUpdate {
                    message: proto::update_language_server::Variant::StatusUpdate(update),
                    ..
                } => {
                    if let Some(proto::status_update::Status::Health(health)) = &update.status {
                        health_events
                            .lock()
                            .push((*health, update.message.clone().unwrap_or_default()));
                    }
                }
                LspStoreEvent::LanguageServerLog(_, _, message) => {
                    server_logs.lock().push(message.clone());
                }
                _ => {}
            })
            .detach();
        }
    });

    fake_server.notify::<JdtlsLog>(json!({
        "type": 1,
        "message": "Could not calculate build plan at /private/host/project: Cannot access central in offline mode and the artifact org.apache.maven.plugins:maven-resources-plugin:jar:3.4.0 has not been downloaded from it before"
    }));
    fake_server.notify::<JdtlsStatus>(json!({
        "type": "ProjectStatus",
        "message": "WARNING"
    }));
    cx.run_until_parked();
    assert!(health_events.lock().iter().any(|(health, message)| {
        *health == proto::ServerHealth::Error as i32
            && message.contains("Java Maven import needs cached dependencies")
            && !message.contains("/private/host")
    }));
    assert!(
        server_logs
            .lock()
            .iter()
            .any(|message| message.contains("/private/host/project"))
    );

    fake_server.notify::<JdtlsStatus>(json!({
        "type": "ProjectStatus",
        "message": "WARNING"
    }));
    cx.run_until_parked();
    let events = health_events.lock();
    assert_eq!(
        events.last(),
        Some(&(
            proto::ServerHealth::Warning as i32,
            "Java project reports errors; view Problems and language server logs".to_string(),
        ))
    );
    drop(events);

    fake_server.notify::<JdtlsLog>(json!({
        "type": 1,
        "message": "Cannot access central in offline mode and the artifact at /private/host/project has not been downloaded from it before"
    }));
    cx.run_until_parked();
    assert!(
        health_events
            .lock()
            .last()
            .is_some_and(|(health, message)| {
                *health == proto::ServerHealth::Error as i32
                    && message.contains("Java Maven import needs cached dependencies")
            })
    );
    fake_server.notify::<JdtlsStatus>(json!({
        "type": "ProjectStatus",
        "message": "OK"
    }));
    fake_server.notify::<JdtlsStatus>(json!({
        "type": "ProjectStatus",
        "message": "WARNING"
    }));
    cx.run_until_parked();
    assert_eq!(
        health_events.lock().last(),
        Some(&(
            proto::ServerHealth::Warning as i32,
            "Java project reports errors; view Problems and language server logs".to_string(),
        ))
    );

    fake_server.notify::<JdtlsLog>(json!({
        "type": 1,
        "message": "Synchronize project /private/host/project failed: Caused by: org.gradle.api.plugins.UnknownPluginException: Plugin [id: 'org.jetbrains.kotlin.jvm', version: '2.0.21', apply: false] was not found; Plugin Repositories (could not resolve plugin artifact 'org.jetbrains.kotlin.jvm:org.jetbrains.kotlin.jvm.gradle.plugin:2.0.21')"
    }));
    cx.run_until_parked();
    assert!(
        health_events
            .lock()
            .last()
            .is_some_and(|(health, message)| {
                *health == proto::ServerHealth::Error as i32
                    && message.contains("Java Gradle import could not resolve a build plugin")
                    && !message.contains("/private/host")
            })
    );
    assert!(server_logs.lock().iter().any(|message| {
        message.contains("UnknownPluginException") && message.contains("/private/host/project")
    }));
    fake_server.notify::<JdtlsStatus>(json!({
        "type": "ProjectStatus",
        "message": "WARNING"
    }));
    cx.run_until_parked();
    assert!(
        health_events
            .lock()
            .last()
            .is_some_and(|(health, message)| {
                *health == proto::ServerHealth::Error as i32
                    && message.contains("Java Gradle import could not resolve a build plugin")
            })
    );
    fake_server.notify::<JdtlsStatus>(json!({
        "type": "ProjectStatus",
        "message": "OK"
    }));
    fake_server.notify::<JdtlsStatus>(json!({
        "type": "ProjectStatus",
        "message": "WARNING"
    }));
    cx.run_until_parked();
    assert_eq!(
        health_events.lock().last(),
        Some(&(
            proto::ServerHealth::Warning as i32,
            "Java project reports errors; view Problems and language server logs".to_string(),
        ))
    );
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn kotlin_import_failure_reaches_shared_lsp_health_without_host_paths(
    cx: &mut TestAppContext,
) {
    use language::{Language, LanguageConfig, LanguageMatcher};
    use rpc::proto;

    struct KotlinImportLog;
    impl lsp::notification::Notification for KotlinImportLog {
        type Params = serde_json::Value;
        const METHOD: &'static str = "intellij/importLog";
    }

    struct KotlinWorkspaceImportState;
    impl lsp::notification::Notification for KotlinWorkspaceImportState {
        type Params = serde_json::Value;
        const METHOD: &'static str = "intellij/workspaceImportState";
    }

    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/kotlin-import"),
        json!({ "Main.kt": "fun main() {}" }),
    )
    .await;
    let project = Project::test(fs, [path!("/kotlin-import").as_ref()], cx).await;
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
    let mut fake_servers = languages.register_fake_lsp(
        "Kotlin",
        FakeLspAdapter {
            name: "kotlin-lsp",
            ..Default::default()
        },
    );
    let (_buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/kotlin-import/Main.kt"), cx)
        })
        .await
        .expect("open Kotlin source");
    let fake_server = fake_servers.next().await.expect("Kotlin provider");
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let health_events = Arc::new(Mutex::new(Vec::<(i32, String)>::new()));
    cx.update({
        let health_events = health_events.clone();
        |cx| {
            cx.subscribe(&lsp_store, move |_, event, _| {
                if let LspStoreEvent::LanguageServerUpdate {
                    message: proto::update_language_server::Variant::StatusUpdate(update),
                    ..
                } = event
                {
                    if let Some(proto::status_update::Status::Health(health)) = &update.status {
                        health_events
                            .lock()
                            .push((*health, update.message.clone().unwrap_or_default()));
                    }
                }
            })
            .detach();
        }
    });

    fake_server.notify::<KotlinImportLog>(json!({
        "message": "Importing with Maven: /private/host/project",
        "tool": "Maven",
        "started": true
    }));
    fake_server.notify::<KotlinImportLog>(json!({
        "message": "Could not transfer artifact from https://private.example/repository"
    }));
    fake_server.notify::<KotlinImportLog>(json!({
        "message": "Failed to import /private/host/project",
        "failed": true
    }));
    fake_server.notify::<KotlinWorkspaceImportState>(json!({
        "phase": "FINISHED",
        "folders": [{"tool": "maven", "status": "FAILED"}]
    }));
    cx.run_until_parked();
    let worktree_id = project.read_with(cx, |project, cx| {
        project
            .worktree_store()
            .read(cx)
            .worktrees()
            .next()
            .expect("Kotlin worktree")
            .read(cx)
            .id()
    });
    project.read_with(cx, |project, cx| {
        assert!(
            project
                .lsp_store()
                .read(cx)
                .kotlin_workspace_import_failed(worktree_id)
        );
    });
    let events = health_events.lock();
    assert!(events.iter().any(|(health, message)| {
        *health == proto::ServerHealth::Error as i32
            && message.contains("could not fetch dependencies")
    }));
    assert!(events.iter().all(|(_, message)| {
        !message.contains("/private/host") && !message.contains("private.example")
    }));
    drop(events);

    fake_server.notify::<KotlinImportLog>(json!({
        "message": "Importing with Maven: /private/host/project",
        "tool": "Maven",
        "started": true
    }));
    cx.run_until_parked();
    project.read_with(cx, |project, cx| {
        assert!(
            !project
                .lsp_store()
                .read(cx)
                .kotlin_workspace_import_failed(worktree_id)
        );
    });
    fake_server.notify::<KotlinWorkspaceImportState>(json!({
        "phase": "FINISHED",
        "folders": [{"tool": "maven", "status": "SUCCESS"}]
    }));
    cx.run_until_parked();
    project.read_with(cx, |project, cx| {
        assert!(
            !project
                .lsp_store()
                .read(cx)
                .kotlin_workspace_import_failed(worktree_id)
        );
    });
    assert!(health_events.lock().iter().any(|(health, message)| {
        *health == proto::ServerHealth::Ok as i32 && message == "Kotlin project imported"
    }));

    health_events.lock().clear();
    fake_server.notify::<KotlinImportLog>(json!({
        "message": "Importing with Gradle: /private/host/project",
        "tool": "Gradle",
        "started": true
    }));
    fake_server.notify::<KotlinImportLog>(json!({
        "message": "Plugin Repositories (could not resolve plugin artifact from https://private.example/plugins)"
    }));
    fake_server.notify::<KotlinImportLog>(json!({
        "message": "org.gradle.api.plugins.UnknownPluginException: Plugin unavailable in /private/host/project"
    }));
    fake_server.notify::<KotlinImportLog>(json!({"failed": true}));
    fake_server.notify::<KotlinWorkspaceImportState>(json!({
        "phase": "FINISHED",
        "folders": [{"tool": "gradle", "status": "FAILED"}]
    }));
    cx.run_until_parked();
    project.read_with(cx, |project, cx| {
        assert!(
            project
                .lsp_store()
                .read(cx)
                .kotlin_workspace_import_failed(worktree_id)
        );
    });
    let events = health_events.lock();
    assert_eq!(
        events
            .iter()
            .filter(|(health, message)| {
                *health == proto::ServerHealth::Error as i32
                    && message.contains("cannot resolve a build plugin")
            })
            .count(),
        2
    );
    assert!(events.iter().all(|(_, message)| {
        !message.contains("/private/host") && !message.contains("private.example")
    }));
    drop(events);
    fake_server.notify::<KotlinImportLog>(json!({"tool": "Gradle", "started": true}));
    fake_server.notify::<KotlinWorkspaceImportState>(json!({
        "phase": "FINISHED",
        "folders": [{"tool": "gradle", "status": "SUCCESS"}]
    }));
    cx.run_until_parked();
    project.read_with(cx, |project, cx| {
        assert!(
            !project
                .lsp_store()
                .read(cx)
                .kotlin_workspace_import_failed(worktree_id)
        );
    });
}

#[cfg(feature = "rust-tests")]
#[gpui::test]
async fn rust_test_source_hint_integration_consumes_typed_lsp_location(cx: &mut TestAppContext) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use collections::HashSet;
    use project::{
        LanguageServerToQuery,
        lsp_store::lsp_ext_command::{
            CargoRunnableArgs, GetLspRunnables, Runnable, RunnableArgs, Runnables,
        },
        trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees},
    };

    init_test(cx);
    cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/fixture"),
        json!({
            "Cargo.toml": "[package]\nname = \"rust-fixture\"\nversion = \"0.1.0\"\n",
            "src": { "lib.rs": "#[test]\nfn unit_passes() {}\n" }
        }),
    )
    .await;
    let project = Project::test_with_worktree_trust(fs, [path!("/fixture").as_ref()], cx).await;
    struct UnavailableCargoFixture;

    #[async_trait::async_trait]
    impl project::cargo_workspace_store::CargoMetadataRunner for UnavailableCargoFixture {
        async fn run(
            &self,
            _request: project::cargo_workspace_store::CargoMetadataRequest,
        ) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("The in-memory LSP fixture has no physical Cargo metadata")
        }
    }

    #[async_trait::async_trait]
    impl project::cargo_workspace_store::CargoConfigurationProbe for UnavailableCargoFixture {
        async fn run(
            &self,
            _request: project::cargo_workspace_store::CargoConfigurationProbeRequest,
        ) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("The in-memory LSP fixture has no physical compiler")
        }
    }

    // This FakeFs scenario exercises semantic hints without a physical Cargo workspace.
    // The real-filesystem scenario below retains the production process owners.
    let cargo_store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
    cargo_store.update(cx, |store, _| {
        store
            .set_local_runners_for_test(
                Arc::new(UnavailableCargoFixture),
                Arc::new(UnavailableCargoFixture),
            )
            .expect("install in-memory Cargo prerequisites");
    });
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .worktrees()
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    let trusted_worktrees =
        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trusted worktrees global"));
    trusted_worktrees.update(cx, |store, cx| {
        store.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        );
    });
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(rust_lang());
    let mut fake_servers = languages.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "rust-analyzer",
            ..FakeLspAdapter::default()
        },
    );
    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/fixture/src/lib.rs"), cx)
        })
        .await
        .expect("open Rust source with LSP");
    let fake_server = fake_servers.next().await.expect("fake Rust Analyzer");
    let request_count = Arc::new(AtomicUsize::new(0));
    let runnable_handler = {
        let request_count = request_count.clone();
        move |_, _| {
            let request_number = request_count.fetch_add(1, Ordering::SeqCst);
            async move {
                if request_number == 0 {
                    return Ok(Vec::new());
                }
                Ok(vec![Runnable {
                    label: "test unit_passes".into(),
                    location: Some(lsp::LocationLink {
                        origin_selection_range: None,
                        target_uri: Uri::from_file_path(path!("/fixture/src/lib.rs"))
                            .expect("source URI"),
                        target_range: lsp::Range::new(
                            lsp::Position::new(1, 0),
                            lsp::Position::new(1, 18),
                        ),
                        target_selection_range: lsp::Range::new(
                            lsp::Position::new(1, 0),
                            lsp::Position::new(1, 18),
                        ),
                    }),
                    args: RunnableArgs::Cargo(CargoRunnableArgs {
                        environment: HashMap::default(),
                        cwd: path!("/fixture").into(),
                        override_cargo: None,
                        workspace_root: None,
                        cargo_args: vec![
                            "test".into(),
                            "--package".into(),
                            "rust-fixture".into(),
                            "--lib".into(),
                        ],
                        executable_args: vec!["unit_passes".into(), "--exact".into()],
                    }),
                }])
            }
        }
    };
    fake_server.set_request_handler::<Runnables, _, _>(runnable_handler.clone());
    cx.run_until_parked();
    let server_id = project.read_with(cx, |project, cx| {
        project
            .lsp_store()
            .read(cx)
            .language_server_statuses()
            .next()
            .expect("Rust Analyzer status")
            .0
    });
    let buffer_id = buffer.read_with(cx, |buffer, _| buffer.remote_id());
    let requested_version = buffer.read_with(cx, |buffer, _| buffer.version());
    let response = project
        .update(cx, |project, cx| {
            project.request_lsp(
                buffer.clone(),
                LanguageServerToQuery::Other(server_id),
                GetLspRunnables {
                    buffer_id,
                    position: None,
                    requested_version,
                },
                cx,
            )
        })
        .await
        .expect("typed runnable response");
    assert!(response.runnables.is_empty());
    let provider = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0
    );
    let discovery = provider.update(cx, |store, cx| store.refresh(cx));
    discovery
        .await
        .expect("Rust discovery refresh after empty response");
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        1
    );
    buffer.update(cx, |buffer, cx| {
        buffer.edit([(0..0, "// edited\n")], None, cx);
    });
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0
    );
    let discovery = provider.update(cx, |store, cx| store.refresh(cx));
    discovery.await.expect("Rust discovery refresh");
    assert_eq!(request_count.load(Ordering::SeqCst), 3);
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        1
    );
    let (release_reply, pending_reply) = futures::channel::oneshot::channel::<()>();
    let pending_reply = Arc::new(Mutex::new(Some(pending_reply)));
    fake_server.set_request_handler::<Runnables, _, _>({
        let runnable_handler = runnable_handler.clone();
        let pending_reply = pending_reply.clone();
        move |params, cx| {
            let response = runnable_handler(params, cx);
            let pending_reply = pending_reply.lock().take();
            async move {
                if let Some(pending_reply) = pending_reply {
                    pending_reply.await.expect("release old trust request");
                }
                response.await
            }
        }
    });
    let pending_request = project.update(cx, |project, cx| {
        project.request_lsp(
            buffer.clone(),
            LanguageServerToQuery::Other(server_id),
            GetLspRunnables {
                buffer_id: buffer.read(cx).remote_id(),
                position: None,
                requested_version: buffer.read(cx).version(),
            },
            cx,
        )
    });
    cx.run_until_parked();
    assert_eq!(request_count.load(Ordering::SeqCst), 4);
    trusted_worktrees.update(cx, |store, cx| {
        store.restrict(
            worktree_store.downgrade(),
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        );
    });
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0,
        "restricting the source worktree must revoke the cached hint"
    );
    release_reply.send(()).expect("release denied late reply");
    pending_request.await.expect("late typed runnable response");
    cx.run_until_parked();
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0,
        "late Rust Analyzer reply must not restore a restricted source hint"
    );
    provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("restricted Rust discovery refresh");
    assert_eq!(
        request_count.load(Ordering::SeqCst),
        4,
        "restricted discovery must not request source runnables"
    );
    trusted_worktrees.update(cx, |store, cx| {
        store.trust(
            &worktree_store,
            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
            cx,
        );
    });
    provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("re-trusted Rust discovery refresh");
    assert_eq!(request_count.load(Ordering::SeqCst), 5);
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        1,
        "re-trusting the worktree should permit a fresh source hint"
    );
    let (release_old_server_reply, held_old_server_reply) = futures::channel::oneshot::channel();
    let held_old_server_reply = Arc::new(Mutex::new(Some(held_old_server_reply)));
    let (old_server_replied, old_server_reply_completed) = futures::channel::oneshot::channel();
    let old_server_replied = Arc::new(Mutex::new(Some(old_server_replied)));
    fake_server.set_request_handler::<Runnables, _, _>({
        let runnable_handler = runnable_handler.clone();
        move |params, cx| {
            let response = runnable_handler(params, cx);
            let held_old_server_reply = held_old_server_reply.lock().take();
            let old_server_replied = old_server_replied.lock().take();
            async move {
                if let Some(held_old_server_reply) = held_old_server_reply {
                    held_old_server_reply
                        .await
                        .expect("release replaced server request");
                }
                let response = response.await;
                if let Some(old_server_replied) = old_server_replied {
                    old_server_replied
                        .send(())
                        .expect("observe completed old server reply");
                }
                response
            }
        }
    });
    let pending_old_server_request = project.update(cx, |project, cx| {
        project.request_lsp(
            buffer.clone(),
            LanguageServerToQuery::Other(server_id),
            GetLspRunnables {
                buffer_id: buffer.read(cx).remote_id(),
                position: None,
                requested_version: buffer.read(cx).version(),
            },
            cx,
        )
    });
    cx.run_until_parked();
    assert_eq!(request_count.load(Ordering::SeqCst), 6);
    project.update(cx, |project, cx| {
        project.restart_language_servers_for_buffers(
            vec![buffer.clone()],
            HashSet::default(),
            true,
            cx,
        );
    });
    let replacement_server = fake_servers
        .next()
        .await
        .expect("replacement Rust Analyzer");
    cx.run_until_parked();
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0,
        "removing the original Rust Analyzer must revoke its source hint"
    );
    release_old_server_reply
        .send(())
        .expect("release replaced server reply");
    old_server_reply_completed
        .await
        .expect("old server handler completed after replacement");
    assert!(
        pending_old_server_request.await.is_err(),
        "old Rust Analyzer request must be rejected after server replacement"
    );
    cx.run_until_parked();
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0,
        "completed reply from the replaced Rust Analyzer must not restore its source hint"
    );
    replacement_server.set_request_handler::<Runnables, _, _>(runnable_handler);
    provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("fresh discovery through replacement Rust Analyzer");
    assert_eq!(request_count.load(Ordering::SeqCst), 7);
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        1,
        "replacement Rust Analyzer should publish a fresh source hint"
    );
}

#[cfg(feature = "rust-tests")]
#[gpui::test]
async fn rust_test_source_hint_integration_projects_real_cargo_case(cx: &mut TestAppContext) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use collections::HashSet;
    use project::{
        LanguageServerToQuery,
        lsp_store::lsp_ext_command::{
            CargoRunnableArgs, GetLspRunnables, Runnable, RunnableArgs, Runnables,
        },
    };

    init_test(cx);
    cx.executor().allow_parking();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test_data/rust_test_provider/physical-workspace");
    let replacement_root = root.clone();
    let source = root.join("tests/integration.rs");
    let project = Project::test(
        Arc::new(fs::RealFs::new(None, cx.executor())),
        [root.as_path()],
        cx,
    )
    .await;
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    languages.add(rust_lang());
    let mut fake_servers = languages.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "rust-analyzer",
            ..FakeLspAdapter::default()
        },
    );
    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(source.as_path(), cx)
        })
        .await
        .expect("open Rust integration source with LSP");
    let fake_server = fake_servers.next().await.expect("fake Rust Analyzer");
    let source_uri = Arc::new(Mutex::new(
        Uri::from_file_path(&source).expect("source URI"),
    ));
    let request_count = Arc::new(AtomicUsize::new(0));
    let runnable_handler = {
        let request_count = request_count.clone();
        let source_uri = source_uri.clone();
        move |_, _| {
            request_count.fetch_add(1, Ordering::SeqCst);
            let source_uri = source_uri.lock().clone();
            let root = root.clone();
            async move {
                Ok(vec![Runnable {
                    label: "test integration_passes".into(),
                    location: Some(lsp::LocationLink {
                        origin_selection_range: None,
                        target_uri: source_uri,
                        target_range: lsp::Range::new(
                            lsp::Position::new(1, 3),
                            lsp::Position::new(1, 21),
                        ),
                        target_selection_range: lsp::Range::new(
                            lsp::Position::new(1, 3),
                            lsp::Position::new(1, 21),
                        ),
                    }),
                    args: RunnableArgs::Cargo(CargoRunnableArgs {
                        environment: HashMap::default(),
                        cwd: root.clone(),
                        override_cargo: None,
                        workspace_root: None,
                        cargo_args: vec![
                            "test".into(),
                            "--package".into(),
                            "zed-rust-tools-physical-fixture".into(),
                            "--test".into(),
                            "integration".into(),
                        ],
                        executable_args: vec![
                            "integration_passes".into(),
                            "--exact".into(),
                            "--nocapture".into(),
                            "--include-ignored".into(),
                        ],
                    }),
                }])
            }
        }
    };
    fake_server.set_request_handler::<Runnables, _, _>(runnable_handler.clone());
    cx.run_until_parked();
    let provider = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
    let snapshot = provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Rust test discovery");
    assert_eq!(
        request_count.load(Ordering::SeqCst),
        1,
        "runnable request count"
    );
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        1,
        "accepted semantic hint count"
    );
    let case = snapshot
        .nodes
        .iter()
        .find(|node| node.label == "integration_passes")
        .expect("discovered integration test");
    assert!(case.path.is_some(), "semantic source path was projected");
    assert_eq!(
        case.path.as_ref().map(|path| path.path.as_unix_str()),
        Some("tests/integration.rs")
    );

    let (release_reply, pending_reply) = futures::channel::oneshot::channel::<()>();
    let pending_reply = Arc::new(Mutex::new(Some(pending_reply)));
    fake_server.set_request_handler::<Runnables, _, _>({
        let runnable_handler = runnable_handler.clone();
        let pending_reply = pending_reply.clone();
        move |params, cx| {
            let response = runnable_handler(params, cx);
            let pending_reply = pending_reply.lock().take();
            async move {
                if let Some(pending_reply) = pending_reply {
                    pending_reply
                        .await
                        .expect("release delayed Rust runnable reply");
                }
                response.await
            }
        }
    });
    let server_id = project.read_with(cx, |project, cx| {
        project
            .lsp_store()
            .read(cx)
            .language_server_statuses()
            .next()
            .expect("Rust Analyzer status")
            .0
    });
    let pending_request = project.update(cx, |project, cx| {
        project.request_lsp(
            buffer.clone(),
            LanguageServerToQuery::Other(server_id),
            GetLspRunnables {
                buffer_id: buffer.read(cx).remote_id(),
                position: None,
                requested_version: buffer.read(cx).version(),
            },
            cx,
        )
    });
    cx.run_until_parked();
    assert_eq!(request_count.load(Ordering::SeqCst), 2);

    let replacement_project = Project::test(
        Arc::new(fs::RealFs::new(None, cx.executor())),
        [replacement_root.as_path()],
        cx,
    )
    .await;
    assert_ne!(project.entity_id(), replacement_project.entity_id());
    // Deliver initial scan notifications before explicitly starting discovery so
    // their debounced invalidation cannot supersede this fixture's refresh.
    cx.run_until_parked();
    let replacement_provider =
        replacement_project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
    let replacement_snapshot = replacement_provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("fresh Project Rust discovery");
    assert_eq!(
        replacement_provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0,
        "new Project must not inherit the old Project's semantic hint"
    );
    let replacement_case = replacement_snapshot
        .nodes
        .iter()
        .find(|node| node.label == "integration_passes")
        .expect("integration test in fresh Project");
    assert!(
        replacement_case.path.is_none(),
        "new Project inherited the old Project's source path"
    );
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        1,
        "old Project retains its own valid hint"
    );
    release_reply
        .send(())
        .expect("release old Project's Rust runnable reply");
    pending_request.await.expect("old Project's delayed reply");
    cx.run_until_parked();
    assert_eq!(
        replacement_provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0,
        "old Project's delayed reply must not update the new Project"
    );
    let mut replacement_after_reply = None;
    for attempt in 0..3 {
        match replacement_provider
            .update(cx, |store, cx| store.refresh(cx))
            .await
        {
            Ok(snapshot) => {
                replacement_after_reply = Some(snapshot);
                break;
            }
            Err(error)
                if attempt < 2 && error.to_string() == "stale Rust test discovery generation" =>
            {
                cx.run_until_parked();
            }
            Err(error) => panic!("replacement discovery after the old reply: {error:#}"),
        }
    }
    let replacement_case = replacement_after_reply
        .expect("replacement discovery after the old reply")
        .nodes
        .into_iter()
        .find(|node| node.label == "integration_passes")
        .expect("integration test in replacement Project");
    assert!(replacement_case.path.is_none());

    project.update(cx, |project, cx| {
        project.restart_language_servers_for_buffers(
            vec![buffer.clone()],
            HashSet::default(),
            true,
            cx,
        );
    });
    let replacement_server = fake_servers
        .next()
        .await
        .expect("replacement Rust Analyzer");
    replacement_server.set_request_handler::<Runnables, _, _>(|_, _| async { Ok(Vec::new()) });
    cx.run_until_parked();
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0,
        "the removed server's source hint must be discarded"
    );
    let after_replacement = provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Rust discovery after language server replacement");
    let case = after_replacement
        .nodes
        .iter()
        .find(|node| node.label == "integration_passes")
        .expect("integration test after server replacement");
    assert!(
        case.path.is_none(),
        "the removed server's source path must not survive discovery"
    );
    replacement_server.set_request_handler::<Runnables, _, _>(runnable_handler);

    let external_directory = tempfile::tempdir().expect("external source directory");
    let external_source = external_directory.path().join("external.rs");
    std::fs::write(&external_source, "fn external() {}\n").expect("external source file");
    *source_uri.lock() = Uri::from_file_path(&external_source).expect("external source URI");
    buffer.update(cx, |buffer, cx| {
        buffer.edit([(0..0, "// edited\n")], None, cx);
    });
    let server_id = project.read_with(cx, |project, cx| {
        project
            .lsp_store()
            .read(cx)
            .language_server_statuses()
            .next()
            .expect("Rust Analyzer status")
            .0
    });
    let buffer_id = buffer.read_with(cx, |buffer, _| buffer.remote_id());
    let requested_version = buffer.read_with(cx, |buffer, _| buffer.version());
    let external_response = project
        .update(cx, |project, cx| {
            project.request_lsp(
                buffer.clone(),
                LanguageServerToQuery::Other(server_id),
                GetLspRunnables {
                    buffer_id,
                    position: None,
                    requested_version,
                },
                cx,
            )
        })
        .await
        .expect("external runnable response");
    assert_eq!(external_response.runnables.len(), 1);
    let external = provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Rust test discovery after external location");
    assert_eq!(request_count.load(Ordering::SeqCst), 3);
    let case = external
        .nodes
        .iter()
        .find(|node| node.label == "integration_passes")
        .expect("discovered integration test after external location");
    assert!(case.path.is_none(), "external source path was projected");

    cx.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files = Some(settings::ExtendingVec(vec![
                    "**/tests/integration.rs".into(),
                ]));
            });
        });
    });
    cx.run_until_parked();
    project.read_with(cx, |project, cx| {
        let worktree = project.worktrees(cx).next().expect("fixture worktree");
        assert!(
            worktree
                .read(cx)
                .snapshot()
                .entry_for_path(util::rel_path::rel_path("tests/integration.rs"))
                .expect("integration source entry")
                .is_private
        );
    });
    let mut restricted = None;
    for attempt in 0..3 {
        match provider.update(cx, |store, cx| store.refresh(cx)).await {
            Ok(snapshot) => {
                restricted = Some(snapshot);
                break;
            }
            Err(error)
                if attempt < 2 && error.to_string() == "stale Rust test discovery generation" =>
            {
                cx.run_until_parked();
            }
            Err(error) => panic!("Rust test discovery after privacy change: {error:#}"),
        }
    }
    let restricted = restricted.expect("Rust test discovery after privacy change");
    let case = restricted
        .nodes
        .iter()
        .find(|node| node.label == "integration_passes")
        .expect("discovered integration test after privacy change");
    assert!(case.path.is_none(), "private source path was projected");

    cx.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files = Some(settings::ExtendingVec(Vec::new()));
            });
        });
    });
    *source_uri.lock() = Uri::from_file_path(&source).expect("restored source URI");
    buffer.update(cx, |buffer, cx| {
        buffer.edit([(0..0, "// restored\n")], None, cx);
    });
    cx.run_until_parked();
    let restored = provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Rust discovery after source restoration");
    let case = restored
        .nodes
        .iter()
        .find(|node| node.label == "integration_passes")
        .expect("integration test after source restoration");
    assert!(
        case.path.is_some(),
        "restored source hint was not projected"
    );

    let worktree_id = project.read_with(cx, |project, cx| {
        project
            .worktrees(cx)
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    project.update(cx, |project, cx| {
        project.remove_worktree(worktree_id, cx);
    });
    cx.run_until_parked();
    assert_eq!(
        provider.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
        0,
        "removed worktree must discard the source hint"
    );
    let removed = provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Rust discovery after worktree removal");
    assert!(
        removed
            .nodes
            .iter()
            .all(|node| node.label != "integration_passes"),
        "removed worktree retained the old test case"
    );

    let replacement = project
        .update(cx, |project, cx| {
            project.create_worktree(replacement_root.as_path(), true, cx)
        })
        .await
        .expect("replacement fixture worktree");
    assert_ne!(
        replacement.read_with(cx, |worktree, _| worktree.id()),
        worktree_id,
        "replacement worktree must have a new identity"
    );
    cx.run_until_parked();
    let replaced = provider
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Rust discovery after worktree replacement");
    let case = replaced
        .nodes
        .iter()
        .find(|node| node.label == "integration_passes")
        .expect("rediscovered test in replacement worktree");
    assert!(
        case.path.is_none(),
        "source path from the removed worktree was projected onto its replacement"
    );
}

#[gpui::test]
async fn test_removing_invisible_worktree_cleans_reused_lsp_bookkeeping(cx: &mut TestAppContext) {
    init_test(cx);
    cx.executor().allow_parking();

    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(path!("/the-root"), json!({ "main.rs": "fn main() {}" }))
        .await;
    fs.insert_tree(
        path!("/the-registry"),
        json!({ "dep": { "src": { "dep.rs": "pub fn dep() {}" } } }),
    )
    .await;

    let project = Project::test(fs, [path!("/the-root").as_ref()], cx).await;
    let language_registry = project.read_with(cx, |project, _| project.languages().clone());
    language_registry.add(rust_lang());
    let mut fake_servers = language_registry.register_fake_lsp("Rust", FakeLspAdapter::default());

    let (_visible_buffer, _visible_handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/the-root/main.rs"), cx)
        })
        .await
        .unwrap();
    fake_servers.next().await.unwrap();
    cx.run_until_parked();

    let server_id = project.read_with(cx, |project, cx| {
        project
            .lsp_store()
            .read(cx)
            .language_server_statuses()
            .next()
            .unwrap()
            .0
    });
    let external_buffer = project
        .update(cx, |project, cx| {
            project.open_local_buffer_via_lsp(
                Uri::from_file_path(path!("/the-registry/dep/src/dep.rs")).unwrap(),
                server_id,
                cx,
            )
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let invisible_worktree_id =
        external_buffer.read_with(cx, |buffer, cx| buffer.file().unwrap().worktree_id(cx));
    project.read_with(cx, |project, cx| {
        let worktree = project.worktree_for_id(invisible_worktree_id, cx).unwrap();
        assert!(!worktree.read(cx).is_visible());
        assert!(
            project
                .lsp_store()
                .read(cx)
                .has_language_server_seed_for_worktree(invisible_worktree_id)
        );
    });

    project.update(cx, |project, cx| {
        project.remove_worktree(invisible_worktree_id, cx);
    });
    cx.run_until_parked();

    project.read_with(cx, |project, cx| {
        let lsp_store = project.lsp_store();
        let lsp_store = lsp_store.read(cx);
        assert!(
            lsp_store
                .language_server_statuses()
                .any(|(status_server_id, _)| status_server_id == server_id)
        );
        assert!(!lsp_store.has_language_server_seed_for_worktree(invisible_worktree_id));
    });
}

#[test]
fn test_rpc_log_grouping_separates_timed_messages() {
    for (received, direction) in [(false, "Send"), (true, "Receive")] {
        let mut header_state = TestRpcLogHeaderState::new();

        assert_eq!(
            header_state.header_for_message(received, None),
            Some(format!("\n// {direction}:"))
        );
        assert_eq!(header_state.header_for_message(received, None), None);
        assert_eq!(
            header_state.header_for_message(received, Some(Duration::from_millis(53))),
            Some(format!("\n// {direction} (took 53.0ms):"))
        );
        assert_eq!(
            header_state.header_for_message(received, None),
            Some(format!("\n// {direction}:"))
        );
        assert_eq!(header_state.header_for_message(received, None), None);
    }
}

#[test]
fn test_rpc_request_tracker_distinguishes_request_directions() {
    let mut tracker = TestRpcRequestTracker::new();
    let started_at = Instant::now();

    assert_eq!(
        tracker.observe(
            false,
            r#"{"jsonrpc":"2.0","id":1,"method":"textDocument/hover"}"#,
            started_at,
        ),
        None
    );
    assert_eq!(
        tracker.observe(
            true,
            r#"{"jsonrpc":"2.0","id":1,"method":"workspace/configuration"}"#,
            started_at + Duration::from_millis(10),
        ),
        None
    );
    assert_eq!(
        tracker.observe(
            false,
            r#"{"jsonrpc":"2.0","id":1,"result":[]}"#,
            started_at + Duration::from_millis(30),
        ),
        Some(Duration::from_millis(20))
    );
    assert_eq!(
        tracker.observe(
            true,
            r#"{"jsonrpc":"2.0","id":1,"result":null}"#,
            started_at + Duration::from_millis(50),
        ),
        Some(Duration::from_millis(50))
    );
}

#[test]
fn test_rpc_request_tracker_decodes_ids_and_times_cancelled_requests() {
    let mut tracker = TestRpcRequestTracker::new();
    let started_at = Instant::now();

    tracker.observe(
        true,
        r#"{"jsonrpc":"2.0","id":"foo\u002fbar","method":"workspace/configuration"}"#,
        started_at,
    );
    assert_eq!(
        tracker.observe(
            false,
            r#"{"jsonrpc":"2.0","id":"foo/bar","result":[]}"#,
            started_at + Duration::from_millis(25),
        ),
        Some(Duration::from_millis(25))
    );

    tracker.observe(
        false,
        r#"{"jsonrpc":"2.0","id":7,"method":"textDocument/hover"}"#,
        started_at,
    );
    tracker.observe(
        false,
        r#"{"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":7}}"#,
        started_at + Duration::from_millis(1),
    );
    assert_eq!(tracker.pending_request_count(), 1);
    assert_eq!(
        tracker.observe(
            true,
            r#"{"jsonrpc":"2.0","id":7,"error":{"code":-32800,"message":"Request was cancelled"}}"#,
            started_at + Duration::from_millis(10),
        ),
        Some(Duration::from_millis(10))
    );
    assert_eq!(tracker.pending_request_count(), 0);
}

#[test]
fn test_rpc_request_tracker_bounds_unanswered_requests() {
    let mut tracker = TestRpcRequestTracker::new();
    let started_at = Instant::now();
    let max_pending_requests = TestRpcRequestTracker::max_pending_requests();

    for id in 0..=max_pending_requests {
        tracker.observe(
            false,
            &format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"textDocument/hover"}}"#),
            started_at + Duration::from_nanos(id as u64),
        );
    }

    assert_eq!(tracker.pending_request_count(), max_pending_requests);
    assert_eq!(
        tracker.observe(
            true,
            r#"{"jsonrpc":"2.0","id":0,"result":null}"#,
            started_at + Duration::from_secs(1),
        ),
        None
    );
    assert!(
        tracker
            .observe(
                true,
                r#"{"jsonrpc":"2.0","id":1,"result":null}"#,
                started_at + Duration::from_secs(1),
            )
            .is_some()
    );
}

#[test]
fn test_rpc_log_duration_proto_roundtrip() {
    let log_type = LanguageServerLogType::Rpc {
        received: true,
        elapsed: Some(Duration::from_micros(1234)),
    };

    assert_eq!(
        LanguageServerLogType::from_proto(log_type.to_proto()),
        log_type
    );
}

#[test]
fn test_glob_literal_prefix() {
    assert_eq!(glob_literal_prefix(Path::new("**/*.js")), Path::new(""));
    assert_eq!(
        glob_literal_prefix(Path::new("node_modules/**/*.js")),
        Path::new("node_modules")
    );
    assert_eq!(
        glob_literal_prefix(Path::new("foo/{bar,baz}.js")),
        Path::new("foo")
    );
    assert_eq!(
        glob_literal_prefix(Path::new("foo/bar/baz.js")),
        Path::new("foo/bar/baz.js")
    );

    #[cfg(target_os = "windows")]
    {
        assert_eq!(glob_literal_prefix(Path::new("**\\*.js")), Path::new(""));
        assert_eq!(
            glob_literal_prefix(Path::new("node_modules\\**/*.js")),
            Path::new("node_modules")
        );
        assert_eq!(
            glob_literal_prefix(Path::new("foo/{bar,baz}.js")),
            Path::new("foo")
        );
        assert_eq!(
            glob_literal_prefix(Path::new("foo\\bar\\baz.js")),
            Path::new("foo/bar/baz.js")
        );
    }
}

#[test]
fn test_multi_len_chars_normalization() {
    let mut label = CodeLabel::new(
        "myElˇ (parameter) myElˇ: {\n    foo: string;\n}".to_string(),
        0..6,
        vec![(0..6, HighlightId::new(1))],
    );
    ensure_uniform_list_compatible_label(&mut label);
    assert_eq!(
        label,
        CodeLabel::new(
            "myElˇ (parameter) myElˇ: { foo: string; }".to_string(),
            0..6,
            vec![(0..6, HighlightId::new(1))],
        )
    );
}

#[test]
fn test_trailing_newline_in_completion_documentation() {
    let doc =
        lsp::Documentation::String("Inappropriate argument value (of correct type).\n".to_string());
    let completion_doc: CompletionDocumentation = doc.into();
    assert!(
        matches!(completion_doc, CompletionDocumentation::SingleLine(s) if s == "Inappropriate argument value (of correct type).")
    );

    let doc = lsp::Documentation::String("  some value  \n".to_string());
    let completion_doc: CompletionDocumentation = doc.into();
    assert!(matches!(
        completion_doc,
        CompletionDocumentation::SingleLine(s) if s == "some value"
    ));
}

#[gpui::test]
async fn test_user_initialization_options_override_adapter_arrays(cx: &mut TestAppContext) {
    init_test(cx);

    let user_settings = serde_json::json!({
        "jvm_builds": {"maven": {"selected_poms": []}, "gradle": {"ignored_projects": [""]}},
        "lsp": {
            "the-fake-language-server": {
                "initialization_options": {
                    "preview": {
                        "background": {
                            "enabled": true,
                            "args": ["--data-plane-host=127.0.0.1:23635", "--invert-colors=never"],
                        },
                    },
                    "plugins": ["user-plugin"],
                    "userOnly": ["user"],
                },
            },
        },
    });

    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/the-root"),
        json!({
            ".zed": {
                "settings.json": user_settings.to_string(),
            },
            "main.rs": "fn main() {}",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/the-root").as_ref()], cx).await;
    let language_registry = project.read_with(cx, |project, _| project.languages().clone());
    language_registry.add(rust_lang());

    let sent_initialization_options = Arc::new(Mutex::new(None));
    let mut fake_servers = language_registry.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "the-fake-language-server",
            initialization_options: Some(json!({
                "preview": {
                    "background": {
                        "args": ["--data-plane-host=127.0.0.1:23635", "--invert-colors=never"],
                        "partialRendering": true,
                    },
                },
                "plugins": ["default-plugin", "user-plugin"],
                "adapterOnly": [1, 2],
            })),
            initializer: Some(Box::new({
                let sent_initialization_options = sent_initialization_options.clone();
                move |fake_server| {
                    let sent_initialization_options = sent_initialization_options.clone();
                    fake_server.set_request_handler::<lsp::request::Initialize, _, _>(
                        move |params, _| {
                            *sent_initialization_options.lock() = params.initialization_options;
                            async move { Ok(lsp::InitializeResult::default()) }
                        },
                    );
                }
            })),
            ..FakeLspAdapter::default()
        },
    );
    cx.run_until_parked();

    project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/the-root/main.rs"), cx)
        })
        .await
        .unwrap();
    fake_servers.next().await.unwrap();
    cx.run_until_parked();

    assert_eq!(
        sent_initialization_options.lock().take(),
        Some(json!({
            "preview": {
                "background": {
                    "enabled": true,
                    "args": ["--data-plane-host=127.0.0.1:23635", "--invert-colors=never"],
                    "partialRendering": true,
                },
            },
            "plugins": ["user-plugin"],
            "adapterOnly": [1, 2],
            "userOnly": ["user"],
        })),
    );
}

#[gpui::test]
async fn test_other_adapters_lsp_configuration_contributions_are_unioned(cx: &mut TestAppContext) {
    init_test(cx);

    let user_settings = serde_json::json!({
        "lsp": {
            "the-fake-language-server": {
                "initialization_options": {
                    "languages": ["user-lang"],
                    "userOnly": true,
                },
            },
        },
    });

    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/the-root"),
        json!({
            ".zed": {
                "settings.json": user_settings.to_string(),
            },
            "main.rs": "fn main() {}",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/the-root").as_ref()], cx).await;
    let language_registry = project.read_with(cx, |project, _| project.languages().clone());
    language_registry.add(rust_lang());

    let main_server_name = LanguageServerName("the-fake-language-server".into());
    for (language, server_name, plugin, lang, memory) in [
        ("Vue", "vue-language-server", "vue-plugin", "vue", 4096),
        (
            "Astro",
            "astro-language-server",
            "astro-plugin",
            "astro",
            2048,
        ),
    ] {
        let contribution = json!({
            "tsserver": {
                "globalPlugins": ["shared-plugin", plugin],
                "maxMemory": memory,
            },
            "languages": [lang],
        });
        language_registry.register_fake_lsp_adapter(
            language,
            FakeLspAdapter {
                name: server_name,
                additional_initialization_options: HashMap::from_iter([(
                    main_server_name.clone(),
                    contribution.clone(),
                )]),
                additional_workspace_configuration: HashMap::from_iter([(
                    main_server_name.clone(),
                    contribution,
                )]),
                ..FakeLspAdapter::default()
            },
        );
    }

    let sent_initialization_options = Arc::new(Mutex::new(None));
    let mut fake_servers = language_registry.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "the-fake-language-server",
            initialization_options: Some(json!({
                "tsserver": {
                    "globalPlugins": ["default-plugin"],
                },
                "languages": ["default-lang"],
            })),
            initializer: Some(Box::new({
                let sent_initialization_options = sent_initialization_options.clone();
                move |fake_server| {
                    let sent_initialization_options = sent_initialization_options.clone();
                    fake_server.set_request_handler::<lsp::request::Initialize, _, _>(
                        move |params, _| {
                            *sent_initialization_options.lock() = params.initialization_options;
                            async move { Ok(lsp::InitializeResult::default()) }
                        },
                    );
                }
            })),
            ..FakeLspAdapter::default()
        },
    );
    cx.run_until_parked();

    project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/the-root/main.rs"), cx)
        })
        .await
        .unwrap();
    let mut fake_server = fake_servers.next().await.unwrap();
    let workspace_configuration = fake_server
        .receive_notification::<lsp::notification::DidChangeConfiguration>()
        .await
        .settings;
    cx.run_until_parked();

    assert_eq!(
        sent_initialization_options.lock().take(),
        Some(json!({
            "tsserver": {
                "globalPlugins": ["default-plugin", "shared-plugin", "astro-plugin", "vue-plugin"],
                "maxMemory": 4096,
            },
            "languages": ["user-lang"],
            "userOnly": true,
        })),
    );
    assert_eq!(
        workspace_configuration,
        json!({
            "tsserver": {
                "globalPlugins": ["shared-plugin", "astro-plugin", "vue-plugin"],
                "maxMemory": 4096,
            },
            "languages": ["astro", "vue"],
        }),
    );
}

#[gpui::test]
async fn test_initialization_options_contributions_without_own_options(cx: &mut TestAppContext) {
    init_test(cx);

    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(path!("/the-root"), json!({ "main.rs": "fn main() {}" }))
        .await;

    let project = Project::test(fs, [path!("/the-root").as_ref()], cx).await;
    let language_registry = project.read_with(cx, |project, _| project.languages().clone());
    language_registry.add(rust_lang());

    let contribution = json!({
        "tsserver": {
            "globalPlugins": ["vue-plugin"],
        },
    });
    language_registry.register_fake_lsp_adapter(
        "Vue",
        FakeLspAdapter {
            name: "vue-language-server",
            additional_initialization_options: HashMap::from_iter([(
                LanguageServerName("the-fake-language-server".into()),
                contribution.clone(),
            )]),
            ..FakeLspAdapter::default()
        },
    );

    let sent_initialization_options = Arc::new(Mutex::new(None));
    let mut fake_servers = language_registry.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "the-fake-language-server",
            initialization_options: None,
            initializer: Some(Box::new({
                let sent_initialization_options = sent_initialization_options.clone();
                move |fake_server| {
                    let sent_initialization_options = sent_initialization_options.clone();
                    fake_server.set_request_handler::<lsp::request::Initialize, _, _>(
                        move |params, _| {
                            *sent_initialization_options.lock() =
                                Some(params.initialization_options);
                            async move { Ok(lsp::InitializeResult::default()) }
                        },
                    );
                }
            })),
            ..FakeLspAdapter::default()
        },
    );
    cx.run_until_parked();

    project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/the-root/main.rs"), cx)
        })
        .await
        .unwrap();
    fake_servers.next().await.unwrap();
    cx.run_until_parked();

    assert_eq!(
        sent_initialization_options.lock().take(),
        Some(Some(contribution)),
    );
}
