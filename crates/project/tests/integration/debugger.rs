mod go_locator {
    use collections::HashMap;
    use dap::{DapLocator, adapters::DebugAdapterName};
    use gpui::TestAppContext;
    use project::debugger::locators::go::{DelveLaunchRequest, GoLocator};
    use task::{HideStrategy, RevealStrategy, RevealTarget, SaveStrategy, Shell, TaskTemplate};
    #[gpui::test]
    async fn test_create_scenario_for_go_build(_: &mut TestAppContext) {
        let locator = GoLocator;
        let task = TaskTemplate {
            experimental_policy_version: 0,
            required_experimental_features: Vec::new(),
            label: "go build".into(),
            command: "go".into(),
            args: vec!["build".into(), ".".into()],
            env: Default::default(),
            cwd: Some("${ZED_WORKTREE_ROOT}".into()),
            use_new_terminal: false,
            allow_concurrent_runs: false,
            reveal: RevealStrategy::Always,
            reveal_target: RevealTarget::Dock,
            hide: HideStrategy::Never,
            shell: Shell::System,
            tags: vec![],
            show_summary: true,
            show_command: true,
            save: SaveStrategy::default(),
            hooks: Default::default(),
            artifact: None,
            artifact_argument_binding: None,
            artifact_group_argument_binding: None,
            stdout_report: None,
            literal_arguments: false,
            artifact_group: None,
        };

        let scenario = locator
            .create_scenario(&task, "test label", &DebugAdapterName("Delve".into()))
            .await;

        assert!(scenario.is_none());
    }

    #[gpui::test]
    async fn test_skip_non_go_commands_with_non_delve_adapter(_: &mut TestAppContext) {
        let locator = GoLocator;
        let task = TaskTemplate {
            experimental_policy_version: 0,
            required_experimental_features: Vec::new(),
            label: "cargo build".into(),
            command: "cargo".into(),
            args: vec!["build".into()],
            env: Default::default(),
            cwd: Some("${ZED_WORKTREE_ROOT}".into()),
            use_new_terminal: false,
            allow_concurrent_runs: false,
            reveal: RevealStrategy::Always,
            reveal_target: RevealTarget::Dock,
            hide: HideStrategy::Never,
            shell: Shell::System,
            tags: vec![],
            show_summary: true,
            show_command: true,
            save: SaveStrategy::default(),
            hooks: Default::default(),
            artifact: None,
            artifact_argument_binding: None,
            artifact_group_argument_binding: None,
            stdout_report: None,
            literal_arguments: false,
            artifact_group: None,
        };

        let scenario = locator
            .create_scenario(
                &task,
                "test label",
                &DebugAdapterName("SomeOtherAdapter".into()),
            )
            .await;
        assert!(scenario.is_none());

        let scenario = locator
            .create_scenario(&task, "test label", &DebugAdapterName("Delve".into()))
            .await;
        assert!(scenario.is_none());
    }
    #[gpui::test]
    async fn test_go_locator_run(_: &mut TestAppContext) {
        let locator = GoLocator;
        let delve = DebugAdapterName("Delve".into());

        let task = TaskTemplate {
            label: "go run with flags".into(),
            command: "go".into(),
            args: vec![
                "run".to_string(),
                "-race".to_string(),
                "-ldflags".to_string(),
                "-X main.version=1.0".to_string(),
                "./cmd/myapp".to_string(),
                "--config".to_string(),
                "production.yaml".to_string(),
                "--verbose".to_string(),
            ],
            env: {
                let mut env = HashMap::default();
                env.insert("GO_ENV".to_string(), "production".to_string());
                env
            },
            cwd: Some("/project/root".into()),
            ..Default::default()
        };

        let scenario = locator
            .create_scenario(&task, "test run label", &delve)
            .await
            .unwrap();

        let config: DelveLaunchRequest = serde_json::from_value(scenario.config).unwrap();

        assert_eq!(
            config,
            DelveLaunchRequest {
                request: "launch".to_string(),
                mode: "debug".to_string(),
                program: "./cmd/myapp".to_string(),
                build_flags: vec![
                    "-race".to_string(),
                    "-ldflags".to_string(),
                    "-X main.version=1.0".to_string()
                ],
                args: vec![
                    "--config".to_string(),
                    "production.yaml".to_string(),
                    "--verbose".to_string(),
                ],
                env: {
                    let mut env = HashMap::default();
                    env.insert("GO_ENV".to_string(), "production".to_string());
                    env
                },
                cwd: Some("/project/root".to_string()),
            }
        );
    }

    #[gpui::test]
    async fn test_go_locator_test(_: &mut TestAppContext) {
        let locator = GoLocator;
        let delve = DebugAdapterName("Delve".into());

        // Test with tags and run flag
        let task_with_tags = TaskTemplate {
            label: "test".into(),
            command: "go".into(),
            args: vec![
                "test".to_string(),
                "-tags".to_string(),
                "integration,unit".to_string(),
                "-run".to_string(),
                "Foo".to_string(),
                ".".to_string(),
            ],
            ..Default::default()
        };
        let result = locator
            .create_scenario(&task_with_tags, "", &delve)
            .await
            .unwrap();

        let config: DelveLaunchRequest = serde_json::from_value(result.config).unwrap();

        assert_eq!(
            config,
            DelveLaunchRequest {
                request: "launch".to_string(),
                mode: "test".to_string(),
                program: ".".to_string(),
                build_flags: vec!["-tags".to_string(), "integration,unit".to_string(),],
                args: vec![
                    "-test.run".to_string(),
                    "Foo".to_string(),
                    "-test.v".to_string()
                ],
                env: Default::default(),
                cwd: None,
            }
        );
    }

    #[gpui::test]
    async fn test_go_locator_unescapes_nested_subtest_regex(_: &mut TestAppContext) {
        let locator = GoLocator;
        let delve = DebugAdapterName("Delve".into());

        // Delve receives the `-run` regex with no shell, so GoLocator must strip
        // the escaping itself.
        let task = TaskTemplate {
            label: "test subtest".into(),
            command: "go".into(),
            args: vec![
                "test".to_string(),
                "-v".to_string(),
                "-run".to_string(),
                "\\^TestFoo\\$/\\^simple_subtest\\$".to_string(),
            ],
            ..Default::default()
        };
        let result = locator.create_scenario(&task, "", &delve).await.unwrap();
        let config: DelveLaunchRequest = serde_json::from_value(result.config).unwrap();
        assert_eq!(
            config,
            DelveLaunchRequest {
                request: "launch".to_string(),
                mode: "test".to_string(),
                program: ".".to_string(),
                build_flags: vec![],
                args: vec![
                    "-test.v".to_string(),
                    "-test.run".to_string(),
                    "^TestFoo$/^simple_subtest$".to_string(),
                ],
                env: Default::default(),
                cwd: None,
            }
        );
    }

    #[gpui::test]
    async fn test_skip_unsupported_go_commands(_: &mut TestAppContext) {
        let locator = GoLocator;
        let task = TaskTemplate {
            experimental_policy_version: 0,
            required_experimental_features: Vec::new(),
            label: "go clean".into(),
            command: "go".into(),
            args: vec!["clean".into()],
            env: Default::default(),
            cwd: Some("${ZED_WORKTREE_ROOT}".into()),
            use_new_terminal: false,
            allow_concurrent_runs: false,
            reveal: RevealStrategy::Always,
            reveal_target: RevealTarget::Dock,
            hide: HideStrategy::Never,
            shell: Shell::System,
            tags: vec![],
            show_summary: true,
            show_command: true,
            save: SaveStrategy::default(),
            hooks: Default::default(),
            artifact: None,
            artifact_argument_binding: None,
            artifact_group_argument_binding: None,
            stdout_report: None,
            literal_arguments: false,
            artifact_group: None,
        };

        let scenario = locator
            .create_scenario(&task, "test label", &DebugAdapterName("Delve".into()))
            .await;
        assert!(scenario.is_none());
    }
}

mod python_locator {
    use dap::{DapLocator, adapters::DebugAdapterName};
    use serde_json::json;

    use project::debugger::locators::python::*;
    use task::{DebugScenario, TaskTemplate};

    #[gpui::test]
    async fn test_python_locator() {
        let adapter = DebugAdapterName("Debugpy".into());
        let build_task = TaskTemplate {
            experimental_policy_version: 0,
            required_experimental_features: Vec::new(),
            label: "run module '$ZED_FILE'".into(),
            command: "$ZED_CUSTOM_PYTHON_ACTIVE_ZED_TOOLCHAIN".into(),
            args: vec!["-m".into(), "$ZED_CUSTOM_PYTHON_MODULE_NAME".into()],
            env: Default::default(),
            cwd: Some("$ZED_WORKTREE_ROOT".into()),
            use_new_terminal: false,
            allow_concurrent_runs: false,
            reveal: task::RevealStrategy::Always,
            reveal_target: task::RevealTarget::Dock,
            hide: task::HideStrategy::Never,
            tags: vec!["python-module-main-method".into()],
            shell: task::Shell::System,
            show_summary: false,
            show_command: false,
            save: task::SaveStrategy::default(),
            hooks: Default::default(),
            artifact: None,
            artifact_argument_binding: None,
            artifact_group_argument_binding: None,
            stdout_report: None,
            literal_arguments: false,
            artifact_group: None,
        };

        let expected_scenario = DebugScenario {
            adapter: "Debugpy".into(),
            label: "run module 'main.py'".into(),
            build: None,
            config: json!({
                "request": "launch",
                "python": "$ZED_CUSTOM_PYTHON_ACTIVE_ZED_TOOLCHAIN",
                "args": [],
                "cwd": "$ZED_WORKTREE_ROOT",
                "module": "$ZED_CUSTOM_PYTHON_MODULE_NAME",
            }),
            tcp_connection: None,
        };

        assert_eq!(
            PythonLocator
                .create_scenario(&build_task, "run module 'main.py'", &adapter)
                .await
                .expect("Failed to create a scenario"),
            expected_scenario
        );
    }
}

mod memory {
    use project::debugger::{
        MemoryCell,
        memory::{MemoryIterator, PageAddress, PageContents},
    };

    #[test]
    fn iterate_over_unmapped_memory() {
        let empty_iterator = MemoryIterator::new(0..=127, Default::default());
        let actual = empty_iterator.collect::<Vec<_>>();
        let expected = vec![MemoryCell(None); 128];
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }

    #[test]
    fn iterate_over_partially_mapped_memory() {
        let it = MemoryIterator::new(
            0..=127,
            vec![(PageAddress(5), PageContents::mapped(vec![1]))].into_iter(),
        );
        let actual = it.collect::<Vec<_>>();
        let expected = std::iter::repeat_n(MemoryCell(None), 5)
            .chain(std::iter::once(MemoryCell(Some(1))))
            .chain(std::iter::repeat_n(MemoryCell(None), 122))
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }

    #[test]
    fn reads_from_the_middle_of_a_page() {
        let partial_iter = MemoryIterator::new(
            20..=30,
            vec![(PageAddress(0), PageContents::mapped((0..255).collect()))].into_iter(),
        );
        let actual = partial_iter.collect::<Vec<_>>();
        let expected = (20..=30)
            .map(|val| MemoryCell(Some(val)))
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }
}

mod session_lifecycle {
    use std::path::Path;

    use dap::adapters::{DebugAdapterBinary, DebugAdapterName};
    use fs::{FakeFs, Fs as _};
    use gpui::{Entity, TestAppContext};
    use project::{Project, debugger::session::Session};
    use serde_json::json;
    use task::SharedTaskContext;

    fn binary() -> DebugAdapterBinary {
        DebugAdapterBinary {
            command: Some("command".into()),
            arguments: Vec::new(),
            envs: Default::default(),
            cwd: None,
            connection: None,
            request_args: dap::StartDebuggingRequestArguments {
                request: dap::StartDebuggingRequestArgumentsRequest::Launch,
                configuration: json!({}),
            },
        }
    }

    fn new_session(
        project: &Entity<Project>,
        adapter: &str,
        cx: &mut TestAppContext,
    ) -> Entity<Session> {
        let store = project.read_with(cx, |project, _| project.dap_store());
        store.update(cx, |store, cx| {
            store.new_session(
                None,
                DebugAdapterName(adapter.to_string().into()),
                SharedTaskContext::default(),
                None,
                Default::default(),
                cx,
            )
        })
    }

    #[gpui::test]
    #[expect(
        clippy::result_large_err,
        reason = "The existing fake DAP callback returns the protocol ErrorResponse"
    )]
    async fn diagnostic_debug_session_records_metadata_and_denies_sensitive_requests(
        cx: &mut TestAppContext,
    ) {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        super::super::init_test(cx);
        let filesystem = FakeFs::new(cx.executor());
        filesystem
            .insert_tree(Path::new("/debug"), json!({"untouched": "original"}))
            .await;
        let project = Project::test(filesystem.clone(), [Path::new("/debug")], cx).await;
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project.visible_worktrees(cx).next().expect("worktree")
        });
        let forbidden = Arc::new(AtomicUsize::new(0));
        let _subscription = project::debugger::test::intercept_debug_sessions(cx, {
            let forbidden = forbidden.clone();
            move |client| {
                client.on_request::<dap::requests::Initialize, _>(|_, arguments| {
                    assert_eq!(arguments.supports_variable_type, Some(false));
                    assert_eq!(arguments.supports_memory_references, Some(false));
                    assert_eq!(arguments.supports_run_in_terminal_request, Some(false));
                    assert_eq!(arguments.supports_start_debugging_request, Some(false));
                    Ok(Default::default())
                });
                client.on_request::<dap::requests::Evaluate, _>({
                    let forbidden = forbidden.clone();
                    move |_, _| {
                        forbidden.fetch_add(1, Ordering::SeqCst);
                        Err(dap::ErrorResponse { error: None })
                    }
                });
                client.on_request::<dap::requests::Variables, _>({
                    let forbidden = forbidden.clone();
                    move |_, _| {
                        forbidden.fetch_add(1, Ordering::SeqCst);
                        Err(dap::ErrorResponse { error: None })
                    }
                });
                client.on_request::<dap::requests::Scopes, _>({
                    let forbidden = forbidden.clone();
                    move |_, _| {
                        forbidden.fetch_add(1, Ordering::SeqCst);
                        Err(dap::ErrorResponse { error: None })
                    }
                });
            }
        });
        let session = new_session(&project, "fake-adapter", cx);
        session
            .update(cx, |session, _| session.restrict_diagnostic_collection())
            .expect("policy");
        assert!(
            session
                .update(cx, |session, _| session.restrict_diagnostic_collection())
                .is_err()
        );
        session
            .update(cx, |session, cx| {
                session.boot(binary(), worktree, store.downgrade(), cx)
            })
            .await
            .expect("existing owner boot");
        assert!(
            session
                .update(cx, |session, _| session.restrict_diagnostic_collection())
                .is_err()
        );
        let client = session.read_with(cx, |session, _| session.adapter_client().expect("adapter"));
        assert!(
            client
                .connect(Box::new(|_| {}), &mut cx.to_async())
                .await
                .is_err()
        );
        assert!(
            client
                .request::<dap::requests::Evaluate>(dap::EvaluateArguments {
                    expression: "sensitive-direct-request".into(),
                    frame_id: None,
                    context: None,
                    format: None,
                    source: None,
                    line: None,
                    column: None,
                })
                .await
                .is_err()
        );
        assert!(
            client
                .send_message(dap::messages::Message::Request(dap::messages::Request {
                    seq: 500,
                    command: "variables".into(),
                    arguments: Some(json!({"variablesReference":1})),
                }))
                .await
                .is_err()
        );
        client
            .on_response::<dap::requests::StartDebugging, _>(|response| assert!(!response.success))
            .await;
        client
            .fake_reverse_request::<dap::requests::StartDebugging>(
                dap::StartDebuggingRequestArguments {
                    request: dap::StartDebuggingRequestArgumentsRequest::Launch,
                    configuration: json!({"program":"unapproved-child"}),
                },
            )
            .await;
        cx.run_until_parked();
        assert!(session.read_with(cx, |session, _| {
            session
                .diagnostic_observation()
                .expect("policy")
                .blocked_reverse_request
        }));
        client
            .fake_event(dap::messages::Events::Other(dap::messages::OtherEvent {
                event: "exited".into(),
                body: json!({"exitCode":"private-runtime-value"}),
            }))
            .await;
        cx.run_until_parked();
        assert!(session.read_with(cx, |session, _| {
            session
                .diagnostic_observation()
                .expect("policy")
                .exit_code
                .is_none()
        }));
        client
            .fake_event(dap::messages::Events::Output(dap::OutputEvent {
                output: "password=private-runtime-value".into(),
                category: None,
                group: None,
                variables_reference: None,
                source: None,
                line: None,
                column: None,
                data: None,
                location_reference: None,
            }))
            .await;
        for reason in [
            dap::StoppedEventReason::Pause,
            dap::StoppedEventReason::Exception,
        ] {
            client
                .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
                    reason,
                    description: Some("secret-sensitive-description".into()),
                    text: Some("secret-sensitive-text".into()),
                    thread_id: Some(1),
                    preserve_focus_hint: None,
                    all_threads_stopped: None,
                    hit_breakpoint_ids: None,
                }))
                .await;
        }
        client
            .fake_event(dap::messages::Events::Exited(dap::ExitedEvent {
                exit_code: 7,
            }))
            .await;
        cx.run_until_parked();
        let watch = session.update(cx, |session, cx| {
            assert!(session.scopes(1, cx).is_empty());
            assert!(session.variables(1, cx).is_empty());
            assert!(session.modules(cx).is_empty());
            assert!(session.loaded_sources(cx).is_empty());
            session
                .evaluate("sensitive_expression".into(), None, None, None, cx)
                .detach();
            session.add_watcher("sensitive_watch".into(), 1, cx)
        });
        assert!(watch.await.is_err());
        cx.run_until_parked();
        assert_eq!(forbidden.load(Ordering::SeqCst), 0);
        session.read_with(cx, |session, _| {
            let observation = session.diagnostic_observation().expect("observation");
            assert_eq!(observation.stop_count, 2);
            assert_eq!(observation.exception_count, 1);
            assert_eq!(observation.exit_code, Some(7));
            assert!(!observation.truncated);
            assert_eq!(
                session
                    .output(project::debugger::session::OutputToken(0))
                    .0
                    .count(),
                0
            );
            assert!(session.watchers().is_empty());
            assert!(session.historic_snapshots().is_empty());
        });
        for _ in 0..127 {
            client
                .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
                    reason: dap::StoppedEventReason::Pause,
                    description: None,
                    text: None,
                    thread_id: Some(1),
                    preserve_focus_hint: None,
                    all_threads_stopped: None,
                    hit_breakpoint_ids: None,
                }))
                .await;
        }
        cx.run_until_parked();
        session.read_with(cx, |session, _| {
            let observation = session.diagnostic_observation().expect("bounded");
            assert_eq!(observation.stop_count, 128);
            assert!(observation.truncated);
        });
        session.update(cx, |session, cx| session.shutdown(cx)).await;
        assert!(session.read_with(cx, |session, _| session.is_terminated()));
        assert_eq!(
            filesystem
                .load(Path::new("/debug/untouched"))
                .await
                .expect("preserved"),
            "original"
        );
    }

    #[gpui::test]
    async fn diagnostic_debug_session_blocks_attach_tcp_and_terminated_reconfiguration(
        cx: &mut TestAppContext,
    ) {
        super::super::init_test(cx);
        let filesystem = FakeFs::new(cx.executor());
        filesystem.insert_tree(Path::new("/debug"), json!({})).await;
        let project = Project::test(filesystem, [Path::new("/debug")], cx).await;
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project.visible_worktrees(cx).next().expect("worktree")
        });
        for attached in [false, true] {
            let session = new_session(&project, "fake-adapter", cx);
            session
                .update(cx, |session, _| session.restrict_diagnostic_collection())
                .expect("policy");
            let mut launch = binary();
            if attached {
                launch.request_args.request = dap::StartDebuggingRequestArgumentsRequest::Attach;
            } else {
                launch.connection = Some(dap::adapters::TcpArguments {
                    host: "127.0.0.1".parse().expect("loopback"),
                    port: 1,
                    timeout: None,
                });
            }
            assert!(
                session
                    .update(cx, |session, cx| session.boot(
                        launch,
                        worktree.clone(),
                        store.downgrade(),
                        cx
                    ))
                    .await
                    .is_err()
            );
            assert!(session.read_with(cx, |session, _| session.adapter_client().is_none()));
            session.update(cx, |session, cx| session.shutdown(cx)).await;
            assert!(
                session
                    .update(cx, |session, _| session.restrict_diagnostic_collection())
                    .is_err()
            );
        }
        let ordinary = new_session(&project, "fake-adapter", cx);
        assert!(ordinary.read_with(cx, |session, _| session.diagnostic_observation().is_none()));
    }

    #[gpui::test]
    async fn terminated_copper_debug_session_cannot_boot_again(cx: &mut TestAppContext) {
        super::super::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/debug"), json!({})).await;
        let project = Project::test(fs, [Path::new("/debug")], cx).await;
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("debug worktree")
        });
        let _subscription = project::debugger::test::intercept_debug_sessions(cx, |_| {});
        let session = new_session(&project, "fake-adapter", cx);
        let session_id = session.read_with(cx, |session, _| session.session_id());
        store
            .update(cx, |store, cx| store.shutdown_session(session_id, cx))
            .await
            .expect("shutdown");
        assert!(session.read_with(cx, |session, _| session.is_terminated()));
        let result = session
            .update(cx, |session, cx| {
                session.boot(binary(), worktree, store.downgrade(), cx)
            })
            .await;
        assert!(
            result
                .expect_err("terminated session must reject boot")
                .to_string()
                .contains("terminated")
        );
        assert!(session.read_with(cx, |session, _| session.adapter_client().is_none()));
    }

    #[gpui::test]
    async fn terminated_copper_debug_session_cannot_resume_pending_boot(cx: &mut TestAppContext) {
        super::super::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/debug"), json!({})).await;
        let project = Project::test(fs, [Path::new("/debug")], cx).await;
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("debug worktree")
        });
        let session = new_session(&project, "fake-adapter", cx);
        let boot = session.update(cx, |session, cx| {
            session.boot(binary(), worktree, store.downgrade(), cx)
        });
        let session_id = session.read_with(cx, |session, _| session.session_id());
        store
            .update(cx, |store, cx| store.shutdown_session(session_id, cx))
            .await
            .expect("pending shutdown");
        assert!(
            boot.await
                .expect_err("late adapter must not revive shutdown")
                .to_string()
                .contains("terminated")
        );
        assert!(session.read_with(cx, |session, _| session.is_terminated()));
        assert!(session.read_with(cx, |session, _| session.adapter_client().is_none()));
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    #[expect(
        clippy::result_large_err,
        reason = "The existing fake DAP callback returns the protocol ErrorResponse"
    )]
    async fn java_debug_trust_restriction_uses_shared_shutdown(cx: &mut TestAppContext) {
        use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        super::super::init_test(cx);
        cx.update(|cx| project::trusted_worktrees::init(Default::default(), cx));
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/debug"), json!({})).await;
        fs.insert_tree(Path::new("/other"), json!({})).await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), [Path::new("/debug")], cx).await;
        let other = Project::test_with_worktree_trust(fs, [Path::new("/other")], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let other_store = other.read_with(cx, |project, _| project.worktree_store());
        let worktree = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("debug worktree")
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let other_id = other.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("other worktree")
                .read(cx)
                .id()
        });
        let trust =
            cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("shared trust service"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                [PathTrust::Worktree(worktree_id)].into_iter().collect(),
                cx,
            )
        });
        cx.run_until_parked();
        let store = project.read_with(cx, |project, _| project.dap_store());
        let disconnects = Arc::new(AtomicUsize::new(0));
        let _subscription = project::debugger::test::intercept_debug_sessions(cx, {
            let disconnects = disconnects.clone();
            move |client| {
                let disconnects = disconnects.clone();
                client.on_request::<dap::requests::Disconnect, _>(move |_, arguments| {
                    assert_eq!(arguments.terminate_debuggee, Some(true));
                    disconnects.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                });
            }
        });
        let java = new_session(&project, "Java", cx);
        let copper = new_session(&project, "fake-adapter", cx);
        for session in [&java, &copper] {
            session
                .update(cx, |session, cx| {
                    session.boot(binary(), worktree.clone(), store.downgrade(), cx)
                })
                .await
                .expect("fake DAP boot");
        }
        let pending_java = new_session(&project, "Java", cx);
        trust.update(cx, |trust, cx| {
            trust.restrict(
                other_store.downgrade(),
                [PathTrust::Worktree(other_id)].into_iter().collect(),
                cx,
            )
        });
        cx.run_until_parked();
        assert!(!java.read_with(cx, |session, _| session.is_terminated()));
        assert!(!pending_java.read_with(cx, |session, _| session.is_terminated()));
        trust.update(cx, |trust, cx| {
            trust.restrict(
                worktree_store.downgrade(),
                [PathTrust::Worktree(worktree_id)].into_iter().collect(),
                cx,
            )
        });
        cx.run_until_parked();
        assert!(java.read_with(cx, |session, _| session.is_terminated()));
        assert!(pending_java.read_with(cx, |session, _| session.is_terminated()));
        assert!(!copper.read_with(cx, |session, _| session.is_terminated()));
        assert_eq!(disconnects.load(Ordering::SeqCst), 1);
        for session in [&java, &pending_java] {
            let id = session.read_with(cx, |session, _| session.session_id());
            assert!(store.read_with(cx, |store, _| store.session_by_id(id).is_none()));
        }
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                [PathTrust::Worktree(worktree_id)].into_iter().collect(),
                cx,
            )
        });
        let late_boot = pending_java
            .update(cx, |session, cx| {
                session.boot(binary(), worktree.clone(), store.downgrade(), cx)
            })
            .await;
        assert!(
            late_boot
                .expect_err("restored trust cannot revive a session")
                .to_string()
                .contains("terminated")
        );
        let fresh = new_session(&project, "Java", cx);
        fresh
            .update(cx, |session, cx| {
                session.boot(binary(), worktree, store.downgrade(), cx)
            })
            .await
            .expect("fresh DAP session");
        store
            .update(cx, |store, cx| store.shutdown_sessions(cx))
            .await;
    }
}

#[cfg(unix)]
mod diagnostic_debug_launch {
    use anyhow::Result;
    use collections::HashMap;
    use dap::{
        DapRegistry,
        adapters::{
            DapDelegate, DebugAdapter, DebugAdapterBinary, DebugAdapterName, DebugTaskDefinition,
        },
    };
    use fs::{FakeFs, RealFs};
    use gpui::{AsyncApp, Entity, TestAppContext, UpdateGlobal as _};
    use project::{Project, debugger::dap_store::DapStore};
    use serde_json::json;
    use settings::SettingsStore;
    use std::{
        os::unix::fs::PermissionsExt as _,
        path::{Path, PathBuf},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    use task::{DebugScenario, ZedDebugConfig};
    use worktree::Worktree;

    struct InstalledAdapter {
        opaque_calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait(?Send)]
    impl DebugAdapter for InstalledAdapter {
        fn name(&self) -> DebugAdapterName {
            "DoctorInstalledFixture".into()
        }
        fn dap_schema(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        async fn config_from_zed_format(&self, _: ZedDebugConfig) -> Result<DebugScenario> {
            anyhow::bail!("Opaque formatter must not run during preparation")
        }
        async fn get_binary(
            &self,
            _: &Arc<dyn DapDelegate>,
            _: &DebugTaskDefinition,
            _: Option<PathBuf>,
            _: Option<Vec<String>>,
            _: Option<HashMap<String, String>>,
            _: &mut AsyncApp,
        ) -> Result<DebugAdapterBinary> {
            self.opaque_calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("Opaque setup must not run during preparation")
        }
        fn diagnostic_installed_binary(
            &self,
            definition: &DebugTaskDefinition,
            root: &Path,
            installed: Option<&Path>,
            _: Option<&[String]>,
            environment: Option<&HashMap<String, String>>,
        ) -> Result<DebugAdapterBinary> {
            Ok(DebugAdapterBinary {
                command: Some(
                    installed
                        .expect("Fixture installed adapter")
                        .to_string_lossy()
                        .into(),
                ),
                arguments: Vec::new(),
                envs: environment.cloned().unwrap_or_default(),
                cwd: Some(root.into()),
                connection: None,
                request_args: dap::StartDebuggingRequestArguments {
                    request: dap::StartDebuggingRequestArgumentsRequest::Launch,
                    configuration: definition.config.clone(),
                },
            })
        }
    }

    struct Fixture {
        _directory: tempfile::TempDir,
        root: PathBuf,
        adapter: PathBuf,
        project: Entity<Project>,
        store: Entity<DapStore>,
        worktree: Entity<Worktree>,
        definition: DebugTaskDefinition,
        opaque_calls: Arc<AtomicUsize>,
    }

    fn native_file(path: &Path, suffix: &[u8]) {
        let mut bytes = b"\x7fELF".to_vec();
        bytes.extend_from_slice(suffix);
        std::fs::write(path, bytes).expect("Native identity fixture");
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .expect("Fixture executable");
    }

    async fn fixture(cx: &mut TestAppContext) -> Fixture {
        super::super::init_test(cx);
        cx.executor().allow_parking();
        let directory = tempfile::tempdir().expect("Native input fixture directory");
        let parent = directory
            .path()
            .canonicalize()
            .expect("Canonical fixture directory");
        let root = parent.join("project");
        std::fs::create_dir(&root).expect("Fixture root");
        native_file(&root.join("program"), b"native one");
        std::fs::write(root.join("untouched"), "original").expect("Unrelated file");
        let adapter = parent.join("adapter");
        native_file(&adapter, b"installed adapter");
        let opaque_calls = Arc::new(AtomicUsize::new(0));
        cx.update(|cx| {
            project::trusted_worktrees::init(Default::default(), cx);
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .session
                        .get_or_insert_with(Default::default)
                        .trust_all_worktrees = Some(true);
                    settings.project.dap.insert(
                        "DoctorInstalledFixture".into(),
                        settings::DapSettingsContent {
                            binary: Some(adapter.to_string_lossy().into()),
                            args: None,
                            env: None,
                        },
                    );
                });
            });
            DapRegistry::global(cx).add_adapter(Arc::new(InstalledAdapter {
                opaque_calls: opaque_calls.clone(),
            }));
        });
        let filesystem = FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                &root,
                json!({"program":"native fixture", "untouched":"original"}),
            )
            .await;
        let project = Project::test(filesystem, [&root as &Path], cx).await;
        let store = project.read_with(cx, |project, _| project.dap_store());
        let native_filesystem = Arc::new(RealFs::new(None, cx.executor()));
        store
            .update(cx, |store, _| {
                store.diagnostic_filesystem_for_test(native_filesystem)
            })
            .expect("Native confined input inspection");
        let worktree = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("Selected worktree")
        });
        let definition = DebugTaskDefinition {
            label: "Reviewed native inputs".into(),
            adapter: "DoctorInstalledFixture".into(),
            tcp_connection: None,
            config: json!({"request":"launch", "program":root.join("program"), "cwd":root, "args":["selected-input"]}),
        };
        Fixture {
            _directory: directory,
            root,
            adapter,
            project,
            store,
            worktree,
            definition,
            opaque_calls,
        }
    }

    #[gpui::test]
    async fn diagnostic_debug_launch_is_passive_and_rejects_changed_native_inputs(
        cx: &mut TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let prepare = |cx: &mut TestAppContext| {
            fixture.store.update(cx, |store, cx| {
                store.prepare_diagnostic_launch(
                    fixture.definition.clone(),
                    fixture.worktree.clone(),
                    cx,
                )
            })
        };
        let plan = prepare(cx).await.expect("Installed input preview");
        fixture
            .store
            .update(cx, |store, cx| {
                store.revalidate_diagnostic_launch(&plan, cx)
            })
            .await
            .expect("Current preview");
        assert_eq!(plan.binary().command.as_deref(), fixture.adapter.to_str());
        assert_eq!(
            fixture
                .store
                .read_with(cx, |store, _| store.sessions().count()),
            0
        );
        assert_eq!(fixture.opaque_calls.load(Ordering::SeqCst), 0);
        let program = fixture.root.join("program");
        let modified = std::fs::metadata(&program)
            .expect("Before metadata")
            .modified()
            .expect("Modification time");
        native_file(&program, b"native two");
        std::fs::File::open(&program)
            .expect("Same input inode")
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .expect("Preserve metadata time");
        assert!(
            fixture
                .store
                .update(cx, |store, cx| store
                    .revalidate_diagnostic_launch(&plan, cx))
                .await
                .is_err()
        );
        let next = prepare(cx).await.expect("Fresh changed input preview");
        native_file(&fixture.adapter, b"replaced adapter");
        assert!(
            fixture
                .store
                .update(cx, |store, cx| store
                    .revalidate_diagnostic_launch(&next, cx))
                .await
                .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("untouched")).expect("Unrelated file"),
            "original"
        );
        assert_eq!(fixture.opaque_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            fixture
                .store
                .read_with(cx, |store, _| store.sessions().count()),
            0
        );
    }

    #[gpui::test]
    async fn diagnostic_debug_launch_rejects_alias_private_and_unsafe_authority(
        cx: &mut TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let alias = fixture.root.join("alias");
        std::os::unix::fs::symlink(fixture.root.join("program"), &alias).expect("Alias fixture");
        let private = fixture.root.join(".env");
        native_file(&private, b"excluded private input");
        for value in [
            json!(alias),
            json!(private),
            json!(fixture.adapter),
            json!("${ZED_WORKTREE_ROOT}/program"),
        ] {
            let mut definition = fixture.definition.clone();
            definition.config["program"] = value;
            assert!(
                fixture
                    .store
                    .update(cx, |store, cx| store.prepare_diagnostic_launch(
                        definition,
                        fixture.worktree.clone(),
                        cx
                    ))
                    .await
                    .is_err()
            );
        }
        for key in ["PASSWORD", "LD_PRELOAD", "BASH_ENV", "INVALID=KEY"] {
            let mut definition = fixture.definition.clone();
            definition.config["env"] = json!({key:"not-retained"});
            assert!(
                fixture
                    .store
                    .update(cx, |store, cx| store.prepare_diagnostic_launch(
                        definition,
                        fixture.worktree.clone(),
                        cx
                    ))
                    .await
                    .is_err(),
                "Environment {key}"
            );
        }
        let plan = fixture
            .store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(
                    fixture.definition.clone(),
                    fixture.worktree.clone(),
                    cx,
                )
            })
            .await
            .expect("Reviewed input");
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .project
                        .dap
                        .get_mut("DoctorInstalledFixture")
                        .expect("Existing adapter settings")
                        .args = Some(Vec::new());
                })
            })
        });
        assert!(
            fixture
                .store
                .update(cx, |store, cx| store
                    .revalidate_diagnostic_launch(&plan, cx))
                .await
                .is_err()
        );
        let plan = fixture
            .store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(
                    fixture.definition.clone(),
                    fixture.worktree.clone(),
                    cx,
                )
            })
            .await
            .expect("Fresh settings preview");
        cx.update(|cx| {
            DapRegistry::global(cx).add_adapter(Arc::new(InstalledAdapter {
                opaque_calls: fixture.opaque_calls.clone(),
            }));
        });
        assert!(
            fixture
                .store
                .update(cx, |store, cx| store
                    .revalidate_diagnostic_launch(&plan, cx))
                .await
                .is_err()
        );
        let next = fixture
            .store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(
                    fixture.definition.clone(),
                    fixture.worktree.clone(),
                    cx,
                )
            })
            .await
            .expect("New installed identity");
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .session
                        .get_or_insert_with(Default::default)
                        .trust_all_worktrees = Some(false);
                })
            })
        });
        assert!(
            fixture
                .store
                .update(cx, |store, cx| store
                    .revalidate_diagnostic_launch(&next, cx))
                .await
                .is_err()
        );
        assert_eq!(fixture.opaque_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            fixture
                .store
                .read_with(cx, |store, _| store.sessions().count()),
            0
        );
        assert_eq!(
            fixture
                .project
                .read_with(cx, |project, _| project.dap_store()),
            fixture.store
        );
    }
    #[gpui::test]
    async fn diagnostic_debug_launch_rejects_other_project_and_missing_capability(
        cx: &mut TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let plan = fixture
            .store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(
                    fixture.definition.clone(),
                    fixture.worktree.clone(),
                    cx,
                )
            })
            .await
            .expect("Selected owner preview");
        let filesystem = FakeFs::new(cx.executor());
        filesystem
            .insert_tree(&fixture.root, json!({"program":"other project"}))
            .await;
        let other_project = Project::test(filesystem, [&fixture.root as &Path], cx).await;
        let other_store = other_project.read_with(cx, |project, _| project.dap_store());
        assert!(
            other_store
                .update(cx, |store, cx| store
                    .revalidate_diagnostic_launch(&plan, cx))
                .await
                .is_err()
        );
        cx.update(|cx| DapRegistry::global(cx).add_adapter(Arc::new(dap::FakeAdapter::new())));
        for adapter in ["MissingDiagnosticAdapter", dap::FakeAdapter::ADAPTER_NAME] {
            let mut definition = fixture.definition.clone();
            definition.adapter = adapter.into();
            assert!(
                fixture
                    .store
                    .update(cx, |store, cx| store.prepare_diagnostic_launch(
                        definition,
                        fixture.worktree.clone(),
                        cx
                    ))
                    .await
                    .is_err()
            );
        }
        let mut attached = fixture.definition.clone();
        attached.config["request"] = json!("attach");
        assert!(
            fixture
                .store
                .update(cx, |store, cx| store.prepare_diagnostic_launch(
                    attached,
                    fixture.worktree.clone(),
                    cx
                ))
                .await
                .is_err()
        );
        assert_eq!(fixture.opaque_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            fixture
                .store
                .read_with(cx, |store, _| store.sessions().count()),
            0
        );
        assert_eq!(
            other_store.read_with(cx, |store, _| store.sessions().count()),
            0
        );
    }
}
