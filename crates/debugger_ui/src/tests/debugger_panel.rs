#![expect(clippy::result_large_err)]
use crate::{
    persistence::DebuggerPaneItem,
    tests::{start_debug_session, start_debug_session_with},
    *,
};
use dap::{
    ErrorResponse, Message, RunInTerminalRequestArguments, SourceBreakpoint,
    StartDebuggingRequestArguments, StartDebuggingRequestArgumentsRequest,
    adapters::DebugTaskDefinition,
    client::SessionId,
    requests::{
        Continue, Disconnect, Launch, Next, RunInTerminal, SetBreakpoints, StackTrace,
        StartDebugging, StepBack, StepIn, StepOut, Threads,
    },
};
use editor::{
    ActiveDebugLine, Editor, EditorMode, MultiBuffer,
    actions::{self},
};
use gpui::{BackgroundExecutor, Focusable, TestAppContext, VisualTestContext};
use project::{
    FakeFs, Project,
    debugger::session::{ThreadId, ThreadStatus},
};
use serde_json::json;
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use terminal_view::terminal_panel::TerminalPanel;
use tests::{active_debug_session_panel, init_test, init_test_workspace};
use util::{path, rel_path::rel_path};
use workspace::item::SaveOptions;
use workspace::pane_group::SplitDirection;
use workspace::{Item, dock::Panel, move_active_item};

#[gpui::test]
async fn test_basic_show_debug_panel(executor: BackgroundExecutor, cx: &mut TestAppContext) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    client.on_request::<Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: 1,
                name: "Thread 1".into(),
            }],
        })
    });

    client.on_request::<StackTrace, _>(move |_, _| {
        Ok(dap::StackTraceResponse {
            stack_frames: Vec::default(),
            total_frames: None,
        })
    });

    cx.run_until_parked();

    // assert we have a debug panel item before the session has stopped
    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();
            let active_session =
                debug_panel.update(cx, |debug_panel, _| debug_panel.active_session().unwrap());

            let running_state = active_session.update(cx, |active_session, _| {
                active_session.running_state().clone()
            });

            debug_panel.update(cx, |this, cx| {
                assert!(this.active_session().is_some());
                assert!(running_state.read(cx).selected_thread_id().is_none());
            });
        })
        .unwrap();

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Pause,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();
            let active_session = debug_panel
                .update(cx, |this, _| this.active_session())
                .unwrap();

            let running_state = active_session.update(cx, |active_session, _| {
                active_session.running_state().clone()
            });

            assert_eq!(client.id(), running_state.read(cx).session_id());
            assert_eq!(
                ThreadId(1),
                running_state.read(cx).selected_thread_id().unwrap()
            );
        })
        .unwrap();

    let shutdown_session = project.update(cx, |project, cx| {
        project.dap_store().update(cx, |dap_store, cx| {
            dap_store.shutdown_session(session.read(cx).session_id(), cx)
        })
    });

    shutdown_session.await.unwrap();

    // assert we still have a debug panel item after the client shutdown
    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();

            let active_session = debug_panel
                .update(cx, |this, _| this.active_session())
                .unwrap();

            let running_state = active_session.update(cx, |active_session, _| {
                active_session.running_state().clone()
            });

            debug_panel.update(cx, |this, cx| {
                assert!(this.active_session().is_some());
                assert_eq!(
                    ThreadId(1),
                    running_state.read(cx).selected_thread_id().unwrap()
                );
            });
        })
        .unwrap();
}

#[gpui::test]
async fn test_debug_toolbar_controls_are_keyboard_reachable(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());
    fs.insert_tree(path!("/project"), json!({ "main.rs": "fn main() {}" }))
        .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());
    client.on_request::<Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: 1,
                name: "Main".into(),
            }],
        })
    });
    client.on_request::<StackTrace, _>(move |_, _| {
        Ok(dap::StackTraceResponse {
            stack_frames: Vec::default(),
            total_frames: None,
        })
    });
    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Pause,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;
    cx.run_until_parked();

    let panel_focus = workspace
        .update(cx, |workspace, _, cx| {
            workspace.panel::<DebugPanel>(cx).unwrap().focus_handle(cx)
        })
        .unwrap();
    cx.dispatch_action(crate::FocusToolbar);
    let toolbar_focus = cx.update(|window, cx| window.focused(cx).unwrap());
    assert_ne!(toolbar_focus, panel_focus);
    assert!(cx.update(|window, _| panel_focus.contains(&toolbar_focus, window)));
    cx.update(|window, cx| window.simulate_next_frame(cx));

    cx.dispatch_action(menu::SelectNext);
    let first_control = cx.update(|window, cx| window.focused(cx).unwrap());
    assert_ne!(first_control, toolbar_focus);
    assert!(cx.update(|window, _| panel_focus.contains(&first_control, window)));

    cx.update(|window, cx| window.simulate_next_frame(cx));
    cx.dispatch_action(menu::SelectNext);
    let second_control = cx.update(|window, cx| window.focused(cx).unwrap());
    assert_ne!(second_control, first_control);
    assert!(cx.update(|window, _| panel_focus.contains(&second_control, window)));

    cx.update(|window, cx| window.simulate_next_frame(cx));
    cx.dispatch_action(menu::SelectPrevious);
    assert_eq!(
        cx.update(|window, cx| window.focused(cx)),
        Some(first_control)
    );
}

#[gpui::test]
async fn test_we_can_only_have_one_panel_per_debug_session(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    client.on_request::<Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: 1,
                name: "Thread 1".into(),
            }],
        })
    });

    client.on_request::<StackTrace, _>(move |_, _| {
        Ok(dap::StackTraceResponse {
            stack_frames: Vec::default(),
            total_frames: None,
        })
    });

    cx.run_until_parked();

    // assert we have a debug panel item before the session has stopped
    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();

            debug_panel.update(cx, |this, _| {
                assert!(this.active_session().is_some());
            });
        })
        .unwrap();

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Pause,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    // assert we added a debug panel item
    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();
            let active_session = debug_panel
                .update(cx, |this, _| this.active_session())
                .unwrap();

            let running_state = active_session.update(cx, |active_session, _| {
                active_session.running_state().clone()
            });

            assert_eq!(client.id(), active_session.read(cx).session_id(cx));
            assert_eq!(
                ThreadId(1),
                running_state.read(cx).selected_thread_id().unwrap()
            );
        })
        .unwrap();

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Pause,
            description: None,
            thread_id: Some(2),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();
            let active_session = debug_panel
                .update(cx, |this, _| this.active_session())
                .unwrap();

            let running_state = active_session.update(cx, |active_session, _| {
                active_session.running_state().clone()
            });

            assert_eq!(client.id(), active_session.read(cx).session_id(cx));
            assert_eq!(
                ThreadId(1),
                running_state.read(cx).selected_thread_id().unwrap()
            );
        })
        .unwrap();

    let shutdown_session = project.update(cx, |project, cx| {
        project.dap_store().update(cx, |dap_store, cx| {
            dap_store.shutdown_session(session.read(cx).session_id(), cx)
        })
    });

    shutdown_session.await.unwrap();

    // assert we still have a debug panel item after the client shutdown
    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();
            let active_session = debug_panel
                .update(cx, |this, _| this.active_session())
                .unwrap();

            let running_state = active_session.update(cx, |active_session, _| {
                active_session.running_state().clone()
            });

            debug_panel.update(cx, |this, cx| {
                assert!(this.active_session().is_some());
                assert_eq!(
                    ThreadId(1),
                    running_state.read(cx).selected_thread_id().unwrap()
                );
            });
        })
        .unwrap();
}

#[gpui::test]
async fn test_handle_successful_run_in_terminal_reverse_request(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    // needed because the debugger launches a terminal which starts a background PTY
    cx.executor().allow_parking();
    init_test(cx);

    let send_response = Arc::new(AtomicBool::new(false));

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    client
        .on_response::<RunInTerminal, _>({
            let send_response = send_response.clone();
            move |response| {
                send_response.store(true, Ordering::SeqCst);

                assert!(response.success);
                assert!(response.body.is_some());
            }
        })
        .await;

    client
        .fake_reverse_request::<RunInTerminal>(RunInTerminalRequestArguments {
            kind: None,
            title: None,
            cwd: std::env::temp_dir().to_string_lossy().into_owned(),
            args: vec![],
            env: None,
            args_can_be_interpreted_by_shell: None,
        })
        .await;

    cx.run_until_parked();

    assert!(
        send_response.load(std::sync::atomic::Ordering::SeqCst),
        "Expected to receive response from reverse request"
    );

    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();
            let session = debug_panel.read(cx).active_session().unwrap();
            let running = session.read(cx).running_state();
            assert_eq!(
                running
                    .read(cx)
                    .pane_items_status(cx)
                    .get(&DebuggerPaneItem::Terminal),
                Some(&true)
            );
            assert!(running.read(cx).debug_terminal.read(cx).terminal.is_some());
        })
        .unwrap();
}

#[cfg(unix)]
#[gpui::test]
async fn test_run_in_terminal_direct_receiver_pty_and_headless(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    use feature_flags::FeatureFlagAppExt as _;
    use futures::StreamExt as _;
    use gpui::UpdateGlobal as _;
    use std::{os::unix::fs::PermissionsExt, time::Duration};

    init_test(cx);
    cx.executor().allow_parking();
    cx.update(|cx| {
        settings::SettingsStore::update_global(cx, |settings, _| {
            settings.register_setting::<feature_flags::FeatureFlagsSettings>();
        });
        feature_flags::FeatureFlagStore::init(cx);
        settings::SettingsStore::update_global(cx, |settings, cx| {
            settings
                .set_user_settings(
                    r#"{
                        "terminal": {"detect_venv": "off"},
                        "feature_flags": {
                            "rust-coverage-experimental": "off",
                            "rust-native-profiling-experimental": "off",
                            "rust-named-cargo-presets-experimental": "off",
                            "rust-exact-doctest-actions-experimental": "off"
                        }
                    }"#,
                    cx,
                )
                .result()
                .expect("Disable activation and experimental Rust workflows");
        });
        cx.set_staff(true);
        for flag in [
            "rust-coverage-experimental",
            "rust-native-profiling-experimental",
            "rust-named-cargo-presets-experimental",
            "rust-exact-doctest-actions-experimental",
        ] {
            assert!(!cx.has_flag_by_name(flag));
        }
    });

    let directory = util::test::TempTree::new(json!({"main.rs": "fn main() {}"}));
    let receiver = directory
        .path()
        .join("debug receiver $DAP_EXPAND ; literal");
    std::fs::write(
        &receiver,
        r#"#!/bin/sh
{
    printf '%s\000' "$#" "$PWD" "$DAP_LITERAL_VALUE" "$$"
    if [ "$#" -gt 0 ]; then
        printf '%s\000' "$@"
    fi
} > "$DAP_RECEIPT"
exit "$DAP_EXIT"
"#,
    )
    .expect("Write actual DAP receiver");
    std::fs::set_permissions(&receiver, std::fs::Permissions::from_mode(0o700))
        .expect("Make the owned receiver executable");
    let project = Project::test(
        Arc::new(project::RealFs::new(None, cx.executor())),
        [directory.path()],
        cx,
    )
    .await;
    let workspace = init_test_workspace(&project, cx).await;
    let session = start_debug_session(&workspace, cx, |_| {}).expect("Ordinary DAP session");
    assert!(session.read_with(cx, |session, _| {
        session.required_experimental_features().is_empty()
    }));
    let client = session
        .update(cx, |session, _| session.adapter_client())
        .expect("Registered fake adapter transport");
    let (responses, mut received_responses) = futures::channel::mpsc::unbounded();
    client
        .on_response::<RunInTerminal, _>(move |response| {
            responses
                .unbounded_send(response)
                .expect("Return the actual reverse-request response");
        })
        .await;
    let running = active_debug_session_panel(workspace, cx)
        .read_with(cx, |session, _| session.running_state().clone());
    let unintended = directory.path().join("unintended-command");
    let operands = vec![
        String::new(),
        "two words".into(),
        "$DAP_EXPAND".into(),
        format!("; printf unintended > {}", unintended.display()),
        "single' and double\" quotes".into(),
    ];
    let literal_value = "literal env $value; preserved";
    let mut cases = 0;

    for headless in [false, true] {
        cx.update(|cx| cx.set_global(terminal::HeadlessTerminal(headless)));
        for (label, arguments, exit_code) in [
            ("zero", Vec::new(), 0),
            ("literal", operands.clone(), 0),
            ("nonzero", operands.clone(), 7),
        ] {
            let receipt = directory.path().join(format!("{headless}-{label}.receipt"));
            let mut request_arguments = vec![receiver.to_string_lossy().into_owned()];
            request_arguments.extend(arguments.clone());
            client
                .fake_reverse_request::<RunInTerminal>(RunInTerminalRequestArguments {
                    kind: None,
                    title: Some(format!("Actual DAP receiver {headless}-{label}")),
                    cwd: directory.path().to_string_lossy().into_owned(),
                    args: request_arguments,
                    env: Some(json!({
                        "DAP_RECEIPT": receipt,
                        "DAP_LITERAL_VALUE": literal_value,
                        "DAP_EXPAND": "must-not-expand",
                        "DAP_EXIT": exit_code.to_string(),
                    })),
                    args_can_be_interpreted_by_shell: None,
                })
                .await;
            let response = {
                let response = received_responses.next();
                let deadline = executor.timer(Duration::from_secs(30));
                futures::pin_mut!(response, deadline);
                match futures::future::select(response, deadline).await {
                    futures::future::Either::Left((response, _)) => {
                        response.expect("Actual DAP reverse response")
                    }
                    futures::future::Either::Right(_) => {
                        let terminals = project
                            .read_with(cx, |project, _| project.local_terminal_handles().clone())
                            .into_iter()
                            .filter_map(|terminal| terminal.upgrade())
                            .collect::<Vec<_>>();
                        for terminal in &terminals {
                            terminal.update(cx, |terminal, _| terminal.kill_active_task());
                        }
                        let completions = terminals
                            .iter()
                            .map(|terminal| {
                                terminal.read_with(cx, |terminal, cx| {
                                    terminal.wait_for_completed_task(cx)
                                })
                            })
                            .collect::<Vec<_>>();
                        let cleanup = futures::future::join_all(completions);
                        let cleanup_deadline = executor.timer(Duration::from_secs(10));
                        let cleanup_result = match futures::future::select(
                            Box::pin(cleanup),
                            Box::pin(cleanup_deadline),
                        )
                        .await
                        {
                            futures::future::Either::Left((statuses, _)) => {
                                if terminals.is_empty() {
                                    "no registered Terminal owner; pending creation remains unverified".into()
                                } else {
                                    format!(
                                        "canonical wait outcomes for {} registered Terminal owners: {statuses:?}",
                                        terminals.len()
                                    )
                                }
                            }
                            futures::future::Either::Right(_) => {
                                format!(
                                    "cleanup deadline exceeded for {} registered Terminal owners",
                                    terminals.len()
                                )
                            }
                        };
                        panic!(
                            "Actual DAP receiver response timed out ({headless}-{label}); canonical cleanup: {cleanup_result}"
                        );
                    }
                }
            };
            let terminal = running.read_with(cx, |running, cx| {
                running
                    .debug_terminal
                    .read(cx)
                    .terminal
                    .as_ref()
                    .expect("Actual DAP terminal view")
                    .read(cx)
                    .terminal()
                    .clone()
            });
            let process_id = terminal.read_with(cx, |terminal, _| {
                assert_eq!(terminal.is_pty(), !headless);
                if headless {
                    assert!(terminal.pid().is_none());
                    assert!(terminal.pid_getter().is_none());
                } else {
                    assert_eq!(terminal.task_process_id(), terminal.pid());
                }
                terminal
                    .task_process_id()
                    .expect("Original Task process identifier")
                    .as_u32()
            });
            let completion =
                terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
            let deadline = executor.timer(Duration::from_secs(30));
            let status = match futures::future::select(Box::pin(completion), Box::pin(deadline))
                .await
            {
                futures::future::Either::Left((status, _)) => {
                    status.expect("Actual kernel Task completion")
                }
                futures::future::Either::Right(_) => {
                    terminal.update(cx, |terminal, _| terminal.kill_active_task());
                    let cleanup =
                        terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                    let deadline = executor.timer(Duration::from_secs(10));
                    let cleanup_result = match futures::future::select(
                        Box::pin(cleanup),
                        Box::pin(deadline),
                    )
                    .await
                    {
                        futures::future::Either::Left((status, _)) => {
                            format!("owned Task completion: {status:?}")
                        }
                        futures::future::Either::Right(_) => "cleanup deadline exceeded".into(),
                    };
                    panic!(
                        "Actual DAP receiver completion timed out ({headless}-{label}); canonical cleanup: {cleanup_result}"
                    );
                }
            };
            assert!(
                response.success,
                "Actual DAP response failed ({headless}-{label}); canonical Task completion: {status:?}"
            );
            let response: dap::RunInTerminalResponse =
                serde_json::from_value(response.body.expect("Actual terminal response body"))
                    .expect("Typed DAP process response");
            assert_eq!(
                response.process_id,
                headless.then_some(u64::from(process_id))
            );
            assert_eq!(
                response.shell_process_id,
                (!headless).then_some(u64::from(process_id))
            );
            assert_eq!(status.code(), Some(exit_code));
            assert!(std::fs::metadata(&receipt).expect("Actual receipt").len() <= 8192);
            let actual = std::fs::read(&receipt).expect("Read the actual receiver receipt");
            let mut expected = Vec::new();
            for value in [
                arguments.len().to_string(),
                directory.path().to_string_lossy().into_owned(),
                literal_value.into(),
                process_id.to_string(),
            ]
            .into_iter()
            .chain(arguments)
            {
                expected.extend_from_slice(value.as_bytes());
                expected.push(0);
            }
            assert_eq!(actual, expected);
            assert!(!unintended.exists());
            let late_completion = terminal
                .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                .await;
            assert_eq!(late_completion, Some(status));
            cx.run_until_parked();
            terminal.read_with(cx, |terminal, _| {
                assert_eq!(
                    terminal.task().expect("Original Task").status,
                    terminal::TaskStatus::Completed {
                        success: exit_code == 0,
                    }
                );
            });
            running.update(cx, |running, cx| {
                running.debug_terminal.update(cx, |terminal, cx| {
                    drop(terminal.terminal.take());
                    cx.notify();
                });
            });
            drop(terminal);
            cx.run_until_parked();
            cases += 1;
        }
    }
    assert_eq!(cases, 6);
    assert!(!unintended.exists());
    session.update(cx, |session, cx| session.shutdown(cx)).await;
    cx.run_until_parked();
}

#[gpui::test]
async fn test_handle_start_debugging_request(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    let fake_config = json!({"one": "two"});
    let launched_with = Arc::new(parking_lot::Mutex::new(None));

    let _subscription = project::debugger::test::intercept_debug_sessions(cx, {
        let launched_with = launched_with.clone();
        move |client| {
            let launched_with = launched_with.clone();
            client.on_request::<dap::requests::Launch, _>(move |_, args| {
                launched_with.lock().replace(args.raw);
                Ok(())
            });
            client.on_request::<dap::requests::Attach, _>(move |_, _| {
                assert!(false, "should not get attach request");
                Ok(())
            });
        }
    });

    let sessions = workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();
            debug_panel.read(cx).sessions().collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(sessions.len(), 1);
    client
        .fake_reverse_request::<StartDebugging>(StartDebuggingRequestArguments {
            request: StartDebuggingRequestArgumentsRequest::Launch,
            configuration: fake_config.clone(),
        })
        .await;

    cx.run_until_parked();

    workspace
        .update(cx, |workspace, _window, cx| {
            let debug_panel = workspace.panel::<DebugPanel>(cx).unwrap();

            // Active session changes on spawn, as the parent has never stopped.
            let active_session = debug_panel
                .read(cx)
                .active_session()
                .unwrap()
                .read(cx)
                .session(cx);
            let current_sessions = debug_panel.read(cx).sessions().collect::<Vec<_>>();
            assert_eq!(active_session, current_sessions[1].read(cx).session(cx));
            assert_eq!(
                active_session.read(cx).parent_session(),
                Some(&current_sessions[0].read(cx).session(cx))
            );

            assert_eq!(current_sessions.len(), 2);
            assert_eq!(current_sessions[0], sessions[0]);

            let parent_session = current_sessions[1]
                .read(cx)
                .session(cx)
                .read(cx)
                .parent_session()
                .unwrap();
            assert_eq!(parent_session, &sessions[0].read(cx).session(cx));

            // We should preserve the original binary (params to spawn process etc.) except for launch params
            // (as they come from reverse spawn request).
            let mut original_binary = parent_session.read(cx).binary().cloned().unwrap();
            original_binary.request_args = StartDebuggingRequestArguments {
                request: StartDebuggingRequestArgumentsRequest::Launch,
                configuration: fake_config.clone(),
            };

            assert_eq!(
                current_sessions[1]
                    .read(cx)
                    .session(cx)
                    .read(cx)
                    .binary()
                    .unwrap(),
                &original_binary
            );
        })
        .unwrap();

    assert_eq!(&fake_config, launched_with.lock().as_ref().unwrap());
}

// // covers that we always send a response back, if something when wrong,
// // while spawning the terminal
#[gpui::test]
async fn test_handle_error_run_in_terminal_reverse_request(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let send_response = Arc::new(AtomicBool::new(false));

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    client
        .on_response::<RunInTerminal, _>({
            let send_response = send_response.clone();
            move |response| {
                send_response.store(true, Ordering::SeqCst);

                assert!(!response.success);
                assert!(response.body.is_some());
            }
        })
        .await;

    client
        .fake_reverse_request::<RunInTerminal>(RunInTerminalRequestArguments {
            kind: None,
            title: None,
            cwd: "".into(),
            args: vec!["oops".into(), "oops".into()],
            env: None,
            args_can_be_interpreted_by_shell: None,
        })
        .await;

    cx.run_until_parked();

    assert!(
        send_response.load(std::sync::atomic::Ordering::SeqCst),
        "Expected to receive response from reverse request"
    );

    workspace
        .update(cx, |workspace, _window, cx| {
            let terminal_panel = workspace.panel::<TerminalPanel>(cx).unwrap();

            assert_eq!(
                0,
                terminal_panel.read(cx).pane().unwrap().read(cx).items_len()
            );
        })
        .unwrap();
}

#[gpui::test]
async fn test_handle_start_debugging_reverse_request(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    cx.executor().allow_parking();
    init_test(cx);

    let send_response = Arc::new(AtomicBool::new(false));

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    client.on_request::<dap::requests::Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: 1,
                name: "Thread 1".into(),
            }],
        })
    });

    client
        .on_response::<StartDebugging, _>({
            let send_response = send_response.clone();
            move |response| {
                send_response.store(true, Ordering::SeqCst);

                assert!(response.success);
                assert!(response.body.is_some());
            }
        })
        .await;
    // Set up handlers for sessions spawned with reverse request too.
    let _reverse_request_subscription =
        project::debugger::test::intercept_debug_sessions(cx, |_| {});
    client
        .fake_reverse_request::<StartDebugging>(StartDebuggingRequestArguments {
            configuration: json!({}),
            request: StartDebuggingRequestArgumentsRequest::Launch,
        })
        .await;

    cx.run_until_parked();

    let child_session = project.update(cx, |project, cx| {
        project
            .dap_store()
            .read(cx)
            .session_by_id(SessionId(1))
            .unwrap()
    });
    let child_client = child_session.update(cx, |session, _| session.adapter_client().unwrap());

    child_client.on_request::<dap::requests::Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: 1,
                name: "Thread 1".into(),
            }],
        })
    });

    child_client.on_request::<Disconnect, _>(move |_, _| Ok(()));

    child_client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Pause,
            description: None,
            thread_id: Some(2),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    assert!(
        send_response.load(std::sync::atomic::Ordering::SeqCst),
        "Expected to receive response from reverse request"
    );
}

#[gpui::test]
async fn test_shutdown_children_when_parent_session_shutdown(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let dap_store = project.update(cx, |project, _| project.dap_store());
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let parent_session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = parent_session.update(cx, |session, _| session.adapter_client().unwrap());

    client.on_request::<dap::requests::Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: 1,
                name: "Thread 1".into(),
            }],
        })
    });

    client.on_response::<StartDebugging, _>(move |_| {}).await;
    // Set up handlers for sessions spawned with reverse request too.
    let _reverse_request_subscription =
        project::debugger::test::intercept_debug_sessions(cx, |_| {});
    // start first child session
    client
        .fake_reverse_request::<StartDebugging>(StartDebuggingRequestArguments {
            configuration: json!({}),
            request: StartDebuggingRequestArgumentsRequest::Launch,
        })
        .await;

    cx.run_until_parked();

    // start second child session
    client
        .fake_reverse_request::<StartDebugging>(StartDebuggingRequestArguments {
            configuration: json!({}),
            request: StartDebuggingRequestArgumentsRequest::Launch,
        })
        .await;

    cx.run_until_parked();

    // configure first child session
    let first_child_session = dap_store.read_with(cx, |dap_store, _| {
        dap_store.session_by_id(SessionId(1)).unwrap()
    });
    let first_child_client =
        first_child_session.update(cx, |session, _| session.adapter_client().unwrap());

    first_child_client.on_request::<Disconnect, _>(move |_, _| Ok(()));

    // configure second child session
    let second_child_session = dap_store.read_with(cx, |dap_store, _| {
        dap_store.session_by_id(SessionId(2)).unwrap()
    });
    let second_child_client =
        second_child_session.update(cx, |session, _| session.adapter_client().unwrap());

    second_child_client.on_request::<Disconnect, _>(move |_, _| Ok(()));

    cx.run_until_parked();

    // shutdown parent session
    dap_store
        .update(cx, |dap_store, cx| {
            dap_store.shutdown_session(parent_session.read(cx).session_id(), cx)
        })
        .await
        .unwrap();

    // assert parent session and all children sessions are shutdown
    dap_store.update(cx, |dap_store, cx| {
        assert!(
            dap_store
                .session_by_id(parent_session.read(cx).session_id())
                .is_none()
        );
        assert!(
            dap_store
                .session_by_id(first_child_session.read(cx).session_id())
                .is_none()
        );
        assert!(
            dap_store
                .session_by_id(second_child_session.read(cx).session_id())
                .is_none()
        );
    });
}

#[gpui::test]
async fn test_shutdown_parent_session_if_all_children_are_shutdown(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let dap_store = project.update(cx, |project, _| project.dap_store());
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let parent_session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = parent_session.update(cx, |session, _| session.adapter_client().unwrap());

    client.on_response::<StartDebugging, _>(move |_| {}).await;
    // Set up handlers for sessions spawned with reverse request too.
    let _reverse_request_subscription =
        project::debugger::test::intercept_debug_sessions(cx, |_| {});
    // start first child session
    client
        .fake_reverse_request::<StartDebugging>(StartDebuggingRequestArguments {
            configuration: json!({}),
            request: StartDebuggingRequestArgumentsRequest::Launch,
        })
        .await;

    cx.run_until_parked();

    // start second child session
    client
        .fake_reverse_request::<StartDebugging>(StartDebuggingRequestArguments {
            configuration: json!({}),
            request: StartDebuggingRequestArgumentsRequest::Launch,
        })
        .await;

    cx.run_until_parked();

    // configure first child session
    let first_child_session = dap_store.read_with(cx, |dap_store, _| {
        dap_store.session_by_id(SessionId(1)).unwrap()
    });
    let first_child_client =
        first_child_session.update(cx, |session, _| session.adapter_client().unwrap());

    first_child_client.on_request::<Disconnect, _>(move |_, _| Ok(()));

    // configure second child session
    let second_child_session = dap_store.read_with(cx, |dap_store, _| {
        dap_store.session_by_id(SessionId(2)).unwrap()
    });
    let second_child_client =
        second_child_session.update(cx, |session, _| session.adapter_client().unwrap());

    second_child_client.on_request::<Disconnect, _>(move |_, _| Ok(()));

    cx.run_until_parked();

    // shutdown first child session
    dap_store
        .update(cx, |dap_store, cx| {
            dap_store.shutdown_session(first_child_session.read(cx).session_id(), cx)
        })
        .await
        .unwrap();

    // assert parent session and second child session still exist
    dap_store.update(cx, |dap_store, cx| {
        assert!(
            dap_store
                .session_by_id(parent_session.read(cx).session_id())
                .is_some()
        );
        assert!(
            dap_store
                .session_by_id(first_child_session.read(cx).session_id())
                .is_none()
        );
        assert!(
            dap_store
                .session_by_id(second_child_session.read(cx).session_id())
                .is_some()
        );
    });

    // shutdown first child session
    dap_store
        .update(cx, |dap_store, cx| {
            dap_store.shutdown_session(second_child_session.read(cx).session_id(), cx)
        })
        .await
        .unwrap();

    // assert parent session got shutdown by second child session
    // because it was the last child
    dap_store.update(cx, |dap_store, cx| {
        assert!(
            dap_store
                .session_by_id(parent_session.read(cx).session_id())
                .is_none()
        );
        assert!(
            dap_store
                .session_by_id(second_child_session.read(cx).session_id())
                .is_none()
        );
    });
}

#[gpui::test]
async fn test_debug_panel_item_thread_status_reset_on_failure(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |client| {
        client.on_request::<dap::requests::Initialize, _>(move |_, _| {
            Ok(dap::Capabilities {
                supports_step_back: Some(true),
                ..Default::default()
            })
        });
    })
    .unwrap();

    let client = session.update(cx, |session, _| session.adapter_client().unwrap());
    const THREAD_ID_NUM: i64 = 1;

    client.on_request::<dap::requests::Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: THREAD_ID_NUM,
                name: "Thread 1".into(),
            }],
        })
    });

    client.on_request::<Launch, _>(move |_, _| Ok(()));

    client.on_request::<StackTrace, _>(move |_, _| {
        Ok(dap::StackTraceResponse {
            stack_frames: Vec::default(),
            total_frames: None,
        })
    });

    client.on_request::<Next, _>(move |_, _| {
        Err(ErrorResponse {
            error: Some(dap::Message {
                id: 1,
                format: "error".into(),
                variables: None,
                send_telemetry: None,
                show_user: None,
                url: None,
                url_label: None,
            }),
        })
    });

    client.on_request::<StepOut, _>(move |_, _| {
        Err(ErrorResponse {
            error: Some(dap::Message {
                id: 1,
                format: "error".into(),
                variables: None,
                send_telemetry: None,
                show_user: None,
                url: None,
                url_label: None,
            }),
        })
    });

    client.on_request::<StepIn, _>(move |_, _| {
        Err(ErrorResponse {
            error: Some(dap::Message {
                id: 1,
                format: "error".into(),
                variables: None,
                send_telemetry: None,
                show_user: None,
                url: None,
                url_label: None,
            }),
        })
    });

    client.on_request::<StepBack, _>(move |_, _| {
        Err(ErrorResponse {
            error: Some(dap::Message {
                id: 1,
                format: "error".into(),
                variables: None,
                send_telemetry: None,
                show_user: None,
                url: None,
                url_label: None,
            }),
        })
    });

    client.on_request::<Continue, _>(move |_, _| {
        Err(ErrorResponse {
            error: Some(dap::Message {
                id: 1,
                format: "error".into(),
                variables: None,
                send_telemetry: None,
                show_user: None,
                url: None,
                url_label: None,
            }),
        })
    });

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Pause,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    let running_state = active_debug_session_panel(workspace, cx)
        .read_with(cx, |item, _| item.running_state().clone());

    cx.run_until_parked();
    let thread_id = ThreadId(1);

    for operation in &[
        "step_over",
        "continue_thread",
        "step_back",
        "step_in",
        "step_out",
    ] {
        running_state.update(cx, |running_state, cx| match *operation {
            "step_over" => running_state.step_over(cx),
            "continue_thread" => running_state.continue_thread(cx),
            "step_back" => running_state.step_back(cx),
            "step_in" => running_state.step_in(cx),
            "step_out" => running_state.step_out(cx),
            _ => unreachable!(),
        });

        // Check that we step the thread status to the correct intermediate state
        running_state.update(cx, |running_state, cx| {
            assert_eq!(
                running_state
                    .thread_status(cx)
                    .expect("There should be an active thread selected"),
                match *operation {
                    "continue_thread" => ThreadStatus::Running,
                    _ => ThreadStatus::Stepping,
                },
                "Thread status was not set to correct intermediate state after {} request",
                operation
            );
        });

        cx.run_until_parked();

        running_state.update(cx, |running_state, cx| {
            assert_eq!(
                running_state
                    .thread_status(cx)
                    .expect("There should be an active thread selected"),
                ThreadStatus::Stopped,
                "Thread status not reset to Stopped after failed {}",
                operation
            );

            // update state to running, so we can test it actually changes the status back to stopped
            running_state
                .session()
                .update(cx, |session, cx| session.continue_thread(thread_id, cx));
        });
    }
}

#[gpui::test]
async fn test_send_breakpoints_when_editor_has_been_saved(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);
    let project_path = Path::new(path!("/project"));
    let worktree = project
        .update(cx, |project, cx| project.find_worktree(project_path, cx))
        .expect("This worktree should exist in project")
        .0;

    let worktree_id = workspace
        .update(cx, |_, _, cx| worktree.read(cx).id())
        .unwrap();

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("main.rs")), cx)
        })
        .await
        .unwrap();

    let (editor, cx) = cx.add_window_view(|window, cx| {
        Editor::new(
            EditorMode::full(),
            MultiBuffer::build_from_buffer(buffer, cx),
            Some(project.clone()),
            window,
            cx,
        )
    });

    client.on_request::<Launch, _>(move |_, _| Ok(()));

    client.on_request::<StackTrace, _>(move |_, _| {
        Ok(dap::StackTraceResponse {
            stack_frames: Vec::default(),
            total_frames: None,
        })
    });

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Pause,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    let called_set_breakpoints = Arc::new(AtomicBool::new(false));
    client.on_request::<SetBreakpoints, _>({
        let called_set_breakpoints = called_set_breakpoints.clone();
        move |_, args| {
            assert_eq!(path!("/project/main.rs"), args.source.path.unwrap());
            assert_eq!(
                vec![SourceBreakpoint {
                    line: 2,
                    column: None,
                    condition: None,
                    hit_condition: None,
                    log_message: None,
                    mode: None
                }],
                args.breakpoints.unwrap()
            );
            assert!(!args.source_modified.unwrap());

            called_set_breakpoints.store(true, Ordering::SeqCst);

            Ok(dap::SetBreakpointsResponse {
                breakpoints: Vec::default(),
            })
        }
    });

    editor.update_in(cx, |editor, window, cx| {
        editor.move_down(&zed_actions::editor::MoveDown, window, cx);
        editor.toggle_breakpoint(&actions::ToggleBreakpoint, window, cx);
    });

    cx.run_until_parked();

    assert!(
        called_set_breakpoints.load(std::sync::atomic::Ordering::SeqCst),
        "SetBreakpoint request must be called"
    );

    let called_set_breakpoints = Arc::new(AtomicBool::new(false));
    client.on_request::<SetBreakpoints, _>({
        let called_set_breakpoints = called_set_breakpoints.clone();
        move |_, args| {
            assert_eq!(path!("/project/main.rs"), args.source.path.unwrap());
            assert_eq!(
                vec![SourceBreakpoint {
                    line: 3,
                    column: None,
                    condition: None,
                    hit_condition: None,
                    log_message: None,
                    mode: None
                }],
                args.breakpoints.unwrap()
            );
            assert!(args.source_modified.unwrap());

            called_set_breakpoints.store(true, Ordering::SeqCst);

            Ok(dap::SetBreakpointsResponse {
                breakpoints: Vec::default(),
            })
        }
    });

    editor.update_in(cx, |editor, window, cx| {
        editor.move_up(&zed_actions::editor::MoveUp, window, cx);
        editor.insert("new text\n", window, cx);
    });

    editor
        .update_in(cx, |editor, window, cx| {
            editor.save(
                SaveOptions {
                    format: true,
                    force_format: false,
                    autosave: false,
                },
                project.clone(),
                window,
                cx,
            )
        })
        .await
        .unwrap();

    cx.run_until_parked();

    assert!(
        called_set_breakpoints.load(std::sync::atomic::Ordering::SeqCst),
        "SetBreakpoint request must be called after editor is saved"
    );
}

#[gpui::test]
async fn test_unsetting_breakpoints_on_clear_breakpoint_action(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
            "second.rs": "First line\nSecond line\nThird line\nFourth line",
            "no_breakpoints.rs": "Used to ensure that we don't unset breakpoint in files with no breakpoints"
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);
    let project_path = Path::new(path!("/project"));
    let worktree = project
        .update(cx, |project, cx| project.find_worktree(project_path, cx))
        .expect("This worktree should exist in project")
        .0;

    let worktree_id = workspace
        .update(cx, |_, _, cx| worktree.read(cx).id())
        .unwrap();

    let first = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("main.rs")), cx)
        })
        .await
        .unwrap();

    let second = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("second.rs")), cx)
        })
        .await
        .unwrap();

    let (first_editor, cx) = cx.add_window_view(|window, cx| {
        Editor::new(
            EditorMode::full(),
            MultiBuffer::build_from_buffer(first, cx),
            Some(project.clone()),
            window,
            cx,
        )
    });

    let (second_editor, cx) = cx.add_window_view(|window, cx| {
        Editor::new(
            EditorMode::full(),
            MultiBuffer::build_from_buffer(second, cx),
            Some(project.clone()),
            window,
            cx,
        )
    });

    first_editor.update_in(cx, |editor, window, cx| {
        editor.move_down(&zed_actions::editor::MoveDown, window, cx);
        editor.toggle_breakpoint(&actions::ToggleBreakpoint, window, cx);
        editor.move_down(&zed_actions::editor::MoveDown, window, cx);
        editor.move_down(&zed_actions::editor::MoveDown, window, cx);
        editor.toggle_breakpoint(&actions::ToggleBreakpoint, window, cx);
    });

    second_editor.update_in(cx, |editor, window, cx| {
        editor.toggle_breakpoint(&actions::ToggleBreakpoint, window, cx);
        editor.move_down(&zed_actions::editor::MoveDown, window, cx);
        editor.move_down(&zed_actions::editor::MoveDown, window, cx);
        editor.move_down(&zed_actions::editor::MoveDown, window, cx);
        editor.toggle_breakpoint(&actions::ToggleBreakpoint, window, cx);
    });

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    let called_set_breakpoints = Arc::new(AtomicBool::new(false));

    client.on_request::<SetBreakpoints, _>({
        move |_, args| {
            assert!(
                args.breakpoints.is_none_or(|bps| bps.is_empty()),
                "Send empty breakpoint sets to clear them from DAP servers"
            );

            match args
                .source
                .path
                .expect("We should always send a breakpoint's path")
                .as_str()
            {
                path!("/project/main.rs") | path!("/project/second.rs") => {}
                _ => {
                    panic!("Unset breakpoints for path that doesn't have any")
                }
            }

            called_set_breakpoints.store(true, Ordering::SeqCst);

            Ok(dap::SetBreakpointsResponse {
                breakpoints: Vec::default(),
            })
        }
    });

    cx.dispatch_action(crate::ClearAllBreakpoints);
    cx.run_until_parked();
}

#[gpui::test]
async fn test_debug_session_is_shutdown_when_attach_and_launch_request_fails(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    start_debug_session(&workspace, cx, |client| {
        client.on_request::<dap::requests::Initialize, _>(|_, _| {
            Err(ErrorResponse {
                error: Some(Message {
                    format: "failed to launch".to_string(),
                    id: 1,
                    variables: None,
                    send_telemetry: None,
                    show_user: None,
                    url: None,
                    url_label: None,
                }),
            })
        });
    })
    .ok();

    cx.run_until_parked();

    project.update(cx, |project, cx| {
        assert!(
            project.dap_store().read(cx).sessions().count() == 0,
            "Session wouldn't exist if it was shutdown"
        );
    });
}

#[gpui::test]
async fn test_we_send_arguments_from_user_config(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);
    let debug_definition = DebugTaskDefinition {
        adapter: "fake-adapter".into(),
        config: json!({
            "request": "launch",
            "program": "main.rs".to_owned(),
            "args": vec!["arg1".to_owned(), "arg2".to_owned()],
            "cwd": path!("/Random_path"),
            "env": json!({ "KEY": "VALUE" }),
        }),
        label: "test".into(),
        tcp_connection: None,
    };

    let launch_handler_called = Arc::new(AtomicBool::new(false));

    start_debug_session_with(&workspace, cx, debug_definition.clone(), {
        let launch_handler_called = launch_handler_called.clone();

        move |client| {
            let debug_definition = debug_definition.clone();
            let launch_handler_called = launch_handler_called.clone();

            client.on_request::<dap::requests::Launch, _>(move |_, args| {
                launch_handler_called.store(true, Ordering::SeqCst);

                assert_eq!(args.raw, debug_definition.config);

                Ok(())
            });
        }
    })
    .ok();

    cx.run_until_parked();

    assert!(
        launch_handler_called.load(Ordering::SeqCst),
        "Launch request handler was not called"
    );
}

#[gpui::test]
async fn test_active_debug_line_setting(executor: BackgroundExecutor, cx: &mut TestAppContext) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
            "second.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);
    let project_path = Path::new(path!("/project"));
    let worktree = project
        .update(cx, |project, cx| project.find_worktree(project_path, cx))
        .expect("This worktree should exist in project")
        .0;

    let worktree_id = workspace
        .update(cx, |_, _, cx| worktree.read(cx).id())
        .unwrap();

    let main_buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("main.rs")), cx)
        })
        .await
        .unwrap();

    let second_buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("second.rs")), cx)
        })
        .await
        .unwrap();

    let (main_editor, cx) = cx.add_window_view(|window, cx| {
        Editor::new(
            EditorMode::full(),
            MultiBuffer::build_from_buffer(main_buffer, cx),
            Some(project.clone()),
            window,
            cx,
        )
    });

    let (second_editor, cx) = cx.add_window_view(|window, cx| {
        Editor::new(
            EditorMode::full(),
            MultiBuffer::build_from_buffer(second_buffer, cx),
            Some(project.clone()),
            window,
            cx,
        )
    });

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    client.on_request::<dap::requests::Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: 1,
                name: "Thread 1".into(),
            }],
        })
    });

    client.on_request::<dap::requests::Scopes, _>(move |_, _| {
        Ok(dap::ScopesResponse {
            scopes: Vec::default(),
        })
    });

    client.on_request::<StackTrace, _>(move |_, args| {
        assert_eq!(args.thread_id, 1);

        Ok(dap::StackTraceResponse {
            stack_frames: vec![dap::StackFrame {
                id: 1,
                name: "frame 1".into(),
                source: Some(dap::Source {
                    name: Some("main.rs".into()),
                    path: Some(path!("/project/main.rs").into()),
                    source_reference: None,
                    presentation_hint: None,
                    origin: None,
                    sources: None,
                    adapter_data: None,
                    checksums: None,
                }),
                line: 2,
                column: 0,
                end_line: None,
                end_column: None,
                can_restart: None,
                instruction_pointer_reference: None,
                module_id: None,
                presentation_hint: None,
            }],
            total_frames: None,
        })
    });

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Breakpoint,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    main_editor.update_in(cx, |editor, window, cx| {
        let active_debug_lines: Vec<_> = editor.highlighted_rows::<ActiveDebugLine>(cx).collect();

        assert_eq!(
            active_debug_lines.len(),
            1,
            "There should be only one active debug line"
        );

        let point = editor
            .snapshot(window, cx)
            .buffer_snapshot()
            .summary_for_anchor::<language::Point>(&active_debug_lines.first().unwrap().0.start);

        assert_eq!(point.row, 1);
    });

    second_editor.update(cx, |editor, cx| {
        let active_debug_lines: Vec<_> = editor.highlighted_rows::<ActiveDebugLine>(cx).collect();

        assert!(
            active_debug_lines.is_empty(),
            "There shouldn't be any active debug lines"
        );
    });

    let handled_second_stacktrace = Arc::new(AtomicBool::new(false));
    client.on_request::<StackTrace, _>({
        let handled_second_stacktrace = handled_second_stacktrace.clone();
        move |_, args| {
            handled_second_stacktrace.store(true, Ordering::SeqCst);
            assert_eq!(args.thread_id, 1);

            Ok(dap::StackTraceResponse {
                stack_frames: vec![dap::StackFrame {
                    id: 2,
                    name: "frame 2".into(),
                    source: Some(dap::Source {
                        name: Some("second.rs".into()),
                        path: Some(path!("/project/second.rs").into()),
                        source_reference: None,
                        presentation_hint: None,
                        origin: None,
                        sources: None,
                        adapter_data: None,
                        checksums: None,
                    }),
                    line: 3,
                    column: 0,
                    end_line: None,
                    end_column: None,
                    can_restart: None,
                    instruction_pointer_reference: None,
                    module_id: None,
                    presentation_hint: None,
                }],
                total_frames: None,
            })
        }
    });

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Breakpoint,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    second_editor.update_in(cx, |editor, window, cx| {
        let active_debug_lines: Vec<_> = editor.highlighted_rows::<ActiveDebugLine>(cx).collect();

        assert_eq!(
            active_debug_lines.len(),
            1,
            "There should be only one active debug line"
        );

        let point = editor
            .snapshot(window, cx)
            .buffer_snapshot()
            .summary_for_anchor::<language::Point>(&active_debug_lines.first().unwrap().0.start);

        assert_eq!(point.row, 2);
    });

    main_editor.update(cx, |editor, cx| {
        let active_debug_lines: Vec<_> = editor.highlighted_rows::<ActiveDebugLine>(cx).collect();

        assert!(
            active_debug_lines.is_empty(),
            "There shouldn't be any active debug lines"
        );
    });

    assert!(
        handled_second_stacktrace.load(Ordering::SeqCst),
        "Second stacktrace request handler was not called"
    );

    client
        .fake_event(dap::messages::Events::Continued(dap::ContinuedEvent {
            thread_id: 0,
            all_threads_continued: Some(true),
        }))
        .await;

    cx.run_until_parked();

    second_editor.update(cx, |editor, cx| {
        let active_debug_lines: Vec<_> = editor.highlighted_rows::<ActiveDebugLine>(cx).collect();

        assert!(
            active_debug_lines.is_empty(),
            "There shouldn't be any active debug lines"
        );
    });

    main_editor.update(cx, |editor, cx| {
        let active_debug_lines: Vec<_> = editor.highlighted_rows::<ActiveDebugLine>(cx).collect();

        assert!(
            active_debug_lines.is_empty(),
            "There shouldn't be any active debug lines"
        );
    });

    // Clean up
    let shutdown_session = project.update(cx, |project, cx| {
        project.dap_store().update(cx, |dap_store, cx| {
            dap_store.shutdown_session(session.read(cx).session_id(), cx)
        })
    });

    shutdown_session.await.unwrap();

    main_editor.update(cx, |editor, cx| {
        let active_debug_lines: Vec<_> = editor.highlighted_rows::<ActiveDebugLine>(cx).collect();

        assert!(
            active_debug_lines.is_empty(),
            "There shouldn't be any active debug lines after session shutdown"
        );
    });

    second_editor.update(cx, |editor, cx| {
        let active_debug_lines: Vec<_> = editor.highlighted_rows::<ActiveDebugLine>(cx).collect();

        assert!(
            active_debug_lines.is_empty(),
            "There shouldn't be any active debug lines after session shutdown"
        );
    });
}

#[gpui::test]
async fn test_debug_adapters_shutdown_on_app_quit(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    let disconnect_request_received = Arc::new(AtomicBool::new(false));
    let disconnect_clone = disconnect_request_received.clone();

    client.on_request::<Disconnect, _>(move |_, _| {
        disconnect_clone.store(true, Ordering::SeqCst);
        Ok(())
    });

    executor.run_until_parked();

    workspace
        .update(cx, |workspace, _, cx| {
            let panel = workspace.panel::<DebugPanel>(cx).unwrap();
            panel.read_with(cx, |panel, _| {
                assert!(
                    panel.sessions().next().is_some(),
                    "Debug session should be active"
                );
            });
        })
        .unwrap();

    cx.update(|_, cx| cx.defer(|cx| cx.shutdown()));

    executor.run_until_parked();

    assert!(
        disconnect_request_received.load(Ordering::SeqCst),
        "Disconnect request should have been sent to the adapter on app shutdown"
    );
}

#[gpui::test]
async fn test_breakpoint_jumps_only_in_proper_split_view(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
            "second.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let project_path = Path::new(path!("/project"));
    let worktree = project
        .update(cx, |project, cx| project.find_worktree(project_path, cx))
        .expect("This worktree should exist in project")
        .0;

    let worktree_id = workspace
        .update(cx, |_, _, cx| worktree.read(cx).id())
        .unwrap();

    // Open main.rs in pane A (the initial pane)
    let pane_a = workspace
        .update(cx, |multi, _window, cx| {
            multi.workspace().read(cx).active_pane().clone()
        })
        .unwrap();

    let open_main = workspace
        .update(cx, |multi, window, cx| {
            multi.workspace().update(cx, |workspace, cx| {
                workspace.open_path((worktree_id, rel_path("main.rs")), None, true, window, cx)
            })
        })
        .unwrap();
    open_main.await.unwrap();

    cx.run_until_parked();

    // Split pane A to the right, creating pane B
    let pane_b = workspace
        .update(cx, |multi, window, cx| {
            multi.workspace().update(cx, |workspace, cx| {
                workspace.split_pane(pane_a.clone(), SplitDirection::Right, window, cx)
            })
        })
        .unwrap();

    cx.run_until_parked();

    // Open main.rs in pane B
    let weak_pane_b = pane_b.downgrade();
    let open_main_in_b = workspace
        .update(cx, |multi, window, cx| {
            multi.workspace().update(cx, |workspace, cx| {
                workspace.open_path(
                    (worktree_id, rel_path("main.rs")),
                    Some(weak_pane_b),
                    true,
                    window,
                    cx,
                )
            })
        })
        .unwrap();
    open_main_in_b.await.unwrap();

    cx.run_until_parked();

    // Also open second.rs in pane B as an inactive tab
    let weak_pane_b = pane_b.downgrade();
    let open_second_in_b = workspace
        .update(cx, |multi, window, cx| {
            multi.workspace().update(cx, |workspace, cx| {
                workspace.open_path(
                    (worktree_id, rel_path("second.rs")),
                    Some(weak_pane_b),
                    true,
                    window,
                    cx,
                )
            })
        })
        .unwrap();
    open_second_in_b.await.unwrap();

    cx.run_until_parked();

    // Switch pane B back to main.rs so second.rs is inactive there
    let weak_pane_b = pane_b.downgrade();
    let reactivate_main_in_b = workspace
        .update(cx, |multi, window, cx| {
            multi.workspace().update(cx, |workspace, cx| {
                workspace.open_path(
                    (worktree_id, rel_path("main.rs")),
                    Some(weak_pane_b),
                    true,
                    window,
                    cx,
                )
            })
        })
        .unwrap();
    reactivate_main_in_b.await.unwrap();

    cx.run_until_parked();

    // Now open second.rs in pane A, making main.rs an inactive tab there
    let weak_pane_a = pane_a.downgrade();
    let open_second = workspace
        .update(cx, |multi, window, cx| {
            multi.workspace().update(cx, |workspace, cx| {
                workspace.open_path(
                    (worktree_id, rel_path("second.rs")),
                    Some(weak_pane_a),
                    true,
                    window,
                    cx,
                )
            })
        })
        .unwrap();
    open_second.await.unwrap();

    cx.run_until_parked();

    // Layout:
    //   Pane A: second.rs (active), main.rs (inactive tab)
    //   Pane B: main.rs (active), second.rs (inactive tab)

    // Verify pane A's active item is second.rs (main.rs is an inactive tab)
    workspace
        .read_with(cx, |_multi, cx| {
            let active = pane_a.read(cx).active_item().unwrap();
            let editor = active.to_any_view().downcast::<Editor>().unwrap();
            let path = editor.read(cx).active_project_path(cx).unwrap();
            assert_eq!(
                path.path.file_name().unwrap(),
                "second.rs",
                "Pane A should have second.rs active",
            );
        })
        .unwrap();

    // Verify pane B's active item is main.rs
    workspace
        .read_with(cx, |_multi, cx| {
            let active = pane_b.read(cx).active_item().unwrap();
            let editor = active.to_any_view().downcast::<Editor>().unwrap();
            let path = editor.read(cx).active_project_path(cx).unwrap();
            assert_eq!(
                path.path.file_name().unwrap(),
                "main.rs",
                "Pane B should have main.rs active",
            );
        })
        .unwrap();

    // Start a debug session and trigger a breakpoint stop on main.rs line 2
    let session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    client.on_request::<Threads, _>(move |_, _| {
        Ok(dap::ThreadsResponse {
            threads: vec![dap::Thread {
                id: 1,
                name: "Thread 1".into(),
            }],
        })
    });

    client.on_request::<dap::requests::Scopes, _>(move |_, _| {
        Ok(dap::ScopesResponse {
            scopes: Vec::default(),
        })
    });

    client.on_request::<StackTrace, _>(move |_, args| {
        assert_eq!(args.thread_id, 1);

        Ok(dap::StackTraceResponse {
            stack_frames: vec![dap::StackFrame {
                id: 1,
                name: "frame 1".into(),
                source: Some(dap::Source {
                    name: Some("main.rs".into()),
                    path: Some(path!("/project/main.rs").into()),
                    source_reference: None,
                    presentation_hint: None,
                    origin: None,
                    sources: None,
                    adapter_data: None,
                    checksums: None,
                }),
                line: 2,
                column: 0,
                end_line: None,
                end_column: None,
                can_restart: None,
                instruction_pointer_reference: None,
                module_id: None,
                presentation_hint: None,
            }],
            total_frames: None,
        })
    });

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Breakpoint,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    // After first breakpoint stop on main.rs:
    // Pane A should still have second.rs as its active item because
    // main.rs was only an inactive tab there. The debugger should have jumped
    // to main.rs only in pane B where it was already the active tab.
    workspace
        .read_with(cx, |_multi, cx| {
            let pane_a_active = pane_a.read(cx).active_item().unwrap();
            let pane_a_editor = pane_a_active.to_any_view().downcast::<Editor>().unwrap();
            let pane_a_path = pane_a_editor.read(cx).active_project_path(cx).unwrap();
            assert_eq!(
                pane_a_path.path.file_name().unwrap(),
                "second.rs",
                "Pane A should still have second.rs as active item. \
                 The debugger should not switch active tabs in panes where the \
                 breakpoint file is not the active tab (issue #40602)",
            );
        })
        .unwrap();

    // There should be exactly one active debug line across all editors in all panes
    workspace
        .read_with(cx, |_multi, cx| {
            let mut total_active_debug_lines = 0;
            for pane in [&pane_a, &pane_b] {
                for item in pane.read(cx).items() {
                    if let Some(editor) = item.to_any_view().downcast::<Editor>().ok() {
                        total_active_debug_lines += editor
                            .read(cx)
                            .highlighted_rows::<ActiveDebugLine>(cx)
                            .count();
                    }
                }
            }
            assert_eq!(
                total_active_debug_lines, 1,
                "There should be exactly one active debug line across all editors in all panes"
            );
        })
        .unwrap();

    // Pane B should show the debug highlight on main.rs
    workspace
        .read_with(cx, |_multi, cx| {
            let pane_b_active = pane_b.read(cx).active_item().unwrap();
            let pane_b_editor = pane_b_active.to_any_view().downcast::<Editor>().unwrap();

            let active_debug_lines: Vec<_> = pane_b_editor
                .read(cx)
                .highlighted_rows::<ActiveDebugLine>(cx)
                .collect();

            assert_eq!(
                active_debug_lines.len(),
                1,
                "Pane B's main.rs editor should have the active debug line"
            );
        })
        .unwrap();

    // Second breakpoint stop: now on second.rs line 3.
    // Even though pane A has second.rs as its active tab, the debug line
    // should open in pane B (the persistent debug pane) because pane B
    // had the last active debug line.
    client.on_request::<StackTrace, _>(move |_, args| {
        assert_eq!(args.thread_id, 1);

        Ok(dap::StackTraceResponse {
            stack_frames: vec![dap::StackFrame {
                id: 2,
                name: "frame 2".into(),
                source: Some(dap::Source {
                    name: Some("second.rs".into()),
                    path: Some(path!("/project/second.rs").into()),
                    source_reference: None,
                    presentation_hint: None,
                    origin: None,
                    sources: None,
                    adapter_data: None,
                    checksums: None,
                }),
                line: 3,
                column: 0,
                end_line: None,
                end_column: None,
                can_restart: None,
                instruction_pointer_reference: None,
                module_id: None,
                presentation_hint: None,
            }],
            total_frames: None,
        })
    });

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Breakpoint,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    // Pane B should now have second.rs as the active tab with the debug line,
    // because pane B was the last pane that had the debug line (persistent debug pane).
    workspace
        .read_with(cx, |_multi, cx| {
            let pane_b_active = pane_b.read(cx).active_item().unwrap();
            let pane_b_editor = pane_b_active.to_any_view().downcast::<Editor>().unwrap();
            let pane_b_path = pane_b_editor.read(cx).active_project_path(cx).unwrap();
            assert_eq!(
                pane_b_path.path.file_name().unwrap(),
                "second.rs",
                "Pane B should have switched to second.rs because it is the persistent debug pane",
            );

            let active_debug_lines: Vec<_> = pane_b_editor
                .read(cx)
                .highlighted_rows::<ActiveDebugLine>(cx)
                .collect();

            assert_eq!(
                active_debug_lines.len(),
                1,
                "Pane B's second.rs editor should have the active debug line"
            );
        })
        .unwrap();

    // There should still be exactly one active debug line across all editors
    workspace
        .read_with(cx, |_multi, cx| {
            let mut total_active_debug_lines = 0;
            for pane in [&pane_a, &pane_b] {
                for item in pane.read(cx).items() {
                    if let Some(editor) = item.to_any_view().downcast::<Editor>().ok() {
                        total_active_debug_lines += editor
                            .read(cx)
                            .highlighted_rows::<ActiveDebugLine>(cx)
                            .count();
                    }
                }
            }
            assert_eq!(
                total_active_debug_lines, 1,
                "There should be exactly one active debug line across all editors after second stop"
            );
        })
        .unwrap();

    // === New case: Move the debug pane (pane B) active item to a new pane C ===
    // This simulates a user dragging the tab with the active debug line to a new split.
    // The debugger should track that the debug line moved to pane C and use pane C
    // for subsequent debug stops.

    // Split pane B to create pane C
    let pane_c = workspace
        .update(cx, |multi, window, cx| {
            multi.workspace().update(cx, |workspace, cx| {
                workspace.split_pane(pane_b.clone(), SplitDirection::Right, window, cx)
            })
        })
        .unwrap();

    cx.run_until_parked();

    // Move the active item (second.rs with debug line) from pane B to pane C
    workspace
        .update(cx, |_multi, window, cx| {
            move_active_item(&pane_b, &pane_c, true, false, window, cx);
        })
        .unwrap();

    cx.run_until_parked();

    // Verify pane C now has second.rs as active item
    workspace
        .read_with(cx, |_multi, cx| {
            let pane_c_active = pane_c.read(cx).active_item().unwrap();
            let pane_c_editor = pane_c_active.to_any_view().downcast::<Editor>().unwrap();
            let pane_c_path = pane_c_editor.read(cx).active_project_path(cx).unwrap();
            assert_eq!(
                pane_c_path.path.file_name().unwrap(),
                "second.rs",
                "Pane C should have second.rs after moving it from pane B",
            );
        })
        .unwrap();

    // Third breakpoint stop: back on main.rs line 2.
    // The debug line should appear in pane C because that's where the debug line
    // was moved to. The debugger should track pane moves.
    client.on_request::<StackTrace, _>(move |_, args| {
        assert_eq!(args.thread_id, 1);

        Ok(dap::StackTraceResponse {
            stack_frames: vec![dap::StackFrame {
                id: 3,
                name: "frame 3".into(),
                source: Some(dap::Source {
                    name: Some("main.rs".into()),
                    path: Some(path!("/project/main.rs").into()),
                    source_reference: None,
                    presentation_hint: None,
                    origin: None,
                    sources: None,
                    adapter_data: None,
                    checksums: None,
                }),
                line: 2,
                column: 0,
                end_line: None,
                end_column: None,
                can_restart: None,
                instruction_pointer_reference: None,
                module_id: None,
                presentation_hint: None,
            }],
            total_frames: None,
        })
    });

    client
        .fake_event(dap::messages::Events::Stopped(dap::StoppedEvent {
            reason: dap::StoppedEventReason::Breakpoint,
            description: None,
            thread_id: Some(1),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: None,
            hit_breakpoint_ids: None,
        }))
        .await;

    cx.run_until_parked();

    // Pane C should now have main.rs as the active tab with the debug line,
    // because pane C is where the debug line was moved to from pane B.
    workspace
        .read_with(cx, |_multi, cx| {
            let pane_c_active = pane_c.read(cx).active_item().unwrap();
            let pane_c_editor = pane_c_active.to_any_view().downcast::<Editor>().unwrap();
            let pane_c_path = pane_c_editor.read(cx).active_project_path(cx).unwrap();
            assert_eq!(
                pane_c_path.path.file_name().unwrap(),
                "main.rs",
                "Pane C should have switched to main.rs because it is now the persistent debug pane \
                 (the debug line was moved here from pane B)",
            );

            let active_debug_lines: Vec<_> = pane_c_editor
                .read(cx)
                .highlighted_rows::<ActiveDebugLine>(cx)
                .collect();

            assert_eq!(
                active_debug_lines.len(),
                1,
                "Pane C's main.rs editor should have the active debug line"
            );
        })
        .unwrap();

    // There should still be exactly one active debug line across all editors
    workspace
        .read_with(cx, |_multi, cx| {
            let mut total_active_debug_lines = 0;
            for pane in [&pane_a, &pane_b, &pane_c] {
                for item in pane.read(cx).items() {
                    if let Some(editor) = item.to_any_view().downcast::<Editor>().ok() {
                        total_active_debug_lines += editor
                            .read(cx)
                            .highlighted_rows::<ActiveDebugLine>(cx)
                            .count();
                    }
                }
            }
            assert_eq!(
                total_active_debug_lines, 1,
                "There should be exactly one active debug line across all editors after third stop"
            );
        })
        .unwrap();

    // Clean up
    let shutdown_session = project.update(cx, |project, cx| {
        project.dap_store().update(cx, |dap_store, cx| {
            dap_store.shutdown_session(session.read(cx).session_id(), cx)
        })
    });

    shutdown_session.await.unwrap();
}

#[gpui::test]
async fn test_adapter_shutdown_with_child_sessions_on_app_quit(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let parent_session = start_debug_session(&workspace, cx, |_| {}).unwrap();
    let parent_session_id = cx.read(|cx| parent_session.read(cx).session_id());
    let parent_client = parent_session.update(cx, |session, _| session.adapter_client().unwrap());

    let disconnect_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let parent_disconnect_called = Arc::new(AtomicBool::new(false));
    let parent_disconnect_clone = parent_disconnect_called.clone();
    let disconnect_count_clone = disconnect_count.clone();

    parent_client.on_request::<Disconnect, _>(move |_, _| {
        parent_disconnect_clone.store(true, Ordering::SeqCst);
        disconnect_count_clone.fetch_add(1, Ordering::SeqCst);

        for _ in 0..50 {
            if disconnect_count_clone.load(Ordering::SeqCst) >= 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        Ok(())
    });

    parent_client
        .on_response::<StartDebugging, _>(move |_| {})
        .await;
    let _subscription = project::debugger::test::intercept_debug_sessions(cx, |_| {});

    parent_client
        .fake_reverse_request::<StartDebugging>(StartDebuggingRequestArguments {
            configuration: json!({}),
            request: StartDebuggingRequestArgumentsRequest::Launch,
        })
        .await;

    cx.run_until_parked();

    let child_session = project.update(cx, |project, cx| {
        project
            .dap_store()
            .read(cx)
            .session_by_id(SessionId(1))
            .unwrap()
    });
    let child_session_id = cx.read(|cx| child_session.read(cx).session_id());
    let child_client = child_session.update(cx, |session, _| session.adapter_client().unwrap());

    let child_disconnect_called = Arc::new(AtomicBool::new(false));
    let child_disconnect_clone = child_disconnect_called.clone();
    let disconnect_count_clone = disconnect_count.clone();

    child_client.on_request::<Disconnect, _>(move |_, _| {
        child_disconnect_clone.store(true, Ordering::SeqCst);
        disconnect_count_clone.fetch_add(1, Ordering::SeqCst);

        for _ in 0..50 {
            if disconnect_count_clone.load(Ordering::SeqCst) >= 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        Ok(())
    });

    executor.run_until_parked();

    project.update(cx, |project, cx| {
        let store = project.dap_store().read(cx);
        assert!(store.session_by_id(parent_session_id).is_some());
        assert!(store.session_by_id(child_session_id).is_some());
    });

    cx.update(|_, cx| cx.defer(|cx| cx.shutdown()));

    executor.run_until_parked();

    let parent_disconnect_check = parent_disconnect_called.clone();
    let child_disconnect_check = child_disconnect_called.clone();
    let executor_clone = executor.clone();
    let both_disconnected = executor
        .spawn(async move {
            let parent_disconnect = parent_disconnect_check;
            let child_disconnect = child_disconnect_check;

            // We only have 100ms to shutdown the app
            for _ in 0..100 {
                if parent_disconnect.load(Ordering::SeqCst)
                    && child_disconnect.load(Ordering::SeqCst)
                {
                    return true;
                }

                executor_clone
                    .timer(std::time::Duration::from_millis(1))
                    .await;
            }

            false
        })
        .await;

    assert!(
        both_disconnected,
        "Both parent and child sessions should receive disconnect requests"
    );

    assert!(
        parent_disconnect_called.load(Ordering::SeqCst),
        "Parent session should have received disconnect request"
    );
    assert!(
        child_disconnect_called.load(Ordering::SeqCst),
        "Child session should have received disconnect request"
    );
}

#[gpui::test]
async fn test_restart_request_is_not_sent_more_than_once_until_response(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    init_test(cx);

    let fs = FakeFs::new(executor.clone());

    fs.insert_tree(
        path!("/project"),
        json!({
            "main.rs": "First line\nSecond line\nThird line\nFourth line",
        }),
    )
    .await;

    let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let cx = &mut VisualTestContext::from_window(*workspace, cx);

    let session = start_debug_session(&workspace, cx, move |client| {
        client.on_request::<dap::requests::Initialize, _>(move |_, _| {
            Ok(dap::Capabilities {
                supports_restart_request: Some(true),
                ..Default::default()
            })
        });
    })
    .unwrap();

    let client = session.update(cx, |session, _| session.adapter_client().unwrap());

    let restart_count = Arc::new(AtomicUsize::new(0));

    client.on_request::<dap::requests::Restart, _>({
        let restart_count = restart_count.clone();
        move |_, _| {
            restart_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    });

    // This works because the restart request sender is on the foreground thread
    // so it will start running after the gpui update stack is cleared
    session.update(cx, |session, cx| {
        session.restart(None, cx);
        session.restart(None, cx);
        session.restart(None, cx);
    });

    cx.run_until_parked();

    assert_eq!(
        restart_count.load(Ordering::SeqCst),
        1,
        "Only one restart request should be sent while a restart is in-flight"
    );

    session.update(cx, |session, cx| {
        session.restart(None, cx);
    });

    cx.run_until_parked();

    assert_eq!(
        restart_count.load(Ordering::SeqCst),
        2,
        "A second restart should be allowed after the first one completes"
    );
}

#[cfg(unix)]
mod diagnostic_debug {
    use super::*;
    use gpui::{Entity, UpdateGlobal as _};
    use settings::SettingsStore;
    use std::os::unix::fs::PermissionsExt as _;
    use task::SharedTaskContext;

    struct InstalledAdapter;

    #[async_trait::async_trait(?Send)]
    impl dap::adapters::DebugAdapter for InstalledAdapter {
        fn name(&self) -> dap::adapters::DebugAdapterName {
            "DoctorUiFixture".into()
        }
        fn dap_schema(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        async fn config_from_zed_format(
            &self,
            _: task::ZedDebugConfig,
        ) -> anyhow::Result<task::DebugScenario> {
            anyhow::bail!("No formatter authority")
        }
        async fn get_binary(
            &self,
            _: &Arc<dyn dap::adapters::DapDelegate>,
            _: &DebugTaskDefinition,
            _: Option<std::path::PathBuf>,
            _: Option<Vec<String>>,
            _: Option<collections::HashMap<String, String>>,
            _: &mut gpui::AsyncApp,
        ) -> anyhow::Result<dap::adapters::DebugAdapterBinary> {
            anyhow::bail!("Opaque setup must never run")
        }
        fn diagnostic_installed_binary(
            &self,
            definition: &DebugTaskDefinition,
            root: &Path,
            installed: Option<&Path>,
            _: Option<&[String]>,
            environment: Option<&collections::HashMap<String, String>>,
        ) -> anyhow::Result<dap::adapters::DebugAdapterBinary> {
            Ok(dap::adapters::DebugAdapterBinary {
                command: Some(
                    installed
                        .expect("Selected fixture adapter")
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

    async fn fixture(
        cx: &mut TestAppContext,
    ) -> (util::test::TempTree, Entity<Project>, DebugTaskDefinition) {
        init_test(cx);
        cx.executor().allow_parking();
        let directory = util::test::TempTree::new(
            json!({"project": {"program":"native", "untouched":"original"}, "adapter":"native"}),
        );
        let root = directory.path().join("project");
        let adapter = directory.path().join("adapter");
        for path in [&adapter, &root.join("program")] {
            std::fs::write(path, b"\x7fELFfixture native input").expect("Native input");
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).expect("Mode");
        }
        cx.update(|cx| {
            project::trusted_worktrees::init(Default::default(), cx);
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .session
                        .get_or_insert_with(Default::default)
                        .trust_all_worktrees = Some(true);
                    settings.project.dap.insert(
                        "DoctorUiFixture".into(),
                        settings::DapSettingsContent {
                            binary: Some(adapter.to_string_lossy().into()),
                            args: None,
                            env: None,
                        },
                    );
                });
            });
            dap::DapRegistry::global(cx).add_adapter(Arc::new(InstalledAdapter));
        });
        let filesystem = FakeFs::new(cx.executor());
        filesystem
            .insert_tree(&root, json!({"program":"native", "untouched":"original"}))
            .await;
        let project = Project::test(filesystem, [&root as &Path], cx).await;
        let native = Arc::new(project::RealFs::new(None, cx.executor()));
        let store = project.read_with(cx, |project, _| project.dap_store());
        store
            .update(cx, |store, _| store.diagnostic_filesystem_for_test(native))
            .expect("Native inspector");
        let definition = DebugTaskDefinition {
            label: "Reviewed Doctor debug".into(),
            adapter: "DoctorUiFixture".into(),
            tcp_connection: None,
            config: json!({"request":"launch", "program":root.join("program"), "cwd":root}),
        };
        (directory, project, definition)
    }

    #[gpui::test]
    async fn diagnostic_debug_handoff_preserves_sessions_and_blocks_restarts(
        cx: &mut TestAppContext,
    ) {
        let (directory, project, definition) = fixture(cx).await;
        let window = init_test_workspace(&project, cx).await;
        let ordinary = start_debug_session(&window, cx, |_| {}).expect("Ordinary session");
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project.visible_worktrees(cx).next().expect("Worktree")
        });
        let selected_worktree = worktree.read_with(cx, |worktree, _| worktree.id());
        let plan = store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(definition, worktree, cx)
            })
            .await
            .expect("Reviewed installed inputs");
        let _interceptor = project::debugger::test::intercept_debug_sessions(cx, |_| {});
        let pending = window
            .update(cx, |multi, window, cx| {
                multi.workspace().update(cx, |workspace, cx| {
                    workspace.start_diagnostic_debug_session(
                        plan,
                        SharedTaskContext::default(),
                        std::rc::Rc::new(|_| Ok(())),
                        window,
                        cx,
                    )
                })
            })
            .expect("Workspace");
        let handle = pending.await.expect("Same existing session handoff");
        handle.boot.await.expect("Fake native DAP boot");
        let selected = handle.session;
        let selected_id = selected.read_with(cx, |session, _| session.session_id());
        assert!(
            selected.read_with(cx, |session, cx| session.diagnostic_observation().is_some()
                && session.worktree().expect("Selected worktree").read(cx).id()
                    == selected_worktree)
        );
        assert!(!ordinary.read_with(cx, |session, _| session.is_terminated()));
        window
            .update(cx, |multi, window, cx| {
                let workspace = multi.workspace().clone();
                let provider = workspace
                    .read(cx)
                    .debugger_provider()
                    .expect("Existing debugger owner");
                assert!(provider.reveal_diagnostic_session(&selected, window, cx));
                let panel = workspace
                    .read(cx)
                    .panel::<DebugPanel>(cx)
                    .expect("Existing panel");
                assert_eq!(
                    panel
                        .read(cx)
                        .active_session()
                        .expect("Exact selected session")
                        .read(cx)
                        .session_id(cx),
                    selected_id
                );
                assert_eq!(
                    store.read(cx).sessions().count(),
                    2,
                    "Navigation creates no session"
                );
            })
            .expect("Native result navigation");
        window
            .update(cx, |multi, window, cx| {
                multi.workspace().update(cx, |workspace, cx| {
                    let panel = workspace
                        .panel::<DebugPanel>(cx)
                        .expect("Native DebugPanel");
                    panel.update(cx, |panel, cx| {
                        assert!(
                            panel
                                .sessions_with_children
                                .keys()
                                .any(|session| session.read(cx).session_id(cx) == selected_id)
                        );
                        panel.handle_restart_request(selected.clone(), window, cx);
                        panel.handle_start_debugging_request(
                            &StartDebuggingRequestArguments {
                                request: StartDebuggingRequestArgumentsRequest::Launch,
                                configuration: json!({"program":"unapproved-child"}),
                            },
                            selected.clone(),
                            window,
                            cx,
                        );
                        let debug_session = panel
                            .sessions_with_children
                            .keys()
                            .find(|session| session.read(cx).session_id(cx) == selected_id)
                            .expect("Registered selected session")
                            .clone();
                        panel.activate_session(debug_session.clone(), window, cx);
                        panel.focus_handle(cx).focus(window, cx);
                        let running = debug_session.read(cx).running_state().clone();
                        running.update(cx, |state, cx| state.rerun_session(window, cx));
                    });
                })
            })
            .expect("Native controls");
        let visual = &mut VisualTestContext::from_window(*window, cx);
        visual.dispatch_action(crate::RerunSession);
        visual.run_until_parked();
        assert_eq!(
            store.read_with(visual, |store, _| store.sessions().count()),
            2
        );
        assert!(!selected.read_with(visual, |session, _| session.is_terminated()));
        store
            .update(visual, |store, cx| store.shutdown_session(selected_id, cx))
            .await
            .expect("Scoped cancellation");
        assert!(!ordinary.read_with(visual, |session, _| session.is_terminated()));
        assert_eq!(
            store.read_with(visual, |store, _| store.sessions().count()),
            1
        );
        assert_eq!(
            std::fs::read_to_string(directory.path().join("project/untouched"))
                .expect("Unrelated file"),
            "original"
        );
        store
            .update(visual, |store, cx| store.shutdown_sessions(cx))
            .await;
    }

    #[gpui::test]
    async fn diagnostic_debug_handoff_rejects_stale_and_unavailable_provider(
        cx: &mut TestAppContext,
    ) {
        let (directory, project, definition) = fixture(cx).await;
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project.visible_worktrees(cx).next().expect("Worktree")
        });
        let plan = store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(definition.clone(), worktree.clone(), cx)
            })
            .await
            .expect("Preview");
        let window = cx.add_window(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let pending = window
            .update(cx, |multi, window, cx| {
                multi.workspace().update(cx, |workspace, cx| {
                    workspace.start_diagnostic_debug_session(
                        plan,
                        SharedTaskContext::default(),
                        std::rc::Rc::new(|_| Ok(())),
                        window,
                        cx,
                    )
                })
            })
            .expect("Window");
        assert!(pending.await.is_err());
        assert_eq!(store.read_with(cx, |store, _| store.sessions().count()), 0);
        let panel = window
            .update(cx, |multi, window, cx| {
                multi.workspace().update(cx, |_, cx| {
                    cx.spawn_in(window, async move |workspace, cx| {
                        DebugPanel::load(workspace, cx).await
                    })
                })
            })
            .expect("Existing workspace")
            .await
            .expect("Existing DebugPanel");
        window
            .update(cx, |multi, window, cx| {
                multi
                    .workspace()
                    .update(cx, |workspace, cx| workspace.add_panel(panel, window, cx));
            })
            .expect("Register existing panel");
        let plan = store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(definition, worktree, cx)
            })
            .await
            .expect("New preview");
        std::fs::write(
            directory.path().join("project/program"),
            b"\x7fELFchanged native input",
        )
        .expect("Changed input");
        let pending = window
            .update(cx, |multi, window, cx| {
                multi.workspace().update(cx, |workspace, cx| {
                    workspace.start_diagnostic_debug_session(
                        plan,
                        SharedTaskContext::default(),
                        std::rc::Rc::new(|_| Ok(())),
                        window,
                        cx,
                    )
                })
            })
            .expect("Window");
        assert!(pending.await.is_err());
        assert_eq!(store.read_with(cx, |store, _| store.sessions().count()), 0);
    }
    #[gpui::test]
    async fn diagnostic_debug_handoff_failure_and_revocation_preserve_other_work(
        cx: &mut TestAppContext,
    ) {
        let (_directory, project, definition) = fixture(cx).await;
        let window = init_test_workspace(&project, cx).await;
        let ordinary = start_debug_session(&window, cx, |_| {}).expect("Ordinary session");
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project.visible_worktrees(cx).next().expect("Worktree")
        });
        let failing = project::debugger::test::intercept_debug_sessions(cx, |client| {
            client.on_request::<dap::requests::Initialize, _>(|_, _| {
                Err(dap::ErrorResponse { error: None })
            });
        });
        let plan = store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(definition.clone(), worktree.clone(), cx)
            })
            .await
            .expect("Review");
        let handle = window
            .update(cx, |multi, window, cx| {
                multi.workspace().update(cx, |workspace, cx| {
                    workspace.start_diagnostic_debug_session(
                        plan,
                        SharedTaskContext::default(),
                        std::rc::Rc::new(|_| Ok(())),
                        window,
                        cx,
                    )
                })
            })
            .expect("Window")
            .await
            .expect("Session handle");
        assert!(handle.boot.await.is_err());
        cx.run_until_parked();
        assert_eq!(store.read_with(cx, |store, _| store.sessions().count()), 1);
        assert!(!ordinary.read_with(cx, |session, _| session.is_terminated()));
        drop(failing);
        let _success = project::debugger::test::intercept_debug_sessions(cx, |_| {});
        let plan = store
            .update(cx, |store, cx| {
                store.prepare_diagnostic_launch(definition, worktree, cx)
            })
            .await
            .expect("Fresh review");
        let handle = window
            .update(cx, |multi, window, cx| {
                multi.workspace().update(cx, |workspace, cx| {
                    workspace.start_diagnostic_debug_session(
                        plan,
                        SharedTaskContext::default(),
                        std::rc::Rc::new(|_| Ok(())),
                        window,
                        cx,
                    )
                })
            })
            .expect("Window")
            .await
            .expect("Session handle");
        handle.boot.await.expect("Restricted session boot");
        let selected = handle.session;
        let selected_id = selected.read_with(cx, |session, _| session.session_id());
        let tree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = selected.read_with(cx, |session, cx| {
            session.worktree().expect("Worktree").read(cx).id()
        });
        let trust = cx.update(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).expect("Trust owner")
        });
        trust.update(cx, |trust, cx| {
            trust.restrict(
                tree_store.downgrade(),
                collections::HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                    worktree_id,
                )]),
                cx,
            )
        });
        cx.run_until_parked();
        assert!(selected.read_with(cx, |session, _| session.is_terminated()));
        assert!(store.read_with(cx, |store, _| store.session_by_id(selected_id).is_none()));
        assert!(!ordinary.read_with(cx, |session, _| session.is_terminated()));
        store
            .update(cx, |store, cx| store.shutdown_sessions(cx))
            .await;
    }
    #[gpui::test]
    async fn diagnostic_debug_handoff_rechecks_caller_at_final_admission(cx: &mut TestAppContext) {
        let (_directory, project, definition) = fixture(cx).await;
        let window = init_test_workspace(&project, cx).await;
        let ordinary = start_debug_session(&window, cx, |_| {}).expect("Ordinary session");
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project.visible_worktrees(cx).next().expect("Worktree")
        });
        for reason in [
            "caller closed",
            "cancelled",
            "inputs changed",
            "deadline elapsed",
        ] {
            let plan = store
                .update(cx, |store, cx| {
                    store.prepare_diagnostic_launch(definition.clone(), worktree.clone(), cx)
                })
                .await
                .expect("Passive preview");
            let allowed = Arc::new(AtomicBool::new(true));
            let calls = Arc::new(AtomicUsize::new(0));
            let authority: project::debugger::dap_store::DiagnosticDebugAuthority =
                std::rc::Rc::new({
                    let allowed = allowed.clone();
                    let calls = calls.clone();
                    let store = store.clone();
                    move |cx| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(store.read(cx).sessions().count(), 1);
                        anyhow::ensure!(allowed.load(Ordering::SeqCst), "{reason}");
                        Ok(())
                    }
                });
            let pending = window
                .update(cx, |multi, window, cx| {
                    multi.workspace().update(cx, |workspace, cx| {
                        workspace.start_diagnostic_debug_session(
                            plan,
                            SharedTaskContext::default(),
                            authority,
                            window,
                            cx,
                        )
                    })
                })
                .expect("Existing workspace handoff");
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            allowed.store(false, Ordering::SeqCst);
            assert!(pending.await.is_err());
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(store.read_with(cx, |store, _| store.sessions().count()), 1);
            assert!(!ordinary.read_with(cx, |session, _| session.is_terminated()));
        }
        store
            .update(cx, |store, cx| store.shutdown_sessions(cx))
            .await;
    }
}

#[gpui::test]
async fn experimental_manual_restart_refuses_off_then_on_during_adapter_shutdown(
    executor: BackgroundExecutor,
    cx: &mut TestAppContext,
) {
    use anyhow::Context as _;
    use gpui::UpdateGlobal as _;
    use std::io::{BufRead as _, Read as _, Write as _};
    use std::time::{Duration, Instant};

    fn set_opt_in(enabled: bool, cx: &mut gpui::App) {
        settings::SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |content| {
                content.feature_flags.get_or_insert_default().insert(
                    "rust-named-cargo-presets-experimental".into(),
                    if enabled { "on" } else { "off" }.into(),
                );
            });
        });
    }

    struct AdapterOwner {
        stop: Arc<AtomicBool>,
        release: std::sync::mpsc::Sender<()>,
        thread: Option<std::thread::JoinHandle<anyhow::Result<()>>>,
    }
    impl Drop for AdapterOwner {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Err(error) = self.release.send(()) {
                log::debug!("Held restart adapter already retired: {error}");
            }
            if let Some(thread) = self.thread.take() {
                match thread.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => log::error!("Held restart adapter failed: {error:#}"),
                    Err(_) => log::error!("Held restart adapter thread panicked"),
                }
            }
        }
    }

    init_test(cx);
    cx.executor().allow_parking();
    cx.update(|cx| {
        settings::SettingsStore::update_global(cx, |store, _| {
            store.register_setting::<feature_flags::FeatureFlagsSettings>();
        });
        feature_flags::FeatureFlagStore::init(cx);
        set_opt_in(true, cx);
        dap::transport::use_real_transport_for_test(cx);
    });
    let filesystem = FakeFs::new(executor.clone());
    filesystem
        .insert_tree(path!("/project"), json!({"main.rs":"fn main() {}"}))
        .await;
    let project = Project::test(filesystem, [path!("/project").as_ref()], cx).await;
    let workspace = init_test_workspace(&project, cx).await;
    let panel = workspace
        .read_with(cx, |multi, cx| {
            multi.workspace().read(cx).panel::<DebugPanel>(cx)
        })
        .expect("Workspace")
        .expect("Existing Debug panel");
    let store = project.read_with(cx, |project, _| project.dap_store());
    let refusals = Arc::new(AtomicUsize::new(0));
    let _notifications = cx.update({
        let refusals = refusals.clone();
        |cx| cx.subscribe(&store, move |_, event, _| {
            if matches!(event, project::debugger::dap_store::DapStoreEvent::Notification(message) if message.contains("revoked")) {
                refusals.fetch_add(1, Ordering::SeqCst);
            }
        })
    });
    let worktree = project.read_with(cx, |project, cx| {
        project
            .visible_worktrees(cx)
            .next()
            .expect("Existing Worktree")
    });
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .expect("Owned adapter listener");
    let address = listener.local_addr().expect("Actual loopback adapter");
    listener.set_nonblocking(true).expect("Bounded accept");
    let stop = Arc::new(AtomicBool::new(false));
    let hold_next = Arc::new(AtomicBool::new(false));
    let held_shutdowns = Arc::new(AtomicUsize::new(0));
    let connections = Arc::new(AtomicUsize::new(0));
    let launches = Arc::new(AtomicUsize::new(0));
    let (release, releases) = std::sync::mpsc::channel();
    let thread = std::thread::spawn({
        let stop = stop.clone();
        let hold_next = hold_next.clone();
        let held_shutdowns = held_shutdowns.clone();
        let connections = connections.clone();
        let launches = launches.clone();
        move || -> anyhow::Result<()> {
            fn write_frame(
                stream: &mut std::net::TcpStream,
                value: serde_json::Value,
            ) -> anyhow::Result<()> {
                let bytes = serde_json::to_vec(&value)?;
                write!(stream, "Content-Length: {}\r\n\r\n", bytes.len())?;
                stream.write_all(&bytes)?;
                stream.flush()?;
                Ok(())
            }
            while !stop.load(Ordering::SeqCst) {
                let (stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(error) => return Err(error.into()),
                };
                connections.fetch_add(1, Ordering::SeqCst);
                stream.set_nonblocking(false)?;
                stream.set_read_timeout(Some(Duration::from_secs(10)))?;
                stream.set_write_timeout(Some(Duration::from_secs(10)))?;
                let mut writer = stream.try_clone()?;
                let mut reader = std::io::BufReader::new(stream);
                loop {
                    let mut content_length = None;
                    let mut header_bytes = 0usize;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line)? == 0 {
                            break;
                        }
                        header_bytes += line.len();
                        anyhow::ensure!(header_bytes <= 8192, "Fixture DAP headers exceeded bound");
                        if line == "\r\n" {
                            break;
                        }
                        if let Some(length) = line.strip_prefix("Content-Length:") {
                            content_length = Some(length.trim().parse::<usize>()?);
                        }
                    }
                    let Some(content_length) = content_length else {
                        break;
                    };
                    anyhow::ensure!(
                        content_length <= 65536,
                        "Fixture DAP request exceeded bound"
                    );
                    let mut bytes = vec![0; content_length];
                    reader.read_exact(&mut bytes)?;
                    let request: serde_json::Value = serde_json::from_slice(&bytes)?;
                    let command = request["command"].as_str().context("Actual DAP request")?;
                    let sequence = request["seq"].as_u64().context("Actual request sequence")?;
                    if command == "disconnect" && hold_next.swap(false, Ordering::SeqCst) {
                        held_shutdowns.fetch_add(1, Ordering::SeqCst);
                        releases.recv_timeout(Duration::from_secs(10))?;
                    }
                    if command == "launch" {
                        launches.fetch_add(1, Ordering::SeqCst);
                    }
                    let body = match command {
                        "initialize" => {
                            json!({"supportsTerminateRequest":false,"supportsConfigurationDoneRequest":false})
                        }
                        "threads" => json!({"threads":[]}),
                        "setBreakpoints" => json!({"breakpoints":[]}),
                        "setExceptionBreakpoints" => json!({}),
                        _ => serde_json::Value::Null,
                    };
                    write_frame(
                        &mut writer,
                        json!({"type":"response","seq":sequence+1,"request_seq":sequence,"success":true,"command":command,"body":body}),
                    )?;
                    if command == "initialize" {
                        write_frame(
                            &mut writer,
                            json!({"type":"event","seq":sequence+2,"event":"initialized","body":{}}),
                        )?;
                    }
                    if command == "disconnect" {
                        break;
                    }
                }
            }
            Ok(())
        }
    });
    let mut adapter = AdapterOwner {
        stop,
        release,
        thread: Some(thread),
    };
    let binary = dap::adapters::DebugAdapterBinary {
        command: None,
        arguments: Vec::new(),
        envs: Default::default(),
        cwd: None,
        connection: Some(dap::adapters::TcpArguments {
            host: address.ip(),
            port: address.port(),
            timeout: Some(10000),
        }),
        request_args: StartDebuggingRequestArguments {
            request: StartDebuggingRequestArgumentsRequest::Launch,
            configuration: json!({"request":"launch"}),
        },
    };
    for (marked, revoke, expected_connections) in
        [(true, true, 1), (true, false, 3), (false, true, 5)]
    {
        cx.update(|cx| set_opt_in(marked || !revoke, cx));
        cx.run_until_parked();
        let session = store.update(cx, |store, cx| {
            let session = store.new_session(
                Some("Held restart".into()),
                "fake-adapter".into(),
                task::SharedTaskContext::default(),
                None,
                Default::default(),
                cx,
            );
            if marked {
                session
                    .update(cx, |session, _| {
                        session.set_required_experimental_features(vec![
                            "rust-named-cargo-presets-experimental".into(),
                        ])
                    })
                    .expect("Original marked requirements");
            }
            session
        });
        let registration = workspace
            .update(cx, |_, window, cx| {
                panel.update(cx, |_, cx| {
                    let session = session.clone();
                    cx.spawn_in(window, async move |panel, cx| {
                        DebugPanel::register_session(panel, session, true, cx).await
                    })
                })
            })
            .expect("Original panel registration");
        registration.await.expect("Original Session registration");
        let boot = session.update(cx, |session, cx| {
            session.boot(binary.clone(), worktree.clone(), store.downgrade(), cx)
        });
        let boot_timeout = executor.timer(Duration::from_secs(10));
        futures::pin_mut!(boot, boot_timeout);
        match futures::future::select(boot, boot_timeout).await {
            futures::future::Either::Left((result, _)) => result.expect("Actual adapter boot"),
            futures::future::Either::Right(_) => {
                panic!("Actual adapter boot exceeded original bound")
            }
        }
        let old_id = session.read_with(cx, |session, _| session.session_id());
        let before_connections = connections.load(Ordering::SeqCst);
        let before_launches = launches.load(Ordering::SeqCst);
        let before_held = held_shutdowns.load(Ordering::SeqCst);
        let before_refusals = refusals.load(Ordering::SeqCst);
        hold_next.store(true, Ordering::SeqCst);
        workspace
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.handle_restart_request(session.clone(), window, cx)
                })
            })
            .expect("Actual restart control");
        let deadline = Instant::now() + Duration::from_secs(10);
        while held_shutdowns.load(Ordering::SeqCst) == before_held {
            cx.run_until_parked();
            assert!(Instant::now() < deadline, "Actual Disconnect was not held");
            executor.timer(Duration::from_millis(2)).await;
        }
        assert_eq!(connections.load(Ordering::SeqCst), before_connections);
        if revoke {
            cx.update(|cx| set_opt_in(false, cx));
            cx.run_until_parked();
            cx.update(|cx| set_opt_in(true, cx));
            cx.run_until_parked();
        }
        adapter
            .release
            .send(())
            .expect("Release actual Disconnect response");
        loop {
            cx.run_until_parked();
            if marked && revoke {
                if refusals.load(Ordering::SeqCst) == before_refusals + 1 {
                    break;
                }
            } else if store.read_with(cx, |store, cx| {
                store.sessions().any(|session| {
                    session.read(cx).session_id() != old_id && session.read(cx).is_started()
                })
            }) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Actual restart completion exceeded original bound"
            );
            executor.timer(Duration::from_millis(2)).await;
        }
        if marked && revoke {
            assert_eq!(
                connections.load(Ordering::SeqCst),
                before_connections,
                "Revoked restart created no replacement adapter"
            );
            assert_eq!(
                launches.load(Ordering::SeqCst),
                before_launches,
                "Revoked restart launched no replacement"
            );
            assert_eq!(store.read_with(cx, |store, _| store.sessions().count()), 0);
        } else {
            assert_eq!(connections.load(Ordering::SeqCst), before_connections + 1);
            assert_eq!(launches.load(Ordering::SeqCst), before_launches + 1);
            assert_eq!(store.read_with(cx, |store, _| store.sessions().count()), 1);
        }
        assert_eq!(connections.load(Ordering::SeqCst), expected_connections);
        store
            .update(cx, |store, cx| store.shutdown_sessions(cx))
            .await;
    }
    adapter.stop.store(true, Ordering::SeqCst);
    adapter
        .thread
        .take()
        .expect("Owned adapter thread")
        .join()
        .expect("Adapter thread did not panic")
        .expect("Actual adapter completed successfully");
    drop(adapter);
}
