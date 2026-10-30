use gpui::AppContext as _;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tasks_ui::jvm_workflow::{
    JvmWorkflowDebugContext, JvmWorkflowDebugHooks, JvmWorkflowDebugTarget,
};

struct CachedJavaDebugAdapter;

#[async_trait::async_trait(?Send)]
impl dap::adapters::DebugAdapter for CachedJavaDebugAdapter {
    fn name(&self) -> dap::adapters::DebugAdapterName {
        "Java".into()
    }
    fn dap_schema(&self) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn config_from_zed_format(
        &self,
        _: task::ZedDebugConfig,
    ) -> anyhow::Result<task::DebugScenario> {
        anyhow::bail!("The cached fixture requires an explicit current Java launch configuration")
    }
    async fn get_binary(
        &self,
        delegate: &Arc<dyn dap::adapters::DapDelegate>,
        definition: &dap::adapters::DebugTaskDefinition,
        path: Option<std::path::PathBuf>,
        arguments: Option<Vec<String>>,
        environment: Option<collections::HashMap<String, String>>,
        cx: &mut gpui::AsyncApp,
    ) -> anyhow::Result<dap::adapters::DebugAdapterBinary> {
        anyhow::ensure!(
            definition
                .tcp_connection
                .as_ref()
                .is_some_and(|connection| connection.port.is_some())
                && definition.config.get("tcp_only") == Some(&serde_json::Value::Bool(true)),
            "The cached Java fixture requires its actual provider TCP endpoint"
        );
        dap::adapters::FakeAdapter {}
            .get_binary(delegate, definition, path, arguments, environment, cx)
            .await
    }
}

#[gpui::test]
#[ignore = "requires explicit cached JVM providers, server/build JDKs and verified GROOVY_JAVA_DEBUG_JAR"]
async fn cached_gradle_editor_build_dependencies_tests_and_debug(cx: &mut gpui::TestAppContext) {
    tasks_ui::jvm_workflow::run_cached_gradle_workflow(
        cx,
        JvmWorkflowDebugHooks {
            initialize: |cx| {
                crate::init(cx);
                dap::transport::use_real_transport_for_test(cx);
            },
            run: |context, cx| Box::pin(run_java_debug(context, cx)),
        },
    )
    .await;
}

async fn run_java_debug(
    context: JvmWorkflowDebugContext,
    cx: &mut gpui::VisualTestContext,
) -> JvmWorkflowDebugTarget {
    let JvmWorkflowDebugContext {
        project,
        workspace,
        project_root,
        worktree_id,
    } = context;
    use crate::debugger_panel::DebugPanel;
    use editor::{MultiBufferOffset, SelectionEffects};
    use language::ToPoint as _;
    use project::debugger::session::ThreadStatus;

    let log_store = cx.new(|cx| debugger_tools::LogStore::new(cx));
    log_store.update(cx, |store, cx| store.add_project(&project, cx));
    let java_path = project_root.join("application/src/main/java/evidence/Main.java");
    let java_source = std::fs::read_to_string(&java_path).expect("current Java Debug source");
    let (java_buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(java_path.clone(), cx)
        })
        .await
        .expect("Java Debug source buffer");
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let server = lsp_store.read_with(cx, |store, _| {
        let id = store
            .jdtls_project_snapshot(worktree_id)
            .expect("current Debug provider")
            .0
            .server_id;
        store
            .language_server_for_id(id)
            .expect("actual Java Debug language server")
    });
    let main_classes = server
        .request::<lsp::request::ExecuteCommand>(
            lsp::ExecuteCommandParams {
                command: "vscode.java.resolveMainClass".into(),
                arguments: vec![serde_json::json!("evidence.Main")],
                ..Default::default()
            },
            Duration::from_secs(60),
        )
        .await
        .into_response()
        .expect("actual Java Debug main resolution");
    let project_name = main_classes
        .as_ref()
        .and_then(|value| value.as_array())
        .and_then(|classes| {
            classes.iter().find(|entry| {
                entry.get("mainClass").and_then(|value| value.as_str()) == Some("evidence.Main")
            })
        })
        .and_then(|entry| entry.get("projectName"))
        .and_then(|value| value.as_str())
        .expect("current Java main project")
        .to_owned();
    let classpaths = server
        .request::<lsp::request::ExecuteCommand>(
            lsp::ExecuteCommandParams {
                command: "vscode.java.resolveClasspath".into(),
                arguments: vec![
                    serde_json::json!("evidence.Main"),
                    serde_json::json!(project_name),
                    serde_json::Value::Null,
                ],
                ..Default::default()
            },
            Duration::from_secs(60),
        )
        .await
        .into_response()
        .expect("actual Java Debug classpath resolution")
        .expect("Java classpath response");
    let classpaths = classpaths
        .as_array()
        .expect("Java classpath groups")
        .iter()
        .flat_map(|group| group.as_array().expect("classpath group").iter())
        .map(|entry| entry.as_str().expect("classpath path").to_owned())
        .collect::<Vec<_>>();
    assert!(!classpaths.is_empty());
    cx.update(|_, cx| dap::DapRegistry::global(cx).add_adapter(Arc::new(CachedJavaDebugAdapter)));
    let mut async_cx = cx.update(|window, cx| window.to_async(cx));
    let debug_panel = DebugPanel::load(workspace.downgrade(), &mut async_cx)
        .await
        .expect("existing Debug panel");
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.add_panel(debug_panel.clone(), window, cx)
    });
    let source_path = project.read_with(cx, |project, cx| {
        project
            .find_project_path(&java_path, cx)
            .expect("current Debug source path")
    });
    let item = workspace
        .update_in(cx, |workspace, window, cx| {
            workspace.open_path(source_path, None, true, window, cx)
        })
        .await
        .expect("open current Debug source in Editor");
    let java_editor = item
        .downcast::<editor::Editor>()
        .expect("Java Debug Editor");
    let breakpoint_offset = java_source
        .find("System.out.println")
        .expect("executable Java line");
    java_editor.update_in(cx, |editor, window, cx| {
        editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
            selections.select_ranges([
                MultiBufferOffset(breakpoint_offset)..MultiBufferOffset(breakpoint_offset)
            ]);
        });
        editor.toggle_breakpoint(&editor::actions::ToggleBreakpoint, window, cx);
    });
    let breakpoints = project.read_with(cx, |project, _| project.breakpoint_store());
    breakpoints.read_with(cx, |store, cx| {
        let all = store.all_source_breakpoints(cx);
        let current = all
            .get(java_path.as_path())
            .expect("Editor Java breakpoint");
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].row, 4);
        assert!(current[0].state.is_enabled());
    });
    let mut previous_session_id = None;
    for (label, restrict) in [
        ("Combined current Java main", false),
        ("Combined trust-restricted Java main", true),
    ] {
        let port = server
            .request::<lsp::request::ExecuteCommand>(
                lsp::ExecuteCommandParams {
                    command: "vscode.java.startDebugSession".into(),
                    ..Default::default()
                },
                Duration::from_secs(60),
            )
            .await
            .into_response()
            .expect("actual Java Debug endpoint")
            .and_then(|value| value.as_u64())
            .and_then(|value| u16::try_from(value).ok())
            .filter(|port| *port > 0)
            .expect("actual Java Debug TCP port");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.start_debug_session(task::DebugScenario {
                adapter: "Java".into(), label: label.into(), build: None,
                config: serde_json::json!({"request":"launch","tcp_only":true,"mainClass":"evidence.Main","projectName":project_name,"classPaths":classpaths,
                    "javaExec":format!("{}/bin/java",std::env::var("GROOVY_BUILD_JAVA_HOME").expect("build JDK")),
                    "cwd":project_root,"stopOnEntry":false,"console":"internalConsole"}),
                tcp_connection: Some(task::TcpArgumentsTemplate { host: Some("127.0.0.1".parse().expect("loopback")), port: Some(port), timeout: None }),
            }, task::SharedTaskContext::default(), Some(java_buffer.clone()), Some(worktree_id), window, cx);
        });
        let deadline = Instant::now() + Duration::from_secs(60);
        let session = loop {
            cx.run_until_parked();
            let session = debug_panel
                .read_with(cx, |panel, cx| {
                    panel.active_session().map(|ui| ui.read(cx).session(cx))
                })
                .filter(|session| {
                    session.read_with(cx, |session, _| {
                        session.any_stopped_thread()
                            && previous_session_id != Some(session.session_id())
                    })
                });
            if let Some(session) = session {
                break session;
            }
            assert!(
                Instant::now() < deadline,
                "Java main did not hit the Editor breakpoint through Debug UI"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            cx.run_until_parked();
            let frames = session.update(cx, |session, cx| {
                let thread = session
                    .threads(cx)
                    .into_iter()
                    .find(|(_, status)| *status == ThreadStatus::Stopped)
                    .expect("stopped Java thread")
                    .0;
                session
                    .stack_frames(project::debugger::session::ThreadId(thread.id), cx)
                    .expect("current Java frames")
            });
            let mapped = frames.iter().any(|frame| {
                frame.dap.line == 5
                    && frame.dap.name.contains("Main.main")
                    && frame
                        .dap
                        .source
                        .as_ref()
                        .and_then(|source| source.path.as_ref())
                        .is_some_and(|path| {
                            std::path::Path::new(path)
                                .canonicalize()
                                .expect("current frame source")
                                == java_path
                        })
            });
            let selected = breakpoints.read_with(cx, |store, cx| {
                store.active_position().is_some_and(|frame| {
                    frame.path.as_ref() == java_path.as_path()
                        && frame
                            .position
                            .to_point(&java_buffer.read(cx).snapshot())
                            .row
                            == 4
                })
            });
            if mapped && selected {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Debug UI did not select the current stopped Java source frame"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        let session_id = session.read_with(cx, |session, _| session.session_id());
        previous_session_id = Some(session_id);
        let deadline = Instant::now() + Duration::from_secs(30);
        let process_id = loop {
            cx.run_until_parked();
            let messages = log_store.read_with(cx, |store, _| {
                store.rpc_messages_for_session_id(&project.downgrade(), session_id)
            });
            let messages = messages
                .iter()
                .filter(|message| message.trim().starts_with('{'))
                .map(|message| {
                    serde_json::from_str::<serde_json::Value>(message).expect("actual DAP JSON log")
                })
                .collect::<Vec<_>>();
            let breakpoint_stop = messages.iter().any(|message| {
                message.get("event").and_then(|value| value.as_str()) == Some("stopped")
                    && message
                        .pointer("/body/reason")
                        .and_then(|value| value.as_str())
                        == Some("breakpoint")
            });
            let process_id = messages
                .iter()
                .find(|message| {
                    // Java Debug reports the debuggee through its custom processid event.
                    message.get("event").and_then(|value| value.as_str()) == Some("processid")
                })
                .and_then(|message| message.pointer("/body/processId"))
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .filter(|pid| *pid > 0);
            if breakpoint_stop && let Some(process_id) = process_id {
                break process_id;
            }
            assert!(
                Instant::now() < deadline,
                "actual DAP logs did not identify the breakpoint reason and debuggee process"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        };
        assert!(
            util::command::new_command("/bin/kill")
                .args(["-0", &process_id.to_string()])
                .stderr(util::command::Stdio::null())
                .status()
                .await
                .expect("live Java debuggee")
                .success()
        );
        if restrict {
            eprintln!(
                "Combined Java Debug second run is paused at the verified current-source breakpoint with actual process PID {process_id}"
            );
            return JvmWorkflowDebugTarget {
                session,
                process_id,
            };
        }
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(crate::FocusToolbar);
        cx.run_until_parked();
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(crate::Continue);
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            cx.run_until_parked();
            if session.read_with(cx, |session, _| session.is_terminated()) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Debug toolbar Continue did not finish the Java main"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
        eprintln!(
            "Combined Java Debug UI hit current Editor breakpoint at Main.java:5, selected the source frame and continued to termination"
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        while util::command::new_command("/bin/kill")
            .args(["-0", &process_id.to_string()])
            .stderr(util::command::Stdio::null())
            .status()
            .await
            .expect("continued Java process")
            .success()
        {
            assert!(
                Instant::now() < deadline,
                "continued Java debuggee process did not exit"
            );
            cx.background_executor
                .timer(Duration::from_millis(100))
                .await;
        }
    }
    panic!("the second current Java session must be returned for trust revocation");
}
