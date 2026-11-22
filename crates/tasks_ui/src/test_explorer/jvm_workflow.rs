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

// Passing the Debug phase in keeps the test support dependency directed from Debug UI to Tasks UI.
pub enum JvmWorkflowDebugHooks {
    JavaMain {
        initialize: fn(&mut App),
        run: for<'a> fn(
            JvmWorkflowDebugContext,
            &'a mut VisualTestContext,
        ) -> futures::future::LocalBoxFuture<'a, JvmWorkflowDebugTarget>,
    },
    SelectedJavaTest {
        initialize: fn(&mut App),
        run: for<'a> fn(
            JvmWorkflowDebugContext,
            &'a mut VisualTestContext,
        ) -> futures::future::LocalBoxFuture<'a, ()>,
    },
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

#[cfg(all(test, unix))]
fn cached_jvm_fixture_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os(name).expect(name))
        .canonicalize()
        .expect(name)
}

#[cfg(all(test, unix))]
async fn start_cached_fixture_jdtls(
    root: &std::path::Path,
    options: serde_json::Value,
    bundles: &[std::path::PathBuf],
    cx: &mut TestAppContext,
) -> (Entity<Project>, WorktreeId, tempfile::TempDir) {
    use language::{FakeLspAdapter, Language, LanguageConfig, LanguageMatcher};
    use project::trusted_worktrees::{self, PathTrust, TrustedWorktrees};

    struct FixtureLogger;
    impl log::Log for FixtureLogger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Warn && metadata.target().starts_with("project::")
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                eprintln!("{}: {}", record.target(), record.args());
            }
        }

        fn flush(&self) {}
    }
    if log::set_logger(&FixtureLogger).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
    let launcher = cached_jvm_fixture_path("GROOVY_JDTLS_LAUNCHER");
    let provider_java = cached_jvm_fixture_path("GROOVY_LSP_JAVA_HOME");
    let provider_data = tempfile::tempdir().expect("fresh JDTLS provider state");
    let settings = serde_json::json!({"lsp": {"jdtls": {
        "binary": {"path": launcher, "arguments": [
            format!("--java-executable={}/bin/java", provider_java.display()),
            "-data", provider_data.path().display().to_string(),
        ], "env": {"JAVA_HOME": provider_java, "JAVA_TOOL_OPTIONS": "-Djava.net.preferIPv4Stack=true"}},
        "settings": {"java": &options},
    }}});
    cx.executor().allow_parking();
    cx.update(|cx| {
        let _app_state = workspace::AppState::test(cx);
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store
                .set_user_settings(&settings.to_string(), cx)
                .expect("offline JDTLS fixture settings");
        });
        trusted_worktrees::init(Default::default(), cx);
        editor::init(cx);
        jvm_ui::init(cx);
    });
    let project = Project::test(Arc::new(fs::RealFs::new(None, cx.executor())), [root], cx).await;
    let (worktrees, worktree_id, languages) = project.read_with(cx, |project, cx| {
        let worktrees = project.worktree_store();
        let worktree_id = worktrees
            .read(cx)
            .worktrees()
            .next()
            .expect("JDTLS graph worktree")
            .read(cx)
            .id();
        (worktrees, worktree_id, project.languages().clone())
    });
    cx.read(|cx| TrustedWorktrees::try_get_global(cx).expect("fixture trust"))
        .update(cx, |trust, cx| {
            trust.trust(
                &worktrees,
                collections::HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
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
        "Java",
        FakeLspAdapter {
            name: "jdtls",
            initialization_options: Some(serde_json::json!({
                "settings": {"java": options},
                "bundles": bundles,
            })),
            ..Default::default()
        },
    );
    (project, worktree_id, provider_data)
}

#[cfg(all(test, unix))]
async fn mount_dependencies_panel(
    project: Entity<Project>,
    cx: &mut TestAppContext,
) -> (
    Entity<Workspace>,
    Entity<jvm_ui::DependenciesPanel>,
    &mut VisualTestContext,
) {
    use jvm_ui::DependenciesPanel;

    let (container, cx) =
        cx.add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
    let workspace = container.read_with(cx, |container, _| container.workspace().clone());
    let async_cx = cx.update(|window, cx| window.to_async(cx));
    let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
        .await
        .expect("existing Dependencies panel");
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.add_panel(panel.clone(), window, cx);
        workspace.right_dock().update(cx, |dock, cx| {
            dock.set_open(true, window, cx);
        });
        workspace.focus_panel::<DependenciesPanel>(window, cx);
        window.activate_a11y_for_test();
    });
    (workspace, panel, cx)
}

#[cfg(all(test, unix))]
fn dispatch_graph_action(cx: &mut VisualTestContext, action_name: &str) {
    cx.update(|window, cx| {
        let action = cx.build_action(action_name, None).expect(action_name);
        window.dispatch_action(action, cx);
    });
}

#[cfg(all(test, unix))]
fn focus_control(cx: &mut VisualTestContext, label: &str) {
    assert!(
        (0..48).any(|_| cx.update(|window, cx| {
            window.focus_next(cx);
            window.draw(cx).clear(cx);
            let tree: serde_json::Value = serde_json::from_str(
                &window
                    .debug_a11y_tree_json()
                    .expect("window accessibility tree"),
            )
            .expect("accessibility JSON");
            tree["gpui_focus"]
                .as_str()
                .is_some_and(|focus| tree["nodes"][focus]["aria"]["label"] == label)
        })),
        "keyboard focus reaches {label}"
    );
}

#[cfg(all(test, unix))]
fn press_enter(cx: &mut VisualTestContext) {
    use gpui::InputEvent as _;

    let keystroke = gpui::Keystroke::parse("enter").expect("activation key");
    cx.update(|window, cx| {
        window.dispatch_event(
            gpui::KeyDownEvent {
                keystroke: keystroke.clone(),
                is_held: false,
                prefer_character_input: false,
            }
            .to_platform_input(),
            cx,
        );
        window.dispatch_event(gpui::KeyUpEvent { keystroke }.to_platform_input(), cx);
    });
    cx.run_until_parked();
}

#[cfg(all(test, unix))]
fn type_fixture_options(cx: &mut VisualTestContext, control: &str, editor: &str, options: &str) {
    focus_control(cx, control);
    press_enter(cx);
    for _ in 0..2 {
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
            window.simulate_next_frame(cx);
        });
    }
    cx.simulate_input(options);
    cx.update(|window, cx| {
        window.draw(cx).clear(cx);
        let tree: serde_json::Value =
            serde_json::from_str(&window.debug_a11y_tree_json().expect("Task options editor"))
                .expect("accessibility JSON");
        assert!(
            tree["nodes"]
                .as_object()
                .expect("accessibility nodes")
                .values()
                .any(|node| node["aria"]["label"] == editor && node["aria"]["value"] == options),
            "keyboard input reaches {editor}"
        );
    });
}

#[cfg(all(test, unix))]
async fn mount_fixture_terminal(
    workspace: &Entity<Workspace>,
    cx: &mut VisualTestContext,
) -> Entity<terminal_view::terminal_panel::TerminalPanel> {
    let async_cx = cx.update(|window, cx| window.to_async(cx));
    let terminal_panel =
        terminal_view::terminal_panel::TerminalPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("production terminal panel");
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.add_panel(terminal_panel.clone(), window, cx);
    });
    terminal_panel
}

#[cfg(all(test, unix))]
async fn assert_fixture_task(
    terminal_panel: &Entity<terminal_view::terminal_panel::TerminalPanel>,
    root: &std::path::Path,
    executable: &std::path::Path,
    arguments: &[String],
    cx: &mut VisualTestContext,
) -> String {
    use futures::future::{Either, select};
    use terminal_view::TerminalView;

    let deadline = Instant::now() + Duration::from_secs(30);
    let terminal = loop {
        cx.run_until_parked();
        let terminal = terminal_panel.read_with(cx, |panel, cx| {
            panel.panes().into_iter().find_map(|pane| {
                pane.read(cx).items().find_map(|item| {
                    let view = item.downcast::<TerminalView>()?;
                    let terminal = view.read(cx).terminal();
                    terminal.read(cx).task().map(|_| terminal.clone())
                })
            })
        });
        if let Some(terminal) = terminal {
            break terminal;
        }
        assert!(
            Instant::now() < deadline,
            "the build did not open a Task terminal"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    };
    let completed = terminal.read_with(cx, |terminal, cx| {
        let task = &terminal.task().expect("build Task").spawned_task;
        let (shell, shell_arguments) = task::ShellBuilder::new(&task.shell, false)
            .build_with_command_intent(
                Some(executable.to_str().expect("executable path").to_owned()),
                arguments,
                true,
                arguments.len(),
            );
        assert_eq!(task.command.as_deref(), Some(shell.as_str()));
        assert_eq!(task.args, shell_arguments);
        assert_eq!(task.cwd.as_deref(), Some(root));
        terminal.wait_for_completed_task(cx)
    });
    let result = match select(
        completed,
        cx.background_executor.timer(Duration::from_secs(70)),
    )
    .await
    {
        Either::Left((result, _)) => result.expect("Task exit status"),
        Either::Right(_) => panic!("the build Task did not finish"),
    };
    let transcript = terminal.read_with(cx, |terminal, _| terminal.get_content());
    assert!(result.success(), "the build Task failed: {transcript}");
    transcript
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires a disposable offline Gradle graph, pinned JDTLS and its Gradle delegate"]
async fn cached_gradle_graph_reaches_mounted_dependency_diagram(cx: &mut TestAppContext) {
    run_cached_gradle_fixture(GradleFixtureRun::Diagram, cx).await;
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires the pinned offline Gradle fixture with the ordered Task probes"]
async fn cached_gradle_ordered_operations_keep_declared_options(cx: &mut TestAppContext) {
    run_cached_gradle_fixture(GradleFixtureRun::OrderedTasks, cx).await;
}

#[cfg(all(test, unix))]
enum GradleFixtureRun {
    Diagram,
    OrderedTasks,
}

#[cfg(all(test, unix))]
async fn run_cached_gradle_fixture(run: GradleFixtureRun, cx: &mut TestAppContext) {
    use self::dispatch_graph_action as dispatch;
    use jvm_ui::{
        DependenciesPanel, ShowGradle,
        dependency_analyzer_item::{DependencyAnalyzerItem, ShowInDependencyDiagram},
        dependency_diagram_item::{DependencyDiagramItem, DependencyDiagramPopup},
        dependency_graph_canvas::{CanvasEdgeVariant, MAX_CANVAS_EDGES, MAX_CANVAS_NODES},
    };
    use language_tools::language_tool_tree::{ExpandAll, Refresh, SelectFirst, SelectNext};

    let root = cached_jvm_fixture_path("GROOVY_GRADLE_GRAPH_ROOT");
    let build_java = cached_jvm_fixture_path("GROOVY_BUILD_JAVA_HOME");
    let gradle_installation = cached_jvm_fixture_path("GROOVY_GRADLE_HOME");
    let gradle_user_home = cached_jvm_fixture_path("GRADLE_USER_HOME");
    let options = serde_json::json!({
        "autobuild": {"enabled": false},
        "configuration": {"updateBuildConfiguration": "disabled"},
        "import": {"maven": {"enabled": false}, "gradle": {
            "enabled": true, "offline": {"enabled": true}, "wrapper": {"enabled": false},
            "home": gradle_installation, "java": {"home": build_java},
            "user": {"home": gradle_user_home}, "arguments": ["--offline"],
        }},
    });
    let (project, worktree_id, _provider_data) = start_cached_fixture_jdtls(
        &root,
        options,
        &[cached_jvm_fixture_path("GROOVY_GRADLE_MODEL_JAR")],
        cx,
    )
    .await;
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let (_source, _open_source) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(
                root.join("application/src/main/java/GraphProbe.java"),
                cx,
            )
        })
        .await
        .expect("Gradle module source");
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut refresh_at = Instant::now();
    loop {
        cx.run_until_parked();
        if lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(snapshot, current)| {
                    current
                        && snapshot
                            .projects
                            .iter()
                            .any(|module| module.module_root.path.as_unix_str() == "application")
                })
        }) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "JDTLS did not import the Gradle module"
        );
        if Instant::now() >= refresh_at {
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            refresh_at = Instant::now() + Duration::from_secs(5);
        }
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
    if matches!(run, GradleFixtureRun::OrderedTasks) {
        cx.update(terminal_view::init);
    }
    let (workspace, panel, cx) = mount_dependencies_panel(project, cx).await;
    cx.run_until_parked();
    cx.update(|window, cx| window.draw(cx).clear(cx));
    cx.dispatch_action(ShowGradle);
    focus_control(
        cx,
        "Synchronize local Gradle metadata; executes trusted build configuration scripts",
    );
    press_enter(cx);
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.focus_panel::<DependenciesPanel>(window, cx);
    });
    cx.update(|window, cx| window.draw(cx).clear(cx));
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        cx.run_until_parked();
        cx.dispatch_action(ExpandAll);
        if panel.read_with(cx, |panel, _| {
            let (_, rows, _) = panel.snapshot_for_test();
            rows.iter().any(|row| row.label == "qualificationGraph")
        }) {
            break;
        }
        if Instant::now() >= deadline
            || lsp_store.read_with(cx, |store, _| store.jdtls_project_query_failed(worktree_id))
        {
            let (status, rows, _) = panel.read_with(cx, |panel, _| panel.snapshot_for_test());
            let modules = lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("imported modules");
                (
                    current,
                    snapshot
                        .projects
                        .iter()
                        .map(|module| {
                            (
                                module.module_root.path.to_string(),
                                module.gradle.as_ref().map(|gradle| {
                                    gradle
                                        .configurations
                                        .iter()
                                        .map(|config| config.name.clone())
                                        .collect::<Vec<_>>()
                                }),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            });
            let labels: Vec<_> = rows.iter().map(|row| row.label.as_str()).collect();
            panic!(
                "the Dependencies tree did not receive the real Gradle configuration: {status:?}, {modules:?}, {labels:?}"
            );
        }
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
    cx.dispatch_action(Refresh);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        cx.run_until_parked();
        if lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)
        }) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "Gradle ownership did not refresh"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
    if matches!(run, GradleFixtureRun::OrderedTasks) {
        verify_gradle_queue(
            &root,
            &build_java,
            &gradle_installation,
            &gradle_user_home,
            &workspace,
            &panel,
            cx,
        )
        .await;
        return;
    }
    let select_configuration = |cx: &mut VisualTestContext| {
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(ExpandAll);
        let configuration = panel.read_with(cx, |panel, _| {
            let (_, rows, _) = panel.snapshot_for_test();
            rows.iter()
                .position(|row| row.label == "qualificationGraph")
                .expect("current Gradle configuration")
        });
        cx.dispatch_action(SelectFirst);
        for _ in 0..configuration {
            cx.dispatch_action(SelectNext);
        }
    };
    select_configuration(cx);
    dispatch(cx, "groovy_dependencies_panel::CollectSelectedGraph");
    let deadline = Instant::now() + Duration::from_secs(90);
    let graph = loop {
        cx.run_until_parked();
        if let Some(graph) = lsp_store.read_with(cx, |store, _| {
            let (snapshot, current) = store.jdtls_project_snapshot(worktree_id)?;
            if !current {
                return None;
            }
            snapshot.projects.iter().find_map(|module| {
                let graph = module.gradle.as_ref()?.dependency_graph.as_ref()?;
                (graph.resolution.configuration == "qualificationGraph")
                    .then(|| graph.resolution.clone())
            })
        }) {
            break graph;
        }
        assert!(
            Instant::now() < deadline,
            "the selected offline Gradle graph was not admitted"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    };
    assert_eq!((graph.components.len(), graph.edges.len()), (126, 413));
    assert!(!graph.resolution_complete);
    eprintln!("Gradle Dependencies admitted 126 components and 413 edges");
    let leaf = graph
        .components
        .iter()
        .find(|component| component.module.as_deref() == Some("leaf-12-08"))
        .expect("late transitive leaf");
    let (incoming_index, incoming) = graph
        .edges
        .iter()
        .enumerate()
        .find(|(_, edge)| edge.selected.as_deref() == Some(&leaf.id))
        .expect("actual provider edge to the leaf");
    select_configuration(cx);
    focus_control(
        cx,
        "Show a read-only diagram of current admitted dependency graph facts in a dedicated editor tab; opens or focuses the tab without resolving dependencies",
    );
    press_enter(cx);
    let diagram = workspace
        .read_with(cx, |workspace, cx| {
            workspace.item_of_type::<DependencyDiagramItem>(cx)
        })
        .unwrap_or_else(|| {
            let notice = cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("window tree"))
                        .expect("accessibility JSON");
                tree["nodes"]
                    .as_object()
                    .expect("accessibility nodes")
                    .values()
                    .find(|node| node["aria"]["label"] == "Dependency graph status")
                    .map(|node| node["aria"]["value"].clone())
            });
            panic!("editor diagram from the selected Gradle configuration: {notice:?}")
        });
    diagram.read_with(cx, |diagram, cx| {
        let canvas = diagram.canvas().read(cx);
        assert!(canvas.graph.truncated);
        assert!(canvas.graph.node(&leaf.id).is_none());
    });

    let search_dependency = |cx: &mut VisualTestContext, query: &str, coordinate: &str| {
        focus_control(cx, "Search dependency diagram nodes");
        press_enter(cx);
        for _ in 0..2 {
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                window.simulate_next_frame(cx);
            });
        }
        cx.simulate_input(query);
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
            let tree: serde_json::Value =
                serde_json::from_str(&window.debug_a11y_tree_json().expect("diagram search"))
                    .expect("accessibility JSON");
            let nodes = tree["nodes"].as_object().expect("accessibility nodes");
            assert_eq!(
                nodes
                    .values()
                    .filter(|node| node["aria"]["label"] == coordinate)
                    .count(),
                1,
                "the picker renders one exact dependency",
            );
            assert!(!nodes.values().any(|node| {
                node["aria"]["label"] == "Dependency graph status"
                    && node["aria"]["value"]
                        .as_str()
                        .is_some_and(|status| status.contains("Diagram unavailable"))
            }));
        });
        dispatch(cx, "menu::Confirm");
        cx.run_until_parked();
    };
    search_dependency(cx, "leaf-12-08", "verification:leaf-12-08:1.0");
    diagram.read_with(cx, |diagram, cx| {
        let canvas = diagram.canvas().read(cx);
        assert_eq!(canvas.focus.as_deref(), Some(leaf.id.as_str()));
        assert!(canvas.selection.contains(&leaf.id));
        assert!(canvas.graph.nodes.len() <= MAX_CANVAS_NODES);
        assert!(canvas.graph.edges.len() <= MAX_CANVAS_EDGES);
        assert!(canvas.graph.edges.iter().any(|edge| {
            edge.from == incoming.from
                && edge.to == leaf.id
                && edge.provider_index == incoming_index
                && edge.variant == CanvasEdgeVariant::Dependency
        }));
    });
    eprintln!(
        "Mounted Gradle Dependencies admitted 126 components and 413 edges and searched to the exact provider edge for verification:leaf-12-08:1.0"
    );
    let viewport = diagram.read_with(cx, |diagram, cx| diagram.canvas().read(cx).viewport.clone());
    dispatch(cx, "groovy_dependencies_panel::InspectGraphNeighbours");
    cx.run_until_parked();
    diagram.read_with(cx, |diagram, cx| {
        let canvas = diagram.canvas().read(cx);
        assert!(canvas.inspected.contains(&incoming.from));
        assert!(canvas.inspected_edges.contains(&incoming_index));
    });
    dispatch(cx, "groovy_dependencies_panel::InspectGraphRootPaths");
    cx.run_until_parked();
    let root_edge = graph
        .edges
        .iter()
        .position(|edge| {
            edge.from == graph.root_component
                && edge.selected.as_deref() == Some(incoming.from.as_str())
        })
        .expect("root edge to the leaf's branch");
    diagram.read_with(cx, |diagram, cx| {
        let canvas = diagram.canvas().read(cx);
        let visible_path = canvas
            .graph
            .edges
            .iter()
            .map(|edge| edge.provider_index)
            .filter(|index| [root_edge, incoming_index].contains(index))
            .collect();
        assert_eq!(canvas.inspected_edges, visible_path);
        assert_eq!(canvas.viewport, viewport);
    });
    let inspection_announces = |cx: &mut VisualTestContext, heading: &str, details: &[&str]| {
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
            let tree: serde_json::Value =
                serde_json::from_str(&window.debug_a11y_tree_json().expect("diagram inspection"))
                    .expect("accessibility JSON");
            tree["nodes"]
                .as_object()
                .expect("accessibility nodes")
                .values()
                .any(|node| {
                    node["aria"]["value"].as_str().is_some_and(|value| {
                        value.starts_with(heading)
                            && details.iter().all(|detail| value.contains(detail))
                    })
                })
        })
    };
    assert!(inspection_announces(
        cx,
        "Shortest root paths:",
        &["Shortest distance 2"]
    ));

    let observer = graph
        .components
        .iter()
        .find(|component| component.module.as_deref() == Some("shared-observer"))
        .expect("dependency with 96 root paths");
    search_dependency(cx, "shared-observer", "verification:shared-observer:1.0");
    dispatch(cx, "groovy_dependencies_panel::InspectGraphAllRootPaths");
    cx.run_until_parked();
    assert!(
        inspection_announces(
            cx,
            "All root paths:",
            &["64 bounded simple paths", "Enumeration reached its limit",]
        ),
        "the diagram announces the actual path truncation"
    );
    let generation = lsp_store.read_with(cx, |store, _| {
        store
            .jdtls_project_snapshot(worktree_id)
            .expect("current graph")
            .0
            .generation
    });
    dispatch(cx, "groovy_dependencies_panel::InspectGraphAllUsages");
    cx.run_until_parked();
    let linked_analyzer = workspace
        .read_with(cx, |workspace, cx| {
            workspace.active_item_as::<DependencyAnalyzerItem>(cx)
        })
        .expect("All usages activates the analyzer");
    linked_analyzer.read_with(cx, |analyzer, _| {
        assert_eq!(analyzer.selected_key(), Some(observer.id.as_str()));
        assert_eq!(analyzer.scope().label(), "qualificationGraph");
        assert!(analyzer.usages_truncated());
    });
    cx.dispatch_action(ShowInDependencyDiagram {
        node_key: observer.id.clone(),
    });
    cx.run_until_parked();
    assert_eq!(
        workspace.read_with(cx, |workspace, cx| workspace
            .active_item_as::<DependencyDiagramItem>(cx)),
        Some(diagram.clone()),
    );
    diagram.read_with(cx, |diagram, cx| {
        assert_eq!(
            diagram.canvas().read(cx).focus.as_deref(),
            Some(observer.id.as_str())
        );
    });
    lsp_store.read_with(cx, |store, _| {
        let (snapshot, current) = store
            .jdtls_project_snapshot(worktree_id)
            .expect("current graph");
        assert!(current);
        assert_eq!(snapshot.generation, generation);
    });
    eprintln!(
        "Mounted Gradle checked neighbours, shortest paths, 64 of 96 root paths and round-trip analyzer navigation without refreshing the provider"
    );
    select_configuration(cx);
    dispatch(cx, "groovy_dependencies_panel::ShowDependencyDiagramPopup");
    cx.run_until_parked();
    assert!(workspace.read_with(cx, |workspace, cx| {
        workspace
            .active_modal::<DependencyDiagramPopup>(cx)
            .is_some()
    }));
    cx.simulate_keystrokes("escape");
    assert!(workspace.read_with(cx, |workspace, cx| {
        workspace
            .active_modal::<DependencyDiagramPopup>(cx)
            .is_none()
    }));
    select_configuration(cx);
    dispatch(cx, "groovy_dependencies_panel::AnalyzeDependencies");
    cx.run_until_parked();
    let analyzer = workspace
        .read_with(cx, |workspace, cx| {
            workspace.item_of_type::<DependencyAnalyzerItem>(cx)
        })
        .expect("analyzer from the selected Gradle configuration");
    analyzer.read_with(cx, |analyzer, _| {
        assert_eq!(analyzer.scope().label(), "qualificationGraph");
    });
    eprintln!("Mounted Gradle opened the diagram popup and analyzer from the same admitted scope");
}

#[cfg(all(test, unix))]
async fn verify_gradle_queue(
    root: &std::path::Path,
    build_java: &std::path::Path,
    gradle_installation: &std::path::Path,
    gradle_user_home: &std::path::Path,
    workspace: &Entity<Workspace>,
    panel: &Entity<jvm_ui::DependenciesPanel>,
    cx: &mut VisualTestContext,
) {
    use jvm_ui::DependenciesPanel;
    use language_tools::language_tool_tree::{ExpandAll, SelectFirst, SelectNext};

    const WRITE: &str = ":application:qualificationQueueWrite";
    const READ: &str = ":application:qualificationQueueRead";
    const MESSAGE: &str = "value $ZED_FILE :library:help";
    let output = root.join("application/build/qualification/queue-message.txt");
    assert!(
        !output.exists(),
        "the disposable fixture has no Task output"
    );
    let terminal_panel = mount_fixture_terminal(workspace, cx).await;

    let select_task = |cx: &mut VisualTestContext, path| {
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(ExpandAll);
        let task_row = panel.read_with(cx, |panel, _| {
            let (_, rows, _) = panel.snapshot_for_test();
            let (index, row) = rows
                .iter()
                .enumerate()
                .find(|(_, row)| row.label == path && row.id.0.contains(":gradle-task:"))
                .unwrap_or_else(|| panic!("missing provider Gradle task {path}"));
            assert!(
                row.secondary_label
                    .as_deref()
                    .is_some_and(|label| label.contains("run offline through Tasks"))
            );
            assert!(row.accessibility_label.contains("or Task options"));
            if path == WRITE {
                assert!(
                    rows.iter().any(|option| {
                        option.label == "--project-dir"
                            && option.id.0.contains(":gradle-option:")
                            && option
                                .id
                                .0
                                .ends_with(":application:qualificationQueueWrite:project-dir")
                    }),
                    "the option is declared by the Gradle provider"
                );
            }
            index
        });
        cx.dispatch_action(SelectFirst);
        for _ in 0..task_row {
            cx.dispatch_action(SelectNext);
        }
        cx.update(|window, cx| window.draw(cx).clear(cx));
    };
    select_task(cx, WRITE);
    let option = format!("--project-dir={MESSAGE}");
    let options = serde_json::to_string(&[&option]).expect("literal Gradle option JSON");
    type_fixture_options(
        cx,
        "Configure declared options for the selected Gradle task",
        "Gradle task options",
        &options,
    );
    dispatch_graph_action(cx, "groovy_dependencies_panel::AddSelectedOrderedOperation");
    select_task(cx, READ);
    dispatch_graph_action(cx, "groovy_dependencies_panel::AddSelectedOrderedOperation");
    dispatch_graph_action(cx, "groovy_dependencies_panel::ShowOrderedOperations");
    let queue = panel.read_with(cx, |panel, _| {
        panel
            .snapshot_for_test()
            .1
            .into_iter()
            .filter(|row| row.id.0.starts_with("groovy-ordered-operation:"))
            .map(|row| row.label.clone())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        queue,
        [format!("1. {WRITE} (1 options)"), format!("2. {READ}")]
    );
    assert!(
        !output.exists(),
        "queueing operations does not start a Task"
    );
    dispatch_graph_action(cx, "groovy_dependencies_panel::RunOrderedOfflineOperations");
    let arguments = [
        "--offline".to_owned(),
        format!("-Dorg.gradle.java.home={}", build_java.display()),
        "--gradle-user-home".to_owned(),
        gradle_user_home
            .to_str()
            .expect("Gradle home path")
            .to_owned(),
        "--".to_owned(),
        WRITE.to_owned(),
        option,
        READ.to_owned(),
    ];
    let transcript = assert_fixture_task(
        &terminal_panel,
        root,
        &gradle_installation.join("bin/gradle"),
        &arguments,
        cx,
    )
    .await;
    let write = transcript
        .find("QUALIFICATION_QUEUE_WRITE")
        .expect("Gradle write task");
    let read = transcript
        .find(&format!("QUALIFICATION_QUEUE_READ={MESSAGE}"))
        .expect("Gradle read task saw the exact option");
    assert!(write < read, "{transcript}");
    assert_eq!(
        std::fs::read_to_string(output).expect("Gradle output"),
        MESSAGE
    );
    eprintln!(
        "Mounted Gradle ran both tasks in order and preserved the literal option in one offline Task"
    );
}

#[cfg(all(test, unix))]
async fn wait_for_maven_module(
    lsp_store: &Entity<project::LspStore>,
    worktree_id: WorktreeId,
    pom: &str,
    ready: impl Fn(&project::jvm_project::MavenProject) -> bool,
    cx: &mut TestAppContext,
) {
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut refresh_at = Instant::now();
    loop {
        cx.run_until_parked();
        if lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(snapshot, current)| {
                    current
                        && snapshot.projects.iter().any(|module| {
                            module.maven.as_ref().is_some_and(|maven| {
                                maven.pom.path.as_unix_str() == pom && ready(maven)
                            })
                        })
                })
        }) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "JDTLS did not import the Maven module"
        );
        if Instant::now() >= refresh_at {
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            refresh_at = Instant::now() + Duration::from_secs(5);
        }
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
}

#[cfg(all(test, unix))]
async fn show_ready_maven_view(
    lsp_store: &Entity<project::LspStore>,
    worktree_id: WorktreeId,
    cx: &mut VisualTestContext,
) {
    use jvm_ui::ShowMaven;
    use language_tools::language_tool_tree::Refresh;

    cx.update(|window, cx| window.draw(cx).clear(cx));
    cx.dispatch_action(ShowMaven);
    cx.run_until_parked();
    cx.dispatch_action(Refresh);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        cx.run_until_parked();
        if lsp_store.read_with(cx, |store, _| {
            !store.jdtls_project_query_pending(worktree_id)
                && store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)
        }) {
            break;
        }
        assert!(Instant::now() < deadline, "Maven ownership did not refresh");
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    }
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires a disposable offline Maven graph, pinned JDTLS and its Maven provider"]
async fn cached_maven_graph_reaches_mounted_dependency_analyzer(cx: &mut TestAppContext) {
    use jvm_ui::{
        DependenciesPanel,
        dependency_analyzer_item::{
            DependencyAnalyzerFilter, DependencyAnalyzerItem, DependencyAnalyzerPresentation,
            SelectDependencyAnalyzerFilter, ShowInDependencyDiagram,
            ToggleDependencyAnalyzerPresentation,
        },
        dependency_diagram_item::{DependencyDiagramItem, SelectDependencyDiagramScope},
        dependency_graph_canvas::{CanvasEdgeVariant, CanvasNodeKind},
    };
    use language_tools::language_tool_tree::{ExpandAll, SelectFirst, SelectNext};

    let root = cached_jvm_fixture_path("GROOVY_MAVEN_GRAPH_ROOT");
    let settings = cached_jvm_fixture_path("GROOVY_MAVEN_GRAPH_SETTINGS");
    let options = serde_json::json!({
        "autobuild": {"enabled": false},
        "configuration": {"maven": {"userSettings": settings}},
        "import": {"gradle": {"enabled": false}, "maven": {
            "enabled": true, "offline": {"enabled": true},
        }},
    });
    let (project, worktree_id, _provider_data) = start_cached_fixture_jdtls(
        &root,
        options,
        &[cached_jvm_fixture_path("GROOVY_MAVEN_MODEL_JAR")],
        cx,
    )
    .await;
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let (_source, _open_source) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(
                root.join("application/src/main/java/evidence/GraphProbe.java"),
                cx,
            )
        })
        .await
        .expect("Maven module source");
    let pom = "application/pom.xml";
    wait_for_maven_module(&lsp_store, worktree_id, pom, |_| true, cx).await;
    let (workspace, panel, cx) = mount_dependencies_panel(project, cx).await;
    show_ready_maven_view(&lsp_store, worktree_id, cx).await;
    let select_application = |cx: &mut VisualTestContext| {
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(ExpandAll);
        let row = panel.read_with(cx, |panel, _| {
            let (status, rows, navigation) = panel.snapshot_for_test();
            rows.iter()
                .position(|row| {
                    row.is_branch
                        && row.id.0.starts_with("jdtls:")
                        && navigation.get(&row.id).is_some_and(|path| {
                            path.worktree_id == worktree_id && path.path.as_unix_str() == pom
                        })
                })
                .unwrap_or_else(|| panic!("imported Maven application is selectable: {status:?}"))
        });
        cx.dispatch_action(SelectFirst);
        for _ in 0..row {
            cx.dispatch_action(SelectNext);
        }
        cx.update(|window, cx| window.draw(cx).clear(cx));
    };
    select_application(cx);
    assert!(!lsp_store.read_with(cx, |store, _| {
        store.jdtls_project_query_pending(worktree_id)
    }));
    dispatch_graph_action(cx, "groovy_dependencies_panel::CollectSelectedGraph");
    if !lsp_store.read_with(cx, |store, _| {
        store.jdtls_project_query_pending(worktree_id)
    }) {
        let status = cx.update(|window, cx| {
            window.draw(cx).clear(cx);
            let tree: serde_json::Value =
                serde_json::from_str(&window.debug_a11y_tree_json().expect("Dependencies tree"))
                    .expect("accessibility JSON");
            tree["nodes"]
                .as_object()
                .expect("accessibility nodes")
                .values()
                .find(|node| node["aria"]["label"] == "Dependency graph status")
                .map(|node| node["aria"]["value"].clone())
        });
        panic!("the Maven graph action did not start a provider request: {status:?}");
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    let (graph, generation) = loop {
        cx.run_until_parked();
        if let Some(graph) = lsp_store.read_with(cx, |store, _| {
            let (snapshot, current) = store.jdtls_project_snapshot(worktree_id)?;
            if !current {
                return None;
            }
            snapshot.projects.iter().find_map(|module| {
                let maven = module.maven.as_ref()?;
                if maven.pom.path.as_unix_str() != pom {
                    return None;
                }
                Some((maven.dependency_graph.clone()?, snapshot.generation))
            })
        }) {
            break graph;
        }
        assert!(
            Instant::now() < deadline
                && !lsp_store
                    .read_with(cx, |store, _| store.jdtls_project_query_failed(worktree_id)),
            "the selected offline Maven graph was not admitted"
        );
        cx.background_executor
            .timer(Duration::from_millis(100))
            .await;
    };
    assert_eq!((graph.components.len(), graph.edges.len()), (7, 6));
    assert!(graph.offline && graph.cached_only && graph.occurrences_known);
    let component = |name: &str, version: &str| {
        graph
            .components
            .iter()
            .find(|component| component.artifact_id == name && component.version == version)
            .expect("fixture component")
    };
    let winner = graph
        .edges
        .iter()
        .find(|edge| edge.component == component("shared", "1").id)
        .expect("first requested version");
    let loser = graph
        .edges
        .iter()
        .find(|edge| edge.component == component("shared", "2").id)
        .expect("second requested version");
    assert!(!winner.omitted && !winner.direct);
    assert!(loser.omitted && !loser.direct);
    assert_ne!(winner.target_node, loser.target_node);
    assert_eq!(loser.selected_node, winner.target_node);
    let missing = component("uncached", "1");
    let missing_edge = graph
        .edges
        .iter()
        .find(|edge| edge.component == missing.id)
        .expect("edge to the occurrence with an unavailable descriptor");
    let test_edge = graph
        .edges
        .iter()
        .find(|edge| edge.component == component("test-only", "1").id)
        .expect("test scope occurrence");
    assert_eq!(test_edge.scope, "test");
    assert!(!graph.collection_complete && !graph.structural_coverage_known);
    assert!(
        graph
            .unavailable_descriptors
            .iter()
            .any(|fact| fact.component == missing.id)
    );

    select_application(cx);
    dispatch_graph_action(cx, "groovy_dependencies_panel::ShowDependencyDiagram");
    cx.run_until_parked();
    let diagram = workspace
        .read_with(cx, |workspace, cx| {
            workspace.active_item_as::<DependencyDiagramItem>(cx)
        })
        .expect("diagram from the selected Maven project");
    let canvas = diagram.read_with(cx, |diagram, _| {
        assert_eq!(diagram.scope().label(), "All scopes");
        diagram.canvas().clone()
    });
    canvas.update(cx, |canvas, _| {
        assert!(canvas.graph.edges.iter().any(|edge| {
            edge.from == loser.from_node
                && edge.to == loser.target_node
                && edge.variant == CanvasEdgeVariant::Omitted
        }));
        assert!(canvas.graph.nodes.iter().any(|node| {
            node.component_id == missing.id && node.kind == CanvasNodeKind::Unresolved
        }));
        assert!(canvas.graph.edges.iter().any(|edge| {
            edge.from == missing_edge.from_node
                && edge.to == missing_edge.target_node
                && edge.variant == CanvasEdgeVariant::Unresolved
        }));
        assert!(
            canvas
                .graph
                .edges
                .iter()
                .any(|edge| edge.to == test_edge.target_node)
        );
    });
    cx.dispatch_action(SelectDependencyDiagramScope {
        label: "compile".to_owned(),
    });
    cx.run_until_parked();
    diagram.read_with(cx, |diagram, _| {
        assert_eq!(diagram.scope().label(), "compile")
    });
    canvas.update(cx, |canvas, _| {
        assert!(
            !canvas
                .graph
                .edges
                .iter()
                .any(|edge| edge.to == test_edge.target_node)
        );
        assert!(canvas.select_node(&winner.target_node, false, false));
    });
    dispatch_graph_action(cx, "groovy_dependencies_panel::InspectGraphAllUsages");
    cx.run_until_parked();
    let analyzer = workspace
        .read_with(cx, |workspace, cx| {
            workspace.active_item_as::<DependencyAnalyzerItem>(cx)
        })
        .expect("Maven All usages opens the analyzer");
    analyzer.read_with(cx, |analyzer, _| {
        assert_eq!(analyzer.scope().label(), "compile");
        assert_eq!(analyzer.selected_key(), Some(winner.target_node.as_str()));
        assert!(!analyzer.usages_truncated());
    });
    cx.dispatch_action(ShowInDependencyDiagram {
        node_key: winner.target_node.clone(),
    });
    cx.run_until_parked();
    assert_eq!(
        workspace.read_with(cx, |workspace, cx| {
            workspace.active_item_as::<DependencyDiagramItem>(cx)
        }),
        Some(diagram.clone())
    );
    diagram.read_with(cx, |diagram, _| {
        assert_eq!(diagram.scope().label(), "compile")
    });
    assert_eq!(
        canvas.read_with(cx, |canvas, _| canvas.focus.clone()),
        Some(winner.target_node.clone())
    );

    canvas.update(cx, |canvas, _| {
        assert!(canvas.select_node(&missing_edge.target_node, false, false));
    });
    dispatch_graph_action(cx, "groovy_dependencies_panel::InspectGraphAllUsages");
    cx.run_until_parked();
    analyzer.read_with(cx, |analyzer, _| {
        assert_eq!(
            analyzer.selected_key(),
            Some(missing_edge.target_node.as_str())
        );
    });
    cx.update(|window, cx| window.draw(cx).clear(cx));
    cx.dispatch_action(ToggleDependencyAnalyzerPresentation);
    analyzer.read_with(cx, |analyzer, _| {
        assert_eq!(
            analyzer.presentation(),
            DependencyAnalyzerPresentation::Flat
        );
    });
    for (filter, expected) in [
        (
            DependencyAnalyzerFilter::All,
            graph
                .edges
                .iter()
                .filter(|edge| edge.scope == "compile")
                .map(|edge| &edge.target_node)
                .collect(),
        ),
        (
            DependencyAnalyzerFilter::Conflicts,
            vec![&winner.target_node, &loser.target_node],
        ),
        (
            DependencyAnalyzerFilter::Unresolved,
            vec![&missing_edge.target_node],
        ),
    ] {
        cx.dispatch_action(SelectDependencyAnalyzerFilter { filter });
        cx.run_until_parked();
        analyzer.read_with(cx, |analyzer, _| assert_eq!(analyzer.filter(), filter));
        cx.dispatch_action(SelectFirst);
        let mut selected = Vec::new();
        for _ in &expected {
            selected.push(analyzer.read_with(cx, |analyzer, _| {
                analyzer
                    .selected_key()
                    .expect("filtered occurrence")
                    .to_owned()
            }));
            cx.dispatch_action(SelectNext);
        }
        assert_eq!(
            analyzer.read_with(cx, |analyzer, _| analyzer.selected_key().map(str::to_owned)),
            selected.last().cloned(),
            "the {filter:?} filter ends after its matching occurrences"
        );
        selected.sort_unstable();
        let mut expected = expected.into_iter().cloned().collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(selected, expected);
    }
    lsp_store.read_with(cx, |store, _| {
        let (snapshot, current) = store
            .jdtls_project_snapshot(worktree_id)
            .expect("admitted graph");
        assert!(current);
        assert_eq!(snapshot.generation, generation);
    });
    eprintln!(
        "Mounted Maven checked mediation and missing descriptors, diagram–analyzer navigation, and conflict and unresolved filters"
    );
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires a cached Maven Help goal, pinned Maven wrapper and pinned JDTLS"]
async fn cached_effective_pom_runs_from_maven_view_through_tasks(cx: &mut TestAppContext) {
    run_cached_maven_goals(MavenGoalRun::EffectivePom, cx).await;
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires cached Maven Help goals, pinned Maven wrapper and pinned JDTLS"]
async fn cached_maven_help_goals_run_in_queue_order_with_task_options(cx: &mut TestAppContext) {
    run_cached_maven_goals(MavenGoalRun::OrderedHelp, cx).await;
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires a cached Maven Dependency goal, pinned Maven wrapper and pinned JDTLS"]
async fn cached_maven_sources_are_reported_in_task_terminal(cx: &mut TestAppContext) {
    run_cached_maven_goals(MavenGoalRun::ResolveSources { cached: true }, cx).await;
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires an offline Maven fixture with its source archive omitted"]
async fn missing_maven_sources_are_reported_in_successful_task(cx: &mut TestAppContext) {
    run_cached_maven_goals(MavenGoalRun::ResolveSources { cached: false }, cx).await;
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires a cached Maven Javadoc archive, pinned Maven wrapper and pinned JDTLS"]
async fn cached_maven_javadoc_is_reported_in_task_terminal(cx: &mut TestAppContext) {
    run_cached_maven_goals(MavenGoalRun::ResolveJavadoc, cx).await;
}

#[cfg(all(test, unix))]
#[gpui::test]
#[ignore = "requires a populated Maven repository, pinned Maven wrapper and pinned JDTLS"]
async fn cached_maven_repository_readiness_is_reported_in_task_terminal(cx: &mut TestAppContext) {
    run_cached_maven_goals(MavenGoalRun::CheckOfflineRepository, cx).await;
}

#[cfg(all(test, unix))]
enum MavenGoalRun {
    EffectivePom,
    OrderedHelp,
    ResolveSources { cached: bool },
    ResolveJavadoc,
    CheckOfflineRepository,
}

#[cfg(all(test, unix))]
async fn run_cached_maven_goals(run: MavenGoalRun, cx: &mut TestAppContext) {
    use jvm_ui::DependenciesPanel;
    use language_tools::language_tool_tree::{ExpandAll, SelectFirst, SelectNext};

    let (plugin, version, root_key, settings_key, source, profile) = match run {
        MavenGoalRun::ResolveSources { .. }
        | MavenGoalRun::ResolveJavadoc
        | MavenGoalRun::CheckOfflineRepository => (
            "maven-dependency-plugin",
            "3.10.0",
            "GROOVY_MAVEN_SOURCE_ROOT",
            "GROOVY_MAVEN_SOURCE_SETTINGS",
            "src/main/java/evidence/ArtifactProbe.java",
            None,
        ),
        _ => (
            "maven-help-plugin",
            "3.5.2",
            "GROOVY_EFFECTIVE_POM_ROOT",
            "GROOVY_EFFECTIVE_POM_SETTINGS",
            "src/main/java/evidence/EffectivePomProbe.java",
            Some("evidence-selected"),
        ),
    };
    let goals: &[&str] = match run {
        MavenGoalRun::EffectivePom => &["effective-pom"],
        MavenGoalRun::OrderedHelp => &["active-profiles", "effective-pom"],
        MavenGoalRun::ResolveSources { .. } => &["resolve-sources"],
        MavenGoalRun::ResolveJavadoc => &["resolve"],
        MavenGoalRun::CheckOfflineRepository => &["go-offline"],
    };
    let coordinates: Vec<_> = goals
        .iter()
        .map(|goal| format!("org.apache.maven.plugins:{plugin}:{version}:{goal}"))
        .collect();
    let root = cached_jvm_fixture_path(root_key);
    let settings = cached_jvm_fixture_path(settings_key);
    let options = serde_json::json!({
        "autobuild": {"enabled": false},
        "configuration": {"maven": {"userSettings": settings}},
        "import": {"gradle": {"enabled": false}, "maven": {
            "enabled": true, "offline": {"enabled": true},
        }},
    });
    let (project, worktree_id, _provider_data) = start_cached_fixture_jdtls(
        &root,
        options,
        &[cached_jvm_fixture_path("GROOVY_MAVEN_MODEL_JAR")],
        cx,
    )
    .await;
    cx.update(terminal_view::init);
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let (_source, _open_source) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(root.join(source), cx)
        })
        .await
        .expect("Maven module source");
    wait_for_maven_module(
        &lsp_store,
        worktree_id,
        "pom.xml",
        |maven| {
            profile.is_none_or(|profile| maven.profiles.get(profile) == Some(&true))
                && maven.plugins.iter().any(|candidate| {
                    candidate.group_id == "org.apache.maven.plugins"
                        && candidate.artifact_id == plugin
                        && candidate.version.as_deref() == Some(version)
                        && !candidate.managed_only
                        && candidate.goals_known
                        && goals
                            .iter()
                            .all(|name| candidate.goals.iter().any(|goal| goal.name == *name))
                })
        },
        cx,
    )
    .await;

    let (workspace, panel, cx) = mount_dependencies_panel(project, cx).await;
    let terminal_panel = mount_fixture_terminal(&workspace, cx).await;
    show_ready_maven_view(&lsp_store, worktree_id, cx).await;
    cx.dispatch_action(ExpandAll);
    let select_goal = |cx: &mut VisualTestContext, goal| {
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        let goal_row = panel.read_with(cx, |panel, _| {
            let (status, rows, _) = panel.snapshot_for_test();
            let (index, row) = rows
                .iter()
                .enumerate()
                .find(|(_, row)| row.label == goal && row.id.0.contains(":maven:goal:"))
                .unwrap_or_else(|| panic!("the {goal} goal is selectable: {status:?}"));
            assert!(
                row.secondary_label.as_deref().is_some_and(|label| {
                    label.contains("offline through Tasks")
                        && !label.contains("execution unavailable")
                }),
                "the visible Maven goal description matches execution availability"
            );
            assert!(row.accessibility_label.contains("or Task options"));
            let guidance = match goal {
                "resolve-sources" => Some(
                    "Check resolved sources in the Task terminal even if Maven succeeds.",
                ),
                "resolve" => Some(
                    "For Javadoc, use Task options to activate a profile with the javadoc classifier and jar type.",
                ),
                "go-offline" => Some(
                    "Check cached dependencies, plugins and reports in the Task terminal.",
                ),
                _ => None,
            };
            if let Some(guidance) = guidance {
                assert!(row.accessibility_label.contains(guidance));
                assert!(
                    row.secondary_label
                        .as_deref()
                        .is_some_and(|label| label.contains(guidance))
                );
            }
            index
        });
        cx.dispatch_action(SelectFirst);
        for _ in 0..goal_row {
            cx.dispatch_action(SelectNext);
        }
        cx.update(|window, cx| window.draw(cx).clear(cx));
    };
    let output = root.join("target/groovy-effective.xml");
    if matches!(run, MavenGoalRun::EffectivePom | MavenGoalRun::OrderedHelp) {
        assert!(
            !output.exists(),
            "the fixture starts without an effective model"
        );
    }
    match run {
        MavenGoalRun::OrderedHelp => {
            select_goal(cx, "active-profiles");
            type_fixture_options(
                cx,
                "Configure profiles, test settings and reactor behavior for the selected Maven operation",
                "Maven operation options",
                r#"["-P!evidence-selected"]"#,
            );
            dispatch_graph_action(cx, "groovy_dependencies_panel::AddSelectedOrderedOperation");
            select_goal(cx, "effective-pom");
            dispatch_graph_action(cx, "groovy_dependencies_panel::AddSelectedOrderedOperation");
            dispatch_graph_action(cx, "groovy_dependencies_panel::ShowOrderedOperations");
            let queue_labels = |cx: &VisualTestContext| {
                panel.read_with(cx, |panel, _| {
                    panel
                        .snapshot_for_test()
                        .1
                        .into_iter()
                        .filter(|row| row.id.0.starts_with("groovy-ordered-operation:"))
                        .map(|row| row.label.clone())
                        .collect::<Vec<_>>()
                })
            };
            let [first, second] = coordinates.as_slice() else {
                panic!("the ordered fixture has two goals");
            };
            let expected_queue = [
                format!("1. {first} (1 Maven options)"),
                format!("2. {second}"),
            ];
            assert_eq!(queue_labels(cx), expected_queue);
            dispatch_graph_action(cx, "groovy_dependencies_panel::RemoveLastOrderedOperation");
            assert_eq!(queue_labels(cx), expected_queue[..1]);
            select_goal(cx, "effective-pom");
            dispatch_graph_action(cx, "groovy_dependencies_panel::AddSelectedOrderedOperation");
            assert_eq!(queue_labels(cx), expected_queue);
            assert!(!output.exists(), "editing the queue does not start a Task");
            dispatch_graph_action(cx, "groovy_dependencies_panel::RunOrderedOfflineOperations");
        }
        _ => {
            let [goal] = goals else {
                panic!("the fixture has exactly one goal");
            };
            select_goal(cx, goal);
            if matches!(run, MavenGoalRun::ResolveJavadoc) {
                type_fixture_options(
                    cx,
                    "Configure profiles, test settings and reactor behavior for the selected Maven operation",
                    "Maven operation options",
                    r#"["-Partifact-javadocs"]"#,
                );
            }
            dispatch_graph_action(cx, "groovy_dependencies_panel::RunSelectedOfflineOperation");
        }
    }
    let mut arguments = [
        "--offline",
        "--settings",
        settings.to_str().expect("settings path"),
        "--file",
        "pom.xml",
    ]
    .map(str::to_owned)
    .to_vec();
    let profile_option = match run {
        MavenGoalRun::OrderedHelp => Some("!evidence-selected"),
        MavenGoalRun::ResolveJavadoc => Some("artifact-javadocs"),
        _ => None,
    };
    if let Some(profile) = profile_option {
        arguments.extend(["--activate-profiles", profile].map(str::to_owned));
    }
    arguments.extend(coordinates);
    let transcript =
        assert_fixture_task(&terminal_panel, &root, &root.join("mvnw"), &arguments, cx).await;
    match run {
        MavenGoalRun::ResolveSources { cached } => {
            let (_, artifacts) = transcript
                .split_once("The following files have been resolved:")
                .expect("Maven's resolved artifact report");
            let (resolved, unresolved) = artifacts
                .split_once("The following files have NOT been resolved:")
                .expect("Maven's unresolved artifact report");
            assert!(transcript.contains("BUILD SUCCESS"));
            let source = "org.apache.commons:commons-lang3:jar:sources:3.17.0";
            if cached {
                assert!(resolved.contains(source), "{transcript}");
            } else {
                assert!(resolved.contains("none"), "{transcript}");
                assert!(!transcript.contains(source), "{transcript}");
                assert!(
                    unresolved.contains("org.apache.commons:commons-lang3:jar:3.17.0:compile"),
                    "{transcript}"
                );
            }
            eprintln!(
                "Mounted Maven reported cached sources: {cached}, with a successful offline Task"
            );
        }
        MavenGoalRun::ResolveJavadoc => {
            let (_, artifacts) = transcript
                .split_once("The following files have been resolved:")
                .expect("Maven's resolved artifact report");
            let (resolved, _) = artifacts
                .split_once("The following files have NOT been resolved:")
                .expect("Maven's unresolved artifact report");
            assert!(resolved.contains("org.apache.commons:commons-lang3:jar:javadoc:3.17.0"));
            assert!(transcript.contains("BUILD SUCCESS"));
        }
        MavenGoalRun::CheckOfflineRepository => {
            assert!(transcript.contains("Resolved plugin: maven-dependency-plugin-3.10.0.jar"));
            assert!(transcript.contains("Resolved dependency: commons-lang3-3.17.0.jar"));
            assert!(transcript.contains("BUILD SUCCESS"));
        }
        MavenGoalRun::EffectivePom | MavenGoalRun::OrderedHelp => {
            let effective = std::fs::read_to_string(output).expect("generated effective POM");
            let (_, properties) = effective
                .split_once("<properties>")
                .expect("effective properties");
            let (properties, _) = properties
                .split_once("</properties>")
                .expect("end effective properties");
            if matches!(run, MavenGoalRun::OrderedHelp) {
                assert!(properties.contains("<effective.fixture>base</effective.fixture>"));
                let profile_report = transcript
                    .find("Active profile report written to:")
                    .expect("active profiles ran in Maven");
                let effective_pom = transcript
                    .find("Effective-POM written to:")
                    .expect("effective POM ran in Maven");
                assert!(profile_report < effective_pom);
                eprintln!(
                    "Mounted Maven ran active-profiles before effective-pom in one offline Task and applied the profile override"
                );
            } else {
                assert!(properties.contains("<effective.fixture>selected</effective.fixture>"));
                eprintln!(
                    "Mounted Maven ran the pinned Effective POM goal through the offline Task terminal with the selected settings profile"
                );
            }
        }
    }
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

pub async fn run_cached_selected_java_debug(
    cx: &mut TestAppContext,
    build_tool: JvmTestBuildTool,
    hooks: JvmWorkflowDebugHooks,
) {
    assert!(matches!(
        hooks,
        JvmWorkflowDebugHooks::SelectedJavaTest { .. }
    ));
    selected_java_case_with_debug_hooks(cx, build_tool, JavaTestOutcome::Pass, false, Some(hooks))
        .await;
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
    let debug_selected = matches!(
        debug_hooks,
        Some(JvmWorkflowDebugHooks::SelectedJavaTest { .. })
    );
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
    if let Some(
        JvmWorkflowDebugHooks::JavaMain { initialize, .. }
        | JvmWorkflowDebugHooks::SelectedJavaTest { initialize, .. },
    ) = debug_hooks.as_ref()
    {
        cx.update(initialize);
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
    let project_root = if with_real_providers || debug_selected {
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
    let provider_state = (with_real_providers || debug_selected)
        .then(|| tempfile::tempdir().expect("isolated JVM provider state"));
    let mut java_settings = serde_json::json!({
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
        if use_gradle {
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
        } else {
            let repository =
                std::env::var("GROOVY_MAVEN_REPOSITORY").expect("explicit primed Maven repository");
            assert!(Path::new(&repository).is_absolute() && Path::new(&repository).is_dir());
            let settings_file = state.path().join("maven-settings.xml");
            let repository_xml = repository.replace('&', "&amp;").replace('<', "&lt;");
            std::fs::write(
                &settings_file,
                format!("<settings><localRepository>{repository_xml}</localRepository><offline>true</offline></settings>"),
            )
            .expect("isolated JDTLS Maven settings");
            std::fs::write(
                root.path().join(".mvn/maven.config"),
                format!("--offline\n-Dmaven.repo.local={repository}\n"),
            )
            .expect("isolated offline Maven wrapper");
            java_settings = serde_json::json!({
                "import": {"maven": {"enabled": true, "offline": {"enabled": true}}},
                "configuration": {"maven": {"userSettings": settings_file}},
            });
        }
        let mut servers = serde_json::Map::new();
        for (name, variable) in [
            ("jdtls", "GROOVY_JDTLS_LAUNCHER"),
            ("kotlin-lsp", "GROOVY_KOTLIN_LSP"),
        ]
        .into_iter()
        .filter(|(name, _)| *name == "jdtls" || with_real_providers)
        {
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
    if debug_selected {
        let path = root
            .path()
            .join("application/src/test/java/evidence/GreetingTest.java");
        let source = std::fs::read_to_string(&path).expect("selected Java test source");
        let (body, suffix) = source.rsplit_once('}').expect("fixture class body");
        std::fs::write(
            path,
            format!("{body}\n    @Test\n    void unselectedTest() {{\n        throw new AssertionError(\"Unselected test ran\");\n    }}\n}}{suffix}"),
        )
        .expect("add a second test that must remain unselected");
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
    if debug_selected {
        let debug_jar = std::env::var("GROOVY_JAVA_DEBUG_JAR").expect("verified Java Debug bundle");
        assert!(Path::new(&debug_jar).is_absolute() && Path::new(&debug_jar).is_file());
        languages.register_fake_lsp_adapter(
            "Java",
            language::FakeLspAdapter {
                name: "jdtls",
                initialization_options: Some(serde_json::json!({
                    "settings": {"java": &java_settings},
                    "bundles": [debug_jar],
                })),
                ..Default::default()
            },
        );
    }

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
    let initial_selection = selection.clone();

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
    if let Some(JvmWorkflowDebugHooks::SelectedJavaTest { run, .. }) = debug_hooks.as_ref() {
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(tests_panel.clone(), window, cx);
            workspace.focus_panel::<TestsPanel>(window, cx);
        });
        tests_panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.refresh(cx);
        });
        cx.run_until_parked();
        tests_panel.update(cx, |panel, cx| {
            let id = find_node(&panel.projection.snapshot.roots, case_label)
                .expect("selected Java case in Tests")
                .id
                .clone();
            assert!(panel.projection.can_run(&id));
            panel.host.expand_all();
            panel.host.select(id);
            cx.notify();
        });
        run(
            JvmWorkflowDebugContext {
                project,
                workspace,
                project_root,
                worktree_id,
            },
            cx,
        )
        .await;
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
        return;
    }
    tests_panel.update(cx, |panel, cx| {
        panel.active = true;
        if with_real_providers {
            panel.discover(cx);
        } else {
            panel.refresh(cx);
        }
    });
    cx.run_until_parked();
    if use_ssh || with_real_providers {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            let ready = tests_panel.read_with(cx, |panel, _| {
                let Some(case) = find_node(&panel.projection.snapshot.roots, case_label) else {
                    return false;
                };
                !with_real_providers
                    || panel
                        .projection
                        .selection(&case.id)
                        .is_some_and(|selection| {
                            selection.node_id == initial_selection.node_id
                                && selection.discovery_generation
                                    != initial_selection.discovery_generation
                                && panel.projection.can_run(&case.id)
                        })
            });
            if ready {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "Tests did not publish a current runnable Java case"
            );
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
        if with_real_providers {
            assert!(panel.action_notice.is_none(), "{:?}", panel.action_notice);
        }
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
        let active_debug = if let Some(JvmWorkflowDebugHooks::JavaMain { run, .. }) = debug_hooks {
            Some(
                run(
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
