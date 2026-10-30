use crate::graph_tests::fixture_model;
use crate::properties_panel::{
    GraphNodePropertyKind, graph_node_properties_with_value, graph_node_property_descriptors,
    openrouter_cost_notice, parse_graph_node_property_value,
};
use crate::{GraphPropertiesPanel, GraphWorkspaceItem, open_for_graph_node};
use gpui::{AppContext as _, Focusable as _, TestAppContext};
use project::{FakeFs, Project};
use sim_runtime::{GraphIdentifier, GraphNode, GraphNodeMode, GraphPaletteColor, GraphPoint};
use workspace::{
    MultiWorkspace, Workspace,
    dock::{DockPosition, Panel as _},
};

fn init_properties_test(cx: &mut TestAppContext) {
    let file_system = FakeFs::new(cx.executor());
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        let settings_store = settings::SettingsStore::test(cx);
        cx.set_global(settings_store);
        <dyn fs::Fs>::set_global(file_system, cx);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        workspace::AppState::test(cx);
        crate::init(cx);
    });
    cx.run_until_parked();
}

#[test]
fn openrouter_properties_disclose_approved_estimate_and_separate_budget() {
    assert!(openrouter_cost_notice("ClaudeNode").is_none());
    let notice = openrouter_cost_notice("OpenRouterLLMNode").expect("OpenRouter disclosure");
    assert!(notice.contains("approval of a USD cost estimate for each run"));
    assert!(notice.contains("Actual charges may exceed the estimate"));
    assert!(notice.contains("key or account budget is a separate backstop"));
    assert!(!notice.contains("guaranteed"));
}

#[gpui::test(seed = 16024)]
async fn properties_panel_is_a_right_dock_bound_to_canonical_graph_commands(
    cx: &mut TestAppContext,
) {
    init_properties_test(cx);
    let file_system = FakeFs::new(cx.executor());
    let project = Project::test(file_system, [], cx).await;
    let (workspace, window) =
        cx.add_window_view(|window, cx| Workspace::test_new(project, window, cx));
    let (graph, panel) = workspace.update_in(window, |workspace, window, cx| {
        let workspace_handle = cx.entity().downgrade();
        let graph = cx.new(|cx| {
            GraphWorkspaceItem::new(
                fixture_model().expect("create native graph properties fixture"),
                workspace_handle.clone(),
                cx,
            )
        });
        workspace.add_item_to_active_pane(Box::new(graph.clone()), None, true, window, cx);
        let panel =
            cx.new(|cx| GraphPropertiesPanel::test_new(workspace_handle.clone(), window, cx));
        workspace.add_panel(panel.clone(), window, cx);
        (graph, panel)
    });
    window.run_until_parked();
    workspace.update_in(window, |workspace, window, cx| {
        open_for_graph_node(
            workspace,
            graph.clone(),
            GraphIdentifier::from("source"),
            window,
            cx,
        )
        .expect("open authoritative native graph properties panel");
    });
    window.run_until_parked();

    workspace.update_in(window, |workspace, window, cx| {
        assert_eq!(panel.read(cx).position(window, cx), DockPosition::Right);
        assert!(workspace.right_dock().read(cx).is_open());
        assert!(panel.focus_handle(cx).contains_focused(window, cx));
    });
    assert_eq!(
        panel.read_with(window, |panel, _| panel.target_for_test()),
        Some(GraphIdentifier::from("source"))
    );

    panel.update_in(window, |panel, window, cx| {
        panel.set_title_for_test("Production Source", window, cx);
        panel.set_mode_for_test(GraphNodeMode::OnTrigger, cx);
        panel.set_color_for_test(Some(GraphPaletteColor::Blue), cx);
        panel.set_properties_for_test(r#"{"priority":7,"owner":"runtime"}"#, window, cx);
    });
    window.run_until_parked();
    graph.read_with(window, |graph, _| {
        let node = graph
            .model()
            .document()
            .expect("editable properties graph")
            .active_graph()
            .expect("active properties graph")
            .nodes
            .get(&GraphIdentifier::from("source"))
            .expect("properties target remains present");
        assert_eq!(node.title, "Production Source");
        assert_eq!(node.mode, GraphNodeMode::OnTrigger);
        assert_eq!(
            node.color.as_deref(),
            Some(GraphPaletteColor::Blue.node_header())
        );
        assert_eq!(
            node.source_fields
                .get("properties")
                .and_then(serde_json::Value::as_object)
                .and_then(|properties| properties.get("priority")),
            Some(&serde_json::Value::from(7))
        );
    });

    assert!(panel.update(window, |panel, cx| panel.delete_for_test(cx)));
    window.run_until_parked();
    graph.read_with(window, |graph, _| {
        assert!(
            !graph
                .model()
                .document()
                .expect("editable properties graph")
                .active_graph()
                .expect("active properties graph")
                .nodes
                .contains_key(&GraphIdentifier::from("source"))
        );
    });
    assert!(!panel.update(window, |panel, cx| panel.delete_for_test(cx)));
    assert!(panel.read_with(window, |panel, cx| {
        panel
            .availability_for_test(cx)
            .is_some_and(|message| message.contains("no longer exists"))
    }));

    graph.update(window, |graph, cx| {
        assert!(graph.model.undo().expect("undo canonical node deletion"));
        cx.notify();
    });
    window.run_until_parked();
    assert!(graph.read_with(window, |graph, _| {
        graph
            .model()
            .document()
            .expect("editable properties graph")
            .active_graph()
            .expect("active properties graph")
            .nodes
            .contains_key(&GraphIdentifier::from("source"))
    }));
}

#[gpui::test]
async fn properties_toggle_and_dock_activation_follow_without_reentrant_workspace_reads(
    cx: &mut TestAppContext,
) {
    init_properties_test(cx);
    let project = Project::test(FakeFs::new(cx.executor()), [], cx).await;
    let (multi_workspace, window) =
        cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
    let workspace = multi_workspace.read_with(window, |multi_workspace, _| {
        multi_workspace.workspace().clone()
    });
    let panel = workspace.update_in(window, |workspace, window, cx| {
        let workspace_handle = cx.entity().downgrade();
        let graph = cx.new(|cx| {
            GraphWorkspaceItem::new(
                fixture_model().expect("graph"),
                workspace_handle.clone(),
                cx,
            )
        });
        workspace.add_item_to_active_pane(Box::new(graph), None, true, window, cx);
        let panel = cx.new(|cx| GraphPropertiesPanel::test_new(workspace_handle, window, cx));
        workspace.add_panel(panel.clone(), window, cx);
        panel
    });
    workspace.update_in(window, |workspace, window, cx| {
        workspace.focus_center_pane(window, cx);
        cx.notify();
    });
    window.run_until_parked();
    window.dispatch_action(crate::actions::ToggleGraphPropertiesPanel);
    window.run_until_parked();
    workspace.update_in(window, |workspace, window, cx| {
        assert!(workspace.right_dock().read(cx).is_open());
        assert!(panel.focus_handle(cx).contains_focused(window, cx));
        workspace.toggle_panel_focus::<GraphPropertiesPanel>(window, cx);
        workspace.reveal_panel::<GraphPropertiesPanel>(window, cx);
    });
    window.run_until_parked();
    assert!(panel.read_with(window, |panel, _| {
        panel
            .status_message_for_test()
            .is_some_and(|message| message.contains("Select exactly one node"))
    }));
}

#[test]
fn zed_loads_properties_panel_through_the_workspace_panel_owner() {
    const ZED_SOURCE: &str = include_str!("../../zed/src/zed.rs");
    assert!(ZED_SOURCE.contains("sim_ui::GraphPropertiesPanel::load"));
    let normalized = ZED_SOURCE.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(normalized.contains("add_panel_when_ready( graph_properties_panel,"));
}

#[test]
fn typed_property_adapter_is_the_shared_bounded_properties_info_owner() {
    let mut node = GraphNode::new(
        GraphIdentifier::from("typed-properties"),
        "NativeFixture",
        "Typed properties",
        GraphPoint::ZERO,
    );
    node.source_fields.insert(
        "properties".to_owned(),
        serde_json::json!({
            "enabled": true,
            "quality": "balanced",
            "scheduler": 0,
            "inferred_bool": false,
            "inferred_number": 3.5,
            "inferred_text": "native",
            "untyped": {"preserved": true}
        }),
    );
    node.source_fields.insert(
        "properties_info".to_owned(),
        serde_json::json!([
            {"name":"enabled","label":"Enabled","type":"boolean","default_value":false},
            {
                "name":"quality",
                "label":"Quality",
                "type":"combo",
                "values":["fast","balanced","high"]
            },
            {
                "name":"scheduler",
                "label":"Scheduler",
                "type":"enum",
                "values":{"Normal":0,"High":1}
            }
        ]),
    );

    let descriptors = graph_node_property_descriptors(&node)
        .expect("parse bounded native node property descriptors");
    assert_eq!(descriptors.len(), 7);
    assert_eq!(descriptors[0].key, "enabled");
    assert_eq!(descriptors[0].kind, GraphNodePropertyKind::Boolean);
    assert_eq!(descriptors[1].key, "quality");
    assert!(matches!(
        descriptors[1].kind,
        GraphNodePropertyKind::Choice { .. }
    ));
    assert_eq!(descriptors[2].key, "scheduler");
    assert!(matches!(
        descriptors[2].kind,
        GraphNodePropertyKind::Choice { .. }
    ));
    assert_eq!(
        descriptors
            .iter()
            .find(|descriptor| descriptor.key == "inferred_bool")
            .map(|descriptor| &descriptor.kind),
        Some(&GraphNodePropertyKind::Boolean)
    );
    assert_eq!(
        descriptors
            .iter()
            .find(|descriptor| descriptor.key == "inferred_number")
            .map(|descriptor| &descriptor.kind),
        Some(&GraphNodePropertyKind::Number)
    );
    assert_eq!(
        descriptors
            .iter()
            .find(|descriptor| descriptor.key == "inferred_text")
            .map(|descriptor| &descriptor.kind),
        Some(&GraphNodePropertyKind::Text)
    );
    assert_eq!(
        descriptors
            .iter()
            .find(|descriptor| descriptor.key == "untyped")
            .map(|descriptor| &descriptor.kind),
        Some(&GraphNodePropertyKind::Json)
    );
    let parsed = parse_graph_node_property_value(&descriptors[1], "high")
        .expect("parse declared combo value");
    assert_eq!(parsed, serde_json::Value::String("high".to_owned()));
    assert!(parse_graph_node_property_value(&descriptors[1], "unsupported").is_err());
    assert_eq!(
        parse_graph_node_property_value(&descriptors[2], "High")
            .expect("parse object-mapped enum label"),
        serde_json::json!(1)
    );
    let updated = graph_node_properties_with_value(&node, "quality", parsed)
        .expect("update typed property through shared adapter");
    assert_eq!(updated.get("quality"), Some(&serde_json::json!("high")));
    assert!(
        graph_node_properties_with_value(&node, "quality", serde_json::json!("unsupported"))
            .is_err()
    );

    let mut duplicate_value_node = node;
    duplicate_value_node.source_fields.insert(
        "properties_info".to_owned(),
        serde_json::json!([
            {"name":"scheduler","type":"enum","values":{"One":1,"Also One":1}}
        ]),
    );
    assert!(graph_node_property_descriptors(&duplicate_value_node).is_err());
}

fn image_import_test_node() -> sim_runtime::GraphNode {
    let mut node = GraphNode::new(
        GraphIdentifier::from("source"),
        "LoadImage",
        "Load Image",
        GraphPoint::default(),
    );
    node.widgets.push(sim_runtime::GraphWidget {
        identifier: "image".into(),
        kind: sim_runtime::GraphWidgetKind::Text { multiline: false },
        value: serde_json::Value::String("before.png".into()),
        prompt_value: serde_json::Value::String("before.png".into()),
        validation: sim_runtime::WidgetValidation::Valid,
        converted_to_input: false,
        visible: true,
        unknown: Default::default(),
    });
    node
}

#[test]
fn image_import_requires_the_canonical_load_image_upload_widget() {
    use crate::properties_panel::load_image_import_widget;
    let node = image_import_test_node();
    assert_eq!(load_image_import_widget(&node), Some("image"));
    for change in [
        "class",
        "name",
        "converted",
        "hidden",
        "fixed-combo",
        "preserved",
    ] {
        let mut candidate = node.clone();
        let widget = candidate.widgets.first_mut().expect("image widget");
        match change {
            "class" => candidate.type_identifier = "OtherImageLoader".into(),
            "name" => widget.identifier = "other".into(),
            "converted" => widget.converted_to_input = true,
            "hidden" => widget.visible = false,
            "fixed-combo" => {
                widget.kind = sim_runtime::GraphWidgetKind::Combo {
                    values: vec!["before.png".into()],
                    dynamic: false,
                }
            }
            "preserved" => {
                widget.kind = sim_runtime::GraphWidgetKind::Preserved {
                    schema: serde_json::json!({"image_upload": true}),
                };
                widget.value = serde_json::Value::Null;
            }
            _ => unreachable!(),
        }
        assert_eq!(load_image_import_widget(&candidate), None, "{change}");
    }
    let mut preserved = node.clone();
    preserved.widgets = vec![sim_runtime::GraphWidget::preserved(
        "image",
        serde_json::Value::String("before.png".into()),
    )];
    assert_eq!(load_image_import_widget(&preserved), Some("image"));
    preserved
        .widgets
        .first_mut()
        .expect("image widget")
        .prompt_value = serde_json::Value::Null;
    assert_eq!(load_image_import_widget(&preserved), None);
    let mut dynamic = node;
    dynamic.widgets.first_mut().expect("image widget").kind = sim_runtime::GraphWidgetKind::Combo {
        values: Vec::new(),
        dynamic: true,
    };
    assert_eq!(load_image_import_widget(&dynamic), Some("image"));
}

#[test]
fn image_import_reader_bounds_bytes_and_propagates_cancellation_and_read_errors()
-> anyhow::Result<()> {
    use crate::properties_panel::read_bounded_image_import;
    use sim_types::CancellationToken;
    use std::io::{Cursor, Read};
    let cancellation = CancellationToken::default();
    assert_eq!(
        read_bounded_image_import(Cursor::new(b"abcd"), 4, &cancellation)?,
        b"abcd"
    );
    assert!(read_bounded_image_import(Cursor::new(b"abcde"), 4, &cancellation).is_err());
    assert!(read_bounded_image_import(Cursor::new([]), u64::MAX, &cancellation).is_err());
    struct InterruptedReader;
    impl Read for InterruptedReader {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "source denied",
            ))
        }
    }
    assert!(
        read_bounded_image_import(InterruptedReader, 4, &cancellation)
            .expect_err("read error")
            .to_string()
            .contains("source denied")
    );
    Ok(())
}

#[test]
fn image_import_reader_observes_its_own_token_between_read_chunks() {
    use crate::properties_panel::read_bounded_image_import;
    use sim_types::CancellationToken;
    use std::io::Read;
    struct CancellingReader(CancellationToken);
    impl Read for CancellingReader {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            self.0.cancel();
            if let Some(byte) = bytes.first_mut() {
                *byte = 1;
                Ok(1)
            } else {
                Ok(0)
            }
        }
    }
    let cancellation = CancellationToken::default();
    assert!(
        read_bounded_image_import(CancellingReader(cancellation.clone()), 4, &cancellation)
            .is_err()
    );
    assert!(cancellation.is_cancelled());
    assert!(!CancellationToken::default().is_cancelled());
}

fn image_import_test_assets() -> anyhow::Result<(tempfile::TempDir, sim_runtime::SharedAssetService)>
{
    use sim_runtime::{AssetNamespace, AssetRoots, AssetService};
    let directory = tempfile::tempdir()?;
    let mut roots = Vec::new();
    for (namespace, name) in [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ] {
        let path = directory.path().join(name);
        std::fs::create_dir(&path)?;
        roots.push((namespace, path));
    }
    let roots = AssetRoots::new(crate::LOCAL_EXECUTION_PROFILE_ID.0.to_string(), roots)?;
    Ok((
        directory,
        std::sync::Arc::new(std::sync::Mutex::new(AssetService::open(roots)?)),
    ))
}

fn image_import_test_model() -> anyhow::Result<crate::GraphWorkspaceModel> {
    let fixture = fixture_model().map_err(|error| anyhow::anyhow!("{error}"))?;
    let mut document = fixture
        .document()
        .ok_or_else(|| anyhow::anyhow!("fixture document"))?
        .clone();
    let source = document
        .root
        .nodes
        .get_mut(&GraphIdentifier::from("source"))
        .ok_or_else(|| anyhow::anyhow!("fixture source"))?;
    source.type_identifier = "LoadImage".into();
    source.widgets = vec![sim_runtime::GraphWidget::preserved(
        "image",
        serde_json::Value::String("before.png".into()),
    )];
    document.profile_identity = Some(crate::LOCAL_EXECUTION_PROFILE_ID.0);
    Ok(crate::GraphWorkspaceModel::open(
        "Image import",
        document.document_identity.to_string(),
        sim_runtime::WorkflowStorageProvider::Draft,
        document.to_workflow_bytes()?,
    )?)
}

#[gpui::test(seed = 16025)]
async fn image_import_completion_updates_canonical_widget_with_one_undo(cx: &mut TestAppContext) {
    init_properties_test(cx);
    let (_directory, assets) = image_import_test_assets().expect("canonical image assets");
    cx.update(|cx| {
        crate::register_native_asset_services(assets, cx).expect("register native importer")
    });
    let file_system = FakeFs::new(cx.executor());
    let bytes = sim_media::encode_png_frame(
        &[1.0, 0.0, 0.0],
        1,
        1,
        1,
        3,
        0,
        &Default::default(),
        sim_media::PngLimits::default(),
    )
    .expect("PNG");
    file_system
        .insert_tree("/selected", serde_json::json!({}))
        .await;
    file_system.insert_file("/selected/actual.png", bytes).await;
    cx.update(|cx| <dyn fs::Fs>::set_global(file_system.clone(), cx));
    let project = Project::test(file_system, [], cx).await;
    let (workspace, window) =
        cx.add_window_view(|window, cx| Workspace::test_new(project, window, cx));
    let (graph, panel) = workspace.update_in(window, |workspace, window, cx| {
        let workspace_handle = cx.entity().downgrade();
        let graph = cx.new(|cx| {
            GraphWorkspaceItem::new(
                image_import_test_model().expect("image graph"),
                workspace_handle.clone(),
                cx,
            )
        });
        workspace.add_item_to_active_pane(Box::new(graph.clone()), None, true, window, cx);
        let panel = cx.new(|cx| GraphPropertiesPanel::test_new(workspace_handle, window, cx));
        workspace.add_panel(panel.clone(), window, cx);
        open_for_graph_node(
            workspace,
            graph.clone(),
            GraphIdentifier::from("source"),
            window,
            cx,
        )
        .expect("bind image properties");
        (graph, panel)
    });
    window.run_until_parked();
    panel.update_in(window, |panel, window, cx| {
        panel.start_image_import_for_test(window, cx)
    });
    assert!(window.did_prompt_for_paths());
    window.simulate_path_prompt_response(|options| {
        assert!(options.files && !options.directories && !options.multiple);
        Some(vec![std::path::PathBuf::from("/selected/actual.png")])
    });
    window.run_until_parked();
    graph.read_with(window, |graph, _| {
        let widget = &graph
            .model()
            .document()
            .expect("document")
            .root
            .nodes
            .get(&GraphIdentifier::from("source"))
            .expect("node")
            .widgets
            .first()
            .expect("image widget");
        assert_eq!(widget.value, serde_json::Value::String("actual.png".into()));
        assert_eq!(widget.value, widget.prompt_value);
        assert!(graph.model().engine().expect("engine").can_undo());
        let saved = graph
            .model()
            .document()
            .expect("document")
            .to_workflow_bytes()
            .expect("serialize imported graph");
        let restored = sim_runtime::GraphDocument::from_workflow_bytes(&saved)
            .expect("restore imported graph");
        let restored_widget = restored
            .root
            .nodes
            .get(&GraphIdentifier::from("source"))
            .expect("restored source")
            .widgets
            .first()
            .expect("restored image widget");
        assert_eq!(restored_widget.value, widget.value);
        assert_eq!(restored_widget.prompt_value, widget.prompt_value);
        let workflow =
            sim_runtime::WorkflowFormatDocument::parse(&saved).expect("canonical saved workflow");
        let descriptors =
            sim_runtime::native_image_frontend_descriptors().expect("canonical image descriptors");
        let submission = sim_runtime::graph_to_prompt(&workflow, &descriptors, "test")
            .expect("project saved graph to execution prompt");
        assert_eq!(
            submission
                .prompt
                .0
                .get(&sim_types::NodeId("source".into()))
                .expect("prompt source")
                .inputs
                .get("image"),
            Some(&serde_json::Value::String("actual.png".into()))
        );
    });
    graph.update_in(window, |graph, window, cx| {
        graph.graph_undo(&crate::GraphUndo, window, cx)
    });
    window.run_until_parked();
    graph.read_with(window, |graph, _| {
        let widget = &graph
            .model()
            .document()
            .expect("document")
            .root
            .nodes
            .get(&GraphIdentifier::from("source"))
            .expect("node")
            .widgets
            .first()
            .expect("image widget");
        assert_eq!(widget.value, serde_json::Value::String("before.png".into()));
        assert_eq!(widget.value, widget.prompt_value);
        assert!(!graph.model().engine().expect("engine").can_undo());
    });
    panel.update_in(window, |panel, window, cx| {
        panel.start_image_import_for_test(window, cx)
    });
    window.simulate_path_prompt_response(|_| None);
    window.run_until_parked();
    assert_eq!(
        panel.read_with(window, |panel, _| panel
            .status_message_for_test()
            .map(str::to_owned)),
        Some("Image selection cancelled.".into())
    );
    assert!(!graph.read_with(window, |graph, _| {
        graph.model().engine().expect("engine").can_undo()
    }));
    panel.update_in(window, |panel, window, cx| {
        panel.start_image_import_for_test(window, cx);
        assert!(panel.has_image_import_task_for_test());
        panel.cancel_image_import_for_test(cx);
        assert!(!panel.has_image_import_task_for_test());
    });
    window.run_until_parked();
    window.simulate_path_prompt_response(|_| {
        Some(vec![std::path::PathBuf::from("/selected/actual.png")])
    });
    window.run_until_parked();
    assert_eq!(
        panel.read_with(window, |panel, _| panel
            .status_message_for_test()
            .map(str::to_owned)),
        Some("Image import cancelled.".into())
    );
    assert!(!graph.read_with(window, |graph, _| {
        graph.model().engine().expect("engine").can_undo()
    }));
}

#[gpui::test(seed = 16026)]
async fn image_import_stale_revision_rebinding_cancel_and_errors_preserve_graph(
    cx: &mut TestAppContext,
) {
    init_properties_test(cx);
    let (_directory, assets) = image_import_test_assets().expect("canonical assets");
    cx.update(|cx| {
        crate::register_native_asset_services(assets.clone(), cx).expect("register importer")
    });
    let file_system = FakeFs::new(cx.executor());
    let project = Project::test(file_system, [], cx).await;
    let (workspace, window) =
        cx.add_window_view(|window, cx| Workspace::test_new(project, window, cx));
    let (graph, panel) = workspace.update_in(window, |workspace, window, cx| {
        let workspace_handle = cx.entity().downgrade();
        let graph = cx.new(|cx| {
            GraphWorkspaceItem::new(
                image_import_test_model().expect("image graph"),
                workspace_handle.clone(),
                cx,
            )
        });
        workspace.add_item_to_active_pane(Box::new(graph.clone()), None, true, window, cx);
        let panel = cx.new(|cx| GraphPropertiesPanel::test_new(workspace_handle, window, cx));
        workspace.add_panel(panel.clone(), window, cx);
        open_for_graph_node(
            workspace,
            graph.clone(),
            GraphIdentifier::from("source"),
            window,
            cx,
        )
        .expect("bind image");
        (graph, panel)
    });
    window.run_until_parked();
    let (_, revision_token) = panel.update(window, |panel, cx| {
        panel
            .stage_image_import_for_test(cx)
            .expect("stage revision")
    });
    graph.update(window, |graph, cx| {
        assert!(graph.apply_graph_command(
            sim_runtime::GraphCommand::RenameNode {
                identifier: GraphIdentifier::from("source"),
                title: "Changed while importing".into(),
            },
            cx
        ))
    });
    window.run_until_parked();
    assert!(revision_token.is_cancelled());
    let (_, generation_token) = panel.update(window, |panel, cx| {
        panel
            .stage_image_import_for_test(cx)
            .expect("stage generation")
    });
    window.update(|_, cx| {
        crate::register_native_asset_services(assets.clone(), cx).expect("rebind importer")
    });
    window.run_until_parked();
    assert!(generation_token.is_cancelled());
    let (identifier, token) = panel.update(window, |panel, cx| {
        panel.stage_image_import_for_test(cx).expect("stage cancel")
    });
    panel.update(window, |panel, cx| {
        panel.cancel_image_import_for_test(cx);
        panel.finish_image_import_for_test(identifier, Err(anyhow::anyhow!("late failure")), cx);
        assert_eq!(
            panel.status_message_for_test(),
            Some("Image import cancelled.")
        );
    });
    assert!(token.is_cancelled());
    let (identifier, _) = panel.update(window, |panel, cx| {
        panel.stage_image_import_for_test(cx).expect("stage error")
    });
    panel.update(window, |panel, cx| {
        panel.finish_image_import_for_test(identifier, Err(anyhow::anyhow!("read denied")), cx);
        assert!(
            panel
                .status_message_for_test()
                .expect("visible error")
                .contains("read denied")
        );
    });
    let (identifier, _) = panel.update(window, |panel, cx| {
        panel
            .stage_image_import_for_test(cx)
            .expect("stage cancelled chooser")
    });
    panel.update(window, |panel, cx| {
        panel.finish_image_import_for_test(identifier, Ok(None), cx)
    });
    graph.read_with(window, |graph, _| {
        assert_eq!(
            graph
                .model()
                .document()
                .expect("document")
                .root
                .nodes
                .get(&GraphIdentifier::from("source"))
                .expect("node")
                .widgets
                .first()
                .expect("image widget")
                .value,
            serde_json::Value::String("before.png".into())
        );
    });
}
