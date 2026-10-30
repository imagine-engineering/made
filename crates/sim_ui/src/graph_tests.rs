use crate::*;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use gpui::{
    ClipboardEntry, ClipboardItem, ExternalPaths, Image, ImageFormat, KeyBinding, Modifiers,
    MouseButton, ScrollDelta, ScrollWheelEvent, TestAppContext, TouchPhase, VisualContext as _,
    WeakEntity, point, px, size,
};
use project::{FakeFs, Fs as _, Project, RemoveOptions};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_runtime::{
    CatalogGraphAction, GraphClipboard, GraphCommand, GraphCommandEngine, GraphDocument,
    GraphIdentifier, GraphNode, GraphPoint, GraphPort, GraphPortType, GraphRect, GraphReroute,
    GraphSelection, GraphSize, GraphViewport, GraphWidget, GraphWidgetKind, GroupToggle,
    SelectionMode, WidgetValidation, WorkflowAuthority, WorkflowSaveCoordinator,
    WorkflowStorageProvider,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use uuid::Uuid;
use workspace::Item as _;

fn fixture_node(
    identifier: &str,
    position: GraphPoint,
    input: Option<GraphPortType>,
    output: Option<GraphPortType>,
) -> GraphNode {
    let mut node = GraphNode::new(
        GraphIdentifier::from(identifier),
        "NativeFixture",
        identifier,
        position,
    );
    if let Some(input) = input {
        node.inputs.push(GraphPort::new("input", input));
    }
    if let Some(output) = output {
        node.outputs.push(GraphPort::new("output", output));
    }
    node
}

fn fixture_widget() -> GraphWidget {
    GraphWidget {
        identifier: "steps".to_owned(),
        kind: GraphWidgetKind::Integer {
            minimum: 1,
            maximum: 100,
            step: 1,
        },
        value: Value::from(20),
        prompt_value: Value::from(20),
        validation: WidgetValidation::Valid,
        converted_to_input: false,
        visible: true,
        unknown: BTreeMap::new(),
    }
}

pub(crate) fn fixture_model() -> Result<GraphWorkspaceModel, Box<dyn Error>> {
    let mut document = GraphDocument::default();
    document.document_identity = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1001);
    document.profile_identity = Some(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1002));
    let image = GraphPortType::Concrete("IMAGE".to_owned());
    let mut source = fixture_node(
        "source",
        GraphPoint { x: 100.0, y: 100.0 },
        None,
        Some(image.clone()),
    );
    source.widgets.push(fixture_widget());
    source.widgets.extend([
        GraphWidget {
            identifier: "enabled".to_owned(),
            kind: GraphWidgetKind::Boolean,
            value: Value::Bool(false),
            prompt_value: Value::Bool(false),
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        },
        GraphWidget {
            identifier: "strength".to_owned(),
            kind: GraphWidgetKind::Float {
                minimum: 0.0,
                maximum: 1.0,
                step: 0.25,
            },
            value: Value::from(0.5),
            prompt_value: Value::from(0.5),
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        },
        GraphWidget {
            identifier: "mode".to_owned(),
            kind: GraphWidgetKind::Combo {
                values: vec!["fast".to_owned(), "quality".to_owned()],
                dynamic: false,
            },
            value: Value::String("fast".to_owned()),
            prompt_value: Value::String("fast".to_owned()),
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        },
        GraphWidget {
            identifier: "label".to_owned(),
            kind: GraphWidgetKind::Text { multiline: false },
            value: Value::String("native".to_owned()),
            prompt_value: Value::String("native".to_owned()),
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        },
        GraphWidget::preserved("extension-widget", json!({"legacy": true})),
    ]);
    source.size.height = 300.0;
    let mut target = fixture_node(
        "target",
        GraphPoint { x: 450.0, y: 100.0 },
        Some(image),
        None,
    );
    if let Some(input) = target.inputs.first_mut() {
        input.dynamic = true;
    }
    document
        .root
        .nodes
        .insert(source.identifier.clone(), source);
    document
        .root
        .nodes
        .insert(target.identifier.clone(), target);
    document.root.selection.nodes = BTreeSet::from([
        GraphIdentifier::from("source"),
        GraphIdentifier::from("target"),
    ]);
    let engine = GraphCommandEngine::new(document)?;
    let bytes = engine.document.to_workflow_bytes()?;
    let save_coordinator =
        WorkflowSaveCoordinator::new("gpui-fixture", WorkflowStorageProvider::Draft, bytes)?;
    Ok(GraphWorkspaceModel {
        schema_version: GRAPH_WORKSPACE_SCHEMA_VERSION,
        title: "Native graph fixture".to_owned(),
        open_state: WorkflowOpenState::Editable(engine),
        save_coordinator,
        execution_association: None,
        canvas_info_visible: false,
        last_error: None,
        operation_errors: Vec::new(),
        announcement: None,
    })
}

fn clipboard_payload(model: &GraphWorkspaceModel) -> Result<Vec<u8>, Box<dyn Error>> {
    let clipboard = GraphClipboard::copy(model.document().ok_or("editable fixture")?)?;
    let json = String::from_utf8(clipboard.encode()?)?;
    Ok(format!("{GRAPH_CLIPBOARD_MEDIA_TYPE}\n{json}").into_bytes())
}

#[test]
fn graph_workspace_no_op_edits_preserve_save_and_history_state() -> Result<(), Box<dyn Error>> {
    let mut model = fixture_model()?;
    let selection = model
        .document()
        .and_then(|document| document.active_graph().ok())
        .ok_or("active graph")?
        .selection
        .clone();
    let document_before = model.document().ok_or("editable document")?.clone();
    let journal_before = model.save_coordinator.encode()?;

    let changed = model.apply_with_change(GraphCommand::SetSelection {
        selection,
        mode: SelectionMode::Replace,
    })?;

    assert!(!changed);
    assert_eq!(model.document(), Some(&document_before));
    assert_eq!(model.save_coordinator.encode()?, journal_before);
    let WorkflowOpenState::Editable(engine) = &model.open_state else {
        return Err("fixture unexpectedly became read-only".into());
    };
    assert!(!engine.can_undo());
    assert!(!engine.can_redo());
    Ok(())
}

fn converted_fixture(
    mut model: GraphWorkspaceModel,
) -> Result<(GraphWorkspaceModel, GraphIdentifier, GraphIdentifier), Box<dyn Error>> {
    GraphCommandModel::execute(
        &mut model,
        CatalogGraphAction::ConvertToSubgraph,
        GraphActionInput::SubgraphName("Catalog fixture".to_owned()),
    )?;
    let graph = model
        .document()
        .and_then(|document| document.active_graph().ok())
        .ok_or("converted graph")?;
    let instance = graph
        .selection
        .nodes
        .iter()
        .next()
        .cloned()
        .ok_or("converted instance")?;
    let definition = graph
        .nodes
        .get(&instance)
        .and_then(|node| node.subgraph_definition.clone())
        .ok_or("converted definition")?;
    Ok((model, instance, definition))
}

fn grouped_fixture(
    mut model: GraphWorkspaceModel,
) -> Result<(GraphWorkspaceModel, GraphIdentifier), Box<dyn Error>> {
    GraphCommandModel::execute(
        &mut model,
        CatalogGraphAction::GroupSelectedNodes,
        GraphActionInput::Group {
            title: "Catalog group".to_owned(),
            padding: 20.0,
        },
    )?;
    let identifier = model
        .document()
        .and_then(|document| document.active_graph().ok())
        .and_then(|graph| graph.groups.keys().next().cloned())
        .ok_or("catalog group")?;
    Ok((model, identifier))
}

fn catalog_case(action: CatalogGraphAction) -> Result<Value, Box<dyn Error>> {
    let mut model = fixture_model()?;
    let input = match action {
        CatalogGraphAction::PasteFromClipboard
        | CatalogGraphAction::PasteFromClipboardWithConnect => {
            model.apply(GraphCommand::SetSelection {
                selection: GraphSelection {
                    nodes: BTreeSet::from([GraphIdentifier::from("target")]),
                    ..GraphSelection::default()
                },
                mode: SelectionMode::Replace,
            })?;
            let bytes = clipboard_payload(&model)?;
            model.apply(GraphCommand::SetSelection {
                selection: GraphSelection {
                    nodes: BTreeSet::from([
                        GraphIdentifier::from("source"),
                        GraphIdentifier::from("target"),
                    ]),
                    ..GraphSelection::default()
                },
                mode: SelectionMode::Replace,
            })?;
            GraphActionInput::Paste {
                bytes,
                offset: GraphPoint { x: 40.0, y: 40.0 },
                connect_from: (action == CatalogGraphAction::PasteFromClipboardWithConnect)
                    .then(|| (GraphIdentifier::from("source"), 0)),
            }
        }
        CatalogGraphAction::Resize => GraphActionInput::Resize {
            identifier: GraphIdentifier::from("source"),
            size: GraphSize {
                width: 320.0,
                height: 180.0,
            },
        },
        CatalogGraphAction::ConvertToSubgraph => {
            GraphActionInput::SubgraphName("Native catalog subgraph".to_owned())
        }
        CatalogGraphAction::EditSubgraphWidgets | CatalogGraphAction::ToggleWidgetPromotion => {
            GraphActionInput::WidgetPromotion {
                node: GraphIdentifier::from("source"),
                widget: "steps".to_owned(),
                promoted: true,
            }
        }
        CatalogGraphAction::ExitSubgraph => {
            let (converted, _, definition) = converted_fixture(model)?;
            model = converted;
            model.apply(GraphCommand::OpenSubgraph {
                definition_identifier: definition,
            })?;
            GraphActionInput::None
        }
        CatalogGraphAction::FitGroupToContents => {
            let (grouped, identifier) = grouped_fixture(model)?;
            model = grouped;
            GraphActionInput::FitGroup {
                identifier,
                padding: 24.0,
            }
        }
        CatalogGraphAction::GroupSelectedNodes => GraphActionInput::Group {
            title: "Native action group".to_owned(),
            padding: 24.0,
        },
        CatalogGraphAction::UnpackSubgraph => {
            let (converted, instance, _) = converted_fixture(model)?;
            model = converted;
            GraphActionInput::SubgraphInstance(instance)
        }
        CatalogGraphAction::PublishSubgraph => GraphActionInput::None,
        CatalogGraphAction::RefreshNodeDefinitions => GraphActionInput::ReconcileNode {
            identifier: GraphIdentifier::from("source"),
            inputs: Vec::new(),
            outputs: vec![GraphPort::new(
                "output",
                GraphPortType::Concrete("IMAGE".to_owned()),
            )],
            widgets: model
                .document()
                .and_then(|document| document.active_graph().ok())
                .and_then(|graph| graph.nodes.get(&GraphIdentifier::from("source")))
                .map(|node| node.widgets.clone())
                .ok_or("source widgets for reconciliation")?,
            confirm_discard: false,
        },
        CatalogGraphAction::SetSubgraphDescription => {
            let (converted, _, definition) = converted_fixture(model)?;
            model = converted;
            GraphActionInput::SubgraphDescription {
                definition,
                description: "A native catalog subgraph".to_owned(),
            }
        }
        CatalogGraphAction::SetSubgraphSearchAliases => {
            let (converted, _, definition) = converted_fixture(model)?;
            model = converted;
            GraphActionInput::SubgraphSearchAliases {
                definition,
                aliases: vec!["native".to_owned(), "catalog".to_owned()],
            }
        }
        CatalogGraphAction::FitView => GraphActionInput::FitAvailable(GraphSize {
            width: 1_000.0,
            height: 700.0,
        }),
        CatalogGraphAction::Unlock => {
            model.apply(GraphCommand::SetViewport {
                viewport: GraphViewport {
                    locked: true,
                    ..GraphViewport::default()
                },
            })?;
            GraphActionInput::None
        }
        CatalogGraphAction::ResetView => {
            model.apply(GraphCommand::PanViewport {
                delta: GraphPoint { x: 80.0, y: 60.0 },
            })?;
            GraphActionInput::None
        }
        CatalogGraphAction::SelectAll => {
            model.apply(GraphCommand::ClearSelection)?;
            GraphActionInput::None
        }
        CatalogGraphAction::ToggleSelectedItemsPin => {
            let (grouped, identifier) = grouped_fixture(model)?;
            model = grouped;
            model.apply(GraphCommand::SetSelection {
                selection: GraphSelection {
                    nodes: BTreeSet::from([
                        GraphIdentifier::from("source"),
                        GraphIdentifier::from("target"),
                    ]),
                    groups: BTreeSet::from([identifier]),
                    ..GraphSelection::default()
                },
                mode: SelectionMode::Replace,
            })?;
            GraphActionInput::None
        }
        _ => GraphActionInput::None,
    };
    let before = model.encode()?;
    let input_evidence = format!(
        "{action:?}\n{input:?}\npre_state_digest={:x}",
        Sha256::digest(&before)
    );
    let effect = match GraphCommandModel::execute(&mut model, action, input) {
        Err(GraphActionError::RequiresAssetService(CatalogGraphAction::PublishSubgraph))
            if action == CatalogGraphAction::PublishSubgraph =>
        {
            let after = model.encode()?;
            if before != after {
                return Err("asset-owned publication mutated the graph command model".into());
            }
            return Ok(json!({
                "name": action.command_id(),
                "passed": true,
                "input_digest": format!("{:x}", Sha256::digest(input_evidence.as_bytes())),
                "pre_state_digest": format!("{:x}", Sha256::digest(&before)),
                "post_state_digest": format!("{:x}", Sha256::digest(&after)),
                "effect_digest": format!("{:x}", Sha256::digest(b"canonical-asset-service")),
                "digest": format!("{:x}", Sha256::digest(&after)),
            }));
        }
        Ok(effect) => effect,
        Err(error) => return Err(error.into()),
    };
    let effect_bytes = match (action, &effect) {
        (CatalogGraphAction::CopySelected, GraphActionEffect::ClipboardText(text)) => {
            decode_clipboard(text.as_bytes())?;
            text.as_bytes().to_vec()
        }
        (CatalogGraphAction::CopySelected, _) => {
            return Err("copy returned no clipboard text".into());
        }
        (_, GraphActionEffect::None) => Vec::new(),
        (_, GraphActionEffect::ClipboardText(_)) => {
            return Err("non-copy action returned clipboard text".into());
        }
    };
    let after = model.encode()?;
    if action != CatalogGraphAction::CopySelected {
        if before == after {
            return Err(format!(
                "{} produced no observable state change",
                action.command_id()
            )
            .into());
        }
    } else if before != after {
        return Err("copy mutated the persisted workspace state".into());
    }
    if let Some(document) = model.document() {
        document.validate()?;
    }
    Ok(json!({
        "name": action.command_id(),
        "passed": true,
        "input_digest": format!("{:x}", Sha256::digest(input_evidence.as_bytes())),
        "pre_state_digest": format!("{:x}", Sha256::digest(&before)),
        "post_state_digest": format!("{:x}", Sha256::digest(&after)),
        "effect_digest": format!("{:x}", Sha256::digest(&effect_bytes)),
        "digest": format!("{:x}", Sha256::digest(&after)),
    }))
}

fn settings_owned_catalog_case(cx: &mut TestAppContext) -> Result<Value, Box<dyn Error>> {
    let action = CatalogGraphAction::ToggleVueNodes;
    let (item, cx) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            fixture_model().expect("create graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    let (workflow_before, can_undo_before, enabled_before) = item.read_with(cx, |item, cx| {
        (
            item.model
                .document()
                .expect("editable catalog graph")
                .to_workflow_bytes()
                .expect("encode catalog workflow"),
            item.model
                .engine()
                .is_some_and(GraphCommandEngine::can_undo),
            item.native_node_renderer_enabled(cx),
        )
    });
    if !enabled_before {
        return Err("catalog settings fixture must start with the native renderer enabled".into());
    }

    let accepted = item.update(cx, |item, cx| {
        item.execute_catalog_action(action, GraphActionInput::None, cx)
    });
    cx.run_until_parked();
    if !accepted {
        let last_error = item.read_with(cx, |item, _| item.model.last_error.clone());
        return Err(format!("settings-owned catalog action was rejected: {last_error:?}").into());
    }

    let (workflow_after, can_undo_after, enabled_after, announcement) =
        item.read_with(cx, |item, cx| {
            (
                item.model
                    .document()
                    .expect("editable catalog graph")
                    .to_workflow_bytes()
                    .expect("encode catalog workflow"),
                item.model
                    .engine()
                    .is_some_and(GraphCommandEngine::can_undo),
                item.native_node_renderer_enabled(cx),
                item.model.announcement.clone(),
            )
        });
    if workflow_after != workflow_before || can_undo_after != can_undo_before {
        return Err("settings-owned catalog action mutated workflow state or history".into());
    }
    if enabled_after {
        return Err("SettingsStore did not commit the native renderer toggle".into());
    }
    if announcement.as_deref() != Some("Compact native node renderer enabled") {
        return Err(
            format!("unexpected settings-owned action announcement: {announcement:?}").into(),
        );
    }

    let input_evidence = format!("{action:?}\n{:?}", GraphActionInput::None);
    let effect_evidence = format!("enabled={enabled_after}\nannouncement={announcement:?}");
    Ok(json!({
        "name": action.command_id(),
        "passed": true,
        "input_digest": format!("{:x}", Sha256::digest(input_evidence.as_bytes())),
        "pre_state_digest": format!("{:x}", Sha256::digest(&workflow_before)),
        "post_state_digest": format!("{:x}", Sha256::digest(&workflow_after)),
        "effect_digest": format!("{:x}", Sha256::digest(effect_evidence.as_bytes())),
        "digest": format!("{:x}", Sha256::digest(effect_evidence.as_bytes())),
    }))
}

fn catalog_cases(cx: &mut TestAppContext) -> Result<Vec<Value>, Box<dyn Error>> {
    CatalogGraphAction::ALL
        .iter()
        .copied()
        .map(|action| {
            if action == CatalogGraphAction::ToggleVueNodes {
                settings_owned_catalog_case(cx)
            } else {
                catalog_case(action)
            }
        })
        .collect()
}

pub(crate) fn state_case(name: &str, state: &[u8]) -> Value {
    let digest = format!("{:x}", Sha256::digest(state));
    json!({
        "name": name,
        "passed": true,
        "post_state_digest": digest,
        "digest": digest,
    })
}

pub(crate) fn write_artifact(
    file_name: &str,
    validation_id: &str,
    fixture_digests: Value,
    cases: Vec<Value>,
) -> Result<(), Box<dyn Error>> {
    if cases.iter().any(|case| case["passed"] != true) {
        return Err(format!("{validation_id} contains a failing case").into());
    }
    let scheduler_seed = match validation_id {
        "VAL-GPUI-001" => 16001,
        "VAL-GPUI-002" => 16002,
        "VAL-GPUI-003" => 16003,
        "VAL-GPUI-004" => 16004,
        "VAL-GPUI-014" => 16014,
        _ => 0,
    };
    let artifact = json!({
        "validation_id": validation_id,
        "environment": {
            "backend": "gpui-test",
            "platform": "mock-window",
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "feature": "test-support",
            "scheduler_seed": scheduler_seed,
            "iterations": std::env::var("ITERATIONS").unwrap_or_else(|_| "1".to_owned()),
        },
        "fixture_digests": fixture_digests,
        "cases": cases,
        "skipped": [],
    });
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"))
        .join("comfy-parity");
    fs::create_dir_all(&target)?;
    fs::write(
        target.join(file_name),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}

#[gpui::test(seed = 16001)]
fn val_gpui_001(cx: &mut TestAppContext) {
    let file_system = FakeFs::new(cx.executor());
    cx.update(|cx| {
        let settings_store = settings::SettingsStore::test(cx);
        cx.set_global(settings_store);
        <dyn project::Fs>::set_global(file_system, cx);
    });
    let catalog_fixture = fixture_model()
        .and_then(|model| model.encode().map_err(Into::into))
        .expect("encode catalog graph fixture");
    let mut cases = catalog_cases(cx).expect("all 37 catalog graph actions must execute");
    let (item, cx) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            fixture_model().expect("create graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    item.update_in(cx, |item, window, cx| {
        item.apply_graph_command(GraphCommand::ClearSelection, cx);
        item.focus_graph(window, cx);
        assert!(item.focus_handle.is_focused(window));
    });
    cx.dispatch_action(GraphSelectAll);
    assert_eq!(
        item.read_with(cx, |item, _| item
            .model
            .selection()
            .map(|selection| selection.nodes.len())),
        Some(2)
    );
    cx.dispatch_action(GraphCopy);
    let clipboard = cx
        .read_from_clipboard()
        .and_then(|item| item.text())
        .expect("graph copy clipboard text");
    assert!(clipboard.starts_with(GRAPH_CLIPBOARD_MEDIA_TYPE));
    let scale_before = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.scale)
    });
    cx.dispatch_action(GraphZoomIn);
    let scale_after = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.scale)
    });
    assert!(scale_after > scale_before);
    cx.dispatch_action(GraphUndo);
    cx.dispatch_action(GraphRedo);
    item.update_in(cx, |item, window, cx| {
        let output_focus = item.control_focus_handle("output:source:0", cx);
        window.focus(&output_focus, cx);
    });
    cx.simulate_keystrokes("enter");
    item.update_in(cx, |item, window, cx| {
        let input_focus = item.control_focus_handle("input:target:0", cx);
        window.focus(&input_focus, cx);
    });
    cx.simulate_keystrokes("enter");
    assert!(item.read_with(cx, |item, _| {
        item.pending_link.is_none()
            && item
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .is_some_and(|graph| graph.links.len() == 1)
    }));
    assert!(cx.debug_bounds("SIM-GRAPH-ANNOUNCEMENT").is_some());
    let keyboard_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode keyboard-authored graph");
    cases.push(json!({
        "name": "focused-action-dispatch-clipboard-and-keyboard-link",
        "passed": true,
        "post_state_digest": format!("{:x}", Sha256::digest(&keyboard_state)),
        "digest": format!("{:x}", Sha256::digest(clipboard.as_bytes())),
    }));
    for (focus_identifier, keystroke) in [
        ("widget:source:enabled", "enter"),
        ("widget:source:steps", "up"),
        ("widget:source:strength", "up"),
        ("widget:source:mode", "right"),
        ("widget:source:label", "x"),
    ] {
        item.update_in(cx, |item, window, cx| {
            let focus = item.control_focus_handle(focus_identifier, cx);
            window.focus(&focus, cx);
        });
        cx.simulate_keystrokes(keystroke);
    }
    item.read_with(cx, |item, _| {
        let widgets = &item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph")
            .nodes[&GraphIdentifier::from("source")]
            .widgets;
        assert_eq!(widgets[0].value, Value::from(21));
        assert_eq!(widgets[1].value, Value::Bool(true));
        assert_eq!(widgets[2].value, Value::from(0.75));
        assert_eq!(widgets[3].value, Value::String("quality".to_owned()));
        assert_eq!(widgets[4].value, Value::String("nativex".to_owned()));
        assert!(
            widgets[..5]
                .iter()
                .all(|widget| widget.value == widget.prompt_value)
        );
        assert!(matches!(widgets[5].kind, GraphWidgetKind::Preserved { .. }));
    });
    let widget_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode keyboard-operated native widget state");
    cases.push(state_case(
        "keyboard-native-widget-operability-and-prompt-values",
        &widget_state,
    ));
    write_artifact(
        "val-gpui-001.json",
        "VAL-GPUI-001",
        json!({
            "workflow": format!("{:x}", Sha256::digest(&catalog_fixture)),
            "catalog_action_count": CatalogGraphAction::ALL.len(),
        }),
        cases,
    )
    .expect("write VAL-GPUI-001 artifact");
}

#[gpui::test(seed = 16002)]
fn val_gpui_002(cx: &mut TestAppContext) {
    let (item, cx) = cx.add_window_view(|_, cx| {
        let mut model = fixture_model().expect("create graph fixture");
        model
            .apply(GraphCommand::ClearSelection)
            .expect("clear selection");
        GraphWorkspaceItem::new(model, WeakEntity::new_invalid(), cx)
    });
    item.update_in(cx, |item, window, cx| item.focus_graph(window, cx));
    let target_bounds = cx
        .debug_bounds("SIM-NODE-target")
        .expect("target node debug bounds");
    let target_click = point(
        target_bounds.origin.x + px(20.0),
        target_bounds.origin.y + px(14.0),
    );
    cx.simulate_click(target_click, Modifiers::default());
    assert_eq!(
        item.read_with(cx, |item, _| item.model.selected_node_identifiers()),
        vec![GraphIdentifier::from("target")]
    );
    let source_bounds = cx
        .debug_bounds("SIM-NODE-source")
        .expect("source node debug bounds");
    let source_click = point(
        source_bounds.origin.x + px(20.0),
        source_bounds.origin.y + px(14.0),
    );
    cx.simulate_click(source_click, Modifiers::shift());
    assert_eq!(
        item.read_with(cx, |item, _| item.model.selected_node_identifiers().len()),
        2
    );
    let selection_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode pointer selection state");
    let drag_end = point(source_click.x + px(53.0), source_click.y + px(27.0));
    cx.simulate_mouse_down(source_click, MouseButton::Left, Modifiers::default());
    cx.simulate_mouse_move(drag_end, MouseButton::Left, Modifiers::default());
    assert!(cx.debug_bounds("SIM-SELECTION-BOX").is_none());
    cx.simulate_mouse_up(drag_end, MouseButton::Left, Modifiers::default());
    let positions = item.read_with(cx, |item, _| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph");
        (
            graph.nodes[&GraphIdentifier::from("source")].position,
            graph.nodes[&GraphIdentifier::from("target")].position,
        )
    });
    assert_eq!(positions.0, GraphPoint { x: 150.0, y: 130.0 });
    assert_eq!(positions.1, GraphPoint { x: 500.0, y: 130.0 });
    let drag_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode drag state");

    let output_bounds = cx
        .debug_bounds("SIM-OUTPUT-source-0")
        .expect("source output debug bounds");
    let input_bounds = cx
        .debug_bounds("SIM-INPUT-target-0")
        .expect("target input debug bounds");
    cx.simulate_mouse_down(
        output_bounds.center(),
        MouseButton::Left,
        Modifiers::default(),
    );
    cx.simulate_mouse_up(
        input_bounds.center(),
        MouseButton::Left,
        Modifiers::default(),
    );
    assert_eq!(
        item.read_with(cx, |item, _| item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.links.len())),
        Some(1)
    );
    let dynamic_input_bounds = cx
        .debug_bounds("SIM-INPUT-target-1")
        .expect("expanded dynamic target input bounds");
    let output_bounds = cx
        .debug_bounds("SIM-OUTPUT-source-0")
        .expect("source output bounds after dynamic expansion");
    cx.simulate_mouse_down(
        output_bounds.center(),
        MouseButton::Left,
        Modifiers::default(),
    );
    cx.simulate_mouse_up(
        dynamic_input_bounds.center(),
        MouseButton::Left,
        Modifiers::default(),
    );
    item.read_with(cx, |item, _| {
        let target = &item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph after dynamic port links")
            .nodes[&GraphIdentifier::from("target")];
        assert_eq!(target.inputs.len(), 3);
        assert!(!target.inputs[0].dynamic);
        assert!(!target.inputs[1].dynamic);
        assert!(target.inputs[2].dynamic);
        assert_eq!(
            item.model
                .document()
                .and_then(|document| document.active_graph().ok())
                .map(|graph| graph.links.len()),
            Some(2)
        );
    });
    cx.simulate_mouse_down(
        output_bounds.center(),
        MouseButton::Left,
        Modifiers::default(),
    );
    item.update_in(cx, |item, window, cx| item.focus_graph(window, cx));
    cx.simulate_keystrokes("escape");
    assert!(item.read_with(cx, |item, _| item.pending_link.is_none()));
    let link_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode link state");

    let viewport_before = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.offset)
    });
    cx.simulate_event(ScrollWheelEvent {
        position: point(px(20.0), px(20.0)),
        delta: ScrollDelta::Pixels(point(px(12.0), px(18.0))),
        modifiers: Modifiers::default(),
        touch_phase: TouchPhase::Moved,
    });
    let viewport_after = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.offset)
    });
    assert_ne!(viewport_before, viewport_after);
    let scale_before = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.scale)
    });
    cx.simulate_event(ScrollWheelEvent {
        position: point(px(20.0), px(20.0)),
        delta: ScrollDelta::Pixels(point(px(0.0), px(-18.0))),
        modifiers: Modifiers {
            control: true,
            ..Modifiers::default()
        },
        touch_phase: TouchPhase::Moved,
    });
    let scale_after = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.scale)
    });
    assert_ne!(scale_before, scale_after);
    let workflow = item
        .read_with(cx, |item, _| {
            item.model.document().expect("document").to_workflow_bytes()
        })
        .expect("workflow bytes");
    let viewport_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode viewport state");
    let mut large_document = GraphDocument::default();
    large_document.document_identity = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_2002);
    for index in 0..1_000usize {
        let identifier = format!("large-{index:04}");
        let column = index % 40;
        let row = index / 40;
        let node = fixture_node(
            &identifier,
            GraphPoint {
                x: column as f32 * 220.0,
                y: row as f32 * 130.0,
            },
            None,
            None,
        );
        large_document
            .root
            .nodes
            .insert(node.identifier.clone(), node);
    }
    let large_engine = GraphCommandEngine::new(large_document).expect("create large graph engine");
    let large_bytes = large_engine
        .document
        .to_workflow_bytes()
        .expect("serialize large graph");
    let large_model = GraphWorkspaceModel::open(
        "large-workflow.json",
        "large-workflow",
        WorkflowStorageProvider::Draft,
        large_bytes,
    )
    .expect("open large graph model");
    let (large_item, large_context) = cx.cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(large_model, WeakEntity::new_invalid(), cx)
    });
    assert!(large_context.debug_bounds("SIM-NODE-large-0000").is_some());
    assert!(large_context.debug_bounds("SIM-NODE-large-0999").is_some());
    large_item.update_in(large_context, |item, window, cx| {
        item.focus_graph(window, cx);
    });
    large_context.dispatch_action(GraphSelectAll);
    large_context.dispatch_action(GraphFitView);
    assert_eq!(
        large_item.read_with(large_context, |item, _| item
            .model
            .selection()
            .map(|selection| selection.nodes.len())),
        Some(1_000)
    );
    let large_state = large_item
        .read_with(large_context, |item, _| item.model.encode())
        .expect("encode large GPUI graph");
    assert_eq!(
        GraphWorkspaceModel::decode(&large_state)
            .expect("restore large GPUI graph")
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len()),
        Some(1_000)
    );
    let renderer = exercise_native_renderer_routes_groups_breadcrumbs_and_minimap(&mut cx.cx);
    write_artifact(
        "val-gpui-002.json",
        "VAL-GPUI-002",
        json!({"workflow": format!("{:x}", Sha256::digest(&workflow))}),
        vec![
            state_case("pointer-selection-and-multiselect", &selection_state),
            state_case("single-transaction-drag-with-preview", &drag_state),
            state_case("typed-port-link-and-cancel", &link_state),
            state_case("wheel-pan-and-zoom-state-change", &viewport_state),
            state_case(
                "thousand-node-gpui-render-selection-fit-and-restore",
                &large_state,
            ),
            state_case("collapsed-group-hide-and-expand", &renderer.group_state),
            state_case(
                "sampled-link-boundary-hit-away-from-node-hitboxes",
                &renderer.boundary_hit_state,
            ),
            state_case(
                "pointer-reroute-insertion-and-keyboard-link-reconnect",
                &renderer.reroute_reconnect_state,
            ),
            state_case(
                "scaled-node-port-and-link-endpoint-alignment",
                &renderer.scaled_geometry_state,
            ),
            state_case(
                "production-node-group-layout-and-reroute-model-actions",
                &renderer.model_action_state,
            ),
            state_case(
                "actual-window-minimap-projection-and-navigation",
                &renderer.minimap_state,
            ),
            state_case(
                "subgraph-breadcrumb-pointer-exit",
                &renderer.breadcrumb_state,
            ),
        ],
    )
    .expect("write VAL-GPUI-002 artifact");
}

#[gpui::test(seed = 16003)]
fn val_gpui_003(cx: &mut TestAppContext) {
    let (item, cx) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            fixture_model().expect("create graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    item.update_in(cx, |item, window, cx| item.focus_graph(window, cx));
    cx.dispatch_action(GraphCopy);
    let copied = cx
        .read_from_clipboard()
        .and_then(|item| item.text())
        .expect("copied graph text");
    let target_window = cx.add_window(|_, cx| {
        GraphWorkspaceItem::new(
            fixture_model().expect("create target-window graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    let target_window_handle = *target_window;
    cx.run_until_parked();
    target_window
        .update(cx, |item, window, cx| {
            let field_focus = item.control_focus_handle("widget:source:label", cx);
            window.focus(&field_focus, cx);
        })
        .expect("focus target-window widget field");
    cx.cx.dispatch_action(target_window_handle, GraphPaste);
    assert_eq!(
        target_window
            .read_with(cx, |item, _| item
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .map(|graph| graph.nodes.len()))
            .expect("read target-window graph"),
        Some(4)
    );
    assert_eq!(
        item.read_with(cx, |item, _| item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len())),
        Some(2),
        "cross-window paste must not mutate the source graph"
    );
    target_window
        .update(cx, |item, window, cx| {
            let field_focus = item.control_focus_handle("widget:source:label", cx);
            assert!(field_focus.is_focused(window));
            let preserved_widget_count = item
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .map(|graph| {
                    graph
                        .nodes
                        .values()
                        .flat_map(|node| &node.widgets)
                        .filter(|widget| matches!(widget.kind, GraphWidgetKind::Preserved { .. }))
                        .count()
                });
            assert_eq!(preserved_widget_count, Some(2));
            assert!(item.focus_handle.contains_focused(window, cx));
        })
        .expect("verify target-window focus and unknown widget preservation");
    item.update_in(cx, |item, window, _| {
        assert!(item.focus_handle.is_focused(window));
    });
    let source_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode source-window graph state");
    let target_state = target_window
        .read_with(cx, |item, _| item.model.encode())
        .expect("read target-window graph")
        .expect("encode target-window pasted graph state");
    let paste_state = [source_state, target_state].concat();

    cx.dispatch_action(GraphCut);
    assert_eq!(
        item.read_with(cx, |item, _| item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len())),
        Some(0)
    );
    let cut_payload = cx
        .read_from_clipboard()
        .and_then(|item| item.text())
        .expect("cut graph clipboard text");
    assert!(cut_payload.starts_with(GRAPH_CLIPBOARD_MEDIA_TYPE));
    let cut_removed_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode cut graph state");
    cx.dispatch_action(GraphUndo);
    let cut_restored_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode cut undo state");
    let cut_state = [cut_removed_state, cut_restored_state].concat();

    let workflow_json = br#"{"version":0.4,"last_node_id":1,"last_link_id":0,"nodes":[{"id":"workflow-source","type":"Source","pos":[20,30],"size":[140,80],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"image","type":"IMAGE","links":[]}],"properties":{},"widgets_values":[]}],"links":[],"groups":[],"config":{},"extra":{}}"#;
    cx.write_to_clipboard(ClipboardItem::new_string(
        String::from_utf8(workflow_json.to_vec()).expect("workflow JSON is UTF-8"),
    ));
    assert!(item.update(cx, |item, cx| item.paste_from_clipboard(false, cx)));
    let nodes_after_workflow = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len())
    });
    assert_eq!(nodes_after_workflow, Some(3));
    let api_prompt_json = br#"{
        "1":{"class_type":"LoadImage","inputs":{"image":"fixture.png"}},
        "2":{"class_type":"PreviewImage","inputs":{"images":["1",0]}}
    }"#;
    cx.write_to_clipboard(ClipboardItem::new_string(
        String::from_utf8(api_prompt_json.to_vec()).expect("API prompt JSON is UTF-8"),
    ));
    assert!(item.update(cx, |item, cx| item.paste_from_clipboard(false, cx)));
    let nodes_after_api_prompt = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len())
    });
    assert_eq!(nodes_after_api_prompt, Some(5));
    let json_import_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode workflow and API-prompt paste state");

    let nodes_before_rejection = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len())
    });
    let mut external_paths = ExternalPaths::default();
    external_paths
        .0
        .push(PathBuf::from("/fixtures/non-empty-workflow.json"));
    external_paths
        .0
        .push(PathBuf::from("/fixtures/non-empty-image.png"));
    let file_payload_evidence = format!("{external_paths:?}").into_bytes();
    cx.write_to_clipboard(ClipboardItem {
        entries: vec![ClipboardEntry::ExternalPaths(external_paths)],
    });
    assert!(!item.update(cx, |item, cx| item.paste_from_clipboard(false, cx)));
    assert_eq!(
        nodes_before_rejection,
        item.read_with(cx, |item, _| item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len()))
    );
    let file_rejection_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode file rejection state");
    let png_bytes = vec![
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
        0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8,
        0xcf, 0xc0, 0x00, 0x00, 0x04, 0x01, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00,
        0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];
    let image = Image::from_bytes(ImageFormat::Png, png_bytes.clone());
    cx.write_to_clipboard(ClipboardItem::new_image(&image));
    assert!(!item.update(cx, |item, cx| item.paste_from_clipboard(false, cx)));
    assert_eq!(
        nodes_before_rejection,
        item.read_with(cx, |item, _| item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len()))
    );
    let media_rejection_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode media rejection state");
    let hostile = format!("{}0{}", "[".repeat(140), "]".repeat(140));
    cx.write_to_clipboard(ClipboardItem::new_string(hostile));
    assert!(!item.update(cx, |item, cx| item.paste_from_clipboard(false, cx)));
    assert_eq!(
        nodes_before_rejection,
        item.read_with(cx, |item, _| item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len()))
    );
    assert!(item.read_with(cx, |item, _| item.model.last_error.is_some()));
    let hostile_rejection_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode hostile rejection state");

    let snapshot = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("workspace snapshot");
    let restored = GraphWorkspaceModel::decode(&snapshot).expect("restore workspace snapshot");
    assert_eq!(
        restored.document().map(|document| document.root.clone()),
        item.read_with(cx, |item, _| item
            .model
            .document()
            .map(|document| document.root.clone()))
    );
    let invalid_bytes = br#"{"version":1,"nodes":[{"id":"broken","pos":[null,0]}]}"#.to_vec();
    let read_only = GraphWorkspaceModel::open(
        "Invalid workflow",
        "invalid-fixture",
        WorkflowStorageProvider::Draft,
        invalid_bytes.clone(),
    )
    .expect("open invalid workflow read-only");
    assert!(read_only.is_read_only());
    assert_eq!(read_only.original_bytes(), invalid_bytes);
    write_artifact(
        "val-gpui-003.json",
        "VAL-GPUI-003",
        json!({
            "clipboard": format!("{:x}", Sha256::digest(copied.as_bytes())),
            "cut_clipboard": format!("{:x}", Sha256::digest(cut_payload.as_bytes())),
            "workflow_json": format!("{:x}", Sha256::digest(workflow_json)),
            "api_prompt_json": format!("{:x}", Sha256::digest(api_prompt_json)),
            "file_payload": format!("{:x}", Sha256::digest(&file_payload_evidence)),
            "media_payload": format!("{:x}", Sha256::digest(&png_bytes)),
            "snapshot": format!("{:x}", Sha256::digest(&snapshot)),
        }),
        vec![
            state_case(
                "copy-paste-across-two-windows-and-widget-focus-restoration",
                &paste_state,
            ),
            state_case("cut-selection-and-undo-restoration", &cut_state),
            state_case("paste-workflow-and-api-prompt-json", &json_import_state),
            state_case("non-empty-file-payload-rejection", &file_rejection_state),
            state_case("non-empty-media-payload-rejection", &media_rejection_state),
            state_case(
                "hostile-payload-rejection-preserves-graph",
                &hostile_rejection_state,
            ),
            state_case("snapshot-and-read-only-original-data-recovery", &snapshot),
        ],
    )
    .expect("write VAL-GPUI-003 artifact");
}

#[gpui::test(seed = 16004)]
async fn val_gpui_004(cx: &mut TestAppContext) {
    let active_profile_id = sim_runtime::ProfileId(Uuid::from_u128(16_004));
    cx.update(|cx| {
        workspace::AppState::test(cx);
        init_execution_ui_model_for_profile(active_profile_id, cx)
            .expect("initialize active profile for graph workspace binding");
    });
    let (item, cx) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            fixture_model().expect("create graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    for selector in [
        "SIM-GRAPH",
        "SIM-NODE-source",
        "SIM-NODE-target",
        "SIM-OUTPUT-source-0",
        "SIM-INPUT-target-0",
    ] {
        assert!(cx.debug_bounds(selector).is_some(), "missing {selector}");
    }
    item.update_in(cx, |item, window, cx| {
        item.focus_graph(window, cx);
        assert!(item.focus_handle.is_focused(window));
    });
    cx.simulate_keystrokes("tab");
    item.update_in(cx, |item, window, cx| {
        assert!(item.focus_handle.contains_focused(window, cx));
    });
    cx.dispatch_action(GraphSelectAll);
    cx.dispatch_action(GraphZoomIn);
    cx.dispatch_action(GraphZoomOut);
    item.update_in(cx, |item, window, cx| {
        assert!(item.focus_handle.contains_focused(window, cx));
    });
    let graph = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .cloned()
            .expect("active graph")
    });
    let labels = graph
        .nodes
        .values()
        .map(|node| crate::graph_render::node_accessibility_label(node, &graph))
        .collect::<Vec<_>>();
    assert!(
        labels
            .iter()
            .all(|label| label.contains("type NativeFixture"))
    );
    assert!(labels.iter().any(|label| label.contains("1 outputs")));
    assert!(labels.iter().any(|label| label.contains("1 inputs")));
    let final_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode workflow tab state");
    let compact_snapshot: Value =
        serde_json::from_slice(&final_state).expect("parse compact workspace snapshot");
    assert_eq!(
        compact_snapshot["schema_version"],
        Value::from(GRAPH_WORKSPACE_SCHEMA_VERSION)
    );
    assert!(compact_snapshot["engine"].is_string());
    assert!(compact_snapshot["save_journal"].is_string());
    let semantics = labels.join("\n");
    let reopened = GraphWorkspaceModel::decode(&final_state).expect("reopen serialized tab state");
    assert_eq!(
        reopened.document().map(|document| document.root.clone()),
        item.read_with(cx, |item, _| item
            .model
            .document()
            .map(|document| document.root.clone()))
    );
    let mut engine_tampered: Value =
        serde_json::from_slice(&final_state).expect("parse engine-tamper snapshot");
    let engine_bytes = BASE64
        .decode(
            engine_tampered["engine"]
                .as_str()
                .expect("persisted graph engine is compact base64"),
        )
        .expect("decode persisted graph engine bytes");
    let mut engine = GraphCommandEngine::decode(&engine_bytes).expect("decode graph engine");
    engine
        .apply(GraphCommand::RenameNode {
            identifier: GraphIdentifier::from("source"),
            title: "Split engine".to_owned(),
        })
        .expect("mutate only the persisted engine");
    engine_tampered["engine"] =
        Value::String(BASE64.encode(engine.encode().expect("encode tampered graph engine")));
    assert_eq!(
        GraphWorkspaceModel::decode(
            &serde_json::to_vec(&engine_tampered).expect("encode engine-tamper snapshot")
        ),
        Err(GraphWorkspaceError::InvalidSnapshotState)
    );

    let mut journal_tampered: Value =
        serde_json::from_slice(&final_state).expect("parse journal-tamper snapshot");
    let journal_bytes = BASE64
        .decode(
            journal_tampered["save_journal"]
                .as_str()
                .expect("persisted save journal is compact base64"),
        )
        .expect("decode save journal bytes");
    let mut coordinator =
        WorkflowSaveCoordinator::decode(&journal_bytes).expect("decode save coordinator");
    let mut journal_document =
        GraphDocument::from_workflow_bytes(coordinator.local_bytes()).expect("parse journal graph");
    journal_document
        .root
        .nodes
        .get_mut(&GraphIdentifier::from("source"))
        .expect("journal source node")
        .title = "Split journal".to_owned();
    coordinator
        .edit(
            journal_document
                .to_workflow_bytes()
                .expect("serialize journal-only mutation"),
        )
        .expect("mutate only the save journal");
    journal_tampered["save_journal"] =
        Value::String(BASE64.encode(coordinator.encode().expect("encode tampered save journal")));
    assert_eq!(
        GraphWorkspaceModel::decode(
            &serde_json::to_vec(&journal_tampered).expect("encode journal-tamper snapshot")
        ),
        Err(GraphWorkspaceError::InvalidSnapshotState)
    );

    let mut ephemeral = GraphWorkspaceModel::decode(&final_state).expect("decode ephemeral model");
    let mut ephemeral_viewport = ephemeral
        .document()
        .and_then(|document| document.active_graph().ok())
        .expect("ephemeral active graph")
        .viewport
        .clone();
    ephemeral_viewport.offset = GraphPoint { x: 91.0, y: 47.0 };
    ephemeral
        .replace_ephemeral_graph_state(GraphSelection::default(), ephemeral_viewport.clone())
        .expect("replace engine-owned workspace projection");
    let ephemeral_snapshot = ephemeral.encode().expect("encode ephemeral snapshot");
    let ephemeral =
        GraphWorkspaceModel::decode(&ephemeral_snapshot).expect("decode coherent ephemeral state");
    let ephemeral_graph = ephemeral
        .document()
        .and_then(|document| document.active_graph().ok())
        .expect("restored ephemeral active graph");
    assert_eq!(ephemeral_graph.selection, GraphSelection::default());
    assert_eq!(ephemeral_graph.viewport, ephemeral_viewport);
    let snapshot_owner_state = [
        serde_json::to_vec(&engine_tampered).expect("snapshot engine evidence"),
        serde_json::to_vec(&journal_tampered).expect("snapshot journal evidence"),
        ephemeral_snapshot,
    ]
    .concat();
    let legacy_model = GraphWorkspaceModel::decode(&final_state).expect("decode legacy source");
    let legacy_snapshot = serde_json::to_vec(&json!({
        "schema_version": 1,
        "title": legacy_model.title.clone(),
        "engine": legacy_model.engine().map(GraphCommandEngine::encode).transpose().expect("encode legacy engine"),
        "read_only_bytes": if legacy_model.is_read_only() { Some(legacy_model.original_bytes().to_vec()) } else { None },
        "read_only_diagnostic": legacy_model.read_only_diagnostic(),
        "save_journal": legacy_model.save_coordinator.encode().expect("encode legacy save journal"),
        "execution_association": legacy_model.execution_association.clone(),
        "canvas_info_visible": legacy_model.canvas_info_visible,
        "native_node_renderer": false,
    }))
    .expect("encode v1 snapshot");
    let migrated_snapshot =
        GraphWorkspaceModel::decode(&legacy_snapshot).expect("migrate v1 workspace snapshot");
    assert_eq!(
        migrated_snapshot.schema_version,
        GRAPH_WORKSPACE_SCHEMA_VERSION
    );
    let migrated_snapshot = migrated_snapshot
        .encode()
        .expect("encode migrated snapshot");
    let migrated_snapshot_json: serde_json::Value =
        serde_json::from_slice(&migrated_snapshot).expect("parse migrated workspace snapshot");
    assert!(
        migrated_snapshot_json.get("native_node_renderer").is_none(),
        "legacy settings-owned renderer state must not be re-emitted by workspace persistence"
    );

    let schema_zero_four = GraphWorkspaceModel::open(
        "schema-0.4.json",
        "schema-0.4",
        WorkflowStorageProvider::Draft,
        br#"{"version":0.4,"last_node_id":0,"last_link_id":0,"nodes":[],"links":[],"groups":[],"config":{},"extra":{}}"#.to_vec(),
    )
    .expect("open schema 0.4 workflow");
    assert!(!schema_zero_four.is_read_only());
    let api_prompt = GraphWorkspaceModel::open(
        "api-prompt.json",
        "api-prompt",
        WorkflowStorageProvider::Draft,
        br#"{"1":{"class_type":"NativeFixture","inputs":{"steps":20}}}"#.to_vec(),
    )
    .expect("open API prompt");
    assert!(!api_prompt.is_read_only());
    let malformed_bytes = br#"{"version":1,"nodes":"invalid","links":[]}"#.to_vec();
    let malformed = GraphWorkspaceModel::open(
        "malformed.json",
        "malformed",
        WorkflowStorageProvider::Draft,
        malformed_bytes.clone(),
    )
    .expect("preserve malformed workflow read-only");
    assert!(malformed.is_read_only());
    assert_eq!(malformed.original_bytes(), malformed_bytes);
    let import_state = [
        migrated_snapshot,
        schema_zero_four.encode().expect("encode schema 0.4"),
        api_prompt.encode().expect("encode API prompt"),
        malformed.encode().expect("encode malformed state"),
    ]
    .concat();

    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(Path::new("/project"), json!({})).await;
    let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
    let save_as_path = project
        .read_with(cx, |project, cx| {
            project.find_project_path("project/native-workflow.json", cx)
        })
        .expect("resolve save-as path");
    let save_as = item.update_in(cx, |item, window, cx| {
        item.save_as(project.clone(), save_as_path, window, cx)
    });
    save_as.await.expect("save workflow as native local file");
    let local_path = Path::new("/project/native-workflow.json");
    let disk_bytes = fs
        .load_bytes(local_path)
        .await
        .expect("read native local workflow");
    assert!(!disk_bytes.is_empty());
    cx.run_until_parked();
    fs.remove_file(local_path, RemoveOptions::default())
        .await
        .expect("delete workflow outside Zed");
    cx.background_executor
        .timer(Duration::from_millis(250))
        .await;
    assert_eq!(
        item.read_with(cx, |item, _| item.model.save_coordinator.authority()),
        WorkflowAuthority::ExternalMissing
    );
    let missing_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode externally deleted workflow state");
    item.update(cx, |item, cx| {
        assert!(item.keep_local_version(cx));
        assert!(item.model.save_coordinator.external_missing());
        assert!(item.model.save_coordinator.missing_recreation_approved());
    });
    let recreate = item.update_in(cx, |item, window, cx| {
        item.save(Default::default(), project.clone(), window, cx)
    });
    recreate
        .await
        .expect("recreate externally deleted workflow after Keep Local");
    assert_eq!(
        fs.load_bytes(local_path)
            .await
            .expect("read recreated workflow"),
        disk_bytes
    );
    assert_eq!(
        item.read_with(cx, |item, _| item.model.save_coordinator.authority()),
        WorkflowAuthority::InSync
    );
    let recreated_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode recreated workflow state");
    let external_deletion_state = [missing_state, recreated_state].concat();
    item.update(cx, |item, cx| {
        assert!(item.apply_graph_command(
            GraphCommand::PanViewport {
                delta: GraphPoint { x: 5.0, y: 7.0 },
            },
            cx,
        ));
    });
    let external = br#"{"version":0.4,"last_node_id":0,"last_link_id":0,"nodes":[],"links":[],"groups":[],"config":{},"extra":{},"external":true}"#.to_vec();
    fs.insert_file(local_path, external.clone()).await;
    cx.background_executor
        .timer(Duration::from_millis(250))
        .await;
    assert_eq!(
        item.read_with(cx, |item, _| item.model.save_coordinator.authority()),
        WorkflowAuthority::Conflict
    );
    let conflict_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode conflict state");
    let comparison = item.read_with(cx, |item, _| item.conflict_comparison());
    assert_ne!(comparison.base, comparison.local);
    assert_eq!(comparison.external.as_deref(), Some(external.as_slice()));
    let conflict_copy_path = project
        .read_with(cx, |project, cx| {
            project.find_project_path("project/conflict-copy.json", cx)
        })
        .expect("resolve conflict copy path");
    let save_copy = item.update(cx, |item, cx| {
        item.export_to_path(project.clone(), conflict_copy_path, cx)
    });
    save_copy.await.expect("save conflict copy");
    assert_eq!(
        fs.load_bytes(Path::new("/project/conflict-copy.json"))
            .await
            .expect("read conflict copy"),
        comparison.local
    );
    let keep_local_window = cx.add_window(|_, cx| {
        GraphWorkspaceItem::new(
            GraphWorkspaceModel::decode(&conflict_state).expect("decode conflict state"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    let kept_local_state = keep_local_window
        .update(cx, |item, _, cx| {
            assert!(item.keep_local_version(cx));
            assert_eq!(
                item.model.save_coordinator.authority(),
                WorkflowAuthority::LocalDirty
            );
            assert!(item.model.save_coordinator.external().is_none());
            item.model.encode()
        })
        .expect("keep local conflict version")
        .expect("encode kept-local state");
    let conflict_state = [
        conflict_state,
        comparison.base,
        comparison.external.unwrap_or_default(),
        kept_local_state,
    ]
    .concat();
    let reload = item.update_in(cx, |item, window, cx| {
        item.reload(project.clone(), window, cx)
    });
    reload.await.expect("reload external workflow");
    assert_eq!(
        item.read_with(cx, |item, _| item
            .model
            .save_coordinator
            .base()
            .bytes
            .clone()),
        external
    );
    let autosave = item.update_in(cx, |item, window, cx| {
        item.save(
            workspace::item::SaveOptions {
                autosave: true,
                ..Default::default()
            },
            project.clone(),
            window,
            cx,
        )
    });
    autosave.await.expect("autosave local workflow");
    let local_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode local workflow state");

    let mut interrupted = fixture_model().expect("create interrupted-save fixture");
    let observed_revision = interrupted.save_coordinator.base().revision.clone();
    interrupted
        .prepare_save(
            Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1604),
            observed_revision,
            "crash-target",
            false,
        )
        .expect("prepare save before crash");
    let interrupted_snapshot = interrupted.encode().expect("encode prepared save");
    let mut interrupted =
        GraphWorkspaceModel::decode(&interrupted_snapshot).expect("restore prepared save");
    interrupted.recover_after_restart();
    assert_eq!(
        interrupted.save_coordinator.authority(),
        WorkflowAuthority::Interrupted
    );
    let interrupted_state = interrupted.encode().expect("encode interrupted state");

    item.update(cx, |item, cx| {
        item.model
            .save_coordinator
            .switch_provider_after_committed_save(WorkflowStorageProvider::Provider {
                identifier: "fixture.provider".to_owned(),
            })
            .expect("switch to fixture provider after committed save");
        cx.notify();
    });
    let provider_save = item.update_in(cx, |item, window, cx| {
        item.save(Default::default(), project.clone(), window, cx)
    });
    assert!(provider_save.await.is_err());
    let provider_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode provider error state");
    let provider_snapshot: Value =
        serde_json::from_slice(&provider_state).expect("parse provider snapshot");
    let provider_engine = GraphCommandEngine::decode(
        &BASE64
            .decode(
                provider_snapshot["engine"]
                    .as_str()
                    .expect("provider engine is compact base64"),
            )
            .expect("decode provider engine bytes"),
    )
    .expect("decode provider engine");
    let provider_coordinator = WorkflowSaveCoordinator::decode(
        &BASE64
            .decode(
                provider_snapshot["save_journal"]
                    .as_str()
                    .expect("provider save journal is compact base64"),
            )
            .expect("decode provider save journal bytes"),
    )
    .expect("decode provider save coordinator");
    let provider_saved_document =
        GraphDocument::from_workflow_bytes(provider_coordinator.local_bytes())
            .expect("parse provider save journal workflow");
    assert!(
        provider_engine
            .document
            .has_same_persisted_workflow(&provider_saved_document)
            .expect("compare provider snapshot owners"),
        "provider snapshot engine={} journal={}",
        provider_engine
            .document
            .to_workflow_value()
            .expect("serialize provider engine for diagnostics"),
        provider_saved_document
            .to_workflow_value()
            .expect("serialize provider journal for diagnostics")
    );
    let restored_provider_state =
        GraphWorkspaceModel::decode(&provider_state).expect("restore provider error state");
    assert!(restored_provider_state.last_error.is_some());
    assert!(!restored_provider_state.operation_errors.is_empty());
    assert!(
        restored_provider_state
            .announcement
            .as_deref()
            .is_some_and(|announcement| announcement.contains("prior failure"))
    );

    let isolated_window = cx.add_window(|_, cx| {
        GraphWorkspaceItem::new(
            fixture_model().expect("create isolated-window graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    cx.run_until_parked();
    isolated_window
        .update(cx, |item, window, cx| {
            item.focus_graph(window, cx);
            assert!(item.focus_handle.is_focused(window));
            assert!(item.apply_graph_command(
                GraphCommand::PanViewport {
                    delta: GraphPoint { x: 19.0, y: 23.0 },
                },
                cx,
            ));
            item.model.execution_association = Some("isolated-attempt".to_owned());
        })
        .expect("edit isolated-window graph");
    let isolated_state = isolated_window
        .read_with(cx, |isolated, cx| {
            assert!(isolated.is_dirty(cx));
            assert!(isolated.can_save(cx));
            assert!(matches!(
                isolated.model.save_coordinator.provider(),
                WorkflowStorageProvider::Draft
            ));
            isolated.model.encode()
        })
        .expect("read isolated-window graph")
        .expect("encode isolated-window graph state");
    let mut restored_isolated =
        GraphWorkspaceModel::decode(&isolated_state).expect("restore isolated window state");
    assert_eq!(
        restored_isolated.execution_association.as_deref(),
        Some("isolated-attempt")
    );
    assert!(
        restored_isolated
            .document()
            .and_then(|document| document.profile_identity)
            .is_some()
    );
    assert!(
        restored_isolated
            .undo()
            .expect("restore isolated undo history")
    );
    let close_dispositions = isolated_window
        .update(cx, |isolated, window, cx| {
            let cancel = isolated.resolve_close_request(GraphDirtyCloseChoice::Cancel, cx);
            assert!(isolated.focus_handle.is_focused(window));
            let save = isolated.resolve_close_request(GraphDirtyCloseChoice::Save, cx);
            assert!(isolated.focus_handle.is_focused(window));
            let discard = isolated.resolve_close_request(GraphDirtyCloseChoice::Discard, cx);
            assert!(isolated.focus_handle.is_focused(window));
            (cancel, save, discard)
        })
        .expect("exercise dirty-close decisions");
    assert_eq!(
        close_dispositions,
        (
            GraphCloseDisposition::KeepOpen,
            GraphCloseDisposition::SaveThenClose,
            GraphCloseDisposition::CloseWithoutSaving,
        )
    );
    let mut close_decision_state = isolated_state.clone();
    close_decision_state.extend_from_slice(format!("{close_dispositions:?}").as_bytes());

    let lifecycle_window = cx.add_window(|_, cx| {
        GraphWorkspaceItem::new_draft("Created workflow", WeakEntity::new_invalid(), cx)
            .expect("create lifecycle workflow")
    });
    cx.run_until_parked();
    let created_path = project
        .read_with(cx, |project, cx| {
            project.find_project_path("project/created-workflow.json", cx)
        })
        .expect("resolve created workflow path");
    let save_created = lifecycle_window
        .update(cx, |lifecycle, window, cx| {
            lifecycle.save_as(project.clone(), created_path, window, cx)
        })
        .expect("start lifecycle workflow save");
    save_created.await.expect("save created workflow");
    assert!(
        fs.is_file(Path::new("/project/created-workflow.json"))
            .await
    );
    fs.insert_file(
        Path::new("/project/existing-workflow.json"),
        b"protected-existing-workflow".to_vec(),
    )
    .await;
    let collision_window = cx.add_window(|_, cx| {
        GraphWorkspaceItem::new_draft("Collision workflow", WeakEntity::new_invalid(), cx)
            .expect("create collision workflow")
    });
    let existing_path = project
        .read_with(cx, |project, cx| {
            project.find_project_path("project/existing-workflow.json", cx)
        })
        .expect("resolve existing workflow path");
    let collision_save = collision_window
        .update(cx, |collision, window, cx| {
            collision.save_as(project.clone(), existing_path, window, cx)
        })
        .expect("start colliding save-as");
    assert!(collision_save.await.is_err());
    assert_eq!(
        fs.load_bytes(Path::new("/project/existing-workflow.json"))
            .await
            .expect("read protected workflow"),
        b"protected-existing-workflow"
    );

    let renamed_path = project
        .read_with(cx, |project, cx| {
            project.find_project_path("project/renamed-workflow.json", cx)
        })
        .expect("resolve renamed workflow path");
    let rename_file = lifecycle_window
        .update(cx, |lifecycle, _, cx| {
            lifecycle.rename_local_file(project.clone(), renamed_path, cx)
        })
        .expect("start workflow file rename");
    rename_file.await.expect("rename workflow file");
    assert!(
        !fs.is_file(Path::new("/project/created-workflow.json"))
            .await
    );
    assert!(
        fs.is_file(Path::new("/project/renamed-workflow.json"))
            .await
    );
    lifecycle_window
        .update(cx, |lifecycle, _, cx| {
            assert!(lifecycle.rename_workflow("Renamed lifecycle workflow", cx));
            assert_eq!(lifecycle.model.title, "Renamed lifecycle workflow");
        })
        .expect("rename workflow title");

    let exported_path = project
        .read_with(cx, |project, cx| {
            project.find_project_path("project/exported-workflow.json", cx)
        })
        .expect("resolve workflow export path");
    let export = lifecycle_window
        .update(cx, |lifecycle, _, cx| {
            lifecycle.export_to_path(project.clone(), exported_path, cx)
        })
        .expect("start workflow export");
    export.await.expect("export workflow copy");
    let exported_bytes = fs
        .load_bytes(Path::new("/project/exported-workflow.json"))
        .await
        .expect("read exported workflow");
    assert!(!exported_bytes.is_empty());
    let duplicate_export = lifecycle_window
        .update(cx, |lifecycle, _, cx| {
            let path = project
                .read(cx)
                .find_project_path("project/exported-workflow.json", cx)
                .expect("resolve duplicate export path");
            lifecycle.export_to_path(project.clone(), path, cx)
        })
        .expect("start duplicate export");
    assert!(duplicate_export.await.is_err());
    assert_eq!(
        fs.load_bytes(Path::new("/project/exported-workflow.json"))
            .await
            .expect("read protected export"),
        exported_bytes
    );

    let cancelled_delete = lifecycle_window
        .update(cx, |lifecycle, _, cx| {
            lifecycle.delete_local_file(project.clone(), GraphDeleteFileChoice::Cancel, cx)
        })
        .expect("cancel workflow file deletion");
    cancelled_delete.await.expect("cancel deletion cleanly");
    assert!(
        fs.is_file(Path::new("/project/renamed-workflow.json"))
            .await
    );
    let delete = lifecycle_window
        .update(cx, |lifecycle, _, cx| {
            lifecycle.delete_local_file(project.clone(), GraphDeleteFileChoice::Confirm, cx)
        })
        .expect("start workflow file deletion");
    delete.await.expect("delete workflow file");
    assert!(
        !fs.is_file(Path::new("/project/renamed-workflow.json"))
            .await
    );
    assert!(
        fs.is_file(Path::new("/project/exported-workflow.json"))
            .await
    );
    let lifecycle_projection = lifecycle_window
        .read_with(cx, |lifecycle, _| {
            assert!(matches!(
                lifecycle.model.save_coordinator.provider(),
                WorkflowStorageProvider::Draft
            ));
            assert_eq!(
                lifecycle.model.save_coordinator.authority(),
                WorkflowAuthority::LocalDirty
            );
            assert_eq!(
                lifecycle
                    .model
                    .document()
                    .and_then(|document| document.profile_identity),
                Some(active_profile_id.0)
            );
            json!({
                "title": lifecycle.model.title,
                "provider": "draft",
                "authority": "local-dirty",
                "profile_identity": active_profile_id.0.to_string(),
            })
        })
        .expect("read lifecycle workflow");
    let mut exported_workflow: Value =
        serde_json::from_slice(&exported_bytes).expect("parse exported lifecycle workflow");
    exported_workflow["id"] = Value::String("generated-document-identity".to_owned());
    let lifecycle_state = serde_json::to_vec(&json!({
        "workspace": lifecycle_projection,
        "exported_workflow": exported_workflow,
    }))
    .expect("encode deterministic lifecycle evidence");
    item.update_in(cx, |item, window, _| {
        assert!(item.focus_handle.is_focused(window));
        assert!(matches!(
            item.model.save_coordinator.provider(),
            WorkflowStorageProvider::Provider { .. }
        ));
    });
    write_artifact(
        "val-gpui-004.json",
        "VAL-GPUI-004",
        json!({
            "semantics": format!("{:x}", Sha256::digest(semantics.as_bytes())),
        }),
        vec![
            state_case(
                "application-node-port-semantics-rendered",
                semantics.as_bytes(),
            ),
            state_case("tab-focus-remains-within-graph-without-trap", &final_state),
            state_case("keyboard-actions-preserve-focus-ownership", &final_state),
            state_case(
                "snapshot-owner-rejects-split-engine-journal-and-preserves-workspace-state",
                &snapshot_owner_state,
            ),
            state_case("create-open-import-and-read-only-recovery", &import_state),
            state_case("save-as-autosave-and-local-reopen", &local_state),
            state_case(
                "external-deletion-requires-keep-local-before-atomic-recreation",
                &external_deletion_state,
            ),
            state_case("external-change-conflict-before-reload", &conflict_state),
            state_case("prepared-save-crash-recovery", &interrupted_state),
            state_case("provider-save-error-propagation", &provider_state),
            state_case(
                "dirty-close-policy-and-multi-window-state-isolation",
                &close_decision_state,
            ),
            state_case(
                "create-rename-export-delete-workflow-lifecycle",
                &lifecycle_state,
            ),
        ],
    )
    .expect("write VAL-GPUI-004 artifact");
}

#[gpui::test(seed = 16005)]
fn native_widgets_are_operable_and_preserve_prompt_values(cx: &mut TestAppContext) {
    let (item, cx) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            fixture_model().expect("create graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    cx.cx.update(|cx| {
        cx.bind_keys([
            KeyBinding::new("r", RefreshNodeDefinitions, Some(SIM_KEYMAP_CONTEXT)),
            KeyBinding::new("w", ToggleWorkflowsSidebar, Some(SIM_KEYMAP_CONTEXT)),
            KeyBinding::new("n", ToggleNodeLibrarySidebar, Some(SIM_KEYMAP_CONTEXT)),
            KeyBinding::new("m", ToggleModelLibrarySidebar, Some(SIM_KEYMAP_CONTEXT)),
            KeyBinding::new("a", ToggleAssetsSidebar, Some(SIM_KEYMAP_CONTEXT)),
            KeyBinding::new(".", GraphFitView, Some(SIM_KEYMAP_CONTEXT)),
            KeyBinding::new("p", ToggleSelectedItemsPin, Some(SIM_KEYMAP_CONTEXT)),
            KeyBinding::new("v", UnlockCanvas, Some(SIM_KEYMAP_CONTEXT)),
            KeyBinding::new("h", LockCanvas, Some(SIM_KEYMAP_CONTEXT)),
        ]);
    });
    assert!(cx.debug_bounds("SIM-GRAPH-ANNOUNCEMENT").is_none());

    let enabled = cx
        .debug_bounds("SIM-WIDGET-source-enabled")
        .expect("boolean widget bounds");
    cx.simulate_click(enabled.center(), Modifiers::default());
    let enabled_values = item.read_with(cx, |item, _| {
        let widget = &item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph")
            .nodes[&GraphIdentifier::from("source")]
            .widgets[1];
        (widget.value.clone(), widget.prompt_value.clone())
    });
    assert_eq!(enabled_values, (Value::Bool(true), Value::Bool(true)));
    assert!(cx.debug_bounds("SIM-GRAPH-ANNOUNCEMENT").is_some());

    let steps = cx
        .debug_bounds("SIM-WIDGET-source-steps")
        .expect("integer widget bounds");
    cx.simulate_click(steps.center(), Modifiers::default());
    cx.simulate_keystrokes("up");
    let steps_values = item.read_with(cx, |item, _| {
        let widget = &item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph")
            .nodes[&GraphIdentifier::from("source")]
            .widgets[0];
        (widget.value.clone(), widget.prompt_value.clone())
    });
    assert_eq!(steps_values, (Value::from(22), Value::from(22)));

    let strength = cx
        .debug_bounds("SIM-WIDGET-source-strength")
        .expect("float widget bounds");
    cx.simulate_click(strength.center(), Modifiers::default());
    let mode = cx
        .debug_bounds("SIM-WIDGET-source-mode")
        .expect("combo widget bounds");
    cx.simulate_click(mode.center(), Modifiers::default());
    let label = cx
        .debug_bounds("SIM-WIDGET-source-label")
        .expect("text widget bounds");
    cx.simulate_click(label.center(), Modifiers::default());
    let (before_text_document, before_trace_length, before_error) =
        item.read_with(cx, |item, _| {
            (
                item.model().document().expect("graph document").clone(),
                item.shell_dispatch_trace_for_test().len(),
                item.model().last_error.clone(),
            )
        });
    cx.simulate_keystrokes("r w n m a . p v h");
    let (mut after_text_document, after_trace_length, after_error) =
        item.read_with(cx, |item, _| {
            (
                item.model().document().expect("graph document").clone(),
                item.shell_dispatch_trace_for_test().len(),
                item.model().last_error.clone(),
            )
        });
    let before_label = before_text_document
        .root
        .nodes
        .get(&GraphIdentifier::from("source"))
        .and_then(|node| node.widgets.get(4))
        .expect("source label before text entry")
        .clone();
    let after_label = after_text_document
        .root
        .nodes
        .get_mut(&GraphIdentifier::from("source"))
        .expect("source node after text entry")
        .widgets
        .get_mut(4)
        .expect("source label after text entry");
    *after_label = before_label;
    assert_eq!(after_text_document, before_text_document);
    assert_eq!(after_trace_length, before_trace_length);
    assert_eq!(after_error, before_error);
    item.read_with(cx, |item, _| {
        let widgets = &item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph")
            .nodes[&GraphIdentifier::from("source")]
            .widgets;
        assert_eq!(widgets[2].value, Value::from(0.75));
        assert_eq!(widgets[2].value, widgets[2].prompt_value);
        assert_eq!(widgets[3].value, Value::String("quality".to_owned()));
        assert_eq!(widgets[3].value, widgets[3].prompt_value);
        assert_eq!(
            widgets[4].value,
            Value::String("nativerwnma.pvh".to_owned())
        );
        assert_eq!(widgets[4].value, widgets[4].prompt_value);
    });

    let before_placeholder = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode before placeholder interaction");
    let placeholder = cx
        .debug_bounds("SIM-WIDGET-source-extension-widget")
        .expect("preserved widget placeholder bounds");
    cx.simulate_click(placeholder.center(), Modifiers::default());
    let after_placeholder = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode after placeholder interaction");
    assert_eq!(before_placeholder, after_placeholder);
}

#[gpui::test(seed = 16007)]
fn control_focus_handles_are_stable_and_pruned_with_rendered_controls(cx: &mut TestAppContext) {
    let (item, cx) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            fixture_model().expect("create graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    let (initial_count, output_focus) = item.update(cx, |item, cx| {
        (
            item.control_focus_handle_count(),
            item.control_focus_handle("output:source:0", cx),
        )
    });
    assert!(initial_count > 0);
    item.update(cx, |item, cx| {
        assert!(item.apply_graph_command(
            GraphCommand::PanViewport {
                delta: GraphPoint { x: 7.0, y: 11.0 },
            },
            cx,
        ));
    });
    cx.run_until_parked();
    let output_focus_after = item.update(cx, |item, cx| {
        item.control_focus_handle("output:source:0", cx)
    });
    assert_eq!(output_focus, output_focus_after);

    item.update_in(cx, |item, window, cx| item.focus_graph(window, cx));
    cx.dispatch_action(GraphDelete);
    assert_eq!(
        item.read_with(cx, |item, _| item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.nodes.len())),
        Some(0)
    );
    item.read_with(cx, |item, _| {
        assert_eq!(item.control_focus_handle_count(), 1);
        assert!(item.contains_control_focus_handle("execution:execute-button"));
        assert!(!item.contains_control_focus_handle("output:source:0"));
    });
}

struct NativeRendererEvidence {
    group_state: Vec<u8>,
    boundary_hit_state: Vec<u8>,
    reroute_reconnect_state: Vec<u8>,
    scaled_geometry_state: Vec<u8>,
    model_action_state: Vec<u8>,
    minimap_state: Vec<u8>,
    breadcrumb_state: Vec<u8>,
}

fn exercise_native_renderer_routes_groups_breadcrumbs_and_minimap(
    cx: &mut TestAppContext,
) -> NativeRendererEvidence {
    let mut model = fixture_model().expect("create graph fixture");
    model
        .apply(GraphCommand::AddNode {
            node: fixture_node(
                "target-two",
                GraphPoint { x: 450.0, y: 460.0 },
                Some(GraphPortType::Concrete("IMAGE".to_owned())),
                None,
            ),
            source: sim_runtime::NodeCreationSource::Library,
        })
        .expect("add reconnect target");
    let group_identifier = GraphIdentifier::from("fixture-group");
    model
        .apply(GraphCommand::CreateGroup {
            group: sim_runtime::GraphGroup {
                identifier: group_identifier.clone(),
                title: "Collapsed fixture".to_owned(),
                bounds: GraphRect {
                    origin: GraphPoint { x: 80.0, y: 80.0 },
                    size: GraphSize {
                        width: 650.0,
                        height: 360.0,
                    },
                },
                node_ids: BTreeSet::from([
                    GraphIdentifier::from("source"),
                    GraphIdentifier::from("target"),
                ]),
                collapsed: true,
                pinned: false,
                color: None,
                source_fields: serde_json::Map::new(),
            },
        })
        .expect("create collapsed group");
    let mut viewport = model
        .document()
        .and_then(|document| document.active_graph().ok())
        .expect("active graph")
        .viewport
        .clone();
    viewport.minimap_visible = true;
    model
        .apply(GraphCommand::SetViewport { viewport })
        .expect("show minimap");
    let collapsed_state = model.encode().expect("encode collapsed-group state");
    let (item, cx) =
        cx.add_window_view(|_, cx| GraphWorkspaceItem::new(model, WeakEntity::new_invalid(), cx));
    assert!(cx.debug_bounds("SIM-GROUP-fixture-group").is_some());
    assert!(cx.debug_bounds("SIM-NODE-source").is_none());
    assert!(cx.debug_bounds("SIM-NODE-target").is_none());
    assert!(cx.debug_bounds("SIM-MINIMAP").is_some());

    item.update(cx, |item, cx| {
        assert!(item.apply_graph_command(
            GraphCommand::ToggleGroups {
                identifiers: BTreeSet::from([group_identifier.clone()]),
                toggle: GroupToggle::Collapse,
            },
            cx,
        ));
    });
    assert!(cx.debug_bounds("SIM-NODE-source").is_some());
    assert!(cx.debug_bounds("SIM-NODE-target").is_some());
    let expanded_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode expanded-group state");
    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle("group:fixture-group", cx);
        window.focus(&focus, cx);
    });
    cx.simulate_keystrokes("f");
    cx.simulate_keystrokes("u");
    assert!(item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .is_some_and(|graph| !graph.groups.contains_key(&group_identifier))
    }));
    let ungrouped_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode keyboard fit-and-ungroup state");
    let group_state = [collapsed_state, expanded_state, ungrouped_state].concat();

    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle("node:source", cx);
        window.focus(&focus, cx);
    });
    cx.simulate_keystrokes("f2");
    cx.simulate_keystrokes("x");
    cx.simulate_keystrokes("enter");
    cx.simulate_keystrokes("c");
    cx.simulate_keystrokes("d");
    item.read_with(cx, |item, _| {
        let source = &item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph after node keyboard actions")
            .nodes[&GraphIdentifier::from("source")];
        assert_eq!(source.title, "sourcex");
        assert_eq!(source.color.as_deref(), Some("#355287"));
        assert_eq!(source.mode, sim_runtime::GraphNodeMode::OnEvent);
    });
    item.update(cx, |item, cx| {
        assert!(item.apply_graph_command(
            GraphCommand::SetSelection {
                selection: GraphSelection {
                    nodes: BTreeSet::from([
                        GraphIdentifier::from("source"),
                        GraphIdentifier::from("target-two"),
                    ]),
                    ..GraphSelection::default()
                },
                mode: SelectionMode::Replace,
            },
            cx,
        ));
    });
    item.update_in(cx, |item, window, cx| item.focus_graph(window, cx));
    cx.simulate_keystrokes("alt-l");
    item.read_with(cx, |item, _| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph after layout shortcut");
        assert_eq!(
            graph.nodes[&GraphIdentifier::from("source")].position.y,
            graph.nodes[&GraphIdentifier::from("target-two")].position.y
        );
    });

    let output = cx
        .debug_bounds("SIM-OUTPUT-source-0")
        .expect("source output bounds");
    let input = cx
        .debug_bounds("SIM-INPUT-target-0")
        .expect("target input bounds");
    cx.simulate_mouse_down(output.center(), MouseButton::Left, Modifiers::default());
    cx.simulate_mouse_move(
        point(output.center().x + px(80.0), output.center().y + px(20.0)),
        MouseButton::Left,
        Modifiers::default(),
    );
    assert!(cx.debug_bounds("SIM-PENDING-LINK").is_some());
    cx.simulate_mouse_up(input.center(), MouseButton::Left, Modifiers::default());
    let link_identifier = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.links.keys().next().cloned())
            .expect("created link")
    });
    let link_bounds = (0..64)
        .find_map(|index| {
            let selector: &'static str =
                Box::leak(format!("SIM-LINK-{}-{index}", link_identifier.text()).into_boxed_str());
            cx.debug_bounds(selector).filter(|bounds| {
                bounds.center().x > output.center().x + px(20.0)
                    && bounds.center().x < input.center().x - px(20.0)
            })
        })
        .expect("link hit target away from node hitboxes");
    let mut boundary_hit_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode boundary-hit graph state");
    boundary_hit_state.extend_from_slice(format!("{link_bounds:?}").as_bytes());
    assert_eq!(
        crate::graph_render::link_at_screen_point(
            &[(
                GraphIdentifier::from("long-link"),
                vec![
                    GraphPoint { x: 0.0, y: 10.0 },
                    GraphPoint {
                        x: 2_000.0,
                        y: 10.0,
                    },
                ],
            )],
            GraphPoint { x: 777.0, y: 16.0 },
        ),
        Some(GraphIdentifier::from("long-link")),
        "continuous geometric hit testing must not leave gaps on long links"
    );
    boundary_hit_state.extend_from_slice(b"continuous-long-link-hit=true");
    cx.simulate_click(link_bounds.center(), Modifiers::alt());
    let reroute_identifier = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.reroutes.keys().next().cloned())
            .expect("inserted reroute")
    });
    let reroute_selector: &'static str =
        Box::leak(format!("SIM-REROUTE-{}", reroute_identifier.text()).into_boxed_str());
    assert!(cx.debug_bounds(reroute_selector).is_some());
    item.update(cx, |item, cx| {
        assert!(item.apply_graph_command(
            GraphCommand::AddReroute {
                reroute: GraphReroute {
                    identifier: GraphIdentifier::from("keyboard-parent"),
                    position: GraphPoint { x: 260.0, y: 280.0 },
                    parent: None,
                    floating_type: None,
                    source_fields: serde_json::Map::new(),
                },
            },
            cx,
        ));
    });
    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle(format!("reroute:{}", reroute_identifier.text()), cx);
        window.focus(&focus, cx);
    });
    let reroute_position_before = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active reroute graph")
            .reroutes[&reroute_identifier]
            .position
    });
    cx.simulate_keystrokes("right");
    cx.simulate_keystrokes("p");
    item.read_with(cx, |item, _| {
        let reroute = &item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph after reroute keyboard actions")
            .reroutes[&reroute_identifier];
        assert_eq!(
            reroute.position,
            GraphPoint {
                x: reroute_position_before.x + 10.0,
                y: reroute_position_before.y,
            }
        );
        assert_eq!(
            reroute.parent,
            Some(GraphIdentifier::from("keyboard-parent"))
        );
    });
    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle("reroute:keyboard-parent", cx);
        window.focus(&focus, cx);
    });
    cx.simulate_keystrokes("delete");
    assert!(item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .is_some_and(|graph| {
                !graph
                    .reroutes
                    .contains_key(&GraphIdentifier::from("keyboard-parent"))
                    && graph.reroutes[&reroute_identifier].parent.is_none()
            })
    }));

    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle(format!("link:{}:0", link_identifier.text()), cx);
        window.focus(&focus, cx);
    });
    cx.simulate_keystrokes("r");
    assert!(item.read_with(cx, |item, _| {
        item.pending_reconnect.as_ref() == Some(&link_identifier)
    }));
    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle("input:target-two:0", cx);
        window.focus(&focus, cx);
    });
    cx.simulate_keystrokes("enter");
    item.read_with(cx, |item, _| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph");
        assert_eq!(graph.links.len(), 1);
        assert_eq!(
            graph.links[&link_identifier].target_node,
            GraphIdentifier::from("target-two")
        );
        assert_eq!(
            graph.links[&link_identifier].parent_reroute,
            Some(reroute_identifier.clone()),
            "reconnecting a routed link must preserve its reroute chain"
        );
    });
    let reroute_reconnect_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode reroute reconnect state");
    let model_action_state = reroute_reconnect_state.clone();

    item.update(cx, |item, cx| {
        let mut viewport = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active graph")
            .viewport
            .clone();
        viewport.scale = 4.0;
        assert!(item.apply_graph_command(GraphCommand::SetViewport { viewport }, cx));
    });
    let scaled_output = cx
        .debug_bounds("SIM-OUTPUT-source-0")
        .expect("scaled source output bounds")
        .center();
    let scaled_input = cx
        .debug_bounds("SIM-INPUT-target-two-0")
        .expect("scaled target input bounds")
        .center();
    let (expected_output, expected_input) = item.read_with(cx, |item, _| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("active scaled graph");
        let source = &graph.nodes[&GraphIdentifier::from("source")];
        let target = &graph.nodes[&GraphIdentifier::from("target-two")];
        (
            graph.viewport.graph_to_screen(GraphPoint {
                x: source.position.x + source.size.width,
                y: source.position.y + 42.0,
            }),
            graph.viewport.graph_to_screen(GraphPoint {
                x: target.position.x,
                y: target.position.y + 42.0,
            }),
        )
    });
    let scaled_output_x: f32 = scaled_output.x.into();
    let scaled_output_y: f32 = scaled_output.y.into();
    let scaled_input_x: f32 = scaled_input.x.into();
    let scaled_input_y: f32 = scaled_input.y.into();
    assert!(
        (scaled_output_x - expected_output.x).abs() < 0.01,
        "scaled output x {scaled_output_x} != {}",
        expected_output.x
    );
    assert!(
        (scaled_output_y - expected_output.y).abs() < 0.01,
        "scaled output y {scaled_output_y} != {}",
        expected_output.y
    );
    assert!(
        (scaled_input_x - expected_input.x).abs() < 0.01,
        "scaled input x {scaled_input_x} != {}",
        expected_input.x
    );
    assert!(
        (scaled_input_y - expected_input.y).abs() < 0.01,
        "scaled input y {scaled_input_y} != {}",
        expected_input.y
    );
    let mut scaled_geometry_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode scaled node-port geometry state");
    scaled_geometry_state.extend_from_slice(
        format!(
            "output={scaled_output_x},{scaled_output_y};input={scaled_input_x},{scaled_input_y}"
        )
        .as_bytes(),
    );
    let window_handle = cx.window_handle();
    cx.simulate_window_resize(window_handle, size(px(640.0), px(480.0)));
    cx.run_until_parked();
    let narrow_viewport_width = cx
        .debug_bounds("SIM-MINIMAP-VIEWPORT")
        .expect("narrow-window minimap viewport bounds")
        .size
        .width;
    cx.simulate_window_resize(window_handle, size(px(1080.0), px(720.0)));
    cx.run_until_parked();
    let wide_viewport_width = cx
        .debug_bounds("SIM-MINIMAP-VIEWPORT")
        .expect("wide-window minimap viewport bounds")
        .size
        .width;
    assert!(wide_viewport_width > narrow_viewport_width);

    let viewport_before = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.clone())
    });
    let minimap = cx.debug_bounds("SIM-MINIMAP").expect("minimap bounds");
    cx.simulate_click(
        point(minimap.origin.x + px(8.0), minimap.origin.y + px(8.0)),
        Modifiers::default(),
    );
    let viewport_after = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.clone())
    });
    assert_ne!(viewport_before, viewport_after);
    for bounds in [
        GraphRect {
            origin: GraphPoint { x: -50.0, y: 20.0 },
            size: GraphSize {
                width: 2_000.0,
                height: 100.0,
            },
        },
        GraphRect {
            origin: GraphPoint { x: 30.0, y: -80.0 },
            size: GraphSize {
                width: 120.0,
                height: 2_400.0,
            },
        },
    ] {
        let transform = crate::graph_render::MinimapTransform::new(bounds, 128.0, 80.0, 6.0);
        let graph_point = GraphPoint {
            x: bounds.origin.x + bounds.size.width * 0.37,
            y: bounds.origin.y + bounds.size.height * 0.61,
        };
        let projected = transform
            .project_rect(GraphRect {
                origin: graph_point,
                size: GraphSize {
                    width: 0.0,
                    height: 0.0,
                },
            })
            .origin;
        let restored = transform.unproject_point(projected);
        assert!((restored.x - graph_point.x).abs() < 0.01);
        assert!((restored.y - graph_point.y).abs() < 0.01);
    }
    let mut minimap_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode minimap navigation state");
    minimap_state.extend_from_slice(
        format!("narrow={narrow_viewport_width:?};wide={wide_viewport_width:?}").as_bytes(),
    );

    let subgraph_instance = item.update(cx, |item, cx| {
        assert!(item.apply_graph_command(
            GraphCommand::SetSelection {
                selection: GraphSelection {
                    nodes: BTreeSet::from([
                        GraphIdentifier::from("source"),
                        GraphIdentifier::from("target"),
                    ]),
                    ..GraphSelection::default()
                },
                mode: SelectionMode::Replace,
            },
            cx,
        ));
        assert!(item.execute_catalog_action(
            CatalogGraphAction::ConvertToSubgraph,
            GraphActionInput::SubgraphName("Nested fixture".to_owned()),
            cx,
        ));
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.selection.nodes.iter().next().cloned())
            .expect("selected subgraph instance")
    });
    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle(format!("node:{}", subgraph_instance.text()), cx);
        window.focus(&focus, cx);
    });
    cx.simulate_keystrokes("o");
    assert!(cx.debug_bounds("SIM-SUBGRAPH-BREADCRUMBS").is_some());
    let root_breadcrumb = cx
        .debug_bounds("SIM-BREADCRUMB-0")
        .expect("root breadcrumb bounds");
    cx.simulate_click(root_breadcrumb.center(), Modifiers::default());
    assert!(item.read_with(cx, |item, _| {
        item.model
            .document()
            .is_some_and(|document| document.navigation.is_empty())
    }));
    let breadcrumb_state = item
        .read_with(cx, |item, _| item.model.encode())
        .expect("encode breadcrumb exit state");
    NativeRendererEvidence {
        group_state,
        boundary_hit_state,
        reroute_reconnect_state,
        scaled_geometry_state,
        model_action_state,
        minimap_state,
        breadcrumb_state,
    }
}

#[gpui::test(seed = 16006)]
fn native_renderer_routes_groups_breadcrumbs_and_minimap(cx: &mut TestAppContext) {
    exercise_native_renderer_routes_groups_breadcrumbs_and_minimap(cx);
}
