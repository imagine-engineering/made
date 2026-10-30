use crate::{
    GeneratedGraphContextAction, GeneratedGraphContextSurface, GraphContextDispatchInput,
    GraphContextDispatchOutcome, GraphContextInputModal, GraphContextInvocation,
    GraphContextTarget, GraphPropertiesPanel, GraphWorkspaceItem, GraphWorkspaceModel,
    build_graph_context_menu, graph_context_action_binding, graph_context_infrastructure_bindings,
    graph_context_menu_entries, graph_context_registry,
};
use gpui::{
    AppContext as _, DismissEvent, Focusable, Modifiers, MouseButton, TestAppContext,
    VisualTestContext, WeakEntity, point, px, size,
};
use project::Project;
use serde_json::json;
use sha2::{Digest, Sha256};
use sim_runtime::{
    AssetCollisionPolicy, AssetIdentity, AssetNamespace, ContentRevision, GraphCommand,
    GraphDocument, GraphGroup, GraphIdentifier, GraphLevel, GraphNode, GraphNodeMode,
    GraphPaletteColor, GraphPoint, GraphPort, GraphPortType, GraphRect, GraphReroute,
    GraphSelection, GraphSize, GraphSlotDirection, GraphVisualShape, LayoutOperation,
    SUBGRAPH_BLUEPRINT_ASSET_TAG, SharedAssetService, SubgraphBlueprintCatalog, SubgraphDefinition,
    SubgraphPort, WorkflowStorageProvider, authorize_native_subgraph_library,
    open_native_profile_asset_service,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
};
use tempfile::TempDir;

use crate::graph_tests::{fixture_model, state_case, write_artifact};

fn reset_history(model: &GraphWorkspaceModel) -> GraphWorkspaceModel {
    let document = model.document().expect("editable graph fixture");
    let bytes = document
        .to_workflow_bytes()
        .expect("serialize graph fixture without history");
    GraphWorkspaceModel::open(
        "Native context fixture",
        document.document_identity.to_string(),
        sim_runtime::WorkflowStorageProvider::Draft,
        bytes,
    )
    .expect("reopen graph fixture without history")
}

fn binding(feature_id: &str) -> crate::GraphContextActionBinding {
    graph_context_action_binding(feature_id).expect("resolve exact graph context binding")
}

fn representative_input(action: GeneratedGraphContextAction) -> GraphContextDispatchInput {
    match action {
        GeneratedGraphContextAction::ChooseNodeShape
        | GeneratedGraphContextAction::ChooseGroupNodeShape => {
            GraphContextDispatchInput::Shape(GraphVisualShape::Box)
        }
        GeneratedGraphContextAction::ChooseNodeColor
        | GeneratedGraphContextAction::ChooseGroupColor => {
            GraphContextDispatchInput::PaletteColor(Some(GraphPaletteColor::Blue))
        }
        GeneratedGraphContextAction::ChooseNodeMode
        | GeneratedGraphContextAction::ChooseGroupMode => {
            GraphContextDispatchInput::NodeMode(GraphNodeMode::Never)
        }
        GeneratedGraphContextAction::AlignSelection => {
            GraphContextDispatchInput::Layout(LayoutOperation::AlignLeft)
        }
        GeneratedGraphContextAction::DistributeSelection => {
            GraphContextDispatchInput::Layout(LayoutOperation::DistributeHorizontally)
        }
        GeneratedGraphContextAction::ChooseGroupFontSize => {
            GraphContextDispatchInput::GroupFontSize(22.0)
        }
        GeneratedGraphContextAction::OpenNodeProperties => {
            GraphContextDispatchInput::NodeProperty {
                key: "enabled".to_owned(),
                value: Some(json!(true)),
            }
        }
        _ => GraphContextDispatchInput::None,
    }
}

fn restore_fixture_selection(mut model: GraphWorkspaceModel) -> GraphWorkspaceModel {
    let viewport = model
        .document()
        .and_then(|document| document.active_graph().ok())
        .expect("reopened context fixture graph")
        .viewport
        .clone();
    model
        .replace_ephemeral_graph_state(
            GraphSelection {
                nodes: BTreeSet::from([
                    GraphIdentifier::from("source"),
                    GraphIdentifier::from("target"),
                ]),
                ..GraphSelection::default()
            },
            viewport,
        )
        .expect("restore context fixture selection");
    model
}

fn context_fixture() -> GraphWorkspaceModel {
    let mut model = fixture_model().expect("create base context fixture");
    model
        .apply(GraphCommand::CreateGroup {
            group: GraphGroup {
                identifier: GraphIdentifier::from("fixture-group"),
                title: "Fixture group".to_owned(),
                bounds: GraphRect {
                    origin: GraphPoint { x: 70.0, y: 70.0 },
                    size: GraphSize {
                        width: 700.0,
                        height: 380.0,
                    },
                },
                node_ids: BTreeSet::new(),
                collapsed: false,
                pinned: false,
                color: None,
                source_fields: Default::default(),
            },
        })
        .expect("create context group");
    model
        .apply(GraphCommand::AddReroute {
            reroute: GraphReroute {
                identifier: GraphIdentifier::from("fixture-reroute"),
                position: GraphPoint { x: 900.0, y: 500.0 },
                parent: None,
                floating_type: Some("output".to_owned()),
                source_fields: serde_json::Map::from_iter([(
                    "floating".to_owned(),
                    json!({"type": "IMAGE"}),
                )]),
            },
        })
        .expect("create context reroute");
    let crate::WorkflowOpenState::Editable(engine) = &mut model.open_state else {
        panic!("context fixture must remain editable");
    };
    let source = engine
        .document
        .root
        .nodes
        .get_mut(&GraphIdentifier::from("source"))
        .expect("context source node");
    source.source_fields.insert(
        "properties".to_owned(),
        json!({
            "enabled": false,
            "strength": 0.5,
            "mode": "fast",
            "label": "native",
            "config": {"depth": 2}
        }),
    );
    source.source_fields.insert(
        "properties_info".to_owned(),
        json!([
            {"property": "enabled", "label": "Enabled", "type": "boolean"},
            {"property": "strength", "label": "Strength", "type": "number"},
            {"property": "mode", "label": "Mode", "type": "combo", "values": ["fast", "quality"]},
            {"property": "label", "label": "Label", "type": "string"},
            {"property": "config", "label": "Config", "type": "object"}
        ]),
    );
    restore_fixture_selection(reset_history(&model))
}

fn scoped_node_fixture() -> GraphWorkspaceModel {
    let mut model = context_fixture();
    let crate::WorkflowOpenState::Editable(engine) = &mut model.open_state else {
        panic!("context fixture must remain editable");
    };
    let graph = engine
        .document
        .active_graph_mut()
        .expect("scoped fixture active graph");
    graph
        .nodes
        .get_mut(&GraphIdentifier::from("target"))
        .expect("scoped fixture target")
        .pinned = true;
    graph
        .nodes
        .get_mut(&GraphIdentifier::from("source"))
        .and_then(|node| node.widgets.first_mut())
        .expect("scoped fixture source widget")
        .unknown
        .insert("advanced".to_owned(), json!(true));
    restore_fixture_selection(reset_history(&model))
}

fn publication_fixture(description: &str) -> GraphWorkspaceModel {
    let definition_identifier = GraphIdentifier::from("publication-definition");
    let internal_identifier = GraphIdentifier::from("publication-internal");
    let mut internal = GraphNode::new(
        internal_identifier.clone(),
        "Fixture",
        "Fixture",
        GraphPoint::ZERO,
    );
    internal.inputs.push(GraphPort::new(
        "image",
        GraphPortType::Concrete("IMAGE".to_owned()),
    ));
    internal.outputs.push(GraphPort::new(
        "image",
        GraphPortType::Concrete("IMAGE".to_owned()),
    ));
    let port = |identifier: &str| SubgraphPort {
        identifier: identifier.to_owned(),
        name: "image".to_owned(),
        port_type: GraphPortType::Concrete("IMAGE".to_owned()),
        internal_node: Some(internal_identifier.clone()),
        internal_slot: 0,
        source_fields: Default::default(),
    };
    let definition = SubgraphDefinition {
        identifier: definition_identifier.clone(),
        name: "Publication source".to_owned(),
        graph: Box::new(GraphLevel {
            nodes: BTreeMap::from([(internal_identifier.clone(), internal)]),
            ..GraphLevel::default()
        }),
        inputs: vec![port("input")],
        outputs: vec![port("output")],
        published: false,
        description: description.to_owned(),
        search_aliases: vec!["native publication".to_owned()],
        exposed_widgets: Vec::new(),
        graph_inline: false,
        unknown: BTreeMap::new(),
    };
    let instance_identifier = GraphIdentifier::from("publication-instance");
    let mut instance = GraphNode::new(
        instance_identifier.clone(),
        definition_identifier.text(),
        "Suggested blueprint",
        GraphPoint::ZERO,
    );
    instance.subgraph_definition = Some(definition_identifier.clone());
    instance.inputs.push(GraphPort::new(
        "image",
        GraphPortType::Concrete("IMAGE".to_owned()),
    ));
    instance.outputs.push(GraphPort::new(
        "image",
        GraphPortType::Concrete("IMAGE".to_owned()),
    ));
    let mut document = GraphDocument::default();
    document
        .root
        .nodes
        .insert(instance_identifier.clone(), instance);
    document
        .root
        .definitions
        .insert(definition_identifier, definition);
    document.root.selection = GraphSelection {
        nodes: BTreeSet::from([instance_identifier]),
        ..GraphSelection::default()
    };
    let bytes = document
        .to_workflow_bytes()
        .expect("serialize publication fixture");
    let mut model = GraphWorkspaceModel::open(
        "Publication fixture",
        document.document_identity.to_string(),
        WorkflowStorageProvider::Draft,
        bytes,
    )
    .expect("open publication fixture");
    let crate::WorkflowOpenState::Editable(engine) = &mut model.open_state else {
        panic!("publication fixture must be editable");
    };
    engine.document.root.selection.nodes =
        BTreeSet::from([GraphIdentifier::from("publication-instance")]);
    model
}

fn subgraph_slot_fixture() -> GraphWorkspaceModel {
    let mut model = publication_fixture("Slot fixture");
    let crate::WorkflowOpenState::Editable(engine) = &mut model.open_state else {
        panic!("slot fixture must be editable");
    };
    let definition = engine
        .document
        .root
        .definitions
        .keys()
        .next()
        .expect("slot fixture definition")
        .clone();
    engine.document.navigation = vec![definition];
    model
}

fn native_asset_service() -> Result<(TempDir, SharedAssetService), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let assets = open_native_profile_asset_service("profile", directory.path(), &[])?;
    Ok((directory, assets))
}

fn invocation(item: &GraphWorkspaceItem, target: GraphContextTarget) -> GraphContextInvocation {
    let document = item.model.document().expect("editable context document");
    let graph = document.active_graph().expect("active context graph");
    let bytes = document
        .to_workflow_bytes()
        .expect("serialize current context graph");
    GraphContextInvocation {
        document_identity: document.document_identity,
        content_revision: ContentRevision::from_bytes(&bytes),
        navigation: document.navigation.clone(),
        selection: graph.selection.clone(),
        target,
        screen_position: GraphPoint { x: 320.0, y: 240.0 },
    }
}

fn pump_context_menu_frames(cx: &mut VisualTestContext) {
    for _ in 0..4 {
        cx.update(|window, cx| {
            let _arena_clear_needed = window.draw(cx);
        });
        cx.cx.refresh().expect("schedule deterministic GPUI frame");
        cx.run_until_parked();
    }
}

fn init_context_test(cx: &mut TestAppContext) {
    let fs = fs::FakeFs::new(cx.executor());
    cx.update(|cx| {
        let settings_store = settings::SettingsStore::test(cx);
        cx.set_global(settings_store);
        <dyn fs::Fs>::set_global(fs, cx);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
    });
    cx.run_until_parked();
}

fn workflow_bytes(item: &GraphWorkspaceItem) -> Vec<u8> {
    item.model
        .document()
        .expect("editable context document")
        .to_workflow_bytes()
        .expect("serialize context document")
}

fn context_input_modal_is_open(
    workspace: &gpui::Entity<workspace::Workspace>,
    cx: &mut VisualTestContext,
) -> bool {
    workspace.read_with(cx, |workspace, cx| {
        workspace
            .active_modal::<GraphContextInputModal>(cx)
            .is_some()
    })
}

fn native_catalog(cx: &mut VisualTestContext) -> SubgraphBlueprintCatalog {
    cx.update(|_, cx| {
        crate::native_subgraph_catalog(cx)
            .expect("native subgraph catalog")
            .clone()
    })
}

#[test]
fn generated_context_registry_has_one_native_owner_and_exact_surface_closure() {
    let rows = graph_context_registry()
        .collect::<Result<Vec<_>, _>>()
        .expect("resolve generated context registry");
    assert_eq!(rows.len(), 63);
    assert!(rows.iter().all(|row| {
        !row.source_condition.trim().is_empty()
            && row.item_kind
                == crate::menu_registration(&row.feature_id)
                    .expect("re-resolve generated menu row")
                    .expect("generated menu row")
                    .item_kind
    }));
    assert!(rows.iter().all(|row| {
        row.owner == crate::GRAPH_CONTEXT_MENU_OWNER
            && row.context_surface.is_some()
            && if row.context_surface == Some(GeneratedGraphContextSurface::Infrastructure) {
                matches!(
                    row.status,
                    crate::CommandNativeStatus::Infrastructure { owner }
                        if owner == crate::GRAPH_CONTEXT_MENU_OWNER
                ) && row.context_action.is_none()
                    && row.context_infrastructure.is_some()
            } else {
                row.status == crate::CommandNativeStatus::Executable
                    && row.context_action.is_some()
                    && row.context_infrastructure.is_none()
            }
    }));
    let feature_ids = rows
        .iter()
        .map(|row| row.feature_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(feature_ids.len(), rows.len());
    let actions = rows
        .iter()
        .filter_map(|row| row.context_action)
        .collect::<BTreeSet<_>>();
    assert_eq!(actions.len(), 42);
    assert_eq!(
        rows.iter()
            .filter(|row| row.context_action.is_some())
            .count(),
        55
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.context_infrastructure.is_some())
            .count(),
        8
    );
    let counts = rows.iter().fold(BTreeMap::new(), |mut counts, row| {
        *counts
            .entry(row.context_surface.expect("typed context surface"))
            .or_insert(0usize) += 1;
        counts
    });
    assert_eq!(counts[&GeneratedGraphContextSurface::Canvas], 3);
    assert_eq!(counts[&GeneratedGraphContextSurface::CanvasMode], 2);
    assert_eq!(counts[&GeneratedGraphContextSurface::Selection], 13);
    assert_eq!(counts[&GeneratedGraphContextSurface::Node], 21);
    assert_eq!(counts[&GeneratedGraphContextSurface::Group], 11);
    assert_eq!(counts[&GeneratedGraphContextSurface::Reroute], 2);
    assert_eq!(counts[&GeneratedGraphContextSurface::Slot], 3);
    assert_eq!(counts[&GeneratedGraphContextSurface::Infrastructure], 8);
    let infrastructure_bindings = graph_context_infrastructure_bindings()
        .expect("map generated infrastructure rows to enforced production prerequisites");
    assert_eq!(infrastructure_bindings.len(), 8);
    assert_eq!(
        infrastructure_bindings
            .iter()
            .map(|binding| binding.source)
            .collect::<BTreeSet<_>>()
            .len(),
        8
    );
    let graph_render_source = include_str!("graph_render.rs");
    assert!(graph_render_source.matches("MouseButton::Right").count() >= 5);
    assert!(graph_render_source.contains("is_context_menu_keystroke(event)"));
    assert!(
        graph_render_source
            .matches(".aria_selected(selected)")
            .count()
            >= 4
    );
    assert!(graph_render_source.contains("focus_handle.is_focused(window)"));
    let context_source = include_str!("context_menu.rs");
    assert!(context_source.contains(".role(Role::Dialog)"));
    assert!(context_source.contains("context_entry.radio(IconPosition::Start, checked)"));
}

#[gpui::test(seed = 16014)]
fn val_gpui_014(cx: &mut TestAppContext) {
    init_context_test(cx);
    let registrations = graph_context_registry()
        .collect::<Result<Vec<_>, _>>()
        .expect("resolve all native context registrations");
    assert_eq!(registrations.len(), 63);
    let parity_matrix = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../crates/sim_test_support/parity/parity-matrix.md"
    ));
    for registration in &registrations {
        let row_prefix = format!("| `{}` |", registration.feature_id);
        let matching_rows = parity_matrix
            .lines()
            .filter(|line| line.starts_with(&row_prefix))
            .collect::<Vec<_>>();
        assert_eq!(
            matching_rows.len(),
            1,
            "{} must have one exact parity-matrix decision",
            registration.feature_id
        );
        let parity_row = matching_rows[0];
        assert!(
            parity_row.contains("owner:comfy-parity-graph-context-menu-surfaces"),
            "{} must retain the Task 25 parity owner",
            registration.feature_id
        );
        if registration.context_infrastructure.is_some() {
            assert!(
                !parity_row.contains("| deferred;"),
                "{} is a consumed native prerequisite, not deferred work",
                registration.feature_id
            );
            assert!(
                parity_row.contains("prerequisite"),
                "{} must document its focused prerequisite adapter",
                registration.feature_id
            );
        }
    }
    let fixture = context_fixture();
    let fixture_bytes = fixture
        .document()
        .expect("context fixture document")
        .to_workflow_bytes()
        .expect("serialize context fixture");
    let window = cx.open_window(size(px(1920.0), px(1080.0)), |_, cx| {
        GraphWorkspaceItem::new(fixture, WeakEntity::new_invalid(), cx)
    });
    let item = window.root(cx).expect("graph context window root");
    let cx = VisualTestContext::from_window(window.into(), cx).into_mut();
    cx.run_until_parked();
    let viewport_size = cx.update(|window, _| window.viewport_size());
    assert_eq!(viewport_size, size(px(1920.0), px(1080.0)));

    let slot_item = cx.update(|_, cx| {
        cx.new(|cx| GraphWorkspaceItem::new(subgraph_slot_fixture(), WeakEntity::new_invalid(), cx))
    });
    let primary_item = item.clone();
    let mut surface_targets = item.read_with(cx, |item, _| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("context graph for rendered accessibility");
        vec![
            (
                primary_item.clone(),
                GraphContextTarget::Canvas {
                    graph_position: GraphPoint::ZERO,
                },
            ),
            (primary_item.clone(), GraphContextTarget::Selection),
            (
                primary_item.clone(),
                GraphContextTarget::Node(GraphIdentifier::from("source")),
            ),
            (
                primary_item.clone(),
                GraphContextTarget::Group(
                    graph.groups.keys().next().expect("context group").clone(),
                ),
            ),
            (
                primary_item.clone(),
                GraphContextTarget::Reroute(
                    graph
                        .reroutes
                        .keys()
                        .next()
                        .expect("context reroute")
                        .clone(),
                ),
            ),
        ]
    });
    surface_targets.push((
        slot_item.clone(),
        GraphContextTarget::Slot {
            direction: GraphSlotDirection::Input,
            slot: 0,
        },
    ));
    let mut rendered_accessibility = BTreeMap::new();
    for (target_item, target) in surface_targets {
        let primary_surfaces: &[GeneratedGraphContextSurface] = match &target {
            GraphContextTarget::Canvas { .. } => &[
                GeneratedGraphContextSurface::Canvas,
                GeneratedGraphContextSurface::CanvasMode,
            ],
            GraphContextTarget::Selection => &[GeneratedGraphContextSurface::Selection],
            GraphContextTarget::Node(_) => &[GeneratedGraphContextSurface::Node],
            GraphContextTarget::Group(_) => &[GeneratedGraphContextSurface::Group],
            GraphContextTarget::Reroute(_) => &[GeneratedGraphContextSurface::Reroute],
            GraphContextTarget::Slot { .. } => &[GeneratedGraphContextSurface::Slot],
        };
        let invocation = target_item.read_with(cx, |item, _| invocation(item, target.clone()));
        let projected_entries = target_item.read_with(cx, |item, cx| {
            graph_context_menu_entries(item, &invocation, cx)
                .expect("project context rows for rendered accessibility")
        });
        target_item.update(cx, |item, _| {
            item.context_menu_state = Some(crate::GraphContextMenuState {
                invocation: invocation.clone(),
                return_focus: None,
            });
        });
        let rendered_menu = cx
            .update(|window, cx| build_graph_context_menu(target_item.downgrade(), window, cx))
            .expect("build production GPUI context menu for accessibility inspection");
        let contracts = rendered_menu.read_with(cx, |menu, _| {
            assert!(
                menu.selected_accessibility_index().is_some(),
                "a deployed menu must expose a visible initial item focus"
            );
            menu.accessibility_contracts()
        });
        assert_eq!(contracts.len(), projected_entries.len());
        for (entry, contract) in projected_entries.into_iter().zip(contracts) {
            if primary_surfaces.contains(&entry.surface) {
                assert!(
                    rendered_accessibility
                        .insert(entry.feature_id.clone(), contract)
                        .is_none(),
                    "{} has more than one primary rendered accessibility owner",
                    entry.feature_id
                );
            }
        }
        rendered_menu.update(cx, |_, cx| cx.emit(DismissEvent));
        cx.run_until_parked();
        assert!(target_item.read_with(cx, |item, _| item.context_menu_state.is_none()));
    }
    assert_eq!(rendered_accessibility.len(), 55);

    let mut visible_feature_ids = BTreeSet::new();
    let mut row_cases = Vec::new();
    let slot_entries = slot_item.read_with(cx, |item, cx| {
        let target = GraphContextTarget::Slot {
            direction: GraphSlotDirection::Input,
            slot: 0,
        };
        graph_context_menu_entries(item, &invocation(item, target), cx)
            .expect("convert canonical slot rows into typed context entries")
    });
    visible_feature_ids.extend(slot_entries.iter().map(|entry| entry.feature_id.clone()));
    item.read_with(cx, |item, cx| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("context graph for catalog closure");
        let group_identifier = graph.groups.keys().next().expect("context group").clone();
        let reroute_identifier = graph
            .reroutes
            .keys()
            .next()
            .expect("context reroute")
            .clone();
        for target in [
            GraphContextTarget::Canvas {
                graph_position: GraphPoint::ZERO,
            },
            GraphContextTarget::Selection,
            GraphContextTarget::Node(GraphIdentifier::from("source")),
            GraphContextTarget::Group(group_identifier.clone()),
            GraphContextTarget::Reroute(reroute_identifier.clone()),
        ] {
            let invocation = invocation(item, target);
            let entries = graph_context_menu_entries(item, &invocation, cx)
                .expect("convert canonical registry into typed context entries");
            for entry in entries {
                visible_feature_ids.insert(entry.feature_id);
            }
        }
        for registration in &registrations {
            let surface = registration
                .context_surface
                .expect("native context row has a typed surface");
            if surface == GeneratedGraphContextSurface::Infrastructure {
                row_cases.push(json!({
                    "name": format!("row:{}", registration.feature_id),
                    "feature_id": registration.feature_id,
                    "owner": registration.owner,
                    "surface": surface,
                    "status": "infrastructure",
                    "infrastructure": registration.context_infrastructure,
                    "source_condition": registration.source_condition,
                    "item_kind": registration.item_kind,
                    "projection_consistent": registration.context_action.is_none()
                        && registration.context_infrastructure.is_some(),
                    "fallback_count": 0,
                }));
                continue;
            }
            let target = match surface {
                GeneratedGraphContextSurface::Canvas | GeneratedGraphContextSurface::CanvasMode => {
                    GraphContextTarget::Canvas {
                        graph_position: GraphPoint::ZERO,
                    }
                }
                GeneratedGraphContextSurface::Selection => GraphContextTarget::Selection,
                GeneratedGraphContextSurface::Node => {
                    GraphContextTarget::Node(GraphIdentifier::from("source"))
                }
                GeneratedGraphContextSurface::Group => {
                    GraphContextTarget::Group(group_identifier.clone())
                }
                GeneratedGraphContextSurface::Reroute => {
                    GraphContextTarget::Reroute(reroute_identifier.clone())
                }
                GeneratedGraphContextSurface::Slot => GraphContextTarget::Slot {
                    direction: GraphSlotDirection::Input,
                    slot: 0,
                },
                GeneratedGraphContextSurface::Infrastructure => unreachable!(),
            };
            let matching = if surface == GeneratedGraphContextSurface::Slot {
                slot_entries
                    .iter()
                    .filter(|entry| entry.feature_id == registration.feature_id)
                    .cloned()
                    .collect::<Vec<_>>()
            } else {
                graph_context_menu_entries(item, &invocation(item, target), cx)
                    .expect("materialize typed context row")
                    .into_iter()
                    .filter(|entry| entry.feature_id == registration.feature_id)
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                matching.len(),
                1,
                "{} row projection",
                registration.feature_id
            );
            let entry = &matching[0];
            assert_eq!(
                entry.action,
                registration.context_action.expect("typed action")
            );
            assert_eq!(entry.surface, surface);
            assert_eq!(entry.catalog_label, registration.label);
            assert!(!entry.label.trim().is_empty());
            assert!(entry.enabled || entry.disabled_reason.is_some());
            if entry.enabled
                && matches!(
                    registration.item_kind,
                    crate::GeneratedMenuItemKind::RadioAction
                        | crate::GeneratedMenuItemKind::ToggleAction
                        | crate::GeneratedMenuItemKind::CheckboxAction
                )
            {
                assert!(
                    entry.checked.is_some(),
                    "{} must expose its checked accessibility state",
                    registration.feature_id
                );
            }
            let expected_accessibility_role = match registration.item_kind {
                crate::GeneratedMenuItemKind::RadioAction => "menu-item-radio",
                crate::GeneratedMenuItemKind::ToggleAction
                | crate::GeneratedMenuItemKind::CheckboxAction => "menu-item-checkbox",
                _ => "menu-item",
            };
            let accessibility = rendered_accessibility
                .get(&registration.feature_id)
                .expect("production-rendered accessibility contract");
            assert_eq!(
                accessibility.role, expected_accessibility_role,
                "{} rendered accessibility role",
                registration.feature_id
            );
            assert_eq!(
                accessibility.disabled,
                !entry.enabled,
                "{} rendered accessibility disabled state",
                registration.feature_id
            );
            assert!(accessibility.label.as_ref().starts_with(&entry.label));
            assert_eq!(accessibility.checked, entry.checked);
            let expected_accessibility_expanded =
                (entry.enabled && entry.has_submenu()).then_some(false);
            assert_eq!(
                accessibility.expanded, expected_accessibility_expanded,
                "{} rendered accessibility expansion state",
                registration.feature_id
            );
            row_cases.push(json!({
                "name": format!("row:{}", registration.feature_id),
                "feature_id": registration.feature_id,
                "owner": registration.owner,
                "surface": surface,
                "action": entry.action,
                "catalog_label": entry.catalog_label,
                "visible_label": accessibility.label,
                "enabled": entry.enabled,
                "checked": accessibility.checked,
                "disabled_reason": entry.disabled_reason,
                "source_condition": registration.source_condition,
                "item_kind": registration.item_kind,
                "accessibility_role": accessibility.role,
                "accessibility_expanded": accessibility.expanded,
                "projection_consistent": entry.action == registration.context_action.expect("typed action")
                    && entry.surface == surface
                    && entry.catalog_label == registration.label
                    && accessibility.role == expected_accessibility_role
                    && accessibility.disabled == !entry.enabled
                    && accessibility.checked == entry.checked
                    && accessibility.expanded == expected_accessibility_expanded,
                "fallback_count": 0,
            }));
        }
    });
    assert_eq!(row_cases.len(), 63);
    assert_eq!(visible_feature_ids.len(), 55);
    let action_expectations = row_cases
        .iter()
        .filter_map(|row| {
            let feature_id = row.get("feature_id")?.as_str()?.to_owned();
            let enabled = row.get("enabled")?.as_bool()?;
            let disabled_reason = row
                .get("disabled_reason")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            Some((feature_id, (enabled, disabled_reason)))
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(action_expectations.len(), 55);

    let infrastructure_evidence = graph_context_infrastructure_bindings()
        .expect("resolve enforced production infrastructure prerequisites")
        .into_iter()
        .map(|binding| (binding.feature_id.clone(), binding))
        .collect::<BTreeMap<_, _>>();
    let mut dispatch_evidence = BTreeMap::new();
    for registration in &registrations {
        let surface = registration
            .context_surface
            .expect("native context row surface");
        if surface == GeneratedGraphContextSurface::Infrastructure {
            let adapter = infrastructure_evidence
                .get(&registration.feature_id)
                .expect("production infrastructure adapter");
            let passed = Some(adapter.source) == registration.context_infrastructure;
            dispatch_evidence.insert(
                registration.feature_id.clone(),
                json!({
                    "binding_resolved": passed,
                    "native_prerequisite": format!("{:?}", adapter.source),
                    "dispatch_outcome": "production-prerequisite-enforced",
                    "state_restored": true,
                    "passed": passed,
                }),
            );
            continue;
        }
        let action = registration.context_action.expect("actionable context row");
        let (expected_enabled, expected_disabled_reason) = action_expectations
            .get(&registration.feature_id)
            .expect("row dispatch availability expectation");
        let dispatch_item = if surface == GeneratedGraphContextSurface::Slot {
            slot_item.clone()
        } else {
            item.clone()
        };
        let target = dispatch_item.read_with(cx, |item, _| {
            let graph = item
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .expect("dispatch probe graph");
            match surface {
                GeneratedGraphContextSurface::Canvas | GeneratedGraphContextSurface::CanvasMode => {
                    GraphContextTarget::Canvas {
                        graph_position: GraphPoint::ZERO,
                    }
                }
                GeneratedGraphContextSurface::Selection => GraphContextTarget::Selection,
                GeneratedGraphContextSurface::Node => {
                    GraphContextTarget::Node(GraphIdentifier::from("source"))
                }
                GeneratedGraphContextSurface::Group => GraphContextTarget::Group(
                    graph
                        .groups
                        .keys()
                        .next()
                        .expect("dispatch probe group")
                        .clone(),
                ),
                GeneratedGraphContextSurface::Reroute => GraphContextTarget::Reroute(
                    graph
                        .reroutes
                        .keys()
                        .next()
                        .expect("dispatch probe reroute")
                        .clone(),
                ),
                GeneratedGraphContextSurface::Slot => GraphContextTarget::Slot {
                    direction: GraphSlotDirection::Input,
                    slot: 0,
                },
                GeneratedGraphContextSurface::Infrastructure => unreachable!(),
            }
        });
        let (before, selection_before, viewport_before) = dispatch_item.read_with(cx, |item, _| {
            let graph = item
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .expect("dispatch probe graph before action");
            (
                workflow_bytes(item),
                graph.selection.clone(),
                graph.viewport.clone(),
            )
        });
        let (outcome, state_restored, exact_binding) =
            dispatch_item.update_in(cx, |item, window, cx| {
                let invocation = invocation(item, target);
                let binding = binding(&registration.feature_id);
                let exact_binding = binding
                    == graph_context_action_binding(&registration.feature_id)
                        .expect("re-resolve dispatch binding");
                let outcome = item.dispatch_context_action(
                    binding,
                    representative_input(action),
                    invocation,
                    window,
                    cx,
                );
                let after = workflow_bytes(item);
                let restored = if after == before {
                    true
                } else {
                    item.model.undo().expect("undo dispatch probe mutation")
                        && workflow_bytes(item) == before
                };
                (outcome, restored, exact_binding)
            });
        let confirmation_observed = outcome == GraphContextDispatchOutcome::ConfirmationPending;
        if confirmation_observed {
            assert!(cx.has_pending_prompt());
            cx.simulate_prompt_answer("Cancel");
        }
        cx.run_until_parked();
        dispatch_item.update(cx, |item, _| {
            item.model
                .replace_ephemeral_graph_state(selection_before.clone(), viewport_before.clone())
                .expect("restore dispatch probe selection");
        });
        let final_state_restored = state_restored
            && dispatch_item.read_with(cx, |item, _| workflow_bytes(item) == before)
            && (!confirmation_observed || !cx.has_pending_prompt());
        let outcome_matches_availability = match &outcome {
            GraphContextDispatchOutcome::Rejected(reason) => {
                !expected_enabled
                    && expected_disabled_reason
                        .as_ref()
                        .is_some_and(|expected| expected == reason)
            }
            GraphContextDispatchOutcome::Executed
            | GraphContextDispatchOutcome::InputPending
            | GraphContextDispatchOutcome::ConfirmationPending => *expected_enabled,
        };
        dispatch_evidence.insert(
            registration.feature_id.clone(),
            json!({
                "binding_resolved": exact_binding,
                "dispatch_outcome": format!("{outcome:?}"),
                "expected_enabled": expected_enabled,
                "expected_disabled_reason": expected_disabled_reason,
                "outcome_matches_availability": outcome_matches_availability,
                "confirmation_observed": confirmation_observed,
                "state_restored": final_state_restored,
                "passed": exact_binding && outcome_matches_availability && final_state_restored,
            }),
        );
    }
    for row_case in &mut row_cases {
        let object = row_case.as_object_mut().expect("row case object");
        let feature_id = object
            .get("feature_id")
            .and_then(serde_json::Value::as_str)
            .expect("row case feature ID");
        let evidence = dispatch_evidence
            .get(feature_id)
            .and_then(serde_json::Value::as_object)
            .expect("row dispatch evidence");
        let projection_consistent = object
            .get("projection_consistent")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        for (key, value) in evidence {
            object.insert(key.clone(), value.clone());
        }
        let dispatch_passed = object
            .get("passed")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        object.insert(
            "passed".to_owned(),
            serde_json::Value::Bool(projection_consistent && dispatch_passed),
        );
    }
    assert!(row_cases.iter().all(|row| row["passed"] == json!(true)));

    pump_context_menu_frames(cx);
    let graph_bounds = cx.debug_bounds("SIM-GRAPH").expect("graph pointer target");
    let canvas_point = point(
        graph_bounds.origin.x + graph_bounds.size.width - px(32.0),
        graph_bounds.origin.y + graph_bounds.size.height - px(32.0),
    );
    cx.simulate_mouse_move(canvas_point, None, Modifiers::default());
    pump_context_menu_frames(cx);
    cx.simulate_mouse_down(canvas_point, MouseButton::Right, Modifiers::default());
    assert!(
        item.read_with(cx, |item, _| item.context_menu_state.is_some()),
        "right click at {canvas_point:?} inside {graph_bounds:?} did not reach graph"
    );
    item.update(cx, |item, _| {
        assert!(matches!(
            item.context_menu_state
                .as_ref()
                .map(|state| &state.invocation.target),
            Some(GraphContextTarget::Canvas { .. })
        ));
    });
    let context_menu_handle = item.read_with(cx, |item, _| item.context_menu_handle.clone());
    pump_context_menu_frames(cx);
    assert!(context_menu_handle.is_deployed());
    cx.update(|window, cx| {
        assert!(context_menu_handle.focus(window, cx));
    });
    assert!(item.read_with(cx, |item, _| item.context_menu_handle.is_deployed()));
    assert!(cx.debug_bounds("MENU_ITEM-Add Group").is_some());
    cx.dispatch_action(menu::Cancel);
    pump_context_menu_frames(cx);

    item.update_in(cx, |item, window, cx| {
        window.focus(&item.focus_handle.clone(), cx);
    });
    cx.simulate_keystrokes("shift-f10");
    item.read_with(cx, |item, _| {
        assert!(matches!(
            item.context_menu_state
                .as_ref()
                .map(|state| &state.invocation.target),
            Some(GraphContextTarget::Canvas { .. })
        ));
    });
    cx.dispatch_action(menu::Cancel);
    pump_context_menu_frames(cx);

    let node_bounds = cx
        .debug_bounds("SIM-NODE-source")
        .expect("rendered source node pointer target");
    let node_point = node_bounds.center();
    cx.simulate_mouse_move(node_point, None, Modifiers::default());
    cx.simulate_mouse_down(node_point, MouseButton::Right, Modifiers::default());
    item.read_with(cx, |item, _| {
        assert_eq!(
            item.context_menu_state
                .as_ref()
                .map(|state| &state.invocation.target),
            Some(&GraphContextTarget::Node(GraphIdentifier::from("source")))
        );
    });
    cx.dispatch_action(menu::Cancel);
    pump_context_menu_frames(cx);

    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle("node:source", cx);
        window.focus(&focus, cx);
    });
    cx.simulate_keystrokes("shift-f10");
    item.read_with(cx, |item, _| {
        assert_eq!(
            item.context_menu_state
                .as_ref()
                .map(|state| &state.invocation.target),
            Some(&GraphContextTarget::Node(GraphIdentifier::from("source")))
        );
    });
    let keyboard_menu_position = item.read_with(cx, |item, _| {
        let position = item
            .context_menu_state
            .as_ref()
            .expect("keyboard context invocation")
            .invocation
            .screen_position;
        point(px(position.x), px(position.y))
    });
    cx.update(|window, cx| {
        assert!(context_menu_handle.show_at(keyboard_menu_position, window, cx));
    });
    pump_context_menu_frames(cx);
    cx.update(|window, cx| {
        assert!(context_menu_handle.focus(window, cx));
    });
    item.update_in(cx, |item, window, cx| {
        assert!(item.context_menu_handle.is_deployed());
        assert!(item.context_menu_handle.is_focused(window, cx));
    });
    cx.dispatch_action(menu::Cancel);
    pump_context_menu_frames(cx);

    let before = item.read_with(cx, |item, _| {
        item.model
            .document()
            .expect("context document before collapse")
            .to_workflow_bytes()
            .expect("serialize pre-collapse context document")
    });
    item.update_in(cx, |item, window, cx| {
        let outcome = item.dispatch_context_action(
            binding("COMFY-GRAPH-125"),
            GraphContextDispatchInput::None,
            invocation(
                item,
                GraphContextTarget::Node(GraphIdentifier::from("source")),
            ),
            window,
            cx,
        );
        assert_eq!(outcome, GraphContextDispatchOutcome::Executed);
        assert!(item.model.undo().expect("undo context collapse"));
        assert!(
            !item
                .model
                .undo()
                .expect("collapse has exactly one undo boundary")
        );
    });
    let restored = item.read_with(cx, |item, _| {
        item.model
            .document()
            .expect("context document after undo")
            .to_workflow_bytes()
            .expect("serialize restored context document")
    });
    assert_eq!(restored, before);

    let stale = item.read_with(cx, |item, _| {
        invocation(
            item,
            GraphContextTarget::Node(GraphIdentifier::from("source")),
        )
    });
    item.update_in(cx, |item, _window, cx| {
        item.apply_graph_command(
            GraphCommand::SetNodeProperties {
                identifier: GraphIdentifier::from("source"),
                properties: serde_json::Map::from_iter([(
                    "changed".to_owned(),
                    serde_json::Value::Bool(true),
                )]),
            },
            cx,
        );
    });
    item.update_in(cx, |item, window, cx| {
        let outcome = item.dispatch_context_action(
            binding("COMFY-GRAPH-128"),
            GraphContextDispatchInput::None,
            stale,
            window,
            cx,
        );
        assert!(matches!(outcome, GraphContextDispatchOutcome::Rejected(_)));
    });
    assert!(cx.debug_bounds("SIM-GRAPH-ERROR").is_some());

    let final_bytes = item.read_with(cx, |item, _| {
        item.model
            .document()
            .expect("final context document")
            .to_workflow_bytes()
            .expect("serialize final context document")
    });
    row_cases.extend([
        state_case("pointer-and-keyboard-native-context-invocation", &before),
        state_case("canonical-dispatch-exactly-one-undo", &restored),
        state_case("content-revision-stale-rejection", &final_bytes),
    ]);
    write_artifact(
        "val-gpui-014.json",
        "VAL-GPUI-014",
        json!({
            "workflow": format!("{:x}", Sha256::digest(&fixture_bytes)),
            "frontend_menus": format!("{:x}", Sha256::digest(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../crates/sim_test_support/parity/catalogs/frontend-menus.csv")))),
            "native_menu_dispositions": format!("{:x}", Sha256::digest(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../crates/sim_test_support/parity/catalogs/native-menu-dispositions.csv")))),
            "parity_matrix": format!("{:x}", Sha256::digest(parity_matrix.as_bytes())),
            "catalog_rows": 63,
            "actionable_rows": visible_feature_ids.len(),
            "infrastructure_rows": 8,
            "row_case_count": 63,
        }),
        row_cases,
    )
    .expect("write VAL-GPUI-014 artifact");
}

#[gpui::test(seed = 16020)]
fn exact_feature_bindings_preserve_clicked_and_selection_scope(cx: &mut TestAppContext) {
    init_context_test(cx);
    let window = cx.open_window(size(px(1280.0), px(900.0)), |_, cx| {
        GraphWorkspaceItem::new(scoped_node_fixture(), WeakEntity::new_invalid(), cx)
    });
    let item = window.root(cx).expect("feature scope fixture");
    let cx = VisualTestContext::from_window(window.into(), cx).into_mut();
    cx.run_until_parked();

    let node_invocation = item.read_with(cx, |item, _| {
        invocation(
            item,
            GraphContextTarget::Node(GraphIdentifier::from("source")),
        )
    });
    item.read_with(cx, |item, cx| {
        let entries = graph_context_menu_entries(item, &node_invocation, cx)
            .expect("feature-scoped node entries");
        let enabled = |feature_id: &str| {
            entries
                .iter()
                .find(|entry| entry.feature_id == feature_id)
                .map(|entry| entry.enabled)
                .expect("exact feature entry")
        };
        assert!(enabled("COMFY-MENU-124"));
        assert!(!enabled("COMFY-GRAPH-124"), "{entries:#?}");
        assert!(enabled("COMFY-MENU-125"));
        assert!(!enabled("COMFY-GRAPH-125"));
        assert!(enabled("COMFY-MENU-126"));
    });

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-125"),
                GraphContextDispatchInput::None,
                invocation(
                    item,
                    GraphContextTarget::Node(GraphIdentifier::from("source")),
                ),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::Executed
        );
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("graph after clicked-node collapse");
        assert!(graph.nodes[&GraphIdentifier::from("source")].collapsed);
        assert!(!graph.nodes[&GraphIdentifier::from("target")].collapsed);
    });
    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-126"),
                GraphContextDispatchInput::None,
                invocation(
                    item,
                    GraphContextTarget::Node(GraphIdentifier::from("source")),
                ),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::Executed
        );
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("graph after clicked-node advanced toggle");
        assert_eq!(
            graph.nodes[&GraphIdentifier::from("source")]
                .source_fields
                .get("show_advanced"),
            Some(&json!(true))
        );
        assert!(
            graph.nodes[&GraphIdentifier::from("target")]
                .source_fields
                .get("show_advanced")
                .is_none()
        );
    });
}

#[gpui::test(seed = 16021)]
fn add_group_for_selection_rechecks_pointer_geometry_for_every_target(cx: &mut TestAppContext) {
    init_context_test(cx);
    let window = cx.open_window(size(px(1280.0), px(900.0)), |_, cx| {
        GraphWorkspaceItem::new(context_fixture(), WeakEntity::new_invalid(), cx)
    });
    let item = window.root(cx).expect("group pointer geometry fixture");
    let cx = VisualTestContext::from_window(window.into(), cx).into_mut();
    cx.run_until_parked();

    let inside = GraphPoint { x: 70.0, y: 70.0 };
    let reroute_identifier = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.reroutes.keys().next())
            .cloned()
            .expect("geometry fixture reroute")
    });
    for target in [
        GraphContextTarget::Canvas {
            graph_position: inside,
        },
        GraphContextTarget::Node(GraphIdentifier::from("source")),
        GraphContextTarget::Reroute(reroute_identifier),
    ] {
        let mut invocation = item.read_with(cx, |item, _| invocation(item, target));
        invocation.screen_position = inside;
        item.read_with(cx, |item, cx| {
            let entry = graph_context_menu_entries(item, &invocation, cx)
                .expect("geometry-aware entries")
                .into_iter()
                .find(|entry| entry.feature_id == "COMFY-MENU-152")
                .expect("selection group creation entry");
            assert!(!entry.enabled);
            assert!(
                entry
                    .disabled_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("existing group")),
                "{entry:#?}"
            );
        });
        item.update_in(cx, |item, window, cx| {
            assert!(matches!(
                item.dispatch_context_action(
                    binding("COMFY-MENU-152"),
                    GraphContextDispatchInput::None,
                    invocation.clone(),
                    window,
                    cx,
                ),
                GraphContextDispatchOutcome::Rejected(_)
            ));
        });
    }

    let outside = GraphPoint { x: 900.0, y: 700.0 };
    let outside_invocation = item.read_with(cx, |item, _| {
        let mut invocation = invocation(
            item,
            GraphContextTarget::Canvas {
                graph_position: outside,
            },
        );
        invocation.screen_position = outside;
        invocation
    });
    item.read_with(cx, |item, cx| {
        assert!(
            graph_context_menu_entries(item, &outside_invocation, cx)
                .expect("outside geometry entries")
                .into_iter()
                .find(|entry| entry.feature_id == "COMFY-MENU-152")
                .is_some_and(|entry| entry.enabled)
        );
    });
}

#[gpui::test(seed = 16017)]
fn context_rows_preserve_surface_scope_confirmation_and_stale_settings(cx: &mut TestAppContext) {
    init_context_test(cx);
    let window = cx.open_window(size(px(1280.0), px(900.0)), |_, cx| {
        GraphWorkspaceItem::new(context_fixture(), WeakEntity::new_invalid(), cx)
    });
    let item = window.root(cx).expect("context scope fixture");
    let cx = VisualTestContext::from_window(window.into(), cx).into_mut();
    cx.run_until_parked();
    let original = item.read_with(cx, |item, _| workflow_bytes(item));

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-133"),
                GraphContextDispatchInput::None,
                invocation(
                    item,
                    GraphContextTarget::Node(GraphIdentifier::from("source")),
                ),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::ConfirmationPending
        );
    });
    assert!(cx.has_pending_prompt());
    cx.simulate_prompt_answer("Cancel");
    cx.run_until_parked();
    assert_eq!(item.read_with(cx, |item, _| workflow_bytes(item)), original);

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-133"),
                GraphContextDispatchInput::None,
                invocation(
                    item,
                    GraphContextTarget::Node(GraphIdentifier::from("source")),
                ),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::ConfirmationPending
        );
    });
    cx.simulate_prompt_answer("Delete");
    cx.run_until_parked();
    item.update(cx, |item, _| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("graph after node-row deletion");
        assert!(!graph.nodes.contains_key(&GraphIdentifier::from("source")));
        assert!(graph.nodes.contains_key(&GraphIdentifier::from("target")));
        assert!(item.model.undo().expect("node-row deletion undo"));
        assert!(!item.model.undo().expect("one node-row undo boundary"));
        assert_eq!(workflow_bytes(item), original);
    });

    item.update(cx, |item, _| {
        let viewport = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("restored graph before selection-row deletion")
            .viewport
            .clone();
        item.model
            .replace_ephemeral_graph_state(
                GraphSelection {
                    nodes: BTreeSet::from([
                        GraphIdentifier::from("source"),
                        GraphIdentifier::from("target"),
                    ]),
                    ..GraphSelection::default()
                },
                viewport,
            )
            .expect("restore two-node selection");
    });

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-GRAPH-141"),
                GraphContextDispatchInput::None,
                invocation(
                    item,
                    GraphContextTarget::Node(GraphIdentifier::from("source")),
                ),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::ConfirmationPending
        );
    });
    cx.simulate_prompt_answer("Delete");
    cx.run_until_parked();
    item.update(cx, |item, _| {
        assert!(
            item.model
                .document()
                .and_then(|document| document.active_graph().ok())
                .expect("graph after selection-row deletion")
                .nodes
                .is_empty()
        );
        assert!(item.model.undo().expect("selection-row deletion undo"));
        assert!(!item.model.undo().expect("one selection-row undo boundary"));
        assert_eq!(workflow_bytes(item), original);
    });

    item.update(cx, |item, _| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("graph for single-selection availability");
        item.model
            .replace_ephemeral_graph_state(
                GraphSelection {
                    nodes: BTreeSet::from([GraphIdentifier::from("source")]),
                    ..GraphSelection::default()
                },
                graph.viewport.clone(),
            )
            .expect("set one-node ephemeral selection");
    });
    item.read_with(cx, |item, cx| {
        let entries =
            graph_context_menu_entries(item, &invocation(item, GraphContextTarget::Selection), cx)
                .expect("single-selection context rows");
        let graph_conversion = entries
            .iter()
            .find(|entry| entry.feature_id == "COMFY-GRAPH-135")
            .expect("canonical selection conversion row");
        let legacy_multi_conversion = entries
            .iter()
            .find(|entry| entry.feature_id == "COMFY-MENU-117")
            .expect("legacy multi-selection conversion row");
        assert!(!graph_conversion.enabled);
        assert!(
            graph_conversion
                .disabled_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("workspace modal layer"))
        );
        assert!(!legacy_multi_conversion.enabled);
        assert!(
            legacy_multi_conversion
                .disabled_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("more than one"))
        );
        let panel = graph_context_menu_entries(
            item,
            &invocation(
                item,
                GraphContextTarget::Node(GraphIdentifier::from("source")),
            ),
            cx,
        )
        .expect("node context without workspace")
        .into_iter()
        .find(|entry| entry.feature_id == "COMFY-MENU-121")
        .expect("properties panel row");
        assert!(!panel.enabled);
        assert!(
            panel
                .disabled_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("workspace properties panel"))
        );
    });

    let stale = item.read_with(cx, |item, _| {
        invocation(
            item,
            GraphContextTarget::Reroute(GraphIdentifier::from("fixture-reroute")),
        )
    });
    item.update(cx, |item, cx| {
        assert!(item.apply_graph_command(
            GraphCommand::SetNodeProperties {
                identifier: GraphIdentifier::from("source"),
                properties: serde_json::Map::from_iter([("stale".to_owned(), json!(true))]),
            },
            cx,
        ));
    });
    item.update_in(cx, |item, window, cx| {
        assert!(matches!(
            item.dispatch_context_action(
                binding("COMFY-MENU-143"),
                GraphContextDispatchInput::None,
                stale,
                window,
                cx,
            ),
            GraphContextDispatchOutcome::Rejected(_)
        ));
        assert!(item.context_settings_task.is_none());
        assert!(item.model.undo().expect("undo stale-fixture mutation"));
        assert!(
            !item
                .model
                .undo()
                .expect("stale settings adds no undo entry")
        );
    });
}

#[gpui::test(seed = 16028)]
async fn settings_owned_context_rows_persist_outside_graph_history(cx: &mut TestAppContext) {
    init_context_test(cx);
    let (item, cx) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(context_fixture(), WeakEntity::new_invalid(), cx)
    });
    let workflow_before = item.read_with(cx, |item, _| workflow_bytes(item));
    let native_renderer_before =
        item.read_with(cx, |item, cx| item.native_node_renderer_enabled(cx));
    let default_reroute_before = cx.update(|_, cx| {
        cx.global::<settings::SettingsStore>()
            .merged_settings()
            .sim_runtime
            .as_ref()
            .and_then(|settings| settings.show_reroute_types)
            .unwrap_or(false)
    });

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-080"),
                GraphContextDispatchInput::None,
                invocation(
                    item,
                    GraphContextTarget::Canvas {
                        graph_position: GraphPoint::ZERO,
                    },
                ),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::Executed
        );
        assert!(item.context_settings_task.is_some());
    });
    cx.run_until_parked();
    item.update(cx, |item, cx| {
        assert!(item.context_settings_task.is_none());
        assert_eq!(
            item.native_node_renderer_enabled(cx),
            !native_renderer_before
        );
        assert!(
            item.model
                .announcement
                .as_deref()
                .is_some_and(|message| message.contains("native node renderer enabled"))
        );
        assert_eq!(workflow_bytes(item), workflow_before);
        assert!(!item.model.undo().expect("settings row adds no graph undo"));
    });

    item.update_in(cx, |item, window, cx| {
        let reroute_identifier = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.reroutes.keys().next().cloned())
            .expect("settings fixture reroute");
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-143"),
                GraphContextDispatchInput::None,
                invocation(item, GraphContextTarget::Reroute(reroute_identifier),),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::Executed
        );
        assert!(item.context_settings_task.is_some());
    });
    cx.run_until_parked();
    item.update(cx, |item, _| {
        assert!(item.context_settings_task.is_none());
        assert!(
            item.model
                .announcement
                .as_deref()
                .is_some_and(|message| message.contains("Default reroute type labels"))
        );
        assert_eq!(workflow_bytes(item), workflow_before);
        assert!(!item.model.undo().expect("settings row adds no graph undo"));
    });
    let default_reroute_after = cx.update(|_, cx| {
        cx.global::<settings::SettingsStore>()
            .merged_settings()
            .sim_runtime
            .as_ref()
            .and_then(|settings| settings.show_reroute_types)
            .unwrap_or(false)
    });
    assert_eq!(default_reroute_after, !default_reroute_before);
}

#[gpui::test(seed = 16018)]
fn slot_surface_uses_pointer_keyboard_and_destructive_confirmation(cx: &mut TestAppContext) {
    init_context_test(cx);
    let window = cx.open_window(size(px(1280.0), px(900.0)), |_, cx| {
        GraphWorkspaceItem::new(subgraph_slot_fixture(), WeakEntity::new_invalid(), cx)
    });
    let item = window.root(cx).expect("slot context fixture");
    let cx = VisualTestContext::from_window(window.into(), cx).into_mut();
    pump_context_menu_frames(cx);
    let bounds = cx
        .debug_bounds("SIM-SUBGRAPH-INPUT-0")
        .expect("rendered subgraph input slot");
    let position = bounds.center();
    cx.simulate_mouse_move(position, None, Modifiers::default());
    cx.simulate_mouse_down(position, MouseButton::Right, Modifiers::default());
    item.read_with(cx, |item, _| {
        assert!(matches!(
            item.context_menu_state
                .as_ref()
                .map(|state| &state.invocation.target),
            Some(GraphContextTarget::Slot {
                direction: GraphSlotDirection::Input,
                slot: 0,
            })
        ));
    });
    cx.dispatch_action(menu::Cancel);
    pump_context_menu_frames(cx);

    item.update_in(cx, |item, window, cx| {
        let focus = item.control_focus_handle("subgraph-input:0", cx);
        window.focus(&focus, cx);
    });
    cx.simulate_keystrokes("shift-f10");
    item.read_with(cx, |item, cx| {
        let invocation = item
            .context_menu_state
            .as_ref()
            .expect("slot keyboard invocation")
            .invocation
            .clone();
        assert!(matches!(
            invocation.target,
            GraphContextTarget::Slot {
                direction: GraphSlotDirection::Input,
                slot: 0,
            }
        ));
        let entries =
            graph_context_menu_entries(item, &invocation, cx).expect("typed slot context entries");
        for feature_id in ["COMFY-MENU-139", "COMFY-MENU-141"] {
            assert!(
                entries
                    .iter()
                    .find(|entry| entry.feature_id == feature_id)
                    .is_some_and(|entry| entry.enabled)
            );
        }
        let rename = entries
            .iter()
            .find(|entry| entry.feature_id == "COMFY-MENU-140")
            .expect("slot rename entry");
        assert!(!rename.enabled);
        assert!(
            rename
                .disabled_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("workspace modal layer"))
        );
    });
    cx.dispatch_action(menu::Cancel);
    pump_context_menu_frames(cx);

    let before = item.read_with(cx, |item, _| workflow_bytes(item));
    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-141"),
                GraphContextDispatchInput::None,
                invocation(
                    item,
                    GraphContextTarget::Slot {
                        direction: GraphSlotDirection::Input,
                        slot: 0,
                    },
                ),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::ConfirmationPending
        );
    });
    assert!(cx.has_pending_prompt());
    cx.simulate_prompt_answer("Cancel");
    cx.run_until_parked();
    assert_eq!(item.read_with(cx, |item, _| workflow_bytes(item)), before);
    item.read_with(cx, |item, _| {
        assert!(
            !item
                .model
                .document()
                .expect("slot document")
                .active_subgraph_definition()
                .expect("active slot definition")
                .inputs
                .is_empty()
        );
    });
    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-141"),
                GraphContextDispatchInput::None,
                invocation(
                    item,
                    GraphContextTarget::Slot {
                        direction: GraphSlotDirection::Input,
                        slot: 0,
                    },
                ),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::ConfirmationPending
        );
    });
    cx.simulate_prompt_answer("Remove Slot");
    cx.run_until_parked();
    item.update(cx, |item, _| {
        assert!(
            item.model
                .document()
                .expect("slot document after deletion")
                .active_subgraph_definition()
                .expect("active definition after slot deletion")
                .inputs
                .is_empty()
        );
        assert!(item.model.undo().expect("slot deletion undo"));
        assert!(!item.model.undo().expect("one slot deletion undo boundary"));
        assert_eq!(workflow_bytes(item), before);
    });

    let stale = item.read_with(cx, |item, _| {
        invocation(
            item,
            GraphContextTarget::Slot {
                direction: GraphSlotDirection::Input,
                slot: 0,
            },
        )
    });
    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-141"),
                GraphContextDispatchInput::None,
                stale,
                window,
                cx,
            ),
            GraphContextDispatchOutcome::ConfirmationPending
        );
        assert!(item.apply_graph_command(
            GraphCommand::RenameSubgraphSlot {
                direction: GraphSlotDirection::Input,
                slot: 0,
                name: "stale-input".to_owned(),
            },
            cx,
        ));
    });
    cx.simulate_prompt_answer("Remove Slot");
    cx.run_until_parked();
    item.update(cx, |item, _| {
        assert!(
            !item
                .model
                .document()
                .expect("slot document after stale confirmation")
                .active_subgraph_definition()
                .expect("slot definition after stale confirmation")
                .inputs
                .is_empty()
        );
        assert!(item.model.undo().expect("undo intervening slot mutation"));
        assert!(!item.model.undo().expect("stale confirmation adds no undo"));
        assert_eq!(workflow_bytes(item), before);
    });
}

#[gpui::test(seed = 16022)]
async fn slot_rename_uses_workspace_modal_and_one_undo_boundary(cx: &mut TestAppContext) {
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
    });
    init_context_test(cx);
    let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
    let (multi_workspace, cx) =
        cx.add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
    let workspace =
        multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
    let item = workspace.update_in(cx, |workspace, window, cx| {
        let weak_workspace = cx.weak_entity();
        let item = cx
            .new(|cx| GraphWorkspaceItem::new(subgraph_slot_fixture(), weak_workspace.clone(), cx));
        workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
        item
    });
    cx.run_until_parked();
    let before = item.read_with(cx, |item, _| workflow_bytes(item));
    let target = GraphContextTarget::Slot {
        direction: GraphSlotDirection::Input,
        slot: 0,
    };

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-140"),
                GraphContextDispatchInput::None,
                invocation(item, target.clone()),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
    });
    pump_context_menu_frames(cx);
    assert!(context_input_modal_is_open(&workspace, cx));
    cx.dispatch_action(menu::Cancel);
    cx.run_until_parked();
    assert_eq!(item.read_with(cx, |item, _| workflow_bytes(item)), before);

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-140"),
                GraphContextDispatchInput::None,
                invocation(item, target),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
    });
    pump_context_menu_frames(cx);
    let editor = item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("slot rename input")
            .editor
            .clone()
    });
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("renamed-input", window, cx)
    });
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    item.update(cx, |item, _| {
        let slot = item
            .model
            .document()
            .expect("renamed slot document")
            .active_subgraph_definition()
            .expect("renamed slot definition")
            .inputs
            .first()
            .expect("renamed input slot");
        assert_eq!(slot.name, "renamed-input");
        assert!(item.model.undo().expect("slot rename undo"));
        assert!(!item.model.undo().expect("one slot rename undo boundary"));
        assert_eq!(workflow_bytes(item), before);
    });
}

#[gpui::test(seed = 16019)]
fn group_and_reroute_surfaces_use_shared_pointer_and_keyboard_routes(cx: &mut TestAppContext) {
    init_context_test(cx);
    let window = cx.open_window(size(px(1280.0), px(900.0)), |_, cx| {
        GraphWorkspaceItem::new(context_fixture(), WeakEntity::new_invalid(), cx)
    });
    let item = window.root(cx).expect("group and reroute context fixture");
    let cx = VisualTestContext::from_window(window.into(), cx).into_mut();
    pump_context_menu_frames(cx);
    let (group_identifier, reroute_identifier) = item.read_with(cx, |item, _| {
        let graph = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .expect("group and reroute graph");
        (
            graph.groups.keys().next().expect("fixture group").clone(),
            graph
                .reroutes
                .keys()
                .next()
                .expect("fixture reroute")
                .clone(),
        )
    });
    for (selector, focus_key, target, use_origin) in [
        (
            Box::leak(format!("SIM-GROUP-{}", group_identifier.text()).into_boxed_str())
                as &'static str,
            format!("group:{}", group_identifier.text()),
            GraphContextTarget::Group(group_identifier),
            true,
        ),
        (
            Box::leak(format!("SIM-REROUTE-{}", reroute_identifier.text()).into_boxed_str())
                as &'static str,
            format!("reroute:{}", reroute_identifier.text()),
            GraphContextTarget::Reroute(reroute_identifier),
            false,
        ),
    ] {
        let bounds = cx
            .debug_bounds(selector)
            .unwrap_or_else(|| panic!("rendered context target {selector}"));
        let position = if use_origin {
            point(bounds.origin.x + px(4.0), bounds.origin.y + px(4.0))
        } else {
            bounds.center()
        };
        cx.simulate_mouse_move(position, None, Modifiers::default());
        cx.simulate_mouse_down(position, MouseButton::Right, Modifiers::default());
        item.read_with(cx, |item, _| {
            assert_eq!(
                item.context_menu_state
                    .as_ref()
                    .map(|state| &state.invocation.target),
                Some(&target)
            );
        });
        cx.dispatch_action(menu::Cancel);
        pump_context_menu_frames(cx);

        item.update_in(cx, |item, window, cx| {
            let focus = item.control_focus_handle(focus_key, cx);
            window.focus(&focus, cx);
        });
        cx.simulate_keystrokes("shift-f10");
        item.read_with(cx, |item, _| {
            assert_eq!(
                item.context_menu_state
                    .as_ref()
                    .map(|state| &state.invocation.target),
                Some(&target)
            );
        });
        cx.dispatch_action(menu::Cancel);
        pump_context_menu_frames(cx);
    }

    let before = item.read_with(cx, |item, _| workflow_bytes(item));
    let group_identifier = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.groups.keys().next())
            .cloned()
            .expect("destructive group target")
    });
    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-138"),
                GraphContextDispatchInput::None,
                invocation(item, GraphContextTarget::Group(group_identifier.clone()),),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::ConfirmationPending
        );
    });
    cx.simulate_prompt_answer("Cancel");
    cx.run_until_parked();
    assert_eq!(item.read_with(cx, |item, _| workflow_bytes(item)), before);
    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-138"),
                GraphContextDispatchInput::None,
                invocation(item, GraphContextTarget::Group(group_identifier.clone()),),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::ConfirmationPending
        );
    });
    cx.simulate_prompt_answer("Remove Group");
    cx.run_until_parked();
    item.update(cx, |item, _| {
        assert!(
            !item
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .expect("graph after group deletion")
                .groups
                .contains_key(&group_identifier)
        );
        assert!(item.model.undo().expect("group deletion undo"));
        assert!(!item.model.undo().expect("one group deletion undo boundary"));
        assert_eq!(workflow_bytes(item), before);
    });
}

#[gpui::test(seed = 16015)]
async fn context_input_uses_workspace_modal_with_exact_transactions(cx: &mut TestAppContext) {
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
    });
    init_context_test(cx);
    let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
    let fixture = context_fixture();
    let (multi_workspace, cx) =
        cx.add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
    let workspace =
        multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
    let (item, properties_panel) = workspace.update_in(cx, |workspace, window, cx| {
        let weak_workspace = cx.weak_entity();
        let item = cx.new(|cx| GraphWorkspaceItem::new(fixture, weak_workspace.clone(), cx));
        workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
        let properties_panel =
            cx.new(|cx| GraphPropertiesPanel::test_new(weak_workspace, window, cx));
        workspace.add_panel(properties_panel.clone(), window, cx);
        (item, properties_panel)
    });
    cx.run_until_parked();
    let original_bytes = item.read_with(cx, |item, _| workflow_bytes(item));
    let rename = |item: &GraphWorkspaceItem| {
        invocation(
            item,
            GraphContextTarget::Node(GraphIdentifier::from("source")),
        )
    };

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-121"),
                GraphContextDispatchInput::None,
                rename(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::Executed
        );
    });
    cx.run_until_parked();
    assert_eq!(
        properties_panel.read_with(cx, |panel, _| panel.target_for_test()),
        Some(GraphIdentifier::from("source"))
    );

    let group_identifier = item.read_with(cx, |item, _| {
        item.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.groups.keys().next())
            .cloned()
            .expect("context input group")
    });
    for (feature_id, target) in [
        ("COMFY-GRAPH-132", GraphContextTarget::Selection),
        ("COMFY-GRAPH-135", GraphContextTarget::Selection),
        ("COMFY-MENU-117", GraphContextTarget::Selection),
        (
            "COMFY-MENU-119",
            GraphContextTarget::Node(GraphIdentifier::from("source")),
        ),
        ("COMFY-GRAPH-138", GraphContextTarget::Selection),
        (
            "COMFY-MENU-135",
            GraphContextTarget::Group(group_identifier.clone()),
        ),
        (
            "COMFY-MENU-137",
            GraphContextTarget::Group(group_identifier),
        ),
    ] {
        item.update_in(cx, |item, window, cx| {
            assert_eq!(
                item.dispatch_context_action(
                    binding(feature_id),
                    GraphContextDispatchInput::None,
                    invocation(item, target.clone()),
                    window,
                    cx,
                ),
                GraphContextDispatchOutcome::InputPending,
                "{feature_id}"
            );
        });
        pump_context_menu_frames(cx);
        assert!(context_input_modal_is_open(&workspace, cx), "{feature_id}");
        cx.dispatch_action(menu::Cancel);
        cx.run_until_parked();
        assert!(!context_input_modal_is_open(&workspace, cx), "{feature_id}");
        item.read_with(cx, |item, _| {
            assert_eq!(workflow_bytes(item), original_bytes, "{feature_id}");
            assert!(item.context_input.is_none(), "{feature_id}");
        });
    }

    item.update_in(cx, |item, window, cx| {
        let outcome = item.dispatch_context_action(
            binding("COMFY-MENU-122"),
            GraphContextDispatchInput::None,
            rename(item),
            window,
            cx,
        );
        assert_eq!(outcome, GraphContextDispatchOutcome::InputPending);
    });
    pump_context_menu_frames(cx);
    assert!(context_input_modal_is_open(&workspace, cx));
    cx.update(|window, cx| {
        let modal = workspace
            .read(cx)
            .active_modal::<GraphContextInputModal>(cx)
            .expect("context input modal");
        assert!(modal.read(cx).focus_handle(cx).is_focused(window));
    });
    cx.dispatch_action(menu::Cancel);
    cx.run_until_parked();
    assert!(!context_input_modal_is_open(&workspace, cx));
    item.update(cx, |item, _| {
        assert_eq!(workflow_bytes(item), original_bytes);
        assert!(!item.model.undo().expect("cancel leaves no undo entry"));
    });

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-122"),
                GraphContextDispatchInput::None,
                rename(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
    });
    pump_context_menu_frames(cx);
    let editor = item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("rename input")
            .editor
            .clone()
    });
    editor.update_in(cx, |editor, window, cx| editor.set_text("", window, cx));
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    assert!(context_input_modal_is_open(&workspace, cx));
    assert_eq!(
        item.read_with(cx, |item, _| workflow_bytes(item)),
        original_bytes
    );
    cx.dispatch_action(menu::Cancel);
    cx.run_until_parked();

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-122"),
                GraphContextDispatchInput::None,
                rename(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
    });
    pump_context_menu_frames(cx);
    let editor = item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("rename input")
            .editor
            .clone()
    });
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("Renamed source", window, cx)
    });
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    assert!(!context_input_modal_is_open(&workspace, cx));
    item.update(cx, |item, _| {
        let source = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.nodes.get(&GraphIdentifier::from("source")))
            .expect("renamed source node");
        assert_eq!(source.title, "Renamed source");
        assert!(item.model.undo().expect("rename has one undo entry"));
        assert!(
            !item
                .model
                .undo()
                .expect("rename has exactly one undo entry")
        );
        assert_eq!(workflow_bytes(item), original_bytes);
    });

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-122"),
                GraphContextDispatchInput::None,
                rename(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
        assert!(item.apply_graph_command(
            GraphCommand::SetNodeProperties {
                identifier: GraphIdentifier::from("source"),
                properties: serde_json::Map::from_iter([(
                    "stale".to_owned(),
                    serde_json::Value::Bool(true),
                )]),
            },
            cx,
        ));
    });
    pump_context_menu_frames(cx);
    let editor = item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("stale rename input")
            .editor
            .clone()
    });
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("Must not apply", window, cx)
    });
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    assert!(!context_input_modal_is_open(&workspace, cx));
    item.update(cx, |item, _| {
        let source = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.nodes.get(&GraphIdentifier::from("source")))
            .expect("source node after stale rejection");
        assert_ne!(source.title, "Must not apply");
        assert!(item.model.undo().expect("undo intervening stale mutation"));
        assert!(!item.model.undo().expect("stale input adds no undo entry"));
        assert_eq!(workflow_bytes(item), original_bytes);
    });

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-120"),
                GraphContextDispatchInput::NodeProperty {
                    key: "enabled".to_owned(),
                    value: Some(serde_json::Value::Bool(true)),
                },
                rename(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::Executed
        );
        let properties = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.nodes.get(&GraphIdentifier::from("source")))
            .and_then(|node| node.source_fields.get("properties"))
            .and_then(serde_json::Value::as_object)
            .expect("typed source properties");
        assert_eq!(
            properties.get("enabled"),
            Some(&serde_json::Value::Bool(true))
        );
        assert!(
            item.model
                .undo()
                .expect("boolean property has one undo entry")
        );
        assert!(
            !item
                .model
                .undo()
                .expect("boolean property has exactly one undo entry")
        );
        assert_eq!(workflow_bytes(item), original_bytes);
    });

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-MENU-120"),
                GraphContextDispatchInput::NodeProperty {
                    key: "strength".to_owned(),
                    value: None,
                },
                rename(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
    });
    pump_context_menu_frames(cx);
    let editor = item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("numeric property input")
            .editor
            .clone()
    });
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("not-a-number", window, cx)
    });
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    assert!(context_input_modal_is_open(&workspace, cx));
    assert_eq!(
        item.read_with(cx, |item, _| workflow_bytes(item)),
        original_bytes
    );
    editor.update_in(cx, |editor, window, cx| editor.set_text("0.75", window, cx));
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    assert!(!context_input_modal_is_open(&workspace, cx));
    item.update(cx, |item, _| {
        let properties = item
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.nodes.get(&GraphIdentifier::from("source")))
            .and_then(|node| node.source_fields.get("properties"))
            .and_then(serde_json::Value::as_object)
            .expect("typed source properties");
        assert_eq!(properties.get("strength"), Some(&json!(0.75)));
        assert!(
            item.model
                .undo()
                .expect("numeric property has one undo entry")
        );
        assert!(
            !item
                .model
                .undo()
                .expect("invalid input added no undo entry")
        );
        assert_eq!(workflow_bytes(item), original_bytes);
    });
}

#[gpui::test(seed = 16016)]
async fn subgraph_publication_is_cancellable_transactional_and_observable(cx: &mut TestAppContext) {
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
    });
    init_context_test(cx);
    let (directory, assets) = native_asset_service().expect("open canonical native asset service");
    let authorization =
        authorize_native_subgraph_library("profile").expect("authorize native blueprint library");
    let malformed_identity =
        AssetIdentity::new("profile", AssetNamespace::Plugin, "subgraphs/Broken.json")
            .expect("canonical malformed fixture identity");
    assets
        .lock()
        .expect("asset service lock")
        .write_exact(
            &malformed_identity,
            b"{not-json",
            BTreeSet::from([SUBGRAPH_BLUEPRINT_ASSET_TAG.to_owned()]),
            AssetCollisionPolicy::Reject,
            &authorization,
            &sim_types::CancellationToken::default(),
        )
        .expect("publish malformed isolation fixture");
    cx.update(|cx| {
        crate::register_native_asset_services(assets.clone(), cx)
            .expect("register canonical native asset services");
    });

    let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
    let fixture = publication_fixture("Published from native GPUI");
    let publication_document = fixture.document().expect("publication document");
    let expected_bytes = publication_document
        .export_selected_subgraph_blueprint("Native Blend")
        .expect("canonical expected blueprint")
        .workflow_bytes;
    let replacement_fixture = publication_fixture("Independent replacement metadata");
    let replacement_expected_bytes = replacement_fixture
        .document()
        .expect("replacement publication document")
        .export_selected_subgraph_blueprint("Native Blend")
        .expect("canonical replacement blueprint")
        .workflow_bytes;
    assert_ne!(replacement_expected_bytes, expected_bytes);
    let (multi_workspace, cx) =
        cx.add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
    let workspace =
        multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
    let (item, observer_item) = workspace.update_in(cx, |workspace, window, cx| {
        let weak_workspace = cx.weak_entity();
        let item = cx.new(|cx| GraphWorkspaceItem::new(fixture, weak_workspace.clone(), cx));
        let observer_item =
            cx.new(|cx| GraphWorkspaceItem::new(replacement_fixture, weak_workspace, cx));
        workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
        (item, observer_item)
    });
    cx.run_until_parked();
    assert!(item.read_with(cx, |item, _| {
        item.model
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("isolated 1 invalid subgraph blueprint"))
    }));
    let original_bytes = item.read_with(cx, |item, _| workflow_bytes(item));
    let original_authority = item.read_with(cx, |item, _| item.model.save_coordinator.authority());
    let publish = |item: &GraphWorkspaceItem| invocation(item, GraphContextTarget::Selection);

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-GRAPH-136"),
                GraphContextDispatchInput::None,
                publish(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
    });
    pump_context_menu_frames(cx);
    let unavailable_editor = item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("unavailable-service publication input")
            .editor
            .clone()
    });
    unavailable_editor.update_in(cx, |editor, window, cx| {
        editor.set_text("Unavailable", window, cx)
    });
    cx.update(|_, cx| crate::remove_native_asset_services_for_test(cx));
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    assert!(context_input_modal_is_open(&workspace, cx));
    assert!(item.read_with(cx, |item, _| {
        item.model
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("canonical native asset service is unavailable"))
    }));
    cx.dispatch_action(menu::Cancel);
    cx.update(|_, cx| {
        crate::register_native_asset_services(assets.clone(), cx)
            .expect("restore canonical native asset services")
    });
    cx.run_until_parked();

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-GRAPH-136"),
                GraphContextDispatchInput::None,
                publish(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
    });
    pump_context_menu_frames(cx);
    let cancelled_editor = item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("cancelled publication input")
            .editor
            .clone()
    });
    cancelled_editor.update_in(cx, |editor, window, cx| {
        editor.set_text("Cancelled", window, cx)
    });
    item.update_in(cx, |item, window, cx| {
        item.confirm_context_input(window, cx);
        assert!(item.subgraph_publish_task.is_some());
        assert!(item.subgraph_publish_cancellation.is_some());
        item.cancel_gesture(cx);
    });
    workspace.update_in(cx, |workspace, window, cx| {
        assert!(workspace.hide_modal(window, cx));
    });
    cx.run_until_parked();
    assert!(
        native_catalog(cx)
            .descriptor("SubgraphBlueprint.Cancelled")
            .is_none()
    );
    let cancelled_identity = AssetIdentity::new(
        "profile",
        AssetNamespace::Plugin,
        "subgraphs/Cancelled.json",
    )
    .expect("cancelled asset identity");
    assert!(
        assets
            .lock()
            .expect("asset service lock")
            .record(&cancelled_identity)
            .is_none()
    );
    assert!(item.read_with(cx, |item, _| {
        item.model
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("cancelled"))
    }));

    item.update_in(cx, |item, window, cx| {
        assert_eq!(
            item.dispatch_context_action(
                binding("COMFY-GRAPH-136"),
                GraphContextDispatchInput::None,
                publish(item),
                window,
                cx,
            ),
            GraphContextDispatchOutcome::InputPending
        );
    });
    pump_context_menu_frames(cx);
    let editor = item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("publication input")
            .editor
            .clone()
    });
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("Native Blend", window, cx)
    });
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    let catalog = native_catalog(cx);
    let published = catalog
        .descriptor("SubgraphBlueprint.Native Blend")
        .expect("published descriptor is immediately observable");
    assert_eq!(published.blueprint.workflow_bytes, expected_bytes);
    assert_eq!(catalog.diagnostics().len(), 1);
    assert_eq!(catalog.diagnostics()[0].identity, malformed_identity);
    assert!(item.read_with(cx, |item, _| {
        item.model
            .announcement
            .as_deref()
            .is_some_and(|announcement| announcement.contains("Published blueprint Native Blend"))
    }));
    assert!(observer_item.read_with(cx, |item, _| {
        item.model
            .announcement
            .as_deref()
            .is_some_and(|announcement| {
                announcement.contains("Native node library updated")
                    && announcement.contains("Native Blend")
            })
    }));

    observer_item.update_in(cx, |item, window, cx| {
        item.begin_shell_publish_subgraph(window, cx);
    });
    pump_context_menu_frames(cx);
    let overwrite_editor = observer_item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("shell publication input")
            .editor
            .clone()
    });
    overwrite_editor.update_in(cx, |editor, window, cx| {
        editor.set_text("Native Blend", window, cx)
    });
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    assert!(cx.has_pending_prompt());
    cx.simulate_prompt_answer("Cancel");
    cx.run_until_parked();
    let before_replace_sha = native_catalog(cx)
        .descriptor("SubgraphBlueprint.Native Blend")
        .map(|entry| entry.asset.sha256.clone())
        .expect("catalog entry before explicit replacement");

    observer_item.update_in(cx, |item, window, cx| {
        item.begin_shell_publish_subgraph(window, cx);
    });
    pump_context_menu_frames(cx);
    let overwrite_editor = observer_item.read_with(cx, |item, _| {
        item.context_input
            .as_ref()
            .expect("replacement publication input")
            .editor
            .clone()
    });
    overwrite_editor.update_in(cx, |editor, window, cx| {
        editor.set_text("Native Blend", window, cx)
    });
    cx.dispatch_action(menu::Confirm);
    cx.run_until_parked();
    assert!(cx.has_pending_prompt());
    cx.simulate_prompt_answer("Replace");
    cx.run_until_parked();
    let after_replace = native_catalog(cx)
        .descriptor("SubgraphBlueprint.Native Blend")
        .cloned()
        .expect("catalog entry after explicit replacement");
    assert_ne!(after_replace.asset.sha256, before_replace_sha);
    assert_eq!(
        after_replace.blueprint.workflow_bytes,
        replacement_expected_bytes
    );
    assert_eq!(
        after_replace.description,
        "Independent replacement metadata"
    );

    item.update(cx, |item, _| {
        assert_eq!(workflow_bytes(item), original_bytes);
        assert_eq!(item.model.save_coordinator.authority(), original_authority);
        assert!(
            !item
                .model
                .undo()
                .expect("publication adds no graph undo entry")
        );
    });
    let restarted = open_native_profile_asset_service("profile", directory.path(), &[])
        .expect("reopen persisted canonical asset index");
    let identity = AssetIdentity::new(
        "profile",
        AssetNamespace::Plugin,
        "subgraphs/Native Blend.json",
    )
    .expect("published asset identity");
    let restarted_record = restarted
        .lock()
        .expect("restarted asset service lock")
        .record(&identity)
        .expect("replacement survives canonical asset-service restart");
    assert_eq!(restarted_record.sha256, after_replace.asset.sha256);
}

#[gpui::test(seed = 16017)]
async fn committed_subgraph_projection_survives_workspace_item_drop(cx: &mut TestAppContext) {
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
    });
    init_context_test(cx);
    let (_directory, assets) = native_asset_service().expect("open canonical native asset service");
    cx.update(|cx| {
        crate::register_native_asset_services(assets.clone(), cx)
            .expect("register canonical native asset services");
    });

    let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
    let (multi_workspace, cx) =
        cx.add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
    let workspace =
        multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
    let item = workspace.update_in(cx, |workspace, window, cx| {
        let weak_workspace = cx.weak_entity();
        let item = cx.new(|cx| {
            GraphWorkspaceItem::new(
                publication_fixture("Projection survives item drop"),
                weak_workspace,
                cx,
            )
        });
        workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
        item
    });
    let (projection_sender, projection_receiver) = futures::channel::oneshot::channel();
    let cancellation = item.update_in(cx, |item, window, cx| {
        item.subgraph_publish_projection_barrier = Some(projection_receiver);
        assert!(item.begin_subgraph_publish_for_test("Drop Safe".to_owned(), window, cx,));
        item.subgraph_publish_cancellation
            .clone()
            .expect("publication cancellation token")
    });
    cx.run_until_parked();

    let identity = AssetIdentity::new(
        "profile",
        AssetNamespace::Plugin,
        "subgraphs/Drop Safe.json",
    )
    .expect("drop-safe asset identity");
    assert!(
        assets
            .lock()
            .expect("asset service lock")
            .record(&identity)
            .is_some(),
        "canonical commit must finish before the projection barrier"
    );
    assert!(
        native_catalog(cx)
            .descriptor("SubgraphBlueprint.Drop Safe")
            .is_none(),
        "projection must still be held behind the deterministic barrier"
    );

    item.update(cx, |item, _| {
        item.cancel_subgraph_publication_for_drop_for_test();
    });
    assert!(
        cancellation.is_cancelled(),
        "Drop must request cancellation"
    );
    let item_id = item.entity_id();
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.active_pane().update(cx, |pane, cx| {
            pane.remove_item(item_id, false, false, window, cx)
        });
    });
    drop(item);
    cx.run_until_parked();
    assert!(projection_sender.send(()).is_ok());
    cx.run_until_parked();

    assert!(
        native_catalog(cx)
            .descriptor("SubgraphBlueprint.Drop Safe")
            .is_some(),
        "the detached app-level coordinator must project a commit that won the cancellation race"
    );
}
