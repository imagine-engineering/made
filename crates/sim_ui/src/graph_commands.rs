use crate::{GraphWorkspaceError, GraphWorkspaceModel};
use serde_json::{Map, Value};
use sim_runtime::{
    CatalogGraphAction, GraphClipboard, GraphCommand, GraphDocument, GraphError, GraphGroup,
    GraphIdentifier, GraphLink, GraphNode, GraphPoint, GraphPort, GraphRect, GraphSelection,
    GraphSize, GraphViewport, GraphWidget, GroupToggle, LayoutOperation, NodeToggle,
};
use std::collections::BTreeSet;
use thiserror::Error;

pub const GRAPH_CLIPBOARD_MEDIA_TYPE: &str = "application/x-zed-sim-graph+json;version=1";

#[derive(Clone, Debug, PartialEq)]
pub enum GraphActionInput {
    None,
    Paste {
        bytes: Vec<u8>,
        offset: GraphPoint,
        connect_from: Option<(GraphIdentifier, usize)>,
    },
    Resize {
        identifier: GraphIdentifier,
        size: GraphSize,
    },
    SubgraphName(String),
    Group {
        title: String,
        padding: f32,
    },
    GroupIdentifier(GraphIdentifier),
    FitGroup {
        identifier: GraphIdentifier,
        padding: f32,
    },
    SubgraphInstance(GraphIdentifier),
    SubgraphDefinition(GraphIdentifier),
    WidgetPromotion {
        node: GraphIdentifier,
        widget: String,
        promoted: bool,
    },
    RenameNode {
        identifier: GraphIdentifier,
        title: String,
    },
    NodeColor {
        identifier: GraphIdentifier,
        color: Option<String>,
    },
    NodeIdentifiers(BTreeSet<GraphIdentifier>),
    Layout {
        operation: LayoutOperation,
        spacing: f32,
    },
    RerouteMove {
        identifier: GraphIdentifier,
        position: GraphPoint,
    },
    RerouteParent {
        identifier: GraphIdentifier,
        parent: Option<GraphIdentifier>,
    },
    RerouteIdentifier(GraphIdentifier),
    RemoveSubgraphDefinition {
        definition: GraphIdentifier,
        remove_instances: bool,
    },
    SubgraphWidgetExposure {
        definition: GraphIdentifier,
        internal_node: GraphIdentifier,
        widget: String,
        exposed: bool,
    },
    ReconcileNode {
        identifier: GraphIdentifier,
        inputs: Vec<GraphPort>,
        outputs: Vec<GraphPort>,
        widgets: Vec<GraphWidget>,
        confirm_discard: bool,
    },
    SubgraphDescription {
        definition: GraphIdentifier,
        description: String,
    },
    SubgraphSearchAliases {
        definition: GraphIdentifier,
        aliases: Vec<String>,
    },
    FitAvailable(GraphSize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphModelAction {
    RenameNode,
    SetNodeColor,
    ToggleNodeDisable,
    LayoutSelection,
    Ungroup,
    MoveReroute,
    ReparentReroute,
    RemoveReroute,
    RemoveSubgraphDefinition,
    SetSubgraphWidgetExposure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphActionEffect {
    None,
    ClipboardText(String),
}

pub struct GraphCommandModel;

impl GraphCommandModel {
    pub fn execute_model_action(
        model: &mut GraphWorkspaceModel,
        action: GraphModelAction,
        input: GraphActionInput,
    ) -> Result<GraphActionEffect, GraphActionError> {
        let command = match (action, input) {
            (GraphModelAction::RenameNode, GraphActionInput::RenameNode { identifier, title }) => {
                GraphCommand::RenameNode { identifier, title }
            }
            (GraphModelAction::SetNodeColor, GraphActionInput::NodeColor { identifier, color }) => {
                GraphCommand::SetNodeColor { identifier, color }
            }
            (
                GraphModelAction::ToggleNodeDisable,
                GraphActionInput::NodeIdentifiers(identifiers),
            ) => GraphCommand::ToggleNodes {
                identifiers,
                toggle: NodeToggle::Disable,
            },
            (
                GraphModelAction::LayoutSelection,
                GraphActionInput::Layout { operation, spacing },
            ) => GraphCommand::LayoutSelection { operation, spacing },
            (GraphModelAction::Ungroup, GraphActionInput::GroupIdentifier(identifier)) => {
                GraphCommand::Ungroup { identifier }
            }
            (
                GraphModelAction::MoveReroute,
                GraphActionInput::RerouteMove {
                    identifier,
                    position,
                },
            ) => GraphCommand::MoveReroute {
                identifier,
                position,
            },
            (
                GraphModelAction::ReparentReroute,
                GraphActionInput::RerouteParent { identifier, parent },
            ) => GraphCommand::ReparentReroute { identifier, parent },
            (GraphModelAction::RemoveReroute, GraphActionInput::RerouteIdentifier(identifier)) => {
                GraphCommand::RemoveReroute { identifier }
            }
            (
                GraphModelAction::RemoveSubgraphDefinition,
                GraphActionInput::RemoveSubgraphDefinition {
                    definition,
                    remove_instances,
                },
            ) => GraphCommand::RemoveSubgraphDefinition {
                definition_identifier: definition,
                remove_instances,
            },
            (
                GraphModelAction::SetSubgraphWidgetExposure,
                GraphActionInput::SubgraphWidgetExposure {
                    definition,
                    internal_node,
                    widget,
                    exposed,
                },
            ) => GraphCommand::SetSubgraphWidgetExposure {
                definition_identifier: definition,
                internal_node,
                widget,
                exposed,
            },
            (_, GraphActionInput::None) => {
                return Err(GraphActionError::ModelInputRequired(action));
            }
            (_, input) => {
                return Err(GraphActionError::ModelUnexpectedInput {
                    action,
                    input: format!("{input:?}"),
                });
            }
        };
        model.apply(command)?;
        Ok(GraphActionEffect::None)
    }

    pub fn execute(
        model: &mut GraphWorkspaceModel,
        action: CatalogGraphAction,
        input: GraphActionInput,
    ) -> Result<GraphActionEffect, GraphActionError> {
        match action {
            CatalogGraphAction::CopySelected => {
                require_none(action, input)?;
                let document = model.document().ok_or(GraphWorkspaceError::ReadOnly)?;
                let clipboard = GraphClipboard::copy(document)?;
                let bytes = clipboard.encode()?;
                let json = String::from_utf8(bytes)
                    .map_err(|error| GraphActionError::Clipboard(error.to_string()))?;
                Ok(GraphActionEffect::ClipboardText(format!(
                    "{GRAPH_CLIPBOARD_MEDIA_TYPE}\n{json}"
                )))
            }
            CatalogGraphAction::DeleteSelectedItems => {
                require_none(action, input)?;
                model.apply(GraphCommand::RemoveSelection)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::FitView => {
                let available = match input {
                    GraphActionInput::None => GraphSize {
                        width: 1_200.0,
                        height: 800.0,
                    },
                    GraphActionInput::FitAvailable(available) => available,
                    other => return Err(unexpected_input(action, other)),
                };
                let bounds = graph_bounds(model).ok_or(GraphActionError::EmptyGraph)?;
                model.apply(GraphCommand::FitViewport {
                    bounds,
                    available,
                    padding: 48.0,
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::Lock => {
                require_none(action, input)?;
                set_viewport_lock(model, true)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::Unlock => {
                require_none(action, input)?;
                set_viewport_lock(model, false)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleLock => {
                require_none(action, input)?;
                let locked = model
                    .document()
                    .and_then(|document| document.active_graph().ok())
                    .ok_or(GraphWorkspaceError::ReadOnly)?
                    .viewport
                    .locked;
                set_viewport_lock(model, !locked)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::MoveSelectedDown => {
                require_none(action, input)?;
                move_selection(model, 0.0, 10.0)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::MoveSelectedLeft => {
                require_none(action, input)?;
                move_selection(model, -10.0, 0.0)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::MoveSelectedRight => {
                require_none(action, input)?;
                move_selection(model, 10.0, 0.0)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::MoveSelectedUp => {
                require_none(action, input)?;
                move_selection(model, 0.0, -10.0)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::PasteFromClipboard
            | CatalogGraphAction::PasteFromClipboardWithConnect => {
                let GraphActionInput::Paste {
                    bytes,
                    offset,
                    connect_from,
                } = input
                else {
                    return Err(GraphActionError::InputRequired(action));
                };
                let clipboard = decode_clipboard(&bytes)?;
                let document = model.document().ok_or(GraphWorkspaceError::ReadOnly)?;
                let paste = clipboard.paste_command(document, offset)?;
                let command = if action == CatalogGraphAction::PasteFromClipboardWithConnect {
                    let (origin_node, origin_slot) =
                        connect_from.ok_or(GraphActionError::InputRequired(action))?;
                    let mut preview = model
                        .engine()
                        .cloned()
                        .ok_or(GraphWorkspaceError::ReadOnly)?;
                    preview.apply(paste.clone())?;
                    let target_node = preview
                        .document
                        .active_graph()?
                        .selection
                        .nodes
                        .iter()
                        .next()
                        .cloned()
                        .ok_or(GraphActionError::Clipboard(
                            "clipboard contains no node to connect".to_owned(),
                        ))?;
                    let mut identifier_source = preview.document.clone();
                    let link_identifier = identifier_source.allocate_identifier();
                    GraphCommand::Batch {
                        commands: vec![
                            paste,
                            GraphCommand::Connect {
                                link: GraphLink {
                                    identifier: link_identifier,
                                    origin_node,
                                    origin_slot,
                                    target_node,
                                    target_slot: 0,
                                    type_name: String::new(),
                                    parent_reroute: None,
                                    source: Value::Null,
                                },
                                replace_existing: false,
                            },
                        ],
                    }
                } else {
                    paste
                };
                model.apply(command)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ResetView => {
                require_none(action, input)?;
                model.apply(GraphCommand::SetViewport {
                    viewport: GraphViewport::default(),
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::Resize => {
                let (identifiers, explicit_size) = match input {
                    GraphActionInput::None => (selection(model)?.nodes, None),
                    GraphActionInput::NodeIdentifiers(identifiers) => (identifiers, None),
                    GraphActionInput::Resize { identifier, size } => {
                        (BTreeSet::from([identifier]), Some(size))
                    }
                    other => return Err(unexpected_input(action, other)),
                };
                if identifiers.is_empty() {
                    return Err(GraphActionError::EmptySelection);
                }
                let graph = model
                    .document()
                    .and_then(|document| document.active_graph().ok())
                    .ok_or(GraphWorkspaceError::ReadOnly)?;
                let commands = identifiers
                    .iter()
                    .map(|identifier| {
                        let node = graph.nodes.get(identifier).ok_or_else(|| {
                            GraphActionError::Graph(GraphError::UnknownNode(identifier.clone()))
                        })?;
                        Ok(GraphCommand::ResizeNode {
                            identifier: identifier.clone(),
                            size: explicit_size.unwrap_or_else(|| optimal_node_size(node)),
                        })
                    })
                    .collect::<Result<Vec<_>, GraphActionError>>()?;
                model.apply(GraphCommand::Batch { commands })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::SelectAll => {
                require_none(action, input)?;
                model.apply(GraphCommand::SelectAll)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleLinkVisibility => {
                require_none(action, input)?;
                update_viewport(model, |viewport| {
                    viewport.links_visible = !viewport.links_visible;
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleMinimap => {
                require_none(action, input)?;
                update_viewport(model, |viewport| {
                    viewport.minimap_visible = !viewport.minimap_visible;
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleSelectedBypass => {
                require_none(action, input)?;
                toggle_selected_nodes(model, NodeToggle::Bypass)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleSelectedCollapse => {
                require_none(action, input)?;
                let selection = selection(model)?;
                let mut commands = Vec::new();
                if !selection.nodes.is_empty() {
                    commands.push(GraphCommand::ToggleNodes {
                        identifiers: selection.nodes,
                        toggle: NodeToggle::Collapse,
                    });
                }
                if !selection.groups.is_empty() {
                    commands.push(GraphCommand::ToggleGroups {
                        identifiers: selection.groups,
                        toggle: GroupToggle::Collapse,
                    });
                }
                apply_nonempty_batch(model, commands)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleSelectedMute => {
                require_none(action, input)?;
                toggle_selected_nodes(model, NodeToggle::Mute)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleSelectedPin | CatalogGraphAction::ToggleSelectedItemsPin => {
                require_none(action, input)?;
                let selection = selection(model)?;
                let mut commands = Vec::new();
                if !selection.nodes.is_empty() {
                    commands.push(GraphCommand::ToggleNodes {
                        identifiers: selection.nodes,
                        toggle: NodeToggle::Pin,
                    });
                }
                if !selection.groups.is_empty() {
                    commands.push(GraphCommand::ToggleGroups {
                        identifiers: selection.groups,
                        toggle: GroupToggle::Pin,
                    });
                }
                apply_nonempty_batch(model, commands)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ZoomIn => {
                require_none(action, input)?;
                model.apply(GraphCommand::ZoomViewport {
                    factor: 1.1,
                    anchor: GraphPoint::ZERO,
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ZoomOut => {
                require_none(action, input)?;
                model.apply(GraphCommand::ZoomViewport {
                    factor: 1.0 / 1.1,
                    anchor: GraphPoint::ZERO,
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ConvertToSubgraph => {
                let GraphActionInput::SubgraphName(name) = input else {
                    return Err(GraphActionError::InputRequired(action));
                };
                let mut identifier_source = model
                    .document()
                    .cloned()
                    .ok_or(GraphWorkspaceError::ReadOnly)?;
                let definition_identifier = identifier_source.allocate_subgraph_identifier();
                let instance_identifier = identifier_source.allocate_identifier();
                model.apply(GraphCommand::ConvertSelectionToSubgraph {
                    definition_identifier,
                    instance_identifier,
                    name,
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::EditSubgraphWidgets => {
                match input {
                    GraphActionInput::SubgraphWidgetExposure {
                        definition,
                        internal_node,
                        widget,
                        exposed,
                    } => model.apply(GraphCommand::SetSubgraphWidgetExposure {
                        definition_identifier: definition,
                        internal_node,
                        widget,
                        exposed,
                    })?,
                    GraphActionInput::WidgetPromotion {
                        node,
                        widget,
                        promoted,
                    } => model.apply(GraphCommand::ConvertWidgetToInput {
                        node,
                        widget,
                        converted: promoted,
                    })?,
                    other => return Err(unexpected_input(action, other)),
                }
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleWidgetPromotion => {
                let GraphActionInput::WidgetPromotion {
                    node,
                    widget,
                    promoted,
                } = input
                else {
                    return Err(GraphActionError::InputRequired(action));
                };
                model.apply(GraphCommand::ConvertWidgetToInput {
                    node,
                    widget,
                    converted: promoted,
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ExitSubgraph => {
                require_none(action, input)?;
                model.apply(GraphCommand::ExitSubgraph)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::FitGroupToContents => {
                let (identifier, padding) = match input {
                    GraphActionInput::FitGroup {
                        identifier,
                        padding,
                    } => (identifier, padding),
                    GraphActionInput::GroupIdentifier(_) | GraphActionInput::None => {
                        return Err(GraphActionError::RequiresSettingsStore(action));
                    }
                    other => return Err(unexpected_input(action, other)),
                };
                model.apply(GraphCommand::FitGroup {
                    identifier,
                    padding,
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::GroupSelectedNodes => {
                let (title, padding) = match input {
                    GraphActionInput::None => {
                        return Err(GraphActionError::RequiresSettingsStore(action));
                    }
                    GraphActionInput::Group { title, padding } => (title, padding),
                    other => return Err(unexpected_input(action, other)),
                };
                let document = model.document().ok_or(GraphWorkspaceError::ReadOnly)?;
                let graph = document.active_graph()?;
                if graph.selection.nodes.is_empty() {
                    return Err(GraphActionError::EmptySelection);
                }
                let node_ids = graph.selection.nodes.clone();
                let bounds = bounds_for_identifiers(graph.nodes.values(), &node_ids)
                    .ok_or(GraphActionError::EmptySelection)?;
                let identifier = document.next_available_identifier();
                model.apply(GraphCommand::CreateGroup {
                    group: GraphGroup {
                        identifier,
                        title,
                        bounds: expanded_bounds(bounds, padding)?,
                        node_ids,
                        collapsed: false,
                        pinned: false,
                        color: None,
                        source_fields: Map::new(),
                    },
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::UnpackSubgraph => {
                let instance_identifier = match input {
                    GraphActionInput::SubgraphInstance(identifier) => identifier,
                    GraphActionInput::None => model
                        .selected_node_identifiers()
                        .into_iter()
                        .next()
                        .ok_or(GraphActionError::InputRequired(action))?,
                    other => return Err(unexpected_input(action, other)),
                };
                model.apply(GraphCommand::UnpackSubgraph {
                    instance_identifier,
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::PublishSubgraph => {
                Err(GraphActionError::RequiresAssetService(action))
            }
            CatalogGraphAction::RefreshNodeDefinitions => {
                let GraphActionInput::ReconcileNode {
                    identifier,
                    inputs,
                    outputs,
                    widgets,
                    confirm_discard,
                } = input
                else {
                    return Err(GraphActionError::InputRequired(action));
                };
                model.apply(GraphCommand::ReconcileNode {
                    identifier,
                    inputs,
                    outputs,
                    widgets,
                    confirm_discard,
                })?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::SetSubgraphDescription => {
                let GraphActionInput::SubgraphDescription {
                    definition,
                    description,
                } = input
                else {
                    return Err(GraphActionError::InputRequired(action));
                };
                set_subgraph_metadata(model, definition, Some(description), None)?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::SetSubgraphSearchAliases => {
                let GraphActionInput::SubgraphSearchAliases {
                    definition,
                    aliases,
                } = input
                else {
                    return Err(GraphActionError::InputRequired(action));
                };
                set_subgraph_metadata(model, definition, None, Some(aliases))?;
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleCanvasInfo => {
                require_none(action, input)?;
                model.canvas_info_visible = !model.canvas_info_visible;
                model.announcement = Some(if model.canvas_info_visible {
                    "Canvas information shown".to_owned()
                } else {
                    "Canvas information hidden".to_owned()
                });
                Ok(GraphActionEffect::None)
            }
            CatalogGraphAction::ToggleVueNodes => {
                require_none(action, input)?;
                Err(GraphActionError::RequiresSettingsStore(action))
            }
        }
    }
}

pub fn decode_clipboard(bytes: &[u8]) -> Result<GraphClipboard, GraphActionError> {
    let marker = format!("{GRAPH_CLIPBOARD_MEDIA_TYPE}\n");
    if let Some(payload) = bytes.strip_prefix(marker.as_bytes()) {
        return GraphClipboard::decode(payload).map_err(GraphActionError::from);
    }
    match GraphClipboard::decode(bytes) {
        Ok(clipboard) => Ok(clipboard),
        Err(clipboard_error) => {
            let mut document = GraphDocument::from_workflow_bytes(bytes).map_err(|workflow_error| {
                GraphActionError::Clipboard(format!(
                    "payload is neither native graph clipboard JSON ({clipboard_error}) nor workflow/API-prompt JSON ({workflow_error})"
                ))
            })?;
            let graph = document.active_graph_mut()?;
            graph.selection.nodes = graph.nodes.keys().cloned().collect();
            graph.selection.groups = graph.groups.keys().cloned().collect();
            graph.selection.reroutes = graph.reroutes.keys().cloned().collect();
            GraphClipboard::copy(&document).map_err(GraphActionError::from)
        }
    }
}

fn require_none(
    action: CatalogGraphAction,
    input: GraphActionInput,
) -> Result<(), GraphActionError> {
    if input == GraphActionInput::None {
        Ok(())
    } else {
        Err(unexpected_input(action, input))
    }
}

fn unexpected_input(action: CatalogGraphAction, input: GraphActionInput) -> GraphActionError {
    GraphActionError::UnexpectedInput {
        action,
        input: format!("{input:?}"),
    }
}

fn selection(model: &GraphWorkspaceModel) -> Result<GraphSelection, GraphActionError> {
    model
        .selection()
        .cloned()
        .ok_or_else(|| GraphWorkspaceError::ReadOnly.into())
}

fn move_selection(model: &mut GraphWorkspaceModel, x: f32, y: f32) -> Result<(), GraphActionError> {
    model.apply(GraphCommand::MoveSelection {
        delta: GraphPoint { x, y },
        snap: Some(10.0),
    })?;
    Ok(())
}

fn toggle_selected_nodes(
    model: &mut GraphWorkspaceModel,
    toggle: NodeToggle,
) -> Result<(), GraphActionError> {
    let identifiers = selection(model)?.nodes;
    if identifiers.is_empty() {
        return Err(GraphActionError::EmptySelection);
    }
    model.apply(GraphCommand::ToggleNodes {
        identifiers,
        toggle,
    })?;
    Ok(())
}

fn apply_nonempty_batch(
    model: &mut GraphWorkspaceModel,
    commands: Vec<GraphCommand>,
) -> Result<(), GraphActionError> {
    if commands.is_empty() {
        return Err(GraphActionError::EmptySelection);
    }
    model.apply(GraphCommand::Batch { commands })?;
    Ok(())
}

fn set_viewport_lock(
    model: &mut GraphWorkspaceModel,
    locked: bool,
) -> Result<(), GraphActionError> {
    update_viewport(model, |viewport| viewport.locked = locked)
}

fn update_viewport(
    model: &mut GraphWorkspaceModel,
    update: impl FnOnce(&mut GraphViewport),
) -> Result<(), GraphActionError> {
    let mut viewport = model
        .document()
        .and_then(|document| document.active_graph().ok())
        .ok_or(GraphWorkspaceError::ReadOnly)?
        .viewport
        .clone();
    update(&mut viewport);
    model.apply(GraphCommand::SetViewport { viewport })?;
    Ok(())
}

fn graph_bounds(model: &GraphWorkspaceModel) -> Option<GraphRect> {
    let graph = model.document()?.active_graph().ok()?;
    let selection = &graph.selection;
    let mut identifiers = selection.nodes.clone();
    for group in selection
        .groups
        .iter()
        .filter_map(|identifier| graph.groups.get(identifier))
    {
        identifiers.extend(group.node_ids.iter().cloned());
    }
    for link in selection
        .links
        .iter()
        .filter_map(|identifier| graph.links.get(identifier))
    {
        identifiers.insert(link.origin_node.clone());
        identifiers.insert(link.target_node.clone());
    }
    if identifiers.is_empty() && selection.groups.is_empty() && selection.reroutes.is_empty() {
        identifiers.extend(graph.nodes.keys().cloned());
    }
    let mut bounds = bounds_for_identifiers(graph.nodes.values(), &identifiers);
    for group in selection
        .groups
        .iter()
        .filter_map(|identifier| graph.groups.get(identifier))
    {
        bounds = Some(union_bounds(bounds, group.bounds));
    }
    for reroute in selection
        .reroutes
        .iter()
        .filter_map(|identifier| graph.reroutes.get(identifier))
    {
        bounds = Some(union_bounds(
            bounds,
            GraphRect {
                origin: GraphPoint {
                    x: reroute.position.x - 6.0,
                    y: reroute.position.y - 6.0,
                },
                size: GraphSize {
                    width: 12.0,
                    height: 12.0,
                },
            },
        ));
    }
    bounds
}

fn union_bounds(existing: Option<GraphRect>, next: GraphRect) -> GraphRect {
    let Some(existing) = existing else {
        return next;
    };
    let minimum_x = existing.origin.x.min(next.origin.x);
    let minimum_y = existing.origin.y.min(next.origin.y);
    let maximum_x = (existing.origin.x + existing.size.width).max(next.origin.x + next.size.width);
    let maximum_y =
        (existing.origin.y + existing.size.height).max(next.origin.y + next.size.height);
    GraphRect {
        origin: GraphPoint {
            x: minimum_x,
            y: minimum_y,
        },
        size: GraphSize {
            width: maximum_x - minimum_x,
            height: maximum_y - minimum_y,
        },
    }
}

fn optimal_node_size(node: &GraphNode) -> GraphSize {
    let content_rows = node
        .inputs
        .len()
        .max(node.outputs.len())
        .saturating_add(node.widgets.iter().filter(|widget| widget.visible).count());
    let longest_label = node
        .inputs
        .iter()
        .map(|port| port.name.len())
        .chain(node.outputs.iter().map(|port| port.name.len()))
        .chain(node.widgets.iter().map(|widget| widget.identifier.len()))
        .chain(std::iter::once(node.title.len()))
        .max()
        .unwrap_or_default();
    GraphSize {
        width: (longest_label as f32 * 8.0 + 72.0).clamp(180.0, 520.0),
        height: if node.collapsed {
            34.0
        } else {
            (58.0 + content_rows as f32 * 24.0).max(96.0)
        },
    }
}

fn bounds_for_identifiers<'a>(
    nodes: impl Iterator<Item = &'a GraphNode>,
    identifiers: &BTreeSet<GraphIdentifier>,
) -> Option<GraphRect> {
    let mut selected = nodes.filter(|node| identifiers.contains(&node.identifier));
    let first = selected.next()?.bounds();
    let mut minimum_x = first.origin.x;
    let mut minimum_y = first.origin.y;
    let mut maximum_x = first.origin.x + first.size.width;
    let mut maximum_y = first.origin.y + first.size.height;
    for bounds in selected.map(GraphNode::bounds) {
        minimum_x = minimum_x.min(bounds.origin.x);
        minimum_y = minimum_y.min(bounds.origin.y);
        maximum_x = maximum_x.max(bounds.origin.x + bounds.size.width);
        maximum_y = maximum_y.max(bounds.origin.y + bounds.size.height);
    }
    Some(GraphRect {
        origin: GraphPoint {
            x: minimum_x,
            y: minimum_y,
        },
        size: GraphSize {
            width: maximum_x - minimum_x,
            height: maximum_y - minimum_y,
        },
    })
}

fn expanded_bounds(bounds: GraphRect, padding: f32) -> Result<GraphRect, GraphActionError> {
    if !padding.is_finite() || padding < 0.0 {
        return Err(GraphActionError::InvalidPadding(padding));
    }
    Ok(GraphRect {
        origin: GraphPoint {
            x: bounds.origin.x - padding,
            y: bounds.origin.y - padding,
        },
        size: GraphSize {
            width: bounds.size.width + padding * 2.0,
            height: bounds.size.height + padding * 2.0,
        },
    })
}

fn set_subgraph_metadata(
    model: &mut GraphWorkspaceModel,
    definition_identifier: GraphIdentifier,
    description: Option<String>,
    aliases: Option<Vec<String>>,
) -> Result<(), GraphActionError> {
    let definition = model
        .document()
        .and_then(|document| document.active_graph().ok())
        .and_then(|graph| graph.definitions.get(&definition_identifier))
        .ok_or_else(|| sim_runtime::GraphError::UnknownSubgraph(definition_identifier.clone()))?;
    model.apply(GraphCommand::SetSubgraphMetadata {
        definition_identifier,
        description: description.unwrap_or_else(|| definition.description.clone()),
        search_aliases: aliases.unwrap_or_else(|| definition.search_aliases.clone()),
    })?;
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum GraphActionError {
    #[error(transparent)]
    Workspace(#[from] GraphWorkspaceError),
    #[error(transparent)]
    Graph(#[from] sim_runtime::GraphError),
    #[error("graph action {0:?} requires typed input")]
    InputRequired(CatalogGraphAction),
    #[error("graph action {0:?} is owned by the Zed settings store")]
    RequiresSettingsStore(CatalogGraphAction),
    #[error("graph action {0:?} requires the canonical native asset service")]
    RequiresAssetService(CatalogGraphAction),
    #[error("graph action {action:?} received unexpected input {input}")]
    UnexpectedInput {
        action: CatalogGraphAction,
        input: String,
    },
    #[error("graph model action {0:?} requires typed input")]
    ModelInputRequired(GraphModelAction),
    #[error("graph model action {action:?} received unexpected input {input}")]
    ModelUnexpectedInput {
        action: GraphModelAction,
        input: String,
    },
    #[error("graph selection is empty")]
    EmptySelection,
    #[error("graph is empty")]
    EmptyGraph,
    #[error("clipboard payload is invalid: {0}")]
    Clipboard(String),
    #[error("group padding {0} must be finite and nonnegative")]
    InvalidPadding(f32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sim_runtime::{
        GraphCommandEngine, GraphLevel, GraphNodeMode, GraphPortType, GraphReroute,
        GraphWidgetKind, SubgraphDefinition, WidgetValidation, WorkflowSaveCoordinator,
    };
    use std::{collections::BTreeMap, error::Error};

    #[test]
    fn clipboard_uses_the_canonical_sim_marker() -> Result<(), Box<dyn Error>> {
        let mut model = model_action_fixture()?;
        let GraphActionEffect::ClipboardText(text) = GraphCommandModel::execute(
            &mut model,
            CatalogGraphAction::CopySelected,
            GraphActionInput::None,
        )?
        else {
            return Err("copy did not produce graph clipboard text".into());
        };
        assert!(text.starts_with("application/x-zed-sim-graph+json;version=1\n"));
        decode_clipboard(text.as_bytes())?;
        let prior_marker = text.replacen(
            "application/x-zed-sim-graph",
            "application/x-zed-comfy-graph",
            1,
        );
        assert!(decode_clipboard(prior_marker.as_bytes()).is_err());
        Ok(())
    }

    fn model_action_fixture() -> Result<GraphWorkspaceModel, GraphWorkspaceError> {
        let mut document = GraphDocument::default();
        let first_identifier = GraphIdentifier::from("first");
        let second_identifier = GraphIdentifier::from("second");
        let first = GraphNode::new(
            first_identifier.clone(),
            "Fixture",
            "First",
            GraphPoint::ZERO,
        );
        let second = GraphNode::new(
            second_identifier.clone(),
            "Fixture",
            "Second",
            GraphPoint { x: 200.0, y: 80.0 },
        );
        document.root.nodes.insert(first_identifier.clone(), first);
        document
            .root
            .nodes
            .insert(second_identifier.clone(), second);
        document.root.selection.nodes =
            BTreeSet::from([first_identifier.clone(), second_identifier.clone()]);
        let group_identifier = GraphIdentifier::from("group");
        document.root.groups.insert(
            group_identifier.clone(),
            GraphGroup {
                identifier: group_identifier,
                title: "Group".to_owned(),
                bounds: GraphRect {
                    origin: GraphPoint::ZERO,
                    size: GraphSize {
                        width: 400.0,
                        height: 240.0,
                    },
                },
                node_ids: BTreeSet::from([first_identifier, second_identifier]),
                collapsed: false,
                pinned: false,
                color: None,
                source_fields: Map::new(),
            },
        );
        document.root.reroutes.insert(
            GraphIdentifier::from("parent-reroute"),
            GraphReroute {
                identifier: GraphIdentifier::from("parent-reroute"),
                position: GraphPoint::ZERO,
                parent: None,
                floating_type: None,
                source_fields: Map::new(),
            },
        );
        document.root.reroutes.insert(
            GraphIdentifier::from("child-reroute"),
            GraphReroute {
                identifier: GraphIdentifier::from("child-reroute"),
                position: GraphPoint { x: 10.0, y: 10.0 },
                parent: None,
                floating_type: None,
                source_fields: Map::new(),
            },
        );
        let internal_identifier = GraphIdentifier::from("internal");
        let mut internal = GraphNode::new(
            internal_identifier.clone(),
            "Fixture",
            "Internal",
            GraphPoint::ZERO,
        );
        internal.inputs.push(GraphPort::new(
            "seed",
            GraphPortType::Concrete("INT".to_owned()),
        ));
        internal.widgets.push(GraphWidget {
            identifier: "seed".to_owned(),
            kind: GraphWidgetKind::Integer {
                minimum: 0,
                maximum: 100,
                step: 1,
            },
            value: json!(4),
            prompt_value: json!(5),
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        });
        let mut definition_graph = GraphLevel::default();
        definition_graph.nodes.insert(internal_identifier, internal);
        let definition_identifier = GraphIdentifier::from("definition");
        document.root.definitions.insert(
            definition_identifier.clone(),
            SubgraphDefinition {
                identifier: definition_identifier,
                name: "Definition".to_owned(),
                graph: Box::new(definition_graph),
                inputs: Vec::new(),
                outputs: Vec::new(),
                published: false,
                description: String::new(),
                search_aliases: Vec::new(),
                exposed_widgets: Vec::new(),
                graph_inline: false,
                unknown: BTreeMap::new(),
            },
        );
        let bytes = document.to_workflow_bytes()?;
        let save_coordinator = WorkflowSaveCoordinator::new(
            "action-fixture",
            sim_runtime::WorkflowStorageProvider::Draft,
            bytes,
        )?;
        Ok(GraphWorkspaceModel {
            schema_version: crate::GRAPH_WORKSPACE_SCHEMA_VERSION,
            title: "Action fixture".to_owned(),
            open_state: crate::WorkflowOpenState::Editable(GraphCommandEngine::new(document)?),
            save_coordinator,
            execution_association: None,
            canvas_info_visible: false,
            last_error: None,
            operation_errors: Vec::new(),
            announcement: None,
        })
    }

    #[test]
    fn typed_model_actions_dispatch_real_atomic_graph_commands() -> Result<(), Box<dyn Error>> {
        let mut model = model_action_fixture()?;
        let before_invalid = model.encode()?;
        assert!(matches!(
            GraphCommandModel::execute_model_action(
                &mut model,
                GraphModelAction::RenameNode,
                GraphActionInput::NodeColor {
                    identifier: GraphIdentifier::from("first"),
                    color: None,
                },
            ),
            Err(GraphActionError::ModelUnexpectedInput { .. })
        ));
        assert_eq!(model.encode()?, before_invalid);

        let cases = [
            (
                GraphModelAction::RenameNode,
                GraphActionInput::RenameNode {
                    identifier: GraphIdentifier::from("first"),
                    title: "Renamed".to_owned(),
                },
            ),
            (
                GraphModelAction::SetNodeColor,
                GraphActionInput::NodeColor {
                    identifier: GraphIdentifier::from("first"),
                    color: Some("#123456".to_owned()),
                },
            ),
            (
                GraphModelAction::ToggleNodeDisable,
                GraphActionInput::NodeIdentifiers(BTreeSet::from([GraphIdentifier::from("first")])),
            ),
            (
                GraphModelAction::LayoutSelection,
                GraphActionInput::Layout {
                    operation: LayoutOperation::AlignTop,
                    spacing: 24.0,
                },
            ),
            (
                GraphModelAction::Ungroup,
                GraphActionInput::GroupIdentifier(GraphIdentifier::from("group")),
            ),
            (
                GraphModelAction::MoveReroute,
                GraphActionInput::RerouteMove {
                    identifier: GraphIdentifier::from("parent-reroute"),
                    position: GraphPoint { x: 40.0, y: 20.0 },
                },
            ),
            (
                GraphModelAction::ReparentReroute,
                GraphActionInput::RerouteParent {
                    identifier: GraphIdentifier::from("child-reroute"),
                    parent: Some(GraphIdentifier::from("parent-reroute")),
                },
            ),
            (
                GraphModelAction::RemoveReroute,
                GraphActionInput::RerouteIdentifier(GraphIdentifier::from("child-reroute")),
            ),
            (
                GraphModelAction::SetSubgraphWidgetExposure,
                GraphActionInput::SubgraphWidgetExposure {
                    definition: GraphIdentifier::from("definition"),
                    internal_node: GraphIdentifier::from("internal"),
                    widget: "seed".to_owned(),
                    exposed: true,
                },
            ),
            (
                GraphModelAction::RemoveSubgraphDefinition,
                GraphActionInput::RemoveSubgraphDefinition {
                    definition: GraphIdentifier::from("definition"),
                    remove_instances: false,
                },
            ),
        ];
        for (action, input) in cases {
            assert_eq!(
                GraphCommandModel::execute_model_action(&mut model, action, input)?,
                GraphActionEffect::None
            );
        }

        let graph = model
            .document()
            .and_then(|document| document.active_graph().ok())
            .ok_or("active graph")?;
        let first = graph
            .nodes
            .get(&GraphIdentifier::from("first"))
            .ok_or("first node")?;
        assert_eq!(first.title, "Renamed");
        assert_eq!(first.color.as_deref(), Some("#123456"));
        assert_eq!(first.mode, GraphNodeMode::OnEvent);
        assert!(!graph.groups.contains_key(&GraphIdentifier::from("group")));
        assert!(
            !graph
                .reroutes
                .contains_key(&GraphIdentifier::from("child-reroute"))
        );
        assert!(
            !graph
                .definitions
                .contains_key(&GraphIdentifier::from("definition"))
        );
        Ok(())
    }

    #[test]
    fn settings_owned_actions_cannot_mutate_graph_workspace_state() -> Result<(), Box<dyn Error>> {
        let mut model = model_action_fixture()?;
        let workflow_before = model
            .document()
            .ok_or("editable graph document")?
            .to_workflow_bytes()?;
        let snapshot_before = model.encode()?;
        let can_undo_before = model.engine().is_some_and(GraphCommandEngine::can_undo);

        for (action, input) in [
            (CatalogGraphAction::ToggleVueNodes, GraphActionInput::None),
            (
                CatalogGraphAction::GroupSelectedNodes,
                GraphActionInput::None,
            ),
            (
                CatalogGraphAction::FitGroupToContents,
                GraphActionInput::GroupIdentifier(GraphIdentifier::from("group")),
            ),
        ] {
            assert_eq!(
                GraphCommandModel::execute(&mut model, action, input),
                Err(GraphActionError::RequiresSettingsStore(action))
            );
        }

        assert_eq!(
            model
                .document()
                .ok_or("editable graph document")?
                .to_workflow_bytes()?,
            workflow_before
        );
        assert_eq!(model.encode()?, snapshot_before);
        assert_eq!(
            model.engine().is_some_and(GraphCommandEngine::can_undo),
            can_undo_before
        );
        Ok(())
    }
}
