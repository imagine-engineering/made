use crate::{
    CommandNativeStatus, GeneratedGraphContextAction, GeneratedGraphContextInfrastructure,
    GeneratedGraphContextSurface, GeneratedMenuItemKind, GraphActionInput, GraphWorkspaceItem,
    graph_context_registry, menu_registration, native_asset_services,
    properties_panel::{
        GraphNodePropertyKind, graph_node_properties_with_value, graph_node_property_descriptors,
        graph_node_property_value_label, parse_graph_node_property_value,
    },
};
use editor::Editor;
use fs::Fs;
use futures::channel::oneshot;
use gpui::{
    App, AppContext as _, Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable,
    PromptLevel, Render, Role, WeakEntity, Window, px,
};
use serde_json::{Map, Value};
use settings::SettingsStore;
use sim_runtime::{
    AssetCollisionPolicy, AssetError, CatalogGraphAction, ContentRevision, GraphCommand,
    GraphGroup, GraphIdentifier, GraphNodeMode, GraphPaletteColor, GraphPoint, GraphRect,
    GraphSelection, GraphSize, GraphSlotDirection, GraphVisualShape, GroupToggle, LayoutOperation,
    NodeToggle, SubgraphBlueprintLibraryError,
};
use sim_types::CancellationToken;
use std::collections::BTreeSet;
use ui::{
    Button, ButtonStyle, ContextMenu, ContextMenuEntry, IconPosition, Modal, ModalFooter,
    ModalHeader, Section, prelude::*,
};
use workspace::{DismissDecision, ModalView};

#[derive(Clone, Debug, PartialEq)]
pub enum GraphContextTarget {
    Canvas {
        graph_position: GraphPoint,
    },
    Selection,
    Node(GraphIdentifier),
    Group(GraphIdentifier),
    Reroute(GraphIdentifier),
    Slot {
        direction: GraphSlotDirection,
        slot: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphContextInvocation {
    pub document_identity: uuid::Uuid,
    pub content_revision: ContentRevision,
    pub navigation: Vec<GraphIdentifier>,
    pub selection: GraphSelection,
    pub target: GraphContextTarget,
    pub screen_position: GraphPoint,
}

#[derive(Clone)]
pub(crate) struct GraphContextMenuState {
    pub invocation: GraphContextInvocation,
    pub return_focus: Option<FocusHandle>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GraphContextInputOperation {
    RenameNode(GraphIdentifier),
    RenameGroup(GraphIdentifier),
    BatchRenameSelection {
        nodes: Vec<GraphIdentifier>,
        groups: Vec<GraphIdentifier>,
    },
    FrameSelection,
    RenameSlot {
        direction: GraphSlotDirection,
        slot: usize,
    },
    SetGroupFontSize(GraphIdentifier),
    EditNodeProperty {
        identifier: GraphIdentifier,
        key: String,
    },
    ConvertToSubgraph,
    PublishSubgraph,
}

enum SubgraphPublishCompletion {
    Published {
        display_name: String,
        diagnostic_message: Option<String>,
    },
    AlreadyExists,
    Failed(String),
}

#[derive(Clone)]
pub(crate) struct GraphContextInputState {
    pub invocation: GraphContextInvocation,
    pub operation: GraphContextInputOperation,
    pub editor: Entity<Editor>,
}

pub(crate) struct GraphContextInputModal {
    item: WeakEntity<GraphWorkspaceItem>,
    operation: GraphContextInputOperation,
    editor: Entity<Editor>,
}

impl EventEmitter<DismissEvent> for GraphContextInputModal {}

impl Focusable for GraphContextInputModal {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.editor.focus_handle(cx)
    }
}

impl ModalView for GraphContextInputModal {
    fn on_before_dismiss(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> DismissDecision {
        if let Err(error) = self.item.update(cx, |item, cx| {
            if item.context_input.take().is_some() {
                item.model.announcement = Some("Cancelled native context input".to_owned());
                cx.notify();
            }
        }) {
            log::error!("graph disappeared while dismissing context input modal: {error}");
        }
        DismissDecision::Dismiss(true)
    }
}

impl GraphContextInputModal {
    fn confirm(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let should_dismiss = match self.item.update(cx, |item, cx| {
            item.confirm_context_input(window, cx);
            item.context_input.is_none()
        }) {
            Ok(should_dismiss) => should_dismiss,
            Err(error) => {
                log::error!("graph disappeared while confirming context input modal: {error}");
                true
            }
        };
        if should_dismiss {
            cx.emit(DismissEvent);
        }
    }

    fn cancel(&mut self, _: &menu::Cancel, _window: &mut Window, cx: &mut Context<Self>) {
        cx.emit(DismissEvent);
    }
}

impl Render for GraphContextInputModal {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let label = context_input_label(&self.operation);
        v_flex()
            .id("sim-context-input-modal")
            .debug_selector(|| "SIM-CONTEXT-INPUT".into())
            .role(Role::Dialog)
            .aria_label(label)
            .key_context("SimContextInputModal ComfyContextInputModal")
            .on_action(cx.listener(Self::confirm))
            .on_action(cx.listener(Self::cancel))
            .w_96()
            .elevation_3(cx)
            .child(
                Modal::new("sim-context-input", None)
                    .header(ModalHeader::new().headline(label).show_dismiss_button(true))
                    .section(Section::new().child(self.editor.clone()))
                    .footer(
                        ModalFooter::new().end_slot(
                            h_flex()
                                .gap_2()
                                .child(
                                    Button::new("sim-context-input-cancel", "Cancel")
                                        .style(ButtonStyle::Subtle)
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.cancel(&menu::Cancel, window, cx);
                                        })),
                                )
                                .child(
                                    Button::new("sim-context-input-confirm", "Apply")
                                        .style(ButtonStyle::Filled)
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.confirm(&menu::Confirm, window, cx);
                                        })),
                                ),
                        ),
                    ),
            )
    }
}

fn context_input_label(operation: &GraphContextInputOperation) -> &'static str {
    match operation {
        GraphContextInputOperation::RenameNode(_) => "Rename node",
        GraphContextInputOperation::RenameGroup(_) => "Rename group",
        GraphContextInputOperation::BatchRenameSelection { .. } => "Rename selection",
        GraphContextInputOperation::FrameSelection => "Frame selected nodes",
        GraphContextInputOperation::RenameSlot { .. } => "Rename slot",
        GraphContextInputOperation::SetGroupFontSize(_) => "Set group font size",
        GraphContextInputOperation::EditNodeProperty { .. } => "Edit node property",
        GraphContextInputOperation::ConvertToSubgraph => "Name subgraph",
        GraphContextInputOperation::PublishSubgraph => "Publish subgraph blueprint",
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphContextDispatchInput {
    None,
    Shape(GraphVisualShape),
    PaletteColor(Option<GraphPaletteColor>),
    NodeMode(GraphNodeMode),
    Layout(LayoutOperation),
    GroupFontSize(f32),
    NodeProperty { key: String, value: Option<Value> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphContextDispatchOutcome {
    Executed,
    InputPending,
    ConfirmationPending,
    Rejected(String),
}

#[derive(Clone, Debug)]
pub struct GraphContextMenuEntry {
    pub action: GeneratedGraphContextAction,
    pub feature_id: String,
    pub catalog_label: String,
    pub surface: GeneratedGraphContextSurface,
    pub label: String,
    pub enabled: bool,
    pub checked: Option<bool>,
    pub disabled_reason: Option<String>,
    binding: GraphContextActionBinding,
}

impl GraphContextMenuEntry {
    pub(crate) fn has_submenu(&self) -> bool {
        action_has_submenu(self.action)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphContextActionBinding {
    feature_id: String,
    action: GeneratedGraphContextAction,
    surface: GeneratedGraphContextSurface,
    item_kind: GeneratedMenuItemKind,
    source_condition: String,
    catalog_label: String,
}

impl GraphContextActionBinding {
    fn from_registration(registration: crate::MenuReconciliation) -> Result<Self, String> {
        if registration.owner != crate::GRAPH_CONTEXT_MENU_OWNER {
            return Err(format!(
                "menu feature {} is owned by {} instead of the native graph context service",
                registration.feature_id, registration.owner
            ));
        }
        if registration.status != CommandNativeStatus::Executable {
            return Err(format!(
                "menu feature {} is not an executable graph context action",
                registration.feature_id
            ));
        }
        let action = registration.context_action.ok_or_else(|| {
            format!(
                "menu feature {} has no typed graph context action",
                registration.feature_id
            )
        })?;
        let surface = registration.context_surface.ok_or_else(|| {
            format!(
                "menu feature {} has no typed graph context surface",
                registration.feature_id
            )
        })?;
        if surface == GeneratedGraphContextSurface::Infrastructure
            || registration.context_infrastructure.is_some()
        {
            return Err(format!(
                "menu feature {} is infrastructure rather than an executable action",
                registration.feature_id
            ));
        }
        Ok(Self {
            feature_id: registration.feature_id,
            action,
            surface,
            item_kind: registration.item_kind,
            source_condition: registration.source_condition,
            catalog_label: registration.label,
        })
    }

    fn revalidate(&self) -> Result<(), String> {
        let registration = menu_registration(&self.feature_id)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("unknown graph context feature {}", self.feature_id))?;
        let current = Self::from_registration(registration)?;
        if current == *self {
            Ok(())
        } else {
            Err(format!(
                "graph context feature {} changed after the menu opened",
                self.feature_id
            ))
        }
    }
}

pub(crate) fn graph_context_action_binding(
    feature_id: &str,
) -> Result<GraphContextActionBinding, String> {
    require_graph_context_infrastructure(GeneratedGraphContextInfrastructure::CommandAdapter)?;
    let registration = menu_registration(feature_id)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("unknown graph context feature {feature_id}"))?;
    GraphContextActionBinding::from_registration(registration)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphContextInfrastructureBinding {
    pub feature_id: String,
    pub source: GeneratedGraphContextInfrastructure,
}

pub(crate) fn graph_context_infrastructure_bindings()
-> Result<Vec<GraphContextInfrastructureBinding>, String> {
    let mut bindings = Vec::new();
    for registration in graph_context_registry() {
        let registration = registration.map_err(|error| error.to_string())?;
        let Some(infrastructure) = registration.context_infrastructure else {
            continue;
        };
        if registration.context_surface != Some(GeneratedGraphContextSurface::Infrastructure)
            || registration.context_action.is_some()
            || registration.item_kind != GeneratedMenuItemKind::Infrastructure
        {
            return Err(format!(
                "graph context infrastructure feature {} has an executable or non-infrastructure projection",
                registration.feature_id
            ));
        }
        bindings.push(GraphContextInfrastructureBinding {
            feature_id: registration.feature_id,
            source: infrastructure,
        });
    }
    let unique = bindings
        .iter()
        .map(|binding| binding.source)
        .collect::<BTreeSet<_>>();
    if bindings.len() != 8 || unique.len() != 8 {
        return Err(format!(
            "graph context infrastructure must resolve to eight unique generated prerequisites; resolved {} rows and {} prerequisites",
            bindings.len(),
            unique.len()
        ));
    }
    Ok(bindings)
}

pub(crate) fn require_graph_context_infrastructure(
    source: GeneratedGraphContextInfrastructure,
) -> Result<GraphContextInfrastructureBinding, String> {
    let mut matching = graph_context_infrastructure_bindings()?
        .into_iter()
        .filter(|binding| binding.source == source);
    let binding = matching.next().ok_or_else(|| {
        format!("native graph context infrastructure {source:?} is not registered")
    })?;
    if matching.next().is_some() {
        return Err(format!(
            "native graph context infrastructure {source:?} has more than one owner"
        ));
    }
    Ok(binding)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ContextActionAvailability {
    enabled: bool,
    checked: Option<bool>,
    reason: Option<String>,
}

impl ContextActionAvailability {
    fn enabled() -> Self {
        Self {
            enabled: true,
            checked: None,
            reason: None,
        }
    }

    fn checked(value: bool) -> Self {
        Self {
            checked: Some(value),
            ..Self::enabled()
        }
    }

    fn disabled(reason: impl Into<String>) -> Self {
        Self {
            enabled: false,
            checked: None,
            reason: Some(reason.into()),
        }
    }
}

fn target_surface_matches(
    invocation: &GraphContextInvocation,
    surface: GeneratedGraphContextSurface,
) -> bool {
    match &invocation.target {
        GraphContextTarget::Canvas { .. } => {
            matches!(
                surface,
                GeneratedGraphContextSurface::Canvas | GeneratedGraphContextSurface::CanvasMode
            ) || (!invocation.selection.is_empty()
                && surface == GeneratedGraphContextSurface::Selection)
        }
        GraphContextTarget::Selection => surface == GeneratedGraphContextSurface::Selection,
        GraphContextTarget::Node(_) => {
            surface == GeneratedGraphContextSurface::Node
                || (!invocation.selection.is_empty()
                    && surface == GeneratedGraphContextSurface::Selection)
        }
        GraphContextTarget::Group(_) => {
            surface == GeneratedGraphContextSurface::Group
                || (!invocation.selection.is_empty()
                    && surface == GeneratedGraphContextSurface::Selection)
        }
        GraphContextTarget::Reroute(_) => {
            surface == GeneratedGraphContextSurface::Reroute
                || (!invocation.selection.is_empty()
                    && surface == GeneratedGraphContextSurface::Selection)
        }
        GraphContextTarget::Slot { .. } => surface == GeneratedGraphContextSurface::Slot,
    }
}

pub fn graph_context_menu_entries(
    item: &GraphWorkspaceItem,
    invocation: &GraphContextInvocation,
    cx: &App,
) -> Result<Vec<GraphContextMenuEntry>, String> {
    for prerequisite in [
        GeneratedGraphContextInfrastructure::RegisterMenuGroup,
        GeneratedGraphContextInfrastructure::CoreMenuLoader,
        GeneratedGraphContextInfrastructure::TranslatedRegistryItems,
        GeneratedGraphContextInfrastructure::ContextMenuConverter,
        GeneratedGraphContextInfrastructure::MergedMoreOptions,
    ] {
        require_graph_context_infrastructure(prerequisite)?;
    }
    let invocation_error = item.validate_context_invocation(invocation, false).err();
    let mut entries = Vec::new();
    for registration in graph_context_registry() {
        let registration = registration.map_err(|error| error.to_string())?;
        if registration.context_infrastructure.is_some() {
            continue;
        }
        let binding = GraphContextActionBinding::from_registration(registration)?;
        let action = binding.action;
        let surface = binding.surface;
        if !target_surface_matches(invocation, surface) {
            continue;
        }
        let availability = invocation_error.as_ref().map_or_else(
            || context_feature_availability(item, invocation, &binding, cx),
            ContextActionAvailability::disabled,
        );
        entries.push(GraphContextMenuEntry {
            action,
            feature_id: binding.feature_id.clone(),
            catalog_label: binding.catalog_label.clone(),
            surface,
            label: context_action_label(item, invocation, &binding, &binding.catalog_label),
            enabled: availability.enabled,
            checked: availability.checked,
            disabled_reason: availability.reason,
            binding,
        });
    }
    Ok(entries)
}

fn context_feature_availability(
    item: &GraphWorkspaceItem,
    invocation: &GraphContextInvocation,
    binding: &GraphContextActionBinding,
    cx: &App,
) -> ContextActionAvailability {
    match binding.feature_id.as_str() {
        "COMFY-MENU-117" if invocation.selection.nodes.len() < 2 => {
            return ContextActionAvailability::disabled(
                "more than one selected node is required for this conversion",
            );
        }
        "COMFY-MENU-152" if context_has_group_at_pointer(item, invocation) => {
            return ContextActionAvailability::disabled(
                "a new group cannot be created under an existing group target",
            );
        }
        "COMFY-MENU-126" => {
            let (has_advanced_widget, advanced_visible) = item
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .and_then(|graph| {
                    let GraphContextTarget::Node(identifier) = &invocation.target else {
                        return None;
                    };
                    let node = graph.nodes.get(identifier)?;
                    Some((
                        node_has_advanced_widgets(node),
                        node.source_fields
                            .get("show_advanced")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    ))
                })
                .unwrap_or((false, false));
            if !has_advanced_widget {
                return ContextActionAvailability {
                    enabled: false,
                    checked: Some(advanced_visible),
                    reason: Some("node has no advanced widgets".to_owned()),
                };
            }
        }
        _ => {}
    }
    context_action_availability(item, invocation, binding, cx)
}

fn context_action_availability(
    item: &GraphWorkspaceItem,
    invocation: &GraphContextInvocation,
    binding: &GraphContextActionBinding,
    cx: &App,
) -> ContextActionAvailability {
    let action = binding.action;
    let surface = binding.surface;
    if is_settings_owned_context_action(action) {
        return settings_context_action_availability(item, invocation, action, cx);
    }
    let Some(document) = item.model.document() else {
        return ContextActionAvailability::disabled("workflow is read-only");
    };
    let Ok(graph) = document.active_graph() else {
        return ContextActionAvailability::disabled("active graph is unavailable");
    };
    let mutating = !matches!(
        action,
        GeneratedGraphContextAction::CanvasHandMode
            | GeneratedGraphContextAction::CanvasSelectMode
            | GeneratedGraphContextAction::CopySelection
            | GeneratedGraphContextAction::OpenNodePropertiesPanel
            | GeneratedGraphContextAction::SelectGroupNodes
            | GeneratedGraphContextAction::ToggleDefaultRerouteType
            | GeneratedGraphContextAction::ToggleNodesTwo
    );
    if mutating && item.model.is_read_only() {
        return ContextActionAvailability::disabled("workflow is read-only");
    }
    if action_requires_context_input_workspace(action) && item.workspace().upgrade().is_none() {
        return ContextActionAvailability::disabled(
            "workspace modal layer is unavailable for context input",
        );
    }
    let node_scope = context_node_scope(binding, invocation);
    match action {
        GeneratedGraphContextAction::CanvasSelectMode => {
            ContextActionAvailability::checked(!item.canvas_is_locked())
        }
        GeneratedGraphContextAction::CanvasHandMode => {
            ContextActionAvailability::checked(item.canvas_is_locked())
        }
        GeneratedGraphContextAction::ToggleNodesTwo => {
            settings_context_action_availability(item, invocation, action, cx)
        }
        GeneratedGraphContextAction::AddGroup | GeneratedGraphContextAction::Paste => {
            target_required(&invocation.target, "canvas", |target| {
                matches!(target, GraphContextTarget::Canvas { .. })
            })
        }
        GeneratedGraphContextAction::AdjustNodeSize => {
            let GraphContextTarget::Node(identifier) = &invocation.target else {
                return ContextActionAvailability::disabled("node target is required");
            };
            if !graph
                .nodes
                .get(identifier)
                .is_some_and(|node| node.is_resizable())
            {
                return ContextActionAvailability::disabled("clicked node cannot be resized");
            }
            let commands = node_scope
                .iter()
                .filter_map(|identifier| {
                    graph
                        .nodes
                        .get(identifier)
                        .map(|node| GraphCommand::ResizeNode {
                            identifier: identifier.clone(),
                            size: node.size,
                        })
                })
                .collect();
            command_availability(item, &GraphCommand::Batch { commands })
        }
        GeneratedGraphContextAction::ChooseNodeColor => command_availability(
            item,
            &GraphCommand::SetNodePalette {
                identifiers: node_scope,
                color: None,
            },
        ),
        GeneratedGraphContextAction::ChooseNodeMode => command_availability(
            item,
            &GraphCommand::SetNodeMode {
                identifiers: node_scope,
                mode: GraphNodeMode::Always,
            },
        ),
        GeneratedGraphContextAction::ChooseNodeShape => command_availability(
            item,
            &GraphCommand::SetNodeShape {
                identifiers: node_scope,
                shape: GraphVisualShape::Default,
            },
        ),
        GeneratedGraphContextAction::ToggleNodeBypass => checked_if_enabled(
            command_availability(
                item,
                &GraphCommand::ToggleNodes {
                    identifiers: node_scope.clone(),
                    toggle: NodeToggle::Bypass,
                },
            ),
            node_scope.iter().all(|identifier| {
                graph
                    .nodes
                    .get(identifier)
                    .is_some_and(|node| node.mode == GraphNodeMode::Bypass)
            }),
        ),
        GeneratedGraphContextAction::ToggleNodeCollapse => {
            let GraphContextTarget::Node(identifier) = &invocation.target else {
                return ContextActionAvailability::disabled("node target is required");
            };
            if !graph
                .nodes
                .get(identifier)
                .is_some_and(|node| node.is_collapsible())
            {
                return ContextActionAvailability::disabled("clicked node cannot be collapsed");
            }
            checked_if_enabled(
                command_availability(
                    item,
                    &GraphCommand::ToggleNodes {
                        identifiers: node_scope.clone(),
                        toggle: NodeToggle::Collapse,
                    },
                ),
                node_scope.iter().all(|identifier| {
                    graph
                        .nodes
                        .get(identifier)
                        .is_some_and(|node| node.collapsed)
                }),
            )
        }
        GeneratedGraphContextAction::ToggleNodePin => checked_if_enabled(
            command_availability(
                item,
                &GraphCommand::ToggleNodes {
                    identifiers: node_scope.clone(),
                    toggle: NodeToggle::Pin,
                },
            ),
            node_scope
                .iter()
                .all(|identifier| graph.nodes.get(identifier).is_some_and(|node| node.pinned)),
        ),
        GeneratedGraphContextAction::DuplicateSelection => command_availability(
            item,
            &GraphCommand::DuplicateSelection {
                selection: invocation.selection.clone(),
                offset: GraphPoint { x: 24.0, y: 24.0 },
            },
        ),
        GeneratedGraphContextAction::DeleteSelection => command_availability(
            item,
            &GraphCommand::RemoveItems {
                selection: if surface == GeneratedGraphContextSurface::Node {
                    let GraphContextTarget::Node(identifier) = &invocation.target else {
                        return ContextActionAvailability::disabled("node target is required");
                    };
                    GraphSelection {
                        nodes: BTreeSet::from([identifier.clone()]),
                        ..GraphSelection::default()
                    }
                } else {
                    invocation.selection.clone()
                },
            },
        ),
        GeneratedGraphContextAction::ToggleAdvancedWidgets => {
            let visible = node_scope.iter().all(|identifier| {
                graph
                    .nodes
                    .get(identifier)
                    .and_then(|node| node.source_fields.get("show_advanced"))
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            });
            match command_availability(
                item,
                &GraphCommand::SetNodeAdvancedVisibility {
                    identifiers: node_scope,
                    visible: !visible,
                },
            ) {
                availability if availability.enabled => ContextActionAvailability::checked(visible),
                availability => availability,
            }
        }
        GeneratedGraphContextAction::OpenNodeProperties => {
            let GraphContextTarget::Node(identifier) = &invocation.target else {
                return ContextActionAvailability::disabled("node target is required");
            };
            let Some(node) = graph.nodes.get(identifier) else {
                return ContextActionAvailability::disabled("node target is unavailable");
            };
            match graph_node_property_descriptors(node) {
                Ok(descriptors) if descriptors.is_empty() => {
                    ContextActionAvailability::disabled("node has no editable properties")
                }
                Ok(_) => ContextActionAvailability::enabled(),
                Err(error) => ContextActionAvailability::disabled(format!(
                    "node property metadata is invalid: {error}"
                )),
            }
        }
        GeneratedGraphContextAction::RenameNode => {
            target_required(&invocation.target, "node", |target| {
                matches!(target, GraphContextTarget::Node(_))
            })
        }
        GeneratedGraphContextAction::OpenNodePropertiesPanel => {
            let target = target_required(&invocation.target, "node", |target| {
                matches!(target, GraphContextTarget::Node(_))
            });
            if !target.enabled {
                return target;
            }
            let available = item
                .workspace()
                .read_with(cx, |workspace, cx| {
                    workspace.panel::<crate::GraphPropertiesPanel>(cx).is_some()
                })
                .unwrap_or(false);
            if available {
                ContextActionAvailability::enabled()
            } else {
                ContextActionAvailability::disabled("workspace properties panel is unavailable")
            }
        }
        GeneratedGraphContextAction::RenameSelection => {
            if !invocation.selection.nodes.is_empty() || !invocation.selection.groups.is_empty() {
                ContextActionAvailability::enabled()
            } else {
                ContextActionAvailability::disabled("selection contains no titled node or group")
            }
        }
        GeneratedGraphContextAction::CopySelection => selection_required(invocation),
        GeneratedGraphContextAction::ConvertToSubgraph => {
            match graph.validate_subgraph_conversion_selection() {
                Ok(()) => ContextActionAvailability::enabled(),
                Err(error) => ContextActionAvailability::disabled(error.to_string()),
            }
        }
        GeneratedGraphContextAction::FrameSelection => {
            if invocation.selection.nodes.len() >= 2 {
                match item.group_selection_command(
                    invocation,
                    "Group".to_owned(),
                    item.group_selected_nodes_padding(cx),
                ) {
                    Ok(command) => command_availability(item, &command),
                    Err(error) => ContextActionAvailability::disabled(error),
                }
            } else {
                ContextActionAvailability::disabled(
                    "at least two selected nodes are required to frame",
                )
            }
        }
        GeneratedGraphContextAction::AddGroupForSelection => {
            if invocation.selection.nodes.is_empty() && invocation.selection.groups.is_empty() {
                ContextActionAvailability::disabled("positionable selection is empty")
            } else {
                match item.group_selection_command(
                    invocation,
                    "Group".to_owned(),
                    item.group_selected_nodes_padding(cx),
                ) {
                    Ok(command) => command_availability(item, &command),
                    Err(error) => ContextActionAvailability::disabled(error),
                }
            }
        }
        GeneratedGraphContextAction::AlignSelection
        | GeneratedGraphContextAction::DistributeSelection => {
            if invocation.selection.nodes.len() < 2 {
                ContextActionAvailability::disabled("at least two selected nodes are required")
            } else {
                command_availability(
                    item,
                    &GraphCommand::LayoutSelection {
                        operation: if action == GeneratedGraphContextAction::AlignSelection {
                            LayoutOperation::AlignLeft
                        } else {
                            LayoutOperation::DistributeHorizontally
                        },
                        spacing: 24.0,
                    },
                )
            }
        }
        GeneratedGraphContextAction::PublishSubgraph => {
            let result = invocation
                .selection
                .nodes
                .iter()
                .next()
                .and_then(|identifier| graph.nodes.get(identifier))
                .map(|node| node.title.as_str())
                .ok_or_else(|| "publishing requires one selected subgraph instance".to_owned())
                .and_then(|display_name| {
                    item.model
                        .document()
                        .ok_or_else(|| "workflow is open read-only".to_owned())?
                        .export_selected_subgraph_blueprint(display_name)
                        .map(drop)
                        .map_err(|error| error.to_string())
                });
            match result {
                Ok(()) => ContextActionAvailability::enabled(),
                Err(error) => ContextActionAvailability::disabled(error),
            }
        }
        GeneratedGraphContextAction::UnpackSubgraph => {
            if invocation.selection.nodes.iter().any(|identifier| {
                graph
                    .nodes
                    .get(identifier)
                    .is_some_and(|node| node.subgraph_definition.is_some())
            }) {
                ContextActionAvailability::enabled()
            } else {
                ContextActionAvailability::disabled("selection contains no subgraph instance")
            }
        }
        GeneratedGraphContextAction::AddSelectionToGroup => {
            if invocation.selection.nodes.is_empty() {
                ContextActionAvailability::disabled("node selection is empty")
            } else {
                let Some(identifier) = invocation.group_identifier() else {
                    return ContextActionAvailability::disabled("group target is required");
                };
                command_availability(
                    item,
                    &GraphCommand::AddNodesToGroup {
                        identifier,
                        nodes: invocation.selection.nodes.clone(),
                        padding: item.group_selected_nodes_padding(cx),
                    },
                )
            }
        }
        GeneratedGraphContextAction::ChooseGroupColor => {
            let Some(identifier) = invocation.group_identifier() else {
                return ContextActionAvailability::disabled("group target is required");
            };
            command_availability(
                item,
                &GraphCommand::SetGroupColor {
                    identifier,
                    color: None,
                },
            )
        }
        GeneratedGraphContextAction::ChooseGroupFontSize => {
            let Some(identifier) = invocation.group_identifier() else {
                return ContextActionAvailability::disabled("group target is required");
            };
            let font_size = graph
                .groups
                .get(&identifier)
                .and_then(|group| group.source_fields.get("font_size"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(20.0) as f32;
            command_availability(
                item,
                &GraphCommand::SetGroupFontSize {
                    identifier,
                    font_size,
                },
            )
        }
        GeneratedGraphContextAction::RenameGroup => {
            let Some(identifier) = invocation.group_identifier() else {
                return ContextActionAvailability::disabled("group target is required");
            };
            let Some(title) = graph
                .groups
                .get(&identifier)
                .map(|group| group.title.clone())
            else {
                return ContextActionAvailability::disabled("group target is unavailable");
            };
            command_availability(item, &GraphCommand::RenameGroup { identifier, title })
        }
        GeneratedGraphContextAction::DeleteGroup => {
            let Some(identifier) = invocation.group_identifier() else {
                return ContextActionAvailability::disabled("group target is required");
            };
            command_availability(item, &GraphCommand::Ungroup { identifier })
        }
        GeneratedGraphContextAction::ToggleGroupPin => {
            let Some(identifier) = invocation.group_identifier() else {
                return ContextActionAvailability::disabled("group target is required");
            };
            let pinned = graph
                .groups
                .get(&identifier)
                .is_some_and(|group| group.pinned);
            checked_if_enabled(
                command_availability(
                    item,
                    &GraphCommand::ToggleGroups {
                        identifiers: BTreeSet::from([identifier]),
                        toggle: GroupToggle::Pin,
                    },
                ),
                pinned,
            )
        }
        GeneratedGraphContextAction::ChooseGroupNodeShape => {
            let Some(nodes) = item.context_group_nodes(invocation) else {
                return ContextActionAvailability::disabled("group target is unavailable");
            };
            command_availability(
                item,
                &GraphCommand::SetNodeShape {
                    identifiers: nodes,
                    shape: GraphVisualShape::Default,
                },
            )
        }
        GeneratedGraphContextAction::FitGroup => {
            let Some(identifier) = invocation.group_identifier() else {
                return ContextActionAvailability::disabled("group target is required");
            };
            command_availability(
                item,
                &GraphCommand::FitGroup {
                    identifier,
                    padding: item.group_selected_nodes_padding(cx),
                },
            )
        }
        GeneratedGraphContextAction::SelectGroupNodes => group_nodes_required(graph, invocation),
        GeneratedGraphContextAction::ChooseGroupMode => {
            let GraphContextTarget::Group(identifier) = &invocation.target else {
                return ContextActionAvailability::disabled("group target is required");
            };
            if graph
                .groups
                .get(identifier)
                .is_some_and(|group| !group.node_ids.is_empty())
            {
                let Some(nodes) = item.context_group_nodes(invocation) else {
                    return ContextActionAvailability::disabled("group target is unavailable");
                };
                command_availability(
                    item,
                    &GraphCommand::SetNodeMode {
                        identifiers: nodes,
                        mode: GraphNodeMode::Always,
                    },
                )
            } else {
                ContextActionAvailability::disabled("group contains no nodes")
            }
        }
        GeneratedGraphContextAction::DisconnectSlot => {
            let GraphContextTarget::Slot { direction, slot } = &invocation.target else {
                return ContextActionAvailability::disabled("slot target is required");
            };
            command_availability(
                item,
                &GraphCommand::DisconnectSubgraphSlot {
                    direction: *direction,
                    slot: *slot,
                },
            )
        }
        GeneratedGraphContextAction::RenameSlot => {
            let GraphContextTarget::Slot { direction, slot } = &invocation.target else {
                return ContextActionAvailability::disabled("slot target is required");
            };
            let Some(name) = document
                .active_subgraph_definition()
                .ok()
                .and_then(|definition| match direction {
                    GraphSlotDirection::Input => definition.inputs.get(*slot),
                    GraphSlotDirection::Output => definition.outputs.get(*slot),
                })
                .map(|port| port.name.clone())
            else {
                return ContextActionAvailability::disabled("subgraph slot is unavailable");
            };
            command_availability(
                item,
                &GraphCommand::RenameSubgraphSlot {
                    direction: *direction,
                    slot: *slot,
                    name,
                },
            )
        }
        GeneratedGraphContextAction::DeleteSlot => {
            let GraphContextTarget::Slot { direction, slot } = &invocation.target else {
                return ContextActionAvailability::disabled("slot target is required");
            };
            command_availability(
                item,
                &GraphCommand::RemoveSubgraphSlot {
                    direction: *direction,
                    slot: *slot,
                },
            )
        }
        GeneratedGraphContextAction::ToggleRerouteType => {
            let GraphContextTarget::Reroute(identifier) = &invocation.target else {
                return ContextActionAvailability::disabled("reroute target is required");
            };
            let visible = item.context_reroute_type_visible(identifier, cx);
            checked_if_enabled(
                command_availability(
                    item,
                    &GraphCommand::SetRerouteTypeVisibility {
                        identifiers: BTreeSet::from([identifier.clone()]),
                        visible: !visible,
                    },
                ),
                visible,
            )
        }
        GeneratedGraphContextAction::ToggleDefaultRerouteType => {
            settings_context_action_availability(item, invocation, action, cx)
        }
    }
}

fn action_requires_context_input_workspace(action: GeneratedGraphContextAction) -> bool {
    matches!(
        action,
        GeneratedGraphContextAction::ChooseGroupFontSize
            | GeneratedGraphContextAction::ConvertToSubgraph
            | GeneratedGraphContextAction::FrameSelection
            | GeneratedGraphContextAction::PublishSubgraph
            | GeneratedGraphContextAction::RenameGroup
            | GeneratedGraphContextAction::RenameNode
            | GeneratedGraphContextAction::RenameSelection
            | GeneratedGraphContextAction::RenameSlot
    )
}

fn is_settings_owned_context_action(action: GeneratedGraphContextAction) -> bool {
    matches!(
        action,
        GeneratedGraphContextAction::ToggleNodesTwo
            | GeneratedGraphContextAction::ToggleDefaultRerouteType
    )
}

fn settings_context_action_availability(
    item: &GraphWorkspaceItem,
    invocation: &GraphContextInvocation,
    action: GeneratedGraphContextAction,
    cx: &App,
) -> ContextActionAvailability {
    if !cx.has_global::<SettingsStore>() {
        return ContextActionAvailability::disabled("Zed settings store is unavailable");
    }
    if item.context_settings_task.is_some() {
        return ContextActionAvailability::disabled("settings update is already in progress");
    }
    match action {
        GeneratedGraphContextAction::ToggleNodesTwo => {
            ContextActionAvailability::checked(item.native_node_renderer_enabled(cx))
        }
        GeneratedGraphContextAction::ToggleDefaultRerouteType => checked_if_enabled(
            reroute_required(invocation),
            item.context_default_reroute_type_visible(cx),
        ),
        _ => ContextActionAvailability::disabled("action is not settings-owned"),
    }
}

fn checked_if_enabled(
    mut availability: ContextActionAvailability,
    checked: bool,
) -> ContextActionAvailability {
    availability.checked = Some(checked);
    availability
}

fn node_has_advanced_widgets(node: &sim_runtime::GraphNode) -> bool {
    node.widgets.iter().any(|widget| {
        widget
            .unknown
            .get("advanced")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    })
}

fn context_has_group_at_pointer(
    item: &GraphWorkspaceItem,
    invocation: &GraphContextInvocation,
) -> bool {
    item.model
        .document()
        .and_then(|document| document.active_graph().ok())
        .is_some_and(|graph| {
            let graph_position = match &invocation.target {
                GraphContextTarget::Canvas { graph_position } => *graph_position,
                _ => graph.viewport.screen_to_graph(invocation.screen_position),
            };
            graph
                .groups
                .values()
                .any(|group| group.bounds.contains(graph_position))
        })
}

fn target_required(
    target: &GraphContextTarget,
    name: &str,
    predicate: impl FnOnce(&GraphContextTarget) -> bool,
) -> ContextActionAvailability {
    if predicate(target) {
        ContextActionAvailability::enabled()
    } else {
        ContextActionAvailability::disabled(format!("{name} target is required"))
    }
}

fn command_availability(
    item: &GraphWorkspaceItem,
    command: &GraphCommand,
) -> ContextActionAvailability {
    let Some(engine) = item.model.engine() else {
        return ContextActionAvailability::disabled("workflow is read-only");
    };
    match engine.validate_command(command) {
        Ok(()) => ContextActionAvailability::enabled(),
        Err(error) => ContextActionAvailability::disabled(error.to_string()),
    }
}

fn selection_required(invocation: &GraphContextInvocation) -> ContextActionAvailability {
    if invocation.selection.is_empty() {
        ContextActionAvailability::disabled("graph selection is empty")
    } else {
        ContextActionAvailability::enabled()
    }
}

fn group_nodes_required(
    graph: &sim_runtime::GraphLevel,
    invocation: &GraphContextInvocation,
) -> ContextActionAvailability {
    let GraphContextTarget::Group(identifier) = &invocation.target else {
        return ContextActionAvailability::disabled("group target is required");
    };
    if graph
        .groups
        .get(identifier)
        .is_some_and(|group| !group.node_ids.is_empty())
    {
        ContextActionAvailability::enabled()
    } else {
        ContextActionAvailability::disabled("group contains no nodes")
    }
}

fn reroute_required(invocation: &GraphContextInvocation) -> ContextActionAvailability {
    target_required(&invocation.target, "reroute", |target| {
        matches!(target, GraphContextTarget::Reroute(_))
    })
}

fn context_action_label(
    item: &GraphWorkspaceItem,
    invocation: &GraphContextInvocation,
    binding: &GraphContextActionBinding,
    fallback: &str,
) -> String {
    let graph = item
        .model
        .document()
        .and_then(|document| document.active_graph().ok());
    let node_scope = context_node_scope(binding, invocation);
    match binding.action {
        GeneratedGraphContextAction::CanvasHandMode => "Hand mode".to_owned(),
        GeneratedGraphContextAction::CanvasSelectMode => "Select mode".to_owned(),
        GeneratedGraphContextAction::ToggleNodeCollapse => {
            let collapsed = node_scope.iter().all(|identifier| {
                graph
                    .and_then(|graph| graph.nodes.get(identifier))
                    .is_some_and(|node| node.collapsed)
            });
            if collapsed {
                "Expand node"
            } else {
                "Collapse node"
            }
            .to_owned()
        }
        GeneratedGraphContextAction::ToggleNodePin => {
            let pinned = node_scope.iter().all(|identifier| {
                graph
                    .and_then(|graph| graph.nodes.get(identifier))
                    .is_some_and(|node| node.pinned)
            });
            if pinned { "Unpin node" } else { "Pin node" }.to_owned()
        }
        GeneratedGraphContextAction::ToggleNodeBypass => {
            let bypassed = node_scope.iter().all(|identifier| {
                graph
                    .and_then(|graph| graph.nodes.get(identifier))
                    .is_some_and(|node| node.mode == GraphNodeMode::Bypass)
            });
            if bypassed {
                "Remove bypass"
            } else {
                "Bypass node"
            }
            .to_owned()
        }
        GeneratedGraphContextAction::ToggleGroupPin => {
            let pinned = match &invocation.target {
                GraphContextTarget::Group(identifier) => graph
                    .and_then(|graph| graph.groups.get(identifier))
                    .is_some_and(|group| group.pinned),
                _ => false,
            };
            if pinned { "Unpin group" } else { "Pin group" }.to_owned()
        }
        _ => fallback.to_owned(),
    }
}

fn context_node_scope(
    binding: &GraphContextActionBinding,
    invocation: &GraphContextInvocation,
) -> BTreeSet<GraphIdentifier> {
    match binding.feature_id.as_str() {
        "COMFY-MENU-124" | "COMFY-MENU-125" | "COMFY-MENU-126" => match &invocation.target {
            GraphContextTarget::Node(identifier) => BTreeSet::from([identifier.clone()]),
            _ => BTreeSet::new(),
        },
        _ => invocation.node_scope(),
    }
}

impl GraphContextInvocation {
    fn node_scope(&self) -> BTreeSet<GraphIdentifier> {
        match &self.target {
            GraphContextTarget::Node(identifier) => {
                if self.selection.nodes.contains(identifier) {
                    self.selection.nodes.clone()
                } else {
                    BTreeSet::from([identifier.clone()])
                }
            }
            GraphContextTarget::Selection => self.selection.nodes.clone(),
            _ => BTreeSet::new(),
        }
    }
}

pub(crate) fn build_graph_context_menu(
    item: WeakEntity<GraphWorkspaceItem>,
    window: &mut Window,
    cx: &mut App,
) -> Option<Entity<ContextMenu>> {
    if let Err(error) =
        require_graph_context_infrastructure(GeneratedGraphContextInfrastructure::DropdownRenderer)
    {
        log::error!("native graph context renderer is unavailable: {error}");
        return None;
    }
    let (invocation, entries) = match item.read_with(cx, |item, _| {
        item.context_menu_state.as_ref().map(|state| {
            (
                state.invocation.clone(),
                graph_context_menu_entries(item, &state.invocation, cx),
            )
        })
    }) {
        Ok(Some((invocation, Ok(entries)))) => (invocation, entries),
        Ok(Some((_, Err(error)))) => {
            log::error!("canonical graph context registry is invalid: {error}");
            return None;
        }
        Ok(None) => return None,
        Err(error) => {
            log::error!(
                "native graph context target disappeared before menu construction: {error}"
            );
            return None;
        }
    };
    let dismiss_invocation = invocation.clone();
    let menu_item = item.clone();
    let menu = ContextMenu::build(window, cx, move |mut menu, _, _| {
        for entry in entries.clone() {
            menu = add_context_entry(menu, menu_item.clone(), invocation.clone(), entry);
        }
        menu
    });
    let dismiss_item = item;
    window
        .subscribe(&menu, cx, move |_, _: &DismissEvent, _, cx| {
            if let Err(error) = dismiss_item.update(cx, |item, cx| {
                let is_current = item
                    .context_menu_state
                    .as_ref()
                    .is_some_and(|state| state.invocation == dismiss_invocation);
                if is_current {
                    item.context_menu_state = None;
                    item.pending_pointer_context_target = None;
                    cx.notify();
                }
            }) {
                log::error!("graph disappeared while clearing dismissed context state: {error}");
            }
        })
        .detach();
    Some(menu)
}

fn add_context_entry(
    menu: ContextMenu,
    item: WeakEntity<GraphWorkspaceItem>,
    invocation: GraphContextInvocation,
    entry: GraphContextMenuEntry,
) -> ContextMenu {
    let visible_label = entry.disabled_reason.as_ref().map_or_else(
        || entry.label.clone(),
        |reason| format!("{} — {reason}", entry.label),
    );
    if entry.has_submenu() {
        if !entry.enabled {
            return menu.item(ContextMenuEntry::new(visible_label).disabled(true));
        }
        let label = visible_label;
        let binding = entry.binding;
        return menu.submenu(label, move |submenu, _, cx| {
            context_submenu(
                submenu,
                item.clone(),
                invocation.clone(),
                binding.clone(),
                cx,
            )
        });
    }
    let binding = entry.binding;
    let item_kind = binding.item_kind;
    let mut context_entry = ContextMenuEntry::new(visible_label)
        .disabled(!entry.enabled)
        .handler(dispatch_handler(
            item,
            invocation,
            binding,
            GraphContextDispatchInput::None,
        ));
    if let Some(checked) = entry.checked {
        context_entry = match item_kind {
            GeneratedMenuItemKind::RadioAction => context_entry.radio(IconPosition::Start, checked),
            GeneratedMenuItemKind::ToggleAction | GeneratedMenuItemKind::CheckboxAction => {
                context_entry.toggleable(IconPosition::Start, checked)
            }
            _ => context_entry,
        };
    }
    menu.item(context_entry)
}

fn action_has_submenu(action: GeneratedGraphContextAction) -> bool {
    matches!(
        action,
        GeneratedGraphContextAction::ChooseNodeShape
            | GeneratedGraphContextAction::ChooseNodeColor
            | GeneratedGraphContextAction::ChooseNodeMode
            | GeneratedGraphContextAction::AlignSelection
            | GeneratedGraphContextAction::DistributeSelection
            | GeneratedGraphContextAction::ChooseGroupNodeShape
            | GeneratedGraphContextAction::ChooseGroupColor
            | GeneratedGraphContextAction::ChooseGroupMode
            | GeneratedGraphContextAction::OpenNodeProperties
    )
}

fn context_submenu(
    mut menu: ContextMenu,
    item: WeakEntity<GraphWorkspaceItem>,
    invocation: GraphContextInvocation,
    binding: GraphContextActionBinding,
    cx: &App,
) -> ContextMenu {
    let action = binding.action;
    if action == GeneratedGraphContextAction::OpenNodeProperties {
        return node_properties_context_submenu(menu, item, invocation, binding, cx);
    }
    let options: Vec<(&'static str, GraphContextDispatchInput)> = match action {
        GeneratedGraphContextAction::ChooseNodeShape
        | GeneratedGraphContextAction::ChooseGroupNodeShape => vec![
            (
                "Default",
                GraphContextDispatchInput::Shape(GraphVisualShape::Default),
            ),
            (
                "Box",
                GraphContextDispatchInput::Shape(GraphVisualShape::Box),
            ),
            (
                "Round",
                GraphContextDispatchInput::Shape(GraphVisualShape::Round),
            ),
            (
                "Card",
                GraphContextDispatchInput::Shape(GraphVisualShape::Card),
            ),
        ],
        GeneratedGraphContextAction::ChooseNodeColor
        | GeneratedGraphContextAction::ChooseGroupColor => {
            let mut options = vec![("No color", GraphContextDispatchInput::PaletteColor(None))];
            options.extend(GraphPaletteColor::ALL.map(|color| {
                (
                    color.label(),
                    GraphContextDispatchInput::PaletteColor(Some(color)),
                )
            }));
            options
        }
        GeneratedGraphContextAction::ChooseNodeMode => vec![
            (
                "Always",
                GraphContextDispatchInput::NodeMode(GraphNodeMode::Always),
            ),
            (
                "On event",
                GraphContextDispatchInput::NodeMode(GraphNodeMode::OnEvent),
            ),
            (
                "Never",
                GraphContextDispatchInput::NodeMode(GraphNodeMode::Never),
            ),
            (
                "On trigger",
                GraphContextDispatchInput::NodeMode(GraphNodeMode::OnTrigger),
            ),
            (
                "Bypass",
                GraphContextDispatchInput::NodeMode(GraphNodeMode::Bypass),
            ),
        ],
        GeneratedGraphContextAction::ChooseGroupMode => {
            let always = (
                "Set group nodes to Always",
                GraphContextDispatchInput::NodeMode(GraphNodeMode::Always),
            );
            let never = (
                "Set group nodes to Never",
                GraphContextDispatchInput::NodeMode(GraphNodeMode::Never),
            );
            let bypass = (
                "Bypass group nodes",
                GraphContextDispatchInput::NodeMode(GraphNodeMode::Bypass),
            );
            vec![always, never, bypass]
        }
        GeneratedGraphContextAction::AlignSelection => vec![
            (
                "Left",
                GraphContextDispatchInput::Layout(LayoutOperation::AlignLeft),
            ),
            (
                "Right",
                GraphContextDispatchInput::Layout(LayoutOperation::AlignRight),
            ),
            (
                "Top",
                GraphContextDispatchInput::Layout(LayoutOperation::AlignTop),
            ),
            (
                "Bottom",
                GraphContextDispatchInput::Layout(LayoutOperation::AlignBottom),
            ),
            (
                "Horizontal centers",
                GraphContextDispatchInput::Layout(LayoutOperation::AlignHorizontalCenters),
            ),
            (
                "Vertical centers",
                GraphContextDispatchInput::Layout(LayoutOperation::AlignVerticalCenters),
            ),
        ],
        GeneratedGraphContextAction::DistributeSelection => vec![
            (
                "Horizontally",
                GraphContextDispatchInput::Layout(LayoutOperation::DistributeHorizontally),
            ),
            (
                "Vertically",
                GraphContextDispatchInput::Layout(LayoutOperation::DistributeVertically),
            ),
            (
                "Grid",
                GraphContextDispatchInput::Layout(LayoutOperation::ArrangeGrid),
            ),
        ],
        _ => Vec::new(),
    };
    for (label, input) in options {
        let checked = context_submenu_checked(&item, &invocation, action, &input, cx);
        let mut entry = ContextMenuEntry::new(label).handler(dispatch_handler(
            item.clone(),
            invocation.clone(),
            binding.clone(),
            input,
        ));
        if let Some(checked) = checked {
            entry = entry.radio(IconPosition::Start, checked);
        }
        menu = menu.item(entry);
    }
    menu
}

fn node_properties_context_submenu(
    mut menu: ContextMenu,
    item: WeakEntity<GraphWorkspaceItem>,
    invocation: GraphContextInvocation,
    binding: GraphContextActionBinding,
    cx: &App,
) -> ContextMenu {
    let descriptors = item
        .read_with(cx, |item, _| {
            let GraphContextTarget::Node(identifier) = &invocation.target else {
                return Err("node target is required".to_owned());
            };
            let node = item
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .and_then(|graph| graph.nodes.get(identifier))
                .ok_or_else(|| "node target is unavailable".to_owned())?;
            graph_node_property_descriptors(node).map_err(|error| error.to_string())
        })
        .map_err(|error| error.to_string())
        .and_then(|descriptors| descriptors);
    let descriptors = match descriptors {
        Ok(descriptors) => descriptors,
        Err(error) => return menu.item(ContextMenuEntry::new(error).disabled(true)),
    };
    for descriptor in descriptors {
        let value_label = graph_node_property_value_label(&descriptor.value);
        let label = format!("{}: {value_label}", descriptor.label);
        match descriptor.kind {
            GraphNodePropertyKind::Boolean => {
                let checked = descriptor.value.as_bool().unwrap_or(false);
                menu = menu.item(
                    ContextMenuEntry::new(label)
                        .toggleable(IconPosition::Start, checked)
                        .handler(dispatch_handler(
                            item.clone(),
                            invocation.clone(),
                            binding.clone(),
                            GraphContextDispatchInput::NodeProperty {
                                key: descriptor.key,
                                value: Some(Value::Bool(!checked)),
                            },
                        )),
                );
            }
            GraphNodePropertyKind::Choice { choices } => {
                let key = descriptor.key;
                let selected = descriptor.value;
                let item = item.clone();
                let invocation = invocation.clone();
                let binding = binding.clone();
                menu = menu.submenu(label, move |mut submenu, _, _| {
                    for choice in choices.clone() {
                        let entry = ContextMenuEntry::new(choice.label)
                            .radio(IconPosition::Start, choice.value == selected)
                            .handler(dispatch_handler(
                                item.clone(),
                                invocation.clone(),
                                binding.clone(),
                                GraphContextDispatchInput::NodeProperty {
                                    key: key.clone(),
                                    value: Some(choice.value),
                                },
                            ));
                        submenu = submenu.item(entry);
                    }
                    submenu
                });
            }
            GraphNodePropertyKind::Number
            | GraphNodePropertyKind::Text
            | GraphNodePropertyKind::Json => {
                menu = menu.item(ContextMenuEntry::new(label).handler(dispatch_handler(
                    item.clone(),
                    invocation.clone(),
                    binding.clone(),
                    GraphContextDispatchInput::NodeProperty {
                        key: descriptor.key,
                        value: None,
                    },
                )));
            }
        }
    }
    menu
}

fn context_submenu_checked(
    item: &WeakEntity<GraphWorkspaceItem>,
    invocation: &GraphContextInvocation,
    action: GeneratedGraphContextAction,
    input: &GraphContextDispatchInput,
    cx: &App,
) -> Option<bool> {
    item.read_with(cx, |item, _| {
        let graph = item.model.document()?.active_graph().ok()?;
        let nodes = match action {
            GeneratedGraphContextAction::ChooseGroupNodeShape
            | GeneratedGraphContextAction::ChooseGroupMode => {
                item.context_group_nodes(invocation)?
            }
            GeneratedGraphContextAction::ChooseNodeShape
            | GeneratedGraphContextAction::ChooseNodeColor
            | GeneratedGraphContextAction::ChooseNodeMode => invocation.node_scope(),
            GeneratedGraphContextAction::ChooseGroupColor => BTreeSet::new(),
            _ => return None,
        };
        match input {
            GraphContextDispatchInput::Shape(shape) => Some(nodes.iter().all(|identifier| {
                graph
                    .nodes
                    .get(identifier)
                    .and_then(|node| node.source_fields.get("shape"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("default")
                    == shape.source_name()
            })),
            GraphContextDispatchInput::PaletteColor(color)
                if action == GeneratedGraphContextAction::ChooseNodeColor =>
            {
                Some(nodes.iter().all(|identifier| {
                    graph.nodes.get(identifier).is_some_and(|node| {
                        node.color.as_deref() == color.map(GraphPaletteColor::node_header)
                    })
                }))
            }
            GraphContextDispatchInput::PaletteColor(color)
                if action == GeneratedGraphContextAction::ChooseGroupColor =>
            {
                let identifier = invocation.group_identifier()?;
                Some(graph.groups.get(&identifier).is_some_and(|group| {
                    group.color.as_deref() == color.map(GraphPaletteColor::group)
                }))
            }
            GraphContextDispatchInput::NodeMode(mode) => Some(nodes.iter().all(|identifier| {
                graph
                    .nodes
                    .get(identifier)
                    .is_some_and(|node| node.mode == *mode)
            })),
            _ => None,
        }
    })
    .ok()
    .flatten()
}

fn dispatch_handler(
    item: WeakEntity<GraphWorkspaceItem>,
    invocation: GraphContextInvocation,
    binding: GraphContextActionBinding,
    input: GraphContextDispatchInput,
) -> impl Fn(&mut Window, &mut App) + 'static {
    move |window, cx| {
        if let Err(error) = item.update(cx, |item, cx| {
            item.dispatch_context_action(
                binding.clone(),
                input.clone(),
                invocation.clone(),
                window,
                cx,
            );
        }) {
            log::error!("native graph context target disappeared during dispatch: {error}");
        }
    }
}

fn restore_focus(return_focus: Option<FocusHandle>, window: &mut Window, cx: &mut App) {
    if let Some(return_focus) = return_focus {
        window.focus(&return_focus, cx);
        window.on_next_frame(move |window, cx| window.focus(&return_focus, cx));
    }
}

impl GraphWorkspaceItem {
    pub(crate) fn canvas_is_locked(&self) -> bool {
        self.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .is_some_and(|graph| graph.viewport.locked)
    }

    pub(crate) fn begin_canvas_pan(&mut self, position: GraphPoint, cx: &mut Context<Self>) {
        let viewport = self
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.clone());
        if let Some(viewport) = viewport {
            self.canvas_pan_anchor = Some((position, viewport));
            self.model.announcement = Some("Canvas hand pan started".to_owned());
            cx.notify();
        }
    }

    pub(crate) fn update_canvas_pan(&mut self, position: GraphPoint, cx: &mut Context<Self>) {
        let Some((anchor, initial_viewport)) = self.canvas_pan_anchor.clone() else {
            return;
        };
        let Some(selection) = self.model.selection().cloned() else {
            return;
        };
        let mut viewport = initial_viewport;
        viewport.offset = viewport.offset.translated(GraphPoint {
            x: position.x - anchor.x,
            y: position.y - anchor.y,
        });
        match self
            .model
            .replace_ephemeral_graph_state(selection, viewport)
        {
            Ok(()) => cx.notify(),
            Err(error) => {
                self.model.report_error(error);
                self.canvas_pan_anchor = None;
                cx.notify();
            }
        }
    }

    pub(crate) fn finish_canvas_pan(&mut self, cx: &mut Context<Self>) {
        let Some((_, initial_viewport)) = self.canvas_pan_anchor.take() else {
            return;
        };
        let Some((selection, final_viewport)) = self
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| (graph.selection.clone(), graph.viewport.clone()))
        else {
            return;
        };
        if final_viewport == initial_viewport {
            return;
        }
        if let Err(error) = self
            .model
            .replace_ephemeral_graph_state(selection, initial_viewport)
        {
            self.model.report_error(error);
            cx.notify();
            return;
        }
        if self.apply_graph_command(
            GraphCommand::SetViewport {
                viewport: final_viewport,
            },
            cx,
        ) {
            self.model.announcement = Some("Canvas hand pan completed".to_owned());
            cx.notify();
        }
    }

    pub(crate) fn open_graph_context_menu(
        &mut self,
        target: GraphContextTarget,
        screen_position: GraphPoint,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.capture_graph_context_menu(target, screen_position, window, cx) {
            return;
        }
        let handle = self.context_menu_handle.clone();
        window.on_next_frame(move |window, cx| {
            handle.show_at(
                gpui::point(px(screen_position.x), px(screen_position.y)),
                window,
                cx,
            );
        });
    }

    pub(crate) fn stage_pointer_graph_context_target(&mut self, target: GraphContextTarget) {
        self.pending_pointer_context_target = Some(target);
    }

    pub(crate) fn capture_pointer_graph_context_menu(
        &mut self,
        screen_position: GraphPoint,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if let Some(target) = self.pending_pointer_context_target.take() {
            self.capture_graph_context_menu(target, screen_position, window, cx)
        } else {
            self.context_menu_state.is_some()
        }
    }

    fn capture_graph_context_menu(
        &mut self,
        target: GraphContextTarget,
        screen_position: GraphPoint,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if let Err(error) = self.prepare_context_target(&target) {
            self.model.report_error(error);
            cx.notify();
            return false;
        }
        let Some(document) = self.model.document() else {
            self.model.report_error("workflow is open read-only");
            cx.notify();
            return false;
        };
        let Ok(graph) = document.active_graph() else {
            self.model.report_error("active graph is unavailable");
            cx.notify();
            return false;
        };
        let content_revision = match document.to_workflow_bytes() {
            Ok(bytes) => ContentRevision::from_bytes(&bytes),
            Err(error) => {
                self.model.report_error(format!(
                    "cannot open graph context menu for an invalid workflow: {error}"
                ));
                cx.notify();
                return false;
            }
        };
        let invocation = GraphContextInvocation {
            document_identity: document.document_identity,
            content_revision,
            navigation: document.navigation.clone(),
            selection: graph.selection.clone(),
            target,
            screen_position,
        };
        self.context_menu_state = Some(GraphContextMenuState {
            invocation,
            return_focus: window.focused(cx),
        });
        self.model.announcement = Some("Opened native graph context menu".to_owned());
        cx.notify();
        true
    }

    fn prepare_context_target(&mut self, target: &GraphContextTarget) -> Result<(), String> {
        let document = self
            .model
            .document()
            .ok_or_else(|| "workflow is open read-only".to_owned())?;
        let graph = document.active_graph().map_err(|error| error.to_string())?;
        let mut selection = graph.selection.clone();
        match target {
            GraphContextTarget::Node(identifier) if !selection.nodes.contains(identifier) => {
                if !graph.nodes.contains_key(identifier) {
                    return Err(format!("node {} is stale", identifier.text()));
                }
                selection = GraphSelection {
                    nodes: BTreeSet::from([identifier.clone()]),
                    ..GraphSelection::default()
                };
            }
            GraphContextTarget::Group(identifier) if !selection.groups.contains(identifier) => {
                if !graph.groups.contains_key(identifier) {
                    return Err(format!("group {} is stale", identifier.text()));
                }
                let selected_nodes = selection.nodes;
                selection = GraphSelection {
                    nodes: selected_nodes,
                    groups: BTreeSet::from([identifier.clone()]),
                    ..GraphSelection::default()
                };
            }
            GraphContextTarget::Reroute(identifier) if !selection.reroutes.contains(identifier) => {
                if !graph.reroutes.contains_key(identifier) {
                    return Err(format!("reroute {} is stale", identifier.text()));
                }
                selection = GraphSelection {
                    reroutes: BTreeSet::from([identifier.clone()]),
                    ..GraphSelection::default()
                };
            }
            _ => {}
        }
        if selection != graph.selection {
            let viewport = graph.viewport.clone();
            self.model
                .replace_ephemeral_graph_state(selection, viewport)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    pub(crate) fn validate_context_invocation(
        &self,
        invocation: &GraphContextInvocation,
        require_selection_match: bool,
    ) -> Result<(), String> {
        let document = self
            .model
            .document()
            .ok_or_else(|| "workflow is open read-only".to_owned())?;
        if document.document_identity != invocation.document_identity
            || document.navigation != invocation.navigation
        {
            return Err("context target belongs to a stale graph view".to_owned());
        }
        let current_revision = document
            .to_workflow_bytes()
            .map(|bytes| ContentRevision::from_bytes(&bytes))
            .map_err(|error| format!("current workflow cannot be validated: {error}"))?;
        if current_revision != invocation.content_revision {
            return Err("workflow content changed while the context action was open".to_owned());
        }
        let graph = document.active_graph().map_err(|error| error.to_string())?;
        if require_selection_match && graph.selection != invocation.selection {
            return Err("graph selection changed while the context action was open".to_owned());
        }
        match &invocation.target {
            GraphContextTarget::Canvas { .. } | GraphContextTarget::Selection => {}
            GraphContextTarget::Node(identifier) => {
                if !graph.nodes.contains_key(identifier) {
                    return Err(format!("node {} is stale", identifier.text()));
                }
            }
            GraphContextTarget::Group(identifier) => {
                if !graph.groups.contains_key(identifier) {
                    return Err(format!("group {} is stale", identifier.text()));
                }
            }
            GraphContextTarget::Reroute(identifier) => {
                if !graph.reroutes.contains_key(identifier) {
                    return Err(format!("reroute {} is stale", identifier.text()));
                }
            }
            GraphContextTarget::Slot { direction, slot } => {
                let definition = document
                    .active_subgraph_definition()
                    .map_err(|error| error.to_string())?;
                let exists = match direction {
                    GraphSlotDirection::Input => definition.inputs.get(*slot).is_some(),
                    GraphSlotDirection::Output => definition.outputs.get(*slot).is_some(),
                };
                if !exists {
                    return Err(format!("subgraph {direction:?} slot {slot} is stale"));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn dispatch_context_action(
        &mut self,
        binding: GraphContextActionBinding,
        input: GraphContextDispatchInput,
        invocation: GraphContextInvocation,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> GraphContextDispatchOutcome {
        if let Err(error) = binding.revalidate() {
            return self.reject_context_action(error, cx);
        }
        if let Err(error) = self.validate_context_invocation(&invocation, true) {
            return self.reject_context_action(error, cx);
        }
        let availability = context_feature_availability(self, &invocation, &binding, cx);
        if !availability.enabled {
            return self.reject_context_action(
                availability
                    .reason
                    .unwrap_or_else(|| "context action is unavailable".to_owned()),
                cx,
            );
        }
        let action = binding.action;
        let surface = binding.surface;
        let node_scope = context_node_scope(&binding, &invocation);
        let outcome = match action {
            GeneratedGraphContextAction::CanvasSelectMode => {
                if !self.canvas_is_locked() {
                    self.model.announcement = Some("Canvas is already in select mode".to_owned());
                    cx.notify();
                    return GraphContextDispatchOutcome::Executed;
                }
                self.execute_catalog_action(CatalogGraphAction::Unlock, GraphActionInput::None, cx)
                    .then_some(GraphContextDispatchOutcome::Executed)
                    .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::CanvasHandMode => {
                if self.canvas_is_locked() {
                    self.model.announcement = Some("Canvas is already in hand mode".to_owned());
                    cx.notify();
                    return GraphContextDispatchOutcome::Executed;
                }
                self.execute_catalog_action(CatalogGraphAction::Lock, GraphActionInput::None, cx)
                    .then_some(GraphContextDispatchOutcome::Executed)
                    .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ToggleNodesTwo => self
                .execute_catalog_action(
                    CatalogGraphAction::ToggleVueNodes,
                    GraphActionInput::None,
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model()),
            GeneratedGraphContextAction::AddGroup => {
                let GraphContextTarget::Canvas { graph_position } = invocation.target else {
                    return self.reject_context_action("canvas target is required", cx);
                };
                let Some(document) = self.model.document() else {
                    return self.reject_context_action("workflow is open read-only", cx);
                };
                let identifier = document.next_available_identifier();
                self.apply_graph_command(
                    GraphCommand::CreateGroup {
                        group: GraphGroup {
                            identifier,
                            title: "Group".to_owned(),
                            bounds: GraphRect {
                                origin: graph_position,
                                size: GraphSize {
                                    width: 140.0,
                                    height: 80.0,
                                },
                            },
                            node_ids: BTreeSet::new(),
                            collapsed: false,
                            pinned: false,
                            color: Some("#3f789e".to_owned()),
                            source_fields: Map::from_iter([(
                                "font_size".to_owned(),
                                serde_json::Value::from(20.0),
                            )]),
                        },
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::Paste => {
                let GraphContextTarget::Canvas { graph_position } = invocation.target else {
                    return self.reject_context_action("canvas target is required", cx);
                };
                self.paste_from_clipboard_at(false, graph_position, cx)
                    .then_some(GraphContextDispatchOutcome::Executed)
                    .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::AdjustNodeSize => self
                .execute_catalog_action(
                    CatalogGraphAction::Resize,
                    GraphActionInput::NodeIdentifiers(node_scope),
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model()),
            GeneratedGraphContextAction::ToggleNodeCollapse => self
                .apply_graph_command(
                    GraphCommand::ToggleNodes {
                        identifiers: node_scope,
                        toggle: NodeToggle::Collapse,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model()),
            GeneratedGraphContextAction::ToggleNodePin => self
                .apply_graph_command(
                    GraphCommand::ToggleNodes {
                        identifiers: node_scope,
                        toggle: NodeToggle::Pin,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model()),
            GeneratedGraphContextAction::ToggleNodeBypass => self
                .apply_graph_command(
                    GraphCommand::ToggleNodes {
                        identifiers: node_scope,
                        toggle: NodeToggle::Bypass,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model()),
            GeneratedGraphContextAction::ChooseNodeShape => {
                let GraphContextDispatchInput::Shape(shape) = input else {
                    return self.reject_context_action("node shape input is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::SetNodeShape {
                        identifiers: node_scope,
                        shape,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ChooseNodeColor => {
                let GraphContextDispatchInput::PaletteColor(color) = input else {
                    return self.reject_context_action("node color input is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::SetNodePalette {
                        identifiers: node_scope,
                        color,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ChooseNodeMode => {
                let GraphContextDispatchInput::NodeMode(mode) = input else {
                    return self.reject_context_action("node mode input is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::SetNodeMode {
                        identifiers: node_scope,
                        mode,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ToggleAdvancedWidgets => {
                let visible = !node_scope.iter().all(|identifier| {
                    self.model
                        .document()
                        .and_then(|document| document.active_graph().ok())
                        .and_then(|graph| graph.nodes.get(identifier))
                        .and_then(|node| node.source_fields.get("show_advanced"))
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                });
                self.apply_graph_command(
                    GraphCommand::SetNodeAdvancedVisibility {
                        identifiers: node_scope,
                        visible,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::CopySelection => self
                .copy_to_clipboard(cx)
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model()),
            GeneratedGraphContextAction::DuplicateSelection => self
                .apply_graph_command(
                    GraphCommand::DuplicateSelection {
                        selection: invocation.selection,
                        offset: GraphPoint { x: 24.0, y: 24.0 },
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model()),
            GeneratedGraphContextAction::RenameNode => {
                let GraphContextTarget::Node(identifier) = &invocation.target else {
                    return self.reject_context_action("node target is required", cx);
                };
                let identifier = identifier.clone();
                self.begin_context_input(
                    invocation,
                    GraphContextInputOperation::RenameNode(identifier),
                    window,
                    cx,
                )
            }
            GeneratedGraphContextAction::RenameSelection => {
                let operation = match (
                    invocation.selection.nodes.len(),
                    invocation.selection.groups.len(),
                ) {
                    (1, 0) => {
                        let Some(identifier) = invocation.selection.nodes.iter().next().cloned()
                        else {
                            return self.reject_context_action("rename target is unavailable", cx);
                        };
                        GraphContextInputOperation::RenameNode(identifier)
                    }
                    (0, 1) => {
                        let Some(identifier) = invocation.selection.groups.iter().next().cloned()
                        else {
                            return self.reject_context_action("rename target is unavailable", cx);
                        };
                        GraphContextInputOperation::RenameGroup(identifier)
                    }
                    _ => GraphContextInputOperation::BatchRenameSelection {
                        nodes: invocation.selection.nodes.iter().cloned().collect(),
                        groups: invocation.selection.groups.iter().cloned().collect(),
                    },
                };
                self.begin_context_input(invocation, operation, window, cx)
            }
            GeneratedGraphContextAction::ConvertToSubgraph => self.begin_context_input(
                invocation,
                GraphContextInputOperation::ConvertToSubgraph,
                window,
                cx,
            ),
            GeneratedGraphContextAction::PublishSubgraph => self.begin_context_input(
                invocation,
                GraphContextInputOperation::PublishSubgraph,
                window,
                cx,
            ),
            GeneratedGraphContextAction::UnpackSubgraph => {
                let Some(graph) = self
                    .model
                    .document()
                    .and_then(|document| document.active_graph().ok())
                else {
                    return self.reject_context_action("active graph is unavailable", cx);
                };
                let commands = invocation
                    .selection
                    .nodes
                    .iter()
                    .filter(|identifier| {
                        graph
                            .nodes
                            .get(*identifier)
                            .is_some_and(|node| node.subgraph_definition.is_some())
                    })
                    .cloned()
                    .map(|instance_identifier| GraphCommand::UnpackSubgraph {
                        instance_identifier,
                    })
                    .collect();
                self.apply_graph_command(GraphCommand::Batch { commands }, cx)
                    .then_some(GraphContextDispatchOutcome::Executed)
                    .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::FrameSelection => self.begin_context_input(
                invocation,
                GraphContextInputOperation::FrameSelection,
                window,
                cx,
            ),
            GeneratedGraphContextAction::AddGroupForSelection => {
                let command = match self.group_selection_command(
                    &invocation,
                    "Group".to_owned(),
                    self.group_selected_nodes_padding(cx),
                ) {
                    Ok(command) => command,
                    Err(error) => return self.reject_context_action(error, cx),
                };
                self.apply_graph_command(command, cx)
                    .then_some(GraphContextDispatchOutcome::Executed)
                    .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::AlignSelection
            | GeneratedGraphContextAction::DistributeSelection => {
                let GraphContextDispatchInput::Layout(operation) = input else {
                    return self.reject_context_action("layout input is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::LayoutSelection {
                        operation,
                        spacing: 24.0,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::DeleteSelection => {
                let command = GraphCommand::RemoveItems {
                    selection: if surface == GeneratedGraphContextSurface::Node {
                        let GraphContextTarget::Node(identifier) = &invocation.target else {
                            return self.reject_context_action("node target is required", cx);
                        };
                        GraphSelection {
                            nodes: BTreeSet::from([identifier.clone()]),
                            ..GraphSelection::default()
                        }
                    } else {
                        invocation.selection.clone()
                    },
                };
                self.begin_context_confirmation(
                    invocation,
                    "Delete the selected graph items?",
                    "This graph mutation can be restored with Undo.",
                    "Delete",
                    command,
                    window,
                    cx,
                )
            }
            GeneratedGraphContextAction::FitGroup => {
                let Some(identifier) = invocation.group_identifier() else {
                    return self.reject_context_action("group target is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::FitGroup {
                        identifier,
                        padding: self.group_selected_nodes_padding(cx),
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ChooseGroupNodeShape => {
                let GraphContextDispatchInput::Shape(shape) = input else {
                    return self.reject_context_action("group shape input is required", cx);
                };
                let Some(nodes) = self.context_group_nodes(&invocation) else {
                    return self.reject_context_action("group target is unavailable", cx);
                };
                self.apply_graph_command(
                    GraphCommand::SetNodeShape {
                        identifiers: nodes,
                        shape,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ChooseGroupMode => {
                let GraphContextDispatchInput::NodeMode(mode) = input else {
                    return self.reject_context_action("group mode input is required", cx);
                };
                let Some(nodes) = self.context_group_nodes(&invocation) else {
                    return self.reject_context_action("group target is unavailable", cx);
                };
                self.apply_graph_command(
                    GraphCommand::SetNodeMode {
                        identifiers: nodes,
                        mode,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ChooseGroupColor => {
                let GraphContextDispatchInput::PaletteColor(color) = input else {
                    return self.reject_context_action("group color input is required", cx);
                };
                let Some(identifier) = invocation.group_identifier() else {
                    return self.reject_context_action("group target is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::SetGroupColor {
                        identifier,
                        color: color.map(|color| color.group().to_owned()),
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ChooseGroupFontSize => {
                let Some(identifier) = invocation.group_identifier() else {
                    return self.reject_context_action("group target is required", cx);
                };
                match input {
                    GraphContextDispatchInput::None => self.begin_context_input(
                        invocation,
                        GraphContextInputOperation::SetGroupFontSize(identifier),
                        window,
                        cx,
                    ),
                    GraphContextDispatchInput::GroupFontSize(font_size) => self
                        .apply_graph_command(
                            GraphCommand::SetGroupFontSize {
                                identifier,
                                font_size,
                            },
                            cx,
                        )
                        .then_some(GraphContextDispatchOutcome::Executed)
                        .unwrap_or_else(|| self.context_rejection_from_model()),
                    _ => self.reject_context_action("numeric group font size is required", cx),
                }
            }
            GeneratedGraphContextAction::ToggleGroupPin => {
                let Some(identifier) = invocation.group_identifier() else {
                    return self.reject_context_action("group target is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::ToggleGroups {
                        identifiers: BTreeSet::from([identifier]),
                        toggle: GroupToggle::Pin,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::RenameGroup => {
                let Some(identifier) = invocation.group_identifier() else {
                    return self.reject_context_action("group target is required", cx);
                };
                self.begin_context_input(
                    invocation,
                    GraphContextInputOperation::RenameGroup(identifier),
                    window,
                    cx,
                )
            }
            GeneratedGraphContextAction::DeleteGroup => {
                let Some(identifier) = invocation.group_identifier() else {
                    return self.reject_context_action("group target is required", cx);
                };
                self.begin_context_confirmation(
                    invocation,
                    "Remove this group?",
                    "The nodes remain in the workflow and the group can be restored with Undo.",
                    "Remove Group",
                    GraphCommand::Ungroup { identifier },
                    window,
                    cx,
                )
            }
            GeneratedGraphContextAction::AddSelectionToGroup => {
                let Some(identifier) = invocation.group_identifier() else {
                    return self.reject_context_action("group target is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::AddNodesToGroup {
                        identifier,
                        nodes: invocation.selection.nodes,
                        padding: self.group_selected_nodes_padding(cx),
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::SelectGroupNodes => {
                let Some(nodes) = self.context_group_nodes(&invocation) else {
                    return self.reject_context_action("group target is unavailable", cx);
                };
                let Some(document) = self.model.document() else {
                    return self.reject_context_action("workflow is open read-only", cx);
                };
                let Ok(graph) = document.active_graph() else {
                    return self.reject_context_action("active graph is unavailable", cx);
                };
                let result = self.model.replace_ephemeral_graph_state(
                    GraphSelection {
                        nodes,
                        ..GraphSelection::default()
                    },
                    graph.viewport.clone(),
                );
                match result {
                    Ok(()) => {
                        self.model.announcement = Some("Selected the group's nodes".to_owned());
                        cx.notify();
                        GraphContextDispatchOutcome::Executed
                    }
                    Err(error) => self.reject_context_action(error, cx),
                }
            }
            GeneratedGraphContextAction::DisconnectSlot => {
                let Some((direction, slot)) = invocation.slot_target() else {
                    return self.reject_context_action("slot target is required", cx);
                };
                self.apply_graph_command(
                    GraphCommand::DisconnectSubgraphSlot { direction, slot },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::RenameSlot => {
                let Some((direction, slot)) = invocation.slot_target() else {
                    return self.reject_context_action("slot target is required", cx);
                };
                self.begin_context_input(
                    invocation,
                    GraphContextInputOperation::RenameSlot { direction, slot },
                    window,
                    cx,
                )
            }
            GeneratedGraphContextAction::DeleteSlot => {
                let Some((direction, slot)) = invocation.slot_target() else {
                    return self.reject_context_action("slot target is required", cx);
                };
                self.begin_context_confirmation(
                    invocation,
                    "Remove this graph slot?",
                    "Connected links are removed in the same undoable transaction.",
                    "Remove Slot",
                    GraphCommand::RemoveSubgraphSlot { direction, slot },
                    window,
                    cx,
                )
            }
            GeneratedGraphContextAction::ToggleRerouteType => {
                let GraphContextTarget::Reroute(identifier) = &invocation.target else {
                    return self.reject_context_action("reroute target is required", cx);
                };
                let visible = !self.context_reroute_type_visible(identifier, cx);
                self.apply_graph_command(
                    GraphCommand::SetRerouteTypeVisibility {
                        identifiers: BTreeSet::from([identifier.clone()]),
                        visible,
                    },
                    cx,
                )
                .then_some(GraphContextDispatchOutcome::Executed)
                .unwrap_or_else(|| self.context_rejection_from_model())
            }
            GeneratedGraphContextAction::ToggleDefaultRerouteType => {
                let visible = !self.context_default_reroute_type_visible(cx);
                if !cx.has_global::<SettingsStore>() {
                    return self.reject_context_action("Zed settings store is unavailable", cx);
                }
                let completion = settings::update_settings_file_with_completion(
                    <dyn Fs>::global(cx),
                    cx,
                    move |settings, _| {
                        let sim_runtime = settings.sim_runtime.get_or_insert_default();
                        sim_runtime.show_reroute_types = Some(visible);
                    },
                );
                self.context_settings_task = Some(cx.spawn(async move |this, cx| {
                    let result = completion.await;
                    if let Err(error) = this.update(cx, |this, cx| {
                        this.context_settings_task = None;
                        match result {
                            Ok(Ok(())) => {
                                this.model.announcement = Some(format!(
                                    "Default reroute type labels {}",
                                    if visible { "enabled" } else { "disabled" }
                                ));
                            }
                            Ok(Err(error)) => this.model.report_error(format!(
                                "failed to update default reroute type labels: {error}"
                            )),
                            Err(error) => this.model.report_error(format!(
                                "default reroute type settings update was cancelled: {error}"
                            )),
                        }
                        cx.notify();
                    }) {
                        log::error!(
                            "graph disappeared while completing reroute type settings update: {error}"
                        );
                    }
                }));
                self.model.announcement = Some("Updating default reroute type labels".to_owned());
                cx.notify();
                GraphContextDispatchOutcome::Executed
            }
            GeneratedGraphContextAction::OpenNodeProperties => {
                let GraphContextTarget::Node(identifier) = &invocation.target else {
                    return self.reject_context_action("node target is required", cx);
                };
                let identifier = identifier.clone();
                let GraphContextDispatchInput::NodeProperty { key, value } = input else {
                    return self
                        .reject_context_action("a typed node property selection is required", cx);
                };
                let Some(node) = self
                    .model
                    .document()
                    .and_then(|document| document.active_graph().ok())
                    .and_then(|graph| graph.nodes.get(&identifier))
                    .cloned()
                else {
                    return self.reject_context_action("node target is unavailable", cx);
                };
                if let Some(value) = value {
                    let properties = match graph_node_properties_with_value(&node, &key, value) {
                        Ok(properties) => properties,
                        Err(error) => {
                            return self.reject_context_action(error.to_string(), cx);
                        }
                    };
                    self.apply_graph_command(
                        GraphCommand::SetNodeProperties {
                            identifier,
                            properties,
                        },
                        cx,
                    )
                    .then_some(GraphContextDispatchOutcome::Executed)
                    .unwrap_or_else(|| self.context_rejection_from_model())
                } else {
                    self.begin_context_input(
                        invocation,
                        GraphContextInputOperation::EditNodeProperty { identifier, key },
                        window,
                        cx,
                    )
                }
            }
            GeneratedGraphContextAction::OpenNodePropertiesPanel => {
                let GraphContextTarget::Node(identifier) = &invocation.target else {
                    return self.reject_context_action("node target is required", cx);
                };
                let graph = cx.entity();
                let Some(workspace) = self.workspace().upgrade() else {
                    return self
                        .reject_context_action("workspace properties panel is unavailable", cx);
                };
                match workspace.update(cx, |workspace, workspace_cx| {
                    crate::open_for_graph_node(
                        workspace,
                        graph,
                        identifier.clone(),
                        window,
                        workspace_cx,
                    )
                }) {
                    Ok(()) => GraphContextDispatchOutcome::Executed,
                    Err(error) => self.reject_context_action(error, cx),
                }
            }
        };
        outcome
    }

    fn begin_context_input(
        &mut self,
        invocation: GraphContextInvocation,
        operation: GraphContextInputOperation,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> GraphContextDispatchOutcome {
        let initial_text = self
            .context_input_initial_text(&operation)
            .unwrap_or_default();
        let multiline = match &operation {
            GraphContextInputOperation::EditNodeProperty { identifier, key } => self
                .model
                .document()
                .and_then(|document| document.active_graph().ok())
                .and_then(|graph| graph.nodes.get(identifier))
                .and_then(|node| graph_node_property_descriptors(node).ok())
                .and_then(|descriptors| {
                    descriptors
                        .into_iter()
                        .find(|descriptor| descriptor.key == key.as_str())
                })
                .is_some_and(|descriptor| matches!(descriptor.kind, GraphNodePropertyKind::Json)),
            _ => false,
        };
        let editor = cx.new(|cx| {
            let mut editor = if multiline {
                Editor::multi_line(window, cx)
            } else {
                Editor::single_line(window, cx)
            };
            editor.set_text(initial_text, window, cx);
            editor.set_placeholder_text(
                match &operation {
                    GraphContextInputOperation::SetGroupFontSize(_) => "Enter a size from 6 to 96",
                    GraphContextInputOperation::EditNodeProperty { .. } => {
                        "Enter a value matching the declared property type"
                    }
                    GraphContextInputOperation::BatchRenameSelection { .. } => {
                        "Enter a base name for the selected items"
                    }
                    GraphContextInputOperation::FrameSelection => "Enter a title for the new frame",
                    GraphContextInputOperation::PublishSubgraph => {
                        "Enter the exact blueprint filename without .json"
                    }
                    _ => "Enter a nonempty name",
                },
                window,
                cx,
            );
            editor
        });
        let Some(workspace) = self.workspace().upgrade() else {
            return self.reject_context_action(
                "workspace modal layer is unavailable for context input",
                cx,
            );
        };
        self.context_input = Some(GraphContextInputState {
            invocation,
            operation: operation.clone(),
            editor: editor.clone(),
        });
        self.model.announcement = Some("Context action requires text input".to_owned());
        cx.notify();
        let item = cx.weak_entity();
        workspace.update(cx, |workspace, cx| {
            workspace.toggle_modal(window, cx, move |_, _| GraphContextInputModal {
                item,
                operation,
                editor,
            });
        });
        GraphContextDispatchOutcome::InputPending
    }

    fn context_input_initial_text(&self, operation: &GraphContextInputOperation) -> Option<String> {
        let graph = self.model.document()?.active_graph().ok()?;
        match operation {
            GraphContextInputOperation::RenameNode(identifier) => {
                graph.nodes.get(identifier).map(|node| node.title.clone())
            }
            GraphContextInputOperation::RenameGroup(identifier) => graph
                .groups
                .get(identifier)
                .map(|group| group.title.clone()),
            GraphContextInputOperation::BatchRenameSelection { .. } => Some("Item".to_owned()),
            GraphContextInputOperation::FrameSelection => Some("Group".to_owned()),
            GraphContextInputOperation::RenameSlot { direction, slot } => self
                .model
                .document()?
                .active_subgraph_definition()
                .ok()
                .and_then(|definition| match direction {
                    GraphSlotDirection::Input => definition.inputs.get(*slot),
                    GraphSlotDirection::Output => definition.outputs.get(*slot),
                })
                .map(|port| port.name.clone()),
            GraphContextInputOperation::SetGroupFontSize(identifier) => graph
                .groups
                .get(identifier)
                .and_then(|group| group.source_fields.get("font_size"))
                .and_then(serde_json::Value::as_f64)
                .map_or_else(|| Some("20".to_owned()), |size| Some(size.to_string())),
            GraphContextInputOperation::EditNodeProperty { identifier, key } => graph
                .nodes
                .get(identifier)
                .and_then(|node| graph_node_property_descriptors(node).ok())
                .and_then(|descriptors| {
                    descriptors
                        .into_iter()
                        .find(|descriptor| descriptor.key == key.as_str())
                })
                .map(|descriptor| graph_node_property_value_label(&descriptor.value)),
            GraphContextInputOperation::ConvertToSubgraph => Some("Subgraph".to_owned()),
            GraphContextInputOperation::PublishSubgraph => graph
                .selection
                .nodes
                .iter()
                .next()
                .and_then(|identifier| graph.nodes.get(identifier))
                .map(|node| node.title.clone()),
        }
    }

    pub(crate) fn confirm_context_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(state) = self.context_input.clone() else {
            return;
        };
        if let Err(error) = self.validate_context_invocation(&state.invocation, true) {
            self.reject_context_action(error, cx);
            self.context_input = None;
            return;
        }
        let text = state.editor.read(cx).text(cx);
        let text = text.trim();
        let requires_label = matches!(
            &state.operation,
            GraphContextInputOperation::RenameNode(_)
                | GraphContextInputOperation::RenameGroup(_)
                | GraphContextInputOperation::BatchRenameSelection { .. }
                | GraphContextInputOperation::FrameSelection
                | GraphContextInputOperation::RenameSlot { .. }
                | GraphContextInputOperation::ConvertToSubgraph
                | GraphContextInputOperation::PublishSubgraph
        );
        if requires_label
            && (text.is_empty() || text.chars().count() > 256 || text.chars().any(char::is_control))
        {
            self.model
                .report_error("graph names must be nonempty, bounded, and contain no controls");
            cx.notify();
            return;
        }
        let succeeded = match state.operation {
            GraphContextInputOperation::RenameNode(identifier) => self.apply_graph_command(
                GraphCommand::RenameNode {
                    identifier,
                    title: text.to_owned(),
                },
                cx,
            ),
            GraphContextInputOperation::RenameGroup(identifier) => self.apply_graph_command(
                GraphCommand::RenameGroup {
                    identifier,
                    title: text.to_owned(),
                },
                cx,
            ),
            GraphContextInputOperation::BatchRenameSelection { nodes, groups } => {
                let commands = nodes
                    .into_iter()
                    .map(|identifier| (true, identifier))
                    .chain(groups.into_iter().map(|identifier| (false, identifier)))
                    .enumerate()
                    .map(|(index, (is_node, identifier))| {
                        let title = format!("{} {}", text, index + 1);
                        if is_node {
                            GraphCommand::RenameNode { identifier, title }
                        } else {
                            GraphCommand::RenameGroup { identifier, title }
                        }
                    })
                    .collect();
                self.apply_graph_command(GraphCommand::Batch { commands }, cx)
            }
            GraphContextInputOperation::FrameSelection => {
                let command = match self.group_selection_command(
                    &state.invocation,
                    text.to_owned(),
                    self.group_selected_nodes_padding(cx),
                ) {
                    Ok(command) => command,
                    Err(error) => {
                        self.model.report_error(error);
                        cx.notify();
                        return;
                    }
                };
                self.apply_graph_command(command, cx)
            }
            GraphContextInputOperation::RenameSlot { direction, slot } => self.apply_graph_command(
                GraphCommand::RenameSubgraphSlot {
                    direction,
                    slot,
                    name: text.to_owned(),
                },
                cx,
            ),
            GraphContextInputOperation::SetGroupFontSize(identifier) => {
                let Ok(font_size) = text.parse::<f32>() else {
                    self.model
                        .report_error("group font size must be a number from 6 to 96");
                    cx.notify();
                    return;
                };
                self.apply_graph_command(
                    GraphCommand::SetGroupFontSize {
                        identifier,
                        font_size,
                    },
                    cx,
                )
            }
            GraphContextInputOperation::EditNodeProperty { identifier, key } => {
                let Some(node) = self
                    .model
                    .document()
                    .and_then(|document| document.active_graph().ok())
                    .and_then(|graph| graph.nodes.get(&identifier))
                    .cloned()
                else {
                    self.model.report_error("node target is unavailable");
                    cx.notify();
                    return;
                };
                let descriptor = match graph_node_property_descriptors(&node)
                    .and_then(|descriptors| {
                        descriptors
                            .into_iter()
                            .find(|descriptor| descriptor.key == key)
                            .ok_or_else(|| {
                                crate::properties_panel::GraphNodePropertyAdapterError::UnknownProperty {
                                    key: key.clone(),
                                }
                            })
                    }) {
                    Ok(descriptor) => descriptor,
                    Err(error) => {
                        self.model.report_error(error.to_string());
                        cx.notify();
                        return;
                    }
                };
                let value = match parse_graph_node_property_value(&descriptor, text) {
                    Ok(value) => value,
                    Err(error) => {
                        self.model.report_error(error.to_string());
                        cx.notify();
                        return;
                    }
                };
                let properties = match graph_node_properties_with_value(&node, &key, value) {
                    Ok(properties) => properties,
                    Err(error) => {
                        self.model.report_error(error.to_string());
                        cx.notify();
                        return;
                    }
                };
                self.apply_graph_command(
                    GraphCommand::SetNodeProperties {
                        identifier,
                        properties,
                    },
                    cx,
                )
            }
            GraphContextInputOperation::ConvertToSubgraph => self.execute_catalog_action(
                CatalogGraphAction::ConvertToSubgraph,
                GraphActionInput::SubgraphName(text.to_owned()),
                cx,
            ),
            GraphContextInputOperation::PublishSubgraph => self.begin_subgraph_publish(
                state.invocation,
                text.to_owned(),
                AssetCollisionPolicy::Reject,
                None,
                window,
                cx,
            ),
        };
        if succeeded {
            self.context_input = None;
            self.model.announcement = Some("Applied native context input".to_owned());
            cx.notify();
        }
    }

    pub(crate) fn begin_shell_publish_subgraph(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let invocation = match self.selection_context_invocation() {
            Ok(invocation) => invocation,
            Err(error) => {
                self.reject_context_action(error, cx);
                return;
            }
        };
        let binding = match graph_context_action_binding("COMFY-GRAPH-136") {
            Ok(binding) => binding,
            Err(error) => {
                self.reject_context_action(error, cx);
                return;
            }
        };
        self.dispatch_context_action(
            binding,
            GraphContextDispatchInput::None,
            invocation,
            window,
            cx,
        );
    }

    fn selection_context_invocation(&self) -> Result<GraphContextInvocation, String> {
        let document = self
            .model
            .document()
            .ok_or_else(|| "workflow is open read-only".to_owned())?;
        let graph = document.active_graph().map_err(|error| error.to_string())?;
        let content_revision = document
            .to_workflow_bytes()
            .map(|bytes| ContentRevision::from_bytes(&bytes))
            .map_err(|error| format!("workflow cannot be published: {error}"))?;
        Ok(GraphContextInvocation {
            document_identity: document.document_identity,
            content_revision,
            navigation: document.navigation.clone(),
            selection: graph.selection.clone(),
            target: GraphContextTarget::Selection,
            screen_position: GraphPoint::ZERO,
        })
    }

    fn begin_subgraph_publish(
        &mut self,
        invocation: GraphContextInvocation,
        display_name: String,
        collision_policy: AssetCollisionPolicy,
        return_focus: Option<FocusHandle>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.subgraph_publish_task.is_some() {
            self.model
                .report_error("another subgraph publication is already in progress");
            cx.notify();
            return false;
        }
        let Some(document) = self.model.document().cloned() else {
            self.model.report_error("workflow is open read-only");
            cx.notify();
            return false;
        };
        let Some(services) = native_asset_services(cx) else {
            self.model
                .report_error("the canonical native asset service is unavailable");
            cx.notify();
            return false;
        };
        let cancellation = CancellationToken::default();
        let publication = cx.background_spawn({
            let library = services.subgraph_blueprints();
            let cancellation = cancellation.clone();
            let display_name = display_name.clone();
            async move {
                library.publish(
                    &document,
                    &display_name,
                    collision_policy,
                    &cancellation,
                )
            }
        });
        let return_focus = return_focus
            .or_else(|| {
                self.context_menu_state
                    .as_ref()
                    .and_then(|state| state.return_focus.clone())
            })
            .or_else(|| Some(self.focus_handle.clone()));
        self.subgraph_publish_cancellation = Some(cancellation);
        self.model.announcement = Some(format!("Publishing blueprint {display_name}"));
        cx.notify();
        let (completion_sender, completion_receiver) = oneshot::channel();
        #[cfg(all(test, feature = "test-support"))]
        let projection_barrier = self.subgraph_publish_projection_barrier.take();
        cx.spawn(async move |_this, cx| {
            let result = publication.await;
            #[cfg(all(test, feature = "test-support"))]
            if let Some(projection_barrier) = projection_barrier
                && projection_barrier.await.is_err()
            {
                log::debug!("subgraph publication projection test barrier was dropped");
            }
            let completion = match result {
                Ok(publication) => {
                    let diagnostic_message =
                        crate::subgraph_catalog_diagnostic_message(&publication.catalog);
                    let display_name = publication.entry.descriptor.display_name.clone();
                    match cx.update(|cx| {
                        crate::replace_native_subgraph_catalog(publication.catalog, cx)
                    }) {
                        Ok(()) => SubgraphPublishCompletion::Published {
                            display_name,
                            diagnostic_message,
                        },
                        Err(error) => {
                            log::error!(
                                "published subgraph blueprint but failed to refresh its catalog projection: {error}"
                            );
                            SubgraphPublishCompletion::Failed(format!(
                                "blueprint was published but its catalog projection failed: {error}"
                            ))
                        }
                    }
                }
                Err(SubgraphBlueprintLibraryError::Asset(AssetError::AlreadyExists(_))) => {
                    SubgraphPublishCompletion::AlreadyExists
                }
                Err(error) => SubgraphPublishCompletion::Failed(format!(
                    "subgraph blueprint publication failed: {error}"
                )),
            };
            if completion_sender.send(completion).is_err() {
                log::debug!(
                    "subgraph publication completed after its workspace item was released"
                );
            }
        })
        .detach();
        self.subgraph_publish_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = completion_receiver.await;
            if let Err(error) = this.update_in(cx, |this, window, cx| {
                this.subgraph_publish_task = None;
                this.subgraph_publish_cancellation = None;
                match result {
                    Ok(SubgraphPublishCompletion::Published {
                        display_name,
                        diagnostic_message,
                    }) => {
                        let announcement = format!("Published blueprint {display_name}");
                        if let Some(message) = diagnostic_message {
                            this.model.report_error(&message);
                            this.model.announcement = Some(format!("{announcement}; {message}"));
                        } else {
                            this.model.announcement = Some(announcement);
                        }
                        cx.notify();
                        restore_focus(return_focus, window, cx);
                    }
                    Ok(SubgraphPublishCompletion::AlreadyExists)
                        if collision_policy == AssetCollisionPolicy::Reject =>
                    {
                        this.begin_subgraph_overwrite_confirmation(
                            invocation,
                            display_name,
                            return_focus,
                            window,
                            cx,
                        );
                    }
                    Ok(SubgraphPublishCompletion::AlreadyExists) => {
                        this.model.report_error(
                            "subgraph blueprint replacement unexpectedly reported an existing asset",
                        );
                        cx.notify();
                        restore_focus(return_focus, window, cx);
                    }
                    Ok(SubgraphPublishCompletion::Failed(error)) => {
                        this.model.report_error(error);
                        cx.notify();
                        restore_focus(return_focus, window, cx);
                    }
                    Err(error) => {
                        this.model.report_error(format!(
                            "subgraph blueprint completion was cancelled: {error}"
                        ));
                        cx.notify();
                        restore_focus(return_focus, window, cx);
                    }
                }
            }) {
                log::error!("graph disappeared while publishing a subgraph blueprint: {error}");
            }
        }));
        true
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn begin_subgraph_publish_for_test(
        &mut self,
        display_name: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let invocation = match self.selection_context_invocation() {
            Ok(invocation) => invocation,
            Err(error) => {
                self.model.report_error(error);
                cx.notify();
                return false;
            }
        };
        self.begin_subgraph_publish(
            invocation,
            display_name,
            AssetCollisionPolicy::Reject,
            None,
            window,
            cx,
        )
    }

    fn begin_subgraph_overwrite_confirmation(
        &mut self,
        invocation: GraphContextInvocation,
        display_name: String,
        return_focus: Option<FocusHandle>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let detail =
            format!("A blueprint named {display_name:?} already exists. Replace that exact asset?");
        let prompt = window.prompt(
            PromptLevel::Warning,
            "Overwrite existing blueprint?",
            Some(&detail),
            &["Replace", "Cancel", "×"],
            cx,
        );
        self.context_confirmation_task = Some(cx.spawn_in(window, async move |this, cx| {
            match prompt.await {
                Ok(0) => {
                    if let Err(error) = this.update_in(cx, |this, window, cx| {
                        this.context_confirmation_task = None;
                        if let Err(error) = this.validate_context_invocation(&invocation, true) {
                            this.reject_context_action(error, cx);
                            restore_focus(return_focus, window, cx);
                            return;
                        }
                        this.begin_subgraph_publish(
                            invocation,
                            display_name,
                            AssetCollisionPolicy::Replace,
                            return_focus,
                            window,
                            cx,
                        );
                    }) {
                        log::error!(
                            "graph disappeared during subgraph overwrite confirmation: {error}"
                        );
                    }
                }
                Ok(_) => {
                    if let Err(error) = this.update_in(cx, |this, window, cx| {
                        this.context_confirmation_task = None;
                        this.model.announcement =
                            Some("Cancelled subgraph blueprint overwrite".to_owned());
                        cx.notify();
                        restore_focus(return_focus, window, cx);
                    }) {
                        log::error!(
                            "graph disappeared while cancelling subgraph overwrite: {error}"
                        );
                    }
                }
                Err(error) => {
                    if let Err(update_error) = this.update_in(cx, |this, window, cx| {
                        this.context_confirmation_task = None;
                        this.model.report_error(format!(
                            "subgraph overwrite confirmation failed: {error}"
                        ));
                        cx.notify();
                        restore_focus(return_focus, window, cx);
                    }) {
                        log::error!(
                            "graph disappeared while reporting overwrite confirmation failure: {update_error}"
                        );
                    }
                }
            }
        }));
    }

    fn begin_context_confirmation(
        &mut self,
        invocation: GraphContextInvocation,
        title: &'static str,
        detail: &'static str,
        confirm_label: &'static str,
        command: GraphCommand,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> GraphContextDispatchOutcome {
        let return_focus = self
            .context_menu_state
            .as_ref()
            .and_then(|state| state.return_focus.clone());
        let prompt = window.prompt(
            PromptLevel::Warning,
            title,
            Some(detail),
            &[confirm_label, "Cancel", "×"],
            cx,
        );
        self.context_confirmation_task = Some(cx.spawn_in(window, async move |this, cx| {
            match prompt.await {
                Ok(0) => {
                    if let Err(error) = this.update_in(cx, |this, window, cx| {
                        this.context_confirmation_task = None;
                        if let Err(error) = this.validate_context_invocation(&invocation, true) {
                            this.reject_context_action(error, cx);
                            restore_focus(return_focus, window, cx);
                            return;
                        }
                        this.apply_graph_command(command, cx);
                        restore_focus(return_focus, window, cx);
                    }) {
                        log::error!("graph disappeared during context confirmation: {error}");
                    }
                }
                Ok(_) => {
                    if let Err(error) = this.update_in(cx, |this, window, cx| {
                        this.context_confirmation_task = None;
                        this.model.announcement =
                            Some("Cancelled destructive graph action".to_owned());
                        cx.notify();
                        restore_focus(return_focus, window, cx);
                    }) {
                        log::error!(
                            "graph disappeared while cancelling context confirmation: {error}"
                        );
                    }
                }
                Err(error) => {
                    if let Err(update_error) = this.update_in(cx, |this, window, cx| {
                        this.context_confirmation_task = None;
                        this.model
                            .report_error(format!("graph confirmation failed: {error}"));
                        cx.notify();
                        restore_focus(return_focus, window, cx);
                    }) {
                        log::error!(
                            "graph disappeared while reporting confirmation failure: {update_error}"
                        );
                    }
                }
            }
        }));
        GraphContextDispatchOutcome::ConfirmationPending
    }

    fn reject_context_action(
        &mut self,
        error: impl std::fmt::Display,
        cx: &mut Context<Self>,
    ) -> GraphContextDispatchOutcome {
        let error = error.to_string();
        self.model.report_error(&error);
        cx.notify();
        GraphContextDispatchOutcome::Rejected(error)
    }

    fn context_rejection_from_model(&self) -> GraphContextDispatchOutcome {
        GraphContextDispatchOutcome::Rejected(
            self.model
                .last_error
                .clone()
                .unwrap_or_else(|| "native context action was rejected".to_owned()),
        )
    }

    fn context_group_nodes(
        &self,
        invocation: &GraphContextInvocation,
    ) -> Option<BTreeSet<GraphIdentifier>> {
        let identifier = invocation.group_identifier()?;
        self.model
            .document()?
            .active_graph()
            .ok()?
            .groups
            .get(&identifier)
            .map(|group| group.node_ids.clone())
    }

    fn group_selection_command(
        &self,
        invocation: &GraphContextInvocation,
        title: String,
        padding: f32,
    ) -> Result<GraphCommand, String> {
        let document = self
            .model
            .document()
            .ok_or_else(|| "workflow is open read-only".to_owned())?;
        let graph = document.active_graph().map_err(|error| error.to_string())?;
        let mut bounds: Option<(f32, f32, f32, f32)> = None;
        let mut include_rect = |rect: GraphRect| {
            let right = rect.origin.x + rect.size.width;
            let bottom = rect.origin.y + rect.size.height;
            bounds = Some(match bounds {
                Some((left, top, existing_right, existing_bottom)) => (
                    left.min(rect.origin.x),
                    top.min(rect.origin.y),
                    existing_right.max(right),
                    existing_bottom.max(bottom),
                ),
                None => (rect.origin.x, rect.origin.y, right, bottom),
            });
        };
        for identifier in &invocation.selection.nodes {
            if let Some(node) = graph.nodes.get(identifier) {
                include_rect(GraphRect {
                    origin: node.position,
                    size: node.size,
                });
            }
        }
        for identifier in &invocation.selection.groups {
            if let Some(group) = graph.groups.get(identifier) {
                include_rect(group.bounds);
            }
        }
        let Some((left, top, right, bottom)) = bounds else {
            return Err("positionable selection is empty".to_owned());
        };
        let mut source_fields = Map::new();
        source_fields.insert("font_size".to_owned(), serde_json::Value::from(20.0));
        Ok(GraphCommand::CreateGroup {
            group: GraphGroup {
                identifier: document.next_available_identifier(),
                title,
                bounds: GraphRect {
                    origin: GraphPoint {
                        x: left - padding,
                        y: top - padding,
                    },
                    size: GraphSize {
                        width: (right - left + padding * 2.0).max(140.0),
                        height: (bottom - top + padding * 2.0).max(80.0),
                    },
                },
                node_ids: invocation.selection.nodes.clone(),
                collapsed: false,
                pinned: false,
                color: Some("#3f789e".to_owned()),
                source_fields,
            },
        })
    }

    pub(crate) fn context_reroute_type_visible(
        &self,
        identifier: &GraphIdentifier,
        cx: &App,
    ) -> bool {
        self.model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.reroutes.get(identifier))
            .map(|reroute| {
                reroute
                    .source_fields
                    .get("show_type")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or_else(|| self.context_default_reroute_type_visible(cx))
            })
            .unwrap_or(false)
    }

    fn context_default_reroute_type_visible(&self, cx: &App) -> bool {
        cx.try_global::<SettingsStore>()
            .and_then(|store| store.merged_settings().sim_runtime.as_ref())
            .and_then(|settings| settings.show_reroute_types)
            .unwrap_or(false)
    }
}

impl GraphContextInvocation {
    fn group_identifier(&self) -> Option<GraphIdentifier> {
        match &self.target {
            GraphContextTarget::Group(identifier) => Some(identifier.clone()),
            _ => None,
        }
    }

    fn slot_target(&self) -> Option<(GraphSlotDirection, usize)> {
        match &self.target {
            GraphContextTarget::Slot { direction, slot } => Some((*direction, *slot)),
            _ => None,
        }
    }
}
