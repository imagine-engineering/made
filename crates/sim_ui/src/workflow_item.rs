use crate::{
    CommandDispatchOutcome, CommandNativeStatus, ExecutionPlanRequest, ExecutionRunMode,
    ExecutionUiModel, GraphActionEffect, GraphActionError, GraphActionInput, GraphCommandModel,
    GraphContextInputState, GraphContextMenuState, GraphModelAction, GraphWorkspaceError,
    GraphWorkspaceModel, command_registration, execution_ui_model, graph_render,
};
use anyhow::{Context as _, anyhow};
use futures::StreamExt as _;
use gpui::{
    Action as _, App, AppContext as _, ClipboardEntry, ClipboardItem, Context, Entity,
    EventEmitter, FocusHandle, Focusable, Render, SharedString, Subscription, Task, WeakEntity,
    Window, actions,
};
use project::{Project, ProjectPath, RemoveOptions, RenameOptions};
use settings::SettingsStore;
use sim_runtime::{
    AttemptState, CatalogGraphAction, ContentRevision, ExecutionCommandOutcome,
    ExecutionControlCommandKind, ExecutionSnapshotStatus, GraphCommand, GraphIdentifier, GraphLink,
    GraphPoint, GraphRect, GraphReroute, GraphSelection, GraphViewport, ProfileId, SelectionMode,
    WorkflowAuthority, WorkflowStorageProvider,
};
use sim_types::{CancellationToken, NodeId};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use workspace::item::ItemEvent;
use workspace::notifications::DetachAndPromptErr as _;
use workspace::{Item, ItemId, SerializableItem, Workspace, WorkspaceId, delete_unloaded_items};

macro_rules! shell_action_handler {
    ($method:ident, $action:ty, $command_id:literal) => {
        pub(crate) fn $method(&mut self, _: &$action, _: &mut Window, cx: &mut Context<Self>) {
            self.dispatch_shell_command($command_id, cx);
        }
    };
}

actions!(
    sim_graph,
    [
        #[action(deprecated_aliases = ["comfy_graph::GraphUndo"])]
        GraphUndo,
        #[action(deprecated_aliases = ["comfy_graph::GraphRedo"])]
        GraphRedo,
        #[action(deprecated_aliases = ["comfy_graph::GraphCopy"])]
        GraphCopy,
        #[action(deprecated_aliases = ["comfy_graph::GraphCut"])]
        GraphCut,
        #[action(deprecated_aliases = ["comfy_graph::GraphPaste"])]
        GraphPaste,
        #[action(deprecated_aliases = ["comfy_graph::GraphDelete"])]
        GraphDelete,
        #[action(deprecated_aliases = ["comfy_graph::GraphSelectAll"])]
        GraphSelectAll,
        #[action(deprecated_aliases = ["comfy_graph::GraphZoomIn"])]
        GraphZoomIn,
        #[action(deprecated_aliases = ["comfy_graph::GraphZoomOut"])]
        GraphZoomOut,
        #[action(deprecated_aliases = ["comfy_graph::GraphFitView"])]
        GraphFitView,
        #[action(deprecated_aliases = ["comfy_graph::GraphCancelGesture"])]
        GraphCancelGesture,
    ]
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphDirtyCloseChoice {
    Save,
    Discard,
    Cancel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphCloseDisposition {
    SaveThenClose,
    CloseWithoutSaving,
    KeepOpen,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphOverwriteChoice {
    RejectExisting,
    ReplaceExisting,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphDeleteFileChoice {
    Cancel,
    Confirm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphConflictComparison {
    pub base: Vec<u8>,
    pub local: Vec<u8>,
    pub external: Option<Vec<u8>>,
}

pub struct GraphWorkspaceItem {
    pub(crate) model: GraphWorkspaceModel,
    pub(crate) focus_handle: FocusHandle,
    workspace: WeakEntity<Workspace>,
    pub(crate) drag_anchor: Option<GraphPoint>,
    pub(crate) drag_delta: GraphPoint,
    pub(crate) box_anchor: Option<GraphPoint>,
    pub(crate) box_current: Option<GraphPoint>,
    pub(crate) box_selection_mode: SelectionMode,
    pub(crate) pending_link: Option<(GraphIdentifier, usize)>,
    pub(crate) pending_link_position: Option<GraphPoint>,
    pub(crate) pending_reconnect: Option<GraphIdentifier>,
    pub(crate) node_rename: Option<(GraphIdentifier, String)>,
    pub(crate) context_menu_handle: ui::RightClickMenuHandle<ui::ContextMenu>,
    pub(crate) context_menu_state: Option<GraphContextMenuState>,
    pub(crate) pending_pointer_context_target: Option<crate::GraphContextTarget>,
    pub(crate) canvas_pan_anchor: Option<(GraphPoint, GraphViewport)>,
    pub(crate) context_input: Option<GraphContextInputState>,
    pub(crate) context_confirmation_task: Option<Task<()>>,
    pub(crate) context_settings_task: Option<Task<()>>,
    pub(crate) subgraph_publish_task: Option<Task<()>>,
    pub(crate) subgraph_publish_cancellation: Option<CancellationToken>,
    #[cfg(all(test, feature = "test-support"))]
    pub(crate) subgraph_publish_projection_barrier: Option<futures::channel::oneshot::Receiver<()>>,
    control_focus_handles: BTreeMap<String, FocusHandle>,
    active_control_focus_handles: BTreeSet<String>,
    file_watch: Option<Task<()>>,
    execution_model: Option<Entity<ExecutionUiModel>>,
    _execution_subscription: Option<Subscription>,
    _settings_subscription: Subscription,
    _initialization_subscription: Subscription,
    _subgraph_catalog_subscription: Option<Subscription>,
    execution_navigation: Vec<ExecutionNavigationState>,
    execution_run_mode: ExecutionRunMode,
    execution_mode_menu_open: bool,
    execute_output_feedback_hovered: bool,
    automatic_execution_task: Option<Task<()>>,
    execution_projection_navigation: Option<Vec<GraphIdentifier>>,
    pub(crate) queue_overlay_visible: bool,
    pub(crate) queue_overlay_tab: QueueOverlayTab,
    pub(crate) queue_details_attempt: Option<sim_runtime::AttemptId>,
    pub(crate) qpov2_enabled: bool,
    pub(crate) show_execution_progress: bool,
    dismissed_execution_error: Option<sim_runtime::AttemptId>,
    #[cfg(feature = "test-support")]
    shell_dispatch_trace: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum QueueOverlayTab {
    #[default]
    All,
    Completed,
    Failed,
}

#[derive(Clone)]
struct ExecutionNavigationState {
    selection: GraphSelection,
    viewport: GraphViewport,
}

impl GraphWorkspaceItem {
    pub fn new(
        mut model: GraphWorkspaceModel,
        workspace: WeakEntity<Workspace>,
        cx: &mut Context<Self>,
    ) -> Self {
        let execution_model = execution_ui_model(cx);
        if let Some(profile_id) = execution_model
            .as_ref()
            .and_then(|execution_model| execution_model.read(cx).active_profile_id())
        {
            model.bind_profile_identity(profile_id);
        }
        if let Some(catalog) = crate::native_subgraph_catalog(cx)
            && let Some(message) = crate::subgraph_catalog_diagnostic_message(catalog)
        {
            model.report_error(message);
        }
        let subgraph_catalog_subscription =
            cx.has_global::<crate::GlobalNativeSubgraphCatalogRevision>()
                .then(|| {
                    cx.observe_global::<crate::GlobalNativeSubgraphCatalogRevision>(
                        |this: &mut GraphWorkspaceItem, cx| {
                            if let Some(catalog) = crate::native_subgraph_catalog(cx) {
                                if let Some(message) =
                                    crate::subgraph_catalog_diagnostic_message(catalog)
                                    && this.model.last_error.as_deref() != Some(message.as_str())
                                {
                                    this.model.report_error(message);
                                }
                                if !this.model.announcement.as_deref().is_some_and(|message| {
                                    message.starts_with("Published blueprint ")
                                }) {
                                    this.model.announcement =
                                        Some(crate::subgraph_catalog_node_library_message(catalog));
                                }
                            }
                            cx.notify();
                        },
                    )
                });
        let execution_subscription = execution_model.as_ref().map(|model| {
            cx.observe(model, |this: &mut GraphWorkspaceItem, _, cx| {
                this.reconcile_execution_capability(cx);
                cx.emit(ItemEvent::UpdateTab);
                cx.notify();
            })
        });
        let settings_subscription =
            cx.observe_global::<SettingsStore>(|_: &mut GraphWorkspaceItem, cx| cx.notify());
        let initialization_subscription =
            cx.observe_global::<crate::SimUiRegistration>(|_: &mut GraphWorkspaceItem, cx| {
                cx.notify()
            });
        Self {
            model,
            focus_handle: cx.focus_handle(),
            workspace,
            drag_anchor: None,
            drag_delta: GraphPoint::ZERO,
            box_anchor: None,
            box_current: None,
            box_selection_mode: SelectionMode::Replace,
            pending_link: None,
            pending_link_position: None,
            pending_reconnect: None,
            node_rename: None,
            context_menu_handle: ui::RightClickMenuHandle::default(),
            context_menu_state: None,
            pending_pointer_context_target: None,
            canvas_pan_anchor: None,
            context_input: None,
            context_confirmation_task: None,
            context_settings_task: None,
            subgraph_publish_task: None,
            subgraph_publish_cancellation: None,
            #[cfg(all(test, feature = "test-support"))]
            subgraph_publish_projection_barrier: None,
            control_focus_handles: BTreeMap::new(),
            active_control_focus_handles: BTreeSet::new(),
            file_watch: None,
            execution_model,
            _execution_subscription: execution_subscription,
            _settings_subscription: settings_subscription,
            _initialization_subscription: initialization_subscription,
            _subgraph_catalog_subscription: subgraph_catalog_subscription,
            execution_navigation: Vec::new(),
            execution_run_mode: ExecutionRunMode::Manual,
            execution_mode_menu_open: false,
            execute_output_feedback_hovered: false,
            automatic_execution_task: None,
            execution_projection_navigation: None,
            queue_overlay_visible: false,
            queue_overlay_tab: QueueOverlayTab::All,
            queue_details_attempt: None,
            qpov2_enabled: true,
            show_execution_progress: true,
            dismissed_execution_error: None,
            #[cfg(feature = "test-support")]
            shell_dispatch_trace: Vec::new(),
        }
    }

    pub fn new_draft(
        title: impl Into<String>,
        workspace: WeakEntity<Workspace>,
        cx: &mut Context<Self>,
    ) -> Result<Self, GraphWorkspaceError> {
        Ok(Self::new(
            GraphWorkspaceModel::create(title)?,
            workspace,
            cx,
        ))
    }

    pub fn model(&self) -> &GraphWorkspaceModel {
        &self.model
    }

    pub fn workspace(&self) -> &WeakEntity<Workspace> {
        &self.workspace
    }

    pub(crate) fn active_execution_presentation(
        &self,
        cx: &App,
    ) -> Option<sim_runtime::AttemptPresentation> {
        let snapshot = self.execution_snapshot(cx)?;
        let associated_attempt_id = self
            .model
            .execution_association
            .as_ref()
            .and_then(|identity| uuid::Uuid::parse_str(identity).ok())
            .map(sim_runtime::AttemptId);
        associated_attempt_id.and_then(|attempt_id| {
            snapshot
                .attempts
                .iter()
                .find(|attempt| attempt.attempt_id == attempt_id)
                .cloned()
                .map(|mut attempt| {
                    let current_navigation = self
                        .model
                        .document()
                        .map(|document| document.navigation.as_slice());
                    if current_navigation != self.execution_projection_navigation.as_deref() {
                        attempt.progress = None;
                        attempt.node_progress.clear();
                        attempt.preview = None;
                        attempt.previews.clear();
                    }
                    attempt
                })
        })
    }

    #[cfg(feature = "test-support")]
    pub fn active_execution_presentation_for_test(
        &self,
        cx: &App,
    ) -> Option<sim_runtime::AttemptPresentation> {
        self.active_execution_presentation(cx)
    }

    pub(crate) fn execution_snapshot(&self, cx: &App) -> Option<sim_runtime::ExecutionSnapshot> {
        let profile_id = self.execution_profile_id().ok()?;
        self.execution_model
            .as_ref()?
            .read(cx)
            .snapshot(profile_id)
            .ok()
    }

    pub(crate) fn execution_run_mode(&self) -> ExecutionRunMode {
        self.execution_run_mode
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn execution_queue_available(&self, cx: &App) -> bool {
        self.execution_queue_unavailable_reason(cx).is_none()
    }

    pub(crate) fn execution_queue_unavailable_reason(&self, cx: &App) -> Option<String> {
        let profile_id = match self.execution_profile_id() {
            Ok(profile_id) => profile_id,
            Err(error) => return Some(error),
        };
        let Some(execution_model) = self.execution_model.as_ref() else {
            return Some("the native execution service is unavailable".to_owned());
        };
        let execution_model = execution_model.read(cx);
        if !execution_model.runtime_controller_available() {
            return Some("the native runtime controller is not connected".to_owned());
        }
        if !execution_model.plan_provider_available() {
            return Some("no native plan provider is registered for this profile".to_owned());
        }
        match execution_model.snapshot(profile_id) {
            Ok(snapshot) if snapshot.status == ExecutionSnapshotStatus::Ready => None,
            Ok(snapshot) => Some(format!(
                "the native execution profile is not ready ({:?})",
                snapshot.status
            )),
            Err(error) => Some(format!(
                "the native execution snapshot is unavailable: {error}"
            )),
        }
    }

    fn execution_control_unavailable_reason(&self, cx: &App) -> Option<String> {
        let profile_id = match self.execution_profile_id() {
            Ok(profile_id) => profile_id,
            Err(error) => return Some(error),
        };
        let Some(execution_model) = self.execution_model.as_ref() else {
            return Some("the native execution service is unavailable".to_owned());
        };
        let execution_model = execution_model.read(cx);
        if !execution_model.runtime_controller_available() {
            return Some("the native runtime controller is not connected".to_owned());
        }
        match execution_model.snapshot(profile_id) {
            Ok(snapshot) if snapshot.status == ExecutionSnapshotStatus::Ready => None,
            Ok(snapshot) => Some(format!(
                "the native execution profile is not ready ({:?})",
                snapshot.status
            )),
            Err(error) => Some(format!(
                "the native execution snapshot is unavailable: {error}"
            )),
        }
    }

    pub(crate) fn set_execute_output_feedback_hovered(
        &mut self,
        hovered: bool,
        cx: &mut Context<Self>,
    ) {
        if self.execute_output_feedback_hovered != hovered {
            self.execute_output_feedback_hovered = hovered;
            cx.notify();
        }
    }

    pub(crate) fn execute_output_feedback_hovered(&self) -> bool {
        self.execute_output_feedback_hovered
    }

    pub(crate) fn associate_execution_attempt(
        &mut self,
        attempt_id: sim_runtime::AttemptId,
        cx: &mut Context<Self>,
    ) {
        self.model.execution_association = Some(attempt_id.0.to_string());
        self.execution_projection_navigation = self
            .model
            .document()
            .map(|document| document.navigation.clone());
        self.dismissed_execution_error = None;
        cx.emit(ItemEvent::Edit);
        cx.emit(ItemEvent::UpdateTab);
        cx.notify();
    }

    pub(crate) fn is_associated_with_execution(&self, attempt_id: sim_runtime::AttemptId) -> bool {
        self.model
            .execution_association
            .as_deref()
            .and_then(|identity| uuid::Uuid::parse_str(identity).ok())
            == Some(attempt_id.0)
    }

    pub(crate) fn can_restore_execution_navigation(&self) -> bool {
        !self.execution_navigation.is_empty()
    }

    pub(crate) fn execution_mode_menu_open(&self) -> bool {
        self.execution_mode_menu_open
    }

    pub(crate) fn toggle_execution_mode_menu(&mut self, cx: &mut Context<Self>) {
        self.execution_mode_menu_open = !self.execution_mode_menu_open;
        cx.notify();
    }

    pub(crate) fn choose_execution_run_mode(
        &mut self,
        mode: ExecutionRunMode,
        cx: &mut Context<Self>,
    ) {
        self.execution_mode_menu_open = false;
        self.set_execution_run_mode(mode, cx);
    }

    pub(crate) fn select_queue_overlay_tab(
        &mut self,
        tab: QueueOverlayTab,
        cx: &mut Context<Self>,
    ) {
        self.queue_overlay_tab = tab;
        self.queue_details_attempt = None;
        cx.notify();
    }

    pub(crate) fn close_queue_overlay(&mut self, cx: &mut Context<Self>) {
        if self.queue_overlay_visible {
            self.queue_overlay_visible = false;
            self.queue_details_attempt = None;
            cx.notify();
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn queue_overlay_attempt_ids_for_test(
        &self,
        cx: &App,
    ) -> Vec<sim_runtime::AttemptId> {
        self.execution_snapshot(cx)
            .map(|snapshot| {
                crate::graph_render::filtered_queue_overlay_attempts(
                    &snapshot,
                    self.queue_overlay_tab,
                )
                .into_iter()
                .map(|attempt| attempt.attempt_id)
                .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn toggle_queue_attempt_details(
        &mut self,
        attempt_id: sim_runtime::AttemptId,
        cx: &mut Context<Self>,
    ) {
        self.queue_details_attempt = if self.queue_details_attempt == Some(attempt_id) {
            None
        } else {
            Some(attempt_id)
        };
        cx.notify();
    }

    pub(crate) fn copy_queue_attempt_id(
        &mut self,
        attempt_id: sim_runtime::AttemptId,
        cx: &mut Context<Self>,
    ) {
        cx.write_to_clipboard(ClipboardItem::new_string(attempt_id.0.to_string()));
        self.model.announcement = Some("Copied native execution job ID".to_owned());
        cx.notify();
    }

    fn current_graph_navigation(&self) -> Option<Vec<GraphIdentifier>> {
        self.model
            .document()
            .map(|document| document.navigation.clone())
    }

    fn invalidate_execution_projection_after_navigation(
        &mut self,
        navigation_before: Option<Vec<GraphIdentifier>>,
    ) {
        if navigation_before != self.current_graph_navigation() {
            self.execution_projection_navigation = None;
        }
    }

    pub(crate) fn copy_execution_error(&mut self, cx: &mut Context<Self>) {
        let Some(attempt) = self.active_execution_presentation(cx) else {
            self.model
                .report_error("no native execution attempt is associated with this graph");
            cx.notify();
            return;
        };
        let Some(failure) = attempt.failure else {
            self.model
                .report_error("the associated native execution attempt has no error");
            cx.notify();
            return;
        };
        let mut text = format!(
            "origin: {:?}\n{}: {}\nretryable: {}",
            failure.origin, failure.code, failure.message, failure.retryable
        );
        if let Some(node_id) = failure.node_id {
            text.push_str(&format!("\nnode: {}", node_id.0));
        }
        for (key, value) in failure.details {
            text.push_str(&format!("\n{key}: {value}"));
        }
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        self.model.announcement = Some("Copied structured execution error".to_owned());
        cx.notify();
    }

    pub(crate) fn locate_active_execution_error(&mut self, cx: &mut Context<Self>) {
        let node_id = self
            .active_execution_presentation(cx)
            .and_then(|attempt| attempt.failure)
            .and_then(|failure| failure.node_id);
        match node_id {
            Some(node_id) => {
                if let Err(error) = self.locate_execution_node(&node_id, cx) {
                    self.model.report_error(error);
                    cx.notify();
                }
            }
            None => {
                self.model
                    .report_error("the structured execution error has no node location");
                cx.notify();
            }
        }
    }

    pub(crate) fn dismiss_active_execution_error(&mut self, cx: &mut Context<Self>) {
        self.dismissed_execution_error = self
            .active_execution_presentation(cx)
            .map(|attempt| attempt.attempt_id);
        self.model.announcement = Some("Dismissed execution error overlay".to_owned());
        cx.notify();
    }

    pub(crate) fn execution_error_is_dismissed(&self, attempt_id: sim_runtime::AttemptId) -> bool {
        self.dismissed_execution_error == Some(attempt_id)
    }

    pub(crate) fn control_focus_handle(
        &mut self,
        identifier: impl Into<String>,
        cx: &mut Context<Self>,
    ) -> FocusHandle {
        let identifier = identifier.into();
        self.active_control_focus_handles.insert(identifier.clone());
        self.control_focus_handles
            .entry(identifier)
            .or_insert_with(|| cx.focus_handle())
            .clone()
    }

    #[cfg(feature = "test-support")]
    pub fn control_focus_handle_for_test(
        &mut self,
        control: &str,
        cx: &mut Context<Self>,
    ) -> FocusHandle {
        self.control_focus_handle(control, cx)
    }

    pub(crate) fn begin_control_focus_handle_render(&mut self) {
        self.active_control_focus_handles.clear();
    }

    pub(crate) fn finish_control_focus_handle_render(&mut self) {
        self.control_focus_handles
            .retain(|identifier, _| self.active_control_focus_handles.contains(identifier));
    }

    pub fn locate_execution_node(
        &mut self,
        node_id: &NodeId,
        cx: &mut Context<Self>,
    ) -> Result<(), GraphWorkspaceError> {
        let (identifier, prior_selection, prior_viewport, target_viewport) = {
            let document = self.model.document().ok_or(GraphWorkspaceError::ReadOnly)?;
            let graph = document.active_graph()?;
            let (identifier, node) = graph
                .nodes
                .iter()
                .find(|(identifier, _)| identifier.text() == node_id.0)
                .ok_or_else(|| {
                    GraphWorkspaceError::Persistence(format!(
                        "execution node {} is not present in the active graph",
                        node_id.0
                    ))
                })?;
            let mut target_viewport = graph.viewport.clone();
            target_viewport.offset = GraphPoint {
                x: 240.0 - (node.position.x + node.size.width / 2.0) * target_viewport.scale,
                y: 160.0 - (node.position.y + node.size.height / 2.0) * target_viewport.scale,
            };
            (
                identifier.clone(),
                graph.selection.clone(),
                graph.viewport.clone(),
                target_viewport,
            )
        };
        let mut selection = GraphSelection::default();
        selection.nodes.insert(identifier);
        self.model
            .replace_ephemeral_graph_state(selection, target_viewport)?;
        self.execution_navigation.push(ExecutionNavigationState {
            selection: prior_selection,
            viewport: prior_viewport,
        });
        if self.execution_navigation.len() > 16 {
            self.execution_navigation.remove(0);
        }
        self.model.announcement = Some(format!(
            "Located execution error at node {}; restore navigation to return",
            node_id.0
        ));
        cx.emit(ItemEvent::UpdateTab);
        cx.notify();
        Ok(())
    }

    pub fn restore_execution_navigation(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Result<bool, GraphWorkspaceError> {
        let Some(state) = self.execution_navigation.pop() else {
            return Ok(false);
        };
        self.model
            .replace_ephemeral_graph_state(state.selection, state.viewport)?;
        self.model.announcement = Some("Restored the previous graph view".to_owned());
        cx.emit(ItemEvent::UpdateTab);
        cx.notify();
        Ok(true)
    }

    pub(crate) fn restore_execution_navigation_action(
        &mut self,
        _: &crate::RestoreExecutionNavigation,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match self.restore_execution_navigation(cx) {
            Ok(true) => {}
            Ok(false) => {
                self.model.announcement = Some("No execution navigation to restore".to_owned());
                cx.notify();
            }
            Err(error) => {
                self.model.report_error(error);
                cx.notify();
            }
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn control_focus_handle_count(&self) -> usize {
        self.control_focus_handles.len()
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn contains_control_focus_handle(&self, identifier: &str) -> bool {
        self.control_focus_handles.contains_key(identifier)
    }

    fn local_file_path(&self) -> anyhow::Result<PathBuf> {
        match self.model.save_coordinator.provider() {
            WorkflowStorageProvider::LocalFile => {
                let identity = self.model.save_coordinator.document_identity().trim();
                if identity.is_empty() {
                    Err(anyhow!("local workflow path is empty"))
                } else {
                    Ok(PathBuf::from(identity))
                }
            }
            WorkflowStorageProvider::Draft => Err(anyhow!(
                "draft workflow has no local path; use Save As to choose one"
            )),
            WorkflowStorageProvider::Provider { identifier } => Err(anyhow!(
                "workflow provider `{identifier}` does not support native local save"
            )),
        }
    }

    pub fn resolve_close_request(
        &mut self,
        choice: GraphDirtyCloseChoice,
        cx: &mut Context<Self>,
    ) -> GraphCloseDisposition {
        let dirty = matches!(
            self.model.save_coordinator.authority(),
            WorkflowAuthority::LocalDirty
                | WorkflowAuthority::Conflict
                | WorkflowAuthority::ExternalMissing
                | WorkflowAuthority::SavePrepared
                | WorkflowAuthority::Interrupted
        );
        let disposition = if !dirty {
            GraphCloseDisposition::CloseWithoutSaving
        } else {
            match choice {
                GraphDirtyCloseChoice::Save
                    if !self.model.is_read_only()
                        && matches!(
                            self.model.save_coordinator.provider(),
                            WorkflowStorageProvider::Draft | WorkflowStorageProvider::LocalFile
                        ) =>
                {
                    GraphCloseDisposition::SaveThenClose
                }
                GraphDirtyCloseChoice::Save => {
                    self.model.report_error(
                        "dirty workflow cannot be saved by its current storage provider",
                    );
                    GraphCloseDisposition::KeepOpen
                }
                GraphDirtyCloseChoice::Discard => GraphCloseDisposition::CloseWithoutSaving,
                GraphDirtyCloseChoice::Cancel => GraphCloseDisposition::KeepOpen,
            }
        };
        self.model.announcement = Some(match disposition {
            GraphCloseDisposition::SaveThenClose => "Save workflow before closing".to_owned(),
            GraphCloseDisposition::CloseWithoutSaving => {
                "Close workflow without another save".to_owned()
            }
            GraphCloseDisposition::KeepOpen => "Workflow close cancelled".to_owned(),
        });
        cx.notify();
        disposition
    }

    pub fn rename_workflow(&mut self, title: impl Into<String>, cx: &mut Context<Self>) -> bool {
        let title = title.into();
        let title = title.trim();
        if title.is_empty() {
            self.model.report_error("workflow title cannot be empty");
            cx.notify();
            return false;
        }
        self.model.title = title.to_owned();
        self.model.announcement = Some(format!("Renamed workflow to {title}"));
        cx.emit(ItemEvent::UpdateTab);
        cx.notify();
        true
    }

    pub fn export_to_path(
        &mut self,
        project: Entity<Project>,
        path: ProjectPath,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        self.export_to_path_with_overwrite(project, path, GraphOverwriteChoice::RejectExisting, cx)
    }

    pub fn export_to_path_with_overwrite(
        &mut self,
        project: Entity<Project>,
        path: ProjectPath,
        overwrite: GraphOverwriteChoice,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        let absolute_path = match project.read(cx).absolute_path(&path, cx) {
            Some(path) => path,
            None => {
                let result = Err(anyhow!("failed to resolve workflow export path"));
                self.report_storage_result(&result, "Workflow exported", cx);
                return Task::ready(result);
            }
        };
        let bytes = match &self.model.open_state {
            crate::WorkflowOpenState::Editable(engine) => engine.document.to_workflow_bytes(),
            crate::WorkflowOpenState::ReadOnly { original_bytes, .. } => Ok(original_bytes.clone()),
        };
        let text = match bytes
            .map_err(anyhow::Error::from)
            .and_then(|bytes| String::from_utf8(bytes).map_err(anyhow::Error::from))
        {
            Ok(text) => text,
            Err(error) => {
                let result = Err(error.context("workflow export is not valid UTF-8 JSON"));
                self.report_storage_result(&result, "Workflow exported", cx);
                return Task::ready(result);
            }
        };
        let fs = project.read(cx).fs().clone();
        let operation_id = uuid::Uuid::new_v4();
        cx.spawn(async move |this, cx| {
            let result: anyhow::Result<()> = async {
                match overwrite {
                    GraphOverwriteChoice::RejectExisting => {
                        atomic_create_workflow_file(&fs, &absolute_path, &text, operation_id).await
                    }
                    GraphOverwriteChoice::ReplaceExisting => fs
                        .atomic_write(absolute_path.clone(), text)
                        .await
                        .with_context(|| {
                            format!("failed to export workflow to {}", absolute_path.display())
                        }),
                }
            }
            .await;
            this.update(cx, |this, cx| {
                this.report_storage_result(&result, "Workflow exported", cx)
            })?;
            result
        })
    }

    pub fn rename_local_file(
        &mut self,
        project: Entity<Project>,
        path: ProjectPath,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        let source = match self.local_file_path() {
            Ok(path) => path,
            Err(error) => {
                let result = Err(error);
                self.report_storage_result(&result, "Workflow file renamed", cx);
                return Task::ready(result);
            }
        };
        let target = match project.read(cx).absolute_path(&path, cx) {
            Some(path) => path,
            None => {
                let result = Err(anyhow!("failed to resolve renamed workflow path"));
                self.report_storage_result(&result, "Workflow file renamed", cx);
                return Task::ready(result);
            }
        };
        let target_identity = match Self::path_identity(&target) {
            Ok(identity) => identity,
            Err(error) => {
                let result = Err(error);
                self.report_storage_result(&result, "Workflow file renamed", cx);
                return Task::ready(result);
            }
        };
        let mut retarget_preflight = self.model.save_coordinator.clone();
        if let Err(error) = retarget_preflight.retarget_local_file(target_identity.clone()) {
            let result = Err(anyhow::Error::from(error));
            self.report_storage_result(&result, "Workflow file renamed", cx);
            return Task::ready(result);
        }
        let title = target
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned);
        let fs = project.read(cx).fs().clone();
        let watch_fs = fs.clone();
        cx.spawn(async move |this, cx| {
            let result: anyhow::Result<()> = async {
                if source != target && fs.is_file(&target).await {
                    return Err(anyhow!(
                        "renamed workflow target already exists: {}",
                        target.display()
                    ));
                }
                fs.rename(
                    &source,
                    &target,
                    RenameOptions {
                        create_parents: true,
                        ..RenameOptions::default()
                    },
                )
                .await
                .with_context(|| {
                    format!(
                        "failed to rename workflow {} to {}",
                        source.display(),
                        target.display()
                    )
                })?;
                let retarget_result = this.update(cx, |this, cx| {
                    this.model
                        .save_coordinator
                        .retarget_local_file(target_identity)?;
                    if let Some(title) = title {
                        this.model.title = title;
                    }
                    this.start_local_file_watch(watch_fs, target.clone(), cx);
                    cx.emit(ItemEvent::UpdateTab);
                    Ok::<(), GraphWorkspaceError>(())
                });
                match retarget_result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        if let Err(rollback_error) = fs
                            .rename(&target, &source, RenameOptions::default())
                            .await
                        {
                            return Err(anyhow!(
                                "workflow rename state update failed: {error}; rollback from {} to {} failed: {rollback_error}",
                                target.display(),
                                source.display()
                            ));
                        }
                        return Err(anyhow::Error::from(error));
                    }
                    Err(error) => {
                        if let Err(rollback_error) = fs
                            .rename(&target, &source, RenameOptions::default())
                            .await
                        {
                            return Err(anyhow!(
                                "workflow rename item update failed: {error}; rollback from {} to {} failed: {rollback_error}",
                                target.display(),
                                source.display()
                            ));
                        }
                        return Err(error);
                    }
                }
                Ok(())
            }
            .await;
            this.update(cx, |this, cx| {
                this.report_storage_result(&result, "Workflow file renamed", cx)
            })?;
            result
        })
    }

    pub fn delete_local_file(
        &mut self,
        project: Entity<Project>,
        choice: GraphDeleteFileChoice,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        if choice == GraphDeleteFileChoice::Cancel {
            self.model.announcement = Some("Workflow deletion cancelled".to_owned());
            cx.notify();
            return Task::ready(Ok(()));
        }
        let path = match self.local_file_path() {
            Ok(path) => path,
            Err(error) => {
                let result = Err(error);
                self.report_storage_result(&result, "Workflow file deleted", cx);
                return Task::ready(result);
            }
        };
        let fs = project.read(cx).fs().clone();
        cx.spawn(async move |this, cx| {
            let result: anyhow::Result<()> = async {
                fs.remove_file(&path, RemoveOptions::default())
                    .await
                    .with_context(|| format!("failed to delete workflow {}", path.display()))?;
                this.update(cx, |this, cx| {
                    this.file_watch = None;
                    let document_identity = this
                        .model
                        .document()
                        .map(|document| document.document_identity.to_string())
                        .unwrap_or_else(|| {
                            this.model.save_coordinator.document_identity().to_owned()
                        });
                    this.model
                        .save_coordinator
                        .detach_local_file_to_draft(document_identity)?;
                    cx.emit(ItemEvent::UpdateTab);
                    Ok::<(), GraphWorkspaceError>(())
                })??;
                Ok(())
            }
            .await;
            this.update(cx, |this, cx| {
                this.report_storage_result(&result, "Workflow file deleted", cx)
            })?;
            result
        })
    }

    pub fn conflict_comparison(&self) -> GraphConflictComparison {
        let comparison = self.model.save_coordinator.comparison();
        GraphConflictComparison {
            base: comparison.base.to_vec(),
            local: comparison.local.to_vec(),
            external: comparison.external.map(|bytes| bytes.to_vec()),
        }
    }

    pub fn keep_local_version(&mut self, cx: &mut Context<Self>) -> bool {
        let result = self.model.keep_local();
        let succeeded = result.is_ok();
        if succeeded {
            cx.emit(ItemEvent::UpdateTab);
        }
        cx.notify();
        succeeded
    }

    fn report_storage_result(
        &mut self,
        result: &anyhow::Result<()>,
        success: &str,
        cx: &mut Context<Self>,
    ) {
        match result {
            Ok(()) => {
                self.model.last_error = None;
                self.model.announcement = Some(success.to_owned());
                cx.emit(ItemEvent::UpdateTab);
            }
            Err(error) => {
                self.model.report_error(error);
                cx.emit(ItemEvent::UpdateTab);
            }
        }
        cx.notify();
    }

    fn path_identity(path: &Path) -> anyhow::Result<String> {
        path.to_str()
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("workflow path is not valid UTF-8: {}", path.display()))
    }

    fn start_local_file_watch(
        &mut self,
        fs: Arc<dyn project::Fs>,
        path: PathBuf,
        cx: &mut Context<Self>,
    ) {
        let watch_path = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.clone());
        self.file_watch = Some(cx.spawn(async move |this, cx| {
            let (mut events, watcher) = fs.watch(&watch_path, Duration::from_millis(100)).await;
            let _watcher = watcher;
            while let Some(events) = events.next().await {
                if !events.iter().any(|event| event.path == path) {
                    continue;
                }
                match fs.load_bytes(&path).await {
                    Ok(bytes) => {
                        if this
                            .update(cx, |this, cx| {
                                match this.model.observe_external_change(bytes) {
                                    Ok(()) => {
                                        cx.emit(ItemEvent::UpdateTab);
                                        cx.notify();
                                    }
                                    Err(error) => {
                                        this.model.report_error(error);
                                        cx.notify();
                                    }
                                }
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(error) => {
                        let missing = !fs.is_file(&path).await;
                        if this
                            .update(cx, |this, cx| {
                                if missing {
                                    match this.model.observe_external_deletion() {
                                        Ok(()) => cx.emit(ItemEvent::UpdateTab),
                                        Err(error) => {
                                            this.model.report_error(error);
                                            cx.emit(ItemEvent::UpdateTab);
                                        }
                                    }
                                } else {
                                    this.model.report_error(format!(
                                        "failed to read externally changed workflow {}: {error}",
                                        path.display()
                                    ));
                                    cx.emit(ItemEvent::UpdateTab);
                                }
                                cx.notify();
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        }));
    }

    pub fn execute_catalog_action(
        &mut self,
        action: CatalogGraphAction,
        input: GraphActionInput,
        cx: &mut Context<Self>,
    ) -> bool {
        let requires_group_padding = matches!(
            (&action, &input),
            (
                CatalogGraphAction::GroupSelectedNodes,
                GraphActionInput::None
            ) | (
                CatalogGraphAction::FitGroupToContents,
                GraphActionInput::None | GraphActionInput::GroupIdentifier(_)
            )
        );
        if requires_group_padding && !cx.has_global::<SettingsStore>() {
            self.model.report_error("Zed settings store is unavailable");
            cx.notify();
            return false;
        }
        let input = match (action, input) {
            (CatalogGraphAction::GroupSelectedNodes, GraphActionInput::None) => {
                GraphActionInput::Group {
                    title: "Group".to_owned(),
                    padding: self.group_selected_nodes_padding(cx),
                }
            }
            (
                CatalogGraphAction::FitGroupToContents,
                GraphActionInput::GroupIdentifier(identifier),
            ) => GraphActionInput::FitGroup {
                identifier,
                padding: self.group_selected_nodes_padding(cx),
            },
            (CatalogGraphAction::FitGroupToContents, GraphActionInput::None) => {
                let Some(identifier) = self
                    .model
                    .selection()
                    .and_then(|selection| selection.groups.iter().next())
                    .cloned()
                else {
                    self.model
                        .report_error("graph selection contains no group to fit");
                    cx.notify();
                    return false;
                };
                GraphActionInput::FitGroup {
                    identifier,
                    padding: self.group_selected_nodes_padding(cx),
                }
            }
            (_, input) => input,
        };
        if action == CatalogGraphAction::ToggleVueNodes {
            return match GraphCommandModel::execute(&mut self.model, action, input) {
                Err(GraphActionError::RequiresSettingsStore(_)) => {
                    self.toggle_native_node_renderer(cx)
                }
                Err(error) => {
                    self.model.report_error(error);
                    cx.notify();
                    false
                }
                Ok(_) => {
                    self.model
                        .report_error("settings-owned action was accepted by the graph model");
                    cx.notify();
                    false
                }
            };
        }
        let navigation_before = self.current_graph_navigation();
        match GraphCommandModel::execute(&mut self.model, action, input) {
            Ok(effect) => {
                self.invalidate_execution_projection_after_navigation(navigation_before);
                if let GraphActionEffect::ClipboardText(text) = effect {
                    cx.write_to_clipboard(ClipboardItem::new_string(text));
                }
                if action_mutates_workflow(action) {
                    cx.emit(ItemEvent::Edit);
                    self.schedule_automatic_execution(cx);
                } else if matches!(
                    action,
                    CatalogGraphAction::ToggleCanvasInfo | CatalogGraphAction::ToggleVueNodes
                ) {
                    cx.emit(ItemEvent::UpdateTab);
                }
                cx.notify();
                true
            }
            Err(error) => {
                self.model.report_error(error);
                cx.notify();
                false
            }
        }
    }

    pub(crate) fn native_node_renderer_enabled(&self, cx: &App) -> bool {
        cx.try_global::<SettingsStore>()
            .and_then(|store| store.merged_settings().sim_runtime.as_ref())
            .and_then(|settings| settings.native_node_renderer)
            .unwrap_or(true)
    }

    pub(crate) fn group_selected_nodes_padding(&self, cx: &App) -> f32 {
        cx.try_global::<SettingsStore>()
            .and_then(|store| store.merged_settings().sim_runtime.as_ref())
            .and_then(|settings| settings.group_selected_nodes_padding)
            .filter(|padding| padding.is_finite())
            .unwrap_or(10.0)
            .clamp(0.0, 100.0)
    }

    fn toggle_native_node_renderer(&mut self, cx: &mut Context<Self>) -> bool {
        if self.context_settings_task.is_some() {
            self.model
                .report_error("another Sim settings update is still in progress");
            cx.notify();
            return false;
        }
        if !cx.has_global::<SettingsStore>() {
            self.model.report_error("Zed settings store is unavailable");
            cx.notify();
            return false;
        }
        let enabled = !self.native_node_renderer_enabled(cx);
        let completion = settings::update_settings_file_with_completion(
            <dyn fs::Fs>::global(cx),
            cx,
            move |settings, _| {
                let sim_runtime = settings.sim_runtime.get_or_insert_default();
                sim_runtime.native_node_renderer = Some(enabled);
            },
        );
        self.context_settings_task = Some(cx.spawn(async move |this, cx| {
            let result = completion.await;
            if let Err(error) = this.update(cx, |this, cx| {
                this.context_settings_task = None;
                match result {
                    Ok(Ok(())) => {
                        this.model.announcement = Some(if enabled {
                            "Detailed native node renderer enabled".to_owned()
                        } else {
                            "Compact native node renderer enabled".to_owned()
                        });
                        cx.emit(ItemEvent::UpdateTab);
                    }
                    Ok(Err(error)) => this
                        .model
                        .report_error(format!("failed to update native node renderer: {error}")),
                    Err(error) => this.model.report_error(format!(
                        "native node renderer settings update was cancelled: {error}"
                    )),
                }
                cx.notify();
            }) {
                log::error!(
                    "graph disappeared while completing native node renderer settings update: {error}"
                );
            }
        }));
        self.model.announcement = Some("Updating native node renderer".to_owned());
        cx.notify();
        true
    }

    pub fn dispatch_shell_command(
        &mut self,
        command_id: &str,
        cx: &mut Context<Self>,
    ) -> CommandDispatchOutcome {
        #[cfg(feature = "test-support")]
        self.shell_dispatch_trace.push(command_id.to_owned());
        let Some(registration) = command_registration(command_id) else {
            let outcome = CommandDispatchOutcome::Unknown {
                command_id: command_id.to_owned(),
            };
            self.report_shell_outcome(&outcome, cx);
            return outcome;
        };

        match registration.status {
            CommandNativeStatus::Infrastructure { owner } => {
                let outcome = CommandDispatchOutcome::Infrastructure {
                    command_id: command_id.to_owned(),
                    owner,
                };
                self.report_shell_outcome(&outcome, cx);
                return outcome;
            }
            CommandNativeStatus::RequiresInput { input, .. } => {
                let outcome = CommandDispatchOutcome::RequiresInput {
                    command_id: command_id.to_owned(),
                    input,
                };
                self.report_shell_outcome(&outcome, cx);
                return outcome;
            }
            CommandNativeStatus::LaterOwned { owner } => {
                let outcome = CommandDispatchOutcome::LaterOwned {
                    command_id: command_id.to_owned(),
                    owner,
                };
                self.report_shell_outcome(&outcome, cx);
                return outcome;
            }
            CommandNativeStatus::Gated { owner, gate } => {
                let outcome = CommandDispatchOutcome::Gated {
                    command_id: command_id.to_owned(),
                    owner,
                    gate,
                };
                self.report_shell_outcome(&outcome, cx);
                return outcome;
            }
            CommandNativeStatus::Legacy { owner } => {
                let outcome = CommandDispatchOutcome::Legacy {
                    command_id: command_id.to_owned(),
                    owner,
                };
                self.report_shell_outcome(&outcome, cx);
                return outcome;
            }
            CommandNativeStatus::Executable => {}
        }

        if matches!(
            command_id,
            "Sim.ClearPendingTasks"
                | "Sim.Interrupt"
                | "Sim.QueuePrompt"
                | "Sim.QueuePromptFront"
                | "Sim.QueueSelectedOutputNodes"
                | "Sim.Queue.ToggleOverlay"
                | "Sim.ToggleQPOV2"
        ) {
            return self.dispatch_execution_command(command_id, cx);
        }

        if command_id == "Sim.PublishSubgraph" {
            let outcome = CommandDispatchOutcome::RequiresInput {
                command_id: command_id.to_owned(),
                input: "subgraph blueprint name through the focused GPUI action",
            };
            self.report_shell_outcome(&outcome, cx);
            return outcome;
        }

        if command_id == "Sim.SaveWorkflow" {
            return self.reject_shell_command(
                command_id,
                "saving a workflow requires its focused GPUI action and owning workspace",
                cx,
            );
        }

        if command_id == "Sim.Undo" {
            let navigation_before = self.current_graph_navigation();
            return match self.model.undo() {
                Ok(true) => {
                    self.invalidate_execution_projection_after_navigation(navigation_before);
                    cx.emit(ItemEvent::Edit);
                    cx.notify();
                    CommandDispatchOutcome::Executed {
                        command_id: command_id.to_owned(),
                    }
                }
                Ok(false) => self.reject_shell_command(command_id, "nothing to undo", cx),
                Err(error) => self.reject_shell_command(command_id, error, cx),
            };
        }
        if command_id == "Sim.Redo" {
            let navigation_before = self.current_graph_navigation();
            return match self.model.redo() {
                Ok(true) => {
                    self.invalidate_execution_projection_after_navigation(navigation_before);
                    cx.emit(ItemEvent::Edit);
                    cx.notify();
                    CommandDispatchOutcome::Executed {
                        command_id: command_id.to_owned(),
                    }
                }
                Ok(false) => self.reject_shell_command(command_id, "nothing to redo", cx),
                Err(error) => self.reject_shell_command(command_id, error, cx),
            };
        }

        let Some(action) = registration.graph_action else {
            return self.reject_shell_command(
                command_id,
                "executable command has no native graph action",
                cx,
            );
        };
        let succeeded = match action {
            CatalogGraphAction::PasteFromClipboard => self.paste_from_clipboard(false, cx),
            CatalogGraphAction::PasteFromClipboardWithConnect => {
                self.paste_from_clipboard(true, cx)
            }
            _ => self.execute_catalog_action(action, GraphActionInput::None, cx),
        };
        if succeeded {
            CommandDispatchOutcome::Executed {
                command_id: command_id.to_owned(),
            }
        } else {
            let error = self
                .model
                .last_error
                .clone()
                .unwrap_or_else(|| "native graph action was rejected".to_owned());
            CommandDispatchOutcome::Rejected {
                command_id: command_id.to_owned(),
                error,
            }
        }
    }

    fn dispatch_execution_command(
        &mut self,
        command_id: &str,
        cx: &mut Context<Self>,
    ) -> CommandDispatchOutcome {
        let result = match command_id {
            "Sim.QueuePrompt" => self.queue_native_execution(false, false, cx),
            "Sim.QueuePromptFront" => self.queue_native_execution(false, true, cx),
            "Sim.QueueSelectedOutputNodes" => self.queue_native_execution(true, false, cx),
            "Sim.Interrupt" => self.interrupt_native_execution(cx),
            "Sim.ClearPendingTasks" => self.dispatch_native_execution_control(
                ExecutionControlCommandKind::ClearPending {
                    reason: "cleared from the native graph command".to_owned(),
                },
                cx,
            ),
            "Sim.Queue.ToggleOverlay" => {
                self.queue_overlay_visible = !self.queue_overlay_visible;
                self.model.announcement = Some(if self.queue_overlay_visible {
                    "Showing native execution overlay".to_owned()
                } else {
                    "Hiding native execution overlay".to_owned()
                });
                cx.notify();
                Ok(())
            }
            "Sim.ToggleQPOV2" => {
                self.qpov2_enabled = !self.qpov2_enabled;
                self.model.announcement = Some(if self.qpov2_enabled {
                    "Native execution panel enabled".to_owned()
                } else {
                    "Native execution panel hidden".to_owned()
                });
                cx.notify();
                Ok(())
            }
            _ => Err(format!("unknown native execution command `{command_id}`")),
        };
        match result {
            Ok(()) => CommandDispatchOutcome::Executed {
                command_id: command_id.to_owned(),
            },
            Err(error) => self.reject_shell_command(command_id, error, cx),
        }
    }

    fn execution_profile_id(&self) -> Result<ProfileId, String> {
        self.model
            .document()
            .and_then(|document| document.profile_identity)
            .map(ProfileId)
            .ok_or_else(|| {
                "the workflow has no native execution profile; select a profile before queuing"
                    .to_owned()
            })
    }

    fn queue_native_execution(
        &mut self,
        selected_outputs_only: bool,
        front: bool,
        cx: &mut Context<Self>,
    ) -> Result<(), String> {
        if let Some(reason) = self.execution_queue_unavailable_reason(cx) {
            return Err(format!("native execution is unavailable: {reason}"));
        }
        let profile_id = self.execution_profile_id()?;
        let execution_model = self
            .execution_model
            .clone()
            .ok_or_else(|| "the native execution service is unavailable".to_owned())?;
        let selected_output_nodes = if selected_outputs_only {
            self.model
                .selected_node_identifiers()
                .into_iter()
                .map(|identifier| NodeId(identifier.text()))
                .collect()
        } else {
            BTreeSet::new()
        };
        if selected_outputs_only && selected_output_nodes.is_empty() {
            return Err("select at least one output node before queuing".to_owned());
        }
        let workflow_bytes = self
            .model
            .document()
            .ok_or_else(|| "the workflow is not editable".to_owned())?
            .to_workflow_bytes()
            .map_err(|error| format!("failed to serialize the native workflow: {error}"))?;
        let request = ExecutionPlanRequest {
            profile_id,
            document_identity: self.model.save_coordinator.document_identity().to_owned(),
            workflow_bytes,
            selected_output_nodes,
        };
        let acknowledgement = execution_model
            .update(cx, |model, cx| model.queue(request, 0, front, cx))
            .map_err(|error| error.to_string())?;
        match acknowledgement.outcome {
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(attempt_id),
            } => {
                self.associate_execution_attempt(attempt_id, cx);
                self.model.announcement =
                    Some(format!("Queued native execution attempt {}", attempt_id.0));
                Ok(())
            }
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: None,
            } => Err("native queue acknowledgement omitted the attempt identity".to_owned()),
            ExecutionCommandOutcome::Rejected { failure } => {
                Err(format!("{}: {}", failure.code, failure.message))
            }
        }
    }

    fn interrupt_native_execution(&mut self, cx: &mut Context<Self>) -> Result<(), String> {
        if let Some(reason) = self.execution_control_unavailable_reason(cx) {
            return Err(format!("native execution control is unavailable: {reason}"));
        }
        let attempt_id = self
            .model
            .execution_association
            .as_deref()
            .and_then(|identity| uuid::Uuid::parse_str(identity).ok())
            .map(sim_runtime::AttemptId)
            .ok_or_else(|| {
                "this workflow has no associated native execution attempt to interrupt".to_owned()
            })?;
        let attempt_state = self
            .execution_snapshot(cx)
            .and_then(|snapshot| {
                snapshot
                    .attempts
                    .into_iter()
                    .find(|attempt| attempt.attempt_id == attempt_id)
            })
            .map(|attempt| attempt.state)
            .ok_or_else(|| {
                "the associated native execution attempt is no longer available".to_owned()
            })?;
        if attempt_state != AttemptState::Running {
            return Err(format!(
                "native interrupt is unavailable while the associated attempt is {attempt_state:?}"
            ));
        }
        self.dispatch_native_execution_control(
            ExecutionControlCommandKind::Interrupt {
                attempt_id,
                reason: "interrupted from the associated native graph".to_owned(),
            },
            cx,
        )?;
        self.model.announcement = Some("Native interrupt acknowledged".to_owned());
        cx.notify();
        Ok(())
    }

    fn dispatch_native_execution_control(
        &mut self,
        kind: ExecutionControlCommandKind,
        cx: &mut Context<Self>,
    ) -> Result<(), String> {
        if let Some(reason) = self.execution_control_unavailable_reason(cx) {
            return Err(format!("native execution control is unavailable: {reason}"));
        }
        let profile_id = self.execution_profile_id()?;
        let execution_model = self
            .execution_model
            .clone()
            .ok_or_else(|| "the native execution service is unavailable".to_owned())?;
        let acknowledgement = execution_model
            .update(cx, |model, cx| model.dispatch(profile_id, kind, cx))
            .map_err(|error| error.to_string())?;
        match acknowledgement.outcome {
            ExecutionCommandOutcome::Accepted { .. } => {
                self.model.announcement = Some("Native execution command acknowledged".to_owned());
                cx.notify();
                Ok(())
            }
            ExecutionCommandOutcome::Rejected { failure } => {
                Err(format!("{}: {}", failure.code, failure.message))
            }
        }
    }

    pub(crate) fn set_execution_run_mode(
        &mut self,
        mode: ExecutionRunMode,
        cx: &mut Context<Self>,
    ) {
        self.automatic_execution_task = None;
        if mode != ExecutionRunMode::Manual
            && let Some(reason) = self.execution_queue_unavailable_reason(cx)
        {
            self.execution_run_mode = ExecutionRunMode::Manual;
            self.model
                .report_error(format!("automatic execution is unavailable: {reason}"));
            cx.notify();
            return;
        }
        self.execution_run_mode = mode;
        self.model.announcement = Some(match mode {
            ExecutionRunMode::Manual => "Execution mode: manual".to_owned(),
            ExecutionRunMode::OnChange => "Execution mode: queue on change".to_owned(),
            ExecutionRunMode::InstantIdle => "Execution mode: queue when idle".to_owned(),
        });
        cx.notify();
    }

    fn reconcile_execution_capability(&mut self, cx: &mut Context<Self>) {
        if self.execution_run_mode == ExecutionRunMode::Manual {
            return;
        }
        let Some(reason) = self.execution_queue_unavailable_reason(cx) else {
            return;
        };
        self.automatic_execution_task = None;
        self.execution_run_mode = ExecutionRunMode::Manual;
        self.model.report_error(format!(
            "automatic execution returned to manual mode: {reason}"
        ));
        cx.notify();
    }

    fn schedule_automatic_execution(&mut self, cx: &mut Context<Self>) {
        let delay = match self.execution_run_mode {
            ExecutionRunMode::Manual => {
                self.automatic_execution_task = None;
                return;
            }
            ExecutionRunMode::OnChange => Duration::from_millis(500),
            ExecutionRunMode::InstantIdle => Duration::from_millis(100),
        };
        self.automatic_execution_task = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(delay).await;
            if let Err(error) = this.update(cx, |this, cx| {
                let is_busy = this
                    .execution_profile_id()
                    .ok()
                    .and_then(|profile_id| {
                        this.execution_model.as_ref().and_then(|model| {
                            model.read(cx).snapshot(profile_id).ok().map(|snapshot| {
                                !snapshot.queue.is_empty()
                                    || snapshot
                                        .attempts
                                        .iter()
                                        .any(|attempt| !attempt.state.is_terminal())
                            })
                        })
                    })
                    .unwrap_or(false);
                if is_busy {
                    this.model.announcement = Some(
                        "Automatic execution deferred while another attempt is active".to_owned(),
                    );
                    cx.notify();
                    return;
                }
                if let Err(error) = this.queue_native_execution(false, false, cx) {
                    this.model
                        .report_error(format!("automatic native execution failed: {error}"));
                    cx.notify();
                }
            }) {
                log::error!("graph closed before scheduled native execution: {error}");
            }
        }));
    }

    #[cfg(feature = "test-support")]
    pub fn shell_dispatch_trace_for_test(&self) -> &[String] {
        &self.shell_dispatch_trace
    }

    fn report_shell_outcome(&mut self, outcome: &CommandDispatchOutcome, cx: &mut Context<Self>) {
        let message = match outcome {
            CommandDispatchOutcome::Infrastructure { command_id, owner } => format!(
                "command `{command_id}` is infrastructure-only and cannot be invoked directly; owner: {owner}"
            ),
            CommandDispatchOutcome::RequiresInput { command_id, input } => {
                format!("command `{command_id}` requires {input}")
            }
            CommandDispatchOutcome::LaterOwned { command_id, owner } => {
                format!(
                    "command `{command_id}` is not yet available; implementation owner: {owner}"
                )
            }
            CommandDispatchOutcome::Gated {
                command_id,
                owner,
                gate,
            } => format!(
                "command `{command_id}` is unavailable behind gate `{gate}`; implementation owner: {owner}"
            ),
            CommandDispatchOutcome::Legacy { command_id, owner } => {
                format!("legacy command `{command_id}` is inactive; compatibility owner: {owner}")
            }
            CommandDispatchOutcome::Unknown { command_id } => {
                format!("unknown Sim command `{command_id}`")
            }
            CommandDispatchOutcome::Executed { .. } | CommandDispatchOutcome::Rejected { .. } => {
                return;
            }
        };
        self.model.report_error(message);
        cx.notify();
    }

    fn reject_shell_command(
        &mut self,
        command_id: &str,
        error: impl std::fmt::Display,
        cx: &mut Context<Self>,
    ) -> CommandDispatchOutcome {
        let error = error.to_string();
        self.model.report_error(&error);
        cx.notify();
        CommandDispatchOutcome::Rejected {
            command_id: command_id.to_owned(),
            error,
        }
    }

    pub fn execute_model_action(
        &mut self,
        action: GraphModelAction,
        input: GraphActionInput,
        cx: &mut Context<Self>,
    ) -> bool {
        let navigation_before = self.current_graph_navigation();
        match GraphCommandModel::execute_model_action(&mut self.model, action, input) {
            Ok(GraphActionEffect::None) => {
                self.invalidate_execution_projection_after_navigation(navigation_before);
                cx.emit(ItemEvent::Edit);
                self.schedule_automatic_execution(cx);
                cx.notify();
                true
            }
            Ok(GraphActionEffect::ClipboardText(_)) => {
                self.model
                    .report_error("typed graph model action produced an invalid clipboard effect");
                cx.notify();
                false
            }
            Err(error) => {
                self.model.report_error(error);
                cx.notify();
                false
            }
        }
    }

    pub fn apply_graph_command(&mut self, command: GraphCommand, cx: &mut Context<Self>) -> bool {
        let navigation_before = self.current_graph_navigation();
        match self.model.apply_with_change(command) {
            Ok(true) => {
                self.invalidate_execution_projection_after_navigation(navigation_before);
                cx.emit(ItemEvent::Edit);
                self.schedule_automatic_execution(cx);
                cx.notify();
                true
            }
            Ok(false) => {
                self.model.announcement = Some("Graph command made no changes".to_owned());
                cx.notify();
                true
            }
            Err(error) => {
                self.model.report_error(error);
                cx.notify();
                false
            }
        }
    }

    pub fn select_node(
        &mut self,
        identifier: GraphIdentifier,
        mode: SelectionMode,
        cx: &mut Context<Self>,
    ) -> bool {
        self.apply_graph_command(
            GraphCommand::SetSelection {
                selection: GraphSelection {
                    nodes: [identifier].into_iter().collect(),
                    ..GraphSelection::default()
                },
                mode,
            },
            cx,
        )
    }

    pub fn select_group(
        &mut self,
        identifier: GraphIdentifier,
        mode: SelectionMode,
        cx: &mut Context<Self>,
    ) -> bool {
        self.apply_graph_command(
            GraphCommand::SetSelection {
                selection: GraphSelection {
                    groups: [identifier].into_iter().collect(),
                    ..GraphSelection::default()
                },
                mode,
            },
            cx,
        )
    }

    pub fn select_reroute(
        &mut self,
        identifier: GraphIdentifier,
        mode: SelectionMode,
        cx: &mut Context<Self>,
    ) -> bool {
        self.apply_graph_command(
            GraphCommand::SetSelection {
                selection: GraphSelection {
                    reroutes: [identifier].into_iter().collect(),
                    ..GraphSelection::default()
                },
                mode,
            },
            cx,
        )
    }

    pub fn select_link(
        &mut self,
        identifier: GraphIdentifier,
        mode: SelectionMode,
        cx: &mut Context<Self>,
    ) -> bool {
        self.apply_graph_command(
            GraphCommand::SetSelection {
                selection: GraphSelection {
                    links: [identifier].into_iter().collect(),
                    ..GraphSelection::default()
                },
                mode,
            },
            cx,
        )
    }

    pub fn select_in_rect(
        &mut self,
        bounds: GraphRect,
        mode: SelectionMode,
        cx: &mut Context<Self>,
    ) -> bool {
        self.apply_graph_command(GraphCommand::SelectInRect { bounds, mode }, cx)
    }

    pub fn drag_selection(
        &mut self,
        delta: GraphPoint,
        snap: Option<f32>,
        cx: &mut Context<Self>,
    ) -> bool {
        self.apply_graph_command(GraphCommand::MoveSelection { delta, snap }, cx)
    }

    pub fn begin_selection_drag(&mut self, position: GraphPoint) {
        self.drag_anchor = Some(position);
        self.drag_delta = GraphPoint::ZERO;
    }

    pub fn update_selection_drag(&mut self, position: GraphPoint, cx: &mut Context<Self>) {
        let Some(anchor) = self.drag_anchor else {
            return;
        };
        self.drag_delta = GraphPoint {
            x: position.x - anchor.x,
            y: position.y - anchor.y,
        };
        cx.notify();
    }

    pub fn finish_selection_drag(&mut self, cx: &mut Context<Self>) -> bool {
        if self.drag_anchor.take().is_none() {
            return false;
        }
        let screen_delta = std::mem::replace(&mut self.drag_delta, GraphPoint::ZERO);
        if screen_delta == GraphPoint::ZERO {
            cx.notify();
            return true;
        }
        let scale = self
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| graph.viewport.scale)
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .unwrap_or(1.0);
        self.drag_selection(
            GraphPoint {
                x: screen_delta.x / scale,
                y: screen_delta.y / scale,
            },
            Some(10.0),
            cx,
        )
    }

    pub fn begin_box_selection(
        &mut self,
        position: GraphPoint,
        mode: SelectionMode,
        cx: &mut Context<Self>,
    ) {
        self.box_anchor = Some(position);
        self.box_current = Some(position);
        self.box_selection_mode = mode;
        cx.notify();
    }

    pub fn update_box_selection(&mut self, position: GraphPoint, cx: &mut Context<Self>) {
        if self.box_anchor.is_some() {
            self.box_current = Some(position);
            cx.notify();
        }
    }

    pub fn finish_box_selection(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(anchor) = self.box_anchor.take() else {
            return false;
        };
        let current = self.box_current.take().unwrap_or(anchor);
        let Some(graph) = self
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
        else {
            self.model.report_error("workflow is open read-only");
            cx.notify();
            return false;
        };
        let first = graph.viewport.screen_to_graph(anchor);
        let second = graph.viewport.screen_to_graph(current);
        let bounds = GraphRect {
            origin: GraphPoint {
                x: first.x.min(second.x),
                y: first.y.min(second.y),
            },
            size: sim_runtime::GraphSize {
                width: (first.x - second.x).abs().max(1.0),
                height: (first.y - second.y).abs().max(1.0),
            },
        };
        self.select_in_rect(bounds, self.box_selection_mode, cx)
    }

    pub fn pan_viewport(&mut self, delta: GraphPoint, cx: &mut Context<Self>) -> bool {
        self.apply_graph_command(GraphCommand::PanViewport { delta }, cx)
    }

    pub fn zoom_viewport(
        &mut self,
        factor: f32,
        anchor: GraphPoint,
        cx: &mut Context<Self>,
    ) -> bool {
        self.apply_graph_command(GraphCommand::ZoomViewport { factor, anchor }, cx)
    }

    pub fn start_link(&mut self, node: GraphIdentifier, output_slot: usize) {
        self.pending_reconnect = None;
        self.pending_link_position = self
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| {
                graph.nodes.get(&node).map(|node| {
                    graph.viewport.graph_to_screen(GraphPoint {
                        x: node.position.x + node.size.width,
                        y: node.position.y + 42.0 + output_slot as f32 * 22.0,
                    })
                })
            });
        self.pending_link = Some((node, output_slot));
        self.model.announcement = Some("Link creation started".to_owned());
    }

    pub fn start_link_reconnect(
        &mut self,
        link_identifier: GraphIdentifier,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some((origin_node, origin_slot)) = self
            .model
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|graph| graph.links.get(&link_identifier))
            .map(|link| (link.origin_node.clone(), link.origin_slot))
        else {
            self.model
                .report_error("link to reconnect no longer exists");
            cx.notify();
            return false;
        };
        self.start_link(origin_node, origin_slot);
        self.pending_reconnect = Some(link_identifier);
        self.model.announcement = Some("Link reconnection started".to_owned());
        cx.notify();
        true
    }

    pub fn update_pending_link_position(&mut self, position: GraphPoint, cx: &mut Context<Self>) {
        if self.pending_link.is_some() {
            self.pending_link_position = Some(position);
            cx.notify();
        }
    }

    pub fn complete_link(
        &mut self,
        identifier: GraphIdentifier,
        target_node: GraphIdentifier,
        target_slot: usize,
        replace_existing: bool,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some((origin_node, origin_slot)) = self.pending_link.take() else {
            self.model.report_error("no link creation is active");
            cx.notify();
            return false;
        };
        self.pending_link_position = None;
        let existing_identifier = self.pending_reconnect.take();
        let preserved_link_state = existing_identifier.as_ref().and_then(|identifier| {
            self.model
                .document()
                .and_then(|document| document.active_graph().ok())
                .and_then(|graph| graph.links.get(identifier))
                .map(|link| (link.parent_reroute.clone(), link.source.clone()))
        });
        let (parent_reroute, source) =
            preserved_link_state.unwrap_or_else(|| (None, serde_json::Value::Null));
        let connect = GraphCommand::Connect {
            link: GraphLink {
                identifier,
                origin_node,
                origin_slot,
                target_node,
                target_slot,
                type_name: String::new(),
                parent_reroute,
                source,
            },
            replace_existing,
        };
        let command = if let Some(existing_identifier) = existing_identifier {
            GraphCommand::Batch {
                commands: vec![
                    GraphCommand::RemoveLink {
                        identifier: existing_identifier,
                    },
                    connect,
                ],
            }
        } else {
            connect
        };
        self.apply_graph_command(command, cx)
    }

    pub fn complete_pending_link(
        &mut self,
        target_node: GraphIdentifier,
        target_slot: usize,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(mut document) = self.model.document().cloned() else {
            self.model.report_error("workflow is open read-only");
            cx.notify();
            return false;
        };
        let link_identifier = self
            .pending_reconnect
            .clone()
            .unwrap_or_else(|| document.allocate_identifier());
        self.complete_link(link_identifier, target_node, target_slot, true, cx)
    }

    pub fn reject_pending_link(&mut self, reason: impl Into<String>, cx: &mut Context<Self>) {
        if self.pending_link.take().is_some() {
            self.pending_link_position = None;
            self.pending_reconnect = None;
            self.model.report_error(reason.into());
            cx.notify();
        }
    }

    pub fn insert_reroute_on_link(
        &mut self,
        link_identifier: GraphIdentifier,
        screen_position: GraphPoint,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(mut document) = self.model.document().cloned() else {
            self.model.report_error("workflow is open read-only");
            cx.notify();
            return false;
        };
        let Some((mut link, position)) = document.active_graph().ok().and_then(|graph| {
            graph
                .links
                .get(&link_identifier)
                .cloned()
                .map(|link| (link, graph.viewport.screen_to_graph(screen_position)))
        }) else {
            self.model.report_error("link to reroute no longer exists");
            cx.notify();
            return false;
        };
        let reroute_identifier = document.allocate_identifier();
        let reroute = GraphReroute {
            identifier: reroute_identifier.clone(),
            position,
            parent: link.parent_reroute.clone(),
            floating_type: None,
            source_fields: serde_json::Map::new(),
        };
        link.parent_reroute = Some(reroute_identifier);
        self.apply_graph_command(
            GraphCommand::Batch {
                commands: vec![
                    GraphCommand::RemoveLink {
                        identifier: link_identifier,
                    },
                    GraphCommand::AddReroute { reroute },
                    GraphCommand::Connect {
                        link,
                        replace_existing: true,
                    },
                ],
            },
            cx,
        )
    }

    pub fn cancel_gesture(&mut self, cx: &mut Context<Self>) {
        let publication_cancelled = self
            .subgraph_publish_cancellation
            .as_ref()
            .is_some_and(|cancellation| cancellation.cancel());
        if let Some((_, initial_viewport)) = self.canvas_pan_anchor.take()
            && let Some(selection) = self.model.selection().cloned()
            && let Err(error) = self
                .model
                .replace_ephemeral_graph_state(selection, initial_viewport)
        {
            self.model.report_error(error);
        }
        self.drag_anchor = None;
        self.drag_delta = GraphPoint::ZERO;
        self.box_anchor = None;
        self.box_current = None;
        self.pending_link = None;
        self.pending_link_position = None;
        self.pending_reconnect = None;
        self.model.announcement = Some(if publication_cancelled {
            "Subgraph blueprint publication cancellation requested".to_owned()
        } else {
            "Graph gesture cancelled".to_owned()
        });
        cx.notify();
    }

    pub fn copy_to_clipboard(&mut self, cx: &mut Context<Self>) -> bool {
        self.execute_catalog_action(CatalogGraphAction::CopySelected, GraphActionInput::None, cx)
    }

    pub fn cut_to_clipboard(&mut self, cx: &mut Context<Self>) -> bool {
        let clipboard_text = match GraphCommandModel::execute(
            &mut self.model,
            CatalogGraphAction::CopySelected,
            GraphActionInput::None,
        ) {
            Ok(GraphActionEffect::ClipboardText(text)) => text,
            Ok(GraphActionEffect::None) => {
                self.model
                    .report_error("native graph cut produced no clipboard payload");
                cx.notify();
                return false;
            }
            Err(error) => {
                self.model.report_error(error);
                cx.notify();
                return false;
            }
        };
        if let Err(error) = self.model.apply(GraphCommand::RemoveSelection) {
            self.model.report_error(error);
            cx.notify();
            return false;
        }
        cx.write_to_clipboard(ClipboardItem::new_string(clipboard_text));
        self.model.announcement = Some("Cut graph selection to clipboard".to_owned());
        cx.emit(ItemEvent::Edit);
        cx.notify();
        true
    }

    pub fn paste_from_clipboard(&mut self, connect: bool, cx: &mut Context<Self>) -> bool {
        let offset = self.model.graph_point_for_paste();
        self.paste_from_clipboard_at(connect, offset, cx)
    }

    pub(crate) fn paste_from_clipboard_at(
        &mut self,
        connect: bool,
        offset: GraphPoint,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(item) = cx.read_from_clipboard() else {
            self.model.report_error("clipboard is empty");
            cx.notify();
            return false;
        };
        if item.entries.is_empty()
            || item
                .entries
                .iter()
                .any(|entry| !matches!(entry, ClipboardEntry::String(_)))
        {
            self.model
                .report_error("clipboard contains media or files instead of graph JSON");
            cx.notify();
            return false;
        }
        let Some(text) = item.text() else {
            self.model.report_error("clipboard text is empty");
            cx.notify();
            return false;
        };
        let action = if connect {
            CatalogGraphAction::PasteFromClipboardWithConnect
        } else {
            CatalogGraphAction::PasteFromClipboard
        };
        let connect_from = if connect {
            self.pending_link.clone()
        } else {
            None
        };
        self.execute_catalog_action(
            action,
            GraphActionInput::Paste {
                bytes: text.into_bytes(),
                offset,
                connect_from,
            },
            cx,
        )
    }

    pub fn focus_graph(&self, window: &mut Window, cx: &mut App) {
        window.focus(&self.focus_handle, cx);
    }

    pub(crate) fn graph_undo(&mut self, _: &GraphUndo, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch_shell_command("Sim.Undo", cx);
    }

    pub(crate) fn graph_redo(&mut self, _: &GraphRedo, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch_shell_command("Sim.Redo", cx);
    }

    pub(crate) fn graph_copy(&mut self, _: &GraphCopy, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch_shell_command("Sim.Canvas.CopySelected", cx);
    }

    pub(crate) fn graph_cut(&mut self, _: &GraphCut, _: &mut Window, cx: &mut Context<Self>) {
        self.cut_to_clipboard(cx);
    }

    pub(crate) fn graph_paste(&mut self, _: &GraphPaste, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch_shell_command("Sim.Canvas.PasteFromClipboard", cx);
    }

    pub(crate) fn graph_delete(&mut self, _: &GraphDelete, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch_shell_command("Sim.Canvas.DeleteSelectedItems", cx);
    }

    pub(crate) fn graph_select_all(
        &mut self,
        _: &GraphSelectAll,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch_shell_command("Sim.Canvas.SelectAll", cx);
    }

    pub(crate) fn graph_zoom_in(
        &mut self,
        _: &GraphZoomIn,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch_shell_command("Sim.Canvas.ZoomIn", cx);
    }

    pub(crate) fn graph_zoom_out(
        &mut self,
        _: &GraphZoomOut,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch_shell_command("Sim.Canvas.ZoomOut", cx);
    }

    pub(crate) fn graph_fit_view(
        &mut self,
        _: &GraphFitView,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch_shell_command("Sim.Canvas.FitView", cx);
    }

    pub(crate) fn graph_cancel_gesture(
        &mut self,
        _: &GraphCancelGesture,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.cancel_gesture(cx);
    }

    shell_action_handler!(shell_queue_prompt, crate::QueuePrompt, "Sim.QueuePrompt");
    shell_action_handler!(
        shell_queue_prompt_front,
        crate::QueuePromptFront,
        "Sim.QueuePromptFront"
    );
    shell_action_handler!(
        shell_queue_selected_output_nodes,
        crate::QueueSelectedOutputNodes,
        "Sim.QueueSelectedOutputNodes"
    );
    shell_action_handler!(shell_interrupt, crate::Interrupt, "Sim.Interrupt");
    shell_action_handler!(
        shell_clear_pending_tasks,
        crate::ClearPendingTasks,
        "Sim.ClearPendingTasks"
    );
    shell_action_handler!(
        shell_toggle_queue_overlay,
        crate::ToggleQueueOverlay,
        "Sim.Queue.ToggleOverlay"
    );
    pub(crate) fn shell_toggle_qpov2(
        &mut self,
        _: &crate::ToggleQpov2,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self
            .dispatch_shell_command("Sim.ToggleQPOV2", cx)
            .is_executed()
        {
            window.dispatch_action(crate::ToggleDockedExecutionHistory.boxed_clone(), cx);
        }
    }

    pub(crate) fn execution_run_manual(
        &mut self,
        _: &crate::ExecutionRunManual,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_execution_run_mode(ExecutionRunMode::Manual, cx);
    }

    pub(crate) fn execution_run_on_change(
        &mut self,
        _: &crate::ExecutionRunOnChange,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_execution_run_mode(ExecutionRunMode::OnChange, cx);
    }

    pub(crate) fn execution_run_instant_idle(
        &mut self,
        _: &crate::ExecutionRunInstantIdle,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_execution_run_mode(ExecutionRunMode::InstantIdle, cx);
    }

    shell_action_handler!(
        shell_refresh_node_definitions,
        crate::RefreshNodeDefinitions,
        "Sim.RefreshNodeDefinitions"
    );
    shell_action_handler!(
        shell_toggle_workflows_sidebar,
        crate::ToggleWorkflowsSidebar,
        "Workspace.ToggleSidebarTab.workflows"
    );
    pub(crate) fn shell_toggle_node_library_sidebar(
        &mut self,
        _: &crate::ToggleNodeLibrarySidebar,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch_shell_command("Workspace.ToggleSidebarTab.node-library", cx);
        if let Some(catalog) = crate::native_subgraph_catalog(cx) {
            self.model.announcement = Some(crate::subgraph_catalog_node_library_message(catalog));
            cx.notify();
        }
    }
    shell_action_handler!(
        shell_toggle_model_library_sidebar,
        crate::ToggleModelLibrarySidebar,
        "Workspace.ToggleSidebarTab.model-library"
    );
    shell_action_handler!(
        shell_toggle_assets_sidebar,
        crate::ToggleAssetsSidebar,
        "Workspace.ToggleSidebarTab.assets"
    );
    shell_action_handler!(shell_toggle_linear, crate::ToggleLinear, "Sim.ToggleLinear");
    pub(crate) fn shell_save_workflow(
        &mut self,
        _: &crate::SaveWorkflow,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !command_registration("Sim.SaveWorkflow")
            .is_some_and(|registration| registration.status == CommandNativeStatus::Executable)
        {
            self.dispatch_shell_command("Sim.SaveWorkflow", cx);
            return;
        }
        #[cfg(feature = "test-support")]
        self.shell_dispatch_trace
            .push("Sim.SaveWorkflow".to_owned());
        if self.workspace.upgrade().is_none() {
            self.model
                .report_error("The workflow's owning workspace is unavailable");
            cx.notify();
            return;
        }
        self.save_workflow_in_workspace(window, cx)
            .detach_and_prompt_err("Failed to save", window, cx, |_, _, _| None);
    }

    fn save_workflow_in_workspace(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        let workspace = self.workspace.clone();
        let graph_id = cx.entity_id();
        // Pane::save_item updates the graph, so release this graph update before entering it.
        cx.spawn_in(window, async move |_, cx| {
            let save = workspace
                .update_in(cx, |workspace, window, cx| {
                    anyhow::ensure!(
                        workspace
                            .active_item_as::<GraphWorkspaceItem>(cx)
                            .is_some_and(|graph| graph.entity_id() == graph_id),
                        "The workflow is no longer active in its owning workspace"
                    );
                    Ok::<_, anyhow::Error>(workspace.save_active_item(
                        workspace::SaveIntent::Save,
                        window,
                        cx,
                    ))
                })
                .context("The workflow's owning workspace is unavailable")??;
            save.await
        })
    }
    shell_action_handler!(shell_open_workflow, crate::OpenWorkflow, "Sim.OpenWorkflow");
    shell_action_handler!(
        shell_group_selected_nodes,
        crate::GroupSelectedNodes,
        "Sim.Graph.GroupSelectedNodes"
    );
    shell_action_handler!(
        shell_show_settings,
        crate::ShowSettings,
        "Sim.ShowSettingsDialog"
    );
    shell_action_handler!(
        shell_show_keybindings,
        crate::ShowKeybindings,
        "Workspace.ToggleBottomPanel.Shortcuts"
    );
    shell_action_handler!(
        shell_toggle_selected_items_pin,
        crate::ToggleSelectedItemsPin,
        "Sim.Canvas.ToggleSelected.Pin"
    );
    shell_action_handler!(
        shell_toggle_selected_collapse,
        crate::ToggleSelectedCollapse,
        "Sim.Canvas.ToggleSelectedNodes.Collapse"
    );
    shell_action_handler!(
        shell_toggle_selected_bypass,
        crate::ToggleSelectedBypass,
        "Sim.Canvas.ToggleSelectedNodes.Bypass"
    );
    shell_action_handler!(
        shell_toggle_selected_mute,
        crate::ToggleSelectedMute,
        "Sim.Canvas.ToggleSelectedNodes.Mute"
    );
    shell_action_handler!(
        shell_toggle_logs_panel,
        crate::ToggleLogsPanel,
        "Workspace.ToggleBottomPanelTab.logs-terminal"
    );
    shell_action_handler!(
        shell_convert_to_subgraph,
        crate::ConvertToSubgraph,
        "Sim.Graph.ConvertToSubgraph"
    );
    shell_action_handler!(
        shell_toggle_minimap,
        crate::ToggleMinimap,
        "Sim.Canvas.ToggleMinimap"
    );
    shell_action_handler!(
        shell_unlock_canvas,
        crate::UnlockCanvas,
        "Sim.Canvas.Unlock"
    );
    shell_action_handler!(shell_lock_canvas, crate::LockCanvas, "Sim.Canvas.Lock");
    shell_action_handler!(
        shell_exit_subgraph,
        crate::ExitSubgraph,
        "Sim.Graph.ExitSubgraph"
    );
    shell_action_handler!(
        shell_paste_with_connect,
        crate::PasteWithConnect,
        "Sim.Canvas.PasteFromClipboardWithConnect"
    );
    shell_action_handler!(
        shell_move_selected_down,
        crate::MoveSelectedDown,
        "Sim.Canvas.MoveSelectedNodes.Down"
    );
    shell_action_handler!(
        shell_move_selected_left,
        crate::MoveSelectedLeft,
        "Sim.Canvas.MoveSelectedNodes.Left"
    );
    shell_action_handler!(
        shell_move_selected_right,
        crate::MoveSelectedRight,
        "Sim.Canvas.MoveSelectedNodes.Right"
    );
    shell_action_handler!(
        shell_move_selected_up,
        crate::MoveSelectedUp,
        "Sim.Canvas.MoveSelectedNodes.Up"
    );
    shell_action_handler!(shell_reset_view, crate::ResetView, "Sim.Canvas.ResetView");
    shell_action_handler!(
        shell_resize_selected_nodes,
        crate::ResizeSelectedNodes,
        "Sim.Canvas.Resize"
    );
    shell_action_handler!(
        shell_toggle_link_visibility,
        crate::ToggleLinkVisibility,
        "Sim.Canvas.ToggleLinkVisibility"
    );
    shell_action_handler!(
        shell_toggle_canvas_lock,
        crate::ToggleCanvasLock,
        "Sim.Canvas.ToggleLock"
    );
    shell_action_handler!(
        shell_toggle_selected_nodes_pin,
        crate::ToggleSelectedNodesPin,
        "Sim.Canvas.ToggleSelectedNodes.Pin"
    );
    shell_action_handler!(
        shell_fit_group_to_contents,
        crate::FitGroupToContents,
        "Sim.Graph.FitGroupToContents"
    );
    shell_action_handler!(
        shell_unpack_subgraph,
        crate::UnpackSubgraph,
        "Sim.Graph.UnpackSubgraph"
    );
    pub(crate) fn shell_publish_subgraph(
        &mut self,
        _: &crate::PublishSubgraph,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(
            self.dispatch_shell_command("Sim.PublishSubgraph", cx),
            CommandDispatchOutcome::RequiresInput { .. }
        ) {
            self.begin_shell_publish_subgraph(window, cx);
        }
    }
}

async fn atomic_create_workflow_file(
    fs: &Arc<dyn project::Fs>,
    target: &Path,
    text: &str,
    operation_id: uuid::Uuid,
) -> anyhow::Result<()> {
    let staging_path = target.with_file_name(format!(".zed-workflow-save-{operation_id}.tmp"));
    fs.write(&staging_path, text.as_bytes())
        .await
        .with_context(|| {
            format!(
                "failed to stage workflow transaction {}",
                staging_path.display()
            )
        })?;
    if let Err(rename_error) = fs
        .rename(&staging_path, target, RenameOptions::default())
        .await
    {
        if let Err(cleanup_error) = fs
            .remove_file(
                &staging_path,
                RemoveOptions {
                    ignore_if_not_exists: true,
                    ..RemoveOptions::default()
                },
            )
            .await
        {
            return Err(anyhow!(
                "failed to create workflow {}: {rename_error}; failed to remove staging file {}: {cleanup_error}",
                target.display(),
                staging_path.display()
            ));
        }
        return Err(rename_error).with_context(|| {
            format!(
                "failed to atomically create workflow {} without replacing an existing file",
                target.display()
            )
        });
    }
    Ok(())
}

async fn rollback_workflow_write(
    fs: &Arc<dyn project::Fs>,
    target: &Path,
    written_bytes: &[u8],
    previous_bytes: Option<&[u8]>,
) -> anyhow::Result<()> {
    let current_bytes = fs.load_bytes(target).await.with_context(|| {
        format!(
            "failed to inspect workflow transaction target {} before rollback",
            target.display()
        )
    })?;
    if current_bytes != written_bytes {
        anyhow::bail!(
            "workflow transaction target {} changed after the write; refusing to overwrite it during rollback",
            target.display()
        );
    }
    if let Some(previous_bytes) = previous_bytes {
        let previous_text = String::from_utf8(previous_bytes.to_vec())
            .context("previous workflow bytes are not valid UTF-8 JSON")?;
        fs.atomic_write(target.to_path_buf(), previous_text)
            .await
            .with_context(|| format!("failed to restore workflow {}", target.display()))
    } else {
        fs.remove_file(target, RemoveOptions::default())
            .await
            .with_context(|| {
                format!(
                    "failed to remove uncommitted workflow transaction {}",
                    target.display()
                )
            })
    }
}

fn action_mutates_workflow(action: CatalogGraphAction) -> bool {
    !matches!(
        action,
        CatalogGraphAction::CopySelected
            | CatalogGraphAction::ToggleCanvasInfo
            | CatalogGraphAction::ToggleVueNodes
    )
}

impl Drop for GraphWorkspaceItem {
    fn drop(&mut self) {
        self.cancel_subgraph_publication_for_drop();
    }
}

impl GraphWorkspaceItem {
    fn cancel_subgraph_publication_for_drop(&mut self) {
        if let Some(cancellation) = &self.subgraph_publish_cancellation {
            cancellation.cancel();
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn cancel_subgraph_publication_for_drop_for_test(&mut self) {
        self.cancel_subgraph_publication_for_drop();
    }
}

impl EventEmitter<ItemEvent> for GraphWorkspaceItem {}

impl Focusable for GraphWorkspaceItem {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for GraphWorkspaceItem {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        use ui::prelude::*;

        let message = crate::initialization_error(cx).map(str::to_owned);
        let graph = graph_render::render_graph_item(self, window, cx);
        let Some(message) = message else {
            return graph;
        };
        div()
            .size_full()
            .relative()
            .child(graph)
            .child(
                v_flex()
                    .id("sim-initialization-error")
                    .debug_selector(|| "SIM-INITIALIZATION-ERROR".into())
                    .role(gpui::Role::Alert)
                    .aria_label(format!("Execution is unavailable: {message}"))
                    .absolute()
                    .left_3()
                    .right_3()
                    .bottom_3()
                    .max_h_48()
                    .overflow_y_scroll()
                    .gap_1()
                    .rounded_md()
                    .border_1()
                    .border_color(cx.theme().status().error_border)
                    .bg(cx.theme().status().error_background)
                    .p_3()
                    .text_ui_sm(cx)
                    .text_color(cx.theme().colors().text)
                    .child("Execution is unavailable")
                    .child(message),
            )
            .into_any_element()
    }
}

impl Item for GraphWorkspaceItem {
    type Event = ItemEvent;

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        let status = self
            .active_execution_presentation(cx)
            .map(|attempt| format!("{:?}", attempt.state));
        status.map_or_else(
            || self.model.title.clone().into(),
            |status| format!("{} — {status}", self.model.title).into(),
        )
    }

    fn telemetry_event_text(&self) -> Option<&'static str> {
        Some("Sim Native Graph Opened")
    }

    fn show_toolbar(&self) -> bool {
        false
    }

    fn is_dirty(&self, _: &App) -> bool {
        matches!(
            self.model.save_coordinator.authority(),
            WorkflowAuthority::LocalDirty
                | WorkflowAuthority::Conflict
                | WorkflowAuthority::ExternalMissing
                | WorkflowAuthority::SavePrepared
                | WorkflowAuthority::Interrupted
        )
    }

    fn has_conflict(&self, _: &App) -> bool {
        matches!(
            self.model.save_coordinator.authority(),
            WorkflowAuthority::Conflict | WorkflowAuthority::ExternalMissing
        )
    }

    fn can_save(&self, _: &App) -> bool {
        !self.model.is_read_only()
            && matches!(
                self.model.save_coordinator.provider(),
                WorkflowStorageProvider::Draft | WorkflowStorageProvider::LocalFile
            )
    }

    fn can_save_as(&self, _: &App) -> bool {
        !self.model.is_read_only()
    }

    fn save(
        &mut self,
        _options: workspace::item::SaveOptions,
        project: Entity<Project>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        if self.model.is_read_only() {
            let result = Err(anyhow!("read-only workflow cannot be saved"));
            self.report_storage_result(&result, "Workflow saved", cx);
            return Task::ready(result);
        }

        match self.model.save_coordinator.provider().clone() {
            WorkflowStorageProvider::Draft => {
                let result = self.model.commit_draft_save().map_err(anyhow::Error::from);
                self.report_storage_result(&result, "Draft workflow saved", cx);
                Task::ready(result)
            }
            WorkflowStorageProvider::Provider { identifier } => {
                let result = Err(anyhow!(
                    "workflow provider `{identifier}` does not support native local save"
                ));
                self.report_storage_result(&result, "Workflow saved", cx);
                Task::ready(result)
            }
            WorkflowStorageProvider::LocalFile => {
                let path = match self.local_file_path() {
                    Ok(path) => path,
                    Err(error) => {
                        let result = Err(error);
                        self.report_storage_result(&result, "Workflow saved", cx);
                        return Task::ready(result);
                    }
                };
                let target_identity = match Self::path_identity(&path) {
                    Ok(identity) => identity,
                    Err(error) => {
                        let result = Err(error);
                        self.report_storage_result(&result, "Workflow saved", cx);
                        return Task::ready(result);
                    }
                };
                let fs = project.read(cx).fs().clone();
                cx.spawn(async move |this, cx| {
                    let result: anyhow::Result<()> = async {
                        let disk_bytes = match fs.load_bytes(&path).await {
                            Ok(bytes) => Some(bytes),
                            Err(_error) if !fs.is_file(&path).await => None,
                            Err(error) => {
                                return Err(error).with_context(|| {
                                    format!("failed to read {} before save", path.display())
                                });
                            }
                        };
                        let operation_id = uuid::Uuid::new_v4();
                        let (observed_revision, prepared, recreate_missing) = this.update(
                            cx,
                            |this, _cx| {
                                let recreate_missing = disk_bytes.is_none();
                                let observed_revision = match disk_bytes {
                                    Some(disk_bytes) => {
                                        let observed_revision =
                                            ContentRevision::from_bytes(&disk_bytes);
                                        this.model.observe_external_change(disk_bytes)?;
                                        observed_revision
                                    }
                                    None => {
                                        if !this.model.save_coordinator.external_missing() {
                                            this.model.observe_external_deletion()?;
                                        }
                                        this.model.save_coordinator.base().revision.clone()
                                    }
                                };
                                let prepared = this.model.prepare_save(
                                    operation_id,
                                    observed_revision.clone(),
                                    target_identity,
                                    false,
                                )?;
                                Ok::<_, GraphWorkspaceError>((
                                    observed_revision,
                                    prepared,
                                    recreate_missing,
                                ))
                            },
                        )??;

                        let rollback_bytes = if recreate_missing {
                            if fs.is_file(&path).await {
                                let reappeared = fs.load_bytes(&path).await.with_context(|| {
                                    format!(
                                        "failed to inspect reappeared workflow {}",
                                        path.display()
                                    )
                                })?;
                                this.update(cx, |this, _cx| {
                                    this.model.observe_external_change(reappeared)
                                })??;
                                anyhow::bail!(
                                    "workflow reappeared on disk during recreation: {}",
                                    path.display()
                                );
                            }
                            None
                        } else {
                            let current_bytes = match fs.load_bytes(&path).await {
                                Ok(bytes) => bytes,
                                Err(error) if !fs.is_file(&path).await => {
                                    this.update(cx, |this, _cx| {
                                        this.model.observe_external_deletion()
                                    })??;
                                    return Err(error).with_context(|| {
                                        format!("workflow disappeared during save: {}", path.display())
                                    });
                                }
                                Err(error) => {
                                    return Err(error).with_context(|| {
                                        format!("failed to verify {} before save", path.display())
                                    });
                                }
                            };
                            let current_revision = ContentRevision::from_bytes(&current_bytes);
                            if current_revision != observed_revision {
                                this.update(cx, |this, _cx| {
                                    this.model.observe_external_change(current_bytes)
                                })??;
                                anyhow::bail!(
                                    "workflow changed on disk during save (expected {}, found {})",
                                    observed_revision.0,
                                    current_revision.0
                                );
                            }
                            Some(current_bytes)
                        };

                        let text = String::from_utf8(prepared.bytes.clone())
                            .context("native workflow serialization is not valid UTF-8")?;
                        if recreate_missing {
                            atomic_create_workflow_file(&fs, &path, &text, operation_id).await?;
                        } else {
                            fs.atomic_write(path.clone(), text)
                                .await
                                .with_context(|| format!("failed to save {}", path.display()))?;
                        }
                        let committed_revision = ContentRevision::from_bytes(&prepared.bytes);
                        let commit_result = this.update(cx, |this, _cx| {
                            let mut candidate = this.model.clone();
                            candidate.commit_save(
                                operation_id,
                                observed_revision,
                                committed_revision,
                            )?;
                            this.model = candidate;
                            Ok::<(), GraphWorkspaceError>(())
                        });
                        match commit_result {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => {
                                if let Err(rollback_error) = rollback_workflow_write(
                                    &fs,
                                    &path,
                                    &prepared.bytes,
                                    rollback_bytes.as_deref(),
                                )
                                .await
                                {
                                    return Err(anyhow!(
                                        "workflow save state commit failed: {error}; disk rollback failed: {rollback_error}"
                                    ));
                                }
                                return Err(anyhow::Error::from(error));
                            }
                            Err(error) => {
                                if let Err(rollback_error) = rollback_workflow_write(
                                    &fs,
                                    &path,
                                    &prepared.bytes,
                                    rollback_bytes.as_deref(),
                                )
                                .await
                                {
                                    return Err(anyhow!(
                                        "workflow save item update failed: {error}; disk rollback failed: {rollback_error}"
                                    ));
                                }
                                return Err(error);
                            }
                        }
                        Ok(())
                    }
                    .await;
                    this.update(cx, |this, cx| {
                        this.report_storage_result(&result, "Workflow saved", cx)
                    })?;
                    result
                })
            }
        }
    }

    fn save_as(
        &mut self,
        project: Entity<Project>,
        path: ProjectPath,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        if self.model.is_read_only() {
            let result = Err(anyhow!(
                "read-only workflow cannot be saved as a local file"
            ));
            self.report_storage_result(&result, "Workflow saved as local file", cx);
            return Task::ready(result);
        }
        let absolute_path = match project.read(cx).absolute_path(&path, cx) {
            Some(path) => path,
            None => {
                let result = Err(anyhow!("failed to resolve local workflow path"));
                self.report_storage_result(&result, "Workflow saved as local file", cx);
                return Task::ready(result);
            }
        };
        let target_identity = match Self::path_identity(&absolute_path) {
            Ok(identity) => identity,
            Err(error) => {
                let result = Err(error);
                self.report_storage_result(&result, "Workflow saved as local file", cx);
                return Task::ready(result);
            }
        };
        let title = absolute_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned);
        let fs = project.read(cx).fs().clone();
        let watch_fs = fs.clone();
        let operation_id = uuid::Uuid::new_v4();
        let observed_revision = self.model.save_coordinator.base().revision.clone();
        cx.spawn(async move |this, cx| {
            let result: anyhow::Result<()> = async {
                if fs.is_file(&absolute_path).await {
                    anyhow::bail!(
                        "workflow save target already exists: {}",
                        absolute_path.display()
                    );
                }
                let prepared = this.update(cx, |this, _cx| {
                    this.model.prepare_save(
                        operation_id,
                        observed_revision.clone(),
                        target_identity,
                        true,
                    )
                })??;
                let text = String::from_utf8(prepared.bytes.clone())
                    .context("native workflow serialization is not valid UTF-8")?;
                atomic_create_workflow_file(&fs, &absolute_path, &text, operation_id).await?;
                let committed_revision = ContentRevision::from_bytes(&prepared.bytes);
                let commit_result = this.update(cx, |this, _cx| {
                    let mut candidate = this.model.clone();
                    candidate.commit_save(
                        operation_id,
                        observed_revision,
                        committed_revision,
                    )?;
                    candidate
                        .save_coordinator
                        .switch_provider_after_committed_save(WorkflowStorageProvider::LocalFile)?;
                    if let Some(title) = title {
                        candidate.title = title;
                    }
                    this.model = candidate;
                    Ok::<(), GraphWorkspaceError>(())
                });
                match commit_result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        if let Err(rollback_error) = rollback_workflow_write(
                            &fs,
                            &absolute_path,
                            &prepared.bytes,
                            None,
                        )
                        .await
                        {
                            return Err(anyhow!(
                                "workflow save-as state commit failed: {error}; disk rollback failed: {rollback_error}"
                            ));
                        }
                        return Err(anyhow::Error::from(error));
                    }
                    Err(error) => {
                        if let Err(rollback_error) = rollback_workflow_write(
                            &fs,
                            &absolute_path,
                            &prepared.bytes,
                            None,
                        )
                        .await
                        {
                            return Err(anyhow!(
                                "workflow save-as item update failed: {error}; disk rollback failed: {rollback_error}"
                            ));
                        }
                        return Err(error);
                    }
                }
                this.update(cx, |this, cx| {
                    this.start_local_file_watch(watch_fs, absolute_path, cx)
                })?;
                Ok(())
            }
            .await;
            this.update(cx, |this, cx| {
                this.report_storage_result(&result, "Workflow saved as local file", cx)
            })?;
            result
        })
    }

    fn reload(
        &mut self,
        project: Entity<Project>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        let path = match self.local_file_path() {
            Ok(path) => path,
            Err(error) => {
                let result = Err(error);
                self.report_storage_result(&result, "Workflow reloaded", cx);
                return Task::ready(result);
            }
        };
        let fs = project.read(cx).fs().clone();
        cx.spawn(async move |this, cx| {
            let result: anyhow::Result<()> = async {
                let bytes = fs
                    .load_bytes(&path)
                    .await
                    .with_context(|| format!("failed to reload {}", path.display()))?;
                this.update(cx, |this, _cx| this.model.reload_from_storage(bytes))??;
                Ok(())
            }
            .await;
            this.update(cx, |this, cx| {
                match &result {
                    Ok(()) => cx.emit(ItemEvent::UpdateTab),
                    Err(error) => this.model.report_error(error),
                }
                cx.notify();
            })?;
            result
        })
    }

    fn to_item_events(event: &Self::Event, emit: &mut dyn FnMut(ItemEvent)) {
        emit(*event);
    }

    fn added_to_workspace(
        &mut self,
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.workspace = workspace.weak_handle();
        if let Ok(path) = self.local_file_path() {
            let fs = workspace.project().read(cx).fs().clone();
            self.start_local_file_watch(fs, path, cx);
        }
        window.focus(&self.focus_handle, cx);
    }
}

impl SerializableItem for GraphWorkspaceItem {
    fn serialized_item_kind() -> &'static str {
        "ComfyNativeGraph"
    }

    fn cleanup(
        workspace_id: WorkspaceId,
        alive_items: Vec<ItemId>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<anyhow::Result<()>> {
        let database = persistence::SimWorkflowDb::global(cx);
        delete_unloaded_items(
            alive_items,
            workspace_id,
            "comfy_workflow_items",
            &database,
            cx,
        )
    }

    fn deserialize(
        _project: Entity<Project>,
        workspace: WeakEntity<Workspace>,
        workspace_id: WorkspaceId,
        item_id: ItemId,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<anyhow::Result<Entity<Self>>> {
        let snapshot_result = persistence::SimWorkflowDb::global(cx)
            .get_snapshot(item_id, workspace_id)
            .context("failed to load native graph workspace snapshot");
        let model_result: anyhow::Result<GraphWorkspaceModel> = (|| {
            Ok(match snapshot_result {
                Ok(Some(snapshot)) => match GraphWorkspaceModel::decode(snapshot.as_bytes()) {
                    Ok(model) => model,
                    Err(error) => {
                        let mut model = GraphWorkspaceModel::open(
                            "Recovered workflow",
                            format!("workspace-{workspace_id:?}-item-{item_id:?}"),
                            WorkflowStorageProvider::Draft,
                            snapshot.into_bytes(),
                        )?;
                        model.report_error(format!(
                            "workspace snapshot could not be decoded: {error}"
                        ));
                        model
                    }
                },
                Ok(None) => {
                    let mut model = GraphWorkspaceModel::open(
                        "Recovered workflow",
                        format!("workspace-{workspace_id:?}-item-{item_id:?}"),
                        WorkflowStorageProvider::Draft,
                        Vec::new(),
                    )?;
                    model.report_error(
                        "workspace snapshot is missing; opened preserved empty data read-only",
                    );
                    model
                }
                Err(error) => {
                    let mut model = GraphWorkspaceModel::open(
                        "Recovered workflow",
                        format!("workspace-{workspace_id:?}-item-{item_id:?}"),
                        WorkflowStorageProvider::Draft,
                        Vec::new(),
                    )?;
                    model.report_error(error);
                    model
                }
            })
        })();
        let model = match model_result {
            Ok(model) => model,
            Err(error) => return Task::ready(Err(error)),
        };
        let entity = cx.new(|cx| Self::new(model, workspace, cx));
        Task::ready(Ok(entity))
    }

    fn serialize(
        &mut self,
        workspace: &mut Workspace,
        item_id: ItemId,
        _closing: bool,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Task<anyhow::Result<()>>> {
        let workspace_id = workspace.database_id()?;
        let snapshot = match self.model.encode().and_then(|bytes| {
            String::from_utf8(bytes)
                .map_err(|error| GraphWorkspaceError::Persistence(error.to_string()))
        }) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.model.report_error(&error);
                cx.notify();
                return Some(Task::ready(Err(anyhow!(error))));
            }
        };
        let database = persistence::SimWorkflowDb::global(cx);
        Some(cx.background_spawn(async move {
            database
                .save_snapshot(item_id, workspace_id, snapshot)
                .await
                .context("failed to persist native graph workspace")
        }))
    }

    fn should_serialize(&self, event: &Self::Event) -> bool {
        matches!(event, ItemEvent::Edit | ItemEvent::UpdateTab)
    }
}

pub(crate) mod persistence {
    use db::{
        query,
        sqlez::{domain::Domain, thread_safe_connection::ThreadSafeConnection},
        sqlez_macros::sql,
    };
    use workspace::{ItemId, WorkspaceDb, WorkspaceId};

    pub struct SimWorkflowDb(ThreadSafeConnection);

    impl Domain for SimWorkflowDb {
        // Preserve the migration domain used by existing workflow databases.
        const NAME: &str = "ComfyWorkflowDb";

        const MIGRATIONS: &[&str] = &[
            sql!(
                CREATE TABLE comfy_workflow_items (
                    workspace_id INTEGER,
                    item_id INTEGER UNIQUE,
                    snapshot_json TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, item_id),
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id)
                    ON DELETE CASCADE
                ) STRICT;
            ),
            sql!(
                CREATE TABLE comfy_workflow_items2 (
                    workspace_id INTEGER,
                    item_id INTEGER,
                    snapshot_json TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, item_id),
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id)
                    ON DELETE CASCADE
                ) STRICT;

                INSERT INTO comfy_workflow_items2 (workspace_id, item_id, snapshot_json)
                SELECT workspace_id, item_id, snapshot_json FROM comfy_workflow_items;

                DROP TABLE comfy_workflow_items;

                ALTER TABLE comfy_workflow_items2 RENAME TO comfy_workflow_items;
            ),
        ];
    }

    db::static_connection!(SimWorkflowDb, [WorkspaceDb]);

    impl SimWorkflowDb {
        query! {
            pub async fn save_snapshot(
                item_id: ItemId,
                workspace_id: WorkspaceId,
                snapshot_json: String
            ) -> Result<()> {
                INSERT INTO comfy_workflow_items(item_id, workspace_id, snapshot_json)
                VALUES (?, ?, ?)
                ON CONFLICT(workspace_id, item_id) DO UPDATE SET
                    snapshot_json = excluded.snapshot_json
            }
        }

        query! {
            pub fn get_snapshot(
                item_id: ItemId,
                workspace_id: WorkspaceId
            ) -> Result<Option<String>> {
                SELECT snapshot_json
                FROM comfy_workflow_items
                WHERE item_id = ? AND workspace_id = ?
            }
        }

        #[cfg(all(test, feature = "test-support"))]
        pub async fn install_write_failure_for_tests(&self) -> anyhow::Result<()> {
            self.0
                .write(|connection| {
                    let mut statement = connection.exec(sql!(
                        CREATE TRIGGER sim_workflow_test_fail_write
                        BEFORE INSERT ON comfy_workflow_items
                        BEGIN
                            SELECT RAISE(ABORT, "injected sim workflow write failure");
                        END;
                    ))?;
                    statement()
                })
                .await
        }

        #[cfg(all(test, feature = "test-support"))]
        pub async fn remove_write_failure_for_tests(&self) -> anyhow::Result<()> {
            self.0
                .write(|connection| {
                    let mut statement = connection.exec(sql!(
                        DROP TRIGGER IF EXISTS sim_workflow_test_fail_write;
                    ))?;
                    statement()
                })
                .await
        }
    }
}

#[cfg(all(test, feature = "test-support"))]
mod settings_ownership_tests {
    use super::*;
    use gpui::{TestAppContext, UpdateGlobal as _};
    use std::{cell::Cell, rc::Rc};

    fn workflow_evidence(item: &GraphWorkspaceItem) -> (Vec<u8>, Vec<u8>, bool) {
        (
            item.model
                .document()
                .expect("editable graph document")
                .to_workflow_bytes()
                .expect("serialize graph workflow"),
            item.model.save_coordinator.local_bytes().to_vec(),
            item.model
                .engine()
                .is_some_and(sim_runtime::GraphCommandEngine::can_undo),
        )
    }

    #[gpui::test(seed = 417)]
    fn initialization_error_banner_updates_and_clears_without_mutating_workflow(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
            crate::init(cx);
        });
        let (item, cx) = cx.add_window_view(|_, cx| {
            let mut model = GraphWorkspaceModel::create("Initialization error delivery")
                .expect("create canonical workflow");
            model.report_error("Independent graph validation failure");
            GraphWorkspaceItem::new(model, WeakEntity::new_invalid(), cx)
        });
        cx.run_until_parked();
        let before = item.read_with(cx, |item, _| workflow_evidence(item));
        let original_canvas = cx.debug_bounds("SIM-GRAPH").expect("actual graph canvas");
        assert!(cx.debug_bounds("SIM-INITIALIZATION-ERROR").is_none());
        let notifications = Rc::new(Cell::new(0));
        let _observation = cx.update(|_, cx| {
            let notifications = notifications.clone();
            cx.observe(&item, move |_, _| {
                notifications.set(notifications.get() + 1)
            })
        });
        cx.run_until_parked();
        let before_notifications = notifications.get();
        cx.update(|_, cx| crate::set_initialization_error("current worker startup failed", cx));
        cx.run_until_parked();
        assert!(notifications.get() > before_notifications);
        let banner = cx
            .debug_bounds("SIM-INITIALIZATION-ERROR")
            .expect("rendered initialization alert");
        assert!(banner.size.width > gpui::px(0.0) && banner.size.height > gpui::px(0.0));
        assert_eq!(cx.debug_bounds("SIM-GRAPH"), Some(original_canvas));
        assert_eq!(
            cx.read(|cx| crate::initialization_error(cx).map(str::to_owned)),
            Some("current worker startup failed".to_owned())
        );
        let after_first_notifications = notifications.get();
        cx.update(|_, cx| crate::set_initialization_error("canonical shutdown failed", cx));
        cx.run_until_parked();
        assert!(notifications.get() > after_first_notifications);
        assert!(cx.debug_bounds("SIM-INITIALIZATION-ERROR").is_some());
        assert_eq!(
            cx.read(|cx| crate::initialization_error(cx).map(str::to_owned)),
            Some("current worker startup failed; canonical shutdown failed".to_owned())
        );
        let after_update_notifications = notifications.get();
        cx.update(|_, cx| crate::clear_initialization_error(cx));
        cx.run_until_parked();
        assert!(notifications.get() > after_update_notifications);
        assert!(cx.debug_bounds("SIM-INITIALIZATION-ERROR").is_none());
        assert_eq!(cx.debug_bounds("SIM-GRAPH"), Some(original_canvas));
        item.read_with(cx, |item, cx| {
            assert!(crate::initialization_error(cx).is_none());
            assert_eq!(
                item.model.last_error.as_deref(),
                Some("Independent graph validation failure")
            );
            assert_eq!(workflow_evidence(item), before);
        });
        cx.update(|_, cx| crate::set_initialization_error("later activation failed", cx));
        cx.run_until_parked();
        assert!(cx.debug_bounds("SIM-INITIALIZATION-ERROR").is_some());
        assert_eq!(cx.debug_bounds("SIM-GRAPH"), Some(original_canvas));
        assert_eq!(
            item.read_with(cx, |item, _| workflow_evidence(item)),
            before
        );
    }

    fn assert_group_padding(item: &GraphWorkspaceItem, identifier: &GraphIdentifier, padding: f32) {
        let graph = item
            .model
            .document()
            .expect("editable graph document")
            .active_graph()
            .expect("active graph");
        let group = graph.groups.get(identifier).expect("group exists");
        let mut node_bounds = group
            .node_ids
            .iter()
            .filter_map(|identifier| graph.nodes.get(identifier))
            .map(|node| node.bounds());
        let first = node_bounds.next().expect("group contains a node");
        let (minimum_x, minimum_y, maximum_x, maximum_y) = node_bounds.fold(
            (
                first.origin.x,
                first.origin.y,
                first.origin.x + first.size.width,
                first.origin.y + first.size.height,
            ),
            |(minimum_x, minimum_y, maximum_x, maximum_y), bounds| {
                (
                    minimum_x.min(bounds.origin.x),
                    minimum_y.min(bounds.origin.y),
                    maximum_x.max(bounds.origin.x + bounds.size.width),
                    maximum_y.max(bounds.origin.y + bounds.size.height),
                )
            },
        );
        assert_eq!(group.bounds.origin.x, minimum_x - padding);
        assert_eq!(group.bounds.origin.y, minimum_y - padding);
        assert_eq!(
            group.bounds.size.width,
            maximum_x - minimum_x + padding * 2.0
        );
        assert_eq!(
            group.bounds.size.height,
            maximum_y - minimum_y + padding * 2.0
        );
    }

    #[gpui::test(seed = 16024)]
    async fn renderer_toggle_is_committed_only_by_settings_store(cx: &mut TestAppContext) {
        let fs = fs::FakeFs::new(cx.executor());
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            <dyn fs::Fs>::set_global(fs, cx);
        });
        let (item, cx) = cx.add_window_view(|_, cx| {
            GraphWorkspaceItem::new(
                GraphWorkspaceModel::create("Settings-owned renderer")
                    .expect("create graph workspace"),
                WeakEntity::new_invalid(),
                cx,
            )
        });
        let before = item.read_with(cx, |item, cx| {
            assert!(item.native_node_renderer_enabled(cx));
            workflow_evidence(item)
        });

        assert!(item.update(cx, |item, cx| item.execute_catalog_action(
            CatalogGraphAction::ToggleVueNodes,
            GraphActionInput::None,
            cx,
        )));
        cx.run_until_parked();

        item.read_with(cx, |item, cx| {
            assert!(!item.native_node_renderer_enabled(cx));
            assert_eq!(item.context_settings_task.is_none(), true);
            assert_eq!(
                item.model.announcement.as_deref(),
                Some("Compact native node renderer enabled")
            );
            assert_eq!(workflow_evidence(item), before);
        });
    }

    #[gpui::test(seed = 16025)]
    async fn unavailable_settings_owner_rejects_toggle_without_graph_mutation(
        cx: &mut TestAppContext,
    ) {
        let (item, cx) = cx.add_window_view(|_, cx| {
            GraphWorkspaceItem::new(
                GraphWorkspaceModel::create("Missing settings owner")
                    .expect("create graph workspace"),
                WeakEntity::new_invalid(),
                cx,
            )
        });
        let before = item.read_with(cx, |item, _| workflow_evidence(item));

        assert!(!item.update(cx, |item, cx| item.execute_catalog_action(
            CatalogGraphAction::ToggleVueNodes,
            GraphActionInput::None,
            cx,
        )));

        item.read_with(cx, |item, _| {
            assert_eq!(
                item.model.last_error.as_deref(),
                Some("Zed settings store is unavailable")
            );
            assert_eq!(workflow_evidence(item), before);
        });
    }

    #[gpui::test(seed = 16026)]
    fn group_selection_padding_uses_settings_store_bounds(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let (item, cx) = cx.add_window_view(|_, cx| {
            GraphWorkspaceItem::new(
                GraphWorkspaceModel::create("Settings-owned group padding")
                    .expect("create graph workspace"),
                WeakEntity::new_invalid(),
                cx,
            )
        });
        assert_eq!(
            item.read_with(cx, |item, cx| item.group_selected_nodes_padding(cx)),
            10.0
        );
        let notify_count = Rc::new(Cell::new(0));
        let _item_observation = cx.update(|_, cx| {
            let notify_count = notify_count.clone();
            cx.observe(&item, move |_, _| notify_count.set(notify_count.get() + 1))
        });

        cx.update(|_, cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    let sim_runtime = settings.sim_runtime.get_or_insert_default();
                    sim_runtime.group_selected_nodes_padding = Some(-5.0);
                    sim_runtime.native_node_renderer = Some(false);
                });
            });
        });
        cx.run_until_parked();
        assert!(
            notify_count.get() > 0,
            "external SettingsStore changes must notify the graph item"
        );
        assert!(!item.read_with(cx, |item, cx| item.native_node_renderer_enabled(cx)));
        assert_eq!(
            item.read_with(cx, |item, cx| item.group_selected_nodes_padding(cx)),
            0.0
        );

        cx.update(|_, cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .sim_runtime
                        .get_or_insert_default()
                        .group_selected_nodes_padding = Some(105.0);
                });
            });
        });
        assert_eq!(
            item.read_with(cx, |item, cx| item.group_selected_nodes_padding(cx)),
            100.0
        );
    }

    #[gpui::test(seed = 16027)]
    fn group_and_fit_actions_use_settings_padding_with_exact_undo(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let mut settings_store = SettingsStore::test(cx);
            settings_store.update_user_settings(cx, |settings| {
                settings
                    .sim_runtime
                    .get_or_insert_default()
                    .group_selected_nodes_padding = Some(37.5);
            });
            cx.set_global(settings_store);
        });
        let (item, cx) = cx.add_window_view(|_, cx| {
            GraphWorkspaceItem::new(
                crate::graph_tests::fixture_model().expect("create selected graph fixture"),
                WeakEntity::new_invalid(),
                cx,
            )
        });
        let workflow_before = item.read_with(cx, |item, _| {
            item.model
                .document()
                .expect("editable graph document")
                .to_workflow_bytes()
                .expect("serialize graph workflow")
        });

        assert!(item.update(cx, |item, cx| item.execute_catalog_action(
            CatalogGraphAction::GroupSelectedNodes,
            GraphActionInput::None,
            cx,
        )));
        let group_identifier = item.read_with(cx, |item, _| {
            let identifier = item
                .model
                .document()
                .expect("editable graph document")
                .active_graph()
                .expect("active graph")
                .groups
                .keys()
                .next()
                .cloned()
                .expect("settings-derived group");
            assert_group_padding(item, &identifier, 37.5);
            identifier
        });

        cx.update(|_, cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .sim_runtime
                        .get_or_insert_default()
                        .group_selected_nodes_padding = Some(5.0);
                });
            });
        });
        assert!(item.update(cx, |item, cx| item.execute_catalog_action(
            CatalogGraphAction::FitGroupToContents,
            GraphActionInput::GroupIdentifier(group_identifier.clone()),
            cx,
        )));
        item.read_with(cx, |item, _| {
            assert_group_padding(item, &group_identifier, 5.0)
        });

        item.update(cx, |item, _| {
            assert!(item.model.undo().expect("undo settings-derived fit"));
            assert_group_padding(item, &group_identifier, 37.5);
            assert!(item.model.undo().expect("undo settings-derived group"));
            assert_eq!(
                item.model
                    .document()
                    .expect("editable graph document")
                    .to_workflow_bytes()
                    .expect("serialize graph workflow"),
                workflow_before
            );
            assert!(!item.model.undo().expect("settings actions have exact undo"));
        });
    }
}

#[cfg(all(test, feature = "test-support"))]
mod lifecycle_tests {
    use super::*;
    use gpui::TestAppContext;
    use project::{FakeFs, Fs as _};
    use serde_json::json;

    const BASE_WORKFLOW: &[u8] = br#"{"version":0.4,"last_node_id":0,"last_link_id":0,"nodes":[],"links":[],"groups":[],"config":{},"extra":{}}"#;

    fn init_lifecycle_test(cx: &mut TestAppContext) -> Arc<workspace::AppState> {
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            workspace::AppState::test(cx)
        })
    }

    fn local_model(path: &Path) -> GraphWorkspaceModel {
        GraphWorkspaceModel::open(
            "workflow.json",
            path.to_string_lossy(),
            WorkflowStorageProvider::LocalFile,
            BASE_WORKFLOW.to_vec(),
        )
        .expect("open local workflow")
    }

    #[gpui::test(seed = 16107)]
    async fn project_opened_workflow_binds_workspace_for_node_properties(cx: &mut TestAppContext) {
        let file_system = FakeFs::new(cx.executor());
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            <dyn fs::Fs>::set_global(file_system.clone(), cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            workspace::AppState::test(cx);
            crate::init(cx);
        });
        file_system
            .insert_tree(
                Path::new("/project"),
                json!({
                    "image.workflow.json": json!({
                        "version": 0.4,
                        "nodes": [{
                            "id": 1,
                            "type": "LoadImage",
                            "pos": [0, 0],
                            "size": [240, 100],
                            "inputs": [],
                            "outputs": [],
                            "properties": {"widget_input_names": ["image"]},
                            "widgets_values": ["selected.png"]
                        }],
                        "links": []
                    }).to_string()
                }),
            )
            .await;
        let project = Project::test(file_system, [Path::new("/project")], cx).await;
        let workflow_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("project/image.workflow.json", cx)
            })
            .expect("resolve the native workflow project file");
        let (workspace, window) =
            cx.add_window_view(|window, cx| Workspace::test_new(project, window, cx));
        let panel = workspace.update_in(window, |workspace, window, cx| {
            let workspace_handle = workspace.weak_handle();
            let panel =
                cx.new(|cx| crate::GraphPropertiesPanel::test_new(workspace_handle, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        let graph = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_path(workflow_path, None, true, window, cx)
            })
            .await
            .expect("open the registered native workflow project item")
            .downcast::<GraphWorkspaceItem>()
            .expect("project file produces the canonical graph workspace item");
        let bound_workspace = graph.read_with(window, |graph, _| {
            graph
                .workspace()
                .upgrade()
                .expect("project-opened graph retains its actual workspace")
        });
        assert_eq!(bound_workspace.entity_id(), workspace.entity_id());
        bound_workspace.update_in(window, |workspace, window, cx| {
            crate::open_for_graph_node(workspace, graph, GraphIdentifier::Integer(1), window, cx)
                .expect("open node properties through the graph's retained workspace");
        });
        window.run_until_parked();
        assert_eq!(
            panel.read_with(window, |panel, _| panel.target_for_test()),
            Some(GraphIdentifier::Integer(1))
        );
    }

    #[gpui::test(seed = 16101)]
    async fn local_save_detects_conflict_and_reload_uses_disk(cx: &mut TestAppContext) {
        init_lifecycle_test(cx);
        let path = Path::new("/project/workflow.json");
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            Path::new("/project"),
            json!({"workflow.json": String::from_utf8_lossy(BASE_WORKFLOW)}),
        )
        .await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let mut model = local_model(path);
        model
            .apply(GraphCommand::PanViewport {
                delta: GraphPoint { x: 20.0, y: 10.0 },
            })
            .expect("edit local workflow");
        let (item, cx) = cx
            .add_window_view(|_, cx| GraphWorkspaceItem::new(model, WeakEntity::new_invalid(), cx));
        item.update(cx, |item, cx| {
            item.start_local_file_watch(fs.clone(), path.to_path_buf(), cx)
        });
        cx.background_executor.timer(Duration::from_millis(1)).await;

        assert!(item.read_with(cx, |item, cx| item.can_save(cx)));
        let save = item.update_in(cx, |item, window, cx| {
            item.save(Default::default(), project.clone(), window, cx)
        });
        save.await.expect("save local workflow");
        let saved_bytes = fs.load_bytes(path).await.expect("read saved workflow");
        item.read_with(cx, |item, _| {
            assert_eq!(
                item.model.save_coordinator.authority(),
                WorkflowAuthority::InSync
            );
            assert_eq!(saved_bytes, item.model.save_coordinator.base().bytes);
            assert!(item.model.last_error.is_none());
        });

        item.update(cx, |item, cx| {
            assert!(item.apply_graph_command(
                GraphCommand::PanViewport {
                    delta: GraphPoint { x: 5.0, y: -5.0 },
                },
                cx,
            ));
        });
        let external = br#"{"version":0.4,"last_node_id":0,"last_link_id":0,"nodes":[],"links":[],"groups":[],"config":{},"extra":{},"external":true}"#.to_vec();
        fs.insert_file(path, external.clone()).await;
        cx.background_executor
            .timer(Duration::from_millis(150))
            .await;
        item.read_with(cx, |item, _| {
            assert_eq!(
                item.model.save_coordinator.authority(),
                WorkflowAuthority::Conflict
            );
            assert_eq!(
                item.model
                    .save_coordinator
                    .external()
                    .map(|external| external.bytes.as_slice()),
                Some(external.as_slice())
            );
        });
        let conflict = item.update_in(cx, |item, window, cx| {
            item.save(Default::default(), project.clone(), window, cx)
        });
        assert!(conflict.await.is_err());
        item.read_with(cx, |item, _| {
            assert_eq!(
                item.model.save_coordinator.authority(),
                WorkflowAuthority::Conflict
            );
            assert!(item.model.last_error.is_some());
            assert!(item.model.save_coordinator.external().is_some());
        });

        let reload = item.update_in(cx, |item, window, cx| {
            item.reload(project.clone(), window, cx)
        });
        reload.await.expect("reload external workflow");
        item.read_with(cx, |item, _| {
            assert_eq!(
                item.model.save_coordinator.authority(),
                WorkflowAuthority::InSync
            );
            assert_eq!(item.model.save_coordinator.base().bytes, external);
            assert!(item.model.last_error.is_none());
        });

        let malformed = br#"{"version":1,"nodes":"not-an-array","links":[]}"#.to_vec();
        fs.insert_file(path, malformed.clone()).await;
        let reload = item.update_in(cx, |item, window, cx| {
            item.reload(project.clone(), window, cx)
        });
        reload
            .await
            .expect("reload malformed workflow as preserved read-only data");
        item.read_with(cx, |item, cx| {
            assert!(item.model.is_read_only());
            assert_eq!(item.model.original_bytes(), malformed);
            assert!(!item.can_save(cx));
            assert!(
                item.model
                    .announcement
                    .as_deref()
                    .is_some_and(|message| message.contains("read-only"))
            );
        });
    }

    #[gpui::test(seed = 16102)]
    async fn save_as_retargets_draft_to_native_local_file(cx: &mut TestAppContext) {
        init_lifecycle_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), json!({})).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let target = project
            .read_with(cx, |project, cx| {
                project.find_project_path("project/saved-workflow.json", cx)
            })
            .expect("resolve save-as path");
        let model = GraphWorkspaceModel::create("Untitled workflow").expect("create draft");
        let (item, cx) = cx
            .add_window_view(|_, cx| GraphWorkspaceItem::new(model, WeakEntity::new_invalid(), cx));

        assert!(item.read_with(cx, |item, cx| item.can_save_as(cx)));
        let save = item.update_in(cx, |item, window, cx| {
            item.save_as(project.clone(), target, window, cx)
        });
        save.await.expect("save draft as local workflow");

        let path = Path::new("/project/saved-workflow.json");
        let bytes = fs.load_bytes(path).await.expect("read save-as target");
        item.read_with(cx, |item, cx| {
            assert_eq!(
                item.model.save_coordinator.provider(),
                &WorkflowStorageProvider::LocalFile
            );
            assert_eq!(
                item.model.save_coordinator.document_identity(),
                path.to_string_lossy()
            );
            assert_eq!(item.model.save_coordinator.base().bytes, bytes);
            assert_eq!(item.model.title, "saved-workflow.json");
            assert!(item.can_save(cx));
        });
    }

    fn init_save_workflow_adapter_test(cx: &mut TestAppContext, file_system: &Arc<FakeFs>) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            <dyn fs::Fs>::set_global(file_system.clone(), cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        init_lifecycle_test(cx);
    }

    #[gpui::test(seed = 16108)]
    async fn save_workflow_adapter_uses_canonical_local_and_draft_save(cx: &mut TestAppContext) {
        let file_system = FakeFs::new(cx.executor());
        init_save_workflow_adapter_test(cx, &file_system);
        let path = Path::new("/project/workflow.json");
        file_system
            .insert_tree(
                Path::new("/project"),
                json!({"workflow.json": String::from_utf8_lossy(BASE_WORKFLOW)}),
            )
            .await;
        let project = Project::test(file_system.clone(), [Path::new("/project")], cx).await;
        let (workspace, window) =
            cx.add_window_view(|window, cx| Workspace::test_new(project, window, cx));
        let mut model = local_model(path);
        model
            .apply(GraphCommand::PanViewport {
                delta: GraphPoint { x: 21.0, y: 8.0 },
            })
            .expect("edit local workflow");
        let graph = workspace.update_in(window, |workspace, window, cx| {
            let graph = cx.new(|cx| GraphWorkspaceItem::new(model, workspace.weak_handle(), cx));
            workspace.add_item_to_active_pane(Box::new(graph.clone()), None, true, window, cx);
            graph
        });
        let save = graph.update_in(window, |graph, window, cx| {
            graph.save_workflow_in_workspace(window, cx)
        });
        save.await
            .expect("save through the canonical workspace and pane");
        let bytes = file_system
            .load_bytes(path)
            .await
            .expect("read canonical save");
        graph.read_with(window, |graph, _| {
            assert_eq!(
                graph.model.save_coordinator.authority(),
                WorkflowAuthority::InSync
            );
            assert_eq!(graph.model.save_coordinator.base().bytes, bytes);
            assert_ne!(bytes, BASE_WORKFLOW);
            assert!(graph.model.last_error.is_none());
        });
        let draft = workspace.update_in(window, |workspace, window, cx| {
            let mut model = GraphWorkspaceModel::create("Draft workflow").expect("create draft");
            model
                .apply(GraphCommand::PanViewport {
                    delta: GraphPoint { x: 6.0, y: 3.0 },
                })
                .expect("edit draft");
            let draft = cx.new(|cx| GraphWorkspaceItem::new(model, workspace.weak_handle(), cx));
            workspace.add_item_to_active_pane(Box::new(draft.clone()), None, true, window, cx);
            draft
        });
        let save = draft.update_in(window, |draft, window, cx| {
            draft.save_workflow_in_workspace(window, cx)
        });
        save.await
            .expect("save draft through the same canonical owner");
        draft.read_with(window, |draft, _| {
            assert_eq!(
                draft.model.save_coordinator.provider(),
                &WorkflowStorageProvider::Draft
            );
            assert_eq!(
                draft.model.save_coordinator.authority(),
                WorkflowAuthority::InSync
            );
            assert!(draft.model.last_error.is_none());
        });
        assert_eq!(
            file_system
                .load_bytes(path)
                .await
                .expect("local file remains unchanged"),
            bytes
        );
    }

    #[gpui::test(seed = 16109)]
    async fn save_workflow_adapter_preserves_conflict_when_cancelled(cx: &mut TestAppContext) {
        let file_system = FakeFs::new(cx.executor());
        init_save_workflow_adapter_test(cx, &file_system);
        let path = Path::new("/project/workflow.json");
        let external = br#"{"version":0.4,"nodes":[],"links":[],"external":true}"#.to_vec();
        file_system
            .insert_tree(
                Path::new("/project"),
                json!({"workflow.json": String::from_utf8_lossy(&external)}),
            )
            .await;
        let project = Project::test(file_system.clone(), [Path::new("/project")], cx).await;
        let (workspace, window) =
            cx.add_window_view(|window, cx| Workspace::test_new(project, window, cx));
        let mut model = local_model(path);
        model
            .apply(GraphCommand::PanViewport {
                delta: GraphPoint { x: 15.0, y: 9.0 },
            })
            .expect("edit local workflow");
        model
            .observe_external_change(external.clone())
            .expect("record real external conflict");
        let local = model.encode().expect("retain edited workflow");
        let graph = workspace.update_in(window, |workspace, window, cx| {
            let graph = cx.new(|cx| GraphWorkspaceItem::new(model, workspace.weak_handle(), cx));
            workspace.add_item_to_active_pane(Box::new(graph.clone()), None, true, window, cx);
            graph
        });
        let save = graph.update_in(window, |graph, window, cx| {
            graph.save_workflow_in_workspace(window, cx)
        });
        window.run_until_parked();
        assert!(
            window
                .pending_prompt()
                .expect("canonical conflict prompt")
                .0
                .contains("changed on disk")
        );
        window.simulate_prompt_answer("Cancel");
        save.await
            .expect("canonical user cancellation is not a save failure");
        assert_eq!(
            file_system
                .load_bytes(path)
                .await
                .expect("read unchanged external bytes"),
            external
        );
        graph.read_with(window, |graph, _| {
            assert_eq!(
                graph.model.save_coordinator.authority(),
                WorkflowAuthority::Conflict
            );
            assert_eq!(graph.model.encode().expect("retain local edit"), local);
            assert_eq!(
                graph
                    .model
                    .save_coordinator
                    .external()
                    .expect("retain external revision")
                    .bytes,
                external
            );
        });
    }

    #[gpui::test(seed = 16110)]
    async fn save_workflow_adapter_rejects_stale_and_missing_workspace(cx: &mut TestAppContext) {
        let file_system = FakeFs::new(cx.executor());
        init_save_workflow_adapter_test(cx, &file_system);
        let project = Project::test(file_system, [], cx).await;
        let (workspace, window) =
            cx.add_window_view(|window, cx| Workspace::test_new(project, window, cx));
        let first = workspace.update_in(window, |workspace, window, cx| {
            let model = GraphWorkspaceModel::create("First draft").expect("first draft");
            let graph = cx.new(|cx| GraphWorkspaceItem::new(model, workspace.weak_handle(), cx));
            workspace.add_item_to_active_pane(Box::new(graph.clone()), None, true, window, cx);
            graph
        });
        let save = first.update_in(window, |graph, window, cx| {
            graph.save_workflow_in_workspace(window, cx)
        });
        let second = workspace.update_in(window, |workspace, window, cx| {
            let mut model = GraphWorkspaceModel::create("Second draft").expect("second draft");
            model
                .apply(GraphCommand::PanViewport {
                    delta: GraphPoint { x: 3.0, y: 7.0 },
                })
                .expect("edit second draft");
            let graph = cx.new(|cx| GraphWorkspaceItem::new(model, workspace.weak_handle(), cx));
            workspace.add_item_to_active_pane(Box::new(graph.clone()), None, true, window, cx);
            graph
        });
        let error = save
            .await
            .expect_err("stale action must not save the new active graph");
        assert!(error.to_string().contains("no longer active"));
        second.read_with(window, |graph, _| {
            assert_eq!(
                graph.model.save_coordinator.authority(),
                WorkflowAuthority::LocalDirty
            );
        });
        first.update(window, |graph, _| {
            graph.workspace = WeakEntity::new_invalid()
        });
        let save = first.update_in(window, |graph, window, cx| {
            graph.save_workflow_in_workspace(window, cx)
        });
        let error = save
            .await
            .expect_err("missing owning workspace must propagate");
        assert!(
            error
                .to_string()
                .contains("owning workspace is unavailable")
        );
    }

    #[gpui::test(seed = 16103)]
    async fn provider_save_is_explicitly_unsupported(cx: &mut TestAppContext) {
        init_lifecycle_test(cx);
        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let model = GraphWorkspaceModel::open(
            "Provider workflow",
            "provider-document",
            WorkflowStorageProvider::Provider {
                identifier: "example.provider".to_owned(),
            },
            BASE_WORKFLOW.to_vec(),
        )
        .expect("open provider workflow");
        let (item, cx) = cx
            .add_window_view(|_, cx| GraphWorkspaceItem::new(model, WeakEntity::new_invalid(), cx));

        item.read_with(cx, |item, cx| {
            assert!(!item.can_save(cx));
            assert!(item.can_save_as(cx));
        });
        let save = item.update_in(cx, |item, window, cx| {
            item.save(Default::default(), project, window, cx)
        });
        let error = save.await.expect_err("provider save must fail");
        assert!(
            error
                .to_string()
                .contains("does not support native local save")
        );
        assert!(item.read_with(cx, |item, _| item.model.last_error.is_some()));
    }

    #[gpui::test(seed = 16104)]
    async fn workflow_snapshots_are_isolated_by_workspace_and_item(cx: &mut TestAppContext) {
        init_lifecycle_test(cx);
        let workspace_database = cx.update(|cx| workspace::WorkspaceDb::global(cx));
        let first_workspace_id = workspace_database
            .next_id()
            .await
            .expect("create first workspace identity");
        let second_workspace_id = workspace_database
            .next_id()
            .await
            .expect("create second workspace identity");
        let database = cx.update(|cx| persistence::SimWorkflowDb::global(cx));
        let shared_item_id = 41;
        let retained_item_id = 42;

        database
            .save_snapshot(
                shared_item_id,
                first_workspace_id,
                "first workspace".to_owned(),
            )
            .await
            .expect("save first workspace snapshot");
        database
            .save_snapshot(
                shared_item_id,
                second_workspace_id,
                "second workspace".to_owned(),
            )
            .await
            .expect("save second workspace snapshot with the same item identity");
        database
            .save_snapshot(
                retained_item_id,
                first_workspace_id,
                "retained snapshot".to_owned(),
            )
            .await
            .expect("save retained snapshot");
        database
            .save_snapshot(
                shared_item_id,
                first_workspace_id,
                "updated first workspace".to_owned(),
            )
            .await
            .expect("update only the first workspace snapshot");

        assert_eq!(
            database
                .get_snapshot(shared_item_id, first_workspace_id)
                .expect("load updated first workspace snapshot")
                .as_deref(),
            Some("updated first workspace")
        );
        assert_eq!(
            database
                .get_snapshot(shared_item_id, second_workspace_id)
                .expect("load second workspace snapshot")
                .as_deref(),
            Some("second workspace")
        );

        let model = GraphWorkspaceModel::create("Cleanup test").expect("create graph model");
        let (_, cx) = cx
            .add_window_view(|_, cx| GraphWorkspaceItem::new(model, WeakEntity::new_invalid(), cx));
        let cleanup = cx.update(|window, cx| {
            <GraphWorkspaceItem as SerializableItem>::cleanup(
                first_workspace_id,
                vec![retained_item_id],
                window,
                cx,
            )
        });
        cleanup.await.expect("clean up first workspace snapshots");

        assert_eq!(
            database
                .get_snapshot(shared_item_id, first_workspace_id)
                .expect("confirm stale first workspace snapshot was removed"),
            None
        );
        assert_eq!(
            database
                .get_snapshot(retained_item_id, first_workspace_id)
                .expect("confirm alive first workspace snapshot was retained")
                .as_deref(),
            Some("retained snapshot")
        );
        assert_eq!(
            database
                .get_snapshot(shared_item_id, second_workspace_id)
                .expect("confirm cleanup did not cross workspace boundary")
                .as_deref(),
            Some("second workspace")
        );
    }

    #[gpui::test(seed = 16105)]
    async fn serializable_item_round_trip_uses_sim_workflow_database(cx: &mut TestAppContext) {
        let app_state = init_lifecycle_test(cx);
        cx.update(crate::init);
        let workspace_database = cx.update(|cx| workspace::WorkspaceDb::global(cx));
        let workspace_id = workspace_database
            .next_id()
            .await
            .expect("create workspace identity");
        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let project_for_workspace = project.clone();
        let (workspace, cx) = cx.add_window_view(|window, cx| {
            Workspace::new(
                Some(workspace_id),
                project_for_workspace,
                app_state.clone(),
                window,
                cx,
            )
        });
        let mut model = GraphWorkspaceModel::create("Persistent workflow")
            .expect("create persistent graph model");
        model
            .apply(GraphCommand::PanViewport {
                delta: GraphPoint { x: 15.0, y: -8.0 },
            })
            .expect("change persistent graph workspace state");
        let item = cx.new(|cx| GraphWorkspaceItem::new(model, workspace.downgrade(), cx));
        let item_id = item.entity_id().as_u64();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
        });
        let expected_document = item.read_with(cx, |item, _| {
            item.model
                .document()
                .cloned()
                .expect("editable workspace has a graph document")
        });
        let expected_snapshot = item.read_with(cx, |item, _| {
            String::from_utf8(
                item.model
                    .encode()
                    .expect("encode expected workspace snapshot"),
            )
            .expect("workspace snapshot is UTF-8 JSON")
        });

        let serialization = workspace.update_in(cx, |workspace, window, cx| {
            item.update(cx, |item, cx| {
                <GraphWorkspaceItem as SerializableItem>::serialize(
                    item, workspace, item_id, false, window, cx,
                )
            })
        });
        serialization
            .expect("workspace has a database identity")
            .await
            .expect("serialize workspace item through ComfyWorkflowDb");

        let flush = workspace.update_in(cx, |workspace, window, cx| {
            workspace.flush_serialization(window, cx)
        });
        flush.await;

        let database = cx.update(|_, cx| persistence::SimWorkflowDb::global(cx));
        assert_eq!(
            database
                .get_snapshot(item_id, workspace_id)
                .expect("load serialized workspace item")
                .as_deref(),
            Some(expected_snapshot.as_str())
        );

        let restored_window = cx
            .update(|_, cx| {
                workspace::open_workspace_by_id(workspace_id, app_state.clone(), None, cx)
            })
            .await
            .expect("restore workspace topology through WorkspaceDb");
        let restored_workspace = restored_window
            .update(cx, |multi_workspace, _, _| {
                multi_workspace.workspace().clone()
            })
            .expect("read restored multi-workspace");
        let restored = restored_workspace
            .read_with(cx, |workspace, cx| {
                workspace.active_item_as::<GraphWorkspaceItem>(cx)
            })
            .expect("SerializableItemRegistry restored the native graph item");
        restored.read_with(cx, |item, _| {
            assert_eq!(item.model.title, "Persistent workflow");
            assert_eq!(item.model.document(), Some(&expected_document));
            assert_eq!(
                item.model.save_coordinator.local_bytes(),
                expected_document
                    .to_workflow_bytes()
                    .expect("encode expected graph document")
            );
            assert_eq!(
                item.model.announcement.as_deref(),
                Some("Workflow state restored")
            );
        });
    }

    #[gpui::test(seed = 16106)]
    async fn failed_serializable_item_write_preserves_prior_workspace_row(cx: &mut TestAppContext) {
        let app_state = init_lifecycle_test(cx);
        cx.update(crate::init);
        let workspace_database = cx.update(|cx| workspace::WorkspaceDb::global(cx));
        let workspace_id = workspace_database
            .next_id()
            .await
            .expect("create workspace identity");
        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let (workspace, cx) = cx.add_window_view(|window, cx| {
            Workspace::new(Some(workspace_id), project, app_state, window, cx)
        });
        let item = cx.new(|cx| {
            GraphWorkspaceItem::new(
                GraphWorkspaceModel::create("Prior workspace row")
                    .expect("create graph workspace model"),
                workspace.downgrade(),
                cx,
            )
        });
        let item_id = item.entity_id().as_u64();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
        });

        let initial_save = workspace.update_in(cx, |workspace, window, cx| {
            item.update(cx, |item, cx| {
                <GraphWorkspaceItem as SerializableItem>::serialize(
                    item, workspace, item_id, false, window, cx,
                )
            })
        });
        initial_save
            .expect("workspace has a database identity")
            .await
            .expect("persist prior workspace row");
        let database = cx.update(|_, cx| persistence::SimWorkflowDb::global(cx));
        let prior_snapshot = database
            .get_snapshot(item_id, workspace_id)
            .expect("read prior workspace row")
            .expect("prior workspace row exists");

        item.update(cx, |item, cx| {
            assert!(item.apply_graph_command(
                GraphCommand::PanViewport {
                    delta: GraphPoint { x: 31.0, y: -12.0 },
                },
                cx,
            ));
        });
        database
            .install_write_failure_for_tests()
            .await
            .expect("install deterministic write failure");
        let failed_save = workspace.update_in(cx, |workspace, window, cx| {
            item.update(cx, |item, cx| {
                <GraphWorkspaceItem as SerializableItem>::serialize(
                    item, workspace, item_id, false, window, cx,
                )
            })
        });
        let failure = failed_save
            .expect("workspace has a database identity")
            .await
            .expect_err("injected database failure must reach the caller");
        assert!(failure.to_string().contains("failed to persist"));
        assert_eq!(
            database
                .get_snapshot(item_id, workspace_id)
                .expect("read row after failed update")
                .as_deref(),
            Some(prior_snapshot.as_str()),
            "the canonical database transaction must retain the prior item row"
        );
        database
            .remove_write_failure_for_tests()
            .await
            .expect("remove deterministic write failure");

        let retry = workspace.update_in(cx, |workspace, window, cx| {
            item.update(cx, |item, cx| {
                <GraphWorkspaceItem as SerializableItem>::serialize(
                    item, workspace, item_id, false, window, cx,
                )
            })
        });
        retry
            .expect("workspace has a database identity")
            .await
            .expect("retry workspace item persistence");
        assert_ne!(
            database
                .get_snapshot(item_id, workspace_id)
                .expect("read retried row")
                .as_deref(),
            Some(prior_snapshot.as_str())
        );
    }
}
