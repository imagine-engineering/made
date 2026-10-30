use crate::execution_surfaces::{
    ErrorExplorerSurface, ErrorOverlaySurface, ExecutionJobTab, ExecutionProgressToastPhase,
    ExecutionSortMode, ExecutionSurfaceAction, ExecutionSurfaceActionHandler,
    ExecutionSurfaceRuntimeState, ExecutionWorkflowFilter, JobDetailsSurface, JobFiltersSurface,
    ProgressToastSurface, QueueNotificationSurface, attempt_matches_query,
    snapshot_allows_controls,
};
use crate::{
    ClearExecutionHistory, ExecutionUiEvent, HistoryPanelAction, HistoryPanelActionHandler,
    HistoryPanelContent, OutputView, OutputViewAction, OutputViewActionHandler, QueuePanelAction,
    QueuePanelActionHandler, QueuePanelContent, ToggleExecutionPanel, execution_ui_model,
};
use anyhow::anyhow;
use db::kvp::KeyValueStore;
use gpui::{
    App, AppContext as _, AsyncWindowContext, ClipboardItem, Context, Entity, EventEmitter,
    FocusHandle, Focusable, IntoElement, KeyDownEvent, Pixels, PromptLevel, Render, Role,
    Subscription, Task, WeakEntity, Window, px,
};
use serde::{Deserialize, Serialize};
use sim_runtime::{
    AttemptId, ExecutionCommandAck, ExecutionCommandOutcome, ExecutionControlCommandKind,
    ExecutionFailure, ExecutionSnapshot, ExecutionSnapshotStatus, ExternalNavigationPolicy,
    ProfileId,
};
#[cfg(all(test, feature = "test-support"))]
use std::collections::VecDeque;
#[cfg(all(test, feature = "test-support"))]
use std::sync::Mutex;
use std::{collections::HashSet, sync::Arc, time::Duration};
use ui::{Button, ButtonCommon, ButtonSize, IconName, prelude::*};
use uuid::Uuid;
use workspace::{
    Toast, Workspace,
    dock::{DockPosition, Panel, PanelEvent},
    notifications::NotificationId,
};

const EXECUTION_PANEL_KEY: &str = "sim-execution-panel";
pub(crate) const EXECUTION_ACTION_UNAVAILABLE_NOTIFICATION_ID: &str =
    "sim-execution-action-unavailable";
const EXECUTION_PANEL_STATE_SCHEMA: u16 = 2;
const EXECUTION_NOTIFICATION_DURATION: Duration = Duration::from_secs(4);
const EXECUTION_TERMINAL_TOAST_DURATION: Duration = Duration::from_secs(5);
const JOB_DETAILS_HOVER_CLOSE_DELAY: Duration = Duration::from_millis(100);
const EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY: usize = crate::EXECUTION_HISTORY_CAPACITY;
const EXECUTION_PANEL_FILTER_CHARACTER_CAPACITY: usize = 512;
const EXECUTION_PANEL_HISTORY_PAGE_CAPACITY: usize = 10_000;
const EXECUTION_PANEL_HISTORY_PAGE_SIZE_CAPACITY: usize = 200;
const EXECUTION_PANEL_SERIALIZED_STATE_BYTE_CAPACITY: usize = 2_097_152;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPanelTab {
    #[default]
    Queue,
    History,
    Output,
    Errors,
}

impl ExecutionPanelTab {
    const ALL: [Self; 4] = [Self::Queue, Self::History, Self::Output, Self::Errors];

    fn label(self) -> &'static str {
        match self {
            Self::Queue => "Queue",
            Self::History => "History",
            Self::Output => "Output",
            Self::Errors => "Errors",
        }
    }

    fn element_id(self) -> &'static str {
        match self {
            Self::Queue => "sim-execution-tab-queue",
            Self::History => "sim-execution-tab-history",
            Self::Output => "sim-execution-tab-output",
            Self::Errors => "sim-execution-tab-errors",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedExecutionPanelState {
    schema_version: u16,
    selected_tab: ExecutionPanelTab,
    selected_profile_id: Option<ProfileId>,
    selected_attempt_id: Option<AttemptId>,
    selected_output_id: Option<Uuid>,
    expanded_error_attempts: HashSet<AttemptId>,
    queue_filter: String,
    history_filter: String,
    history_page: usize,
    history_page_size: usize,
    show_progress: bool,
    #[serde(default = "default_output_auto_follow")]
    output_auto_follow: bool,
    docked_history: bool,
    #[serde(default)]
    selected_job_tab: ExecutionJobTab,
    #[serde(default)]
    job_search_query: String,
    #[serde(default)]
    workflow_filter: ExecutionWorkflowFilter,
    #[serde(default)]
    sort_mode: ExecutionSortMode,
    #[serde(default)]
    error_search_query: String,
    #[serde(default)]
    errors_all_collapsed: bool,
    #[serde(default)]
    collapsed_error_attempts: HashSet<AttemptId>,
    #[serde(default)]
    dismissed_error_overlay_attempts: HashSet<AttemptId>,
}

impl Default for PersistedExecutionPanelState {
    fn default() -> Self {
        Self {
            schema_version: EXECUTION_PANEL_STATE_SCHEMA,
            selected_tab: ExecutionPanelTab::Queue,
            selected_profile_id: None,
            selected_attempt_id: None,
            selected_output_id: None,
            expanded_error_attempts: HashSet::new(),
            queue_filter: String::new(),
            history_filter: String::new(),
            history_page: 0,
            history_page_size: 50,
            show_progress: true,
            output_auto_follow: true,
            docked_history: true,
            selected_job_tab: ExecutionJobTab::All,
            job_search_query: String::new(),
            workflow_filter: ExecutionWorkflowFilter::All,
            sort_mode: ExecutionSortMode::MostRecent,
            error_search_query: String::new(),
            errors_all_collapsed: false,
            collapsed_error_attempts: HashSet::new(),
            dismissed_error_overlay_attempts: HashSet::new(),
        }
    }
}

struct RecoveredExecutionPanelState {
    state: PersistedExecutionPanelState,
    persistence_error: Option<String>,
    rewrite: bool,
}

fn recover_execution_panel_state(serialized: Option<&str>) -> RecoveredExecutionPanelState {
    let Some(serialized) = serialized else {
        return RecoveredExecutionPanelState {
            state: PersistedExecutionPanelState::default(),
            persistence_error: None,
            rewrite: false,
        };
    };
    if serialized.len() > EXECUTION_PANEL_SERIALIZED_STATE_BYTE_CAPACITY {
        return RecoveredExecutionPanelState {
            state: PersistedExecutionPanelState::default(),
            persistence_error: Some(format!(
                "Native execution panel state exceeded the {} byte limit and was recovered with bounded defaults",
                EXECUTION_PANEL_SERIALIZED_STATE_BYTE_CAPACITY
            )),
            rewrite: true,
        };
    }
    match serde_json::from_str::<PersistedExecutionPanelState>(serialized) {
        Ok(mut state) if matches!(state.schema_version, 1 | EXECUTION_PANEL_STATE_SCHEMA) => {
            let migrated = state.schema_version == 1;
            state.schema_version = EXECUTION_PANEL_STATE_SCHEMA;
            let sanitized = sanitize_persisted_execution_panel_state(&mut state);
            RecoveredExecutionPanelState {
                state,
                persistence_error: sanitized.then(|| {
                    "Native execution panel state exceeded bounded limits and was safely normalized"
                        .to_owned()
                }),
                rewrite: migrated || sanitized,
            }
        }
        Ok(state) => RecoveredExecutionPanelState {
            persistence_error: Some(format!(
                "Unsupported native execution panel state schema {}; recovered with bounded defaults",
                state.schema_version
            )),
            state: PersistedExecutionPanelState::default(),
            rewrite: true,
        },
        Err(error) => RecoveredExecutionPanelState {
            state: PersistedExecutionPanelState::default(),
            persistence_error: Some(format!(
                "Malformed native execution panel state; recovered with bounded defaults: {error}"
            )),
            rewrite: true,
        },
    }
}

fn sanitize_persisted_execution_panel_state(state: &mut PersistedExecutionPanelState) -> bool {
    let mut changed = false;
    changed |= truncate_string_characters(
        &mut state.queue_filter,
        EXECUTION_PANEL_FILTER_CHARACTER_CAPACITY,
    );
    changed |= truncate_string_characters(
        &mut state.history_filter,
        EXECUTION_PANEL_FILTER_CHARACTER_CAPACITY,
    );
    changed |= truncate_string_characters(
        &mut state.job_search_query,
        EXECUTION_PANEL_FILTER_CHARACTER_CAPACITY,
    );
    changed |= truncate_string_characters(
        &mut state.error_search_query,
        EXECUTION_PANEL_FILTER_CHARACTER_CAPACITY,
    );
    let page = state
        .history_page
        .min(EXECUTION_PANEL_HISTORY_PAGE_CAPACITY);
    changed |= page != state.history_page;
    state.history_page = page;
    let page_size = state
        .history_page_size
        .clamp(1, EXECUTION_PANEL_HISTORY_PAGE_SIZE_CAPACITY);
    changed |= page_size != state.history_page_size;
    state.history_page_size = page_size;
    changed |= cap_attempt_id_set(&mut state.expanded_error_attempts);
    changed |= cap_attempt_id_set(&mut state.collapsed_error_attempts);
    changed |= cap_attempt_id_set(&mut state.dismissed_error_overlay_attempts);
    changed
}

fn truncate_string_characters(value: &mut String, capacity: usize) -> bool {
    if value.chars().count() <= capacity {
        return false;
    }
    *value = value.chars().take(capacity).collect();
    true
}

fn cap_attempt_id_set(attempt_ids: &mut HashSet<AttemptId>) -> bool {
    if attempt_ids.len() <= EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY {
        return false;
    }
    let mut bounded = attempt_ids.iter().copied().collect::<Vec<_>>();
    bounded.sort_by_key(|attempt_id| attempt_id.0);
    bounded.truncate(EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY);
    *attempt_ids = bounded.into_iter().collect();
    true
}

pub struct ExecutionPanel {
    workspace: WeakEntity<Workspace>,
    model: Entity<crate::ExecutionUiModel>,
    focus_handle: FocusHandle,
    clear_queue_focus_handle: FocusHandle,
    error_overlay_view_focus_handle: FocusHandle,
    error_overlay_dismiss_focus_handle: FocusHandle,
    selected_output_view_focus_handle: FocusHandle,
    external_navigation_policy: ExternalNavigationPolicy,
    state: PersistedExecutionPanelState,
    active: bool,
    status_message: Option<String>,
    persistence_error: Option<String>,
    pending_serialization: Option<Task<()>>,
    confirmation_task: Option<Task<()>>,
    confirmation_profile_id: Option<ProfileId>,
    surface_state: ExecutionSurfaceRuntimeState,
    notification_task: Option<Task<()>>,
    notification_task_identity: Option<u64>,
    progress_toast_task: Option<Task<()>>,
    progress_toast_task_identity: Option<u64>,
    job_details_attempt_id: Option<AttemptId>,
    job_details_hover_attempt_id: Option<AttemptId>,
    job_details_trigger_hovered: Option<AttemptId>,
    job_details_content_hovered: Option<AttemptId>,
    job_details_hover_close_task: Option<Task<()>>,
    job_context_attempt_id: Option<AttemptId>,
    known_in_progress_attempts: HashSet<AttemptId>,
    #[cfg(all(test, feature = "test-support"))]
    surface_action_trace: VecDeque<ExecutionSurfaceAction>,
    #[cfg(all(test, feature = "test-support"))]
    surface_action_invocation_trace: Arc<Mutex<VecDeque<ExecutionSurfaceAction>>>,
    #[cfg(all(test, feature = "test-support"))]
    surface_action_update_error: Arc<Mutex<Option<String>>>,
    _subscriptions: Vec<Subscription>,
}

const fn default_output_auto_follow() -> bool {
    true
}

fn in_progress_attempt_ids(snapshot: &ExecutionSnapshot) -> HashSet<AttemptId> {
    snapshot
        .attempts
        .iter()
        .rev()
        .filter(|attempt| !attempt.state.is_terminal())
        .take(EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY)
        .map(|attempt| attempt.attempt_id)
        .collect()
}

fn follow_latest_in_progress(
    state: &mut PersistedExecutionPanelState,
    snapshot: &ExecutionSnapshot,
) -> bool {
    let latest_attempt_id = snapshot
        .attempts
        .iter()
        .filter(|attempt| !attempt.state.is_terminal())
        .max_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.attempt_id.0.cmp(&right.attempt_id.0))
        })
        .map(|attempt| attempt.attempt_id);
    let Some(latest_attempt_id) = latest_attempt_id else {
        return false;
    };
    if state.selected_attempt_id == Some(latest_attempt_id) {
        return false;
    }
    state.selected_attempt_id = Some(latest_attempt_id);
    state.selected_output_id = None;
    true
}

#[cfg(all(test, feature = "test-support"))]
pub(crate) struct ExecutionPanelSurfaceStateForTest {
    pub schema_version: u16,
    pub selected_profile_id: Option<ProfileId>,
    pub selected_tab: ExecutionPanelTab,
    pub active: bool,
    pub docked_history: bool,
    pub persistence_error: Option<String>,
    pub selected_job_tab: ExecutionJobTab,
    pub job_search_query: String,
    pub workflow_filter: ExecutionWorkflowFilter,
    pub sort_mode: ExecutionSortMode,
    pub show_progress: bool,
    pub output_auto_follow: bool,
    pub selected_attempt_id: Option<AttemptId>,
    pub in_progress_attempt_ids: HashSet<AttemptId>,
    pub job_details_attempt_id: Option<AttemptId>,
    pub job_details_hover_attempt_id: Option<AttemptId>,
    pub job_details_trigger_hovered: Option<AttemptId>,
    pub job_details_content_hovered: Option<AttemptId>,
    pub job_context_attempt_id: Option<AttemptId>,
    pub error_search_query: String,
    pub errors_all_collapsed: bool,
    pub collapsed_error_attempts: HashSet<AttemptId>,
    pub dismissed_error_overlay_attempts: HashSet<AttemptId>,
    pub notification_count: usize,
    pub current_notification_identity: Option<u64>,
    pub current_notification: Option<(
        crate::execution_surfaces::ExecutionNotificationKind,
        Option<sim_types::RequestId>,
        usize,
        String,
    )>,
    pub progress_toast: Option<(u64, ExecutionProgressToastPhase)>,
    pub bounded_counts: crate::execution_surfaces::ExecutionSurfaceBoundedCountsForTest,
}

#[cfg(all(test, feature = "test-support"))]
pub(crate) struct PersistedExecutionPanelRecoveryForTest {
    pub schema_version: u16,
    pub selected_tab: ExecutionPanelTab,
    pub persistence_error: Option<String>,
    pub serialized_after_recovery: Option<String>,
}

impl ExecutionPanel {
    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn test_new(model: Entity<crate::ExecutionUiModel>, cx: &mut Context<Self>) -> Self {
        let model_subscription = cx.observe(&model, |this: &mut ExecutionPanel, _, cx| {
            this.reconcile_execution_surfaces(cx);
            cx.notify();
        });
        let model_event_subscription = cx.subscribe(
            &model,
            |this: &mut ExecutionPanel, _, event: &ExecutionUiEvent, cx| {
                this.handle_model_event(event, cx);
            },
        );
        let mut state = PersistedExecutionPanelState::default();
        state.selected_profile_id = model.read(cx).active_profile_id();
        let mut surface_state = ExecutionSurfaceRuntimeState::default();
        let mut known_in_progress_attempts = HashSet::new();
        if let Some(profile_id) = state.selected_profile_id
            && let Ok(snapshot) = model.read(cx).snapshot(profile_id)
        {
            surface_state.prime(&snapshot);
            known_in_progress_attempts = in_progress_attempt_ids(&snapshot);
            follow_latest_in_progress(&mut state, &snapshot);
        }
        Self {
            workspace: WeakEntity::new_invalid(),
            model,
            focus_handle: cx.focus_handle(),
            clear_queue_focus_handle: cx.focus_handle(),
            error_overlay_view_focus_handle: cx.focus_handle(),
            error_overlay_dismiss_focus_handle: cx.focus_handle(),
            selected_output_view_focus_handle: cx.focus_handle(),
            external_navigation_policy: ExternalNavigationPolicy::https_user_gesture(),
            state,
            active: true,
            status_message: None,
            persistence_error: None,
            pending_serialization: None,
            confirmation_task: None,
            confirmation_profile_id: None,
            surface_state,
            notification_task: None,
            notification_task_identity: None,
            progress_toast_task: None,
            progress_toast_task_identity: None,
            job_details_attempt_id: None,
            job_details_hover_attempt_id: None,
            job_details_trigger_hovered: None,
            job_details_content_hovered: None,
            job_details_hover_close_task: None,
            job_context_attempt_id: None,
            known_in_progress_attempts,
            surface_action_trace: VecDeque::new(),
            surface_action_invocation_trace: Arc::new(Mutex::new(VecDeque::new())),
            surface_action_update_error: Arc::new(Mutex::new(None)),
            _subscriptions: vec![model_subscription, model_event_subscription],
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn set_test_state(
        &mut self,
        tab: ExecutionPanelTab,
        selected_attempt_id: Option<AttemptId>,
        queue_filter: &str,
        history_filter: &str,
        history_page: usize,
        cx: &mut Context<Self>,
    ) {
        self.state.selected_tab = tab;
        self.state.selected_attempt_id = selected_attempt_id;
        if selected_attempt_id.is_some() {
            self.state.output_auto_follow = false;
        }
        self.state.queue_filter = queue_filter.to_owned();
        self.state.history_filter = history_filter.to_owned();
        self.state.history_page = history_page;
        cx.notify();
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn set_output_auto_follow_for_test(
        &mut self,
        output_auto_follow: bool,
        cx: &mut Context<Self>,
    ) {
        self.state.output_auto_follow = output_auto_follow;
        cx.notify();
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn set_external_navigation_policy_for_test(
        &mut self,
        policy: ExternalNavigationPolicy,
    ) {
        self.external_navigation_policy = policy;
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn persisted_state_round_trip_for_test(&self) -> Result<String, serde_json::Error> {
        let encoded = serde_json::to_string(&self.state)?;
        let decoded = serde_json::from_str::<PersistedExecutionPanelState>(&encoded)?;
        serde_json::to_string(&decoded)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn handle_output_action_for_test(
        &mut self,
        action: OutputViewAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.handle_output_action(action, window, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn handle_surface_action_for_test(
        &mut self,
        action: ExecutionSurfaceAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.handle_surface_action(action, window, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn surface_state_for_test(&self) -> ExecutionPanelSurfaceStateForTest {
        ExecutionPanelSurfaceStateForTest {
            schema_version: self.state.schema_version,
            selected_profile_id: self.state.selected_profile_id,
            selected_tab: self.state.selected_tab,
            active: self.active,
            docked_history: self.state.docked_history,
            persistence_error: self.persistence_error.clone(),
            selected_job_tab: self.state.selected_job_tab,
            job_search_query: self.state.job_search_query.clone(),
            workflow_filter: self.state.workflow_filter,
            sort_mode: self.state.sort_mode,
            show_progress: self.state.show_progress,
            output_auto_follow: self.state.output_auto_follow,
            selected_attempt_id: self.state.selected_attempt_id,
            in_progress_attempt_ids: self.known_in_progress_attempts.clone(),
            job_details_attempt_id: self.job_details_attempt_id,
            job_details_hover_attempt_id: self.job_details_hover_attempt_id,
            job_details_trigger_hovered: self.job_details_trigger_hovered,
            job_details_content_hovered: self.job_details_content_hovered,
            job_context_attempt_id: self.job_context_attempt_id,
            error_search_query: self.state.error_search_query.clone(),
            errors_all_collapsed: self.state.errors_all_collapsed,
            collapsed_error_attempts: self.state.collapsed_error_attempts.clone(),
            dismissed_error_overlay_attempts: self.state.dismissed_error_overlay_attempts.clone(),
            notification_count: self.surface_state.notification_count_for_test(),
            current_notification_identity: self
                .surface_state
                .current_notification()
                .map(|notification| notification.identity),
            current_notification: self
                .surface_state
                .current_notification()
                .map(|notification| {
                    (
                        notification.kind,
                        notification.request_id,
                        notification.batch_count,
                        notification.message.clone(),
                    )
                }),
            progress_toast: self
                .surface_state
                .progress_toast()
                .map(|toast| (toast.identity, toast.phase)),
            bounded_counts: self.surface_state.bounded_counts_for_test(),
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn is_focused_for_test(&self, window: &Window) -> bool {
        self.focus_handle.is_focused(window)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn focus_clear_queue_for_test(&self, window: &mut Window, cx: &mut App) {
        self.clear_queue_focus_handle.focus(window, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn clear_queue_is_focused_for_test(&self, window: &Window) -> bool {
        self.clear_queue_focus_handle.is_focused(window)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn focus_error_overlay_view_for_test(&self, window: &mut Window, cx: &mut App) {
        self.error_overlay_view_focus_handle.focus(window, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn error_overlay_view_is_focused_for_test(&self, window: &Window) -> bool {
        self.error_overlay_view_focus_handle.is_focused(window)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn focus_error_overlay_dismiss_for_test(&self, window: &mut Window, cx: &mut App) {
        self.error_overlay_dismiss_focus_handle.focus(window, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn error_overlay_dismiss_is_focused_for_test(&self, window: &Window) -> bool {
        self.error_overlay_dismiss_focus_handle.is_focused(window)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn focus_selected_output_view_for_test(&self, window: &mut Window, cx: &mut App) {
        self.selected_output_view_focus_handle.focus(window, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn selected_output_view_is_focused_for_test(&self, window: &Window) -> bool {
        self.selected_output_view_focus_handle.is_focused(window)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn surface_action_trace_for_test(&self) -> Vec<ExecutionSurfaceAction> {
        self.surface_action_trace.iter().cloned().collect()
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn surface_action_invocation_trace_for_test(&self) -> Vec<ExecutionSurfaceAction> {
        match self.surface_action_invocation_trace.lock() {
            Ok(trace) => trace.iter().cloned().collect(),
            Err(error) => error.into_inner().iter().cloned().collect(),
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn surface_action_update_error_for_test(&self) -> Option<String> {
        match self.surface_action_update_error.lock() {
            Ok(error) => error.clone(),
            Err(error) => error.into_inner().clone(),
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn confirm_clear_history_for_test(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.confirm_clear_history(window, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn status_message_for_test(&self) -> Option<&str> {
        self.status_message.as_deref()
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn observe_prompt_queueing_for_test(
        &mut self,
        request_id: sim_types::RequestId,
        batch_count: usize,
        cx: &mut Context<Self>,
    ) {
        self.surface_state
            .observe_prompt_queueing(request_id, batch_count);
        self.schedule_notification_task(cx);
        cx.notify();
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn observe_prompt_queued_for_test(
        &mut self,
        request_id: sim_types::RequestId,
        batch_count: usize,
        attempt_id: AttemptId,
        cx: &mut Context<Self>,
    ) {
        self.surface_state
            .observe_prompt_queued(request_id, batch_count, attempt_id);
        self.schedule_notification_task(cx);
        cx.notify();
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn observe_prompt_queue_failed_for_test(
        &mut self,
        request_id: sim_types::RequestId,
        batch_count: usize,
        failure: &ExecutionFailure,
        cx: &mut Context<Self>,
    ) {
        self.surface_state
            .observe_prompt_queue_failed(request_id, batch_count, failure);
        self.schedule_notification_task(cx);
        cx.notify();
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn current_error_overlay_attempt_id_for_test(&self, cx: &App) -> Option<AttemptId> {
        self.snapshot(cx)
            .ok()
            .and_then(|snapshot| self.error_overlay_for_snapshot(&snapshot))
            .map(|(attempt_id, _)| attempt_id)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn filtered_failure_attempt_ids_for_test(&self, cx: &App) -> Vec<AttemptId> {
        let Ok(snapshot) = self.snapshot(cx) else {
            return Vec::new();
        };
        let mut failures = snapshot
            .attempts
            .iter()
            .filter_map(|attempt| attempt.failure.as_ref().map(|failure| (attempt, failure)))
            .filter(|(attempt, failure)| {
                crate::execution_surfaces::error_matches_query(
                    attempt,
                    failure,
                    &self.state.error_search_query,
                )
            })
            .collect::<Vec<_>>();
        failures.sort_by(|(left, _), (right, _)| {
            right
                .finished_at
                .cmp(&left.finished_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| right.attempt_id.0.cmp(&left.attempt_id.0))
        });
        failures
            .into_iter()
            .map(|(attempt, _)| attempt.attempt_id)
            .collect()
    }

    pub fn load(
        workspace: WeakEntity<Workspace>,
        cx: &mut AsyncWindowContext,
    ) -> Task<anyhow::Result<Entity<Self>>> {
        cx.spawn(async move |cx| {
            let serialization_key =
                workspace.read_with(cx, |workspace, _| Self::serialization_key(workspace))?;
            let mut persistence_error = None;
            let mut persisted = PersistedExecutionPanelState::default();
            if let Some(serialization_key) = serialization_key {
                match cx.update(|_, cx| KeyValueStore::global(cx)) {
                    Ok(key_value_store) => {
                        let read_key = serialization_key.clone();
                        let (key_value_store, serialized) = cx
                            .background_spawn(async move {
                                let serialized = key_value_store.read_kvp(&read_key);
                                (key_value_store, serialized)
                            })
                            .await;
                        match serialized {
                            Ok(serialized) => {
                                let recovery =
                                    recover_execution_panel_state(serialized.as_deref());
                                persisted = recovery.state;
                                persistence_error = recovery.persistence_error;
                                if recovery.rewrite {
                                    match serde_json::to_string(&persisted) {
                                        Ok(serialized) => {
                                            if let Err(error) = cx
                                                .background_spawn(async move {
                                                    key_value_store
                                                        .write_kvp(serialization_key, serialized)
                                                        .await
                                                })
                                                .await
                                            {
                                                let recovery_message = persistence_error
                                                    .get_or_insert_with(|| {
                                                        "Recovered native execution panel state with bounded defaults"
                                                            .to_owned()
                                                    });
                                                recovery_message.push_str(&format!(
                                                    "; failed to persist the recovered state: {error}"
                                                ));
                                            }
                                        }
                                        Err(error) => {
                                            persistence_error = Some(format!(
                                                "Recovered native execution panel state but failed to encode bounded defaults: {error}"
                                            ));
                                        }
                                    }
                                }
                            }
                            Err(error) => {
                                persistence_error = Some(format!(
                                    "Failed to read native execution panel state; recovered with bounded defaults: {error}"
                                ));
                            }
                        }
                    }
                    Err(error) => {
                        persistence_error = Some(format!(
                            "Failed to access native execution panel persistence; recovered with bounded defaults: {error}"
                        ));
                    }
                }
            }
            workspace.update_in(cx, |_workspace, _window, cx| {
                let workspace_handle = cx.entity().downgrade();
                let model = execution_ui_model(cx)
                    .ok_or_else(|| anyhow!("native execution UI model is not initialized"))?;
                Ok(cx.new(|cx| {
                    let model_subscription =
                        cx.observe(&model, |this: &mut ExecutionPanel, _, cx| {
                            this.reconcile_execution_surfaces(cx);
                            cx.notify();
                        });
                    let model_event_subscription = cx.subscribe(
                        &model,
                        |this: &mut ExecutionPanel, _, event: &ExecutionUiEvent, cx| {
                            this.handle_model_event(event, cx);
                        },
                    );
                    let mut state = persisted;
                    state.selected_profile_id = model.read(cx).active_profile_id();
                    let mut surface_state = ExecutionSurfaceRuntimeState::default();
                    let mut known_in_progress_attempts = HashSet::new();
                    if let Some(profile_id) = state.selected_profile_id
                        && let Ok(snapshot) = model.read(cx).snapshot(profile_id)
                    {
                        surface_state.prime(&snapshot);
                        known_in_progress_attempts = in_progress_attempt_ids(&snapshot);
                        follow_latest_in_progress(&mut state, &snapshot);
                    }
                    Self {
                        workspace: workspace_handle,
                        model,
                        focus_handle: cx.focus_handle(),
                        clear_queue_focus_handle: cx.focus_handle(),
                        error_overlay_view_focus_handle: cx.focus_handle(),
                        error_overlay_dismiss_focus_handle: cx.focus_handle(),
                        selected_output_view_focus_handle: cx.focus_handle(),
                        external_navigation_policy:
                            ExternalNavigationPolicy::https_user_gesture(),
                        state,
                        active: false,
                        status_message: None,
                        persistence_error,
                        pending_serialization: None,
                        confirmation_task: None,
                        confirmation_profile_id: None,
                        surface_state,
                        notification_task: None,
                        notification_task_identity: None,
                        progress_toast_task: None,
                        progress_toast_task_identity: None,
                        job_details_attempt_id: None,
                        job_details_hover_attempt_id: None,
                        job_details_trigger_hovered: None,
                        job_details_content_hovered: None,
                        job_details_hover_close_task: None,
                        job_context_attempt_id: None,
                        known_in_progress_attempts,
                        #[cfg(all(test, feature = "test-support"))]
                        surface_action_trace: VecDeque::new(),
                        #[cfg(all(test, feature = "test-support"))]
                        surface_action_invocation_trace: Arc::new(Mutex::new(VecDeque::new())),
                        #[cfg(all(test, feature = "test-support"))]
                        surface_action_update_error: Arc::new(Mutex::new(None)),
                        _subscriptions: vec![model_subscription, model_event_subscription],
                    }
                }))
            })?
        })
    }

    fn serialization_key(workspace: &Workspace) -> Option<String> {
        workspace
            .database_id()
            .map(|identifier| i64::from(identifier).to_string())
            .or(workspace.session_id())
            .map(|identifier| format!("{EXECUTION_PANEL_KEY}-{identifier:?}"))
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn recover_persisted_state_for_test(
        serialization_key: String,
        cx: &App,
    ) -> Task<PersistedExecutionPanelRecoveryForTest> {
        let key_value_store = KeyValueStore::global(cx);
        cx.background_spawn(async move {
            let mut recovery = match key_value_store.read_kvp(&serialization_key) {
                Ok(serialized) => recover_execution_panel_state(serialized.as_deref()),
                Err(error) => RecoveredExecutionPanelState {
                    state: PersistedExecutionPanelState::default(),
                    persistence_error: Some(format!(
                        "Failed to read native execution panel state; recovered with bounded defaults: {error}"
                    )),
                    rewrite: false,
                },
            };
            if recovery.rewrite {
                match serde_json::to_string(&recovery.state) {
                    Ok(serialized) => {
                        if let Err(error) = key_value_store
                            .write_kvp(serialization_key.clone(), serialized)
                            .await
                        {
                            let message = recovery.persistence_error.get_or_insert_with(|| {
                                "Recovered native execution panel state with bounded defaults"
                                    .to_owned()
                            });
                            message.push_str(&format!(
                                "; failed to persist the recovered state: {error}"
                            ));
                        }
                    }
                    Err(error) => {
                        recovery.persistence_error = Some(format!(
                            "Recovered native execution panel state but failed to encode bounded defaults: {error}"
                        ));
                    }
                }
            }
            let serialized_after_recovery = match key_value_store.read_kvp(&serialization_key) {
                Ok(serialized) => serialized,
                Err(error) => {
                    let message = recovery.persistence_error.get_or_insert_with(|| {
                        "Recovered native execution panel state with bounded defaults".to_owned()
                    });
                    message.push_str(&format!(
                        "; failed to verify the recovered persisted state: {error}"
                    ));
                    None
                }
            };
            PersistedExecutionPanelRecoveryForTest {
                schema_version: recovery.state.schema_version,
                selected_tab: recovery.state.selected_tab,
                persistence_error: recovery.persistence_error,
                serialized_after_recovery,
            }
        })
    }

    fn snapshot(&self, cx: &App) -> Result<ExecutionSnapshot, crate::ExecutionUiModelError> {
        self.model.read(cx).active_snapshot()
    }

    fn selected_attempt_id(&self, cx: &App) -> Option<AttemptId> {
        self.state.selected_attempt_id.or_else(|| {
            self.snapshot(cx)
                .ok()
                .and_then(|snapshot| snapshot.attempts.into_iter().rev().next())
                .map(|attempt| attempt.attempt_id)
        })
    }

    fn error_overlay_for_snapshot(
        &self,
        snapshot: &ExecutionSnapshot,
    ) -> Option<(AttemptId, usize)> {
        let failure_count = snapshot
            .attempts
            .iter()
            .filter(|attempt| attempt.failure.is_some())
            .count();
        snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                attempt.failure.is_some()
                    && !self
                        .state
                        .dismissed_error_overlay_attempts
                        .contains(&attempt.attempt_id)
            })
            .max_by(|left, right| {
                left.created_at
                    .cmp(&right.created_at)
                    .then_with(|| left.attempt_id.0.cmp(&right.attempt_id.0))
            })
            .map(|attempt| (attempt.attempt_id, failure_count))
    }

    fn reconcile_execution_surfaces(&mut self, cx: &mut Context<Self>) {
        let profile_changed = self.synchronize_active_profile(cx);
        let Ok(snapshot) = self.snapshot(cx) else {
            return;
        };
        let current_in_progress_attempts = in_progress_attempt_ids(&snapshot);
        let new_attempt_started = current_in_progress_attempts
            .iter()
            .any(|attempt_id| !self.known_in_progress_attempts.contains(attempt_id));
        self.known_in_progress_attempts = current_in_progress_attempts;
        let mut persistence_changed =
            profile_changed || self.prune_persisted_attempt_state(&snapshot);
        if new_attempt_started && !self.state.output_auto_follow {
            self.state.output_auto_follow = true;
            persistence_changed = true;
        }
        if self.state.output_auto_follow && follow_latest_in_progress(&mut self.state, &snapshot) {
            persistence_changed = true;
        }
        self.surface_state.reconcile(&snapshot);
        self.schedule_notification_task(cx);
        self.schedule_progress_toast_task(cx);
        if persistence_changed {
            self.serialize(cx);
        }
    }

    fn synchronize_active_profile(&mut self, cx: &App) -> bool {
        let active_profile_id = self.model.read(cx).active_profile_id();
        if self.state.selected_profile_id == active_profile_id {
            return false;
        }
        if self.confirmation_profile_id.take().is_some() {
            self.status_message = Some(
                "Destructive confirmation cancelled because the active execution profile changed"
                    .to_owned(),
            );
        }
        self.state.selected_profile_id = active_profile_id;
        self.state.selected_attempt_id = None;
        self.state.selected_output_id = None;
        self.state.expanded_error_attempts.clear();
        self.state.collapsed_error_attempts.clear();
        self.state.dismissed_error_overlay_attempts.clear();
        self.state.errors_all_collapsed = false;
        self.job_details_attempt_id = None;
        self.job_details_hover_attempt_id = None;
        self.job_context_attempt_id = None;
        self.known_in_progress_attempts.clear();
        self.surface_state = ExecutionSurfaceRuntimeState::default();
        true
    }

    fn prune_persisted_attempt_state(&mut self, snapshot: &ExecutionSnapshot) -> bool {
        let retained_attempts = snapshot
            .attempts
            .iter()
            .rev()
            .take(EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY)
            .map(|attempt| attempt.attempt_id)
            .collect::<HashSet<_>>();
        let retained_failures = snapshot
            .attempts
            .iter()
            .rev()
            .filter(|attempt| attempt.failure.is_some())
            .take(EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY)
            .map(|attempt| attempt.attempt_id)
            .collect::<HashSet<_>>();
        let previous_expanded = self.state.expanded_error_attempts.len();
        let previous_collapsed = self.state.collapsed_error_attempts.len();
        let previous_dismissed = self.state.dismissed_error_overlay_attempts.len();
        self.state
            .expanded_error_attempts
            .retain(|attempt_id| retained_attempts.contains(attempt_id));
        self.state
            .collapsed_error_attempts
            .retain(|attempt_id| retained_failures.contains(attempt_id));
        self.state
            .dismissed_error_overlay_attempts
            .retain(|attempt_id| retained_failures.contains(attempt_id));

        let mut changed = previous_expanded != self.state.expanded_error_attempts.len()
            || previous_collapsed != self.state.collapsed_error_attempts.len()
            || previous_dismissed != self.state.dismissed_error_overlay_attempts.len();
        if let Some(selected_attempt_id) = self.state.selected_attempt_id {
            if let Some(attempt) = snapshot
                .attempts
                .iter()
                .find(|attempt| attempt.attempt_id == selected_attempt_id)
            {
                if self
                    .state
                    .selected_output_id
                    .is_some_and(|selected_output_id| {
                        !attempt
                            .outputs
                            .iter()
                            .any(|output| output.output_id == selected_output_id)
                    })
                {
                    self.state.selected_output_id = None;
                    changed = true;
                }
            } else {
                self.state.selected_attempt_id = None;
                self.state.selected_output_id = None;
                changed = true;
            }
        } else if self.state.selected_output_id.take().is_some() {
            changed = true;
        }

        let failure_count = snapshot
            .attempts
            .iter()
            .filter(|attempt| attempt.failure.is_some())
            .count();
        let all_collapsed = failure_count > 0
            && failure_count <= EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY
            && self.state.collapsed_error_attempts.len() == failure_count;
        if self.state.errors_all_collapsed != all_collapsed {
            self.state.errors_all_collapsed = all_collapsed;
            changed = true;
        }
        changed
    }

    fn set_job_details_hover(
        &mut self,
        attempt_id: AttemptId,
        hovered: bool,
        content: bool,
        cx: &mut Context<Self>,
    ) {
        let hovered_source = if content {
            &mut self.job_details_content_hovered
        } else {
            &mut self.job_details_trigger_hovered
        };
        if hovered {
            self.job_details_hover_close_task = None;
            *hovered_source = Some(attempt_id);
            self.job_details_hover_attempt_id = Some(attempt_id);
            cx.notify();
            return;
        }
        if *hovered_source == Some(attempt_id) {
            *hovered_source = None;
        }
        if self.job_details_trigger_hovered.is_some() || self.job_details_content_hovered.is_some()
        {
            return;
        }
        let timer = cx
            .background_executor()
            .timer(JOB_DETAILS_HOVER_CLOSE_DELAY);
        self.job_details_hover_close_task = Some(cx.spawn(async move |this, cx| {
            timer.await;
            if let Err(error) = this.update(cx, |this, cx| {
                if this.job_details_trigger_hovered.is_none()
                    && this.job_details_content_hovered.is_none()
                {
                    this.job_details_hover_attempt_id = None;
                    this.job_details_hover_close_task = None;
                    cx.notify();
                }
            }) {
                log::error!("execution panel disappeared while closing job details: {error}");
            }
        }));
    }

    fn handle_model_event(&mut self, event: &ExecutionUiEvent, cx: &mut Context<Self>) {
        let profile_changed = self.synchronize_active_profile(cx);
        if profile_changed {
            self.reconcile_execution_surfaces(cx);
        }
        let selected_profile_id = self.state.selected_profile_id;
        match event {
            ExecutionUiEvent::CommandSubmitted {
                profile_id,
                request_id,
                queue_batch_count: Some(batch_count),
            } if selected_profile_id == Some(*profile_id) => {
                self.surface_state
                    .observe_prompt_queueing(*request_id, *batch_count);
                self.schedule_notification_task(cx);
                cx.notify();
            }
            ExecutionUiEvent::CommandAcknowledged {
                profile_id,
                request_id,
                queue_batch_count: Some(batch_count),
                outcome:
                    ExecutionCommandOutcome::Accepted {
                        assigned_attempt_id: Some(attempt_id),
                    },
            } if selected_profile_id == Some(*profile_id) => {
                self.surface_state
                    .observe_prompt_queued(*request_id, *batch_count, *attempt_id);
                self.schedule_notification_task(cx);
                cx.notify();
            }
            ExecutionUiEvent::CommandAcknowledged {
                profile_id,
                request_id,
                queue_batch_count: Some(batch_count),
                outcome: ExecutionCommandOutcome::Rejected { failure },
            } if selected_profile_id == Some(*profile_id) => {
                self.surface_state
                    .observe_prompt_queue_failed(*request_id, *batch_count, failure);
                self.schedule_notification_task(cx);
                cx.notify();
            }
            ExecutionUiEvent::Changed { .. }
            | ExecutionUiEvent::CommandSubmitted { .. }
            | ExecutionUiEvent::CommandAcknowledged { .. }
            | ExecutionUiEvent::Error(_) => {}
        }
    }

    fn schedule_notification_task(&mut self, cx: &mut Context<Self>) {
        let identity = self
            .surface_state
            .current_notification()
            .map(|notification| notification.identity);
        if identity == self.notification_task_identity {
            return;
        }
        self.notification_task = None;
        self.notification_task_identity = identity;
        let Some(identity) = identity else {
            return;
        };
        let timer = cx
            .background_executor()
            .timer(EXECUTION_NOTIFICATION_DURATION);
        self.notification_task = Some(cx.spawn(async move |this, cx| {
            timer.await;
            if let Err(error) = this.update(cx, |this, cx| {
                if this.surface_state.dismiss_notification(identity) {
                    this.notification_task_identity = None;
                    this.schedule_notification_task(cx);
                    cx.notify();
                }
            }) {
                log::error!("execution panel disappeared while dismissing notification: {error}");
            }
        }));
    }

    fn schedule_progress_toast_task(&mut self, cx: &mut Context<Self>) {
        let identity = self.surface_state.progress_toast().and_then(|toast| {
            (toast.phase != ExecutionProgressToastPhase::Running).then_some(toast.identity)
        });
        if identity == self.progress_toast_task_identity {
            return;
        }
        self.progress_toast_task = None;
        self.progress_toast_task_identity = identity;
        let Some(identity) = identity else {
            return;
        };
        let timer = cx
            .background_executor()
            .timer(EXECUTION_TERMINAL_TOAST_DURATION);
        self.progress_toast_task = Some(cx.spawn(async move |this, cx| {
            timer.await;
            if let Err(error) = this.update(cx, |this, cx| {
                if this.surface_state.dismiss_progress_toast(identity) {
                    this.progress_toast_task_identity = None;
                    this.reconcile_execution_surfaces(cx);
                    cx.notify();
                }
            }) {
                log::error!("execution panel disappeared while dismissing progress toast: {error}");
            }
        }));
    }

    fn surface_handler(&self, cx: &Context<Self>) -> ExecutionSurfaceActionHandler {
        let this = cx.entity().downgrade();
        #[cfg(all(test, feature = "test-support"))]
        let invocation_trace = self.surface_action_invocation_trace.clone();
        #[cfg(all(test, feature = "test-support"))]
        let update_error = self.surface_action_update_error.clone();
        Arc::new(move |action, window, cx| {
            #[cfg(all(test, feature = "test-support"))]
            {
                let mut trace = match invocation_trace.lock() {
                    Ok(trace) => trace,
                    Err(error) => error.into_inner(),
                };
                if trace.len() == 64 {
                    trace.pop_front();
                }
                trace.push_back(action.clone());
            }
            if let Err(error) = this.update(cx, |this, cx| {
                #[cfg(all(test, feature = "test-support"))]
                {
                    if this.surface_action_trace.len() == 64 {
                        this.surface_action_trace.pop_front();
                    }
                    this.surface_action_trace.push_back(action.clone());
                }
                this.handle_surface_action(action, window, cx)
            }) {
                #[cfg(all(test, feature = "test-support"))]
                {
                    let mut last_error = match update_error.lock() {
                        Ok(error) => error,
                        Err(error) => error.into_inner(),
                    };
                    *last_error = Some(error.to_string());
                }
                log::error!("execution surface action targeted a closed panel: {error}");
            }
        })
    }

    fn handle_surface_action(
        &mut self,
        action: ExecutionSurfaceAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match action {
            ExecutionSurfaceAction::SelectJobTab(tab) => {
                self.state.selected_job_tab = tab;
                self.state.selected_tab = match tab {
                    ExecutionJobTab::Completed => ExecutionPanelTab::History,
                    ExecutionJobTab::All | ExecutionJobTab::Active => ExecutionPanelTab::Queue,
                };
                self.state.history_page = 0;
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::SetJobSearch(query) => {
                self.state.job_search_query = query
                    .chars()
                    .take(EXECUTION_PANEL_FILTER_CHARACTER_CAPACITY)
                    .collect();
                self.state.history_page = 0;
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::ToggleWorkflowFilter => {
                self.state.workflow_filter = match self.state.workflow_filter {
                    ExecutionWorkflowFilter::All => ExecutionWorkflowFilter::Selected,
                    ExecutionWorkflowFilter::Selected => ExecutionWorkflowFilter::All,
                };
                self.state.history_page = 0;
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::CycleSortMode => {
                self.state.sort_mode = match self.state.sort_mode {
                    ExecutionSortMode::MostRecent => ExecutionSortMode::Oldest,
                    ExecutionSortMode::Oldest => ExecutionSortMode::TotalGenerationTime,
                    ExecutionSortMode::TotalGenerationTime => ExecutionSortMode::MostRecent,
                };
                self.state.history_page = 0;
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::ToggleShowProgress => {
                self.state.show_progress = !self.state.show_progress;
                if let Ok(Some(graph)) = self.workspace.read_with(cx, |workspace, cx| {
                    workspace.active_item_as::<crate::GraphWorkspaceItem>(cx)
                }) {
                    graph.update(cx, |graph, cx| {
                        graph.show_execution_progress = self.state.show_progress;
                        cx.notify();
                    });
                }
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::ToggleJobDetails(attempt_id) => {
                let opening = self.job_details_attempt_id != Some(attempt_id);
                self.job_details_attempt_id = opening.then_some(attempt_id);
                if opening {
                    self.job_context_attempt_id = None;
                    self.job_details_hover_attempt_id = None;
                    self.job_details_trigger_hovered = None;
                    self.job_details_content_hovered = None;
                    self.job_details_hover_close_task = None;
                }
                cx.notify();
            }
            ExecutionSurfaceAction::SetJobDetailsTriggerHovered(attempt_id, hovered) => {
                self.set_job_details_hover(attempt_id, hovered, false, cx);
            }
            ExecutionSurfaceAction::SetJobDetailsContentHovered(attempt_id, hovered) => {
                self.set_job_details_hover(attempt_id, hovered, true, cx);
            }
            ExecutionSurfaceAction::ToggleJobContextMenu(attempt_id) => {
                let opening = self.job_context_attempt_id != Some(attempt_id);
                self.job_context_attempt_id = opening.then_some(attempt_id);
                if opening {
                    self.job_details_attempt_id = None;
                    self.job_details_hover_attempt_id = None;
                    self.job_details_trigger_hovered = None;
                    self.job_details_content_hovered = None;
                    self.job_details_hover_close_task = None;
                }
                cx.notify();
            }
            ExecutionSurfaceAction::CopyAttemptId(attempt_id) => {
                cx.write_to_clipboard(ClipboardItem::new_string(attempt_id.0.to_string()));
                self.job_context_attempt_id = None;
                self.status_message = Some("Copied native execution attempt ID".to_owned());
                cx.notify();
            }
            ExecutionSurfaceAction::InspectAttempt(attempt_id) => {
                self.state.selected_attempt_id = Some(attempt_id);
                self.state.selected_tab = ExecutionPanelTab::Output;
                self.state.output_auto_follow = false;
                self.job_context_attempt_id = None;
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::CancelAttempt(attempt_id) => {
                let result = self.dispatch(
                    ExecutionControlCommandKind::Cancel {
                        attempt_id,
                        reason: "cancelled from native execution job actions".to_owned(),
                    },
                    cx,
                );
                self.job_context_attempt_id = None;
                self.handle_command_result(result, cx);
            }
            ExecutionSurfaceAction::RetryAttempt(attempt_id) => {
                let result = self.with_profile(
                    |profile_id, model, cx| model.retry(profile_id, attempt_id, cx),
                    cx,
                );
                self.job_context_attempt_id = None;
                self.handle_retry_result(attempt_id, result, window, cx);
            }
            ExecutionSurfaceAction::RemoveAttempt(attempt_id) => {
                let result = self.dispatch(
                    ExecutionControlCommandKind::RemoveHistory { attempt_id },
                    cx,
                );
                self.job_context_attempt_id = None;
                self.handle_command_result(result, cx);
            }
            ExecutionSurfaceAction::ClearPending => {
                let result = self.dispatch(
                    ExecutionControlCommandKind::ClearPending {
                        reason: "cleared from native output-history queue item".to_owned(),
                    },
                    cx,
                );
                self.handle_command_result(result, cx);
            }
            ExecutionSurfaceAction::DismissNotification(identity) => {
                if self.surface_state.dismiss_notification(identity) {
                    self.notification_task_identity = None;
                    self.schedule_notification_task(cx);
                    cx.notify();
                }
            }
            ExecutionSurfaceAction::ViewErrors => {
                self.state.selected_tab = ExecutionPanelTab::Errors;
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::DismissErrorOverlay(attempt_id) => {
                self.state
                    .dismissed_error_overlay_attempts
                    .insert(attempt_id);
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::SetErrorSearch(query) => {
                self.state.error_search_query = query
                    .chars()
                    .take(EXECUTION_PANEL_FILTER_CHARACTER_CAPACITY)
                    .collect();
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::ToggleAllErrors => {
                let failure_ids = self.snapshot(cx).map_or_else(
                    |_| Vec::new(),
                    |snapshot| {
                        snapshot
                            .attempts
                            .iter()
                            .rev()
                            .filter(|attempt| attempt.failure.is_some())
                            .take(EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY)
                            .map(|attempt| attempt.attempt_id)
                            .collect::<Vec<_>>()
                    },
                );
                let all_collapsed = !failure_ids.is_empty()
                    && failure_ids
                        .iter()
                        .all(|attempt_id| self.state.collapsed_error_attempts.contains(attempt_id));
                if all_collapsed {
                    self.state.collapsed_error_attempts.clear();
                    self.state.errors_all_collapsed = false;
                } else {
                    self.state
                        .collapsed_error_attempts
                        .extend(failure_ids.iter().copied());
                    self.state.errors_all_collapsed = !failure_ids.is_empty()
                        && self.snapshot(cx).is_ok_and(|snapshot| {
                            snapshot
                                .attempts
                                .iter()
                                .filter(|attempt| attempt.failure.is_some())
                                .count()
                                == failure_ids.len()
                        });
                }
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::ToggleErrorDetails(attempt_id) => {
                if !self.state.collapsed_error_attempts.remove(&attempt_id) {
                    self.state.collapsed_error_attempts.insert(attempt_id);
                }
                self.state.errors_all_collapsed = self.snapshot(cx).is_ok_and(|snapshot| {
                    let failure_ids = snapshot
                        .attempts
                        .iter()
                        .filter(|attempt| attempt.failure.is_some())
                        .map(|attempt| attempt.attempt_id)
                        .collect::<Vec<_>>();
                    !failure_ids.is_empty()
                        && failure_ids.len() <= EXECUTION_PANEL_ATTEMPT_STATE_CAPACITY
                        && failure_ids.iter().all(|attempt_id| {
                            self.state.collapsed_error_attempts.contains(attempt_id)
                        })
                });
                self.serialize(cx);
                cx.notify();
            }
            ExecutionSurfaceAction::CopyError(attempt_id) => {
                self.copy_error(attempt_id, cx);
                self.job_context_attempt_id = None;
            }
            ExecutionSurfaceAction::LocateError(attempt_id) => {
                self.navigate_to_error(attempt_id, window, cx)
            }
            ExecutionSurfaceAction::OpenErrorHelp(_) => {
                self.open_external_navigation(
                    "https://docs.comfy.org/troubleshooting/overview",
                    "Opened execution troubleshooting help",
                    cx,
                );
            }
            ExecutionSurfaceAction::OpenErrorGitHub(_) => {
                self.open_external_navigation(
                    "https://github.com/Comfy-Org/ComfyUI_frontend/issues",
                    "Opened ComfyUI frontend GitHub issues",
                    cx,
                );
            }
        }
    }

    fn open_external_navigation(
        &mut self,
        url: &'static str,
        success_message: &'static str,
        cx: &mut Context<Self>,
    ) {
        match self.external_navigation_policy.authorize(url, true) {
            Ok(()) => {
                cx.open_url(url);
                self.status_message = Some(success_message.to_owned());
            }
            Err(error) => {
                self.status_message = Some(format!("Blocked unsafe execution navigation: {error}"));
            }
        }
        cx.notify();
    }

    fn cycle_queue_filter(&mut self, cx: &mut Context<Self>) {
        self.state.queue_filter = match self.state.queue_filter.as_str() {
            "queued" => "active",
            "active" => "all",
            _ => "queued",
        }
        .to_owned();
        self.serialize(cx);
        cx.notify();
    }

    fn cycle_history_filter(&mut self, cx: &mut Context<Self>) {
        self.state.history_filter = match self.state.history_filter.as_str() {
            "failed" => "succeeded",
            "succeeded" => "interrupted",
            "interrupted" => "all",
            _ => "failed",
        }
        .to_owned();
        self.state.history_page = 0;
        self.serialize(cx);
        cx.notify();
    }

    fn change_history_page(&mut self, next: bool, cx: &mut Context<Self>) {
        let page_count = self
            .snapshot(cx)
            .map(|snapshot| self.history_page_count(&snapshot))
            .unwrap_or(1);
        self.state.history_page = if next {
            self.state
                .history_page
                .saturating_add(1)
                .min(page_count.saturating_sub(1))
        } else {
            self.state.history_page.saturating_sub(1)
        };
        self.serialize(cx);
        cx.notify();
    }

    fn filtered_queue_snapshot(&self, mut snapshot: ExecutionSnapshot) -> ExecutionSnapshot {
        match self.state.queue_filter.as_str() {
            "queued" => snapshot.attempts.retain(|attempt| {
                snapshot
                    .queue
                    .iter()
                    .any(|queued| queued.attempt_id == attempt.attempt_id)
            }),
            "active" => {
                let queued_attempts = snapshot
                    .queue
                    .iter()
                    .map(|queued| queued.attempt_id)
                    .collect::<HashSet<_>>();
                snapshot.queue.clear();
                snapshot.attempts.retain(|attempt| {
                    !attempt.state.is_terminal() && !queued_attempts.contains(&attempt.attempt_id)
                });
            }
            _ => {}
        }
        let selected_prompt_id = self.state.selected_attempt_id.and_then(|attempt_id| {
            snapshot
                .attempts
                .iter()
                .find(|attempt| attempt.attempt_id == attempt_id)
                .map(|attempt| attempt.prompt_id)
        });
        let allowed_attempts = snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                self.state.selected_job_tab != ExecutionJobTab::Completed
                    && attempt_matches_query(attempt, &self.state.job_search_query)
                    && (self.state.workflow_filter == ExecutionWorkflowFilter::All
                        || selected_prompt_id == Some(attempt.prompt_id))
            })
            .map(|attempt| attempt.attempt_id)
            .collect::<HashSet<_>>();
        snapshot
            .queue
            .retain(|queued| allowed_attempts.contains(&queued.attempt_id));
        snapshot
            .attempts
            .retain(|attempt| allowed_attempts.contains(&attempt.attempt_id));
        snapshot
    }

    fn filtered_history_attempts(
        &self,
        snapshot: &ExecutionSnapshot,
    ) -> Vec<sim_runtime::AttemptPresentation> {
        let mut attempts = snapshot
            .attempts
            .iter()
            .filter(|attempt| attempt.state.is_terminal())
            .filter(|attempt| match self.state.history_filter.as_str() {
                "failed" => attempt.state == sim_runtime::AttemptState::Failed,
                "succeeded" => attempt.state == sim_runtime::AttemptState::Succeeded,
                "interrupted" => attempt.state == sim_runtime::AttemptState::Interrupted,
                _ => true,
            })
            .filter(|attempt| {
                self.state.selected_job_tab != ExecutionJobTab::Active
                    && attempt_matches_query(attempt, &self.state.job_search_query)
            })
            .cloned()
            .collect::<Vec<_>>();
        if self.state.workflow_filter == ExecutionWorkflowFilter::Selected {
            let selected_prompt_id = self.state.selected_attempt_id.and_then(|attempt_id| {
                snapshot
                    .attempts
                    .iter()
                    .find(|attempt| attempt.attempt_id == attempt_id)
                    .map(|attempt| attempt.prompt_id)
            });
            attempts.retain(|attempt| selected_prompt_id == Some(attempt.prompt_id));
        }
        attempts.sort_by(|left, right| match self.state.sort_mode {
            ExecutionSortMode::MostRecent => right
                .finished_at
                .cmp(&left.finished_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| right.attempt_id.0.cmp(&left.attempt_id.0)),
            ExecutionSortMode::Oldest => left
                .finished_at
                .cmp(&right.finished_at)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| left.attempt_id.0.cmp(&right.attempt_id.0)),
            ExecutionSortMode::TotalGenerationTime => {
                let left_duration = left
                    .finished_at
                    .map(|finished| finished.signed_duration_since(left.created_at));
                let right_duration = right
                    .finished_at
                    .map(|finished| finished.signed_duration_since(right.created_at));
                right_duration
                    .cmp(&left_duration)
                    .then_with(|| right.created_at.cmp(&left.created_at))
                    .then_with(|| right.attempt_id.0.cmp(&left.attempt_id.0))
            }
        });
        attempts
    }

    fn history_page_count(&self, snapshot: &ExecutionSnapshot) -> usize {
        let page_size = self.state.history_page_size.max(1);
        let attempt_count = self.filtered_history_attempts(snapshot).len();
        attempt_count.saturating_add(page_size.saturating_sub(1)) / page_size
    }

    fn filtered_history_snapshot(&self, mut snapshot: ExecutionSnapshot) -> ExecutionSnapshot {
        let page_size = self.state.history_page_size.max(1);
        let attempts = self.filtered_history_attempts(&snapshot);
        let page_count = attempts.len().saturating_add(page_size.saturating_sub(1)) / page_size;
        let page = self.state.history_page.min(page_count.saturating_sub(1));
        let start = page.saturating_mul(page_size).min(attempts.len());
        let end = start.saturating_add(page_size).min(attempts.len());
        snapshot.attempts = attempts[start..end].to_vec();
        snapshot
    }

    fn select_tab(&mut self, tab: ExecutionPanelTab, cx: &mut Context<Self>) {
        self.state.selected_tab = tab;
        self.serialize(cx);
        cx.notify();
    }

    fn serialize(&mut self, cx: &mut Context<Self>) {
        let serialization_key = self
            .workspace
            .read_with(cx, |workspace, _| Self::serialization_key(workspace))
            .ok()
            .flatten();
        let Some(serialization_key) = serialization_key else {
            return;
        };
        let serialized = match serde_json::to_string(&self.state) {
            Ok(serialized) => serialized,
            Err(error) => {
                self.persistence_error = Some(format!(
                    "failed to encode native execution panel state: {error}"
                ));
                cx.notify();
                return;
            }
        };
        let key_value_store = KeyValueStore::global(cx);
        let write = cx.background_spawn(async move {
            key_value_store
                .write_kvp(serialization_key, serialized)
                .await
        });
        self.pending_serialization = Some(cx.spawn(async move |this, cx| {
            if let Err(error) = write.await
                && let Err(update_error) = this.update(cx, |this, cx| {
                    this.persistence_error = Some(format!(
                        "failed to persist native execution panel state: {error}"
                    ));
                    cx.notify();
                })
            {
                log::error!(
                    "execution panel disappeared while reporting persistence error: {update_error}"
                );
            }
        }));
    }

    fn handle_queue_action(
        &mut self,
        action: QueuePanelAction,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let result = match action {
            QueuePanelAction::SelectAttempt(attempt_id) => {
                self.state.selected_attempt_id = Some(attempt_id);
                self.state.selected_tab = ExecutionPanelTab::Output;
                self.state.output_auto_follow = false;
                self.serialize(cx);
                cx.notify();
                return;
            }
            QueuePanelAction::MoveEarlier(attempt_id) => self.reorder(attempt_id, true, cx),
            QueuePanelAction::MoveLater(attempt_id) => self.reorder(attempt_id, false, cx),
            QueuePanelAction::Cancel(attempt_id) => self.dispatch(
                ExecutionControlCommandKind::Cancel {
                    attempt_id,
                    reason: "cancelled from native execution queue".to_owned(),
                },
                cx,
            ),
            QueuePanelAction::Interrupt(attempt_id) => self.dispatch(
                ExecutionControlCommandKind::Interrupt {
                    attempt_id,
                    reason: "interrupted from native execution queue".to_owned(),
                },
                cx,
            ),
            QueuePanelAction::ClearPending => self.dispatch(
                ExecutionControlCommandKind::ClearPending {
                    reason: "cleared from native execution queue".to_owned(),
                },
                cx,
            ),
        };
        self.handle_command_result(result, cx);
    }

    fn handle_history_action(
        &mut self,
        action: HistoryPanelAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match action {
            HistoryPanelAction::SelectAttempt(attempt_id) => {
                self.state.selected_attempt_id = Some(attempt_id);
                self.state.selected_tab = ExecutionPanelTab::Output;
                self.state.output_auto_follow = false;
                self.serialize(cx);
                cx.notify();
            }
            HistoryPanelAction::ToggleErrorDetails(attempt_id) => {
                if !self.state.expanded_error_attempts.remove(&attempt_id) {
                    self.state.expanded_error_attempts.insert(attempt_id);
                }
                self.serialize(cx);
                cx.notify();
            }
            HistoryPanelAction::CopyError(attempt_id) => self.copy_error(attempt_id, cx),
            HistoryPanelAction::NavigateToError(attempt_id) => {
                self.navigate_to_error(attempt_id, window, cx)
            }
            HistoryPanelAction::Retry(attempt_id) => {
                let result = self.with_profile(
                    |profile_id, model, cx| model.retry(profile_id, attempt_id, cx),
                    cx,
                );
                self.handle_retry_result(attempt_id, result, window, cx);
            }
            HistoryPanelAction::Remove(attempt_id) => {
                let result = self.dispatch(
                    ExecutionControlCommandKind::RemoveHistory { attempt_id },
                    cx,
                );
                self.handle_command_result(result, cx);
            }
            HistoryPanelAction::ClearHistory => self.confirm_clear_history(window, cx),
        }
    }

    fn handle_output_action(
        &mut self,
        action: OutputViewAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match action {
            OutputViewAction::SelectAttempt(attempt_id) => {
                self.state.selected_attempt_id = Some(attempt_id);
                self.state.selected_output_id = None;
                self.state.output_auto_follow = false;
                self.serialize(cx);
                cx.notify();
            }
            OutputViewAction::CancelAttempt(attempt_id) => {
                let result = self.dispatch(
                    ExecutionControlCommandKind::Cancel {
                        attempt_id,
                        reason: "cancelled from native output history".to_owned(),
                    },
                    cx,
                );
                self.handle_command_result(result, cx);
            }
            OutputViewAction::InterruptAttempt(attempt_id) => {
                let result = self.dispatch(
                    ExecutionControlCommandKind::Interrupt {
                        attempt_id,
                        reason: "interrupted from native output history".to_owned(),
                    },
                    cx,
                );
                self.handle_command_result(result, cx);
            }
            OutputViewAction::SelectOutput(output_id) => {
                self.state.selected_output_id = Some(output_id);
                self.state.output_auto_follow = false;
                self.serialize(cx);
                cx.notify();
            }
            OutputViewAction::ToggleErrorDetails(attempt_id) => {
                if !self.state.expanded_error_attempts.remove(&attempt_id) {
                    self.state.expanded_error_attempts.insert(attempt_id);
                }
                self.serialize(cx);
                cx.notify();
            }
            OutputViewAction::CopyError(attempt_id) => self.copy_error(attempt_id, cx),
            OutputViewAction::NavigateToError(attempt_id) => {
                self.navigate_to_error(attempt_id, window, cx)
            }
            OutputViewAction::CopyReference { reference, .. } => {
                cx.write_to_clipboard(ClipboardItem::new_string(reference));
                self.status_message = Some("Copied native output reference".to_owned());
                cx.notify();
            }
            OutputViewAction::ViewReference { reference, .. } => {
                let result = self.with_profile(
                    |profile_id, model, _cx| {
                        model.handle_output_reference(
                            profile_id,
                            crate::ExecutionOutputReferenceAction::View,
                            &reference,
                        )
                    },
                    cx,
                );
                self.handle_result(result, cx);
            }
            OutputViewAction::DownloadReference { reference, .. } => {
                let result = self.with_profile(
                    |profile_id, model, _cx| {
                        model.handle_output_reference(
                            profile_id,
                            crate::ExecutionOutputReferenceAction::Download,
                            &reference,
                        )
                    },
                    cx,
                );
                self.handle_result(result, cx);
            }
            OutputViewAction::RecoverOutput(output_id) => {
                let Some(attempt_id) = self.selected_attempt_id(cx) else {
                    self.handle_result(Err(crate::ExecutionUiModelError::NoSelectedAttempt), cx);
                    return;
                };
                let result = self.with_profile(
                    |profile_id, model, cx| {
                        model.handle_output_operation(
                            profile_id,
                            attempt_id,
                            output_id,
                            crate::ExecutionOutputOperationAction::Recover,
                            cx,
                        )
                    },
                    cx,
                );
                self.handle_result(result, cx);
            }
            OutputViewAction::RemoveOutput(output_id) => {
                let Some(attempt_id) = self.selected_attempt_id(cx) else {
                    self.handle_result(Err(crate::ExecutionUiModelError::NoSelectedAttempt), cx);
                    return;
                };
                self.confirm_remove_output(attempt_id, output_id, window, cx);
            }
        }
    }

    fn dispatch(
        &mut self,
        kind: ExecutionControlCommandKind,
        cx: &mut Context<Self>,
    ) -> Result<ExecutionCommandAck, crate::ExecutionUiModelError> {
        self.with_profile(
            |profile_id, model, cx| model.dispatch(profile_id, kind, cx),
            cx,
        )
    }

    fn with_profile<T>(
        &self,
        operation: impl FnOnce(
            ProfileId,
            &mut crate::ExecutionUiModel,
            &mut Context<crate::ExecutionUiModel>,
        ) -> Result<T, crate::ExecutionUiModelError>,
        cx: &mut Context<Self>,
    ) -> Result<T, crate::ExecutionUiModelError> {
        let profile_id = self
            .model
            .read(cx)
            .active_profile_id()
            .ok_or(crate::ExecutionUiModelError::NoActiveProfile)?;
        self.model
            .update(cx, |model, cx| operation(profile_id, model, cx))
    }

    fn reorder(
        &self,
        attempt_id: AttemptId,
        earlier: bool,
        cx: &mut Context<Self>,
    ) -> Result<ExecutionCommandAck, crate::ExecutionUiModelError> {
        let snapshot = self.snapshot(cx)?;
        let position = snapshot
            .queue
            .iter()
            .position(|item| item.attempt_id == attempt_id)
            .ok_or_else(|| crate::ExecutionUiModelError::PlanCompilation {
                code: "attempt_not_queued".to_owned(),
                message: "The selected attempt is no longer in the queue.".to_owned(),
            })?;
        let new_position = if earlier {
            position.saturating_sub(1)
        } else {
            position
                .checked_add(1)
                .map(|position| position.min(snapshot.queue.len().saturating_sub(1)))
                .unwrap_or(position)
        };
        self.with_profile(
            |profile_id, model, cx| {
                model.dispatch(
                    profile_id,
                    ExecutionControlCommandKind::Reorder {
                        attempt_id,
                        position: new_position,
                    },
                    cx,
                )
            },
            cx,
        )
    }

    fn copy_error(&mut self, attempt_id: AttemptId, cx: &mut Context<Self>) {
        match self.snapshot(cx).and_then(|snapshot| {
            snapshot
                .attempts
                .into_iter()
                .find(|attempt| attempt.attempt_id == attempt_id)
                .and_then(|attempt| attempt.failure)
                .ok_or_else(|| crate::ExecutionUiModelError::PlanCompilation {
                    code: "error_not_available".to_owned(),
                    message: "The selected attempt has no structured error.".to_owned(),
                })
        }) {
            Ok(failure) => {
                cx.write_to_clipboard(ClipboardItem::new_string(format_failure(&failure)));
                self.status_message = Some("Copied structured execution error".to_owned());
                cx.notify();
            }
            Err(error) => self.handle_result(Err(error), cx),
        }
    }

    fn navigate_to_error(
        &mut self,
        attempt_id: AttemptId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let node_id = self.snapshot(cx).ok().and_then(|snapshot| {
            snapshot
                .attempts
                .into_iter()
                .find(|attempt| attempt.attempt_id == attempt_id)
                .and_then(|attempt| attempt.failure)
                .and_then(|failure| failure.node_id)
        });
        match node_id {
            Some(node_id) => {
                let graph_item = self
                    .workspace
                    .read_with(cx, |workspace, cx| {
                        workspace.active_item_as::<crate::GraphWorkspaceItem>(cx)
                    })
                    .ok()
                    .flatten();
                match graph_item {
                    Some(graph_item)
                        if graph_item.read(cx).is_associated_with_execution(attempt_id) =>
                    {
                        let result = graph_item.update(cx, |graph_item, cx| {
                            graph_item.locate_execution_node(&node_id, cx)
                        });
                        match result {
                            Ok(()) => {
                                graph_item.focus_handle(cx).focus(window, cx);
                                self.status_message = Some(format!(
                                    "Focused execution error at node {}; dispatch RestoreExecutionNavigation to return",
                                    node_id.0
                                ));
                            }
                            Err(error) => {
                                self.status_message =
                                    Some(format!("Could not locate execution node: {error}"));
                            }
                        }
                    }
                    _ => {
                        self.status_message = Some(
                            "Open the associated native graph before locating this execution error"
                                .to_owned(),
                        );
                    }
                }
            }
            None => {
                self.status_message =
                    Some("The structured execution error has no node location".to_owned());
            }
        }
        cx.notify();
    }

    fn confirm_clear_history(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(originating_profile_id) = self.model.read(cx).active_profile_id() else {
            self.status_message = Some(
                "Clear-history confirmation unavailable because no execution profile is active"
                    .to_owned(),
            );
            cx.notify();
            return;
        };
        self.confirmation_profile_id = Some(originating_profile_id);
        let prompt = window.prompt(
            PromptLevel::Warning,
            "Clear native execution history?",
            Some("This removes all terminal attempts for the selected execution profile."),
            &["Clear History", "Cancel", "×"],
            cx,
        );
        self.confirmation_task = Some(cx.spawn_in(window, async move |this, cx| {
            match prompt.await {
                Ok(0) => {
                    if let Err(error) = this.update_in(cx, |this, _window, cx| {
                        if !this.take_profile_confirmation(originating_profile_id, cx) {
                            return;
                        }
                        let result = this.model.update(cx, |model, cx| {
                            model.dispatch(
                                originating_profile_id,
                                ExecutionControlCommandKind::ClearHistory,
                                cx,
                            )
                        });
                        this.handle_command_result(result, cx);
                    }) {
                        log::error!("execution panel disappeared while clearing history: {error}");
                    }
                }
                Ok(_) => {
                    if let Err(error) = this.update_in(cx, |this, _window, _cx| {
                        this.clear_profile_confirmation(originating_profile_id);
                    }) {
                        log::error!(
                            "execution panel disappeared while cancelling clear-history confirmation: {error}"
                        );
                    }
                }
                Err(error) => {
                    if let Err(update_error) = this.update_in(cx, |this, _window, cx| {
                        this.clear_profile_confirmation(originating_profile_id);
                        this.status_message = Some(format!(
                            "Clear-history confirmation failed: {error}"
                        ));
                        cx.notify();
                    }) {
                        log::error!("execution panel disappeared while reporting prompt error: {update_error}");
                    }
                }
            }
        }));
    }

    fn confirm_remove_output(
        &mut self,
        attempt_id: AttemptId,
        output_id: Uuid,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(originating_profile_id) = self.model.read(cx).active_profile_id() else {
            self.status_message = Some(
                "Output-removal confirmation unavailable because no execution profile is active"
                    .to_owned(),
            );
            cx.notify();
            return;
        };
        self.confirmation_profile_id = Some(originating_profile_id);
        let prompt = window.prompt(
            PromptLevel::Warning,
            "Remove this native execution output?",
            Some(
                "This removes an existing native asset and retains an inspectable removed record in execution history.",
            ),
            &["Remove Output", "Cancel", "×"],
            cx,
        );
        self.confirmation_task = Some(cx.spawn_in(window, async move |this, cx| {
            match prompt.await {
                Ok(0) => {
                    if let Err(error) = this.update_in(cx, |this, _window, cx| {
                        if !this.take_profile_confirmation(originating_profile_id, cx) {
                            return;
                        }
                        let result = this.model.update(cx, |model, cx| {
                            model.handle_output_operation(
                                originating_profile_id,
                                attempt_id,
                                output_id,
                                crate::ExecutionOutputOperationAction::Remove,
                                cx,
                            )
                        });
                        this.handle_result(result, cx);
                    }) {
                        log::error!("execution panel disappeared while removing output: {error}");
                    }
                }
                Ok(_) => {
                    if let Err(error) = this.update_in(cx, |this, _window, _cx| {
                        this.clear_profile_confirmation(originating_profile_id);
                    }) {
                        log::error!(
                            "execution panel disappeared while cancelling output-removal confirmation: {error}"
                        );
                    }
                }
                Err(error) => {
                    if let Err(update_error) = this.update_in(cx, |this, _window, cx| {
                        this.clear_profile_confirmation(originating_profile_id);
                        this.status_message =
                            Some(format!("Output-removal confirmation failed: {error}"));
                        cx.notify();
                    }) {
                        log::error!(
                            "execution panel disappeared while reporting prompt error: {update_error}"
                        );
                    }
                }
            }
        }));
    }

    fn take_profile_confirmation(
        &mut self,
        originating_profile_id: ProfileId,
        cx: &mut Context<Self>,
    ) -> bool {
        let confirmation_profile_id = self.confirmation_profile_id.take();
        let active_profile_id = self.model.read(cx).active_profile_id();
        if confirmation_profile_id == Some(originating_profile_id)
            && active_profile_id == Some(originating_profile_id)
        {
            return true;
        }
        self.status_message = Some(
            "Destructive confirmation cancelled because the active execution profile changed"
                .to_owned(),
        );
        cx.notify();
        false
    }

    fn clear_profile_confirmation(&mut self, originating_profile_id: ProfileId) {
        if self.confirmation_profile_id == Some(originating_profile_id) {
            self.confirmation_profile_id = None;
        }
    }

    fn handle_result(
        &mut self,
        result: Result<(), crate::ExecutionUiModelError>,
        cx: &mut Context<Self>,
    ) {
        self.status_message = Some(match result {
            Ok(()) => "Native execution command acknowledged".to_owned(),
            Err(error) => format!("Native execution command failed: {error}"),
        });
        cx.notify();
    }

    fn handle_command_result(
        &mut self,
        result: Result<ExecutionCommandAck, crate::ExecutionUiModelError>,
        cx: &mut Context<Self>,
    ) {
        self.status_message = Some(match result {
            Ok(ExecutionCommandAck {
                outcome: ExecutionCommandOutcome::Accepted { .. },
                ..
            }) => "Native execution command acknowledged".to_owned(),
            Ok(ExecutionCommandAck {
                outcome: ExecutionCommandOutcome::Rejected { failure },
                ..
            }) => format!(
                "Native execution command rejected ({}): {}",
                failure.code, failure.message
            ),
            Err(error) => format!("Native execution command failed: {error}"),
        });
        cx.notify();
    }

    fn handle_retry_result(
        &mut self,
        original_attempt_id: AttemptId,
        result: Result<ExecutionCommandAck, crate::ExecutionUiModelError>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Ok(ExecutionCommandAck {
            outcome:
                ExecutionCommandOutcome::Accepted {
                    assigned_attempt_id: Some(attempt_id),
                },
            ..
        }) = &result
            && let Ok(Some(graph_item)) = self.workspace.read_with(cx, |workspace, cx| {
                workspace.active_item_as::<crate::GraphWorkspaceItem>(cx)
            })
            && graph_item
                .read(cx)
                .is_associated_with_execution(original_attempt_id)
        {
            graph_item.update(cx, |graph_item, cx| {
                graph_item.associate_execution_attempt(*attempt_id, cx)
            });
            graph_item.focus_handle(cx).focus(window, cx);
        }
        self.handle_command_result(result, cx);
    }

    fn queue_handler(&self, cx: &Context<Self>) -> QueuePanelActionHandler {
        let this = cx.entity().downgrade();
        Arc::new(move |action, window, cx| {
            if let Err(error) =
                this.update(cx, |this, cx| this.handle_queue_action(action, window, cx))
            {
                log::error!("execution queue action targeted a closed panel: {error}");
            }
        })
    }

    fn history_handler(&self, cx: &Context<Self>) -> HistoryPanelActionHandler {
        let this = cx.entity().downgrade();
        Arc::new(move |action, window, cx| {
            if let Err(error) = this.update(cx, |this, cx| {
                this.handle_history_action(action, window, cx)
            }) {
                log::error!("execution history action targeted a closed panel: {error}");
            }
        })
    }

    fn output_handler(&self, cx: &Context<Self>) -> OutputViewActionHandler {
        let this = cx.entity().downgrade();
        Arc::new(move |action, window, cx| {
            if let Err(error) =
                this.update(cx, |this, cx| this.handle_output_action(action, window, cx))
            {
                log::error!("execution output action targeted a closed panel: {error}");
            }
        })
    }
}

impl EventEmitter<PanelEvent> for ExecutionPanel {}

impl Focusable for ExecutionPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Panel for ExecutionPanel {
    fn persistent_name() -> &'static str {
        "Native Execution"
    }

    fn panel_key() -> &'static str {
        EXECUTION_PANEL_KEY
    }

    fn position(&self, _window: &Window, _cx: &App) -> DockPosition {
        DockPosition::Bottom
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        position == DockPosition::Bottom
    }

    fn set_position(
        &mut self,
        position: DockPosition,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.position_is_valid(position) {
            self.status_message = Some("The native execution panel is bottom-docked".to_owned());
            cx.notify();
        }
    }

    fn default_size(&self, _window: &Window, _cx: &App) -> Pixels {
        px(320.)
    }

    fn icon(&self, _window: &Window, _cx: &App) -> Option<IconName> {
        Some(IconName::QueueMessage)
    }

    fn icon_tooltip(&self, _window: &Window, _cx: &App) -> Option<&'static str> {
        Some("Native Execution")
    }

    fn toggle_action(&self) -> Box<dyn gpui::Action> {
        Box::new(ToggleExecutionPanel)
    }

    fn starts_open(&self, _window: &Window, _cx: &App) -> bool {
        false
    }

    fn set_active(&mut self, active: bool, _window: &mut Window, cx: &mut Context<Self>) {
        // Workspace decides focus after activation; taking it here closes the first toggle.
        self.active = active;
        cx.notify();
    }

    fn activation_priority(&self) -> u32 {
        8
    }
}

impl Render for ExecutionPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.reconcile_execution_surfaces(cx);
        let snapshot = self.snapshot(cx);
        let (reference_actions_available, output_operations_available, runtime_controls_available) = {
            let model = self.model.read(cx);
            (
                model.output_reference_actions_available(),
                model.output_operations_available(),
                model.runtime_controller_available(),
            )
        };
        let selected_tab = self.state.selected_tab;
        let selected_attempt_id = match selected_tab {
            ExecutionPanelTab::Errors => snapshot.as_ref().ok().and_then(|snapshot| {
                snapshot
                    .attempts
                    .iter()
                    .rev()
                    .find(|attempt| attempt.failure.is_some())
                    .map(|attempt| attempt.attempt_id)
            }),
            _ => self.state.selected_attempt_id,
        };
        let selected_attempt = snapshot.as_ref().ok().and_then(|snapshot| {
            self.state.selected_attempt_id.and_then(|attempt_id| {
                snapshot
                    .attempts
                    .iter()
                    .find(|attempt| attempt.attempt_id == attempt_id)
                    .cloned()
            })
        });
        let job_counts = snapshot.as_ref().map_or((0, 0, 0), |snapshot| {
            (
                snapshot
                    .attempts
                    .iter()
                    .filter(|attempt| !attempt.state.is_terminal())
                    .count(),
                snapshot
                    .attempts
                    .iter()
                    .filter(|attempt| attempt.state.is_terminal())
                    .count(),
                snapshot.queue.len(),
            )
        });
        let notification = self.surface_state.current_notification().cloned();
        let progress_toast = self.surface_state.progress_toast().cloned();
        let error_overlay = snapshot
            .as_ref()
            .ok()
            .and_then(|snapshot| self.error_overlay_for_snapshot(snapshot));
        let error_overlay_keyboard_handler = self.surface_handler(cx);
        let error_overlay_view_focus_handle = self.error_overlay_view_focus_handle.clone();
        let error_overlay_dismiss_focus_handle = self.error_overlay_dismiss_focus_handle.clone();
        let panel_content = match snapshot.as_ref() {
            Ok(snapshot) => match selected_tab {
                ExecutionPanelTab::Queue => QueuePanelContent::new(
                    self.filtered_queue_snapshot(snapshot.clone()),
                    self.state.selected_attempt_id,
                    self.queue_handler(cx),
                )
                .with_show_progress(self.state.show_progress)
                .with_runtime_controls_available(runtime_controls_available)
                .into_any_element(),
                ExecutionPanelTab::History => HistoryPanelContent::new(
                    self.filtered_history_snapshot(snapshot.clone()),
                    self.state.selected_attempt_id,
                    self.history_handler(cx),
                )
                .with_expanded_error_attempts(self.state.expanded_error_attempts.iter().copied())
                .with_runtime_controls_available(runtime_controls_available)
                .into_any_element(),
                ExecutionPanelTab::Output => OutputView::new(
                    snapshot.clone(),
                    selected_attempt_id,
                    self.state.selected_output_id,
                    self.output_handler(cx),
                )
                .with_capabilities(reference_actions_available, output_operations_available)
                .with_runtime_controls_available(runtime_controls_available)
                .with_show_progress(self.state.show_progress)
                .with_selected_view_focus_handle(self.selected_output_view_focus_handle.clone())
                .with_expanded_error_attempts(self.state.expanded_error_attempts.iter().copied())
                .into_any_element(),
                ExecutionPanelTab::Errors => ErrorExplorerSurface {
                    snapshot: snapshot.clone(),
                    search_query: self.state.error_search_query.clone(),
                    all_collapsed: self.state.errors_all_collapsed,
                    collapsed_attempts: self.state.collapsed_error_attempts.clone(),
                    runtime_controls_available,
                    on_action: self.surface_handler(cx),
                }
                .into_any_element(),
            },
            Err(error) => div()
                .id("sim-execution-unavailable")
                .role(Role::Alert)
                .aria_label(format!("Native execution view unavailable: {error}"))
                .flex_1()
                .p_2()
                .child(format!("Native execution view unavailable: {error}"))
                .into_any_element(),
        };
        v_flex()
            .id("sim-execution-panel")
            .key_context("SimExecutionPanel ComfyExecutionPanel")
            .track_focus(&self.focus_handle)
            .on_key_down(move |event: &KeyDownEvent, window, cx| {
                if !matches!(event.keystroke.key.as_str(), "enter" | "space") {
                    return;
                }
                let action = if error_overlay_view_focus_handle.is_focused(window) {
                    Some(ExecutionSurfaceAction::ViewErrors)
                } else if error_overlay_dismiss_focus_handle.is_focused(window) {
                    error_overlay.map(|(attempt_id, _)| {
                        ExecutionSurfaceAction::DismissErrorOverlay(attempt_id)
                    })
                } else {
                    None
                };
                if let Some(action) = action {
                    cx.stop_propagation();
                    error_overlay_keyboard_handler(action, window, cx);
                }
            })
            .relative()
            .size_full()
            .min_h_0()
            .bg(cx.theme().colors().panel_background)
            .child(
                h_flex()
                    .id("sim-execution-tabs")
                    .role(Role::TabList)
                    .aria_label("Native execution views")
                    .w_full()
                    .px_2()
                    .py_1()
                    .gap_1()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .children(
                        ExecutionPanelTab::ALL
                            .into_iter()
                            .enumerate()
                            .map(|(index, tab)| {
                                div()
                                    .id(tab.element_id())
                                    .debug_selector(move || {
                                        match tab {
                                            ExecutionPanelTab::Queue => "SIM-EXECUTION-TAB-QUEUE",
                                            ExecutionPanelTab::History => {
                                                "SIM-EXECUTION-TAB-HISTORY"
                                            }
                                            ExecutionPanelTab::Output => "SIM-EXECUTION-TAB-OUTPUT",
                                            ExecutionPanelTab::Errors => "SIM-EXECUTION-TAB-ERRORS",
                                        }
                                        .into()
                                    })
                                    .role(Role::Tab)
                                    .aria_selected(tab == selected_tab)
                                    .aria_label(format!("{} execution view", tab.label()))
                                    .tab_index(index as isize)
                                    .px_2()
                                    .py_1()
                                    .rounded_sm()
                                    .cursor_pointer()
                                    .when(tab == selected_tab, |this| {
                                        this.bg(cx.theme().colors().element_selected)
                                    })
                                    .on_click(
                                        cx.listener(move |this, _, _, cx| this.select_tab(tab, cx)),
                                    )
                                    .on_key_down(cx.listener(
                                        move |this, event: &KeyDownEvent, _, cx| {
                                            if matches!(
                                                event.keystroke.key.as_str(),
                                                "enter" | "space"
                                            ) {
                                                cx.stop_propagation();
                                                this.select_tab(tab, cx);
                                            }
                                        },
                                    ))
                                    .child(tab.label())
                            }),
                    )
                    .when(selected_tab == ExecutionPanelTab::Queue, |this| {
                        let filter = match self.state.queue_filter.as_str() {
                            "queued" => "Queued",
                            "active" => "Active",
                            _ => "All",
                        };
                        this.child(
                            Button::new("sim-queue-filter", format!("Filter: {filter}"))
                                .size(ButtonSize::Compact)
                                .aria_label(format!(
                                    "Queue filter is {filter}; activate to choose the next filter"
                                ))
                                .on_click(
                                    cx.listener(|this, _, _, cx| this.cycle_queue_filter(cx)),
                                ),
                        )
                    })
                    .when(selected_tab == ExecutionPanelTab::History, |this| {
                        let filter = match self.state.history_filter.as_str() {
                            "failed" => "Failed",
                            "succeeded" => "Succeeded",
                            "interrupted" => "Interrupted",
                            _ => "All",
                        };
                        let (page, page_count) = snapshot.as_ref().map_or((1, 1), |snapshot| {
                            let page_count = self.history_page_count(snapshot).max(1);
                            (
                                self.state
                                    .history_page
                                    .min(page_count.saturating_sub(1))
                                    .saturating_add(1),
                                page_count,
                            )
                        });
                        this.child(
                            Button::new("sim-history-filter", format!("Filter: {filter}"))
                                .size(ButtonSize::Compact)
                                .aria_label(format!(
                                    "History filter is {filter}; activate to choose the next filter"
                                ))
                                .on_click(
                                    cx.listener(|this, _, _, cx| this.cycle_history_filter(cx)),
                                ),
                        )
                        .child(
                            Button::new("sim-history-page-previous", "Previous")
                                .size(ButtonSize::Compact)
                                .disabled(page <= 1)
                                .aria_label("Show the previous execution history page")
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.change_history_page(false, cx)
                                })),
                        )
                        .child(
                            div()
                                .id("sim-history-page-status")
                                .role(Role::Status)
                                .aria_label(format!(
                                    "Execution history page {page} of {page_count}"
                                ))
                                .text_ui_sm(cx)
                                .child(format!("{page}/{page_count}")),
                        )
                        .child(
                            Button::new("sim-history-page-next", "Next")
                                .size(ButtonSize::Compact)
                                .disabled(page >= page_count)
                                .aria_label("Show the next execution history page")
                                .on_click(
                                    cx.listener(|this, _, _, cx| {
                                        this.change_history_page(true, cx)
                                    }),
                                ),
                        )
                    }),
            )
            .when(
                matches!(
                    selected_tab,
                    ExecutionPanelTab::Queue | ExecutionPanelTab::History
                ),
                |this| {
                    this.child(JobFiltersSurface {
                        selected_tab: self.state.selected_job_tab,
                        search_query: self.state.job_search_query.clone(),
                        workflow_filter: self.state.workflow_filter,
                        sort_mode: self.state.sort_mode,
                        selected_workflow_available: selected_attempt.is_some(),
                        show_progress: self.state.show_progress,
                        active_count: job_counts.0,
                        completed_count: job_counts.1,
                        queued_count: job_counts.2,
                        runtime_controls_available: runtime_controls_available
                            && snapshot
                                .as_ref()
                                .is_ok_and(|snapshot| snapshot_allows_controls(&snapshot.status)),
                        clear_queue_focus_handle: self.clear_queue_focus_handle.clone(),
                        on_action: self.surface_handler(cx),
                    })
                },
            )
            .when_some(selected_attempt, |this, attempt| {
                this.child(JobDetailsSurface {
                    details_open: self.job_details_attempt_id == Some(attempt.attempt_id)
                        || self.job_details_hover_attempt_id == Some(attempt.attempt_id),
                    context_menu_open: self.job_context_attempt_id == Some(attempt.attempt_id),
                    runtime_controls_available,
                    snapshot_status: snapshot.as_ref().map_or_else(
                        |error| ExecutionSnapshotStatus::Unavailable {
                            failure: ExecutionFailure::new(
                                "execution_snapshot_unavailable",
                                error.to_string(),
                            ),
                        },
                        |snapshot| snapshot.status.clone(),
                    ),
                    attempt,
                    on_action: self.surface_handler(cx),
                })
            })
            .child(
                v_flex()
                    .debug_selector(|| "SIM-EXECUTION-MAIN-REGION".into())
                    .flex_1()
                    .min_h_0()
                    .child(
                        v_flex()
                            .debug_selector(|| "SIM-EXECUTION-MAIN-CONTENT".into())
                            .flex_1()
                            .min_h_0()
                            .overflow_hidden()
                            .child(panel_content),
                    ),
            )
            .when_some(notification, |this, notification| {
                this.child(QueueNotificationSurface {
                    notification,
                    on_action: self.surface_handler(cx),
                })
            })
            .when_some(
                self.state.show_progress.then_some(progress_toast).flatten(),
                |this, toast| this.child(ProgressToastSurface { toast }),
            )
            .when_some(self.status_message.clone(), |this, message| {
                this.child(
                    div()
                        .id("sim-execution-status")
                        .debug_selector(|| "SIM-EXECUTION-STATUS".into())
                        .role(Role::Status)
                        .aria_label(message.clone())
                        .px_2()
                        .py_1()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text_muted)
                        .child(message),
                )
            })
            .when_some(self.persistence_error.clone(), |this, message| {
                this.child(
                    div()
                        .id("sim-execution-persistence-error")
                        .debug_selector(|| "SIM-EXECUTION-PERSISTENCE-ERROR".into())
                        .role(Role::Alert)
                        .aria_label(message.clone())
                        .px_2()
                        .py_1()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text)
                        .child(message),
                )
            })
            .when_some(
                error_overlay.filter(|_| selected_tab != ExecutionPanelTab::Errors),
                |this, (attempt_id, failure_count)| {
                    this.child(ErrorOverlaySurface {
                        attempt_id,
                        failure_count,
                        view_focus_handle: self.error_overlay_view_focus_handle.clone(),
                        dismiss_focus_handle: self.error_overlay_dismiss_focus_handle.clone(),
                        on_action: self.surface_handler(cx),
                    })
                },
            )
    }
}

fn prepare_docked_execution_history(
    graph: Option<Entity<crate::GraphWorkspaceItem>>,
    panel: Entity<ExecutionPanel>,
    window: &mut Window,
    cx: &mut App,
) {
    if let Some(graph) = graph {
        graph.update(cx, |graph, cx| graph.close_queue_overlay(cx));
    }
    panel.update(cx, |panel, cx| {
        panel.state.docked_history = true;
        panel.state.selected_tab = ExecutionPanelTab::History;
        panel.active = true;
        panel.focus_handle.focus(window, cx);
        panel.serialize(cx);
        cx.notify();
    });
}

pub(crate) fn open_docked_execution_history(
    workspace: &mut Workspace,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let graph = workspace.active_item_as::<crate::GraphWorkspaceItem>(cx);
    if let Some(panel) = execution_panel_for_action(workspace, "open execution history", cx) {
        prepare_docked_execution_history(graph, panel, window, cx);
        workspace.focus_panel::<ExecutionPanel>(window, cx);
    }
}

pub(crate) fn execution_panel_for_action(
    workspace: &mut Workspace,
    action: &'static str,
    cx: &mut Context<Workspace>,
) -> Option<Entity<ExecutionPanel>> {
    let panel = workspace.panel::<ExecutionPanel>(cx);
    if panel.is_none() {
        workspace.show_toast(
            Toast::new(
                NotificationId::named(EXECUTION_ACTION_UNAVAILABLE_NOTIFICATION_ID.into()),
                format!("Cannot {action}: the native execution panel is not available"),
            )
            .autohide(),
            cx,
        );
    }
    panel
}

#[cfg(all(test, feature = "test-support"))]
pub(crate) fn open_docked_execution_history_for_test(
    graph: Entity<crate::GraphWorkspaceItem>,
    panel: Entity<ExecutionPanel>,
    window: &mut Window,
    cx: &mut App,
) {
    prepare_docked_execution_history(Some(graph), panel, window, cx);
}

pub fn init_execution_panel(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|workspace, _: &ToggleExecutionPanel, window, cx| {
            if execution_panel_for_action(workspace, "toggle the execution panel", cx).is_some() {
                workspace.toggle_panel_focus::<ExecutionPanel>(window, cx);
            }
        });
        workspace.register_action(|workspace, _: &ClearExecutionHistory, window, cx| {
            if let Some(panel) =
                execution_panel_for_action(workspace, "clear execution history", cx)
            {
                panel.update(cx, |panel, cx| panel.confirm_clear_history(window, cx));
            }
        });
        workspace.register_action(
            |workspace, _: &crate::CancelSelectedExecution, _window, cx| {
                if let Some(panel) =
                    execution_panel_for_action(workspace, "cancel the selected execution", cx)
                {
                    panel.update(cx, |panel, cx| {
                        let Some(attempt_id) = panel.selected_attempt_id(cx) else {
                            panel.status_message =
                                Some("No native execution attempt is selected".to_owned());
                            cx.notify();
                            return;
                        };
                        let result = panel.dispatch(
                            ExecutionControlCommandKind::Cancel {
                                attempt_id,
                                reason: "cancelled from the native selected-job menu".to_owned(),
                            },
                            cx,
                        );
                        panel.handle_command_result(result, cx);
                    });
                }
            },
        );
        workspace.register_action(|workspace, _: &crate::RetrySelectedExecution, window, cx| {
            if let Some(panel) =
                execution_panel_for_action(workspace, "retry the selected execution", cx)
            {
                panel.update(cx, |panel, cx| {
                    let Some(attempt_id) = panel.selected_attempt_id(cx) else {
                        panel.status_message =
                            Some("No native execution attempt is selected".to_owned());
                        cx.notify();
                        return;
                    };
                    let result = panel.with_profile(
                        |profile_id, model, cx| model.retry(profile_id, attempt_id, cx),
                        cx,
                    );
                    panel.handle_retry_result(attempt_id, result, window, cx);
                });
            }
        });
        workspace.register_action(
            |workspace, _: &crate::RemoveSelectedExecution, _window, cx| {
                if let Some(panel) =
                    execution_panel_for_action(workspace, "remove the selected execution", cx)
                {
                    panel.update(cx, |panel, cx| {
                        let Some(attempt_id) = panel.selected_attempt_id(cx) else {
                            panel.status_message =
                                Some("No native execution attempt is selected".to_owned());
                            cx.notify();
                            return;
                        };
                        let result = panel.dispatch(
                            ExecutionControlCommandKind::RemoveHistory { attempt_id },
                            cx,
                        );
                        panel.handle_command_result(result, cx);
                    });
                }
            },
        );
        workspace.register_action(
            |workspace, _: &crate::CopySelectedExecutionId, _window, cx| {
                if let Some(panel) =
                    execution_panel_for_action(workspace, "copy the selected execution ID", cx)
                {
                    panel.update(cx, |panel, cx| match panel.selected_attempt_id(cx) {
                        Some(attempt_id) => {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                attempt_id.0.to_string(),
                            ));
                            panel.status_message =
                                Some("Copied native execution attempt ID".to_owned());
                            cx.notify();
                        }
                        None => {
                            panel.status_message =
                                Some("No native execution attempt is selected".to_owned());
                            cx.notify();
                        }
                    });
                }
            },
        );
        workspace.register_action(
            |workspace, _: &crate::CopySelectedExecutionError, _window, cx| {
                if let Some(panel) =
                    execution_panel_for_action(workspace, "copy the selected execution error", cx)
                {
                    panel.update(cx, |panel, cx| match panel.selected_attempt_id(cx) {
                        Some(attempt_id) => panel.copy_error(attempt_id, cx),
                        None => {
                            panel.status_message =
                                Some("No native execution attempt is selected".to_owned());
                            cx.notify();
                        }
                    });
                }
            },
        );
        workspace.register_action(
            |workspace, _: &crate::ToggleDockedExecutionHistory, window, cx| {
                open_docked_execution_history(workspace, window, cx);
            },
        );
        workspace.register_action(
            |workspace, _: &crate::ToggleExecutionProgress, _window, cx| {
                let panel = workspace.panel::<ExecutionPanel>(cx);
                let graph = workspace.active_item_as::<crate::GraphWorkspaceItem>(cx);
                if panel.is_none() && graph.is_none() {
                    execution_panel_for_action(workspace, "toggle execution progress", cx);
                    return;
                }
                if let Some(panel) = panel {
                    panel.update(cx, |panel, cx| {
                        panel.state.show_progress = !panel.state.show_progress;
                        panel.serialize(cx);
                        cx.notify();
                    });
                }
                if let Some(graph) = graph {
                    graph.update(cx, |graph, cx| {
                        graph.show_execution_progress = !graph.show_execution_progress;
                        cx.notify();
                    });
                }
            },
        );
    })
    .detach();
}

fn format_failure(failure: &ExecutionFailure) -> String {
    let mut text = format!(
        "origin: {:?}\n{}: {}\nretryable: {}",
        failure.origin, failure.code, failure.message, failure.retryable
    );
    if let Some(node_id) = &failure.node_id {
        text.push_str(&format!("\nnode: {}", node_id.0));
    }
    if !failure.details.is_empty() {
        let details = &failure.details;
        let details = match serde_json::to_string(details) {
            Ok(details) => details,
            Err(error) => format!("<failed to encode details: {error}>"),
        };
        text.push_str(&format!("\ndetails: {details}"));
    }
    text
}
