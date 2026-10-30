use gpui::{
    AnyElement, App, FocusHandle, IntoElement, KeyDownEvent, MouseButton, RenderOnce, Role, Window,
};
use serde::{Deserialize, Serialize};
use sim_runtime::{
    AttemptId, AttemptPresentation, AttemptState, ExecutionControlCommandKind, ExecutionFailure,
    ExecutionSnapshot, ExecutionSnapshotStatus,
};
use sim_types::RequestId;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};
use ui::{Button, ButtonCommon, ButtonSize, Disableable, prelude::*};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionJobTab {
    #[default]
    All,
    Active,
    Completed,
}

impl ExecutionJobTab {
    const ALL: [Self; 3] = [Self::All, Self::Active, Self::Completed];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Active => "Active",
            Self::Completed => "Completed",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionWorkflowFilter {
    #[default]
    All,
    Selected,
}

impl ExecutionWorkflowFilter {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::All => "All workflows",
            Self::Selected => "Selected workflow",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionSortMode {
    #[default]
    MostRecent,
    Oldest,
    TotalGenerationTime,
}

impl ExecutionSortMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::MostRecent => "Most recent",
            Self::Oldest => "Oldest",
            Self::TotalGenerationTime => "Generation time",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ExecutionSurfaceAction {
    SelectJobTab(ExecutionJobTab),
    SetJobSearch(String),
    ToggleWorkflowFilter,
    CycleSortMode,
    ToggleShowProgress,
    ToggleJobDetails(AttemptId),
    SetJobDetailsTriggerHovered(AttemptId, bool),
    SetJobDetailsContentHovered(AttemptId, bool),
    ToggleJobContextMenu(AttemptId),
    CopyAttemptId(AttemptId),
    InspectAttempt(AttemptId),
    CancelAttempt(AttemptId),
    RetryAttempt(AttemptId),
    RemoveAttempt(AttemptId),
    ClearPending,
    DismissNotification(u64),
    ViewErrors,
    DismissErrorOverlay(AttemptId),
    SetErrorSearch(String),
    ToggleAllErrors,
    ToggleErrorDetails(AttemptId),
    CopyError(AttemptId),
    LocateError(AttemptId),
    OpenErrorHelp(AttemptId),
    OpenErrorGitHub(AttemptId),
}

pub(crate) type ExecutionSurfaceActionHandler =
    Arc<dyn Fn(ExecutionSurfaceAction, &mut Window, &mut App) + 'static>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExecutionNotificationKind {
    Queueing,
    Queued,
    Success,
    Failure,
    Cancelled,
    Interrupted,
    Summary,
}

pub(crate) const EXECUTION_REQUEST_TRACKING_CAPACITY: usize = 256;
pub(crate) const EXECUTION_ATTEMPT_TRACKING_CAPACITY: usize = 1_024;
pub(crate) const EXECUTION_NOTIFICATION_FIFO_CAPACITY: usize = 32;

#[cfg(all(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExecutionSurfaceBoundedCountsForTest {
    pub observed_attempt_states: usize,
    pub observed_queueing_requests: usize,
    pub observed_queued_requests: usize,
    pub queue_request_batch_counts: usize,
    pub pending_notifications: usize,
    pub current_notification: usize,
    pub coalesced_notifications: usize,
    pub coalesced_failures: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExecutionNotification {
    pub identity: u64,
    pub attempt_id: AttemptId,
    pub request_id: Option<RequestId>,
    pub batch_count: usize,
    pub kind: ExecutionNotificationKind,
    pub title: String,
    pub message: String,
    pub thumbnail_count: usize,
    pub show_view_errors: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExecutionProgressToastPhase {
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExecutionProgressToast {
    pub identity: u64,
    pub attempt_id: AttemptId,
    pub phase: ExecutionProgressToastPhase,
    pub completed: u64,
    pub total: u64,
    pub node_label: Option<String>,
    pub message: String,
}

#[derive(Default)]
pub(crate) struct ExecutionSurfaceRuntimeState {
    observed_attempt_states: HashMap<AttemptId, AttemptState>,
    observed_queueing_requests: HashSet<RequestId>,
    observed_queueing_order: VecDeque<RequestId>,
    observed_queued_requests: HashSet<RequestId>,
    observed_queued_order: VecDeque<RequestId>,
    queue_request_batch_counts: HashMap<RequestId, usize>,
    queue_request_batch_order: VecDeque<RequestId>,
    pending_notifications: VecDeque<ExecutionNotification>,
    current_notification: Option<ExecutionNotification>,
    coalesced_notification_count: usize,
    coalesced_failure_count: usize,
    progress_toast: Option<ExecutionProgressToast>,
    next_identity: u64,
}

impl ExecutionSurfaceRuntimeState {
    pub(crate) fn prime(&mut self, snapshot: &ExecutionSnapshot) {
        self.observed_attempt_states = tracked_attempts(snapshot)
            .into_iter()
            .map(|attempt| (attempt.attempt_id, attempt.state))
            .collect();
        for pending in snapshot
            .pending_commands
            .iter()
            .rev()
            .take(EXECUTION_REQUEST_TRACKING_CAPACITY)
            .rev()
        {
            if let ExecutionControlCommandKind::Queue { plan, .. } = &pending.command.kind {
                remember_request(
                    &mut self.observed_queueing_requests,
                    &mut self.observed_queueing_order,
                    pending.command.request_id,
                );
                remember_batch_count(
                    &mut self.queue_request_batch_counts,
                    &mut self.queue_request_batch_order,
                    pending.command.request_id,
                    batch_count_from_plan(plan),
                );
            }
        }
        for request_id in snapshot
            .recent_command_results
            .iter()
            .rev()
            .take(EXECUTION_REQUEST_TRACKING_CAPACITY)
            .rev()
            .filter(|acknowledgement| {
                matches!(
                    acknowledgement.outcome,
                    sim_runtime::ExecutionCommandOutcome::Accepted {
                        assigned_attempt_id: Some(_)
                    }
                )
            })
            .map(|acknowledgement| acknowledgement.request_id)
        {
            remember_request(
                &mut self.observed_queued_requests,
                &mut self.observed_queued_order,
                request_id,
            );
        }
        let identity = self.next_identity();
        self.progress_toast = running_toast(snapshot, identity);
    }

    pub(crate) fn reconcile(&mut self, snapshot: &ExecutionSnapshot) {
        for pending in snapshot
            .pending_commands
            .iter()
            .rev()
            .take(EXECUTION_REQUEST_TRACKING_CAPACITY)
            .rev()
        {
            let ExecutionControlCommandKind::Queue { plan, .. } = &pending.command.kind else {
                continue;
            };
            let request_id = pending.command.request_id;
            let batch_count = batch_count_from_plan(plan);
            remember_batch_count(
                &mut self.queue_request_batch_counts,
                &mut self.queue_request_batch_order,
                request_id,
                batch_count,
            );
            if !self.observed_queueing_requests.contains(&request_id) {
                self.observe_prompt_queueing(request_id, batch_count);
            }
        }
        for acknowledgement in snapshot
            .recent_command_results
            .iter()
            .rev()
            .take(EXECUTION_REQUEST_TRACKING_CAPACITY)
            .rev()
        {
            let sim_runtime::ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(attempt_id),
            } = &acknowledgement.outcome
            else {
                continue;
            };
            if self
                .observed_queued_requests
                .contains(&acknowledgement.request_id)
            {
                continue;
            }
            let batch_count = remove_batch_count(
                &mut self.queue_request_batch_counts,
                &mut self.queue_request_batch_order,
                acknowledgement.request_id,
            )
            .or_else(|| {
                snapshot
                    .queue
                    .iter()
                    .find(|queued| queued.attempt_id == *attempt_id)
                    .map(|queued| batch_count_from_plan(&queued.plan))
            })
            .unwrap_or(1);
            self.observe_prompt_queued(acknowledgement.request_id, batch_count, *attempt_id);
        }

        let tracked_attempts = tracked_attempts(snapshot);
        let current_attempts = tracked_attempts
            .iter()
            .map(|attempt| attempt.attempt_id)
            .collect::<HashSet<_>>();
        self.observed_attempt_states
            .retain(|attempt_id, _| current_attempts.contains(attempt_id));

        for attempt in tracked_attempts {
            let previous = self
                .observed_attempt_states
                .insert(attempt.attempt_id, attempt.state);
            if previous.is_some_and(|previous| previous != attempt.state)
                && attempt.state.is_terminal()
            {
                let identity = self.next_identity();
                self.enqueue_notification(notification_for_attempt(attempt, identity));
            }
        }
        if self.current_notification.is_none() {
            self.current_notification = self.pending_notifications.pop_front();
        }

        let toast_attempt = self.progress_toast.as_ref().and_then(|toast| {
            snapshot
                .attempts
                .iter()
                .find(|attempt| attempt.attempt_id == toast.attempt_id)
                .cloned()
        });
        if let Some(attempt) = toast_attempt {
            let terminal_identity = self
                .progress_toast
                .as_ref()
                .is_some_and(|toast| {
                    toast.phase == ExecutionProgressToastPhase::Running
                        && attempt.state.is_terminal()
                })
                .then(|| self.next_identity());
            let Some(toast) = self.progress_toast.as_mut() else {
                return;
            };
            if let Some(progress) = attempt.progress.as_ref() {
                toast.completed = progress.completed;
                toast.total = progress.total.max(1);
                toast.node_label = progress.node_id.as_ref().map(|node_id| node_id.0.clone());
            }
            if let Some(identity) = terminal_identity {
                toast.identity = identity;
                toast.phase = toast_phase(attempt.state);
                toast.message = attempt.failure.as_ref().map_or_else(
                    || state_label(attempt.state).to_owned(),
                    |failure| format!("{}: {}", failure.code, failure.message),
                );
            }
            return;
        }

        if self.progress_toast.is_some() {
            self.progress_toast = None;
        }

        if self.progress_toast.is_none() {
            let identity = self.next_identity();
            self.progress_toast = running_toast(snapshot, identity);
        }
    }

    pub(crate) fn current_notification(&self) -> Option<&ExecutionNotification> {
        self.current_notification.as_ref()
    }

    pub(crate) fn observe_prompt_queueing(&mut self, request_id: RequestId, batch_count: usize) {
        let batch_count = batch_count.max(1);
        remember_batch_count(
            &mut self.queue_request_batch_counts,
            &mut self.queue_request_batch_order,
            request_id,
            batch_count,
        );
        if self.observed_queueing_requests.contains(&request_id) {
            return;
        }
        remember_request(
            &mut self.observed_queueing_requests,
            &mut self.observed_queueing_order,
            request_id,
        );
        if self.notification_for_request(request_id).is_some() {
            return;
        }
        let identity = self.next_identity();
        let job_label = if batch_count == 1 { "job" } else { "jobs" };
        self.enqueue_notification(ExecutionNotification {
            identity,
            attempt_id: AttemptId(uuid::Uuid::nil()),
            request_id: Some(request_id),
            batch_count,
            kind: ExecutionNotificationKind::Queueing,
            title: if batch_count == 1 {
                "Queuing job".to_owned()
            } else {
                format!("Queuing {batch_count} jobs")
            },
            message: format!("{batch_count} {job_label} being added to the native queue"),
            thumbnail_count: 0,
            show_view_errors: false,
        });
    }

    pub(crate) fn observe_prompt_queued(
        &mut self,
        request_id: RequestId,
        batch_count: usize,
        attempt_id: AttemptId,
    ) {
        let batch_count = batch_count.max(1);
        remove_batch_count(
            &mut self.queue_request_batch_counts,
            &mut self.queue_request_batch_order,
            request_id,
        );
        if self.observed_queued_requests.contains(&request_id) {
            return;
        }
        remember_request(
            &mut self.observed_queued_requests,
            &mut self.observed_queued_order,
            request_id,
        );
        if let Some(notification) = self.current_notification.as_mut()
            && notification.kind == ExecutionNotificationKind::Queueing
            && notification.request_id == Some(request_id)
        {
            upgrade_to_queued(notification, batch_count, attempt_id);
            return;
        }
        if let Some(notification) = self.pending_notifications.iter_mut().find(|notification| {
            notification.kind == ExecutionNotificationKind::Queueing
                && notification.request_id == Some(request_id)
        }) {
            upgrade_to_queued(notification, batch_count, attempt_id);
            return;
        }
        let identity = self.next_identity();
        let mut notification = ExecutionNotification {
            identity,
            attempt_id,
            request_id: Some(request_id),
            batch_count,
            kind: ExecutionNotificationKind::Queued,
            title: String::new(),
            message: String::new(),
            thumbnail_count: 0,
            show_view_errors: false,
        };
        upgrade_to_queued(&mut notification, batch_count, attempt_id);
        self.enqueue_notification(notification);
    }

    pub(crate) fn observe_prompt_queue_failed(
        &mut self,
        request_id: RequestId,
        batch_count: usize,
        failure: &ExecutionFailure,
    ) {
        let batch_count = batch_count.max(1);
        remove_batch_count(
            &mut self.queue_request_batch_counts,
            &mut self.queue_request_batch_order,
            request_id,
        );
        if self.observed_queued_requests.contains(&request_id) {
            return;
        }
        remember_request(
            &mut self.observed_queued_requests,
            &mut self.observed_queued_order,
            request_id,
        );
        if let Some(notification) = self.current_notification.as_mut()
            && notification.kind == ExecutionNotificationKind::Queueing
            && notification.request_id == Some(request_id)
        {
            upgrade_to_queue_failure(notification, batch_count, failure);
            return;
        }
        if let Some(notification) = self.pending_notifications.iter_mut().find(|notification| {
            notification.kind == ExecutionNotificationKind::Queueing
                && notification.request_id == Some(request_id)
        }) {
            upgrade_to_queue_failure(notification, batch_count, failure);
            return;
        }
        let identity = self.next_identity();
        let mut notification = ExecutionNotification {
            identity,
            attempt_id: AttemptId(uuid::Uuid::nil()),
            request_id: Some(request_id),
            batch_count,
            kind: ExecutionNotificationKind::Failure,
            title: String::new(),
            message: String::new(),
            thumbnail_count: 0,
            show_view_errors: false,
        };
        upgrade_to_queue_failure(&mut notification, batch_count, failure);
        self.enqueue_notification(notification);
    }

    fn notification_for_request(&self, request_id: RequestId) -> Option<&ExecutionNotification> {
        self.current_notification
            .iter()
            .chain(self.pending_notifications.iter())
            .find(|notification| notification.request_id == Some(request_id))
    }

    fn enqueue_notification(&mut self, notification: ExecutionNotification) {
        if self.current_notification.is_none() {
            self.current_notification = Some(notification);
            return;
        }
        if self.pending_notifications.len() < EXECUTION_NOTIFICATION_FIFO_CAPACITY {
            self.pending_notifications.push_back(notification);
            return;
        }

        if notification.kind == ExecutionNotificationKind::Failure {
            if let Some(position) = self
                .pending_notifications
                .iter()
                .position(|queued| queued.kind != ExecutionNotificationKind::Failure)
            {
                self.pending_notifications.remove(position);
            } else {
                self.pending_notifications.pop_front();
                self.coalesced_failure_count = self.coalesced_failure_count.saturating_add(1);
            }
            self.coalesced_notification_count = self.coalesced_notification_count.saturating_add(1);
            self.pending_notifications.push_back(notification);
        } else {
            self.coalesced_notification_count = self.coalesced_notification_count.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn notification_count_for_test(&self) -> usize {
        self.pending_notifications.len()
            + usize::from(self.current_notification.is_some())
            + usize::from(self.coalesced_notification_count > 0)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn bounded_counts_for_test(&self) -> ExecutionSurfaceBoundedCountsForTest {
        ExecutionSurfaceBoundedCountsForTest {
            observed_attempt_states: self.observed_attempt_states.len(),
            observed_queueing_requests: self.observed_queueing_requests.len(),
            observed_queued_requests: self.observed_queued_requests.len(),
            queue_request_batch_counts: self.queue_request_batch_counts.len(),
            pending_notifications: self.pending_notifications.len(),
            current_notification: usize::from(self.current_notification.is_some()),
            coalesced_notifications: self.coalesced_notification_count,
            coalesced_failures: self.coalesced_failure_count,
        }
    }

    pub(crate) fn dismiss_notification(&mut self, identity: u64) -> bool {
        if self
            .current_notification
            .as_ref()
            .is_none_or(|notification| notification.identity != identity)
        {
            return false;
        }
        self.current_notification = self.pending_notifications.pop_front();
        if self.current_notification.is_none() && self.coalesced_notification_count > 0 {
            let omitted = self.coalesced_notification_count;
            let failures = self.coalesced_failure_count;
            self.coalesced_notification_count = 0;
            self.coalesced_failure_count = 0;
            let identity = self.next_identity();
            self.current_notification = Some(ExecutionNotification {
                identity,
                attempt_id: AttemptId(uuid::Uuid::nil()),
                request_id: None,
                batch_count: omitted.max(1),
                kind: ExecutionNotificationKind::Summary,
                title: "Additional execution activity".to_owned(),
                message: if failures == 0 {
                    format!("{omitted} additional notifications were coalesced")
                } else {
                    format!(
                        "{omitted} additional notifications were coalesced, including {failures} failures; full execution errors remain in the Errors tab"
                    )
                },
                thumbnail_count: 0,
                show_view_errors: failures > 0,
            });
        }
        true
    }

    pub(crate) fn progress_toast(&self) -> Option<&ExecutionProgressToast> {
        self.progress_toast.as_ref()
    }

    pub(crate) fn dismiss_progress_toast(&mut self, identity: u64) -> bool {
        if self
            .progress_toast
            .as_ref()
            .is_none_or(|toast| toast.identity != identity)
        {
            return false;
        }
        self.progress_toast = None;
        true
    }

    fn next_identity(&mut self) -> u64 {
        self.next_identity = self.next_identity.checked_add(1).unwrap_or(1);
        self.next_identity
    }
}

fn notification_for_attempt(attempt: &AttemptPresentation, identity: u64) -> ExecutionNotification {
    let (kind, title, message) = match attempt.state {
        AttemptState::Succeeded => (
            ExecutionNotificationKind::Success,
            "Execution completed".to_owned(),
            format!(
                "Attempt {} produced {} outputs",
                attempt.attempt_id.0,
                attempt.outputs.len()
            ),
        ),
        AttemptState::Failed => {
            let message = attempt.failure.as_ref().map_or_else(
                || "The attempt failed without structured details".to_owned(),
                |failure| format!("{}: {}", failure.code, failure.message),
            );
            (
                ExecutionNotificationKind::Failure,
                "Execution failed".to_owned(),
                message,
            )
        }
        AttemptState::Cancelled => (
            ExecutionNotificationKind::Cancelled,
            "Execution cancelled".to_owned(),
            format!("Attempt {} was cancelled", attempt.attempt_id.0),
        ),
        AttemptState::Interrupted => (
            ExecutionNotificationKind::Interrupted,
            "Execution interrupted".to_owned(),
            attempt
                .interrupted_reason
                .clone()
                .unwrap_or_else(|| format!("Attempt {} was interrupted", attempt.attempt_id.0)),
        ),
        AttemptState::Queued | AttemptState::Running | AttemptState::Cancelling => (
            ExecutionNotificationKind::Interrupted,
            "Execution state changed".to_owned(),
            state_label(attempt.state).to_owned(),
        ),
    };
    ExecutionNotification {
        identity,
        attempt_id: attempt.attempt_id,
        request_id: None,
        batch_count: 1,
        kind,
        title,
        message,
        thumbnail_count: attempt
            .outputs
            .iter()
            .filter(|output| {
                matches!(
                    output.media_kind,
                    sim_runtime::OutputMediaKind::Image
                        | sim_runtime::OutputMediaKind::Animation
                        | sim_runtime::OutputMediaKind::Video
                        | sim_runtime::OutputMediaKind::ThreeD
                )
            })
            .count(),
        show_view_errors: attempt.failure.is_some(),
    }
}

fn upgrade_to_queued(
    notification: &mut ExecutionNotification,
    batch_count: usize,
    attempt_id: AttemptId,
) {
    notification.kind = ExecutionNotificationKind::Queued;
    notification.attempt_id = attempt_id;
    notification.batch_count = batch_count;
    if batch_count == 1 {
        notification.title = "Job queued".to_owned();
        notification.message = "1 job added to queue".to_owned();
    } else {
        notification.title = format!("{batch_count} jobs queued");
        notification.message = format!("{batch_count} jobs added to queue");
    }
}

fn upgrade_to_queue_failure(
    notification: &mut ExecutionNotification,
    batch_count: usize,
    failure: &ExecutionFailure,
) {
    notification.kind = ExecutionNotificationKind::Failure;
    notification.batch_count = batch_count;
    notification.title = if batch_count == 1 {
        "Job could not be queued".to_owned()
    } else {
        format!("{batch_count} jobs could not be queued")
    };
    notification.message = format!("{}: {}", failure.code, failure.message);
}

fn batch_count_from_plan(plan: &sim_runtime::CompiledPlan) -> usize {
    ["batch_count", "batchCount"]
        .into_iter()
        .find_map(|key| plan.extra_data.get(key).and_then(serde_json::Value::as_u64))
        .and_then(|count| usize::try_from(count).ok())
        .filter(|count| *count > 0)
        .unwrap_or(1)
}

fn tracked_attempts(snapshot: &ExecutionSnapshot) -> Vec<&AttemptPresentation> {
    let mut tracked = Vec::with_capacity(
        snapshot
            .attempts
            .len()
            .min(EXECUTION_ATTEMPT_TRACKING_CAPACITY),
    );
    let mut seen = HashSet::with_capacity(tracked.capacity());
    for attempt in snapshot
        .attempts
        .iter()
        .rev()
        .filter(|attempt| !attempt.state.is_terminal())
        .chain(
            snapshot
                .attempts
                .iter()
                .rev()
                .filter(|attempt| attempt.failure.is_some()),
        )
        .chain(
            snapshot
                .attempts
                .iter()
                .rev()
                .filter(|attempt| attempt.state.is_terminal()),
        )
    {
        if seen.insert(attempt.attempt_id) {
            tracked.push(attempt);
            if tracked.len() == EXECUTION_ATTEMPT_TRACKING_CAPACITY {
                break;
            }
        }
    }
    tracked
}

fn remember_request(
    requests: &mut HashSet<RequestId>,
    order: &mut VecDeque<RequestId>,
    request_id: RequestId,
) {
    requests.insert(request_id);
    order.retain(|candidate| *candidate != request_id);
    order.push_back(request_id);
    while order.len() > EXECUTION_REQUEST_TRACKING_CAPACITY {
        if let Some(expired) = order.pop_front() {
            requests.remove(&expired);
        }
    }
}

fn remember_batch_count(
    counts: &mut HashMap<RequestId, usize>,
    order: &mut VecDeque<RequestId>,
    request_id: RequestId,
    batch_count: usize,
) {
    counts.insert(request_id, batch_count.max(1));
    order.retain(|candidate| *candidate != request_id);
    order.push_back(request_id);
    while order.len() > EXECUTION_REQUEST_TRACKING_CAPACITY {
        if let Some(expired) = order.pop_front() {
            counts.remove(&expired);
        }
    }
}

fn remove_batch_count(
    counts: &mut HashMap<RequestId, usize>,
    order: &mut VecDeque<RequestId>,
    request_id: RequestId,
) -> Option<usize> {
    order.retain(|candidate| *candidate != request_id);
    counts.remove(&request_id)
}

fn running_toast(snapshot: &ExecutionSnapshot, identity: u64) -> Option<ExecutionProgressToast> {
    let attempt = snapshot
        .attempts
        .iter()
        .find(|attempt| attempt.state == AttemptState::Running && attempt.progress.is_some())?;
    let progress = attempt.progress.as_ref()?;
    Some(ExecutionProgressToast {
        identity,
        attempt_id: attempt.attempt_id,
        phase: ExecutionProgressToastPhase::Running,
        completed: progress.completed,
        total: progress.total.max(1),
        node_label: progress.node_id.as_ref().map(|node_id| node_id.0.clone()),
        message: "native execution in progress".to_owned(),
    })
}

fn toast_phase(state: AttemptState) -> ExecutionProgressToastPhase {
    match state {
        AttemptState::Succeeded => ExecutionProgressToastPhase::Succeeded,
        AttemptState::Failed => ExecutionProgressToastPhase::Failed,
        AttemptState::Cancelled => ExecutionProgressToastPhase::Cancelled,
        AttemptState::Interrupted => ExecutionProgressToastPhase::Interrupted,
        AttemptState::Queued | AttemptState::Running | AttemptState::Cancelling => {
            ExecutionProgressToastPhase::Running
        }
    }
}

#[derive(IntoElement)]
pub(crate) struct JobFiltersSurface {
    pub selected_tab: ExecutionJobTab,
    pub search_query: String,
    pub workflow_filter: ExecutionWorkflowFilter,
    pub sort_mode: ExecutionSortMode,
    pub selected_workflow_available: bool,
    pub show_progress: bool,
    pub active_count: usize,
    pub completed_count: usize,
    pub queued_count: usize,
    pub runtime_controls_available: bool,
    pub clear_queue_focus_handle: FocusHandle,
    pub on_action: ExecutionSurfaceActionHandler,
}

impl RenderOnce for JobFiltersSurface {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let search_value = self.search_query.clone();
        let search_handler = self.on_action.clone();
        let workflow_handler = self.on_action.clone();
        let sort_handler = self.on_action.clone();
        let progress_handler = self.on_action.clone();
        let clear_handler = self.on_action.clone();
        let clear_mouse_focus_handle = self.clear_queue_focus_handle.clone();
        let clear_available = self.queued_count > 0 && self.runtime_controls_available;
        let clear_unavailable_reason = if self.queued_count == 0 {
            Some("queue is empty")
        } else if !self.runtime_controls_available {
            Some("native runtime controller is not connected")
        } else {
            None
        };

        v_flex()
            .id("sim-job-filters-bar")
            .debug_selector(|| "SIM-SURFACE-JOB-FILTERS-BAR".into())
            .w_full()
            .gap_1()
            .px_2()
            .py_1()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(
                h_flex()
                    .id("sim-job-filter-tabs")
                    .debug_selector(|| "SIM-SURFACE-JOB-FILTER-TABS".into())
                    .role(Role::TabList)
                    .aria_label("Execution job state filters")
                    .w_full()
                    .gap_1()
                    .children(ExecutionJobTab::ALL.into_iter().map(|tab| {
                        let selected = tab == self.selected_tab;
                        let count = match tab {
                            ExecutionJobTab::All => self.active_count + self.completed_count,
                            ExecutionJobTab::Active => self.active_count,
                            ExecutionJobTab::Completed => self.completed_count,
                        };
                        let handler = self.on_action.clone();
                        Button::new(
                            format!("sim-job-filter-tab-{}", tab.label().to_lowercase()),
                            format!("{} ({count})", tab.label()),
                        )
                        .size(ButtonSize::Compact)
                        .aria_role(Role::Tab)
                        .toggle_state(selected)
                        .aria_label(format!(
                            "Show {} execution jobs, {count} available",
                            tab.label()
                        ))
                        .on_click(move |_, window, cx| {
                            handler(ExecutionSurfaceAction::SelectJobTab(tab), window, cx);
                        })
                    })),
            )
            .child(
                h_flex()
                    .id("sim-job-filter-actions")
                    .debug_selector(|| "SIM-SURFACE-JOB-FILTER-ACTIONS".into())
                    .w_full()
                    .gap_1()
                    .child(
                        div()
                            .id("sim-job-search-input")
                            .role(Role::SearchInput)
                            .tab_stop(true)
                            .aria_label("Search execution jobs by attempt, prompt, state, or error")
                            .aria_value(if self.search_query.is_empty() {
                                "No search filter".to_owned()
                            } else {
                                self.search_query.clone()
                            })
                            .min_w_32()
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .border_1()
                            .border_color(cx.theme().colors().border)
                            .text_ui_sm(cx)
                            .text_color(if self.search_query.is_empty() {
                                cx.theme().colors().text_muted
                            } else {
                                cx.theme().colors().text
                            })
                            .on_key_down(move |event: &KeyDownEvent, window, cx| {
                                if let Some(value) = edit_search_query(&search_value, event) {
                                    cx.stop_propagation();
                                    search_handler(
                                        ExecutionSurfaceAction::SetJobSearch(value),
                                        window,
                                        cx,
                                    );
                                }
                            })
                            .child(if self.search_query.is_empty() {
                                "Type to search jobs".to_owned()
                            } else {
                                self.search_query.clone()
                            }),
                    )
                    .child(
                        Button::new("sim-job-workflow-filter", self.workflow_filter.label())
                            .size(ButtonSize::Compact)
                            .disabled(!self.selected_workflow_available)
                            .aria_label(if self.selected_workflow_available {
                                format!(
                                    "Workflow filter is {}; activate to change it",
                                    self.workflow_filter.label()
                                )
                            } else {
                                "Selected-workflow filter unavailable: select an execution attempt"
                                    .to_owned()
                            })
                            .on_click(move |_, window, cx| {
                                workflow_handler(
                                    ExecutionSurfaceAction::ToggleWorkflowFilter,
                                    window,
                                    cx,
                                );
                            }),
                    )
                    .child(
                        Button::new(
                            "sim-job-sort-mode",
                            format!("Sort: {}", self.sort_mode.label()),
                        )
                        .size(ButtonSize::Compact)
                        .aria_label(format!(
                            "Execution sort is {}; activate to choose the next mode",
                            self.sort_mode.label()
                        ))
                        .on_click(move |_, window, cx| {
                            sort_handler(ExecutionSurfaceAction::CycleSortMode, window, cx);
                        }),
                    )
                    .child(
                        Button::new(
                            "sim-toggle-execution-progress",
                            if self.show_progress {
                                "Hide Progress"
                            } else {
                                "Show Progress"
                            },
                        )
                        .size(ButtonSize::Compact)
                        .toggle_state(self.show_progress)
                        .aria_label(if self.show_progress {
                            "Hide native execution progress surfaces"
                        } else {
                            "Show native execution progress surfaces"
                        })
                        .on_click(move |_, window, cx| {
                            progress_handler(
                                ExecutionSurfaceAction::ToggleShowProgress,
                                window,
                                cx,
                            );
                        }),
                    )
                    .child(
                        h_flex()
                            .id("sim-output-history-queue-summary")
                            .role(Role::Status)
                            .aria_label(format!("{0} queued execution jobs", self.queued_count))
                            .gap_1()
                            .child(
                                div()
                                    .text_ui_sm(cx)
                                    .child(format!("Queued {}", self.queued_count)),
                            )
                            .child(
                                div()
                                    .id("sim-output-history-clear-queue")
                                    .debug_selector(|| "SIM-SURFACE-CLEAR-QUEUE".into())
                                    .role(Role::Button)
                                    .tab_stop(true)
                                    .tab_index(0)
                                    .track_focus(&self.clear_queue_focus_handle)
                                    .aria_label(clear_unavailable_reason.map_or_else(
                                        || "Clear all queued execution jobs".to_owned(),
                                        |reason| {
                                            format!(
                                                "Clear queued execution jobs unavailable: {reason}"
                                            )
                                        },
                                    ))
                                    .rounded_sm()
                                    .px_2()
                                    .py_1()
                                    .text_ui_sm(cx)
                                    .when(clear_available, |this| {
                                        let keyboard_clear_handler = clear_handler.clone();
                                        this.cursor_pointer()
                                            .on_mouse_down(MouseButton::Left, |_, window, _| {
                                                window.prevent_default();
                                            })
                                            .on_click(move |_, window, cx| {
                                                cx.stop_propagation();
                                                clear_mouse_focus_handle.focus(window, cx);
                                                clear_handler(
                                                    ExecutionSurfaceAction::ClearPending,
                                                    window,
                                                    cx,
                                                );
                                            })
                                            .on_key_down(move |event: &KeyDownEvent, window, cx| {
                                                if matches!(
                                                    event.keystroke.key.as_str(),
                                                    "enter" | "space"
                                                ) {
                                                    cx.stop_propagation();
                                                    keyboard_clear_handler(
                                                        ExecutionSurfaceAction::ClearPending,
                                                        window,
                                                        cx,
                                                    );
                                                }
                                            })
                                    })
                                    .when(!clear_available, |this| {
                                        this.cursor_not_allowed().opacity(0.5)
                                    })
                                    .child("Clear"),
                            )
                            .when_some(clear_unavailable_reason, |this, reason| {
                                this.child(
                                    div()
                                        .id("sim-clear-queue-unavailable-reason")
                                        .debug_selector(|| {
                                            "SIM-SURFACE-CLEAR-QUEUE-UNAVAILABLE-REASON".into()
                                        })
                                        .role(Role::Status)
                                        .aria_label(format!("Clear Queue unavailable: {reason}"))
                                        .text_ui_sm(cx)
                                        .text_color(cx.theme().colors().text_muted)
                                        .child(reason),
                                )
                            }),
                    ),
            )
    }
}

#[derive(IntoElement)]
pub(crate) struct JobDetailsSurface {
    pub attempt: AttemptPresentation,
    pub details_open: bool,
    pub context_menu_open: bool,
    pub runtime_controls_available: bool,
    pub snapshot_status: ExecutionSnapshotStatus,
    pub on_action: ExecutionSurfaceActionHandler,
}

impl RenderOnce for JobDetailsSurface {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let attempt_id = self.attempt.attempt_id;
        let label = short_attempt_label(attempt_id);
        let details_handler = self.on_action.clone();
        let details_keyboard_handler = self.on_action.clone();
        let details_hover_handler = self.on_action.clone();
        let content_hover_handler = self.on_action.clone();
        let menu_handler = self.on_action.clone();
        let copy_handler = self.on_action.clone();
        let status_ready = snapshot_allows_controls(&self.snapshot_status);

        v_flex()
            .id("sim-job-details-hover-popover")
            .debug_selector(|| "SIM-SURFACE-JOB-DETAILS-HOVER-POPOVER".into())
            .w_full()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(
                h_flex()
                    .w_full()
                    .px_2()
                    .py_1()
                    .gap_1()
                    .child(
                        div()
                            .id("sim-selected-job-status")
                            .role(Role::Status)
                            .aria_label(format!(
                                "Selected execution attempt {}, {}",
                                attempt_id.0,
                                state_label(self.attempt.state)
                            ))
                            .text_ui_sm(cx)
                            .child(format!(
                                "Selected {label} · {}",
                                state_label(self.attempt.state)
                            )),
                    )
                    .child(
                        div()
                            .id("sim-job-details-hover-trigger")
                            .debug_selector(|| "SIM-SURFACE-JOB-DETAILS-HOVER-TRIGGER".into())
                            .role(Role::Button)
                            .tab_stop(true)
                            .tab_index(0)
                            .aria_expanded(self.details_open)
                            .aria_label(if self.details_open {
                                format!("Hide details for execution attempt {label}")
                            } else {
                                format!("Show details for execution attempt {label}")
                            })
                            .rounded_sm()
                            .px_1()
                            .text_ui_sm(cx)
                            .cursor_pointer()
                            .hover(|style| style.bg(cx.theme().colors().element_hover))
                            .on_mouse_down(MouseButton::Left, |_, window, _| {
                                window.prevent_default();
                            })
                            .on_hover(move |&hovered, window, cx| {
                                details_hover_handler(
                                    ExecutionSurfaceAction::SetJobDetailsTriggerHovered(
                                        attempt_id, hovered,
                                    ),
                                    window,
                                    cx,
                                );
                            })
                            .on_click(move |_, window, cx| {
                                cx.stop_propagation();
                                details_handler(
                                    ExecutionSurfaceAction::ToggleJobDetails(attempt_id),
                                    window,
                                    cx,
                                );
                            })
                            .on_key_down(move |event: &KeyDownEvent, window, cx| {
                                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                    cx.stop_propagation();
                                    details_keyboard_handler(
                                        ExecutionSurfaceAction::ToggleJobDetails(attempt_id),
                                        window,
                                        cx,
                                    );
                                }
                            })
                            .child(if self.details_open {
                                "Hide Details"
                            } else {
                                "Details"
                            }),
                    )
                    .child(
                        div()
                            .debug_selector(|| "SIM-SURFACE-JOB-CONTEXT-TRIGGER".into())
                            .child(
                                Button::new(
                                    "sim-job-context-menu-toggle",
                                    if self.context_menu_open {
                                        "Close Actions"
                                    } else {
                                        "Actions"
                                    },
                                )
                                .size(ButtonSize::Compact)
                                .aria_expanded(self.context_menu_open)
                                .aria_label(format!("Execution actions for attempt {label}"))
                                .on_click(move |_, window, cx| {
                                    menu_handler(
                                        ExecutionSurfaceAction::ToggleJobContextMenu(attempt_id),
                                        window,
                                        cx,
                                    );
                                }),
                            ),
                    )
                    .child(
                        Button::new("sim-job-copy-id", "Copy ID")
                            .size(ButtonSize::Compact)
                            .aria_label(format!("Copy execution attempt ID {label}"))
                            .on_click(move |_, window, cx| {
                                copy_handler(
                                    ExecutionSurfaceAction::CopyAttemptId(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                    ),
            )
            .when(self.details_open, |this| {
                this.child(render_job_details(
                    &self.attempt,
                    content_hover_handler,
                    window,
                    cx,
                ))
            })
            .when(self.context_menu_open, |this| {
                this.child(render_job_context_menu(
                    &self.attempt,
                    self.runtime_controls_available && status_ready,
                    self.on_action,
                    cx,
                ))
            })
    }
}

fn render_job_details(
    attempt: &AttemptPresentation,
    on_hover: ExecutionSurfaceActionHandler,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let attempt_id = attempt.attempt_id;
    let progress = attempt.progress.as_ref().map_or_else(
        || "not active".to_owned(),
        |progress| format!("{} of {}", progress.completed, progress.total),
    );
    let failure = attempt.failure.as_ref().map_or_else(
        || "none".to_owned(),
        |failure| {
            format!(
                "{:?} · {} · {}",
                failure.origin, failure.code, failure.message
            )
        },
    );
    let effective_backend = attempt.effective_backend.as_ref();

    v_flex()
        .id("sim-job-details-popover")
        .debug_selector(|| "SIM-SURFACE-JOB-DETAILS-POPOVER".into())
        .role(Role::Group)
        .aria_label(format!(
            "Execution job details for attempt {}",
            attempt_id.0
        ))
        .max_h(vh(0.42, window))
        .overflow_y_scroll()
        .on_hover(move |&hovered, window, cx| {
            on_hover(
                ExecutionSurfaceAction::SetJobDetailsContentHovered(attempt_id, hovered),
                window,
                cx,
            );
        })
        .px_2()
        .py_1()
        .gap_0p5()
        .bg(cx.theme().colors().elevated_surface_background)
        .child(detail_line("Attempt ID", attempt_id.0.to_string(), cx))
        .child(detail_line(
            "Prompt ID",
            attempt.prompt_id.0.to_string(),
            cx,
        ))
        .child(detail_line("State", state_label(attempt.state), cx))
        .child(detail_line("Progress", progress, cx))
        .child(identified_detail_line(
            "sim-job-effective-device",
            "Device",
            effective_backend.map_or_else(|| "not selected".to_owned(), effective_device_label),
            cx,
        ))
        .child(identified_detail_line(
            "sim-job-effective-memory",
            "Memory",
            effective_backend.map_or_else(|| "not selected".to_owned(), effective_memory_label),
            cx,
        ))
        .child(detail_line(
            "Outputs",
            attempt.outputs.len().to_string(),
            cx,
        ))
        .child(detail_line("Created", attempt.created_at.to_rfc3339(), cx))
        .child(detail_line(
            "Finished",
            attempt
                .finished_at
                .map(|finished| finished.to_rfc3339())
                .unwrap_or_else(|| "not finished".to_owned()),
            cx,
        ))
        .child(detail_line(
            "Retry of",
            attempt
                .retry_of
                .map(|retry_of| retry_of.0.to_string())
                .unwrap_or_else(|| "none".to_owned()),
            cx,
        ))
        .child(detail_line("Failure", failure, cx))
        .into_any_element()
}

fn effective_device_label(backend: &sim_runtime::EffectiveNativeBackendState) -> String {
    backend.architecture.as_ref().map_or_else(
        || backend.device_name.clone(),
        |architecture| format!("{} · {architecture}", backend.device_name),
    )
}

fn effective_memory_label(backend: &sim_runtime::EffectiveNativeBackendState) -> String {
    backend.total_memory_bytes.map_or_else(
        || {
            format!(
                "{} / {} bytes · {}",
                backend.memory_in_use_bytes,
                backend.memory_limit_bytes,
                backend.memory_policy.as_str()
            )
        },
        |total_memory_bytes| match backend.allocation_limit_bytes {
            Some(allocation_limit_bytes) if allocation_limit_bytes != total_memory_bytes => {
                format!(
                    "{} / {} bytes · {} bytes device ceiling · {} bytes physical · {}",
                    backend.memory_in_use_bytes,
                    backend.memory_limit_bytes,
                    allocation_limit_bytes,
                    total_memory_bytes,
                    backend.memory_policy.as_str()
                )
            }
            _ => format!(
                "{} / {} bytes · {} bytes physical · {}",
                backend.memory_in_use_bytes,
                backend.memory_limit_bytes,
                total_memory_bytes,
                backend.memory_policy.as_str()
            ),
        },
    )
}

#[cfg(all(test, feature = "test-support"))]
pub(crate) fn effective_device_label_for_test(
    backend: &sim_runtime::EffectiveNativeBackendState,
) -> String {
    effective_device_label(backend)
}

#[cfg(all(test, feature = "test-support"))]
pub(crate) fn effective_memory_label_for_test(
    backend: &sim_runtime::EffectiveNativeBackendState,
) -> String {
    effective_memory_label(backend)
}

fn detail_line(label: &'static str, value: impl Into<String>, cx: &App) -> AnyElement {
    h_flex()
        .w_full()
        .gap_2()
        .child(
            div()
                .w_24()
                .text_ui_sm(cx)
                .text_color(cx.theme().colors().text_muted)
                .child(label),
        )
        .child(div().min_w_0().text_ui_sm(cx).child(value.into()))
        .into_any_element()
}

fn identified_detail_line(
    id: &'static str,
    label: &'static str,
    value: impl Into<String>,
    cx: &App,
) -> AnyElement {
    let value = value.into();
    h_flex()
        .id(id)
        .debug_selector(move || format!("SIM-JOB-{}", label.to_ascii_uppercase()))
        .role(Role::Status)
        .aria_label(format!("{label}: {value}"))
        .w_full()
        .gap_2()
        .child(
            div()
                .w_24()
                .text_ui_sm(cx)
                .text_color(cx.theme().colors().text_muted)
                .child(label),
        )
        .child(div().min_w_0().text_ui_sm(cx).child(value))
        .into_any_element()
}

fn render_job_context_menu(
    attempt: &AttemptPresentation,
    runtime_controls_available: bool,
    on_action: ExecutionSurfaceActionHandler,
    cx: &mut App,
) -> AnyElement {
    let attempt_id = attempt.attempt_id;
    let inspect_handler = on_action.clone();
    let copy_handler = on_action.clone();
    let copy_error_handler = on_action.clone();
    let cancel_handler = on_action.clone();
    let retry_handler = on_action.clone();
    let remove_handler = on_action;
    let can_cancel = runtime_controls_available
        && matches!(
            attempt.state,
            AttemptState::Queued | AttemptState::Running | AttemptState::Cancelling
        );
    let can_retry = runtime_controls_available && attempt.retry_eligibility().is_allowed();
    let can_remove = attempt.removal_eligibility().is_allowed();

    h_flex()
        .id("sim-job-context-menu")
        .debug_selector(|| "SIM-SURFACE-JOB-CONTEXT-MENU".into())
        .role(Role::Menu)
        .aria_label(format!("Actions for execution attempt {}", attempt_id.0))
        .w_full()
        .px_2()
        .py_1()
        .gap_1()
        .bg(cx.theme().colors().elevated_surface_background)
        .child(
            Button::new("sim-job-menu-inspect", "Inspect")
                .size(ButtonSize::Compact)
                .aria_role(Role::MenuItem)
                .aria_label("Inspect this execution attempt")
                .on_click(move |_, window, cx| {
                    inspect_handler(
                        ExecutionSurfaceAction::InspectAttempt(attempt_id),
                        window,
                        cx,
                    );
                }),
        )
        .child(
            Button::new("sim-job-menu-copy-id", "Copy ID")
                .size(ButtonSize::Compact)
                .aria_role(Role::MenuItem)
                .aria_label("Copy this execution attempt ID")
                .on_click(move |_, window, cx| {
                    copy_handler(
                        ExecutionSurfaceAction::CopyAttemptId(attempt_id),
                        window,
                        cx,
                    );
                }),
        )
        .when(attempt.failure.is_some(), |this| {
            this.child(
                div()
                    .debug_selector(|| "SIM-SURFACE-JOB-CONTEXT-COPY-ERROR".into())
                    .child(
                        Button::new("sim-job-menu-copy-error", "Copy Error Message")
                            .size(ButtonSize::Compact)
                            .aria_role(Role::MenuItem)
                            .aria_label("Copy this failed attempt's structured error message")
                            .on_click(move |_, window, cx| {
                                copy_error_handler(
                                    ExecutionSurfaceAction::CopyError(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                    ),
            )
        })
        .child(
            Button::new("sim-job-menu-cancel", "Cancel")
                .size(ButtonSize::Compact)
                .aria_role(Role::MenuItem)
                .disabled(!can_cancel)
                .aria_label(if can_cancel {
                    "Cancel or interrupt this execution attempt"
                } else if !runtime_controls_available {
                    "Cancel unavailable: native runtime controller is not connected or snapshot is not ready"
                } else {
                    "Cancel unavailable: execution attempt is terminal"
                })
                .on_click(move |_, window, cx| {
                    cancel_handler(
                        ExecutionSurfaceAction::CancelAttempt(attempt_id),
                        window,
                        cx,
                    );
                }),
        )
        .child(
            Button::new("sim-job-menu-retry", "Retry")
                .size(ButtonSize::Compact)
                .aria_role(Role::MenuItem)
                .disabled(!can_retry)
                .aria_label(if can_retry {
                    "Retry this execution attempt"
                } else if !runtime_controls_available {
                    "Retry unavailable: native runtime controller is not connected or snapshot is not ready"
                } else {
                    "Retry unavailable: attempt recovery evidence does not permit retry"
                })
                .on_click(move |_, window, cx| {
                    retry_handler(
                        ExecutionSurfaceAction::RetryAttempt(attempt_id),
                        window,
                        cx,
                    );
                }),
        )
        .child(
            Button::new("sim-job-menu-remove", "Remove")
                .size(ButtonSize::Compact)
                .aria_role(Role::MenuItem)
                .disabled(!can_remove)
                .aria_label(if can_remove {
                    "Remove this terminal attempt from execution history"
                } else {
                    "Remove unavailable: only terminal attempts can be removed"
                })
                .on_click(move |_, window, cx| {
                    remove_handler(
                        ExecutionSurfaceAction::RemoveAttempt(attempt_id),
                        window,
                        cx,
                    );
                }),
        )
        .into_any_element()
}

#[derive(IntoElement)]
pub(crate) struct QueueNotificationSurface {
    pub notification: ExecutionNotification,
    pub on_action: ExecutionSurfaceActionHandler,
}

impl RenderOnce for QueueNotificationSurface {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let identity = self.notification.identity;
        let show_view_errors = self.notification.show_view_errors;
        let view_errors_handler = self.on_action.clone();
        let dismiss_handler = self.on_action;
        let role = if self.notification.kind == ExecutionNotificationKind::Failure {
            Role::Alert
        } else {
            Role::Status
        };
        let kind = match self.notification.kind {
            ExecutionNotificationKind::Queueing => "queueing",
            ExecutionNotificationKind::Queued => "queued",
            ExecutionNotificationKind::Success => "success",
            ExecutionNotificationKind::Failure => "failure",
            ExecutionNotificationKind::Cancelled => "cancelled",
            ExecutionNotificationKind::Interrupted => "interrupted",
            ExecutionNotificationKind::Summary => "summary",
        };

        v_flex()
            .id("sim-queue-notification-banner-host")
            .debug_selector(|| "SIM-SURFACE-QUEUE-NOTIFICATION-BANNER-HOST".into())
            .w_full()
            .child(
                h_flex()
                    .id("sim-queue-notification-banner")
                    .debug_selector(|| "SIM-SURFACE-QUEUE-NOTIFICATION-BANNER".into())
                    .role(role)
                    .aria_label(format!(
                        "Execution {kind}: {}. {}",
                        self.notification.title, self.notification.message
                    ))
                    .w_full()
                    .px_2()
                    .py_1()
                    .gap_2()
                    .justify_between()
                    .bg(match self.notification.kind {
                        ExecutionNotificationKind::Failure => {
                            cx.theme().status().error_background.opacity(0.20)
                        }
                        ExecutionNotificationKind::Success => {
                            cx.theme().status().success_background.opacity(0.20)
                        }
                        ExecutionNotificationKind::Queueing | ExecutionNotificationKind::Queued => {
                            cx.theme().status().info_background.opacity(0.20)
                        }
                        ExecutionNotificationKind::Cancelled
                        | ExecutionNotificationKind::Interrupted
                        | ExecutionNotificationKind::Summary => {
                            cx.theme().colors().element_background
                        }
                    })
                    .child(
                        v_flex()
                            .min_w_0()
                            .child(div().text_ui_sm(cx).child(self.notification.title))
                            .child(
                                div()
                                    .text_ui_sm(cx)
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(self.notification.message),
                            )
                            .when(self.notification.thumbnail_count > 0, |this| {
                                this.child(
                                    div()
                                        .id("sim-queue-notification-thumbnails")
                                        .role(Role::List)
                                        .aria_label(format!(
                                            "{} execution output thumbnails",
                                            self.notification.thumbnail_count
                                        ))
                                        .text_ui_sm(cx)
                                        .child(format!(
                                            "{} output previews",
                                            self.notification.thumbnail_count
                                        )),
                                )
                            }),
                    )
                    .child(
                        h_flex()
                            .gap_1()
                            .when(show_view_errors, |this| {
                                this.child(
                                    div()
                                        .debug_selector(|| {
                                            "SIM-SURFACE-NOTIFICATION-VIEW-ERRORS".into()
                                        })
                                        .child(
                                            Button::new(
                                                "sim-notification-view-errors",
                                                "View Errors",
                                            )
                                            .size(ButtonSize::Compact)
                                            .aria_label("Open durable structured execution errors")
                                            .on_click(
                                                move |_, window, cx| {
                                                    view_errors_handler(
                                                        ExecutionSurfaceAction::ViewErrors,
                                                        window,
                                                        cx,
                                                    );
                                                },
                                            ),
                                        ),
                                )
                            })
                            .child(
                                Button::new("sim-dismiss-queue-notification", "Dismiss")
                                    .size(ButtonSize::Compact)
                                    .aria_label("Dismiss this execution notification")
                                    .on_click(move |_, window, cx| {
                                        dismiss_handler(
                                            ExecutionSurfaceAction::DismissNotification(identity),
                                            window,
                                            cx,
                                        );
                                    }),
                            ),
                    ),
            )
    }
}

#[derive(IntoElement)]
pub(crate) struct ProgressToastSurface {
    pub toast: ExecutionProgressToast,
}

impl RenderOnce for ProgressToastSurface {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let total = self.toast.total.max(1);
        let completed = self.toast.completed.min(total);
        let failed = self.toast.phase == ExecutionProgressToastPhase::Failed;
        let phase = match self.toast.phase {
            ExecutionProgressToastPhase::Running => "running",
            ExecutionProgressToastPhase::Succeeded => "succeeded",
            ExecutionProgressToastPhase::Failed => "failed",
            ExecutionProgressToastPhase::Cancelled => "cancelled",
            ExecutionProgressToastPhase::Interrupted => "interrupted",
        };

        v_flex()
            .id("sim-progress-toast-item")
            .debug_selector(|| "SIM-SURFACE-PROGRESS-TOAST".into())
            .role(if failed { Role::Alert } else { Role::Status })
            .aria_label(format!(
                "Execution attempt {} {phase}: {}. Progress {completed} of {total}",
                self.toast.attempt_id.0, self.toast.message
            ))
            .w_full()
            .px_2()
            .py_1()
            .gap_0p5()
            .bg(if failed {
                cx.theme().status().error_background.opacity(0.18)
            } else {
                cx.theme().colors().element_background
            })
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .child(div().text_ui_sm(cx).child(format!("Execution {phase}")))
                    .child(
                        div()
                            .text_ui_sm(cx)
                            .text_color(cx.theme().colors().text_muted)
                            .child(format!("{completed} / {total}")),
                    ),
            )
            .when_some(self.toast.node_label, |this, node| {
                this.child(
                    div()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text_muted)
                        .child(format!("Node {node}")),
                )
            })
            .child(
                div()
                    .id("sim-linear-progress-bar")
                    .debug_selector(|| "SIM-SURFACE-LINEAR-PROGRESS-BAR".into())
                    .role(Role::ProgressIndicator)
                    .aria_label("Overall execution progress")
                    .aria_value(format!("{completed} of {total}"))
                    .aria_numeric_value(completed as f64)
                    .aria_min_numeric_value(0.0)
                    .aria_max_numeric_value(total as f64)
                    .h_1()
                    .w_full()
                    .rounded_full()
                    .bg(cx.theme().colors().panel_background)
                    .child(
                        div()
                            .h_full()
                            .rounded_full()
                            .bg(if failed {
                                cx.theme().status().error
                            } else {
                                cx.theme().status().info
                            })
                            .w(relative(completed as f32 / total as f32)),
                    ),
            )
    }
}

#[derive(IntoElement)]
pub(crate) struct ErrorExplorerSurface {
    pub snapshot: ExecutionSnapshot,
    pub search_query: String,
    pub all_collapsed: bool,
    pub collapsed_attempts: HashSet<AttemptId>,
    pub runtime_controls_available: bool,
    pub on_action: ExecutionSurfaceActionHandler,
}

impl RenderOnce for ErrorExplorerSurface {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let mut failures = self
            .snapshot
            .attempts
            .iter()
            .filter_map(|attempt| attempt.failure.as_ref().map(|failure| (attempt, failure)))
            .filter(|(attempt, failure)| error_matches_query(attempt, failure, &self.search_query))
            .collect::<Vec<_>>();
        failures.sort_by(|(left, _), (right, _)| {
            right
                .finished_at
                .cmp(&left.finished_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| right.attempt_id.0.cmp(&left.attempt_id.0))
        });
        let total_failures = self
            .snapshot
            .attempts
            .iter()
            .filter(|attempt| attempt.failure.is_some())
            .count();
        let search_value = self.search_query.clone();
        let search_handler = self.on_action.clone();
        let collapse_handler = self.on_action.clone();
        let github_handler = self.on_action.clone();
        let first_failure_attempt_id = failures.first().map(|(attempt, _)| attempt.attempt_id);

        let header = h_flex()
            .w_full()
            .px_2()
            .py_1()
            .gap_1()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(
                div()
                    .id("sim-error-search-input")
                    .role(Role::SearchInput)
                    .tab_stop(true)
                    .aria_label("Filter structured execution errors")
                    .aria_value(if self.search_query.is_empty() {
                        "No error filter".to_owned()
                    } else {
                        self.search_query.clone()
                    })
                    .min_w_32()
                    .px_2()
                    .py_1()
                    .rounded_sm()
                    .border_1()
                    .border_color(cx.theme().colors().border)
                    .text_ui_sm(cx)
                    .text_color(if self.search_query.is_empty() {
                        cx.theme().colors().text_muted
                    } else {
                        cx.theme().colors().text
                    })
                    .on_key_down(move |event: &KeyDownEvent, window, cx| {
                        if let Some(value) = edit_search_query(&search_value, event) {
                            cx.stop_propagation();
                            search_handler(
                                ExecutionSurfaceAction::SetErrorSearch(value),
                                window,
                                cx,
                            );
                        }
                    })
                    .child(if self.search_query.is_empty() {
                        "Type to filter errors".to_owned()
                    } else {
                        self.search_query.clone()
                    }),
            )
            .child(
                div()
                    .debug_selector(|| "SIM-SURFACE-ERROR-COLLAPSE-ALL".into())
                    .child(
                        Button::new(
                            "sim-error-collapse-all",
                            if self.all_collapsed {
                                "Expand All"
                            } else {
                                "Collapse All"
                            },
                        )
                        .size(ButtonSize::Compact)
                        .aria_expanded(!self.all_collapsed)
                        .aria_label(if self.all_collapsed {
                            "Expand all structured execution error sections"
                        } else {
                            "Collapse all structured execution error sections"
                        })
                        .on_click(move |_, window, cx| {
                            collapse_handler(ExecutionSurfaceAction::ToggleAllErrors, window, cx);
                        }),
                    ),
            )
            .child(
                Button::new("sim-error-panel-github", "GitHub Issues")
                    .size(ButtonSize::Compact)
                    .aria_label("Open ComfyUI frontend GitHub issues in the system browser")
                    .on_click(move |_, window, cx| {
                        github_handler(
                            ExecutionSurfaceAction::OpenErrorGitHub(
                                first_failure_attempt_id.unwrap_or(AttemptId(uuid::Uuid::nil())),
                            ),
                            window,
                            cx,
                        );
                    }),
            );

        v_flex()
            .id("sim-tab-errors")
            .debug_selector(|| "SIM-SURFACE-TAB-ERRORS".into())
            .size_full()
            .min_h_0()
            .child(header)
            .when(total_failures == 0, |this| {
                this.child(
                    div()
                        .id("sim-error-group-list-empty")
                        .debug_selector(|| "SIM-SURFACE-ERROR-GROUP-LIST".into())
                        .role(Role::Status)
                        .aria_label("No structured execution errors")
                        .flex_1()
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text_muted)
                        .child("No execution errors"),
                )
            })
            .when(total_failures > 0 && failures.is_empty(), |this| {
                this.child(
                    div()
                        .id("sim-error-group-list-no-matches")
                        .debug_selector(|| "SIM-SURFACE-ERROR-GROUP-LIST".into())
                        .role(Role::Status)
                        .aria_label(format!(
                            "No structured execution errors match {}",
                            self.search_query
                        ))
                        .flex_1()
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text_muted)
                        .child("No errors match this filter"),
                )
            })
            .when(!failures.is_empty(), |this| {
                this.child(
                    v_flex()
                        .id("sim-error-group-list")
                        .debug_selector(|| "SIM-SURFACE-ERROR-GROUP-LIST".into())
                        .role(Role::List)
                        .aria_label(format!(
                            "{} filtered structured execution errors",
                            failures.len()
                        ))
                        .max_h(vh(0.70, window))
                        .flex_1()
                        .min_h_0()
                        .overflow_y_scroll()
                        .children(failures.into_iter().enumerate().map(
                            |(index, (attempt, failure))| {
                                let expanded =
                                    !self.collapsed_attempts.contains(&attempt.attempt_id);
                                render_error_card(
                                    attempt,
                                    failure,
                                    expanded,
                                    index == 0,
                                    self.runtime_controls_available
                                        && snapshot_allows_controls(&self.snapshot.status),
                                    self.on_action.clone(),
                                    cx,
                                )
                            },
                        )),
                )
            })
    }
}

fn render_error_card(
    attempt: &AttemptPresentation,
    failure: &ExecutionFailure,
    expanded: bool,
    is_first: bool,
    runtime_controls_available: bool,
    on_action: ExecutionSurfaceActionHandler,
    cx: &mut App,
) -> AnyElement {
    let attempt_id = attempt.attempt_id;
    let toggle_handler = on_action.clone();
    let toggle_keyboard_handler = on_action.clone();
    let copy_handler = on_action.clone();
    let locate_handler = on_action.clone();
    let help_handler = on_action.clone();
    let github_handler = on_action.clone();
    let retry_handler = on_action;
    let retry_allowed = runtime_controls_available && attempt.retry_eligibility().is_allowed();
    let details_trigger = div()
        .id(format!("sim-error-details-{}", attempt_id.0))
        .debug_selector(|| "SIM-SURFACE-ERROR-DETAILS-TRIGGER".into())
        .role(Role::Button)
        .tab_stop(true)
        .tab_index(0)
        .aria_expanded(expanded)
        .aria_label(if expanded {
            "Collapse structured execution error details"
        } else {
            "Expand structured execution error details"
        })
        .rounded_sm()
        .px_1()
        .text_ui_sm(cx)
        .cursor_pointer()
        .hover(|style| style.bg(cx.theme().colors().element_hover))
        .on_click(move |_, window, cx| {
            cx.stop_propagation();
            toggle_handler(
                ExecutionSurfaceAction::ToggleErrorDetails(attempt_id),
                window,
                cx,
            );
        })
        .on_key_down(move |event: &KeyDownEvent, window, cx| {
            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                cx.stop_propagation();
                toggle_keyboard_handler(
                    ExecutionSurfaceAction::ToggleErrorDetails(attempt_id),
                    window,
                    cx,
                );
            }
        })
        .child(if expanded { "Hide Details" } else { "Details" });
    let details_trigger = if is_first {
        div()
            .debug_selector(|| "SIM-SURFACE-ERROR-DETAILS-TRIGGER-FIRST".into())
            .child(details_trigger)
            .into_any_element()
    } else {
        details_trigger.into_any_element()
    };
    let copy_action = div()
        .debug_selector(|| "SIM-SURFACE-ERROR-COPY".into())
        .child(
            Button::new(format!("sim-error-copy-{}", attempt_id.0), "Copy")
                .size(ButtonSize::Compact)
                .aria_label("Copy structured execution error")
                .on_click(move |_, window, cx| {
                    copy_handler(ExecutionSurfaceAction::CopyError(attempt_id), window, cx);
                }),
        );
    let copy_action = if is_first {
        div()
            .debug_selector(|| "SIM-SURFACE-ERROR-COPY-FIRST".into())
            .child(copy_action)
            .into_any_element()
    } else {
        copy_action.into_any_element()
    };

    let card = v_flex()
        .id(format!("sim-error-card-section-{}", attempt_id.0))
        .debug_selector(|| "SIM-SURFACE-ERROR-CARD-SECTION".into())
        .role(Role::ListItem)
        .aria_label(format!(
            "Execution {:?} error {}: {}. Retryable: {}",
            failure.origin, failure.code, failure.message, failure.retryable
        ))
        .w_full()
        .px_2()
        .py_1()
        .gap_1()
        .border_b_1()
        .border_color(cx.theme().colors().border_variant)
        .child(
            h_flex()
                .id(format!("sim-error-node-card-{}", attempt_id.0))
                .debug_selector(|| "SIM-SURFACE-ERROR-NODE-CARD".into())
                .w_full()
                .justify_between()
                .gap_2()
                .child(
                    v_flex()
                        .min_w_0()
                        .child(div().text_ui_sm(cx).child(format!(
                            "{:?} · {}",
                            failure.origin, failure.code
                        )))
                        .child(
                            div()
                                .text_ui_sm(cx)
                                .text_color(cx.theme().colors().text_muted)
                                .child(failure.message.clone()),
                        ),
                )
                .child(
                    h_flex()
                        .gap_1()
                        .child(details_trigger)
                        .child(copy_action)
                        .child(
                            Button::new(
                                format!("sim-error-locate-{}", attempt_id.0),
                                "Locate",
                            )
                            .size(ButtonSize::Compact)
                            .disabled(failure.node_id.is_none())
                            .aria_label(if failure.node_id.is_some() {
                                "Select and reveal the affected graph node"
                            } else {
                                "Locate unavailable: structured error has no affected node"
                            })
                            .on_click(move |_, window, cx| {
                                locate_handler(
                                    ExecutionSurfaceAction::LocateError(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            Button::new(
                                format!("sim-error-help-{}", attempt_id.0),
                                "Help",
                            )
                            .size(ButtonSize::Compact)
                            .aria_label("Open execution troubleshooting help in the system browser")
                            .on_click(move |_, window, cx| {
                                help_handler(
                                    ExecutionSurfaceAction::OpenErrorHelp(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            div()
                                .debug_selector(|| "SIM-SURFACE-ERROR-GITHUB".into())
                                .child(
                                    Button::new(
                                        format!("sim-error-github-{}", attempt_id.0),
                                        "GitHub",
                                    )
                                    .size(ButtonSize::Compact)
                                    .aria_label("Find related issues on GitHub in the system browser")
                                    .on_click(move |_, window, cx| {
                                        github_handler(
                                            ExecutionSurfaceAction::OpenErrorGitHub(attempt_id),
                                            window,
                                            cx,
                                        );
                                    }),
                                ),
                        )
                        .child(
                            Button::new(
                                format!("sim-error-retry-{}", attempt_id.0),
                                "Retry",
                            )
                            .size(ButtonSize::Compact)
                            .disabled(!retry_allowed)
                            .aria_label(if retry_allowed {
                                "Retry this failed execution attempt"
                            } else if !runtime_controls_available {
                                "Retry unavailable: native runtime controller is not connected or snapshot is not ready"
                            } else {
                                "Retry unavailable: structured failure is not retryable"
                            })
                            .on_click(move |_, window, cx| {
                                retry_handler(
                                    ExecutionSurfaceAction::RetryAttempt(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        ),
                ),
        )
        .when(expanded, |this| {
            this.child(
                v_flex()
                    .id(format!("sim-error-runtime-details-{}", attempt_id.0))
                    .debug_selector(|| "SIM-SURFACE-ERROR-DETAILS-CONTENT".into())
                    .role(Role::Group)
                    .aria_label(format!(
                        "Structured runtime details for execution attempt {}",
                        attempt_id.0
                    ))
                    .gap_0p5()
                    .child(detail_line("Attempt", attempt_id.0.to_string(), cx))
                    .child(detail_line("Origin", format!("{:?}", failure.origin), cx))
                    .child(detail_line("Code", failure.code.clone(), cx))
                    .child(detail_line("Retryable", failure.retryable.to_string(), cx))
                    .when_some(failure.node_id.as_ref(), |this, node_id| {
                        this.child(detail_line("Node", node_id.0.clone(), cx))
                    })
                    .children(
                        failure
                            .details
                            .iter()
                            .map(|(key, value)| detail_line("Detail", format!("{key}: {value}"), cx)),
                    ),
            )
        });
    if is_first {
        div()
            .debug_selector(|| "SIM-SURFACE-ERROR-CARD-SECTION-FIRST".into())
            .w_full()
            .child(card)
            .into_any_element()
    } else {
        card.into_any_element()
    }
}

#[derive(IntoElement)]
pub(crate) struct ErrorOverlaySurface {
    pub attempt_id: AttemptId,
    pub failure_count: usize,
    pub view_focus_handle: FocusHandle,
    pub dismiss_focus_handle: FocusHandle,
    pub on_action: ExecutionSurfaceActionHandler,
}

impl RenderOnce for ErrorOverlaySurface {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let view_handler = self.on_action.clone();
        let view_keyboard_handler = self.on_action.clone();
        let dismiss_handler = self.on_action.clone();
        let dismiss_keyboard_handler = self.on_action;
        let attempt_id = self.attempt_id;

        div().when(self.failure_count > 0, |this| {
            this.child(
                h_flex()
                    .id("sim-error-overlay")
                    .debug_selector(|| "SIM-SURFACE-ERROR-OVERLAY".into())
                    .role(Role::Alert)
                    .aria_label(format!(
                        "{} structured execution errors are available",
                        self.failure_count
                    ))
                    .w_full()
                    .px_2()
                    .py_1()
                    .gap_1()
                    .justify_between()
                    .bg(cx.theme().status().error_background.opacity(0.18))
                    .child(
                        div()
                            .text_ui_sm(cx)
                            .child(format!("{} execution errors", self.failure_count)),
                    )
                    .child(
                        h_flex()
                            .gap_1()
                            .child(
                                div()
                                    .id("sim-error-overlay-view")
                                    .debug_selector(|| "SIM-SURFACE-ERROR-OVERLAY-VIEW".into())
                                    .role(Role::Button)
                                    .tab_stop(true)
                                    .tab_index(0)
                                    .track_focus(&self.view_focus_handle)
                                    .aria_label("Open the structured execution errors view")
                                    .rounded_sm()
                                    .px_1()
                                    .text_ui_sm(cx)
                                    .cursor_pointer()
                                    .hover(|style| style.bg(cx.theme().colors().element_hover))
                                    .on_click(move |_, window, cx| {
                                        cx.stop_propagation();
                                        view_handler(
                                            ExecutionSurfaceAction::ViewErrors,
                                            window,
                                            cx,
                                        );
                                    })
                                    .on_key_down(move |event: &KeyDownEvent, window, cx| {
                                        if matches!(event.keystroke.key.as_str(), "enter" | "space")
                                        {
                                            cx.stop_propagation();
                                            view_keyboard_handler(
                                                ExecutionSurfaceAction::ViewErrors,
                                                window,
                                                cx,
                                            );
                                        }
                                    })
                                    .child("View Errors"),
                            )
                            .child(
                                div()
                                    .id("sim-error-overlay-dismiss")
                                    .debug_selector(|| "SIM-SURFACE-ERROR-OVERLAY-DISMISS".into())
                                    .role(Role::Button)
                                    .tab_stop(true)
                                    .tab_index(0)
                                    .track_focus(&self.dismiss_focus_handle)
                                    .aria_label(
                                        "Dismiss this error overlay; errors remain in history",
                                    )
                                    .rounded_sm()
                                    .px_1()
                                    .text_ui_sm(cx)
                                    .cursor_pointer()
                                    .hover(|style| style.bg(cx.theme().colors().element_hover))
                                    .on_click(move |_, window, cx| {
                                        cx.stop_propagation();
                                        dismiss_handler(
                                            ExecutionSurfaceAction::DismissErrorOverlay(attempt_id),
                                            window,
                                            cx,
                                        );
                                    })
                                    .on_key_down(move |event: &KeyDownEvent, window, cx| {
                                        if matches!(event.keystroke.key.as_str(), "enter" | "space")
                                        {
                                            cx.stop_propagation();
                                            dismiss_keyboard_handler(
                                                ExecutionSurfaceAction::DismissErrorOverlay(
                                                    attempt_id,
                                                ),
                                                window,
                                                cx,
                                            );
                                        }
                                    })
                                    .child("Dismiss"),
                            ),
                    ),
            )
        })
    }
}

pub(crate) fn edit_search_query(current: &str, event: &KeyDownEvent) -> Option<String> {
    if event.keystroke.modifiers.control
        || event.keystroke.modifiers.alt
        || event.keystroke.modifiers.platform
    {
        return None;
    }
    let mut value = current.to_owned();
    if event.keystroke.key == "backspace" {
        value.pop()?;
    } else if event.keystroke.key == "escape" {
        if value.is_empty() {
            return None;
        }
        value.clear();
    } else {
        let text = event.keystroke.key_char.as_deref()?;
        if text.chars().any(char::is_control) {
            return None;
        }
        value.push_str(text);
    }
    Some(value)
}

pub(crate) fn error_matches_query(
    attempt: &AttemptPresentation,
    failure: &ExecutionFailure,
    query: &str,
) -> bool {
    let query = query.trim().to_lowercase();
    if query.is_empty() {
        return true;
    }
    let details = failure
        .details
        .iter()
        .map(|(key, value)| format!("{key} {value}"))
        .collect::<Vec<_>>()
        .join(" ");
    [
        attempt.attempt_id.0.to_string(),
        attempt.prompt_id.0.to_string(),
        format!("{:?}", failure.origin),
        failure.code.clone(),
        failure.message.clone(),
        failure
            .node_id
            .as_ref()
            .map(|node_id| node_id.0.clone())
            .unwrap_or_default(),
        details,
    ]
    .into_iter()
    .any(|value| value.to_lowercase().contains(&query))
}

pub(crate) fn attempt_matches_query(attempt: &AttemptPresentation, query: &str) -> bool {
    let query = query.trim().to_lowercase();
    if query.is_empty() {
        return true;
    }
    [
        attempt.attempt_id.0.to_string(),
        attempt.prompt_id.0.to_string(),
        state_label(attempt.state).to_owned(),
        attempt
            .failure
            .as_ref()
            .map(|failure| format!("{:?} {} {}", failure.origin, failure.code, failure.message))
            .unwrap_or_default(),
    ]
    .into_iter()
    .any(|value| value.to_lowercase().contains(&query))
}

pub(crate) fn snapshot_allows_controls(status: &ExecutionSnapshotStatus) -> bool {
    matches!(
        status,
        ExecutionSnapshotStatus::Ready | ExecutionSnapshotStatus::Partial { .. }
    )
}

pub(crate) fn state_label(state: AttemptState) -> &'static str {
    match state {
        AttemptState::Queued => "queued",
        AttemptState::Running => "running",
        AttemptState::Cancelling => "cancelling",
        AttemptState::Succeeded => "succeeded",
        AttemptState::Failed => "failed",
        AttemptState::Cancelled => "cancelled",
        AttemptState::Interrupted => "interrupted",
    }
}

fn short_attempt_label(attempt_id: AttemptId) -> String {
    attempt_id.0.to_string().chars().take(8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{Keystroke, Modifiers};
    use uuid::Uuid;

    #[test]
    fn search_editor_appends_deletes_and_clears_without_shortcuts() {
        let typed = KeyDownEvent {
            keystroke: Keystroke {
                key: "x".to_owned(),
                key_char: Some("x".to_owned()),
                modifiers: Modifiers::default(),
            },
            is_held: false,
            prefer_character_input: false,
        };
        assert_eq!(edit_search_query("job", &typed).as_deref(), Some("jobx"));

        let backspace = KeyDownEvent {
            keystroke: Keystroke {
                key: "backspace".to_owned(),
                key_char: None,
                modifiers: Modifiers::default(),
            },
            is_held: false,
            prefer_character_input: false,
        };
        assert_eq!(edit_search_query("job", &backspace).as_deref(), Some("jo"));

        let escape = KeyDownEvent {
            keystroke: Keystroke {
                key: "escape".to_owned(),
                key_char: None,
                modifiers: Modifiers::default(),
            },
            is_held: false,
            prefer_character_input: false,
        };
        assert_eq!(edit_search_query("job", &escape).as_deref(), Some(""));
    }

    #[test]
    fn queue_lifecycle_correlates_requests_pluralizes_and_preserves_fifo() {
        let primary = RequestId(Uuid::from_u128(1));
        let mismatch = RequestId(Uuid::from_u128(999));
        let primary_attempt = AttemptId(Uuid::from_u128(10));
        let mismatch_attempt = AttemptId(Uuid::from_u128(11));
        let mut state = ExecutionSurfaceRuntimeState::default();

        state.observe_prompt_queueing(primary, 1);
        let pending = state
            .current_notification()
            .expect("queueing notification should become active");
        let pending_identity = pending.identity;
        assert_eq!(pending.kind, ExecutionNotificationKind::Queueing);
        assert_eq!(pending.request_id, Some(primary));

        state.observe_prompt_queued(primary, 1, primary_attempt);
        let upgraded = state
            .current_notification()
            .expect("matching queued notification should remain active");
        assert_eq!(upgraded.identity, pending_identity);
        assert_eq!(upgraded.kind, ExecutionNotificationKind::Queued);
        assert_eq!(upgraded.message, "1 job added to queue");

        state.observe_prompt_queued(mismatch, 3, mismatch_attempt);
        assert_eq!(state.notification_count_for_test(), 2);
        assert_eq!(
            state
                .current_notification()
                .map(|notification| notification.request_id),
            Some(Some(primary))
        );

        assert!(state.dismiss_notification(pending_identity));
        let next = state
            .current_notification()
            .expect("mismatched request should remain queued in FIFO order");
        assert_eq!(next.request_id, Some(mismatch));
        assert_eq!(next.batch_count, 3);
        assert_eq!(next.message, "3 jobs added to queue");
    }

    #[test]
    fn rejected_queue_acknowledgement_upgrades_correlated_banner_to_failure() {
        let request_id = RequestId(Uuid::from_u128(2));
        let mut state = ExecutionSurfaceRuntimeState::default();
        state.observe_prompt_queueing(request_id, 2);
        let queueing_identity = state
            .current_notification()
            .expect("queueing notification should become active")
            .identity;

        state.observe_prompt_queue_failed(
            request_id,
            2,
            &ExecutionFailure::new("runtime_unavailable", "native controller is disconnected"),
        );

        let failure = state
            .current_notification()
            .expect("rejected queue notification should remain visible");
        assert_eq!(failure.identity, queueing_identity);
        assert_eq!(failure.kind, ExecutionNotificationKind::Failure);
        assert_eq!(failure.request_id, Some(request_id));
        assert_eq!(failure.batch_count, 2);
        assert_eq!(failure.title, "2 jobs could not be queued");
        assert_eq!(
            failure.message,
            "runtime_unavailable: native controller is disconnected"
        );
    }
}
