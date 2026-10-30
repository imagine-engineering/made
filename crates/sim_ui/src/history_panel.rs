use gpui::{AnyElement, App, IntoElement, RenderOnce, Role, SharedString, Window};
use sim_runtime::{
    AttemptId, AttemptPresentation, AttemptState, ExecutionControlCommandKind, ExecutionDataSource,
    ExecutionSnapshot, ExecutionSnapshotStatus,
};
use std::{collections::HashSet, sync::Arc};
use ui::{Button, ButtonCommon, ButtonSize, Disableable, prelude::*};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryPanelAction {
    SelectAttempt(AttemptId),
    ToggleErrorDetails(AttemptId),
    CopyError(AttemptId),
    NavigateToError(AttemptId),
    Retry(AttemptId),
    Remove(AttemptId),
    ClearHistory,
}

pub type HistoryPanelActionHandler =
    Arc<dyn Fn(HistoryPanelAction, &mut Window, &mut App) + 'static>;

#[derive(IntoElement)]
pub struct HistoryPanelContent {
    snapshot: ExecutionSnapshot,
    selected_attempt_id: Option<AttemptId>,
    expanded_error_attempts: HashSet<AttemptId>,
    runtime_controls_available: bool,
    on_action: HistoryPanelActionHandler,
}

impl HistoryPanelContent {
    pub fn new(
        snapshot: ExecutionSnapshot,
        selected_attempt_id: Option<AttemptId>,
        on_action: HistoryPanelActionHandler,
    ) -> Self {
        Self {
            snapshot,
            selected_attempt_id,
            expanded_error_attempts: HashSet::new(),
            runtime_controls_available: true,
            on_action,
        }
    }

    pub fn with_expanded_error_attempts(
        mut self,
        expanded_error_attempts: impl IntoIterator<Item = AttemptId>,
    ) -> Self {
        self.expanded_error_attempts = expanded_error_attempts.into_iter().collect();
        self
    }

    pub fn with_runtime_controls_available(mut self, available: bool) -> Self {
        self.runtime_controls_available = available;
        self
    }
}

impl RenderOnce for HistoryPanelContent {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let pending_attempts = pending_attempts(&self.snapshot);
        let mut attempts = self
            .snapshot
            .attempts
            .iter()
            .filter(|attempt| attempt.state.is_terminal())
            .cloned()
            .collect::<Vec<_>>();
        attempts.sort_by(|left, right| {
            right
                .finished_at
                .cmp(&left.finished_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| right.attempt_id.0.cmp(&left.attempt_id.0))
        });
        let pending_clear = self.snapshot.pending_commands.iter().any(|pending| {
            matches!(
                &pending.command.kind,
                ExecutionControlCommandKind::ClearHistory
            )
        });
        let clear_disabled = attempts.is_empty() || pending_clear;
        let clear_handler = self.on_action.clone();
        let profile_label = self
            .snapshot
            .profile_id
            .0
            .to_string()
            .chars()
            .take(8)
            .collect::<String>();

        v_flex()
            .id("sim-execution-history")
            .size_full()
            .min_h_0()
            .bg(cx.theme().colors().panel_background)
            .child(
                h_flex()
                    .w_full()
                    .px_2()
                    .py_1()
                    .gap_2()
                    .justify_between()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .child(div().text_ui_sm(cx).child(format!(
                        "{} execution history · profile {profile_label}",
                        source_name(self.snapshot.source)
                    )))
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                div()
                                    .id("sim-history-summary")
                                    .role(Role::Status)
                                    .aria_label(format!(
                                        "{} terminal attempts, {} command requests pending",
                                        attempts.len(),
                                        self.snapshot.pending_commands.len()
                                    ))
                                    .text_ui_sm(cx)
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(format!("{} attempts", attempts.len())),
                            )
                            .child(
                                Button::new("sim-clear-execution-history", "Clear History")
                                    .size(ButtonSize::Compact)
                                    .disabled(clear_disabled)
                                    .aria_label(if pending_clear {
                                        "Clear execution history (request pending)"
                                    } else if attempts.is_empty() {
                                        "Clear execution history (history is empty)"
                                    } else {
                                        "Clear all terminal attempts from execution history"
                                    })
                                    .on_click(move |_, window, cx| {
                                        clear_handler(HistoryPanelAction::ClearHistory, window, cx);
                                    }),
                            ),
                    ),
            )
            .child(snapshot_status(&self.snapshot.status, cx))
            .when(
                attempts.is_empty() && snapshot_can_show_content(&self.snapshot.status),
                |this| {
                    this.child(
                        div()
                            .id("sim-history-empty")
                            .role(Role::Status)
                            .aria_label("Execution history is empty")
                            .flex_1()
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_ui_sm(cx)
                            .text_color(cx.theme().colors().text_muted)
                            .child("No completed or interrupted attempts"),
                    )
                },
            )
            .when(
                !attempts.is_empty() && snapshot_can_show_content(&self.snapshot.status),
                |this| {
                    this.child(
                        v_flex()
                            .id("sim-history-list")
                            .role(Role::List)
                            .aria_label("Execution attempt history")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .children(attempts.iter().map(|attempt| {
                                render_history_row(
                                    attempt,
                                    self.selected_attempt_id == Some(attempt.attempt_id),
                                    pending_attempts.contains(&attempt.attempt_id),
                                    self.expanded_error_attempts.contains(&attempt.attempt_id),
                                    self.runtime_controls_available,
                                    self.on_action.clone(),
                                    cx,
                                )
                            })),
                    )
                },
            )
    }
}

fn render_history_row(
    attempt: &AttemptPresentation,
    selected: bool,
    command_pending: bool,
    error_expanded: bool,
    runtime_controls_available: bool,
    on_action: HistoryPanelActionHandler,
    cx: &mut App,
) -> AnyElement {
    let attempt_id = attempt.attempt_id;
    let attempt_label = attempt_label(attempt_id);
    let state_label = projected_state_name(attempt);
    let select_handler = on_action.clone();
    let retry_handler = on_action.clone();
    let error_handler = on_action.clone();
    let remove_handler = on_action;
    let retry = attempt.retry_eligibility();
    let retry_allowed = runtime_controls_available && retry.is_allowed();
    let retry_reason = if runtime_controls_available {
        eligibility_reason(&retry)
    } else {
        "the native runtime controller is not connected"
    };
    let removal = attempt.removal_eligibility();
    let remove_allowed = removal.is_allowed();
    let remove_reason = eligibility_reason(&removal);
    let output_count = attempt.outputs.len();
    let pending_suffix = command_pending
        .then_some("; command request pending")
        .unwrap_or("");

    v_flex()
        .id(format!("sim-history-row-{}", attempt_id.0))
        .role(Role::ListItem)
        .aria_selected(selected)
        .aria_label(format!(
            "Attempt {attempt_label}, {state_label}, {output_count} outputs, {} canonical events{pending_suffix}",
            attempt.canonical_event_count
        ))
        .w_full()
        .px_2()
        .py_1()
        .gap_1()
        .border_b_1()
        .border_color(cx.theme().colors().border_variant)
        .when(selected, |this| this.bg(cx.theme().colors().element_selected))
        .child(
            h_flex()
                .w_full()
                .justify_between()
                .gap_2()
                .child(
                    v_flex()
                        .min_w_0()
                        .child(
                            h_flex()
                                .gap_1()
                                .child(
                                    div()
                                        .text_ui_sm(cx)
                                        .child(format!("Attempt {attempt_label}")),
                                )
                                .child(
                                    div()
                                        .id(format!("sim-history-state-{}", attempt_id.0))
                                        .role(Role::Status)
                                        .aria_label(format!(
                                            "Attempt {attempt_label} {state_label}"
                                        ))
                                        .text_ui_sm(cx)
                                        .text_color(state_color(attempt.state, cx))
                                        .child(state_label),
                                ),
                        )
                        .child(
                            div()
                                .text_ui_sm(cx)
                                .text_color(cx.theme().colors().text_muted)
                                .child(format!(
                                    "{} outputs · {} events · {}",
                                    output_count,
                                    attempt.canonical_event_count,
                                    finished_label(attempt)
                                )),
                        )
                        .when_some(attempt.retry_of, |this, prior_attempt_id| {
                            this.child(
                                div()
                                    .text_ui_sm(cx)
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(format!(
                                        "Retry of attempt {}",
                                        self::attempt_label(prior_attempt_id)
                                    )),
                            )
                        }),
                )
                .child(
                    h_flex()
                        .gap_1()
                        .child(
                            Button::new(
                                format!("sim-inspect-history-{}", attempt_id.0),
                                "Inspect",
                            )
                            .size(ButtonSize::Compact)
                            .aria_label(format!("Inspect attempt {attempt_label}"))
                            .on_click(move |_, window, cx| {
                                select_handler(
                                    HistoryPanelAction::SelectAttempt(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            Button::new(
                                format!("sim-retry-history-{}", attempt_id.0),
                                if command_pending { "Pending" } else { "Retry" },
                            )
                            .size(ButtonSize::Compact)
                            .disabled(!retry_allowed || command_pending)
                            .aria_label(if command_pending {
                                format!("Retry attempt {attempt_label} (request pending)")
                            } else if retry_allowed {
                                format!("Retry attempt {attempt_label} from its original prompt")
                            } else {
                                format!("Retry attempt {attempt_label} unavailable: {retry_reason}")
                            })
                            .on_click(move |_, window, cx| {
                                retry_handler(
                                    HistoryPanelAction::Retry(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            Button::new(
                                format!("sim-remove-history-{}", attempt_id.0),
                                "Remove",
                            )
                            .size(ButtonSize::Compact)
                            .disabled(!remove_allowed || command_pending)
                            .aria_label(if command_pending {
                                format!("Remove attempt {attempt_label} (request pending)")
                            } else if remove_allowed {
                                format!("Remove attempt {attempt_label} from execution history")
                            } else {
                                format!("Remove attempt {attempt_label} unavailable: {remove_reason}")
                            })
                            .on_click(move |_, window, cx| {
                                remove_handler(
                                    HistoryPanelAction::Remove(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        ),
                ),
        )
        .when_some(attempt.interrupted_reason.as_ref(), |this, reason| {
            this.child(
                div()
                    .id(format!("sim-history-interrupted-reason-{}", attempt_id.0))
                    .role(Role::Status)
                    .aria_label(format!("Attempt {attempt_label} interrupted: {reason}"))
                    .text_ui_sm(cx)
                    .text_color(cx.theme().colors().text_muted)
                    .child(format!("Interrupted · {reason}")),
            )
        })
        .when_some(attempt.failure.as_ref(), |this, failure| {
            this.child(render_failure(
                attempt_id,
                failure,
                error_expanded,
                error_handler,
                cx,
            ))
        })
        .into_any_element()
}

fn render_failure(
    attempt_id: AttemptId,
    failure: &sim_runtime::ExecutionFailure,
    expanded: bool,
    on_action: HistoryPanelActionHandler,
    cx: &mut App,
) -> AnyElement {
    let detail_summary = failure
        .details
        .iter()
        .map(|(key, value)| format!("{key}: {value}"))
        .collect::<Vec<_>>()
        .join("; ");
    let aria_details = if detail_summary.is_empty() {
        String::new()
    } else {
        format!(". Details: {detail_summary}")
    };
    let toggle_handler = on_action.clone();
    let copy_handler = on_action.clone();
    let navigate_handler = on_action;

    v_flex()
        .id(format!("sim-history-error-{}", attempt_id.0))
        .role(Role::Alert)
        .aria_label(format!(
            "Execution {:?} error {}: {}{}. Retryable: {}",
            failure.origin, failure.code, failure.message, aria_details, failure.retryable
        ))
        .w_full()
        .p_1()
        .gap_0p5()
        .rounded_sm()
        .bg(cx.theme().status().error_background.opacity(0.16))
        .child(
            h_flex()
                .w_full()
                .justify_between()
                .gap_2()
                .child(div().min_w_0().text_ui_sm(cx).child(format!(
                    "{:?} · {} · {}",
                    failure.origin, failure.code, failure.message
                )))
                .child(
                    h_flex()
                        .gap_1()
                        .child(
                            Button::new(
                                format!("sim-history-error-details-{}", attempt_id.0),
                                if expanded { "Hide Details" } else { "Details" },
                            )
                            .size(ButtonSize::Compact)
                            .aria_expanded(expanded)
                            .aria_label(if expanded {
                                "Hide structured execution error details"
                            } else {
                                "Show structured execution error details"
                            })
                            .on_click(move |_, window, cx| {
                                toggle_handler(
                                    HistoryPanelAction::ToggleErrorDetails(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            Button::new(format!("sim-history-copy-error-{}", attempt_id.0), "Copy")
                                .size(ButtonSize::Compact)
                                .aria_label("Copy structured execution error")
                                .on_click(move |_, window, cx| {
                                    copy_handler(
                                        HistoryPanelAction::CopyError(attempt_id),
                                        window,
                                        cx,
                                    );
                                }),
                        )
                        .child(
                            Button::new(
                                format!("sim-history-locate-error-{}", attempt_id.0),
                                "Locate",
                            )
                            .size(ButtonSize::Compact)
                            .disabled(failure.node_id.is_none())
                            .aria_label(if failure.node_id.is_some() {
                                "Select and reveal the graph node for this execution error"
                            } else {
                                "Locate execution error unavailable: no affected graph node"
                            })
                            .on_click(move |_, window, cx| {
                                navigate_handler(
                                    HistoryPanelAction::NavigateToError(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        ),
                ),
        )
        .when(expanded, |this| {
            this.when_some(failure.node_id.as_ref(), |this, node_id| {
                this.child(
                    div()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text_muted)
                        .child(format!("Node {}", node_id.0)),
                )
            })
            .children(failure.details.iter().map(|(key, value)| {
                div()
                    .text_ui_sm(cx)
                    .text_color(cx.theme().colors().text_muted)
                    .child(format!("{key}: {value}"))
            }))
        })
        .into_any_element()
}

fn pending_attempts(snapshot: &ExecutionSnapshot) -> HashSet<AttemptId> {
    snapshot
        .pending_commands
        .iter()
        .filter_map(|pending| match &pending.command.kind {
            ExecutionControlCommandKind::Reorder { attempt_id, .. }
            | ExecutionControlCommandKind::Cancel { attempt_id, .. }
            | ExecutionControlCommandKind::Interrupt { attempt_id, .. }
            | ExecutionControlCommandKind::Retry { attempt_id, .. }
            | ExecutionControlCommandKind::RemoveHistory { attempt_id } => Some(*attempt_id),
            ExecutionControlCommandKind::Queue { .. }
            | ExecutionControlCommandKind::ClearPending { .. }
            | ExecutionControlCommandKind::ClearHistory => None,
        })
        .collect()
}

fn snapshot_status(status: &ExecutionSnapshotStatus, cx: &mut App) -> AnyElement {
    match status {
        ExecutionSnapshotStatus::Loading => div()
            .id("sim-history-loading")
            .role(Role::Status)
            .aria_label("Loading native execution history")
            .w_full()
            .p_2()
            .text_ui_sm(cx)
            .text_color(cx.theme().colors().text_muted)
            .child("Loading execution history…")
            .into_any_element(),
        ExecutionSnapshotStatus::Ready => div().into_any_element(),
        ExecutionSnapshotStatus::Stale {
            source_revision,
            failure,
        } => {
            let revision = source_revision.map_or_else(
                || "unknown source revision".to_owned(),
                |revision| format!("source revision {revision}"),
            );
            div()
                .id("sim-history-stale")
                .role(Role::Alert)
                .aria_label(format!(
                    "Execution history is stale at {revision}: {}: {}. The last known history remains available for inspection",
                    failure.code, failure.message
                ))
                .w_full()
                .p_2()
                .bg(cx.theme().status().warning_background.opacity(0.2))
                .text_ui_sm(cx)
                .child(format!(
                    "Stale history data · {revision} · {}: {} · showing the last known history",
                    failure.code, failure.message
                ))
                .into_any_element()
        }
        ExecutionSnapshotStatus::Partial { failure } => div()
            .id("sim-history-partial")
            .role(Role::Alert)
            .aria_label(format!(
                "Execution history is partially available: {}: {}",
                failure.code, failure.message
            ))
            .w_full()
            .p_2()
            .bg(cx.theme().status().warning_background.opacity(0.2))
            .text_ui_sm(cx)
            .child(format!(
                "Partial history data · {}: {}",
                failure.code, failure.message
            ))
            .into_any_element(),
        ExecutionSnapshotStatus::Unavailable { failure } => div()
            .id("sim-history-unavailable")
            .role(Role::Alert)
            .aria_label(format!(
                "Execution history is unavailable: {}: {}",
                failure.code, failure.message
            ))
            .w_full()
            .p_2()
            .bg(cx.theme().status().error_background.opacity(0.2))
            .text_ui_sm(cx)
            .child(format!(
                "History unavailable · {}: {}",
                failure.code, failure.message
            ))
            .into_any_element(),
    }
}

fn snapshot_can_show_content(status: &ExecutionSnapshotStatus) -> bool {
    matches!(
        status,
        ExecutionSnapshotStatus::Ready
            | ExecutionSnapshotStatus::Stale { .. }
            | ExecutionSnapshotStatus::Partial { .. }
    )
}

fn eligibility_reason(eligibility: &sim_runtime::OperationEligibility) -> &str {
    match eligibility {
        sim_runtime::OperationEligibility::Allowed => "available",
        sim_runtime::OperationEligibility::Unavailable { reason } => reason,
    }
}

fn source_name(source: ExecutionDataSource) -> &'static str {
    match source {
        ExecutionDataSource::Live => "Live",
        ExecutionDataSource::Persisted => "Persisted",
        ExecutionDataSource::Recovery => "Recovered",
    }
}

fn state_name(state: AttemptState) -> &'static str {
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

fn projected_state_name(attempt: &AttemptPresentation) -> String {
    match attempt.source_projection.as_ref() {
        None => state_name(attempt.state).to_owned(),
        Some(sim_runtime::AttemptSourceProjection::Provider { provider_id, state }) => {
            format!("provider {provider_id}: {}", provider_state_name(state))
        }
        Some(sim_runtime::AttemptSourceProjection::Unknown {
            source_id,
            raw_state,
        }) => source_id.as_ref().map_or_else(
            || format!("unknown source state: {raw_state}"),
            |source_id| format!("unknown source {source_id}: {raw_state}"),
        ),
    }
}

fn provider_state_name(state: &sim_runtime::ProviderAttemptState) -> &str {
    match state {
        sim_runtime::ProviderAttemptState::Queued => "queued",
        sim_runtime::ProviderAttemptState::Running => "running",
        sim_runtime::ProviderAttemptState::Cancelling => "cancelling",
        sim_runtime::ProviderAttemptState::Succeeded => "succeeded",
        sim_runtime::ProviderAttemptState::Failed => "failed",
        sim_runtime::ProviderAttemptState::Cancelled => "cancelled",
        sim_runtime::ProviderAttemptState::Interrupted => "interrupted",
        sim_runtime::ProviderAttemptState::Unknown { raw_state } => raw_state,
    }
}

fn state_color(state: AttemptState, cx: &App) -> gpui::Hsla {
    match state {
        AttemptState::Succeeded => cx.theme().status().success,
        AttemptState::Failed => cx.theme().status().error,
        AttemptState::Interrupted | AttemptState::Cancelled => cx.theme().status().warning,
        AttemptState::Queued | AttemptState::Running | AttemptState::Cancelling => {
            cx.theme().colors().text_muted
        }
    }
}

fn finished_label(attempt: &AttemptPresentation) -> String {
    attempt.finished_at.as_ref().map_or_else(
        || "finish time unavailable".to_owned(),
        |finished_at| finished_at.to_rfc3339(),
    )
}

fn attempt_label(attempt_id: AttemptId) -> SharedString {
    attempt_id
        .0
        .to_string()
        .chars()
        .take(8)
        .collect::<String>()
        .into()
}
