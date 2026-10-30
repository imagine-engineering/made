use gpui::{AnyElement, App, IntoElement, RenderOnce, Role, SharedString, Window};
use sim_runtime::{
    AttemptId, AttemptPresentation, AttemptState, ExecutionControlCommandKind, ExecutionDataSource,
    ExecutionSnapshot, ExecutionSnapshotStatus, QueuedPrompt,
};
use std::{collections::HashSet, sync::Arc};
use ui::{Button, ButtonCommon, ButtonSize, Disableable, prelude::*};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueuePanelAction {
    SelectAttempt(AttemptId),
    MoveEarlier(AttemptId),
    MoveLater(AttemptId),
    Cancel(AttemptId),
    Interrupt(AttemptId),
    ClearPending,
}

pub type QueuePanelActionHandler = Arc<dyn Fn(QueuePanelAction, &mut Window, &mut App) + 'static>;

#[derive(IntoElement)]
pub struct QueuePanelContent {
    snapshot: ExecutionSnapshot,
    selected_attempt_id: Option<AttemptId>,
    show_progress: bool,
    runtime_controls_available: bool,
    on_action: QueuePanelActionHandler,
}

impl QueuePanelContent {
    pub fn new(
        snapshot: ExecutionSnapshot,
        selected_attempt_id: Option<AttemptId>,
        on_action: QueuePanelActionHandler,
    ) -> Self {
        Self {
            snapshot,
            selected_attempt_id,
            show_progress: true,
            runtime_controls_available: true,
            on_action,
        }
    }

    pub fn with_show_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    pub fn with_runtime_controls_available(mut self, available: bool) -> Self {
        self.runtime_controls_available = available;
        self
    }
}

impl RenderOnce for QueuePanelContent {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let pending_attempts = pending_attempts(&self.snapshot);
        let pending_clear = self.snapshot.pending_commands.iter().any(|pending| {
            matches!(
                &pending.command.kind,
                ExecutionControlCommandKind::ClearPending { .. }
            )
        });
        let active_attempts = self
            .snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                !attempt.state.is_terminal()
                    && !self
                        .snapshot
                        .queue
                        .iter()
                        .any(|queued| queued.attempt_id == attempt.attempt_id)
            })
            .cloned()
            .collect::<Vec<_>>();
        let profile_label = self
            .snapshot
            .profile_id
            .0
            .to_string()
            .chars()
            .take(8)
            .collect::<String>();
        let item_count = self.snapshot.queue.len() + active_attempts.len();
        let clear_disabled =
            self.snapshot.queue.is_empty() || pending_clear || !self.runtime_controls_available;
        let clear_label = if pending_clear {
            "Clear pending queue (request pending)"
        } else if self.snapshot.queue.is_empty() {
            "Clear pending queue (queue is empty)"
        } else if !self.runtime_controls_available {
            "Clear pending queue unavailable: native runtime controller is not connected"
        } else {
            "Clear pending queue"
        };
        let clear_handler = self.on_action.clone();

        v_flex()
            .id("sim-execution-queue")
            .size_full()
            .min_h_0()
            .bg(cx.theme().colors().panel_background)
            .child(
                h_flex()
                    .id("sim-execution-dock-queue-overlay-header")
                    .debug_selector(|| "SIM-SURFACE-QUEUE-OVERLAY-HEADER".into())
                    .w_full()
                    .px_2()
                    .py_1()
                    .gap_2()
                    .justify_between()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .child(
                        h_flex()
                            .id("sim-execution-dock-queue-progress-overlay")
                            .debug_selector(|| "SIM-SURFACE-QUEUE-PROGRESS-OVERLAY".into())
                            .gap_2()
                            .child(div().text_ui_sm(cx).child(format!(
                                "{} execution queue · profile {profile_label}",
                                source_name(self.snapshot.source)
                            )))
                            .child(
                                div()
                                    .id("sim-execution-dock-queue-overlay-active")
                                    .debug_selector(|| "SIM-SURFACE-QUEUE-OVERLAY-ACTIVE".into())
                                    .role(Role::Status)
                                    .aria_label(format!(
                                        "{} queued, {} active, {} command requests pending",
                                        self.snapshot.queue.len(),
                                        active_attempts.len(),
                                        self.snapshot.pending_commands.len()
                                    ))
                                    .text_ui_sm(cx)
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(format!(
                                        "{} queued · {} active",
                                        self.snapshot.queue.len(),
                                        active_attempts.len()
                                    )),
                            ),
                    )
                    .child(
                        Button::new("sim-clear-pending-queue", "Clear Pending")
                            .size(ButtonSize::Compact)
                            .disabled(clear_disabled)
                            .aria_label(clear_label)
                            .on_click(move |_, window, cx| {
                                clear_handler(QueuePanelAction::ClearPending, window, cx);
                            }),
                    ),
            )
            .child(snapshot_status(&self.snapshot.status, cx))
            .when(
                item_count == 0 && snapshot_can_show_content(&self.snapshot.status),
                |this| {
                    this.child(
                        div()
                            .id("sim-queue-empty")
                            .role(Role::Status)
                            .aria_label("Execution queue is empty")
                            .flex_1()
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_ui_sm(cx)
                            .text_color(cx.theme().colors().text_muted)
                            .child("No queued or running attempts"),
                    )
                },
            )
            .when(
                item_count > 0 && snapshot_can_show_content(&self.snapshot.status),
                |this| {
                    this.child(
                        v_flex()
                            .id("sim-queue-list")
                            .role(Role::List)
                            .aria_label("Queued and active execution attempts")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .child(
                                div()
                                    .id("sim-execution-dock-queue-overlay-expanded")
                                    .debug_selector(|| "SIM-SURFACE-QUEUE-OVERLAY-EXPANDED".into())
                                    .role(Role::Status)
                                    .aria_label(format!(
                                        "Expanded execution queue with {item_count} jobs"
                                    ))
                                    .px_2()
                                    .py_1()
                                    .text_ui_sm(cx)
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(format!("Showing {item_count} execution jobs")),
                            )
                            .children(self.snapshot.queue.iter().enumerate().map(
                                |(position, item)| {
                                    render_queued_row(
                                        item,
                                        position,
                                        self.snapshot.queue.len(),
                                        self.selected_attempt_id == Some(item.attempt_id),
                                        pending_attempts.contains(&item.attempt_id),
                                        self.runtime_controls_available,
                                        self.on_action.clone(),
                                        cx,
                                    )
                                },
                            ))
                            .children(active_attempts.iter().map(|attempt| {
                                render_active_row(
                                    attempt,
                                    self.selected_attempt_id == Some(attempt.attempt_id),
                                    pending_attempts.contains(&attempt.attempt_id),
                                    self.show_progress,
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

fn render_queued_row(
    item: &QueuedPrompt,
    position: usize,
    queue_length: usize,
    selected: bool,
    command_pending: bool,
    runtime_controls_available: bool,
    on_action: QueuePanelActionHandler,
    cx: &mut App,
) -> AnyElement {
    let attempt_id = item.attempt_id;
    let attempt_label = attempt_label(attempt_id);
    let select_handler = on_action.clone();
    let earlier_handler = on_action.clone();
    let later_handler = on_action.clone();
    let cancel_handler = on_action;
    let earlier_disabled = position == 0 || command_pending || !runtime_controls_available;
    let later_disabled = position
        .checked_add(1)
        .is_none_or(|next| next >= queue_length)
        || command_pending
        || !runtime_controls_available;
    let pending_suffix = command_pending
        .then_some("; command request pending")
        .unwrap_or("");

    v_flex()
        .id(format!("sim-queue-row-{}", attempt_id.0))
        .role(Role::ListItem)
        .aria_selected(selected)
        .aria_label(format!(
            "Queued attempt {attempt_label}, position {} of {}, priority {}, {} nodes{pending_suffix}",
            position.saturating_add(1),
            queue_length,
            item.priority,
            item.plan.nodes.len()
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
                            div()
                                .text_ui_sm(cx)
                                .child(format!("Attempt {attempt_label}")),
                        )
                        .child(
                            div()
                                .text_ui_sm(cx)
                                .text_color(cx.theme().colors().text_muted)
                                .child(format!(
                                    "Position {} · priority {} · {} nodes",
                                    position.saturating_add(1),
                                    item.priority,
                                    item.plan.nodes.len()
                                )),
                        ),
                )
                .child(
                    h_flex()
                        .gap_1()
                        .child(
                            Button::new(
                                format!("sim-inspect-queued-{}", attempt_id.0),
                                "Inspect",
                            )
                            .size(ButtonSize::Compact)
                            .aria_label(format!("Inspect queued attempt {attempt_label}"))
                            .on_click(move |_, window, cx| {
                                select_handler(
                                    QueuePanelAction::SelectAttempt(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            Button::new(
                                format!("sim-move-earlier-{}", attempt_id.0),
                                "Earlier",
                            )
                            .size(ButtonSize::Compact)
                            .disabled(earlier_disabled)
                            .aria_label(if command_pending {
                                format!("Move attempt {attempt_label} earlier (request pending)")
                            } else if !runtime_controls_available {
                                format!(
                                    "Move attempt {attempt_label} earlier unavailable: native runtime controller is not connected"
                                )
                            } else if position == 0 {
                                format!("Move attempt {attempt_label} earlier (already first)")
                            } else {
                                format!("Move attempt {attempt_label} earlier")
                            })
                            .on_click(move |_, window, cx| {
                                earlier_handler(
                                    QueuePanelAction::MoveEarlier(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            Button::new(
                                format!("sim-move-later-{}", attempt_id.0),
                                "Later",
                            )
                            .size(ButtonSize::Compact)
                            .disabled(later_disabled)
                            .aria_label(if command_pending {
                                format!("Move attempt {attempt_label} later (request pending)")
                            } else if !runtime_controls_available {
                                format!(
                                    "Move attempt {attempt_label} later unavailable: native runtime controller is not connected"
                                )
                            } else if position.checked_add(1).is_none_or(|next| next >= queue_length) {
                                format!("Move attempt {attempt_label} later (already last)")
                            } else {
                                format!("Move attempt {attempt_label} later")
                            })
                            .on_click(move |_, window, cx| {
                                later_handler(
                                    QueuePanelAction::MoveLater(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            Button::new(
                                format!("sim-cancel-queued-{}", attempt_id.0),
                                if command_pending { "Pending" } else { "Cancel" },
                            )
                            .size(ButtonSize::Compact)
                            .disabled(command_pending || !runtime_controls_available)
                            .aria_label(if command_pending {
                                format!("Cancel attempt {attempt_label} (request pending)")
                            } else if !runtime_controls_available {
                                format!(
                                    "Cancel attempt {attempt_label} unavailable: native runtime controller is not connected"
                                )
                            } else {
                                format!("Cancel queued attempt {attempt_label}")
                            })
                            .on_click(move |_, window, cx| {
                                cancel_handler(
                                    QueuePanelAction::Cancel(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        ),
                ),
        )
        .into_any_element()
}

fn render_active_row(
    attempt: &AttemptPresentation,
    selected: bool,
    command_pending: bool,
    show_progress: bool,
    runtime_controls_available: bool,
    on_action: QueuePanelActionHandler,
    cx: &mut App,
) -> AnyElement {
    let attempt_id = attempt.attempt_id;
    let attempt_label = attempt_label(attempt_id);
    let select_handler = on_action.clone();
    let interrupt_handler = on_action;
    let state_label = projected_state_name(attempt);

    v_flex()
        .id(format!("sim-active-row-{}", attempt_id.0))
        .role(Role::ListItem)
        .aria_selected(selected)
        .aria_label(format!("Attempt {attempt_label}, {state_label}"))
        .w_full()
        .px_2()
        .py_1()
        .gap_1()
        .border_b_1()
        .border_color(cx.theme().colors().border_variant)
        .when(selected, |this| {
            this.bg(cx.theme().colors().element_selected)
        })
        .child(
            h_flex()
                .w_full()
                .justify_between()
                .gap_2()
                .child(
                    v_flex()
                        .min_w_0()
                        .child(
                            div()
                                .text_ui_sm(cx)
                                .child(format!("Attempt {attempt_label}")),
                        )
                        .child(
                            div()
                                .id(format!("sim-attempt-status-{}", attempt_id.0))
                                .role(Role::Status)
                                .aria_label(format!("Attempt {attempt_label} is {state_label}"))
                                .text_ui_sm(cx)
                                .text_color(cx.theme().colors().text_muted)
                                .child(state_label),
                        ),
                )
                .child(
                    h_flex()
                        .gap_1()
                        .child(
                            div()
                                .debug_selector(|| "SIM-QUEUE-ACTIVE-INSPECT".into())
                                .child(
                                    Button::new(
                                        format!("sim-inspect-active-{}", attempt_id.0),
                                        "Inspect",
                                    )
                                    .size(ButtonSize::Compact)
                                    .aria_label(format!(
                                        "Inspect active attempt {attempt_label}"
                                    ))
                                    .on_click(move |_, window, cx| {
                                        select_handler(
                                            QueuePanelAction::SelectAttempt(attempt_id),
                                            window,
                                            cx,
                                        );
                                    }),
                                ),
                        )
                        .child(
                            Button::new(
                                format!("sim-interrupt-{}", attempt_id.0),
                                if attempt.state == AttemptState::Cancelling {
                                    "Cancelling"
                                } else if command_pending {
                                    "Pending"
                                } else {
                                    "Interrupt"
                                },
                            )
                            .size(ButtonSize::Compact)
                            .disabled(
                                command_pending
                                    || attempt.state == AttemptState::Cancelling
                                    || !runtime_controls_available,
                            )
                            .aria_label(if attempt.state == AttemptState::Cancelling {
                                format!(
                                    "Interrupt attempt {attempt_label} (cancellation in progress)"
                                )
                            } else if command_pending {
                                format!("Interrupt attempt {attempt_label} (request pending)")
                            } else if !runtime_controls_available {
                                format!(
                                    "Interrupt attempt {attempt_label} unavailable: native runtime controller is not connected"
                                )
                            } else {
                                format!("Interrupt running attempt {attempt_label}")
                            })
                            .on_click(move |_, window, cx| {
                                interrupt_handler(
                                    QueuePanelAction::Interrupt(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        ),
                ),
        )
        .when_some(
            show_progress.then_some(attempt.progress.as_ref()).flatten(),
            |this, progress| {
                let total = progress.total.max(1);
                let completed = progress.completed.min(total);
                this.child(
                    v_flex()
                        .id(format!("sim-queue-inline-progress-{}", attempt_id.0))
                        .debug_selector(|| "SIM-SURFACE-QUEUE-INLINE-PROGRESS".into())
                        .role(Role::ProgressIndicator)
                        .aria_label(progress.node_id.as_ref().map_or_else(
                            || format!("Attempt {attempt_label} progress"),
                            |node_id| {
                                format!("Attempt {attempt_label} progress for node {}", node_id.0)
                            },
                        ))
                        .aria_value(format!("{completed} of {total}"))
                        .aria_numeric_value(completed as f64)
                        .aria_min_numeric_value(0.0)
                        .aria_max_numeric_value(total as f64)
                        .gap_0p5()
                        .child(
                            div()
                                .id(format!(
                                    "sim-queue-inline-progress-summary-{}",
                                    attempt_id.0
                                ))
                                .debug_selector(|| {
                                    "SIM-SURFACE-QUEUE-INLINE-PROGRESS-SUMMARY".into()
                                })
                                .role(Role::Status)
                                .text_ui_sm(cx)
                                .text_color(cx.theme().colors().text_muted)
                                .child(format!("{completed} / {total}")),
                        )
                        .child(
                            div()
                                .id(format!("sim-queue-linear-progress-bar-{}", attempt_id.0))
                                .h_1()
                                .w_full()
                                .rounded_full()
                                .bg(cx.theme().colors().element_background)
                                .child(
                                    div()
                                        .h_full()
                                        .rounded_full()
                                        .bg(cx.theme().status().info)
                                        .w(relative(completed as f32 / total as f32)),
                                ),
                        ),
                )
            },
        )
        .when_some(attempt.preview.as_ref(), |this, preview| {
            let preview_identity = preview_identity(preview.frame_index, preview.output_index);
            this.child(
                div()
                    .id(format!("sim-preview-summary-{}", attempt_id.0))
                    .role(Role::Status)
                    .aria_label(format!(
                        "Latest preview for node {}, revision {}, media type {}{}",
                        preview.node_id.0, preview.revision, preview.media_type, preview_identity
                    ))
                    .text_ui_sm(cx)
                    .text_color(cx.theme().colors().text_muted)
                    .child(format!(
                        "Preview · {} · revision {}{}{}",
                        preview.media_type,
                        preview.revision,
                        dimensions(preview.width, preview.height),
                        preview_identity
                    )),
            )
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
            .id("sim-queue-loading")
            .role(Role::Status)
            .aria_label("Loading native execution queue")
            .w_full()
            .p_2()
            .text_ui_sm(cx)
            .text_color(cx.theme().colors().text_muted)
            .child("Loading execution queue…")
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
                .id("sim-queue-stale")
                .role(Role::Alert)
                .aria_label(format!(
                    "Execution queue is stale at {revision}: {}: {}. The last known queue remains available for inspection",
                    failure.code, failure.message
                ))
                .w_full()
                .p_2()
                .bg(cx.theme().status().warning_background.opacity(0.2))
                .text_ui_sm(cx)
                .child(format!(
                    "Stale queue data · {revision} · {}: {} · showing the last known queue",
                    failure.code, failure.message
                ))
                .into_any_element()
        }
        ExecutionSnapshotStatus::Partial { failure } => div()
            .id("sim-queue-partial")
            .role(Role::Alert)
            .aria_label(format!(
                "Execution queue is partially available: {}: {}",
                failure.code, failure.message
            ))
            .w_full()
            .p_2()
            .bg(cx.theme().status().warning_background.opacity(0.2))
            .text_ui_sm(cx)
            .child(format!(
                "Partial queue data · {}: {}",
                failure.code, failure.message
            ))
            .into_any_element(),
        ExecutionSnapshotStatus::Unavailable { failure } => div()
            .id("sim-queue-unavailable")
            .role(Role::Alert)
            .aria_label(format!(
                "Execution queue is unavailable: {}: {}",
                failure.code, failure.message
            ))
            .w_full()
            .p_2()
            .bg(cx.theme().status().error_background.opacity(0.2))
            .text_ui_sm(cx)
            .child(format!(
                "Queue unavailable · {}: {}",
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

fn attempt_label(attempt_id: AttemptId) -> SharedString {
    attempt_id
        .0
        .to_string()
        .chars()
        .take(8)
        .collect::<String>()
        .into()
}

fn dimensions(width: Option<u32>, height: Option<u32>) -> String {
    match (width, height) {
        (Some(width), Some(height)) => format!(" · {width}×{height}"),
        _ => String::new(),
    }
}

fn preview_identity(frame_index: Option<u64>, output_index: Option<usize>) -> String {
    match (frame_index, output_index) {
        (Some(frame_index), Some(output_index)) => {
            format!(" · frame {frame_index} · output {output_index}")
        }
        (Some(frame_index), None) => format!(" · frame {frame_index}"),
        (None, Some(output_index)) => format!(" · output {output_index}"),
        (None, None) => String::new(),
    }
}
