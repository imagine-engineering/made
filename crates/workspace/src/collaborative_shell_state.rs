use gpui::{Action, App, Context, Entity, EventEmitter, RenderOnce, Role, SharedString};
use ui::{Banner, Button, ButtonStyle, Severity, Window, prelude::*};

use crate::SwitchToEditorWorkspace;
use crate::collaborative_accessibility::{CollaborativeAnnouncementRole, shell_announcement};

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "collaboration service bindings will construct scoped shell updates"
    )
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CollaborativeShellScope {
    Workspace,
    Timeline,
    Realtime,
}

impl CollaborativeShellScope {
    fn label(self) -> &'static str {
        match self {
            Self::Workspace => "Multiplayer Workspace",
            Self::Timeline => "timeline",
            Self::Realtime => "realtime synchronization",
        }
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "collaboration service bindings will construct shell failure phases"
    )
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CollaborativeShellPhase {
    Ready,
    Loading {
        scope: CollaborativeShellScope,
        last_trustworthy_state: Option<SharedString>,
    },
    PartialFailure {
        scope: CollaborativeShellScope,
        summary: SharedString,
        last_trustworthy_state: SharedString,
    },
    InitializationFailed {
        scope: CollaborativeShellScope,
        summary: SharedString,
        last_trustworthy_state: Option<SharedString>,
    },
    Retrying {
        scope: CollaborativeShellScope,
        attempt: u32,
        last_trustworthy_state: Option<SharedString>,
    },
}

impl CollaborativeShellPhase {
    fn scope(&self) -> Option<CollaborativeShellScope> {
        match self {
            Self::Ready => None,
            Self::Loading { scope, .. }
            | Self::PartialFailure { scope, .. }
            | Self::InitializationFailed { scope, .. }
            | Self::Retrying { scope, .. } => Some(*scope),
        }
    }

    fn last_trustworthy_state(&self) -> Option<SharedString> {
        match self {
            Self::Ready => None,
            Self::Loading {
                last_trustworthy_state,
                ..
            }
            | Self::InitializationFailed {
                last_trustworthy_state,
                ..
            }
            | Self::Retrying {
                last_trustworthy_state,
                ..
            } => last_trustworthy_state.clone(),
            Self::PartialFailure {
                last_trustworthy_state,
                ..
            } => Some(last_trustworthy_state.clone()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CollaborativeShellRetryRequested {
    pub(crate) scope: CollaborativeShellScope,
    pub(crate) attempt: u32,
}

pub(crate) struct CollaborativeShellState {
    phase: CollaborativeShellPhase,
    retry_attempt: u32,
}

impl CollaborativeShellState {
    pub(crate) fn new() -> Self {
        Self {
            phase: CollaborativeShellPhase::Ready,
            retry_attempt: 0,
        }
    }

    pub(crate) fn phase(&self) -> &CollaborativeShellPhase {
        &self.phase
    }

    pub(crate) fn transition(&mut self, phase: CollaborativeShellPhase, cx: &mut Context<Self>) {
        if self.phase != phase {
            if matches!(phase, CollaborativeShellPhase::Ready) {
                self.retry_attempt = 0;
            }
            self.phase = phase;
            cx.notify();
        }
    }

    fn retry(&mut self, cx: &mut Context<Self>) {
        let retryable = matches!(
            self.phase,
            CollaborativeShellPhase::PartialFailure { .. }
                | CollaborativeShellPhase::InitializationFailed { .. }
        );
        if !retryable {
            return;
        }

        let Some(scope) = self.phase.scope() else {
            return;
        };
        let last_trustworthy_state = self.phase.last_trustworthy_state();
        self.retry_attempt = self.retry_attempt.saturating_add(1);
        self.transition(
            CollaborativeShellPhase::Retrying {
                scope,
                attempt: self.retry_attempt,
                last_trustworthy_state,
            },
            cx,
        );
        cx.emit(CollaborativeShellRetryRequested {
            scope,
            attempt: self.retry_attempt,
        });
    }
}

impl EventEmitter<CollaborativeShellRetryRequested> for CollaborativeShellState {}

#[derive(IntoElement)]
pub(crate) struct CollaborativeShellStatus {
    state: Entity<CollaborativeShellState>,
}

impl CollaborativeShellStatus {
    pub(crate) fn new(state: Entity<CollaborativeShellState>) -> Self {
        Self { state }
    }

    fn last_trustworthy_label(last_trustworthy_state: Option<SharedString>) -> Option<Label> {
        last_trustworthy_state.map(|state| {
            Label::new(format!("Last trustworthy state: {state}"))
                .size(LabelSize::XSmall)
                .color(Color::Muted)
        })
    }
}

impl RenderOnce for CollaborativeShellStatus {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let phase = self.state.read(cx).phase().clone();
        let announcement = shell_announcement(&phase);
        let state = self.state;

        let banner = match phase {
            CollaborativeShellPhase::Ready => None,
            CollaborativeShellPhase::Loading {
                scope,
                last_trustworthy_state,
            } => Some(
                Banner::new()
                    .severity(Severity::Info)
                    .child(
                        v_flex()
                            .min_w_0()
                            .child(LoadingLabel::new(format!("Loading {}", scope.label())))
                            .children(Self::last_trustworthy_label(last_trustworthy_state)),
                    )
                    .into_any_element(),
            ),
            CollaborativeShellPhase::PartialFailure {
                scope,
                summary,
                last_trustworthy_state,
            } => {
                let retry_state = state;
                Some(
                    Banner::new()
                        .severity(Severity::Warning)
                        .wrap_content(true)
                        .child(
                            v_flex()
                                .min_w_0()
                                .child(Label::new(summary).size(LabelSize::Small))
                                .child(
                                    Label::new(format!("Affected scope: {}", scope.label()))
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                                .children(Self::last_trustworthy_label(Some(
                                    last_trustworthy_state,
                                ))),
                        )
                        .action_slot(
                            div()
                                .debug_selector(|| "COLLABORATIVE-SHELL-RETRY".to_owned())
                                .child(
                                    Button::new("collaborative-shell-retry", "Retry")
                                        .style(ButtonStyle::Outlined)
                                        .aria_label(format!("Retry {}", scope.label()))
                                        .on_click(move |_, _, cx| {
                                            retry_state.update(cx, CollaborativeShellState::retry);
                                        }),
                                ),
                        )
                        .into_any_element(),
                )
            }
            CollaborativeShellPhase::InitializationFailed {
                scope,
                summary,
                last_trustworthy_state,
            } => {
                let retry_state = state;
                Some(
                    Banner::new()
                        .severity(Severity::Error)
                        .wrap_content(true)
                        .child(
                            v_flex()
                                .min_w_0()
                                .child(Label::new(summary).size(LabelSize::Small))
                                .child(
                                    Label::new(format!("Affected scope: {}", scope.label()))
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                                .children(Self::last_trustworthy_label(last_trustworthy_state)),
                        )
                        .action_slot(
                            h_flex()
                                .gap_1()
                                .child(
                                    div()
                                        .debug_selector(|| "COLLABORATIVE-SHELL-RETRY".to_owned())
                                        .child(
                                            Button::new("collaborative-shell-retry", "Retry")
                                                .style(ButtonStyle::Outlined)
                                                .aria_label(format!("Retry {}", scope.label()))
                                                .on_click(move |_, _, cx| {
                                                    retry_state.update(
                                                        cx,
                                                        CollaborativeShellState::retry,
                                                    );
                                                }),
                                        ),
                                )
                                .child(
                                    div()
                                        .debug_selector(|| {
                                            "COLLABORATIVE-SHELL-EDITOR-FALLBACK".to_owned()
                                        })
                                        .child(
                                            Button::new(
                                                "collaborative-shell-editor-fallback",
                                                "Open Editor Workspace",
                                            )
                                            .style(ButtonStyle::Filled)
                                            .aria_label(
                                                "Open Editor Workspace and keep the current project",
                                            )
                                            .on_click(move |_, window, cx| {
                                                window.dispatch_action(
                                                    SwitchToEditorWorkspace.boxed_clone(),
                                                    cx,
                                                );
                                            }),
                                        ),
                                ),
                        )
                        .into_any_element(),
                )
            }
            CollaborativeShellPhase::Retrying {
                scope,
                attempt,
                last_trustworthy_state,
            } => Some(
                Banner::new()
                    .severity(Severity::Info)
                    .child(
                        v_flex()
                            .min_w_0()
                            .child(LoadingLabel::new(format!(
                                "Retrying {} (attempt {attempt})",
                                scope.label()
                            )))
                            .children(Self::last_trustworthy_label(last_trustworthy_state)),
                    )
                    .into_any_element(),
            ),
        };

        div()
            .id("collaborative-shell-status")
            .debug_selector(|| "COLLABORATIVE-SHELL-STATUS".to_owned())
            .when_some(announcement, |this, announcement| {
                this.role(match announcement.role {
                    CollaborativeAnnouncementRole::Status => Role::Status,
                    CollaborativeAnnouncementRole::Alert => Role::Alert,
                })
                .aria_label(announcement.label)
            })
            .w_full()
            .flex_none()
            .px_2()
            .py_1()
            .when_some(banner, |this, banner| this.child(banner))
    }
}
