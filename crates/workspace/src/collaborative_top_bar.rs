use gpui::{Action, App, Entity, RenderOnce, Role};
use project::Project;
use ui::{Avatar, ButtonStyle, Tooltip, Window, prelude::*};

use crate::{
    CopyRoomId, OpenCollaborators, ShareProject, SwitchToEditorWorkspace,
    collaborative_accessibility::{
        TOP_BAR_LABEL, participant_label as accessibility_participant_label,
    },
    collaborative_layout::{CollaborativeLayout, RESIZE_HANDLE_WIDTH},
    collaborative_participants::{
        CollaborativeConnectionState, CollaborativeParticipantIdentity,
        CollaborativeParticipantProviderState,
    },
    collaborative_review::ToggleCollaborativeReview,
};

const NO_ACTIVE_TASK_LABEL: &str = "Select a task or thread";
const PARTICIPANTS_UNAVAILABLE_LABEL: &str = "Participants unavailable";
const CONNECTION_UNAVAILABLE_LABEL: &str = "Connection unavailable";

#[derive(Clone, Copy)]
struct CollaborativeTopBarActionAvailability {
    share: bool,
    invite: bool,
    connection_details: bool,
    review_layout: bool,
    editor_layout: bool,
}

#[derive(IntoElement)]
pub(crate) struct CollaborativeTopBar {
    project: Entity<Project>,
    layout: Entity<CollaborativeLayout>,
    participant_state: CollaborativeParticipantProviderState,
}

impl CollaborativeTopBar {
    pub(crate) fn new(
        project: Entity<Project>,
        layout: Entity<CollaborativeLayout>,
        participant_state: CollaborativeParticipantProviderState,
    ) -> Self {
        Self {
            project,
            layout,
            participant_state,
        }
    }

    fn project_title(&self, cx: &App) -> SharedString {
        self.project
            .read(cx)
            .visible_worktrees(cx)
            .next()
            .and_then(|worktree| {
                worktree
                    .read(cx)
                    .root_name()
                    .file_name()
                    .map(|name| name.to_string().into())
            })
            .unwrap_or_else(|| "Untitled Project".into())
    }

    fn action_availability(&self) -> CollaborativeTopBarActionAvailability {
        let room_actions = matches!(
            &self.participant_state,
            CollaborativeParticipantProviderState::Ready(view_data)
                if view_data.connection.supports_room_actions()
        );
        CollaborativeTopBarActionAvailability {
            share: room_actions,
            invite: true,
            connection_details: room_actions,
            review_layout: true,
            editor_layout: true,
        }
    }

    #[cfg(test)]
    fn participant_label(&self) -> SharedString {
        match &self.participant_state {
            CollaborativeParticipantProviderState::Ready(view_data)
                if view_data.participants.is_empty() =>
            {
                "No participants".into()
            }
            CollaborativeParticipantProviderState::Ready(view_data) => {
                match view_data.participants.as_slice() {
                    [participant] => participant.display_name.clone(),
                    participants => format!("{} participants", participants.len()).into(),
                }
            }
            CollaborativeParticipantProviderState::Failed(_) => "Participants unavailable".into(),
            CollaborativeParticipantProviderState::Unavailable => {
                PARTICIPANTS_UNAVAILABLE_LABEL.into()
            }
        }
    }

    fn task_title(&self) -> SharedString {
        match &self.participant_state {
            CollaborativeParticipantProviderState::Ready(view_data) => view_data
                .task_title
                .as_ref()
                .filter(|title| !title.trim().is_empty())
                .cloned()
                .unwrap_or_else(|| NO_ACTIVE_TASK_LABEL.into()),
            CollaborativeParticipantProviderState::Failed(_)
            | CollaborativeParticipantProviderState::Unavailable => NO_ACTIVE_TASK_LABEL.into(),
        }
    }

    fn connection_label(&self) -> SharedString {
        match &self.participant_state {
            CollaborativeParticipantProviderState::Ready(view_data) => {
                let connection = view_data.connection.label();
                view_data.execution.as_ref().map_or_else(
                    || connection.into(),
                    |execution| {
                        format!(
                            "{connection} · {} · {}",
                            execution.runtime_label(),
                            execution.location_label()
                        )
                        .into()
                    },
                )
            }
            CollaborativeParticipantProviderState::Failed(message) => message.clone(),
            CollaborativeParticipantProviderState::Unavailable => {
                CONNECTION_UNAVAILABLE_LABEL.into()
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn test_snapshot(&self, cx: &App) -> CollaborativeTopBarTestSnapshot {
        let action_availability = self.action_availability();
        CollaborativeTopBarTestSnapshot {
            project_title: self.project_title(cx),
            task_title: self.task_title(),
            participants: self.participant_label(),
            connection: self.connection_label(),
            share_enabled: action_availability.share,
            invite_enabled: action_availability.invite,
            connection_details_enabled: action_availability.connection_details,
            review_layout_enabled: action_availability.review_layout,
            editor_layout_enabled: action_availability.editor_layout,
        }
    }
}

#[cfg(test)]
pub(crate) struct CollaborativeTopBarTestSnapshot {
    pub(crate) project_title: SharedString,
    pub(crate) task_title: SharedString,
    pub(crate) participants: SharedString,
    pub(crate) connection: SharedString,
    pub(crate) share_enabled: bool,
    pub(crate) invite_enabled: bool,
    pub(crate) connection_details_enabled: bool,
    pub(crate) review_layout_enabled: bool,
    pub(crate) editor_layout_enabled: bool,
}

impl RenderOnce for CollaborativeTopBar {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let action_availability = self.action_availability();
        let review_requested = self.layout.read(cx).review_requested();
        let task_title = self.task_title();
        let connection_label = self.connection_label();
        let connection_icon = match &self.participant_state {
            CollaborativeParticipantProviderState::Ready(view_data) => match view_data.connection {
                CollaborativeConnectionState::Connected => IconName::UserGroup,
                CollaborativeConnectionState::Connecting => IconName::ArrowCircle,
                CollaborativeConnectionState::Disconnected
                | CollaborativeConnectionState::Failed => IconName::Disconnected,
            },
            CollaborativeParticipantProviderState::Failed(_)
            | CollaborativeParticipantProviderState::Unavailable => IconName::Disconnected,
        };
        let geometry = self.layout.read(cx).geometry(window);
        let mut layout_actions = Some(
            h_flex()
                .flex_none()
                .gap(DynamicSpacing::Base04.rems(cx))
                .child(
                    div()
                        .debug_selector(|| "COLLABORATIVE-TOP-BAR-REVIEW-LAYOUT".to_owned())
                        .child(
                            IconButton::new(
                                "collaborative-review-layout",
                                if review_requested {
                                    IconName::ThreadsSidebarRightOpen
                                } else {
                                    IconName::ThreadsSidebarRightClosed
                                },
                            )
                            .style(ButtonStyle::Subtle)
                            .size(ButtonSize::Compact)
                            .icon_size(IconSize::Small)
                            .aria_label("Toggle review layout")
                            .disabled(!action_availability.review_layout)
                            .tooltip(Tooltip::text(if review_requested {
                                "Hide Review Changes"
                            } else {
                                "Show Review Changes"
                            }))
                            .on_click(move |_, window, cx| {
                                window.dispatch_action(ToggleCollaborativeReview.boxed_clone(), cx);
                            }),
                        ),
                )
                .child(
                    div()
                        .debug_selector(|| "COLLABORATIVE-TOP-BAR-EDITOR-LAYOUT".to_owned())
                        .child(
                            IconButton::new("collaborative-editor-layout", IconName::Code)
                                .style(ButtonStyle::Subtle)
                                .size(ButtonSize::Compact)
                                .icon_size(IconSize::Small)
                                .aria_label("Switch to Editor Workspace")
                                .disabled(!action_availability.editor_layout)
                                .tooltip(Tooltip::text("Switch to Editor Workspace"))
                                .on_click(|_, window, cx| {
                                    window
                                        .dispatch_action(SwitchToEditorWorkspace.boxed_clone(), cx);
                                }),
                        ),
                )
                .into_any_element(),
        );
        let project_title = self.project_title(cx);
        let participant_state = self.participant_state;
        let participant_accessibility_label = accessibility_participant_label(&participant_state);
        let participant_role = if matches!(
            &participant_state,
            CollaborativeParticipantProviderState::Failed(_)
        ) {
            Role::Alert
        } else {
            Role::Group
        };
        let conversation = h_flex()
            .id("collaborative-conversation-header")
            .debug_selector(|| "COLLABORATIVE-CONVERSATION-HEADER".to_owned())
            .min_h(rems_from_px(ui::fractal::CONTROL_MEDIUM))
            .flex_1()
            .min_w_0()
            .overflow_hidden()
            .px(DynamicSpacing::Base04.rems(cx))
            .gap(DynamicSpacing::Base04.rems(cx))
            .bg(cx.theme().colors().title_bar_background)
            .role(Role::Group)
            .aria_label(format!("{TOP_BAR_LABEL}: {}", project_title))
            .child(
                h_flex()
                    .id("collaborative-top-bar-title")
                    .debug_selector(|| "COLLABORATIVE-TOP-BAR-TITLE".to_owned())
                    .flex_shrink_1()
                    .min_w_0()
                    .gap(DynamicSpacing::Base04.rems(cx))
                    .child(Label::new(task_title).size(LabelSize::Small).truncate()),
            )
            .child(
                h_flex()
                    .id("collaborative-top-bar-participants")
                    .debug_selector(|| "COLLABORATIVE-TOP-BAR-PARTICIPANTS".to_owned())
                    .gap(DynamicSpacing::Base04.rems(cx))
                    .role(participant_role)
                    .aria_label(participant_accessibility_label)
                    .child(h_flex().gap(DynamicSpacing::Base04.rems(cx)).map(|this| {
                        match participant_state {
                            CollaborativeParticipantProviderState::Ready(view_data) => this
                                .debug_selector(|| {
                                    "COLLABORATIVE-TOP-BAR-PARTICIPANTS-READY".to_owned()
                                })
                                .children(view_data.participants.iter().take(3).map(
                                    |participant| {
                                        if let Some(avatar_uri) = &participant.avatar_uri {
                                            div()
                                                .debug_selector(|| {
                                                    "COLLABORATIVE-PARTICIPANT-AVATAR".to_owned()
                                                })
                                                .child(
                                                    Avatar::new(avatar_uri.clone())
                                                        .size(rems(1.25)),
                                                )
                                                .into_any_element()
                                        } else {
                                            h_flex()
                                                .debug_selector(|| {
                                                    "COLLABORATIVE-PARTICIPANT-AVATAR-FALLBACK"
                                                        .to_owned()
                                                })
                                                .size(rems(1.25))
                                                .justify_center()
                                                .rounded_full()
                                                .bg(cx.theme().colors().element_background)
                                                .child(
                                                    Icon::new(match &participant.identity {
                                                        CollaborativeParticipantIdentity::Human(
                                                            _,
                                                        ) => IconName::Person,
                                                        CollaborativeParticipantIdentity::Agent(
                                                            _,
                                                        ) => IconName::ZedAssistant,
                                                    })
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted),
                                                )
                                                .into_any_element()
                                        }
                                    },
                                )),
                            CollaborativeParticipantProviderState::Failed(message) => this
                                .debug_selector(|| {
                                    "COLLABORATIVE-TOP-BAR-PARTICIPANTS-FAILED".to_owned()
                                })
                                .child(
                                    Icon::new(IconName::Warning)
                                        .size(IconSize::Small)
                                        .color(Color::Error),
                                )
                                .child(
                                    Label::new(message)
                                        .size(LabelSize::XSmall)
                                        .color(Color::Error),
                                ),
                            CollaborativeParticipantProviderState::Unavailable => this
                                .debug_selector(|| {
                                    "COLLABORATIVE-TOP-BAR-PARTICIPANTS-UNAVAILABLE".to_owned()
                                })
                                .child(
                                    Icon::new(IconName::UserGroup)
                                        .size(IconSize::Small)
                                        .color(Color::Muted),
                                )
                                .child(
                                    Label::new(PARTICIPANTS_UNAVAILABLE_LABEL)
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                ),
                        }
                    })),
            )
            .child(div().flex_1())
            .child(
                div()
                    .debug_selector(|| "COLLABORATIVE-TOP-BAR-SHARE".to_owned())
                    .child(
                        IconButton::new("collaborative-share", IconName::Share)
                            .style(ButtonStyle::Subtle)
                            .size(ButtonSize::Compact)
                            .icon_size(IconSize::Small)
                            .aria_label("Share Multiplayer Workspace")
                            .disabled(!action_availability.share)
                            .tooltip(Tooltip::text("Share or unshare the current project"))
                            .on_click(|_, window, cx| {
                                window.dispatch_action(ShareProject.boxed_clone(), cx);
                            }),
                    ),
            )
            .child(
                div()
                    .debug_selector(|| "COLLABORATIVE-TOP-BAR-INVITE".to_owned())
                    .child(
                        IconButton::new("collaborative-invite", IconName::UserArrowUp)
                            .style(ButtonStyle::Subtle)
                            .size(ButtonSize::Compact)
                            .icon_size(IconSize::Small)
                            .aria_label("Invite participants")
                            .disabled(!action_availability.invite)
                            .tooltip(Tooltip::text("Open collaborators and invitations"))
                            .on_click(|_, window, cx| {
                                window.dispatch_action(OpenCollaborators.boxed_clone(), cx);
                            }),
                    ),
            )
            .child(
                div()
                    .debug_selector(|| "COLLABORATIVE-TOP-BAR-CONNECTION".to_owned())
                    .child(
                        IconButton::new("collaborative-connection", connection_icon)
                            .style(ButtonStyle::Subtle)
                            .size(ButtonSize::Compact)
                            .icon_size(IconSize::Small)
                            .aria_label(connection_label.clone())
                            .disabled(!action_availability.connection_details)
                            .tooltip(Tooltip::text(connection_label))
                            .on_click(|_, window, cx| {
                                window.dispatch_action(CopyRoomId.boxed_clone(), cx);
                            }),
                    ),
            )
            .when(!geometry.review_visible, |this| {
                this.children(layout_actions.take())
            });
        h_flex()
            .id("collaborative-top-bar")
            .debug_selector(|| "COLLABORATIVE-TOP-BAR".to_owned())
            .w_full()
            .min_h(rems_from_px(ui::fractal::CONTROL_MEDIUM))
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .bg(cx.theme().colors().title_bar_background)
            .when(geometry.timeline_width > px(0.), |this| {
                this.child(conversation)
            })
            .when(geometry.review_visible, |this| {
                this.when(geometry.timeline_width > px(0.), |this| {
                    this.child(
                        div()
                            .w(RESIZE_HANDLE_WIDTH)
                            .h_full()
                            .flex_none()
                            .border_l_1()
                            .border_color(cx.theme().colors().border),
                    )
                })
                .child(
                    h_flex()
                        .debug_selector(|| "COLLABORATIVE-REVIEW-HEADER".to_owned())
                        .w(geometry.review_width)
                        .flex_none()
                        .child(self.layout.read(cx).render_review_header())
                        .children(layout_actions),
                )
            })
    }
}
