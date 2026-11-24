use gpui::SharedString;

use crate::{
    collaborative_participants::{
        CollaborativeExecutionPhase, CollaborativeParticipantPresence,
        CollaborativeParticipantProviderState,
    },
    collaborative_shell_state::{CollaborativeShellPhase, CollaborativeShellScope},
};

pub const WORKSPACE_LABEL: &str = "Multiplayer Workspace";
pub const TOP_BAR_LABEL: &str = "Multiplayer Workspace controls";
pub const NAVIGATION_LABEL: &str = "Multiplayer Workspace navigation";
pub const TIMELINE_LABEL: &str = "Multiplayer Workspace activity timeline";
pub const COMPOSER_LABEL: &str = "Message and agent prompt composer";
pub const REVIEW_LABEL: &str = "Review changes";
pub const STATUS_LABEL: &str = "Multiplayer Workspace status";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeAnnouncementRole {
    Status,
    Alert,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborativeAnnouncement {
    pub role: CollaborativeAnnouncementRole,
    pub label: SharedString,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborativeAccessibilityContract {
    pub region: &'static str,
    pub role: &'static str,
    pub label: &'static str,
}

pub const COLLABORATIVE_ACCESSIBILITY_CONTRACTS: [CollaborativeAccessibilityContract; 7] = [
    CollaborativeAccessibilityContract {
        region: "workspace",
        role: "main",
        label: WORKSPACE_LABEL,
    },
    CollaborativeAccessibilityContract {
        region: "top_bar",
        role: "group",
        label: TOP_BAR_LABEL,
    },
    CollaborativeAccessibilityContract {
        region: "navigation",
        role: "navigation",
        label: NAVIGATION_LABEL,
    },
    CollaborativeAccessibilityContract {
        region: "timeline",
        role: "document",
        label: TIMELINE_LABEL,
    },
    CollaborativeAccessibilityContract {
        region: "composer",
        role: "group",
        label: COMPOSER_LABEL,
    },
    CollaborativeAccessibilityContract {
        region: "review",
        role: "complementary",
        label: REVIEW_LABEL,
    },
    CollaborativeAccessibilityContract {
        region: "status",
        role: "status",
        label: STATUS_LABEL,
    },
];

pub fn participant_label(state: &CollaborativeParticipantProviderState) -> SharedString {
    match state {
        CollaborativeParticipantProviderState::Ready(view_data)
            if view_data.participants.is_empty() =>
        {
            "No participants".into()
        }
        CollaborativeParticipantProviderState::Ready(view_data) => {
            let names = view_data
                .participants
                .iter()
                .take(3)
                .map(|participant| {
                    format!(
                        "{} ({})",
                        participant.display_name,
                        presence_label(participant.presence)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let remaining = view_data.participants.len().saturating_sub(3);
            if remaining == 0 {
                format!("Participants: {names}").into()
            } else {
                format!("Participants: {names}, and {remaining} more").into()
            }
        }
        CollaborativeParticipantProviderState::Failed(_) => "Participant status unavailable".into(),
        CollaborativeParticipantProviderState::Unavailable => "Participants unavailable".into(),
    }
}

pub fn execution_announcement(
    state: &CollaborativeParticipantProviderState,
) -> Option<CollaborativeAnnouncement> {
    match state {
        CollaborativeParticipantProviderState::Failed(message) => Some(CollaborativeAnnouncement {
            role: CollaborativeAnnouncementRole::Alert,
            label: format!("Participant status unavailable: {message}").into(),
        }),
        CollaborativeParticipantProviderState::Ready(view_data) => {
            let execution = view_data.execution.as_ref()?;
            let (role, label) = match execution.phase {
                CollaborativeExecutionPhase::Running => {
                    (CollaborativeAnnouncementRole::Status, "Agent task running")
                }
                CollaborativeExecutionPhase::WaitingForUser => (
                    CollaborativeAnnouncementRole::Status,
                    "Agent task waiting for user",
                ),
                CollaborativeExecutionPhase::Failed => {
                    (CollaborativeAnnouncementRole::Alert, "Agent task failed")
                }
                CollaborativeExecutionPhase::Completed => (
                    CollaborativeAnnouncementRole::Status,
                    "Agent task completed",
                ),
                CollaborativeExecutionPhase::Idle | CollaborativeExecutionPhase::Unknown => {
                    return None;
                }
            };
            Some(CollaborativeAnnouncement {
                role,
                label: label.into(),
            })
        }
        CollaborativeParticipantProviderState::Unavailable => None,
    }
}

pub(crate) fn shell_announcement(
    phase: &CollaborativeShellPhase,
) -> Option<CollaborativeAnnouncement> {
    let (role, label) = match phase {
        CollaborativeShellPhase::Ready => return None,
        CollaborativeShellPhase::Loading { scope, .. } => (
            CollaborativeAnnouncementRole::Status,
            format!("Loading {}", scope_label(*scope)),
        ),
        CollaborativeShellPhase::PartialFailure { scope, summary, .. }
        | CollaborativeShellPhase::InitializationFailed { scope, summary, .. } => (
            CollaborativeAnnouncementRole::Alert,
            format!("{summary}. Affected scope: {}", scope_label(*scope)),
        ),
        CollaborativeShellPhase::Retrying { scope, attempt, .. } => (
            CollaborativeAnnouncementRole::Status,
            format!("Retrying {}. Attempt {attempt}", scope_label(*scope)),
        ),
    };
    Some(CollaborativeAnnouncement {
        role,
        label: label.into(),
    })
}

fn scope_label(scope: CollaborativeShellScope) -> &'static str {
    match scope {
        CollaborativeShellScope::Workspace => "Multiplayer Workspace",
        CollaborativeShellScope::Timeline => "timeline",
        CollaborativeShellScope::Realtime => "realtime synchronization",
    }
}

fn presence_label(presence: CollaborativeParticipantPresence) -> &'static str {
    match presence {
        CollaborativeParticipantPresence::Online => "online",
        CollaborativeParticipantPresence::Away => "away",
        CollaborativeParticipantPresence::Busy => "busy",
        CollaborativeParticipantPresence::Offline => "offline",
        CollaborativeParticipantPresence::Unknown => "presence unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collaborative_participants::{
        CollaborativeExecutionLocation, CollaborativeExecutionStatus, CollaborativeParticipant,
        CollaborativeParticipantViewData,
    };

    #[test]
    fn collaborative_accessibility() {
        assert_eq!(COLLABORATIVE_ACCESSIBILITY_CONTRACTS.len(), 7);
        for required in [
            "workspace",
            "top_bar",
            "navigation",
            "timeline",
            "composer",
            "review",
            "status",
        ] {
            assert!(
                COLLABORATIVE_ACCESSIBILITY_CONTRACTS
                    .iter()
                    .any(|contract| contract.region == required && !contract.label.is_empty())
            );
        }

        let running =
            CollaborativeParticipantProviderState::Ready(CollaborativeParticipantViewData {
                participants: vec![CollaborativeParticipant::agent(
                    "agent-1",
                    "Builder",
                    None,
                    CollaborativeParticipantPresence::Online,
                )],
                execution: Some(CollaborativeExecutionStatus {
                    phase: CollaborativeExecutionPhase::Running,
                    model: None,
                    runtime: None,
                    location: CollaborativeExecutionLocation::Local,
                }),
                task_title: Some("Run checks".into()),
                connection: Default::default(),
            });
        assert_eq!(
            participant_label(&running).as_ref(),
            "Participants: Builder (online)"
        );
        assert_eq!(
            execution_announcement(&running),
            Some(CollaborativeAnnouncement {
                role: CollaborativeAnnouncementRole::Status,
                label: "Agent task running".into(),
            })
        );

        let failed = CollaborativeParticipantProviderState::failed("runtime disconnected");
        assert_eq!(
            execution_announcement(&failed),
            Some(CollaborativeAnnouncement {
                role: CollaborativeAnnouncementRole::Alert,
                label: "Participant status unavailable: runtime disconnected".into(),
            })
        );
        let shell_failure = CollaborativeShellPhase::InitializationFailed {
            scope: CollaborativeShellScope::Workspace,
            summary: "Unable to initialize collaboration".into(),
            last_trustworthy_state: None,
        };
        assert_eq!(
            shell_announcement(&shell_failure),
            Some(CollaborativeAnnouncement {
                role: CollaborativeAnnouncementRole::Alert,
                label: "Unable to initialize collaboration. Affected scope: Multiplayer Workspace"
                    .into(),
            })
        );
    }
}
