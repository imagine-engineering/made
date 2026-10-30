use std::{error::Error, fmt};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivitySourceKind {
    Acp,
    NativeAction,
    Nostr,
    Git,
    Workflow,
    Ci,
    Moderation,
    System,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivityItemId {
    source_kind: ActivitySourceKind,
    source_id: String,
}

impl ActivityItemId {
    pub fn new(
        source_kind: ActivitySourceKind,
        source_id: impl Into<String>,
    ) -> Result<Self, ActivityProjectionContractError> {
        let source_id = source_id.into();
        if source_id.trim().is_empty() {
            return Err(ActivityProjectionContractError::EmptySourceId);
        }
        Ok(Self {
            source_kind,
            source_id,
        })
    }

    pub fn source_kind(&self) -> ActivitySourceKind {
        self.source_kind
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActivityProjectionContractError {
    EmptySourceId,
}

impl fmt::Display for ActivityProjectionContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySourceId => formatter.write_str("activity source id must not be empty"),
        }
    }
}

impl Error for ActivityProjectionContractError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivitySemanticClass {
    Message,
    PlatformOperation,
    FileEdit,
    ShellCommand,
    Lifecycle,
    Thought,
    Plan,
    Permission,
    Error,
    Generic,
    Raw,
    Suppressed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityActorKind {
    Human,
    Agent,
    System,
    Service,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivityActor {
    pub kind: ActivityActorKind,
    pub id: String,
    pub label: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityObjectKind {
    Message,
    Channel,
    Thread,
    File,
    Repository,
    Command,
    TestSuite,
    Permission,
    Plan,
    Tool,
    Session,
    Workflow,
    Review,
    Identity,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivityObject {
    pub kind: ActivityObjectKind,
    pub id: Option<String>,
    pub label: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityOutcomeStatus {
    Pending,
    Success,
    Failure,
    Cancelled,
    TimedOut,
    NoChange,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivityOutcome {
    pub status: ActivityOutcomeStatus,
    pub summary: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityLifecycle {
    Pending,
    Running,
    WaitingForUser,
    Idle,
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
    Disconnected,
    Suppressed,
}

impl ActivityLifecycle {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::TimedOut | Self::Suppressed
        )
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivityContext {
    pub community_id: Option<String>,
    pub project_id: Option<String>,
    pub thread_id: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityVisibility {
    Public,
    Community,
    Project,
    Participants,
    Private,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActivityDetailHandle {
    AcpEntry {
        session_id: String,
        entry_id: String,
    },
    NativeAction {
        action_id: String,
    },
    ProtocolEvent {
        event_id: String,
    },
    GitChange {
        repository_id: String,
        change_id: String,
    },
    WorkflowRun {
        run_id: String,
        step_id: Option<String>,
    },
    RawSource {
        item_id: ActivityItemId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActivityLink {
    Action {
        action_id: String,
    },
    GitChange {
        repository_id: String,
        change_id: String,
    },
    Entity {
        entity_kind: String,
        entity_id: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivityItem {
    pub id: ActivityItemId,
    pub source_version: u64,
    pub class: ActivitySemanticClass,
    pub actor: ActivityActor,
    pub verb: String,
    pub object: ActivityObject,
    pub outcome: ActivityOutcome,
    pub lifecycle: ActivityLifecycle,
    pub occurred_at: DateTime<Utc>,
    pub projected_at: DateTime<Utc>,
    pub context: ActivityContext,
    pub visibility: ActivityVisibility,
    pub details: Option<ActivityDetailHandle>,
    pub links: Vec<ActivityLink>,
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone as _;

    use super::*;

    fn test_item(id: ActivityItemId, source_version: u64) -> ActivityItem {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 8, 14, 12, 0, 0)
            .single()
            .expect("test timestamp should be valid");
        ActivityItem {
            id,
            source_version,
            class: ActivitySemanticClass::ShellCommand,
            actor: ActivityActor {
                kind: ActivityActorKind::Agent,
                id: "agent-1".into(),
                label: "Builder".into(),
            },
            verb: "ran".into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Command,
                id: Some("command-1".into()),
                label: "cargo test".into(),
            },
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: None,
            },
            lifecycle: ActivityLifecycle::Running,
            occurred_at: timestamp,
            projected_at: timestamp,
            context: ActivityContext {
                project_id: Some("project-1".into()),
                thread_id: Some("thread-1".into()),
                session_id: Some("session-1".into()),
                ..ActivityContext::default()
            },
            visibility: ActivityVisibility::Project,
            details: Some(ActivityDetailHandle::AcpEntry {
                session_id: "session-1".into(),
                entry_id: "entry-1".into(),
            }),
            links: vec![ActivityLink::Action {
                action_id: "action-1".into(),
            }],
        }
    }

    #[test]
    fn activity_projection_contract_keeps_identity_stable_across_versions() {
        let id = ActivityItemId::new(ActivitySourceKind::Acp, "session-1/tool-call-1")
            .expect("non-empty source id should be valid");
        let running = test_item(id.clone(), 1);
        let mut completed = test_item(id, 2);
        completed.lifecycle = ActivityLifecycle::Succeeded;
        completed.outcome = ActivityOutcome {
            status: ActivityOutcomeStatus::Success,
            summary: Some("12 tests passed".into()),
        };

        assert_eq!(running.id, completed.id);
        assert_ne!(running.source_version, completed.source_version);
        assert!(!running.lifecycle.is_terminal());
        assert!(completed.lifecycle.is_terminal());
    }

    #[test]
    fn activity_projection_contract_round_trips_serializable_detail_handles() {
        let handles = vec![
            ActivityDetailHandle::AcpEntry {
                session_id: "session-1".into(),
                entry_id: "entry-1".into(),
            },
            ActivityDetailHandle::NativeAction {
                action_id: "action-1".into(),
            },
            ActivityDetailHandle::ProtocolEvent {
                event_id: "event-1".into(),
            },
            ActivityDetailHandle::GitChange {
                repository_id: "repository-1".into(),
                change_id: "change-1".into(),
            },
            ActivityDetailHandle::WorkflowRun {
                run_id: "run-1".into(),
                step_id: Some("step-1".into()),
            },
            ActivityDetailHandle::RawSource {
                item_id: ActivityItemId::new(ActivitySourceKind::System, "raw-1")
                    .expect("non-empty source id should be valid"),
            },
        ];

        let json = serde_json::to_string(&handles).expect("detail handles should serialize");
        let decoded =
            serde_json::from_str::<Vec<ActivityDetailHandle>>(&json).expect("valid round trip");
        assert_eq!(decoded, handles);
    }

    #[test]
    fn activity_projection_contract_round_trips_complete_item() {
        let id = ActivityItemId::new(ActivitySourceKind::Acp, "session-1/tool-call-1")
            .expect("non-empty source id should be valid");
        let item = test_item(id, 1);

        let json = serde_json::to_string(&item).expect("activity item should serialize");
        let decoded = serde_json::from_str::<ActivityItem>(&json).expect("valid round trip");
        assert_eq!(decoded, item);
    }

    #[test]
    fn activity_projection_contract_rejects_empty_source_identity() {
        assert_eq!(
            ActivityItemId::new(ActivitySourceKind::System, "  "),
            Err(ActivityProjectionContractError::EmptySourceId)
        );
    }
}
