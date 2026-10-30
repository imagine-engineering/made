use std::{error::Error, fmt};

use acp_thread::{
    AcpThreadEvent, AssistantMessage, AssistantMessageChunk, ClientUserMessageId, ThreadStatus,
    UserMessage,
};
use agent_client_protocol::schema::v1 as acp;
use chrono::{DateTime, Utc};
use gpui::App;

use crate::activity_projection::{
    ActivityActor, ActivityContext, ActivityDetailHandle, ActivityItem, ActivityItemId,
    ActivityLifecycle, ActivityObject, ActivityObjectKind, ActivityOutcome, ActivityOutcomeStatus,
    ActivityProjectionContractError, ActivitySemanticClass, ActivitySourceKind, ActivityVisibility,
};

#[derive(Clone, Debug)]
pub struct AcpActivityProjectionContext {
    pub session_id: acp::SessionId,
    pub human_actor: ActivityActor,
    pub agent_actor: ActivityActor,
    pub context: ActivityContext,
    pub visibility: ActivityVisibility,
    pub occurred_at: DateTime<Utc>,
    pub projected_at: DateTime<Utc>,
}

impl AcpActivityProjectionContext {
    fn activity_context(&self) -> ActivityContext {
        let mut context = self.context.clone();
        context.session_id = Some(self.session_id.0.to_string());
        context
    }
}

#[derive(Clone, Debug)]
pub struct AcpLifecycleActivity {
    pub event_id: String,
    pub kind: AcpLifecycleKind,
}

#[derive(Clone, Debug)]
pub enum AcpLifecycleKind {
    Started,
    Idle,
    Stopped(acp::StopReason),
    Disconnected { reason: Option<String> },
    Failed { message: String },
}

#[derive(Debug, Eq, PartialEq)]
pub enum AcpActivityProjectionError {
    Contract(ActivityProjectionContractError),
    InvalidClientMessageId,
    InvalidLifecycleEventId,
}

impl fmt::Display for AcpActivityProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => error.fmt(formatter),
            Self::InvalidClientMessageId => {
                formatter.write_str("ACP client message id did not serialize as a string")
            }
            Self::InvalidLifecycleEventId => {
                formatter.write_str("ACP lifecycle event id must not be empty")
            }
        }
    }
}

impl Error for AcpActivityProjectionError {}

impl From<ActivityProjectionContractError> for AcpActivityProjectionError {
    fn from(error: ActivityProjectionContractError) -> Self {
        Self::Contract(error)
    }
}

pub fn project_acp_user_message(
    projection_context: &AcpActivityProjectionContext,
    entry_index: usize,
    source_version: u64,
    message: &UserMessage,
    cx: &App,
) -> Result<ActivityItem, AcpActivityProjectionError> {
    let source_id = user_message_source_id(
        &projection_context.session_id,
        entry_index,
        message.client_id.as_ref(),
        message.protocol_id.as_ref(),
    )?;
    let label = message.content.to_markdown(cx).trim().to_owned();
    project_message(
        projection_context,
        source_id,
        source_version,
        ActivitySemanticClass::Message,
        projection_context.human_actor.clone(),
        "said",
        label,
    )
}

pub fn project_acp_assistant_message(
    projection_context: &AcpActivityProjectionContext,
    entry_index: usize,
    source_version: u64,
    message: &AssistantMessage,
    cx: &App,
) -> Result<Vec<ActivityItem>, AcpActivityProjectionError> {
    message
        .chunks
        .iter()
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let (message_id, block, class, verb, source_label) = match chunk {
                AssistantMessageChunk::Message { id, block } => (
                    id.as_ref(),
                    block,
                    ActivitySemanticClass::Message,
                    "said",
                    "message",
                ),
                AssistantMessageChunk::Thought { id, block } => (
                    id.as_ref(),
                    block,
                    ActivitySemanticClass::Thought,
                    "thought",
                    "thought",
                ),
            };
            let source_id = assistant_message_source_id(
                &projection_context.session_id,
                entry_index,
                chunk_index,
                source_label,
                message_id,
            );
            project_message(
                projection_context,
                source_id,
                source_version,
                class,
                projection_context.agent_actor.clone(),
                verb,
                block.to_markdown(cx).trim().to_owned(),
            )
        })
        .collect()
}

pub fn project_acp_lifecycle(
    projection_context: &AcpActivityProjectionContext,
    source_version: u64,
    event: &AcpLifecycleActivity,
) -> Result<ActivityItem, AcpActivityProjectionError> {
    if event.event_id.trim().is_empty() {
        return Err(AcpActivityProjectionError::InvalidLifecycleEventId);
    }
    let source_id = format!(
        "{}/lifecycle/{}",
        projection_context.session_id.0, event.event_id
    );
    let (class, verb, lifecycle, outcome) = lifecycle_semantics(&event.kind);
    build_item(
        projection_context,
        source_id,
        source_version,
        ActivityItemSemantics {
            class,
            actor: projection_context.agent_actor.clone(),
            verb: verb.into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Session,
                id: Some(projection_context.session_id.0.to_string()),
                label: "agent session".into(),
            },
            outcome,
            lifecycle,
        },
    )
}

pub fn lifecycle_activity_for_thread_event(
    event_id: impl Into<String>,
    status: &ThreadStatus,
    event: &AcpThreadEvent,
) -> Option<AcpLifecycleActivity> {
    let kind = match event {
        AcpThreadEvent::StatusChanged => match status {
            ThreadStatus::Generating => AcpLifecycleKind::Started,
            ThreadStatus::Idle => AcpLifecycleKind::Idle,
        },
        AcpThreadEvent::Stopped(reason) => AcpLifecycleKind::Stopped(*reason),
        AcpThreadEvent::Error => AcpLifecycleKind::Failed {
            message: "Agent session failed".into(),
        },
        AcpThreadEvent::LoadError(error) => AcpLifecycleKind::Failed {
            message: error.to_string(),
        },
        _ => return None,
    };
    Some(AcpLifecycleActivity {
        event_id: event_id.into(),
        kind,
    })
}

fn project_message(
    projection_context: &AcpActivityProjectionContext,
    source_id: String,
    source_version: u64,
    class: ActivitySemanticClass,
    actor: ActivityActor,
    verb: &str,
    label: String,
) -> Result<ActivityItem, AcpActivityProjectionError> {
    build_item(
        projection_context,
        source_id,
        source_version,
        ActivityItemSemantics {
            class,
            actor,
            verb: verb.into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Message,
                id: None,
                label,
            },
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Success,
                summary: None,
            },
            lifecycle: ActivityLifecycle::Succeeded,
        },
    )
}

struct ActivityItemSemantics {
    class: ActivitySemanticClass,
    actor: ActivityActor,
    verb: String,
    object: ActivityObject,
    outcome: ActivityOutcome,
    lifecycle: ActivityLifecycle,
}

fn build_item(
    projection_context: &AcpActivityProjectionContext,
    source_id: String,
    source_version: u64,
    semantics: ActivityItemSemantics,
) -> Result<ActivityItem, AcpActivityProjectionError> {
    let id = ActivityItemId::new(ActivitySourceKind::Acp, source_id)?;
    Ok(ActivityItem {
        details: Some(ActivityDetailHandle::AcpEntry {
            session_id: projection_context.session_id.0.to_string(),
            entry_id: id.source_id().to_owned(),
        }),
        id,
        source_version,
        class: semantics.class,
        actor: semantics.actor,
        verb: semantics.verb,
        object: semantics.object,
        outcome: semantics.outcome,
        lifecycle: semantics.lifecycle,
        occurred_at: projection_context.occurred_at,
        projected_at: projection_context.projected_at,
        context: projection_context.activity_context(),
        visibility: projection_context.visibility,
        links: Vec::new(),
    })
}

fn user_message_source_id(
    session_id: &acp::SessionId,
    entry_index: usize,
    client_id: Option<&ClientUserMessageId>,
    protocol_id: Option<&acp::MessageId>,
) -> Result<String, AcpActivityProjectionError> {
    let message_key = if let Some(client_id) = client_id {
        let value = serde_json::to_value(client_id)
            .map_err(|_| AcpActivityProjectionError::InvalidClientMessageId)?;
        let value = value
            .as_str()
            .ok_or(AcpActivityProjectionError::InvalidClientMessageId)?;
        format!("client/{value}")
    } else if let Some(protocol_id) = protocol_id {
        format!("protocol/{}", protocol_id.0)
    } else {
        format!("entry/{entry_index}")
    };
    Ok(format!("{}/human/message/{message_key}", session_id.0))
}

fn assistant_message_source_id(
    session_id: &acp::SessionId,
    entry_index: usize,
    chunk_index: usize,
    source_label: &str,
    protocol_id: Option<&acp::MessageId>,
) -> String {
    let message_key = protocol_id.map_or_else(
        || format!("entry/{entry_index}/chunk/{chunk_index}"),
        |protocol_id| format!("protocol/{}", protocol_id.0),
    );
    format!("{}/agent/{source_label}/{message_key}", session_id.0)
}

fn lifecycle_semantics(
    kind: &AcpLifecycleKind,
) -> (
    ActivitySemanticClass,
    &'static str,
    ActivityLifecycle,
    ActivityOutcome,
) {
    match kind {
        AcpLifecycleKind::Started => (
            ActivitySemanticClass::Lifecycle,
            "started",
            ActivityLifecycle::Running,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Agent session started".into()),
            },
        ),
        AcpLifecycleKind::Idle => (
            ActivitySemanticClass::Lifecycle,
            "is idle",
            ActivityLifecycle::Idle,
            ActivityOutcome {
                status: ActivityOutcomeStatus::NoChange,
                summary: Some("Waiting for work".into()),
            },
        ),
        AcpLifecycleKind::Disconnected { reason } => (
            ActivitySemanticClass::Lifecycle,
            "disconnected",
            ActivityLifecycle::Disconnected,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Failure,
                summary: Some(
                    reason
                        .as_ref()
                        .map_or_else(|| "Connection lost".into(), Clone::clone),
                ),
            },
        ),
        AcpLifecycleKind::Failed { message } => (
            ActivitySemanticClass::Error,
            "failed",
            ActivityLifecycle::Failed,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Failure,
                summary: Some(message.clone()),
            },
        ),
        AcpLifecycleKind::Stopped(reason) => stop_reason_semantics(*reason),
    }
}

fn stop_reason_semantics(
    reason: acp::StopReason,
) -> (
    ActivitySemanticClass,
    &'static str,
    ActivityLifecycle,
    ActivityOutcome,
) {
    let (verb, lifecycle, status, summary) = match reason {
        acp::StopReason::EndTurn => (
            "completed",
            ActivityLifecycle::Succeeded,
            ActivityOutcomeStatus::Success,
            "Turn completed",
        ),
        acp::StopReason::MaxTokens => (
            "stopped",
            ActivityLifecycle::Failed,
            ActivityOutcomeStatus::Failure,
            "Stopped after reaching the token limit",
        ),
        acp::StopReason::MaxTurnRequests => (
            "stopped",
            ActivityLifecycle::Failed,
            ActivityOutcomeStatus::Failure,
            "Stopped after reaching the turn request limit",
        ),
        acp::StopReason::Refusal => (
            "refused",
            ActivityLifecycle::Failed,
            ActivityOutcomeStatus::Failure,
            "Agent refused to continue",
        ),
        acp::StopReason::Cancelled => (
            "cancelled",
            ActivityLifecycle::Cancelled,
            ActivityOutcomeStatus::Cancelled,
            "Cancelled by the user",
        ),
        _ => (
            "stopped",
            ActivityLifecycle::Failed,
            ActivityOutcomeStatus::Unknown,
            "Stopped for an unknown reason",
        ),
    };
    (
        ActivitySemanticClass::Lifecycle,
        verb,
        lifecycle,
        ActivityOutcome {
            status,
            summary: Some(summary.into()),
        },
    )
}

#[cfg(test)]
mod tests {
    use acp_thread::ContentBlock;
    use gpui::TestAppContext;

    use super::*;
    use crate::activity_projection::{ActivityActorKind, ActivitySemanticClass};

    fn projection_context() -> AcpActivityProjectionContext {
        let timestamp = DateTime::parse_from_rfc3339("2026-08-14T12:00:00Z")
            .expect("test timestamp should parse")
            .with_timezone(&Utc);
        AcpActivityProjectionContext {
            session_id: acp::SessionId::new("session-1"),
            human_actor: ActivityActor {
                kind: ActivityActorKind::Human,
                id: "human-1".into(),
                label: "Human".into(),
            },
            agent_actor: ActivityActor {
                kind: ActivityActorKind::Agent,
                id: "agent-1".into(),
                label: "Agent".into(),
            },
            context: ActivityContext {
                project_id: Some("project-1".into()),
                thread_id: Some("thread-1".into()),
                ..ActivityContext::default()
            },
            visibility: ActivityVisibility::Project,
            occurred_at: timestamp,
            projected_at: timestamp,
        }
    }

    fn resource_block(uri: &str) -> ContentBlock {
        ContentBlock::ResourceLink {
            resource_link: acp::ResourceLink::new("fixture", uri),
        }
    }

    #[gpui::test]
    fn activity_acp_mapping_maps_human_message_and_preserves_optimistic_identity(
        cx: &mut TestAppContext,
    ) {
        let client_id = ClientUserMessageId::new();
        let before = UserMessage {
            protocol_id: None,
            client_id: Some(client_id.clone()),
            is_optimistic: true,
            content: resource_block("file:///project/prompt.md"),
            chunks: Vec::new(),
            checkpoint: None,
            indented: false,
        };
        let after = UserMessage {
            protocol_id: Some(acp::MessageId::new("protocol-message-1")),
            client_id: Some(client_id),
            is_optimistic: true,
            content: resource_block("file:///project/prompt.md"),
            chunks: Vec::new(),
            checkpoint: None,
            indented: false,
        };

        cx.update(|cx| {
            let before = project_acp_user_message(&projection_context(), 0, 1, &before, cx)
                .expect("optimistic message should project");
            let after = project_acp_user_message(&projection_context(), 0, 2, &after, cx)
                .expect("acknowledged message should project");

            assert_eq!(before.id, after.id);
            assert_eq!(before.actor.kind, ActivityActorKind::Human);
            assert_eq!(before.class, ActivitySemanticClass::Message);
            assert_eq!(before.object.label, "file:///project/prompt.md");
            assert_eq!(before.context.session_id.as_deref(), Some("session-1"));
        });
    }

    #[gpui::test]
    fn activity_acp_mapping_maps_agent_message_and_thought_once_each(cx: &mut TestAppContext) {
        let message = AssistantMessage {
            chunks: vec![
                AssistantMessageChunk::Message {
                    id: Some(acp::MessageId::new("message-1")),
                    block: resource_block("file:///project/result.md"),
                },
                AssistantMessageChunk::Thought {
                    id: Some(acp::MessageId::new("thought-1")),
                    block: resource_block("file:///project/reasoning.md"),
                },
            ],
            indented: false,
            is_subagent_output: false,
        };

        cx.update(|cx| {
            let items = project_acp_assistant_message(&projection_context(), 1, 1, &message, cx)
                .expect("assistant message should project");
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].class, ActivitySemanticClass::Message);
            assert_eq!(items[0].verb, "said");
            assert_eq!(items[1].class, ActivitySemanticClass::Thought);
            assert_eq!(items[1].verb, "thought");
            assert_ne!(items[0].id, items[1].id);
            assert!(
                items
                    .iter()
                    .all(|item| item.actor.kind == ActivityActorKind::Agent)
            );
        });
    }

    #[test]
    fn activity_acp_mapping_exhausts_lifecycle_and_stop_fixtures() {
        let fixtures = [
            (
                AcpLifecycleKind::Started,
                ActivityLifecycle::Running,
                ActivityOutcomeStatus::Pending,
            ),
            (
                AcpLifecycleKind::Idle,
                ActivityLifecycle::Idle,
                ActivityOutcomeStatus::NoChange,
            ),
            (
                AcpLifecycleKind::Disconnected {
                    reason: Some("relay unavailable".into()),
                },
                ActivityLifecycle::Disconnected,
                ActivityOutcomeStatus::Failure,
            ),
            (
                AcpLifecycleKind::Failed {
                    message: "provider exited".into(),
                },
                ActivityLifecycle::Failed,
                ActivityOutcomeStatus::Failure,
            ),
            (
                AcpLifecycleKind::Stopped(acp::StopReason::EndTurn),
                ActivityLifecycle::Succeeded,
                ActivityOutcomeStatus::Success,
            ),
            (
                AcpLifecycleKind::Stopped(acp::StopReason::MaxTokens),
                ActivityLifecycle::Failed,
                ActivityOutcomeStatus::Failure,
            ),
            (
                AcpLifecycleKind::Stopped(acp::StopReason::MaxTurnRequests),
                ActivityLifecycle::Failed,
                ActivityOutcomeStatus::Failure,
            ),
            (
                AcpLifecycleKind::Stopped(acp::StopReason::Refusal),
                ActivityLifecycle::Failed,
                ActivityOutcomeStatus::Failure,
            ),
            (
                AcpLifecycleKind::Stopped(acp::StopReason::Cancelled),
                ActivityLifecycle::Cancelled,
                ActivityOutcomeStatus::Cancelled,
            ),
        ];

        let mut item_ids = std::collections::HashSet::new();
        for (index, (kind, expected_lifecycle, expected_outcome)) in
            fixtures.into_iter().enumerate()
        {
            let item = project_acp_lifecycle(
                &projection_context(),
                1,
                &AcpLifecycleActivity {
                    event_id: format!("event-{index}"),
                    kind,
                },
            )
            .expect("lifecycle fixture should project");
            assert_eq!(item.lifecycle, expected_lifecycle);
            assert_eq!(item.outcome.status, expected_outcome);
            assert!(item_ids.insert(item.id));
        }
    }

    #[test]
    fn activity_acp_mapping_keeps_lifecycle_identity_stable_across_updates() {
        let event = lifecycle_activity_for_thread_event(
            "turn-1",
            &ThreadStatus::Generating,
            &AcpThreadEvent::StatusChanged,
        )
        .expect("status change should map");
        let started = project_acp_lifecycle(&projection_context(), 1, &event)
            .expect("started lifecycle should project");
        let stopped_event = lifecycle_activity_for_thread_event(
            "turn-1",
            &ThreadStatus::Idle,
            &AcpThreadEvent::Stopped(acp::StopReason::EndTurn),
        )
        .expect("stopped event should map");
        let completed = project_acp_lifecycle(&projection_context(), 2, &stopped_event)
            .expect("completed lifecycle should project");

        assert_eq!(started.id, completed.id);
        assert_ne!(started.source_version, completed.source_version);
        assert_eq!(completed.lifecycle, ActivityLifecycle::Succeeded);
        assert!(
            lifecycle_activity_for_thread_event(
                "entry-1",
                &ThreadStatus::Idle,
                &AcpThreadEvent::NewEntry,
            )
            .is_none()
        );
        assert_eq!(
            project_acp_lifecycle(
                &projection_context(),
                1,
                &AcpLifecycleActivity {
                    event_id: "  ".into(),
                    kind: AcpLifecycleKind::Idle,
                },
            ),
            Err(AcpActivityProjectionError::InvalidLifecycleEventId)
        );
    }
}
