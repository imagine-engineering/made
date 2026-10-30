use std::{error::Error, fmt};

use chrono::{DateTime, Utc};
use collaboration_domain::{
    AggregateVersion, Job, JobState, Message, MessageAuthor, MessageLifecycleState, NostrEventId,
    PresenceSnapshot, PresenceStatus, PresenceSubject, PrincipalId,
};

use crate::activity_projection::{
    ActivityActor, ActivityActorKind, ActivityContext, ActivityDetailHandle, ActivityItem,
    ActivityItemId, ActivityLifecycle, ActivityLink, ActivityObject, ActivityObjectKind,
    ActivityOutcome, ActivityOutcomeStatus, ActivityProjectionContractError, ActivitySemanticClass,
    ActivitySourceKind, ActivityVisibility,
};

const MAX_ACTOR_LABEL_BYTES: usize = 1_024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationActorPresentation {
    pub principal_id: PrincipalId,
    pub kind: ActivityActorKind,
    pub label: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CollaborationActivityScope {
    pub project_id: Option<String>,
    pub thread_id: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageActivityContext {
    pub author: CollaborationActorPresentation,
    pub reply_to_event_id: Option<NostrEventId>,
    pub projected_at: DateTime<Utc>,
    pub scope: CollaborationActivityScope,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresenceActivityContext {
    pub actor: CollaborationActorPresentation,
    pub source_version: AggregateVersion,
    pub observed_at_millis: u64,
    pub projected_at: DateTime<Utc>,
    pub waiting_for_user: bool,
    pub scope: CollaborationActivityScope,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobActivityContext {
    pub requester: CollaborationActorPresentation,
    pub executor: CollaborationActorPresentation,
    pub transition_actor: Option<CollaborationActorPresentation>,
    pub projected_at: DateTime<Utc>,
    pub scope: CollaborationActivityScope,
}

pub fn project_message_activity(
    message: &Message,
    context: &MessageActivityContext,
) -> Result<ActivityItem, CollaborationActivityProjectionError> {
    let fields = message.fields();
    let author_principal_id = fields.author.principal_id();
    validate_actor(&context.author, author_principal_id)?;
    match fields.author {
        MessageAuthor::Principal(_) => validate_human_or_agent(&context.author)?,
        MessageAuthor::OwnerAttestedAgent { .. }
            if context.author.kind != ActivityActorKind::Agent =>
        {
            return Err(CollaborationActivityProjectionError::ActorKindMismatch);
        }
        MessageAuthor::OwnerAttestedAgent { .. } => {}
    }

    let source_event_id = nostr_event_id(fields.source.event_id);
    let current_event_id = nostr_event_id(fields.current_source.event_id);
    let reply_to_event_id = context.reply_to_event_id.map(nostr_event_id);
    let lifecycle_state = fields.lifecycle_state;
    let content = message.visible_content().map(|content| content.as_str());
    let occurred_at = timestamp_seconds(fields.source.event_created_at)?;

    let mut outcome_details = Vec::new();
    if let Some(reply_to_event_id) = &reply_to_event_id {
        outcome_details.push(format!("Reply to {reply_to_event_id}"));
    }
    if lifecycle_state == MessageLifecycleState::Edited {
        outcome_details.push("Edited".into());
    }

    let mut scope = context.scope.clone();
    if scope.thread_id.is_none() {
        scope.thread_id = Some(fields.channel_id.to_string());
    }
    Ok(ActivityItem {
        id: ActivityItemId::new(ActivitySourceKind::Nostr, source_event_id.clone())?,
        source_version: fields.version.get(),
        class: ActivitySemanticClass::Message,
        actor: project_actor(&context.author),
        verb: match lifecycle_state {
            MessageLifecycleState::Deleted => "deleted".into(),
            _ if reply_to_event_id.is_some() => "replied with".into(),
            MessageLifecycleState::Edited => "edited".into(),
            MessageLifecycleState::Active => "said".into(),
        },
        object: ActivityObject {
            kind: ActivityObjectKind::Message,
            id: Some(source_event_id.clone()),
            label: content.unwrap_or("Deleted message").into(),
        },
        outcome: ActivityOutcome {
            status: ActivityOutcomeStatus::Success,
            summary: (!outcome_details.is_empty()).then(|| outcome_details.join(" · ")),
        },
        lifecycle: ActivityLifecycle::Succeeded,
        occurred_at,
        projected_at: context.projected_at,
        context: project_context(fields.community_id.to_string(), scope),
        visibility: ActivityVisibility::Community,
        details: Some(ActivityDetailHandle::ProtocolEvent {
            event_id: current_event_id,
        }),
        links: message_links(&source_event_id, reply_to_event_id.as_deref()),
    })
}

pub fn project_presence_activity(
    subject: PresenceSubject,
    snapshot: PresenceSnapshot,
    context: &PresenceActivityContext,
) -> Result<ActivityItem, CollaborationActivityProjectionError> {
    validate_actor(&context.actor, subject.principal_id)?;
    validate_human_or_agent(&context.actor)?;
    let occurred_at = timestamp_millis(context.observed_at_millis)?;
    let source_id = format!("presence:{}:{}", subject.community_id, subject.principal_id);
    let waiting_for_user = context.waiting_for_user && snapshot.status != PresenceStatus::Offline;
    let (verb, lifecycle, outcome) = if waiting_for_user {
        (
            "is waiting for",
            ActivityLifecycle::WaitingForUser,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Waiting for a response".into()),
            },
        )
    } else {
        match snapshot.status {
            PresenceStatus::Online => (
                "is",
                ActivityLifecycle::Running,
                ActivityOutcome {
                    status: ActivityOutcomeStatus::NoChange,
                    summary: Some("Online".into()),
                },
            ),
            PresenceStatus::Away => (
                "is",
                ActivityLifecycle::Idle,
                ActivityOutcome {
                    status: ActivityOutcomeStatus::NoChange,
                    summary: Some("Away".into()),
                },
            ),
            PresenceStatus::Offline => (
                "is",
                ActivityLifecycle::Disconnected,
                ActivityOutcome {
                    status: ActivityOutcomeStatus::NoChange,
                    summary: Some("Offline".into()),
                },
            ),
        }
    };

    Ok(ActivityItem {
        id: ActivityItemId::new(ActivitySourceKind::System, source_id)?,
        source_version: context.source_version.get(),
        class: ActivitySemanticClass::Lifecycle,
        actor: project_actor(&context.actor),
        verb: verb.into(),
        object: ActivityObject {
            kind: if waiting_for_user {
                ActivityObjectKind::Message
            } else {
                ActivityObjectKind::Identity
            },
            id: Some(subject.principal_id.to_string()),
            label: if waiting_for_user {
                "a response".into()
            } else {
                "available".into()
            },
        },
        outcome,
        lifecycle,
        occurred_at,
        projected_at: context.projected_at,
        context: project_context(subject.community_id.to_string(), context.scope.clone()),
        visibility: ActivityVisibility::Community,
        details: None,
        links: vec![ActivityLink::Entity {
            entity_kind: "principal".into(),
            entity_id: subject.principal_id.to_string(),
        }],
    })
}

pub fn project_job_activity(
    job: &Job,
    context: &JobActivityContext,
) -> Result<ActivityItem, CollaborationActivityProjectionError> {
    validate_actor(&context.requester, job.requester_principal_id())?;
    validate_actor(&context.executor, job.target_executor_principal_id())?;
    validate_human_or_agent(&context.requester)?;
    validate_human_or_agent(&context.executor)?;
    if context.executor.kind != ActivityActorKind::Agent {
        return Err(CollaborationActivityProjectionError::ActorKindMismatch);
    }
    if let Some(transition_actor) = &context.transition_actor {
        validate_actor_label(transition_actor)?;
        validate_human_or_agent(transition_actor)?;
    }

    let (actor, verb, lifecycle, outcome) = match job.state() {
        JobState::Requested => (
            &context.requester,
            "requested",
            ActivityLifecycle::Pending,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Waiting for the delegated agent".into()),
            },
        ),
        JobState::Accepted {
            executor_principal_id,
        } => (
            actor_for(context, executor_principal_id)?,
            "accepted",
            ActivityLifecycle::Pending,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Waiting to start".into()),
            },
        ),
        JobState::InProgress {
            executor_principal_id,
        } => (
            actor_for(context, executor_principal_id)?,
            "is working on",
            ActivityLifecycle::Running,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("Delegated work is in progress".into()),
            },
        ),
        JobState::Completed {
            executor_principal_id,
        } => (
            actor_for(context, executor_principal_id)?,
            "completed",
            ActivityLifecycle::Succeeded,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Success,
                summary: Some("Delegated work completed".into()),
            },
        ),
        JobState::Cancelled {
            cancelled_by_principal_id,
            ..
        } => (
            actor_for(context, cancelled_by_principal_id)?,
            "cancelled",
            ActivityLifecycle::Cancelled,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Cancelled,
                summary: Some("Delegated work was cancelled".into()),
            },
        ),
        JobState::Failed {
            reported_by_principal_id,
            ..
        } => (
            actor_for(context, reported_by_principal_id)?,
            "reported a failure for",
            ActivityLifecycle::Failed,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Failure,
                summary: Some("Delegated work failed".into()),
            },
        ),
    };

    let identity = job.identity();
    let job_id = identity.job_id().to_string();
    let source_id = format!("job:{}:{job_id}", identity.community_id());
    Ok(ActivityItem {
        id: ActivityItemId::new(ActivitySourceKind::System, source_id)?,
        source_version: job.version().get(),
        class: ActivitySemanticClass::PlatformOperation,
        actor: project_actor(actor),
        verb: verb.into(),
        object: ActivityObject {
            kind: ActivityObjectKind::Workflow,
            id: Some(job_id.clone()),
            label: "delegated work".into(),
        },
        outcome,
        lifecycle,
        occurred_at: timestamp_millis(job.updated_at_millis())?,
        projected_at: context.projected_at,
        context: project_context(identity.community_id().to_string(), context.scope.clone()),
        visibility: ActivityVisibility::Participants,
        details: None,
        links: vec![ActivityLink::Entity {
            entity_kind: "job".into(),
            entity_id: job_id,
        }],
    })
}

fn actor_for(
    context: &JobActivityContext,
    principal_id: PrincipalId,
) -> Result<&CollaborationActorPresentation, CollaborationActivityProjectionError> {
    [&context.requester, &context.executor]
        .into_iter()
        .chain(context.transition_actor.as_ref())
        .find(|actor| actor.principal_id == principal_id)
        .ok_or(CollaborationActivityProjectionError::MissingJobActor)
}

fn validate_actor(
    actor: &CollaborationActorPresentation,
    expected_principal_id: PrincipalId,
) -> Result<(), CollaborationActivityProjectionError> {
    validate_actor_label(actor)?;
    if actor.principal_id != expected_principal_id {
        return Err(CollaborationActivityProjectionError::ActorMismatch);
    }
    Ok(())
}

fn validate_actor_label(
    actor: &CollaborationActorPresentation,
) -> Result<(), CollaborationActivityProjectionError> {
    let label = actor.label.trim();
    if label.is_empty()
        || label.len() > MAX_ACTOR_LABEL_BYTES
        || label.chars().any(char::is_control)
    {
        return Err(CollaborationActivityProjectionError::InvalidActorLabel);
    }
    Ok(())
}

fn validate_human_or_agent(
    actor: &CollaborationActorPresentation,
) -> Result<(), CollaborationActivityProjectionError> {
    if matches!(
        actor.kind,
        ActivityActorKind::Human | ActivityActorKind::Agent
    ) {
        Ok(())
    } else {
        Err(CollaborationActivityProjectionError::ActorKindMismatch)
    }
}

fn project_actor(actor: &CollaborationActorPresentation) -> ActivityActor {
    ActivityActor {
        kind: actor.kind,
        id: actor.principal_id.to_string(),
        label: actor.label.trim().to_owned(),
    }
}

fn project_context(community_id: String, scope: CollaborationActivityScope) -> ActivityContext {
    ActivityContext {
        community_id: Some(community_id),
        project_id: scope.project_id,
        thread_id: scope.thread_id,
        session_id: scope.session_id,
    }
}

fn message_links(source_event_id: &str, reply_to_event_id: Option<&str>) -> Vec<ActivityLink> {
    let mut links = vec![ActivityLink::Entity {
        entity_kind: "message".into(),
        entity_id: source_event_id.into(),
    }];
    if let Some(reply_to_event_id) = reply_to_event_id {
        links.push(ActivityLink::Entity {
            entity_kind: "message".into(),
            entity_id: reply_to_event_id.into(),
        });
    }
    links
}

fn nostr_event_id(event_id: NostrEventId) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in event_id.as_bytes() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn timestamp_seconds(seconds: u64) -> Result<DateTime<Utc>, CollaborationActivityProjectionError> {
    i64::try_from(seconds)
        .ok()
        .and_then(|seconds| DateTime::from_timestamp(seconds, 0))
        .ok_or(CollaborationActivityProjectionError::InvalidTimestamp)
}

fn timestamp_millis(millis: u64) -> Result<DateTime<Utc>, CollaborationActivityProjectionError> {
    i64::try_from(millis)
        .ok()
        .and_then(DateTime::from_timestamp_millis)
        .ok_or(CollaborationActivityProjectionError::InvalidTimestamp)
}

#[derive(Debug)]
pub enum CollaborationActivityProjectionError {
    ActorMismatch,
    ActorKindMismatch,
    InvalidActorLabel,
    MissingJobActor,
    InvalidTimestamp,
    Contract(ActivityProjectionContractError),
}

impl fmt::Display for CollaborationActivityProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActorMismatch => formatter.write_str("collaboration activity actor mismatch"),
            Self::ActorKindMismatch => {
                formatter.write_str("collaboration activity actor kind mismatch")
            }
            Self::InvalidActorLabel => {
                formatter.write_str("collaboration activity actor label is invalid")
            }
            Self::MissingJobActor => {
                formatter.write_str("collaboration job transition actor is missing")
            }
            Self::InvalidTimestamp => formatter
                .write_str("collaboration activity timestamp is outside the supported range"),
            Self::Contract(error) => error.fmt(formatter),
        }
    }
}

impl Error for CollaborationActivityProjectionError {}

impl From<ActivityProjectionContractError> for CollaborationActivityProjectionError {
    fn from(error: ActivityProjectionContractError) -> Self {
        Self::Contract(error)
    }
}

#[cfg(test)]
mod tests {
    use collaboration_domain::{
        AggregateId, CommunityId, JobCommand, JobCommandKind, JobIdentity, MessageContent,
        MessageMutation, MessageMutationKind, MessageRecordFields, MessageSource, OperationId,
        PresenceSources,
    };
    use uuid::Uuid;

    use crate::activity_reducer::{ActivityReducer, ActivityReduction};

    use super::*;

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn actor(value: u128, kind: ActivityActorKind, label: &str) -> CollaborationActorPresentation {
        CollaborationActorPresentation {
            principal_id: principal(value),
            kind,
            label: label.into(),
        }
    }

    fn event(value: u8) -> NostrEventId {
        NostrEventId::from_bytes([value; 32])
    }

    fn projected_at() -> DateTime<Utc> {
        DateTime::from_timestamp(1_800_000_000, 0).expect("test timestamp should be valid")
    }

    fn message(edited: bool) -> Message {
        let author = principal(3);
        let source = MessageSource {
            event_id: event(1),
            event_created_at: 1_700_000_000,
        };
        let mutation_source = MessageSource {
            event_id: event(2),
            event_created_at: 1_700_000_001,
        };
        let mutations = edited
            .then(|| MessageMutation {
                source: mutation_source,
                actor_principal_id: author,
                kind: MessageMutationKind::Edit,
                resulting_version: AggregateVersion::new(2).expect("valid version"),
            })
            .into_iter()
            .collect();
        Message::from_record(MessageRecordFields {
            community_id: CommunityId::from_uuid(Uuid::from_u128(1)),
            channel_id: AggregateId::from_uuid(Uuid::from_u128(2)),
            message_id: AggregateId::from_uuid(Uuid::from_u128(4)),
            author: MessageAuthor::principal(author),
            content: MessageContent::new(if edited { "edited" } else { "original" })
                .expect("valid message"),
            lifecycle_state: if edited {
                MessageLifecycleState::Edited
            } else {
                MessageLifecycleState::Active
            },
            source,
            current_source: if edited { mutation_source } else { source },
            mutations,
            version: AggregateVersion::new(if edited { 2 } else { 1 }).expect("valid version"),
        })
        .expect("valid message record")
    }

    fn message_context(reply_to_event_id: Option<NostrEventId>) -> MessageActivityContext {
        MessageActivityContext {
            author: actor(3, ActivityActorKind::Human, "Avery"),
            reply_to_event_id,
            projected_at: projected_at(),
            scope: CollaborationActivityScope::default(),
        }
    }

    fn job_command(version: u64, kind: JobCommandKind) -> JobCommand {
        JobCommand::new(
            JobIdentity::new(
                CommunityId::from_uuid(Uuid::from_u128(1)),
                AggregateId::from_uuid(Uuid::from_u128(20)),
            )
            .expect("valid identity"),
            OperationId::from_uuid(Uuid::from_u128(30 + u128::from(version))),
            AggregateVersion::new(version).expect("valid version"),
            1_700_000_000_000 + version,
            kind,
        )
        .expect("valid command")
    }

    fn requested_job() -> Job {
        Job::request(job_command(
            1,
            JobCommandKind::Request {
                requester_principal_id: principal(3),
                target_executor_principal_id: principal(4),
            },
        ))
        .expect("valid job")
    }

    fn job_context() -> JobActivityContext {
        JobActivityContext {
            requester: actor(3, ActivityActorKind::Human, "Avery"),
            executor: actor(4, ActivityActorKind::Agent, "Builder"),
            transition_actor: None,
            projected_at: projected_at(),
            scope: CollaborationActivityScope::default(),
        }
    }

    #[test]
    fn message_versions_reconcile_in_place_and_reply_stays_on_one_row() {
        let reply_to = event(9);
        let initial = project_message_activity(&message(false), &message_context(Some(reply_to)))
            .expect("initial message should project");
        let edited = project_message_activity(&message(true), &message_context(Some(reply_to)))
            .expect("edit should project");
        assert_eq!(initial.id, edited.id);
        assert_eq!(initial.links.len(), 2);
        assert_eq!(initial.verb, "replied with");

        let mut reducer = ActivityReducer::new();
        assert!(matches!(
            reducer.reduce(initial.clone()),
            Ok(ActivityReduction::Inserted { index: 0 })
        ));
        assert!(matches!(
            reducer.reduce(initial),
            Ok(ActivityReduction::Duplicate { index: 0 })
        ));
        assert!(matches!(
            reducer.reduce(edited),
            Ok(ActivityReduction::Updated { index: 0 })
        ));
        assert_eq!(reducer.items().len(), 1);
        assert_eq!(reducer.items()[0].object.label, "edited");
    }

    #[test]
    fn presence_waits_and_resumes_on_one_principal_row() {
        let subject = PresenceSubject {
            community_id: CommunityId::from_uuid(Uuid::from_u128(1)),
            principal_id: principal(4),
            nostr_public_key: collaboration_domain::NostrPublicKey::from_bytes([5; 32]),
        };
        let snapshot = PresenceSnapshot {
            status: PresenceStatus::Online,
            active_sources: PresenceSources {
                signed: true,
                room: false,
            },
            refresh_at_millis: Some(1_700_000_100_000),
            membership_version: AggregateVersion::FIRST,
        };
        let waiting = project_presence_activity(
            subject,
            snapshot,
            &PresenceActivityContext {
                actor: actor(4, ActivityActorKind::Agent, "Builder"),
                source_version: AggregateVersion::FIRST,
                observed_at_millis: 1_700_000_000_000,
                projected_at: projected_at(),
                waiting_for_user: true,
                scope: CollaborationActivityScope::default(),
            },
        )
        .expect("waiting presence should project");
        let resumed = project_presence_activity(
            subject,
            snapshot,
            &PresenceActivityContext {
                actor: actor(4, ActivityActorKind::Agent, "Builder"),
                source_version: AggregateVersion::new(2).expect("valid version"),
                observed_at_millis: 1_700_000_001_000,
                projected_at: projected_at(),
                waiting_for_user: false,
                scope: CollaborationActivityScope::default(),
            },
        )
        .expect("resumed presence should project");
        assert_eq!(waiting.id, resumed.id);
        assert_eq!(waiting.lifecycle, ActivityLifecycle::WaitingForUser);
        assert_eq!(resumed.lifecycle, ActivityLifecycle::Running);

        let mut reducer = ActivityReducer::new();
        reducer.reduce(waiting).expect("wait should insert");
        reducer.reduce(resumed).expect("resume should update");
        assert_eq!(reducer.items().len(), 1);
        assert_eq!(reducer.items()[0].lifecycle, ActivityLifecycle::Running);
    }

    #[test]
    fn delegated_job_advances_to_each_terminal_outcome_in_place() {
        for (terminal_command, expected_lifecycle, expected_status) in [
            (
                JobCommandKind::Result {
                    executor_principal_id: principal(4),
                },
                ActivityLifecycle::Succeeded,
                ActivityOutcomeStatus::Success,
            ),
            (
                JobCommandKind::Cancel {
                    actor_principal_id: principal(3),
                },
                ActivityLifecycle::Cancelled,
                ActivityOutcomeStatus::Cancelled,
            ),
            (
                JobCommandKind::Error {
                    actor_principal_id: principal(4),
                },
                ActivityLifecycle::Failed,
                ActivityOutcomeStatus::Failure,
            ),
        ] {
            let mut job = requested_job();
            let requested =
                project_job_activity(&job, &job_context()).expect("requested job should project");
            job.apply(job_command(
                2,
                JobCommandKind::Accept {
                    executor_principal_id: principal(4),
                },
            ))
            .expect("accept should apply");
            job.apply(job_command(3, terminal_command))
                .expect("terminal transition should apply");
            let terminal =
                project_job_activity(&job, &job_context()).expect("terminal job should project");
            assert_eq!(requested.id, terminal.id);
            assert_eq!(terminal.lifecycle, expected_lifecycle);
            assert_eq!(terminal.outcome.status, expected_status);
            let mut reducer = ActivityReducer::new();
            reducer.reduce(requested).expect("request should insert");
            reducer.reduce(terminal).expect("terminal should update");
            assert_eq!(reducer.items().len(), 1);
        }
    }

    #[test]
    fn owner_attested_agent_message_projects_as_agent_and_rejects_human_presentation() {
        let mut fields = message(false).fields().clone();
        fields.author = MessageAuthor::OwnerAttestedAgent {
            agent_principal_id: principal(3),
            owner_principal_id: principal(8),
            proof_event_id: event(7),
        };
        let message = Message::from_record(fields).expect("valid owner-attested message");
        let mut context = message_context(None);
        context.author.kind = ActivityActorKind::Agent;
        let item = project_message_activity(&message, &context)
            .expect("owner-attested agent message should project");
        assert_eq!(item.actor.kind, ActivityActorKind::Agent);

        let error = project_message_activity(&message, &message_context(None))
            .expect_err("agent authorship must not be presented as human");
        assert!(matches!(
            error,
            CollaborationActivityProjectionError::ActorKindMismatch
        ));
    }
}
