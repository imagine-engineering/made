use std::{error::Error, fmt};

use chrono::{DateTime, Utc};
use collaboration_domain::CommunityId;

use crate::{
    activity_git::{
        CodeActivityProjectionContext, CodeActivityProjectionError, CollaborationCodeActivity,
        project_code_activity,
    },
    activity_projection::{
        ActivityActor, ActivityActorKind, ActivityContext, ActivityDetailHandle, ActivityItem,
        ActivityItemId, ActivityLifecycle, ActivityLink, ActivityObject, ActivityObjectKind,
        ActivityOutcome, ActivityOutcomeStatus, ActivityProjectionContractError,
        ActivitySemanticClass, ActivitySourceKind, ActivityVisibility,
    },
};

pub const PLATFORM_ACTIVITY_SCHEMA_VERSION: u16 = 1;
const MAX_PRESENTATION_FIELD_BYTES: usize = 1_024;

#[derive(Clone, Debug)]
pub struct PlatformActivityProjectionContext {
    pub actor_kind: ActivityActorKind,
    pub actor_label: String,
    pub project_id: Option<String>,
    pub thread_id: Option<String>,
    pub session_id: Option<String>,
    pub visibility: ActivityVisibility,
    pub projected_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlatformActivityRecord {
    pub source_kind: ActivitySourceKind,
    pub source_id: String,
    pub source_version: u64,
    pub schema_version: u16,
    pub actor_id: String,
    pub community_id: Option<CommunityId>,
    pub event_kind: PlatformEventKind,
    pub object_id: Option<String>,
    pub object_label: String,
    pub occurred_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlatformEventKind {
    Registered(RegisteredPlatformEventKind),
    Future(String),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegisteredPlatformEventKind {
    WorkflowPending,
    WorkflowRunning,
    WorkflowWaitingApproval,
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowCancelled,
    ModerationReportOpened,
    ModerationReportResolved,
    ModerationReportDismissed,
    ModerationRestrictionApplied,
    ModerationRestrictionLifted,
    SystemConnected,
    SystemDisconnected,
    SystemTimedOut,
    SystemFailed,
}

impl RegisteredPlatformEventKind {
    pub const ALL: [Self; 15] = [
        Self::WorkflowPending,
        Self::WorkflowRunning,
        Self::WorkflowWaitingApproval,
        Self::WorkflowCompleted,
        Self::WorkflowFailed,
        Self::WorkflowCancelled,
        Self::ModerationReportOpened,
        Self::ModerationReportResolved,
        Self::ModerationReportDismissed,
        Self::ModerationRestrictionApplied,
        Self::ModerationRestrictionLifted,
        Self::SystemConnected,
        Self::SystemDisconnected,
        Self::SystemTimedOut,
        Self::SystemFailed,
    ];

    pub const fn source_kind(self) -> ActivitySourceKind {
        match self {
            Self::WorkflowPending
            | Self::WorkflowRunning
            | Self::WorkflowWaitingApproval
            | Self::WorkflowCompleted
            | Self::WorkflowFailed
            | Self::WorkflowCancelled => ActivitySourceKind::Workflow,
            Self::ModerationReportOpened
            | Self::ModerationReportResolved
            | Self::ModerationReportDismissed
            | Self::ModerationRestrictionApplied
            | Self::ModerationRestrictionLifted => ActivitySourceKind::Moderation,
            Self::SystemConnected
            | Self::SystemDisconnected
            | Self::SystemTimedOut
            | Self::SystemFailed => ActivitySourceKind::System,
        }
    }

    pub const fn catalog_name(self) -> &'static str {
        match self {
            Self::WorkflowPending => "workflow_pending",
            Self::WorkflowRunning => "workflow_running",
            Self::WorkflowWaitingApproval => "workflow_waiting_approval",
            Self::WorkflowCompleted => "workflow_completed",
            Self::WorkflowFailed => "workflow_failed",
            Self::WorkflowCancelled => "workflow_cancelled",
            Self::ModerationReportOpened => "moderation_report_opened",
            Self::ModerationReportResolved => "moderation_report_resolved",
            Self::ModerationReportDismissed => "moderation_report_dismissed",
            Self::ModerationRestrictionApplied => "moderation_restriction_applied",
            Self::ModerationRestrictionLifted => "moderation_restriction_lifted",
            Self::SystemConnected => "system_connected",
            Self::SystemDisconnected => "system_disconnected",
            Self::SystemTimedOut => "system_timed_out",
            Self::SystemFailed => "system_failed",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlatformActivity {
    Code(CollaborationCodeActivity),
    Platform(PlatformActivityRecord),
}

pub fn project_platform_activity(
    context: &PlatformActivityProjectionContext,
    activity: &PlatformActivity,
) -> Result<ActivityItem, PlatformActivityProjectionError> {
    match activity {
        PlatformActivity::Code(activity) => project_code_activity(
            &CodeActivityProjectionContext {
                actor_kind: context.actor_kind,
                actor_label: context.actor_label.clone(),
                project_id: context.project_id.clone(),
                thread_id: context.thread_id.clone(),
                visibility: context.visibility,
                projected_at: context.projected_at,
            },
            activity,
        )
        .map_err(PlatformActivityProjectionError::Code),
        PlatformActivity::Platform(activity) => project_registered_or_fallback(context, activity),
    }
}

fn project_registered_or_fallback(
    context: &PlatformActivityProjectionContext,
    record: &PlatformActivityRecord,
) -> Result<ActivityItem, PlatformActivityProjectionError> {
    validate_context(context)?;
    validate_record(record)?;
    let id = ActivityItemId::new(record.source_kind, record.source_id.clone())?;
    let occurred_at = timestamp(record.occurred_at_millis)?;
    let event_kind_name = match &record.event_kind {
        PlatformEventKind::Registered(kind) => kind.catalog_name(),
        PlatformEventKind::Future(kind) => kind.as_str(),
    };
    let semantics = match record.event_kind {
        PlatformEventKind::Registered(kind)
            if record.schema_version == PLATFORM_ACTIVITY_SCHEMA_VERSION =>
        {
            if kind.source_kind() != record.source_kind {
                return Err(PlatformActivityProjectionError::SourceKindMismatch);
            }
            registered_semantics(kind)
        }
        _ => fallback_semantics(record.schema_version),
    };
    let details = match record.event_kind {
        PlatformEventKind::Registered(kind)
            if record.schema_version == PLATFORM_ACTIVITY_SCHEMA_VERSION
                && kind.source_kind() == ActivitySourceKind::Workflow =>
        {
            Some(ActivityDetailHandle::WorkflowRun {
                run_id: record
                    .object_id
                    .clone()
                    .ok_or(PlatformActivityProjectionError::MissingObjectId)?,
                step_id: None,
            })
        }
        _ => Some(ActivityDetailHandle::RawSource {
            item_id: id.clone(),
        }),
    };
    let links = record
        .object_id
        .as_ref()
        .map(|object_id| ActivityLink::Entity {
            entity_kind: semantics.entity_kind.into(),
            entity_id: object_id.clone(),
        })
        .into_iter()
        .collect();

    Ok(ActivityItem {
        id,
        source_version: record.source_version,
        class: semantics.class,
        actor: ActivityActor {
            kind: context.actor_kind,
            id: record.actor_id.trim().into(),
            label: context.actor_label.trim().into(),
        },
        verb: semantics.verb.into(),
        object: ActivityObject {
            kind: semantics.object_kind,
            id: record.object_id.clone(),
            label: if semantics.use_event_kind_as_label {
                event_kind_name.into()
            } else {
                record.object_label.trim().into()
            },
        },
        outcome: ActivityOutcome {
            status: semantics.outcome_status,
            summary: Some(semantics.outcome_summary.into()),
        },
        lifecycle: semantics.lifecycle,
        occurred_at,
        projected_at: context.projected_at,
        context: ActivityContext {
            community_id: record
                .community_id
                .map(|community_id| community_id.to_string()),
            project_id: context.project_id.clone(),
            thread_id: context.thread_id.clone(),
            session_id: context.session_id.clone(),
        },
        visibility: context.visibility,
        details,
        links,
    })
}

struct PlatformSemantics {
    class: ActivitySemanticClass,
    verb: &'static str,
    object_kind: ActivityObjectKind,
    outcome_status: ActivityOutcomeStatus,
    outcome_summary: &'static str,
    lifecycle: ActivityLifecycle,
    entity_kind: &'static str,
    use_event_kind_as_label: bool,
}

fn registered_semantics(kind: RegisteredPlatformEventKind) -> PlatformSemantics {
    match kind {
        RegisteredPlatformEventKind::WorkflowPending => workflow_semantics(
            "queued",
            ActivityOutcomeStatus::Pending,
            "Workflow is pending",
            ActivityLifecycle::Pending,
        ),
        RegisteredPlatformEventKind::WorkflowRunning => workflow_semantics(
            "is running",
            ActivityOutcomeStatus::Pending,
            "Workflow is running",
            ActivityLifecycle::Running,
        ),
        RegisteredPlatformEventKind::WorkflowWaitingApproval => PlatformSemantics {
            class: ActivitySemanticClass::Permission,
            verb: "is waiting for approval on",
            object_kind: ActivityObjectKind::Workflow,
            outcome_status: ActivityOutcomeStatus::Pending,
            outcome_summary: "Approval required",
            lifecycle: ActivityLifecycle::WaitingForUser,
            entity_kind: "workflow_run",
            use_event_kind_as_label: false,
        },
        RegisteredPlatformEventKind::WorkflowCompleted => workflow_semantics(
            "completed",
            ActivityOutcomeStatus::Success,
            "Workflow completed",
            ActivityLifecycle::Succeeded,
        ),
        RegisteredPlatformEventKind::WorkflowFailed => PlatformSemantics {
            class: ActivitySemanticClass::Error,
            verb: "failed",
            object_kind: ActivityObjectKind::Workflow,
            outcome_status: ActivityOutcomeStatus::Failure,
            outcome_summary: "Workflow failed",
            lifecycle: ActivityLifecycle::Failed,
            entity_kind: "workflow_run",
            use_event_kind_as_label: false,
        },
        RegisteredPlatformEventKind::WorkflowCancelled => workflow_semantics(
            "cancelled",
            ActivityOutcomeStatus::Cancelled,
            "Workflow was cancelled",
            ActivityLifecycle::Cancelled,
        ),
        RegisteredPlatformEventKind::ModerationReportOpened => PlatformSemantics {
            class: ActivitySemanticClass::Permission,
            verb: "opened",
            object_kind: ActivityObjectKind::Other,
            outcome_status: ActivityOutcomeStatus::Pending,
            outcome_summary: "Moderation review required",
            lifecycle: ActivityLifecycle::WaitingForUser,
            entity_kind: "moderation_report",
            use_event_kind_as_label: false,
        },
        RegisteredPlatformEventKind::ModerationReportResolved => moderation_semantics(
            "resolved",
            ActivityObjectKind::Other,
            ActivityOutcomeStatus::Success,
            "Moderation report resolved",
            "moderation_report",
        ),
        RegisteredPlatformEventKind::ModerationReportDismissed => moderation_semantics(
            "dismissed",
            ActivityObjectKind::Other,
            ActivityOutcomeStatus::NoChange,
            "Moderation report dismissed",
            "moderation_report",
        ),
        RegisteredPlatformEventKind::ModerationRestrictionApplied => moderation_semantics(
            "restricted",
            ActivityObjectKind::Identity,
            ActivityOutcomeStatus::Success,
            "Community restriction applied",
            "principal",
        ),
        RegisteredPlatformEventKind::ModerationRestrictionLifted => moderation_semantics(
            "lifted restriction on",
            ActivityObjectKind::Identity,
            ActivityOutcomeStatus::Success,
            "Community restriction lifted",
            "principal",
        ),
        RegisteredPlatformEventKind::SystemConnected => system_semantics(
            ActivitySemanticClass::Lifecycle,
            "connected to",
            ActivityOutcomeStatus::NoChange,
            "System connected",
            ActivityLifecycle::Running,
        ),
        RegisteredPlatformEventKind::SystemDisconnected => system_semantics(
            ActivitySemanticClass::Error,
            "disconnected from",
            ActivityOutcomeStatus::Failure,
            "System disconnected",
            ActivityLifecycle::Disconnected,
        ),
        RegisteredPlatformEventKind::SystemTimedOut => system_semantics(
            ActivitySemanticClass::Error,
            "timed out on",
            ActivityOutcomeStatus::TimedOut,
            "System operation timed out",
            ActivityLifecycle::TimedOut,
        ),
        RegisteredPlatformEventKind::SystemFailed => system_semantics(
            ActivitySemanticClass::Error,
            "failed on",
            ActivityOutcomeStatus::Failure,
            "System operation failed",
            ActivityLifecycle::Failed,
        ),
    }
}

fn workflow_semantics(
    verb: &'static str,
    outcome_status: ActivityOutcomeStatus,
    outcome_summary: &'static str,
    lifecycle: ActivityLifecycle,
) -> PlatformSemantics {
    PlatformSemantics {
        class: ActivitySemanticClass::PlatformOperation,
        verb,
        object_kind: ActivityObjectKind::Workflow,
        outcome_status,
        outcome_summary,
        lifecycle,
        entity_kind: "workflow_run",
        use_event_kind_as_label: false,
    }
}

fn moderation_semantics(
    verb: &'static str,
    object_kind: ActivityObjectKind,
    outcome_status: ActivityOutcomeStatus,
    outcome_summary: &'static str,
    entity_kind: &'static str,
) -> PlatformSemantics {
    PlatformSemantics {
        class: ActivitySemanticClass::Permission,
        verb,
        object_kind,
        outcome_status,
        outcome_summary,
        lifecycle: ActivityLifecycle::Succeeded,
        entity_kind,
        use_event_kind_as_label: false,
    }
}

fn system_semantics(
    class: ActivitySemanticClass,
    verb: &'static str,
    outcome_status: ActivityOutcomeStatus,
    outcome_summary: &'static str,
    lifecycle: ActivityLifecycle,
) -> PlatformSemantics {
    PlatformSemantics {
        class,
        verb,
        object_kind: ActivityObjectKind::Other,
        outcome_status,
        outcome_summary,
        lifecycle,
        entity_kind: "system",
        use_event_kind_as_label: false,
    }
}

fn fallback_semantics(schema_version: u16) -> PlatformSemantics {
    PlatformSemantics {
        class: ActivitySemanticClass::Generic,
        verb: "reported",
        object_kind: ActivityObjectKind::Other,
        outcome_status: ActivityOutcomeStatus::Unknown,
        outcome_summary: if schema_version == PLATFORM_ACTIVITY_SCHEMA_VERSION {
            "Unsupported platform activity kind"
        } else {
            "Unsupported platform activity version"
        },
        lifecycle: ActivityLifecycle::Succeeded,
        entity_kind: "platform_activity",
        use_event_kind_as_label: true,
    }
}

fn validate_context(
    context: &PlatformActivityProjectionContext,
) -> Result<(), PlatformActivityProjectionError> {
    if !valid_field(&context.actor_label) {
        return Err(PlatformActivityProjectionError::InvalidContext);
    }
    Ok(())
}

fn validate_record(record: &PlatformActivityRecord) -> Result<(), PlatformActivityProjectionError> {
    if !matches!(
        record.source_kind,
        ActivitySourceKind::Workflow | ActivitySourceKind::Moderation | ActivitySourceKind::System
    ) || record.source_version == 0
        || record.schema_version == 0
        || matches!(
            record.source_kind,
            ActivitySourceKind::Workflow | ActivitySourceKind::Moderation
        ) && record.community_id.is_none()
        || record
            .community_id
            .is_some_and(|community_id| community_id.as_uuid().is_nil())
        || !valid_field(&record.source_id)
        || !valid_field(&record.actor_id)
        || !valid_field(&record.object_label)
        || record.object_id.as_ref().is_some_and(|id| !valid_field(id))
        || matches!(&record.event_kind, PlatformEventKind::Future(kind) if !valid_field(kind))
    {
        return Err(PlatformActivityProjectionError::InvalidRecord);
    }
    if matches!(
        record.event_kind,
        PlatformEventKind::Registered(kind)
            if record.schema_version == PLATFORM_ACTIVITY_SCHEMA_VERSION
                && matches!(
                    kind.source_kind(),
                    ActivitySourceKind::Workflow | ActivitySourceKind::Moderation
                )
                && record.object_id.is_none()
    ) {
        return Err(PlatformActivityProjectionError::MissingObjectId);
    }
    Ok(())
}

fn valid_field(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && value.len() <= MAX_PRESENTATION_FIELD_BYTES
        && !value.chars().any(char::is_control)
}

fn timestamp(millis: u64) -> Result<DateTime<Utc>, PlatformActivityProjectionError> {
    i64::try_from(millis)
        .ok()
        .and_then(DateTime::from_timestamp_millis)
        .ok_or(PlatformActivityProjectionError::InvalidTimestamp)
}

#[derive(Debug)]
pub enum PlatformActivityProjectionError {
    InvalidContext,
    InvalidRecord,
    MissingObjectId,
    SourceKindMismatch,
    InvalidTimestamp,
    Contract(ActivityProjectionContractError),
    Code(CodeActivityProjectionError),
}

impl fmt::Display for PlatformActivityProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidContext => "platform activity projection context is invalid",
            Self::InvalidRecord => "platform activity record is invalid",
            Self::MissingObjectId => "platform activity object ID is required",
            Self::SourceKindMismatch => "registered platform activity source kind mismatch",
            Self::InvalidTimestamp => "platform activity timestamp is outside the supported range",
            Self::Contract(_) => "platform activity violates the activity projection contract",
            Self::Code(_) => "code activity projection failed",
        };
        formatter.write_str(message)
    }
}

impl Error for PlatformActivityProjectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            Self::Code(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ActivityProjectionContractError> for PlatformActivityProjectionError {
    fn from(error: ActivityProjectionContractError) -> Self {
        Self::Contract(error)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use collaboration_domain::AggregateId;
    use uuid::Uuid;

    use crate::activity_reducer::{ActivityReducer, ActivityReduction};

    use super::*;
    use crate::activity_git::GenericCodeActivity;

    fn context() -> PlatformActivityProjectionContext {
        PlatformActivityProjectionContext {
            actor_kind: ActivityActorKind::Service,
            actor_label: "Collaboration service".into(),
            project_id: Some("project-1".into()),
            thread_id: Some("thread-1".into()),
            session_id: None,
            visibility: ActivityVisibility::Community,
            projected_at: DateTime::from_timestamp_millis(1_900_000_010_000)
                .expect("valid timestamp"),
        }
    }

    fn record(index: usize, kind: RegisteredPlatformEventKind) -> PlatformActivityRecord {
        PlatformActivityRecord {
            source_kind: kind.source_kind(),
            source_id: format!("platform-{index}"),
            source_version: 1,
            schema_version: PLATFORM_ACTIVITY_SCHEMA_VERSION,
            actor_id: "service-1".into(),
            community_id: Some(CommunityId::from_uuid(Uuid::from_u128(1))),
            event_kind: PlatformEventKind::Registered(kind),
            object_id: Some(format!("object-{index}")),
            object_label: format!("Platform object {index}"),
            occurred_at_millis: 1_900_000_000_000 + index as u64,
        }
    }

    #[test]
    fn platform_mapping_catalog_handles_every_registered_kind_exactly_once() {
        let mut reducer = ActivityReducer::new();
        let mut names = HashSet::new();
        for (index, kind) in RegisteredPlatformEventKind::ALL.into_iter().enumerate() {
            assert!(names.insert(kind.catalog_name()));
            let item = project_platform_activity(
                &context(),
                &PlatformActivity::Platform(record(index, kind)),
            )
            .expect("registered platform event should project");
            assert_ne!(item.class, ActivitySemanticClass::Generic);
            assert_ne!(item.class, ActivitySemanticClass::Raw);
            assert_eq!(item.id.source_kind(), kind.source_kind());
            assert!(matches!(
                reducer.reduce(item.clone()).expect("first delivery"),
                ActivityReduction::Inserted { .. }
            ));
            assert!(matches!(
                reducer.reduce(item).expect("duplicate delivery"),
                ActivityReduction::Duplicate { .. }
            ));
        }
        assert_eq!(names.len(), RegisteredPlatformEventKind::ALL.len());
        assert_eq!(
            reducer.items().len(),
            RegisteredPlatformEventKind::ALL.len()
        );
    }

    #[test]
    fn platform_consequence_mapping_surfaces_intervention_and_failures() {
        for (kind, class, lifecycle, status) in [
            (
                RegisteredPlatformEventKind::WorkflowWaitingApproval,
                ActivitySemanticClass::Permission,
                ActivityLifecycle::WaitingForUser,
                ActivityOutcomeStatus::Pending,
            ),
            (
                RegisteredPlatformEventKind::ModerationReportOpened,
                ActivitySemanticClass::Permission,
                ActivityLifecycle::WaitingForUser,
                ActivityOutcomeStatus::Pending,
            ),
            (
                RegisteredPlatformEventKind::SystemDisconnected,
                ActivitySemanticClass::Error,
                ActivityLifecycle::Disconnected,
                ActivityOutcomeStatus::Failure,
            ),
            (
                RegisteredPlatformEventKind::SystemTimedOut,
                ActivitySemanticClass::Error,
                ActivityLifecycle::TimedOut,
                ActivityOutcomeStatus::TimedOut,
            ),
        ] {
            let item =
                project_platform_activity(&context(), &PlatformActivity::Platform(record(1, kind)))
                    .expect("consequential event should project");
            assert_eq!(item.class, class);
            assert_eq!(item.lifecycle, lifecycle);
            assert_eq!(item.outcome.status, status);
        }
    }

    #[test]
    fn future_kind_and_schema_version_use_truthful_raw_fallback() {
        let mut future_kind = record(1, RegisteredPlatformEventKind::WorkflowRunning);
        future_kind.event_kind = PlatformEventKind::Future("workflow_paused_v2".into());
        let mut future_version = record(2, RegisteredPlatformEventKind::ModerationReportResolved);
        future_version.schema_version = 2;
        for record in [future_kind, future_version] {
            let item =
                project_platform_activity(&context(), &PlatformActivity::Platform(record.clone()))
                    .expect("future record should degrade to a generic item");
            assert_eq!(item.class, ActivitySemanticClass::Generic);
            assert_eq!(item.outcome.status, ActivityOutcomeStatus::Unknown);
            assert!(matches!(
                item.details,
                Some(ActivityDetailHandle::RawSource { item_id }) if item_id == item.id
            ));
            assert_eq!(
                item.object.label,
                match record.event_kind {
                    PlatformEventKind::Registered(kind) => kind.catalog_name(),
                    PlatformEventKind::Future(ref kind) => kind,
                }
            );
        }

        let mut mismatched = record(3, RegisteredPlatformEventKind::WorkflowRunning);
        mismatched.source_kind = ActivitySourceKind::Moderation;
        assert!(matches!(
            project_platform_activity(&context(), &PlatformActivity::Platform(mismatched)),
            Err(PlatformActivityProjectionError::SourceKindMismatch)
        ));

        let mut unscoped = record(4, RegisteredPlatformEventKind::ModerationReportOpened);
        unscoped.community_id = None;
        assert!(matches!(
            project_platform_activity(&context(), &PlatformActivity::Platform(unscoped)),
            Err(PlatformActivityProjectionError::InvalidRecord)
        ));
    }

    #[test]
    fn code_activity_delegates_to_existing_git_workflow_fallback() {
        let code = CollaborationCodeActivity::Unsupported(GenericCodeActivity {
            source_kind: ActivitySourceKind::Workflow,
            source_id: "future-repository-workflow".into(),
            source_version: 1,
            actor_id: "service-1".into(),
            community_id: CommunityId::from_uuid(Uuid::from_u128(1)),
            repository_id: AggregateId::from_uuid(Uuid::from_u128(2)),
            event_kind: "future_repository_workflow".into(),
            occurred_at_millis: 1_900_000_000_000,
        });
        let expected = project_code_activity(
            &CodeActivityProjectionContext {
                actor_kind: context().actor_kind,
                actor_label: context().actor_label,
                project_id: context().project_id,
                thread_id: context().thread_id,
                visibility: context().visibility,
                projected_at: context().projected_at,
            },
            &code,
        )
        .expect("existing code mapper should project");
        let actual = project_platform_activity(&context(), &PlatformActivity::Code(code))
            .expect("platform mapper should delegate code activity");
        assert_eq!(actual, expected);
    }
}
