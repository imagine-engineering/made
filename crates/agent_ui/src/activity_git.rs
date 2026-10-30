use std::{error::Error, fmt};

use chrono::{DateTime, Utc};
use collaboration_domain::{
    AggregateId, AggregateVersion, BranchCollaborationIdentity, BranchRefName, BranchUpdateKind,
    CiCheckStatus, CiCheckSuite, CommunityId, GitCommitId, PatchRevision, PrincipalId,
    ReviewApproval, ReviewComment, ReviewDecision,
};

use crate::activity_projection::{
    ActivityActor, ActivityActorKind, ActivityContext, ActivityDetailHandle, ActivityItem,
    ActivityItemId, ActivityLifecycle, ActivityLink, ActivityObject, ActivityObjectKind,
    ActivityOutcome, ActivityOutcomeStatus, ActivityProjectionContractError, ActivitySemanticClass,
    ActivitySourceKind, ActivityVisibility,
};

const MAX_FALLBACK_FIELD_BYTES: usize = 1_024;

#[derive(Clone, Debug)]
pub struct CodeActivityProjectionContext {
    pub actor_kind: ActivityActorKind,
    pub actor_label: String,
    pub project_id: Option<String>,
    pub thread_id: Option<String>,
    pub visibility: ActivityVisibility,
    pub projected_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BranchCodeActivity {
    pub event_id: AggregateId,
    pub actor_principal_id: PrincipalId,
    pub branch: BranchCollaborationIdentity,
    pub version: AggregateVersion,
    pub occurred_at_millis: u64,
    pub kind: BranchCodeActivityKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BranchCodeActivityKind {
    Created {
        commit: GitCommitId,
    },
    Updated {
        previous_commit: GitCommitId,
        current_commit: GitCommitId,
        update_kind: BranchUpdateKind,
    },
    Merged {
        source_commit: GitCommitId,
        target_branch: BranchRefName,
        result_commit: GitCommitId,
    },
    Deleted {
        commit: GitCommitId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewDecisionActivity {
    pub approval: ReviewApproval,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericCodeActivity {
    pub source_kind: ActivitySourceKind,
    pub source_id: String,
    pub source_version: u64,
    pub actor_id: String,
    pub community_id: CommunityId,
    pub repository_id: AggregateId,
    pub event_kind: String,
    pub occurred_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborationCodeActivity {
    Branch(BranchCodeActivity),
    PatchSubmitted(PatchRevision),
    ReviewCommented(ReviewComment),
    ReviewDecisionRecorded(ReviewDecisionActivity),
    CiStatusChanged(CiCheckSuite),
    Unsupported(GenericCodeActivity),
}

pub fn project_code_activity(
    context: &CodeActivityProjectionContext,
    event: &CollaborationCodeActivity,
) -> Result<ActivityItem, CodeActivityProjectionError> {
    let parts = projection_parts(context, event)?;
    let occurred_at = timestamp(parts.occurred_at_millis)?;
    let id = ActivityItemId::new(parts.source_kind, parts.source_id)?;
    Ok(ActivityItem {
        id,
        source_version: parts.source_version,
        class: parts.semantics.class,
        actor: ActivityActor {
            kind: parts.actor_kind,
            id: parts.actor_id,
            label: parts.actor_label,
        },
        verb: parts.semantics.verb,
        object: parts.semantics.object,
        outcome: parts.semantics.outcome,
        lifecycle: parts.semantics.lifecycle,
        occurred_at,
        projected_at: context.projected_at,
        context: ActivityContext {
            community_id: Some(parts.community_id.to_string()),
            project_id: context.project_id.clone(),
            thread_id: context.thread_id.clone(),
            session_id: None,
        },
        visibility: context.visibility,
        details: parts.details,
        links: parts.links,
    })
}

struct ProjectionParts {
    source_kind: ActivitySourceKind,
    source_id: String,
    source_version: u64,
    actor_kind: ActivityActorKind,
    actor_id: String,
    actor_label: String,
    community_id: CommunityId,
    occurred_at_millis: u64,
    semantics: ActivitySemantics,
    details: Option<ActivityDetailHandle>,
    links: Vec<ActivityLink>,
}

struct ActivitySemantics {
    class: ActivitySemanticClass,
    verb: String,
    object: ActivityObject,
    outcome: ActivityOutcome,
    lifecycle: ActivityLifecycle,
}

fn projection_parts(
    context: &CodeActivityProjectionContext,
    event: &CollaborationCodeActivity,
) -> Result<ProjectionParts, CodeActivityProjectionError> {
    match event {
        CollaborationCodeActivity::Branch(activity) => project_branch(context, activity),
        CollaborationCodeActivity::PatchSubmitted(revision) => project_patch(context, revision),
        CollaborationCodeActivity::ReviewCommented(comment) => project_comment(context, comment),
        CollaborationCodeActivity::ReviewDecisionRecorded(decision) => {
            project_decision(context, decision)
        }
        CollaborationCodeActivity::CiStatusChanged(suite) => project_ci(suite),
        CollaborationCodeActivity::Unsupported(activity) => project_fallback(context, activity),
    }
}

fn project_branch(
    context: &CodeActivityProjectionContext,
    activity: &BranchCodeActivity,
) -> Result<ProjectionParts, CodeActivityProjectionError> {
    validate_known_actor(context)?;
    if activity.event_id.as_uuid().is_nil() || activity.actor_principal_id.as_uuid().is_nil() {
        return Err(CodeActivityProjectionError::InvalidRecord);
    }
    match &activity.kind {
        BranchCodeActivityKind::Updated {
            previous_commit,
            current_commit,
            ..
        } if previous_commit == current_commit => {
            return Err(CodeActivityProjectionError::InvalidRecord);
        }
        BranchCodeActivityKind::Merged { target_branch, .. }
            if target_branch == activity.branch.branch_ref() =>
        {
            return Err(CodeActivityProjectionError::InvalidRecord);
        }
        _ => {}
    }
    let repository_id = activity.branch.repository_id().to_string();
    let branch_id = branch_id(&activity.branch);
    let branch_label = short_branch(activity.branch.branch_ref()).to_owned();
    let source_id = format!("branch:{}", activity.event_id);
    let mut links = vec![
        entity_link("repository", repository_id.clone()),
        entity_link("branch", branch_id),
        ActivityLink::GitChange {
            repository_id: repository_id.clone(),
            change_id: source_id.clone(),
        },
    ];
    let (verb, outcome, commits) = match &activity.kind {
        BranchCodeActivityKind::Created { commit } => (
            "created".to_owned(),
            successful_outcome(format!("Created at {}", short_commit(commit))),
            vec![commit],
        ),
        BranchCodeActivityKind::Updated {
            previous_commit,
            current_commit,
            update_kind,
        } => {
            let (verb, summary) = match update_kind {
                BranchUpdateKind::FastForward => (
                    "updated".to_owned(),
                    format!(
                        "Fast-forwarded {} to {}",
                        short_commit(previous_commit),
                        short_commit(current_commit)
                    ),
                ),
                BranchUpdateKind::Force => (
                    "force-updated".to_owned(),
                    format!(
                        "Force-updated {} to {}",
                        short_commit(previous_commit),
                        short_commit(current_commit)
                    ),
                ),
            };
            (
                verb,
                successful_outcome(summary),
                vec![previous_commit, current_commit],
            )
        }
        BranchCodeActivityKind::Merged {
            source_commit,
            target_branch,
            result_commit,
        } => (
            "merged".to_owned(),
            successful_outcome(format!(
                "Merged into {} at {}",
                short_branch(target_branch),
                short_commit(result_commit)
            )),
            vec![source_commit, result_commit],
        ),
        BranchCodeActivityKind::Deleted { commit } => (
            "deleted".to_owned(),
            successful_outcome(format!("Deleted at {}", short_commit(commit))),
            vec![commit],
        ),
    };
    links.extend(
        commits
            .into_iter()
            .map(|commit| entity_link("commit", commit.as_str().to_owned())),
    );
    Ok(ProjectionParts {
        source_kind: ActivitySourceKind::Git,
        source_id: source_id.clone(),
        source_version: activity.version.get(),
        actor_kind: context.actor_kind,
        actor_id: activity.actor_principal_id.to_string(),
        actor_label: normalized_actor_label(context)?,
        community_id: activity.branch.community_id(),
        occurred_at_millis: activity.occurred_at_millis,
        semantics: ActivitySemantics {
            class: ActivitySemanticClass::Lifecycle,
            verb,
            object: ActivityObject {
                kind: ActivityObjectKind::Repository,
                id: Some(repository_id.clone()),
                label: branch_label,
            },
            outcome,
            lifecycle: ActivityLifecycle::Succeeded,
        },
        details: Some(ActivityDetailHandle::GitChange {
            repository_id,
            change_id: source_id,
        }),
        links,
    })
}

fn project_patch(
    context: &CodeActivityProjectionContext,
    revision: &PatchRevision,
) -> Result<ProjectionParts, CodeActivityProjectionError> {
    validate_known_actor(context)?;
    if revision.revision_id.as_uuid().is_nil()
        || revision.author_principal_id.as_uuid().is_nil()
        || revision.base_commit == revision.head_commit
    {
        return Err(CodeActivityProjectionError::InvalidRecord);
    }
    let identity = &revision.review;
    let repository_id = identity.repository_id().to_string();
    let change_id = review_change_id(identity.review_id(), revision.number.get());
    let links = review_links(identity.branch(), identity.review_id(), &change_id)
        .into_iter()
        .chain([
            entity_link("revision", revision.revision_id.to_string()),
            entity_link("commit", revision.base_commit.as_str().to_owned()),
            entity_link("commit", revision.head_commit.as_str().to_owned()),
        ])
        .collect();
    Ok(ProjectionParts {
        source_kind: ActivitySourceKind::Git,
        source_id: format!("patch:{}", revision.revision_id),
        source_version: revision.number.get(),
        actor_kind: context.actor_kind,
        actor_id: revision.author_principal_id.to_string(),
        actor_label: normalized_actor_label(context)?,
        community_id: identity.branch().community_id(),
        occurred_at_millis: revision.created_at_millis,
        semantics: ActivitySemantics {
            class: ActivitySemanticClass::FileEdit,
            verb: "submitted".into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Review,
                id: Some(identity.review_id().to_string()),
                label: format!("patch revision {}", revision.number.get()),
            },
            outcome: successful_outcome(format!(
                "Updated review head to {}",
                short_commit(&revision.head_commit)
            )),
            lifecycle: ActivityLifecycle::Succeeded,
        },
        details: Some(ActivityDetailHandle::GitChange {
            repository_id,
            change_id,
        }),
        links,
    })
}

fn project_comment(
    context: &CodeActivityProjectionContext,
    comment: &ReviewComment,
) -> Result<ProjectionParts, CodeActivityProjectionError> {
    validate_known_actor(context)?;
    if comment.comment_id.as_uuid().is_nil()
        || comment.author_principal_id.as_uuid().is_nil()
        || comment.anchor.end_line < comment.anchor.start_line
    {
        return Err(CodeActivityProjectionError::InvalidRecord);
    }
    let identity = &comment.review;
    let repository_id = identity.repository_id().to_string();
    let change_id = review_change_id(identity.review_id(), comment.anchor.revision.get());
    let links = review_links(identity.branch(), identity.review_id(), &change_id)
        .into_iter()
        .chain([
            entity_link("comment", comment.comment_id.to_string()),
            entity_link("commit", comment.anchor.commit.as_str().to_owned()),
            entity_link("file", comment.anchor.file_path.as_str().to_owned()),
            entity_link("hunk", comment.anchor.hunk_id.as_str().to_owned()),
        ])
        .collect();
    Ok(ProjectionParts {
        source_kind: ActivitySourceKind::Git,
        source_id: format!("review-comment:{}", comment.comment_id),
        source_version: 1,
        actor_kind: context.actor_kind,
        actor_id: comment.author_principal_id.to_string(),
        actor_label: normalized_actor_label(context)?,
        community_id: identity.branch().community_id(),
        occurred_at_millis: comment.created_at_millis,
        semantics: ActivitySemantics {
            class: ActivitySemanticClass::Message,
            verb: "commented on".into(),
            object: ActivityObject {
                kind: ActivityObjectKind::File,
                id: Some(comment.anchor.file_path.as_str().to_owned()),
                label: format!(
                    "{}:{}-{}",
                    comment.anchor.file_path.as_str(),
                    comment.anchor.start_line,
                    comment.anchor.end_line
                ),
            },
            outcome: successful_outcome("Review comment recorded".into()),
            lifecycle: ActivityLifecycle::Succeeded,
        },
        details: Some(ActivityDetailHandle::GitChange {
            repository_id,
            change_id,
        }),
        links,
    })
}

fn project_decision(
    context: &CodeActivityProjectionContext,
    decision: &ReviewDecisionActivity,
) -> Result<ProjectionParts, CodeActivityProjectionError> {
    validate_known_actor(context)?;
    let approval = &decision.approval;
    if approval.approval_id.as_uuid().is_nil() || approval.approver_principal_id.as_uuid().is_nil()
    {
        return Err(CodeActivityProjectionError::InvalidRecord);
    }
    let identity = &approval.review;
    let repository_id = identity.repository_id().to_string();
    let change_id = review_change_id(identity.review_id(), approval.revision.get());
    let (verb, current_summary) = match approval.decision {
        ReviewDecision::Approve => ("approved", "Approval recorded"),
        ReviewDecision::RequestChanges => ("requested changes on", "Changes requested"),
    };
    let links = review_links(identity.branch(), identity.review_id(), &change_id)
        .into_iter()
        .chain([
            entity_link("approval", approval.approval_id.to_string()),
            entity_link("commit", approval.head_commit.as_str().to_owned()),
        ])
        .collect();
    Ok(ProjectionParts {
        source_kind: ActivitySourceKind::Git,
        source_id: format!("review-decision:{}", approval.approval_id),
        source_version: 1,
        actor_kind: context.actor_kind,
        actor_id: approval.approver_principal_id.to_string(),
        actor_label: normalized_actor_label(context)?,
        community_id: identity.branch().community_id(),
        occurred_at_millis: approval.created_at_millis,
        semantics: ActivitySemantics {
            class: ActivitySemanticClass::Permission,
            verb: verb.into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Review,
                id: Some(identity.review_id().to_string()),
                label: format!("review revision {}", approval.revision.get()),
            },
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Success,
                summary: Some(current_summary.into()),
            },
            lifecycle: ActivityLifecycle::Succeeded,
        },
        details: Some(ActivityDetailHandle::GitChange {
            repository_id,
            change_id,
        }),
        links,
    })
}

fn project_ci(suite: &CiCheckSuite) -> Result<ProjectionParts, CodeActivityProjectionError> {
    let fields = suite.fields();
    let identity = &fields.identity;
    let review = identity.review();
    let change_id = review_change_id(review.review_id(), identity.revision().get());
    let status = suite.status();
    let (verb, lifecycle, outcome_status, summary) = match status {
        CiCheckStatus::Pending => (
            "queued",
            ActivityLifecycle::Pending,
            ActivityOutcomeStatus::Pending,
            "CI checks are pending",
        ),
        CiCheckStatus::Running => (
            "is running",
            ActivityLifecycle::Running,
            ActivityOutcomeStatus::Pending,
            "CI checks are running",
        ),
        CiCheckStatus::Success => (
            "passed",
            ActivityLifecycle::Succeeded,
            ActivityOutcomeStatus::Success,
            "All CI checks passed",
        ),
        CiCheckStatus::Failure => (
            "failed",
            ActivityLifecycle::Failed,
            ActivityOutcomeStatus::Failure,
            "One or more CI checks failed",
        ),
        CiCheckStatus::Cancelled => (
            "was cancelled",
            ActivityLifecycle::Cancelled,
            ActivityOutcomeStatus::Cancelled,
            "CI checks were cancelled",
        ),
    };
    let links = review_links(review.branch(), review.review_id(), &change_id)
        .into_iter()
        .chain([
            entity_link("check_suite", identity.suite_id().to_string()),
            entity_link("commit", identity.head_commit().as_str().to_owned()),
            entity_link("workflow", fields.workflow.workflow_id.to_string()),
            entity_link("workflow_run", fields.workflow.workflow_run_id.to_string()),
        ])
        .collect();
    Ok(ProjectionParts {
        source_kind: ActivitySourceKind::Ci,
        source_id: format!("ci-suite:{}", identity.suite_id()),
        source_version: fields.version.get(),
        actor_kind: ActivityActorKind::Service,
        actor_id: fields.workflow.workflow_id.to_string(),
        actor_label: fields.workflow.label.as_str().to_owned(),
        community_id: review.branch().community_id(),
        occurred_at_millis: ci_observed_at_millis(suite),
        semantics: ActivitySemantics {
            class: ActivitySemanticClass::PlatformOperation,
            verb: verb.into(),
            object: ActivityObject {
                kind: ActivityObjectKind::TestSuite,
                id: Some(identity.suite_id().to_string()),
                label: fields.workflow.label.as_str().to_owned(),
            },
            outcome: ActivityOutcome {
                status: outcome_status,
                summary: Some(summary.into()),
            },
            lifecycle,
        },
        details: Some(ActivityDetailHandle::WorkflowRun {
            run_id: fields.workflow.workflow_run_id.to_string(),
            step_id: None,
        }),
        links,
    })
}

fn project_fallback(
    context: &CodeActivityProjectionContext,
    activity: &GenericCodeActivity,
) -> Result<ProjectionParts, CodeActivityProjectionError> {
    validate_known_actor(context)?;
    if !matches!(
        activity.source_kind,
        ActivitySourceKind::Git | ActivitySourceKind::Workflow | ActivitySourceKind::Ci
    ) || activity.source_version == 0
        || activity.community_id.as_uuid().is_nil()
        || activity.repository_id.as_uuid().is_nil()
        || !valid_fallback_field(&activity.source_id)
        || !valid_fallback_field(&activity.actor_id)
        || !valid_fallback_field(&activity.event_kind)
    {
        return Err(CodeActivityProjectionError::InvalidFallback);
    }
    let repository_id = activity.repository_id.to_string();
    Ok(ProjectionParts {
        source_kind: activity.source_kind,
        source_id: activity.source_id.clone(),
        source_version: activity.source_version,
        actor_kind: context.actor_kind,
        actor_id: activity.actor_id.clone(),
        actor_label: normalized_actor_label(context)?,
        community_id: activity.community_id,
        occurred_at_millis: activity.occurred_at_millis,
        semantics: ActivitySemantics {
            class: ActivitySemanticClass::Generic,
            verb: "reported".into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Other,
                id: None,
                label: activity.event_kind.clone(),
            },
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Unknown,
                summary: Some("Unsupported code activity kind".into()),
            },
            lifecycle: ActivityLifecycle::Succeeded,
        },
        details: Some(ActivityDetailHandle::RawSource {
            item_id: ActivityItemId::new(activity.source_kind, activity.source_id.clone())?,
        }),
        links: vec![entity_link("repository", repository_id)],
    })
}

fn review_links(
    branch: &BranchCollaborationIdentity,
    review_id: AggregateId,
    change_id: &str,
) -> Vec<ActivityLink> {
    let repository_id = branch.repository_id().to_string();
    vec![
        entity_link("repository", repository_id.clone()),
        entity_link("branch", branch_id(branch)),
        entity_link("review", review_id.to_string()),
        ActivityLink::GitChange {
            repository_id,
            change_id: change_id.to_owned(),
        },
    ]
}

fn entity_link(entity_kind: &str, entity_id: String) -> ActivityLink {
    ActivityLink::Entity {
        entity_kind: entity_kind.into(),
        entity_id,
    }
}

fn branch_id(branch: &BranchCollaborationIdentity) -> String {
    format!(
        "{}:{}:{}",
        branch.repository_id(),
        branch.branch_ref().as_str(),
        branch.generation().get()
    )
}

fn review_change_id(review_id: AggregateId, revision: u64) -> String {
    format!("{review_id}/revision/{revision}")
}

fn short_branch(branch: &BranchRefName) -> &str {
    branch
        .as_str()
        .strip_prefix("refs/heads/")
        .unwrap_or(branch.as_str())
}

fn short_commit(commit: &GitCommitId) -> &str {
    commit.as_str().get(..12).unwrap_or(commit.as_str())
}

fn successful_outcome(summary: String) -> ActivityOutcome {
    ActivityOutcome {
        status: ActivityOutcomeStatus::Success,
        summary: Some(summary),
    }
}

fn ci_observed_at_millis(suite: &CiCheckSuite) -> u64 {
    suite
        .fields()
        .runs
        .iter()
        .flat_map(|run| {
            [
                Some(run.queued_at_millis),
                run.started_at_millis,
                run.completed_at_millis,
            ]
        })
        .flatten()
        .max()
        .unwrap_or(suite.fields().created_at_millis)
}

fn normalized_actor_label(
    context: &CodeActivityProjectionContext,
) -> Result<String, CodeActivityProjectionError> {
    let label = context.actor_label.trim();
    if !valid_display_field(label) {
        return Err(CodeActivityProjectionError::InvalidActorLabel);
    }
    Ok(label.to_owned())
}

fn validate_known_actor(
    context: &CodeActivityProjectionContext,
) -> Result<(), CodeActivityProjectionError> {
    normalized_actor_label(context).map(|_| ())
}

fn valid_fallback_field(value: &str) -> bool {
    valid_display_field(value.trim())
}

fn valid_display_field(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_FALLBACK_FIELD_BYTES
        && !value.chars().any(char::is_control)
}

fn timestamp(millis: u64) -> Result<DateTime<Utc>, CodeActivityProjectionError> {
    i64::try_from(millis)
        .ok()
        .and_then(DateTime::from_timestamp_millis)
        .ok_or(CodeActivityProjectionError::InvalidTimestamp)
}

#[derive(Debug)]
pub enum CodeActivityProjectionError {
    InvalidActorLabel,
    InvalidRecord,
    InvalidFallback,
    InvalidTimestamp,
    Contract(ActivityProjectionContractError),
}

impl fmt::Display for CodeActivityProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidActorLabel => "code activity actor label is invalid",
            Self::InvalidRecord => "code activity record is invalid",
            Self::InvalidFallback => "generic code activity fallback is invalid",
            Self::InvalidTimestamp => "code activity timestamp is outside the supported range",
            Self::Contract(_) => "code activity violates the activity projection contract",
        };
        formatter.write_str(message)
    }
}

impl Error for CodeActivityProjectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ActivityProjectionContractError> for CodeActivityProjectionError {
    fn from(error: ActivityProjectionContractError) -> Self {
        Self::Contract(error)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, num::NonZeroU32};

    use collaboration_domain::{
        BranchGeneration, CiCheckRunCompletionInput, CiCheckRunInput, CiCheckSuiteIdentity,
        CiLabel, CiOutputText, CiWorkflowLink, PatchRevisionNumber, ReviewCommentAnchor,
        ReviewCommentBody, ReviewDiffSide, ReviewFilePath, ReviewHunkId, ReviewIdentity,
    };
    use uuid::Uuid;

    use super::*;
    use crate::activity_reducer::{ActivityReducer, ActivityReduction};

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn commit(value: u64) -> GitCommitId {
        GitCommitId::parse(format!("{value:040x}")).expect("valid commit")
    }

    fn version(value: u64) -> AggregateVersion {
        AggregateVersion::new(value).expect("positive version")
    }

    fn branch() -> BranchCollaborationIdentity {
        BranchCollaborationIdentity::new(
            CommunityId::from_uuid(Uuid::from_u128(1)),
            aggregate(2),
            BranchRefName::parse("refs/heads/feature/activity").expect("valid branch"),
            BranchGeneration::FIRST,
        )
        .expect("valid branch identity")
    }

    fn review_identity() -> ReviewIdentity {
        ReviewIdentity::new(aggregate(3), branch()).expect("valid review identity")
    }

    fn revision() -> PatchRevision {
        PatchRevision {
            revision_id: aggregate(10),
            review: review_identity(),
            number: PatchRevisionNumber::FIRST,
            base_commit: commit(100),
            head_commit: commit(101),
            author_principal_id: principal(20),
            created_at_millis: 1_900_000_000_000,
        }
    }

    fn comment() -> ReviewComment {
        ReviewComment {
            comment_id: aggregate(10),
            review: review_identity(),
            author_principal_id: principal(21),
            body: ReviewCommentBody::new("Please retain the tenant fence").expect("valid body"),
            anchor: ReviewCommentAnchor::new(
                PatchRevisionNumber::FIRST,
                commit(101),
                ReviewFilePath::new("src/activity.rs").expect("valid path"),
                ReviewHunkId::parse("a".repeat(64)).expect("valid hunk"),
                ReviewDiffSide::Head,
                NonZeroU32::new(20).expect("nonzero line"),
                NonZeroU32::new(24).expect("nonzero line"),
            )
            .expect("valid anchor"),
            created_at_millis: 1_900_000_001_000,
        }
    }

    fn approval(id: u128, decision: ReviewDecision) -> ReviewApproval {
        ReviewApproval {
            approval_id: aggregate(id),
            review: review_identity(),
            revision: PatchRevisionNumber::FIRST,
            head_commit: commit(101),
            approver_principal_id: principal(id + 100),
            decision,
            created_at_millis: 1_900_000_002_000 + id as u64,
        }
    }

    fn suite(id: u128, status: CiCheckStatus) -> CiCheckSuite {
        let revision = revision();
        let mut suite = CiCheckSuite::create(
            CiCheckSuiteIdentity::for_revision(aggregate(id), &revision)
                .expect("valid suite identity"),
            CiWorkflowLink::new(
                aggregate(200),
                aggregate(id + 300),
                CiLabel::from_untrusted("build and test").expect("valid workflow label"),
                None,
            )
            .expect("valid workflow link"),
            1_900_000_003_000,
        );
        if status == CiCheckStatus::Pending {
            return suite;
        }
        let run_id = aggregate(id + 400);
        suite
            .add_run(
                AggregateVersion::FIRST,
                CiCheckRunInput {
                    check_run_id: run_id,
                    label: CiLabel::from_untrusted("tests").expect("valid run label"),
                    queued_at_millis: 1_900_000_004_000,
                },
            )
            .expect("add run");
        if status == CiCheckStatus::Running {
            suite
                .start_run(
                    version(2),
                    run_id,
                    AggregateVersion::FIRST,
                    1_900_000_005_000,
                )
                .expect("start run");
            return suite;
        }
        suite
            .complete_run(
                version(2),
                run_id,
                AggregateVersion::FIRST,
                &commit(101),
                CiCheckRunCompletionInput {
                    status,
                    output: CiOutputText::from_untrusted("finished"),
                    artifacts: Vec::new(),
                    completed_at_millis: 1_900_000_006_000,
                },
            )
            .expect("complete run");
        suite
    }

    fn context() -> CodeActivityProjectionContext {
        CodeActivityProjectionContext {
            actor_kind: ActivityActorKind::Human,
            actor_label: "Ada".into(),
            project_id: Some("project-1".into()),
            thread_id: Some("thread-1".into()),
            visibility: ActivityVisibility::Project,
            projected_at: DateTime::from_timestamp_millis(1_900_000_010_000)
                .expect("valid timestamp"),
        }
    }

    fn branch_activity(
        event_id: u128,
        version_value: u64,
        kind: BranchCodeActivityKind,
    ) -> CollaborationCodeActivity {
        CollaborationCodeActivity::Branch(BranchCodeActivity {
            event_id: aggregate(event_id),
            actor_principal_id: principal(9),
            branch: branch(),
            version: version(version_value),
            occurred_at_millis: 1_900_000_000_000 + event_id as u64,
            kind,
        })
    }

    #[test]
    fn activity_git_fixture_maps_every_code_kind_exactly_once() {
        let fixtures = vec![
            (
                "branch_created",
                branch_activity(
                    100,
                    1,
                    BranchCodeActivityKind::Created {
                        commit: commit(101),
                    },
                ),
            ),
            (
                "branch_fast_forwarded",
                branch_activity(
                    101,
                    2,
                    BranchCodeActivityKind::Updated {
                        previous_commit: commit(101),
                        current_commit: commit(102),
                        update_kind: BranchUpdateKind::FastForward,
                    },
                ),
            ),
            (
                "branch_force_updated",
                branch_activity(
                    102,
                    3,
                    BranchCodeActivityKind::Updated {
                        previous_commit: commit(102),
                        current_commit: commit(103),
                        update_kind: BranchUpdateKind::Force,
                    },
                ),
            ),
            (
                "branch_merged",
                branch_activity(
                    103,
                    4,
                    BranchCodeActivityKind::Merged {
                        source_commit: commit(103),
                        target_branch: BranchRefName::parse("refs/heads/main")
                            .expect("valid target"),
                        result_commit: commit(104),
                    },
                ),
            ),
            (
                "branch_deleted",
                branch_activity(
                    104,
                    5,
                    BranchCodeActivityKind::Deleted {
                        commit: commit(103),
                    },
                ),
            ),
            (
                "patch_submitted",
                CollaborationCodeActivity::PatchSubmitted(revision()),
            ),
            (
                "review_commented",
                CollaborationCodeActivity::ReviewCommented(comment()),
            ),
            (
                "review_approved",
                CollaborationCodeActivity::ReviewDecisionRecorded(ReviewDecisionActivity {
                    approval: approval(12, ReviewDecision::Approve),
                }),
            ),
            (
                "review_changes_requested",
                CollaborationCodeActivity::ReviewDecisionRecorded(ReviewDecisionActivity {
                    approval: approval(13, ReviewDecision::RequestChanges),
                }),
            ),
            (
                "ci_pending",
                CollaborationCodeActivity::CiStatusChanged(suite(30, CiCheckStatus::Pending)),
            ),
            (
                "ci_running",
                CollaborationCodeActivity::CiStatusChanged(suite(31, CiCheckStatus::Running)),
            ),
            (
                "ci_success",
                CollaborationCodeActivity::CiStatusChanged(suite(32, CiCheckStatus::Success)),
            ),
            (
                "ci_failure",
                CollaborationCodeActivity::CiStatusChanged(suite(33, CiCheckStatus::Failure)),
            ),
            (
                "ci_cancelled",
                CollaborationCodeActivity::CiStatusChanged(suite(34, CiCheckStatus::Cancelled)),
            ),
            (
                "unsupported_workflow",
                CollaborationCodeActivity::Unsupported(GenericCodeActivity {
                    source_kind: ActivitySourceKind::Workflow,
                    source_id: "future-workflow-kind-1".into(),
                    source_version: 1,
                    actor_id: "service-1".into(),
                    community_id: branch().community_id(),
                    repository_id: branch().repository_id(),
                    event_kind: "future_workflow_kind".into(),
                    occurred_at_millis: 1_900_000_009_000,
                }),
            ),
        ];
        let expected = HashSet::from([
            "branch_created",
            "branch_fast_forwarded",
            "branch_force_updated",
            "branch_merged",
            "branch_deleted",
            "patch_submitted",
            "review_commented",
            "review_approved",
            "review_changes_requested",
            "ci_pending",
            "ci_running",
            "ci_success",
            "ci_failure",
            "ci_cancelled",
            "unsupported_workflow",
        ]);
        assert_eq!(
            fixtures
                .iter()
                .map(|(name, _)| *name)
                .collect::<HashSet<_>>(),
            expected
        );

        let mut reducer = ActivityReducer::new();
        let expected_community_id = branch().community_id().to_string();
        for (name, event) in fixtures {
            let item = project_code_activity(&context(), &event).expect("fixture should project");
            assert_eq!(
                item.context.community_id.as_deref(),
                Some(expected_community_id.as_str())
            );
            assert!(
                item.links.iter().any(|link| matches!(link, ActivityLink::Entity { entity_kind, .. } if entity_kind == "repository")),
                "{name} should retain its repository link"
            );
            assert!(matches!(
                reducer.reduce(item.clone()).expect("first delivery"),
                ActivityReduction::Inserted { .. }
            ));
            assert!(matches!(
                reducer.reduce(item).expect("duplicate delivery"),
                ActivityReduction::Duplicate { .. }
            ));
        }
        assert_eq!(reducer.items().len(), expected.len());
        assert_eq!(
            reducer
                .items()
                .iter()
                .map(|item| item.id.clone())
                .collect::<HashSet<_>>()
                .len(),
            expected.len()
        );
    }

    #[test]
    fn activity_git_updates_running_ci_in_place_and_keeps_decisions_immutable() {
        let pending = suite(40, CiCheckStatus::Pending);
        let running = suite(40, CiCheckStatus::Running);
        let pending = project_code_activity(
            &context(),
            &CollaborationCodeActivity::CiStatusChanged(pending),
        )
        .expect("pending suite should project");
        let running = project_code_activity(
            &context(),
            &CollaborationCodeActivity::CiStatusChanged(running),
        )
        .expect("running suite should project");
        assert_eq!(pending.id, running.id);
        assert!(running.source_version > pending.source_version);

        let approval = approval(14, ReviewDecision::Approve);
        let decision = project_code_activity(
            &context(),
            &CollaborationCodeActivity::ReviewDecisionRecorded(ReviewDecisionActivity { approval }),
        )
        .expect("approval should project");
        assert_eq!(decision.source_version, 1);
        assert_eq!(decision.outcome.status, ActivityOutcomeStatus::Success);

        let mut reducer = ActivityReducer::new();
        assert!(matches!(
            reducer.reduce(pending).expect("insert pending"),
            ActivityReduction::Inserted { .. }
        ));
        assert!(matches!(
            reducer.reduce(running).expect("update running"),
            ActivityReduction::Updated { .. }
        ));
        assert!(matches!(
            reducer.reduce(decision.clone()).expect("insert approval"),
            ActivityReduction::Inserted { .. }
        ));
        assert!(matches!(
            reducer.reduce(decision).expect("deduplicate approval"),
            ActivityReduction::Duplicate { .. }
        ));
        assert_eq!(reducer.items().len(), 2);
    }

    #[test]
    fn activity_git_rejects_invalid_fallback_and_timestamp() {
        let unsupported = CollaborationCodeActivity::Unsupported(GenericCodeActivity {
            source_kind: ActivitySourceKind::Acp,
            source_id: "wrong-source".into(),
            source_version: 1,
            actor_id: "actor-1".into(),
            community_id: branch().community_id(),
            repository_id: branch().repository_id(),
            event_kind: "unknown".into(),
            occurred_at_millis: 1_900_000_009_000,
        });
        assert!(matches!(
            project_code_activity(&context(), &unsupported),
            Err(CodeActivityProjectionError::InvalidFallback)
        ));

        let invalid_timestamp = branch_activity(
            105,
            6,
            BranchCodeActivityKind::Created {
                commit: commit(105),
            },
        );
        let invalid_timestamp = match invalid_timestamp {
            CollaborationCodeActivity::Branch(mut activity) => {
                activity.occurred_at_millis = u64::MAX;
                CollaborationCodeActivity::Branch(activity)
            }
            _ => unreachable!(),
        };
        assert!(matches!(
            project_code_activity(&context(), &invalid_timestamp),
            Err(CodeActivityProjectionError::InvalidTimestamp)
        ));
    }
}
