use std::{fmt, num::NonZeroU32};

use collaboration_domain::{
    AggregateId, AggregateVersion, CiCheckStatus, CiCheckSuiteRecordFields, GitCommitId,
    MergeEligibility, MergeReadiness, OperationId, PatchRevisionInput, PatchRevisionNumber,
    ReviewCommentInput, ReviewDecisionInput, ReviewDiffSide, ReviewIdentity, ReviewRecordFields,
};
use nostr_compat::nip34_collaboration::{
    GitIssue, GitPatch, GitPullRequest, GitPullRequestUpdate, PatchPosition,
};
use nostr_compat::nip34_repository::{
    RepositoryCoordinate, RepositoryStatus, RepositoryStatusEvent,
};
use nostr_compat::{EventId, PublicKey};
use serde_json::{Value, json};

use super::contracts::{ErrorClass, error_contract};

#[derive(Clone, Eq, PartialEq)]
pub enum ReviewCliCommand {
    CreatePatch {
        patch: GitPatch,
        author: PublicKey,
        created_at: u64,
        operation_id: OperationId,
    },
    GetPatch {
        event_id: EventId,
    },
    ListPatches {
        repository: RepositoryCoordinate,
        limit: NonZeroU32,
    },
    OpenPullRequest {
        pull_request: GitPullRequest,
        author: PublicKey,
        created_at: u64,
        operation_id: OperationId,
    },
    UpdatePullRequest {
        update: GitPullRequestUpdate,
        author: PublicKey,
        created_at: u64,
        operation_id: OperationId,
    },
    GetPullRequest {
        event_id: EventId,
    },
    ListPullRequests {
        repository: RepositoryCoordinate,
        limit: NonZeroU32,
    },
    CreateIssue {
        issue: GitIssue,
        author: PublicKey,
        created_at: u64,
        operation_id: OperationId,
    },
    GetIssue {
        event_id: EventId,
    },
    ListIssues {
        repository: RepositoryCoordinate,
        limit: NonZeroU32,
    },
    SetRecordStatus {
        status: RepositoryStatusEvent,
        author: PublicKey,
        created_at: u64,
        operation_id: OperationId,
    },
    GetReview {
        identity: ReviewIdentity,
    },
    OpenReview {
        identity: ReviewIdentity,
        required_approvals: u16,
        initial_revision: PatchRevisionInput,
        operation_id: OperationId,
    },
    SubmitRevision {
        identity: ReviewIdentity,
        expected_version: AggregateVersion,
        expected_revision: PatchRevisionNumber,
        revision: PatchRevisionInput,
        operation_id: OperationId,
    },
    AddReviewComment {
        identity: ReviewIdentity,
        expected_version: AggregateVersion,
        comment: ReviewCommentInput,
        operation_id: OperationId,
    },
    RecordReviewDecision {
        identity: ReviewIdentity,
        expected_version: AggregateVersion,
        decision: ReviewDecisionInput,
        operation_id: OperationId,
    },
    GetMergeReadiness {
        identity: ReviewIdentity,
        expected_revision: PatchRevisionNumber,
        expected_head_commit: GitCommitId,
    },
    GetCiStatus {
        identity: ReviewIdentity,
    },
    PublishCiStatus {
        suite: CiCheckSuiteRecordFields,
        expected_version: Option<AggregateVersion>,
        operation_id: OperationId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReviewCliVerb {
    CreatePatch,
    GetPatch,
    ListPatches,
    OpenPullRequest,
    UpdatePullRequest,
    GetPullRequest,
    ListPullRequests,
    CreateIssue,
    GetIssue,
    ListIssues,
    SetRecordStatus,
    GetReview,
    OpenReview,
    SubmitRevision,
    AddReviewComment,
    RecordReviewDecision,
    GetMergeReadiness,
    GetCiStatus,
    PublishCiStatus,
}

impl ReviewCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::CreatePatch => "patch.create",
            Self::GetPatch => "patch.get",
            Self::ListPatches => "patch.list",
            Self::OpenPullRequest => "pull_request.open",
            Self::UpdatePullRequest => "pull_request.update",
            Self::GetPullRequest => "pull_request.get",
            Self::ListPullRequests => "pull_request.list",
            Self::CreateIssue => "issue.create",
            Self::GetIssue => "issue.get",
            Self::ListIssues => "issue.list",
            Self::SetRecordStatus => "git_record.status.set",
            Self::GetReview => "review.get",
            Self::OpenReview => "review.open",
            Self::SubmitRevision => "review.revision.submit",
            Self::AddReviewComment => "review.comment.add",
            Self::RecordReviewDecision => "review.decision.record",
            Self::GetMergeReadiness => "review.readiness.get",
            Self::GetCiStatus => "ci.status.get",
            Self::PublishCiStatus => "ci.status.publish",
        }
    }
}

impl ReviewCliCommand {
    const fn verb(&self) -> ReviewCliVerb {
        match self {
            Self::CreatePatch { .. } => ReviewCliVerb::CreatePatch,
            Self::GetPatch { .. } => ReviewCliVerb::GetPatch,
            Self::ListPatches { .. } => ReviewCliVerb::ListPatches,
            Self::OpenPullRequest { .. } => ReviewCliVerb::OpenPullRequest,
            Self::UpdatePullRequest { .. } => ReviewCliVerb::UpdatePullRequest,
            Self::GetPullRequest { .. } => ReviewCliVerb::GetPullRequest,
            Self::ListPullRequests { .. } => ReviewCliVerb::ListPullRequests,
            Self::CreateIssue { .. } => ReviewCliVerb::CreateIssue,
            Self::GetIssue { .. } => ReviewCliVerb::GetIssue,
            Self::ListIssues { .. } => ReviewCliVerb::ListIssues,
            Self::SetRecordStatus { .. } => ReviewCliVerb::SetRecordStatus,
            Self::GetReview { .. } => ReviewCliVerb::GetReview,
            Self::OpenReview { .. } => ReviewCliVerb::OpenReview,
            Self::SubmitRevision { .. } => ReviewCliVerb::SubmitRevision,
            Self::AddReviewComment { .. } => ReviewCliVerb::AddReviewComment,
            Self::RecordReviewDecision { .. } => ReviewCliVerb::RecordReviewDecision,
            Self::GetMergeReadiness { .. } => ReviewCliVerb::GetMergeReadiness,
            Self::GetCiStatus { .. } => ReviewCliVerb::GetCiStatus,
            Self::PublishCiStatus { .. } => ReviewCliVerb::PublishCiStatus,
        }
    }
}

impl fmt::Debug for ReviewCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReviewCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum CanonicalGitRecord {
    Patch {
        event_id: EventId,
        author: PublicKey,
        created_at: u64,
        patch: GitPatch,
    },
    PullRequest {
        event_id: EventId,
        author: PublicKey,
        created_at: u64,
        pull_request: GitPullRequest,
    },
    Issue {
        event_id: EventId,
        author: PublicKey,
        created_at: u64,
        issue: GitIssue,
    },
}

impl CanonicalGitRecord {
    const fn kind(&self) -> GitRecordKind {
        match self {
            Self::Patch { .. } => GitRecordKind::Patch,
            Self::PullRequest { .. } => GitRecordKind::PullRequest,
            Self::Issue { .. } => GitRecordKind::Issue,
        }
    }
}

impl fmt::Debug for CanonicalGitRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CanonicalGitRecord")
            .field("kind", &self.kind())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GitRecordKind {
    Patch,
    PullRequest,
    Issue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalGitStatus {
    pub event_id: EventId,
    pub author: PublicKey,
    pub created_at: u64,
    pub status: RepositoryStatusEvent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReviewResourceId {
    Event(EventId),
    Review(AggregateId),
    CiSuite(AggregateId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReviewWriteReceipt {
    pub operation_id: OperationId,
    pub resource_id: ReviewResourceId,
    pub version: Option<AggregateVersion>,
}

#[derive(Clone, Eq, PartialEq)]
pub enum ReviewCliOutcome {
    GitRecord(CanonicalGitRecord),
    GitRecords(Vec<CanonicalGitRecord>),
    GitStatus(CanonicalGitStatus),
    Review(ReviewRecordFields),
    MergeReadiness(MergeReadiness),
    CiStatus(CiCheckSuiteRecordFields),
    Applied(ReviewWriteReceipt),
}

impl fmt::Debug for ReviewCliOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let variant = match self {
            Self::GitRecord(_) => "GitRecord",
            Self::GitRecords(_) => "GitRecords",
            Self::GitStatus(_) => "GitStatus",
            Self::Review(_) => "Review",
            Self::MergeReadiness(_) => "MergeReadiness",
            Self::CiStatus(_) => "CiStatus",
            Self::Applied(_) => "Applied",
        };
        formatter
            .debug_struct("ReviewCliOutcome")
            .field("variant", &variant)
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReviewCliError {
    InvalidRequest,
    NotFound,
    Unavailable,
    AuthorizationDenied,
    ApprovalDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl ReviewCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "review_cli_invalid_request",
            Self::NotFound => "review_cli_not_found",
            Self::Unavailable => "review_cli_unavailable",
            Self::AuthorizationDenied => "review_cli_authorization_denied",
            Self::ApprovalDenied => "review_cli_approval_denied",
            Self::PartialFailure => "review_cli_completion_unknown",
            Self::Unexpected => "review_cli_unexpected_response",
            Self::Conflict => "review_cli_stale_state",
        }
    }

    const fn common_class(self) -> ErrorClass {
        match self {
            Self::InvalidRequest => ErrorClass::Usage,
            Self::NotFound => ErrorClass::NotFound,
            Self::Unavailable => ErrorClass::Network { retryable: true },
            Self::AuthorizationDenied | Self::ApprovalDenied => ErrorClass::Authorization,
            Self::PartialFailure => ErrorClass::DeliveryUnknown,
            Self::Unexpected => ErrorClass::Unexpected,
            Self::Conflict => ErrorClass::Conflict,
        }
    }
}

pub trait ReviewCliExecutor {
    fn execute(&self, command: ReviewCliCommand) -> Result<ReviewCliOutcome, ReviewCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_review_command(
    executor: &impl ReviewCliExecutor,
    command: ReviewCliCommand,
) -> ReviewCliExecution {
    let verb = command.verb();
    match executor.execute(command) {
        Ok(outcome) => success_output(verb, outcome)
            .map(ReviewCliExecution::success)
            .unwrap_or_else(|| error_output(verb, ReviewCliError::Unexpected)),
        Err(error) => error_output(verb, error),
    }
}

impl ReviewCliExecution {
    fn success(value: Value) -> Self {
        Self {
            stdout: format!("{value}\n"),
            stderr: String::new(),
            exit_code: 0,
        }
    }

    fn failure(value: Value, exit_code: i32) -> Self {
        Self {
            stdout: String::new(),
            stderr: format!("{value}\n"),
            exit_code,
        }
    }
}

fn success_output(verb: ReviewCliVerb, outcome: ReviewCliOutcome) -> Option<Value> {
    match (verb, outcome) {
        (ReviewCliVerb::GetPatch, ReviewCliOutcome::GitRecord(record))
            if record.kind() == GitRecordKind::Patch =>
        {
            Some(git_record_output(verb, &record))
        }
        (ReviewCliVerb::GetPullRequest, ReviewCliOutcome::GitRecord(record))
            if record.kind() == GitRecordKind::PullRequest =>
        {
            Some(git_record_output(verb, &record))
        }
        (ReviewCliVerb::GetIssue, ReviewCliOutcome::GitRecord(record))
            if record.kind() == GitRecordKind::Issue =>
        {
            Some(git_record_output(verb, &record))
        }
        (ReviewCliVerb::ListPatches, ReviewCliOutcome::GitRecords(records))
            if records
                .iter()
                .all(|record| record.kind() == GitRecordKind::Patch) =>
        {
            Some(git_records_output(verb, &records))
        }
        (ReviewCliVerb::ListPullRequests, ReviewCliOutcome::GitRecords(records))
            if records
                .iter()
                .all(|record| record.kind() == GitRecordKind::PullRequest) =>
        {
            Some(git_records_output(verb, &records))
        }
        (ReviewCliVerb::ListIssues, ReviewCliOutcome::GitRecords(records))
            if records
                .iter()
                .all(|record| record.kind() == GitRecordKind::Issue) =>
        {
            Some(git_records_output(verb, &records))
        }
        (ReviewCliVerb::SetRecordStatus, ReviewCliOutcome::GitStatus(status)) => {
            Some(git_status_output(verb, &status))
        }
        (ReviewCliVerb::GetReview, ReviewCliOutcome::Review(review)) => {
            Some(review_output(verb, &review))
        }
        (ReviewCliVerb::GetMergeReadiness, ReviewCliOutcome::MergeReadiness(readiness)) => {
            Some(readiness_output(verb, &readiness))
        }
        (ReviewCliVerb::GetCiStatus, ReviewCliOutcome::CiStatus(status)) => {
            Some(ci_status_output(verb, &status))
        }
        (
            ReviewCliVerb::CreatePatch
            | ReviewCliVerb::OpenPullRequest
            | ReviewCliVerb::UpdatePullRequest
            | ReviewCliVerb::CreateIssue
            | ReviewCliVerb::OpenReview
            | ReviewCliVerb::SubmitRevision
            | ReviewCliVerb::AddReviewComment
            | ReviewCliVerb::RecordReviewDecision
            | ReviewCliVerb::PublishCiStatus,
            ReviewCliOutcome::Applied(receipt),
        ) => Some(write_output(verb, receipt)),
        _ => None,
    }
}

fn error_output(verb: ReviewCliVerb, error: ReviewCliError) -> ReviewCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    ReviewCliExecution::failure(
        json!({
            "command": verb.as_str(),
            "error": contract.category,
            "error_code": diagnostic,
            "message": diagnostic,
            "ok": false,
            "retryable": contract.retryable,
        }),
        contract.exit_class as i32,
    )
}

fn git_records_output(verb: ReviewCliVerb, records: &[CanonicalGitRecord]) -> Value {
    json!({
        "command": verb.as_str(),
        "ok": true,
        "records": records.iter().map(|record| git_record_output(verb, record)).collect::<Vec<_>>(),
    })
}

fn git_record_output(verb: ReviewCliVerb, record: &CanonicalGitRecord) -> Value {
    match record {
        CanonicalGitRecord::Patch {
            event_id,
            author,
            created_at,
            patch,
        } => json!({
            "author": author.to_hex(),
            "command": verb.as_str(),
            "commit": patch.commit.as_ref().map(|commit| commit.as_hex()),
            "content": patch.content,
            "created_at": created_at,
            "event_id": event_id.to_hex(),
            "kind": "patch",
            "ok": true,
            "parent_commit": patch.parent_commit.as_ref().map(|commit| commit.as_hex()),
            "position": match patch.position {
                PatchPosition::Continuation => "continuation",
                PatchPosition::Root => "root",
                PatchPosition::RootRevision => "root_revision",
            },
            "repository": patch.repository.value(),
        }),
        CanonicalGitRecord::PullRequest {
            event_id,
            author,
            created_at,
            pull_request,
        } => json!({
            "author": author.to_hex(),
            "branch_name": pull_request.branch_name,
            "channel_id": pull_request.channel_id,
            "clone_urls": pull_request.clone_urls,
            "command": verb.as_str(),
            "content": pull_request.content,
            "created_at": created_at,
            "event_id": event_id.to_hex(),
            "kind": "pull_request",
            "labels": pull_request.labels,
            "ok": true,
            "repository": pull_request.repository.value(),
            "subject": pull_request.subject,
            "tip_commit": pull_request.tip_commit.as_hex(),
        }),
        CanonicalGitRecord::Issue {
            event_id,
            author,
            created_at,
            issue,
        } => json!({
            "author": author.to_hex(),
            "command": verb.as_str(),
            "content": issue.content,
            "created_at": created_at,
            "event_id": event_id.to_hex(),
            "kind": "issue",
            "labels": issue.labels,
            "ok": true,
            "repository": issue.repository.value(),
            "subject": issue.subject,
        }),
    }
}

fn git_status_output(verb: ReviewCliVerb, record: &CanonicalGitStatus) -> Value {
    json!({
        "accepted_revision_root": record.status.accepted_revision_root.map(EventId::to_hex),
        "applied_patches": record.status.applied_patches.iter().map(|patch| patch.event_id.to_hex()).collect::<Vec<_>>(),
        "author": record.author.to_hex(),
        "command": verb.as_str(),
        "content": record.status.content,
        "created_at": record.created_at,
        "event_id": record.event_id.to_hex(),
        "merge_commit": record.status.merge_commit.as_ref().map(|commit| commit.as_hex()),
        "ok": true,
        "root_event": record.status.root_event.to_hex(),
        "status": repository_status(record.status.status),
    })
}

fn review_output(verb: ReviewCliVerb, review: &ReviewRecordFields) -> Value {
    json!({
        "approvals": review.approvals.iter().map(|approval| json!({
            "approval_id": approval.approval_id,
            "approver_principal_id": approval.approver_principal_id,
            "created_at_millis": approval.created_at_millis,
            "decision": match approval.decision {
                collaboration_domain::ReviewDecision::Approve => "approve",
                collaboration_domain::ReviewDecision::RequestChanges => "request_changes",
            },
            "head_commit": approval.head_commit.as_str(),
            "revision": approval.revision.get(),
        })).collect::<Vec<_>>(),
        "comments": review.comments.iter().map(|comment| json!({
            "anchor": {
                "commit": comment.anchor.commit.as_str(),
                "end_line": comment.anchor.end_line.get(),
                "file_path": comment.anchor.file_path.as_str(),
                "hunk_id": comment.anchor.hunk_id.as_str(),
                "revision": comment.anchor.revision.get(),
                "side": match comment.anchor.side {
                    ReviewDiffSide::Base => "base",
                    ReviewDiffSide::Head => "head",
                },
                "start_line": comment.anchor.start_line.get(),
            },
            "author_principal_id": comment.author_principal_id,
            "body": comment.body.as_str(),
            "comment_id": comment.comment_id,
            "created_at_millis": comment.created_at_millis,
        })).collect::<Vec<_>>(),
        "command": verb.as_str(),
        "current_revision": review.revisions.last().map(|revision| json!({
            "author_principal_id": revision.author_principal_id,
            "base_commit": revision.base_commit.as_str(),
            "created_at_millis": revision.created_at_millis,
            "head_commit": revision.head_commit.as_str(),
            "number": revision.number.get(),
            "revision_id": revision.revision_id,
        })),
        "ok": true,
        "repository_id": review.identity.repository_id(),
        "required_approvals": review.required_approvals,
        "review_id": review.identity.review_id(),
        "version": review.version,
    })
}

fn readiness_output(verb: ReviewCliVerb, readiness: &MergeReadiness) -> Value {
    json!({
        "approval_ids": readiness.approval_ids,
        "change_request_ids": readiness.change_request_ids,
        "command": verb.as_str(),
        "eligibility": match readiness.eligibility {
            MergeEligibility::Ready => "ready",
            MergeEligibility::Blocked => "blocked",
        },
        "head_commit": readiness.head_commit.as_str(),
        "ok": true,
        "required_approvals": readiness.required_approvals,
        "review_id": readiness.review.review_id(),
        "revision": readiness.revision.get(),
    })
}

fn ci_status_output(verb: ReviewCliVerb, suite: &CiCheckSuiteRecordFields) -> Value {
    json!({
        "command": verb.as_str(),
        "created_at_millis": suite.created_at_millis,
        "head_commit": suite.identity.head_commit().as_str(),
        "ok": true,
        "repository_id": suite.identity.repository_id(),
        "review_id": suite.identity.review().review_id(),
        "revision": suite.identity.revision().get(),
        "runs": suite.runs.iter().map(|run| json!({
            "artifacts": run.artifacts.iter().map(|artifact| json!({
                "artifact_id": artifact.artifact_id,
                "digest": artifact.digest.as_ref().map(|digest| digest.as_str()),
                "label": artifact.label.as_str(),
                "url": artifact.url.as_str(),
            })).collect::<Vec<_>>(),
            "check_run_id": run.check_run_id,
            "completed_at_millis": run.completed_at_millis,
            "label": run.label.as_str(),
            "output": run.output.as_ref().map(|output| output.as_str()),
            "queued_at_millis": run.queued_at_millis,
            "started_at_millis": run.started_at_millis,
            "status": ci_check_status(run.status),
            "version": run.version,
        })).collect::<Vec<_>>(),
        "suite_id": suite.identity.suite_id(),
        "version": suite.version,
        "workflow": {
            "label": suite.workflow.label.as_str(),
            "url": suite.workflow.url.as_ref().map(|url| url.as_str()),
            "workflow_id": suite.workflow.workflow_id,
            "workflow_run_id": suite.workflow.workflow_run_id,
        },
    })
}

fn write_output(verb: ReviewCliVerb, receipt: ReviewWriteReceipt) -> Value {
    let (resource_kind, resource_id) = resource_output(receipt.resource_id);
    json!({
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.operation_id,
        "resource_id": resource_id,
        "resource_kind": resource_kind,
        "version": receipt.version,
    })
}

fn resource_output(resource: ReviewResourceId) -> (&'static str, String) {
    match resource {
        ReviewResourceId::Event(event_id) => ("event", event_id.to_hex()),
        ReviewResourceId::Review(review_id) => ("review", review_id.to_string()),
        ReviewResourceId::CiSuite(suite_id) => ("ci_suite", suite_id.to_string()),
    }
}

const fn repository_status(status: RepositoryStatus) -> &'static str {
    match status {
        RepositoryStatus::Open => "open",
        RepositoryStatus::AppliedOrResolved => "applied_or_resolved",
        RepositoryStatus::Closed => "closed",
        RepositoryStatus::Draft => "draft",
    }
}

const fn ci_check_status(status: CiCheckStatus) -> &'static str {
    match status {
        CiCheckStatus::Pending => "pending",
        CiCheckStatus::Running => "running",
        CiCheckStatus::Success => "success",
        CiCheckStatus::Failure => "failure",
        CiCheckStatus::Cancelled => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use collaboration_domain::{
        BranchCollaborationIdentity, BranchGeneration, BranchRefName, CiCheckRun,
        CiCheckSuiteIdentity, CiLabel, CiOutputText, CiWorkflowLink, CommunityId, PatchRevision,
        PrincipalId, ReviewApproval, ReviewComment, ReviewCommentAnchor, ReviewCommentBody,
        ReviewDecision,
    };
    use nostr_compat::nip34_repository::GitObjectId;
    use uuid::Uuid;

    use super::*;

    struct TestExecutor(RefCell<Option<Result<ReviewCliOutcome, ReviewCliError>>>);

    impl TestExecutor {
        fn returning(result: Result<ReviewCliOutcome, ReviewCliError>) -> Self {
            Self(RefCell::new(Some(result)))
        }
    }

    impl ReviewCliExecutor for TestExecutor {
        fn execute(&self, _command: ReviewCliCommand) -> Result<ReviewCliOutcome, ReviewCliError> {
            self.0
                .borrow_mut()
                .take()
                .expect("the test executor is called once")
        }
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn operation_id() -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(20))
    }

    fn event_id(value: u8) -> EventId {
        EventId::from_bytes([value; 32])
    }

    fn author(value: u8) -> PublicKey {
        PublicKey::from_bytes([value; 32])
    }

    fn repository() -> RepositoryCoordinate {
        RepositoryCoordinate::parse(&format!("30617:{}:zed", author(1).to_hex()))
            .expect("repository")
    }

    fn object_id(value: u8) -> GitObjectId {
        GitObjectId::from_hex(&format!("{value:02x}{}", "00".repeat(19))).expect("object id")
    }

    fn domain_commit(value: u8) -> GitCommitId {
        GitCommitId::parse(format!("{value:02x}{}", "00".repeat(19))).expect("commit")
    }

    fn review_identity() -> ReviewIdentity {
        ReviewIdentity::new(
            aggregate_id(4),
            BranchCollaborationIdentity::new(
                CommunityId::from_uuid(Uuid::from_u128(1)),
                aggregate_id(3),
                BranchRefName::parse("refs/heads/feature/review").expect("branch"),
                BranchGeneration::FIRST,
            )
            .expect("branch identity"),
        )
        .expect("review identity")
    }

    fn revision() -> PatchRevision {
        PatchRevision {
            revision_id: aggregate_id(5),
            review: review_identity(),
            number: PatchRevisionNumber::FIRST,
            base_commit: domain_commit(1),
            head_commit: domain_commit(2),
            author_principal_id: PrincipalId::from_uuid(Uuid::from_u128(6)),
            created_at_millis: 10,
        }
    }

    fn git_records() -> Vec<CanonicalGitRecord> {
        vec![
            CanonicalGitRecord::Patch {
                event_id: event_id(2),
                author: author(1),
                created_at: 10,
                patch: GitPatch {
                    repository: repository(),
                    earliest_unique_commit: None,
                    recipients: vec![author(1)],
                    reply_to: None,
                    position: PatchPosition::Root,
                    commit: Some(object_id(2)),
                    parent_commit: Some(object_id(1)),
                    commit_pgp_signature: None,
                    committer: None,
                    content: "diff --git a/a b/a".into(),
                    extra_tags: Vec::new(),
                },
            },
            CanonicalGitRecord::PullRequest {
                event_id: event_id(3),
                author: author(1),
                created_at: 11,
                pull_request: GitPullRequest {
                    repository: repository(),
                    earliest_unique_commit: None,
                    recipients: vec![author(1)],
                    subject: "Improve review".into(),
                    labels: vec!["enhancement".into()],
                    tip_commit: object_id(3),
                    clone_urls: vec!["https://example.com/zed.git".into()],
                    channel_id: None,
                    branch_name: Some("feature/review".into()),
                    merge_base: Some(object_id(1)),
                    revision_of: None,
                    content: "Pull request body".into(),
                    extra_tags: Vec::new(),
                },
            },
            CanonicalGitRecord::Issue {
                event_id: event_id(4),
                author: author(1),
                created_at: 12,
                issue: GitIssue {
                    repository: repository(),
                    recipients: vec![author(1)],
                    subject: "Broken review".into(),
                    labels: vec!["bug".into()],
                    content: "Issue body".into(),
                    extra_tags: Vec::new(),
                },
            },
        ]
    }

    #[test]
    fn patch_pull_request_and_issue_outputs_are_canonical() {
        for (command, record, kind, marker) in [
            (
                ReviewCliCommand::GetPatch {
                    event_id: event_id(2),
                },
                git_records().remove(0),
                "patch",
                "diff --git",
            ),
            (
                ReviewCliCommand::GetPullRequest {
                    event_id: event_id(3),
                },
                git_records().remove(1),
                "pull_request",
                "Improve review",
            ),
            (
                ReviewCliCommand::GetIssue {
                    event_id: event_id(4),
                },
                git_records().remove(2),
                "issue",
                "Broken review",
            ),
        ] {
            assert!(!format!("{record:?}").contains(marker));
            let output = execute_review_command(
                &TestExecutor::returning(Ok(ReviewCliOutcome::GitRecord(record))),
                command,
            );
            let value: Value = serde_json::from_str(&output.stdout).expect("JSON");
            assert_eq!(value["kind"], kind);
            assert!(output.stdout.contains(marker));
        }
    }

    #[test]
    fn repository_status_output_is_stable() {
        let status = RepositoryStatusEvent {
            status: RepositoryStatus::AppliedOrResolved,
            root_event: event_id(3),
            accepted_revision_root: Some(event_id(2)),
            recipients: vec![author(1)],
            repository: Some(repository()),
            earliest_unique_commit: Some(object_id(1)),
            applied_patches: Vec::new(),
            merge_commit: Some(object_id(4)),
            applied_as_commits: vec![object_id(4)],
            content: "Merged safely".into(),
            extra_tags: Vec::new(),
        };
        let command = ReviewCliCommand::SetRecordStatus {
            status: status.clone(),
            author: author(1),
            created_at: 20,
            operation_id: operation_id(),
        };
        assert!(!format!("{command:?}").contains("Merged safely"));
        let output = execute_review_command(
            &TestExecutor::returning(Ok(ReviewCliOutcome::GitStatus(CanonicalGitStatus {
                event_id: event_id(5),
                author: author(1),
                created_at: 20,
                status,
            }))),
            command,
        );
        let value: Value = serde_json::from_str(&output.stdout).expect("JSON");
        assert_eq!(value["status"], "applied_or_resolved");
        assert_eq!(value["merge_commit"], object_id(4).as_hex());
    }

    #[test]
    fn review_comments_and_approvals_have_stable_projection() {
        let revision = revision();
        let anchor = ReviewCommentAnchor::new(
            PatchRevisionNumber::FIRST,
            revision.head_commit.clone(),
            collaboration_domain::ReviewFilePath::new("src/main.rs").expect("path"),
            collaboration_domain::ReviewHunkId::parse("a".repeat(64)).expect("hunk"),
            ReviewDiffSide::Head,
            NonZeroU32::new(2).expect("line"),
            NonZeroU32::new(3).expect("line"),
        )
        .expect("anchor");
        let record = ReviewRecordFields {
            identity: review_identity(),
            required_approvals: 1,
            revisions: vec![revision.clone()],
            comments: vec![ReviewComment {
                comment_id: aggregate_id(7),
                review: review_identity(),
                author_principal_id: PrincipalId::from_uuid(Uuid::from_u128(8)),
                body: ReviewCommentBody::new("Please cover this path").expect("body"),
                anchor,
                created_at_millis: 11,
            }],
            approvals: vec![ReviewApproval {
                approval_id: aggregate_id(9),
                review: review_identity(),
                revision: PatchRevisionNumber::FIRST,
                head_commit: revision.head_commit,
                approver_principal_id: PrincipalId::from_uuid(Uuid::from_u128(10)),
                decision: ReviewDecision::Approve,
                created_at_millis: 12,
            }],
            version: AggregateVersion::new(3).expect("version"),
        };
        let output = execute_review_command(
            &TestExecutor::returning(Ok(ReviewCliOutcome::Review(record))),
            ReviewCliCommand::GetReview {
                identity: review_identity(),
            },
        );
        let value: Value = serde_json::from_str(&output.stdout).expect("JSON");
        assert_eq!(value["comments"][0]["anchor"]["file_path"], "src/main.rs");
        assert_eq!(value["approvals"][0]["decision"], "approve");
        assert_eq!(value["current_revision"]["number"], 1);
    }

    #[test]
    fn ci_status_preserves_runs_and_bounded_output() {
        let revision = revision();
        let identity = CiCheckSuiteIdentity::for_revision(aggregate_id(11), &revision)
            .expect("suite identity");
        let workflow = CiWorkflowLink::new(
            aggregate_id(12),
            aggregate_id(13),
            CiLabel::from_untrusted("CI").expect("label"),
            None,
        )
        .expect("workflow");
        let run = CiCheckRun {
            check_run_id: aggregate_id(14),
            suite: identity.clone(),
            label: CiLabel::from_untrusted("test").expect("label"),
            status: CiCheckStatus::Success,
            output: Some(CiOutputText::from_untrusted("all tests passed")),
            artifacts: Vec::new(),
            queued_at_millis: 10,
            started_at_millis: Some(11),
            completed_at_millis: Some(12),
            version: AggregateVersion::new(3).expect("version"),
        };
        let suite = CiCheckSuiteRecordFields {
            identity,
            workflow,
            runs: vec![run],
            created_at_millis: 10,
            version: AggregateVersion::new(2).expect("version"),
        };
        let output = execute_review_command(
            &TestExecutor::returning(Ok(ReviewCliOutcome::CiStatus(suite))),
            ReviewCliCommand::GetCiStatus {
                identity: review_identity(),
            },
        );
        let value: Value = serde_json::from_str(&output.stdout).expect("JSON");
        assert_eq!(value["runs"][0]["status"], "success");
        assert_eq!(value["runs"][0]["output"], "all tests passed");
    }

    #[test]
    fn stale_conflict_denied_approval_and_exit_matrix_are_stable() {
        let cases = [
            (ReviewCliError::InvalidRequest, "user_error", 1, false),
            (ReviewCliError::NotFound, "not_found", 1, false),
            (ReviewCliError::Unavailable, "network_error", 2, true),
            (ReviewCliError::PartialFailure, "delivery_unknown", 2, false),
            (ReviewCliError::AuthorizationDenied, "auth_error", 3, false),
            (ReviewCliError::ApprovalDenied, "auth_error", 3, false),
            (ReviewCliError::Unexpected, "error", 4, false),
            (ReviewCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output = execute_review_command(
                &TestExecutor::returning(Err(error)),
                ReviewCliCommand::GetReview {
                    identity: review_identity(),
                },
            );
            let value: Value = serde_json::from_str(&output.stderr).expect("error JSON");
            assert_eq!(value["error"], category);
            assert_eq!(value["retryable"], retryable);
            assert_eq!(output.exit_code, exit_code);
        }
    }

    #[test]
    fn mismatched_record_kind_fails_closed() {
        let issue = git_records().remove(2);
        let output = execute_review_command(
            &TestExecutor::returning(Ok(ReviewCliOutcome::GitRecord(issue))),
            ReviewCliCommand::GetPatch {
                event_id: event_id(4),
            },
        );
        assert_eq!(output.exit_code, 4);
        assert!(output.stdout.is_empty());
    }
}
