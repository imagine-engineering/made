use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    num::{NonZeroU32, NonZeroU64},
};

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::{AggregateId, AggregateVersion, BranchCollaborationIdentity, GitCommitId, PrincipalId};

const MAX_REVIEW_FILE_PATH_BYTES: usize = 4_096;
const MAX_REVIEW_COMMENT_BYTES: usize = 65_536;
const REVIEW_HUNK_ID_BYTES: usize = 64;
const MAX_REQUIRED_APPROVALS: u16 = 128;
const MAX_PATCH_REVISIONS: usize = 10_000;
const MAX_REVIEW_COMMENTS: usize = 100_000;
const MAX_REVIEW_DECISIONS: usize = 100_000;

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ReviewIdentity {
    review_id: AggregateId,
    branch: BranchCollaborationIdentity,
}

impl ReviewIdentity {
    pub fn new(
        review_id: AggregateId,
        branch: BranchCollaborationIdentity,
    ) -> Result<Self, ReviewError> {
        if review_id.as_uuid().is_nil() {
            return Err(ReviewError::InvalidReviewId);
        }
        Ok(Self { review_id, branch })
    }

    pub const fn review_id(&self) -> AggregateId {
        self.review_id
    }

    pub const fn branch(&self) -> &BranchCollaborationIdentity {
        &self.branch
    }

    pub const fn repository_id(&self) -> AggregateId {
        self.branch.repository_id()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PatchRevisionNumber(NonZeroU64);

impl PatchRevisionNumber {
    pub const FIRST: Self = Self(NonZeroU64::MIN);

    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }

    pub fn next(self) -> Option<Self> {
        self.get().checked_add(1).and_then(Self::new)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PatchRevision {
    pub revision_id: AggregateId,
    pub review: ReviewIdentity,
    pub number: PatchRevisionNumber,
    pub base_commit: GitCommitId,
    pub head_commit: GitCommitId,
    pub author_principal_id: PrincipalId,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchRevisionInput {
    pub revision_id: AggregateId,
    pub base_commit: GitCommitId,
    pub head_commit: GitCommitId,
    pub author_principal_id: PrincipalId,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ReviewFilePath(String);

impl ReviewFilePath {
    pub fn new(value: impl Into<String>) -> Result<Self, ReviewError> {
        let value = value.into();
        if !is_safe_review_path(&value) {
            return Err(ReviewError::InvalidFilePath);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ReviewFilePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ReviewHunkId(String);

impl ReviewHunkId {
    pub fn parse(value: impl Into<String>) -> Result<Self, ReviewError> {
        let value = value.into();
        if value.len() != REVIEW_HUNK_ID_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ReviewError::InvalidHunkId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ReviewHunkId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ReviewCommentBody(String);

impl ReviewCommentBody {
    pub fn new(value: impl Into<String>) -> Result<Self, ReviewError> {
        let value = value.into();
        if value.trim().is_empty() || value.len() > MAX_REVIEW_COMMENT_BYTES {
            return Err(ReviewError::InvalidCommentBody);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ReviewCommentBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewDiffSide {
    Base,
    Head,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReviewCommentAnchor {
    pub revision: PatchRevisionNumber,
    pub commit: GitCommitId,
    pub file_path: ReviewFilePath,
    pub hunk_id: ReviewHunkId,
    pub side: ReviewDiffSide,
    pub start_line: NonZeroU32,
    pub end_line: NonZeroU32,
}

impl ReviewCommentAnchor {
    pub fn new(
        revision: PatchRevisionNumber,
        commit: GitCommitId,
        file_path: ReviewFilePath,
        hunk_id: ReviewHunkId,
        side: ReviewDiffSide,
        start_line: NonZeroU32,
        end_line: NonZeroU32,
    ) -> Result<Self, ReviewError> {
        if end_line < start_line {
            return Err(ReviewError::InvalidCommentAnchor);
        }
        Ok(Self {
            revision,
            commit,
            file_path,
            hunk_id,
            side,
            start_line,
            end_line,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReviewComment {
    pub comment_id: AggregateId,
    pub review: ReviewIdentity,
    pub author_principal_id: PrincipalId,
    pub body: ReviewCommentBody,
    pub anchor: ReviewCommentAnchor,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewCommentInput {
    pub comment_id: AggregateId,
    pub author_principal_id: PrincipalId,
    pub body: ReviewCommentBody,
    pub anchor: ReviewCommentAnchor,
    pub created_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewDecision {
    Approve,
    RequestChanges,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReviewApproval {
    pub approval_id: AggregateId,
    pub review: ReviewIdentity,
    pub revision: PatchRevisionNumber,
    pub head_commit: GitCommitId,
    pub approver_principal_id: PrincipalId,
    pub decision: ReviewDecision,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewDecisionInput {
    pub approval_id: AggregateId,
    pub revision: PatchRevisionNumber,
    pub head_commit: GitCommitId,
    pub approver_principal_id: PrincipalId,
    pub decision: ReviewDecision,
    pub created_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalApplicability {
    Current,
    SupersededByRevision {
        current_revision: PatchRevisionNumber,
    },
    SupersededByDecision {
        current_decision_id: AggregateId,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeEligibility {
    Ready,
    Blocked,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MergeReadiness {
    pub review: ReviewIdentity,
    pub revision: PatchRevisionNumber,
    pub head_commit: GitCommitId,
    pub eligibility: MergeEligibility,
    pub required_approvals: u16,
    pub approval_ids: Vec<AggregateId>,
    pub change_request_ids: Vec<AggregateId>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReviewRecordFields {
    pub identity: ReviewIdentity,
    pub required_approvals: u16,
    pub revisions: Vec<PatchRevision>,
    pub comments: Vec<ReviewComment>,
    pub approvals: Vec<ReviewApproval>,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReviewCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Review {
    fields: ReviewRecordFields,
}

impl Review {
    pub fn open(
        identity: ReviewIdentity,
        required_approvals: u16,
        initial_revision: PatchRevisionInput,
    ) -> Result<Self, ReviewError> {
        let revision = PatchRevision {
            revision_id: initial_revision.revision_id,
            review: identity.clone(),
            number: PatchRevisionNumber::FIRST,
            base_commit: initial_revision.base_commit,
            head_commit: initial_revision.head_commit,
            author_principal_id: initial_revision.author_principal_id,
            created_at_millis: initial_revision.created_at_millis,
        };
        let review = Self {
            fields: ReviewRecordFields {
                identity,
                required_approvals,
                revisions: vec![revision],
                comments: Vec::new(),
                approvals: Vec::new(),
                version: AggregateVersion::FIRST,
            },
        };
        validate_record(&review.fields)?;
        Ok(review)
    }

    pub fn from_record(fields: ReviewRecordFields) -> Result<Self, ReviewError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &ReviewRecordFields {
        &self.fields
    }

    pub fn current_revision(&self) -> Option<&PatchRevision> {
        self.fields.revisions.last()
    }

    pub fn submit_revision(
        &mut self,
        expected_version: AggregateVersion,
        expected_revision: PatchRevisionNumber,
        revision: PatchRevisionInput,
    ) -> Result<ReviewCommandOutcome, ReviewError> {
        let current = self.current_revision_or_error()?.clone();
        let next_number = expected_revision
            .next()
            .ok_or(ReviewError::RevisionExhausted)?;
        let proposed = PatchRevision {
            revision_id: revision.revision_id,
            review: self.fields.identity.clone(),
            number: next_number,
            base_commit: revision.base_commit,
            head_commit: revision.head_commit,
            author_principal_id: revision.author_principal_id,
            created_at_millis: revision.created_at_millis,
        };
        if let Some(existing) = self
            .fields
            .revisions
            .iter()
            .find(|revision| revision.revision_id == proposed.revision_id)
        {
            return if existing == &proposed {
                Ok(ReviewCommandOutcome::Unchanged)
            } else {
                Err(ReviewError::ConflictingRecordId)
            };
        }
        self.require_version(expected_version)?;
        self.require_revision(expected_revision)?;
        if self.fields.revisions.len() >= MAX_PATCH_REVISIONS {
            return Err(ReviewError::TooManyRevisions);
        }
        validate_revision(&self.fields.identity, &proposed)?;
        if proposed.created_at_millis < current.created_at_millis {
            return Err(ReviewError::InvalidTimestamp);
        }
        self.fields.version = self.next_version()?;
        self.fields.revisions.push(proposed);
        Ok(ReviewCommandOutcome::Applied)
    }

    pub fn add_comment(
        &mut self,
        expected_version: AggregateVersion,
        comment: ReviewCommentInput,
    ) -> Result<ReviewCommandOutcome, ReviewError> {
        let proposed = ReviewComment {
            comment_id: comment.comment_id,
            review: self.fields.identity.clone(),
            author_principal_id: comment.author_principal_id,
            body: comment.body,
            anchor: comment.anchor,
            created_at_millis: comment.created_at_millis,
        };
        if let Some(existing) = self
            .fields
            .comments
            .iter()
            .find(|comment| comment.comment_id == proposed.comment_id)
        {
            return if existing == &proposed {
                Ok(ReviewCommandOutcome::Unchanged)
            } else {
                Err(ReviewError::ConflictingRecordId)
            };
        }
        self.require_version(expected_version)?;
        let current = self.current_revision_or_error()?;
        validate_comment(&self.fields.identity, &proposed, current)?;
        if proposed.created_at_millis < current.created_at_millis
            || self
                .fields
                .comments
                .last()
                .is_some_and(|comment| proposed.created_at_millis < comment.created_at_millis)
        {
            return Err(ReviewError::InvalidTimestamp);
        }
        if self.fields.comments.len() >= MAX_REVIEW_COMMENTS {
            return Err(ReviewError::TooManyComments);
        }
        self.fields.version = self.next_version()?;
        self.fields.comments.push(proposed);
        Ok(ReviewCommandOutcome::Applied)
    }

    pub fn record_decision(
        &mut self,
        expected_version: AggregateVersion,
        decision: ReviewDecisionInput,
    ) -> Result<ReviewCommandOutcome, ReviewError> {
        let proposed = ReviewApproval {
            approval_id: decision.approval_id,
            review: self.fields.identity.clone(),
            revision: decision.revision,
            head_commit: decision.head_commit,
            approver_principal_id: decision.approver_principal_id,
            decision: decision.decision,
            created_at_millis: decision.created_at_millis,
        };
        if let Some(existing) = self
            .fields
            .approvals
            .iter()
            .find(|approval| approval.approval_id == proposed.approval_id)
        {
            return if existing == &proposed {
                Ok(ReviewCommandOutcome::Unchanged)
            } else {
                Err(ReviewError::ConflictingRecordId)
            };
        }
        self.require_version(expected_version)?;
        self.require_revision(proposed.revision)?;
        self.require_head(&proposed.head_commit)?;
        if self.fields.approvals.len() >= MAX_REVIEW_DECISIONS {
            return Err(ReviewError::TooManyDecisions);
        }
        validate_approval(
            &self.fields.identity,
            &proposed,
            self.current_revision_or_error()?,
        )?;
        if proposed.created_at_millis < self.current_revision_or_error()?.created_at_millis
            || self
                .fields
                .approvals
                .last()
                .is_some_and(|approval| proposed.created_at_millis < approval.created_at_millis)
        {
            return Err(ReviewError::InvalidTimestamp);
        }
        self.fields.version = self.next_version()?;
        self.fields.approvals.push(proposed);
        Ok(ReviewCommandOutcome::Applied)
    }

    pub fn approval_applicability(
        &self,
        approval_id: AggregateId,
    ) -> Option<ApprovalApplicability> {
        let current_revision = self.current_revision()?.number;
        let (approval_index, approval) = self
            .fields
            .approvals
            .iter()
            .enumerate()
            .find(|(_, approval)| approval.approval_id == approval_id)?;
        if approval.revision != current_revision {
            return Some(ApprovalApplicability::SupersededByRevision { current_revision });
        }
        let next_index = approval_index.checked_add(1)?;
        if let Some(replacement) =
            self.fields
                .approvals
                .iter()
                .skip(next_index)
                .rev()
                .find(|replacement| {
                    replacement.revision == approval.revision
                        && replacement.head_commit == approval.head_commit
                        && replacement.approver_principal_id == approval.approver_principal_id
                })
        {
            return Some(ApprovalApplicability::SupersededByDecision {
                current_decision_id: replacement.approval_id,
            });
        }
        Some(ApprovalApplicability::Current)
    }

    pub fn merge_readiness(
        &self,
        expected_revision: PatchRevisionNumber,
        expected_head_commit: &GitCommitId,
    ) -> Result<MergeReadiness, ReviewError> {
        self.require_revision(expected_revision)?;
        self.require_head(expected_head_commit)?;
        let mut current_decisions = BTreeMap::new();
        for approval in &self.fields.approvals {
            if approval.revision == expected_revision
                && approval.head_commit == *expected_head_commit
            {
                current_decisions.insert(approval.approver_principal_id, approval);
            }
        }
        let mut approval_ids = Vec::new();
        let mut change_request_ids = Vec::new();
        for decision in current_decisions.values() {
            match decision.decision {
                ReviewDecision::Approve => approval_ids.push(decision.approval_id),
                ReviewDecision::RequestChanges => {
                    change_request_ids.push(decision.approval_id);
                }
            }
        }
        let enough_approvals = usize::from(self.fields.required_approvals) <= approval_ids.len();
        let eligibility = if enough_approvals && change_request_ids.is_empty() {
            MergeEligibility::Ready
        } else {
            MergeEligibility::Blocked
        };
        Ok(MergeReadiness {
            review: self.fields.identity.clone(),
            revision: expected_revision,
            head_commit: expected_head_commit.clone(),
            eligibility,
            required_approvals: self.fields.required_approvals,
            approval_ids,
            change_request_ids,
        })
    }

    fn require_version(&self, expected: AggregateVersion) -> Result<(), ReviewError> {
        if self.fields.version != expected {
            return Err(ReviewError::StaleVersion {
                expected,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn require_revision(&self, expected: PatchRevisionNumber) -> Result<(), ReviewError> {
        let actual = self.current_revision_or_error()?.number;
        if actual != expected {
            return Err(ReviewError::StaleRevision { expected, actual });
        }
        Ok(())
    }

    fn require_head(&self, expected: &GitCommitId) -> Result<(), ReviewError> {
        let actual = &self.current_revision_or_error()?.head_commit;
        if actual != expected {
            return Err(ReviewError::StaleCommit {
                expected: expected.clone(),
                actual: actual.clone(),
            });
        }
        Ok(())
    }

    fn next_version(&self) -> Result<AggregateVersion, ReviewError> {
        self.fields
            .version
            .next()
            .ok_or(ReviewError::VersionExhausted)
    }

    fn current_revision_or_error(&self) -> Result<&PatchRevision, ReviewError> {
        self.current_revision()
            .ok_or(ReviewError::InvalidRevisionSequence)
    }
}

fn validate_record(fields: &ReviewRecordFields) -> Result<(), ReviewError> {
    ReviewIdentity::new(fields.identity.review_id, fields.identity.branch.clone())?;
    if fields.required_approvals > MAX_REQUIRED_APPROVALS {
        return Err(ReviewError::TooManyRequiredApprovals);
    }
    if fields.revisions.is_empty() || fields.revisions.len() > MAX_PATCH_REVISIONS {
        return Err(ReviewError::TooManyRevisions);
    }
    if fields.comments.len() > MAX_REVIEW_COMMENTS {
        return Err(ReviewError::TooManyComments);
    }
    if fields.approvals.len() > MAX_REVIEW_DECISIONS {
        return Err(ReviewError::TooManyDecisions);
    }
    let mut revision_ids = BTreeSet::new();
    let mut previous_revision: Option<PatchRevisionNumber> = None;
    let mut previous_timestamp = None;
    for revision in &fields.revisions {
        validate_revision(&fields.identity, revision)?;
        if !revision_ids.insert(revision.revision_id) {
            return Err(ReviewError::ConflictingRecordId);
        }
        match previous_revision {
            None if revision.number != PatchRevisionNumber::FIRST => {
                return Err(ReviewError::InvalidRevisionSequence);
            }
            Some(previous)
                if revision.number != previous.next().ok_or(ReviewError::RevisionExhausted)? =>
            {
                return Err(ReviewError::InvalidRevisionSequence);
            }
            _ => {}
        }
        if previous_timestamp.is_some_and(|timestamp| revision.created_at_millis < timestamp) {
            return Err(ReviewError::InvalidTimestamp);
        }
        previous_revision = Some(revision.number);
        previous_timestamp = Some(revision.created_at_millis);
    }
    let revisions_by_number = fields
        .revisions
        .iter()
        .map(|revision| (revision.number, revision))
        .collect::<BTreeMap<_, _>>();
    let mut record_ids = revision_ids;
    let mut previous_comment_timestamp = None;
    for comment in &fields.comments {
        if !record_ids.insert(comment.comment_id) {
            return Err(ReviewError::ConflictingRecordId);
        }
        let revision = revisions_by_number
            .get(&comment.anchor.revision)
            .ok_or(ReviewError::InvalidCommentAnchor)?;
        validate_comment(&fields.identity, comment, revision)?;
        if comment.created_at_millis < revision.created_at_millis
            || previous_comment_timestamp
                .is_some_and(|timestamp| comment.created_at_millis < timestamp)
        {
            return Err(ReviewError::InvalidTimestamp);
        }
        previous_comment_timestamp = Some(comment.created_at_millis);
    }
    let mut previous_approval_timestamp = None;
    for approval in &fields.approvals {
        if !record_ids.insert(approval.approval_id) {
            return Err(ReviewError::ConflictingRecordId);
        }
        let revision = revisions_by_number
            .get(&approval.revision)
            .ok_or(ReviewError::InvalidApproval)?;
        validate_approval(&fields.identity, approval, revision)?;
        if approval.created_at_millis < revision.created_at_millis
            || previous_approval_timestamp
                .is_some_and(|timestamp| approval.created_at_millis < timestamp)
        {
            return Err(ReviewError::InvalidTimestamp);
        }
        previous_approval_timestamp = Some(approval.created_at_millis);
    }
    let mutation_count = fields
        .revisions
        .len()
        .checked_sub(1)
        .and_then(|count| count.checked_add(fields.comments.len()))
        .and_then(|count| count.checked_add(fields.approvals.len()))
        .ok_or(ReviewError::VersionExhausted)?;
    let expected_version = u64::try_from(mutation_count)
        .ok()
        .and_then(|count| count.checked_add(1))
        .and_then(AggregateVersion::new)
        .ok_or(ReviewError::VersionExhausted)?;
    if fields.version != expected_version {
        return Err(ReviewError::InvalidRecordVersion);
    }
    Ok(())
}

fn validate_revision(
    identity: &ReviewIdentity,
    revision: &PatchRevision,
) -> Result<(), ReviewError> {
    if revision.revision_id.as_uuid().is_nil()
        || revision.author_principal_id.as_uuid().is_nil()
        || &revision.review != identity
        || revision.base_commit == revision.head_commit
    {
        return Err(ReviewError::InvalidRevision);
    }
    Ok(())
}

fn validate_comment(
    identity: &ReviewIdentity,
    comment: &ReviewComment,
    revision: &PatchRevision,
) -> Result<(), ReviewError> {
    if comment.comment_id.as_uuid().is_nil()
        || comment.author_principal_id.as_uuid().is_nil()
        || &comment.review != identity
        || comment.anchor.revision != revision.number
        || comment.anchor.end_line < comment.anchor.start_line
    {
        return Err(ReviewError::InvalidCommentAnchor);
    }
    let expected_commit = match comment.anchor.side {
        ReviewDiffSide::Base => &revision.base_commit,
        ReviewDiffSide::Head => &revision.head_commit,
    };
    if &comment.anchor.commit != expected_commit {
        return Err(ReviewError::InvalidCommentAnchor);
    }
    Ok(())
}

fn validate_approval(
    identity: &ReviewIdentity,
    approval: &ReviewApproval,
    revision: &PatchRevision,
) -> Result<(), ReviewError> {
    if approval.approval_id.as_uuid().is_nil()
        || approval.approver_principal_id.as_uuid().is_nil()
        || &approval.review != identity
        || approval.revision != revision.number
        || approval.head_commit != revision.head_commit
    {
        return Err(ReviewError::InvalidApproval);
    }
    Ok(())
}

fn is_safe_review_path(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_REVIEW_FILE_PATH_BYTES
        && !value.starts_with('/')
        && !value.ends_with('/')
        && !value.contains('\0')
        && !value.chars().any(char::is_control)
        && value
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReviewError {
    InvalidReviewId,
    InvalidRevision,
    InvalidRevisionSequence,
    InvalidFilePath,
    InvalidHunkId,
    InvalidCommentBody,
    InvalidCommentAnchor,
    InvalidApproval,
    InvalidTimestamp,
    InvalidRecordVersion,
    ConflictingRecordId,
    TooManyRequiredApprovals,
    TooManyRevisions,
    TooManyComments,
    TooManyDecisions,
    StaleVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    StaleRevision {
        expected: PatchRevisionNumber,
        actual: PatchRevisionNumber,
    },
    StaleCommit {
        expected: GitCommitId,
        actual: GitCommitId,
    },
    RevisionExhausted,
    VersionExhausted,
}

impl fmt::Display for ReviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReviewId => formatter.write_str("review identifier is invalid"),
            Self::InvalidRevision => formatter.write_str("patch revision is invalid"),
            Self::InvalidRevisionSequence => {
                formatter.write_str("patch revision sequence is invalid")
            }
            Self::InvalidFilePath => formatter.write_str("review file path is invalid"),
            Self::InvalidHunkId => formatter.write_str("review hunk identifier is invalid"),
            Self::InvalidCommentBody => formatter.write_str("review comment body is invalid"),
            Self::InvalidCommentAnchor => formatter.write_str("review comment anchor is invalid"),
            Self::InvalidApproval => formatter.write_str("review approval is invalid"),
            Self::InvalidTimestamp => formatter.write_str("review timestamp is invalid"),
            Self::InvalidRecordVersion => formatter.write_str("review record version is invalid"),
            Self::ConflictingRecordId => formatter.write_str("review record identifier conflicts"),
            Self::TooManyRequiredApprovals => {
                formatter.write_str("required review approval count is too large")
            }
            Self::TooManyRevisions => formatter.write_str("review has too many revisions"),
            Self::TooManyComments => formatter.write_str("review has too many comments"),
            Self::TooManyDecisions => formatter.write_str("review has too many decisions"),
            Self::StaleVersion { .. } => formatter.write_str("review version is stale"),
            Self::StaleRevision { .. } => formatter.write_str("patch revision is stale"),
            Self::StaleCommit { .. } => formatter.write_str("patch commit is stale"),
            Self::RevisionExhausted => formatter.write_str("patch revision is exhausted"),
            Self::VersionExhausted => formatter.write_str("review version is exhausted"),
        }
    }
}

impl Error for ReviewError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BranchGeneration, BranchRefName, CommunityId};
    use uuid::Uuid;

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn commit(value: u64) -> GitCommitId {
        GitCommitId::parse(format!("{value:040x}")).expect("valid commit")
    }

    fn identity() -> ReviewIdentity {
        ReviewIdentity::new(
            aggregate(3),
            BranchCollaborationIdentity::new(
                CommunityId::from_uuid(Uuid::from_u128(1)),
                aggregate(2),
                BranchRefName::parse("refs/heads/feature/review").expect("valid branch"),
                BranchGeneration::FIRST,
            )
            .expect("valid branch identity"),
        )
        .expect("valid review identity")
    }

    fn review(required_approvals: u16) -> Review {
        Review::open(
            identity(),
            required_approvals,
            revision_input(10, 101, 1_000),
        )
        .expect("valid review")
    }

    fn revision_input(
        revision_id: u128,
        head_commit: u64,
        created_at_millis: u64,
    ) -> PatchRevisionInput {
        PatchRevisionInput {
            revision_id: aggregate(revision_id),
            base_commit: commit(100),
            head_commit: commit(head_commit),
            author_principal_id: principal(10),
            created_at_millis,
        }
    }

    fn decision_input(
        approval_id: u128,
        approver: u128,
        decision: ReviewDecision,
        created_at_millis: u64,
    ) -> ReviewDecisionInput {
        ReviewDecisionInput {
            approval_id: aggregate(approval_id),
            revision: PatchRevisionNumber::FIRST,
            head_commit: commit(101),
            approver_principal_id: principal(approver),
            decision,
            created_at_millis,
        }
    }

    fn comment_input(
        comment_id: u128,
        body: &str,
        anchor: ReviewCommentAnchor,
        created_at_millis: u64,
    ) -> ReviewCommentInput {
        ReviewCommentInput {
            comment_id: aggregate(comment_id),
            author_principal_id: principal(20),
            body: ReviewCommentBody::new(body).expect("valid body"),
            anchor,
            created_at_millis,
        }
    }

    fn head_anchor(revision: PatchRevisionNumber, head: GitCommitId) -> ReviewCommentAnchor {
        ReviewCommentAnchor::new(
            revision,
            head,
            ReviewFilePath::new("src/review.rs").expect("valid path"),
            ReviewHunkId::parse("a".repeat(REVIEW_HUNK_ID_BYTES)).expect("valid hunk"),
            ReviewDiffSide::Head,
            NonZeroU32::new(40).expect("nonzero line"),
            NonZeroU32::new(45).expect("nonzero line"),
        )
        .expect("valid anchor")
    }

    #[test]
    fn stale_revision_cannot_accept_comment_or_approval() {
        let mut review = review(1);
        assert_eq!(
            review.submit_revision(
                AggregateVersion::FIRST,
                PatchRevisionNumber::FIRST,
                revision_input(11, 102, 2_000),
            ),
            Ok(ReviewCommandOutcome::Applied)
        );
        let current_version = review.fields().version;
        assert!(matches!(
            review.record_decision(
                current_version,
                decision_input(20, 20, ReviewDecision::Approve, 2_100),
            ),
            Err(ReviewError::StaleRevision { .. })
        ));
        assert!(matches!(
            review.add_comment(
                current_version,
                comment_input(
                    21,
                    "stale note",
                    head_anchor(PatchRevisionNumber::FIRST, commit(101)),
                    2_100,
                ),
            ),
            Err(ReviewError::InvalidCommentAnchor)
        ));
        assert_eq!(review.fields().version, current_version);
    }

    #[test]
    fn new_patch_revision_supersedes_existing_approval() {
        let mut review = review(1);
        let approval_id = aggregate(20);
        assert_eq!(
            review.record_decision(
                AggregateVersion::FIRST,
                decision_input(20, 20, ReviewDecision::Approve, 1_100),
            ),
            Ok(ReviewCommandOutcome::Applied)
        );
        assert_eq!(
            review.submit_revision(
                AggregateVersion::new(2).expect("version two"),
                PatchRevisionNumber::FIRST,
                revision_input(11, 102, 2_000),
            ),
            Ok(ReviewCommandOutcome::Applied)
        );
        assert_eq!(
            review.submit_revision(
                AggregateVersion::new(2).expect("version two"),
                PatchRevisionNumber::FIRST,
                revision_input(11, 102, 2_000),
            ),
            Ok(ReviewCommandOutcome::Unchanged)
        );
        assert_eq!(
            review.approval_applicability(approval_id),
            Some(ApprovalApplicability::SupersededByRevision {
                current_revision: PatchRevisionNumber::new(2).expect("revision two"),
            })
        );
        assert_eq!(
            review
                .merge_readiness(
                    PatchRevisionNumber::new(2).expect("revision two"),
                    &commit(102),
                )
                .expect("current readiness")
                .eligibility,
            MergeEligibility::Blocked
        );
    }

    #[test]
    fn comment_anchor_binds_revision_commit_file_hunk_side_and_range() {
        let mut review = review(0);
        let anchor = head_anchor(PatchRevisionNumber::FIRST, commit(101));
        assert_eq!(
            review.add_comment(
                AggregateVersion::FIRST,
                comment_input(30, "Please keep this error visible.", anchor.clone(), 1_100,),
            ),
            Ok(ReviewCommandOutcome::Applied)
        );
        assert_eq!(
            review
                .fields()
                .comments
                .first()
                .map(|comment| &comment.anchor),
            Some(&anchor)
        );
        let stale_commit_anchor = head_anchor(PatchRevisionNumber::FIRST, commit(999));
        assert_eq!(
            review.add_comment(
                AggregateVersion::new(2).expect("version two"),
                comment_input(31, "wrong commit", stale_commit_anchor, 1_200),
            ),
            Err(ReviewError::InvalidCommentAnchor)
        );
    }

    #[test]
    fn merge_eligibility_requires_current_distinct_approvals_and_no_change_request() {
        let mut review = review(2);
        assert_eq!(
            review
                .merge_readiness(PatchRevisionNumber::FIRST, &commit(101))
                .expect("readiness")
                .eligibility,
            MergeEligibility::Blocked
        );
        review
            .record_decision(
                AggregateVersion::FIRST,
                decision_input(40, 20, ReviewDecision::Approve, 1_100),
            )
            .expect("first approval");
        review
            .record_decision(
                AggregateVersion::new(2).expect("version two"),
                decision_input(41, 21, ReviewDecision::RequestChanges, 1_200),
            )
            .expect("change request");
        let blocked = review
            .merge_readiness(PatchRevisionNumber::FIRST, &commit(101))
            .expect("blocked readiness");
        assert_eq!(blocked.eligibility, MergeEligibility::Blocked);
        assert_eq!(blocked.approval_ids, vec![aggregate(40)]);
        assert_eq!(blocked.change_request_ids, vec![aggregate(41)]);

        review
            .record_decision(
                AggregateVersion::new(3).expect("version three"),
                decision_input(42, 21, ReviewDecision::Approve, 1_300),
            )
            .expect("replacement approval");
        let ready = review
            .merge_readiness(PatchRevisionNumber::FIRST, &commit(101))
            .expect("ready review");
        assert_eq!(ready.eligibility, MergeEligibility::Ready);
        assert_eq!(ready.approval_ids, vec![aggregate(40), aggregate(42)]);
        assert!(ready.change_request_ids.is_empty());
        assert_eq!(
            review.approval_applicability(aggregate(41)),
            Some(ApprovalApplicability::SupersededByDecision {
                current_decision_id: aggregate(42),
            })
        );
        assert!(matches!(
            review.merge_readiness(PatchRevisionNumber::FIRST, &commit(999)),
            Err(ReviewError::StaleCommit { .. })
        ));
    }
}
