use std::{collections::BTreeSet, error::Error, fmt};

use crate::{
    AggregateId, AggregateVersion, GitCommitId, PatchRevision, PatchRevisionNumber, ReviewIdentity,
};

const MAX_CI_LABEL_BYTES: usize = 256;
const MAX_CI_OUTPUT_BYTES: usize = 16 * 1_024;
const MAX_CI_LINK_BYTES: usize = 2_048;
const MAX_CI_RUNS: usize = 1_000;
const MAX_CI_ARTIFACTS_PER_RUN: usize = 256;
const SHA256_HEX_BYTES: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiLabel {
    value: String,
    truncated: bool,
    sanitized: bool,
}

impl CiLabel {
    pub fn from_untrusted(value: &str) -> Result<Self, CiStatusError> {
        let bounded = bound_untrusted_text(value, MAX_CI_LABEL_BYTES);
        if bounded.value.trim().is_empty() {
            return Err(CiStatusError::InvalidText);
        }
        Ok(Self {
            value: bounded.value,
            truncated: bounded.truncated,
            sanitized: bounded.sanitized,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }

    pub fn from_record(
        value: String,
        truncated: bool,
        sanitized: bool,
    ) -> Result<Self, CiStatusError> {
        validate_stored_text(&value, MAX_CI_LABEL_BYTES, false)?;
        Ok(Self {
            value,
            truncated,
            sanitized,
        })
    }

    pub const fn was_truncated(&self) -> bool {
        self.truncated
    }

    pub const fn was_sanitized(&self) -> bool {
        self.sanitized
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiOutputText {
    value: String,
    truncated: bool,
    sanitized: bool,
}

impl CiOutputText {
    pub fn from_untrusted(value: &str) -> Self {
        let bounded = bound_untrusted_text(value, MAX_CI_OUTPUT_BYTES);
        Self {
            value: bounded.value,
            truncated: bounded.truncated,
            sanitized: bounded.sanitized,
        }
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }

    pub fn from_record(
        value: String,
        truncated: bool,
        sanitized: bool,
    ) -> Result<Self, CiStatusError> {
        validate_stored_text(&value, MAX_CI_OUTPUT_BYTES, true)?;
        Ok(Self {
            value,
            truncated,
            sanitized,
        })
    }

    pub const fn was_truncated(&self) -> bool {
        self.truncated
    }

    pub const fn was_sanitized(&self) -> bool {
        self.sanitized
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CiExternalLink(String);

impl CiExternalLink {
    pub fn parse(value: impl Into<String>) -> Result<Self, CiStatusError> {
        let value = value.into();
        let target = value
            .strip_prefix("https://")
            .ok_or(CiStatusError::InvalidLink)?;
        if target.is_empty()
            || value.len() > MAX_CI_LINK_BYTES
            || value.chars().any(|character| {
                character.is_control() || character.is_whitespace() || character == '\\'
            })
        {
            return Err(CiStatusError::InvalidLink);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CiArtifactDigest(String);

impl CiArtifactDigest {
    pub fn parse(value: impl Into<String>) -> Result<Self, CiStatusError> {
        let value = value.into();
        if value.len() != SHA256_HEX_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CiStatusError::InvalidArtifactDigest);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiArtifactLink {
    pub artifact_id: AggregateId,
    pub label: CiLabel,
    pub url: CiExternalLink,
    pub digest: Option<CiArtifactDigest>,
}

impl CiArtifactLink {
    pub fn new(
        artifact_id: AggregateId,
        label: CiLabel,
        url: CiExternalLink,
        digest: Option<CiArtifactDigest>,
    ) -> Result<Self, CiStatusError> {
        if artifact_id.as_uuid().is_nil() {
            return Err(CiStatusError::InvalidArtifact);
        }
        Ok(Self {
            artifact_id,
            label,
            url,
            digest,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiWorkflowLink {
    pub workflow_id: AggregateId,
    pub workflow_run_id: AggregateId,
    pub label: CiLabel,
    pub url: Option<CiExternalLink>,
}

impl CiWorkflowLink {
    pub fn new(
        workflow_id: AggregateId,
        workflow_run_id: AggregateId,
        label: CiLabel,
        url: Option<CiExternalLink>,
    ) -> Result<Self, CiStatusError> {
        if workflow_id.as_uuid().is_nil() || workflow_run_id.as_uuid().is_nil() {
            return Err(CiStatusError::InvalidWorkflowLink);
        }
        Ok(Self {
            workflow_id,
            workflow_run_id,
            label,
            url,
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CiCheckSuiteIdentity {
    suite_id: AggregateId,
    review: ReviewIdentity,
    revision: PatchRevisionNumber,
    head_commit: GitCommitId,
}

impl CiCheckSuiteIdentity {
    pub fn for_revision(
        suite_id: AggregateId,
        revision: &PatchRevision,
    ) -> Result<Self, CiStatusError> {
        Self::new(
            suite_id,
            revision.review.clone(),
            revision.number,
            revision.head_commit.clone(),
        )
    }

    pub fn new(
        suite_id: AggregateId,
        review: ReviewIdentity,
        revision: PatchRevisionNumber,
        head_commit: GitCommitId,
    ) -> Result<Self, CiStatusError> {
        if suite_id.as_uuid().is_nil() {
            return Err(CiStatusError::InvalidSuiteId);
        }
        Ok(Self {
            suite_id,
            review,
            revision,
            head_commit,
        })
    }

    pub const fn suite_id(&self) -> AggregateId {
        self.suite_id
    }

    pub const fn review(&self) -> &ReviewIdentity {
        &self.review
    }

    pub const fn revision(&self) -> PatchRevisionNumber {
        self.revision
    }

    pub const fn head_commit(&self) -> &GitCommitId {
        &self.head_commit
    }

    pub const fn repository_id(&self) -> AggregateId {
        self.review.repository_id()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CiCheckStatus {
    Pending,
    Running,
    Success,
    Failure,
    Cancelled,
}

impl CiCheckStatus {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Success | Self::Failure | Self::Cancelled)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiCheckRun {
    pub check_run_id: AggregateId,
    pub suite: CiCheckSuiteIdentity,
    pub label: CiLabel,
    pub status: CiCheckStatus,
    pub output: Option<CiOutputText>,
    pub artifacts: Vec<CiArtifactLink>,
    pub queued_at_millis: u64,
    pub started_at_millis: Option<u64>,
    pub completed_at_millis: Option<u64>,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiCheckRunInput {
    pub check_run_id: AggregateId,
    pub label: CiLabel,
    pub queued_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiCheckRunCompletionInput {
    pub status: CiCheckStatus,
    pub output: CiOutputText,
    pub artifacts: Vec<CiArtifactLink>,
    pub completed_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiCheckSuiteRecordFields {
    pub identity: CiCheckSuiteIdentity,
    pub workflow: CiWorkflowLink,
    pub runs: Vec<CiCheckRun>,
    pub created_at_millis: u64,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CiStatusCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiCheckSuite {
    fields: CiCheckSuiteRecordFields,
}

impl CiCheckSuite {
    pub fn create(
        identity: CiCheckSuiteIdentity,
        workflow: CiWorkflowLink,
        created_at_millis: u64,
    ) -> Self {
        Self {
            fields: CiCheckSuiteRecordFields {
                identity,
                workflow,
                runs: Vec::new(),
                created_at_millis,
                version: AggregateVersion::FIRST,
            },
        }
    }

    pub fn from_record(fields: CiCheckSuiteRecordFields) -> Result<Self, CiStatusError> {
        validate_record(&fields)?;
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &CiCheckSuiteRecordFields {
        &self.fields
    }

    pub fn status(&self) -> CiCheckStatus {
        if self.fields.runs.is_empty() {
            return CiCheckStatus::Pending;
        }
        if self
            .fields
            .runs
            .iter()
            .any(|run| run.status == CiCheckStatus::Running)
        {
            return CiCheckStatus::Running;
        }
        if self
            .fields
            .runs
            .iter()
            .any(|run| run.status == CiCheckStatus::Pending)
        {
            return CiCheckStatus::Pending;
        }
        if self
            .fields
            .runs
            .iter()
            .any(|run| run.status == CiCheckStatus::Failure)
        {
            return CiCheckStatus::Failure;
        }
        if self
            .fields
            .runs
            .iter()
            .any(|run| run.status == CiCheckStatus::Cancelled)
        {
            return CiCheckStatus::Cancelled;
        }
        CiCheckStatus::Success
    }

    pub fn add_run(
        &mut self,
        expected_version: AggregateVersion,
        input: CiCheckRunInput,
    ) -> Result<CiStatusCommandOutcome, CiStatusError> {
        if let Some(existing) = self
            .fields
            .runs
            .iter()
            .find(|run| run.check_run_id == input.check_run_id)
        {
            return if existing.suite == self.fields.identity
                && existing.label == input.label
                && existing.queued_at_millis == input.queued_at_millis
            {
                Ok(CiStatusCommandOutcome::Unchanged)
            } else {
                Err(CiStatusError::ConflictingRunId)
            };
        }
        self.require_version(expected_version)?;
        if input.check_run_id.as_uuid().is_nil() {
            return Err(CiStatusError::InvalidRun);
        }
        if self.fields.runs.len() >= MAX_CI_RUNS {
            return Err(CiStatusError::TooManyRuns);
        }
        if input.queued_at_millis < self.fields.created_at_millis {
            return Err(CiStatusError::InvalidTimestamp);
        }
        self.fields.version = self.next_version()?;
        self.fields.runs.push(CiCheckRun {
            check_run_id: input.check_run_id,
            suite: self.fields.identity.clone(),
            label: input.label,
            status: CiCheckStatus::Pending,
            output: None,
            artifacts: Vec::new(),
            queued_at_millis: input.queued_at_millis,
            started_at_millis: None,
            completed_at_millis: None,
            version: AggregateVersion::FIRST,
        });
        Ok(CiStatusCommandOutcome::Applied)
    }

    pub fn start_run(
        &mut self,
        expected_suite_version: AggregateVersion,
        check_run_id: AggregateId,
        expected_run_version: AggregateVersion,
        started_at_millis: u64,
    ) -> Result<CiStatusCommandOutcome, CiStatusError> {
        let run_index = self.run_index(check_run_id)?;
        let run = self
            .fields
            .runs
            .get(run_index)
            .ok_or(CiStatusError::RunNotFound)?;
        if run.status == CiCheckStatus::Running && run.started_at_millis == Some(started_at_millis)
        {
            return Ok(CiStatusCommandOutcome::Unchanged);
        }
        self.require_version(expected_suite_version)?;
        if run.version != expected_run_version {
            return Err(CiStatusError::StaleRunVersion {
                expected: expected_run_version,
                actual: run.version,
            });
        }
        if run.status != CiCheckStatus::Pending {
            return Err(CiStatusError::InvalidTransition);
        }
        if started_at_millis < run.queued_at_millis {
            return Err(CiStatusError::InvalidTimestamp);
        }
        let next_suite_version = self.next_version()?;
        let next_run_version = run.version.next().ok_or(CiStatusError::VersionExhausted)?;
        let run = self
            .fields
            .runs
            .get_mut(run_index)
            .ok_or(CiStatusError::RunNotFound)?;
        run.status = CiCheckStatus::Running;
        run.started_at_millis = Some(started_at_millis);
        run.version = next_run_version;
        self.fields.version = next_suite_version;
        Ok(CiStatusCommandOutcome::Applied)
    }

    pub fn complete_run(
        &mut self,
        expected_suite_version: AggregateVersion,
        check_run_id: AggregateId,
        expected_run_version: AggregateVersion,
        expected_head_commit: &GitCommitId,
        completion: CiCheckRunCompletionInput,
    ) -> Result<CiStatusCommandOutcome, CiStatusError> {
        self.require_head(expected_head_commit)?;
        validate_completion(&completion)?;
        let run_index = self.run_index(check_run_id)?;
        let run = self
            .fields
            .runs
            .get(run_index)
            .ok_or(CiStatusError::RunNotFound)?;
        if run.status == completion.status
            && run.output.as_ref() == Some(&completion.output)
            && run.artifacts == completion.artifacts
            && run.completed_at_millis == Some(completion.completed_at_millis)
        {
            return Ok(CiStatusCommandOutcome::Unchanged);
        }
        self.require_version(expected_suite_version)?;
        if run.version != expected_run_version {
            return Err(CiStatusError::StaleRunVersion {
                expected: expected_run_version,
                actual: run.version,
            });
        }
        if run.status.is_terminal() {
            return Err(CiStatusError::InvalidTransition);
        }
        let earliest_completion = run.started_at_millis.unwrap_or(run.queued_at_millis);
        if completion.completed_at_millis < earliest_completion {
            return Err(CiStatusError::InvalidTimestamp);
        }
        let next_suite_version = self.next_version()?;
        let next_run_version = run.version.next().ok_or(CiStatusError::VersionExhausted)?;
        let run = self
            .fields
            .runs
            .get_mut(run_index)
            .ok_or(CiStatusError::RunNotFound)?;
        run.status = completion.status;
        run.output = Some(completion.output);
        run.artifacts = completion.artifacts;
        run.completed_at_millis = Some(completion.completed_at_millis);
        run.version = next_run_version;
        self.fields.version = next_suite_version;
        Ok(CiStatusCommandOutcome::Applied)
    }

    fn run_index(&self, check_run_id: AggregateId) -> Result<usize, CiStatusError> {
        self.fields
            .runs
            .iter()
            .position(|run| run.check_run_id == check_run_id)
            .ok_or(CiStatusError::RunNotFound)
    }

    fn require_version(&self, expected: AggregateVersion) -> Result<(), CiStatusError> {
        if self.fields.version != expected {
            return Err(CiStatusError::StaleSuiteVersion {
                expected,
                actual: self.fields.version,
            });
        }
        Ok(())
    }

    fn require_head(&self, expected: &GitCommitId) -> Result<(), CiStatusError> {
        if self.fields.identity.head_commit != *expected {
            return Err(CiStatusError::StaleCommit {
                expected: expected.clone(),
                actual: self.fields.identity.head_commit.clone(),
            });
        }
        Ok(())
    }

    fn next_version(&self) -> Result<AggregateVersion, CiStatusError> {
        self.fields
            .version
            .next()
            .ok_or(CiStatusError::VersionExhausted)
    }
}

fn validate_record(fields: &CiCheckSuiteRecordFields) -> Result<(), CiStatusError> {
    CiCheckSuiteIdentity::new(
        fields.identity.suite_id,
        fields.identity.review.clone(),
        fields.identity.revision,
        fields.identity.head_commit.clone(),
    )?;
    CiWorkflowLink::new(
        fields.workflow.workflow_id,
        fields.workflow.workflow_run_id,
        fields.workflow.label.clone(),
        fields.workflow.url.clone(),
    )?;
    if fields.runs.len() > MAX_CI_RUNS {
        return Err(CiStatusError::TooManyRuns);
    }
    let mut run_ids = BTreeSet::new();
    let mut mutation_count = fields.runs.len();
    for run in &fields.runs {
        if !run_ids.insert(run.check_run_id) {
            return Err(CiStatusError::ConflictingRunId);
        }
        validate_run(&fields.identity, fields.created_at_millis, run)?;
        let run_mutations = usize::try_from(run.version.get())
            .ok()
            .and_then(|version| version.checked_sub(1))
            .ok_or(CiStatusError::VersionExhausted)?;
        mutation_count = mutation_count
            .checked_add(run_mutations)
            .ok_or(CiStatusError::VersionExhausted)?;
    }
    let expected_version = u64::try_from(mutation_count)
        .ok()
        .and_then(|count| count.checked_add(1))
        .and_then(AggregateVersion::new)
        .ok_or(CiStatusError::VersionExhausted)?;
    if fields.version != expected_version {
        return Err(CiStatusError::InvalidRecordVersion);
    }
    Ok(())
}

fn validate_run(
    identity: &CiCheckSuiteIdentity,
    suite_created_at_millis: u64,
    run: &CiCheckRun,
) -> Result<(), CiStatusError> {
    if run.check_run_id.as_uuid().is_nil()
        || &run.suite != identity
        || run.queued_at_millis < suite_created_at_millis
    {
        return Err(CiStatusError::InvalidRun);
    }
    let expected_version = match run.status {
        CiCheckStatus::Pending => {
            if run.started_at_millis.is_some()
                || run.completed_at_millis.is_some()
                || run.output.is_some()
                || !run.artifacts.is_empty()
            {
                return Err(CiStatusError::InvalidRun);
            }
            AggregateVersion::FIRST
        }
        CiCheckStatus::Running => {
            if run.started_at_millis.is_none()
                || run.completed_at_millis.is_some()
                || run.output.is_some()
                || !run.artifacts.is_empty()
            {
                return Err(CiStatusError::InvalidRun);
            }
            AggregateVersion::new(2).ok_or(CiStatusError::VersionExhausted)?
        }
        CiCheckStatus::Success | CiCheckStatus::Failure | CiCheckStatus::Cancelled => {
            if run.completed_at_millis.is_none() || run.output.is_none() {
                return Err(CiStatusError::InvalidRun);
            }
            validate_artifacts(&run.artifacts)?;
            if run.started_at_millis.is_some() {
                AggregateVersion::new(3).ok_or(CiStatusError::VersionExhausted)?
            } else {
                AggregateVersion::new(2).ok_or(CiStatusError::VersionExhausted)?
            }
        }
    };
    if run.version != expected_version
        || run
            .started_at_millis
            .is_some_and(|started| started < run.queued_at_millis)
        || run.completed_at_millis.is_some_and(|completed| {
            completed < run.started_at_millis.unwrap_or(run.queued_at_millis)
        })
    {
        return Err(CiStatusError::InvalidRun);
    }
    Ok(())
}

fn validate_completion(completion: &CiCheckRunCompletionInput) -> Result<(), CiStatusError> {
    if !completion.status.is_terminal() {
        return Err(CiStatusError::InvalidTransition);
    }
    validate_artifacts(&completion.artifacts)
}

fn validate_artifacts(artifacts: &[CiArtifactLink]) -> Result<(), CiStatusError> {
    if artifacts.len() > MAX_CI_ARTIFACTS_PER_RUN {
        return Err(CiStatusError::TooManyArtifacts);
    }
    let mut artifact_ids = BTreeSet::new();
    for artifact in artifacts {
        if artifact.artifact_id.as_uuid().is_nil() || !artifact_ids.insert(artifact.artifact_id) {
            return Err(CiStatusError::InvalidArtifact);
        }
    }
    Ok(())
}

struct BoundedText {
    value: String,
    truncated: bool,
    sanitized: bool,
}

fn bound_untrusted_text(value: &str, max_bytes: usize) -> BoundedText {
    let inspection_limit = max_bytes.saturating_mul(4);
    let mut bounded = String::with_capacity(value.len().min(max_bytes));
    let mut inspected_bytes = 0usize;
    let mut truncated = false;
    let mut sanitized = false;
    for character in value.chars() {
        inspected_bytes = inspected_bytes.saturating_add(character.len_utf8());
        if inspected_bytes > inspection_limit {
            truncated = true;
            break;
        }
        if character.is_control() && !matches!(character, '\n' | '\t') {
            sanitized = true;
            continue;
        }
        if bounded.len().saturating_add(character.len_utf8()) > max_bytes {
            truncated = true;
            break;
        }
        bounded.push(character);
    }
    if !truncated && inspected_bytes < value.len() {
        truncated = true;
    }
    BoundedText {
        value: bounded,
        truncated,
        sanitized,
    }
}

fn validate_stored_text(
    value: &str,
    max_bytes: usize,
    allow_empty: bool,
) -> Result<(), CiStatusError> {
    if value.len() > max_bytes
        || (!allow_empty && value.trim().is_empty())
        || value
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\t'))
    {
        return Err(CiStatusError::InvalidText);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CiStatusError {
    InvalidSuiteId,
    InvalidWorkflowLink,
    InvalidRun,
    InvalidArtifact,
    InvalidArtifactDigest,
    InvalidLink,
    InvalidText,
    InvalidTimestamp,
    InvalidTransition,
    InvalidRecordVersion,
    ConflictingRunId,
    RunNotFound,
    TooManyRuns,
    TooManyArtifacts,
    StaleSuiteVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    StaleRunVersion {
        expected: AggregateVersion,
        actual: AggregateVersion,
    },
    StaleCommit {
        expected: GitCommitId,
        actual: GitCommitId,
    },
    VersionExhausted,
}

impl fmt::Display for CiStatusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSuiteId => formatter.write_str("CI suite identifier is invalid"),
            Self::InvalidWorkflowLink => formatter.write_str("CI workflow link is invalid"),
            Self::InvalidRun => formatter.write_str("CI check run is invalid"),
            Self::InvalidArtifact => formatter.write_str("CI artifact is invalid"),
            Self::InvalidArtifactDigest => formatter.write_str("CI artifact digest is invalid"),
            Self::InvalidLink => formatter.write_str("CI external link is invalid"),
            Self::InvalidText => formatter.write_str("CI text is invalid"),
            Self::InvalidTimestamp => formatter.write_str("CI timestamp is invalid"),
            Self::InvalidTransition => formatter.write_str("CI status transition is invalid"),
            Self::InvalidRecordVersion => formatter.write_str("CI record version is invalid"),
            Self::ConflictingRunId => formatter.write_str("CI check run identifier conflicts"),
            Self::RunNotFound => formatter.write_str("CI check run was not found"),
            Self::TooManyRuns => formatter.write_str("CI suite has too many check runs"),
            Self::TooManyArtifacts => formatter.write_str("CI check run has too many artifacts"),
            Self::StaleSuiteVersion { .. } => formatter.write_str("CI suite version is stale"),
            Self::StaleRunVersion { .. } => formatter.write_str("CI check run version is stale"),
            Self::StaleCommit { .. } => formatter.write_str("CI commit is stale"),
            Self::VersionExhausted => formatter.write_str("CI version is exhausted"),
        }
    }
}

impl Error for CiStatusError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BranchCollaborationIdentity, BranchGeneration, BranchRefName, CommunityId,
        PatchRevisionInput, PrincipalId, Review,
    };
    use uuid::Uuid;

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn commit(value: u64) -> GitCommitId {
        GitCommitId::parse(format!("{value:040x}")).expect("valid commit")
    }

    fn revision() -> PatchRevision {
        let identity = ReviewIdentity::new(
            aggregate(3),
            BranchCollaborationIdentity::new(
                CommunityId::from_uuid(Uuid::from_u128(1)),
                aggregate(2),
                BranchRefName::parse("refs/heads/feature/ci").expect("valid branch"),
                BranchGeneration::FIRST,
            )
            .expect("valid branch identity"),
        )
        .expect("valid review identity");
        let review = Review::open(
            identity,
            1,
            PatchRevisionInput {
                revision_id: aggregate(4),
                base_commit: commit(100),
                head_commit: commit(101),
                author_principal_id: PrincipalId::from_uuid(Uuid::from_u128(5)),
                created_at_millis: 1_000,
            },
        )
        .expect("valid review");
        review.current_revision().expect("current revision").clone()
    }

    fn suite() -> CiCheckSuite {
        CiCheckSuite::create(
            CiCheckSuiteIdentity::for_revision(aggregate(10), &revision())
                .expect("valid suite identity"),
            CiWorkflowLink::new(
                aggregate(11),
                aggregate(12),
                CiLabel::from_untrusted("CI").expect("valid label"),
                Some(
                    CiExternalLink::parse("https://ci.example/runs/12")
                        .expect("valid workflow URL"),
                ),
            )
            .expect("valid workflow link"),
            1_100,
        )
    }

    fn run_input(run_id: u128, label: &str) -> CiCheckRunInput {
        CiCheckRunInput {
            check_run_id: aggregate(run_id),
            label: CiLabel::from_untrusted(label).expect("valid label"),
            queued_at_millis: 1_200,
        }
    }

    fn completion(status: CiCheckStatus, completed_at_millis: u64) -> CiCheckRunCompletionInput {
        CiCheckRunCompletionInput {
            status,
            output: CiOutputText::from_untrusted("finished"),
            artifacts: Vec::new(),
            completed_at_millis,
        }
    }

    #[test]
    fn pending_run_transitions_to_success_with_workflow_and_artifact_links() {
        let mut suite = suite();
        assert_eq!(suite.status(), CiCheckStatus::Pending);
        suite
            .add_run(AggregateVersion::FIRST, run_input(20, "build"))
            .expect("add run");
        assert_eq!(suite.status(), CiCheckStatus::Pending);
        suite
            .start_run(
                AggregateVersion::new(2).expect("suite version two"),
                aggregate(20),
                AggregateVersion::FIRST,
                1_300,
            )
            .expect("start run");
        assert_eq!(suite.status(), CiCheckStatus::Running);
        let artifact = CiArtifactLink::new(
            aggregate(30),
            CiLabel::from_untrusted("test report").expect("valid artifact label"),
            CiExternalLink::parse("https://artifacts.example/report").expect("valid artifact URL"),
            Some(CiArtifactDigest::parse("a".repeat(64)).expect("valid digest")),
        )
        .expect("valid artifact");
        suite
            .complete_run(
                AggregateVersion::new(3).expect("suite version three"),
                aggregate(20),
                AggregateVersion::new(2).expect("run version two"),
                &commit(101),
                CiCheckRunCompletionInput {
                    status: CiCheckStatus::Success,
                    output: CiOutputText::from_untrusted("47 tests passed"),
                    artifacts: vec![artifact.clone()],
                    completed_at_millis: 1_400,
                },
            )
            .expect("complete run");
        assert_eq!(suite.status(), CiCheckStatus::Success);
        assert_eq!(
            suite
                .fields()
                .runs
                .first()
                .map(|run| run.artifacts.as_slice()),
            Some([artifact].as_slice())
        );
    }

    #[test]
    fn terminal_suite_status_covers_failure_and_cancel_without_reopening_runs() {
        let mut failed = suite();
        failed
            .add_run(AggregateVersion::FIRST, run_input(20, "test"))
            .expect("add failure run");
        failed
            .complete_run(
                AggregateVersion::new(2).expect("suite version two"),
                aggregate(20),
                AggregateVersion::FIRST,
                &commit(101),
                completion(CiCheckStatus::Failure, 1_300),
            )
            .expect("fail run");
        assert_eq!(failed.status(), CiCheckStatus::Failure);
        assert_eq!(
            failed.complete_run(
                AggregateVersion::new(3).expect("suite version three"),
                aggregate(20),
                AggregateVersion::new(2).expect("run version two"),
                &commit(101),
                completion(CiCheckStatus::Success, 1_400),
            ),
            Err(CiStatusError::InvalidTransition)
        );

        let mut cancelled = suite();
        cancelled
            .add_run(AggregateVersion::FIRST, run_input(21, "lint"))
            .expect("add cancelled run");
        cancelled
            .complete_run(
                AggregateVersion::new(2).expect("suite version two"),
                aggregate(21),
                AggregateVersion::FIRST,
                &commit(101),
                completion(CiCheckStatus::Cancelled, 1_300),
            )
            .expect("cancel run");
        assert_eq!(cancelled.status(), CiCheckStatus::Cancelled);
    }

    #[test]
    fn stale_commit_rejects_terminal_result_without_mutating_pending_run() {
        let mut suite = suite();
        suite
            .add_run(AggregateVersion::FIRST, run_input(20, "build"))
            .expect("add run");
        let before = suite.clone();
        assert!(matches!(
            suite.complete_run(
                AggregateVersion::new(2).expect("suite version two"),
                aggregate(20),
                AggregateVersion::FIRST,
                &commit(999),
                completion(CiCheckStatus::Success, 1_300),
            ),
            Err(CiStatusError::StaleCommit { .. })
        ));
        assert_eq!(suite, before);
    }

    #[test]
    fn malicious_provider_output_is_sanitized_and_truncated_at_utf8_boundary() {
        let malicious = format!(
            "\u{1b}[31m\0<script>{}</script>🦀",
            "é".repeat(MAX_CI_OUTPUT_BYTES)
        );
        let output = CiOutputText::from_untrusted(&malicious);

        assert!(output.was_sanitized());
        assert!(output.was_truncated());
        assert!(output.as_str().len() <= MAX_CI_OUTPUT_BYTES);
        assert!(output.as_str().is_char_boundary(output.as_str().len()));
        assert!(!output.as_str().contains('\u{1b}'));
        assert!(!output.as_str().contains('\0'));
        assert!(CiExternalLink::parse("http://artifacts.example/report").is_err());
        assert!(CiExternalLink::parse("https://artifacts.example/report\nsecret").is_err());
    }
}
