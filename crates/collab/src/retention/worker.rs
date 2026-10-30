use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    CommunityRetentionPolicy, LegalHoldSnapshot, RetentionArchiveSnapshot, RetentionDecision,
    RetentionDisposition, RetentionError, RetentionReason, RetentionRecord, RetentionRequest,
    RetentionSnapshot, RetentionVisibility, TenantContext, resolve_retention,
};

pub const MAX_RETENTION_BATCH_SIZE: usize = 1_000;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct RetentionSourcePosition {
    sequence: u64,
    token: [u8; 32],
}

impl RetentionSourcePosition {
    pub fn new(sequence: u64, token: [u8; 32]) -> Result<Self, RetentionWorkerError> {
        if sequence == 0 || token == [0; 32] {
            return Err(RetentionWorkerError::InvalidInput);
        }
        Ok(Self { sequence, token })
    }

    pub const fn initial() -> Self {
        Self {
            sequence: 0,
            token: [0; 32],
        }
    }

    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    pub const fn token(self) -> [u8; 32] {
        self.token
    }
}

impl fmt::Debug for RetentionSourcePosition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetentionSourcePosition")
            .field("sequence", &self.sequence)
            .field("token", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionAuthoritySnapshot {
    pub policy: RetentionSnapshot<CommunityRetentionPolicy>,
    pub legal_hold: RetentionSnapshot<LegalHoldSnapshot>,
    pub community_archive: RetentionSnapshot<RetentionArchiveSnapshot>,
}

#[derive(Clone, Eq, PartialEq)]
pub struct RetentionWorkItem {
    position: RetentionSourcePosition,
    record: RetentionRecord,
    authority: RetentionAuthoritySnapshot,
}

impl RetentionWorkItem {
    pub const fn new(
        position: RetentionSourcePosition,
        record: RetentionRecord,
        authority: RetentionAuthoritySnapshot,
    ) -> Self {
        Self {
            position,
            record,
            authority,
        }
    }

    pub const fn position(&self) -> RetentionSourcePosition {
        self.position
    }

    pub const fn record(&self) -> RetentionRecord {
        self.record
    }

    pub const fn authority(&self) -> &RetentionAuthoritySnapshot {
        &self.authority
    }
}

impl fmt::Debug for RetentionWorkItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetentionWorkItem")
            .field("position", &self.position)
            .field("community_id", &self.record.community_id)
            .field("record_id", &"<redacted>")
            .field("event_kind", &self.record.event_kind)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetentionWorkerCounts {
    pub scanned: u64,
    pub retained_live: u64,
    pub retained_archive_only: u64,
    pub deleted: u64,
}

impl RetentionWorkerCounts {
    fn validate(self) -> Result<Self, RetentionWorkerError> {
        let accounted = self
            .retained_live
            .checked_add(self.retained_archive_only)
            .and_then(|count| count.checked_add(self.deleted))
            .ok_or(RetentionWorkerError::InvalidInput)?;
        if accounted != self.scanned {
            return Err(RetentionWorkerError::InvalidInput);
        }
        Ok(self)
    }

    fn add(self, batch: Self) -> Result<Self, RetentionWorkerError> {
        Self {
            scanned: self
                .scanned
                .checked_add(batch.scanned)
                .ok_or(RetentionWorkerError::CountExhausted)?,
            retained_live: self
                .retained_live
                .checked_add(batch.retained_live)
                .ok_or(RetentionWorkerError::CountExhausted)?,
            retained_archive_only: self
                .retained_archive_only
                .checked_add(batch.retained_archive_only)
                .ok_or(RetentionWorkerError::CountExhausted)?,
            deleted: self
                .deleted
                .checked_add(batch.deleted)
                .ok_or(RetentionWorkerError::CountExhausted)?,
        }
        .validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionWorkerCheckpointFields {
    pub community_id: collaboration_domain::CommunityId,
    pub checkpoint_version: u64,
    pub sweep_generation: u64,
    pub completed_sweeps: u64,
    pub cursor: RetentionSourcePosition,
    pub counts: RetentionWorkerCounts,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionWorkerCheckpoint {
    community_id: collaboration_domain::CommunityId,
    checkpoint_version: u64,
    sweep_generation: u64,
    completed_sweeps: u64,
    cursor: RetentionSourcePosition,
    counts: RetentionWorkerCounts,
}

impl RetentionWorkerCheckpoint {
    pub fn from_record(
        fields: RetentionWorkerCheckpointFields,
    ) -> Result<Self, RetentionWorkerError> {
        let checkpoint = Self {
            community_id: fields.community_id,
            checkpoint_version: fields.checkpoint_version,
            sweep_generation: fields.sweep_generation,
            completed_sweeps: fields.completed_sweeps,
            cursor: fields.cursor,
            counts: fields.counts,
        };
        checkpoint.validate(fields.community_id)?;
        Ok(checkpoint)
    }

    pub const fn initial(community_id: collaboration_domain::CommunityId) -> Self {
        Self {
            community_id,
            checkpoint_version: 0,
            sweep_generation: 1,
            completed_sweeps: 0,
            cursor: RetentionSourcePosition::initial(),
            counts: RetentionWorkerCounts {
                scanned: 0,
                retained_live: 0,
                retained_archive_only: 0,
                deleted: 0,
            },
        }
    }

    pub const fn community_id(&self) -> collaboration_domain::CommunityId {
        self.community_id
    }

    pub const fn checkpoint_version(&self) -> u64 {
        self.checkpoint_version
    }

    pub const fn sweep_generation(&self) -> u64 {
        self.sweep_generation
    }

    pub const fn completed_sweeps(&self) -> u64 {
        self.completed_sweeps
    }

    pub const fn cursor(&self) -> RetentionSourcePosition {
        self.cursor
    }

    pub const fn counts(&self) -> RetentionWorkerCounts {
        self.counts
    }

    fn validate(
        &self,
        expected_community_id: collaboration_domain::CommunityId,
    ) -> Result<(), RetentionWorkerError> {
        let expected_sweep_generation = self
            .completed_sweeps
            .checked_add(1)
            .ok_or(RetentionWorkerError::InvalidCheckpoint)?;
        if self.community_id != expected_community_id
            || self.community_id.as_uuid().is_nil()
            || self.sweep_generation == 0
            || self.sweep_generation != expected_sweep_generation
            || self.completed_sweeps > self.checkpoint_version
            || (self.checkpoint_version == 0
                && (self.sweep_generation != 1
                    || self.completed_sweeps != 0
                    || self.cursor != RetentionSourcePosition::initial()
                    || self.counts != RetentionWorkerCounts::default()))
            || (self.cursor.sequence == 0) != (self.cursor.token == [0; 32])
            || (self.cursor.sequence > 0 && self.counts.scanned == 0)
        {
            return Err(RetentionWorkerError::InvalidCheckpoint);
        }
        if self.counts.validate().is_err() {
            return Err(RetentionWorkerError::InvalidCheckpoint);
        }
        Ok(())
    }

    fn advance(
        &self,
        position: RetentionSourcePosition,
        batch_counts: RetentionWorkerCounts,
        completed_sweep: bool,
    ) -> Result<Self, RetentionWorkerError> {
        if position.sequence <= self.cursor.sequence {
            return Err(RetentionWorkerError::InvalidBatch);
        }
        let checkpoint_version = self
            .checkpoint_version
            .checked_add(1)
            .ok_or(RetentionWorkerError::VersionExhausted)?;
        let (sweep_generation, completed_sweeps, cursor) = if completed_sweep {
            (
                self.sweep_generation
                    .checked_add(1)
                    .ok_or(RetentionWorkerError::VersionExhausted)?,
                self.completed_sweeps
                    .checked_add(1)
                    .ok_or(RetentionWorkerError::CountExhausted)?,
                RetentionSourcePosition::initial(),
            )
        } else {
            (self.sweep_generation, self.completed_sweeps, position)
        };
        Ok(Self {
            community_id: self.community_id,
            checkpoint_version,
            sweep_generation,
            completed_sweeps,
            cursor,
            counts: self.counts.add(batch_counts)?,
        })
    }

    fn complete_empty_sweep(&self) -> Result<Self, RetentionWorkerError> {
        Ok(Self {
            community_id: self.community_id,
            checkpoint_version: self
                .checkpoint_version
                .checked_add(1)
                .ok_or(RetentionWorkerError::VersionExhausted)?,
            sweep_generation: self
                .sweep_generation
                .checked_add(1)
                .ok_or(RetentionWorkerError::VersionExhausted)?,
            completed_sweeps: self
                .completed_sweeps
                .checked_add(1)
                .ok_or(RetentionWorkerError::CountExhausted)?,
            cursor: RetentionSourcePosition::initial(),
            counts: self.counts,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionDeleteCause {
    Ephemeral,
    Policy(RetentionReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionAuthorityAction {
    SetVisibility(RetentionVisibility),
    Delete(RetentionDeleteCause),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionEvaluation {
    position: RetentionSourcePosition,
    decision: RetentionDecision,
    action: RetentionAuthorityAction,
}

impl RetentionEvaluation {
    pub const fn position(&self) -> RetentionSourcePosition {
        self.position
    }

    pub const fn decision(&self) -> &RetentionDecision {
        &self.decision
    }

    pub const fn action(&self) -> RetentionAuthorityAction {
        self.action
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionBatchCommit {
    expected_checkpoint: RetentionWorkerCheckpoint,
    next_checkpoint: RetentionWorkerCheckpoint,
    evaluations: Vec<RetentionEvaluation>,
}

impl RetentionBatchCommit {
    pub const fn expected_checkpoint(&self) -> &RetentionWorkerCheckpoint {
        &self.expected_checkpoint
    }

    pub const fn next_checkpoint(&self) -> &RetentionWorkerCheckpoint {
        &self.next_checkpoint
    }

    pub fn evaluations(&self) -> &[RetentionEvaluation] {
        &self.evaluations
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionCommitOutcome {
    Committed,
    AlreadyCommitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionBackendError {
    Unavailable,
    StaleCheckpoint,
    InvalidData,
    OutcomeUnknown,
}

impl fmt::Display for RetentionBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Unavailable => "retention backend is unavailable",
            Self::StaleCheckpoint => "retention backend checkpoint is stale",
            Self::InvalidData => "retention backend data is invalid",
            Self::OutcomeUnknown => "retention backend outcome is unknown",
        };
        formatter.write_str(message)
    }
}

impl Error for RetentionBackendError {}

#[async_trait]
pub trait RetentionAuthorityBackend: Send + Sync {
    async fn load_checkpoint(
        &self,
        tenant: &TenantContext,
    ) -> Result<Option<RetentionWorkerCheckpoint>, RetentionBackendError>;

    async fn load_batch(
        &self,
        tenant: &TenantContext,
        checkpoint: &RetentionWorkerCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionWorkItem>, RetentionBackendError>;

    /// Applies every authority action and advances the checkpoint in one transaction.
    /// An unobservable atomic result must return `OutcomeUnknown`, never expose a partial commit.
    async fn commit_batch(
        &self,
        tenant: &TenantContext,
        commit: &RetentionBatchCommit,
    ) -> Result<RetentionCommitOutcome, RetentionBackendError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionBatchHalt {
    pub position: RetentionSourcePosition,
    pub reason: RetentionError,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionBatchOutcome {
    checkpoint: RetentionWorkerCheckpoint,
    batch_counts: RetentionWorkerCounts,
    completed_sweep: bool,
    halt: Option<RetentionBatchHalt>,
    commit_outcome: Option<RetentionCommitOutcome>,
}

impl RetentionBatchOutcome {
    pub const fn checkpoint(&self) -> &RetentionWorkerCheckpoint {
        &self.checkpoint
    }

    pub const fn batch_counts(&self) -> RetentionWorkerCounts {
        self.batch_counts
    }

    pub const fn completed_sweep(&self) -> bool {
        self.completed_sweep
    }

    pub const fn halt(&self) -> Option<RetentionBatchHalt> {
        self.halt
    }

    pub const fn commit_outcome(&self) -> Option<RetentionCommitOutcome> {
        self.commit_outcome
    }
}

#[derive(Debug)]
pub enum RetentionWorkerError {
    InvalidInput,
    InvalidCheckpoint,
    InvalidBatch,
    CountExhausted,
    VersionExhausted,
    Backend(RetentionBackendError),
}

impl fmt::Display for RetentionWorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidInput => "retention worker input is invalid",
            Self::InvalidCheckpoint => "retention worker checkpoint is invalid",
            Self::InvalidBatch => "retention worker batch is invalid",
            Self::CountExhausted => "retention worker count is exhausted",
            Self::VersionExhausted => "retention worker version is exhausted",
            Self::Backend(error) => return error.fmt(formatter),
        };
        formatter.write_str(message)
    }
}

impl Error for RetentionWorkerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RetentionBackendError> for RetentionWorkerError {
    fn from(error: RetentionBackendError) -> Self {
        Self::Backend(error)
    }
}

pub struct RetentionWorker<Backend> {
    backend: Backend,
}

impl<Backend> RetentionWorker<Backend>
where
    Backend: RetentionAuthorityBackend,
{
    pub const fn new(backend: Backend) -> Self {
        Self { backend }
    }

    pub fn into_backend(self) -> Backend {
        self.backend
    }

    pub async fn run_batch(
        &self,
        tenant: &TenantContext,
        now_millis: u64,
        limit: usize,
    ) -> Result<RetentionBatchOutcome, RetentionWorkerError> {
        if limit == 0 || limit > MAX_RETENTION_BATCH_SIZE {
            return Err(RetentionWorkerError::InvalidInput);
        }
        let checkpoint = self
            .backend
            .load_checkpoint(tenant)
            .await?
            .unwrap_or_else(|| RetentionWorkerCheckpoint::initial(tenant.community_id()));
        checkpoint.validate(tenant.community_id())?;
        let items = self.backend.load_batch(tenant, &checkpoint, limit).await?;
        validate_batch(tenant, &checkpoint, &items, limit)?;

        let mut evaluations = Vec::with_capacity(items.len());
        let mut batch_counts = RetentionWorkerCounts::default();
        let mut halt = None;
        for item in &items {
            match evaluate_item(item, now_millis) {
                Ok(evaluation) => {
                    add_action_count(&mut batch_counts, evaluation.action)?;
                    evaluations.push(evaluation);
                }
                Err(reason) => {
                    halt = Some(RetentionBatchHalt {
                        position: item.position,
                        reason,
                    });
                    break;
                }
            }
        }

        let completed_sweep = halt.is_none() && items.len() < limit;
        if evaluations.is_empty() && !completed_sweep {
            return Ok(RetentionBatchOutcome {
                checkpoint,
                batch_counts,
                completed_sweep: false,
                halt,
                commit_outcome: None,
            });
        }

        let next_checkpoint = if let Some(last) = evaluations.last() {
            checkpoint.advance(last.position, batch_counts, completed_sweep)?
        } else {
            checkpoint.complete_empty_sweep()?
        };
        let commit = RetentionBatchCommit {
            expected_checkpoint: checkpoint,
            next_checkpoint: next_checkpoint.clone(),
            evaluations,
        };
        validate_commit(tenant, &commit)?;
        let commit_outcome = self.backend.commit_batch(tenant, &commit).await?;
        Ok(RetentionBatchOutcome {
            checkpoint: next_checkpoint,
            batch_counts,
            completed_sweep,
            halt,
            commit_outcome: Some(commit_outcome),
        })
    }
}

fn validate_batch(
    tenant: &TenantContext,
    checkpoint: &RetentionWorkerCheckpoint,
    items: &[RetentionWorkItem],
    limit: usize,
) -> Result<(), RetentionWorkerError> {
    if items.len() > limit {
        return Err(RetentionWorkerError::InvalidBatch);
    }
    let mut previous_position = checkpoint.cursor;
    for item in items {
        if item.record.community_id != tenant.community_id()
            || item.record.record_id.as_uuid().is_nil()
            || item.position.sequence <= previous_position.sequence
            || item.position.token == previous_position.token
        {
            return Err(RetentionWorkerError::InvalidBatch);
        }
        previous_position = item.position;
    }
    Ok(())
}

fn evaluate_item(
    item: &RetentionWorkItem,
    now_millis: u64,
) -> Result<RetentionEvaluation, RetentionError> {
    let policy = match &item.authority.policy {
        RetentionSnapshot::Absent => RetentionSnapshot::Absent,
        RetentionSnapshot::Current(policy) => RetentionSnapshot::Current(policy),
        RetentionSnapshot::Unavailable => RetentionSnapshot::Unavailable,
        RetentionSnapshot::Ambiguous => RetentionSnapshot::Ambiguous,
    };
    let resolution = resolve_retention(&RetentionRequest {
        record: item.record,
        policy,
        legal_hold: item.authority.legal_hold,
        community_archive: item.authority.community_archive,
        now_millis,
        previous_decision: None,
    })?;
    let decision = resolution.decision().clone();
    let action = match decision.disposition() {
        RetentionDisposition::DoNotPersist => {
            RetentionAuthorityAction::Delete(RetentionDeleteCause::Ephemeral)
        }
        RetentionDisposition::Retain { visibility, .. }
        | RetentionDisposition::DeleteAt { visibility, .. } => {
            RetentionAuthorityAction::SetVisibility(visibility)
        }
        RetentionDisposition::DeleteNow { reason } => {
            RetentionAuthorityAction::Delete(RetentionDeleteCause::Policy(reason))
        }
    };
    Ok(RetentionEvaluation {
        position: item.position,
        decision,
        action,
    })
}

fn add_action_count(
    counts: &mut RetentionWorkerCounts,
    action: RetentionAuthorityAction,
) -> Result<(), RetentionWorkerError> {
    counts.scanned = counts
        .scanned
        .checked_add(1)
        .ok_or(RetentionWorkerError::CountExhausted)?;
    let counter = match action {
        RetentionAuthorityAction::SetVisibility(RetentionVisibility::Live) => {
            &mut counts.retained_live
        }
        RetentionAuthorityAction::SetVisibility(RetentionVisibility::ArchiveOnly) => {
            &mut counts.retained_archive_only
        }
        RetentionAuthorityAction::Delete(_) => &mut counts.deleted,
    };
    *counter = counter
        .checked_add(1)
        .ok_or(RetentionWorkerError::CountExhausted)?;
    Ok(())
}

fn validate_commit(
    tenant: &TenantContext,
    commit: &RetentionBatchCommit,
) -> Result<(), RetentionWorkerError> {
    commit.expected_checkpoint.validate(tenant.community_id())?;
    commit.next_checkpoint.validate(tenant.community_id())?;
    if commit.next_checkpoint.checkpoint_version
        != commit
            .expected_checkpoint
            .checkpoint_version
            .checked_add(1)
            .ok_or(RetentionWorkerError::VersionExhausted)?
        || commit
            .evaluations
            .windows(2)
            .any(|pair| pair[0].position.sequence >= pair[1].position.sequence)
    {
        return Err(RetentionWorkerError::InvalidCheckpoint);
    }
    Ok(())
}
