use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{CommunityId, TenantContext};

use crate::{
    push::outbox::{PushOutboxError, PushOutboxRepository, PushRetentionCancellationOutcome},
    retention::worker::{MAX_RETENTION_BATCH_SIZE, RetentionSourcePosition},
};

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct RetentionDerivedSourceId([u8; 32]);

impl RetentionDerivedSourceId {
    pub fn new(value: [u8; 32]) -> Result<Self, RetentionDerivedCleanupError> {
        if value == [0; 32] {
            return Err(RetentionDerivedCleanupError::InvalidInput);
        }
        Ok(Self(value))
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for RetentionDerivedSourceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RetentionDerivedSourceId([REDACTED])")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionFinalVisibility {
    ArchiveOnly,
    Deleted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionDerivedCleanupItem {
    community_id: CommunityId,
    source_position: RetentionSourcePosition,
    source_id: RetentionDerivedSourceId,
    final_visibility: RetentionFinalVisibility,
    decided_at_millis: u64,
}

impl RetentionDerivedCleanupItem {
    pub fn new(
        community_id: CommunityId,
        source_position: RetentionSourcePosition,
        source_id: RetentionDerivedSourceId,
        final_visibility: RetentionFinalVisibility,
        decided_at_millis: u64,
    ) -> Result<Self, RetentionDerivedCleanupError> {
        if community_id.as_uuid().is_nil() || decided_at_millis == 0 {
            return Err(RetentionDerivedCleanupError::InvalidInput);
        }
        Ok(Self {
            community_id,
            source_position,
            source_id,
            final_visibility,
            decided_at_millis,
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn source_position(self) -> RetentionSourcePosition {
        self.source_position
    }

    pub const fn source_id(self) -> RetentionDerivedSourceId {
        self.source_id
    }

    pub const fn final_visibility(self) -> RetentionFinalVisibility {
        self.final_visibility
    }

    pub const fn decided_at_millis(self) -> u64 {
        self.decided_at_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionDerivedCheckpointFields {
    pub community_id: CommunityId,
    pub checkpoint_version: u64,
    pub source_position: RetentionSourcePosition,
    pub converged: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionDerivedCheckpoint {
    community_id: CommunityId,
    checkpoint_version: u64,
    source_position: RetentionSourcePosition,
    converged: u64,
}

impl RetentionDerivedCheckpoint {
    pub fn from_record(
        fields: RetentionDerivedCheckpointFields,
    ) -> Result<Self, RetentionDerivedCleanupError> {
        let checkpoint = Self {
            community_id: fields.community_id,
            checkpoint_version: fields.checkpoint_version,
            source_position: fields.source_position,
            converged: fields.converged,
        };
        checkpoint.validate(fields.community_id)?;
        Ok(checkpoint)
    }

    pub const fn initial(community_id: CommunityId) -> Self {
        Self {
            community_id,
            checkpoint_version: 0,
            source_position: RetentionSourcePosition::initial(),
            converged: 0,
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn checkpoint_version(self) -> u64 {
        self.checkpoint_version
    }

    pub const fn source_position(self) -> RetentionSourcePosition {
        self.source_position
    }

    pub const fn converged(self) -> u64 {
        self.converged
    }

    fn validate(self, community_id: CommunityId) -> Result<(), RetentionDerivedCleanupError> {
        if self.community_id != community_id
            || self.community_id.as_uuid().is_nil()
            || self.checkpoint_version != self.converged
            || (self.checkpoint_version == 0)
                != (self.source_position == RetentionSourcePosition::initial())
        {
            return Err(RetentionDerivedCleanupError::InvalidCheckpoint);
        }
        Ok(())
    }

    fn advance(
        self,
        item: RetentionDerivedCleanupItem,
    ) -> Result<Self, RetentionDerivedCleanupError> {
        if item.community_id != self.community_id
            || item.source_position.sequence() <= self.source_position.sequence()
        {
            return Err(RetentionDerivedCleanupError::InvalidBatch);
        }
        Ok(Self {
            community_id: self.community_id,
            checkpoint_version: self
                .checkpoint_version
                .checked_add(1)
                .ok_or(RetentionDerivedCleanupError::VersionExhausted)?,
            source_position: item.source_position,
            converged: self
                .converged
                .checked_add(1)
                .ok_or(RetentionDerivedCleanupError::CountExhausted)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionDerivedMutationOutcome {
    Cleared,
    AlreadyClear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionDerivedTargetError {
    Unavailable,
    InvalidData,
}

impl fmt::Display for RetentionDerivedTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "retention derived-state target is unavailable",
            Self::InvalidData => "retention derived-state target data is invalid",
        })
    }
}

impl Error for RetentionDerivedTargetError {}

#[async_trait]
pub trait RetentionCacheInvalidator: Send + Sync {
    async fn invalidate_cache_and_presence(
        &self,
        tenant: &TenantContext,
        item: RetentionDerivedCleanupItem,
    ) -> Result<RetentionDerivedMutationOutcome, RetentionDerivedTargetError>;
}

#[async_trait]
pub trait RetentionPushQueue: Send + Sync {
    async fn cancel_obsolete_wakes(
        &self,
        tenant: &TenantContext,
        item: RetentionDerivedCleanupItem,
    ) -> Result<RetentionDerivedMutationOutcome, RetentionDerivedTargetError>;
}

#[async_trait]
impl RetentionPushQueue for PushOutboxRepository {
    async fn cancel_obsolete_wakes(
        &self,
        tenant: &TenantContext,
        item: RetentionDerivedCleanupItem,
    ) -> Result<RetentionDerivedMutationOutcome, RetentionDerivedTargetError> {
        match self
            .cancel_source_wakes_after_retention(tenant, item.source_id.as_bytes())
            .await
        {
            Ok(PushRetentionCancellationOutcome::Cancelled(_)) => {
                Ok(RetentionDerivedMutationOutcome::Cleared)
            }
            Ok(PushRetentionCancellationOutcome::AlreadyClear) => {
                Ok(RetentionDerivedMutationOutcome::AlreadyClear)
            }
            Err(PushOutboxError::Unavailable(_)) => Err(RetentionDerivedTargetError::Unavailable),
            Err(_) => Err(RetentionDerivedTargetError::InvalidData),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionDerivedCheckpointCommitOutcome {
    Committed,
    AlreadyCommitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionDerivedBackendError {
    Unavailable,
    StaleCheckpoint,
    InvalidData,
    OutcomeUnknown,
}

impl fmt::Display for RetentionDerivedBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "retention derived-state backend is unavailable",
            Self::StaleCheckpoint => "retention derived-state checkpoint is stale",
            Self::InvalidData => "retention derived-state backend data is invalid",
            Self::OutcomeUnknown => "retention derived-state checkpoint outcome is unknown",
        })
    }
}

impl Error for RetentionDerivedBackendError {}

#[async_trait]
pub trait RetentionDerivedBackend: Send + Sync {
    async fn load_checkpoint(
        &self,
        tenant: &TenantContext,
    ) -> Result<Option<RetentionDerivedCheckpoint>, RetentionDerivedBackendError>;

    async fn load_batch(
        &self,
        tenant: &TenantContext,
        checkpoint: RetentionDerivedCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionDerivedCleanupItem>, RetentionDerivedBackendError>;

    async fn advance_checkpoint(
        &self,
        tenant: &TenantContext,
        expected: RetentionDerivedCheckpoint,
        next: RetentionDerivedCheckpoint,
    ) -> Result<RetentionDerivedCheckpointCommitOutcome, RetentionDerivedBackendError>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetentionDerivedCleanupCounts {
    pub cache_cleared: u64,
    pub cache_already_clear: u64,
    pub push_cleared: u64,
    pub push_already_clear: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionDerivedCleanupOutcome {
    checkpoint: RetentionDerivedCheckpoint,
    counts: RetentionDerivedCleanupCounts,
    completed_batch: bool,
    final_visibility: Option<RetentionFinalVisibility>,
}

impl RetentionDerivedCleanupOutcome {
    pub const fn checkpoint(self) -> RetentionDerivedCheckpoint {
        self.checkpoint
    }

    pub const fn counts(self) -> RetentionDerivedCleanupCounts {
        self.counts
    }

    pub const fn completed_batch(self) -> bool {
        self.completed_batch
    }

    pub const fn final_visibility(self) -> Option<RetentionFinalVisibility> {
        self.final_visibility
    }
}

#[derive(Debug)]
pub enum RetentionDerivedCleanupError {
    InvalidInput,
    InvalidCheckpoint,
    InvalidBatch,
    CountExhausted,
    VersionExhausted,
    Cache(RetentionDerivedTargetError),
    Push(RetentionDerivedTargetError),
    Backend(RetentionDerivedBackendError),
}

impl fmt::Display for RetentionDerivedCleanupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidInput => "retention derived-state cleanup input is invalid",
            Self::InvalidCheckpoint => "retention derived-state checkpoint is invalid",
            Self::InvalidBatch => "retention derived-state cleanup batch is invalid",
            Self::CountExhausted => "retention derived-state cleanup count is exhausted",
            Self::VersionExhausted => "retention derived-state cleanup version is exhausted",
            Self::Cache(error) | Self::Push(error) => return error.fmt(formatter),
            Self::Backend(error) => return error.fmt(formatter),
        };
        formatter.write_str(message)
    }
}

impl Error for RetentionDerivedCleanupError {}

impl From<RetentionDerivedBackendError> for RetentionDerivedCleanupError {
    fn from(error: RetentionDerivedBackendError) -> Self {
        Self::Backend(error)
    }
}

pub struct RetentionDerivedCleanup<Backend, Cache, Push> {
    backend: Backend,
    cache: Cache,
    push: Push,
}

impl<Backend, Cache, Push> RetentionDerivedCleanup<Backend, Cache, Push>
where
    Backend: RetentionDerivedBackend,
    Cache: RetentionCacheInvalidator,
    Push: RetentionPushQueue,
{
    pub const fn new(backend: Backend, cache: Cache, push: Push) -> Self {
        Self {
            backend,
            cache,
            push,
        }
    }

    pub async fn run_batch(
        &self,
        tenant: &TenantContext,
        limit: usize,
    ) -> Result<RetentionDerivedCleanupOutcome, RetentionDerivedCleanupError> {
        if limit == 0 || limit > MAX_RETENTION_BATCH_SIZE {
            return Err(RetentionDerivedCleanupError::InvalidInput);
        }
        let mut checkpoint = self
            .backend
            .load_checkpoint(tenant)
            .await?
            .unwrap_or_else(|| RetentionDerivedCheckpoint::initial(tenant.community_id()));
        checkpoint.validate(tenant.community_id())?;
        let items = self.backend.load_batch(tenant, checkpoint, limit).await?;
        validate_batch(tenant, checkpoint, &items, limit)?;

        let mut counts = RetentionDerivedCleanupCounts::default();
        let mut final_visibility = None;
        for item in items.iter().copied() {
            let cache = self
                .cache
                .invalidate_cache_and_presence(tenant, item)
                .await
                .map_err(RetentionDerivedCleanupError::Cache)?;
            add_count(&mut counts, cache, true)?;
            let push = self
                .push
                .cancel_obsolete_wakes(tenant, item)
                .await
                .map_err(RetentionDerivedCleanupError::Push)?;
            add_count(&mut counts, push, false)?;
            let next = checkpoint.advance(item)?;
            self.backend
                .advance_checkpoint(tenant, checkpoint, next)
                .await?;
            checkpoint = next;
            final_visibility = Some(item.final_visibility);
        }
        Ok(RetentionDerivedCleanupOutcome {
            checkpoint,
            counts,
            completed_batch: items.len() < limit,
            final_visibility,
        })
    }
}

fn add_count(
    counts: &mut RetentionDerivedCleanupCounts,
    outcome: RetentionDerivedMutationOutcome,
    cache: bool,
) -> Result<(), RetentionDerivedCleanupError> {
    let counter = match (cache, outcome) {
        (true, RetentionDerivedMutationOutcome::Cleared) => &mut counts.cache_cleared,
        (true, RetentionDerivedMutationOutcome::AlreadyClear) => &mut counts.cache_already_clear,
        (false, RetentionDerivedMutationOutcome::Cleared) => &mut counts.push_cleared,
        (false, RetentionDerivedMutationOutcome::AlreadyClear) => &mut counts.push_already_clear,
    };
    *counter = counter
        .checked_add(1)
        .ok_or(RetentionDerivedCleanupError::CountExhausted)?;
    Ok(())
}

fn validate_batch(
    tenant: &TenantContext,
    checkpoint: RetentionDerivedCheckpoint,
    items: &[RetentionDerivedCleanupItem],
    limit: usize,
) -> Result<(), RetentionDerivedCleanupError> {
    if items.len() > limit {
        return Err(RetentionDerivedCleanupError::InvalidBatch);
    }
    let mut previous = checkpoint.source_position;
    for item in items {
        if item.community_id != tenant.community_id()
            || item.source_position.sequence() <= previous.sequence()
        {
            return Err(RetentionDerivedCleanupError::InvalidBatch);
        }
        previous = item.source_position;
    }
    Ok(())
}
