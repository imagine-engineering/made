use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{CommunityId, TenantContext};

use crate::{
    retention::worker::{MAX_RETENTION_BATCH_SIZE, RetentionSourcePosition},
    search::indexer::{
        CollaborationSearchIndexer, SearchExclusionReason, SearchIndexerError,
        SearchIndexingOutcome,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionSearchDelivery {
    community_id: CommunityId,
    source_position: RetentionSourcePosition,
    outbox_sequence: u64,
}

impl RetentionSearchDelivery {
    pub fn new(
        community_id: CommunityId,
        source_position: RetentionSourcePosition,
        outbox_sequence: u64,
    ) -> Result<Self, RetentionSearchCleanupError> {
        if community_id.as_uuid().is_nil() || outbox_sequence == 0 {
            return Err(RetentionSearchCleanupError::InvalidInput);
        }
        Ok(Self {
            community_id,
            source_position,
            outbox_sequence,
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn source_position(self) -> RetentionSourcePosition {
        self.source_position
    }

    pub const fn outbox_sequence(self) -> u64 {
        self.outbox_sequence
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionSearchCheckpointFields {
    pub community_id: CommunityId,
    pub checkpoint_version: u64,
    pub source_position: RetentionSourcePosition,
    pub outbox_sequence: u64,
    pub converged: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionSearchCheckpoint {
    community_id: CommunityId,
    checkpoint_version: u64,
    source_position: RetentionSourcePosition,
    outbox_sequence: u64,
    converged: u64,
}

impl RetentionSearchCheckpoint {
    pub fn from_record(
        fields: RetentionSearchCheckpointFields,
    ) -> Result<Self, RetentionSearchCleanupError> {
        let checkpoint = Self {
            community_id: fields.community_id,
            checkpoint_version: fields.checkpoint_version,
            source_position: fields.source_position,
            outbox_sequence: fields.outbox_sequence,
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
            outbox_sequence: 0,
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

    pub const fn outbox_sequence(self) -> u64 {
        self.outbox_sequence
    }

    pub const fn converged(self) -> u64 {
        self.converged
    }

    fn validate(
        self,
        expected_community_id: CommunityId,
    ) -> Result<(), RetentionSearchCleanupError> {
        if self.community_id != expected_community_id
            || self.community_id.as_uuid().is_nil()
            || (self.source_position.sequence() == 0) != (self.outbox_sequence == 0)
            || (self.checkpoint_version == 0
                && (self.source_position != RetentionSourcePosition::initial()
                    || self.outbox_sequence != 0
                    || self.converged != 0))
            || (self.checkpoint_version > 0
                && (self.source_position == RetentionSourcePosition::initial()
                    || self.outbox_sequence == 0
                    || self.converged == 0))
            || self.converged != self.checkpoint_version
        {
            return Err(RetentionSearchCleanupError::InvalidCheckpoint);
        }
        Ok(())
    }

    fn advance(
        self,
        delivery: RetentionSearchDelivery,
    ) -> Result<Self, RetentionSearchCleanupError> {
        if delivery.community_id != self.community_id
            || delivery.source_position.sequence() <= self.source_position.sequence()
            || delivery.outbox_sequence <= self.outbox_sequence
        {
            return Err(RetentionSearchCleanupError::InvalidBatch);
        }
        Ok(Self {
            community_id: self.community_id,
            checkpoint_version: self
                .checkpoint_version
                .checked_add(1)
                .ok_or(RetentionSearchCleanupError::VersionExhausted)?,
            source_position: delivery.source_position,
            outbox_sequence: delivery.outbox_sequence,
            converged: self
                .converged
                .checked_add(1)
                .ok_or(RetentionSearchCleanupError::CountExhausted)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionSearchProjectionOutcome {
    Excluded,
    AlreadyConverged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionSearchProjectionError {
    Unavailable,
    InvalidData,
}

impl fmt::Display for RetentionSearchProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "retention search projection is unavailable",
            Self::InvalidData => "retention search projection data is invalid",
        })
    }
}

impl Error for RetentionSearchProjectionError {}

#[async_trait]
pub trait RetentionSearchProjection: Send + Sync {
    async fn exclude_after_retention(
        &self,
        tenant: &TenantContext,
        outbox_sequence: u64,
    ) -> Result<RetentionSearchProjectionOutcome, RetentionSearchProjectionError>;
}

#[async_trait]
impl RetentionSearchProjection for CollaborationSearchIndexer {
    async fn exclude_after_retention(
        &self,
        tenant: &TenantContext,
        outbox_sequence: u64,
    ) -> Result<RetentionSearchProjectionOutcome, RetentionSearchProjectionError> {
        match self
            .index_retention_expiry_outbox_sequence(tenant, outbox_sequence)
            .await
        {
            Ok(SearchIndexingOutcome::Excluded(SearchExclusionReason::RetentionExpired)) => {
                Ok(RetentionSearchProjectionOutcome::Excluded)
            }
            Ok(SearchIndexingOutcome::IgnoredReplay) => {
                Ok(RetentionSearchProjectionOutcome::AlreadyConverged)
            }
            Ok(_) => Err(RetentionSearchProjectionError::InvalidData),
            Err(SearchIndexerError::Unavailable(_)) => {
                Err(RetentionSearchProjectionError::Unavailable)
            }
            Err(_) => Err(RetentionSearchProjectionError::InvalidData),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionSearchCheckpointCommitOutcome {
    Committed,
    AlreadyCommitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionSearchBackendError {
    Unavailable,
    StaleCheckpoint,
    InvalidData,
    OutcomeUnknown,
}

impl fmt::Display for RetentionSearchBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "retention search checkpoint backend is unavailable",
            Self::StaleCheckpoint => "retention search checkpoint is stale",
            Self::InvalidData => "retention search checkpoint data is invalid",
            Self::OutcomeUnknown => "retention search checkpoint outcome is unknown",
        })
    }
}

impl Error for RetentionSearchBackendError {}

#[async_trait]
pub trait RetentionSearchBackend: Send + Sync {
    async fn load_checkpoint(
        &self,
        tenant: &TenantContext,
    ) -> Result<Option<RetentionSearchCheckpoint>, RetentionSearchBackendError>;

    async fn load_batch(
        &self,
        tenant: &TenantContext,
        checkpoint: RetentionSearchCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionSearchDelivery>, RetentionSearchBackendError>;

    /// Persists the exact compare-and-set transition or reports `OutcomeUnknown` when its
    /// durable result cannot be observed; implementations must never expose a partial checkpoint.
    async fn advance_checkpoint(
        &self,
        tenant: &TenantContext,
        expected: RetentionSearchCheckpoint,
        next: RetentionSearchCheckpoint,
    ) -> Result<RetentionSearchCheckpointCommitOutcome, RetentionSearchBackendError>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetentionSearchCleanupCounts {
    pub excluded: u64,
    pub already_converged: u64,
}

impl RetentionSearchCleanupCounts {
    pub fn total(self) -> Result<u64, RetentionSearchCleanupError> {
        self.excluded
            .checked_add(self.already_converged)
            .ok_or(RetentionSearchCleanupError::CountExhausted)
    }

    fn add(
        &mut self,
        outcome: RetentionSearchProjectionOutcome,
    ) -> Result<(), RetentionSearchCleanupError> {
        let counter = match outcome {
            RetentionSearchProjectionOutcome::Excluded => &mut self.excluded,
            RetentionSearchProjectionOutcome::AlreadyConverged => &mut self.already_converged,
        };
        *counter = counter
            .checked_add(1)
            .ok_or(RetentionSearchCleanupError::CountExhausted)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionSearchCleanupOutcome {
    checkpoint: RetentionSearchCheckpoint,
    counts: RetentionSearchCleanupCounts,
    completed_batch: bool,
}

impl RetentionSearchCleanupOutcome {
    pub const fn checkpoint(self) -> RetentionSearchCheckpoint {
        self.checkpoint
    }

    pub const fn counts(self) -> RetentionSearchCleanupCounts {
        self.counts
    }

    pub const fn completed_batch(self) -> bool {
        self.completed_batch
    }
}

#[derive(Debug)]
pub enum RetentionSearchCleanupError {
    InvalidInput,
    InvalidCheckpoint,
    InvalidBatch,
    CountExhausted,
    VersionExhausted,
    Projection(RetentionSearchProjectionError),
    Backend(RetentionSearchBackendError),
}

impl fmt::Display for RetentionSearchCleanupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidInput => "retention search cleanup input is invalid",
            Self::InvalidCheckpoint => "retention search cleanup checkpoint is invalid",
            Self::InvalidBatch => "retention search cleanup batch is invalid",
            Self::CountExhausted => "retention search cleanup count is exhausted",
            Self::VersionExhausted => "retention search cleanup version is exhausted",
            Self::Projection(error) => return error.fmt(formatter),
            Self::Backend(error) => return error.fmt(formatter),
        };
        formatter.write_str(message)
    }
}

impl Error for RetentionSearchCleanupError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Projection(error) => Some(error),
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RetentionSearchProjectionError> for RetentionSearchCleanupError {
    fn from(error: RetentionSearchProjectionError) -> Self {
        Self::Projection(error)
    }
}

impl From<RetentionSearchBackendError> for RetentionSearchCleanupError {
    fn from(error: RetentionSearchBackendError) -> Self {
        Self::Backend(error)
    }
}

pub struct RetentionSearchCleanup<Backend, Projection> {
    backend: Backend,
    projection: Projection,
}

impl<Backend, Projection> RetentionSearchCleanup<Backend, Projection>
where
    Backend: RetentionSearchBackend,
    Projection: RetentionSearchProjection,
{
    pub const fn new(backend: Backend, projection: Projection) -> Self {
        Self {
            backend,
            projection,
        }
    }

    pub fn into_parts(self) -> (Backend, Projection) {
        (self.backend, self.projection)
    }

    pub async fn run_batch(
        &self,
        tenant: &TenantContext,
        limit: usize,
    ) -> Result<RetentionSearchCleanupOutcome, RetentionSearchCleanupError> {
        if limit == 0 || limit > MAX_RETENTION_BATCH_SIZE {
            return Err(RetentionSearchCleanupError::InvalidInput);
        }
        let mut checkpoint = self
            .backend
            .load_checkpoint(tenant)
            .await?
            .unwrap_or_else(|| RetentionSearchCheckpoint::initial(tenant.community_id()));
        checkpoint.validate(tenant.community_id())?;
        let deliveries = self.backend.load_batch(tenant, checkpoint, limit).await?;
        validate_batch(tenant, checkpoint, &deliveries, limit)?;

        let mut counts = RetentionSearchCleanupCounts::default();
        for delivery in deliveries.iter().copied() {
            let projection_outcome = self
                .projection
                .exclude_after_retention(tenant, delivery.outbox_sequence)
                .await?;
            let next = checkpoint.advance(delivery)?;
            self.backend
                .advance_checkpoint(tenant, checkpoint, next)
                .await?;
            checkpoint = next;
            counts.add(projection_outcome)?;
        }
        Ok(RetentionSearchCleanupOutcome {
            checkpoint,
            counts,
            completed_batch: deliveries.len() < limit,
        })
    }
}

fn validate_batch(
    tenant: &TenantContext,
    checkpoint: RetentionSearchCheckpoint,
    deliveries: &[RetentionSearchDelivery],
    limit: usize,
) -> Result<(), RetentionSearchCleanupError> {
    if deliveries.len() > limit {
        return Err(RetentionSearchCleanupError::InvalidBatch);
    }
    let mut previous_position = checkpoint.source_position;
    let mut previous_outbox_sequence = checkpoint.outbox_sequence;
    for delivery in deliveries {
        if delivery.community_id != tenant.community_id()
            || delivery.source_position.sequence() <= previous_position.sequence()
            || delivery.outbox_sequence <= previous_outbox_sequence
        {
            return Err(RetentionSearchCleanupError::InvalidBatch);
        }
        previous_position = delivery.source_position;
        previous_outbox_sequence = delivery.outbox_sequence;
    }
    Ok(())
}
