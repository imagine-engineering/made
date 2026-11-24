use std::{collections::BTreeSet, error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    CommunityId, MediaAttachmentLink, MediaContentHash, OperationId, TenantContext,
};

use crate::{
    media::object_store::{
        MediaObjectBackend, MediaObjectBackendError, MediaObjectDeleteOutcome,
        MediaOrphanDeletionLease, MediaOrphanFinalization,
    },
    retention::worker::{MAX_RETENTION_BATCH_SIZE, RetentionSourcePosition},
};

pub const MAX_RETENTION_MEDIA_ATTACHMENTS: usize = 32;
pub const MAX_RETENTION_MEDIA_OBJECTS: usize = MAX_RETENTION_MEDIA_ATTACHMENTS * 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionMediaCleanupItem {
    community_id: CommunityId,
    source_position: RetentionSourcePosition,
    operation_id: OperationId,
    decided_at_millis: u64,
    attachments: Vec<MediaAttachmentLink>,
}

impl RetentionMediaCleanupItem {
    pub fn new(
        community_id: CommunityId,
        source_position: RetentionSourcePosition,
        operation_id: OperationId,
        decided_at_millis: u64,
        attachments: Vec<MediaAttachmentLink>,
    ) -> Result<Self, RetentionMediaCleanupError> {
        if community_id.as_uuid().is_nil()
            || operation_id.as_uuid().is_nil()
            || decided_at_millis == 0
            || attachments.is_empty()
            || attachments.len() > MAX_RETENTION_MEDIA_ATTACHMENTS
        {
            return Err(RetentionMediaCleanupError::InvalidInput);
        }
        let first = attachments
            .first()
            .copied()
            .ok_or(RetentionMediaCleanupError::InvalidInput)?;
        let mut identities = BTreeSet::new();
        for attachment in &attachments {
            if attachment.media_identity().community_id() != community_id
                || attachment.channel_id() != first.channel_id()
                || attachment.message_id() != first.message_id()
                || !identities.insert(attachment.media_identity())
            {
                return Err(RetentionMediaCleanupError::InvalidInput);
            }
        }
        Ok(Self {
            community_id,
            source_position,
            operation_id,
            decided_at_millis,
            attachments,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn source_position(&self) -> RetentionSourcePosition {
        self.source_position
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn decided_at_millis(&self) -> u64 {
        self.decided_at_millis
    }

    pub fn attachments(&self) -> &[MediaAttachmentLink] {
        &self.attachments
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionMediaCheckpointFields {
    pub community_id: CommunityId,
    pub checkpoint_version: u64,
    pub source_position: RetentionSourcePosition,
    pub converged: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionMediaCheckpoint {
    community_id: CommunityId,
    checkpoint_version: u64,
    source_position: RetentionSourcePosition,
    converged: u64,
}

impl RetentionMediaCheckpoint {
    pub fn from_record(
        fields: RetentionMediaCheckpointFields,
    ) -> Result<Self, RetentionMediaCleanupError> {
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

    fn validate(self, community_id: CommunityId) -> Result<(), RetentionMediaCleanupError> {
        if self.community_id != community_id
            || self.community_id.as_uuid().is_nil()
            || self.checkpoint_version != self.converged
            || (self.checkpoint_version == 0)
                != (self.source_position == RetentionSourcePosition::initial())
        {
            return Err(RetentionMediaCleanupError::InvalidCheckpoint);
        }
        Ok(())
    }

    fn advance(self, item: &RetentionMediaCleanupItem) -> Result<Self, RetentionMediaCleanupError> {
        if item.community_id != self.community_id
            || item.source_position.sequence() <= self.source_position.sequence()
        {
            return Err(RetentionMediaCleanupError::InvalidBatch);
        }
        Ok(Self {
            community_id: self.community_id,
            checkpoint_version: self
                .checkpoint_version
                .checked_add(1)
                .ok_or(RetentionMediaCleanupError::VersionExhausted)?,
            source_position: item.source_position,
            converged: self
                .converged
                .checked_add(1)
                .ok_or(RetentionMediaCleanupError::CountExhausted)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionMediaReferenceOutcome {
    Detached,
    AlreadyDetached,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionMediaRemovalPlan {
    operation_id: OperationId,
    reference_outcome: RetentionMediaReferenceOutcome,
    orphan_leases: Vec<MediaOrphanDeletionLease>,
}

impl RetentionMediaRemovalPlan {
    pub fn new(
        operation_id: OperationId,
        reference_outcome: RetentionMediaReferenceOutcome,
        orphan_leases: Vec<MediaOrphanDeletionLease>,
    ) -> Result<Self, RetentionMediaCleanupError> {
        if operation_id.as_uuid().is_nil() || orphan_leases.len() > MAX_RETENTION_MEDIA_OBJECTS {
            return Err(RetentionMediaCleanupError::InvalidBackendData);
        }
        Ok(Self {
            operation_id,
            reference_outcome,
            orphan_leases,
        })
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn reference_outcome(&self) -> RetentionMediaReferenceOutcome {
        self.reference_outcome
    }

    pub fn orphan_leases(&self) -> &[MediaOrphanDeletionLease] {
        &self.orphan_leases
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionMediaCheckpointCommitOutcome {
    Committed,
    AlreadyCommitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionMediaBackendError {
    Unavailable,
    StaleCheckpoint,
    Conflict,
    InvalidData,
    OutcomeUnknown,
}

impl fmt::Display for RetentionMediaBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "retention media backend is unavailable",
            Self::StaleCheckpoint => "retention media checkpoint is stale",
            Self::Conflict => "retention media operation conflicts with durable state",
            Self::InvalidData => "retention media backend data is invalid",
            Self::OutcomeUnknown => "retention media backend outcome is unknown",
        })
    }
}

impl Error for RetentionMediaBackendError {}

#[async_trait]
pub trait RetentionMediaBackend: Send + Sync {
    async fn load_checkpoint(
        &self,
        tenant: &TenantContext,
    ) -> Result<Option<RetentionMediaCheckpoint>, RetentionMediaBackendError>;

    async fn load_batch(
        &self,
        tenant: &TenantContext,
        checkpoint: RetentionMediaCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionMediaCleanupItem>, RetentionMediaBackendError>;

    async fn detach_and_reserve_orphans(
        &self,
        tenant: &TenantContext,
        item: &RetentionMediaCleanupItem,
    ) -> Result<RetentionMediaRemovalPlan, RetentionMediaBackendError>;

    async fn finalize_orphan_deletion(
        &self,
        tenant: &TenantContext,
        item: &RetentionMediaCleanupItem,
        lease: &MediaOrphanDeletionLease,
        outcome: MediaOrphanFinalization,
    ) -> Result<(), RetentionMediaBackendError>;

    async fn advance_checkpoint(
        &self,
        tenant: &TenantContext,
        expected: RetentionMediaCheckpoint,
        next: RetentionMediaCheckpoint,
    ) -> Result<RetentionMediaCheckpointCommitOutcome, RetentionMediaBackendError>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetentionMediaCleanupCounts {
    pub detached_references: u64,
    pub already_detached_references: u64,
    pub deleted_objects: u64,
    pub already_missing_objects: u64,
    pub concurrently_changed_objects: u64,
}

impl RetentionMediaCleanupCounts {
    fn add_references(
        &mut self,
        outcome: RetentionMediaReferenceOutcome,
        count: usize,
    ) -> Result<(), RetentionMediaCleanupError> {
        let count = u64::try_from(count).map_err(|_| RetentionMediaCleanupError::CountExhausted)?;
        let target = match outcome {
            RetentionMediaReferenceOutcome::Detached => &mut self.detached_references,
            RetentionMediaReferenceOutcome::AlreadyDetached => {
                &mut self.already_detached_references
            }
        };
        *target = target
            .checked_add(count)
            .ok_or(RetentionMediaCleanupError::CountExhausted)?;
        Ok(())
    }

    fn add_object(
        &mut self,
        outcome: MediaObjectDeleteOutcome,
    ) -> Result<(), RetentionMediaCleanupError> {
        let target = match outcome {
            MediaObjectDeleteOutcome::Deleted => &mut self.deleted_objects,
            MediaObjectDeleteOutcome::AlreadyMissing => &mut self.already_missing_objects,
            MediaObjectDeleteOutcome::PreconditionFailed => &mut self.concurrently_changed_objects,
            MediaObjectDeleteOutcome::VersionArtifact => {
                return Err(RetentionMediaCleanupError::UnsafeObjectStore);
            }
        };
        *target = target
            .checked_add(1)
            .ok_or(RetentionMediaCleanupError::CountExhausted)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionMediaCleanupOutcome {
    checkpoint: RetentionMediaCheckpoint,
    counts: RetentionMediaCleanupCounts,
    completed_batch: bool,
}

impl RetentionMediaCleanupOutcome {
    pub const fn checkpoint(self) -> RetentionMediaCheckpoint {
        self.checkpoint
    }

    pub const fn counts(self) -> RetentionMediaCleanupCounts {
        self.counts
    }

    pub const fn completed_batch(self) -> bool {
        self.completed_batch
    }
}

#[derive(Debug)]
pub enum RetentionMediaCleanupError {
    InvalidInput,
    InvalidCheckpoint,
    InvalidBatch,
    InvalidBackendData,
    CountExhausted,
    VersionExhausted,
    UnsafeObjectStore,
    ObjectBackend(MediaObjectBackendError),
    Backend(RetentionMediaBackendError),
}

impl fmt::Display for RetentionMediaCleanupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidInput => "retention media cleanup input is invalid",
            Self::InvalidCheckpoint => "retention media cleanup checkpoint is invalid",
            Self::InvalidBatch => "retention media cleanup batch is invalid",
            Self::InvalidBackendData => "retention media cleanup backend data is invalid",
            Self::CountExhausted => "retention media cleanup count is exhausted",
            Self::VersionExhausted => "retention media cleanup version is exhausted",
            Self::UnsafeObjectStore => "retention media cleanup cannot safely continue",
            Self::ObjectBackend(error) => return error.fmt(formatter),
            Self::Backend(error) => return error.fmt(formatter),
        })
    }
}

impl Error for RetentionMediaCleanupError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ObjectBackend(error) => Some(error),
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MediaObjectBackendError> for RetentionMediaCleanupError {
    fn from(error: MediaObjectBackendError) -> Self {
        Self::ObjectBackend(error)
    }
}

impl From<RetentionMediaBackendError> for RetentionMediaCleanupError {
    fn from(error: RetentionMediaBackendError) -> Self {
        Self::Backend(error)
    }
}

pub struct RetentionMediaCleanup<Backend, ObjectBackend> {
    backend: Backend,
    object_backend: ObjectBackend,
}

impl<Backend, ObjectBackend> RetentionMediaCleanup<Backend, ObjectBackend>
where
    Backend: RetentionMediaBackend,
    ObjectBackend: MediaObjectBackend,
{
    pub const fn new(backend: Backend, object_backend: ObjectBackend) -> Self {
        Self {
            backend,
            object_backend,
        }
    }

    pub fn into_parts(self) -> (Backend, ObjectBackend) {
        (self.backend, self.object_backend)
    }

    pub async fn run_batch(
        &self,
        tenant: &TenantContext,
        limit: usize,
    ) -> Result<RetentionMediaCleanupOutcome, RetentionMediaCleanupError> {
        if limit == 0 || limit > MAX_RETENTION_BATCH_SIZE {
            return Err(RetentionMediaCleanupError::InvalidInput);
        }
        let mut checkpoint = self
            .backend
            .load_checkpoint(tenant)
            .await?
            .unwrap_or_else(|| RetentionMediaCheckpoint::initial(tenant.community_id()));
        checkpoint.validate(tenant.community_id())?;
        let items = self.backend.load_batch(tenant, checkpoint, limit).await?;
        validate_batch(tenant, checkpoint, &items, limit)?;

        let mut counts = RetentionMediaCleanupCounts::default();
        for item in &items {
            let plan = self
                .backend
                .detach_and_reserve_orphans(tenant, item)
                .await?;
            validate_plan(item, &plan)?;
            counts.add_references(plan.reference_outcome, item.attachments.len())?;
            for lease in &plan.orphan_leases {
                let outcome = self
                    .object_backend
                    .delete_if_match(lease.content_hash(), lease.object_version())
                    .await?;
                let finalization = match outcome {
                    MediaObjectDeleteOutcome::Deleted
                    | MediaObjectDeleteOutcome::AlreadyMissing => MediaOrphanFinalization::Deleted,
                    MediaObjectDeleteOutcome::PreconditionFailed => {
                        MediaOrphanFinalization::Preserved
                    }
                    MediaObjectDeleteOutcome::VersionArtifact => {
                        return Err(RetentionMediaCleanupError::UnsafeObjectStore);
                    }
                };
                self.backend
                    .finalize_orphan_deletion(tenant, item, lease, finalization)
                    .await?;
                counts.add_object(outcome)?;
            }
            let next = checkpoint.advance(item)?;
            self.backend
                .advance_checkpoint(tenant, checkpoint, next)
                .await?;
            checkpoint = next;
        }
        Ok(RetentionMediaCleanupOutcome {
            checkpoint,
            counts,
            completed_batch: items.len() < limit,
        })
    }
}

fn validate_batch(
    tenant: &TenantContext,
    checkpoint: RetentionMediaCheckpoint,
    items: &[RetentionMediaCleanupItem],
    limit: usize,
) -> Result<(), RetentionMediaCleanupError> {
    if items.len() > limit {
        return Err(RetentionMediaCleanupError::InvalidBatch);
    }
    let mut previous = checkpoint.source_position;
    let mut operations = BTreeSet::new();
    let mut attachments = BTreeSet::new();
    for item in items {
        if item.community_id != tenant.community_id()
            || item.source_position.sequence() <= previous.sequence()
            || !operations.insert(item.operation_id.as_uuid())
        {
            return Err(RetentionMediaCleanupError::InvalidBatch);
        }
        for attachment in &item.attachments {
            if !attachments.insert((
                attachment.channel_id().as_uuid(),
                attachment.message_id().as_uuid(),
                attachment.media_identity(),
            )) {
                return Err(RetentionMediaCleanupError::InvalidBatch);
            }
        }
        previous = item.source_position;
    }
    Ok(())
}

fn validate_plan(
    item: &RetentionMediaCleanupItem,
    plan: &RetentionMediaRemovalPlan,
) -> Result<(), RetentionMediaCleanupError> {
    if plan.operation_id != item.operation_id
        || plan.orphan_leases.len() > item.attachments.len() * 3
    {
        return Err(RetentionMediaCleanupError::InvalidBackendData);
    }
    let mut hashes = BTreeSet::<MediaContentHash>::new();
    for lease in &plan.orphan_leases {
        if lease.job_id() != item.operation_id.as_uuid() || !hashes.insert(lease.content_hash()) {
            return Err(RetentionMediaCleanupError::InvalidBackendData);
        }
    }
    Ok(())
}
