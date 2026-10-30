use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityDeletion, CommunityDeletionActiveState,
    CommunityDeletionCompletion, CommunityDeletionState, CommunityDeletionTransition, CommunityId,
    DeletionEvidenceDigest, DeletionFenceGeneration, TenantContext,
};

pub const COMMUNITY_DELETION_PHASES: [CommunityDeletionPhase; 6] = [
    CommunityDeletionPhase::Database,
    CommunityDeletionPhase::Search,
    CommunityDeletionPhase::Cache,
    CommunityDeletionPhase::Push,
    CommunityDeletionPhase::ObjectStorage,
    CommunityDeletionPhase::Git,
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CommunityDeletionPhase {
    Database,
    Search,
    Cache,
    Push,
    ObjectStorage,
    Git,
}

impl CommunityDeletionPhase {
    const fn index(self) -> usize {
        match self {
            Self::Database => 0,
            Self::Search => 1,
            Self::Cache => 2,
            Self::Push => 3,
            Self::ObjectStorage => 4,
            Self::Git => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommunityDeletionCheckpointFields {
    pub community_id: CommunityId,
    pub deletion_id: AggregateId,
    pub boundary_version: AggregateVersion,
    pub checkpoint_version: u64,
    pub fence_generation: DeletionFenceGeneration,
    pub boundary_digest: DeletionEvidenceDigest,
    pub evidence_digest: DeletionEvidenceDigest,
    pub completed_phases: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommunityDeletionCheckpoint {
    fields: CommunityDeletionCheckpointFields,
}

impl CommunityDeletionCheckpoint {
    pub fn from_record(
        fields: CommunityDeletionCheckpointFields,
    ) -> Result<Self, CommunityDeletionExecutorError> {
        let checkpoint = Self { fields };
        checkpoint.validate_shape()?;
        Ok(checkpoint)
    }

    pub fn from_irreversible(
        deletion: &CommunityDeletion,
    ) -> Result<Self, CommunityDeletionExecutorError> {
        let (boundary_version, fence_generation, boundary_digest) = irreversible_boundary(deletion)
            .ok_or(CommunityDeletionExecutorError::InvalidExecution)?;
        let checkpoint = Self {
            fields: CommunityDeletionCheckpointFields {
                community_id: deletion.fields().community_id,
                deletion_id: deletion.fields().deletion_id,
                boundary_version,
                checkpoint_version: 1,
                fence_generation,
                boundary_digest,
                evidence_digest: boundary_digest,
                completed_phases: 0,
            },
        };
        checkpoint.validate_for(deletion)?;
        Ok(checkpoint)
    }

    pub const fn fields(self) -> CommunityDeletionCheckpointFields {
        self.fields
    }

    pub const fn community_id(self) -> CommunityId {
        self.fields.community_id
    }

    pub const fn deletion_id(self) -> AggregateId {
        self.fields.deletion_id
    }

    pub const fn checkpoint_version(self) -> u64 {
        self.fields.checkpoint_version
    }

    pub const fn fence_generation(self) -> DeletionFenceGeneration {
        self.fields.fence_generation
    }

    pub const fn evidence_digest(self) -> DeletionEvidenceDigest {
        self.fields.evidence_digest
    }

    pub const fn completed_phases(self) -> u8 {
        self.fields.completed_phases
    }

    pub fn next_phase(self) -> Option<CommunityDeletionPhase> {
        COMMUNITY_DELETION_PHASES
            .get(usize::from(self.fields.completed_phases))
            .copied()
    }

    pub fn advance(
        self,
        phase: CommunityDeletionPhase,
        evidence_digest: DeletionEvidenceDigest,
    ) -> Result<Self, CommunityDeletionExecutorError> {
        if self.next_phase() != Some(phase) || evidence_digest == self.fields.evidence_digest {
            return Err(CommunityDeletionExecutorError::InvalidCheckpoint);
        }
        let checkpoint = Self {
            fields: CommunityDeletionCheckpointFields {
                checkpoint_version: self
                    .fields
                    .checkpoint_version
                    .checked_add(1)
                    .ok_or(CommunityDeletionExecutorError::VersionExhausted)?,
                evidence_digest,
                completed_phases: self
                    .fields
                    .completed_phases
                    .checked_add(1)
                    .ok_or(CommunityDeletionExecutorError::VersionExhausted)?,
                ..self.fields
            },
        };
        checkpoint.validate_shape()?;
        Ok(checkpoint)
    }

    pub fn is_complete(self) -> bool {
        usize::from(self.fields.completed_phases) == COMMUNITY_DELETION_PHASES.len()
    }

    fn validate_shape(self) -> Result<(), CommunityDeletionExecutorError> {
        if self.fields.community_id.as_uuid().is_nil()
            || self.fields.deletion_id.as_uuid().is_nil()
            || usize::from(self.fields.completed_phases) > COMMUNITY_DELETION_PHASES.len()
            || self.fields.checkpoint_version
                != u64::from(self.fields.completed_phases)
                    .checked_add(1)
                    .ok_or(CommunityDeletionExecutorError::VersionExhausted)?
        {
            return Err(CommunityDeletionExecutorError::InvalidCheckpoint);
        }
        Ok(())
    }

    fn validate_for(
        self,
        deletion: &CommunityDeletion,
    ) -> Result<(), CommunityDeletionExecutorError> {
        self.validate_shape()?;
        let fields = deletion.fields();
        let (boundary_version, fence_generation, boundary_digest) = irreversible_boundary(deletion)
            .ok_or(CommunityDeletionExecutorError::InvalidExecution)?;
        if self.fields.community_id != fields.community_id
            || self.fields.deletion_id != fields.deletion_id
            || self.fields.boundary_version != boundary_version
            || self.fields.fence_generation != fence_generation
            || self.fields.boundary_digest != boundary_digest
            || fields.version < boundary_version
            || (matches!(
                deletion.state(),
                CommunityDeletionState::Completed(CommunityDeletionCompletion::Deleted)
            ) && (!self.is_complete()
                || fields.version
                    != boundary_version
                        .next()
                        .ok_or(CommunityDeletionExecutorError::VersionExhausted)?))
        {
            return Err(CommunityDeletionExecutorError::InvalidCheckpoint);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityDeletionExecutionRecord {
    pub deletion: CommunityDeletion,
    pub checkpoint: Option<CommunityDeletionCheckpoint>,
}

impl CommunityDeletionExecutionRecord {
    pub(crate) fn validate(
        &self,
        tenant: &TenantContext,
        deletion_id: AggregateId,
    ) -> Result<(), CommunityDeletionExecutorError> {
        if self.deletion.fields().community_id != tenant.community_id()
            || self.deletion.fields().deletion_id != deletion_id
            || deletion_id.as_uuid().is_nil()
        {
            return Err(CommunityDeletionExecutorError::InvalidExecution);
        }
        match self.deletion.state() {
            CommunityDeletionState::Irreversible
            | CommunityDeletionState::Failed {
                failed_from: CommunityDeletionActiveState::Irreversible,
                ..
            }
            | CommunityDeletionState::Completed(CommunityDeletionCompletion::Deleted) => self
                .checkpoint
                .ok_or(CommunityDeletionExecutorError::InvalidCheckpoint)?
                .validate_for(&self.deletion),
            CommunityDeletionState::Requested
            | CommunityDeletionState::Verified
            | CommunityDeletionState::Reversible
            | CommunityDeletionState::Failed {
                failed_from:
                    CommunityDeletionActiveState::Requested
                    | CommunityDeletionActiveState::Verified
                    | CommunityDeletionActiveState::Reversible,
                ..
            }
            | CommunityDeletionState::Completed(CommunityDeletionCompletion::RolledBack) => {
                if self.checkpoint.is_some() {
                    Err(CommunityDeletionExecutorError::InvalidCheckpoint)
                } else {
                    Ok(())
                }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityDeletionBoundaryCommit {
    pub deletion: CommunityDeletion,
    pub checkpoint: CommunityDeletionCheckpoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommunityDeletionPhaseAttempt {
    checkpoint: CommunityDeletionCheckpoint,
    phase: CommunityDeletionPhase,
}

impl CommunityDeletionPhaseAttempt {
    pub const fn checkpoint(self) -> CommunityDeletionCheckpoint {
        self.checkpoint
    }

    pub const fn phase(self) -> CommunityDeletionPhase {
        self.phase
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionPhaseCommitOutcome {
    Committed,
    AlreadyCommitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommunityDeletionPhaseCommit {
    pub checkpoint: CommunityDeletionCheckpoint,
    pub outcome: CommunityDeletionPhaseCommitOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionBackendError {
    Unavailable,
    StaleCheckpoint,
    FenceLost,
    InvalidData,
    OutcomeUnknown,
}

impl fmt::Display for CommunityDeletionBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "community deletion backend is unavailable",
            Self::StaleCheckpoint => "community deletion checkpoint is stale",
            Self::FenceLost => "community deletion fence was lost",
            Self::InvalidData => "community deletion backend data is invalid",
            Self::OutcomeUnknown => "community deletion backend outcome is unknown",
        })
    }
}

impl Error for CommunityDeletionBackendError {}

#[async_trait]
pub trait CommunityDeletionExecutorBackend: Send + Sync {
    async fn load_execution(
        &self,
        tenant: &TenantContext,
        deletion_id: AggregateId,
    ) -> Result<CommunityDeletionExecutionRecord, CommunityDeletionBackendError>;

    /// Appends the irreversible aggregate transition and creates its checkpoint atomically.
    async fn record_irreversible_boundary(
        &self,
        tenant: &TenantContext,
        expected_deletion: &CommunityDeletion,
    ) -> Result<CommunityDeletionBoundaryCommit, CommunityDeletionBackendError>;

    /// The effect and receipt use the exact deletion, fence, phase and checkpoint version as an
    /// idempotency key. An unobservable receipt must return `OutcomeUnknown` and remain reloadable.
    async fn commit_phase(
        &self,
        tenant: &TenantContext,
        attempt: CommunityDeletionPhaseAttempt,
    ) -> Result<CommunityDeletionPhaseCommit, CommunityDeletionBackendError>;

    /// Appends terminal deletion only after every phase receipt and its final absence evidence.
    async fn complete(
        &self,
        tenant: &TenantContext,
        expected_deletion: &CommunityDeletion,
        checkpoint: CommunityDeletionCheckpoint,
    ) -> Result<CommunityDeletion, CommunityDeletionBackendError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommunityDeletionStepOutcome {
    BoundaryRecorded(CommunityDeletionCheckpoint),
    PhaseCommitted {
        phase: CommunityDeletionPhase,
        checkpoint: CommunityDeletionCheckpoint,
        outcome: CommunityDeletionPhaseCommitOutcome,
    },
    Completed(CommunityDeletionCompletion),
    NotReady(CommunityDeletionState),
}

#[derive(Debug)]
pub enum CommunityDeletionExecutorError {
    InvalidInput,
    InvalidExecution,
    InvalidCheckpoint,
    InvalidTransition,
    VersionExhausted,
    Backend(CommunityDeletionBackendError),
}

impl fmt::Display for CommunityDeletionExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidInput => "community deletion executor input is invalid",
            Self::InvalidExecution => "community deletion execution record is invalid",
            Self::InvalidCheckpoint => "community deletion executor checkpoint is invalid",
            Self::InvalidTransition => "community deletion executor transition is invalid",
            Self::VersionExhausted => "community deletion executor version is exhausted",
            Self::Backend(error) => return error.fmt(formatter),
        })
    }
}

impl Error for CommunityDeletionExecutorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CommunityDeletionBackendError> for CommunityDeletionExecutorError {
    fn from(error: CommunityDeletionBackendError) -> Self {
        Self::Backend(error)
    }
}

pub struct CommunityDeletionExecutor<Backend> {
    backend: Backend,
}

impl<Backend> CommunityDeletionExecutor<Backend>
where
    Backend: CommunityDeletionExecutorBackend,
{
    pub const fn new(backend: Backend) -> Self {
        Self { backend }
    }

    pub fn into_backend(self) -> Backend {
        self.backend
    }

    pub async fn run_step(
        &self,
        tenant: &TenantContext,
        deletion_id: AggregateId,
    ) -> Result<CommunityDeletionStepOutcome, CommunityDeletionExecutorError> {
        if deletion_id.as_uuid().is_nil() || tenant.community_id().as_uuid().is_nil() {
            return Err(CommunityDeletionExecutorError::InvalidInput);
        }
        let execution = self.backend.load_execution(tenant, deletion_id).await?;
        execution.validate(tenant, deletion_id)?;
        match execution.deletion.state() {
            CommunityDeletionState::Reversible => {
                let commit = self
                    .backend
                    .record_irreversible_boundary(tenant, &execution.deletion)
                    .await?;
                validate_boundary_commit(&execution.deletion, &commit)?;
                Ok(CommunityDeletionStepOutcome::BoundaryRecorded(
                    commit.checkpoint,
                ))
            }
            CommunityDeletionState::Irreversible => {
                let checkpoint = execution
                    .checkpoint
                    .ok_or(CommunityDeletionExecutorError::InvalidCheckpoint)?;
                if let Some(phase) = checkpoint.next_phase() {
                    let attempt = CommunityDeletionPhaseAttempt { checkpoint, phase };
                    let commit = self.backend.commit_phase(tenant, attempt).await?;
                    validate_phase_commit(attempt, commit)?;
                    Ok(CommunityDeletionStepOutcome::PhaseCommitted {
                        phase,
                        checkpoint: commit.checkpoint,
                        outcome: commit.outcome,
                    })
                } else {
                    let deletion = self
                        .backend
                        .complete(tenant, &execution.deletion, checkpoint)
                        .await?;
                    validate_completion(&execution.deletion, checkpoint, &deletion)?;
                    Ok(CommunityDeletionStepOutcome::Completed(
                        CommunityDeletionCompletion::Deleted,
                    ))
                }
            }
            CommunityDeletionState::Completed(completion) => {
                Ok(CommunityDeletionStepOutcome::Completed(completion))
            }
            state => Ok(CommunityDeletionStepOutcome::NotReady(state)),
        }
    }
}

fn irreversible_boundary(
    deletion: &CommunityDeletion,
) -> Option<(
    AggregateVersion,
    DeletionFenceGeneration,
    DeletionEvidenceDigest,
)> {
    deletion.fields().transitions.iter().enumerate().find_map(
        |(index, transition)| match transition {
            CommunityDeletionTransition::EnteredIrreversible {
                fence_generation,
                boundary_digest,
                ..
            } => Some((
                AggregateVersion::new(u64::try_from(index).ok()?.checked_add(1)?)?,
                *fence_generation,
                *boundary_digest,
            )),
            _ => None,
        },
    )
}

fn validate_boundary_commit(
    expected: &CommunityDeletion,
    commit: &CommunityDeletionBoundaryCommit,
) -> Result<(), CommunityDeletionExecutorError> {
    let expected_fields = expected.fields();
    let actual_fields = commit.deletion.fields();
    if expected.state() != CommunityDeletionState::Reversible
        || commit.deletion.state() != CommunityDeletionState::Irreversible
        || actual_fields.community_id != expected_fields.community_id
        || actual_fields.deletion_id != expected_fields.deletion_id
        || actual_fields.version
            != expected_fields
                .version
                .next()
                .ok_or(CommunityDeletionExecutorError::VersionExhausted)?
        || actual_fields.transitions.len() != expected_fields.transitions.len() + 1
        || !actual_fields
            .transitions
            .starts_with(&expected_fields.transitions)
    {
        return Err(CommunityDeletionExecutorError::InvalidTransition);
    }
    commit.checkpoint.validate_for(&commit.deletion)?;
    if commit.checkpoint.completed_phases() != 0
        || commit.checkpoint.fields.boundary_version != actual_fields.version
    {
        return Err(CommunityDeletionExecutorError::InvalidCheckpoint);
    }
    Ok(())
}

fn validate_phase_commit(
    attempt: CommunityDeletionPhaseAttempt,
    commit: CommunityDeletionPhaseCommit,
) -> Result<(), CommunityDeletionExecutorError> {
    let expected = attempt
        .checkpoint
        .advance(attempt.phase, commit.checkpoint.evidence_digest())?;
    if commit.checkpoint != expected
        || attempt.phase.index() + 1 != usize::from(expected.completed_phases())
    {
        return Err(CommunityDeletionExecutorError::InvalidCheckpoint);
    }
    Ok(())
}

fn validate_completion(
    expected: &CommunityDeletion,
    checkpoint: CommunityDeletionCheckpoint,
    completed: &CommunityDeletion,
) -> Result<(), CommunityDeletionExecutorError> {
    let expected_fields = expected.fields();
    let completed_fields = completed.fields();
    if expected.state() != CommunityDeletionState::Irreversible
        || !checkpoint.is_complete()
        || completed.state()
            != CommunityDeletionState::Completed(CommunityDeletionCompletion::Deleted)
        || completed_fields.community_id != expected_fields.community_id
        || completed_fields.deletion_id != expected_fields.deletion_id
        || completed_fields.version
            != expected_fields
                .version
                .next()
                .ok_or(CommunityDeletionExecutorError::VersionExhausted)?
        || completed_fields.transitions.len() != expected_fields.transitions.len() + 1
        || !completed_fields
            .transitions
            .starts_with(&expected_fields.transitions)
        || !matches!(
            completed_fields.transitions.last(),
            Some(CommunityDeletionTransition::Completed {
                outcome: CommunityDeletionCompletion::Deleted,
                verification_digest,
                ..
            }) if *verification_digest == checkpoint.evidence_digest()
        )
    {
        return Err(CommunityDeletionExecutorError::InvalidTransition);
    }
    checkpoint.validate_for(completed)
}
