use std::{collections::BTreeSet, error::Error, fmt, num::NonZeroU64};

use crate::{
    AggregateId, AggregateVersion, CommunityArchivePolicyState, CommunityArchiveSnapshot,
    CommunityId, OperationId, PrincipalId,
};

const MAX_DELETION_TRANSITIONS: usize = 1_000;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct DeletionEvidenceDigest([u8; 32]);

impl DeletionEvidenceDigest {
    pub fn new(value: [u8; 32]) -> Result<Self, CommunityDeletionError> {
        if value == [0; 32] {
            return Err(CommunityDeletionError::InvalidEvidence);
        }
        Ok(Self(value))
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for DeletionEvidenceDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DeletionEvidenceDigest([REDACTED])")
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeletionFenceGeneration(NonZeroU64);

impl DeletionFenceGeneration {
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct CommunityDeletionAuthorityEvidence {
    community_archive: CommunityArchiveSnapshot,
    actor_principal_id: PrincipalId,
    operation_id: OperationId,
    observed_at_millis: u64,
}

impl CommunityDeletionAuthorityEvidence {
    pub fn new(
        community_archive: CommunityArchiveSnapshot,
        actor_principal_id: PrincipalId,
        operation_id: OperationId,
        observed_at_millis: u64,
    ) -> Result<Self, CommunityDeletionError> {
        let evidence = Self {
            community_archive,
            actor_principal_id,
            operation_id,
            observed_at_millis,
        };
        evidence.validate(community_archive.community_id)?;
        Ok(evidence)
    }

    pub const fn community_archive(self) -> CommunityArchiveSnapshot {
        self.community_archive
    }

    pub const fn actor_principal_id(self) -> PrincipalId {
        self.actor_principal_id
    }

    pub const fn operation_id(self) -> OperationId {
        self.operation_id
    }

    pub const fn observed_at_millis(self) -> u64 {
        self.observed_at_millis
    }

    fn validate(self, community_id: CommunityId) -> Result<(), CommunityDeletionError> {
        if self.community_archive.community_id != community_id
            || self.community_archive.state != CommunityArchivePolicyState::Archived
            || self.actor_principal_id.as_uuid().is_nil()
            || self.operation_id.as_uuid().is_nil()
            || self.observed_at_millis == 0
        {
            return Err(CommunityDeletionError::InvalidAuthority);
        }
        Ok(())
    }
}

impl fmt::Debug for CommunityDeletionAuthorityEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CommunityDeletionAuthorityEvidence")
            .field("community_id", &self.community_archive.community_id)
            .field("archive_state", &self.community_archive.state)
            .field("archive_version", &self.community_archive.version)
            .field("actor_principal_id", &"[REDACTED]")
            .field("operation_id", &"[REDACTED]")
            .field("observed_at_millis", &self.observed_at_millis)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionActiveState {
    Requested,
    Verified,
    Reversible,
    Irreversible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionCompletion {
    Deleted,
    RolledBack,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionFailureReason {
    AuthorityUnavailable,
    InventoryMismatch,
    DependencyUnavailable,
    FenceLost,
    VerificationFailed,
    ExecutionConflict,
}

impl CommunityDeletionFailureReason {
    pub const fn retryable(self) -> bool {
        matches!(
            self,
            Self::AuthorityUnavailable
                | Self::DependencyUnavailable
                | Self::FenceLost
                | Self::ExecutionConflict
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionState {
    Requested,
    Verified,
    Reversible,
    Irreversible,
    Completed(CommunityDeletionCompletion),
    Failed {
        failed_from: CommunityDeletionActiveState,
        reason: CommunityDeletionFailureReason,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionTransition {
    Requested {
        authority: CommunityDeletionAuthorityEvidence,
    },
    Verified {
        authority: CommunityDeletionAuthorityEvidence,
        inventory_digest: DeletionEvidenceDigest,
    },
    EnteredReversible {
        authority: CommunityDeletionAuthorityEvidence,
        fence_generation: DeletionFenceGeneration,
    },
    EnteredIrreversible {
        authority: CommunityDeletionAuthorityEvidence,
        fence_generation: DeletionFenceGeneration,
        boundary_digest: DeletionEvidenceDigest,
    },
    Completed {
        authority: CommunityDeletionAuthorityEvidence,
        outcome: CommunityDeletionCompletion,
        verification_digest: DeletionEvidenceDigest,
    },
    Failed {
        authority: CommunityDeletionAuthorityEvidence,
        reason: CommunityDeletionFailureReason,
    },
    Resumed {
        authority: CommunityDeletionAuthorityEvidence,
    },
}

impl CommunityDeletionTransition {
    const fn authority(self) -> CommunityDeletionAuthorityEvidence {
        match self {
            Self::Requested { authority }
            | Self::Verified { authority, .. }
            | Self::EnteredReversible { authority, .. }
            | Self::EnteredIrreversible { authority, .. }
            | Self::Completed { authority, .. }
            | Self::Failed { authority, .. }
            | Self::Resumed { authority } => authority,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityDeletionRecordFields {
    pub deletion_id: AggregateId,
    pub community_id: CommunityId,
    pub transitions: Vec<CommunityDeletionTransition>,
    pub version: AggregateVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunityDeletionCommandOutcome {
    Applied,
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommunityDeletion {
    fields: CommunityDeletionRecordFields,
    state: CommunityDeletionState,
}

impl CommunityDeletion {
    pub fn request(
        deletion_id: AggregateId,
        community_id: CommunityId,
        authority: CommunityDeletionAuthorityEvidence,
    ) -> Result<Self, CommunityDeletionError> {
        if deletion_id.as_uuid().is_nil() || community_id.as_uuid().is_nil() {
            return Err(CommunityDeletionError::InvalidIdentity);
        }
        let transitions = vec![CommunityDeletionTransition::Requested { authority }];
        let state = project_transitions(community_id, &transitions)?;
        Ok(Self {
            fields: CommunityDeletionRecordFields {
                deletion_id,
                community_id,
                transitions,
                version: AggregateVersion::FIRST,
            },
            state,
        })
    }

    pub fn from_record(
        fields: CommunityDeletionRecordFields,
    ) -> Result<Self, CommunityDeletionError> {
        if fields.deletion_id.as_uuid().is_nil()
            || fields.community_id.as_uuid().is_nil()
            || fields.transitions.is_empty()
            || fields.transitions.len() > MAX_DELETION_TRANSITIONS
            || u64::try_from(fields.transitions.len()).ok() != Some(fields.version.get())
        {
            return Err(CommunityDeletionError::InvalidRecord);
        }
        let state = project_transitions(fields.community_id, &fields.transitions)?;
        Ok(Self { fields, state })
    }

    pub const fn fields(&self) -> &CommunityDeletionRecordFields {
        &self.fields
    }

    pub const fn state(&self) -> CommunityDeletionState {
        self.state
    }

    pub fn verify(
        &mut self,
        expected_version: AggregateVersion,
        authority: CommunityDeletionAuthorityEvidence,
        inventory_digest: DeletionEvidenceDigest,
    ) -> Result<CommunityDeletionCommandOutcome, CommunityDeletionError> {
        self.apply(
            expected_version,
            CommunityDeletionTransition::Verified {
                authority,
                inventory_digest,
            },
        )
    }

    pub fn enter_reversible(
        &mut self,
        expected_version: AggregateVersion,
        authority: CommunityDeletionAuthorityEvidence,
        fence_generation: DeletionFenceGeneration,
    ) -> Result<CommunityDeletionCommandOutcome, CommunityDeletionError> {
        self.apply(
            expected_version,
            CommunityDeletionTransition::EnteredReversible {
                authority,
                fence_generation,
            },
        )
    }

    pub fn enter_irreversible(
        &mut self,
        expected_version: AggregateVersion,
        authority: CommunityDeletionAuthorityEvidence,
        fence_generation: DeletionFenceGeneration,
        boundary_digest: DeletionEvidenceDigest,
    ) -> Result<CommunityDeletionCommandOutcome, CommunityDeletionError> {
        self.apply(
            expected_version,
            CommunityDeletionTransition::EnteredIrreversible {
                authority,
                fence_generation,
                boundary_digest,
            },
        )
    }

    pub fn complete(
        &mut self,
        expected_version: AggregateVersion,
        authority: CommunityDeletionAuthorityEvidence,
        verification_digest: DeletionEvidenceDigest,
    ) -> Result<CommunityDeletionCommandOutcome, CommunityDeletionError> {
        self.apply(
            expected_version,
            CommunityDeletionTransition::Completed {
                authority,
                outcome: CommunityDeletionCompletion::Deleted,
                verification_digest,
            },
        )
    }

    pub fn rollback(
        &mut self,
        expected_version: AggregateVersion,
        authority: CommunityDeletionAuthorityEvidence,
        verification_digest: DeletionEvidenceDigest,
    ) -> Result<CommunityDeletionCommandOutcome, CommunityDeletionError> {
        self.apply(
            expected_version,
            CommunityDeletionTransition::Completed {
                authority,
                outcome: CommunityDeletionCompletion::RolledBack,
                verification_digest,
            },
        )
    }

    pub fn fail(
        &mut self,
        expected_version: AggregateVersion,
        authority: CommunityDeletionAuthorityEvidence,
        reason: CommunityDeletionFailureReason,
    ) -> Result<CommunityDeletionCommandOutcome, CommunityDeletionError> {
        self.apply(
            expected_version,
            CommunityDeletionTransition::Failed { authority, reason },
        )
    }

    pub fn resume(
        &mut self,
        expected_version: AggregateVersion,
        authority: CommunityDeletionAuthorityEvidence,
    ) -> Result<CommunityDeletionCommandOutcome, CommunityDeletionError> {
        self.apply(
            expected_version,
            CommunityDeletionTransition::Resumed { authority },
        )
    }

    fn apply(
        &mut self,
        expected_version: AggregateVersion,
        transition: CommunityDeletionTransition,
    ) -> Result<CommunityDeletionCommandOutcome, CommunityDeletionError> {
        if self.fields.transitions.last() == Some(&transition) {
            return Ok(CommunityDeletionCommandOutcome::Unchanged);
        }
        let operation_id = transition.authority().operation_id;
        if self
            .fields
            .transitions
            .iter()
            .any(|existing| existing.authority().operation_id == operation_id)
        {
            return Err(CommunityDeletionError::OperationConflict);
        }
        if expected_version != self.fields.version {
            return Err(CommunityDeletionError::VersionConflict);
        }
        if self.fields.transitions.len() >= MAX_DELETION_TRANSITIONS {
            return Err(CommunityDeletionError::TransitionLimitReached);
        }
        let next_version = self
            .fields
            .version
            .next()
            .ok_or(CommunityDeletionError::VersionExhausted)?;
        let mut transitions = self.fields.transitions.clone();
        transitions.push(transition);
        let state = project_transitions(self.fields.community_id, &transitions)?;
        self.fields.transitions = transitions;
        self.fields.version = next_version;
        self.state = state;
        Ok(CommunityDeletionCommandOutcome::Applied)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum CommunityDeletionError {
    InvalidIdentity,
    InvalidAuthority,
    InvalidEvidence,
    InvalidRecord,
    InvalidTransition,
    InvalidRollback,
    StaleAuthority,
    VersionConflict,
    OperationConflict,
    TransitionLimitReached,
    VersionExhausted,
}

impl fmt::Display for CommunityDeletionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidIdentity => "community deletion identity is invalid",
            Self::InvalidAuthority => "community deletion authority is invalid",
            Self::InvalidEvidence => "community deletion evidence is invalid",
            Self::InvalidRecord => "community deletion record is invalid",
            Self::InvalidTransition => "community deletion transition is invalid",
            Self::InvalidRollback => "community deletion rollback is invalid",
            Self::StaleAuthority => "community deletion authority is stale",
            Self::VersionConflict => "community deletion version conflicts",
            Self::OperationConflict => "community deletion operation conflicts",
            Self::TransitionLimitReached => "community deletion transition limit was reached",
            Self::VersionExhausted => "community deletion version is exhausted",
        })
    }
}

impl Error for CommunityDeletionError {}

#[derive(Clone, Copy)]
struct ProjectedDeletion {
    state: CommunityDeletionState,
    last_authority: Option<CommunityDeletionAuthorityEvidence>,
    fence_generation: Option<DeletionFenceGeneration>,
}

fn project_transitions(
    community_id: CommunityId,
    transitions: &[CommunityDeletionTransition],
) -> Result<CommunityDeletionState, CommunityDeletionError> {
    let mut projected = ProjectedDeletion {
        state: CommunityDeletionState::Requested,
        last_authority: None,
        fence_generation: None,
    };
    let mut operation_ids = BTreeSet::new();
    for (index, transition) in transitions.iter().copied().enumerate() {
        let authority = transition.authority();
        authority.validate(community_id)?;
        if !operation_ids.insert(authority.operation_id) {
            return Err(CommunityDeletionError::OperationConflict);
        }
        if let Some(previous) = projected.last_authority {
            if authority.community_archive.version < previous.community_archive.version {
                return Err(CommunityDeletionError::StaleAuthority);
            }
            if authority.observed_at_millis <= previous.observed_at_millis {
                return Err(CommunityDeletionError::InvalidAuthority);
            }
        }
        projected.state = project_transition(index, projected, transition)?;
        projected.last_authority = Some(authority);
        match transition {
            CommunityDeletionTransition::EnteredReversible {
                fence_generation, ..
            } => projected.fence_generation = Some(fence_generation),
            CommunityDeletionTransition::EnteredIrreversible {
                fence_generation, ..
            } if projected.fence_generation != Some(fence_generation) => {
                return Err(CommunityDeletionError::InvalidTransition);
            }
            _ => {}
        }
    }
    Ok(projected.state)
}

fn project_transition(
    index: usize,
    projected: ProjectedDeletion,
    transition: CommunityDeletionTransition,
) -> Result<CommunityDeletionState, CommunityDeletionError> {
    match transition {
        CommunityDeletionTransition::Requested { .. } if index == 0 => {
            Ok(CommunityDeletionState::Requested)
        }
        CommunityDeletionTransition::Verified { .. }
            if projected.state == CommunityDeletionState::Requested =>
        {
            Ok(CommunityDeletionState::Verified)
        }
        CommunityDeletionTransition::EnteredReversible { .. }
            if projected.state == CommunityDeletionState::Verified =>
        {
            Ok(CommunityDeletionState::Reversible)
        }
        CommunityDeletionTransition::EnteredIrreversible { .. }
            if projected.state == CommunityDeletionState::Reversible =>
        {
            Ok(CommunityDeletionState::Irreversible)
        }
        CommunityDeletionTransition::Completed {
            outcome: CommunityDeletionCompletion::Deleted,
            ..
        } if projected.state == CommunityDeletionState::Irreversible => Ok(
            CommunityDeletionState::Completed(CommunityDeletionCompletion::Deleted),
        ),
        CommunityDeletionTransition::Completed {
            outcome: CommunityDeletionCompletion::RolledBack,
            ..
        } => project_rollback(projected.state),
        CommunityDeletionTransition::Failed { reason, .. } => {
            let failed_from =
                active_state(projected.state).ok_or(CommunityDeletionError::InvalidTransition)?;
            Ok(CommunityDeletionState::Failed {
                failed_from,
                reason,
            })
        }
        CommunityDeletionTransition::Resumed { .. } => match projected.state {
            CommunityDeletionState::Failed {
                failed_from,
                reason,
            } if reason.retryable() => Ok(state_from_active(failed_from)),
            _ => Err(CommunityDeletionError::InvalidTransition),
        },
        _ => Err(CommunityDeletionError::InvalidTransition),
    }
}

fn project_rollback(
    state: CommunityDeletionState,
) -> Result<CommunityDeletionState, CommunityDeletionError> {
    let rollback_allowed = matches!(
        state,
        CommunityDeletionState::Verified
            | CommunityDeletionState::Reversible
            | CommunityDeletionState::Failed {
                failed_from: CommunityDeletionActiveState::Verified
                    | CommunityDeletionActiveState::Reversible,
                ..
            }
    );
    if !rollback_allowed {
        return Err(CommunityDeletionError::InvalidRollback);
    }
    Ok(CommunityDeletionState::Completed(
        CommunityDeletionCompletion::RolledBack,
    ))
}

const fn active_state(state: CommunityDeletionState) -> Option<CommunityDeletionActiveState> {
    match state {
        CommunityDeletionState::Requested => Some(CommunityDeletionActiveState::Requested),
        CommunityDeletionState::Verified => Some(CommunityDeletionActiveState::Verified),
        CommunityDeletionState::Reversible => Some(CommunityDeletionActiveState::Reversible),
        CommunityDeletionState::Irreversible => Some(CommunityDeletionActiveState::Irreversible),
        CommunityDeletionState::Completed(_) | CommunityDeletionState::Failed { .. } => None,
    }
}

const fn state_from_active(state: CommunityDeletionActiveState) -> CommunityDeletionState {
    match state {
        CommunityDeletionActiveState::Requested => CommunityDeletionState::Requested,
        CommunityDeletionActiveState::Verified => CommunityDeletionState::Verified,
        CommunityDeletionActiveState::Reversible => CommunityDeletionState::Reversible,
        CommunityDeletionActiveState::Irreversible => CommunityDeletionState::Irreversible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn authority(
        community_id: CommunityId,
        archive_version: u64,
        sequence: u128,
    ) -> CommunityDeletionAuthorityEvidence {
        CommunityDeletionAuthorityEvidence::new(
            CommunityArchiveSnapshot {
                community_id,
                state: CommunityArchivePolicyState::Archived,
                version: AggregateVersion::new(archive_version).expect("archive version"),
            },
            PrincipalId::from_uuid(Uuid::from_u128(100 + sequence)),
            OperationId::from_uuid(Uuid::from_u128(200 + sequence)),
            u64::try_from(1_000 + sequence).expect("observed time"),
        )
        .expect("authority")
    }

    fn digest(value: u8) -> DeletionEvidenceDigest {
        DeletionEvidenceDigest::new([value; 32]).expect("digest")
    }

    fn deletion() -> CommunityDeletion {
        let community_id = community(1);
        CommunityDeletion::request(
            AggregateId::from_uuid(Uuid::from_u128(10)),
            community_id,
            authority(community_id, 1, 1),
        )
        .expect("request")
    }

    #[test]
    fn lifecycle_preserves_verified_reversible_and_irreversible_evidence() {
        let community_id = community(1);
        let mut deletion = deletion();
        deletion
            .verify(
                AggregateVersion::FIRST,
                authority(community_id, 1, 2),
                digest(1),
            )
            .expect("verify");
        deletion
            .enter_reversible(
                AggregateVersion::new(2).expect("version"),
                authority(community_id, 2, 3),
                DeletionFenceGeneration::new(7).expect("fence"),
            )
            .expect("reversible");
        deletion
            .enter_irreversible(
                AggregateVersion::new(3).expect("version"),
                authority(community_id, 2, 4),
                DeletionFenceGeneration::new(7).expect("fence"),
                digest(2),
            )
            .expect("irreversible");
        deletion
            .complete(
                AggregateVersion::new(4).expect("version"),
                authority(community_id, 3, 5),
                digest(3),
            )
            .expect("complete");
        assert_eq!(
            deletion.state(),
            CommunityDeletionState::Completed(CommunityDeletionCompletion::Deleted)
        );
        assert_eq!(deletion.fields().version.get(), 5);
        assert_eq!(deletion.fields().transitions.len(), 5);
        assert_eq!(
            CommunityDeletion::from_record(deletion.fields().clone()).expect("hydrate"),
            deletion
        );
    }

    #[test]
    fn rollback_is_terminal_only_before_the_irreversible_boundary() {
        let community_id = community(1);
        let mut rolled_back = deletion();
        rolled_back
            .verify(
                AggregateVersion::FIRST,
                authority(community_id, 1, 2),
                digest(1),
            )
            .expect("verify");
        rolled_back
            .rollback(
                AggregateVersion::new(2).expect("version"),
                authority(community_id, 2, 3),
                digest(2),
            )
            .expect("rollback");
        assert_eq!(
            rolled_back.state(),
            CommunityDeletionState::Completed(CommunityDeletionCompletion::RolledBack)
        );

        let mut irreversible = deletion();
        irreversible
            .verify(
                AggregateVersion::FIRST,
                authority(community_id, 1, 2),
                digest(1),
            )
            .expect("verify");
        irreversible
            .enter_reversible(
                AggregateVersion::new(2).expect("version"),
                authority(community_id, 1, 3),
                DeletionFenceGeneration::new(1).expect("fence"),
            )
            .expect("reversible");
        irreversible
            .enter_irreversible(
                AggregateVersion::new(3).expect("version"),
                authority(community_id, 1, 4),
                DeletionFenceGeneration::new(1).expect("fence"),
                digest(2),
            )
            .expect("irreversible");
        let fields = irreversible.fields().clone();
        assert_eq!(
            irreversible.rollback(
                AggregateVersion::new(4).expect("version"),
                authority(community_id, 1, 5),
                digest(3),
            ),
            Err(CommunityDeletionError::InvalidRollback)
        );
        assert_eq!(irreversible.fields(), &fields);
    }

    #[test]
    fn failures_resume_only_when_retryable_and_retain_the_exact_phase() {
        let community_id = community(1);
        let mut deletion = deletion();
        deletion
            .verify(
                AggregateVersion::FIRST,
                authority(community_id, 1, 2),
                digest(1),
            )
            .expect("verify");
        deletion
            .fail(
                AggregateVersion::new(2).expect("version"),
                authority(community_id, 2, 3),
                CommunityDeletionFailureReason::DependencyUnavailable,
            )
            .expect("fail");
        assert_eq!(
            deletion.state(),
            CommunityDeletionState::Failed {
                failed_from: CommunityDeletionActiveState::Verified,
                reason: CommunityDeletionFailureReason::DependencyUnavailable,
            }
        );
        deletion
            .resume(
                AggregateVersion::new(3).expect("version"),
                authority(community_id, 2, 4),
            )
            .expect("resume");
        assert_eq!(deletion.state(), CommunityDeletionState::Verified);
        deletion
            .fail(
                AggregateVersion::new(4).expect("version"),
                authority(community_id, 2, 5),
                CommunityDeletionFailureReason::InventoryMismatch,
            )
            .expect("permanent failure");
        let fields = deletion.fields().clone();
        assert_eq!(
            deletion.resume(
                AggregateVersion::new(5).expect("version"),
                authority(community_id, 2, 6),
            ),
            Err(CommunityDeletionError::InvalidTransition)
        );
        assert_eq!(deletion.fields(), &fields);
    }

    #[test]
    fn stale_authority_and_operation_reuse_are_non_mutating() {
        let community_id = community(1);
        let mut deletion = CommunityDeletion::request(
            AggregateId::from_uuid(Uuid::from_u128(10)),
            community_id,
            authority(community_id, 2, 1),
        )
        .expect("request");
        let fields = deletion.fields().clone();
        assert_eq!(
            deletion.verify(
                AggregateVersion::FIRST,
                authority(community_id, 1, 2),
                digest(1),
            ),
            Err(CommunityDeletionError::StaleAuthority)
        );
        assert_eq!(deletion.fields(), &fields);
        let reused = CommunityDeletionAuthorityEvidence::new(
            CommunityArchiveSnapshot {
                community_id,
                state: CommunityArchivePolicyState::Archived,
                version: AggregateVersion::new(2).expect("archive version"),
            },
            PrincipalId::from_uuid(Uuid::from_u128(999)),
            fields.transitions[0].authority().operation_id,
            2_000,
        )
        .expect("reused authority");
        assert_eq!(
            deletion.verify(AggregateVersion::FIRST, reused, digest(1)),
            Err(CommunityDeletionError::OperationConflict)
        );
        assert_eq!(deletion.fields(), &fields);
    }

    #[test]
    fn hydration_rejects_partial_cross_tenant_and_skipped_state_records() {
        let community_id = community(1);
        let foreign_id = community(2);
        let mut fields = deletion().fields().clone();
        fields
            .transitions
            .push(CommunityDeletionTransition::Verified {
                authority: authority(foreign_id, 1, 2),
                inventory_digest: digest(1),
            });
        fields.version = AggregateVersion::new(2).expect("version");
        assert_eq!(
            CommunityDeletion::from_record(fields),
            Err(CommunityDeletionError::InvalidAuthority)
        );

        let mut skipped = deletion().fields().clone();
        skipped
            .transitions
            .push(CommunityDeletionTransition::EnteredIrreversible {
                authority: authority(community_id, 1, 2),
                fence_generation: DeletionFenceGeneration::new(1).expect("fence"),
                boundary_digest: digest(2),
            });
        skipped.version = AggregateVersion::new(2).expect("version");
        assert_eq!(
            CommunityDeletion::from_record(skipped),
            Err(CommunityDeletionError::InvalidTransition)
        );
    }
}
