use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use collab::deletion::executor::{
    COMMUNITY_DELETION_PHASES, CommunityDeletionBackendError, CommunityDeletionBoundaryCommit,
    CommunityDeletionCheckpoint, CommunityDeletionExecutionRecord, CommunityDeletionExecutor,
    CommunityDeletionExecutorBackend, CommunityDeletionExecutorError, CommunityDeletionPhase,
    CommunityDeletionPhaseAttempt, CommunityDeletionPhaseCommit,
    CommunityDeletionPhaseCommitOutcome, CommunityDeletionStepOutcome,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityArchivePolicyState, CommunityArchiveSnapshot,
    CommunityDeletion, CommunityDeletionAuthorityEvidence, CommunityDeletionCompletion,
    CommunityId, DeletionEvidenceDigest, DeletionFenceGeneration, OperationId, PrincipalId,
    TenantContext, TrustedTenantRoute,
};
use uuid::Uuid;

#[derive(Clone, Copy)]
enum InjectedFault {
    None,
    BeforePhase(CommunityDeletionPhase),
    AfterPhase(CommunityDeletionPhase),
    AfterCompletion,
}

struct BackendState {
    deletion: CommunityDeletion,
    checkpoint: Option<CommunityDeletionCheckpoint>,
    fault: InjectedFault,
    fault_fired: bool,
    effects: [u32; COMMUNITY_DELETION_PHASES.len()],
    phase_attempts: [u32; COMMUNITY_DELETION_PHASES.len()],
    committed_order: Vec<CommunityDeletionPhase>,
    boundary_commits: u32,
    completion_effects: u32,
}

#[derive(Clone)]
struct TestBackend {
    state: Arc<Mutex<BackendState>>,
}

impl TestBackend {
    fn new(fault: InjectedFault) -> Self {
        Self {
            state: Arc::new(Mutex::new(BackendState {
                deletion: reversible_deletion(),
                checkpoint: None,
                fault,
                fault_fired: false,
                effects: [0; COMMUNITY_DELETION_PHASES.len()],
                phase_attempts: [0; COMMUNITY_DELETION_PHASES.len()],
                committed_order: Vec::new(),
                boundary_commits: 0,
                completion_effects: 0,
            })),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, BackendState> {
        self.state.lock().expect("backend lock")
    }
}

#[async_trait]
impl CommunityDeletionExecutorBackend for TestBackend {
    async fn load_execution(
        &self,
        _tenant: &TenantContext,
        _deletion_id: AggregateId,
    ) -> Result<CommunityDeletionExecutionRecord, CommunityDeletionBackendError> {
        let state = self.state();
        Ok(CommunityDeletionExecutionRecord {
            deletion: state.deletion.clone(),
            checkpoint: state.checkpoint,
        })
    }

    async fn record_irreversible_boundary(
        &self,
        _tenant: &TenantContext,
        expected_deletion: &CommunityDeletion,
    ) -> Result<CommunityDeletionBoundaryCommit, CommunityDeletionBackendError> {
        let mut state = self.state();
        if &state.deletion != expected_deletion || state.checkpoint.is_some() {
            return Err(CommunityDeletionBackendError::StaleCheckpoint);
        }
        let mut deletion = state.deletion.clone();
        deletion
            .enter_irreversible(
                deletion.fields().version,
                authority(4),
                DeletionFenceGeneration::new(7)
                    .ok_or(CommunityDeletionBackendError::InvalidData)?,
                digest(4),
            )
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        let checkpoint = CommunityDeletionCheckpoint::from_irreversible(&deletion)
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        state.deletion = deletion.clone();
        state.checkpoint = Some(checkpoint);
        state.boundary_commits += 1;
        Ok(CommunityDeletionBoundaryCommit {
            deletion,
            checkpoint,
        })
    }

    async fn commit_phase(
        &self,
        _tenant: &TenantContext,
        attempt: CommunityDeletionPhaseAttempt,
    ) -> Result<CommunityDeletionPhaseCommit, CommunityDeletionBackendError> {
        let mut state = self.state();
        let phase_index = phase_index(attempt.phase());
        state.phase_attempts[phase_index] += 1;
        if matches!(state.fault, InjectedFault::BeforePhase(phase) if phase == attempt.phase())
            && !state.fault_fired
        {
            state.fault_fired = true;
            return Err(CommunityDeletionBackendError::Unavailable);
        }

        let current = state
            .checkpoint
            .ok_or(CommunityDeletionBackendError::InvalidData)?;
        if current != attempt.checkpoint() {
            let already_committed = attempt
                .checkpoint()
                .advance(attempt.phase(), current.evidence_digest())
                .is_ok_and(|expected| expected == current);
            if already_committed {
                return Ok(CommunityDeletionPhaseCommit {
                    checkpoint: current,
                    outcome: CommunityDeletionPhaseCommitOutcome::AlreadyCommitted,
                });
            }
            return Err(CommunityDeletionBackendError::StaleCheckpoint);
        }

        state.effects[phase_index] += 1;
        if state.effects[phase_index] != 1 {
            return Err(CommunityDeletionBackendError::InvalidData);
        }
        state.committed_order.push(attempt.phase());
        let checkpoint = current
            .advance(
                attempt.phase(),
                digest(10 + u8::try_from(phase_index).unwrap_or(0)),
            )
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        state.checkpoint = Some(checkpoint);

        if matches!(state.fault, InjectedFault::AfterPhase(phase) if phase == attempt.phase())
            && !state.fault_fired
        {
            state.fault_fired = true;
            return Err(CommunityDeletionBackendError::OutcomeUnknown);
        }
        Ok(CommunityDeletionPhaseCommit {
            checkpoint,
            outcome: CommunityDeletionPhaseCommitOutcome::Committed,
        })
    }

    async fn complete(
        &self,
        _tenant: &TenantContext,
        expected_deletion: &CommunityDeletion,
        checkpoint: CommunityDeletionCheckpoint,
    ) -> Result<CommunityDeletion, CommunityDeletionBackendError> {
        let mut state = self.state();
        if &state.deletion != expected_deletion
            || state.checkpoint != Some(checkpoint)
            || !checkpoint.is_complete()
        {
            return Err(CommunityDeletionBackendError::StaleCheckpoint);
        }
        let mut deletion = state.deletion.clone();
        deletion
            .complete(
                deletion.fields().version,
                authority(5),
                checkpoint.evidence_digest(),
            )
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        state.deletion = deletion.clone();
        state.completion_effects += 1;
        if matches!(state.fault, InjectedFault::AfterCompletion) && !state.fault_fired {
            state.fault_fired = true;
            return Err(CommunityDeletionBackendError::OutcomeUnknown);
        }
        Ok(deletion)
    }
}

#[test]
fn every_phase_resumes_after_precommit_failure_without_skipping() {
    for failed_phase in COMMUNITY_DELETION_PHASES {
        let backend = TestBackend::new(InjectedFault::BeforePhase(failed_phase));
        drive_to_completion(&backend);
        let state = backend.state();
        assert_eq!(state.boundary_commits, 1);
        assert_eq!(state.effects, [1; COMMUNITY_DELETION_PHASES.len()]);
        assert_eq!(state.committed_order, COMMUNITY_DELETION_PHASES);
        assert_eq!(state.phase_attempts[phase_index(failed_phase)], 2);
        assert_eq!(state.completion_effects, 1);
    }
}

#[test]
fn every_phase_reloads_unknown_commit_without_repeating_effect() {
    for unknown_phase in COMMUNITY_DELETION_PHASES {
        let backend = TestBackend::new(InjectedFault::AfterPhase(unknown_phase));
        drive_to_completion(&backend);
        let state = backend.state();
        assert_eq!(state.effects, [1; COMMUNITY_DELETION_PHASES.len()]);
        assert_eq!(state.committed_order, COMMUNITY_DELETION_PHASES);
        assert_eq!(state.phase_attempts, [1; COMMUNITY_DELETION_PHASES.len()]);
        assert_eq!(state.completion_effects, 1);
    }
}

#[test]
fn unknown_terminal_commit_reloads_as_completed() {
    let backend = TestBackend::new(InjectedFault::AfterCompletion);
    drive_to_completion(&backend);
    let state = backend.state();
    assert_eq!(state.effects, [1; COMMUNITY_DELETION_PHASES.len()]);
    assert_eq!(state.completion_effects, 1);
    assert_eq!(
        state.deletion.state(),
        collaboration_domain::CommunityDeletionState::Completed(
            CommunityDeletionCompletion::Deleted
        )
    );
}

#[test]
fn executor_rejects_irreversible_state_without_atomic_checkpoint() {
    let backend = TestBackend::new(InjectedFault::None);
    {
        let mut state = backend.state();
        state
            .deletion
            .enter_irreversible(
                AggregateVersion::new(3).expect("version"),
                authority(4),
                DeletionFenceGeneration::new(7).expect("fence"),
                digest(4),
            )
            .expect("irreversible");
    }
    let executor = CommunityDeletionExecutor::new(backend);
    let error = futures::executor::block_on(executor.run_step(&tenant(), deletion_id()))
        .expect_err("missing checkpoint must fail");
    assert!(matches!(
        error,
        CommunityDeletionExecutorError::InvalidCheckpoint
    ));
}

fn drive_to_completion(backend: &TestBackend) {
    let executor = CommunityDeletionExecutor::new(backend.clone());
    let mut completed = false;
    for _ in 0..16 {
        match futures::executor::block_on(executor.run_step(&tenant(), deletion_id())) {
            Ok(CommunityDeletionStepOutcome::Completed(CommunityDeletionCompletion::Deleted)) => {
                completed = true;
                break;
            }
            Ok(
                CommunityDeletionStepOutcome::BoundaryRecorded(_)
                | CommunityDeletionStepOutcome::PhaseCommitted { .. },
            ) => {}
            Ok(outcome) => panic!("unexpected executor outcome: {outcome:?}"),
            Err(CommunityDeletionExecutorError::Backend(
                CommunityDeletionBackendError::Unavailable
                | CommunityDeletionBackendError::OutcomeUnknown,
            )) => {}
            Err(error) => panic!("unexpected executor error: {error}"),
        }
    }
    assert!(completed, "executor did not reach terminal deletion");
}

fn reversible_deletion() -> CommunityDeletion {
    let mut deletion = CommunityDeletion::request(deletion_id(), community_id(), authority(1))
        .expect("deletion request");
    deletion
        .verify(AggregateVersion::FIRST, authority(2), digest(2))
        .expect("verified deletion");
    deletion
        .enter_reversible(
            AggregateVersion::new(2).expect("version"),
            authority(3),
            DeletionFenceGeneration::new(7).expect("fence"),
        )
        .expect("reversible deletion");
    deletion
}

fn tenant() -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id(), "deletion-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn authority(sequence: u64) -> CommunityDeletionAuthorityEvidence {
    CommunityDeletionAuthorityEvidence::new(
        CommunityArchiveSnapshot {
            community_id: community_id(),
            state: CommunityArchivePolicyState::Archived,
            version: AggregateVersion::new(sequence).expect("archive version"),
        },
        PrincipalId::from_uuid(Uuid::from_u128(100)),
        OperationId::from_uuid(Uuid::from_u128(200 + u128::from(sequence))),
        1_000 + sequence,
    )
    .expect("authority")
}

fn digest(value: u8) -> DeletionEvidenceDigest {
    DeletionEvidenceDigest::new([value; 32]).expect("digest")
}

fn community_id() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(1))
}

fn deletion_id() -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(2))
}

fn phase_index(phase: CommunityDeletionPhase) -> usize {
    COMMUNITY_DELETION_PHASES
        .iter()
        .position(|candidate| *candidate == phase)
        .expect("known phase")
}
