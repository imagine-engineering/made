use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::{
    deletion::{
        executor::{
            COMMUNITY_DELETION_PHASES, CommunityDeletionBackendError,
            CommunityDeletionBoundaryCommit, CommunityDeletionCheckpoint,
            CommunityDeletionExecutionRecord, CommunityDeletionExecutor,
            CommunityDeletionExecutorBackend, CommunityDeletionExecutorError,
            CommunityDeletionPhase, CommunityDeletionPhaseAttempt, CommunityDeletionPhaseCommit,
            CommunityDeletionPhaseCommitOutcome, CommunityDeletionStepOutcome,
        },
        recovery::{
            CommunityDeletionOperatorApi, CommunityDeletionOperatorCommand,
            CommunityDeletionOperatorOutcome, CommunityDeletionOperatorStage,
            CommunityDeletionRecoveryAction, CommunityDeletionRecoveryBackend,
            CommunityDeletionRecoveryError,
        },
    },
    retention::worker::{
        RetentionAuthorityAction, RetentionAuthorityBackend, RetentionAuthoritySnapshot,
        RetentionBackendError, RetentionBatchCommit, RetentionCommitOutcome, RetentionDeleteCause,
        RetentionSourcePosition, RetentionWorkItem, RetentionWorker, RetentionWorkerCheckpoint,
        RetentionWorkerError,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, ArchiveRetentionRule, AuthenticatedPrincipal,
    AuthorizationAction, AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind,
    AuthorizationScope, CommunityArchivePolicyState, CommunityArchiveSnapshot, CommunityDeletion,
    CommunityDeletionAuthorityEvidence, CommunityDeletionCompletion, CommunityId,
    CommunityMembership, CommunityRetentionPolicy, CommunityRetentionPolicyFields,
    DeletionEvidenceDigest, DeletionFenceGeneration, MembershipRole, MembershipStatus, OperationId,
    PrincipalId, PrincipalScopes, RetentionArchiveSnapshot, RetentionEventKind,
    RetentionPersistenceClass, RetentionPolicySchemaVersion, RetentionReason, RetentionRecord,
    RetentionSnapshot, RetentionTtl, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

#[derive(Deserialize)]
struct MigrationFixtureDocument {
    format_version: u8,
    fixtures: Vec<MigrationFixture>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct MigrationFixture {
    fixture_id: String,
    data_source_id: String,
    version: String,
    migration_state: String,
    contains_private_key_material: bool,
    records: Vec<Value>,
    expected: Value,
}

#[derive(Clone, Copy)]
enum RetentionFault {
    BeforeCommit,
    AfterCommit,
}

struct RetentionState {
    checkpoint: Option<RetentionWorkerCheckpoint>,
    items: Vec<RetentionWorkItem>,
    actions: BTreeMap<u64, RetentionAuthorityAction>,
    effects: BTreeMap<u64, u32>,
    fault: Option<RetentionFault>,
}

#[derive(Clone)]
struct FaultingRetentionBackend {
    state: Arc<Mutex<RetentionState>>,
}

impl FaultingRetentionBackend {
    fn new(items: Vec<RetentionWorkItem>, fault: RetentionFault) -> Self {
        Self {
            state: Arc::new(Mutex::new(RetentionState {
                checkpoint: None,
                items,
                actions: BTreeMap::new(),
                effects: BTreeMap::new(),
                fault: Some(fault),
            })),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, RetentionState> {
        self.state.lock().expect("retention backend lock")
    }
}

#[async_trait]
impl RetentionAuthorityBackend for FaultingRetentionBackend {
    async fn load_checkpoint(
        &self,
        _tenant: &TenantContext,
    ) -> Result<Option<RetentionWorkerCheckpoint>, RetentionBackendError> {
        Ok(self.state().checkpoint.clone())
    }

    async fn load_batch(
        &self,
        _tenant: &TenantContext,
        checkpoint: &RetentionWorkerCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionWorkItem>, RetentionBackendError> {
        Ok(self
            .state()
            .items
            .iter()
            .filter(|item| item.position().sequence() > checkpoint.cursor().sequence())
            .take(limit)
            .cloned()
            .collect())
    }

    async fn commit_batch(
        &self,
        _tenant: &TenantContext,
        commit: &RetentionBatchCommit,
    ) -> Result<RetentionCommitOutcome, RetentionBackendError> {
        let mut state = self.state();
        if matches!(state.fault, Some(RetentionFault::BeforeCommit)) {
            state.fault = None;
            return Err(RetentionBackendError::Unavailable);
        }
        let current = state.checkpoint.clone().unwrap_or_else(|| {
            RetentionWorkerCheckpoint::initial(commit.expected_checkpoint().community_id())
        });
        if current == *commit.next_checkpoint() {
            return Ok(RetentionCommitOutcome::AlreadyCommitted);
        }
        if current != *commit.expected_checkpoint() {
            return Err(RetentionBackendError::StaleCheckpoint);
        }
        for evaluation in commit.evaluations() {
            let sequence = evaluation.position().sequence();
            if let Some(existing) = state.actions.get(&sequence) {
                if *existing != evaluation.action() {
                    return Err(RetentionBackendError::InvalidData);
                }
                continue;
            }
            state.actions.insert(sequence, evaluation.action());
            *state.effects.entry(sequence).or_default() += 1;
        }
        state.checkpoint = Some(commit.next_checkpoint().clone());
        if matches!(state.fault, Some(RetentionFault::AfterCommit)) {
            state.fault = None;
            return Err(RetentionBackendError::OutcomeUnknown);
        }
        Ok(RetentionCommitOutcome::Committed)
    }
}

#[derive(Clone, Copy)]
enum DeletionFault {
    AfterPhase(CommunityDeletionPhase),
    AfterRestore,
}

struct DeletionState {
    execution: CommunityDeletionExecutionRecord,
    fault: DeletionFault,
    fault_fired: bool,
    phase_effects: [u32; COMMUNITY_DELETION_PHASES.len()],
    restore_effects: u32,
    completion_effects: u32,
}

#[derive(Clone)]
struct FaultingDeletionBackend {
    state: Arc<Mutex<DeletionState>>,
}

impl FaultingDeletionBackend {
    fn new(fault: DeletionFault) -> Self {
        Self {
            state: Arc::new(Mutex::new(DeletionState {
                execution: CommunityDeletionExecutionRecord {
                    deletion: reversible_deletion(),
                    checkpoint: None,
                },
                fault,
                fault_fired: false,
                phase_effects: [0; COMMUNITY_DELETION_PHASES.len()],
                restore_effects: 0,
                completion_effects: 0,
            })),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, DeletionState> {
        self.state.lock().expect("deletion backend lock")
    }
}

#[async_trait]
impl CommunityDeletionExecutorBackend for FaultingDeletionBackend {
    async fn load_execution(
        &self,
        _tenant: &TenantContext,
        _deletion_id: AggregateId,
    ) -> Result<CommunityDeletionExecutionRecord, CommunityDeletionBackendError> {
        Ok(self.state().execution.clone())
    }

    async fn record_irreversible_boundary(
        &self,
        _tenant: &TenantContext,
        expected_deletion: &CommunityDeletion,
    ) -> Result<CommunityDeletionBoundaryCommit, CommunityDeletionBackendError> {
        let mut state = self.state();
        if &state.execution.deletion != expected_deletion || state.execution.checkpoint.is_some() {
            return Err(CommunityDeletionBackendError::StaleCheckpoint);
        }
        let mut deletion = state.execution.deletion.clone();
        deletion
            .enter_irreversible(
                deletion.fields().version,
                deletion_authority(4),
                DeletionFenceGeneration::new(7)
                    .ok_or(CommunityDeletionBackendError::InvalidData)?,
                deletion_digest(4),
            )
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        let checkpoint = CommunityDeletionCheckpoint::from_irreversible(&deletion)
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        state.execution = CommunityDeletionExecutionRecord {
            deletion: deletion.clone(),
            checkpoint: Some(checkpoint),
        };
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
        let current = state
            .execution
            .checkpoint
            .ok_or(CommunityDeletionBackendError::InvalidData)?;
        if current != attempt.checkpoint() {
            return Err(CommunityDeletionBackendError::StaleCheckpoint);
        }
        let index = phase_index(attempt.phase());
        state.phase_effects[index] += 1;
        if state.phase_effects[index] != 1 {
            return Err(CommunityDeletionBackendError::InvalidData);
        }
        let checkpoint = current
            .advance(
                attempt.phase(),
                deletion_digest(20 + u8::try_from(index).unwrap_or(0)),
            )
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        state.execution.checkpoint = Some(checkpoint);
        if matches!(state.fault, DeletionFault::AfterPhase(phase) if phase == attempt.phase())
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
        if &state.execution.deletion != expected_deletion
            || state.execution.checkpoint != Some(checkpoint)
        {
            return Err(CommunityDeletionBackendError::StaleCheckpoint);
        }
        let mut deletion = state.execution.deletion.clone();
        deletion
            .complete(
                deletion.fields().version,
                deletion_authority(50),
                checkpoint.evidence_digest(),
            )
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        state.execution.deletion = deletion.clone();
        state.completion_effects += 1;
        Ok(deletion)
    }
}

#[async_trait]
impl CommunityDeletionRecoveryBackend for FaultingDeletionBackend {
    async fn load_execution(
        &self,
        _tenant: &TenantContext,
        _deletion_id: AggregateId,
    ) -> Result<CommunityDeletionExecutionRecord, CommunityDeletionBackendError> {
        Ok(self.state().execution.clone())
    }

    async fn restore_pre_irreversible(
        &self,
        _tenant: &TenantContext,
        expected_deletion: &CommunityDeletion,
        authority: CommunityDeletionAuthorityEvidence,
    ) -> Result<CommunityDeletion, CommunityDeletionBackendError> {
        let mut state = self.state();
        if &state.execution.deletion != expected_deletion || state.execution.checkpoint.is_some() {
            return Err(CommunityDeletionBackendError::StaleCheckpoint);
        }
        let mut deletion = state.execution.deletion.clone();
        deletion
            .rollback(deletion.fields().version, authority, deletion_digest(60))
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        state.execution.deletion = deletion.clone();
        state.restore_effects += 1;
        if matches!(state.fault, DeletionFault::AfterRestore) && !state.fault_fired {
            state.fault_fired = true;
            return Err(CommunityDeletionBackendError::OutcomeUnknown);
        }
        Ok(deletion)
    }
}

#[test]
fn frozen_retention_versions_converge_after_before_and_after_commit_faults() {
    let document: MigrationFixtureDocument = serde_json::from_str(include_str!(
        "../../../.agents/specs/collaborative-workspace/fixtures/migrations/desktop-stores.json"
    ))
    .expect("migration fixture document");
    assert_eq!(document.format_version, 1);
    let fixtures = [
        (
            "retention-global-v0",
            "global-v0",
            RetentionFault::BeforeCommit,
        ),
        (
            "retention-scoped-v1",
            "relay-owner-scoped-v1",
            RetentionFault::AfterCommit,
        ),
    ];
    for (fixture_id, version, fault) in fixtures {
        let fixture = document
            .fixtures
            .iter()
            .find(|fixture| fixture.fixture_id == fixture_id)
            .expect("retention fixture");
        assert_eq!(fixture.data_source_id, "DESKTOP-RETENTION-001");
        assert_eq!(fixture.version, version);
        assert!(!fixture.contains_private_key_material);
        assert_eq!(fixture.records.len(), 2);
        assert!(!fixture.migration_state.is_empty());
        assert!(fixture.expected.is_object());

        let backend = FaultingRetentionBackend::new(retention_items(), fault);
        drive_retention(&backend);
        let state = backend.state();
        assert_eq!(
            state.effects.values().copied().collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(
            state.actions.get(&1),
            Some(&RetentionAuthorityAction::Delete(
                RetentionDeleteCause::Policy(RetentionReason::CommunityArchive,)
            ))
        );
        assert_eq!(
            state.actions.get(&2),
            Some(&RetentionAuthorityAction::Delete(
                RetentionDeleteCause::Policy(RetentionReason::CommunityPolicy,)
            ))
        );
        let checkpoint = state.checkpoint.as_ref().expect("retention checkpoint");
        assert_eq!(checkpoint.completed_sweeps(), 1);
        assert_eq!(checkpoint.counts().deleted, 2);
    }
}

#[test]
fn unknown_recovery_outcome_reloads_one_rolled_back_state() {
    let backend = FaultingDeletionBackend::new(DeletionFault::AfterRestore);
    let fixture = OperatorFixture::new();
    let restore_authorization =
        fixture.authorization(MembershipRole::Owner, AuthorizationAction::Delete);
    let error =
        futures::executor::block_on(CommunityDeletionOperatorApi::new(backend.clone()).execute(
            &restore_authorization,
            deletion_id(),
            CommunityDeletionOperatorCommand::Restore {
                expected_version: AggregateVersion::new(3).expect("version"),
                authority: deletion_authority(60),
            },
        ))
        .expect_err("unknown recovery outcome");
    assert_eq!(error, CommunityDeletionRecoveryError::OutcomeUnknown);

    let status_authorization =
        fixture.authorization(MembershipRole::Owner, AuthorizationAction::Read);
    let outcome =
        futures::executor::block_on(CommunityDeletionOperatorApi::new(backend.clone()).execute(
            &status_authorization,
            deletion_id(),
            CommunityDeletionOperatorCommand::Status,
        ))
        .expect("reload recovered status");
    let CommunityDeletionOperatorOutcome::Status(status) = outcome else {
        panic!("expected status");
    };
    assert_eq!(status.stage(), CommunityDeletionOperatorStage::RolledBack);
    assert_eq!(
        status.recovery_action(),
        CommunityDeletionRecoveryAction::None
    );
    let state = backend.state();
    assert_eq!(state.restore_effects, 1);
    assert_eq!(state.phase_effects, [0; COMMUNITY_DELETION_PHASES.len()]);
}

#[test]
fn every_irreversible_fault_reaches_one_deleted_state_independently() {
    for failed_phase in COMMUNITY_DELETION_PHASES {
        let backend = FaultingDeletionBackend::new(DeletionFault::AfterPhase(failed_phase));
        drive_deletion(&backend);
        let state = backend.state();
        assert_eq!(state.phase_effects, [1; COMMUNITY_DELETION_PHASES.len()]);
        assert_eq!(state.restore_effects, 0);
        assert_eq!(state.completion_effects, 1);
        assert_eq!(
            state.execution.deletion.state(),
            collaboration_domain::CommunityDeletionState::Completed(
                CommunityDeletionCompletion::Deleted,
            )
        );
    }
}

fn drive_retention(backend: &FaultingRetentionBackend) {
    let worker = RetentionWorker::new(backend.clone());
    let mut completed = false;
    for _ in 0..5 {
        match futures::executor::block_on(worker.run_batch(&tenant(), 1_000, 1)) {
            Ok(outcome) if outcome.completed_sweep() => {
                completed = true;
                break;
            }
            Ok(_) => {}
            Err(RetentionWorkerError::Backend(
                RetentionBackendError::Unavailable | RetentionBackendError::OutcomeUnknown,
            )) => {}
            Err(error) => panic!("unexpected retention failure: {error}"),
        }
    }
    assert!(completed, "retention did not converge");
}

fn drive_deletion(backend: &FaultingDeletionBackend) {
    let executor = CommunityDeletionExecutor::new(backend.clone());
    let mut completed = false;
    for _ in 0..10 {
        match futures::executor::block_on(executor.run_step(&tenant(), deletion_id())) {
            Ok(CommunityDeletionStepOutcome::Completed(CommunityDeletionCompletion::Deleted)) => {
                completed = true;
                break;
            }
            Ok(
                CommunityDeletionStepOutcome::BoundaryRecorded(_)
                | CommunityDeletionStepOutcome::PhaseCommitted { .. },
            ) => {}
            Ok(outcome) => panic!("unexpected deletion outcome: {outcome:?}"),
            Err(CommunityDeletionExecutorError::Backend(
                CommunityDeletionBackendError::OutcomeUnknown,
            )) => {}
            Err(error) => panic!("unexpected deletion failure: {error}"),
        }
    }
    assert!(completed, "deletion did not converge");
}

fn retention_items() -> Vec<RetentionWorkItem> {
    vec![
        retention_item(
            1,
            RetentionSnapshot::Current(RetentionArchiveSnapshot {
                archive: CommunityArchiveSnapshot {
                    community_id: community_id(),
                    state: CommunityArchivePolicyState::Archived,
                    version: AggregateVersion::FIRST,
                },
                archived_at_millis: Some(500),
            }),
        ),
        retention_item(2, RetentionSnapshot::Absent),
    ]
}

fn retention_item(
    sequence: u64,
    community_archive: RetentionSnapshot<RetentionArchiveSnapshot>,
) -> RetentionWorkItem {
    RetentionWorkItem::new(
        RetentionSourcePosition::new(sequence, [u8::try_from(sequence).expect("sequence"); 32])
            .expect("source position"),
        RetentionRecord {
            community_id: community_id(),
            record_id: AggregateId::from_uuid(Uuid::from_u128(500 + u128::from(sequence))),
            event_kind: RetentionEventKind::from_registry(1, RetentionPersistenceClass::Durable)
                .expect("event kind"),
            retention_started_at_millis: 100,
        },
        RetentionAuthoritySnapshot {
            policy: RetentionSnapshot::Current(retention_policy()),
            legal_hold: RetentionSnapshot::Absent,
            community_archive,
        },
    )
}

fn retention_policy() -> CommunityRetentionPolicy {
    CommunityRetentionPolicy::from_record(CommunityRetentionPolicyFields {
        community_id: community_id(),
        schema_version: RetentionPolicySchemaVersion::new(1).expect("schema version"),
        version: AggregateVersion::FIRST,
        default_ttl: Some(RetentionTtl::from_millis(100).expect("retention TTL")),
        archive_rule: ArchiveRetentionRule::DeleteOnArchive,
        kind_rules: Vec::new(),
    })
    .expect("retention policy")
}

fn reversible_deletion() -> CommunityDeletion {
    let mut deletion =
        CommunityDeletion::request(deletion_id(), community_id(), deletion_authority(1))
            .expect("deletion request");
    deletion
        .verify(
            AggregateVersion::FIRST,
            deletion_authority(2),
            deletion_digest(2),
        )
        .expect("verified deletion");
    deletion
        .enter_reversible(
            AggregateVersion::new(2).expect("version"),
            deletion_authority(3),
            DeletionFenceGeneration::new(7).expect("fence"),
        )
        .expect("reversible deletion");
    deletion
}

struct OperatorFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    scope: AuthorizationScope,
}

impl OperatorFixture {
    fn new() -> Self {
        let scope = AuthorizationScope::new("communities:manage").expect("scope");
        Self {
            tenant: tenant(),
            principal: AuthenticatedPrincipal::zed_account(
                principal_id(),
                community_id(),
                ServiceAccountId::new(2),
                PrincipalScopes::new([scope.clone()]).expect("scopes"),
            ),
            scope,
        }
    }

    fn authorization(
        &self,
        role: MembershipRole,
        action: AuthorizationAction,
    ) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(community_id().as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: community_id(),
                principal_id: principal_id(),
                role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 10_000,
        }
    }
}

fn deletion_authority(sequence: u64) -> CommunityDeletionAuthorityEvidence {
    CommunityDeletionAuthorityEvidence::new(
        CommunityArchiveSnapshot {
            community_id: community_id(),
            state: CommunityArchivePolicyState::Archived,
            version: AggregateVersion::new(sequence).expect("archive version"),
        },
        principal_id(),
        OperationId::from_uuid(Uuid::from_u128(700 + u128::from(sequence))),
        1_000 + sequence,
    )
    .expect("deletion authority")
}

fn deletion_digest(value: u8) -> DeletionEvidenceDigest {
    DeletionEvidenceDigest::new([value; 32]).expect("deletion digest")
}

fn tenant() -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id(), "retention-deletion-faults")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn community_id() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(1))
}

fn principal_id() -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(2))
}

fn deletion_id() -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(3))
}

fn phase_index(phase: CommunityDeletionPhase) -> usize {
    COMMUNITY_DELETION_PHASES
        .iter()
        .position(|candidate| *candidate == phase)
        .expect("known phase")
}
