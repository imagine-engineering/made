use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use collab::deletion::{
    executor::{
        COMMUNITY_DELETION_PHASES, CommunityDeletionBackendError, CommunityDeletionCheckpoint,
        CommunityDeletionExecutionRecord, CommunityDeletionPhase,
    },
    recovery::{
        CommunityDeletionOperatorApi, CommunityDeletionOperatorCommand,
        CommunityDeletionOperatorHaltReason, CommunityDeletionOperatorOutcome,
        CommunityDeletionOperatorStage, CommunityDeletionRecoveryAction,
        CommunityDeletionRecoveryBackend, CommunityDeletionRecoveryError,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityArchivePolicyState, CommunityArchiveSnapshot, CommunityDeletion,
    CommunityDeletionAuthorityEvidence, CommunityDeletionFailureReason, CommunityId,
    CommunityMembership, DeletionEvidenceDigest, DeletionFenceGeneration, MembershipRole,
    MembershipStatus, OperationId, PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext,
    TrustedTenantRoute,
};
use uuid::Uuid;

struct BackendState {
    execution: CommunityDeletionExecutionRecord,
    load_calls: u32,
    restore_calls: u32,
}

#[derive(Clone)]
struct TestBackend {
    state: Arc<Mutex<BackendState>>,
}

impl TestBackend {
    fn new(deletion: CommunityDeletion, checkpoint: Option<CommunityDeletionCheckpoint>) -> Self {
        Self {
            state: Arc::new(Mutex::new(BackendState {
                execution: CommunityDeletionExecutionRecord {
                    deletion,
                    checkpoint,
                },
                load_calls: 0,
                restore_calls: 0,
            })),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, BackendState> {
        self.state.lock().expect("backend lock")
    }
}

#[async_trait]
impl CommunityDeletionRecoveryBackend for TestBackend {
    async fn load_execution(
        &self,
        _tenant: &TenantContext,
        _deletion_id: AggregateId,
    ) -> Result<CommunityDeletionExecutionRecord, CommunityDeletionBackendError> {
        let mut state = self.state();
        state.load_calls += 1;
        Ok(state.execution.clone())
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
        let mut restored = state.execution.deletion.clone();
        restored
            .rollback(restored.fields().version, authority, digest(90))
            .map_err(|_| CommunityDeletionBackendError::InvalidData)?;
        state.execution.deletion = restored.clone();
        state.restore_calls += 1;
        Ok(restored)
    }
}

#[test]
fn restore_completes_from_every_reversible_state() {
    let deletions = [
        verified_deletion(),
        reversible_deletion(),
        failed_verified_deletion(),
        failed_reversible_deletion(),
    ];
    for deletion in deletions {
        let backend = TestBackend::new(deletion.clone(), None);
        let fixture = Fixture::new();
        let authorization =
            fixture.authorization(MembershipRole::Owner, AuthorizationAction::Delete);
        let authority = authority(deletion.fields().version.get() + 10);
        let outcome = futures::executor::block_on(
            CommunityDeletionOperatorApi::new(backend.clone()).execute(
                &authorization,
                deletion_id(),
                CommunityDeletionOperatorCommand::Restore {
                    expected_version: deletion.fields().version,
                    authority,
                },
            ),
        )
        .expect("restore reversible deletion");
        let CommunityDeletionOperatorOutcome::Restored(status) = outcome else {
            panic!("expected restored status");
        };
        assert_eq!(status.stage(), CommunityDeletionOperatorStage::RolledBack);
        assert_eq!(
            status.recovery_action(),
            CommunityDeletionRecoveryAction::None
        );
        assert_eq!(backend.state().restore_calls, 1);
    }
}

#[test]
fn restore_refuses_every_checkpoint_at_or_beyond_boundary() {
    for completed_phases in 0..=COMMUNITY_DELETION_PHASES.len() {
        let deletion = irreversible_deletion();
        let mut checkpoint =
            CommunityDeletionCheckpoint::from_irreversible(&deletion).expect("checkpoint");
        for (index, phase) in COMMUNITY_DELETION_PHASES
            .iter()
            .take(completed_phases)
            .enumerate()
        {
            checkpoint = checkpoint
                .advance(
                    *phase,
                    digest(20 + u8::try_from(index).expect("phase index")),
                )
                .expect("advance checkpoint");
        }
        let backend = TestBackend::new(deletion.clone(), Some(checkpoint));
        let fixture = Fixture::new();
        let authorization =
            fixture.authorization(MembershipRole::Owner, AuthorizationAction::Delete);
        let error = futures::executor::block_on(
            CommunityDeletionOperatorApi::new(backend.clone()).execute(
                &authorization,
                deletion_id(),
                CommunityDeletionOperatorCommand::Restore {
                    expected_version: deletion.fields().version,
                    authority: authority(20),
                },
            ),
        )
        .expect_err("irreversible recovery must fail");
        assert_eq!(error, CommunityDeletionRecoveryError::IrreversibleBoundary);
        assert_eq!(backend.state().restore_calls, 0);
    }
}

#[test]
fn status_exposes_redacted_progress_halt_and_recovery() {
    let deletion = irreversible_deletion();
    let mut checkpoint =
        CommunityDeletionCheckpoint::from_irreversible(&deletion).expect("checkpoint");
    for (index, phase) in COMMUNITY_DELETION_PHASES.iter().take(3).enumerate() {
        checkpoint = checkpoint
            .advance(
                *phase,
                digest(30 + u8::try_from(index).expect("phase index")),
            )
            .expect("advance checkpoint");
    }
    let backend = TestBackend::new(deletion, Some(checkpoint));
    let fixture = Fixture::new();
    let authorization = fixture.authorization(MembershipRole::Admin, AuthorizationAction::Read);
    let outcome = futures::executor::block_on(CommunityDeletionOperatorApi::new(backend).execute(
        &authorization,
        deletion_id(),
        CommunityDeletionOperatorCommand::Status,
    ))
    .expect("operator status");
    let CommunityDeletionOperatorOutcome::Status(status) = outcome else {
        panic!("expected status");
    };
    assert_eq!(status.stage(), CommunityDeletionOperatorStage::Irreversible);
    assert_eq!(
        status.last_trustworthy_stage(),
        CommunityDeletionOperatorStage::Irreversible
    );
    assert_eq!(status.completed_phases(), 3);
    assert_eq!(status.total_phases(), 6);
    assert_eq!(status.next_phase(), Some(CommunityDeletionPhase::Push));
    assert_eq!(status.checkpoint_version(), Some(4));
    assert_eq!(status.halt_reason(), None);
    assert_eq!(
        status.recovery_action(),
        CommunityDeletionRecoveryAction::Resume
    );

    let failed = failed_reversible_deletion_with(CommunityDeletionFailureReason::InventoryMismatch);
    let backend = TestBackend::new(failed, None);
    let outcome = futures::executor::block_on(CommunityDeletionOperatorApi::new(backend).execute(
        &authorization,
        deletion_id(),
        CommunityDeletionOperatorCommand::Status,
    ))
    .expect("failed status");
    let CommunityDeletionOperatorOutcome::Status(status) = outcome else {
        panic!("expected failed status");
    };
    assert_eq!(status.stage(), CommunityDeletionOperatorStage::Failed);
    assert_eq!(
        status.last_trustworthy_stage(),
        CommunityDeletionOperatorStage::Reversible
    );
    assert_eq!(
        status.halt_reason(),
        Some(CommunityDeletionOperatorHaltReason::InventoryMismatch)
    );
    assert_eq!(
        status.recovery_action(),
        CommunityDeletionRecoveryAction::Restore
    );
    let debug = format!("{status:?}");
    assert!(!debug.contains(&community_id().to_string()));
    assert!(!debug.contains(&deletion_id().to_string()));
}

#[test]
fn authorization_precedes_lookup_and_separates_status_from_restore() {
    let backend = TestBackend::new(verified_deletion(), None);
    let fixture = Fixture::new();
    let member_status = fixture.authorization(MembershipRole::Member, AuthorizationAction::Read);
    let error =
        futures::executor::block_on(CommunityDeletionOperatorApi::new(backend.clone()).execute(
            &member_status,
            deletion_id(),
            CommunityDeletionOperatorCommand::Status,
        ))
        .expect_err("member status must fail");
    assert_eq!(error, CommunityDeletionRecoveryError::AuthorizationDenied);
    assert_eq!(backend.state().load_calls, 0);

    let admin_restore = fixture.authorization(MembershipRole::Admin, AuthorizationAction::Delete);
    let error =
        futures::executor::block_on(CommunityDeletionOperatorApi::new(backend.clone()).execute(
            &admin_restore,
            deletion_id(),
            CommunityDeletionOperatorCommand::Restore {
                expected_version: AggregateVersion::new(2).expect("version"),
                authority: authority(20),
            },
        ))
        .expect_err("admin restore must fail");
    assert_eq!(error, CommunityDeletionRecoveryError::AuthorizationDenied);
    assert_eq!(backend.state().load_calls, 0);

    let admin_status = fixture.authorization(MembershipRole::Admin, AuthorizationAction::Read);
    futures::executor::block_on(CommunityDeletionOperatorApi::new(backend.clone()).execute(
        &admin_status,
        deletion_id(),
        CommunityDeletionOperatorCommand::Status,
    ))
    .expect("admin may read status");
    assert_eq!(backend.state().load_calls, 1);
}

struct Fixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    scope: AuthorizationScope,
}

impl Fixture {
    fn new() -> Self {
        let scope = AuthorizationScope::new("communities:manage").expect("scope");
        let tenant = TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id(), "deletion-recovery-test")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant");
        let principal = AuthenticatedPrincipal::zed_account(
            principal_id(),
            community_id(),
            ServiceAccountId::new(2),
            PrincipalScopes::new([scope.clone()]).expect("scopes"),
        );
        Self {
            tenant,
            principal,
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
            now_millis: 50_000,
        }
    }
}

fn requested_deletion() -> CommunityDeletion {
    CommunityDeletion::request(deletion_id(), community_id(), authority(1))
        .expect("deletion request")
}

fn verified_deletion() -> CommunityDeletion {
    let mut deletion = requested_deletion();
    deletion
        .verify(AggregateVersion::FIRST, authority(2), digest(2))
        .expect("verified deletion");
    deletion
}

fn reversible_deletion() -> CommunityDeletion {
    let mut deletion = verified_deletion();
    deletion
        .enter_reversible(
            AggregateVersion::new(2).expect("version"),
            authority(3),
            DeletionFenceGeneration::new(7).expect("fence"),
        )
        .expect("reversible deletion");
    deletion
}

fn irreversible_deletion() -> CommunityDeletion {
    let mut deletion = reversible_deletion();
    deletion
        .enter_irreversible(
            AggregateVersion::new(3).expect("version"),
            authority(4),
            DeletionFenceGeneration::new(7).expect("fence"),
            digest(4),
        )
        .expect("irreversible deletion");
    deletion
}

fn failed_verified_deletion() -> CommunityDeletion {
    let mut deletion = verified_deletion();
    deletion
        .fail(
            AggregateVersion::new(2).expect("version"),
            authority(3),
            CommunityDeletionFailureReason::DependencyUnavailable,
        )
        .expect("failed verified deletion");
    deletion
}

fn failed_reversible_deletion() -> CommunityDeletion {
    failed_reversible_deletion_with(CommunityDeletionFailureReason::DependencyUnavailable)
}

fn failed_reversible_deletion_with(reason: CommunityDeletionFailureReason) -> CommunityDeletion {
    let mut deletion = reversible_deletion();
    deletion
        .fail(
            AggregateVersion::new(3).expect("version"),
            authority(4),
            reason,
        )
        .expect("failed reversible deletion");
    deletion
}

fn authority(sequence: u64) -> CommunityDeletionAuthorityEvidence {
    CommunityDeletionAuthorityEvidence::new(
        CommunityArchiveSnapshot {
            community_id: community_id(),
            state: CommunityArchivePolicyState::Archived,
            version: AggregateVersion::new(sequence).expect("archive version"),
        },
        principal_id(),
        OperationId::from_uuid(Uuid::from_u128(100 + u128::from(sequence))),
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

fn principal_id() -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(2))
}

fn deletion_id() -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(3))
}
