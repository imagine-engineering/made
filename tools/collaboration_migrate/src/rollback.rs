use std::error::Error;

use async_trait::async_trait;
use collab::migration::{
    cutover_checkpoint::{
        CutoverAuthority, CutoverCheckpoint, CutoverPhase, CutoverReversibleBoundary,
    },
    divergence::{DivergenceOperatorDiagnostic, DivergenceRecoveryAction},
};
use collaboration_domain::{OperationId, ScopedAggregateId, TenantContext};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RollbackPlan {
    aggregate: ScopedAggregateId,
    operation_id: OperationId,
    approval_hash: [u8; 32],
    checkpoint_version: collaboration_domain::AggregateVersion,
    boundary: CutoverReversibleBoundary,
    halt_operation_id: OperationId,
    plan_hash: [u8; 32],
}

impl RollbackPlan {
    pub fn new(
        tenant: &TenantContext,
        checkpoint: &CutoverCheckpoint,
        diagnostic: DivergenceOperatorDiagnostic,
        operation_id: OperationId,
        approval_hash: [u8; 32],
    ) -> Result<Self, RollbackError> {
        if tenant.community_id() != checkpoint.aggregate().community_id()
            || diagnostic.aggregate() != checkpoint.aggregate()
        {
            return Err(RollbackError::TenantBoundaryViolation);
        }
        if operation_id.as_uuid().is_nil() || approval_hash == [0; 32] {
            return Err(RollbackError::InvalidInput);
        }
        let last_trustworthy = diagnostic.last_trustworthy();
        if checkpoint.authority() != CutoverAuthority::Canonical
            || checkpoint.phase() >= CutoverPhase::Retirement
            || last_trustworthy.checkpoint_version() != checkpoint.version()
            || last_trustworthy.phase() != checkpoint.phase()
            || last_trustworthy.authority() != CutoverAuthority::Canonical
            || !last_trustworthy.has_reversible_boundary()
            || diagnostic.rollback().aggregate() != checkpoint.aggregate()
            || diagnostic.rollback().actions() != canonical_rollback_actions()
        {
            return Err(RollbackError::UnsafeBoundary);
        }
        let boundary = checkpoint
            .last_reversible_boundary()
            .ok_or(RollbackError::UnsafeBoundary)?
            .clone();
        if boundary.authority() != CutoverAuthority::Legacy
            || boundary.checkpoint_version() >= checkpoint.version()
            || boundary.phase() > checkpoint.phase()
        {
            return Err(RollbackError::UnsafeBoundary);
        }
        let plan_hash = hash_plan(
            checkpoint.aggregate(),
            operation_id,
            approval_hash,
            checkpoint.version(),
            &boundary,
            diagnostic.operation_id(),
        );
        Ok(Self {
            aggregate: checkpoint.aggregate(),
            operation_id,
            approval_hash,
            checkpoint_version: checkpoint.version(),
            boundary,
            halt_operation_id: diagnostic.operation_id(),
            plan_hash,
        })
    }

    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn approval_hash(&self) -> [u8; 32] {
        self.approval_hash
    }

    pub const fn checkpoint_version(&self) -> collaboration_domain::AggregateVersion {
        self.checkpoint_version
    }

    pub fn boundary(&self) -> &CutoverReversibleBoundary {
        &self.boundary
    }

    pub const fn halt_operation_id(&self) -> OperationId {
        self.halt_operation_id
    }

    pub const fn plan_hash(&self) -> [u8; 32] {
        self.plan_hash
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RollbackStage {
    Requested,
    WritesQuiesced,
    OutboxDrained,
    Verified,
    RoutingRestored,
    Completed,
}

impl RollbackStage {
    const fn next(self) -> Option<Self> {
        match self {
            Self::Requested => Some(Self::WritesQuiesced),
            Self::WritesQuiesced => Some(Self::OutboxDrained),
            Self::OutboxDrained => Some(Self::Verified),
            Self::Verified => Some(Self::RoutingRestored),
            Self::RoutingRestored => Some(Self::Completed),
            Self::Completed => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RollbackProgressFields {
    pub plan_hash: [u8; 32],
    pub stage: RollbackStage,
    pub write_fence_generation: Option<u64>,
    pub drained_outbox_sequence: Option<u64>,
    pub pending_outbox_count: Option<u64>,
    pub verification_hash: Option<[u8; 32]>,
    pub unexplained_divergence_count: Option<u64>,
    pub target_only_mutation_count: Option<u64>,
    pub restored_authority: Option<CutoverAuthority>,
    pub routing_generation: Option<u64>,
    pub completed_at_millis: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RollbackProgress {
    fields: RollbackProgressFields,
}

impl RollbackProgress {
    pub fn requested(plan_hash: [u8; 32]) -> Result<Self, RollbackError> {
        Self::from_record(RollbackProgressFields {
            plan_hash,
            stage: RollbackStage::Requested,
            write_fence_generation: None,
            drained_outbox_sequence: None,
            pending_outbox_count: None,
            verification_hash: None,
            unexplained_divergence_count: None,
            target_only_mutation_count: None,
            restored_authority: None,
            routing_generation: None,
            completed_at_millis: None,
        })
    }

    pub fn from_record(fields: RollbackProgressFields) -> Result<Self, RollbackError> {
        let progress = Self { fields };
        if !progress.valid() {
            return Err(RollbackError::InvalidBackendResponse);
        }
        Ok(progress)
    }

    pub const fn record(self) -> RollbackProgressFields {
        self.fields
    }

    pub const fn plan_hash(self) -> [u8; 32] {
        self.fields.plan_hash
    }

    pub const fn stage(self) -> RollbackStage {
        self.fields.stage
    }

    pub const fn restored_authority(self) -> Option<CutoverAuthority> {
        self.fields.restored_authority
    }

    pub const fn routing_generation(self) -> Option<u64> {
        self.fields.routing_generation
    }

    fn follows(self, previous: Self) -> bool {
        self.fields.plan_hash == previous.fields.plan_hash
            && previous.fields.stage.next() == Some(self.fields.stage)
            && retains_prior_evidence(previous.fields, self.fields)
            && self.valid()
    }

    fn valid(self) -> bool {
        if self.fields.plan_hash == [0; 32] {
            return false;
        }
        let quiesced = self
            .fields
            .write_fence_generation
            .is_some_and(|value| value > 0);
        let drain_absent = self.fields.drained_outbox_sequence.is_none()
            && self.fields.pending_outbox_count.is_none();
        let drained = self.fields.drained_outbox_sequence.is_some()
            && self.fields.pending_outbox_count == Some(0);
        let verification_absent = self.fields.verification_hash.is_none()
            && self.fields.unexplained_divergence_count.is_none()
            && self.fields.target_only_mutation_count.is_none();
        let verified = self
            .fields
            .verification_hash
            .is_some_and(|hash| hash != [0; 32])
            && self.fields.unexplained_divergence_count == Some(0)
            && self.fields.target_only_mutation_count == Some(0);
        let restoration_absent =
            self.fields.restored_authority.is_none() && self.fields.routing_generation.is_none();
        let restored = self.fields.restored_authority == Some(CutoverAuthority::Legacy)
            && self
                .fields
                .routing_generation
                .is_some_and(|value| value > 0);
        let completion_absent = self.fields.completed_at_millis.is_none();
        let completed = self
            .fields
            .completed_at_millis
            .is_some_and(|value| value > 0);
        match self.fields.stage {
            RollbackStage::Requested => {
                self.fields.write_fence_generation.is_none()
                    && drain_absent
                    && verification_absent
                    && restoration_absent
                    && completion_absent
            }
            RollbackStage::WritesQuiesced => {
                quiesced
                    && drain_absent
                    && verification_absent
                    && restoration_absent
                    && completion_absent
            }
            RollbackStage::OutboxDrained => {
                quiesced
                    && drained
                    && verification_absent
                    && restoration_absent
                    && completion_absent
            }
            RollbackStage::Verified => {
                quiesced && drained && verified && restoration_absent && completion_absent
            }
            RollbackStage::RoutingRestored => {
                quiesced && drained && verified && restored && completion_absent
            }
            RollbackStage::Completed => quiesced && drained && verified && restored && completed,
        }
    }
}

fn retains_prior_evidence(previous: RollbackProgressFields, next: RollbackProgressFields) -> bool {
    (previous.write_fence_generation.is_none()
        || previous.write_fence_generation == next.write_fence_generation)
        && (previous.drained_outbox_sequence.is_none()
            || previous.drained_outbox_sequence == next.drained_outbox_sequence)
        && (previous.pending_outbox_count.is_none()
            || previous.pending_outbox_count == next.pending_outbox_count)
        && (previous.verification_hash.is_none()
            || previous.verification_hash == next.verification_hash)
        && (previous.unexplained_divergence_count.is_none()
            || previous.unexplained_divergence_count == next.unexplained_divergence_count)
        && (previous.target_only_mutation_count.is_none()
            || previous.target_only_mutation_count == next.target_only_mutation_count)
        && (previous.restored_authority.is_none()
            || previous.restored_authority == next.restored_authority)
        && (previous.routing_generation.is_none()
            || previous.routing_generation == next.routing_generation)
        && (previous.completed_at_millis.is_none()
            || previous.completed_at_millis == next.completed_at_millis)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RollbackBackendError {
    Unavailable,
    OutcomeUnknown,
    Conflict,
    InvalidData,
}

impl std::fmt::Display for RollbackBackendError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "rollback backend is unavailable",
            Self::OutcomeUnknown => "rollback backend outcome is unknown",
            Self::Conflict => "rollback backend rejected a conflicting operation",
            Self::InvalidData => "rollback backend returned invalid data",
        })
    }
}

impl Error for RollbackBackendError {}

#[async_trait]
pub trait RollbackBackend: Send + Sync {
    async fn load_or_create(
        &self,
        tenant: &TenantContext,
        plan: &RollbackPlan,
    ) -> Result<RollbackProgress, RollbackBackendError>;

    /// Atomically performs exactly the next stage and persists its evidence. Implementations map
    /// Requested→quiesce, WritesQuiesced→drain, OutboxDrained→verify,
    /// Verified→restore the recorded boundary, and RoutingRestored→complete.
    async fn advance(
        &self,
        tenant: &TenantContext,
        plan: &RollbackPlan,
        current: RollbackProgress,
    ) -> Result<RollbackProgress, RollbackBackendError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RollbackReceipt {
    aggregate: ScopedAggregateId,
    operation_id: OperationId,
    plan_hash: [u8; 32],
    restored_authority: CutoverAuthority,
    routing_generation: u64,
    completed_at_millis: u64,
}

impl RollbackReceipt {
    pub const fn aggregate(self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn operation_id(self) -> OperationId {
        self.operation_id
    }

    pub const fn plan_hash(self) -> [u8; 32] {
        self.plan_hash
    }

    pub const fn restored_authority(self) -> CutoverAuthority {
        self.restored_authority
    }

    pub const fn routing_generation(self) -> u64 {
        self.routing_generation
    }

    pub const fn completed_at_millis(self) -> u64 {
        self.completed_at_millis
    }
}

pub struct RollbackCoordinator<Backend> {
    backend: Backend,
}

impl<Backend> RollbackCoordinator<Backend>
where
    Backend: RollbackBackend,
{
    pub const fn new(backend: Backend) -> Self {
        Self { backend }
    }

    pub fn into_backend(self) -> Backend {
        self.backend
    }

    pub async fn execute(
        &self,
        tenant: &TenantContext,
        plan: &RollbackPlan,
    ) -> Result<RollbackReceipt, RollbackError> {
        if tenant.community_id() != plan.aggregate.community_id() {
            return Err(RollbackError::TenantBoundaryViolation);
        }
        let mut progress = self
            .backend
            .load_or_create(tenant, plan)
            .await
            .map_err(RollbackError::from_backend)?;
        validate_progress(plan, progress)?;
        while progress.stage() != RollbackStage::Completed {
            let next = self
                .backend
                .advance(tenant, plan, progress)
                .await
                .map_err(RollbackError::from_backend)?;
            validate_progress(plan, next)?;
            if !next.follows(progress) {
                return Err(RollbackError::InvalidBackendResponse);
            }
            progress = next;
        }
        let fields = progress.record();
        Ok(RollbackReceipt {
            aggregate: plan.aggregate,
            operation_id: plan.operation_id,
            plan_hash: plan.plan_hash,
            restored_authority: fields
                .restored_authority
                .ok_or(RollbackError::InvalidBackendResponse)?,
            routing_generation: fields
                .routing_generation
                .ok_or(RollbackError::InvalidBackendResponse)?,
            completed_at_millis: fields
                .completed_at_millis
                .ok_or(RollbackError::InvalidBackendResponse)?,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RollbackError {
    #[error("rollback input is invalid")]
    InvalidInput,
    #[error("rollback crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("rollback is unsafe past the recorded boundary")]
    UnsafeBoundary,
    #[error("rollback backend is unavailable")]
    BackendUnavailable,
    #[error("rollback outcome is unknown")]
    OutcomeUnknown,
    #[error("rollback operation conflicts with durable state")]
    Conflict,
    #[error("rollback backend returned invalid data")]
    InvalidBackendResponse,
}

impl RollbackError {
    const fn from_backend(error: RollbackBackendError) -> Self {
        match error {
            RollbackBackendError::Unavailable => Self::BackendUnavailable,
            RollbackBackendError::OutcomeUnknown => Self::OutcomeUnknown,
            RollbackBackendError::Conflict => Self::Conflict,
            RollbackBackendError::InvalidData => Self::InvalidBackendResponse,
        }
    }
}

fn validate_progress(plan: &RollbackPlan, progress: RollbackProgress) -> Result<(), RollbackError> {
    if progress.plan_hash() != plan.plan_hash || !progress.valid() {
        return Err(RollbackError::InvalidBackendResponse);
    }
    Ok(())
}

fn canonical_rollback_actions() -> [DivergenceRecoveryAction; 4] {
    [
        DivergenceRecoveryAction::QuiesceCanonicalWrites,
        DivergenceRecoveryAction::DrainCanonicalOutbox,
        DivergenceRecoveryAction::VerifyReversibleBoundary,
        DivergenceRecoveryAction::RestorePriorAuthority,
    ]
}

fn hash_plan(
    aggregate: ScopedAggregateId,
    operation_id: OperationId,
    approval_hash: [u8; 32],
    checkpoint_version: collaboration_domain::AggregateVersion,
    boundary: &CutoverReversibleBoundary,
    halt_operation_id: OperationId,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(aggregate.community_id().as_uuid().as_bytes());
    hasher.update(aggregate.aggregate_id().as_uuid().as_bytes());
    hasher.update(aggregate_type_name(aggregate.aggregate_type()));
    hasher.update(operation_id.as_uuid().as_bytes());
    hasher.update(approval_hash);
    hasher.update(checkpoint_version.get().to_be_bytes());
    hasher.update(boundary.label().as_bytes());
    hasher.update(boundary.checkpoint_version().get().to_be_bytes());
    hasher.update(phase_name(boundary.phase()));
    hasher.update(authority_name(boundary.authority()));
    hasher.update(boundary.source_cursor().sequence().to_be_bytes());
    hasher.update(boundary.source_cursor().token_hash());
    hasher.update(boundary.target_cursor().sequence().to_be_bytes());
    hasher.update(boundary.target_cursor().token_hash());
    hasher.update(halt_operation_id.as_uuid().as_bytes());
    hasher.finalize().into()
}

fn aggregate_type_name(aggregate_type: collaboration_domain::AggregateType) -> &'static [u8] {
    use collaboration_domain::AggregateType;

    match aggregate_type {
        AggregateType::Community => b"community",
        AggregateType::Project => b"project",
        AggregateType::Conversation => b"conversation",
        AggregateType::AgentSession => b"agent_session",
        AggregateType::Activity => b"activity",
        AggregateType::GitChange => b"git_change",
        AggregateType::Workflow => b"workflow",
        AggregateType::Identity => b"identity",
        AggregateType::Presence => b"presence",
    }
}

fn phase_name(phase: CutoverPhase) -> &'static [u8] {
    match phase {
        CutoverPhase::Baseline => b"baseline",
        CutoverPhase::NativePresentation => b"native_presentation",
        CutoverPhase::Foundations => b"foundations",
        CutoverPhase::CommunicationReadShadow => b"communication_read_shadow",
        CutoverPhase::CommunicationWriteCutover => b"communication_write_cutover",
        CutoverPhase::ProjectGitAgentIntegration => b"project_git_agent_integration",
        CutoverPhase::WorkflowInfrastructureCutover => b"workflow_infrastructure_cutover",
        CutoverPhase::ClientDeploymentMigration => b"client_deployment_migration",
        CutoverPhase::Retirement => b"retirement",
    }
}

fn authority_name(authority: CutoverAuthority) -> &'static [u8] {
    match authority {
        CutoverAuthority::Legacy => b"legacy",
        CutoverAuthority::Canonical => b"canonical",
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Mutex};

    use async_trait::async_trait;
    use collab::{
        audit::events::SecurityAuditEvent,
        migration::{
            cutover_checkpoint::{
                CutoverCursor, CutoverGateEvidence, CutoverIntegrity, CutoverTransition,
                CutoverTransitionOutcome,
            },
            divergence::{
                DivergenceHalt, DivergenceHaltStore, DivergenceSignal, DivergenceStopCoordinator,
                DivergenceStopOutcome, DivergenceStoreError, DivergenceStoreOutcome,
            },
        },
    };
    use collaboration_domain::{
        AggregateId, AggregateType, AuthenticatedPrincipal, CommunityId, PrincipalId,
        PrincipalScopes, TrustedTenantRoute,
    };
    use uuid::Uuid;

    use super::*;

    fn hash(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate(community_id: CommunityId) -> ScopedAggregateId {
        ScopedAggregateId::new(
            community_id,
            AggregateType::Conversation,
            AggregateId::from_uuid(Uuid::from_u128(100)),
        )
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "rollback-test")
                    .expect("trusted tenant route"),
            ),
            &[],
        )
        .expect("tenant context")
    }

    fn actor(community_id: CommunityId) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::service(
            PrincipalId::from_uuid(Uuid::from_u128(900)),
            community_id,
            "rollback-test",
            PrincipalScopes::default(),
        )
        .expect("service actor")
    }

    fn canonical_checkpoint(community_id: CommunityId) -> CutoverCheckpoint {
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community_id),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");
        let cursor = CutoverCursor::new(5, hash(5)).expect("cursor");
        let integrity = CutoverIntegrity::new(Some(hash(6)), Some(hash(6))).expect("integrity");
        let gates =
            CutoverGateEvidence::new(Some(hash(7)), Some(hash(8)), Some(hash(9)), Some(hash(10)))
                .expect("gates");
        let transition = CutoverTransition {
            operation_id: OperationId::from_uuid(Uuid::from_u128(500)),
            expected_version: checkpoint.version(),
            phase: CutoverPhase::CommunicationWriteCutover,
            authority: CutoverAuthority::Canonical,
            source_cursor: cursor,
            target_cursor: cursor,
            integrity,
            gates,
            reversible_boundary_label: Some("before-rollback".to_string()),
        };
        let CutoverTransitionOutcome::Advanced(checkpoint) = checkpoint
            .transition(&transition)
            .expect("canonical transition")
        else {
            panic!("authority must advance");
        };
        checkpoint
    }

    #[derive(Default)]
    struct HaltStore;

    #[async_trait]
    impl DivergenceHaltStore for HaltStore {
        async fn halt_and_record(
            &self,
            _tenant: &TenantContext,
            _halt: DivergenceHalt,
            _audit_event: SecurityAuditEvent,
        ) -> Result<DivergenceStoreOutcome, DivergenceStoreError> {
            Ok(DivergenceStoreOutcome::Halted)
        }
    }

    async fn diagnostic(
        tenant: &TenantContext,
        checkpoint: &CutoverCheckpoint,
    ) -> DivergenceOperatorDiagnostic {
        let coordinator = DivergenceStopCoordinator::new(HaltStore);
        let DivergenceStopOutcome::Halted(diagnostic) = coordinator
            .halt(
                tenant,
                checkpoint,
                DivergenceSignal::signature(hash(20)).expect("signal"),
                OperationId::from_uuid(Uuid::from_u128(700)),
                &actor(tenant.community_id()),
                1_900_000_000_000,
            )
            .await
            .expect("halt")
        else {
            panic!("first signal must halt");
        };
        diagnostic
    }

    #[derive(Default)]
    struct TestBackendState {
        progress: BTreeMap<[u8; 32], RollbackProgress>,
        unknown_once_after_quiesce: bool,
    }

    #[derive(Default)]
    struct TestBackend {
        state: Mutex<TestBackendState>,
    }

    impl TestBackend {
        fn unknown_once() -> Self {
            Self {
                state: Mutex::new(TestBackendState {
                    unknown_once_after_quiesce: true,
                    ..TestBackendState::default()
                }),
            }
        }

        fn progress(&self, hash: [u8; 32]) -> Option<RollbackProgress> {
            self.state
                .lock()
                .expect("backend lock")
                .progress
                .get(&hash)
                .copied()
        }
    }

    #[async_trait]
    impl RollbackBackend for TestBackend {
        async fn load_or_create(
            &self,
            _tenant: &TenantContext,
            plan: &RollbackPlan,
        ) -> Result<RollbackProgress, RollbackBackendError> {
            let mut state = self
                .state
                .lock()
                .map_err(|_| RollbackBackendError::Unavailable)?;
            if let Some(progress) = state.progress.get(&plan.plan_hash()) {
                return Ok(*progress);
            }
            let progress = RollbackProgress::requested(plan.plan_hash())
                .map_err(|_| RollbackBackendError::InvalidData)?;
            state.progress.insert(plan.plan_hash(), progress);
            Ok(progress)
        }

        async fn advance(
            &self,
            _tenant: &TenantContext,
            plan: &RollbackPlan,
            current: RollbackProgress,
        ) -> Result<RollbackProgress, RollbackBackendError> {
            let mut state = self
                .state
                .lock()
                .map_err(|_| RollbackBackendError::Unavailable)?;
            if state.progress.get(&plan.plan_hash()) != Some(&current) {
                return Err(RollbackBackendError::Conflict);
            }
            let mut fields = current.record();
            match current.stage() {
                RollbackStage::Requested => {
                    fields.stage = RollbackStage::WritesQuiesced;
                    fields.write_fence_generation = Some(1);
                }
                RollbackStage::WritesQuiesced => {
                    fields.stage = RollbackStage::OutboxDrained;
                    fields.drained_outbox_sequence = Some(20);
                    fields.pending_outbox_count = Some(0);
                }
                RollbackStage::OutboxDrained => {
                    fields.stage = RollbackStage::Verified;
                    fields.verification_hash = Some(hash(30));
                    fields.unexplained_divergence_count = Some(0);
                    fields.target_only_mutation_count = Some(0);
                }
                RollbackStage::Verified => {
                    fields.stage = RollbackStage::RoutingRestored;
                    fields.restored_authority = Some(plan.boundary().authority());
                    fields.routing_generation = Some(2);
                }
                RollbackStage::RoutingRestored => {
                    fields.stage = RollbackStage::Completed;
                    fields.completed_at_millis = Some(1_900_000_000_100);
                }
                RollbackStage::Completed => return Err(RollbackBackendError::Conflict),
            }
            let next = RollbackProgress::from_record(fields)
                .map_err(|_| RollbackBackendError::InvalidData)?;
            state.progress.insert(plan.plan_hash(), next);
            if current.stage() == RollbackStage::Requested && state.unknown_once_after_quiesce {
                state.unknown_once_after_quiesce = false;
                return Err(RollbackBackendError::OutcomeUnknown);
            }
            Ok(next)
        }
    }

    #[tokio::test]
    async fn isolated_rollback_restores_recorded_prior_authority() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id);
        let diagnostic = diagnostic(&tenant, &checkpoint).await;
        let plan = RollbackPlan::new(
            &tenant,
            &checkpoint,
            diagnostic,
            OperationId::from_uuid(Uuid::from_u128(800)),
            hash(40),
        )
        .expect("rollback plan");
        let coordinator = RollbackCoordinator::new(TestBackend::default());
        let receipt = coordinator.execute(&tenant, &plan).await.expect("rollback");
        assert_eq!(receipt.aggregate(), aggregate(community_id));
        assert_eq!(receipt.restored_authority(), CutoverAuthority::Legacy);
        assert_eq!(receipt.routing_generation(), 2);
        assert_eq!(
            coordinator
                .backend
                .progress(plan.plan_hash())
                .map(RollbackProgress::stage),
            Some(RollbackStage::Completed)
        );
    }

    #[tokio::test]
    async fn unknown_stage_outcome_resumes_without_repeating_prior_stage() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id);
        let plan = RollbackPlan::new(
            &tenant,
            &checkpoint,
            diagnostic(&tenant, &checkpoint).await,
            OperationId::from_uuid(Uuid::from_u128(800)),
            hash(40),
        )
        .expect("rollback plan");
        let coordinator = RollbackCoordinator::new(TestBackend::unknown_once());
        assert!(matches!(
            coordinator.execute(&tenant, &plan).await,
            Err(RollbackError::OutcomeUnknown)
        ));
        assert_eq!(
            coordinator
                .backend
                .progress(plan.plan_hash())
                .map(RollbackProgress::stage),
            Some(RollbackStage::WritesQuiesced)
        );
        assert_eq!(
            coordinator
                .execute(&tenant, &plan)
                .await
                .expect("resumed rollback")
                .restored_authority(),
            CutoverAuthority::Legacy
        );
    }

    #[tokio::test]
    async fn rollback_rejects_missing_or_stale_boundary() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let legacy = CutoverCheckpoint::new(
            aggregate(community_id),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("legacy checkpoint");
        let canonical = canonical_checkpoint(community_id);
        let canonical_diagnostic = diagnostic(&tenant, &canonical).await;
        assert!(matches!(
            RollbackPlan::new(
                &tenant,
                &legacy,
                canonical_diagnostic,
                OperationId::from_uuid(Uuid::from_u128(800)),
                hash(40),
            ),
            Err(RollbackError::TenantBoundaryViolation | RollbackError::UnsafeBoundary)
        ));

        let transition = CutoverTransition {
            operation_id: OperationId::from_uuid(Uuid::from_u128(501)),
            expected_version: canonical.version(),
            phase: CutoverPhase::Retirement,
            authority: CutoverAuthority::Canonical,
            source_cursor: canonical.source_cursor(),
            target_cursor: canonical.target_cursor(),
            integrity: canonical.integrity(),
            gates: canonical.gates(),
            reversible_boundary_label: None,
        };
        let CutoverTransitionOutcome::Advanced(retired) = canonical
            .transition(&transition)
            .expect("retirement transition")
        else {
            panic!("phase must advance");
        };
        assert!(matches!(
            RollbackPlan::new(
                &tenant,
                &retired,
                canonical_diagnostic,
                OperationId::from_uuid(Uuid::from_u128(801)),
                hash(41),
            ),
            Err(RollbackError::UnsafeBoundary)
        ));
    }
}
