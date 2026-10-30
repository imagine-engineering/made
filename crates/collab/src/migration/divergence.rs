use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    AuditOutcome, AuthenticatedPrincipal, OperationId, ScopedAggregateId, TenantContext,
};
use sha2::{Digest, Sha256};

use crate::{
    audit::events::{
        AuditEventContext, AuditFailureClass, MigrationAuditOperation, MigrationAuditSource,
        SecurityAuditEvent,
    },
    migration::{
        cutover_checkpoint::{CutoverAuthority, CutoverCheckpoint, CutoverCursor, CutoverPhase},
        legacy_mirror::LegacyMirrorError,
        shadow_read::{ShadowComparison, ShadowDivergence},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DivergenceReason {
    Authorization,
    Signature,
    Count,
    Hash,
    LegacyOnlyWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DivergenceSignal {
    reason: DivergenceReason,
    evidence_hash: [u8; 32],
    expected_count: Option<u64>,
    observed_count: Option<u64>,
}

impl DivergenceSignal {
    pub fn authorization(comparison: &ShadowComparison) -> Result<Self, DivergenceStopError> {
        if !comparison
            .divergences()
            .contains(&ShadowDivergence::Authorization)
        {
            return Err(DivergenceStopError::InvalidSignal);
        }
        Ok(Self::hashed(
            DivergenceReason::Authorization,
            &[
                &comparison.legacy().authorization_hash(),
                &comparison.canonical().authorization_hash(),
            ],
        ))
    }

    pub fn signature(evidence_hash: [u8; 32]) -> Result<Self, DivergenceStopError> {
        Self::direct(DivergenceReason::Signature, evidence_hash)
    }

    pub fn count(expected: u64, observed: u64) -> Result<Self, DivergenceStopError> {
        if expected == observed {
            return Err(DivergenceStopError::InvalidSignal);
        }
        let mut hasher = Sha256::new();
        hasher.update(expected.to_be_bytes());
        hasher.update(observed.to_be_bytes());
        Ok(Self {
            reason: DivergenceReason::Count,
            evidence_hash: hasher.finalize().into(),
            expected_count: Some(expected),
            observed_count: Some(observed),
        })
    }

    pub fn hash(comparison: &ShadowComparison) -> Result<Self, DivergenceStopError> {
        if !comparison.divergences().iter().any(|divergence| {
            matches!(
                divergence,
                ShadowDivergence::Content
                    | ShadowDivergence::Order
                    | ShadowDivergence::Cursor
                    | ShadowDivergence::Overlay
            )
        }) {
            return Err(DivergenceStopError::InvalidSignal);
        }
        let legacy = comparison.legacy();
        let canonical = comparison.canonical();
        let legacy_cursor = legacy.cursor_hash().unwrap_or([0; 32]);
        let canonical_cursor = canonical.cursor_hash().unwrap_or([0; 32]);
        Ok(Self::hashed(
            DivergenceReason::Hash,
            &[
                &legacy.content_hash(),
                &canonical.content_hash(),
                &legacy.order_hash(),
                &canonical.order_hash(),
                &legacy_cursor,
                &canonical_cursor,
                &legacy.overlay_hash(),
                &canonical.overlay_hash(),
            ],
        ))
    }

    pub fn legacy_only_write(
        mirror_error: LegacyMirrorError,
        operation_id: OperationId,
        source_reference_hash: [u8; 32],
    ) -> Result<Self, DivergenceStopError> {
        if mirror_error != LegacyMirrorError::ReverseReconciliationForbidden
            || operation_id.as_uuid().is_nil()
            || source_reference_hash == [0; 32]
        {
            return Err(DivergenceStopError::InvalidSignal);
        }
        let operation_bytes = operation_id.as_uuid().into_bytes();
        Ok(Self::hashed(
            DivergenceReason::LegacyOnlyWrite,
            &[&operation_bytes, &source_reference_hash],
        ))
    }

    pub const fn reason(self) -> DivergenceReason {
        self.reason
    }

    pub const fn evidence_hash(self) -> [u8; 32] {
        self.evidence_hash
    }

    pub const fn expected_count(self) -> Option<u64> {
        self.expected_count
    }

    pub const fn observed_count(self) -> Option<u64> {
        self.observed_count
    }

    fn direct(
        reason: DivergenceReason,
        evidence_hash: [u8; 32],
    ) -> Result<Self, DivergenceStopError> {
        if evidence_hash == [0; 32] {
            return Err(DivergenceStopError::InvalidSignal);
        }
        Ok(Self {
            reason,
            evidence_hash,
            expected_count: None,
            observed_count: None,
        })
    }

    fn hashed(reason: DivergenceReason, parts: &[&[u8]]) -> Self {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update(part);
        }
        Self {
            reason,
            evidence_hash: hasher.finalize().into(),
            expected_count: None,
            observed_count: None,
        }
    }

    fn valid(self) -> bool {
        self.evidence_hash != [0; 32]
            && match self.reason {
                DivergenceReason::Count => matches!(
                    (self.expected_count, self.observed_count),
                    (Some(expected), Some(observed)) if expected != observed
                ),
                _ => self.expected_count.is_none() && self.observed_count.is_none(),
            }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LastTrustworthyCutoverState {
    checkpoint_version: collaboration_domain::AggregateVersion,
    phase: CutoverPhase,
    authority: CutoverAuthority,
    source_cursor: CutoverCursor,
    target_cursor: CutoverCursor,
    has_reversible_boundary: bool,
}

impl LastTrustworthyCutoverState {
    fn from_checkpoint(checkpoint: &CutoverCheckpoint) -> Self {
        Self {
            checkpoint_version: checkpoint.version(),
            phase: checkpoint.phase(),
            authority: checkpoint.authority(),
            source_cursor: checkpoint.source_cursor(),
            target_cursor: checkpoint.target_cursor(),
            has_reversible_boundary: checkpoint.last_reversible_boundary().is_some(),
        }
    }

    pub const fn checkpoint_version(self) -> collaboration_domain::AggregateVersion {
        self.checkpoint_version
    }

    pub const fn phase(self) -> CutoverPhase {
        self.phase
    }

    pub const fn authority(self) -> CutoverAuthority {
        self.authority
    }

    pub const fn source_cursor(self) -> CutoverCursor {
        self.source_cursor
    }

    pub const fn target_cursor(self) -> CutoverCursor {
        self.target_cursor
    }

    pub const fn has_reversible_boundary(self) -> bool {
        self.has_reversible_boundary
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DivergenceRecoveryAction {
    KeepLegacyAuthority,
    DisableCanonicalReads,
    RebuildCanonicalProjection,
    VerifyZeroDivergence,
    QuiesceCanonicalWrites,
    DrainCanonicalOutbox,
    VerifyReversibleBoundary,
    RestorePriorAuthority,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DivergenceRollbackGuidance {
    aggregate: ScopedAggregateId,
    actions: [DivergenceRecoveryAction; 4],
}

impl DivergenceRollbackGuidance {
    fn for_checkpoint(checkpoint: &CutoverCheckpoint) -> Self {
        let actions = match checkpoint.authority() {
            CutoverAuthority::Legacy => [
                DivergenceRecoveryAction::KeepLegacyAuthority,
                DivergenceRecoveryAction::DisableCanonicalReads,
                DivergenceRecoveryAction::RebuildCanonicalProjection,
                DivergenceRecoveryAction::VerifyZeroDivergence,
            ],
            CutoverAuthority::Canonical => [
                DivergenceRecoveryAction::QuiesceCanonicalWrites,
                DivergenceRecoveryAction::DrainCanonicalOutbox,
                DivergenceRecoveryAction::VerifyReversibleBoundary,
                DivergenceRecoveryAction::RestorePriorAuthority,
            ],
        };
        Self {
            aggregate: checkpoint.aggregate(),
            actions,
        }
    }

    pub const fn aggregate(self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn actions(self) -> [DivergenceRecoveryAction; 4] {
        self.actions
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DivergenceHalt {
    aggregate: ScopedAggregateId,
    operation_id: OperationId,
    reason: DivergenceReason,
    evidence_hash: [u8; 32],
    expected_count: Option<u64>,
    observed_count: Option<u64>,
    detected_at_millis: u64,
    last_trustworthy: LastTrustworthyCutoverState,
    rollback: DivergenceRollbackGuidance,
}

impl DivergenceHalt {
    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn reason(&self) -> DivergenceReason {
        self.reason
    }

    pub const fn evidence_hash(&self) -> [u8; 32] {
        self.evidence_hash
    }

    pub const fn expected_count(&self) -> Option<u64> {
        self.expected_count
    }

    pub const fn observed_count(&self) -> Option<u64> {
        self.observed_count
    }

    pub const fn detected_at_millis(&self) -> u64 {
        self.detected_at_millis
    }

    pub const fn last_trustworthy(&self) -> LastTrustworthyCutoverState {
        self.last_trustworthy
    }

    pub const fn rollback(&self) -> DivergenceRollbackGuidance {
        self.rollback
    }

    fn valid(&self) -> bool {
        !self.aggregate.community_id().as_uuid().is_nil()
            && !self.aggregate.aggregate_id().as_uuid().is_nil()
            && !self.operation_id.as_uuid().is_nil()
            && self.detected_at_millis > 0
            && self.evidence_hash != [0; 32]
            && self.rollback.aggregate == self.aggregate
            && match self.reason {
                DivergenceReason::Count => matches!(
                    (self.expected_count, self.observed_count),
                    (Some(expected), Some(observed)) if expected != observed
                ),
                _ => self.expected_count.is_none() && self.observed_count.is_none(),
            }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DivergenceOperatorDiagnostic {
    aggregate: ScopedAggregateId,
    operation_id: OperationId,
    reason: DivergenceReason,
    detected_at_millis: u64,
    last_trustworthy: LastTrustworthyCutoverState,
    rollback: DivergenceRollbackGuidance,
}

impl DivergenceOperatorDiagnostic {
    fn from_halt(halt: &DivergenceHalt) -> Self {
        Self {
            aggregate: halt.aggregate,
            operation_id: halt.operation_id,
            reason: halt.reason,
            detected_at_millis: halt.detected_at_millis,
            last_trustworthy: halt.last_trustworthy,
            rollback: halt.rollback,
        }
    }

    pub const fn aggregate(self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn operation_id(self) -> OperationId {
        self.operation_id
    }

    pub const fn reason(self) -> DivergenceReason {
        self.reason
    }

    pub const fn detected_at_millis(self) -> u64 {
        self.detected_at_millis
    }

    pub const fn last_trustworthy(self) -> LastTrustworthyCutoverState {
        self.last_trustworthy
    }

    pub const fn rollback(self) -> DivergenceRollbackGuidance {
        self.rollback
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DivergenceStoreOutcome {
    Halted,
    AlreadyHalted(DivergenceHalt),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DivergenceStoreError {
    Unavailable,
    OutcomeUnknown,
    InvalidData,
}

impl fmt::Display for DivergenceStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "divergence halt storage is unavailable",
            Self::OutcomeUnknown => "divergence halt outcome is unknown",
            Self::InvalidData => "divergence halt storage returned invalid data",
        })
    }
}

impl Error for DivergenceStoreError {}

#[async_trait]
pub trait DivergenceHaltStore: Send + Sync {
    /// Atomically persists the first halt for `halt.aggregate` and its canonical audit event.
    async fn halt_and_record(
        &self,
        tenant: &TenantContext,
        halt: DivergenceHalt,
        audit_event: SecurityAuditEvent,
    ) -> Result<DivergenceStoreOutcome, DivergenceStoreError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DivergenceStopOutcome {
    Halted(DivergenceOperatorDiagnostic),
    AlreadyHalted(DivergenceOperatorDiagnostic),
}

pub struct DivergenceStopCoordinator<Store> {
    store: Store,
}

impl<Store> DivergenceStopCoordinator<Store>
where
    Store: DivergenceHaltStore,
{
    pub const fn new(store: Store) -> Self {
        Self { store }
    }

    pub fn into_store(self) -> Store {
        self.store
    }

    pub async fn halt(
        &self,
        tenant: &TenantContext,
        checkpoint: &CutoverCheckpoint,
        signal: DivergenceSignal,
        operation_id: OperationId,
        actor: &AuthenticatedPrincipal,
        detected_at_millis: u64,
    ) -> Result<DivergenceStopOutcome, DivergenceStopError> {
        validate_boundary(tenant, checkpoint, signal, operation_id, detected_at_millis)?;
        if signal.reason == DivergenceReason::LegacyOnlyWrite
            && checkpoint.authority() != CutoverAuthority::Canonical
        {
            return Err(DivergenceStopError::InvalidSignal);
        }
        let halt = DivergenceHalt {
            aggregate: checkpoint.aggregate(),
            operation_id,
            reason: signal.reason,
            evidence_hash: signal.evidence_hash,
            expected_count: signal.expected_count,
            observed_count: signal.observed_count,
            detected_at_millis,
            last_trustworthy: LastTrustworthyCutoverState::from_checkpoint(checkpoint),
            rollback: DivergenceRollbackGuidance::for_checkpoint(checkpoint),
        };
        let failure_class = match signal.reason {
            DivergenceReason::Authorization => AuditFailureClass::AuthorizationDenied,
            DivergenceReason::Signature
            | DivergenceReason::Count
            | DivergenceReason::Hash
            | DivergenceReason::LegacyOnlyWrite => AuditFailureClass::IntegrityViolation,
        };
        let audit_context = AuditEventContext::new(
            tenant,
            operation_id,
            Some(actor),
            AuditOutcome::Failed,
            Some(failure_class),
            detected_at_millis,
        )?;
        let audit_event = SecurityAuditEvent::Migration {
            context: audit_context,
            operation: MigrationAuditOperation::Checkpoint,
            source: MigrationAuditSource::Native,
            migration_id: checkpoint.aggregate().aggregate_id().as_uuid(),
            checkpoint: Some(checkpoint.version().get()),
        };
        match self
            .store
            .halt_and_record(tenant, halt.clone(), audit_event)
            .await
        {
            Ok(DivergenceStoreOutcome::Halted) => Ok(DivergenceStopOutcome::Halted(
                DivergenceOperatorDiagnostic::from_halt(&halt),
            )),
            Ok(DivergenceStoreOutcome::AlreadyHalted(existing)) => {
                if !existing.valid()
                    || existing.aggregate != halt.aggregate
                    || existing.aggregate.community_id() != tenant.community_id()
                {
                    return Err(DivergenceStopError::InvalidBackendResponse);
                }
                Ok(DivergenceStopOutcome::AlreadyHalted(
                    DivergenceOperatorDiagnostic::from_halt(&existing),
                ))
            }
            Err(DivergenceStoreError::Unavailable) => Err(DivergenceStopError::StorageUnavailable),
            Err(DivergenceStoreError::OutcomeUnknown) => Err(DivergenceStopError::OutcomeUnknown),
            Err(DivergenceStoreError::InvalidData) => {
                Err(DivergenceStopError::InvalidBackendResponse)
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DivergenceStopError {
    #[error("divergence signal is invalid")]
    InvalidSignal,
    #[error("divergence stop crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("divergence stop is unavailable after retirement")]
    RetirementComplete,
    #[error("divergence halt storage is unavailable")]
    StorageUnavailable,
    #[error("divergence halt outcome is unknown")]
    OutcomeUnknown,
    #[error("divergence halt storage returned invalid data")]
    InvalidBackendResponse,
    #[error("divergence audit event is invalid")]
    Audit(#[from] crate::audit::events::AuditEventError),
}

fn validate_boundary(
    tenant: &TenantContext,
    checkpoint: &CutoverCheckpoint,
    signal: DivergenceSignal,
    operation_id: OperationId,
    detected_at_millis: u64,
) -> Result<(), DivergenceStopError> {
    if tenant.community_id() != checkpoint.aggregate().community_id()
        || checkpoint.aggregate().community_id() != tenant.community_id()
    {
        return Err(DivergenceStopError::TenantBoundaryViolation);
    }
    if !signal.valid() || operation_id.as_uuid().is_nil() || detected_at_millis == 0 {
        return Err(DivergenceStopError::InvalidSignal);
    }
    if checkpoint.phase() >= CutoverPhase::Retirement {
        return Err(DivergenceStopError::RetirementComplete);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Mutex};

    use collaboration_domain::{
        AggregateId, AggregateType, CommunityId, PrincipalId, PrincipalScopes, TrustedTenantRoute,
    };
    use uuid::Uuid;

    use super::*;
    use crate::migration::{
        cutover_checkpoint::{
            CutoverGateEvidence, CutoverIntegrity, CutoverTransition, CutoverTransitionOutcome,
        },
        shadow_read::{
            ShadowAuthorization, ShadowAuthorizationDenial, ShadowReadResult, ShadowReadRow,
        },
    };

    fn hash(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate(community_id: CommunityId, value: u128) -> ScopedAggregateId {
        ScopedAggregateId::new(
            community_id,
            AggregateType::Conversation,
            AggregateId::from_uuid(Uuid::from_u128(value)),
        )
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "divergence-stop")
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
            "migration-divergence",
            PrincipalScopes::default(),
        )
        .expect("service actor")
    }

    fn canonical_checkpoint(aggregate: ScopedAggregateId) -> CutoverCheckpoint {
        let checkpoint = CutoverCheckpoint::new(aggregate, CutoverPhase::CommunicationReadShadow)
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
            reversible_boundary_label: Some("before-divergence".to_string()),
        };
        let CutoverTransitionOutcome::Advanced(checkpoint) = checkpoint
            .transition(&transition)
            .expect("canonical transition")
        else {
            panic!("authority must advance");
        };
        checkpoint
    }

    fn row(coordinate: u8, content: u8) -> ShadowReadRow {
        ShadowReadRow::new(hash(coordinate), 1, hash(content)).expect("shadow row")
    }

    fn shadow_comparison(aggregate: ScopedAggregateId) -> ShadowComparison {
        let legacy = ShadowReadResult::new(
            aggregate,
            ShadowAuthorization::Allowed,
            vec![row(1, 11)],
            None,
            Vec::new(),
        )
        .expect("legacy result");
        let canonical = ShadowReadResult::new(
            aggregate,
            ShadowAuthorization::Denied(ShadowAuthorizationDenial::NotMember),
            Vec::new(),
            None,
            Vec::new(),
        )
        .expect("canonical result");
        ShadowComparison::compare(&legacy, &canonical)
    }

    #[derive(Default)]
    struct TestStore {
        halts: Mutex<BTreeMap<ScopedAggregateId, DivergenceHalt>>,
        audit_records: Mutex<u64>,
    }

    impl TestStore {
        fn halt(&self, aggregate: ScopedAggregateId) -> Option<DivergenceHalt> {
            self.halts
                .lock()
                .expect("halt lock")
                .get(&aggregate)
                .cloned()
        }

        fn audit_records(&self) -> u64 {
            *self.audit_records.lock().expect("audit lock")
        }
    }

    #[async_trait]
    impl DivergenceHaltStore for TestStore {
        async fn halt_and_record(
            &self,
            tenant: &TenantContext,
            halt: DivergenceHalt,
            audit_event: SecurityAuditEvent,
        ) -> Result<DivergenceStoreOutcome, DivergenceStoreError> {
            if halt.aggregate.community_id() != tenant.community_id()
                || audit_event.context().community_id() != tenant.community_id()
                || audit_event.into_record().is_err()
            {
                return Err(DivergenceStoreError::InvalidData);
            }
            let mut halts = self
                .halts
                .lock()
                .map_err(|_| DivergenceStoreError::Unavailable)?;
            if let Some(existing) = halts.get(&halt.aggregate) {
                return Ok(DivergenceStoreOutcome::AlreadyHalted(existing.clone()));
            }
            halts.insert(halt.aggregate, halt);
            let mut audit_records = self
                .audit_records
                .lock()
                .map_err(|_| DivergenceStoreError::Unavailable)?;
            *audit_records = audit_records
                .checked_add(1)
                .ok_or(DivergenceStoreError::InvalidData)?;
            Ok(DivergenceStoreOutcome::Halted)
        }
    }

    #[tokio::test]
    async fn every_stop_reason_halts_only_its_scoped_aggregate() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let actor = actor(community_id);
        let store = TestStore::default();
        let coordinator = DivergenceStopCoordinator::new(store);
        let reasons = [
            DivergenceReason::Authorization,
            DivergenceReason::Signature,
            DivergenceReason::Count,
            DivergenceReason::Hash,
            DivergenceReason::LegacyOnlyWrite,
        ];

        for (index, reason) in reasons.into_iter().enumerate() {
            let aggregate = aggregate(community_id, 100 + index as u128);
            let checkpoint = canonical_checkpoint(aggregate);
            let comparison = shadow_comparison(aggregate);
            let signal = match reason {
                DivergenceReason::Authorization => {
                    DivergenceSignal::authorization(&comparison).expect("authorization signal")
                }
                DivergenceReason::Signature => {
                    DivergenceSignal::signature(hash(20)).expect("signature signal")
                }
                DivergenceReason::Count => DivergenceSignal::count(10, 9).expect("count signal"),
                DivergenceReason::Hash => DivergenceSignal::hash(&comparison).expect("hash signal"),
                DivergenceReason::LegacyOnlyWrite => DivergenceSignal::legacy_only_write(
                    LegacyMirrorError::ReverseReconciliationForbidden,
                    OperationId::from_uuid(Uuid::from_u128(800)),
                    hash(21),
                )
                .expect("legacy-only signal"),
            };
            let operation_id = OperationId::from_uuid(Uuid::from_u128(1000 + index as u128));
            let DivergenceStopOutcome::Halted(diagnostic) = coordinator
                .halt(
                    &tenant,
                    &checkpoint,
                    signal,
                    operation_id,
                    &actor,
                    1_900_000_000_000 + index as u64,
                )
                .await
                .expect("halt")
            else {
                panic!("first signal must halt");
            };
            assert_eq!(diagnostic.aggregate(), aggregate);
            assert_eq!(diagnostic.reason(), reason);
            assert_eq!(
                diagnostic.last_trustworthy().authority(),
                CutoverAuthority::Canonical
            );
            assert_eq!(
                diagnostic.rollback().actions(),
                [
                    DivergenceRecoveryAction::QuiesceCanonicalWrites,
                    DivergenceRecoveryAction::DrainCanonicalOutbox,
                    DivergenceRecoveryAction::VerifyReversibleBoundary,
                    DivergenceRecoveryAction::RestorePriorAuthority,
                ]
            );
            assert_eq!(
                coordinator.store.halt(aggregate).map(|halt| halt.reason()),
                Some(reason)
            );
        }
        assert_eq!(coordinator.store.audit_records(), 5);
        assert!(
            coordinator
                .store
                .halt(aggregate(community_id, 999))
                .is_none()
        );
    }

    #[tokio::test]
    async fn exact_retry_returns_first_halt_without_duplicate_audit() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let aggregate = aggregate(community_id, 100);
        let checkpoint = canonical_checkpoint(aggregate);
        let coordinator = DivergenceStopCoordinator::new(TestStore::default());
        let signal = DivergenceSignal::count(10, 9).expect("count signal");
        let operation_id = OperationId::from_uuid(Uuid::from_u128(1000));
        assert!(matches!(
            coordinator
                .halt(
                    &tenant,
                    &checkpoint,
                    signal,
                    operation_id,
                    &actor(community_id),
                    1_900_000_000_000,
                )
                .await
                .expect("first halt"),
            DivergenceStopOutcome::Halted(_)
        ));
        assert!(matches!(
            coordinator
                .halt(
                    &tenant,
                    &checkpoint,
                    signal,
                    operation_id,
                    &actor(community_id),
                    1_900_000_000_000,
                )
                .await
                .expect("retry"),
            DivergenceStopOutcome::AlreadyHalted(_)
        ));
        assert_eq!(coordinator.store.audit_records(), 1);
    }

    #[tokio::test]
    async fn tenant_and_invalid_signal_reject_before_store() {
        let community_id = community(1);
        let aggregate = aggregate(community_id, 100);
        let checkpoint = canonical_checkpoint(aggregate);
        let coordinator = DivergenceStopCoordinator::new(TestStore::default());
        let signal = DivergenceSignal::signature(hash(20)).expect("signature signal");
        assert!(matches!(
            coordinator
                .halt(
                    &tenant(community(2)),
                    &checkpoint,
                    signal,
                    OperationId::from_uuid(Uuid::from_u128(1000)),
                    &actor(community(2)),
                    1_900_000_000_000,
                )
                .await,
            Err(DivergenceStopError::TenantBoundaryViolation)
        ));
        assert_eq!(coordinator.store.audit_records(), 0);
        assert!(matches!(
            DivergenceSignal::count(10, 10),
            Err(DivergenceStopError::InvalidSignal)
        ));
    }
}
