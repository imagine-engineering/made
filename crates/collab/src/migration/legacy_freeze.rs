use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{AggregateVersion, OperationId, ScopedAggregateId, TenantContext};

use crate::migration::cutover_checkpoint::{CutoverAuthority, CutoverCheckpoint, CutoverPhase};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LegacyComponent {
    Desktop,
    AgentRuntime,
    Relay,
    Database,
    PubSub,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyWritePath {
    CompatibilityAdapter,
    DirectLegacyStore,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyWriteTrafficKind {
    CanonicalAdapterWrite,
    RejectedDirectWrite,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyWriteTrafficEvent {
    aggregate: ScopedAggregateId,
    checkpoint_version: AggregateVersion,
    component: LegacyComponent,
    operation_id: OperationId,
    observed_at_ms: i64,
    kind: LegacyWriteTrafficKind,
}

impl LegacyWriteTrafficEvent {
    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn checkpoint_version(&self) -> AggregateVersion {
        self.checkpoint_version
    }

    pub const fn component(&self) -> LegacyComponent {
        self.component
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn observed_at_ms(&self) -> i64 {
        self.observed_at_ms
    }

    pub const fn kind(&self) -> LegacyWriteTrafficKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyUsageRecordOutcome {
    Recorded,
    AlreadyRecorded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyUsageStoreError {
    Unavailable,
    OutcomeUnknown,
    ConflictingOperation,
}

impl fmt::Display for LegacyUsageStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "legacy usage storage is unavailable",
            Self::OutcomeUnknown => "legacy usage storage outcome is unknown",
            Self::ConflictingOperation => "legacy usage operation conflicts with prior input",
        })
    }
}

impl Error for LegacyUsageStoreError {}

#[async_trait]
pub trait LegacyUsageStore: Send + Sync {
    /// Records one operation-keyed attempt. `AlreadyRecorded` is valid only for the exact event.
    async fn record_write_attempt(
        &self,
        tenant: &TenantContext,
        event: &LegacyWriteTrafficEvent,
    ) -> Result<LegacyUsageRecordOutcome, LegacyUsageStoreError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalLegacyAdapterWritePermit {
    event: LegacyWriteTrafficEvent,
}

impl CanonicalLegacyAdapterWritePermit {
    pub fn event(&self) -> &LegacyWriteTrafficEvent {
        &self.event
    }
}

pub struct LegacyWriteFreeze<Store> {
    usage_store: Store,
}

impl<Store> LegacyWriteFreeze<Store>
where
    Store: LegacyUsageStore,
{
    pub const fn new(usage_store: Store) -> Self {
        Self { usage_store }
    }

    pub fn into_usage_store(self) -> Store {
        self.usage_store
    }

    pub async fn admit(
        &self,
        tenant: &TenantContext,
        checkpoint: &CutoverCheckpoint,
        component: LegacyComponent,
        path: LegacyWritePath,
        operation_id: OperationId,
        observed_at_ms: i64,
    ) -> Result<CanonicalLegacyAdapterWritePermit, LegacyFreezeError> {
        validate_checkpoint(tenant, checkpoint, false)?;
        if operation_id.as_uuid().is_nil() || observed_at_ms < 0 {
            return Err(LegacyFreezeError::InvalidInput);
        }
        let kind = match path {
            LegacyWritePath::CompatibilityAdapter => LegacyWriteTrafficKind::CanonicalAdapterWrite,
            LegacyWritePath::DirectLegacyStore => LegacyWriteTrafficKind::RejectedDirectWrite,
        };
        let event = LegacyWriteTrafficEvent {
            aggregate: checkpoint.aggregate(),
            checkpoint_version: checkpoint.version(),
            component,
            operation_id,
            observed_at_ms,
            kind,
        };
        self.usage_store
            .record_write_attempt(tenant, &event)
            .await
            .map_err(LegacyFreezeError::from_store)?;

        if path == LegacyWritePath::DirectLegacyStore {
            return Err(LegacyFreezeError::DirectLegacyWriteRejected);
        }
        Ok(CanonicalLegacyAdapterWritePermit { event })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LegacyTrafficCounts {
    adapter_reads: u64,
    adapter_writes: u64,
    active_client_high_watermark: u64,
    rejected_direct_writes: u64,
}

impl LegacyTrafficCounts {
    pub const fn new(
        adapter_reads: u64,
        adapter_writes: u64,
        active_client_high_watermark: u64,
        rejected_direct_writes: u64,
    ) -> Self {
        Self {
            adapter_reads,
            adapter_writes,
            active_client_high_watermark,
            rejected_direct_writes,
        }
    }

    pub const fn adapter_reads(self) -> u64 {
        self.adapter_reads
    }

    pub const fn adapter_writes(self) -> u64 {
        self.adapter_writes
    }

    pub const fn active_client_high_watermark(self) -> u64 {
        self.active_client_high_watermark
    }

    pub const fn rejected_direct_writes(self) -> u64 {
        self.rejected_direct_writes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyUsageSnapshot {
    aggregate: ScopedAggregateId,
    checkpoint_version: AggregateVersion,
    component: LegacyComponent,
    window_started_at_ms: i64,
    window_ended_at_ms: i64,
    counts: LegacyTrafficCounts,
}

impl LegacyUsageSnapshot {
    pub fn new(
        aggregate: ScopedAggregateId,
        checkpoint_version: AggregateVersion,
        component: LegacyComponent,
        window_started_at_ms: i64,
        window_ended_at_ms: i64,
        counts: LegacyTrafficCounts,
    ) -> Result<Self, LegacyFreezeError> {
        if aggregate.community_id().as_uuid().is_nil()
            || aggregate.aggregate_id().as_uuid().is_nil()
            || window_started_at_ms < 0
            || window_ended_at_ms <= window_started_at_ms
        {
            return Err(LegacyFreezeError::InvalidInput);
        }
        Ok(Self {
            aggregate,
            checkpoint_version,
            component,
            window_started_at_ms,
            window_ended_at_ms,
            counts,
        })
    }

    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn checkpoint_version(&self) -> AggregateVersion {
        self.checkpoint_version
    }

    pub const fn component(&self) -> LegacyComponent {
        self.component
    }

    pub const fn window_started_at_ms(&self) -> i64 {
        self.window_started_at_ms
    }

    pub const fn window_ended_at_ms(&self) -> i64 {
        self.window_ended_at_ms
    }

    pub const fn counts(&self) -> LegacyTrafficCounts {
        self.counts
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegacyRemovalThresholds {
    approval_hash: [u8; 32],
    minimum_observation_window_ms: u64,
    minimum_rollback_window_ms: u64,
    maximum_adapter_reads: u64,
    maximum_adapter_writes: u64,
    maximum_active_clients: u64,
}

impl LegacyRemovalThresholds {
    pub fn new(
        approval_hash: [u8; 32],
        minimum_observation_window_ms: u64,
        minimum_rollback_window_ms: u64,
        maximum_adapter_reads: u64,
        maximum_adapter_writes: u64,
        maximum_active_clients: u64,
    ) -> Result<Self, LegacyFreezeError> {
        if approval_hash == [0; 32]
            || minimum_observation_window_ms == 0
            || minimum_rollback_window_ms == 0
        {
            return Err(LegacyFreezeError::InvalidInput);
        }
        Ok(Self {
            approval_hash,
            minimum_observation_window_ms,
            minimum_rollback_window_ms,
            maximum_adapter_reads,
            maximum_adapter_writes,
            maximum_active_clients,
        })
    }

    pub const fn approval_hash(self) -> [u8; 32] {
        self.approval_hash
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RollbackWindowEvidence {
    started_at_ms: i64,
    evaluated_at_ms: i64,
    evidence_hash: [u8; 32],
}

impl RollbackWindowEvidence {
    pub fn new(
        started_at_ms: i64,
        evaluated_at_ms: i64,
        evidence_hash: [u8; 32],
    ) -> Result<Self, LegacyFreezeError> {
        if started_at_ms < 0 || evaluated_at_ms <= started_at_ms || evidence_hash == [0; 32] {
            return Err(LegacyFreezeError::InvalidInput);
        }
        Ok(Self {
            started_at_ms,
            evaluated_at_ms,
            evidence_hash,
        })
    }

    pub const fn started_at_ms(self) -> i64 {
        self.started_at_ms
    }

    pub const fn evaluated_at_ms(self) -> i64 {
        self.evaluated_at_ms
    }

    pub const fn evidence_hash(self) -> [u8; 32] {
        self.evidence_hash
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyRemovalGateReceipt {
    aggregate: ScopedAggregateId,
    checkpoint_version: AggregateVersion,
    component: LegacyComponent,
    usage: LegacyTrafficCounts,
    threshold_approval_hash: [u8; 32],
    rollback_evidence_hash: [u8; 32],
}

impl LegacyRemovalGateReceipt {
    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn checkpoint_version(&self) -> AggregateVersion {
        self.checkpoint_version
    }

    pub const fn component(&self) -> LegacyComponent {
        self.component
    }

    pub const fn usage(&self) -> LegacyTrafficCounts {
        self.usage
    }

    pub const fn threshold_approval_hash(&self) -> [u8; 32] {
        self.threshold_approval_hash
    }

    pub const fn rollback_evidence_hash(&self) -> [u8; 32] {
        self.rollback_evidence_hash
    }
}

pub struct LegacyRemovalGate;

impl LegacyRemovalGate {
    pub fn evaluate(
        tenant: &TenantContext,
        checkpoint: &CutoverCheckpoint,
        usage: &LegacyUsageSnapshot,
        thresholds: LegacyRemovalThresholds,
        rollback: RollbackWindowEvidence,
    ) -> Result<LegacyRemovalGateReceipt, LegacyFreezeError> {
        validate_checkpoint(tenant, checkpoint, true)?;
        if usage.aggregate != checkpoint.aggregate()
            || usage.checkpoint_version != checkpoint.version()
            || rollback.started_at_ms > usage.window_started_at_ms
            || rollback.evaluated_at_ms != usage.window_ended_at_ms
        {
            return Err(LegacyFreezeError::InvalidInput);
        }
        let observation_window = elapsed_ms(usage.window_started_at_ms, usage.window_ended_at_ms)?;
        if observation_window < thresholds.minimum_observation_window_ms {
            return Err(LegacyFreezeError::ObservationWindowIncomplete);
        }
        let rollback_window = elapsed_ms(rollback.started_at_ms, rollback.evaluated_at_ms)?;
        if rollback_window < thresholds.minimum_rollback_window_ms {
            return Err(LegacyFreezeError::RollbackWindowIncomplete);
        }
        if usage.counts.rejected_direct_writes > 0 {
            return Err(LegacyFreezeError::DirectWriteTrafficAboveThreshold);
        }
        if usage.counts.adapter_reads > thresholds.maximum_adapter_reads {
            return Err(LegacyFreezeError::AdapterReadTrafficAboveThreshold);
        }
        if usage.counts.adapter_writes > thresholds.maximum_adapter_writes {
            return Err(LegacyFreezeError::AdapterWriteTrafficAboveThreshold);
        }
        if usage.counts.active_client_high_watermark > thresholds.maximum_active_clients {
            return Err(LegacyFreezeError::ActiveClientTrafficAboveThreshold);
        }
        Ok(LegacyRemovalGateReceipt {
            aggregate: usage.aggregate,
            checkpoint_version: usage.checkpoint_version,
            component: usage.component,
            usage: usage.counts,
            threshold_approval_hash: thresholds.approval_hash,
            rollback_evidence_hash: rollback.evidence_hash,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyFreezeError {
    InvalidInput,
    TenantBoundaryViolation,
    CheckpointNotReady,
    DirectLegacyWriteRejected,
    UsageUnavailable,
    UsageOutcomeUnknown,
    UsageOperationConflict,
    ObservationWindowIncomplete,
    RollbackWindowIncomplete,
    DirectWriteTrafficAboveThreshold,
    AdapterReadTrafficAboveThreshold,
    AdapterWriteTrafficAboveThreshold,
    ActiveClientTrafficAboveThreshold,
}

impl LegacyFreezeError {
    const fn from_store(error: LegacyUsageStoreError) -> Self {
        match error {
            LegacyUsageStoreError::Unavailable => Self::UsageUnavailable,
            LegacyUsageStoreError::OutcomeUnknown => Self::UsageOutcomeUnknown,
            LegacyUsageStoreError::ConflictingOperation => Self::UsageOperationConflict,
        }
    }
}

impl fmt::Display for LegacyFreezeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidInput => "legacy freeze input is invalid",
            Self::TenantBoundaryViolation => "legacy freeze crossed its tenant boundary",
            Self::CheckpointNotReady => "legacy freeze checkpoint is not ready",
            Self::DirectLegacyWriteRejected => "direct legacy writes are frozen",
            Self::UsageUnavailable => "legacy usage storage is unavailable",
            Self::UsageOutcomeUnknown => "legacy usage storage outcome is unknown",
            Self::UsageOperationConflict => "legacy usage operation conflicts with prior input",
            Self::ObservationWindowIncomplete => "legacy usage observation window is incomplete",
            Self::RollbackWindowIncomplete => "legacy rollback window is incomplete",
            Self::DirectWriteTrafficAboveThreshold => {
                "direct legacy write traffic is above the removal threshold"
            }
            Self::AdapterReadTrafficAboveThreshold => {
                "legacy adapter read traffic is above the removal threshold"
            }
            Self::AdapterWriteTrafficAboveThreshold => {
                "legacy adapter write traffic is above the removal threshold"
            }
            Self::ActiveClientTrafficAboveThreshold => {
                "active legacy client traffic is above the removal threshold"
            }
        })
    }
}

impl Error for LegacyFreezeError {}

fn validate_checkpoint(
    tenant: &TenantContext,
    checkpoint: &CutoverCheckpoint,
    retirement_required: bool,
) -> Result<(), LegacyFreezeError> {
    if tenant.community_id() != checkpoint.aggregate().community_id() {
        return Err(LegacyFreezeError::TenantBoundaryViolation);
    }
    let phase_ready = if retirement_required {
        checkpoint.phase() == CutoverPhase::Retirement
    } else {
        checkpoint.phase() >= CutoverPhase::CommunicationWriteCutover
    };
    if !phase_ready
        || checkpoint.authority() != CutoverAuthority::Canonical
        || checkpoint.last_reversible_boundary().is_none()
    {
        return Err(LegacyFreezeError::CheckpointNotReady);
    }
    Ok(())
}

fn elapsed_ms(started_at_ms: i64, ended_at_ms: i64) -> Result<u64, LegacyFreezeError> {
    let elapsed = ended_at_ms
        .checked_sub(started_at_ms)
        .ok_or(LegacyFreezeError::InvalidInput)?;
    u64::try_from(elapsed).map_err(|_| LegacyFreezeError::InvalidInput)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Mutex};

    use collaboration_domain::{AggregateId, AggregateType, CommunityId, TrustedTenantRoute};
    use uuid::Uuid;

    use super::*;
    use crate::migration::cutover_checkpoint::{
        CutoverCursor, CutoverGateEvidence, CutoverIntegrity, CutoverTransition,
        CutoverTransitionOutcome,
    };

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
                TrustedTenantRoute::from_listener(community_id, "legacy-freeze")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn canonical_checkpoint(community_id: CommunityId, phase: CutoverPhase) -> CutoverCheckpoint {
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
            reversible_boundary_label: Some("before-legacy-freeze".to_string()),
        };
        let CutoverTransitionOutcome::Advanced(checkpoint) = checkpoint
            .transition(&transition)
            .expect("canonical transition")
        else {
            panic!("authority must advance");
        };
        if phase == CutoverPhase::CommunicationWriteCutover {
            return checkpoint;
        }
        let transition = CutoverTransition {
            operation_id: OperationId::from_uuid(Uuid::from_u128(501)),
            expected_version: checkpoint.version(),
            phase,
            authority: CutoverAuthority::Canonical,
            source_cursor: cursor,
            target_cursor: cursor,
            integrity,
            gates,
            reversible_boundary_label: None,
        };
        let CutoverTransitionOutcome::Advanced(checkpoint) = checkpoint
            .transition(&transition)
            .expect("phase transition")
        else {
            panic!("phase must advance");
        };
        checkpoint
    }

    #[derive(Default)]
    struct TestUsageStore {
        events: Mutex<BTreeMap<OperationId, LegacyWriteTrafficEvent>>,
    }

    impl TestUsageStore {
        fn events(&self) -> Vec<LegacyWriteTrafficEvent> {
            self.events
                .lock()
                .expect("usage lock")
                .values()
                .cloned()
                .collect()
        }
    }

    #[async_trait]
    impl LegacyUsageStore for TestUsageStore {
        async fn record_write_attempt(
            &self,
            _tenant: &TenantContext,
            event: &LegacyWriteTrafficEvent,
        ) -> Result<LegacyUsageRecordOutcome, LegacyUsageStoreError> {
            let mut events = self.events.lock().expect("usage lock");
            if let Some(existing) = events.get(&event.operation_id()) {
                return if existing == event {
                    Ok(LegacyUsageRecordOutcome::AlreadyRecorded)
                } else {
                    Err(LegacyUsageStoreError::ConflictingOperation)
                };
            }
            events.insert(event.operation_id(), event.clone());
            Ok(LegacyUsageRecordOutcome::Recorded)
        }
    }

    #[gpui::test]
    async fn direct_legacy_writes_are_recorded_and_never_receive_a_permit() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint =
            canonical_checkpoint(community_id, CutoverPhase::CommunicationWriteCutover);
        let freeze = LegacyWriteFreeze::new(TestUsageStore::default());

        let error = freeze
            .admit(
                &tenant,
                &checkpoint,
                LegacyComponent::Database,
                LegacyWritePath::DirectLegacyStore,
                OperationId::from_uuid(Uuid::from_u128(700)),
                1_000,
            )
            .await
            .expect_err("direct write must be rejected");
        assert_eq!(error, LegacyFreezeError::DirectLegacyWriteRejected);
        let events = freeze.into_usage_store().events();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].kind(),
            LegacyWriteTrafficKind::RejectedDirectWrite
        );
    }

    #[gpui::test]
    async fn canonical_adapter_write_is_measured_before_permit_and_retries_exactly() {
        let community_id = community(2);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id, CutoverPhase::Retirement);
        let freeze = LegacyWriteFreeze::new(TestUsageStore::default());
        let operation_id = OperationId::from_uuid(Uuid::from_u128(701));

        let first = freeze
            .admit(
                &tenant,
                &checkpoint,
                LegacyComponent::Desktop,
                LegacyWritePath::CompatibilityAdapter,
                operation_id,
                2_000,
            )
            .await
            .expect("adapter permit");
        let retry = freeze
            .admit(
                &tenant,
                &checkpoint,
                LegacyComponent::Desktop,
                LegacyWritePath::CompatibilityAdapter,
                operation_id,
                2_000,
            )
            .await
            .expect("adapter retry");
        assert_eq!(first, retry);
        assert_eq!(
            first.event().kind(),
            LegacyWriteTrafficKind::CanonicalAdapterWrite
        );
        assert_eq!(freeze.into_usage_store().events().len(), 1);
    }

    fn thresholds() -> LegacyRemovalThresholds {
        LegacyRemovalThresholds::new(hash(20), 1_000, 10_000, 2, 1, 1).expect("thresholds")
    }

    fn usage(checkpoint: &CutoverCheckpoint, counts: LegacyTrafficCounts) -> LegacyUsageSnapshot {
        LegacyUsageSnapshot::new(
            checkpoint.aggregate(),
            checkpoint.version(),
            LegacyComponent::Relay,
            10_000,
            20_000,
            counts,
        )
        .expect("usage")
    }

    #[test]
    fn removal_gate_rejects_every_traffic_threshold_excess() {
        let community_id = community(3);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id, CutoverPhase::Retirement);
        let rollback =
            RollbackWindowEvidence::new(5_000, 20_000, hash(21)).expect("rollback evidence");
        let cases = [
            (
                LegacyTrafficCounts::new(3, 0, 0, 0),
                LegacyFreezeError::AdapterReadTrafficAboveThreshold,
            ),
            (
                LegacyTrafficCounts::new(0, 2, 0, 0),
                LegacyFreezeError::AdapterWriteTrafficAboveThreshold,
            ),
            (
                LegacyTrafficCounts::new(0, 0, 2, 0),
                LegacyFreezeError::ActiveClientTrafficAboveThreshold,
            ),
            (
                LegacyTrafficCounts::new(0, 0, 0, 1),
                LegacyFreezeError::DirectWriteTrafficAboveThreshold,
            ),
        ];

        for (counts, expected) in cases {
            assert_eq!(
                LegacyRemovalGate::evaluate(
                    &tenant,
                    &checkpoint,
                    &usage(&checkpoint, counts),
                    thresholds(),
                    rollback,
                ),
                Err(expected)
            );
        }
    }

    #[test]
    fn removal_gate_requires_complete_observation_and_rollback_windows() {
        let community_id = community(4);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id, CutoverPhase::Retirement);
        let counts = LegacyTrafficCounts::new(2, 1, 1, 0);
        let short_observation = LegacyUsageSnapshot::new(
            checkpoint.aggregate(),
            checkpoint.version(),
            LegacyComponent::Relay,
            19_500,
            20_000,
            counts,
        )
        .expect("usage");
        let rollback =
            RollbackWindowEvidence::new(5_000, 20_000, hash(21)).expect("rollback evidence");
        assert_eq!(
            LegacyRemovalGate::evaluate(
                &tenant,
                &checkpoint,
                &short_observation,
                thresholds(),
                rollback,
            ),
            Err(LegacyFreezeError::ObservationWindowIncomplete)
        );

        let rollback_window_usage = LegacyUsageSnapshot::new(
            checkpoint.aggregate(),
            checkpoint.version(),
            LegacyComponent::Relay,
            15_000,
            20_000,
            counts,
        )
        .expect("usage");
        let short_rollback =
            RollbackWindowEvidence::new(14_000, 20_000, hash(21)).expect("rollback evidence");
        assert_eq!(
            LegacyRemovalGate::evaluate(
                &tenant,
                &checkpoint,
                &rollback_window_usage,
                thresholds(),
                short_rollback,
            ),
            Err(LegacyFreezeError::RollbackWindowIncomplete)
        );
    }

    #[test]
    fn removal_gate_accepts_exact_approved_thresholds_after_retirement_window() {
        let community_id = community(5);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id, CutoverPhase::Retirement);
        let counts = LegacyTrafficCounts::new(2, 1, 1, 0);
        let rollback =
            RollbackWindowEvidence::new(5_000, 20_000, hash(21)).expect("rollback evidence");

        let receipt = LegacyRemovalGate::evaluate(
            &tenant,
            &checkpoint,
            &usage(&checkpoint, counts),
            thresholds(),
            rollback,
        )
        .expect("eligible for removal");
        assert_eq!(receipt.component(), LegacyComponent::Relay);
        assert_eq!(receipt.usage(), counts);
        assert_eq!(receipt.threshold_approval_hash(), hash(20));
        assert_eq!(receipt.rollback_evidence_hash(), hash(21));
    }
}
