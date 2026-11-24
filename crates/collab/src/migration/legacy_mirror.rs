use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    AggregateVersion, OperationId, ScopedAggregateId, SourceRecordId, SourceSystem, TenantContext,
};
use sha2::{Digest, Sha256};

use crate::{
    db::collaboration::outbox::OutboxOperation,
    migration::cutover_checkpoint::{CutoverAuthority, CutoverCheckpoint, CutoverPhase},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalOutboxMirrorItem {
    aggregate: ScopedAggregateId,
    outbox_sequence: u64,
    operation_id: OperationId,
    authoritative_version: AggregateVersion,
    operation: OutboxOperation,
    source_version: String,
    payload_hash: [u8; 32],
}

impl CanonicalOutboxMirrorItem {
    pub fn new(
        aggregate: ScopedAggregateId,
        outbox_sequence: u64,
        operation_id: OperationId,
        authoritative_version: AggregateVersion,
        operation: OutboxOperation,
    ) -> Result<Self, LegacyMirrorError> {
        let source_version = operation
            .provenance()
            .source_version
            .as_deref()
            .ok_or(LegacyMirrorError::InvalidInput)?
            .to_owned();
        if aggregate.community_id().as_uuid().is_nil()
            || aggregate.aggregate_id().as_uuid().is_nil()
            || outbox_sequence == 0
            || operation_id.as_uuid().is_nil()
            || source_version.is_empty()
            || source_version.len() > 1024
            || source_version.trim() != source_version
            || source_version.chars().any(char::is_control)
        {
            return Err(LegacyMirrorError::InvalidInput);
        }
        let payload_hash = Sha256::digest(operation.payload()).into();
        Ok(Self {
            aggregate,
            outbox_sequence,
            operation_id,
            authoritative_version,
            operation,
            source_version,
            payload_hash,
        })
    }

    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn outbox_sequence(&self) -> u64 {
        self.outbox_sequence
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn authoritative_version(&self) -> AggregateVersion {
        self.authoritative_version
    }

    pub fn topic(&self) -> &str {
        self.operation.topic()
    }

    pub const fn source_system(&self) -> SourceSystem {
        self.operation.provenance().source_system
    }

    pub fn source_record_id(&self) -> &SourceRecordId {
        &self.operation.provenance().source_record_id
    }

    pub fn source_version(&self) -> &str {
        &self.source_version
    }

    pub fn payload(&self) -> &[u8] {
        self.operation.payload()
    }

    pub const fn payload_hash(&self) -> [u8; 32] {
        self.payload_hash
    }

    pub fn expected_receipt(&self) -> LegacyMirrorReceipt {
        LegacyMirrorReceipt {
            aggregate: self.aggregate,
            outbox_sequence: self.outbox_sequence,
            operation_id: self.operation_id,
            authoritative_version: self.authoritative_version,
            source_system: self.operation.provenance().source_system,
            source_record_id: self.operation.provenance().source_record_id.clone(),
            source_version: self.source_version().to_owned(),
            payload_hash: self.payload_hash,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyMirrorReceipt {
    aggregate: ScopedAggregateId,
    outbox_sequence: u64,
    operation_id: OperationId,
    authoritative_version: AggregateVersion,
    source_system: SourceSystem,
    source_record_id: SourceRecordId,
    source_version: String,
    payload_hash: [u8; 32],
}

impl LegacyMirrorReceipt {
    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn outbox_sequence(&self) -> u64 {
        self.outbox_sequence
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn authoritative_version(&self) -> AggregateVersion {
        self.authoritative_version
    }

    pub const fn source_system(&self) -> SourceSystem {
        self.source_system
    }

    pub fn source_record_id(&self) -> &SourceRecordId {
        &self.source_record_id
    }

    pub fn source_version(&self) -> &str {
        &self.source_version
    }

    pub const fn payload_hash(&self) -> [u8; 32] {
        self.payload_hash
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LegacyProjectionApplyOutcome {
    Applied(LegacyMirrorReceipt),
    AlreadyApplied(LegacyMirrorReceipt),
    Delayed {
        last_applied_version: Option<AggregateVersion>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyProjectionWriteError {
    Unavailable,
    OutcomeUnknown,
    LegacyAhead,
    ConflictingProjection,
    InvalidData,
}

impl fmt::Display for LegacyProjectionWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "legacy projection writer is unavailable",
            Self::OutcomeUnknown => "legacy projection write outcome is unknown",
            Self::LegacyAhead => "legacy projection is ahead of canonical input",
            Self::ConflictingProjection => "legacy projection conflicts with canonical input",
            Self::InvalidData => "legacy projection writer returned invalid data",
        })
    }
}

impl Error for LegacyProjectionWriteError {}

#[async_trait]
pub trait LegacyProjectionWriter: Send + Sync {
    /// Applies only canonical input to a temporary legacy projection. Implementations must return
    /// `AlreadyApplied` only for the exact receipt and must never write legacy state upstream.
    async fn apply_canonical(
        &self,
        tenant: &TenantContext,
        item: &CanonicalOutboxMirrorItem,
    ) -> Result<LegacyProjectionApplyOutcome, LegacyProjectionWriteError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LegacyMirrorOutcome {
    Mirrored(LegacyMirrorReceipt),
    Duplicate(LegacyMirrorReceipt),
    Delayed {
        item: LegacyMirrorReceipt,
        last_applied_version: Option<AggregateVersion>,
    },
}

pub struct LegacyMirror<Writer> {
    writer: Writer,
}

impl<Writer> LegacyMirror<Writer>
where
    Writer: LegacyProjectionWriter,
{
    pub const fn new(writer: Writer) -> Self {
        Self { writer }
    }

    pub fn into_writer(self) -> Writer {
        self.writer
    }

    pub async fn mirror_one(
        &self,
        tenant: &TenantContext,
        checkpoint: &CutoverCheckpoint,
        item: CanonicalOutboxMirrorItem,
    ) -> Result<LegacyMirrorOutcome, LegacyMirrorError> {
        validate_boundary(tenant, checkpoint, &item)?;
        let expected_receipt = item.expected_receipt();
        match self.writer.apply_canonical(tenant, &item).await {
            Ok(LegacyProjectionApplyOutcome::Applied(receipt)) => {
                validate_receipt(&expected_receipt, &receipt)?;
                Ok(LegacyMirrorOutcome::Mirrored(receipt))
            }
            Ok(LegacyProjectionApplyOutcome::AlreadyApplied(receipt)) => {
                validate_receipt(&expected_receipt, &receipt)?;
                Ok(LegacyMirrorOutcome::Duplicate(receipt))
            }
            Ok(LegacyProjectionApplyOutcome::Delayed {
                last_applied_version,
            }) => {
                if last_applied_version.is_some_and(|version| version >= item.authoritative_version)
                {
                    return Err(LegacyMirrorError::InvalidBackendResponse);
                }
                Ok(LegacyMirrorOutcome::Delayed {
                    item: expected_receipt,
                    last_applied_version,
                })
            }
            Err(LegacyProjectionWriteError::Unavailable) => {
                Err(LegacyMirrorError::ProjectionUnavailable)
            }
            Err(LegacyProjectionWriteError::OutcomeUnknown) => {
                Err(LegacyMirrorError::OutcomeUnknown)
            }
            Err(LegacyProjectionWriteError::LegacyAhead) => {
                Err(LegacyMirrorError::ReverseReconciliationForbidden)
            }
            Err(LegacyProjectionWriteError::ConflictingProjection) => {
                Err(LegacyMirrorError::ProjectionConflict)
            }
            Err(LegacyProjectionWriteError::InvalidData) => {
                Err(LegacyMirrorError::InvalidBackendResponse)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyMirrorError {
    InvalidInput,
    TenantBoundaryViolation,
    CheckpointMismatch,
    MirrorNotPermitted,
    ProjectionUnavailable,
    OutcomeUnknown,
    ReverseReconciliationForbidden,
    ProjectionConflict,
    InvalidBackendResponse,
}

impl fmt::Display for LegacyMirrorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidInput => "legacy mirror input is invalid",
            Self::TenantBoundaryViolation => "legacy mirror crossed its tenant boundary",
            Self::CheckpointMismatch => "legacy mirror checkpoint does not match its item",
            Self::MirrorNotPermitted => "legacy mirroring is not permitted for this checkpoint",
            Self::ProjectionUnavailable => "legacy mirror projection is unavailable",
            Self::OutcomeUnknown => "legacy mirror projection outcome is unknown",
            Self::ReverseReconciliationForbidden => {
                "legacy state cannot overwrite canonical authority"
            }
            Self::ProjectionConflict => "legacy mirror projection diverged",
            Self::InvalidBackendResponse => "legacy mirror writer returned invalid data",
        })
    }
}

impl Error for LegacyMirrorError {}

fn validate_boundary(
    tenant: &TenantContext,
    checkpoint: &CutoverCheckpoint,
    item: &CanonicalOutboxMirrorItem,
) -> Result<(), LegacyMirrorError> {
    if tenant.community_id() != item.aggregate.community_id() {
        return Err(LegacyMirrorError::TenantBoundaryViolation);
    }
    if checkpoint.aggregate() != item.aggregate {
        return Err(LegacyMirrorError::CheckpointMismatch);
    }
    if checkpoint.authority() != CutoverAuthority::Canonical
        || checkpoint.phase() < CutoverPhase::CommunicationWriteCutover
        || checkpoint.phase() > CutoverPhase::WorkflowInfrastructureCutover
    {
        return Err(LegacyMirrorError::MirrorNotPermitted);
    }
    Ok(())
}

fn validate_receipt(
    expected: &LegacyMirrorReceipt,
    actual: &LegacyMirrorReceipt,
) -> Result<(), LegacyMirrorError> {
    if actual != expected {
        return Err(LegacyMirrorError::InvalidBackendResponse);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Mutex};

    use collaboration_domain::{
        AggregateId, AggregateType, CommunityId, Provenance, SourceRecordId, TrustedTenantRoute,
    };
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
                TrustedTenantRoute::from_listener(community_id, "legacy-mirror")
                    .expect("trusted tenant route"),
            ),
            &[],
        )
        .expect("tenant context")
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
            reversible_boundary_label: Some("before-legacy-mirror".to_string()),
        };
        let CutoverTransitionOutcome::Advanced(checkpoint) = checkpoint
            .transition(&transition)
            .expect("canonical transition")
        else {
            panic!("authority must advance");
        };
        checkpoint
    }

    fn item(
        community_id: CommunityId,
        sequence: u64,
        version: u64,
        operation: u128,
        payload: u8,
    ) -> CanonicalOutboxMirrorItem {
        let operation_id = OperationId::from_uuid(Uuid::from_u128(operation));
        let provenance = Provenance::new(
            SourceSystem::Zed,
            SourceRecordId::new("conversation:100").expect("source ID"),
            1_900_000_000_000,
        )
        .with_source_version(version.to_string());
        let operation = OutboxOperation::new("conversation.accepted", provenance, vec![payload])
            .expect("outbox operation");
        CanonicalOutboxMirrorItem::new(
            aggregate(community_id),
            sequence,
            operation_id,
            AggregateVersion::new(version).expect("version"),
            operation,
        )
        .expect("mirror item")
    }

    #[derive(Default)]
    struct TestWriterState {
        receipts: BTreeMap<OperationId, LegacyMirrorReceipt>,
        current_version: Option<AggregateVersion>,
        unavailable_once: bool,
        legacy_ahead: bool,
        writes: u64,
    }

    #[derive(Default)]
    struct TestWriter {
        state: Mutex<TestWriterState>,
    }

    impl TestWriter {
        fn unavailable_once() -> Self {
            Self {
                state: Mutex::new(TestWriterState {
                    unavailable_once: true,
                    ..TestWriterState::default()
                }),
            }
        }

        fn legacy_ahead(version: AggregateVersion) -> Self {
            Self {
                state: Mutex::new(TestWriterState {
                    current_version: Some(version),
                    legacy_ahead: true,
                    ..TestWriterState::default()
                }),
            }
        }

        fn writes(&self) -> u64 {
            self.state.lock().expect("writer lock").writes
        }
    }

    #[async_trait]
    impl LegacyProjectionWriter for TestWriter {
        async fn apply_canonical(
            &self,
            _tenant: &TenantContext,
            item: &CanonicalOutboxMirrorItem,
        ) -> Result<LegacyProjectionApplyOutcome, LegacyProjectionWriteError> {
            let mut state = self
                .state
                .lock()
                .map_err(|_| LegacyProjectionWriteError::Unavailable)?;
            if state.unavailable_once {
                state.unavailable_once = false;
                return Err(LegacyProjectionWriteError::Unavailable);
            }
            if let Some(receipt) = state.receipts.get(&item.operation_id()) {
                return if receipt == &item.expected_receipt() {
                    Ok(LegacyProjectionApplyOutcome::AlreadyApplied(
                        receipt.clone(),
                    ))
                } else {
                    Err(LegacyProjectionWriteError::ConflictingProjection)
                };
            }
            if state.legacy_ahead {
                return Err(LegacyProjectionWriteError::LegacyAhead);
            }
            let expected = match state.current_version {
                Some(version) => version.next(),
                None => Some(AggregateVersion::FIRST),
            };
            if expected != Some(item.authoritative_version()) {
                return Ok(LegacyProjectionApplyOutcome::Delayed {
                    last_applied_version: state.current_version,
                });
            }
            let receipt = item.expected_receipt();
            state.current_version = Some(item.authoritative_version());
            state.receipts.insert(item.operation_id(), receipt.clone());
            state.writes = state
                .writes
                .checked_add(1)
                .ok_or(LegacyProjectionWriteError::InvalidData)?;
            Ok(LegacyProjectionApplyOutcome::Applied(receipt))
        }
    }

    #[tokio::test]
    async fn unavailable_retry_and_exact_duplicate_write_once() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id);
        let mirror = LegacyMirror::new(TestWriter::unavailable_once());
        let accepted = item(community_id, 10, 1, 1000, 42);

        assert!(matches!(
            mirror
                .mirror_one(&tenant, &checkpoint, accepted.clone())
                .await,
            Err(LegacyMirrorError::ProjectionUnavailable)
        ));
        assert!(matches!(
            mirror
                .mirror_one(&tenant, &checkpoint, accepted.clone())
                .await
                .expect("retry"),
            LegacyMirrorOutcome::Mirrored(_)
        ));
        assert!(matches!(
            mirror
                .mirror_one(&tenant, &checkpoint, accepted)
                .await
                .expect("duplicate"),
            LegacyMirrorOutcome::Duplicate(_)
        ));
        assert_eq!(mirror.writer.writes(), 1);
    }

    #[tokio::test]
    async fn delayed_projection_waits_for_predecessor_then_converges() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id);
        let mirror = LegacyMirror::new(TestWriter::default());
        let first = item(community_id, 10, 1, 1000, 41);
        let second = item(community_id, 20, 2, 1001, 42);

        assert!(matches!(
            mirror
                .mirror_one(&tenant, &checkpoint, second.clone())
                .await
                .expect("delay"),
            LegacyMirrorOutcome::Delayed {
                last_applied_version: None,
                ..
            }
        ));
        assert!(matches!(
            mirror
                .mirror_one(&tenant, &checkpoint, first)
                .await
                .expect("first"),
            LegacyMirrorOutcome::Mirrored(_)
        ));
        assert!(matches!(
            mirror
                .mirror_one(&tenant, &checkpoint, second)
                .await
                .expect("second"),
            LegacyMirrorOutcome::Mirrored(_)
        ));
        assert_eq!(mirror.writer.writes(), 2);
    }

    #[tokio::test]
    async fn stable_operation_reuse_with_changed_source_conflicts() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id);
        let mirror = LegacyMirror::new(TestWriter::default());
        let first = item(community_id, 10, 1, 1000, 41);
        let changed = item(community_id, 11, 1, 1000, 42);
        assert!(matches!(
            mirror
                .mirror_one(&tenant, &checkpoint, first)
                .await
                .expect("first"),
            LegacyMirrorOutcome::Mirrored(_)
        ));
        assert!(matches!(
            mirror.mirror_one(&tenant, &checkpoint, changed).await,
            Err(LegacyMirrorError::ProjectionConflict)
        ));
        assert_eq!(mirror.writer.writes(), 1);
    }

    #[tokio::test]
    async fn legacy_ahead_never_flows_back_to_canonical_authority() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = canonical_checkpoint(community_id);
        let mirror = LegacyMirror::new(TestWriter::legacy_ahead(
            AggregateVersion::new(2).expect("version"),
        ));
        assert!(matches!(
            mirror
                .mirror_one(&tenant, &checkpoint, item(community_id, 10, 1, 1000, 41),)
                .await,
            Err(LegacyMirrorError::ReverseReconciliationForbidden)
        ));
        assert_eq!(checkpoint.authority(), CutoverAuthority::Canonical);
        assert_eq!(mirror.writer.writes(), 0);
    }

    #[tokio::test]
    async fn tenant_and_phase_reject_before_legacy_projection() {
        let community_id = community(1);
        let legacy_checkpoint = CutoverCheckpoint::new(
            aggregate(community_id),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("legacy checkpoint");
        let mirror = LegacyMirror::new(TestWriter::default());
        let accepted = item(community_id, 10, 1, 1000, 41);
        assert!(matches!(
            mirror
                .mirror_one(&tenant(community(2)), &legacy_checkpoint, accepted.clone(),)
                .await,
            Err(LegacyMirrorError::TenantBoundaryViolation)
        ));
        assert!(matches!(
            mirror
                .mirror_one(&tenant(community_id), &legacy_checkpoint, accepted,)
                .await,
            Err(LegacyMirrorError::MirrorNotPermitted)
        ));
        assert_eq!(mirror.writer.writes(), 0);
    }
}
