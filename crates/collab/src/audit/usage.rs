use std::fmt;

use collaboration_domain::{
    AuditAction, AuditEntry, AuditError, AuditField, AuditFieldName, AuditFields, AuditHash,
    AuditIdentifier, AuditOutcome, AuditRecord, AuditValue, CommunityId, OperationId,
    TenantContext,
};
use sha2::{Digest as _, Sha256};
use uuid::Uuid;

use super::repository::{
    AuditExportCursor, AuditHead, AuditRepository, AuditRepositoryError, ExpectedAuditHead,
};

const USAGE_SOURCE_DOMAIN: &[u8] = b"zed.collaboration.usage.source.v1";
const MAX_USAGE_DURATION_MILLIS: u64 = 7 * 24 * 60 * 60 * 1_000;
const MAX_WORKFLOW_STEPS: u32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClientUsageTelemetryIntegration {
    DisabledByDesign,
}

pub const CLIENT_USAGE_TELEMETRY_INTEGRATION: ClientUsageTelemetryIntegration =
    ClientUsageTelemetryIntegration::DisabledByDesign;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UsageKind {
    AgentJob,
    AgentTurn,
    WorkflowRun,
}

impl UsageKind {
    const fn code(self) -> u8 {
        match self {
            Self::AgentJob => 1,
            Self::AgentTurn => 2,
            Self::WorkflowRun => 3,
        }
    }

    const fn action(self) -> &'static str {
        match self {
            Self::AgentJob => "usage.agent_job",
            Self::AgentTurn => "usage.agent_turn",
            Self::WorkflowRun => "usage.workflow_run",
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct UsageSourceReference {
    community_id: CommunityId,
    kind: UsageKind,
    digest: [u8; 32],
}

impl UsageSourceReference {
    pub fn agent_job(tenant: &TenantContext, job_id: Uuid) -> Result<Self, UsageAccountingError> {
        if job_id.is_nil() {
            return Err(UsageAccountingError::InvalidRecord);
        }
        Ok(Self::from_source(
            tenant.community_id(),
            UsageKind::AgentJob,
            job_id.as_bytes(),
        ))
    }

    pub fn agent_turn(
        tenant: &TenantContext,
        nip_am_event_id: [u8; 32],
    ) -> Result<Self, UsageAccountingError> {
        if nip_am_event_id == [0; 32] {
            return Err(UsageAccountingError::InvalidRecord);
        }
        Ok(Self::from_source(
            tenant.community_id(),
            UsageKind::AgentTurn,
            &nip_am_event_id,
        ))
    }

    pub fn workflow_run(
        tenant: &TenantContext,
        run_id: Uuid,
    ) -> Result<Self, UsageAccountingError> {
        if run_id.is_nil() {
            return Err(UsageAccountingError::InvalidRecord);
        }
        Ok(Self::from_source(
            tenant.community_id(),
            UsageKind::WorkflowRun,
            run_id.as_bytes(),
        ))
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn digest(self) -> [u8; 32] {
        self.digest
    }

    fn from_source(community_id: CommunityId, kind: UsageKind, source: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(USAGE_SOURCE_DOMAIN);
        hasher.update(community_id.as_uuid().as_bytes());
        hasher.update([kind.code()]);
        hasher.update(source);
        Self {
            community_id,
            kind,
            digest: hasher.finalize().into(),
        }
    }

    const fn from_digest(community_id: CommunityId, kind: UsageKind, digest: [u8; 32]) -> Self {
        Self {
            community_id,
            kind,
            digest,
        }
    }
}

impl fmt::Debug for UsageSourceReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UsageSourceReference")
            .field("kind", &self.kind)
            .field("digest", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UsageDetails {
    AgentJob {
        attempt: u32,
        duration_millis: u64,
    },
    AgentTurn,
    WorkflowRun {
        step_count: u32,
        duration_millis: u64,
    },
}

impl UsageDetails {
    const fn kind(self) -> UsageKind {
        match self {
            Self::AgentJob { .. } => UsageKind::AgentJob,
            Self::AgentTurn => UsageKind::AgentTurn,
            Self::WorkflowRun { .. } => UsageKind::WorkflowRun,
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct UsageRecord {
    community_id: CommunityId,
    operation_id: OperationId,
    source: UsageSourceReference,
    details: UsageDetails,
    outcome: AuditOutcome,
    occurred_at_millis: u64,
}

impl fmt::Debug for UsageRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UsageRecord")
            .field("community_id", &"<redacted>")
            .field("operation_id", &"<redacted>")
            .field("source", &self.source)
            .field("details", &self.details)
            .field("outcome", &self.outcome)
            .field("occurred_at_millis", &self.occurred_at_millis)
            .finish()
    }
}

impl UsageRecord {
    pub fn agent_job(
        tenant: &TenantContext,
        operation_id: OperationId,
        source: UsageSourceReference,
        attempt: u32,
        duration_millis: u64,
        outcome: AuditOutcome,
        occurred_at_millis: u64,
    ) -> Result<Self, UsageAccountingError> {
        Self::new(
            tenant,
            operation_id,
            source,
            UsageDetails::AgentJob {
                attempt,
                duration_millis,
            },
            outcome,
            occurred_at_millis,
        )
    }

    pub fn agent_turn(
        tenant: &TenantContext,
        operation_id: OperationId,
        source: UsageSourceReference,
        outcome: AuditOutcome,
        occurred_at_millis: u64,
    ) -> Result<Self, UsageAccountingError> {
        Self::new(
            tenant,
            operation_id,
            source,
            UsageDetails::AgentTurn,
            outcome,
            occurred_at_millis,
        )
    }

    pub fn workflow_run(
        tenant: &TenantContext,
        operation_id: OperationId,
        source: UsageSourceReference,
        step_count: u32,
        duration_millis: u64,
        outcome: AuditOutcome,
        occurred_at_millis: u64,
    ) -> Result<Self, UsageAccountingError> {
        Self::new(
            tenant,
            operation_id,
            source,
            UsageDetails::WorkflowRun {
                step_count,
                duration_millis,
            },
            outcome,
            occurred_at_millis,
        )
    }

    fn new(
        tenant: &TenantContext,
        operation_id: OperationId,
        source: UsageSourceReference,
        details: UsageDetails,
        outcome: AuditOutcome,
        occurred_at_millis: u64,
    ) -> Result<Self, UsageAccountingError> {
        if source.community_id != tenant.community_id()
            || source.kind != details.kind()
            || operation_id.as_uuid().is_nil()
            || occurred_at_millis == 0
            || !valid_details(details)
        {
            return Err(UsageAccountingError::InvalidRecord);
        }
        Ok(Self {
            community_id: tenant.community_id(),
            operation_id,
            source,
            details,
            outcome,
            occurred_at_millis,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn source(&self) -> UsageSourceReference {
        self.source
    }

    pub const fn details(&self) -> UsageDetails {
        self.details
    }

    pub const fn outcome(&self) -> AuditOutcome {
        self.outcome
    }

    pub const fn occurred_at_millis(&self) -> u64 {
        self.occurred_at_millis
    }

    fn to_audit_record(&self) -> Result<AuditRecord, UsageAccountingError> {
        let mut fields = vec![identifier_field("source_digest", self.source.digest)?];
        match self.details {
            UsageDetails::AgentJob {
                attempt,
                duration_millis,
            } => {
                fields.push(unsigned_field("attempt", u64::from(attempt))?);
                fields.push(unsigned_field("duration_millis", duration_millis)?);
            }
            UsageDetails::AgentTurn => {}
            UsageDetails::WorkflowRun {
                step_count,
                duration_millis,
            } => {
                fields.push(unsigned_field("duration_millis", duration_millis)?);
                fields.push(unsigned_field("step_count", u64::from(step_count))?);
            }
        }
        AuditRecord::new(
            self.operation_id,
            AuditAction::new(self.details.kind().action())?,
            None,
            self.outcome,
            self.occurred_at_millis,
            AuditFields::new(fields)?,
        )
        .map_err(Into::into)
    }

    fn from_entry(entry: &AuditEntry) -> Result<Option<Self>, UsageAccountingError> {
        let kind = match entry.record().action().as_str() {
            "usage.agent_job" => UsageKind::AgentJob,
            "usage.agent_turn" => UsageKind::AgentTurn,
            "usage.workflow_run" => UsageKind::WorkflowRun,
            _ => return Ok(None),
        };
        if entry.record().actor_principal_id().is_some() {
            return Err(UsageAccountingError::InvalidStoredRecord);
        }
        let fields = entry.record().fields().as_slice();
        let digest = digest_field(fields, "source_digest")?;
        let details = match kind {
            UsageKind::AgentJob if fields.len() == 3 => UsageDetails::AgentJob {
                attempt: u32_field(fields, "attempt")?,
                duration_millis: unsigned_value(fields, "duration_millis")?,
            },
            UsageKind::AgentTurn if fields.len() == 1 => UsageDetails::AgentTurn,
            UsageKind::WorkflowRun if fields.len() == 3 => UsageDetails::WorkflowRun {
                step_count: u32_field(fields, "step_count")?,
                duration_millis: unsigned_value(fields, "duration_millis")?,
            },
            _ => return Err(UsageAccountingError::InvalidStoredRecord),
        };
        let source = UsageSourceReference::from_digest(entry.community_id(), kind, digest);
        let record = Self {
            community_id: entry.community_id(),
            operation_id: entry.record().operation_id(),
            source,
            details,
            outcome: entry.record().outcome(),
            occurred_at_millis: entry.record().occurred_at_millis(),
        };
        valid_details(record.details)
            .then_some(Some(record))
            .ok_or(UsageAccountingError::InvalidStoredRecord)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UsageWriteOutcome {
    Applied(AuditEntry),
    AlreadyRecorded(AuditEntry),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UsageExportRecord {
    sequence: u64,
    hash: AuditHash,
    record: UsageRecord,
}

impl UsageExportRecord {
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn hash(&self) -> AuditHash {
        self.hash
    }

    pub const fn record(&self) -> &UsageRecord {
        &self.record
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UsageExportBatch {
    records: Vec<UsageExportRecord>,
    scanned_through: Option<AuditHead>,
}

impl UsageExportBatch {
    pub fn records(&self) -> &[UsageExportRecord] {
        &self.records
    }

    pub const fn scanned_through(&self) -> Option<AuditHead> {
        self.scanned_through
    }

    pub fn aggregate(&self) -> Result<UsageAggregate, UsageAccountingError> {
        UsageAggregate::from_records(self.records.iter().map(UsageExportRecord::record))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UsageAggregate {
    pub agent_jobs: u64,
    pub agent_turns: u64,
    pub workflow_runs: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub denied: u64,
    pub cancelled: u64,
    pub agent_job_duration_millis: u64,
    pub workflow_duration_millis: u64,
    pub workflow_steps: u64,
}

impl UsageAggregate {
    fn from_records<'a>(
        records: impl IntoIterator<Item = &'a UsageRecord>,
    ) -> Result<Self, UsageAccountingError> {
        let mut aggregate = Self::default();
        for record in records {
            match record.details {
                UsageDetails::AgentJob {
                    duration_millis, ..
                } => {
                    checked_increment(&mut aggregate.agent_jobs, 1)?;
                    checked_increment(&mut aggregate.agent_job_duration_millis, duration_millis)?;
                }
                UsageDetails::AgentTurn => checked_increment(&mut aggregate.agent_turns, 1)?,
                UsageDetails::WorkflowRun {
                    step_count,
                    duration_millis,
                } => {
                    checked_increment(&mut aggregate.workflow_runs, 1)?;
                    checked_increment(&mut aggregate.workflow_steps, u64::from(step_count))?;
                    checked_increment(&mut aggregate.workflow_duration_millis, duration_millis)?;
                }
            }
            let outcome = match record.outcome {
                AuditOutcome::Succeeded => &mut aggregate.succeeded,
                AuditOutcome::Failed => &mut aggregate.failed,
                AuditOutcome::Denied => &mut aggregate.denied,
                AuditOutcome::Cancelled => &mut aggregate.cancelled,
            };
            checked_increment(outcome, 1)?;
        }
        Ok(aggregate)
    }
}

pub struct UsageAccounting {
    repository: AuditRepository,
}

impl UsageAccounting {
    pub const fn new(repository: AuditRepository) -> Self {
        Self { repository }
    }

    pub fn into_repository(self) -> AuditRepository {
        self.repository
    }

    pub async fn record(
        &self,
        tenant: &TenantContext,
        expected_head: ExpectedAuditHead,
        record: UsageRecord,
    ) -> Result<UsageWriteOutcome, UsageAccountingError> {
        if record.community_id != tenant.community_id() {
            return Err(UsageAccountingError::TenantBoundaryViolation);
        }
        if let Some(existing) = self.find_operation(tenant, record.operation_id).await? {
            return compare_existing(existing, record);
        }
        let audit_record = record.to_audit_record()?;
        match self
            .repository
            .append(tenant, expected_head, audit_record)
            .await
        {
            Ok(entry) => Ok(UsageWriteOutcome::Applied(entry)),
            Err(stale @ AuditRepositoryError::StaleHead) => {
                match self.find_operation(tenant, record.operation_id).await? {
                    Some(existing) => compare_existing(existing, record),
                    None => Err(UsageAccountingError::Repository(stale)),
                }
            }
            Err(error) => Err(error.into()),
        }
    }

    pub async fn export_segment(
        &self,
        tenant: &TenantContext,
        cursor: Option<&AuditExportCursor>,
        limit: u32,
    ) -> Result<UsageExportBatch, UsageAccountingError> {
        let segment = self
            .repository
            .export_segment(tenant, cursor, limit)
            .await?;
        let scanned_through = segment.end_head();
        let mut records = Vec::new();
        for entry in segment.entries() {
            if let Some(record) = UsageRecord::from_entry(entry)? {
                records.push(UsageExportRecord {
                    sequence: entry.sequence(),
                    hash: entry.hash(),
                    record,
                });
            }
        }
        Ok(UsageExportBatch {
            records,
            scanned_through,
        })
    }

    pub async fn load_export_cursor(
        &self,
        tenant: &TenantContext,
        exporter_id: &str,
    ) -> Result<Option<AuditExportCursor>, UsageAccountingError> {
        self.repository
            .load_export_cursor(tenant, exporter_id)
            .await
            .map_err(Into::into)
    }

    pub async fn advance_export_cursor(
        &self,
        tenant: &TenantContext,
        exporter_id: &str,
        expected: Option<&AuditExportCursor>,
        batch: &UsageExportBatch,
    ) -> Result<AuditExportCursor, UsageAccountingError> {
        let scanned_through = batch
            .scanned_through
            .ok_or(UsageAccountingError::EmptyExportBatch)?;
        self.repository
            .advance_export_cursor(tenant, exporter_id, expected, scanned_through)
            .await
            .map_err(Into::into)
    }

    async fn find_operation(
        &self,
        tenant: &TenantContext,
        operation_id: OperationId,
    ) -> Result<Option<AuditEntry>, UsageAccountingError> {
        self.repository
            .load_operation(tenant, operation_id)
            .await
            .map_err(Into::into)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UsageAccountingError {
    #[error("usage accounting request crossed its admitted community boundary")]
    TenantBoundaryViolation,
    #[error("usage accounting record is invalid")]
    InvalidRecord,
    #[error("stored usage accounting record is invalid")]
    InvalidStoredRecord,
    #[error("usage operation ID was reused for different accounting data")]
    DuplicateConflict,
    #[error("usage export batch did not scan an audit entry")]
    EmptyExportBatch,
    #[error("usage aggregation overflowed")]
    AggregateOverflow,
    #[error("usage record violates the canonical audit contract")]
    Domain(#[from] AuditError),
    #[error("usage accounting repository is unavailable")]
    Repository(#[from] AuditRepositoryError),
}

fn compare_existing(
    entry: AuditEntry,
    expected: UsageRecord,
) -> Result<UsageWriteOutcome, UsageAccountingError> {
    match UsageRecord::from_entry(&entry)? {
        Some(stored) if stored == expected => Ok(UsageWriteOutcome::AlreadyRecorded(entry)),
        _ => Err(UsageAccountingError::DuplicateConflict),
    }
}

fn valid_details(details: UsageDetails) -> bool {
    match details {
        UsageDetails::AgentJob {
            attempt,
            duration_millis,
        } => attempt > 0 && duration_millis <= MAX_USAGE_DURATION_MILLIS,
        UsageDetails::AgentTurn => true,
        UsageDetails::WorkflowRun {
            step_count,
            duration_millis,
        } => {
            step_count > 0
                && step_count <= MAX_WORKFLOW_STEPS
                && duration_millis <= MAX_USAGE_DURATION_MILLIS
        }
    }
}

fn identifier_field(
    name: &'static str,
    value: [u8; 32],
) -> Result<AuditField, UsageAccountingError> {
    Ok(AuditField::new(
        AuditFieldName::new(name)?,
        AuditValue::Identifier(AuditIdentifier::new(hex::encode(value))?),
    ))
}

fn unsigned_field(name: &'static str, value: u64) -> Result<AuditField, UsageAccountingError> {
    Ok(AuditField::new(
        AuditFieldName::new(name)?,
        AuditValue::Unsigned(value),
    ))
}

fn field_value<'a>(
    fields: &'a [AuditField],
    name: &str,
) -> Result<&'a AuditValue, UsageAccountingError> {
    fields
        .iter()
        .find(|field| field.name().as_str() == name)
        .map(AuditField::value)
        .ok_or(UsageAccountingError::InvalidStoredRecord)
}

fn digest_field(fields: &[AuditField], name: &str) -> Result<[u8; 32], UsageAccountingError> {
    let AuditValue::Identifier(value) = field_value(fields, name)? else {
        return Err(UsageAccountingError::InvalidStoredRecord);
    };
    let bytes =
        hex::decode(value.as_str()).map_err(|_| UsageAccountingError::InvalidStoredRecord)?;
    bytes
        .try_into()
        .map_err(|_| UsageAccountingError::InvalidStoredRecord)
}

fn unsigned_value(fields: &[AuditField], name: &str) -> Result<u64, UsageAccountingError> {
    let AuditValue::Unsigned(value) = field_value(fields, name)? else {
        return Err(UsageAccountingError::InvalidStoredRecord);
    };
    Ok(*value)
}

fn u32_field(fields: &[AuditField], name: &str) -> Result<u32, UsageAccountingError> {
    u32::try_from(unsigned_value(fields, name)?)
        .map_err(|_| UsageAccountingError::InvalidStoredRecord)
}

fn checked_increment(target: &mut u64, value: u64) -> Result<(), UsageAccountingError> {
    *target = target
        .checked_add(value)
        .ok_or(UsageAccountingError::AggregateOverflow)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use collaboration_domain::{
        AuditChainPosition, AuditField, AuditRedaction, PrincipalId, TrustedTenantRoute,
    };
    use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};

    use super::*;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn tenant(value: u128) -> TenantContext {
        let community_id = community(value);
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "usage-accounting")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn repository(
        query_results: Vec<Vec<BTreeMap<String, SeaValue>>>,
        affected_rows: &[u64],
    ) -> AuditRepository {
        let database = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results(query_results)
            .append_exec_results(affected_rows.iter().copied().map(|rows_affected| {
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected,
                }
            }))
            .into_connection();
        AuditRepository::new(database).expect("repository")
    }

    fn tenant_setting_row(tenant: &TenantContext) -> BTreeMap<String, SeaValue> {
        BTreeMap::from([(
            "community_id".into(),
            tenant.community_id().to_string().into(),
        )])
    }

    fn job_record(tenant: &TenantContext, operation: u128, duration: u64) -> UsageRecord {
        UsageRecord::agent_job(
            tenant,
            OperationId::from_uuid(Uuid::from_u128(operation)),
            UsageSourceReference::agent_job(tenant, Uuid::from_u128(100)).expect("job source"),
            1,
            duration,
            AuditOutcome::Succeeded,
            1_900_000_000_000 + u64::try_from(operation).expect("fixture operation"),
        )
        .expect("job record")
    }

    fn entry(position: AuditChainPosition, record: UsageRecord) -> AuditEntry {
        AuditEntry::append(position, record.to_audit_record().expect("audit record"))
            .expect("audit entry")
    }

    fn entry_row(entry: &AuditEntry) -> BTreeMap<String, SeaValue> {
        let bridge = entry.chain_bridge();
        BTreeMap::from([
            ("community_id".into(), entry.community_id().as_uuid().into()),
            ("sequence_text".into(), entry.sequence().to_string().into()),
            ("entry_hash".into(), entry.hash().as_bytes().to_vec().into()),
            (
                "previous_hash".into(),
                entry
                    .previous_hash()
                    .map(|hash| hash.as_bytes().to_vec())
                    .into(),
            ),
            (
                "operation_id".into(),
                entry.record().operation_id().as_uuid().into(),
            ),
            (
                "action".into(),
                entry.record().action().as_str().to_owned().into(),
            ),
            (
                "actor_principal_id".into(),
                entry
                    .record()
                    .actor_principal_id()
                    .map(PrincipalId::as_uuid)
                    .into(),
            ),
            (
                "outcome".into(),
                outcome_name(entry.record().outcome()).to_owned().into(),
            ),
            (
                "occurred_at_millis_text".into(),
                entry.record().occurred_at_millis().to_string().into(),
            ),
            (
                "fields_json".into(),
                serde_json::to_string(entry.record().fields())
                    .expect("fields JSON")
                    .into(),
            ),
            (
                "bridge_source".into(),
                bridge.map(|_| "buzz_v1".to_owned()).into(),
            ),
            (
                "bridge_source_sequence_text".into(),
                bridge
                    .map(|bridge| bridge.source_sequence().to_string())
                    .into(),
            ),
            (
                "bridge_source_head".into(),
                bridge
                    .map(|bridge| bridge.source_head().as_bytes().to_vec())
                    .into(),
            ),
        ])
    }

    fn head_row(head: AuditHead) -> BTreeMap<String, SeaValue> {
        BTreeMap::from([
            ("sequence_text".into(), head.sequence().to_string().into()),
            ("entry_hash".into(), head.hash().as_bytes().to_vec().into()),
        ])
    }

    fn outcome_name(outcome: AuditOutcome) -> &'static str {
        match outcome {
            AuditOutcome::Succeeded => "succeeded",
            AuditOutcome::Failed => "failed",
            AuditOutcome::Denied => "denied",
            AuditOutcome::Cancelled => "cancelled",
        }
    }

    #[tokio::test]
    async fn exact_retry_is_deduplicated_and_conflicting_reuse_is_rejected() {
        let tenant = tenant(1);
        let record = job_record(&tenant, 10, 50);
        let expected_entry = entry(
            AuditChainPosition::genesis(tenant.community_id()).expect("genesis"),
            record.clone(),
        );
        let first = UsageAccounting::new(repository(
            vec![vec![tenant_setting_row(&tenant)], vec![], vec![]],
            &[1, 1, 1, 1, 1],
        ));
        let applied = first
            .record(&tenant, ExpectedAuditHead::Empty, record.clone())
            .await
            .expect("first write");
        assert!(matches!(applied, UsageWriteOutcome::Applied(entry) if entry == expected_entry));

        let retry = UsageAccounting::new(repository(
            vec![
                vec![tenant_setting_row(&tenant)],
                vec![entry_row(&expected_entry)],
            ],
            &[1],
        ));
        let duplicate = retry
            .record(&tenant, ExpectedAuditHead::Empty, record.clone())
            .await
            .expect("exact retry");
        assert!(
            matches!(duplicate, UsageWriteOutcome::AlreadyRecorded(entry) if entry == expected_entry)
        );
        let retry_log = format!(
            "{:#?}",
            retry
                .into_repository()
                .into_connection()
                .into_transaction_log()
        );
        assert!(retry_log.contains("operation_id = $2"));
        assert!(!retry_log.contains("INSERT INTO public.collaboration_audit_entries"));

        let racing_retry = UsageAccounting::new(repository(
            vec![
                vec![tenant_setting_row(&tenant)],
                vec![],
                vec![head_row(AuditHead::from_entry(&expected_entry))],
                vec![tenant_setting_row(&tenant)],
                vec![entry_row(&expected_entry)],
            ],
            &[1, 1, 1, 1],
        ));
        assert!(matches!(
            racing_retry
                .record(&tenant, ExpectedAuditHead::Empty, record.clone())
                .await,
            Ok(UsageWriteOutcome::AlreadyRecorded(entry)) if entry == expected_entry
        ));
        let racing_log = format!(
            "{:#?}",
            racing_retry
                .into_repository()
                .into_connection()
                .into_transaction_log()
        );
        assert!(!racing_log.contains("INSERT INTO public.collaboration_audit_entries"));

        let conflict = UsageAccounting::new(repository(
            vec![
                vec![tenant_setting_row(&tenant)],
                vec![entry_row(&expected_entry)],
            ],
            &[1],
        ));
        assert!(matches!(
            conflict
                .record(
                    &tenant,
                    ExpectedAuditHead::Empty,
                    job_record(&tenant, 10, 51)
                )
                .await,
            Err(UsageAccountingError::DuplicateConflict)
        ));
    }

    #[test]
    fn private_agent_turn_projects_only_a_tenant_bound_digest() {
        let tenant = tenant(1);
        let event_id = [7; 32];
        let source = UsageSourceReference::agent_turn(&tenant, event_id).expect("turn source");
        let record = UsageRecord::agent_turn(
            &tenant,
            OperationId::from_uuid(Uuid::from_u128(20)),
            source,
            AuditOutcome::Succeeded,
            1_900_000_000_020,
        )
        .expect("turn record");
        let audit = record.to_audit_record().expect("audit record");
        assert_eq!(audit.action().as_str(), "usage.agent_turn");
        assert_eq!(audit.actor_principal_id(), None);
        assert_eq!(audit.fields().as_slice().len(), 1);
        assert_eq!(
            audit.fields().as_slice()[0].name().as_str(),
            "source_digest"
        );
        let serialized = serde_json::to_string(audit.fields()).expect("fields JSON");
        assert!(!serialized.contains(&hex::encode(event_id)));
        assert!(!format!("{source:?}").contains(&hex::encode(source.digest())));
        let rendered = format!("{record:?}");
        for private_value in [
            tenant.community_id().to_string(),
            record.operation_id().to_string(),
            hex::encode(source.digest()),
            hex::encode(event_id),
        ] {
            assert!(!rendered.contains(&private_value));
        }
        assert_eq!(
            CLIENT_USAGE_TELEMETRY_INTEGRATION,
            ClientUsageTelemetryIntegration::DisabledByDesign
        );
    }

    #[tokio::test]
    async fn export_filters_non_usage_entries_and_aggregates_closed_counters() {
        let tenant = tenant(1);
        let job = job_record(&tenant, 30, 50);
        let first = entry(
            AuditChainPosition::genesis(tenant.community_id()).expect("genesis"),
            job,
        );
        let unrelated = AuditEntry::append(
            AuditChainPosition::after(&first).expect("second position"),
            AuditRecord::new(
                OperationId::from_uuid(Uuid::from_u128(31)),
                AuditAction::new("auth.authenticate").expect("action"),
                Some(PrincipalId::from_uuid(Uuid::from_u128(3))),
                AuditOutcome::Succeeded,
                1_900_000_000_031,
                AuditFields::new(vec![AuditField::new(
                    AuditFieldName::new("credential_detail").expect("field name"),
                    AuditValue::Redacted(AuditRedaction::Credential),
                )])
                .expect("fields"),
            )
            .expect("record"),
        )
        .expect("unrelated entry");
        let turn_record = UsageRecord::agent_turn(
            &tenant,
            OperationId::from_uuid(Uuid::from_u128(32)),
            UsageSourceReference::agent_turn(&tenant, [8; 32]).expect("turn source"),
            AuditOutcome::Failed,
            1_900_000_000_032,
        )
        .expect("turn record");
        let turn = entry(
            AuditChainPosition::after(&unrelated).expect("third position"),
            turn_record,
        );
        let workflow_record = UsageRecord::workflow_run(
            &tenant,
            OperationId::from_uuid(Uuid::from_u128(33)),
            UsageSourceReference::workflow_run(&tenant, Uuid::from_u128(130))
                .expect("workflow source"),
            3,
            70,
            AuditOutcome::Succeeded,
            1_900_000_000_033,
        )
        .expect("workflow record");
        let workflow = entry(
            AuditChainPosition::after(&turn).expect("fourth position"),
            workflow_record,
        );
        let accounting = UsageAccounting::new(repository(
            vec![vec![
                entry_row(&first),
                entry_row(&unrelated),
                entry_row(&turn),
                entry_row(&workflow),
            ]],
            &[1],
        ));

        let batch = accounting
            .export_segment(&tenant, None, 10)
            .await
            .expect("usage export");
        assert_eq!(batch.records().len(), 3);
        assert_eq!(
            batch.scanned_through(),
            Some(AuditHead::from_entry(&workflow))
        );
        assert_eq!(
            batch.aggregate().expect("aggregate"),
            UsageAggregate {
                agent_jobs: 1,
                agent_turns: 1,
                workflow_runs: 1,
                succeeded: 2,
                failed: 1,
                agent_job_duration_millis: 50,
                workflow_duration_millis: 70,
                workflow_steps: 3,
                ..UsageAggregate::default()
            }
        );
        let rendered = format!("{batch:?}");
        assert!(!rendered.contains("credential_detail"));
    }

    #[test]
    fn usage_records_reject_cross_tenant_sources_and_invalid_bounds() {
        let admitted_tenant = tenant(1);
        let other_tenant = tenant(2);
        let source = UsageSourceReference::agent_job(&other_tenant, Uuid::from_u128(100))
            .expect("other source");
        assert!(matches!(
            UsageRecord::agent_job(
                &admitted_tenant,
                OperationId::from_uuid(Uuid::from_u128(40)),
                source,
                1,
                10,
                AuditOutcome::Succeeded,
                1_900_000_000_040,
            ),
            Err(UsageAccountingError::InvalidRecord)
        ));
        let workflow = UsageSourceReference::workflow_run(&admitted_tenant, Uuid::from_u128(140))
            .expect("workflow source");
        assert!(matches!(
            UsageRecord::workflow_run(
                &admitted_tenant,
                OperationId::from_uuid(Uuid::from_u128(41)),
                workflow,
                MAX_WORKFLOW_STEPS + 1,
                10,
                AuditOutcome::Succeeded,
                1_900_000_000_041,
            ),
            Err(UsageAccountingError::InvalidRecord)
        ));
    }

    #[test]
    fn aggregate_detects_overflow() {
        let tenant = tenant(1);
        let mut record = job_record(&tenant, 50, 1);
        record.details = UsageDetails::AgentJob {
            attempt: 1,
            duration_millis: u64::MAX,
        };
        let records = vec![record.clone(), record];
        assert!(matches!(
            UsageAggregate::from_records(&records),
            Err(UsageAccountingError::AggregateOverflow)
        ));
    }
}
