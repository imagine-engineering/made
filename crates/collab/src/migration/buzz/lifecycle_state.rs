use std::collections::{HashMap, HashSet};

use collaboration_domain::{CommunityId, TenantContext};
use serde::Serialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

const SUPPORTED_BUZZ_SCHEMA_VERSION: u32 = 30;
const MAX_IMPORT_RECORDS: usize = 100_000;
const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_JSON_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzWorkflowStatus {
    Active,
    Disabled,
    Archived,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzWorkflowRunStatus {
    Pending,
    Running,
    WaitingApproval,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzWorkflowApprovalStatus {
    Pending,
    Granted,
    Denied,
    Expired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzModerationReportStatus {
    Open,
    Resolved,
    Dismissed,
    Escalated,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzModerationTarget {
    Event([u8; 32]),
    PublicKey([u8; 32]),
    Blob([u8; 32]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzCommunityDeletionState {
    Active,
    Quiescing,
    Fenced,
    Tombstone,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzDeletionStage {
    Submitted,
    Inventoried,
    Approved,
    Fenced,
    Drained,
    BindingsRemoved,
    PostgresPurged,
    CachePurged,
    LogicallyVerified,
    RetentionPending,
    Aborted,
}

impl BuzzDeletionStage {
    const fn progress(self) -> Option<u8> {
        match self {
            Self::Submitted => Some(0),
            Self::Inventoried => Some(1),
            Self::Approved => Some(2),
            Self::Fenced => Some(3),
            Self::Drained => Some(4),
            Self::BindingsRemoved => Some(5),
            Self::PostgresPurged => Some(6),
            Self::CachePurged => Some(7),
            Self::LogicallyVerified => Some(8),
            Self::RetentionPending => Some(9),
            Self::Aborted => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzDeletionCheckpointStatus {
    Started,
    Completed,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzWorkflowRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub workflow_id: Uuid,
    pub name: String,
    pub owner_public_key: [u8; 32],
    pub channel_id: Option<Uuid>,
    pub definition: serde_json::Value,
    pub definition_hash: [u8; 32],
    pub status: BuzzWorkflowStatus,
    pub enabled: bool,
    pub created_at_millis: u64,
    pub updated_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzWorkflowRunRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub run_id: Uuid,
    pub workflow_id: Uuid,
    pub status: BuzzWorkflowRunStatus,
    pub trigger_event_id: Option<[u8; 32]>,
    pub current_step: u32,
    pub execution_trace: serde_json::Value,
    pub trigger_context: Option<serde_json::Value>,
    pub started_at_millis: Option<u64>,
    pub completed_at_millis: Option<u64>,
    pub error_message: Option<String>,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzWorkflowApprovalRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub token_hash: [u8; 32],
    pub workflow_id: Uuid,
    pub run_id: Uuid,
    pub step_id: String,
    pub step_index: u32,
    pub approver_spec: String,
    pub status: BuzzWorkflowApprovalStatus,
    pub approver_public_key: Option<[u8; 32]>,
    pub note: Option<String>,
    pub granted_at_millis: Option<u64>,
    pub denied_at_millis: Option<u64>,
    pub expires_at_millis: u64,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzModerationReportRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub report_id: Uuid,
    pub report_event_id: [u8; 32],
    pub reporter_public_key: [u8; 32],
    pub target: BuzzModerationTarget,
    pub channel_id: Option<Uuid>,
    pub report_type: String,
    pub note: Option<String>,
    pub status: BuzzModerationReportStatus,
    pub resolved_by: Option<[u8; 32]>,
    pub resolved_at_millis: Option<u64>,
    pub action_id: Option<Uuid>,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzModerationRestrictionRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub public_key: [u8; 32],
    pub banned: bool,
    pub ban_expires_at_millis: Option<u64>,
    pub ban_reason: Option<String>,
    pub muted_until_millis: Option<u64>,
    pub mute_reason: Option<String>,
    pub actor_public_key: [u8; 32],
    pub created_at_millis: u64,
    pub updated_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzModerationActionRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub action_id: Uuid,
    pub actor_public_key: [u8; 32],
    pub action: String,
    pub target_public_key: Option<[u8; 32]>,
    pub target_event_id: Option<[u8; 32]>,
    pub channel_id: Option<Uuid>,
    pub reason_code: Option<String>,
    pub public_reason: Option<String>,
    pub private_reason: Option<String>,
    pub matched_principal: Option<String>,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzRetentionWatermarkRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub kind: u32,
    pub author_public_key: [u8; 32],
    pub discriminator: String,
    pub event_created_at_millis: u64,
    pub event_id: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzCommunityDeletionRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub deletion_state: BuzzCommunityDeletionState,
    pub fence_generation: u64,
    pub deleted_at_millis: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzDeletionRequestRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub request_id: Uuid,
    pub community_host: String,
    pub stage: BuzzDeletionStage,
    pub requested_by: String,
    pub inventory_digest: Option<[u8; 32]>,
    pub inventory_frozen_at_millis: Option<u64>,
    pub destructive_storage_frozen_at_millis: Option<u64>,
    pub fence_generation: Option<u64>,
    pub lease_generation: u64,
    pub attempts: u32,
    pub retry_count: u32,
    pub retry_stage: Option<BuzzDeletionStage>,
    pub blocked_reason: Option<String>,
    pub created_at_millis: u64,
    pub updated_at_millis: u64,
    pub completed_at_millis: Option<u64>,
    pub aborted_by: Option<String>,
    pub abort_reason: Option<String>,
    pub aborted_at_millis: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzDeletionApprovalRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub request_id: Uuid,
    pub inventory_digest: [u8; 32],
    pub approved_by: String,
    pub note: Option<String>,
    pub approved_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzDeletionCheckpointRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub request_id: Uuid,
    pub stage: BuzzDeletionStage,
    pub unit_key: String,
    pub status: BuzzDeletionCheckpointStatus,
    pub lease_generation: u64,
    pub attempts: u32,
    pub detail: serde_json::Value,
    pub error: Option<String>,
    pub started_at_millis: u64,
    pub completed_at_millis: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzLifecycleStateRecord {
    Workflow(BuzzWorkflowRecord),
    WorkflowRun(BuzzWorkflowRunRecord),
    WorkflowApproval(BuzzWorkflowApprovalRecord),
    ModerationReport(BuzzModerationReportRecord),
    ModerationRestriction(BuzzModerationRestrictionRecord),
    ModerationAction(BuzzModerationActionRecord),
    RetentionWatermark(BuzzRetentionWatermarkRecord),
    CommunityDeletion(BuzzCommunityDeletionRecord),
    DeletionRequest(BuzzDeletionRequestRecord),
    DeletionApproval(BuzzDeletionApprovalRecord),
    DeletionCheckpoint(BuzzDeletionCheckpointRecord),
}

impl BuzzLifecycleStateRecord {
    fn community_id(&self) -> CommunityId {
        match self {
            Self::Workflow(record) => record.community_id,
            Self::WorkflowRun(record) => record.community_id,
            Self::WorkflowApproval(record) => record.community_id,
            Self::ModerationReport(record) => record.community_id,
            Self::ModerationRestriction(record) => record.community_id,
            Self::ModerationAction(record) => record.community_id,
            Self::RetentionWatermark(record) => record.community_id,
            Self::CommunityDeletion(record) => record.community_id,
            Self::DeletionRequest(record) => record.community_id,
            Self::DeletionApproval(record) => record.community_id,
            Self::DeletionCheckpoint(record) => record.community_id,
        }
    }

    fn source_sequence(&self) -> u64 {
        match self {
            Self::Workflow(record) => record.source_sequence,
            Self::WorkflowRun(record) => record.source_sequence,
            Self::WorkflowApproval(record) => record.source_sequence,
            Self::ModerationReport(record) => record.source_sequence,
            Self::ModerationRestriction(record) => record.source_sequence,
            Self::ModerationAction(record) => record.source_sequence,
            Self::RetentionWatermark(record) => record.source_sequence,
            Self::CommunityDeletion(record) => record.source_sequence,
            Self::DeletionRequest(record) => record.source_sequence,
            Self::DeletionApproval(record) => record.source_sequence,
            Self::DeletionCheckpoint(record) => record.source_sequence,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzLifecycleStateBatch {
    schema_version: u32,
    records: Vec<BuzzLifecycleStateRecord>,
}

impl BuzzLifecycleStateBatch {
    pub fn new(
        schema_version: u32,
        records: Vec<BuzzLifecycleStateRecord>,
    ) -> Result<Self, BuzzLifecycleStateImportError> {
        if schema_version != SUPPORTED_BUZZ_SCHEMA_VERSION {
            return Err(BuzzLifecycleStateImportError::UnsupportedSchemaVersion(
                schema_version,
            ));
        }
        if records.is_empty() || records.len() > MAX_IMPORT_RECORDS {
            return Err(BuzzLifecycleStateImportError::InvalidBatch);
        }
        let mut previous_sequence = 0;
        for record in &records {
            if record.source_sequence() <= previous_sequence {
                return Err(BuzzLifecycleStateImportError::InvalidBatch);
            }
            validate_record(record)?;
            previous_sequence = record.source_sequence();
        }
        Ok(Self {
            schema_version,
            records,
        })
    }

    pub fn records(&self) -> &[BuzzLifecycleStateRecord] {
        &self.records
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzLifecycleWorkerState {
    pub workflow_scheduler_enabled: bool,
    pub workflow_executor_enabled: bool,
    pub retention_worker_enabled: bool,
    pub deletion_worker_enabled: bool,
}

impl BuzzLifecycleWorkerState {
    pub const fn all_disabled() -> Self {
        Self {
            workflow_scheduler_enabled: false,
            workflow_executor_enabled: false,
            retention_worker_enabled: false,
            deletion_worker_enabled: false,
        }
    }

    pub const fn any_enabled(self) -> bool {
        self.workflow_scheduler_enabled
            || self.workflow_executor_enabled
            || self.retention_worker_enabled
            || self.deletion_worker_enabled
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzLifecycleCheckpointProgress {
    pub final_source_sequence: u64,
    pub source_hash: [u8; 32],
    pub staged_hash: [u8; 32],
    pub scanned: u64,
    pub staged: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzLifecycleStagingPlan {
    pub records: Vec<BuzzLifecycleStateRecord>,
    pub workers: BuzzLifecycleWorkerState,
    pub checkpoint: BuzzLifecycleCheckpointProgress,
}

#[derive(Debug, thiserror::Error)]
pub enum BuzzLifecycleStateImportError {
    #[error("Buzz lifecycle-state schema version {0} is unsupported")]
    UnsupportedSchemaVersion(u32),
    #[error("Buzz lifecycle-state batch is empty, oversized or out of order")]
    InvalidBatch,
    #[error("Buzz lifecycle-state source record is invalid")]
    InvalidSourceRecord,
    #[error("Buzz lifecycle-state import crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("Buzz lifecycle-state source contains duplicate durable identity")]
    DuplicateIdentity,
    #[error("Buzz lifecycle-state source contains a missing or inconsistent reference")]
    InvalidReference,
    #[error("Buzz lifecycle-state source contains an impossible state transition")]
    InvalidState,
    #[error("Buzz lifecycle-state source could not be hashed")]
    Hashing(#[source] serde_json::Error),
}

pub struct BuzzLifecycleStateImporter;

impl BuzzLifecycleStateImporter {
    pub fn stage(
        tenant: &TenantContext,
        batch: &BuzzLifecycleStateBatch,
    ) -> Result<BuzzLifecycleStagingPlan, BuzzLifecycleStateImportError> {
        if batch
            .records
            .iter()
            .any(|record| record.community_id() != tenant.community_id())
        {
            return Err(BuzzLifecycleStateImportError::TenantBoundaryViolation);
        }
        validate_references(&batch.records)?;
        let mut source_hasher = Sha256::new();
        let mut staged_hasher = Sha256::new();
        for record in &batch.records {
            let bytes =
                serde_json::to_vec(record).map_err(BuzzLifecycleStateImportError::Hashing)?;
            hash_part(&mut source_hasher, &batch.schema_version.to_be_bytes());
            hash_part(&mut source_hasher, &record.source_sequence().to_be_bytes());
            hash_part(&mut source_hasher, &bytes);
            hash_part(&mut staged_hasher, &batch.schema_version.to_be_bytes());
            hash_part(&mut staged_hasher, &record.source_sequence().to_be_bytes());
            hash_part(&mut staged_hasher, &bytes);
        }
        let scanned = u64::try_from(batch.records.len())
            .map_err(|_| BuzzLifecycleStateImportError::InvalidBatch)?;
        let final_source_sequence = batch
            .records
            .last()
            .map(BuzzLifecycleStateRecord::source_sequence)
            .ok_or(BuzzLifecycleStateImportError::InvalidBatch)?;
        Ok(BuzzLifecycleStagingPlan {
            records: batch.records.clone(),
            workers: BuzzLifecycleWorkerState::all_disabled(),
            checkpoint: BuzzLifecycleCheckpointProgress {
                final_source_sequence,
                source_hash: source_hasher.finalize().into(),
                staged_hash: staged_hasher.finalize().into(),
                scanned,
                staged: scanned,
            },
        })
    }
}

fn validate_record(record: &BuzzLifecycleStateRecord) -> Result<(), BuzzLifecycleStateImportError> {
    if record.source_sequence() == 0 {
        return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
    }
    match record {
        BuzzLifecycleStateRecord::Workflow(record) => {
            if !valid_text(&record.name)
                || !valid_json(&record.definition)
                || !valid_timestamp(record.created_at_millis)
                || !valid_timestamp(record.updated_at_millis)
                || record.updated_at_millis < record.created_at_millis
                || (record.enabled && record.status != BuzzWorkflowStatus::Active)
            {
                return Err(BuzzLifecycleStateImportError::InvalidState);
            }
            let definition = serde_json::to_vec(&record.definition)
                .map_err(BuzzLifecycleStateImportError::Hashing)?;
            if Sha256::digest(definition).as_slice() != record.definition_hash {
                return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
            }
        }
        BuzzLifecycleStateRecord::WorkflowRun(record) => validate_workflow_run(record)?,
        BuzzLifecycleStateRecord::WorkflowApproval(record) => validate_workflow_approval(record)?,
        BuzzLifecycleStateRecord::ModerationReport(record) => {
            if !valid_text(&record.report_type)
                || !valid_optional_text(record.note.as_deref())
                || !valid_timestamp(record.created_at_millis)
            {
                return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
            }
            let has_resolution =
                record.resolved_by.is_some() && record.resolved_at_millis.is_some();
            match record.status {
                BuzzModerationReportStatus::Open
                    if has_resolution || record.action_id.is_some() =>
                {
                    return Err(BuzzLifecycleStateImportError::InvalidState);
                }
                BuzzModerationReportStatus::Resolved
                | BuzzModerationReportStatus::Dismissed
                | BuzzModerationReportStatus::Escalated
                    if !has_resolution =>
                {
                    return Err(BuzzLifecycleStateImportError::InvalidState);
                }
                _ => {}
            }
        }
        BuzzLifecycleStateRecord::ModerationRestriction(record) => {
            if !valid_optional_text(record.ban_reason.as_deref())
                || !valid_optional_text(record.mute_reason.as_deref())
                || !valid_timestamp(record.created_at_millis)
                || !valid_timestamp(record.updated_at_millis)
                || record.updated_at_millis < record.created_at_millis
                || (!record.banned && record.ban_expires_at_millis.is_some())
            {
                return Err(BuzzLifecycleStateImportError::InvalidState);
            }
        }
        BuzzLifecycleStateRecord::ModerationAction(record) => {
            const ACTIONS: &[&str] = &[
                "delete_message",
                "kick",
                "ban",
                "unban",
                "timeout",
                "untimeout",
                "dismiss_report",
                "escalate",
                "resolve:delete",
                "resolve:kick",
                "resolve:ban",
                "resolve:timeout",
            ];
            if !ACTIONS.contains(&record.action.as_str())
                || !valid_optional_text(record.reason_code.as_deref())
                || !valid_optional_text(record.public_reason.as_deref())
                || !valid_optional_text(record.private_reason.as_deref())
                || !matches!(
                    record.matched_principal.as_deref(),
                    None | Some("self" | "owner")
                )
                || !valid_timestamp(record.created_at_millis)
            {
                return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
            }
        }
        BuzzLifecycleStateRecord::RetentionWatermark(record) => {
            let discriminator = record
                .discriminator
                .strip_prefix("read-state:")
                .unwrap_or_default();
            if record.kind != 30_078
                || discriminator.len() != 32
                || !discriminator.bytes().all(|byte| byte.is_ascii_hexdigit())
                || !valid_timestamp(record.event_created_at_millis)
            {
                return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
            }
        }
        BuzzLifecycleStateRecord::CommunityDeletion(record) => {
            if record.deletion_state == BuzzCommunityDeletionState::Active
                && record.deleted_at_millis.is_some()
                || record.deletion_state == BuzzCommunityDeletionState::Tombstone
                    && (record.fence_generation == 0 || record.deleted_at_millis.is_none())
            {
                return Err(BuzzLifecycleStateImportError::InvalidState);
            }
        }
        BuzzLifecycleStateRecord::DeletionRequest(record) => validate_deletion_request(record)?,
        BuzzLifecycleStateRecord::DeletionApproval(record) => {
            if !valid_text(&record.approved_by)
                || !valid_optional_text(record.note.as_deref())
                || !valid_timestamp(record.approved_at_millis)
            {
                return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
            }
        }
        BuzzLifecycleStateRecord::DeletionCheckpoint(record) => {
            if record.stage == BuzzDeletionStage::Aborted
                || !valid_text(&record.unit_key)
                || record.lease_generation == 0
                || record.attempts == 0
                || !valid_json(&record.detail)
                || !valid_timestamp(record.started_at_millis)
                || !valid_optional_text(record.error.as_deref())
            {
                return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
            }
            match record.status {
                BuzzDeletionCheckpointStatus::Started
                    if record.completed_at_millis.is_some() || record.error.is_some() =>
                {
                    return Err(BuzzLifecycleStateImportError::InvalidState);
                }
                BuzzDeletionCheckpointStatus::Completed
                    if record.completed_at_millis.is_none() || record.error.is_some() =>
                {
                    return Err(BuzzLifecycleStateImportError::InvalidState);
                }
                BuzzDeletionCheckpointStatus::Failed
                    if record.error.is_none() || record.completed_at_millis.is_some() =>
                {
                    return Err(BuzzLifecycleStateImportError::InvalidState);
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn validate_workflow_run(
    record: &BuzzWorkflowRunRecord,
) -> Result<(), BuzzLifecycleStateImportError> {
    if !valid_json(&record.execution_trace)
        || record
            .trigger_context
            .as_ref()
            .is_some_and(|value| !valid_json(value))
        || !valid_timestamp(record.created_at_millis)
        || !valid_optional_text(record.error_message.as_deref())
    {
        return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
    }
    let legal = match record.status {
        BuzzWorkflowRunStatus::Pending => {
            record.started_at_millis.is_none()
                && record.completed_at_millis.is_none()
                && record.error_message.is_none()
        }
        BuzzWorkflowRunStatus::Running | BuzzWorkflowRunStatus::WaitingApproval => {
            record.started_at_millis.is_some()
                && record.completed_at_millis.is_none()
                && record.error_message.is_none()
        }
        BuzzWorkflowRunStatus::Completed => {
            record.started_at_millis.is_some()
                && record.completed_at_millis.is_some()
                && record.error_message.is_none()
        }
        BuzzWorkflowRunStatus::Failed => {
            record.started_at_millis.is_some()
                && record.completed_at_millis.is_some()
                && record.error_message.is_some()
        }
        BuzzWorkflowRunStatus::Cancelled => {
            record.completed_at_millis.is_some() && record.error_message.is_none()
        }
    };
    if !legal {
        return Err(BuzzLifecycleStateImportError::InvalidState);
    }
    Ok(())
}

fn validate_workflow_approval(
    record: &BuzzWorkflowApprovalRecord,
) -> Result<(), BuzzLifecycleStateImportError> {
    if !valid_text(&record.step_id)
        || !valid_text(&record.approver_spec)
        || !valid_optional_text(record.note.as_deref())
        || !valid_timestamp(record.created_at_millis)
        || !valid_timestamp(record.expires_at_millis)
        || record.expires_at_millis < record.created_at_millis
    {
        return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
    }
    let legal = match record.status {
        BuzzWorkflowApprovalStatus::Pending => {
            record.approver_public_key.is_none()
                && record.granted_at_millis.is_none()
                && record.denied_at_millis.is_none()
        }
        BuzzWorkflowApprovalStatus::Granted => {
            record.approver_public_key.is_some()
                && record.granted_at_millis.is_some()
                && record.denied_at_millis.is_none()
        }
        BuzzWorkflowApprovalStatus::Denied => {
            record.approver_public_key.is_some()
                && record.granted_at_millis.is_none()
                && record.denied_at_millis.is_some()
        }
        BuzzWorkflowApprovalStatus::Expired => {
            record.granted_at_millis.is_none() && record.denied_at_millis.is_none()
        }
    };
    if !legal {
        return Err(BuzzLifecycleStateImportError::InvalidState);
    }
    Ok(())
}

fn validate_deletion_request(
    record: &BuzzDeletionRequestRecord,
) -> Result<(), BuzzLifecycleStateImportError> {
    if !valid_text(&record.community_host)
        || !valid_text(&record.requested_by)
        || !valid_optional_text(record.blocked_reason.as_deref())
        || !valid_optional_text(record.aborted_by.as_deref())
        || !valid_optional_text(record.abort_reason.as_deref())
        || !valid_timestamp(record.created_at_millis)
        || !valid_timestamp(record.updated_at_millis)
        || record.updated_at_millis < record.created_at_millis
        || record.retry_stage == Some(BuzzDeletionStage::Aborted)
    {
        return Err(BuzzLifecycleStateImportError::InvalidSourceRecord);
    }
    let progress = record.stage.progress();
    let requires_inventory =
        record.stage == BuzzDeletionStage::Aborted || progress.is_some_and(|value| value >= 1);
    let requires_fence =
        record.stage == BuzzDeletionStage::Aborted || progress.is_some_and(|value| value >= 3);
    if requires_inventory
        != (record.inventory_digest.is_some() && record.inventory_frozen_at_millis.is_some())
        || requires_fence != record.fence_generation.is_some()
        || record.stage == BuzzDeletionStage::RetentionPending
            && record.completed_at_millis.is_none()
    {
        return Err(BuzzLifecycleStateImportError::InvalidState);
    }
    let aborted_evidence = record.aborted_by.is_some()
        && record.abort_reason.is_some()
        && record.aborted_at_millis.is_some();
    if (record.stage == BuzzDeletionStage::Aborted) != aborted_evidence {
        return Err(BuzzLifecycleStateImportError::InvalidState);
    }
    if record.retry_stage.is_some_and(|retry| {
        let Some(retry_progress) = retry.progress() else {
            return true;
        };
        progress.is_some_and(|current| retry_progress > current)
    }) {
        return Err(BuzzLifecycleStateImportError::InvalidState);
    }
    Ok(())
}

fn validate_references(
    records: &[BuzzLifecycleStateRecord],
) -> Result<(), BuzzLifecycleStateImportError> {
    let mut workflows = HashSet::new();
    let mut runs = HashMap::new();
    let mut actions = HashSet::new();
    let mut requests = HashMap::new();
    let mut approvals = HashMap::new();
    let mut durable_identities = HashSet::new();
    let mut community_deletion = None;
    for record in records {
        let identity = durable_identity(record);
        if !durable_identities.insert(identity) {
            return Err(BuzzLifecycleStateImportError::DuplicateIdentity);
        }
        match record {
            BuzzLifecycleStateRecord::Workflow(record) => {
                workflows.insert(record.workflow_id);
            }
            BuzzLifecycleStateRecord::WorkflowRun(record) => {
                runs.insert(record.run_id, (record.workflow_id, record.status));
            }
            BuzzLifecycleStateRecord::ModerationAction(record) => {
                actions.insert(record.action_id);
            }
            BuzzLifecycleStateRecord::CommunityDeletion(record) => {
                if community_deletion.replace(record).is_some() {
                    return Err(BuzzLifecycleStateImportError::DuplicateIdentity);
                }
            }
            BuzzLifecycleStateRecord::DeletionRequest(record) => {
                requests.insert(record.request_id, record);
            }
            BuzzLifecycleStateRecord::DeletionApproval(record) => {
                approvals.insert(record.request_id, record);
            }
            _ => {}
        }
    }
    for record in records {
        match record {
            BuzzLifecycleStateRecord::WorkflowRun(record)
                if !workflows.contains(&record.workflow_id) =>
            {
                return Err(BuzzLifecycleStateImportError::InvalidReference);
            }
            BuzzLifecycleStateRecord::WorkflowApproval(record) => {
                let Some((workflow_id, run_status)) = runs.get(&record.run_id) else {
                    return Err(BuzzLifecycleStateImportError::InvalidReference);
                };
                if *workflow_id != record.workflow_id
                    || !workflows.contains(workflow_id)
                    || record.status == BuzzWorkflowApprovalStatus::Pending
                        && *run_status != BuzzWorkflowRunStatus::WaitingApproval
                {
                    return Err(BuzzLifecycleStateImportError::InvalidReference);
                }
            }
            BuzzLifecycleStateRecord::ModerationReport(record)
                if record.action_id.is_some_and(|id| !actions.contains(&id)) =>
            {
                return Err(BuzzLifecycleStateImportError::InvalidReference);
            }
            BuzzLifecycleStateRecord::DeletionApproval(record) => {
                let Some(request) = requests.get(&record.request_id) else {
                    return Err(BuzzLifecycleStateImportError::InvalidReference);
                };
                if request.inventory_digest != Some(record.inventory_digest) {
                    return Err(BuzzLifecycleStateImportError::InvalidReference);
                }
            }
            BuzzLifecycleStateRecord::DeletionCheckpoint(record) => {
                let Some(request) = requests.get(&record.request_id) else {
                    return Err(BuzzLifecycleStateImportError::InvalidReference);
                };
                let checkpoint_progress = record
                    .stage
                    .progress()
                    .ok_or(BuzzLifecycleStateImportError::InvalidState)?;
                let maximum_progress = request.stage.progress().unwrap_or_else(|| {
                    if request.stage == BuzzDeletionStage::Aborted {
                        BuzzDeletionStage::Fenced.progress().unwrap_or(0)
                    } else {
                        request
                            .retry_stage
                            .and_then(BuzzDeletionStage::progress)
                            .unwrap_or(0)
                    }
                });
                if checkpoint_progress > maximum_progress
                    || record.lease_generation > request.lease_generation
                {
                    return Err(BuzzLifecycleStateImportError::InvalidState);
                }
            }
            _ => {}
        }
    }
    for request in requests.values() {
        let approved = request.stage == BuzzDeletionStage::Aborted
            || request
                .stage
                .progress()
                .is_some_and(|progress| progress >= 2);
        if approved != approvals.contains_key(&request.request_id) {
            return Err(BuzzLifecycleStateImportError::InvalidReference);
        }
        if let Some(community) = community_deletion {
            let state_is_legal = match request.stage {
                BuzzDeletionStage::Submitted
                | BuzzDeletionStage::Inventoried
                | BuzzDeletionStage::Aborted => {
                    community.deletion_state == BuzzCommunityDeletionState::Active
                }
                BuzzDeletionStage::Approved => matches!(
                    community.deletion_state,
                    BuzzCommunityDeletionState::Active | BuzzCommunityDeletionState::Quiescing
                ),
                BuzzDeletionStage::Fenced
                | BuzzDeletionStage::Drained
                | BuzzDeletionStage::BindingsRemoved
                | BuzzDeletionStage::PostgresPurged
                | BuzzDeletionStage::CachePurged
                | BuzzDeletionStage::LogicallyVerified => {
                    community.deletion_state == BuzzCommunityDeletionState::Fenced
                }
                BuzzDeletionStage::RetentionPending => {
                    community.deletion_state == BuzzCommunityDeletionState::Tombstone
                }
            };
            if !state_is_legal
                || request
                    .fence_generation
                    .is_some_and(|generation| generation != community.fence_generation)
            {
                return Err(BuzzLifecycleStateImportError::InvalidState);
            }
        }
    }
    Ok(())
}

fn durable_identity(record: &BuzzLifecycleStateRecord) -> String {
    match record {
        BuzzLifecycleStateRecord::Workflow(record) => format!("workflow:{}", record.workflow_id),
        BuzzLifecycleStateRecord::WorkflowRun(record) => format!("run:{}", record.run_id),
        BuzzLifecycleStateRecord::WorkflowApproval(record) => {
            format!("approval:{}", hex::encode(record.token_hash))
        }
        BuzzLifecycleStateRecord::ModerationReport(record) => {
            format!("report:{}", record.report_id)
        }
        BuzzLifecycleStateRecord::ModerationRestriction(record) => {
            format!("restriction:{}", hex::encode(record.public_key))
        }
        BuzzLifecycleStateRecord::ModerationAction(record) => {
            format!("action:{}", record.action_id)
        }
        BuzzLifecycleStateRecord::RetentionWatermark(record) => format!(
            "watermark:{}:{}:{}",
            record.kind,
            hex::encode(record.author_public_key),
            record.discriminator
        ),
        BuzzLifecycleStateRecord::CommunityDeletion(_) => "community-deletion".to_owned(),
        BuzzLifecycleStateRecord::DeletionRequest(record) => {
            format!("deletion-request:{}", record.request_id)
        }
        BuzzLifecycleStateRecord::DeletionApproval(record) => {
            format!("deletion-approval:{}", record.request_id)
        }
        BuzzLifecycleStateRecord::DeletionCheckpoint(record) => format!(
            "deletion-checkpoint:{}:{:?}:{}",
            record.request_id, record.stage, record.unit_key
        ),
    }
}

fn valid_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TEXT_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn valid_optional_text(value: Option<&str>) -> bool {
    value.is_none_or(|value| value.len() <= MAX_TEXT_BYTES && !value.chars().any(char::is_control))
}

fn valid_timestamp(value: u64) -> bool {
    value > 0 && i64::try_from(value).is_ok()
}

fn valid_json(value: &serde_json::Value) -> bool {
    serde_json::to_vec(value).is_ok_and(|bytes| bytes.len() <= MAX_JSON_BYTES)
}

fn hash_part(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(bytes.len().to_be_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use collaboration_domain::{TenantContext, TrustedTenantRoute};
    use serde_json::json;

    use super::*;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "buzz-lifecycle-import")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn fixture_records(community_id: CommunityId) -> Vec<BuzzLifecycleStateRecord> {
        let workflow_id = Uuid::from_u128(10);
        let run_id = Uuid::from_u128(11);
        let action_id = Uuid::from_u128(12);
        let request_id = Uuid::from_u128(13);
        let definition = json!({"steps": [{"type": "approval"}]});
        let definition_hash = Sha256::digest(serde_json::to_vec(&definition).expect("json")).into();
        vec![
            BuzzLifecycleStateRecord::Workflow(BuzzWorkflowRecord {
                community_id,
                source_sequence: 1,
                workflow_id,
                name: "review".to_owned(),
                owner_public_key: [1; 32],
                channel_id: None,
                definition,
                definition_hash,
                status: BuzzWorkflowStatus::Active,
                enabled: true,
                created_at_millis: 1_000,
                updated_at_millis: 2_000,
            }),
            BuzzLifecycleStateRecord::WorkflowRun(BuzzWorkflowRunRecord {
                community_id,
                source_sequence: 2,
                run_id,
                workflow_id,
                status: BuzzWorkflowRunStatus::WaitingApproval,
                trigger_event_id: Some([2; 32]),
                current_step: 0,
                execution_trace: json!([]),
                trigger_context: Some(json!({"channel": "builders"})),
                started_at_millis: Some(2_000),
                completed_at_millis: None,
                error_message: None,
                created_at_millis: 2_000,
            }),
            BuzzLifecycleStateRecord::WorkflowApproval(BuzzWorkflowApprovalRecord {
                community_id,
                source_sequence: 3,
                token_hash: [3; 32],
                workflow_id,
                run_id,
                step_id: "approval".to_owned(),
                step_index: 0,
                approver_spec: "role:owner".to_owned(),
                status: BuzzWorkflowApprovalStatus::Pending,
                approver_public_key: None,
                note: None,
                granted_at_millis: None,
                denied_at_millis: None,
                expires_at_millis: 10_000,
                created_at_millis: 2_000,
            }),
            BuzzLifecycleStateRecord::ModerationAction(BuzzModerationActionRecord {
                community_id,
                source_sequence: 4,
                action_id,
                actor_public_key: [4; 32],
                action: "dismiss_report".to_owned(),
                target_public_key: Some([5; 32]),
                target_event_id: None,
                channel_id: None,
                reason_code: Some("not-actionable".to_owned()),
                public_reason: None,
                private_reason: Some("duplicate".to_owned()),
                matched_principal: None,
                created_at_millis: 3_000,
            }),
            BuzzLifecycleStateRecord::ModerationReport(BuzzModerationReportRecord {
                community_id,
                source_sequence: 5,
                report_id: Uuid::from_u128(14),
                report_event_id: [6; 32],
                reporter_public_key: [7; 32],
                target: BuzzModerationTarget::PublicKey([5; 32]),
                channel_id: None,
                report_type: "spam".to_owned(),
                note: Some("repeated posts".to_owned()),
                status: BuzzModerationReportStatus::Dismissed,
                resolved_by: Some([4; 32]),
                resolved_at_millis: Some(3_000),
                action_id: Some(action_id),
                created_at_millis: 2_500,
            }),
            BuzzLifecycleStateRecord::RetentionWatermark(BuzzRetentionWatermarkRecord {
                community_id,
                source_sequence: 6,
                kind: 30_078,
                author_public_key: [8; 32],
                discriminator: "read-state:00112233445566778899aabbccddeeff".to_owned(),
                event_created_at_millis: 4_000,
                event_id: [9; 32],
            }),
            BuzzLifecycleStateRecord::CommunityDeletion(BuzzCommunityDeletionRecord {
                community_id,
                source_sequence: 7,
                deletion_state: BuzzCommunityDeletionState::Fenced,
                fence_generation: 2,
                deleted_at_millis: None,
            }),
            BuzzLifecycleStateRecord::DeletionRequest(BuzzDeletionRequestRecord {
                community_id,
                source_sequence: 8,
                request_id,
                community_host: "community.example".to_owned(),
                stage: BuzzDeletionStage::Drained,
                requested_by: "operator@example".to_owned(),
                inventory_digest: Some([10; 32]),
                inventory_frozen_at_millis: Some(5_000),
                destructive_storage_frozen_at_millis: Some(6_000),
                fence_generation: Some(2),
                lease_generation: 3,
                attempts: 1,
                retry_count: 0,
                retry_stage: None,
                blocked_reason: None,
                created_at_millis: 4_500,
                updated_at_millis: 7_000,
                completed_at_millis: None,
                aborted_by: None,
                abort_reason: None,
                aborted_at_millis: None,
            }),
            BuzzLifecycleStateRecord::DeletionApproval(BuzzDeletionApprovalRecord {
                community_id,
                source_sequence: 9,
                request_id,
                inventory_digest: [10; 32],
                approved_by: "reviewer@example".to_owned(),
                note: None,
                approved_at_millis: 5_500,
            }),
            BuzzLifecycleStateRecord::DeletionCheckpoint(BuzzDeletionCheckpointRecord {
                community_id,
                source_sequence: 10,
                request_id,
                stage: BuzzDeletionStage::Fenced,
                unit_key: "postgres-write-fence".to_owned(),
                status: BuzzDeletionCheckpointStatus::Completed,
                lease_generation: 3,
                attempts: 1,
                detail: json!({"generation": 2}),
                error: None,
                started_at_millis: 6_000,
                completed_at_millis: Some(6_500),
            }),
        ]
    }

    #[test]
    fn stages_legal_state_with_every_worker_disabled() {
        let community_id = community(1);
        let batch = BuzzLifecycleStateBatch::new(30, fixture_records(community_id))
            .expect("legal lifecycle snapshot");
        let staged = BuzzLifecycleStateImporter::stage(&tenant(community_id), &batch)
            .expect("stage lifecycle state");

        assert_eq!(staged.records, batch.records);
        assert_eq!(staged.checkpoint.scanned, 10);
        assert_eq!(staged.checkpoint.staged, 10);
        assert_eq!(staged.checkpoint.source_hash, staged.checkpoint.staged_hash);
        assert!(!staged.workers.any_enabled());
    }

    #[test]
    fn rejects_cross_tenant_and_impossible_deletion_progress() {
        let source_community = community(1);
        let batch = BuzzLifecycleStateBatch::new(30, fixture_records(source_community))
            .expect("legal lifecycle snapshot");
        assert!(matches!(
            BuzzLifecycleStateImporter::stage(&tenant(community(2)), &batch),
            Err(BuzzLifecycleStateImportError::TenantBoundaryViolation)
        ));

        let mut records = fixture_records(source_community);
        let BuzzLifecycleStateRecord::DeletionCheckpoint(checkpoint) =
            records.last_mut().expect("checkpoint")
        else {
            panic!("last fixture must be a checkpoint")
        };
        checkpoint.stage = BuzzDeletionStage::PostgresPurged;
        let batch = BuzzLifecycleStateBatch::new(30, records).expect("well-formed records");
        assert!(matches!(
            BuzzLifecycleStateImporter::stage(&tenant(source_community), &batch),
            Err(BuzzLifecycleStateImportError::InvalidState)
        ));
    }

    #[test]
    fn preserves_terminal_abort_evidence_without_rearming_workers() {
        let community_id = community(1);
        let mut records = fixture_records(community_id);
        let BuzzLifecycleStateRecord::CommunityDeletion(community) = &mut records[6] else {
            panic!("fixture community deletion")
        };
        community.deletion_state = BuzzCommunityDeletionState::Active;
        let BuzzLifecycleStateRecord::DeletionRequest(request) = &mut records[7] else {
            panic!("fixture deletion request")
        };
        request.stage = BuzzDeletionStage::Aborted;
        request.lease_generation = 4;
        request.completed_at_millis = Some(7_000);
        request.aborted_by = Some("operator@example".to_owned());
        request.abort_reason = Some("cancel deletion".to_owned());
        request.aborted_at_millis = Some(7_000);

        let batch = BuzzLifecycleStateBatch::new(30, records).expect("abort evidence");
        let staged = BuzzLifecycleStateImporter::stage(&tenant(community_id), &batch)
            .expect("stage terminal abort evidence");
        assert!(!staged.workers.any_enabled());
    }

    #[test]
    fn rejects_mismatched_approval_and_report_action_references() {
        let community_id = community(1);
        let mut records = fixture_records(community_id);
        let BuzzLifecycleStateRecord::DeletionApproval(approval) = &mut records[8] else {
            panic!("fixture approval")
        };
        approval.inventory_digest = [99; 32];
        let batch = BuzzLifecycleStateBatch::new(30, records).expect("well-formed records");
        assert!(matches!(
            BuzzLifecycleStateImporter::stage(&tenant(community_id), &batch),
            Err(BuzzLifecycleStateImportError::InvalidReference)
        ));

        let mut records = fixture_records(community_id);
        let BuzzLifecycleStateRecord::ModerationReport(report) = &mut records[4] else {
            panic!("fixture report")
        };
        report.action_id = Some(Uuid::from_u128(999));
        let batch = BuzzLifecycleStateBatch::new(30, records).expect("well-formed records");
        assert!(matches!(
            BuzzLifecycleStateImporter::stage(&tenant(community_id), &batch),
            Err(BuzzLifecycleStateImportError::InvalidReference)
        ));
    }
}
