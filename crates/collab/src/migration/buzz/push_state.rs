use std::collections::{HashMap, HashSet};

use collaboration_domain::{CommunityId, TenantContext};
use serde::Serialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

const PUSH_IMPORT_FORMAT_VERSION: u32 = 1;
const MAX_IMPORT_RECORDS: usize = 100_000;
const MAX_INSTALLATION_ID_BYTES: usize = 64;
const MAX_OPAQUE_VALUE_BYTES: usize = 16 * 1024;
const MAX_SUBSCRIPTIONS_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzPushClass {
    Silent,
    Default,
    TimeSensitive,
    Urgent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzWakeState {
    Pending,
    Sending,
    Delivered,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzPushMatchState {
    Pending,
    Matching,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzPushLeaseRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub author_public_key: [u8; 32],
    pub installation_id: String,
    pub source_event_id: [u8; 32],
    pub source_created_at_seconds: i64,
    pub generation: u64,
    pub active: bool,
    pub endpoint_enabled: bool,
    pub app_profile: Option<String>,
    pub endpoint_hash: Option<[u8; 32]>,
    pub endpoint_grant: Option<String>,
    pub max_class: Option<BuzzPushClass>,
    pub subscriptions: Option<serde_json::Value>,
    pub expires_at_seconds: i64,
    pub updated_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzPushWakeRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub wake_id: Uuid,
    pub author_public_key: [u8; 32],
    pub installation_id: String,
    pub lease_generation: u64,
    pub endpoint_hash: [u8; 32],
    pub event_id: [u8; 32],
    pub class: BuzzPushClass,
    pub expires_at_seconds: i64,
    pub state: BuzzWakeState,
    pub attempts: u32,
    pub next_attempt_at_millis: u64,
    pub lease_until_millis: Option<u64>,
    pub claim_id: Option<Uuid>,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzPushMatchRecord {
    pub community_id: CommunityId,
    pub source_sequence: u64,
    pub event_id: [u8; 32],
    pub state: BuzzPushMatchState,
    pub attempts: u32,
    pub next_attempt_at_millis: u64,
    pub lease_until_millis: Option<u64>,
    pub claim_id: Option<Uuid>,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzPushStateRecord {
    Lease(BuzzPushLeaseRecord),
    Wake(BuzzPushWakeRecord),
    Match(BuzzPushMatchRecord),
}

impl BuzzPushStateRecord {
    fn community_id(&self) -> CommunityId {
        match self {
            Self::Lease(record) => record.community_id,
            Self::Wake(record) => record.community_id,
            Self::Match(record) => record.community_id,
        }
    }

    fn source_sequence(&self) -> u64 {
        match self {
            Self::Lease(record) => record.source_sequence,
            Self::Wake(record) => record.source_sequence,
            Self::Match(record) => record.source_sequence,
        }
    }

    fn recover_claim(&self) -> Self {
        match self {
            Self::Wake(record) if record.state == BuzzWakeState::Sending => {
                let mut record = record.clone();
                record.state = BuzzWakeState::Pending;
                record.claim_id = None;
                record.lease_until_millis = None;
                Self::Wake(record)
            }
            Self::Match(record) if record.state == BuzzPushMatchState::Matching => {
                let mut record = record.clone();
                record.state = BuzzPushMatchState::Pending;
                record.claim_id = None;
                record.lease_until_millis = None;
                Self::Match(record)
            }
            _ => self.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzPushStateBatch {
    format_version: u32,
    records: Vec<BuzzPushStateRecord>,
}

impl BuzzPushStateBatch {
    pub fn new(
        format_version: u32,
        records: Vec<BuzzPushStateRecord>,
    ) -> Result<Self, BuzzPushStateImportError> {
        if format_version != PUSH_IMPORT_FORMAT_VERSION {
            return Err(BuzzPushStateImportError::UnsupportedFormatVersion(
                format_version,
            ));
        }
        if records.is_empty() || records.len() > MAX_IMPORT_RECORDS {
            return Err(BuzzPushStateImportError::InvalidBatch);
        }
        let mut previous_sequence = 0;
        for record in &records {
            if record.source_sequence() <= previous_sequence {
                return Err(BuzzPushStateImportError::InvalidBatch);
            }
            validate_record(record)?;
            previous_sequence = record.source_sequence();
        }
        Ok(Self {
            format_version,
            records,
        })
    }

    pub fn records(&self) -> &[BuzzPushStateRecord] {
        &self.records
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzPushExecutionState {
    pub matcher_enabled: bool,
    pub wake_dispatch_enabled: bool,
    pub provider_contact_allowed: bool,
}

impl BuzzPushExecutionState {
    pub const fn all_disabled() -> Self {
        Self {
            matcher_enabled: false,
            wake_dispatch_enabled: false,
            provider_contact_allowed: false,
        }
    }

    pub const fn can_contact_provider(self) -> bool {
        self.matcher_enabled || self.wake_dispatch_enabled || self.provider_contact_allowed
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzPushCheckpointProgress {
    pub final_source_sequence: u64,
    pub source_hash: [u8; 32],
    pub staged_hash: [u8; 32],
    pub scanned: u64,
    pub staged: u64,
    pub recovered_claims: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzPushStagingPlan {
    pub records: Vec<BuzzPushStateRecord>,
    pub execution: BuzzPushExecutionState,
    pub checkpoint: BuzzPushCheckpointProgress,
}

#[derive(Debug, thiserror::Error)]
pub enum BuzzPushStateImportError {
    #[error("Buzz push import format version {0} is unsupported")]
    UnsupportedFormatVersion(u32),
    #[error("Buzz push-state batch is empty, oversized or out of order")]
    InvalidBatch,
    #[error("Buzz push-state source record is invalid")]
    InvalidSourceRecord,
    #[error("Buzz push-state import crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("Buzz push-state source contains a duplicate durable identity")]
    DuplicateIdentity,
    #[error("Buzz push-state source contains an invalid lease or queue reference")]
    InvalidReference,
    #[error("Buzz push-state source contains an impossible state")]
    InvalidState,
    #[error("Buzz push-state source could not be hashed")]
    Hashing(#[source] serde_json::Error),
}

pub struct BuzzPushStateImporter;

impl BuzzPushStateImporter {
    pub fn stage(
        tenant: &TenantContext,
        batch: &BuzzPushStateBatch,
    ) -> Result<BuzzPushStagingPlan, BuzzPushStateImportError> {
        if batch
            .records
            .iter()
            .any(|record| record.community_id() != tenant.community_id())
        {
            return Err(BuzzPushStateImportError::TenantBoundaryViolation);
        }
        validate_references(&batch.records)?;
        let records: Vec<_> = batch
            .records
            .iter()
            .map(BuzzPushStateRecord::recover_claim)
            .collect();
        let recovered_claims = batch
            .records
            .iter()
            .zip(&records)
            .filter(|(source, staged)| source != staged)
            .count();
        let source_hash = hash_records(batch.format_version, &batch.records)?;
        let staged_hash = hash_records(batch.format_version, &records)?;
        let scanned = u64::try_from(batch.records.len())
            .map_err(|_| BuzzPushStateImportError::InvalidBatch)?;
        Ok(BuzzPushStagingPlan {
            records,
            execution: BuzzPushExecutionState::all_disabled(),
            checkpoint: BuzzPushCheckpointProgress {
                final_source_sequence: batch
                    .records
                    .last()
                    .map(BuzzPushStateRecord::source_sequence)
                    .ok_or(BuzzPushStateImportError::InvalidBatch)?,
                source_hash,
                staged_hash,
                scanned,
                staged: scanned,
                recovered_claims: u64::try_from(recovered_claims)
                    .map_err(|_| BuzzPushStateImportError::InvalidBatch)?,
            },
        })
    }
}

fn validate_record(record: &BuzzPushStateRecord) -> Result<(), BuzzPushStateImportError> {
    if record.source_sequence() == 0 {
        return Err(BuzzPushStateImportError::InvalidSourceRecord);
    }
    match record {
        BuzzPushStateRecord::Lease(record) => {
            if !valid_installation_id(&record.installation_id)
                || record.source_created_at_seconds <= 0
                || record.generation == 0
                || record.expires_at_seconds <= 0
                || !valid_timestamp(record.updated_at_millis)
            {
                return Err(BuzzPushStateImportError::InvalidSourceRecord);
            }
            let complete_active = record.app_profile.as_deref().is_some_and(valid_opaque)
                && record.endpoint_hash.is_some()
                && record.endpoint_grant.as_deref().is_some_and(valid_opaque)
                && record.max_class.is_some()
                && record
                    .subscriptions
                    .as_ref()
                    .is_some_and(valid_subscriptions);
            let complete_tombstone = record.app_profile.is_none()
                && record.endpoint_hash.is_none()
                && record.endpoint_grant.is_none()
                && record.max_class.is_none()
                && record.subscriptions.is_none();
            if record.active != complete_active || !record.active && !complete_tombstone {
                return Err(BuzzPushStateImportError::InvalidState);
            }
        }
        BuzzPushStateRecord::Wake(record) => {
            if !valid_installation_id(&record.installation_id)
                || record.lease_generation == 0
                || record.expires_at_seconds <= 0
                || !valid_timestamp(record.next_attempt_at_millis)
                || !valid_timestamp(record.created_at_millis)
            {
                return Err(BuzzPushStateImportError::InvalidSourceRecord);
            }
            validate_claim_state(
                record.state == BuzzWakeState::Sending,
                record.claim_id,
                record.lease_until_millis,
            )?;
        }
        BuzzPushStateRecord::Match(record) => {
            if !valid_timestamp(record.next_attempt_at_millis)
                || !valid_timestamp(record.created_at_millis)
            {
                return Err(BuzzPushStateImportError::InvalidSourceRecord);
            }
            validate_claim_state(
                record.state == BuzzPushMatchState::Matching,
                record.claim_id,
                record.lease_until_millis,
            )?;
        }
    }
    Ok(())
}

fn validate_claim_state(
    claimed: bool,
    claim_id: Option<Uuid>,
    lease_until_millis: Option<u64>,
) -> Result<(), BuzzPushStateImportError> {
    if claimed != (claim_id.is_some() && lease_until_millis.is_some_and(valid_timestamp))
        || !claimed && (claim_id.is_some() || lease_until_millis.is_some())
    {
        return Err(BuzzPushStateImportError::InvalidState);
    }
    Ok(())
}

fn validate_references(records: &[BuzzPushStateRecord]) -> Result<(), BuzzPushStateImportError> {
    let mut leases = HashMap::new();
    let mut source_events = HashSet::new();
    let mut active_endpoints = HashSet::new();
    let mut wake_ids = HashSet::new();
    let mut wake_keys = HashSet::new();
    let mut match_events = HashSet::new();
    for record in records {
        match record {
            BuzzPushStateRecord::Lease(record) => {
                let address = (record.author_public_key, record.installation_id.clone());
                if leases.insert(address, record).is_some()
                    || !source_events.insert(record.source_event_id)
                {
                    return Err(BuzzPushStateImportError::DuplicateIdentity);
                }
                if record.active {
                    let endpoint = (
                        record.author_public_key,
                        record
                            .app_profile
                            .clone()
                            .ok_or(BuzzPushStateImportError::InvalidState)?,
                        record
                            .endpoint_hash
                            .ok_or(BuzzPushStateImportError::InvalidState)?,
                    );
                    if !active_endpoints.insert(endpoint) {
                        return Err(BuzzPushStateImportError::DuplicateIdentity);
                    }
                }
            }
            BuzzPushStateRecord::Wake(record) => {
                if !wake_ids.insert(record.wake_id)
                    || !wake_keys.insert((record.endpoint_hash, record.event_id))
                {
                    return Err(BuzzPushStateImportError::DuplicateIdentity);
                }
            }
            BuzzPushStateRecord::Match(record) => {
                if !match_events.insert(record.event_id) {
                    return Err(BuzzPushStateImportError::DuplicateIdentity);
                }
            }
        }
    }
    for record in records {
        let BuzzPushStateRecord::Wake(wake) = record else {
            continue;
        };
        let Some(lease) = leases.get(&(wake.author_public_key, wake.installation_id.clone()))
        else {
            return Err(BuzzPushStateImportError::InvalidReference);
        };
        if wake.lease_generation > lease.generation {
            return Err(BuzzPushStateImportError::InvalidReference);
        }
        if wake.lease_generation == lease.generation
            && lease.endpoint_hash != Some(wake.endpoint_hash)
        {
            return Err(BuzzPushStateImportError::InvalidReference);
        }
    }
    Ok(())
}

fn hash_records(
    format_version: u32,
    records: &[BuzzPushStateRecord],
) -> Result<[u8; 32], BuzzPushStateImportError> {
    let mut hasher = Sha256::new();
    hash_part(&mut hasher, &format_version.to_be_bytes());
    for record in records {
        hash_part(&mut hasher, &record.source_sequence().to_be_bytes());
        let bytes = serde_json::to_vec(record).map_err(BuzzPushStateImportError::Hashing)?;
        hash_part(&mut hasher, &bytes);
    }
    Ok(hasher.finalize().into())
}

fn valid_installation_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_INSTALLATION_ID_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn valid_opaque(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_OPAQUE_VALUE_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_subscriptions(value: &serde_json::Value) -> bool {
    value.is_array()
        && serde_json::to_vec(value).is_ok_and(|bytes| bytes.len() <= MAX_SUBSCRIPTIONS_BYTES)
}

fn valid_timestamp(value: u64) -> bool {
    value > 0 && i64::try_from(value).is_ok()
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
                TrustedTenantRoute::from_listener(community_id, "buzz-push-import")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn fixture_records(community_id: CommunityId) -> Vec<BuzzPushStateRecord> {
        vec![
            BuzzPushStateRecord::Lease(BuzzPushLeaseRecord {
                community_id,
                source_sequence: 1,
                author_public_key: [1; 32],
                installation_id: "ios-primary".to_owned(),
                source_event_id: [2; 32],
                source_created_at_seconds: 1_700_000_000,
                generation: 7,
                active: true,
                endpoint_enabled: true,
                app_profile: Some("buzz-ios-production".to_owned()),
                endpoint_hash: Some([3; 32]),
                endpoint_grant: Some("opaque-encrypted-endpoint-grant".to_owned()),
                max_class: Some(BuzzPushClass::TimeSensitive),
                subscriptions: Some(json!([{"kinds": [7, 9], "#p": ["self"]}])),
                expires_at_seconds: 1_800_000_000,
                updated_at_millis: 1_700_000_001_000,
            }),
            BuzzPushStateRecord::Wake(BuzzPushWakeRecord {
                community_id,
                source_sequence: 2,
                wake_id: Uuid::from_u128(10),
                author_public_key: [1; 32],
                installation_id: "ios-primary".to_owned(),
                lease_generation: 7,
                endpoint_hash: [3; 32],
                event_id: [4; 32],
                class: BuzzPushClass::Default,
                expires_at_seconds: 1_800_000_000,
                state: BuzzWakeState::Sending,
                attempts: 2,
                next_attempt_at_millis: 1_700_000_002_000,
                lease_until_millis: Some(1_700_000_012_000),
                claim_id: Some(Uuid::from_u128(11)),
                created_at_millis: 1_700_000_001_000,
            }),
            BuzzPushStateRecord::Match(BuzzPushMatchRecord {
                community_id,
                source_sequence: 3,
                event_id: [4; 32],
                state: BuzzPushMatchState::Matching,
                attempts: 1,
                next_attempt_at_millis: 1_700_000_002_000,
                lease_until_millis: Some(1_700_000_012_000),
                claim_id: Some(Uuid::from_u128(12)),
                created_at_millis: 1_700_000_001_000,
            }),
        ]
    }

    #[test]
    fn rejects_unknown_import_format_without_staging_records() {
        assert!(matches!(
            BuzzPushStateBatch::new(2, fixture_records(community(1))),
            Err(BuzzPushStateImportError::UnsupportedFormatVersion(2))
        ));
    }

    #[test]
    fn preserves_generations_and_opaque_values_without_sending_wakes() {
        let community_id = community(1);
        let batch =
            BuzzPushStateBatch::new(1, fixture_records(community_id)).expect("valid push state");
        let staged =
            BuzzPushStateImporter::stage(&tenant(community_id), &batch).expect("stage push state");

        let BuzzPushStateRecord::Lease(lease) = &staged.records[0] else {
            panic!("lease fixture")
        };
        assert_eq!(lease.generation, 7);
        assert_eq!(
            lease.endpoint_grant.as_deref(),
            Some("opaque-encrypted-endpoint-grant")
        );
        assert_eq!(
            lease.subscriptions,
            Some(json!([{"kinds": [7, 9], "#p": ["self"]}]))
        );
        assert_eq!(staged.checkpoint.recovered_claims, 2);
        assert_ne!(staged.checkpoint.source_hash, staged.checkpoint.staged_hash);
        assert!(!staged.execution.can_contact_provider());
    }

    #[test]
    fn recovers_nonportable_claims_to_pending_without_resetting_attempts() {
        let community_id = community(1);
        let batch =
            BuzzPushStateBatch::new(1, fixture_records(community_id)).expect("valid push state");
        let staged =
            BuzzPushStateImporter::stage(&tenant(community_id), &batch).expect("stage push state");

        let BuzzPushStateRecord::Wake(wake) = &staged.records[1] else {
            panic!("wake fixture")
        };
        assert_eq!(wake.state, BuzzWakeState::Pending);
        assert_eq!(wake.attempts, 2);
        assert_eq!(wake.claim_id, None);
        assert_eq!(wake.lease_until_millis, None);
        let BuzzPushStateRecord::Match(job) = &staged.records[2] else {
            panic!("match fixture")
        };
        assert_eq!(job.state, BuzzPushMatchState::Pending);
        assert_eq!(job.attempts, 1);
        assert_eq!(job.claim_id, None);
        assert_eq!(job.lease_until_millis, None);
    }

    #[test]
    fn rejects_cross_tenant_and_future_generation_wakes() {
        let community_id = community(1);
        let batch =
            BuzzPushStateBatch::new(1, fixture_records(community_id)).expect("valid push state");
        assert!(matches!(
            BuzzPushStateImporter::stage(&tenant(community(2)), &batch),
            Err(BuzzPushStateImportError::TenantBoundaryViolation)
        ));

        let mut records = fixture_records(community_id);
        let BuzzPushStateRecord::Wake(wake) = &mut records[1] else {
            panic!("wake fixture")
        };
        wake.lease_generation = 8;
        let batch = BuzzPushStateBatch::new(1, records).expect("well-formed push state");
        assert!(matches!(
            BuzzPushStateImporter::stage(&tenant(community_id), &batch),
            Err(BuzzPushStateImportError::InvalidReference)
        ));
    }
}
