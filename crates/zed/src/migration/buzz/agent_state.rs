use agent::{
    AgentMemoryRepository, AgentMemoryRepositoryError, AgentUsageRepository,
    AgentUsageRepositoryError, ManagedAgentSnapshotDocuments, ManagedAgentSnapshotError,
    ManagedAgentSnapshotRepository, MemoryWriteOutcome, StoredEncryptedMemory, StoredTurnUsage,
    UsageWriteOutcome,
};
use agent_settings::managed_agent::PrivateManagedAgentRecord;
use agent_settings::team::NostrPublicKey;
use db::sqlez::domain::Domain;
use db::sqlez_macros::sql;
use nostr_compat::PublicKey;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::HashSet;
use thiserror::Error;

use super::agent_staging::{
    BuzzAgentJsonKind, BuzzAgentPrivacyClass, BuzzAgentStagingBatch, BuzzAgentStagingImporter,
    BuzzAgentStagingPlan, BuzzAgentStagingRecord,
};

const MAX_MAPPINGS: usize = 10_000;
const MAX_TARGETS_PER_SOURCE: usize = 1_024;

pub struct BuzzAgentStateImportDatabase(db::sqlez::thread_safe_connection::ThreadSafeConnection);

impl Domain for BuzzAgentStateImportDatabase {
    const NAME: &str = stringify!(BuzzAgentStateImportDatabase);

    const MIGRATIONS: &[&str] = &[sql!(
        CREATE TABLE buzz_agent_state_import_records (
            owner_profile_id TEXT NOT NULL,
            source_idempotency_key BLOB NOT NULL,
            source_sequence INTEGER NOT NULL,
            source_path TEXT NOT NULL,
            source_content_hash BLOB NOT NULL,
            source_privacy_hash BLOB NOT NULL,
            target_manifest_json TEXT NOT NULL,
            target_hash BLOB NOT NULL,
            verified_at INTEGER NOT NULL,
            PRIMARY KEY (owner_profile_id, source_idempotency_key),
            CHECK (length(owner_profile_id) > 0),
            CHECK (length(source_idempotency_key) = 32),
            CHECK (source_sequence > 0),
            CHECK (length(source_path) > 0),
            CHECK (length(source_content_hash) = 32),
            CHECK (length(source_privacy_hash) = 32),
            CHECK (length(target_manifest_json) <= 1048576),
            CHECK (length(target_hash) = 32),
            CHECK (verified_at > 0)
        ) STRICT;

        CREATE TABLE buzz_agent_state_import_batches (
            owner_profile_id TEXT NOT NULL,
            source_hash BLOB NOT NULL,
            staged_hash BLOB NOT NULL,
            privacy_hash BLOB NOT NULL,
            target_hash BLOB NOT NULL,
            final_source_sequence INTEGER NOT NULL,
            imported_records INTEGER NOT NULL,
            retained_sources INTEGER NOT NULL,
            verified_at INTEGER NOT NULL,
            PRIMARY KEY (owner_profile_id, source_hash),
            CHECK (length(owner_profile_id) > 0),
            CHECK (length(source_hash) = 32),
            CHECK (length(staged_hash) = 32),
            CHECK (length(privacy_hash) = 32),
            CHECK (length(target_hash) = 32),
            CHECK (final_source_sequence > 0),
            CHECK (imported_records > 0),
            CHECK (retained_sources > 0),
            CHECK (verified_at > 0)
        ) STRICT;
    )];
}

db::static_connection!(BuzzAgentStateImportDatabase, []);

impl BuzzAgentStateImportDatabase {
    fn from_app_database(database: &db::AppDatabase) -> Self {
        Self(database.0.clone())
    }
}

#[derive(Clone)]
pub enum BuzzAgentCanonicalWrite {
    EncryptedMemory {
        authenticated_owner: PublicKey,
        record: StoredEncryptedMemory,
    },
    ManagedAgentSnapshot {
        authenticated_owner: NostrPublicKey,
        runtime: PrivateManagedAgentRecord,
        persona: Value,
        team: Option<Value>,
        created_at: u64,
    },
    TurnUsage {
        authenticated_owner: PublicKey,
        record: StoredTurnUsage,
    },
    RetainSourceEvidence,
}

impl std::fmt::Debug for BuzzAgentCanonicalWrite {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EncryptedMemory { record, .. } => formatter
                .debug_struct("EncryptedMemory")
                .field("coordinate", record.coordinate())
                .field("event_id", &record.event_id())
                .finish(),
            Self::ManagedAgentSnapshot { runtime, .. } => formatter
                .debug_struct("ManagedAgentSnapshot")
                .field("owner", runtime.owner_public_key())
                .field("agent", runtime.agent_public_key())
                .field("documents", &"<redacted>")
                .finish(),
            Self::TurnUsage { record, .. } => formatter
                .debug_struct("TurnUsage")
                .field("owner", &record.owner())
                .field("agent", &record.agent())
                .field("event_id", &record.event_id())
                .finish(),
            Self::RetainSourceEvidence => formatter.write_str("RetainSourceEvidence"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuzzAgentImportMapping {
    pub source_idempotency_key: [u8; 32],
    pub writes: Vec<BuzzAgentCanonicalWrite>,
}

impl BuzzAgentImportMapping {
    pub fn new(
        source_idempotency_key: [u8; 32],
        writes: Vec<BuzzAgentCanonicalWrite>,
    ) -> Result<Self, BuzzAgentStateImportError> {
        if writes.is_empty() || writes.len() > MAX_TARGETS_PER_SOURCE {
            return Err(BuzzAgentStateImportError::InvalidMapping);
        }
        Ok(Self {
            source_idempotency_key,
            writes,
        })
    }
}

pub struct BuzzAgentStateImportRequest {
    pub owner_profile_id: String,
    pub owner_public_key: PublicKey,
    pub plan: BuzzAgentStagingPlan,
    pub mappings: Vec<BuzzAgentImportMapping>,
    pub verified_at: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzAgentStateImportReceipt {
    pub owner_profile_id: String,
    pub source_hash: [u8; 32],
    pub staged_hash: [u8; 32],
    pub privacy_hash: [u8; 32],
    pub target_hash: [u8; 32],
    pub final_source_sequence: u64,
    pub imported_records: u64,
    pub retained_sources: u64,
    pub verified_at: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuzzAgentStateImportOutcome {
    Imported(BuzzAgentStateImportReceipt),
    AlreadyVerified(BuzzAgentStateImportReceipt),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzAgentRollbackEvidence {
    pub receipt: BuzzAgentStateImportReceipt,
    pub original_source_required: bool,
    pub activation_remains_disabled: bool,
}

#[derive(Debug, Error)]
pub enum BuzzAgentStateImportError {
    #[error("Buzz agent-state import request is invalid")]
    InvalidRequest,
    #[error("Buzz agent-state import mapping is invalid or incomplete")]
    InvalidMapping,
    #[error("Buzz agent-state import owner does not match the staged or canonical owner")]
    OwnerMismatch,
    #[error("Buzz agent-state import canonical target conflicts with existing state")]
    CanonicalConflict,
    #[error("Buzz agent-state import canonical target could not be verified")]
    VerificationFailed,
    #[error("Buzz agent-state import checkpoint conflicts with prior evidence")]
    CheckpointConflict,
    #[error("Buzz agent-state import is unavailable")]
    Unavailable(#[source] anyhow::Error),
}

#[derive(Clone)]
pub struct BuzzAgentStateImporter {
    database: BuzzAgentStateImportDatabase,
    memory_repository: AgentMemoryRepository,
    snapshot_repository: ManagedAgentSnapshotRepository,
    usage_repository: AgentUsageRepository,
}

impl BuzzAgentStateImporter {
    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self {
            database: BuzzAgentStateImportDatabase::from_app_database(database),
            memory_repository: AgentMemoryRepository::from_app_database(database),
            snapshot_repository: ManagedAgentSnapshotRepository::from_app_database(database),
            usage_repository: AgentUsageRepository::from_app_database(database),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_file_database(database_directory: &std::path::Path) -> Self {
        Self {
            database: BuzzAgentStateImportDatabase(
                db::open_db::<BuzzAgentStateImportDatabase>(database_directory, db::GlobalDbScope)
                    .await,
            ),
            memory_repository: AgentMemoryRepository::open_test_file_database(database_directory)
                .await,
            snapshot_repository: ManagedAgentSnapshotRepository::open_test_file_database(
                database_directory,
            )
            .await,
            usage_repository: AgentUsageRepository::open_test_file_database(database_directory)
                .await,
        }
    }

    pub async fn import(
        &self,
        request: BuzzAgentStateImportRequest,
    ) -> Result<BuzzAgentStateImportOutcome, BuzzAgentStateImportError> {
        validate_request(&request)?;
        let mut target_hasher = Sha256::new();
        let mut retained_sources = 0_u64;

        for (source, mapping) in request.plan.records.iter().zip(&request.mappings) {
            validate_mapping(
                source,
                mapping,
                request.owner_public_key,
                request.verified_at,
            )?;
            let checkpoint = self.load_record_checkpoint(
                &request.owner_profile_id,
                source,
                request.verified_at,
            )?;
            let (target_manifest, target_hash, retained) = match checkpoint {
                Some(checkpoint) => checkpoint,
                None => {
                    self.apply_mapping(
                        source,
                        mapping,
                        request.owner_public_key,
                        request.verified_at,
                    )
                    .await?
                }
            };
            hash_part(&mut target_hasher, &target_hash);
            retained_sources = retained_sources
                .checked_add(u64::from(retained))
                .ok_or(BuzzAgentStateImportError::InvalidRequest)?;
            self.checkpoint_record(
                &request.owner_profile_id,
                source,
                target_manifest,
                target_hash,
                request.verified_at,
            )
            .await?;
        }

        let target_hash = target_hasher.finalize().into();
        let receipt = BuzzAgentStateImportReceipt {
            owner_profile_id: request.owner_profile_id,
            source_hash: request.plan.checkpoint.source_hash,
            staged_hash: request.plan.checkpoint.staged_hash,
            privacy_hash: request.plan.checkpoint.privacy_hash,
            target_hash,
            final_source_sequence: request.plan.checkpoint.final_source_sequence,
            imported_records: request.plan.checkpoint.staged,
            retained_sources,
            verified_at: request.verified_at,
        };
        self.checkpoint_batch(receipt).await
    }

    pub fn rollback_evidence(
        &self,
        owner_profile_id: &str,
        source_hash: [u8; 32],
    ) -> Result<Option<BuzzAgentRollbackEvidence>, BuzzAgentStateImportError> {
        let mut select_receipt = self
            .database
            .select_row_bound::<(&str, &[u8]), BatchReceiptRow>(
                "SELECT staged_hash, privacy_hash, target_hash, final_source_sequence, \
                        imported_records, retained_sources, verified_at \
                 FROM buzz_agent_state_import_batches \
                 WHERE owner_profile_id = ? AND source_hash = ?",
            )
            .map_err(BuzzAgentStateImportError::Unavailable)?;
        let row = select_receipt((owner_profile_id, &source_hash))
            .map_err(BuzzAgentStateImportError::Unavailable)?;
        row.map(|row| decode_batch_receipt(owner_profile_id, source_hash, row))
            .transpose()
            .map(|receipt| {
                receipt.map(|receipt| BuzzAgentRollbackEvidence {
                    receipt,
                    original_source_required: true,
                    activation_remains_disabled: true,
                })
            })
    }

    fn load_record_checkpoint(
        &self,
        owner_profile_id: &str,
        source: &BuzzAgentStagingRecord,
        verified_at: u64,
    ) -> Result<Option<(String, [u8; 32], bool)>, BuzzAgentStateImportError> {
        let mut select_checkpoint = self
            .database
            .select_row_bound::<(&str, &[u8]), ExistingRecordReceiptRow>(
                "SELECT source_sequence, source_path, source_content_hash, \
                        source_privacy_hash, target_manifest_json, target_hash, verified_at \
                 FROM buzz_agent_state_import_records \
                 WHERE owner_profile_id = ? AND source_idempotency_key = ?",
            )
            .map_err(BuzzAgentStateImportError::Unavailable)?;
        let Some(row) = select_checkpoint((owner_profile_id, &source_idempotency_key(source)))
            .map_err(BuzzAgentStateImportError::Unavailable)?
        else {
            return Ok(None);
        };
        let (content_hash, privacy_hash) = source_hashes(source);
        if row.0 != to_i64(source_sequence(source))?
            || row.1 != source_path(source)
            || row.2 != content_hash
            || row.3 != privacy_hash
            || positive_u64(row.6)? > verified_at
        {
            return Err(BuzzAgentStateImportError::CheckpointConflict);
        }
        let target_hash = fixed_hash(row.5)?;
        if hash_bytes(row.4.as_bytes()) != target_hash {
            return Err(BuzzAgentStateImportError::CheckpointConflict);
        }
        let evidence: Vec<TargetEvidence> = serde_json::from_str(&row.4)
            .map_err(|_| BuzzAgentStateImportError::CheckpointConflict)?;
        let retained_source = evidence
            .iter()
            .filter(|item| item.kind == "retained_source");
        if retained_source.count() != 1
            || !evidence.iter().any(|item| {
                item.kind == "retained_source"
                    && item.target_id == source_path(source)
                    && item.target_hash == content_hash
            })
        {
            return Err(BuzzAgentStateImportError::CheckpointConflict);
        }
        Ok(Some((row.4, target_hash, true)))
    }

    async fn apply_mapping(
        &self,
        source: &BuzzAgentStagingRecord,
        mapping: &BuzzAgentImportMapping,
        owner_public_key: PublicKey,
        verified_at: u64,
    ) -> Result<(String, [u8; 32], bool), BuzzAgentStateImportError> {
        validate_mapping(source, mapping, owner_public_key, verified_at)?;
        let mut evidence = Vec::with_capacity(mapping.writes.len());
        let mut retained = false;
        for write in &mapping.writes {
            match write {
                BuzzAgentCanonicalWrite::EncryptedMemory {
                    authenticated_owner,
                    record,
                } => {
                    match self
                        .memory_repository
                        .store(*authenticated_owner, record)
                        .await
                        .map_err(map_memory_error)?
                    {
                        MemoryWriteOutcome::Stored | MemoryWriteOutcome::AlreadyCurrent => {}
                        MemoryWriteOutcome::Stale => {
                            return Err(BuzzAgentStateImportError::CanonicalConflict);
                        }
                    }
                    let loaded = self
                        .memory_repository
                        .load_for_owner(*authenticated_owner, record.coordinate(), verified_at)
                        .map_err(map_memory_error)?
                        .ok_or(BuzzAgentStateImportError::VerificationFailed)?;
                    if loaded != *record {
                        return Err(BuzzAgentStateImportError::VerificationFailed);
                    }
                    evidence.push(TargetEvidence::new(
                        "encrypted_memory",
                        record.event_id().to_hex(),
                        memory_target_hash(record),
                    ));
                }
                BuzzAgentCanonicalWrite::ManagedAgentSnapshot {
                    authenticated_owner,
                    runtime,
                    persona,
                    team,
                    created_at,
                } => {
                    let documents =
                        ManagedAgentSnapshotDocuments::new(persona.clone(), team.clone())
                            .map_err(map_snapshot_error)?;
                    let snapshot_id = self
                        .snapshot_repository
                        .create(authenticated_owner, runtime, documents, *created_at)
                        .await
                        .map_err(map_snapshot_error)?;
                    let loaded = self
                        .snapshot_repository
                        .load(
                            authenticated_owner,
                            runtime.agent_public_key(),
                            &snapshot_id,
                        )
                        .map_err(map_snapshot_error)?
                        .ok_or(BuzzAgentStateImportError::VerificationFailed)?;
                    if loaded.runtime() != runtime
                        || loaded.persona() != persona
                        || loaded.team() != team.as_ref()
                    {
                        return Err(BuzzAgentStateImportError::VerificationFailed);
                    }
                    evidence.push(TargetEvidence::new(
                        "managed_agent_snapshot",
                        snapshot_id.as_str().to_owned(),
                        hash_bytes(snapshot_id.as_str().as_bytes()),
                    ));
                }
                BuzzAgentCanonicalWrite::TurnUsage {
                    authenticated_owner,
                    record,
                } => {
                    match self
                        .usage_repository
                        .store(*authenticated_owner, record)
                        .await
                        .map_err(map_usage_error)?
                    {
                        UsageWriteOutcome::Stored | UsageWriteOutcome::AlreadyStored => {}
                        UsageWriteOutcome::Conflict => {
                            return Err(BuzzAgentStateImportError::CanonicalConflict);
                        }
                    }
                    let loaded = self
                        .usage_repository
                        .load_for_owner(*authenticated_owner, record.event_id(), verified_at)
                        .map_err(map_usage_error)?
                        .ok_or(BuzzAgentStateImportError::VerificationFailed)?;
                    if loaded != *record {
                        return Err(BuzzAgentStateImportError::VerificationFailed);
                    }
                    evidence.push(TargetEvidence::new(
                        "turn_usage",
                        record.event_id().to_hex(),
                        usage_target_hash(record)?,
                    ));
                }
                BuzzAgentCanonicalWrite::RetainSourceEvidence => {
                    retained = true;
                    let (content_hash, _) = source_hashes(source);
                    evidence.push(TargetEvidence::new(
                        "retained_source",
                        source_path(source).to_owned(),
                        content_hash,
                    ));
                }
            }
        }
        let manifest = serde_json::to_string(&evidence)
            .map_err(|error| BuzzAgentStateImportError::Unavailable(error.into()))?;
        let target_hash = hash_bytes(manifest.as_bytes());
        Ok((manifest, target_hash, retained))
    }

    async fn checkpoint_record(
        &self,
        owner_profile_id: &str,
        source: &BuzzAgentStagingRecord,
        target_manifest: String,
        target_hash: [u8; 32],
        verified_at: u64,
    ) -> Result<(), BuzzAgentStateImportError> {
        let owner_profile_id = owner_profile_id.to_owned();
        let source_idempotency_key = source_idempotency_key(source);
        let source_sequence = to_i64(source_sequence(source))?;
        let source_path = source_path(source).to_owned();
        let (source_content_hash, source_privacy_hash) = source_hashes(source);
        let verified_at = positive_i64(verified_at)?;
        self.database
            .write(move |connection| -> anyhow::Result<()> {
                connection
                    .exec_bound::<(&str, &[u8], i64, &str, &[u8], &[u8], &str, &[u8], i64)>(
                        "INSERT OR IGNORE INTO buzz_agent_state_import_records( \
                        owner_profile_id, source_idempotency_key, source_sequence, source_path, \
                        source_content_hash, source_privacy_hash, target_manifest_json, \
                        target_hash, verified_at \
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    )?((
                    &owner_profile_id,
                    &source_idempotency_key,
                    source_sequence,
                    &source_path,
                    &source_content_hash,
                    &source_privacy_hash,
                    &target_manifest,
                    &target_hash,
                    verified_at,
                ))?;
                let existing = connection.select_row_bound::<(&str, &[u8]), RecordReceiptRow>(
                    "SELECT source_sequence, source_path, source_content_hash, \
                                source_privacy_hash, target_manifest_json, target_hash \
                         FROM buzz_agent_state_import_records \
                         WHERE owner_profile_id = ? AND source_idempotency_key = ?",
                )?((&owner_profile_id, &source_idempotency_key))?
                .ok_or_else(|| anyhow::anyhow!("agent import receipt disappeared"))?;
                if existing
                    != (
                        source_sequence,
                        source_path,
                        source_content_hash.to_vec(),
                        source_privacy_hash.to_vec(),
                        target_manifest,
                        target_hash.to_vec(),
                    )
                {
                    anyhow::bail!("agent import record checkpoint conflict");
                }
                Ok(())
            })
            .await
            .map_err(|error| {
                if error.to_string().contains("checkpoint conflict") {
                    BuzzAgentStateImportError::CheckpointConflict
                } else {
                    BuzzAgentStateImportError::Unavailable(error)
                }
            })
    }

    async fn checkpoint_batch(
        &self,
        receipt: BuzzAgentStateImportReceipt,
    ) -> Result<BuzzAgentStateImportOutcome, BuzzAgentStateImportError> {
        let row = PersistedBatchReceipt::from_receipt(&receipt)?;
        self.database
            .write(
                move |connection| -> anyhow::Result<(bool, BatchReceiptRow)> {
                    connection
                        .exec_bound::<(&str, &[u8], &[u8], &[u8], &[u8], i64, i64, i64, i64)>(
                            "INSERT OR IGNORE INTO buzz_agent_state_import_batches( \
                        owner_profile_id, source_hash, staged_hash, privacy_hash, target_hash, \
                        final_source_sequence, imported_records, retained_sources, verified_at \
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        )?((
                        &row.owner_profile_id,
                        &row.source_hash,
                        &row.staged_hash,
                        &row.privacy_hash,
                        &row.target_hash,
                        row.final_source_sequence,
                        row.imported_records,
                        row.retained_sources,
                        row.verified_at,
                    ))?;
                    let inserted = changed_rows(connection)? == 1;
                    let existing =
                        connection.select_row_bound::<(&str, &[u8]), BatchReceiptRow>(
                            "SELECT staged_hash, privacy_hash, target_hash, final_source_sequence, \
                                imported_records, retained_sources, verified_at \
                         FROM buzz_agent_state_import_batches \
                         WHERE owner_profile_id = ? AND source_hash = ?",
                        )?((&row.owner_profile_id, &row.source_hash))?
                        .ok_or_else(|| anyhow::anyhow!("agent import batch receipt disappeared"))?;
                    Ok((inserted, existing))
                },
            )
            .await
            .map_err(BuzzAgentStateImportError::Unavailable)
            .and_then(|(inserted, existing)| {
                let existing =
                    decode_batch_receipt(&receipt.owner_profile_id, receipt.source_hash, existing)?;
                if !same_batch(&receipt, &existing) {
                    return Err(BuzzAgentStateImportError::CheckpointConflict);
                }
                Ok(if inserted {
                    BuzzAgentStateImportOutcome::Imported(existing)
                } else {
                    BuzzAgentStateImportOutcome::AlreadyVerified(existing)
                })
            })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct TargetEvidence {
    kind: String,
    target_id: String,
    target_hash: [u8; 32],
}

impl TargetEvidence {
    fn new(kind: &str, target_id: String, target_hash: [u8; 32]) -> Self {
        Self {
            kind: kind.to_owned(),
            target_id,
            target_hash,
        }
    }
}

type RecordReceiptRow = (i64, String, Vec<u8>, Vec<u8>, String, Vec<u8>);
type ExistingRecordReceiptRow = (i64, String, Vec<u8>, Vec<u8>, String, Vec<u8>, i64);
type BatchReceiptRow = (Vec<u8>, Vec<u8>, Vec<u8>, i64, i64, i64, i64);

struct PersistedBatchReceipt {
    owner_profile_id: String,
    source_hash: Vec<u8>,
    staged_hash: Vec<u8>,
    privacy_hash: Vec<u8>,
    target_hash: Vec<u8>,
    final_source_sequence: i64,
    imported_records: i64,
    retained_sources: i64,
    verified_at: i64,
}

impl PersistedBatchReceipt {
    fn from_receipt(
        receipt: &BuzzAgentStateImportReceipt,
    ) -> Result<Self, BuzzAgentStateImportError> {
        Ok(Self {
            owner_profile_id: receipt.owner_profile_id.clone(),
            source_hash: receipt.source_hash.to_vec(),
            staged_hash: receipt.staged_hash.to_vec(),
            privacy_hash: receipt.privacy_hash.to_vec(),
            target_hash: receipt.target_hash.to_vec(),
            final_source_sequence: to_i64(receipt.final_source_sequence)?,
            imported_records: to_i64(receipt.imported_records)?,
            retained_sources: to_i64(receipt.retained_sources)?,
            verified_at: positive_i64(receipt.verified_at)?,
        })
    }
}

fn validate_request(
    request: &BuzzAgentStateImportRequest,
) -> Result<(), BuzzAgentStateImportError> {
    if request.owner_profile_id.is_empty()
        || request.owner_profile_id.len() > 512
        || request.verified_at == 0
        || request.plan.records.is_empty()
        || request.plan.records.len() > MAX_MAPPINGS
        || request.plan.records.len() != request.mappings.len()
        || request.plan.activation.can_activate()
        || request.plan.checkpoint.staged
            != u64::try_from(request.plan.records.len())
                .map_err(|_| BuzzAgentStateImportError::InvalidRequest)?
    {
        return Err(BuzzAgentStateImportError::InvalidRequest);
    }
    let first_source = request
        .plan
        .records
        .first()
        .ok_or(BuzzAgentStateImportError::InvalidRequest)?;
    let staged_batch =
        BuzzAgentStagingBatch::new(1, source_owner(first_source), request.plan.records.clone())
            .map_err(|_| BuzzAgentStateImportError::InvalidRequest)?;
    let verified_plan = BuzzAgentStagingImporter::stage(source_owner(first_source), &staged_batch)
        .map_err(|_| BuzzAgentStateImportError::InvalidRequest)?;
    if verified_plan != request.plan {
        return Err(BuzzAgentStateImportError::InvalidRequest);
    }
    let mut source_keys = HashSet::new();
    for (source, mapping) in request.plan.records.iter().zip(&request.mappings) {
        if source_owner(source) != request.owner_profile_id
            || source_idempotency_key(source) != mapping.source_idempotency_key
            || !source_keys.insert(mapping.source_idempotency_key)
        {
            return Err(BuzzAgentStateImportError::InvalidMapping);
        }
    }
    Ok(())
}

fn validate_mapping(
    source: &BuzzAgentStagingRecord,
    mapping: &BuzzAgentImportMapping,
    owner_public_key: PublicKey,
    verified_at: u64,
) -> Result<(), BuzzAgentStateImportError> {
    let mut memory_count = 0_usize;
    let mut snapshot_count = 0_usize;
    let mut usage_count = 0_usize;
    let mut retained_source_count = 0_usize;
    for write in &mapping.writes {
        match write {
            BuzzAgentCanonicalWrite::EncryptedMemory {
                authenticated_owner,
                record,
            } => {
                if *authenticated_owner != owner_public_key
                    || record.coordinate().owner() != owner_public_key
                {
                    return Err(BuzzAgentStateImportError::OwnerMismatch);
                }
                if record.created_at() > verified_at
                    || record
                        .retention()
                        .expires_at()
                        .is_some_and(|expiry| expiry <= verified_at)
                {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
                memory_count += 1;
            }
            BuzzAgentCanonicalWrite::ManagedAgentSnapshot {
                authenticated_owner,
                runtime,
                created_at,
                ..
            } => {
                if authenticated_owner.as_str() != owner_public_key.to_hex()
                    || runtime.owner_public_key() != authenticated_owner
                {
                    return Err(BuzzAgentStateImportError::OwnerMismatch);
                }
                if *created_at == 0 || *created_at > verified_at {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
                snapshot_count += 1;
            }
            BuzzAgentCanonicalWrite::TurnUsage {
                authenticated_owner,
                record,
            } => {
                if *authenticated_owner != owner_public_key || record.owner() != owner_public_key {
                    return Err(BuzzAgentStateImportError::OwnerMismatch);
                }
                if record.created_at() > verified_at
                    || record
                        .retention()
                        .expires_at()
                        .is_some_and(|expiry| expiry <= verified_at)
                {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
                usage_count += 1;
            }
            BuzzAgentCanonicalWrite::RetainSourceEvidence => retained_source_count += 1,
        }
    }
    if retained_source_count != 1 {
        return Err(BuzzAgentStateImportError::InvalidMapping);
    }
    match source {
        BuzzAgentStagingRecord::Json(record) => match record.kind {
            BuzzAgentJsonKind::AgentSnapshot => {
                if snapshot_count != 1
                    || record.privacy_class == BuzzAgentPrivacyClass::PrivateMemory
                        && memory_count == 0
                {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
            }
            BuzzAgentJsonKind::TeamSnapshot => {
                if snapshot_count == 0 {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
            }
            BuzzAgentJsonKind::EncryptedAgentSnapshot
            | BuzzAgentJsonKind::ManagedAgent
            | BuzzAgentJsonKind::Persona
            | BuzzAgentJsonKind::Team => {
                if memory_count != 0 || snapshot_count != 0 {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
            }
        },
        BuzzAgentStagingRecord::ArchivedEvidence(record) => {
            if PublicKey::from_bytes(record.identity_public_key) != owner_public_key {
                return Err(BuzzAgentStateImportError::OwnerMismatch);
            }
            if u64::try_from(record.archived_at_seconds)
                .map_err(|_| BuzzAgentStateImportError::InvalidMapping)?
                > verified_at
                || memory_count != 0
                || snapshot_count != 0
            {
                return Err(BuzzAgentStateImportError::InvalidMapping);
            }
            match record.event_kind {
                44_200 if usage_count != 1 => {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
                24_200 if usage_count != 0 => {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
                _ => {}
            }
            for write in &mapping.writes {
                if let BuzzAgentCanonicalWrite::TurnUsage { record: usage, .. } = write
                    && (usage.event_id().as_bytes() != &record.event_id
                        || usage.agent().as_bytes() != &record.agent_public_key)
                {
                    return Err(BuzzAgentStateImportError::InvalidMapping);
                }
            }
        }
    }
    if usage_count != 0 && !matches!(source, BuzzAgentStagingRecord::ArchivedEvidence(_)) {
        return Err(BuzzAgentStateImportError::InvalidMapping);
    }
    Ok(())
}

fn memory_target_hash(record: &StoredEncryptedMemory) -> [u8; 32] {
    hash_parts(&[
        record.coordinate().owner().as_bytes(),
        record.coordinate().agent().as_bytes(),
        record.event_id().as_bytes(),
        &record.created_at().to_be_bytes(),
        record.ciphertext().wire_value().as_bytes(),
        record.ciphertext_sha256(),
        &record.retention().generation().to_be_bytes(),
        &record
            .retention()
            .expires_at()
            .unwrap_or_default()
            .to_be_bytes(),
    ])
}

fn usage_target_hash(record: &StoredTurnUsage) -> Result<[u8; 32], BuzzAgentStateImportError> {
    let payload = serde_json::to_vec(record.payload())
        .map_err(|error| BuzzAgentStateImportError::Unavailable(error.into()))?;
    Ok(hash_parts(&[
        record.owner().as_bytes(),
        record.agent().as_bytes(),
        record.event_id().as_bytes(),
        &record.created_at().to_be_bytes(),
        &payload,
        record.ciphertext().wire_value().as_bytes(),
        &record.retention().generation().to_be_bytes(),
        &record
            .retention()
            .expires_at()
            .unwrap_or_default()
            .to_be_bytes(),
    ]))
}

fn source_owner(source: &BuzzAgentStagingRecord) -> &str {
    match source {
        BuzzAgentStagingRecord::Json(record) => &record.owner_profile_id,
        BuzzAgentStagingRecord::ArchivedEvidence(record) => &record.owner_profile_id,
    }
}

fn source_sequence(source: &BuzzAgentStagingRecord) -> u64 {
    match source {
        BuzzAgentStagingRecord::Json(record) => record.source_sequence,
        BuzzAgentStagingRecord::ArchivedEvidence(record) => record.source_sequence,
    }
}

fn source_path(source: &BuzzAgentStagingRecord) -> &str {
    match source {
        BuzzAgentStagingRecord::Json(record) => &record.source_path,
        BuzzAgentStagingRecord::ArchivedEvidence(record) => &record.source_path,
    }
}

fn source_idempotency_key(source: &BuzzAgentStagingRecord) -> [u8; 32] {
    match source {
        BuzzAgentStagingRecord::Json(record) => record.idempotency_key,
        BuzzAgentStagingRecord::ArchivedEvidence(record) => record.idempotency_key,
    }
}

fn source_hashes(source: &BuzzAgentStagingRecord) -> ([u8; 32], [u8; 32]) {
    match source {
        BuzzAgentStagingRecord::Json(record) => (record.source_hash, record.privacy_hash),
        BuzzAgentStagingRecord::ArchivedEvidence(record) => (
            record.archived_payload_hash,
            hash_parts(&[
                b"private_telemetry",
                &record.event_kind.to_be_bytes(),
                &record.event_id,
            ]),
        ),
    }
}

fn decode_batch_receipt(
    owner_profile_id: &str,
    source_hash: [u8; 32],
    row: BatchReceiptRow,
) -> Result<BuzzAgentStateImportReceipt, BuzzAgentStateImportError> {
    Ok(BuzzAgentStateImportReceipt {
        owner_profile_id: owner_profile_id.to_owned(),
        source_hash,
        staged_hash: fixed_hash(row.0)?,
        privacy_hash: fixed_hash(row.1)?,
        target_hash: fixed_hash(row.2)?,
        final_source_sequence: positive_u64(row.3)?,
        imported_records: positive_u64(row.4)?,
        retained_sources: positive_u64(row.5)?,
        verified_at: positive_u64(row.6)?,
    })
}

fn same_batch(
    expected: &BuzzAgentStateImportReceipt,
    actual: &BuzzAgentStateImportReceipt,
) -> bool {
    expected.owner_profile_id == actual.owner_profile_id
        && expected.source_hash == actual.source_hash
        && expected.staged_hash == actual.staged_hash
        && expected.privacy_hash == actual.privacy_hash
        && expected.target_hash == actual.target_hash
        && expected.final_source_sequence == actual.final_source_sequence
        && expected.imported_records == actual.imported_records
        && expected.retained_sources == actual.retained_sources
}

fn map_memory_error(error: AgentMemoryRepositoryError) -> BuzzAgentStateImportError {
    BuzzAgentStateImportError::Unavailable(error.into())
}

fn map_snapshot_error(error: ManagedAgentSnapshotError) -> BuzzAgentStateImportError {
    BuzzAgentStateImportError::Unavailable(error.into())
}

fn map_usage_error(error: AgentUsageRepositoryError) -> BuzzAgentStateImportError {
    BuzzAgentStateImportError::Unavailable(error.into())
}

fn changed_rows(connection: &db::sqlez::connection::Connection) -> anyhow::Result<i64> {
    connection.select_row_bound::<(), i64>("SELECT changes()")?(())?
        .ok_or_else(|| anyhow::anyhow!("SQLite did not report a changed-row count"))
}

fn fixed_hash(value: Vec<u8>) -> Result<[u8; 32], BuzzAgentStateImportError> {
    value
        .try_into()
        .map_err(|_| BuzzAgentStateImportError::CheckpointConflict)
}

fn to_i64(value: u64) -> Result<i64, BuzzAgentStateImportError> {
    i64::try_from(value).map_err(|_| BuzzAgentStateImportError::InvalidRequest)
}

fn positive_i64(value: u64) -> Result<i64, BuzzAgentStateImportError> {
    if value == 0 {
        return Err(BuzzAgentStateImportError::InvalidRequest);
    }
    to_i64(value)
}

fn positive_u64(value: i64) -> Result<u64, BuzzAgentStateImportError> {
    let value = u64::try_from(value).map_err(|_| BuzzAgentStateImportError::CheckpointConflict)?;
    if value == 0 {
        return Err(BuzzAgentStateImportError::CheckpointConflict);
    }
    Ok(value)
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hash_parts(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hash_part(&mut hasher, part);
    }
    hasher.finalize().into()
}

fn hash_part(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(bytes.len().to_be_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::{MemoryRetention, UsageRetention, encrypt_turn_usage_for_owner};
    use agent_settings::managed_agent::{ManagedAgentConfiguration, RuntimeId};
    use agent_settings::team::NostrEventId;
    use nostr_compat::EventId;
    use nostr_compat::agent_memory::encrypt_engram_for_owner;
    use nostr_compat::buzz_nips::agent_activity::{
        AgentTurnMetricPayload, EngramBody, StopReason, TokenCounts,
    };
    use serde_json::json;
    use std::collections::BTreeMap;

    use super::super::agent_staging::{
        BuzzAgentJsonSource, BuzzAgentJsonStagingRecord, BuzzAgentStagingBatch,
        BuzzAgentStagingImporter, BuzzArchivedAgentEvidenceRecord, BuzzArchivedAgentEvidenceSource,
    };

    const OWNER_PROFILE_ID: &str = "profile-a";
    const VERIFIED_AT: u64 = 1_800_000_000;
    const AGENT_SECRET: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ];

    fn owner() -> PublicKey {
        PublicKey::from_hex("c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5")
            .expect("fixture owner public key must be valid")
    }

    fn agent() -> PublicKey {
        PublicKey::from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")
            .expect("fixture agent public key must be valid")
    }

    fn snapshot_source() -> BuzzAgentStagingRecord {
        let source_bytes = serde_json::to_vec(&json!({
            "format": "buzz-agent-snapshot",
            "version": 1,
            "definition": {"name": "Builder"},
            "profile": {"displayName": "Builder"},
            "memory": {
                "level": "core",
                "entries": [{"slug": "role", "body": "Build safely"}]
            }
        }))
        .expect("serialize snapshot fixture");
        BuzzAgentStagingRecord::Json(
            BuzzAgentJsonStagingRecord::from_source(
                BuzzAgentJsonSource {
                    owner_profile_id: OWNER_PROFILE_ID,
                    source_sequence: 1,
                    source_path: "agents/snapshots/agent-a.agent.json",
                    semantic_id: "agent-a-snapshot",
                    kind: BuzzAgentJsonKind::AgentSnapshot,
                    source_schema_version: 1,
                    source_bytes: &source_bytes,
                },
                Vec::new(),
            )
            .expect("stage snapshot fixture"),
        )
    }

    fn archived_source(
        source_sequence: u64,
        source_path: &str,
        event_kind: u32,
        event_id: [u8; 32],
        payload_hash: [u8; 32],
    ) -> BuzzAgentStagingRecord {
        BuzzAgentStagingRecord::ArchivedEvidence(
            BuzzArchivedAgentEvidenceRecord::new(BuzzArchivedAgentEvidenceSource {
                owner_profile_id: OWNER_PROFILE_ID,
                source_sequence,
                source_path,
                event_id,
                event_kind,
                identity_public_key: *owner().as_bytes(),
                agent_public_key: *agent().as_bytes(),
                relay_url: "wss://relay.example.com",
                archived_at_seconds: 1_790_000_000,
                archived_payload_hash: payload_hash,
            })
            .expect("stage archived fixture"),
        )
    }

    fn memory() -> StoredEncryptedMemory {
        let encrypted = encrypt_engram_for_owner(
            &AGENT_SECRET,
            owner(),
            &EngramBody::Memory {
                slug: "mem/role".to_owned(),
                value: Some("Build safely".to_owned()),
            },
        )
        .expect("encrypt memory fixture");
        StoredEncryptedMemory::new(
            &encrypted,
            EventId::from_bytes([11; 32]),
            1_790_000_001,
            MemoryRetention::new(1, None).expect("memory retention fixture"),
        )
        .expect("stored memory fixture")
    }

    fn runtime() -> PrivateManagedAgentRecord {
        PrivateManagedAgentRecord::new(
            NostrPublicKey::parse(owner().to_hex()).expect("snapshot owner fixture"),
            NostrPublicKey::parse(agent().to_hex()).expect("snapshot agent fixture"),
            NostrEventId::parse("33".repeat(32)).expect("snapshot event fixture"),
            ManagedAgentConfiguration::new(
                RuntimeId::parse("claude-code").expect("runtime fixture"),
                None,
                None,
                BTreeMap::new(),
            )
            .expect("configuration fixture"),
        )
        .expect("runtime snapshot fixture")
    }

    fn usage() -> StoredTurnUsage {
        let payload = AgentTurnMetricPayload {
            harness: "buzz-import".to_owned(),
            model: Some("claude-sonnet-4-5".to_owned()),
            channel_id: Some("private-channel".to_owned()),
            session_id: None,
            turn_id: Some("turn-1".to_owned()),
            turn_seq: None,
            timestamp: "2026-08-23T10:00:00Z".to_owned(),
            turn: Some(TokenCounts {
                input_tokens: Some(100),
                output_tokens: Some(20),
                total_tokens: Some(120),
                cost_usd: Some(0.03),
                cache_read_tokens: None,
                cache_write_tokens: None,
            }),
            cumulative: None,
            delta_reliable: true,
            stop_reason: Some(StopReason::EndTurn),
            pricing_identity: None,
        };
        let encrypted = encrypt_turn_usage_for_owner(&AGENT_SECRET, owner(), &payload)
            .expect("encrypt usage fixture");
        let created_at = 1_790_000_003;
        let event_id = encrypted
            .to_canonical_event(created_at)
            .event_id()
            .expect("derive usage fixture event ID");
        StoredTurnUsage::new(
            &encrypted,
            event_id,
            created_at,
            payload,
            UsageRetention::new(1, None).expect("usage retention fixture"),
        )
        .expect("stored usage fixture")
    }

    fn fixture_plan() -> (BuzzAgentStagingPlan, Vec<BuzzAgentImportMapping>) {
        let memory = memory();
        let runtime = runtime();
        let usage = usage();
        let records = vec![
            snapshot_source(),
            archived_source(2, "archives/agent-frame-1.json", 24_200, [42; 32], [52; 32]),
            archived_source(
                3,
                "archives/agent-usage-1.json",
                44_200,
                *usage.event_id().as_bytes(),
                [53; 32],
            ),
        ];
        let batch = BuzzAgentStagingBatch::new(1, OWNER_PROFILE_ID, records)
            .expect("build staged fixture batch");
        let plan =
            BuzzAgentStagingImporter::stage(OWNER_PROFILE_ID, &batch).expect("stage fixture batch");
        let mappings = vec![
            BuzzAgentImportMapping::new(
                source_idempotency_key(&plan.records[0]),
                vec![
                    BuzzAgentCanonicalWrite::EncryptedMemory {
                        authenticated_owner: owner(),
                        record: memory,
                    },
                    BuzzAgentCanonicalWrite::ManagedAgentSnapshot {
                        authenticated_owner: NostrPublicKey::parse(owner().to_hex())
                            .expect("snapshot owner fixture"),
                        runtime,
                        persona: json!({"name": "Builder"}),
                        team: None,
                        created_at: 1_790_000_002,
                    },
                    BuzzAgentCanonicalWrite::RetainSourceEvidence,
                ],
            )
            .expect("snapshot mapping fixture"),
            BuzzAgentImportMapping::new(
                source_idempotency_key(&plan.records[1]),
                vec![BuzzAgentCanonicalWrite::RetainSourceEvidence],
            )
            .expect("observer archive mapping fixture"),
            BuzzAgentImportMapping::new(
                source_idempotency_key(&plan.records[2]),
                vec![
                    BuzzAgentCanonicalWrite::TurnUsage {
                        authenticated_owner: owner(),
                        record: usage,
                    },
                    BuzzAgentCanonicalWrite::RetainSourceEvidence,
                ],
            )
            .expect("usage archive mapping fixture"),
        ];
        (plan, mappings)
    }

    fn request(
        plan: BuzzAgentStagingPlan,
        mappings: Vec<BuzzAgentImportMapping>,
    ) -> BuzzAgentStateImportRequest {
        BuzzAgentStateImportRequest {
            owner_profile_id: OWNER_PROFILE_ID.to_owned(),
            owner_public_key: owner(),
            plan,
            mappings,
            verified_at: VERIFIED_AT,
        }
    }

    #[gpui::test]
    async fn imports_all_agent_state_fixtures_idempotently_across_restart(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let database_directory = tempfile::tempdir().expect("create fixture database directory");
        let importer =
            BuzzAgentStateImporter::open_test_file_database(database_directory.path()).await;
        let (plan, mappings) = fixture_plan();
        let checkpoint = plan.checkpoint;
        let outcome = importer
            .import(request(plan.clone(), mappings.clone()))
            .await
            .expect("import agent state fixtures");
        let BuzzAgentStateImportOutcome::Imported(receipt) = outcome else {
            panic!("first fixture import must create a receipt");
        };
        assert_eq!(receipt.source_hash, checkpoint.source_hash);
        assert_eq!(receipt.staged_hash, checkpoint.staged_hash);
        assert_eq!(receipt.privacy_hash, checkpoint.privacy_hash);
        assert_eq!(receipt.imported_records, 3);
        assert_eq!(receipt.retained_sources, 3);
        assert_ne!(receipt.target_hash, [0; 32]);
        drop(importer);

        let restarted =
            BuzzAgentStateImporter::open_test_file_database(database_directory.path()).await;
        assert_eq!(
            restarted
                .import(request(plan, mappings))
                .await
                .expect("replay agent state fixtures"),
            BuzzAgentStateImportOutcome::AlreadyVerified(receipt.clone())
        );
        let rollback = restarted
            .rollback_evidence(OWNER_PROFILE_ID, checkpoint.source_hash)
            .expect("load rollback evidence")
            .expect("rollback evidence exists");
        assert_eq!(rollback.receipt, receipt);
        assert!(rollback.original_source_required);
        assert!(rollback.activation_remains_disabled);
    }

    #[gpui::test]
    async fn failed_archive_mapping_resumes_from_verified_record_checkpoints(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let database_directory = tempfile::tempdir().expect("create fixture database directory");
        let importer =
            BuzzAgentStateImporter::open_test_file_database(database_directory.path()).await;
        let (plan, mappings) = fixture_plan();
        let mut incomplete = mappings.clone();
        incomplete[2] = BuzzAgentImportMapping::new(
            source_idempotency_key(&plan.records[2]),
            vec![BuzzAgentCanonicalWrite::RetainSourceEvidence],
        )
        .expect("incomplete mapping fixture");
        assert!(matches!(
            importer.import(request(plan.clone(), incomplete)).await,
            Err(BuzzAgentStateImportError::InvalidMapping)
        ));
        let checkpoint_count = importer
            .database
            .select_row_bound::<(), i64>("SELECT COUNT(*) FROM buzz_agent_state_import_records")
            .expect("prepare checkpoint count query")(())
        .expect("count checkpoint records")
        .expect("checkpoint count row");
        assert_eq!(checkpoint_count, 2);
        assert!(
            importer
                .rollback_evidence(OWNER_PROFILE_ID, plan.checkpoint.source_hash)
                .expect("query rollback evidence")
                .is_none()
        );

        let outcome = importer
            .import(request(plan.clone(), mappings))
            .await
            .expect("resume agent state import");
        let BuzzAgentStateImportOutcome::Imported(receipt) = outcome else {
            panic!("resumed fixture import must complete the batch");
        };
        assert_eq!(receipt.imported_records, 3);
        assert_eq!(receipt.retained_sources, 3);
        assert_eq!(receipt.source_hash, plan.checkpoint.source_hash);
    }

    #[gpui::test]
    async fn rejects_cross_owner_canonical_materialization_before_writing(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let database_directory = tempfile::tempdir().expect("create fixture database directory");
        let importer =
            BuzzAgentStateImporter::open_test_file_database(database_directory.path()).await;
        let (plan, mut mappings) = fixture_plan();
        let wrong_owner =
            PublicKey::from_hex("f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9")
                .expect("fixture wrong owner public key");
        if let BuzzAgentCanonicalWrite::EncryptedMemory {
            authenticated_owner,
            ..
        } = &mut mappings[0].writes[0]
        {
            *authenticated_owner = wrong_owner;
        }
        assert!(matches!(
            importer.import(request(plan, mappings)).await,
            Err(BuzzAgentStateImportError::OwnerMismatch)
        ));
        let checkpoint_count = importer
            .database
            .select_row_bound::<(), i64>("SELECT COUNT(*) FROM buzz_agent_state_import_records")
            .expect("prepare checkpoint count query")(())
        .expect("count checkpoint records")
        .expect("checkpoint count row");
        assert_eq!(checkpoint_count, 0);

        let (mut tampered_plan, valid_mappings) = fixture_plan();
        tampered_plan.checkpoint.staged_hash[0] ^= 1;
        assert!(matches!(
            importer
                .import(request(tampered_plan, valid_mappings))
                .await,
            Err(BuzzAgentStateImportError::InvalidRequest)
        ));
    }
}
