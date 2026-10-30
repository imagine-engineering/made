use agent_settings::{
    managed_agent::{ManagedAgentVersion, PrivateManagedAgentRecord},
    team::{NostrEventId, NostrPublicKey},
};
use db::sqlez_macros::sql;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest as _, Sha256};
use sqlez::domain::Domain;
use std::fmt;
use thiserror::Error;

use crate::managed_agents::{decode_snapshot, encode_snapshot};

const MAX_DOCUMENT_BYTES: usize = 1_048_576;
const MAX_RETAINED_SNAPSHOTS: usize = 1_024;

pub struct ManagedAgentSnapshotDatabase(db::sqlez::thread_safe_connection::ThreadSafeConnection);

impl Domain for ManagedAgentSnapshotDatabase {
    const NAME: &str = stringify!(ManagedAgentSnapshotDatabase);

    const MIGRATIONS: &[&str] = &[sql!(
        CREATE TABLE managed_agent_lifecycle_snapshots (
            owner_public_key TEXT NOT NULL,
            agent_public_key TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            provenance_json TEXT NOT NULL,
            persona_json TEXT NOT NULL,
            team_json TEXT,
            runtime_json TEXT NOT NULL,
            integrity_json TEXT NOT NULL,
            PRIMARY KEY (owner_public_key, agent_public_key, snapshot_id),
            CHECK (length(owner_public_key) = 64),
            CHECK (length(agent_public_key) = 64),
            CHECK (length(snapshot_id) = 64),
            CHECK (created_at > 0),
            CHECK (length(provenance_json) <= 1024),
            CHECK (length(persona_json) <= 1048576),
            CHECK (team_json IS NULL OR length(team_json) <= 1048576),
            CHECK (length(runtime_json) <= 1048576),
            CHECK (length(integrity_json) <= 512)
        ) STRICT;

        CREATE INDEX managed_agent_lifecycle_snapshots_order
            ON managed_agent_lifecycle_snapshots(
                owner_public_key, agent_public_key, created_at, snapshot_id
            );

        CREATE TABLE managed_agent_snapshot_heads (
            owner_public_key TEXT NOT NULL,
            agent_public_key TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            PRIMARY KEY (owner_public_key, agent_public_key),
            FOREIGN KEY (owner_public_key, agent_public_key, snapshot_id)
                REFERENCES managed_agent_lifecycle_snapshots(
                    owner_public_key, agent_public_key, snapshot_id
                ) ON DELETE RESTRICT
        ) STRICT;

        CREATE TABLE managed_agent_snapshot_compactions (
            owner_public_key TEXT NOT NULL,
            agent_public_key TEXT NOT NULL,
            compaction_id TEXT NOT NULL,
            compacted_at INTEGER NOT NULL,
            head_snapshot_id TEXT NOT NULL,
            removed_snapshot_ids_json TEXT NOT NULL,
            removed_chain_sha256 BLOB NOT NULL,
            PRIMARY KEY (owner_public_key, agent_public_key, compaction_id),
            CHECK (length(compaction_id) = 64),
            CHECK (compacted_at > 0),
            CHECK (length(head_snapshot_id) = 64),
            CHECK (length(removed_chain_sha256) = 32)
        ) STRICT;
    )];
}

db::static_connection!(ManagedAgentSnapshotDatabase, []);

impl ManagedAgentSnapshotDatabase {
    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self(database.0.clone())
    }
}

#[derive(Clone)]
pub struct ManagedAgentSnapshotRepository {
    database: ManagedAgentSnapshotDatabase,
}

impl ManagedAgentSnapshotRepository {
    pub fn global(cx: &gpui::App) -> Self {
        Self {
            database: ManagedAgentSnapshotDatabase::global(cx),
        }
    }

    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self {
            database: ManagedAgentSnapshotDatabase::from_app_database(database),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_database(name: &'static str) -> Self {
        Self {
            database: ManagedAgentSnapshotDatabase::open_test_db(name).await,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_file_database(database_directory: &std::path::Path) -> Self {
        Self {
            database: ManagedAgentSnapshotDatabase(
                db::open_db::<ManagedAgentSnapshotDatabase>(database_directory, db::GlobalDbScope)
                    .await,
            ),
        }
    }

    pub async fn create(
        &self,
        authenticated_owner: &NostrPublicKey,
        runtime: &PrivateManagedAgentRecord,
        documents: ManagedAgentSnapshotDocuments,
        created_at: u64,
    ) -> Result<ManagedAgentSnapshotId, ManagedAgentSnapshotError> {
        authorize_owner(authenticated_owner, runtime.owner_public_key())?;
        if created_at == 0 {
            return Err(ManagedAgentSnapshotError::InvalidTime);
        }
        let created_at = to_i64(created_at)?;
        let owner_public_key = runtime.owner_public_key().as_str().to_string();
        let agent_public_key = runtime.agent_public_key().as_str().to_string();
        let source_generation = to_i64(runtime.version().generation())?;
        let source_event_id = runtime.version().event_id().as_str().to_string();
        let persona_json = documents.persona_json;
        let team_json = documents.team_json;
        let runtime_json = encode_snapshot(runtime)
            .map_err(|error| ManagedAgentSnapshotError::Unavailable(error.into()))?;
        validate_document_bytes(runtime_json.as_bytes())?;
        let persona_hash = hash_bytes(persona_json.as_bytes());
        let team_hash = team_json.as_ref().map(|team| hash_bytes(team.as_bytes()));
        let runtime_hash = hash_bytes(runtime_json.as_bytes());
        let snapshot_id = creation_fingerprint(
            &owner_public_key,
            &agent_public_key,
            source_generation,
            &source_event_id,
            created_at,
            &persona_hash,
            team_hash.as_ref(),
            &runtime_hash,
        );
        let snapshot_id_string = snapshot_id.as_str().to_string();

        self.database
            .write(
                move |connection| -> anyhow::Result<ManagedAgentSnapshotId> {
                    connection.with_savepoint("create_managed_agent_lifecycle_snapshot", || {
                        if let Some(existing) = select_snapshot_row(
                            connection,
                            &owner_public_key,
                            &agent_public_key,
                            &snapshot_id_string,
                        )? {
                            decode_row(
                                &owner_public_key,
                                &agent_public_key,
                                &snapshot_id_string,
                                existing,
                            )?;
                            return Ok(ManagedAgentSnapshotId(snapshot_id_string));
                        }
                        let predecessor_snapshot_id =
                            select_head(connection, &owner_public_key, &agent_public_key)?;
                        let aggregate_hash = aggregate_hash(
                            &owner_public_key,
                            &agent_public_key,
                            &snapshot_id_string,
                            created_at,
                            source_generation,
                            &source_event_id,
                            predecessor_snapshot_id.as_deref(),
                            &persona_hash,
                            team_hash.as_ref(),
                            &runtime_hash,
                        );
                        let provenance_json = serde_json::to_string(&PersistedProvenance {
                            schema_version: 1,
                            source_generation: from_i64(source_generation)?,
                            source_event_id: source_event_id.clone(),
                            predecessor_snapshot_id,
                        })?;
                        let integrity_json = serde_json::to_string(&PersistedIntegrity {
                            schema_version: 1,
                            persona_sha256: hex_hash(persona_hash),
                            team_sha256: team_hash.map(hex_hash),
                            runtime_sha256: hex_hash(runtime_hash),
                            aggregate_sha256: hex_hash(aggregate_hash),
                        })?;
                        connection.exec_bound::<SnapshotInsert>(
                            "INSERT INTO managed_agent_lifecycle_snapshots(\
                            owner_public_key, agent_public_key, snapshot_id, created_at, \
                            provenance_json, persona_json, team_json, runtime_json, integrity_json\
                         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        )?((
                            owner_public_key.clone(),
                            agent_public_key.clone(),
                            snapshot_id_string.clone(),
                            created_at,
                            provenance_json,
                            persona_json,
                            team_json,
                            runtime_json,
                            integrity_json,
                        ))?;
                        connection.exec_bound::<(String, String, String)>(
                            "INSERT INTO managed_agent_snapshot_heads(\
                            owner_public_key, agent_public_key, snapshot_id\
                         ) VALUES (?, ?, ?) \
                         ON CONFLICT(owner_public_key, agent_public_key) DO UPDATE SET \
                            snapshot_id = excluded.snapshot_id",
                        )?((
                            owner_public_key.clone(),
                            agent_public_key.clone(),
                            snapshot_id_string.clone(),
                        ))?;
                        let stored = select_snapshot_row(
                            connection,
                            &owner_public_key,
                            &agent_public_key,
                            &snapshot_id_string,
                        )?
                        .ok_or_else(|| anyhow::anyhow!("created snapshot is missing"))?;
                        decode_row(
                            &owner_public_key,
                            &agent_public_key,
                            &snapshot_id_string,
                            stored,
                        )?;
                        Ok(ManagedAgentSnapshotId(snapshot_id_string))
                    })
                },
            )
            .await
            .map_err(map_database_error)
    }

    pub fn load(
        &self,
        authenticated_owner: &NostrPublicKey,
        agent_public_key: &NostrPublicKey,
        snapshot_id: &ManagedAgentSnapshotId,
    ) -> Result<Option<ManagedAgentSnapshot>, ManagedAgentSnapshotError> {
        let owner_public_key = authenticated_owner.as_str();
        let row = self
            .database
            .select_row_bound::<(&str, &str, &str), SnapshotRow>(
                "SELECT created_at, provenance_json, persona_json, team_json, runtime_json, \
                        integrity_json \
                 FROM managed_agent_lifecycle_snapshots \
                 WHERE owner_public_key = ? AND agent_public_key = ? AND snapshot_id = ?",
            )?((
            owner_public_key,
            agent_public_key.as_str(),
            snapshot_id.as_str(),
        ))
        .map_err(ManagedAgentSnapshotError::Unavailable)?;
        row.map(|row| {
            decode_row(
                owner_public_key,
                agent_public_key.as_str(),
                snapshot_id.as_str(),
                row,
            )
        })
        .transpose()
    }

    pub fn compare(
        &self,
        authenticated_owner: &NostrPublicKey,
        agent_public_key: &NostrPublicKey,
        before: &ManagedAgentSnapshotId,
        after: &ManagedAgentSnapshotId,
    ) -> Result<ManagedAgentSnapshotComparison, ManagedAgentSnapshotError> {
        let before = self
            .load(authenticated_owner, agent_public_key, before)?
            .ok_or(ManagedAgentSnapshotError::NotFound)?;
        let after = self
            .load(authenticated_owner, agent_public_key, after)?
            .ok_or(ManagedAgentSnapshotError::NotFound)?;
        Ok(ManagedAgentSnapshotComparison {
            persona_changed: before.persona_hash != after.persona_hash,
            team_changed: before.team_hash != after.team_hash,
            runtime_changed: before.runtime_hash != after.runtime_hash,
        })
    }

    pub fn restore(
        &self,
        authenticated_owner: &NostrPublicKey,
        current_runtime: &PrivateManagedAgentRecord,
        expected_current_version: &ManagedAgentVersion,
        snapshot_id: &ManagedAgentSnapshotId,
    ) -> Result<ManagedAgentSnapshot, ManagedAgentSnapshotError> {
        authorize_owner(authenticated_owner, current_runtime.owner_public_key())?;
        if current_runtime.version() != expected_current_version {
            return Err(ManagedAgentSnapshotError::StaleRestore);
        }
        self.load(
            authenticated_owner,
            current_runtime.agent_public_key(),
            snapshot_id,
        )?
        .ok_or(ManagedAgentSnapshotError::NotFound)
    }

    pub async fn compact(
        &self,
        authenticated_owner: &NostrPublicKey,
        agent_public_key: &NostrPublicKey,
        expected_head: &ManagedAgentSnapshotId,
        retained_snapshots: usize,
        compacted_at: u64,
    ) -> Result<ManagedAgentSnapshotCompactionOutcome, ManagedAgentSnapshotError> {
        if retained_snapshots == 0 || retained_snapshots > MAX_RETAINED_SNAPSHOTS {
            return Err(ManagedAgentSnapshotError::InvalidCompaction);
        }
        if compacted_at == 0 {
            return Err(ManagedAgentSnapshotError::InvalidTime);
        }
        let owner_public_key = authenticated_owner.as_str().to_string();
        let agent_public_key = agent_public_key.as_str().to_string();
        let expected_head = expected_head.as_str().to_string();
        let compacted_at = to_i64(compacted_at)?;
        self.database
            .write(
                move |connection| -> anyhow::Result<ManagedAgentSnapshotCompactionOutcome> {
                    connection.with_savepoint("compact_managed_agent_lifecycle_snapshots", || {
                        if select_head(connection, &owner_public_key, &agent_public_key)?.as_deref()
                            != Some(&expected_head)
                        {
                            return Ok(ManagedAgentSnapshotCompactionOutcome::Stale);
                        }
                        let rows = connection
                            .select_bound::<(&str, &str, &str), SnapshotWithIdRow>(
                                "SELECT snapshot_id, created_at, provenance_json, persona_json, \
                                    team_json, runtime_json, integrity_json \
                             FROM managed_agent_lifecycle_snapshots \
                             WHERE owner_public_key = ? AND agent_public_key = ? \
                             ORDER BY snapshot_id = ? ASC, created_at ASC, snapshot_id ASC",
                            )?((
                            &owner_public_key,
                            &agent_public_key,
                            &expected_head,
                        ))?;
                        for row in &rows {
                            decode_row(
                                &owner_public_key,
                                &agent_public_key,
                                &row.0,
                                row_without_id(row.clone()),
                            )?;
                        }
                        if rows.len() <= retained_snapshots {
                            return Ok(ManagedAgentSnapshotCompactionOutcome::Unchanged);
                        }
                        let remove_count = rows.len() - retained_snapshots;
                        let removed = &rows[..remove_count];
                        if removed.iter().any(|row| row.0 == expected_head) {
                            anyhow::bail!("snapshot compaction selected the current head");
                        }
                        let removed_ids =
                            removed.iter().map(|row| row.0.clone()).collect::<Vec<_>>();
                        let removed_chain_hash = removed_chain_hash(removed);
                        let compaction_id = compaction_fingerprint(
                            &owner_public_key,
                            &agent_public_key,
                            &expected_head,
                            compacted_at,
                            &removed_chain_hash,
                        );
                        let removed_ids_json = serde_json::to_string(&removed_ids)?;
                        connection
                            .exec_bound::<(String, String, String, i64, String, String, Vec<u8>)>(
                                "INSERT INTO managed_agent_snapshot_compactions(\
                                owner_public_key, agent_public_key, compaction_id, compacted_at, \
                                head_snapshot_id, removed_snapshot_ids_json, removed_chain_sha256\
                             ) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            )?((
                            owner_public_key.clone(),
                            agent_public_key.clone(),
                            compaction_id.as_str().to_string(),
                            compacted_at,
                            expected_head.clone(),
                            removed_ids_json.clone(),
                            removed_chain_hash.to_vec(),
                        ))?;
                        let journal = connection
                            .select_row_bound::<(&str, &str, &str), (String, Vec<u8>)>(
                                "SELECT removed_snapshot_ids_json, removed_chain_sha256 \
                                 FROM managed_agent_snapshot_compactions \
                                 WHERE owner_public_key = ? AND agent_public_key = ? \
                                   AND compaction_id = ?",
                            )?((
                            &owner_public_key,
                            &agent_public_key,
                            compaction_id.as_str(),
                        ))?
                        .ok_or_else(|| {
                            anyhow::anyhow!("snapshot compaction evidence is missing")
                        })?;
                        anyhow::ensure!(
                            journal.0 == removed_ids_json
                                && journal.1.as_slice() == removed_chain_hash,
                            "snapshot compaction evidence failed verification"
                        );
                        let mut delete = connection.exec_bound::<(&str, &str, &str)>(
                            "DELETE FROM managed_agent_lifecycle_snapshots \
                             WHERE owner_public_key = ? AND agent_public_key = ? \
                               AND snapshot_id = ?",
                        )?;
                        for snapshot_id in &removed_ids {
                            delete((&owner_public_key, &agent_public_key, snapshot_id))?;
                            anyhow::ensure!(
                                changed_rows(connection)? == 1,
                                "snapshot compaction source changed before deletion"
                            );
                        }
                        Ok(ManagedAgentSnapshotCompactionOutcome::Compacted {
                            removed: remove_count,
                            compaction_id,
                        })
                    })
                },
            )
            .await
            .map_err(map_database_error)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn corrupt_persona_for_test(
        &self,
        authenticated_owner: &NostrPublicKey,
        agent_public_key: &NostrPublicKey,
        snapshot_id: &ManagedAgentSnapshotId,
    ) -> Result<(), ManagedAgentSnapshotError> {
        let owner_public_key = authenticated_owner.as_str().to_string();
        let agent_public_key = agent_public_key.as_str().to_string();
        let snapshot_id = snapshot_id.as_str().to_string();
        self.database
            .write(move |connection| -> anyhow::Result<()> {
                connection.exec_bound::<(String, String, String)>(
                    "UPDATE managed_agent_lifecycle_snapshots SET persona_json = '{\"partial\":' \
                     WHERE owner_public_key = ? AND agent_public_key = ? AND snapshot_id = ?",
                )?((owner_public_key, agent_public_key, snapshot_id))?;
                Ok(())
            })
            .await
            .map_err(map_database_error)
    }
}

pub struct ManagedAgentSnapshotDocuments {
    persona_json: String,
    team_json: Option<String>,
}

impl ManagedAgentSnapshotDocuments {
    pub fn new(persona: Value, team: Option<Value>) -> Result<Self, ManagedAgentSnapshotError> {
        let persona_json = canonical_document(persona)?;
        let team_json = team.map(canonical_document).transpose()?;
        Ok(Self {
            persona_json,
            team_json,
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ManagedAgentSnapshotId(String);

impl ManagedAgentSnapshotId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, PartialEq)]
pub struct ManagedAgentSnapshot {
    snapshot_id: ManagedAgentSnapshotId,
    created_at: u64,
    predecessor_snapshot_id: Option<ManagedAgentSnapshotId>,
    source_version: ManagedAgentVersion,
    persona: Value,
    team: Option<Value>,
    runtime: PrivateManagedAgentRecord,
    persona_hash: [u8; 32],
    team_hash: Option<[u8; 32]>,
    runtime_hash: [u8; 32],
}

impl ManagedAgentSnapshot {
    pub fn snapshot_id(&self) -> &ManagedAgentSnapshotId {
        &self.snapshot_id
    }

    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    pub fn predecessor_snapshot_id(&self) -> Option<&ManagedAgentSnapshotId> {
        self.predecessor_snapshot_id.as_ref()
    }

    pub fn source_version(&self) -> &ManagedAgentVersion {
        &self.source_version
    }

    pub fn persona(&self) -> &Value {
        &self.persona
    }

    pub fn team(&self) -> Option<&Value> {
        self.team.as_ref()
    }

    pub fn runtime(&self) -> &PrivateManagedAgentRecord {
        &self.runtime
    }
}

impl fmt::Debug for ManagedAgentSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedAgentSnapshot")
            .field("snapshot_id", &self.snapshot_id)
            .field("created_at", &self.created_at)
            .field("predecessor_snapshot_id", &self.predecessor_snapshot_id)
            .field("source_version", &self.source_version)
            .field("persona", &"<redacted>")
            .field("team", &self.team.as_ref().map(|_| "<redacted>"))
            .field("runtime", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagedAgentSnapshotComparison {
    pub persona_changed: bool,
    pub team_changed: bool,
    pub runtime_changed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManagedAgentSnapshotCompactionOutcome {
    Compacted {
        removed: usize,
        compaction_id: ManagedAgentSnapshotId,
    },
    Unchanged,
    Stale,
}

#[derive(Debug, Error)]
pub enum ManagedAgentSnapshotError {
    #[error("managed-agent snapshot repository is unavailable")]
    Unavailable(#[source] anyhow::Error),
    #[error("managed-agent snapshot owner does not match")]
    OwnerMismatch,
    #[error("managed-agent snapshot document is invalid")]
    InvalidDocument,
    #[error("managed-agent snapshot time is invalid")]
    InvalidTime,
    #[error("managed-agent snapshot was not found")]
    NotFound,
    #[error("managed-agent snapshot is corrupt")]
    CorruptSnapshot,
    #[error("managed-agent snapshot restore is stale")]
    StaleRestore,
    #[error("managed-agent snapshot compaction request is invalid")]
    InvalidCompaction,
}

impl From<anyhow::Error> for ManagedAgentSnapshotError {
    fn from(error: anyhow::Error) -> Self {
        Self::Unavailable(error)
    }
}

type SnapshotInsert = (
    String,
    String,
    String,
    i64,
    String,
    String,
    Option<String>,
    String,
    String,
);

type SnapshotRow = (i64, String, String, Option<String>, String, String);

type SnapshotWithIdRow = (String, i64, String, String, Option<String>, String, String);

fn row_without_id(row: SnapshotWithIdRow) -> SnapshotRow {
    (row.1, row.2, row.3, row.4, row.5, row.6)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedProvenance {
    schema_version: u32,
    source_generation: u64,
    source_event_id: String,
    predecessor_snapshot_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedIntegrity {
    schema_version: u32,
    persona_sha256: String,
    team_sha256: Option<String>,
    runtime_sha256: String,
    aggregate_sha256: String,
}

fn select_snapshot_row(
    connection: &db::sqlez::connection::Connection,
    owner_public_key: &str,
    agent_public_key: &str,
    snapshot_id: &str,
) -> anyhow::Result<Option<SnapshotRow>> {
    connection.select_row_bound::<(&str, &str, &str), SnapshotRow>(
        "SELECT created_at, provenance_json, persona_json, team_json, runtime_json, \
                integrity_json \
         FROM managed_agent_lifecycle_snapshots \
         WHERE owner_public_key = ? AND agent_public_key = ? AND snapshot_id = ?",
    )?((owner_public_key, agent_public_key, snapshot_id))
}

fn select_head(
    connection: &db::sqlez::connection::Connection,
    owner_public_key: &str,
    agent_public_key: &str,
) -> anyhow::Result<Option<String>> {
    connection.select_row_bound::<(&str, &str), String>(
        "SELECT snapshot_id FROM managed_agent_snapshot_heads \
         WHERE owner_public_key = ? AND agent_public_key = ?",
    )?((owner_public_key, agent_public_key))
}

fn decode_row(
    owner_public_key: &str,
    agent_public_key: &str,
    snapshot_id: &str,
    row: SnapshotRow,
) -> Result<ManagedAgentSnapshot, ManagedAgentSnapshotError> {
    let (created_at, provenance_json, persona_json, team_json, runtime_json, integrity_json) = row;
    let created_at = from_i64(created_at)?;
    let provenance: PersistedProvenance = serde_json::from_str(&provenance_json)
        .map_err(|_| ManagedAgentSnapshotError::CorruptSnapshot)?;
    let integrity: PersistedIntegrity = serde_json::from_str(&integrity_json)
        .map_err(|_| ManagedAgentSnapshotError::CorruptSnapshot)?;
    if provenance.schema_version != 1 || integrity.schema_version != 1 {
        return Err(ManagedAgentSnapshotError::CorruptSnapshot);
    }
    let source_generation = provenance.source_generation;
    let snapshot_id = parse_id(snapshot_id)?;
    let predecessor_snapshot_id = provenance
        .predecessor_snapshot_id
        .map(|snapshot_id| parse_id(&snapshot_id))
        .transpose()?;
    let source_event_id = NostrEventId::parse(provenance.source_event_id)
        .map_err(|_| ManagedAgentSnapshotError::CorruptSnapshot)?;
    let source_version = ManagedAgentVersion::new(source_generation, source_event_id.clone())
        .map_err(|_| ManagedAgentSnapshotError::CorruptSnapshot)?;
    let persona_hash = parse_hex_hash(&integrity.persona_sha256)?;
    let team_hash = integrity
        .team_sha256
        .as_deref()
        .map(parse_hex_hash)
        .transpose()?;
    let runtime_hash = parse_hex_hash(&integrity.runtime_sha256)?;
    let aggregate_hash_stored = parse_hex_hash(&integrity.aggregate_sha256)?;
    if hash_bytes(persona_json.as_bytes()) != persona_hash
        || team_json.as_ref().map(|team| hash_bytes(team.as_bytes())) != team_hash
        || hash_bytes(runtime_json.as_bytes()) != runtime_hash
    {
        return Err(ManagedAgentSnapshotError::CorruptSnapshot);
    }
    let aggregate_hash_computed = aggregate_hash(
        owner_public_key,
        agent_public_key,
        snapshot_id.as_str(),
        to_i64(created_at)?,
        to_i64(source_generation)?,
        source_event_id.as_str(),
        predecessor_snapshot_id
            .as_ref()
            .map(ManagedAgentSnapshotId::as_str),
        &persona_hash,
        team_hash.as_ref(),
        &runtime_hash,
    );
    if aggregate_hash_computed != aggregate_hash_stored {
        return Err(ManagedAgentSnapshotError::CorruptSnapshot);
    }
    let persona = decode_document(&persona_json)?;
    let team = team_json.as_deref().map(decode_document).transpose()?;
    let runtime =
        decode_snapshot(&runtime_json).map_err(|_| ManagedAgentSnapshotError::CorruptSnapshot)?;
    if runtime.owner_public_key().as_str() != owner_public_key
        || runtime.agent_public_key().as_str() != agent_public_key
        || runtime.version() != &source_version
    {
        return Err(ManagedAgentSnapshotError::CorruptSnapshot);
    }
    Ok(ManagedAgentSnapshot {
        snapshot_id,
        created_at,
        predecessor_snapshot_id,
        source_version,
        persona,
        team,
        runtime,
        persona_hash,
        team_hash,
        runtime_hash,
    })
}

fn canonical_document(value: Value) -> Result<String, ManagedAgentSnapshotError> {
    if !value.is_object() {
        return Err(ManagedAgentSnapshotError::InvalidDocument);
    }
    let canonical = canonicalize(value);
    let encoded = serde_json::to_string(&canonical)
        .map_err(|_| ManagedAgentSnapshotError::InvalidDocument)?;
    validate_document_bytes(encoded.as_bytes())?;
    Ok(encoded)
}

fn decode_document(encoded: &str) -> Result<Value, ManagedAgentSnapshotError> {
    validate_document_bytes(encoded.as_bytes())?;
    let value: Value =
        serde_json::from_str(encoded).map_err(|_| ManagedAgentSnapshotError::CorruptSnapshot)?;
    let canonical = canonical_document(value.clone())
        .map_err(|_| ManagedAgentSnapshotError::CorruptSnapshot)?;
    if !value.is_object() || canonical != encoded {
        return Err(ManagedAgentSnapshotError::CorruptSnapshot);
    }
    Ok(value)
}

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, canonicalize(value)))
                    .collect::<Map<_, _>>(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize).collect()),
        other => other,
    }
}

fn validate_document_bytes(bytes: &[u8]) -> Result<(), ManagedAgentSnapshotError> {
    if bytes.is_empty() || bytes.len() > MAX_DOCUMENT_BYTES {
        return Err(ManagedAgentSnapshotError::InvalidDocument);
    }
    Ok(())
}

fn authorize_owner(
    authenticated_owner: &NostrPublicKey,
    record_owner: &NostrPublicKey,
) -> Result<(), ManagedAgentSnapshotError> {
    if authenticated_owner != record_owner {
        return Err(ManagedAgentSnapshotError::OwnerMismatch);
    }
    Ok(())
}

fn creation_fingerprint(
    owner_public_key: &str,
    agent_public_key: &str,
    source_generation: i64,
    source_event_id: &str,
    created_at: i64,
    persona_hash: &[u8; 32],
    team_hash: Option<&[u8; 32]>,
    runtime_hash: &[u8; 32],
) -> ManagedAgentSnapshotId {
    let mut hasher = Sha256::new();
    update_hash(&mut hasher, b"managed-agent-snapshot-id-v1");
    update_hash(&mut hasher, owner_public_key.as_bytes());
    update_hash(&mut hasher, agent_public_key.as_bytes());
    update_hash(&mut hasher, &source_generation.to_be_bytes());
    update_hash(&mut hasher, source_event_id.as_bytes());
    update_hash(&mut hasher, &created_at.to_be_bytes());
    update_hash(&mut hasher, persona_hash);
    update_optional_hash(&mut hasher, team_hash);
    update_hash(&mut hasher, runtime_hash);
    ManagedAgentSnapshotId(hex_hash(hasher.finalize().into()))
}

#[allow(
    clippy::too_many_arguments,
    reason = "the integrity hash must bind every provenance field in one explicit order"
)]
fn aggregate_hash(
    owner_public_key: &str,
    agent_public_key: &str,
    snapshot_id: &str,
    created_at: i64,
    source_generation: i64,
    source_event_id: &str,
    predecessor_snapshot_id: Option<&str>,
    persona_hash: &[u8; 32],
    team_hash: Option<&[u8; 32]>,
    runtime_hash: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    update_hash(&mut hasher, b"managed-agent-snapshot-aggregate-v1");
    update_hash(&mut hasher, owner_public_key.as_bytes());
    update_hash(&mut hasher, agent_public_key.as_bytes());
    update_hash(&mut hasher, snapshot_id.as_bytes());
    update_hash(&mut hasher, &created_at.to_be_bytes());
    update_hash(&mut hasher, &source_generation.to_be_bytes());
    update_hash(&mut hasher, source_event_id.as_bytes());
    update_optional_bytes(&mut hasher, predecessor_snapshot_id.map(str::as_bytes));
    update_hash(&mut hasher, persona_hash);
    update_optional_hash(&mut hasher, team_hash);
    update_hash(&mut hasher, runtime_hash);
    hasher.finalize().into()
}

fn removed_chain_hash(rows: &[SnapshotWithIdRow]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    update_hash(&mut hasher, b"managed-agent-snapshot-compaction-chain-v1");
    for row in rows {
        update_hash(&mut hasher, row.0.as_bytes());
        update_hash(&mut hasher, row.6.as_bytes());
    }
    hasher.finalize().into()
}

fn compaction_fingerprint(
    owner_public_key: &str,
    agent_public_key: &str,
    head_snapshot_id: &str,
    compacted_at: i64,
    removed_chain_hash: &[u8; 32],
) -> ManagedAgentSnapshotId {
    let mut hasher = Sha256::new();
    update_hash(&mut hasher, b"managed-agent-snapshot-compaction-id-v1");
    update_hash(&mut hasher, owner_public_key.as_bytes());
    update_hash(&mut hasher, agent_public_key.as_bytes());
    update_hash(&mut hasher, head_snapshot_id.as_bytes());
    update_hash(&mut hasher, &compacted_at.to_be_bytes());
    update_hash(&mut hasher, removed_chain_hash);
    ManagedAgentSnapshotId(hex_hash(hasher.finalize().into()))
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn update_hash(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn update_optional_hash(hasher: &mut Sha256, hash: Option<&[u8; 32]>) {
    update_optional_bytes(hasher, hash.map(|hash| hash.as_slice()));
}

fn update_optional_bytes(hasher: &mut Sha256, bytes: Option<&[u8]>) {
    match bytes {
        Some(bytes) => {
            hasher.update([1]);
            update_hash(hasher, bytes);
        }
        None => hasher.update([0]),
    }
}

fn parse_id(value: &str) -> Result<ManagedAgentSnapshotId, ManagedAgentSnapshotError> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err(ManagedAgentSnapshotError::CorruptSnapshot);
    }
    Ok(ManagedAgentSnapshotId(value.to_string()))
}

fn parse_hex_hash(value: &str) -> Result<[u8; 32], ManagedAgentSnapshotError> {
    if value.len() != 64 {
        return Err(ManagedAgentSnapshotError::CorruptSnapshot);
    }
    let mut hash = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).ok_or(ManagedAgentSnapshotError::CorruptSnapshot)?;
        let low = hex_nibble(pair[1]).ok_or(ManagedAgentSnapshotError::CorruptSnapshot)?;
        hash[index] = (high << 4) | low;
    }
    Ok(hash)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn hex_hash(hash: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in hash {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn to_i64(value: u64) -> Result<i64, ManagedAgentSnapshotError> {
    i64::try_from(value).map_err(|_| ManagedAgentSnapshotError::InvalidTime)
}

fn from_i64(value: i64) -> Result<u64, ManagedAgentSnapshotError> {
    u64::try_from(value).map_err(|_| ManagedAgentSnapshotError::CorruptSnapshot)
}

fn changed_rows(connection: &db::sqlez::connection::Connection) -> anyhow::Result<i64> {
    connection.select_row::<i64>("SELECT changes()")?()?
        .ok_or_else(|| anyhow::anyhow!("SQLite did not report its changed row count"))
}

fn map_database_error(error: anyhow::Error) -> ManagedAgentSnapshotError {
    if error
        .chain()
        .any(|source| source.downcast_ref::<ManagedAgentSnapshotError>().is_some())
    {
        ManagedAgentSnapshotError::CorruptSnapshot
    } else {
        ManagedAgentSnapshotError::Unavailable(error)
    }
}
