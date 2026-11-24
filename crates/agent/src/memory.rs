use db::sqlez_macros::sql;
use nostr_compat::agent_memory::{EncryptedEngram, EngramCoordinate};
use nostr_compat::dm::Nip44Ciphertext;
use nostr_compat::{EventId, PublicKey};
use sha2::{Digest as _, Sha256};
use sqlez::domain::Domain;
use std::fmt;
use thiserror::Error;

const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

pub struct AgentMemoryDatabase(db::sqlez::thread_safe_connection::ThreadSafeConnection);

impl Domain for AgentMemoryDatabase {
    const NAME: &str = stringify!(AgentMemoryDatabase);

    const MIGRATIONS: &[&str] = &[sql!(
        CREATE TABLE encrypted_agent_memories (
            owner_public_key TEXT NOT NULL,
            agent_public_key TEXT NOT NULL,
            d_tag TEXT NOT NULL,
            event_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            ciphertext TEXT NOT NULL,
            ciphertext_sha256 BLOB NOT NULL,
            retention_generation INTEGER NOT NULL,
            expires_at INTEGER,
            PRIMARY KEY (owner_public_key, agent_public_key, d_tag),
            CHECK (length(owner_public_key) = 64),
            CHECK (length(agent_public_key) = 64),
            CHECK (length(d_tag) = 64),
            CHECK (length(event_id) = 64),
            CHECK (created_at >= 0),
            CHECK (length(ciphertext) >= 132),
            CHECK (length(ciphertext) <= 87472),
            CHECK (length(ciphertext_sha256) = 32),
            CHECK (retention_generation > 0),
            CHECK (retention_generation <= 9007199254740991),
            CHECK (expires_at IS NULL OR expires_at > 0)
        ) STRICT;

        CREATE INDEX encrypted_agent_memories_expiry
            ON encrypted_agent_memories(expires_at)
            WHERE expires_at IS NOT NULL;
    )];
}

db::static_connection!(AgentMemoryDatabase, []);

impl AgentMemoryDatabase {
    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self(database.0.clone())
    }
}

#[derive(Clone)]
pub struct AgentMemoryRepository {
    database: AgentMemoryDatabase,
}

impl AgentMemoryRepository {
    pub fn global(cx: &gpui::App) -> Self {
        Self {
            database: AgentMemoryDatabase::global(cx),
        }
    }

    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self {
            database: AgentMemoryDatabase::from_app_database(database),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_database(name: &'static str) -> Self {
        Self {
            database: AgentMemoryDatabase::open_test_db(name).await,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_file_database(database_directory: &std::path::Path) -> Self {
        Self {
            database: AgentMemoryDatabase(
                db::open_db::<AgentMemoryDatabase>(database_directory, db::GlobalDbScope).await,
            ),
        }
    }

    pub async fn store(
        &self,
        authenticated_owner: PublicKey,
        record: &StoredEncryptedMemory,
    ) -> Result<MemoryWriteOutcome, AgentMemoryRepositoryError> {
        authorize_owner(authenticated_owner, record.coordinate())?;
        validate_record(record)?;
        let row = PersistedMemoryRow::from_record(record)?;
        self.database
            .write(move |connection| insert_or_replace(connection, &row))
            .await
            .map_err(AgentMemoryRepositoryError::Unavailable)
    }

    pub fn load_for_owner(
        &self,
        authenticated_owner: PublicKey,
        coordinate: &EngramCoordinate,
        now: u64,
    ) -> Result<Option<StoredEncryptedMemory>, AgentMemoryRepositoryError> {
        authorize_owner(authenticated_owner, coordinate)?;
        let now = to_i64(now)?;
        let row = self
            .database
            .select_row_bound::<(&str, &str, &str), PersistedMemoryTuple>(
                "SELECT event_id, created_at, ciphertext, ciphertext_sha256, \
                        retention_generation, expires_at \
                 FROM encrypted_agent_memories \
                 WHERE owner_public_key = ? AND agent_public_key = ? AND d_tag = ?",
            )?((
            &coordinate.owner().to_hex(),
            &coordinate.agent().to_hex(),
            coordinate.d_tag(),
        ))
        .map_err(AgentMemoryRepositoryError::Unavailable)?;
        let Some(row) = row else {
            return Ok(None);
        };
        if row.5.is_some_and(|expires_at| expires_at <= now) {
            return Ok(None);
        }
        decode_record(coordinate, row).map(Some)
    }

    pub async fn expire(
        &self,
        authenticated_owner: PublicKey,
        coordinate: &EngramCoordinate,
        expected_retention_generation: u64,
        expires_at: u64,
    ) -> Result<MemoryRetentionOutcome, AgentMemoryRepositoryError> {
        authorize_owner(authenticated_owner, coordinate)?;
        let expected_generation = retention_generation_to_i64(expected_retention_generation)?;
        let next_generation = expected_retention_generation
            .checked_add(1)
            .ok_or(AgentMemoryRepositoryError::InvalidRetention)
            .and_then(retention_generation_to_i64)?;
        let expires_at = positive_i64(expires_at)?;
        let owner_public_key = coordinate.owner().to_hex();
        let agent_public_key = coordinate.agent().to_hex();
        let d_tag = coordinate.d_tag().to_owned();
        self.database
            .write(
                move |connection| -> anyhow::Result<MemoryRetentionOutcome> {
                    connection.exec_bound::<(i64, i64, i64, String, String, String, i64)>(
                        "UPDATE encrypted_agent_memories \
                     SET retention_generation = ?, \
                         expires_at = CASE \
                             WHEN expires_at IS NULL OR expires_at > ? THEN ? \
                             ELSE expires_at \
                         END \
                     WHERE owner_public_key = ? AND agent_public_key = ? AND d_tag = ? \
                       AND retention_generation = ?",
                    )?((
                        next_generation,
                        expires_at,
                        expires_at,
                        owner_public_key,
                        agent_public_key,
                        d_tag,
                        expected_generation,
                    ))?;
                    Ok(match changed_rows(connection)? {
                        1 => MemoryRetentionOutcome::Applied,
                        0 => MemoryRetentionOutcome::Stale,
                        _ => anyhow::bail!("memory expiry affected an invalid row count"),
                    })
                },
            )
            .await
            .map_err(AgentMemoryRepositoryError::Unavailable)
    }

    pub async fn rotate_owner(
        &self,
        authenticated_owner: PublicKey,
        previous_coordinate: &EngramCoordinate,
        expected_retention_generation: u64,
        replacement: &StoredEncryptedMemory,
        rotated_at: u64,
    ) -> Result<MemoryRotationOutcome, AgentMemoryRepositoryError> {
        authorize_owner(authenticated_owner, previous_coordinate)?;
        validate_record(replacement)?;
        if previous_coordinate.agent() != replacement.coordinate().agent()
            || previous_coordinate.owner() == replacement.coordinate().owner()
        {
            return Err(AgentMemoryRepositoryError::InvalidRotation);
        }
        if replacement
            .retention()
            .expires_at()
            .is_some_and(|expires_at| expires_at <= rotated_at)
        {
            return Err(AgentMemoryRepositoryError::InvalidRotation);
        }
        let expected_generation = retention_generation_to_i64(expected_retention_generation)?;
        let next_generation = expected_retention_generation
            .checked_add(1)
            .ok_or(AgentMemoryRepositoryError::InvalidRetention)
            .and_then(retention_generation_to_i64)?;
        let rotated_at = positive_i64(rotated_at)?;
        let previous_owner = previous_coordinate.owner().to_hex();
        let previous_agent = previous_coordinate.agent().to_hex();
        let previous_d_tag = previous_coordinate.d_tag().to_owned();
        let replacement = PersistedMemoryRow::from_record(replacement)?;

        self.database
            .write(move |connection| -> anyhow::Result<MemoryRotationOutcome> {
                connection.with_savepoint("encrypted_memory_owner_rotation", || {
                    let previous_state = connection
                        .select_row_bound::<(&str, &str, &str), (i64, Option<i64>)>(
                            "SELECT retention_generation, expires_at \
                             FROM encrypted_agent_memories \
                             WHERE owner_public_key = ? AND agent_public_key = ? AND d_tag = ?",
                        )?((&previous_owner, &previous_agent, &previous_d_tag))?;
                    let replacement_state = connection
                        .select_row_bound::<(&str, &str, &str), PersistedMemoryTuple>(
                            "SELECT event_id, created_at, ciphertext, ciphertext_sha256, \
                                    retention_generation, expires_at \
                             FROM encrypted_agent_memories \
                             WHERE owner_public_key = ? AND agent_public_key = ? AND d_tag = ?",
                        )?((
                        &replacement.owner_public_key,
                        &replacement.agent_public_key,
                        &replacement.d_tag,
                    ))?;

                    let replacement_is_exact = replacement_state
                        .as_ref()
                        .is_some_and(|state| replacement.matches_tuple(state));
                    if previous_state == Some((next_generation, Some(rotated_at)))
                        && replacement_is_exact
                    {
                        return Ok(MemoryRotationOutcome::AlreadyApplied);
                    }
                    if previous_state != Some((expected_generation, None))
                        && !matches!(previous_state, Some((generation, Some(expires_at))) if generation == expected_generation && expires_at > rotated_at)
                    {
                        return Ok(MemoryRotationOutcome::Stale);
                    }
                    if replacement_state.is_some() && !replacement_is_exact {
                        return Ok(MemoryRotationOutcome::Conflict);
                    }

                    connection.exec_bound::<(i64, i64, &str, &str, &str, i64)>(
                        "UPDATE encrypted_agent_memories \
                         SET retention_generation = ?, expires_at = ? \
                         WHERE owner_public_key = ? AND agent_public_key = ? AND d_tag = ? \
                           AND retention_generation = ?",
                    )?((
                        next_generation,
                        rotated_at,
                        &previous_owner,
                        &previous_agent,
                        &previous_d_tag,
                        expected_generation,
                    ))?;
                    if changed_rows(connection)? != 1 {
                        anyhow::bail!("memory owner rotation lost its retention claim");
                    }
                    if replacement_state.is_none() {
                        insert_exact(connection, &replacement)?;
                    }
                    Ok(MemoryRotationOutcome::Applied)
                })
            })
            .await
            .map_err(AgentMemoryRepositoryError::Unavailable)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn corrupt_ciphertext_for_test(
        &self,
        coordinate: &EngramCoordinate,
    ) -> Result<(), AgentMemoryRepositoryError> {
        let owner_public_key = coordinate.owner().to_hex();
        let agent_public_key = coordinate.agent().to_hex();
        let d_tag = coordinate.d_tag().to_owned();
        self.database
            .write(move |connection| -> anyhow::Result<()> {
                connection.exec_bound::<(String, String, String)>(
                    "UPDATE encrypted_agent_memories \
                     SET ciphertext = CASE substr(ciphertext, 1, 1) \
                         WHEN 'A' THEN 'B' ELSE 'A' END || substr(ciphertext, 2) \
                     WHERE owner_public_key = ? AND agent_public_key = ? AND d_tag = ?",
                )?((owner_public_key, agent_public_key, d_tag))?;
                Ok(())
            })
            .await
            .map_err(AgentMemoryRepositoryError::Unavailable)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryRetention {
    generation: u64,
    expires_at: Option<u64>,
}

impl MemoryRetention {
    pub fn new(
        generation: u64,
        expires_at: Option<u64>,
    ) -> Result<Self, AgentMemoryRepositoryError> {
        retention_generation_to_i64(generation)?;
        if let Some(expires_at) = expires_at {
            positive_i64(expires_at)?;
        }
        Ok(Self {
            generation,
            expires_at,
        })
    }

    pub fn generation(self) -> u64 {
        self.generation
    }

    pub fn expires_at(self) -> Option<u64> {
        self.expires_at
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct StoredEncryptedMemory {
    coordinate: EngramCoordinate,
    event_id: EventId,
    created_at: u64,
    ciphertext: Nip44Ciphertext,
    ciphertext_sha256: [u8; 32],
    retention: MemoryRetention,
}

impl StoredEncryptedMemory {
    pub fn new(
        encrypted: &EncryptedEngram,
        event_id: EventId,
        created_at: u64,
        retention: MemoryRetention,
    ) -> Result<Self, AgentMemoryRepositoryError> {
        to_i64(created_at)?;
        let ciphertext = encrypted.ciphertext().clone();
        Ok(Self {
            coordinate: encrypted.coordinate().clone(),
            event_id,
            created_at,
            ciphertext_sha256: ciphertext_digest(&ciphertext),
            ciphertext,
            retention,
        })
    }

    pub fn coordinate(&self) -> &EngramCoordinate {
        &self.coordinate
    }

    pub fn event_id(&self) -> EventId {
        self.event_id
    }

    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    pub fn ciphertext(&self) -> &Nip44Ciphertext {
        &self.ciphertext
    }

    pub fn ciphertext_sha256(&self) -> &[u8; 32] {
        &self.ciphertext_sha256
    }

    pub fn retention(&self) -> MemoryRetention {
        self.retention
    }
}

impl fmt::Debug for StoredEncryptedMemory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoredEncryptedMemory")
            .field("coordinate", &self.coordinate)
            .field("event_id", &self.event_id)
            .field("created_at", &self.created_at)
            .field("ciphertext", &"<redacted>")
            .field("ciphertext_sha256", &self.ciphertext_sha256)
            .field("retention", &self.retention)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryWriteOutcome {
    Stored,
    AlreadyCurrent,
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryRetentionOutcome {
    Applied,
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryRotationOutcome {
    Applied,
    AlreadyApplied,
    Stale,
    Conflict,
}

#[derive(Debug, Error)]
pub enum AgentMemoryRepositoryError {
    #[error("agent-memory repository is unavailable")]
    Unavailable(#[source] anyhow::Error),
    #[error("agent-memory owner does not match the authenticated owner")]
    OwnerMismatch,
    #[error("agent-memory record is corrupt")]
    CorruptRecord,
    #[error("agent-memory ciphertext failed integrity verification")]
    CorruptCiphertext,
    #[error("agent-memory retention state is invalid")]
    InvalidRetention,
    #[error("agent-memory owner rotation is invalid")]
    InvalidRotation,
}

impl From<anyhow::Error> for AgentMemoryRepositoryError {
    fn from(error: anyhow::Error) -> Self {
        Self::Unavailable(error)
    }
}

type PersistedMemoryTuple = (String, i64, String, Vec<u8>, i64, Option<i64>);

struct PersistedMemoryRow {
    owner_public_key: String,
    agent_public_key: String,
    d_tag: String,
    event_id: String,
    created_at: i64,
    ciphertext: String,
    ciphertext_sha256: Vec<u8>,
    retention_generation: i64,
    expires_at: Option<i64>,
}

impl PersistedMemoryRow {
    fn from_record(record: &StoredEncryptedMemory) -> Result<Self, AgentMemoryRepositoryError> {
        Ok(Self {
            owner_public_key: record.coordinate.owner().to_hex(),
            agent_public_key: record.coordinate.agent().to_hex(),
            d_tag: record.coordinate.d_tag().to_owned(),
            event_id: record.event_id.to_hex(),
            created_at: to_i64(record.created_at)?,
            ciphertext: record.ciphertext.wire_value().to_owned(),
            ciphertext_sha256: record.ciphertext_sha256.to_vec(),
            retention_generation: retention_generation_to_i64(record.retention.generation)?,
            expires_at: record.retention.expires_at.map(positive_i64).transpose()?,
        })
    }

    fn matches_tuple(&self, row: &PersistedMemoryTuple) -> bool {
        self.event_id == row.0
            && self.created_at == row.1
            && self.ciphertext == row.2
            && self.ciphertext_sha256 == row.3
            && self.retention_generation == row.4
            && self.expires_at == row.5
    }
}

fn insert_or_replace(
    connection: &db::sqlez::connection::Connection,
    row: &PersistedMemoryRow,
) -> anyhow::Result<MemoryWriteOutcome> {
    connection.exec_bound::<(&str, &str, &str, &str, i64, &str, &[u8], i64, Option<i64>)>(
        "INSERT INTO encrypted_agent_memories(\
            owner_public_key, agent_public_key, d_tag, event_id, created_at, ciphertext, \
            ciphertext_sha256, retention_generation, expires_at\
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) \
         ON CONFLICT(owner_public_key, agent_public_key, d_tag) DO UPDATE SET \
            event_id = excluded.event_id, created_at = excluded.created_at, \
            ciphertext = excluded.ciphertext, ciphertext_sha256 = excluded.ciphertext_sha256, \
            retention_generation = encrypted_agent_memories.retention_generation, \
            expires_at = encrypted_agent_memories.expires_at \
         WHERE excluded.created_at > encrypted_agent_memories.created_at \
            OR (excluded.created_at = encrypted_agent_memories.created_at \
                AND excluded.event_id < encrypted_agent_memories.event_id)",
    )?((
        &row.owner_public_key,
        &row.agent_public_key,
        &row.d_tag,
        &row.event_id,
        row.created_at,
        &row.ciphertext,
        &row.ciphertext_sha256,
        row.retention_generation,
        row.expires_at,
    ))?;
    if changed_rows(connection)? == 1 {
        return Ok(MemoryWriteOutcome::Stored);
    }
    let current_event = connection.select_row_bound::<(&str, &str, &str), String>(
        "SELECT event_id FROM encrypted_agent_memories \
             WHERE owner_public_key = ? AND agent_public_key = ? AND d_tag = ?",
    )?((&row.owner_public_key, &row.agent_public_key, &row.d_tag))?;
    Ok(if current_event.as_deref() == Some(&row.event_id) {
        MemoryWriteOutcome::AlreadyCurrent
    } else {
        MemoryWriteOutcome::Stale
    })
}

fn insert_exact(
    connection: &db::sqlez::connection::Connection,
    row: &PersistedMemoryRow,
) -> anyhow::Result<()> {
    connection.exec_bound::<(&str, &str, &str, &str, i64, &str, &[u8], i64, Option<i64>)>(
        "INSERT INTO encrypted_agent_memories(\
            owner_public_key, agent_public_key, d_tag, event_id, created_at, ciphertext, \
            ciphertext_sha256, retention_generation, expires_at\
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )?((
        &row.owner_public_key,
        &row.agent_public_key,
        &row.d_tag,
        &row.event_id,
        row.created_at,
        &row.ciphertext,
        &row.ciphertext_sha256,
        row.retention_generation,
        row.expires_at,
    ))?;
    Ok(())
}

fn decode_record(
    coordinate: &EngramCoordinate,
    row: PersistedMemoryTuple,
) -> Result<StoredEncryptedMemory, AgentMemoryRepositoryError> {
    let event_id =
        EventId::from_hex(&row.0).map_err(|_| AgentMemoryRepositoryError::CorruptRecord)?;
    let created_at = from_i64(row.1)?;
    let ciphertext =
        Nip44Ciphertext::parse(row.2).map_err(|_| AgentMemoryRepositoryError::CorruptCiphertext)?;
    let stored_digest: [u8; 32] = row
        .3
        .try_into()
        .map_err(|_| AgentMemoryRepositoryError::CorruptCiphertext)?;
    let ciphertext_sha256 = ciphertext_digest(&ciphertext);
    if stored_digest != ciphertext_sha256 {
        return Err(AgentMemoryRepositoryError::CorruptCiphertext);
    }
    let retention_generation = retention_generation_from_i64(row.4)?;
    let expires_at = row.5.map(from_positive_i64).transpose()?;
    Ok(StoredEncryptedMemory {
        coordinate: coordinate.clone(),
        event_id,
        created_at,
        ciphertext,
        ciphertext_sha256,
        retention: MemoryRetention {
            generation: retention_generation,
            expires_at,
        },
    })
}

fn validate_record(record: &StoredEncryptedMemory) -> Result<(), AgentMemoryRepositoryError> {
    to_i64(record.created_at)?;
    retention_generation_to_i64(record.retention.generation)?;
    if let Some(expires_at) = record.retention.expires_at {
        positive_i64(expires_at)?;
    }
    if record.ciphertext_sha256 != ciphertext_digest(&record.ciphertext) {
        return Err(AgentMemoryRepositoryError::CorruptCiphertext);
    }
    Ok(())
}

fn authorize_owner(
    authenticated_owner: PublicKey,
    coordinate: &EngramCoordinate,
) -> Result<(), AgentMemoryRepositoryError> {
    if authenticated_owner != coordinate.owner() {
        return Err(AgentMemoryRepositoryError::OwnerMismatch);
    }
    Ok(())
}

fn ciphertext_digest(ciphertext: &Nip44Ciphertext) -> [u8; 32] {
    Sha256::digest(ciphertext.wire_value().as_bytes()).into()
}

fn changed_rows(connection: &db::sqlez::connection::Connection) -> anyhow::Result<i64> {
    connection.select_row_bound::<(), i64>("SELECT changes()")?(())?
        .ok_or_else(|| anyhow::anyhow!("SQLite did not report a changed-row count"))
}

fn retention_generation_to_i64(value: u64) -> Result<i64, AgentMemoryRepositoryError> {
    if value == 0 || value > MAX_SAFE_INTEGER {
        return Err(AgentMemoryRepositoryError::InvalidRetention);
    }
    i64::try_from(value).map_err(|_| AgentMemoryRepositoryError::InvalidRetention)
}

fn retention_generation_from_i64(value: i64) -> Result<u64, AgentMemoryRepositoryError> {
    let value = u64::try_from(value).map_err(|_| AgentMemoryRepositoryError::CorruptRecord)?;
    if value == 0 || value > MAX_SAFE_INTEGER {
        return Err(AgentMemoryRepositoryError::CorruptRecord);
    }
    Ok(value)
}

fn to_i64(value: u64) -> Result<i64, AgentMemoryRepositoryError> {
    i64::try_from(value).map_err(|_| AgentMemoryRepositoryError::CorruptRecord)
}

fn from_i64(value: i64) -> Result<u64, AgentMemoryRepositoryError> {
    u64::try_from(value).map_err(|_| AgentMemoryRepositoryError::CorruptRecord)
}

fn positive_i64(value: u64) -> Result<i64, AgentMemoryRepositoryError> {
    if value == 0 {
        return Err(AgentMemoryRepositoryError::InvalidRetention);
    }
    i64::try_from(value).map_err(|_| AgentMemoryRepositoryError::InvalidRetention)
}

fn from_positive_i64(value: i64) -> Result<u64, AgentMemoryRepositoryError> {
    let value = u64::try_from(value).map_err(|_| AgentMemoryRepositoryError::CorruptRecord)?;
    if value == 0 {
        return Err(AgentMemoryRepositoryError::CorruptRecord);
    }
    Ok(value)
}
