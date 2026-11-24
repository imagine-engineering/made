use db::sqlez_macros::sql;
use nostr::nips::nip44::{self, Version};
use nostr::{Keys, SecretKey};
use nostr_compat::buzz_nips::agent_activity::{AgentTurnMetricPayload, TokenCounts};
use nostr_compat::dm::Nip44Ciphertext;
use nostr_compat::generated_kinds::KIND_AGENT_TURN_METRIC;
use nostr_compat::{CanonicalEvent, EventId, PublicKey};
use sha2::{Digest as _, Sha256};
use sqlez::domain::Domain;
use std::collections::BTreeMap;
use std::fmt;
use thiserror::Error;
use zeroize::Zeroizing;

const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

pub struct AgentUsageDatabase(db::sqlez::thread_safe_connection::ThreadSafeConnection);

impl Domain for AgentUsageDatabase {
    const NAME: &str = stringify!(AgentUsageDatabase);

    const MIGRATIONS: &[&str] = &[sql!(
        CREATE TABLE private_agent_turn_usage (
            owner_public_key TEXT NOT NULL,
            agent_public_key TEXT NOT NULL,
            event_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 BLOB NOT NULL,
            ciphertext TEXT NOT NULL,
            ciphertext_sha256 BLOB NOT NULL,
            retention_generation INTEGER NOT NULL,
            expires_at INTEGER,
            PRIMARY KEY (owner_public_key, event_id),
            CHECK (length(owner_public_key) = 64),
            CHECK (length(agent_public_key) = 64),
            CHECK (length(event_id) = 64),
            CHECK (created_at > 0),
            CHECK (length(payload_json) <= 65535),
            CHECK (length(payload_sha256) = 32),
            CHECK (length(ciphertext) >= 132),
            CHECK (length(ciphertext) <= 87472),
            CHECK (length(ciphertext_sha256) = 32),
            CHECK (retention_generation > 0),
            CHECK (retention_generation <= 9007199254740991),
            CHECK (expires_at IS NULL OR expires_at > 0)
        ) STRICT;

        CREATE INDEX private_agent_turn_usage_owner_time
            ON private_agent_turn_usage(owner_public_key, created_at, event_id);

        CREATE INDEX private_agent_turn_usage_expiry
            ON private_agent_turn_usage(owner_public_key, expires_at)
            WHERE expires_at IS NOT NULL;
    )];
}

db::static_connection!(AgentUsageDatabase, []);

impl AgentUsageDatabase {
    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self(database.0.clone())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClientTelemetryIntegration {
    DisabledByDesign,
}

pub const CLIENT_TELEMETRY_INTEGRATION: ClientTelemetryIntegration =
    ClientTelemetryIntegration::DisabledByDesign;

#[derive(Clone, Eq, PartialEq)]
pub struct EncryptedTurnUsage {
    agent: PublicKey,
    owner: PublicKey,
    ciphertext: Nip44Ciphertext,
    plaintext_sha256: [u8; 32],
}

impl EncryptedTurnUsage {
    pub fn agent(&self) -> PublicKey {
        self.agent
    }

    pub fn owner(&self) -> PublicKey {
        self.owner
    }

    pub fn ciphertext(&self) -> &Nip44Ciphertext {
        &self.ciphertext
    }

    pub fn to_canonical_event(&self, created_at: u64) -> CanonicalEvent {
        metric_event(
            self.agent,
            self.owner,
            created_at,
            self.ciphertext.wire_value().to_owned(),
        )
    }
}

impl fmt::Debug for EncryptedTurnUsage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncryptedTurnUsage")
            .field("agent", &self.agent)
            .field("owner", &self.owner)
            .field("ciphertext", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UsageRetention {
    generation: u64,
    expires_at: Option<u64>,
}

impl UsageRetention {
    pub fn new(
        generation: u64,
        expires_at: Option<u64>,
    ) -> Result<Self, AgentUsageRepositoryError> {
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

#[derive(Clone, PartialEq)]
pub struct StoredTurnUsage {
    owner: PublicKey,
    agent: PublicKey,
    event_id: EventId,
    created_at: u64,
    payload: AgentTurnMetricPayload,
    payload_sha256: [u8; 32],
    ciphertext: Nip44Ciphertext,
    ciphertext_sha256: [u8; 32],
    retention: UsageRetention,
}

impl StoredTurnUsage {
    pub fn new(
        encrypted: &EncryptedTurnUsage,
        event_id: EventId,
        created_at: u64,
        payload: AgentTurnMetricPayload,
        retention: UsageRetention,
    ) -> Result<Self, AgentUsageRepositoryError> {
        if created_at == 0 {
            return Err(AgentUsageRepositoryError::InvalidTime);
        }
        if encrypted
            .to_canonical_event(created_at)
            .event_id()
            .map_err(|error| AgentUsageRepositoryError::Unavailable(error.into()))?
            != event_id
        {
            return Err(AgentUsageRepositoryError::CorruptRecord);
        }
        let payload_json = encode_payload(&payload)?;
        let payload_sha256 = hash_bytes(&payload_json);
        if payload_sha256 != encrypted.plaintext_sha256 {
            return Err(AgentUsageRepositoryError::PayloadMismatch);
        }
        let ciphertext = encrypted.ciphertext.clone();
        Ok(Self {
            owner: encrypted.owner,
            agent: encrypted.agent,
            event_id,
            created_at,
            payload,
            payload_sha256,
            ciphertext_sha256: hash_bytes(ciphertext.wire_value().as_bytes()),
            ciphertext,
            retention,
        })
    }

    pub fn owner(&self) -> PublicKey {
        self.owner
    }

    pub fn agent(&self) -> PublicKey {
        self.agent
    }

    pub fn event_id(&self) -> EventId {
        self.event_id
    }

    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    pub fn payload(&self) -> &AgentTurnMetricPayload {
        &self.payload
    }

    pub fn ciphertext(&self) -> &Nip44Ciphertext {
        &self.ciphertext
    }

    pub fn retention(&self) -> UsageRetention {
        self.retention
    }

    pub fn to_canonical_event(&self) -> CanonicalEvent {
        metric_event(
            self.agent,
            self.owner,
            self.created_at,
            self.ciphertext.wire_value().to_owned(),
        )
    }
}

impl fmt::Debug for StoredTurnUsage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoredTurnUsage")
            .field("owner", &self.owner)
            .field("agent", &self.agent)
            .field("event_id", &self.event_id)
            .field("created_at", &self.created_at)
            .field("payload", &"<redacted>")
            .field("ciphertext", &"<redacted>")
            .field("retention", &self.retention)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UsageWriteOutcome {
    Stored,
    AlreadyStored,
    Conflict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UsageRetentionOutcome {
    Applied,
    Stale,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UsageAggregate {
    pub record_count: u64,
    pub accounted_turn_count: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cost_usd: Option<f64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UsageQuery {
    pub agent: Option<PublicKey>,
    pub created_at_or_after: Option<u64>,
    pub created_before: Option<u64>,
}

#[derive(Clone, Eq, PartialEq)]
pub struct EncryptedUsageExportRecord {
    owner: PublicKey,
    agent: PublicKey,
    event_id: EventId,
    created_at: u64,
    ciphertext: Nip44Ciphertext,
    retention: UsageRetention,
}

impl EncryptedUsageExportRecord {
    pub fn owner(&self) -> PublicKey {
        self.owner
    }

    pub fn agent(&self) -> PublicKey {
        self.agent
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

    pub fn retention(&self) -> UsageRetention {
        self.retention
    }

    pub fn to_canonical_event(&self) -> CanonicalEvent {
        metric_event(
            self.agent,
            self.owner,
            self.created_at,
            self.ciphertext.wire_value().to_owned(),
        )
    }
}

impl fmt::Debug for EncryptedUsageExportRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncryptedUsageExportRecord")
            .field("owner", &self.owner)
            .field("agent", &self.agent)
            .field("event_id", &self.event_id)
            .field("created_at", &self.created_at)
            .field("ciphertext", &"<redacted>")
            .field("retention", &self.retention)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum AgentUsageCryptoError {
    #[error("invalid usage secret key")]
    InvalidSecretKey,
    #[error("usage participants must be distinct")]
    SameParticipant,
    #[error("the supplied key cannot read this usage metric")]
    WrongReader,
    #[error("usage payload is invalid")]
    InvalidPayload,
    #[error("usage encryption failed")]
    EncryptionFailed,
    #[error("usage decryption failed")]
    DecryptionFailed,
}

#[derive(Debug, Error)]
pub enum AgentUsageRepositoryError {
    #[error("agent-usage repository is unavailable")]
    Unavailable(#[source] anyhow::Error),
    #[error("agent-usage owner does not match the authenticated owner")]
    OwnerMismatch,
    #[error("agent-usage record is corrupt")]
    CorruptRecord,
    #[error("agent-usage payload failed integrity verification")]
    CorruptPayload,
    #[error("agent-usage ciphertext failed integrity verification")]
    CorruptCiphertext,
    #[error("agent-usage encrypted and local payloads do not match")]
    PayloadMismatch,
    #[error("agent-usage payload is invalid")]
    InvalidPayload,
    #[error("agent-usage retention state is invalid")]
    InvalidRetention,
    #[error("agent-usage time is invalid")]
    InvalidTime,
    #[error("agent-usage aggregate overflowed")]
    AggregateOverflow,
    #[error("agent-usage cumulative series is ambiguous")]
    AmbiguousSeries,
}

impl From<anyhow::Error> for AgentUsageRepositoryError {
    fn from(error: anyhow::Error) -> Self {
        Self::Unavailable(error)
    }
}

pub fn encrypt_turn_usage_for_owner(
    agent_secret: &[u8; 32],
    owner: PublicKey,
    payload: &AgentTurnMetricPayload,
) -> Result<EncryptedTurnUsage, AgentUsageCryptoError> {
    let secret_key =
        SecretKey::from_slice(agent_secret).map_err(|_| AgentUsageCryptoError::InvalidSecretKey)?;
    let keys = Keys::new(secret_key.clone());
    let agent = PublicKey::from_bytes(keys.public_key().to_bytes());
    if agent == owner {
        return Err(AgentUsageCryptoError::SameParticipant);
    }
    let plaintext =
        Zeroizing::new(encode_payload(payload).map_err(|_| AgentUsageCryptoError::InvalidPayload)?);
    let owner_key = nostr_public_key(owner).map_err(|_| AgentUsageCryptoError::InvalidPayload)?;
    let ciphertext = nip44::encrypt(&secret_key, &owner_key, plaintext.as_slice(), Version::V2)
        .map_err(|_| AgentUsageCryptoError::EncryptionFailed)?;
    let ciphertext =
        Nip44Ciphertext::parse(ciphertext).map_err(|_| AgentUsageCryptoError::EncryptionFailed)?;
    Ok(EncryptedTurnUsage {
        agent,
        owner,
        plaintext_sha256: hash_bytes(plaintext.as_slice()),
        ciphertext,
    })
}

pub fn decrypt_turn_usage_as_owner(
    owner_secret: &[u8; 32],
    encrypted: &EncryptedTurnUsage,
) -> Result<AgentTurnMetricPayload, AgentUsageCryptoError> {
    decrypt_turn_usage(owner_secret, encrypted.owner, encrypted.agent, encrypted)
}

pub fn decrypt_turn_usage_as_agent(
    agent_secret: &[u8; 32],
    encrypted: &EncryptedTurnUsage,
) -> Result<AgentTurnMetricPayload, AgentUsageCryptoError> {
    decrypt_turn_usage(agent_secret, encrypted.agent, encrypted.owner, encrypted)
}

fn decrypt_turn_usage(
    reader_secret: &[u8; 32],
    expected_reader: PublicKey,
    counterparty: PublicKey,
    encrypted: &EncryptedTurnUsage,
) -> Result<AgentTurnMetricPayload, AgentUsageCryptoError> {
    let secret_key = SecretKey::from_slice(reader_secret)
        .map_err(|_| AgentUsageCryptoError::InvalidSecretKey)?;
    let keys = Keys::new(secret_key.clone());
    if PublicKey::from_bytes(keys.public_key().to_bytes()) != expected_reader {
        return Err(AgentUsageCryptoError::WrongReader);
    }
    let counterparty =
        nostr_public_key(counterparty).map_err(|_| AgentUsageCryptoError::DecryptionFailed)?;
    let plaintext = Zeroizing::new(
        nip44::decrypt_to_bytes(
            &secret_key,
            &counterparty,
            encrypted.ciphertext.wire_value(),
        )
        .map_err(|_| AgentUsageCryptoError::DecryptionFailed)?,
    );
    if hash_bytes(plaintext.as_slice()) != encrypted.plaintext_sha256 {
        return Err(AgentUsageCryptoError::DecryptionFailed);
    }
    AgentTurnMetricPayload::parse(plaintext.as_slice())
        .map_err(|_| AgentUsageCryptoError::DecryptionFailed)
}

#[derive(Clone)]
pub struct AgentUsageRepository {
    database: AgentUsageDatabase,
}

impl AgentUsageRepository {
    pub fn global(cx: &gpui::App) -> Self {
        Self {
            database: AgentUsageDatabase::global(cx),
        }
    }

    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self {
            database: AgentUsageDatabase::from_app_database(database),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_database(name: &'static str) -> Self {
        Self {
            database: AgentUsageDatabase::open_test_db(name).await,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_file_database(database_directory: &std::path::Path) -> Self {
        Self {
            database: AgentUsageDatabase(
                db::open_db::<AgentUsageDatabase>(database_directory, db::GlobalDbScope).await,
            ),
        }
    }

    pub async fn store(
        &self,
        authenticated_owner: PublicKey,
        record: &StoredTurnUsage,
    ) -> Result<UsageWriteOutcome, AgentUsageRepositoryError> {
        authorize_owner(authenticated_owner, record.owner)?;
        validate_record(record)?;
        let row = PersistedUsageRow::from_record(record)?;
        self.database
            .write(move |connection| insert_record(connection, &row))
            .await
            .map_err(AgentUsageRepositoryError::Unavailable)
    }

    pub fn load_for_owner(
        &self,
        authenticated_owner: PublicKey,
        event_id: EventId,
        now: u64,
    ) -> Result<Option<StoredTurnUsage>, AgentUsageRepositoryError> {
        let owner = authenticated_owner.to_hex();
        let event_id = event_id.to_hex();
        let now = to_i64(now)?;
        let row = self
            .database
            .select_row_bound::<(&str, &str), PersistedUsageTuple>(
                "SELECT agent_public_key, created_at, payload_json, payload_sha256, ciphertext, \
                    ciphertext_sha256, retention_generation, expires_at \
             FROM private_agent_turn_usage WHERE owner_public_key = ? AND event_id = ?",
            )?((&owner, &event_id))?;
        let Some(row) = row else {
            return Ok(None);
        };
        if row.7.is_some_and(|expires_at| expires_at <= now) {
            return Ok(None);
        }
        decode_record(&owner, &event_id, row).map(Some)
    }

    pub fn aggregate_for_owner(
        &self,
        authenticated_owner: PublicKey,
        query: UsageQuery,
        now: u64,
    ) -> Result<UsageAggregate, AgentUsageRepositoryError> {
        let records = self.records_for_owner(authenticated_owner, query, now)?;
        aggregate_records(&records)
    }

    pub fn export_encrypted_for_owner(
        &self,
        authenticated_owner: PublicKey,
        query: UsageQuery,
        now: u64,
    ) -> Result<Vec<EncryptedUsageExportRecord>, AgentUsageRepositoryError> {
        Ok(self
            .records_for_owner(authenticated_owner, query, now)?
            .into_iter()
            .map(|record| EncryptedUsageExportRecord {
                owner: record.owner,
                agent: record.agent,
                event_id: record.event_id,
                created_at: record.created_at,
                ciphertext: record.ciphertext,
                retention: record.retention,
            })
            .collect())
    }

    pub async fn expire(
        &self,
        authenticated_owner: PublicKey,
        event_id: EventId,
        expected_retention_generation: u64,
        expires_at: u64,
    ) -> Result<UsageRetentionOutcome, AgentUsageRepositoryError> {
        let expected_generation = retention_generation_to_i64(expected_retention_generation)?;
        let next_generation = expected_retention_generation
            .checked_add(1)
            .ok_or(AgentUsageRepositoryError::InvalidRetention)
            .and_then(retention_generation_to_i64)?;
        let expires_at = positive_i64(expires_at)?;
        let owner = authenticated_owner.to_hex();
        let event_id = event_id.to_hex();
        self.database
            .write(move |connection| -> anyhow::Result<UsageRetentionOutcome> {
                connection.exec_bound::<(i64, i64, i64, &str, &str, i64)>(
                    "UPDATE private_agent_turn_usage SET retention_generation = ?, \
                         expires_at = CASE WHEN expires_at IS NULL OR expires_at > ? \
                                           THEN ? ELSE expires_at END \
                     WHERE owner_public_key = ? AND event_id = ? AND retention_generation = ?",
                )?((
                    next_generation,
                    expires_at,
                    expires_at,
                    &owner,
                    &event_id,
                    expected_generation,
                ))?;
                Ok(match changed_rows(connection)? {
                    1 => UsageRetentionOutcome::Applied,
                    0 => UsageRetentionOutcome::Stale,
                    _ => anyhow::bail!("usage expiry affected an invalid row count"),
                })
            })
            .await
            .map_err(AgentUsageRepositoryError::Unavailable)
    }

    pub async fn purge_expired(
        &self,
        authenticated_owner: PublicKey,
        now: u64,
    ) -> Result<u64, AgentUsageRepositoryError> {
        let owner = authenticated_owner.to_hex();
        let now = to_i64(now)?;
        self.database
            .write(move |connection| -> anyhow::Result<u64> {
                connection.exec_bound::<(&str, i64)>(
                    "DELETE FROM private_agent_turn_usage \
                     WHERE owner_public_key = ? AND expires_at IS NOT NULL AND expires_at <= ?",
                )?((&owner, now))?;
                let changed = changed_rows(connection)?;
                u64::try_from(changed).map_err(Into::into)
            })
            .await
            .map_err(AgentUsageRepositoryError::Unavailable)
    }

    fn records_for_owner(
        &self,
        authenticated_owner: PublicKey,
        query: UsageQuery,
        now: u64,
    ) -> Result<Vec<StoredTurnUsage>, AgentUsageRepositoryError> {
        validate_query(query)?;
        let owner = authenticated_owner.to_hex();
        let now = to_i64(now)?;
        let rows = self
            .database
            .select_bound::<(&str, i64), PersistedUsageWithIdTuple>(
                "SELECT event_id, agent_public_key, created_at, payload_json, payload_sha256, \
                    ciphertext, ciphertext_sha256, retention_generation, expires_at \
             FROM private_agent_turn_usage \
             WHERE owner_public_key = ? AND (expires_at IS NULL OR expires_at > ?) \
             ORDER BY created_at ASC, event_id ASC",
            )?((&owner, now))?;
        rows.into_iter()
            .map(|row| {
                let event_id = row.0.clone();
                let record = decode_record(&owner, &event_id, without_event_id(row))?;
                Ok(record)
            })
            .filter(|record| match record {
                Ok(record) => record_matches_query(record, query),
                Err(_) => true,
            })
            .collect()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn corrupt_payload_for_test(
        &self,
        authenticated_owner: PublicKey,
        event_id: EventId,
    ) -> Result<(), AgentUsageRepositoryError> {
        let owner = authenticated_owner.to_hex();
        let event_id = event_id.to_hex();
        self.database
            .write(move |connection| -> anyhow::Result<()> {
                connection.exec_bound::<(&str, &str)>(
                    "UPDATE private_agent_turn_usage SET payload_json = '{}' \
                     WHERE owner_public_key = ? AND event_id = ?",
                )?((&owner, &event_id))?;
                Ok(())
            })
            .await
            .map_err(AgentUsageRepositoryError::Unavailable)
    }
}

type PersistedUsageTuple = (
    String,
    i64,
    String,
    Vec<u8>,
    String,
    Vec<u8>,
    i64,
    Option<i64>,
);
type PersistedUsageWithIdTuple = (
    String,
    String,
    i64,
    String,
    Vec<u8>,
    String,
    Vec<u8>,
    i64,
    Option<i64>,
);

struct PersistedUsageRow {
    owner_public_key: String,
    agent_public_key: String,
    event_id: String,
    created_at: i64,
    payload_json: String,
    payload_sha256: Vec<u8>,
    ciphertext: String,
    ciphertext_sha256: Vec<u8>,
    retention_generation: i64,
    expires_at: Option<i64>,
}

impl PersistedUsageRow {
    fn from_record(record: &StoredTurnUsage) -> Result<Self, AgentUsageRepositoryError> {
        Ok(Self {
            owner_public_key: record.owner.to_hex(),
            agent_public_key: record.agent.to_hex(),
            event_id: record.event_id.to_hex(),
            created_at: to_i64(record.created_at)?,
            payload_json: String::from_utf8(encode_payload(&record.payload)?)
                .map_err(|_| AgentUsageRepositoryError::InvalidPayload)?,
            payload_sha256: record.payload_sha256.to_vec(),
            ciphertext: record.ciphertext.wire_value().to_owned(),
            ciphertext_sha256: record.ciphertext_sha256.to_vec(),
            retention_generation: retention_generation_to_i64(record.retention.generation)?,
            expires_at: record.retention.expires_at.map(positive_i64).transpose()?,
        })
    }

    fn matches_tuple(&self, row: &PersistedUsageTuple) -> bool {
        self.agent_public_key == row.0
            && self.created_at == row.1
            && self.payload_json == row.2
            && self.payload_sha256 == row.3
            && self.ciphertext == row.4
            && self.ciphertext_sha256 == row.5
            && self.retention_generation == row.6
            && self.expires_at == row.7
    }
}

fn insert_record(
    connection: &db::sqlez::connection::Connection,
    row: &PersistedUsageRow,
) -> anyhow::Result<UsageWriteOutcome> {
    connection.exec_bound::<(
        &str,
        &str,
        &str,
        i64,
        &str,
        &[u8],
        &str,
        &[u8],
        i64,
        Option<i64>,
    )>(
        "INSERT OR IGNORE INTO private_agent_turn_usage( \
            owner_public_key, agent_public_key, event_id, created_at, payload_json, \
            payload_sha256, ciphertext, ciphertext_sha256, retention_generation, expires_at \
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )?((
        &row.owner_public_key,
        &row.agent_public_key,
        &row.event_id,
        row.created_at,
        &row.payload_json,
        &row.payload_sha256,
        &row.ciphertext,
        &row.ciphertext_sha256,
        row.retention_generation,
        row.expires_at,
    ))?;
    if changed_rows(connection)? == 1 {
        return Ok(UsageWriteOutcome::Stored);
    }
    let current = connection.select_row_bound::<(&str, &str), PersistedUsageTuple>(
        "SELECT agent_public_key, created_at, payload_json, payload_sha256, ciphertext, \
                    ciphertext_sha256, retention_generation, expires_at \
             FROM private_agent_turn_usage WHERE owner_public_key = ? AND event_id = ?",
    )?((&row.owner_public_key, &row.event_id))?;
    Ok(
        if current
            .as_ref()
            .is_some_and(|current| row.matches_tuple(current))
        {
            UsageWriteOutcome::AlreadyStored
        } else {
            UsageWriteOutcome::Conflict
        },
    )
}

fn decode_record(
    owner: &str,
    event_id: &str,
    row: PersistedUsageTuple,
) -> Result<StoredTurnUsage, AgentUsageRepositoryError> {
    let owner = PublicKey::from_hex(owner).map_err(|_| AgentUsageRepositoryError::CorruptRecord)?;
    let agent =
        PublicKey::from_hex(&row.0).map_err(|_| AgentUsageRepositoryError::CorruptRecord)?;
    let event_id =
        EventId::from_hex(event_id).map_err(|_| AgentUsageRepositoryError::CorruptRecord)?;
    let created_at = from_positive_i64(row.1)?;
    let payload_digest: [u8; 32] = row
        .3
        .try_into()
        .map_err(|_| AgentUsageRepositoryError::CorruptPayload)?;
    if hash_bytes(row.2.as_bytes()) != payload_digest {
        return Err(AgentUsageRepositoryError::CorruptPayload);
    }
    let payload = AgentTurnMetricPayload::parse(row.2.as_bytes())
        .map_err(|_| AgentUsageRepositoryError::CorruptPayload)?;
    validate_observed_usage(&payload).map_err(|_| AgentUsageRepositoryError::CorruptPayload)?;
    let ciphertext =
        Nip44Ciphertext::parse(row.4).map_err(|_| AgentUsageRepositoryError::CorruptCiphertext)?;
    let ciphertext_digest: [u8; 32] = row
        .5
        .try_into()
        .map_err(|_| AgentUsageRepositoryError::CorruptCiphertext)?;
    if hash_bytes(ciphertext.wire_value().as_bytes()) != ciphertext_digest {
        return Err(AgentUsageRepositoryError::CorruptCiphertext);
    }
    if metric_event(agent, owner, created_at, ciphertext.wire_value().to_owned())
        .event_id()
        .map_err(|_| AgentUsageRepositoryError::CorruptRecord)?
        != event_id
    {
        return Err(AgentUsageRepositoryError::CorruptRecord);
    }
    let retention = UsageRetention {
        generation: retention_generation_from_i64(row.6)?,
        expires_at: row.7.map(from_positive_i64).transpose()?,
    };
    Ok(StoredTurnUsage {
        owner,
        agent,
        event_id,
        created_at,
        payload,
        payload_sha256: payload_digest,
        ciphertext,
        ciphertext_sha256: ciphertext_digest,
        retention,
    })
}

fn without_event_id(row: PersistedUsageWithIdTuple) -> PersistedUsageTuple {
    (row.1, row.2, row.3, row.4, row.5, row.6, row.7, row.8)
}

fn aggregate_records(
    records: &[StoredTurnUsage],
) -> Result<UsageAggregate, AgentUsageRepositoryError> {
    let mut aggregate = UsageAggregate {
        record_count: u64::try_from(records.len())
            .map_err(|_| AgentUsageRepositoryError::AggregateOverflow)?,
        ..UsageAggregate::default()
    };
    let mut standalone = Vec::new();
    let mut cumulative: BTreeMap<(PublicKey, String), Vec<&StoredTurnUsage>> = BTreeMap::new();
    for record in records {
        if record.payload.cumulative.is_some() {
            let session_id = record
                .payload
                .session_id
                .as_ref()
                .ok_or(AgentUsageRepositoryError::CorruptPayload)?;
            cumulative
                .entry((record.agent, session_id.clone()))
                .or_default()
                .push(record);
        } else {
            standalone.push(record);
        }
    }
    for record in standalone {
        if record.payload.delta_reliable
            && let Some(turn) = &record.payload.turn
        {
            aggregate.add_counts(turn)?;
        }
    }
    for series in cumulative.values_mut() {
        series.sort_by_key(|record| (record.payload.turn_seq, record.event_id));
        let mut previous: Option<&TokenCounts> = None;
        let mut previous_sequence = None;
        for record in series {
            let sequence = record
                .payload
                .turn_seq
                .ok_or(AgentUsageRepositoryError::CorruptPayload)?;
            if previous_sequence == Some(sequence) {
                return Err(AgentUsageRepositoryError::AmbiguousSeries);
            }
            let current = record
                .payload
                .cumulative
                .as_ref()
                .ok_or(AgentUsageRepositoryError::CorruptPayload)?;
            if let Some(previous) = previous {
                aggregate.add_counts(&delta_counts(previous, current))?;
            } else if record.payload.delta_reliable
                && let Some(turn) = &record.payload.turn
            {
                aggregate.add_counts(turn)?;
            }
            previous = Some(current);
            previous_sequence = Some(sequence);
        }
    }
    Ok(aggregate)
}

impl UsageAggregate {
    fn add_counts(&mut self, counts: &TokenCounts) -> Result<(), AgentUsageRepositoryError> {
        if counts_are_empty(counts) {
            return Ok(());
        }
        self.accounted_turn_count = self
            .accounted_turn_count
            .checked_add(1)
            .ok_or(AgentUsageRepositoryError::AggregateOverflow)?;
        add_optional_u64(&mut self.input_tokens, counts.input_tokens)?;
        add_optional_u64(&mut self.output_tokens, counts.output_tokens)?;
        add_optional_u64(&mut self.total_tokens, counts.total_tokens)?;
        add_optional_cost(&mut self.cost_usd, counts.cost_usd)?;
        add_optional_u64(&mut self.cache_read_tokens, counts.cache_read_tokens)?;
        add_optional_u64(&mut self.cache_write_tokens, counts.cache_write_tokens)?;
        Ok(())
    }
}

fn delta_counts(previous: &TokenCounts, current: &TokenCounts) -> TokenCounts {
    TokenCounts {
        input_tokens: checked_delta(previous.input_tokens, current.input_tokens),
        output_tokens: checked_delta(previous.output_tokens, current.output_tokens),
        total_tokens: checked_delta(previous.total_tokens, current.total_tokens),
        cost_usd: checked_cost_delta(previous.cost_usd, current.cost_usd),
        cache_read_tokens: checked_delta(previous.cache_read_tokens, current.cache_read_tokens),
        cache_write_tokens: checked_delta(previous.cache_write_tokens, current.cache_write_tokens),
    }
}

fn checked_delta(previous: Option<u64>, current: Option<u64>) -> Option<u64> {
    current
        .zip(previous)
        .and_then(|(current, previous)| current.checked_sub(previous))
}

fn checked_cost_delta(previous: Option<f64>, current: Option<f64>) -> Option<f64> {
    current
        .zip(previous)
        .and_then(|(current, previous)| (current >= previous).then_some(current - previous))
}

fn add_optional_u64(
    total: &mut Option<u64>,
    value: Option<u64>,
) -> Result<(), AgentUsageRepositoryError> {
    let Some(value) = value else {
        return Ok(());
    };
    *total = Some(
        total
            .unwrap_or_default()
            .checked_add(value)
            .ok_or(AgentUsageRepositoryError::AggregateOverflow)?,
    );
    Ok(())
}

fn add_optional_cost(
    total: &mut Option<f64>,
    value: Option<f64>,
) -> Result<(), AgentUsageRepositoryError> {
    let Some(value) = value else {
        return Ok(());
    };
    let next = total.unwrap_or_default() + value;
    if !next.is_finite() {
        return Err(AgentUsageRepositoryError::AggregateOverflow);
    }
    *total = Some(next);
    Ok(())
}

fn validate_record(record: &StoredTurnUsage) -> Result<(), AgentUsageRepositoryError> {
    if record.owner == record.agent {
        return Err(AgentUsageRepositoryError::CorruptRecord);
    }
    from_positive_i64(to_i64(record.created_at)?)?;
    retention_generation_to_i64(record.retention.generation)?;
    if let Some(expires_at) = record.retention.expires_at {
        positive_i64(expires_at)?;
    }
    let payload = encode_payload(&record.payload)?;
    if hash_bytes(&payload) != record.payload_sha256 {
        return Err(AgentUsageRepositoryError::CorruptPayload);
    }
    if hash_bytes(record.ciphertext.wire_value().as_bytes()) != record.ciphertext_sha256 {
        return Err(AgentUsageRepositoryError::CorruptCiphertext);
    }
    if record
        .to_canonical_event()
        .event_id()
        .map_err(|error| AgentUsageRepositoryError::Unavailable(error.into()))?
        != record.event_id
    {
        return Err(AgentUsageRepositoryError::CorruptRecord);
    }
    Ok(())
}

fn encode_payload(payload: &AgentTurnMetricPayload) -> Result<Vec<u8>, AgentUsageRepositoryError> {
    payload
        .validate()
        .map_err(|_| AgentUsageRepositoryError::InvalidPayload)?;
    validate_observed_usage(payload)?;
    let bytes =
        serde_json::to_vec(payload).map_err(|_| AgentUsageRepositoryError::InvalidPayload)?;
    AgentTurnMetricPayload::parse(&bytes).map_err(|_| AgentUsageRepositoryError::InvalidPayload)?;
    Ok(bytes)
}

fn validate_observed_usage(
    payload: &AgentTurnMetricPayload,
) -> Result<(), AgentUsageRepositoryError> {
    if payload
        .turn
        .as_ref()
        .is_some_and(|counts| !counts_are_empty(counts))
        || payload
            .cumulative
            .as_ref()
            .is_some_and(|counts| !counts_are_empty(counts))
    {
        Ok(())
    } else {
        Err(AgentUsageRepositoryError::InvalidPayload)
    }
}

fn counts_are_empty(counts: &TokenCounts) -> bool {
    counts.input_tokens.is_none()
        && counts.output_tokens.is_none()
        && counts.total_tokens.is_none()
        && counts.cost_usd.is_none()
        && counts.cache_read_tokens.is_none()
        && counts.cache_write_tokens.is_none()
}

fn validate_query(query: UsageQuery) -> Result<(), AgentUsageRepositoryError> {
    if query
        .created_at_or_after
        .zip(query.created_before)
        .is_some_and(|(start, end)| start >= end)
    {
        return Err(AgentUsageRepositoryError::InvalidTime);
    }
    Ok(())
}

fn record_matches_query(record: &StoredTurnUsage, query: UsageQuery) -> bool {
    query.agent.is_none_or(|agent| agent == record.agent)
        && query
            .created_at_or_after
            .is_none_or(|start| record.created_at >= start)
        && query
            .created_before
            .is_none_or(|end| record.created_at < end)
}

fn authorize_owner(
    authenticated_owner: PublicKey,
    record_owner: PublicKey,
) -> Result<(), AgentUsageRepositoryError> {
    if authenticated_owner != record_owner {
        return Err(AgentUsageRepositoryError::OwnerMismatch);
    }
    Ok(())
}

fn nostr_public_key(public_key: PublicKey) -> Result<nostr::PublicKey, AgentUsageCryptoError> {
    nostr::PublicKey::from_slice(public_key.as_bytes())
        .map_err(|_| AgentUsageCryptoError::InvalidPayload)
}

fn metric_event(
    agent: PublicKey,
    owner: PublicKey,
    created_at: u64,
    ciphertext: String,
) -> CanonicalEvent {
    CanonicalEvent::new(
        agent,
        created_at,
        KIND_AGENT_TURN_METRIC as u16,
        vec![
            vec!["p".to_owned(), owner.to_hex()],
            vec!["agent".to_owned(), agent.to_hex()],
        ],
        ciphertext,
    )
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn changed_rows(connection: &db::sqlez::connection::Connection) -> anyhow::Result<i64> {
    connection.select_row_bound::<(), i64>("SELECT changes()")?(())?
        .ok_or_else(|| anyhow::anyhow!("SQLite did not report a changed-row count"))
}

fn retention_generation_to_i64(value: u64) -> Result<i64, AgentUsageRepositoryError> {
    if value == 0 || value > MAX_SAFE_INTEGER {
        return Err(AgentUsageRepositoryError::InvalidRetention);
    }
    i64::try_from(value).map_err(|_| AgentUsageRepositoryError::InvalidRetention)
}

fn retention_generation_from_i64(value: i64) -> Result<u64, AgentUsageRepositoryError> {
    let value = u64::try_from(value).map_err(|_| AgentUsageRepositoryError::CorruptRecord)?;
    if value == 0 || value > MAX_SAFE_INTEGER {
        return Err(AgentUsageRepositoryError::CorruptRecord);
    }
    Ok(value)
}

fn to_i64(value: u64) -> Result<i64, AgentUsageRepositoryError> {
    i64::try_from(value).map_err(|_| AgentUsageRepositoryError::InvalidTime)
}

fn positive_i64(value: u64) -> Result<i64, AgentUsageRepositoryError> {
    if value == 0 {
        return Err(AgentUsageRepositoryError::InvalidRetention);
    }
    i64::try_from(value).map_err(|_| AgentUsageRepositoryError::InvalidRetention)
}

fn from_positive_i64(value: i64) -> Result<u64, AgentUsageRepositoryError> {
    let value = u64::try_from(value).map_err(|_| AgentUsageRepositoryError::CorruptRecord)?;
    if value == 0 {
        return Err(AgentUsageRepositoryError::CorruptRecord);
    }
    Ok(value)
}
