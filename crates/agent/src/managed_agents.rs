use agent_settings::{
    managed_agent::{
        EnvironmentReference, EnvironmentVariableName, ManagedAgentConfiguration,
        ManagedAgentState, ManagedAgentVersion, ModelId, PrivateManagedAgentRecord,
        ProtectedCredentialReference, ProviderId, RuntimeId,
    },
    team::{NostrEventId as SettingsEventId, NostrPublicKey as SettingsPublicKey},
};
use collaboration_domain::PublicAgentCatalogProjection;
use db::sqlez_macros::sql;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlez::domain::Domain;
use std::collections::BTreeMap;
use thiserror::Error;

const MANAGED_AGENT_SNAPSHOT_VERSION: u32 = 1;
const PUBLIC_PROJECTION_SNAPSHOT_VERSION: u32 = 1;
const MAX_SAFE_GENERATION: u64 = (1_u64 << 53) - 1;

pub struct ManagedAgentDatabase(db::sqlez::thread_safe_connection::ThreadSafeConnection);

impl Domain for ManagedAgentDatabase {
    const NAME: &str = stringify!(ManagedAgentDatabase);

    const MIGRATIONS: &[&str] = &[sql!(
        CREATE TABLE managed_agent_snapshots (
            owner_public_key TEXT NOT NULL,
            agent_public_key TEXT NOT NULL,
            generation INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            PRIMARY KEY (owner_public_key, agent_public_key),
            CHECK (generation > 0),
            CHECK (generation <= 9007199254740991)
        ) STRICT;

        CREATE TABLE managed_agent_projection_snapshots (
            owner_public_key TEXT NOT NULL,
            agent_public_key TEXT NOT NULL,
            source_generation INTEGER NOT NULL,
            source_event_id TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            projection_revision INTEGER NOT NULL,
            projected_at INTEGER NOT NULL,
            projection_json TEXT NOT NULL,
            PRIMARY KEY (owner_public_key, agent_public_key),
            FOREIGN KEY (owner_public_key, agent_public_key)
                REFERENCES managed_agent_snapshots(owner_public_key, agent_public_key)
                ON DELETE CASCADE,
            CHECK (source_generation > 0),
            CHECK (source_generation <= 9007199254740991),
            CHECK (schema_version = 1),
            CHECK (projection_revision > 0),
            CHECK (projection_revision <= 9007199254740991),
            CHECK (projected_at > 0)
        ) STRICT;
    )];
}

db::static_connection!(ManagedAgentDatabase, []);

impl ManagedAgentDatabase {
    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self(database.0.clone())
    }
}

#[derive(Clone)]
pub struct ManagedAgentRepository {
    database: ManagedAgentDatabase,
}

impl ManagedAgentRepository {
    pub fn global(cx: &gpui::App) -> Self {
        Self {
            database: ManagedAgentDatabase::global(cx),
        }
    }

    pub fn from_app_database(database: &db::AppDatabase) -> Self {
        Self {
            database: ManagedAgentDatabase::from_app_database(database),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_database(name: &'static str) -> Self {
        Self {
            database: ManagedAgentDatabase::open_test_db(name).await,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_file_database(database_directory: &std::path::Path) -> Self {
        Self {
            database: ManagedAgentDatabase(
                db::open_db::<ManagedAgentDatabase>(database_directory, db::GlobalDbScope).await,
            ),
        }
    }

    pub async fn insert(
        &self,
        record: &PrivateManagedAgentRecord,
    ) -> Result<ManagedAgentInsertOutcome, ManagedAgentRepositoryError> {
        let snapshot = encode_snapshot(record)?;
        let owner_public_key = record.owner_public_key().as_str().to_string();
        let agent_public_key = record.agent_public_key().as_str().to_string();
        let generation = to_i64(record.version().generation())?;
        let event_id = record.version().event_id().as_str().to_string();

        self.database
            .write(
                move |connection| -> anyhow::Result<ManagedAgentInsertOutcome> {
                    connection.exec_bound::<(String, String, i64, String, String)>(
                        "INSERT OR IGNORE INTO managed_agent_snapshots(\
                        owner_public_key, agent_public_key, generation, event_id, snapshot_json\
                    ) VALUES (?, ?, ?, ?, ?)",
                    )?((
                        owner_public_key,
                        agent_public_key,
                        generation,
                        event_id,
                        snapshot,
                    ))?;
                    Ok(match changed_rows(connection)? {
                        1 => ManagedAgentInsertOutcome::Inserted,
                        0 => ManagedAgentInsertOutcome::AlreadyExists,
                        _ => anyhow::bail!("managed-agent insert affected an invalid row count"),
                    })
                },
            )
            .await
            .map_err(ManagedAgentRepositoryError::Unavailable)
    }

    pub fn load(
        &self,
        owner_public_key: &SettingsPublicKey,
        agent_public_key: &SettingsPublicKey,
    ) -> Result<Option<PrivateManagedAgentRecord>, ManagedAgentRepositoryError> {
        let row = self
            .database
            .select_row_bound::<(&str, &str), (i64, String, String)>(
                "SELECT generation, event_id, snapshot_json \
                 FROM managed_agent_snapshots \
                 WHERE owner_public_key = ? AND agent_public_key = ?",
            )?((owner_public_key.as_str(), agent_public_key.as_str()))
        .map_err(ManagedAgentRepositoryError::Unavailable)?;
        let Some((generation, event_id, snapshot)) = row else {
            return Ok(None);
        };
        let record = decode_snapshot(&snapshot)?;
        if record.owner_public_key() != owner_public_key
            || record.agent_public_key() != agent_public_key
            || record.version().generation() != from_i64(generation)?
            || record.version().event_id().as_str() != event_id
        {
            return Err(ManagedAgentRepositoryError::CorruptSnapshot);
        }
        Ok(Some(record))
    }

    pub async fn compare_and_swap(
        &self,
        expected_version: &ManagedAgentVersion,
        next_record: &PrivateManagedAgentRecord,
    ) -> Result<ManagedAgentCasOutcome, ManagedAgentRepositoryError> {
        validate_transition(expected_version, next_record)?;
        let snapshot = encode_snapshot(next_record)?;
        let owner_public_key = next_record.owner_public_key().as_str().to_string();
        let agent_public_key = next_record.agent_public_key().as_str().to_string();
        let next_generation = to_i64(next_record.version().generation())?;
        let next_event_id = next_record.version().event_id().as_str().to_string();
        let expected_generation = to_i64(expected_version.generation())?;
        let expected_event_id = expected_version.event_id().as_str().to_string();

        self.database
            .write(
                move |connection| -> anyhow::Result<ManagedAgentCasOutcome> {
                    connection.with_savepoint("managed_agent_compare_and_swap", || {
                        connection
                            .exec_bound::<(i64, String, String, String, String, i64, String)>(
                                "UPDATE managed_agent_snapshots \
                         SET generation = ?, event_id = ?, snapshot_json = ? \
                         WHERE owner_public_key = ? AND agent_public_key = ? \
                           AND generation = ? AND event_id = ?",
                            )?((
                            next_generation,
                            next_event_id,
                            snapshot,
                            owner_public_key.clone(),
                            agent_public_key.clone(),
                            expected_generation,
                            expected_event_id,
                        ))?;
                        match changed_rows(connection)? {
                            0 => return Ok(ManagedAgentCasOutcome::Stale),
                            1 => {}
                            _ => anyhow::bail!("managed-agent CAS affected an invalid row count"),
                        }
                        connection.exec_bound::<(&str, &str)>(
                            "DELETE FROM managed_agent_projection_snapshots \
                         WHERE owner_public_key = ? AND agent_public_key = ?",
                        )?((&owner_public_key, &agent_public_key))?;
                        Ok(ManagedAgentCasOutcome::Applied)
                    })
                },
            )
            .await
            .map_err(ManagedAgentRepositoryError::Unavailable)
    }

    pub async fn rebuild_public_projection(
        &self,
        source_record: &PrivateManagedAgentRecord,
        projection: &PublicAgentCatalogProjection,
        projected_at: u64,
    ) -> Result<ProjectionWriteOutcome, ManagedAgentRepositoryError> {
        if !matches!(source_record.state(), ManagedAgentState::Active(_)) {
            return Err(ManagedAgentRepositoryError::DeletedRecord);
        }
        if projected_at == 0 {
            return Err(ManagedAgentRepositoryError::InvalidProjection);
        }
        if lower_hex(projection.owner_public_key.as_bytes())
            != source_record.owner_public_key().as_str()
        {
            return Err(ManagedAgentRepositoryError::ProjectionOwnerMismatch);
        }
        let projection_value = serde_json::to_value(projection)
            .map_err(|error| ManagedAgentRepositoryError::Unavailable(error.into()))?;
        validate_public_projection_value(&projection_value)?;
        let projection_json = serde_json::to_string(&projection_value)
            .map_err(|error| ManagedAgentRepositoryError::Unavailable(error.into()))?;
        let owner_public_key = source_record.owner_public_key().as_str().to_string();
        let agent_public_key = source_record.agent_public_key().as_str().to_string();
        let source_generation = to_i64(source_record.version().generation())?;
        let source_event_id = source_record.version().event_id().as_str().to_string();
        let schema_version = i64::from(PUBLIC_PROJECTION_SNAPSHOT_VERSION);
        let projection_revision = source_generation;
        let projected_at = to_i64(projected_at)?;

        self.database
            .write(
                move |connection| -> anyhow::Result<ProjectionWriteOutcome> {
                    connection.with_savepoint("managed_agent_projection_rebuild", || {
                        let current =
                            connection.select_row_bound::<(&str, &str), (i64, String)>(
                                "SELECT generation, event_id FROM managed_agent_snapshots \
                         WHERE owner_public_key = ? AND agent_public_key = ?",
                            )?((&owner_public_key, &agent_public_key))?;
                        if current.as_ref() != Some(&(source_generation, source_event_id.clone())) {
                            return Ok(ProjectionWriteOutcome::Stale);
                        }
                        connection
                            .exec_bound::<(String, String, i64, String, i64, i64, i64, String)>(
                                "INSERT INTO managed_agent_projection_snapshots(\
                            owner_public_key, agent_public_key, source_generation, \
                            source_event_id, schema_version, projection_revision, projected_at, \
                            projection_json\
                         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
                         ON CONFLICT(owner_public_key, agent_public_key) DO UPDATE SET \
                            source_generation = excluded.source_generation, \
                            source_event_id = excluded.source_event_id, \
                            schema_version = excluded.schema_version, \
                            projection_revision = excluded.projection_revision, \
                            projected_at = excluded.projected_at, \
                            projection_json = excluded.projection_json",
                            )?((
                            owner_public_key,
                            agent_public_key,
                            source_generation,
                            source_event_id,
                            schema_version,
                            projection_revision,
                            projected_at,
                            projection_json,
                        ))?;
                        Ok(ProjectionWriteOutcome::Stored)
                    })
                },
            )
            .await
            .map_err(ManagedAgentRepositoryError::Unavailable)
    }

    pub fn load_public_projection(
        &self,
        owner_public_key: &SettingsPublicKey,
        agent_public_key: &SettingsPublicKey,
    ) -> Result<Option<StoredPublicProjection>, ManagedAgentRepositoryError> {
        let row = self
            .database
            .select_row_bound::<(&str, &str), (i64, String, i64, i64, i64, String)>(
                "SELECT source_generation, source_event_id, schema_version, projection_revision, \
                        projected_at, projection_json \
                 FROM managed_agent_projection_snapshots \
                 WHERE owner_public_key = ? AND agent_public_key = ?",
            )?((owner_public_key.as_str(), agent_public_key.as_str()))
        .map_err(ManagedAgentRepositoryError::Unavailable)?;
        let Some((
            source_generation,
            source_event_id,
            schema_version,
            revision,
            projected_at,
            projection_json,
        )) = row
        else {
            return Ok(None);
        };
        let source_generation = from_i64(source_generation)
            .map_err(|_| ManagedAgentRepositoryError::CorruptProjection)?;
        let revision =
            from_i64(revision).map_err(|_| ManagedAgentRepositoryError::CorruptProjection)?;
        let projected_at =
            from_i64(projected_at).map_err(|_| ManagedAgentRepositoryError::CorruptProjection)?;
        if schema_version != i64::from(PUBLIC_PROJECTION_SNAPSHOT_VERSION)
            || revision != source_generation
            || projected_at == 0
        {
            return Err(ManagedAgentRepositoryError::CorruptProjection);
        }
        let event_id = SettingsEventId::parse(source_event_id)
            .map_err(|_| ManagedAgentRepositoryError::CorruptProjection)?;
        let projection: Value = serde_json::from_str(&projection_json)
            .map_err(|_| ManagedAgentRepositoryError::CorruptProjection)?;
        validate_public_projection_value(&projection)
            .map_err(|_| ManagedAgentRepositoryError::CorruptProjection)?;
        Ok(Some(StoredPublicProjection {
            schema_version: PUBLIC_PROJECTION_SNAPSHOT_VERSION,
            source_version: ManagedAgentVersion::new(source_generation, event_id)
                .map_err(|_| ManagedAgentRepositoryError::CorruptProjection)?,
            projection_revision: revision,
            projected_at,
            projection,
        }))
    }

    pub async fn invalidate_public_projection(
        &self,
        owner_public_key: &SettingsPublicKey,
        agent_public_key: &SettingsPublicKey,
    ) -> Result<(), ManagedAgentRepositoryError> {
        let owner_public_key = owner_public_key.as_str().to_string();
        let agent_public_key = agent_public_key.as_str().to_string();
        self.database
            .write(move |connection| -> anyhow::Result<()> {
                connection.exec_bound::<(String, String)>(
                    "DELETE FROM managed_agent_projection_snapshots \
                     WHERE owner_public_key = ? AND agent_public_key = ?",
                )?((owner_public_key, agent_public_key))?;
                Ok(())
            })
            .await
            .map_err(ManagedAgentRepositoryError::Unavailable)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn corrupt_snapshot_for_test(
        &self,
        owner_public_key: &SettingsPublicKey,
        agent_public_key: &SettingsPublicKey,
    ) -> Result<(), ManagedAgentRepositoryError> {
        let owner_public_key = owner_public_key.as_str().to_string();
        let agent_public_key = agent_public_key.as_str().to_string();
        self.database
            .write(move |connection| -> anyhow::Result<()> {
                connection.exec_bound::<(String, String)>(
                    "UPDATE managed_agent_snapshots SET snapshot_json = '{\"schema_version\":1}' \
                     WHERE owner_public_key = ? AND agent_public_key = ?",
                )?((owner_public_key, agent_public_key))?;
                Ok(())
            })
            .await
            .map_err(ManagedAgentRepositoryError::Unavailable)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedAgentInsertOutcome {
    Inserted,
    AlreadyExists,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedAgentCasOutcome {
    Applied,
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionWriteOutcome {
    Stored,
    Stale,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoredPublicProjection {
    pub schema_version: u32,
    pub source_version: ManagedAgentVersion,
    pub projection_revision: u64,
    pub projected_at: u64,
    pub projection: Value,
}

#[derive(Debug, Error)]
pub enum ManagedAgentRepositoryError {
    #[error("managed-agent repository is unavailable")]
    Unavailable(#[source] anyhow::Error),
    #[error("managed-agent snapshot is corrupt")]
    CorruptSnapshot,
    #[error("managed-agent public projection is corrupt")]
    CorruptProjection,
    #[error("managed-agent transition is invalid")]
    InvalidTransition,
    #[error("managed-agent public projection is invalid")]
    InvalidProjection,
    #[error("managed-agent projection owner does not match")]
    ProjectionOwnerMismatch,
    #[error("managed-agent is deleted")]
    DeletedRecord,
}

impl From<anyhow::Error> for ManagedAgentRepositoryError {
    fn from(error: anyhow::Error) -> Self {
        Self::Unavailable(error)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedManagedAgentSnapshot {
    schema_version: u32,
    owner_public_key: String,
    agent_public_key: String,
    generation: u64,
    event_id: String,
    previous_event_id: Option<String>,
    state: PersistedManagedAgentState,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum PersistedManagedAgentState {
    Active {
        runtime: String,
        provider: Option<String>,
        model: Option<String>,
        environment: Vec<PersistedEnvironmentBinding>,
    },
    Deleted {
        deleted_at: u64,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedEnvironmentBinding {
    name: String,
    reference: PersistedEnvironmentReference,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
enum PersistedEnvironmentReference {
    ProcessEnvironment { variable: String },
    ProtectedCredential { reference: String },
}

pub(crate) fn encode_snapshot(
    record: &PrivateManagedAgentRecord,
) -> Result<String, ManagedAgentRepositoryError> {
    let state = match record.state() {
        ManagedAgentState::Active(configuration) => PersistedManagedAgentState::Active {
            runtime: configuration.runtime().as_str().to_string(),
            provider: configuration
                .provider()
                .map(|provider| provider.as_str().to_string()),
            model: configuration
                .model()
                .map(|model| model.as_str().to_string()),
            environment: configuration
                .environment()
                .iter()
                .map(|(name, reference)| PersistedEnvironmentBinding {
                    name: name.as_str().to_string(),
                    reference: match reference {
                        EnvironmentReference::ProcessEnvironment(variable) => {
                            PersistedEnvironmentReference::ProcessEnvironment {
                                variable: variable.as_str().to_string(),
                            }
                        }
                        EnvironmentReference::ProtectedCredential(reference) => {
                            PersistedEnvironmentReference::ProtectedCredential {
                                reference: reference.as_str().to_string(),
                            }
                        }
                    },
                })
                .collect(),
        },
        ManagedAgentState::Deleted { deleted_at } => PersistedManagedAgentState::Deleted {
            deleted_at: *deleted_at,
        },
    };
    serde_json::to_string(&PersistedManagedAgentSnapshot {
        schema_version: MANAGED_AGENT_SNAPSHOT_VERSION,
        owner_public_key: record.owner_public_key().as_str().to_string(),
        agent_public_key: record.agent_public_key().as_str().to_string(),
        generation: record.version().generation(),
        event_id: record.version().event_id().as_str().to_string(),
        previous_event_id: record
            .previous_event_id()
            .map(|event_id| event_id.as_str().to_string()),
        state,
    })
    .map_err(|error| ManagedAgentRepositoryError::Unavailable(error.into()))
}

pub(crate) fn decode_snapshot(
    snapshot: &str,
) -> Result<PrivateManagedAgentRecord, ManagedAgentRepositoryError> {
    let snapshot: PersistedManagedAgentSnapshot =
        serde_json::from_str(snapshot).map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?;
    if snapshot.schema_version != MANAGED_AGENT_SNAPSHOT_VERSION {
        return Err(ManagedAgentRepositoryError::CorruptSnapshot);
    }
    let owner_public_key = SettingsPublicKey::parse(snapshot.owner_public_key)
        .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?;
    let agent_public_key = SettingsPublicKey::parse(snapshot.agent_public_key)
        .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?;
    let event_id = SettingsEventId::parse(snapshot.event_id)
        .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?;
    let version = ManagedAgentVersion::new(snapshot.generation, event_id)
        .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?;
    let previous_event_id = snapshot
        .previous_event_id
        .map(SettingsEventId::parse)
        .transpose()
        .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?;
    let state = match snapshot.state {
        PersistedManagedAgentState::Active {
            runtime,
            provider,
            model,
            environment,
        } => {
            let mut environment_map = BTreeMap::new();
            for binding in environment {
                let name = EnvironmentVariableName::parse(binding.name)
                    .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?;
                let reference = match binding.reference {
                    PersistedEnvironmentReference::ProcessEnvironment { variable } => {
                        EnvironmentReference::ProcessEnvironment(
                            EnvironmentVariableName::parse(variable)
                                .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?,
                        )
                    }
                    PersistedEnvironmentReference::ProtectedCredential { reference } => {
                        EnvironmentReference::ProtectedCredential(
                            ProtectedCredentialReference::parse(reference)
                                .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?,
                        )
                    }
                };
                if environment_map.insert(name, reference).is_some() {
                    return Err(ManagedAgentRepositoryError::CorruptSnapshot);
                }
            }
            ManagedAgentState::Active(
                ManagedAgentConfiguration::new(
                    RuntimeId::parse(runtime)
                        .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?,
                    provider
                        .map(ProviderId::parse)
                        .transpose()
                        .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?,
                    model
                        .map(ModelId::parse)
                        .transpose()
                        .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?,
                    environment_map,
                )
                .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?,
            )
        }
        PersistedManagedAgentState::Deleted { deleted_at } => {
            ManagedAgentState::Deleted { deleted_at }
        }
    };
    PrivateManagedAgentRecord::hydrate(
        owner_public_key,
        agent_public_key,
        version,
        previous_event_id,
        state,
    )
    .map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)
}

fn validate_transition(
    expected_version: &ManagedAgentVersion,
    next_record: &PrivateManagedAgentRecord,
) -> Result<(), ManagedAgentRepositoryError> {
    let expected_next_generation = expected_version
        .generation()
        .checked_add(1)
        .filter(|generation| *generation <= MAX_SAFE_GENERATION)
        .ok_or(ManagedAgentRepositoryError::InvalidTransition)?;
    if next_record.version().generation() != expected_next_generation
        || next_record.previous_event_id() != Some(expected_version.event_id())
        || next_record.version().event_id() == expected_version.event_id()
    {
        return Err(ManagedAgentRepositoryError::InvalidTransition);
    }
    Ok(())
}

fn changed_rows(connection: &db::sqlez::connection::Connection) -> anyhow::Result<i64> {
    connection.select_row_bound::<(), i64>("SELECT changes()")?(())?
        .ok_or_else(|| anyhow::anyhow!("SQLite did not report a changed-row count"))
}

fn to_i64(value: u64) -> Result<i64, ManagedAgentRepositoryError> {
    i64::try_from(value).map_err(|_| ManagedAgentRepositoryError::InvalidTransition)
}

fn from_i64(value: i64) -> Result<u64, ManagedAgentRepositoryError> {
    let value = u64::try_from(value).map_err(|_| ManagedAgentRepositoryError::CorruptSnapshot)?;
    if value == 0 || value > MAX_SAFE_GENERATION {
        return Err(ManagedAgentRepositoryError::CorruptSnapshot);
    }
    Ok(value)
}

fn validate_public_projection_value(value: &Value) -> Result<(), ManagedAgentRepositoryError> {
    const PRIVATE_KEYS: &[&str] = &[
        "managed_agents",
        "environment",
        "environment_references",
        "credential_references",
        "local_source_path",
        "backend_reference",
        "respond_to_allowlist",
        "generation",
        "current_event_id",
    ];
    match value {
        Value::Object(object) => {
            if object
                .keys()
                .any(|key| PRIVATE_KEYS.contains(&key.as_str()))
            {
                return Err(ManagedAgentRepositoryError::InvalidProjection);
            }
            for value in object.values() {
                validate_public_projection_value(value)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                validate_public_projection_value(value)?;
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
    Ok(())
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}
