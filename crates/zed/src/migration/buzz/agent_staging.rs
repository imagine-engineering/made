use std::collections::{BTreeMap, HashSet};

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const AGENT_STAGING_FORMAT_VERSION: u32 = 1;
const MAX_RECORDS: usize = 10_000;
const MAX_SOURCE_BYTES: usize = 10 * 1024 * 1024;
const MAX_IDENTIFIER_BYTES: usize = 512;
const AGENT_SNAPSHOT_FORMAT: &str = "buzz-agent-snapshot";
const LOCKED_AGENT_SNAPSHOT_FORMAT: &str = "buzz-agent-snapshot-encrypted";
const TEAM_SNAPSHOT_FORMAT: &str = "buzz-team-snapshot";
const LOCKED_SNAPSHOT_SCHEME: &str = "nip44-v2";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzAgentJsonKind {
    ManagedAgent,
    Persona,
    Team,
    AgentSnapshot,
    EncryptedAgentSnapshot,
    TeamSnapshot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzAgentPrivacyClass {
    ProtectedIdentity,
    PrivateDefinition,
    PrivateMemory,
    OwnerEncrypted,
    PrivateTelemetry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzSecretBinding {
    pub json_pointer: String,
    pub protected_credential_id: String,
}

impl BuzzSecretBinding {
    pub fn new(
        json_pointer: impl Into<String>,
        protected_credential_id: impl Into<String>,
    ) -> Result<Self, BuzzAgentStagingError> {
        let json_pointer = json_pointer.into();
        let protected_credential_id = protected_credential_id.into();
        if !valid_json_pointer(&json_pointer) || !valid_identifier(&protected_credential_id) {
            return Err(BuzzAgentStagingError::InvalidSourceRecord);
        }
        Ok(Self {
            json_pointer,
            protected_credential_id,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzProtectedSecretReference {
    pub json_pointer: String,
    pub protected_credential_id: String,
    pub source_value_hash: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzAgentJsonStagingRecord {
    pub owner_profile_id: String,
    pub source_sequence: u64,
    pub source_path: String,
    pub semantic_id: String,
    pub kind: BuzzAgentJsonKind,
    pub source_schema_version: u32,
    pub privacy_class: BuzzAgentPrivacyClass,
    pub sanitized_payload: Value,
    pub protected_secrets: Vec<BuzzProtectedSecretReference>,
    pub source_hash: [u8; 32],
    pub privacy_hash: [u8; 32],
    pub idempotency_key: [u8; 32],
}

pub struct BuzzAgentJsonSource<'a> {
    pub owner_profile_id: &'a str,
    pub source_sequence: u64,
    pub source_path: &'a str,
    pub semantic_id: &'a str,
    pub kind: BuzzAgentJsonKind,
    pub source_schema_version: u32,
    pub source_bytes: &'a [u8],
}

impl BuzzAgentJsonStagingRecord {
    pub fn from_source(
        source: BuzzAgentJsonSource<'_>,
        secret_bindings: Vec<BuzzSecretBinding>,
    ) -> Result<Self, BuzzAgentStagingError> {
        if source.source_sequence == 0
            || !valid_identifier(source.owner_profile_id)
            || !valid_source_path(source.source_path)
            || !valid_identifier(source.semantic_id)
            || source.source_bytes.is_empty()
            || source.source_bytes.len() > MAX_SOURCE_BYTES
        {
            return Err(BuzzAgentStagingError::InvalidSourceRecord);
        }
        let mut payload: Value = serde_json::from_slice(source.source_bytes)
            .map_err(BuzzAgentStagingError::InvalidJson)?;
        validate_kind_payload(
            source.kind,
            source.source_schema_version,
            source.semantic_id,
            &payload,
        )?;
        let source_hash = sha256(source.source_bytes);
        let protected_secrets = protect_secrets(&mut payload, secret_bindings)?;
        reject_unprotected_secrets(&payload, "")?;
        let privacy_class = privacy_class(source.kind, &payload);
        let sanitized_bytes = serde_json::to_vec(&payload).map_err(BuzzAgentStagingError::Json)?;
        let privacy_hash = hash_parts(&[
            privacy_label(privacy_class),
            &source.source_schema_version.to_be_bytes(),
            &sanitized_bytes,
            &secret_reference_bytes(&protected_secrets)?,
        ]);
        let idempotency_key = hash_parts(&[
            kind_label(source.kind),
            source.source_path.as_bytes(),
            source.semantic_id.as_bytes(),
            &source_hash,
            &privacy_hash,
        ]);
        Ok(Self {
            owner_profile_id: source.owner_profile_id.to_owned(),
            source_sequence: source.source_sequence,
            source_path: source.source_path.to_owned(),
            semantic_id: source.semantic_id.to_owned(),
            kind: source.kind,
            source_schema_version: source.source_schema_version,
            privacy_class,
            sanitized_payload: payload,
            protected_secrets,
            source_hash,
            privacy_hash,
            idempotency_key,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzArchivedAgentEvidenceRecord {
    pub owner_profile_id: String,
    pub source_sequence: u64,
    pub source_path: String,
    pub event_id: [u8; 32],
    pub event_kind: u32,
    pub identity_public_key: [u8; 32],
    pub agent_public_key: [u8; 32],
    pub relay_url: String,
    pub archived_at_seconds: i64,
    pub archived_payload_hash: [u8; 32],
    pub privacy_class: BuzzAgentPrivacyClass,
    pub idempotency_key: [u8; 32],
}

pub struct BuzzArchivedAgentEvidenceSource<'a> {
    pub owner_profile_id: &'a str,
    pub source_sequence: u64,
    pub source_path: &'a str,
    pub event_id: [u8; 32],
    pub event_kind: u32,
    pub identity_public_key: [u8; 32],
    pub agent_public_key: [u8; 32],
    pub relay_url: &'a str,
    pub archived_at_seconds: i64,
    pub archived_payload_hash: [u8; 32],
}

impl BuzzArchivedAgentEvidenceRecord {
    pub fn new(source: BuzzArchivedAgentEvidenceSource<'_>) -> Result<Self, BuzzAgentStagingError> {
        if source.source_sequence == 0
            || !valid_identifier(source.owner_profile_id)
            || !valid_source_path(source.source_path)
            || !matches!(source.event_kind, 24_200 | 44_200)
            || !valid_identifier(source.relay_url)
            || source.archived_at_seconds <= 0
        {
            return Err(BuzzAgentStagingError::InvalidSourceRecord);
        }
        let idempotency_key = hash_parts(&[
            b"archive_agent_evidence",
            source.source_path.as_bytes(),
            &source.event_id,
            &source.event_kind.to_be_bytes(),
            &source.archived_payload_hash,
        ]);
        Ok(Self {
            owner_profile_id: source.owner_profile_id.to_owned(),
            source_sequence: source.source_sequence,
            source_path: source.source_path.to_owned(),
            event_id: source.event_id,
            event_kind: source.event_kind,
            identity_public_key: source.identity_public_key,
            agent_public_key: source.agent_public_key,
            relay_url: source.relay_url.to_owned(),
            archived_at_seconds: source.archived_at_seconds,
            archived_payload_hash: source.archived_payload_hash,
            privacy_class: BuzzAgentPrivacyClass::PrivateTelemetry,
            idempotency_key,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzAgentStagingRecord {
    Json(BuzzAgentJsonStagingRecord),
    ArchivedEvidence(BuzzArchivedAgentEvidenceRecord),
}

impl BuzzAgentStagingRecord {
    fn owner_profile_id(&self) -> &str {
        match self {
            Self::Json(record) => &record.owner_profile_id,
            Self::ArchivedEvidence(record) => &record.owner_profile_id,
        }
    }

    fn source_sequence(&self) -> u64 {
        match self {
            Self::Json(record) => record.source_sequence,
            Self::ArchivedEvidence(record) => record.source_sequence,
        }
    }

    fn source_path(&self) -> &str {
        match self {
            Self::Json(record) => &record.source_path,
            Self::ArchivedEvidence(record) => &record.source_path,
        }
    }

    fn idempotency_key(&self) -> [u8; 32] {
        match self {
            Self::Json(record) => record.idempotency_key,
            Self::ArchivedEvidence(record) => record.idempotency_key,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzAgentStagingBatch {
    format_version: u32,
    owner_profile_id: String,
    records: Vec<BuzzAgentStagingRecord>,
}

impl BuzzAgentStagingBatch {
    pub fn new(
        format_version: u32,
        owner_profile_id: impl Into<String>,
        records: Vec<BuzzAgentStagingRecord>,
    ) -> Result<Self, BuzzAgentStagingError> {
        let owner_profile_id = owner_profile_id.into();
        if format_version != AGENT_STAGING_FORMAT_VERSION {
            return Err(BuzzAgentStagingError::UnsupportedFormatVersion(
                format_version,
            ));
        }
        if !valid_identifier(&owner_profile_id) || records.is_empty() || records.len() > MAX_RECORDS
        {
            return Err(BuzzAgentStagingError::InvalidBatch);
        }
        let mut previous_sequence = 0;
        let mut source_paths = HashSet::new();
        let mut idempotency_keys = HashSet::new();
        for record in &records {
            if record.owner_profile_id() != owner_profile_id
                || record.source_sequence() <= previous_sequence
                || !source_paths.insert(record.source_path().to_owned())
                || !idempotency_keys.insert(record.idempotency_key())
            {
                return Err(BuzzAgentStagingError::InvalidBatch);
            }
            previous_sequence = record.source_sequence();
        }
        Ok(Self {
            format_version,
            owner_profile_id,
            records,
        })
    }

    pub fn records(&self) -> &[BuzzAgentStagingRecord] {
        &self.records
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzAgentActivationState {
    pub agent_execution_enabled: bool,
    pub automatic_start_enabled: bool,
    pub credential_use_enabled: bool,
}

impl BuzzAgentActivationState {
    pub const fn all_disabled() -> Self {
        Self {
            agent_execution_enabled: false,
            automatic_start_enabled: false,
            credential_use_enabled: false,
        }
    }

    pub const fn can_activate(self) -> bool {
        self.agent_execution_enabled || self.automatic_start_enabled || self.credential_use_enabled
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzAgentStagingCheckpoint {
    pub final_source_sequence: u64,
    pub source_hash: [u8; 32],
    pub staged_hash: [u8; 32],
    pub privacy_hash: [u8; 32],
    pub staged: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzAgentStagingPlan {
    pub records: Vec<BuzzAgentStagingRecord>,
    pub activation: BuzzAgentActivationState,
    pub checkpoint: BuzzAgentStagingCheckpoint,
}

#[derive(Debug, thiserror::Error)]
pub enum BuzzAgentStagingError {
    #[error("Buzz agent staging format version {0} is unsupported")]
    UnsupportedFormatVersion(u32),
    #[error("Buzz agent staging source record is invalid")]
    InvalidSourceRecord,
    #[error("Buzz agent staging batch is invalid, duplicated or out of order")]
    InvalidBatch,
    #[error("Buzz agent staging JSON is invalid")]
    InvalidJson(#[source] serde_json::Error),
    #[error("Buzz agent staging JSON operation failed")]
    Json(#[source] serde_json::Error),
    #[error("Buzz agent record format or schema version is unsupported")]
    UnsupportedSourceFormat,
    #[error("Buzz agent record semantic identity does not match its payload")]
    SemanticIdentityMismatch,
    #[error("Buzz agent record contains a secret without a protected credential binding")]
    UnprotectedSecret,
    #[error("Buzz agent record secret binding does not resolve to a scalar source value")]
    InvalidSecretBinding,
}

pub struct BuzzAgentStagingImporter;

impl BuzzAgentStagingImporter {
    pub fn stage(
        expected_owner_profile_id: &str,
        batch: &BuzzAgentStagingBatch,
    ) -> Result<BuzzAgentStagingPlan, BuzzAgentStagingError> {
        if batch.owner_profile_id != expected_owner_profile_id {
            return Err(BuzzAgentStagingError::InvalidBatch);
        }
        let mut source_hasher = Sha256::new();
        let mut staged_hasher = Sha256::new();
        let mut privacy_hasher = Sha256::new();
        hash_part(&mut source_hasher, &batch.format_version.to_be_bytes());
        hash_part(&mut staged_hasher, &batch.format_version.to_be_bytes());
        for record in &batch.records {
            match record {
                BuzzAgentStagingRecord::Json(record) => {
                    hash_part(&mut source_hasher, &record.source_hash);
                    hash_part(&mut privacy_hasher, &record.privacy_hash);
                }
                BuzzAgentStagingRecord::ArchivedEvidence(record) => {
                    hash_part(&mut source_hasher, &record.archived_payload_hash);
                    hash_part(&mut privacy_hasher, privacy_label(record.privacy_class));
                }
            }
            let staged = serde_json::to_vec(record).map_err(BuzzAgentStagingError::Json)?;
            hash_part(&mut staged_hasher, &staged);
        }
        let staged =
            u64::try_from(batch.records.len()).map_err(|_| BuzzAgentStagingError::InvalidBatch)?;
        Ok(BuzzAgentStagingPlan {
            records: batch.records.clone(),
            activation: BuzzAgentActivationState::all_disabled(),
            checkpoint: BuzzAgentStagingCheckpoint {
                final_source_sequence: batch
                    .records
                    .last()
                    .map(BuzzAgentStagingRecord::source_sequence)
                    .ok_or(BuzzAgentStagingError::InvalidBatch)?,
                source_hash: source_hasher.finalize().into(),
                staged_hash: staged_hasher.finalize().into(),
                privacy_hash: privacy_hasher.finalize().into(),
                staged,
            },
        })
    }
}

fn validate_kind_payload(
    kind: BuzzAgentJsonKind,
    source_schema_version: u32,
    semantic_id: &str,
    payload: &Value,
) -> Result<(), BuzzAgentStagingError> {
    let object = payload
        .as_object()
        .ok_or(BuzzAgentStagingError::InvalidSourceRecord)?;
    match kind {
        BuzzAgentJsonKind::ManagedAgent => {
            require_staging_version(source_schema_version)?;
            let public_key = object.get("pubkey").and_then(Value::as_str);
            if public_key.is_some_and(|value| !value.is_empty() && !canonical_public_key(value)) {
                return Err(BuzzAgentStagingError::InvalidSourceRecord);
            }
            let payload_id = public_key
                .filter(|value| !value.is_empty())
                .or_else(|| object.get("slug").and_then(Value::as_str));
            if payload_id != Some(semantic_id) {
                return Err(BuzzAgentStagingError::SemanticIdentityMismatch);
            }
        }
        BuzzAgentJsonKind::Persona | BuzzAgentJsonKind::Team => {
            require_staging_version(source_schema_version)?;
            if object.get("id").and_then(Value::as_str) != Some(semantic_id) {
                return Err(BuzzAgentStagingError::SemanticIdentityMismatch);
            }
        }
        BuzzAgentJsonKind::AgentSnapshot => {
            require_wire_format(object, AGENT_SNAPSHOT_FORMAT, source_schema_version)?;
            require_nonempty_pointer(payload, "/definition/name")?;
            require_nonempty_pointer(payload, "/profile/displayName")?;
            validate_snapshot_memory(payload)?;
        }
        BuzzAgentJsonKind::EncryptedAgentSnapshot => {
            require_wire_format(object, LOCKED_AGENT_SNAPSHOT_FORMAT, source_schema_version)?;
            let owner_public_key = payload
                .pointer("/encryption/ownerPubkey")
                .and_then(Value::as_str);
            let agent_public_key = payload
                .pointer("/encryption/agentPubkey")
                .and_then(Value::as_str);
            if payload
                .pointer("/encryption/scheme")
                .and_then(Value::as_str)
                != Some(LOCKED_SNAPSHOT_SCHEME)
                || payload
                    .pointer("/encryption/ciphertext")
                    .and_then(Value::as_str)
                    .is_none_or(|value| value.is_empty() || value.len() > 90_000)
                || owner_public_key.is_none_or(|value| !canonical_public_key(value))
                || agent_public_key.is_none_or(|value| !canonical_public_key(value))
                || owner_public_key == agent_public_key
            {
                return Err(BuzzAgentStagingError::UnsupportedSourceFormat);
            }
        }
        BuzzAgentJsonKind::TeamSnapshot => {
            require_wire_format(object, TEAM_SNAPSHOT_FORMAT, source_schema_version)?;
            require_nonempty_pointer(payload, "/team/name")?;
            let members = payload
                .get("members")
                .and_then(Value::as_array)
                .filter(|members| !members.is_empty())
                .ok_or(BuzzAgentStagingError::InvalidSourceRecord)?;
            for member in members {
                let member_object = member
                    .as_object()
                    .ok_or(BuzzAgentStagingError::InvalidSourceRecord)?;
                require_wire_format(member_object, AGENT_SNAPSHOT_FORMAT, 1)?;
                require_nonempty_pointer(member, "/definition/name")?;
                require_nonempty_pointer(member, "/profile/displayName")?;
                validate_snapshot_memory(member)?;
            }
            if members.len() > MAX_RECORDS {
                return Err(BuzzAgentStagingError::InvalidSourceRecord);
            }
        }
    }
    Ok(())
}

fn require_staging_version(source_schema_version: u32) -> Result<(), BuzzAgentStagingError> {
    if source_schema_version != 1 {
        return Err(BuzzAgentStagingError::UnsupportedSourceFormat);
    }
    Ok(())
}

fn require_wire_format(
    object: &Map<String, Value>,
    expected_format: &str,
    source_schema_version: u32,
) -> Result<(), BuzzAgentStagingError> {
    require_staging_version(source_schema_version)?;
    if object.get("format").and_then(Value::as_str) != Some(expected_format)
        || object.get("version").and_then(Value::as_u64) != Some(1)
    {
        return Err(BuzzAgentStagingError::UnsupportedSourceFormat);
    }
    Ok(())
}

fn require_nonempty_pointer(payload: &Value, pointer: &str) -> Result<(), BuzzAgentStagingError> {
    if payload
        .pointer(pointer)
        .and_then(Value::as_str)
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err(BuzzAgentStagingError::InvalidSourceRecord);
    }
    Ok(())
}

fn validate_snapshot_memory(payload: &Value) -> Result<(), BuzzAgentStagingError> {
    let level = payload.pointer("/memory/level").and_then(Value::as_str);
    let entries = payload.pointer("/memory/entries").and_then(Value::as_array);
    if !matches!(level, Some("none" | "core" | "everything"))
        || entries.is_none()
        || level == Some("none") && entries.is_some_and(|entries| !entries.is_empty())
    {
        return Err(BuzzAgentStagingError::InvalidSourceRecord);
    }
    Ok(())
}

fn protect_secrets(
    payload: &mut Value,
    bindings: Vec<BuzzSecretBinding>,
) -> Result<Vec<BuzzProtectedSecretReference>, BuzzAgentStagingError> {
    let mut seen = HashSet::new();
    let mut protected = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if !seen.insert(binding.json_pointer.clone()) {
            return Err(BuzzAgentStagingError::InvalidSecretBinding);
        }
        let value = payload
            .pointer_mut(&binding.json_pointer)
            .ok_or(BuzzAgentStagingError::InvalidSecretBinding)?;
        if value.is_null() || value.is_array() || value.is_object() {
            return Err(BuzzAgentStagingError::InvalidSecretBinding);
        }
        let source_value = serde_json::to_vec(value).map_err(BuzzAgentStagingError::Json)?;
        let source_value_hash = sha256(&source_value);
        *value = Value::Object(Map::from_iter([(
            "$protectedCredential".to_owned(),
            Value::String(binding.protected_credential_id.clone()),
        )]));
        protected.push(BuzzProtectedSecretReference {
            json_pointer: binding.json_pointer,
            protected_credential_id: binding.protected_credential_id,
            source_value_hash,
        });
    }
    protected.sort_by(|left, right| left.json_pointer.cmp(&right.json_pointer));
    Ok(protected)
}

fn reject_unprotected_secrets(value: &Value, key: &str) -> Result<(), BuzzAgentStagingError> {
    match value {
        Value::Object(object) => {
            if object.len() == 1 && object.contains_key("$protectedCredential") {
                return Ok(());
            }
            for (child_key, child_value) in object {
                let is_protected_reference = child_value.as_object().is_some_and(|object| {
                    object.len() == 1 && object.contains_key("$protectedCredential")
                });
                if secret_key(child_key)
                    && !is_protected_reference
                    && !child_value.is_null()
                    && child_value != ""
                {
                    return Err(BuzzAgentStagingError::UnprotectedSecret);
                }
                reject_unprotected_secrets(child_value, child_key)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                reject_unprotected_secrets(child, key)?;
            }
        }
        Value::String(string) if key == "private_key_nsec" || string.starts_with("nsec1") => {
            return Err(BuzzAgentStagingError::UnprotectedSecret);
        }
        _ => {}
    }
    Ok(())
}

fn secret_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "private_key_nsec"
            | "nsec"
            | "api_key"
            | "access_token"
            | "refresh_token"
            | "password"
            | "secret"
    ) || normalized.ends_with("_api_key")
        || normalized.ends_with("_token")
        || normalized.ends_with("_secret")
        || normalized.ends_with("_password")
}

fn privacy_class(kind: BuzzAgentJsonKind, payload: &Value) -> BuzzAgentPrivacyClass {
    match kind {
        BuzzAgentJsonKind::ManagedAgent => BuzzAgentPrivacyClass::ProtectedIdentity,
        BuzzAgentJsonKind::Persona | BuzzAgentJsonKind::Team => {
            BuzzAgentPrivacyClass::PrivateDefinition
        }
        BuzzAgentJsonKind::EncryptedAgentSnapshot => BuzzAgentPrivacyClass::OwnerEncrypted,
        BuzzAgentJsonKind::AgentSnapshot | BuzzAgentJsonKind::TeamSnapshot => {
            if snapshot_contains_memory(payload) {
                BuzzAgentPrivacyClass::PrivateMemory
            } else {
                BuzzAgentPrivacyClass::PrivateDefinition
            }
        }
    }
}

fn snapshot_contains_memory(payload: &Value) -> bool {
    payload
        .pointer("/memory/entries")
        .and_then(Value::as_array)
        .is_some_and(|entries| !entries.is_empty())
        || payload
            .get("members")
            .and_then(Value::as_array)
            .is_some_and(|members| {
                members.iter().any(|member| {
                    member
                        .pointer("/memory/entries")
                        .and_then(Value::as_array)
                        .is_some_and(|entries| !entries.is_empty())
                })
            })
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTIFIER_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn canonical_public_key(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_source_path(value: &str) -> bool {
    valid_identifier(value)
        && !value.starts_with('/')
        && !value.split('/').any(|component| component == "..")
}

fn valid_json_pointer(value: &str) -> bool {
    value.starts_with('/') && value.len() <= MAX_IDENTIFIER_BYTES && !value.contains("//")
}

fn privacy_label(privacy: BuzzAgentPrivacyClass) -> &'static [u8] {
    match privacy {
        BuzzAgentPrivacyClass::ProtectedIdentity => b"protected_identity",
        BuzzAgentPrivacyClass::PrivateDefinition => b"private_definition",
        BuzzAgentPrivacyClass::PrivateMemory => b"private_memory",
        BuzzAgentPrivacyClass::OwnerEncrypted => b"owner_encrypted",
        BuzzAgentPrivacyClass::PrivateTelemetry => b"private_telemetry",
    }
}

fn kind_label(kind: BuzzAgentJsonKind) -> &'static [u8] {
    match kind {
        BuzzAgentJsonKind::ManagedAgent => b"managed_agent",
        BuzzAgentJsonKind::Persona => b"persona",
        BuzzAgentJsonKind::Team => b"team",
        BuzzAgentJsonKind::AgentSnapshot => b"agent_snapshot",
        BuzzAgentJsonKind::EncryptedAgentSnapshot => b"encrypted_agent_snapshot",
        BuzzAgentJsonKind::TeamSnapshot => b"team_snapshot",
    }
}

fn secret_reference_bytes(
    references: &[BuzzProtectedSecretReference],
) -> Result<Vec<u8>, BuzzAgentStagingError> {
    let ordered: BTreeMap<_, _> = references
        .iter()
        .map(|reference| {
            (
                reference.json_pointer.as_str(),
                (
                    reference.protected_credential_id.as_str(),
                    reference.source_value_hash,
                ),
            )
        })
        .collect();
    serde_json::to_vec(&ordered).map_err(BuzzAgentStagingError::Json)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
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
    use serde_json::json;

    use super::*;

    fn json_record(
        source_sequence: u64,
        source_path: &str,
        semantic_id: &str,
        kind: BuzzAgentJsonKind,
        payload: Value,
        secret_bindings: Vec<BuzzSecretBinding>,
    ) -> BuzzAgentStagingRecord {
        let bytes = serde_json::to_vec(&payload).expect("fixture JSON");
        BuzzAgentStagingRecord::Json(
            BuzzAgentJsonStagingRecord::from_source(
                BuzzAgentJsonSource {
                    owner_profile_id: "profile-1",
                    source_sequence,
                    source_path,
                    semantic_id,
                    kind,
                    source_schema_version: 1,
                    source_bytes: &bytes,
                },
                secret_bindings,
            )
            .expect("staging record"),
        )
    }

    fn fixture_records() -> Vec<BuzzAgentStagingRecord> {
        let agent_public_key = "11".repeat(32);
        let owner_public_key = "22".repeat(32);
        vec![
            json_record(
                1,
                "agents/managed-agents.json#agent-a",
                &agent_public_key,
                BuzzAgentJsonKind::ManagedAgent,
                json!({
                    "pubkey": agent_public_key,
                    "name": "Builder",
                    "private_key_nsec": "nsec1private",
                    "env_vars": {"OPENAI_API_KEY": "private-provider-key"}
                }),
                vec![
                    BuzzSecretBinding::new("/private_key_nsec", "buzz-agent-signing-key:agent-a")
                        .expect("binding"),
                    BuzzSecretBinding::new(
                        "/env_vars/OPENAI_API_KEY",
                        "buzz-agent-env:agent-a:openai",
                    )
                    .expect("binding"),
                ],
            ),
            json_record(
                2,
                "agents/managed-agents.json#persona-a",
                "persona-a",
                BuzzAgentJsonKind::Persona,
                json!({"id": "persona-a", "display_name": "Reviewer", "system_prompt": "Review changes"}),
                vec![],
            ),
            json_record(
                3,
                "agents/teams.json#team-a",
                "team-a",
                BuzzAgentJsonKind::Team,
                json!({"id": "team-a", "name": "Core", "persona_ids": ["persona-a"]}),
                vec![],
            ),
            json_record(
                4,
                "agents/snapshots/agent-a.agent.json",
                "agent-a-snapshot",
                BuzzAgentJsonKind::AgentSnapshot,
                json!({
                    "format": "buzz-agent-snapshot",
                    "version": 1,
                    "definition": {"name": "Builder"},
                    "profile": {"displayName": "Builder"},
                    "memory": {"level": "core", "entries": [{"slug": "role", "body": "Build safely"}]}
                }),
                vec![],
            ),
            json_record(
                5,
                "agents/snapshots/agent-a.locked.json",
                "agent-a-locked-snapshot",
                BuzzAgentJsonKind::EncryptedAgentSnapshot,
                json!({
                    "format": "buzz-agent-snapshot-encrypted",
                    "version": 1,
                    "encryption": {"scheme": "nip44-v2", "ownerPubkey": owner_public_key, "agentPubkey": "11".repeat(32), "ciphertext": "ciphertext"}
                }),
                vec![],
            ),
            json_record(
                6,
                "agents/snapshots/team-a.team.json",
                "team-a-snapshot",
                BuzzAgentJsonKind::TeamSnapshot,
                json!({
                    "format": "buzz-team-snapshot",
                    "version": 1,
                    "team": {"name": "Core"},
                    "members": [{
                        "format": "buzz-agent-snapshot",
                        "version": 1,
                        "definition": {"name": "Builder"},
                        "profile": {"displayName": "Builder"},
                        "memory": {"level": "none", "entries": []}
                    }]
                }),
                vec![],
            ),
            BuzzAgentStagingRecord::ArchivedEvidence(
                BuzzArchivedAgentEvidenceRecord::new(BuzzArchivedAgentEvidenceSource {
                    owner_profile_id: "profile-1",
                    source_sequence: 7,
                    source_path: "archive/archive.db#event-1",
                    event_id: [1; 32],
                    event_kind: 44_200,
                    identity_public_key: [2; 32],
                    agent_public_key: [3; 32],
                    relay_url: "wss://relay.example",
                    archived_at_seconds: 1_700_000_000,
                    archived_payload_hash: [4; 32],
                })
                .expect("archive evidence"),
            ),
        ]
    }

    #[test]
    fn stages_every_agent_fixture_idempotently_with_privacy_hashes() {
        let batch =
            BuzzAgentStagingBatch::new(1, "profile-1", fixture_records()).expect("staging batch");
        let first = BuzzAgentStagingImporter::stage("profile-1", &batch).expect("first stage");
        let replay = BuzzAgentStagingImporter::stage("profile-1", &batch).expect("replay stage");

        assert_eq!(first, replay);
        assert_eq!(first.records.len(), 7);
        assert_ne!(first.checkpoint.source_hash, [0; 32]);
        assert_ne!(first.checkpoint.staged_hash, [0; 32]);
        assert_ne!(first.checkpoint.privacy_hash, [0; 32]);
        assert!(!first.activation.can_activate());
    }

    #[test]
    fn redacts_source_secrets_without_mutating_source_bytes() {
        let agent_public_key = "11".repeat(32);
        let source = serde_json::to_vec(&json!({
            "pubkey": agent_public_key,
            "private_key_nsec": "nsec1private"
        }))
        .expect("source JSON");
        let original = source.clone();
        let record = BuzzAgentJsonStagingRecord::from_source(
            BuzzAgentJsonSource {
                owner_profile_id: "profile-1",
                source_sequence: 1,
                source_path: "agents/managed-agents.json#agent-a",
                semantic_id: &agent_public_key,
                kind: BuzzAgentJsonKind::ManagedAgent,
                source_schema_version: 1,
                source_bytes: &source,
            },
            vec![
                BuzzSecretBinding::new("/private_key_nsec", "buzz-agent-signing-key:agent-a")
                    .expect("binding"),
            ],
        )
        .expect("protected record");

        assert_eq!(source, original);
        assert_eq!(record.source_hash, sha256(&original));
        assert_eq!(record.protected_secrets.len(), 1);
        let staged = serde_json::to_string(&record.sanitized_payload).expect("staged JSON");
        assert!(!staged.contains("nsec1private"));
        assert!(staged.contains("buzz-agent-signing-key:agent-a"));
    }

    #[test]
    fn rejects_unprotected_secret_and_unknown_snapshot_version() {
        let agent_public_key = "11".repeat(32);
        let source = serde_json::to_vec(&json!({
            "pubkey": agent_public_key,
            "private_key_nsec": "nsec1private"
        }))
        .expect("source JSON");
        assert!(matches!(
            BuzzAgentJsonStagingRecord::from_source(
                BuzzAgentJsonSource {
                    owner_profile_id: "profile-1",
                    source_sequence: 1,
                    source_path: "agents/managed-agents.json#agent-a",
                    semantic_id: &agent_public_key,
                    kind: BuzzAgentJsonKind::ManagedAgent,
                    source_schema_version: 1,
                    source_bytes: &source,
                },
                vec![],
            ),
            Err(BuzzAgentStagingError::UnprotectedSecret)
        ));

        let snapshot = serde_json::to_vec(&json!({
            "format": "buzz-agent-snapshot",
            "version": 2,
            "definition": {"name": "Builder"},
            "profile": {"displayName": "Builder"},
            "memory": {"level": "none", "entries": []}
        }))
        .expect("snapshot JSON");
        assert!(matches!(
            BuzzAgentJsonStagingRecord::from_source(
                BuzzAgentJsonSource {
                    owner_profile_id: "profile-1",
                    source_sequence: 1,
                    source_path: "agents/snapshot.agent.json",
                    semantic_id: "snapshot",
                    kind: BuzzAgentJsonKind::AgentSnapshot,
                    source_schema_version: 1,
                    source_bytes: &snapshot,
                },
                vec![],
            ),
            Err(BuzzAgentStagingError::UnsupportedSourceFormat)
        ));
    }

    #[test]
    fn rejects_owner_scope_mismatch_before_staging() {
        let batch =
            BuzzAgentStagingBatch::new(1, "profile-1", fixture_records()).expect("staging batch");
        assert!(matches!(
            BuzzAgentStagingImporter::stage("profile-2", &batch),
            Err(BuzzAgentStagingError::InvalidBatch)
        ));
    }
}
