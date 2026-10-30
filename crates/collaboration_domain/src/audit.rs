use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};
use sha2::{Digest, Sha256};

use crate::{CommunityId, OperationId, PrincipalId};

const AUDIT_HASH_DOMAIN: &[u8] = b"zed.collaboration.audit.entry.v1";
const MAX_ACTION_BYTES: usize = 128;
const MAX_FIELD_NAME_BYTES: usize = 64;
const MAX_IDENTIFIER_BYTES: usize = 256;
const MAX_CATEGORY_BYTES: usize = 64;
const MAX_FIELDS: usize = 32;
const MAX_ENCODED_ENTRY_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AuditHash([u8; 32]);

impl AuditHash {
    pub const fn from_bytes(value: [u8; 32]) -> Self {
        Self(value)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for AuditHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AuditHash([32 bytes])")
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AuditAction(String);

impl AuditAction {
    pub fn new(value: impl Into<String>) -> Result<Self, AuditError> {
        let value = value.into();
        validate_category_token(&value, MAX_ACTION_BYTES)
            .then_some(Self(value))
            .ok_or(AuditError::InvalidAction)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AuditAction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct AuditFieldName(String);

impl AuditFieldName {
    pub fn new(value: impl Into<String>) -> Result<Self, AuditError> {
        let value = value.into();
        validate_category_token(&value, MAX_FIELD_NAME_BYTES)
            .then_some(Self(value))
            .ok_or(AuditError::InvalidFieldName)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn requires_redaction(&self) -> bool {
        self.0
            .split(['.', '_', '-'])
            .any(|segment| SENSITIVE_FIELD_SEGMENTS.contains(&segment))
    }
}

impl<'de> Deserialize<'de> for AuditFieldName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

const SENSITIVE_FIELD_SEGMENTS: &[&str] = &[
    "body",
    "ciphertext",
    "content",
    "credential",
    "detail",
    "key",
    "message",
    "mnemonic",
    "note",
    "password",
    "payload",
    "private",
    "prompt",
    "secret",
    "seed",
    "token",
];

#[derive(Clone, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AuditIdentifier(String);

impl AuditIdentifier {
    pub fn new(value: impl Into<String>) -> Result<Self, AuditError> {
        let value = value.into();
        let valid = !value.is_empty()
            && value.len() <= MAX_IDENTIFIER_BYTES
            && value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
            });
        valid
            .then_some(Self(value))
            .ok_or(AuditError::InvalidIdentifier)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AuditIdentifier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

impl fmt::Debug for AuditIdentifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AuditIdentifier(<redacted>)")
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AuditCategory(String);

impl AuditCategory {
    pub fn new(value: impl Into<String>) -> Result<Self, AuditError> {
        let value = value.into();
        validate_category_token(&value, MAX_CATEGORY_BYTES)
            .then_some(Self(value))
            .ok_or(AuditError::InvalidCategory)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AuditCategory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    Succeeded,
    Failed,
    Denied,
    Cancelled,
}

impl AuditOutcome {
    const fn code(self) -> u8 {
        match self {
            Self::Succeeded => 1,
            Self::Failed => 2,
            Self::Denied => 3,
            Self::Cancelled => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditRedaction {
    Credential,
    ErrorDetail,
    KeyMaterial,
    PersonalData,
    PrivateContent,
}

impl AuditRedaction {
    const fn code(self) -> u8 {
        match self {
            Self::Credential => 1,
            Self::ErrorDetail => 2,
            Self::KeyMaterial => 3,
            Self::PersonalData => 4,
            Self::PrivateContent => 5,
        }
    }
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum AuditValue {
    Identifier(AuditIdentifier),
    Category(AuditCategory),
    Unsigned(u64),
    Signed(i64),
    Boolean(bool),
    Redacted(AuditRedaction),
}

impl fmt::Debug for AuditValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identifier(_) => formatter.write_str("Identifier(<redacted>)"),
            Self::Category(value) => formatter.debug_tuple("Category").field(value).finish(),
            Self::Unsigned(value) => formatter.debug_tuple("Unsigned").field(value).finish(),
            Self::Signed(value) => formatter.debug_tuple("Signed").field(value).finish(),
            Self::Boolean(value) => formatter.debug_tuple("Boolean").field(value).finish(),
            Self::Redacted(value) => formatter.debug_tuple("Redacted").field(value).finish(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AuditField {
    name: AuditFieldName,
    value: AuditValue,
}

impl AuditField {
    pub fn new(name: AuditFieldName, value: AuditValue) -> Self {
        Self { name, value }
    }

    pub fn name(&self) -> &AuditFieldName {
        &self.name
    }

    pub fn value(&self) -> &AuditValue {
        &self.value
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AuditFields(Vec<AuditField>);

impl AuditFields {
    pub fn new(mut fields: Vec<AuditField>) -> Result<Self, AuditError> {
        if fields.len() > MAX_FIELDS {
            return Err(AuditError::TooManyFields);
        }
        fields.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        if fields.windows(2).any(|pair| pair[0].name == pair[1].name) {
            return Err(AuditError::DuplicateField);
        }
        if fields.iter().any(|field| {
            field.name.requires_redaction() && !matches!(field.value, AuditValue::Redacted(_))
        }) {
            return Err(AuditError::SensitiveFieldNotRedacted);
        }
        Ok(Self(fields))
    }

    pub fn as_slice(&self) -> &[AuditField] {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AuditFields {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let fields = Vec::<AuditField>::deserialize(deserializer)?;
        Self::new(fields).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditRecord {
    operation_id: OperationId,
    action: AuditAction,
    actor_principal_id: Option<PrincipalId>,
    outcome: AuditOutcome,
    occurred_at_millis: u64,
    fields: AuditFields,
}

impl AuditRecord {
    pub fn new(
        operation_id: OperationId,
        action: AuditAction,
        actor_principal_id: Option<PrincipalId>,
        outcome: AuditOutcome,
        occurred_at_millis: u64,
        fields: AuditFields,
    ) -> Result<Self, AuditError> {
        if operation_id.as_uuid().is_nil()
            || actor_principal_id.is_some_and(|actor| actor.as_uuid().is_nil())
            || occurred_at_millis == 0
        {
            return Err(AuditError::InvalidRecord);
        }
        Ok(Self {
            operation_id,
            action,
            actor_principal_id,
            outcome,
            occurred_at_millis,
            fields,
        })
    }

    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub fn action(&self) -> &AuditAction {
        &self.action
    }

    pub const fn actor_principal_id(&self) -> Option<PrincipalId> {
        self.actor_principal_id
    }

    pub const fn outcome(&self) -> AuditOutcome {
        self.outcome
    }

    pub const fn occurred_at_millis(&self) -> u64 {
        self.occurred_at_millis
    }

    pub fn fields(&self) -> &AuditFields {
        &self.fields
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditChainSource {
    BuzzV1,
}

impl AuditChainSource {
    const fn code(self) -> u8 {
        match self {
            Self::BuzzV1 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct AuditChainBridge {
    source: AuditChainSource,
    source_sequence: u64,
    source_head: AuditHash,
}

impl<'de> Deserialize<'de> for AuditChainBridge {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct AuditChainBridgeFields {
            source: AuditChainSource,
            source_sequence: u64,
            source_head: AuditHash,
        }

        let fields = AuditChainBridgeFields::deserialize(deserializer)?;
        Self::new(fields.source, fields.source_sequence, fields.source_head)
            .map_err(de::Error::custom)
    }
}

impl AuditChainBridge {
    pub fn new(
        source: AuditChainSource,
        source_sequence: u64,
        source_head: AuditHash,
    ) -> Result<Self, AuditError> {
        if source_sequence == 0 {
            return Err(AuditError::InvalidBridge);
        }
        Ok(Self {
            source,
            source_sequence,
            source_head,
        })
    }

    pub const fn source(self) -> AuditChainSource {
        self.source
    }

    pub const fn source_sequence(self) -> u64 {
        self.source_sequence
    }

    pub const fn source_head(self) -> AuditHash {
        self.source_head
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuditPredecessor {
    Genesis,
    Entry(AuditHash),
    Imported(AuditChainBridge),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuditChainPosition {
    community_id: CommunityId,
    next_sequence: u64,
    predecessor: AuditPredecessor,
}

impl AuditChainPosition {
    pub fn genesis(community_id: CommunityId) -> Result<Self, AuditError> {
        if community_id.as_uuid().is_nil() {
            return Err(AuditError::InvalidCommunity);
        }
        Ok(Self {
            community_id,
            next_sequence: 1,
            predecessor: AuditPredecessor::Genesis,
        })
    }

    pub fn from_imported(
        community_id: CommunityId,
        bridge: AuditChainBridge,
    ) -> Result<Self, AuditError> {
        if community_id.as_uuid().is_nil() {
            return Err(AuditError::InvalidCommunity);
        }
        let next_sequence = bridge
            .source_sequence
            .checked_add(1)
            .ok_or(AuditError::SequenceExhausted)?;
        Ok(Self {
            community_id,
            next_sequence,
            predecessor: AuditPredecessor::Imported(bridge),
        })
    }

    pub fn after(entry: &AuditEntry) -> Result<Self, AuditError> {
        entry.verify()?;
        Self::after_stored_head(entry.community_id, entry.sequence, entry.hash)
    }

    pub fn after_stored_head(
        community_id: CommunityId,
        sequence: u64,
        hash: AuditHash,
    ) -> Result<Self, AuditError> {
        if community_id.as_uuid().is_nil() || sequence == 0 {
            return Err(AuditError::InvalidCommunity);
        }
        let next_sequence = sequence
            .checked_add(1)
            .ok_or(AuditError::SequenceExhausted)?;
        Ok(Self {
            community_id,
            next_sequence,
            predecessor: AuditPredecessor::Entry(hash),
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn next_sequence(self) -> u64 {
        self.next_sequence
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditEntry {
    community_id: CommunityId,
    sequence: u64,
    record: AuditRecord,
    predecessor: AuditPredecessor,
    hash: AuditHash,
}

impl AuditEntry {
    pub fn append(position: AuditChainPosition, record: AuditRecord) -> Result<Self, AuditError> {
        let mut entry = Self {
            community_id: position.community_id,
            sequence: position.next_sequence,
            record,
            predecessor: position.predecessor,
            hash: AuditHash::from_bytes([0; 32]),
        };
        entry.hash = entry.recompute_hash()?;
        Ok(entry)
    }

    pub fn from_stored(
        position: AuditChainPosition,
        record: AuditRecord,
        hash: AuditHash,
    ) -> Result<Self, AuditError> {
        let entry = Self {
            community_id: position.community_id,
            sequence: position.next_sequence,
            record,
            predecessor: position.predecessor,
            hash,
        };
        entry.verify()?;
        Ok(entry)
    }

    pub fn verify(&self) -> Result<(), AuditError> {
        if self.recompute_hash()? == self.hash {
            Ok(())
        } else {
            Err(AuditError::HashMismatch)
        }
    }

    pub fn recompute_hash(&self) -> Result<AuditHash, AuditError> {
        let mut encoded = Vec::with_capacity(1024);
        write_bytes(&mut encoded, AUDIT_HASH_DOMAIN)?;
        encoded.extend_from_slice(self.community_id.as_uuid().as_bytes());
        encoded.extend_from_slice(&self.sequence.to_be_bytes());
        encoded.extend_from_slice(self.record.operation_id.as_uuid().as_bytes());
        write_bytes(&mut encoded, self.record.action.as_str().as_bytes())?;
        match self.record.actor_principal_id {
            Some(actor) => {
                encoded.push(1);
                encoded.extend_from_slice(actor.as_uuid().as_bytes());
            }
            None => encoded.push(0),
        }
        encoded.push(self.record.outcome.code());
        encoded.extend_from_slice(&self.record.occurred_at_millis.to_be_bytes());
        encoded.extend_from_slice(
            &u16::try_from(self.record.fields.0.len())
                .map_err(|_| AuditError::EntryTooLarge)?
                .to_be_bytes(),
        );
        for field in &self.record.fields.0 {
            write_bytes(&mut encoded, field.name.as_str().as_bytes())?;
            write_value(&mut encoded, &field.value)?;
        }
        match self.predecessor {
            AuditPredecessor::Genesis => encoded.push(0),
            AuditPredecessor::Entry(hash) => {
                encoded.push(1);
                encoded.extend_from_slice(hash.as_bytes());
            }
            AuditPredecessor::Imported(bridge) => {
                encoded.push(2);
                encoded.push(bridge.source.code());
                encoded.extend_from_slice(&bridge.source_sequence.to_be_bytes());
                encoded.extend_from_slice(bridge.source_head.as_bytes());
            }
        }
        if encoded.len() > MAX_ENCODED_ENTRY_BYTES {
            return Err(AuditError::EntryTooLarge);
        }
        Ok(AuditHash(Sha256::digest(encoded).into()))
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn record(&self) -> &AuditRecord {
        &self.record
    }

    pub const fn hash(&self) -> AuditHash {
        self.hash
    }

    pub const fn previous_hash(&self) -> Option<AuditHash> {
        match self.predecessor {
            AuditPredecessor::Entry(hash) => Some(hash),
            AuditPredecessor::Genesis | AuditPredecessor::Imported(_) => None,
        }
    }

    pub const fn chain_bridge(&self) -> Option<AuditChainBridge> {
        match self.predecessor {
            AuditPredecessor::Imported(bridge) => Some(bridge),
            AuditPredecessor::Genesis | AuditPredecessor::Entry(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuditError {
    InvalidAction,
    InvalidFieldName,
    InvalidIdentifier,
    InvalidCategory,
    TooManyFields,
    DuplicateField,
    SensitiveFieldNotRedacted,
    InvalidRecord,
    InvalidCommunity,
    InvalidBridge,
    SequenceExhausted,
    EntryTooLarge,
    HashMismatch,
}

impl fmt::Display for AuditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidAction => "audit action is invalid",
            Self::InvalidFieldName => "audit field name is invalid",
            Self::InvalidIdentifier => "audit identifier is invalid",
            Self::InvalidCategory => "audit category is invalid",
            Self::TooManyFields => "audit field count exceeds its bound",
            Self::DuplicateField => "audit field names must be unique",
            Self::SensitiveFieldNotRedacted => "sensitive audit field is not redacted",
            Self::InvalidRecord => "audit record is invalid",
            Self::InvalidCommunity => "audit community is invalid",
            Self::InvalidBridge => "audit chain bridge is invalid",
            Self::SequenceExhausted => "audit chain sequence is exhausted",
            Self::EntryTooLarge => "audit entry exceeds its encoded bound",
            Self::HashMismatch => "audit entry hash does not match its canonical preimage",
        })
    }
}

impl std::error::Error for AuditError {}

fn validate_category_token(value: &str, maximum_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum_bytes
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
}

fn write_bytes(encoded: &mut Vec<u8>, value: &[u8]) -> Result<(), AuditError> {
    let length = u32::try_from(value.len()).map_err(|_| AuditError::EntryTooLarge)?;
    encoded.extend_from_slice(&length.to_be_bytes());
    encoded.extend_from_slice(value);
    Ok(())
}

fn write_value(encoded: &mut Vec<u8>, value: &AuditValue) -> Result<(), AuditError> {
    match value {
        AuditValue::Identifier(value) => {
            encoded.push(1);
            write_bytes(encoded, value.as_str().as_bytes())?;
        }
        AuditValue::Category(value) => {
            encoded.push(2);
            write_bytes(encoded, value.as_str().as_bytes())?;
        }
        AuditValue::Unsigned(value) => {
            encoded.push(3);
            encoded.extend_from_slice(&value.to_be_bytes());
        }
        AuditValue::Signed(value) => {
            encoded.push(4);
            encoded.extend_from_slice(&value.to_be_bytes());
        }
        AuditValue::Boolean(value) => {
            encoded.push(5);
            encoded.push(u8::from(*value));
        }
        AuditValue::Redacted(value) => {
            encoded.push(6);
            encoded.push(value.code());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn field(name: &str, value: AuditValue) -> AuditField {
        AuditField::new(AuditFieldName::new(name).expect("field name"), value)
    }

    fn identifier(value: &str) -> AuditValue {
        AuditValue::Identifier(AuditIdentifier::new(value).expect("identifier"))
    }

    fn record(fields: Vec<AuditField>) -> AuditRecord {
        AuditRecord::new(
            OperationId::from_uuid(Uuid::from_u128(20)),
            AuditAction::new("workflow.action.completed").expect("action"),
            Some(PrincipalId::from_uuid(Uuid::from_u128(30))),
            AuditOutcome::Succeeded,
            1_900_000_000_000,
            AuditFields::new(fields).expect("fields"),
        )
        .expect("record")
    }

    fn entry(fields: Vec<AuditField>) -> AuditEntry {
        AuditEntry::append(
            AuditChainPosition::genesis(community(1)).expect("genesis"),
            record(fields),
        )
        .expect("entry")
    }

    #[test]
    fn canonical_hash_vector_is_stable_across_field_order() {
        let first = entry(vec![
            field("workflow_id", identifier("workflow:10")),
            field("attempt", AuditValue::Unsigned(2)),
            field(
                "failure_detail",
                AuditValue::Redacted(AuditRedaction::ErrorDetail),
            ),
        ]);
        let reordered = entry(vec![
            field(
                "failure_detail",
                AuditValue::Redacted(AuditRedaction::ErrorDetail),
            ),
            field("attempt", AuditValue::Unsigned(2)),
            field("workflow_id", identifier("workflow:10")),
        ]);

        assert_eq!(first.hash(), reordered.hash());
        assert_eq!(
            first.hash().as_bytes(),
            &[
                49, 173, 108, 51, 82, 187, 148, 153, 90, 223, 33, 197, 226, 43, 231, 201, 176, 35,
                82, 58, 80, 14, 35, 193, 185, 97, 236, 184, 157, 126, 146, 67,
            ]
        );
    }

    #[test]
    fn sensitive_fields_require_structural_redaction() {
        let unredacted = AuditFields::new(vec![field(
            "credential_token",
            identifier("credential:private"),
        )]);
        assert_eq!(unredacted, Err(AuditError::SensitiveFieldNotRedacted));

        let redacted = record(vec![field(
            "credential_token",
            AuditValue::Redacted(AuditRedaction::Credential),
        )]);
        let rendered = format!("{redacted:?}");
        assert!(rendered.contains("Redacted(Credential)"));
        assert!(!rendered.contains("credential:private"));

        let identifier = AuditIdentifier::new("principal:private-reference").expect("identifier");
        assert!(!format!("{identifier:?}").contains("private-reference"));

        assert!(
            serde_json::from_str::<AuditFields>(
                r#"[{"name":"private_payload","value":{"type":"identifier","value":"secret:value"}}]"#,
            )
            .is_err()
        );
        assert!(serde_json::from_str::<AuditAction>(r#""Invalid Action""#).is_err());
        assert!(
            serde_json::from_str::<AuditChainBridge>(
                r#"{"source":"buzz_v1","source_sequence":0,"source_head":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn every_canonical_identity_or_outcome_mutation_changes_the_hash() {
        let base = entry(vec![field("attempt", AuditValue::Unsigned(2))]);

        let other_community = AuditEntry::append(
            AuditChainPosition::genesis(community(2)).expect("genesis"),
            base.record.clone(),
        )
        .expect("entry");
        assert_ne!(base.hash(), other_community.hash());

        let mut changed_record = base.record.clone();
        changed_record.outcome = AuditOutcome::Failed;
        let changed_outcome = AuditEntry::append(
            AuditChainPosition::genesis(community(1)).expect("genesis"),
            changed_record,
        )
        .expect("entry");
        assert_ne!(base.hash(), changed_outcome.hash());

        let mut corrupted = base;
        corrupted.record.fields =
            AuditFields::new(vec![field("attempt", AuditValue::Unsigned(3))]).expect("fields");
        assert_eq!(corrupted.verify(), Err(AuditError::HashMismatch));
        assert_eq!(
            AuditChainPosition::after(&corrupted),
            Err(AuditError::HashMismatch)
        );
    }

    #[test]
    fn imported_head_is_bound_as_an_explicit_chain_bridge() {
        let bridge =
            AuditChainBridge::new(AuditChainSource::BuzzV1, 42, AuditHash::from_bytes([7; 32]))
                .expect("bridge");
        let imported = AuditEntry::append(
            AuditChainPosition::from_imported(community(1), bridge).expect("position"),
            record(vec![]),
        )
        .expect("imported entry");
        assert_eq!(imported.sequence(), 43);
        assert_eq!(imported.chain_bridge(), Some(bridge));
        assert_eq!(imported.previous_hash(), None);

        let changed_bridge =
            AuditChainBridge::new(AuditChainSource::BuzzV1, 42, AuditHash::from_bytes([8; 32]))
                .expect("bridge");
        let changed = AuditEntry::append(
            AuditChainPosition::from_imported(community(1), changed_bridge).expect("position"),
            record(vec![]),
        )
        .expect("changed entry");
        assert_ne!(imported.hash(), changed.hash());

        let next = AuditEntry::append(
            AuditChainPosition::after(&imported).expect("next position"),
            AuditRecord::new(
                OperationId::from_uuid(Uuid::from_u128(21)),
                AuditAction::new("workflow.action.failed").expect("action"),
                None,
                AuditOutcome::Failed,
                1_900_000_000_001,
                AuditFields::default(),
            )
            .expect("record"),
        )
        .expect("next entry");
        assert_eq!(next.sequence(), 44);
        assert_eq!(next.previous_hash(), Some(imported.hash()));
        assert_eq!(next.chain_bridge(), None);
        assert_ne!(next.hash(), imported.hash());
    }

    #[test]
    fn field_and_identity_bounds_reject_ambiguous_preimages() {
        assert_eq!(
            AuditAction::new("Workflow Action"),
            Err(AuditError::InvalidAction)
        );
        assert_eq!(
            AuditIdentifier::new("identifier with spaces"),
            Err(AuditError::InvalidIdentifier)
        );
        assert_eq!(
            AuditFields::new(vec![
                field("attempt", AuditValue::Unsigned(1)),
                field("attempt", AuditValue::Unsigned(2)),
            ]),
            Err(AuditError::DuplicateField)
        );
        let too_many = (0..=MAX_FIELDS)
            .map(|index| {
                field(
                    &format!("field_{index}"),
                    AuditValue::Boolean(index % 2 == 0),
                )
            })
            .collect();
        assert_eq!(AuditFields::new(too_many), Err(AuditError::TooManyFields));
        assert_eq!(
            AuditChainPosition::after_stored_head(
                CommunityId::from_uuid(Uuid::nil()),
                1,
                AuditHash::from_bytes([1; 32]),
            ),
            Err(AuditError::InvalidCommunity)
        );
        assert_eq!(
            AuditChainPosition::after_stored_head(community(1), 0, AuditHash::from_bytes([1; 32]),),
            Err(AuditError::InvalidCommunity)
        );
    }
}
