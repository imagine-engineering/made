use crate::buzz_nips::identity::OwnerAttestation;
use crate::generated_kinds::{
    KIND_MANAGED_AGENT, KIND_PERSONA, KIND_PRIVATE_MANAGED_AGENT, KIND_TEAM_CATALOG,
};
use crate::{
    CanonicalEvent, EventId, EventSignature, PublicKey, SignedEvent, TimestampPolicy,
    verify_signed_event,
};
use secp256k1::{Keypair, Secp256k1, SecretKey, XOnlyPublicKey};
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

pub const PRIVATE_MANAGED_AGENT_FORMAT: &str = "buzz-private-managed-agent";
pub const PRIVATE_MANAGED_AGENT_VERSION: u32 = 1;
pub const PRIVATE_MANAGED_AGENT_INGEST_ENABLED: bool = false;
pub const MAX_PERSONA_CONTENT_BYTES: usize = 65_535;
pub const MAX_PRIVATE_PLAINTEXT_BYTES: usize = 65_535;
pub const MAX_PRIVATE_CIPHERTEXT_BYTES: usize = 87_472;
pub const MAX_SAFE_GENERATION: u64 = (1_u64 << 53) - 1;
const MAX_VALUE_BYTES: usize = 32_768;
const MAX_ENV_VARS: usize = 256;
const MAX_ENV_KEY_BYTES: usize = 256;
const MAX_ENV_VALUE_BYTES: usize = 16_384;
const MAX_AGENT_ARGS: usize = 256;
const MAX_AGENT_ARG_BYTES: usize = 8_192;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AgentConfigCodecError {
    #[error("unsupported agent configuration kind {0}")]
    UnsupportedKind(u16),
    #[error("invalid agent configuration envelope: {0}")]
    InvalidEnvelope(String),
    #[error("invalid agent configuration content: {0}")]
    InvalidContent(String),
    #[error("invalid private managed-agent payload: {0}")]
    InvalidPayload(String),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersonaContent {
    pub display_name: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub avatar_url: Option<String>,
    #[serde(default)]
    pub runtime: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub name_pool: Vec<String>,
    #[serde(default)]
    pub respond_to: Option<String>,
    #[serde(default)]
    pub respond_to_allowlist: Vec<PublicKey>,
    #[serde(default)]
    pub parallelism: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersonaProjection {
    pub slug: String,
    pub shared: bool,
    pub content: PersonaContent,
}

impl PersonaProjection {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, AgentConfigCodecError> {
        if u32::from(event.kind) != KIND_PERSONA {
            return Err(AgentConfigCodecError::UnsupportedKind(event.kind));
        }
        if event.content.len() > MAX_PERSONA_CONTENT_BYTES {
            return Err(invalid_content("persona content exceeds 65,535 bytes"));
        }
        let slug = parse_single_d(&event.tags)?;
        if !valid_persona_slug(&slug) {
            return Err(invalid_envelope("invalid persona slug"));
        }
        if event
            .tags
            .iter()
            .any(|tag| tag.first().map(String::as_str) == Some("p"))
        {
            return Err(invalid_envelope("persona must not carry a p tag"));
        }
        let shared = parse_shared(&event.tags)?;
        let value: Value = serde_json::from_str(&event.content)
            .map_err(|error| invalid_content(format!("persona JSON: {error}")))?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid_content("persona content must be an object"))?;
        if object.contains_key("env_vars") {
            return Err(invalid_content(
                "public persona content must not contain env_vars",
            ));
        }
        let content: PersonaContent = serde_json::from_value(value)
            .map_err(|error| invalid_content(format!("persona schema: {error}")))?;
        validate_persona_content(&content)?;
        Ok(Self {
            slug,
            shared,
            content,
        })
    }

    pub fn to_event(
        &self,
        owner: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, AgentConfigCodecError> {
        if !valid_persona_slug(&self.slug) {
            return Err(invalid_envelope("invalid persona slug"));
        }
        validate_persona_content(&self.content)?;
        let content = serde_json::to_string(&self.content)
            .map_err(|error| invalid_content(format!("persona JSON: {error}")))?;
        if content.len() > MAX_PERSONA_CONTENT_BYTES {
            return Err(invalid_content("persona content exceeds 65,535 bytes"));
        }
        let mut tags = vec![vec!["d".into(), self.slug.clone()]];
        if self.shared {
            tags.push(vec!["shared".into(), "true".into()]);
        }
        Ok(CanonicalEvent::new(
            owner,
            created_at,
            KIND_PERSONA as u16,
            tags,
            content,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TeamCatalogProjection {
    pub team_id: String,
    pub shared: bool,
    pub content: Value,
}

impl TeamCatalogProjection {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, AgentConfigCodecError> {
        if u32::from(event.kind) != KIND_TEAM_CATALOG {
            return Err(AgentConfigCodecError::UnsupportedKind(event.kind));
        }
        let team_id = parse_single_d(&event.tags)?;
        if !valid_team_id(&team_id) {
            return Err(invalid_envelope("invalid team-catalog coordinate"));
        }
        let shared = parse_shared(&event.tags)?;
        let content: Value = serde_json::from_str(&event.content)
            .map_err(|error| invalid_content(format!("team catalog JSON: {error}")))?;
        if !content.is_object() || event.content.len() > MAX_PERSONA_CONTENT_BYTES {
            return Err(invalid_content(
                "team catalog content must be a bounded JSON object",
            ));
        }
        if contains_private_projection_field(&content) {
            return Err(invalid_content(
                "team catalog content contains a private or local-only field",
            ));
        }
        Ok(Self {
            team_id,
            shared,
            content,
        })
    }

    pub fn to_event(
        &self,
        owner: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, AgentConfigCodecError> {
        if !valid_team_id(&self.team_id)
            || !self.content.is_object()
            || contains_private_projection_field(&self.content)
        {
            return Err(invalid_content("invalid team catalog projection"));
        }
        let content = serde_json::to_string(&self.content)
            .map_err(|error| invalid_content(format!("team catalog JSON: {error}")))?;
        if content.len() > MAX_PERSONA_CONTENT_BYTES {
            return Err(invalid_content("team catalog content exceeds limit"));
        }
        let mut tags = vec![vec!["d".into(), self.team_id.clone()]];
        if self.shared {
            tags.push(vec!["shared".into(), "true".into()]);
        }
        Ok(CanonicalEvent::new(
            owner,
            created_at,
            KIND_TEAM_CATALOG as u16,
            tags,
            content,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagedAgentProjection {
    pub agent: PublicKey,
    pub content: serde_json::Map<String, Value>,
}

impl ManagedAgentProjection {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, AgentConfigCodecError> {
        if u32::from(event.kind) != KIND_MANAGED_AGENT {
            return Err(AgentConfigCodecError::UnsupportedKind(event.kind));
        }
        let coordinate = parse_single_d(&event.tags)?;
        let agent = PublicKey::from_hex(&coordinate)
            .map_err(|error| invalid_envelope(format!("managed-agent d tag: {error}")))?;
        let content: serde_json::Map<String, Value> = serde_json::from_str(&event.content)
            .map_err(|error| invalid_content(format!("managed-agent JSON: {error}")))?;
        Ok(Self { agent, content })
    }

    pub fn to_event(
        &self,
        owner: PublicKey,
        created_at: u64,
    ) -> Result<CanonicalEvent, AgentConfigCodecError> {
        let content = serde_json::to_string(&self.content)
            .map_err(|error| invalid_content(format!("managed-agent JSON: {error}")))?;
        Ok(CanonicalEvent::new(
            owner,
            created_at,
            KIND_MANAGED_AGENT as u16,
            vec![vec!["d".into(), self.agent.to_hex()]],
            content,
        ))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PrivateAgentState {
    Active,
    Deleted,
}

impl PrivateAgentState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Deleted => "deleted",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateManagedAgentEnvelope {
    pub agent: PublicKey,
    pub owner: PublicKey,
    pub generation: u64,
    pub previous_event_id: Option<EventId>,
    pub state: PrivateAgentState,
}

impl PrivateManagedAgentEnvelope {
    pub fn parse_signed_event(
        event: &SignedEvent,
        expected_owner: PublicKey,
    ) -> Result<Self, AgentConfigCodecError> {
        verify_signed_event(event, TimestampPolicy::Historical)
            .map_err(|error| invalid_envelope(format!("invalid signed event: {error}")))?;
        let canonical = &event.event;
        if u32::from(canonical.kind) != KIND_PRIVATE_MANAGED_AGENT {
            return Err(AgentConfigCodecError::UnsupportedKind(canonical.kind));
        }
        if canonical.public_key != expected_owner {
            return Err(invalid_envelope("author is not the expected owner"));
        }
        if canonical.content.is_empty() || canonical.content.len() > MAX_PRIVATE_CIPHERTEXT_BYTES {
            return Err(invalid_envelope("invalid ciphertext length"));
        }
        let mut d = None;
        let mut generation = None;
        let mut previous = None;
        let mut state = None;
        for tag in &canonical.tags {
            if tag.len() != 2 {
                return Err(invalid_envelope(
                    "every private aggregate tag must have two elements",
                ));
            }
            let slot = match tag[0].as_str() {
                "d" => &mut d,
                "g" => &mut generation,
                "prev" => &mut previous,
                "state" => &mut state,
                name => return Err(invalid_envelope(format!("unexpected tag {name}"))),
            };
            if slot.replace(tag[1].clone()).is_some() {
                return Err(invalid_envelope(format!("duplicate {} tag", tag[0])));
            }
        }
        let agent = PublicKey::from_hex(
            d.as_deref()
                .ok_or_else(|| invalid_envelope("missing d tag"))?,
        )
        .map_err(|error| invalid_envelope(format!("invalid d tag: {error}")))?;
        validate_curve_key(&agent, "agent")?;
        let generation = parse_generation(
            generation
                .as_deref()
                .ok_or_else(|| invalid_envelope("missing g tag"))?,
        )?;
        let previous_event_id = previous
            .as_deref()
            .map(EventId::from_hex)
            .transpose()
            .map_err(|error| invalid_envelope(format!("invalid prev tag: {error}")))?;
        validate_generation_predecessor(generation, previous_event_id.as_ref())
            .map_err(|error| invalid_envelope(format!("invalid CAS predecessor: {error}")))?;
        let state = match state.as_deref() {
            Some("active") => PrivateAgentState::Active,
            Some("deleted") => PrivateAgentState::Deleted,
            Some(_) => return Err(invalid_envelope("invalid state tag")),
            None => return Err(invalid_envelope("missing state tag")),
        };
        Ok(Self {
            agent,
            owner: expected_owner,
            generation,
            previous_event_id,
            state,
        })
    }

    pub fn unsigned_event(&self, created_at: u64, ciphertext: String) -> CanonicalEvent {
        let mut tags = vec![
            vec!["d".into(), self.agent.to_hex()],
            vec!["g".into(), self.generation.to_string()],
            vec!["state".into(), self.state.as_str().into()],
        ];
        if let Some(previous) = self.previous_event_id {
            tags.push(vec!["prev".into(), previous.to_hex()]);
        }
        CanonicalEvent::new(
            self.owner,
            created_at,
            KIND_PRIVATE_MANAGED_AGENT as u16,
            tags,
            ciphertext,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionEvent {
    pub id: EventId,
    pub pubkey: PublicKey,
    pub created_at: u64,
    pub kind: u16,
    pub tags: Vec<Vec<String>>,
    pub content: String,
    pub sig: EventSignature,
}

impl ProjectionEvent {
    fn as_signed_event(&self) -> SignedEvent {
        SignedEvent {
            claimed_id: self.id,
            event: CanonicalEvent::new(
                self.pubkey,
                self.created_at,
                self.kind,
                self.tags.clone(),
                self.content.clone(),
            ),
            signature: self.sig,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionRecoveryV1 {
    pub version: u32,
    pub signed_event: ProjectionEvent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DefinitionBinding {
    pub revision: u64,
    pub event_id: EventId,
    pub content_sha256: String,
    pub recovery: ProjectionRecoveryV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InstanceBinding {
    pub event_id: EventId,
    pub content_sha256: String,
    pub recovery: ProjectionRecoveryV1,
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivateIdentity {
    pub private_key_nsec: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_tag: Option<String>,
}

impl fmt::Debug for PrivateIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrivateIdentity")
            .field("private_key_nsec", &"<redacted>")
            .field("auth_tag", &self.auth_tag.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivateConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition_coordinate: Option<String>,
    pub relay_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_command_override: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_args: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turn_duration_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env_vars: BTreeMap<String, String>,
    pub backend: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona_name_in_team: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_mesh: Option<Value>,
}

impl fmt::Debug for PrivateConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrivateConfig")
            .field("contents", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivePrivateAgent {
    pub definition: DefinitionBinding,
    pub instance_projection: InstanceBinding,
    pub identity: PrivateIdentity,
    pub config: PrivateConfig,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivateManagedAgentPayload {
    pub format: String,
    pub version: u32,
    pub agent_pubkey: PublicKey,
    pub owner_pubkey: PublicKey,
    pub generation: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_event_id: Option<EventId>,
    pub state: PrivateAgentState,
    pub updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<ActivePrivateAgent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

impl PrivateManagedAgentPayload {
    pub fn parse_decrypted(bytes: &[u8]) -> Result<Self, AgentConfigCodecError> {
        if bytes.len() > MAX_PRIVATE_PLAINTEXT_BYTES {
            return Err(invalid_payload("plaintext exceeds NIP-44 limit"));
        }
        let value = parse_strict_json(bytes)?;
        let payload: Self = serde_json::from_value(value)
            .map_err(|error| invalid_payload(format!("schema: {error}")))?;
        payload.validate()?;
        Ok(payload)
    }

    pub fn encode_decrypted(&self) -> Result<Vec<u8>, AgentConfigCodecError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| invalid_payload(format!("serialization: {error}")))?;
        if bytes.len() > MAX_PRIVATE_PLAINTEXT_BYTES {
            return Err(invalid_payload("plaintext exceeds NIP-44 limit"));
        }
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), AgentConfigCodecError> {
        if self.format != PRIVATE_MANAGED_AGENT_FORMAT
            || self.version != PRIVATE_MANAGED_AGENT_VERSION
        {
            return Err(invalid_payload("unsupported format or version"));
        }
        validate_curve_key(&self.agent_pubkey, "agent")?;
        validate_curve_key(&self.owner_pubkey, "owner")?;
        validate_generation_predecessor(self.generation, self.previous_event_id.as_ref())?;
        parse_rfc3339("updated_at", &self.updated_at)?;
        for (name, value) in &self.extensions {
            if name.is_empty() || name.len() > 128 || !name.contains(':') {
                return Err(invalid_payload(
                    "extension names must be namespaced and at most 128 bytes",
                ));
            }
            validate_value_size("extension", value)?;
        }
        match self.state {
            PrivateAgentState::Active => {
                if self.deleted_at.is_some() {
                    return Err(invalid_payload("active payload contains deleted_at"));
                }
                let active = self
                    .active
                    .as_ref()
                    .ok_or_else(|| invalid_payload("active payload is missing active body"))?;
                validate_active(active, self)?;
            }
            PrivateAgentState::Deleted => {
                if self.active.is_some() {
                    return Err(invalid_payload("deleted payload contains active body"));
                }
                parse_rfc3339(
                    "deleted_at",
                    self.deleted_at
                        .as_deref()
                        .ok_or_else(|| invalid_payload("deleted payload is missing deleted_at"))?,
                )?;
            }
        }
        Ok(())
    }

    pub fn matches_envelope(&self, envelope: &PrivateManagedAgentEnvelope) -> bool {
        self.agent_pubkey == envelope.agent
            && self.owner_pubkey == envelope.owner
            && self.generation == envelope.generation
            && self.previous_event_id == envelope.previous_event_id
            && self.state == envelope.state
    }
}

fn validate_active(
    active: &ActivePrivateAgent,
    payload: &PrivateManagedAgentPayload,
) -> Result<(), AgentConfigCodecError> {
    if active.definition.revision == 0 || active.definition.revision > MAX_SAFE_GENERATION {
        return Err(invalid_payload("invalid definition revision"));
    }
    let definition_slug = parse_definition_coordinate(
        active.config.definition_coordinate.as_deref(),
        payload.owner_pubkey,
    )?;
    validate_projection_binding(
        "definition",
        KIND_PERSONA as u16,
        payload.owner_pubkey,
        &definition_slug,
        active.definition.event_id,
        &active.definition.content_sha256,
        &active.definition.recovery,
    )?;
    validate_projection_binding(
        "instance projection",
        KIND_MANAGED_AGENT as u16,
        payload.owner_pubkey,
        &payload.agent_pubkey.to_hex(),
        active.instance_projection.event_id,
        &active.instance_projection.content_sha256,
        &active.instance_projection.recovery,
    )?;
    let secret = decode_nsec(&active.identity.private_key_nsec)?;
    let secret_key =
        SecretKey::from_slice(&secret).map_err(|_| invalid_payload("invalid agent nsec secret"))?;
    let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret_key);
    let (derived, _) = XOnlyPublicKey::from_keypair(&keypair);
    if derived.serialize() != *payload.agent_pubkey.as_bytes() {
        return Err(invalid_payload("agent nsec does not derive d coordinate"));
    }
    if let Some(encoded) = &active.identity.auth_tag {
        if encoded.len() > 4096 {
            return Err(invalid_payload("attestation exceeds limit"));
        }
        let tag: Vec<String> = serde_json::from_str(encoded)
            .map_err(|_| invalid_payload("invalid owner attestation JSON"))?;
        let attestation = OwnerAttestation::parse_tag(&tag)
            .map_err(|error| invalid_payload(format!("owner attestation: {error}")))?;
        if attestation.owner != payload.owner_pubkey || !attestation.conditions.as_str().is_empty()
        {
            return Err(invalid_payload(
                "attestation must be unconditional and authored by owner",
            ));
        }
        attestation
            .verify_identity_binding(&payload.owner_pubkey, &payload.agent_pubkey, None)
            .map_err(|error| invalid_payload(format!("owner attestation: {error}")))?;
    }
    let config = &active.config;
    if config.relay_url.is_empty() || config.relay_url.len() > 4096 {
        return Err(invalid_payload("invalid relay_url length"));
    }
    if config.agent_args.len() > MAX_AGENT_ARGS
        || config
            .agent_args
            .iter()
            .any(|argument| argument.len() > MAX_AGENT_ARG_BYTES)
    {
        return Err(invalid_payload("agent_args exceed limits"));
    }
    if config.env_vars.len() > MAX_ENV_VARS
        || config.env_vars.iter().any(|(name, value)| {
            name.is_empty() || name.len() > MAX_ENV_KEY_BYTES || value.len() > MAX_ENV_VALUE_BYTES
        })
    {
        return Err(invalid_payload("env_vars exceed limits"));
    }
    validate_value_size("backend", &config.backend)?;
    if let Some(relay_mesh) = &config.relay_mesh {
        validate_value_size("relay_mesh", relay_mesh)?;
    }
    Ok(())
}

fn validate_projection_binding(
    label: &str,
    expected_kind: u16,
    owner: PublicKey,
    expected_d: &str,
    event_id: EventId,
    content_hash: &str,
    recovery: &ProjectionRecoveryV1,
) -> Result<(), AgentConfigCodecError> {
    if recovery.version != 1 {
        return Err(invalid_payload(format!(
            "unsupported {label} recovery version"
        )));
    }
    if !valid_lower_hash(content_hash) {
        return Err(invalid_payload(format!("invalid {label} content hash")));
    }
    let signed_event = recovery.signed_event.as_signed_event();
    verify_signed_event(&signed_event, TimestampPolicy::Historical)
        .map_err(|error| invalid_payload(format!("invalid {label} recovery: {error}")))?;
    if signed_event.claimed_id != event_id
        || signed_event.event.kind != expected_kind
        || signed_event.event.public_key != owner
        || content_sha256(signed_event.event.content.as_bytes()) != content_hash
    {
        return Err(invalid_payload(format!(
            "{label} recovery does not match binding"
        )));
    }
    let d = parse_single_d(&signed_event.event.tags)
        .map_err(|_| invalid_payload(format!("invalid {label} recovery coordinate")))?;
    if d != expected_d {
        return Err(invalid_payload(format!(
            "{label} recovery has wrong coordinate"
        )));
    }
    validate_value_size(
        label,
        &serde_json::to_value(recovery)
            .map_err(|error| invalid_payload(format!("invalid {label}: {error}")))?,
    )
}

fn validate_persona_content(content: &PersonaContent) -> Result<(), AgentConfigCodecError> {
    if content.display_name.is_empty() {
        return Err(invalid_content("display_name must not be empty"));
    }
    if !matches!(
        content.respond_to.as_deref(),
        None | Some("anyone" | "owner-only" | "allowlist")
    ) {
        return Err(invalid_content("invalid respond_to policy"));
    }
    if content.parallelism == Some(0) {
        return Err(invalid_content("parallelism must be positive"));
    }
    Ok(())
}

fn parse_single_d(tags: &[Vec<String>]) -> Result<String, AgentConfigCodecError> {
    let matching: Vec<_> = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some("d"))
        .collect();
    if matching.len() != 1 || matching[0].len() != 2 || matching[0][1].is_empty() {
        return Err(invalid_envelope(
            "expected exactly one non-empty two-element d tag",
        ));
    }
    Ok(matching[0][1].clone())
}

fn parse_shared(tags: &[Vec<String>]) -> Result<bool, AgentConfigCodecError> {
    let matching: Vec<_> = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some("shared"))
        .collect();
    match matching.as_slice() {
        [] => Ok(false),
        [tag] if tag.len() == 2 && tag[1] == "true" => Ok(true),
        _ => Err(invalid_envelope(
            "shared must be absent or exactly [\"shared\",\"true\"] once",
        )),
    }
}

fn valid_persona_slug(slug: &str) -> bool {
    let bytes = slug.as_bytes();
    (1..=64).contains(&bytes.len())
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && bytes.iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
}

fn valid_team_id(team_id: &str) -> bool {
    !team_id.is_empty()
        && team_id.chars().count() <= 64
        && !team_id
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
}

fn parse_generation(value: &str) -> Result<u64, AgentConfigCodecError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid_envelope("g must be canonical decimal"));
    }
    let generation = value
        .parse::<u64>()
        .map_err(|_| invalid_envelope("invalid g tag"))?;
    if generation == 0 || generation > MAX_SAFE_GENERATION {
        return Err(invalid_envelope("g must be a positive safe integer"));
    }
    Ok(generation)
}

fn validate_generation_predecessor(
    generation: u64,
    previous: Option<&EventId>,
) -> Result<(), AgentConfigCodecError> {
    if generation == 0 || generation > MAX_SAFE_GENERATION {
        return Err(invalid_payload(
            "generation must be a positive safe integer",
        ));
    }
    if (generation == 1) != previous.is_none() {
        return Err(invalid_payload(
            "predecessor must be absent exactly at generation one",
        ));
    }
    Ok(())
}

fn parse_definition_coordinate(
    coordinate: Option<&str>,
    owner: PublicKey,
) -> Result<String, AgentConfigCodecError> {
    let coordinate = coordinate.ok_or_else(|| invalid_payload("missing definition coordinate"))?;
    let owner_hex = owner.to_hex();
    let mut parts = coordinate.splitn(3, ':');
    let kind = parts.next();
    let coordinate_owner = parts.next();
    let slug = parts.next();
    if kind != Some("30175")
        || coordinate_owner != Some(owner_hex.as_str())
        || slug.is_none_or(|slug| !valid_persona_slug(slug))
    {
        return Err(invalid_payload("invalid definition coordinate"));
    }
    Ok(slug.unwrap_or_default().to_owned())
}

fn validate_curve_key(key: &PublicKey, label: &str) -> Result<(), AgentConfigCodecError> {
    XOnlyPublicKey::from_slice(key.as_bytes())
        .map(|_| ())
        .map_err(|_| invalid_payload(format!("invalid {label} curve point")))
}

fn parse_rfc3339(label: &str, value: &str) -> Result<(), AgentConfigCodecError> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|_| ())
        .map_err(|_| invalid_payload(format!("{label} must be RFC3339")))
}

fn validate_value_size(label: &str, value: &Value) -> Result<(), AgentConfigCodecError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| invalid_payload(format!("invalid {label}: {error}")))?;
    if bytes.len() > MAX_VALUE_BYTES {
        return Err(invalid_payload(format!("{label} exceeds size limit")));
    }
    Ok(())
}

fn valid_lower_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn content_sha256(content: &[u8]) -> String {
    hex::encode(Sha256::digest(content))
}

fn contains_private_projection_field(value: &Value) -> bool {
    const FORBIDDEN: &[&str] = &[
        "env_vars",
        "respond_to_allowlist",
        "source_id",
        "local_id",
        "filesystem_path",
        "private_key_nsec",
        "auth_tag",
    ];
    match value {
        Value::Object(object) => object.iter().any(|(name, value)| {
            FORBIDDEN.contains(&name.as_str()) || contains_private_projection_field(value)
        }),
        Value::Array(values) => values.iter().any(contains_private_projection_field),
        _ => false,
    }
}

fn decode_nsec(value: &str) -> Result<[u8; 32], AgentConfigCodecError> {
    if value.to_ascii_lowercase() != value || !value.starts_with("nsec1") {
        return Err(invalid_payload("private key must be lowercase nsec"));
    }
    let separator = value
        .rfind('1')
        .ok_or_else(|| invalid_payload("invalid nsec separator"))?;
    let (human, encoded) = value.split_at(separator);
    let encoded = &encoded[1..];
    if human != "nsec" || encoded.len() < 6 {
        return Err(invalid_payload("invalid nsec encoding"));
    }
    const CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    let values: Vec<u8> = encoded
        .bytes()
        .map(|character| {
            CHARSET
                .iter()
                .position(|candidate| *candidate == character)
                .and_then(|position| u8::try_from(position).ok())
                .ok_or_else(|| invalid_payload("invalid nsec character"))
        })
        .collect::<Result<_, _>>()?;
    let mut checksum_input = Vec::new();
    checksum_input.extend(human.bytes().map(|byte| byte >> 5));
    checksum_input.push(0);
    checksum_input.extend(human.bytes().map(|byte| byte & 31));
    checksum_input.extend(&values);
    if bech32_polymod(&checksum_input) != 1 {
        return Err(invalid_payload("invalid nsec checksum"));
    }
    let payload = &values[..values.len() - 6];
    let mut output = Vec::new();
    let mut accumulator = 0_u32;
    let mut bits = 0_u8;
    for value in payload {
        accumulator = ((accumulator << 5) | u32::from(*value)) & 0x0fff;
        bits += 5;
        while bits >= 8 {
            bits -= 8;
            output.push(((accumulator >> bits) & 0xff) as u8);
        }
    }
    if bits >= 5 || ((accumulator << (8 - bits)) & 0xff) != 0 || output.len() != 32 {
        return Err(invalid_payload("invalid nsec payload length"));
    }
    let mut secret = [0; 32];
    secret.copy_from_slice(&output);
    Ok(secret)
}

fn bech32_polymod(values: &[u8]) -> u32 {
    let mut checksum = 1_u32;
    for value in values {
        let top = checksum >> 25;
        checksum = (checksum & 0x1ff_ffff) << 5 ^ u32::from(*value);
        for (bit, generator) in [
            0x3b6a_57b2,
            0x2650_8e6d,
            0x1ea1_19fa,
            0x3d42_33dd,
            0x2a14_62b3,
        ]
        .iter()
        .enumerate()
        {
            if (top >> bit) & 1 == 1 {
                checksum ^= generator;
            }
        }
    }
    checksum
}

fn parse_strict_json(bytes: &[u8]) -> Result<Value, AgentConfigCodecError> {
    struct StrictValue;

    impl<'de> DeserializeSeed<'de> for StrictValue {
        type Value = Value;

        fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
            deserializer.deserialize_any(self)
        }
    }

    impl<'de> Visitor<'de> for StrictValue {
        type Value = Value;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("JSON with unique object keys")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
            Ok(Value::Bool(value))
        }

        fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
            Ok(Value::Number(value.into()))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
            Ok(Value::Number(value.into()))
        }

        fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Value, E> {
            serde_json::Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| E::custom("non-finite number"))
        }

        fn visit_str<E>(self, value: &str) -> Result<Value, E> {
            Ok(Value::String(value.to_owned()))
        }

        fn visit_string<E>(self, value: String) -> Result<Value, E> {
            Ok(Value::String(value))
        }

        fn visit_unit<E>(self) -> Result<Value, E> {
            Ok(Value::Null)
        }

        fn visit_none<E>(self) -> Result<Value, E> {
            Ok(Value::Null)
        }

        fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
            deserializer.deserialize_any(self)
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
            let mut values = Vec::new();
            while let Some(value) = sequence.next_element_seed(StrictValue)? {
                values.push(value);
            }
            Ok(Value::Array(values))
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
            let mut seen = HashSet::new();
            let mut values = serde_json::Map::new();
            while let Some(key) = map.next_key::<String>()? {
                if !seen.insert(key.clone()) {
                    return Err(serde::de::Error::custom(format!("duplicate key: {key}")));
                }
                values.insert(key, map.next_value_seed(StrictValue)?);
            }
            Ok(Value::Object(values))
        }
    }

    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValue
        .deserialize(&mut deserializer)
        .map_err(|error| invalid_payload(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| invalid_payload(error.to_string()))?;
    Ok(value)
}

fn invalid_envelope(reason: impl Into<String>) -> AgentConfigCodecError {
    AgentConfigCodecError::InvalidEnvelope(reason.into())
}

fn invalid_content(reason: impl Into<String>) -> AgentConfigCodecError {
    AgentConfigCodecError::InvalidContent(reason.into())
}

fn invalid_payload(reason: impl Into<String>) -> AgentConfigCodecError {
    AgentConfigCodecError::InvalidPayload(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use secp256k1::Message;

    const OWNER_SECRET: [u8; 32] = [1; 32];
    const AGENT_SECRET: [u8; 32] = [2; 32];

    fn keypair(secret: [u8; 32]) -> Keypair {
        let secret = SecretKey::from_slice(&secret).expect("fixture secret");
        Keypair::from_secret_key(&Secp256k1::new(), &secret)
    }

    fn public_key(secret: [u8; 32]) -> PublicKey {
        let (public, _) = XOnlyPublicKey::from_keypair(&keypair(secret));
        PublicKey::from_bytes(public.serialize())
    }

    fn sign(event: CanonicalEvent, secret: [u8; 32]) -> SignedEvent {
        let id = event.event_id().expect("event id");
        let signature = Secp256k1::new()
            .sign_schnorr_no_aux_rand(&Message::from_digest(*id.as_bytes()), &keypair(secret));
        SignedEvent {
            claimed_id: id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
        }
    }

    #[test]
    fn persona_and_team_privacy_envelopes_round_trip_and_fail_closed() {
        let persona = PersonaProjection {
            slug: "test-agent".into(),
            shared: true,
            content: PersonaContent {
                display_name: "Test Agent".into(),
                system_prompt: Some("You are a test assistant.".into()),
                avatar_url: None,
                runtime: Some("goose".into()),
                model: None,
                provider: None,
                name_pool: vec!["Alpha".into()],
                respond_to: Some("owner-only".into()),
                respond_to_allowlist: Vec::new(),
                parallelism: Some(1),
            },
        };
        let event = persona
            .to_event(public_key(OWNER_SECRET), 1_700_000_000)
            .expect("persona event");
        assert_eq!(
            PersonaProjection::parse_event(&event).expect("persona"),
            persona
        );

        let mut malformed_shared = event.clone();
        malformed_shared
            .tags
            .push(vec!["shared".into(), "true".into()]);
        assert!(matches!(
            PersonaProjection::parse_event(&malformed_shared),
            Err(AgentConfigCodecError::InvalidEnvelope(_))
        ));
        let mut leaked_secret = event;
        leaked_secret.content = r#"{"display_name":"bad","env_vars":{"TOKEN":"secret"}}"#.into();
        assert!(matches!(
            PersonaProjection::parse_event(&leaked_secret),
            Err(AgentConfigCodecError::InvalidContent(_))
        ));

        let team = TeamCatalogProjection {
            team_id: "builtin-team:welcome".into(),
            shared: false,
            content: serde_json::json!({"version": 1, "members": []}),
        };
        let team_event = team
            .to_event(public_key(OWNER_SECRET), 1_700_000_001)
            .expect("team event");
        assert_eq!(
            TeamCatalogProjection::parse_event(&team_event).expect("team projection"),
            team
        );
        let private_team = TeamCatalogProjection {
            team_id: "team-private".into(),
            shared: true,
            content: serde_json::json!({"members": [{"env_vars": {"TOKEN": "secret"}}]}),
        };
        assert!(
            private_team
                .to_event(public_key(OWNER_SECRET), 1_700_000_002)
                .is_err()
        );
    }

    #[test]
    fn managed_agent_projection_accepts_slim_and_legacy_content() {
        for content in [
            serde_json::json!({"name":"agent", "definition_id":"persona"}),
            serde_json::json!({"name":"agent", "system_prompt":"legacy", "model":"old"}),
        ] {
            let projection = ManagedAgentProjection {
                agent: public_key(AGENT_SECRET),
                content: content.as_object().expect("object").clone(),
            };
            let event = projection
                .to_event(public_key(OWNER_SECRET), 1_700_000_000)
                .expect("projection");
            assert_eq!(
                ManagedAgentProjection::parse_event(&event).expect("managed agent"),
                projection
            );
        }
    }

    #[test]
    fn private_envelope_enforces_cas_predecessors_and_signed_owner() {
        let owner = public_key(OWNER_SECRET);
        let first = PrivateManagedAgentEnvelope {
            agent: public_key(AGENT_SECRET),
            owner,
            generation: 1,
            previous_event_id: None,
            state: PrivateAgentState::Active,
        };
        let signed = sign(first.unsigned_event(10, "ciphertext".into()), OWNER_SECRET);
        assert_eq!(
            PrivateManagedAgentEnvelope::parse_signed_event(&signed, owner).expect("envelope"),
            first
        );

        let mut missing_previous = signed.event;
        missing_previous.tags[1][1] = "2".into();
        let missing_previous = sign(missing_previous, OWNER_SECRET);
        assert!(matches!(
            PrivateManagedAgentEnvelope::parse_signed_event(&missing_previous, owner),
            Err(AgentConfigCodecError::InvalidEnvelope(_))
        ));
        assert!(!PRIVATE_MANAGED_AGENT_INGEST_ENABLED);
    }

    #[test]
    fn private_payload_rejects_versions_duplicates_and_invalid_tombstones() {
        let payload = PrivateManagedAgentPayload {
            format: PRIVATE_MANAGED_AGENT_FORMAT.into(),
            version: 1,
            agent_pubkey: public_key(AGENT_SECRET),
            owner_pubkey: public_key(OWNER_SECRET),
            generation: 2,
            previous_event_id: Some(EventId::from_bytes([3; 32])),
            state: PrivateAgentState::Deleted,
            updated_at: "2026-08-15T12:00:00Z".into(),
            active: None,
            deleted_at: Some("2026-08-15T12:00:00Z".into()),
            extensions: BTreeMap::new(),
        };
        let bytes = payload.encode_decrypted().expect("tombstone");
        assert_eq!(
            PrivateManagedAgentPayload::parse_decrypted(&bytes).expect("payload"),
            payload
        );

        let mut value: Value = serde_json::from_slice(&bytes).expect("JSON");
        value["version"] = Value::from(2);
        assert!(
            PrivateManagedAgentPayload::parse_decrypted(&serde_json::to_vec(&value).expect("JSON"))
                .is_err()
        );
        assert!(
            PrivateManagedAgentPayload::parse_decrypted(br#"{"format":"a","format":"b"}"#).is_err()
        );
        let known_nsec = "nsec1vl029mgpspedva04g90vltkh6fvh240zqtv9k0t9af8935ke9laqsnlfe5";
        assert!(decode_nsec(known_nsec).is_ok());
        let mut invalid_nsec = known_nsec.to_owned();
        invalid_nsec.replace_range(invalid_nsec.len() - 1.., "q");
        assert!(decode_nsec(&invalid_nsec).is_err());
    }

    #[test]
    fn projection_recovery_rejects_wrong_coordinate_and_content_binding() {
        let owner = public_key(OWNER_SECRET);
        let projection = sign(
            CanonicalEvent::new(
                owner,
                10,
                KIND_PERSONA as u16,
                vec![vec!["d".into(), "persona".into()]],
                "definition".into(),
            ),
            OWNER_SECRET,
        );
        let recovery = ProjectionRecoveryV1 {
            version: 1,
            signed_event: ProjectionEvent {
                id: projection.claimed_id,
                pubkey: projection.event.public_key,
                created_at: projection.event.created_at,
                kind: projection.event.kind,
                tags: projection.event.tags,
                content: projection.event.content,
                sig: projection.signature,
            },
        };
        let hash = content_sha256(recovery.signed_event.content.as_bytes());
        assert!(
            validate_projection_binding(
                "definition",
                KIND_PERSONA as u16,
                owner,
                "persona",
                recovery.signed_event.id,
                &hash,
                &recovery,
            )
            .is_ok()
        );
        assert!(
            validate_projection_binding(
                "definition",
                KIND_PERSONA as u16,
                owner,
                "wrong",
                recovery.signed_event.id,
                &hash,
                &recovery,
            )
            .is_err()
        );
        assert!(
            validate_projection_binding(
                "definition",
                KIND_PERSONA as u16,
                owner,
                "persona",
                recovery.signed_event.id,
                &content_sha256(b"tampered"),
                &recovery,
            )
            .is_err()
        );
    }
}
