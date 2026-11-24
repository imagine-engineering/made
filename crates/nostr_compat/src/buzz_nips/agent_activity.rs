use crate::generated_kinds::{
    KIND_AGENT_ENGRAM, KIND_AGENT_OBSERVER_FRAME, KIND_AGENT_TURN_METRIC,
};
use crate::{PublicKey, SignedEvent, TimestampPolicy, verify_signed_event};
use secp256k1::{Parity, Scalar, SecretKey, XOnlyPublicKey};
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::HashSet;
use std::fmt;

pub const ENGRAM_D_TAG_DOMAIN: &[u8] = b"agent-memory/v1/d-tag";
pub const MAX_ENGRAM_SLUG_BYTES: usize = 255;
pub const MAX_AGENT_PLAINTEXT_BYTES: usize = 65_535;
pub const MIN_NIP44_CIPHERTEXT_BYTES: usize = 132;
pub const MAX_NIP44_CIPHERTEXT_BYTES: usize = 87_472;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AgentActivityCodecError {
    #[error("unsupported agent activity kind {0}")]
    UnsupportedKind(u16),
    #[error("invalid agent activity envelope: {0}")]
    InvalidEnvelope(String),
    #[error("invalid agent activity payload: {0}")]
    InvalidPayload(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngramBody {
    Core { profile: String },
    Memory { slug: String, value: Option<String> },
}

impl EngramBody {
    pub fn slug(&self) -> &str {
        match self {
            Self::Core { .. } => "core",
            Self::Memory { slug, .. } => slug,
        }
    }

    pub fn is_tombstone(&self) -> bool {
        matches!(self, Self::Memory { value: None, .. })
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, AgentActivityCodecError> {
        if bytes.len() > MAX_AGENT_PLAINTEXT_BYTES {
            return Err(invalid_payload("engram plaintext exceeds NIP-44 limit"));
        }
        let value = parse_strict_json(bytes)?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid_payload("engram body must be an object"))?;
        let slug = object
            .get("slug")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_payload("engram body is missing string slug"))?;
        validate_engram_slug(slug)?;
        if slug == "core" {
            let profile = object
                .get("profile")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_payload("core engram is missing string profile"))?;
            Ok(Self::Core {
                profile: profile.to_owned(),
            })
        } else {
            let value = match object.get("value") {
                Some(Value::String(value)) => Some(value.clone()),
                Some(Value::Null) => None,
                Some(_) => return Err(invalid_payload("memory value must be string or null")),
                None => return Err(invalid_payload("memory engram is missing value")),
            };
            Ok(Self::Memory {
                slug: slug.to_owned(),
                value,
            })
        }
    }
}

pub fn validate_engram_slug(slug: &str) -> Result<(), AgentActivityCodecError> {
    if slug == "core" {
        return Ok(());
    }
    if slug.len() > MAX_ENGRAM_SLUG_BYTES {
        return Err(invalid_payload("engram slug exceeds 255 bytes"));
    }
    let Some(path) = slug.strip_prefix("mem/") else {
        return Err(invalid_payload("engram slug must be core or mem/..."));
    };
    if path.is_empty() {
        return Err(invalid_payload("engram memory path is empty"));
    }
    for segment in path.split('/') {
        let bytes = segment.as_bytes();
        if bytes.is_empty()
            || bytes.len() > 64
            || !(bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
            || !bytes.iter().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
            })
        {
            return Err(invalid_payload("invalid engram slug segment"));
        }
    }
    Ok(())
}

pub fn nip44_conversation_key(
    secret: [u8; 32],
    other: PublicKey,
) -> Result<[u8; 32], AgentActivityCodecError> {
    let secret_key =
        SecretKey::from_slice(&secret).map_err(|_| invalid_payload("invalid NIP-44 secret key"))?;
    let xonly = XOnlyPublicKey::from_slice(other.as_bytes())
        .map_err(|_| invalid_payload("invalid NIP-44 public key"))?;
    let public = secp256k1::PublicKey::from_x_only_public_key(xonly, Parity::Even);
    let scalar = Scalar::from_be_bytes(secret_key.secret_bytes())
        .map_err(|_| invalid_payload("invalid NIP-44 secret scalar"))?;
    let shared = public
        .mul_tweak(&secp256k1::Secp256k1::verification_only(), &scalar)
        .map_err(|_| invalid_payload("NIP-44 ECDH failed"))?;
    let serialized = shared.serialize_uncompressed();
    let mut shared_x = [0; 32];
    shared_x.copy_from_slice(&serialized[1..33]);
    Ok(hmac_sha256(b"nip44-v2", &shared_x))
}

pub fn derive_engram_d_tag(conversation_key: &[u8; 32], slug: &str) -> String {
    let mut message = Vec::with_capacity(ENGRAM_D_TAG_DOMAIN.len() + 1 + slug.len());
    message.extend_from_slice(ENGRAM_D_TAG_DOMAIN);
    message.push(0);
    message.extend_from_slice(slug.as_bytes());
    hex::encode(hmac_sha256(conversation_key, &message))
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut key_block = [0_u8; 64];
    if key.len() > key_block.len() {
        key_block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }
    let mut inner_pad = [0x36_u8; 64];
    let mut outer_pad = [0x5c_u8; 64];
    for index in 0..64 {
        inner_pad[index] ^= key_block[index];
        outer_pad[index] ^= key_block[index];
    }
    let mut inner = Sha256::new();
    inner.update(inner_pad);
    inner.update(message);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(outer_pad);
    outer.update(inner);
    let digest = outer.finalize();
    let mut output = [0; 32];
    output.copy_from_slice(&digest);
    output
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngramEnvelope {
    pub agent: PublicKey,
    pub owner: PublicKey,
    pub d_tag: String,
}

impl EngramEnvelope {
    pub fn parse_signed_event(
        event: &SignedEvent,
        expected_agent: PublicKey,
        expected_owner: PublicKey,
    ) -> Result<Self, AgentActivityCodecError> {
        verify_signed_event(event, TimestampPolicy::Historical)
            .map_err(|error| invalid_envelope(format!("invalid signed event: {error}")))?;
        if u32::from(event.event.kind) != KIND_AGENT_ENGRAM {
            return Err(AgentActivityCodecError::UnsupportedKind(event.event.kind));
        }
        if event.event.public_key != expected_agent {
            return Err(invalid_envelope("engram author is not expected agent"));
        }
        validate_nip44_ciphertext(&event.event.content)?;
        let d_tag = parse_single_text_tag(&event.event.tags, "d")?;
        if !valid_lower_hex(&d_tag, 64) {
            return Err(invalid_envelope("engram d tag must be 64 lowercase hex"));
        }
        let owner = PublicKey::from_hex(&parse_single_text_tag(&event.event.tags, "p")?)
            .map_err(|error| invalid_envelope(format!("engram owner: {error}")))?;
        if owner != expected_owner {
            return Err(invalid_envelope(
                "engram owner does not match expected owner",
            ));
        }
        Ok(Self {
            agent: expected_agent,
            owner,
            d_tag,
        })
    }

    pub fn validate_decrypted(
        &self,
        plaintext: &[u8],
        conversation_key: &[u8; 32],
    ) -> Result<EngramBody, AgentActivityCodecError> {
        let body = EngramBody::parse(plaintext)?;
        if derive_engram_d_tag(conversation_key, body.slug()) != self.d_tag {
            return Err(invalid_envelope(
                "engram body slug does not match blinded coordinate",
            ));
        }
        Ok(body)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenCounts {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cost_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    Cancelled,
    Error,
    Unknown,
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "end_turn" => Self::EndTurn,
            "max_tokens" => Self::MaxTokens,
            "cancelled" => Self::Cancelled,
            "error" => Self::Error,
            _ => Self::Unknown,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PricingIdentity {
    pub authority: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_class: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTurnMetricPayload {
    pub harness: String,
    pub model: Option<String>,
    pub channel_id: Option<String>,
    pub session_id: Option<String>,
    pub turn_id: Option<String>,
    pub turn_seq: Option<u64>,
    pub timestamp: String,
    pub turn: Option<TokenCounts>,
    pub cumulative: Option<TokenCounts>,
    #[serde(default = "default_true")]
    pub delta_reliable: bool,
    pub stop_reason: Option<StopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing_identity: Option<PricingIdentity>,
}

impl AgentTurnMetricPayload {
    pub fn parse(bytes: &[u8]) -> Result<Self, AgentActivityCodecError> {
        if bytes.len() > MAX_AGENT_PLAINTEXT_BYTES {
            return Err(invalid_payload("turn metric plaintext exceeds limit"));
        }
        let value = parse_strict_json(bytes)?;
        reject_explicit_null(&value, "pricingIdentity")?;
        for window in ["turn", "cumulative"] {
            if let Some(Value::Object(counts)) = value.get(window) {
                for field in ["cacheReadTokens", "cacheWriteTokens"] {
                    if counts.get(field) == Some(&Value::Null) {
                        return Err(invalid_payload(format!(
                            "{window}.{field} must be omitted when unavailable"
                        )));
                    }
                }
            }
        }
        let payload: Self = serde_json::from_value(value)
            .map_err(|error| invalid_payload(format!("turn metric schema: {error}")))?;
        payload.validate()?;
        Ok(payload)
    }

    pub fn validate(&self) -> Result<(), AgentActivityCodecError> {
        if self.harness.is_empty() {
            return Err(invalid_payload("turn metric harness is required"));
        }
        parse_rfc3339("turn metric timestamp", &self.timestamp)?;
        if self.cumulative.is_some() && (self.session_id.is_none() || self.turn_seq.is_none()) {
            return Err(invalid_payload(
                "cumulative metrics require sessionId and turnSeq",
            ));
        }
        for (label, counts) in [("turn", &self.turn), ("cumulative", &self.cumulative)] {
            if let Some(counts) = counts {
                if counts
                    .cost_usd
                    .is_some_and(|cost| !cost.is_finite() || cost < 0.0)
                {
                    return Err(invalid_payload(format!(
                        "{label}.costUsd must be finite and non-negative"
                    )));
                }
            }
        }
        if let Some(pricing) = &self.pricing_identity {
            if !matches!(
                pricing.authority.as_str(),
                "api.anthropic.com" | "api.openai.com" | "openrouter.ai"
            ) || pricing.model.is_empty()
            {
                return Err(invalid_payload("invalid pricing identity"));
            }
        }
        Ok(())
    }
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentTurnMetricEnvelope {
    pub agent: PublicKey,
    pub owner: PublicKey,
}

impl AgentTurnMetricEnvelope {
    pub fn parse_signed_event(event: &SignedEvent) -> Result<Self, AgentActivityCodecError> {
        verify_signed_event(event, TimestampPolicy::Historical)
            .map_err(|error| invalid_envelope(format!("invalid signed event: {error}")))?;
        if u32::from(event.event.kind) != KIND_AGENT_TURN_METRIC {
            return Err(AgentActivityCodecError::UnsupportedKind(event.event.kind));
        }
        validate_nip44_ciphertext(&event.event.content)?;
        if event
            .event
            .tags
            .iter()
            .any(|tag| tag.first().map(String::as_str) == Some("h"))
        {
            return Err(invalid_envelope(
                "turn metrics must not expose a channel tag",
            ));
        }
        let owner = PublicKey::from_hex(&parse_single_text_tag(&event.event.tags, "p")?)
            .map_err(|error| invalid_envelope(format!("metric owner: {error}")))?;
        let agent = PublicKey::from_hex(&parse_single_text_tag(&event.event.tags, "agent")?)
            .map_err(|error| invalid_envelope(format!("metric agent: {error}")))?;
        if agent != event.event.public_key {
            return Err(invalid_envelope("metric agent tag does not equal author"));
        }
        Ok(Self { agent, owner })
    }

    pub fn visible_to(&self, reader: PublicKey) -> bool {
        reader == self.owner
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObserverFrame {
    Telemetry,
    Control,
    Unknown(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObserverEnvelope {
    pub sender: PublicKey,
    pub recipient: PublicKey,
    pub agent: PublicKey,
    pub frame: ObserverFrame,
    pub channel: Option<String>,
}

impl ObserverEnvelope {
    pub fn parse_signed_event(event: &SignedEvent) -> Result<Self, AgentActivityCodecError> {
        verify_signed_event(event, TimestampPolicy::Historical)
            .map_err(|error| invalid_envelope(format!("invalid signed event: {error}")))?;
        if u32::from(event.event.kind) != KIND_AGENT_OBSERVER_FRAME {
            return Err(AgentActivityCodecError::UnsupportedKind(event.event.kind));
        }
        validate_nip44_ciphertext(&event.event.content)?;
        let recipient = PublicKey::from_hex(&parse_single_text_tag(&event.event.tags, "p")?)
            .map_err(|error| invalid_envelope(format!("observer recipient: {error}")))?;
        let agent = PublicKey::from_hex(&parse_single_text_tag(&event.event.tags, "agent")?)
            .map_err(|error| invalid_envelope(format!("observer agent: {error}")))?;
        let frame = match parse_single_text_tag(&event.event.tags, "frame")?.as_str() {
            "telemetry" => ObserverFrame::Telemetry,
            "control" => ObserverFrame::Control,
            value => ObserverFrame::Unknown(value.to_owned()),
        };
        match frame {
            ObserverFrame::Telemetry if event.event.public_key != agent => {
                return Err(invalid_envelope("telemetry author must equal agent"));
            }
            ObserverFrame::Control if recipient != agent => {
                return Err(invalid_envelope("control recipient must equal agent"));
            }
            _ => {}
        }
        let channel = parse_optional_text_tag(&event.event.tags, "h")?;
        Ok(Self {
            sender: event.event.public_key,
            recipient,
            agent,
            frame,
            channel,
        })
    }

    pub fn visible_to(&self, reader: PublicKey) -> bool {
        reader == self.recipient
    }

    pub fn is_recognized(&self) -> bool {
        !matches!(self.frame, ObserverFrame::Unknown(_))
    }
}

#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObserverTelemetry {
    pub seq: u64,
    pub timestamp: String,
    pub kind: String,
    pub agent_index: Option<u64>,
    pub channel_id: Option<String>,
    pub session_id: Option<String>,
    pub turn_id: Option<String>,
    pub payload: Value,
}

impl fmt::Debug for ObserverTelemetry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ObserverTelemetry")
            .field("seq", &self.seq)
            .field("timestamp", &self.timestamp)
            .field("kind", &self.kind)
            .field("payload", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObserverControl {
    #[serde(rename = "type")]
    pub control_type: String,
    pub channel_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ObserverPayload {
    Telemetry(ObserverTelemetry),
    Control(ObserverControl),
    Ignored,
}

impl ObserverPayload {
    pub fn parse(frame: &ObserverFrame, bytes: &[u8]) -> Result<Self, AgentActivityCodecError> {
        if bytes.len() > MAX_AGENT_PLAINTEXT_BYTES {
            return Err(invalid_payload("observer plaintext exceeds limit"));
        }
        match frame {
            ObserverFrame::Unknown(_) => Ok(Self::Ignored),
            ObserverFrame::Telemetry => {
                let telemetry: ObserverTelemetry =
                    serde_json::from_value(parse_strict_json(bytes)?)
                        .map_err(|error| invalid_payload(format!("telemetry schema: {error}")))?;
                parse_rfc3339("observer timestamp", &telemetry.timestamp)?;
                if !telemetry.payload.is_object() {
                    return Err(invalid_payload("observer payload must be an object"));
                }
                Ok(Self::Telemetry(telemetry))
            }
            ObserverFrame::Control => {
                let control: ObserverControl = serde_json::from_value(parse_strict_json(bytes)?)
                    .map_err(|error| invalid_payload(format!("control schema: {error}")))?;
                if control.control_type == "cancel_turn" && control.channel_id.is_none() {
                    return Err(invalid_payload("cancel_turn requires channelId"));
                }
                if control.control_type == "cancel_turn" {
                    Ok(Self::Control(control))
                } else {
                    Ok(Self::Ignored)
                }
            }
        }
    }
}

fn validate_nip44_ciphertext(content: &str) -> Result<(), AgentActivityCodecError> {
    if !(MIN_NIP44_CIPHERTEXT_BYTES..=MAX_NIP44_CIPHERTEXT_BYTES).contains(&content.len()) {
        return Err(invalid_envelope("invalid NIP-44 ciphertext length"));
    }
    Ok(())
}

fn parse_single_text_tag(
    tags: &[Vec<String>],
    name: &'static str,
) -> Result<String, AgentActivityCodecError> {
    parse_optional_text_tag(tags, name)?
        .ok_or_else(|| invalid_envelope(format!("missing {name} tag")))
}

fn parse_optional_text_tag(
    tags: &[Vec<String>],
    name: &'static str,
) -> Result<Option<String>, AgentActivityCodecError> {
    let matching: Vec<_> = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some(name))
        .collect();
    match matching.as_slice() {
        [] => Ok(None),
        [tag] if tag.len() == 2 && !tag[1].is_empty() => Ok(Some(tag[1].clone())),
        [_] => Err(invalid_envelope(format!("malformed {name} tag"))),
        _ => Err(invalid_envelope(format!("duplicate {name} tag"))),
    }
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_rfc3339(label: &str, value: &str) -> Result<(), AgentActivityCodecError> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|_| ())
        .map_err(|_| invalid_payload(format!("{label} must be RFC3339")))
}

fn reject_explicit_null(value: &Value, field: &str) -> Result<(), AgentActivityCodecError> {
    if value.get(field) == Some(&Value::Null) {
        return Err(invalid_payload(format!(
            "{field} must be omitted, not null"
        )));
    }
    Ok(())
}

fn parse_strict_json(bytes: &[u8]) -> Result<Value, AgentActivityCodecError> {
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

fn invalid_envelope(reason: impl Into<String>) -> AgentActivityCodecError {
    AgentActivityCodecError::InvalidEnvelope(reason.into())
}

fn invalid_payload(reason: impl Into<String>) -> AgentActivityCodecError {
    AgentActivityCodecError::InvalidPayload(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalEvent, EventId, EventSignature};
    use secp256k1::{Keypair, Message};

    const AGENT_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 1;
        secret
    };
    const OWNER_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 2;
        secret
    };
    const AGENT: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const OWNER: &str = "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5";

    fn key(value: &str) -> PublicKey {
        PublicKey::from_hex(value).expect("fixture public key")
    }

    fn keypair(secret: [u8; 32]) -> Keypair {
        let secret = SecretKey::from_slice(&secret).expect("fixture secret");
        Keypair::from_secret_key(&secp256k1::Secp256k1::new(), &secret)
    }

    fn sign(event: CanonicalEvent, secret: [u8; 32]) -> SignedEvent {
        let claimed_id = event.event_id().expect("event id");
        let signature = secp256k1::Secp256k1::new().sign_schnorr_no_aux_rand(
            &Message::from_digest(*claimed_id.as_bytes()),
            &keypair(secret),
        );
        SignedEvent {
            claimed_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
        }
    }

    #[test]
    fn engram_coordinate_and_encrypted_vector_match_nip_ae() {
        let conversation = nip44_conversation_key(AGENT_SECRET, key(OWNER)).expect("conversation");
        assert_eq!(
            hex::encode(conversation),
            "c41c775356fd92eadc63ff5a0dc1da211b268cbea22316767095b2871ea1412d"
        );
        assert_eq!(
            nip44_conversation_key(OWNER_SECRET, key(AGENT)).expect("symmetric conversation"),
            conversation
        );
        assert_eq!(
            derive_engram_d_tag(&conversation, "mem/example"),
            "72d4f9629106451505d7d341ea85bb3ebad4f654fcfd2aad100d5a35f8a85cba"
        );

        let content = "AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABedgcxyfmpph68LBjCWZsTI5lb0Cbg8dIPVYVe/WVj/l4Yd8HGgzC8awyBi9bn9ClRdtd2IPsmont0jN/cajVSQhahTOwuNNwoJtZIg35aSsUzeCq4tQfd8E+fLoKomdPxjs=";
        let event = SignedEvent {
            claimed_id: EventId::from_hex(
                "f4a594177b7aeea4fe99a09efbf74ae85f0126244f322135682c405888a38689",
            )
            .expect("event id"),
            event: CanonicalEvent::new(
                key(AGENT),
                1_700_000_000,
                KIND_AGENT_ENGRAM as u16,
                vec![
                    vec![
                        "d".into(),
                        "72d4f9629106451505d7d341ea85bb3ebad4f654fcfd2aad100d5a35f8a85cba".into(),
                    ],
                    vec!["p".into(), OWNER.into()],
                ],
                content.into(),
            ),
            signature: EventSignature::from_hex("0a4582f0bc5995b9a010afda5984f568055988ebbe4552b4e0ec6d11aeb2b303af940f3d84726a7edd1763badb284eb3aa8457664ceba85a90d6252ed4b494cb").expect("signature"),
        };
        let envelope =
            EngramEnvelope::parse_signed_event(&event, key(AGENT), key(OWNER)).expect("engram");
        let body = envelope
            .validate_decrypted(
                br#"{"slug":"mem/example","value":"hello, agent memory"}"#,
                &conversation,
            )
            .expect("decrypted body");
        assert_eq!(
            body,
            EngramBody::Memory {
                slug: "mem/example".into(),
                value: Some("hello, agent memory".into()),
            }
        );
    }

    #[test]
    fn engram_body_rejects_duplicate_members_and_wrong_coordinate() {
        assert!(EngramBody::parse(br#"{"slug":"core","profile":"a","profile":"b"}"#).is_err());
        let envelope = EngramEnvelope {
            agent: key(AGENT),
            owner: key(OWNER),
            d_tag: "0".repeat(64),
        };
        let conversation = nip44_conversation_key(OWNER_SECRET, key(AGENT)).expect("conversation");
        assert!(
            envelope
                .validate_decrypted(br#"{"slug":"mem/example","value":"hello"}"#, &conversation,)
                .is_err()
        );
    }

    #[test]
    fn turn_metric_enforces_owner_only_envelope_and_payload_rules() {
        let event = sign(
            CanonicalEvent::new(
                key(AGENT),
                20,
                KIND_AGENT_TURN_METRIC as u16,
                vec![
                    vec!["p".into(), OWNER.into()],
                    vec!["agent".into(), AGENT.into()],
                ],
                "A".repeat(MIN_NIP44_CIPHERTEXT_BYTES),
            ),
            AGENT_SECRET,
        );
        let envelope = AgentTurnMetricEnvelope::parse_signed_event(&event).expect("metric");
        assert!(envelope.visible_to(key(OWNER)));
        assert!(!envelope.visible_to(key(AGENT)));

        let payload = AgentTurnMetricPayload::parse(
            br#"{"harness":"goose","model":null,"channelId":null,"sessionId":"s","turnId":null,"turnSeq":1,"timestamp":"2026-07-01T20:11:03.213Z","turn":{"inputTokens":1,"outputTokens":2,"totalTokens":3,"costUsd":0.1},"cumulative":{"inputTokens":10,"outputTokens":20,"totalTokens":30,"costUsd":0.2},"deltaReliable":true,"stopReason":"future_reason"}"#,
        )
        .expect("metric payload");
        assert_eq!(payload.stop_reason, Some(StopReason::Unknown));
        assert!(
            AgentTurnMetricPayload::parse(
                br#"{"harness":"goose","timestamp":"2026-07-01T20:11:03Z","pricingIdentity":null}"#
            )
            .is_err()
        );
        assert!(AgentTurnMetricPayload::parse(
            br#"{"harness":"goose","timestamp":"2026-07-01T20:11:03Z","turn":{"inputTokens":1,"outputTokens":2,"totalTokens":3,"costUsd":0,"cacheReadTokens":null}}"#
        )
        .is_err());
    }

    #[test]
    fn observer_frames_enforce_direction_privacy_and_redacted_payloads() {
        let telemetry_event = sign(
            CanonicalEvent::new(
                key(AGENT),
                30,
                KIND_AGENT_OBSERVER_FRAME as u16,
                vec![
                    vec!["p".into(), OWNER.into()],
                    vec!["agent".into(), AGENT.into()],
                    vec!["frame".into(), "telemetry".into()],
                ],
                "A".repeat(MIN_NIP44_CIPHERTEXT_BYTES),
            ),
            AGENT_SECRET,
        );
        let envelope = ObserverEnvelope::parse_signed_event(&telemetry_event).expect("telemetry");
        assert!(envelope.visible_to(key(OWNER)));
        assert!(!envelope.visible_to(key(AGENT)));
        let payload = ObserverPayload::parse(
            &envelope.frame,
            br#"{"seq":42,"timestamp":"2026-04-29T12:00:41.500Z","kind":"acp_write","agentIndex":0,"channelId":null,"sessionId":"s","turnId":"t","payload":{"secret":"do-not-log"}}"#,
        )
        .expect("observer payload");
        let debug = format!("{payload:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("do-not-log"));

        let control_event = sign(
            CanonicalEvent::new(
                key(OWNER),
                31,
                KIND_AGENT_OBSERVER_FRAME as u16,
                vec![
                    vec!["p".into(), AGENT.into()],
                    vec!["agent".into(), AGENT.into()],
                    vec!["frame".into(), "control".into()],
                ],
                "A".repeat(MIN_NIP44_CIPHERTEXT_BYTES),
            ),
            OWNER_SECRET,
        );
        let control = ObserverEnvelope::parse_signed_event(&control_event).expect("control");
        assert!(
            ObserverPayload::parse(
                &control.frame,
                br#"{"type":"cancel_turn","channelId":"channel"}"#
            )
            .is_ok()
        );
    }

    #[test]
    fn observer_unknown_frames_are_ignored_and_malformed_directions_fail() {
        let unknown = sign(
            CanonicalEvent::new(
                key(AGENT),
                40,
                KIND_AGENT_OBSERVER_FRAME as u16,
                vec![
                    vec!["p".into(), OWNER.into()],
                    vec!["agent".into(), AGENT.into()],
                    vec!["frame".into(), "future".into()],
                ],
                "A".repeat(MIN_NIP44_CIPHERTEXT_BYTES),
            ),
            AGENT_SECRET,
        );
        let unknown = ObserverEnvelope::parse_signed_event(&unknown).expect("unknown frame");
        assert!(!unknown.is_recognized());
        assert_eq!(
            ObserverPayload::parse(&unknown.frame, b"not even JSON").expect("ignored"),
            ObserverPayload::Ignored
        );

        let wrong_control = sign(
            CanonicalEvent::new(
                key(OWNER),
                41,
                KIND_AGENT_OBSERVER_FRAME as u16,
                vec![
                    vec!["p".into(), OWNER.into()],
                    vec!["agent".into(), AGENT.into()],
                    vec!["frame".into(), "control".into()],
                ],
                "A".repeat(MIN_NIP44_CIPHERTEXT_BYTES),
            ),
            OWNER_SECRET,
        );
        assert!(ObserverEnvelope::parse_signed_event(&wrong_control).is_err());
    }
}
