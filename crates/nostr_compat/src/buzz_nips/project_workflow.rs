use crate::buzz_nips::identity::{AttestationConditions, OwnerAttestation};
use crate::generated_kinds::{KIND_PROJECT, RELAY_ADMIN_SET_WORKSPACE_PROFILE};
use crate::{EventSignature, PublicKey, SignedEvent, TimestampPolicy, verify_signed_event};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use secp256k1::{Message, Secp256k1, XOnlyPublicKey, schnorr::Signature};
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use sha2::{Digest as _, Sha256};
use std::collections::HashSet;
use std::fmt;

const GIT_SIGNING_DOMAIN: &[u8] = b"nostr:git:v1:";
const ARMOR_BEGIN: &str = "-----BEGIN SIGNED MESSAGE-----";
const ARMOR_END: &str = "-----END SIGNED MESSAGE-----";
const MAX_GIT_PAYLOAD_BYTES: usize = 100 * 1024 * 1024;
const MAX_GIT_ENVELOPE_BYTES: usize = 2_048;
const MAX_GIT_BASE64_BYTES: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ProjectWorkflowCodecError {
    #[error("unsupported project/workflow kind {0}")]
    UnsupportedKind(u16),
    #[error("invalid project/workflow envelope: {0}")]
    InvalidEnvelope(String),
    #[error("invalid git signature: {0}")]
    InvalidGitSignature(String),
    #[error("git payload is {actual} bytes, maximum is {maximum}")]
    GitPayloadTooLarge { actual: usize, maximum: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitOwnerAttestation {
    pub owner: PublicKey,
    pub conditions: AttestationConditions,
    pub signature: EventSignature,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitSignatureEnvelope {
    pub signer: PublicKey,
    pub signature: EventSignature,
    pub timestamp: u32,
    pub owner_attestation: Option<GitOwnerAttestation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnerAttestationStatus {
    Absent,
    Valid,
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitSignatureVerification {
    pub signer: PublicKey,
    pub timestamp: u32,
    pub owner: Option<PublicKey>,
    pub owner_attestation: OwnerAttestationStatus,
}

impl GitSignatureEnvelope {
    pub fn parse_armored(armored: &str) -> Result<Self, ProjectWorkflowCodecError> {
        let without_final_newline = armored
            .strip_suffix('\n')
            .ok_or_else(|| invalid_git("armor must end with one LF"))?;
        if without_final_newline.ends_with('\n') {
            return Err(invalid_git("armor contains extra trailing data"));
        }
        let mut lines = without_final_newline.split('\n');
        let begin = lines.next().unwrap_or_default();
        let encoded = lines.next().unwrap_or_default();
        let end = lines.next().unwrap_or_default();
        if lines.next().is_some() || begin != ARMOR_BEGIN || end != ARMOR_END {
            return Err(invalid_git(
                "armor must contain exactly BEGIN, base64, and END lines",
            ));
        }
        if encoded.is_empty()
            || encoded.len() > MAX_GIT_BASE64_BYTES
            || encoded.ends_with([' ', '\t', '\r'])
        {
            return Err(invalid_git("invalid armored base64 line"));
        }
        let decoded = STANDARD
            .decode(encoded)
            .map_err(|error| invalid_git(format!("invalid base64: {error}")))?;
        if decoded.len() > MAX_GIT_ENVELOPE_BYTES {
            return Err(invalid_git("decoded envelope exceeds 2048 bytes"));
        }
        let decoded = std::str::from_utf8(&decoded)
            .map_err(|_| invalid_git("decoded envelope is not UTF-8"))?;
        let value = parse_strict_json(decoded.as_bytes())?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid_git("git signature envelope must be an object"))?;
        let allowed = ["v", "pk", "sig", "t", "oa"];
        if object.keys().any(|key| !allowed.contains(&key.as_str())) {
            return Err(invalid_git("unknown v1 envelope field"));
        }
        if object.get("v").and_then(Value::as_u64) != Some(1) {
            return Err(invalid_git("v must be integer 1"));
        }
        let signer = PublicKey::from_hex(required_string(object, "pk")?)
            .map_err(|error| invalid_git(format!("invalid signer: {error}")))?;
        let signature = EventSignature::from_hex(required_string(object, "sig")?)
            .map_err(|error| invalid_git(format!("invalid signature: {error}")))?;
        let timestamp = object
            .get("t")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| invalid_git("t must be a uint32 integer"))?;
        let owner_attestation = object
            .get("oa")
            .map(|value| GitOwnerAttestation::from_value(value, signer))
            .transpose()?;
        let envelope = Self {
            signer,
            signature,
            timestamp,
            owner_attestation,
        };
        if envelope.canonical_json() != decoded {
            return Err(invalid_git("envelope is not canonical compact JSON"));
        }
        Ok(envelope)
    }

    pub fn canonical_json(&self) -> String {
        let signer = self.signer.to_hex();
        let signature = self.signature.to_hex();
        match &self.owner_attestation {
            Some(owner) => format!(
                r#"{{"v":1,"pk":"{signer}","sig":"{signature}","t":{},"oa":["{}","{}","{}"]}}"#,
                self.timestamp,
                owner.owner.to_hex(),
                owner.conditions.as_str(),
                owner.signature.to_hex()
            ),
            None => format!(
                r#"{{"v":1,"pk":"{signer}","sig":"{signature}","t":{}}}"#,
                self.timestamp
            ),
        }
    }

    pub fn to_armored(&self) -> String {
        let encoded = STANDARD.encode(self.canonical_json());
        format!("{ARMOR_BEGIN}\n{encoded}\n{ARMOR_END}\n")
    }

    pub fn signing_hash(&self, payload: &[u8]) -> Result<[u8; 32], ProjectWorkflowCodecError> {
        compute_git_signing_hash(self.timestamp, self.owner_attestation.as_ref(), payload)
    }

    pub fn verify(
        &self,
        payload: &[u8],
    ) -> Result<GitSignatureVerification, ProjectWorkflowCodecError> {
        let digest = self.signing_hash(payload)?;
        let signer = XOnlyPublicKey::from_slice(self.signer.as_bytes())
            .map_err(|_| invalid_git("invalid signer public key"))?;
        let signature = Signature::from_slice(self.signature.as_bytes())
            .map_err(|_| invalid_git("invalid Schnorr signature"))?;
        Secp256k1::verification_only()
            .verify_schnorr(&signature, &Message::from_digest(digest), &signer)
            .map_err(|_| invalid_git("git object signature verification failed"))?;
        let (owner, owner_attestation) = match &self.owner_attestation {
            None => (None, OwnerAttestationStatus::Absent),
            Some(owner) => {
                let attestation = OwnerAttestation {
                    owner: owner.owner,
                    conditions: owner.conditions.clone(),
                    signature: owner.signature,
                };
                let status = if attestation
                    .verify_for_membership(&self.signer, u64::from(self.timestamp))
                    .is_ok()
                {
                    OwnerAttestationStatus::Valid
                } else {
                    OwnerAttestationStatus::Invalid
                };
                (Some(owner.owner), status)
            }
        };
        Ok(GitSignatureVerification {
            signer: self.signer,
            timestamp: self.timestamp,
            owner,
            owner_attestation,
        })
    }
}

impl GitOwnerAttestation {
    fn from_value(value: &Value, signer: PublicKey) -> Result<Self, ProjectWorkflowCodecError> {
        let values = value
            .as_array()
            .filter(|values| values.len() == 3)
            .ok_or_else(|| invalid_git("oa must be an array of exactly three strings"))?;
        let owner = PublicKey::from_hex(
            values[0]
                .as_str()
                .ok_or_else(|| invalid_git("oa owner must be a string"))?,
        )
        .map_err(|error| invalid_git(format!("invalid oa owner: {error}")))?;
        if owner == signer {
            return Err(invalid_git("owner attestation cannot be self-authored"));
        }
        let conditions = AttestationConditions::parse(
            values[1]
                .as_str()
                .ok_or_else(|| invalid_git("oa conditions must be a string"))?,
        )
        .map_err(|error| invalid_git(format!("invalid oa conditions: {error}")))?;
        let signature = EventSignature::from_hex(
            values[2]
                .as_str()
                .ok_or_else(|| invalid_git("oa signature must be a string"))?,
        )
        .map_err(|error| invalid_git(format!("invalid oa signature: {error}")))?;
        Ok(Self {
            owner,
            conditions,
            signature,
        })
    }
}

pub fn compute_git_signing_hash(
    timestamp: u32,
    owner_attestation: Option<&GitOwnerAttestation>,
    payload: &[u8],
) -> Result<[u8; 32], ProjectWorkflowCodecError> {
    if payload.len() > MAX_GIT_PAYLOAD_BYTES {
        return Err(ProjectWorkflowCodecError::GitPayloadTooLarge {
            actual: payload.len(),
            maximum: MAX_GIT_PAYLOAD_BYTES,
        });
    }
    let mut hash = Sha256::new();
    hash.update(GIT_SIGNING_DOMAIN);
    hash.update(timestamp.to_string().as_bytes());
    hash.update(b":");
    if let Some(owner) = owner_attestation {
        hash.update(owner.owner.to_hex().as_bytes());
        hash.update(b":");
        hash.update(owner.conditions.as_str().as_bytes());
        hash.update(b":");
        hash.update(owner.signature.to_hex().as_bytes());
        hash.update(b":");
    }
    hash.update(payload);
    let digest = hash.finalize();
    let mut output = [0; 32];
    output.copy_from_slice(&digest);
    Ok(output)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepositoryCoordinate {
    pub owner_hex: String,
    pub discriminator: String,
    pub relay_hint: Option<String>,
}

impl RepositoryCoordinate {
    pub fn parse_tag(tag: &[String]) -> Result<Self, ProjectWorkflowCodecError> {
        if !(2..=3).contains(&tag.len()) || tag.first().map(String::as_str) != Some("a") {
            return Err(invalid_envelope(
                "project member a tag must have two or three elements",
            ));
        }
        let mut parts = tag[1].splitn(3, ':');
        if parts.next() != Some("30617") {
            return Err(invalid_envelope("project member must reference kind 30617"));
        }
        let owner_hex = parts.next().unwrap_or_default();
        let discriminator = parts.next().unwrap_or_default();
        if !valid_lower_hex(owner_hex, 64) || discriminator.is_empty() {
            return Err(invalid_envelope("malformed repository coordinate"));
        }
        Ok(Self {
            owner_hex: owner_hex.to_owned(),
            discriminator: discriminator.to_owned(),
            relay_hint: tag.get(2).cloned(),
        })
    }

    pub fn coordinate(&self) -> String {
        format!("30617:{}:{}", self.owner_hex, self.discriminator)
    }

    pub fn to_tag(&self) -> Vec<String> {
        let mut tag = vec!["a".into(), self.coordinate()];
        if let Some(relay_hint) = &self.relay_hint {
            tag.push(relay_hint.clone());
        }
        tag
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectVisibility {
    Listed,
    Unlisted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectEnvelope {
    pub signer: PublicKey,
    pub slug: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub members: Vec<RepositoryCoordinate>,
    pub channel: Option<String>,
    pub visibility_value: Option<String>,
}

impl ProjectEnvelope {
    pub fn parse_signed_event(event: &SignedEvent) -> Result<Self, ProjectWorkflowCodecError> {
        verify_kind(event, KIND_PROJECT)?;
        let slug = single_tag_value(&event.event.tags, "d", false, 1_024)?;
        let name = optional_tag_value(&event.event.tags, "name", 256)?;
        let description = optional_tag_value(&event.event.tags, "description", 2_048)?;
        let channel = optional_tag_value(&event.event.tags, "buzz-channel", 256)?;
        let visibility_value = optional_tag_value(&event.event.tags, "buzz-visibility", 256)?;
        let member_tags = event
            .event
            .tags
            .iter()
            .filter(|tag| tag.first().map(String::as_str) == Some("a"))
            .collect::<Vec<_>>();
        if member_tags.len() > 64 {
            return Err(invalid_envelope("project exceeds 64 member tags"));
        }
        let mut seen = HashSet::new();
        let mut members = Vec::with_capacity(member_tags.len());
        for tag in member_tags {
            let member = RepositoryCoordinate::parse_tag(tag)?;
            if !seen.insert(member.coordinate()) {
                return Err(invalid_envelope("duplicate project member coordinate"));
            }
            members.push(member);
        }
        Ok(Self {
            signer: event.event.public_key,
            slug,
            name,
            description,
            members,
            channel,
            visibility_value,
        })
    }

    pub fn visibility(&self) -> ProjectVisibility {
        if self.visibility_value.as_deref() == Some("unlisted") {
            ProjectVisibility::Unlisted
        } else {
            ProjectVisibility::Listed
        }
    }

    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.slug)
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        let mut tags = vec![vec!["d".into(), self.slug.clone()]];
        if let Some(name) = &self.name {
            tags.push(vec!["name".into(), name.clone()]);
        }
        if let Some(description) = &self.description {
            tags.push(vec!["description".into(), description.clone()]);
        }
        tags.extend(self.members.iter().map(RepositoryCoordinate::to_tag));
        if let Some(channel) = &self.channel {
            tags.push(vec!["buzz-channel".into(), channel.clone()]);
        }
        if let Some(visibility) = &self.visibility_value {
            tags.push(vec!["buzz-visibility".into(), visibility.clone()]);
        }
        tags
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceProfileCommand {
    pub actor: PublicKey,
    pub icon: Option<String>,
}

impl WorkspaceProfileCommand {
    pub fn parse_signed_event(event: &SignedEvent) -> Result<Self, ProjectWorkflowCodecError> {
        verify_kind(event, RELAY_ADMIN_SET_WORKSPACE_PROFILE)?;
        if !event.event.content.is_empty() {
            return Err(invalid_envelope("workspace-profile content must be empty"));
        }
        let icon = optional_tag_value(&event.event.tags, "icon", 98_304)?;
        if let Some(icon) = icon.as_deref() {
            validate_workspace_icon(icon)?;
        }
        Ok(Self {
            actor: event.event.public_key,
            icon: icon.filter(|value| !value.is_empty()),
        })
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        vec![vec!["icon".into(), self.icon.clone().unwrap_or_default()]]
    }
}

fn validate_workspace_icon(icon: &str) -> Result<(), ProjectWorkflowCodecError> {
    if icon.is_empty() {
        return Ok(());
    }
    if icon
        .chars()
        .any(|character| character.is_control() || character.is_whitespace())
    {
        return Err(invalid_envelope(
            "workspace icon contains invalid characters",
        ));
    }
    if icon.starts_with("data:image/") {
        if icon.len() > 98_304 {
            return Err(invalid_envelope("workspace icon data URL exceeds 96 KiB"));
        }
        return Ok(());
    }
    if !(icon.starts_with("https://") || icon.starts_with("http://")) {
        return Err(invalid_envelope(
            "workspace icon must be http(s) or data:image/*",
        ));
    }
    if icon.len() > 2_048 {
        return Err(invalid_envelope("workspace icon URL exceeds 2048 bytes"));
    }
    Ok(())
}

fn verify_kind(event: &SignedEvent, kind: u32) -> Result<(), ProjectWorkflowCodecError> {
    verify_signed_event(event, TimestampPolicy::Historical)
        .map_err(|error| invalid_envelope(format!("invalid signed event: {error}")))?;
    if u32::from(event.event.kind) != kind {
        return Err(ProjectWorkflowCodecError::UnsupportedKind(event.event.kind));
    }
    Ok(())
}

fn single_tag_value(
    tags: &[Vec<String>],
    name: &str,
    allow_empty: bool,
    maximum_bytes: usize,
) -> Result<String, ProjectWorkflowCodecError> {
    optional_tag_value(tags, name, maximum_bytes)?
        .filter(|value| allow_empty || !value.is_empty())
        .ok_or_else(|| invalid_envelope(format!("missing or empty {name} tag")))
}

fn optional_tag_value(
    tags: &[Vec<String>],
    name: &str,
    maximum_bytes: usize,
) -> Result<Option<String>, ProjectWorkflowCodecError> {
    let mut matching = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some(name));
    let Some(tag) = matching.next() else {
        return Ok(None);
    };
    if matching.next().is_some() {
        return Err(invalid_envelope(format!("duplicate {name} tag")));
    }
    if tag.len() != 2 {
        return Err(invalid_envelope(format!("malformed {name} tag")));
    }
    if tag[1].len() > maximum_bytes {
        return Err(invalid_envelope(format!(
            "{name} tag exceeds {maximum_bytes} bytes"
        )));
    }
    Ok(Some(tag[1].clone()))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a str, ProjectWorkflowCodecError> {
    object
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_git(format!("{name} must be a string")))
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_strict_json(bytes: &[u8]) -> Result<Value, ProjectWorkflowCodecError> {
    struct StrictValue;
    impl<'de> DeserializeSeed<'de> for StrictValue {
        type Value = Value;

        fn deserialize<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            deserializer.deserialize_any(self)
        }
    }

    impl<'de> Visitor<'de> for StrictValue {
        type Value = Value;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("JSON with unique object keys")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
            Ok(Value::Bool(value))
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
            Ok(Value::Number(value.into()))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
            Ok(Value::Number(value.into()))
        }

        fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
            serde_json::Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| E::custom("non-finite number"))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
            Ok(Value::String(value.to_owned()))
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
            Ok(Value::String(value))
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(Value::Null)
        }

        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(Value::Null)
        }

        fn visit_some<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            deserializer.deserialize_any(self)
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let mut values = Vec::new();
            while let Some(value) = sequence.next_element_seed(StrictValue)? {
                values.push(value);
            }
            Ok(Value::Array(values))
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut seen = HashSet::new();
            let mut values = Map::new();
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
        .map_err(|error| invalid_git(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| invalid_git(error.to_string()))?;
    Ok(value)
}

fn invalid_envelope(reason: impl Into<String>) -> ProjectWorkflowCodecError {
    ProjectWorkflowCodecError::InvalidEnvelope(reason.into())
}

fn invalid_git(reason: impl Into<String>) -> ProjectWorkflowCodecError {
    ProjectWorkflowCodecError::InvalidGitSignature(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CanonicalEvent;
    use secp256k1::{Keypair, SecretKey};

    const SIGNER_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 1;
        secret
    };
    const SIGNER: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const GIT_SIGNER: &str = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9";
    const GIT_SIGNATURE: &str = "c35062148d95b820068c18ab9cf69a8dd2322c606890366d084df7617570b96b7a1aca0a8fcabb2eb4032ebbdf5b43e6bf8633e0d85bcecce28a9e08705b875f";
    const GIT_ARMOR: &str = "-----BEGIN SIGNED MESSAGE-----\neyJ2IjoxLCJwayI6ImY5MzA4YTAxOTI1OGMzMTA0OTM0NGY4NWY4OWQ1MjI5YjUzMWM4NDU4MzZmOTliMDg2MDFmMTEzYmNlMDM2ZjkiLCJzaWciOiJjMzUwNjIxNDhkOTViODIwMDY4YzE4YWI5Y2Y2OWE4ZGQyMzIyYzYwNjg5MDM2NmQwODRkZjc2MTc1NzBiOTZiN2ExYWNhMGE4ZmNhYmIyZWI0MDMyZWJiZGY1YjQzZTZiZjg2MzNlMGQ4NWJjZWNjZTI4YTllMDg3MDViODc1ZiIsInQiOjE3MDAwMDAwMDB9\n-----END SIGNED MESSAGE-----\n";
    const GIT_PAYLOAD: &[u8] = b"tree 4b825dc642cb6eb9a060e54bf899d69f7cb46101\nauthor Test User <test@example.com> 1700000000 +0000\ncommitter Test User <test@example.com> 1700000000 +0000\n\nInitial commit";

    fn key(value: &str) -> PublicKey {
        PublicKey::from_hex(value).expect("fixture public key")
    }

    fn sign(event: CanonicalEvent) -> SignedEvent {
        let claimed_id = event.event_id().expect("event id");
        let secret = SecretKey::from_slice(&SIGNER_SECRET).expect("secret");
        let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret);
        let signature = Secp256k1::new()
            .sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
        SignedEvent {
            claimed_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
        }
    }

    #[test]
    fn git_signature_matches_published_hash_armor_and_signature_vector() {
        let envelope = GitSignatureEnvelope::parse_armored(GIT_ARMOR).expect("published armor");
        assert_eq!(envelope.signer, key(GIT_SIGNER));
        assert_eq!(envelope.signature.to_hex(), GIT_SIGNATURE);
        assert_eq!(envelope.to_armored(), GIT_ARMOR);
        assert_eq!(
            hex::encode(envelope.signing_hash(GIT_PAYLOAD).expect("hash")),
            "a11a32173aa35125aaefaad8854f2eda5a144268a4a355905c841f79ff44aa18"
        );
        assert_eq!(
            envelope.verify(GIT_PAYLOAD).expect("verified"),
            GitSignatureVerification {
                signer: key(GIT_SIGNER),
                timestamp: 1_700_000_000,
                owner: None,
                owner_attestation: OwnerAttestationStatus::Absent,
            }
        );
    }

    #[test]
    fn git_owner_attestation_vector_is_bound_and_reported_separately() {
        let json = format!(
            r#"{{"v":1,"pk":"{GIT_SIGNER}","sig":"{}","t":1700000000,"oa":["{SIGNER}","","{}"]}}"#,
            "15592857980b8656ff50303d86acaffcbda397b9c0bb40aebd2fb87a723e466fdb1a74404d39f9eb7ac220b4f2e061f27523f1af24cbdf991cf42ff9b47034c0",
            "54b97dfd2b7d61c1bc1b5facab9d12a991fe0ac3dcb9044b3176f63bebb6f67340eb0ad866f2d5568b78b58ba234ee9f490f8c41e64a949c200315801520ed25",
        );
        let armored = format!("{ARMOR_BEGIN}\n{}\n{ARMOR_END}\n", STANDARD.encode(json));
        let envelope = GitSignatureEnvelope::parse_armored(&armored).expect("owner armor");
        assert_eq!(
            hex::encode(envelope.signing_hash(GIT_PAYLOAD).expect("hash")),
            "b61f1658836a4f63a2d2f5d621014a064435dde0765dd9c1dc79c9530fe879f0"
        );
        let verified = envelope.verify(GIT_PAYLOAD).expect("commit signature");
        assert_eq!(verified.owner, Some(key(SIGNER)));
        assert_eq!(verified.owner_attestation, OwnerAttestationStatus::Valid);

        let reordered = format!(
            "{ARMOR_BEGIN}\n{}\n{ARMOR_END}\n",
            STANDARD.encode(format!(
                r#"{{"pk":"{GIT_SIGNER}","v":1,"sig":"{GIT_SIGNATURE}","t":1700000000}}"#
            ))
        );
        assert!(GitSignatureEnvelope::parse_armored(&reordered).is_err());
        let duplicate = format!(
            "{ARMOR_BEGIN}\n{}\n{ARMOR_END}\n",
            STANDARD.encode(format!(
                r#"{{"v":1,"v":1,"pk":"{GIT_SIGNER}","sig":"{GIT_SIGNATURE}","t":1700000000}}"#
            ))
        );
        assert!(GitSignatureEnvelope::parse_armored(&duplicate).is_err());
    }

    #[test]
    fn project_codec_matches_every_nip_mp_ingest_fixture() {
        let fixtures: Value = serde_json::from_str(include_str!(
            "../../../../projects/buzz/docs/nips/NIP-MP.fixtures.json"
        ))
        .expect("fixtures");
        let cases = fixtures["cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 31);
        for case in cases {
            let template = &case["template"];
            let tags: Vec<Vec<String>> =
                serde_json::from_value(template["tags"].clone()).expect("tags");
            let event = sign(CanonicalEvent::new(
                key(SIGNER),
                1_700_000_000,
                template["kind"].as_u64().expect("kind") as u16,
                tags,
                template["content"].as_str().expect("content").into(),
            ));
            let result = ProjectEnvelope::parse_signed_event(&event);
            let accepted = case["expect"].as_str() == Some("accept");
            assert_eq!(
                result.is_ok(),
                accepted,
                "fixture {}",
                case["name"].as_str().unwrap_or("unnamed")
            );
        }
    }

    #[test]
    fn project_coordinates_preserve_colons_and_never_confer_repository_authority() {
        let event = sign(CanonicalEvent::new(
            key(SIGNER),
            1_700_000_000,
            KIND_PROJECT as u16,
            vec![
                vec!["d".into(), "platform".into()],
                vec![
                    "a".into(),
                    format!("30617:{}:repo:with:colons", "a".repeat(64)),
                    "not a parsed relay hint".into(),
                ],
                vec!["buzz-visibility".into(), "future-token".into()],
                vec!["future-metadata".into(), "retained-by-signer".into()],
            ],
            "future content is ignored".into(),
        ));
        let project = ProjectEnvelope::parse_signed_event(&event).expect("project");
        assert_eq!(project.members[0].discriminator, "repo:with:colons");
        assert_eq!(project.visibility(), ProjectVisibility::Listed);
        assert_eq!(project.display_name(), "platform");
        assert_eq!(project.signer, key(SIGNER));
    }

    #[test]
    fn workspace_profile_codec_validates_image_sinks_and_clear_shape() {
        let event = sign(CanonicalEvent::new(
            key(SIGNER),
            1_700_000_000,
            RELAY_ADMIN_SET_WORKSPACE_PROFILE as u16,
            vec![vec![
                "icon".into(),
                "data:image/webp;base64,UklGRg==".into(),
            ]],
            String::new(),
        ));
        let command = WorkspaceProfileCommand::parse_signed_event(&event).expect("profile");
        assert_eq!(
            command.icon.as_deref(),
            Some("data:image/webp;base64,UklGRg==")
        );
        let clear = sign(CanonicalEvent::new(
            key(SIGNER),
            1_700_000_001,
            RELAY_ADMIN_SET_WORKSPACE_PROFILE as u16,
            Vec::new(),
            String::new(),
        ));
        assert_eq!(
            WorkspaceProfileCommand::parse_signed_event(&clear)
                .expect("clear")
                .icon,
            None
        );
        let invalid = sign(CanonicalEvent::new(
            key(SIGNER),
            1_700_000_002,
            RELAY_ADMIN_SET_WORKSPACE_PROFILE as u16,
            vec![vec!["icon".into(), "javascript:alert(1)".into()]],
            String::new(),
        ));
        assert!(WorkspaceProfileCommand::parse_signed_event(&invalid).is_err());
    }
}
