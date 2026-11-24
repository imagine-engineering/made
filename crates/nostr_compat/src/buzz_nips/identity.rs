use crate::generated_kinds::{
    KIND_AUTH, KIND_IA_ARCHIVE_REQUEST, KIND_IA_ARCHIVED, KIND_IA_ARCHIVED_LIST,
    KIND_IA_UNARCHIVE_REQUEST, KIND_IA_UNARCHIVED,
};
use crate::{CanonicalEvent, EventId, EventSignature, PublicKey};
use secp256k1::{Message, Secp256k1, XOnlyPublicKey, schnorr::Signature};
use sha2::{Digest as _, Sha256};

const MAX_REASON_BYTES: usize = 64;
const MAX_REQUEST_CONTENT_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum IdentityCodecError {
    #[error("unsupported identity event kind {0}")]
    UnsupportedKind(u16),
    #[error("missing required {0} tag")]
    MissingTag(&'static str),
    #[error("expected exactly one {0} tag")]
    DuplicateTag(&'static str),
    #[error("malformed {tag} tag: {reason}")]
    MalformedTag { tag: &'static str, reason: String },
    #[error("invalid attestation conditions: {0}")]
    InvalidConditions(String),
    #[error("invalid owner attestation: {0}")]
    InvalidAttestation(String),
    #[error("owner attestation condition is not satisfied")]
    ConditionNotSatisfied,
    #[error("request content is {actual} bytes, maximum is {maximum}")]
    ContentTooLarge { actual: usize, maximum: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttestationCondition {
    Kind(u16),
    CreatedBefore(u32),
    CreatedAfter(u32),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttestationConditions {
    raw: String,
    clauses: Vec<AttestationCondition>,
}

impl AttestationConditions {
    pub fn parse(raw: &str) -> Result<Self, IdentityCodecError> {
        if raw.is_empty() {
            return Ok(Self {
                raw: String::new(),
                clauses: Vec::new(),
            });
        }
        if !raw.is_ascii() || raw.bytes().any(|byte| byte.is_ascii_whitespace()) {
            return Err(IdentityCodecError::InvalidConditions(
                "conditions must be ASCII without whitespace".into(),
            ));
        }
        let mut clauses = Vec::new();
        for clause in raw.split('&') {
            if clause.is_empty() {
                return Err(IdentityCodecError::InvalidConditions(
                    "conditions contain an empty clause".into(),
                ));
            }
            let parsed = if let Some(value) = clause.strip_prefix("kind=") {
                AttestationCondition::Kind(parse_decimal(value, u16::MAX as u64)? as u16)
            } else if let Some(value) = clause.strip_prefix("created_at<") {
                AttestationCondition::CreatedBefore(parse_decimal(value, u32::MAX as u64)? as u32)
            } else if let Some(value) = clause.strip_prefix("created_at>") {
                AttestationCondition::CreatedAfter(parse_decimal(value, u32::MAX as u64)? as u32)
            } else {
                return Err(IdentityCodecError::InvalidConditions(format!(
                    "unsupported clause {clause:?}"
                )));
            };
            clauses.push(parsed);
        }
        Ok(Self {
            raw: raw.to_owned(),
            clauses,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn clauses(&self) -> &[AttestationCondition] {
        &self.clauses
    }

    fn event_matches(&self, kind: u16, created_at: u64) -> bool {
        self.clauses.iter().all(|clause| match clause {
            AttestationCondition::Kind(required) => kind == *required,
            AttestationCondition::CreatedBefore(limit) => created_at < u64::from(*limit),
            AttestationCondition::CreatedAfter(limit) => created_at > u64::from(*limit),
        })
    }

    fn timestamp_matches(&self, created_at: u64) -> bool {
        self.clauses.iter().all(|clause| match clause {
            AttestationCondition::Kind(_) => true,
            AttestationCondition::CreatedBefore(limit) => created_at < u64::from(*limit),
            AttestationCondition::CreatedAfter(limit) => created_at > u64::from(*limit),
        })
    }
}

fn parse_decimal(value: &str, maximum: u64) -> Result<u64, IdentityCodecError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(IdentityCodecError::InvalidConditions(format!(
            "non-canonical decimal {value:?}"
        )));
    }
    let parsed = value.parse::<u64>().map_err(|error| {
        IdentityCodecError::InvalidConditions(format!("invalid decimal {value:?}: {error}"))
    })?;
    if parsed > maximum {
        return Err(IdentityCodecError::InvalidConditions(format!(
            "decimal {value:?} exceeds {maximum}"
        )));
    }
    Ok(parsed)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnerAttestation {
    pub owner: PublicKey,
    pub conditions: AttestationConditions,
    pub signature: EventSignature,
}

impl OwnerAttestation {
    pub fn parse_tag(tag: &[String]) -> Result<Self, IdentityCodecError> {
        if tag.len() != 4 || tag.first().map(String::as_str) != Some("auth") {
            return Err(malformed("auth", "expected exactly four elements"));
        }
        let owner =
            PublicKey::from_hex(&tag[1]).map_err(|error| malformed("auth", error.to_string()))?;
        let conditions = AttestationConditions::parse(&tag[2])?;
        let signature = EventSignature::from_hex(&tag[3])
            .map_err(|error| malformed("auth", error.to_string()))?;
        Ok(Self {
            owner,
            conditions,
            signature,
        })
    }

    pub fn to_tag(&self) -> Vec<String> {
        vec![
            "auth".into(),
            self.owner.to_hex(),
            self.conditions.as_str().into(),
            self.signature.to_hex(),
        ]
    }

    pub fn verify_for_event(&self, event: &CanonicalEvent) -> Result<(), IdentityCodecError> {
        self.verify_signature(&event.public_key)?;
        if !self.conditions.event_matches(event.kind, event.created_at) {
            return Err(IdentityCodecError::ConditionNotSatisfied);
        }
        Ok(())
    }

    pub fn verify_for_membership(
        &self,
        agent: &PublicKey,
        auth_created_at: u64,
    ) -> Result<(), IdentityCodecError> {
        self.verify_signature(agent)?;
        if !self.conditions.timestamp_matches(auth_created_at) {
            return Err(IdentityCodecError::ConditionNotSatisfied);
        }
        Ok(())
    }

    pub fn verify_identity_binding(
        &self,
        actor: &PublicKey,
        target: &PublicKey,
        request_created_at: Option<u64>,
    ) -> Result<(), IdentityCodecError> {
        if self.owner != *actor {
            return Err(IdentityCodecError::InvalidAttestation(
                "attesting owner does not match request actor".into(),
            ));
        }
        self.verify_signature(target)?;
        if request_created_at
            .is_some_and(|created_at| !self.conditions.timestamp_matches(created_at))
        {
            return Err(IdentityCodecError::ConditionNotSatisfied);
        }
        Ok(())
    }

    fn verify_signature(&self, agent: &PublicKey) -> Result<(), IdentityCodecError> {
        if self.owner == *agent {
            return Err(IdentityCodecError::InvalidAttestation(
                "self-attestation is prohibited".into(),
            ));
        }
        let preimage = format!(
            "nostr:agent-auth:{}:{}",
            agent.to_hex(),
            self.conditions.as_str()
        );
        let digest = Sha256::digest(preimage.as_bytes());
        let mut message_bytes = [0; 32];
        message_bytes.copy_from_slice(&digest);
        let message = Message::from_digest(message_bytes);
        let owner = XOnlyPublicKey::from_slice(self.owner.as_bytes()).map_err(|_| {
            IdentityCodecError::InvalidAttestation("invalid owner public key".into())
        })?;
        let signature = Signature::from_slice(self.signature.as_bytes()).map_err(|_| {
            IdentityCodecError::InvalidAttestation("invalid Schnorr signature".into())
        })?;
        Secp256k1::verification_only()
            .verify_schnorr(&signature, &message, &owner)
            .map_err(|_| {
                IdentityCodecError::InvalidAttestation(
                    "owner signature does not verify for agent".into(),
                )
            })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentAuthentication {
    pub relay: String,
    pub challenge: String,
    pub attestation: OwnerAttestation,
}

impl AgentAuthentication {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, IdentityCodecError> {
        if u32::from(event.kind) != KIND_AUTH {
            return Err(IdentityCodecError::UnsupportedKind(event.kind));
        }
        let relay = parse_single_text_tag(&event.tags, "relay", false)?;
        let challenge = parse_single_text_tag(&event.tags, "challenge", false)?;
        let auth = single_tag(&event.tags, "auth")?;
        let attestation = OwnerAttestation::parse_tag(auth)?;
        Ok(Self {
            relay,
            challenge,
            attestation,
        })
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        vec![
            vec!["relay".into(), self.relay.clone()],
            vec!["challenge".into(), self.challenge.clone()],
            self.attestation.to_tag(),
        ]
    }

    pub fn verify_attestation(&self, event: &CanonicalEvent) -> Result<(), IdentityCodecError> {
        self.attestation
            .verify_for_membership(&event.public_key, event.created_at)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveAction {
    Archive,
    Unarchive,
}

impl ArchiveAction {
    fn request_kind(self) -> u16 {
        match self {
            Self::Archive => KIND_IA_ARCHIVE_REQUEST as u16,
            Self::Unarchive => KIND_IA_UNARCHIVE_REQUEST as u16,
        }
    }

    fn delta_kind(self) -> u16 {
        match self {
            Self::Archive => KIND_IA_ARCHIVED as u16,
            Self::Unarchive => KIND_IA_UNARCHIVED as u16,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityArchiveRequest {
    pub action: ArchiveAction,
    pub target: PublicKey,
    pub reason: Option<String>,
    pub replaced_by: Option<PublicKey>,
    pub attestation: Option<OwnerAttestation>,
    pub content: String,
}

impl IdentityArchiveRequest {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, IdentityCodecError> {
        let action = match u32::from(event.kind) {
            KIND_IA_ARCHIVE_REQUEST => ArchiveAction::Archive,
            KIND_IA_UNARCHIVE_REQUEST => ArchiveAction::Unarchive,
            _ => return Err(IdentityCodecError::UnsupportedKind(event.kind)),
        };
        require_protected_marker(&event.tags)?;
        validate_content(&event.content)?;
        let target = parse_single_public_key_tag(&event.tags, "p")?;
        let reason = parse_optional_reason(&event.tags)?;
        let replaced_by = optional_tag(&event.tags, "replaced-by")?
            .map(|tag| parse_public_key_tag(tag, "replaced-by"))
            .transpose()?;
        if action == ArchiveAction::Unarchive && replaced_by.is_some() {
            return Err(malformed(
                "replaced-by",
                "unarchive requests cannot carry a replacement",
            ));
        }
        if replaced_by == Some(target) {
            return Err(malformed("replaced-by", "replacement equals target"));
        }
        let attestation = optional_tag(&event.tags, "auth")?
            .map(OwnerAttestation::parse_tag)
            .transpose()?;
        Ok(Self {
            action,
            target,
            reason,
            replaced_by,
            attestation,
            content: event.content.clone(),
        })
    }

    pub fn to_event(&self, actor: PublicKey, created_at: u64) -> CanonicalEvent {
        let mut tags = vec![vec!["-".into()], vec!["p".into(), self.target.to_hex()]];
        if let Some(reason) = &self.reason {
            tags.push(vec!["reason".into(), reason.clone()]);
        }
        if let Some(replaced_by) = self.replaced_by {
            tags.push(vec!["replaced-by".into(), replaced_by.to_hex()]);
        }
        if let Some(attestation) = &self.attestation {
            tags.push(attestation.to_tag());
        }
        CanonicalEvent::new(
            actor,
            created_at,
            self.action.request_kind(),
            tags,
            self.content.clone(),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsentKind {
    SelfSigned,
    Owner,
    Admin,
    Relay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveConsent {
    pub kind: ConsentKind,
    pub actor: Option<PublicKey>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfileProof {
    pub event_id: EventId,
    pub target: PublicKey,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityArchiveDelta {
    pub action: ArchiveAction,
    pub target: PublicKey,
    pub consent: ArchiveConsent,
    pub request_id: Option<EventId>,
    pub profile_proof: Option<ProfileProof>,
    pub reason: Option<String>,
    pub replaced_by: Option<PublicKey>,
    pub content: String,
}

impl IdentityArchiveDelta {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, IdentityCodecError> {
        let action = match u32::from(event.kind) {
            KIND_IA_ARCHIVED => ArchiveAction::Archive,
            KIND_IA_UNARCHIVED => ArchiveAction::Unarchive,
            _ => return Err(IdentityCodecError::UnsupportedKind(event.kind)),
        };
        require_protected_marker(&event.tags)?;
        let target = parse_single_public_key_tag(&event.tags, "p")?;
        let consent = parse_consent(single_tag(&event.tags, "consent")?, target)?;
        let (request_id, profile_proof) = parse_delta_references(&event.tags, target)?;
        if consent.kind != ConsentKind::Relay && request_id.is_none() {
            return Err(IdentityCodecError::MissingTag("e"));
        }
        let reason = parse_optional_reason(&event.tags)?;
        let replaced_by = optional_tag(&event.tags, "replaced-by")?
            .map(|tag| parse_public_key_tag(tag, "replaced-by"))
            .transpose()?;
        if action == ArchiveAction::Unarchive && replaced_by.is_some() {
            return Err(malformed(
                "replaced-by",
                "unarchive deltas cannot carry a replacement",
            ));
        }
        if replaced_by == Some(target) {
            return Err(malformed("replaced-by", "replacement equals target"));
        }
        Ok(Self {
            action,
            target,
            consent,
            request_id,
            profile_proof,
            reason,
            replaced_by,
            content: event.content.clone(),
        })
    }

    pub fn to_event(&self, relay: PublicKey, created_at: u64) -> CanonicalEvent {
        let mut tags = vec![vec!["-".into()], vec!["p".into(), self.target.to_hex()]];
        tags.push(self.consent.to_tag());
        if let Some(request_id) = self.request_id {
            tags.push(vec!["e".into(), request_id.to_hex()]);
        }
        if let Some(proof) = self.profile_proof {
            tags.push(vec![
                "e".into(),
                proof.event_id.to_hex(),
                String::new(),
                "proof".into(),
                proof.target.to_hex(),
            ]);
        }
        if let Some(reason) = &self.reason {
            tags.push(vec!["reason".into(), reason.clone()]);
        }
        if let Some(replaced_by) = self.replaced_by {
            tags.push(vec!["replaced-by".into(), replaced_by.to_hex()]);
        }
        CanonicalEvent::new(
            relay,
            created_at,
            self.action.delta_kind(),
            tags,
            self.content.clone(),
        )
    }
}

impl ArchiveConsent {
    fn to_tag(self) -> Vec<String> {
        let name = match self.kind {
            ConsentKind::SelfSigned => "self",
            ConsentKind::Owner => "owner",
            ConsentKind::Admin => "admin",
            ConsentKind::Relay => "relay",
        };
        let mut tag = vec!["consent".into(), name.into()];
        if let Some(actor) = self.actor {
            tag.push(actor.to_hex());
        }
        tag
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchivedIdentitySnapshot {
    pub identities: Vec<PublicKey>,
}

impl ArchivedIdentitySnapshot {
    pub fn parse_event(event: &CanonicalEvent) -> Result<Self, IdentityCodecError> {
        if u32::from(event.kind) != KIND_IA_ARCHIVED_LIST {
            return Err(IdentityCodecError::UnsupportedKind(event.kind));
        }
        require_protected_marker(&event.tags)?;
        if !event.content.is_empty() {
            return Err(malformed("content", "snapshot content must be empty"));
        }
        let identities = event
            .tags
            .iter()
            .filter(|tag| tag.first().map(String::as_str) == Some("p"))
            .filter_map(|tag| tag.get(1))
            .filter_map(|value| PublicKey::from_hex(value).ok())
            .collect();
        Ok(Self { identities })
    }

    pub fn to_event(&self, relay: PublicKey, created_at: u64) -> CanonicalEvent {
        let mut tags = vec![vec!["-".into()]];
        tags.extend(
            self.identities
                .iter()
                .map(|identity| vec!["p".into(), identity.to_hex()]),
        );
        CanonicalEvent::new(
            relay,
            created_at,
            KIND_IA_ARCHIVED_LIST as u16,
            tags,
            String::new(),
        )
    }
}

fn parse_consent(tag: &[String], target: PublicKey) -> Result<ArchiveConsent, IdentityCodecError> {
    let Some(path) = tag.get(1).map(String::as_str) else {
        return Err(malformed("consent", "missing consent path"));
    };
    let actor = tag
        .get(2)
        .map(|value| {
            PublicKey::from_hex(value).map_err(|error| malformed("consent", error.to_string()))
        })
        .transpose()?;
    if tag.len() > 3 {
        return Err(malformed("consent", "too many elements"));
    }
    let kind = match path {
        "self" => {
            if actor != Some(target) {
                return Err(malformed("consent", "self actor must equal target"));
            }
            ConsentKind::SelfSigned
        }
        "owner" => {
            if actor.is_none() {
                return Err(malformed("consent", "owner actor is required"));
            }
            ConsentKind::Owner
        }
        "admin" => {
            if actor.is_none() {
                return Err(malformed("consent", "admin actor is required"));
            }
            ConsentKind::Admin
        }
        "relay" => ConsentKind::Relay,
        _ => return Err(malformed("consent", "unknown consent path")),
    };
    Ok(ArchiveConsent { kind, actor })
}

fn parse_delta_references(
    tags: &[Vec<String>],
    target: PublicKey,
) -> Result<(Option<EventId>, Option<ProfileProof>), IdentityCodecError> {
    let mut request_id = None;
    let mut profile_proof = None;
    for tag in tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some("e"))
    {
        if tag.len() == 2 {
            if request_id.is_some() {
                return Err(IdentityCodecError::DuplicateTag("e request"));
            }
            request_id = Some(
                EventId::from_hex(&tag[1]).map_err(|error| malformed("e", error.to_string()))?,
            );
        } else if tag.len() == 5 && tag[2].is_empty() && tag[3] == "proof" {
            if profile_proof.is_some() {
                return Err(IdentityCodecError::DuplicateTag("e proof"));
            }
            let proof_target =
                PublicKey::from_hex(&tag[4]).map_err(|error| malformed("e", error.to_string()))?;
            if proof_target != target {
                return Err(malformed("e", "profile proof target does not match delta"));
            }
            profile_proof = Some(ProfileProof {
                event_id: EventId::from_hex(&tag[1])
                    .map_err(|error| malformed("e", error.to_string()))?,
                target: proof_target,
            });
        } else {
            return Err(malformed("e", "unsupported archive-delta reference"));
        }
    }
    Ok((request_id, profile_proof))
}

fn parse_optional_reason(tags: &[Vec<String>]) -> Result<Option<String>, IdentityCodecError> {
    let Some(tag) = optional_tag(tags, "reason")? else {
        return Ok(None);
    };
    if tag.len() != 2 {
        return Err(malformed("reason", "expected exactly two elements"));
    }
    let reason = &tag[1];
    if reason.len() > MAX_REASON_BYTES || reason.chars().any(char::is_control) {
        return Err(malformed(
            "reason",
            "must be at most 64 UTF-8 bytes without control characters",
        ));
    }
    Ok(Some(reason.clone()))
}

fn validate_content(content: &str) -> Result<(), IdentityCodecError> {
    if content.len() > MAX_REQUEST_CONTENT_BYTES {
        return Err(IdentityCodecError::ContentTooLarge {
            actual: content.len(),
            maximum: MAX_REQUEST_CONTENT_BYTES,
        });
    }
    Ok(())
}

fn require_protected_marker(tags: &[Vec<String>]) -> Result<(), IdentityCodecError> {
    let markers: Vec<_> = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some("-"))
        .collect();
    match markers.as_slice() {
        [] => Err(IdentityCodecError::MissingTag("-")),
        [marker] if marker.len() == 1 => Ok(()),
        [_] => Err(malformed("-", "marker must contain one element")),
        _ => Err(IdentityCodecError::DuplicateTag("-")),
    }
}

fn parse_single_public_key_tag(
    tags: &[Vec<String>],
    name: &'static str,
) -> Result<PublicKey, IdentityCodecError> {
    parse_public_key_tag(single_tag(tags, name)?, name)
}

fn parse_public_key_tag(
    tag: &[String],
    name: &'static str,
) -> Result<PublicKey, IdentityCodecError> {
    if tag.len() != 2 {
        return Err(malformed(name, "expected exactly two elements"));
    }
    PublicKey::from_hex(&tag[1]).map_err(|error| malformed(name, error.to_string()))
}

fn parse_single_text_tag(
    tags: &[Vec<String>],
    name: &'static str,
    allow_empty: bool,
) -> Result<String, IdentityCodecError> {
    let tag = single_tag(tags, name)?;
    if tag.len() != 2 || (!allow_empty && tag[1].is_empty()) {
        return Err(malformed(name, "expected one non-empty value"));
    }
    Ok(tag[1].clone())
}

fn single_tag<'a>(
    tags: &'a [Vec<String>],
    name: &'static str,
) -> Result<&'a [String], IdentityCodecError> {
    optional_tag(tags, name)?.ok_or(IdentityCodecError::MissingTag(name))
}

fn optional_tag<'a>(
    tags: &'a [Vec<String>],
    name: &'static str,
) -> Result<Option<&'a [String]>, IdentityCodecError> {
    let mut matches = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some(name));
    let first = matches.next().map(Vec::as_slice);
    if matches.next().is_some() {
        return Err(IdentityCodecError::DuplicateTag(name));
    }
    Ok(first)
}

fn malformed(tag: &'static str, reason: impl Into<String>) -> IdentityCodecError {
    IdentityCodecError::MalformedTag {
        tag,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const AGENT: &str = "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5";
    const RELAY: &str = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9";
    const CONDITIONS: &str = "kind=1&created_at<1713957000";
    const AUTH_SIGNATURE: &str = "8b7df2575caf0a108374f8471722b233c53f9ff827a8b0f91861966c3b9dd5cb2e189eae9f49d72187674c2f5bd244145e10ff86c9f257ffe65a1ee5f108b369";

    fn key(value: &str) -> PublicKey {
        PublicKey::from_hex(value).expect("fixture public key")
    }

    fn auth_tag() -> Vec<String> {
        vec![
            "auth".into(),
            OWNER.into(),
            CONDITIONS.into(),
            AUTH_SIGNATURE.into(),
        ]
    }

    #[test]
    fn owner_attestation_vector_round_trips_and_preserves_context_rules() {
        let attestation = OwnerAttestation::parse_tag(&auth_tag()).expect("attestation");
        assert_eq!(attestation.to_tag(), auth_tag());
        let event = CanonicalEvent::new(
            key(AGENT),
            1_713_956_400,
            1,
            vec![auth_tag()],
            "owner-attested agent event".into(),
        );
        assert!(attestation.verify_for_event(&event).is_ok());
        assert!(
            attestation
                .verify_for_membership(&key(AGENT), event.created_at)
                .is_ok()
        );
        assert!(
            attestation
                .verify_identity_binding(&key(OWNER), &key(AGENT), None)
                .is_ok()
        );
        assert_eq!(
            AttestationConditions::parse("kind=01"),
            Err(IdentityCodecError::InvalidConditions(
                "non-canonical decimal \"01\"".into()
            ))
        );
    }

    #[test]
    fn agent_authentication_round_trips_and_rejects_duplicate_credentials() {
        let authentication = AgentAuthentication {
            relay: "wss://relay.example".into(),
            challenge: "nonce".into(),
            attestation: OwnerAttestation::parse_tag(&auth_tag()).expect("attestation"),
        };
        let event = CanonicalEvent::new(
            key(AGENT),
            1_713_956_400,
            KIND_AUTH as u16,
            authentication.to_tags(),
            String::new(),
        );
        assert_eq!(
            AgentAuthentication::parse_event(&event).expect("NIP-AA event"),
            authentication
        );
        assert!(authentication.verify_attestation(&event).is_ok());

        let mut duplicate = event;
        duplicate.tags.push(auth_tag());
        assert_eq!(
            AgentAuthentication::parse_event(&duplicate),
            Err(IdentityCodecError::DuplicateTag("auth"))
        );
    }

    #[test]
    fn archive_request_matches_nip_ia_vector_and_rejects_malformed_tags() {
        let event = CanonicalEvent::new(
            key(OWNER),
            1_713_956_400,
            KIND_IA_ARCHIVE_REQUEST as u16,
            vec![
                vec!["-".into()],
                vec!["p".into(), AGENT.into()],
                vec!["reason".into(), "bot-rebuilt".into()],
                auth_tag(),
            ],
            "Archiving zombie agent after rebuild.".into(),
        );
        assert_eq!(
            event.event_id().expect("vector id").to_hex(),
            "3eb98c5200ee3b0280471131c0e63b5a3a3b6049a3c51ee4f425e649a45389d8"
        );
        let request = IdentityArchiveRequest::parse_event(&event).expect("archive request");
        assert_eq!(request.to_event(key(OWNER), event.created_at), event);
        assert!(
            request
                .attestation
                .as_ref()
                .expect("owner proof")
                .verify_identity_binding(&key(OWNER), &key(AGENT), Some(event.created_at))
                .is_ok()
        );

        let mut missing_marker = event.clone();
        missing_marker.tags.remove(0);
        assert_eq!(
            IdentityArchiveRequest::parse_event(&missing_marker),
            Err(IdentityCodecError::MissingTag("-"))
        );
        let mut duplicate_target = event.clone();
        duplicate_target.tags.push(vec!["p".into(), OWNER.into()]);
        assert_eq!(
            IdentityArchiveRequest::parse_event(&duplicate_target),
            Err(IdentityCodecError::DuplicateTag("p"))
        );
        let mut invalid_auth = event;
        invalid_auth.tags[3][3] = "A".repeat(128);
        assert!(matches!(
            IdentityArchiveRequest::parse_event(&invalid_auth),
            Err(IdentityCodecError::MalformedTag { tag: "auth", .. })
        ));
    }

    #[test]
    fn archive_deltas_and_snapshot_match_nip_ia_vectors() {
        let request_id =
            EventId::from_hex("3eb98c5200ee3b0280471131c0e63b5a3a3b6049a3c51ee4f425e649a45389d8")
                .expect("request id");
        let delta = IdentityArchiveDelta {
            action: ArchiveAction::Archive,
            target: key(AGENT),
            consent: ArchiveConsent {
                kind: ConsentKind::Owner,
                actor: Some(key(OWNER)),
            },
            request_id: Some(request_id),
            profile_proof: None,
            reason: Some("bot-rebuilt".into()),
            replaced_by: None,
            content: "Archiving zombie agent after rebuild.".into(),
        };
        let event = delta.to_event(key(RELAY), 1_713_956_401);
        assert_eq!(
            event.event_id().expect("delta id").to_hex(),
            "cf4f9376861f90af3edcfabc8f6363e5e0894f0f1234592663352ec8977c4d86"
        );
        assert_eq!(
            IdentityArchiveDelta::parse_event(&event).expect("archive delta"),
            delta
        );

        let snapshot = ArchivedIdentitySnapshot {
            identities: vec![key(AGENT)],
        };
        let snapshot_event = snapshot.to_event(key(RELAY), 1_713_956_402);
        assert_eq!(
            snapshot_event.event_id().expect("snapshot id").to_hex(),
            "263a4e89f569146af145adea1630194a1f35e1290ae08b776d51237012cba9a7"
        );
        assert_eq!(
            ArchivedIdentitySnapshot::parse_event(&snapshot_event).expect("snapshot"),
            snapshot
        );
    }

    #[test]
    fn archive_codec_rejects_invalid_action_specific_and_consent_shapes() {
        let unarchive = CanonicalEvent::new(
            key(AGENT),
            1_713_956_500,
            KIND_IA_UNARCHIVE_REQUEST as u16,
            vec![
                vec!["-".into()],
                vec!["p".into(), AGENT.into()],
                vec!["replaced-by".into(), OWNER.into()],
            ],
            String::new(),
        );
        assert!(matches!(
            IdentityArchiveRequest::parse_event(&unarchive),
            Err(IdentityCodecError::MalformedTag {
                tag: "replaced-by",
                ..
            })
        ));

        let malformed_delta = CanonicalEvent::new(
            key(RELAY),
            1_713_956_501,
            KIND_IA_UNARCHIVED as u16,
            vec![
                vec!["-".into()],
                vec!["p".into(), AGENT.into()],
                vec!["consent".into(), "self".into(), OWNER.into()],
                vec!["e".into(), "a".repeat(64)],
            ],
            String::new(),
        );
        assert!(matches!(
            IdentityArchiveDelta::parse_event(&malformed_delta),
            Err(IdentityCodecError::MalformedTag { tag: "consent", .. })
        ));
    }
}
