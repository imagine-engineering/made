use std::collections::BTreeMap;
use std::fmt;

use nostr::nips::nip44::{self, Version};
use nostr::{RelayUrl, SecretKey};
use serde::Serialize;

use crate::PublicKey;
use crate::buzz_nips::agent_activity::{
    EngramBody, EngramEnvelope, MAX_AGENT_PLAINTEXT_BYTES, derive_engram_d_tag,
    nip44_conversation_key, validate_engram_slug,
};
use crate::dm::Nip44Ciphertext;
use crate::generated_kinds::KIND_AGENT_ENGRAM;

const MAX_RELAY_CANDIDATES: usize = 256;
const MAX_RELAY_URL_BYTES: usize = 2_048;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AgentMemoryCodecError {
    #[error("invalid engram secret key")]
    InvalidSecretKey,
    #[error("invalid engram coordinate")]
    InvalidCoordinate,
    #[error("engram participant identities must be distinct")]
    SameParticipant,
    #[error("the supplied key cannot read this engram")]
    WrongReader,
    #[error("the engram body is invalid")]
    InvalidBody,
    #[error("engram encryption failed")]
    EncryptionFailed,
    #[error("engram decryption failed")]
    DecryptionFailed,
    #[error("relay scope contains too many candidates")]
    TooManyRelays,
    #[error("relay scope contains no usable write relay")]
    NoUsableRelay,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EngramCoordinate {
    agent: PublicKey,
    owner: PublicKey,
    d_tag: String,
}

impl EngramCoordinate {
    pub fn derive(
        agent: PublicKey,
        owner: PublicKey,
        conversation_key: &[u8; 32],
        slug: &str,
    ) -> Result<Self, AgentMemoryCodecError> {
        validate_engram_slug(slug).map_err(|_| AgentMemoryCodecError::InvalidBody)?;
        Self::from_d_tag(agent, owner, derive_engram_d_tag(conversation_key, slug))
    }

    pub fn parse(value: &str, owner: PublicKey) -> Result<Self, AgentMemoryCodecError> {
        let mut parts = value.split(':');
        let kind = parts.next();
        let agent = parts.next();
        let d_tag = parts.next();
        if parts.next().is_some() || kind != Some("30174") || agent.is_none() || d_tag.is_none() {
            return Err(AgentMemoryCodecError::InvalidCoordinate);
        }
        let agent = PublicKey::from_hex(agent.unwrap_or_default())
            .map_err(|_| AgentMemoryCodecError::InvalidCoordinate)?;
        Self::from_d_tag(agent, owner, d_tag.unwrap_or_default().to_owned())
    }

    pub fn value(&self) -> String {
        format!("{}:{}:{}", KIND_AGENT_ENGRAM, self.agent, self.d_tag)
    }

    pub fn agent(&self) -> PublicKey {
        self.agent
    }

    pub fn owner(&self) -> PublicKey {
        self.owner
    }

    pub fn d_tag(&self) -> &str {
        &self.d_tag
    }

    fn from_d_tag(
        agent: PublicKey,
        owner: PublicKey,
        d_tag: String,
    ) -> Result<Self, AgentMemoryCodecError> {
        if agent == owner || !is_lower_hex_32(&d_tag) {
            return Err(if agent == owner {
                AgentMemoryCodecError::SameParticipant
            } else {
                AgentMemoryCodecError::InvalidCoordinate
            });
        }
        Ok(Self {
            agent,
            owner,
            d_tag,
        })
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct EncryptedEngram {
    coordinate: EngramCoordinate,
    ciphertext: Nip44Ciphertext,
}

impl EncryptedEngram {
    pub fn coordinate(&self) -> &EngramCoordinate {
        &self.coordinate
    }

    pub fn ciphertext(&self) -> &Nip44Ciphertext {
        &self.ciphertext
    }
}

impl fmt::Debug for EncryptedEngram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncryptedEngram")
            .field("coordinate", &self.coordinate)
            .field("ciphertext", &"<redacted>")
            .finish()
    }
}

pub fn encrypt_engram_for_owner(
    agent_secret: &[u8; 32],
    owner: PublicKey,
    body: &EngramBody,
) -> Result<EncryptedEngram, AgentMemoryCodecError> {
    let secret_key =
        SecretKey::from_slice(agent_secret).map_err(|_| AgentMemoryCodecError::InvalidSecretKey)?;
    let agent = public_key_for_secret(&secret_key);
    if agent == owner {
        return Err(AgentMemoryCodecError::SameParticipant);
    }
    let plaintext = encode_body(body)?;
    let owner_key = nostr_public_key(owner)?;
    let conversation_key = nip44_conversation_key(*agent_secret, owner)
        .map_err(|_| AgentMemoryCodecError::InvalidSecretKey)?;
    let coordinate = EngramCoordinate::derive(agent, owner, &conversation_key, body.slug())?;
    let ciphertext = nip44::encrypt(&secret_key, &owner_key, plaintext, Version::V2)
        .map_err(|_| AgentMemoryCodecError::EncryptionFailed)?;
    let ciphertext =
        Nip44Ciphertext::parse(ciphertext).map_err(|_| AgentMemoryCodecError::EncryptionFailed)?;
    Ok(EncryptedEngram {
        coordinate,
        ciphertext,
    })
}

pub fn decrypt_engram_as_owner(
    owner_secret: &[u8; 32],
    coordinate: &EngramCoordinate,
    ciphertext: &Nip44Ciphertext,
) -> Result<EngramBody, AgentMemoryCodecError> {
    decrypt_engram(
        owner_secret,
        coordinate.owner,
        coordinate.agent,
        coordinate,
        ciphertext,
    )
}

pub fn decrypt_engram_as_agent(
    agent_secret: &[u8; 32],
    coordinate: &EngramCoordinate,
    ciphertext: &Nip44Ciphertext,
) -> Result<EngramBody, AgentMemoryCodecError> {
    decrypt_engram(
        agent_secret,
        coordinate.agent,
        coordinate.owner,
        coordinate,
        ciphertext,
    )
}

fn decrypt_engram(
    reader_secret: &[u8; 32],
    expected_reader: PublicKey,
    counterparty: PublicKey,
    coordinate: &EngramCoordinate,
    ciphertext: &Nip44Ciphertext,
) -> Result<EngramBody, AgentMemoryCodecError> {
    let secret_key = SecretKey::from_slice(reader_secret)
        .map_err(|_| AgentMemoryCodecError::InvalidSecretKey)?;
    if public_key_for_secret(&secret_key) != expected_reader {
        return Err(AgentMemoryCodecError::WrongReader);
    }
    let counterparty_key = nostr_public_key(counterparty)?;
    let plaintext =
        nip44::decrypt_to_bytes(&secret_key, &counterparty_key, ciphertext.wire_value())
            .map_err(|_| AgentMemoryCodecError::DecryptionFailed)?;
    let conversation_key = nip44_conversation_key(*reader_secret, counterparty)
        .map_err(|_| AgentMemoryCodecError::InvalidSecretKey)?;
    EngramEnvelope {
        agent: coordinate.agent,
        owner: coordinate.owner,
        d_tag: coordinate.d_tag.clone(),
    }
    .validate_decrypted(&plaintext, &conversation_key)
    .map_err(|_| AgentMemoryCodecError::DecryptionFailed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentRelayAccess {
    Read,
    Write,
    ReadWrite,
}

impl AgentRelayAccess {
    fn permits_write(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentRelay {
    pub url: String,
    pub access: AgentRelayAccess,
}

impl AgentRelay {
    pub fn new(url: impl Into<String>, access: AgentRelayAccess) -> Self {
        Self {
            url: url.into(),
            access,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngramRelayScopeSource {
    AgentRelayList,
    OutOfBandFallback,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngramRelay {
    connection_url: String,
    canonical_url: String,
}

impl EngramRelay {
    pub fn connection_url(&self) -> &str {
        &self.connection_url
    }

    pub fn canonical_url(&self) -> &str {
        &self.canonical_url
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngramRelayScope {
    source: EngramRelayScopeSource,
    relays: Vec<EngramRelay>,
}

impl EngramRelayScope {
    pub fn resolve(
        advertised: &[AgentRelay],
        fallback: &[String],
    ) -> Result<Self, AgentMemoryCodecError> {
        if advertised.len() > MAX_RELAY_CANDIDATES || fallback.len() > MAX_RELAY_CANDIDATES {
            return Err(AgentMemoryCodecError::TooManyRelays);
        }
        let advertised = advertised
            .iter()
            .filter(|relay| relay.access.permits_write())
            .map(|relay| relay.url.as_str());
        let relays = canonical_relays(advertised);
        if !relays.is_empty() {
            return Ok(Self {
                source: EngramRelayScopeSource::AgentRelayList,
                relays,
            });
        }
        let relays = canonical_relays(fallback.iter().map(String::as_str));
        if relays.is_empty() {
            return Err(AgentMemoryCodecError::NoUsableRelay);
        }
        Ok(Self {
            source: EngramRelayScopeSource::OutOfBandFallback,
            relays,
        })
    }

    pub fn source(&self) -> EngramRelayScopeSource {
        self.source
    }

    pub fn relays(&self) -> &[EngramRelay] {
        &self.relays
    }

    pub fn rotate_to(&self, next: &Self) -> EngramRelayRotation {
        let previous = owned_relay_map(&self.relays);
        let current = owned_relay_map(&next.relays);
        let departing = previous
            .iter()
            .filter(|(canonical, _)| !current.contains_key(*canonical))
            .map(|(_, relay)| relay.clone())
            .collect();
        let added = current
            .iter()
            .filter(|(canonical, _)| !previous.contains_key(*canonical))
            .map(|(_, relay)| relay.clone())
            .collect();
        let mut publication_targets = previous;
        for (canonical, relay) in current {
            publication_targets.entry(canonical).or_insert(relay);
        }
        EngramRelayRotation {
            departing,
            added,
            publication_targets: publication_targets.into_values().collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngramRelayRotation {
    departing: Vec<EngramRelay>,
    added: Vec<EngramRelay>,
    publication_targets: Vec<EngramRelay>,
}

impl EngramRelayRotation {
    pub fn departing(&self) -> &[EngramRelay] {
        &self.departing
    }

    pub fn added(&self) -> &[EngramRelay] {
        &self.added
    }

    pub fn publication_targets(&self) -> &[EngramRelay] {
        &self.publication_targets
    }

    pub fn requires_republication(&self) -> bool {
        !self.departing.is_empty() || !self.added.is_empty()
    }
}

fn encode_body(body: &EngramBody) -> Result<Vec<u8>, AgentMemoryCodecError> {
    #[derive(Serialize)]
    struct CoreBody<'a> {
        slug: &'static str,
        profile: &'a str,
    }
    #[derive(Serialize)]
    struct MemoryBody<'a> {
        slug: &'a str,
        value: Option<&'a str>,
    }
    let bytes = match body {
        EngramBody::Core { profile } => serde_json::to_vec(&CoreBody {
            slug: "core",
            profile,
        }),
        EngramBody::Memory { slug, value } => {
            validate_engram_slug(slug).map_err(|_| AgentMemoryCodecError::InvalidBody)?;
            serde_json::to_vec(&MemoryBody {
                slug,
                value: value.as_deref(),
            })
        }
    }
    .map_err(|_| AgentMemoryCodecError::InvalidBody)?;
    if bytes.len() > MAX_AGENT_PLAINTEXT_BYTES {
        return Err(AgentMemoryCodecError::InvalidBody);
    }
    Ok(bytes)
}

fn public_key_for_secret(secret: &SecretKey) -> PublicKey {
    let secp = secp256k1::Secp256k1::new();
    let keypair = secp256k1::Keypair::from_secret_key(&secp, secret);
    let (public_key, _) = secp256k1::XOnlyPublicKey::from_keypair(&keypair);
    PublicKey::from_bytes(public_key.serialize())
}

fn nostr_public_key(public_key: PublicKey) -> Result<nostr::PublicKey, AgentMemoryCodecError> {
    nostr::PublicKey::from_slice(public_key.as_bytes())
        .map_err(|_| AgentMemoryCodecError::InvalidCoordinate)
}

fn canonical_relays<'a>(values: impl Iterator<Item = &'a str>) -> Vec<EngramRelay> {
    let mut relays = BTreeMap::new();
    for value in values {
        if value.len() > MAX_RELAY_URL_BYTES {
            continue;
        }
        let Ok(parsed) = RelayUrl::parse(value) else {
            continue;
        };
        let url: &nostr::Url = (&parsed).into();
        let canonical_url = if url.path() == "/" {
            parsed.as_str_without_trailing_slash().to_owned()
        } else {
            url.as_str().to_owned()
        };
        relays.entry(parsed).or_insert_with(|| EngramRelay {
            connection_url: value.to_owned(),
            canonical_url,
        });
    }
    relays.into_values().collect()
}

fn owned_relay_map(relays: &[EngramRelay]) -> BTreeMap<String, EngramRelay> {
    relays
        .iter()
        .map(|relay| (relay.canonical_url.clone(), relay.clone()))
        .collect()
}

fn is_lower_hex_32(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGENT_SECRET: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ];
    const OWNER_SECRET: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 2,
    ];
    const ROTATED_OWNER_SECRET: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 3,
    ];

    fn public_key(secret: &[u8; 32]) -> PublicKey {
        let secret = SecretKey::from_slice(secret).expect("fixture secret must be valid");
        public_key_for_secret(&secret)
    }

    #[test]
    fn owner_and_agent_round_trip_encrypted_engram() {
        let owner = public_key(&OWNER_SECRET);
        let body = EngramBody::Memory {
            slug: "mem/example".to_owned(),
            value: Some("hello, agent memory".to_owned()),
        };
        let encrypted =
            encrypt_engram_for_owner(&AGENT_SECRET, owner, &body).expect("encrypt engram");
        assert_eq!(
            encrypted.coordinate().value(),
            "30174:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:72d4f9629106451505d7d341ea85bb3ebad4f654fcfd2aad100d5a35f8a85cba"
        );
        assert!(!encrypted.ciphertext().wire_value().contains("hello"));
        assert_eq!(
            decrypt_engram_as_owner(
                &OWNER_SECRET,
                encrypted.coordinate(),
                encrypted.ciphertext()
            )
            .expect("owner decrypt"),
            body
        );
        assert_eq!(
            decrypt_engram_as_agent(
                &AGENT_SECRET,
                encrypted.coordinate(),
                encrypted.ciphertext()
            )
            .expect("agent decrypt"),
            body
        );
    }

    #[test]
    fn wrong_owner_cannot_read_engram() {
        let owner = public_key(&OWNER_SECRET);
        let encrypted = encrypt_engram_for_owner(
            &AGENT_SECRET,
            owner,
            &EngramBody::Core {
                profile: "private profile".to_owned(),
            },
        )
        .expect("encrypt engram");
        assert_eq!(
            decrypt_engram_as_owner(
                &ROTATED_OWNER_SECRET,
                encrypted.coordinate(),
                encrypted.ciphertext(),
            ),
            Err(AgentMemoryCodecError::WrongReader)
        );
        assert!(!format!("{encrypted:?}").contains("private profile"));
        assert!(!format!("{encrypted:?}").contains(encrypted.ciphertext().wire_value()));
    }

    #[test]
    fn owner_and_relay_rotation_create_explicit_migration_scope() {
        let old_owner = public_key(&OWNER_SECRET);
        let new_owner = public_key(&ROTATED_OWNER_SECRET);
        let body = EngramBody::Memory {
            slug: "mem/rotation".to_owned(),
            value: Some("rotated".to_owned()),
        };
        let old = encrypt_engram_for_owner(&AGENT_SECRET, old_owner, &body).expect("old owner");
        let rotated = encrypt_engram_for_owner(&AGENT_SECRET, new_owner, &body).expect("new owner");
        assert_ne!(old.coordinate().d_tag(), rotated.coordinate().d_tag());
        assert_eq!(
            decrypt_engram_as_owner(&OWNER_SECRET, rotated.coordinate(), rotated.ciphertext(),),
            Err(AgentMemoryCodecError::WrongReader)
        );
        assert_eq!(
            decrypt_engram_as_owner(
                &ROTATED_OWNER_SECRET,
                rotated.coordinate(),
                rotated.ciphertext(),
            )
            .expect("rotated owner decrypt"),
            body
        );

        let previous = EngramRelayScope::resolve(
            &[
                AgentRelay::new("WSS://Relay.Example:443/", AgentRelayAccess::Write),
                AgentRelay::new("wss://relay.example", AgentRelayAccess::ReadWrite),
                AgentRelay::new("wss://departing.example", AgentRelayAccess::Write),
                AgentRelay::new("wss://read-only.example", AgentRelayAccess::Read),
            ],
            &[],
        )
        .expect("previous scope");
        let next = EngramRelayScope::resolve(
            &[
                AgentRelay::new("wss://relay.example", AgentRelayAccess::Write),
                AgentRelay::new("wss://new.example/path/", AgentRelayAccess::Write),
            ],
            &[],
        )
        .expect("next scope");
        assert_eq!(previous.relays().len(), 2);
        assert!(
            previous
                .relays()
                .iter()
                .any(|relay| relay.canonical_url() == "wss://relay.example")
        );
        let rotation = previous.rotate_to(&next);
        assert!(rotation.requires_republication());
        assert_eq!(rotation.departing().len(), 1);
        assert_eq!(
            rotation.departing()[0].canonical_url(),
            "wss://departing.example"
        );
        assert_eq!(rotation.added().len(), 1);
        assert_eq!(rotation.publication_targets().len(), 3);
        assert!(rotation.publication_targets().iter().any(|relay| {
            relay.connection_url() == "wss://new.example/path/"
                && relay.canonical_url() == "wss://new.example/path/"
        }));
    }

    #[test]
    fn malformed_coordinates_and_empty_relay_scope_fail_closed() {
        let owner = public_key(&OWNER_SECRET);
        for coordinate in [
            "30174",
            "030174:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:72d4f9629106451505d7d341ea85bb3ebad4f654fcfd2aad100d5a35f8a85cba",
            "30174:bad:00",
            "30175:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:72d4f9629106451505d7d341ea85bb3ebad4f654fcfd2aad100d5a35f8a85cba",
            "30174:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:72D4F9629106451505D7D341EA85BB3EBAD4F654FCFD2AAD100D5A35F8A85CBA",
            "30174:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:72d4f9629106451505d7d341ea85bb3ebad4f654fcfd2aad100d5a35f8a85cba:extra",
        ] {
            assert_eq!(
                EngramCoordinate::parse(coordinate, owner),
                Err(AgentMemoryCodecError::InvalidCoordinate),
                "accepted malformed coordinate {coordinate}"
            );
        }
        assert_eq!(
            EngramRelayScope::resolve(
                &[AgentRelay::new(
                    "https://not-a-relay.example",
                    AgentRelayAccess::Write,
                )],
                &[],
            ),
            Err(AgentMemoryCodecError::NoUsableRelay)
        );
        let fallback = EngramRelayScope::resolve(
            &[AgentRelay::new(
                "wss://read-only.example",
                AgentRelayAccess::Read,
            )],
            &["WSS://Fallback.Example:443/".to_owned()],
        )
        .expect("fallback scope must be usable");
        assert_eq!(fallback.source(), EngramRelayScopeSource::OutOfBandFallback);
        assert_eq!(
            fallback.relays()[0].canonical_url(),
            "wss://fallback.example"
        );
    }
}
