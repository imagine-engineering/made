use std::{collections::HashMap, fmt};

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit, Payload},
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use collaboration_domain::{PushCapabilityReference, PushEndpointGeneration, PushLeaseGeneration};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zeroize::Zeroizing;

use super::{ApprovedPushProfile, apns};
use crate::executor::PushDeliveryRequest;

const GRANT_AAD_PREFIX: &[u8] = b"buzz-stateful-delivery-capability-v1:";
const TOKEN_AAD_PREFIX: &[u8] = b"buzz-apns-token-v1:";
const ENDPOINT_FINGERPRINT_DOMAIN: &[u8] = b"buzz-apns-endpoint-v1\0";
const MAX_KEY_ID_BYTES: usize = 32;
const MAX_GRANT_BYTES: usize = 4_096;
const MAX_TOKEN_CIPHERTEXT_BYTES: usize = 2_048;
const APNS_DEVICE_TOKEN_BYTES: usize = 32;

pub struct PushPlatformKey {
    id: String,
    key: Zeroizing<[u8; 32]>,
}

impl PushPlatformKey {
    pub fn new(
        id: impl Into<String>,
        key: Zeroizing<[u8; 32]>,
    ) -> Result<Self, PlatformGrantError> {
        let id = id.into();
        if id.is_empty()
            || id.len() > MAX_KEY_ID_BYTES
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(PlatformGrantError::Invalid);
        }
        Ok(Self { id, key })
    }
}

impl fmt::Debug for PushPlatformKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PushPlatformKey([redacted])")
    }
}

#[derive(Clone)]
struct AeadKey {
    id: String,
    cipher: Aes256Gcm,
    material_fingerprint: [u8; 32],
}

#[derive(Clone)]
struct AeadKeyring {
    current: AeadKey,
    predecessors: HashMap<String, AeadKey>,
    aad_prefix: &'static [u8],
}

impl AeadKeyring {
    fn new(
        keys: Vec<PushPlatformKey>,
        aad_prefix: &'static [u8],
    ) -> Result<Self, PlatformGrantError> {
        let mut keys = keys.into_iter();
        let current = keys.next().ok_or(PlatformGrantError::EmptyKeyring)?;
        let current = Self::build_key(current)?;
        let mut predecessors = HashMap::new();
        for key in keys {
            let key = Self::build_key(key)?;
            if key.id == current.id || predecessors.insert(key.id.clone(), key).is_some() {
                return Err(PlatformGrantError::DuplicateKeyId);
            }
        }
        Ok(Self {
            current,
            predecessors,
            aad_prefix,
        })
    }

    fn build_key(key: PushPlatformKey) -> Result<AeadKey, PlatformGrantError> {
        let material_fingerprint = Sha256::digest(key.key.as_slice()).into();
        let cipher = Aes256Gcm::new_from_slice(key.key.as_slice())
            .map_err(|_| PlatformGrantError::Invalid)?;
        Ok(AeadKey {
            id: key.id,
            cipher,
            material_fingerprint,
        })
    }

    fn aad(&self, key: &AeadKey) -> Vec<u8> {
        [self.aad_prefix, key.id.as_bytes()].concat()
    }

    fn seal(&self, plaintext: &[u8], maximum: usize) -> Result<Vec<u8>, PlatformGrantError> {
        if plaintext.is_empty() || plaintext.len() > maximum {
            return Err(PlatformGrantError::Invalid);
        }
        let mut nonce = [0_u8; 12];
        getrandom::fill(&mut nonce).map_err(|_| PlatformGrantError::Unavailable)?;
        let mut ciphertext = nonce.to_vec();
        ciphertext.extend(
            self.current
                .cipher
                .encrypt(
                    Nonce::from_slice(&nonce),
                    Payload {
                        msg: plaintext,
                        aad: &self.aad(&self.current),
                    },
                )
                .map_err(|_| PlatformGrantError::Invalid)?,
        );
        let encoded =
            format!("{}.{}", self.current.id, URL_SAFE_NO_PAD.encode(ciphertext)).into_bytes();
        if encoded.len() > maximum {
            return Err(PlatformGrantError::Invalid);
        }
        Ok(encoded)
    }

    fn open(
        &self,
        encoded: &[u8],
        maximum: usize,
    ) -> Result<Zeroizing<Vec<u8>>, PlatformGrantError> {
        if encoded.is_empty() || encoded.len() > maximum {
            return Err(PlatformGrantError::Invalid);
        }
        let encoded = std::str::from_utf8(encoded).map_err(|_| PlatformGrantError::Invalid)?;
        let (key_id, body) = encoded.split_once('.').ok_or(PlatformGrantError::Invalid)?;
        let key = if key_id == self.current.id {
            &self.current
        } else {
            self.predecessors
                .get(key_id)
                .ok_or(PlatformGrantError::Invalid)?
        };
        let bytes = URL_SAFE_NO_PAD
            .decode(body)
            .map_err(|_| PlatformGrantError::Invalid)?;
        if bytes.len() < 13 {
            return Err(PlatformGrantError::Invalid);
        }
        key.cipher
            .decrypt(
                Nonce::from_slice(&bytes[..12]),
                Payload {
                    msg: &bytes[12..],
                    aad: &self.aad(key),
                },
            )
            .map(Zeroizing::new)
            .map_err(|_| PlatformGrantError::Invalid)
    }

    fn material_fingerprints(&self) -> impl Iterator<Item = &[u8; 32]> {
        std::iter::once(&self.current.material_fingerprint).chain(
            self.predecessors
                .values()
                .map(|key| &key.material_fingerprint),
        )
    }
}

#[derive(Clone)]
pub struct GrantKeyring(AeadKeyring);

impl GrantKeyring {
    pub fn new(keys: Vec<PushPlatformKey>) -> Result<Self, PlatformGrantError> {
        AeadKeyring::new(keys, GRANT_AAD_PREFIX).map(Self)
    }

    pub fn issue(&self, grant: &EndpointGrant) -> Result<SealedEndpointGrant, PlatformGrantError> {
        let plaintext =
            Zeroizing::new(serde_json::to_vec(grant).map_err(|_| PlatformGrantError::Invalid)?);
        self.0
            .seal(&plaintext, MAX_GRANT_BYTES)
            .map(SealedEndpointGrant)
    }

    pub fn open(&self, grant: &SealedEndpointGrant) -> Result<EndpointGrant, PlatformGrantError> {
        let plaintext = self.0.open(&grant.0, MAX_GRANT_BYTES)?;
        let grant: EndpointGrant =
            serde_json::from_slice(&plaintext).map_err(|_| PlatformGrantError::Invalid)?;
        grant.validate()?;
        Ok(grant)
    }
}

#[derive(Clone)]
pub struct TokenKeyring(AeadKeyring);

impl TokenKeyring {
    pub fn new(keys: Vec<PushPlatformKey>) -> Result<Self, PlatformGrantError> {
        AeadKeyring::new(keys, TOKEN_AAD_PREFIX).map(Self)
    }

    pub fn seal(
        &self,
        token: &ApnsDeviceToken,
    ) -> Result<EncryptedApnsDeviceToken, PlatformGrantError> {
        self.0
            .seal(token.0.as_slice(), MAX_TOKEN_CIPHERTEXT_BYTES)
            .map(EncryptedApnsDeviceToken)
    }

    pub fn open(
        &self,
        token: &EncryptedApnsDeviceToken,
    ) -> Result<ApnsDeviceToken, PlatformGrantError> {
        let plaintext = self.0.open(&token.0, MAX_TOKEN_CIPHERTEXT_BYTES)?;
        let token: [u8; APNS_DEVICE_TOKEN_BYTES] = plaintext
            .as_slice()
            .try_into()
            .map_err(|_| PlatformGrantError::Invalid)?;
        Ok(ApnsDeviceToken(Zeroizing::new(token)))
    }
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointGrant {
    v: u8,
    delegation_id: Uuid,
    relay_pubkey: String,
    app_profile: ApprovedPushProfile,
    endpoint_epoch: i64,
    generation: i64,
    expires_at: i64,
}

impl EndpointGrant {
    pub fn new(
        delegation_id: Uuid,
        relay_pubkey: String,
        profile: ApprovedPushProfile,
        lease_generation: PushLeaseGeneration,
        endpoint_generation: PushEndpointGeneration,
        expires_at_millis: u64,
    ) -> Result<Self, PlatformGrantError> {
        let generation =
            i64::try_from(lease_generation.get()).map_err(|_| PlatformGrantError::Invalid)?;
        let endpoint_epoch =
            i64::try_from(endpoint_generation.get()).map_err(|_| PlatformGrantError::Invalid)?;
        let expires_at =
            i64::try_from(expires_at_millis / 1_000).map_err(|_| PlatformGrantError::Invalid)?;
        if delegation_id.is_nil()
            || relay_pubkey.len() != 64
            || !relay_pubkey.bytes().all(|byte| byte.is_ascii_hexdigit())
            || expires_at_millis == 0
            || !expires_at_millis.is_multiple_of(1_000)
        {
            return Err(PlatformGrantError::Invalid);
        }
        let grant = Self {
            v: 1,
            delegation_id,
            relay_pubkey,
            app_profile: profile,
            endpoint_epoch,
            generation,
            expires_at,
        };
        grant.validate()?;
        Ok(grant)
    }

    fn validate(&self) -> Result<(), PlatformGrantError> {
        if self.v != 1
            || self.delegation_id.is_nil()
            || self.relay_pubkey.len() != 64
            || !self
                .relay_pubkey
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || self.endpoint_epoch < 1
            || self.generation < 1
            || self.expires_at < 1
        {
            return Err(PlatformGrantError::Invalid);
        }
        Ok(())
    }
}

impl fmt::Debug for EndpointGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointGrant")
            .field("profile", &self.app_profile)
            .field("lease_generation", &self.generation)
            .field("endpoint_generation", &self.endpoint_epoch)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct SealedEndpointGrant(Vec<u8>);

impl SealedEndpointGrant {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, PlatformGrantError> {
        if bytes.is_empty() || bytes.len() > MAX_GRANT_BYTES {
            return Err(PlatformGrantError::Invalid);
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn capability_reference(&self) -> Result<PushCapabilityReference, PlatformGrantError> {
        PushCapabilityReference::from_digest(Sha256::digest(&self.0).into())
            .ok_or(PlatformGrantError::Invalid)
    }
}

impl fmt::Debug for SealedEndpointGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SealedEndpointGrant([redacted])")
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct EncryptedApnsDeviceToken(Vec<u8>);

impl EncryptedApnsDeviceToken {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, PlatformGrantError> {
        if bytes.is_empty() || bytes.len() > MAX_TOKEN_CIPHERTEXT_BYTES {
            return Err(PlatformGrantError::Invalid);
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for EncryptedApnsDeviceToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("EncryptedApnsDeviceToken([redacted])")
    }
}

#[derive(Clone)]
pub struct ApnsDeviceToken(Zeroizing<[u8; APNS_DEVICE_TOKEN_BYTES]>);

impl ApnsDeviceToken {
    pub fn from_hex(value: &str) -> Result<Self, PlatformGrantError> {
        if value.len() != APNS_DEVICE_TOKEN_BYTES * 2
            || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(PlatformGrantError::Invalid);
        }
        let mut bytes = Zeroizing::new([0_u8; APNS_DEVICE_TOKEN_BYTES]);
        hex::decode_to_slice(value, bytes.as_mut_slice())
            .map_err(|_| PlatformGrantError::Invalid)?;
        Ok(Self(bytes))
    }

    pub fn encode_hex(&self) -> Zeroizing<String> {
        Zeroizing::new(hex::encode(self.0.as_slice()))
    }

    pub fn fingerprint(&self, profile: ApprovedPushProfile) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(ENDPOINT_FINGERPRINT_DOMAIN);
        hasher.update(profile.as_str().as_bytes());
        hasher.update([0]);
        hasher.update(self.0.as_slice());
        hasher.finalize().into()
    }
}

impl fmt::Debug for ApnsDeviceToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApnsDeviceToken([redacted])")
    }
}

#[derive(Clone, Debug)]
pub struct SealedApnsAuthorityRecord {
    pub grant: SealedEndpointGrant,
    pub token: EncryptedApnsDeviceToken,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PlatformAuthorityStoreError {
    #[error("platform endpoint authority rejected the request")]
    Rejected,
    #[error("platform endpoint authority is unavailable")]
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlatformAuthorityQuery {
    capability_reference: PushCapabilityReference,
    lease_generation: PushLeaseGeneration,
    endpoint_generation: PushEndpointGeneration,
    expires_at_millis: u64,
}

impl PlatformAuthorityQuery {
    fn from_request(request: &PushDeliveryRequest) -> Self {
        Self {
            capability_reference: request.capability_reference(),
            lease_generation: request.lease_generation(),
            endpoint_generation: request.endpoint_generation(),
            expires_at_millis: request.expires_at_millis(),
        }
    }

    pub const fn capability_reference(self) -> PushCapabilityReference {
        self.capability_reference
    }

    pub const fn lease_generation(self) -> PushLeaseGeneration {
        self.lease_generation
    }

    pub const fn endpoint_generation(self) -> PushEndpointGeneration {
        self.endpoint_generation
    }

    pub const fn expires_at_millis(self) -> u64 {
        self.expires_at_millis
    }
}

#[async_trait]
pub trait PlatformAuthorityStore: Send + Sync {
    /// This call is the last-hop send-begin boundary: implementations atomically
    /// reject revoked, rotated, expired or replayed installation authority.
    async fn authority(
        &self,
        query: PlatformAuthorityQuery,
    ) -> Result<SealedApnsAuthorityRecord, PlatformAuthorityStoreError>;
}

pub struct SealedEndpointAuthority<S> {
    store: S,
    grants: GrantKeyring,
    tokens: TokenKeyring,
}

impl<S> SealedEndpointAuthority<S> {
    pub fn new(
        store: S,
        grants: GrantKeyring,
        tokens: TokenKeyring,
    ) -> Result<Self, PlatformGrantError> {
        if grants
            .0
            .material_fingerprints()
            .any(|grant| tokens.0.material_fingerprints().any(|token| token == grant))
        {
            return Err(PlatformGrantError::KeyReuse);
        }
        Ok(Self {
            store,
            grants,
            tokens,
        })
    }
}

#[async_trait]
impl<S> apns::ApnsEndpointAuthority for SealedEndpointAuthority<S>
where
    S: PlatformAuthorityStore,
{
    async fn resolve(
        &self,
        request: &PushDeliveryRequest,
    ) -> Result<apns::AuthorizedApnsEndpoint, apns::ApnsEndpointAuthorityError> {
        let record = self
            .store
            .authority(PlatformAuthorityQuery::from_request(request))
            .await
            .map_err(|error| match error {
                PlatformAuthorityStoreError::Rejected => apns::ApnsEndpointAuthorityError::Rejected,
                PlatformAuthorityStoreError::Unavailable => {
                    apns::ApnsEndpointAuthorityError::Unavailable
                }
            })?;
        if record
            .grant
            .capability_reference()
            .map_err(|_| apns::ApnsEndpointAuthorityError::Rejected)?
            != request.capability_reference()
        {
            return Err(apns::ApnsEndpointAuthorityError::Rejected);
        }
        let grant = self
            .grants
            .open(&record.grant)
            .map_err(|_| apns::ApnsEndpointAuthorityError::Rejected)?;
        let lease_generation = u64::try_from(grant.generation)
            .ok()
            .and_then(PushLeaseGeneration::new);
        let endpoint_generation = u64::try_from(grant.endpoint_epoch)
            .ok()
            .and_then(PushEndpointGeneration::new);
        let expires_at_millis = u64::try_from(grant.expires_at)
            .ok()
            .and_then(|seconds| seconds.checked_mul(1_000));
        if lease_generation != Some(request.lease_generation())
            || endpoint_generation != Some(request.endpoint_generation())
            || expires_at_millis.is_none_or(|expires_at| expires_at < request.expires_at_millis())
        {
            return Err(apns::ApnsEndpointAuthorityError::Rejected);
        }
        let token = self
            .tokens
            .open(&record.token)
            .map_err(|_| apns::ApnsEndpointAuthorityError::Rejected)?;
        Ok(apns::AuthorizedApnsEndpoint::new(
            grant.app_profile,
            token,
            request.endpoint_generation(),
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PlatformGrantError {
    #[error("invalid platform endpoint authority")]
    Invalid,
    #[error("platform endpoint keyring is empty")]
    EmptyKeyring,
    #[error("platform endpoint key ids must be unique")]
    DuplicateKeyId,
    #[error("grant and token custody keys must be independent")]
    KeyReuse,
    #[error("platform endpoint cryptography is unavailable")]
    Unavailable,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::apns::ApnsEndpointAuthority;

    fn key(id: &str, byte: u8) -> PushPlatformKey {
        PushPlatformKey::new(id, Zeroizing::new([byte; 32])).expect("test key")
    }

    fn grant(expires_at_millis: u64) -> EndpointGrant {
        EndpointGrant::new(
            Uuid::from_u128(1),
            "11".repeat(32),
            ApprovedPushProfile::BuzzIosProduction,
            PushLeaseGeneration::new(2).expect("lease generation"),
            PushEndpointGeneration::new(3).expect("endpoint generation"),
            expires_at_millis,
        )
        .expect("test grant")
    }

    #[test]
    fn token_and_grant_custody_rotate_without_cross_key_reuse() {
        let old_grants = GrantKeyring::new(vec![key("old-grant", 1)]).expect("old grants");
        let grant = grant(20_000);
        assert_eq!(
            serde_json::to_value(&grant).expect("grant JSON"),
            serde_json::json!({
                "v": 1,
                "delegation_id": Uuid::from_u128(1),
                "relay_pubkey": "11".repeat(32),
                "app_profile": "buzz-ios-production",
                "endpoint_epoch": 3,
                "generation": 2,
                "expires_at": 20
            })
        );
        let sealed_grant = old_grants.issue(&grant).expect("sealed grant");
        assert_eq!(
            sealed_grant
                .capability_reference()
                .expect("capability")
                .as_digest(),
            &<[u8; 32]>::from(Sha256::digest(sealed_grant.as_bytes()))
        );
        let old_tokens = TokenKeyring::new(vec![key("old-token", 2)]).expect("old tokens");
        let token = ApnsDeviceToken::from_hex(&"ab".repeat(32)).expect("token");
        let sealed_token = old_tokens.seal(&token).expect("sealed token");

        let grants = GrantKeyring::new(vec![key("current-grant", 3), key("old-grant", 1)])
            .expect("rotated grants");
        let tokens = TokenKeyring::new(vec![key("current-token", 4), key("old-token", 2)])
            .expect("rotated tokens");
        assert_eq!(grants.open(&sealed_grant).expect("opened grant"), grant);
        assert_eq!(
            tokens
                .open(&sealed_token)
                .expect("opened token")
                .encode_hex()
                .as_str(),
            "ab".repeat(32)
        );

        let store = FakeStore {
            record: SealedApnsAuthorityRecord {
                grant: sealed_grant,
                token: sealed_token,
            },
        };
        assert!(SealedEndpointAuthority::new(store, grants.clone(), tokens).is_ok());
        let reused = TokenKeyring::new(vec![key("reused-token", 3)]).expect("reused token key");
        assert!(matches!(
            SealedEndpointAuthority::new(FakeStore::default(), grants, reused),
            Err(PlatformGrantError::KeyReuse)
        ));
    }

    #[tokio::test]
    async fn sealed_authority_fences_capability_generations_and_expiry() {
        let grants = GrantKeyring::new(vec![key("grant", 1)]).expect("grant keys");
        let tokens = TokenKeyring::new(vec![key("token", 2)]).expect("token keys");
        let sealed_grant = grants.issue(&grant(20_000)).expect("sealed grant");
        let capability_reference = sealed_grant
            .capability_reference()
            .expect("capability reference");
        let token = ApnsDeviceToken::from_hex(&"ab".repeat(32)).expect("token");
        let store = FakeStore {
            record: SealedApnsAuthorityRecord {
                grant: sealed_grant,
                token: tokens.seal(&token).expect("sealed token"),
            },
        };
        let authority =
            SealedEndpointAuthority::new(store, grants, tokens).expect("sealed authority");
        let request = PushDeliveryRequest::for_test(
            Uuid::from_u128(7),
            PushLeaseGeneration::new(2).expect("lease generation"),
            PushEndpointGeneration::new(3).expect("endpoint generation"),
            capability_reference,
            20_000,
        );
        assert!(authority.resolve(&request).await.is_ok());

        let expired_request = PushDeliveryRequest::for_test(
            Uuid::from_u128(8),
            PushLeaseGeneration::new(2).expect("lease generation"),
            PushEndpointGeneration::new(3).expect("endpoint generation"),
            capability_reference,
            20_001,
        );
        assert_eq!(
            authority.resolve(&expired_request).await.unwrap_err(),
            apns::ApnsEndpointAuthorityError::Rejected
        );
    }

    #[derive(Clone)]
    struct FakeStore {
        record: SealedApnsAuthorityRecord,
    }

    impl Default for FakeStore {
        fn default() -> Self {
            let grants = GrantKeyring::new(vec![key("unused-grant", 9)]).expect("grant key");
            let tokens = TokenKeyring::new(vec![key("unused-token", 8)]).expect("token key");
            let token = ApnsDeviceToken::from_hex(&"00".repeat(32)).expect("token");
            Self {
                record: SealedApnsAuthorityRecord {
                    grant: grants.issue(&grant(20_000)).expect("grant"),
                    token: tokens.seal(&token).expect("token"),
                },
            }
        }
    }

    #[async_trait]
    impl PlatformAuthorityStore for FakeStore {
        async fn authority(
            &self,
            _query: PlatformAuthorityQuery,
        ) -> Result<SealedApnsAuthorityRecord, PlatformAuthorityStoreError> {
            Ok(self.record.clone())
        }
    }
}
