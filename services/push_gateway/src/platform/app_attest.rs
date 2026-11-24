use std::{fmt, sync::Arc};

use appattest::{assertion::Assertion, attestation::Attestation};
use async_trait::async_trait;
use base64::{
    Engine as _, engine::general_purpose::STANDARD, engine::general_purpose::URL_SAFE_NO_PAD,
};
use byteorder::{BigEndian, ByteOrder};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::ApprovedPushProfile;

const MAX_ATTESTATION_BYTES: usize = 16 * 1024;
const MAX_ASSERTION_BYTES: usize = 1_024;
const MAX_ATTESTATION_BASE64_BYTES: usize = MAX_ATTESTATION_BYTES.div_ceil(3) * 4;
const MAX_ASSERTION_BASE64_BYTES: usize = MAX_ASSERTION_BYTES.div_ceil(3) * 4;
const MAX_KEY_ID_BASE64_BYTES: usize = 64;
const MAX_TRANSCRIPT_BYTES: usize = 8 * 1_024;
const MAX_APP_IDENTIFIER_BYTES: usize = 255;
const APPLE_APP_ATTEST_ROOT_PEM_SHA256: [u8; 32] = [
    0xc7, 0x78, 0xd0, 0x9a, 0xc3, 0x41, 0xf7, 0xfd, 0x9f, 0x8f, 0x3b, 0x19, 0xe2, 0xb8, 0x15, 0xaf,
    0x6a, 0xed, 0x4a, 0xd4, 0x49, 0x0e, 0x1e, 0x92, 0xc0, 0x5c, 0xb3, 0x55, 0x21, 0x2a, 0x50, 0x13,
];

#[derive(Clone, Eq, PartialEq)]
pub struct VerifiedAppAttestation {
    key_id: [u8; 32],
    public_key: Vec<u8>,
}

impl VerifiedAppAttestation {
    pub fn new(key_id: [u8; 32], public_key: Vec<u8>) -> Result<Self, AppAttestError> {
        if public_key.is_empty() || public_key.len() > 1_024 {
            return Err(AppAttestError::Rejected);
        }
        Ok(Self { key_id, public_key })
    }

    pub const fn key_id(&self) -> &[u8; 32] {
        &self.key_id
    }

    pub fn public_key(&self) -> &[u8] {
        &self.public_key
    }
}

impl fmt::Debug for VerifiedAppAttestation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("VerifiedAppAttestation([redacted])")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifiedAppAssertion {
    counter: u32,
}

impl VerifiedAppAssertion {
    pub fn new(counter: u32) -> Result<Self, AppAttestError> {
        if counter == 0 {
            return Err(AppAttestError::Rejected);
        }
        Ok(Self { counter })
    }

    pub const fn counter(self) -> u32 {
        self.counter
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AppAttestError {
    #[error("invalid app attestation or assertion")]
    Rejected,
    #[error("app attestation authority is unavailable")]
    Unavailable,
}

pub trait AppAttestCryptography: Send + Sync {
    fn verify_attestation(
        &self,
        attestation_base64: &str,
        key_id_base64: &str,
        client_data: &[u8],
    ) -> Result<VerifiedAppAttestation, AppAttestError>;

    fn verify_assertion(
        &self,
        assertion_base64: &str,
        client_data: &[u8],
        public_key: &[u8],
        previous_counter: u32,
        challenge: &str,
    ) -> Result<VerifiedAppAssertion, AppAttestError>;
}

#[derive(Clone)]
pub struct AppleAppAttestVerifier {
    app_identifier: String,
    apple_root_certificate_pem: Arc<[u8]>,
}

impl AppleAppAttestVerifier {
    pub fn new(
        app_identifier: impl Into<String>,
        apple_root_certificate_pem: Vec<u8>,
    ) -> Result<Self, AppAttestError> {
        let app_identifier = app_identifier.into();
        let root_digest: [u8; 32] = Sha256::digest(&apple_root_certificate_pem).into();
        if app_identifier.is_empty()
            || app_identifier.len() > MAX_APP_IDENTIFIER_BYTES
            || app_identifier.chars().any(char::is_control)
            || root_digest != APPLE_APP_ATTEST_ROOT_PEM_SHA256
        {
            return Err(AppAttestError::Rejected);
        }
        Ok(Self {
            app_identifier,
            apple_root_certificate_pem: apple_root_certificate_pem.into(),
        })
    }
}

impl AppAttestCryptography for AppleAppAttestVerifier {
    fn verify_attestation(
        &self,
        attestation_base64: &str,
        key_id_base64: &str,
        client_data: &[u8],
    ) -> Result<VerifiedAppAttestation, AppAttestError> {
        validate_transcript(client_data)?;
        if attestation_base64.is_empty()
            || attestation_base64.len() > MAX_ATTESTATION_BASE64_BYTES
            || key_id_base64.is_empty()
            || key_id_base64.len() > MAX_KEY_ID_BASE64_BYTES
        {
            return Err(AppAttestError::Rejected);
        }
        let attestation = STANDARD
            .decode(attestation_base64)
            .map_err(|_| AppAttestError::Rejected)?;
        if attestation.is_empty() || attestation.len() > MAX_ATTESTATION_BYTES {
            return Err(AppAttestError::Rejected);
        }
        let challenge = std::str::from_utf8(client_data).map_err(|_| AppAttestError::Rejected)?;
        let attestation =
            Attestation::from_cbor_bytes(&attestation).map_err(|_| AppAttestError::Rejected)?;
        let (public_key, _) = attestation
            .verify(
                challenge,
                &self.app_identifier,
                key_id_base64,
                &self.apple_root_certificate_pem,
            )
            .map_err(|_| AppAttestError::Rejected)?;
        let key_id: [u8; 32] = STANDARD
            .decode(key_id_base64)
            .map_err(|_| AppAttestError::Rejected)?
            .try_into()
            .map_err(|_| AppAttestError::Rejected)?;
        VerifiedAppAttestation::new(key_id, public_key.to_vec())
    }

    fn verify_assertion(
        &self,
        assertion_base64: &str,
        client_data: &[u8],
        public_key: &[u8],
        previous_counter: u32,
        challenge: &str,
    ) -> Result<VerifiedAppAssertion, AppAttestError> {
        validate_transcript(client_data)?;
        if assertion_base64.is_empty() || assertion_base64.len() > MAX_ASSERTION_BASE64_BYTES {
            return Err(AppAttestError::Rejected);
        }
        let assertion = STANDARD
            .decode(assertion_base64)
            .map_err(|_| AppAttestError::Rejected)?;
        if assertion.is_empty() || assertion.len() > MAX_ASSERTION_BYTES {
            return Err(AppAttestError::Rejected);
        }
        let counter = assertion_counter(&assertion)?;
        let client_data_hash = Sha256::digest(client_data);
        Assertion::from_assertion(&assertion)
            .map_err(|_| AppAttestError::Rejected)?
            .verify(
                client_data_hash,
                challenge,
                &self.app_identifier,
                public_key,
                previous_counter,
                challenge,
            )
            .map_err(|_| AppAttestError::Rejected)?;
        VerifiedAppAssertion::new(counter)
    }
}

fn validate_transcript(transcript: &[u8]) -> Result<(), AppAttestError> {
    if transcript.is_empty()
        || transcript.len() > MAX_TRANSCRIPT_BYTES
        || std::str::from_utf8(transcript).is_err()
    {
        return Err(AppAttestError::Rejected);
    }
    Ok(())
}

fn assertion_counter(cbor: &[u8]) -> Result<u32, AppAttestError> {
    let mut decoder = minicbor::Decoder::new(cbor);
    let count = decoder
        .map()
        .map_err(|_| AppAttestError::Rejected)?
        .ok_or(AppAttestError::Rejected)?;
    let mut authenticator_data = None;
    for _ in 0..count {
        let key = decoder.str().map_err(|_| AppAttestError::Rejected)?;
        match key {
            "authenticatorData" => {
                authenticator_data = Some(decoder.bytes().map_err(|_| AppAttestError::Rejected)?);
            }
            "signature" => {
                decoder.bytes().map_err(|_| AppAttestError::Rejected)?;
            }
            _ => return Err(AppAttestError::Rejected),
        }
    }
    let authenticator_data = authenticator_data
        .filter(|data| data.len() == 37)
        .ok_or(AppAttestError::Rejected)?;
    VerifiedAppAssertion::new(BigEndian::read_u32(&authenticator_data[33..37]))?;
    Ok(BigEndian::read_u32(&authenticator_data[33..37]))
}

#[derive(Clone, Eq, PartialEq)]
pub struct AppAttestChallenge {
    id: Uuid,
    value: [u8; 32],
    expires_at_millis: u64,
}

impl fmt::Debug for AppAttestChallenge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AppAttestChallenge")
            .field("expires_at_millis", &self.expires_at_millis)
            .finish_non_exhaustive()
    }
}

impl AppAttestChallenge {
    pub fn new(id: Uuid, value: [u8; 32], expires_at_millis: u64) -> Result<Self, AppAttestError> {
        if id.is_nil()
            || value.iter().all(|byte| *byte == 0)
            || expires_at_millis == 0
            || i64::try_from(expires_at_millis).is_err()
        {
            return Err(AppAttestError::Rejected);
        }
        Ok(Self {
            id,
            value,
            expires_at_millis,
        })
    }

    pub const fn id(&self) -> Uuid {
        self.id
    }

    pub const fn value(&self) -> &[u8; 32] {
        &self.value
    }

    pub const fn expires_at_millis(&self) -> u64 {
        self.expires_at_millis
    }

    fn encoded_value(&self) -> String {
        URL_SAFE_NO_PAD.encode(self.value)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct AppAttestInstallation {
    id: Uuid,
    profile: ApprovedPushProfile,
    public_key: Vec<u8>,
    assertion_counter: u32,
}

impl AppAttestInstallation {
    pub fn new(
        id: Uuid,
        profile: ApprovedPushProfile,
        public_key: Vec<u8>,
        assertion_counter: u32,
    ) -> Result<Self, AppAttestError> {
        if id.is_nil() || public_key.is_empty() || public_key.len() > 1_024 {
            return Err(AppAttestError::Rejected);
        }
        Ok(Self {
            id,
            profile,
            public_key,
            assertion_counter,
        })
    }
}

impl fmt::Debug for AppAttestInstallation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AppAttestInstallation")
            .field("profile", &self.profile)
            .field("assertion_counter", &self.assertion_counter)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AppAttestStoreError {
    #[error("app attestation authority rejected the request")]
    Rejected,
    #[error("app attestation authority is unavailable")]
    Unavailable,
}

#[async_trait]
pub trait AppAttestAuthorityStore: Send + Sync {
    async fn installation(
        &self,
        installation_id: Uuid,
        now_millis: u64,
    ) -> Result<AppAttestInstallation, AppAttestStoreError>;

    async fn consume_enrollment_challenge(
        &self,
        challenge: &AppAttestChallenge,
        now_millis: u64,
    ) -> Result<(), AppAttestStoreError>;

    async fn consume_assertion_and_advance(
        &self,
        installation_id: Uuid,
        challenge: &AppAttestChallenge,
        expected_counter: u32,
        next_counter: u32,
        now_millis: u64,
    ) -> Result<(), AppAttestStoreError>;
}

pub struct AppAttestAdapter<C, S> {
    cryptography: C,
    store: S,
}

impl<C, S> AppAttestAdapter<C, S>
where
    C: AppAttestCryptography,
    S: AppAttestAuthorityStore,
{
    pub const fn new(cryptography: C, store: S) -> Self {
        Self {
            cryptography,
            store,
        }
    }

    pub async fn admit_enrollment(
        &self,
        challenge: &AppAttestChallenge,
        attestation_base64: &str,
        key_id_base64: &str,
        transcript: &[u8],
        now_millis: u64,
    ) -> Result<VerifiedAppAttestation, AppAttestError> {
        validate_challenge(challenge, now_millis)?;
        let verified =
            self.cryptography
                .verify_attestation(attestation_base64, key_id_base64, transcript)?;
        self.store
            .consume_enrollment_challenge(challenge, now_millis)
            .await
            .map_err(map_store_error)?;
        Ok(verified)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn admit_assertion(
        &self,
        installation_id: Uuid,
        expected_profile: ApprovedPushProfile,
        challenge: &AppAttestChallenge,
        assertion_base64: &str,
        transcript: &[u8],
        now_millis: u64,
    ) -> Result<VerifiedAppAssertion, AppAttestError> {
        validate_challenge(challenge, now_millis)?;
        let installation = self
            .store
            .installation(installation_id, now_millis)
            .await
            .map_err(map_store_error)?;
        if installation.id != installation_id || installation.profile != expected_profile {
            return Err(AppAttestError::Rejected);
        }
        let verified = self.cryptography.verify_assertion(
            assertion_base64,
            transcript,
            &installation.public_key,
            installation.assertion_counter,
            &challenge.encoded_value(),
        )?;
        if verified.counter() <= installation.assertion_counter {
            return Err(AppAttestError::Rejected);
        }
        self.store
            .consume_assertion_and_advance(
                installation_id,
                challenge,
                installation.assertion_counter,
                verified.counter(),
                now_millis,
            )
            .await
            .map_err(map_store_error)?;
        Ok(verified)
    }
}

fn validate_challenge(
    challenge: &AppAttestChallenge,
    now_millis: u64,
) -> Result<(), AppAttestError> {
    if now_millis == 0 || now_millis > challenge.expires_at_millis {
        return Err(AppAttestError::Rejected);
    }
    Ok(())
}

fn map_store_error(error: AppAttestStoreError) -> AppAttestError {
    match error {
        AppAttestStoreError::Rejected => AppAttestError::Rejected,
        AppAttestStoreError::Unavailable => AppAttestError::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Mutex};

    use super::*;

    #[derive(Clone)]
    struct FakeCryptography {
        counter: u32,
    }

    impl AppAttestCryptography for FakeCryptography {
        fn verify_attestation(
            &self,
            _attestation_base64: &str,
            _key_id_base64: &str,
            client_data: &[u8],
        ) -> Result<VerifiedAppAttestation, AppAttestError> {
            validate_transcript(client_data)?;
            VerifiedAppAttestation::new([7; 32], vec![8; 65])
        }

        fn verify_assertion(
            &self,
            _assertion_base64: &str,
            client_data: &[u8],
            _public_key: &[u8],
            _previous_counter: u32,
            _challenge: &str,
        ) -> Result<VerifiedAppAssertion, AppAttestError> {
            validate_transcript(client_data)?;
            VerifiedAppAssertion::new(self.counter)
        }
    }

    struct MemoryStore(Mutex<MemoryState>);

    struct MemoryState {
        challenges: HashMap<Uuid, AppAttestChallenge>,
        installation: AppAttestInstallation,
    }

    #[async_trait]
    impl AppAttestAuthorityStore for MemoryStore {
        async fn installation(
            &self,
            installation_id: Uuid,
            _now_millis: u64,
        ) -> Result<AppAttestInstallation, AppAttestStoreError> {
            let state = self
                .0
                .lock()
                .map_err(|_| AppAttestStoreError::Unavailable)?;
            if state.installation.id != installation_id {
                return Err(AppAttestStoreError::Rejected);
            }
            Ok(state.installation.clone())
        }

        async fn consume_enrollment_challenge(
            &self,
            challenge: &AppAttestChallenge,
            now_millis: u64,
        ) -> Result<(), AppAttestStoreError> {
            let mut state = self
                .0
                .lock()
                .map_err(|_| AppAttestStoreError::Unavailable)?;
            consume_challenge(&mut state, challenge, now_millis)
        }

        async fn consume_assertion_and_advance(
            &self,
            installation_id: Uuid,
            challenge: &AppAttestChallenge,
            expected_counter: u32,
            next_counter: u32,
            now_millis: u64,
        ) -> Result<(), AppAttestStoreError> {
            let mut state = self
                .0
                .lock()
                .map_err(|_| AppAttestStoreError::Unavailable)?;
            if state.installation.id != installation_id
                || state.installation.assertion_counter != expected_counter
                || next_counter <= expected_counter
            {
                return Err(AppAttestStoreError::Rejected);
            }
            consume_challenge(&mut state, challenge, now_millis)?;
            state.installation.assertion_counter = next_counter;
            Ok(())
        }
    }

    fn consume_challenge(
        state: &mut MemoryState,
        challenge: &AppAttestChallenge,
        now_millis: u64,
    ) -> Result<(), AppAttestStoreError> {
        let stored = state
            .challenges
            .remove(&challenge.id)
            .ok_or(AppAttestStoreError::Rejected)?;
        if stored != *challenge || now_millis > stored.expires_at_millis {
            return Err(AppAttestStoreError::Rejected);
        }
        Ok(())
    }

    fn challenge(id: u128) -> AppAttestChallenge {
        AppAttestChallenge::new(Uuid::from_u128(id), [id as u8; 32], 20_000).expect("challenge")
    }

    fn store(challenges: Vec<AppAttestChallenge>) -> MemoryStore {
        MemoryStore(Mutex::new(MemoryState {
            challenges: challenges
                .into_iter()
                .map(|challenge| (challenge.id, challenge))
                .collect(),
            installation: AppAttestInstallation::new(
                Uuid::from_u128(9),
                ApprovedPushProfile::BuzzIosProduction,
                vec![4; 65],
                4,
            )
            .expect("installation"),
        }))
    }

    #[tokio::test]
    async fn enrollment_challenge_is_single_use() {
        let challenge = challenge(1);
        let adapter = AppAttestAdapter::new(
            FakeCryptography { counter: 5 },
            store(vec![challenge.clone()]),
        );
        adapter
            .admit_enrollment(
                &challenge,
                "attestation",
                "key",
                b"enroll transcript",
                10_000,
            )
            .await
            .expect("first enrollment");
        assert_eq!(
            adapter
                .admit_enrollment(
                    &challenge,
                    "attestation",
                    "key",
                    b"enroll transcript",
                    10_000
                )
                .await,
            Err(AppAttestError::Rejected)
        );
    }

    #[tokio::test]
    async fn assertion_counter_profile_and_challenge_replay_fail_closed() {
        let first = challenge(2);
        let second = challenge(3);
        let adapter = AppAttestAdapter::new(
            FakeCryptography { counter: 5 },
            store(vec![first.clone(), second.clone()]),
        );
        adapter
            .admit_assertion(
                Uuid::from_u128(9),
                ApprovedPushProfile::BuzzIosProduction,
                &first,
                "assertion",
                b"rotate transcript",
                10_000,
            )
            .await
            .expect("first assertion");
        assert_eq!(
            adapter
                .admit_assertion(
                    Uuid::from_u128(9),
                    ApprovedPushProfile::BuzzIosProduction,
                    &first,
                    "assertion",
                    b"rotate transcript",
                    10_000,
                )
                .await,
            Err(AppAttestError::Rejected)
        );
        assert_eq!(
            adapter
                .admit_assertion(
                    Uuid::from_u128(9),
                    ApprovedPushProfile::BuzzIosSandbox,
                    &second,
                    "assertion",
                    b"rotate transcript",
                    10_000,
                )
                .await,
            Err(AppAttestError::Rejected)
        );
        assert_eq!(
            adapter
                .admit_assertion(
                    Uuid::from_u128(9),
                    ApprovedPushProfile::BuzzIosProduction,
                    &second,
                    "assertion",
                    b"rotate transcript",
                    10_000,
                )
                .await,
            Err(AppAttestError::Rejected)
        );
    }
}
