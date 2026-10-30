use std::future::Future;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

use bech32::Hrp;
use credentials_provider::CredentialsProvider;
use gpui::AsyncApp;
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use sha2::{Digest as _, Sha256};
use uuid::Uuid;
use zeroize::Zeroizing;

const CHALLENGE_DOMAIN: &[u8] = b"zed.collaboration.nostr-import.v1\0";
const MAX_CHALLENGE_LIFETIME_SECONDS: i64 = 5 * 60;
const MAX_SOURCE_IDENTIFIER_BYTES: usize = 1024;
const MAX_ENCODED_SECRET_BYTES: usize = 256;
const NSEC_HRP: &str = "nsec";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NostrImportChallenge {
    nonce: [u8; 32],
    issued_at: i64,
    expires_at: i64,
}

impl NostrImportChallenge {
    pub fn new(nonce: [u8; 32], issued_at: i64, expires_at: i64) -> Self {
        Self {
            nonce,
            issued_at,
            expires_at,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzSigningKeyImportRequest {
    pub community_id: Uuid,
    pub service_account_id: u64,
    pub profile_id: Uuid,
    pub source_identifier: String,
    pub expected_public_key: [u8; 32],
    pub challenge: NostrImportChallenge,
}

impl BuzzSigningKeyImportRequest {
    pub fn credential_identifier(&self) -> String {
        nostr_credential_identifier(
            self.community_id,
            self.service_account_id,
            self.profile_id,
            self.expected_public_key,
        )
    }

    fn validate(&self, now: i64) -> Result<(), NostrImportError> {
        if self.source_identifier.is_empty()
            || self.source_identifier.trim() != self.source_identifier
            || self.source_identifier.len() > MAX_SOURCE_IDENTIFIER_BYTES
            || self.source_identifier.chars().any(char::is_control)
        {
            return Err(NostrImportError::InvalidSourceIdentifier);
        }
        if self.challenge.nonce == [0; 32]
            || self.challenge.expires_at <= self.challenge.issued_at
            || self.challenge.expires_at - self.challenge.issued_at > MAX_CHALLENGE_LIFETIME_SECONDS
        {
            return Err(NostrImportError::InvalidChallenge);
        }
        if now < self.challenge.issued_at || now >= self.challenge.expires_at {
            return Err(NostrImportError::ExpiredChallenge);
        }
        XOnlyPublicKey::from_slice(&self.expected_public_key)
            .map_err(|_| NostrImportError::InvalidExpectedPublicKey)?;
        Ok(())
    }

    fn challenge_digest(&self, credential_identifier: &str) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(CHALLENGE_DOMAIN);
        digest.update(self.community_id.as_bytes());
        digest.update(self.service_account_id.to_be_bytes());
        digest.update(self.profile_id.as_bytes());
        digest.update(self.expected_public_key);
        digest.update(self.challenge.nonce);
        digest.update(self.challenge.issued_at.to_be_bytes());
        digest.update(self.challenge.expires_at.to_be_bytes());
        digest.update((self.source_identifier.len() as u64).to_be_bytes());
        digest.update(self.source_identifier.as_bytes());
        digest.update((credential_identifier.len() as u64).to_be_bytes());
        digest.update(credential_identifier.as_bytes());
        digest.finalize().into()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportDisposition {
    Imported,
    AlreadyPresent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedSigningKey {
    pub credential_identifier: String,
    pub public_key: [u8; 32],
    pub disposition: ImportDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SigningKeySourceError {
    #[error("Buzz signing-key source is unavailable")]
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum ProtectedSigningKeyStoreError {
    #[error("protected signing-key storage is unavailable")]
    Unavailable,
}

pub trait BuzzSigningKeySource: Send + Sync {
    fn read_signing_key<'a>(
        &'a self,
        source_identifier: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Zeroizing<Vec<u8>>>, SigningKeySourceError>> + 'a>>;
}

pub(crate) struct StoredSigningKey {
    username: String,
    secret: Zeroizing<Vec<u8>>,
}

impl StoredSigningKey {
    pub(crate) fn new(username: String, secret: Zeroizing<Vec<u8>>) -> Self {
        Self { username, secret }
    }

    pub(crate) fn username(&self) -> &str {
        &self.username
    }

    pub(crate) fn secret(&self) -> &[u8] {
        &self.secret
    }
}

pub(crate) trait ProtectedSigningKeyStore {
    fn read<'a>(
        &'a self,
        credential_identifier: &'a str,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<StoredSigningKey>, ProtectedSigningKeyStoreError>>
                + 'a,
        >,
    >;

    fn write<'a>(
        &'a self,
        credential_identifier: &'a str,
        username: &'a str,
        secret: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), ProtectedSigningKeyStoreError>> + 'a>>;

    fn delete<'a>(
        &'a self,
        credential_identifier: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProtectedSigningKeyStoreError>> + 'a>>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NostrImportError {
    #[error("invalid Buzz signing-key source identifier")]
    InvalidSourceIdentifier,
    #[error("invalid signing challenge")]
    InvalidChallenge,
    #[error("signing challenge is not currently valid")]
    ExpiredChallenge,
    #[error("system clock is unavailable")]
    ClockUnavailable,
    #[error("expected Nostr public key is invalid")]
    InvalidExpectedPublicKey,
    #[error("Buzz signing-key source is unavailable")]
    SourceUnavailable,
    #[error("Buzz signing-key source does not contain the requested key")]
    SourceMissing,
    #[error("Buzz signing-key source is corrupt or unsupported")]
    CorruptSource,
    #[error("source key failed the bound signing challenge")]
    ChallengeMismatch,
    #[error("protected signing-key storage is unavailable")]
    ProtectedStorageUnavailable,
    #[error("protected signing-key destination already contains another value")]
    DestinationConflict,
    #[error("protected signing-key read-back did not return the imported key")]
    ReadbackMismatch,
    #[error("failed to remove an unverified destination value")]
    CleanupFailed,
}

pub(crate) struct CredentialsProviderSigningKeyStore<'a> {
    provider: &'a dyn CredentialsProvider,
    cx: &'a AsyncApp,
}

impl<'a> CredentialsProviderSigningKeyStore<'a> {
    pub(crate) fn new(provider: &'a dyn CredentialsProvider, cx: &'a AsyncApp) -> Self {
        Self { provider, cx }
    }
}

impl ProtectedSigningKeyStore for CredentialsProviderSigningKeyStore<'_> {
    fn read<'a>(
        &'a self,
        credential_identifier: &'a str,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<StoredSigningKey>, ProtectedSigningKeyStoreError>>
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.provider
                .read_credentials(credential_identifier, self.cx)
                .await
                .map(|record| {
                    record.map(|(username, secret)| {
                        StoredSigningKey::new(username, Zeroizing::new(secret))
                    })
                })
                .map_err(|_| ProtectedSigningKeyStoreError::Unavailable)
        })
    }

    fn write<'a>(
        &'a self,
        credential_identifier: &'a str,
        username: &'a str,
        secret: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), ProtectedSigningKeyStoreError>> + 'a>> {
        Box::pin(async move {
            self.provider
                .write_credentials(credential_identifier, username, secret, self.cx)
                .await
                .map_err(|_| ProtectedSigningKeyStoreError::Unavailable)
        })
    }

    fn delete<'a>(
        &'a self,
        credential_identifier: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProtectedSigningKeyStoreError>> + 'a>> {
        Box::pin(async move {
            self.provider
                .delete_credentials(credential_identifier, self.cx)
                .await
                .map_err(|_| ProtectedSigningKeyStoreError::Unavailable)
        })
    }
}

pub async fn import_buzz_signing_key(
    source: &dyn BuzzSigningKeySource,
    provider: &dyn CredentialsProvider,
    request: &BuzzSigningKeyImportRequest,
    cx: &AsyncApp,
) -> Result<ImportedSigningKey, NostrImportError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
        .ok_or(NostrImportError::ClockUnavailable)?;
    let store = CredentialsProviderSigningKeyStore::new(provider, cx);
    import_buzz_signing_key_with_store(source, &store, request, now).await
}

async fn import_buzz_signing_key_with_store(
    source: &dyn BuzzSigningKeySource,
    store: &dyn ProtectedSigningKeyStore,
    request: &BuzzSigningKeyImportRequest,
    now: i64,
) -> Result<ImportedSigningKey, NostrImportError> {
    request.validate(now)?;
    let credential_identifier = request.credential_identifier();
    let expected_username = hex::encode(request.expected_public_key);
    let challenge_digest = request.challenge_digest(&credential_identifier);

    if let Some(existing) = store
        .read(&credential_identifier)
        .await
        .map_err(|_| NostrImportError::ProtectedStorageUnavailable)?
    {
        if existing.username != expected_username
            || verify_secret(
                &existing.secret,
                request.expected_public_key,
                challenge_digest,
            )
            .is_err()
        {
            return Err(NostrImportError::DestinationConflict);
        }
        return Ok(ImportedSigningKey {
            credential_identifier,
            public_key: request.expected_public_key,
            disposition: ImportDisposition::AlreadyPresent,
        });
    }

    let source_secret = source
        .read_signing_key(&request.source_identifier)
        .await
        .map_err(|_| NostrImportError::SourceUnavailable)?
        .ok_or(NostrImportError::SourceMissing)?;
    let secret = parse_secret(&source_secret)?;
    verify_secret_key(&secret, request.expected_public_key, challenge_digest)?;

    persist_new_signing_key(
        store,
        &credential_identifier,
        request.expected_public_key,
        &secret,
        challenge_digest,
    )
    .await?;

    Ok(ImportedSigningKey {
        credential_identifier,
        public_key: request.expected_public_key,
        disposition: ImportDisposition::Imported,
    })
}

pub(crate) fn nostr_credential_identifier(
    community_id: Uuid,
    service_account_id: u64,
    profile_id: Uuid,
    public_key: [u8; 32],
) -> String {
    format!(
        "zed-nostr://credential/v1/{community_id}/{service_account_id}/{profile_id}/{}",
        hex::encode(public_key)
    )
}

pub(crate) async fn persist_new_signing_key(
    store: &dyn ProtectedSigningKeyStore,
    credential_identifier: &str,
    expected_public_key: [u8; 32],
    secret: &SecretKey,
    challenge_digest: [u8; 32],
) -> Result<(), NostrImportError> {
    let expected_username = hex::encode(expected_public_key);
    let canonical_secret = Zeroizing::new(secret.secret_bytes());
    if store
        .write(
            credential_identifier,
            &expected_username,
            canonical_secret.as_ref(),
        )
        .await
        .is_err()
    {
        return Err(cleanup_unverified_destination(
            store,
            credential_identifier,
            NostrImportError::ProtectedStorageUnavailable,
        )
        .await);
    }

    let readback = match store.read(credential_identifier).await {
        Ok(Some(readback)) => readback,
        Ok(None) | Err(_) => {
            return Err(cleanup_unverified_destination(
                store,
                credential_identifier,
                NostrImportError::ReadbackMismatch,
            )
            .await);
        }
    };
    if readback.username != expected_username
        || verify_secret(&readback.secret, expected_public_key, challenge_digest).is_err()
    {
        return Err(cleanup_unverified_destination(
            store,
            credential_identifier,
            NostrImportError::ReadbackMismatch,
        )
        .await);
    }
    Ok(())
}

pub(crate) async fn verify_stored_signing_key(
    store: &dyn ProtectedSigningKeyStore,
    credential_identifier: &str,
    expected_public_key: [u8; 32],
    challenge_digest: [u8; 32],
) -> Result<(), NostrImportError> {
    let expected_username = hex::encode(expected_public_key);
    let stored = store
        .read(credential_identifier)
        .await
        .map_err(|_| NostrImportError::ProtectedStorageUnavailable)?
        .ok_or(NostrImportError::ReadbackMismatch)?;
    if stored.username != expected_username
        || verify_secret(&stored.secret, expected_public_key, challenge_digest).is_err()
    {
        return Err(NostrImportError::ReadbackMismatch);
    }
    Ok(())
}

async fn cleanup_unverified_destination(
    store: &dyn ProtectedSigningKeyStore,
    credential_identifier: &str,
    error: NostrImportError,
) -> NostrImportError {
    match store.delete(credential_identifier).await {
        Ok(()) => error,
        Err(_) => NostrImportError::CleanupFailed,
    }
}

fn verify_secret(
    encoded: &[u8],
    expected_public_key: [u8; 32],
    challenge_digest: [u8; 32],
) -> Result<(), NostrImportError> {
    let secret = parse_secret(encoded)?;
    verify_secret_key(&secret, expected_public_key, challenge_digest)
}

pub(crate) fn verify_secret_key(
    secret: &SecretKey,
    expected_public_key: [u8; 32],
    challenge_digest: [u8; 32],
) -> Result<(), NostrImportError> {
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, secret);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    if public_key.serialize() != expected_public_key {
        return Err(NostrImportError::ChallengeMismatch);
    }
    let message = Message::from_digest(challenge_digest);
    let signature = secp.sign_schnorr_no_aux_rand(&message, &keypair);
    secp.verify_schnorr(&signature, &message, &public_key)
        .map_err(|_| NostrImportError::ChallengeMismatch)
}

fn parse_secret(encoded: &[u8]) -> Result<SecretKey, NostrImportError> {
    if encoded.len() > MAX_ENCODED_SECRET_BYTES {
        return Err(NostrImportError::CorruptSource);
    }
    if encoded.len() == 32 {
        return SecretKey::from_slice(encoded).map_err(|_| NostrImportError::CorruptSource);
    }
    let text = std::str::from_utf8(encoded).map_err(|_| NostrImportError::CorruptSource)?;
    let text = text.trim();
    let secret = if text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        let mut bytes = Zeroizing::new([0; 32]);
        hex::decode_to_slice(text, bytes.as_mut()).map_err(|_| NostrImportError::CorruptSource)?;
        SecretKey::from_slice(bytes.as_ref()).map_err(|_| NostrImportError::CorruptSource)?
    } else {
        let (human_readable_part, bytes) =
            bech32::decode(text).map_err(|_| NostrImportError::CorruptSource)?;
        let bytes = Zeroizing::new(bytes);
        if human_readable_part
            != Hrp::parse(NSEC_HRP).map_err(|_| NostrImportError::CorruptSource)?
            || bytes.len() != 32
        {
            return Err(NostrImportError::CorruptSource);
        }
        SecretKey::from_slice(bytes.as_ref()).map_err(|_| NostrImportError::CorruptSource)?
    };
    Ok(secret)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;

    struct MemorySource {
        entries: Mutex<HashMap<String, Vec<u8>>>,
        unavailable: bool,
    }

    impl MemorySource {
        fn containing(identifier: &str, secret: Vec<u8>) -> Self {
            Self {
                entries: Mutex::new(HashMap::from([(identifier.to_owned(), secret)])),
                unavailable: false,
            }
        }

        fn still_contains(&self, identifier: &str) -> bool {
            self.entries
                .lock()
                .expect("source lock")
                .contains_key(identifier)
        }
    }

    impl BuzzSigningKeySource for MemorySource {
        fn read_signing_key<'a>(
            &'a self,
            source_identifier: &'a str,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Option<Zeroizing<Vec<u8>>>, SigningKeySourceError>> + 'a,
            >,
        > {
            Box::pin(async move {
                if self.unavailable {
                    return Err(SigningKeySourceError::Unavailable);
                }
                Ok(self
                    .entries
                    .lock()
                    .expect("source lock")
                    .get(source_identifier)
                    .cloned()
                    .map(Zeroizing::new))
            })
        }
    }

    #[derive(Default)]
    struct MemoryStore {
        entries: Mutex<HashMap<String, (String, Vec<u8>)>>,
        unavailable: bool,
    }

    impl MemoryStore {
        fn contains(&self, identifier: &str) -> bool {
            self.entries
                .lock()
                .expect("store lock")
                .contains_key(identifier)
        }
    }

    impl ProtectedSigningKeyStore for MemoryStore {
        fn read<'a>(
            &'a self,
            credential_identifier: &'a str,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Option<StoredSigningKey>, ProtectedSigningKeyStoreError>>
                    + 'a,
            >,
        > {
            Box::pin(async move {
                if self.unavailable {
                    return Err(ProtectedSigningKeyStoreError::Unavailable);
                }
                Ok(self
                    .entries
                    .lock()
                    .expect("store lock")
                    .get(credential_identifier)
                    .cloned()
                    .map(|(username, secret)| {
                        StoredSigningKey::new(username, Zeroizing::new(secret))
                    }))
            })
        }

        fn write<'a>(
            &'a self,
            credential_identifier: &'a str,
            username: &'a str,
            secret: &'a [u8],
        ) -> Pin<Box<dyn Future<Output = Result<(), ProtectedSigningKeyStoreError>> + 'a>> {
            Box::pin(async move {
                if self.unavailable {
                    return Err(ProtectedSigningKeyStoreError::Unavailable);
                }
                self.entries.lock().expect("store lock").insert(
                    credential_identifier.to_owned(),
                    (username.to_owned(), secret.to_vec()),
                );
                Ok(())
            })
        }

        fn delete<'a>(
            &'a self,
            credential_identifier: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProtectedSigningKeyStoreError>> + 'a>> {
            Box::pin(async move {
                if self.unavailable {
                    return Err(ProtectedSigningKeyStoreError::Unavailable);
                }
                self.entries
                    .lock()
                    .expect("store lock")
                    .remove(credential_identifier);
                Ok(())
            })
        }
    }

    fn public_key(secret: [u8; 32]) -> [u8; 32] {
        let secp = Secp256k1::new();
        let secret = SecretKey::from_slice(&secret).expect("fixture secret");
        let keypair = Keypair::from_secret_key(&secp, &secret);
        XOnlyPublicKey::from_keypair(&keypair).0.serialize()
    }

    fn request(secret: [u8; 32]) -> BuzzSigningKeyImportRequest {
        BuzzSigningKeyImportRequest {
            community_id: Uuid::from_u128(1),
            service_account_id: 7,
            profile_id: Uuid::from_u128(2),
            source_identifier: "buzz-desktop:identity".into(),
            expected_public_key: public_key(secret),
            challenge: NostrImportChallenge::new([9; 32], 100, 200),
        }
    }

    fn nsec(secret: [u8; 32]) -> Vec<u8> {
        bech32::encode::<bech32::Bech32>(Hrp::parse(NSEC_HRP).expect("nsec HRP"), &secret)
            .expect("encode fixture nsec")
            .into_bytes()
    }

    #[test]
    fn imports_nsec_after_protected_readback_and_preserves_source() {
        let secret = [1; 32];
        let request = request(secret);
        let source = MemorySource::containing(&request.source_identifier, nsec(secret));
        let store = MemoryStore::default();

        let imported = futures::executor::block_on(import_buzz_signing_key_with_store(
            &source, &store, &request, 150,
        ))
        .expect("import succeeds");

        assert_eq!(imported.disposition, ImportDisposition::Imported);
        assert!(store.contains(&request.credential_identifier()));
        assert!(source.still_contains(&request.source_identifier));
    }

    #[test]
    fn rejects_corrupt_source_without_writing_destination() {
        let request = request([1; 32]);
        let source = MemorySource::containing(&request.source_identifier, b"nsec1corrupt".to_vec());
        let store = MemoryStore::default();

        let error = futures::executor::block_on(import_buzz_signing_key_with_store(
            &source, &store, &request, 150,
        ))
        .expect_err("corrupt source rejected");

        assert_eq!(error, NostrImportError::CorruptSource);
        assert!(!store.contains(&request.credential_identifier()));
        assert!(source.still_contains(&request.source_identifier));
    }

    #[test]
    fn unavailable_protected_storage_does_not_remove_source() {
        let secret = [1; 32];
        let request = request(secret);
        let source = MemorySource::containing(&request.source_identifier, nsec(secret));
        let store = MemoryStore {
            unavailable: true,
            ..MemoryStore::default()
        };

        let error = futures::executor::block_on(import_buzz_signing_key_with_store(
            &source, &store, &request, 150,
        ))
        .expect_err("unavailable storage rejected");

        assert_eq!(error, NostrImportError::ProtectedStorageUnavailable);
        assert!(source.still_contains(&request.source_identifier));
    }

    #[test]
    fn challenge_public_key_mismatch_never_writes_or_removes_source() {
        let source_secret = [1; 32];
        let request = request([2; 32]);
        let source = MemorySource::containing(&request.source_identifier, nsec(source_secret));
        let store = MemoryStore::default();

        let error = futures::executor::block_on(import_buzz_signing_key_with_store(
            &source, &store, &request, 150,
        ))
        .expect_err("mismatched challenge rejected");

        assert_eq!(error, NostrImportError::ChallengeMismatch);
        assert!(!store.contains(&request.credential_identifier()));
        assert!(source.still_contains(&request.source_identifier));
    }

    #[test]
    fn repeated_import_is_idempotent_and_preserves_source() {
        let secret = [1; 32];
        let request = request(secret);
        let source = MemorySource::containing(&request.source_identifier, nsec(secret));
        let store = MemoryStore::default();
        futures::executor::block_on(import_buzz_signing_key_with_store(
            &source, &store, &request, 150,
        ))
        .expect("first import succeeds");

        let imported = futures::executor::block_on(import_buzz_signing_key_with_store(
            &source, &store, &request, 150,
        ))
        .expect("second import succeeds");

        assert_eq!(imported.disposition, ImportDisposition::AlreadyPresent);
        assert!(source.still_contains(&request.source_identifier));
    }
}
