use collaboration_domain::{BindingId, CommunityId, NostrPublicKey, ProfileId, ServiceAccountId};
use credentials_provider::CredentialsProvider;
use gpui::AsyncApp;
use nostr::nips::nip19::{FromBech32, ToBech32};
use nostr::nips::nip49::{EncryptedSecretKey, KeySecurity};
use secp256k1::{Keypair, Secp256k1, SecretKey, XOnlyPublicKey};
use sha2::{Digest as _, Sha256};
use zeroize::{Zeroize, Zeroizing};

use crate::nostr_import::{
    CredentialsProviderSigningKeyStore, NostrImportError, ProtectedSigningKeyStore,
    nostr_credential_identifier, persist_new_signing_key, verify_stored_signing_key,
};
use crate::nostr_lifecycle::{
    ActiveSigningCredential, IdentityLifecycleRepository, NostrLifecycleError,
    resolve_active_signing_credential,
};

const BACKUP_CHALLENGE_DOMAIN: &[u8] = b"zed.collaboration.nostr-backup-restore.v1\0";
const NCRYPTSEC_HRP: &str = "ncryptsec1";
const MAX_BACKUP_BYTES: usize = 512;
const MAX_PASSWORD_BYTES: usize = 1024;
const MIN_PASSWORD_CHARACTERS: usize = 12;

pub const BACKUP_LOG_N: u8 = 18;
pub const MAX_RESTORE_LOG_N: u8 = BACKUP_LOG_N;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoreSigningKeyRequest {
    pub community_id: CommunityId,
    pub service_account_id: ServiceAccountId,
    pub profile_id: ProfileId,
    pub expected_public_key: NostrPublicKey,
}

impl RestoreSigningKeyRequest {
    pub fn credential_identifier(&self) -> String {
        nostr_credential_identifier(
            self.community_id.as_uuid(),
            self.service_account_id.get(),
            self.profile_id.as_uuid(),
            *self.expected_public_key.as_bytes(),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RestoreDisposition {
    Restored,
    AlreadyPresent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoredSigningKey {
    pub credential_identifier: String,
    pub public_key: NostrPublicKey,
    pub disposition: RestoreDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NostrBackupError {
    #[error("current signing identity is unavailable")]
    CurrentIdentityUnavailable,
    #[error("protected signing-key storage is unavailable")]
    ProtectedStorageUnavailable,
    #[error("protected signing-key storage does not match the current identity")]
    ProtectedStorageMismatch,
    #[error("encrypted signing-key backup could not be created")]
    BackupCreationFailed,
    #[error("backup passphrase must contain at least 12 characters")]
    PasswordTooShort,
    #[error("backup passphrase is too large")]
    PasswordTooLarge,
    #[error("encrypted signing-key backup is malformed or truncated")]
    MalformedBackup,
    #[error("encrypted signing-key backup requests an unsupported KDF cost")]
    UnsupportedKdfCost,
    #[error("wrong backup password or damaged signing-key backup")]
    WrongPasswordOrDamagedBackup,
    #[error("restored signing key does not match the expected identity")]
    IdentityMismatch,
    #[error("protected signing-key destination already contains another value")]
    DestinationConflict,
    #[error("protected signing-key read-back verification failed")]
    ReadbackMismatch,
    #[error("failed to remove an unverified signing-key destination")]
    CleanupFailed,
}

pub async fn create_nostr_backup(
    provider: &dyn CredentialsProvider,
    repository: &dyn IdentityLifecycleRepository,
    community_id: CommunityId,
    binding_id: BindingId,
    password: Zeroizing<String>,
    cx: &AsyncApp,
) -> Result<String, NostrBackupError> {
    validate_export_password(&password)?;
    let active =
        resolve_active_signing_credential(provider, repository, community_id, binding_id, cx)
            .await
            .map_err(map_lifecycle_error)?;
    let store = CredentialsProviderSigningKeyStore::new(provider, cx);
    let secret = read_active_secret(&store, &active).await?;
    cx.background_executor()
        .spawn(async move { encrypt_verified_backup(secret, password, BACKUP_LOG_N) })
        .await
}

pub async fn restore_nostr_backup(
    provider: &dyn CredentialsProvider,
    request: &RestoreSigningKeyRequest,
    backup: String,
    password: Zeroizing<String>,
    cx: &AsyncApp,
) -> Result<RestoredSigningKey, NostrBackupError> {
    validate_restore_password(&password)?;
    validate_backup_text(&backup)?;
    let expected_public_key = *request.expected_public_key.as_bytes();
    let backup_digest: [u8; 32] = Sha256::digest(backup.as_bytes()).into();
    let secret = cx
        .background_executor()
        .spawn(async move { decrypt_backup(&backup, &password, expected_public_key) })
        .await?;
    let store = CredentialsProviderSigningKeyStore::new(provider, cx);
    restore_secret_with_store(&store, request, secret, backup_digest).await
}

async fn read_active_secret(
    store: &dyn ProtectedSigningKeyStore,
    active: &ActiveSigningCredential,
) -> Result<Zeroizing<[u8; 32]>, NostrBackupError> {
    let stored = store
        .read(&active.credential_identifier)
        .await
        .map_err(|_| NostrBackupError::ProtectedStorageUnavailable)?
        .ok_or(NostrBackupError::ProtectedStorageMismatch)?;
    let expected_public_key = *active.public_key.as_bytes();
    if stored.username() != hex::encode(expected_public_key) {
        return Err(NostrBackupError::ProtectedStorageMismatch);
    }
    let secret = SecretKey::from_slice(stored.secret())
        .map_err(|_| NostrBackupError::ProtectedStorageMismatch)?;
    if public_key(&secret) != expected_public_key {
        return Err(NostrBackupError::ProtectedStorageMismatch);
    }
    Ok(Zeroizing::new(secret.secret_bytes()))
}

fn encrypt_verified_backup(
    secret_bytes: Zeroizing<[u8; 32]>,
    password: Zeroizing<String>,
    log_n: u8,
) -> Result<String, NostrBackupError> {
    let secret = nostr::SecretKey::from_slice(secret_bytes.as_ref())
        .map_err(|_| NostrBackupError::ProtectedStorageMismatch)?;
    let encrypted = EncryptedSecretKey::new(&secret, &password, log_n, KeySecurity::Unknown)
        .map_err(|_| NostrBackupError::BackupCreationFailed)?;
    let backup = encrypted
        .to_bech32()
        .map_err(|_| NostrBackupError::BackupCreationFailed)?;
    let expected_public_key = public_key(
        &SecretKey::from_slice(secret_bytes.as_ref())
            .map_err(|_| NostrBackupError::ProtectedStorageMismatch)?,
    );
    let mut recovered = decrypt_backup(&backup, &password, expected_public_key)?;
    recovered.zeroize();
    Ok(backup)
}

fn decrypt_backup(
    backup: &str,
    password: &str,
    expected_public_key: [u8; 32],
) -> Result<Zeroizing<[u8; 32]>, NostrBackupError> {
    validate_backup_text(backup)?;
    let encrypted = EncryptedSecretKey::from_bech32(backup.trim())
        .map_err(|_| NostrBackupError::MalformedBackup)?;
    if encrypted.log_n() > MAX_RESTORE_LOG_N {
        return Err(NostrBackupError::UnsupportedKdfCost);
    }
    let recovered = encrypted
        .decrypt(password)
        .map_err(|_| NostrBackupError::WrongPasswordOrDamagedBackup)?;
    let secret = SecretKey::from_slice(recovered.as_secret_bytes())
        .map_err(|_| NostrBackupError::WrongPasswordOrDamagedBackup)?;
    if public_key(&secret) != expected_public_key {
        return Err(NostrBackupError::IdentityMismatch);
    }
    Ok(Zeroizing::new(secret.secret_bytes()))
}

async fn restore_secret_with_store(
    store: &dyn ProtectedSigningKeyStore,
    request: &RestoreSigningKeyRequest,
    secret_bytes: Zeroizing<[u8; 32]>,
    backup_digest: [u8; 32],
) -> Result<RestoredSigningKey, NostrBackupError> {
    let credential_identifier = request.credential_identifier();
    let expected_public_key = *request.expected_public_key.as_bytes();
    let challenge_digest =
        restore_challenge_digest(&credential_identifier, expected_public_key, backup_digest);
    if store
        .read(&credential_identifier)
        .await
        .map_err(|_| NostrBackupError::ProtectedStorageUnavailable)?
        .is_some()
    {
        verify_stored_signing_key(
            store,
            &credential_identifier,
            expected_public_key,
            challenge_digest,
        )
        .await
        .map_err(|error| match error {
            NostrImportError::ProtectedStorageUnavailable => {
                NostrBackupError::ProtectedStorageUnavailable
            }
            _ => NostrBackupError::DestinationConflict,
        })?;
        return Ok(RestoredSigningKey {
            credential_identifier,
            public_key: request.expected_public_key,
            disposition: RestoreDisposition::AlreadyPresent,
        });
    }
    let secret = SecretKey::from_slice(secret_bytes.as_ref())
        .map_err(|_| NostrBackupError::WrongPasswordOrDamagedBackup)?;
    persist_new_signing_key(
        store,
        &credential_identifier,
        expected_public_key,
        &secret,
        challenge_digest,
    )
    .await
    .map_err(map_import_error)?;
    Ok(RestoredSigningKey {
        credential_identifier,
        public_key: request.expected_public_key,
        disposition: RestoreDisposition::Restored,
    })
}

fn validate_export_password(password: &str) -> Result<(), NostrBackupError> {
    validate_restore_password(password)?;
    if password.chars().count() < MIN_PASSWORD_CHARACTERS {
        return Err(NostrBackupError::PasswordTooShort);
    }
    Ok(())
}

fn validate_restore_password(password: &str) -> Result<(), NostrBackupError> {
    if password.len() > MAX_PASSWORD_BYTES {
        return Err(NostrBackupError::PasswordTooLarge);
    }
    Ok(())
}

fn validate_backup_text(backup: &str) -> Result<(), NostrBackupError> {
    let trimmed = backup.trim();
    if trimmed.is_empty()
        || trimmed.len() > MAX_BACKUP_BYTES
        || !trimmed
            .get(..NCRYPTSEC_HRP.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(NCRYPTSEC_HRP))
    {
        return Err(NostrBackupError::MalformedBackup);
    }
    Ok(())
}

fn public_key(secret: &SecretKey) -> [u8; 32] {
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, secret);
    XOnlyPublicKey::from_keypair(&keypair).0.serialize()
}

fn restore_challenge_digest(
    credential_identifier: &str,
    expected_public_key: [u8; 32],
    backup_digest: [u8; 32],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(BACKUP_CHALLENGE_DOMAIN);
    digest.update((credential_identifier.len() as u64).to_be_bytes());
    digest.update(credential_identifier.as_bytes());
    digest.update(expected_public_key);
    digest.update(backup_digest);
    digest.finalize().into()
}

fn map_lifecycle_error(_error: NostrLifecycleError) -> NostrBackupError {
    NostrBackupError::CurrentIdentityUnavailable
}

fn map_import_error(error: NostrImportError) -> NostrBackupError {
    match error {
        NostrImportError::ProtectedStorageUnavailable => {
            NostrBackupError::ProtectedStorageUnavailable
        }
        NostrImportError::ReadbackMismatch => NostrBackupError::ReadbackMismatch,
        NostrImportError::CleanupFailed => NostrBackupError::CleanupFailed,
        _ => NostrBackupError::DestinationConflict,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    use super::*;
    use crate::nostr_import::{ProtectedSigningKeyStoreError, StoredSigningKey};

    const SPEC_NCRYPTSEC: &str = "ncryptsec1qgg9947rlpvqu76pj5ecreduf9jxhselq2nae2kghhvd5g7dgjtcxfqtd67p9m0w57lspw8gsq6yphnm8623nsl8xn9j4jdzz84zm3frztj3z7s35vpzmqf6ksu8r89qk5z2zxfmu5gv8th8wclt0h4p";
    const SPEC_SECRET: [u8; 32] = [
        0x35, 0x01, 0x45, 0x41, 0x35, 0x01, 0x45, 0x41, 0x35, 0x01, 0x45, 0x41, 0x35, 0x01, 0x45,
        0x3f, 0xef, 0xb0, 0x22, 0x27, 0xe4, 0x49, 0xe5, 0x7c, 0xf4, 0xd3, 0xa3, 0xce, 0x05, 0x37,
        0x86, 0x83,
    ];
    const FAST_LOG_N: u8 = 16;

    #[derive(Default)]
    struct MemoryStore {
        entries: Mutex<HashMap<String, (String, Vec<u8>)>>,
    }

    impl MemoryStore {
        fn insert(&self, identifier: String, public_key: [u8; 32], secret: [u8; 32]) {
            self.entries
                .lock()
                .expect("memory store lock")
                .insert(identifier, (hex::encode(public_key), secret.to_vec()));
        }

        fn secret(&self, identifier: &str) -> Option<Vec<u8>> {
            self.entries
                .lock()
                .expect("memory store lock")
                .get(identifier)
                .map(|(_, secret)| secret.clone())
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
                Ok(self
                    .entries
                    .lock()
                    .expect("memory store lock")
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
                self.entries.lock().expect("memory store lock").insert(
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
                self.entries
                    .lock()
                    .expect("memory store lock")
                    .remove(credential_identifier);
                Ok(())
            })
        }
    }

    fn request() -> RestoreSigningKeyRequest {
        let secret = SecretKey::from_slice(&SPEC_SECRET).expect("fixture secret");
        RestoreSigningKeyRequest {
            community_id: CommunityId::from_uuid(uuid::Uuid::from_u128(1)),
            service_account_id: ServiceAccountId::new(7),
            profile_id: ProfileId::from_uuid(uuid::Uuid::from_u128(2)),
            expected_public_key: NostrPublicKey::from_bytes(public_key(&secret)),
        }
    }

    #[test]
    fn buzz_backup_round_trips_through_canonical_storage() {
        let request = request();
        let source = MemoryStore::default();
        source.insert(
            request.credential_identifier(),
            *request.expected_public_key.as_bytes(),
            SPEC_SECRET,
        );
        let active = ActiveSigningCredential {
            credential_identifier: request.credential_identifier(),
            public_key: request.expected_public_key,
        };
        let secret = futures::executor::block_on(read_active_secret(&source, &active))
            .expect("read active secret");
        let backup = encrypt_verified_backup(
            secret,
            Zeroizing::new("correct horse battery".to_owned()),
            FAST_LOG_N,
        )
        .expect("create backup");
        let recovered = decrypt_backup(
            &backup,
            "correct horse battery",
            *request.expected_public_key.as_bytes(),
        )
        .expect("decrypt backup");
        let destination = MemoryStore::default();
        let digest = Sha256::digest(backup.as_bytes()).into();
        let restored = futures::executor::block_on(restore_secret_with_store(
            &destination,
            &request,
            recovered,
            digest,
        ))
        .expect("restore backup");

        assert_eq!(restored.disposition, RestoreDisposition::Restored);
        assert_eq!(
            destination.secret(&request.credential_identifier()),
            Some(SPEC_SECRET.to_vec())
        );
    }

    #[test]
    fn restores_the_buzz_nip49_fixture() {
        let request = request();
        assert_eq!(validate_restore_password("nostr"), Ok(()));
        assert_eq!(
            validate_export_password("nostr"),
            Err(NostrBackupError::PasswordTooShort)
        );
        let secret = decrypt_backup(
            SPEC_NCRYPTSEC,
            "nostr",
            *request.expected_public_key.as_bytes(),
        )
        .expect("decrypt Buzz fixture");
        assert_eq!(secret.as_ref(), &SPEC_SECRET);
    }

    #[test]
    fn wrong_password_and_truncated_backup_fail_closed() {
        let request = request();
        assert_eq!(
            decrypt_backup(
                SPEC_NCRYPTSEC,
                "wrong password",
                *request.expected_public_key.as_bytes(),
            ),
            Err(NostrBackupError::WrongPasswordOrDamagedBackup)
        );
        assert_eq!(
            decrypt_backup(
                &SPEC_NCRYPTSEC[..SPEC_NCRYPTSEC.len() - 10],
                "nostr",
                *request.expected_public_key.as_bytes(),
            ),
            Err(NostrBackupError::MalformedBackup)
        );
    }

    #[test]
    fn restore_does_not_overwrite_a_conflicting_destination() {
        let request = request();
        let destination = MemoryStore::default();
        let conflicting_secret = [8; 32];
        destination.insert(
            request.credential_identifier(),
            public_key(&SecretKey::from_slice(&conflicting_secret).expect("conflicting secret")),
            conflicting_secret,
        );
        let result = futures::executor::block_on(restore_secret_with_store(
            &destination,
            &request,
            Zeroizing::new(SPEC_SECRET),
            [3; 32],
        ));

        assert_eq!(result, Err(NostrBackupError::DestinationConflict));
        assert_eq!(
            destination.secret(&request.credential_identifier()),
            Some(conflicting_secret.to_vec())
        );
    }

    #[test]
    fn matching_destination_is_an_idempotent_restore() {
        let request = request();
        let destination = MemoryStore::default();
        destination.insert(
            request.credential_identifier(),
            *request.expected_public_key.as_bytes(),
            SPEC_SECRET,
        );
        let result = futures::executor::block_on(restore_secret_with_store(
            &destination,
            &request,
            Zeroizing::new(SPEC_SECRET),
            [3; 32],
        ))
        .expect("idempotent restore");

        assert_eq!(result.disposition, RestoreDisposition::AlreadyPresent);
        assert_eq!(
            destination.secret(&request.credential_identifier()),
            Some(SPEC_SECRET.to_vec())
        );
    }

    #[test]
    fn rejects_excessive_kdf_cost_before_password_work() {
        let request = request();
        let encrypted = EncryptedSecretKey::from_bech32(SPEC_NCRYPTSEC).expect("fixture backup");
        let mut payload = encrypted.as_vec();
        payload[1] = MAX_RESTORE_LOG_N + 1;
        let excessive = EncryptedSecretKey::from_slice(&payload)
            .expect("structural backup")
            .to_bech32()
            .expect("encoded backup");

        assert_eq!(
            decrypt_backup(
                &excessive,
                "redaction sentinel password",
                *request.expected_public_key.as_bytes(),
            ),
            Err(NostrBackupError::UnsupportedKdfCost)
        );
    }

    #[test]
    fn diagnostics_redact_password_secret_and_backup_bytes() {
        let password = "redaction sentinel password";
        let secret = hex::encode(SPEC_SECRET);
        for error in [
            NostrBackupError::BackupCreationFailed,
            NostrBackupError::MalformedBackup,
            NostrBackupError::UnsupportedKdfCost,
            NostrBackupError::WrongPasswordOrDamagedBackup,
            NostrBackupError::IdentityMismatch,
            NostrBackupError::DestinationConflict,
        ] {
            let diagnostic = format!("{error:?}: {error}");
            assert!(!diagnostic.contains(password));
            assert!(!diagnostic.contains(&secret));
            assert!(!diagnostic.contains(SPEC_NCRYPTSEC));
        }
    }
}
