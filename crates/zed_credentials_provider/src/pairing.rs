use collaboration_domain::{CommunityId, NostrPublicKey, ProfileId, ServiceAccountId};
use credentials_provider::CredentialsProvider;
use gpui::AsyncApp;
use nostr_compat::pairing::{PairingPayload, PairingPayloadType};
use secp256k1::{Keypair, Secp256k1, SecretKey, XOnlyPublicKey};
use sha2::{Digest as _, Sha256};
use zeroize::Zeroizing;

use crate::nostr_import::{
    CredentialsProviderSigningKeyStore, NostrImportError, ProtectedSigningKeyStore,
    nostr_credential_identifier, persist_new_signing_key, verify_stored_signing_key,
};

const PAIRING_IMPORT_CHALLENGE_DOMAIN: &[u8] = b"zed.collaboration.nostr-pairing-import.v1\0";
const MAX_PAIRING_NSEC_BYTES: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedIdentityImportRequest {
    pub community_id: CommunityId,
    pub service_account_id: ServiceAccountId,
    pub profile_id: ProfileId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairingImportDisposition {
    Imported,
    AlreadyPresent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedPairedIdentity {
    pub credential_identifier: String,
    pub public_key: NostrPublicKey,
    pub disposition: PairingImportDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PairingCredentialError {
    #[error("pairing payload type is not supported for local signing-key custody")]
    UnsupportedPayload,
    #[error("paired signing identity is malformed or unsupported")]
    InvalidIdentity,
    #[error("protected signing-key storage is unavailable")]
    ProtectedStorageUnavailable,
    #[error("protected signing-key destination already contains another value")]
    DestinationConflict,
    #[error("protected signing-key read-back verification failed")]
    ReadbackMismatch,
    #[error("failed to remove an unverified signing-key destination")]
    CleanupFailed,
}

pub async fn import_paired_identity(
    provider: &dyn CredentialsProvider,
    request: &PairedIdentityImportRequest,
    payload: &PairingPayload,
    cx: &AsyncApp,
) -> Result<ImportedPairedIdentity, PairingCredentialError> {
    let store = CredentialsProviderSigningKeyStore::new(provider, cx);
    import_paired_identity_with_store(&store, request, payload).await
}

async fn import_paired_identity_with_store(
    store: &dyn ProtectedSigningKeyStore,
    request: &PairedIdentityImportRequest,
    payload: &PairingPayload,
) -> Result<ImportedPairedIdentity, PairingCredentialError> {
    let secret = parse_paired_secret(payload)?;
    let public_key_bytes = public_key(&secret);
    let public_key = NostrPublicKey::from_bytes(public_key_bytes);
    let credential_identifier = nostr_credential_identifier(
        request.community_id.as_uuid(),
        request.service_account_id.get(),
        request.profile_id.as_uuid(),
        public_key_bytes,
    );
    let challenge_digest =
        pairing_import_challenge_digest(request, &credential_identifier, public_key_bytes);

    if store
        .read(&credential_identifier)
        .await
        .map_err(|_| PairingCredentialError::ProtectedStorageUnavailable)?
        .is_some()
    {
        verify_stored_signing_key(
            store,
            &credential_identifier,
            public_key_bytes,
            challenge_digest,
        )
        .await
        .map_err(|error| match error {
            NostrImportError::ProtectedStorageUnavailable => {
                PairingCredentialError::ProtectedStorageUnavailable
            }
            _ => PairingCredentialError::DestinationConflict,
        })?;
        return Ok(ImportedPairedIdentity {
            credential_identifier,
            public_key,
            disposition: PairingImportDisposition::AlreadyPresent,
        });
    }

    persist_new_signing_key(
        store,
        &credential_identifier,
        public_key_bytes,
        &secret,
        challenge_digest,
    )
    .await
    .map_err(map_import_error)?;

    Ok(ImportedPairedIdentity {
        credential_identifier,
        public_key,
        disposition: PairingImportDisposition::Imported,
    })
}

fn parse_paired_secret(payload: &PairingPayload) -> Result<SecretKey, PairingCredentialError> {
    if payload.payload_type() != PairingPayloadType::Nsec {
        return Err(PairingCredentialError::UnsupportedPayload);
    }
    let encoded = payload.secret();
    if encoded.is_empty()
        || encoded.len() > MAX_PAIRING_NSEC_BYTES
        || encoded.trim() != encoded
        || !encoded.starts_with("nsec1")
    {
        return Err(PairingCredentialError::InvalidIdentity);
    }
    let (human_readable_part, bytes) =
        bech32::decode(encoded).map_err(|_| PairingCredentialError::InvalidIdentity)?;
    let bytes = Zeroizing::new(bytes);
    if human_readable_part
        != bech32::Hrp::parse("nsec").map_err(|_| PairingCredentialError::InvalidIdentity)?
        || bytes.len() != 32
    {
        return Err(PairingCredentialError::InvalidIdentity);
    }
    SecretKey::from_slice(bytes.as_ref()).map_err(|_| PairingCredentialError::InvalidIdentity)
}

fn public_key(secret: &SecretKey) -> [u8; 32] {
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, secret);
    XOnlyPublicKey::from_keypair(&keypair).0.serialize()
}

fn pairing_import_challenge_digest(
    request: &PairedIdentityImportRequest,
    credential_identifier: &str,
    public_key: [u8; 32],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(PAIRING_IMPORT_CHALLENGE_DOMAIN);
    digest.update(request.community_id.as_uuid().as_bytes());
    digest.update(request.service_account_id.get().to_be_bytes());
    digest.update(request.profile_id.as_uuid().as_bytes());
    digest.update(public_key);
    digest.update((credential_identifier.len() as u64).to_be_bytes());
    digest.update(credential_identifier.as_bytes());
    digest.finalize().into()
}

fn map_import_error(error: NostrImportError) -> PairingCredentialError {
    match error {
        NostrImportError::ProtectedStorageUnavailable => {
            PairingCredentialError::ProtectedStorageUnavailable
        }
        NostrImportError::ReadbackMismatch => PairingCredentialError::ReadbackMismatch,
        NostrImportError::CleanupFailed => PairingCredentialError::CleanupFailed,
        _ => PairingCredentialError::DestinationConflict,
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

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    enum StoreFailure {
        #[default]
        None,
        Locked,
        WriteInterruptedAfterCommit,
    }

    #[derive(Default)]
    struct MemoryStore {
        entries: Mutex<HashMap<String, (String, Vec<u8>)>>,
        failure: StoreFailure,
        writes: Mutex<usize>,
    }

    impl MemoryStore {
        fn with_failure(failure: StoreFailure) -> Self {
            Self {
                failure,
                ..Self::default()
            }
        }

        fn insert(&self, identifier: impl Into<String>, username: String, secret: Vec<u8>) {
            self.entries
                .lock()
                .expect("memory store lock")
                .insert(identifier.into(), (username, secret));
        }

        fn entry(&self, identifier: &str) -> Option<(String, Vec<u8>)> {
            self.entries
                .lock()
                .expect("memory store lock")
                .get(identifier)
                .cloned()
        }

        fn writes(&self) -> usize {
            *self.writes.lock().expect("write counter lock")
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
                if self.failure == StoreFailure::Locked {
                    return Err(ProtectedSigningKeyStoreError::Unavailable);
                }
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
                *self.writes.lock().expect("write counter lock") += 1;
                if self.failure == StoreFailure::Locked {
                    return Err(ProtectedSigningKeyStoreError::Unavailable);
                }
                self.entries.lock().expect("memory store lock").insert(
                    credential_identifier.to_owned(),
                    (username.to_owned(), secret.to_vec()),
                );
                if self.failure == StoreFailure::WriteInterruptedAfterCommit {
                    return Err(ProtectedSigningKeyStoreError::Unavailable);
                }
                Ok(())
            })
        }

        fn delete<'a>(
            &'a self,
            credential_identifier: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProtectedSigningKeyStoreError>> + 'a>> {
            Box::pin(async move {
                if self.failure == StoreFailure::Locked {
                    return Err(ProtectedSigningKeyStoreError::Unavailable);
                }
                self.entries
                    .lock()
                    .expect("memory store lock")
                    .remove(credential_identifier);
                Ok(())
            })
        }
    }

    fn request() -> PairedIdentityImportRequest {
        PairedIdentityImportRequest {
            community_id: CommunityId::from_uuid(uuid::Uuid::from_u128(1)),
            service_account_id: ServiceAccountId::new(7),
            profile_id: ProfileId::from_uuid(uuid::Uuid::from_u128(2)),
        }
    }

    fn payload(secret: [u8; 32]) -> PairingPayload {
        let encoded = bech32::encode::<bech32::Bech32>(
            bech32::Hrp::parse("nsec").expect("nsec HRP"),
            &secret,
        )
        .expect("fixture nsec");
        PairingPayload::new(PairingPayloadType::Nsec, Zeroizing::new(encoded))
            .expect("pairing payload")
    }

    fn destination(request: &PairedIdentityImportRequest, secret: [u8; 32]) -> String {
        let secret = SecretKey::from_slice(&secret).expect("fixture secret");
        nostr_credential_identifier(
            request.community_id.as_uuid(),
            request.service_account_id.get(),
            request.profile_id.as_uuid(),
            public_key(&secret),
        )
    }

    fn prior_fixture(store: &MemoryStore) -> (&'static str, Vec<u8>) {
        let identifier = "zed-nostr://credential/v1/prior";
        let secret = vec![9; 32];
        store.insert(identifier, "prior-public-key".into(), secret.clone());
        (identifier, secret)
    }

    #[test]
    fn verified_pairing_import_uses_canonical_storage_and_round_trip() {
        let secret = [1; 32];
        let request = request();
        let payload = payload(secret);
        let store = MemoryStore::default();

        let imported = futures::executor::block_on(import_paired_identity_with_store(
            &store, &request, &payload,
        ))
        .expect("verified import");

        let expected_public_key = public_key(&SecretKey::from_slice(&secret).expect("fixture"));
        assert_eq!(
            imported.public_key,
            NostrPublicKey::from_bytes(expected_public_key)
        );
        assert_eq!(imported.disposition, PairingImportDisposition::Imported);
        assert_eq!(
            store.entry(&imported.credential_identifier),
            Some((hex::encode(expected_public_key), secret.to_vec()))
        );
        assert_eq!(store.writes(), 1);
    }

    #[test]
    fn interrupted_write_is_reconciled_without_changing_prior_credentials() {
        let secret = [1; 32];
        let request = request();
        let payload = payload(secret);
        let store = MemoryStore::with_failure(StoreFailure::WriteInterruptedAfterCommit);
        let (prior_identifier, prior_secret) = prior_fixture(&store);

        let error = futures::executor::block_on(import_paired_identity_with_store(
            &store, &request, &payload,
        ))
        .expect_err("interrupted import fails closed");

        assert_eq!(error, PairingCredentialError::ProtectedStorageUnavailable);
        assert_eq!(
            store.entry(prior_identifier).map(|entry| entry.1),
            Some(prior_secret)
        );
        assert_eq!(store.entry(&destination(&request, secret)), None);
    }

    #[test]
    fn locked_keyring_preserves_prior_credentials_without_attempting_a_write() {
        let secret = [1; 32];
        let request = request();
        let payload = payload(secret);
        let store = MemoryStore::with_failure(StoreFailure::Locked);
        let (prior_identifier, prior_secret) = prior_fixture(&store);

        let error = futures::executor::block_on(import_paired_identity_with_store(
            &store, &request, &payload,
        ))
        .expect_err("locked keyring fails closed");

        assert_eq!(error, PairingCredentialError::ProtectedStorageUnavailable);
        assert_eq!(
            store.entry(prior_identifier).map(|entry| entry.1),
            Some(prior_secret)
        );
        assert_eq!(store.writes(), 0);
    }

    #[test]
    fn source_payload_is_preserved_on_success_and_failure() {
        let secret = [1; 32];
        let request = request();
        let payload = payload(secret);
        let source_before = payload.secret().to_owned();
        let store = MemoryStore::default();

        futures::executor::block_on(import_paired_identity_with_store(
            &store, &request, &payload,
        ))
        .expect("first import");
        let repeated = futures::executor::block_on(import_paired_identity_with_store(
            &store, &request, &payload,
        ))
        .expect("idempotent import");

        assert_eq!(
            repeated.disposition,
            PairingImportDisposition::AlreadyPresent
        );
        assert_eq!(payload.secret(), source_before);

        let unsupported = PairingPayload::new(
            PairingPayloadType::Bunker,
            Zeroizing::new("bunker://redacted".into()),
        )
        .expect("unsupported fixture");
        assert_eq!(
            futures::executor::block_on(import_paired_identity_with_store(
                &MemoryStore::default(),
                &request,
                &unsupported,
            )),
            Err(PairingCredentialError::UnsupportedPayload)
        );
        assert_eq!(unsupported.secret(), "bunker://redacted");
    }
}
