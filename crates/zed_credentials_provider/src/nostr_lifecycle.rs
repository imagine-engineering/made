use std::future::Future;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

use collaboration_domain::{
    AccountBinding, AccountBindingFields, AgentProfile, AggregateVersion, BindingId, BindingStatus,
    BindingVerification, BindingVerificationMethod, BindingVersionReference, CommunityId,
    EvidenceReference, IdentityProfile, NostrPublicKey, OperationId, OrganizationPolicyVersion,
    PrincipalId, ProfileId, ProfileKind, ProfileRecordFields, ServiceAccountId,
};
use credentials_provider::CredentialsProvider;
use gpui::AsyncApp;
use secp256k1::{Keypair, Secp256k1, SecretKey, XOnlyPublicKey};
use sha2::{Digest as _, Sha256};
use zeroize::Zeroizing;

use crate::nostr_import::{
    CredentialsProviderSigningKeyStore, NostrImportError, ProtectedSigningKeyStore,
    nostr_credential_identifier, persist_new_signing_key, verify_stored_signing_key,
};

const CREDENTIAL_PROBE_IDENTIFIER: &str = "zed-nostr://credential/probe/v1";
const GENERATION_CHALLENGE_DOMAIN: &[u8] = b"zed.collaboration.nostr-generation.v1\0";
const RESOLUTION_CHALLENGE_DOMAIN: &[u8] = b"zed.collaboration.nostr-resolution.v1\0";
const MAX_SECRET_GENERATION_ATTEMPTS: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NewProfileKind {
    Human,
    Agent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CreateSigningIdentityRequest {
    pub binding_id: BindingId,
    pub community_id: CommunityId,
    pub service_account_id: ServiceAccountId,
    pub profile_id: ProfileId,
    pub profile_kind: NewProfileKind,
    pub organization_policy_version: OrganizationPolicyVersion,
    pub actor_principal_id: PrincipalId,
    pub audit_reference: OperationId,
    pub evidence_reference: EvidenceReference,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RotateSigningIdentityRequest {
    pub current_binding: AccountBinding,
    pub current_profile: IdentityProfile,
    pub successor_binding_id: BindingId,
    pub successor_profile_id: ProfileId,
    pub organization_policy_version: OrganizationPolicyVersion,
    pub actor_principal_id: PrincipalId,
    pub audit_reference: OperationId,
    pub evidence_reference: EvidenceReference,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TerminalBindingRequest {
    pub current_binding: AccountBinding,
    pub organization_policy_version: OrganizationPolicyVersion,
    pub actor_principal_id: PrincipalId,
    pub audit_reference: OperationId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IdentityLifecycleMutation {
    Create {
        binding: AccountBinding,
        profile: IdentityProfile,
    },
    Rotate {
        predecessor: AccountBinding,
        historical_profile: IdentityProfile,
        successor: AccountBinding,
        successor_profile: IdentityProfile,
    },
    Revoke {
        binding: AccountBinding,
    },
    Archive {
        binding: AccountBinding,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppliedIdentityLifecycle {
    pub mutation: IdentityLifecycleMutation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSigningCredential {
    pub credential_identifier: String,
    pub public_key: NostrPublicKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum IdentityLifecycleRepositoryError {
    #[error("identity lifecycle repository rejected the optimistic mutation")]
    Rejected,
    #[error("identity lifecycle repository cannot determine whether the mutation committed")]
    OutcomeUnknown,
    #[error("identity lifecycle repository is unavailable")]
    Unavailable,
}

pub trait IdentityLifecycleRepository: Send + Sync {
    fn commit<'a>(
        &'a self,
        mutation: IdentityLifecycleMutation,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdentityLifecycleRepositoryError>> + 'a>>;

    fn current_binding<'a>(
        &'a self,
        community_id: CommunityId,
        binding_id: BindingId,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AccountBinding>, IdentityLifecycleRepositoryError>>
                + 'a,
        >,
    >;
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NostrLifecycleError {
    #[error("system clock is unavailable")]
    ClockUnavailable,
    #[error("operating-system entropy is unavailable")]
    EntropyUnavailable,
    #[error("protected signing-key storage is unavailable")]
    ProtectedStorageUnavailable,
    #[error("protected signing-key verification failed")]
    ProtectedStorageVerificationFailed,
    #[error("protected signing-key destination already exists")]
    CredentialConflict,
    #[error("identity lifecycle repository rejected the mutation")]
    RepositoryRejected,
    #[error("identity lifecycle repository outcome is unknown and requires reconciliation")]
    RepositoryOutcomeUnknown,
    #[error(
        "identity lifecycle repository outcome is unknown for generated key {credential_identifier} and requires reconciliation"
    )]
    GeneratedKeyRepositoryOutcomeUnknown { credential_identifier: String },
    #[error("identity lifecycle repository is unavailable")]
    RepositoryUnavailable,
    #[error("failed to clean up uncommitted signing key {credential_identifier}")]
    CleanupFailed { credential_identifier: String },
    #[error("current binding is not active")]
    InactiveBinding,
    #[error("current binding and profile do not identify the same author")]
    BindingProfileMismatch,
    #[error("identity aggregate version is exhausted")]
    VersionExhausted,
    #[error("identity lifecycle transition is invalid")]
    InvalidTransition,
}

trait SigningKeyEntropy {
    fn generate_secret(&self) -> Result<SecretKey, NostrLifecycleError>;
    fn generate_nonce(&self) -> Result<Zeroizing<[u8; 32]>, NostrLifecycleError>;
}

struct OperatingSystemEntropy;

impl SigningKeyEntropy for OperatingSystemEntropy {
    fn generate_secret(&self) -> Result<SecretKey, NostrLifecycleError> {
        for _ in 0..MAX_SECRET_GENERATION_ATTEMPTS {
            let mut bytes = Zeroizing::new([0; 32]);
            getrandom::fill(bytes.as_mut()).map_err(|_| NostrLifecycleError::EntropyUnavailable)?;
            if let Ok(secret) = SecretKey::from_slice(bytes.as_ref()) {
                return Ok(secret);
            }
        }
        Err(NostrLifecycleError::EntropyUnavailable)
    }

    fn generate_nonce(&self) -> Result<Zeroizing<[u8; 32]>, NostrLifecycleError> {
        let mut nonce = Zeroizing::new([0; 32]);
        getrandom::fill(nonce.as_mut()).map_err(|_| NostrLifecycleError::EntropyUnavailable)?;
        if nonce.as_ref() == &[0; 32] {
            return Err(NostrLifecycleError::EntropyUnavailable);
        }
        Ok(nonce)
    }
}

pub async fn create_signing_identity(
    provider: &dyn CredentialsProvider,
    repository: &dyn IdentityLifecycleRepository,
    request: &CreateSigningIdentityRequest,
    cx: &AsyncApp,
) -> Result<AppliedIdentityLifecycle, NostrLifecycleError> {
    let store = CredentialsProviderSigningKeyStore::new(provider, cx);
    create_signing_identity_with(
        &store,
        repository,
        &OperatingSystemEntropy,
        request,
        current_time_millis()?,
    )
    .await
}

pub async fn rotate_signing_identity(
    provider: &dyn CredentialsProvider,
    repository: &dyn IdentityLifecycleRepository,
    request: &RotateSigningIdentityRequest,
    cx: &AsyncApp,
) -> Result<AppliedIdentityLifecycle, NostrLifecycleError> {
    let store = CredentialsProviderSigningKeyStore::new(provider, cx);
    rotate_signing_identity_with(
        &store,
        repository,
        &OperatingSystemEntropy,
        request,
        current_time_millis()?,
    )
    .await
}

pub async fn revoke_signing_identity(
    repository: &dyn IdentityLifecycleRepository,
    request: &TerminalBindingRequest,
) -> Result<AppliedIdentityLifecycle, NostrLifecycleError> {
    apply_terminal_transition(
        repository,
        request,
        BindingStatus::Revoked,
        current_time_millis()?,
    )
    .await
}

pub async fn archive_signing_identity(
    repository: &dyn IdentityLifecycleRepository,
    request: &TerminalBindingRequest,
) -> Result<AppliedIdentityLifecycle, NostrLifecycleError> {
    apply_terminal_transition(
        repository,
        request,
        BindingStatus::Archived,
        current_time_millis()?,
    )
    .await
}

pub async fn resolve_active_signing_credential(
    provider: &dyn CredentialsProvider,
    repository: &dyn IdentityLifecycleRepository,
    community_id: CommunityId,
    binding_id: BindingId,
    cx: &AsyncApp,
) -> Result<ActiveSigningCredential, NostrLifecycleError> {
    let store = CredentialsProviderSigningKeyStore::new(provider, cx);
    resolve_active_signing_credential_with(&store, repository, community_id, binding_id).await
}

async fn create_signing_identity_with(
    store: &dyn ProtectedSigningKeyStore,
    repository: &dyn IdentityLifecycleRepository,
    entropy: &dyn SigningKeyEntropy,
    request: &CreateSigningIdentityRequest,
    now_millis: u64,
) -> Result<AppliedIdentityLifecycle, NostrLifecycleError> {
    probe_protected_storage(store).await?;
    let secret = entropy.generate_secret()?;
    let nonce = entropy.generate_nonce()?;
    let public_key = public_key(&secret);
    let credential_identifier = nostr_credential_identifier(
        request.community_id.as_uuid(),
        request.service_account_id.get(),
        request.profile_id.as_uuid(),
        public_key,
    );
    ensure_destination_absent(store, &credential_identifier).await?;
    let challenge_digest = generation_challenge_digest(
        b"create",
        request.community_id,
        request.service_account_id,
        request.profile_id,
        public_key,
        &nonce,
        now_millis,
    );
    persist_new_signing_key(
        store,
        &credential_identifier,
        public_key,
        &secret,
        challenge_digest,
    )
    .await
    .map_err(|error| map_import_error(error, &credential_identifier))?;

    let mutation = match create_mutation(request, public_key, now_millis) {
        Ok(mutation) => mutation,
        Err(error) => return Err(cleanup_uncommitted(store, &credential_identifier, error).await),
    };
    commit_or_cleanup(store, repository, &credential_identifier, mutation).await
}

async fn rotate_signing_identity_with(
    store: &dyn ProtectedSigningKeyStore,
    repository: &dyn IdentityLifecycleRepository,
    entropy: &dyn SigningKeyEntropy,
    request: &RotateSigningIdentityRequest,
    now_millis: u64,
) -> Result<AppliedIdentityLifecycle, NostrLifecycleError> {
    validate_rotation_source(request)?;
    probe_protected_storage(store).await?;
    let secret = entropy.generate_secret()?;
    let nonce = entropy.generate_nonce()?;
    let public_key = public_key(&secret);
    let credential_identifier = nostr_credential_identifier(
        request.current_binding.community_id().as_uuid(),
        request.current_binding.service_account_id().get(),
        request.successor_profile_id.as_uuid(),
        public_key,
    );
    ensure_destination_absent(store, &credential_identifier).await?;
    let challenge_digest = generation_challenge_digest(
        b"rotate",
        request.current_binding.community_id(),
        request.current_binding.service_account_id(),
        request.successor_profile_id,
        public_key,
        &nonce,
        now_millis,
    );
    persist_new_signing_key(
        store,
        &credential_identifier,
        public_key,
        &secret,
        challenge_digest,
    )
    .await
    .map_err(|error| map_import_error(error, &credential_identifier))?;

    let mutation = match rotation_mutation(request, public_key, now_millis) {
        Ok(mutation) => mutation,
        Err(error) => return Err(cleanup_uncommitted(store, &credential_identifier, error).await),
    };
    commit_or_cleanup(store, repository, &credential_identifier, mutation).await
}

async fn commit_or_cleanup(
    store: &dyn ProtectedSigningKeyStore,
    repository: &dyn IdentityLifecycleRepository,
    credential_identifier: &str,
    mutation: IdentityLifecycleMutation,
) -> Result<AppliedIdentityLifecycle, NostrLifecycleError> {
    match repository.commit(mutation.clone()).await {
        Ok(()) => Ok(AppliedIdentityLifecycle { mutation }),
        Err(IdentityLifecycleRepositoryError::Rejected) => Err(cleanup_uncommitted(
            store,
            credential_identifier,
            NostrLifecycleError::RepositoryRejected,
        )
        .await),
        Err(IdentityLifecycleRepositoryError::OutcomeUnknown) => {
            Err(NostrLifecycleError::GeneratedKeyRepositoryOutcomeUnknown {
                credential_identifier: credential_identifier.to_owned(),
            })
        }
        Err(IdentityLifecycleRepositoryError::Unavailable) => Err(cleanup_uncommitted(
            store,
            credential_identifier,
            NostrLifecycleError::RepositoryUnavailable,
        )
        .await),
    }
}

async fn cleanup_uncommitted(
    store: &dyn ProtectedSigningKeyStore,
    credential_identifier: &str,
    original_error: NostrLifecycleError,
) -> NostrLifecycleError {
    match store.delete(credential_identifier).await {
        Ok(()) => original_error,
        Err(_) => NostrLifecycleError::CleanupFailed {
            credential_identifier: credential_identifier.to_owned(),
        },
    }
}

async fn probe_protected_storage(
    store: &dyn ProtectedSigningKeyStore,
) -> Result<(), NostrLifecycleError> {
    store
        .read(CREDENTIAL_PROBE_IDENTIFIER)
        .await
        .map(|_| ())
        .map_err(|_| NostrLifecycleError::ProtectedStorageUnavailable)
}

async fn ensure_destination_absent(
    store: &dyn ProtectedSigningKeyStore,
    credential_identifier: &str,
) -> Result<(), NostrLifecycleError> {
    match store.read(credential_identifier).await {
        Ok(None) => Ok(()),
        Ok(Some(_)) => Err(NostrLifecycleError::CredentialConflict),
        Err(_) => Err(NostrLifecycleError::ProtectedStorageUnavailable),
    }
}

fn create_mutation(
    request: &CreateSigningIdentityRequest,
    public_key: [u8; 32],
    now_millis: u64,
) -> Result<IdentityLifecycleMutation, NostrLifecycleError> {
    let public_key = NostrPublicKey::from_bytes(public_key);
    let verification = BindingVerification {
        method: BindingVerificationMethod::GeneratedKeyChallenge,
        evidence_reference: request.evidence_reference.clone(),
        verified_at_millis: now_millis,
    };
    let binding = AccountBinding::new(AccountBindingFields {
        binding_id: request.binding_id,
        community_id: request.community_id,
        service_account_id: request.service_account_id,
        profile_id: request.profile_id,
        public_key,
        status: BindingStatus::Active,
        verification: Some(verification),
        predecessor: None,
        successor: None,
        created_at_millis: now_millis,
        activated_at_millis: Some(now_millis),
        terminal_at_millis: None,
        organization_policy_version: request.organization_policy_version,
        actor_principal_id: request.actor_principal_id,
        version: AggregateVersion::FIRST,
        audit_reference: request.audit_reference,
    })
    .map_err(|_| NostrLifecycleError::InvalidTransition)?;
    let profile = IdentityProfile::new(ProfileRecordFields {
        profile_id: request.profile_id,
        community_id: request.community_id,
        author_public_key: public_key,
        kind: profile_kind(request.profile_kind),
        metadata: None,
        statuses: Vec::new(),
        social_lists: Vec::new(),
        relay_archive_states: Vec::new(),
        version: AggregateVersion::FIRST,
    })
    .map_err(|_| NostrLifecycleError::InvalidTransition)?;
    Ok(IdentityLifecycleMutation::Create { binding, profile })
}

fn rotation_mutation(
    request: &RotateSigningIdentityRequest,
    successor_public_key: [u8; 32],
    now_millis: u64,
) -> Result<IdentityLifecycleMutation, NostrLifecycleError> {
    let predecessor_version = request
        .current_binding
        .version()
        .next()
        .ok_or(NostrLifecycleError::VersionExhausted)?;
    let successor_reference = BindingVersionReference {
        binding_id: request.successor_binding_id,
        version: AggregateVersion::FIRST,
    };
    let mut predecessor_fields = request.current_binding.fields().clone();
    predecessor_fields.status = BindingStatus::Rotated;
    predecessor_fields.successor = Some(successor_reference);
    predecessor_fields.terminal_at_millis = Some(now_millis);
    predecessor_fields.organization_policy_version = request.organization_policy_version;
    predecessor_fields.actor_principal_id = request.actor_principal_id;
    predecessor_fields.version = predecessor_version;
    predecessor_fields.audit_reference = request.audit_reference;
    let predecessor = AccountBinding::new(predecessor_fields)
        .map_err(|_| NostrLifecycleError::InvalidTransition)?;

    let successor_public_key = NostrPublicKey::from_bytes(successor_public_key);
    let successor = AccountBinding::new(AccountBindingFields {
        binding_id: request.successor_binding_id,
        community_id: request.current_binding.community_id(),
        service_account_id: request.current_binding.service_account_id(),
        profile_id: request.successor_profile_id,
        public_key: successor_public_key,
        status: BindingStatus::Active,
        verification: Some(BindingVerification {
            method: BindingVerificationMethod::GeneratedKeyChallenge,
            evidence_reference: request.evidence_reference.clone(),
            verified_at_millis: now_millis,
        }),
        predecessor: Some(BindingVersionReference {
            binding_id: predecessor.binding_id(),
            version: predecessor.version(),
        }),
        successor: None,
        created_at_millis: now_millis,
        activated_at_millis: Some(now_millis),
        terminal_at_millis: None,
        organization_policy_version: request.organization_policy_version,
        actor_principal_id: request.actor_principal_id,
        version: AggregateVersion::FIRST,
        audit_reference: request.audit_reference,
    })
    .map_err(|_| NostrLifecycleError::InvalidTransition)?;
    let successor_kind = match request.current_profile.kind() {
        ProfileKind::Human => ProfileKind::Human,
        ProfileKind::Agent(_) => ProfileKind::Agent(AgentProfile {
            claimed_owner: None,
            owner_attestation: None,
        }),
    };
    let successor_profile = IdentityProfile::new(ProfileRecordFields {
        profile_id: request.successor_profile_id,
        community_id: request.current_profile.community_id(),
        author_public_key: successor_public_key,
        kind: successor_kind,
        metadata: None,
        statuses: Vec::new(),
        social_lists: Vec::new(),
        relay_archive_states: Vec::new(),
        version: AggregateVersion::FIRST,
    })
    .map_err(|_| NostrLifecycleError::InvalidTransition)?;

    Ok(IdentityLifecycleMutation::Rotate {
        predecessor,
        historical_profile: request.current_profile.clone(),
        successor,
        successor_profile,
    })
}

fn validate_rotation_source(
    request: &RotateSigningIdentityRequest,
) -> Result<(), NostrLifecycleError> {
    if !request.current_binding.can_sign() {
        return Err(NostrLifecycleError::InactiveBinding);
    }
    if request.current_binding.community_id() != request.current_profile.community_id()
        || request.current_binding.profile_id() != request.current_profile.profile_id()
        || request.current_binding.public_key() != request.current_profile.author_public_key()
        || request.successor_binding_id == request.current_binding.binding_id()
        || request.successor_profile_id == request.current_profile.profile_id()
    {
        return Err(NostrLifecycleError::BindingProfileMismatch);
    }
    Ok(())
}

async fn apply_terminal_transition(
    repository: &dyn IdentityLifecycleRepository,
    request: &TerminalBindingRequest,
    status: BindingStatus,
    now_millis: u64,
) -> Result<AppliedIdentityLifecycle, NostrLifecycleError> {
    if !request.current_binding.can_sign() {
        return Err(NostrLifecycleError::InactiveBinding);
    }
    let mut fields = request.current_binding.fields().clone();
    fields.status = status;
    fields.terminal_at_millis = Some(now_millis);
    fields.organization_policy_version = request.organization_policy_version;
    fields.actor_principal_id = request.actor_principal_id;
    fields.version = fields
        .version
        .next()
        .ok_or(NostrLifecycleError::VersionExhausted)?;
    fields.audit_reference = request.audit_reference;
    let binding =
        AccountBinding::new(fields).map_err(|_| NostrLifecycleError::InvalidTransition)?;
    let mutation = match status {
        BindingStatus::Revoked => IdentityLifecycleMutation::Revoke { binding },
        BindingStatus::Archived => IdentityLifecycleMutation::Archive { binding },
        _ => return Err(NostrLifecycleError::InvalidTransition),
    };
    repository
        .commit(mutation.clone())
        .await
        .map_err(map_repository_error)?;
    Ok(AppliedIdentityLifecycle { mutation })
}

async fn resolve_active_signing_credential_with(
    store: &dyn ProtectedSigningKeyStore,
    repository: &dyn IdentityLifecycleRepository,
    community_id: CommunityId,
    binding_id: BindingId,
) -> Result<ActiveSigningCredential, NostrLifecycleError> {
    let binding = repository
        .current_binding(community_id, binding_id)
        .await
        .map_err(|_| NostrLifecycleError::RepositoryUnavailable)?
        .ok_or(NostrLifecycleError::InactiveBinding)?;
    if !binding.can_sign() {
        return Err(NostrLifecycleError::InactiveBinding);
    }
    let credential_identifier = nostr_credential_identifier(
        binding.community_id().as_uuid(),
        binding.service_account_id().get(),
        binding.profile_id().as_uuid(),
        *binding.public_key().as_bytes(),
    );
    let challenge_digest = resolution_challenge_digest(&binding, &credential_identifier);
    verify_stored_signing_key(
        store,
        &credential_identifier,
        *binding.public_key().as_bytes(),
        challenge_digest,
    )
    .await
    .map_err(|error| map_import_error(error, &credential_identifier))?;
    Ok(ActiveSigningCredential {
        credential_identifier,
        public_key: binding.public_key(),
    })
}

fn profile_kind(kind: NewProfileKind) -> ProfileKind {
    match kind {
        NewProfileKind::Human => ProfileKind::Human,
        NewProfileKind::Agent => ProfileKind::Agent(AgentProfile {
            claimed_owner: None,
            owner_attestation: None,
        }),
    }
}

fn public_key(secret: &SecretKey) -> [u8; 32] {
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, secret);
    XOnlyPublicKey::from_keypair(&keypair).0.serialize()
}

fn generation_challenge_digest(
    operation: &[u8],
    community_id: CommunityId,
    service_account_id: ServiceAccountId,
    profile_id: ProfileId,
    public_key: [u8; 32],
    nonce: &[u8; 32],
    now_millis: u64,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(GENERATION_CHALLENGE_DOMAIN);
    digest.update((operation.len() as u64).to_be_bytes());
    digest.update(operation);
    digest.update(community_id.as_uuid().as_bytes());
    digest.update(service_account_id.get().to_be_bytes());
    digest.update(profile_id.as_uuid().as_bytes());
    digest.update(public_key);
    digest.update(nonce);
    digest.update(now_millis.to_be_bytes());
    digest.finalize().into()
}

fn resolution_challenge_digest(binding: &AccountBinding, credential_identifier: &str) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(RESOLUTION_CHALLENGE_DOMAIN);
    digest.update(binding.community_id().as_uuid().as_bytes());
    digest.update(binding.binding_id().as_uuid().as_bytes());
    digest.update(binding.version().get().to_be_bytes());
    digest.update(binding.public_key().as_bytes());
    digest.update((credential_identifier.len() as u64).to_be_bytes());
    digest.update(credential_identifier.as_bytes());
    digest.finalize().into()
}

fn current_time_millis() -> Result<u64, NostrLifecycleError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .ok_or(NostrLifecycleError::ClockUnavailable)
}

fn map_import_error(error: NostrImportError, credential_identifier: &str) -> NostrLifecycleError {
    match error {
        NostrImportError::ProtectedStorageUnavailable => {
            NostrLifecycleError::ProtectedStorageUnavailable
        }
        NostrImportError::CleanupFailed => NostrLifecycleError::CleanupFailed {
            credential_identifier: credential_identifier.to_owned(),
        },
        _ => NostrLifecycleError::ProtectedStorageVerificationFailed,
    }
}

fn map_repository_error(error: IdentityLifecycleRepositoryError) -> NostrLifecycleError {
    match error {
        IdentityLifecycleRepositoryError::Rejected => NostrLifecycleError::RepositoryRejected,
        IdentityLifecycleRepositoryError::OutcomeUnknown => {
            NostrLifecycleError::RepositoryOutcomeUnknown
        }
        IdentityLifecycleRepositoryError::Unavailable => NostrLifecycleError::RepositoryUnavailable,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::nostr_import::{ProtectedSigningKeyStoreError, StoredSigningKey};

    use super::*;

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

    #[derive(Default)]
    struct MemoryRepository {
        mutations: Mutex<Vec<IdentityLifecycleMutation>>,
        current_bindings: Mutex<HashMap<(CommunityId, BindingId), AccountBinding>>,
        reject: bool,
        outcome_unknown: bool,
    }

    impl IdentityLifecycleRepository for MemoryRepository {
        fn commit<'a>(
            &'a self,
            mutation: IdentityLifecycleMutation,
        ) -> Pin<Box<dyn Future<Output = Result<(), IdentityLifecycleRepositoryError>> + 'a>>
        {
            Box::pin(async move {
                if self.reject {
                    return Err(IdentityLifecycleRepositoryError::Rejected);
                }
                if self.outcome_unknown {
                    return Err(IdentityLifecycleRepositoryError::OutcomeUnknown);
                }
                {
                    let mut current_bindings =
                        self.current_bindings.lock().expect("repository lock");
                    match &mutation {
                        IdentityLifecycleMutation::Create { binding, .. } => {
                            current_bindings.insert(
                                (binding.community_id(), binding.binding_id()),
                                binding.clone(),
                            );
                        }
                        IdentityLifecycleMutation::Rotate {
                            predecessor,
                            successor,
                            ..
                        } => {
                            current_bindings.insert(
                                (predecessor.community_id(), predecessor.binding_id()),
                                predecessor.clone(),
                            );
                            current_bindings.insert(
                                (successor.community_id(), successor.binding_id()),
                                successor.clone(),
                            );
                        }
                        IdentityLifecycleMutation::Revoke { binding }
                        | IdentityLifecycleMutation::Archive { binding } => {
                            current_bindings.insert(
                                (binding.community_id(), binding.binding_id()),
                                binding.clone(),
                            );
                        }
                    }
                }
                self.mutations
                    .lock()
                    .expect("repository lock")
                    .push(mutation);
                Ok(())
            })
        }

        fn current_binding<'a>(
            &'a self,
            community_id: CommunityId,
            binding_id: BindingId,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<Option<AccountBinding>, IdentityLifecycleRepositoryError>,
                    > + 'a,
            >,
        > {
            Box::pin(async move {
                Ok(self
                    .current_bindings
                    .lock()
                    .expect("repository lock")
                    .get(&(community_id, binding_id))
                    .cloned())
            })
        }
    }

    struct FixedEntropy {
        secret: [u8; 32],
        calls: AtomicUsize,
    }

    impl FixedEntropy {
        fn new(secret: [u8; 32]) -> Self {
            Self {
                secret,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl SigningKeyEntropy for FixedEntropy {
        fn generate_secret(&self) -> Result<SecretKey, NostrLifecycleError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            SecretKey::from_slice(&self.secret).map_err(|_| NostrLifecycleError::EntropyUnavailable)
        }

        fn generate_nonce(&self) -> Result<Zeroizing<[u8; 32]>, NostrLifecycleError> {
            Ok(Zeroizing::new([7; 32]))
        }
    }

    fn evidence() -> EvidenceReference {
        EvidenceReference::new("audit:generated-key-challenge").expect("bounded evidence reference")
    }

    fn create_request() -> CreateSigningIdentityRequest {
        CreateSigningIdentityRequest {
            binding_id: BindingId::from_uuid(uuid::Uuid::from_u128(10)),
            community_id: CommunityId::from_uuid(uuid::Uuid::from_u128(1)),
            service_account_id: ServiceAccountId::new(7),
            profile_id: ProfileId::from_uuid(uuid::Uuid::from_u128(2)),
            profile_kind: NewProfileKind::Human,
            organization_policy_version: OrganizationPolicyVersion::FIRST,
            actor_principal_id: PrincipalId::from_uuid(uuid::Uuid::from_u128(3)),
            audit_reference: OperationId::from_uuid(uuid::Uuid::from_u128(4)),
            evidence_reference: evidence(),
        }
    }

    fn created_records(mutation: IdentityLifecycleMutation) -> (AccountBinding, IdentityProfile) {
        match mutation {
            IdentityLifecycleMutation::Create { binding, profile } => (binding, profile),
            _ => panic!("expected create mutation"),
        }
    }

    #[test]
    fn generation_persists_verified_key_and_identity_records() {
        let store = MemoryStore::default();
        let repository = MemoryRepository::default();
        let entropy = FixedEntropy::new([1; 32]);
        let applied = futures::executor::block_on(create_signing_identity_with(
            &store,
            &repository,
            &entropy,
            &create_request(),
            100,
        ))
        .expect("generation succeeds");
        let (binding, profile) = created_records(applied.mutation);

        assert!(binding.can_sign());
        assert_eq!(profile.author_public_key(), binding.public_key());
        let identifier = nostr_credential_identifier(
            binding.community_id().as_uuid(),
            binding.service_account_id().get(),
            binding.profile_id().as_uuid(),
            *binding.public_key().as_bytes(),
        );
        assert!(store.contains(&identifier));
    }

    #[test]
    fn unavailable_storage_fails_before_key_generation() {
        let store = MemoryStore {
            unavailable: true,
            ..MemoryStore::default()
        };
        let repository = MemoryRepository::default();
        let entropy = FixedEntropy::new([1; 32]);

        let error = futures::executor::block_on(create_signing_identity_with(
            &store,
            &repository,
            &entropy,
            &create_request(),
            100,
        ))
        .expect_err("unavailable storage rejected");

        assert_eq!(error, NostrLifecycleError::ProtectedStorageUnavailable);
        assert_eq!(entropy.calls.load(Ordering::SeqCst), 0);
        assert!(
            repository
                .mutations
                .lock()
                .expect("repository lock")
                .is_empty()
        );
    }

    #[test]
    fn repository_rejection_removes_uncommitted_generated_key() {
        let store = MemoryStore::default();
        let repository = MemoryRepository {
            reject: true,
            ..MemoryRepository::default()
        };
        let entropy = FixedEntropy::new([1; 32]);
        let request = create_request();

        let error = futures::executor::block_on(create_signing_identity_with(
            &store,
            &repository,
            &entropy,
            &request,
            100,
        ))
        .expect_err("repository rejection surfaced");

        assert_eq!(error, NostrLifecycleError::RepositoryRejected);
        assert!(store.entries.lock().expect("store lock").is_empty());
    }

    #[test]
    fn unknown_repository_outcome_retains_key_for_reconciliation() {
        let store = MemoryStore::default();
        let repository = MemoryRepository {
            outcome_unknown: true,
            ..MemoryRepository::default()
        };
        let entropy = FixedEntropy::new([1; 32]);

        let error = futures::executor::block_on(create_signing_identity_with(
            &store,
            &repository,
            &entropy,
            &create_request(),
            100,
        ))
        .expect_err("unknown repository outcome surfaced");

        let NostrLifecycleError::GeneratedKeyRepositoryOutcomeUnknown {
            credential_identifier,
        } = error
        else {
            panic!("expected generated-key unknown outcome");
        };
        assert!(store.contains(&credential_identifier));
    }

    #[test]
    fn rotation_preserves_old_authorship_and_moves_active_signing() {
        let store = MemoryStore::default();
        let repository = MemoryRepository::default();
        let original_entropy = FixedEntropy::new([1; 32]);
        let created = futures::executor::block_on(create_signing_identity_with(
            &store,
            &repository,
            &original_entropy,
            &create_request(),
            100,
        ))
        .expect("initial generation succeeds");
        let (current_binding, current_profile) = created_records(created.mutation);
        let old_author = current_profile.author_public_key();
        let request = RotateSigningIdentityRequest {
            current_binding,
            current_profile,
            successor_binding_id: BindingId::from_uuid(uuid::Uuid::from_u128(11)),
            successor_profile_id: ProfileId::from_uuid(uuid::Uuid::from_u128(12)),
            organization_policy_version: OrganizationPolicyVersion::FIRST,
            actor_principal_id: PrincipalId::from_uuid(uuid::Uuid::from_u128(3)),
            audit_reference: OperationId::from_uuid(uuid::Uuid::from_u128(5)),
            evidence_reference: evidence(),
        };
        let successor_entropy = FixedEntropy::new([2; 32]);

        let rotated = futures::executor::block_on(rotate_signing_identity_with(
            &store,
            &repository,
            &successor_entropy,
            &request,
            200,
        ))
        .expect("rotation succeeds");
        let IdentityLifecycleMutation::Rotate {
            predecessor,
            historical_profile,
            successor,
            successor_profile,
        } = rotated.mutation
        else {
            panic!("expected rotation mutation");
        };

        assert_eq!(historical_profile.author_public_key(), old_author);
        assert_eq!(predecessor.public_key(), old_author);
        assert_eq!(predecessor.status(), BindingStatus::Rotated);
        assert!(!predecessor.can_sign());
        assert!(successor.can_sign());
        assert_eq!(
            successor_profile.author_public_key(),
            successor.public_key()
        );
        assert_ne!(successor.public_key(), old_author);
        assert_eq!(
            futures::executor::block_on(resolve_active_signing_credential_with(
                &store,
                &repository,
                predecessor.community_id(),
                predecessor.binding_id(),
            )),
            Err(NostrLifecycleError::InactiveBinding)
        );
        assert!(
            futures::executor::block_on(resolve_active_signing_credential_with(
                &store,
                &repository,
                successor.community_id(),
                successor.binding_id(),
            ))
            .is_ok()
        );
    }

    #[test]
    fn revoke_and_archive_disable_signing_without_rewriting_author() {
        let store = MemoryStore::default();
        let creation_repository = MemoryRepository::default();
        let entropy = FixedEntropy::new([1; 32]);
        let created = futures::executor::block_on(create_signing_identity_with(
            &store,
            &creation_repository,
            &entropy,
            &create_request(),
            100,
        ))
        .expect("initial generation succeeds");
        let (binding, profile) = created_records(created.mutation);
        let author = profile.author_public_key();

        for (status, audit) in [
            (BindingStatus::Revoked, 20_u128),
            (BindingStatus::Archived, 21_u128),
        ] {
            let repository = MemoryRepository::default();
            repository
                .current_bindings
                .lock()
                .expect("repository lock")
                .insert(
                    (binding.community_id(), binding.binding_id()),
                    binding.clone(),
                );
            let request = TerminalBindingRequest {
                current_binding: binding.clone(),
                organization_policy_version: OrganizationPolicyVersion::FIRST,
                actor_principal_id: PrincipalId::from_uuid(uuid::Uuid::from_u128(3)),
                audit_reference: OperationId::from_uuid(uuid::Uuid::from_u128(audit)),
            };
            let applied = futures::executor::block_on(apply_terminal_transition(
                &repository,
                &request,
                status,
                300,
            ))
            .expect("terminal transition succeeds");
            let terminal = match applied.mutation {
                IdentityLifecycleMutation::Revoke { binding }
                | IdentityLifecycleMutation::Archive { binding } => binding,
                _ => panic!("expected terminal mutation"),
            };
            assert_eq!(terminal.public_key(), author);
            assert!(!terminal.can_sign());
            assert_eq!(
                futures::executor::block_on(resolve_active_signing_credential_with(
                    &store,
                    &repository,
                    terminal.community_id(),
                    terminal.binding_id(),
                )),
                Err(NostrLifecycleError::InactiveBinding)
            );
        }
        assert_eq!(profile.author_public_key(), author);
    }
}
