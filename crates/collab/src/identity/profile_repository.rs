use async_trait::async_trait;
use collaboration_domain::{
    AgentProfile, AggregateVersion, AuthenticatedPrincipal, AuthoredValue, CommunityId,
    IdentityProfile, NostrAuthenticationMethod, NostrEventId, NostrPublicKey,
    OwnerAttestationEvidence, PrincipalId, PrincipalScopes, ProfileId, ProfileKind,
    ProfileMetadata, ProfileRecordFields, TenantContext, validate_profile_update,
};
use nostr_compat::{
    CanonicalEvent, EventId, PublicKey, TimestampPolicy, buzz_nips::identity::OwnerAttestation,
    generated_kinds::KIND_PROFILE, head::ReplacementCoordinate, verify_signed_event,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbErr, QueryResult,
    Statement, TransactionTrait,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::str::FromStr as _;
use uuid::Uuid;

use crate::{
    db::collaboration::{
        event_repository::{
            EventRepository, EventRepositoryError, EventStoreOutcome, VerifiedEventRecord,
        },
        persistence_policy::{EventPersistencePolicy, PrivacyAdmission},
    },
    nostr::{
        auth::{NostrPrincipalResolutionError, NostrPrincipalResolver, VerifiedNip42Identity},
        http::{Nip98PrincipalResolver, NostrHttpDirectoryError, VerifiedNip98Identity},
    },
};

const SELECT_PROFILE: &str = r#"
SELECT community_id, profile_id, principal_id, author_public_key, profile_kind,
       owner_public_key, first_owner_proof_event_id, source_event_id,
       aggregate_version::text AS aggregate_version, profile_payload, profile_hash,
       first_owner_auth_proof, latest_owner_auth_proof, registration_source_event_id,
       registration_verified_at_seconds::text AS registration_verified_at_seconds,
       projection_hash_version, observation_proof, observation_source_event_id,
       observation_verified_at_millis::text AS observation_verified_at_millis, observation_moderation_operation_id,
       accepted_nip05_name, accepted_nip05_source_event_id, nip05_processed_event_id, nip05_projection_host
FROM public.collaboration_identity_profiles
WHERE community_id = $1 AND author_public_key = $2
"#;

#[derive(Debug, thiserror::Error)]
pub enum IdentityProfileRepositoryError {
    #[error("identity-profile repository requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("identity-profile request crossed its community boundary")]
    TenantBoundaryViolation,
    #[error("identity-profile source or projection is invalid")]
    InvalidRecord,
    #[error("identity-profile classification or first owner conflicts with current state")]
    IdentityConflict,
    #[error("identity-profile aggregate version is exhausted")]
    VersionExhausted,
    #[error("identity-profile source repository is unavailable or invalid")]
    EventRepository(#[from] EventRepositoryError),
    #[error("identity-profile repository is unavailable")]
    Unavailable(#[from] DbErr),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedIdentityProfile {
    principal_id: PrincipalId,
    profile: IdentityProfile,
    first_owner_proof_event_id: NostrEventId,
    source_event_id: NostrEventId,
}

impl PersistedIdentityProfile {
    pub const fn principal_id(&self) -> PrincipalId {
        self.principal_id
    }
    pub const fn profile(&self) -> &IdentityProfile {
        &self.profile
    }
    pub const fn first_owner_proof_event_id(&self) -> NostrEventId {
        self.first_owner_proof_event_id
    }
    pub const fn source_event_id(&self) -> NostrEventId {
        self.source_event_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct DetachedOwnerProof {
    source_event_id: NostrEventId,
    source_created_at: u64,
    verified_at_seconds: u64,
    exact_tag: Vec<String>,
}

impl DetachedOwnerProof {
    fn from_receipt(identity: &VerifiedNip42Identity) -> Option<Self> {
        identity.owner_attestation().map(|proof| Self {
            source_event_id: NostrEventId::from_bytes(*identity.event_id().as_bytes()),
            source_created_at: identity.signed_authentication_event().event.created_at,
            verified_at_seconds: identity.verified_at_seconds(),
            exact_tag: proof.attestation().to_tag(),
        })
    }

    fn verify(
        &self,
        public_key: NostrPublicKey,
    ) -> Result<OwnerAttestation, IdentityProfileRepositoryError> {
        let proof = OwnerAttestation::parse_tag(&self.exact_tag)
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
        proof
            .verify_for_membership(
                &PublicKey::from_bytes(*public_key.as_bytes()),
                self.source_created_at,
            )
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
        Ok(proof)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source_kind", deny_unknown_fields)]
enum ObservationEvidence {
    RetainedEventAuthor {
        event_id: NostrEventId,
        verified_at_millis: u64,
    },
    VerifiedNip42Key {
        event_id: NostrEventId,
        verified_at_seconds: u64,
    },
    AdmittedModerationTarget {
        event: CanonicalEvent,
        event_id: NostrEventId,
        signature: nostr_compat::EventSignature,
        operation_id: collaboration_domain::OperationId,
        actor_principal_id: PrincipalId,
        verified_at_millis: u64,
    },
    VerifiedNip42PresentedOwner {
        authentication_event_id: NostrEventId,
        source_created_at_seconds: u64,
        verified_at_seconds: u64,
        agent_public_key: NostrPublicKey,
        exact_attestation_tag: Vec<String>,
    },
}

impl ObservationEvidence {
    fn retained_event_id(&self) -> Option<NostrEventId> {
        match self {
            Self::RetainedEventAuthor { event_id, .. } => Some(*event_id),
            _ => None,
        }
    }

    fn moderation_operation_id(&self) -> Option<collaboration_domain::OperationId> {
        match self {
            Self::AdmittedModerationTarget { operation_id, .. } => Some(*operation_id),
            _ => None,
        }
    }

    fn verified_at_millis(&self) -> Result<u64, IdentityProfileRepositoryError> {
        match self {
            Self::RetainedEventAuthor {
                verified_at_millis, ..
            }
            | Self::AdmittedModerationTarget {
                verified_at_millis, ..
            } => Ok(*verified_at_millis),
            Self::VerifiedNip42Key {
                verified_at_seconds,
                ..
            }
            | Self::VerifiedNip42PresentedOwner {
                verified_at_seconds,
                ..
            } => verified_at_seconds
                .checked_mul(1000)
                .ok_or(IdentityProfileRepositoryError::InvalidRecord),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentitySubjectStatus {
    ObservedOnly,
    RegisteredUnclassified,
    ClassifiedProfile,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedIdentitySubject {
    record: RegisteredIdentity,
}

impl ObservedIdentitySubject {
    pub const fn principal_id(&self) -> PrincipalId {
        self.record.principal_id
    }
    pub const fn profile_id(&self) -> ProfileId {
        self.record.profile_id
    }
    pub const fn public_key(&self) -> NostrPublicKey {
        self.record.public_key
    }
    pub const fn version(&self) -> AggregateVersion {
        self.record.version
    }
    pub fn status(&self) -> IdentitySubjectStatus {
        if self.record.profile.is_some() {
            IdentitySubjectStatus::ClassifiedProfile
        } else if self.record.registration_source_event_id.is_some() {
            IdentitySubjectStatus::RegisteredUnclassified
        } else {
            IdentitySubjectStatus::ObservedOnly
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source_kind", deny_unknown_fields)]
enum ImmutableAgentOwnerProof {
    RetainedProfile { event_id: NostrEventId },
    VerifiedAuthentication { proof: DetachedOwnerProof },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImmutableAgentMessageAttribution {
    community_id: CommunityId,
    profile_id: ProfileId,
    profile_version: AggregateVersion,
    agent_principal_id: PrincipalId,
    agent_public_key: NostrPublicKey,
    owner_principal_id: PrincipalId,
    owner_public_key: NostrPublicKey,
    proof_event_id: NostrEventId,
    proof: ImmutableAgentOwnerProof,
    owner_membership_version: AggregateVersion,
    channel_membership_version: AggregateVersion,
    channel_role: String,
}

pub(crate) struct VerifiedAgentMessageAuthor {
    record: ImmutableAgentMessageAttribution,
}

impl VerifiedAgentMessageAuthor {
    pub(crate) const fn owner_principal_id(&self) -> PrincipalId {
        self.record.owner_principal_id
    }
    pub(crate) const fn profile_id(&self) -> ProfileId {
        self.record.profile_id
    }
    pub(crate) const fn proof_event_id(&self) -> NostrEventId {
        self.record.proof_event_id
    }
    pub(crate) fn author(&self) -> collaboration_domain::MessageAuthor {
        collaboration_domain::MessageAuthor::OwnerAttestedAgent {
            agent_principal_id: self.record.agent_principal_id,
            owner_principal_id: self.record.owner_principal_id,
            proof_event_id: self.record.proof_event_id,
        }
    }
    pub(crate) fn attribution(
        &self,
        owner: collaboration_domain::CommunityMembership,
        channel: collaboration_domain::ChannelMembership,
    ) -> Result<serde_json::Value, IdentityProfileRepositoryError> {
        if owner.community_id != self.record.community_id
            || owner.principal_id != self.record.owner_principal_id
            || channel.community_id != self.record.community_id
            || channel.principal_id != self.record.agent_principal_id
        {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        let mut record = self.record.clone();
        record.owner_membership_version = owner.version;
        record.channel_membership_version = channel.version;
        record.channel_role = format!("{:?}", channel.role);
        serde_json::to_value(record).map_err(|_| IdentityProfileRepositoryError::InvalidRecord)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct AcceptedNameProjection {
    name: Option<String>,
    source_event_id: Option<NostrEventId>,
    processed_event_id: NostrEventId,
    host: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RegisteredIdentity {
    community_id: CommunityId,
    profile_id: ProfileId,
    principal_id: PrincipalId,
    public_key: NostrPublicKey,
    version: AggregateVersion,
    profile: Option<IdentityProfile>,
    first_owner_proof_event_id: Option<NostrEventId>,
    source_event_id: Option<NostrEventId>,
    first_owner_auth_proof: Option<DetachedOwnerProof>,
    latest_owner_auth_proof: Option<DetachedOwnerProof>,
    registration_source_event_id: Option<NostrEventId>,
    registration_verified_at_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observation: Option<ObservationEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    accepted_name: Option<AcceptedNameProjection>,
}

impl RegisteredIdentity {
    pub const fn principal_id(&self) -> PrincipalId {
        self.principal_id
    }
    pub const fn profile_id(&self) -> ProfileId {
        self.profile_id
    }
    pub const fn public_key(&self) -> NostrPublicKey {
        self.public_key
    }
    pub const fn profile(&self) -> Option<&IdentityProfile> {
        self.profile.as_ref()
    }
    pub const fn version(&self) -> AggregateVersion {
        self.version
    }

    fn persisted_profile(
        &self,
    ) -> Result<Option<PersistedIdentityProfile>, IdentityProfileRepositoryError> {
        let Some(profile) = &self.profile else {
            return Ok(None);
        };
        let first_owner_proof_event_id = self
            .first_owner_proof_event_id
            .or_else(|| {
                self.first_owner_auth_proof
                    .as_ref()
                    .map(|proof| proof.source_event_id)
            })
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        let source_event_id = self
            .source_event_id
            .or_else(|| match profile.kind() {
                ProfileKind::Agent(agent) => agent
                    .owner_attestation
                    .as_ref()
                    .map(|proof| proof.proof_event_id),
                ProfileKind::Human => None,
            })
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        Ok(Some(PersistedIdentityProfile {
            principal_id: self.principal_id,
            profile: profile.clone(),
            first_owner_proof_event_id,
            source_event_id,
        }))
    }
}

pub(crate) struct PreparedModerationSubject {
    record: RegisteredIdentity,
    new_observation: bool,
}

impl PreparedModerationSubject {
    pub(crate) const fn principal_id(&self) -> PrincipalId {
        self.record.principal_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthoredProfileMetadata {
    community_id: CommunityId,
    metadata: AuthoredValue<ProfileMetadata>,
    subject: Option<ObservedIdentitySubject>,
    owner_public_key: Option<NostrPublicKey>,
}

impl AuthoredProfileMetadata {
    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn metadata(&self) -> &AuthoredValue<ProfileMetadata> {
        &self.metadata
    }

    pub const fn subject(&self) -> Option<&ObservedIdentitySubject> {
        self.subject.as_ref()
    }

    pub const fn owner_public_key(&self) -> Option<NostrPublicKey> {
        self.owner_public_key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IdentityProfileProjectionOutcome {
    Unclassified,
    Materialized(Box<PersistedIdentityProfile>),
    Unchanged(Box<PersistedIdentityProfile>),
    Stale,
}

pub struct IdentityProfileRepository {
    connection: DatabaseConnection,
    events: EventRepository,
}

enum ProfileOwnerEvidencePolicy {
    StrictProjection,
    OptionalAuthoredMetadata,
}

impl IdentityProfileRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, IdentityProfileRepositoryError> {
        if !matches!(
            &connection,
            DatabaseConnection::SqlxPostgresPoolConnection(_)
        ) {
            return Err(IdentityProfileRepositoryError::UnsupportedDatabase);
        }
        let events = EventRepository::new(DatabaseConnection::from(
            connection.get_postgres_connection_pool().clone(),
        ))?;
        Ok(Self { connection, events })
    }

    pub async fn current_public_community_metadata(
        &self,
        tenant: &TenantContext,
    ) -> Result<
        Option<crate::messages::channel_admission::CurrentCommunityServingState>,
        IdentityProfileRepositoryError,
    > {
        let transaction = self
            .connection
            .begin_with_config(Some(sea_orm::IsolationLevel::ReadCommitted), None)
            .await?;
        let result = self
            .current_public_community_metadata_in_transaction(&transaction, tenant)
            .await;
        finish_transaction(transaction, result).await
    }

    pub async fn current_public_community_metadata_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
    ) -> Result<
        Option<crate::messages::channel_admission::CurrentCommunityServingState>,
        IdentityProfileRepositoryError,
    > {
        ensure_transaction_tenant(transaction, tenant).await?;
        let community =
            crate::messages::channel_admission::current_community_serving_state_in_transaction(
                transaction,
                tenant,
            )
            .await
            .map_err(|error| match error.downcast::<DbErr>() {
                Ok(error) => IdentityProfileRepositoryError::Unavailable(error),
                Err(_) => IdentityProfileRepositoryError::InvalidRecord,
            })?;
        let Some(community) = community else {
            return Ok(None);
        };
        let authority = axum::http::uri::Authority::from_str(tenant.route_reference())
            .map_err(|_| IdentityProfileRepositoryError::TenantBoundaryViolation)?;
        if community.lifecycle != collaboration_domain::CommunityLifecycleState::Active
            || authority.host().to_ascii_lowercase() != community.host.as_str()
        {
            return Ok(None);
        }
        Ok(Some(community))
    }

    pub async fn current_accepted_nip05(
        &self,
        tenant: &TenantContext,
        canonical_name: &str,
    ) -> Result<Option<PublicKey>, IdentityProfileRepositoryError> {
        let transaction = self
            .connection
            .begin_with_config(Some(sea_orm::IsolationLevel::ReadCommitted), None)
            .await?;
        let result = self
            .current_accepted_nip05_in_transaction(&transaction, tenant, canonical_name)
            .await;
        finish_transaction(transaction, result).await
    }

    pub async fn current_accepted_nip05_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        canonical_name: &str,
    ) -> Result<Option<PublicKey>, IdentityProfileRepositoryError> {
        if crate::nostr::http::canonical_nip05_name(canonical_name, None).as_deref()
            != Some(canonical_name)
        {
            return Ok(None);
        }
        let Some(community) = self
            .current_public_community_metadata_in_transaction(transaction, tenant)
            .await?
        else {
            return Ok(None);
        };
        let handle = format!("{canonical_name}@{}", community.host.as_str());
        let row = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT author_public_key FROM public.collaboration_identity_profiles WHERE community_id=$1 AND lower(accepted_nip05_name)=lower($2)",
            [tenant.community_id().as_uuid().into(), handle.clone().into()])).await?;
        let Some(row) = row else { return Ok(None) };
        let key = NostrPublicKey::from_bytes(
            row.try_get::<Vec<u8>>("", "author_public_key")?
                .try_into()
                .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
        );
        // Do not hold a row lock before the canonical identity lock: profile writers take those locks in the opposite order.
        lock_identity(transaction, tenant, key).await?;
        let Some(record) = select_profile(transaction, tenant, key, false).await? else {
            return Ok(None);
        };
        let Some(accepted) = &record.accepted_name else {
            return Ok(None);
        };
        if accepted.name.as_deref() != Some(handle.as_str()) {
            return Ok(None);
        }
        self.verify_retained_proof(transaction, tenant, &record)
            .await?;
        let head = self
            .events
            .head_in_transaction(
                transaction,
                tenant,
                &ReplacementCoordinate {
                    kind: 0,
                    author: PublicKey::from_bytes(*key.as_bytes()),
                    discriminator: None,
                },
            )
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        if head.signed_event().claimed_id.as_bytes() != accepted.processed_event_id.as_bytes() {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        Ok(Some(PublicKey::from_bytes(*key.as_bytes())))
    }

    async fn project_admitted_nip05_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &VerifiedEventRecord,
    ) -> Result<(), IdentityProfileRepositoryError> {
        ensure_transaction_tenant(transaction, tenant).await?;
        let public_key =
            NostrPublicKey::from_bytes(*source.signed_event().event.public_key.as_bytes());
        lock_identity(transaction, tenant, public_key).await?;
        let mut record = select_profile(transaction, tenant, public_key, true)
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        self.verify_retained_proof(transaction, tenant, &record)
            .await?;
        let source_id = NostrEventId::from_bytes(*source.signed_event().claimed_id.as_bytes());
        let head = self
            .events
            .head_in_transaction(
                transaction,
                tenant,
                &ReplacementCoordinate {
                    kind: 0,
                    author: PublicKey::from_bytes(*public_key.as_bytes()),
                    discriminator: None,
                },
            )
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        if head.signed_event().claimed_id.as_bytes() != source_id.as_bytes()
            || record
                .accepted_name
                .as_ref()
                .is_some_and(|accepted| accepted.processed_event_id == source_id)
        {
            return Ok(());
        }
        let host = axum::http::uri::Authority::from_str(tenant.route_reference())
            .map_err(|_| IdentityProfileRepositoryError::TenantBoundaryViolation)?
            .host()
            .to_ascii_lowercase();
        let claimed = profile_metadata(&source.signed_event().event)?
            .nip05_handle
            .as_deref()
            .and_then(|claim| crate::nostr::http::canonical_nip05_name(claim, Some(host.as_str())));
        let exceeds_storage_bound = claimed.as_ref().is_some_and(|claim| {
            claim.chars().count() > crate::nostr::http::MAX_NIP05_HANDLE_CHARACTERS
        });
        if exceeds_storage_bound {
            log::warn!(
                "NIP05 claim exceeds accepted-name storage limit; retaining prior projection"
            );
        }
        let claimed = claimed.filter(|_| !exceeds_storage_bound);
        let previous = record
            .accepted_name
            .as_ref()
            .and_then(|accepted| accepted.name.clone());
        let mut locked_names = previous
            .iter()
            .chain(claimed.iter())
            .cloned()
            .collect::<Vec<_>>();
        locked_names.sort();
        locked_names.dedup();
        for name in locked_names {
            transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
                    [format!(
                        "zed.collaboration.identity.accepted-nip05:{}:{name}",
                        tenant.community_id()
                    )
                    .into()],
                ))
                .await?;
        }
        let occupied = match claimed.as_ref() {
            Some(name) => transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,
                "SELECT author_public_key FROM public.collaboration_identity_profiles WHERE community_id=$1 AND lower(accepted_nip05_name)=lower($2) AND author_public_key<>$3",
                [tenant.community_id().as_uuid().into(), name.clone().into(), public_key.as_bytes().to_vec().into()])).await?.is_some(),
            None => false,
        };
        let (name, source_event_id) = if occupied || exceeds_storage_bound {
            (
                previous,
                record
                    .accepted_name
                    .as_ref()
                    .and_then(|accepted| accepted.source_event_id),
            )
        } else {
            (claimed.clone(), claimed.map(|_| source_id))
        };
        let projection_host = match name.as_ref() {
            Some(name) => name
                .split_once('@')
                .map(|(_, domain)| domain.to_owned())
                .ok_or(IdentityProfileRepositoryError::InvalidRecord)?,
            None => host,
        };
        record.accepted_name = Some(AcceptedNameProjection {
            name,
            source_event_id,
            processed_event_id: source_id,
            host: projection_host,
        });
        record.version = record
            .version
            .next()
            .ok_or(IdentityProfileRepositoryError::VersionExhausted)?;
        if let Some(profile) = &record.profile {
            let mut fields = profile.fields().clone();
            fields.version = record.version;
            let updated = IdentityProfile::new(fields)
                .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
            validate_profile_update(profile, &updated)
                .map_err(|_| IdentityProfileRepositoryError::IdentityConflict)?;
            record.profile = Some(updated);
        }
        write_profile(transaction, tenant, &record).await
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn current_authored_metadata_by_public_key(
        &self,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
    ) -> Result<Option<AuthoredProfileMetadata>, IdentityProfileRepositoryError> {
        let transaction = self.connection.begin().await?;
        let result = self
            .current_authored_metadata_in_transaction(&transaction, tenant, public_key)
            .await;
        finish_transaction(transaction, result).await
    }

    pub async fn current_authored_metadata_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
    ) -> Result<Option<AuthoredProfileMetadata>, IdentityProfileRepositoryError> {
        ensure_transaction_tenant(transaction, tenant).await?;
        lock_identity(transaction, tenant, public_key).await?;
        let subject = self
            .current_subject_in_transaction(transaction, tenant, public_key)
            .await?;
        let head = self
            .events
            .head_in_transaction(
                transaction,
                tenant,
                &ReplacementCoordinate {
                    kind: u16::try_from(KIND_PROFILE)
                        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
                    author: PublicKey::from_bytes(*public_key.as_bytes()),
                    discriminator: None,
                },
            )
            .await?;
        let Some(head) = head else {
            return Ok(None);
        };
        let signed = head.signed_event();
        verify_signed_event(signed, TimestampPolicy::Historical)
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
        if head.community_id() != tenant.community_id()
            || signed.event.public_key.as_bytes() != public_key.as_bytes()
            || u32::from(signed.event.kind) != KIND_PROFILE
        {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        let metadata = profile_metadata(&signed.event)?;
        metadata
            .validate()
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
        let owner_public_key = subject.as_ref().and_then(|subject| {
            subject
                .record
                .profile
                .as_ref()
                .and_then(|profile| match profile.kind() {
                    ProfileKind::Agent(agent) => agent.claimed_owner,
                    ProfileKind::Human => None,
                })
        });
        Ok(Some(AuthoredProfileMetadata {
            community_id: tenant.community_id(),
            metadata: AuthoredValue {
                source_event_id: NostrEventId::from_bytes(*signed.claimed_id.as_bytes()),
                source_author: public_key,
                source_created_at: signed.event.created_at,
                value: metadata,
            },
            subject,
            owner_public_key,
        }))
    }

    pub(crate) async fn verified_message_agent_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        now_seconds: u64,
    ) -> Result<Option<VerifiedAgentMessageAuthor>, IdentityProfileRepositoryError> {
        use collaboration_domain::AuthenticatedPrincipalKind;
        ensure_transaction_tenant(transaction, tenant).await?;
        if let AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } = principal.kind() {
            let registered = self
                .current_registration_in_transaction(transaction, tenant, *public_key)
                .await?
                .ok_or(IdentityProfileRepositoryError::IdentityConflict)?;
            if registered.principal_id != principal.principal_id() {
                return Err(IdentityProfileRepositoryError::IdentityConflict);
            }
            return Ok(None);
        }
        let AuthenticatedPrincipalKind::OwnerAttestedAgent {
            profile_id,
            agent_public_key,
            owner_public_key,
            proof_event_id,
            ..
        } = principal.kind()
        else {
            return Ok(None);
        };
        if principal.community_id() != tenant.community_id() {
            return Err(IdentityProfileRepositoryError::TenantBoundaryViolation);
        }
        ensure_transaction_tenant(transaction, tenant).await?;
        // Match the profile writer's lock before reading either association; renewal cannot race admission.
        for key in std::collections::BTreeSet::from([*agent_public_key, *owner_public_key]) {
            lock_identity(transaction, tenant, key).await?;
        }
        let record = select_profile(transaction, tenant, *agent_public_key, true)
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        self.verify_retained_proof(transaction, tenant, &record)
            .await?;
        let profile = record
            .profile
            .as_ref()
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        let ProfileKind::Agent(agent) = profile.kind() else {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        };
        let evidence = agent
            .owner_attestation
            .as_ref()
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        if record.profile_id != *profile_id
            || record.principal_id != principal.principal_id()
            || record.public_key != *agent_public_key
            || agent.claimed_owner != Some(*owner_public_key)
            || evidence.proof_event_id != *proof_event_id
        {
            return Err(IdentityProfileRepositoryError::IdentityConflict);
        }
        let owner = self
            .current_registration_in_transaction(transaction, tenant, *owner_public_key)
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        if owner.principal_id == record.principal_id {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        let (proof, attestation) = if let Some(detached) = record
            .latest_owner_auth_proof
            .as_ref()
            .filter(|proof| proof.source_event_id == *proof_event_id)
        {
            (
                ImmutableAgentOwnerProof::VerifiedAuthentication {
                    proof: detached.clone(),
                },
                detached.verify(*agent_public_key)?,
            )
        } else {
            let source = self
                .events
                .exact_in_transaction(
                    transaction,
                    tenant,
                    EventId::from_bytes(*proof_event_id.as_bytes()),
                )
                .await?
                .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
            verify_signed_event(source.signed_event(), TimestampPolicy::Historical)
                .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
            (
                ImmutableAgentOwnerProof::RetainedProfile {
                    event_id: *proof_event_id,
                },
                verified_owner_attestation(&source.signed_event().event)?
                    .ok_or(IdentityProfileRepositoryError::InvalidRecord)?,
            )
        };
        attestation
            .verify_for_membership(
                &PublicKey::from_bytes(*agent_public_key.as_bytes()),
                now_seconds,
            )
            .map_err(|_| IdentityProfileRepositoryError::IdentityConflict)?;
        Ok(Some(VerifiedAgentMessageAuthor {
            record: ImmutableAgentMessageAttribution {
                community_id: tenant.community_id(),
                profile_id: record.profile_id,
                profile_version: record.version,
                agent_principal_id: record.principal_id,
                agent_public_key: *agent_public_key,
                owner_principal_id: owner.principal_id,
                owner_public_key: *owner_public_key,
                proof_event_id: *proof_event_id,
                proof,
                owner_membership_version: AggregateVersion::FIRST,
                channel_membership_version: AggregateVersion::FIRST,
                channel_role: String::new(),
            },
        }))
    }

    pub(crate) async fn message_author_from_attribution_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        agent_principal_id: PrincipalId,
        profile_id: ProfileId,
        owner_principal_id: PrincipalId,
        proof_event_id: NostrEventId,
        message_source_event_id: NostrEventId,
        attribution: serde_json::Value,
    ) -> Result<collaboration_domain::MessageAuthor, IdentityProfileRepositoryError> {
        ensure_transaction_tenant(transaction, tenant).await?;
        let record: ImmutableAgentMessageAttribution = serde_json::from_value(attribution)
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
        if record.community_id != tenant.community_id()
            || record.agent_principal_id != agent_principal_id
            || record.profile_id != profile_id
            || record.owner_principal_id != owner_principal_id
            || record.proof_event_id != proof_event_id
            || !matches!(record.channel_role.as_str(), "Owner" | "Admin" | "Member")
        {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        let source = self
            .events
            .exact_in_transaction(
                transaction,
                tenant,
                EventId::from_bytes(*message_source_event_id.as_bytes()),
            )
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        verify_signed_event(source.signed_event(), TimestampPolicy::Historical)
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
        if source.signed_event().event.public_key.as_bytes() != record.agent_public_key.as_bytes()
            || !matches!(
                u32::from(source.signed_event().event.kind),
                nostr_compat::generated_kinds::KIND_STREAM_MESSAGE
                    | nostr_compat::generated_kinds::KIND_STREAM_MESSAGE_V2
            )
        {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        let current = select_profile(transaction, tenant, record.agent_public_key, false)
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        let owner = self
            .current_registration_in_transaction(transaction, tenant, record.owner_public_key)
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        if current.profile_id != record.profile_id
            || current.principal_id != record.agent_principal_id
            || owner.principal_id != record.owner_principal_id
            || !matches!(current.profile.as_ref().map(IdentityProfile::kind), Some(ProfileKind::Agent(agent)) if agent.claimed_owner == Some(record.owner_public_key))
        {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        let attestation = match &record.proof {
            ImmutableAgentOwnerProof::VerifiedAuthentication { proof } => {
                if proof.source_event_id != record.proof_event_id {
                    return Err(IdentityProfileRepositoryError::InvalidRecord);
                }
                proof.verify(record.agent_public_key)?
            }
            ImmutableAgentOwnerProof::RetainedProfile { event_id } => {
                if *event_id != record.proof_event_id {
                    return Err(IdentityProfileRepositoryError::InvalidRecord);
                }
                let source = self
                    .events
                    .exact_in_transaction(
                        transaction,
                        tenant,
                        EventId::from_bytes(*event_id.as_bytes()),
                    )
                    .await?
                    .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
                let signed = source.signed_event();
                verify_signed_event(signed, TimestampPolicy::Historical)
                    .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
                if signed.event.kind != 0
                    || signed.event.public_key.as_bytes() != record.agent_public_key.as_bytes()
                {
                    return Err(IdentityProfileRepositoryError::InvalidRecord);
                }
                verified_owner_attestation(&signed.event)?
                    .ok_or(IdentityProfileRepositoryError::InvalidRecord)?
            }
        };
        if attestation.owner.as_bytes() != record.owner_public_key.as_bytes() {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        Ok(VerifiedAgentMessageAuthor { record }.author())
    }

    pub(crate) async fn lock_nostr_channel_membership_identities_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        actor: &AuthenticatedPrincipal,
        target_public_key: NostrPublicKey,
    ) -> Result<(), IdentityProfileRepositoryError> {
        use collaboration_domain::AuthenticatedPrincipalKind;
        ensure_transaction_tenant(transaction, tenant).await?;
        if actor.community_id() != tenant.community_id() {
            return Err(IdentityProfileRepositoryError::TenantBoundaryViolation);
        }
        let mut keys = std::collections::BTreeSet::from([target_public_key]);
        match actor.kind() {
            AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } => {
                keys.insert(*public_key);
            }
            AuthenticatedPrincipalKind::OwnerAttestedAgent {
                agent_public_key,
                owner_public_key,
                ..
            } => {
                keys.extend([*agent_public_key, *owner_public_key]);
            }
            _ => return Err(IdentityProfileRepositoryError::IdentityConflict),
        }
        // AUTH/profile writers take these locks in key order; an actor-first prefix can deadlock a target renewal.
        for key in keys {
            lock_identity(transaction, tenant, key).await?;
        }
        Ok(())
    }

    pub(crate) async fn current_channel_add_policy_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
    ) -> Result<
        collaboration_domain::membership::AgentChannelAddPolicy,
        IdentityProfileRepositoryError,
    > {
        use collaboration_domain::membership::AgentChannelAddPolicy;
        ensure_transaction_tenant(transaction, tenant).await?;
        lock_identity(transaction, tenant, public_key).await?;
        self.current_registration_in_transaction(transaction, tenant, public_key)
            .await?
            .ok_or(IdentityProfileRepositoryError::IdentityConflict)?;
        // A current policy head must remain stable through the membership commit.
        transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT head_event_id FROM public.collaboration_event_heads WHERE community_id=$1 AND kind=$2 AND author_public_key=$3 AND discriminator='' FOR SHARE",
            [tenant.community_id().as_uuid().into(), i32::from(nostr_compat::generated_kinds::KIND_AGENT_PROFILE as u16).into(), public_key.as_bytes().to_vec().into()])).await?;
        let head = self
            .events
            .head_in_transaction(
                transaction,
                tenant,
                &ReplacementCoordinate {
                    kind: nostr_compat::generated_kinds::KIND_AGENT_PROFILE as u16,
                    author: PublicKey::from_bytes(*public_key.as_bytes()),
                    discriminator: None,
                },
            )
            .await?;
        match head {
            None => Ok(AgentChannelAddPolicy::Anyone),
            Some(head) => validate_channel_add_policy(&head.signed_event().event.content),
        }
    }

    pub(crate) async fn store_channel_add_policy_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &VerifiedEventRecord,
    ) -> Result<EventStoreOutcome, IdentityProfileRepositoryError> {
        let event = &source.signed_event().event;
        if source.community_id() != tenant.community_id()
            || u32::from(event.kind) != nostr_compat::generated_kinds::KIND_AGENT_PROFILE
        {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        validate_channel_add_policy(&event.content)?;
        ensure_transaction_tenant(transaction, tenant).await?;
        let key = NostrPublicKey::from_bytes(*event.public_key.as_bytes());
        lock_identity(transaction, tenant, key).await?;
        self.current_registration_in_transaction(transaction, tenant, key)
            .await?
            .ok_or(IdentityProfileRepositoryError::IdentityConflict)?;
        Ok(self
            .events
            .store_in_transaction(
                transaction,
                tenant,
                source,
                EventPersistencePolicy::evaluate(event.kind, PrivacyAdmission::author())
                    .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
            )
            .await?)
    }

    pub async fn current_registration_by_public_key(
        &self,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
    ) -> Result<Option<RegisteredIdentity>, IdentityProfileRepositoryError> {
        let transaction = self.connection.begin().await?;
        let result = self
            .current_registration_in_transaction(&transaction, tenant, public_key)
            .await;
        finish_transaction(transaction, result).await
    }

    pub async fn current_registration_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
    ) -> Result<Option<RegisteredIdentity>, IdentityProfileRepositoryError> {
        Ok(self
            .current_subject_in_transaction(transaction, tenant, public_key)
            .await?
            .filter(|subject| subject.status() != IdentitySubjectStatus::ObservedOnly)
            .map(|subject| subject.record))
    }

    pub async fn current_subject_by_public_key(
        &self,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
    ) -> Result<Option<ObservedIdentitySubject>, IdentityProfileRepositoryError> {
        let transaction = self.connection.begin().await?;
        let result = self
            .current_subject_in_transaction(&transaction, tenant, public_key)
            .await;
        finish_transaction(transaction, result).await
    }

    pub async fn current_subject_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
    ) -> Result<Option<ObservedIdentitySubject>, IdentityProfileRepositoryError> {
        ensure_transaction_tenant(transaction, tenant).await?;
        let current = select_profile(transaction, tenant, public_key, false).await?;
        if let Some(current) = &current {
            self.verify_retained_proof(transaction, tenant, current)
                .await?;
        }
        Ok(current.map(|record| ObservedIdentitySubject { record }))
    }

    pub(crate) async fn prepare_admitted_moderation_target_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        admitted: &crate::admin::moderation::AdmittedModerationTargetObservation,
    ) -> Result<PreparedModerationSubject, IdentityProfileRepositoryError> {
        ensure_transaction_tenant(transaction, tenant).await?;
        let authorization = admitted.authorization();
        if authorization.community_id() != tenant.community_id() {
            return Err(IdentityProfileRepositoryError::TenantBoundaryViolation);
        }
        let public_key = admitted.target_public_key();
        lock_identity(transaction, tenant, public_key).await?;
        if let Some(record) = select_profile(transaction, tenant, public_key, true).await? {
            self.verify_retained_proof(transaction, tenant, &record)
                .await?;
            return Ok(PreparedModerationSubject {
                record,
                new_observation: false,
            });
        }
        let signed = admitted.signed_event();
        Ok(PreparedModerationSubject {
            record: RegisteredIdentity {
                community_id: tenant.community_id(),
                profile_id: ProfileId::from_uuid(Uuid::new_v4()),
                principal_id: PrincipalId::from_uuid(Uuid::new_v4()),
                public_key,
                version: AggregateVersion::FIRST,
                profile: None,
                first_owner_proof_event_id: None,
                source_event_id: None,
                first_owner_auth_proof: None,
                latest_owner_auth_proof: None,
                registration_source_event_id: None,
                registration_verified_at_seconds: None,
                accepted_name: None,
                observation: Some(ObservationEvidence::AdmittedModerationTarget {
                    event: signed.event.clone(),
                    event_id: NostrEventId::from_bytes(*signed.claimed_id.as_bytes()),
                    signature: signed.signature,
                    operation_id: authorization.source().operation_id,
                    actor_principal_id: authorization.principal_id(),
                    verified_at_millis: authorization.authorized_at_millis(),
                }),
            },
            new_observation: true,
        })
    }

    pub(crate) async fn commit_prepared_moderation_subject_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        prepared: PreparedModerationSubject,
    ) -> Result<(), IdentityProfileRepositoryError> {
        ensure_transaction_tenant(transaction, tenant).await?;
        if prepared.record.community_id != tenant.community_id() {
            return Err(IdentityProfileRepositoryError::TenantBoundaryViolation);
        }
        self.verify_retained_proof(transaction, tenant, &prepared.record)
            .await?;
        if prepared.new_observation {
            write_profile(transaction, tenant, &prepared.record).await?;
        }
        Ok(())
    }

    pub async fn observe_retained_event_author_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source_event_id: EventId,
    ) -> Result<ObservedIdentitySubject, IdentityProfileRepositoryError> {
        ensure_transaction_tenant(transaction, tenant).await?;
        let source = self
            .events
            .exact_in_transaction(transaction, tenant, source_event_id)
            .await?
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        let key = NostrPublicKey::from_bytes(*source.signed_event().event.public_key.as_bytes());
        self.observe_source_in_transaction(
            transaction,
            tenant,
            key,
            ObservationEvidence::RetainedEventAuthor {
                event_id: NostrEventId::from_bytes(*source_event_id.as_bytes()),
                verified_at_millis: source.verified_at_millis(),
            },
        )
        .await
    }

    pub async fn observe_verified_nip42_key_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<ObservedIdentitySubject, IdentityProfileRepositoryError> {
        identity
            .for_tenant(tenant)
            .map_err(|_| IdentityProfileRepositoryError::TenantBoundaryViolation)?;
        self.observe_source_in_transaction(
            transaction,
            tenant,
            identity.public_key(),
            ObservationEvidence::VerifiedNip42Key {
                event_id: NostrEventId::from_bytes(*identity.event_id().as_bytes()),
                verified_at_seconds: identity.verified_at_seconds(),
            },
        )
        .await
    }

    pub async fn observe_verified_nip42_presented_owner_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<Option<ObservedIdentitySubject>, IdentityProfileRepositoryError> {
        identity
            .for_tenant(tenant)
            .map_err(|_| IdentityProfileRepositoryError::TenantBoundaryViolation)?;
        ensure_transaction_tenant(transaction, tenant).await?;
        let Some(proof) = identity.owner_attestation() else {
            return Ok(None);
        };
        self.observe_source_in_transaction(
            transaction,
            tenant,
            proof.owner_public_key(),
            ObservationEvidence::VerifiedNip42PresentedOwner {
                authentication_event_id: NostrEventId::from_bytes(*identity.event_id().as_bytes()),
                source_created_at_seconds: identity.signed_authentication_event().event.created_at,
                verified_at_seconds: identity.verified_at_seconds(),
                agent_public_key: identity.public_key(),
                exact_attestation_tag: proof.attestation().to_tag(),
            },
        )
        .await
        .map(Some)
    }

    async fn observe_source_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
        observation: ObservationEvidence,
    ) -> Result<ObservedIdentitySubject, IdentityProfileRepositoryError> {
        ensure_transaction_tenant(transaction, tenant).await?;
        observation.verified_at_millis()?;
        lock_identity(transaction, tenant, public_key).await?;
        if let Some(current) = select_profile(transaction, tenant, public_key, true).await? {
            self.verify_retained_proof(transaction, tenant, &current)
                .await?;
            return Ok(ObservedIdentitySubject { record: current });
        }
        let record = RegisteredIdentity {
            community_id: tenant.community_id(),
            profile_id: ProfileId::from_uuid(Uuid::new_v4()),
            principal_id: PrincipalId::from_uuid(Uuid::new_v4()),
            public_key,
            version: AggregateVersion::FIRST,
            profile: None,
            first_owner_proof_event_id: None,
            source_event_id: None,
            first_owner_auth_proof: None,
            latest_owner_auth_proof: None,
            registration_source_event_id: None,
            registration_verified_at_seconds: None,
            accepted_name: None,
            observation: Some(observation),
        };
        self.verify_retained_proof(transaction, tenant, &record)
            .await?;
        write_profile(transaction, tenant, &record).await?;
        Ok(ObservedIdentitySubject { record })
    }

    pub async fn current_by_public_key(
        &self,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
    ) -> Result<Option<PersistedIdentityProfile>, IdentityProfileRepositoryError> {
        self.current_registration_by_public_key(tenant, public_key)
            .await?
            .map(|record| record.persisted_profile())
            .transpose()
            .map(Option::flatten)
    }

    async fn register_key(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<RegisteredIdentity, IdentityProfileRepositoryError> {
        identity
            .for_tenant(tenant)
            .map_err(|_| IdentityProfileRepositoryError::TenantBoundaryViolation)?;
        self.register_verified_key(
            tenant,
            identity.public_key(),
            identity.event_id(),
            identity.verified_at_seconds(),
        )
        .await
    }

    async fn register_verified_key(
        &self,
        tenant: &TenantContext,
        public_key: NostrPublicKey,
        source_event_id: EventId,
        verified_at_seconds: u64,
    ) -> Result<RegisteredIdentity, IdentityProfileRepositoryError> {
        let transaction = self.connection.begin().await?;
        let result = async {
            lock_identity(&transaction, tenant, public_key).await?;
            if let Some(mut current) =
                select_profile(&transaction, tenant, public_key, true).await?
            {
                self.verify_retained_proof(&transaction, tenant, &current)
                    .await?;
                if current.registration_source_event_id.is_none() && current.profile.is_none() {
                    current.version = current
                        .version
                        .next()
                        .ok_or(IdentityProfileRepositoryError::VersionExhausted)?;
                    current.registration_source_event_id =
                        Some(NostrEventId::from_bytes(*source_event_id.as_bytes()));
                    current.registration_verified_at_seconds = Some(verified_at_seconds);
                    write_profile(&transaction, tenant, &current).await?;
                }
                return Ok(current);
            }
            let record = RegisteredIdentity {
                community_id: tenant.community_id(),
                profile_id: ProfileId::from_uuid(Uuid::new_v4()),
                principal_id: PrincipalId::from_uuid(Uuid::new_v4()),
                public_key,
                version: AggregateVersion::FIRST,
                profile: None,
                first_owner_proof_event_id: None,
                source_event_id: None,
                first_owner_auth_proof: None,
                latest_owner_auth_proof: None,
                registration_source_event_id: Some(NostrEventId::from_bytes(
                    *source_event_id.as_bytes(),
                )),
                registration_verified_at_seconds: Some(verified_at_seconds),
                accepted_name: None,
                observation: None,
            };
            write_profile(&transaction, tenant, &record).await?;
            Ok(record)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn register_owner(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<RegisteredIdentity, IdentityProfileRepositoryError> {
        identity
            .for_tenant(tenant)
            .map_err(|_| IdentityProfileRepositoryError::TenantBoundaryViolation)?;
        let detached = DetachedOwnerProof::from_receipt(identity)
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        let attestation = detached.verify(identity.public_key())?;
        let transaction = self.connection.begin().await?;
        let result = async {
            lock_identity(&transaction, tenant, identity.public_key()).await?;
            let mut record = select_profile(&transaction, tenant, identity.public_key(), true)
                .await?
                .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
            self.verify_retained_proof(&transaction, tenant, &record)
                .await?;
            if let Some(profile) = &record.profile {
                let ProfileKind::Agent(agent) = profile.kind() else {
                    return Err(IdentityProfileRepositoryError::IdentityConflict);
                };
                if agent.claimed_owner
                    != Some(NostrPublicKey::from_bytes(*attestation.owner.as_bytes()))
                {
                    return Err(IdentityProfileRepositoryError::IdentityConflict);
                }
            }
            if record.latest_owner_auth_proof.as_ref() == Some(&detached) {
                return Ok(record);
            }
            let previous = record.profile.clone();
            record.version = record
                .version
                .next()
                .ok_or(IdentityProfileRepositoryError::VersionExhausted)?;
            let mut fields = previous
                .as_ref()
                .map(|profile| profile.fields().clone())
                .unwrap_or_else(|| profile_fields(&record));
            fields.version = record.version;
            fields.kind = ProfileKind::Agent(AgentProfile {
                claimed_owner: Some(NostrPublicKey::from_bytes(*attestation.owner.as_bytes())),
                owner_attestation: Some(owner_evidence(
                    record.public_key,
                    &attestation,
                    detached.source_event_id,
                    detached.verified_at_seconds,
                )),
            });
            if previous.is_none() {
                record.first_owner_auth_proof = Some(detached.clone());
                let head = self
                    .events
                    .head_in_transaction(
                        &transaction,
                        tenant,
                        &ReplacementCoordinate {
                            kind: 0,
                            author: PublicKey::from_bytes(*record.public_key.as_bytes()),
                            discriminator: None,
                        },
                    )
                    .await?;
                if let Some(head) = head {
                    let event = &head.signed_event().event;
                    verify_signed_event(head.signed_event(), TimestampPolicy::Historical)
                        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
                    if let Some(proof) = verified_owner_attestation(event)? {
                        if proof.owner != attestation.owner {
                            return Err(IdentityProfileRepositoryError::IdentityConflict);
                        }
                    }
                    let source =
                        NostrEventId::from_bytes(*head.signed_event().claimed_id.as_bytes());
                    record.source_event_id = Some(source);
                    fields.metadata = Some(AuthoredValue {
                        source_event_id: source,
                        source_author: record.public_key,
                        source_created_at: event.created_at,
                        value: profile_metadata(event)?,
                    });
                }
            }
            let profile = IdentityProfile::new(fields)
                .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
            if let Some(previous) = &previous {
                validate_profile_update(previous, &profile)
                    .map_err(|_| IdentityProfileRepositoryError::IdentityConflict)?;
            }
            record.profile = Some(profile);
            record.latest_owner_auth_proof = Some(detached);
            write_profile(&transaction, tenant, &record).await?;
            Ok(record)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn project_profile_event(
        &self,
        tenant: &TenantContext,
        source: &VerifiedEventRecord,
    ) -> Result<IdentityProfileProjectionOutcome, IdentityProfileRepositoryError> {
        let transaction = self.connection.begin().await?;
        let result = self
            .project_profile_event_in_transaction(&transaction, tenant, source)
            .await;
        finish_transaction(transaction, result).await
    }

    pub async fn project_profile_event_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &VerifiedEventRecord,
    ) -> Result<IdentityProfileProjectionOutcome, IdentityProfileRepositoryError> {
        self.project_profile_with_owner_policy_in_transaction(
            transaction,
            tenant,
            source,
            ProfileOwnerEvidencePolicy::StrictProjection,
        )
        .await
    }

    async fn project_profile_with_owner_policy_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &VerifiedEventRecord,
        owner_policy: ProfileOwnerEvidencePolicy,
    ) -> Result<IdentityProfileProjectionOutcome, IdentityProfileRepositoryError> {
        if source.community_id() != tenant.community_id() {
            return Err(IdentityProfileRepositoryError::TenantBoundaryViolation);
        }
        let event = &source.signed_event().event;
        if u32::from(event.kind) != KIND_PROFILE {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        let metadata = profile_metadata(event)?;
        // Ordinary signed metadata may retain an invalid optional claim without authority;
        // explicit profile projection must reject invalid supplied owner proof.
        let attestation = match owner_policy {
            ProfileOwnerEvidencePolicy::StrictProjection => verified_owner_attestation(event)?,
            ProfileOwnerEvidencePolicy::OptionalAuthoredMetadata => {
                optional_profile_owner_attestation(event)
            }
        };
        let public_key = NostrPublicKey::from_bytes(*event.public_key.as_bytes());
        let source_event_id =
            NostrEventId::from_bytes(*source.signed_event().claimed_id.as_bytes());
        let persistence = EventPersistencePolicy::evaluate(event.kind, PrivacyAdmission::author())
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
        ensure_transaction_tenant(transaction, tenant).await?;
        lock_identity(transaction, tenant, public_key).await?;
        let previous = select_profile(transaction, tenant, public_key, true).await?;
        if let Some(previous) = &previous {
            self.verify_retained_proof(transaction, tenant, previous)
                .await?;
            if let Some(profile) = &previous.profile {
                let ProfileKind::Agent(agent) = profile.kind() else {
                    return Err(IdentityProfileRepositoryError::IdentityConflict);
                };
                if attestation.as_ref().is_some_and(|proof| {
                    agent.claimed_owner != Some(NostrPublicKey::from_bytes(*proof.owner.as_bytes()))
                }) {
                    return Err(IdentityProfileRepositoryError::IdentityConflict);
                }
            }
        }
        let outcome = self
            .events
            .store_in_transaction(transaction, tenant, source, persistence)
            .await?;
        if outcome == EventStoreOutcome::Stale {
            return Ok(IdentityProfileProjectionOutcome::Stale);
        }
        if let Some(previous) = &previous {
            if previous.source_event_id == Some(source_event_id) {
                let profile = previous
                    .persisted_profile()?
                    .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
                return Ok(IdentityProfileProjectionOutcome::Unchanged(Box::new(
                    profile,
                )));
            }
        }
        if previous
            .as_ref()
            .is_none_or(|record| record.profile.is_none())
            && attestation.is_none()
        {
            return Ok(IdentityProfileProjectionOutcome::Unclassified);
        }
        let mut record = previous.clone().unwrap_or_else(|| RegisteredIdentity {
            community_id: tenant.community_id(),
            profile_id: ProfileId::from_uuid(Uuid::new_v4()),
            principal_id: PrincipalId::from_uuid(Uuid::new_v4()),
            public_key,
            version: AggregateVersion::FIRST,
            profile: None,
            first_owner_proof_event_id: Some(source_event_id),
            source_event_id: None,
            first_owner_auth_proof: None,
            latest_owner_auth_proof: None,
            registration_source_event_id: None,
            registration_verified_at_seconds: None,
            accepted_name: None,
            observation: None,
        });
        if previous.is_some() {
            record.version = record
                .version
                .next()
                .ok_or(IdentityProfileRepositoryError::VersionExhausted)?;
        }
        let mut fields = record
            .profile
            .as_ref()
            .map(|profile| profile.fields().clone())
            .unwrap_or_else(|| profile_fields(&record));
        fields.version = record.version;
        if let Some(attestation) = attestation {
            let verified_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?
                .as_secs();
            fields.kind = ProfileKind::Agent(AgentProfile {
                claimed_owner: Some(NostrPublicKey::from_bytes(*attestation.owner.as_bytes())),
                owner_attestation: Some(owner_evidence(
                    public_key,
                    &attestation,
                    source_event_id,
                    verified_at,
                )),
            });
            record.latest_owner_auth_proof = None;
            if record.first_owner_proof_event_id.is_none()
                && record.first_owner_auth_proof.is_none()
            {
                record.first_owner_proof_event_id = Some(source_event_id);
            }
        }
        fields.metadata = Some(AuthoredValue {
            source_event_id,
            source_author: public_key,
            source_created_at: event.created_at,
            value: metadata,
        });
        let profile = IdentityProfile::new(fields)
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
        if let Some(previous) = previous
            .as_ref()
            .and_then(|previous| previous.profile.as_ref())
        {
            validate_profile_update(previous, &profile)
                .map_err(|_| IdentityProfileRepositoryError::IdentityConflict)?;
        }
        record.profile = Some(profile);
        record.source_event_id = Some(source_event_id);
        write_profile(transaction, tenant, &record).await?;
        Ok(IdentityProfileProjectionOutcome::Materialized(Box::new(
            record
                .persisted_profile()?
                .ok_or(IdentityProfileRepositoryError::InvalidRecord)?,
        )))
    }

    pub(crate) async fn current_closed_nip42_membership_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        now_seconds: u64,
    ) -> Result<collaboration_domain::CommunityMembership, NostrPrincipalResolutionError> {
        use crate::messages::channel_admission::{
            current_community_serving_state_in_transaction,
            current_subject_memberships_in_transaction,
        };
        use collaboration_domain::{
            AuthenticatedPrincipalKind, CommunityArchivePolicyState, CommunityArchiveSnapshot,
            CommunityLifecycleState, IdentityArchivePolicyState, IdentityArchiveSnapshot,
            MembershipStatus, ModerationSnapshot,
        };
        let denied = || NostrPrincipalResolutionError::Denied;
        let unavailable = || NostrPrincipalResolutionError::Unavailable;
        if principal.community_id() != tenant.community_id()
            || now_seconds == 0
            || !matches!(
                principal.kind(),
                AuthenticatedPrincipalKind::NostrIdentity {
                    authentication_method: NostrAuthenticationMethod::Nip42,
                    ..
                } | AuthenticatedPrincipalKind::OwnerAttestedAgent {
                    authentication_method: NostrAuthenticationMethod::Nip42,
                    ..
                }
            )
        {
            return Err(denied());
        }
        ensure_transaction_tenant(transaction, tenant)
            .await
            .map_err(|_| unavailable())?;
        let author = self
            .verified_message_agent_in_transaction(transaction, tenant, principal, now_seconds)
            .await
            .map_err(|error| match error {
                IdentityProfileRepositoryError::IdentityConflict
                | IdentityProfileRepositoryError::TenantBoundaryViolation => denied(),
                _ => unavailable(),
            })?;
        let community = current_community_serving_state_in_transaction(transaction, tenant)
            .await
            .map_err(|_| unavailable())?
            .ok_or_else(denied)?;
        let authority =
            axum::http::uri::Authority::from_str(tenant.route_reference()).map_err(|_| denied())?;
        if community.lifecycle != CommunityLifecycleState::Active
            || authority.host().to_ascii_lowercase() != community.host.as_str()
        {
            return Err(denied());
        }
        let owner = author.as_ref().map(|author| author.owner_principal_id());
        transaction.query_all(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT principal_id FROM public.collaboration_community_memberships WHERE community_id=$1 AND (principal_id=$2 OR principal_id=$3) ORDER BY principal_id FOR SHARE",
            [tenant.community_id().as_uuid().into(), principal.principal_id().as_uuid().into(), owner.map(|owner| owner.as_uuid()).into()])).await.map_err(|_| unavailable())?;
        let current = current_subject_memberships_in_transaction(
            transaction,
            tenant,
            principal.principal_id(),
            None,
        )
        .await
        .map_err(|_| unavailable())?
        .community;
        if current.is_some_and(|member| member.status != MembershipStatus::Active) {
            return Err(denied());
        }
        for subject in std::collections::BTreeSet::from_iter(
            std::iter::once(principal.principal_id()).chain(owner),
        ) {
            let (identity_archive, community_archive) =
                crate::admin::moderation::current_actor_archives_in_transaction(
                    transaction,
                    tenant,
                    subject,
                )
                .await
                .map_err(|_| unavailable())?;
            if !matches!(
                identity_archive,
                ModerationSnapshot::Absent
                    | ModerationSnapshot::Current(IdentityArchiveSnapshot {
                        state: IdentityArchivePolicyState::Visible,
                        ..
                    })
            ) || !matches!(
                community_archive,
                ModerationSnapshot::Absent
                    | ModerationSnapshot::Current(CommunityArchiveSnapshot {
                        state: CommunityArchivePolicyState::Active,
                        ..
                    })
            ) {
                return Err(denied());
            }
        }
        if let Some(owner) = owner {
            let owner_member =
                current_subject_memberships_in_transaction(transaction, tenant, owner, None)
                    .await
                    .map_err(|_| unavailable())?
                    .community
                    .ok_or_else(denied)?;
            return collaboration_domain::VirtualAgentMembershipEvidence::derive(
                principal,
                owner,
                owner_member,
                owner_member.version,
            )
            .map(|evidence| evidence.policy_membership_snapshot())
            .map_err(|_| denied());
        }
        current.ok_or_else(denied)
    }

    async fn admit_profile_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        action: collaboration_domain::AuthorizationAction,
        now_millis: u64,
    ) -> Result<crate::tenant_admission::AuthorizedRpcRequest, NostrHttpDirectoryError> {
        use crate::{
            admin::moderation::{
                current_actor_archives_in_transaction, current_restriction_in_transaction,
                lock_moderation_in_transaction,
            },
            messages::channel_admission::{
                current_community_serving_state_in_transaction,
                current_principal_memberships_in_transaction,
            },
        };
        use collaboration_domain::{
            AggregateId, AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind,
            AuthorizationScope, CommunityLifecycleState, ModerationAuthorizationDecision,
            ModerationAuthorizationRequest, ModerationResourceContext, ModerationSnapshot,
            authorize_with_moderation,
        };
        let unavailable = || NostrHttpDirectoryError::Unavailable;
        let denied = || NostrHttpDirectoryError::Denied;
        if principal.community_id() != tenant.community_id() || now_millis == 0 {
            return Err(denied());
        }
        lock_moderation_in_transaction(transaction, tenant)
            .await
            .map_err(|_| unavailable())?;
        let community = current_community_serving_state_in_transaction(transaction, tenant)
            .await
            .map_err(|_| unavailable())?
            .ok_or_else(denied)?;
        if community.lifecycle != CommunityLifecycleState::Active {
            return Err(denied());
        }
        // The routing authority is explicit deployment configuration; a conflicting SQL host is corruption, never a second tenant fallback.
        let route_host =
            axum::http::uri::Authority::from_str(tenant.route_reference()).map_err(|_| denied())?;
        if route_host.host().to_ascii_lowercase() != community.host.as_str() {
            return Err(denied());
        }
        let key = match principal.kind() {
            collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity {
                public_key, ..
            } => *public_key,
            _ => return Err(denied()),
        };
        let subject = self
            .current_subject_in_transaction(transaction, tenant, key)
            .await
            .map_err(|_| unavailable())?
            .ok_or_else(denied)?;
        if subject.principal_id() != principal.principal_id()
            || subject.status() == IdentitySubjectStatus::ObservedOnly
        {
            return Err(denied());
        }
        transaction.query_all(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT principal_id FROM public.collaboration_community_memberships WHERE community_id = $1 AND principal_id = $2 FOR SHARE",
            [tenant.community_id().as_uuid().into(), principal.principal_id().as_uuid().into()]))
            .await.map_err(|_| unavailable())?;
        let members =
            current_principal_memberships_in_transaction(transaction, tenant, principal, None)
                .await
                .map_err(|_| unavailable())?;
        let member = members.community.ok_or_else(denied)?;
        let scope = AuthorizationScope::new(match action {
            collaboration_domain::AuthorizationAction::Read => "users:read",
            collaboration_domain::AuthorizationAction::Write => "users:write",
            _ => return Err(denied()),
        })
        .map_err(|_| denied())?;
        let authorization = AuthorizationRequest {
            tenant,
            principal,
            required_scope: &scope,
            action,
            resource: AuthorizationResource {
                community_id: tenant.community_id(),
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(tenant.community_id().as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: member.version,
            community_membership: Some(member),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis,
        };
        let restriction =
            current_restriction_in_transaction(transaction, tenant, principal.principal_id())
                .await
                .map_err(|_| unavailable())?;
        let (principal_archive, community_archive) =
            current_actor_archives_in_transaction(transaction, tenant, principal.principal_id())
                .await
                .map_err(|_| unavailable())?;
        match authorize_with_moderation(&ModerationAuthorizationRequest {
            authorization: &authorization,
            restriction: restriction
                .as_ref()
                .map_or(ModerationSnapshot::Absent, ModerationSnapshot::Current),
            principal_archive,
            community_archive,
            resource_context: ModerationResourceContext::Current,
        }) {
            ModerationAuthorizationDecision::Allowed(_) => {
                crate::tenant_admission::AuthorizedRpcRequest::authorize(&authorization)
                    .map_err(|_| denied())
            }
            ModerationAuthorizationDecision::Denied(_) => Err(denied()),
        }
    }

    pub async fn admit_profile_transport(
        &self,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        now_millis: u64,
    ) -> Result<crate::tenant_admission::AuthorizedRpcRequest, NostrHttpDirectoryError> {
        let transaction = self
            .connection
            .begin_with_config(Some(sea_orm::IsolationLevel::ReadCommitted), None)
            .await
            .map_err(|_| NostrHttpDirectoryError::Unavailable)?;
        let result = self
            .admit_profile_in_transaction(
                &transaction,
                tenant,
                principal,
                collaboration_domain::AuthorizationAction::Write,
                now_millis,
            )
            .await;
        match result {
            Ok(admission) => {
                transaction
                    .commit()
                    .await
                    .map_err(|_| NostrHttpDirectoryError::Unavailable)?;
                Ok(admission)
            }
            Err(error) => {
                transaction
                    .rollback()
                    .await
                    .map_err(|_| NostrHttpDirectoryError::Unavailable)?;
                Err(error)
            }
        }
    }

    pub async fn query_authored_profiles(
        &self,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        queries: &[crate::db::collaboration::event_repository::EventRepositoryQuery],
        now_millis: u64,
    ) -> Result<
        Vec<crate::db::collaboration::event_repository::StoredEventRecord>,
        NostrHttpDirectoryError,
    > {
        if queries.is_empty()
            || queries.len() > crate::nostr::subscriptions::MAX_FILTERS_PER_REQUEST
        {
            return Err(NostrHttpDirectoryError::Denied);
        }
        let transaction = self
            .connection
            .begin_with_config(Some(sea_orm::IsolationLevel::ReadCommitted), None)
            .await
            .map_err(|_| NostrHttpDirectoryError::Unavailable)?;
        let result = async {
            self.admit_profile_in_transaction(
                &transaction,
                tenant,
                principal,
                collaboration_domain::AuthorizationAction::Read,
                now_millis,
            )
            .await?;
            let mut records = Vec::new();
            for query in queries {
                let filter_records = self
                    .events
                    .query_in_transaction(&transaction, tenant, query)
                    .await
                    .map_err(|_| NostrHttpDirectoryError::Unavailable)?;
                if filter_records.len()
                    > crate::nostr::subscriptions::MAX_HISTORICAL_EVENTS_PER_REQUEST
                        .saturating_sub(records.len())
                {
                    return Err(NostrHttpDirectoryError::Unavailable);
                }
                for record in &filter_records {
                    if u32::from(record.signed_event().event.kind) != KIND_PROFILE {
                        return Err(NostrHttpDirectoryError::Denied);
                    }
                    let key = NostrPublicKey::from_bytes(
                        *record.signed_event().event.public_key.as_bytes(),
                    );
                    let authored = self
                        .current_authored_metadata_in_transaction(&transaction, tenant, key)
                        .await
                        .map_err(|_| NostrHttpDirectoryError::Unavailable)?
                        .ok_or(NostrHttpDirectoryError::Unavailable)?;
                    if authored.metadata().source_event_id.as_bytes()
                        != record.signed_event().claimed_id.as_bytes()
                    {
                        return Err(NostrHttpDirectoryError::Unavailable);
                    }
                }
                records.extend(filter_records);
            }
            Ok(records)
        }
        .await;
        match result {
            Ok(records) => {
                transaction
                    .commit()
                    .await
                    .map_err(|_| NostrHttpDirectoryError::Unavailable)?;
                Ok(records)
            }
            Err(error) => {
                transaction
                    .rollback()
                    .await
                    .map_err(|_| NostrHttpDirectoryError::Unavailable)?;
                Err(error)
            }
        }
    }

    async fn verify_retained_proof(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        record: &RegisteredIdentity,
    ) -> Result<(), IdentityProfileRepositoryError> {
        if let Some(accepted) = &record.accepted_name {
            for event_id in [Some(accepted.processed_event_id), accepted.source_event_id]
                .into_iter()
                .flatten()
            {
                let source = self
                    .events
                    .exact_in_transaction(
                        transaction,
                        tenant,
                        EventId::from_bytes(*event_id.as_bytes()),
                    )
                    .await?
                    .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
                if source.community_id() != tenant.community_id()
                    || source.signed_event().event.kind != 0
                    || source.signed_event().event.public_key.as_bytes()
                        != record.public_key.as_bytes()
                {
                    return Err(IdentityProfileRepositoryError::InvalidRecord);
                }
                verify_signed_event(source.signed_event(), TimestampPolicy::Historical)
                    .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
                let metadata = profile_metadata(&source.signed_event().event)?;
                metadata
                    .validate()
                    .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
                if accepted.source_event_id == Some(event_id) {
                    let name = accepted
                        .name
                        .as_ref()
                        .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
                    let (_, domain) = name
                        .split_once('@')
                        .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
                    if domain != accepted.host
                        || metadata.nip05_handle.as_ref().and_then(|claim| {
                            crate::nostr::http::canonical_nip05_name(claim, Some(domain))
                        }) != Some(name.clone())
                    {
                        return Err(IdentityProfileRepositoryError::InvalidRecord);
                    }
                }
            }
        }
        if let Some(observation) = &record.observation {
            match observation {
                ObservationEvidence::RetainedEventAuthor {
                    event_id,
                    verified_at_millis,
                } => {
                    let source = self
                        .events
                        .exact_in_transaction(
                            transaction,
                            tenant,
                            EventId::from_bytes(*event_id.as_bytes()),
                        )
                        .await?
                        .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
                    verify_signed_event(source.signed_event(), TimestampPolicy::Historical)
                        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
                    if source.community_id() != tenant.community_id()
                        || source.signed_event().event.public_key.as_bytes()
                            != record.public_key.as_bytes()
                        || source.verified_at_millis() != *verified_at_millis
                    {
                        return Err(IdentityProfileRepositoryError::InvalidRecord);
                    }
                }
                ObservationEvidence::AdmittedModerationTarget {
                    event,
                    event_id,
                    signature,
                    operation_id,
                    actor_principal_id,
                    verified_at_millis,
                } => {
                    let invalid = || IdentityProfileRepositoryError::InvalidRecord;
                    let signed = nostr_compat::SignedEvent {
                        event: event.clone(),
                        claimed_id: EventId::from_bytes(*event_id.as_bytes()),
                        signature: *signature,
                    };
                    verify_signed_event(&signed, TimestampPolicy::Historical)
                        .map_err(|_| invalid())?;
                    if !matches!(
                        u32::from(event.kind),
                        nostr_compat::generated_kinds::KIND_MODERATION_BAN
                            | nostr_compat::generated_kinds::KIND_MODERATION_UNBAN
                            | nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT
                            | nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT
                    ) || crate::nostr::event_ingest::operation_id(
                        tenant.community_id().as_uuid(),
                        signed.claimed_id,
                    ) != *operation_id
                        || event.created_at.abs_diff(verified_at_millis / 1000) > 120
                    {
                        return Err(invalid());
                    }
                    let target = crate::admin::moderation::signed_restriction_target(event)
                        .ok_or_else(invalid)?;
                    if target != record.public_key {
                        return Err(invalid());
                    }
                    let entry =
                        crate::audit::repository::AuditRepository::load_operation_in_transaction(
                            transaction,
                            tenant,
                            *operation_id,
                        )
                        .await
                        .map_err(|error| match error {
                            crate::audit::repository::AuditRepositoryError::Unavailable(error) => {
                                IdentityProfileRepositoryError::Unavailable(error)
                            }
                            _ => invalid(),
                        })?
                        .ok_or_else(invalid)?;
                    let audit = entry.record();
                    let expected_action = if matches!(
                        u32::from(event.kind),
                        nostr_compat::generated_kinds::KIND_MODERATION_UNBAN
                            | nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT
                    ) {
                        crate::audit::events::ModerationAuditOperation::LiftRestriction.action()
                    } else {
                        crate::audit::events::ModerationAuditOperation::ApplyRestriction.action()
                    };
                    if audit.operation_id() != *operation_id
                        || audit.actor_principal_id() != Some(*actor_principal_id)
                        || audit.action().as_str() != expected_action
                        || audit.outcome() != collaboration_domain::AuditOutcome::Succeeded
                        || audit.occurred_at_millis() != *verified_at_millis
                    {
                        return Err(invalid());
                    }
                    for name in ["subject_principal_id", "moderation_record_id"] {
                        let field = audit
                            .fields()
                            .as_slice()
                            .iter()
                            .find(|field| field.name().as_str() == name)
                            .ok_or_else(invalid)?;
                        if !matches!(field.value(), collaboration_domain::AuditValue::Identifier(value) if value.as_str() == record.principal_id.to_string())
                        {
                            return Err(invalid());
                        }
                    }
                    let action = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres, r#"
SELECT action_kind, actor_principal_id, target_principal_id, record_kind, record_id, source_system, source_record_id,
       integrity_algorithm, integrity_value, (extract(epoch FROM occurred_at) * 1000)::text AS occurred_at
FROM public.collaboration_moderation_actions WHERE community_id = $1 AND operation_id = $2
"#, [tenant.community_id().as_uuid().into(), operation_id.as_uuid().into()])).await?.ok_or_else(invalid)?;
                    let action_kind: String = action.try_get("", "action_kind")?;
                    let expected_kind = match u32::from(event.kind) {
                        nostr_compat::generated_kinds::KIND_MODERATION_BAN => "apply_ban",
                        nostr_compat::generated_kinds::KIND_MODERATION_UNBAN => "lift_ban",
                        nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT => "apply_timeout",
                        nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT => "lift_timeout",
                        _ => return Err(invalid()),
                    };
                    if action_kind != expected_kind
                        || action.try_get::<Uuid>("", "actor_principal_id")?
                            != actor_principal_id.as_uuid()
                        || action.try_get::<Option<Uuid>>("", "target_principal_id")?
                            != Some(record.principal_id.as_uuid())
                        || action.try_get::<Uuid>("", "record_id")? != record.principal_id.as_uuid()
                        || action.try_get::<String>("", "record_kind")? != "restriction"
                        || action.try_get::<String>("", "source_system")? != "nostr"
                        || action.try_get::<String>("", "source_record_id")?
                            != signed.claimed_id.to_hex()
                        || action
                            .try_get::<Option<String>>("", "integrity_algorithm")?
                            .as_deref()
                            != Some("nostr_event_id")
                        || action.try_get::<Option<String>>("", "integrity_value")?
                            != Some(signed.claimed_id.to_hex())
                        || crate::admin::moderation::exact_milliseconds(
                            &action.try_get::<String>("", "occurred_at")?,
                        )
                        .map_err(|_| invalid())?
                            != *verified_at_millis
                    {
                        return Err(invalid());
                    }
                }
                ObservationEvidence::VerifiedNip42Key { .. } => {
                    // Transient AUTH provenance is supplied only by the actual verifier; AUTH originals are not persisted.
                    observation.verified_at_millis()?;
                }
                ObservationEvidence::VerifiedNip42PresentedOwner {
                    source_created_at_seconds,
                    agent_public_key,
                    exact_attestation_tag,
                    ..
                } => {
                    let proof = OwnerAttestation::parse_tag(exact_attestation_tag)
                        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
                    proof
                        .verify_for_membership(
                            &PublicKey::from_bytes(*agent_public_key.as_bytes()),
                            *source_created_at_seconds,
                        )
                        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
                    if proof.owner.as_bytes() != record.public_key.as_bytes()
                        || agent_public_key == &record.public_key
                    {
                        return Err(IdentityProfileRepositoryError::InvalidRecord);
                    }
                    observation.verified_at_millis()?;
                }
            }
        }
        let Some(profile) = &record.profile else {
            return Ok(());
        };
        let ProfileKind::Agent(agent) = profile.kind() else {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        };
        let current = agent
            .owner_attestation
            .as_ref()
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        let owner = agent
            .claimed_owner
            .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
        if let Some(first) = &record.first_owner_auth_proof {
            if NostrPublicKey::from_bytes(*first.verify(record.public_key)?.owner.as_bytes())
                != owner
            {
                return Err(IdentityProfileRepositoryError::InvalidRecord);
            }
        }
        for id in [
            record.first_owner_proof_event_id,
            Some(current.proof_event_id),
        ] {
            let Some(id) = id else {
                continue;
            };
            if let Some(detached) = &record.latest_owner_auth_proof {
                if detached.source_event_id == id {
                    let proof = detached.verify(record.public_key)?;
                    if NostrPublicKey::from_bytes(*proof.owner.as_bytes()) != owner
                        || current.exact_conditions != proof.conditions.as_str()
                        || current.verified_at != detached.verified_at_seconds
                    {
                        return Err(IdentityProfileRepositoryError::InvalidRecord);
                    }
                    continue;
                }
            }
            let proof = self
                .events
                .exact_in_transaction(transaction, tenant, EventId::from_bytes(*id.as_bytes()))
                .await?
                .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
            verify_signed_event(proof.signed_event(), TimestampPolicy::Historical)
                .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
            let event = &proof.signed_event().event;
            let attestation = verified_owner_attestation(event)?
                .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
            if u32::from(event.kind) != KIND_PROFILE
                || NostrPublicKey::from_bytes(*event.public_key.as_bytes()) != record.public_key
                || NostrPublicKey::from_bytes(*attestation.owner.as_bytes()) != owner
                || (current.proof_event_id == id
                    && current.exact_conditions != attestation.conditions.as_str())
            {
                return Err(IdentityProfileRepositoryError::InvalidRecord);
            }
        }
        if let Some(id) = record.source_event_id {
            let source = self
                .events
                .exact_in_transaction(transaction, tenant, EventId::from_bytes(*id.as_bytes()))
                .await?
                .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
            verify_signed_event(source.signed_event(), TimestampPolicy::Historical)
                .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
            let metadata = profile
                .fields()
                .metadata
                .as_ref()
                .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
            let event = &source.signed_event().event;
            if u32::from(event.kind) != KIND_PROFILE
                || NostrPublicKey::from_bytes(*event.public_key.as_bytes()) != record.public_key
                || event.created_at != metadata.source_created_at
                || profile_metadata(event)? != metadata.value
            {
                return Err(IdentityProfileRepositoryError::InvalidRecord);
            }
        } else if profile.fields().metadata.is_some() {
            return Err(IdentityProfileRepositoryError::InvalidRecord);
        }
        Ok(())
    }
}

#[async_trait]
impl NostrPrincipalResolver for IdentityProfileRepository {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<AuthenticatedPrincipal, NostrPrincipalResolutionError> {
        identity.for_tenant(tenant)?;
        let registered = self
            .register_key(tenant, identity)
            .await
            .map_err(resolution_error)?;
        let scopes = PrincipalScopes::for_nip42_authenticated_key()
            .map_err(|_| NostrPrincipalResolutionError::Denied)?;
        if identity.owner_attestation().is_some() {
            match self.register_owner(tenant, identity).await {
                Ok(record) => {
                    let profile = record
                        .profile
                        .as_ref()
                        .ok_or(NostrPrincipalResolutionError::Denied)?;
                    return AuthenticatedPrincipal::owner_attested_agent(
                        record.principal_id,
                        tenant.community_id(),
                        profile,
                        NostrAuthenticationMethod::Nip42,
                        scopes,
                    )
                    .map_err(|_| NostrPrincipalResolutionError::Denied);
                }
                Err(
                    IdentityProfileRepositoryError::IdentityConflict
                    | IdentityProfileRepositoryError::Unavailable(_)
                    | IdentityProfileRepositoryError::EventRepository(
                        EventRepositoryError::Unavailable(_),
                    ),
                ) => {
                    // Buzz keeps successful exact-key AUTH independent of a denied or unavailable owner backfill.
                }
                Err(error) => return Err(resolution_error(error)),
            }
        }
        Ok(AuthenticatedPrincipal::nostr_identity(
            registered.principal_id,
            tenant.community_id(),
            identity.public_key(),
            NostrAuthenticationMethod::Nip42,
            scopes,
        ))
    }
}

fn resolution_error(error: IdentityProfileRepositoryError) -> NostrPrincipalResolutionError {
    match error {
        IdentityProfileRepositoryError::Unavailable(_)
        | IdentityProfileRepositoryError::EventRepository(EventRepositoryError::Unavailable(_)) => {
            NostrPrincipalResolutionError::Unavailable
        }
        _ => NostrPrincipalResolutionError::Denied,
    }
}

#[async_trait]
impl Nip98PrincipalResolver for IdentityProfileRepository {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip98Identity,
    ) -> Result<AuthenticatedPrincipal, NostrHttpDirectoryError> {
        identity.for_tenant(tenant)?;
        let registered = self
            .register_verified_key(
                tenant,
                identity.public_key(),
                identity.event_id(),
                identity.verified_at_seconds(),
            )
            .await
            .map_err(|error| match resolution_error(error) {
                NostrPrincipalResolutionError::Denied
                | NostrPrincipalResolutionError::Banned
                | NostrPrincipalResolutionError::UnrepresentedSubject => {
                    NostrHttpDirectoryError::Denied
                }
                NostrPrincipalResolutionError::Unavailable => NostrHttpDirectoryError::Unavailable,
            })?;
        let scopes = PrincipalScopes::for_nip42_authenticated_key()
            .map_err(|_| NostrHttpDirectoryError::Denied)?;
        Ok(AuthenticatedPrincipal::nostr_identity(
            registered.principal_id,
            tenant.community_id(),
            identity.public_key(),
            NostrAuthenticationMethod::Nip98,
            scopes,
        ))
    }
}

fn profile_fields(record: &RegisteredIdentity) -> ProfileRecordFields {
    ProfileRecordFields {
        profile_id: record.profile_id,
        community_id: record.community_id,
        author_public_key: record.public_key,
        kind: ProfileKind::Agent(AgentProfile {
            claimed_owner: None,
            owner_attestation: None,
        }),
        metadata: None,
        statuses: Vec::new(),
        social_lists: Vec::new(),
        relay_archive_states: Vec::new(),
        version: record.version,
    }
}

fn owner_evidence(
    public_key: NostrPublicKey,
    attestation: &OwnerAttestation,
    source: NostrEventId,
    verified_at: u64,
) -> OwnerAttestationEvidence {
    OwnerAttestationEvidence {
        owner_public_key: NostrPublicKey::from_bytes(*attestation.owner.as_bytes()),
        agent_public_key: public_key,
        proof_event_id: source,
        exact_conditions: attestation.conditions.as_str().to_owned(),
        verified_at,
    }
}

fn profile_metadata(
    event: &CanonicalEvent,
) -> Result<ProfileMetadata, IdentityProfileRepositoryError> {
    let metadata: serde_json::Value = serde_json::from_str(&event.content)
        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
    let text = |field: &str, fallback: Option<&str>| {
        metadata
            .get(field)
            .or_else(|| fallback.and_then(|field| metadata.get(field)))
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    Ok(ProfileMetadata {
        display_name: text("display_name", Some("name")),
        name: text("name", None),
        avatar_url: text("picture", Some("image")),
        about: text("about", None),
        nip05_handle: text("nip05", None),
    })
}

fn optional_profile_owner_attestation(event: &CanonicalEvent) -> Option<OwnerAttestation> {
    match verified_owner_attestation(event) {
        Ok(proof) => proof,
        Err(_) => {
            tracing::warn!(
                reason = "optional_profile_owner_evidence_invalid",
                "profile metadata has no verified optional owner authority"
            );
            None
        }
    }
}

fn verified_owner_attestation(
    event: &CanonicalEvent,
) -> Result<Option<OwnerAttestation>, IdentityProfileRepositoryError> {
    let mut tags = event
        .tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some("auth"));
    let Some(tag) = tags.next() else {
        return Ok(None);
    };
    if tags.next().is_some() {
        return Err(IdentityProfileRepositoryError::InvalidRecord);
    }
    let attestation = OwnerAttestation::parse_tag(tag)
        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
    attestation
        .verify_for_event(event)
        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
    Ok(Some(attestation))
}

async fn ensure_transaction_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), IdentityProfileRepositoryError> {
    if transaction.get_database_backend() != DatabaseBackend::Postgres {
        return Err(IdentityProfileRepositoryError::UnsupportedDatabase);
    }
    let setting = transaction
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT NULLIF(current_setting('app.community_id', true), '') AS community_id"
                .to_owned(),
        ))
        .await?
        .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
    let configured: Option<String> = setting.try_get("", "community_id")?;
    if configured.is_some_and(|configured| configured != tenant.community_id().to_string()) {
        return Err(IdentityProfileRepositoryError::TenantBoundaryViolation);
    }
    set_tenant(transaction, tenant).await
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), IdentityProfileRepositoryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id', $1, true)",
            [tenant.community_id().to_string().into()],
        ))
        .await?;
    Ok(())
}

async fn lock_identity(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    public_key: NostrPublicKey,
) -> Result<(), IdentityProfileRepositoryError> {
    set_tenant(transaction, tenant).await?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            [format!(
                "collaboration-profile:{}:{}",
                tenant.community_id(),
                hex::encode(public_key.as_bytes())
            )
            .into()],
        ))
        .await?;
    Ok(())
}

async fn select_profile(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    public_key: NostrPublicKey,
    for_update: bool,
) -> Result<Option<RegisteredIdentity>, IdentityProfileRepositoryError> {
    let sql = if for_update {
        format!("{SELECT_PROFILE} FOR UPDATE")
    } else {
        SELECT_PROFILE.to_owned()
    };
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            sql,
            [
                tenant.community_id().as_uuid().into(),
                public_key.as_bytes().to_vec().into(),
            ],
        ))
        .await?
        .map(|row| hydrate_profile(row, tenant, public_key))
        .transpose()
}

fn hydrate_profile(
    row: QueryResult,
    tenant: &TenantContext,
    public_key: NostrPublicKey,
) -> Result<RegisteredIdentity, IdentityProfileRepositoryError> {
    let profile: Option<IdentityProfile> = row
        .try_get::<Option<serde_json::Value>>("", "profile_payload")?
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
    let principal_id: Uuid = row.try_get("", "principal_id")?;
    let profile_id: Uuid = row.try_get("", "profile_id")?;
    let version: String = row.try_get("", "aggregate_version")?;
    let version = AggregateVersion::new(
        version
            .parse()
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
    )
    .ok_or(IdentityProfileRepositoryError::InvalidRecord)?;
    let mut record = RegisteredIdentity {
        community_id: CommunityId::from_uuid(row.try_get("", "community_id")?),
        profile_id: ProfileId::from_uuid(profile_id),
        principal_id: PrincipalId::from_uuid(principal_id),
        public_key,
        version,
        profile,
        first_owner_proof_event_id: row
            .try_get::<Option<Vec<u8>>>("", "first_owner_proof_event_id")?
            .map(bytes_id)
            .transpose()?,
        source_event_id: row
            .try_get::<Option<Vec<u8>>>("", "source_event_id")?
            .map(bytes_id)
            .transpose()?,
        first_owner_auth_proof: row
            .try_get::<Option<serde_json::Value>>("", "first_owner_auth_proof")?
            .map(serde_json::from_value)
            .transpose()
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
        latest_owner_auth_proof: row
            .try_get::<Option<serde_json::Value>>("", "latest_owner_auth_proof")?
            .map(serde_json::from_value)
            .transpose()
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
        registration_source_event_id: row
            .try_get::<Option<Vec<u8>>>("", "registration_source_event_id")?
            .map(bytes_id)
            .transpose()?,
        registration_verified_at_seconds: row
            .try_get::<Option<String>>("", "registration_verified_at_seconds")?
            .map(|value| value.parse())
            .transpose()
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
        observation: row
            .try_get::<Option<serde_json::Value>>("", "observation_proof")?
            .map(serde_json::from_value)
            .transpose()
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
        accepted_name: None,
    };
    let name: Option<String> = row.try_get("", "accepted_nip05_name")?;
    let accepted_source = row
        .try_get::<Option<Vec<u8>>>("", "accepted_nip05_source_event_id")?
        .map(bytes_id)
        .transpose()?;
    let processed = row
        .try_get::<Option<Vec<u8>>>("", "nip05_processed_event_id")?
        .map(bytes_id)
        .transpose()?;
    let host: Option<String> = row.try_get("", "nip05_projection_host")?;
    record.accepted_name = match (processed, host) {
        (None, None) if name.is_none() && accepted_source.is_none() => None,
        (Some(processed_event_id), Some(host))
            if name.is_some() == accepted_source.is_some() && !host.is_empty() =>
        {
            Some(AcceptedNameProjection {
                name,
                source_event_id: accepted_source,
                processed_event_id,
                host,
            })
        }
        _ => return Err(IdentityProfileRepositoryError::InvalidRecord),
    };
    let observed_event = row
        .try_get::<Option<Vec<u8>>>("", "observation_source_event_id")?
        .map(bytes_id)
        .transpose()?;
    let observed_at = row
        .try_get::<Option<String>>("", "observation_verified_at_millis")?
        .map(|value| value.parse::<u64>())
        .transpose()
        .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
    let observed_operation = row
        .try_get::<Option<Uuid>>("", "observation_moderation_operation_id")?
        .map(collaboration_domain::OperationId::from_uuid);
    if observed_operation
        != record
            .observation
            .as_ref()
            .and_then(ObservationEvidence::moderation_operation_id)
        || observed_event
            != record
                .observation
                .as_ref()
                .and_then(ObservationEvidence::retained_event_id)
        || observed_at
            != record
                .observation
                .as_ref()
                .map(ObservationEvidence::verified_at_millis)
                .transpose()?
    {
        return Err(IdentityProfileRepositoryError::InvalidRecord);
    }
    let stored_author: Vec<u8> = row.try_get("", "author_public_key")?;
    let kind: Option<String> = row.try_get("", "profile_kind")?;
    let owner: Option<Vec<u8>> = row.try_get("", "owner_public_key")?;
    if record.community_id != tenant.community_id()
        || principal_id.is_nil()
        || profile_id.is_nil()
        || stored_author.as_slice() != public_key.as_bytes()
        || record.registration_source_event_id.is_some()
            != record.registration_verified_at_seconds.is_some()
    {
        return Err(IdentityProfileRepositoryError::InvalidRecord);
    }
    match record.profile.as_ref() {
        Some(profile) => {
            let ProfileKind::Agent(agent) = profile.kind() else {
                return Err(IdentityProfileRepositoryError::InvalidRecord);
            };
            if kind.as_deref() != Some("agent")
                || profile.community_id() != record.community_id
                || profile.profile_id() != record.profile_id
                || profile.author_public_key() != record.public_key
                || profile.version() != record.version
                || agent
                    .claimed_owner
                    .is_none_or(|key| Some(key.as_bytes().to_vec()) != owner)
                || record.first_owner_proof_event_id.is_some()
                    == record.first_owner_auth_proof.is_some()
                || profile
                    .fields()
                    .metadata
                    .as_ref()
                    .map(|metadata| metadata.source_event_id)
                    != record.source_event_id
            {
                return Err(IdentityProfileRepositoryError::InvalidRecord);
            }
        }
        None => {
            if kind.is_some()
                || owner.is_some()
                || record.first_owner_proof_event_id.is_some()
                || record.first_owner_auth_proof.is_some()
                || record.latest_owner_auth_proof.is_some()
                || record.source_event_id.is_some()
                || (record.registration_source_event_id.is_none() && record.observation.is_none())
            {
                return Err(IdentityProfileRepositoryError::InvalidRecord);
            }
        }
    }
    let hash: Vec<u8> = row.try_get("", "profile_hash")?;
    let expected = match row.try_get::<i16>("", "projection_hash_version")? {
        1 if record.accepted_name.is_none() => {
            legacy_profile_hash(&record)?.ok_or(IdentityProfileRepositoryError::InvalidRecord)?
        }
        2 if record.observation.is_none() && record.accepted_name.is_none() => {
            registration_hash(&record)?
        }
        3 if record.observation.is_some() && record.accepted_name.is_none() => {
            observation_hash(&record)?
        }
        4 if record.accepted_name.is_some() => accepted_name_hash(&record)?,
        _ => return Err(IdentityProfileRepositoryError::InvalidRecord),
    };
    if expected != hash {
        return Err(IdentityProfileRepositoryError::InvalidRecord);
    }
    Ok(record)
}

fn bytes_id(bytes: Vec<u8>) -> Result<NostrEventId, IdentityProfileRepositoryError> {
    Ok(NostrEventId::from_bytes(bytes.try_into().map_err(
        |_| IdentityProfileRepositoryError::InvalidRecord,
    )?))
}

async fn write_profile(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
    record: &RegisteredIdentity,
) -> Result<(), IdentityProfileRepositoryError> {
    let (kind, owner) = match record.profile.as_ref().map(IdentityProfile::kind) {
        Some(ProfileKind::Agent(agent)) => (
            Some("agent".to_owned()),
            agent.claimed_owner.map(|key| key.as_bytes().to_vec()),
        ),
        Some(ProfileKind::Human) => return Err(IdentityProfileRepositoryError::InvalidRecord),
        None => (None, None),
    };
    let json = |value| {
        serde_json::to_value(value).map_err(|_| IdentityProfileRepositoryError::InvalidRecord)
    };
    let result = transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres, r#"
INSERT INTO public.collaboration_identity_profiles (
    community_id, profile_id, principal_id, author_public_key, profile_kind,
    owner_public_key, first_owner_proof_event_id, source_event_id, aggregate_version,
    profile_payload, profile_hash, first_owner_auth_proof, latest_owner_auth_proof,
    registration_source_event_id, registration_verified_at_seconds, projection_hash_version,
    legacy_profile_hash, observation_proof, observation_source_event_id, observation_verified_at_millis, observation_moderation_operation_id,
    accepted_nip05_name, accepted_nip05_source_event_id, nip05_processed_event_id, nip05_projection_host
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,CAST($9 AS numeric),$10,$11,$12,$13,$14,CAST($15 AS numeric),$17,$16,$18,$19,CAST($20 AS numeric),$21,$22,$23,$24,$25)
ON CONFLICT (community_id, author_public_key) DO UPDATE SET
    profile_kind = EXCLUDED.profile_kind, owner_public_key = EXCLUDED.owner_public_key,
    first_owner_proof_event_id = EXCLUDED.first_owner_proof_event_id,
    source_event_id = EXCLUDED.source_event_id, aggregate_version = EXCLUDED.aggregate_version,
    profile_payload = EXCLUDED.profile_payload, profile_hash = EXCLUDED.profile_hash,
    first_owner_auth_proof = EXCLUDED.first_owner_auth_proof,
    latest_owner_auth_proof = EXCLUDED.latest_owner_auth_proof,
    registration_source_event_id = EXCLUDED.registration_source_event_id,
    registration_verified_at_seconds = EXCLUDED.registration_verified_at_seconds,
    projection_hash_version = EXCLUDED.projection_hash_version,
    legacy_profile_hash = EXCLUDED.legacy_profile_hash,
    observation_proof = EXCLUDED.observation_proof,
    observation_source_event_id = EXCLUDED.observation_source_event_id,
    observation_verified_at_millis = EXCLUDED.observation_verified_at_millis,
    observation_moderation_operation_id = EXCLUDED.observation_moderation_operation_id,
    accepted_nip05_name = EXCLUDED.accepted_nip05_name,
    accepted_nip05_source_event_id = EXCLUDED.accepted_nip05_source_event_id,
    nip05_processed_event_id = EXCLUDED.nip05_processed_event_id,
    nip05_projection_host = EXCLUDED.nip05_projection_host,
    updated_at = clock_timestamp()
WHERE collaboration_identity_profiles.profile_id = EXCLUDED.profile_id
  AND collaboration_identity_profiles.principal_id = EXCLUDED.principal_id
  AND collaboration_identity_profiles.aggregate_version + 1 = EXCLUDED.aggregate_version
"#, [
        tenant.community_id().as_uuid().into(), record.profile_id.as_uuid().into(), record.principal_id.as_uuid().into(),
        record.public_key.as_bytes().to_vec().into(), kind.into(), owner.into(),
        record.first_owner_proof_event_id.map(|id| id.as_bytes().to_vec()).into(), record.source_event_id.map(|id| id.as_bytes().to_vec()).into(),
        record.version.get().to_string().into(), record.profile.as_ref().map(json).transpose()?.into(), if record.accepted_name.is_some() { accepted_name_hash(record)? } else if record.observation.is_some() { observation_hash(record)? } else { registration_hash(record)? }.into(),
        record.first_owner_auth_proof.as_ref().map(|value| serde_json::to_value(value)).transpose().map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?.into(),
        record.latest_owner_auth_proof.as_ref().map(|value| serde_json::to_value(value)).transpose().map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?.into(),
        record.registration_source_event_id.map(|id| id.as_bytes().to_vec()).into(),
        record.registration_verified_at_seconds.map(|seconds| seconds.to_string()).into(), legacy_profile_hash(record)?.into(),
        if record.accepted_name.is_some() { 4_i16 } else if record.observation.is_some() { 3_i16 } else { 2_i16 }.into(),
        record.observation.as_ref().map(serde_json::to_value).transpose().map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?.into(),
        record.observation.as_ref().and_then(ObservationEvidence::retained_event_id).map(|id| id.as_bytes().to_vec()).into(),
        record.observation.as_ref().map(ObservationEvidence::verified_at_millis).transpose()?.map(|value| value.to_string()).into(),
        record.observation.as_ref().and_then(ObservationEvidence::moderation_operation_id).map(|id| id.as_uuid()).into(),
        record.accepted_name.as_ref().and_then(|accepted| accepted.name.clone()).into(),
        record.accepted_name.as_ref().and_then(|accepted| accepted.source_event_id).map(|id| id.as_bytes().to_vec()).into(),
        record.accepted_name.as_ref().map(|accepted| accepted.processed_event_id.as_bytes().to_vec()).into(),
        record.accepted_name.as_ref().map(|accepted| accepted.host.clone()).into(),
    ])).await?;
    if result.rows_affected() != 1 {
        return Err(IdentityProfileRepositoryError::IdentityConflict);
    }
    Ok(())
}

fn accepted_name_hash(
    record: &RegisteredIdentity,
) -> Result<Vec<u8>, IdentityProfileRepositoryError> {
    Ok(Sha256::digest(
        serde_json::to_vec(&("zed.collaboration.identity.accepted-name.v4", record))
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
    )
    .to_vec())
}

fn registration_hash(
    record: &RegisteredIdentity,
) -> Result<Vec<u8>, IdentityProfileRepositoryError> {
    Ok(Sha256::digest(
        serde_json::to_vec(record).map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
    )
    .to_vec())
}

fn observation_hash(
    record: &RegisteredIdentity,
) -> Result<Vec<u8>, IdentityProfileRepositoryError> {
    Ok(Sha256::digest(
        serde_json::to_vec(&("zed.collaboration.identity.observation.v3", record))
            .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?,
    )
    .to_vec())
}

fn legacy_profile_hash(
    record: &RegisteredIdentity,
) -> Result<Option<Vec<u8>>, IdentityProfileRepositoryError> {
    if record.accepted_name.is_some()
        || record.observation.is_some()
        || record.first_owner_auth_proof.is_some()
        || record.latest_owner_auth_proof.is_some()
        || record.registration_source_event_id.is_some()
    {
        return Ok(None);
    }
    let (Some(profile), Some(first), Some(source)) = (
        &record.profile,
        record.first_owner_proof_event_id,
        record.source_event_id,
    ) else {
        return Ok(None);
    };
    let encoded = serde_json::to_vec(&(
        record.principal_id.as_uuid(),
        profile,
        first.as_bytes(),
        source.as_bytes(),
    ))
    .map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
    Ok(Some(Sha256::digest(encoded).to_vec()))
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, IdentityProfileRepositoryError>,
) -> Result<T, IdentityProfileRepositoryError> {
    match result {
        Ok(result) => {
            transaction.commit().await?;
            Ok(result)
        }
        Err(error) => {
            transaction.rollback().await?;
            Err(error)
        }
    }
}

#[async_trait]
impl crate::collaboration_command::DomainCommandSink<crate::nostr::event_ingest::NostrEventCommand>
    for IdentityProfileRepository
{
    async fn submit(
        &self,
        command: crate::collaboration_command::DomainCommand<
            crate::nostr::event_ingest::NostrEventCommand,
        >,
    ) -> Result<
        crate::collaboration_command::DomainCommandReceipt,
        crate::collaboration_command::DomainCommandSubmissionError,
    > {
        use crate::collaboration_command::{DomainCommandReceipt, DomainCommandSubmissionError};
        let rejected = || DomainCommandSubmissionError::Rejected;
        let unavailable = || DomainCommandSubmissionError::Unavailable;
        let signed = command.payload().signed_event();
        if u32::from(signed.event.kind) != KIND_PROFILE
            || command.contract_version()
                != crate::collaboration_command::CURRENT_COMMAND_CONTRACT_VERSION
            || command.expected_version().is_some()
            || command.predecessor().is_some()
            || command.operation_id()
                != crate::nostr::event_ingest::operation_id(
                    command.tenant().community_id().as_uuid(),
                    signed.claimed_id,
                )
            || !crate::nostr::event_ingest::principal_matches_author(
                command.principal().kind(),
                signed.event.public_key,
            )
            || signed.event.content.len() > 256 * 1024
        {
            return Err(rejected());
        }
        let now_millis = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| unavailable())?
                .as_millis(),
        )
        .map_err(|_| unavailable())?;
        let record = VerifiedEventRecord::new(
            command.tenant().community_id(),
            signed.clone(),
            crate::db::collaboration::event_repository::EventVerificationState::Live,
            now_millis,
            TimestampPolicy::Bounded {
                now: now_millis / 1000,
                max_past_seconds: 900,
                max_future_seconds: 900,
            },
        )
        .map_err(|_| rejected())?;
        profile_metadata(&signed.event)
            .map_err(|_| rejected())?
            .validate()
            .map_err(|_| rejected())?;
        let transaction = self
            .connection
            .begin_with_config(Some(sea_orm::IsolationLevel::ReadCommitted), None)
            .await
            .map_err(|_| unavailable())?;
        let result = async {
            self.admit_profile_in_transaction(
                &transaction,
                command.tenant(),
                command.principal(),
                collaboration_domain::AuthorizationAction::Write,
                now_millis,
            )
            .await
            .map_err(|error| match error {
                NostrHttpDirectoryError::Denied => rejected(),
                NostrHttpDirectoryError::Unavailable => unavailable(),
            })?;
            let duplicate = self
                .events
                .exact_in_transaction(&transaction, command.tenant(), signed.claimed_id)
                .await
                .map_err(|_| unavailable())?
                .is_some();
            self.project_profile_with_owner_policy_in_transaction(
                &transaction,
                command.tenant(),
                &record,
                ProfileOwnerEvidencePolicy::OptionalAuthoredMetadata,
            )
            .await
            .map_err(|error| match error {
                IdentityProfileRepositoryError::IdentityConflict
                | IdentityProfileRepositoryError::InvalidRecord => rejected(),
                _ => unavailable(),
            })?;
            self.project_admitted_nip05_in_transaction(&transaction, command.tenant(), &record)
                .await
                .map_err(|error| match error {
                    IdentityProfileRepositoryError::IdentityConflict
                    | IdentityProfileRepositoryError::InvalidRecord => rejected(),
                    _ => unavailable(),
                })?;
            let subject = self
                .current_subject_in_transaction(
                    &transaction,
                    command.tenant(),
                    NostrPublicKey::from_bytes(*signed.event.public_key.as_bytes()),
                )
                .await
                .map_err(|_| unavailable())?
                .ok_or_else(unavailable)?;
            Ok(if duplicate {
                DomainCommandReceipt::duplicate(command.operation_id(), subject.version())
            } else {
                DomainCommandReceipt::new(command.operation_id(), subject.version())
            })
        }
        .await;
        match result {
            Ok(receipt) => {
                transaction.commit().await.map_err(|_| unavailable())?;
                Ok(receipt)
            }
            Err(error) => {
                transaction.rollback().await.map_err(|_| unavailable())?;
                Err(error)
            }
        }
    }
}

pub(crate) fn validate_channel_add_policy(
    content: &str,
) -> Result<collaboration_domain::membership::AgentChannelAddPolicy, IdentityProfileRepositoryError>
{
    use collaboration_domain::membership::AgentChannelAddPolicy;
    let content: serde_json::Value =
        serde_json::from_str(content).map_err(|_| IdentityProfileRepositoryError::InvalidRecord)?;
    match content
        .get("channel_add_policy")
        .and_then(serde_json::Value::as_str)
    {
        Some("anyone") => Ok(AgentChannelAddPolicy::Anyone),
        Some("owner_only") => Ok(AgentChannelAddPolicy::OwnerOnly),
        Some("nobody") => Ok(AgentChannelAddPolicy::Nobody),
        _ => Err(IdentityProfileRepositoryError::InvalidRecord),
    }
}
