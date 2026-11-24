use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    AuthenticatedPrincipal, AuthenticatedPrincipalKind, CommunityId, NostrAuthenticationMethod,
    NostrPublicKey, TenantContext,
};
use nostr_compat::{
    EventId, SignedEvent, TimestampPolicy,
    buzz_nips::identity::{AgentAuthentication, OwnerAttestation},
    generated_kinds::KIND_AUTH,
    verify_signed_event,
};
use rand::random;
use url::Url;

pub const NIP42_AUTH_TIMEOUT_SECONDS: u64 = 5;
pub const NIP42_EVENT_FRESHNESS_SECONDS: u64 = 60;

#[derive(Clone, Eq, PartialEq)]
pub struct NostrAuthChallenge(String);

impl NostrAuthChallenge {
    pub fn generate() -> Self {
        Self(hex::encode(random::<[u8; 32]>()))
    }

    pub fn parse(value: impl Into<String>) -> Result<Self, NostrAuthenticationError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(NostrAuthenticationError::InvalidChallenge);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for NostrAuthChallenge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("NostrAuthChallenge([redacted])")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedNip42OwnerAttestation {
    attestation: OwnerAttestation,
}

impl VerifiedNip42OwnerAttestation {
    pub const fn owner_public_key(&self) -> NostrPublicKey {
        NostrPublicKey::from_bytes(*self.attestation.owner.as_bytes())
    }

    pub fn exact_conditions(&self) -> &str {
        self.attestation.conditions.as_str()
    }

    pub const fn attestation(&self) -> &OwnerAttestation {
        &self.attestation
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct VerifiedNip42Identity {
    community_id: CommunityId,
    signed_authentication_event: SignedEvent,
    verified_at_seconds: u64,
    owner_attestation: Option<VerifiedNip42OwnerAttestation>,
}

impl VerifiedNip42Identity {
    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn public_key(&self) -> NostrPublicKey {
        NostrPublicKey::from_bytes(*self.signed_authentication_event.event.public_key.as_bytes())
    }

    pub const fn event_id(&self) -> EventId {
        self.signed_authentication_event.claimed_id
    }

    pub const fn signed_authentication_event(&self) -> &SignedEvent {
        &self.signed_authentication_event
    }

    pub const fn verified_at_seconds(&self) -> u64 {
        self.verified_at_seconds
    }

    pub const fn owner_attestation(&self) -> Option<&VerifiedNip42OwnerAttestation> {
        self.owner_attestation.as_ref()
    }

    pub fn for_tenant(
        &self,
        tenant: &TenantContext,
    ) -> Result<&Self, NostrPrincipalResolutionError> {
        if self.community_id != tenant.community_id() {
            return Err(NostrPrincipalResolutionError::Denied);
        }
        Ok(self)
    }
}

impl fmt::Debug for VerifiedNip42Identity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedNip42Identity")
            .field("community_id", &self.community_id)
            .field("public_key", &self.public_key())
            .field("event_id", &self.event_id())
            .field("verified_at_seconds", &self.verified_at_seconds)
            .field("owner_attestation", &self.owner_attestation)
            .field("signed_authentication_event", &"[redacted]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayClaim {
    Claimed,
    Replay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrAuthenticationInfrastructureError {
    Unavailable,
}

#[async_trait]
pub trait NostrAuthReplayStore: Send + Sync {
    async fn claim(
        &self,
        community_id: CommunityId,
        event_id: EventId,
        expires_at_seconds: u64,
    ) -> Result<ReplayClaim, NostrAuthenticationInfrastructureError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NostrPrincipalResolutionError {
    #[error("Nostr principal admission denied")]
    Denied,
    #[error("Nostr principal is banned")]
    Banned,
    #[error("Nostr subject is not represented")]
    UnrepresentedSubject,
    #[error("Nostr principal resolution is unavailable")]
    Unavailable,
}

#[async_trait]
pub trait NostrPrincipalResolver: Send + Sync {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<AuthenticatedPrincipal, NostrPrincipalResolutionError>;

    async fn validate_current_in_transaction(
        &self,
        _transaction: &sea_orm::DatabaseTransaction,
        _tenant: &TenantContext,
        _identity: &VerifiedNip42Identity,
        _principal: &AuthenticatedPrincipal,
        _now_seconds: u64,
    ) -> Result<(), NostrPrincipalResolutionError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NostrAuthenticationState {
    Pending {
        challenge: NostrAuthChallenge,
        deadline_seconds: u64,
    },
    Authenticated(AuthenticatedPrincipal),
    Failed,
    TimedOut,
}

pub struct NostrConnectionAuthenticator<R, P> {
    tenant: TenantContext,
    relay_url: String,
    replay_store: R,
    principal_resolver: P,
    state: NostrAuthenticationState,
}

impl<R, P> NostrConnectionAuthenticator<R, P>
where
    R: NostrAuthReplayStore,
    P: NostrPrincipalResolver,
{
    pub fn new(
        tenant: TenantContext,
        relay_url: impl Into<String>,
        challenge: NostrAuthChallenge,
        issued_at_seconds: u64,
        replay_store: R,
        principal_resolver: P,
    ) -> Result<Self, NostrAuthenticationError> {
        let relay_url = relay_url.into();
        if normalize_relay_url(&relay_url).is_none() {
            return Err(NostrAuthenticationError::InvalidRelayConfiguration);
        }
        Ok(Self {
            tenant,
            relay_url,
            replay_store,
            principal_resolver,
            state: NostrAuthenticationState::Pending {
                challenge,
                deadline_seconds: issued_at_seconds.saturating_add(NIP42_AUTH_TIMEOUT_SECONDS),
            },
        })
    }

    pub const fn state(&self) -> &NostrAuthenticationState {
        &self.state
    }

    pub fn challenge(&self) -> Option<&NostrAuthChallenge> {
        let NostrAuthenticationState::Pending { challenge, .. } = &self.state else {
            return None;
        };
        Some(challenge)
    }

    pub fn challenge_frame(&self) -> Option<String> {
        self.challenge()
            .map(|challenge| serde_json::json!(["AUTH", challenge.as_str()]).to_string())
    }

    pub fn expire(&mut self, now_seconds: u64) -> bool {
        let NostrAuthenticationState::Pending {
            deadline_seconds, ..
        } = &self.state
        else {
            return false;
        };
        if now_seconds < *deadline_seconds {
            return false;
        }
        self.state = NostrAuthenticationState::TimedOut;
        true
    }

    pub async fn authenticate(
        &mut self,
        signed_event: &SignedEvent,
        now_seconds: u64,
    ) -> Result<&AuthenticatedPrincipal, NostrAuthenticationError> {
        let (challenge, deadline_seconds) = match &self.state {
            NostrAuthenticationState::Pending {
                challenge,
                deadline_seconds,
            } => (challenge.clone(), *deadline_seconds),
            NostrAuthenticationState::Authenticated(_) => {
                return Err(NostrAuthenticationError::AlreadyAuthenticated);
            }
            NostrAuthenticationState::Failed => {
                return Err(NostrAuthenticationError::AuthenticationAlreadyFailed);
            }
            NostrAuthenticationState::TimedOut => {
                return Err(NostrAuthenticationError::TimedOut);
            }
        };
        if self.expire(now_seconds) {
            return Err(NostrAuthenticationError::TimedOut);
        }

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(deadline_seconds.saturating_sub(now_seconds)),
            self.verify_and_resolve(signed_event, &challenge, now_seconds),
        )
        .await;
        let result = match result {
            Ok(result) => result,
            Err(_) => {
                self.state = NostrAuthenticationState::TimedOut;
                return Err(NostrAuthenticationError::TimedOut);
            }
        };
        match result {
            Ok(principal) => {
                self.state = NostrAuthenticationState::Authenticated(principal);
                match &self.state {
                    NostrAuthenticationState::Authenticated(principal) => Ok(principal),
                    _ => Err(NostrAuthenticationError::InfrastructureUnavailable),
                }
            }
            Err(error) => {
                self.state = NostrAuthenticationState::Failed;
                Err(error)
            }
        }
    }

    async fn verify_and_resolve(
        &self,
        signed_event: &SignedEvent,
        challenge: &NostrAuthChallenge,
        now_seconds: u64,
    ) -> Result<AuthenticatedPrincipal, NostrAuthenticationError> {
        if u32::from(signed_event.event.kind) != KIND_AUTH {
            return Err(NostrAuthenticationError::InvalidEvent);
        }
        verify_signed_event(
            signed_event,
            TimestampPolicy::Bounded {
                now: now_seconds,
                max_past_seconds: NIP42_EVENT_FRESHNESS_SECONDS,
                max_future_seconds: NIP42_EVENT_FRESHNESS_SECONDS,
            },
        )
        .map_err(|_| NostrAuthenticationError::InvalidEvent)?;

        let presented_challenge = single_text_tag(&signed_event.event.tags, "challenge")?;
        if presented_challenge != challenge.as_str() {
            return Err(NostrAuthenticationError::ChallengeMismatch);
        }
        let presented_relay = single_text_tag(&signed_event.event.tags, "relay")?;
        if normalize_relay_url(presented_relay) != normalize_relay_url(&self.relay_url) {
            return Err(NostrAuthenticationError::RelayMismatch);
        }

        let owner_attestation = if signed_event
            .event
            .tags
            .iter()
            .any(|tag| tag.first().map(String::as_str) == Some("auth"))
        {
            match AgentAuthentication::parse_event(&signed_event.event) {
                Ok(authentication) => {
                    match authentication.verify_attestation(&signed_event.event) {
                        Ok(()) => Some(VerifiedNip42OwnerAttestation {
                            attestation: authentication.attestation,
                        }),
                        Err(_) => {
                            tracing::warn!(
                                reason = "unverified_optional_owner_attestation",
                                "NIP-42 key authentication carries no verified owner authority"
                            );
                            None
                        }
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        reason = "malformed_optional_owner_attestation",
                        "NIP-42 key authentication carries no verified owner authority"
                    );
                    None
                }
            }
        } else {
            None
        };
        let identity = VerifiedNip42Identity {
            community_id: self.tenant.community_id(),
            signed_authentication_event: signed_event.clone(),
            verified_at_seconds: now_seconds,
            owner_attestation,
        };

        match self
            .replay_store
            .claim(
                self.tenant.community_id(),
                signed_event.claimed_id,
                now_seconds
                    .saturating_add(NIP42_EVENT_FRESHNESS_SECONDS)
                    .max(
                        signed_event
                            .event
                            .created_at
                            .saturating_add(NIP42_EVENT_FRESHNESS_SECONDS),
                    ),
            )
            .await
            .map_err(|_| NostrAuthenticationError::InfrastructureUnavailable)?
        {
            ReplayClaim::Claimed => {}
            ReplayClaim::Replay => return Err(NostrAuthenticationError::Replay),
        }

        let principal = self
            .principal_resolver
            .resolve(&self.tenant, &identity)
            .await
            .map_err(|error| match error {
                NostrPrincipalResolutionError::Denied => NostrAuthenticationError::IdentityDenied,
                NostrPrincipalResolutionError::Banned => NostrAuthenticationError::Banned,
                NostrPrincipalResolutionError::UnrepresentedSubject => {
                    NostrAuthenticationError::UnrepresentedSubject
                }
                NostrPrincipalResolutionError::Unavailable => {
                    NostrAuthenticationError::InfrastructureUnavailable
                }
            })?;
        if principal.community_id() != self.tenant.community_id() {
            return Err(NostrAuthenticationError::IdentityDenied);
        }
        if !principal_matches_identity(&principal, &identity) {
            return Err(NostrAuthenticationError::IdentityDenied);
        }
        Ok(principal)
    }
}

fn single_text_tag<'a>(
    tags: &'a [Vec<String>],
    name: &'static str,
) -> Result<&'a str, NostrAuthenticationError> {
    let mut matches = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some(name));
    let Some(tag) = matches.next() else {
        return Err(NostrAuthenticationError::InvalidEvent);
    };
    if matches.next().is_some() || tag.len() != 2 || tag[1].is_empty() {
        return Err(NostrAuthenticationError::InvalidEvent);
    }
    Ok(&tag[1])
}

fn normalize_relay_url(raw: &str) -> Option<String> {
    let mut parsed = Url::parse(raw).ok()?;
    if !matches!(parsed.scheme(), "ws" | "wss")
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
    {
        return None;
    }
    if matches!(parsed.host_str(), Some("localhost" | "::1"))
        && parsed.set_host(Some("127.0.0.1")).is_err()
    {
        return None;
    }
    let path = parsed.path().trim_end_matches('/').to_owned();
    parsed.set_path(&path);
    Some(parsed.to_string())
}

fn principal_matches_identity(
    principal: &AuthenticatedPrincipal,
    identity: &VerifiedNip42Identity,
) -> bool {
    match (principal.kind(), identity.owner_attestation()) {
        (
            AuthenticatedPrincipalKind::NostrIdentity {
                public_key,
                authentication_method,
                ..
            },
            _,
        ) => {
            *public_key == identity.public_key()
                && *authentication_method == NostrAuthenticationMethod::Nip42
        }
        (
            AuthenticatedPrincipalKind::OwnerAttestedAgent {
                agent_public_key,
                owner_public_key,
                authentication_method,
                ..
            },
            Some(attestation),
        ) => {
            *agent_public_key == identity.public_key()
                && *owner_public_key == attestation.owner_public_key()
                && *authentication_method == NostrAuthenticationMethod::Nip42
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrAuthenticationError {
    InvalidChallenge,
    InvalidRelayConfiguration,
    InvalidEvent,
    ChallengeMismatch,
    RelayMismatch,
    InvalidOwnerAttestation,
    Replay,
    IdentityDenied,
    Banned,
    UnrepresentedSubject,
    InfrastructureUnavailable,
    TimedOut,
    AlreadyAuthenticated,
    AuthenticationAlreadyFailed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrAuthProtocolDisposition {
    Reject {
        reason: &'static str,
        close_connection: bool,
    },
    CloseWithoutResponse,
}

impl NostrAuthenticationError {
    pub const fn protocol_disposition(self) -> NostrAuthProtocolDisposition {
        match self {
            Self::TimedOut => NostrAuthProtocolDisposition::CloseWithoutResponse,
            Self::AlreadyAuthenticated => NostrAuthProtocolDisposition::Reject {
                reason: "auth-required: already authenticated",
                close_connection: false,
            },
            Self::AuthenticationAlreadyFailed => NostrAuthProtocolDisposition::Reject {
                reason: "auth-required: authentication already failed",
                close_connection: false,
            },
            Self::Banned => NostrAuthProtocolDisposition::Reject {
                reason: "blocked: you are banned from this community",
                close_connection: true,
            },
            Self::UnrepresentedSubject => NostrAuthProtocolDisposition::Reject {
                reason: "error: restriction target is not represented",
                close_connection: true,
            },
            Self::IdentityDenied | Self::InvalidOwnerAttestation => {
                NostrAuthProtocolDisposition::Reject {
                    reason: "restricted: not a relay member",
                    close_connection: false,
                }
            }
            Self::InfrastructureUnavailable | Self::InvalidRelayConfiguration => {
                NostrAuthProtocolDisposition::Reject {
                    reason: "error: internal",
                    close_connection: true,
                }
            }
            Self::InvalidChallenge
            | Self::InvalidEvent
            | Self::ChallengeMismatch
            | Self::RelayMismatch
            | Self::Replay => NostrAuthProtocolDisposition::Reject {
                reason: "auth-required: verification failed",
                close_connection: false,
            },
        }
    }
}

impl fmt::Display for NostrAuthenticationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidChallenge => "NIP-42 challenge is invalid",
            Self::InvalidRelayConfiguration => "NIP-42 relay configuration is invalid",
            Self::InvalidEvent => "NIP-42 authentication event is invalid",
            Self::ChallengeMismatch => "NIP-42 challenge does not match",
            Self::RelayMismatch => "NIP-42 relay does not match",
            Self::InvalidOwnerAttestation => "NIP-42 owner attestation is invalid",
            Self::Replay => "NIP-42 authentication event was already used",
            Self::IdentityDenied => "NIP-42 identity is not authorized",
            Self::Banned => "NIP-42 identity is banned",
            Self::UnrepresentedSubject => "NIP-42 restriction target is not represented",
            Self::InfrastructureUnavailable => "NIP-42 authentication is unavailable",
            Self::TimedOut => "NIP-42 authentication timed out",
            Self::AlreadyAuthenticated => "NIP-42 connection is already authenticated",
            Self::AuthenticationAlreadyFailed => "NIP-42 authentication already failed",
        })
    }
}

impl Error for NostrAuthenticationError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NostrRestrictionVersionFence {
    pub principal_id: collaboration_domain::PrincipalId,
    pub profile_version: collaboration_domain::AggregateVersion,
    pub restriction_version: Option<collaboration_domain::AggregateVersion>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NostrStructuralBanSnapshot {
    pub community_id: CommunityId,
    pub verified_at_seconds: u64,
    pub identity: NostrRestrictionVersionFence,
    pub presented_owner: Option<NostrRestrictionVersionFence>,
}

pub struct NostrStructuralBanResolver<P> {
    connection: sea_orm::DatabaseConnection,
    profiles: crate::identity::profile_repository::IdentityProfileRepository,
    delegate: P,
}

impl<P: NostrPrincipalResolver> NostrStructuralBanResolver<P> {
    pub fn new(
        connection: sea_orm::DatabaseConnection,
        delegate: P,
    ) -> Result<Self, crate::identity::profile_repository::IdentityProfileRepositoryError> {
        if !matches!(
            &connection,
            sea_orm::DatabaseConnection::SqlxPostgresPoolConnection(_)
        ) {
            return Err(crate::identity::profile_repository::IdentityProfileRepositoryError::UnsupportedDatabase);
        }
        let profiles = crate::identity::profile_repository::IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(connection.get_postgres_connection_pool().clone()),
        )?;
        Ok(Self {
            connection,
            profiles,
            delegate,
        })
    }

    pub async fn check_restrictions(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<NostrStructuralBanSnapshot, NostrPrincipalResolutionError> {
        self.check_restrictions_at(tenant, identity, identity.verified_at_seconds(), None)
            .await
    }

    async fn check_restrictions_at(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
        current_seconds: u64,
        principal: Option<&AuthenticatedPrincipal>,
    ) -> Result<NostrStructuralBanSnapshot, NostrPrincipalResolutionError> {
        use sea_orm::{IsolationLevel, TransactionTrait as _};
        identity.for_tenant(tenant)?;
        let transaction = self
            .connection
            .begin_with_config(Some(IsolationLevel::ReadCommitted), None)
            .await
            .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
        let result = async {
            crate::admin::moderation::lock_moderation_in_transaction(&transaction, tenant)
                .await
                .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
            let subject = self
                .profiles
                .observe_verified_nip42_key_in_transaction(&transaction, tenant, identity)
                .await
                .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
            let identity_fence = self
                .check_key_in_transaction(&transaction, tenant, &subject, current_seconds)
                .await?;
            let owner_fence = match identity.owner_attestation() {
                Some(_) => {
                    let owner = self
                        .profiles
                        .observe_verified_nip42_presented_owner_in_transaction(
                            &transaction,
                            tenant,
                            identity,
                        )
                        .await
                        .map_err(|_| NostrPrincipalResolutionError::Unavailable)?
                        .ok_or(NostrPrincipalResolutionError::Unavailable)?;
                    Some(
                        self.check_key_in_transaction(
                            &transaction,
                            tenant,
                            &owner,
                            current_seconds,
                        )
                        .await?,
                    )
                }
                None => None,
            };
            if let Some(principal) = principal {
                // The final membership/profile decision shares the ban-reader lock and transaction, not a previously released delegate snapshot.
                self.delegate
                    .validate_current_in_transaction(
                        &transaction,
                        tenant,
                        identity,
                        principal,
                        current_seconds,
                    )
                    .await?;
            }
            Ok(NostrStructuralBanSnapshot {
                community_id: tenant.community_id(),
                verified_at_seconds: identity.verified_at_seconds(),
                identity: identity_fence,
                presented_owner: owner_fence,
            })
        }
        .await;
        match result {
            Ok(snapshot) => {
                transaction
                    .commit()
                    .await
                    .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
                Ok(snapshot)
            }
            Err(error) => {
                transaction
                    .rollback()
                    .await
                    .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
                Err(error)
            }
        }
    }

    async fn check_key_in_transaction(
        &self,
        transaction: &sea_orm::DatabaseTransaction,
        tenant: &TenantContext,
        subject: &crate::identity::profile_repository::ObservedIdentitySubject,
        verified_at_seconds: u64,
    ) -> Result<NostrRestrictionVersionFence, NostrPrincipalResolutionError> {
        let restriction = crate::admin::moderation::current_restriction_in_transaction(
            transaction,
            tenant,
            subject.principal_id(),
        )
        .await
        .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
        let now_millis = verified_at_seconds
            .checked_mul(1000)
            .ok_or(NostrPrincipalResolutionError::Unavailable)?;
        if restriction
            .as_ref()
            .is_some_and(|restriction| restriction.fields().ban.is_active_at(now_millis))
        {
            return Err(NostrPrincipalResolutionError::Banned);
        }
        Ok(NostrRestrictionVersionFence {
            principal_id: subject.principal_id(),
            profile_version: subject.version(),
            restriction_version: restriction.map(|restriction| restriction.fields().version),
        })
    }
}

#[async_trait]
impl<P: NostrPrincipalResolver> NostrPrincipalResolver for NostrStructuralBanResolver<P> {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<AuthenticatedPrincipal, NostrPrincipalResolutionError> {
        let started = std::time::Instant::now();
        self.check_restrictions(tenant, identity).await?;
        // The delegate owns its transaction, so retaining the policy lock here would deadlock a single-connection pool.
        let principal = self.delegate.resolve(tenant, identity).await?;
        let current_seconds = identity
            .verified_at_seconds()
            .checked_add(started.elapsed().as_secs())
            .ok_or(NostrPrincipalResolutionError::Unavailable)?;
        let final_snapshot = self
            .check_restrictions_at(tenant, identity, current_seconds, Some(&principal))
            .await?;
        if principal.community_id() != tenant.community_id()
            || principal.principal_id() != final_snapshot.identity.principal_id
        {
            return Err(NostrPrincipalResolutionError::Denied);
        }
        Ok(principal)
    }
}

pub struct ClosedNostrPrincipalResolver {
    profiles: crate::identity::profile_repository::IdentityProfileRepository,
    connection: sea_orm::DatabaseConnection,
}

impl ClosedNostrPrincipalResolver {
    pub fn new(
        connection: sea_orm::DatabaseConnection,
    ) -> Result<Self, crate::identity::profile_repository::IdentityProfileRepositoryError> {
        if !matches!(
            &connection,
            sea_orm::DatabaseConnection::SqlxPostgresPoolConnection(_)
        ) {
            return Err(crate::identity::profile_repository::IdentityProfileRepositoryError::UnsupportedDatabase);
        }
        Ok(Self {
            profiles: crate::identity::profile_repository::IdentityProfileRepository::new(
                sea_orm::DatabaseConnection::from(
                    connection.get_postgres_connection_pool().clone(),
                ),
            )?,
            connection,
        })
    }
}

#[async_trait]
impl NostrPrincipalResolver for ClosedNostrPrincipalResolver {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<AuthenticatedPrincipal, NostrPrincipalResolutionError> {
        use sea_orm::TransactionTrait as _;
        let principal = self.profiles.resolve(tenant, identity).await?;
        let transaction = self
            .connection
            .begin_with_config(Some(sea_orm::IsolationLevel::ReadCommitted), None)
            .await
            .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
        let result = async {
            crate::admin::moderation::lock_moderation_in_transaction(&transaction, tenant)
                .await
                .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
            self.validate_current_in_transaction(
                &transaction,
                tenant,
                identity,
                &principal,
                identity.verified_at_seconds(),
            )
            .await
        }
        .await;
        match result {
            Ok(()) => {
                transaction
                    .commit()
                    .await
                    .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
                Ok(principal)
            }
            Err(error) => {
                transaction
                    .rollback()
                    .await
                    .map_err(|_| NostrPrincipalResolutionError::Unavailable)?;
                Err(error)
            }
        }
    }

    async fn validate_current_in_transaction(
        &self,
        transaction: &sea_orm::DatabaseTransaction,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
        principal: &AuthenticatedPrincipal,
        now_seconds: u64,
    ) -> Result<(), NostrPrincipalResolutionError> {
        if !principal_matches_identity(principal, identity) {
            return Err(NostrPrincipalResolutionError::Denied);
        }
        self.profiles
            .current_closed_nip42_membership_in_transaction(
                transaction,
                tenant,
                principal,
                now_seconds,
            )
            .await
            .map(|_| ())
    }
}
