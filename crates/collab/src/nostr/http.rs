use std::{collections::BTreeMap, error::Error, fmt, str::FromStr, sync::Arc};

use async_trait::async_trait;
use axum::{
    Json, Router,
    extract::{Query, State},
    http::{HeaderMap, StatusCode, Uri, header},
    response::{IntoResponse, Response},
    routing::get,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use collaboration_domain::{
    AuthenticatedPrincipal, AuthenticatedPrincipalKind, CommunityId, NostrAuthenticationMethod,
    NostrPublicKey, TenantContext, TrustedTenantRoute, TrustedTenantRouteSource,
};
use nostr_compat::{EventId, PublicKey, SignedEvent, TimestampPolicy, verify_signed_event};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use url::Url;

use crate::nostr::{
    MAX_NOSTR_FRAME_BYTES,
    event_ingest::parse_wire_signed_event,
    subscriptions::{
        MAX_ACTIVE_SUBSCRIPTIONS, MAX_FILTER_LIMIT, MAX_FILTERS_PER_REQUEST,
        MAX_SUBSCRIPTION_ID_BYTES,
    },
};

const NIP98_KIND: u16 = 27_235;
const NIP98_FRESHNESS_SECONDS: u64 = 60;
const NIP98_REPLAY_SECONDS: u64 = 120;
const MAX_AUTHORIZATION_BYTES: usize = MAX_NOSTR_FRAME_BYTES * 2;
const MAX_NIP98_BODY_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_NIP05_HANDLE_CHARACTERS: usize = 255;
const MAX_METADATA_TEXT_BYTES: usize = 1_024;
const MAX_METADATA_URL_BYTES: usize = 2_048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrHttpDirectoryError {
    Denied,
    Unavailable,
}

#[async_trait]
pub trait NostrHttpHostResolver: Send + Sync {
    async fn resolve_host(
        &self,
        canonical_host: &str,
    ) -> Result<TenantContext, NostrHttpDirectoryError>;
}

#[derive(Clone)]
pub struct NostrDeploymentHostResolver {
    routes: BTreeMap<String, CommunityId>,
}

impl NostrDeploymentHostResolver {
    pub fn new(
        routes: impl IntoIterator<Item = (CommunityId, Url)>,
    ) -> Result<Self, NostrHttpConfigurationError> {
        let mut admitted = BTreeMap::new();
        for (route_index, (community, endpoint)) in routes.into_iter().enumerate() {
            let host = endpoint
                .host_str()
                .ok_or(NostrHttpConfigurationError::InvalidMetadata)?;
            let tls = matches!(endpoint.scheme(), "https" | "wss");
            let loopback = host == "localhost"
                || host
                    .trim_matches(['[', ']'])
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback());
            if route_index >= 256
                || !tls && (!matches!(endpoint.scheme(), "http" | "ws") || !loopback)
                || !endpoint.username().is_empty()
                || endpoint.password().is_some()
                || endpoint.query().is_some()
                || endpoint.fragment().is_some()
                || !matches!(endpoint.path(), "" | "/")
            {
                return Err(NostrHttpConfigurationError::InvalidMetadata);
            }
            let authority = match endpoint.port() {
                Some(port) => format!("{host}:{port}"),
                None => host.to_owned(),
            };
            if admitted.insert(authority, community).is_some() {
                return Err(NostrHttpConfigurationError::InvalidMetadata);
            }
            if endpoint.port().is_none()
                && admitted
                    .insert(format!("{host}:{}", if tls { 443 } else { 80 }), community)
                    .is_some()
            {
                return Err(NostrHttpConfigurationError::InvalidMetadata);
            }
        }
        if admitted.is_empty() {
            return Err(NostrHttpConfigurationError::InvalidMetadata);
        }
        Ok(Self { routes: admitted })
    }
}

#[async_trait]
impl NostrHttpHostResolver for NostrDeploymentHostResolver {
    async fn resolve_host(
        &self,
        canonical_host: &str,
    ) -> Result<TenantContext, NostrHttpDirectoryError> {
        let community = self
            .routes
            .get(canonical_host)
            .copied()
            .ok_or(NostrHttpDirectoryError::Denied)?;
        let route = TrustedTenantRoute::from_direct_host(community, canonical_host)
            .map_err(|_| NostrHttpDirectoryError::Denied)?;
        TenantContext::establish(Some(route), &[]).map_err(|_| NostrHttpDirectoryError::Denied)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NostrTenantRelayMetadata {
    icon: Option<String>,
}

impl NostrTenantRelayMetadata {
    pub fn new(icon: Option<String>) -> Result<Self, NostrHttpConfigurationError> {
        if icon.as_ref().is_some_and(|value| {
            value.is_empty() || value.len() > MAX_METADATA_URL_BYTES || Url::parse(value).is_err()
        }) {
            return Err(NostrHttpConfigurationError::InvalidMetadata);
        }
        Ok(Self { icon })
    }

    pub fn icon(&self) -> Option<&str> {
        self.icon.as_deref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Nip05Identity {
    public_key: PublicKey,
}

impl Nip05Identity {
    pub const fn new(public_key: PublicKey) -> Self {
        Self { public_key }
    }

    pub const fn public_key(self) -> PublicKey {
        self.public_key
    }
}

#[async_trait]
pub trait NostrHttpDirectory: Send + Sync {
    async fn relay_metadata(
        &self,
        tenant: &TenantContext,
    ) -> Result<NostrTenantRelayMetadata, NostrHttpDirectoryError>;

    async fn resolve_nip05(
        &self,
        tenant: &TenantContext,
        canonical_name: &str,
    ) -> Result<Option<Nip05Identity>, NostrHttpDirectoryError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NostrHttpConfiguration {
    name: String,
    description: String,
    software: String,
    version: String,
    relay_self: Option<PublicKey>,
    supported_nips: Vec<u32>,
    supported_extensions: Vec<String>,
    public_tls: bool,
}

impl NostrHttpConfiguration {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        software: impl Into<String>,
        version: impl Into<String>,
        relay_self: Option<PublicKey>,
        supported_nips: Vec<u32>,
        supported_extensions: Vec<String>,
        public_tls: bool,
    ) -> Result<Self, NostrHttpConfigurationError> {
        let name = name.into();
        let description = description.into();
        let software = software.into();
        let version = version.into();
        if !valid_metadata_text(&name)
            || !valid_metadata_text(&description)
            || !valid_metadata_text(&version)
            || software.len() > MAX_METADATA_URL_BYTES
            || Url::parse(&software).is_err()
            || supported_nips.len() > 128
            || supported_extensions.len() > 128
            || supported_extensions.iter().any(|value| {
                value.is_empty() || value.len() > 128 || value.chars().any(char::is_control)
            })
        {
            return Err(NostrHttpConfigurationError::InvalidMetadata);
        }
        let mut supported_nips = supported_nips;
        supported_nips.sort_unstable();
        supported_nips.dedup();
        let mut supported_extensions = supported_extensions;
        supported_extensions.sort();
        supported_extensions.dedup();
        Ok(Self {
            name,
            description,
            software,
            version,
            relay_self,
            supported_nips,
            supported_extensions,
            public_tls,
        })
    }

    pub fn buzz_compatible(
        version: impl Into<String>,
        relay_self: Option<PublicKey>,
        public_tls: bool,
    ) -> Result<Self, NostrHttpConfigurationError> {
        Self::new(
            "Zed Collaborative Relay",
            "Zed collaborative workspace relay",
            "https://github.com/imagine-engineering/made",
            version,
            relay_self,
            vec![1, 2, 10, 11, 16, 17, 23, 25, 29, 33, 38, 42, 50, 56],
            vec!["nip-er".into()],
            public_tls,
        )
    }
}

fn valid_metadata_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_METADATA_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

#[derive(Clone)]
pub struct NostrPublicHttpState {
    configuration: NostrHttpConfiguration,
    host_resolver: Arc<dyn NostrHttpHostResolver>,
    directory: Arc<dyn NostrHttpDirectory>,
}

impl NostrPublicHttpState {
    pub fn new(
        configuration: NostrHttpConfiguration,
        host_resolver: Arc<dyn NostrHttpHostResolver>,
        directory: Arc<dyn NostrHttpDirectory>,
    ) -> Self {
        Self {
            configuration,
            host_resolver,
            directory,
        }
    }
}

pub fn router(state: Arc<NostrPublicHttpState>) -> Router {
    Router::new()
        .route("/", get(nip11_root))
        .route("/info", get(nip11_info))
        .route("/.well-known/nostr.json", get(nip05))
        .with_state(state)
}

async fn nip11_root(
    State(state): State<Arc<NostrPublicHttpState>>,
    headers: HeaderMap,
) -> Response {
    if !accepts_nostr_metadata(&headers) {
        return StatusCode::NOT_FOUND.into_response();
    }
    nip11_response(&state, &headers).await
}

fn accepts_nostr_metadata(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .any(|part| part.trim() == "application/nostr+json")
        })
}

async fn nip11_info(
    State(state): State<Arc<NostrPublicHttpState>>,
    headers: HeaderMap,
) -> Response {
    nip11_response(&state, &headers).await
}

async fn nip11_response(state: &NostrPublicHttpState, headers: &HeaderMap) -> Response {
    let tenant = resolve_request_tenant(state.host_resolver.as_ref(), headers).await;
    let tenant_metadata = match tenant.as_ref() {
        Some(tenant) => state.directory.relay_metadata(tenant).await.ok(),
        None => None,
    };
    nip11_document_response(&state.configuration, tenant_metadata.as_ref())
}

fn nip11_document_response(
    configuration: &NostrHttpConfiguration,
    metadata: Option<&NostrTenantRelayMetadata>,
) -> Response {
    Json(Nip11Document::from_configuration(configuration, metadata)).into_response()
}

pub struct NostrPublicMetadataHttpState {
    configuration: NostrHttpConfiguration,
    host_resolver: NostrDeploymentHostResolver,
    profiles: Arc<crate::identity::profile_repository::IdentityProfileRepository>,
}

impl NostrPublicMetadataHttpState {
    async fn response(&self, headers: &HeaderMap) -> Response {
        let metadata = match resolve_request_tenant(&self.host_resolver, headers).await {
            Some(tenant) => match self
                .profiles
                .current_public_community_metadata(&tenant)
                .await
            {
                Ok(community) => community.map(|community| NostrTenantRelayMetadata {
                    // Canonical community icons may be data URLs or other bounded scalars, not the older fixture constructor's URL-only metadata.
                    icon: community
                        .icon
                        .filter(|icon| !icon.as_str().is_empty())
                        .map(|icon| icon.as_str().to_owned()),
                }),
                Err(_) => {
                    log::warn!("NIP11 community metadata unavailable; returning generic document");
                    None
                }
            },
            None => None,
        };
        nip11_document_response(&self.configuration, metadata.as_ref())
    }
}

pub fn metadata_http_router(state: Arc<NostrPublicMetadataHttpState>) -> Router {
    Router::new()
        .route("/info", get(metadata_http_info))
        .route("/.well-known/nostr.json", get(metadata_http_nip05))
        .with_state(state)
}

async fn metadata_http_info(
    State(state): State<Arc<NostrPublicMetadataHttpState>>,
    headers: HeaderMap,
) -> Response {
    state.response(&headers).await
}

async fn metadata_http_nip05(
    State(state): State<Arc<NostrPublicMetadataHttpState>>,
    headers: HeaderMap,
    Query(query): Query<Nip05Query>,
) -> Response {
    let identity = match (
        query
            .name
            .and_then(|name| canonical_nip05_name(&name, None)),
        resolve_request_tenant(&state.host_resolver, &headers).await,
    ) {
        (Some(name), Some(tenant)) => {
            match state.profiles.current_accepted_nip05(&tenant, &name).await {
                Ok(identity) => {
                    identity.map(|key| (name, key, tenant.route_reference().to_owned()))
                }
                Err(_) => {
                    log::warn!("NIP05 accepted identity unavailable; returning empty directory");
                    None
                }
            }
        }
        _ => None,
    };
    nip05_document_response(&state.configuration, identity)
}

pub async fn metadata_http_root_response(
    state: Option<&NostrPublicMetadataHttpState>,
    headers: &HeaderMap,
    status: String,
) -> Response {
    match state {
        Some(state) if accepts_nostr_metadata(headers) => state.response(headers).await,
        _ => status.into_response(),
    }
}

#[derive(Deserialize)]
struct Nip05Query {
    name: Option<String>,
}

async fn nip05(
    State(state): State<Arc<NostrPublicHttpState>>,
    headers: HeaderMap,
    Query(query): Query<Nip05Query>,
) -> Response {
    let identity = match (
        query
            .name
            .and_then(|value| canonical_nip05_name(&value, None)),
        resolve_request_tenant(state.host_resolver.as_ref(), &headers).await,
    ) {
        (Some(name), Some(tenant)) => match state.directory.resolve_nip05(&tenant, &name).await {
            Ok(identity) => identity.map(|identity| {
                (
                    name,
                    identity.public_key(),
                    tenant.route_reference().to_owned(),
                )
            }),
            Err(_) => {
                log::warn!("NIP05 accepted identity unavailable; returning empty directory");
                None
            }
        },
        _ => None,
    };
    nip05_document_response(&state.configuration, identity)
}

fn nip05_document_response(
    configuration: &NostrHttpConfiguration,
    identity: Option<(String, PublicKey, String)>,
) -> Response {
    let mut body = Nip05Document::default();
    if let Some((name, public_key, host)) = identity {
        let key = public_key.to_hex();
        body.names.insert(name, key.clone());
        body.relays.insert(
            key,
            vec![format!(
                "{}://{}",
                if configuration.public_tls {
                    "wss"
                } else {
                    "ws"
                },
                host
            )],
        );
    }
    let mut response = Json(body).into_response();
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        axum::http::HeaderValue::from_static("*"),
    );
    response
}

async fn resolve_request_tenant(
    resolver: &dyn NostrHttpHostResolver,
    headers: &HeaderMap,
) -> Option<TenantContext> {
    let host = canonical_host(headers)?;
    let tenant = resolver.resolve_host(&host).await.ok()?;
    matches!(
        tenant.route_source(),
        TrustedTenantRouteSource::DirectHost | TrustedTenantRouteSource::TrustedForwardedHost
    )
    .then_some(tenant)
}

fn canonical_host(headers: &HeaderMap) -> Option<String> {
    let mut hosts = headers.get_all(header::HOST).iter();
    let raw = hosts.next()?.to_str().ok()?;
    if hosts.next().is_some()
        || raw.is_empty()
        || raw.len() > 1_024
        || raw.contains('@')
        || raw.chars().any(char::is_control)
    {
        return None;
    }
    let authority = axum::http::uri::Authority::from_str(raw).ok()?;
    let host = authority.host();
    if host.is_empty() {
        return None;
    }
    // URI Authority accepts user-info and can hide an invalid port behind None.
    // Host admission must reject those forms instead of resolving a different host.
    let suffix = raw.strip_prefix(host)?;
    let host = host.to_ascii_lowercase();
    if suffix.is_empty() {
        Some(host)
    } else {
        let port = suffix.strip_prefix(':')?;
        if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        Some(format!("{host}:{}", port.parse::<u16>().ok()?))
    }
}

pub(crate) fn canonical_nip05_name(value: &str, expected_host: Option<&str>) -> Option<String> {
    let canonical = match expected_host {
        Some(host) => {
            let (local, domain) = value.trim().split_once('@')?;
            if local.is_empty() || domain.is_empty() || domain.to_lowercase() != host {
                return None;
            }
            format!("{}@{}", local.to_lowercase(), domain.to_lowercase())
        }
        None => value.to_lowercase(),
    };
    (!canonical.is_empty()
        && (expected_host.is_some() || canonical.chars().count() <= MAX_NIP05_HANDLE_CHARACTERS))
        .then_some(canonical)
}

#[derive(Serialize)]
struct Nip11Document {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    icon: Option<String>,
    pubkey: Option<String>,
    contact: Option<String>,
    supported_nips: Vec<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    supported_extensions: Vec<String>,
    software: String,
    version: String,
    limitation: Nip11Limitation,
    #[serde(rename = "self", skip_serializing_if = "Option::is_none")]
    relay_self: Option<String>,
}

impl Nip11Document {
    fn from_configuration(
        configuration: &NostrHttpConfiguration,
        tenant_metadata: Option<&NostrTenantRelayMetadata>,
    ) -> Self {
        Self {
            name: configuration.name.clone(),
            description: configuration.description.clone(),
            icon: tenant_metadata.and_then(|metadata| metadata.icon().map(str::to_owned)),
            pubkey: None,
            contact: None,
            supported_nips: configuration.supported_nips.clone(),
            supported_extensions: configuration.supported_extensions.clone(),
            software: configuration.software.clone(),
            version: configuration.version.clone(),
            limitation: Nip11Limitation {
                max_message_length: MAX_NOSTR_FRAME_BYTES as u64,
                max_subscriptions: configuration
                    .supported_nips
                    .contains(&1)
                    .then_some(MAX_ACTIVE_SUBSCRIPTIONS as u32),
                max_filters: MAX_FILTERS_PER_REQUEST as u32,
                max_limit: MAX_FILTER_LIMIT,
                max_subid_length: configuration
                    .supported_nips
                    .contains(&1)
                    .then_some(MAX_SUBSCRIPTION_ID_BYTES as u32),
                auth_required: true,
                payment_required: false,
                restricted_writes: true,
            },
            relay_self: configuration.relay_self.map(PublicKey::to_hex),
        }
    }
}

#[derive(Serialize)]
struct Nip11Limitation {
    max_message_length: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_subscriptions: Option<u32>,
    max_filters: u32,
    max_limit: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_subid_length: Option<u32>,
    auth_required: bool,
    payment_required: bool,
    restricted_writes: bool,
}

#[derive(Default, Serialize)]
struct Nip05Document {
    names: BTreeMap<String, String>,
    relays: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nip98ReplayClaim {
    Claimed,
    Replay,
}

#[async_trait]
pub trait Nip98ReplayStore: Send + Sync {
    async fn claim(
        &self,
        tenant: &TenantContext,
        event_id: EventId,
        expires_at_seconds: u64,
    ) -> Result<Nip98ReplayClaim, NostrHttpDirectoryError>;
}

#[derive(Clone)]
pub struct VerifiedNip98Identity {
    community_id: CommunityId,
    signed_authentication_event: SignedEvent,
    verified_at_seconds: u64,
    request_url: String,
    request_method: String,
    request_body_sha256: [u8; 32],
    request_payload_verified: bool,
}

impl VerifiedNip98Identity {
    pub const fn public_key(&self) -> NostrPublicKey {
        NostrPublicKey::from_bytes(*self.signed_authentication_event.event.public_key.as_bytes())
    }

    pub const fn event_id(&self) -> EventId {
        self.signed_authentication_event.claimed_id
    }

    pub const fn verified_at_seconds(&self) -> u64 {
        self.verified_at_seconds
    }

    pub const fn signed_authentication_event(&self) -> &SignedEvent {
        &self.signed_authentication_event
    }

    pub fn request_url(&self) -> &str {
        &self.request_url
    }

    pub fn request_method(&self) -> &str {
        &self.request_method
    }

    pub const fn request_body_sha256(&self) -> &[u8; 32] {
        &self.request_body_sha256
    }

    pub const fn request_payload_verified(&self) -> bool {
        self.request_payload_verified
    }

    pub fn for_tenant(&self, tenant: &TenantContext) -> Result<&Self, NostrHttpDirectoryError> {
        if self.community_id != tenant.community_id() {
            return Err(NostrHttpDirectoryError::Denied);
        }
        Ok(self)
    }
}

impl fmt::Debug for VerifiedNip98Identity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("VerifiedNip98Identity([redacted])")
    }
}

#[async_trait]
pub trait Nip98PrincipalResolver: Send + Sync {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip98Identity,
    ) -> Result<AuthenticatedPrincipal, NostrHttpDirectoryError>;
}

pub struct Nip98Authenticator {
    public_tls: bool,
    replay_store: Arc<dyn Nip98ReplayStore>,
    principal_resolver: Arc<dyn Nip98PrincipalResolver>,
}

impl Nip98Authenticator {
    pub fn new(
        public_tls: bool,
        replay_store: Arc<dyn Nip98ReplayStore>,
        principal_resolver: Arc<dyn Nip98PrincipalResolver>,
    ) -> Self {
        Self {
            public_tls,
            replay_store,
            principal_resolver,
        }
    }

    pub async fn authenticate(
        &self,
        tenant: &TenantContext,
        authorization: &str,
        method: &str,
        path_and_query: &str,
        body: &[u8],
        require_payload: bool,
        now: u64,
    ) -> Result<AuthenticatedPrincipal, Nip98AuthenticationError> {
        self.authenticate_internal(
            tenant,
            authorization,
            method,
            path_and_query,
            body,
            require_payload,
            now,
            None,
        )
        .await
    }

    async fn authenticate_admitted(
        &self,
        tenant: &TenantContext,
        authorization: &str,
        method: &str,
        path_and_query: &str,
        body: &[u8],
        now: u64,
        quota: &crate::nostr::replay::RedisHttpAdmissionQuota,
    ) -> Result<AuthenticatedPrincipal, Nip98AuthenticationError> {
        self.authenticate_internal(
            tenant,
            authorization,
            method,
            path_and_query,
            body,
            false,
            now,
            Some(quota),
        )
        .await
    }

    async fn authenticate_internal(
        &self,
        tenant: &TenantContext,
        authorization: &str,
        method: &str,
        path_and_query: &str,
        body: &[u8],
        require_payload: bool,
        now: u64,
        quota: Option<&crate::nostr::replay::RedisHttpAdmissionQuota>,
    ) -> Result<AuthenticatedPrincipal, Nip98AuthenticationError> {
        validate_http_tenant(tenant)?;
        if authorization.len() > MAX_AUTHORIZATION_BYTES || body.len() > MAX_NIP98_BODY_BYTES {
            return Err(Nip98AuthenticationError::Malformed);
        }
        let encoded = authorization
            .strip_prefix("Nostr ")
            .ok_or(Nip98AuthenticationError::Missing)?;
        let decoded = BASE64
            .decode(encoded)
            .map_err(|_| Nip98AuthenticationError::Malformed)?;
        if decoded.len() > MAX_NOSTR_FRAME_BYTES {
            return Err(Nip98AuthenticationError::Malformed);
        }
        let wire_event: serde_json::Value =
            serde_json::from_slice(&decoded).map_err(|_| Nip98AuthenticationError::Malformed)?;
        let (signed_event, _) =
            parse_wire_signed_event(wire_event).map_err(|_| Nip98AuthenticationError::Malformed)?;
        if signed_event.event.kind != NIP98_KIND {
            return Err(Nip98AuthenticationError::Invalid);
        }
        let verification_event = Arc::new(signed_event);
        let blocking_event = Arc::clone(&verification_event);
        let verification = tokio::task::spawn_blocking(move || {
            verify_signed_event(
                &blocking_event,
                TimestampPolicy::Bounded {
                    now,
                    max_past_seconds: NIP98_FRESHNESS_SECONDS,
                    max_future_seconds: NIP98_FRESHNESS_SECONDS,
                },
            )
        })
        .await
        .map_err(|_| Nip98AuthenticationError::Unavailable)?;
        verification.map_err(|_| Nip98AuthenticationError::Invalid)?;
        let signed_event =
            Arc::try_unwrap(verification_event).unwrap_or_else(|event| (*event).clone());

        let expected_url = expected_http_url(self.public_tls, tenant, path_and_query)?;
        validate_nip98_tags(
            &signed_event.event.tags,
            &expected_url,
            method,
            body,
            require_payload,
        )?;

        if let Some(quota) = quota {
            quota
                .check(tenant, signed_event.event.public_key)
                .await
                .map_err(|error| match error {
                    crate::nostr::replay::HttpQuotaAdmissionError::Exceeded {
                        reset_in_seconds,
                    } => Nip98AuthenticationError::RateLimited { reset_in_seconds },
                    crate::nostr::replay::HttpQuotaAdmissionError::Unavailable => {
                        Nip98AuthenticationError::Unavailable
                    }
                })?;
        }

        match self
            .replay_store
            .claim(
                tenant,
                signed_event.claimed_id,
                now.saturating_add(NIP98_REPLAY_SECONDS),
            )
            .await
        {
            Ok(Nip98ReplayClaim::Claimed) => {}
            Ok(Nip98ReplayClaim::Replay) => return Err(Nip98AuthenticationError::Replay),
            Err(_) => return Err(Nip98AuthenticationError::Unavailable),
        }

        let request_payload_verified = signed_event
            .event
            .tags
            .iter()
            .any(|tag| tag.first().is_some_and(|name| name == "payload"));
        let identity = VerifiedNip98Identity {
            community_id: tenant.community_id(),
            signed_authentication_event: signed_event,
            verified_at_seconds: now,
            request_url: expected_url,
            request_method: method.to_owned(),
            request_body_sha256: Sha256::digest(body).into(),
            request_payload_verified,
        };
        let principal = self
            .principal_resolver
            .resolve(tenant, &identity)
            .await
            .map_err(|error| match error {
                NostrHttpDirectoryError::Denied => Nip98AuthenticationError::Denied,
                NostrHttpDirectoryError::Unavailable => Nip98AuthenticationError::Unavailable,
            })?;
        validate_nip98_principal(tenant, &principal, identity.public_key())?;
        Ok(principal)
    }
}

fn validate_http_tenant(tenant: &TenantContext) -> Result<(), Nip98AuthenticationError> {
    if !matches!(
        tenant.route_source(),
        TrustedTenantRouteSource::DirectHost | TrustedTenantRouteSource::TrustedForwardedHost
    ) || canonical_authority(tenant.route_reference()).as_deref()
        != Some(tenant.route_reference())
    {
        return Err(Nip98AuthenticationError::TenantBinding);
    }
    Ok(())
}

fn canonical_authority(raw: &str) -> Option<String> {
    let authority = axum::http::uri::Authority::from_str(raw).ok()?;
    let host = authority.host().to_ascii_lowercase();
    if host.is_empty() {
        return None;
    }
    Some(match authority.port_u16() {
        Some(port) => format!("{host}:{port}"),
        None => host,
    })
}

fn expected_http_url(
    public_tls: bool,
    tenant: &TenantContext,
    path_and_query: &str,
) -> Result<String, Nip98AuthenticationError> {
    if !path_and_query.starts_with('/')
        || path_and_query.len() > 8_192
        || path_and_query.chars().any(char::is_control)
    {
        return Err(Nip98AuthenticationError::Invalid);
    }
    let value = format!(
        "{}://{}{}",
        if public_tls { "https" } else { "http" },
        tenant.route_reference(),
        path_and_query
    );
    normalize_http_url(&value).ok_or(Nip98AuthenticationError::Invalid)
}

fn validate_nip98_tags(
    tags: &[Vec<String>],
    expected_url: &str,
    method: &str,
    body: &[u8],
    require_payload: bool,
) -> Result<(), Nip98AuthenticationError> {
    if method.is_empty()
        || method.len() > 32
        || !method.bytes().all(|byte| byte.is_ascii_alphabetic())
    {
        return Err(Nip98AuthenticationError::Invalid);
    }
    let url = unique_tag_value(tags, "u")?;
    let signed_url = normalize_http_url(url).ok_or(Nip98AuthenticationError::Invalid)?;
    if signed_url != expected_url {
        return Err(Nip98AuthenticationError::TenantBinding);
    }
    let signed_method = unique_tag_value(tags, "method")?;
    if !signed_method.eq_ignore_ascii_case(method) {
        return Err(Nip98AuthenticationError::Invalid);
    }
    let payloads = tag_values(tags, "payload")?;
    if payloads.len() > 1 || (require_payload && payloads.is_empty()) {
        return Err(Nip98AuthenticationError::Invalid);
    }
    if let Some(payload) = payloads.first() {
        let computed = hex::encode(Sha256::digest(body));
        if payload.len() != 64
            || !payload
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || *payload != computed
        {
            return Err(Nip98AuthenticationError::Invalid);
        }
    }
    Ok(())
}

fn unique_tag_value<'a>(
    tags: &'a [Vec<String>],
    name: &str,
) -> Result<&'a str, Nip98AuthenticationError> {
    let values = tag_values(tags, name)?;
    if values.len() != 1 {
        return Err(Nip98AuthenticationError::Invalid);
    }
    Ok(values[0])
}

fn tag_values<'a>(
    tags: &'a [Vec<String>],
    name: &str,
) -> Result<Vec<&'a str>, Nip98AuthenticationError> {
    tags.iter()
        .filter(|tag| tag.first().is_some_and(|value| value == name))
        .map(|tag| {
            if tag.len() != 2 || tag[1].is_empty() {
                Err(Nip98AuthenticationError::Invalid)
            } else {
                Ok(tag[1].as_str())
            }
        })
        .collect()
}

fn normalize_http_url(raw: &str) -> Option<String> {
    let mut url = Url::parse(raw).ok()?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.host_str().is_none()
    {
        return None;
    }
    let path = url.path().trim_end_matches('/').to_owned();
    url.set_path(&path);
    Some(url.to_string())
}

fn validate_nip98_principal(
    tenant: &TenantContext,
    principal: &AuthenticatedPrincipal,
    public_key: NostrPublicKey,
) -> Result<(), Nip98AuthenticationError> {
    if principal.community_id() != tenant.community_id() {
        return Err(Nip98AuthenticationError::TenantBinding);
    }
    match principal.kind() {
        AuthenticatedPrincipalKind::NostrIdentity {
            public_key: resolved,
            authentication_method: NostrAuthenticationMethod::Nip98,
            ..
        } if *resolved == public_key => Ok(()),
        AuthenticatedPrincipalKind::OwnerAttestedAgent {
            agent_public_key,
            authentication_method: NostrAuthenticationMethod::Nip98,
            ..
        } if *agent_public_key == public_key => Ok(()),
        _ => Err(Nip98AuthenticationError::Denied),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nip98AuthenticationError {
    Missing,
    Malformed,
    Invalid,
    TenantBinding,
    Replay,
    RateLimited { reset_in_seconds: u64 },
    Denied,
    Unavailable,
}

impl fmt::Display for Nip98AuthenticationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Missing => "NIP-98 authorization is missing",
            Self::Malformed => "NIP-98 authorization is malformed",
            Self::Invalid => "NIP-98 authorization is invalid",
            Self::TenantBinding => "NIP-98 authorization does not match the request tenant",
            Self::Replay => "NIP-98 authorization was already used",
            Self::RateLimited { .. } => "shared HTTP admission quota exceeded",
            Self::Denied => "NIP-98 principal is not authorized",
            Self::Unavailable => "NIP-98 authorization is unavailable",
        })
    }
}

impl Error for Nip98AuthenticationError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrHttpConfigurationError {
    InvalidMetadata,
}

impl fmt::Display for NostrHttpConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Nostr HTTP metadata configuration is invalid")
    }
}

impl Error for NostrHttpConfigurationError {}

pub fn request_path_and_query(uri: &Uri) -> &str {
    uri.path_and_query()
        .map(axum::http::uri::PathAndQuery::as_str)
        .unwrap_or("/")
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileHttpDeployment {
    community_id: uuid::Uuid,
    public_url: Url,
}

pub struct NostrProfileHttpState {
    host_resolver: NostrDeploymentHostResolver,
    metadata: Arc<NostrPublicMetadataHttpState>,
    authenticator: Nip98Authenticator,
    quota: crate::nostr::replay::RedisHttpAdmissionQuota,
    profiles: Arc<crate::identity::profile_repository::IdentityProfileRepository>,
    ingress: crate::nostr::event_ingest::NostrEventIngress<
        crate::identity::profile_repository::IdentityProfileRepository,
    >,
    socket_connection: sea_orm::DatabaseConnection,
    socket_replay: crate::nostr::replay::RedisAuthenticationReplayStore,
    public_tls: bool,
    closed_websocket: bool,
}

impl NostrProfileHttpState {
    pub fn metadata(&self) -> Arc<NostrPublicMetadataHttpState> {
        self.metadata.clone()
    }

    pub fn new(
        connection: sea_orm::DatabaseConnection,
        routes: Vec<(CommunityId, Url)>,
        redis_endpoint: &str,
        calls_per_minute: u64,
    ) -> anyhow::Result<Self> {
        use crate::{
            identity::profile_repository::IdentityProfileRepository,
            nostr::{
                event_ingest::NostrEventIngress,
                ingress::NostrIngressDeployment,
                replay::{RedisAuthenticationReplayStore, RedisHttpAdmissionQuota},
            },
        };
        let first = routes
            .first()
            .ok_or_else(|| anyhow::anyhow!("Nostr profile routes are required"))?;
        let public_tls = matches!(first.1.scheme(), "https" | "wss");
        if routes.iter().any(|(community, endpoint)| {
            community.as_uuid().is_nil()
                || matches!(endpoint.scheme(), "https" | "wss") != public_tls
        }) {
            anyhow::bail!(
                "Nostr profile routes require nonempty communities and one public transport scheme"
            );
        }
        let host_resolver = NostrDeploymentHostResolver::new(routes)?;
        if !matches!(
            &connection,
            sea_orm::DatabaseConnection::SqlxPostgresPoolConnection(_)
        ) {
            anyhow::bail!("Nostr profile HTTP requires PostgreSQL");
        }
        let pool = connection.get_postgres_connection_pool().clone();
        let profiles = Arc::new(IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?);
        let replay = Arc::new(RedisAuthenticationReplayStore::new(redis_endpoint)?);
        let socket_replay = replay.as_ref().clone();
        let authenticator = Nip98Authenticator::new(public_tls, replay, profiles.clone());
        let quota = RedisHttpAdmissionQuota::new(redis_endpoint, calls_per_minute)?;
        let ingress = NostrEventIngress::new(
            IdentityProfileRepository::new(sea_orm::DatabaseConnection::from(pool.clone()))?,
            NostrIngressDeployment::InProcess,
        );
        let metadata = Arc::new(NostrPublicMetadataHttpState {
            configuration: NostrHttpConfiguration::new(
                "Made Collaborative Service",
                "Made collaborative workspace profile service",
                "https://github.com/simtropolis/made",
                crate::VERSION,
                None,
                vec![5, 11, 98],
                Vec::new(),
                public_tls,
            )?,
            host_resolver: host_resolver.clone(),
            profiles: profiles.clone(),
        });
        Ok(Self {
            host_resolver,
            metadata,
            authenticator,
            quota,
            profiles,
            ingress,
            socket_connection: sea_orm::DatabaseConnection::from(pool),
            socket_replay,
            public_tls,
            closed_websocket: false,
        })
    }

    pub fn enable_closed_websocket(mut self) -> anyhow::Result<Self> {
        let metadata = Arc::get_mut(&mut self.metadata)
            .ok_or_else(|| anyhow::anyhow!("socket configuration must precede state sharing"))?;
        metadata.configuration.supported_nips.push(42);
        metadata.configuration.supported_nips.sort_unstable();
        metadata.configuration.supported_nips.dedup();
        self.closed_websocket = true;
        Ok(self)
    }
}

pub fn profile_http_router(state: Arc<NostrProfileHttpState>) -> Router {
    Router::new()
        .route("/events", axum::routing::post(submit_profile_http))
        .route("/query", axum::routing::post(query_profile_http))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_NOSTR_FRAME_BYTES))
        .with_state(state)
}

pub fn configured_profile_http_router(
    config: &crate::Config,
    connection: sea_orm::DatabaseConnection,
) -> anyhow::Result<Option<Router>> {
    Ok(configured_profile_http_state(config, connection)?.map(profile_http_router))
}

pub fn configured_profile_http_state(
    config: &crate::Config,
    connection: sea_orm::DatabaseConnection,
) -> anyhow::Result<Option<Arc<NostrProfileHttpState>>> {
    let Some(routes) = &config.collaboration_nostr_routes else {
        if config.collaboration_nostr_closed_websocket == Some(true) {
            anyhow::bail!("closed NIP42 WebSocket requires configured Nostr routes");
        }
        if config.collaboration_nostr_api_calls_per_minute.is_some() {
            anyhow::bail!("Nostr HTTP quota requires explicit deployment routes");
        }
        return Ok(None);
    };
    let deployments: Vec<ProfileHttpDeployment> = serde_json::from_str(routes)?;
    let redis = config
        .collaboration_redis_url
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Nostr HTTP requires shared Redis admission"))?;
    let limit = config
        .collaboration_nostr_api_calls_per_minute
        .ok_or_else(|| anyhow::anyhow!("Nostr HTTP requires an explicit API quota"))?;
    let routes = deployments
        .into_iter()
        .map(|deployment| {
            (
                CommunityId::from_uuid(deployment.community_id),
                deployment.public_url,
            )
        })
        .collect();
    let state = NostrProfileHttpState::new(connection, routes, redis, limit)?;
    let state = if config.collaboration_nostr_closed_websocket == Some(true) {
        state.enable_closed_websocket()?
    } else {
        state
    };
    Ok(Some(Arc::new(state)))
}

fn profile_http_failure(status: StatusCode, message: &'static str) -> Response {
    (status, Json(serde_json::json!({"error": message}))).into_response()
}

fn profile_directory_failure(error: NostrHttpDirectoryError) -> Response {
    match error {
        NostrHttpDirectoryError::Denied => {
            profile_http_failure(StatusCode::FORBIDDEN, "profile access denied")
        }
        NostrHttpDirectoryError::Unavailable => profile_http_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "profile authority unavailable",
        ),
    }
}

async fn profile_http_principal(
    state: &NostrProfileHttpState,
    headers: &HeaderMap,
    uri: &Uri,
    body: &[u8],
) -> Result<(TenantContext, AuthenticatedPrincipal, u64), Box<Response>> {
    let tenant = resolve_request_tenant(&state.host_resolver, headers)
        .await
        .ok_or_else(|| profile_http_failure(StatusCode::NOT_FOUND, "relay not found"))?;
    let mut headers = headers.get_all(header::AUTHORIZATION).iter();
    let authorization = headers
        .next()
        .and_then(|header| header.to_str().ok())
        .filter(|_| headers.next().is_none())
        .ok_or_else(|| {
            profile_http_failure(StatusCode::UNAUTHORIZED, "Nostr authorization required")
        })?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| profile_http_failure(StatusCode::SERVICE_UNAVAILABLE, "clock unavailable"))?
        .as_secs();
    let principal = state
        .authenticator
        .authenticate_admitted(
            &tenant,
            authorization,
            "POST",
            uri.path_and_query()
                .map(|path| path.as_str())
                .unwrap_or(uri.path()),
            body,
            now,
            &state.quota,
        )
        .await
        .map_err(|error| match error {
            Nip98AuthenticationError::RateLimited { reset_in_seconds } => {
                let mut response = profile_http_failure(
                    StatusCode::TOO_MANY_REQUESTS,
                    "shared HTTP quota exceeded",
                );
                if let Ok(value) = axum::http::HeaderValue::from_str(&reset_in_seconds.to_string())
                {
                    response.headers_mut().insert(header::RETRY_AFTER, value);
                }
                response
            }
            Nip98AuthenticationError::Unavailable => profile_http_failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "authentication unavailable",
            ),
            Nip98AuthenticationError::Denied => {
                profile_http_failure(StatusCode::FORBIDDEN, "authentication denied")
            }
            _ => profile_http_failure(StatusCode::UNAUTHORIZED, "Nostr authorization invalid"),
        })?;
    let now_millis = now.checked_mul(1000).ok_or_else(|| {
        profile_http_failure(StatusCode::SERVICE_UNAVAILABLE, "clock unavailable")
    })?;
    Ok((tenant, principal, now_millis))
}

async fn submit_profile_http(
    State(state): State<Arc<NostrProfileHttpState>>,
    headers: HeaderMap,
    uri: Uri,
    body: axum::body::Bytes,
) -> Response {
    let (tenant, principal, now_millis) =
        match profile_http_principal(&state, &headers, &uri, &body).await {
            Ok(authenticated) => authenticated,
            Err(response) => return *response,
        };
    let event: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(event) => event,
        Err(_) => return profile_http_failure(StatusCode::BAD_REQUEST, "invalid event JSON"),
    };
    if event.get("kind").and_then(serde_json::Value::as_u64) != Some(0) {
        return profile_http_failure(
            StatusCode::UNPROCESSABLE_ENTITY,
            "event kind unsupported by profile service",
        );
    }
    let admission = match state
        .profiles
        .admit_profile_transport(&tenant, &principal, now_millis)
        .await
    {
        Ok(admission) => admission,
        Err(error) => return profile_directory_failure(error),
    };
    let frame = serde_json::json!(["EVENT", event]).to_string();
    let outcome = match state
        .ingress
        .handle_frame(admission, &frame, now_millis / 1000)
        .await
    {
        Ok(outcome) => outcome,
        Err(_) => return profile_http_failure(StatusCode::BAD_REQUEST, "invalid event frame"),
    };
    let response: serde_json::Value = match serde_json::from_str(outcome.frame()) {
        Ok(response) => response,
        Err(_) => {
            return profile_http_failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "invalid ingest response",
            );
        }
    };
    let status = match outcome.status() {
        crate::nostr::event_ingest::NostrEventIngestStatus::Accepted
        | crate::nostr::event_ingest::NostrEventIngestStatus::Duplicate => StatusCode::OK,
        crate::nostr::event_ingest::NostrEventIngestStatus::Invalid => StatusCode::BAD_REQUEST,
        crate::nostr::event_ingest::NostrEventIngestStatus::Unauthorized => StatusCode::FORBIDDEN,
        crate::nostr::event_ingest::NostrEventIngestStatus::Unavailable => {
            StatusCode::SERVICE_UNAVAILABLE
        }
    };
    (status, Json(serde_json::json!({"event_id":response.get(1),"accepted":response.get(2),"message":response.get(3)}))).into_response()
}

async fn query_profile_http(
    State(state): State<Arc<NostrProfileHttpState>>,
    headers: HeaderMap,
    uri: Uri,
    body: axum::body::Bytes,
) -> Response {
    use crate::nostr::subscriptions::NostrSubscriptionFrame;
    let (tenant, principal, now_millis) =
        match profile_http_principal(&state, &headers, &uri, &body).await {
            Ok(authenticated) => authenticated,
            Err(response) => return *response,
        };
    let values: Vec<serde_json::Value> = match serde_json::from_slice(&body) {
        Ok(filters) => filters,
        Err(_) => return profile_http_failure(StatusCode::BAD_REQUEST, "invalid profile filters"),
    };
    let mut frame = vec![serde_json::json!("REQ"), serde_json::json!("http-profile")];
    frame.extend(values);
    let filters = match NostrSubscriptionFrame::parse(&serde_json::Value::Array(frame).to_string())
    {
        Ok(NostrSubscriptionFrame::Req { filters, .. }) if !filters.is_empty() => filters,
        _ => return profile_http_failure(StatusCode::BAD_REQUEST, "invalid profile filters"),
    };
    if filters.iter().any(|filter| {
        filter.event_filter().kinds != [0]
            || filter.search().is_some()
            || !filter.event_filter().generic_tags.is_empty()
    }) {
        return profile_http_failure(
            StatusCode::UNPROCESSABLE_ENTITY,
            "filters unsupported by profile service",
        );
    }
    let queries = match filters
        .into_iter()
        .map(|filter| {
            crate::db::collaboration::event_repository::EventRepositoryQuery::new(
                vec![filter.event_filter().clone()],
                filter.limit(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(queries) => queries,
        Err(_) => return profile_http_failure(StatusCode::BAD_REQUEST, "invalid profile query"),
    };
    let records = match state
        .profiles
        .query_authored_profiles(&tenant, &principal, &queries, now_millis)
        .await
    {
        Ok(records) => records,
        Err(error) => return profile_directory_failure(error),
    };
    let events = records.iter().map(|record| {
        let signed = record.signed_event();
        serde_json::json!({"id":signed.claimed_id.to_hex(),"pubkey":signed.event.public_key.to_hex(),
            "created_at":signed.event.created_at,"kind":signed.event.kind,"tags":signed.event.tags,
            "content":signed.event.content,"sig":signed.signature.to_hex()})
    }).collect::<Vec<_>>();
    Json(events).into_response()
}

pub async fn closed_websocket_root_response(
    state: Option<&Arc<NostrProfileHttpState>>,
    headers: &HeaderMap,
    upgrade: Option<axum::extract::ws::WebSocketUpgrade>,
) -> Option<Response> {
    let upgrade = upgrade?;
    let Some(state) = state.filter(|state| state.closed_websocket).cloned() else {
        return Some(StatusCode::NOT_FOUND.into_response());
    };
    let tenant = match resolve_request_tenant(&state.host_resolver, headers).await {
        Some(tenant) => tenant,
        None => return Some(StatusCode::NOT_FOUND.into_response()),
    };
    Some(
        upgrade
            .max_message_size(MAX_NOSTR_FRAME_BYTES)
            .max_frame_size(MAX_NOSTR_FRAME_BYTES)
            .on_upgrade(move |socket| async move {
                if run_closed_nostr_socket(state, tenant, socket)
                    .await
                    .is_err()
                {
                    tracing::warn!("closed Nostr connection failed");
                }
            })
            .into_response(),
    )
}

fn nostr_socket_now() -> anyhow::Result<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs())
}

async fn send_nostr_socket(
    socket: &mut axum::extract::ws::WebSocket,
    message: axum::extract::ws::Message,
) -> anyhow::Result<()> {
    tokio::time::timeout(std::time::Duration::from_secs(5), socket.send(message))
        .await
        .map_err(|_| anyhow::anyhow!("socket send timed out"))??;
    Ok(())
}

async fn run_closed_nostr_socket(
    state: Arc<NostrProfileHttpState>,
    tenant: TenantContext,
    mut socket: axum::extract::ws::WebSocket,
) -> anyhow::Result<()> {
    use crate::nostr::auth::{
        ClosedNostrPrincipalResolver, NostrAuthChallenge, NostrAuthProtocolDisposition,
        NostrAuthenticationState, NostrConnectionAuthenticator, NostrStructuralBanResolver,
    };
    use axum::extract::ws::Message;
    if !matches!(
        &state.socket_connection,
        sea_orm::DatabaseConnection::SqlxPostgresPoolConnection(_)
    ) {
        anyhow::bail!("closed Nostr socket requires PostgreSQL");
    }
    let pool = state.socket_connection.get_postgres_connection_pool();
    let resolver = NostrStructuralBanResolver::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
        ClosedNostrPrincipalResolver::new(sea_orm::DatabaseConnection::from(pool.clone()))?,
    )?;
    let relay_url = format!(
        "{}://{}",
        if state.public_tls { "wss" } else { "ws" },
        tenant.route_reference()
    );
    let mut authenticator = NostrConnectionAuthenticator::new(
        tenant.clone(),
        relay_url,
        NostrAuthChallenge::generate(),
        nostr_socket_now()?,
        state.socket_replay.clone(),
        resolver,
    )?;
    let ingress = crate::nostr::event_ingest::NostrEventIngress::new(
        crate::messages::channel_service::NostrChannelCommandSink::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?,
        crate::nostr::ingress::NostrIngressDeployment::InProcess,
    );
    let challenge = authenticator
        .challenge_frame()
        .ok_or_else(|| anyhow::anyhow!("missing AUTH challenge"))?;
    send_nostr_socket(&mut socket, Message::Text(challenge)).await?;
    loop {
        let message = tokio::select! {
            message = socket.recv() => message,
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                if authenticator.expire(nostr_socket_now()?) { break; }
                continue;
            }
        };
        let Some(message) = message else {
            return Ok(());
        };
        let raw = match message? {
            Message::Text(raw) if raw.len() <= MAX_NOSTR_FRAME_BYTES => raw,
            Message::Ping(payload) => {
                send_nostr_socket(&mut socket, Message::Pong(payload)).await?;
                continue;
            }
            Message::Pong(_) => continue,
            Message::Close(_) => return Ok(()),
            _ => break,
        };
        let frame: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(frame) => frame,
            Err(_) => {
                send_nostr_socket(
                    &mut socket,
                    Message::Text(
                        serde_json::json!(["NOTICE", "invalid: malformed frame"]).to_string(),
                    ),
                )
                .await?;
                continue;
            }
        };
        let Some(parts) = frame.as_array().filter(|parts| parts.len() == 2) else {
            send_nostr_socket(
                &mut socket,
                Message::Text(
                    serde_json::json!(["NOTICE", "invalid: unsupported frame"]).to_string(),
                ),
            )
            .await?;
            continue;
        };
        let Some(payload) = parts.get(1) else {
            continue;
        };
        match parts.first().and_then(serde_json::Value::as_str) {
            Some("AUTH") => {
                let signed = match parse_wire_signed_event(payload.clone()) {
                    Ok((signed, _wire)) => signed,
                    Err(_) => {
                        send_nostr_socket(
                            &mut socket,
                            Message::Text(
                                serde_json::json!(["NOTICE", "invalid: malformed AUTH"])
                                    .to_string(),
                            ),
                        )
                        .await?;
                        continue;
                    }
                };
                let event_id = signed.claimed_id.to_hex();
                let (accepted, reason, close) = match authenticator
                    .authenticate(&signed, nostr_socket_now()?)
                    .await
                {
                    Ok(_) => (true, "", false),
                    Err(error) => match error.protocol_disposition() {
                        NostrAuthProtocolDisposition::Reject {
                            reason,
                            close_connection,
                        } => (false, reason, close_connection),
                        NostrAuthProtocolDisposition::CloseWithoutResponse => break,
                    },
                };
                send_nostr_socket(
                    &mut socket,
                    Message::Text(
                        serde_json::json!(["OK", event_id, accepted, reason]).to_string(),
                    ),
                )
                .await?;
                if close {
                    break;
                }
            }
            Some("EVENT") => {
                let NostrAuthenticationState::Authenticated(principal) = authenticator.state()
                else {
                    send_nostr_socket(
                        &mut socket,
                        Message::Text(
                            serde_json::json!(["NOTICE", "auth-required: authenticate first"])
                                .to_string(),
                        ),
                    )
                    .await?;
                    continue;
                };
                let signed = match parse_wire_signed_event(payload.clone()) {
                    Ok((signed, _wire)) => signed,
                    Err(_) => {
                        send_nostr_socket(
                            &mut socket,
                            Message::Text(
                                serde_json::json!(["NOTICE", "invalid: malformed EVENT"])
                                    .to_string(),
                            ),
                        )
                        .await?;
                        continue;
                    }
                };
                let event_id = signed.claimed_id.to_hex();
                if !matches!(
                    u32::from(signed.event.kind),
                    nostr_compat::generated_kinds::KIND_MODERATION_BAN
                        | nostr_compat::generated_kinds::KIND_MODERATION_UNBAN
                        | nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT
                        | nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT
                        | nostr_compat::generated_kinds::KIND_NIP29_PUT_USER
                        | nostr_compat::generated_kinds::KIND_AGENT_PROFILE
                        | nostr_compat::generated_kinds::KIND_STREAM_MESSAGE
                        | nostr_compat::generated_kinds::KIND_DELETION
                ) {
                    send_nostr_socket(
                        &mut socket,
                        Message::Text(
                            serde_json::json!([
                                "OK",
                                event_id,
                                false,
                                "invalid: unsupported event kind"
                            ])
                            .to_string(),
                        ),
                    )
                    .await?;
                    continue;
                }
                let now = nostr_socket_now()?;
                let admission = if matches!(
                    u32::from(signed.event.kind),
                    nostr_compat::generated_kinds::KIND_MODERATION_BAN
                        | nostr_compat::generated_kinds::KIND_MODERATION_UNBAN
                        | nostr_compat::generated_kinds::KIND_MODERATION_TIMEOUT
                        | nostr_compat::generated_kinds::KIND_MODERATION_UNTIMEOUT
                ) {
                    crate::admin::moderation::admit_signed_restriction_transport(
                        &state.socket_connection,
                        &state.profiles,
                        &tenant,
                        principal,
                        now,
                    )
                    .await
                } else {
                    crate::messages::channel_admission::admit_nostr_channel_transport(
                        &state.socket_connection,
                        &state.profiles,
                        &tenant,
                        principal,
                        signed.event.kind,
                        now,
                    )
                    .await
                    .map_err(|error| {
                        if crate::messages::channel_admission::channel_command_error(error) == crate::collaboration_command::DomainCommandSubmissionError::Unavailable {
                            crate::nostr::auth::NostrPrincipalResolutionError::Unavailable
                        } else {
                            crate::nostr::auth::NostrPrincipalResolutionError::Denied
                        }
                    })
                };
                let admission = match admission {
                    Ok(admission) => admission,
                    Err(error) => {
                        let reason = if error
                            == crate::nostr::auth::NostrPrincipalResolutionError::Unavailable
                        {
                            "error: internal"
                        } else {
                            "restricted: transport admission denied"
                        };
                        send_nostr_socket(
                            &mut socket,
                            Message::Text(
                                serde_json::json!(["OK", event_id, false, reason]).to_string(),
                            ),
                        )
                        .await?;
                        continue;
                    }
                };
                let outcome = ingress.handle_frame(admission, &raw, now).await?;
                send_nostr_socket(&mut socket, Message::Text(outcome.frame().to_owned())).await?;
            }
            _ => {
                send_nostr_socket(
                    &mut socket,
                    Message::Text(
                        serde_json::json!(["NOTICE", "invalid: unsupported frame"]).to_string(),
                    ),
                )
                .await?
            }
        }
    }
    send_nostr_socket(&mut socket, Message::Close(None)).await
}
