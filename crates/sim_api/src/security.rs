use crate::http::MutationIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_model::ArtifactRoot;
use sim_runtime::{
    AuthorizedCapabilities, Capability, CapabilitySet, NativeApiExposure, PermissionGrant,
    PermissionPolicy, RemoteExposureApproval,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    net::IpAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

const IDEMPOTENCY_SNAPSHOT_VERSION: u16 = 1;
const MAX_IDEMPOTENCY_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TlsPolicy {
    Disabled,
    Required { certificate_identity: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RequestLimits {
    pub maximum_body_bytes: usize,
    pub maximum_header_bytes: usize,
    pub maximum_header_count: usize,
    pub maximum_concurrent_requests: usize,
    pub requests_per_minute: u32,
    pub maximum_rate_identities: usize,
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            maximum_body_bytes: 16 * 1024 * 1024,
            maximum_header_bytes: 64 * 1024,
            maximum_header_count: 128,
            maximum_concurrent_requests: 64,
            requests_per_minute: 600,
            maximum_rate_identities: 4_096,
        }
    }
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BearerCredential {
    pub principal: String,
    secret_digest: [u8; 32],
    pub scopes: BTreeSet<String>,
    pub expires_at_epoch_seconds: Option<u64>,
}

impl BearerCredential {
    pub fn new(
        principal: impl Into<String>,
        secret: impl Into<String>,
        scopes: impl IntoIterator<Item = String>,
        expires_at_epoch_seconds: Option<u64>,
    ) -> Result<Self, ApiSecurityError> {
        let principal = principal.into();
        let secret = secret.into();
        if principal.trim().is_empty() || secret.len() < 16 || secret.len() > 4096 {
            return Err(ApiSecurityError::InvalidConfiguration(
                "bearer credentials require a principal and a 16-4096 byte secret".into(),
            ));
        }
        Ok(Self {
            principal,
            secret_digest: Sha256::digest(secret.as_bytes()).into(),
            scopes: scopes.into_iter().collect(),
            expires_at_epoch_seconds,
        })
    }

    fn matches(&self, candidate: &[u8]) -> bool {
        let candidate_digest: [u8; 32] = Sha256::digest(candidate).into();
        constant_time_equals(&self.secret_digest, &candidate_digest)
    }
}

impl fmt::Debug for BearerCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BearerCredential")
            .field("principal", &self.principal)
            .field("secret_digest", &"[REDACTED]")
            .field("scopes", &self.scopes)
            .field("expires_at_epoch_seconds", &self.expires_at_epoch_seconds)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PluginRouteGrant {
    pub profile_id: String,
    pub principal: String,
    pub plugin_id: String,
    pub plugin_digest: String,
    pub methods: BTreeSet<String>,
    pub route_prefixes: BTreeSet<String>,
    pub capabilities: BTreeSet<String>,
}

impl PluginRouteGrant {
    fn capability_set(&self) -> Result<CapabilitySet, ApiSecurityError> {
        self.capabilities
            .iter()
            .map(|identifier| {
                Capability::parse_wire_identifier(identifier).map_err(|error| {
                    ApiSecurityError::InvalidConfiguration(format!(
                        "plugin route capability `{identifier}` is invalid: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(CapabilitySet::new)
    }

    fn subject_id(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"zed-sim-api-plugin-route-subject-v1\0");
        for component in [&self.principal, &self.plugin_id, &self.plugin_digest] {
            digest.update(
                u64::try_from(component.len())
                    .unwrap_or(u64::MAX)
                    .to_le_bytes(),
            );
            digest.update(component.as_bytes());
        }
        format!("api-plugin-route.{:x}", digest.finalize())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuthorizedPluginRoute {
    profile_id: String,
    principal: String,
    plugin_id: String,
    plugin_digest: String,
    methods: BTreeSet<String>,
    route_prefixes: BTreeSet<String>,
    required_capabilities: CapabilitySet,
    authorization: AuthorizedCapabilities,
}

impl AuthorizedPluginRoute {
    fn permits(
        &self,
        principal: &str,
        request: &PluginRouteRequest,
        method: &str,
        path: &str,
    ) -> bool {
        self.profile_id == request.profile_id
            && self.principal == principal
            && self.plugin_id == request.plugin_id
            && self.plugin_digest == request.plugin_digest
            && self.methods.contains(method)
            && request.required_capabilities == self.required_capabilities
            && request
                .required_capabilities
                .iter()
                .all(|capability| self.authorization.require(capability).is_ok())
            && self.route_prefixes.iter().any(|prefix| {
                path == prefix
                    || path
                        .strip_prefix(prefix)
                        .is_some_and(|suffix| suffix.starts_with('/'))
            })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginRouteRequest {
    pub profile_id: String,
    pub plugin_id: String,
    pub plugin_digest: String,
    pub required_capabilities: CapabilitySet,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ApiSecurityConfig {
    pub bind_address: IpAddr,
    pub explicit_remote_exposure: bool,
    pub trust_reverse_proxy: bool,
    pub trusted_reverse_proxies: BTreeSet<IpAddr>,
    pub tls: TlsPolicy,
    pub allowed_origins: BTreeSet<String>,
    pub require_authentication: bool,
    pub credentials: Vec<BearerCredential>,
    pub limits: RequestLimits,
    pub plugin_route_grants: Vec<PluginRouteGrant>,
}

impl ApiSecurityConfig {
    pub fn loopback() -> Self {
        Self {
            bind_address: IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            explicit_remote_exposure: false,
            trust_reverse_proxy: false,
            trusted_reverse_proxies: BTreeSet::new(),
            tls: TlsPolicy::Disabled,
            allowed_origins: BTreeSet::new(),
            require_authentication: false,
            credentials: Vec::new(),
            limits: RequestLimits::default(),
            plugin_route_grants: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ApiSecurityError> {
        let remote = !self.bind_address.is_loopback();
        if remote && !self.explicit_remote_exposure {
            return Err(ApiSecurityError::RemoteExposureNotAcknowledged);
        }
        if remote && !matches!(self.tls, TlsPolicy::Required { .. }) {
            return Err(ApiSecurityError::TlsRequired);
        }
        if matches!(
            &self.tls,
            TlsPolicy::Required {
                certificate_identity
            } if certificate_identity.trim().is_empty()
        ) {
            return Err(ApiSecurityError::InvalidConfiguration(
                "TLS certificate identity cannot be empty".into(),
            ));
        }
        if remote && (!self.require_authentication || self.credentials.is_empty()) {
            return Err(ApiSecurityError::AuthenticationRequired);
        }
        if remote && self.allowed_origins.is_empty() {
            return Err(ApiSecurityError::OriginDenied);
        }
        self.native_exposure()?;
        if self.trust_reverse_proxy && self.trusted_reverse_proxies.is_empty() {
            return Err(ApiSecurityError::InvalidConfiguration(
                "reverse-proxy trust requires at least one exact trusted peer".into(),
            ));
        }
        if self.limits.maximum_body_bytes == 0
            || self.limits.maximum_header_bytes == 0
            || self.limits.maximum_header_count == 0
            || self.limits.maximum_concurrent_requests == 0
            || self.limits.requests_per_minute == 0
            || self.limits.maximum_rate_identities == 0
        {
            return Err(ApiSecurityError::InvalidConfiguration(
                "request limits must be non-zero".into(),
            ));
        }
        for grant in &self.plugin_route_grants {
            if !is_exact_identifier(&grant.profile_id)
                || !is_exact_identifier(&grant.principal)
                || !is_exact_identifier(&grant.plugin_id)
                || !is_exact_identifier(&grant.plugin_digest)
                || grant.methods.is_empty()
                || grant
                    .methods
                    .iter()
                    .any(|method| !is_exact_identifier(method))
                || grant.route_prefixes.is_empty()
                || grant.route_prefixes.iter().any(|path| !is_safe_route(path))
            {
                return Err(ApiSecurityError::InvalidConfiguration(
                    "plugin route grants require an exact identity, method, and safe route prefix"
                        .into(),
                ));
            }
            grant.capability_set()?;
        }
        Ok(())
    }

    fn native_exposure(&self) -> Result<NativeApiExposure, ApiSecurityError> {
        NativeApiExposure::new(
            self.bind_address,
            matches!(self.tls, TlsPolicy::Required { .. }),
            self.allowed_origins.iter().cloned(),
            self.trust_reverse_proxy,
            if self.explicit_remote_exposure {
                RemoteExposureApproval::Approved
            } else {
                RemoteExposureApproval::LoopbackOnly
            },
        )
        .map_err(|error| {
            ApiSecurityError::InvalidConfiguration(format!(
                "native API exposure policy rejected the configuration: {error}"
            ))
        })
    }

    fn into_validated(mut self) -> Result<Self, ApiSecurityError> {
        self.validate()?;
        let exposure = self.native_exposure()?;
        self.allowed_origins = exposure.allowed_origins().clone();
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSecurityContext {
    pub method: String,
    pub canonical_path: String,
    pub body_bytes: usize,
    pub header_bytes: usize,
    pub header_count: usize,
    pub origin: Option<String>,
    pub authorization: Option<String>,
    pub peer_address: IpAddr,
    pub forwarded_for: Option<IpAddr>,
    pub transport_tls: bool,
    pub required_scope: Option<String>,
    pub plugin: Option<PluginRouteRequest>,
    pub mutation_identity: Option<MutationIdentity>,
    pub now_epoch_seconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreflightSecurityContext {
    pub canonical_path: String,
    pub header_bytes: usize,
    pub header_count: usize,
    pub origin: Option<String>,
    pub peer_address: IpAddr,
    pub forwarded_for: Option<IpAddr>,
    pub transport_tls: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorizedPreflight {
    pub effective_client_address: IpAddr,
    pub allowed_origin: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedPrincipal {
    pub identity: String,
    pub scopes: BTreeSet<String>,
}

#[derive(Debug)]
pub struct AuthorizedRequest {
    pub principal: AuthenticatedPrincipal,
    pub effective_client_address: IpAddr,
    pub allowed_origin: Option<String>,
    _permit: ConcurrencyPermit,
}

#[derive(Debug)]
struct RateEntry {
    minute: u64,
    requests: u32,
}

#[derive(Debug)]
pub struct ApiSecurityGate {
    config: ApiSecurityConfig,
    permission_policy: Arc<PermissionPolicy>,
    authorized_plugin_routes: Vec<AuthorizedPluginRoute>,
    active_requests: Arc<AtomicUsize>,
    rate_entries: Mutex<BTreeMap<String, RateEntry>>,
}

impl ApiSecurityGate {
    pub fn new(
        config: ApiSecurityConfig,
        permission_policy: Arc<PermissionPolicy>,
    ) -> Result<Self, ApiSecurityError> {
        let config = config.into_validated()?;
        let authorized_plugin_routes =
            authorize_plugin_routes(&config.plugin_route_grants, &permission_policy)?;
        Ok(Self {
            config,
            permission_policy,
            authorized_plugin_routes,
            active_requests: Arc::new(AtomicUsize::new(0)),
            rate_entries: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn config(&self) -> &ApiSecurityConfig {
        &self.config
    }

    pub fn permission_policy(&self) -> &Arc<PermissionPolicy> {
        &self.permission_policy
    }

    pub fn authorize(
        &self,
        request: &RequestSecurityContext,
    ) -> Result<AuthorizedRequest, ApiSecurityError> {
        if !is_safe_route(&request.canonical_path) {
            return Err(ApiSecurityError::UnsafePath);
        }
        if matches!(self.config.tls, TlsPolicy::Required { .. }) && !request.transport_tls {
            return Err(ApiSecurityError::TlsRequired);
        }
        if request.body_bytes > self.config.limits.maximum_body_bytes {
            return Err(ApiSecurityError::BodyTooLarge);
        }
        if request.header_bytes > self.config.limits.maximum_header_bytes
            || request.header_count > self.config.limits.maximum_header_count
        {
            return Err(ApiSecurityError::HeadersTooLarge);
        }

        let effective_client_address = self.effective_client_address(request)?;
        let allowed_origin = self.authorize_origin(request.origin.as_deref())?;
        let principal = self.authenticate(request)?;
        if let Some(required_scope) = &request.required_scope
            && !principal.scopes.contains(required_scope)
        {
            return Err(ApiSecurityError::ForbiddenScope(required_scope.clone()));
        }
        if let Some(plugin_request) = &request.plugin
            && !self.authorized_plugin_routes.iter().any(|grant| {
                grant.permits(
                    &principal.identity,
                    plugin_request,
                    request.method.as_str(),
                    request.canonical_path.as_str(),
                )
            })
        {
            return Err(ApiSecurityError::PluginRouteDenied);
        }
        if request.mutation_identity == Some(MutationIdentity::Untracked) {
            return Err(ApiSecurityError::MutationIdentityRequired);
        }
        self.apply_rate_limit(&principal.identity, effective_client_address, request)?;
        let permit = ConcurrencyPermit::acquire(
            self.active_requests.clone(),
            self.config.limits.maximum_concurrent_requests,
        )?;
        Ok(AuthorizedRequest {
            principal,
            effective_client_address,
            allowed_origin,
            _permit: permit,
        })
    }

    pub fn authorize_preflight(
        &self,
        request: &PreflightSecurityContext,
    ) -> Result<AuthorizedPreflight, ApiSecurityError> {
        if !is_safe_route(&request.canonical_path) {
            return Err(ApiSecurityError::UnsafePath);
        }
        if matches!(self.config.tls, TlsPolicy::Required { .. }) && !request.transport_tls {
            return Err(ApiSecurityError::TlsRequired);
        }
        if request.header_bytes > self.config.limits.maximum_header_bytes
            || request.header_count > self.config.limits.maximum_header_count
        {
            return Err(ApiSecurityError::HeadersTooLarge);
        }
        let effective_client_address =
            self.effective_client_address_parts(request.peer_address, request.forwarded_for)?;
        let origin = request
            .origin
            .as_deref()
            .ok_or(ApiSecurityError::OriginRequired)?;
        let allowed_origin = self
            .authorize_origin(Some(origin))?
            .ok_or(ApiSecurityError::OriginRequired)?;
        Ok(AuthorizedPreflight {
            effective_client_address,
            allowed_origin,
        })
    }

    fn effective_client_address(
        &self,
        request: &RequestSecurityContext,
    ) -> Result<IpAddr, ApiSecurityError> {
        self.effective_client_address_parts(request.peer_address, request.forwarded_for)
    }

    fn effective_client_address_parts(
        &self,
        peer_address: IpAddr,
        forwarded_for: Option<IpAddr>,
    ) -> Result<IpAddr, ApiSecurityError> {
        match forwarded_for {
            None => Ok(peer_address),
            Some(forwarded) => {
                if !self.config.trust_reverse_proxy
                    || !self.config.trusted_reverse_proxies.contains(&peer_address)
                {
                    return Err(ApiSecurityError::UntrustedForwardedAddress);
                }
                Ok(forwarded)
            }
        }
    }

    fn authorize_origin(&self, origin: Option<&str>) -> Result<Option<String>, ApiSecurityError> {
        match origin {
            None => Ok(None),
            Some(origin) if self.config.allowed_origins.contains(origin) => {
                Ok(Some(origin.to_owned()))
            }
            Some(_) => Err(ApiSecurityError::OriginDenied),
        }
    }

    fn authenticate(
        &self,
        request: &RequestSecurityContext,
    ) -> Result<AuthenticatedPrincipal, ApiSecurityError> {
        let authorization = request.authorization.as_deref();
        if authorization.is_none() && !self.config.require_authentication {
            return Ok(AuthenticatedPrincipal {
                identity: format!("anonymous:{}", request.peer_address),
                scopes: BTreeSet::new(),
            });
        }
        let token = authorization
            .and_then(|value| value.strip_prefix("Bearer "))
            .ok_or(ApiSecurityError::AuthenticationRequired)?;
        let credential = self
            .config
            .credentials
            .iter()
            .find(|credential| credential.matches(token.as_bytes()))
            .ok_or(ApiSecurityError::InvalidCredential)?;
        if credential
            .expires_at_epoch_seconds
            .is_some_and(|expiry| request.now_epoch_seconds >= expiry)
        {
            return Err(ApiSecurityError::ExpiredCredential);
        }
        Ok(AuthenticatedPrincipal {
            identity: credential.principal.clone(),
            scopes: credential.scopes.clone(),
        })
    }

    fn apply_rate_limit(
        &self,
        principal: &str,
        address: IpAddr,
        request: &RequestSecurityContext,
    ) -> Result<(), ApiSecurityError> {
        let minute = request.now_epoch_seconds / 60;
        let key = format!("{principal}@{address}");
        let mut entries = self
            .rate_entries
            .lock()
            .map_err(|_| ApiSecurityError::SecurityStateUnavailable)?;
        entries.retain(|_, entry| entry.minute.saturating_add(1) >= minute);
        if !entries.contains_key(&key)
            && entries.len() >= self.config.limits.maximum_rate_identities
        {
            return Err(ApiSecurityError::RateLimited);
        }
        let entry = entries.entry(key).or_insert(RateEntry {
            minute,
            requests: 0,
        });
        if entry.minute != minute {
            entry.minute = minute;
            entry.requests = 0;
        }
        if entry.requests >= self.config.limits.requests_per_minute {
            return Err(ApiSecurityError::RateLimited);
        }
        entry.requests = entry.requests.saturating_add(1);
        Ok(())
    }
}

pub fn plugin_route_permission_grants(
    grants: &[PluginRouteGrant],
) -> Result<Vec<PermissionGrant>, ApiSecurityError> {
    let mut capabilities_by_subject = BTreeMap::<(String, String), Vec<Capability>>::new();
    for grant in grants {
        let key = (grant.profile_id.clone(), grant.subject_id());
        capabilities_by_subject
            .entry(key)
            .or_default()
            .extend(grant.capability_set()?.iter().cloned());
    }

    capabilities_by_subject
        .into_iter()
        .map(|((profile_id, subject_id), capabilities)| {
            PermissionGrant::new(
                profile_id,
                subject_id,
                CapabilitySet::new(capabilities),
                "native-api-plugin-route-config",
            )
            .map_err(permission_configuration_error)
        })
        .collect()
}

fn authorize_plugin_routes(
    grants: &[PluginRouteGrant],
    permission_policy: &PermissionPolicy,
) -> Result<Vec<AuthorizedPluginRoute>, ApiSecurityError> {
    grants
        .iter()
        .map(|grant| {
            let capabilities = grant.capability_set()?;
            if permission_policy.profile_id() != grant.profile_id {
                return Err(ApiSecurityError::InvalidConfiguration(
                    "plugin route permission profile does not match the canonical policy".into(),
                ));
            }
            let authorization = permission_policy
                .authorize(&grant.subject_id(), &capabilities)
                .map_err(permission_configuration_error)?;
            Ok(AuthorizedPluginRoute {
                profile_id: grant.profile_id.clone(),
                principal: grant.principal.clone(),
                plugin_id: grant.plugin_id.clone(),
                plugin_digest: grant.plugin_digest.clone(),
                methods: grant.methods.clone(),
                route_prefixes: grant.route_prefixes.clone(),
                required_capabilities: capabilities,
                authorization,
            })
        })
        .collect()
}

fn permission_configuration_error(error: impl fmt::Display) -> ApiSecurityError {
    ApiSecurityError::InvalidConfiguration(format!(
        "plugin route permission configuration is invalid: {error}"
    ))
}

#[derive(Debug)]
struct ConcurrencyPermit {
    active_requests: Arc<AtomicUsize>,
}

impl ConcurrencyPermit {
    fn acquire(
        active_requests: Arc<AtomicUsize>,
        maximum: usize,
    ) -> Result<Self, ApiSecurityError> {
        let acquired =
            active_requests.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < maximum).then_some(current + 1)
            });
        acquired.map_err(|_| ApiSecurityError::TooManyConcurrentRequests)?;
        Ok(Self { active_requests })
    }
}

impl Drop for ConcurrencyPermit {
    fn drop(&mut self) {
        self.active_requests.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IdempotencyState {
    Prepared,
    Committed { status: u16, response_body: Vec<u8> },
    Failed { status: u16, response_body: Vec<u8> },
    Interrupted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IdempotencyRecord {
    pub key: String,
    pub operation_id: String,
    pub request_digest: String,
    pub state: IdempotencyState,
    pub updated_at_epoch_seconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IdempotencySnapshot {
    pub version: u16,
    pub profile_id: String,
    pub records: Vec<IdempotencyRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IdempotencyDecision {
    Begin,
    Replay { status: u16, response_body: Vec<u8> },
    Pending { operation_id: String },
    Reconcile { operation_id: String },
}

#[derive(Debug)]
pub struct IdempotencyLedger {
    profile_id: String,
    maximum_records: usize,
    maximum_response_bytes: usize,
    records: BTreeMap<String, IdempotencyRecord>,
}

pub trait IdempotencySnapshotStore: Send + Sync {
    fn load(&self) -> Result<Option<IdempotencySnapshot>, ApiSecurityError>;
    fn save(&self, snapshot: &IdempotencySnapshot) -> Result<(), ApiSecurityError>;
}

#[derive(Clone, Debug)]
pub struct ArtifactIdempotencySnapshotStore {
    root: ArtifactRoot,
    relative_path: PathBuf,
}

impl ArtifactIdempotencySnapshotStore {
    pub fn new(
        root: ArtifactRoot,
        relative_path: impl Into<PathBuf>,
    ) -> Result<Self, ApiSecurityError> {
        let relative_path = relative_path.into();
        if relative_path.file_name().is_none() {
            return Err(ApiSecurityError::InvalidConfiguration(
                "idempotency snapshot identity must identify a relative file".into(),
            ));
        }
        root.resolve_for_create(&relative_path)
            .map_err(|error| ApiSecurityError::Persistence(error.to_string()))?;
        Ok(Self {
            root,
            relative_path,
        })
    }

    pub fn from_directory(
        root_directory: &Path,
        relative_path: impl Into<PathBuf>,
    ) -> Result<Self, ApiSecurityError> {
        let root = ArtifactRoot::canonical_with_path_identity(
            "native-api-idempotency",
            "native-api-private-state",
            root_directory,
            ["json"],
        )
        .map_err(|error| ApiSecurityError::Persistence(error.to_string()))?;
        Self::new(root, relative_path)
    }
}

impl IdempotencySnapshotStore for ArtifactIdempotencySnapshotStore {
    fn load(&self) -> Result<Option<IdempotencySnapshot>, ApiSecurityError> {
        self.root
            .read_private_file(&self.relative_path, MAX_IDEMPOTENCY_SNAPSHOT_BYTES)
            .map_err(|error| ApiSecurityError::Persistence(error.to_string()))?
            .map(|bytes| {
                serde_json::from_slice(&bytes)
                    .map_err(|_| ApiSecurityError::InvalidIdempotencySnapshot)
            })
            .transpose()
    }

    fn save(&self, snapshot: &IdempotencySnapshot) -> Result<(), ApiSecurityError> {
        let bytes = serde_json::to_vec(snapshot).map_err(|error| {
            ApiSecurityError::Persistence(format!("failed to encode idempotency snapshot: {error}"))
        })?;
        if bytes.len() > MAX_IDEMPOTENCY_SNAPSHOT_BYTES {
            return Err(ApiSecurityError::InvalidIdempotencySnapshot);
        }
        self.root
            .write_private_file(&self.relative_path, &bytes)
            .map_err(|error| ApiSecurityError::Persistence(error.to_string()))
    }
}

impl IdempotencyLedger {
    pub fn new(
        profile_id: impl Into<String>,
        maximum_records: usize,
        maximum_response_bytes: usize,
    ) -> Result<Self, ApiSecurityError> {
        let profile_id = profile_id.into();
        if profile_id.is_empty() || maximum_records == 0 || maximum_response_bytes == 0 {
            return Err(ApiSecurityError::InvalidConfiguration(
                "idempotency ledgers require a profile and non-zero bounds".into(),
            ));
        }
        Ok(Self {
            profile_id,
            maximum_records,
            maximum_response_bytes,
            records: BTreeMap::new(),
        })
    }

    pub fn restore(
        snapshot: IdempotencySnapshot,
        maximum_records: usize,
        maximum_response_bytes: usize,
    ) -> Result<Self, ApiSecurityError> {
        if snapshot.version != IDEMPOTENCY_SNAPSHOT_VERSION
            || snapshot.profile_id.is_empty()
            || snapshot.records.len() > maximum_records
        {
            return Err(ApiSecurityError::InvalidIdempotencySnapshot);
        }
        let mut ledger = Self::new(snapshot.profile_id, maximum_records, maximum_response_bytes)?;
        for mut record in snapshot.records {
            if record.key.is_empty()
                || record.operation_id.is_empty()
                || record.request_digest.is_empty()
                || response_length(&record.state) > maximum_response_bytes
                || ledger.records.contains_key(&record.key)
            {
                return Err(ApiSecurityError::InvalidIdempotencySnapshot);
            }
            if matches!(record.state, IdempotencyState::Prepared) {
                record.state = IdempotencyState::Interrupted;
            }
            ledger.records.insert(record.key.clone(), record);
        }
        Ok(ledger)
    }

    pub fn snapshot(&self) -> IdempotencySnapshot {
        IdempotencySnapshot {
            version: IDEMPOTENCY_SNAPSHOT_VERSION,
            profile_id: self.profile_id.clone(),
            records: self.records.values().cloned().collect(),
        }
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn begin(
        &mut self,
        key: impl Into<String>,
        operation_id: impl Into<String>,
        request_digest: impl Into<String>,
        now_epoch_seconds: u64,
    ) -> Result<IdempotencyDecision, ApiSecurityError> {
        let key = key.into();
        let operation_id = operation_id.into();
        let request_digest = request_digest.into();
        if key.is_empty() || operation_id.is_empty() || request_digest.is_empty() {
            return Err(ApiSecurityError::MutationIdentityRequired);
        }
        if let Some(record) = self.records.get(&key) {
            if record.request_digest != request_digest || record.operation_id != operation_id {
                return Err(ApiSecurityError::IdempotencyConflict);
            }
            return Ok(match &record.state {
                IdempotencyState::Committed {
                    status,
                    response_body,
                }
                | IdempotencyState::Failed {
                    status,
                    response_body,
                } => IdempotencyDecision::Replay {
                    status: *status,
                    response_body: response_body.clone(),
                },
                IdempotencyState::Prepared => IdempotencyDecision::Pending {
                    operation_id: record.operation_id.clone(),
                },
                IdempotencyState::Interrupted => IdempotencyDecision::Reconcile {
                    operation_id: record.operation_id.clone(),
                },
            });
        }
        if self.records.len() >= self.maximum_records {
            let removable = self
                .records
                .values()
                .filter(|record| {
                    matches!(
                        record.state,
                        IdempotencyState::Committed { .. } | IdempotencyState::Failed { .. }
                    )
                })
                .min_by_key(|record| record.updated_at_epoch_seconds)
                .map(|record| record.key.clone())
                .ok_or(ApiSecurityError::IdempotencyLedgerFull)?;
            self.records.remove(&removable);
        }
        self.records.insert(
            key.clone(),
            IdempotencyRecord {
                key,
                operation_id,
                request_digest,
                state: IdempotencyState::Prepared,
                updated_at_epoch_seconds: now_epoch_seconds,
            },
        );
        Ok(IdempotencyDecision::Begin)
    }

    pub fn complete(
        &mut self,
        key: &str,
        state: IdempotencyState,
        now_epoch_seconds: u64,
    ) -> Result<(), ApiSecurityError> {
        if matches!(state, IdempotencyState::Prepared)
            || response_length(&state) > self.maximum_response_bytes
        {
            return Err(ApiSecurityError::InvalidIdempotencyTransition);
        }
        let record = self
            .records
            .get_mut(key)
            .ok_or(ApiSecurityError::UnknownIdempotencyKey)?;
        if !matches!(
            record.state,
            IdempotencyState::Prepared | IdempotencyState::Interrupted
        ) {
            return Err(ApiSecurityError::InvalidIdempotencyTransition);
        }
        record.state = state;
        record.updated_at_epoch_seconds = now_epoch_seconds;
        Ok(())
    }

    pub fn reopen_after_not_applied(
        &mut self,
        key: &str,
        now_epoch_seconds: u64,
    ) -> Result<(), ApiSecurityError> {
        let record = self
            .records
            .get_mut(key)
            .ok_or(ApiSecurityError::UnknownIdempotencyKey)?;
        if !matches!(record.state, IdempotencyState::Interrupted) {
            return Err(ApiSecurityError::InvalidIdempotencyTransition);
        }
        record.state = IdempotencyState::Prepared;
        record.updated_at_epoch_seconds = now_epoch_seconds;
        Ok(())
    }
}

fn response_length(state: &IdempotencyState) -> usize {
    match state {
        IdempotencyState::Committed { response_body, .. }
        | IdempotencyState::Failed { response_body, .. } => response_body.len(),
        IdempotencyState::Prepared | IdempotencyState::Interrupted => 0,
    }
}

fn is_safe_route(path: &str) -> bool {
    if !path.starts_with('/')
        || path.contains('\0')
        || path.contains('\\')
        || path.contains('?')
        || path.contains('#')
    {
        return false;
    }
    let lowercase = path.to_ascii_lowercase();
    if lowercase.contains("%2e") || lowercase.contains("%2f") || lowercase.contains("%5c") {
        return false;
    }
    !path
        .split('/')
        .any(|segment| segment == "." || segment == "..")
}

fn is_exact_identifier(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= 4_096
        && !value.bytes().any(|byte| byte.is_ascii_control())
}

fn constant_time_equals(expected: &[u8], candidate: &[u8]) -> bool {
    let mut difference = expected.len() ^ candidate.len();
    let maximum_length = expected.len().max(candidate.len());
    for index in 0..maximum_length {
        let expected_byte = expected.get(index).copied().unwrap_or_default();
        let candidate_byte = candidate.get(index).copied().unwrap_or_default();
        difference |= usize::from(expected_byte ^ candidate_byte);
    }
    difference == 0
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ApiSecurityError {
    #[error("invalid native API security configuration: {0}")]
    InvalidConfiguration(String),
    #[error("remote API exposure requires explicit acknowledgement")]
    RemoteExposureNotAcknowledged,
    #[error("TLS is required for non-loopback API exposure")]
    TlsRequired,
    #[error("authentication is required")]
    AuthenticationRequired,
    #[error("the supplied credential is invalid")]
    InvalidCredential,
    #[error("the supplied credential has expired")]
    ExpiredCredential,
    #[error("the request origin is not permitted")]
    OriginDenied,
    #[error("CORS preflight requires an Origin header")]
    OriginRequired,
    #[error("the requested authorization scope is not granted: {0}")]
    ForbiddenScope(String),
    #[error("the forwarded client address came from an untrusted peer")]
    UntrustedForwardedAddress,
    #[error("the request body exceeds the configured bound")]
    BodyTooLarge,
    #[error("the request headers exceed the configured bound")]
    HeadersTooLarge,
    #[error("the request path is unsafe")]
    UnsafePath,
    #[error("the request rate limit was exceeded")]
    RateLimited,
    #[error("the concurrent request limit was exceeded")]
    TooManyConcurrentRequests,
    #[error("the security state is temporarily unavailable")]
    SecurityStateUnavailable,
    #[error("the plugin route is not covered by an exact signed grant")]
    PluginRouteDenied,
    #[error("a mutation requires an idempotency key or durable operation identity")]
    MutationIdentityRequired,
    #[error("the idempotency key was reused for a different request or operation")]
    IdempotencyConflict,
    #[error("the idempotency ledger is full of in-flight operations")]
    IdempotencyLedgerFull,
    #[error("the idempotency snapshot is malformed or incompatible")]
    InvalidIdempotencySnapshot,
    #[error("the idempotency transition is not legal")]
    InvalidIdempotencyTransition,
    #[error("the idempotency key is unknown")]
    UnknownIdempotencyKey,
    #[error("native API persistence failed: {0}")]
    Persistence(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        net::{Ipv4Addr, Ipv6Addr},
    };

    fn credential() -> BearerCredential {
        BearerCredential::new(
            "automation",
            "0123456789abcdef0123456789abcdef",
            ["queue:write".to_owned()],
            None,
        )
        .expect("valid test credential")
    }

    fn permission_policy(config: &ApiSecurityConfig) -> Arc<PermissionPolicy> {
        let grants = plugin_route_permission_grants(&config.plugin_route_grants)
            .expect("valid plugin route grants");
        Arc::new(
            PermissionPolicy::native_runtime_services("profile-a")
                .expect("valid native permission policy")
                .with_additional_grants(grants)
                .expect("valid route grant extension"),
        )
    }

    fn request() -> RequestSecurityContext {
        RequestSecurityContext {
            method: "POST".into(),
            canonical_path: "/api/prompt".into(),
            body_bytes: 128,
            header_bytes: 256,
            header_count: 4,
            origin: None,
            authorization: None,
            peer_address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            forwarded_for: None,
            transport_tls: false,
            required_scope: None,
            plugin: None,
            mutation_identity: Some(MutationIdentity::IdempotencyKey("request-1".into())),
            now_epoch_seconds: 120,
        }
    }

    #[test]
    fn remote_exposure_requires_tls_cors_and_authentication() {
        let mut config = ApiSecurityConfig::loopback();
        config.bind_address = IpAddr::V6(Ipv6Addr::UNSPECIFIED);
        assert_eq!(
            config.validate(),
            Err(ApiSecurityError::RemoteExposureNotAcknowledged)
        );
        config.explicit_remote_exposure = true;
        assert_eq!(config.validate(), Err(ApiSecurityError::TlsRequired));
        config.tls = TlsPolicy::Required {
            certificate_identity: "api.example.test".into(),
        };
        assert_eq!(
            config.validate(),
            Err(ApiSecurityError::AuthenticationRequired)
        );
        config.require_authentication = true;
        config.credentials.push(credential());
        assert_eq!(config.validate(), Err(ApiSecurityError::OriginDenied));
        config
            .allowed_origins
            .insert("https://client.example.test".into());
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn authentication_scopes_origins_and_limits_fail_closed() {
        let mut config = ApiSecurityConfig::loopback();
        config.require_authentication = true;
        config.credentials.push(credential());
        config.allowed_origins.insert("https://client.test".into());
        config.limits.maximum_concurrent_requests = 1;
        let policy = permission_policy(&config);
        let gate = ApiSecurityGate::new(config, policy).expect("valid gate");
        let mut request = request();
        request.origin = Some("https://client.test".into());
        request.authorization = Some("Bearer 0123456789abcdef0123456789abcdef".into());
        request.required_scope = Some("queue:write".into());
        let permit = gate.authorize(&request).expect("authorized request");
        assert_eq!(
            gate.authorize(&request).expect_err("concurrency bound"),
            ApiSecurityError::TooManyConcurrentRequests
        );
        drop(permit);
        request.canonical_path = "/api/%2e%2e/secret".into();
        assert_eq!(
            gate.authorize(&request).expect_err("unsafe path"),
            ApiSecurityError::UnsafePath
        );
    }

    #[test]
    fn plugin_routes_require_exact_profile_digest_scope_and_prefix() {
        let provider = Capability::parse_wire_identifier(
            "provider_network:provider-a|https://provider-a.invalid/v1/run",
        )
        .expect("canonical provider capability");
        let mut config = ApiSecurityConfig::loopback();
        config.plugin_route_grants.push(PluginRouteGrant {
            profile_id: "profile-a".into(),
            principal: "anonymous:127.0.0.1".into(),
            plugin_id: "plugin-a".into(),
            plugin_digest: "sha256:1234".into(),
            methods: ["POST".into()].into_iter().collect(),
            route_prefixes: ["/plugin/a".into()].into_iter().collect(),
            capabilities: [provider.wire_identifier()].into_iter().collect(),
        });
        let policy = permission_policy(&config);
        let gate = ApiSecurityGate::new(config, policy).expect("valid gate");
        let mut request = request();
        request.canonical_path = "/plugin/a/run".into();
        request.plugin = Some(PluginRouteRequest {
            profile_id: "profile-a".into(),
            plugin_id: "plugin-a".into(),
            plugin_digest: "sha256:1234".into(),
            required_capabilities: CapabilitySet::new([provider]),
        });
        assert!(gate.authorize(&request).is_ok());
        request
            .plugin
            .as_mut()
            .expect("plugin request")
            .plugin_digest = "sha256:different".into();
        assert_eq!(
            gate.authorize(&request).expect_err("digest mismatch"),
            ApiSecurityError::PluginRouteDenied
        );
        request
            .plugin
            .as_mut()
            .expect("plugin request")
            .plugin_digest = "sha256:1234".into();
        request.peer_address = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));
        assert_eq!(
            gate.authorize(&request).expect_err("principal mismatch"),
            ApiSecurityError::PluginRouteDenied
        );
    }

    #[test]
    fn plugin_route_boundary_capabilities_are_canonical_and_checked() {
        let capability = Capability::ProviderNetwork {
            provider: "payments".into(),
            endpoint: "https://payments.invalid/v1/charge".into(),
        };
        let grant = PluginRouteGrant {
            profile_id: "profile-a".into(),
            principal: "anonymous:127.0.0.1".into(),
            plugin_id: "plugin-a".into(),
            plugin_digest: "sha256:1234".into(),
            methods: ["POST".into()].into_iter().collect(),
            route_prefixes: ["/plugin/a".into()].into_iter().collect(),
            capabilities: [capability.wire_identifier()].into_iter().collect(),
        };
        let encoded = serde_json::to_vec(&grant).expect("encode route boundary DTO");
        let decoded: PluginRouteGrant =
            serde_json::from_slice(&encoded).expect("decode route boundary DTO");
        assert_eq!(decoded, grant);
        assert_eq!(
            decoded
                .capability_set()
                .expect("parse canonical capability"),
            CapabilitySet::new([capability])
        );

        let mut malformed = ApiSecurityConfig::loopback();
        malformed.plugin_route_grants.push(PluginRouteGrant {
            capabilities: ["provider:payments".into()].into_iter().collect(),
            ..grant
        });
        let policy = Arc::new(
            PermissionPolicy::native_runtime_services("profile-a")
                .expect("valid native permission policy"),
        );
        assert!(matches!(
            ApiSecurityGate::new(malformed, policy),
            Err(ApiSecurityError::InvalidConfiguration(message))
                if message.contains("plugin route capability")
        ));
    }

    #[test]
    fn denied_plugin_capabilities_do_not_consume_rate_limit_capacity() {
        let granted = Capability::ProviderNetwork {
            provider: "payments".into(),
            endpoint: "https://payments.invalid/v1/charge".into(),
        };
        let mut config = ApiSecurityConfig::loopback();
        config.limits.requests_per_minute = 1;
        config.plugin_route_grants.push(PluginRouteGrant {
            profile_id: "profile-a".into(),
            principal: "anonymous:127.0.0.1".into(),
            plugin_id: "plugin-a".into(),
            plugin_digest: "sha256:1234".into(),
            methods: ["POST".into()].into_iter().collect(),
            route_prefixes: ["/plugin/a".into()].into_iter().collect(),
            capabilities: [granted.wire_identifier()].into_iter().collect(),
        });
        config.plugin_route_grants.push(PluginRouteGrant {
            profile_id: "profile-a".into(),
            principal: "anonymous:127.0.0.1".into(),
            plugin_id: "plugin-a".into(),
            plugin_digest: "sha256:1234".into(),
            methods: ["POST".into()].into_iter().collect(),
            route_prefixes: ["/plugin/b".into()].into_iter().collect(),
            capabilities: ["secret:secret.payments".into()].into_iter().collect(),
        });
        let policy = permission_policy(&config);
        let gate = ApiSecurityGate::new(config, policy).expect("valid gate");
        let mut request = request();
        request.canonical_path = "/plugin/a/run".into();
        request.plugin = Some(PluginRouteRequest {
            profile_id: "profile-a".into(),
            plugin_id: "plugin-a".into(),
            plugin_digest: "sha256:1234".into(),
            required_capabilities: CapabilitySet::default(),
        });
        assert_eq!(
            gate.authorize(&request)
                .expect_err("omitted route capability requirement"),
            ApiSecurityError::PluginRouteDenied
        );

        request
            .plugin
            .as_mut()
            .expect("plugin request")
            .required_capabilities = CapabilitySet::new([Capability::Secret {
            secret_id: "secret.payments".into(),
        }]);
        assert_eq!(
            gate.authorize(&request)
                .expect_err("untrusted capability request"),
            ApiSecurityError::PluginRouteDenied
        );

        request
            .plugin
            .as_mut()
            .expect("plugin request")
            .required_capabilities = CapabilitySet::new([granted]);
        assert!(gate.authorize(&request).is_ok());
    }

    #[test]
    fn durable_idempotency_replays_and_reconciles_ambiguous_mutations() {
        let mut ledger =
            IdempotencyLedger::new("profile-a", 2, 1024).expect("valid idempotency ledger");
        assert_eq!(
            ledger.begin("key-a", "operation-a", "digest-a", 1),
            Ok(IdempotencyDecision::Begin)
        );
        let snapshot = ledger.snapshot();
        let mut recovered = IdempotencyLedger::restore(snapshot, 2, 1024).expect("restored ledger");
        assert_eq!(
            recovered.begin("key-a", "operation-a", "digest-a", 2),
            Ok(IdempotencyDecision::Reconcile {
                operation_id: "operation-a".into()
            })
        );
        recovered
            .complete(
                "key-a",
                IdempotencyState::Committed {
                    status: 200,
                    response_body: br#"{"ok":true}"#.to_vec(),
                },
                3,
            )
            .expect("commit recovery");
        assert_eq!(
            recovered.begin("key-a", "operation-a", "digest-a", 4),
            Ok(IdempotencyDecision::Replay {
                status: 200,
                response_body: br#"{"ok":true}"#.to_vec()
            })
        );
        assert_eq!(
            recovered.begin("key-a", "operation-a", "different", 5),
            Err(ApiSecurityError::IdempotencyConflict)
        );
    }

    #[test]
    fn interrupted_idempotency_records_cannot_be_evicted_by_new_work() {
        let mut ledger = IdempotencyLedger::new("profile-a", 1, 1024).expect("valid ledger");
        ledger
            .begin("key-a", "operation-a", "digest-a", 1)
            .expect("prepare first operation");
        let mut recovered = IdempotencyLedger::restore(ledger.snapshot(), 1, 1024)
            .expect("restore interrupted operation");
        assert_eq!(
            recovered.begin("key-b", "operation-b", "digest-b", 2),
            Err(ApiSecurityError::IdempotencyLedgerFull)
        );
        assert_eq!(
            recovered.begin("key-a", "operation-a", "digest-a", 3),
            Ok(IdempotencyDecision::Reconcile {
                operation_id: "operation-a".into()
            })
        );
    }

    #[test]
    fn artifact_snapshot_store_delegates_secure_atomic_replacement()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = std::env::temp_dir().join(format!(
            "sim-api-idempotency-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        if directory.try_exists()? {
            fs::remove_dir_all(&directory)?;
        }
        fs::create_dir(&directory)?;
        let store = ArtifactIdempotencySnapshotStore::from_directory(&directory, "ledger.json")?;
        let mut ledger = IdempotencyLedger::new("profile-a", 2, 1024)?;
        assert_eq!(
            ledger.begin("key-a", "operation-a", "digest-a", 1)?,
            IdempotencyDecision::Begin
        );
        let first = ledger.snapshot();
        store.save(&first)?;
        assert_eq!(store.load()?, Some(first));

        ledger.complete(
            "key-a",
            IdempotencyState::Committed {
                status: 200,
                response_body: br#"{"ok":true}"#.to_vec(),
            },
            2,
        )?;
        let replacement = ledger.snapshot();
        store.save(&replacement)?;
        assert_eq!(store.load()?, Some(replacement));
        assert_eq!(
            fs::read_dir(&directory)?.count(),
            1,
            "ArtifactRoot must clean its private temporary publication file"
        );

        assert!(
            ArtifactIdempotencySnapshotStore::from_directory(&directory, "../escape.json").is_err(),
            "the adapter must preserve ArtifactRoot traversal rejection"
        );
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn credential_debug_output_never_contains_the_secret() {
        let debug = format!("{:?}", credential());
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("0123456789abcdef"));
    }
}
