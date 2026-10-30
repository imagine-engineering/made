use std::{
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use collaboration_domain::{AuthorizationRequest, TenantContext};
use collaboration_workflow::definition::WorkflowTrigger;
use futures::{Stream, StreamExt};
use hmac::{Hmac, Mac};
use serde_json::{Map, Value as JsonValue, json};
use sha2::{Digest, Sha256};
use url::{Host, Url};

use super::{
    repository::{
        StoredWorkflowDefinition, WorkflowIdentity, WorkflowStoreOutcome, WorkflowTriggerKind,
    },
    triggers::{
        WorkflowRunClaimer, WorkflowTriggerAdmissionError, WorkflowTriggerAdmissionOutcome,
        WorkflowTriggerAdmissionStatus, run_request, validate_current_definition,
        validate_owner_authorization,
    },
};

pub const MAX_WEBHOOK_BODY_BYTES: usize = 1024 * 1024;
pub const WEBHOOK_TOTAL_TIMEOUT: Duration = Duration::from_secs(10);
pub const MAX_WEBHOOK_FIELDS: usize = 64;
pub const MAX_WEBHOOK_FIELD_NAME_BYTES: usize = 64;
pub const MAX_WEBHOOK_FIELD_VALUE_BYTES: usize = 4_096;
pub const MAX_WEBHOOK_IDEMPOTENCY_KEY_BYTES: usize = 128;
pub const MAX_WEBHOOK_TIMESTAMP_SKEW_MILLIS: u64 = 5 * 60 * 1_000;

const SIGNATURE_DOMAIN: &[u8] = b"zed-workflow-webhook-v1\0";
const MIN_WEBHOOK_SECRET_BYTES: usize = 32;
const MAX_WEBHOOK_SECRET_BYTES: usize = 128;
const MAX_CREDENTIAL_REFERENCE_BYTES: usize = 256;
const MAX_CREDENTIAL_VERSION_BYTES: usize = 64;

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebhookCredentialReference(String);

impl WebhookCredentialReference {
    pub fn new(value: impl Into<String>) -> Result<Self, WebhookAdmissionError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_CREDENTIAL_REFERENCE_BYTES
            || value.trim() != value
            || value.chars().any(char::is_control)
        {
            return Err(WebhookAdmissionError::InvalidAuthentication);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub struct ResolvedWebhookCredential {
    version: String,
    secret: Vec<u8>,
}

impl ResolvedWebhookCredential {
    pub fn new(version: impl Into<String>, secret: Vec<u8>) -> Result<Self, WebhookAdmissionError> {
        let version = version.into();
        if version.is_empty()
            || version.len() > MAX_CREDENTIAL_VERSION_BYTES
            || version.trim() != version
            || version.chars().any(char::is_control)
            || !(MIN_WEBHOOK_SECRET_BYTES..=MAX_WEBHOOK_SECRET_BYTES).contains(&secret.len())
        {
            return Err(WebhookAdmissionError::InvalidCredential);
        }
        Ok(Self { version, secret })
    }

    pub fn version(&self) -> &str {
        &self.version
    }
}

impl fmt::Debug for ResolvedWebhookCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedWebhookCredential")
            .field("version", &self.version)
            .field("secret", &"[REDACTED]")
            .finish()
    }
}

impl Drop for ResolvedWebhookCredential {
    fn drop(&mut self) {
        self.secret.fill(0);
    }
}

#[async_trait]
pub trait WebhookCredentialResolver: Send + Sync {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        workflow: WorkflowIdentity,
        reference: &WebhookCredentialReference,
    ) -> Result<ResolvedWebhookCredential, WebhookAdmissionError>;
}

#[derive(Clone, Eq, PartialEq)]
pub struct WebhookAuthentication {
    timestamp_millis: u64,
    idempotency_key: String,
    content_sha256: [u8; 32],
    signature: [u8; 32],
}

impl fmt::Debug for WebhookAuthentication {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WebhookAuthentication")
            .field("timestamp_millis", &self.timestamp_millis)
            .field("idempotency_key", &self.idempotency_key)
            .field("content_sha256", &hex::encode(self.content_sha256))
            .field("signature", &"[REDACTED]")
            .finish()
    }
}

impl WebhookAuthentication {
    pub fn new(
        timestamp_millis: u64,
        idempotency_key: impl Into<String>,
        content_sha256_hex: &str,
        signature_hex: &str,
    ) -> Result<Self, WebhookAdmissionError> {
        let idempotency_key = idempotency_key.into();
        if timestamp_millis == 0 || !valid_idempotency_key(&idempotency_key) {
            return Err(WebhookAdmissionError::InvalidAuthentication);
        }
        Ok(Self {
            timestamp_millis,
            idempotency_key,
            content_sha256: decode_sha256(content_sha256_hex)?,
            signature: decode_sha256(signature_hex)?,
        })
    }

    pub const fn timestamp_millis(&self) -> u64 {
        self.timestamp_millis
    }

    pub fn idempotency_key(&self) -> &str {
        &self.idempotency_key
    }

    pub const fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }
}

impl Drop for WebhookAuthentication {
    fn drop(&mut self) {
        self.signature.fill(0);
    }
}

pub fn webhook_signature_v1(
    secret: &[u8],
    tenant: &TenantContext,
    workflow: WorkflowIdentity,
    timestamp_millis: u64,
    idempotency_key: &str,
    content_sha256: [u8; 32],
) -> Result<[u8; 32], WebhookAdmissionError> {
    if !(MIN_WEBHOOK_SECRET_BYTES..=MAX_WEBHOOK_SECRET_BYTES).contains(&secret.len())
        || timestamp_millis == 0
        || !valid_idempotency_key(idempotency_key)
        || workflow.community_id() != tenant.community_id()
    {
        return Err(WebhookAdmissionError::InvalidAuthentication);
    }
    let mut mac =
        HmacSha256::new_from_slice(secret).map_err(|_| WebhookAdmissionError::InvalidCredential)?;
    mac.update(&signature_payload(
        tenant,
        workflow,
        timestamp_millis,
        idempotency_key,
        content_sha256,
    ));
    Ok(mac.finalize().into_bytes().into())
}

#[derive(Clone, Copy, Debug)]
pub struct WebhookIngressLimits {
    total_timeout: Duration,
}

impl Default for WebhookIngressLimits {
    fn default() -> Self {
        Self {
            total_timeout: WEBHOOK_TOTAL_TIMEOUT,
        }
    }
}

impl WebhookIngressLimits {
    pub fn with_total_timeout(total_timeout: Duration) -> Result<Self, WebhookAdmissionError> {
        if total_timeout.is_zero() || total_timeout > WEBHOOK_TOTAL_TIMEOUT {
            return Err(WebhookAdmissionError::InvalidLimits);
        }
        Ok(Self { total_timeout })
    }
}

pub struct WorkflowWebhookAdmission<R, C> {
    run_claimer: R,
    credential_resolver: C,
    limits: WebhookIngressLimits,
}

impl<R, C> WorkflowWebhookAdmission<R, C>
where
    R: WorkflowRunClaimer,
    C: WebhookCredentialResolver,
{
    pub fn new(run_claimer: R, credential_resolver: C) -> Self {
        Self {
            run_claimer,
            credential_resolver,
            limits: WebhookIngressLimits::default(),
        }
    }

    pub fn with_limits(
        run_claimer: R,
        credential_resolver: C,
        limits: WebhookIngressLimits,
    ) -> Self {
        Self {
            run_claimer,
            credential_resolver,
            limits,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn admit<S>(
        &self,
        tenant: &TenantContext,
        definition: &StoredWorkflowDefinition,
        credential_reference: &WebhookCredentialReference,
        authentication: &WebhookAuthentication,
        content_length: Option<u64>,
        body: S,
        owner_authorization: &AuthorizationRequest<'_>,
        received_at_millis: u64,
    ) -> Result<WorkflowTriggerAdmissionOutcome, WebhookAdmissionError>
    where
        S: Stream<Item = Result<Bytes, WebhookBodyReadError>> + Unpin,
    {
        validate_webhook_definition(tenant, definition)?;
        if content_length.is_some_and(|length| length > MAX_WEBHOOK_BODY_BYTES as u64) {
            return Err(WebhookAdmissionError::BodyTooLarge);
        }
        if received_at_millis == 0
            || received_at_millis.abs_diff(authentication.timestamp_millis)
                > MAX_WEBHOOK_TIMESTAMP_SKEW_MILLIS
        {
            return Err(WebhookAdmissionError::StaleAuthentication);
        }

        let trigger_context = tokio::time::timeout(
            self.limits.total_timeout,
            self.authenticate_and_read(
                tenant,
                definition,
                credential_reference,
                authentication,
                body,
                owner_authorization,
            ),
        )
        .await
        .map_err(|_| WebhookAdmissionError::Timeout)??;

        let source_id = format!("webhook:{}", authentication.idempotency_key);
        let request = run_request(
            definition,
            WorkflowTriggerKind::Webhook,
            &source_id,
            trigger_context,
            authentication.timestamp_millis,
        )?;
        let run_identity = request.identity;
        let outcome = self.run_claimer.claim_run(tenant, &request).await?;
        Ok(WorkflowTriggerAdmissionOutcome {
            status: match outcome {
                WorkflowStoreOutcome::Applied => WorkflowTriggerAdmissionStatus::Claimed,
                WorkflowStoreOutcome::Duplicate => WorkflowTriggerAdmissionStatus::Duplicate,
            },
            run_identity: Some(run_identity),
        })
    }

    async fn authenticate_and_read<S>(
        &self,
        tenant: &TenantContext,
        definition: &StoredWorkflowDefinition,
        credential_reference: &WebhookCredentialReference,
        authentication: &WebhookAuthentication,
        body: S,
        owner_authorization: &AuthorizationRequest<'_>,
    ) -> Result<JsonValue, WebhookAdmissionError>
    where
        S: Stream<Item = Result<Bytes, WebhookBodyReadError>> + Unpin,
    {
        let credential = self
            .credential_resolver
            .resolve(tenant, definition.identity, credential_reference)
            .await?;
        verify_authentication(tenant, definition, authentication, &credential)?;
        validate_owner_authorization(tenant, definition, owner_authorization)?;
        let fields = read_webhook_body(body, authentication.content_sha256).await?;
        Ok(json!({
            "idempotency_key": authentication.idempotency_key,
            "timestamp": authentication.timestamp_millis.to_string(),
            "webhook": fields,
        }))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebhookRedirectPolicy {
    Reject,
}

#[async_trait]
pub trait WebhookDnsResolver: Send + Sync {
    async fn resolve(
        &self,
        host: &str,
        port: u16,
    ) -> Result<Vec<IpAddr>, WebhookTransportPolicyError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemWebhookDnsResolver;

#[async_trait]
impl WebhookDnsResolver for SystemWebhookDnsResolver {
    async fn resolve(
        &self,
        host: &str,
        port: u16,
    ) -> Result<Vec<IpAddr>, WebhookTransportPolicyError> {
        let mut addresses = tokio::net::lookup_host((host, port))
            .await
            .map_err(|_| WebhookTransportPolicyError::DnsUnavailable)?
            .map(|address| address.ip())
            .collect::<Vec<_>>();
        addresses.sort_unstable();
        addresses.dedup();
        Ok(addresses)
    }
}

pub struct WebhookNetworkPolicy<D> {
    dns_resolver: D,
}

impl WebhookNetworkPolicy<SystemWebhookDnsResolver> {
    pub fn system() -> Self {
        Self::new(SystemWebhookDnsResolver)
    }
}

impl<D> WebhookNetworkPolicy<D>
where
    D: WebhookDnsResolver,
{
    pub fn new(dns_resolver: D) -> Self {
        Self { dns_resolver }
    }

    pub const fn redirect_policy(&self) -> WebhookRedirectPolicy {
        WebhookRedirectPolicy::Reject
    }

    pub const fn proxies_enabled(&self) -> bool {
        false
    }

    pub async fn pin(
        &self,
        target: &str,
    ) -> Result<PinnedWebhookDestination, WebhookTransportPolicyError> {
        let url = Url::parse(target).map_err(|_| WebhookTransportPolicyError::InvalidTarget)?;
        if url.scheme() != "https"
            || url.username() != ""
            || url.password().is_some()
            || url.fragment().is_some()
            || url.port_or_known_default() != Some(443)
        {
            return Err(WebhookTransportPolicyError::InvalidTarget);
        }
        let host = url
            .host()
            .ok_or(WebhookTransportPolicyError::InvalidTarget)?;
        let host_string = host.to_string();
        let addresses = match host {
            Host::Domain(domain) => tokio::time::timeout(
                WEBHOOK_TOTAL_TIMEOUT,
                self.dns_resolver.resolve(domain, 443),
            )
            .await
            .map_err(|_| WebhookTransportPolicyError::Timeout)??,
            Host::Ipv4(address) => vec![IpAddr::V4(address)],
            Host::Ipv6(address) => vec![IpAddr::V6(address)],
        };
        if addresses.is_empty() || addresses.iter().any(|address| !is_public_ip(*address)) {
            return Err(WebhookTransportPolicyError::UnsafeAddress);
        }
        Ok(PinnedWebhookDestination {
            url,
            host: host_string,
            socket_address: SocketAddr::new(addresses[0], 443),
        })
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct PinnedWebhookDestination {
    url: Url,
    host: String,
    socket_address: SocketAddr,
}

impl PinnedWebhookDestination {
    pub fn url(&self) -> &Url {
        &self.url
    }

    pub fn socket_address(&self) -> SocketAddr {
        self.socket_address
    }

    pub fn build_client(&self) -> Result<reqwest::Client, WebhookTransportPolicyError> {
        reqwest::Client::builder()
            .timeout(WEBHOOK_TOTAL_TIMEOUT)
            .connect_timeout(WEBHOOK_TOTAL_TIMEOUT)
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .resolve(&self.host, self.socket_address)
            .build()
            .map_err(|_| WebhookTransportPolicyError::ClientConfiguration)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("webhook body stream failed")]
pub struct WebhookBodyReadError;

#[derive(Debug, thiserror::Error)]
pub enum WebhookAdmissionError {
    #[error("webhook authentication is invalid")]
    InvalidAuthentication,
    #[error("webhook credential is invalid")]
    InvalidCredential,
    #[error("webhook credential is unavailable")]
    CredentialUnavailable,
    #[error("webhook authentication is stale")]
    StaleAuthentication,
    #[error("webhook body exceeds its byte limit")]
    BodyTooLarge,
    #[error("webhook body could not be read")]
    BodyRead,
    #[error("webhook body digest does not match its signed digest")]
    BodyDigestMismatch,
    #[error("webhook body is not a valid bounded object")]
    InvalidBody,
    #[error("webhook admission timed out")]
    Timeout,
    #[error("webhook admission limits are invalid")]
    InvalidLimits,
    #[error(transparent)]
    Trigger(#[from] WorkflowTriggerAdmissionError),
    #[error(transparent)]
    Repository(#[from] super::repository::WorkflowRepositoryError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WebhookTransportPolicyError {
    #[error("webhook target is invalid")]
    InvalidTarget,
    #[error("webhook DNS resolution failed")]
    DnsUnavailable,
    #[error("webhook target resolved to an unsafe address")]
    UnsafeAddress,
    #[error("webhook target resolution timed out")]
    Timeout,
    #[error("webhook HTTP client could not be configured")]
    ClientConfiguration,
}

fn verify_authentication(
    tenant: &TenantContext,
    definition: &StoredWorkflowDefinition,
    authentication: &WebhookAuthentication,
    credential: &ResolvedWebhookCredential,
) -> Result<(), WebhookAdmissionError> {
    let mut mac = HmacSha256::new_from_slice(&credential.secret)
        .map_err(|_| WebhookAdmissionError::InvalidCredential)?;
    mac.update(&signature_payload(
        tenant,
        definition.identity,
        authentication.timestamp_millis,
        &authentication.idempotency_key,
        authentication.content_sha256,
    ));
    mac.verify_slice(&authentication.signature)
        .map_err(|_| WebhookAdmissionError::InvalidAuthentication)
}

fn signature_payload(
    tenant: &TenantContext,
    workflow: WorkflowIdentity,
    timestamp_millis: u64,
    idempotency_key: &str,
    content_sha256: [u8; 32],
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(256);
    payload.extend_from_slice(SIGNATURE_DOMAIN);
    payload.extend_from_slice(tenant.community_id().as_uuid().as_bytes());
    payload.extend_from_slice(workflow.workflow_id().as_bytes());
    payload.extend_from_slice(&timestamp_millis.to_be_bytes());
    payload.extend_from_slice(&(idempotency_key.len() as u16).to_be_bytes());
    payload.extend_from_slice(idempotency_key.as_bytes());
    payload.extend_from_slice(&content_sha256);
    payload
}

async fn read_webhook_body<S>(
    mut body: S,
    expected_sha256: [u8; 32],
) -> Result<Map<String, JsonValue>, WebhookAdmissionError>
where
    S: Stream<Item = Result<Bytes, WebhookBodyReadError>> + Unpin,
{
    let mut bytes = Vec::new();
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|_| WebhookAdmissionError::BodyRead)?;
        if bytes.len().saturating_add(chunk.len()) > MAX_WEBHOOK_BODY_BYTES {
            return Err(WebhookAdmissionError::BodyTooLarge);
        }
        bytes.extend_from_slice(&chunk);
    }
    let actual_sha256: [u8; 32] = Sha256::digest(&bytes).into();
    if actual_sha256 != expected_sha256 {
        return Err(WebhookAdmissionError::BodyDigestMismatch);
    }
    if bytes.is_empty() {
        return Ok(Map::new());
    }
    let value: JsonValue =
        serde_json::from_slice(&bytes).map_err(|_| WebhookAdmissionError::InvalidBody)?;
    validate_webhook_fields(value)
}

fn validate_webhook_fields(
    value: JsonValue,
) -> Result<Map<String, JsonValue>, WebhookAdmissionError> {
    let JsonValue::Object(fields) = value else {
        return Err(WebhookAdmissionError::InvalidBody);
    };
    if fields.len() > MAX_WEBHOOK_FIELDS {
        return Err(WebhookAdmissionError::InvalidBody);
    }
    for (name, value) in &fields {
        if !valid_field_name(name)
            || !matches!(
                value,
                JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_)
            )
            || serde_json::to_vec(value).map_or(true, |encoded| {
                encoded.len() > MAX_WEBHOOK_FIELD_VALUE_BYTES
            })
        {
            return Err(WebhookAdmissionError::InvalidBody);
        }
    }
    Ok(fields)
}

fn validate_webhook_definition(
    tenant: &TenantContext,
    definition: &StoredWorkflowDefinition,
) -> Result<(), WebhookAdmissionError> {
    validate_current_definition(tenant, definition)?;
    if !matches!(definition.definition.trigger(), WorkflowTrigger::Webhook) {
        return Err(WorkflowTriggerAdmissionError::TriggerMismatch.into());
    }
    Ok(())
}

fn decode_sha256(value: &str) -> Result<[u8; 32], WebhookAdmissionError> {
    let bytes = hex::decode(value).map_err(|_| WebhookAdmissionError::InvalidAuthentication)?;
    bytes
        .try_into()
        .map_err(|_| WebhookAdmissionError::InvalidAuthentication)
}

fn valid_idempotency_key(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_WEBHOOK_IDEMPOTENCY_KEY_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn valid_field_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    value.len() <= MAX_WEBHOOK_FIELD_NAME_BYTES
        && first.is_ascii_alphabetic()
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        && !value.starts_with("trigger_")
        && !matches!(value, "idempotency_key" | "timestamp" | "webhook")
}

fn is_public_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_ipv4(address),
        IpAddr::V6(address) => is_public_ipv6(address),
    }
}

fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let [first, second, third, _] = address.octets();
    !(first == 0
        || first == 10
        || first == 127
        || first >= 224
        || (first == 100 && (64..=127).contains(&second))
        || (first == 169 && second == 254)
        || (first == 172 && (16..=31).contains(&second))
        || (first == 192 && second == 0)
        || (first == 192 && second == 168)
        || (first == 198 && (second == 18 || second == 19))
        || (first == 198 && second == 51 && third == 100)
        || (first == 203 && second == 0 && third == 113))
}

fn is_public_ipv6(address: Ipv6Addr) -> bool {
    let segments = address.segments();
    if address.is_unspecified()
        || address.is_loopback()
        || address.is_multicast()
        || (segments[0] & 0xfe00) == 0xfc00
        || (segments[0] & 0xffc0) == 0xfe80
        || !(0x2000..=0x3fff).contains(&segments[0])
        || (segments[0] == 0x2001 && segments[1] == 0)
        || (segments[0] == 0x2001 && segments[1] == 2)
        || (segments[0] == 0x2001 && (segments[1] & 0xfff0) == 0x0010)
        || (segments[0] == 0x2001 && (segments[1] & 0xfff0) == 0x0020)
        || (segments[0] == 0x2001 && segments[1] == 0x0db8)
        || segments[0] == 0x2002
        || (segments[0] == 0x3fff && (segments[1] & 0xf000) == 0)
    {
        return false;
    }
    match address.to_ipv4_mapped() {
        Some(address) => is_public_ipv4(address),
        None => true,
    }
}
