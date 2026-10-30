use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_plugin_sdk::{
    ArtifactValue, CapabilityCall, CapabilityKind, CapabilityQuota, CapabilityResponse,
    InvocationError, ModelValue, PluginManifest,
};
use sim_runtime::{
    AssetError, AssetIdentity, AssetNamespace, AssetService, AuthorizedCapabilities, Capability,
    OutputCommitError, OutputCommitReceipt, OutputCommitter, OutputExecutionScope, OutputProposal,
    PluginAuthorization, PluginCapabilityBroker, PluginCapabilityInvocation, PluginModelHandle,
    PluginServiceError, PluginServiceInvocationContext, SecretId, SharedAssetService,
};
pub use sim_types::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use thiserror::Error;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityLimits {
    pub maximum_operations: u64,
    pub maximum_request_bytes: u64,
    pub maximum_response_bytes: u64,
    pub maximum_total_bytes: u64,
    pub maximum_handles: u32,
    pub maximum_timeout_milliseconds: u64,
}

impl Default for CapabilityLimits {
    fn default() -> Self {
        Self {
            maximum_operations: 1_000_000,
            maximum_request_bytes: 64 * 1024 * 1024,
            maximum_response_bytes: 64 * 1024 * 1024,
            maximum_total_bytes: 256 * 1024 * 1024,
            maximum_handles: 65_536,
            maximum_timeout_milliseconds: 60_000,
        }
    }
}

impl CapabilityLimits {
    pub fn validate(&self) -> Result<(), InvocationError> {
        if self.maximum_operations == 0
            || self.maximum_request_bytes == 0
            || self.maximum_response_bytes == 0
            || self.maximum_total_bytes == 0
            || self.maximum_handles == 0
            || self.maximum_timeout_milliseconds == 0
            || self.maximum_request_bytes > self.maximum_total_bytes
            || self.maximum_response_bytes > self.maximum_total_bytes
        {
            return Err(InvocationError::HostFailure(
                "invalid host capability limits".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn validate_quota(&self, quota: CapabilityQuota) -> Result<(), InvocationError> {
        self.validate()?;
        if quota.maximum_operations > self.maximum_operations
            || quota.maximum_request_bytes > self.maximum_request_bytes
            || quota.maximum_response_bytes > self.maximum_response_bytes
            || quota.maximum_total_bytes > self.maximum_total_bytes
            || quota.maximum_handles > self.maximum_handles
            || quota.timeout_milliseconds > self.maximum_timeout_milliseconds
        {
            return Err(InvocationError::InvalidCapabilityRequest(
                "requested capability quota exceeds a host ceiling".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct CapabilityServiceContext {
    cancellation: CancellationToken,
    deadline: Instant,
    maximum_response_bytes: u64,
    kind: CapabilityKind,
}

impl CapabilityServiceContext {
    fn new(
        cancellation: CancellationToken,
        started_at: Instant,
        timeout_milliseconds: u64,
        maximum_response_bytes: u64,
        kind: CapabilityKind,
    ) -> Result<Self, InvocationError> {
        let deadline = started_at
            .checked_add(Duration::from_millis(timeout_milliseconds))
            .ok_or_else(|| {
                InvocationError::InvalidCapabilityRequest("capability deadline overflow".to_owned())
            })?;
        Ok(Self {
            cancellation,
            deadline,
            maximum_response_bytes,
            kind,
        })
    }

    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    pub fn maximum_response_bytes(&self) -> u64 {
        self.maximum_response_bytes
    }

    pub fn check_active(&self) -> Result<(), InvocationError> {
        check_plugin_cancellation(&self.cancellation)?;
        if Instant::now() > self.deadline {
            return Err(InvocationError::TimedOut);
        }
        Ok(())
    }

    pub fn validate_response_length(&self, length: usize) -> Result<(), InvocationError> {
        let length = u64::try_from(length).map_err(|_| quota_error(self.kind, "response-byte"))?;
        if length > self.maximum_response_bytes {
            return Err(quota_error(self.kind, "response-byte"));
        }
        Ok(())
    }
}

pub fn check_plugin_cancellation(cancellation: &CancellationToken) -> Result<(), InvocationError> {
    cancellation.check().map_err(|_| InvocationError::Cancelled)
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CapabilityKey {
    kind: CapabilityKind,
    scope: String,
}

pub trait PluginCapabilityServices: Send + Sync {
    fn read_asset(
        &self,
        identity: &AssetIdentity,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError>;

    fn call_provider(
        &self,
        provider: &str,
        endpoint: &str,
        body: &[u8],
        secret_id: Option<&SecretId>,
    ) -> Result<Vec<u8>, InvocationError>;

    fn secret_exists(&self, identifier: &str) -> Result<bool, InvocationError>;

    fn clock_milliseconds(&self, clock: &str) -> Result<u64, InvocationError>;

    fn random_bytes(&self, stream: &str, length: u32) -> Result<Vec<u8>, InvocationError>;

    fn open_model(&self, identifier: &str) -> Result<ModelValue, InvocationError>;

    fn sanitize_log(&self, level: &str, message: &str) -> Result<String, InvocationError>;

    fn finish_invocation(&self) -> Result<(), InvocationError> {
        Ok(())
    }

    fn abort_invocation(&self) {}

    fn call_provider_with_context(
        &self,
        provider: &str,
        endpoint: &str,
        body: &[u8],
        secret_id: Option<&SecretId>,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        context.check_active()?;
        let bytes = self.call_provider(provider, endpoint, body, secret_id)?;
        context.validate_response_length(bytes.len())?;
        context.check_active()?;
        Ok(bytes)
    }

    fn secret_exists_with_context(
        &self,
        identifier: &str,
        context: &CapabilityServiceContext,
    ) -> Result<bool, InvocationError> {
        context.check_active()?;
        let exists = self.secret_exists(identifier)?;
        context.check_active()?;
        Ok(exists)
    }

    fn clock_milliseconds_with_context(
        &self,
        clock: &str,
        context: &CapabilityServiceContext,
    ) -> Result<u64, InvocationError> {
        context.check_active()?;
        let milliseconds = self.clock_milliseconds(clock)?;
        context.check_active()?;
        Ok(milliseconds)
    }

    fn random_bytes_with_context(
        &self,
        stream: &str,
        length: u32,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        context.check_active()?;
        let bytes = self.random_bytes(stream, length)?;
        context.validate_response_length(bytes.len())?;
        context.check_active()?;
        Ok(bytes)
    }

    fn open_model_with_context(
        &self,
        identifier: &str,
        context: &CapabilityServiceContext,
    ) -> Result<ModelValue, InvocationError> {
        context.check_active()?;
        let model = self.open_model(identifier)?;
        context.check_active()?;
        Ok(model)
    }

    fn sanitize_log_with_context(
        &self,
        level: &str,
        message: &str,
        context: &CapabilityServiceContext,
    ) -> Result<String, InvocationError> {
        context.check_active()?;
        let message = self.sanitize_log(level, message)?;
        context.validate_response_length(message.len())?;
        context.check_active()?;
        Ok(message)
    }
}

#[derive(Default)]
pub struct UnavailablePluginCapabilityServices;

impl PluginCapabilityServices for UnavailablePluginCapabilityServices {
    fn read_asset(
        &self,
        _identity: &AssetIdentity,
        _context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        Err(unavailable_service("artifact"))
    }

    fn call_provider(
        &self,
        _provider: &str,
        _endpoint: &str,
        _body: &[u8],
        _secret_id: Option<&SecretId>,
    ) -> Result<Vec<u8>, InvocationError> {
        Err(unavailable_service("provider"))
    }

    fn secret_exists(&self, _identifier: &str) -> Result<bool, InvocationError> {
        Err(unavailable_service("credentials"))
    }

    fn clock_milliseconds(&self, _clock: &str) -> Result<u64, InvocationError> {
        Err(unavailable_service("clock"))
    }

    fn random_bytes(&self, _stream: &str, _length: u32) -> Result<Vec<u8>, InvocationError> {
        Err(unavailable_service("randomness"))
    }

    fn open_model(&self, _identifier: &str) -> Result<ModelValue, InvocationError> {
        Err(unavailable_service("model"))
    }

    fn sanitize_log(&self, level: &str, message: &str) -> Result<String, InvocationError> {
        let _ = level;
        Ok(sanitize_log(message, &BTreeSet::new()))
    }
}

pub struct AssetPluginCapabilityServices {
    assets: SharedAssetService,
    authorization: AuthorizedCapabilities,
}

fn plugin_asset_identity(
    profile_id: &str,
    root: &str,
    relative_path: &str,
) -> Result<AssetIdentity, InvocationError> {
    let namespace = AssetNamespace::from_plugin_root(root).map_err(|_| {
        InvocationError::InvalidCapabilityRequest(format!(
            "plugin asset root `{root}` is not supported"
        ))
    })?;
    AssetIdentity::new(profile_id, namespace, relative_path)
        .map_err(|error| InvocationError::InvalidCapabilityRequest(error.to_string()))
}

pub(crate) fn artifact_value_identity(
    profile_id: &str,
    value: &ArtifactValue,
) -> Result<AssetIdentity, InvocationError> {
    plugin_asset_identity(profile_id, value.namespace(), value.identifier())
}

fn model_store_handle_value(model: &PluginModelHandle) -> Result<ModelValue, InvocationError> {
    ModelValue::new(
        model.model_id(),
        model.model_format(),
        model.model_identity(),
    )
    .map_err(|_| {
        InvocationError::HostFailure(
            "canonical model handle cannot be represented by the plugin ABI".to_owned(),
        )
    })
}

fn map_asset_read_error(identity: &AssetIdentity, error: AssetError) -> InvocationError {
    match error {
        AssetError::PermissionDenied { .. } | AssetError::ProfileMismatch { .. } => {
            InvocationError::CapabilityDenied {
                kind: CapabilityKind::Filesystem,
                scope: identity.namespace.locator_type().to_owned(),
            }
        }
        error => InvocationError::HostFailure(error.to_string()),
    }
}

impl AssetPluginCapabilityServices {
    pub fn new(
        assets: SharedAssetService,
        authorization: AuthorizedCapabilities,
    ) -> Result<Self, InvocationError> {
        let profile_id = assets
            .lock()
            .map_err(|error| InvocationError::HostFailure(error.to_string()))?
            .roots()
            .profile_id
            .clone();
        if authorization.profile_id() != profile_id {
            return Err(InvocationError::HostFailure(
                "plugin asset authorization belongs to another profile".to_owned(),
            ));
        }
        Ok(Self {
            assets,
            authorization,
        })
    }

    pub fn assets(&self) -> &SharedAssetService {
        &self.assets
    }
}

impl PluginCapabilityServices for AssetPluginCapabilityServices {
    fn read_asset(
        &self,
        identity: &AssetIdentity,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        context.check_active()?;
        let bytes = self
            .assets
            .lock()
            .map_err(|error| InvocationError::HostFailure(error.to_string()))?
            .read_verified(
                identity,
                &self.authorization,
                context.cancellation(),
                context.maximum_response_bytes(),
            )
            .map_err(|error| map_asset_read_error(identity, error))?;
        context.validate_response_length(bytes.len())?;
        context.check_active()?;
        Ok(bytes)
    }

    fn call_provider(
        &self,
        _provider: &str,
        _endpoint: &str,
        _body: &[u8],
        _secret_id: Option<&SecretId>,
    ) -> Result<Vec<u8>, InvocationError> {
        Err(unavailable_service("provider"))
    }

    fn secret_exists(&self, _identifier: &str) -> Result<bool, InvocationError> {
        Err(unavailable_service("credentials"))
    }

    fn clock_milliseconds(&self, _clock: &str) -> Result<u64, InvocationError> {
        Err(unavailable_service("clock"))
    }

    fn random_bytes(&self, _stream: &str, _length: u32) -> Result<Vec<u8>, InvocationError> {
        Err(unavailable_service("randomness"))
    }

    fn open_model(&self, _identifier: &str) -> Result<ModelValue, InvocationError> {
        Err(unavailable_service("model"))
    }

    fn sanitize_log(&self, level: &str, message: &str) -> Result<String, InvocationError> {
        let _ = level;
        Ok(sanitize_log(message, &BTreeSet::new()))
    }
}

pub struct BrokerPluginCapabilityServices {
    session: Mutex<Option<PluginCapabilityInvocation>>,
    log_redactions: BTreeSet<String>,
}

impl BrokerPluginCapabilityServices {
    pub fn new(
        broker: &PluginCapabilityBroker,
        invocation: PluginServiceInvocationContext,
        log_redactions: BTreeSet<String>,
    ) -> Result<Self, InvocationError> {
        let session = broker
            .begin_invocation(invocation)
            .map_err(|error| map_broker_error(error, CapabilityKind::SanitizedLog, "invocation"))?;
        Ok(Self {
            session: Mutex::new(Some(session)),
            log_redactions,
        })
    }

    fn with_session<T>(
        &self,
        kind: CapabilityKind,
        scope: &str,
        operation: impl FnOnce(&mut PluginCapabilityInvocation) -> Result<T, PluginServiceError>,
    ) -> Result<T, InvocationError> {
        let mut session = self.session.lock().map_err(|_| {
            InvocationError::HostFailure(
                "canonical plugin capability session is unavailable".to_owned(),
            )
        })?;
        let session = session.as_mut().ok_or(InvocationError::RevokedHandle)?;
        operation(session).map_err(|error| map_broker_error(error, kind, scope))
    }

    fn abort_session(&self) {
        let mut session = match self.session.lock() {
            Ok(session) => session,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(session) = session.take() {
            session.abort();
        }
    }
}

impl PluginCapabilityServices for BrokerPluginCapabilityServices {
    fn read_asset(
        &self,
        identity: &AssetIdentity,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        let outcome = BrokerAdapterOutcomeGuard::new(self);
        context.check_active()?;
        let reference = identity.to_reference().map_err(|_| {
            InvocationError::InvalidCapabilityRequest(
                "asset reference is not available to this plugin".to_owned(),
            )
        })?;
        let scope = identity.namespace.locator_type();
        let bytes = self.with_session(CapabilityKind::Filesystem, scope, |session| {
            session.read_asset(identity.namespace, &reference)
        })?;
        context.validate_response_length(bytes.len())?;
        context.check_active()?;
        Ok(outcome.succeed(bytes))
    }

    fn call_provider(
        &self,
        _provider: &str,
        _endpoint: &str,
        _body: &[u8],
        _secret_id: Option<&SecretId>,
    ) -> Result<Vec<u8>, InvocationError> {
        self.abort_session();
        Err(context_required_service("provider"))
    }

    fn secret_exists(&self, _identifier: &str) -> Result<bool, InvocationError> {
        self.abort_session();
        Err(context_required_service("credentials"))
    }

    fn clock_milliseconds(&self, _clock: &str) -> Result<u64, InvocationError> {
        self.abort_session();
        Err(context_required_service("clock"))
    }

    fn random_bytes(&self, _stream: &str, _length: u32) -> Result<Vec<u8>, InvocationError> {
        self.abort_session();
        Err(context_required_service("randomness"))
    }

    fn open_model(&self, _identifier: &str) -> Result<ModelValue, InvocationError> {
        self.abort_session();
        Err(context_required_service("model"))
    }

    fn sanitize_log(&self, _level: &str, _message: &str) -> Result<String, InvocationError> {
        self.abort_session();
        Err(context_required_service("log"))
    }

    fn call_provider_with_context(
        &self,
        provider: &str,
        endpoint: &str,
        body: &[u8],
        secret_id: Option<&SecretId>,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        let outcome = BrokerAdapterOutcomeGuard::new(self);
        context.check_active()?;
        let scope = format!("{provider}|{endpoint}");
        let bytes = self.with_session(CapabilityKind::NetworkProvider, &scope, |session| {
            session.execute_provider_request(provider, endpoint, secret_id, body)
        })?;
        context.validate_response_length(bytes.len())?;
        context.check_active()?;
        Ok(outcome.succeed(bytes))
    }

    fn secret_exists_with_context(
        &self,
        identifier: &str,
        context: &CapabilityServiceContext,
    ) -> Result<bool, InvocationError> {
        let outcome = BrokerAdapterOutcomeGuard::new(self);
        context.check_active()?;
        let secret_id = SecretId::new(identifier).map_err(|_| {
            InvocationError::InvalidCapabilityRequest("secret identifier is invalid".to_owned())
        })?;
        let exists = self.with_session(CapabilityKind::Secret, identifier, |session| {
            session.credential_is_present(&secret_id)
        })?;
        context.check_active()?;
        Ok(outcome.succeed(exists))
    }

    fn clock_milliseconds_with_context(
        &self,
        clock: &str,
        context: &CapabilityServiceContext,
    ) -> Result<u64, InvocationError> {
        let outcome = BrokerAdapterOutcomeGuard::new(self);
        context.check_active()?;
        let milliseconds = self.with_session(CapabilityKind::Clock, clock, |session| {
            session.monotonic_milliseconds(clock)
        })?;
        context.check_active()?;
        Ok(outcome.succeed(milliseconds))
    }

    fn random_bytes_with_context(
        &self,
        stream: &str,
        length: u32,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        let outcome = BrokerAdapterOutcomeGuard::new(self);
        context.check_active()?;
        let length = usize::try_from(length)
            .map_err(|_| quota_error(CapabilityKind::Randomness, "response-byte"))?;
        let bytes = self.with_session(CapabilityKind::Randomness, stream, |session| {
            session.random_bytes(stream, length)
        })?;
        context.validate_response_length(bytes.len())?;
        context.check_active()?;
        Ok(outcome.succeed(bytes))
    }

    fn open_model_with_context(
        &self,
        identifier: &str,
        context: &CapabilityServiceContext,
    ) -> Result<ModelValue, InvocationError> {
        let outcome = BrokerAdapterOutcomeGuard::new(self);
        context.check_active()?;
        let model = self.with_session(CapabilityKind::Model, identifier, |session| {
            session.load_model(identifier)
        })?;
        let value = model_store_handle_value(&model)?;
        context.check_active()?;
        Ok(outcome.succeed(value))
    }

    fn sanitize_log_with_context(
        &self,
        _level: &str,
        message: &str,
        context: &CapabilityServiceContext,
    ) -> Result<String, InvocationError> {
        let outcome = BrokerAdapterOutcomeGuard::new(self);
        context.check_active()?;
        let message = sanitize_log(message, &self.log_redactions);
        context.validate_response_length(message.len())?;
        context.check_active()?;
        Ok(outcome.succeed(message))
    }

    fn finish_invocation(&self) -> Result<(), InvocationError> {
        let session = self
            .session
            .lock()
            .map_err(|_| {
                InvocationError::HostFailure(
                    "canonical plugin capability session is unavailable".to_owned(),
                )
            })?
            .take()
            .ok_or(InvocationError::RevokedHandle)?;
        session
            .finish()
            .map_err(|error| map_broker_error(error, CapabilityKind::Randomness, "invocation"))
    }

    fn abort_invocation(&self) {
        self.abort_session();
    }
}

struct BrokerAdapterOutcomeGuard<'a> {
    services: &'a BrokerPluginCapabilityServices,
    succeeded: bool,
}

impl<'a> BrokerAdapterOutcomeGuard<'a> {
    fn new(services: &'a BrokerPluginCapabilityServices) -> Self {
        Self {
            services,
            succeeded: false,
        }
    }

    fn succeed<T>(mut self, value: T) -> T {
        self.succeeded = true;
        value
    }
}

impl Drop for BrokerAdapterOutcomeGuard<'_> {
    fn drop(&mut self) {
        if !self.succeeded {
            self.services.abort_session();
        }
    }
}

impl Drop for BrokerPluginCapabilityServices {
    fn drop(&mut self) {
        self.abort_session();
    }
}

fn map_broker_error(
    error: PluginServiceError,
    kind: CapabilityKind,
    scope: &str,
) -> InvocationError {
    match error {
        PluginServiceError::CapabilityDenied(capability) => capability
            .plugin_capability_key()
            .map(|(kind, scope)| InvocationError::CapabilityDenied { kind, scope })
            .unwrap_or_else(|| InvocationError::CapabilityDenied {
                kind,
                scope: scope.to_owned(),
            }),
        PluginServiceError::Cancelled => InvocationError::Cancelled,
        PluginServiceError::DeadlineExceeded => InvocationError::TimedOut,
        PluginServiceError::RequestTooLarge { .. } => quota_error(kind, "request-byte"),
        PluginServiceError::ResponseTooLarge { .. }
        | PluginServiceError::InvalidResponseLimit { .. }
        | PluginServiceError::ResponseAllocationFailed => quota_error(kind, "response-byte"),
        PluginServiceError::InvalidIdentity { .. }
        | PluginServiceError::InvalidWirePayload
        | PluginServiceError::AssetNamespaceMismatch
        | PluginServiceError::ModelNamespaceRequired => InvocationError::InvalidCapabilityRequest(
            "canonical capability identifier is invalid".to_owned(),
        ),
        PluginServiceError::ProviderPolicyDenied
        | PluginServiceError::ProviderCostAcceptanceRequired
        | PluginServiceError::ProviderCostPrincipalRequired
        | PluginServiceError::ProviderCostAcceptanceDenied
        | PluginServiceError::ProviderCostAcceptanceReused
        | PluginServiceError::ProviderResultReceiptAuthorityRequired
        | PluginServiceError::ProviderResultReceiptAuthorityDenied
        | PluginServiceError::ProviderSessionUnavailable
        | PluginServiceError::ProviderRuntimeAuthorityDenied
        | PluginServiceError::ProviderRuntimeForeignSession
        | PluginServiceError::ProviderRuntimeStaleSession
        | PluginServiceError::ProviderRuntimeForeignInvocation
        | PluginServiceError::ProviderRuntimeStaleInvocation
        | PluginServiceError::ProviderStreamingContract(_) => InvocationError::CapabilityDenied {
            kind: CapabilityKind::NetworkProvider,
            scope: scope.to_owned(),
        },
        PluginServiceError::InvocationFinished | PluginServiceError::InvocationFailed => {
            InvocationError::RevokedHandle
        }
        PluginServiceError::ProfileMismatch
        | PluginServiceError::AuthorizationIdentityMismatch
        | PluginServiceError::ClockMovedBackwards
        | PluginServiceError::ClockOverflow
        | PluginServiceError::AssetServiceUnavailable
        | PluginServiceError::CredentialUnavailable
        | PluginServiceError::AssetOperationFailed { .. }
        | PluginServiceError::ActuatorFailed { .. }
        | PluginServiceError::RandomnessStreamBusy
        | PluginServiceError::RandomnessFailed => {
            InvocationError::HostFailure("canonical plugin capability operation failed".to_owned())
        }
    }
}

fn context_required_service(name: &str) -> InvocationError {
    InvocationError::HostFailure(format!(
        "canonical {name} service requires an invocation context"
    ))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PluginOutputProposal {
    pub identifier: String,
    pub namespace: String,
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PluginOutputPublicationAdapter;

impl PluginOutputPublicationAdapter {
    pub fn map_proposals(
        &self,
        execution_scope: &OutputExecutionScope,
        proposals: &[PluginOutputProposal],
    ) -> Result<Vec<OutputProposal>, PluginOutputPublicationError> {
        proposals
            .iter()
            .enumerate()
            .map(|(index, proposal)| {
                let batch_index = u32::try_from(index).map_err(|_| {
                    PluginOutputPublicationError::ProposalCountExceeded(proposals.len())
                })?;
                self.map_proposal(execution_scope, proposal, batch_index)
            })
            .collect()
    }

    pub fn publish(
        &self,
        execution_scope: &OutputExecutionScope,
        proposals: &[PluginOutputProposal],
        committer: &mut OutputCommitter,
        assets: &mut AssetService,
        capabilities: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<OutputCommitReceipt>, PluginOutputPublicationError> {
        let proposals = self.map_proposals(execution_scope, proposals)?;
        committer
            .commit_scoped_proposal_batch_and_register_now(
                execution_scope,
                &proposals,
                assets,
                capabilities,
                cancellation,
            )
            .map_err(Into::into)
    }

    fn map_proposal(
        &self,
        execution_scope: &OutputExecutionScope,
        proposal: &PluginOutputProposal,
        batch_index: u32,
    ) -> Result<OutputProposal, PluginOutputPublicationError> {
        let namespace = match proposal.namespace.as_str() {
            "output" | "outputs" => AssetNamespace::Output,
            "temp" | "temporary" => AssetNamespace::Temporary,
            namespace => {
                return Err(PluginOutputPublicationError::UnsupportedNamespace(
                    namespace.to_owned(),
                ));
            }
        };
        validate_output_segment(&proposal.name)
            .map_err(|_| PluginOutputPublicationError::InvalidName(proposal.name.clone()))?;
        let (filename_prefix, extension) = proposal
            .name
            .rsplit_once('.')
            .ok_or_else(|| PluginOutputPublicationError::MissingExtension(proposal.name.clone()))?;
        if filename_prefix.is_empty() {
            return Err(PluginOutputPublicationError::InvalidName(
                proposal.name.clone(),
            ));
        }
        if extension.is_empty() {
            return Err(PluginOutputPublicationError::MissingExtension(
                proposal.name.clone(),
            ));
        }
        let proposal_id = Uuid::new_v5(
            &execution_scope.attempt_id.0,
            proposal.identifier.as_bytes(),
        );
        OutputProposal::new(
            proposal_id,
            namespace,
            filename_prefix,
            extension,
            batch_index,
            0,
            0,
            proposal.bytes.clone(),
        )
        .map_err(Into::into)
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PluginOutputPublicationError {
    #[error("plugin output namespace {0:?} is unsupported")]
    UnsupportedNamespace(String),
    #[error("plugin output name {0:?} is invalid")]
    InvalidName(String),
    #[error("plugin output name {0:?} has no valid extension")]
    MissingExtension(String),
    #[error("plugin output proposal count {0} exceeds the supported batch index")]
    ProposalCountExceeded(usize),
    #[error(transparent)]
    Commit(#[from] OutputCommitError),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RouteEffect {
    pub route: String,
    pub status: u16,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityEffects {
    pub outputs: Vec<PluginOutputProposal>,
    pub logs: Vec<String>,
    pub ui_state: BTreeMap<String, Vec<u8>>,
    pub routes: Vec<RouteEffect>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CapabilityUsage {
    operations: u64,
    request_bytes: u64,
    response_bytes: u64,
    handles: u32,
}

#[derive(Clone, Debug)]
struct OutputProposalBuffer {
    namespace: String,
    name: String,
    bytes: Vec<u8>,
}

pub struct CapabilityState {
    quotas: BTreeMap<CapabilityKey, CapabilityQuota>,
    usage: BTreeMap<CapabilityKey, CapabilityUsage>,
    services: Arc<dyn PluginCapabilityServices>,
    profile_id: String,
    cancellation: CancellationToken,
    started_at: Instant,
    next_output_buffer: u64,
    output_buffers: BTreeMap<u64, OutputProposalBuffer>,
    ui_contributions: BTreeSet<String>,
    route_response_limits: BTreeMap<String, u64>,
    pending_effects: CapabilityEffects,
    limits: CapabilityLimits,
    terminal: bool,
}

impl CapabilityState {
    pub fn new(
        authorization: &PluginAuthorization,
        manifest: &PluginManifest,
        services: Arc<dyn PluginCapabilityServices>,
        cancellation: CancellationToken,
    ) -> Result<Self, InvocationError> {
        Self::with_declarations(
            authorization,
            manifest,
            services,
            cancellation,
            CapabilityLimits::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )
    }

    pub(crate) fn with_declarations(
        authorization: &PluginAuthorization,
        manifest: &PluginManifest,
        services: Arc<dyn PluginCapabilityServices>,
        cancellation: CancellationToken,
        limits: CapabilityLimits,
        ui_contributions: BTreeSet<String>,
        route_response_limits: BTreeMap<String, u64>,
    ) -> Result<Self, InvocationError> {
        limits.validate()?;
        authorization
            .require_manifest(manifest)
            .map_err(|error| InvocationError::HostFailure(error.to_string()))?;
        let mut grant_map = BTreeMap::new();
        for request in &manifest.capabilities {
            request
                .quota
                .validate()
                .map_err(|error| InvocationError::HostFailure(error.to_string()))?;
            limits.validate_quota(request.quota)?;
            let capability = Capability::from_plugin_request(request)
                .map_err(|error| InvocationError::InvalidCapabilityRequest(error.to_string()))?;
            authorization
                .capabilities()
                .require(&capability)
                .map_err(|_| InvocationError::CapabilityDenied {
                    kind: request.kind,
                    scope: request.scope.clone(),
                })?;
            let key = CapabilityKey {
                kind: request.kind,
                scope: request.scope.clone(),
            };
            if grant_map.insert(key, request.quota).is_some() {
                return Err(InvocationError::HostFailure(
                    "duplicate capability declaration".to_owned(),
                ));
            }
        }
        Ok(Self {
            quotas: grant_map,
            usage: BTreeMap::new(),
            services,
            profile_id: authorization.capabilities().profile_id().to_owned(),
            cancellation,
            started_at: Instant::now(),
            next_output_buffer: 1,
            output_buffers: BTreeMap::new(),
            ui_contributions,
            route_response_limits,
            pending_effects: CapabilityEffects::default(),
            limits,
            terminal: false,
        })
    }

    #[cfg(test)]
    fn with_test_declarations(
        declarations: impl IntoIterator<Item = (CapabilityKind, String, CapabilityQuota)>,
        profile_id: String,
        services: Arc<dyn PluginCapabilityServices>,
        cancellation: CancellationToken,
        ui_contributions: BTreeSet<String>,
        route_response_limits: BTreeMap<String, u64>,
    ) -> Result<Self, InvocationError> {
        let limits = CapabilityLimits::default();
        limits.validate()?;
        let mut quotas = BTreeMap::new();
        for (kind, scope, quota) in declarations {
            quota
                .validate()
                .map_err(|error| InvocationError::HostFailure(error.to_string()))?;
            limits.validate_quota(quota)?;
            let key = CapabilityKey { kind, scope };
            if quotas.insert(key, quota).is_some() {
                return Err(InvocationError::HostFailure(
                    "duplicate test capability declaration".to_owned(),
                ));
            }
        }
        Ok(Self {
            quotas,
            usage: BTreeMap::new(),
            services,
            profile_id,
            cancellation,
            started_at: Instant::now(),
            next_output_buffer: 1,
            output_buffers: BTreeMap::new(),
            ui_contributions,
            route_response_limits,
            pending_effects: CapabilityEffects::default(),
            limits,
            terminal: false,
        })
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub fn check_cancelled(&self) -> Result<(), InvocationError> {
        self.check_active()?;
        self.check_cancellation()
    }

    pub fn execute(&mut self, call: CapabilityCall) -> Result<CapabilityResponse, InvocationError> {
        self.check_active()?;
        self.check_cancellation()?;
        let provider_secret = validate_provider_secret(&call)?;
        let kind = call.kind();
        let scope = scope_for_call(&call)?;
        let request_bytes = request_size(&call)?;
        let key = CapabilityKey { kind, scope };
        let (quota, usage) = self.stage_request_charge(&key, request_bytes)?;
        let secret_charge = if let CapabilityCall::NetworkProvider {
            secret_id: Some(secret_id),
            ..
        } = &call
        {
            let secret_key = CapabilityKey {
                kind: CapabilityKind::Secret,
                scope: secret_id.clone(),
            };
            let secret_request_bytes = u64::try_from(secret_id.len()).map_err(|_| {
                InvocationError::HostFailure("secret identifier is too large".to_owned())
            })?;
            let (secret_quota, secret_usage) =
                self.stage_request_charge(&secret_key, secret_request_bytes)?;
            Some((secret_key, secret_quota, secret_usage))
        } else {
            None
        };
        self.usage.insert(key.clone(), usage);
        if let Some((secret_key, _secret_quota, secret_usage)) = secret_charge {
            self.usage.insert(secret_key, secret_usage);
        }
        self.preflight_stateful_limits(&call, &key, quota)?;
        let maximum_response_bytes = self
            .remaining_response_allowance(&key, quota)?
            .min(self.limits.maximum_response_bytes);
        let known_response_bytes = known_response_size(&call);
        if known_response_bytes > maximum_response_bytes {
            return Err(quota_error(kind, "response-byte"));
        }

        let expected_random_length = if let CapabilityCall::RandomBytes { length, .. } = &call {
            self.charge_response(&key, quota, u64::from(*length))?;
            Some(*length)
        } else {
            None
        };
        let service_context = CapabilityServiceContext::new(
            self.cancellation.clone(),
            self.started_at,
            quota.timeout_milliseconds,
            expected_random_length.map_or(maximum_response_bytes, u64::from),
            kind,
        )?;

        let prepared = self.prepare_call(call, provider_secret.as_ref(), &service_context)?;
        if let Some(expected_length) = expected_random_length {
            validate_random_response(&prepared, expected_length)?;
        }
        let response_bytes = prepared.response_size()?;
        if expected_random_length.is_none() {
            self.charge_response(&key, quota, response_bytes)?;
        }
        self.check_cancellation()?;
        self.apply_prepared(prepared)
    }

    pub fn has_open_output_buffers(&self) -> bool {
        !self.output_buffers.is_empty()
    }

    pub fn finish(mut self) -> Result<CapabilityEffects, InvocationError> {
        self.check_active()?;
        self.check_cancellation()?;
        if self.has_open_output_buffers() {
            self.rollback();
            return Err(InvocationError::HostFailure(
                "plugin invocation left an output transaction open".to_owned(),
            ));
        }
        self.services.finish_invocation()?;
        self.terminal = true;
        Ok(std::mem::take(&mut self.pending_effects))
    }

    pub fn rollback(&mut self) {
        self.services.abort_invocation();
        self.output_buffers.clear();
        self.pending_effects = CapabilityEffects::default();
        self.terminal = true;
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    fn check_active(&self) -> Result<(), InvocationError> {
        if self.terminal {
            Err(InvocationError::RevokedHandle)
        } else {
            Ok(())
        }
    }

    fn check_cancellation(&self) -> Result<(), InvocationError> {
        check_plugin_cancellation(&self.cancellation)
    }

    #[cfg(test)]
    fn charge_request(
        &mut self,
        key: &CapabilityKey,
        request_bytes: u64,
    ) -> Result<CapabilityQuota, InvocationError> {
        let (quota, usage) = self.stage_request_charge(key, request_bytes)?;
        self.usage.insert(key.clone(), usage);
        Ok(quota)
    }

    fn stage_request_charge(
        &self,
        key: &CapabilityKey,
        request_bytes: u64,
    ) -> Result<(CapabilityQuota, CapabilityUsage), InvocationError> {
        let quota =
            self.quotas
                .get(key)
                .copied()
                .ok_or_else(|| InvocationError::CapabilityDenied {
                    kind: key.kind,
                    scope: key.scope.clone(),
                })?;
        if self.started_at.elapsed() > Duration::from_millis(quota.timeout_milliseconds) {
            return Err(InvocationError::TimedOut);
        }
        if request_bytes > quota.maximum_request_bytes {
            return Err(quota_error(key.kind, "request-byte"));
        }
        let mut usage = self.usage.get(key).copied().unwrap_or_default();
        let operations = usage
            .operations
            .checked_add(1)
            .ok_or_else(|| quota_error(key.kind, "operation"))?;
        let total_request_bytes = usage
            .request_bytes
            .checked_add(request_bytes)
            .ok_or_else(|| quota_error(key.kind, "request-byte"))?;
        let total_bytes = total_request_bytes
            .checked_add(usage.response_bytes)
            .ok_or_else(|| quota_error(key.kind, "total-byte"))?;
        if operations > quota.maximum_operations {
            return Err(quota_error(key.kind, "operation"));
        }
        if total_bytes > quota.maximum_total_bytes {
            return Err(quota_error(key.kind, "total-byte"));
        }
        usage.operations = operations;
        usage.request_bytes = total_request_bytes;
        Ok((quota, usage))
    }

    fn charge_response(
        &mut self,
        key: &CapabilityKey,
        quota: CapabilityQuota,
        response_bytes: u64,
    ) -> Result<(), InvocationError> {
        if self.started_at.elapsed() > Duration::from_millis(quota.timeout_milliseconds) {
            return Err(InvocationError::TimedOut);
        }
        if response_bytes > quota.maximum_response_bytes {
            return Err(quota_error(key.kind, "response-byte"));
        }
        let usage = self.usage.get_mut(key).ok_or_else(|| {
            InvocationError::HostFailure("capability usage was not initialized".to_owned())
        })?;
        let total_response_bytes = usage
            .response_bytes
            .checked_add(response_bytes)
            .ok_or_else(|| quota_error(key.kind, "response-byte"))?;
        let total_bytes = usage
            .request_bytes
            .checked_add(total_response_bytes)
            .ok_or_else(|| quota_error(key.kind, "total-byte"))?;
        if total_bytes > quota.maximum_total_bytes {
            return Err(quota_error(key.kind, "total-byte"));
        }
        usage.response_bytes = total_response_bytes;
        Ok(())
    }

    fn remaining_response_allowance(
        &self,
        key: &CapabilityKey,
        quota: CapabilityQuota,
    ) -> Result<u64, InvocationError> {
        let usage = self.usage.get(key).ok_or_else(|| {
            InvocationError::HostFailure("capability usage was not initialized".to_owned())
        })?;
        let used_total = usage
            .request_bytes
            .checked_add(usage.response_bytes)
            .ok_or_else(|| quota_error(key.kind, "total-byte"))?;
        let remaining_total = quota
            .maximum_total_bytes
            .checked_sub(used_total)
            .ok_or_else(|| quota_error(key.kind, "total-byte"))?;
        Ok(quota.maximum_response_bytes.min(remaining_total))
    }

    fn preflight_stateful_limits(
        &self,
        call: &CapabilityCall,
        key: &CapabilityKey,
        quota: CapabilityQuota,
    ) -> Result<(), InvocationError> {
        if matches!(
            call,
            CapabilityCall::ModelOpen { .. } | CapabilityCall::OutputBegin { .. }
        ) {
            let handles = self
                .usage
                .get(key)
                .ok_or_else(|| {
                    InvocationError::HostFailure("capability usage was not initialized".to_owned())
                })?
                .handles;
            if handles >= quota.maximum_handles {
                return Err(quota_error(key.kind, "handle"));
            }
        }
        Ok(())
    }

    fn prepare_call(
        &self,
        call: CapabilityCall,
        provider_secret: Option<&SecretId>,
        context: &CapabilityServiceContext,
    ) -> Result<PreparedCall, InvocationError> {
        match call {
            CapabilityCall::FilesystemRead {
                root,
                relative_path,
            } => {
                let identity = plugin_asset_identity(&self.profile_id, &root, &relative_path)?;
                let bytes = self.services.read_asset(&identity, context)?;
                Ok(PreparedCall::Response(CapabilityResponse::Bytes(bytes)))
            }
            CapabilityCall::NetworkProvider {
                provider,
                endpoint,
                body,
                secret_id,
            } => {
                if secret_id.is_some() && provider_secret.is_none() {
                    return Err(InvocationError::HostFailure(
                        "provider authorization context is missing".to_owned(),
                    ));
                }
                let bytes = self.services.call_provider_with_context(
                    &provider,
                    &endpoint,
                    &body,
                    provider_secret,
                    context,
                )?;
                Ok(PreparedCall::Response(CapabilityResponse::Bytes(bytes)))
            }
            CapabilityCall::SecretExists { identifier } => {
                Ok(PreparedCall::Response(CapabilityResponse::Boolean(
                    self.services
                        .secret_exists_with_context(&identifier, context)?,
                )))
            }
            CapabilityCall::ClockNow { clock } => {
                let milliseconds = self
                    .services
                    .clock_milliseconds_with_context(&clock, context)?;
                Ok(PreparedCall::Response(
                    CapabilityResponse::TimestampMilliseconds(milliseconds),
                ))
            }
            CapabilityCall::RandomBytes { stream, length } => {
                let bytes = self
                    .services
                    .random_bytes_with_context(&stream, length, context)?;
                Ok(PreparedCall::Response(CapabilityResponse::Bytes(bytes)))
            }
            CapabilityCall::ModelOpen { identifier } => {
                let model = self
                    .services
                    .open_model_with_context(&identifier, context)?;
                Ok(PreparedCall::Model { model })
            }
            CapabilityCall::OutputBegin { namespace, name } => {
                validate_output_segment(&namespace)?;
                validate_output_segment(&name)?;
                Ok(PreparedCall::OutputBegin { namespace, name })
            }
            CapabilityCall::OutputWrite { transaction, bytes } => {
                if !self.output_buffers.contains_key(&transaction) {
                    return Err(InvocationError::InvalidHandle);
                }
                Ok(PreparedCall::OutputWrite { transaction, bytes })
            }
            CapabilityCall::OutputCommit { transaction } => {
                let output = self
                    .output_buffers
                    .get(&transaction)
                    .cloned()
                    .ok_or(InvocationError::InvalidHandle)?;
                let identifier = output_identifier(&output);
                Ok(PreparedCall::OutputCommit {
                    transaction,
                    identifier,
                })
            }
            CapabilityCall::Log { level, message } => {
                if !matches!(level.as_str(), "debug" | "info" | "warn" | "error") {
                    return Err(InvocationError::InvalidCapabilityRequest(
                        "unsupported log level".to_owned(),
                    ));
                }
                let message = self
                    .services
                    .sanitize_log_with_context(&level, &message, context)?;
                Ok(PreparedCall::Log { level, message })
            }
            CapabilityCall::UiSet {
                contribution,
                state,
            } => {
                if !self.ui_contributions.is_empty()
                    && !self.ui_contributions.contains(&contribution)
                {
                    return Err(InvocationError::CapabilityDenied {
                        kind: CapabilityKind::DeclarativeUi,
                        scope: contribution,
                    });
                }
                Ok(PreparedCall::Ui {
                    contribution,
                    state,
                })
            }
            CapabilityCall::RouteRespond {
                route,
                status,
                body,
            } => {
                if !(100..=599).contains(&status) {
                    return Err(InvocationError::InvalidCapabilityRequest(
                        "invalid route response status".to_owned(),
                    ));
                }
                if !self.route_response_limits.is_empty() {
                    let maximum_bytes =
                        self.route_response_limits.get(&route).ok_or_else(|| {
                            InvocationError::CapabilityDenied {
                                kind: CapabilityKind::Route,
                                scope: route.clone(),
                            }
                        })?;
                    let body_bytes = u64::try_from(body.len()).map_err(|_| {
                        InvocationError::InvalidCapabilityRequest(
                            "route response body is too large".to_owned(),
                        )
                    })?;
                    if body_bytes > *maximum_bytes {
                        return Err(quota_error(CapabilityKind::Route, "route-response-byte"));
                    }
                }
                Ok(PreparedCall::Route {
                    route,
                    status,
                    body,
                })
            }
        }
    }

    fn apply_prepared(
        &mut self,
        prepared: PreparedCall,
    ) -> Result<CapabilityResponse, InvocationError> {
        match prepared {
            PreparedCall::Response(response) => Ok(response),
            PreparedCall::Model { model } => {
                let key = CapabilityKey {
                    kind: CapabilityKind::Model,
                    scope: model.identifier().to_owned(),
                };
                let quota = self.quotas.get(&key).copied().ok_or_else(|| {
                    InvocationError::CapabilityDenied {
                        kind: key.kind,
                        scope: key.scope.clone(),
                    }
                })?;
                let usage = self.usage.get_mut(&key).ok_or_else(|| {
                    InvocationError::HostFailure("model capability usage is missing".to_owned())
                })?;
                usage.handles = usage
                    .handles
                    .checked_add(1)
                    .ok_or_else(|| quota_error(CapabilityKind::Model, "handle"))?;
                if usage.handles > quota.maximum_handles {
                    return Err(quota_error(CapabilityKind::Model, "handle"));
                }
                Ok(CapabilityResponse::Handle(u64::from(usage.handles)))
            }
            PreparedCall::OutputBegin { namespace, name } => {
                let key = CapabilityKey {
                    kind: CapabilityKind::TransactionalOutput,
                    scope: namespace.clone(),
                };
                let quota = self.quotas.get(&key).copied().ok_or_else(|| {
                    InvocationError::CapabilityDenied {
                        kind: key.kind,
                        scope: key.scope.clone(),
                    }
                })?;
                let usage = self.usage.get_mut(&key).ok_or_else(|| {
                    InvocationError::HostFailure("output capability usage is missing".to_owned())
                })?;
                usage.handles = usage
                    .handles
                    .checked_add(1)
                    .ok_or_else(|| quota_error(CapabilityKind::TransactionalOutput, "handle"))?;
                if usage.handles > quota.maximum_handles {
                    return Err(quota_error(CapabilityKind::TransactionalOutput, "handle"));
                }
                let transaction = self.next_output_buffer;
                self.next_output_buffer = self
                    .next_output_buffer
                    .checked_add(1)
                    .ok_or_else(|| quota_error(CapabilityKind::TransactionalOutput, "handle"))?;
                self.output_buffers.insert(
                    transaction,
                    OutputProposalBuffer {
                        namespace,
                        name,
                        bytes: Vec::new(),
                    },
                );
                Ok(CapabilityResponse::Handle(transaction))
            }
            PreparedCall::OutputWrite { transaction, bytes } => {
                let output = self
                    .output_buffers
                    .get_mut(&transaction)
                    .ok_or(InvocationError::InvalidHandle)?;
                output.bytes.extend_from_slice(&bytes);
                Ok(CapabilityResponse::Unit)
            }
            PreparedCall::OutputCommit {
                transaction,
                identifier,
            } => {
                let output = self
                    .output_buffers
                    .remove(&transaction)
                    .ok_or(InvocationError::InvalidHandle)?;
                self.pending_effects.outputs.push(PluginOutputProposal {
                    identifier: identifier.clone(),
                    namespace: output.namespace,
                    name: output.name,
                    bytes: output.bytes,
                });
                Ok(CapabilityResponse::CommittedArtifact(identifier))
            }
            PreparedCall::Log { level, message } => {
                self.pending_effects
                    .logs
                    .push(format!("{level}: {message}"));
                Ok(CapabilityResponse::Unit)
            }
            PreparedCall::Ui {
                contribution,
                state,
            } => {
                self.pending_effects.ui_state.insert(contribution, state);
                Ok(CapabilityResponse::Unit)
            }
            PreparedCall::Route {
                route,
                status,
                body,
            } => {
                self.pending_effects.routes.push(RouteEffect {
                    route,
                    status,
                    body,
                });
                Ok(CapabilityResponse::Unit)
            }
        }
    }
}

impl Drop for CapabilityState {
    fn drop(&mut self) {
        if !self.terminal {
            self.services.abort_invocation();
            self.output_buffers.clear();
            self.pending_effects = CapabilityEffects::default();
            self.terminal = true;
        }
    }
}

enum PreparedCall {
    Response(CapabilityResponse),
    Model {
        model: ModelValue,
    },
    OutputBegin {
        namespace: String,
        name: String,
    },
    OutputWrite {
        transaction: u64,
        bytes: Vec<u8>,
    },
    OutputCommit {
        transaction: u64,
        identifier: String,
    },
    Log {
        level: String,
        message: String,
    },
    Ui {
        contribution: String,
        state: Vec<u8>,
    },
    Route {
        route: String,
        status: u16,
        body: Vec<u8>,
    },
}

impl PreparedCall {
    fn response_size(&self) -> Result<u64, InvocationError> {
        let length = match self {
            Self::Response(CapabilityResponse::Bytes(bytes)) => bytes.len(),
            Self::Response(CapabilityResponse::Boolean(_)) => 1,
            Self::Response(CapabilityResponse::TimestampMilliseconds(_))
            | Self::Response(CapabilityResponse::Handle(_))
            | Self::Model { .. }
            | Self::OutputBegin { .. } => 8,
            Self::Response(CapabilityResponse::CommittedArtifact(identifier))
            | Self::OutputCommit { identifier, .. } => identifier.len(),
            Self::Response(CapabilityResponse::Unit)
            | Self::OutputWrite { .. }
            | Self::Log { .. }
            | Self::Ui { .. }
            | Self::Route { .. } => 0,
        };
        u64::try_from(length)
            .map_err(|_| InvocationError::HostFailure("response is too large".to_owned()))
    }
}

fn validate_random_response(
    prepared: &PreparedCall,
    expected_length: u32,
) -> Result<(), InvocationError> {
    let PreparedCall::Response(CapabilityResponse::Bytes(bytes)) = prepared else {
        return Err(InvocationError::HostFailure(
            "randomness service returned an invalid response".to_owned(),
        ));
    };
    let expected_length = usize::try_from(expected_length).map_err(|_| {
        InvocationError::InvalidCapabilityRequest("random response length overflow".to_owned())
    })?;
    if bytes.len() != expected_length {
        return Err(InvocationError::HostFailure(
            "randomness service returned an unexpected byte count".to_owned(),
        ));
    }
    Ok(())
}

fn validate_provider_secret(call: &CapabilityCall) -> Result<Option<SecretId>, InvocationError> {
    let CapabilityCall::NetworkProvider { secret_id, .. } = call else {
        return Ok(None);
    };
    secret_id
        .as_deref()
        .map(SecretId::new)
        .transpose()
        .map_err(|_| {
            InvocationError::InvalidCapabilityRequest("secret identifier is invalid".to_owned())
        })
}

fn scope_for_call(call: &CapabilityCall) -> Result<String, InvocationError> {
    let scope = match call {
        CapabilityCall::FilesystemRead { root, .. } => root.clone(),
        CapabilityCall::NetworkProvider {
            provider, endpoint, ..
        } => format!("{provider}|{endpoint}"),
        CapabilityCall::SecretExists { identifier } => identifier.clone(),
        CapabilityCall::ClockNow { clock } => clock.clone(),
        CapabilityCall::RandomBytes { stream, .. } => stream.clone(),
        CapabilityCall::ModelOpen { identifier } => identifier.clone(),
        CapabilityCall::OutputBegin { namespace, .. } => namespace.clone(),
        CapabilityCall::OutputWrite { .. } | CapabilityCall::OutputCommit { .. } => {
            "output-transaction".to_owned()
        }
        CapabilityCall::Log { level, .. } => level.clone(),
        CapabilityCall::UiSet { contribution, .. } => contribution.clone(),
        CapabilityCall::RouteRespond { route, .. } => route.clone(),
    };
    if scope.is_empty() || scope.len() > 1_024 {
        return Err(InvocationError::InvalidCapabilityRequest(
            "capability scope is empty or too long".to_owned(),
        ));
    }
    Ok(scope)
}

fn request_size(call: &CapabilityCall) -> Result<u64, InvocationError> {
    let length = match call {
        CapabilityCall::FilesystemRead {
            root,
            relative_path,
        } => root.len().checked_add(relative_path.len()),
        CapabilityCall::NetworkProvider {
            provider,
            endpoint,
            body,
            secret_id,
        } => provider
            .len()
            .checked_add(endpoint.len())
            .and_then(|length| length.checked_add(body.len()))
            .and_then(|length| {
                length.checked_add(secret_id.as_ref().map_or(0, std::string::String::len))
            }),
        CapabilityCall::SecretExists { identifier } | CapabilityCall::ModelOpen { identifier } => {
            Some(identifier.len())
        }
        CapabilityCall::ClockNow { clock } => Some(clock.len()),
        CapabilityCall::RandomBytes { stream, .. } => Some(stream.len()),
        CapabilityCall::OutputBegin { namespace, name } => namespace.len().checked_add(name.len()),
        CapabilityCall::OutputWrite { bytes, .. } => bytes.len().checked_add(8),
        CapabilityCall::OutputCommit { .. } => Some(8),
        CapabilityCall::Log { level, message } => level.len().checked_add(message.len()),
        CapabilityCall::UiSet {
            contribution,
            state,
        } => contribution.len().checked_add(state.len()),
        CapabilityCall::RouteRespond { route, body, .. } => route
            .len()
            .checked_add(body.len())
            .and_then(|size| size.checked_add(2)),
    }
    .ok_or_else(|| InvocationError::HostFailure("capability request size overflow".to_owned()))?;
    u64::try_from(length)
        .map_err(|_| InvocationError::HostFailure("capability request is too large".to_owned()))
}

fn known_response_size(call: &CapabilityCall) -> u64 {
    match call {
        CapabilityCall::SecretExists { .. } => 1,
        CapabilityCall::ClockNow { .. }
        | CapabilityCall::ModelOpen { .. }
        | CapabilityCall::OutputBegin { .. } => 8,
        CapabilityCall::RandomBytes { length, .. } => u64::from(*length),
        CapabilityCall::FilesystemRead { .. }
        | CapabilityCall::NetworkProvider { .. }
        | CapabilityCall::OutputWrite { .. }
        | CapabilityCall::OutputCommit { .. }
        | CapabilityCall::Log { .. }
        | CapabilityCall::UiSet { .. }
        | CapabilityCall::RouteRespond { .. } => 0,
    }
}

fn quota_error(kind: CapabilityKind, limit: &str) -> InvocationError {
    InvocationError::QuotaExceeded {
        kind,
        limit: limit.to_owned(),
    }
}

fn validate_output_segment(value: &str) -> Result<(), InvocationError> {
    if value.is_empty()
        || value.len() > 256
        || value.contains('/')
        || value.contains('\\')
        || value.contains('\0')
        || value == "."
        || value == ".."
    {
        return Err(InvocationError::InvalidCapabilityRequest(
            "invalid output namespace or name".to_owned(),
        ));
    }
    Ok(())
}

fn unavailable_service(name: &str) -> InvocationError {
    InvocationError::HostFailure(format!("canonical {name} service is unavailable"))
}

fn output_identifier(output: &OutputProposalBuffer) -> String {
    let digest = Sha256::digest(&output.bytes);
    format!(
        "{}/{}#{}",
        output.namespace,
        output.name,
        encode_hex(&digest)
    )
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

fn sanitize_log(message: &str, secret_identifiers: &BTreeSet<String>) -> String {
    let mut sanitized: String = message
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .take(8_192)
        .collect();
    for secret in secret_identifiers {
        sanitized = sanitized.replace(secret, "[REDACTED]");
    }
    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_plugin_sdk::ProviderStreamingContractError;
    use sim_runtime::{
        AssetQuery, AssetRoots, AttemptId, ProfileId, PromptId, authorize_native_output_committer,
        authorize_native_plugin_asset_broker,
    };
    use std::{
        error::Error,
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn artifact_abi_value_maps_once_to_canonical_asset_identity() -> Result<(), Box<dyn Error>> {
        let value = ArtifactValue::new("input", "nested/fixture.bin", 7, "1".repeat(64))?;
        let identity = artifact_value_identity("profile-a", &value)?;
        assert_eq!(identity.profile_id, "profile-a");
        assert_eq!(identity.namespace, AssetNamespace::Input);
        assert_eq!(identity.relative_path, Path::new("nested/fixture.bin"));

        let traversal = ArtifactValue::new("input", "../escape.bin", 1, "2".repeat(64))?;
        assert!(matches!(
            artifact_value_identity("profile-a", &traversal),
            Err(InvocationError::InvalidCapabilityRequest(_))
        ));
        let unsupported_root = ArtifactValue::new("private", "fixture.bin", 1, "3".repeat(64))?;
        assert!(matches!(
            artifact_value_identity("profile-a", &unsupported_root),
            Err(InvocationError::InvalidCapabilityRequest(_))
        ));
        assert!(matches!(
            artifact_value_identity("", &value),
            Err(InvocationError::InvalidCapabilityRequest(_))
        ));
        Ok(())
    }

    #[derive(Default)]
    struct TestServices {
        provider_responses: BTreeMap<(String, String), Vec<u8>>,
    }

    impl PluginCapabilityServices for TestServices {
        fn read_asset(
            &self,
            _identity: &AssetIdentity,
            _context: &CapabilityServiceContext,
        ) -> Result<Vec<u8>, InvocationError> {
            Err(unavailable_service("artifact"))
        }

        fn call_provider(
            &self,
            provider: &str,
            endpoint: &str,
            _body: &[u8],
            _secret_id: Option<&SecretId>,
        ) -> Result<Vec<u8>, InvocationError> {
            self.provider_responses
                .get(&(provider.to_owned(), endpoint.to_owned()))
                .cloned()
                .ok_or_else(|| unavailable_service("provider"))
        }

        fn secret_exists(&self, _identifier: &str) -> Result<bool, InvocationError> {
            Ok(false)
        }

        fn clock_milliseconds(&self, _clock: &str) -> Result<u64, InvocationError> {
            Err(unavailable_service("clock"))
        }

        fn random_bytes(&self, _stream: &str, length: u32) -> Result<Vec<u8>, InvocationError> {
            let length = usize::try_from(length)
                .map_err(|_| InvocationError::HostFailure("random length overflow".to_owned()))?;
            Ok(vec![7; length])
        }

        fn open_model(&self, _identifier: &str) -> Result<ModelValue, InvocationError> {
            Err(unavailable_service("model"))
        }

        fn sanitize_log(&self, _level: &str, message: &str) -> Result<String, InvocationError> {
            Ok(sanitize_log(message, &BTreeSet::new()))
        }
    }

    #[derive(Default)]
    struct RecordingServices {
        file_reads: AtomicUsize,
        provider_calls: AtomicUsize,
        model_opens: AtomicUsize,
        random_calls: AtomicUsize,
        finishes: AtomicUsize,
        aborts: AtomicUsize,
    }

    impl PluginCapabilityServices for RecordingServices {
        fn read_asset(
            &self,
            _identity: &AssetIdentity,
            _context: &CapabilityServiceContext,
        ) -> Result<Vec<u8>, InvocationError> {
            self.file_reads.fetch_add(1, Ordering::SeqCst);
            Ok(b"file".to_vec())
        }

        fn call_provider(
            &self,
            _provider: &str,
            _endpoint: &str,
            _body: &[u8],
            _secret_id: Option<&SecretId>,
        ) -> Result<Vec<u8>, InvocationError> {
            self.provider_calls.fetch_add(1, Ordering::SeqCst);
            Ok(b"provider".to_vec())
        }

        fn secret_exists(&self, _identifier: &str) -> Result<bool, InvocationError> {
            Ok(false)
        }

        fn clock_milliseconds(&self, _clock: &str) -> Result<u64, InvocationError> {
            Ok(0)
        }

        fn random_bytes(&self, _stream: &str, length: u32) -> Result<Vec<u8>, InvocationError> {
            self.random_calls.fetch_add(1, Ordering::SeqCst);
            let length = usize::try_from(length)
                .map_err(|_| InvocationError::HostFailure("random length overflow".to_owned()))?;
            Ok(vec![0; length.saturating_add(1)])
        }

        fn open_model(&self, identifier: &str) -> Result<ModelValue, InvocationError> {
            self.model_opens.fetch_add(1, Ordering::SeqCst);
            ModelValue::new(identifier, "safetensors", "0".repeat(64))
                .map_err(|error| InvocationError::HostFailure(error.to_string()))
        }

        fn sanitize_log(&self, _level: &str, message: &str) -> Result<String, InvocationError> {
            Ok(message.to_owned())
        }

        fn finish_invocation(&self) -> Result<(), InvocationError> {
            self.finishes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn abort_invocation(&self) {
            self.aborts.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn quota() -> CapabilityQuota {
        CapabilityQuota {
            maximum_operations: 8,
            maximum_request_bytes: 1_024,
            maximum_response_bytes: 1_024,
            maximum_total_bytes: 8_192,
            maximum_handles: 2,
            timeout_milliseconds: 1_000,
        }
    }

    struct OutputHarness {
        directory: PathBuf,
        scope: OutputExecutionScope,
        committer: OutputCommitter,
        assets: AssetService,
        capabilities: AuthorizedCapabilities,
        asset_reader: AuthorizedCapabilities,
    }

    fn output_harness(label: &str) -> Result<OutputHarness, Box<dyn Error>> {
        let profile_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, label.as_bytes());
        let profile_id = profile_uuid.to_string();
        let directory = std::env::temp_dir().join(format!(
            "sim-plugin-output-{label}-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        let paths = [
            AssetNamespace::Input,
            AssetNamespace::Output,
            AssetNamespace::Temporary,
            AssetNamespace::Model,
            AssetNamespace::Plugin,
        ]
        .into_iter()
        .map(|namespace| {
            let path = directory.join(namespace.locator_type());
            fs::create_dir_all(&path)?;
            Ok((namespace, path))
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
        let roots = AssetRoots::new(profile_id.clone(), paths)?;
        let committer = OutputCommitter::open(roots.clone())?;
        let assets = AssetService::open(roots)?;
        let capabilities = authorize_native_output_committer(profile_id.clone())?;
        let asset_reader = authorize_native_plugin_asset_broker(profile_id)?;
        Ok(OutputHarness {
            directory,
            scope: OutputExecutionScope {
                profile_id: ProfileId(profile_uuid),
                prompt_id: PromptId(Uuid::new_v5(&profile_uuid, b"prompt")),
                attempt_id: AttemptId(Uuid::new_v5(&profile_uuid, b"attempt")),
            },
            committer,
            assets,
            capabilities,
            asset_reader,
        })
    }

    fn staged_plugin_output(
        namespace: &str,
        name: &str,
        bytes: &[u8],
    ) -> Result<PluginOutputProposal, InvocationError> {
        let mut state = CapabilityState::with_test_declarations(
            [
                (
                    CapabilityKind::TransactionalOutput,
                    namespace.to_owned(),
                    quota(),
                ),
                (
                    CapabilityKind::TransactionalOutput,
                    "output-transaction".to_owned(),
                    quota(),
                ),
            ],
            "test-profile".to_owned(),
            Arc::new(TestServices::default()),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        let CapabilityResponse::Handle(transaction) =
            state.execute(CapabilityCall::OutputBegin {
                namespace: namespace.to_owned(),
                name: name.to_owned(),
            })?
        else {
            return Err(InvocationError::HostFailure(
                "plugin output begin returned no transaction handle".to_owned(),
            ));
        };
        state.execute(CapabilityCall::OutputWrite {
            transaction,
            bytes: bytes.to_vec(),
        })?;
        state.execute(CapabilityCall::OutputCommit { transaction })?;
        state.finish()?.outputs.pop().ok_or_else(|| {
            InvocationError::HostFailure("plugin output proposal is missing".to_owned())
        })
    }

    fn regular_file_count(directory: &Path) -> Result<usize, std::io::Error> {
        let mut count = 0;
        for entry in fs::read_dir(directory)? {
            if entry?.file_type()?.is_file() {
                count += 1;
            }
        }
        Ok(count)
    }

    #[test]
    fn denied_calls_fail_before_effects() -> Result<(), InvocationError> {
        let mut state = CapabilityState::with_test_declarations(
            [],
            "test-profile".to_owned(),
            Arc::new(TestServices::default()),
            Default::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        assert!(matches!(
            state.execute(CapabilityCall::Log {
                level: "info".to_owned(),
                message: "not written".to_owned(),
            }),
            Err(InvocationError::CapabilityDenied { .. })
        ));
        assert!(state.pending_effects.logs.is_empty());
        Ok(())
    }

    #[test]
    fn val_cancel_001_plugin_adapter_rolls_back_staged_effects() -> Result<(), InvocationError> {
        let cancellation = CancellationToken::default();
        let mut state = CapabilityState::with_test_declarations(
            [(CapabilityKind::SanitizedLog, "info".to_owned(), quota())],
            "test-profile".to_owned(),
            Arc::new(TestServices::default()),
            cancellation.clone(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        state.execute(CapabilityCall::Log {
            level: "info".to_owned(),
            message: "staged".to_owned(),
        })?;
        cancellation.cancel();
        assert!(matches!(state.finish(), Err(InvocationError::Cancelled)));
        Ok(())
    }

    #[test]
    fn timeout_is_rechecked_before_response_effects() -> Result<(), InvocationError> {
        let key = CapabilityKey {
            kind: CapabilityKind::SanitizedLog,
            scope: "info".to_owned(),
        };
        let mut state = CapabilityState::with_test_declarations(
            [(CapabilityKind::SanitizedLog, "info".to_owned(), quota())],
            "test-profile".to_owned(),
            Arc::new(TestServices::default()),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        let quota = state.charge_request(&key, 1)?;
        state.started_at = Instant::now() - Duration::from_secs(2);
        assert!(matches!(
            state.charge_response(&key, quota, 1),
            Err(InvocationError::TimedOut)
        ));
        Ok(())
    }

    #[test]
    fn provider_calls_delegate_authorization_to_the_canonical_service()
    -> Result<(), InvocationError> {
        let mut provider_quota = quota();
        provider_quota.maximum_operations = 1;
        let mut services = TestServices::default();
        services.provider_responses.insert(
            (
                "demo".to_owned(),
                "https://demo.invalid/v1/generate".to_owned(),
            ),
            b"ok".to_vec(),
        );
        let mut state = CapabilityState::with_test_declarations(
            [(
                CapabilityKind::NetworkProvider,
                "demo|https://demo.invalid/v1/generate".to_owned(),
                provider_quota,
            )],
            "test-profile".to_owned(),
            Arc::new(services),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        let call = || CapabilityCall::NetworkProvider {
            provider: "demo".to_owned(),
            endpoint: "https://demo.invalid/v1/generate".to_owned(),
            body: Vec::new(),
            secret_id: None,
        };
        assert_eq!(
            state.execute(call())?,
            CapabilityResponse::Bytes(b"ok".to_vec())
        );
        assert!(matches!(
            state.execute(call()),
            Err(InvocationError::QuotaExceeded { limit, .. }) if limit == "operation"
        ));
        Ok(())
    }

    #[test]
    fn provider_capability_must_be_declared_before_reaching_the_canonical_service()
    -> Result<(), InvocationError> {
        let mut state = CapabilityState::with_test_declarations(
            [(CapabilityKind::Secret, "secret.demo".to_owned(), quota())],
            "test-profile".to_owned(),
            Arc::new(TestServices::default()),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;

        assert!(matches!(
            state.execute(CapabilityCall::NetworkProvider {
                provider: "demo".to_owned(),
                endpoint: "https://demo.invalid/v1/generate".to_owned(),
                body: Vec::new(),
                secret_id: Some("secret.demo".to_owned()),
            }),
            Err(InvocationError::CapabilityDenied {
                kind: CapabilityKind::NetworkProvider,
                ..
            })
        ));
        assert!(state.usage.is_empty());
        Ok(())
    }

    #[test]
    fn declarations_and_random_allocation_are_bounded_before_effects() -> Result<(), InvocationError>
    {
        let mut route_quota = quota();
        route_quota.maximum_response_bytes = 4;
        let declarations = [
            (
                CapabilityKind::DeclarativeUi,
                "panel.other".to_owned(),
                quota(),
            ),
            (CapabilityKind::Route, "route.demo".to_owned(), quota()),
            (CapabilityKind::Randomness, "stream".to_owned(), route_quota),
        ];
        let mut state = CapabilityState::with_test_declarations(
            declarations,
            "test-profile".to_owned(),
            Arc::new(TestServices::default()),
            CancellationToken::default(),
            BTreeSet::from(["panel.demo".to_owned()]),
            BTreeMap::from([("route.demo".to_owned(), 2)]),
        )?;
        assert!(matches!(
            state.execute(CapabilityCall::UiSet {
                contribution: "panel.other".to_owned(),
                state: Vec::new(),
            }),
            Err(InvocationError::CapabilityDenied {
                kind: CapabilityKind::DeclarativeUi,
                ..
            })
        ));
        assert!(matches!(
            state.execute(CapabilityCall::RouteRespond {
                route: "route.demo".to_owned(),
                status: 200,
                body: vec![1, 2, 3],
            }),
            Err(InvocationError::QuotaExceeded { limit, .. })
                if limit == "route-response-byte"
        ));
        assert!(matches!(
            state.execute(CapabilityCall::RandomBytes {
                stream: "stream".to_owned(),
                length: 1_000_000,
            }),
            Err(InvocationError::QuotaExceeded { limit, .. })
                if limit == "response-byte"
        ));
        assert!(state.pending_effects.ui_state.is_empty());
        assert!(state.pending_effects.routes.is_empty());
        Ok(())
    }

    #[test]
    fn val_plugin_001_host_ceilings_and_preflight_failures_do_not_reach_services()
    -> Result<(), InvocationError> {
        let services = Arc::new(RecordingServices::default());
        let mut denied = CapabilityState::with_test_declarations(
            [],
            "test-profile".to_owned(),
            services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        assert!(matches!(
            denied.execute(CapabilityCall::FilesystemRead {
                root: "input".to_owned(),
                relative_path: "fixture.bin".to_owned(),
            }),
            Err(InvocationError::CapabilityDenied { .. })
        ));
        assert_eq!(services.file_reads.load(Ordering::SeqCst), 0);

        let mut excessive = quota();
        excessive.maximum_total_bytes = CapabilityLimits::default()
            .maximum_total_bytes
            .checked_add(1)
            .ok_or_else(|| InvocationError::HostFailure("test quota overflow".to_owned()))?;
        excessive.maximum_response_bytes = excessive.maximum_total_bytes;
        assert!(matches!(
            CapabilityState::with_test_declarations(
                [(CapabilityKind::Filesystem, "input".to_owned(), excessive)],
                "test-profile".to_owned(),
                services.clone(),
                CancellationToken::default(),
                BTreeSet::new(),
                BTreeMap::new(),
            ),
            Err(InvocationError::InvalidCapabilityRequest(_))
        ));
        assert_eq!(services.file_reads.load(Ordering::SeqCst), 0);

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let mut cancelled = CapabilityState::with_test_declarations(
            [(CapabilityKind::Filesystem, "input".to_owned(), quota())],
            "test-profile".to_owned(),
            services.clone(),
            cancellation,
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        assert!(matches!(
            cancelled.execute(CapabilityCall::FilesystemRead {
                root: "input".to_owned(),
                relative_path: "fixture.bin".to_owned(),
            }),
            Err(InvocationError::Cancelled)
        ));
        assert_eq!(services.file_reads.load(Ordering::SeqCst), 0);

        let mut timed_out = CapabilityState::with_test_declarations(
            [(CapabilityKind::Filesystem, "input".to_owned(), quota())],
            "test-profile".to_owned(),
            services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        timed_out.started_at = Instant::now() - Duration::from_secs(2);
        assert!(matches!(
            timed_out.execute(CapabilityCall::FilesystemRead {
                root: "input".to_owned(),
                relative_path: "fixture.bin".to_owned(),
            }),
            Err(InvocationError::TimedOut)
        ));
        assert_eq!(services.file_reads.load(Ordering::SeqCst), 0);

        let mut model_quota = quota();
        model_quota.maximum_handles = 1;
        let mut models = CapabilityState::with_test_declarations(
            [(CapabilityKind::Model, "model.demo".to_owned(), model_quota)],
            "test-profile".to_owned(),
            services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        models.execute(CapabilityCall::ModelOpen {
            identifier: "model.demo".to_owned(),
        })?;
        assert!(matches!(
            models.execute(CapabilityCall::ModelOpen {
                identifier: "model.demo".to_owned(),
            }),
            Err(InvocationError::QuotaExceeded { limit, .. }) if limit == "handle"
        ));
        assert_eq!(services.model_opens.load(Ordering::SeqCst), 1);

        let mut response_limited_model_quota = quota();
        response_limited_model_quota.maximum_response_bytes = 1;
        let mut response_limited_model = CapabilityState::with_test_declarations(
            [(
                CapabilityKind::Model,
                "model.demo".to_owned(),
                response_limited_model_quota,
            )],
            "test-profile".to_owned(),
            services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        assert!(matches!(
            response_limited_model.execute(CapabilityCall::ModelOpen {
                identifier: "model.demo".to_owned(),
            }),
            Err(InvocationError::QuotaExceeded { limit, .. }) if limit == "response-byte"
        ));
        assert_eq!(services.model_opens.load(Ordering::SeqCst), 1);

        let mut provider_quota = quota();
        provider_quota.maximum_operations = 1;
        let mut provider = CapabilityState::with_test_declarations(
            [(
                CapabilityKind::NetworkProvider,
                "demo|https://demo.invalid/v1/generate".to_owned(),
                provider_quota,
            )],
            "test-profile".to_owned(),
            services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        let provider_call = || CapabilityCall::NetworkProvider {
            provider: "demo".to_owned(),
            endpoint: "https://demo.invalid/v1/generate".to_owned(),
            body: Vec::new(),
            secret_id: None,
        };
        provider.execute(provider_call())?;
        assert!(matches!(
            provider.execute(provider_call()),
            Err(InvocationError::QuotaExceeded { limit, .. }) if limit == "operation"
        ));
        assert_eq!(services.provider_calls.load(Ordering::SeqCst), 1);

        let mut randomness = CapabilityState::with_test_declarations(
            [(CapabilityKind::Randomness, "stream".to_owned(), quota())],
            "test-profile".to_owned(),
            services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        assert!(matches!(
            randomness.execute(CapabilityCall::RandomBytes {
                stream: "stream".to_owned(),
                length: 4,
            }),
            Err(InvocationError::QuotaExceeded { limit, .. }) if limit == "response-byte"
        ));
        assert_eq!(services.random_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn plugin_output_mapping_is_deterministic_and_attempt_scoped() -> Result<(), Box<dyn Error>> {
        let scope = OutputExecutionScope {
            profile_id: ProfileId(Uuid::from_u128(1)),
            prompt_id: PromptId(Uuid::from_u128(2)),
            attempt_id: AttemptId(Uuid::from_u128(3)),
        };
        let proposals = [
            PluginOutputProposal {
                identifier: "outputs/result.png#digest-a".to_owned(),
                namespace: "outputs".to_owned(),
                name: "result.PNG".to_owned(),
                bytes: b"image".to_vec(),
            },
            PluginOutputProposal {
                identifier: "temporary/preview.bin#digest-b".to_owned(),
                namespace: "temporary".to_owned(),
                name: "preview.bin".to_owned(),
                bytes: b"preview".to_vec(),
            },
        ];
        let adapter = PluginOutputPublicationAdapter;
        let first = adapter.map_proposals(&scope, &proposals)?;
        let second = adapter.map_proposals(&scope, &proposals)?;
        assert_eq!(first, second);
        assert_eq!(first[0].namespace(), AssetNamespace::Output);
        assert_eq!(first[0].filename_prefix(), "result");
        assert_eq!(first[0].extension(), "png");
        assert_eq!(first[0].batch_index(), 0);
        assert_eq!(first[0].content(), b"image");
        assert_eq!(first[1].namespace(), AssetNamespace::Temporary);
        assert_eq!(first[1].batch_index(), 1);
        assert_eq!(
            first[0].proposal_id(),
            Uuid::new_v5(&scope.attempt_id.0, proposals[0].identifier.as_bytes())
        );

        let other_scope = OutputExecutionScope {
            attempt_id: AttemptId(Uuid::from_u128(4)),
            ..scope
        };
        let other = adapter.map_proposals(&other_scope, &proposals)?;
        assert_ne!(first[0].proposal_id(), other[0].proposal_id());
        Ok(())
    }

    #[test]
    fn plugin_output_mapping_rejects_unsafe_namespace_name_and_extension() {
        let scope = OutputExecutionScope {
            profile_id: ProfileId(Uuid::from_u128(11)),
            prompt_id: PromptId(Uuid::from_u128(12)),
            attempt_id: AttemptId(Uuid::from_u128(13)),
        };
        let adapter = PluginOutputPublicationAdapter;
        let proposal = |namespace: &str, name: &str| PluginOutputProposal {
            identifier: "proposal".to_owned(),
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            bytes: Vec::new(),
        };

        assert!(matches!(
            adapter.map_proposals(&scope, &[proposal("input", "result.png")]),
            Err(PluginOutputPublicationError::UnsupportedNamespace(namespace))
                if namespace == "input"
        ));
        assert!(matches!(
            adapter.map_proposals(&scope, &[proposal("output", "../escape.png")]),
            Err(PluginOutputPublicationError::InvalidName(name))
                if name == "../escape.png"
        ));
        assert!(matches!(
            adapter.map_proposals(&scope, &[proposal("output", "result")]),
            Err(PluginOutputPublicationError::MissingExtension(name)) if name == "result"
        ));
        assert!(matches!(
            adapter.map_proposals(&scope, &[proposal("output", "result.bad-extension")]),
            Err(PluginOutputPublicationError::Commit(
                OutputCommitError::InvalidExtension(extension)
            )) if extension == "bad-extension"
        ));
    }

    #[test]
    fn cancelled_plugin_output_publication_rolls_back_without_visibility()
    -> Result<(), Box<dyn Error>> {
        let mut harness = output_harness("cancelled")?;
        let proposal = staged_plugin_output("outputs", "cancelled.bin", b"cancelled")?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let error = PluginOutputPublicationAdapter
            .publish(
                &harness.scope,
                &[proposal],
                &mut harness.committer,
                &mut harness.assets,
                &harness.capabilities,
                &cancellation,
            )
            .expect_err("cancelled plugin output must not publish");
        assert!(matches!(
            error,
            PluginOutputPublicationError::Commit(OutputCommitError::Asset(AssetError::Cancelled))
        ));
        assert!(harness.committer.operations().is_empty());
        assert!(
            harness
                .committer
                .committed_receipts_for_scope(&harness.scope)?
                .is_empty()
        );
        assert_eq!(
            harness
                .assets
                .list_authorized(&AssetQuery::default(), &harness.asset_reader)?
                .total,
            0
        );
        assert_eq!(
            regular_file_count(
                &harness
                    .directory
                    .join(AssetNamespace::Output.locator_type())
            )?,
            0
        );
        fs::remove_dir_all(harness.directory)?;
        Ok(())
    }

    #[test]
    fn plugin_output_effect_has_zero_filesystem_commits_and_one_host_commit()
    -> Result<(), Box<dyn Error>> {
        let mut harness = output_harness("exact-host-commit")?;
        let proposal = staged_plugin_output("outputs", "result.bin", b"plugin-result")?;
        assert!(harness.committer.operations().is_empty());
        assert_eq!(
            harness
                .assets
                .list_authorized(&AssetQuery::default(), &harness.asset_reader)?
                .total,
            0
        );
        assert_eq!(
            regular_file_count(
                &harness
                    .directory
                    .join(AssetNamespace::Output.locator_type())
            )?,
            0
        );

        let expected_proposal_id =
            Uuid::new_v5(&harness.scope.attempt_id.0, proposal.identifier.as_bytes());
        let receipts = PluginOutputPublicationAdapter.publish(
            &harness.scope,
            &[proposal],
            &mut harness.committer,
            &mut harness.assets,
            &harness.capabilities,
            &CancellationToken::default(),
        )?;
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].proposal_id(), expected_proposal_id);
        assert_eq!(harness.committer.operations().len(), 1);
        let scoped_receipts = harness
            .committer
            .committed_receipts_for_scope(&harness.scope)?;
        assert_eq!(scoped_receipts, receipts);
        assert_eq!(
            harness
                .assets
                .list_authorized(&AssetQuery::default(), &harness.asset_reader)?
                .total,
            1
        );
        let output_path = harness
            .directory
            .join(AssetNamespace::Output.locator_type())
            .join(&receipts[0].operation().identity.relative_path);
        assert_eq!(fs::read(output_path)?, b"plugin-result");
        assert_eq!(
            regular_file_count(
                &harness
                    .directory
                    .join(AssetNamespace::Output.locator_type())
            )?,
            1
        );
        fs::remove_dir_all(harness.directory)?;
        Ok(())
    }

    #[test]
    fn capability_state_closes_one_service_session_on_finish_abort_or_drop()
    -> Result<(), InvocationError> {
        let finished_services = Arc::new(RecordingServices::default());
        let finished = CapabilityState::with_test_declarations(
            [],
            "test-profile".to_owned(),
            finished_services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        finished.finish()?;
        assert_eq!(finished_services.finishes.load(Ordering::SeqCst), 1);
        assert_eq!(finished_services.aborts.load(Ordering::SeqCst), 0);

        let aborted_services = Arc::new(RecordingServices::default());
        let mut aborted = CapabilityState::with_test_declarations(
            [],
            "test-profile".to_owned(),
            aborted_services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        aborted.rollback();
        drop(aborted);
        assert_eq!(aborted_services.finishes.load(Ordering::SeqCst), 0);
        assert_eq!(aborted_services.aborts.load(Ordering::SeqCst), 1);

        let dropped_services = Arc::new(RecordingServices::default());
        let dropped = CapabilityState::with_test_declarations(
            [],
            "test-profile".to_owned(),
            dropped_services.clone(),
            CancellationToken::default(),
            BTreeSet::new(),
            BTreeMap::new(),
        )?;
        drop(dropped);
        assert_eq!(dropped_services.finishes.load(Ordering::SeqCst), 0);
        assert_eq!(dropped_services.aborts.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn broker_adapter_has_no_contextless_production_service_path() -> Result<(), InvocationError> {
        let services = BrokerPluginCapabilityServices {
            session: Mutex::new(None),
            log_redactions: BTreeSet::new(),
        };
        let secret = SecretId::new("secret.demo")
            .map_err(|_| InvocationError::HostFailure("fixture secret is invalid".to_owned()))?;

        for result in [
            services
                .call_provider("provider", "/endpoint", b"body", Some(&secret))
                .map(|_| ()),
            services.secret_exists("secret.demo").map(|_| ()),
            services.clock_milliseconds("workflow").map(|_| ()),
            services.random_bytes("noise", 8).map(|_| ()),
            services.open_model("model.demo").map(|_| ()),
            services.sanitize_log("info", "message").map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(InvocationError::HostFailure(message))
                    if message.contains("requires an invocation context")
            ));
        }
        Ok(())
    }

    #[test]
    fn broker_error_mapping_redacts_actuator_paths_and_secret_material() {
        let error = map_broker_error(
            PluginServiceError::ActuatorFailed {
                service: "provider",
                message: "/private/model secret-value".to_owned(),
            },
            CapabilityKind::NetworkProvider,
            "provider|/endpoint",
        );
        let text = error.to_string();
        assert!(!text.contains("/private/model"));
        assert!(!text.contains("secret-value"));
        assert_eq!(
            map_broker_error(
                PluginServiceError::Cancelled,
                CapabilityKind::Randomness,
                "noise",
            ),
            InvocationError::Cancelled
        );
        assert_eq!(
            map_broker_error(
                PluginServiceError::DeadlineExceeded,
                CapabilityKind::Clock,
                "workflow",
            ),
            InvocationError::TimedOut
        );
        assert_eq!(
            map_broker_error(
                PluginServiceError::ResponseTooLarge { maximum: 1 },
                CapabilityKind::Randomness,
                "noise",
            ),
            InvocationError::QuotaExceeded {
                kind: CapabilityKind::Randomness,
                limit: "response-byte".to_owned(),
            }
        );

        for error in [
            PluginServiceError::ProviderSessionUnavailable,
            PluginServiceError::ProviderRuntimeAuthorityDenied,
            PluginServiceError::ProviderRuntimeForeignSession,
            PluginServiceError::ProviderRuntimeStaleSession,
            PluginServiceError::ProviderRuntimeForeignInvocation,
            PluginServiceError::ProviderRuntimeStaleInvocation,
            PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidContract,
            ),
        ] {
            assert_eq!(
                map_broker_error(error, CapabilityKind::NetworkProvider, "provider|/endpoint"),
                InvocationError::CapabilityDenied {
                    kind: CapabilityKind::NetworkProvider,
                    scope: "provider|/endpoint".to_owned(),
                }
            );
        }
    }
}
