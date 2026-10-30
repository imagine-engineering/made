use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use clock::SystemClock;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_model::{
    LoadedModel, ModelFormat, ModelFormatError, ModelLoadAccounting, ModelStore, ModelStoreError,
};
use sim_nodes::NativeSchemaValue;
use sim_plugin_sdk::{
    ProviderCostRequestV2, ProviderCostResponseV2, ProviderHeaderV2, ProviderHttpMethodV2,
    ProviderInvocationContextV2, ProviderProgressV2, ProviderRequestChunkV2, ProviderRequestHeadV2,
    ProviderResponseChunkV2, ProviderResponseFrameEventV2, ProviderResponseFrameV2,
    ProviderResponseHeadV2, ProviderResultReceiptSet, ProviderStreamHandleV2,
    ProviderStreamTerminalV2, ProviderStreamValidatorV2, ProviderStreamingContractError,
    ProviderUploadRequestV2, ProviderUploadValidatorV2, ProviderWaitOutcomeV2,
    ProviderWaitRequestV2,
};
use sim_tensor::{
    CancellationToken, RetryRngPolicy, RngAlgorithm, RngCheckpoint, RngProfileVersion, RngStream,
    RngStreamAddress, RngTransaction,
};
use sim_types::{
    AttemptId, NodeId, ProfileId, PromptId, WorkerProviderCostRequest, WorkerProviderCostResponse,
    WorkerProviderHeader, WorkerProviderHttpMethod, WorkerProviderInvocationContext,
    WorkerProviderProgress, WorkerProviderRequestChunk, WorkerProviderRequestHead,
    WorkerProviderResponseChunk, WorkerProviderResponseFrameEvent, WorkerProviderStreamHandle,
    WorkerProviderTerminal, WorkerProviderUploadCompletion, WorkerProviderUploadCompletionRequest,
    WorkerProviderUploadRequest, WorkerProviderWaitOutcome, WorkerProviderWaitRequest,
    validate_provider_upload_url,
};
use thiserror::Error;

use crate::{
    AssetError, AssetNamespace, AssetOperation, AuthorizedProviderRequest, Capability,
    PluginAuthorization, ProviderCostAcceptance, ProviderCostAcceptanceIssuer,
    ProviderCostAcceptanceScope, ProviderCostAcceptanceVerifier, ProviderCostNonce,
    ProviderInvocationIdentity, ProviderManifestAuthorizationV2, ProviderMaterializationError,
    ProviderPolicy, ProviderPriceBound, ProviderResultReceiptAuthority,
    ProviderResultReceiptSession, ProviderRuntimeReceiptIdentityV2, ProviderRuntimeReceiptIssuerV2,
    ProviderRuntimeReceiptV2, ProviderRuntimeReceiptVerifierV2, ResolvedProviderResult, SecretId,
    SecretValue, SharedAssetService, VerifiedProviderRuntimeReceiptV2,
    WorkerRegistryDeploymentPlan,
};

pub const MAX_PLUGIN_SERVICE_REQUEST_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_PLUGIN_SERVICE_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_PROVIDER_RUNTIME_SESSIONS: usize = 256;
pub const MAX_PROVIDER_RUNTIME_HTTP_REQUESTS: u32 = 8_192;
const MAX_PROVIDER_RUNTIME_IDENTITY_BYTES: usize = 1_024;
const MAX_PROVIDER_RUNTIME_RETRY_AFTER_SECONDS: u64 = 86_400;
const PROVIDER_PROGRESS_MINIMUM_INTERVAL: Duration = Duration::from_millis(50);
const MAX_PLUGIN_SERVICE_IDENTITY_BYTES: usize = 1_024;
const MAX_CONSUMED_PROVIDER_COST_NONCES: usize = 65_536;
const RNG_DEADLINE_CHECK_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProviderWorkerSessionStart {
    pub session_id: String,
    pub registry_generation: u64,
    pub registry_digest_sha256: String,
    pub extension_id: String,
    pub extension_version: String,
    pub plugin_identifier: String,
    pub plugin_version: String,
    pub manifest_digest_sha256: String,
    pub component_digest_sha256: String,
    pub authorization_generation_sha256: String,
    pub binding_set_sha256: String,
    pub node_id: String,
    pub compiled_plan_sha256: String,
    pub maximum_response_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum NativeProviderWorkerRequest {
    Begin(NativeProviderWorkerSessionStart),
    Call {
        session_id: String,
        request: Vec<u8>,
    },
    Resolve {
        session_id: String,
        receipt_set: Vec<u8>,
    },
    Finish {
        session_id: String,
    },
    Abort {
        session_id: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum NativeProviderWorkerResponse {
    Begun,
    Call(Vec<u8>),
    Resolved(Vec<Vec<u8>>),
    Finished,
    Aborted,
    Failure(PluginServiceWireFailure),
}

impl NativeProviderWorkerRequest {
    pub fn to_bytes(&self) -> Result<Vec<u8>, PluginServiceError> {
        let bytes =
            postcard::to_stdvec(self).map_err(|_| PluginServiceError::InvalidWirePayload)?;
        check_request_size(bytes.len())?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PluginServiceError> {
        check_request_size(bytes.len())?;
        postcard::from_bytes(bytes).map_err(|_| PluginServiceError::InvalidWirePayload)
    }
}

impl NativeProviderWorkerResponse {
    pub fn to_bytes(&self) -> Result<Vec<u8>, PluginServiceError> {
        let bytes =
            postcard::to_stdvec(self).map_err(|_| PluginServiceError::InvalidWirePayload)?;
        check_response_size(bytes.len(), MAX_PLUGIN_SERVICE_RESPONSE_BYTES)?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PluginServiceError> {
        check_response_size(bytes.len(), MAX_PLUGIN_SERVICE_RESPONSE_BYTES)?;
        postcard::from_bytes(bytes).map_err(|_| PluginServiceError::InvalidWirePayload)
    }
}

#[derive(Clone)]
pub struct NativeProviderInvocationScope {
    pub profile_id: ProfileId,
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub node_id: NodeId,
    pub cancellation: CancellationToken,
    pub start: NativeProviderWorkerSessionStart,
}

pub trait NativeProviderInvocationAuthority: Send + Sync {
    fn begin(
        &self,
        scope: NativeProviderInvocationScope,
    ) -> Result<PluginCapabilityInvocation, PluginServiceError>;
}

pub struct ProviderRuntimeActivationGrant {
    host_context: Option<WorkerProviderInvocationContext>,
    service_identity: Option<Arc<()>>,
    claim: ProviderRuntimeClaimGuard,
    cancellation: Option<CancellationToken>,
    profile_id: String,
    principal_id: String,
    prompt_id: String,
    prompt_sha256: String,
    attempt_id: String,
    node_id: String,
    request_ordinal: u32,
    registry_generation: u64,
    registry_digest_sha256: String,
    component_generation: u64,
    component_digest_sha256: String,
    provider_manifest_sha256: String,
    authorization_generation_sha256: String,
    binding_generation: u64,
    binding_set_sha256: String,
    compiled_plan_sha256: String,
}

#[derive(Default)]
struct ProviderRuntimeClaimGuard {
    claim: Option<Arc<AtomicBool>>,
}

impl ProviderRuntimeClaimGuard {
    fn armed(claim: Arc<AtomicBool>) -> Self {
        Self { claim: Some(claim) }
    }

    fn disarm(&mut self) -> Result<Arc<AtomicBool>, PluginServiceError> {
        self.claim
            .take()
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)
    }
}

impl Drop for ProviderRuntimeClaimGuard {
    fn drop(&mut self) {
        revoke_provider_runtime_claim(self.claim.as_ref());
    }
}

pub struct PreflightedProviderRuntimeActivationGrant {
    grant: Option<ProviderRuntimeActivationGrant>,
    manifest_authorization: Option<ProviderManifestAuthorizationV2>,
}

impl fmt::Debug for ProviderRuntimeActivationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderRuntimeActivationGrant([SEALED])")
    }
}

impl fmt::Debug for PreflightedProviderRuntimeActivationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PreflightedProviderRuntimeActivationGrant([SEALED])")
    }
}

struct ProviderRuntimeActivationGrantTable {
    session_generations: BTreeMap<uuid::Uuid, u64>,
    grants: BTreeMap<(uuid::Uuid, u64, u64), (u32, Option<ProviderRuntimeActivationGrant>)>,
}

struct ProviderRuntimeStreamState {
    service_identity: Arc<()>,
    owner: ProviderRuntimeStreamOwner,
    grants: ProviderRuntimeActivationGrantTable,
    activation_claims: BTreeMap<(uuid::Uuid, u64, u64, u32), Arc<AtomicBool>>,
    invocation_bindings: BTreeMap<(uuid::Uuid, u64, u64, u32), u64>,
    handles: BTreeMap<WorkerProviderStreamHandle, ProviderStreamHandleV2>,
    main_handles: BTreeSet<WorkerProviderStreamHandle>,
    upload_parents: BTreeMap<WorkerProviderStreamHandle, WorkerProviderStreamHandle>,
    cancellations: BTreeMap<WorkerProviderStreamHandle, CancellationToken>,
    next_slots: BTreeMap<(uuid::Uuid, u64, u64, u32), u32>,
    next_internal_invocation: u64,
    http_invocations: BTreeMap<(uuid::Uuid, u64, u64, u32), ProviderRuntimeHttpInvocationState>,
}

impl ProviderRuntimeStreamState {
    fn new() -> Self {
        Self {
            service_identity: Arc::new(()),
            owner: ProviderRuntimeStreamOwner::new(),
            grants: ProviderRuntimeActivationGrantTable {
                session_generations: BTreeMap::new(),
                grants: BTreeMap::new(),
            },
            activation_claims: BTreeMap::new(),
            invocation_bindings: BTreeMap::new(),
            handles: BTreeMap::new(),
            main_handles: BTreeSet::new(),
            upload_parents: BTreeMap::new(),
            cancellations: BTreeMap::new(),
            next_slots: BTreeMap::new(),
            next_internal_invocation: 0,
            http_invocations: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderRuntimeHttpInvocationIdentity {
    pub context: WorkerProviderInvocationContext,
    pub profile_id: String,
    pub principal_id: String,
    pub prompt_id: String,
    pub prompt_sha256: String,
    pub attempt_id: String,
    pub node_id: String,
    pub compiled_plan_sha256: String,
    pub activation_request_ordinal: u32,
    pub registry_generation: u64,
    pub registry_digest_sha256: String,
    pub component_generation: u64,
    pub component_digest_sha256: String,
    pub authorization_generation_sha256: String,
    pub binding_generation: u64,
    pub binding_set_sha256: String,
    pub provider: String,
    pub provider_manifest_sha256: String,
}

struct ProviderRuntimeHttpInvocationState {
    identity: Arc<()>,
    scope: ProviderRuntimeHttpInvocationIdentity,
    cancellation: CancellationToken,
    maximum_requests: u32,
    ordinal: u32,
    active: Option<WorkerProviderStreamHandle>,
    retained_authority: Option<ProviderRuntimeAuthorityInput>,
    predecessor: Option<ProviderRuntimeReceiptV2>,
    transcript: Sha256,
}

pub struct ProviderRuntimeHttpInvocation {
    service: ProviderRuntimeStreamService,
    service_identity: Arc<()>,
    identity: Arc<()>,
    scope: ProviderRuntimeHttpInvocationIdentity,
    armed: bool,
}

impl ProviderRuntimeHttpInvocation {
    pub fn identity(&self) -> &ProviderRuntimeHttpInvocationIdentity {
        &self.scope
    }
}

impl Drop for ProviderRuntimeHttpInvocation {
    fn drop(&mut self) {
        if self.armed {
            let mut state = self.service.state.lock();
            revoke_http_invocation(&mut state, &self.scope.context, &self.identity);
        }
    }
}

pub struct ProviderRuntimeHttpRequestCompletion {
    invocation_identity: Arc<()>,
    ordinal: u32,
    verified: VerifiedProviderRuntimeReceiptV2,
}

impl ProviderRuntimeHttpRequestCompletion {
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn receipt(&self) -> &VerifiedProviderRuntimeReceiptV2 {
        &self.verified
    }
}

pub struct ProviderRuntimeHttpReceiptSeal<'a> {
    pub issuer: &'a ProviderRuntimeReceiptIssuerV2,
    pub verifier: &'a ProviderRuntimeReceiptVerifierV2,
    pub issued_at: Instant,
    pub expires_at: Instant,
    pub nonce: [u8; 32],
    pub verification_time: Instant,
}

pub struct ProviderRuntimeHttpContinuation<'a> {
    pub ordinal: u32,
    pub request_head: WorkerProviderRequestHead,
    pub provider_policy: &'a ProviderPolicy,
    pub verifier: &'a ProviderRuntimeReceiptVerifierV2,
    pub now: Instant,
}

pub struct ProviderRuntimeHttpInvocationCompletion {
    identity: ProviderRuntimeHttpInvocationIdentity,
    request_count: u32,
    transcript_sha256: String,
    last_request: VerifiedProviderRuntimeReceiptV2,
}

impl ProviderRuntimeHttpInvocationCompletion {
    pub fn identity(&self) -> &ProviderRuntimeHttpInvocationIdentity {
        &self.identity
    }

    pub fn request_count(&self) -> u32 {
        self.request_count
    }

    pub fn transcript_sha256(&self) -> &str {
        &self.transcript_sha256
    }

    pub fn last_request(&self) -> &VerifiedProviderRuntimeReceiptV2 {
        &self.last_request
    }
}

#[derive(Clone)]
pub struct ProviderRuntimeStreamService {
    state: Arc<Mutex<ProviderRuntimeStreamState>>,
}

impl ProviderRuntimeStreamService {
    pub(crate) fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(ProviderRuntimeStreamState::new())),
        }
    }

    pub fn activation_grants(&self) -> ProviderRuntimeActivationGrantSource {
        ProviderRuntimeActivationGrantSource {
            state: self.state.clone(),
        }
    }

    pub fn claim_activation(
        &self,
        context: &WorkerProviderInvocationContext,
        cancellation: &CancellationToken,
    ) -> Result<ProviderRuntimeActivationGrant, PluginServiceError> {
        self.activation_grants().claim(context, cancellation)
    }

    #[allow(dead_code)]
    pub(crate) fn register_activation(
        &self,
        context: &WorkerProviderInvocationContext,
        grant: ProviderRuntimeActivationGrant,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        self.activation_grants()
            .insert(context, grant, cancellation)
    }
}

impl fmt::Debug for ProviderRuntimeStreamService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderRuntimeStreamService([SEALED])")
    }
}

#[derive(Clone)]
pub struct ProviderRuntimeActivationGrantSource {
    state: Arc<Mutex<ProviderRuntimeStreamState>>,
}

impl ProviderRuntimeActivationGrantSource {
    #[cfg(test)]
    fn new() -> Self {
        ProviderRuntimeStreamService::new().activation_grants()
    }

    pub fn claim(
        &self,
        context: &WorkerProviderInvocationContext,
        cancellation: &CancellationToken,
    ) -> Result<ProviderRuntimeActivationGrant, PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        validate_worker_provider_context(context)?;
        let mut state = self.state.lock();
        let table = &mut state.grants;
        let key = (
            context.session_id,
            context.session_generation,
            context.invocation,
        );
        let Some(current_session_generation) =
            table.session_generations.get(&context.session_id).copied()
        else {
            return Err(PluginServiceError::ProviderRuntimeForeignSession);
        };
        if current_session_generation != context.session_generation {
            return Err(PluginServiceError::ProviderRuntimeStaleSession);
        }
        let Some((generation, grant)) = table.grants.get_mut(&key) else {
            return Err(PluginServiceError::ProviderRuntimeForeignInvocation);
        };
        if *generation != context.generation {
            return Err(PluginServiceError::ProviderRuntimeStaleInvocation);
        }
        if grant.is_none() {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::RevokedHandle,
            ));
        }
        grant
            .as_ref()
            .and_then(|grant| grant.cancellation.as_ref())
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let mut grant = grant
            .take()
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let claim = Arc::new(AtomicBool::new(true));
        state.activation_claims.insert(
            (
                context.session_id,
                context.session_generation,
                context.invocation,
                context.generation,
            ),
            claim.clone(),
        );
        grant.host_context = Some(context.clone());
        grant.service_identity = Some(state.service_identity.clone());
        grant.claim = ProviderRuntimeClaimGuard::armed(claim);
        Ok(grant)
    }

    #[allow(dead_code)]
    pub(crate) fn insert(
        &self,
        context: &WorkerProviderInvocationContext,
        mut grant: ProviderRuntimeActivationGrant,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        validate_worker_provider_context(context)?;
        grant.cancellation = Some(cancellation.clone());
        let mut state = self.state.lock();
        let key = (
            context.session_id,
            context.session_generation,
            context.invocation,
        );
        if let Some(current_generation) = state
            .grants
            .session_generations
            .get(&context.session_id)
            .copied()
        {
            if context.session_generation < current_generation {
                return Err(PluginServiceError::ProviderRuntimeStaleSession);
            }
            if context.session_generation > current_generation {
                revoke_replaced_host_context(&mut state, context, true);
                state
                    .grants
                    .grants
                    .retain(|(session_id, _, _), _| *session_id != context.session_id);
                state
                    .grants
                    .session_generations
                    .insert(context.session_id, context.session_generation);
            }
        } else {
            if state.grants.grants.len() >= MAX_PROVIDER_RUNTIME_SESSIONS {
                return Err(PluginServiceError::ProviderSessionUnavailable);
            }
            state
                .grants
                .session_generations
                .insert(context.session_id, context.session_generation);
        }
        if !state.grants.grants.contains_key(&key)
            && state.grants.grants.len() >= MAX_PROVIDER_RUNTIME_SESSIONS
        {
            return Err(PluginServiceError::ProviderSessionUnavailable);
        }
        if state
            .grants
            .grants
            .get(&key)
            .is_some_and(|(generation, _)| *generation >= context.generation)
        {
            return Err(PluginServiceError::ProviderRuntimeStaleInvocation);
        }
        if state.grants.grants.contains_key(&key) {
            revoke_replaced_host_context(&mut state, context, false);
        }
        state
            .grants
            .grants
            .insert(key, (context.generation, Some(grant)));
        Ok(())
    }
}

impl fmt::Debug for ProviderRuntimeActivationGrantSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderRuntimeActivationGrantSource([SEALED])")
    }
}

impl ProviderRuntimeActivationGrant {
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn checked_from_active_deployment(
        profile_id: impl Into<String>,
        principal_id: impl Into<String>,
        prompt_id: impl Into<String>,
        prompt_sha256: impl Into<String>,
        attempt_id: impl Into<String>,
        node_id: impl Into<String>,
        request_ordinal: u32,
        worker_deployment: &WorkerRegistryDeploymentPlan,
        component_digest_sha256: impl Into<String>,
        provider_manifest_sha256: impl Into<String>,
        authorization_generation_sha256: impl Into<String>,
        binding_set_sha256: impl Into<String>,
        compiled_plan_sha256: impl Into<String>,
    ) -> Result<Self, PluginServiceError> {
        let profile_id = profile_id.into();
        let principal_id = principal_id.into();
        let prompt_id = prompt_id.into();
        let prompt_sha256 = prompt_sha256.into();
        let attempt_id = attempt_id.into();
        let node_id = node_id.into();
        let registry_generation = worker_deployment.begin().generation().get();
        let registry_digest_sha256 = worker_deployment
            .begin()
            .registry_digest_sha256()
            .as_str()
            .to_owned();
        let component_generation = registry_generation;
        let binding_generation = registry_generation;
        let component_digest_sha256 = component_digest_sha256.into();
        let provider_manifest_sha256 = provider_manifest_sha256.into();
        let authorization_generation_sha256 = authorization_generation_sha256.into();
        let binding_set_sha256 = binding_set_sha256.into();
        let compiled_plan_sha256 = compiled_plan_sha256.into();
        if [
            profile_id.as_str(),
            principal_id.as_str(),
            prompt_id.as_str(),
            attempt_id.as_str(),
            node_id.as_str(),
        ]
        .into_iter()
        .any(|value| !valid_provider_runtime_identity(value))
            || [
                prompt_sha256.as_str(),
                registry_digest_sha256.as_str(),
                component_digest_sha256.as_str(),
                provider_manifest_sha256.as_str(),
                authorization_generation_sha256.as_str(),
                binding_set_sha256.as_str(),
                compiled_plan_sha256.as_str(),
            ]
            .into_iter()
            .any(|value| !is_lower_sha256(value))
        {
            return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
        }
        Ok(Self {
            host_context: None,
            service_identity: None,
            claim: ProviderRuntimeClaimGuard::default(),
            cancellation: None,
            profile_id,
            principal_id,
            prompt_id,
            prompt_sha256,
            attempt_id,
            node_id,
            request_ordinal,
            registry_generation,
            registry_digest_sha256,
            component_generation,
            component_digest_sha256,
            provider_manifest_sha256,
            authorization_generation_sha256,
            binding_generation,
            binding_set_sha256,
            compiled_plan_sha256,
        })
    }

    pub fn preflight_installed_component(
        self,
        worker_context: &WorkerProviderInvocationContext,
        worker_deployment: &WorkerRegistryDeploymentPlan,
        worker_invocation: &NativeProviderWorkerSessionStart,
        manifest_authorization: ProviderManifestAuthorizationV2,
    ) -> Result<PreflightedProviderRuntimeActivationGrant, PluginServiceError> {
        let result = self.preflight_installed_component_inner(
            worker_context,
            worker_deployment,
            worker_invocation,
            &manifest_authorization,
        );
        match result {
            Ok(()) => Ok(PreflightedProviderRuntimeActivationGrant {
                grant: Some(self),
                manifest_authorization: Some(manifest_authorization),
            }),
            Err(error) => Err(error),
        }
    }

    fn preflight_installed_component_inner(
        &self,
        worker_context: &WorkerProviderInvocationContext,
        worker_deployment: &WorkerRegistryDeploymentPlan,
        worker_invocation: &NativeProviderWorkerSessionStart,
        manifest_authorization: &ProviderManifestAuthorizationV2,
    ) -> Result<(), PluginServiceError> {
        let cancellation = self
            .cancellation
            .as_ref()
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let claimed_context = self
            .host_context
            .as_ref()
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        if worker_context.session_id != claimed_context.session_id
            || worker_context.session_generation != claimed_context.session_generation
            || worker_context.invocation != claimed_context.invocation
            || worker_context.generation != claimed_context.generation
        {
            return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
        }
        let deployment = worker_deployment.begin();
        let component = deployment
            .components()
            .iter()
            .find(|component| component.extension_id() == worker_invocation.extension_id)
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        let authorization = manifest_authorization.authorization();
        let binding = manifest_authorization.provider_binding();
        let provider_manifest_sha256 =
            encode_lower_hex(manifest_authorization.outer_signing_payload_sha256());
        let generation = deployment.generation().get();
        if worker_invocation.registry_generation != generation
            || worker_invocation.registry_digest_sha256
                != deployment.registry_digest_sha256().as_str()
            || worker_invocation.extension_version != component.extension_version()
            || worker_invocation.plugin_identifier != component.plugin_identifier()
            || worker_invocation.plugin_version != component.plugin_version()
            || worker_invocation.manifest_digest_sha256
                != component.manifest_digest_sha256().as_str()
            || worker_invocation.component_digest_sha256
                != component.component_digest_sha256().as_str()
            || worker_invocation.authorization_generation_sha256
                != component.authorization_generation().as_str()
            || self.registry_generation != generation
            || self.registry_digest_sha256 != deployment.registry_digest_sha256().as_str()
            || self.component_generation != generation
            || self.component_digest_sha256 != component.component_digest_sha256().as_str()
            || self.provider_manifest_sha256 != provider_manifest_sha256
            || self.authorization_generation_sha256 != component.authorization_generation().as_str()
            || self.binding_generation != generation
            || self.binding_set_sha256 != binding.bindings_sha256
            || self.binding_set_sha256 != worker_invocation.binding_set_sha256
            || self.node_id != worker_invocation.node_id
            || self.compiled_plan_sha256 != worker_invocation.compiled_plan_sha256
            || authorization.plugin_id() != component.plugin_identifier()
            || authorization.digest_sha256() != component.component_digest_sha256().as_str()
            || binding.implementation_namespace != authorization.plugin_id()
            || !binding
                .bindings
                .iter()
                .any(|claim| claim.node_id == worker_invocation.node_id)
        {
            return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
        }
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)
    }
}

impl PreflightedProviderRuntimeActivationGrant {
    pub fn bind(
        mut self,
        request_head: &ProviderRequestHeadV2,
        provider_policy: &ProviderPolicy,
    ) -> Result<ProviderRuntimeAuthorityInput, PluginServiceError> {
        let grant = self
            .grant
            .take()
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        let manifest_authorization = self
            .manifest_authorization
            .take()
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        ProviderRuntimeAuthorityInput::checked_from_preflighted_activation_grant(
            grant,
            manifest_authorization,
            request_head,
            provider_policy,
        )
    }
}

fn revoke_provider_runtime_claim(claim: Option<&Arc<AtomicBool>>) {
    if let Some(claim) = claim {
        claim.store(false, Ordering::Release);
    }
}

pub struct ProviderRuntimeAuthorityInput {
    service_identity: Arc<()>,
    claim: Arc<AtomicBool>,
    http_request_ordinal: Option<u32>,
    cancellation: CancellationToken,
    host_context: WorkerProviderInvocationContext,
    profile_id: String,
    principal_id: String,
    prompt_id: String,
    prompt_sha256: String,
    attempt_id: String,
    node_id: String,
    request_ordinal: u32,
    registry_generation: u64,
    registry_digest_sha256: String,
    component_generation: u64,
    component_digest_sha256: String,
    authorization_generation_sha256: String,
    binding_generation: u64,
    binding_set_sha256: String,
    compiled_plan_sha256: String,
    provider: String,
    request: AuthorizedProviderRequest,
    request_head: ProviderRequestHeadV2,
    request_head_sha256: String,
    provider_manifest_sha256: String,
    manifest_authorization: ProviderManifestAuthorizationV2,
}

impl ProviderRuntimeAuthorityInput {
    fn checked_from_preflighted_activation_grant(
        mut grant: ProviderRuntimeActivationGrant,
        manifest_authorization: ProviderManifestAuthorizationV2,
        request_head: &ProviderRequestHeadV2,
        provider_policy: &ProviderPolicy,
    ) -> Result<Self, PluginServiceError> {
        let host_context = grant
            .host_context
            .clone()
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        let service_identity = grant
            .service_identity
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        let cancellation = grant
            .cancellation
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let (provider, provider_manifest_sha256, request, request_head_sha256) =
            authorize_runtime_request(
                &grant.profile_id,
                &manifest_authorization,
                request_head,
                provider_policy,
            )?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let claim = grant.claim.disarm()?;
        Ok(Self {
            service_identity,
            claim,
            http_request_ordinal: None,
            cancellation,
            host_context,
            profile_id: grant.profile_id,
            principal_id: grant.principal_id,
            prompt_id: grant.prompt_id,
            prompt_sha256: grant.prompt_sha256,
            attempt_id: grant.attempt_id,
            node_id: grant.node_id,
            request_ordinal: grant.request_ordinal,
            registry_generation: grant.registry_generation,
            registry_digest_sha256: grant.registry_digest_sha256,
            component_generation: grant.component_generation,
            component_digest_sha256: grant.component_digest_sha256,
            authorization_generation_sha256: grant.authorization_generation_sha256,
            binding_generation: grant.binding_generation,
            binding_set_sha256: grant.binding_set_sha256,
            compiled_plan_sha256: grant.compiled_plan_sha256,
            provider,
            request,
            request_head: request_head.clone(),
            request_head_sha256,
            provider_manifest_sha256,
            manifest_authorization,
        })
    }

    fn streaming_contract(&self) -> &sim_plugin_sdk::ProviderStreamingContractV2 {
        self.manifest_authorization.streaming_contract()
    }

    fn idempotency_identity_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"zed-sim-provider-runtime-authority-v2\0");
        for value in [
            self.profile_id.as_str(),
            self.principal_id.as_str(),
            self.prompt_id.as_str(),
            self.prompt_sha256.as_str(),
            self.attempt_id.as_str(),
            self.node_id.as_str(),
            self.registry_digest_sha256.as_str(),
            self.component_digest_sha256.as_str(),
            self.authorization_generation_sha256.as_str(),
            self.binding_set_sha256.as_str(),
            self.compiled_plan_sha256.as_str(),
            self.provider.as_str(),
            self.request.endpoint(),
            self.request_head_sha256.as_str(),
            self.provider_manifest_sha256.as_str(),
        ] {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        digest.update(self.registry_generation.to_le_bytes());
        digest.update(self.component_generation.to_le_bytes());
        digest.update(self.binding_generation.to_le_bytes());
        digest.update(self.request_ordinal.to_le_bytes());
        if let Some(ordinal) = self.http_request_ordinal {
            digest.update(b"retained-http-request\0");
            digest.update(ordinal.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }
}

fn authorize_runtime_request(
    profile_id: &str,
    manifest_authorization: &ProviderManifestAuthorizationV2,
    request_head: &ProviderRequestHeadV2,
    provider_policy: &ProviderPolicy,
) -> Result<(String, String, AuthorizedProviderRequest, String), PluginServiceError> {
    request_head
        .validate_for_contract(manifest_authorization.streaming_contract())
        .map_err(map_provider_streaming_error)?;
    let binding = manifest_authorization.provider_binding();
    let provider = binding.implementation_namespace.clone();
    let authorization = manifest_authorization.authorization();
    let provider_manifest_sha256 =
        encode_lower_hex(manifest_authorization.outer_signing_payload_sha256());
    if authorization.plugin_id() != provider {
        return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
    }
    authorization
        .capabilities()
        .require(&Capability::ProviderNetwork {
            provider: provider.clone(),
            endpoint: request_head.endpoint.clone(),
        })
        .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
    let secret_id = request_head
        .secret_id
        .as_deref()
        .map(SecretId::new)
        .transpose()
        .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
    if let Some(secret_id) = &secret_id {
        authorization
            .capabilities()
            .require(&Capability::Secret {
                secret_id: secret_id.as_str().to_owned(),
            })
            .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
    }
    let request = provider_policy
        .authorize(
            profile_id,
            authorization.plugin_id(),
            &provider,
            &request_head.endpoint,
            secret_id.as_ref(),
        )
        .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
    let request_head_sha256 = format!(
        "{:x}",
        Sha256::digest(
            request_head
                .canonical_bytes(manifest_authorization.streaming_contract())
                .map_err(map_provider_streaming_error)?
        )
    );
    Ok((
        provider,
        provider_manifest_sha256,
        request,
        request_head_sha256,
    ))
}

impl fmt::Debug for ProviderRuntimeAuthorityInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntimeAuthorityInput")
            .field("provider", &self.provider)
            .field("node_id", &self.node_id)
            .field("request_head_sha256", &self.request_head_sha256)
            .field("provider_manifest_sha256", &self.provider_manifest_sha256)
            .finish_non_exhaustive()
    }
}

struct ProviderRuntimeStreamingSession {
    authority: ProviderRuntimeAuthorityInput,
    validator: ProviderStreamValidatorV2,
    request_body_digest: Sha256,
    request_finished: bool,
    response_status: Option<u16>,
    response_headers_sha256: Option<String>,
    ordered_chunks_digest: Sha256,
    terminal_receipt_sha256: Option<String>,
    accepted_cost_microunits: u64,
    upload_validators: BTreeMap<ProviderStreamHandleV2, ProviderUploadValidatorV2>,
    upload_download_urls: BTreeMap<ProviderStreamHandleV2, String>,
    ordered_uploads: Vec<ProviderRuntimeUploadIdentity>,
    last_published_progress: Option<(Instant, u64, u64)>,
    retry_after_seconds: Option<u64>,
    completed: bool,
    actuation_proposal: Option<ProviderRuntimeActuationProposal>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProviderRuntimeUploadIdentity {
    handle: ProviderStreamHandleV2,
    port_id: String,
    media_type: String,
    byte_length: u64,
    content_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderRuntimeActuationProposal {
    request: AuthorizedProviderRequest,
    request_head: ProviderRequestHeadV2,
    request_head_sha256: String,
    request_body_sha256: String,
    ordered_uploads_sha256: String,
    accepted_cost_microunits: u64,
    idempotency_identity_sha256: String,
}

impl ProviderRuntimeActuationProposal {
    pub fn request(&self) -> &AuthorizedProviderRequest {
        &self.request
    }

    pub fn request_head(&self) -> &ProviderRequestHeadV2 {
        &self.request_head
    }

    pub fn request_head_sha256(&self) -> &str {
        &self.request_head_sha256
    }

    pub fn request_body_sha256(&self) -> &str {
        &self.request_body_sha256
    }

    pub fn ordered_uploads_sha256(&self) -> &str {
        &self.ordered_uploads_sha256
    }

    pub fn accepted_cost_microunits(&self) -> u64 {
        self.accepted_cost_microunits
    }

    pub fn idempotency_identity_sha256(&self) -> &str {
        &self.idempotency_identity_sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderRuntimeProgressProjection {
    pub handle: WorkerProviderStreamHandle,
    pub sequence: u64,
    pub completed: u64,
    pub total: u64,
    pub message: Option<String>,
}

struct ProviderRuntimeProgressUpdate {
    sequence: u64,
    completed: u64,
    total: u64,
    message: Option<String>,
}

enum ProviderRuntimeWaitUpdate {
    None,
    Head {
        status: u16,
        headers_sha256: String,
        retry_after_seconds: Option<u64>,
    },
    Chunk(Sha256),
    Terminal {
        receipt_sha256: String,
        completed: bool,
    },
}

struct ProviderRuntimeStreamOwner {
    legacy_sessions: BTreeMap<String, PluginCapabilityInvocation>,
    streaming_sessions: BTreeMap<ProviderStreamHandleV2, ProviderRuntimeStreamingSession>,
    streaming_upload_parents: BTreeMap<ProviderStreamHandleV2, ProviderStreamHandleV2>,
    consumed_streaming_cost_nonces: BTreeMap<ProviderCostNonce, Instant>,
}

impl ProviderRuntimeStreamOwner {
    pub(crate) fn new() -> Self {
        Self {
            legacy_sessions: BTreeMap::new(),
            streaming_sessions: BTreeMap::new(),
            streaming_upload_parents: BTreeMap::new(),
            consumed_streaming_cost_nonces: BTreeMap::new(),
        }
    }

    fn begin_streaming(
        &mut self,
        authority: ProviderRuntimeAuthorityInput,
        context: ProviderInvocationContextV2,
        handle: ProviderStreamHandleV2,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        if self.legacy_sessions.len() + self.streaming_sessions.len()
            >= MAX_PROVIDER_RUNTIME_SESSIONS
            || self.streaming_sessions.contains_key(&handle)
            || self.streaming_upload_parents.contains_key(&handle)
        {
            return Err(PluginServiceError::ProviderSessionUnavailable);
        }
        let validator = ProviderStreamValidatorV2::checked(
            authority.streaming_contract().clone(),
            context,
            handle,
            &authority.request_head,
        )
        .map_err(map_provider_streaming_error)?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        self.streaming_sessions.insert(
            handle,
            ProviderRuntimeStreamingSession {
                authority,
                validator,
                request_body_digest: Sha256::new(),
                request_finished: false,
                response_status: None,
                response_headers_sha256: None,
                ordered_chunks_digest: ordered_chunks_digest(),
                terminal_receipt_sha256: None,
                accepted_cost_microunits: 0,
                upload_validators: BTreeMap::new(),
                upload_download_urls: BTreeMap::new(),
                ordered_uploads: Vec::new(),
                last_published_progress: None,
                retry_after_seconds: None,
                completed: false,
                actuation_proposal: None,
            },
        );
        Ok(())
    }

    fn write_streaming_request_chunk(
        &mut self,
        chunk: &ProviderRequestChunkV2,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let session = self.streaming_session_mut(chunk.handle)?;
        if session.actuation_proposal.is_some() {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidOrder,
            ));
        }
        session
            .validator
            .write_request_chunk(chunk)
            .map_err(map_provider_streaming_error)?;
        session.request_body_digest.update(&chunk.bytes);
        session.request_finished = chunk.end;
        Ok(())
    }

    fn accept_streaming_wait(
        &mut self,
        request: &ProviderWaitRequestV2,
        outcome: ProviderWaitOutcomeV2,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let retry_after_seconds = retry_after_seconds(&outcome)?;
        let revoke_after_accept = matches!(&outcome, ProviderWaitOutcomeV2::Cancelled)
            || matches!(
                &outcome,
                ProviderWaitOutcomeV2::Frame(sim_plugin_sdk::ProviderResponseFrameV2 {
                    event: ProviderResponseFrameEventV2::Terminal(
                        sim_plugin_sdk::ProviderStreamTerminalV2::Failed { .. }
                            | sim_plugin_sdk::ProviderStreamTerminalV2::Cancelled
                    ),
                    ..
                })
            );
        {
            let session = self.streaming_session_mut(request.handle)?;
            if matches!(
                &outcome,
                ProviderWaitOutcomeV2::Frame(sim_plugin_sdk::ProviderResponseFrameV2 {
                    event: ProviderResponseFrameEventV2::Head(_),
                    ..
                })
            ) && session.actuation_proposal.is_none()
            {
                return Err(PluginServiceError::ProviderStreamingContract(
                    ProviderStreamingContractError::InvalidOrder,
                ));
            }
            let update = match &outcome {
                ProviderWaitOutcomeV2::Frame(frame) => match &frame.event {
                    ProviderResponseFrameEventV2::Head(head) => ProviderRuntimeWaitUpdate::Head {
                        status: head.status,
                        headers_sha256: ordered_headers_sha256(&head.headers),
                        retry_after_seconds,
                    },
                    ProviderResponseFrameEventV2::Chunk(chunk) => {
                        ProviderRuntimeWaitUpdate::Chunk(updated_ordered_chunks_digest(
                            &session.ordered_chunks_digest,
                            frame.sequence,
                            chunk,
                        ))
                    }
                    ProviderResponseFrameEventV2::Terminal(terminal) => {
                        ProviderRuntimeWaitUpdate::Terminal {
                            receipt_sha256: terminal_event_sha256(terminal)?,
                            completed: matches!(
                                terminal,
                                sim_plugin_sdk::ProviderStreamTerminalV2::Completed { .. }
                            ),
                        }
                    }
                },
                ProviderWaitOutcomeV2::TimedOut | ProviderWaitOutcomeV2::Cancelled => {
                    ProviderRuntimeWaitUpdate::None
                }
            };
            session
                .validator
                .accept_wait(request, outcome)
                .map_err(map_provider_streaming_error)?;
            match update {
                ProviderRuntimeWaitUpdate::None => {}
                ProviderRuntimeWaitUpdate::Head {
                    status,
                    headers_sha256,
                    retry_after_seconds,
                } => {
                    session.response_status = Some(status);
                    session.response_headers_sha256 = Some(headers_sha256);
                    session.retry_after_seconds = retry_after_seconds;
                }
                ProviderRuntimeWaitUpdate::Terminal {
                    receipt_sha256,
                    completed,
                } => {
                    session.terminal_receipt_sha256 = Some(receipt_sha256);
                    session.completed = completed;
                }
                ProviderRuntimeWaitUpdate::Chunk(updated) => {
                    session.ordered_chunks_digest = updated;
                }
            }
        }
        if revoke_after_accept {
            self.remove_streaming_session(request.handle);
        }
        Ok(())
    }

    fn start_streaming_upload(
        &mut self,
        request: &ProviderUploadRequestV2,
        upload_handle: ProviderStreamHandleV2,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        self.streaming_session_mut(request.handle)?;
        if self.streaming_sessions.contains_key(&upload_handle)
            || self.streaming_upload_parents.contains_key(&upload_handle)
        {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidUpload,
            ));
        }
        let session = self.streaming_session_mut(request.handle)?;
        if session.actuation_proposal.is_some() {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidOrder,
            ));
        }
        let upload = session
            .validator
            .start_upload(request, upload_handle)
            .map_err(map_provider_streaming_error)?;
        session.upload_validators.insert(upload_handle, upload);
        session.ordered_uploads.push(ProviderRuntimeUploadIdentity {
            handle: upload_handle,
            port_id: request.port_id.clone(),
            media_type: request.media_type.clone(),
            byte_length: request.byte_length,
            content_sha256: request.content_sha256.clone(),
        });
        self.streaming_upload_parents
            .insert(upload_handle, request.handle);
        Ok(())
    }

    fn write_streaming_upload_chunk(
        &mut self,
        chunk: &ProviderRequestChunkV2,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let parent = *self
            .streaming_upload_parents
            .get(&chunk.handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        self.streaming_sessions
            .get_mut(&parent)
            .and_then(|session| session.upload_validators.get_mut(&chunk.handle))
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?
            .write_chunk(chunk)
            .map_err(map_provider_streaming_error)
    }

    fn accept_streaming_upload_completion(
        &mut self,
        upload_handle: ProviderStreamHandleV2,
        download_url: String,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let parent = *self
            .streaming_upload_parents
            .get(&upload_handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let session = self
            .streaming_sessions
            .get_mut(&parent)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        if !session
            .upload_validators
            .get(&upload_handle)
            .is_some_and(ProviderUploadValidatorV2::is_terminal)
            || session.upload_download_urls.contains_key(&upload_handle)
        {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidOrder,
            ));
        }
        session
            .upload_download_urls
            .insert(upload_handle, download_url);
        Ok(())
    }

    fn accept_streaming_cost(
        &mut self,
        request: &ProviderCostRequestV2,
        cost_request_sha256: String,
        authorization: ProviderCostAuthorization,
        verifier: &ProviderCostAcceptanceVerifier,
        now: Instant,
        cancellation: &CancellationToken,
    ) -> Result<ProviderCostResponseV2, PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let session = self
            .streaming_sessions
            .get(&request.handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        if session.actuation_proposal.is_some()
            || !session.request_finished
            || session
                .upload_validators
                .values()
                .any(|upload| !upload.is_terminal())
            || session.upload_validators.len() != session.upload_download_urls.len()
            || request.currency != authorization.price_bound().currency_code()
            || request.maximum_microunits != authorization.price_bound().maximum_microunits()
        {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        let expected_scope = provider_streaming_cost_scope_with_request_sha256(
            session,
            cost_request_sha256,
            authorization.price_bound().clone(),
        )?;
        let verified = verifier
            .verify(authorization.acceptance(), &expected_scope, now)
            .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
        if verified.nonce() != authorization.nonce() {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        self.consumed_streaming_cost_nonces
            .retain(|_, expires_at| *expires_at > now);
        if self
            .consumed_streaming_cost_nonces
            .contains_key(&verified.nonce())
        {
            return Err(PluginServiceError::ProviderCostAcceptanceReused);
        }
        if self.consumed_streaming_cost_nonces.len() >= MAX_CONSUMED_PROVIDER_COST_NONCES {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        let accepted_cost_microunits = session
            .accepted_cost_microunits
            .checked_add(request.maximum_microunits)
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        let response = ProviderCostResponseV2 {
            accepted: true,
            approved_microunits: request.maximum_microunits,
            receipt: authorization
                .acceptance()
                .receipt_bytes()
                .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?,
        };
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        if let Some(previous_expiration) = self
            .consumed_streaming_cost_nonces
            .insert(verified.nonce(), verified.expires_at())
        {
            self.consumed_streaming_cost_nonces
                .insert(verified.nonce(), previous_expiration);
            return Err(PluginServiceError::ProviderCostAcceptanceReused);
        }
        let validation = self
            .streaming_session_mut(request.handle)?
            .validator
            .accept_cost_request(request, &response)
            .map_err(map_provider_streaming_error);
        if let Err(error) = validation {
            self.consumed_streaming_cost_nonces
                .remove(&verified.nonce());
            return Err(error);
        }
        self.streaming_session_mut(request.handle)?
            .accepted_cost_microunits = accepted_cost_microunits;
        Ok(response)
    }

    fn deny_streaming_cost(
        &mut self,
        request: &ProviderCostRequestV2,
        cancellation: &CancellationToken,
    ) -> Result<ProviderCostResponseV2, PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let response = ProviderCostResponseV2 {
            accepted: false,
            approved_microunits: 0,
            receipt: Vec::new(),
        };
        let session = self.streaming_session_mut(request.handle)?;
        if session.actuation_proposal.is_some() {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidOrder,
            ));
        }
        session
            .validator
            .accept_cost_request(request, &response)
            .map_err(map_provider_streaming_error)?;
        Ok(response)
    }

    fn report_streaming_progress(
        &mut self,
        progress: &ProviderProgressV2,
        now: Instant,
        cancellation: &CancellationToken,
    ) -> Result<Option<ProviderRuntimeProgressUpdate>, PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let message = progress
            .message
            .as_ref()
            .map(|message| {
                let mut copy = String::new();
                copy.try_reserve_exact(message.len())
                    .map_err(|_| PluginServiceError::ResponseAllocationFailed)?;
                copy.push_str(message);
                Ok(copy)
            })
            .transpose()?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let session = self.streaming_session_mut(progress.handle)?;
        session
            .validator
            .accept_progress(progress)
            .map_err(map_provider_streaming_error)?;
        let publish = session
            .last_published_progress
            .is_none_or(|(last, completed, total)| {
                progress.completed == progress.total
                    || progress.total != total
                    || progress.completed > completed
                        && now.saturating_duration_since(last) >= PROVIDER_PROGRESS_MINIMUM_INTERVAL
            });
        if !publish {
            return Ok(None);
        }
        session.last_published_progress = Some((now, progress.completed, progress.total));
        Ok(Some(ProviderRuntimeProgressUpdate {
            sequence: progress.sequence,
            completed: progress.completed,
            total: progress.total,
            message,
        }))
    }

    fn prepare_streaming_actuation(
        &mut self,
        handle: ProviderStreamHandleV2,
        canonical_ordered_uploads_sha256: Option<&str>,
        cancellation: &CancellationToken,
    ) -> Result<ProviderRuntimeActuationProposal, PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let session = self.streaming_session_mut(handle)?;
        if let Some(proposal) = &session.actuation_proposal {
            return Ok(proposal.clone());
        }
        if !session.request_finished
            || session
                .upload_validators
                .values()
                .any(|upload| !upload.is_terminal())
            || session.upload_validators.len() != session.upload_download_urls.len()
            || session.response_status.is_some()
            || session.validator.is_terminal()
        {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidOrder,
            ));
        }
        let request_body_sha256 = request_body_sha256(session);
        let internal_ordered_uploads_sha256 = ordered_uploads_sha256(session)?;
        let ordered_uploads_sha256 = canonical_ordered_uploads_sha256
            .unwrap_or(&internal_ordered_uploads_sha256)
            .to_owned();
        let idempotency_identity_sha256 = provider_runtime_mutation_identity_sha256(
            &session.authority,
            &request_body_sha256,
            &ordered_uploads_sha256,
            session.accepted_cost_microunits,
        );
        let request = session
            .authority
            .request
            .clone()
            .with_idempotency_key_sha256(idempotency_identity_sha256.clone())
            .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let proposal = ProviderRuntimeActuationProposal {
            request,
            request_head: session.authority.request_head.clone(),
            request_head_sha256: session.authority.request_head_sha256.clone(),
            request_body_sha256,
            ordered_uploads_sha256,
            accepted_cost_microunits: session.accepted_cost_microunits,
            idempotency_identity_sha256,
        };
        session.actuation_proposal = Some(proposal.clone());
        Ok(proposal)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_streaming(
        &mut self,
        handle: ProviderStreamHandleV2,
        issuer: &ProviderRuntimeReceiptIssuerV2,
        issued_at: Instant,
        expires_at: Instant,
        nonce: [u8; 32],
        cancellation: &CancellationToken,
    ) -> Result<ProviderRuntimeReceiptV2, PluginServiceError> {
        self.finish_streaming_retained(handle, issuer, issued_at, expires_at, nonce, cancellation)
            .map(|(receipt, _authority)| receipt)
    }

    fn finish_streaming_retained(
        &mut self,
        handle: ProviderStreamHandleV2,
        issuer: &ProviderRuntimeReceiptIssuerV2,
        issued_at: Instant,
        expires_at: Instant,
        nonce: [u8; 32],
        cancellation: &CancellationToken,
    ) -> Result<(ProviderRuntimeReceiptV2, ProviderRuntimeAuthorityInput), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let session = self
            .streaming_sessions
            .get(&handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        if !session.validator.is_terminal() || !session.completed {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidTerminal,
            ));
        }
        let proposal = session
            .actuation_proposal
            .as_ref()
            .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        let authority = &session.authority;
        let ordered_chunks_sha256 =
            format!("{:x}", session.ordered_chunks_digest.clone().finalize());
        let identity = ProviderRuntimeReceiptIdentityV2 {
            provider: authority.provider.clone(),
            method: authority.request_head.method,
            endpoint: proposal.request.endpoint().to_owned(),
            ordered_headers_sha256: ordered_headers_sha256(&authority.request_head.headers),
            secret_id: proposal
                .request
                .secret_id()
                .map(|secret| secret.as_str().to_owned()),
            request_head_sha256: proposal.request_head_sha256.clone(),
            request_body_sha256: proposal.request_body_sha256.clone(),
            provider_manifest_sha256: authority.provider_manifest_sha256.clone(),
            component_generation: authority.component_generation,
            component_digest_sha256: authority.component_digest_sha256.clone(),
            binding_generation: authority.binding_generation,
            binding_set_sha256: authority.binding_set_sha256.clone(),
            accepted_cost_microunits: proposal.accepted_cost_microunits,
            request_ordinal: authority.request_ordinal,
            response_status: session
                .response_status
                .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?,
            response_headers_sha256: session
                .response_headers_sha256
                .clone()
                .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?,
            ordered_uploads_sha256: proposal.ordered_uploads_sha256.clone(),
            ordered_chunks_sha256,
            terminal_receipt_sha256: session
                .terminal_receipt_sha256
                .clone()
                .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?,
            idempotency_identity_sha256: proposal.idempotency_identity_sha256.clone(),
        };
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let receipt = issuer
            .issue(identity, issued_at, expires_at, nonce)
            .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        let session = self
            .remove_streaming_session(handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        Ok((receipt, session.authority))
    }

    fn streaming_retry_after_seconds(
        &self,
        handle: ProviderStreamHandleV2,
    ) -> Result<Option<u64>, PluginServiceError> {
        self.streaming_sessions
            .get(&handle)
            .map(|session| session.retry_after_seconds)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)
    }

    fn retry_streaming(
        &mut self,
        handle: ProviderStreamHandleV2,
        retry_handle: ProviderStreamHandleV2,
        cancellation: &CancellationToken,
    ) -> Result<(), PluginServiceError> {
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        if self.streaming_sessions.contains_key(&retry_handle) {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidHandle,
            ));
        }
        let mut session = self
            .streaming_sessions
            .remove(&handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let result = session
            .validator
            .retry(retry_handle)
            .map_err(map_provider_streaming_error);
        if let Err(error) = result {
            self.streaming_sessions.insert(handle, session);
            return Err(error);
        }
        session.request_body_digest = Sha256::new();
        session.request_finished = false;
        session.response_status = None;
        session.response_headers_sha256 = None;
        session.ordered_chunks_digest = ordered_chunks_digest();
        session.terminal_receipt_sha256 = None;
        session.retry_after_seconds = None;
        session.completed = false;
        session.actuation_proposal = None;
        self.streaming_sessions.insert(retry_handle, session);
        Ok(())
    }

    fn streaming_session_mut(
        &mut self,
        handle: ProviderStreamHandleV2,
    ) -> Result<&mut ProviderRuntimeStreamingSession, PluginServiceError> {
        self.streaming_sessions
            .get_mut(&handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)
    }

    fn remove_streaming_session(
        &mut self,
        handle: ProviderStreamHandleV2,
    ) -> Option<ProviderRuntimeStreamingSession> {
        let mut session = self.streaming_sessions.remove(&handle)?;
        session.validator.revoke();
        for upload in &session.ordered_uploads {
            self.streaming_upload_parents.remove(&upload.handle);
        }
        Some(session)
    }

    fn revoke_streaming(&mut self, handle: ProviderStreamHandleV2) {
        self.remove_streaming_session(handle);
    }

    fn begin_legacy(
        &mut self,
        authority: &dyn NativeProviderInvocationAuthority,
        scope: NativeProviderInvocationScope,
    ) -> Result<(), PluginServiceError> {
        if !valid_provider_runtime_identity(&scope.start.session_id)
            || self.legacy_sessions.len() + self.streaming_sessions.len()
                >= MAX_PROVIDER_RUNTIME_SESSIONS
            || self.legacy_sessions.contains_key(&scope.start.session_id)
        {
            return Err(PluginServiceError::ProviderSessionUnavailable);
        }
        let session_id = scope.start.session_id.clone();
        let invocation = authority.begin(scope)?;
        self.legacy_sessions.insert(session_id, invocation);
        Ok(())
    }

    fn call_legacy(
        &mut self,
        session_id: &str,
        request: PluginServiceWireRequest,
    ) -> Result<PluginServiceWireResponse, PluginServiceError> {
        if !valid_provider_runtime_identity(session_id) {
            return Err(PluginServiceError::ProviderSessionUnavailable);
        }
        let invocation = self
            .legacy_sessions
            .get_mut(session_id)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        Ok(invocation.handle_wire_request(request))
    }

    fn resolve_legacy(
        &mut self,
        session_id: &str,
        receipt_set: &ProviderResultReceiptSet,
    ) -> Result<Vec<Vec<u8>>, PluginServiceError> {
        if !valid_provider_runtime_identity(session_id) {
            return Err(PluginServiceError::ProviderSessionUnavailable);
        }
        self.legacy_sessions
            .get_mut(session_id)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?
            .resolve_provider_result_receipt_set(receipt_set)
            .map(|results| {
                results
                    .into_iter()
                    .map(ResolvedProviderResult::into_response)
                    .collect()
            })
    }

    fn finish_legacy(&mut self, session_id: &str) -> Result<(), PluginServiceError> {
        if !valid_provider_runtime_identity(session_id) {
            return Err(PluginServiceError::ProviderSessionUnavailable);
        }
        self.legacy_sessions
            .remove(session_id)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?
            .finish()
    }

    fn abort_legacy(&mut self, session_id: &str) {
        if !valid_provider_runtime_identity(session_id) {
            return;
        }
        if let Some(invocation) = self.legacy_sessions.remove(session_id) {
            invocation.abort();
        }
    }

    fn revoke_all(&mut self) {
        for (_, invocation) in std::mem::take(&mut self.legacy_sessions) {
            invocation.abort();
        }
        for (_, mut session) in std::mem::take(&mut self.streaming_sessions) {
            session.validator.revoke();
        }
        self.streaming_upload_parents.clear();
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.legacy_sessions.is_empty()
            && self.streaming_sessions.is_empty()
            && self.streaming_upload_parents.is_empty()
    }
}

impl ProviderRuntimeStreamService {
    pub fn begin_http_invocation(
        &self,
        mut authority: ProviderRuntimeAuthorityInput,
        request_head: WorkerProviderRequestHead,
        maximum_requests: u32,
    ) -> Result<(ProviderRuntimeHttpInvocation, WorkerProviderStreamHandle), PluginServiceError>
    {
        if !Arc::ptr_eq(
            &authority.service_identity,
            &self.state.lock().service_identity,
        ) {
            return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
        }
        let scope = http_invocation_scope(&authority);
        let cancellation = authority.cancellation.clone();
        if maximum_requests == 0 || maximum_requests > MAX_PROVIDER_RUNTIME_HTTP_REQUESTS {
            self.revoke_invocation(&scope.context)?;
            return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
        }
        authority.http_request_ordinal = Some(1);
        let handle = self.start_request(authority, request_head)?;
        let mut state = self.state.lock();
        let key = worker_provider_context_key(&scope.context);
        if cancellation.check().is_err() {
            revoke_worker_stream(&mut state, handle);
            return Err(PluginServiceError::Cancelled);
        }
        if !state.main_handles.contains(&handle)
            || state.http_invocations.len() >= MAX_PROVIDER_RUNTIME_SESSIONS
        {
            revoke_worker_stream(&mut state, handle);
            return Err(PluginServiceError::ProviderSessionUnavailable);
        }
        let identity = Arc::new(());
        let mut transcript = Sha256::new();
        transcript.update(b"zed-sim-provider-http-transcript-v2\0");
        state.http_invocations.insert(
            key,
            ProviderRuntimeHttpInvocationState {
                identity: Arc::clone(&identity),
                scope: scope.clone(),
                cancellation,
                maximum_requests,
                ordinal: 1,
                active: Some(handle),
                retained_authority: None,
                predecessor: None,
                transcript,
            },
        );
        Ok((
            ProviderRuntimeHttpInvocation {
                service: self.clone(),
                service_identity: Arc::clone(&state.service_identity),
                identity,
                scope,
                armed: true,
            },
            handle,
        ))
    }

    pub fn check_http_invocation_cancelled(
        &self,
        invocation: &ProviderRuntimeHttpInvocation,
    ) -> Result<(), PluginServiceError> {
        let mut state = self.state.lock();
        let result = checked_http_invocation(&state, invocation).map(|_| ());
        if result.is_err() {
            revoke_http_invocation(&mut state, &invocation.scope.context, &invocation.identity);
        }
        result
    }

    pub fn complete_http_request(
        &self,
        invocation: &ProviderRuntimeHttpInvocation,
        handle: WorkerProviderStreamHandle,
        seal: ProviderRuntimeHttpReceiptSeal<'_>,
    ) -> Result<ProviderRuntimeHttpRequestCompletion, PluginServiceError> {
        let mut state = self.state.lock();
        let result = (|| {
            let entry = checked_http_invocation(&state, invocation)?;
            if entry.active != Some(handle) {
                return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
            }
            let ordinal = entry.ordinal;
            let cancellation = entry.cancellation.clone();
            let (sdk_handle, main_handle) = resolve_worker_handle(&state, handle, Some(true))?;
            let (receipt, authority) = state.owner.finish_streaming_retained(
                sdk_handle,
                seal.issuer,
                seal.issued_at,
                seal.expires_at,
                seal.nonce,
                &cancellation,
            )?;
            let verified = seal
                .verifier
                .verify(&receipt, receipt.identity(), seal.verification_time)
                .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
            if http_invocation_scope(&authority) != invocation.scope
                || authority.http_request_ordinal != Some(ordinal)
            {
                return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
            }
            cancellation
                .check()
                .map_err(|_| PluginServiceError::Cancelled)?;
            remove_worker_stream(&mut state, main_handle);
            let entry = state
                .http_invocations
                .get_mut(&worker_provider_context_key(&invocation.scope.context))
                .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
            entry.active = None;
            entry.retained_authority = Some(authority);
            entry.predecessor = Some(receipt);
            entry.transcript.update(ordinal.to_le_bytes());
            entry.transcript.update(Sha256::digest(verified.bytes()));
            Ok(ProviderRuntimeHttpRequestCompletion {
                invocation_identity: Arc::clone(&invocation.identity),
                ordinal,
                verified,
            })
        })();
        if result.is_err() {
            revoke_http_invocation(&mut state, &invocation.scope.context, &invocation.identity);
        }
        result
    }

    pub fn continue_http_request(
        &self,
        invocation: &ProviderRuntimeHttpInvocation,
        predecessor: &ProviderRuntimeHttpRequestCompletion,
        continuation: ProviderRuntimeHttpContinuation<'_>,
    ) -> Result<WorkerProviderStreamHandle, PluginServiceError> {
        let mut state = self.state.lock();
        let result = (|| {
            let entry = checked_http_invocation(&state, invocation)?;
            verify_http_predecessor(entry, predecessor, continuation.verifier, continuation.now)?;
            if entry.ordinal.checked_add(1) != Some(continuation.ordinal)
                || continuation.ordinal > entry.maximum_requests
            {
                return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
            }
            let key = worker_provider_context_key(&invocation.scope.context);
            let internal_invocation = state
                .invocation_bindings
                .get(&key)
                .copied()
                .ok_or(PluginServiceError::ProviderRuntimeStaleInvocation)?;
            let slot = state
                .next_slots
                .get(&key)
                .copied()
                .ok_or(PluginServiceError::ProviderRuntimeStaleInvocation)?;
            let next_slot = slot
                .checked_add(1)
                .filter(|slot| *slot != 0)
                .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
            let entry = state
                .http_invocations
                .get_mut(&key)
                .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
            let mut authority = entry
                .retained_authority
                .take()
                .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
            let request_head = ProviderRequestHeadV2 {
                endpoint: continuation.request_head.endpoint,
                secret_id: continuation.request_head.secret_id,
                method: provider_http_method(continuation.request_head.method),
                headers: continuation
                    .request_head
                    .headers
                    .into_iter()
                    .map(provider_header)
                    .collect(),
                declared_body_bytes: continuation.request_head.declared_body_bytes,
            };
            let (provider, manifest_sha256, request, request_head_sha256) =
                authorize_runtime_request(
                    &authority.profile_id,
                    &authority.manifest_authorization,
                    &request_head,
                    continuation.provider_policy,
                )?;
            if provider != authority.provider
                || manifest_sha256 != authority.provider_manifest_sha256
                || http_invocation_scope(&authority) != invocation.scope
            {
                return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
            }
            authority.request = request;
            authority.request_head = request_head;
            authority.request_head_sha256 = request_head_sha256;
            authority.http_request_ordinal = Some(continuation.ordinal);
            let cancellation = authority.cancellation.clone();
            cancellation
                .check()
                .map_err(|_| PluginServiceError::Cancelled)?;
            let handle = worker_provider_handle(&invocation.scope.context, slot);
            let sdk_handle = ProviderStreamHandleV2 {
                invocation: internal_invocation,
                slot,
                generation: invocation.scope.context.generation,
            };
            state.owner.begin_streaming(
                authority,
                ProviderInvocationContextV2 {
                    invocation: internal_invocation,
                    generation: invocation.scope.context.generation,
                },
                sdk_handle,
                &cancellation,
            )?;
            state.next_slots.insert(key, next_slot);
            state.handles.insert(handle, sdk_handle);
            state.main_handles.insert(handle);
            state.cancellations.insert(handle, cancellation);
            let entry = state
                .http_invocations
                .get_mut(&key)
                .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
            entry.ordinal = continuation.ordinal;
            entry.active = Some(handle);
            entry.predecessor = None;
            Ok(handle)
        })();
        if result.is_err() {
            revoke_http_invocation(&mut state, &invocation.scope.context, &invocation.identity);
        }
        result
    }

    // The caller finalizes transport authority only after checking the component result and
    // materialization acknowledgement. An HTTP terminal alone leaves this guard armed.
    pub fn finish_http_invocation(
        &self,
        mut invocation: ProviderRuntimeHttpInvocation,
        predecessor: ProviderRuntimeHttpRequestCompletion,
        verifier: &ProviderRuntimeReceiptVerifierV2,
        now: Instant,
    ) -> Result<ProviderRuntimeHttpInvocationCompletion, PluginServiceError> {
        let mut state = self.state.lock();
        let result = (|| {
            let entry = checked_http_invocation(&state, &invocation)?;
            let verified = verify_http_predecessor(entry, &predecessor, verifier, now)?;
            let completion = ProviderRuntimeHttpInvocationCompletion {
                identity: entry.scope.clone(),
                request_count: entry.ordinal,
                transcript_sha256: format!("{:x}", entry.transcript.clone().finalize()),
                last_request: verified,
            };
            state
                .http_invocations
                .remove(&worker_provider_context_key(&invocation.scope.context));
            Ok(completion)
        })();
        if result.is_err() {
            revoke_http_invocation(&mut state, &invocation.scope.context, &invocation.identity);
        }
        if result.is_ok() {
            invocation.armed = false;
        }
        result
    }
}

fn http_invocation_scope(
    authority: &ProviderRuntimeAuthorityInput,
) -> ProviderRuntimeHttpInvocationIdentity {
    ProviderRuntimeHttpInvocationIdentity {
        context: authority.host_context.clone(),
        profile_id: authority.profile_id.clone(),
        principal_id: authority.principal_id.clone(),
        prompt_id: authority.prompt_id.clone(),
        prompt_sha256: authority.prompt_sha256.clone(),
        attempt_id: authority.attempt_id.clone(),
        node_id: authority.node_id.clone(),
        compiled_plan_sha256: authority.compiled_plan_sha256.clone(),
        activation_request_ordinal: authority.request_ordinal,
        registry_generation: authority.registry_generation,
        registry_digest_sha256: authority.registry_digest_sha256.clone(),
        component_generation: authority.component_generation,
        component_digest_sha256: authority.component_digest_sha256.clone(),
        authorization_generation_sha256: authority.authorization_generation_sha256.clone(),
        binding_generation: authority.binding_generation,
        binding_set_sha256: authority.binding_set_sha256.clone(),
        provider: authority.provider.clone(),
        provider_manifest_sha256: authority.provider_manifest_sha256.clone(),
    }
}

fn checked_http_invocation<'a>(
    state: &'a ProviderRuntimeStreamState,
    invocation: &ProviderRuntimeHttpInvocation,
) -> Result<&'a ProviderRuntimeHttpInvocationState, PluginServiceError> {
    if !Arc::ptr_eq(&state.service_identity, &invocation.service_identity) {
        return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
    }
    let entry = state
        .http_invocations
        .get(&worker_provider_context_key(&invocation.scope.context))
        .ok_or(PluginServiceError::ProviderRuntimeStaleInvocation)?;
    if !Arc::ptr_eq(&entry.identity, &invocation.identity) || entry.scope != invocation.scope {
        return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
    }
    entry
        .cancellation
        .check()
        .map_err(|_| PluginServiceError::Cancelled)?;
    Ok(entry)
}

fn verify_http_predecessor(
    entry: &ProviderRuntimeHttpInvocationState,
    predecessor: &ProviderRuntimeHttpRequestCompletion,
    verifier: &ProviderRuntimeReceiptVerifierV2,
    now: Instant,
) -> Result<VerifiedProviderRuntimeReceiptV2, PluginServiceError> {
    if entry.active.is_some()
        || entry.ordinal != predecessor.ordinal
        || !Arc::ptr_eq(&entry.identity, &predecessor.invocation_identity)
    {
        return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
    }
    let receipt = entry
        .predecessor
        .as_ref()
        .ok_or(PluginServiceError::ProviderRuntimeAuthorityDenied)?;
    let verified = verifier
        .verify(receipt, predecessor.verified.identity(), now)
        .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
    if verified.bytes() != predecessor.verified.bytes() {
        return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
    }
    Ok(verified)
}

fn revoke_http_invocation(
    state: &mut ProviderRuntimeStreamState,
    context: &WorkerProviderInvocationContext,
    identity: &Arc<()>,
) {
    let key = worker_provider_context_key(context);
    if state
        .http_invocations
        .get(&key)
        .is_some_and(|entry| Arc::ptr_eq(&entry.identity, identity))
    {
        if let Some(entry) = state.http_invocations.remove(&key)
            && let Some(handle) = entry.active
        {
            revoke_worker_stream(state, handle);
        }
    }
}

impl ProviderRuntimeStreamService {
    pub fn start_request(
        &self,
        authority: ProviderRuntimeAuthorityInput,
        request_head: WorkerProviderRequestHead,
    ) -> Result<WorkerProviderStreamHandle, PluginServiceError> {
        let context = authority.host_context.clone();
        validate_worker_provider_context(&context)?;
        let key = worker_provider_context_key(&context);
        let mut state = self.state.lock();
        if !Arc::ptr_eq(&authority.service_identity, &state.service_identity) {
            return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
        }
        let Some(active_claim) = state.activation_claims.get(&key) else {
            return Err(classify_inactive_authority(&state, &context));
        };
        if !Arc::ptr_eq(active_claim, &authority.claim)
            || authority
                .claim
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::RevokedHandle,
            ));
        }
        state.activation_claims.remove(&key);
        let cancellation = authority.cancellation.clone();
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        if !worker_request_head_matches(&request_head, &authority.request_head) {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidRequestAuthority,
            ));
        }
        if state.invocation_bindings.contains_key(&key) {
            return Err(PluginServiceError::ProviderRuntimeStaleInvocation);
        }
        let internal_invocation = state
            .next_internal_invocation
            .checked_add(1)
            .filter(|invocation| *invocation != 0)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let slot = state.next_slots.get(&key).copied().unwrap_or(1);
        let next_slot = slot
            .checked_add(1)
            .filter(|slot| *slot != 0)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let worker_handle = worker_provider_handle(&context, slot);
        let sdk_context = ProviderInvocationContextV2 {
            invocation: internal_invocation,
            generation: context.generation,
        };
        let sdk_handle = ProviderStreamHandleV2 {
            invocation: internal_invocation,
            slot,
            generation: context.generation,
        };
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        state
            .owner
            .begin_streaming(authority, sdk_context, sdk_handle, &cancellation)?;
        state.next_internal_invocation = internal_invocation;
        state.invocation_bindings.insert(key, internal_invocation);
        state.next_slots.insert(key, next_slot);
        state.handles.insert(worker_handle, sdk_handle);
        state.main_handles.insert(worker_handle);
        state.cancellations.insert(worker_handle, cancellation);
        Ok(worker_handle)
    }

    pub fn write_request_chunk(
        &self,
        chunk: WorkerProviderRequestChunk,
    ) -> Result<(), PluginServiceError> {
        let mut state = self.state.lock();
        let worker_handle = chunk.handle;
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, worker_handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let chunk = provider_request_chunk(chunk, sdk_handle);
        let result = state
            .owner
            .write_streaming_request_chunk(&chunk, &cancellation);
        finish_worker_operation(&mut state, main_handle, result)
    }

    pub fn retry_request(
        &self,
        handle: WorkerProviderStreamHandle,
    ) -> Result<WorkerProviderStreamHandle, PluginServiceError> {
        let mut state = self.state.lock();
        if state
            .http_invocations
            .contains_key(&worker_provider_context_key(
                &worker_provider_context_from_handle(handle),
            ))
        {
            return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
        }
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let context = worker_provider_context_from_handle(main_handle);
        let key = worker_provider_context_key(&context);
        let slot = state.next_slots.get(&key).copied().unwrap_or(1);
        let next_slot = slot
            .checked_add(1)
            .filter(|slot| *slot != 0)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let retry_handle = worker_provider_handle(&context, slot);
        let retry_sdk_handle = ProviderStreamHandleV2 {
            invocation: sdk_handle.invocation,
            slot,
            generation: sdk_handle.generation,
        };
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        state
            .owner
            .retry_streaming(sdk_handle, retry_sdk_handle, &cancellation)?;
        let uploads = state
            .upload_parents
            .iter()
            .filter_map(|(upload, parent)| (*parent == main_handle).then_some(*upload))
            .collect::<Vec<_>>();
        state.handles.remove(&main_handle);
        state.main_handles.remove(&main_handle);
        state.cancellations.remove(&main_handle);
        for upload in uploads {
            state.upload_parents.remove(&upload);
            state.handles.remove(&upload);
        }
        state.next_slots.insert(key, next_slot);
        state.handles.insert(retry_handle, retry_sdk_handle);
        state.main_handles.insert(retry_handle);
        state.cancellations.insert(retry_handle, cancellation);
        Ok(retry_handle)
    }

    pub fn accept_wait(
        &self,
        request: WorkerProviderWaitRequest,
        outcome: WorkerProviderWaitOutcome,
    ) -> Result<(), PluginServiceError> {
        let mut state = self.state.lock();
        let (_, main_handle) = resolve_worker_handle(&state, request.handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let sdk_request_handle = state
            .handles
            .get(&request.handle)
            .copied()
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let request = ProviderWaitRequestV2 {
            handle: sdk_request_handle,
            after_sequence: request.after_sequence,
            timeout_milliseconds: request.timeout_milliseconds,
        };
        let revoke_after_accept = worker_wait_outcome_is_noncompleted_terminal(&outcome);
        let outcome = provider_wait_outcome(&state, outcome)?;
        let result = state
            .owner
            .accept_streaming_wait(&request, outcome, &cancellation);
        if result.is_ok() && revoke_after_accept {
            remove_worker_stream(&mut state, main_handle);
        }
        finish_worker_operation(&mut state, main_handle, result)
    }

    pub fn start_upload(
        &self,
        request: WorkerProviderUploadRequest,
    ) -> Result<WorkerProviderStreamHandle, PluginServiceError> {
        let mut state = self.state.lock();
        let (_, main_handle) = resolve_worker_handle(&state, request.handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let sdk_parent = state
            .handles
            .get(&main_handle)
            .copied()
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let context = worker_provider_context_from_handle(main_handle);
        let key = worker_provider_context_key(&context);
        let slot = state.next_slots.get(&key).copied().unwrap_or(1);
        let next_slot = slot
            .checked_add(1)
            .filter(|slot| *slot != 0)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        let worker_upload = worker_provider_handle(&context, slot);
        if state.handles.contains_key(&worker_upload) {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidUpload,
            ));
        }
        let sdk_upload = ProviderStreamHandleV2 {
            invocation: sdk_parent.invocation,
            slot,
            generation: sdk_parent.generation,
        };
        let request = ProviderUploadRequestV2 {
            handle: sdk_parent,
            port_id: request.port_id,
            media_type: request.media_type,
            byte_length: request.byte_length,
            content_sha256: request.content_sha256,
        };
        state
            .owner
            .start_streaming_upload(&request, sdk_upload, &cancellation)?;
        state.next_slots.insert(key, next_slot);
        state.handles.insert(worker_upload, sdk_upload);
        state.upload_parents.insert(worker_upload, main_handle);
        Ok(worker_upload)
    }

    pub fn write_upload_chunk(
        &self,
        chunk: WorkerProviderRequestChunk,
    ) -> Result<(), PluginServiceError> {
        let mut state = self.state.lock();
        let worker_handle = chunk.handle;
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, worker_handle, Some(false))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let chunk = provider_request_chunk(chunk, sdk_handle);
        let result = state
            .owner
            .write_streaming_upload_chunk(&chunk, &cancellation);
        finish_worker_operation(&mut state, main_handle, result)
    }

    pub fn accept_upload_completion(
        &self,
        request: WorkerProviderUploadCompletionRequest,
        completion: WorkerProviderUploadCompletion,
    ) -> Result<(), PluginServiceError> {
        let mut state = self.state.lock();
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, request.handle, Some(false))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        validate_provider_upload_url(&completion.download_url).map_err(|_| {
            PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidUpload,
            )
        })?;
        let result = state.owner.accept_streaming_upload_completion(
            sdk_handle,
            completion.download_url,
            &cancellation,
        );
        finish_worker_operation(&mut state, main_handle, result)
    }

    pub fn accept_streaming_cost(
        &self,
        request: WorkerProviderCostRequest,
        authorization: ProviderCostAuthorization,
        verifier: &ProviderCostAcceptanceVerifier,
        now: Instant,
    ) -> Result<WorkerProviderCostResponse, PluginServiceError> {
        let mut state = self.state.lock();
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, request.handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let request = provider_cost_request(request, sdk_handle);
        let cost_request_sha256 =
            worker_provider_cost_request_sha256(&state, main_handle, &request)?;
        let result = state.owner.accept_streaming_cost(
            &request,
            cost_request_sha256,
            authorization,
            verifier,
            now,
            &cancellation,
        );
        match result {
            Ok(response) => Ok(worker_cost_response(response)),
            Err(error) => {
                revoke_if_cancelled(&mut state, main_handle, &error);
                Err(error)
            }
        }
    }

    pub fn prepare_cost_acceptance(
        &self,
        request: &WorkerProviderCostRequest,
        price_bound: ProviderPriceBound,
    ) -> Result<ProviderCostAcceptanceScope, PluginServiceError> {
        let mut state = self.state.lock();
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, request.handle, Some(true))?;
        active_stream_cancellation(&mut state, main_handle)?;
        let sdk_request = ProviderCostRequestV2 {
            handle: sdk_handle,
            operation: request.operation.clone(),
            currency: request.currency.clone(),
            maximum_microunits: request.maximum_microunits,
        };
        let session = state
            .owner
            .streaming_sessions
            .get(&sdk_handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        if session.actuation_proposal.is_some()
            || !session.request_finished
            || session
                .upload_validators
                .values()
                .any(|upload| !upload.is_terminal())
            || session.upload_validators.len() != session.upload_download_urls.len()
            || request.currency != price_bound.currency_code()
            || request.maximum_microunits != price_bound.maximum_microunits()
        {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        let cost_request_sha256 =
            worker_provider_cost_request_sha256(&state, main_handle, &sdk_request)?;
        provider_streaming_cost_scope_with_request_sha256(session, cost_request_sha256, price_bound)
    }

    pub fn deny_streaming_cost(
        &self,
        request: WorkerProviderCostRequest,
    ) -> Result<WorkerProviderCostResponse, PluginServiceError> {
        let mut state = self.state.lock();
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, request.handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let request = provider_cost_request(request, sdk_handle);
        let result = state.owner.deny_streaming_cost(&request, &cancellation);
        match result {
            Ok(response) => Ok(worker_cost_response(response)),
            Err(error) => {
                revoke_if_cancelled(&mut state, main_handle, &error);
                Err(error)
            }
        }
    }

    pub fn report_progress(
        &self,
        progress: WorkerProviderProgress,
        now: Instant,
    ) -> Result<Option<ProviderRuntimeProgressProjection>, PluginServiceError> {
        let mut state = self.state.lock();
        let worker_handle = progress.handle;
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, worker_handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let progress = ProviderProgressV2 {
            handle: sdk_handle,
            sequence: progress.sequence,
            completed: progress.completed,
            total: progress.total,
            message: progress.message,
        };
        let result = state
            .owner
            .report_streaming_progress(&progress, now, &cancellation);
        match result {
            Ok(Some(update)) => Ok(Some(ProviderRuntimeProgressProjection {
                handle: worker_handle,
                sequence: update.sequence,
                completed: update.completed,
                total: update.total,
                message: update.message,
            })),
            Ok(None) => Ok(None),
            Err(error) => {
                revoke_if_cancelled(&mut state, main_handle, &error);
                Err(error)
            }
        }
    }

    pub fn prepare_streaming_actuation(
        &self,
        handle: WorkerProviderStreamHandle,
    ) -> Result<ProviderRuntimeActuationProposal, PluginServiceError> {
        let mut state = self.state.lock();
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let ordered_uploads_sha256 = worker_ordered_uploads_sha256(&state, main_handle)?;
        let result = state.owner.prepare_streaming_actuation(
            sdk_handle,
            Some(&ordered_uploads_sha256),
            &cancellation,
        );
        if let Err(error) = &result {
            revoke_if_cancelled(&mut state, main_handle, error);
        }
        result
    }

    pub fn finish_streaming(
        &self,
        handle: WorkerProviderStreamHandle,
        issuer: &ProviderRuntimeReceiptIssuerV2,
        issued_at: Instant,
        expires_at: Instant,
        nonce: [u8; 32],
    ) -> Result<ProviderRuntimeReceiptV2, PluginServiceError> {
        let mut state = self.state.lock();
        if state
            .http_invocations
            .contains_key(&worker_provider_context_key(
                &worker_provider_context_from_handle(handle),
            ))
        {
            return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
        }
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, handle, Some(true))?;
        let cancellation = active_stream_cancellation(&mut state, main_handle)?;
        let result = state.owner.finish_streaming(
            sdk_handle,
            issuer,
            issued_at,
            expires_at,
            nonce,
            &cancellation,
        );
        match result {
            Ok(receipt) => {
                remove_worker_stream(&mut state, main_handle);
                Ok(receipt)
            }
            Err(error) => {
                revoke_if_cancelled(&mut state, main_handle, &error);
                Err(error)
            }
        }
    }

    pub fn retry_after_seconds(
        &self,
        handle: WorkerProviderStreamHandle,
    ) -> Result<Option<u64>, PluginServiceError> {
        let mut state = self.state.lock();
        let (sdk_handle, main_handle) = resolve_worker_handle(&state, handle, Some(true))?;
        active_stream_cancellation(&mut state, main_handle)?;
        state.owner.streaming_retry_after_seconds(sdk_handle)
    }

    pub fn check_cancelled(
        &self,
        handle: WorkerProviderStreamHandle,
    ) -> Result<(), PluginServiceError> {
        let mut state = self.state.lock();
        let (_, main_handle) = resolve_worker_handle(&state, handle, None)?;
        active_stream_cancellation(&mut state, main_handle).map(|_| ())
    }

    pub fn revoke_stream(&self, handle: WorkerProviderStreamHandle) {
        let mut state = self.state.lock();
        if let Ok((_, main_handle)) = resolve_worker_handle(&state, handle, None) {
            revoke_worker_stream(&mut state, main_handle);
        }
    }

    pub fn revoke_invocation(
        &self,
        context: &WorkerProviderInvocationContext,
    ) -> Result<(), PluginServiceError> {
        validate_worker_provider_context(context)?;
        let mut state = self.state.lock();
        let main_handles = state
            .main_handles
            .iter()
            .copied()
            .filter(|handle| worker_handle_matches_context(*handle, context))
            .collect::<Vec<_>>();
        for main_handle in main_handles {
            revoke_worker_stream(&mut state, main_handle);
        }
        let grant_key = (
            context.session_id,
            context.session_generation,
            context.invocation,
        );
        if let Some((generation, grant)) = state.grants.grants.get_mut(&grant_key)
            && *generation == context.generation
        {
            *grant = None;
        }
        let claim_key = worker_provider_context_key(context);
        state.http_invocations.remove(&claim_key);
        if let Some(claim) = state.activation_claims.remove(&claim_key) {
            claim.store(false, Ordering::Release);
        }
        Ok(())
    }

    pub(crate) fn begin_legacy(
        &self,
        authority: &dyn NativeProviderInvocationAuthority,
        scope: NativeProviderInvocationScope,
    ) -> Result<(), PluginServiceError> {
        self.state.lock().owner.begin_legacy(authority, scope)
    }

    pub(crate) fn call_legacy(
        &self,
        session_id: &str,
        request: PluginServiceWireRequest,
    ) -> Result<PluginServiceWireResponse, PluginServiceError> {
        self.state.lock().owner.call_legacy(session_id, request)
    }

    pub(crate) fn resolve_legacy(
        &self,
        session_id: &str,
        receipt_set: &ProviderResultReceiptSet,
    ) -> Result<Vec<Vec<u8>>, PluginServiceError> {
        self.state
            .lock()
            .owner
            .resolve_legacy(session_id, receipt_set)
    }

    pub(crate) fn finish_legacy(&self, session_id: &str) -> Result<(), PluginServiceError> {
        self.state.lock().owner.finish_legacy(session_id)
    }

    pub(crate) fn abort_legacy(&self, session_id: &str) {
        self.state.lock().owner.abort_legacy(session_id);
    }

    pub fn revoke_all(&self) {
        let mut state = self.state.lock();
        state.owner.revoke_all();
        state.grants.grants.clear();
        state.grants.session_generations.clear();
        for (_, claim) in std::mem::take(&mut state.activation_claims) {
            claim.store(false, Ordering::Release);
        }
        state.http_invocations.clear();
        state.invocation_bindings.clear();
        state.handles.clear();
        state.main_handles.clear();
        state.upload_parents.clear();
        state.cancellations.clear();
        state.next_slots.clear();
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        let state = self.state.lock();
        state.owner.is_empty()
            && state.grants.grants.is_empty()
            && state.grants.session_generations.is_empty()
            && state.http_invocations.is_empty()
            && state.activation_claims.is_empty()
            && state.invocation_bindings.is_empty()
            && state.handles.is_empty()
            && state.main_handles.is_empty()
            && state.upload_parents.is_empty()
            && state.cancellations.is_empty()
            && state.next_slots.is_empty()
    }
}

fn worker_provider_context_key(
    context: &WorkerProviderInvocationContext,
) -> (uuid::Uuid, u64, u64, u32) {
    (
        context.session_id,
        context.session_generation,
        context.invocation,
        context.generation,
    )
}

fn worker_provider_context_from_handle(
    handle: WorkerProviderStreamHandle,
) -> WorkerProviderInvocationContext {
    WorkerProviderInvocationContext {
        session_id: handle.session_id,
        session_generation: handle.session_generation,
        invocation: handle.invocation,
        generation: handle.generation,
    }
}

fn worker_provider_handle(
    context: &WorkerProviderInvocationContext,
    slot: u32,
) -> WorkerProviderStreamHandle {
    WorkerProviderStreamHandle {
        session_id: context.session_id,
        session_generation: context.session_generation,
        invocation: context.invocation,
        slot,
        generation: context.generation,
    }
}

fn worker_handle_matches_context(
    handle: WorkerProviderStreamHandle,
    context: &WorkerProviderInvocationContext,
) -> bool {
    handle.session_id == context.session_id
        && handle.session_generation == context.session_generation
        && handle.invocation == context.invocation
        && handle.generation == context.generation
}

fn resolve_worker_handle(
    state: &ProviderRuntimeStreamState,
    handle: WorkerProviderStreamHandle,
    main: Option<bool>,
) -> Result<(ProviderStreamHandleV2, WorkerProviderStreamHandle), PluginServiceError> {
    validate_worker_provider_context(&worker_provider_context_from_handle(handle))?;
    if handle.slot == 0 {
        return Err(PluginServiceError::ProviderStreamingContract(
            ProviderStreamingContractError::InvalidHandle,
        ));
    }
    if let Some(sdk_handle) = state.handles.get(&handle).copied() {
        let is_main = state.main_handles.contains(&handle);
        if main.is_some_and(|expected| expected != is_main) {
            return Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::ForeignHandle,
            ));
        }
        let main_handle = if is_main {
            handle
        } else {
            *state
                .upload_parents
                .get(&handle)
                .ok_or(PluginServiceError::ProviderSessionUnavailable)?
        };
        return Ok((sdk_handle, main_handle));
    }
    Err(classify_missing_worker_handle(state, handle))
}

fn classify_missing_worker_handle(
    state: &ProviderRuntimeStreamState,
    handle: WorkerProviderStreamHandle,
) -> PluginServiceError {
    let Some(session_generation) = state
        .grants
        .session_generations
        .get(&handle.session_id)
        .copied()
        .or_else(|| {
            state
                .invocation_bindings
                .keys()
                .find_map(|(session_id, generation, _, _)| {
                    (*session_id == handle.session_id).then_some(*generation)
                })
        })
    else {
        return PluginServiceError::ProviderRuntimeForeignSession;
    };
    if session_generation != handle.session_generation {
        return PluginServiceError::ProviderRuntimeStaleSession;
    }
    let invocation_exists =
        state
            .invocation_bindings
            .keys()
            .any(|(session_id, generation, invocation, _)| {
                *session_id == handle.session_id
                    && *generation == handle.session_generation
                    && *invocation == handle.invocation
            })
            || state
                .grants
                .grants
                .keys()
                .any(|(session_id, generation, invocation)| {
                    *session_id == handle.session_id
                        && *generation == handle.session_generation
                        && *invocation == handle.invocation
                });
    if !invocation_exists {
        return PluginServiceError::ProviderRuntimeForeignInvocation;
    }
    let key = (
        handle.session_id,
        handle.session_generation,
        handle.invocation,
        handle.generation,
    );
    let current_invocation_generation = state
        .grants
        .grants
        .get(&(
            handle.session_id,
            handle.session_generation,
            handle.invocation,
        ))
        .map(|(generation, _)| *generation);
    if !state.invocation_bindings.contains_key(&key) {
        if current_invocation_generation == Some(handle.generation) {
            return PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidHandle,
            );
        }
        return PluginServiceError::ProviderRuntimeStaleInvocation;
    }
    if state
        .next_slots
        .get(&key)
        .is_none_or(|next_slot| handle.slot >= *next_slot)
    {
        return PluginServiceError::ProviderStreamingContract(
            ProviderStreamingContractError::InvalidHandle,
        );
    }
    PluginServiceError::ProviderStreamingContract(ProviderStreamingContractError::RevokedHandle)
}

fn classify_inactive_authority(
    state: &ProviderRuntimeStreamState,
    context: &WorkerProviderInvocationContext,
) -> PluginServiceError {
    let Some(session_generation) = state
        .grants
        .session_generations
        .get(&context.session_id)
        .copied()
    else {
        return PluginServiceError::ProviderRuntimeForeignSession;
    };
    if session_generation != context.session_generation {
        return PluginServiceError::ProviderRuntimeStaleSession;
    }
    let Some((generation, _)) = state.grants.grants.get(&(
        context.session_id,
        context.session_generation,
        context.invocation,
    )) else {
        return PluginServiceError::ProviderRuntimeForeignInvocation;
    };
    if *generation != context.generation {
        return PluginServiceError::ProviderRuntimeStaleInvocation;
    }
    PluginServiceError::ProviderStreamingContract(ProviderStreamingContractError::RevokedHandle)
}

fn active_stream_cancellation(
    state: &mut ProviderRuntimeStreamState,
    main_handle: WorkerProviderStreamHandle,
) -> Result<CancellationToken, PluginServiceError> {
    let cancellation = state
        .cancellations
        .get(&main_handle)
        .cloned()
        .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
    if cancellation.check().is_err() {
        revoke_worker_stream(state, main_handle);
        return Err(PluginServiceError::Cancelled);
    }
    Ok(cancellation)
}

fn finish_worker_operation(
    state: &mut ProviderRuntimeStreamState,
    main_handle: WorkerProviderStreamHandle,
    result: Result<(), PluginServiceError>,
) -> Result<(), PluginServiceError> {
    if let Err(error) = &result {
        revoke_if_cancelled(state, main_handle, error);
    }
    result
}

fn revoke_if_cancelled(
    state: &mut ProviderRuntimeStreamState,
    main_handle: WorkerProviderStreamHandle,
    error: &PluginServiceError,
) {
    if matches!(error, PluginServiceError::Cancelled) {
        revoke_worker_stream(state, main_handle);
    }
}

fn remove_worker_stream(
    state: &mut ProviderRuntimeStreamState,
    main_handle: WorkerProviderStreamHandle,
) {
    state.handles.remove(&main_handle);
    state.main_handles.remove(&main_handle);
    state.cancellations.remove(&main_handle);
    let uploads = state
        .upload_parents
        .iter()
        .filter_map(|(upload, parent)| (*parent == main_handle).then_some(*upload))
        .collect::<Vec<_>>();
    for upload in uploads {
        state.upload_parents.remove(&upload);
        state.handles.remove(&upload);
    }
}

fn revoke_worker_stream(
    state: &mut ProviderRuntimeStreamState,
    main_handle: WorkerProviderStreamHandle,
) {
    if let Some(sdk_handle) = state.handles.get(&main_handle).copied() {
        state.owner.revoke_streaming(sdk_handle);
    }
    remove_worker_stream(state, main_handle);
    state.http_invocations.remove(&worker_provider_context_key(
        &worker_provider_context_from_handle(main_handle),
    ));
}

fn provider_request_chunk(
    chunk: WorkerProviderRequestChunk,
    handle: ProviderStreamHandleV2,
) -> ProviderRequestChunkV2 {
    ProviderRequestChunkV2 {
        handle,
        sequence: chunk.sequence,
        bytes: chunk.bytes,
        end: chunk.end,
    }
}

fn worker_request_head_matches(
    worker: &WorkerProviderRequestHead,
    sdk: &ProviderRequestHeadV2,
) -> bool {
    worker.endpoint == sdk.endpoint
        && worker.secret_id == sdk.secret_id
        && provider_http_method(worker.method) == sdk.method
        && worker.declared_body_bytes == sdk.declared_body_bytes
        && worker.headers.len() == sdk.headers.len()
        && worker
            .headers
            .iter()
            .zip(&sdk.headers)
            .all(|(worker, sdk)| worker.name == sdk.name && worker.value == sdk.value)
}

fn provider_cost_request(
    request: WorkerProviderCostRequest,
    handle: ProviderStreamHandleV2,
) -> ProviderCostRequestV2 {
    ProviderCostRequestV2 {
        handle,
        operation: request.operation,
        currency: request.currency,
        maximum_microunits: request.maximum_microunits,
    }
}

fn worker_cost_response(response: ProviderCostResponseV2) -> WorkerProviderCostResponse {
    WorkerProviderCostResponse {
        accepted: response.accepted,
        approved_microunits: response.approved_microunits,
        receipt: response.receipt,
    }
}

fn provider_wait_outcome(
    state: &ProviderRuntimeStreamState,
    outcome: WorkerProviderWaitOutcome,
) -> Result<ProviderWaitOutcomeV2, PluginServiceError> {
    match outcome {
        WorkerProviderWaitOutcome::Frame(frame) => {
            let (handle, _) = resolve_worker_handle(state, frame.handle, Some(true))?;
            Ok(ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                handle,
                sequence: frame.sequence,
                event: provider_response_event(frame.event)?,
            }))
        }
        WorkerProviderWaitOutcome::TimedOut => Ok(ProviderWaitOutcomeV2::TimedOut),
        WorkerProviderWaitOutcome::Cancelled => Ok(ProviderWaitOutcomeV2::Cancelled),
    }
}

fn provider_response_event(
    event: WorkerProviderResponseFrameEvent,
) -> Result<ProviderResponseFrameEventV2, PluginServiceError> {
    Ok(match event {
        WorkerProviderResponseFrameEvent::Head(head) => {
            let mut headers = Vec::new();
            headers
                .try_reserve_exact(head.headers.len())
                .map_err(|_| PluginServiceError::ResponseAllocationFailed)?;
            headers.extend(head.headers.into_iter().map(provider_header));
            ProviderResponseFrameEventV2::Head(ProviderResponseHeadV2 {
                status: head.status,
                headers,
            })
        }
        WorkerProviderResponseFrameEvent::Chunk(chunk) => {
            ProviderResponseFrameEventV2::Chunk(match chunk {
                WorkerProviderResponseChunk::Binary(bytes) => {
                    ProviderResponseChunkV2::Binary(bytes)
                }
                WorkerProviderResponseChunk::Text(text) => ProviderResponseChunkV2::Text(text),
                WorkerProviderResponseChunk::NdjsonLine(line) => {
                    ProviderResponseChunkV2::NdjsonLine(line)
                }
            })
        }
        WorkerProviderResponseFrameEvent::Terminal(terminal) => {
            ProviderResponseFrameEventV2::Terminal(match terminal {
                WorkerProviderTerminal::Completed(receipt) => {
                    ProviderStreamTerminalV2::Completed { receipt }
                }
                WorkerProviderTerminal::Failed { code, message } => {
                    ProviderStreamTerminalV2::Failed { code, message }
                }
                WorkerProviderTerminal::Cancelled => ProviderStreamTerminalV2::Cancelled,
            })
        }
    })
}

fn provider_header(header: WorkerProviderHeader) -> ProviderHeaderV2 {
    ProviderHeaderV2 {
        name: header.name,
        value: header.value,
    }
}

fn provider_http_method(method: WorkerProviderHttpMethod) -> ProviderHttpMethodV2 {
    match method {
        WorkerProviderHttpMethod::Delete => ProviderHttpMethodV2::Delete,
        WorkerProviderHttpMethod::Get => ProviderHttpMethodV2::Get,
        WorkerProviderHttpMethod::Head => ProviderHttpMethodV2::Head,
        WorkerProviderHttpMethod::Options => ProviderHttpMethodV2::Options,
        WorkerProviderHttpMethod::Patch => ProviderHttpMethodV2::Patch,
        WorkerProviderHttpMethod::Post => ProviderHttpMethodV2::Post,
        WorkerProviderHttpMethod::Put => ProviderHttpMethodV2::Put,
    }
}

fn worker_wait_outcome_is_noncompleted_terminal(outcome: &WorkerProviderWaitOutcome) -> bool {
    matches!(outcome, WorkerProviderWaitOutcome::Cancelled)
        || matches!(
            outcome,
            WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                event: WorkerProviderResponseFrameEvent::Terminal(
                    WorkerProviderTerminal::Failed { .. } | WorkerProviderTerminal::Cancelled
                ),
                ..
            })
        )
}

#[allow(dead_code)]
fn revoke_replaced_host_context(
    state: &mut ProviderRuntimeStreamState,
    context: &WorkerProviderInvocationContext,
    session_replaced: bool,
) {
    let replaced_claims = state
        .activation_claims
        .keys()
        .copied()
        .filter(|(session_id, session_generation, invocation, generation)| {
            *session_id == context.session_id
                && (session_replaced && *session_generation < context.session_generation
                    || !session_replaced
                        && *session_generation == context.session_generation
                        && *invocation == context.invocation
                        && *generation < context.generation)
        })
        .collect::<Vec<_>>();
    for key in replaced_claims {
        if let Some(claim) = state.activation_claims.remove(&key) {
            claim.store(false, Ordering::Release);
        }
    }
    let replaced_main_handles = state
        .main_handles
        .iter()
        .copied()
        .filter(|handle| {
            handle.session_id == context.session_id
                && (session_replaced && handle.session_generation < context.session_generation
                    || !session_replaced
                        && handle.session_generation == context.session_generation
                        && handle.invocation == context.invocation
                        && handle.generation < context.generation)
        })
        .collect::<Vec<_>>();
    for main_handle in replaced_main_handles {
        if let Some(handle) = state.handles.remove(&main_handle) {
            state.owner.revoke_streaming(handle);
        }
        state.main_handles.remove(&main_handle);
        state.cancellations.remove(&main_handle);
        let uploads = state
            .upload_parents
            .iter()
            .filter_map(|(upload, parent)| (*parent == main_handle).then_some(*upload))
            .collect::<Vec<_>>();
        for upload in uploads {
            state.upload_parents.remove(&upload);
            state.handles.remove(&upload);
        }
    }
    state.http_invocations.retain(
        |(session_id, session_generation, invocation, generation), _| {
            !(*session_id == context.session_id
                && (session_replaced && *session_generation < context.session_generation
                    || !session_replaced
                        && *session_generation == context.session_generation
                        && *invocation == context.invocation
                        && *generation < context.generation))
        },
    );
    state.invocation_bindings.retain(
        |(session_id, session_generation, invocation, generation), _| {
            !(*session_id == context.session_id
                && (session_replaced && *session_generation < context.session_generation
                    || !session_replaced
                        && *session_generation == context.session_generation
                        && *invocation == context.invocation
                        && *generation < context.generation))
        },
    );
    state.next_slots.retain(
        |(session_id, session_generation, invocation, generation), _| {
            !(*session_id == context.session_id
                && (session_replaced && *session_generation < context.session_generation
                    || !session_replaced
                        && *session_generation == context.session_generation
                        && *invocation == context.invocation
                        && *generation < context.generation))
        },
    );
}

fn provider_streaming_cost_request_sha256(
    worker_handle: WorkerProviderStreamHandle,
    request_body_sha256: &str,
    request: &ProviderCostRequestV2,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-streaming-cost-request-v2\0");
    digest.update(worker_handle.session_id.as_bytes());
    digest.update(worker_handle.session_generation.to_le_bytes());
    digest.update(worker_handle.invocation.to_le_bytes());
    digest.update(worker_handle.slot.to_le_bytes());
    digest.update(worker_handle.generation.to_le_bytes());
    for value in [
        request_body_sha256,
        request.operation.as_str(),
        request.currency.as_str(),
    ] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    digest.update(request.maximum_microunits.to_le_bytes());
    format!("{:x}", digest.finalize())
}

fn provider_streaming_cost_scope_with_request_sha256(
    session: &ProviderRuntimeStreamingSession,
    cost_request_sha256: String,
    price_bound: ProviderPriceBound,
) -> Result<ProviderCostAcceptanceScope, PluginServiceError> {
    let plugin_authorization = session.authority.manifest_authorization.authorization();
    ProviderCostAcceptanceScope::new(
        session.authority.principal_id.as_str(),
        session.authority.profile_id.as_str(),
        session.authority.prompt_id.as_str(),
        session.authority.prompt_sha256.as_str(),
        session.authority.attempt_id.as_str(),
        session.authority.node_id.as_str(),
        session.authority.request_ordinal,
        cost_request_sha256,
        plugin_authorization.plugin_id(),
        plugin_authorization.digest_sha256(),
        session.authority.binding_set_sha256.as_str(),
        session.authority.provider.as_str(),
        session.authority.request.endpoint(),
        price_bound,
    )
    .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)
}

fn worker_provider_cost_request_sha256(
    state: &ProviderRuntimeStreamState,
    main_handle: WorkerProviderStreamHandle,
    request: &ProviderCostRequestV2,
) -> Result<String, PluginServiceError> {
    let sdk_main = state
        .handles
        .get(&main_handle)
        .copied()
        .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
    let session = state
        .owner
        .streaming_sessions
        .get(&sdk_main)
        .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
    let ordered_uploads_sha256 = worker_ordered_uploads_sha256(state, main_handle)?;
    let mut request_identity = Sha256::new();
    request_identity.update(b"zed-sim-provider-worker-streaming-request-v2\0");
    request_identity.update(request_body_sha256(session).as_bytes());
    request_identity.update(ordered_uploads_sha256.as_bytes());
    Ok(provider_streaming_cost_request_sha256(
        main_handle,
        &format!("{:x}", request_identity.finalize()),
        request,
    ))
}

fn worker_ordered_uploads_sha256(
    state: &ProviderRuntimeStreamState,
    main_handle: WorkerProviderStreamHandle,
) -> Result<String, PluginServiceError> {
    let sdk_main = state
        .handles
        .get(&main_handle)
        .copied()
        .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
    let session = state
        .owner
        .streaming_sessions
        .get(&sdk_main)
        .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
    if session
        .upload_validators
        .values()
        .any(|upload| !upload.is_terminal())
        || session.upload_validators.len() != session.upload_download_urls.len()
    {
        return Err(PluginServiceError::ProviderStreamingContract(
            ProviderStreamingContractError::InvalidUpload,
        ));
    }
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-semantic-ordered-uploads-v2\0");
    digest.update((session.ordered_uploads.len() as u64).to_le_bytes());
    for (ordinal, upload) in session.ordered_uploads.iter().enumerate() {
        digest.update((ordinal as u64).to_le_bytes());
        for value in [
            upload.port_id.as_str(),
            upload.media_type.as_str(),
            upload.content_sha256.as_str(),
        ] {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        digest.update(upload.byte_length.to_le_bytes());
        let download_url = session
            .upload_download_urls
            .get(&upload.handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        digest.update((download_url.len() as u64).to_le_bytes());
        digest.update(download_url.as_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn valid_provider_runtime_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_PROVIDER_RUNTIME_IDENTITY_BYTES
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn validate_worker_provider_context(
    context: &WorkerProviderInvocationContext,
) -> Result<(), PluginServiceError> {
    if context.session_id.is_nil()
        || context.session_generation == 0
        || context.invocation == 0
        || context.generation == 0
    {
        return Err(PluginServiceError::ProviderRuntimeAuthorityDenied);
    }
    Ok(())
}

#[allow(dead_code)]
fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn map_provider_streaming_error(error: ProviderStreamingContractError) -> PluginServiceError {
    PluginServiceError::ProviderStreamingContract(error)
}

fn ordered_headers_sha256(headers: &[ProviderHeaderV2]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-ordered-headers-v2\0");
    digest.update((headers.len() as u64).to_le_bytes());
    for header in headers {
        digest.update((header.name.len() as u64).to_le_bytes());
        digest.update(header.name.as_bytes());
        digest.update((header.value.len() as u64).to_le_bytes());
        digest.update(header.value.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn request_body_sha256(session: &ProviderRuntimeStreamingSession) -> String {
    format!("{:x}", session.request_body_digest.clone().finalize())
}

fn ordered_uploads_sha256(
    session: &ProviderRuntimeStreamingSession,
) -> Result<String, PluginServiceError> {
    if session
        .upload_validators
        .values()
        .any(|upload| !upload.is_terminal())
        || session.upload_validators.len() != session.upload_download_urls.len()
    {
        return Err(PluginServiceError::ProviderStreamingContract(
            ProviderStreamingContractError::InvalidUpload,
        ));
    }
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-ordered-uploads-v2\0");
    digest.update((session.ordered_uploads.len() as u64).to_le_bytes());
    for upload in &session.ordered_uploads {
        digest.update(upload.handle.invocation.to_le_bytes());
        digest.update(upload.handle.slot.to_le_bytes());
        digest.update(upload.handle.generation.to_le_bytes());
        for value in [
            upload.port_id.as_str(),
            upload.media_type.as_str(),
            upload.content_sha256.as_str(),
        ] {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        digest.update(upload.byte_length.to_le_bytes());
        let download_url = session
            .upload_download_urls
            .get(&upload.handle)
            .ok_or(PluginServiceError::ProviderSessionUnavailable)?;
        digest.update((download_url.len() as u64).to_le_bytes());
        digest.update(download_url.as_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn ordered_chunks_digest() -> Sha256 {
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-ordered-response-chunks-v2\0");
    digest
}

fn updated_ordered_chunks_digest(
    current: &Sha256,
    sequence: u64,
    chunk: &sim_plugin_sdk::ProviderResponseChunkV2,
) -> Sha256 {
    let (tag, bytes): (u8, &[u8]) = match chunk {
        sim_plugin_sdk::ProviderResponseChunkV2::Binary(bytes) => (0, bytes),
        sim_plugin_sdk::ProviderResponseChunkV2::Text(text) => (1, text.as_bytes()),
        sim_plugin_sdk::ProviderResponseChunkV2::NdjsonLine(line) => (2, line.as_bytes()),
    };
    let mut digest = current.clone();
    digest.update(sequence.to_le_bytes());
    digest.update([tag]);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    digest
}

fn terminal_event_sha256(
    terminal: &sim_plugin_sdk::ProviderStreamTerminalV2,
) -> Result<String, PluginServiceError> {
    if let sim_plugin_sdk::ProviderStreamTerminalV2::Completed { receipt } = terminal {
        return Ok(provider_terminal_completed_receipt_sha256(receipt));
    }
    let encoded = serde_json::to_vec(terminal)
        .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-terminal-event-v2\0");
    digest.update((encoded.len() as u64).to_le_bytes());
    digest.update(encoded);
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn provider_terminal_completed_receipt_sha256(receipt: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-terminal-completed-receipt-v2\0");
    digest.update((receipt.len() as u64).to_le_bytes());
    digest.update(receipt);
    format!("{:x}", digest.finalize())
}

fn retry_after_seconds(outcome: &ProviderWaitOutcomeV2) -> Result<Option<u64>, PluginServiceError> {
    let ProviderWaitOutcomeV2::Frame(frame) = outcome else {
        return Ok(None);
    };
    let ProviderResponseFrameEventV2::Head(head) = &frame.event else {
        return Ok(None);
    };
    let values = head
        .headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case("retry-after"))
        .map(|header| header.value.as_str())
        .collect::<Vec<_>>();
    match values.as_slice() {
        [] => Ok(None),
        [value]
            if !value.is_empty()
                && value.bytes().all(|byte| byte.is_ascii_digit())
                && (!value.starts_with('0') || *value == "0") =>
        {
            value
                .parse::<u64>()
                .ok()
                .filter(|seconds| *seconds <= MAX_PROVIDER_RUNTIME_RETRY_AFTER_SECONDS)
                .map(Some)
                .ok_or(PluginServiceError::ProviderStreamingContract(
                    ProviderStreamingContractError::InvalidHeaders,
                ))
        }
        _ => Err(PluginServiceError::ProviderStreamingContract(
            ProviderStreamingContractError::InvalidHeaders,
        )),
    }
}

fn provider_runtime_mutation_identity_sha256(
    authority: &ProviderRuntimeAuthorityInput,
    request_body_sha256: &str,
    ordered_uploads_sha256: &str,
    accepted_cost_microunits: u64,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-runtime-mutation-v2\0");
    digest.update(authority.idempotency_identity_sha256().as_bytes());
    digest.update(request_body_sha256.as_bytes());
    digest.update(ordered_uploads_sha256.as_bytes());
    digest.update(authority.request_ordinal.to_le_bytes());
    digest.update(accepted_cost_microunits.to_le_bytes());
    format!("{:x}", digest.finalize())
}

impl Drop for ProviderRuntimeStreamOwner {
    fn drop(&mut self) {
        self.revoke_all();
    }
}

pub struct ProviderCostAuthorizationRequest<'a> {
    identity: &'a ProviderInvocationIdentity,
    price_badge: &'a NativeSchemaValue,
}

impl<'a> ProviderCostAuthorizationRequest<'a> {
    pub fn identity(&self) -> &'a ProviderInvocationIdentity {
        self.identity
    }

    pub fn price_badge(&self) -> &'a NativeSchemaValue {
        self.price_badge
    }
}

pub struct ProviderCostAuthorization {
    price_bound: ProviderPriceBound,
    nonce: ProviderCostNonce,
    acceptance: ProviderCostAcceptance,
}

impl ProviderCostAuthorization {
    pub fn new(
        price_bound: ProviderPriceBound,
        nonce: ProviderCostNonce,
        acceptance: ProviderCostAcceptance,
    ) -> Result<Self, PluginServiceError> {
        if acceptance.nonce() != nonce {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        Ok(Self {
            price_bound,
            nonce,
            acceptance,
        })
    }

    pub fn price_bound(&self) -> &ProviderPriceBound {
        &self.price_bound
    }

    pub fn nonce(&self) -> ProviderCostNonce {
        self.nonce
    }

    pub fn acceptance(&self) -> &ProviderCostAcceptance {
        &self.acceptance
    }
}

impl fmt::Debug for ProviderCostAuthorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderCostAuthorization([SEALED])")
    }
}

pub trait ProviderCostAuthorizationAuthority: Send + Sync {
    fn authorize(
        &self,
        request: ProviderCostAuthorizationRequest<'_>,
    ) -> Result<ProviderCostAuthorization, PluginServiceError>;
}

#[derive(Clone, Debug)]
pub struct ProviderCostApproval {
    principal_id: String,
    profile_id: String,
    prompt_sha256: String,
    node_id: String,
    request_ordinal: u32,
    plugin_id: String,
    plugin_digest_sha256: String,
    provider_binding_sha256: String,
    provider: String,
    endpoint: String,
    price_badge_sha256: String,
    price_bound: ProviderPriceBound,
    expires_at: Instant,
    nonce: ProviderCostNonce,
}

impl ProviderCostApproval {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        principal_id: impl Into<String>,
        profile_id: impl Into<String>,
        prompt_sha256: impl Into<String>,
        node_id: impl Into<String>,
        request_ordinal: u32,
        plugin_id: impl Into<String>,
        plugin_digest_sha256: impl Into<String>,
        provider_binding_sha256: impl Into<String>,
        provider: impl Into<String>,
        endpoint: impl Into<String>,
        price_badge: &NativeSchemaValue,
        price_bound: ProviderPriceBound,
        expires_at: Instant,
        nonce: ProviderCostNonce,
    ) -> Result<Self, PluginServiceError> {
        price_badge
            .validate()
            .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
        let price_badge_sha256 = schema_value_sha256(price_badge)?;
        let approval = Self {
            principal_id: principal_id.into(),
            profile_id: profile_id.into(),
            prompt_sha256: prompt_sha256.into(),
            node_id: node_id.into(),
            request_ordinal,
            plugin_id: plugin_id.into(),
            plugin_digest_sha256: plugin_digest_sha256.into(),
            provider_binding_sha256: provider_binding_sha256.into(),
            provider: provider.into(),
            endpoint: endpoint.into(),
            price_badge_sha256,
            price_bound,
            expires_at,
            nonce,
        };
        approval.key()?;
        Ok(approval)
    }

    fn key(&self) -> Result<String, PluginServiceError> {
        approval_key(
            &self.principal_id,
            &self.profile_id,
            &self.prompt_sha256,
            &self.node_id,
            self.request_ordinal,
            &self.plugin_id,
            &self.plugin_digest_sha256,
            &self.provider_binding_sha256,
            &self.provider,
            &self.endpoint,
            &self.price_badge_sha256,
        )
    }
}

pub struct ProviderCostApprovalAuthority {
    issuer: Arc<ProviderCostAcceptanceIssuer>,
    clock: Arc<dyn SystemClock>,
    approvals: Mutex<BTreeMap<String, ProviderCostApproval>>,
}

impl ProviderCostApprovalAuthority {
    pub fn new(issuer: Arc<ProviderCostAcceptanceIssuer>, clock: Arc<dyn SystemClock>) -> Self {
        Self {
            issuer,
            clock,
            approvals: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn verifier(&self) -> Result<ProviderCostAcceptanceVerifier, PluginServiceError> {
        self.issuer
            .verifier()
            .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)
    }

    pub fn approve(&self, approval: ProviderCostApproval) -> Result<(), PluginServiceError> {
        if approval.expires_at <= self.clock.utc_now() {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        let key = approval.key()?;
        let mut approvals = self.approvals.lock();
        approvals.retain(|_, approval| approval.expires_at > self.clock.utc_now());
        if approvals.contains_key(&key) {
            return Err(PluginServiceError::ProviderCostAcceptanceReused);
        }
        approvals.insert(key, approval);
        Ok(())
    }
}

impl fmt::Debug for ProviderCostApprovalAuthority {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderCostApprovalAuthority([REDACTED])")
    }
}

impl ProviderCostAuthorizationAuthority for ProviderCostApprovalAuthority {
    fn authorize(
        &self,
        request: ProviderCostAuthorizationRequest<'_>,
    ) -> Result<ProviderCostAuthorization, PluginServiceError> {
        let identity = request.identity();
        let price_badge_sha256 = schema_value_sha256(request.price_badge())?;
        let key = approval_key(
            identity.principal_id(),
            identity.profile_id(),
            identity.prompt_sha256(),
            identity.node_id(),
            identity.request_ordinal(),
            identity.plugin_id(),
            identity.plugin_digest_sha256(),
            identity.provider_binding_sha256(),
            identity.provider(),
            identity.endpoint(),
            &price_badge_sha256,
        )?;
        let approval = self
            .approvals
            .lock()
            .remove(&key)
            .ok_or(PluginServiceError::ProviderCostAcceptanceRequired)?;
        let now = self.clock.utc_now();
        if approval.expires_at <= now {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        let scope = ProviderCostAcceptanceScope::new(
            identity.principal_id(),
            identity.profile_id(),
            identity.prompt_id(),
            identity.prompt_sha256(),
            identity.attempt_id(),
            identity.node_id(),
            identity.request_ordinal(),
            identity.request_sha256(),
            identity.plugin_id(),
            identity.plugin_digest_sha256(),
            identity.provider_binding_sha256(),
            identity.provider(),
            identity.endpoint(),
            approval.price_bound.clone(),
        )
        .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
        let maximum_expiration = now
            .checked_add(Duration::from_secs(5 * 60))
            .ok_or(PluginServiceError::ProviderCostAcceptanceDenied)?;
        let acceptance = self
            .issuer
            .issue(
                scope,
                now,
                approval.expires_at.min(maximum_expiration),
                approval.nonce,
            )
            .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
        ProviderCostAuthorization::new(approval.price_bound, approval.nonce, acceptance)
    }
}

#[allow(clippy::too_many_arguments)]
fn approval_key(
    principal_id: &str,
    profile_id: &str,
    prompt_sha256: &str,
    node_id: &str,
    request_ordinal: u32,
    plugin_id: &str,
    plugin_digest_sha256: &str,
    provider_binding_sha256: &str,
    provider: &str,
    endpoint: &str,
    price_badge_sha256: &str,
) -> Result<String, PluginServiceError> {
    let identity = ProviderInvocationIdentity::new(
        principal_id,
        profile_id,
        "provider-cost-approval",
        prompt_sha256,
        "provider-cost-approval",
        node_id,
        request_ordinal,
        "0000000000000000000000000000000000000000000000000000000000000000",
        plugin_id,
        plugin_digest_sha256,
        provider_binding_sha256,
        provider,
        endpoint,
    )
    .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
    let bytes = serde_json::to_vec(&(
        identity.principal_id(),
        identity.profile_id(),
        identity.prompt_sha256(),
        identity.node_id(),
        identity.request_ordinal(),
        identity.plugin_id(),
        identity.plugin_digest_sha256(),
        identity.provider_binding_sha256(),
        identity.provider(),
        identity.endpoint(),
        price_badge_sha256,
    ))
    .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn schema_value_sha256(value: &NativeSchemaValue) -> Result<String, PluginServiceError> {
    let bytes =
        serde_json::to_vec(value).map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[derive(Clone)]
struct ProviderInvocationCostAuthority {
    price_badge: NativeSchemaValue,
    authority: Option<Arc<dyn ProviderCostAuthorizationAuthority>>,
}

impl fmt::Debug for ProviderInvocationCostAuthority {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderInvocationCostAuthority")
            .field("price_badge", &self.price_badge)
            .field("authority", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum PluginServiceWireRequest {
    ReadAsset {
        namespace: String,
        asset_reference: String,
    },
    ExecuteProvider {
        provider: String,
        endpoint: String,
        secret_id: Option<String>,
        body: Vec<u8>,
    },
    CredentialIsPresent {
        secret_id: String,
    },
    MonotonicMilliseconds {
        clock_id: String,
    },
    RandomBytes {
        stream_id: String,
        length: u32,
    },
    LoadModel {
        model_id: String,
    },
    SanitizeLog {
        level: String,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginServiceWireFailure {
    InvalidRequest,
    CapabilityDenied,
    Cancelled,
    DeadlineExceeded,
    ResponseTooLarge,
    ServiceUnavailable,
    ProviderDenied,
    ActuatorFailed,
    RandomnessFailed,
    InvocationFailed,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum PluginServiceWireResponse {
    Bytes(Vec<u8>),
    Boolean(bool),
    TimestampMilliseconds(u64),
    Model {
        identifier: String,
        format: String,
        digest_sha256: String,
    },
    SanitizedLog(String),
    Failure(PluginServiceWireFailure),
}

impl PluginServiceWireRequest {
    pub fn to_bytes(&self) -> Result<Vec<u8>, PluginServiceError> {
        let bytes =
            postcard::to_stdvec(self).map_err(|_| PluginServiceError::InvalidWirePayload)?;
        check_request_size(bytes.len())?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PluginServiceError> {
        if bytes.is_empty() {
            return Err(PluginServiceError::InvalidWirePayload);
        }
        check_request_size(bytes.len())?;
        postcard::from_bytes(bytes).map_err(|_| PluginServiceError::InvalidWirePayload)
    }
}

impl PluginServiceWireResponse {
    pub fn to_bytes(&self, maximum: u64) -> Result<Vec<u8>, PluginServiceError> {
        let bytes =
            postcard::to_stdvec(self).map_err(|_| PluginServiceError::InvalidWirePayload)?;
        check_response_size(bytes.len(), maximum)?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8], maximum: u64) -> Result<Self, PluginServiceError> {
        if bytes.is_empty() {
            return Err(PluginServiceError::InvalidWirePayload);
        }
        check_response_size(bytes.len(), maximum)?;
        postcard::from_bytes(bytes).map_err(|_| PluginServiceError::InvalidWirePayload)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PluginRngPolicy {
    profile: RngProfileVersion,
    algorithm: RngAlgorithm,
    seed: u64,
}

impl PluginRngPolicy {
    pub fn new(profile: RngProfileVersion, algorithm: RngAlgorithm, seed: u64) -> Self {
        Self {
            profile,
            algorithm,
            seed,
        }
    }

    pub fn profile(&self) -> RngProfileVersion {
        self.profile
    }

    pub fn algorithm(&self) -> RngAlgorithm {
        self.algorithm
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

#[derive(Clone, Debug)]
pub struct PluginServiceInvocationContext {
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: NodeId,
    principal_id: Option<String>,
    authorization: PluginAuthorization,
    cancellation: CancellationToken,
    deadline: Instant,
    maximum_response_bytes: u64,
    provider_result_authority: Option<ProviderResultReceiptAuthority>,
    provider_cost_authority: Option<ProviderInvocationCostAuthority>,
}

impl PluginServiceInvocationContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: NodeId,
        authorization: PluginAuthorization,
        cancellation: CancellationToken,
        deadline: Instant,
        maximum_response_bytes: u64,
    ) -> Result<Self, PluginServiceError> {
        Self::checked(
            profile_id,
            prompt_id,
            attempt_id,
            node_id,
            None,
            authorization,
            cancellation,
            deadline,
            maximum_response_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_principal(
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: NodeId,
        principal_id: impl Into<String>,
        authorization: PluginAuthorization,
        cancellation: CancellationToken,
        deadline: Instant,
        maximum_response_bytes: u64,
    ) -> Result<Self, PluginServiceError> {
        let principal_id = principal_id.into();
        if principal_id.is_empty()
            || principal_id.len() > MAX_PLUGIN_SERVICE_IDENTITY_BYTES
            || principal_id != principal_id.trim()
            || principal_id.chars().any(char::is_control)
        {
            return Err(PluginServiceError::InvalidIdentity { kind: "principal" });
        }
        Self::checked(
            profile_id,
            prompt_id,
            attempt_id,
            node_id,
            Some(principal_id),
            authorization,
            cancellation,
            deadline,
            maximum_response_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn checked(
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: NodeId,
        principal_id: Option<String>,
        authorization: PluginAuthorization,
        cancellation: CancellationToken,
        deadline: Instant,
        maximum_response_bytes: u64,
    ) -> Result<Self, PluginServiceError> {
        validate_identity("node", &node_id.0)?;
        let canonical_profile_id = profile_id.0.to_string();
        if authorization.capabilities().profile_id() != canonical_profile_id {
            return Err(PluginServiceError::ProfileMismatch);
        }
        if authorization.capabilities().subject_id() != authorization.plugin_id() {
            return Err(PluginServiceError::AuthorizationIdentityMismatch);
        }
        if maximum_response_bytes == 0 || maximum_response_bytes > MAX_PLUGIN_SERVICE_RESPONSE_BYTES
        {
            return Err(PluginServiceError::InvalidResponseLimit {
                maximum: MAX_PLUGIN_SERVICE_RESPONSE_BYTES,
            });
        }
        Ok(Self {
            profile_id,
            prompt_id,
            attempt_id,
            node_id,
            principal_id,
            authorization,
            cancellation,
            deadline,
            maximum_response_bytes,
            provider_result_authority: None,
            provider_cost_authority: None,
        })
    }

    pub fn with_provider_result_authority(
        mut self,
        authority: ProviderResultReceiptAuthority,
    ) -> Result<Self, PluginServiceError> {
        if self.principal_id() != Some(authority.principal_id()) {
            return Err(PluginServiceError::ProviderResultReceiptAuthorityDenied);
        }
        self.provider_result_authority = Some(authority);
        Ok(self)
    }

    pub fn with_provider_cost_authority(
        self,
        price_badge: NativeSchemaValue,
        authority: Arc<dyn ProviderCostAuthorizationAuthority>,
    ) -> Result<Self, PluginServiceError> {
        self.with_provider_cost_requirement(price_badge, Some(authority))
    }

    pub fn with_provider_cost_requirement(
        mut self,
        price_badge: NativeSchemaValue,
        authority: Option<Arc<dyn ProviderCostAuthorizationAuthority>>,
    ) -> Result<Self, PluginServiceError> {
        price_badge
            .validate()
            .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
        if self.principal_id().is_none() || self.provider_result_authority().is_none() {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        self.provider_cost_authority = Some(ProviderInvocationCostAuthority {
            price_badge,
            authority,
        });
        Ok(self)
    }

    pub fn profile_id(&self) -> ProfileId {
        self.profile_id
    }

    pub fn prompt_id(&self) -> PromptId {
        self.prompt_id
    }

    pub fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn principal_id(&self) -> Option<&str> {
        self.principal_id.as_deref()
    }

    pub fn plugin_id(&self) -> &str {
        self.authorization.plugin_id()
    }

    pub fn plugin_digest_sha256(&self) -> &str {
        self.authorization.digest_sha256()
    }

    pub fn authorization(&self) -> &PluginAuthorization {
        &self.authorization
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

    pub fn provider_result_authority(&self) -> Option<&ProviderResultReceiptAuthority> {
        self.provider_result_authority.as_ref()
    }

    fn provider_cost_authority(&self) -> Option<&ProviderInvocationCostAuthority> {
        self.provider_cost_authority.as_ref()
    }
}

pub struct PluginServiceOperationContext<'a> {
    invocation: &'a PluginServiceInvocationContext,
    clock: &'a dyn SystemClock,
    clock_origin: Instant,
}

impl PluginServiceOperationContext<'_> {
    pub fn invocation(&self) -> &PluginServiceInvocationContext {
        self.invocation
    }

    pub fn cancellation(&self) -> &CancellationToken {
        self.invocation.cancellation()
    }

    pub fn maximum_response_bytes(&self) -> u64 {
        self.invocation.maximum_response_bytes()
    }

    pub fn check_active(&self) -> Result<(), PluginServiceError> {
        if self.invocation.cancellation.is_cancelled() {
            return Err(PluginServiceError::Cancelled);
        }
        let now = self.clock.utc_now();
        now.checked_duration_since(self.clock_origin)
            .ok_or(PluginServiceError::ClockMovedBackwards)?;
        if now >= self.invocation.deadline {
            return Err(PluginServiceError::DeadlineExceeded);
        }
        Ok(())
    }

    pub fn remaining_time(&self) -> Result<Duration, PluginServiceError> {
        self.check_active()?;
        let now = self.clock.utc_now();
        self.invocation
            .deadline
            .checked_duration_since(now)
            .ok_or(PluginServiceError::DeadlineExceeded)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginServiceActuatorError {
    message: String,
}

impl PluginServiceActuatorError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: sanitize_actuator_message(&message.into()),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for PluginServiceActuatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PluginServiceActuatorError {}

pub trait ProviderRequestActuator: Send + Sync {
    fn execute(
        &self,
        request: &AuthorizedProviderRequest,
        secret: Option<&SecretValue>,
        body: &[u8],
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Vec<u8>, PluginServiceActuatorError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorizedCredentialPresenceRequest {
    profile_id: ProfileId,
    plugin_id: String,
    secret_id: SecretId,
}

impl AuthorizedCredentialPresenceRequest {
    pub fn profile_id(&self) -> ProfileId {
        self.profile_id
    }

    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    pub fn secret_id(&self) -> &SecretId {
        &self.secret_id
    }
}

pub trait CredentialPresenceActuator: Send + Sync {
    fn is_present(
        &self,
        request: &AuthorizedCredentialPresenceRequest,
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<bool, PluginServiceActuatorError>;

    fn read_for_provider(
        &self,
        request: &AuthorizedCredentialPresenceRequest,
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Option<SecretValue>, PluginServiceActuatorError>;
}

#[derive(Clone)]
pub struct PluginModelHandle {
    model_id: String,
    model_identity: String,
    model_format: &'static str,
    accounting: ModelLoadAccounting,
    model: Arc<LoadedModel>,
}

impl PluginModelHandle {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }

    pub fn model_format(&self) -> &'static str {
        self.model_format
    }

    pub fn accounting(&self) -> &ModelLoadAccounting {
        &self.accounting
    }

    pub fn model(&self) -> Arc<LoadedModel> {
        self.model.clone()
    }
}

impl fmt::Debug for PluginModelHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PluginModelHandle")
            .field("model_id", &self.model_id)
            .field("model_identity", &self.model_identity)
            .field("model_format", &self.model_format)
            .field("accounting", &self.accounting)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct PluginCapabilityBroker {
    inner: Arc<PluginCapabilityBrokerInner>,
}

struct PluginCapabilityBrokerInner {
    assets: SharedAssetService,
    model_store: Mutex<ModelStore>,
    provider_policy: ProviderPolicy,
    provider_cost_acceptance_verifier: Option<ProviderCostAcceptanceVerifier>,
    consumed_provider_cost_nonces: Mutex<BTreeMap<ProviderCostNonce, Instant>>,
    provider_actuator: Arc<dyn ProviderRequestActuator>,
    credential_presence_actuator: Arc<dyn CredentialPresenceActuator>,
    clock: Arc<dyn SystemClock>,
    clock_origin: Instant,
    rng_policy: PluginRngPolicy,
    rng_state: Mutex<PluginRngState>,
}

#[derive(Default)]
struct PluginRngState {
    checkpoints: BTreeMap<PluginRngKey, RngCheckpoint>,
    active_streams: BTreeSet<PluginRngKey>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PluginRngKey {
    profile_id: String,
    prompt_id: String,
    attempt_id: String,
    node_id: NodeId,
    plugin_id: String,
    plugin_digest_sha256: String,
    stream_id: String,
}

impl PluginCapabilityBroker {
    pub fn new(
        assets: SharedAssetService,
        model_store: ModelStore,
        provider_policy: ProviderPolicy,
        provider_actuator: Arc<dyn ProviderRequestActuator>,
        credential_presence_actuator: Arc<dyn CredentialPresenceActuator>,
        clock: Arc<dyn SystemClock>,
        rng_policy: PluginRngPolicy,
    ) -> Self {
        Self::new_internal(
            assets,
            model_store,
            provider_policy,
            None,
            provider_actuator,
            credential_presence_actuator,
            clock,
            rng_policy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_provider_cost_acceptance(
        assets: SharedAssetService,
        model_store: ModelStore,
        provider_policy: ProviderPolicy,
        provider_cost_acceptance_verifier: ProviderCostAcceptanceVerifier,
        provider_actuator: Arc<dyn ProviderRequestActuator>,
        credential_presence_actuator: Arc<dyn CredentialPresenceActuator>,
        clock: Arc<dyn SystemClock>,
        rng_policy: PluginRngPolicy,
    ) -> Self {
        Self::new_internal(
            assets,
            model_store,
            provider_policy,
            Some(provider_cost_acceptance_verifier),
            provider_actuator,
            credential_presence_actuator,
            clock,
            rng_policy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_internal(
        assets: SharedAssetService,
        model_store: ModelStore,
        provider_policy: ProviderPolicy,
        provider_cost_acceptance_verifier: Option<ProviderCostAcceptanceVerifier>,
        provider_actuator: Arc<dyn ProviderRequestActuator>,
        credential_presence_actuator: Arc<dyn CredentialPresenceActuator>,
        clock: Arc<dyn SystemClock>,
        rng_policy: PluginRngPolicy,
    ) -> Self {
        let clock_origin = clock.utc_now();
        Self {
            inner: Arc::new(PluginCapabilityBrokerInner {
                assets,
                model_store: Mutex::new(model_store),
                provider_policy,
                provider_cost_acceptance_verifier,
                consumed_provider_cost_nonces: Mutex::new(BTreeMap::new()),
                provider_actuator,
                credential_presence_actuator,
                clock,
                clock_origin,
                rng_policy,
                rng_state: Mutex::new(PluginRngState::default()),
            }),
        }
    }

    pub fn begin_invocation(
        &self,
        context: PluginServiceInvocationContext,
    ) -> Result<PluginCapabilityInvocation, PluginServiceError> {
        let operation_context = PluginServiceOperationContext {
            invocation: &context,
            clock: self.inner.clock.as_ref(),
            clock_origin: self.inner.clock_origin,
        };
        operation_context.check_active()?;
        let provider_result_receipts = context
            .provider_result_authority()
            .map(|authority| {
                usize::try_from(context.maximum_response_bytes())
                    .map_err(|_| PluginServiceError::ProviderResultReceiptAuthorityDenied)
                    .and_then(|maximum| {
                        authority
                            .begin_session(maximum)
                            .map_err(map_provider_materialization_error)
                    })
            })
            .transpose()?;
        Ok(PluginCapabilityInvocation {
            broker: self.clone(),
            context,
            rng_transactions: BTreeMap::new(),
            provider_result_receipts,
            operation_failed: Arc::new(AtomicBool::new(false)),
            terminal: false,
        })
    }
}

pub struct PluginCapabilityInvocation {
    broker: PluginCapabilityBroker,
    context: PluginServiceInvocationContext,
    rng_transactions: BTreeMap<PluginRngKey, RngTransaction>,
    provider_result_receipts: Option<ProviderResultReceiptSession>,
    operation_failed: Arc<AtomicBool>,
    terminal: bool,
}

struct ProviderCostNonceClaim<'a> {
    broker: &'a PluginCapabilityBrokerInner,
    nonce: ProviderCostNonce,
    expires_at: Instant,
    retained: bool,
}

impl ProviderCostNonceClaim<'_> {
    fn retain(mut self) {
        self.retained = true;
    }
}

impl Drop for ProviderCostNonceClaim<'_> {
    fn drop(&mut self) {
        if self.retained {
            return;
        }
        let mut consumed_nonces = self.broker.consumed_provider_cost_nonces.lock();
        if consumed_nonces.get(&self.nonce) == Some(&self.expires_at) {
            consumed_nonces.remove(&self.nonce);
        }
    }
}

impl PluginCapabilityBrokerInner {
    fn claim_provider_cost_nonce(
        &self,
        nonce: ProviderCostNonce,
        expires_at: Instant,
        now: Instant,
    ) -> Result<ProviderCostNonceClaim<'_>, PluginServiceError> {
        let mut consumed_nonces = self.consumed_provider_cost_nonces.lock();
        consumed_nonces.retain(|_, expiration| *expiration > now);
        if expires_at <= now {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        if consumed_nonces.contains_key(&nonce) {
            return Err(PluginServiceError::ProviderCostAcceptanceReused);
        }
        if consumed_nonces.len() >= MAX_CONSUMED_PROVIDER_COST_NONCES {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        consumed_nonces.insert(nonce, expires_at);
        Ok(ProviderCostNonceClaim {
            broker: self,
            nonce,
            expires_at,
            retained: false,
        })
    }
}

impl PluginCapabilityInvocation {
    pub fn context(&self) -> &PluginServiceInvocationContext {
        &self.context
    }

    pub fn read_asset(
        &self,
        namespace: AssetNamespace,
        asset_reference: &str,
    ) -> Result<Vec<u8>, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        self.require_capability(&Capability::Asset {
            namespace: namespace.locator_type().to_owned(),
            action: AssetOperation::Read,
        })?;
        let operation_context = self.operation_context();
        operation_context.check_active()?;
        let assets = self
            .broker
            .inner
            .assets
            .lock()
            .map_err(|_| PluginServiceError::AssetServiceUnavailable)?;
        let identity = assets
            .roots()
            .identity_from_reference(asset_reference)
            .map_err(map_asset_error)?;
        if identity.namespace != namespace {
            return Err(PluginServiceError::AssetNamespaceMismatch);
        }
        let bytes = assets
            .read_verified(
                &identity,
                self.context.authorization.capabilities(),
                self.context.cancellation(),
                self.context.maximum_response_bytes,
            )
            .map_err(map_asset_error)?;
        check_response_size(bytes.len(), self.context.maximum_response_bytes)?;
        operation_context.check_active()?;
        Ok(outcome.succeed(bytes))
    }

    pub fn load_model(&self, model_id: &str) -> Result<PluginModelHandle, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        validate_identity("model", model_id)?;
        self.require_capability(&Capability::ModelHandle {
            model_id: model_id.to_owned(),
        })?;
        self.require_capability(&Capability::Asset {
            namespace: AssetNamespace::Model.locator_type().to_owned(),
            action: AssetOperation::Read,
        })?;
        let operation_context = self.operation_context();
        operation_context.check_active()?;
        let assets = self
            .broker
            .inner
            .assets
            .lock()
            .map_err(|_| PluginServiceError::AssetServiceUnavailable)?;
        let identity = assets
            .roots()
            .identity_from_reference(model_id)
            .map_err(map_asset_error)?;
        if identity.namespace != AssetNamespace::Model {
            return Err(PluginServiceError::ModelNamespaceRequired);
        }
        let mut model_store = self.broker.inner.model_store.lock();
        let model = assets
            .load_model(
                &identity,
                &mut model_store,
                self.context.authorization.capabilities(),
                self.context.cancellation(),
            )
            .map_err(map_asset_error)?;
        operation_context.check_active()?;
        let model_format = model
            .documents()
            .first()
            .map(|document| model_format_name(&document.format))
            .ok_or(PluginServiceError::AssetOperationFailed {
                operation: "model_projection",
            })?;
        Ok(outcome.succeed(PluginModelHandle {
            model_id: model_id.to_owned(),
            model_identity: model.identity().to_owned(),
            model_format,
            accounting: model.accounting().clone(),
            model,
        }))
    }

    pub fn execute_provider_request(
        &self,
        provider: &str,
        endpoint: &str,
        secret_id: Option<&SecretId>,
        body: &[u8],
    ) -> Result<Vec<u8>, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        check_request_size(body.len())?;
        self.require_capability(&Capability::ProviderNetwork {
            provider: provider.to_owned(),
            endpoint: endpoint.to_owned(),
        })?;
        if let Some(secret_id) = secret_id {
            self.require_capability(&Capability::Secret {
                secret_id: secret_id.as_str().to_owned(),
            })?;
        }
        let operation_context = self.operation_context();
        operation_context.check_active()?;
        let authorized_request = self
            .broker
            .inner
            .provider_policy
            .authorize(
                &self.context.profile_id.0.to_string(),
                self.context.plugin_id(),
                provider,
                endpoint,
                secret_id,
            )
            .map_err(|_| PluginServiceError::ProviderPolicyDenied)?;
        let secret = secret_id
            .map(|secret_id| {
                let request = AuthorizedCredentialPresenceRequest {
                    profile_id: self.context.profile_id,
                    plugin_id: self.context.plugin_id().to_owned(),
                    secret_id: secret_id.clone(),
                };
                match self
                    .broker
                    .inner
                    .credential_presence_actuator
                    .read_for_provider(&request, &operation_context)
                {
                    Ok(secret) => secret.ok_or(PluginServiceError::CredentialUnavailable),
                    Err(error) => {
                        operation_context.check_active()?;
                        Err(PluginServiceError::ActuatorFailed {
                            service: "credential_read",
                            message: error.message,
                        })
                    }
                }
            })
            .transpose()?;
        let response = match self.broker.inner.provider_actuator.execute(
            &authorized_request,
            secret.as_ref(),
            body,
            &operation_context,
        ) {
            Ok(response) => response,
            Err(error) => {
                operation_context.check_active()?;
                return Err(PluginServiceError::ActuatorFailed {
                    service: "provider",
                    message: error.message,
                });
            }
        };
        check_response_size(response.len(), self.context.maximum_response_bytes)?;
        operation_context.check_active()?;
        Ok(outcome.succeed(response))
    }

    pub fn execute_provider_request_with_receipt(
        &mut self,
        provider: &str,
        endpoint: &str,
        secret_id: Option<&SecretId>,
        body: &[u8],
    ) -> Result<Vec<u8>, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        self.require_capability(&Capability::ProviderUpload {
            provider: provider.to_owned(),
            endpoint: endpoint.to_owned(),
        })?;
        let authority = self
            .context
            .provider_result_authority()
            .cloned()
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityRequired)?;
        let request_ordinal = self
            .provider_result_receipts
            .as_ref()
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityRequired)?
            .next_request_ordinal()
            .map_err(map_provider_materialization_error)?;
        let identity = ProviderInvocationIdentity::new(
            authority.principal_id(),
            self.context.profile_id().0.to_string(),
            self.context.prompt_id().0.to_string(),
            authority.prompt_sha256(),
            self.context.attempt_id().0.to_string(),
            self.context.node_id().0.clone(),
            request_ordinal,
            format!("{:x}", Sha256::digest(body)),
            self.context.plugin_id(),
            self.context.plugin_digest_sha256(),
            authority.provider_binding_sha256(),
            provider,
            endpoint,
        )
        .map_err(|_| PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        let cost_authority = self.context.provider_cost_authority().cloned();
        let response = if let Some(cost_authority) = cost_authority {
            self.require_capability(&Capability::ProviderCost {
                provider: provider.to_owned(),
                endpoint: endpoint.to_owned(),
            })?;
            let authorization = cost_authority
                .authority
                .as_ref()
                .ok_or(PluginServiceError::ProviderCostAcceptanceRequired)?
                .authorize(ProviderCostAuthorizationRequest {
                    identity: &identity,
                    price_badge: &cost_authority.price_badge,
                })?;
            self.execute_priced_provider_request(
                authority.provider_binding_sha256(),
                authority.prompt_sha256(),
                request_ordinal,
                provider,
                endpoint,
                secret_id,
                authorization.price_bound(),
                authorization.nonce(),
                Some(authorization.acceptance()),
                body,
            )?
        } else {
            self.execute_provider_request(provider, endpoint, secret_id, body)?
        };
        let issued_at = self.broker.inner.clock.utc_now();
        let lifetime_expiration = issued_at
            .checked_add(authority.receipt_lifetime())
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        let expires_at = lifetime_expiration.min(self.context.deadline());
        if expires_at <= issued_at {
            return Err(PluginServiceError::DeadlineExceeded);
        }
        let receipt = self
            .provider_result_receipts
            .as_mut()
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityRequired)?
            .issue(identity, response, issued_at, expires_at)
            .map_err(map_provider_materialization_error)?;
        check_response_size(receipt.len(), self.context.maximum_response_bytes())?;
        Ok(outcome.succeed(receipt))
    }

    fn check_provider_result_read_identity(
        &self,
        expected_identity: &ProviderInvocationIdentity,
    ) -> Result<(), PluginServiceError> {
        self.operation_context().check_active()?;
        self.require_capability(&Capability::ProviderNetwork {
            provider: expected_identity.provider().to_owned(),
            endpoint: expected_identity.endpoint().to_owned(),
        })?;
        self.require_capability(&Capability::ProviderUpload {
            provider: expected_identity.provider().to_owned(),
            endpoint: expected_identity.endpoint().to_owned(),
        })?;
        let authority = self
            .context
            .provider_result_authority()
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityRequired)?;
        if self.context.principal_id() != Some(expected_identity.principal_id())
            || self.context.profile_id().0.to_string() != expected_identity.profile_id()
            || self.context.prompt_id().0.to_string() != expected_identity.prompt_id()
            || self.context.attempt_id().0.to_string() != expected_identity.attempt_id()
            || self.context.node_id().0 != expected_identity.node_id()
            || self.context.plugin_id() != expected_identity.plugin_id()
            || self.context.plugin_digest_sha256() != expected_identity.plugin_digest_sha256()
            || authority.prompt_sha256() != expected_identity.prompt_sha256()
            || authority.provider_binding_sha256() != expected_identity.provider_binding_sha256()
        {
            return Err(PluginServiceError::ProviderResultReceiptAuthorityDenied);
        }
        Ok(())
    }

    pub fn is_provider_result_structured_transport_response(
        &self,
        receipt: &[u8],
        expected_identity: &ProviderInvocationIdentity,
        context: &sim_nodes::NativeNodeContext,
    ) -> Result<bool, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        self.check_provider_result_read_identity(expected_identity)?;
        let structured = self
            .provider_result_receipts
            .as_ref()
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityRequired)?
            .is_structured_transport_response(
                receipt,
                expected_identity,
                self.broker.inner.clock.utc_now(),
                context,
            )
            .map_err(map_provider_materialization_error)?;
        self.operation_context().check_active()?;
        context
            .cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        Ok(outcome.succeed(structured))
    }

    pub fn materialize_provider_downloaded_image(
        &self,
        receipt: &[u8],
        expected_identity: &ProviderInvocationIdentity,
        context: &sim_nodes::NativeNodeContext,
    ) -> Result<sim_nodes::NativeStoredPayload, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        self.check_provider_result_read_identity(expected_identity)?;
        let payload = self
            .provider_result_receipts
            .as_ref()
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityRequired)?
            .materialize_downloaded_image(
                receipt,
                expected_identity,
                self.broker.inner.clock.utc_now(),
                context,
            )
            .map_err(map_provider_materialization_error)?;
        self.operation_context().check_active()?;
        context
            .cancellation
            .check()
            .map_err(|_| PluginServiceError::Cancelled)?;
        Ok(outcome.succeed(payload))
    }

    pub fn resolve_provider_result_receipt_set(
        &mut self,
        receipt_set: &ProviderResultReceiptSet,
    ) -> Result<Vec<ResolvedProviderResult>, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        self.operation_context().check_active()?;
        let now = self.broker.inner.clock.utc_now();
        let resolved = self
            .provider_result_receipts
            .as_mut()
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityRequired)?
            .resolve_receipt_set(receipt_set, now)
            .map_err(map_provider_materialization_error)?;
        self.operation_context().check_active()?;
        Ok(outcome.succeed(resolved))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_priced_provider_request(
        &self,
        provider_binding_sha256: &str,
        prompt_sha256: &str,
        request_ordinal: u32,
        provider: &str,
        endpoint: &str,
        secret_id: Option<&SecretId>,
        price_bound: &ProviderPriceBound,
        nonce: ProviderCostNonce,
        acceptance: Option<&ProviderCostAcceptance>,
        body: &[u8],
    ) -> Result<Vec<u8>, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        check_request_size(body.len())?;
        self.require_capability(&Capability::ProviderUpload {
            provider: provider.to_owned(),
            endpoint: endpoint.to_owned(),
        })?;
        self.require_capability(&Capability::ProviderCost {
            provider: provider.to_owned(),
            endpoint: endpoint.to_owned(),
        })?;
        self.require_capability(&Capability::ProviderNetwork {
            provider: provider.to_owned(),
            endpoint: endpoint.to_owned(),
        })?;
        if let Some(secret_id) = secret_id {
            self.require_capability(&Capability::Secret {
                secret_id: secret_id.as_str().to_owned(),
            })?;
        }
        let operation_context = self.operation_context();
        operation_context.check_active()?;
        let acceptance = acceptance.ok_or(PluginServiceError::ProviderCostAcceptanceRequired)?;
        let verifier = self
            .broker
            .inner
            .provider_cost_acceptance_verifier
            .as_ref()
            .ok_or(PluginServiceError::ProviderCostAcceptanceRequired)?;
        let principal_id = self
            .context
            .principal_id()
            .ok_or(PluginServiceError::ProviderCostPrincipalRequired)?;
        let request_sha256 = format!("{:x}", Sha256::digest(body));
        let expected_scope = ProviderCostAcceptanceScope::new(
            principal_id,
            self.context.profile_id().0.to_string(),
            self.context.prompt_id().0.to_string(),
            prompt_sha256,
            self.context.attempt_id().0.to_string(),
            self.context.node_id().0.clone(),
            request_ordinal,
            request_sha256,
            self.context.plugin_id(),
            self.context.plugin_digest_sha256(),
            provider_binding_sha256,
            provider,
            endpoint,
            price_bound.clone(),
        )
        .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
        let verified_acceptance = verifier
            .verify(
                acceptance,
                &expected_scope,
                self.broker.inner.clock.utc_now(),
            )
            .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
        if verified_acceptance.nonce() != nonce {
            return Err(PluginServiceError::ProviderCostAcceptanceDenied);
        }
        let authorized_request = self
            .broker
            .inner
            .provider_policy
            .authorize(
                &self.context.profile_id.0.to_string(),
                self.context.plugin_id(),
                provider,
                endpoint,
                secret_id,
            )
            .map_err(|_| PluginServiceError::ProviderPolicyDenied)?
            .with_idempotency_key_sha256(expected_scope.identity().idempotency_key_sha256())
            .map_err(|_| PluginServiceError::ProviderCostAcceptanceDenied)?;
        let cost_nonce_claim = self.broker.inner.claim_provider_cost_nonce(
            nonce,
            verified_acceptance.expires_at(),
            self.broker.inner.clock.utc_now(),
        )?;
        let secret = secret_id
            .map(|secret_id| {
                let request = AuthorizedCredentialPresenceRequest {
                    profile_id: self.context.profile_id,
                    plugin_id: self.context.plugin_id().to_owned(),
                    secret_id: secret_id.clone(),
                };
                match self
                    .broker
                    .inner
                    .credential_presence_actuator
                    .read_for_provider(&request, &operation_context)
                {
                    Ok(secret) => secret.ok_or(PluginServiceError::CredentialUnavailable),
                    Err(error) => {
                        operation_context.check_active()?;
                        Err(PluginServiceError::ActuatorFailed {
                            service: "credential_read",
                            message: error.message,
                        })
                    }
                }
            })
            .transpose()?;
        operation_context.check_active()?;
        cost_nonce_claim.retain();
        let response = match self.broker.inner.provider_actuator.execute(
            &authorized_request,
            secret.as_ref(),
            body,
            &operation_context,
        ) {
            Ok(response) => response,
            Err(error) => {
                operation_context.check_active()?;
                return Err(PluginServiceError::ActuatorFailed {
                    service: "provider",
                    message: error.message,
                });
            }
        };
        check_response_size(response.len(), self.context.maximum_response_bytes)?;
        operation_context.check_active()?;
        Ok(outcome.succeed(response))
    }

    pub fn credential_is_present(&self, secret_id: &SecretId) -> Result<bool, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        self.require_capability(&Capability::Secret {
            secret_id: secret_id.as_str().to_owned(),
        })?;
        let operation_context = self.operation_context();
        operation_context.check_active()?;
        let request = AuthorizedCredentialPresenceRequest {
            profile_id: self.context.profile_id,
            plugin_id: self.context.plugin_id().to_owned(),
            secret_id: secret_id.clone(),
        };
        let present = match self
            .broker
            .inner
            .credential_presence_actuator
            .is_present(&request, &operation_context)
        {
            Ok(present) => present,
            Err(error) => {
                operation_context.check_active()?;
                return Err(PluginServiceError::ActuatorFailed {
                    service: "credential_presence",
                    message: error.message,
                });
            }
        };
        operation_context.check_active()?;
        Ok(outcome.succeed(present))
    }

    pub fn monotonic_milliseconds(&self, clock_id: &str) -> Result<u64, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        validate_identity("clock", clock_id)?;
        self.require_capability(&Capability::Clock {
            clock_id: clock_id.to_owned(),
        })?;
        let operation_context = self.operation_context();
        operation_context.check_active()?;
        let elapsed = self
            .broker
            .inner
            .clock
            .utc_now()
            .checked_duration_since(self.broker.inner.clock_origin)
            .ok_or(PluginServiceError::ClockMovedBackwards)?;
        let milliseconds =
            u64::try_from(elapsed.as_millis()).map_err(|_| PluginServiceError::ClockOverflow)?;
        Ok(outcome.succeed(milliseconds))
    }

    pub fn random_bytes(
        &mut self,
        stream_id: &str,
        length: usize,
    ) -> Result<Vec<u8>, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        validate_identity("randomness stream", stream_id)?;
        self.require_capability(&Capability::Randomness {
            stream_id: stream_id.to_owned(),
        })?;
        check_response_size(length, self.context.maximum_response_bytes)?;
        self.operation_context().check_active()?;
        let key = self.rng_key(stream_id);
        if !self.rng_transactions.contains_key(&key) {
            let checkpoint = {
                let mut state = self.broker.inner.rng_state.lock();
                if !state.active_streams.insert(key.clone()) {
                    return Err(PluginServiceError::RandomnessStreamBusy);
                }
                state.checkpoints.get(&key).cloned()
            };
            let transaction = match self.rng_stream(stream_id).and_then(|stream| {
                stream
                    .begin(checkpoint)
                    .map_err(|_| PluginServiceError::RandomnessFailed)
            }) {
                Ok(transaction) => transaction,
                Err(error) => {
                    self.broker
                        .inner
                        .rng_state
                        .lock()
                        .active_streams
                        .remove(&key);
                    return Err(error);
                }
            };
            self.rng_transactions.insert(key.clone(), transaction);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| PluginServiceError::ResponseAllocationFailed)?;
        bytes.resize(length, 0);
        for chunk in bytes.chunks_mut(RNG_DEADLINE_CHECK_CHUNK_BYTES) {
            self.rng_transactions
                .get_mut(&key)
                .ok_or(PluginServiceError::RandomnessFailed)?
                .fill_bytes(chunk, self.context.cancellation())
                .map_err(|_| {
                    if self.context.cancellation.is_cancelled() {
                        PluginServiceError::Cancelled
                    } else {
                        PluginServiceError::RandomnessFailed
                    }
                })?;
            self.operation_context().check_active()?;
        }
        Ok(outcome.succeed(bytes))
    }

    pub fn sanitize_log(&self, level: &str, message: &str) -> Result<String, PluginServiceError> {
        self.check_terminal()?;
        let outcome = OperationOutcomeGuard::new(self.operation_failed.clone());
        validate_identity("log level", level)?;
        check_request_size(message.len())?;
        self.require_capability(&Capability::SanitizedLog {
            level: level.to_owned(),
        })?;
        let operation_context = self.operation_context();
        operation_context.check_active()?;
        let redactions = self
            .context
            .authorization
            .capabilities()
            .capabilities()
            .iter()
            .filter_map(|capability| match capability {
                Capability::Secret { secret_id } => Some(secret_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut sanitized: String = message
            .chars()
            .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
            .take(8_192)
            .collect();
        for secret_id in redactions {
            sanitized = sanitized.replace(secret_id, "[REDACTED]");
        }
        check_response_size(sanitized.len(), self.context.maximum_response_bytes)?;
        operation_context.check_active()?;
        Ok(outcome.succeed(sanitized))
    }

    pub fn handle_wire_request(
        &mut self,
        request: PluginServiceWireRequest,
    ) -> PluginServiceWireResponse {
        let response = match request {
            PluginServiceWireRequest::ReadAsset {
                namespace,
                asset_reference,
            } => AssetNamespace::from_locator_type(&namespace)
                .map_err(|_| PluginServiceError::InvalidWirePayload)
                .and_then(|namespace| self.read_asset(namespace, &asset_reference))
                .map(PluginServiceWireResponse::Bytes),
            PluginServiceWireRequest::ExecuteProvider {
                provider,
                endpoint,
                secret_id,
                body,
            } => {
                let provider_result_receipts = self.provider_result_receipts.is_some();
                secret_id
                    .map(SecretId::new)
                    .transpose()
                    .map_err(|_| PluginServiceError::InvalidWirePayload)
                    .and_then(|secret_id| {
                        if provider_result_receipts {
                            self.execute_provider_request_with_receipt(
                                &provider,
                                &endpoint,
                                secret_id.as_ref(),
                                &body,
                            )
                        } else {
                            self.execute_provider_request(
                                &provider,
                                &endpoint,
                                secret_id.as_ref(),
                                &body,
                            )
                        }
                    })
                    .map(PluginServiceWireResponse::Bytes)
            }
            PluginServiceWireRequest::CredentialIsPresent { secret_id } => SecretId::new(secret_id)
                .map_err(|_| PluginServiceError::InvalidWirePayload)
                .and_then(|secret_id| self.credential_is_present(&secret_id))
                .map(PluginServiceWireResponse::Boolean),
            PluginServiceWireRequest::MonotonicMilliseconds { clock_id } => self
                .monotonic_milliseconds(&clock_id)
                .map(PluginServiceWireResponse::TimestampMilliseconds),
            PluginServiceWireRequest::RandomBytes { stream_id, length } => usize::try_from(length)
                .map_err(|_| PluginServiceError::InvalidWirePayload)
                .and_then(|length| self.random_bytes(&stream_id, length))
                .map(PluginServiceWireResponse::Bytes),
            PluginServiceWireRequest::LoadModel { model_id } => {
                self.load_model(&model_id)
                    .map(|model| PluginServiceWireResponse::Model {
                        identifier: model.model_id().to_owned(),
                        format: model.model_format().to_owned(),
                        digest_sha256: model.model_identity().to_owned(),
                    })
            }
            PluginServiceWireRequest::SanitizeLog { level, message } => self
                .sanitize_log(&level, &message)
                .map(PluginServiceWireResponse::SanitizedLog),
        };
        if response.is_err() {
            self.operation_failed.store(true, Ordering::Release);
        }
        response.unwrap_or_else(|error| PluginServiceWireResponse::Failure(error.into()))
    }

    pub fn finish(mut self) -> Result<(), PluginServiceError> {
        self.check_terminal()?;
        if self.operation_failed.load(Ordering::Acquire) {
            return Err(PluginServiceError::InvocationFailed);
        }
        self.operation_context().check_active()?;
        if let Some(receipt_session) = self.provider_result_receipts.take() {
            receipt_session
                .finish()
                .map_err(map_provider_materialization_error)?;
        }
        let committed = std::mem::take(&mut self.rng_transactions)
            .into_iter()
            .map(|(key, transaction)| (key, transaction.commit()))
            .collect::<Vec<_>>();
        let mut state = self.broker.inner.rng_state.lock();
        for (key, checkpoint) in committed {
            state.checkpoints.insert(key.clone(), checkpoint);
            state.active_streams.remove(&key);
        }
        drop(state);
        self.terminal = true;
        Ok(())
    }

    pub fn abort(mut self) {
        self.abort_internal();
    }

    fn operation_context(&self) -> PluginServiceOperationContext<'_> {
        PluginServiceOperationContext {
            invocation: &self.context,
            clock: self.broker.inner.clock.as_ref(),
            clock_origin: self.broker.inner.clock_origin,
        }
    }

    fn check_terminal(&self) -> Result<(), PluginServiceError> {
        if self.terminal {
            Err(PluginServiceError::InvocationFinished)
        } else {
            Ok(())
        }
    }

    fn require_capability(&self, capability: &Capability) -> Result<(), PluginServiceError> {
        self.context
            .authorization
            .capabilities()
            .require(capability)
            .map_err(|_| PluginServiceError::CapabilityDenied(capability.clone()))
    }

    fn rng_key(&self, stream_id: &str) -> PluginRngKey {
        PluginRngKey {
            profile_id: self.context.profile_id.0.to_string(),
            prompt_id: self.context.prompt_id.0.to_string(),
            attempt_id: self.context.attempt_id.0.to_string(),
            node_id: self.context.node_id.clone(),
            plugin_id: self.context.plugin_id().to_owned(),
            plugin_digest_sha256: self.context.plugin_digest_sha256().to_owned(),
            stream_id: stream_id.to_owned(),
        }
    }

    fn rng_stream(&self, stream_id: &str) -> Result<RngStream, PluginServiceError> {
        let address = RngStreamAddress::new(
            self.context.prompt_id.0.to_string(),
            self.context.attempt_id.0.to_string(),
            self.context.node_id.0.clone(),
            0,
            format!("plugin:{}:{stream_id}", self.context.plugin_id()),
            0,
            0,
            RetryRngPolicy::Replay,
        )
        .map_err(|_| PluginServiceError::RandomnessFailed)?;
        RngStream::new(
            self.broker.inner.rng_policy.profile,
            self.broker.inner.rng_policy.algorithm,
            self.broker.inner.rng_policy.seed,
            address,
        )
        .map_err(|_| PluginServiceError::RandomnessFailed)
    }

    fn abort_internal(&mut self) {
        if let Some(receipt_session) = self.provider_result_receipts.take() {
            receipt_session.abort();
        }
        let transactions = std::mem::take(&mut self.rng_transactions);
        let mut state = self.broker.inner.rng_state.lock();
        for (key, transaction) in transactions {
            transaction.abort();
            state.active_streams.remove(&key);
        }
        drop(state);
        self.terminal = true;
    }
}

struct OperationOutcomeGuard {
    operation_failed: Arc<AtomicBool>,
    succeeded: bool,
}

impl OperationOutcomeGuard {
    fn new(operation_failed: Arc<AtomicBool>) -> Self {
        Self {
            operation_failed,
            succeeded: false,
        }
    }

    fn succeed<T>(mut self, value: T) -> T {
        self.succeeded = true;
        value
    }
}

impl Drop for OperationOutcomeGuard {
    fn drop(&mut self) {
        if !self.succeeded {
            self.operation_failed.store(true, Ordering::Release);
        }
    }
}

impl Drop for PluginCapabilityInvocation {
    fn drop(&mut self) {
        if !self.terminal {
            self.abort_internal();
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PluginServiceError {
    #[error("plugin service context belongs to a different runtime profile")]
    ProfileMismatch,
    #[error("plugin authorization subject does not match the signed plugin identity")]
    AuthorizationIdentityMismatch,
    #[error("plugin service {kind} identity is invalid")]
    InvalidIdentity { kind: &'static str },
    #[error("plugin response limit must be between 1 and {maximum} bytes")]
    InvalidResponseLimit { maximum: u64 },
    #[error("plugin capability denied: {0:?}")]
    CapabilityDenied(Capability),
    #[error("plugin invocation was cancelled")]
    Cancelled,
    #[error("plugin invocation deadline was exceeded")]
    DeadlineExceeded,
    #[error("the injected monotonic clock moved backwards")]
    ClockMovedBackwards,
    #[error("the injected monotonic clock overflowed milliseconds")]
    ClockOverflow,
    #[error("plugin request exceeds the {maximum}-byte limit")]
    RequestTooLarge { maximum: u64 },
    #[error("plugin response exceeds the {maximum}-byte invocation limit")]
    ResponseTooLarge { maximum: u64 },
    #[error("plugin response allocation failed")]
    ResponseAllocationFailed,
    #[error("the canonical asset service is unavailable")]
    AssetServiceUnavailable,
    #[error("asset reference namespace does not match the authorized namespace")]
    AssetNamespaceMismatch,
    #[error("model loading requires a canonical model asset reference")]
    ModelNamespaceRequired,
    #[error("canonical asset operation failed: {operation}")]
    AssetOperationFailed { operation: &'static str },
    #[error("provider policy denied the request")]
    ProviderPolicyDenied,
    #[error("priced provider request requires a host-issued cost acceptance")]
    ProviderCostAcceptanceRequired,
    #[error("priced provider request requires a host-bound principal")]
    ProviderCostPrincipalRequired,
    #[error("provider cost acceptance is invalid, expired, or bound to another request")]
    ProviderCostAcceptanceDenied,
    #[error("provider cost acceptance nonce was already consumed")]
    ProviderCostAcceptanceReused,
    #[error("provider result receipts require host-owned invocation authority")]
    ProviderResultReceiptAuthorityRequired,
    #[error("provider result receipt authority is invalid or belongs to another invocation")]
    ProviderResultReceiptAuthorityDenied,
    #[error("provider runtime stream session is unavailable, duplicated, or over quota")]
    ProviderSessionUnavailable,
    #[error("provider runtime authority is not bound to the verified activation")]
    ProviderRuntimeAuthorityDenied,
    #[error("provider runtime request belongs to a foreign host session")]
    ProviderRuntimeForeignSession,
    #[error("provider runtime request belongs to a stale host session generation")]
    ProviderRuntimeStaleSession,
    #[error("provider runtime request belongs to a foreign invocation")]
    ProviderRuntimeForeignInvocation,
    #[error("provider runtime request belongs to a stale invocation generation")]
    ProviderRuntimeStaleInvocation,
    #[error("provider streaming contract rejected the operation: {0}")]
    ProviderStreamingContract(ProviderStreamingContractError),
    #[error("the authorized provider credential is unavailable")]
    CredentialUnavailable,
    #[error("{service} actuator failed: {message}")]
    ActuatorFailed {
        service: &'static str,
        message: String,
    },
    #[error("randomness stream is already in use by another invocation")]
    RandomnessStreamBusy,
    #[error("canonical randomness operation failed")]
    RandomnessFailed,
    #[error("plugin invocation has already finished")]
    InvocationFinished,
    #[error("plugin invocation cannot commit after a failed capability operation")]
    InvocationFailed,
    #[error("plugin capability wire payload is malformed")]
    InvalidWirePayload,
}

impl From<PluginServiceError> for PluginServiceWireFailure {
    fn from(error: PluginServiceError) -> Self {
        match error {
            PluginServiceError::CapabilityDenied(_) => Self::CapabilityDenied,
            PluginServiceError::Cancelled => Self::Cancelled,
            PluginServiceError::DeadlineExceeded => Self::DeadlineExceeded,
            PluginServiceError::ResponseTooLarge { .. }
            | PluginServiceError::ResponseAllocationFailed => Self::ResponseTooLarge,
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
            | PluginServiceError::ProviderStreamingContract(_) => Self::ProviderDenied,
            PluginServiceError::ActuatorFailed { .. } => Self::ActuatorFailed,
            PluginServiceError::RandomnessStreamBusy | PluginServiceError::RandomnessFailed => {
                Self::RandomnessFailed
            }
            PluginServiceError::InvocationFailed | PluginServiceError::InvocationFinished => {
                Self::InvocationFailed
            }
            PluginServiceError::AssetServiceUnavailable
            | PluginServiceError::CredentialUnavailable
            | PluginServiceError::AssetOperationFailed { .. } => Self::ServiceUnavailable,
            PluginServiceError::ProfileMismatch
            | PluginServiceError::AuthorizationIdentityMismatch
            | PluginServiceError::InvalidIdentity { .. }
            | PluginServiceError::InvalidResponseLimit { .. }
            | PluginServiceError::ClockMovedBackwards
            | PluginServiceError::ClockOverflow
            | PluginServiceError::RequestTooLarge { .. }
            | PluginServiceError::AssetNamespaceMismatch
            | PluginServiceError::ModelNamespaceRequired
            | PluginServiceError::InvalidWirePayload => Self::InvalidRequest,
        }
    }
}

fn validate_identity(kind: &'static str, value: &str) -> Result<(), PluginServiceError> {
    if value.is_empty()
        || value != value.trim()
        || value.len() > MAX_PLUGIN_SERVICE_IDENTITY_BYTES
        || value.chars().any(char::is_control)
    {
        Err(PluginServiceError::InvalidIdentity { kind })
    } else {
        Ok(())
    }
}

fn check_request_size(length: usize) -> Result<(), PluginServiceError> {
    if u64::try_from(length).unwrap_or(u64::MAX) > MAX_PLUGIN_SERVICE_REQUEST_BYTES {
        Err(PluginServiceError::RequestTooLarge {
            maximum: MAX_PLUGIN_SERVICE_REQUEST_BYTES,
        })
    } else {
        Ok(())
    }
}

fn check_response_size(length: usize, maximum: u64) -> Result<(), PluginServiceError> {
    if u64::try_from(length).unwrap_or(u64::MAX) > maximum {
        Err(PluginServiceError::ResponseTooLarge { maximum })
    } else {
        Ok(())
    }
}

fn map_asset_error(error: AssetError) -> PluginServiceError {
    match error {
        AssetError::Cancelled
        | AssetError::Model(ModelStoreError::Cancelled)
        | AssetError::Model(ModelStoreError::Format(ModelFormatError::Cancelled)) => {
            PluginServiceError::Cancelled
        }
        AssetError::PermissionDenied { namespace, action } => {
            PluginServiceError::CapabilityDenied(Capability::Asset {
                namespace: namespace.locator_type().to_owned(),
                action: action.into(),
            })
        }
        AssetError::TooLarge { limit, .. } => {
            PluginServiceError::ResponseTooLarge { maximum: limit }
        }
        AssetError::ModelNamespaceRequired(_) => PluginServiceError::ModelNamespaceRequired,
        AssetError::InvalidReference(_) => PluginServiceError::AssetOperationFailed {
            operation: "reference_validation",
        },
        _ => PluginServiceError::AssetOperationFailed {
            operation: "canonical_asset_service",
        },
    }
}

fn map_provider_materialization_error(error: ProviderMaterializationError) -> PluginServiceError {
    match error {
        ProviderMaterializationError::Cancelled => PluginServiceError::Cancelled,
        ProviderMaterializationError::ImageDecode(message) => PluginServiceError::ActuatorFailed {
            service: "provider_image_decode",
            message: sanitize_actuator_message(&message),
        },
        ProviderMaterializationError::ImageAllocation(message) => {
            PluginServiceError::ActuatorFailed {
                service: "provider_image_allocation",
                message: sanitize_actuator_message(&message),
            }
        }
        ProviderMaterializationError::ResponseTooLarge => PluginServiceError::ResponseTooLarge {
            maximum: MAX_PLUGIN_SERVICE_RESPONSE_BYTES,
        },
        ProviderMaterializationError::ReceiptSessionFinished
        | ProviderMaterializationError::RequestOrdinalOutOfOrder
        | ProviderMaterializationError::ReceiptRejected
        | ProviderMaterializationError::UnknownReceipt
        | ProviderMaterializationError::ReceiptOutOfOrder
        | ProviderMaterializationError::UnresolvedReceipts
        | ProviderMaterializationError::InvalidReceiptAuthority
        | ProviderMaterializationError::InvalidTransportProjection
        | ProviderMaterializationError::InvalidNativePayload
        | ProviderMaterializationError::UnsupportedTransportSchema
        | ProviderMaterializationError::UnsupportedMaterializerSchema => {
            PluginServiceError::ProviderResultReceiptAuthorityDenied
        }
    }
}

fn sanitize_actuator_message(message: &str) -> String {
    let mut sanitized = String::new();
    for character in message.chars() {
        if sanitized.len().saturating_add(character.len_utf8()) > MAX_PLUGIN_SERVICE_IDENTITY_BYTES
        {
            break;
        }
        if character.is_control() {
            sanitized.push(' ');
        } else {
            sanitized.push(character);
        }
    }
    if sanitized.trim().is_empty() {
        "operation failed".to_owned()
    } else {
        sanitized
    }
}

fn model_format_name(format: &ModelFormat) -> &'static str {
    match format {
        ModelFormat::Safetensors => "safetensors",
        ModelFormat::PytorchArchive => "pytorch-archive",
        ModelFormat::Gguf => "gguf",
        ModelFormat::JsonConfig => "json-config",
        ModelFormat::JsonTokenizer => "json-tokenizer",
        ModelFormat::YamlConfig => "yaml-config",
        ModelFormat::SentencePiece => "sentencepiece",
        ModelFormat::Tiktoken => "tiktoken",
    }
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        fs,
        sync::{
            Arc, Barrier,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use sim_model::ParserLimits;
    use sim_plugin_sdk::{
        ApiRequirement, ApiVersion, CachePolicy, CapabilityKind, CapabilityQuota,
        CapabilityRequest, DeterminismPolicy, ED25519_SIGNATURE_BYTES, EffectPolicy,
        ManifestProvenance, ManifestSignature, PLUGIN_SIGNATURE_ALGORITHM,
        PROVIDER_BINDING_SCHEMA_VERSION, PluginManifest, PluginNode, PluginSigningKey,
        ProviderBindingClaim, ProviderBindingSet, ProviderHttpMethodV2,
        ProviderResponseFrameEventV2, ProviderResponseFrameV2, ProviderResponseHeadV2,
        ProviderStreamTerminalV2, ProviderStreamingContractV2,
    };
    use sim_types::{
        WorkerComponentContent, WorkerComponentDescriptor, WorkerProviderInvocationContext,
        WorkerRegistryDeploymentBegin, WorkerRegistryDeploymentChunk, WorkerRegistryGeneration,
        WorkerSha256Digest,
    };
    use tempfile::TempDir;
    use uuid::Uuid;

    use crate::{
        AssetRoots, AssetService, CapabilitySet, PermissionGrant, PermissionPolicy,
        PluginTrustPolicy, PluginVerificationKey, ProviderCostAcceptanceIssuer, ProviderEndpoint,
        ProviderMode, ProviderResultReceiptIssuer, TrustError,
    };

    use super::*;

    const SIGNING_KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";
    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const PROMPT_SHA256: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const REQUEST_ORDINAL: u32 = 7;
    const REQUEST_BODY: &[u8] = b"request";
    const PROFILE_UUID: Uuid = Uuid::from_u128(1);
    const PROMPT_UUID: Uuid = Uuid::from_u128(2);
    const ATTEMPT_UUID: Uuid = Uuid::from_u128(3);
    const PROVIDER: &str = "fixture-provider";
    const ENDPOINT: &str = "https://provider.invalid/v1";
    const SECRET: &str = "fixture-secret";

    struct TestClock {
        now: Mutex<Instant>,
    }

    impl TestClock {
        fn new(now: Instant) -> Self {
            Self {
                now: Mutex::new(now),
            }
        }

        fn now(&self) -> Instant {
            *self.now.lock()
        }

        fn advance(&self, duration: Duration) {
            *self.now.lock() += duration;
        }
    }

    impl SystemClock for TestClock {
        fn utc_now(&self) -> Instant {
            self.now()
        }
    }

    #[derive(Default)]
    struct TestProviderActuator {
        calls: AtomicUsize,
        response: Mutex<Vec<u8>>,
        cancel_during_call: AtomicBool,
        fail_call: AtomicBool,
        last_authorized_request: Mutex<Option<(String, String, Option<String>)>>,
        last_idempotency_key: Mutex<Option<String>>,
        last_secret: Mutex<Option<Vec<u8>>>,
    }

    impl TestProviderActuator {
        fn set_response(&self, response: Vec<u8>) {
            *self.response.lock() = response;
        }

        fn set_cancel_during_call(&self, value: bool) {
            self.cancel_during_call.store(value, Ordering::Release);
        }

        fn set_fail_call(&self, value: bool) {
            self.fail_call.store(value, Ordering::Release);
        }
    }

    impl ProviderRequestActuator for TestProviderActuator {
        fn execute(
            &self,
            request: &AuthorizedProviderRequest,
            secret: Option<&SecretValue>,
            _body: &[u8],
            context: &PluginServiceOperationContext<'_>,
        ) -> Result<Vec<u8>, PluginServiceActuatorError> {
            context
                .check_active()
                .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
            self.calls.fetch_add(1, Ordering::AcqRel);
            *self.last_authorized_request.lock() = Some((
                request.provider().to_owned(),
                request.endpoint().to_owned(),
                request
                    .secret_id()
                    .map(|secret_id| secret_id.as_str().to_owned()),
            ));
            *self.last_idempotency_key.lock() = request.idempotency_key_sha256().map(str::to_owned);
            *self.last_secret.lock() = secret.map(|secret| secret.expose_to(<[u8]>::to_vec));
            if self.cancel_during_call.load(Ordering::Acquire) {
                context.cancellation().cancel();
            }
            if self.fail_call.load(Ordering::Acquire) {
                return Err(PluginServiceActuatorError::new("injected provider failure"));
            }
            Ok(self.response.lock().clone())
        }
    }

    #[derive(Default)]
    struct TestCredentialPresenceActuator {
        calls: AtomicUsize,
        present: AtomicBool,
        last_secret_id: Mutex<Option<String>>,
    }

    impl CredentialPresenceActuator for TestCredentialPresenceActuator {
        fn is_present(
            &self,
            request: &AuthorizedCredentialPresenceRequest,
            context: &PluginServiceOperationContext<'_>,
        ) -> Result<bool, PluginServiceActuatorError> {
            context
                .check_active()
                .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
            self.calls.fetch_add(1, Ordering::AcqRel);
            *self.last_secret_id.lock() = Some(request.secret_id().as_str().to_owned());
            Ok(self.present.load(Ordering::Acquire))
        }

        fn read_for_provider(
            &self,
            request: &AuthorizedCredentialPresenceRequest,
            context: &PluginServiceOperationContext<'_>,
        ) -> Result<Option<SecretValue>, PluginServiceActuatorError> {
            context
                .check_active()
                .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
            self.calls.fetch_add(1, Ordering::AcqRel);
            *self.last_secret_id.lock() = Some(request.secret_id().as_str().to_owned());
            Ok(self
                .present
                .load(Ordering::Acquire)
                .then(|| SecretValue::new(b"fixture-secret-value".to_vec())))
        }
    }

    fn capability(kind: CapabilityKind, scope: &str) -> CapabilityRequest {
        CapabilityRequest {
            kind,
            scope: scope.to_owned(),
            quota: CapabilityQuota {
                maximum_operations: 64,
                maximum_request_bytes: MAX_PLUGIN_SERVICE_REQUEST_BYTES,
                maximum_response_bytes: MAX_PLUGIN_SERVICE_RESPONSE_BYTES,
                maximum_total_bytes: MAX_PLUGIN_SERVICE_RESPONSE_BYTES,
                maximum_handles: 16,
                timeout_milliseconds: 60_000,
            },
        }
    }

    fn provider_capabilities(include_cost: bool) -> Vec<CapabilityRequest> {
        let scope = format!("{PROVIDER}|{ENDPOINT}");
        let mut capabilities = vec![
            capability(CapabilityKind::NetworkProvider, &scope),
            capability(CapabilityKind::ProviderUpload, &scope),
        ];
        if include_cost {
            capabilities.push(capability(CapabilityKind::ProviderCost, &scope));
        }
        capabilities
    }

    fn authorization(
        requested_capabilities: Vec<CapabilityRequest>,
    ) -> Result<PluginAuthorization, Box<dyn Error>> {
        let signing_key = PluginSigningKey::new("fixture.key", SIGNING_KEY)?;
        let trust = PluginTrustPolicy::new([PluginVerificationKey::new(
            "fixture.key",
            signing_key.verification_key_bytes()?,
        )?])?;
        let mut manifest = PluginManifest {
            schema_version: 1,
            identifier: "plugin.fixture".to_owned(),
            plugin_version: ApiVersion::new(1, 0, 0),
            api: ApiRequirement {
                major: 1,
                minimum_minor: 0,
                maximum_minor: 0,
                required_features: Vec::new(),
            },
            digest_sha256: DIGEST.to_owned(),
            signature: ManifestSignature {
                algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
                key_id: "fixture.key".to_owned(),
                value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
            },
            provenance: ManifestProvenance {
                source: "fixture://plugin.fixture".to_owned(),
                publisher: "fixture publisher".to_owned(),
                registry: Some("fixture://registry".to_owned()),
            },
            provider_binding: None,
            nodes: vec![PluginNode {
                id: "node.fixture".to_owned(),
                version: ApiVersion::new(1, 0, 0),
                display_name: "Fixture".to_owned(),
                category: "tests".to_owned(),
                ports: Vec::new(),
                determinism: DeterminismPolicy::Deterministic,
                cache: CachePolicy::InputIdentity,
                effects: EffectPolicy::Pure,
            }],
            capabilities: requested_capabilities.clone(),
            ui: Vec::new(),
            routes: Vec::new(),
            legacy_mappings: Vec::new(),
        };
        manifest.signature.value = signing_key.sign_manifest(&manifest)?;
        let domain_capabilities = requested_capabilities
            .iter()
            .map(Capability::from_plugin_request)
            .collect::<Result<Vec<_>, _>>()?;
        let permissions = PermissionPolicy::new(
            PROFILE_UUID.to_string(),
            [PermissionGrant::new(
                PROFILE_UUID.to_string(),
                "plugin.fixture",
                CapabilitySet::new(domain_capabilities),
                "test-profile-settings",
            )?],
        )?;
        Ok(trust.authorize_manifest(&manifest, &permissions)?)
    }

    fn asset_service() -> Result<(TempDir, SharedAssetService), Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        let input_root = directory.path().join("input");
        let output_root = directory.path().join("output");
        let temporary_root = directory.path().join("temporary");
        let model_root = directory.path().join("model");
        let plugin_root = directory.path().join("plugin");
        for root in [
            &input_root,
            &output_root,
            &temporary_root,
            &model_root,
            &plugin_root,
        ] {
            fs::create_dir_all(root)?;
        }
        fs::write(input_root.join("fixture.bin"), b"canonical asset bytes")?;
        fs::write(
            model_root.join("fixture.json"),
            br#"{"model_type":"fixture","hidden_size":4}"#,
        )?;
        let roots = AssetRoots::new(
            PROFILE_UUID.to_string(),
            [
                (AssetNamespace::Input, input_root),
                (AssetNamespace::Output, output_root),
                (AssetNamespace::Temporary, temporary_root),
                (AssetNamespace::Model, model_root),
                (AssetNamespace::Plugin, plugin_root),
            ],
        )?;
        let mut service = AssetService::open(roots)?;
        let setup_capabilities = CapabilitySet::new(
            [
                AssetNamespace::Input,
                AssetNamespace::Output,
                AssetNamespace::Temporary,
                AssetNamespace::Model,
                AssetNamespace::Plugin,
            ]
            .into_iter()
            .map(|namespace| Capability::Asset {
                namespace: namespace.locator_type().to_owned(),
                action: AssetOperation::Read,
            }),
        );
        let setup_policy = PermissionPolicy::new(
            PROFILE_UUID.to_string(),
            [PermissionGrant::new(
                PROFILE_UUID.to_string(),
                "test.asset-indexer",
                setup_capabilities.clone(),
                "test-fixture-setup",
            )?],
        )?;
        let setup_authorization =
            setup_policy.authorize("test.asset-indexer", &setup_capabilities)?;
        service.scan(&setup_authorization, &CancellationToken::default())?;
        let service = Arc::new(std::sync::Mutex::new(service));
        Ok((directory, service))
    }

    type BrokerFixture = (
        TempDir,
        PluginCapabilityBroker,
        Arc<TestProviderActuator>,
        Arc<TestCredentialPresenceActuator>,
    );

    fn broker(
        authorization: &PluginAuthorization,
        clock: Arc<TestClock>,
    ) -> Result<BrokerFixture, Box<dyn Error>> {
        broker_with_cost_acceptance(authorization, clock, None)
    }

    fn broker_with_cost_acceptance(
        authorization: &PluginAuthorization,
        clock: Arc<TestClock>,
        verifier: Option<ProviderCostAcceptanceVerifier>,
    ) -> Result<BrokerFixture, Box<dyn Error>> {
        assert_eq!(
            authorization.capabilities().profile_id(),
            PROFILE_UUID.to_string()
        );
        let (directory, assets) = asset_service()?;
        let provider = Arc::new(TestProviderActuator::default());
        let credential = Arc::new(TestCredentialPresenceActuator::default());
        let provider_policy = ProviderPolicy::new(
            PROFILE_UUID.to_string(),
            ProviderMode::Enabled,
            [ProviderEndpoint::new(PROVIDER, ENDPOINT)?],
            [crate::CredentialScope::new(
                PROFILE_UUID.to_string(),
                "plugin.fixture",
                PROVIDER,
                SecretId::new(SECRET)?,
            )?],
        )?;
        let model_store = ModelStore::new(ParserLimits::default())?;
        let rng_policy =
            PluginRngPolicy::new(RngProfileVersion::V2, RngAlgorithm::Philox4x32_10, 7);
        let broker = if let Some(verifier) = verifier {
            PluginCapabilityBroker::new_with_provider_cost_acceptance(
                assets,
                model_store,
                provider_policy,
                verifier,
                provider.clone(),
                credential.clone(),
                clock,
                rng_policy,
            )
        } else {
            PluginCapabilityBroker::new(
                assets,
                model_store,
                provider_policy,
                provider.clone(),
                credential.clone(),
                clock,
                rng_policy,
            )
        };
        Ok((directory, broker, provider, credential))
    }

    fn context(
        authorization: PluginAuthorization,
        clock: &TestClock,
        cancellation: CancellationToken,
        maximum_response_bytes: u64,
    ) -> Result<PluginServiceInvocationContext, PluginServiceError> {
        PluginServiceInvocationContext::new(
            ProfileId(PROFILE_UUID),
            PromptId(PROMPT_UUID),
            AttemptId(ATTEMPT_UUID),
            NodeId("node.fixture".to_owned()),
            authorization,
            cancellation,
            clock.now() + Duration::from_secs(30),
            maximum_response_bytes,
        )
    }

    fn priced_context(
        authorization: PluginAuthorization,
        clock: &TestClock,
    ) -> Result<PluginServiceInvocationContext, PluginServiceError> {
        PluginServiceInvocationContext::new_with_principal(
            ProfileId(PROFILE_UUID),
            PromptId(PROMPT_UUID),
            AttemptId(ATTEMPT_UUID),
            NodeId("node.fixture".to_owned()),
            "principal-a",
            authorization,
            CancellationToken::default(),
            clock.now() + Duration::from_secs(30),
            1_024,
        )
    }

    fn provider_receipt_context(
        authorization: PluginAuthorization,
        clock: &TestClock,
    ) -> Result<PluginServiceInvocationContext, PluginServiceError> {
        PluginServiceInvocationContext::new_with_principal(
            ProfileId(PROFILE_UUID),
            PromptId(PROMPT_UUID),
            AttemptId(ATTEMPT_UUID),
            NodeId("node.fixture".to_owned()),
            "principal-a",
            authorization,
            CancellationToken::default(),
            clock.now() + Duration::from_secs(30),
            4 * 1_024,
        )
    }

    fn provider_cost_scope(
        provider_binding_sha256: &str,
        provider: &str,
        endpoint: &str,
        price_bound: ProviderPriceBound,
    ) -> Result<ProviderCostAcceptanceScope, TrustError> {
        ProviderCostAcceptanceScope::new(
            "principal-a",
            PROFILE_UUID.to_string(),
            PROMPT_UUID.to_string(),
            PROMPT_SHA256,
            ATTEMPT_UUID.to_string(),
            "node.fixture",
            REQUEST_ORDINAL,
            format!("{:x}", Sha256::digest(REQUEST_BODY)),
            "plugin.fixture",
            DIGEST,
            provider_binding_sha256,
            provider,
            endpoint,
            price_bound,
        )
    }

    fn fixture_price_badge() -> NativeSchemaValue {
        NativeSchemaValue::PreservedExpression {
            source: r#"{"type":"usd","usd":0.025}"#.to_owned(),
            sha256: format!(
                "{:x}",
                Sha256::digest(r#"{"type":"usd","usd":0.025}"#.as_bytes())
            ),
        }
    }

    #[test]
    fn provider_world_paid_request_requires_and_consumes_exact_host_approval()
    -> Result<(), Box<dyn Error>> {
        let clock = Arc::new(TestClock::new(Instant::now()));
        let mut capabilities = provider_capabilities(true);
        capabilities.push(capability(CapabilityKind::Secret, SECRET));
        let authorization = authorization(capabilities)?;
        let cost_issuer = Arc::new(ProviderCostAcceptanceIssuer::from_seed(
            [81; 32],
            clock.now(),
        )?);
        let cost_authority = Arc::new(ProviderCostApprovalAuthority::new(
            cost_issuer,
            clock.clone(),
        ));
        let (_directory, broker, provider, credential) = broker_with_cost_acceptance(
            &authorization,
            clock.clone(),
            Some(cost_authority.verifier()?),
        )?;
        credential.present.store(true, Ordering::Release);
        provider.set_response(b"paid-provider-response".to_vec());
        let receipt_issuer = Arc::new(ProviderResultReceiptIssuer::from_seed(
            [82; 32],
            clock.now(),
        )?);
        let receipt_authority = ProviderResultReceiptAuthority::new(
            "principal-a",
            PROMPT_SHA256,
            DIGEST,
            receipt_issuer,
            Duration::from_secs(30),
        )?;
        let paid_context = |authorization| {
            provider_receipt_context(authorization, &clock)?
                .with_provider_result_authority(receipt_authority.clone())?
                .with_provider_cost_requirement(fixture_price_badge(), Some(cost_authority.clone()))
        };

        let mut denied = broker.begin_invocation(paid_context(authorization.clone())?)?;
        assert_eq!(
            denied.handle_wire_request(PluginServiceWireRequest::ExecuteProvider {
                provider: PROVIDER.to_owned(),
                endpoint: ENDPOINT.to_owned(),
                secret_id: Some(SECRET.to_owned()),
                body: REQUEST_BODY.to_vec(),
            }),
            PluginServiceWireResponse::Failure(PluginServiceWireFailure::ProviderDenied)
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 0);
        denied.abort();

        cost_authority.approve(ProviderCostApproval::new(
            "principal-a",
            PROFILE_UUID.to_string(),
            PROMPT_SHA256,
            "node.fixture",
            0,
            "plugin.fixture",
            DIGEST,
            DIGEST,
            PROVIDER,
            ENDPOINT,
            &fixture_price_badge(),
            ProviderPriceBound::new("USD", 25_000)?,
            clock.now() + Duration::from_secs(30),
            ProviderCostNonce::new([83; 32])?,
        )?)?;
        let mut invocation = broker.begin_invocation(paid_context(authorization)?)?;
        let PluginServiceWireResponse::Bytes(receipt) =
            invocation.handle_wire_request(PluginServiceWireRequest::ExecuteProvider {
                provider: PROVIDER.to_owned(),
                endpoint: ENDPOINT.to_owned(),
                secret_id: Some(SECRET.to_owned()),
                body: REQUEST_BODY.to_vec(),
            })
        else {
            return Err("paid provider request did not return a receipt".into());
        };
        assert_eq!(provider.calls.load(Ordering::Acquire), 1);
        let receipt_set = ProviderResultReceiptSet::new(vec![receipt])?;
        let resolved = invocation.resolve_provider_result_receipt_set(&receipt_set)?;
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].response(), b"paid-provider-response");
        let expected_idempotency_key = resolved[0].identity().idempotency_key_sha256();
        assert_eq!(
            provider.last_idempotency_key.lock().as_deref(),
            Some(expected_idempotency_key.as_str())
        );
        invocation.finish()?;
        Ok(())
    }

    #[test]
    fn capability_denials_happen_before_provider_or_credential_actuators()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(Vec::new())?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, provider, credential) = broker(&authorization, clock.clone())?;
        let invocation = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            1_024,
        )?)?;
        let secret = SecretId::new(SECRET)?;

        assert!(matches!(
            invocation.execute_provider_request(PROVIDER, ENDPOINT, Some(&secret), b"request"),
            Err(PluginServiceError::CapabilityDenied(
                Capability::ProviderNetwork { .. }
            ))
        ));
        assert!(matches!(
            invocation.credential_is_present(&secret),
            Err(PluginServiceError::CapabilityDenied(
                Capability::Secret { .. }
            ))
        ));
        assert_eq!(provider.calls.load(Ordering::Acquire), 0);
        assert_eq!(credential.calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn provider_upload_and_cost_denials_happen_before_actuation() -> Result<(), Box<dyn Error>> {
        let origin = Instant::now();
        let clock = Arc::new(TestClock::new(origin));
        let scope = format!("{PROVIDER}|{ENDPOINT}");

        let authorization_without_upload =
            authorization(vec![capability(CapabilityKind::NetworkProvider, &scope)])?;
        let (_directory, broker_without_upload, provider_without_upload, credential_without_upload) =
            broker(&authorization_without_upload, clock.clone())?;
        let authority = ProviderResultReceiptAuthority::new(
            "principal-a",
            PROMPT_SHA256,
            "f".repeat(64),
            Arc::new(ProviderResultReceiptIssuer::from_seed([61; 32], origin)?),
            Duration::from_secs(20),
        )?;
        let context = provider_receipt_context(authorization_without_upload, &clock)?
            .with_provider_result_authority(authority)?;
        let mut invocation = broker_without_upload.begin_invocation(context)?;
        assert_eq!(
            invocation.handle_wire_request(PluginServiceWireRequest::ExecuteProvider {
                provider: PROVIDER.to_owned(),
                endpoint: ENDPOINT.to_owned(),
                secret_id: None,
                body: REQUEST_BODY.to_vec(),
            }),
            PluginServiceWireResponse::Failure(PluginServiceWireFailure::CapabilityDenied)
        );
        assert_eq!(provider_without_upload.calls.load(Ordering::Acquire), 0);
        assert_eq!(credential_without_upload.calls.load(Ordering::Acquire), 0);

        let authorization = authorization(provider_capabilities(false))?;
        let (_directory, broker, provider, credential) = broker(&authorization, clock.clone())?;
        let authority = ProviderResultReceiptAuthority::new(
            "principal-a",
            PROMPT_SHA256,
            "f".repeat(64),
            Arc::new(ProviderResultReceiptIssuer::from_seed([62; 32], origin)?),
            Duration::from_secs(20),
        )?;
        let context = provider_receipt_context(authorization, &clock)?
            .with_provider_result_authority(authority)?
            .with_provider_cost_requirement(fixture_price_badge(), None)?;
        let mut invocation = broker.begin_invocation(context)?;
        assert_eq!(
            invocation.handle_wire_request(PluginServiceWireRequest::ExecuteProvider {
                provider: PROVIDER.to_owned(),
                endpoint: ENDPOINT.to_owned(),
                secret_id: None,
                body: REQUEST_BODY.to_vec(),
            }),
            PluginServiceWireResponse::Failure(PluginServiceWireFailure::CapabilityDenied)
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 0);
        assert_eq!(credential.calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn provider_wire_requests_return_app_owned_receipts_until_exact_settlement()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(provider_capabilities(false))?;
        let origin = Instant::now();
        let clock = Arc::new(TestClock::new(origin));
        let (_directory, broker, provider, _credential) = broker(&authorization, clock.clone())?;
        let response_body = b"app-side-provider-response".to_vec();
        *provider.response.lock() = response_body.clone();
        let authority = ProviderResultReceiptAuthority::new(
            "principal-a",
            PROMPT_SHA256,
            "f".repeat(64),
            Arc::new(ProviderResultReceiptIssuer::from_seed([31; 32], origin)?),
            Duration::from_secs(20),
        )?;
        let context = provider_receipt_context(authorization.clone(), &clock)?
            .with_provider_result_authority(authority)?;
        let mut invocation = broker.begin_invocation(context)?;
        let response = invocation.handle_wire_request(PluginServiceWireRequest::ExecuteProvider {
            provider: PROVIDER.to_owned(),
            endpoint: ENDPOINT.to_owned(),
            secret_id: None,
            body: REQUEST_BODY.to_vec(),
        });
        let PluginServiceWireResponse::Bytes(receipt) = response else {
            return Err(format!("provider receipt response was not returned: {response:?}").into());
        };
        assert_eq!(provider.calls.load(Ordering::Acquire), 1);
        assert!(
            !receipt
                .windows(response_body.len())
                .any(|window| window == response_body)
        );
        let receipt_set = ProviderResultReceiptSet::new(vec![receipt])?;
        let resolved = invocation.resolve_provider_result_receipt_set(&receipt_set)?;
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].response(), response_body);
        assert_eq!(resolved[0].identity().principal_id(), "principal-a");
        assert_eq!(resolved[0].identity().prompt_sha256(), PROMPT_SHA256);
        assert_eq!(resolved[0].identity().request_ordinal(), 0);
        assert_eq!(
            resolved[0].identity().request_sha256(),
            format!("{:x}", Sha256::digest(REQUEST_BODY))
        );
        invocation.finish()?;

        let mut unresolved = broker.begin_invocation(
            provider_receipt_context(authorization, &clock)?.with_provider_result_authority(
                ProviderResultReceiptAuthority::new(
                    "principal-a",
                    PROMPT_SHA256,
                    "f".repeat(64),
                    Arc::new(ProviderResultReceiptIssuer::from_seed([32; 32], origin)?),
                    Duration::from_secs(20),
                )?,
            )?,
        )?;
        let response = unresolved.handle_wire_request(PluginServiceWireRequest::ExecuteProvider {
            provider: PROVIDER.to_owned(),
            endpoint: ENDPOINT.to_owned(),
            secret_id: None,
            body: REQUEST_BODY.to_vec(),
        });
        assert!(matches!(response, PluginServiceWireResponse::Bytes(_)));
        assert_eq!(
            unresolved.finish(),
            Err(PluginServiceError::ProviderResultReceiptAuthorityDenied)
        );
        Ok(())
    }

    #[test]
    fn provider_cost_acceptance_denials_make_zero_priced_actuator_calls()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(provider_capabilities(true))?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let issuer = ProviderCostAcceptanceIssuer::from_seed([11; 32], clock.now())?;
        let (_directory, broker, provider, credential) =
            broker_with_cost_acceptance(&authorization, clock.clone(), Some(issuer.verifier()?))?;
        let binding = "a".repeat(64);
        let other_binding = "b".repeat(64);
        let price = ProviderPriceBound::new("USD", 25_000)?;
        let nonce = ProviderCostNonce::new([21; 32])?;
        let acceptance = issuer.issue(
            provider_cost_scope(&binding, PROVIDER, ENDPOINT, price.clone())?,
            clock.now(),
            clock.now() + Duration::from_secs(10),
            nonce,
        )?;

        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                nonce,
                None,
                b"request",
            ),
            Err(PluginServiceError::ProviderCostAcceptanceRequired)
        );

        let foreign_issuer = ProviderCostAcceptanceIssuer::from_seed([12; 32], clock.now())?;
        let forged = foreign_issuer.issue(
            provider_cost_scope(&binding, PROVIDER, ENDPOINT, price.clone())?,
            clock.now(),
            clock.now() + Duration::from_secs(10),
            ProviderCostNonce::new([22; 32])?,
        )?;
        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                forged.nonce(),
                Some(&forged),
                b"request",
            ),
            Err(PluginServiceError::ProviderCostAcceptanceDenied)
        );

        let expired_nonce = ProviderCostNonce::new([23; 32])?;
        let expired = issuer.issue(
            provider_cost_scope(&binding, PROVIDER, ENDPOINT, price.clone())?,
            clock.now(),
            clock.now() + Duration::from_millis(1),
            expired_nonce,
        )?;
        clock.advance(Duration::from_millis(1));
        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                expired_nonce,
                Some(&expired),
                b"request",
            ),
            Err(PluginServiceError::ProviderCostAcceptanceDenied)
        );

        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &other_binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                nonce,
                Some(&acceptance),
                b"request",
            ),
            Err(PluginServiceError::ProviderCostAcceptanceDenied)
        );
        for (prompt_sha256, request_ordinal, body) in [
            ("e".repeat(64), REQUEST_ORDINAL, REQUEST_BODY),
            (PROMPT_SHA256.to_owned(), REQUEST_ORDINAL + 1, REQUEST_BODY),
            (PROMPT_SHA256.to_owned(), REQUEST_ORDINAL, b"other-request"),
        ] {
            let invocation =
                broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
            assert_eq!(
                invocation.execute_priced_provider_request(
                    &binding,
                    &prompt_sha256,
                    request_ordinal,
                    PROVIDER,
                    ENDPOINT,
                    None,
                    &price,
                    nonce,
                    Some(&acceptance),
                    body,
                ),
                Err(PluginServiceError::ProviderCostAcceptanceDenied)
            );
        }
        let invocation = broker.begin_invocation(priced_context(authorization, &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                ProviderCostNonce::new([24; 32])?,
                Some(&acceptance),
                b"request",
            ),
            Err(PluginServiceError::ProviderCostAcceptanceDenied)
        );

        assert_eq!(provider.calls.load(Ordering::Acquire), 0);
        assert_eq!(credential.calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn provider_cost_acceptance_is_single_use_and_legacy_requests_stay_unpriced()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(provider_capabilities(true))?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let issuer = ProviderCostAcceptanceIssuer::from_seed([31; 32], clock.now())?;
        let (_directory, broker, provider, _credential) =
            broker_with_cost_acceptance(&authorization, clock.clone(), Some(issuer.verifier()?))?;
        provider.set_response(b"provider-response".to_vec());
        let binding = "c".repeat(64);
        let price = ProviderPriceBound::new("USD", 30_000)?;
        let nonce = ProviderCostNonce::new([32; 32])?;
        let acceptance = issuer.issue(
            provider_cost_scope(&binding, PROVIDER, ENDPOINT, price.clone())?,
            clock.now(),
            clock.now() + Duration::from_secs(10),
            nonce,
        )?;

        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                nonce,
                Some(&acceptance),
                b"request",
            )?,
            b"provider-response"
        );
        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                nonce,
                Some(&acceptance),
                b"request",
            ),
            Err(PluginServiceError::ProviderCostAcceptanceReused)
        );

        let legacy = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            1_024,
        )?)?;
        assert_eq!(
            legacy.execute_provider_request(PROVIDER, ENDPOINT, None, b"request")?,
            b"provider-response"
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 2);

        let wire = serde_json::to_value(PluginServiceWireRequest::ExecuteProvider {
            provider: PROVIDER.to_owned(),
            endpoint: ENDPOINT.to_owned(),
            secret_id: None,
            body: b"request".to_vec(),
        })?;
        let wire_text = serde_json::to_string(&wire)?;
        assert!(!wire_text.contains("acceptance"));
        assert!(!wire_text.contains("nonce"));
        assert!(!wire_text.contains("price"));
        Ok(())
    }

    fn provider_cost_nonce(index: u64) -> Result<ProviderCostNonce, TrustError> {
        let mut bytes = [0_u8; 32];
        bytes[..8].copy_from_slice(&index.checked_add(1).unwrap_or(u64::MAX).to_le_bytes());
        ProviderCostNonce::new(bytes)
    }

    #[test]
    fn provider_cost_nonce_claims_expire_roll_back_and_remain_bounded() -> Result<(), Box<dyn Error>>
    {
        let authorization = authorization(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("{PROVIDER}|{ENDPOINT}"),
        )])?;
        let now = Instant::now();
        let clock = Arc::new(TestClock::new(now));
        let (_directory, broker, _provider, _credential) =
            broker_with_cost_acceptance(&authorization, clock, None)?;

        assert!(matches!(
            broker
                .inner
                .claim_provider_cost_nonce(provider_cost_nonce(0)?, now, now,),
            Err(PluginServiceError::ProviderCostAcceptanceDenied)
        ));
        assert!(broker.inner.consumed_provider_cost_nonces.lock().is_empty());

        {
            let _claim = broker.inner.claim_provider_cost_nonce(
                provider_cost_nonce(1)?,
                now + Duration::from_secs(1),
                now,
            )?;
        }
        assert!(broker.inner.consumed_provider_cost_nonces.lock().is_empty());

        {
            let mut consumed_nonces = broker.inner.consumed_provider_cost_nonces.lock();
            for index in 0..MAX_CONSUMED_PROVIDER_COST_NONCES {
                consumed_nonces.insert(
                    provider_cost_nonce(u64::try_from(index)?.checked_add(10).unwrap_or(u64::MAX))?,
                    now,
                );
            }
        }
        broker
            .inner
            .claim_provider_cost_nonce(provider_cost_nonce(2)?, now + Duration::from_secs(1), now)?
            .retain();
        let consumed_nonces = broker.inner.consumed_provider_cost_nonces.lock();
        assert_eq!(consumed_nonces.len(), 1);
        assert_eq!(
            consumed_nonces.get(&provider_cost_nonce(2)?),
            Some(&(now + Duration::from_secs(1)))
        );
        Ok(())
    }

    #[test]
    fn concurrent_provider_cost_nonce_claims_admit_one_caller() -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("{PROVIDER}|{ENDPOINT}"),
        )])?;
        let now = Instant::now();
        let clock = Arc::new(TestClock::new(now));
        let (_directory, broker, _provider, _credential) =
            broker_with_cost_acceptance(&authorization, clock, None)?;
        let barrier = Arc::new(Barrier::new(3));
        let nonce = provider_cost_nonce(100)?;
        let handles = (0..2)
            .map(|_| {
                let broker = broker.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || -> Result<bool, PluginServiceError> {
                    barrier.wait();
                    match broker.inner.claim_provider_cost_nonce(
                        nonce,
                        now + Duration::from_secs(1),
                        now,
                    ) {
                        Ok(claim) => {
                            claim.retain();
                            Ok(true)
                        }
                        Err(PluginServiceError::ProviderCostAcceptanceReused) => Ok(false),
                        Err(error) => Err(error),
                    }
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let admitted = handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "provider cost claim thread panicked".to_owned())?
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        assert_eq!(admitted.into_iter().filter(|admitted| *admitted).count(), 1);
        assert_eq!(broker.inner.consumed_provider_cost_nonces.lock().len(), 1);
        Ok(())
    }

    #[test]
    fn provider_cost_nonce_rolls_back_before_actuation_and_is_retained_after_attempt()
    -> Result<(), Box<dyn Error>> {
        let mut capabilities = provider_capabilities(true);
        capabilities.push(capability(CapabilityKind::Secret, SECRET));
        let authorization = authorization(capabilities)?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let issuer = ProviderCostAcceptanceIssuer::from_seed([41; 32], clock.now())?;
        let (_directory, broker, provider, credential) =
            broker_with_cost_acceptance(&authorization, clock.clone(), Some(issuer.verifier()?))?;
        provider.set_response(b"provider-response".to_vec());
        let binding = "d".repeat(64);
        let price = ProviderPriceBound::new("USD", 40_000)?;
        let credential_nonce = ProviderCostNonce::new([42; 32])?;
        let credential_acceptance = issuer.issue(
            provider_cost_scope(&binding, PROVIDER, ENDPOINT, price.clone())?,
            clock.now(),
            clock.now() + Duration::from_secs(10),
            credential_nonce,
        )?;
        let secret = SecretId::new(SECRET)?;

        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                Some(&secret),
                &price,
                credential_nonce,
                Some(&credential_acceptance),
                b"request",
            ),
            Err(PluginServiceError::CredentialUnavailable)
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 0);
        assert_eq!(credential.calls.load(Ordering::Acquire), 1);

        credential.present.store(true, Ordering::Release);
        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                Some(&secret),
                &price,
                credential_nonce,
                Some(&credential_acceptance),
                b"request",
            )?,
            b"provider-response"
        );

        let failed_nonce = ProviderCostNonce::new([43; 32])?;
        let failed_acceptance = issuer.issue(
            provider_cost_scope(&binding, PROVIDER, ENDPOINT, price.clone())?,
            clock.now(),
            clock.now() + Duration::from_secs(10),
            failed_nonce,
        )?;
        provider.set_fail_call(true);
        let invocation = broker.begin_invocation(priced_context(authorization.clone(), &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                failed_nonce,
                Some(&failed_acceptance),
                b"request",
            ),
            Err(PluginServiceError::ActuatorFailed {
                service: "provider",
                message: "injected provider failure".to_owned(),
            })
        );

        provider.set_fail_call(false);
        let invocation = broker.begin_invocation(priced_context(authorization, &clock)?)?;
        assert_eq!(
            invocation.execute_priced_provider_request(
                &binding,
                PROMPT_SHA256,
                REQUEST_ORDINAL,
                PROVIDER,
                ENDPOINT,
                None,
                &price,
                failed_nonce,
                Some(&failed_acceptance),
                b"request",
            ),
            Err(PluginServiceError::ProviderCostAcceptanceReused)
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 2);
        assert_eq!(credential.calls.load(Ordering::Acquire), 2);
        Ok(())
    }

    #[test]
    fn provider_policy_seals_requests_and_cancellation_and_bounds_are_rechecked()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![
            capability(
                CapabilityKind::NetworkProvider,
                &format!("{PROVIDER}|{ENDPOINT}"),
            ),
            capability(CapabilityKind::Secret, SECRET),
        ])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, provider, credential) = broker(&authorization, clock.clone())?;
        credential.present.store(true, Ordering::Release);
        let secret = SecretId::new(SECRET)?;
        provider.set_response(vec![0; 17]);
        let invocation = broker.begin_invocation(context(
            authorization.clone(),
            &clock,
            CancellationToken::default(),
            16,
        )?)?;
        assert_eq!(
            invocation.execute_provider_request(PROVIDER, ENDPOINT, Some(&secret), b"request"),
            Err(PluginServiceError::ResponseTooLarge { maximum: 16 })
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 1);
        assert_eq!(credential.calls.load(Ordering::Acquire), 1);
        assert_eq!(
            *provider.last_secret.lock(),
            Some(b"fixture-secret-value".to_vec())
        );
        assert_eq!(
            *provider.last_authorized_request.lock(),
            Some((
                PROVIDER.to_owned(),
                ENDPOINT.to_owned(),
                Some(SECRET.to_owned())
            ))
        );

        provider.set_response(b"ok".to_vec());
        provider.set_cancel_during_call(true);
        let cancellation = CancellationToken::default();
        let invocation =
            broker.begin_invocation(context(authorization, &clock, cancellation.clone(), 16)?)?;
        assert_eq!(
            invocation.execute_provider_request(PROVIDER, ENDPOINT, Some(&secret), b"request"),
            Err(PluginServiceError::Cancelled)
        );
        assert!(cancellation.is_cancelled());
        Ok(())
    }

    #[test]
    fn deadlines_and_response_limits_fail_before_work_starts() -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("{PROVIDER}|{ENDPOINT}"),
        )])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, provider, _credential) = broker(&authorization, clock.clone())?;
        assert!(matches!(
            PluginServiceInvocationContext::new(
                ProfileId(PROFILE_UUID),
                PromptId(PROMPT_UUID),
                AttemptId(ATTEMPT_UUID),
                NodeId("node.fixture".to_owned()),
                authorization.clone(),
                CancellationToken::default(),
                clock.now() + Duration::from_secs(1),
                MAX_PLUGIN_SERVICE_RESPONSE_BYTES + 1,
            ),
            Err(PluginServiceError::InvalidResponseLimit {
                maximum: MAX_PLUGIN_SERVICE_RESPONSE_BYTES
            })
        ));
        let context = PluginServiceInvocationContext::new(
            ProfileId(PROFILE_UUID),
            PromptId(PROMPT_UUID),
            AttemptId(ATTEMPT_UUID),
            NodeId("node.fixture".to_owned()),
            authorization,
            CancellationToken::default(),
            clock.now() + Duration::from_millis(1),
            16,
        )?;
        clock.advance(Duration::from_millis(1));
        assert!(matches!(
            broker.begin_invocation(context),
            Err(PluginServiceError::DeadlineExceeded)
        ));
        assert_eq!(provider.calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn rng_aborts_roll_back_and_successful_invocations_commit() -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![capability(CapabilityKind::Randomness, "noise")])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let invocation_context =
            context(authorization, &clock, CancellationToken::default(), 1_024)?;

        let mut aborted = broker.begin_invocation(invocation_context.clone())?;
        let first = aborted.random_bytes("noise", 32)?;
        aborted.abort();

        let mut committed = broker.begin_invocation(invocation_context.clone())?;
        let replayed = committed.random_bytes("noise", 32)?;
        assert_eq!(replayed, first);
        committed.finish()?;

        let mut advanced = broker.begin_invocation(invocation_context)?;
        let next = advanced.random_bytes("noise", 32)?;
        assert_ne!(next, first);
        advanced.finish()?;
        Ok(())
    }

    #[test]
    fn asset_and_model_operations_delegate_to_canonical_index_without_path_exposure()
    -> Result<(), Box<dyn Error>> {
        let model_id = "zed-asset://model/fixture.json";
        let authorization = authorization(vec![
            capability(CapabilityKind::Filesystem, "input"),
            capability(CapabilityKind::Filesystem, "model"),
            capability(CapabilityKind::Model, model_id),
        ])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let invocation = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            4_096,
        )?)?;

        assert_eq!(
            invocation.read_asset(AssetNamespace::Input, "zed-asset://input/fixture.bin")?,
            b"canonical asset bytes"
        );
        let model = invocation.load_model(model_id)?;
        assert_eq!(model.model_id(), model_id);
        assert_eq!(model.model().documents().len(), 1);
        let debug = format!("{model:?}");
        assert!(!debug.contains(_directory.path().to_string_lossy().as_ref()));

        let error = invocation
            .read_asset(
                AssetNamespace::Input,
                "zed-asset://input/../../outside-secret",
            )
            .expect_err("canonical root must reject traversal");
        let error_text = error.to_string();
        assert!(!error_text.contains("outside-secret"));
        assert!(!error_text.contains(_directory.path().to_string_lossy().as_ref()));
        Ok(())
    }

    #[test]
    fn credential_presence_and_clock_use_injected_operation_only_owners()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![
            capability(CapabilityKind::Secret, SECRET),
            capability(CapabilityKind::Clock, "monotonic"),
        ])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, credential) = broker(&authorization, clock.clone())?;
        credential.present.store(true, Ordering::Release);
        let invocation = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            128,
        )?)?;
        let secret = SecretId::new(SECRET)?;

        assert!(invocation.credential_is_present(&secret)?);
        assert_eq!(credential.calls.load(Ordering::Acquire), 1);
        assert_eq!(*credential.last_secret_id.lock(), Some(SECRET.to_owned()));
        assert_eq!(invocation.monotonic_milliseconds("monotonic")?, 0);
        clock.advance(Duration::from_millis(25));
        assert_eq!(invocation.monotonic_milliseconds("monotonic")?, 25);
        Ok(())
    }

    #[test]
    fn cancelled_rng_work_does_not_advance_the_stream() -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![capability(CapabilityKind::Randomness, "noise")])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let baseline_context = context(
            authorization.clone(),
            &clock,
            CancellationToken::default(),
            1_024,
        )?;
        let mut baseline = broker.begin_invocation(baseline_context)?;
        let expected = baseline.random_bytes("noise", 16)?;
        baseline.abort();

        let cancellation = CancellationToken::default();
        let cancelled_context =
            context(authorization.clone(), &clock, cancellation.clone(), 1_024)?;
        let mut cancelled = broker.begin_invocation(cancelled_context)?;
        cancellation.cancel();
        assert_eq!(
            cancelled.random_bytes("noise", 16),
            Err(PluginServiceError::Cancelled)
        );
        cancelled.abort();

        let mut replay = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            1_024,
        )?)?;
        assert_eq!(replay.random_bytes("noise", 16)?, expected);
        replay.finish()?;
        Ok(())
    }

    #[test]
    fn asset_namespace_adapter_cannot_expand_a_signed_grant() -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![capability(CapabilityKind::Filesystem, "input")])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let invocation = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            1_024,
        )?)?;

        assert!(matches!(
            invocation.read_asset(
                AssetNamespace::Model,
                "zed-asset://model/fixture.json"
            ),
            Err(PluginServiceError::CapabilityDenied(Capability::Asset {
                namespace,
                action: AssetOperation::Read,
            })) if namespace == "model"
        ));
        Ok(())
    }

    #[test]
    fn model_loading_requires_both_opaque_handle_and_canonical_asset_grants()
    -> Result<(), Box<dyn Error>> {
        let model_id = "zed-asset://model/fixture.json";
        let authorization = authorization(vec![capability(CapabilityKind::Model, model_id)])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let invocation = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            4_096,
        )?)?;

        assert!(matches!(
            invocation.load_model(model_id),
            Err(PluginServiceError::CapabilityDenied(Capability::Asset {
                namespace,
                action: AssetOperation::Read,
            })) if namespace == "model"
        ));
        Ok(())
    }

    #[test]
    fn provider_endpoint_policy_cannot_be_bypassed_by_a_broader_capability()
    -> Result<(), Box<dyn Error>> {
        let denied_endpoint = "https://provider.invalid/v2";
        let authorization = authorization(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("{PROVIDER}|{denied_endpoint}"),
        )])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, provider, _credential) = broker(&authorization, clock.clone())?;
        let invocation = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            1_024,
        )?)?;

        assert_eq!(
            invocation.execute_provider_request(PROVIDER, denied_endpoint, None, b"request"),
            Err(PluginServiceError::ProviderPolicyDenied)
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn requested_random_response_is_bounded_before_allocation() -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![capability(CapabilityKind::Randomness, "noise")])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let mut invocation = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            8,
        )?)?;

        assert_eq!(
            invocation.random_bytes("noise", 9),
            Err(PluginServiceError::ResponseTooLarge { maximum: 8 })
        );
        invocation.abort();
        Ok(())
    }

    #[test]
    fn plugin_model_handle_does_not_expose_mutable_store_or_paths() -> Result<(), Box<dyn Error>> {
        let model_id = "zed-asset://model/fixture.json";
        let authorization = authorization(vec![
            capability(CapabilityKind::Filesystem, "model"),
            capability(CapabilityKind::Model, model_id),
        ])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let invocation = broker.begin_invocation(context(
            authorization,
            &clock,
            CancellationToken::default(),
            4_096,
        )?)?;
        let model = invocation.load_model(model_id)?;

        assert_eq!(model.model_id(), model_id);
        assert_eq!(model.model_format(), "json-config");
        assert!(!model.model_identity().is_empty());
        assert!(!format!("{model:?}").contains(directory.path().to_string_lossy().as_ref()));
        assert_eq!(model.model().accounting(), model.accounting());
        Ok(())
    }

    #[test]
    fn dropped_invocations_release_rng_leases_without_committing() -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![capability(CapabilityKind::Randomness, "noise")])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let invocation_context =
            context(authorization, &clock, CancellationToken::default(), 1_024)?;
        let expected = {
            let mut invocation = broker.begin_invocation(invocation_context.clone())?;
            invocation.random_bytes("noise", 8)?
        };
        let mut replay = broker.begin_invocation(invocation_context)?;
        assert_eq!(replay.random_bytes("noise", 8)?, expected);
        replay.finish()?;
        Ok(())
    }

    #[test]
    fn failed_capability_operations_prevent_rng_commit() -> Result<(), Box<dyn Error>> {
        let authorization = authorization(vec![capability(CapabilityKind::Randomness, "noise")])?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, _provider, _credential) = broker(&authorization, clock.clone())?;
        let invocation_context =
            context(authorization, &clock, CancellationToken::default(), 1_024)?;
        let mut failed = broker.begin_invocation(invocation_context.clone())?;
        let expected = failed.random_bytes("noise", 8)?;
        assert!(matches!(
            failed.read_asset(AssetNamespace::Input, "zed-asset://input/fixture.bin"),
            Err(PluginServiceError::CapabilityDenied(
                Capability::Asset { .. }
            ))
        ));
        assert_eq!(failed.finish(), Err(PluginServiceError::InvocationFailed));

        let mut replay = broker.begin_invocation(invocation_context)?;
        assert_eq!(replay.random_bytes("noise", 8)?, expected);
        replay.finish()?;
        Ok(())
    }

    fn provider_streaming_contract() -> ProviderStreamingContractV2 {
        ProviderStreamingContractV2 {
            methods: vec![ProviderHttpMethodV2::Post],
            maximum_headers: 8,
            maximum_header_bytes: 1_024,
            maximum_request_body_bytes: 1_024,
            maximum_response_body_bytes: 1_024,
            maximum_chunk_bytes: 256,
            maximum_ndjson_line_bytes: 256,
            maximum_wait_milliseconds: 1_000,
            maximum_uploads: 1,
            maximum_upload_body_bytes: 1_024,
            maximum_cost_requests: 1,
            maximum_progress_total: 100,
            uploads: true,
            cost_requests: true,
        }
    }

    fn provider_binding() -> Result<ProviderBindingSet, Box<dyn Error>> {
        let mut binding = ProviderBindingSet {
            schema_version: PROVIDER_BINDING_SCHEMA_VERSION,
            implementation_namespace: "plugin.fixture".to_owned(),
            bindings_sha256: "0".repeat(64),
            bindings: vec![ProviderBindingClaim {
                feature_id: "COMFY-NODE-0001".to_owned(),
                node_id: "node.fixture".to_owned(),
                contract_sha256: "3".repeat(64),
                transport_schema: "zed:sim-provider-transport@1".parse()?,
                materializer_schema: "zed:sim-provider-materializer@1".parse()?,
            }],
        };
        binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
        Ok(binding)
    }

    fn worker_registry_deployment() -> Result<WorkerRegistryDeploymentPlan, Box<dyn Error>> {
        worker_registry_deployment_at(7)
    }

    fn worker_registry_deployment_at(
        generation: u64,
    ) -> Result<WorkerRegistryDeploymentPlan, Box<dyn Error>> {
        let generation = WorkerRegistryGeneration::new(generation)?;
        let component = WorkerComponentDescriptor::new(
            "fixture.extension",
            "1.0.0",
            "plugin.fixture",
            "1.0.0",
            WorkerSha256Digest::new("b".repeat(64))?,
            WorkerSha256Digest::new("d".repeat(64))?,
            WorkerSha256Digest::new(DIGEST)?,
            1,
            1,
            1,
        )?;
        let begin = WorkerRegistryDeploymentBegin::new(
            generation,
            WorkerSha256Digest::new("a".repeat(64))?,
            vec![component],
        )?;
        let verifier = crate::PluginAuthorizationSealer::from_seed(
            [0x71; 32],
            crate::PermissionPolicyGeneration::new(1)?,
        )?
        .verifier()?;
        let chunks = [
            WorkerComponentContent::Manifest,
            WorkerComponentContent::Authorization,
            WorkerComponentContent::Component,
        ]
        .into_iter()
        .map(|content| WorkerRegistryDeploymentChunk::new(generation, 0, content, 0, vec![1]))
        .collect::<Result<Vec<_>, _>>()?;
        Ok(WorkerRegistryDeploymentPlan::new(begin, chunks, verifier)?)
    }

    fn activation_start(
        deployment: &WorkerRegistryDeploymentPlan,
        binding: &ProviderBindingSet,
    ) -> NativeProviderWorkerSessionStart {
        let component = &deployment.begin().components()[0];
        NativeProviderWorkerSessionStart {
            session_id: "session.fixture".to_owned(),
            registry_generation: deployment.begin().generation().get(),
            registry_digest_sha256: deployment
                .begin()
                .registry_digest_sha256()
                .as_str()
                .to_owned(),
            extension_id: component.extension_id().to_owned(),
            extension_version: component.extension_version().to_owned(),
            plugin_identifier: component.plugin_identifier().to_owned(),
            plugin_version: component.plugin_version().to_owned(),
            manifest_digest_sha256: component.manifest_digest_sha256().as_str().to_owned(),
            component_digest_sha256: component.component_digest_sha256().as_str().to_owned(),
            authorization_generation_sha256: component
                .authorization_generation()
                .as_str()
                .to_owned(),
            binding_set_sha256: binding.bindings_sha256.clone(),
            node_id: "node.fixture".to_owned(),
            compiled_plan_sha256: "c".repeat(64),
            maximum_response_bytes: MAX_PLUGIN_SERVICE_RESPONSE_BYTES,
        }
    }

    #[test]
    fn legacy_provider_worker_wire_fixture_is_frozen() -> Result<(), Box<dyn Error>> {
        let deployment = worker_registry_deployment()?;
        let binding = provider_binding()?;
        let request = NativeProviderWorkerRequest::Begin(activation_start(&deployment, &binding))
            .to_bytes()?;
        let response = NativeProviderWorkerResponse::Begun.to_bytes()?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&request)),
            "c39833ba648209526b0a1f87110e72503857e8a5e4058a2aec53f1d7709a02de"
        );
        assert_eq!(response, [0]);
        assert_eq!(
            format!("{:x}", Sha256::digest(&response)),
            "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
        );
        assert!(matches!(
            NativeProviderWorkerRequest::from_bytes(&request)?,
            NativeProviderWorkerRequest::Begin(_)
        ));
        assert_eq!(
            NativeProviderWorkerResponse::from_bytes(&response)?,
            NativeProviderWorkerResponse::Begun
        );
        Ok(())
    }

    fn activation_grant(
        binding: &ProviderBindingSet,
        outer_signing_payload_sha256: [u8; 32],
    ) -> Result<ProviderRuntimeActivationGrant, Box<dyn Error>> {
        activation_grant_for_deployment(
            binding,
            outer_signing_payload_sha256,
            &worker_registry_deployment()?,
        )
    }

    fn activation_grant_for_deployment(
        binding: &ProviderBindingSet,
        outer_signing_payload_sha256: [u8; 32],
        worker_deployment: &WorkerRegistryDeploymentPlan,
    ) -> Result<ProviderRuntimeActivationGrant, Box<dyn Error>> {
        Ok(
            ProviderRuntimeActivationGrant::checked_from_active_deployment(
                PROFILE_UUID.to_string(),
                "principal-a",
                PROMPT_UUID.to_string(),
                PROMPT_SHA256,
                ATTEMPT_UUID.to_string(),
                "node.fixture",
                REQUEST_ORDINAL,
                worker_deployment,
                DIGEST,
                encode_lower_hex(&outer_signing_payload_sha256),
                "b".repeat(64),
                binding.bindings_sha256.clone(),
                "c".repeat(64),
            )?,
        )
    }

    fn preflight_activation(
        grant: ProviderRuntimeActivationGrant,
        worker_context: &WorkerProviderInvocationContext,
        manifest: ProviderManifestAuthorizationV2,
    ) -> Result<PreflightedProviderRuntimeActivationGrant, PluginServiceError> {
        let deployment = worker_registry_deployment()
            .map_err(|_| PluginServiceError::ProviderRuntimeAuthorityDenied)?;
        let start = activation_start(&deployment, manifest.provider_binding());
        grant.preflight_installed_component(worker_context, &deployment, &start, manifest)
    }

    fn bound_streaming_authority(
        binding: &ProviderBindingSet,
        outer_digest: [u8; 32],
    ) -> Result<(ProviderRuntimeAuthorityInput, ProviderRequestHeadV2), Box<dyn Error>> {
        let endpoint = "https://provider.invalid/v2";
        let authorization = authorization(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("plugin.fixture|{endpoint}"),
        )])?;
        let manifest = ProviderManifestAuthorizationV2::fixture(
            authorization,
            outer_digest,
            binding.clone(),
            provider_streaming_contract(),
        );
        let policy = ProviderPolicy::new(
            PROFILE_UUID.to_string(),
            ProviderMode::Enabled,
            [ProviderEndpoint::new("plugin.fixture", endpoint)?],
            std::iter::empty(),
        )?;
        let head = ProviderRequestHeadV2 {
            endpoint: endpoint.to_owned(),
            secret_id: None,
            method: ProviderHttpMethodV2::Post,
            headers: Vec::new(),
            declared_body_bytes: Some(0),
        };
        let service = ProviderRuntimeStreamService::new();
        let source = service.activation_grants();
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x4120),
            session_generation: 1,
            invocation: 1,
            generation: 1,
        };
        source.insert(
            &context,
            activation_grant(binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        let authority = source.claim(&context, &CancellationToken::default())?;
        let authority =
            preflight_activation(authority, &context, manifest)?.bind(&head, &policy)?;
        Ok((authority, head))
    }

    fn bound_streaming_service(
        context: &WorkerProviderInvocationContext,
        cancellation: &CancellationToken,
    ) -> Result<
        (
            ProviderRuntimeStreamService,
            ProviderRuntimeAuthorityInput,
            WorkerProviderRequestHead,
        ),
        Box<dyn Error>,
    > {
        bound_streaming_service_with_declared_body(context, cancellation, Some(0))
    }

    fn bound_streaming_service_with_declared_body(
        context: &WorkerProviderInvocationContext,
        cancellation: &CancellationToken,
        declared_body_bytes: Option<u64>,
    ) -> Result<
        (
            ProviderRuntimeStreamService,
            ProviderRuntimeAuthorityInput,
            WorkerProviderRequestHead,
        ),
        Box<dyn Error>,
    > {
        bound_streaming_service_with_contract(
            context,
            cancellation,
            declared_body_bytes,
            provider_streaming_contract(),
        )
    }

    fn bound_streaming_service_with_contract(
        context: &WorkerProviderInvocationContext,
        cancellation: &CancellationToken,
        declared_body_bytes: Option<u64>,
        streaming_contract: ProviderStreamingContractV2,
    ) -> Result<
        (
            ProviderRuntimeStreamService,
            ProviderRuntimeAuthorityInput,
            WorkerProviderRequestHead,
        ),
        Box<dyn Error>,
    > {
        let endpoint = "https://provider.invalid/v2";
        let binding = provider_binding()?;
        let outer_digest = [0x81; 32];
        let authorization = authorization(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("plugin.fixture|{endpoint}"),
        )])?;
        let manifest = ProviderManifestAuthorizationV2::fixture(
            authorization,
            outer_digest,
            binding.clone(),
            streaming_contract,
        );
        let policy = ProviderPolicy::new(
            PROFILE_UUID.to_string(),
            ProviderMode::Enabled,
            [ProviderEndpoint::new("plugin.fixture", endpoint)?],
            std::iter::empty(),
        )?;
        let sdk_head = ProviderRequestHeadV2 {
            endpoint: endpoint.to_owned(),
            secret_id: None,
            method: ProviderHttpMethodV2::Post,
            headers: vec![ProviderHeaderV2 {
                name: "x-fixture".to_owned(),
                value: "one".to_owned(),
            }],
            declared_body_bytes,
        };
        let worker_head = WorkerProviderRequestHead {
            endpoint: endpoint.to_owned(),
            secret_id: None,
            method: WorkerProviderHttpMethod::Post,
            headers: vec![WorkerProviderHeader {
                name: "x-fixture".to_owned(),
                value: "one".to_owned(),
            }],
            declared_body_bytes,
        };
        let service = ProviderRuntimeStreamService::new();
        service.register_activation(
            context,
            activation_grant(&binding, outer_digest)?,
            cancellation,
        )?;
        let authority = service.claim_activation(context, cancellation)?;
        let authority =
            preflight_activation(authority, context, manifest)?.bind(&sdk_head, &policy)?;
        Ok((service, authority, worker_head))
    }

    fn started_streaming_service(
        context: &WorkerProviderInvocationContext,
        cancellation: &CancellationToken,
    ) -> Result<(ProviderRuntimeStreamService, WorkerProviderStreamHandle), Box<dyn Error>> {
        let (service, authority, head) = bound_streaming_service(context, cancellation)?;
        let handle = service.start_request(authority, head)?;
        service.write_request_chunk(WorkerProviderRequestChunk {
            handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        Ok((service, handle))
    }

    fn complete_streaming_upload(
        service: &ProviderRuntimeStreamService,
        handle: WorkerProviderStreamHandle,
        bytes: &[u8],
    ) -> Result<WorkerProviderStreamHandle, Box<dyn Error>> {
        let upload = service.start_upload(WorkerProviderUploadRequest {
            handle,
            port_id: "image.input".to_owned(),
            media_type: "application/octet-stream".to_owned(),
            byte_length: u64::try_from(bytes.len())?,
            content_sha256: format!("{:x}", Sha256::digest(bytes)),
        })?;
        service.write_upload_chunk(WorkerProviderRequestChunk {
            handle: upload,
            sequence: 0,
            bytes: bytes.to_vec(),
            end: true,
        })?;
        service.accept_upload_completion(
            WorkerProviderUploadCompletionRequest { handle: upload },
            WorkerProviderUploadCompletion {
                download_url: format!("https://fixture.invalid/storage/{}.bin", upload.slot),
            },
        )?;
        Ok(upload)
    }

    fn prepared_mutation_proposal(
        context: &WorkerProviderInvocationContext,
        body: &[u8],
        upload_port: &str,
        upload_media_type: &str,
        upload_bytes: &[u8],
        upload_download_url: &str,
    ) -> Result<ProviderRuntimeActuationProposal, Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (service, authority, head) = bound_streaming_service_with_declared_body(
            context,
            &cancellation,
            Some(u64::try_from(body.len())?),
        )?;
        let handle = service.start_request(authority, head)?;
        service.write_request_chunk(WorkerProviderRequestChunk {
            handle,
            sequence: 0,
            bytes: body.to_vec(),
            end: true,
        })?;
        let upload = service.start_upload(WorkerProviderUploadRequest {
            handle,
            port_id: upload_port.to_owned(),
            media_type: upload_media_type.to_owned(),
            byte_length: u64::try_from(upload_bytes.len())?,
            content_sha256: format!("{:x}", Sha256::digest(upload_bytes)),
        })?;
        service.write_upload_chunk(WorkerProviderRequestChunk {
            handle: upload,
            sequence: 0,
            bytes: upload_bytes.to_vec(),
            end: true,
        })?;
        service.accept_upload_completion(
            WorkerProviderUploadCompletionRequest { handle: upload },
            WorkerProviderUploadCompletion {
                download_url: upload_download_url.to_owned(),
            },
        )?;
        Ok(service.prepare_streaming_actuation(handle)?)
    }

    fn prepared_cost_scope(
        context: &WorkerProviderInvocationContext,
        operation: &str,
        currency: &str,
        maximum_microunits: u64,
    ) -> Result<ProviderCostAcceptanceScope, Box<dyn Error>> {
        let (service, handle) = started_streaming_service(context, &CancellationToken::default())?;
        complete_streaming_upload(&service, handle, b"priced-upload")?;
        Ok(service.prepare_cost_acceptance(
            &WorkerProviderCostRequest {
                handle,
                operation: operation.to_owned(),
                currency: currency.to_owned(),
                maximum_microunits,
            },
            ProviderPriceBound::new(currency, maximum_microunits)?,
        )?)
    }

    #[test]
    fn provider_runtime_activation_grants_are_host_scoped_one_shot_and_manifest_bound()
    -> Result<(), Box<dyn Error>> {
        let endpoint = "https://provider.invalid/v2";
        let authorization = authorization(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("plugin.fixture|{endpoint}"),
        )])?;
        let binding = provider_binding()?;
        let outer_digest = [0x41; 32];
        let manifest = ProviderManifestAuthorizationV2::fixture(
            authorization.clone(),
            outer_digest,
            binding.clone(),
            provider_streaming_contract(),
        );
        let policy = ProviderPolicy::new(
            PROFILE_UUID.to_string(),
            ProviderMode::Enabled,
            [ProviderEndpoint::new("plugin.fixture", endpoint)?],
            std::iter::empty(),
        )?;
        let head = ProviderRequestHeadV2 {
            endpoint: endpoint.to_owned(),
            secret_id: None,
            method: ProviderHttpMethodV2::Post,
            headers: Vec::new(),
            declared_body_bytes: Some(0),
        };
        let source = ProviderRuntimeActivationGrantSource::new();
        let rightful = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412),
            session_generation: 4,
            invocation: 7,
            generation: 9,
        };
        source.insert(
            &rightful,
            activation_grant(&binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        for (context, expected) in [
            (
                WorkerProviderInvocationContext {
                    session_id: Uuid::from_u128(0x413),
                    ..rightful.clone()
                },
                PluginServiceError::ProviderRuntimeForeignSession,
            ),
            (
                WorkerProviderInvocationContext {
                    session_generation: 5,
                    ..rightful
                },
                PluginServiceError::ProviderRuntimeStaleSession,
            ),
            (
                WorkerProviderInvocationContext {
                    invocation: 8,
                    ..rightful
                },
                PluginServiceError::ProviderRuntimeForeignInvocation,
            ),
            (
                WorkerProviderInvocationContext {
                    generation: 10,
                    ..rightful
                },
                PluginServiceError::ProviderRuntimeStaleInvocation,
            ),
        ] {
            assert_eq!(
                source
                    .claim(&context, &CancellationToken::default())
                    .expect_err("foreign or stale claim must fail")
                    .to_string(),
                expected.to_string()
            );
        }
        let grant = source.claim(&rightful, &CancellationToken::default())?;
        let authority = preflight_activation(grant, &rightful, manifest)?.bind(&head, &policy)?;
        assert_eq!(authority.request_head, head);
        assert!(matches!(
            source.claim(&rightful, &CancellationToken::default()),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::RevokedHandle
            ))
        ));

        let replaced = ProviderRuntimeActivationGrantSource::new();
        replaced.insert(
            &rightful,
            activation_grant(&binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        let newer = WorkerProviderInvocationContext {
            session_generation: rightful.session_generation + 1,
            ..rightful
        };
        replaced.insert(
            &newer,
            activation_grant(&binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        assert!(matches!(
            replaced.claim(&rightful, &CancellationToken::default()),
            Err(PluginServiceError::ProviderRuntimeStaleSession)
        ));
        assert!(
            replaced
                .claim(&newer, &CancellationToken::default())
                .is_ok()
        );

        let changed_outer = ProviderManifestAuthorizationV2::fixture(
            authorization.clone(),
            [0x42; 32],
            binding.clone(),
            provider_streaming_contract(),
        );
        let changed_outer_source = ProviderRuntimeActivationGrantSource::new();
        changed_outer_source.insert(
            &rightful,
            activation_grant(&binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        assert!(matches!(
            preflight_activation(
                changed_outer_source.claim(&rightful, &CancellationToken::default())?,
                &rightful,
                changed_outer,
            ),
            Err(PluginServiceError::ProviderRuntimeAuthorityDenied)
        ));
        let changed_inner = ProviderManifestAuthorizationV2::fixture(
            authorization.fixture_with_digest_sha256("d".repeat(64)),
            outer_digest,
            binding.clone(),
            provider_streaming_contract(),
        );
        let changed_inner_source = ProviderRuntimeActivationGrantSource::new();
        changed_inner_source.insert(
            &rightful,
            activation_grant(&binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        assert!(matches!(
            preflight_activation(
                changed_inner_source.claim(&rightful, &CancellationToken::default())?,
                &rightful,
                changed_inner,
            ),
            Err(PluginServiceError::ProviderRuntimeAuthorityDenied)
        ));

        let cancelled = CancellationToken::default();
        let cancelled_source = ProviderRuntimeActivationGrantSource::new();
        cancelled_source.insert(
            &rightful,
            activation_grant(&binding, outer_digest)?,
            &cancelled,
        )?;
        cancelled.cancel();
        assert_eq!(
            cancelled_source
                .claim(&rightful, &CancellationToken::default())
                .expect_err("cancelled claim must fail")
                .to_string(),
            PluginServiceError::Cancelled.to_string()
        );
        let bind_cancellation = CancellationToken::default();
        let bind_cancelled_source = ProviderRuntimeActivationGrantSource::new();
        bind_cancelled_source.insert(
            &rightful,
            activation_grant(&binding, outer_digest)?,
            &bind_cancellation,
        )?;
        let bind_cancelled_grant =
            bind_cancelled_source.claim(&rightful, &CancellationToken::default())?;
        bind_cancellation.cancel();
        assert!(matches!(
            preflight_activation(
                bind_cancelled_grant,
                &rightful,
                ProviderManifestAuthorizationV2::fixture(
                    authorization,
                    outer_digest,
                    binding,
                    provider_streaming_contract(),
                ),
            ),
            Err(PluginServiceError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn provider_runtime_component_preflight_rejects_every_sealed_identity_mismatch()
    -> Result<(), Box<dyn Error>> {
        let endpoint = "https://provider.invalid/v2";
        let authorization = authorization(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("plugin.fixture|{endpoint}"),
        )])?;
        let binding = provider_binding()?;
        let outer_digest = [0x47; 32];
        let manifest = ProviderManifestAuthorizationV2::fixture(
            authorization.clone(),
            outer_digest,
            binding.clone(),
            provider_streaming_contract(),
        );
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x417),
            session_generation: 1,
            invocation: 1,
            generation: 1,
        };
        let context_mutations: [fn(&mut WorkerProviderInvocationContext); 4] = [
            |context| context.session_id = Uuid::from_u128(0x418),
            |context| context.session_generation += 1,
            |context| context.invocation += 1,
            |context| context.generation += 1,
        ];
        for mutate in context_mutations {
            let source = ProviderRuntimeActivationGrantSource::new();
            let cancellation = CancellationToken::default();
            source.insert(
                &context,
                activation_grant(&binding, outer_digest)?,
                &cancellation,
            )?;
            let deployment = worker_registry_deployment()?;
            let start = activation_start(&deployment, &binding);
            let mut supplied_context = context.clone();
            mutate(&mut supplied_context);
            let grant = source.claim(&context, &cancellation)?;
            assert!(matches!(
                grant.preflight_installed_component(
                    &supplied_context,
                    &deployment,
                    &start,
                    manifest.clone(),
                ),
                Err(PluginServiceError::ProviderRuntimeAuthorityDenied)
            ));
            let claim_key = (
                context.session_id,
                context.session_generation,
                context.invocation,
                context.generation,
            );
            assert!(
                !source
                    .state
                    .lock()
                    .activation_claims
                    .get(&claim_key)
                    .ok_or("activation claim disappeared")?
                    .load(Ordering::Acquire)
            );
            assert!(matches!(
                source.claim(&context, &cancellation),
                Err(PluginServiceError::ProviderStreamingContract(
                    ProviderStreamingContractError::RevokedHandle
                ))
            ));
        }

        let exact_source = ProviderRuntimeActivationGrantSource::new();
        let exact_cancellation = CancellationToken::default();
        exact_source.insert(
            &context,
            activation_grant(&binding, outer_digest)?,
            &exact_cancellation,
        )?;
        let exact_deployment = worker_registry_deployment()?;
        let exact_start = activation_start(&exact_deployment, &binding);
        exact_source
            .claim(&context, &exact_cancellation)?
            .preflight_installed_component(
                &context,
                &exact_deployment,
                &exact_start,
                manifest.clone(),
            )?;
        assert!(matches!(
            exact_source.claim(&context, &exact_cancellation),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::RevokedHandle
            ))
        ));

        let mutations: [fn(&mut ProviderRuntimeActivationGrant); 10] = [
            |grant| grant.registry_generation += 1,
            |grant| grant.registry_digest_sha256 = "0".repeat(64),
            |grant| grant.component_generation += 1,
            |grant| grant.component_digest_sha256 = "0".repeat(64),
            |grant| grant.provider_manifest_sha256 = "0".repeat(64),
            |grant| grant.authorization_generation_sha256 = "0".repeat(64),
            |grant| grant.binding_generation += 1,
            |grant| grant.binding_set_sha256 = "0".repeat(64),
            |grant| grant.node_id = "node.changed".to_owned(),
            |grant| grant.compiled_plan_sha256 = "0".repeat(64),
        ];
        for mutate in mutations {
            let source = ProviderRuntimeActivationGrantSource::new();
            let cancellation = CancellationToken::default();
            let mut grant = activation_grant(&binding, outer_digest)?;
            mutate(&mut grant);
            source.insert(&context, grant, &cancellation)?;
            let grant = source.claim(&context, &cancellation)?;
            assert!(matches!(
                preflight_activation(grant, &context, manifest.clone()),
                Err(PluginServiceError::ProviderRuntimeAuthorityDenied)
            ));
            let claim_key = (
                context.session_id,
                context.session_generation,
                context.invocation,
                context.generation,
            );
            assert!(
                !source
                    .state
                    .lock()
                    .activation_claims
                    .get(&claim_key)
                    .ok_or("activation claim disappeared")?
                    .load(Ordering::Acquire)
            );
            assert!(matches!(
                source.claim(&context, &cancellation),
                Err(PluginServiceError::ProviderStreamingContract(
                    ProviderStreamingContractError::RevokedHandle
                ))
            ));
        }

        let start_mutations: [fn(&mut NativeProviderWorkerSessionStart); 12] = [
            |start| start.registry_generation += 1,
            |start| start.registry_digest_sha256 = "0".repeat(64),
            |start| start.extension_id = "fixture.changed".to_owned(),
            |start| start.extension_version = "2.0.0".to_owned(),
            |start| start.plugin_identifier = "plugin.changed".to_owned(),
            |start| start.plugin_version = "2.0.0".to_owned(),
            |start| start.manifest_digest_sha256 = "0".repeat(64),
            |start| start.component_digest_sha256 = "0".repeat(64),
            |start| start.authorization_generation_sha256 = "0".repeat(64),
            |start| start.binding_set_sha256 = "0".repeat(64),
            |start| start.node_id = "node.changed".to_owned(),
            |start| start.compiled_plan_sha256 = "0".repeat(64),
        ];
        for mutate in start_mutations {
            let source = ProviderRuntimeActivationGrantSource::new();
            let cancellation = CancellationToken::default();
            source.insert(
                &context,
                activation_grant(&binding, outer_digest)?,
                &cancellation,
            )?;
            let deployment = worker_registry_deployment()?;
            let mut start = activation_start(&deployment, &binding);
            mutate(&mut start);
            let grant = source.claim(&context, &cancellation)?;
            assert!(matches!(
                grant.preflight_installed_component(
                    &context,
                    &deployment,
                    &start,
                    manifest.clone()
                ),
                Err(PluginServiceError::ProviderRuntimeAuthorityDenied)
            ));
            assert!(
                !source
                    .state
                    .lock()
                    .activation_claims
                    .values()
                    .next()
                    .ok_or("activation claim disappeared")?
                    .load(Ordering::Acquire)
            );
        }

        let later_deployment = worker_registry_deployment_at(8)?;
        let source = ProviderRuntimeActivationGrantSource::new();
        let cancellation = CancellationToken::default();
        source.insert(
            &context,
            activation_grant_for_deployment(&binding, outer_digest, &later_deployment)?,
            &cancellation,
        )?;
        assert!(matches!(
            preflight_activation(source.claim(&context, &cancellation)?, &context, manifest),
            Err(PluginServiceError::ProviderRuntimeAuthorityDenied)
        ));

        let mut changed_binding = binding.clone();
        changed_binding.bindings[0].node_id = "node.changed".to_owned();
        changed_binding.bindings_sha256 = changed_binding.canonical_bindings_sha256()?;
        let changed_binding_manifest = ProviderManifestAuthorizationV2::fixture(
            authorization.clone(),
            outer_digest,
            changed_binding.clone(),
            provider_streaming_contract(),
        );
        let source = ProviderRuntimeActivationGrantSource::new();
        source.insert(
            &context,
            activation_grant(&changed_binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        assert!(matches!(
            preflight_activation(
                source.claim(&context, &CancellationToken::default())?,
                &context,
                changed_binding_manifest,
            ),
            Err(PluginServiceError::ProviderRuntimeAuthorityDenied)
        ));

        let mut changed_streaming = provider_streaming_contract();
        changed_streaming.maximum_progress_total -= 1;
        let changed_streaming_manifest = ProviderManifestAuthorizationV2::fixture(
            authorization,
            [0x48; 32],
            binding.clone(),
            changed_streaming,
        );
        let source = ProviderRuntimeActivationGrantSource::new();
        source.insert(
            &context,
            activation_grant(&binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        assert!(matches!(
            preflight_activation(
                source.claim(&context, &CancellationToken::default())?,
                &context,
                changed_streaming_manifest,
            ),
            Err(PluginServiceError::ProviderRuntimeAuthorityDenied)
        ));

        let dropped_source = ProviderRuntimeActivationGrantSource::new();
        dropped_source.insert(
            &context,
            activation_grant(&binding, outer_digest)?,
            &CancellationToken::default(),
        )?;
        drop(dropped_source.claim(&context, &CancellationToken::default())?);
        assert!(
            !dropped_source
                .state
                .lock()
                .activation_claims
                .values()
                .next()
                .ok_or("dropped activation claim disappeared")?
                .load(Ordering::Acquire)
        );
        Ok(())
    }

    #[test]
    fn provider_runtime_activation_grant_capacity_and_cleanup_are_atomic()
    -> Result<(), Box<dyn Error>> {
        let binding = provider_binding()?;
        let service = ProviderRuntimeStreamService::new();
        let source = service.activation_grants();
        let cancellation = CancellationToken::default();
        for invocation in 1..=u64::try_from(MAX_PROVIDER_RUNTIME_SESSIONS)? {
            source.insert(
                &WorkerProviderInvocationContext {
                    session_id: Uuid::from_u128(0x500 + u128::from(invocation)),
                    session_generation: 1,
                    invocation,
                    generation: 1,
                },
                activation_grant(&binding, [0x51; 32])?,
                &cancellation,
            )?;
        }
        let overflow = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x999),
            session_generation: 1,
            invocation: 999,
            generation: 1,
        };
        assert!(matches!(
            source.insert(
                &overflow,
                activation_grant(&binding, [0x51; 32])?,
                &cancellation,
            ),
            Err(PluginServiceError::ProviderSessionUnavailable)
        ));
        let first = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x501),
            session_generation: 1,
            invocation: 1,
            generation: 1,
        };
        assert!(source.claim(&first, &cancellation).is_ok());
        service.revoke_all();
        assert!(matches!(
            source.claim(
                &WorkerProviderInvocationContext {
                    session_id: Uuid::from_u128(0x502),
                    session_generation: 1,
                    invocation: 2,
                    generation: 1,
                },
                &cancellation,
            ),
            Err(PluginServiceError::ProviderRuntimeForeignSession)
        ));
        Ok(())
    }

    #[test]
    fn provider_runtime_service_owns_one_grant_and_legacy_state_table() -> Result<(), Box<dyn Error>>
    {
        let service = ProviderRuntimeStreamService::new();
        let clone = service.clone();
        let source = service.activation_grants();
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_900),
            session_generation: 3,
            invocation: 7,
            generation: 11,
        };
        let binding = provider_binding()?;
        service.register_activation(
            &context,
            activation_grant(&binding, [0x61; 32])?,
            &CancellationToken::default(),
        )?;
        assert!(
            clone
                .claim_activation(&context, &CancellationToken::default())
                .is_ok()
        );
        assert!(matches!(
            source.claim(&context, &CancellationToken::default()),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::RevokedHandle
            ))
        ));

        let old_main = WorkerProviderStreamHandle {
            session_id: context.session_id,
            session_generation: context.session_generation,
            invocation: context.invocation,
            slot: 1,
            generation: context.generation,
        };
        let old_upload = WorkerProviderStreamHandle {
            slot: 2,
            ..old_main
        };
        let (old_authority, _) = bound_streaming_authority(&binding, [0x61; 32])?;
        {
            let mut state = service.state.lock();
            let key = (
                context.session_id,
                context.session_generation,
                context.invocation,
                context.generation,
            );
            state.invocation_bindings.insert(key, 71);
            state.next_slots.insert(key, 2);
            state.handles.insert(
                old_main,
                ProviderStreamHandleV2 {
                    invocation: 71,
                    slot: 1,
                    generation: context.generation,
                },
            );
            state.owner.begin_streaming(
                old_authority,
                ProviderInvocationContextV2 {
                    invocation: 71,
                    generation: context.generation,
                },
                ProviderStreamHandleV2 {
                    invocation: 71,
                    slot: 1,
                    generation: context.generation,
                },
                &CancellationToken::default(),
            )?;
            state.handles.insert(
                old_upload,
                ProviderStreamHandleV2 {
                    invocation: 71,
                    slot: 2,
                    generation: context.generation,
                },
            );
            state.main_handles.insert(old_main);
            state.upload_parents.insert(old_upload, old_main);
        }
        let replacement = WorkerProviderInvocationContext {
            session_generation: context.session_generation + 1,
            generation: context.generation + 1,
            ..context
        };
        service.register_activation(
            &replacement,
            activation_grant(&binding, [0x61; 32])?,
            &CancellationToken::default(),
        )?;
        {
            let state = service.state.lock();
            assert!(!state.handles.contains_key(&old_main));
            assert!(!state.handles.contains_key(&old_upload));
            assert!(!state.main_handles.contains(&old_main));
            assert!(!state.upload_parents.contains_key(&old_upload));
            assert!(state.invocation_bindings.is_empty());
            assert!(state.next_slots.is_empty());
            assert!(state.owner.is_empty());
        }
        assert!(matches!(
            clone.claim_activation(&context, &CancellationToken::default()),
            Err(PluginServiceError::ProviderRuntimeStaleSession)
        ));
        let replacement_main = WorkerProviderStreamHandle {
            session_id: replacement.session_id,
            session_generation: replacement.session_generation,
            invocation: replacement.invocation,
            slot: 1,
            generation: replacement.generation,
        };
        let (replacement_authority, _) = bound_streaming_authority(&binding, [0x61; 32])?;
        {
            let mut state = service.state.lock();
            let key = (
                replacement.session_id,
                replacement.session_generation,
                replacement.invocation,
                replacement.generation,
            );
            state.invocation_bindings.insert(key, 72);
            state.next_slots.insert(key, 1);
            state.handles.insert(
                replacement_main,
                ProviderStreamHandleV2 {
                    invocation: 72,
                    slot: 1,
                    generation: replacement.generation,
                },
            );
            state.main_handles.insert(replacement_main);
            state.owner.begin_streaming(
                replacement_authority,
                ProviderInvocationContextV2 {
                    invocation: 72,
                    generation: replacement.generation,
                },
                ProviderStreamHandleV2 {
                    invocation: 72,
                    slot: 1,
                    generation: replacement.generation,
                },
                &CancellationToken::default(),
            )?;
        }
        let newer_invocation_generation = WorkerProviderInvocationContext {
            generation: replacement.generation + 1,
            ..replacement
        };
        service.register_activation(
            &newer_invocation_generation,
            activation_grant(&binding, [0x61; 32])?,
            &CancellationToken::default(),
        )?;
        {
            let state = service.state.lock();
            assert!(!state.handles.contains_key(&replacement_main));
            assert!(!state.main_handles.contains(&replacement_main));
            assert!(state.invocation_bindings.is_empty());
            assert!(state.next_slots.is_empty());
            assert!(state.owner.is_empty());
        }
        assert!(matches!(
            clone.claim_activation(&replacement, &CancellationToken::default()),
            Err(PluginServiceError::ProviderRuntimeStaleInvocation)
        ));
        assert!(
            clone
                .claim_activation(&newer_invocation_generation, &CancellationToken::default())
                .is_ok()
        );
        service.revoke_all();
        assert!(clone.is_empty());
        Ok(())
    }

    fn http_test_context(invocation: u64) -> WorkerProviderInvocationContext {
        WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_aa77),
            session_generation: 2,
            invocation,
            generation: 4,
        }
    }

    fn bound_http_test_service(
        context: &WorkerProviderInvocationContext,
        cancellation: &CancellationToken,
    ) -> Result<
        (
            ProviderRuntimeStreamService,
            ProviderRuntimeAuthorityInput,
            WorkerProviderRequestHead,
        ),
        Box<dyn Error>,
    > {
        let mut contract = provider_streaming_contract();
        contract.methods.push(ProviderHttpMethodV2::Get);
        contract.methods.sort();
        bound_streaming_service_with_contract(context, cancellation, Some(0), contract)
    }

    fn http_test_policy() -> Result<ProviderPolicy, Box<dyn Error>> {
        Ok(ProviderPolicy::new(
            PROFILE_UUID.to_string(),
            ProviderMode::Enabled,
            [ProviderEndpoint::new(
                "plugin.fixture",
                "https://provider.invalid/v2",
            )?],
            std::iter::empty(),
        )?)
    }

    fn complete_http_test_request(
        service: &ProviderRuntimeStreamService,
        invocation: &ProviderRuntimeHttpInvocation,
        handle: WorkerProviderStreamHandle,
        seal: ProviderRuntimeHttpReceiptSeal<'_>,
    ) -> Result<ProviderRuntimeHttpRequestCompletion, PluginServiceError> {
        service.write_request_chunk(WorkerProviderRequestChunk {
            handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        service.prepare_streaming_actuation(handle)?;
        service.accept_wait(
            WorkerProviderWaitRequest {
                handle,
                after_sequence: None,
                timeout_milliseconds: 100,
            },
            WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                handle,
                sequence: 0,
                event: WorkerProviderResponseFrameEvent::Head(
                    sim_types::WorkerProviderResponseHead {
                        status: 200,
                        headers: Vec::new(),
                    },
                ),
            }),
        )?;
        service.accept_wait(
            WorkerProviderWaitRequest {
                handle,
                after_sequence: Some(0),
                timeout_milliseconds: 100,
            },
            WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                handle,
                sequence: 1,
                event: WorkerProviderResponseFrameEvent::Terminal(
                    WorkerProviderTerminal::Completed(b"http-terminal".to_vec()),
                ),
            }),
        )?;
        service.complete_http_request(invocation, handle, seal)
    }

    #[test]
    fn provider_http_invocation_retains_one_activation_across_checked_requests()
    -> Result<(), Box<dyn Error>> {
        let context = http_test_context(1);
        let cancellation = CancellationToken::default();
        let (service, authority, mut head) = bound_http_test_service(&context, &cancellation)?;
        let activation_ordinal = authority.request_ordinal;
        let expected_scope = http_invocation_scope(&authority);
        let (invocation, mut handle) = service.begin_http_invocation(authority, head.clone(), 3)?;
        assert_eq!(invocation.identity(), &expected_scope);
        assert!(service.claim_activation(&context, &cancellation).is_err());
        let origin = Instant::now();
        let issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x85; 32], origin)?;
        let verifier = issuer.verifier()?;
        let policy = http_test_policy()?;
        let mut transcript = Sha256::new();
        transcript.update(b"zed-sim-provider-http-transcript-v2\0");
        let mut idempotency_identities = BTreeSet::new();
        for ordinal in 1..=3 {
            let predecessor = complete_http_test_request(
                &service,
                &invocation,
                handle,
                ProviderRuntimeHttpReceiptSeal {
                    issuer: &issuer,
                    verifier: &verifier,
                    issued_at: origin,
                    expires_at: origin + Duration::from_secs(30),
                    nonce: [ordinal as u8; 32],
                    verification_time: origin,
                },
            )?;
            assert_eq!(predecessor.ordinal(), ordinal);
            assert_eq!(
                predecessor.receipt().identity().request_ordinal,
                activation_ordinal
            );
            assert!(
                idempotency_identities.insert(
                    predecessor
                        .receipt()
                        .identity()
                        .idempotency_identity_sha256
                        .clone()
                )
            );
            transcript.update(ordinal.to_le_bytes());
            transcript.update(Sha256::digest(predecessor.receipt().bytes()));
            assert!(service.state.lock().owner.is_empty());
            service.check_http_invocation_cancelled(&invocation)?;
            assert_eq!(service.state.lock().http_invocations.len(), 1);
            assert!(service.check_cancelled(handle).is_err());
            if ordinal == 3 {
                let completion =
                    service.finish_http_invocation(invocation, predecessor, &verifier, origin)?;
                assert_eq!(completion.identity(), &expected_scope);
                assert_eq!(completion.request_count(), 3);
                assert_eq!(
                    completion.transcript_sha256(),
                    format!("{:x}", transcript.finalize())
                );
                assert_eq!(
                    completion.last_request().identity().method,
                    ProviderHttpMethodV2::Get
                );
                assert!(service.state.lock().http_invocations.is_empty());
                service.revoke_all();
                assert!(service.is_empty());
                return Ok(());
            }
            head.method = WorkerProviderHttpMethod::Get;
            handle = service.continue_http_request(
                &invocation,
                &predecessor,
                ProviderRuntimeHttpContinuation {
                    ordinal: ordinal + 1,
                    request_head: head.clone(),
                    provider_policy: &policy,
                    verifier: &verifier,
                    now: origin,
                },
            )?;
        }
        Err("invocation did not finalize".into())
    }

    #[test]
    fn provider_http_invocation_rejects_invalid_predecessor_and_bounds_atomically()
    -> Result<(), Box<dyn Error>> {
        // Each rejected continuation must revoke both its retained authority and all handles.
        for case in 0..9 {
            let context = http_test_context(10 + case);
            let cancellation = CancellationToken::default();
            let (service, authority, mut head) = bound_streaming_service(&context, &cancellation)?;
            let (invocation, handle) = service.begin_http_invocation(
                authority,
                head.clone(),
                if case == 2 { 1 } else { 3 },
            )?;
            let origin = Instant::now();
            let issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x86; 32], origin)?;
            let verifier = issuer.verifier()?;
            let foreign_issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x87; 32], origin)?;
            let foreign_verifier = foreign_issuer.verifier()?;
            let policy = http_test_policy()?;
            let predecessor = complete_http_test_request(
                &service,
                &invocation,
                handle,
                ProviderRuntimeHttpReceiptSeal {
                    issuer: &issuer,
                    verifier: &verifier,
                    issued_at: origin,
                    expires_at: origin + Duration::from_secs(30),
                    nonce: [0x88; 32],
                    verification_time: origin,
                },
            )?;
            let denied_policy = ProviderPolicy::new(
                PROFILE_UUID.to_string(),
                ProviderMode::Disabled,
                [ProviderEndpoint::new(
                    "plugin.fixture",
                    "https://provider.invalid/v2",
                )?],
                std::iter::empty(),
            )?;
            let ordinal = match case {
                0 => 1,
                1 => 3,
                _ => 2,
            };
            if case == 3 {
                head.endpoint = "https://other.invalid/v2".to_owned();
            }
            if case == 4 {
                head.secret_id = Some("undeclared.secret".to_owned());
            }
            if case == 6 {
                cancellation.cancel();
            }
            if case == 7 {
                head.method = WorkerProviderHttpMethod::Get;
            }
            let result = service.continue_http_request(
                &invocation,
                &predecessor,
                ProviderRuntimeHttpContinuation {
                    ordinal,
                    request_head: head,
                    provider_policy: if case == 8 { &denied_policy } else { &policy },
                    verifier: if case == 5 {
                        &foreign_verifier
                    } else {
                        &verifier
                    },
                    now: origin,
                },
            );
            assert!(result.is_err(), "case {case}");
            let state = service.state.lock();
            assert!(state.http_invocations.is_empty(), "case {case}");
            assert!(state.owner.is_empty());
            assert!(state.main_handles.is_empty());
            assert!(state.handles.is_empty());
        }
        Ok(())
    }

    #[test]
    fn provider_http_invocation_replay_expiration_and_early_finish_revoke()
    -> Result<(), Box<dyn Error>> {
        for case in 0..3 {
            let context = http_test_context(30 + case);
            let cancellation = CancellationToken::default();
            let (service, authority, head) = bound_streaming_service(&context, &cancellation)?;
            let (invocation, handle) = service.begin_http_invocation(authority, head.clone(), 3)?;
            let origin = Instant::now();
            let issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x89; 32], origin)?;
            let verifier = issuer.verifier()?;
            let policy = http_test_policy()?;
            let predecessor = complete_http_test_request(
                &service,
                &invocation,
                handle,
                ProviderRuntimeHttpReceiptSeal {
                    issuer: &issuer,
                    verifier: &verifier,
                    issued_at: origin,
                    expires_at: origin + Duration::from_secs(30),
                    nonce: [0x90; 32],
                    verification_time: origin,
                },
            )?;
            if case == 0 {
                assert!(
                    service
                        .continue_http_request(
                            &invocation,
                            &predecessor,
                            ProviderRuntimeHttpContinuation {
                                ordinal: 2,
                                request_head: head,
                                provider_policy: &policy,
                                verifier: &verifier,
                                now: origin + Duration::from_secs(30),
                            }
                        )
                        .is_err()
                );
            } else {
                let next = service.continue_http_request(
                    &invocation,
                    &predecessor,
                    ProviderRuntimeHttpContinuation {
                        ordinal: 2,
                        request_head: head.clone(),
                        provider_policy: &policy,
                        verifier: &verifier,
                        now: origin,
                    },
                )?;
                if case == 1 {
                    assert!(
                        service
                            .continue_http_request(
                                &invocation,
                                &predecessor,
                                ProviderRuntimeHttpContinuation {
                                    ordinal: 2,
                                    request_head: head,
                                    provider_policy: &policy,
                                    verifier: &verifier,
                                    now: origin,
                                }
                            )
                            .is_err()
                    );
                } else {
                    assert!(
                        service
                            .finish_http_invocation(invocation, predecessor, &verifier, origin)
                            .is_err()
                    );
                }
                assert!(service.check_cancelled(next).is_err());
            }
            let state = service.state.lock();
            assert!(state.http_invocations.is_empty());
            assert!(state.owner.is_empty());
            assert!(state.handles.is_empty());
        }
        Ok(())
    }

    #[test]
    fn provider_http_invocation_rejects_foreign_receipts_and_bad_seals()
    -> Result<(), Box<dyn Error>> {
        let origin = Instant::now();
        let issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x91; 32], origin)?;
        let verifier = issuer.verifier()?;
        let foreign_issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x92; 32], origin)?;
        let foreign_verifier = foreign_issuer.verifier()?;
        let cancellation = CancellationToken::default();
        let (first, authority, head) =
            bound_streaming_service(&http_test_context(41), &cancellation)?;
        let (first_invocation, first_handle) =
            first.begin_http_invocation(authority, head.clone(), 2)?;
        let (second, authority, second_head) =
            bound_streaming_service(&http_test_context(42), &cancellation)?;
        let (second_invocation, second_handle) =
            second.begin_http_invocation(authority, second_head, 2)?;
        let first_predecessor = complete_http_test_request(
            &first,
            &first_invocation,
            first_handle,
            ProviderRuntimeHttpReceiptSeal {
                issuer: &issuer,
                verifier: &verifier,
                issued_at: origin,
                expires_at: origin + Duration::from_secs(30),
                nonce: [0x93; 32],
                verification_time: origin,
            },
        )?;
        let second_predecessor = complete_http_test_request(
            &second,
            &second_invocation,
            second_handle,
            ProviderRuntimeHttpReceiptSeal {
                issuer: &issuer,
                verifier: &verifier,
                issued_at: origin,
                expires_at: origin + Duration::from_secs(30),
                nonce: [0x94; 32],
                verification_time: origin,
            },
        )?;
        let policy = http_test_policy()?;
        assert!(
            first
                .continue_http_request(
                    &first_invocation,
                    &second_predecessor,
                    ProviderRuntimeHttpContinuation {
                        ordinal: 2,
                        request_head: head,
                        provider_policy: &policy,
                        verifier: &verifier,
                        now: origin,
                    }
                )
                .is_err()
        );
        assert!(
            second
                .check_http_invocation_cancelled(&second_invocation)
                .is_ok()
        );
        assert!(
            second
                .finish_http_invocation(first_invocation, first_predecessor, &verifier, origin)
                .is_err()
        );
        drop(second_invocation);
        assert!(second.state.lock().http_invocations.is_empty());
        let (service, authority, head) =
            bound_streaming_service(&http_test_context(43), &cancellation)?;
        let (invocation, handle) = service.begin_http_invocation(authority, head, 2)?;
        assert!(
            complete_http_test_request(
                &service,
                &invocation,
                handle,
                ProviderRuntimeHttpReceiptSeal {
                    issuer: &issuer,
                    verifier: &foreign_verifier,
                    issued_at: origin,
                    expires_at: origin + Duration::from_secs(30),
                    nonce: [0x95; 32],
                    verification_time: origin,
                }
            )
            .is_err()
        );
        assert!(service.state.lock().owner.is_empty());
        assert!(service.state.lock().handles.is_empty());
        assert!(service.state.lock().http_invocations.is_empty());
        Ok(())
    }

    #[test]
    fn provider_http_invocation_drop_and_context_replacement_revoke_between_requests()
    -> Result<(), Box<dyn Error>> {
        for completed in [false, true] {
            for replacement in [false, true] {
                let context = http_test_context(51);
                let cancellation = CancellationToken::default();
                let (service, authority, head) = bound_streaming_service(&context, &cancellation)?;
                let (invocation, handle) = service.begin_http_invocation(authority, head, 2)?;
                if completed {
                    let origin = Instant::now();
                    let issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x96; 32], origin)?;
                    let verifier = issuer.verifier()?;
                    complete_http_test_request(
                        &service,
                        &invocation,
                        handle,
                        ProviderRuntimeHttpReceiptSeal {
                            issuer: &issuer,
                            verifier: &verifier,
                            issued_at: origin,
                            expires_at: origin + Duration::from_secs(30),
                            nonce: [0x97; 32],
                            verification_time: origin,
                        },
                    )?;
                }
                if replacement {
                    let next_context = WorkerProviderInvocationContext {
                        generation: context.generation + 1,
                        ..context
                    };
                    let binding = provider_binding()?;
                    service.register_activation(
                        &next_context,
                        activation_grant(&binding, [0x81; 32])?,
                        &CancellationToken::default(),
                    )?;
                    assert!(
                        service
                            .check_http_invocation_cancelled(&invocation)
                            .is_err()
                    );
                }
                drop(invocation);
                let state = service.state.lock();
                assert!(state.http_invocations.is_empty());
                assert!(state.owner.is_empty());
                assert!(state.handles.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn provider_runtime_stream_service_maps_worker_frames_without_actuation()
    -> Result<(), Box<dyn Error>> {
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_aa01),
            session_generation: 2,
            invocation: 3,
            generation: 4,
        };
        let cancellation = CancellationToken::default();
        let (service, authority, head) = bound_streaming_service(&context, &cancellation)?;
        let handle = service.start_request(authority, head)?;
        assert_eq!(handle.session_id, context.session_id);
        assert_eq!(handle.session_generation, context.session_generation);
        assert_eq!(handle.invocation, context.invocation);
        assert_eq!(handle.generation, context.generation);
        assert_ne!(handle.slot, 0);

        service.write_request_chunk(WorkerProviderRequestChunk {
            handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        let upload_bytes = b"upload".to_vec();
        let upload = service.start_upload(WorkerProviderUploadRequest {
            handle,
            port_id: "image.input".to_owned(),
            media_type: "application/octet-stream".to_owned(),
            byte_length: u64::try_from(upload_bytes.len())?,
            content_sha256: format!("{:x}", Sha256::digest(&upload_bytes)),
        })?;
        assert_ne!(upload.slot, handle.slot);
        service.write_upload_chunk(WorkerProviderRequestChunk {
            handle: upload,
            sequence: 0,
            bytes: upload_bytes,
            end: true,
        })?;
        service.accept_upload_completion(
            WorkerProviderUploadCompletionRequest { handle: upload },
            WorkerProviderUploadCompletion {
                download_url: "https://fixture.invalid/storage/upload.bin".to_owned(),
            },
        )?;

        let proposal = service.prepare_streaming_actuation(handle)?;
        assert_eq!(proposal.request_head().method, ProviderHttpMethodV2::Post);
        assert_eq!(proposal.request_head().headers.len(), 1);
        assert_eq!(proposal.accepted_cost_microunits(), 0);
        service.accept_wait(
            WorkerProviderWaitRequest {
                handle,
                after_sequence: None,
                timeout_milliseconds: 100,
            },
            WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                handle,
                sequence: 0,
                event: WorkerProviderResponseFrameEvent::Head(
                    sim_types::WorkerProviderResponseHead {
                        status: 200,
                        headers: Vec::new(),
                    },
                ),
            }),
        )?;
        let progress = service
            .report_progress(
                WorkerProviderProgress {
                    handle,
                    sequence: 0,
                    completed: 1,
                    total: 1,
                    message: Some("done".to_owned()),
                },
                Instant::now(),
            )?
            .ok_or("final progress must be projected")?;
        assert_eq!(progress.handle, handle);
        assert_eq!(progress.message.as_deref(), Some("done"));
        service.accept_wait(
            WorkerProviderWaitRequest {
                handle,
                after_sequence: Some(0),
                timeout_milliseconds: 100,
            },
            WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                handle,
                sequence: 1,
                event: WorkerProviderResponseFrameEvent::Terminal(
                    WorkerProviderTerminal::Completed(b"terminal-receipt".to_vec()),
                ),
            }),
        )?;
        let origin = Instant::now();
        let issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x82; 32], origin)?;
        let receipt = service.finish_streaming(
            handle,
            &issuer,
            origin,
            origin + Duration::from_secs(30),
            [0x83; 32],
        )?;
        assert_eq!(
            receipt.identity().terminal_receipt_sha256,
            provider_terminal_completed_receipt_sha256(b"terminal-receipt")
        );
        assert!(matches!(
            service.check_cancelled(handle),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::RevokedHandle
            ))
        ));
        assert!(matches!(
            service.check_cancelled(upload),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::RevokedHandle
            ))
        ));
        assert!(matches!(
            service.check_cancelled(WorkerProviderStreamHandle {
                slot: upload.slot + 1,
                ..handle
            }),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidHandle
            ))
        ));
        let state = service.state.lock();
        assert!(state.owner.is_empty());
        assert!(state.handles.is_empty());
        assert!(state.main_handles.is_empty());
        assert!(state.upload_parents.is_empty());
        assert!(state.cancellations.is_empty());
        Ok(())
    }

    #[test]
    fn provider_runtime_stream_retry_revokes_old_handles_and_retains_completed_uploads()
    -> Result<(), Box<dyn Error>> {
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_aa11),
            session_generation: 2,
            invocation: 3,
            generation: 4,
        };
        let (service, handle) = started_streaming_service(&context, &CancellationToken::default())?;
        let upload = complete_streaming_upload(&service, handle, b"upload")?;
        let first_proposal = service.prepare_streaming_actuation(handle)?;
        service.accept_wait(
            WorkerProviderWaitRequest {
                handle,
                after_sequence: None,
                timeout_milliseconds: 100,
            },
            WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                handle,
                sequence: 0,
                event: WorkerProviderResponseFrameEvent::Head(
                    sim_types::WorkerProviderResponseHead {
                        status: 503,
                        headers: Vec::new(),
                    },
                ),
            }),
        )?;
        let retry_handle = service.retry_request(handle)?;
        assert_ne!(retry_handle, handle);
        for revoked in [handle, upload] {
            assert!(matches!(
                service.check_cancelled(revoked),
                Err(PluginServiceError::ProviderStreamingContract(
                    ProviderStreamingContractError::RevokedHandle
                ))
            ));
        }
        assert_eq!(service.check_cancelled(retry_handle), Ok(()));
        assert!(matches!(
            service.prepare_streaming_actuation(retry_handle),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidOrder
            ))
        ));
        service.write_request_chunk(WorkerProviderRequestChunk {
            handle: retry_handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        let retry_proposal = service.prepare_streaming_actuation(retry_handle)?;
        assert_eq!(
            retry_proposal.ordered_uploads_sha256(),
            first_proposal.ordered_uploads_sha256()
        );
        assert_eq!(retry_proposal.accepted_cost_microunits(), 0);
        service.revoke_stream(retry_handle);
        Ok(())
    }

    #[test]
    fn provider_runtime_stream_service_consumes_only_current_canonical_activation()
    -> Result<(), Box<dyn Error>> {
        let binding = provider_binding()?;
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_aa02),
            session_generation: 5,
            invocation: 7,
            generation: 9,
        };

        let cancelled = CancellationToken::default();
        let (service, authority, head) = bound_streaming_service(&context, &cancelled)?;
        cancelled.cancel();
        assert_eq!(
            service.start_request(authority, head),
            Err(PluginServiceError::Cancelled)
        );
        assert!(service.state.lock().owner.is_empty());

        let cancellation = CancellationToken::default();
        let (service, authority, head) = bound_streaming_service(&context, &cancellation)?;
        let newer_session = WorkerProviderInvocationContext {
            session_generation: context.session_generation + 1,
            ..context
        };
        service.register_activation(
            &newer_session,
            activation_grant(&binding, [0x81; 32])?,
            &CancellationToken::default(),
        )?;
        assert_eq!(
            service.start_request(authority, head),
            Err(PluginServiceError::ProviderRuntimeStaleSession)
        );
        assert!(service.state.lock().owner.is_empty());

        let (service, authority, head) =
            bound_streaming_service(&context, &CancellationToken::default())?;
        let newer_invocation = WorkerProviderInvocationContext {
            generation: context.generation + 1,
            ..context
        };
        service.register_activation(
            &newer_invocation,
            activation_grant(&binding, [0x81; 32])?,
            &CancellationToken::default(),
        )?;
        assert_eq!(
            service.start_request(authority, head),
            Err(PluginServiceError::ProviderRuntimeStaleInvocation)
        );
        assert!(service.state.lock().owner.is_empty());

        let (service, authority, head) =
            bound_streaming_service(&context, &CancellationToken::default())?;
        service.revoke_invocation(&context)?;
        assert!(matches!(
            service.start_request(authority, head),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::RevokedHandle
            ))
        ));
        assert!(service.state.lock().owner.is_empty());

        let (service, authority, mut head) =
            bound_streaming_service(&context, &CancellationToken::default())?;
        head.headers[0].value = "changed".to_owned();
        assert!(matches!(
            service.start_request(authority, head),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidRequestAuthority
            ))
        ));
        assert!(service.state.lock().owner.is_empty());
        Ok(())
    }

    #[test]
    fn provider_runtime_stream_service_verifies_cost_offline_before_sealing_proposal()
    -> Result<(), Box<dyn Error>> {
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_aa03),
            session_generation: 1,
            invocation: 2,
            generation: 3,
        };
        let cancellation = CancellationToken::default();
        let (service, authority, head) = bound_streaming_service(&context, &cancellation)?;
        let handle = service.start_request(authority, head)?;
        service.write_request_chunk(WorkerProviderRequestChunk {
            handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        complete_streaming_upload(&service, handle, b"priced-upload")?;
        let request = WorkerProviderCostRequest {
            handle,
            operation: "provider.cost".to_owned(),
            currency: "USD".to_owned(),
            maximum_microunits: 5,
        };
        let price = ProviderPriceBound::new("USD", 5)?;
        let nonce = ProviderCostNonce::new([0x84; 32])?;
        let now = Instant::now();
        let issuer = ProviderCostAcceptanceIssuer::from_seed([0x85; 32], now)?;
        let scope = service.prepare_cost_acceptance(&request, price.clone())?;
        let acceptance = issuer.issue(scope.clone(), now, now + Duration::from_secs(30), nonce)?;
        let replay = issuer.issue(scope.clone(), now, now + Duration::from_secs(30), nonce)?;
        let verifier = issuer.verifier()?;
        let response = service.accept_streaming_cost(
            request.clone(),
            ProviderCostAuthorization::new(price.clone(), nonce, acceptance)?,
            &verifier,
            now,
        )?;
        assert!(response.accepted);
        assert_eq!(response.approved_microunits, 5);
        assert!(!response.receipt.is_empty());
        assert_eq!(
            service.accept_streaming_cost(
                request.clone(),
                ProviderCostAuthorization::new(ProviderPriceBound::new("USD", 5)?, nonce, replay,)?,
                &verifier,
                now,
            ),
            Err(PluginServiceError::ProviderCostAcceptanceReused)
        );
        let proposal = service.prepare_streaming_actuation(handle)?;
        assert_eq!(proposal.accepted_cost_microunits(), 5);
        assert!(matches!(
            service.deny_streaming_cost(request.clone()),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidOrder
            ))
        ));
        assert!(service.check_cancelled(handle).is_ok());
        service.revoke_stream(handle);
        assert!(service.state.lock().owner.is_empty());

        let (reordered_service, reordered_authority, reordered_head) =
            bound_streaming_service(&context, &CancellationToken::default())?;
        reordered_service.state.lock().next_internal_invocation = 100;
        let reordered_handle =
            reordered_service.start_request(reordered_authority, reordered_head)?;
        reordered_service.write_request_chunk(WorkerProviderRequestChunk {
            handle: reordered_handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        complete_streaming_upload(&reordered_service, reordered_handle, b"priced-upload")?;
        let reordered_request = WorkerProviderCostRequest {
            handle: reordered_handle,
            ..request.clone()
        };
        let reordered_scope =
            reordered_service.prepare_cost_acceptance(&reordered_request, price.clone())?;
        assert_eq!(reordered_scope, scope);
        let reordered_nonce = ProviderCostNonce::new([0x86; 32])?;
        let reordered_acceptance = issuer.issue(
            reordered_scope,
            now,
            now + Duration::from_secs(30),
            reordered_nonce,
        )?;
        reordered_service.accept_streaming_cost(
            reordered_request,
            ProviderCostAuthorization::new(price.clone(), reordered_nonce, reordered_acceptance)?,
            &verifier,
            now,
        )?;
        let reordered_proposal = reordered_service.prepare_streaming_actuation(reordered_handle)?;
        assert_eq!(
            reordered_proposal.ordered_uploads_sha256(),
            proposal.ordered_uploads_sha256()
        );
        assert_eq!(
            reordered_proposal.idempotency_identity_sha256(),
            proposal.idempotency_identity_sha256()
        );

        let changed_context = WorkerProviderInvocationContext {
            generation: context.generation + 1,
            ..context
        };
        let (changed_service, changed_handle) =
            started_streaming_service(&changed_context, &CancellationToken::default())?;
        complete_streaming_upload(&changed_service, changed_handle, b"priced-upload")?;
        let changed_request = WorkerProviderCostRequest {
            handle: changed_handle,
            ..request
        };
        let changed_scope =
            changed_service.prepare_cost_acceptance(&changed_request, price.clone())?;
        assert_ne!(changed_scope, scope);
        let changed_nonce = ProviderCostNonce::new([0x87; 32])?;
        let stale_acceptance =
            issuer.issue(scope, now, now + Duration::from_secs(30), changed_nonce)?;
        assert_eq!(
            changed_service.accept_streaming_cost(
                changed_request,
                ProviderCostAuthorization::new(price, changed_nonce, stale_acceptance)?,
                &verifier,
                now,
            ),
            Err(PluginServiceError::ProviderCostAcceptanceDenied)
        );

        let denial_context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_aa04),
            ..context
        };
        let (denial_service, denial_handle) =
            started_streaming_service(&denial_context, &CancellationToken::default())?;
        let denied = denial_service.deny_streaming_cost(WorkerProviderCostRequest {
            handle: denial_handle,
            operation: "provider.cost".to_owned(),
            currency: "USD".to_owned(),
            maximum_microunits: 5,
        })?;
        assert_eq!(denied.approved_microunits, 0);
        assert!(!denied.accepted);
        assert!(denied.receipt.is_empty());
        assert_eq!(
            denial_service
                .prepare_streaming_actuation(denial_handle)?
                .accepted_cost_microunits(),
            0
        );
        assert!(
            denial_service
                .state
                .lock()
                .owner
                .consumed_streaming_cost_nonces
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn provider_runtime_service_identities_bind_body_upload_and_cost_mutations()
    -> Result<(), Box<dyn Error>> {
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_aa05),
            session_generation: 1,
            invocation: 2,
            generation: 3,
        };
        let baseline = prepared_mutation_proposal(
            &context,
            b"request-a",
            "image.input",
            "application/octet-stream",
            b"upload-a",
            "https://fixture.invalid/storage/upload-a.bin",
        )?;
        for proposal in [
            prepared_mutation_proposal(
                &context,
                b"request-b",
                "image.input",
                "application/octet-stream",
                b"upload-a",
                "https://fixture.invalid/storage/upload-a.bin",
            )?,
            prepared_mutation_proposal(
                &context,
                b"request-a",
                "image.reference",
                "application/octet-stream",
                b"upload-a",
                "https://fixture.invalid/storage/upload-a.bin",
            )?,
            prepared_mutation_proposal(
                &context,
                b"request-a",
                "image.input",
                "application/json",
                b"upload-a",
                "https://fixture.invalid/storage/upload-a.bin",
            )?,
            prepared_mutation_proposal(
                &context,
                b"request-a",
                "image.input",
                "application/octet-stream",
                b"upload-b",
                "https://fixture.invalid/storage/upload-a.bin",
            )?,
            prepared_mutation_proposal(
                &context,
                b"request-a",
                "image.input",
                "application/octet-stream",
                b"upload-a",
                "https://fixture.invalid/storage/upload-b.bin",
            )?,
        ] {
            assert_ne!(
                proposal.idempotency_identity_sha256(),
                baseline.idempotency_identity_sha256()
            );
        }

        let baseline_scope = prepared_cost_scope(&context, "provider.cost", "USD", 5)?;
        for scope in [
            prepared_cost_scope(&context, "provider.cost.alt", "USD", 5)?,
            prepared_cost_scope(&context, "provider.cost", "EUR", 5)?,
            prepared_cost_scope(&context, "provider.cost", "USD", 6)?,
        ] {
            assert_ne!(scope, baseline_scope);
        }
        Ok(())
    }

    #[test]
    fn provider_runtime_stream_service_preserves_retry_progress_and_terminal_state()
    -> Result<(), Box<dyn Error>> {
        for (index, invalid_headers) in [
            vec![
                WorkerProviderHeader {
                    name: "retry-after".to_owned(),
                    value: "1".to_owned(),
                },
                WorkerProviderHeader {
                    name: "Retry-After".to_owned(),
                    value: "2".to_owned(),
                },
            ],
            vec![WorkerProviderHeader {
                name: "retry-after".to_owned(),
                value: "seconds".to_owned(),
            }],
            vec![WorkerProviderHeader {
                name: "retry-after".to_owned(),
                value: "01".to_owned(),
            }],
            vec![WorkerProviderHeader {
                name: "retry-after".to_owned(),
                value: "86401".to_owned(),
            }],
        ]
        .into_iter()
        .enumerate()
        {
            let context = WorkerProviderInvocationContext {
                session_id: Uuid::from_u128(0x412_bb00 + u128::try_from(index)?),
                session_generation: 1,
                invocation: 1,
                generation: 1,
            };
            let (service, handle) =
                started_streaming_service(&context, &CancellationToken::default())?;
            service.prepare_streaming_actuation(handle)?;
            let wait = WorkerProviderWaitRequest {
                handle,
                after_sequence: None,
                timeout_milliseconds: 100,
            };
            assert!(matches!(
                service.accept_wait(
                    wait,
                    WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                        handle,
                        sequence: 0,
                        event: WorkerProviderResponseFrameEvent::Head(
                            sim_types::WorkerProviderResponseHead {
                                status: 429,
                                headers: invalid_headers,
                            },
                        ),
                    }),
                ),
                Err(PluginServiceError::ProviderStreamingContract(
                    ProviderStreamingContractError::InvalidHeaders
                ))
            ));
            assert_eq!(service.retry_after_seconds(handle)?, None);
            service.accept_wait(
                wait,
                WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                    handle,
                    sequence: 0,
                    event: WorkerProviderResponseFrameEvent::Head(
                        sim_types::WorkerProviderResponseHead {
                            status: 429,
                            headers: vec![WorkerProviderHeader {
                                name: "Retry-After".to_owned(),
                                value: "120".to_owned(),
                            }],
                        },
                    ),
                }),
            )?;
            assert_eq!(service.retry_after_seconds(handle)?, Some(120));
            service.revoke_stream(handle);
        }

        let progress_context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_bb10),
            session_generation: 1,
            invocation: 1,
            generation: 1,
        };
        let (progress_service, progress_handle) =
            started_streaming_service(&progress_context, &CancellationToken::default())?;
        progress_service.prepare_streaming_actuation(progress_handle)?;
        progress_service.accept_wait(
            WorkerProviderWaitRequest {
                handle: progress_handle,
                after_sequence: None,
                timeout_milliseconds: 100,
            },
            WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                handle: progress_handle,
                sequence: 0,
                event: WorkerProviderResponseFrameEvent::Head(
                    sim_types::WorkerProviderResponseHead {
                        status: 200,
                        headers: Vec::new(),
                    },
                ),
            }),
        )?;
        let progress_origin = Instant::now();
        assert!(
            progress_service
                .report_progress(
                    WorkerProviderProgress {
                        handle: progress_handle,
                        sequence: 0,
                        completed: 1,
                        total: 10,
                        message: None,
                    },
                    progress_origin,
                )?
                .is_some()
        );
        assert!(
            progress_service
                .report_progress(
                    WorkerProviderProgress {
                        handle: progress_handle,
                        sequence: 1,
                        completed: 2,
                        total: 10,
                        message: None,
                    },
                    progress_origin + Duration::from_millis(10),
                )?
                .is_none()
        );
        assert!(
            progress_service
                .report_progress(
                    WorkerProviderProgress {
                        handle: progress_handle,
                        sequence: 2,
                        completed: 10,
                        total: 10,
                        message: Some("complete".to_owned()),
                    },
                    progress_origin + Duration::from_millis(20),
                )?
                .is_some()
        );
        assert!(matches!(
            progress_service.report_progress(
                WorkerProviderProgress {
                    handle: progress_handle,
                    sequence: 2,
                    completed: 10,
                    total: 10,
                    message: None,
                },
                progress_origin + Duration::from_millis(60),
            ),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidProgress
            ))
        ));

        let cancellation = CancellationToken::default();
        let cancelled_context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x412_bb11),
            ..progress_context
        };
        let (cancelled_service, cancelled_handle) =
            started_streaming_service(&cancelled_context, &cancellation)?;
        complete_streaming_upload(&cancelled_service, cancelled_handle, b"cancelled")?;
        cancellation.cancel();
        assert_eq!(
            cancelled_service.check_cancelled(cancelled_handle),
            Err(PluginServiceError::Cancelled)
        );
        {
            let state = cancelled_service.state.lock();
            assert!(state.owner.is_empty());
            assert!(state.handles.is_empty());
            assert!(state.upload_parents.is_empty());
            assert!(state.cancellations.is_empty());
        }

        for (index, terminal) in [
            WorkerProviderTerminal::Failed {
                code: "fixture.failed".to_owned(),
                message: "failed".to_owned(),
            },
            WorkerProviderTerminal::Cancelled,
        ]
        .into_iter()
        .enumerate()
        {
            let context = WorkerProviderInvocationContext {
                session_id: Uuid::from_u128(0x412_bb20 + u128::try_from(index)?),
                ..progress_context.clone()
            };
            let (service, handle) =
                started_streaming_service(&context, &CancellationToken::default())?;
            let upload = complete_streaming_upload(&service, handle, b"terminal")?;
            service.prepare_streaming_actuation(handle)?;
            service.accept_wait(
                WorkerProviderWaitRequest {
                    handle,
                    after_sequence: None,
                    timeout_milliseconds: 100,
                },
                WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                    handle,
                    sequence: 0,
                    event: WorkerProviderResponseFrameEvent::Head(
                        sim_types::WorkerProviderResponseHead {
                            status: 500,
                            headers: Vec::new(),
                        },
                    ),
                }),
            )?;
            service.accept_wait(
                WorkerProviderWaitRequest {
                    handle,
                    after_sequence: Some(0),
                    timeout_milliseconds: 100,
                },
                WorkerProviderWaitOutcome::Frame(sim_types::WorkerProviderResponseFrame {
                    handle,
                    sequence: 1,
                    event: WorkerProviderResponseFrameEvent::Terminal(terminal),
                }),
            )?;
            assert!(matches!(
                service.check_cancelled(handle),
                Err(PluginServiceError::ProviderStreamingContract(
                    ProviderStreamingContractError::RevokedHandle
                ))
            ));
            assert!(matches!(
                service.check_cancelled(upload),
                Err(PluginServiceError::ProviderStreamingContract(
                    ProviderStreamingContractError::RevokedHandle
                ))
            ));
            let state = service.state.lock();
            assert!(state.owner.is_empty());
            assert!(state.handles.is_empty());
            assert!(state.upload_parents.is_empty());
            assert!(state.cancellations.is_empty());
        }
        Ok(())
    }

    #[test]
    fn provider_runtime_stream_owner_is_transactional_progressive_and_receipt_bound()
    -> Result<(), Box<dyn Error>> {
        let binding = provider_binding()?;
        let (authority, head) = bound_streaming_authority(&binding, [0x61; 32])?;
        let handle = ProviderStreamHandleV2 {
            invocation: 10,
            slot: 1,
            generation: 2,
        };
        let cancellation = CancellationToken::default();
        let mut owner = ProviderRuntimeStreamOwner::new();
        owner.begin_streaming(
            authority,
            ProviderInvocationContextV2 {
                invocation: 10,
                generation: 2,
            },
            handle,
            &cancellation,
        )?;
        owner.write_streaming_request_chunk(
            &ProviderRequestChunkV2 {
                handle,
                sequence: 0,
                bytes: Vec::new(),
                end: true,
            },
            &cancellation,
        )?;
        let wait = ProviderWaitRequestV2 {
            handle,
            after_sequence: None,
            timeout_milliseconds: 100,
        };
        let response_head = ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
            handle,
            sequence: 0,
            event: ProviderResponseFrameEventV2::Head(ProviderResponseHeadV2 {
                status: 200,
                headers: Vec::new(),
            }),
        });
        assert!(matches!(
            owner.accept_streaming_wait(&wait, response_head.clone(), &cancellation),
            Err(PluginServiceError::ProviderStreamingContract(
                ProviderStreamingContractError::InvalidOrder
            ))
        ));
        let proposal = owner.prepare_streaming_actuation(handle, None, &cancellation)?;
        assert_eq!(proposal.request_head(), &head);
        assert_eq!(
            proposal.request().idempotency_key_sha256(),
            Some(proposal.idempotency_identity_sha256())
        );
        owner.accept_streaming_wait(&wait, response_head, &cancellation)?;
        let progress = ProviderProgressV2 {
            handle,
            sequence: 0,
            completed: 1,
            total: 2,
            message: Some("working".to_owned()),
        };
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            owner.report_streaming_progress(&progress, Instant::now(), &cancelled),
            Err(PluginServiceError::Cancelled)
        ));
        assert!(
            owner
                .report_streaming_progress(&progress, Instant::now(), &cancellation)?
                .is_some()
        );
        owner.accept_streaming_wait(
            &ProviderWaitRequestV2 {
                after_sequence: Some(0),
                ..wait
            },
            ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                handle,
                sequence: 1,
                event: ProviderResponseFrameEventV2::Terminal(
                    ProviderStreamTerminalV2::Completed {
                        receipt: b"terminal-receipt".to_vec(),
                    },
                ),
            }),
            &cancellation,
        )?;
        let origin = Instant::now();
        let issuer = ProviderRuntimeReceiptIssuerV2::from_seed([0x62; 32], origin)?;
        let receipt = owner.finish_streaming(
            handle,
            &issuer,
            origin,
            origin + Duration::from_secs(30),
            [0x63; 32],
            &cancellation,
        )?;
        assert_eq!(
            receipt.identity().terminal_receipt_sha256,
            provider_terminal_completed_receipt_sha256(b"terminal-receipt")
        );
        assert!(owner.is_empty());
        Ok(())
    }

    #[test]
    fn provider_runtime_failed_and_cancelled_terminals_release_capacity()
    -> Result<(), Box<dyn Error>> {
        let binding = provider_binding()?;
        let cancellation = CancellationToken::default();
        let mut owner = ProviderRuntimeStreamOwner::new();
        for invocation in 1..=64_u64 {
            let (authority, _) = bound_streaming_authority(&binding, [0x71; 32])?;
            let handle = ProviderStreamHandleV2 {
                invocation,
                slot: 1,
                generation: 1,
            };
            owner.begin_streaming(
                authority,
                ProviderInvocationContextV2 {
                    invocation,
                    generation: 1,
                },
                handle,
                &cancellation,
            )?;
            let outcome = if invocation % 2 == 0 {
                ProviderWaitOutcomeV2::Cancelled
            } else {
                ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                    handle,
                    sequence: 0,
                    event: ProviderResponseFrameEventV2::Terminal(
                        ProviderStreamTerminalV2::Failed {
                            code: "provider.failed".to_owned(),
                            message: "failed".to_owned(),
                        },
                    ),
                })
            };
            owner.accept_streaming_wait(
                &ProviderWaitRequestV2 {
                    handle,
                    after_sequence: None,
                    timeout_milliseconds: 100,
                },
                outcome,
                &cancellation,
            )?;
            assert!(matches!(
                owner.write_streaming_request_chunk(
                    &ProviderRequestChunkV2 {
                        handle,
                        sequence: 0,
                        bytes: Vec::new(),
                        end: true,
                    },
                    &cancellation,
                ),
                Err(PluginServiceError::ProviderSessionUnavailable)
            ));
        }
        assert!(owner.is_empty());
        Ok(())
    }
    const SCRIPTED_DOWNLOADED_PNG: &[u8] =
        include_bytes!("../../sim_test_support/fixtures/providers/bfl/downloaded-image.png");

    fn image_receipt_identity() -> Result<ProviderInvocationIdentity, TrustError> {
        ProviderInvocationIdentity::new(
            "principal-a",
            PROFILE_UUID.to_string(),
            PROMPT_UUID.to_string(),
            PROMPT_SHA256,
            ATTEMPT_UUID.to_string(),
            "node.fixture",
            0,
            format!("{:x}", Sha256::digest(REQUEST_BODY)),
            "plugin.fixture",
            DIGEST,
            DIGEST,
            PROVIDER,
            ENDPOINT,
        )
    }

    fn image_receipt_native_context(
        cancellation: CancellationToken,
        memory_limit: u64,
    ) -> Result<
        (
            sim_nodes::NativeNodeContext,
            crate::NativeHandleStoreGeneration,
        ),
        Box<dyn Error>,
    > {
        use sim_nodes::{
            NativeNodeComputeSession, NativeNodeContext, NativeNodeServiceIdentity,
            NativeNodeServices,
        };
        use sim_tensor::CpuWorkspaceAuthority;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(memory_limit)?;
        let generation = crate::NativeHandleStoreGeneration::with_capacities(1, 1024)?;
        let attempt = AttemptId(ATTEMPT_UUID);
        let node = NodeId::from("node.fixture");
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(Uuid::from_u128(0x437901), attempt, node.clone())?,
            backend,
            sim_plugin_sdk::StreamId::DEFAULT,
            &scratch,
        )?;
        let context = NativeNodeContext::new_with_services(
            PromptId(PROMPT_UUID),
            attempt,
            node,
            cancellation,
            scratch,
            generation.handle_store_for_attempt(attempt),
            NativeNodeServices::checked(None, None, Some(compute))?.with_provider_execution(
                sim_nodes::NativeProviderExecutionIdentity::checked(PROMPT_SHA256)?,
            ),
        )?;
        Ok((context, generation))
    }

    fn image_receipt_service_context(
        authorization: PluginAuthorization,
        clock: &TestClock,
        cancellation: CancellationToken,
        issuer_seed: u8,
    ) -> Result<PluginServiceInvocationContext, Box<dyn Error>> {
        let authority = ProviderResultReceiptAuthority::new(
            "principal-a",
            PROMPT_SHA256,
            DIGEST,
            Arc::new(ProviderResultReceiptIssuer::from_seed(
                [issuer_seed; 32],
                clock.now(),
            )?),
            Duration::from_secs(20),
        )?;
        Ok(PluginServiceInvocationContext::new_with_principal(
            ProfileId(PROFILE_UUID),
            PromptId(PROMPT_UUID),
            AttemptId(ATTEMPT_UUID),
            NodeId::from("node.fixture"),
            "principal-a",
            authorization,
            cancellation,
            clock.now() + Duration::from_secs(30),
            4096,
        )?
        .with_provider_result_authority(authority)?)
    }

    fn wire_image_receipt(
        invocation: &mut PluginCapabilityInvocation,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        let response = invocation.handle_wire_request(PluginServiceWireRequest::ExecuteProvider {
            provider: PROVIDER.to_owned(),
            endpoint: ENDPOINT.to_owned(),
            secret_id: None,
            body: REQUEST_BODY.to_vec(),
        });
        match response {
            PluginServiceWireResponse::Bytes(receipt) => Ok(receipt),
            response => Err(format!("scripted PNG route returned {response:?}").into()),
        }
    }

    #[test]
    fn retained_worker_capability_png_receipt_materializes_and_settles_same_session()
    -> Result<(), Box<dyn Error>> {
        use sim_nodes::NativeStoredPayload;
        let authorization = authorization(provider_capabilities(false))?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, provider, credential) = broker(&authorization, clock.clone())?;
        provider.set_response(SCRIPTED_DOWNLOADED_PNG.to_vec());
        let mut invocation = broker.begin_invocation(image_receipt_service_context(
            authorization,
            &clock,
            CancellationToken::default(),
            91,
        )?)?;
        let receipt = wire_image_receipt(&mut invocation)?;
        let mut retained = crate::RetainedPluginExecution::with_capability_invocation_for_test(
            sim_types::WorkerPluginExecutionOutcome::Succeeded(receipt.clone()),
            invocation,
        );
        let (context, generation) =
            image_receipt_native_context(CancellationToken::default(), 1024 * 1024)?;
        let payload = retained.materialize_provider_downloaded_image(
            &receipt,
            &image_receipt_identity()?,
            &context,
        )?;
        assert!(generation.is_empty());
        let NativeStoredPayload::Tensor(image) = &payload else {
            return Err("decoded payload is not IMAGE".into());
        };
        assert_eq!(image.tensor().descriptor().shape(), [1, 1, 1, 4]);
        assert_eq!(
            image.tensor().descriptor().dtype(),
            sim_plugin_sdk::DType::F32
        );
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let values = sim_tensor::generated_native_diffusion::tensor_to_f32(
            compute.backend(),
            image.tensor(),
            &execution,
        )?;
        assert_eq!(&*values, &[128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 1.0]);
        drop(values);
        let expected_type = payload.handle_type()?;
        let registry = sim_plugin_sdk::TypeRegistry::built_in()?;
        let transport = crate::ProviderTransportValue::from_native_payload(
            registry.resolve("IMAGE")?.to_string(),
            sim_plugin_sdk::ValueFamily::Tensor,
            &payload,
        )?;
        let response = crate::ProviderTransportResponse::checked(
            "node.fixture",
            vec![crate::ProviderTransportPort::checked(
                "output_0",
                true,
                vec![transport],
            )?],
        )?;
        let response = crate::ProviderTransportResponse::from_bytes(&response.to_bytes()?)?;
        let restored = response
            .ports()
            .first()
            .and_then(|port| port.values().first())
            .ok_or("missing canonical IMAGE output transport")?
            .materialize_native_payload(&expected_type, "plugin.fixture", &context)?;
        drop(payload);
        let payload = restored;
        let NativeStoredPayload::Tensor(image) = &payload else {
            return Err("restored transport is not IMAGE".into());
        };
        let values = sim_tensor::generated_native_diffusion::tensor_to_f32(
            compute.backend(),
            image.tensor(),
            &execution,
        )?;
        assert_eq!(&*values, &[128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 1.0]);
        drop(values);

        let resolved = retained
            .resolve_provider_result_receipt_set(&ProviderResultReceiptSet::new(vec![receipt])?)?;
        assert_eq!(resolved.len(), 1);
        assert_eq!(
            resolved
                .first()
                .ok_or("missing retained result")?
                .response(),
            SCRIPTED_DOWNLOADED_PNG
        );
        assert_eq!(
            resolved
                .first()
                .ok_or("missing retained result")?
                .identity(),
            &image_receipt_identity()?
        );
        assert!(matches!(
            retained.finish()?,
            sim_types::WorkerPluginExecutionOutcome::Succeeded(_)
        ));
        context.cancellation.check()?;
        let handle = context
            .handle_store()
            .publish(payload, &context.cancellation)?;
        assert_eq!(generation.len(), 1);
        context
            .handle_store()
            .revoke(&handle, &context.cancellation)?;
        assert!(generation.is_empty());
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(
            context
                .compute_session()?
                .backend()
                .memory_snapshot()
                .current_bytes,
            0
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 1);
        assert_eq!(credential.calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn broker_png_receipt_cancellation_and_capacity_roll_back_then_recover()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(provider_capabilities(false))?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, provider, _credential) = broker(&authorization, clock.clone())?;
        provider.set_response(SCRIPTED_DOWNLOADED_PNG.to_vec());
        for (seed, memory_limit, cancelled) in [(92, 1024 * 1024, true), (93, 15, false)] {
            let cancellation = CancellationToken::default();
            let mut invocation = broker.begin_invocation(image_receipt_service_context(
                authorization.clone(),
                &clock,
                cancellation.clone(),
                seed,
            )?)?;
            let receipt = wire_image_receipt(&mut invocation)?;
            let (context, generation) =
                image_receipt_native_context(cancellation.clone(), memory_limit)?;
            if cancelled {
                assert!(cancellation.cancel());
            }
            let result = invocation.materialize_provider_downloaded_image(
                &receipt,
                &image_receipt_identity()?,
                &context,
            );
            if cancelled {
                assert!(matches!(result, Err(PluginServiceError::Cancelled)));
            } else {
                assert!(matches!(
                    result,
                    Err(PluginServiceError::ActuatorFailed {
                        service: "provider_image_allocation",
                        ..
                    })
                ));
            }
            assert!(generation.is_empty());
            assert_eq!(context.scratch.in_use_bytes(), 0);
            assert_eq!(
                context
                    .compute_session()?
                    .backend()
                    .memory_snapshot()
                    .current_bytes,
                0
            );
            assert_eq!(
                invocation.finish(),
                Err(PluginServiceError::InvocationFailed)
            );
        }
        let mut recovery = broker.begin_invocation(image_receipt_service_context(
            authorization,
            &clock,
            CancellationToken::default(),
            94,
        )?)?;
        let receipt = wire_image_receipt(&mut recovery)?;
        let (context, generation) =
            image_receipt_native_context(CancellationToken::default(), 1024 * 1024)?;
        let payload = recovery.materialize_provider_downloaded_image(
            &receipt,
            &image_receipt_identity()?,
            &context,
        )?;
        drop(payload);
        recovery
            .resolve_provider_result_receipt_set(&ProviderResultReceiptSet::new(vec![receipt])?)?;
        recovery.finish()?;
        assert!(generation.is_empty());
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(
            context
                .compute_session()?
                .backend()
                .memory_snapshot()
                .current_bytes,
            0
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 3);
        Ok(())
    }

    #[test]
    fn broker_png_receipt_foreign_identity_and_unsettled_finish_are_denied()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(provider_capabilities(false))?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, provider, _credential) = broker(&authorization, clock.clone())?;
        provider.set_response(SCRIPTED_DOWNLOADED_PNG.to_vec());
        let mut original = broker.begin_invocation(image_receipt_service_context(
            authorization.clone(),
            &clock,
            CancellationToken::default(),
            95,
        )?)?;
        let receipt = wire_image_receipt(&mut original)?;
        let (context, generation) =
            image_receipt_native_context(CancellationToken::default(), 1024 * 1024)?;
        let foreign = broker.begin_invocation(image_receipt_service_context(
            authorization.clone(),
            &clock,
            CancellationToken::default(),
            96,
        )?)?;
        assert!(matches!(
            foreign.materialize_provider_downloaded_image(
                &receipt,
                &image_receipt_identity()?,
                &context
            ),
            Err(PluginServiceError::ProviderResultReceiptAuthorityDenied)
        ));
        assert_eq!(foreign.finish(), Err(PluginServiceError::InvocationFailed));
        let altered = ProviderInvocationIdentity::new(
            "principal-a",
            PROFILE_UUID.to_string(),
            PROMPT_UUID.to_string(),
            PROMPT_SHA256,
            ATTEMPT_UUID.to_string(),
            "node.fixture",
            0,
            "f".repeat(64),
            "plugin.fixture",
            DIGEST,
            DIGEST,
            PROVIDER,
            ENDPOINT,
        )?;
        assert!(matches!(
            original.materialize_provider_downloaded_image(&receipt, &altered, &context),
            Err(PluginServiceError::ProviderResultReceiptAuthorityDenied)
        ));
        assert_eq!(original.finish(), Err(PluginServiceError::InvocationFailed));
        let mut unsettled = broker.begin_invocation(image_receipt_service_context(
            authorization,
            &clock,
            CancellationToken::default(),
            97,
        )?)?;
        let pending_receipt = wire_image_receipt(&mut unsettled)?;
        let payload = unsettled.materialize_provider_downloaded_image(
            &pending_receipt,
            &image_receipt_identity()?,
            &context,
        )?;
        drop(payload);
        assert_eq!(
            unsettled.finish(),
            Err(PluginServiceError::ProviderResultReceiptAuthorityDenied)
        );
        assert!(generation.is_empty());
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(
            context
                .compute_session()?
                .backend()
                .memory_snapshot()
                .current_bytes,
            0
        );
        assert_eq!(provider.calls.load(Ordering::Acquire), 2);
        Ok(())
    }
    #[test]
    fn retained_response_discrimination_preserves_success_and_failed_operation_state()
    -> Result<(), Box<dyn Error>> {
        let authorization = authorization(provider_capabilities(false))?;
        let clock = Arc::new(TestClock::new(Instant::now()));
        let (_directory, broker, provider, _) = broker(&authorization, clock.clone())?;
        let structured = crate::ProviderTransportResponse::checked(
            "node.fixture",
            vec![crate::ProviderTransportPort::checked(
                "output_0",
                false,
                Vec::new(),
            )?],
        )?
        .to_bytes()?;
        provider.set_response(structured.clone());
        let mut invocation = broker.begin_invocation(image_receipt_service_context(
            authorization.clone(),
            &clock,
            CancellationToken::default(),
            139,
        )?)?;
        let receipt = wire_image_receipt(&mut invocation)?;
        let (context, generation) = image_receipt_native_context(CancellationToken::default(), 15)?;
        assert!(invocation.is_provider_result_structured_transport_response(
            &receipt,
            &image_receipt_identity()?,
            &context
        )?);
        let set = ProviderResultReceiptSet::new(vec![receipt])?;
        let resolved = invocation.resolve_provider_result_receipt_set(&set)?;
        assert_eq!(
            resolved
                .first()
                .ok_or("missing structured result")?
                .response(),
            structured
        );
        invocation.finish()?;
        assert!(generation.is_empty());
        assert_eq!(
            context
                .compute_session()?
                .backend()
                .memory_snapshot()
                .current_bytes,
            0
        );
        provider.set_response(SCRIPTED_DOWNLOADED_PNG.to_vec());
        let mut failed = broker.begin_invocation(image_receipt_service_context(
            authorization.clone(),
            &clock,
            CancellationToken::default(),
            140,
        )?)?;
        let receipt = wire_image_receipt(&mut failed)?;
        assert!(!failed.is_provider_result_structured_transport_response(
            &receipt,
            &image_receipt_identity()?,
            &context
        )?);
        assert!(
            failed
                .materialize_provider_downloaded_image(
                    &receipt,
                    &image_receipt_identity()?,
                    &context
                )
                .is_err()
        );
        assert_eq!(failed.finish(), Err(PluginServiceError::InvocationFailed));
        let mut recovered = broker.begin_invocation(image_receipt_service_context(
            authorization,
            &clock,
            CancellationToken::default(),
            141,
        )?)?;
        let receipt = wire_image_receipt(&mut recovered)?;
        let (recovery_context, recovered_generation) =
            image_receipt_native_context(CancellationToken::default(), 1024 * 1024)?;
        assert!(!recovered.is_provider_result_structured_transport_response(
            &receipt,
            &image_receipt_identity()?,
            &recovery_context
        )?);
        let payload = recovered.materialize_provider_downloaded_image(
            &receipt,
            &image_receipt_identity()?,
            &recovery_context,
        )?;
        drop(payload);
        recovered
            .resolve_provider_result_receipt_set(&ProviderResultReceiptSet::new(vec![receipt])?)?;
        recovered.finish()?;
        assert!(recovered_generation.is_empty());
        assert_eq!(
            recovery_context
                .compute_session()?
                .backend()
                .memory_snapshot()
                .current_bytes,
            0
        );
        Ok(())
    }
}
