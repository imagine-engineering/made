mod capabilities;
mod component_host;
mod legacy_mapping;
mod private_worker;
mod registry_adapter;

pub use capabilities::{
    AssetPluginCapabilityServices, BrokerPluginCapabilityServices, CancellationToken,
    CapabilityEffects, CapabilityLimits, CapabilityServiceContext, CapabilityState,
    PluginCapabilityServices, PluginOutputProposal, PluginOutputPublicationAdapter,
    PluginOutputPublicationError, RouteEffect, UnavailablePluginCapabilityServices,
    check_plugin_cancellation,
};
pub use component_host::{
    ComponentExecutionBoundary, ComponentHost, ComponentHostError,
    ComponentHostProviderInvocationAuthority, ComponentHostRouter, InstalledComponentBinding,
    InstalledVerifiedPlugin, PluginInvocationExecutor, PreparedPluginInvocation,
    VerifiedComponentDeployment, VerifiedComponentGeneration, WorkerPluginInvocation,
};
pub use legacy_mapping::{
    AcceptedRewrite, InstalledMappingProjection, LegacyCompatibilityProjection,
    LegacyInputPortTranslation, LegacyInputSourceProjection, LegacyMappingError,
    LegacyMappingResolver, LegacyNodeReference, LegacyOutputPortTranslation, LegacyPortTranslation,
    LegacyProviderProjection, LegacyProviderScope, LegacyResolution, MAX_LEGACY_REFERENCE_BYTES,
    MappingCandidate, MappingProvenance, MappingSource, MappingTarget,
};
pub use private_worker::PrivateWorkerPluginExecutor;
#[cfg(feature = "test-support")]
pub use private_worker::{
    PrivateWorkerProviderV2ActuatorAttachment, private_worker_provider_v2_actuator_route,
};
pub use registry_adapter::{
    PluginRegistryAdapterError, PreparedNativeProviderInvocation,
    materialize_native_provider_response, prepare_native_provider_invocation,
    registry_with_installed_plugins, rollback_native_provider_outputs,
};

use extension_host::ComponentRuntime;
use sha2::{Digest, Sha256};
use sim_plugin_sdk::{
    COMPONENT_API_VERSION, CancelReason, CanonicalTypeId, CapabilityCall, CapabilityResponse,
    ComponentManifestProjection, InputState, InvocationError, NegotiatedApi,
    PROVIDER_COMPONENT_WORLD, PROVIDER_STREAMING_API_FEATURE_V2, PluginContractError,
    PluginInvocation, PluginManifest, PluginNode, PluginValue, PluginValueRepresentation,
    PortCardinality, PortDirection, PortPresence, PortSerialization, ProviderBindingClaim,
    ProviderBindingSet, ProviderEncodedValueV2, ProviderHttpMethodV2, ProviderInvocationResultV2,
    ProviderMaterializedOutputV2, ProviderPluginManifestV2, ProviderResultReceiptSet,
    ProviderStreamingContractV2, RustSimPlugin, TypeRegistry, ValueFamily, ValueHandle,
};
use sim_runtime::{
    AssetIdentity, AssetNamespace, PluginAuthorization, ProviderInvocationIdentity,
    ProviderManifestAuthorizationV2, ProviderRuntimeHttpInvocationIdentity,
    ProviderTransportResponse, ResolvedProviderResult, TrustError,
    WorkerProviderV2InvocationEnvelope,
};
use sim_types::{
    MAX_WORKER_PROVIDER_PENDING_CALLS, MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS,
    MAX_WORKER_PROVIDER_WAIT_MILLISECONDS, WorkerProviderCostRequest, WorkerProviderHeader,
    WorkerProviderHttpContinuationChallenge, WorkerProviderHttpMethod,
    WorkerProviderInvocationContext, WorkerProviderProgress, WorkerProviderRequestChunk,
    WorkerProviderRequestHead, WorkerProviderResponseChunk, WorkerProviderResponseFrame,
    WorkerProviderResponseFrameEvent, WorkerProviderRetryRequest, WorkerProviderStreamError,
    WorkerProviderStreamHandle, WorkerProviderStreamRequest, WorkerProviderStreamResponse,
    WorkerProviderStreamTransportValidator, WorkerProviderStreamingContract,
    WorkerProviderTerminal, WorkerProviderUploadCompletionRequest, WorkerProviderUploadRequest,
    WorkerProviderV2ProposalFinalization, WorkerProviderWaitOutcome, WorkerProviderWaitRequest,
    WorkerSha256Digest,
};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    mem,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel},
    },
    time::{Duration, Instant, SystemTime},
};
use thiserror::Error;
use wasmtime::{
    Store, StoreLimits, StoreLimitsBuilder,
    component::{Component, Linker},
};

mod wit_contract {
    wasmtime::component::bindgen!({
        path: "../sim_plugin_sdk/wit/sim-plugin.wit",
        world: "sim-plugin",
    });
}

mod provider_wit_contract {
    wasmtime::component::bindgen!({
        path: "../sim_plugin_sdk/wit/sim-plugin.wit",
        world: "sim-provider-plugin",
        with: {
            "zed:sim-plugin/types@1.0.0": super::wit_contract::zed::sim_plugin::types,
            "zed:sim-plugin/host@1.0.0": super::wit_contract::zed::sim_plugin::host,
        },
    });
}

mod provider_v2_wit_contract {
    wasmtime::component::bindgen!({
        path: "../sim_plugin_sdk/wit/provider-v2",
        world: "sim-provider-plugin",
        with: {
            "zed:sim-plugin/types@1.0.0": super::wit_contract::zed::sim_plugin::types,
        },
    });
}

type WitInvocationError = wit_contract::zed::sim_plugin::types::InvocationError;

pub const DEFAULT_API_FEATURES: &[&str] = &[
    "capabilities.transactional",
    "handles.revocation",
    "legacy.non-destructive",
    "ports.list",
    "provider.bindings.v1",
    PROVIDER_STREAMING_API_FEATURE_V2,
];

static NEXT_INVOCATION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentLimits {
    pub maximum_component_bytes: usize,
    pub maximum_memory_bytes: usize,
    pub maximum_table_elements: usize,
    pub maximum_instances: usize,
    pub maximum_tables: usize,
    pub maximum_memories: usize,
    pub maximum_fuel: u64,
    pub epoch_deadline_ticks: u64,
    pub maximum_values_per_port: usize,
    pub maximum_value_bytes: u64,
    pub maximum_value_handles: usize,
    pub maximum_invocation_value_bytes: u64,
    pub maximum_port_operations: u64,
    pub maximum_port_response_bytes: u64,
    pub capability_limits: CapabilityLimits,
}

impl Default for ComponentLimits {
    fn default() -> Self {
        Self {
            maximum_component_bytes: 64 * 1024 * 1024,
            maximum_memory_bytes: 256 * 1024 * 1024,
            maximum_table_elements: 100_000,
            maximum_instances: 32,
            maximum_tables: 32,
            maximum_memories: 16,
            maximum_fuel: 10_000_000,
            epoch_deadline_ticks: 1,
            maximum_values_per_port: 16_384,
            maximum_value_bytes: 256 * 1024 * 1024,
            maximum_value_handles: 65_536,
            maximum_invocation_value_bytes: 512 * 1024 * 1024,
            maximum_port_operations: 1_000_000,
            maximum_port_response_bytes: 512 * 1024 * 1024,
            capability_limits: CapabilityLimits::default(),
        }
    }
}

impl ComponentLimits {
    pub(crate) fn validate(&self) -> Result<(), PluginError> {
        if self.maximum_component_bytes == 0
            || self.maximum_memory_bytes == 0
            || self.maximum_table_elements == 0
            || self.maximum_instances == 0
            || self.maximum_tables == 0
            || self.maximum_memories == 0
            || self.maximum_fuel == 0
            || self.epoch_deadline_ticks == 0
            || self.maximum_values_per_port == 0
            || self.maximum_value_bytes == 0
            || self.maximum_value_handles == 0
            || self.maximum_invocation_value_bytes == 0
            || self.maximum_port_operations == 0
            || self.maximum_port_response_bytes == 0
        {
            return Err(PluginError::InvalidHostLimits);
        }
        self.capability_limits
            .validate()
            .map_err(|_| PluginError::InvalidHostLimits)?;
        Ok(())
    }
}

pub struct VerifiedManifest<'a> {
    pub manifest: &'a PluginManifest,
    pub authorization: &'a PluginAuthorization,
    pub negotiated_api: NegotiatedApi,
}

#[derive(Debug, Error)]
pub enum PluginError {
    #[error(transparent)]
    Contract(#[from] PluginContractError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("invalid plugin host resource limits")]
    InvalidHostLimits,
    #[error("plugin component is larger than the configured byte limit")]
    ComponentTooLarge,
    #[error("plugin component compilation failed: {0}")]
    ComponentCompilation(String),
    #[error("plugin invocation failed: {0}")]
    Invocation(#[from] InvocationError),
    #[error("plugin component trapped: {0}")]
    WasmTrap(String),
    #[error("plugin node `{0}` was not declared by its signed manifest")]
    UndeclaredNode(String),
    #[error("plugin manifest and Rust implementation disagree")]
    ManifestProjectionMismatch,
    #[error("compiled plugin and invocation authorization disagree")]
    InvocationBindingMismatch,
    #[error("plugin component world disagrees with its signed manifest")]
    ComponentWorldMismatch,
    #[error("provider component binding set disagrees with its signed manifest")]
    ProviderBindingMismatch,
    #[error("provider invocation is unavailable for this component")]
    ProviderInvocationUnavailable,
    #[error("provider runtime activation was denied")]
    ProviderRuntimeActivationDenied,
    #[error("provider streaming operation failed: {0}")]
    ProviderStreaming(WorkerProviderStreamError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ComponentWorld {
    Legacy,
    ProviderV1,
    ProviderV2,
}

pub struct CompiledPlugin {
    component: Component,
    identifier: String,
    digest_sha256: String,
    manifest_projection: ComponentManifestProjection,
    provider_binding: Option<ProviderBindingSet>,
    provider_manifest_v2: Option<ProviderPluginManifestV2>,
    world: ComponentWorld,
}

impl CompiledPlugin {
    pub fn component(&self) -> &Component {
        &self.component
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }
}

pub struct WasmStoreState {
    limits: StoreLimits,
    invocation: Option<InvocationHost>,
    provider_runtime: Option<ProviderV2RuntimeHost>,
}

pub struct WasmPluginInstance {
    store: Store<WasmStoreState>,
    bindings: WasmBindings,
    expected_manifest_projection: ComponentManifestProjection,
    provider_binding: Option<ProviderBindingSet>,
    expected_provider_manifest_v2: Option<ProviderPluginManifestV2>,
    terminal: bool,
}

enum WasmBindings {
    Legacy(wit_contract::SimPlugin),
    ProviderV1(provider_wit_contract::SimProviderPlugin),
    ProviderV2(provider_v2_wit_contract::SimProviderPlugin),
}

#[derive(Debug)]
pub(crate) struct ProviderV2StreamRouteCall {
    call_id: u64,
    request: WorkerProviderStreamRequest,
    reply: SyncSender<WorkerProviderStreamResponse>,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "production consumer is sim-parity-provider-worker-stream-bridge"
    )
)]
impl ProviderV2StreamRouteCall {
    pub(crate) fn call_id(&self) -> u64 {
        self.call_id
    }

    pub(crate) fn request(&self) -> &WorkerProviderStreamRequest {
        &self.request
    }

    pub(crate) fn respond(
        self,
        response: WorkerProviderStreamResponse,
    ) -> Result<(), WorkerProviderStreamError> {
        if matches!(
            self.request,
            WorkerProviderStreamRequest::StartRequest { .. }
        ) {
            return Err(WorkerProviderStreamError::InvalidRequestAuthority);
        }
        self.reply.try_send(response).map_err(|error| match error {
            TrySendError::Full(_) => WorkerProviderStreamError::InvalidOrder,
            TrySendError::Disconnected(_) => WorkerProviderStreamError::RevokedHandle,
        })
    }

    fn into_start(
        self,
    ) -> Result<
        (
            u64,
            WorkerProviderInvocationContext,
            WorkerProviderRequestHead,
            SyncSender<WorkerProviderStreamResponse>,
        ),
        WorkerProviderStreamError,
    > {
        let WorkerProviderStreamRequest::StartRequest { context, head } = self.request else {
            return Err(WorkerProviderStreamError::InvalidOrder);
        };
        Ok((self.call_id, context, head, self.reply))
    }
}

#[derive(Debug)]
#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
pub(crate) struct ProviderV2BoundStartCall {
    call_id: u64,
    reply: SyncSender<WorkerProviderStreamResponse>,
}

#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
impl ProviderV2BoundStartCall {
    pub(crate) fn respond(
        self,
        response: Result<WorkerProviderStreamHandle, WorkerProviderStreamError>,
    ) -> Result<(), WorkerProviderStreamError> {
        self.reply
            .try_send(WorkerProviderStreamResponse::Stream(response))
            .map_err(|error| match error {
                TrySendError::Full(_) => WorkerProviderStreamError::InvalidOrder,
                TrySendError::Disconnected(_) => WorkerProviderStreamError::RevokedHandle,
            })
    }

    pub(crate) fn call_id(&self) -> u64 {
        self.call_id
    }
}

#[derive(Debug)]
pub(crate) enum ProviderV2StreamRouteMessage {
    Request(ProviderV2StreamRouteCall),
    Revoke {
        reply: SyncSender<Result<(), WorkerProviderStreamError>>,
    },
}

pub(crate) struct ProviderV2StreamRouteReceiver {
    receiver: Receiver<ProviderV2StreamRouteMessage>,
    revoke_receiver: Receiver<ProviderV2StreamRouteMessage>,
    revoked: Arc<AtomicBool>,
}

impl ProviderV2StreamRouteReceiver {
    pub(crate) fn try_receive(&self) -> Result<ProviderV2StreamRouteMessage, TryRecvError> {
        if self.revoked.load(Ordering::Acquire) {
            while self.receiver.try_recv().is_ok() {}
            return self.revoke_receiver.try_recv();
        }
        match self.revoke_receiver.try_recv() {
            Ok(message) => Ok(message),
            Err(TryRecvError::Disconnected) => Err(TryRecvError::Disconnected),
            Err(TryRecvError::Empty) => self.receiver.try_recv(),
        }
    }

    pub(crate) fn mark_revoked(&self) {
        self.revoked.store(true, Ordering::Release);
    }
}

impl Drop for ProviderV2StreamRouteReceiver {
    fn drop(&mut self) {
        self.mark_revoked();
    }
}

struct ProviderV2StreamRoute {
    sender: SyncSender<ProviderV2StreamRouteMessage>,
    revoke_sender: SyncSender<ProviderV2StreamRouteMessage>,
    revoked: Arc<AtomicBool>,
}

const MAX_PROVIDER_V2_SECRET_QUERIES: u32 = 32;

type ProviderV2HttpContinuationVerifier = Box<
    dyn for<'request> FnMut(
            &ProviderRuntimeHttpInvocationIdentity,
            &WorkerProviderHttpContinuationChallenge<'request>,
        ) -> Result<(), WorkerProviderStreamError>
        + Send,
>;

struct ProviderV2HttpContinuationAuthority {
    scope: ProviderRuntimeHttpInvocationIdentity,
    binding_sha256: WorkerSha256Digest,
    verify: ProviderV2HttpContinuationVerifier,
}

pub(crate) struct ProviderV2RuntimeHost {
    context: WorkerProviderInvocationContext,
    validator: WorkerProviderStreamTransportValidator,
    route: ProviderV2StreamRoute,
    cancellation: CancellationToken,
    reply_deadline: Duration,
    next_call_id: u64,
    secret_query_count: u32,
    present_secrets: BTreeSet<String>,
    bound: bool,
    stream_terminal: bool,
    revocation_complete: bool,
    terminal_failure: Option<WorkerProviderStreamError>,
    http_continuation: Option<ProviderV2HttpContinuationAuthority>,
}

pub struct ProviderV2WorkerStreamCall {
    call_id: u64,
    request: WorkerProviderStreamRequest,
    response: async_channel::Sender<WorkerProviderStreamResponse>,
}

impl ProviderV2WorkerStreamCall {
    pub fn call_id(&self) -> u64 {
        self.call_id
    }

    pub fn request(&self) -> &WorkerProviderStreamRequest {
        &self.request
    }

    pub fn respond(
        self,
        response: WorkerProviderStreamResponse,
    ) -> Result<(), WorkerProviderStreamError> {
        self.response
            .try_send(response)
            .map_err(|error| match error {
                async_channel::TrySendError::Full(_) => WorkerProviderStreamError::InvalidOrder,
                async_channel::TrySendError::Closed(_) => WorkerProviderStreamError::RevokedHandle,
            })
    }
}

impl ProviderV2RuntimeHost {
    fn route_disconnection_error(&self) -> WorkerProviderStreamError {
        if self.cancellation.is_cancelled() {
            WorkerProviderStreamError::Cancelled
        } else {
            WorkerProviderStreamError::RevokedHandle
        }
    }

    fn checked_for_worker_bridge(
        context: WorkerProviderInvocationContext,
        contract: WorkerProviderStreamingContract,
        cancellation: CancellationToken,
        sender: async_channel::Sender<ProviderV2WorkerStreamCall>,
    ) -> Result<Self, WorkerProviderStreamError> {
        let (route, receiver) = provider_v2_stream_route();
        let runtime = Self::checked_from_certified_capsule(context, contract, cancellation, route)?;
        drop(
            std::thread::Builder::new()
                .name("sim-provider-v2-worker-route".to_owned())
                .spawn(move || relay_provider_v2_worker_route(receiver, sender))
                .map_err(|_| WorkerProviderStreamError::HostFailure)?,
        );
        Ok(runtime)
    }

    fn checked_from_certified_capsule(
        context: WorkerProviderInvocationContext,
        contract: WorkerProviderStreamingContract,
        cancellation: CancellationToken,
        route: ProviderV2StreamRoute,
    ) -> Result<Self, WorkerProviderStreamError> {
        let reply_deadline = Duration::from_millis(
            contract
                .maximum_wait_milliseconds
                .min(MAX_WORKER_PROVIDER_WAIT_MILLISECONDS),
        );
        let validator = WorkerProviderStreamTransportValidator::checked_for_host_session(
            context.clone(),
            contract,
            cancellation.clone(),
        )?;
        Ok(Self {
            context,
            validator,
            route,
            cancellation,
            reply_deadline,
            next_call_id: 1,
            secret_query_count: 0,
            present_secrets: BTreeSet::new(),
            bound: false,
            stream_terminal: false,
            revocation_complete: false,
            terminal_failure: None,
            http_continuation: None,
        })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "certified multi-request coordinator wiring is pending"
        )
    )]
    pub(crate) fn bind_checked_http_continuations(
        &mut self,
        scope: ProviderRuntimeHttpInvocationIdentity,
        maximum_requests: u32,
        verify: ProviderV2HttpContinuationVerifier,
    ) -> Result<(), WorkerProviderStreamError> {
        self.check_active()?;
        if scope.context != self.context || self.bound || self.http_continuation.is_some() {
            return Err(WorkerProviderStreamError::InvalidRequestAuthority);
        }
        let binding_sha256 = http_invocation_binding_sha256(&scope)?;
        self.validator
            .bind_checked_http_invocation(binding_sha256.clone(), maximum_requests)?;
        self.http_continuation = Some(ProviderV2HttpContinuationAuthority {
            scope,
            binding_sha256,
            verify,
        });
        Ok(())
    }

    fn prepare_http_continuation(
        &mut self,
        context: &WorkerProviderInvocationContext,
        head: &WorkerProviderRequestHead,
    ) -> Result<(), WorkerProviderStreamError> {
        self.check_invocation_active()?;
        let ordinal = self.validator.next_http_request_ordinal()?;
        let authority = self
            .http_continuation
            .as_mut()
            .ok_or(WorkerProviderStreamError::InvalidRequestAuthority)?;
        if authority.scope.context != *context
            || http_invocation_binding_sha256(&authority.scope)? != authority.binding_sha256
        {
            return Err(WorkerProviderStreamError::InvalidRequestAuthority);
        }
        self.validator.prepare_checked_http_continuation(
            context,
            &authority.binding_sha256,
            ordinal,
            head,
            |challenge| (authority.verify)(&authority.scope, challenge),
        )
    }

    fn check_invocation_active(&self) -> Result<(), WorkerProviderStreamError> {
        self.cancellation
            .check()
            .map_err(|_| WorkerProviderStreamError::Cancelled)?;
        if self.revocation_complete || self.route.revoked.load(Ordering::Acquire) {
            return Err(self.route_disconnection_error());
        }
        if let Some(error) = &self.terminal_failure {
            return Err(error.clone());
        }
        Ok(())
    }

    fn exchange(
        &mut self,
        request: WorkerProviderStreamRequest,
    ) -> Result<WorkerProviderStreamResponse, WorkerProviderStreamError> {
        match self.exchange_active(request) {
            Ok(response) => Ok(response),
            Err(error) => {
                if (!self.stream_terminal || self.http_continuation.is_some())
                    && let Err(revoke_error) = self.revoke()
                {
                    eprintln!("provider-v2 route revocation after {error} failed: {revoke_error}");
                }
                Err(error)
            }
        }
    }

    fn wait_retry_delay(
        &self,
        request: &WorkerProviderRetryRequest,
    ) -> Result<(), WorkerProviderStreamError> {
        self.validator.validate_retry_request(request)?;
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(request.delay_milliseconds))
            .ok_or(WorkerProviderStreamError::WaitLimit)?;
        loop {
            self.cancellation
                .check()
                .map_err(|_| WorkerProviderStreamError::Cancelled)?;
            if self.route.revoked.load(Ordering::Acquire) {
                return Err(self.route_disconnection_error());
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::sleep(
                deadline
                    .saturating_duration_since(now)
                    .min(Duration::from_millis(10)),
            );
        }
        self.cancellation
            .check()
            .map_err(|_| WorkerProviderStreamError::Cancelled)
    }

    fn checked_retry_request(
        &self,
        handle: WorkerProviderStreamHandle,
        fallback_milliseconds: u64,
        now: SystemTime,
    ) -> Result<WorkerProviderRetryRequest, WorkerProviderStreamError> {
        let mut request = WorkerProviderRetryRequest {
            handle,
            delay_milliseconds: fallback_milliseconds,
        };
        let (status, retry_after) = self.validator.retry_response_head(&request)?;
        request.delay_milliseconds =
            resolve_retry_delay(status, fallback_milliseconds, retry_after, now)?;
        self.validator.validate_retry_request(&request)?;
        Ok(request)
    }

    fn begin_secret_query(&mut self) -> Result<(), WorkerProviderStreamError> {
        if self.http_continuation.is_some() {
            self.check_invocation_active()?;
        } else {
            self.check_active()?;
        }
        self.secret_query_count = self
            .secret_query_count
            .checked_add(1)
            .filter(|count| *count <= MAX_PROVIDER_V2_SECRET_QUERIES)
            .ok_or(WorkerProviderStreamError::InvalidRequestAuthority)?;
        Ok(())
    }

    fn record_secret_query_result(&mut self, identifier: String, present: bool) {
        if present {
            self.present_secrets.insert(identifier);
        } else {
            self.present_secrets.remove(&identifier);
        }
    }

    fn require_present_secret(
        &self,
        identifier: Option<&str>,
    ) -> Result<(), WorkerProviderStreamError> {
        if identifier.is_some_and(|identifier| !self.present_secrets.contains(identifier)) {
            return Err(WorkerProviderStreamError::InvalidRequestAuthority);
        }
        Ok(())
    }

    fn exchange_active(
        &mut self,
        request: WorkerProviderStreamRequest,
    ) -> Result<WorkerProviderStreamResponse, WorkerProviderStreamError> {
        let continuing = self.stream_terminal
            && matches!(&request, WorkerProviderStreamRequest::StartRequest { .. });
        if continuing {
            let WorkerProviderStreamRequest::StartRequest { context, head } = &request else {
                return Err(WorkerProviderStreamError::InvalidOrder);
            };
            self.prepare_http_continuation(context, head)?;
        }
        let final_progress = self.stream_terminal
            && self.terminal_failure.is_none()
            && matches!(
                &request,
                WorkerProviderStreamRequest::ReportProgress(progress)
                    if progress.completed == progress.total
            );
        let intermediate_progress = self.stream_terminal
            && self.http_continuation.is_some()
            && self.terminal_failure.is_none()
            && matches!(&request, WorkerProviderStreamRequest::ReportProgress(_));
        if final_progress || intermediate_progress {
            self.cancellation
                .check()
                .map_err(|_| WorkerProviderStreamError::Cancelled)?;
            if self.revocation_complete || self.route.revoked.load(Ordering::Acquire) {
                return Err(self.route_disconnection_error());
            }
        } else if continuing {
            self.check_invocation_active()?;
        } else {
            self.check_active()?;
        }
        let call_id = self.next_call_id;
        self.validator.validate_request(call_id, &request)?;
        let (reply, receiver) = sync_channel(1);
        match self
            .route
            .sender
            .try_send(ProviderV2StreamRouteMessage::Request(
                ProviderV2StreamRouteCall {
                    call_id,
                    request,
                    reply,
                },
            )) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(WorkerProviderStreamError::InvalidOrder),
            Err(TrySendError::Disconnected(_)) => {
                return Err(self.route_disconnection_error());
            }
        }
        let deadline = Instant::now()
            .checked_add(self.reply_deadline)
            .ok_or(WorkerProviderStreamError::TimedOut)?;
        let response = loop {
            self.cancellation
                .check()
                .map_err(|_| WorkerProviderStreamError::Cancelled)?;
            if self.route.revoked.load(Ordering::Acquire) {
                return Err(self.route_disconnection_error());
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(WorkerProviderStreamError::TimedOut);
            }
            match receiver.recv_timeout(
                deadline
                    .saturating_duration_since(now)
                    .min(Duration::from_millis(10)),
            ) {
                Ok(response) => break response,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(self.route_disconnection_error());
                }
            }
        };
        self.cancellation
            .check()
            .map_err(|_| WorkerProviderStreamError::Cancelled)?;
        self.validator.validate_response(call_id, &response)?;
        self.next_call_id = self
            .next_call_id
            .checked_add(1)
            .ok_or(WorkerProviderStreamError::InvalidOrder)?;
        if matches!(response, WorkerProviderStreamResponse::Stream(Ok(_))) {
            if !self.bound {
                self.bound = true;
            }
            if continuing {
                self.stream_terminal = false;
            }
        }
        response_result(&response)?;
        if let Some(disposition) = worker_response_terminal_disposition(&response) {
            self.stream_terminal = true;
            if let Err(error) = disposition {
                self.terminal_failure = Some(error);
                if let Err(revoke_error) = self.revoke() {
                    eprintln!("provider-v2 terminal route revocation failed: {revoke_error}");
                }
            }
        }
        Ok(response)
    }

    #[cfg(test)]
    fn require_bound(&self) -> Result<(), WorkerProviderStreamError> {
        self.check_active()?;
        if !self.bound {
            return Err(WorkerProviderStreamError::InvalidRequestAuthority);
        }
        Ok(())
    }

    fn worker_handle(
        &self,
        handle: provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamHandle,
    ) -> WorkerProviderStreamHandle {
        WorkerProviderStreamHandle {
            session_id: self.context.session_id,
            session_generation: self.context.session_generation,
            invocation: handle.invocation,
            slot: handle.slot,
            generation: handle.generation,
        }
    }

    fn ensure_completed(&self) -> Result<(), WorkerProviderStreamError> {
        if self.http_continuation.is_some() {
            self.check_invocation_active()?;
        }
        if let Some(error) = &self.terminal_failure {
            return Err(error.clone());
        }
        if !self.stream_terminal || self.revocation_complete {
            return Err(WorkerProviderStreamError::InvalidTerminal);
        }
        Ok(())
    }

    fn completed_handle(&self) -> Result<WorkerProviderStreamHandle, WorkerProviderStreamError> {
        self.ensure_completed()?;
        self.validator
            .primary_handle()
            .ok_or(WorkerProviderStreamError::InvalidHandle)
    }

    fn complete_non_revoking(&mut self) -> Result<(), WorkerProviderStreamError> {
        self.ensure_completed()?;
        if self.http_continuation.is_some() {
            self.validator.finish_checked_http_invocation()?;
        }
        self.http_continuation = None;
        self.revocation_complete = true;
        Ok(())
    }

    fn check_active(&self) -> Result<(), WorkerProviderStreamError> {
        self.cancellation
            .check()
            .map_err(|_| WorkerProviderStreamError::Cancelled)?;
        if self.stream_terminal
            || self.revocation_complete
            || self.route.revoked.load(Ordering::Acquire)
        {
            return Err(WorkerProviderStreamError::RevokedHandle);
        }
        Ok(())
    }

    fn revoke(&mut self) -> Result<(), WorkerProviderStreamError> {
        if self.revocation_complete {
            return Ok(());
        }
        self.stream_terminal = true;
        self.revocation_complete = true;
        self.http_continuation = None;
        self.validator.revoke();
        self.route.revoked.store(true, Ordering::Release);
        let (reply, receiver) = sync_channel(1);
        match self
            .route
            .revoke_sender
            .try_send(ProviderV2StreamRouteMessage::Revoke { reply })
        {
            Ok(()) => receiver
                .recv_timeout(Duration::from_millis(10))
                .map_err(|_| WorkerProviderStreamError::HostFailure)?,
            Err(TrySendError::Full(_)) => Err(WorkerProviderStreamError::InvalidOrder),
            Err(TrySendError::Disconnected(_)) => Err(WorkerProviderStreamError::RevokedHandle),
        }
    }
}

fn http_invocation_binding_sha256(
    scope: &ProviderRuntimeHttpInvocationIdentity,
) -> Result<WorkerSha256Digest, WorkerProviderStreamError> {
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-provider-certified-http-binding-v2\0");
    digest.update(scope.context.session_id.as_bytes());
    digest.update(scope.context.session_generation.to_le_bytes());
    digest.update(scope.context.invocation.to_le_bytes());
    digest.update(scope.context.generation.to_le_bytes());
    for value in [
        &scope.profile_id,
        &scope.principal_id,
        &scope.prompt_id,
        &scope.prompt_sha256,
        &scope.attempt_id,
        &scope.node_id,
        &scope.compiled_plan_sha256,
        &scope.registry_digest_sha256,
        &scope.component_digest_sha256,
        &scope.authorization_generation_sha256,
        &scope.binding_set_sha256,
        &scope.provider,
        &scope.provider_manifest_sha256,
    ] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    digest.update(scope.activation_request_ordinal.to_le_bytes());
    digest.update(scope.registry_generation.to_le_bytes());
    digest.update(scope.component_generation.to_le_bytes());
    digest.update(scope.binding_generation.to_le_bytes());
    WorkerSha256Digest::new(encode_hex(&digest.finalize()))
        .map_err(|_| WorkerProviderStreamError::InvalidRequestAuthority)
}

fn relay_provider_v2_worker_route(
    receiver: ProviderV2StreamRouteReceiver,
    sender: async_channel::Sender<ProviderV2WorkerStreamCall>,
) {
    loop {
        match receiver.try_receive() {
            Ok(ProviderV2StreamRouteMessage::Request(call)) => {
                let (response, response_receiver) = async_channel::bounded(1);
                let worker_call = ProviderV2WorkerStreamCall {
                    call_id: call.call_id,
                    request: call.request.clone(),
                    response,
                };
                let mut worker_call = worker_call;
                loop {
                    if relay_provider_v2_revoke(&receiver) {
                        return;
                    }
                    match sender.try_send(worker_call) {
                        Ok(()) => break,
                        Err(async_channel::TrySendError::Full(returned)) => {
                            worker_call = returned;
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(async_channel::TrySendError::Closed(_)) => {
                            receiver.mark_revoked();
                            return;
                        }
                    }
                }
                let response = loop {
                    if relay_provider_v2_revoke(&receiver) {
                        return;
                    }
                    match response_receiver.try_recv() {
                        Ok(response) => break response,
                        Err(async_channel::TryRecvError::Empty) => {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(async_channel::TryRecvError::Closed) => {
                            receiver.mark_revoked();
                            return;
                        }
                    }
                };
                if call.reply.try_send(response).is_err() {
                    receiver.mark_revoked();
                    break;
                }
            }
            Ok(ProviderV2StreamRouteMessage::Revoke { reply }) => {
                receiver.mark_revoked();
                if let Err(error) = reply.try_send(Ok(())) {
                    eprintln!("provider-v2 worker route revoke reply was lost: {error}");
                }
                break;
            }
            Err(TryRecvError::Empty) => std::thread::sleep(Duration::from_millis(1)),
            Err(TryRecvError::Disconnected) => break,
        }
    }
}

fn relay_provider_v2_revoke(receiver: &ProviderV2StreamRouteReceiver) -> bool {
    match receiver.revoke_receiver.try_recv() {
        Ok(ProviderV2StreamRouteMessage::Revoke { reply }) => {
            receiver.mark_revoked();
            if let Err(error) = reply.try_send(Ok(())) {
                eprintln!("provider-v2 worker route revoke reply was lost: {error}");
            }
            true
        }
        Ok(ProviderV2StreamRouteMessage::Request(_)) => {
            receiver.mark_revoked();
            true
        }
        Err(TryRecvError::Empty) => receiver.revoked.load(Ordering::Acquire),
        Err(TryRecvError::Disconnected) => true,
    }
}

impl Drop for ProviderV2RuntimeHost {
    fn drop(&mut self) {
        if let Err(error) = self.revoke() {
            eprintln!("provider-v2 route revocation failed: {error}");
        }
    }
}

fn response_result(
    response: &WorkerProviderStreamResponse,
) -> Result<(), WorkerProviderStreamError> {
    match response {
        WorkerProviderStreamResponse::Stream(result) => {
            result.as_ref().map(|_| ()).map_err(Clone::clone)
        }
        WorkerProviderStreamResponse::Unit(result) => result.clone(),
        WorkerProviderStreamResponse::Wait(result) => {
            result.as_ref().map(|_| ()).map_err(Clone::clone)
        }
        WorkerProviderStreamResponse::UploadCompletion(result) => {
            result.as_ref().map(|_| ()).map_err(Clone::clone)
        }
        WorkerProviderStreamResponse::Cost(result) => {
            result.as_ref().map(|_| ()).map_err(Clone::clone)
        }
    }
}

fn worker_response_terminal_disposition(
    response: &WorkerProviderStreamResponse,
) -> Option<Result<(), WorkerProviderStreamError>> {
    match response {
        WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Cancelled)) => {
            Some(Err(WorkerProviderStreamError::Cancelled))
        }
        WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
            WorkerProviderResponseFrame {
                event: WorkerProviderResponseFrameEvent::Terminal(terminal),
                ..
            },
        ))) => Some(match terminal {
            WorkerProviderTerminal::Completed(_) => Ok(()),
            WorkerProviderTerminal::Failed { .. } => {
                Err(WorkerProviderStreamError::InvalidTerminal)
            }
            WorkerProviderTerminal::Cancelled => Err(WorkerProviderStreamError::Cancelled),
        }),
        _ => None,
    }
}

fn provider_v2_stream_route() -> (ProviderV2StreamRoute, ProviderV2StreamRouteReceiver) {
    let (sender, receiver) = sync_channel(MAX_WORKER_PROVIDER_PENDING_CALLS);
    let (revoke_sender, revoke_receiver) = sync_channel(1);
    let revoked = Arc::new(AtomicBool::new(false));
    (
        ProviderV2StreamRoute {
            sender,
            revoke_sender,
            revoked: revoked.clone(),
        },
        ProviderV2StreamRouteReceiver {
            receiver,
            revoke_receiver,
            revoked,
        },
    )
}

fn worker_streaming_contract(
    contract: &ProviderStreamingContractV2,
) -> WorkerProviderStreamingContract {
    WorkerProviderStreamingContract {
        methods: contract
            .methods
            .iter()
            .copied()
            .map(worker_http_method)
            .collect(),
        maximum_headers: contract.maximum_headers,
        maximum_header_bytes: contract.maximum_header_bytes,
        maximum_request_body_bytes: contract.maximum_request_body_bytes,
        maximum_response_body_bytes: contract.maximum_response_body_bytes,
        maximum_chunk_bytes: contract.maximum_chunk_bytes,
        maximum_ndjson_line_bytes: contract.maximum_ndjson_line_bytes,
        maximum_wait_milliseconds: contract.maximum_wait_milliseconds,
        maximum_uploads: contract.maximum_uploads,
        maximum_upload_body_bytes: contract.maximum_upload_body_bytes,
        maximum_cost_requests: contract.maximum_cost_requests,
        maximum_progress_total: contract.maximum_progress_total,
        uploads: contract.uploads,
        cost_requests: contract.cost_requests,
    }
}

fn sdk_provider_streaming_contract(
    contract: provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamingContract,
) -> ProviderStreamingContractV2 {
    use provider_v2_wit_contract::zed::sim_provider_plugin::types::HttpMethod;
    ProviderStreamingContractV2 {
        methods: contract
            .methods
            .into_iter()
            .map(|method| match method {
                HttpMethod::Delete => ProviderHttpMethodV2::Delete,
                HttpMethod::Get => ProviderHttpMethodV2::Get,
                HttpMethod::Head => ProviderHttpMethodV2::Head,
                HttpMethod::Options => ProviderHttpMethodV2::Options,
                HttpMethod::Patch => ProviderHttpMethodV2::Patch,
                HttpMethod::Post => ProviderHttpMethodV2::Post,
                HttpMethod::Put => ProviderHttpMethodV2::Put,
            })
            .collect(),
        maximum_headers: contract.maximum_headers,
        maximum_header_bytes: contract.maximum_header_bytes,
        maximum_request_body_bytes: contract.maximum_request_body_bytes,
        maximum_response_body_bytes: contract.maximum_response_body_bytes,
        maximum_chunk_bytes: contract.maximum_chunk_bytes,
        maximum_ndjson_line_bytes: contract.maximum_ndjson_line_bytes,
        maximum_wait_milliseconds: contract.maximum_wait_milliseconds,
        maximum_uploads: contract.maximum_uploads,
        maximum_upload_body_bytes: contract.maximum_upload_body_bytes,
        maximum_cost_requests: contract.maximum_cost_requests,
        maximum_progress_total: contract.maximum_progress_total,
        uploads: contract.uploads,
        cost_requests: contract.cost_requests,
    }
}

fn worker_http_method(method: ProviderHttpMethodV2) -> WorkerProviderHttpMethod {
    match method {
        ProviderHttpMethodV2::Delete => WorkerProviderHttpMethod::Delete,
        ProviderHttpMethodV2::Get => WorkerProviderHttpMethod::Get,
        ProviderHttpMethodV2::Head => WorkerProviderHttpMethod::Head,
        ProviderHttpMethodV2::Options => WorkerProviderHttpMethod::Options,
        ProviderHttpMethodV2::Patch => WorkerProviderHttpMethod::Patch,
        ProviderHttpMethodV2::Post => WorkerProviderHttpMethod::Post,
        ProviderHttpMethodV2::Put => WorkerProviderHttpMethod::Put,
    }
}

fn worker_provider_request_head(
    request: provider_v2_wit_contract::zed::sim_provider_plugin::types::RequestHead,
) -> WorkerProviderRequestHead {
    use provider_v2_wit_contract::zed::sim_provider_plugin::types::HttpMethod;
    WorkerProviderRequestHead {
        endpoint: request.endpoint,
        secret_id: request.secret_id,
        method: match request.method {
            HttpMethod::Delete => WorkerProviderHttpMethod::Delete,
            HttpMethod::Get => WorkerProviderHttpMethod::Get,
            HttpMethod::Head => WorkerProviderHttpMethod::Head,
            HttpMethod::Options => WorkerProviderHttpMethod::Options,
            HttpMethod::Patch => WorkerProviderHttpMethod::Patch,
            HttpMethod::Post => WorkerProviderHttpMethod::Post,
            HttpMethod::Put => WorkerProviderHttpMethod::Put,
        },
        headers: request
            .headers
            .into_iter()
            .map(|header| WorkerProviderHeader {
                name: header.name,
                value: header.value,
            })
            .collect(),
        declared_body_bytes: request.declared_body_bytes,
    }
}

fn wit_provider_stream_handle(
    handle: WorkerProviderStreamHandle,
) -> provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamHandle {
    provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamHandle {
        invocation: handle.invocation,
        slot: handle.slot,
        generation: handle.generation,
    }
}

fn wit_provider_wait_outcome(
    outcome: WorkerProviderWaitOutcome,
) -> provider_v2_wit_contract::zed::sim_provider_plugin::types::WaitOutcome {
    use provider_v2_wit_contract::zed::sim_provider_plugin::types as wit;
    match outcome {
        WorkerProviderWaitOutcome::TimedOut => wit::WaitOutcome::TimedOut,
        WorkerProviderWaitOutcome::Cancelled => wit::WaitOutcome::Cancelled,
        WorkerProviderWaitOutcome::Frame(frame) => wit::WaitOutcome::Frame(wit::ResponseFrame {
            handle: wit_provider_stream_handle(frame.handle),
            sequence: frame.sequence,
            event: match frame.event {
                WorkerProviderResponseFrameEvent::Head(head) => {
                    wit::ResponseFrameEvent::Head(wit::ResponseHead {
                        status: head.status,
                        headers: head
                            .headers
                            .into_iter()
                            .map(|header| wit::Header {
                                name: header.name,
                                value: header.value,
                            })
                            .collect(),
                    })
                }
                WorkerProviderResponseFrameEvent::Chunk(chunk) => {
                    wit::ResponseFrameEvent::Chunk(match chunk {
                        WorkerProviderResponseChunk::Binary(bytes) => {
                            wit::ResponseChunkValue::Binary(bytes)
                        }
                        WorkerProviderResponseChunk::Text(text) => {
                            wit::ResponseChunkValue::Text(text)
                        }
                        WorkerProviderResponseChunk::NdjsonLine(line) => {
                            wit::ResponseChunkValue::NdjsonLine(line)
                        }
                    })
                }
                WorkerProviderResponseFrameEvent::Terminal(terminal) => {
                    wit::ResponseFrameEvent::Terminal(match terminal {
                        WorkerProviderTerminal::Completed(receipt) => {
                            wit::Terminal::Completed(receipt)
                        }
                        WorkerProviderTerminal::Failed { code, message } => {
                            wit::Terminal::Failed((code, message))
                        }
                        WorkerProviderTerminal::Cancelled => wit::Terminal::Cancelled,
                    })
                }
            },
        }),
    }
}

fn wit_provider_stream_error(
    error: WorkerProviderStreamError,
) -> provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError {
    use provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError;
    match error {
        WorkerProviderStreamError::Cancelled => StreamError::Cancelled,
        WorkerProviderStreamError::TimedOut => StreamError::TimedOut,
        WorkerProviderStreamError::HostFailure => StreamError::HostFailure,
        WorkerProviderStreamError::InvalidContract => StreamError::InvalidContract,
        WorkerProviderStreamError::InvalidHandle => StreamError::InvalidHandle,
        WorkerProviderStreamError::ForeignHandle
        | WorkerProviderStreamError::ForeignSession
        | WorkerProviderStreamError::ForeignInvocation => StreamError::ForeignHandle,
        WorkerProviderStreamError::RevokedHandle
        | WorkerProviderStreamError::StaleSession
        | WorkerProviderStreamError::StaleGeneration => StreamError::RevokedHandle,
        WorkerProviderStreamError::InvalidMethod => StreamError::InvalidMethod,
        WorkerProviderStreamError::InvalidHeaders => StreamError::InvalidHeaders,
        WorkerProviderStreamError::BodyLimit => StreamError::BodyLimit,
        WorkerProviderStreamError::ChunkLimit => StreamError::ChunkLimit,
        WorkerProviderStreamError::InvalidNdjsonLine => StreamError::InvalidNdjsonLine,
        WorkerProviderStreamError::InvalidSequence => StreamError::InvalidSequence,
        WorkerProviderStreamError::InvalidOrder => StreamError::InvalidOrder,
        WorkerProviderStreamError::WaitLimit => StreamError::WaitLimit,
        WorkerProviderStreamError::InvalidUpload => StreamError::InvalidUpload,
        WorkerProviderStreamError::InvalidCostRequest => StreamError::InvalidCostRequest,
        WorkerProviderStreamError::InvalidProgress => StreamError::InvalidProgress,
        WorkerProviderStreamError::InvalidTerminal => StreamError::InvalidTerminal,
        WorkerProviderStreamError::InvalidInvocationResult => StreamError::InvalidInvocationResult,
        WorkerProviderStreamError::InvalidRequestAuthority => StreamError::InvalidRequestAuthority,
    }
}

fn worker_provider_stream_error(
    error: provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError,
) -> WorkerProviderStreamError {
    use provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError;
    match error {
        StreamError::Cancelled => WorkerProviderStreamError::Cancelled,
        StreamError::TimedOut => WorkerProviderStreamError::TimedOut,
        StreamError::HostFailure => WorkerProviderStreamError::HostFailure,
        StreamError::InvalidContract => WorkerProviderStreamError::InvalidContract,
        StreamError::InvalidHandle => WorkerProviderStreamError::InvalidHandle,
        StreamError::ForeignHandle => WorkerProviderStreamError::ForeignHandle,
        StreamError::RevokedHandle => WorkerProviderStreamError::RevokedHandle,
        StreamError::InvalidMethod => WorkerProviderStreamError::InvalidMethod,
        StreamError::InvalidHeaders => WorkerProviderStreamError::InvalidHeaders,
        StreamError::BodyLimit => WorkerProviderStreamError::BodyLimit,
        StreamError::ChunkLimit => WorkerProviderStreamError::ChunkLimit,
        StreamError::InvalidNdjsonLine => WorkerProviderStreamError::InvalidNdjsonLine,
        StreamError::InvalidSequence => WorkerProviderStreamError::InvalidSequence,
        StreamError::InvalidOrder => WorkerProviderStreamError::InvalidOrder,
        StreamError::WaitLimit => WorkerProviderStreamError::WaitLimit,
        StreamError::InvalidUpload => WorkerProviderStreamError::InvalidUpload,
        StreamError::InvalidCostRequest => WorkerProviderStreamError::InvalidCostRequest,
        StreamError::InvalidProgress => WorkerProviderStreamError::InvalidProgress,
        StreamError::InvalidTerminal => WorkerProviderStreamError::InvalidTerminal,
        StreamError::InvalidInvocationResult => WorkerProviderStreamError::InvalidInvocationResult,
        StreamError::InvalidRequestAuthority => WorkerProviderStreamError::InvalidRequestAuthority,
    }
}

fn sdk_provider_invocation_result(
    result: provider_v2_wit_contract::zed::sim_provider_plugin::types::InvocationResult,
    invocation: &Option<InvocationHost>,
) -> Result<ProviderInvocationResultV2, PluginError> {
    let registry = invocation
        .as_ref()
        .map(|invocation| &invocation.registry)
        .ok_or_else(|| PluginError::Invocation(InvocationError::RevokedHandle))?;
    let result = ProviderInvocationResultV2 {
        outputs: result
            .outputs
            .into_iter()
            .map(|output| {
                Ok(ProviderMaterializedOutputV2 {
                    port_id: output.port_id,
                    value: ProviderEncodedValueV2 {
                        type_id: CanonicalTypeId::from_str(&output.value.type_id).map_err(
                            |_| {
                                PluginError::ProviderStreaming(
                                    WorkerProviderStreamError::InvalidInvocationResult,
                                )
                            },
                        )?,
                        family: sdk_value_family(output.value.family),
                        abi_bytes: output.value.abi_bytes,
                    },
                })
            })
            .collect::<Result<Vec<_>, PluginError>>()?,
        receipt: result.receipt,
    };
    result.validate(registry).map_err(|_| {
        PluginError::ProviderStreaming(WorkerProviderStreamError::InvalidInvocationResult)
    })?;
    Ok(result)
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInvocationResult {
    pub outputs: BTreeMap<String, Vec<PluginValue>>,
    pub output_presence: BTreeMap<String, bool>,
    pub effects: CapabilityEffects,
    receipts: Vec<Vec<u8>>,
    #[serde(skip)]
    resolved_provider_results: Vec<ResolvedProviderResult>,
    #[serde(skip)]
    materialized_provider_response: Option<HostMaterializedProviderResponse>,
}

pub struct ProviderV2InvocationProposal {
    result: ProviderInvocationResultV2,
    runtime: ProviderV2RuntimeHost,
}

pub struct ProviderV2WorkerPendingInvocation {
    proposal: ProviderV2InvocationProposal,
    envelope: WorkerProviderV2InvocationEnvelope,
    registry: TypeRegistry,
    cancellation: CancellationToken,
}

impl ProviderV2WorkerPendingInvocation {
    pub fn result(&self) -> &ProviderInvocationResultV2 {
        self.proposal.result()
    }

    pub fn finalize(
        self,
        finalization: &WorkerProviderV2ProposalFinalization,
    ) -> Result<(), WorkerProviderStreamError> {
        if finalization.context != *self.envelope.context()
            || finalization.proposal_generation != self.envelope.proposal_generation()
            || finalization.finalization_nonce != self.envelope.finalization_nonce()
        {
            return Err(WorkerProviderStreamError::StaleGeneration);
        }
        finalization.validate()?;
        let receipt_identity_sha256 = encode_hex(&Sha256::digest(&self.proposal.result.receipt));
        let materialization_identity_sha256 = encode_hex(
            &sim_runtime::provider_invocation_result_v2_materialization_identity(
                &self.proposal.result,
                &self.registry,
                &self.cancellation,
            )
            .map_err(|_| WorkerProviderStreamError::InvalidInvocationResult)?,
        );
        if finalization.receipt_identity_sha256.as_str() != receipt_identity_sha256
            || finalization.materialization_identity_sha256.as_str()
                != materialization_identity_sha256
        {
            return Err(WorkerProviderStreamError::InvalidInvocationResult);
        }
        let ProviderV2InvocationProposal {
            result: _,
            mut runtime,
        } = self.proposal;
        if runtime.completed_handle()? != finalization.handle {
            return Err(WorkerProviderStreamError::ForeignHandle);
        }
        runtime.complete_non_revoking()
    }
}

impl ProviderV2InvocationProposal {
    pub fn result(&self) -> &ProviderInvocationResultV2 {
        &self.result
    }

    #[expect(
        dead_code,
        reason = "production consumer is sim-parity-provider-worker-stream-bridge"
    )]
    pub(crate) fn into_parts(self) -> (ProviderInvocationResultV2, ProviderV2RuntimeHost) {
        (self.result, self.runtime)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct HostMaterializedProviderResponse {
    original_identity: ProviderInvocationIdentity,
    original_receipt: Vec<u8>,
    original_result_sha256: String,
    response: ProviderTransportResponse,
}

impl ProviderInvocationResult {
    pub(crate) fn set_materialized_provider_response(
        &mut self,
        original_identity: ProviderInvocationIdentity,
        original_receipt: Vec<u8>,
        original_result_sha256: String,
        response: ProviderTransportResponse,
    ) {
        self.materialized_provider_response = Some(HostMaterializedProviderResponse {
            original_identity,
            original_receipt,
            original_result_sha256,
            response,
        });
    }

    pub(crate) fn verify_materialized_provider_response(
        &self,
        signed_namespace: &str,
        class_type: &str,
    ) -> Result<(), PluginError> {
        let Some(materialized) = self.materialized_provider_response.as_ref() else {
            return Ok(());
        };
        let [resolved] = self.resolved_provider_results.as_slice() else {
            return Err(PluginError::Invocation(InvocationError::HostFailure(
                "native image materialization lost its exact original settlement".to_owned(),
            )));
        };
        let [receipt] = self.receipts.as_slice() else {
            return Err(PluginError::Invocation(InvocationError::HostFailure(
                "native image materialization lost its exact issued receipt".to_owned(),
            )));
        };
        if receipt != &materialized.original_receipt
            || materialized.original_identity.plugin_id() != signed_namespace
            || materialized.response.class_type() != class_type
            || resolved.identity() != &materialized.original_identity
            || format!("{:x}", Sha256::digest(resolved.response()))
                != materialized.original_result_sha256
        {
            return Err(PluginError::Invocation(InvocationError::HostFailure(
                "native image materialization disagrees with its authenticated original response"
                    .to_owned(),
            )));
        }
        Ok(())
    }

    pub(crate) fn take_materialized_provider_response(
        &mut self,
        signed_namespace: &str,
        class_type: &str,
    ) -> Result<Option<ProviderTransportResponse>, PluginError> {
        self.verify_materialized_provider_response(signed_namespace, class_type)?;
        Ok(self
            .materialized_provider_response
            .take()
            .map(|materialized| materialized.response))
    }

    pub fn receipts(&self) -> &[Vec<u8>] {
        &self.receipts
    }

    pub fn resolved_provider_results(&self) -> &[ResolvedProviderResult] {
        &self.resolved_provider_results
    }

    pub(crate) fn set_resolved_provider_results(
        &mut self,
        resolved_provider_results: Vec<ResolvedProviderResult>,
    ) {
        self.resolved_provider_results = resolved_provider_results;
    }

    pub(crate) fn take_resolved_provider_results(&mut self) -> Vec<ResolvedProviderResult> {
        mem::take(&mut self.resolved_provider_results)
    }
}

impl WasmPluginInstance {
    pub fn manifest_bytes(&mut self) -> Result<ComponentManifestProjection, PluginError> {
        self.check_active()?;
        let result = match &self.bindings {
            WasmBindings::Legacy(bindings) => bindings
                .zed_sim_plugin_plugin()
                .call_manifest(&mut self.store),
            WasmBindings::ProviderV1(bindings) => bindings
                .zed_sim_plugin_plugin()
                .call_manifest(&mut self.store),
            WasmBindings::ProviderV2(_) => return self.provider_v2_manifest_projection(),
        };
        match result {
            Ok(projection) => {
                let projection = match sdk_manifest_projection(projection) {
                    Ok(projection) => projection,
                    Err(_) => {
                        self.abort();
                        return Err(PluginError::ManifestProjectionMismatch);
                    }
                };
                if let Err(error) =
                    validate_component_projection(&self.expected_manifest_projection, &projection)
                {
                    self.abort();
                    return Err(error);
                }
                Ok(projection)
            }
            Err(error) => Err(self.wasm_call_error(error)),
        }
    }

    pub fn create_node(&mut self, node_id: &str) -> Result<u64, PluginError> {
        self.check_active()?;
        let declared_for_invocation = self
            .store
            .data()
            .invocation
            .as_ref()
            .is_some_and(|invocation| invocation.node.id == node_id);
        if !declared_for_invocation {
            self.abort();
            return Err(PluginError::UndeclaredNode(node_id.to_owned()));
        }
        let result = match &self.bindings {
            WasmBindings::Legacy(bindings) => bindings
                .zed_sim_plugin_plugin()
                .call_create_node(&mut self.store, node_id),
            WasmBindings::ProviderV1(bindings) => bindings
                .zed_sim_plugin_plugin()
                .call_create_node(&mut self.store, node_id),
            WasmBindings::ProviderV2(_) => {
                self.abort();
                return Err(PluginError::ProviderInvocationUnavailable);
            }
        };
        match result {
            Ok(Ok(instance)) => Ok(instance),
            Ok(Err(error)) => {
                self.abort();
                Err(PluginError::Invocation(sdk_error(error)))
            }
            Err(error) => Err(self.wasm_call_error(error)),
        }
    }

    pub fn provider_binding_set(&mut self) -> Result<ProviderBindingSet, PluginError> {
        self.check_active()?;
        let expected = self
            .provider_binding
            .clone()
            .ok_or(PluginError::ProviderInvocationUnavailable)?;
        let WasmBindings::ProviderV1(bindings) = &self.bindings else {
            return Err(PluginError::ProviderInvocationUnavailable);
        };
        let binding_set = bindings
            .zed_sim_plugin_provider_binding()
            .call_binding_set(&mut self.store)
            .map_err(|error| self.wasm_call_error(error))?;
        let actual = match sdk_provider_binding_set(binding_set) {
            Ok(actual) => actual,
            Err(error) => {
                self.abort();
                return Err(error);
            }
        };
        if actual != expected {
            self.abort();
            return Err(PluginError::ProviderBindingMismatch);
        }
        Ok(actual)
    }

    fn provider_v2_manifest_projection(
        &mut self,
    ) -> Result<ComponentManifestProjection, PluginError> {
        self.check_active()?;
        let expected = self
            .expected_provider_manifest_v2
            .clone()
            .ok_or(PluginError::ProviderInvocationUnavailable)?;
        let WasmBindings::ProviderV2(bindings) = &self.bindings else {
            return Err(PluginError::ProviderInvocationUnavailable);
        };
        let projection = bindings
            .zed_sim_provider_plugin_provider_node()
            .call_manifest(&mut self.store)
            .map_err(|error| self.wasm_call_error(error))?;
        let manifest_projection = sdk_manifest_projection(projection.manifest)?;
        let provider_binding = sdk_provider_binding_set(projection.provider_binding)?;
        let streaming = sdk_provider_streaming_contract(projection.streaming);
        if projection.schema_version != expected.schema_version
            || projection.component_world != expected.component_world
            || manifest_projection != expected.manifest.component_projection()
            || provider_binding
                != *expected
                    .manifest
                    .provider_binding
                    .as_ref()
                    .ok_or(PluginError::ProviderBindingMismatch)?
            || streaming != expected.streaming
        {
            self.abort();
            return Err(PluginError::ManifestProjectionMismatch);
        }
        Ok(manifest_projection)
    }

    pub(crate) fn invoke_provider_v2(
        mut self,
        node_id: &str,
    ) -> Result<ProviderV2InvocationProposal, PluginError> {
        self.check_active()?;
        let context = self
            .store
            .data()
            .provider_runtime
            .as_ref()
            .map(|runtime| runtime.context.clone())
            .ok_or(PluginError::ProviderRuntimeActivationDenied)?;
        if !self
            .expected_manifest_projection
            .nodes
            .iter()
            .any(|node| node.id == node_id)
        {
            self.abort();
            return Err(PluginError::ProviderRuntimeActivationDenied);
        }
        let WasmBindings::ProviderV2(bindings) = &self.bindings else {
            self.abort();
            return Err(PluginError::ProviderInvocationUnavailable);
        };
        let wit_context =
            provider_v2_wit_contract::zed::sim_provider_plugin::types::InvocationContext {
                invocation: context.invocation,
                generation: context.generation,
            };
        let result = match bindings
            .zed_sim_provider_plugin_provider_node()
            .call_invoke(&mut self.store, wit_context, node_id)
        {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => {
                self.abort();
                return Err(PluginError::ProviderStreaming(
                    worker_provider_stream_error(error),
                ));
            }
            Err(error) => return Err(self.wasm_call_error(error)),
        };
        if let Err(error) = self
            .store
            .data()
            .provider_runtime
            .as_ref()
            .ok_or(PluginError::ProviderRuntimeActivationDenied)?
            .ensure_completed()
        {
            self.abort();
            return Err(PluginError::ProviderStreaming(error));
        }
        let result = sdk_provider_invocation_result(result, &self.store.data().invocation)?;
        let invocation = self
            .store
            .data_mut()
            .invocation
            .take()
            .ok_or_else(|| PluginError::Invocation(InvocationError::RevokedHandle))?;
        invocation.finish_provider_v2_inputs()?;
        let runtime = self
            .store
            .data_mut()
            .provider_runtime
            .take()
            .ok_or(PluginError::ProviderRuntimeActivationDenied)?;
        self.terminal = true;
        Ok(ProviderV2InvocationProposal { result, runtime })
    }

    pub fn invoke(&mut self, instance: u64) -> Result<(), PluginError> {
        self.check_active()?;
        let result = match &self.bindings {
            WasmBindings::Legacy(bindings) => bindings
                .zed_sim_plugin_plugin()
                .call_invoke(&mut self.store, instance),
            WasmBindings::ProviderV1(bindings) => bindings
                .zed_sim_plugin_plugin()
                .call_invoke(&mut self.store, instance),
            WasmBindings::ProviderV2(_) => {
                self.abort();
                return Err(PluginError::ProviderInvocationUnavailable);
            }
        };
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                self.abort();
                Err(PluginError::Invocation(sdk_error(error)))
            }
            Err(error) => Err(self.wasm_call_error(error)),
        }
    }

    pub fn cancel(&mut self, instance: u64, reason: CancelReason) -> Result<(), PluginError> {
        self.check_active()?;
        let reason = wit_cancel_reason(reason);
        let result = match &self.bindings {
            WasmBindings::Legacy(bindings) => {
                bindings
                    .zed_sim_plugin_plugin()
                    .call_cancel(&mut self.store, instance, reason)
            }
            WasmBindings::ProviderV1(bindings) => {
                bindings
                    .zed_sim_plugin_plugin()
                    .call_cancel(&mut self.store, instance, reason)
            }
            WasmBindings::ProviderV2(_) => {
                self.abort();
                return Err(PluginError::ProviderInvocationUnavailable);
            }
        };
        match result {
            Ok(Ok(())) => {
                self.abort();
                Ok(())
            }
            Ok(Err(error)) => {
                self.abort();
                Err(PluginError::Invocation(sdk_error(error)))
            }
            Err(error) => Err(self.wasm_call_error(error)),
        }
    }

    pub fn drop_node(&mut self, instance: u64) -> Result<(), PluginError> {
        self.check_active()?;
        let result = match &self.bindings {
            WasmBindings::Legacy(bindings) => bindings
                .zed_sim_plugin_plugin()
                .call_drop_node(&mut self.store, instance),
            WasmBindings::ProviderV1(bindings) => bindings
                .zed_sim_plugin_plugin()
                .call_drop_node(&mut self.store, instance),
            WasmBindings::ProviderV2(_) => {
                self.abort();
                return Err(PluginError::ProviderInvocationUnavailable);
            }
        };
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                self.abort();
                Err(PluginError::WasmTrap(sanitize_diagnostic(
                    &error.to_string(),
                )))
            }
        }
    }

    pub fn finish(mut self) -> Result<InvocationResult, PluginError> {
        let invocation = self
            .store
            .data_mut()
            .invocation
            .take()
            .ok_or_else(|| PluginError::Invocation(InvocationError::RevokedHandle))?;
        let result = invocation.finish().map_err(PluginError::from);
        self.terminal = true;
        result
    }

    pub fn invoke_provider(
        mut self,
        class_type: &str,
        request: &[u8],
    ) -> Result<ProviderInvocationResult, PluginError> {
        self.check_active()?;
        let provider_binding = self
            .provider_binding
            .as_ref()
            .ok_or(PluginError::ProviderInvocationUnavailable)?;
        let invocation = self
            .store
            .data()
            .invocation
            .as_ref()
            .ok_or_else(|| PluginError::Invocation(InvocationError::RevokedHandle))?;
        if invocation.node.id != class_type
            || !provider_binding
                .bindings
                .iter()
                .any(|binding| binding.node_id == class_type)
        {
            self.abort();
            return Err(PluginError::UndeclaredNode(class_type.to_owned()));
        }
        if request.is_empty()
            || request.len()
                > usize::try_from(invocation.limits.maximum_value_bytes).unwrap_or(usize::MAX)
        {
            self.abort();
            return Err(PluginError::Invocation(value_quota_error()));
        }
        invocation.check_cancellation()?;
        let WasmBindings::ProviderV1(bindings) = &self.bindings else {
            self.abort();
            return Err(PluginError::ProviderInvocationUnavailable);
        };
        let response = match bindings
            .zed_sim_plugin_provider_binding()
            .call_invoke_provider(&mut self.store, class_type, request)
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                self.abort();
                return Err(PluginError::Invocation(sdk_error(error)));
            }
            Err(error) => return Err(self.wasm_call_error(error)),
        };
        let invocation = self
            .store
            .data_mut()
            .invocation
            .take()
            .ok_or_else(|| PluginError::Invocation(InvocationError::RevokedHandle))?;
        let result = invocation
            .finish_provider_response(response)
            .map_err(PluginError::from);
        self.terminal = true;
        result
    }

    pub fn abort(&mut self) {
        if let Some(invocation) = self.store.data_mut().invocation.as_mut() {
            invocation.abort();
        }
        if let Some(provider_runtime) = self.store.data_mut().provider_runtime.as_mut() {
            if let Err(error) = provider_runtime.revoke() {
                eprintln!("provider-v2 runtime revocation failed: {error}");
            }
        }
        self.terminal = true;
    }

    fn check_active(&self) -> Result<(), PluginError> {
        if self.terminal {
            Err(PluginError::Invocation(InvocationError::RevokedHandle))
        } else {
            Ok(())
        }
    }

    fn wasm_call_error(&mut self, error: impl std::fmt::Display) -> PluginError {
        let cancelled = self
            .store
            .data()
            .invocation
            .as_ref()
            .is_some_and(InvocationHost::is_cancelled);
        self.abort();
        if cancelled {
            PluginError::Invocation(InvocationError::Cancelled)
        } else {
            PluginError::WasmTrap(sanitize_diagnostic(&format!("{error:#}")))
        }
    }
}

impl Drop for WasmPluginInstance {
    fn drop(&mut self) {
        if !self.terminal {
            self.abort();
        }
    }
}

pub struct PluginHost {
    runtime: ComponentRuntime,
    registry: TypeRegistry,
    api_features: BTreeSet<String>,
    limits: ComponentLimits,
}

impl PluginHost {
    pub fn new() -> Result<Self, PluginError> {
        Self::with_configuration(
            ComponentLimits::default(),
            DEFAULT_API_FEATURES
                .iter()
                .map(|feature| (*feature).to_owned()),
        )
    }

    pub fn with_configuration(
        limits: ComponentLimits,
        api_features: impl IntoIterator<Item = String>,
    ) -> Result<Self, PluginError> {
        let runtime = ComponentRuntime::no_wasi()
            .map_err(|error| PluginError::ComponentCompilation(error.to_string()))?;
        Self::with_component_runtime(runtime, limits, api_features)
    }

    pub fn with_component_runtime(
        runtime: ComponentRuntime,
        limits: ComponentLimits,
        api_features: impl IntoIterator<Item = String>,
    ) -> Result<Self, PluginError> {
        limits.validate()?;
        Ok(Self {
            runtime,
            registry: TypeRegistry::built_in().map_err(PluginContractError::from)?,
            api_features: api_features.into_iter().collect(),
            limits,
        })
    }

    pub fn registry(&self) -> &TypeRegistry {
        &self.registry
    }

    pub fn limits(&self) -> &ComponentLimits {
        &self.limits
    }

    pub fn validate<'a>(
        &self,
        manifest: &'a PluginManifest,
        authorization: &'a PluginAuthorization,
    ) -> Result<VerifiedManifest<'a>, PluginError> {
        manifest.validate(&self.registry)?;
        for request in &manifest.capabilities {
            self.limits
                .capability_limits
                .validate_quota(request.quota)
                .map_err(PluginError::Invocation)?;
        }
        authorization.require_manifest(manifest)?;
        let negotiated_api = manifest
            .api
            .negotiate(COMPONENT_API_VERSION, &self.api_features)?;
        Ok(VerifiedManifest {
            manifest,
            authorization,
            negotiated_api,
        })
    }

    pub fn compile_component(
        &self,
        bytes: &[u8],
        manifest: &PluginManifest,
        authorization: &PluginAuthorization,
    ) -> Result<CompiledPlugin, PluginError> {
        self.validate(manifest, authorization)?;
        if bytes.len() > self.limits.maximum_component_bytes {
            return Err(PluginError::ComponentTooLarge);
        }
        let digest = encode_hex(&Sha256::digest(bytes));
        if !constant_time_equal(digest.as_bytes(), manifest.digest_sha256.as_bytes()) {
            return Err(PluginError::ManifestProjectionMismatch);
        }
        let component = self.runtime.compile_component(bytes).map_err(|error| {
            PluginError::ComponentCompilation(sanitize_diagnostic(&error.to_string()))
        })?;
        let manifest_projection = manifest.component_projection();
        let world = if manifest.provider_binding.is_some() {
            if manifest_projection.component_world != PROVIDER_COMPONENT_WORLD {
                return Err(PluginError::ComponentWorldMismatch);
            }
            ComponentWorld::ProviderV1
        } else {
            ComponentWorld::Legacy
        };
        let compiled = CompiledPlugin {
            component,
            identifier: manifest.identifier.clone(),
            digest_sha256: digest,
            manifest_projection,
            provider_binding: manifest.provider_binding.clone(),
            provider_manifest_v2: None,
            world,
        };
        self.preflight_component(&compiled)?;
        Ok(compiled)
    }

    pub fn compile_provider_component_v2(
        &self,
        bytes: &[u8],
        manifest: &ProviderPluginManifestV2,
        authorization: &ProviderManifestAuthorizationV2,
    ) -> Result<CompiledPlugin, PluginError> {
        manifest.validate(&self.registry)?;
        self.validate(&manifest.manifest, authorization.authorization())?;
        if bytes.len() > self.limits.maximum_component_bytes {
            return Err(PluginError::ComponentTooLarge);
        }
        let digest = encode_hex(&Sha256::digest(bytes));
        let provider_binding = manifest
            .manifest
            .provider_binding
            .as_ref()
            .ok_or(PluginError::ProviderBindingMismatch)?;
        let outer_signing_payload_sha256 = Sha256::digest(manifest.signing_payload()?);
        if !constant_time_equal(
            digest.as_bytes(),
            manifest.manifest.digest_sha256.as_bytes(),
        ) || authorization.provider_binding() != provider_binding
            || authorization.streaming_contract() != &manifest.streaming
            || authorization.outer_signing_payload_sha256()
                != outer_signing_payload_sha256.as_slice()
        {
            return Err(PluginError::ManifestProjectionMismatch);
        }
        let component = self.runtime.compile_component(bytes).map_err(|error| {
            PluginError::ComponentCompilation(sanitize_diagnostic(&error.to_string()))
        })?;
        let compiled = CompiledPlugin {
            component,
            identifier: manifest.manifest.identifier.clone(),
            digest_sha256: digest,
            manifest_projection: manifest.manifest.component_projection(),
            provider_binding: manifest.manifest.provider_binding.clone(),
            provider_manifest_v2: Some(manifest.clone()),
            world: ComponentWorld::ProviderV2,
        };
        self.preflight_component(&compiled)?;
        Ok(compiled)
    }

    pub fn compile_provider_component_v2_for_worker(
        &self,
        bytes: &[u8],
        manifest: &ProviderPluginManifestV2,
        authorization: &PluginAuthorization,
    ) -> Result<CompiledPlugin, PluginError> {
        manifest.validate(&self.registry)?;
        self.validate(&manifest.manifest, authorization)?;
        if bytes.len() > self.limits.maximum_component_bytes {
            return Err(PluginError::ComponentTooLarge);
        }
        let digest = encode_hex(&Sha256::digest(bytes));
        if !constant_time_equal(
            digest.as_bytes(),
            manifest.manifest.digest_sha256.as_bytes(),
        ) {
            return Err(PluginError::ManifestProjectionMismatch);
        }
        let component = self.runtime.compile_component(bytes).map_err(|error| {
            PluginError::ComponentCompilation(sanitize_diagnostic(&error.to_string()))
        })?;
        let compiled = CompiledPlugin {
            component,
            identifier: manifest.manifest.identifier.clone(),
            digest_sha256: digest,
            manifest_projection: manifest.manifest.component_projection(),
            provider_binding: manifest.manifest.provider_binding.clone(),
            provider_manifest_v2: Some(manifest.clone()),
            world: ComponentWorld::ProviderV2,
        };
        self.preflight_component(&compiled)?;
        Ok(compiled)
    }

    pub fn invoke_provider_component_v2_for_worker(
        &self,
        plugin: &CompiledPlugin,
        invocation: InvocationHost,
        node_id: &str,
        envelope: WorkerProviderV2InvocationEnvelope,
        cancellation: CancellationToken,
        sender: async_channel::Sender<ProviderV2WorkerStreamCall>,
    ) -> Result<ProviderV2WorkerPendingInvocation, PluginError> {
        let registry = invocation.registry.clone();
        let expected_manifest = plugin
            .provider_manifest_v2
            .as_ref()
            .ok_or(PluginError::ProviderInvocationUnavailable)?;
        let expected_manifest_sha256 =
            encode_hex(&Sha256::digest(expected_manifest.signing_payload()?));
        if envelope.provider_manifest_sha256().as_str() != expected_manifest_sha256
            || envelope.streaming_contract()
                != &worker_streaming_contract(&expected_manifest.streaming)
        {
            return Err(PluginError::ProviderRuntimeActivationDenied);
        }
        let runtime = ProviderV2RuntimeHost::checked_for_worker_bridge(
            envelope.context().clone(),
            envelope.streaming_contract().clone(),
            cancellation.clone(),
            sender,
        )
        .map_err(PluginError::ProviderStreaming)?;
        let instance = self.instantiate_provider_component_v2(plugin, invocation, runtime)?;
        let proposal = instance.invoke_provider_v2(node_id)?;
        Ok(ProviderV2WorkerPendingInvocation {
            proposal,
            envelope,
            registry,
            cancellation,
        })
    }

    fn new_wasm_store(&self) -> Result<Store<WasmStoreState>, PluginError> {
        self.make_wasm_store(None, None)
    }

    fn new_wasm_invocation_store(
        &self,
        invocation: InvocationHost,
    ) -> Result<Store<WasmStoreState>, PluginError> {
        self.make_wasm_store(Some(invocation), None)
    }

    fn new_wasm_provider_v2_store(
        &self,
        invocation: InvocationHost,
        provider_runtime: ProviderV2RuntimeHost,
    ) -> Result<Store<WasmStoreState>, PluginError> {
        self.make_wasm_store(Some(invocation), Some(provider_runtime))
    }

    pub fn instantiate_component(
        &self,
        plugin: &CompiledPlugin,
        invocation: InvocationHost,
    ) -> Result<WasmPluginInstance, PluginError> {
        if plugin.world == ComponentWorld::ProviderV2 {
            return Err(PluginError::ProviderInvocationUnavailable);
        }
        if invocation.plugin_identifier != plugin.identifier
            || invocation.plugin_digest_sha256 != plugin.digest_sha256
            || !plugin.manifest_projection.nodes.iter().any(|node| {
                node.id == invocation.node.id && node.version == invocation.node.version
            })
        {
            return Err(PluginError::InvocationBindingMismatch);
        }
        let store = self.new_wasm_invocation_store(invocation)?;
        let (store, bindings) = self.instantiate_bindings(plugin, store)?;
        let mut instance = WasmPluginInstance {
            store,
            bindings,
            expected_manifest_projection: plugin.manifest_projection.clone(),
            provider_binding: plugin.provider_binding.clone(),
            expected_provider_manifest_v2: plugin.provider_manifest_v2.clone(),
            terminal: false,
        };
        instance.manifest_bytes()?;
        if plugin.world == ComponentWorld::ProviderV1 {
            instance.provider_binding_set()?;
        }
        Ok(instance)
    }

    pub(crate) fn instantiate_provider_component_v2(
        &self,
        plugin: &CompiledPlugin,
        invocation: InvocationHost,
        provider_runtime: ProviderV2RuntimeHost,
    ) -> Result<WasmPluginInstance, PluginError> {
        if plugin.world != ComponentWorld::ProviderV2
            || invocation.plugin_identifier != plugin.identifier
            || invocation.plugin_digest_sha256 != plugin.digest_sha256
            || !plugin.manifest_projection.nodes.iter().any(|node| {
                node.id == invocation.node.id && node.version == invocation.node.version
            })
        {
            return Err(PluginError::InvocationBindingMismatch);
        }
        let store = self.new_wasm_provider_v2_store(invocation, provider_runtime)?;
        let (store, bindings) = self.instantiate_bindings(plugin, store)?;
        let mut instance = WasmPluginInstance {
            store,
            bindings,
            expected_manifest_projection: plugin.manifest_projection.clone(),
            provider_binding: plugin.provider_binding.clone(),
            expected_provider_manifest_v2: plugin.provider_manifest_v2.clone(),
            terminal: false,
        };
        instance.provider_v2_manifest_projection()?;
        Ok(instance)
    }

    fn preflight_component(&self, plugin: &CompiledPlugin) -> Result<(), PluginError> {
        let store = self.new_wasm_store()?;
        let phase_started = Instant::now();
        let (mut store, bindings) = self.instantiate_bindings(plugin, store).map_err(|error| {
            preflight_component_error(preflight_timed_phase_error(
                ComponentPreflightPhase::Instantiation,
                error,
                phase_started.elapsed(),
            ))
        })?;
        // Host-side instantiation must not exhaust the metadata call phase's
        // epoch budget. Instantiation retains its deadline and fuel stays shared.
        store.set_epoch_deadline(self.limits.epoch_deadline_ticks);
        let mut instance = WasmPluginInstance {
            store,
            bindings,
            expected_manifest_projection: plugin.manifest_projection.clone(),
            provider_binding: plugin.provider_binding.clone(),
            expected_provider_manifest_v2: plugin.provider_manifest_v2.clone(),
            terminal: false,
        };
        let phase_started = Instant::now();
        instance.manifest_bytes().map_err(|error| {
            preflight_timed_phase_error(
                ComponentPreflightPhase::Manifest,
                error,
                phase_started.elapsed(),
            )
        })?;
        if plugin.world == ComponentWorld::ProviderV1 {
            let phase_started = Instant::now();
            instance.provider_binding_set().map_err(|error| {
                preflight_timed_phase_error(
                    ComponentPreflightPhase::ProviderBinding,
                    error,
                    phase_started.elapsed(),
                )
            })?;
        }
        instance.terminal = true;
        Ok(())
    }

    fn instantiate_bindings(
        &self,
        plugin: &CompiledPlugin,
        mut store: Store<WasmStoreState>,
    ) -> Result<(Store<WasmStoreState>, WasmBindings), PluginError> {
        let mut linker = Linker::<WasmStoreState>::new(self.runtime.engine());
        let bindings = match plugin.world {
            ComponentWorld::Legacy => {
                wit_contract::SimPlugin::add_to_linker::<
                    WasmStoreState,
                    wasmtime::component::HasSelf<WasmStoreState>,
                >(&mut linker, |state| state)
                .map_err(component_compilation_error)?;
                WasmBindings::Legacy(
                    wit_contract::SimPlugin::instantiate(&mut store, plugin.component(), &linker)
                        .map_err(component_instantiation_error)?,
                )
            }
            ComponentWorld::ProviderV1 => {
                provider_wit_contract::SimProviderPlugin::add_to_linker::<
                    WasmStoreState,
                    wasmtime::component::HasSelf<WasmStoreState>,
                >(&mut linker, |state| state)
                .map_err(component_compilation_error)?;
                WasmBindings::ProviderV1(
                    provider_wit_contract::SimProviderPlugin::instantiate(
                        &mut store,
                        plugin.component(),
                        &linker,
                    )
                    .map_err(component_instantiation_error)?,
                )
            }
            ComponentWorld::ProviderV2 => {
                provider_v2_wit_contract::SimProviderPlugin::add_to_linker::<
                    WasmStoreState,
                    wasmtime::component::HasSelf<WasmStoreState>,
                >(&mut linker, |state| state)
                .map_err(component_compilation_error)?;
                WasmBindings::ProviderV2(
                    provider_v2_wit_contract::SimProviderPlugin::instantiate(
                        &mut store,
                        plugin.component(),
                        &linker,
                    )
                    .map_err(component_instantiation_error)?,
                )
            }
        };
        Ok((store, bindings))
    }

    fn make_wasm_store(
        &self,
        invocation: Option<InvocationHost>,
        provider_runtime: Option<ProviderV2RuntimeHost>,
    ) -> Result<Store<WasmStoreState>, PluginError> {
        let limits = StoreLimitsBuilder::new()
            .memory_size(self.limits.maximum_memory_bytes)
            .table_elements(self.limits.maximum_table_elements)
            .instances(self.limits.maximum_instances)
            .tables(self.limits.maximum_tables)
            .memories(self.limits.maximum_memories)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(
            self.runtime.engine(),
            WasmStoreState {
                limits,
                invocation,
                provider_runtime,
            },
        );
        store.limiter(|state| &mut state.limits);
        store
            .set_fuel(self.limits.maximum_fuel)
            .map_err(|error| PluginError::ComponentCompilation(error.to_string()))?;
        store.set_epoch_deadline(self.limits.epoch_deadline_ticks);
        Ok(store)
    }

    pub fn interrupt_wasm(&self) {
        self.runtime.increment_epoch();
    }

    pub fn begin_invocation(
        &self,
        manifest: &PluginManifest,
        authorization: &PluginAuthorization,
        node_id: &str,
        inputs: InvocationInputs,
        services: Arc<dyn PluginCapabilityServices>,
        cancellation: CancellationToken,
    ) -> Result<InvocationHost, PluginError> {
        self.validate(manifest, authorization)?;
        let node = manifest
            .nodes
            .iter()
            .find(|node| node.id == node_id)
            .ok_or_else(|| PluginError::UndeclaredNode(node_id.to_owned()))?;
        let ui_contributions = manifest
            .ui
            .iter()
            .map(|contribution| contribution.id.clone())
            .collect();
        let route_response_limits = manifest
            .routes
            .iter()
            .map(|route| (route.id.clone(), route.maximum_response_bytes))
            .collect();
        InvocationHost::new(
            node,
            inputs,
            authorization,
            manifest,
            services,
            cancellation,
            self.registry.clone(),
            self.limits.clone(),
            ui_contributions,
            route_response_limits,
        )
        .map_err(PluginError::from)
    }

    pub fn invoke_rust(
        &self,
        plugin: &dyn RustSimPlugin,
        authorization: &PluginAuthorization,
        node_id: &str,
        inputs: InvocationInputs,
        services: Arc<dyn PluginCapabilityServices>,
        cancellation: CancellationToken,
    ) -> Result<InvocationResult, PluginError> {
        let manifest = plugin.manifest();
        let mut invocation = self.begin_invocation(
            manifest,
            authorization,
            node_id,
            inputs,
            services,
            cancellation,
        )?;
        let mut node = plugin.create_node(node_id)?;
        if let Err(error) = node.invoke(&mut invocation) {
            match &error {
                InvocationError::Cancelled => node.cancel(CancelReason::User),
                InvocationError::TimedOut => node.cancel(CancelReason::Timeout),
                _ => {}
            }
            invocation.abort();
            return Err(PluginError::Invocation(error));
        }
        invocation.finish().map_err(PluginError::from)
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct InvocationInputs {
    values: BTreeMap<String, Option<Vec<PluginValue>>>,
    #[serde(default)]
    media: BTreeMap<String, BTreeMap<u32, InvocationMediaContent>>,
    #[serde(default)]
    media_batches: BTreeMap<String, BTreeMap<u32, BTreeMap<u32, InvocationMediaContent>>>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct InvocationMediaContent {
    media_type: String,
    source_value_sha256: String,
    content_sha256: String,
    bytes: Vec<u8>,
}

impl InvocationMediaContent {
    pub fn new(
        media_type: impl Into<String>,
        source_value_sha256: impl Into<String>,
        content_sha256: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            media_type: media_type.into(),
            source_value_sha256: source_value_sha256.into(),
            content_sha256: content_sha256.into(),
            bytes,
        }
    }
}

impl InvocationInputs {
    pub fn set_absent(&mut self, port_id: impl Into<String>) {
        self.values.insert(port_id.into(), None);
    }

    pub fn set_present(&mut self, port_id: impl Into<String>, values: Vec<PluginValue>) {
        self.values.insert(port_id.into(), Some(values));
    }

    pub fn register_media(
        &mut self,
        port_id: impl Into<String>,
        index: u32,
        content: InvocationMediaContent,
    ) {
        self.media
            .entry(port_id.into())
            .or_default()
            .insert(index, content);
    }

    pub fn register_media_batch_item(
        &mut self,
        port_id: impl Into<String>,
        value_index: u32,
        batch_index: u32,
        content: InvocationMediaContent,
    ) -> Result<(), InvocationError> {
        if batch_index >= 128 {
            return Err(InvocationError::InvalidCardinality(
                "media input batch".to_owned(),
            ));
        }
        let batch = self
            .media_batches
            .entry(port_id.into())
            .or_default()
            .entry(value_index)
            .or_default();
        if batch.contains_key(&batch_index) {
            return Err(InvocationError::InvalidCardinality(
                "media input batch".to_owned(),
            ));
        }
        batch.insert(batch_index, content);
        Ok(())
    }

    pub(crate) fn values(&self) -> &BTreeMap<String, Option<Vec<PluginValue>>> {
        &self.values
    }
}

struct InputPortState {
    present: bool,
    values: Vec<Option<PluginValue>>,
    media: Vec<Vec<InvocationMediaContent>>,
    first_image_only: bool,
}

struct OutputPortState {
    values: Vec<PluginValue>,
    value_bytes: u64,
    present: Option<bool>,
    finished: bool,
}

struct HandleEntry {
    generation: u32,
    value: PluginValue,
    media: Vec<InvocationMediaContent>,
    first_image_only: bool,
}

#[derive(Clone, Copy, Default)]
struct PortCallUsage {
    operations: u64,
    response_bytes: u64,
}

pub struct InvocationHost {
    invocation_id: u64,
    plugin_identifier: String,
    plugin_digest_sha256: String,
    node: PluginNode,
    registry: TypeRegistry,
    limits: ComponentLimits,
    inputs: BTreeMap<String, InputPortState>,
    outputs: BTreeMap<String, OutputPortState>,
    handles: BTreeMap<u32, HandleEntry>,
    next_handle: u32,
    invocation_value_bytes: u64,
    port_call_usage: Cell<PortCallUsage>,
    capabilities: Option<CapabilityState>,
    terminal: bool,
}

impl InvocationHost {
    fn new(
        node: &PluginNode,
        supplied_inputs: InvocationInputs,
        authorization: &PluginAuthorization,
        manifest: &PluginManifest,
        services: Arc<dyn PluginCapabilityServices>,
        cancellation: CancellationToken,
        registry: TypeRegistry,
        limits: ComponentLimits,
        ui_contributions: BTreeSet<String>,
        route_response_limits: BTreeMap<String, u64>,
    ) -> Result<Self, InvocationError> {
        let invocation_id = next_invocation_id()?;
        let InvocationInputs {
            values: mut supplied,
            media: mut supplied_media,
            media_batches: mut supplied_media_batches,
        } = supplied_inputs;
        let mut inputs = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        let mut total_value_bytes = 0_u64;
        for port in &node.ports {
            match port.direction {
                PortDirection::Input => {
                    let supplied_values = supplied.remove(&port.id).unwrap_or(None);
                    let present = supplied_values.is_some();
                    let values = supplied_values.unwrap_or_default();
                    let port_value_bytes =
                        validate_input_port(port, present, &values, &registry, &limits)?;
                    total_value_bytes = total_value_bytes
                        .checked_add(port_value_bytes)
                        .ok_or_else(invocation_value_quota_error)?;
                    if total_value_bytes > limits.maximum_invocation_value_bytes {
                        return Err(invocation_value_quota_error());
                    }
                    let mut port_media = supplied_media.remove(&port.id).unwrap_or_default();
                    let mut port_media_batches =
                        supplied_media_batches.remove(&port.id).unwrap_or_default();
                    let first_image_only = manifest.identifier == "zed.sim.provider.beeble"
                        && manifest.provider_binding.as_ref().is_some_and(|binding| {
                            binding.implementation_namespace == manifest.identifier
                                && binding
                                    .bindings
                                    .iter()
                                    .any(|claim| claim.node_id == node.id)
                        })
                        && registry_adapter::beeble_uses_first_image(&node.id, &port.id)
                        || manifest.identifier == "zed.sim.provider.hitpaw"
                            && manifest.provider_binding.as_ref().is_some_and(|binding| {
                                binding.implementation_namespace == manifest.identifier
                                    && binding.bindings.iter().any(|claim| {
                                        claim.node_id == node.id
                                            && claim.feature_id == "COMFY-NODE-0226"
                                    })
                            })
                            && registry_adapter::hitpaw_uses_first_image(&node.id, &port.id)
                        || manifest.identifier == "zed.sim.provider.bfl"
                            && manifest.provider_binding.as_ref().is_some_and(|binding| {
                                binding.implementation_namespace == manifest.identifier
                                    && binding.bindings.iter().any(|claim| {
                                        claim.node_id == node.id
                                            && claim.feature_id == "COMFY-NODE-0188"
                                    })
                            })
                            && registry_adapter::bfl_ultra_uses_first_image(&node.id, &port.id);
                    let mut media = Vec::with_capacity(values.len());
                    for (index, value) in values.iter().enumerate() {
                        let index = u32::try_from(index)
                            .map_err(|_| InvocationError::InvalidCardinality(port.id.clone()))?;
                        let legacy_content = port_media.remove(&index);
                        let batch_content = port_media_batches.remove(&index).unwrap_or_default();
                        if legacy_content.is_some() && !batch_content.is_empty() {
                            return Err(invalid_value("duplicate media input registration"));
                        }
                        let content =
                            if let Some(content) = legacy_content {
                                vec![content]
                            } else if batch_content.is_empty() {
                                Vec::new()
                            } else {
                                let expected = media_batch_count(value, first_image_only)?;
                                if batch_content.len() != expected
                                    || batch_content.keys().copied().ne(0..u32::try_from(expected)
                                        .map_err(|_| value_quota_error())?)
                                {
                                    return Err(invalid_value("media batch registration"));
                                }
                                batch_content.into_values().collect()
                            };
                        if !content.is_empty()
                            && content.len() != media_batch_count(value, first_image_only)?
                        {
                            return Err(invalid_value("media batch cardinality"));
                        }
                        for content in &content {
                            let media_bytes = validate_media_content(value, content, &limits)?;
                            total_value_bytes = total_value_bytes
                                .checked_add(media_bytes)
                                .ok_or_else(invocation_value_quota_error)?;
                            if total_value_bytes > limits.maximum_invocation_value_bytes {
                                return Err(invocation_value_quota_error());
                            }
                        }
                        media.push(content);
                    }
                    if !port_media.is_empty() || !port_media_batches.is_empty() {
                        let index = port_media
                            .keys()
                            .next()
                            .copied()
                            .or_else(|| port_media_batches.keys().next().copied())
                            .ok_or_else(|| {
                                InvocationError::HostFailure(
                                    "media input index disappeared".to_owned(),
                                )
                            })?;
                        return Err(InvocationError::IndexOutOfBounds {
                            port: port.id.clone(),
                            index,
                        });
                    }
                    inputs.insert(
                        port.id.clone(),
                        InputPortState {
                            present,
                            values: values.into_iter().map(Some).collect(),
                            media,
                            first_image_only,
                        },
                    );
                }
                PortDirection::Output => {
                    outputs.insert(
                        port.id.clone(),
                        OutputPortState {
                            values: Vec::new(),
                            value_bytes: 0,
                            present: None,
                            finished: false,
                        },
                    );
                }
            }
        }
        if let Some((unknown, _)) = supplied.into_iter().next() {
            return Err(InvocationError::UnknownPort(unknown));
        }
        if let Some((unknown, _)) = supplied_media.into_iter().next() {
            return Err(InvocationError::UnknownPort(unknown));
        }
        if let Some((unknown, _)) = supplied_media_batches.into_iter().next() {
            return Err(InvocationError::UnknownPort(unknown));
        }
        let capabilities = CapabilityState::with_declarations(
            authorization,
            manifest,
            services,
            cancellation,
            limits.capability_limits,
            ui_contributions,
            route_response_limits,
        )?;
        Ok(Self {
            invocation_id,
            plugin_identifier: manifest.identifier.clone(),
            plugin_digest_sha256: manifest.digest_sha256.clone(),
            node: node.clone(),
            registry,
            limits,
            inputs,
            outputs,
            handles: BTreeMap::new(),
            next_handle: 1,
            invocation_value_bytes: total_value_bytes,
            port_call_usage: Cell::new(PortCallUsage::default()),
            capabilities: Some(capabilities),
            terminal: false,
        })
    }

    pub fn abort(&mut self) {
        if let Some(capabilities) = self.capabilities.as_mut() {
            capabilities.rollback();
        }
        self.inputs.clear();
        self.outputs.clear();
        self.handles.clear();
        self.terminal = true;
    }

    pub fn finish(mut self) -> Result<InvocationResult, InvocationError> {
        self.check_active()?;
        self.check_cancellation()?;
        let unfinished_output = self
            .node
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .find_map(|port| {
                self.outputs
                    .get(&port.id)
                    .filter(|state| !state.finished)
                    .map(|_| port.id.clone())
            });
        if let Some(port_id) = unfinished_output {
            self.abort();
            return Err(InvocationError::UnfinishedOutput(port_id));
        }
        let mut capabilities = self.capabilities.take().ok_or_else(|| {
            InvocationError::HostFailure("invocation capability state is missing".to_owned())
        })?;
        if capabilities.has_open_output_buffers() {
            capabilities.rollback();
            self.abort();
            return Err(InvocationError::HostFailure(
                "plugin invocation left an output transaction open".to_owned(),
            ));
        }
        let effects = capabilities.finish()?;
        let mut outputs = BTreeMap::new();
        let mut output_presence = BTreeMap::new();
        for (port, state) in mem::take(&mut self.outputs) {
            let present = state.present.ok_or_else(|| {
                InvocationError::HostFailure(format!(
                    "finished plugin output `{port}` has no presence state"
                ))
            })?;
            output_presence.insert(port.clone(), present);
            outputs.insert(port, state.values);
        }
        self.inputs.clear();
        self.handles.clear();
        self.terminal = true;
        Ok(InvocationResult {
            outputs,
            output_presence,
            effects,
        })
    }

    fn finish_provider_response(
        mut self,
        response: wit_contract::zed::sim_plugin::types::ProviderInvocationResponse,
    ) -> Result<ProviderInvocationResult, InvocationError> {
        self.check_active()?;
        self.check_cancellation()?;
        if response.receipt.is_empty()
            || u64::try_from(response.receipt.len()).unwrap_or(u64::MAX)
                > self.limits.maximum_port_response_bytes
        {
            return Err(InvocationError::InvocationQuotaExceeded {
                limit: "provider-receipt-byte".to_owned(),
            });
        }
        let receipts = ProviderResultReceiptSet::from_bytes(&response.receipt)
            .map_err(|_| {
                InvocationError::HostFailure(
                    "provider component returned an invalid result receipt set".to_owned(),
                )
            })?
            .into_receipts();
        if !self.handles.is_empty()
            || self
                .outputs
                .values()
                .any(|state| !state.values.is_empty() || state.present.is_some() || state.finished)
        {
            return Err(InvocationError::HostFailure(
                "provider component mixed handle-port output with its typed response".to_owned(),
            ));
        }
        if !response.outputs.is_empty() {
            return Err(InvocationError::HostFailure(
                "provider component attempted to author materialized output metadata".to_owned(),
            ));
        }
        self.check_cancellation()?;
        let mut capabilities = self.capabilities.take().ok_or_else(|| {
            InvocationError::HostFailure("invocation capability state is missing".to_owned())
        })?;
        if capabilities.has_open_output_buffers() {
            capabilities.rollback();
            return Err(InvocationError::HostFailure(
                "provider invocation left an output transaction open".to_owned(),
            ));
        }
        let effects = capabilities.finish()?;
        self.outputs.clear();
        self.inputs.clear();
        self.terminal = true;
        Ok(ProviderInvocationResult {
            outputs: BTreeMap::new(),
            output_presence: BTreeMap::new(),
            effects,
            receipts,
            resolved_provider_results: Vec::new(),
            materialized_provider_response: None,
        })
    }

    fn finish_provider_v2_inputs(mut self) -> Result<(), InvocationError> {
        self.check_active()?;
        self.check_cancellation()?;
        if self
            .outputs
            .values()
            .any(|state| !state.values.is_empty() || state.present.is_some() || state.finished)
        {
            self.abort();
            return Err(InvocationError::HostFailure(
                "provider-v2 input host retained output authority".to_owned(),
            ));
        }
        let mut capabilities = self.capabilities.take().ok_or_else(|| {
            InvocationError::HostFailure("invocation capability state is missing".to_owned())
        })?;
        if capabilities.has_open_output_buffers() {
            capabilities.rollback();
            self.abort();
            return Err(InvocationError::HostFailure(
                "provider-v2 input host retained an output transaction".to_owned(),
            ));
        }
        let effects = capabilities.finish()?;
        if !effects.outputs.is_empty()
            || !effects.logs.is_empty()
            || !effects.ui_state.is_empty()
            || !effects.routes.is_empty()
        {
            self.abort();
            return Err(InvocationError::HostFailure(
                "provider-v2 input host produced a legacy capability effect".to_owned(),
            ));
        }
        self.inputs.clear();
        self.handles.clear();
        self.outputs.clear();
        self.terminal = true;
        Ok(())
    }

    fn check_active(&self) -> Result<(), InvocationError> {
        if self.terminal {
            Err(InvocationError::RevokedHandle)
        } else {
            Ok(())
        }
    }

    fn check_cancellation(&self) -> Result<(), InvocationError> {
        self.capabilities
            .as_ref()
            .ok_or_else(|| InvocationError::HostFailure("capability state is missing".to_owned()))?
            .check_cancelled()
    }

    fn is_cancelled(&self) -> bool {
        self.capabilities
            .as_ref()
            .is_some_and(CapabilityState::is_cancelled)
    }

    fn begin_port_call(&self) -> Result<(), InvocationError> {
        self.check_active()?;
        self.check_cancellation()?;
        let usage = self.port_call_usage.get();
        let operations = usage.operations.checked_add(1).ok_or_else(|| {
            InvocationError::InvocationQuotaExceeded {
                limit: "port-operation".to_owned(),
            }
        })?;
        if operations > self.limits.maximum_port_operations {
            return Err(InvocationError::InvocationQuotaExceeded {
                limit: "port-operation".to_owned(),
            });
        }
        self.port_call_usage.set(PortCallUsage {
            operations,
            response_bytes: usage.response_bytes,
        });
        Ok(())
    }

    fn charge_port_response(&self, response_bytes: u64) -> Result<(), InvocationError> {
        self.check_active()?;
        self.check_cancellation()?;
        let usage = self.port_call_usage.get();
        let response_bytes = usage
            .response_bytes
            .checked_add(response_bytes)
            .ok_or_else(|| InvocationError::InvocationQuotaExceeded {
                limit: "port-response-byte".to_owned(),
            })?;
        if response_bytes > self.limits.maximum_port_response_bytes {
            return Err(InvocationError::InvocationQuotaExceeded {
                limit: "port-response-byte".to_owned(),
            });
        }
        self.port_call_usage.set(PortCallUsage {
            operations: usage.operations,
            response_bytes,
        });
        Ok(())
    }

    fn available_value_handle(&self) -> Result<(ValueHandle, u32), InvocationError> {
        if self.handles.len() >= self.limits.maximum_value_handles {
            return Err(InvocationError::InvocationQuotaExceeded {
                limit: "invocation-handle".to_owned(),
            });
        }
        let next_handle = self.next_handle.checked_add(1).ok_or_else(|| {
            InvocationError::InvocationQuotaExceeded {
                limit: "invocation-handle".to_owned(),
            }
        })?;
        Ok((
            ValueHandle {
                invocation: self.invocation_id,
                slot: self.next_handle,
                generation: 1,
            },
            next_handle,
        ))
    }

    fn create_output_value_after_begin(
        &mut self,
        value: PluginValue,
    ) -> Result<ValueHandle, InvocationError> {
        let expected_family = self.registry.family(value.type_id()).map_err(|error| {
            InvocationError::HostFailure(format!("plugin type registry failed: {error}"))
        })?;
        if expected_family != value.family() {
            return Err(InvocationError::HostFailure(
                "plugin output value type and representation disagree".to_owned(),
            ));
        }
        let value_bytes = value_size(&value)?;
        if value_bytes > self.limits.maximum_value_bytes {
            return Err(value_quota_error());
        }
        let invocation_value_bytes = self
            .invocation_value_bytes
            .checked_add(value_bytes)
            .ok_or_else(invocation_value_quota_error)?;
        if invocation_value_bytes > self.limits.maximum_invocation_value_bytes {
            return Err(invocation_value_quota_error());
        }
        let (handle, next_handle) = self.available_value_handle()?;
        self.charge_port_response(16)?;
        self.next_handle = next_handle;
        self.invocation_value_bytes = invocation_value_bytes;
        self.handles.insert(
            handle.slot,
            HandleEntry {
                generation: handle.generation,
                value,
                media: Vec::new(),
                first_image_only: false,
            },
        );
        Ok(handle)
    }

    fn read_media_input(
        &self,
        handle: ValueHandle,
    ) -> Result<InvocationMediaContent, InvocationError> {
        self.begin_port_call()?;
        let entry = self.media_handle_entry(handle)?;
        if entry.media.len() != 1 || media_batch_count(&entry.value, entry.first_image_only)? != 1 {
            return Err(InvocationError::InvalidCardinality(
                "media input batch".to_owned(),
            ));
        }
        let content = entry.media.first().ok_or_else(|| {
            InvocationError::HostFailure(
                "media content is not registered for this handle".to_owned(),
            )
        })?;
        let response_bytes = validate_media_content(&entry.value, content, &self.limits)?
            .checked_add(u64::try_from(content.media_type.len()).unwrap_or(u64::MAX))
            .and_then(|size| size.checked_add(128))
            .ok_or_else(value_quota_error)?;
        self.charge_port_response(response_bytes)?;
        Ok(content.clone())
    }

    fn media_input_count(&self, handle: ValueHandle) -> Result<u32, InvocationError> {
        self.begin_port_call()?;
        let entry = self.media_handle_entry(handle)?;
        let expected = media_batch_count(&entry.value, entry.first_image_only)?;
        if entry.media.is_empty() || entry.media.len() != expected {
            return Err(InvocationError::InvalidCardinality(
                "media input batch".to_owned(),
            ));
        }
        self.charge_port_response(4)?;
        u32::try_from(expected).map_err(|_| value_quota_error())
    }

    fn read_media_input_at(
        &self,
        handle: ValueHandle,
        batch_index: u32,
    ) -> Result<InvocationMediaContent, InvocationError> {
        self.begin_port_call()?;
        let entry = self.media_handle_entry(handle)?;
        let expected = media_batch_count(&entry.value, entry.first_image_only)?;
        if entry.media.is_empty() || entry.media.len() != expected {
            return Err(InvocationError::InvalidCardinality(
                "media input batch".to_owned(),
            ));
        }
        let batch_index =
            usize::try_from(batch_index).map_err(|_| InvocationError::IndexOutOfBounds {
                port: "media input batch".to_owned(),
                index: batch_index,
            })?;
        let content =
            entry
                .media
                .get(batch_index)
                .ok_or_else(|| InvocationError::IndexOutOfBounds {
                    port: "media input batch".to_owned(),
                    index: batch_index as u32,
                })?;
        let response_bytes = validate_media_content(&entry.value, content, &self.limits)?
            .checked_add(u64::try_from(content.media_type.len()).unwrap_or(u64::MAX))
            .and_then(|size| size.checked_add(128))
            .ok_or_else(value_quota_error)?;
        self.charge_port_response(response_bytes)?;
        Ok(content.clone())
    }

    fn media_handle_entry(&self, handle: ValueHandle) -> Result<&HandleEntry, InvocationError> {
        let generation = self.handles.get(&handle.slot).map(|entry| entry.generation);
        validate_media_handle_identity(self.invocation_id, handle, generation)?;
        self.handles
            .get(&handle.slot)
            .ok_or(InvocationError::RevokedHandle)
    }

    fn port(
        &self,
        port_id: &str,
        direction: PortDirection,
    ) -> Result<&sim_plugin_sdk::PluginPort, InvocationError> {
        let port = self
            .node
            .ports
            .iter()
            .find(|port| port.id == port_id)
            .ok_or_else(|| InvocationError::UnknownPort(port_id.to_owned()))?;
        if port.direction != direction {
            return Err(InvocationError::WrongDirection(port_id.to_owned()));
        }
        Ok(port)
    }
}

fn validate_media_handle_identity(
    invocation_id: u64,
    handle: ValueHandle,
    retained_generation: Option<u32>,
) -> Result<(), InvocationError> {
    if handle.invocation != invocation_id {
        return Err(InvocationError::InvalidHandle);
    }
    if retained_generation != Some(handle.generation) {
        return Err(InvocationError::RevokedHandle);
    }
    Ok(())
}

impl PluginInvocation for InvocationHost {
    fn input_state(&self, port_id: &str) -> Result<InputState, InvocationError> {
        self.begin_port_call()?;
        let port = self.port(port_id, PortDirection::Input)?;
        let state = self
            .inputs
            .get(port_id)
            .ok_or_else(|| InvocationError::UnknownPort(port_id.to_owned()))?;
        let family = self.registry.family(&port.type_id).map_err(|error| {
            InvocationError::HostFailure(format!("plugin type registry failed: {error}"))
        })?;
        let result = InputState {
            present: state.present,
            length: u32::try_from(state.values.len()).map_err(|_| {
                InvocationError::HostFailure("input length exceeds the ABI limit".to_owned())
            })?,
            type_id: port.type_id.clone(),
            family,
            cardinality: port.cardinality,
            presence: port.presence,
            serialization: port.serialization,
            lazy: port.lazy,
        };
        let type_id_bytes = u64::try_from(port.type_id.to_string().len()).map_err(|_| {
            InvocationError::HostFailure("input state type identifier is too large".to_owned())
        })?;
        self.charge_port_response(type_id_bytes.saturating_add(12))?;
        Ok(result)
    }

    fn read_scalar_input(&self, port_id: &str, index: u32) -> Result<PluginValue, InvocationError> {
        self.begin_port_call()?;
        let port = self.port(port_id, PortDirection::Input)?;
        let family = self.registry.family(&port.type_id).map_err(|error| {
            InvocationError::HostFailure(format!("plugin type registry failed: {error}"))
        })?;
        if family != ValueFamily::Scalar || port.serialization != PortSerialization::Inline {
            return Err(InvocationError::HostFailure(format!(
                "input port `{port_id}` does not use inline scalar ownership"
            )));
        }
        let state = self
            .inputs
            .get(port_id)
            .ok_or_else(|| InvocationError::UnknownPort(port_id.to_owned()))?;
        let index = usize::try_from(index).map_err(|_| InvocationError::IndexOutOfBounds {
            port: port_id.to_owned(),
            index,
        })?;
        let value = state
            .values
            .get(index)
            .ok_or_else(|| InvocationError::IndexOutOfBounds {
                port: port_id.to_owned(),
                index: index as u32,
            })?
            .as_ref()
            .ok_or_else(|| InvocationError::AlreadyTaken {
                port: port_id.to_owned(),
                index: index as u32,
            })?;
        require_exact_port_type(port, value)?;
        self.charge_port_response(encoded_value_response_size(value)?)?;
        Ok(value.clone())
    }

    fn take_input(&mut self, port_id: &str, index: u32) -> Result<ValueHandle, InvocationError> {
        self.begin_port_call()?;
        let port = self.port(port_id, PortDirection::Input)?;
        let family = self.registry.family(&port.type_id).map_err(|error| {
            InvocationError::HostFailure(format!("plugin type registry failed: {error}"))
        })?;
        if family == ValueFamily::Scalar || port.serialization == PortSerialization::Inline {
            return Err(InvocationError::HostFailure(format!(
                "input port `{port_id}` uses inline scalar ownership and cannot be taken"
            )));
        }
        let state = self
            .inputs
            .get(port_id)
            .ok_or_else(|| InvocationError::UnknownPort(port_id.to_owned()))?;
        let index = usize::try_from(index).map_err(|_| InvocationError::IndexOutOfBounds {
            port: port_id.to_owned(),
            index,
        })?;
        let value = state
            .values
            .get(index)
            .ok_or_else(|| InvocationError::IndexOutOfBounds {
                port: port_id.to_owned(),
                index: index as u32,
            })?
            .as_ref()
            .ok_or_else(|| InvocationError::AlreadyTaken {
                port: port_id.to_owned(),
                index: index as u32,
            })?;
        require_exact_port_type(port, value)?;
        let first_image_only = state.first_image_only;
        let (handle, next_handle) = self.available_value_handle()?;
        self.charge_port_response(16)?;
        let value = self
            .inputs
            .get_mut(port_id)
            .and_then(|state| state.values.get_mut(index))
            .and_then(Option::take)
            .ok_or_else(|| {
                InvocationError::HostFailure(
                    "validated plugin input disappeared before handle creation".to_owned(),
                )
            })?;
        let media = self
            .inputs
            .get_mut(port_id)
            .and_then(|state| state.media.get_mut(index))
            .map(mem::take)
            .ok_or_else(|| {
                InvocationError::HostFailure(
                    "validated plugin media disappeared before handle creation".to_owned(),
                )
            })?;
        self.next_handle = next_handle;
        self.handles.insert(
            handle.slot,
            HandleEntry {
                generation: handle.generation,
                value,
                media,
                first_image_only,
            },
        );
        Ok(handle)
    }

    fn read_handle(&self, handle: ValueHandle) -> Result<&PluginValue, InvocationError> {
        self.begin_port_call()?;
        if handle.invocation != self.invocation_id {
            return Err(InvocationError::InvalidHandle);
        }
        let entry = self
            .handles
            .get(&handle.slot)
            .ok_or(InvocationError::RevokedHandle)?;
        if entry.generation != handle.generation {
            return Err(InvocationError::RevokedHandle);
        }
        self.charge_port_response(encoded_value_response_size(&entry.value)?)?;
        Ok(&entry.value)
    }

    fn create_output_value(&mut self, value: PluginValue) -> Result<ValueHandle, InvocationError> {
        self.begin_port_call()?;
        self.create_output_value_after_begin(value)
    }

    fn push_output(&mut self, port_id: &str, handle: ValueHandle) -> Result<(), InvocationError> {
        self.begin_port_call()?;
        let port = self.port(port_id, PortDirection::Output)?.clone();
        if handle.invocation != self.invocation_id {
            return Err(InvocationError::InvalidHandle);
        }
        let entry = self
            .handles
            .get(&handle.slot)
            .ok_or(InvocationError::RevokedHandle)?;
        if entry.generation != handle.generation {
            return Err(InvocationError::RevokedHandle);
        }
        require_exact_port_type(&port, &entry.value)?;
        let value_bytes = value_size(&entry.value)?;
        let state = self
            .outputs
            .get(port_id)
            .ok_or_else(|| InvocationError::UnknownPort(port_id.to_owned()))?;
        if state.finished {
            return Err(InvocationError::OutputAlreadyFinished(port_id.to_owned()));
        }
        if port.cardinality == PortCardinality::Singular && !state.values.is_empty() {
            return Err(InvocationError::InvalidCardinality(port_id.to_owned()));
        }
        if state.values.len() >= self.limits.maximum_values_per_port {
            return Err(InvocationError::InvalidCardinality(port_id.to_owned()));
        }
        let output_value_bytes = state
            .value_bytes
            .checked_add(value_bytes)
            .ok_or_else(value_quota_error)?;
        if output_value_bytes > self.limits.maximum_value_bytes {
            return Err(value_quota_error());
        }
        let entry = self
            .handles
            .remove(&handle.slot)
            .ok_or(InvocationError::RevokedHandle)?;
        let state = self.outputs.get_mut(port_id).ok_or_else(|| {
            InvocationError::HostFailure("validated output disappeared".to_owned())
        })?;
        state.value_bytes = output_value_bytes;
        state.values.push(entry.value);
        Ok(())
    }

    fn finish_output(&mut self, port_id: &str, present: bool) -> Result<(), InvocationError> {
        self.begin_port_call()?;
        let port = self.port(port_id, PortDirection::Output)?.clone();
        let state = self
            .outputs
            .get_mut(port_id)
            .ok_or_else(|| InvocationError::UnknownPort(port_id.to_owned()))?;
        if state.finished {
            return Err(InvocationError::OutputAlreadyFinished(port_id.to_owned()));
        }
        validate_finished_output(&port, present, state.values.len())?;
        let state = self.outputs.get_mut(port_id).ok_or_else(|| {
            InvocationError::HostFailure("validated output disappeared".to_owned())
        })?;
        state.present = Some(present);
        state.finished = true;
        Ok(())
    }

    fn call(&mut self, call: CapabilityCall) -> Result<CapabilityResponse, InvocationError> {
        self.check_active()?;
        self.capabilities
            .as_mut()
            .ok_or_else(|| InvocationError::HostFailure("capability state is missing".to_owned()))?
            .execute(call)
    }

    fn check_cancelled(&self) -> Result<(), InvocationError> {
        self.begin_port_call()
    }
}

impl wit_contract::zed::sim_plugin::types::Host for WasmStoreState {}

impl wit_contract::zed::sim_plugin::host::Host for WasmStoreState {
    fn get_input_state(
        &mut self,
        port_id: String,
    ) -> Result<wit_contract::zed::sim_plugin::host::InputState, WitInvocationError> {
        let state = self
            .invocation_mut()?
            .input_state(&port_id)
            .map_err(wit_error)?;
        Ok(wit_contract::zed::sim_plugin::host::InputState {
            present: state.present,
            length: state.length,
            type_id: state.type_id.to_string(),
            family: wit_value_family(state.family),
            cardinality: wit_port_cardinality(state.cardinality),
            presence: wit_port_presence(state.presence),
            serialization: wit_port_serialization(state.serialization),
            lazy: state.lazy,
        })
    }

    fn read_scalar_input(
        &mut self,
        port_id: String,
        index: u32,
    ) -> Result<wit_contract::zed::sim_plugin::host::EncodedValue, WitInvocationError> {
        let value = self
            .invocation_mut()?
            .read_scalar_input(&port_id, index)
            .map_err(wit_error)?;
        let abi_bytes = value.abi_bytes().map_err(|error| {
            wit_host_failure(&format!("plugin value ABI encoding failed: {error}"))
        })?;
        Ok(wit_contract::zed::sim_plugin::host::EncodedValue {
            type_id: value.type_id().to_string(),
            family: wit_value_family(value.family()),
            abi_bytes,
        })
    }

    fn take_input(
        &mut self,
        port_id: String,
        index: u32,
    ) -> Result<wit_contract::zed::sim_plugin::host::ValueHandle, WitInvocationError> {
        self.invocation_mut()?
            .take_input(&port_id, index)
            .map(wit_value_handle)
            .map_err(wit_error)
    }

    fn read_handle(
        &mut self,
        handle: wit_contract::zed::sim_plugin::host::ValueHandle,
    ) -> Result<wit_contract::zed::sim_plugin::host::EncodedValue, WitInvocationError> {
        let handle = sdk_value_handle(handle);
        let value = self
            .invocation_mut()?
            .read_handle(handle)
            .map_err(wit_error)?;
        let abi_bytes = value.abi_bytes().map_err(|error| {
            wit_host_failure(&format!("plugin value ABI encoding failed: {error}"))
        })?;
        Ok(wit_contract::zed::sim_plugin::host::EncodedValue {
            type_id: value.type_id().to_string(),
            family: wit_value_family(value.family()),
            abi_bytes,
        })
    }

    fn create_output_value(
        &mut self,
        value: wit_contract::zed::sim_plugin::host::EncodedValue,
    ) -> Result<wit_contract::zed::sim_plugin::host::ValueHandle, WitInvocationError> {
        let invocation = self.invocation_mut()?;
        invocation.begin_port_call().map_err(wit_error)?;
        let decoded = PluginValue::from_abi_bytes(&value.abi_bytes, &invocation.registry)
            .map_err(|error| wit_host_failure(&format!("invalid plugin output value: {error}")))?;
        if decoded.type_id().to_string() != value.type_id
            || decoded.family() != sdk_value_family(value.family)
        {
            return Err(wit_host_failure(
                "encoded plugin output metadata disagrees with its canonical ABI value",
            ));
        }
        invocation
            .create_output_value_after_begin(decoded)
            .map(wit_value_handle)
            .map_err(wit_error)
    }

    fn push_output(
        &mut self,
        port_id: String,
        handle: wit_contract::zed::sim_plugin::host::ValueHandle,
    ) -> Result<(), WitInvocationError> {
        self.invocation_mut()?
            .push_output(&port_id, sdk_value_handle(handle))
            .map_err(wit_error)
    }

    fn finish_output(&mut self, port_id: String, present: bool) -> Result<(), WitInvocationError> {
        self.invocation_mut()?
            .finish_output(&port_id, present)
            .map_err(wit_error)
    }

    fn check_cancelled(&mut self) -> Result<(), WitInvocationError> {
        self.invocation_mut()?.check_cancelled().map_err(wit_error)
    }

    fn filesystem_read(
        &mut self,
        root: String,
        relative_path: String,
    ) -> Result<Vec<u8>, WitInvocationError> {
        match self.call(CapabilityCall::FilesystemRead {
            root,
            relative_path,
        })? {
            CapabilityResponse::Bytes(bytes) => Ok(bytes),
            _ => Err(wit_host_failure(
                "filesystem host returned the wrong response type",
            )),
        }
    }

    fn provider_request(
        &mut self,
        provider: String,
        endpoint: String,
        body: Vec<u8>,
        secret_id: Option<String>,
    ) -> Result<Vec<u8>, WitInvocationError> {
        match self.call(CapabilityCall::NetworkProvider {
            provider,
            endpoint,
            body,
            secret_id,
        })? {
            CapabilityResponse::Bytes(bytes) => Ok(bytes),
            _ => Err(wit_host_failure(
                "provider host returned the wrong response type",
            )),
        }
    }

    fn secret_exists(&mut self, identifier: String) -> Result<bool, WitInvocationError> {
        match self.call(CapabilityCall::SecretExists { identifier })? {
            CapabilityResponse::Boolean(exists) => Ok(exists),
            _ => Err(wit_host_failure(
                "secret host returned the wrong response type",
            )),
        }
    }

    fn clock_now(&mut self, clock: String) -> Result<u64, WitInvocationError> {
        match self.call(CapabilityCall::ClockNow { clock })? {
            CapabilityResponse::TimestampMilliseconds(milliseconds) => Ok(milliseconds),
            _ => Err(wit_host_failure(
                "clock host returned the wrong response type",
            )),
        }
    }

    fn random_bytes(&mut self, stream: String, length: u32) -> Result<Vec<u8>, WitInvocationError> {
        match self.call(CapabilityCall::RandomBytes { stream, length })? {
            CapabilityResponse::Bytes(bytes) => Ok(bytes),
            _ => Err(wit_host_failure(
                "random host returned the wrong response type",
            )),
        }
    }

    fn model_open(&mut self, identifier: String) -> Result<u64, WitInvocationError> {
        match self.call(CapabilityCall::ModelOpen { identifier })? {
            CapabilityResponse::Handle(handle) => Ok(handle),
            _ => Err(wit_host_failure(
                "model host returned the wrong response type",
            )),
        }
    }

    fn output_begin(&mut self, namespace: String, name: String) -> Result<u64, WitInvocationError> {
        match self.call(CapabilityCall::OutputBegin { namespace, name })? {
            CapabilityResponse::Handle(handle) => Ok(handle),
            _ => Err(wit_host_failure(
                "output host returned the wrong response type",
            )),
        }
    }

    fn output_write(&mut self, transaction: u64, bytes: Vec<u8>) -> Result<(), WitInvocationError> {
        match self.call(CapabilityCall::OutputWrite { transaction, bytes })? {
            CapabilityResponse::Unit => Ok(()),
            _ => Err(wit_host_failure(
                "output host returned the wrong response type",
            )),
        }
    }

    fn output_commit(&mut self, transaction: u64) -> Result<String, WitInvocationError> {
        match self.call(CapabilityCall::OutputCommit { transaction })? {
            CapabilityResponse::CommittedArtifact(identifier) => Ok(identifier),
            _ => Err(wit_host_failure(
                "output host returned the wrong response type",
            )),
        }
    }

    fn log(&mut self, level: String, message: String) -> Result<(), WitInvocationError> {
        match self.call(CapabilityCall::Log { level, message })? {
            CapabilityResponse::Unit => Ok(()),
            _ => Err(wit_host_failure(
                "log host returned the wrong response type",
            )),
        }
    }

    fn ui_set(&mut self, contribution: String, state: Vec<u8>) -> Result<(), WitInvocationError> {
        match self.call(CapabilityCall::UiSet {
            contribution,
            state,
        })? {
            CapabilityResponse::Unit => Ok(()),
            _ => Err(wit_host_failure("UI host returned the wrong response type")),
        }
    }

    fn route_respond(
        &mut self,
        route: String,
        status: u16,
        body: Vec<u8>,
    ) -> Result<(), WitInvocationError> {
        match self.call(CapabilityCall::RouteRespond {
            route,
            status,
            body,
        })? {
            CapabilityResponse::Unit => Ok(()),
            _ => Err(wit_host_failure(
                "route host returned the wrong response type",
            )),
        }
    }
}

impl provider_v2_wit_contract::zed::sim_provider_plugin::invocation_input_host::Host
    for WasmStoreState
{
    fn get_input_state(
        &mut self,
        port_id: String,
    ) -> Result<wit_contract::zed::sim_plugin::types::InputState, WitInvocationError> {
        let state = self
            .invocation_mut()?
            .input_state(&port_id)
            .map_err(wit_error)?;
        Ok(wit_contract::zed::sim_plugin::types::InputState {
            present: state.present,
            length: state.length,
            type_id: state.type_id.to_string(),
            family: wit_value_family(state.family),
            cardinality: wit_port_cardinality(state.cardinality),
            presence: wit_port_presence(state.presence),
            serialization: wit_port_serialization(state.serialization),
            lazy: state.lazy,
        })
    }

    fn read_scalar_input(
        &mut self,
        port_id: String,
        index: u32,
    ) -> Result<wit_contract::zed::sim_plugin::types::EncodedValue, WitInvocationError> {
        let value = self
            .invocation_mut()?
            .read_scalar_input(&port_id, index)
            .map_err(wit_error)?;
        Ok(wit_contract::zed::sim_plugin::types::EncodedValue {
            type_id: value.type_id().to_string(),
            family: wit_value_family(value.family()),
            abi_bytes: value.abi_bytes().map_err(|error| {
                wit_host_failure(&format!("plugin value ABI encoding failed: {error}"))
            })?,
        })
    }

    fn read_media_input(
        &mut self,
        handle: wit_contract::zed::sim_plugin::types::ValueHandle,
    ) -> Result<
        provider_v2_wit_contract::zed::sim_provider_plugin::types::MediaContent,
        WitInvocationError,
    > {
        let content = self
            .invocation_mut()?
            .read_media_input(sdk_value_handle(handle))
            .map_err(wit_error)?;
        Ok(
            provider_v2_wit_contract::zed::sim_provider_plugin::types::MediaContent {
                media_type: content.media_type,
                source_value_sha256: content.source_value_sha256,
                content_sha256: content.content_sha256,
                bytes: content.bytes,
            },
        )
    }

    fn media_input_count(
        &mut self,
        handle: wit_contract::zed::sim_plugin::types::ValueHandle,
    ) -> Result<u32, WitInvocationError> {
        self.invocation_mut()?
            .media_input_count(sdk_value_handle(handle))
            .map_err(wit_error)
    }

    fn read_media_input_at(
        &mut self,
        handle: wit_contract::zed::sim_plugin::types::ValueHandle,
        batch_index: u32,
    ) -> Result<
        provider_v2_wit_contract::zed::sim_provider_plugin::types::MediaContent,
        WitInvocationError,
    > {
        let content = self
            .invocation_mut()?
            .read_media_input_at(sdk_value_handle(handle), batch_index)
            .map_err(wit_error)?;
        Ok(
            provider_v2_wit_contract::zed::sim_provider_plugin::types::MediaContent {
                media_type: content.media_type,
                source_value_sha256: content.source_value_sha256,
                content_sha256: content.content_sha256,
                bytes: content.bytes,
            },
        )
    }

    fn secret_exists(&mut self, identifier: String) -> Result<bool, WitInvocationError> {
        self.provider_runtime_mut()
            .map_err(|error| wit_host_failure(&error.to_string()))?
            .begin_secret_query()
            .map_err(|error| match error {
                WorkerProviderStreamError::Cancelled => WitInvocationError::Cancelled,
                WorkerProviderStreamError::TimedOut => WitInvocationError::TimedOut,
                error => wit_host_failure(&error.to_string()),
            })?;
        let present = match self.call(CapabilityCall::SecretExists {
            identifier: identifier.clone(),
        })? {
            CapabilityResponse::Boolean(present) => present,
            _ => {
                return Err(wit_host_failure(
                    "secret host returned the wrong response type",
                ));
            }
        };
        self.provider_runtime_mut()
            .map_err(|error| wit_host_failure(&error.to_string()))?
            .record_secret_query_result(identifier, present);
        Ok(present)
    }

    fn take_input(
        &mut self,
        port_id: String,
        index: u32,
    ) -> Result<wit_contract::zed::sim_plugin::types::ValueHandle, WitInvocationError> {
        self.invocation_mut()?
            .take_input(&port_id, index)
            .map(wit_value_handle)
            .map_err(wit_error)
    }

    fn read_handle(
        &mut self,
        handle: wit_contract::zed::sim_plugin::types::ValueHandle,
    ) -> Result<wit_contract::zed::sim_plugin::types::EncodedValue, WitInvocationError> {
        let value = self
            .invocation_mut()?
            .read_handle(sdk_value_handle(handle))
            .map_err(wit_error)?;
        Ok(wit_contract::zed::sim_plugin::types::EncodedValue {
            type_id: value.type_id().to_string(),
            family: wit_value_family(value.family()),
            abi_bytes: value.abi_bytes().map_err(|error| {
                wit_host_failure(&format!("plugin value ABI encoding failed: {error}"))
            })?,
        })
    }

    fn check_cancelled(&mut self) -> Result<(), WitInvocationError> {
        self.invocation_mut()?.check_cancelled().map_err(wit_error)
    }
}

impl provider_v2_wit_contract::zed::sim_provider_plugin::provider_streaming_host::Host
    for WasmStoreState
{
    fn start_request(
        &mut self,
        context: provider_v2_wit_contract::zed::sim_provider_plugin::types::InvocationContext,
        request: provider_v2_wit_contract::zed::sim_provider_plugin::types::RequestHead,
    ) -> Result<
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamHandle,
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError,
    > {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        if context.invocation != runtime.context.invocation
            || context.generation != runtime.context.generation
        {
            if let Err(revoke_error) = runtime.revoke() {
                eprintln!(
                    "provider-v2 route revocation after invalid invocation context failed: {revoke_error}"
                );
            }
            return Err(wit_provider_stream_error(
                WorkerProviderStreamError::InvalidRequestAuthority,
            ));
        }
        runtime
            .require_present_secret(request.secret_id.as_deref())
            .map_err(|error| {
                if let Err(revoke_error) = runtime.revoke() {
                    eprintln!(
                        "provider-v2 route revocation after unchecked secret failed: {revoke_error}"
                    );
                }
                wit_provider_stream_error(error)
            })?;
        let context = runtime.context.clone();
        let response = runtime
            .exchange(WorkerProviderStreamRequest::StartRequest {
                context,
                head: worker_provider_request_head(request),
            })
            .map_err(wit_provider_stream_error)?;
        match response {
            WorkerProviderStreamResponse::Stream(Ok(handle)) => {
                Ok(wit_provider_stream_handle(handle))
            }
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn write_request_chunk(
        &mut self,
        chunk: provider_v2_wit_contract::zed::sim_provider_plugin::types::RequestChunk,
    ) -> Result<(), provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError> {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let chunk = WorkerProviderRequestChunk {
            handle: runtime.worker_handle(chunk.handle),
            sequence: chunk.sequence,
            bytes: chunk.bytes,
            end: chunk.end,
        };
        match runtime
            .exchange(WorkerProviderStreamRequest::WriteRequestChunk(chunk))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::Unit(Ok(())) => Ok(()),
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn wait_response(
        &mut self,
        request: provider_v2_wit_contract::zed::sim_provider_plugin::types::WaitRequest,
    ) -> Result<
        provider_v2_wit_contract::zed::sim_provider_plugin::types::WaitOutcome,
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError,
    > {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let request = WorkerProviderWaitRequest {
            handle: runtime.worker_handle(request.handle),
            after_sequence: request.after_sequence,
            timeout_milliseconds: request.timeout_milliseconds,
        };
        match runtime
            .exchange(WorkerProviderStreamRequest::WaitResponse(request))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::Wait(Ok(outcome)) => {
                Ok(wit_provider_wait_outcome(outcome))
            }
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn start_upload(
        &mut self,
        request: provider_v2_wit_contract::zed::sim_provider_plugin::types::UploadRequest,
    ) -> Result<
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamHandle,
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError,
    > {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let request = WorkerProviderUploadRequest {
            handle: runtime.worker_handle(request.handle),
            port_id: request.port_id,
            media_type: request.media_type,
            byte_length: request.byte_length,
            content_sha256: request.content_sha256,
        };
        match runtime
            .exchange(WorkerProviderStreamRequest::StartUpload(request))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::Stream(Ok(handle)) => {
                Ok(wit_provider_stream_handle(handle))
            }
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn write_upload_chunk(
        &mut self,
        chunk: provider_v2_wit_contract::zed::sim_provider_plugin::types::RequestChunk,
    ) -> Result<(), provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError> {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let chunk = WorkerProviderRequestChunk {
            handle: runtime.worker_handle(chunk.handle),
            sequence: chunk.sequence,
            bytes: chunk.bytes,
            end: chunk.end,
        };
        match runtime
            .exchange(WorkerProviderStreamRequest::WriteUploadChunk(chunk))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::Unit(Ok(())) => Ok(()),
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn complete_upload(
        &mut self,
        request: provider_v2_wit_contract::zed::sim_provider_plugin::types::UploadCompletionRequest,
    ) -> Result<
        provider_v2_wit_contract::zed::sim_provider_plugin::types::UploadCompletion,
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError,
    > {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let request = WorkerProviderUploadCompletionRequest {
            handle: runtime.worker_handle(request.handle),
        };
        match runtime
            .exchange(WorkerProviderStreamRequest::CompleteUpload(request))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::UploadCompletion(Ok(completion)) => Ok(
                provider_v2_wit_contract::zed::sim_provider_plugin::types::UploadCompletion {
                    download_url: completion.download_url,
                },
            ),
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn request_cost(
        &mut self,
        request: provider_v2_wit_contract::zed::sim_provider_plugin::types::CostRequest,
    ) -> Result<
        provider_v2_wit_contract::zed::sim_provider_plugin::types::CostResponse,
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError,
    > {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let request = WorkerProviderCostRequest {
            handle: runtime.worker_handle(request.handle),
            operation: request.operation,
            currency: request.currency,
            maximum_microunits: request.maximum_microunits,
        };
        match runtime
            .exchange(WorkerProviderStreamRequest::RequestCost(request))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::Cost(Ok(response)) => Ok(
                provider_v2_wit_contract::zed::sim_provider_plugin::types::CostResponse {
                    accepted: response.accepted,
                    approved_microunits: response.approved_microunits,
                    receipt: response.receipt,
                },
            ),
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn report_progress(
        &mut self,
        progress: provider_v2_wit_contract::zed::sim_provider_plugin::types::Progress,
    ) -> Result<(), provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError> {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let progress = WorkerProviderProgress {
            handle: runtime.worker_handle(progress.handle),
            sequence: progress.sequence,
            completed: progress.completed,
            total: progress.total,
            message: progress.message,
        };
        match runtime
            .exchange(WorkerProviderStreamRequest::ReportProgress(progress))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::Unit(Ok(())) => Ok(()),
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn check_cancelled(
        &mut self,
        handle: provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamHandle,
    ) -> Result<(), provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError> {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let handle = runtime.worker_handle(handle);
        match runtime
            .exchange(WorkerProviderStreamRequest::CheckCancelled(handle))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::Unit(Ok(())) => Ok(()),
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }

    fn retry_request(
        &mut self,
        request: provider_v2_wit_contract::zed::sim_provider_plugin::types::RetryRequestOptions,
    ) -> Result<
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamHandle,
        provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError,
    > {
        let runtime = self
            .provider_runtime_mut()
            .map_err(wit_provider_stream_error)?;
        let request = runtime
            .checked_retry_request(
                runtime.worker_handle(request.handle),
                request.delay_milliseconds,
                SystemTime::now(),
            )
            .map_err(wit_provider_stream_error)?;
        runtime
            .wait_retry_delay(&request)
            .map_err(wit_provider_stream_error)?;
        match runtime
            .exchange(WorkerProviderStreamRequest::RetryRequest(request))
            .map_err(wit_provider_stream_error)?
        {
            WorkerProviderStreamResponse::Stream(Ok(handle)) => {
                Ok(wit_provider_stream_handle(handle))
            }
            _ => Err(wit_provider_stream_error(
                WorkerProviderStreamError::HostFailure,
            )),
        }
    }
}

fn resolve_retry_delay(
    _status: u16,
    fallback_milliseconds: u64,
    retry_after: Option<&str>,
    now: SystemTime,
) -> Result<u64, WorkerProviderStreamError> {
    let retry_after_milliseconds = retry_after.and_then(|value| {
        let value = value.trim();
        if !value.is_empty() && value.is_ascii() && value.bytes().all(|byte| byte.is_ascii_digit())
        {
            let seconds = value.parse::<u64>().unwrap_or(u64::MAX);
            return Some(
                seconds
                    .saturating_mul(1_000)
                    .min(MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS),
            );
        }
        if value.is_empty() {
            return None;
        }
        httpdate::parse_http_date(value).ok().map(|date| {
            let milliseconds = date.duration_since(now).unwrap_or_default().as_millis();
            u64::try_from(milliseconds)
                .unwrap_or(u64::MAX)
                .min(MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS)
        })
    });
    let milliseconds = retry_after_milliseconds.unwrap_or(fallback_milliseconds);
    if milliseconds > MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS {
        return Err(WorkerProviderStreamError::WaitLimit);
    }
    Ok(milliseconds)
}

impl WasmStoreState {
    fn invocation_mut(&mut self) -> Result<&mut InvocationHost, WitInvocationError> {
        self.invocation
            .as_mut()
            .ok_or_else(|| wit_host_failure("WASM store has no active plugin invocation"))
    }

    fn call(&mut self, call: CapabilityCall) -> Result<CapabilityResponse, WitInvocationError> {
        self.invocation_mut()?.call(call).map_err(wit_error)
    }

    fn provider_runtime_mut(
        &mut self,
    ) -> Result<&mut ProviderV2RuntimeHost, WorkerProviderStreamError> {
        self.provider_runtime
            .as_mut()
            .ok_or(WorkerProviderStreamError::InvalidRequestAuthority)
    }
}

impl provider_v2_wit_contract::zed::sim_provider_plugin::types::Host for WasmStoreState {}

fn wit_value_handle(handle: ValueHandle) -> wit_contract::zed::sim_plugin::host::ValueHandle {
    wit_contract::zed::sim_plugin::host::ValueHandle {
        invocation: handle.invocation,
        slot: handle.slot,
        generation: handle.generation,
    }
}

fn sdk_value_handle(handle: wit_contract::zed::sim_plugin::host::ValueHandle) -> ValueHandle {
    ValueHandle {
        invocation: handle.invocation,
        slot: handle.slot,
        generation: handle.generation,
    }
}

fn sdk_manifest_projection(
    projection: wit_contract::zed::sim_plugin::types::ManifestProjection,
) -> Result<ComponentManifestProjection, PluginError> {
    Ok(ComponentManifestProjection {
        component_world: projection.component_world,
        schema_version: projection.schema_version,
        identifier: projection.identifier,
        plugin_version: sdk_api_version(projection.plugin_version),
        api: sim_plugin_sdk::ApiRequirement {
            major: projection.api.major,
            minimum_minor: projection.api.minimum_minor,
            maximum_minor: projection.api.maximum_minor,
            required_features: projection.api.required_features,
        },
        nodes: projection
            .nodes
            .into_iter()
            .map(sdk_projected_node)
            .collect::<Result<Vec<_>, PluginError>>()?,
        capabilities: projection
            .capabilities
            .into_iter()
            .map(|request| sim_plugin_sdk::CapabilityRequest {
                kind: sdk_capability_kind(request.kind),
                scope: request.scope,
                quota: sim_plugin_sdk::CapabilityQuota {
                    maximum_operations: request.quota.maximum_operations,
                    maximum_request_bytes: request.quota.maximum_request_bytes,
                    maximum_response_bytes: request.quota.maximum_response_bytes,
                    maximum_total_bytes: request.quota.maximum_total_bytes,
                    maximum_handles: request.quota.maximum_handles,
                    timeout_milliseconds: request.quota.timeout_milliseconds,
                },
            })
            .collect(),
        ui: projection
            .ui
            .into_iter()
            .map(|contribution| sim_plugin_sdk::UiContribution {
                id: contribution.id,
                surface: contribution.surface,
                state_schema: contribution.state_schema,
            })
            .collect(),
        routes: projection
            .routes
            .into_iter()
            .map(|route| sim_plugin_sdk::RouteDeclaration {
                id: route.id,
                method: route.method,
                path: route.path,
                maximum_request_bytes: route.maximum_request_bytes,
                maximum_response_bytes: route.maximum_response_bytes,
            })
            .collect(),
        legacy_mappings: projection
            .legacy_mappings
            .into_iter()
            .map(|mapping| sim_plugin_sdk::ComponentLegacyMapping {
                legacy_identifier: mapping.legacy_identifier,
                node_id: mapping.node_id,
                node_version: sdk_api_version(mapping.node_version),
            })
            .collect(),
    })
}

fn sdk_provider_binding_set(
    binding_set: wit_contract::zed::sim_plugin::types::ProviderBindingSet,
) -> Result<ProviderBindingSet, PluginError> {
    Ok(ProviderBindingSet {
        schema_version: binding_set.schema_version,
        implementation_namespace: binding_set.implementation_namespace,
        bindings_sha256: binding_set.bindings_sha256,
        bindings: binding_set
            .bindings
            .into_iter()
            .map(|binding| {
                Ok(ProviderBindingClaim {
                    feature_id: binding.feature_id,
                    node_id: binding.node_id,
                    contract_sha256: binding.contract_sha256,
                    transport_schema: CanonicalTypeId::from_str(&binding.transport_schema)
                        .map_err(|_| PluginError::ProviderBindingMismatch)?,
                    materializer_schema: CanonicalTypeId::from_str(&binding.materializer_schema)
                        .map_err(|_| PluginError::ProviderBindingMismatch)?,
                })
            })
            .collect::<Result<Vec<_>, PluginError>>()?,
    })
}

fn sdk_projected_node(
    node: wit_contract::zed::sim_plugin::types::Node,
) -> Result<PluginNode, PluginError> {
    Ok(PluginNode {
        id: node.id,
        version: sdk_api_version(node.version),
        display_name: node.display_name,
        category: node.category,
        ports: node
            .ports
            .into_iter()
            .map(sdk_projected_port)
            .collect::<Result<Vec<_>, PluginError>>()?,
        determinism: match node.determinism {
            wit_contract::zed::sim_plugin::types::DeterminismPolicy::Deterministic => {
                sim_plugin_sdk::DeterminismPolicy::Deterministic
            }
            wit_contract::zed::sim_plugin::types::DeterminismPolicy::Seeded => {
                sim_plugin_sdk::DeterminismPolicy::Seeded
            }
            wit_contract::zed::sim_plugin::types::DeterminismPolicy::External => {
                sim_plugin_sdk::DeterminismPolicy::External
            }
        },
        cache: match node.cache {
            wit_contract::zed::sim_plugin::types::CachePolicy::InputIdentity => {
                sim_plugin_sdk::CachePolicy::InputIdentity
            }
            wit_contract::zed::sim_plugin::types::CachePolicy::Never => {
                sim_plugin_sdk::CachePolicy::Never
            }
            wit_contract::zed::sim_plugin::types::CachePolicy::PluginKey => {
                sim_plugin_sdk::CachePolicy::PluginKey
            }
        },
        effects: match node.effects {
            wit_contract::zed::sim_plugin::types::EffectPolicy::Pure => {
                sim_plugin_sdk::EffectPolicy::Pure
            }
            wit_contract::zed::sim_plugin::types::EffectPolicy::Transactional => {
                sim_plugin_sdk::EffectPolicy::Transactional
            }
            wit_contract::zed::sim_plugin::types::EffectPolicy::Provider => {
                sim_plugin_sdk::EffectPolicy::Provider
            }
        },
    })
}

fn sdk_projected_port(
    port: wit_contract::zed::sim_plugin::types::Port,
) -> Result<sim_plugin_sdk::PluginPort, PluginError> {
    let type_id = port.type_id.parse().map_err(PluginContractError::from)?;
    let default = port.default.map(sdk_projected_scalar).transpose()?;
    Ok(sim_plugin_sdk::PluginPort {
        id: port.id,
        name: port.name,
        direction: match port.direction {
            wit_contract::zed::sim_plugin::types::PortDirection::Input => PortDirection::Input,
            wit_contract::zed::sim_plugin::types::PortDirection::Output => PortDirection::Output,
        },
        type_id,
        cardinality: match port.cardinality {
            wit_contract::zed::sim_plugin::types::PortCardinality::Singular => {
                PortCardinality::Singular
            }
            wit_contract::zed::sim_plugin::types::PortCardinality::List => PortCardinality::List,
        },
        presence: match port.presence {
            wit_contract::zed::sim_plugin::types::PortPresence::Required => PortPresence::Required,
            wit_contract::zed::sim_plugin::types::PortPresence::Optional => PortPresence::Optional,
            wit_contract::zed::sim_plugin::types::PortPresence::Hidden => PortPresence::Hidden,
        },
        hidden: port.hidden,
        lazy: port.lazy,
        default,
        serialization: match port.serialization {
            wit_contract::zed::sim_plugin::types::PortSerialization::Inline => {
                PortSerialization::Inline
            }
            wit_contract::zed::sim_plugin::types::PortSerialization::Handle => {
                PortSerialization::Handle
            }
            wit_contract::zed::sim_plugin::types::PortSerialization::ArtifactReference => {
                PortSerialization::ArtifactReference
            }
            wit_contract::zed::sim_plugin::types::PortSerialization::OpaquePreserved => {
                PortSerialization::OpaquePreserved
            }
        },
        accepted_legacy_names: port.accepted_legacy_names,
    })
}

fn sdk_projected_scalar(
    scalar: wit_contract::zed::sim_plugin::types::ScalarValue,
) -> Result<sim_plugin_sdk::ScalarValue, PluginError> {
    const MAX_SCALAR_NODES: usize = 65_536;
    const MAX_SCALAR_DEPTH: usize = 64;
    const MAX_SCALAR_BYTES: usize = 8 * 1024 * 1024;

    let root =
        usize::try_from(scalar.root_node).map_err(|_| PluginError::ManifestProjectionMismatch)?;
    if scalar.nodes.is_empty()
        || scalar.nodes.len() > MAX_SCALAR_NODES
        || root >= scalar.nodes.len()
    {
        return Err(PluginError::ManifestProjectionMismatch);
    }

    let mut parent_counts = vec![0_u8; scalar.nodes.len()];
    let mut encoded_bytes = 0_usize;
    for node in &scalar.nodes {
        use wit_contract::zed::sim_plugin::types::ScalarNode;
        let children: &[u32] = match node {
            ScalarNode::ListValue(children) => children,
            ScalarNode::RecordValue(entries) => {
                for entry in entries {
                    encoded_bytes = encoded_bytes
                        .checked_add(entry.key.len())
                        .ok_or(PluginError::ManifestProjectionMismatch)?;
                    let child = usize::try_from(entry.value_node)
                        .map_err(|_| PluginError::ManifestProjectionMismatch)?;
                    let parent_count = parent_counts
                        .get_mut(child)
                        .ok_or(PluginError::ManifestProjectionMismatch)?;
                    *parent_count = parent_count
                        .checked_add(1)
                        .ok_or(PluginError::ManifestProjectionMismatch)?;
                    if *parent_count > 1 {
                        return Err(PluginError::ManifestProjectionMismatch);
                    }
                }
                &[]
            }
            ScalarNode::TextValue(value) => {
                encoded_bytes = encoded_bytes
                    .checked_add(value.len())
                    .ok_or(PluginError::ManifestProjectionMismatch)?;
                &[]
            }
            ScalarNode::BytesValue(value) => {
                encoded_bytes = encoded_bytes
                    .checked_add(value.len())
                    .ok_or(PluginError::ManifestProjectionMismatch)?;
                &[]
            }
            ScalarNode::NullValue
            | ScalarNode::BooleanValue(_)
            | ScalarNode::IntegerValue(_)
            | ScalarNode::FloatValue(_) => &[],
        };
        for child in children {
            let child =
                usize::try_from(*child).map_err(|_| PluginError::ManifestProjectionMismatch)?;
            let parent_count = parent_counts
                .get_mut(child)
                .ok_or(PluginError::ManifestProjectionMismatch)?;
            *parent_count = parent_count
                .checked_add(1)
                .ok_or(PluginError::ManifestProjectionMismatch)?;
            if *parent_count > 1 {
                return Err(PluginError::ManifestProjectionMismatch);
            }
        }
        if encoded_bytes > MAX_SCALAR_BYTES {
            return Err(PluginError::ManifestProjectionMismatch);
        }
    }
    if parent_counts[root] != 0
        || parent_counts
            .iter()
            .enumerate()
            .any(|(index, count)| index != root && *count != 1)
    {
        return Err(PluginError::ManifestProjectionMismatch);
    }

    let mut states = vec![0_u8; scalar.nodes.len()];
    let mut order = Vec::with_capacity(scalar.nodes.len());
    let mut stack = vec![(root, 0_usize, false)];
    while let Some((index, depth, exiting)) = stack.pop() {
        if depth > MAX_SCALAR_DEPTH {
            return Err(PluginError::ManifestProjectionMismatch);
        }
        if exiting {
            states[index] = 2;
            order.push(index);
            continue;
        }
        match states[index] {
            1 => return Err(PluginError::ManifestProjectionMismatch),
            2 => continue,
            _ => states[index] = 1,
        }
        stack.push((index, depth, true));
        let children = projected_scalar_children(&scalar.nodes[index]);
        for child in children.into_iter().rev() {
            let child =
                usize::try_from(child).map_err(|_| PluginError::ManifestProjectionMismatch)?;
            if child >= scalar.nodes.len() {
                return Err(PluginError::ManifestProjectionMismatch);
            }
            stack.push((child, depth + 1, false));
        }
    }
    if states.iter().any(|state| *state != 2) {
        return Err(PluginError::ManifestProjectionMismatch);
    }

    let mut values = vec![None; scalar.nodes.len()];
    for index in order {
        use wit_contract::zed::sim_plugin::types::ScalarNode;
        let value = match &scalar.nodes[index] {
            ScalarNode::NullValue => sim_plugin_sdk::ScalarValue::Null,
            ScalarNode::BooleanValue(value) => sim_plugin_sdk::ScalarValue::Boolean(*value),
            ScalarNode::IntegerValue(value) => sim_plugin_sdk::ScalarValue::Integer(*value),
            ScalarNode::FloatValue(value) => sim_plugin_sdk::ScalarValue::Float(*value),
            ScalarNode::TextValue(value) => sim_plugin_sdk::ScalarValue::String(value.clone()),
            ScalarNode::BytesValue(value) => sim_plugin_sdk::ScalarValue::Bytes(value.clone()),
            ScalarNode::ListValue(children) => sim_plugin_sdk::ScalarValue::List(
                take_projected_scalar_children(children, &mut values)?,
            ),
            ScalarNode::RecordValue(entries) => sim_plugin_sdk::ScalarValue::Record(
                entries
                    .iter()
                    .map(|entry| {
                        let child = take_projected_scalar_child(entry.value_node, &mut values)?;
                        Ok((entry.key.clone(), child))
                    })
                    .collect::<Result<Vec<_>, PluginError>>()?,
            ),
        };
        values[index] = Some(value);
    }
    let value = values[root]
        .take()
        .ok_or(PluginError::ManifestProjectionMismatch)?;
    value
        .abi_bytes()
        .map_err(|_| PluginError::ManifestProjectionMismatch)?;
    Ok(value)
}

fn projected_scalar_children(node: &wit_contract::zed::sim_plugin::types::ScalarNode) -> Vec<u32> {
    use wit_contract::zed::sim_plugin::types::ScalarNode;
    match node {
        ScalarNode::ListValue(children) => children.clone(),
        ScalarNode::RecordValue(entries) => entries.iter().map(|entry| entry.value_node).collect(),
        ScalarNode::NullValue
        | ScalarNode::BooleanValue(_)
        | ScalarNode::IntegerValue(_)
        | ScalarNode::FloatValue(_)
        | ScalarNode::TextValue(_)
        | ScalarNode::BytesValue(_) => Vec::new(),
    }
}

fn take_projected_scalar_children(
    children: &[u32],
    values: &mut [Option<sim_plugin_sdk::ScalarValue>],
) -> Result<Vec<sim_plugin_sdk::ScalarValue>, PluginError> {
    children
        .iter()
        .map(|child| take_projected_scalar_child(*child, values))
        .collect()
}

fn take_projected_scalar_child(
    child: u32,
    values: &mut [Option<sim_plugin_sdk::ScalarValue>],
) -> Result<sim_plugin_sdk::ScalarValue, PluginError> {
    let child = usize::try_from(child).map_err(|_| PluginError::ManifestProjectionMismatch)?;
    values
        .get_mut(child)
        .and_then(Option::take)
        .ok_or(PluginError::ManifestProjectionMismatch)
}

fn sdk_api_version(
    version: wit_contract::zed::sim_plugin::types::ApiVersion,
) -> sim_plugin_sdk::ApiVersion {
    sim_plugin_sdk::ApiVersion::new(version.major, version.minor, version.patch)
}

fn wit_error(error: InvocationError) -> WitInvocationError {
    use wit_contract::zed::sim_plugin::types::{
        CapabilityError, InvocationError as WitError, PortIndexError, QuotaError, ValueFamilyError,
    };
    match error {
        InvocationError::Cancelled => WitError::Cancelled,
        InvocationError::TimedOut => WitError::TimedOut,
        InvocationError::UnknownPort(port) => WitError::UnknownPort(sanitize_diagnostic(&port)),
        InvocationError::WrongDirection(port) => {
            WitError::WrongDirection(sanitize_diagnostic(&port))
        }
        InvocationError::MissingRequiredPort(port) => {
            WitError::MissingRequiredPort(sanitize_diagnostic(&port))
        }
        InvocationError::InvalidCardinality(port) => {
            WitError::InvalidCardinality(sanitize_diagnostic(&port))
        }
        InvocationError::IndexOutOfBounds { port, index } => {
            WitError::IndexOutOfBounds(PortIndexError {
                port: sanitize_diagnostic(&port),
                index,
            })
        }
        InvocationError::AlreadyTaken { port, index } => WitError::AlreadyTaken(PortIndexError {
            port: sanitize_diagnostic(&port),
            index,
        }),
        InvocationError::InvalidHandle => WitError::InvalidHandle,
        InvocationError::RevokedHandle => WitError::RevokedHandle,
        InvocationError::WrongValueFamily {
            port,
            expected,
            actual,
        } => WitError::WrongValueFamily(ValueFamilyError {
            port: sanitize_diagnostic(&port),
            expected: wit_value_family(expected),
            actual: wit_value_family(actual),
        }),
        InvocationError::OutputAlreadyFinished(port) => {
            WitError::OutputAlreadyFinished(sanitize_diagnostic(&port))
        }
        InvocationError::UnfinishedOutput(port) => {
            WitError::UnfinishedOutput(sanitize_diagnostic(&port))
        }
        InvocationError::CapabilityDenied { kind, scope } => {
            WitError::CapabilityDenied(CapabilityError {
                kind: wit_capability_kind(kind),
                scope: sanitize_diagnostic(&scope),
            })
        }
        InvocationError::QuotaExceeded { kind, limit } => WitError::QuotaExceeded(QuotaError {
            kind: wit_capability_kind(kind),
            limit: sanitize_diagnostic(&limit),
        }),
        InvocationError::InvocationQuotaExceeded { limit } => {
            WitError::InvocationQuotaExceeded(sanitize_diagnostic(&limit))
        }
        InvocationError::InvalidCapabilityRequest(message) => {
            WitError::InvalidCapabilityRequest(sanitize_diagnostic(&message))
        }
        InvocationError::HostFailure(message) => {
            WitError::HostFailure(sanitize_diagnostic(&message))
        }
        InvocationError::PluginFailure(message) => {
            WitError::PluginFailure(sanitize_diagnostic(&message))
        }
    }
}

fn sdk_error(error: WitInvocationError) -> InvocationError {
    use wit_contract::zed::sim_plugin::types::InvocationError as WitError;
    match error {
        WitError::Cancelled => InvocationError::Cancelled,
        WitError::TimedOut => InvocationError::TimedOut,
        WitError::UnknownPort(port) => InvocationError::UnknownPort(sanitize_diagnostic(&port)),
        WitError::WrongDirection(port) => {
            InvocationError::WrongDirection(sanitize_diagnostic(&port))
        }
        WitError::MissingRequiredPort(port) => {
            InvocationError::MissingRequiredPort(sanitize_diagnostic(&port))
        }
        WitError::InvalidCardinality(port) => {
            InvocationError::InvalidCardinality(sanitize_diagnostic(&port))
        }
        WitError::IndexOutOfBounds(error) => InvocationError::IndexOutOfBounds {
            port: sanitize_diagnostic(&error.port),
            index: error.index,
        },
        WitError::AlreadyTaken(error) => InvocationError::AlreadyTaken {
            port: sanitize_diagnostic(&error.port),
            index: error.index,
        },
        WitError::InvalidHandle => InvocationError::InvalidHandle,
        WitError::RevokedHandle => InvocationError::RevokedHandle,
        WitError::WrongValueFamily(error) => InvocationError::WrongValueFamily {
            port: sanitize_diagnostic(&error.port),
            expected: sdk_value_family(error.expected),
            actual: sdk_value_family(error.actual),
        },
        WitError::OutputAlreadyFinished(port) => {
            InvocationError::OutputAlreadyFinished(sanitize_diagnostic(&port))
        }
        WitError::UnfinishedOutput(port) => {
            InvocationError::UnfinishedOutput(sanitize_diagnostic(&port))
        }
        WitError::CapabilityDenied(error) => InvocationError::CapabilityDenied {
            kind: sdk_capability_kind(error.kind),
            scope: sanitize_diagnostic(&error.scope),
        },
        WitError::QuotaExceeded(error) => InvocationError::QuotaExceeded {
            kind: sdk_capability_kind(error.kind),
            limit: sanitize_diagnostic(&error.limit),
        },
        WitError::InvocationQuotaExceeded(limit) => InvocationError::InvocationQuotaExceeded {
            limit: sanitize_diagnostic(&limit),
        },
        WitError::InvalidCapabilityRequest(message) => {
            InvocationError::InvalidCapabilityRequest(sanitize_diagnostic(&message))
        }
        WitError::HostFailure(message) => {
            InvocationError::HostFailure(sanitize_diagnostic(&message))
        }
        WitError::PluginFailure(message) => {
            InvocationError::PluginFailure(sanitize_diagnostic(&message))
        }
    }
}

fn wit_host_failure(message: &str) -> WitInvocationError {
    WitInvocationError::HostFailure(sanitize_diagnostic(message))
}

fn wit_value_family(
    family: sim_plugin_sdk::ValueFamily,
) -> wit_contract::zed::sim_plugin::types::ValueFamily {
    use sim_plugin_sdk::ValueFamily;
    use wit_contract::zed::sim_plugin::types::ValueFamily as WitFamily;
    match family {
        ValueFamily::Scalar => WitFamily::Scalar,
        ValueFamily::Tensor => WitFamily::Tensor,
        ValueFamily::Artifact => WitFamily::Artifact,
        ValueFamily::Model => WitFamily::Model,
    }
}

fn sdk_value_family(
    family: wit_contract::zed::sim_plugin::types::ValueFamily,
) -> sim_plugin_sdk::ValueFamily {
    use sim_plugin_sdk::ValueFamily;
    use wit_contract::zed::sim_plugin::types::ValueFamily as WitFamily;
    match family {
        WitFamily::Scalar => ValueFamily::Scalar,
        WitFamily::Tensor => ValueFamily::Tensor,
        WitFamily::Artifact => ValueFamily::Artifact,
        WitFamily::Model => ValueFamily::Model,
    }
}

fn wit_port_cardinality(
    cardinality: PortCardinality,
) -> wit_contract::zed::sim_plugin::types::PortCardinality {
    use wit_contract::zed::sim_plugin::types::PortCardinality as WitCardinality;
    match cardinality {
        PortCardinality::Singular => WitCardinality::Singular,
        PortCardinality::List => WitCardinality::List,
    }
}

fn wit_port_presence(presence: PortPresence) -> wit_contract::zed::sim_plugin::types::PortPresence {
    use wit_contract::zed::sim_plugin::types::PortPresence as WitPresence;
    match presence {
        PortPresence::Required => WitPresence::Required,
        PortPresence::Optional => WitPresence::Optional,
        PortPresence::Hidden => WitPresence::Hidden,
    }
}

fn wit_port_serialization(
    serialization: PortSerialization,
) -> wit_contract::zed::sim_plugin::types::PortSerialization {
    use wit_contract::zed::sim_plugin::types::PortSerialization as WitSerialization;
    match serialization {
        PortSerialization::Inline => WitSerialization::Inline,
        PortSerialization::Handle => WitSerialization::Handle,
        PortSerialization::ArtifactReference => WitSerialization::ArtifactReference,
        PortSerialization::OpaquePreserved => WitSerialization::OpaquePreserved,
    }
}

fn wit_capability_kind(
    kind: sim_plugin_sdk::CapabilityKind,
) -> wit_contract::zed::sim_plugin::types::CapabilityKind {
    use sim_plugin_sdk::CapabilityKind;
    use wit_contract::zed::sim_plugin::types::CapabilityKind as WitKind;
    match kind {
        CapabilityKind::Filesystem => WitKind::Filesystem,
        CapabilityKind::NetworkProvider => WitKind::NetworkProvider,
        CapabilityKind::Secret => WitKind::Secret,
        CapabilityKind::Clock => WitKind::Clock,
        CapabilityKind::Randomness => WitKind::Randomness,
        CapabilityKind::Model => WitKind::Model,
        CapabilityKind::TransactionalOutput => WitKind::TransactionalOutput,
        CapabilityKind::SanitizedLog => WitKind::SanitizedLog,
        CapabilityKind::DeclarativeUi => WitKind::DeclarativeUi,
        CapabilityKind::Route => WitKind::Route,
        CapabilityKind::ProviderUpload => WitKind::ProviderUpload,
        CapabilityKind::ProviderCost => WitKind::ProviderCost,
    }
}

fn sdk_capability_kind(
    kind: wit_contract::zed::sim_plugin::types::CapabilityKind,
) -> sim_plugin_sdk::CapabilityKind {
    use sim_plugin_sdk::CapabilityKind;
    use wit_contract::zed::sim_plugin::types::CapabilityKind as WitKind;
    match kind {
        WitKind::Filesystem => CapabilityKind::Filesystem,
        WitKind::NetworkProvider => CapabilityKind::NetworkProvider,
        WitKind::Secret => CapabilityKind::Secret,
        WitKind::Clock => CapabilityKind::Clock,
        WitKind::Randomness => CapabilityKind::Randomness,
        WitKind::Model => CapabilityKind::Model,
        WitKind::TransactionalOutput => CapabilityKind::TransactionalOutput,
        WitKind::SanitizedLog => CapabilityKind::SanitizedLog,
        WitKind::DeclarativeUi => CapabilityKind::DeclarativeUi,
        WitKind::Route => CapabilityKind::Route,
        WitKind::ProviderUpload => CapabilityKind::ProviderUpload,
        WitKind::ProviderCost => CapabilityKind::ProviderCost,
    }
}

fn wit_cancel_reason(reason: CancelReason) -> wit_contract::zed::sim_plugin::types::CancelReason {
    use wit_contract::zed::sim_plugin::types::CancelReason as WitReason;
    match reason {
        CancelReason::User => WitReason::User,
        CancelReason::Timeout => WitReason::Timeout,
        CancelReason::HostShutdown => WitReason::HostShutdown,
        CancelReason::CapabilityRevoked => WitReason::CapabilityRevoked,
    }
}

impl Drop for InvocationHost {
    fn drop(&mut self) {
        if !self.terminal {
            self.abort();
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct InvocationResult {
    pub outputs: BTreeMap<String, Vec<PluginValue>>,
    pub output_presence: BTreeMap<String, bool>,
    pub effects: CapabilityEffects,
}

fn validate_input_port(
    port: &sim_plugin_sdk::PluginPort,
    present: bool,
    values: &[PluginValue],
    registry: &TypeRegistry,
    limits: &ComponentLimits,
) -> Result<u64, InvocationError> {
    if port.presence == PortPresence::Required && !present {
        return Err(InvocationError::MissingRequiredPort(port.id.clone()));
    }
    if !present && !values.is_empty() {
        return Err(InvocationError::InvalidCardinality(port.id.clone()));
    }
    match port.cardinality {
        PortCardinality::Singular if present && values.len() != 1 => {
            return Err(InvocationError::InvalidCardinality(port.id.clone()));
        }
        PortCardinality::List if values.len() > limits.maximum_values_per_port => {
            return Err(InvocationError::InvalidCardinality(port.id.clone()));
        }
        _ => {}
    }
    let expected = registry.family(&port.type_id).map_err(|error| {
        InvocationError::HostFailure(format!("plugin type registry failed: {error}"))
    })?;
    let mut total_bytes = 0_u64;
    for value in values {
        if value.family() != expected {
            return Err(InvocationError::WrongValueFamily {
                port: port.id.clone(),
                expected,
                actual: value.family(),
            });
        }
        require_exact_port_type(port, value)?;
        total_bytes = total_bytes
            .checked_add(value_size(value)?)
            .ok_or_else(|| value_quota_error())?;
        if total_bytes > limits.maximum_value_bytes {
            return Err(value_quota_error());
        }
    }
    Ok(total_bytes)
}

fn validate_media_content(
    value: &PluginValue,
    content: &InvocationMediaContent,
    limits: &ComponentLimits,
) -> Result<u64, InvocationError> {
    if !matches!(value.representation(), PluginValueRepresentation::Tensor(_)) {
        return Err(InvocationError::HostFailure(
            "media content requires a retained tensor value".to_owned(),
        ));
    }
    if !valid_media_type(&content.media_type)
        || !valid_sha256(&content.source_value_sha256)
        || !valid_sha256(&content.content_sha256)
    {
        return Err(invalid_value("media content metadata"));
    }
    let source_abi = value.abi_bytes().map_err(|error| {
        InvocationError::HostFailure(format!("plugin value ABI encoding failed: {error}"))
    })?;
    if encode_hex(&Sha256::digest(&source_abi)) != content.source_value_sha256
        || encode_hex(&Sha256::digest(&content.bytes)) != content.content_sha256
    {
        return Err(invalid_value("media content digest"));
    }
    let byte_length = u64::try_from(content.bytes.len()).map_err(|_| value_quota_error())?;
    if byte_length == 0 || byte_length > limits.maximum_value_bytes {
        return Err(value_quota_error());
    }
    Ok(byte_length)
}

fn media_batch_count(
    value: &PluginValue,
    first_image_only: bool,
) -> Result<usize, InvocationError> {
    const MAX_MEDIA_BATCH_ITEMS: usize = 128;
    let PluginValueRepresentation::Tensor(tensor) = value.representation() else {
        return Err(InvocationError::HostFailure(
            "media content requires a retained tensor value".to_owned(),
        ));
    };
    let type_id = value.type_id();
    let shape = tensor.descriptor().shape();
    let count = if shape.len() > 3 { shape[0] } else { 1 };
    // Video frames form one stream; Beeble image uploads select one frame while
    // retaining the original tensor identity for digest validation.
    let count = if count > 0
        && type_id.namespace() == "sim"
        && (type_id.name() == "video" || (first_image_only && type_id.name() == "image"))
        && type_id.major() == 1
    {
        1
    } else {
        usize::try_from(count).map_err(|_| value_quota_error())?
    };
    if count == 0 || count > MAX_MEDIA_BATCH_ITEMS {
        return Err(InvocationError::InvalidCardinality(
            "media input batch".to_owned(),
        ));
    }
    Ok(count)
}

fn valid_media_type(value: &str) -> bool {
    value.len() <= 128
        && value.split_once('/').is_some_and(|(kind, subtype)| {
            !kind.is_empty()
                && !subtype.is_empty()
                && kind
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                && subtype
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'+' | b'.'))
        })
}

fn validate_finished_output(
    port: &sim_plugin_sdk::PluginPort,
    present: bool,
    length: usize,
) -> Result<(), InvocationError> {
    if !present {
        if port.presence != PortPresence::Optional {
            return Err(InvocationError::MissingRequiredPort(port.id.clone()));
        }
        if length != 0 {
            return Err(InvocationError::InvalidCardinality(port.id.clone()));
        }
        return Ok(());
    }
    if port.cardinality == PortCardinality::Singular && length != 1 {
        return Err(if length == 0 {
            InvocationError::MissingRequiredPort(port.id.clone())
        } else {
            InvocationError::InvalidCardinality(port.id.clone())
        });
    }
    Ok(())
}

fn value_size(value: &PluginValue) -> Result<u64, InvocationError> {
    let size = match value.representation() {
        PluginValueRepresentation::Scalar(value) => scalar_size(value)?,
        PluginValueRepresentation::Tensor(value) => value.byte_length(),
        PluginValueRepresentation::Artifact(value) => {
            if canonical_artifact_identity(value.namespace(), value.identifier()).is_err()
                || !valid_sha256(value.digest())
            {
                return Err(invalid_value("artifact metadata"));
            }
            value.byte_length()
        }
        PluginValueRepresentation::Model(value) => {
            if !valid_value_text(value.identifier(), 4_096)
                || !valid_value_text(value.format(), 128)
                || !valid_sha256(value.digest())
            {
                return Err(invalid_value("model metadata"));
            }
            value
                .identifier()
                .len()
                .checked_add(value.format().len())
                .and_then(|length| length.checked_add(value.digest().len()))
                .and_then(|length| u64::try_from(length).ok())
                .ok_or_else(|| {
                    InvocationError::HostFailure("model value size overflow".to_owned())
                })?
        }
    };
    Ok(size)
}

fn encoded_value_response_size(value: &PluginValue) -> Result<u64, InvocationError> {
    let abi_bytes = value.abi_bytes().map_err(|error| {
        InvocationError::HostFailure(format!("plugin value ABI encoding failed: {error}"))
    })?;
    let type_id_bytes = value.type_id().to_string().len();
    type_id_bytes
        .checked_add(abi_bytes.len())
        .and_then(|bytes| bytes.checked_add(1))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| InvocationError::HostFailure("plugin value ABI size overflow".to_owned()))
}

pub(crate) fn require_exact_port_type(
    port: &sim_plugin_sdk::PluginPort,
    value: &PluginValue,
) -> Result<(), InvocationError> {
    if value.type_id() != &port.type_id {
        return Err(InvocationError::HostFailure(format!(
            "port `{}` expects canonical type `{}`, received `{}`",
            port.id,
            port.type_id,
            value.type_id()
        )));
    }
    Ok(())
}

fn scalar_size(value: &sim_plugin_sdk::ScalarValue) -> Result<u64, InvocationError> {
    use sim_plugin_sdk::ScalarValue;
    let mut stack = vec![(value, 0_usize)];
    let mut total = 0_u64;
    let mut value_count = 0_usize;
    while let Some((value, depth)) = stack.pop() {
        if depth > 64 || value_count >= 65_536 {
            return Err(InvocationError::HostFailure(
                "scalar nesting or value count exceeds the host limit".to_owned(),
            ));
        }
        value_count += 1;
        let bytes = match value {
            ScalarValue::Null => 0,
            ScalarValue::Boolean(_) => 1,
            ScalarValue::Integer(_) => 8,
            ScalarValue::Float(value) => {
                if !value.is_finite() {
                    return Err(invalid_value("non-finite scalar float"));
                }
                8
            }
            ScalarValue::String(value) => u64::try_from(value.len())
                .map_err(|_| InvocationError::HostFailure("scalar size overflow".to_owned()))?,
            ScalarValue::Bytes(value) => u64::try_from(value.len())
                .map_err(|_| InvocationError::HostFailure("scalar size overflow".to_owned()))?,
            ScalarValue::List(values) => {
                stack.extend(values.iter().map(|value| (value, depth + 1)));
                0
            }
            ScalarValue::Record(values) => {
                let mut keys = BTreeSet::new();
                for (key, value) in values {
                    if key.is_empty() || key.len() > 1_024 || !keys.insert(key) {
                        return Err(invalid_value("scalar record key"));
                    }
                    total = total
                        .checked_add(u64::try_from(key.len()).map_err(|_| {
                            InvocationError::HostFailure("scalar size overflow".to_owned())
                        })?)
                        .ok_or_else(|| {
                            InvocationError::HostFailure("scalar size overflow".to_owned())
                        })?;
                    stack.push((value, depth + 1));
                }
                0
            }
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| InvocationError::HostFailure("scalar size overflow".to_owned()))?;
    }
    Ok(total)
}

fn next_invocation_id() -> Result<u64, InvocationError> {
    NEXT_INVOCATION_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |identifier| {
            identifier.checked_add(1)
        })
        .map_err(|_| {
            InvocationError::HostFailure("invocation identifier space exhausted".to_owned())
        })
}

fn invocation_value_quota_error() -> InvocationError {
    InvocationError::InvocationQuotaExceeded {
        limit: "invocation-value-byte".to_owned(),
    }
}

fn value_quota_error() -> InvocationError {
    InvocationError::InvocationQuotaExceeded {
        limit: "value-byte".to_owned(),
    }
}

fn invalid_value(subject: &str) -> InvocationError {
    InvocationError::HostFailure(format!("invalid plugin {subject}"))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_value_text(value: &str, maximum_bytes: usize) -> bool {
    !value.is_empty() && value.len() <= maximum_bytes && !value.chars().any(char::is_control)
}

fn canonical_artifact_identity(namespace: &str, identifier: &str) -> Result<AssetIdentity, ()> {
    let namespace = AssetNamespace::from_locator_type(namespace).map_err(|_| ())?;
    AssetIdentity::new("plugin-wire", namespace, identifier).map_err(|_| ())
}

fn validate_component_projection(
    expected: &ComponentManifestProjection,
    actual: &ComponentManifestProjection,
) -> Result<(), PluginError> {
    if actual.canonical_bytes().is_err() || expected != actual {
        return Err(PluginError::ManifestProjectionMismatch);
    }
    Ok(())
}

fn component_compilation_error(error: wasmtime::Error) -> PluginError {
    PluginError::ComponentCompilation(sanitize_diagnostic(&error.to_string()))
}

fn component_instantiation_error(error: wasmtime::Error) -> PluginError {
    PluginError::WasmTrap(sanitize_diagnostic(&error.to_string()))
}

enum ComponentPreflightPhase {
    Instantiation,
    Manifest,
    ProviderBinding,
}

fn preflight_phase_error(phase: ComponentPreflightPhase, error: PluginError) -> PluginError {
    let phase = match phase {
        ComponentPreflightPhase::Instantiation => "instantiation",
        ComponentPreflightPhase::Manifest => "manifest",
        ComponentPreflightPhase::ProviderBinding => "provider-binding",
    };
    match error {
        PluginError::WasmTrap(message) => PluginError::WasmTrap(sanitize_diagnostic(&format!(
            "preflight {phase}: {message}"
        ))),
        PluginError::ComponentCompilation(message) => PluginError::ComponentCompilation(
            sanitize_diagnostic(&format!("preflight {phase}: {message}")),
        ),
        error => error,
    }
}

fn preflight_timed_phase_error(
    phase: ComponentPreflightPhase,
    error: PluginError,
    elapsed: Duration,
) -> PluginError {
    let error = match error {
        PluginError::WasmTrap(message) => {
            PluginError::WasmTrap(format!("elapsed_us={}: {message}", elapsed.as_micros()))
        }
        PluginError::ComponentCompilation(message) => PluginError::ComponentCompilation(format!(
            "elapsed_us={}: {message}",
            elapsed.as_micros()
        )),
        error => error,
    };
    preflight_phase_error(phase, error)
}

fn preflight_component_error(error: PluginError) -> PluginError {
    match error {
        PluginError::WasmTrap(message) => PluginError::ComponentCompilation(message),
        error => error,
    }
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
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

fn sanitize_diagnostic(message: &str) -> String {
    message
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .take(4_096)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{error::Error, thread};
    use uuid::Uuid;

    #[test]
    fn provider_v2_media_handles_reject_foreign_stale_and_revoked_authority() {
        let valid = ValueHandle {
            invocation: 41,
            slot: 2,
            generation: 7,
        };
        assert_eq!(validate_media_handle_identity(41, valid, Some(7)), Ok(()));
        assert_eq!(
            validate_media_handle_identity(
                41,
                ValueHandle {
                    invocation: 42,
                    ..valid
                },
                Some(7),
            ),
            Err(InvocationError::InvalidHandle)
        );
        assert_eq!(
            validate_media_handle_identity(
                41,
                ValueHandle {
                    generation: 8,
                    ..valid
                },
                Some(7),
            ),
            Err(InvocationError::RevokedHandle)
        );
        assert_eq!(
            validate_media_handle_identity(41, valid, None),
            Err(InvocationError::RevokedHandle)
        );
    }

    #[test]
    fn provider_v2_media_batch_registration_rejects_duplicate_and_excess_indices() {
        let content =
            InvocationMediaContent::new("image/png", "a".repeat(64), "b".repeat(64), vec![1]);
        let expected = content.clone();
        let mut inputs = InvocationInputs::default();
        assert_eq!(
            inputs.register_media_batch_item("images", 0, 0, content.clone()),
            Ok(())
        );
        assert!(matches!(
            inputs.register_media_batch_item("images", 0, 0, content.clone()),
            Err(InvocationError::InvalidCardinality(_))
        ));
        assert!(matches!(
            inputs.register_media_batch_item("images", 0, 128, content),
            Err(InvocationError::InvalidCardinality(_))
        ));
        assert_eq!(inputs.media_batches["images"][&0][&0], expected);
    }

    #[test]
    fn provider_v2_video_media_is_one_stream_while_image_batches_are_individual_items()
    -> Result<(), Box<dyn Error>> {
        use sim_plugin_sdk::{DType, DeviceId, StreamId, TensorDescriptor, TensorValue};

        let registry = TypeRegistry::built_in()?;
        for frame_count in [0_u64, 1, 2, 128, 240] {
            let descriptor = TensorDescriptor::contiguous(
                vec![frame_count, 1, 1, 3],
                DType::F32,
                DeviceId::from_source_device("cpu")?,
                StreamId::new(0),
            )?;
            let tensor = TensorValue::new(descriptor, frame_count * 12, "a".repeat(64))?;
            let video = PluginValue::tensor(
                registry.resolve("Video")?.clone(),
                tensor.clone(),
                &registry,
            )?;
            let image = PluginValue::tensor(registry.resolve("Image")?.clone(), tensor, &registry)?;
            if frame_count == 0 {
                for value in [video, image] {
                    assert!(matches!(
                        media_batch_count(&value, false),
                        Err(InvocationError::InvalidCardinality(_))
                    ));
                }
                continue;
            }
            assert_eq!(media_batch_count(&video, false)?, 1);
            if frame_count <= 128 {
                assert_eq!(
                    media_batch_count(&image, false)?,
                    usize::try_from(frame_count)?
                );
            } else {
                assert!(matches!(
                    media_batch_count(&image, false),
                    Err(InvocationError::InvalidCardinality(_))
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn beeble_first_image_media_survives_host_admission_and_handle_reads()
    -> Result<(), Box<dyn Error>> {
        use sim_plugin_sdk::{
            ApiRequirement, ApiVersion, CachePolicy, DType, DeterminismPolicy, DeviceId,
            EffectPolicy, ManifestProvenance, ManifestSignature, PLUGIN_SIGNATURE_ALGORITHM,
            PROVIDER_BINDING_API_FEATURE, PluginPort, PluginSigningKey, StreamId, TensorDescriptor,
            TensorValue,
        };
        use sim_runtime::{
            CapabilitySet, PermissionGrant, PermissionPolicy, PluginTrustPolicy,
            PluginVerificationKey,
        };

        let registry = TypeRegistry::built_in()?;
        let key = PluginSigningKey::new("test.beeble", b"0123456789abcdef0123456789abcdef")?;
        let trust = PluginTrustPolicy::new([PluginVerificationKey::new(
            "test.beeble",
            key.verification_key_bytes()?,
        )?])?;
        let host = PluginHost::with_configuration(
            ComponentLimits::default(),
            DEFAULT_API_FEATURES
                .iter()
                .map(|feature| (*feature).to_owned()),
        )?;
        for (node_id, port_id) in [
            ("BeebleSwitchXImageEdit", "image"),
            ("BeebleSwitchXImageEdit", "reference_image"),
            ("BeebleSwitchXVideoEdit", "reference_image"),
        ] {
            for (namespace, bound) in [
                ("zed.sim.provider.beeble", true),
                ("zed.sim.provider.beeble", false),
                ("test.other", true),
            ] {
                let mut binding = ProviderBindingSet {
                    schema_version: 1,
                    implementation_namespace: namespace.to_owned(),
                    bindings_sha256: "0".repeat(64),
                    bindings: vec![ProviderBindingClaim {
                        feature_id: "COMFY-NODE-0020".to_owned(),
                        node_id: node_id.to_owned(),
                        contract_sha256: "a".repeat(64),
                        transport_schema: "zed:sim-provider-transport@1".parse()?,
                        materializer_schema: "zed:sim-provider-materializer@1".parse()?,
                    }],
                };
                binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
                let mut manifest = PluginManifest {
                    schema_version: 1,
                    identifier: namespace.to_owned(),
                    plugin_version: ApiVersion::new(1, 0, 0),
                    api: ApiRequirement {
                        major: 1,
                        minimum_minor: 0,
                        maximum_minor: 0,
                        required_features: vec![PROVIDER_BINDING_API_FEATURE.to_owned()],
                    },
                    digest_sha256: "0".repeat(64),
                    signature: ManifestSignature {
                        algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
                        key_id: "test.beeble".to_owned(),
                        value: "0".repeat(128),
                    },
                    provenance: ManifestProvenance {
                        source: "fixture://beeble".to_owned(),
                        publisher: "Test".to_owned(),
                        registry: None,
                    },
                    provider_binding: bound.then_some(binding),
                    nodes: vec![PluginNode {
                        id: node_id.to_owned(),
                        version: ApiVersion::new(1, 0, 0),
                        display_name: "Beeble".to_owned(),
                        category: "test".to_owned(),
                        ports: vec![PluginPort {
                            id: port_id.to_owned(),
                            name: port_id.to_owned(),
                            direction: PortDirection::Input,
                            type_id: registry.resolve("IMAGE")?.clone(),
                            cardinality: PortCardinality::Singular,
                            presence: PortPresence::Required,
                            hidden: false,
                            lazy: false,
                            default: None,
                            serialization: PortSerialization::Handle,
                            accepted_legacy_names: Vec::new(),
                        }],
                        determinism: DeterminismPolicy::External,
                        cache: CachePolicy::Never,
                        effects: EffectPolicy::Provider,
                    }],
                    capabilities: Vec::new(),
                    ui: Vec::new(),
                    routes: Vec::new(),
                    legacy_mappings: Vec::new(),
                };
                manifest.signature.value = key.sign_manifest(&manifest)?;
                let authorization = trust.authorize_manifest(
                    &manifest,
                    &PermissionPolicy::new(
                        "test",
                        [PermissionGrant::new(
                            "test",
                            namespace,
                            CapabilitySet::new([]),
                            "signed-test-fixture",
                        )?],
                    )?,
                )?;
                for batch in [2_u64, 240] {
                    let value = PluginValue::tensor(
                        registry.resolve("IMAGE")?.clone(),
                        TensorValue::new(
                            TensorDescriptor::contiguous(
                                vec![batch, 1, 1, 3],
                                DType::F32,
                                DeviceId::from_source_device("cpu")?,
                                StreamId::new(0),
                            )?,
                            batch * 12,
                            "a".repeat(64),
                        )?,
                        &registry,
                    )?;
                    let bytes = b"retained image attachment".to_vec();
                    let content = InvocationMediaContent::new(
                        "image/png",
                        encode_hex(&Sha256::digest(value.abi_bytes()?)),
                        encode_hex(&Sha256::digest(&bytes)),
                        bytes,
                    );
                    let mut inputs = InvocationInputs::default();
                    inputs.set_present(port_id, vec![value.clone()]);
                    inputs.register_media_batch_item(port_id, 0, 0, content.clone())?;
                    let begin = |inputs| {
                        host.begin_invocation(
                            &manifest,
                            &authorization,
                            node_id,
                            inputs,
                            Arc::new(UnavailablePluginCapabilityServices),
                            CancellationToken::default(),
                        )
                    };
                    let result = begin(inputs.clone());
                    if !bound || namespace != "zed.sim.provider.beeble" {
                        assert!(result.is_err());
                        continue;
                    }
                    let mut invocation = result?;
                    let handle = invocation.take_input(port_id, 0)?;
                    assert_eq!(invocation.read_handle(handle)?, &value);
                    assert_eq!(invocation.media_input_count(handle)?, 1);
                    assert_eq!(invocation.read_media_input(handle)?, content);
                    assert_eq!(invocation.read_media_input_at(handle, 0)?, content);
                    assert!(invocation.read_media_input_at(handle, 1).is_err());
                    let mut extra = inputs.clone();
                    extra.register_media_batch_item(port_id, 0, 1, content.clone())?;
                    assert!(begin(extra).is_err());
                    let mut corrupt = inputs;
                    corrupt
                        .media_batches
                        .get_mut(port_id)
                        .ok_or("missing port")?
                        .get_mut(&0)
                        .ok_or("missing value")?
                        .get_mut(&0)
                        .ok_or("missing media")?
                        .source_value_sha256 = "f".repeat(64);
                    assert!(begin(corrupt).is_err());
                    let mut missing = InvocationInputs::default();
                    missing.set_present(port_id, vec![value]);
                    let mut invocation = begin(missing)?;
                    let handle = invocation.take_input(port_id, 0)?;
                    assert!(invocation.media_input_count(handle).is_err());
                    assert!(invocation.read_media_input(handle).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn bfl_ultra_first_image_admission_is_bound_to_signed_namespace_node_port_and_feature()
    -> Result<(), Box<dyn Error>> {
        use sim_plugin_sdk::{
            ApiRequirement, ApiVersion, CachePolicy, DType, DeterminismPolicy, DeviceId,
            EffectPolicy, ManifestProvenance, ManifestSignature, PLUGIN_SIGNATURE_ALGORITHM,
            PROVIDER_BINDING_API_FEATURE, PluginPort, PluginSigningKey, StreamId, TensorDescriptor,
            TensorValue,
        };
        use sim_runtime::{
            CapabilitySet, PermissionGrant, PermissionPolicy, PluginTrustPolicy,
            PluginVerificationKey,
        };

        let registry = TypeRegistry::built_in()?;
        let native = sim_runtime::generated_native_node_registry_projection(None)?;
        let contract = native
            .provider_binding_contract_sha256(
                "FluxProUltraImageNode",
                "zed:sim-provider-transport@1",
                "zed:sim-provider-materializer@1",
            )?
            .ok_or("missing canonical BFL contract")?;
        let key = PluginSigningKey::new("test.bfl.ultra", b"0123456789abcdef0123456789abcdef")?;
        let trust = PluginTrustPolicy::new([PluginVerificationKey::new(
            "test.bfl.ultra",
            key.verification_key_bytes()?,
        )?])?;
        let host = PluginHost::with_configuration(
            ComponentLimits::default(),
            DEFAULT_API_FEATURES
                .iter()
                .map(|feature| (*feature).to_owned()),
        )?;
        for (node_id, port_id) in [
            ("FluxProUltraImageNode", "image_prompt"),
            ("Flux2ImageNode", "image_prompt"),
            ("FluxProUltraImageNode", "reference_image"),
        ] {
            for (namespace, bound, feature_id) in [
                ("zed.sim.provider.bfl", true, "COMFY-NODE-0188"),
                ("zed.sim.provider.bfl", false, "COMFY-NODE-0188"),
                ("test.other", true, "COMFY-NODE-0188"),
                ("zed.sim.provider.bfl", true, "COMFY-NODE-0174"),
            ] {
                let mut binding = ProviderBindingSet {
                    schema_version: 1,
                    implementation_namespace: namespace.to_owned(),
                    bindings_sha256: "0".repeat(64),
                    bindings: vec![ProviderBindingClaim {
                        feature_id: feature_id.to_owned(),
                        node_id: node_id.to_owned(),
                        contract_sha256: contract.clone(),
                        transport_schema: "zed:sim-provider-transport@1".parse()?,
                        materializer_schema: "zed:sim-provider-materializer@1".parse()?,
                    }],
                };
                binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
                let mut manifest = PluginManifest {
                    schema_version: 1,
                    identifier: namespace.to_owned(),
                    plugin_version: ApiVersion::new(1, 0, 0),
                    api: ApiRequirement {
                        major: 1,
                        minimum_minor: 0,
                        maximum_minor: 0,
                        required_features: vec![PROVIDER_BINDING_API_FEATURE.to_owned()],
                    },
                    digest_sha256: "0".repeat(64),
                    signature: ManifestSignature {
                        algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
                        key_id: "test.bfl.ultra".to_owned(),
                        value: "0".repeat(128),
                    },
                    provenance: ManifestProvenance {
                        source: "fixture://bfl-ultra".to_owned(),
                        publisher: "Test".to_owned(),
                        registry: None,
                    },
                    provider_binding: bound.then_some(binding),
                    nodes: vec![PluginNode {
                        id: node_id.to_owned(),
                        version: ApiVersion::new(1, 0, 0),
                        display_name: "BFL Ultra".to_owned(),
                        category: "test".to_owned(),
                        ports: vec![PluginPort {
                            id: port_id.to_owned(),
                            name: port_id.to_owned(),
                            direction: PortDirection::Input,
                            type_id: registry.resolve("IMAGE")?.clone(),
                            cardinality: PortCardinality::Singular,
                            presence: PortPresence::Required,
                            hidden: false,
                            lazy: false,
                            default: None,
                            serialization: PortSerialization::Handle,
                            accepted_legacy_names: Vec::new(),
                        }],
                        determinism: DeterminismPolicy::External,
                        cache: CachePolicy::Never,
                        effects: EffectPolicy::Provider,
                    }],
                    capabilities: Vec::new(),
                    ui: Vec::new(),
                    routes: Vec::new(),
                    legacy_mappings: Vec::new(),
                };
                manifest.signature.value = key.sign_manifest(&manifest)?;
                let authorization = trust.authorize_manifest(
                    &manifest,
                    &PermissionPolicy::new(
                        "test",
                        [PermissionGrant::new(
                            "test",
                            namespace,
                            CapabilitySet::new([]),
                            "signed-test-fixture",
                        )?],
                    )?,
                )?;
                for batch in [2_u64, 240] {
                    let value = PluginValue::tensor(
                        registry.resolve("IMAGE")?.clone(),
                        TensorValue::new(
                            TensorDescriptor::contiguous(
                                vec![batch, 1, 1, 3],
                                DType::F32,
                                DeviceId::from_source_device("cpu")?,
                                StreamId::new(0),
                            )?,
                            batch * 12,
                            "a".repeat(64),
                        )?,
                        &registry,
                    )?;
                    let bytes = include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
                    ))
                    .to_vec();
                    let content = InvocationMediaContent::new(
                        "image/png",
                        encode_hex(&Sha256::digest(value.abi_bytes()?)),
                        encode_hex(&Sha256::digest(&bytes)),
                        bytes,
                    );
                    let mut inputs = InvocationInputs::default();
                    inputs.set_present(port_id, vec![value.clone()]);
                    inputs.register_media_batch_item(port_id, 0, 0, content.clone())?;
                    let begin = |inputs| {
                        host.begin_invocation(
                            &manifest,
                            &authorization,
                            node_id,
                            inputs,
                            Arc::new(UnavailablePluginCapabilityServices),
                            CancellationToken::default(),
                        )
                    };
                    let result = begin(inputs.clone());
                    if !bound
                        || namespace != "zed.sim.provider.bfl"
                        || node_id != "FluxProUltraImageNode"
                        || port_id != "image_prompt"
                        || feature_id != "COMFY-NODE-0188"
                    {
                        assert!(result.is_err());
                        continue;
                    }
                    let mut invocation = result?;
                    let handle = invocation.take_input(port_id, 0)?;
                    assert_eq!(invocation.read_handle(handle)?, &value);
                    assert_eq!(invocation.media_input_count(handle)?, 1);
                    assert_eq!(invocation.read_media_input(handle)?, content);
                    assert_eq!(invocation.read_media_input_at(handle, 0)?, content);
                    assert!(invocation.read_media_input_at(handle, 1).is_err());
                    let mut extra = inputs.clone();
                    extra.register_media_batch_item(port_id, 0, 1, content.clone())?;
                    assert!(begin(extra).is_err());
                    let mut corrupt = inputs;
                    corrupt
                        .media_batches
                        .get_mut(port_id)
                        .ok_or("missing port")?
                        .get_mut(&0)
                        .ok_or("missing value")?
                        .get_mut(&0)
                        .ok_or("missing media")?
                        .source_value_sha256 = "f".repeat(64);
                    assert!(begin(corrupt).is_err());
                    let mut missing = InvocationInputs::default();
                    missing.set_present(port_id, vec![value]);
                    let mut invocation = begin(missing)?;
                    let handle = invocation.take_input(port_id, 0)?;
                    assert!(invocation.media_input_count(handle).is_err());
                    assert!(invocation.read_media_input(handle).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn hitpaw_first_image_admission_is_bound_to_signed_namespace_node_port_and_feature()
    -> Result<(), Box<dyn Error>> {
        use sim_plugin_sdk::{
            ApiRequirement, ApiVersion, CachePolicy, DType, DeterminismPolicy, DeviceId,
            EffectPolicy, ManifestProvenance, ManifestSignature, PLUGIN_SIGNATURE_ALGORITHM,
            PROVIDER_BINDING_API_FEATURE, PluginPort, PluginSigningKey, StreamId, TensorDescriptor,
            TensorValue,
        };
        use sim_runtime::{
            CapabilitySet, PermissionGrant, PermissionPolicy, PluginTrustPolicy,
            PluginVerificationKey,
        };

        let registry = TypeRegistry::built_in()?;
        let key = PluginSigningKey::new("test.hitpaw", b"0123456789abcdef0123456789abcdef")?;
        let trust = PluginTrustPolicy::new([PluginVerificationKey::new(
            "test.hitpaw",
            key.verification_key_bytes()?,
        )?])?;
        let host = PluginHost::with_configuration(
            ComponentLimits::default(),
            DEFAULT_API_FEATURES
                .iter()
                .map(|feature| (*feature).to_owned()),
        )?;
        for (node_id, port_id) in [
            ("HitPawGeneralImageEnhance", "image"),
            ("HitPawVideoEnhance", "image"),
            ("HitPawGeneralImageEnhance", "reference_image"),
        ] {
            for (namespace, bound, feature_id) in [
                ("zed.sim.provider.hitpaw", true, "COMFY-NODE-0226"),
                ("zed.sim.provider.hitpaw", false, "COMFY-NODE-0226"),
                ("test.other", true, "COMFY-NODE-0226"),
                ("zed.sim.provider.hitpaw", true, "COMFY-NODE-0227"),
            ] {
                let mut binding = ProviderBindingSet {
                    schema_version: 1,
                    implementation_namespace: namespace.to_owned(),
                    bindings_sha256: "0".repeat(64),
                    bindings: vec![ProviderBindingClaim {
                        feature_id: feature_id.to_owned(),
                        node_id: node_id.to_owned(),
                        contract_sha256: "a".repeat(64),
                        transport_schema: "zed:sim-provider-transport@1".parse()?,
                        materializer_schema: "zed:sim-provider-materializer@1".parse()?,
                    }],
                };
                binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
                let mut manifest = PluginManifest {
                    schema_version: 1,
                    identifier: namespace.to_owned(),
                    plugin_version: ApiVersion::new(1, 0, 0),
                    api: ApiRequirement {
                        major: 1,
                        minimum_minor: 0,
                        maximum_minor: 0,
                        required_features: vec![PROVIDER_BINDING_API_FEATURE.to_owned()],
                    },
                    digest_sha256: "0".repeat(64),
                    signature: ManifestSignature {
                        algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
                        key_id: "test.hitpaw".to_owned(),
                        value: "0".repeat(128),
                    },
                    provenance: ManifestProvenance {
                        source: "fixture://hitpaw".to_owned(),
                        publisher: "Test".to_owned(),
                        registry: None,
                    },
                    provider_binding: bound.then_some(binding),
                    nodes: vec![PluginNode {
                        id: node_id.to_owned(),
                        version: ApiVersion::new(1, 0, 0),
                        display_name: "HitPaw".to_owned(),
                        category: "test".to_owned(),
                        ports: vec![PluginPort {
                            id: port_id.to_owned(),
                            name: port_id.to_owned(),
                            direction: PortDirection::Input,
                            type_id: registry.resolve("IMAGE")?.clone(),
                            cardinality: PortCardinality::Singular,
                            presence: PortPresence::Required,
                            hidden: false,
                            lazy: false,
                            default: None,
                            serialization: PortSerialization::Handle,
                            accepted_legacy_names: Vec::new(),
                        }],
                        determinism: DeterminismPolicy::External,
                        cache: CachePolicy::Never,
                        effects: EffectPolicy::Provider,
                    }],
                    capabilities: Vec::new(),
                    ui: Vec::new(),
                    routes: Vec::new(),
                    legacy_mappings: Vec::new(),
                };
                manifest.signature.value = key.sign_manifest(&manifest)?;
                let authorization = trust.authorize_manifest(
                    &manifest,
                    &PermissionPolicy::new(
                        "test",
                        [PermissionGrant::new(
                            "test",
                            namespace,
                            CapabilitySet::new([]),
                            "signed-test-fixture",
                        )?],
                    )?,
                )?;
                for batch in [2_u64, 240] {
                    let value = PluginValue::tensor(
                        registry.resolve("IMAGE")?.clone(),
                        TensorValue::new(
                            TensorDescriptor::contiguous(
                                vec![batch, 1, 1, 3],
                                DType::F32,
                                DeviceId::from_source_device("cpu")?,
                                StreamId::new(0),
                            )?,
                            batch * 12,
                            "a".repeat(64),
                        )?,
                        &registry,
                    )?;
                    let bytes = include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../sim_test_support/fixtures/providers/hitpaw/downloaded-image.png"
                    ))
                    .to_vec();
                    let content = InvocationMediaContent::new(
                        "image/png",
                        encode_hex(&Sha256::digest(value.abi_bytes()?)),
                        encode_hex(&Sha256::digest(&bytes)),
                        bytes,
                    );
                    let mut inputs = InvocationInputs::default();
                    inputs.set_present(port_id, vec![value.clone()]);
                    inputs.register_media_batch_item(port_id, 0, 0, content.clone())?;
                    let begin = |inputs| {
                        host.begin_invocation(
                            &manifest,
                            &authorization,
                            node_id,
                            inputs,
                            Arc::new(UnavailablePluginCapabilityServices),
                            CancellationToken::default(),
                        )
                    };
                    let result = begin(inputs.clone());
                    if !bound
                        || namespace != "zed.sim.provider.hitpaw"
                        || node_id != "HitPawGeneralImageEnhance"
                        || port_id != "image"
                        || feature_id != "COMFY-NODE-0226"
                    {
                        assert!(result.is_err());
                        continue;
                    }
                    let mut invocation = result?;
                    let handle = invocation.take_input(port_id, 0)?;
                    assert_eq!(invocation.read_handle(handle)?, &value);
                    assert_eq!(invocation.media_input_count(handle)?, 1);
                    assert_eq!(invocation.read_media_input(handle)?, content);
                    assert_eq!(invocation.read_media_input_at(handle, 0)?, content);
                    assert!(invocation.read_media_input_at(handle, 1).is_err());
                    let mut extra = inputs.clone();
                    extra.register_media_batch_item(port_id, 0, 1, content.clone())?;
                    assert!(begin(extra).is_err());
                    let mut corrupt = inputs;
                    corrupt
                        .media_batches
                        .get_mut(port_id)
                        .ok_or("missing port")?
                        .get_mut(&0)
                        .ok_or("missing value")?
                        .get_mut(&0)
                        .ok_or("missing media")?
                        .source_value_sha256 = "f".repeat(64);
                    assert!(begin(corrupt).is_err());
                    let mut missing = InvocationInputs::default();
                    missing.set_present(port_id, vec![value]);
                    let mut invocation = begin(missing)?;
                    let handle = invocation.take_input(port_id, 0)?;
                    assert!(invocation.media_input_count(handle).is_err());
                    assert!(invocation.read_media_input(handle).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn provider_v2_retry_delay_parses_delta_and_http_date_with_one_cap() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        assert_eq!(resolve_retry_delay(503, 17, None, now), Ok(17));
        assert_eq!(resolve_retry_delay(503, 17, Some("2"), now), Ok(2_000));
        assert_eq!(resolve_retry_delay(429, 17, Some("2"), now), Ok(2_000));
        let date = httpdate::fmt_http_date(now + Duration::from_secs(3));
        assert_eq!(resolve_retry_delay(429, 17, Some(&date), now), Ok(3_000));
        let past_date = httpdate::fmt_http_date(now - Duration::from_secs(3));
        assert_eq!(resolve_retry_delay(503, 17, Some(&past_date), now), Ok(0));
        assert_eq!(resolve_retry_delay(429, 17, None, now), Ok(17));
        assert_eq!(
            resolve_retry_delay(429, 17, Some("151"), now),
            Ok(MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS)
        );
        assert_eq!(
            resolve_retry_delay(429, 17, Some("not-a-date"), now),
            Ok(17)
        );
        assert_eq!(
            resolve_retry_delay(503, 17, Some("999999999999999999999999999"), now),
            Ok(MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS)
        );
        assert_eq!(
            resolve_retry_delay(
                503,
                MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS + 1,
                None,
                now,
            ),
            Err(WorkerProviderStreamError::WaitLimit)
        );
    }

    fn provider_context() -> WorkerProviderInvocationContext {
        WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x421),
            session_generation: 7,
            invocation: 11,
            generation: 13,
        }
    }

    fn provider_handle(slot: u32) -> WorkerProviderStreamHandle {
        let context = provider_context();
        WorkerProviderStreamHandle {
            session_id: context.session_id,
            session_generation: context.session_generation,
            invocation: context.invocation,
            slot,
            generation: context.generation,
        }
    }

    fn check_cancelled_request(slot: u32) -> WorkerProviderStreamRequest {
        WorkerProviderStreamRequest::CheckCancelled(provider_handle(slot))
    }

    fn provider_head() -> WorkerProviderRequestHead {
        WorkerProviderRequestHead {
            endpoint: "https://provider.invalid/v2".to_owned(),
            secret_id: Some("provider-secret".to_owned()),
            method: WorkerProviderHttpMethod::Post,
            headers: vec![WorkerProviderHeader {
                name: "content-type".to_owned(),
                value: "application/json".to_owned(),
            }],
            declared_body_bytes: Some(3),
        }
    }

    fn provider_contract() -> WorkerProviderStreamingContract {
        WorkerProviderStreamingContract {
            methods: vec![WorkerProviderHttpMethod::Post],
            maximum_headers: 4,
            maximum_header_bytes: 1024,
            maximum_request_body_bytes: 1024,
            maximum_response_body_bytes: 1024,
            maximum_chunk_bytes: 128,
            maximum_ndjson_line_bytes: 128,
            maximum_wait_milliseconds: 1000,
            maximum_uploads: 1,
            maximum_upload_body_bytes: 1024,
            maximum_cost_requests: 1,
            maximum_progress_total: 100,
            uploads: true,
            cost_requests: true,
        }
    }

    fn started_provider_runtime(
        cancellation: CancellationToken,
    ) -> Result<(ProviderV2RuntimeHost, ProviderV2StreamRouteReceiver), WorkerProviderStreamError>
    {
        let (mut runtime, receiver) = checked_provider_runtime(cancellation)?;
        runtime.validator.validate_request(
            1,
            &WorkerProviderStreamRequest::StartRequest {
                context: provider_context(),
                head: provider_head(),
            },
        )?;
        runtime.validator.validate_response(
            1,
            &WorkerProviderStreamResponse::Stream(Ok(provider_handle(1))),
        )?;
        runtime.next_call_id = 2;
        runtime.bound = true;
        Ok((runtime, receiver))
    }

    fn checked_provider_runtime(
        cancellation: CancellationToken,
    ) -> Result<(ProviderV2RuntimeHost, ProviderV2StreamRouteReceiver), WorkerProviderStreamError>
    {
        let (route, receiver) = provider_v2_stream_route();
        let runtime = ProviderV2RuntimeHost::checked_from_certified_capsule(
            provider_context(),
            provider_contract(),
            cancellation,
            route,
        )?;
        Ok((runtime, receiver))
    }

    fn admit_provider_retry_head(
        runtime: &mut ProviderV2RuntimeHost,
        status: u16,
        headers: Vec<WorkerProviderHeader>,
    ) -> Result<(), WorkerProviderStreamError> {
        runtime.validator.validate_request(
            2,
            &WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                handle: provider_handle(1),
                sequence: 0,
                bytes: b"abc".to_vec(),
                end: true,
            }),
        )?;
        runtime
            .validator
            .validate_response(2, &WorkerProviderStreamResponse::Unit(Ok(())))?;
        runtime.validator.validate_request(
            3,
            &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                handle: provider_handle(1),
                after_sequence: None,
                timeout_milliseconds: 10,
            }),
        )?;
        runtime.validator.validate_response(
            3,
            &WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: provider_handle(1),
                    sequence: 0,
                    event: WorkerProviderResponseFrameEvent::Head(
                        sim_types::WorkerProviderResponseHead { status, headers },
                    ),
                },
            ))),
        )
    }

    #[test]
    fn provider_v2_retry_uses_the_retained_head_and_cancels_before_a_fresh_handle()
    -> Result<(), WorkerProviderStreamError> {
        let (mut rate_limited, _receiver) = started_provider_runtime(CancellationToken::default())?;
        admit_provider_retry_head(
            &mut rate_limited,
            503,
            vec![WorkerProviderHeader {
                name: "Retry-After".to_owned(),
                value: "120".to_owned(),
            }],
        )?;
        let request =
            rate_limited.checked_retry_request(provider_handle(1), 1, SystemTime::UNIX_EPOCH)?;
        assert_eq!(request.delay_milliseconds, 120_000);

        let (mut duplicated, _receiver) = started_provider_runtime(CancellationToken::default())?;
        assert_eq!(
            admit_provider_retry_head(
                &mut duplicated,
                429,
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
            ),
            Err(WorkerProviderStreamError::InvalidHeaders)
        );

        let cancellation = CancellationToken::default();
        let (mut cancelled, _receiver) = started_provider_runtime(cancellation.clone())?;
        admit_provider_retry_head(&mut cancelled, 503, Vec::new())?;
        let request =
            cancelled.checked_retry_request(provider_handle(1), 50, SystemTime::UNIX_EPOCH)?;
        let cancel = thread::spawn(move || {
            thread::sleep(Duration::from_millis(2));
            cancellation.cancel();
        });
        assert_eq!(
            cancelled.wait_retry_delay(&request),
            Err(WorkerProviderStreamError::Cancelled)
        );
        cancel
            .join()
            .map_err(|_| WorkerProviderStreamError::HostFailure)?;
        assert_eq!(
            cancelled.validator.primary_handle(),
            Some(provider_handle(1))
        );
        Ok(())
    }

    #[test]
    fn provider_v2_request_head_projection_preserves_every_signed_field() {
        use provider_v2_wit_contract::zed::sim_provider_plugin::types::{
            Header, HttpMethod, RequestHead,
        };

        for (wit_method, worker_method) in [
            (HttpMethod::Delete, WorkerProviderHttpMethod::Delete),
            (HttpMethod::Get, WorkerProviderHttpMethod::Get),
            (HttpMethod::Head, WorkerProviderHttpMethod::Head),
            (HttpMethod::Options, WorkerProviderHttpMethod::Options),
            (HttpMethod::Patch, WorkerProviderHttpMethod::Patch),
            (HttpMethod::Post, WorkerProviderHttpMethod::Post),
            (HttpMethod::Put, WorkerProviderHttpMethod::Put),
        ] {
            let projected = worker_provider_request_head(RequestHead {
                endpoint: "https://provider.invalid/v2".to_owned(),
                secret_id: Some("provider-secret".to_owned()),
                method: wit_method,
                headers: vec![
                    Header {
                        name: "x-first".to_owned(),
                        value: "1".to_owned(),
                    },
                    Header {
                        name: "x-second".to_owned(),
                        value: "2".to_owned(),
                    },
                ],
                declared_body_bytes: Some(41),
            });
            assert_eq!(projected.endpoint, "https://provider.invalid/v2");
            assert_eq!(projected.secret_id.as_deref(), Some("provider-secret"));
            assert_eq!(projected.method, worker_method);
            assert_eq!(
                projected.headers,
                vec![
                    WorkerProviderHeader {
                        name: "x-first".to_owned(),
                        value: "1".to_owned(),
                    },
                    WorkerProviderHeader {
                        name: "x-second".to_owned(),
                        value: "2".to_owned(),
                    },
                ]
            );
            assert_eq!(projected.declared_body_bytes, Some(41));
        }

        let without_secret = worker_provider_request_head(RequestHead {
            endpoint: "https://provider.invalid/no-secret".to_owned(),
            secret_id: None,
            method: HttpMethod::Get,
            headers: Vec::new(),
            declared_body_bytes: None,
        });
        assert_eq!(without_secret.secret_id, None);
        assert_eq!(without_secret.declared_body_bytes, None);
    }

    #[test]
    fn provider_v2_route_is_capacity_one_and_revocation_discards_queued_work() {
        let (route, receiver) = provider_v2_stream_route();
        let (first_reply, _first_receiver) = sync_channel(1);
        route
            .sender
            .try_send(ProviderV2StreamRouteMessage::Request(
                ProviderV2StreamRouteCall {
                    call_id: 1,
                    request: check_cancelled_request(1),
                    reply: first_reply,
                },
            ))
            .expect("first capacity-one request must be admitted");
        let (second_reply, _second_receiver) = sync_channel(1);
        assert!(matches!(
            route.sender.try_send(ProviderV2StreamRouteMessage::Request(
                ProviderV2StreamRouteCall {
                    call_id: 2,
                    request: check_cancelled_request(1),
                    reply: second_reply,
                },
            )),
            Err(TrySendError::Full(_))
        ));

        route.revoked.store(true, Ordering::Release);
        let (revoke_reply, _revoke_receiver) = sync_channel(1);
        route
            .revoke_sender
            .try_send(ProviderV2StreamRouteMessage::Revoke {
                reply: revoke_reply,
            })
            .expect("out-of-band revocation must not share request capacity");
        assert!(matches!(
            receiver.try_receive(),
            Ok(ProviderV2StreamRouteMessage::Revoke { .. })
        ));
        assert!(matches!(
            receiver.receiver.try_recv(),
            Err(TryRecvError::Empty)
        ));
    }

    #[test]
    fn provider_v2_route_carries_every_typed_request_and_response_without_erasure()
    -> Result<(), Box<dyn Error>> {
        let requests = vec![
            WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                handle: provider_handle(1),
                sequence: 0,
                bytes: vec![1, 2, 3],
                end: true,
            }),
            WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                handle: provider_handle(1),
                after_sequence: None,
                timeout_milliseconds: 10,
            }),
            WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                handle: provider_handle(1),
                port_id: "reference".to_owned(),
                media_type: "application/octet-stream".to_owned(),
                byte_length: 3,
                content_sha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
                    .to_owned(),
            }),
            WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                handle: provider_handle(2),
                sequence: 0,
                bytes: vec![1, 2, 3],
                end: true,
            }),
            WorkerProviderStreamRequest::CompleteUpload(WorkerProviderUploadCompletionRequest {
                handle: provider_handle(2),
            }),
            WorkerProviderStreamRequest::RetryRequest(WorkerProviderRetryRequest {
                handle: provider_handle(1),
                delay_milliseconds: 1,
            }),
            WorkerProviderStreamRequest::RequestCost(WorkerProviderCostRequest {
                handle: provider_handle(1),
                operation: "fixture".to_owned(),
                currency: "USD".to_owned(),
                maximum_microunits: 1000,
            }),
            WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
                handle: provider_handle(1),
                sequence: 1,
                completed: 1,
                total: 1,
                message: Some("complete".to_owned()),
            }),
            check_cancelled_request(1),
        ];
        let responses = vec![
            WorkerProviderStreamResponse::Unit(Ok(())),
            WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::TimedOut)),
            WorkerProviderStreamResponse::Stream(Ok(provider_handle(2))),
            WorkerProviderStreamResponse::Unit(Ok(())),
            WorkerProviderStreamResponse::UploadCompletion(Ok(
                sim_types::WorkerProviderUploadCompletion {
                    download_url: "https://fixture.invalid/storage/reference.bin".to_owned(),
                },
            )),
            WorkerProviderStreamResponse::Stream(Ok(provider_handle(3))),
            WorkerProviderStreamResponse::Cost(Ok(sim_types::WorkerProviderCostResponse {
                accepted: true,
                approved_microunits: 900,
                receipt: vec![9],
            })),
            WorkerProviderStreamResponse::Unit(Ok(())),
            WorkerProviderStreamResponse::Unit(Ok(())),
        ];
        for (index, (request, response)) in requests.into_iter().zip(responses).enumerate() {
            let (route, receiver) = provider_v2_stream_route();
            let (reply, reply_receiver) = sync_channel(1);
            let call_id = u64::try_from(index).expect("fixture index fits") + 1;
            route
                .sender
                .try_send(ProviderV2StreamRouteMessage::Request(
                    ProviderV2StreamRouteCall {
                        call_id,
                        request: request.clone(),
                        reply,
                    },
                ))
                .expect("typed route accepts one request");
            let ProviderV2StreamRouteMessage::Request(call) =
                receiver.try_receive().expect("typed request is available")
            else {
                return Err("request route yielded a control message".into());
            };
            assert_eq!(call.call_id(), call_id);
            assert_eq!(call.request(), &request);
            call.respond(response.clone())
                .expect("typed response is delivered once");
            assert_eq!(
                reply_receiver
                    .recv()
                    .expect("typed response remains connected"),
                response
            );
        }

        let (route, receiver) = provider_v2_stream_route();
        let (reply, reply_receiver) = sync_channel(1);
        let start = WorkerProviderStreamRequest::StartRequest {
            context: provider_context(),
            head: provider_head(),
        };
        route
            .sender
            .try_send(ProviderV2StreamRouteMessage::Request(
                ProviderV2StreamRouteCall {
                    call_id: 1,
                    request: start.clone(),
                    reply,
                },
            ))
            .expect("start request is admitted");
        let ProviderV2StreamRouteMessage::Request(call) =
            receiver.try_receive().expect("start request is available")
        else {
            return Err("start route yielded a control message".into());
        };
        assert_eq!(call.request(), &start);
        let (call_id, context, head, reply) =
            call.into_start().expect("exact start DTO is retained");
        assert_eq!(call_id, 1);
        assert_eq!(context, provider_context());
        assert_eq!(head, provider_head());
        ProviderV2BoundStartCall { call_id, reply }
            .respond(Ok(provider_handle(1)))
            .expect("bound start response is delivered once");
        assert_eq!(
            reply_receiver
                .recv()
                .expect("start response remains connected"),
            WorkerProviderStreamResponse::Stream(Ok(provider_handle(1)))
        );
        Ok(())
    }

    #[test]
    fn provider_v2_validator_rejects_every_exposed_handle_identity_mutation() {
        let mut validator = WorkerProviderStreamTransportValidator::checked_for_host_session(
            provider_context(),
            provider_contract(),
            CancellationToken::default(),
        )
        .expect("fixture streaming contract is valid");
        validator
            .validate_request(
                1,
                &WorkerProviderStreamRequest::StartRequest {
                    context: provider_context(),
                    head: provider_head(),
                },
            )
            .expect("exact start is valid");
        validator
            .validate_response(
                1,
                &WorkerProviderStreamResponse::Stream(Ok(provider_handle(1))),
            )
            .expect("exact start response is valid");

        let mut mutations = Vec::new();
        let mut foreign_session = provider_handle(1);
        foreign_session.session_id = Uuid::from_u128(0x999);
        mutations.push((foreign_session, WorkerProviderStreamError::ForeignSession));
        let mut stale_session = provider_handle(1);
        stale_session.session_generation += 1;
        mutations.push((stale_session, WorkerProviderStreamError::StaleSession));
        let mut foreign_invocation = provider_handle(1);
        foreign_invocation.invocation += 1;
        mutations.push((
            foreign_invocation,
            WorkerProviderStreamError::ForeignInvocation,
        ));
        let mut stale_generation = provider_handle(1);
        stale_generation.generation += 1;
        mutations.push((stale_generation, WorkerProviderStreamError::StaleGeneration));
        let mut foreign_slot = provider_handle(1);
        foreign_slot.slot += 1;
        mutations.push((foreign_slot, WorkerProviderStreamError::ForeignHandle));
        for (handle, error) in mutations {
            assert_eq!(
                validator
                    .validate_request(2, &WorkerProviderStreamRequest::CheckCancelled(handle),),
                Err(error)
            );
        }
    }

    #[test]
    fn provider_v2_inputs_are_denied_until_the_certified_start_is_bound()
    -> Result<(), WorkerProviderStreamError> {
        let (mut runtime, _receiver) = started_provider_runtime(CancellationToken::default())?;
        runtime.bound = false;
        assert_eq!(
            runtime.require_bound(),
            Err(WorkerProviderStreamError::InvalidRequestAuthority)
        );
        runtime.bound = true;
        assert_eq!(runtime.require_bound(), Ok(()));
        Ok(())
    }

    #[test]
    fn provider_v2_secret_presence_is_query_bound_and_identifier_exact()
    -> Result<(), WorkerProviderStreamError> {
        let (mut runtime, _receiver) = started_provider_runtime(CancellationToken::default())?;
        runtime.begin_secret_query()?;
        runtime.record_secret_query_result("secret-a".to_owned(), false);
        assert_eq!(
            runtime.require_present_secret(Some("secret-a")),
            Err(WorkerProviderStreamError::InvalidRequestAuthority)
        );
        runtime.begin_secret_query()?;
        runtime.record_secret_query_result("secret-a".to_owned(), true);
        assert_eq!(runtime.require_present_secret(Some("secret-a")), Ok(()));
        runtime.begin_secret_query()?;
        runtime.record_secret_query_result("secret-a".to_owned(), false);
        assert_eq!(
            runtime.require_present_secret(Some("secret-a")),
            Err(WorkerProviderStreamError::InvalidRequestAuthority)
        );
        runtime.begin_secret_query()?;
        runtime.record_secret_query_result("secret-a".to_owned(), true);
        assert_eq!(
            runtime.require_present_secret(Some("secret-b")),
            Err(WorkerProviderStreamError::InvalidRequestAuthority)
        );
        assert_eq!(runtime.require_present_secret(None), Ok(()));
        for _ in runtime.secret_query_count..MAX_PROVIDER_V2_SECRET_QUERIES {
            runtime.begin_secret_query()?;
        }
        assert_eq!(
            runtime.begin_secret_query(),
            Err(WorkerProviderStreamError::InvalidRequestAuthority)
        );
        assert_eq!(runtime.secret_query_count, MAX_PROVIDER_V2_SECRET_QUERIES);
        assert_eq!(runtime.require_present_secret(Some("secret-a")), Ok(()));
        Ok(())
    }

    #[test]
    fn provider_v2_wit_secret_query_preserves_cancelled_runtime_without_querying_capabilities()
    -> Result<(), WorkerProviderStreamError> {
        use provider_v2_wit_contract::zed::sim_provider_plugin::invocation_input_host::Host;

        for interrupted in [false, true] {
            let cancellation = CancellationToken::default();
            let (mut runtime, receiver) = checked_provider_runtime(cancellation.clone())?;
            if interrupted {
                assert!(cancellation.cancel());
            } else {
                runtime.secret_query_count = MAX_PROVIDER_V2_SECRET_QUERIES;
            }
            let query_count = runtime.secret_query_count;
            let mut state = WasmStoreState {
                limits: StoreLimitsBuilder::new().build(),
                invocation: None,
                provider_runtime: Some(runtime),
            };
            let error = Host::secret_exists(&mut state, "sim_org.auth_token".to_owned());
            if interrupted {
                assert!(matches!(error, Err(WitInvocationError::Cancelled)));
            } else {
                assert!(
                    matches!(error, Err(WitInvocationError::HostFailure(message))
                    if message == sanitize_diagnostic(&WorkerProviderStreamError::InvalidRequestAuthority.to_string()))
                );
            }
            let runtime = state
                .provider_runtime
                .as_ref()
                .ok_or(WorkerProviderStreamError::HostFailure)?;
            assert_eq!(runtime.secret_query_count, query_count);
            assert!(runtime.present_secrets.is_empty());
            assert!(!runtime.bound);
            assert!(state.invocation.is_none());
            assert!(matches!(
                receiver.receiver.try_recv(),
                Err(TryRecvError::Empty)
            ));
        }
        Ok(())
    }

    #[test]
    fn provider_v2_non_start_operations_validate_before_route_enqueue()
    -> Result<(), WorkerProviderStreamError> {
        let requests = vec![
            WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                handle: provider_handle(1),
                sequence: 0,
                bytes: vec![1],
                end: true,
            }),
            WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                handle: provider_handle(1),
                after_sequence: None,
                timeout_milliseconds: 1,
            }),
            WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                handle: provider_handle(1),
                port_id: "image".to_owned(),
                media_type: "application/octet-stream".to_owned(),
                byte_length: 1,
                content_sha256: "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7c24e9bd663e6c"
                    .to_owned(),
            }),
            WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                handle: provider_handle(2),
                sequence: 0,
                bytes: vec![1],
                end: true,
            }),
            WorkerProviderStreamRequest::RequestCost(WorkerProviderCostRequest {
                handle: provider_handle(1),
                operation: "fixture".to_owned(),
                currency: "USD".to_owned(),
                maximum_microunits: 1,
            }),
            WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
                handle: provider_handle(1),
                sequence: 0,
                completed: 0,
                total: 1,
                message: None,
            }),
            WorkerProviderStreamRequest::CheckCancelled(provider_handle(1)),
        ];
        for request in requests {
            let (mut runtime, receiver) = checked_provider_runtime(CancellationToken::default())?;
            assert_eq!(
                runtime.exchange(request),
                Err(WorkerProviderStreamError::ForeignHandle)
            );
            assert!(runtime.route.revoked.load(Ordering::Acquire));
            assert!(matches!(
                receiver.receiver.try_recv(),
                Err(TryRecvError::Empty)
            ));
            assert!(matches!(
                receiver.try_receive(),
                Ok(ProviderV2StreamRouteMessage::Revoke { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn provider_v2_wit_start_context_mismatch_revokes_before_route_enqueue()
    -> Result<(), WorkerProviderStreamError> {
        use provider_v2_wit_contract::zed::sim_provider_plugin::provider_streaming_host::Host;
        use provider_v2_wit_contract::zed::sim_provider_plugin::types::{
            Header, HttpMethod, InvocationContext, RequestHead, StreamError,
        };

        for context in [
            InvocationContext {
                invocation: provider_context().invocation + 1,
                generation: provider_context().generation,
            },
            InvocationContext {
                invocation: provider_context().invocation,
                generation: provider_context().generation + 1,
            },
        ] {
            let (runtime, receiver) = checked_provider_runtime(CancellationToken::default())?;
            let mut state = WasmStoreState {
                limits: StoreLimitsBuilder::new().build(),
                invocation: None,
                provider_runtime: Some(runtime),
            };
            assert!(matches!(
                Host::start_request(
                    &mut state,
                    context,
                    RequestHead {
                        endpoint: "https://provider.invalid/v2".to_owned(),
                        secret_id: Some("provider-secret".to_owned()),
                        method: HttpMethod::Post,
                        headers: vec![Header {
                            name: "content-type".to_owned(),
                            value: "application/json".to_owned(),
                        }],
                        declared_body_bytes: Some(3),
                    },
                ),
                Err(StreamError::InvalidRequestAuthority)
            ));
            let runtime = state
                .provider_runtime
                .as_ref()
                .expect("test runtime remains retained after rejection");
            assert!(runtime.route.revoked.load(Ordering::Acquire));
            assert!(matches!(
                receiver.receiver.try_recv(),
                Err(TryRecvError::Empty)
            ));
            assert!(matches!(
                receiver.try_receive(),
                Ok(ProviderV2StreamRouteMessage::Revoke { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn provider_v2_full_disconnected_and_cancelled_routes_preserve_primary_errors()
    -> Result<(), WorkerProviderStreamError> {
        let (mut full_runtime, full_receiver) =
            started_provider_runtime(CancellationToken::default())?;
        let (reply, _reply_receiver) = sync_channel(1);
        full_runtime
            .route
            .sender
            .try_send(ProviderV2StreamRouteMessage::Request(
                ProviderV2StreamRouteCall {
                    call_id: 99,
                    request: check_cancelled_request(1),
                    reply,
                },
            ))
            .map_err(|_| WorkerProviderStreamError::HostFailure)?;
        assert_eq!(
            full_runtime.exchange(check_cancelled_request(1)),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        assert!(full_runtime.route.revoked.load(Ordering::Acquire));
        drop(full_receiver);

        let (mut disconnected_runtime, disconnected_receiver) =
            started_provider_runtime(CancellationToken::default())?;
        assert_eq!(
            disconnected_runtime.route_disconnection_error(),
            WorkerProviderStreamError::RevokedHandle
        );
        drop(disconnected_receiver);
        assert_eq!(
            disconnected_runtime.exchange(check_cancelled_request(1)),
            Err(WorkerProviderStreamError::RevokedHandle)
        );
        assert!(disconnected_runtime.route.revoked.load(Ordering::Acquire));

        let cancellation = CancellationToken::default();
        let (mut cancelled_runtime, cancelled_receiver) =
            started_provider_runtime(cancellation.clone())?;
        assert!(cancellation.cancel());
        assert_eq!(
            cancelled_runtime.route_disconnection_error(),
            WorkerProviderStreamError::Cancelled
        );
        assert_eq!(
            cancelled_runtime.exchange(check_cancelled_request(1)),
            Err(WorkerProviderStreamError::Cancelled)
        );
        assert!(cancelled_runtime.route.revoked.load(Ordering::Acquire));
        drop(cancelled_receiver);

        let blocked_cancellation = CancellationToken::default();
        let (mut blocked_runtime, blocked_receiver) =
            started_provider_runtime(blocked_cancellation.clone())?;
        let blocked_exchange =
            std::thread::spawn(move || blocked_runtime.exchange(check_cancelled_request(1)));
        let blocked_call = receive_provider_route_request(&blocked_receiver)?;
        assert!(blocked_cancellation.cancel());
        drop(blocked_call);
        drop(blocked_receiver);
        assert_eq!(
            blocked_exchange
                .join()
                .expect("blocked provider exchange thread remains available"),
            Err(WorkerProviderStreamError::Cancelled)
        );
        Ok(())
    }

    fn provider_http_scope() -> ProviderRuntimeHttpInvocationIdentity {
        ProviderRuntimeHttpInvocationIdentity {
            context: provider_context(),
            profile_id: Uuid::from_u128(1).to_string(),
            principal_id: "fixture-principal".to_owned(),
            prompt_id: Uuid::from_u128(2).to_string(),
            prompt_sha256: "a".repeat(64),
            attempt_id: Uuid::from_u128(3).to_string(),
            node_id: "node.fixture".to_owned(),
            compiled_plan_sha256: "b".repeat(64),
            activation_request_ordinal: 5,
            registry_generation: 7,
            registry_digest_sha256: "c".repeat(64),
            component_generation: 11,
            component_digest_sha256: "d".repeat(64),
            authorization_generation_sha256: "e".repeat(64),
            binding_generation: 13,
            binding_set_sha256: "f".repeat(64),
            provider: "plugin.fixture".to_owned(),
            provider_manifest_sha256: "1".repeat(64),
        }
    }

    fn checked_http_provider_runtime(
        maximum_requests: u32,
        cancellation: CancellationToken,
        verifier: ProviderV2HttpContinuationVerifier,
    ) -> Result<(ProviderV2RuntimeHost, ProviderV2StreamRouteReceiver), WorkerProviderStreamError>
    {
        let (route, receiver) = provider_v2_stream_route();
        let mut contract = provider_contract();
        contract.methods.push(WorkerProviderHttpMethod::Get);
        contract.methods.sort();
        let mut runtime = ProviderV2RuntimeHost::checked_from_certified_capsule(
            provider_context(),
            contract,
            cancellation,
            route,
        )?;
        runtime.bind_checked_http_continuations(
            provider_http_scope(),
            maximum_requests,
            verifier,
        )?;
        Ok((runtime, receiver))
    }

    fn spawn_checked_http_route(
        receiver: ProviderV2StreamRouteReceiver,
    ) -> thread::JoinHandle<Result<(), WorkerProviderStreamError>> {
        thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut ordinal = 0;
            loop {
                if Instant::now() >= deadline {
                    return Err(WorkerProviderStreamError::TimedOut);
                }
                match receiver.try_receive() {
                    Ok(ProviderV2StreamRouteMessage::Request(call)) => {
                        if matches!(
                            &call.request,
                            WorkerProviderStreamRequest::StartRequest { .. }
                        ) {
                            let (call_id, context, _head, reply) = call.into_start()?;
                            if context != provider_context() {
                                return Err(WorkerProviderStreamError::ForeignInvocation);
                            }
                            ordinal += 1;
                            ProviderV2BoundStartCall { call_id, reply }
                                .respond(Ok(provider_handle(ordinal)))?;
                            continue;
                        }
                        let response = match &call.request {
                            WorkerProviderStreamRequest::WriteRequestChunk(_)
                            | WorkerProviderStreamRequest::ReportProgress(_) => {
                                WorkerProviderStreamResponse::Unit(Ok(()))
                            }
                            WorkerProviderStreamRequest::WaitResponse(request) => {
                                let (sequence, event) = if request.after_sequence.is_none() {
                                    (
                                        0,
                                        WorkerProviderResponseFrameEvent::Head(
                                            sim_types::WorkerProviderResponseHead {
                                                status: 200,
                                                headers: Vec::new(),
                                            },
                                        ),
                                    )
                                } else {
                                    (
                                        1,
                                        WorkerProviderResponseFrameEvent::Terminal(
                                            WorkerProviderTerminal::Completed(
                                                b"verified-http-terminal".to_vec(),
                                            ),
                                        ),
                                    )
                                };
                                WorkerProviderStreamResponse::Wait(Ok(
                                    WorkerProviderWaitOutcome::Frame(WorkerProviderResponseFrame {
                                        handle: request.handle,
                                        sequence,
                                        event,
                                    }),
                                ))
                            }
                            _ => return Err(WorkerProviderStreamError::InvalidOrder),
                        };
                        call.respond(response)?;
                    }
                    Ok(ProviderV2StreamRouteMessage::Revoke { reply }) => {
                        receiver.mark_revoked();
                        reply
                            .try_send(Ok(()))
                            .map_err(|_| WorkerProviderStreamError::HostFailure)?;
                        return Ok(());
                    }
                    Err(TryRecvError::Empty) => thread::yield_now(),
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            }
        })
    }

    fn complete_checked_http_host_request(
        runtime: &mut ProviderV2RuntimeHost,
        handle: WorkerProviderStreamHandle,
        body: Vec<u8>,
    ) -> Result<(), WorkerProviderStreamError> {
        runtime.exchange(WorkerProviderStreamRequest::WriteRequestChunk(
            WorkerProviderRequestChunk {
                handle,
                sequence: 0,
                bytes: body,
                end: true,
            },
        ))?;
        for after_sequence in [None, Some(0)] {
            runtime.exchange(WorkerProviderStreamRequest::WaitResponse(
                WorkerProviderWaitRequest {
                    handle,
                    after_sequence,
                    timeout_milliseconds: 10,
                },
            ))?;
        }
        Ok(())
    }

    #[test]
    fn provider_v2_http_lifecycle_checks_retained_scope_and_preserves_intermediate_progress()
    -> Result<(), Box<dyn Error>> {
        let checked_count = Arc::new(AtomicU64::new(0));
        let count_for_verifier = checked_count.clone();
        let expected_scope = provider_http_scope();
        let expected_binding = http_invocation_binding_sha256(&expected_scope)?;
        let (mut runtime, receiver) = checked_http_provider_runtime(
            3,
            CancellationToken::default(),
            Box::new(move |scope, challenge| {
                assert_eq!(scope, &expected_scope);
                assert_eq!(challenge.context(), &expected_scope.context);
                assert_eq!(challenge.certified_binding_sha256(), &expected_binding);
                let ordinal = count_for_verifier.fetch_add(1, Ordering::AcqRel) + 2;
                assert_eq!(u64::from(challenge.ordinal()), ordinal);
                assert_eq!(u64::from(challenge.predecessor_handle().slot), ordinal - 1);
                assert_eq!(
                    challenge.predecessor_receipt_sha256(),
                    "0279fa674c7f6e8aedd6f3d0390a478f61b326b2a9282ab246fbb16633f64678"
                );
                assert_eq!(
                    challenge.request_head().method,
                    WorkerProviderHttpMethod::Get
                );
                Ok(())
            }),
        )?;
        let worker = spawn_checked_http_route(receiver);
        for ordinal in 1..=3 {
            let mut head = provider_head();
            let body = if ordinal == 1 {
                b"abc".to_vec()
            } else {
                head.method = WorkerProviderHttpMethod::Get;
                head.declared_body_bytes = Some(0);
                Vec::new()
            };
            let response = runtime.exchange(WorkerProviderStreamRequest::StartRequest {
                context: provider_context(),
                head,
            })?;
            let WorkerProviderStreamResponse::Stream(Ok(handle)) = response else {
                return Err("start response did not supply handle".into());
            };
            assert_eq!(handle, provider_handle(ordinal));
            assert!(!runtime.stream_terminal);
            complete_checked_http_host_request(&mut runtime, handle, body)?;
            runtime.exchange(WorkerProviderStreamRequest::ReportProgress(
                WorkerProviderProgress {
                    handle,
                    sequence: u64::from(ordinal - 1),
                    completed: u64::from(ordinal),
                    total: 3,
                    message: None,
                },
            ))?;
            assert!(runtime.stream_terminal);
            assert!(!runtime.revocation_complete);
            assert!(!runtime.route.revoked.load(Ordering::Acquire));
            assert_eq!(runtime.completed_handle()?, handle);
            runtime.begin_secret_query()?;
        }
        assert_eq!(checked_count.load(Ordering::Acquire), 2);
        runtime.complete_non_revoking()?;
        assert!(runtime.revocation_complete);
        assert!(runtime.http_continuation.is_none());
        assert_eq!(
            runtime.check_active(),
            Err(WorkerProviderStreamError::RevokedHandle)
        );
        drop(runtime);
        worker
            .join()
            .map_err(|_| "multi-request route worker panicked")??;
        Ok(())
    }

    #[test]
    fn provider_v2_http_lifecycle_denies_unverified_cancelled_and_stale_continuations_before_enqueue()
    -> Result<(), Box<dyn Error>> {
        for case in 0..5 {
            let cancellation = CancellationToken::default();
            let verification_count = Arc::new(AtomicU64::new(0));
            let count_for_verifier = verification_count.clone();
            let (mut runtime, receiver) = checked_http_provider_runtime(
                if case == 4 { 1 } else { 3 },
                cancellation.clone(),
                Box::new(move |_, _| {
                    count_for_verifier.fetch_add(1, Ordering::AcqRel);
                    Err(WorkerProviderStreamError::InvalidRequestAuthority)
                }),
            )?;
            let worker = spawn_checked_http_route(receiver);
            runtime.exchange(WorkerProviderStreamRequest::StartRequest {
                context: provider_context(),
                head: provider_head(),
            })?;
            complete_checked_http_host_request(&mut runtime, provider_handle(1), b"abc".to_vec())?;
            let mut context = provider_context();
            if case == 1 {
                context.generation += 1;
            }
            if case == 2 {
                cancellation.cancel();
            }
            if case == 3 {
                let authority = runtime.http_continuation.as_mut().ok_or("missing scope")?;
                authority.scope.node_id = "foreign.node".to_owned();
            }
            assert!(
                runtime
                    .exchange(WorkerProviderStreamRequest::StartRequest {
                        context,
                        head: provider_head()
                    })
                    .is_err()
            );
            assert_eq!(
                verification_count.load(Ordering::Acquire),
                u64::from(case == 0)
            );
            assert!(runtime.revocation_complete);
            assert!(runtime.http_continuation.is_none());
            assert!(runtime.route.revoked.load(Ordering::Acquire));
            drop(runtime);
            worker
                .join()
                .map_err(|_| "rejected continuation worker panicked")??;
        }
        Ok(())
    }

    #[test]
    fn provider_v2_http_binding_rejects_cross_invocation_and_detects_every_scope_mutation()
    -> Result<(), Box<dyn Error>> {
        let scope = provider_http_scope();
        let baseline = http_invocation_binding_sha256(&scope)?;
        for field in 0..22 {
            let mut changed = scope.clone();
            match field {
                0 => changed.context.session_id = Uuid::from_u128(99),
                1 => changed.context.session_generation += 1,
                2 => changed.context.invocation += 1,
                3 => changed.context.generation += 1,
                4 => changed.profile_id.push('x'),
                5 => changed.principal_id.push('x'),
                6 => changed.prompt_id.push('x'),
                7 => changed.prompt_sha256.push('x'),
                8 => changed.attempt_id.push('x'),
                9 => changed.node_id.push('x'),
                10 => changed.compiled_plan_sha256.push('x'),
                11 => changed.activation_request_ordinal += 1,
                12 => changed.registry_generation += 1,
                13 => changed.registry_digest_sha256.push('x'),
                14 => changed.component_generation += 1,
                15 => changed.component_digest_sha256.push('x'),
                16 => changed.authorization_generation_sha256.push('x'),
                17 => changed.binding_generation += 1,
                18 => changed.binding_set_sha256.push('x'),
                19 => changed.provider.push('x'),
                20 => changed.provider_manifest_sha256.push('x'),
                _ => {
                    changed.prompt_id = scope.attempt_id.clone();
                    changed.attempt_id = scope.prompt_id.clone();
                }
            }
            assert_ne!(
                http_invocation_binding_sha256(&changed)?,
                baseline,
                "field {field}"
            );
        }
        let (mut runtime, receiver) = checked_provider_runtime(CancellationToken::default())?;
        let mut foreign = scope;
        foreign.context.invocation += 1;
        assert_eq!(
            runtime.bind_checked_http_continuations(foreign, 3, Box::new(|_, _| Ok(()))),
            Err(WorkerProviderStreamError::InvalidRequestAuthority)
        );
        assert!(runtime.http_continuation.is_none());
        runtime.revocation_complete = true;
        drop(runtime);
        drop(receiver);
        Ok(())
    }

    #[test]
    fn provider_v2_completed_route_blocks_late_calls_and_drop_revokes() -> Result<(), Box<dyn Error>>
    {
        let (mut runtime, receiver) = started_provider_runtime(CancellationToken::default())?;
        runtime.validator.validate_request(
            2,
            &WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                handle: provider_handle(1),
                sequence: 0,
                bytes: vec![1, 2, 3],
                end: true,
            }),
        )?;
        runtime
            .validator
            .validate_response(2, &WorkerProviderStreamResponse::Unit(Ok(())))?;
        runtime.next_call_id = 3;
        let revoked = Arc::new(AtomicBool::new(false));
        let revoked_for_thread = revoked.clone();
        let worker = std::thread::spawn(move || -> Result<(), WorkerProviderStreamError> {
            for sequence in 0..2 {
                let request = loop {
                    match receiver.try_receive() {
                        Ok(message) => break message,
                        Err(TryRecvError::Empty) => std::thread::yield_now(),
                        Err(TryRecvError::Disconnected) => {
                            return Err(WorkerProviderStreamError::RevokedHandle);
                        }
                    }
                };
                let ProviderV2StreamRouteMessage::Request(call) = request else {
                    return Err(WorkerProviderStreamError::InvalidOrder);
                };
                let event = if sequence == 0 {
                    WorkerProviderResponseFrameEvent::Head(sim_types::WorkerProviderResponseHead {
                        status: 200,
                        headers: Vec::new(),
                    })
                } else {
                    WorkerProviderResponseFrameEvent::Terminal(WorkerProviderTerminal::Completed(
                        vec![4, 2, 1],
                    ))
                };
                call.respond(WorkerProviderStreamResponse::Wait(Ok(
                    WorkerProviderWaitOutcome::Frame(WorkerProviderResponseFrame {
                        handle: provider_handle(1),
                        sequence,
                        event,
                    }),
                )))?;
            }
            loop {
                match receiver.try_receive() {
                    Ok(ProviderV2StreamRouteMessage::Revoke { reply }) => {
                        reply
                            .try_send(Ok(()))
                            .map_err(|_| WorkerProviderStreamError::RevokedHandle)?;
                        revoked_for_thread.store(true, Ordering::Release);
                        return Ok(());
                    }
                    Ok(ProviderV2StreamRouteMessage::Request(_)) => {
                        return Err(WorkerProviderStreamError::InvalidOrder);
                    }
                    Err(TryRecvError::Empty) => std::thread::yield_now(),
                    Err(TryRecvError::Disconnected) => {
                        return Err(WorkerProviderStreamError::RevokedHandle);
                    }
                }
            }
        });
        assert!(matches!(
            runtime.exchange(WorkerProviderStreamRequest::WaitResponse(
                WorkerProviderWaitRequest {
                    handle: provider_handle(1),
                    after_sequence: None,
                    timeout_milliseconds: 10,
                },
            )),
            Ok(WorkerProviderStreamResponse::Wait(Ok(
                WorkerProviderWaitOutcome::Frame(WorkerProviderResponseFrame {
                    event: WorkerProviderResponseFrameEvent::Head(_),
                    ..
                })
            )))
        ));
        assert!(matches!(
            runtime.exchange(WorkerProviderStreamRequest::WaitResponse(
                WorkerProviderWaitRequest {
                    handle: provider_handle(1),
                    after_sequence: Some(0),
                    timeout_milliseconds: 10,
                },
            )),
            Ok(WorkerProviderStreamResponse::Wait(Ok(
                WorkerProviderWaitOutcome::Frame(WorkerProviderResponseFrame {
                    event: WorkerProviderResponseFrameEvent::Terminal(
                        WorkerProviderTerminal::Completed(_)
                    ),
                    ..
                })
            )))
        ));
        assert_eq!(
            runtime.exchange(check_cancelled_request(1)),
            Err(WorkerProviderStreamError::RevokedHandle)
        );
        assert!(!revoked.load(Ordering::Acquire));
        drop(runtime);
        worker
            .join()
            .map_err(|_| "provider route worker panicked")??;
        assert!(revoked.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn provider_v2_failed_cancelled_malformed_and_deadline_paths_revoke()
    -> Result<(), Box<dyn Error>> {
        let terminal_cases = [
            (
                WorkerProviderWaitOutcome::Cancelled,
                WorkerProviderStreamError::Cancelled,
            ),
            (
                WorkerProviderWaitOutcome::Frame(WorkerProviderResponseFrame {
                    handle: provider_handle(1),
                    sequence: 0,
                    event: WorkerProviderResponseFrameEvent::Terminal(
                        WorkerProviderTerminal::Failed {
                            code: "fixture".to_owned(),
                            message: "failed".to_owned(),
                        },
                    ),
                }),
                WorkerProviderStreamError::InvalidTerminal,
            ),
        ];
        for (outcome, expected) in terminal_cases {
            let (mut runtime, receiver) = started_provider_runtime(CancellationToken::default())?;
            let worker = std::thread::spawn(move || -> Result<(), WorkerProviderStreamError> {
                let call = receive_provider_route_request(&receiver)?;
                call.respond(WorkerProviderStreamResponse::Wait(Ok(outcome)))?;
                acknowledge_provider_route_revoke(&receiver)
            });
            assert!(matches!(
                runtime.exchange(WorkerProviderStreamRequest::WaitResponse(
                    WorkerProviderWaitRequest {
                        handle: provider_handle(1),
                        after_sequence: None,
                        timeout_milliseconds: 10,
                    },
                )),
                Ok(WorkerProviderStreamResponse::Wait(Ok(_)))
            ));
            assert_eq!(runtime.ensure_completed(), Err(expected));
            assert!(runtime.revocation_complete);
            worker
                .join()
                .map_err(|_| "provider terminal worker panicked")??;
        }

        let (mut malformed_runtime, malformed_receiver) =
            started_provider_runtime(CancellationToken::default())?;
        let malformed_worker =
            std::thread::spawn(move || -> Result<(), WorkerProviderStreamError> {
                let call = receive_provider_route_request(&malformed_receiver)?;
                call.respond(WorkerProviderStreamResponse::Cost(Ok(
                    sim_types::WorkerProviderCostResponse {
                        accepted: true,
                        approved_microunits: 1,
                        receipt: vec![1],
                    },
                )))?;
                acknowledge_provider_route_revoke(&malformed_receiver)
            });
        assert_eq!(
            malformed_runtime.exchange(check_cancelled_request(1)),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        assert!(malformed_runtime.revocation_complete);
        malformed_worker
            .join()
            .map_err(|_| "provider malformed-response worker panicked")??;

        let (mut deadline_runtime, deadline_receiver) =
            started_provider_runtime(CancellationToken::default())?;
        deadline_runtime.reply_deadline = Duration::from_millis(1);
        let deadline_worker =
            std::thread::spawn(move || -> Result<(), WorkerProviderStreamError> {
                let mut pending_call = None;
                loop {
                    match deadline_receiver.try_receive() {
                        Ok(ProviderV2StreamRouteMessage::Request(call)) => {
                            if pending_call.replace(call).is_some() {
                                return Err(WorkerProviderStreamError::InvalidOrder);
                            }
                        }
                        Ok(ProviderV2StreamRouteMessage::Revoke { reply }) => {
                            // Revocation may overtake the request before this thread runs.
                            drop(pending_call);
                            return match reply.try_send(Ok(())) {
                                Ok(()) | Err(TrySendError::Disconnected(_)) => Ok(()),
                                Err(TrySendError::Full(_)) => {
                                    Err(WorkerProviderStreamError::InvalidOrder)
                                }
                            };
                        }
                        Err(TryRecvError::Empty) => std::thread::yield_now(),
                        Err(TryRecvError::Disconnected) => {
                            return Err(WorkerProviderStreamError::RevokedHandle);
                        }
                    }
                }
            });
        assert_eq!(
            deadline_runtime.exchange(check_cancelled_request(1)),
            Err(WorkerProviderStreamError::TimedOut)
        );
        assert!(deadline_runtime.revocation_complete);
        deadline_worker
            .join()
            .map_err(|_| "provider deadline worker panicked")??;
        Ok(())
    }

    fn receive_provider_route_request(
        receiver: &ProviderV2StreamRouteReceiver,
    ) -> Result<ProviderV2StreamRouteCall, WorkerProviderStreamError> {
        loop {
            match receiver.try_receive() {
                Ok(ProviderV2StreamRouteMessage::Request(call)) => return Ok(call),
                Ok(ProviderV2StreamRouteMessage::Revoke { .. }) => {
                    return Err(WorkerProviderStreamError::InvalidOrder);
                }
                Err(TryRecvError::Empty) => std::thread::yield_now(),
                Err(TryRecvError::Disconnected) => {
                    return Err(WorkerProviderStreamError::RevokedHandle);
                }
            }
        }
    }

    fn acknowledge_provider_route_revoke(
        receiver: &ProviderV2StreamRouteReceiver,
    ) -> Result<(), WorkerProviderStreamError> {
        loop {
            match receiver.try_receive() {
                Ok(ProviderV2StreamRouteMessage::Revoke { reply }) => {
                    return reply
                        .try_send(Ok(()))
                        .map_err(|_| WorkerProviderStreamError::RevokedHandle);
                }
                Ok(ProviderV2StreamRouteMessage::Request(_)) => {
                    return Err(WorkerProviderStreamError::InvalidOrder);
                }
                Err(TryRecvError::Empty) => std::thread::yield_now(),
                Err(TryRecvError::Disconnected) => {
                    return Err(WorkerProviderStreamError::RevokedHandle);
                }
            }
        }
    }

    #[test]
    fn provider_v2_terminal_disposition_distinguishes_completion_and_failure() {
        let completed = WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
            WorkerProviderResponseFrame {
                handle: provider_handle(1),
                sequence: 3,
                event: WorkerProviderResponseFrameEvent::Terminal(
                    WorkerProviderTerminal::Completed(vec![1, 2, 3]),
                ),
            },
        )));
        assert_eq!(
            worker_response_terminal_disposition(&completed),
            Some(Ok(()))
        );

        let failed = WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
            WorkerProviderResponseFrame {
                handle: provider_handle(1),
                sequence: 3,
                event: WorkerProviderResponseFrameEvent::Terminal(WorkerProviderTerminal::Failed {
                    code: "provider".to_owned(),
                    message: "failed".to_owned(),
                }),
            },
        )));
        assert_eq!(
            worker_response_terminal_disposition(&failed),
            Some(Err(WorkerProviderStreamError::InvalidTerminal))
        );
        assert_eq!(
            worker_response_terminal_disposition(&WorkerProviderStreamResponse::Wait(Ok(
                WorkerProviderWaitOutcome::Cancelled,
            ))),
            Some(Err(WorkerProviderStreamError::Cancelled))
        );
        assert_eq!(
            worker_response_terminal_disposition(&WorkerProviderStreamResponse::Wait(Ok(
                WorkerProviderWaitOutcome::TimedOut,
            ))),
            None
        );
    }

    #[test]
    fn provider_v2_wait_projection_preserves_every_frame_variant() -> Result<(), Box<dyn Error>> {
        use provider_v2_wit_contract::zed::sim_provider_plugin::types as wit;

        assert!(matches!(
            wit_provider_wait_outcome(WorkerProviderWaitOutcome::TimedOut),
            wit::WaitOutcome::TimedOut
        ));
        assert!(matches!(
            wit_provider_wait_outcome(WorkerProviderWaitOutcome::Cancelled),
            wit::WaitOutcome::Cancelled
        ));

        let cases = [
            WorkerProviderResponseFrameEvent::Head(sim_types::WorkerProviderResponseHead {
                status: 207,
                headers: vec![
                    WorkerProviderHeader {
                        name: "x-first".to_owned(),
                        value: "1".to_owned(),
                    },
                    WorkerProviderHeader {
                        name: "x-second".to_owned(),
                        value: "2".to_owned(),
                    },
                ],
            }),
            WorkerProviderResponseFrameEvent::Chunk(WorkerProviderResponseChunk::Binary(vec![
                0, 1, 2,
            ])),
            WorkerProviderResponseFrameEvent::Chunk(WorkerProviderResponseChunk::Text(
                "text".to_owned(),
            )),
            WorkerProviderResponseFrameEvent::Chunk(WorkerProviderResponseChunk::NdjsonLine(
                "{\"line\":1}".to_owned(),
            )),
            WorkerProviderResponseFrameEvent::Terminal(WorkerProviderTerminal::Completed(vec![
                4, 2,
            ])),
            WorkerProviderResponseFrameEvent::Terminal(WorkerProviderTerminal::Failed {
                code: "fixture".to_owned(),
                message: "failed".to_owned(),
            }),
            WorkerProviderResponseFrameEvent::Terminal(WorkerProviderTerminal::Cancelled),
        ];
        for (index, event) in cases.into_iter().enumerate() {
            let outcome = wit_provider_wait_outcome(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: provider_handle(1),
                    sequence: u64::try_from(index)?,
                    event,
                },
            ));
            let wit::WaitOutcome::Frame(frame) = outcome else {
                return Err("frame projection changed its wait outcome".into());
            };
            assert_eq!(frame.handle.invocation, provider_context().invocation);
            assert_eq!(frame.handle.slot, 1);
            assert_eq!(frame.handle.generation, provider_context().generation);
            assert_eq!(frame.sequence, u64::try_from(index)?);
            match (index, frame.event) {
                (0, wit::ResponseFrameEvent::Head(head)) => {
                    assert_eq!(head.status, 207);
                    let [first, second] = head.headers.as_slice() else {
                        return Err("ordered response headers changed length".into());
                    };
                    assert_eq!(first.name, "x-first");
                    assert_eq!(first.value, "1");
                    assert_eq!(second.name, "x-second");
                    assert_eq!(second.value, "2");
                }
                (1, wit::ResponseFrameEvent::Chunk(wit::ResponseChunkValue::Binary(bytes))) => {
                    assert_eq!(bytes, vec![0, 1, 2]);
                }
                (2, wit::ResponseFrameEvent::Chunk(wit::ResponseChunkValue::Text(text))) => {
                    assert_eq!(text, "text");
                }
                (3, wit::ResponseFrameEvent::Chunk(wit::ResponseChunkValue::NdjsonLine(line))) => {
                    assert_eq!(line, "{\"line\":1}")
                }
                (4, wit::ResponseFrameEvent::Terminal(wit::Terminal::Completed(receipt))) => {
                    assert_eq!(receipt, vec![4, 2]);
                }
                (5, wit::ResponseFrameEvent::Terminal(wit::Terminal::Failed(failure))) => {
                    assert_eq!(failure, ("fixture".to_owned(), "failed".to_owned()));
                }
                (6, wit::ResponseFrameEvent::Terminal(wit::Terminal::Cancelled)) => {}
                _ => return Err("response frame projection changed its exact variant".into()),
            }
        }
        Ok(())
    }

    #[test]
    fn provider_v2_stream_error_projection_is_exhaustive_and_intentionally_collapsed() {
        use provider_v2_wit_contract::zed::sim_provider_plugin::types::StreamError as WitError;

        let one_to_one = [
            (WorkerProviderStreamError::Cancelled, WitError::Cancelled),
            (WorkerProviderStreamError::TimedOut, WitError::TimedOut),
            (
                WorkerProviderStreamError::HostFailure,
                WitError::HostFailure,
            ),
            (
                WorkerProviderStreamError::InvalidContract,
                WitError::InvalidContract,
            ),
            (
                WorkerProviderStreamError::InvalidHandle,
                WitError::InvalidHandle,
            ),
            (
                WorkerProviderStreamError::ForeignHandle,
                WitError::ForeignHandle,
            ),
            (
                WorkerProviderStreamError::RevokedHandle,
                WitError::RevokedHandle,
            ),
            (
                WorkerProviderStreamError::InvalidMethod,
                WitError::InvalidMethod,
            ),
            (
                WorkerProviderStreamError::InvalidHeaders,
                WitError::InvalidHeaders,
            ),
            (WorkerProviderStreamError::BodyLimit, WitError::BodyLimit),
            (WorkerProviderStreamError::ChunkLimit, WitError::ChunkLimit),
            (
                WorkerProviderStreamError::InvalidNdjsonLine,
                WitError::InvalidNdjsonLine,
            ),
            (
                WorkerProviderStreamError::InvalidSequence,
                WitError::InvalidSequence,
            ),
            (
                WorkerProviderStreamError::InvalidOrder,
                WitError::InvalidOrder,
            ),
            (WorkerProviderStreamError::WaitLimit, WitError::WaitLimit),
            (
                WorkerProviderStreamError::InvalidUpload,
                WitError::InvalidUpload,
            ),
            (
                WorkerProviderStreamError::InvalidCostRequest,
                WitError::InvalidCostRequest,
            ),
            (
                WorkerProviderStreamError::InvalidProgress,
                WitError::InvalidProgress,
            ),
            (
                WorkerProviderStreamError::InvalidTerminal,
                WitError::InvalidTerminal,
            ),
            (
                WorkerProviderStreamError::InvalidInvocationResult,
                WitError::InvalidInvocationResult,
            ),
            (
                WorkerProviderStreamError::InvalidRequestAuthority,
                WitError::InvalidRequestAuthority,
            ),
        ];
        for (worker, wit) in one_to_one {
            assert_eq!(wit_provider_stream_error(worker.clone()), wit);
            assert_eq!(worker_provider_stream_error(wit), worker);
        }
        for worker in [
            WorkerProviderStreamError::ForeignSession,
            WorkerProviderStreamError::ForeignInvocation,
        ] {
            assert_eq!(wit_provider_stream_error(worker), WitError::ForeignHandle);
        }
        for worker in [
            WorkerProviderStreamError::StaleSession,
            WorkerProviderStreamError::StaleGeneration,
        ] {
            assert_eq!(wit_provider_stream_error(worker), WitError::RevokedHandle);
        }
    }

    #[test]
    fn provider_v2_adapter_source_contains_no_runtime_service_or_public_transport_authority() {
        let source = include_str!("sim_plugin_host.rs");
        let route = source
            .split("struct ProviderV2StreamRoute {")
            .nth(1)
            .and_then(|source| source.split("pub struct ProviderInvocationResult").next())
            .expect("provider-v2 route source must exist");
        for forbidden in [
            "ProviderRuntimeStreamService",
            "ProviderPolicy",
            "ProviderRuntimeAuthorityInput",
            "dyn Provider",
            "pub trait",
            "SyncSender::send",
            ".send(",
        ] {
            assert!(
                !route.contains(forbidden),
                "forbidden route authority: {forbidden}"
            );
        }
        assert!(route.contains("MAX_WORKER_PROVIDER_PENDING_CALLS"));
        assert!(route.contains("try_send"));
        assert!(route.contains("WorkerProviderStreamTransportValidator"));
    }

    #[test]
    fn component_projection_requires_an_exact_bounded_match() {
        let signed = ComponentManifestProjection {
            component_world: sim_plugin_sdk::COMPONENT_WORLD.to_owned(),
            schema_version: sim_plugin_sdk::MANIFEST_SCHEMA_VERSION,
            identifier: "test.plugin".to_owned(),
            plugin_version: sim_plugin_sdk::ApiVersion::new(1, 0, 0),
            api: sim_plugin_sdk::ApiRequirement {
                major: 1,
                minimum_minor: 0,
                maximum_minor: 0,
                required_features: Vec::new(),
            },
            nodes: Vec::new(),
            capabilities: Vec::new(),
            ui: Vec::new(),
            routes: Vec::new(),
            legacy_mappings: Vec::new(),
        };
        assert!(validate_component_projection(&signed, &signed).is_ok());
        let mut changed = signed.clone();
        changed.identifier = "test.changed".to_owned();
        assert!(matches!(
            validate_component_projection(&signed, &changed),
            Err(PluginError::ManifestProjectionMismatch)
        ));
        let mut oversized = signed;
        oversized.identifier = "x".repeat(sim_plugin_sdk::MAX_MANIFEST_BYTES + 1);
        assert!(matches!(
            validate_component_projection(&oversized, &oversized),
            Err(PluginError::ManifestProjectionMismatch)
        ));
    }

    #[test]
    fn typed_scalar_projection_rejects_hostile_graphs() -> Result<(), PluginError> {
        use wit_contract::zed::sim_plugin::types::{
            ScalarNode as WitScalarNode, ScalarRecordEntry, ScalarValue as WitScalarValue,
        };

        let valid = sdk_projected_scalar(WitScalarValue {
            root_node: 0,
            nodes: vec![
                WitScalarNode::RecordValue(vec![ScalarRecordEntry {
                    key: "items".to_owned(),
                    value_node: 1,
                }]),
                WitScalarNode::ListValue(vec![2]),
                WitScalarNode::TextValue("value".to_owned()),
            ],
        })?;
        assert_eq!(
            valid,
            sim_plugin_sdk::ScalarValue::Record(vec![(
                "items".to_owned(),
                sim_plugin_sdk::ScalarValue::List(vec![sim_plugin_sdk::ScalarValue::String(
                    "value".to_owned()
                ),]),
            )])
        );

        let hostile = [
            WitScalarValue {
                root_node: 1,
                nodes: vec![WitScalarNode::NullValue],
            },
            WitScalarValue {
                root_node: 0,
                nodes: vec![WitScalarNode::ListValue(vec![0])],
            },
            WitScalarValue {
                root_node: 0,
                nodes: vec![WitScalarNode::NullValue, WitScalarNode::NullValue],
            },
            WitScalarValue {
                root_node: 0,
                nodes: vec![WitScalarNode::ListValue(vec![3])],
            },
            WitScalarValue {
                root_node: 0,
                nodes: vec![
                    WitScalarNode::RecordValue(vec![
                        ScalarRecordEntry {
                            key: "duplicate".to_owned(),
                            value_node: 1,
                        },
                        ScalarRecordEntry {
                            key: "duplicate".to_owned(),
                            value_node: 2,
                        },
                    ]),
                    WitScalarNode::NullValue,
                    WitScalarNode::NullValue,
                ],
            },
        ];
        for scalar in hostile {
            assert!(matches!(
                sdk_projected_scalar(scalar),
                Err(PluginError::ManifestProjectionMismatch)
            ));
        }

        let mut deep_nodes = (0..=64)
            .map(|index| WitScalarNode::ListValue(vec![index + 1]))
            .collect::<Vec<_>>();
        deep_nodes.push(WitScalarNode::NullValue);
        assert!(matches!(
            sdk_projected_scalar(WitScalarValue {
                root_node: 0,
                nodes: deep_nodes,
            }),
            Err(PluginError::ManifestProjectionMismatch)
        ));
        Ok(())
    }

    #[test]
    fn typed_wit_errors_round_trip_every_sdk_variant() {
        use sim_plugin_sdk::{CapabilityKind, ValueFamily};
        let errors = vec![
            InvocationError::Cancelled,
            InvocationError::TimedOut,
            InvocationError::UnknownPort("input".to_owned()),
            InvocationError::WrongDirection("output".to_owned()),
            InvocationError::MissingRequiredPort("required".to_owned()),
            InvocationError::InvalidCardinality("list".to_owned()),
            InvocationError::IndexOutOfBounds {
                port: "items".to_owned(),
                index: 7,
            },
            InvocationError::AlreadyTaken {
                port: "items".to_owned(),
                index: 3,
            },
            InvocationError::InvalidHandle,
            InvocationError::RevokedHandle,
            InvocationError::WrongValueFamily {
                port: "image".to_owned(),
                expected: ValueFamily::Tensor,
                actual: ValueFamily::Scalar,
            },
            InvocationError::OutputAlreadyFinished("output".to_owned()),
            InvocationError::UnfinishedOutput("output".to_owned()),
            InvocationError::CapabilityDenied {
                kind: CapabilityKind::NetworkProvider,
                scope: "provider".to_owned(),
            },
            InvocationError::QuotaExceeded {
                kind: CapabilityKind::TransactionalOutput,
                limit: "total-byte".to_owned(),
            },
            InvocationError::InvocationQuotaExceeded {
                limit: "port-operation".to_owned(),
            },
            InvocationError::InvalidCapabilityRequest("invalid".to_owned()),
            InvocationError::HostFailure("host".to_owned()),
            InvocationError::PluginFailure("plugin".to_owned()),
        ];
        for error in errors {
            assert_eq!(sdk_error(wit_error(error.clone())), error);
        }
    }

    #[test]
    fn wasm_store_has_fuel_and_resource_limits() -> Result<(), PluginError> {
        let host = PluginHost::new()?;
        let store = host.new_wasm_store()?;
        assert_eq!(store.get_fuel().ok(), Some(host.limits.maximum_fuel));
        Ok(())
    }

    #[test]
    fn preflight_phase_diagnostics_preserve_trap_and_instantiation_classes() {
        let error = preflight_component_error(preflight_phase_error(
            ComponentPreflightPhase::Instantiation,
            PluginError::WasmTrap("wasm trap: interrupt".to_owned()),
        ));
        assert!(matches!(error, PluginError::ComponentCompilation(message)
            if message == "preflight instantiation: wasm trap: interrupt"));
        for (phase, label) in [
            (ComponentPreflightPhase::Manifest, "manifest"),
            (ComponentPreflightPhase::ProviderBinding, "provider-binding"),
        ] {
            let error = preflight_phase_error(
                phase,
                PluginError::WasmTrap("wasm trap: interrupt".to_owned()),
            );
            assert!(matches!(error, PluginError::WasmTrap(message)
                if message == format!("preflight {label}: wasm trap: interrupt")));
        }
    }

    #[test]
    fn preflight_phase_diagnostics_keep_typed_cancellation_and_projection_errors() {
        assert!(matches!(
            preflight_phase_error(
                ComponentPreflightPhase::Manifest,
                PluginError::Invocation(InvocationError::Cancelled),
            ),
            PluginError::Invocation(InvocationError::Cancelled)
        ));
        assert!(matches!(
            preflight_phase_error(
                ComponentPreflightPhase::ProviderBinding,
                PluginError::ProviderBindingMismatch,
            ),
            PluginError::ProviderBindingMismatch
        ));
        assert!(matches!(
            preflight_phase_error(
                ComponentPreflightPhase::Manifest,
                PluginError::ManifestProjectionMismatch,
            ),
            PluginError::ManifestProjectionMismatch
        ));
    }

    #[test]
    fn preflight_phase_diagnostics_reapply_control_filter_and_character_bound() {
        let error = preflight_phase_error(
            ComponentPreflightPhase::Manifest,
            PluginError::WasmTrap(format!("\0{}", "é".repeat(5_000))),
        );
        assert!(matches!(error, PluginError::WasmTrap(message)
            if message.starts_with("preflight manifest: ")
                && message.chars().count() == 4_096
                && !message.contains('\0')));
    }

    #[test]
    fn preflight_timing_diagnostic_preserves_original_error_class_and_elapsed() {
        let error = preflight_timed_phase_error(
            ComponentPreflightPhase::Manifest,
            PluginError::WasmTrap("wasm trap: interrupt".to_owned()),
            Duration::from_micros(17),
        );
        assert!(matches!(error, PluginError::WasmTrap(message)
            if message == "preflight manifest: elapsed_us=17: wasm trap: interrupt"));
        let error = preflight_component_error(preflight_timed_phase_error(
            ComponentPreflightPhase::Instantiation,
            PluginError::WasmTrap("wasm trap: interrupt".to_owned()),
            Duration::from_micros(100_001),
        ));
        assert!(matches!(error, PluginError::ComponentCompilation(message)
            if message == "preflight instantiation: elapsed_us=100001: wasm trap: interrupt"));
        assert!(matches!(
            preflight_timed_phase_error(
                ComponentPreflightPhase::Manifest,
                PluginError::Invocation(InvocationError::Cancelled),
                Duration::from_micros(17),
            ),
            PluginError::Invocation(InvocationError::Cancelled)
        ));
        assert!(matches!(
            preflight_timed_phase_error(
                ComponentPreflightPhase::Manifest,
                PluginError::ManifestProjectionMismatch,
                Duration::from_micros(17),
            ),
            PluginError::ManifestProjectionMismatch
        ));
    }

    #[test]
    fn preflight_timing_diagnostic_retains_control_filter_and_size_bound() {
        let error = preflight_timed_phase_error(
            ComponentPreflightPhase::ProviderBinding,
            PluginError::ComponentCompilation(format!("\0{}", "é".repeat(5_000))),
            Duration::from_micros(99_999),
        );
        assert!(matches!(error, PluginError::ComponentCompilation(message)
            if message.starts_with("preflight provider-binding: elapsed_us=99999: ")
                && message.chars().count() == 4_096
                && !message.contains('\0')));
    }

    #[test]
    fn metadata_epoch_refresh_preserves_fuel_and_interrupts_within_each_phase()
    -> Result<(), Box<dyn Error>> {
        const STEP_MODULE: &[u8] = &[
            0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 7, 8, 1, 4, 115, 116, 101,
            112, 0, 0, 10, 7, 1, 5, 0, 65, 1, 26, 11,
        ];
        const START_MODULE: &[u8] = &[
            0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 7, 8, 1, 4, 115, 116, 101,
            112, 0, 0, 8, 1, 0, 10, 7, 1, 5, 0, 65, 1, 26, 11,
        ];
        let mut configuration = wasmtime::Config::new();
        configuration.consume_fuel(true).epoch_interruption(true);
        let engine = wasmtime::Engine::new(&configuration)?;
        let module = wasmtime::Module::new(&engine, STEP_MODULE)?;
        let mut store = wasmtime::Store::new(&engine, ());
        store.set_fuel(20)?;
        store.set_epoch_deadline(1);
        let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
        let step = instance.get_typed_func::<(), ()>(&mut store, "step")?;
        engine.increment_epoch();
        let expired = step.call(&mut store, ()).expect_err("expired setup epoch");
        assert_eq!(
            expired.downcast_ref::<wasmtime::Trap>(),
            Some(&wasmtime::Trap::Interrupt)
        );
        let remaining_fuel = store.get_fuel()?;
        store.set_epoch_deadline(1);
        assert_eq!(store.get_fuel()?, remaining_fuel);
        step.call(&mut store, ())?;
        assert!(store.get_fuel()? < remaining_fuel);
        engine.increment_epoch();
        let interrupted = step
            .call(&mut store, ())
            .expect_err("metadata epoch expires");
        assert_eq!(
            interrupted.downcast_ref::<wasmtime::Trap>(),
            Some(&wasmtime::Trap::Interrupt)
        );
        store.set_fuel(0)?;
        store.set_epoch_deadline(1);
        let exhausted = step
            .call(&mut store, ())
            .expect_err("shared fuel remains exhausted");
        assert_eq!(
            exhausted.downcast_ref::<wasmtime::Trap>(),
            Some(&wasmtime::Trap::OutOfFuel)
        );
        assert_eq!(store.get_fuel()?, 0);
        let start_module = wasmtime::Module::new(&engine, START_MODULE)?;
        let mut start_store = wasmtime::Store::new(&engine, ());
        start_store.set_fuel(20)?;
        start_store.set_epoch_deadline(1);
        engine.increment_epoch();
        let expired_start = wasmtime::Instance::new(&mut start_store, &start_module, &[])
            .expect_err("instantiation retains its own deadline");
        assert_eq!(
            expired_start.downcast_ref::<wasmtime::Trap>(),
            Some(&wasmtime::Trap::Interrupt)
        );
        Ok(())
    }

    #[test]
    fn wasm_traps_and_hangs_are_contained() -> Result<(), Box<dyn Error>> {
        const TRAP_MODULE: &[u8] = &[
            0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 7, 8, 1, 4, 116, 114, 97,
            112, 0, 0, 10, 5, 1, 3, 0, 0, 11,
        ];
        const HANG_MODULE: &[u8] = &[
            0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 7, 8, 1, 4, 104, 97, 110,
            103, 0, 0, 10, 9, 1, 7, 0, 3, 64, 12, 0, 11, 11,
        ];

        let host = PluginHost::new()?;
        let trap_module = wasmtime::Module::new(host.runtime.engine(), TRAP_MODULE)?;
        let mut trap_store = host.new_wasm_store()?;
        let trap_instance = wasmtime::Instance::new(&mut trap_store, &trap_module, &[])?;
        let trap = trap_instance.get_typed_func::<(), ()>(&mut trap_store, "trap")?;
        let trap_error = trap.call(&mut trap_store, ()).expect_err("guest trap");
        assert!(matches!(
            trap_error.downcast_ref::<wasmtime::Trap>(),
            Some(wasmtime::Trap::UnreachableCodeReached | wasmtime::Trap::Interrupt)
        ));

        let hang_module = wasmtime::Module::new(host.runtime.engine(), HANG_MODULE)?;
        let mut hang_store = host.new_wasm_store()?;
        let hang_instance = wasmtime::Instance::new(&mut hang_store, &hang_module, &[])?;
        let hang = hang_instance.get_typed_func::<(), ()>(&mut hang_store, "hang")?;
        let hang_error = hang
            .call(&mut hang_store, ())
            .expect_err("guest loop must be contained");
        let remaining_fuel = hang_store.get_fuel()?;
        match hang_error.downcast_ref::<wasmtime::Trap>() {
            Some(wasmtime::Trap::OutOfFuel) => assert_eq!(remaining_fuel, 0),
            // The epoch may expire during host setup before the guest spends fuel.
            Some(wasmtime::Trap::Interrupt) => {
                assert!(remaining_fuel <= host.limits.maximum_fuel);
            }
            cause => {
                return Err(format!("unexpected guest-loop termination: {cause:?}").into());
            }
        }

        let mut interrupted_store = host.new_wasm_store()?;
        let interrupted_instance =
            wasmtime::Instance::new(&mut interrupted_store, &hang_module, &[])?;
        let interrupted =
            interrupted_instance.get_typed_func::<(), ()>(&mut interrupted_store, "hang")?;
        host.interrupt_wasm();
        let interrupted_error = interrupted
            .call(&mut interrupted_store, ())
            .expect_err("explicit epoch interruption");
        assert_eq!(
            interrupted_error.downcast_ref::<wasmtime::Trap>(),
            Some(&wasmtime::Trap::Interrupt)
        );
        assert_eq!(
            interrupted_store.get_fuel().ok(),
            Some(host.limits.maximum_fuel)
        );
        Ok(())
    }

    #[test]
    fn wasm_memory_and_instance_growth_are_enforced() -> Result<(), Box<dyn Error>> {
        const MEMORY_GROW_MODULE: &[u8] = &[
            0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 127, 1, 127, 3, 2, 1, 0, 5, 3, 1, 0, 1, 7,
            8, 1, 4, 103, 114, 111, 119, 0, 0, 10, 8, 1, 6, 0, 32, 0, 64, 0, 11,
        ];
        const TABLE_GROW_MODULE: &[u8] = &[
            0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 127, 1, 127, 3, 2, 1, 0, 4, 4, 1, 112, 0,
            1, 7, 8, 1, 4, 103, 114, 111, 119, 0, 0, 10, 11, 1, 9, 0, 208, 112, 32, 0, 252, 15, 0,
            11,
        ];
        const EMPTY_MODULE: &[u8] = &[0, 97, 115, 109, 1, 0, 0, 0];
        let host = PluginHost::new()?;
        let memory_module = wasmtime::Module::new(host.runtime.engine(), MEMORY_GROW_MODULE)?;
        let mut memory_store = host.new_wasm_store()?;
        let memory_instance = wasmtime::Instance::new(&mut memory_store, &memory_module, &[])?;
        let grow = memory_instance.get_typed_func::<i32, i32>(&mut memory_store, "grow")?;
        assert!(grow.call(&mut memory_store, i32::MAX).is_err());

        let table_module = wasmtime::Module::new(host.runtime.engine(), TABLE_GROW_MODULE)?;
        let mut table_store = host.new_wasm_store()?;
        let table_instance = wasmtime::Instance::new(&mut table_store, &table_module, &[])?;
        let grow = table_instance.get_typed_func::<i32, i32>(&mut table_store, "grow")?;
        assert!(grow.call(&mut table_store, i32::MAX).is_err());

        let instance_module = wasmtime::Module::new(host.runtime.engine(), EMPTY_MODULE)?;
        let mut instance_store = host.new_wasm_store()?;
        let mut instances = Vec::new();
        for _ in 0..host.limits.maximum_instances {
            instances.push(wasmtime::Instance::new(
                &mut instance_store,
                &instance_module,
                &[],
            )?);
        }
        assert!(wasmtime::Instance::new(&mut instance_store, &instance_module, &[]).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod legacy_png_settlement_binding_tests {
    use super::*;
    use sim_runtime::{
        ProviderResultReceiptIssuer, ProviderResultReceiptSession, ProviderTransportPort,
    };
    use std::time::{Duration, Instant};
    const NAMESPACE: &str = "zed.sim.provider.bfl";
    const CLASS: &str = "FluxProUltraImageNode";

    fn settled_result() -> Result<
        (
            ProviderInvocationResult,
            ProviderInvocationIdentity,
            String,
            ProviderTransportResponse,
        ),
        Box<dyn std::error::Error>,
    > {
        settled_result_with_original(include_bytes!(
            "../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
        ))
    }

    fn settled_result_with_original(
        original: &[u8],
    ) -> Result<
        (
            ProviderInvocationResult,
            ProviderInvocationIdentity,
            String,
            ProviderTransportResponse,
        ),
        Box<dyn std::error::Error>,
    > {
        let now = Instant::now();
        let identity = ProviderInvocationIdentity::new(
            "principal.fixture",
            "profile.fixture",
            "prompt.fixture",
            "a".repeat(64),
            "attempt.fixture",
            "node.fixture",
            0,
            "b".repeat(64),
            NAMESPACE,
            "c".repeat(64),
            "d".repeat(64),
            "fixture",
            "https://fixture.invalid/v1/generate",
        )?;
        let issuer = Arc::new(ProviderResultReceiptIssuer::from_seed([89; 32], now)?);
        let mut session = ProviderResultReceiptSession::new(issuer, 1024, 0)?;
        let receipt = session.issue(
            identity.clone(),
            original.to_vec(),
            now,
            now + Duration::from_secs(5),
        )?;
        let set = ProviderResultReceiptSet::new(vec![receipt.clone()])?;
        let resolved = session.resolve_receipt_set(&set, now)?;
        session.finish()?;
        let response = ProviderTransportResponse::checked(
            CLASS,
            vec![ProviderTransportPort::checked(
                "output_0",
                false,
                Vec::new(),
            )?],
        )?;
        Ok((
            ProviderInvocationResult {
                outputs: BTreeMap::new(),
                output_presence: BTreeMap::new(),
                effects: CapabilityEffects::default(),
                receipts: vec![receipt],
                resolved_provider_results: resolved,
                materialized_provider_response: None,
            },
            identity,
            format!("{:x}", Sha256::digest(original)),
            response,
        ))
    }
    #[test]
    fn host_png_materialization_requires_exact_settled_response_and_namespace()
    -> Result<(), Box<dyn std::error::Error>> {
        let (mut result, identity, digest, response) = settled_result()?;
        result.set_materialized_provider_response(
            identity.clone(),
            result
                .receipts
                .first()
                .ok_or("missing original receipt")?
                .clone(),
            digest.clone(),
            response.clone(),
        );
        assert_eq!(
            result.take_materialized_provider_response(NAMESPACE, CLASS)?,
            Some(response.clone())
        );
        for (namespace, class_type, recorded_digest) in [
            ("zed.sim.provider.hitpaw", CLASS, digest.as_str()),
            (NAMESPACE, "FluxProImageNode", digest.as_str()),
            (
                NAMESPACE,
                CLASS,
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            ),
        ] {
            result.set_materialized_provider_response(
                identity.clone(),
                result
                    .receipts
                    .first()
                    .ok_or("missing original receipt")?
                    .clone(),
                recorded_digest.to_owned(),
                response.clone(),
            );
            assert!(
                result
                    .take_materialized_provider_response(namespace, class_type)
                    .is_err()
            );
        }
        let foreign_request = ProviderInvocationIdentity::new(
            identity.principal_id(),
            identity.profile_id(),
            identity.prompt_id(),
            identity.prompt_sha256(),
            identity.attempt_id(),
            identity.node_id(),
            identity.request_ordinal(),
            "f".repeat(64),
            identity.plugin_id(),
            identity.plugin_digest_sha256(),
            identity.provider_binding_sha256(),
            identity.provider(),
            identity.endpoint(),
        )?;
        result.set_materialized_provider_response(
            foreign_request,
            result
                .receipts
                .first()
                .ok_or("missing original receipt")?
                .clone(),
            digest.clone(),
            response.clone(),
        );
        assert!(
            result
                .take_materialized_provider_response(NAMESPACE, CLASS)
                .is_err()
        );
        let (mut original, original_identity, original_digest, original_response) =
            settled_result()?;
        original.set_materialized_provider_response(
            original_identity,
            original
                .receipts
                .first()
                .ok_or("missing original receipt")?
                .clone(),
            original_digest,
            original_response,
        );
        let (mut another_session, _, _, _) = settled_result()?;
        assert_ne!(original.receipts, another_session.receipts);
        another_session.materialized_provider_response =
            original.materialized_provider_response.clone();
        assert!(
            another_session
                .take_materialized_provider_response(NAMESPACE, CLASS)
                .is_err()
        );
        let (mut changed_response, _, _, _) =
            settled_result_with_original(b"different authenticated original response")?;
        changed_response.materialized_provider_response =
            original.materialized_provider_response.clone();
        assert!(
            changed_response
                .take_materialized_provider_response(NAMESPACE, CLASS)
                .is_err()
        );
        let foreign_node = ProviderInvocationIdentity::new(
            identity.principal_id(),
            identity.profile_id(),
            identity.prompt_id(),
            identity.prompt_sha256(),
            identity.attempt_id(),
            "different.native.node",
            identity.request_ordinal(),
            identity.request_sha256(),
            identity.plugin_id(),
            identity.plugin_digest_sha256(),
            identity.provider_binding_sha256(),
            identity.provider(),
            identity.endpoint(),
        )?;
        result.set_materialized_provider_response(
            foreign_node,
            result
                .receipts
                .first()
                .ok_or("missing original receipt")?
                .clone(),
            digest.clone(),
            response.clone(),
        );
        assert!(
            result
                .take_materialized_provider_response(NAMESPACE, CLASS)
                .is_err()
        );
        original.receipts.clear();
        assert!(
            original
                .take_materialized_provider_response(NAMESPACE, CLASS)
                .is_err()
        );
        // Recovery uses a newly settled genuine record, never the rejected copied record.
        let (mut recovered, recovered_identity, recovered_digest, recovered_response) =
            settled_result()?;
        recovered.set_materialized_provider_response(
            recovered_identity,
            recovered
                .receipts
                .first()
                .ok_or("missing original receipt")?
                .clone(),
            recovered_digest,
            recovered_response.clone(),
        );
        assert_eq!(
            recovered.take_materialized_provider_response(NAMESPACE, CLASS)?,
            Some(recovered_response)
        );
        result.set_materialized_provider_response(
            identity,
            result
                .receipts
                .first()
                .ok_or("missing original receipt")?
                .clone(),
            digest,
            response,
        );
        result.resolved_provider_results.clear();
        assert!(
            result
                .take_materialized_provider_response(NAMESPACE, CLASS)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn guest_wire_cannot_supply_host_png_materialization() -> Result<(), Box<dyn std::error::Error>>
    {
        let (mut result, identity, digest, response) = settled_result()?;
        result.set_materialized_provider_response(
            identity,
            result
                .receipts
                .first()
                .ok_or("missing original receipt")?
                .clone(),
            digest,
            response,
        );
        let mut wire = serde_json::to_value(&result)?;
        assert!(wire.get("materialized_provider_response").is_none());
        let decoded: ProviderInvocationResult = serde_json::from_value(wire.clone())?;
        assert!(decoded.materialized_provider_response.is_none());
        wire.as_object_mut()
            .ok_or("result wire is not an object")?
            .insert(
                "materialized_provider_response".to_owned(),
                serde_json::json!({"response":"unverified guest object"}),
            );
        assert!(serde_json::from_value::<ProviderInvocationResult>(wire).is_err());
        Ok(())
    }
}
