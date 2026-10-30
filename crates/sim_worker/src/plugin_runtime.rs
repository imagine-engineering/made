use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use sha2::{Digest, Sha256};
use sim_nodes::{
    NativeCacheDependencies, NativeNode, NativeNodeContext, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeValue,
};
use sim_plugin_host::{
    CapabilityServiceContext, ComponentLimits, InvocationInputs, InvocationResult,
    PluginCapabilityServices, PluginError, PluginHost, ProviderInvocationResult,
    ProviderV2WorkerPendingInvocation, ProviderV2WorkerStreamCall, WorkerPluginInvocation,
    materialize_native_provider_response, prepare_native_provider_invocation,
    rollback_native_provider_outputs,
};
use sim_plugin_sdk::{
    CapabilityKind, InvocationError, ModelValue, PluginManifest, PluginNode,
    ProviderPluginManifestV2, ProviderResultReceiptSet,
};
use sim_runtime::{
    AssetIdentity, NativeNodeRegistry, NativeProviderBindingActivation,
    NativeProviderBindingActivationSet, NativeProviderRegistryPin, NativeProviderWorkerRequest,
    NativeProviderWorkerResponse, PluginAuthorization, PluginAuthorizationVerifier,
    PluginServiceWireFailure, PluginServiceWireRequest, PluginServiceWireResponse,
    ProviderTransportResponse, SecretId,
};
use sim_tensor::CancellationToken;
use sim_types::{
    MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES, MAX_WORKER_PLUGIN_RESULT_BYTES, ProfileId,
    WorkerPluginExecutionFailure, WorkerPluginExecutionOutcome, WorkerProviderStreamError,
    WorkerRegistryDeploymentRejectionReason,
};
use thiserror::Error;
use uuid::Uuid;

use crate::AssembledWorkerRegistry;

const CAPABILITY_BRIDGE_POLL_INTERVAL: Duration = Duration::from_millis(5);

pub(crate) enum WorkerPluginInvocationResult {
    Node(InvocationResult),
    Provider(ProviderInvocationResult),
    ProviderV2(ProviderV2WorkerPendingInvocation),
}

pub(crate) enum WorkerPluginTaskEvent {
    Terminal(WorkerPluginExecutionOutcome),
    ProviderV2Proposal {
        outcome: WorkerPluginExecutionOutcome,
        pending: ProviderV2WorkerPendingInvocation,
    },
}

pub(crate) struct WorkerCapabilityBridgeRequest {
    pub call_id: u64,
    pub request: Vec<u8>,
    pub response_sender: async_channel::Sender<Vec<u8>>,
}

#[derive(Clone)]
pub(crate) struct WorkerCapabilityBridge {
    sender: async_channel::Sender<WorkerCapabilityBridgeRequest>,
    provider_v2_sender: Option<async_channel::Sender<ProviderV2WorkerStreamCall>>,
    next_call_id: Arc<AtomicU64>,
    native_provider_session_id: Option<String>,
}

impl WorkerCapabilityBridge {
    pub fn new(sender: async_channel::Sender<WorkerCapabilityBridgeRequest>) -> Self {
        Self {
            sender,
            provider_v2_sender: None,
            next_call_id: Arc::new(AtomicU64::new(1)),
            native_provider_session_id: None,
        }
    }

    pub fn with_provider_v2_sender(
        mut self,
        provider_v2_sender: async_channel::Sender<ProviderV2WorkerStreamCall>,
    ) -> Self {
        self.provider_v2_sender = Some(provider_v2_sender);
        self
    }

    fn provider_v2_sender(
        &self,
    ) -> Result<async_channel::Sender<ProviderV2WorkerStreamCall>, WorkerPluginRuntimeError> {
        self.provider_v2_sender
            .clone()
            .ok_or(WorkerPluginRuntimeError::InvocationFailed)
    }

    pub fn for_native_provider(&self, native_provider_session_id: impl Into<String>) -> Self {
        Self {
            sender: self.sender.clone(),
            provider_v2_sender: self.provider_v2_sender.clone(),
            next_call_id: self.next_call_id.clone(),
            native_provider_session_id: Some(native_provider_session_id.into()),
        }
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "the private worker process has no GPUI dispatcher; its blocking native provider thread must cooperatively poll the async IPC bridge"
    )]
    pub fn native_provider_control(
        &self,
        request: NativeProviderWorkerRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeProviderWorkerResponse, WorkerPluginRuntimeError> {
        cancellation
            .check()
            .map_err(|_| WorkerPluginRuntimeError::InvocationFailed)?;
        let request = request
            .to_bytes()
            .map_err(|_| WorkerPluginRuntimeError::InvalidDeployment)?;
        if request.len() > MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES {
            return Err(WorkerPluginRuntimeError::InvalidDeployment);
        }
        let call_id = self.next_call_id.fetch_add(1, Ordering::Relaxed);
        if call_id == 0 {
            return Err(WorkerPluginRuntimeError::InvocationFailed);
        }
        let (response_sender, response_receiver) = async_channel::bounded(1);
        let mut bridge_request = WorkerCapabilityBridgeRequest {
            call_id,
            request,
            response_sender,
        };
        loop {
            cancellation
                .check()
                .map_err(|_| WorkerPluginRuntimeError::InvocationFailed)?;
            match self.sender.try_send(bridge_request) {
                Ok(()) => break,
                Err(async_channel::TrySendError::Full(returned)) => {
                    bridge_request = returned;
                    smol::block_on(async_io::Timer::after(CAPABILITY_BRIDGE_POLL_INTERVAL));
                }
                Err(async_channel::TrySendError::Closed(_)) => {
                    return Err(WorkerPluginRuntimeError::InvocationFailed);
                }
            }
        }
        let response = loop {
            cancellation
                .check()
                .map_err(|_| WorkerPluginRuntimeError::InvocationFailed)?;
            let received = smol::block_on(smol::future::race(
                async { Some(response_receiver.recv().await) },
                async {
                    async_io::Timer::after(CAPABILITY_BRIDGE_POLL_INTERVAL).await;
                    None
                },
            ));
            match received {
                Some(Ok(response)) => break response,
                Some(Err(_)) => return Err(WorkerPluginRuntimeError::InvocationFailed),
                None => {}
            }
        };
        cancellation
            .check()
            .map_err(|_| WorkerPluginRuntimeError::InvocationFailed)?;
        NativeProviderWorkerResponse::from_bytes(&response)
            .map_err(|_| WorkerPluginRuntimeError::InvocationFailed)
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "the private worker process has no GPUI dispatcher; its blocking component thread must cooperatively poll the async IPC bridge"
    )]
    fn call(
        &self,
        request: PluginServiceWireRequest,
        context: &CapabilityServiceContext,
        kind: CapabilityKind,
        scope: &str,
    ) -> Result<PluginServiceWireResponse, InvocationError> {
        context.check_active()?;
        let request = request.to_bytes().map_err(|_| {
            InvocationError::InvalidCapabilityRequest(
                "plugin capability request cannot be represented".to_owned(),
            )
        })?;
        let request = match &self.native_provider_session_id {
            Some(session_id) => NativeProviderWorkerRequest::Call {
                session_id: session_id.clone(),
                request,
            }
            .to_bytes()
            .map_err(|_| {
                InvocationError::InvalidCapabilityRequest(
                    "native provider capability request cannot be represented".to_owned(),
                )
            })?,
            None => request,
        };
        if request.len() > MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES {
            return Err(InvocationError::InvalidCapabilityRequest(
                "plugin capability request exceeds the worker transport bound".to_owned(),
            ));
        }
        let call_id = self.next_call_id.fetch_add(1, Ordering::Relaxed);
        if call_id == 0 {
            return Err(InvocationError::HostFailure(
                "plugin capability call identifier overflowed".to_owned(),
            ));
        }
        let (response_sender, response_receiver) = async_channel::bounded(1);
        let mut bridge_request = WorkerCapabilityBridgeRequest {
            call_id,
            request,
            response_sender,
        };
        loop {
            context.check_active()?;
            match self.sender.try_send(bridge_request) {
                Ok(()) => break,
                Err(async_channel::TrySendError::Full(returned)) => {
                    bridge_request = returned;
                    smol::block_on(async_io::Timer::after(CAPABILITY_BRIDGE_POLL_INTERVAL));
                }
                Err(async_channel::TrySendError::Closed(_)) => {
                    return Err(InvocationError::HostFailure(
                        "private worker capability bridge is unavailable".to_owned(),
                    ));
                }
            }
        }
        let response = loop {
            context.check_active()?;
            let received = smol::block_on(smol::future::race(
                async { Some(response_receiver.recv().await) },
                async {
                    async_io::Timer::after(CAPABILITY_BRIDGE_POLL_INTERVAL).await;
                    None
                },
            ));
            match received {
                Some(Ok(response)) => break response,
                Some(Err(_)) => {
                    return Err(InvocationError::HostFailure(
                        "private worker capability response was lost".to_owned(),
                    ));
                }
                None => {}
            }
        };
        let transport_maximum = context
            .maximum_response_bytes()
            .saturating_add(64 * 1024)
            .min(u64::try_from(MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES).unwrap_or(u64::MAX));
        let response = match &self.native_provider_session_id {
            Some(_) => match NativeProviderWorkerResponse::from_bytes(&response).map_err(|_| {
                InvocationError::HostFailure(
                    "native provider capability response is malformed".to_owned(),
                )
            })? {
                NativeProviderWorkerResponse::Call(response) => response,
                NativeProviderWorkerResponse::Failure(failure) => {
                    return Err(map_wire_failure(failure, kind, scope));
                }
                _ => {
                    return Err(InvocationError::HostFailure(
                        "native provider capability response has the wrong phase".to_owned(),
                    ));
                }
            },
            None => response,
        };
        let response = PluginServiceWireResponse::from_bytes(&response, transport_maximum)
            .map_err(|_| {
                InvocationError::HostFailure(
                    "private worker capability response is malformed".to_owned(),
                )
            })?;
        context.check_active()?;
        match response {
            PluginServiceWireResponse::Failure(failure) => {
                Err(map_wire_failure(failure, kind, scope))
            }
            response => Ok(response),
        }
    }
}

impl PluginCapabilityServices for WorkerCapabilityBridge {
    fn read_asset(
        &self,
        identity: &AssetIdentity,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        let reference = identity.to_reference().map_err(|_| {
            InvocationError::InvalidCapabilityRequest(
                "asset reference is not available to this plugin".to_owned(),
            )
        })?;
        let namespace = identity.namespace.locator_type();
        match self.call(
            PluginServiceWireRequest::ReadAsset {
                namespace: namespace.to_owned(),
                asset_reference: reference,
            },
            context,
            CapabilityKind::Filesystem,
            namespace,
        )? {
            PluginServiceWireResponse::Bytes(bytes) => {
                context.validate_response_length(bytes.len())?;
                Ok(bytes)
            }
            _ => Err(invalid_wire_response("asset")),
        }
    }

    fn call_provider(
        &self,
        _provider: &str,
        _endpoint: &str,
        _body: &[u8],
        _secret_id: Option<&SecretId>,
    ) -> Result<Vec<u8>, InvocationError> {
        Err(context_required("provider"))
    }

    fn secret_exists(&self, _identifier: &str) -> Result<bool, InvocationError> {
        Err(context_required("credentials"))
    }

    fn clock_milliseconds(&self, _clock: &str) -> Result<u64, InvocationError> {
        Err(context_required("clock"))
    }

    fn random_bytes(&self, _stream: &str, _length: u32) -> Result<Vec<u8>, InvocationError> {
        Err(context_required("randomness"))
    }

    fn open_model(&self, _identifier: &str) -> Result<ModelValue, InvocationError> {
        Err(context_required("model"))
    }

    fn sanitize_log(&self, _level: &str, _message: &str) -> Result<String, InvocationError> {
        Err(context_required("log"))
    }

    fn call_provider_with_context(
        &self,
        provider: &str,
        endpoint: &str,
        body: &[u8],
        secret_id: Option<&SecretId>,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        let scope = format!("{provider}|{endpoint}");
        match self.call(
            PluginServiceWireRequest::ExecuteProvider {
                provider: provider.to_owned(),
                endpoint: endpoint.to_owned(),
                secret_id: secret_id.map(|secret_id| secret_id.as_str().to_owned()),
                body: body.to_vec(),
            },
            context,
            CapabilityKind::NetworkProvider,
            &scope,
        )? {
            PluginServiceWireResponse::Bytes(bytes) => {
                context.validate_response_length(bytes.len())?;
                Ok(bytes)
            }
            _ => Err(invalid_wire_response("provider")),
        }
    }

    fn secret_exists_with_context(
        &self,
        identifier: &str,
        context: &CapabilityServiceContext,
    ) -> Result<bool, InvocationError> {
        match self.call(
            PluginServiceWireRequest::CredentialIsPresent {
                secret_id: identifier.to_owned(),
            },
            context,
            CapabilityKind::Secret,
            identifier,
        )? {
            PluginServiceWireResponse::Boolean(exists) => Ok(exists),
            _ => Err(invalid_wire_response("credentials")),
        }
    }

    fn clock_milliseconds_with_context(
        &self,
        clock: &str,
        context: &CapabilityServiceContext,
    ) -> Result<u64, InvocationError> {
        match self.call(
            PluginServiceWireRequest::MonotonicMilliseconds {
                clock_id: clock.to_owned(),
            },
            context,
            CapabilityKind::Clock,
            clock,
        )? {
            PluginServiceWireResponse::TimestampMilliseconds(milliseconds) => Ok(milliseconds),
            _ => Err(invalid_wire_response("clock")),
        }
    }

    fn random_bytes_with_context(
        &self,
        stream: &str,
        length: u32,
        context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        match self.call(
            PluginServiceWireRequest::RandomBytes {
                stream_id: stream.to_owned(),
                length,
            },
            context,
            CapabilityKind::Randomness,
            stream,
        )? {
            PluginServiceWireResponse::Bytes(bytes) => {
                context.validate_response_length(bytes.len())?;
                Ok(bytes)
            }
            _ => Err(invalid_wire_response("randomness")),
        }
    }

    fn open_model_with_context(
        &self,
        identifier: &str,
        context: &CapabilityServiceContext,
    ) -> Result<ModelValue, InvocationError> {
        match self.call(
            PluginServiceWireRequest::LoadModel {
                model_id: identifier.to_owned(),
            },
            context,
            CapabilityKind::Model,
            identifier,
        )? {
            PluginServiceWireResponse::Model {
                identifier,
                format,
                digest_sha256,
            } => ModelValue::new(identifier, format, digest_sha256).map_err(|_| {
                InvocationError::HostFailure(
                    "canonical model handle cannot be represented by the plugin ABI".to_owned(),
                )
            }),
            _ => Err(invalid_wire_response("model")),
        }
    }

    fn sanitize_log_with_context(
        &self,
        level: &str,
        message: &str,
        context: &CapabilityServiceContext,
    ) -> Result<String, InvocationError> {
        match self.call(
            PluginServiceWireRequest::SanitizeLog {
                level: level.to_owned(),
                message: message.to_owned(),
            },
            context,
            CapabilityKind::SanitizedLog,
            level,
        )? {
            PluginServiceWireResponse::SanitizedLog(message) => {
                context.validate_response_length(message.len())?;
                Ok(message)
            }
            _ => Err(invalid_wire_response("log")),
        }
    }
}

pub(crate) struct WorkerPluginRegistry {
    profile_id: ProfileId,
    generation: sim_types::WorkerRegistryGeneration,
    registry_digest_sha256: sim_types::WorkerSha256Digest,
    host: Arc<PluginHost>,
    component_limits: ComponentLimits,
    components: BTreeMap<String, WorkerCompiledPlugin>,
}

struct WorkerCompiledPlugin {
    extension_version: String,
    plugin_identifier: String,
    plugin_version: String,
    manifest_digest_sha256: sim_types::WorkerSha256Digest,
    component_digest_sha256: sim_types::WorkerSha256Digest,
    authorization_generation: sim_types::WorkerSha256Digest,
    manifest: Arc<PluginManifest>,
    authorization: Arc<PluginAuthorization>,
    compiled: Arc<sim_plugin_host::CompiledPlugin>,
    provider_v2: bool,
}

struct WorkerNativeProviderNode {
    registry: Arc<WorkerPluginRegistry>,
    extension_id: String,
    node: PluginNode,
    implementation_version: String,
    descriptor: sim_nodes::NativeNodeDescriptor,
    bridge: Arc<WorkerCapabilityBridge>,
}

impl NativeNode for WorkerNativeProviderNode {
    fn class_type(&self) -> &str {
        &self.node.id
    }

    fn implementation_version(&self) -> &str {
        &self.implementation_version
    }

    fn implementation_namespace(&self) -> &str {
        self.registry
            .components
            .get(&self.extension_id)
            .map(|component| component.plugin_identifier.as_str())
            .unwrap_or("invalid.provider")
    }

    fn cache_change_token(
        &self,
        _inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        Ok(format!(
            "provider:{}:{}",
            self.registry.registry_digest_sha256.as_str(),
            self.node.id
        ))
    }

    fn cache_dependencies(
        &self,
        _context: &NativeNodeContext,
        _inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> futures::future::BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            let component = self
                .registry
                .components
                .get(&self.extension_id)
                .ok_or_else(|| provider_node_failure("provider component is unavailable"))?;
            let binding_set = component
                .manifest
                .provider_binding
                .as_ref()
                .ok_or_else(|| provider_node_failure("provider binding is unavailable"))?;
            let prepared = prepare_native_provider_invocation(
                &self.node,
                &self.descriptor,
                inputs,
                &self.registry.profile_id.0.to_string(),
                &context,
            )
            .await?;
            let session_id = Uuid::new_v4().to_string();
            let start = sim_runtime::NativeProviderWorkerSessionStart {
                session_id: session_id.clone(),
                registry_generation: self.registry.generation.get(),
                registry_digest_sha256: self.registry.registry_digest_sha256.as_str().to_owned(),
                extension_id: self.extension_id.clone(),
                extension_version: component.extension_version.clone(),
                plugin_identifier: component.plugin_identifier.clone(),
                plugin_version: component.plugin_version.clone(),
                manifest_digest_sha256: component.manifest_digest_sha256.as_str().to_owned(),
                component_digest_sha256: component.component_digest_sha256.as_str().to_owned(),
                authorization_generation_sha256: component
                    .authorization_generation
                    .as_str()
                    .to_owned(),
                binding_set_sha256: binding_set.bindings_sha256.clone(),
                node_id: self.node.id.clone(),
                compiled_plan_sha256: context
                    .provider_execution()
                    .map_err(provider_node_failure)?
                    .compiled_plan_sha256()
                    .to_owned(),
                maximum_response_bytes: sim_runtime::MAX_PLUGIN_SERVICE_RESPONSE_BYTES,
            };
            match self.bridge.native_provider_control(
                NativeProviderWorkerRequest::Begin(start),
                &context.cancellation,
            ) {
                Ok(NativeProviderWorkerResponse::Begun) => {}
                _ => {
                    return Err(provider_node_failure(
                        WorkerPluginRuntimeError::InvocationFailed,
                    ));
                }
            }
            let invocation = (|| {
                let provider_bridge = Arc::new(self.bridge.for_native_provider(session_id.clone()));
                let result = self
                    .registry
                    .execute_provider_node(
                        &self.extension_id,
                        &self.node.id,
                        prepared.inputs,
                        prepared.request.to_bytes().map_err(provider_node_failure)?,
                        provider_bridge,
                        context.cancellation.clone(),
                    )
                    .map_err(provider_node_failure)?;
                if !result.outputs.is_empty()
                    || !result.output_presence.is_empty()
                    || !result.effects.outputs.is_empty()
                    || !result.effects.logs.is_empty()
                    || !result.effects.ui_state.is_empty()
                    || !result.effects.routes.is_empty()
                {
                    return Err(provider_node_failure(
                        WorkerPluginRuntimeError::InvalidDeployment,
                    ));
                }
                let receipt_set = ProviderResultReceiptSet::new(result.receipts().to_vec())
                    .and_then(|receipts| receipts.to_bytes())
                    .map_err(provider_node_failure)?;
                let resolved = self
                    .bridge
                    .native_provider_control(
                        NativeProviderWorkerRequest::Resolve {
                            session_id: session_id.clone(),
                            receipt_set,
                        },
                        &context.cancellation,
                    )
                    .map_err(provider_node_failure)?;
                let NativeProviderWorkerResponse::Resolved(mut resolved) = resolved else {
                    return Err(provider_node_failure(
                        WorkerPluginRuntimeError::InvocationFailed,
                    ));
                };
                if resolved.len() != 1 {
                    return Err(provider_node_failure(
                        WorkerPluginRuntimeError::InvalidDeployment,
                    ));
                }
                let response = ProviderTransportResponse::from_bytes(&resolved.remove(0))
                    .map_err(provider_node_failure)?;
                let values = materialize_native_provider_response(
                    &self.node,
                    &response,
                    &component.plugin_identifier,
                    &context,
                )?;
                match self.bridge.native_provider_control(
                    NativeProviderWorkerRequest::Finish {
                        session_id: session_id.clone(),
                    },
                    &context.cancellation,
                ) {
                    Ok(NativeProviderWorkerResponse::Finished) => {}
                    _ => {
                        rollback_native_provider_outputs(&context, &values)?;
                        return Err(provider_node_failure(
                            WorkerPluginRuntimeError::InvocationFailed,
                        ));
                    }
                }
                Ok(NativeNodeOutcome::Values {
                    outputs: values,
                    ui: None,
                    effects: Vec::new(),
                })
            })();
            match invocation {
                Ok(outcome) => Ok(outcome),
                Err(error) => match self.bridge.native_provider_control(
                    NativeProviderWorkerRequest::Abort { session_id },
                    &CancellationToken::default(),
                ) {
                    Ok(NativeProviderWorkerResponse::Aborted) => Err(error),
                    Ok(_) | Err(_) => Err(provider_node_failure(format!(
                        "{error}; provider session cleanup failed"
                    ))),
                },
            }
        })
    }
}

fn provider_node_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "native_provider_invocation_failed".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

impl WorkerPluginRegistry {
    pub fn from_assembled(
        profile_id: ProfileId,
        registry: &AssembledWorkerRegistry,
        component_limits: ComponentLimits,
        authorization_verifier: &PluginAuthorizationVerifier,
    ) -> Result<Self, WorkerPluginRuntimeError> {
        let host = Arc::new(PluginHost::with_configuration(
            component_limits.clone(),
            sim_plugin_host::DEFAULT_API_FEATURES
                .iter()
                .map(|feature| (*feature).to_owned()),
        )?);
        let mut components = BTreeMap::new();
        for component in registry.components() {
            if !digest_matches(
                component.manifest_bytes(),
                component.manifest_digest_sha256(),
            ) || !digest_matches(
                component.authorization_bytes(),
                component.authorization_generation(),
            ) || !digest_matches(
                component.component_bytes(),
                component.component_digest_sha256(),
            ) {
                return Err(WorkerPluginRuntimeError::InvalidDeployment);
            }
            let provider_manifest_v2 =
                serde_json::from_slice::<ProviderPluginManifestV2>(component.manifest_bytes()).ok();
            let manifest: PluginManifest = match &provider_manifest_v2 {
                Some(provider_manifest) => provider_manifest.manifest.clone(),
                None => serde_json::from_slice(component.manifest_bytes())
                    .map_err(|_| WorkerPluginRuntimeError::InvalidDeployment)?,
            };
            if manifest.identifier != component.plugin_identifier()
                || manifest.plugin_version.to_string() != component.plugin_version()
                || manifest.digest_sha256 != component.component_digest_sha256().as_str()
            {
                return Err(WorkerPluginRuntimeError::InvalidDeployment);
            }
            let authorization = PluginAuthorization::from_sealed_bytes(
                component.authorization_bytes(),
                &manifest,
                authorization_verifier,
                authorization_verifier.policy_generation(),
                &profile_id.0.to_string(),
            )?;
            if authorization.capabilities().profile_id() != profile_id.0.to_string() {
                return Err(WorkerPluginRuntimeError::InvalidDeployment);
            }
            let compiled = match &provider_manifest_v2 {
                Some(provider_manifest) => host.compile_provider_component_v2_for_worker(
                    component.component_bytes(),
                    provider_manifest,
                    &authorization,
                )?,
                None => {
                    host.compile_component(component.component_bytes(), &manifest, &authorization)?
                }
            };
            let extension_id = component.extension_id().to_owned();
            if components
                .insert(
                    extension_id,
                    WorkerCompiledPlugin {
                        extension_version: component.extension_version().to_owned(),
                        plugin_identifier: component.plugin_identifier().to_owned(),
                        plugin_version: component.plugin_version().to_owned(),
                        manifest_digest_sha256: component.manifest_digest_sha256().clone(),
                        component_digest_sha256: component.component_digest_sha256().clone(),
                        authorization_generation: component.authorization_generation().clone(),
                        manifest: Arc::new(manifest),
                        authorization: Arc::new(authorization),
                        compiled: Arc::new(compiled),
                        provider_v2: provider_manifest_v2.is_some(),
                    },
                )
                .is_some()
            {
                return Err(WorkerPluginRuntimeError::InvalidDeployment);
            }
        }
        Ok(Self {
            profile_id,
            generation: registry.generation(),
            registry_digest_sha256: registry.registry_digest_sha256().clone(),
            host,
            component_limits,
            components,
        })
    }

    pub fn uses_component_limits(&self, component_limits: &ComponentLimits) -> bool {
        &self.component_limits == component_limits
    }

    pub fn matches_provider_registry_pin(&self, pin: &NativeProviderRegistryPin) -> bool {
        let mut binding_digests = self
            .components
            .values()
            .filter_map(|component| {
                component
                    .manifest
                    .provider_binding
                    .as_ref()
                    .map(|binding| binding.bindings_sha256.clone())
            })
            .collect::<Vec<_>>();
        binding_digests.sort();
        binding_digests.dedup();
        self.generation.get() == pin.generation()
            && self.registry_digest_sha256.as_str() == pin.registry_digest_sha256()
            && binding_digests == pin.binding_digests_sha256()
    }

    pub fn activate_native_provider_nodes(
        self: &Arc<Self>,
        registry: &mut NativeNodeRegistry,
        bridge: Arc<WorkerCapabilityBridge>,
    ) -> Result<(), WorkerPluginRuntimeError> {
        for (extension_id, component) in &self.components {
            let Some(binding_set) = component.manifest.provider_binding.clone() else {
                continue;
            };
            let nodes = component
                .manifest
                .nodes
                .iter()
                .map(|node| (node.id.as_str(), node))
                .collect::<BTreeMap<_, _>>();
            let mut bindings = Vec::with_capacity(binding_set.bindings.len());
            for claim in &binding_set.bindings {
                let node = nodes
                    .get(claim.node_id.as_str())
                    .copied()
                    .ok_or(WorkerPluginRuntimeError::InvalidDeployment)?
                    .clone();
                let descriptor = registry
                    .descriptor(&node.id)
                    .cloned()
                    .ok_or(WorkerPluginRuntimeError::InvalidDeployment)?;
                bindings.push(NativeProviderBindingActivation::new(
                    claim.clone(),
                    Arc::new(WorkerNativeProviderNode {
                        registry: self.clone(),
                        extension_id: extension_id.clone(),
                        implementation_version: node.version.to_string(),
                        descriptor,
                        node,
                        bridge: bridge.clone(),
                    }),
                ));
            }
            let activation = NativeProviderBindingActivationSet::checked(
                self.profile_id.0.to_string(),
                self.generation.get(),
                self.registry_digest_sha256.as_str().to_owned(),
                component.component_digest_sha256.as_str().to_owned(),
                component.authorization_generation.as_str().to_owned(),
                binding_set,
                bindings,
            )
            .map_err(|_| WorkerPluginRuntimeError::InvalidDeployment)?;
            registry
                .activate_provider_binding_set(activation)
                .map_err(|_| WorkerPluginRuntimeError::InvalidDeployment)?;
        }
        Ok(())
    }

    fn execute_provider_node(
        &self,
        extension_id: &str,
        node_id: &str,
        inputs: InvocationInputs,
        provider_request: Vec<u8>,
        bridge: Arc<dyn PluginCapabilityServices>,
        cancellation: CancellationToken,
    ) -> Result<ProviderInvocationResult, WorkerPluginRuntimeError> {
        let component = self
            .components
            .get(extension_id)
            .ok_or(WorkerPluginRuntimeError::MissingComponent)?;
        let binding_set = component
            .manifest
            .provider_binding
            .as_ref()
            .ok_or(WorkerPluginRuntimeError::InvalidDeployment)?;
        if !binding_set
            .bindings
            .iter()
            .any(|claim| claim.node_id == node_id)
        {
            return Err(WorkerPluginRuntimeError::InvalidDeployment);
        }
        let invocation_host = self.host.begin_invocation(
            &component.manifest,
            &component.authorization,
            node_id,
            inputs,
            bridge,
            cancellation,
        )?;
        let instance = self
            .host
            .instantiate_component(&component.compiled, invocation_host)?;
        instance
            .invoke_provider(node_id, &provider_request)
            .map_err(Into::into)
    }

    #[cfg(test)]
    pub(crate) fn empty_for_test(
        profile_id: ProfileId,
        generation: sim_types::WorkerRegistryGeneration,
        registry_digest_sha256: sim_types::WorkerSha256Digest,
    ) -> Result<Self, WorkerPluginRuntimeError> {
        let component_limits = ComponentLimits::default();
        let host = PluginHost::with_configuration(
            component_limits.clone(),
            sim_plugin_host::DEFAULT_API_FEATURES
                .iter()
                .map(|feature| (*feature).to_owned()),
        )?;
        Ok(Self {
            profile_id,
            generation,
            registry_digest_sha256,
            host: Arc::new(host),
            component_limits,
            components: BTreeMap::new(),
        })
    }

    pub fn execute(
        &self,
        invocation: WorkerPluginInvocation,
        bridge: Arc<WorkerCapabilityBridge>,
        cancellation: CancellationToken,
    ) -> Result<WorkerPluginInvocationResult, WorkerPluginRuntimeError> {
        if invocation.registry_generation() != self.generation
            || invocation.registry_digest_sha256() != &self.registry_digest_sha256
        {
            return Err(WorkerPluginRuntimeError::StaleGeneration);
        }
        let component = self
            .components
            .get(invocation.extension_id())
            .ok_or(WorkerPluginRuntimeError::MissingComponent)?;
        if invocation.component_digest_sha256() != &component.component_digest_sha256
            || invocation.extension_version() != component.extension_version
            || invocation.plugin_identifier() != component.plugin_identifier
            || invocation.plugin_version() != component.plugin_version
            || invocation.manifest_digest_sha256() != &component.manifest_digest_sha256
            || invocation.authorization_generation() != &component.authorization_generation
            || component.authorization.capabilities().profile_id() != self.profile_id.0.to_string()
        {
            return Err(WorkerPluginRuntimeError::StaleGeneration);
        }
        let node_id = invocation.node_id().to_owned();
        let (inputs, provider_request, provider_v2) = invocation.into_execution_parts();
        let node_is_provider_bound =
            component
                .manifest
                .provider_binding
                .as_ref()
                .is_some_and(|binding| {
                    binding
                        .bindings
                        .iter()
                        .any(|claim| claim.node_id == node_id)
                });
        if component.provider_v2 {
            if !node_is_provider_bound || provider_request.is_some() || provider_v2.is_none() {
                return Err(WorkerPluginRuntimeError::InvalidDeployment);
            }
        } else if provider_v2.is_some() || node_is_provider_bound != provider_request.is_some() {
            return Err(WorkerPluginRuntimeError::InvalidDeployment);
        }
        let invocation_host = self.host.begin_invocation(
            &component.manifest,
            &component.authorization,
            &node_id,
            inputs,
            bridge.clone(),
            cancellation.clone(),
        )?;
        if let Some(provider_v2) = provider_v2 {
            return self
                .host
                .invoke_provider_component_v2_for_worker(
                    &component.compiled,
                    invocation_host,
                    &node_id,
                    provider_v2,
                    cancellation,
                    bridge.provider_v2_sender()?,
                )
                .map(WorkerPluginInvocationResult::ProviderV2)
                .map_err(Into::into);
        }
        let mut instance = self
            .host
            .instantiate_component(&component.compiled, invocation_host)?;
        if let Some(provider_request) = provider_request {
            return instance
                .invoke_provider(&node_id, &provider_request)
                .map(WorkerPluginInvocationResult::Provider)
                .map_err(Into::into);
        }
        let node = instance.create_node(&node_id)?;
        if let Err(error) = instance.invoke(node) {
            instance.abort();
            return Err(error.into());
        }
        if let Err(error) = instance.drop_node(node) {
            instance.abort();
            return Err(error.into());
        }
        instance
            .finish()
            .map(WorkerPluginInvocationResult::Node)
            .map_err(Into::into)
    }
}

fn digest_matches(bytes: &[u8], expected: &sim_types::WorkerSha256Digest) -> bool {
    let actual: [u8; 32] = Sha256::digest(bytes).into();
    actual == expected.bytes()
}

pub(crate) fn encode_plugin_task_event(
    result: Result<WorkerPluginInvocationResult, WorkerPluginRuntimeError>,
) -> WorkerPluginTaskEvent {
    match result {
        Ok(WorkerPluginInvocationResult::ProviderV2(pending)) => {
            match serde_json::to_vec(pending.result()) {
                Ok(bytes) if !bytes.is_empty() && bytes.len() <= MAX_WORKER_PLUGIN_RESULT_BYTES => {
                    WorkerPluginTaskEvent::ProviderV2Proposal {
                        outcome: WorkerPluginExecutionOutcome::Succeeded(bytes),
                        pending,
                    }
                }
                Ok(_) | Err(_) => WorkerPluginTaskEvent::Terminal(
                    WorkerPluginExecutionOutcome::Failed(WorkerPluginExecutionFailure::HostFailure),
                ),
            }
        }
        result => WorkerPluginTaskEvent::Terminal(encode_plugin_outcome(result)),
    }
}

pub(crate) fn encode_plugin_outcome(
    result: Result<WorkerPluginInvocationResult, WorkerPluginRuntimeError>,
) -> WorkerPluginExecutionOutcome {
    match result {
        Ok(WorkerPluginInvocationResult::ProviderV2(_)) => {
            WorkerPluginExecutionOutcome::Failed(WorkerPluginExecutionFailure::HostFailure)
        }
        Ok(result) => match match result {
            WorkerPluginInvocationResult::Node(result) => serde_json::to_vec(&result),
            WorkerPluginInvocationResult::Provider(result) => serde_json::to_vec(&result),
            WorkerPluginInvocationResult::ProviderV2(_) => {
                return WorkerPluginExecutionOutcome::Failed(
                    WorkerPluginExecutionFailure::HostFailure,
                );
            }
        } {
            Ok(bytes) if !bytes.is_empty() && bytes.len() <= MAX_WORKER_PLUGIN_RESULT_BYTES => {
                WorkerPluginExecutionOutcome::Succeeded(bytes)
            }
            Ok(_) | Err(_) => {
                WorkerPluginExecutionOutcome::Failed(WorkerPluginExecutionFailure::HostFailure)
            }
        },
        Err(error) => WorkerPluginExecutionOutcome::Failed(error.failure()),
    }
}

#[derive(Debug, Error)]
pub(crate) enum WorkerPluginRuntimeError {
    #[error("worker component deployment is invalid")]
    InvalidDeployment,
    #[error("worker plugin invocation targets a stale component generation")]
    StaleGeneration,
    #[error("worker plugin invocation targets an unavailable component")]
    MissingComponent,
    #[error("worker native provider capability session failed")]
    InvocationFailed,
    #[error(transparent)]
    Trust(#[from] sim_runtime::TrustError),
    #[error(transparent)]
    Plugin(#[from] PluginError),
}

impl WorkerPluginRuntimeError {
    pub(crate) const fn deployment_rejection_reason(
        &self,
    ) -> WorkerRegistryDeploymentRejectionReason {
        match self {
            Self::InvalidDeployment
            | Self::StaleGeneration
            | Self::MissingComponent
            | Self::InvocationFailed => WorkerRegistryDeploymentRejectionReason::InvalidCandidate,
            Self::Trust(_) => WorkerRegistryDeploymentRejectionReason::InvalidAuthorization,
            Self::Plugin(_) => WorkerRegistryDeploymentRejectionReason::ComponentCompilationFailed,
        }
    }

    fn failure(&self) -> WorkerPluginExecutionFailure {
        match self {
            Self::StaleGeneration
            | Self::MissingComponent
            | Self::InvalidDeployment
            | Self::InvocationFailed => WorkerPluginExecutionFailure::InvalidInvocation,
            Self::Plugin(PluginError::WasmTrap(diagnostic)) => WorkerPluginExecutionFailure::Trap {
                diagnostic: diagnostic.clone(),
            },
            Self::Plugin(PluginError::Invocation(InvocationError::Cancelled)) => {
                WorkerPluginExecutionFailure::Cancelled
            }
            Self::Plugin(PluginError::Invocation(InvocationError::TimedOut)) => {
                WorkerPluginExecutionFailure::TimedOut
            }
            Self::Plugin(PluginError::ProviderStreaming(WorkerProviderStreamError::Cancelled)) => {
                WorkerPluginExecutionFailure::Cancelled
            }
            Self::Plugin(PluginError::ProviderStreaming(WorkerProviderStreamError::TimedOut)) => {
                WorkerPluginExecutionFailure::TimedOut
            }
            Self::Plugin(PluginError::Invocation(InvocationError::CapabilityDenied { .. })) => {
                WorkerPluginExecutionFailure::CapabilityDenied
            }
            Self::Trust(_) | Self::Plugin(_) => WorkerPluginExecutionFailure::HostFailure,
        }
    }
}

fn map_wire_failure(
    failure: PluginServiceWireFailure,
    kind: CapabilityKind,
    scope: &str,
) -> InvocationError {
    match failure {
        PluginServiceWireFailure::CapabilityDenied | PluginServiceWireFailure::ProviderDenied => {
            InvocationError::CapabilityDenied {
                kind,
                scope: scope.to_owned(),
            }
        }
        PluginServiceWireFailure::Cancelled => InvocationError::Cancelled,
        PluginServiceWireFailure::DeadlineExceeded => InvocationError::TimedOut,
        PluginServiceWireFailure::ResponseTooLarge => InvocationError::QuotaExceeded {
            kind,
            limit: "response-byte".to_owned(),
        },
        PluginServiceWireFailure::InvalidRequest => InvocationError::InvalidCapabilityRequest(
            "canonical plugin capability request was rejected".to_owned(),
        ),
        PluginServiceWireFailure::ServiceUnavailable
        | PluginServiceWireFailure::ActuatorFailed
        | PluginServiceWireFailure::RandomnessFailed
        | PluginServiceWireFailure::InvocationFailed => {
            InvocationError::HostFailure("canonical plugin capability service failed".to_owned())
        }
    }
}

fn invalid_wire_response(service: &str) -> InvocationError {
    InvocationError::HostFailure(format!(
        "canonical {service} service returned an incompatible response"
    ))
}

fn context_required(service: &str) -> InvocationError {
    InvocationError::HostFailure(format!(
        "canonical {service} service requires invocation context"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_stream_terminal_dispositions_preserve_typed_worker_failures() {
        assert_eq!(
            WorkerPluginRuntimeError::Plugin(PluginError::ProviderStreaming(
                WorkerProviderStreamError::Cancelled,
            ))
            .failure(),
            WorkerPluginExecutionFailure::Cancelled
        );
        assert_eq!(
            WorkerPluginRuntimeError::Plugin(PluginError::ProviderStreaming(
                WorkerProviderStreamError::TimedOut,
            ))
            .failure(),
            WorkerPluginExecutionFailure::TimedOut
        );
        assert_eq!(
            WorkerPluginRuntimeError::Plugin(PluginError::ProviderStreaming(
                WorkerProviderStreamError::InvalidOrder,
            ))
            .failure(),
            WorkerPluginExecutionFailure::HostFailure
        );
    }

    #[test]
    fn native_provider_capability_bridge_routes_the_complete_session_protocol() {
        let (sender, receiver) = async_channel::bounded(1);
        let bridge = WorkerCapabilityBridge::new(sender);
        let service = std::thread::spawn(move || {
            let mut previous_call_id = 0;
            for expected in ["begin", "call", "resolve", "finish", "abort"] {
                let request = receiver
                    .recv_blocking()
                    .expect("native provider bridge request is available");
                assert!(request.call_id > previous_call_id);
                previous_call_id = request.call_id;
                let request_value = NativeProviderWorkerRequest::from_bytes(&request.request)
                    .expect("native provider request is canonical");
                let response = match (expected, request_value) {
                    ("begin", NativeProviderWorkerRequest::Begin(start)) => {
                        assert_eq!(start.session_id, "provider-session");
                        NativeProviderWorkerResponse::Begun
                    }
                    (
                        "call",
                        NativeProviderWorkerRequest::Call {
                            session_id,
                            request,
                        },
                    ) => {
                        assert_eq!(session_id, "provider-session");
                        assert_eq!(request, b"provider-call");
                        NativeProviderWorkerResponse::Call(b"provider-response".to_vec())
                    }
                    (
                        "resolve",
                        NativeProviderWorkerRequest::Resolve {
                            session_id,
                            receipt_set,
                        },
                    ) => {
                        assert_eq!(session_id, "provider-session");
                        assert_eq!(receipt_set, b"receipt-set");
                        NativeProviderWorkerResponse::Resolved(vec![b"materialized".to_vec()])
                    }
                    ("finish", NativeProviderWorkerRequest::Finish { session_id }) => {
                        assert_eq!(session_id, "provider-session");
                        NativeProviderWorkerResponse::Finished
                    }
                    ("abort", NativeProviderWorkerRequest::Abort { session_id }) => {
                        assert_eq!(session_id, "provider-session");
                        NativeProviderWorkerResponse::Aborted
                    }
                    _ => panic!("native provider session phase changed"),
                };
                request
                    .response_sender
                    .send_blocking(
                        response
                            .to_bytes()
                            .expect("native provider response is canonical"),
                    )
                    .expect("native provider bridge receives its response");
            }
        });
        let cancellation = CancellationToken::default();
        let begin =
            NativeProviderWorkerRequest::Begin(sim_runtime::NativeProviderWorkerSessionStart {
                session_id: "provider-session".to_owned(),
                registry_generation: 7,
                registry_digest_sha256: "a".repeat(64),
                extension_id: "provider-extension".to_owned(),
                extension_version: "1.0.0".to_owned(),
                plugin_identifier: "provider.plugin".to_owned(),
                plugin_version: "1.0.0".to_owned(),
                manifest_digest_sha256: "b".repeat(64),
                component_digest_sha256: "c".repeat(64),
                authorization_generation_sha256: "d".repeat(64),
                binding_set_sha256: "e".repeat(64),
                node_id: "ProviderNode".to_owned(),
                compiled_plan_sha256: "f".repeat(64),
                maximum_response_bytes: 1_024,
            });
        assert_eq!(
            bridge
                .native_provider_control(begin, &cancellation)
                .expect("provider session begins"),
            NativeProviderWorkerResponse::Begun
        );
        assert_eq!(
            bridge
                .native_provider_control(
                    NativeProviderWorkerRequest::Call {
                        session_id: "provider-session".to_owned(),
                        request: b"provider-call".to_vec(),
                    },
                    &cancellation,
                )
                .expect("provider session routes a capability call"),
            NativeProviderWorkerResponse::Call(b"provider-response".to_vec())
        );
        assert_eq!(
            bridge
                .native_provider_control(
                    NativeProviderWorkerRequest::Resolve {
                        session_id: "provider-session".to_owned(),
                        receipt_set: b"receipt-set".to_vec(),
                    },
                    &cancellation,
                )
                .expect("provider session resolves app-owned receipts"),
            NativeProviderWorkerResponse::Resolved(vec![b"materialized".to_vec()])
        );
        assert_eq!(
            bridge
                .native_provider_control(
                    NativeProviderWorkerRequest::Finish {
                        session_id: "provider-session".to_owned(),
                    },
                    &cancellation,
                )
                .expect("provider session finishes"),
            NativeProviderWorkerResponse::Finished
        );
        assert_eq!(
            bridge
                .native_provider_control(
                    NativeProviderWorkerRequest::Abort {
                        session_id: "provider-session".to_owned(),
                    },
                    &cancellation,
                )
                .expect("provider session abort is idempotently routable"),
            NativeProviderWorkerResponse::Aborted
        );
        service.join().expect("provider bridge service joins");
    }

    #[test]
    fn provider_registry_pin_requires_the_exact_committed_generation() {
        let registry = WorkerPluginRegistry::empty_for_test(
            ProfileId(uuid::Uuid::nil()),
            sim_types::WorkerRegistryGeneration::new(2).expect("generation is non-zero"),
            sim_types::WorkerSha256Digest::new("a".repeat(64)).expect("registry digest is valid"),
        )
        .expect("empty worker registry is valid");
        let pin = NativeProviderRegistryPin::checked(2, "a".repeat(64), vec!["b".repeat(64)])
            .expect("provider registry pin is valid");

        assert!(
            !registry.matches_provider_registry_pin(&pin),
            "an empty committed registry cannot satisfy a provider binding pin"
        );
    }

    #[test]
    fn stale_worker_generation_fails_before_component_or_capability_access() {
        let current_digest = sim_types::WorkerSha256Digest::new("b".repeat(64))
            .expect("current registry digest is valid");
        let component_limits = ComponentLimits::default();
        let host = PluginHost::with_configuration(
            component_limits.clone(),
            sim_plugin_host::DEFAULT_API_FEATURES
                .iter()
                .map(|feature| (*feature).to_owned()),
        )
        .expect("no-WASI plugin host is valid");
        let registry = WorkerPluginRegistry {
            profile_id: ProfileId(uuid::Uuid::nil()),
            generation: sim_types::WorkerRegistryGeneration::new(2)
                .expect("generation is non-zero"),
            registry_digest_sha256: current_digest,
            host: Arc::new(host),
            component_limits: component_limits.clone(),
            components: BTreeMap::new(),
        };
        let invocation_bytes = serde_json::to_vec(&serde_json::json!({
            "registry_generation": 1,
            "registry_digest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "extension_id": "test.echo-extension",
            "extension_version": "1.0.0",
            "plugin_identifier": "test.echo-plugin",
            "plugin_version": "1.0.0",
            "manifest_digest_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "component_digest_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "authorization_generation": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "node_id": "echo",
            "inputs": { "values": {} },
            "provider_request": [1, 2, 3],
            "timeout_milliseconds": 1_000,
            "maximum_response_bytes": 1_024,
            "component_limits": component_limits
        }))
        .expect("worker invocation fixture serializes");
        let invocation = WorkerPluginInvocation::from_bytes(&invocation_bytes)
            .expect("worker invocation fixture is valid");
        let round_trip = invocation
            .to_bytes()
            .and_then(|bytes| WorkerPluginInvocation::from_bytes(&bytes))
            .expect("tagged invocation values round trip through the bounded JSON DTO");
        assert_eq!(round_trip, invocation);
        assert_eq!(round_trip.provider_request(), Some([1, 2, 3].as_slice()));

        let (sender, _receiver) = async_channel::bounded(1);
        let result = registry.execute(
            invocation,
            Arc::new(WorkerCapabilityBridge::new(sender)),
            CancellationToken::default(),
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("stale invocation unexpectedly executed"),
        };
        assert!(matches!(error, WorkerPluginRuntimeError::StaleGeneration));
        assert_eq!(
            error.failure(),
            WorkerPluginExecutionFailure::InvalidInvocation
        );
    }
}
