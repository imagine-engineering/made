use crate::{
    ComponentLimits, InvocationInputs, InvocationResult, PluginCapabilityServices, PluginError,
    PluginHost, ProviderInvocationResult, WasmPluginInstance,
};
use extension_host::{ComponentLifecycleAdapter, ComponentRuntime, InstalledComponent};
use sha2::{Digest, Sha256};
use sim_nodes::NativeSchemaValue;
use sim_plugin_sdk::{PluginManifest, ProviderPluginManifestV2};
use sim_runtime::{
    NativeNodeRegistry, NativeProviderInvocationAuthority, NativeProviderInvocationScope,
    NativeProviderRegistryPin, NativeProviderWorkerV2Activation,
    NativeProviderWorkerV2RouteAuthority, NativeProviderWorkerV2RouteSession, NodeContext,
    PermissionPolicy, PluginAuthorization, PluginAuthorizationSealer, PluginAuthorizationVerifier,
    PluginCapabilityBroker, PluginCapabilityInvocation, PluginServiceError,
    PluginServiceInvocationContext, PluginTrustPolicy, ProviderCostAuthorizationAuthority,
    ProviderManifestAuthorizationV2, ProviderPolicy, ProviderResultReceiptAuthority,
    ProviderResultReceiptIssuer, WorkerProviderV2InvocationEnvelope, WorkerRegistryDeploymentPlan,
};
#[cfg(feature = "test-support")]
use sim_types::WorkerPluginExecutionOutcome;
use sim_types::{
    CancellationToken, MAX_WORKER_COMPONENT_CHUNK_BYTES,
    MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES, MAX_WORKER_PLUGIN_INVOCATION_BYTES,
    WorkerComponentContent, WorkerComponentDescriptor, WorkerProviderInvocationContext,
    WorkerRegistryDeploymentBegin, WorkerRegistryDeploymentChunk, WorkerRegistryGeneration,
    WorkerSha256Digest,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    pin::Pin,
    sync::{Arc, Condvar, Mutex, RwLock},
    time::{Duration, Instant},
};
use thiserror::Error;

pub const SIM_COMPONENT_ADAPTER_ID: &str = "zed.sim.component-host.v1";
pub const MAX_WORKER_PLUGIN_TIMEOUT_MILLISECONDS: u64 = 60_000;

#[derive(Debug, Error)]
pub enum ComponentHostError {
    #[error("component invocation was cancelled")]
    Cancelled,
    #[error("component host state is unavailable")]
    StateUnavailable,
    #[error("component manifest for extension `{extension_id}` is invalid: {message}")]
    InvalidManifest {
        extension_id: Arc<str>,
        message: String,
    },
    #[error("component verification for extension `{extension_id}` failed: {message}")]
    Verification {
        extension_id: Arc<str>,
        message: String,
    },
    #[error("extension `{0}` repeated a component identity")]
    DuplicatePlugin(String),
    #[error("component node `{0}` is owned by more than one installed component")]
    DuplicateNode(String),
    #[error("extension `{0}` has no active verified component")]
    MissingExtension(String),
    #[error("component handle for extension `{0}` was revoked")]
    Revoked(String),
    #[error("plugin execution boundary failed: {0}")]
    ExecutionBoundary(String),
    #[error(
        "installed extension `{extension_id}` version `{extension_version}` does not match signed plugin version `{plugin_version}`"
    )]
    ExtensionVersionMismatch {
        extension_id: Arc<str>,
        extension_version: Arc<str>,
        plugin_version: String,
    },
    #[error(transparent)]
    Plugin(#[from] PluginError),
}

pub trait PluginInvocationExecutor: Send + Sync {
    fn execute<'a>(
        &'a self,
        invocation: PreparedPluginInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<InvocationResult, ComponentHostError>> + Send + 'a>>;

    fn execute_provider<'a>(
        &'a self,
        invocation: PreparedPluginInvocation,
    ) -> Pin<
        Box<dyn Future<Output = Result<ProviderInvocationResult, ComponentHostError>> + Send + 'a>,
    >;

    fn execute_provider_v2<'a>(
        &'a self,
        invocation: PreparedPluginInvocation,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<sim_runtime::ProviderTransportResponse, ComponentHostError>>
                + Send
                + 'a,
        >,
    >;
}

#[derive(Clone)]
pub enum ComponentExecutionBoundary {
    PrivateWorker(Arc<dyn PluginInvocationExecutor>),
    ConformanceInProcess(Arc<dyn PluginCapabilityServices>),
}

impl ComponentExecutionBoundary {
    pub fn private_worker(executor: Arc<dyn PluginInvocationExecutor>) -> Self {
        Self::PrivateWorker(executor)
    }

    pub fn conformance_in_process(services: Arc<dyn PluginCapabilityServices>) -> Self {
        Self::ConformanceInProcess(services)
    }
}

struct VerifiedPlugin {
    binding: InstalledComponentBinding,
    manifest: Arc<PluginManifest>,
    authorization: Arc<PluginAuthorization>,
    provider_manifest_v2: Option<Arc<ProviderPluginManifestV2>>,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes retained provider-v2 authorization"
        )
    )]
    provider_authorization_v2: Option<Arc<ProviderManifestAuthorizationV2>>,
    compiled: Arc<crate::CompiledPlugin>,
    manifest_bytes: Arc<[u8]>,
    component_bytes: Arc<[u8]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedComponentDeployment {
    extension_id: Arc<str>,
    extension_version: Arc<str>,
    plugin_identifier: Arc<str>,
    plugin_version: Arc<str>,
    manifest_sha256: Arc<str>,
    component_sha256: Arc<str>,
    authorization_generation: Arc<str>,
    manifest_bytes: Arc<[u8]>,
    authorization_bytes: Arc<[u8]>,
    component_bytes: Arc<[u8]>,
}

impl VerifiedComponentDeployment {
    pub fn extension_id(&self) -> &str {
        &self.extension_id
    }

    pub fn extension_version(&self) -> &str {
        &self.extension_version
    }

    pub fn plugin_identifier(&self) -> &str {
        &self.plugin_identifier
    }

    pub fn plugin_version(&self) -> &str {
        &self.plugin_version
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn component_sha256(&self) -> &str {
        &self.component_sha256
    }

    pub fn authorization_generation(&self) -> &str {
        &self.authorization_generation
    }

    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }

    pub fn component_bytes(&self) -> &[u8] {
        &self.component_bytes
    }

    pub fn authorization_bytes(&self) -> &[u8] {
        &self.authorization_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedComponentGeneration {
    profile_id: Arc<str>,
    generation: u64,
    snapshot_sha256: Arc<str>,
    authorization_verifier: PluginAuthorizationVerifier,
    components: Arc<[VerifiedComponentDeployment]>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPluginInvocation {
    registry_generation: WorkerRegistryGeneration,
    registry_digest_sha256: WorkerSha256Digest,
    extension_id: String,
    extension_version: String,
    plugin_identifier: String,
    plugin_version: String,
    manifest_digest_sha256: WorkerSha256Digest,
    component_digest_sha256: WorkerSha256Digest,
    authorization_generation: WorkerSha256Digest,
    node_id: String,
    inputs: InvocationInputs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_request: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_v2: Option<WorkerProviderV2InvocationEnvelope>,
    timeout_milliseconds: u64,
    maximum_response_bytes: u64,
    component_limits: ComponentLimits,
}

impl WorkerPluginInvocation {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ComponentHostError> {
        if bytes.is_empty() || bytes.len() > MAX_WORKER_PLUGIN_INVOCATION_BYTES {
            return Err(worker_deployment_error(
                "worker plugin invocation exceeds its transport bound",
            ));
        }
        let invocation: Self = serde_json::from_slice(bytes).map_err(worker_deployment_error)?;
        invocation.validate()?;
        Ok(invocation)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ComponentHostError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(worker_deployment_error)?;
        if bytes.is_empty() || bytes.len() > MAX_WORKER_PLUGIN_INVOCATION_BYTES {
            return Err(worker_deployment_error(
                "worker plugin invocation exceeds its transport bound",
            ));
        }
        Ok(bytes)
    }

    pub const fn registry_generation(&self) -> WorkerRegistryGeneration {
        self.registry_generation
    }

    pub fn registry_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.registry_digest_sha256
    }

    pub fn extension_id(&self) -> &str {
        &self.extension_id
    }

    pub fn extension_version(&self) -> &str {
        &self.extension_version
    }

    pub fn plugin_identifier(&self) -> &str {
        &self.plugin_identifier
    }

    pub fn plugin_version(&self) -> &str {
        &self.plugin_version
    }

    pub fn manifest_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.manifest_digest_sha256
    }

    pub fn component_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.component_digest_sha256
    }

    pub fn authorization_generation(&self) -> &WorkerSha256Digest {
        &self.authorization_generation
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn inputs(&self) -> &InvocationInputs {
        &self.inputs
    }

    pub fn into_inputs(self) -> InvocationInputs {
        self.inputs
    }

    pub fn provider_request(&self) -> Option<&[u8]> {
        self.provider_request.as_deref()
    }

    pub fn provider_v2(&self) -> Option<&WorkerProviderV2InvocationEnvelope> {
        self.provider_v2.as_ref()
    }

    pub fn with_provider_v2(
        mut self,
        provider_v2: WorkerProviderV2InvocationEnvelope,
    ) -> Result<Self, ComponentHostError> {
        if self.provider_request.is_some() || self.provider_v2.is_some() {
            return Err(worker_deployment_error(
                "worker provider invocation modes are mutually exclusive",
            ));
        }
        self.provider_v2 = Some(provider_v2);
        self.validate()?;
        Ok(self)
    }

    pub fn into_execution_parts(
        self,
    ) -> (
        InvocationInputs,
        Option<Vec<u8>>,
        Option<WorkerProviderV2InvocationEnvelope>,
    ) {
        (self.inputs, self.provider_request, self.provider_v2)
    }

    pub const fn timeout_milliseconds(&self) -> u64 {
        self.timeout_milliseconds
    }

    pub const fn maximum_response_bytes(&self) -> u64 {
        self.maximum_response_bytes
    }

    pub fn component_limits(&self) -> &ComponentLimits {
        &self.component_limits
    }

    fn validate(&self) -> Result<(), ComponentHostError> {
        if !valid_worker_plugin_identity(&self.extension_id)
            || !valid_worker_plugin_identity(&self.extension_version)
            || !valid_worker_plugin_identity(&self.plugin_identifier)
            || !valid_worker_plugin_identity(&self.plugin_version)
            || !valid_worker_plugin_identity(&self.node_id)
            || self.timeout_milliseconds == 0
            || self.timeout_milliseconds > MAX_WORKER_PLUGIN_TIMEOUT_MILLISECONDS
            || self.maximum_response_bytes == 0
            || self.maximum_response_bytes
                > u64::try_from(MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES)
                    .map_err(worker_deployment_error)?
            || self.provider_request.as_ref().is_some_and(|request| {
                request.is_empty() || request.len() > MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES
            })
            || (self.provider_request.is_some() && self.provider_v2.is_some())
        {
            return Err(worker_deployment_error(
                "worker plugin invocation metadata is invalid",
            ));
        }
        self.component_limits.validate()?;
        if let Some(provider_v2) = &self.provider_v2 {
            provider_v2.validate().map_err(worker_deployment_error)?;
        }
        Ok(())
    }
}

impl VerifiedComponentGeneration {
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn snapshot_sha256(&self) -> &str {
        &self.snapshot_sha256
    }

    pub fn components(&self) -> &[VerifiedComponentDeployment] {
        &self.components
    }

    pub fn provider_registry_pin(
        &self,
    ) -> Result<Option<NativeProviderRegistryPin>, ComponentHostError> {
        let mut binding_digests = Vec::new();
        for component in self.components.iter() {
            let (manifest, _) =
                parse_component_manifest(component.manifest_bytes()).map_err(|error| {
                    ComponentHostError::InvalidManifest {
                        extension_id: component.extension_id.clone(),
                        message: error.to_string(),
                    }
                })?;
            if let Some(provider_binding) = manifest.provider_binding {
                binding_digests.push(provider_binding.bindings_sha256);
            }
        }
        if binding_digests.is_empty() {
            return Ok(None);
        }
        binding_digests.sort();
        binding_digests.dedup();
        let deployment = self.worker_deployment_plan()?;
        NativeProviderRegistryPin::checked(
            deployment.begin().generation().get(),
            deployment
                .begin()
                .registry_digest_sha256()
                .as_str()
                .to_owned(),
            binding_digests,
        )
        .map(Some)
        .map_err(worker_deployment_error)
    }

    pub fn prepare_worker_invocation(
        &self,
        extension_id: &str,
        node_id: &str,
        inputs: InvocationInputs,
        timeout_milliseconds: u64,
        maximum_response_bytes: u64,
        component_limits: ComponentLimits,
    ) -> Result<WorkerPluginInvocation, ComponentHostError> {
        self.prepare_worker_invocation_with_provider_request(
            extension_id,
            node_id,
            inputs,
            None,
            timeout_milliseconds,
            maximum_response_bytes,
            component_limits,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_worker_provider_invocation(
        &self,
        extension_id: &str,
        node_id: &str,
        inputs: InvocationInputs,
        provider_request: Vec<u8>,
        timeout_milliseconds: u64,
        maximum_response_bytes: u64,
        component_limits: ComponentLimits,
    ) -> Result<WorkerPluginInvocation, ComponentHostError> {
        self.prepare_worker_invocation_with_provider_request(
            extension_id,
            node_id,
            inputs,
            Some(provider_request),
            timeout_milliseconds,
            maximum_response_bytes,
            component_limits,
            false,
        )
    }

    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator prepares provider-v2 worker invocations"
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_worker_provider_v2_invocation(
        &self,
        extension_id: &str,
        node_id: &str,
        inputs: InvocationInputs,
        timeout_milliseconds: u64,
        maximum_response_bytes: u64,
        component_limits: ComponentLimits,
    ) -> Result<WorkerPluginInvocation, ComponentHostError> {
        self.prepare_worker_invocation_with_provider_request(
            extension_id,
            node_id,
            inputs,
            None,
            timeout_milliseconds,
            maximum_response_bytes,
            component_limits,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_worker_invocation_with_provider_request(
        &self,
        extension_id: &str,
        node_id: &str,
        inputs: InvocationInputs,
        provider_request: Option<Vec<u8>>,
        timeout_milliseconds: u64,
        maximum_response_bytes: u64,
        component_limits: ComponentLimits,
        provider_v2_route: bool,
    ) -> Result<WorkerPluginInvocation, ComponentHostError> {
        let component = self
            .components
            .iter()
            .find(|component| component.extension_id() == extension_id)
            .ok_or_else(|| ComponentHostError::MissingExtension(extension_id.to_owned()))?;
        let (manifest, provider_manifest_v2) = parse_component_manifest(component.manifest_bytes())
            .map_err(|error| ComponentHostError::InvalidManifest {
                extension_id: component.extension_id.clone(),
                message: error.to_string(),
            })?;
        if provider_manifest_v2.is_some() != provider_v2_route {
            return Err(ComponentHostError::ExecutionBoundary(
                "worker invocation route disagrees with the signed provider world".to_owned(),
            ));
        }
        if !manifest.nodes.iter().any(|node| node.id == node_id) {
            return Err(ComponentHostError::Plugin(PluginError::UndeclaredNode(
                node_id.to_owned(),
            )));
        }
        let node_is_provider_bound = manifest.provider_binding.as_ref().is_some_and(|binding| {
            binding
                .bindings
                .iter()
                .any(|claim| claim.node_id == node_id)
        });
        if node_is_provider_bound != (provider_request.is_some() || provider_v2_route) {
            return Err(ComponentHostError::ExecutionBoundary(
                "worker invocation mode disagrees with the signed provider binding".to_owned(),
            ));
        }
        let deployment = self.worker_deployment_plan()?;
        let descriptor = deployment
            .begin()
            .components()
            .iter()
            .find(|descriptor| descriptor.extension_id() == extension_id)
            .ok_or_else(|| worker_deployment_error("worker descriptor vanished"))?;
        let invocation = WorkerPluginInvocation {
            registry_generation: deployment.begin().generation(),
            registry_digest_sha256: deployment.begin().registry_digest_sha256().clone(),
            extension_id: extension_id.to_owned(),
            extension_version: descriptor.extension_version().to_owned(),
            plugin_identifier: descriptor.plugin_identifier().to_owned(),
            plugin_version: descriptor.plugin_version().to_owned(),
            manifest_digest_sha256: descriptor.manifest_digest_sha256().clone(),
            component_digest_sha256: descriptor.component_digest_sha256().clone(),
            authorization_generation: descriptor.authorization_generation().clone(),
            node_id: node_id.to_owned(),
            inputs,
            provider_request,
            provider_v2: None,
            timeout_milliseconds,
            maximum_response_bytes,
            component_limits,
        };
        invocation.validate()?;
        Ok(invocation)
    }

    pub fn worker_deployment_plan(
        &self,
    ) -> Result<WorkerRegistryDeploymentPlan, ComponentHostError> {
        let generation =
            WorkerRegistryGeneration::new(self.generation).map_err(worker_deployment_error)?;
        let mut descriptors = Vec::with_capacity(self.components.len());
        for component in self.components.iter() {
            descriptors.push(
                WorkerComponentDescriptor::new(
                    component.extension_id.to_string(),
                    component.extension_version.to_string(),
                    component.plugin_identifier.to_string(),
                    component.plugin_version.to_string(),
                    WorkerSha256Digest::new(component.authorization_generation.to_string())
                        .map_err(worker_deployment_error)?,
                    WorkerSha256Digest::new(component.manifest_sha256.to_string())
                        .map_err(worker_deployment_error)?,
                    WorkerSha256Digest::new(component.component_sha256.to_string())
                        .map_err(worker_deployment_error)?,
                    u64::try_from(component.manifest_bytes.len())
                        .map_err(|error| worker_deployment_error(error.to_string()))?,
                    u64::try_from(component.authorization_bytes.len())
                        .map_err(|error| worker_deployment_error(error.to_string()))?,
                    u64::try_from(component.component_bytes.len())
                        .map_err(|error| worker_deployment_error(error.to_string()))?,
                )
                .map_err(worker_deployment_error)?,
            );
        }
        descriptors.sort_by(|left, right| {
            (left.extension_id(), left.component_digest_sha256())
                .cmp(&(right.extension_id(), right.component_digest_sha256()))
        });
        let provisional = WorkerRegistryDeploymentBegin::new(
            generation,
            WorkerSha256Digest::new("0".repeat(64)).map_err(worker_deployment_error)?,
            descriptors,
        )
        .map_err(worker_deployment_error)?;
        let registry_digest = hex_sha256(&provisional.digest_material());
        let begin = WorkerRegistryDeploymentBegin::new(
            generation,
            WorkerSha256Digest::new(registry_digest).map_err(worker_deployment_error)?,
            provisional.components().to_vec(),
        )
        .map_err(worker_deployment_error)?;
        let mut chunks = Vec::new();
        for (component_index, descriptor) in begin.components().iter().enumerate() {
            let component_index = u32::try_from(component_index)
                .map_err(|error| worker_deployment_error(error.to_string()))?;
            let component = self
                .components
                .iter()
                .find(|component| component.extension_id() == descriptor.extension_id())
                .ok_or_else(|| worker_deployment_error("verified component identity vanished"))?;
            append_worker_chunks(
                &mut chunks,
                generation,
                component_index,
                WorkerComponentContent::Manifest,
                component.manifest_bytes(),
            )?;
            append_worker_chunks(
                &mut chunks,
                generation,
                component_index,
                WorkerComponentContent::Authorization,
                component.authorization_bytes(),
            )?;
            append_worker_chunks(
                &mut chunks,
                generation,
                component_index,
                WorkerComponentContent::Component,
                component.component_bytes(),
            )?;
        }
        WorkerRegistryDeploymentPlan::new(begin, chunks, self.authorization_verifier.clone())
            .map_err(worker_deployment_error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstalledComponentBinding {
    lifecycle_extension_id: Arc<str>,
    lifecycle_extension_version: Arc<str>,
    signed_plugin_identifier: Arc<str>,
    signed_plugin_version: Arc<str>,
    signed_digest_sha256: Arc<str>,
    signed_provenance_source: Arc<str>,
    signed_provenance_publisher: Arc<str>,
}

impl InstalledComponentBinding {
    fn checked(
        component: &InstalledComponent,
        manifest: &PluginManifest,
    ) -> Result<Self, ComponentHostError> {
        let extension_id: Arc<str> = Arc::from(component.extension_id());
        let extension_version: Arc<str> = Arc::from(component.extension_version());
        let signed_plugin_version = manifest.plugin_version.to_string();
        if extension_version.as_ref() != signed_plugin_version {
            return Err(ComponentHostError::ExtensionVersionMismatch {
                extension_id,
                extension_version,
                plugin_version: signed_plugin_version,
            });
        }
        Ok(Self {
            lifecycle_extension_id: extension_id,
            lifecycle_extension_version: extension_version,
            signed_plugin_identifier: Arc::from(manifest.identifier.as_str()),
            signed_plugin_version: Arc::from(signed_plugin_version),
            signed_digest_sha256: Arc::from(manifest.digest_sha256.as_str()),
            signed_provenance_source: Arc::from(manifest.provenance.source.as_str()),
            signed_provenance_publisher: Arc::from(manifest.provenance.publisher.as_str()),
        })
    }

    pub fn lifecycle_extension_id(&self) -> &str {
        &self.lifecycle_extension_id
    }

    pub fn lifecycle_extension_version(&self) -> &str {
        &self.lifecycle_extension_version
    }

    pub fn signed_plugin_identifier(&self) -> &str {
        &self.signed_plugin_identifier
    }

    pub fn signed_plugin_version(&self) -> &str {
        &self.signed_plugin_version
    }

    pub fn signed_digest_sha256(&self) -> &str {
        &self.signed_digest_sha256
    }

    pub fn signed_provenance_source(&self) -> &str {
        &self.signed_provenance_source
    }

    pub fn signed_provenance_publisher(&self) -> &str {
        &self.signed_provenance_publisher
    }
}

#[derive(Clone)]
pub struct InstalledVerifiedPlugin {
    inner: Arc<VerifiedPlugin>,
}

impl InstalledVerifiedPlugin {
    pub fn extension_id(&self) -> &str {
        self.inner.binding.lifecycle_extension_id()
    }

    pub fn extension_version(&self) -> &str {
        self.inner.binding.signed_plugin_version()
    }

    pub fn binding(&self) -> &InstalledComponentBinding {
        &self.inner.binding
    }

    pub fn manifest(&self) -> &PluginManifest {
        &self.inner.manifest
    }

    pub fn authorization(&self) -> &PluginAuthorization {
        &self.inner.authorization
    }

    pub(crate) fn is_provider_v2(&self) -> bool {
        self.inner.provider_manifest_v2.is_some()
    }

    #[expect(
        dead_code,
        reason = "production consumer is sim-parity-provider-worker-stream-bridge"
    )]
    pub(crate) fn provider_manifest_v2(&self) -> Option<&ProviderPluginManifestV2> {
        self.inner.provider_manifest_v2.as_deref()
    }

    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes retained provider-v2 authorization"
        )
    )]
    pub(crate) fn provider_authorization_v2(&self) -> Option<&ProviderManifestAuthorizationV2> {
        self.inner.provider_authorization_v2.as_deref()
    }

    fn require_legacy_invocation_route(&self) -> Result<(), ComponentHostError> {
        if self.is_provider_v2() {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider-v8 worker routing is unavailable until the canonical bridge is active"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

pub struct PreparedPluginInvocation {
    worker_invocation: WorkerPluginInvocation,
    deployment: WorkerRegistryDeploymentPlan,
    authorization: PluginAuthorization,
    context: NodeContext,
    plugin: InstalledVerifiedPlugin,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the provider-v2 component generation"
        )
    )]
    generation: VerifiedComponentGeneration,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the provider-v2 plugin host"
        )
    )]
    plugin_host: Arc<PluginHost>,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the provider-v2 lease host"
        )
    )]
    lease_host: Arc<ComponentHostInner>,
    provider_price_badge: Option<NativeSchemaValue>,
    _lease: Option<InvocationLease>,
}

pub(crate) struct PreflightedProviderComponentCapsule {
    route_authority: NativeProviderWorkerV2RouteAuthority,
    envelope: WorkerProviderV2InvocationEnvelope,
    manifest_authorization: ProviderManifestAuthorizationV2,
    plugin: InstalledVerifiedPlugin,
    generation: VerifiedComponentGeneration,
    node_id: Arc<str>,
    plugin_host: Arc<PluginHost>,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the preflighted worker invocation"
        )
    )]
    worker_invocation: WorkerPluginInvocation,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the preflighted deployment"
        )
    )]
    deployment: WorkerRegistryDeploymentPlan,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the preflighted authorization"
        )
    )]
    authorization: PluginAuthorization,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the preflighted node context"
        )
    )]
    context: NodeContext,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the preflighted price badge"
        )
    )]
    provider_price_badge: Option<NativeSchemaValue>,
    _lease: InvocationLease,
}

#[cfg_attr(
    not(feature = "test-support"),
    expect(
        dead_code,
        reason = "Task427 deployment actuator consumes prepared provider-v2 worker execution"
    )
)]
pub(crate) struct PreparedProviderV2WorkerExecution {
    worker_invocation: WorkerPluginInvocation,
    deployment: WorkerRegistryDeploymentPlan,
    authorization: PluginAuthorization,
    context: NodeContext,
    plugin: InstalledVerifiedPlugin,
    generation: VerifiedComponentGeneration,
    route_authority: NativeProviderWorkerV2RouteAuthority,
    provider_price_badge: Option<NativeSchemaValue>,
    _lease: InvocationLease,
}

#[cfg(feature = "test-support")]
pub(crate) struct PreparedProviderV2SupervisorExecution {
    invocation: Vec<u8>,
    prompt_id: sim_types::PromptId,
    attempt_id: sim_types::AttemptId,
    bridge: sim_runtime::NativeProviderWorkerV2SupervisorBridge,
    capability_invocation: PluginCapabilityInvocation,
    _deployment: WorkerRegistryDeploymentPlan,
    _authorization: PluginAuthorization,
    _context: NodeContext,
    _plugin: InstalledVerifiedPlugin,
    _generation: VerifiedComponentGeneration,
    _provider_price_badge: Option<NativeSchemaValue>,
    _lease: InvocationLease,
}

impl PreparedProviderV2WorkerExecution {
    #[cfg(feature = "test-support")]
    pub fn into_supervised_parts(
        self,
        broker: &PluginCapabilityBroker,
    ) -> Result<
        (
            PreparedProviderV2SupervisorExecution,
            sim_runtime::NativeProviderWorkerV2ActuatorRoute,
        ),
        ComponentHostError,
    > {
        let timeout = Duration::from_millis(self.worker_invocation.timeout_milliseconds());
        let cancellation = self.context.cancellation.clone();
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
            ComponentHostError::ExecutionBoundary("provider-v2 deadline overflowed".to_owned())
        })?;
        let profile_id = uuid::Uuid::parse_str(self.authorization.capabilities().profile_id())
            .map(sim_types::ProfileId)
            .map_err(worker_deployment_error)?;
        let service_context = PluginServiceInvocationContext::new(
            profile_id,
            self.context.prompt_id,
            self.context.attempt_id,
            self.context.node_id.clone(),
            self.authorization.clone(),
            cancellation.clone(),
            deadline,
            self.worker_invocation.maximum_response_bytes(),
        )
        .map_err(worker_deployment_error)?;
        let capability_invocation = broker
            .begin_invocation(service_context)
            .map_err(worker_deployment_error)?;
        let (bridge, actuator) = self
            .route_authority
            .into_supervised_route(timeout, cancellation)
            .map_err(worker_deployment_error)?;
        let invocation = self.worker_invocation.to_bytes()?;
        Ok((
            PreparedProviderV2SupervisorExecution {
                invocation,
                prompt_id: self.context.prompt_id,
                attempt_id: self.context.attempt_id,
                bridge,
                capability_invocation,
                _deployment: self.deployment,
                _authorization: self.authorization,
                _context: self.context,
                _plugin: self.plugin,
                _generation: self.generation,
                _provider_price_badge: self.provider_price_badge,
                _lease: self._lease,
            },
            actuator,
        ))
    }
}

#[cfg(feature = "test-support")]
impl PreparedProviderV2SupervisorExecution {
    pub async fn execute(
        self,
        supervisor: &mut sim_runtime::RuntimeSupervisor,
    ) -> Result<
        (
            WorkerPluginExecutionOutcome,
            Option<sim_runtime::ProviderTransportResponse>,
        ),
        ComponentHostError,
    > {
        let mut capability_invocation = self.capability_invocation;
        let result = supervisor
            .execute_provider_v2(
                self.prompt_id,
                self.attempt_id,
                self.invocation,
                self.bridge,
                &mut capability_invocation,
            )
            .await
            .map_err(worker_deployment_error);
        if matches!(
            &result,
            Ok((WorkerPluginExecutionOutcome::Succeeded(_), Some(_)))
        ) {
            capability_invocation
                .finish()
                .map_err(worker_deployment_error)?;
        } else {
            capability_invocation.abort();
        }
        result
    }
}

impl PreflightedProviderComponentCapsule {
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes the provider-v2 preflight capsule"
        )
    )]
    pub(crate) fn new(
        prepared: PreparedPluginInvocation,
        activation: NativeProviderWorkerV2Activation,
        worker_start: &sim_runtime::NativeProviderWorkerSessionStart,
        manifest_authorization: ProviderManifestAuthorizationV2,
    ) -> Result<Self, ComponentHostError> {
        let PreparedPluginInvocation {
            worker_invocation,
            deployment,
            authorization,
            context,
            plugin,
            generation,
            plugin_host,
            lease_host,
            provider_price_badge,
            _lease,
        } = prepared;
        if _lease.is_some() {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider-v2 preflight received an invocation lease too early".to_owned(),
            ));
        }
        let node_id: Arc<str> = Arc::from(worker_invocation.node_id());
        if worker_start.node_id != node_id.as_ref()
            || worker_start.extension_id != plugin.extension_id()
            || deployment.begin().generation() != worker_invocation.registry_generation()
        {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider-v2 prepared invocation differs from its retained capsule".to_owned(),
            ));
        }
        if !plugin
            .manifest()
            .nodes
            .iter()
            .any(|node| node.id == node_id.as_ref())
        {
            return Err(ComponentHostError::Plugin(PluginError::UndeclaredNode(
                node_id.to_string(),
            )));
        }
        let preflight = activation
            .preflight_installed_component(
                &deployment,
                worker_start,
                manifest_authorization.clone(),
            )
            .map_err(|error| match error {
                PluginServiceError::Cancelled => ComponentHostError::Cancelled,
                error => ComponentHostError::ExecutionBoundary(format!(
                    "provider component activation preflight failed: {error}"
                )),
            })?;
        let lease = begin_invocation_lease_for_host(&lease_host, &plugin)?;
        let (envelope, route_authority) = preflight
            .into_transport_parts()
            .map_err(|error| ComponentHostError::ExecutionBoundary(error.to_string()))?;
        let worker_invocation = worker_invocation.with_provider_v2(envelope.clone())?;
        Ok(Self {
            route_authority,
            envelope,
            manifest_authorization,
            plugin,
            generation,
            node_id,
            plugin_host,
            worker_invocation,
            deployment,
            authorization,
            context,
            provider_price_badge,
            _lease: lease,
        })
    }

    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes preflighted provider-v2 execution"
        )
    )]
    pub(crate) fn into_worker_execution(self) -> PreparedProviderV2WorkerExecution {
        PreparedProviderV2WorkerExecution {
            worker_invocation: self.worker_invocation,
            deployment: self.deployment,
            authorization: self.authorization,
            context: self.context,
            plugin: self.plugin,
            generation: self.generation,
            route_authority: self.route_authority,
            provider_price_badge: self.provider_price_badge,
            _lease: self._lease,
        }
    }
}

#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
pub(crate) struct PreparedProviderV2Invocation {
    route_authority: Option<NativeProviderWorkerV2RouteAuthority>,
    worker_context: WorkerProviderInvocationContext,
    manifest_authorization: ProviderManifestAuthorizationV2,
    plugin: InstalledVerifiedPlugin,
    generation: VerifiedComponentGeneration,
    node_id: Arc<str>,
    route: crate::ProviderV2StreamRouteReceiver,
    runtime: crate::ProviderV2RuntimeHost,
    cancellation: CancellationToken,
    plugin_host: Arc<PluginHost>,
    _lease: InvocationLease,
}

#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
pub(crate) struct ProviderV2ComponentInvocation {
    instance: WasmPluginInstance,
    node_id: Arc<str>,
    _plugin: InstalledVerifiedPlugin,
    _generation: VerifiedComponentGeneration,
    _lease: InvocationLease,
}

#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
pub(crate) struct ProviderV2AppRoute {
    route_authority: Option<NativeProviderWorkerV2RouteAuthority>,
    worker_context: WorkerProviderInvocationContext,
    manifest_authorization: ProviderManifestAuthorizationV2,
    route: crate::ProviderV2StreamRouteReceiver,
}

#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
impl PreflightedProviderComponentCapsule {
    pub(crate) fn prepare_stream_route(
        self,
        cancellation: CancellationToken,
    ) -> Result<PreparedProviderV2Invocation, ComponentHostError> {
        let contract =
            crate::worker_streaming_contract(self.manifest_authorization.streaming_contract());
        let (route, receiver) = crate::provider_v2_stream_route();
        let runtime = crate::ProviderV2RuntimeHost::checked_from_certified_capsule(
            self.envelope.context().clone(),
            contract,
            cancellation.clone(),
            route,
        )
        .map_err(|error| ComponentHostError::ExecutionBoundary(error.to_string()))?;
        Ok(PreparedProviderV2Invocation {
            route_authority: Some(self.route_authority),
            worker_context: self.envelope.context().clone(),
            manifest_authorization: self.manifest_authorization,
            plugin: self.plugin,
            generation: self.generation,
            node_id: self.node_id,
            route: receiver,
            runtime,
            cancellation,
            plugin_host: self.plugin_host,
            _lease: self._lease,
        })
    }
}

#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
impl PreparedProviderV2Invocation {
    pub(crate) fn into_execution(
        self,
        inputs: InvocationInputs,
        services: Arc<dyn PluginCapabilityServices>,
    ) -> Result<(ProviderV2ComponentInvocation, ProviderV2AppRoute), ComponentHostError> {
        let invocation = self.plugin_host.begin_invocation(
            self.plugin.manifest(),
            self.plugin.authorization(),
            self.node_id.as_ref(),
            inputs,
            services,
            self.cancellation,
        )?;
        let instance = self.plugin_host.instantiate_provider_component_v2(
            &self.plugin.inner.compiled,
            invocation,
            self.runtime,
        )?;
        Ok((
            ProviderV2ComponentInvocation {
                instance,
                node_id: self.node_id,
                _plugin: self.plugin,
                _generation: self.generation,
                _lease: self._lease,
            },
            ProviderV2AppRoute {
                route_authority: self.route_authority,
                worker_context: self.worker_context,
                manifest_authorization: self.manifest_authorization,
                route: self.route,
            },
        ))
    }
}

#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
impl ProviderV2ComponentInvocation {
    pub(crate) fn invoke(self) -> Result<crate::ProviderV2InvocationProposal, ComponentHostError> {
        self.instance
            .invoke_provider_v2(self.node_id.as_ref())
            .map_err(ComponentHostError::from)
    }
}

#[expect(
    dead_code,
    reason = "production consumer is sim-parity-provider-worker-stream-bridge"
)]
impl ProviderV2AppRoute {
    pub(crate) fn bind_start_request(
        &mut self,
        call: crate::ProviderV2StreamRouteCall,
        policy: &ProviderPolicy,
    ) -> Result<
        (
            NativeProviderWorkerV2RouteSession,
            crate::ProviderV2BoundStartCall,
        ),
        ComponentHostError,
    > {
        let (call_id, context, head, reply) = call
            .into_start()
            .map_err(|error| ComponentHostError::ExecutionBoundary(error.to_string()))?;
        if context != self.worker_context {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider-v2 route context differs from its certified activation".to_owned(),
            ));
        }
        let session = self
            .route_authority
            .take()
            .ok_or_else(|| {
                ComponentHostError::ExecutionBoundary(
                    "provider-v2 activation grant was already consumed".to_owned(),
                )
            })?
            .start(head, policy)
            .map_err(|error| ComponentHostError::ExecutionBoundary(error.to_string()))?;
        Ok((session, crate::ProviderV2BoundStartCall { call_id, reply }))
    }

    pub(crate) fn try_receive_stream_call(
        &self,
    ) -> Result<crate::ProviderV2StreamRouteMessage, std::sync::mpsc::TryRecvError> {
        self.route.try_receive()
    }
}

#[cfg(test)]
mod activation_preflight_tests {
    #[test]
    fn provider_component_capsule_is_the_only_preflight_callsite() {
        let source = include_str!("component_host.rs");
        let production = source
            .split("#[cfg(test)]\nmod activation_preflight_tests")
            .next()
            .expect("component host production source is missing");
        let call = [".preflight_installed_", "component("].concat();
        assert_eq!(production.matches(&call).count(), 1);
        let fields = source
            .split("struct PreflightedProviderComponentCapsule")
            .nth(1)
            .and_then(|source| {
                source
                    .split("impl PreflightedProviderComponentCapsule")
                    .next()
            })
            .expect("private provider component capsule is missing");
        for retained in [
            "route_authority:",
            "envelope:",
            "manifest_authorization:",
            "plugin:",
            "generation:",
            "node_id:",
            "plugin_host:",
            "worker_invocation:",
            "deployment:",
            "authorization:",
            "context:",
            "_lease:",
        ] {
            assert!(fields.contains(retained));
        }
        assert!(!fields.contains("pub route_authority:"));
        assert!(!fields.contains("pub envelope:"));
        assert!(!fields.contains("pub manifest_authorization:"));
        let capsule = source
            .split("impl PreflightedProviderComponentCapsule")
            .nth(1)
            .and_then(|source| source.split("#[cfg(test)]").next())
            .expect("private provider component capsule implementation is missing");
        let constructor_signature = capsule
            .split("fn new(")
            .nth(1)
            .and_then(|source| source.split(") -> Result").next())
            .expect("private capsule constructor signature is missing");
        assert_eq!(constructor_signature.matches("prepared").count(), 1);
        assert_eq!(constructor_signature.matches("activation").count(), 1);
        let compact_capsule = capsule.split_whitespace().collect::<String>();
        assert!(compact_capsule.contains("plugin.manifest().nodes.iter().any"));
        assert!(compact_capsule.contains(
            "activation.preflight_installed_component(&deployment,worker_start,manifest_authorization.clone(),)"
        ));
        assert!(capsule.contains(&call));
        let ordered = [
            "let PreparedPluginInvocation",
            "node.id == node_id.as_ref()",
            call.as_str(),
            "into_transport_parts",
            "worker_invocation.with_provider_v2",
            "Ok(Self",
        ];
        let mut previous = 0;
        for marker in ordered {
            let position = capsule
                .find(marker)
                .expect("capsule ordering marker is missing");
            assert!(position >= previous);
            previous = position;
        }
        for forbidden in [
            "pub fn",
            "begin_invocation(",
            "get_input_state",
            "read_scalar_input",
            "take_input",
            "read_handle",
            "instantiate_component",
            "create_node",
            ".invoke(",
        ] {
            assert!(!capsule.contains(forbidden));
        }
    }

    #[test]
    fn every_legacy_component_entrypoint_uses_the_early_provider_v2_guard() {
        let source = include_str!("component_host.rs");
        assert!(source.matches("require_legacy_invocation_route()?").count() >= 4);
        for function in [
            "pub fn prepare_plugin_invocation",
            "pub fn prepare_provider_invocation",
            "pub fn execute_plugin",
            "pub fn invoke(",
        ] {
            let body = source
                .split(function)
                .nth(1)
                .and_then(|source| source.split("\n    pub fn ").next())
                .expect("legacy component entrypoint is missing");
            let guard = body
                .find("require_legacy_invocation_route")
                .expect("provider-v2 legacy-route guard is missing");
            for exposure in [
                "begin_invocation(",
                "instantiate_component(",
                "execute_prepared_plugin(",
                ".invoke(",
            ] {
                if let Some(position) = body.find(exposure) {
                    assert!(
                        guard < position,
                        "{function} exposes state before its v2 guard"
                    );
                }
            }
        }
    }

    #[test]
    fn provider_v2_prepared_invocation_has_one_consuming_selector_free_execution_split() {
        let source = include_str!("component_host.rs");
        let prepared = source
            .split("struct PreparedProviderV2Invocation")
            .nth(1)
            .and_then(|source| {
                source
                    .split("impl PreflightedProviderComponentCapsule")
                    .next()
            })
            .expect("prepared provider-v2 invocation fields are missing");
        for retained in [
            "route_authority:",
            "worker_context:",
            "manifest_authorization:",
            "plugin:",
            "generation:",
            "node_id:",
            "route:",
            "runtime:",
            "cancellation:",
            "plugin_host:",
            "_lease:",
        ] {
            assert!(
                prepared.contains(retained),
                "missing retained field {retained}"
            );
        }
        assert!(!prepared.contains("grant:"));
        assert!(!prepared.contains("ProviderRuntimeActivationGrant"));
        assert!(!prepared.contains("Clone"));

        let execution = source
            .split("fn into_execution(")
            .nth(1)
            .and_then(|source| source.split("impl ProviderV2ComponentInvocation").next())
            .expect("consuming provider-v2 execution split is missing");
        let signature = execution
            .split(") -> Result")
            .next()
            .expect("execution split signature is missing");
        for selector in [
            "CompiledPlugin",
            "WasmPluginInstance",
            "InvocationHost",
            "WorkerProviderInvocationContext",
            "ProviderManifestAuthorizationV2",
            "node_id",
            "generation",
            "plugin",
            "runtime",
        ] {
            assert!(
                !signature.contains(selector),
                "caller can replace {selector}"
            );
        }
        let compact = execution.split_whitespace().collect::<String>();
        assert!(compact.contains("self.plugin_host.begin_invocation(self.plugin.manifest(),self.plugin.authorization(),self.node_id.as_ref()"));
        assert!(compact.contains("&self.plugin.inner.compiled,invocation,self.runtime"));

        let production_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("component-host production source must exist");
        let host_source = include_str!("sim_plugin_host.rs");
        let production_host_source = host_source
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("plugin-host production source must exist");
        let component_host_instantiations = production_source
            .matches("instantiate_provider_component_v2(")
            .count();
        let plugin_host_instantiations = production_host_source
            .matches("instantiate_provider_component_v2(")
            .count();
        assert_eq!(component_host_instantiations, 1);
        assert_eq!(plugin_host_instantiations, 2);
        assert_eq!(
            component_host_instantiations + plugin_host_instantiations,
            3,
            "v2 instantiation must have one private definition and two certified callsites"
        );
        assert!(
            production_host_source.contains("pub(crate) fn instantiate_provider_component_v2(")
        );
        assert!(!production_host_source.contains("pub fn instantiate_provider_component_v2("));

        let worker_invocation = production_host_source
            .split("pub fn invoke_provider_component_v2_for_worker(")
            .nth(1)
            .and_then(|source| source.split("\n    fn new_wasm_store(").next())
            .expect("provider-v2 worker invocation owner is missing");
        let ordered_worker_gates = [
            ".provider_manifest_v2",
            "Sha256::digest(expected_manifest.signing_payload()?)",
            "envelope.provider_manifest_sha256().as_str() != expected_manifest_sha256",
            "envelope.streaming_contract()",
            "ProviderV2RuntimeHost::checked_for_worker_bridge(",
            "self.instantiate_provider_component_v2(",
        ];
        let mut previous = 0;
        for gate in ordered_worker_gates {
            let position = worker_invocation
                .find(gate)
                .unwrap_or_else(|| panic!("provider-v2 worker gate is missing: {gate}"));
            assert!(
                position >= previous,
                "provider-v2 worker gate is out of order: {gate}"
            );
            previous = position;
        }
    }
}

impl PreparedPluginInvocation {
    pub fn worker_invocation(&self) -> &WorkerPluginInvocation {
        &self.worker_invocation
    }

    pub fn deployment(&self) -> &WorkerRegistryDeploymentPlan {
        &self.deployment
    }

    pub fn authorization(&self) -> &PluginAuthorization {
        &self.authorization
    }

    pub fn context(&self) -> &NodeContext {
        &self.context
    }

    pub fn provider_binding_sha256(&self) -> Option<&str> {
        self.plugin
            .manifest()
            .provider_binding
            .as_ref()
            .map(|binding| binding.bindings_sha256.as_str())
    }

    pub(crate) fn provider_namespace(&self) -> Option<&str> {
        self.plugin
            .manifest()
            .provider_binding
            .as_ref()
            .map(|binding| binding.implementation_namespace.as_str())
    }

    pub(crate) fn provider_node(&self) -> Result<&sim_plugin_sdk::PluginNode, ComponentHostError> {
        self.plugin
            .manifest()
            .nodes
            .iter()
            .find(|node| node.id == self.worker_invocation.node_id())
            .ok_or_else(|| {
                ComponentHostError::ExecutionBoundary("signed provider node is absent".to_owned())
            })
    }

    pub fn provider_price_badge(&self) -> Option<&NativeSchemaValue> {
        self.provider_price_badge.as_ref()
    }

    pub(crate) fn is_provider_v2(&self) -> bool {
        self.plugin.is_provider_v2()
    }

    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes provider-v2 activation"
        )
    )]
    pub(crate) fn activate_provider_v2(
        self,
        attachment: &sim_runtime::NativeProviderWorkerBridgeAttachment,
    ) -> Result<PreparedProviderV2WorkerExecution, ComponentHostError> {
        if !self.plugin.is_provider_v2()
            || self.worker_invocation.provider_request().is_some()
            || self.worker_invocation.provider_v2().is_some()
        {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider-v2 worker activation mode is invalid".to_owned(),
            ));
        }
        let manifest_authorization = self
            .plugin
            .provider_authorization_v2()
            .cloned()
            .ok_or_else(|| {
                ComponentHostError::ExecutionBoundary(
                    "provider-v2 worker activation omitted signed authorization".to_owned(),
                )
            })?;
        let binding_set_sha256 = self
            .plugin
            .manifest()
            .provider_binding
            .as_ref()
            .map(|binding| binding.bindings_sha256.clone())
            .ok_or_else(|| {
                ComponentHostError::ExecutionBoundary(
                    "provider-v2 worker activation omitted its binding set".to_owned(),
                )
            })?;
        let compiled_plan_sha256 = self
            .context
            .provider_execution()
            .map_err(worker_deployment_error)?
            .compiled_plan_sha256()
            .to_owned();
        let worker_start = sim_runtime::NativeProviderWorkerSessionStart {
            session_id: "provider-v2-controller-owned".to_owned(),
            registry_generation: self.worker_invocation.registry_generation().get(),
            registry_digest_sha256: self
                .worker_invocation
                .registry_digest_sha256()
                .as_str()
                .to_owned(),
            extension_id: self.worker_invocation.extension_id().to_owned(),
            extension_version: self.worker_invocation.extension_version().to_owned(),
            plugin_identifier: self.worker_invocation.plugin_identifier().to_owned(),
            plugin_version: self.worker_invocation.plugin_version().to_owned(),
            manifest_digest_sha256: self
                .worker_invocation
                .manifest_digest_sha256()
                .as_str()
                .to_owned(),
            component_digest_sha256: self
                .worker_invocation
                .component_digest_sha256()
                .as_str()
                .to_owned(),
            authorization_generation_sha256: self
                .worker_invocation
                .authorization_generation()
                .as_str()
                .to_owned(),
            binding_set_sha256,
            node_id: self.worker_invocation.node_id().to_owned(),
            compiled_plan_sha256,
            maximum_response_bytes: self.worker_invocation.maximum_response_bytes(),
        };
        let profile_id = uuid::Uuid::parse_str(self.authorization.capabilities().profile_id())
            .map(sim_types::ProfileId)
            .map_err(worker_deployment_error)?;
        let activation = attachment
            .activate_provider_v2(
                profile_id,
                self.context.prompt_id,
                self.context.attempt_id,
                &self.context.node_id.0,
                self.worker_invocation.node_id(),
                &self.deployment,
                &worker_start,
                manifest_authorization.clone(),
                crate::worker_streaming_contract(manifest_authorization.streaming_contract()),
            )
            .map_err(worker_deployment_error)?;
        PreflightedProviderComponentCapsule::new(
            self,
            activation,
            &worker_start,
            manifest_authorization,
        )
        .map(PreflightedProviderComponentCapsule::into_worker_execution)
    }
}

struct ComponentState {
    by_extension: BTreeMap<Arc<str>, InstalledVerifiedPlugin>,
    node_owners: BTreeMap<String, Arc<str>>,
    registry: NativeNodeRegistry,
    generation: VerifiedComponentGeneration,
}

struct ComponentHostInner {
    plugin_host: Arc<PluginHost>,
    trust_policy: PluginTrustPolicy,
    permission_policy: PermissionPolicy,
    authorization_sealer: PluginAuthorizationSealer,
    executor: Arc<dyn PluginInvocationExecutor>,
    conformance_services: Option<Arc<dyn PluginCapabilityServices>>,
    invocation_timeout_milliseconds: u64,
    invocation_maximum_response_bytes: u64,
    base_registry: NativeNodeRegistry,
    state: RwLock<ComponentState>,
    invocation_gate: Mutex<InvocationGate>,
    invocation_gate_changed: Condvar,
}

#[derive(Default)]
struct InvocationGate {
    active: usize,
    quiescing: bool,
}

struct InvocationLease {
    host: Arc<ComponentHostInner>,
}

fn begin_invocation_lease_for_host(
    host: &Arc<ComponentHostInner>,
    plugin: &InstalledVerifiedPlugin,
) -> Result<InvocationLease, ComponentHostError> {
    let mut gate = host
        .invocation_gate
        .lock()
        .map_err(|_| ComponentHostError::StateUnavailable)?;
    while gate.quiescing {
        gate = host
            .invocation_gate_changed
            .wait(gate)
            .map_err(|_| ComponentHostError::StateUnavailable)?;
    }
    let state = host
        .state
        .read()
        .map_err(|_| ComponentHostError::StateUnavailable)?;
    let active = state
        .by_extension
        .get(plugin.extension_id())
        .ok_or_else(|| ComponentHostError::Revoked(plugin.extension_id().to_owned()))?;
    if !Arc::ptr_eq(&active.inner, &plugin.inner) {
        return Err(ComponentHostError::Revoked(
            plugin.extension_id().to_owned(),
        ));
    }
    gate.active = gate
        .active
        .checked_add(1)
        .ok_or(ComponentHostError::StateUnavailable)?;
    drop(state);
    drop(gate);
    Ok(InvocationLease { host: host.clone() })
}

struct ConformanceInProcessExecutor {
    plugin_host: Arc<PluginHost>,
    services: Arc<dyn PluginCapabilityServices>,
}

impl PluginInvocationExecutor for ConformanceInProcessExecutor {
    fn execute<'a>(
        &'a self,
        invocation: PreparedPluginInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<InvocationResult, ComponentHostError>> + Send + 'a>>
    {
        Box::pin(async move {
            invocation
                .context
                .cancellation
                .check()
                .map_err(|_| PluginError::Invocation(sim_plugin_sdk::InvocationError::Cancelled))?;
            let host_invocation = self.plugin_host.begin_invocation(
                &invocation.plugin.inner.manifest,
                &invocation.plugin.inner.authorization,
                invocation.worker_invocation.node_id(),
                invocation.worker_invocation.inputs().clone(),
                self.services.clone(),
                invocation.context.cancellation.clone(),
            )?;
            let mut instance = self
                .plugin_host
                .instantiate_component(&invocation.plugin.inner.compiled, host_invocation)?;
            let node = instance.create_node(invocation.worker_invocation.node_id())?;
            if let Err(error) = instance.invoke(node) {
                instance.abort();
                return Err(error.into());
            }
            if let Err(error) = instance.drop_node(node) {
                instance.abort();
                return Err(error.into());
            }
            instance.finish().map_err(ComponentHostError::from)
        })
    }

    fn execute_provider<'a>(
        &'a self,
        invocation: PreparedPluginInvocation,
    ) -> Pin<
        Box<dyn Future<Output = Result<ProviderInvocationResult, ComponentHostError>> + Send + 'a>,
    > {
        Box::pin(async move {
            invocation
                .context
                .cancellation
                .check()
                .map_err(|_| PluginError::Invocation(sim_plugin_sdk::InvocationError::Cancelled))?;
            let provider_request = invocation
                .worker_invocation
                .provider_request()
                .ok_or_else(|| {
                    ComponentHostError::ExecutionBoundary(
                        "provider invocation omitted its bounded request".to_owned(),
                    )
                })?
                .to_vec();
            let host_invocation = self.plugin_host.begin_invocation(
                &invocation.plugin.inner.manifest,
                &invocation.plugin.inner.authorization,
                invocation.worker_invocation.node_id(),
                invocation.worker_invocation.inputs().clone(),
                self.services.clone(),
                invocation.context.cancellation.clone(),
            )?;
            let instance = self
                .plugin_host
                .instantiate_component(&invocation.plugin.inner.compiled, host_invocation)?;
            instance
                .invoke_provider(invocation.worker_invocation.node_id(), &provider_request)
                .map_err(ComponentHostError::from)
        })
    }

    fn execute_provider_v2<'a>(
        &'a self,
        _invocation: PreparedPluginInvocation,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<sim_runtime::ProviderTransportResponse, ComponentHostError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async {
            Err(ComponentHostError::ExecutionBoundary(
                "provider-v2 private worker route is unavailable in conformance mode".to_owned(),
            ))
        })
    }
}

struct ComponentHostRouterState {
    host: ComponentHost,
    extension_store_replay_snapshot: Vec<InstalledComponent>,
    next_generation: u64,
    bundle_subscribers: Vec<async_channel::Sender<sim_runtime::NativeExecutionRegistryBundle>>,
}

impl ComponentHostRouterState {
    fn revoke(&self) -> Result<(), ComponentHostError> {
        self.host.synchronize_components(Vec::new())
    }
}

#[derive(Clone)]
pub struct ComponentHostRouter {
    state: Arc<Mutex<ComponentHostRouterState>>,
}

impl ComponentHostRouter {
    pub fn new(host: ComponentHost) -> Self {
        Self {
            state: Arc::new(Mutex::new(ComponentHostRouterState {
                host,
                extension_store_replay_snapshot: Vec::new(),
                next_generation: 1,
                bundle_subscribers: Vec::new(),
            })),
        }
    }

    pub fn with_initial_generation(
        host: ComponentHost,
        initial_generation: u64,
    ) -> Result<Self, ComponentHostError> {
        if initial_generation == 0 {
            return Err(ComponentHostError::Verification {
                extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
                message: "component generation must be nonzero".to_owned(),
            });
        }
        host.synchronize_components_at_generation(Vec::new(), initial_generation)?;
        Ok(Self {
            state: Arc::new(Mutex::new(ComponentHostRouterState {
                host,
                extension_store_replay_snapshot: Vec::new(),
                next_generation: initial_generation,
                bundle_subscribers: Vec::new(),
            })),
        })
    }

    pub fn current(&self) -> Result<ComponentHost, ComponentHostError> {
        self.state
            .lock()
            .map(|state| state.host.clone())
            .map_err(|_| ComponentHostError::StateUnavailable)
    }

    pub fn active_execution_registry_bundle(
        &self,
    ) -> Result<sim_runtime::NativeExecutionRegistryBundle, ComponentHostError> {
        self.current()?.execution_registry_bundle()
    }

    pub fn subscribe_execution_registry_bundles(
        &self,
    ) -> Result<
        async_channel::Receiver<sim_runtime::NativeExecutionRegistryBundle>,
        ComponentHostError,
    > {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        let bundle = state.host.execution_registry_bundle()?;
        let (sender, receiver) = async_channel::bounded(8);
        sender
            .try_send(bundle)
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        state.bundle_subscribers.push(sender);
        Ok(receiver)
    }

    pub fn replace(&self, host: ComponentHost) -> Result<(), ComponentHostError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        replace_component_host(&mut state, host)
    }

    pub fn replace_with_initial_generation(
        &self,
        host: ComponentHost,
        initial_generation: u64,
    ) -> Result<(), ComponentHostError> {
        if initial_generation == 0 {
            return Err(ComponentHostError::Verification {
                extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
                message: "component generation must be nonzero".to_owned(),
            });
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        state.next_generation = state.next_generation.max(initial_generation);
        replace_component_host(&mut state, host)
    }
}

fn replace_component_host(
    state: &mut ComponentHostRouterState,
    host: ComponentHost,
) -> Result<(), ComponentHostError> {
    if Arc::ptr_eq(&state.host.inner, &host.inner) {
        return Ok(());
    }
    let generation = state.next_generation;
    host.synchronize_components_at_generation(
        state.extension_store_replay_snapshot.clone(),
        generation,
    )?;
    state.revoke()?;
    state.next_generation = state
        .next_generation
        .checked_add(1)
        .ok_or(ComponentHostError::StateUnavailable)?;
    state.host = host;
    Ok(())
}

impl Drop for InvocationLease {
    fn drop(&mut self) {
        let mut gate = match self.host.invocation_gate.lock() {
            Ok(gate) => gate,
            Err(poisoned) => poisoned.into_inner(),
        };
        gate.active = gate.active.saturating_sub(1);
        self.host.invocation_gate_changed.notify_all();
    }
}

#[derive(Clone)]
pub struct ComponentHost {
    inner: Arc<ComponentHostInner>,
}

pub struct ComponentHostProviderInvocationAuthority {
    host: ComponentHost,
    broker: PluginCapabilityBroker,
    principal_id: String,
    receipt_issuer: Arc<ProviderResultReceiptIssuer>,
    receipt_lifetime: Duration,
    cost_authority: Option<Arc<dyn ProviderCostAuthorizationAuthority>>,
}

impl ComponentHostProviderInvocationAuthority {
    pub fn new(
        host: ComponentHost,
        broker: PluginCapabilityBroker,
        principal_id: impl Into<String>,
        receipt_issuer: Arc<ProviderResultReceiptIssuer>,
        receipt_lifetime: Duration,
    ) -> Result<Self, PluginServiceError> {
        let principal_id = principal_id.into();
        if principal_id.is_empty() || receipt_lifetime.is_zero() {
            return Err(PluginServiceError::ProviderResultReceiptAuthorityDenied);
        }
        Ok(Self {
            host,
            broker,
            principal_id,
            receipt_issuer,
            receipt_lifetime,
            cost_authority: None,
        })
    }

    pub fn with_cost_authority(
        mut self,
        cost_authority: Arc<dyn ProviderCostAuthorizationAuthority>,
    ) -> Self {
        self.cost_authority = Some(cost_authority);
        self
    }
}

impl NativeProviderInvocationAuthority for ComponentHostProviderInvocationAuthority {
    fn begin(
        &self,
        scope: NativeProviderInvocationScope,
    ) -> Result<PluginCapabilityInvocation, PluginServiceError> {
        let state = self
            .host
            .inner
            .state
            .read()
            .map_err(|_| PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        let deployment = state
            .generation
            .worker_deployment_plan()
            .map_err(|_| PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        if deployment.begin().generation().get() != scope.start.registry_generation
            || deployment.begin().registry_digest_sha256().as_str()
                != scope.start.registry_digest_sha256
            || scope.node_id.0 != scope.start.node_id
        {
            return Err(PluginServiceError::ProviderResultReceiptAuthorityDenied);
        }
        let descriptor = deployment
            .begin()
            .components()
            .iter()
            .find(|descriptor| descriptor.extension_id() == scope.start.extension_id)
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        if descriptor.extension_version() != scope.start.extension_version
            || descriptor.plugin_identifier() != scope.start.plugin_identifier
            || descriptor.plugin_version() != scope.start.plugin_version
            || descriptor.manifest_digest_sha256().as_str() != scope.start.manifest_digest_sha256
            || descriptor.component_digest_sha256().as_str() != scope.start.component_digest_sha256
            || descriptor.authorization_generation().as_str()
                != scope.start.authorization_generation_sha256
        {
            return Err(PluginServiceError::ProviderResultReceiptAuthorityDenied);
        }
        let plugin = state
            .by_extension
            .get(scope.start.extension_id.as_str())
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        let price_badge = state
            .registry
            .descriptor(&scope.start.node_id)
            .and_then(|descriptor| descriptor.source_schema.as_ref())
            .and_then(|schema| schema.node.price_badge.clone());
        let binding_set = plugin
            .manifest()
            .provider_binding
            .as_ref()
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        if binding_set.bindings_sha256 != scope.start.binding_set_sha256
            || !binding_set
                .bindings
                .iter()
                .any(|claim| claim.node_id == scope.start.node_id)
            || scope.start.maximum_response_bytes == 0
            || scope.start.maximum_response_bytes
                > self.host.inner.invocation_maximum_response_bytes
        {
            return Err(PluginServiceError::ProviderResultReceiptAuthorityDenied);
        }
        let authority = ProviderResultReceiptAuthority::new(
            &self.principal_id,
            &scope.start.compiled_plan_sha256,
            &scope.start.binding_set_sha256,
            self.receipt_issuer.clone(),
            self.receipt_lifetime,
        )
        .map_err(|_| PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(
                self.host.inner.invocation_timeout_milliseconds,
            ))
            .ok_or(PluginServiceError::ProviderResultReceiptAuthorityDenied)?;
        let context = PluginServiceInvocationContext::new_with_principal(
            scope.profile_id,
            scope.prompt_id,
            scope.attempt_id,
            scope.node_id,
            &self.principal_id,
            plugin.authorization().clone(),
            scope.cancellation,
            deadline,
            scope.start.maximum_response_bytes,
        )?
        .with_provider_result_authority(authority)?;
        let context = match price_badge {
            Some(price_badge) => {
                context.with_provider_cost_requirement(price_badge, self.cost_authority.clone())?
            }
            None => context,
        };
        self.broker.begin_invocation(context)
    }
}

impl ComponentHost {
    pub fn new(
        runtime: ComponentRuntime,
        trust_policy: PluginTrustPolicy,
        permission_policy: PermissionPolicy,
        execution_boundary: ComponentExecutionBoundary,
        limits: ComponentLimits,
        base_registry: NativeNodeRegistry,
    ) -> Result<Self, ComponentHostError> {
        let profile_id = permission_policy.profile_id().to_owned();
        let authorization_sealer = PluginAuthorizationSealer::generate(
            permission_policy.generation(),
        )
        .map_err(|error| ComponentHostError::Verification {
            extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
            message: format!("failed to initialize authorization sealing: {error}"),
        })?;
        let authorization_verifier =
            authorization_sealer
                .verifier()
                .map_err(|error| ComponentHostError::Verification {
                    extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
                    message: format!("failed to initialize authorization verification: {error}"),
                })?;
        let invocation_timeout_milliseconds = limits
            .capability_limits
            .maximum_timeout_milliseconds
            .min(MAX_WORKER_PLUGIN_TIMEOUT_MILLISECONDS);
        let invocation_maximum_response_bytes =
            limits.capability_limits.maximum_response_bytes.min(
                u64::try_from(MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES)
                    .map_err(|error| ComponentHostError::ExecutionBoundary(error.to_string()))?,
            );
        let plugin_host = PluginHost::with_component_runtime(
            runtime,
            limits,
            crate::DEFAULT_API_FEATURES
                .iter()
                .map(|feature| (*feature).to_owned()),
        )?;
        let plugin_host = Arc::new(plugin_host);
        let (executor, conformance_services): (
            Arc<dyn PluginInvocationExecutor>,
            Option<Arc<dyn PluginCapabilityServices>>,
        ) = match execution_boundary {
            ComponentExecutionBoundary::PrivateWorker(executor) => (executor, None),
            ComponentExecutionBoundary::ConformanceInProcess(services) => (
                Arc::new(ConformanceInProcessExecutor {
                    plugin_host: plugin_host.clone(),
                    services: services.clone(),
                }),
                Some(services),
            ),
        };
        Ok(Self {
            inner: Arc::new(ComponentHostInner {
                plugin_host,
                trust_policy,
                permission_policy,
                authorization_sealer,
                executor,
                conformance_services,
                invocation_timeout_milliseconds,
                invocation_maximum_response_bytes,
                base_registry: base_registry.clone(),
                state: RwLock::new(ComponentState {
                    by_extension: BTreeMap::new(),
                    node_owners: BTreeMap::new(),
                    registry: base_registry,
                    generation: empty_component_generation(&profile_id, 0, authorization_verifier),
                }),
                invocation_gate: Mutex::new(InvocationGate::default()),
                invocation_gate_changed: Condvar::new(),
            }),
        })
    }

    pub fn installed_plugins(&self) -> Result<Vec<InstalledVerifiedPlugin>, ComponentHostError> {
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        Ok(state.by_extension.values().cloned().collect())
    }

    pub fn installed_plugin(
        &self,
        extension_id: &str,
    ) -> Result<InstalledVerifiedPlugin, ComponentHostError> {
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        state
            .by_extension
            .get(extension_id)
            .cloned()
            .ok_or_else(|| ComponentHostError::MissingExtension(extension_id.to_owned()))
    }

    pub fn plugin_for_node(
        &self,
        node_id: &str,
    ) -> Result<InstalledVerifiedPlugin, ComponentHostError> {
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        let extension_id = state
            .node_owners
            .get(node_id)
            .ok_or_else(|| ComponentHostError::MissingExtension(node_id.to_owned()))?;
        state
            .by_extension
            .get(extension_id)
            .cloned()
            .ok_or(ComponentHostError::StateUnavailable)
    }

    pub fn registry_snapshot(&self) -> Result<NativeNodeRegistry, ComponentHostError> {
        self.inner
            .state
            .read()
            .map(|state| state.registry.clone())
            .map_err(|_| ComponentHostError::StateUnavailable)
    }

    pub fn verified_generation(&self) -> Result<VerifiedComponentGeneration, ComponentHostError> {
        self.inner
            .state
            .read()
            .map(|state| state.generation.clone())
            .map_err(|_| ComponentHostError::StateUnavailable)
    }

    pub fn execution_registry_bundle(
        &self,
    ) -> Result<sim_runtime::NativeExecutionRegistryBundle, ComponentHostError> {
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        let profile_id = uuid::Uuid::parse_str(state.generation.profile_id()).map_err(|_| {
            ComponentHostError::Verification {
                extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
                message: "verified component profile identity is invalid".to_owned(),
            }
        })?;
        let deployment = state.generation.worker_deployment_plan()?;
        let provider_registry = state.generation.provider_registry_pin()?;
        sim_runtime::NativeExecutionRegistryBundle::checked(
            sim_types::ProfileId(profile_id),
            state.registry.clone(),
            deployment,
            provider_registry,
        )
        .map_err(|error| ComponentHostError::Verification {
            extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
            message: error.to_string(),
        })
    }

    pub(crate) fn prepare_plugin_invocation(
        &self,
        plugin: &InstalledVerifiedPlugin,
        node_id: &str,
        inputs: InvocationInputs,
        context: NodeContext,
    ) -> Result<PreparedPluginInvocation, ComponentHostError> {
        plugin.require_legacy_invocation_route()?;
        let lease = self.begin_invocation_lease(plugin)?;
        let generation = self.verified_generation()?;
        let worker_invocation = generation.prepare_worker_invocation(
            plugin.extension_id(),
            node_id,
            inputs,
            self.inner.invocation_timeout_milliseconds,
            self.inner.invocation_maximum_response_bytes,
            self.inner.plugin_host.limits().clone(),
        )?;
        let deployment = generation.worker_deployment_plan()?;
        Ok(PreparedPluginInvocation {
            worker_invocation,
            deployment,
            authorization: plugin.authorization().clone(),
            context,
            plugin: plugin.clone(),
            generation,
            plugin_host: self.inner.plugin_host.clone(),
            lease_host: self.inner.clone(),
            provider_price_badge: None,
            _lease: Some(lease),
        })
    }

    pub(crate) fn prepare_provider_invocation(
        &self,
        plugin: &InstalledVerifiedPlugin,
        node_id: &str,
        inputs: InvocationInputs,
        provider_request: Vec<u8>,
        context: NodeContext,
    ) -> Result<PreparedPluginInvocation, ComponentHostError> {
        plugin.require_legacy_invocation_route()?;
        let lease = self.begin_invocation_lease(plugin)?;
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        let generation = state.generation.clone();
        let provider_price_badge = state
            .registry
            .descriptor(node_id)
            .and_then(|descriptor| descriptor.source_schema.as_ref())
            .and_then(|schema| schema.node.price_badge.clone());
        drop(state);
        let worker_invocation = generation.prepare_worker_provider_invocation(
            plugin.extension_id(),
            node_id,
            inputs,
            provider_request,
            self.inner.invocation_timeout_milliseconds,
            self.inner.invocation_maximum_response_bytes,
            self.inner.plugin_host.limits().clone(),
        )?;
        let deployment = generation.worker_deployment_plan()?;
        Ok(PreparedPluginInvocation {
            worker_invocation,
            deployment,
            authorization: plugin.authorization().clone(),
            context,
            plugin: plugin.clone(),
            generation,
            plugin_host: self.inner.plugin_host.clone(),
            lease_host: self.inner.clone(),
            provider_price_badge,
            _lease: Some(lease),
        })
    }

    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator prepares provider-v2 worker invocations"
        )
    )]
    pub(crate) fn prepare_provider_v2_worker_invocation(
        &self,
        plugin: &InstalledVerifiedPlugin,
        node_id: &str,
        inputs: InvocationInputs,
        context: NodeContext,
    ) -> Result<PreparedPluginInvocation, ComponentHostError> {
        if !plugin.is_provider_v2() {
            return Err(ComponentHostError::ExecutionBoundary(
                "provider-v2 worker route requires a signed provider-v8 component".to_owned(),
            ));
        }
        let generation = self.verified_generation()?;
        let worker_invocation = generation.prepare_worker_provider_v2_invocation(
            plugin.extension_id(),
            node_id,
            inputs,
            self.inner.invocation_timeout_milliseconds,
            self.inner.invocation_maximum_response_bytes,
            self.inner.plugin_host.limits().clone(),
        )?;
        let deployment = generation.worker_deployment_plan()?;
        Ok(PreparedPluginInvocation {
            worker_invocation,
            deployment,
            authorization: plugin.authorization().clone(),
            context,
            plugin: plugin.clone(),
            generation,
            plugin_host: self.inner.plugin_host.clone(),
            lease_host: self.inner.clone(),
            provider_price_badge: None,
            _lease: None,
        })
    }

    pub(crate) fn executor(&self) -> Arc<dyn PluginInvocationExecutor> {
        self.inner.executor.clone()
    }

    pub async fn execute_plugin(
        &self,
        plugin: &InstalledVerifiedPlugin,
        node_id: &str,
        inputs: InvocationInputs,
        context: NodeContext,
    ) -> Result<InvocationResult, ComponentHostError> {
        plugin.require_legacy_invocation_route()?;
        let prepared = self.prepare_plugin_invocation(plugin, node_id, inputs, context)?;
        self.executor().execute(prepared).await
    }

    pub async fn execute_provider(
        &self,
        plugin: &InstalledVerifiedPlugin,
        node_id: &str,
        inputs: InvocationInputs,
        provider_request: Vec<u8>,
        context: NodeContext,
    ) -> Result<ProviderInvocationResult, ComponentHostError> {
        let prepared =
            self.prepare_provider_invocation(plugin, node_id, inputs, provider_request, context)?;
        self.executor().execute_provider(prepared).await
    }

    #[cfg(feature = "test-support")]
    pub async fn execute_provider_v2_worker(
        &self,
        plugin: &InstalledVerifiedPlugin,
        node_id: &str,
        inputs: InvocationInputs,
        context: NodeContext,
    ) -> Result<sim_runtime::ProviderTransportResponse, ComponentHostError> {
        let prepared =
            self.prepare_provider_v2_worker_invocation(plugin, node_id, inputs, context)?;
        self.executor().execute_provider_v2(prepared).await
    }

    pub fn invoke(
        &self,
        plugin: &InstalledVerifiedPlugin,
        node_id: &str,
        inputs: InvocationInputs,
        cancellation: CancellationToken,
    ) -> Result<InvocationResult, ComponentHostError> {
        plugin.require_legacy_invocation_route()?;
        let _invocation_lease = self.begin_invocation_lease(plugin)?;
        let services = self.inner.conformance_services.clone().ok_or_else(|| {
            ComponentHostError::ExecutionBoundary(
                "direct component invocation is only available to the conformance oracle"
                    .to_owned(),
            )
        })?;
        cancellation
            .check()
            .map_err(|_| PluginError::Invocation(sim_plugin_sdk::InvocationError::Cancelled))?;
        let invocation = self.inner.plugin_host.begin_invocation(
            &plugin.inner.manifest,
            &plugin.inner.authorization,
            node_id,
            inputs,
            services,
            cancellation,
        )?;
        let mut instance = self
            .inner
            .plugin_host
            .instantiate_component(&plugin.inner.compiled, invocation)?;
        let node = instance.create_node(node_id)?;
        if let Err(error) = instance.invoke(node) {
            instance.abort();
            return Err(error.into());
        }
        if let Err(error) = instance.drop_node(node) {
            instance.abort();
            return Err(error.into());
        }
        instance.finish().map_err(ComponentHostError::from)
    }

    fn begin_invocation_lease(
        &self,
        plugin: &InstalledVerifiedPlugin,
    ) -> Result<InvocationLease, ComponentHostError> {
        begin_invocation_lease_for_host(&self.inner, plugin)
    }

    fn synchronize_components(
        &self,
        components: Vec<InstalledComponent>,
    ) -> Result<(), ComponentHostError> {
        let generation = self
            .inner
            .state
            .read()
            .map_err(|_| ComponentHostError::StateUnavailable)?
            .generation
            .generation
            .saturating_add(1)
            .max(1);
        self.synchronize_components_at_generation(components, generation)
    }

    fn synchronize_components_at_generation(
        &self,
        components: Vec<InstalledComponent>,
        generation: u64,
    ) -> Result<(), ComponentHostError> {
        let mut by_extension = BTreeMap::new();
        let mut plugin_ids = BTreeSet::new();
        let mut node_owners = BTreeMap::new();
        for component in components {
            let extension_id: Arc<str> = Arc::from(component.extension_id());
            let (manifest, provider_manifest_v2) =
                parse_component_manifest(component.manifest_bytes()).map_err(|error| {
                    ComponentHostError::InvalidManifest {
                        extension_id: extension_id.clone(),
                        message: error.to_string(),
                    }
                })?;
            let binding = InstalledComponentBinding::checked(&component, &manifest)?;
            if !plugin_ids.insert(manifest.identifier.clone()) {
                return Err(ComponentHostError::DuplicatePlugin(manifest.identifier));
            }
            let provider_authorization_v2 = provider_manifest_v2
                .as_ref()
                .map(|provider_manifest| {
                    self.inner.trust_policy.authorize_provider_manifest_v2(
                        provider_manifest,
                        &self.inner.permission_policy,
                    )
                })
                .transpose()
                .map_err(|error| ComponentHostError::Verification {
                    extension_id: extension_id.clone(),
                    message: error.to_string(),
                })?;
            let authorization = match &provider_authorization_v2 {
                Some(authorization) => authorization.authorization().clone(),
                None => self
                    .inner
                    .trust_policy
                    .authorize_manifest(&manifest, &self.inner.permission_policy)
                    .map_err(|error| ComponentHostError::Verification {
                        extension_id: extension_id.clone(),
                        message: error.to_string(),
                    })?,
            };
            let compiled = match (&provider_manifest_v2, &provider_authorization_v2) {
                (Some(provider_manifest), Some(provider_authorization)) => {
                    self.inner.plugin_host.compile_provider_component_v2(
                        component.component_bytes(),
                        provider_manifest,
                        provider_authorization,
                    )
                }
                (None, None) => self.inner.plugin_host.compile_component(
                    component.component_bytes(),
                    &manifest,
                    &authorization,
                ),
                _ => Err(PluginError::ProviderRuntimeActivationDenied),
            }
            .map_err(|error| ComponentHostError::Verification {
                extension_id: extension_id.clone(),
                message: error.to_string(),
            })?;
            for node in &manifest.nodes {
                if node_owners
                    .insert(node.id.clone(), extension_id.clone())
                    .is_some()
                {
                    return Err(ComponentHostError::DuplicateNode(node.id.clone()));
                }
            }
            let plugin = InstalledVerifiedPlugin {
                inner: Arc::new(VerifiedPlugin {
                    binding,
                    manifest: Arc::new(manifest),
                    authorization: Arc::new(authorization),
                    provider_manifest_v2: provider_manifest_v2.map(Arc::new),
                    provider_authorization_v2: provider_authorization_v2.map(Arc::new),
                    compiled: Arc::new(compiled),
                    manifest_bytes: Arc::from(component.manifest_bytes()),
                    component_bytes: Arc::from(component.component_bytes()),
                }),
            };
            if by_extension.insert(extension_id.clone(), plugin).is_some() {
                return Err(ComponentHostError::DuplicatePlugin(
                    extension_id.to_string(),
                ));
            }
        }
        let plugins = by_extension.values().cloned().collect();
        let verified_generation = component_generation(
            self.inner.permission_policy.profile_id(),
            generation,
            by_extension.values(),
            &self.inner.authorization_sealer,
        )?;
        let registry = crate::registry_adapter::registry_with_plugins(
            &self.inner.base_registry,
            self,
            plugins,
            verified_generation.clone(),
        )
        .map_err(|error| ComponentHostError::Verification {
            extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
            message: error.to_string(),
        })?;
        let mut gate = self
            .inner
            .invocation_gate
            .lock()
            .map_err(|_| ComponentHostError::StateUnavailable)?;
        while gate.quiescing {
            gate = self
                .inner
                .invocation_gate_changed
                .wait(gate)
                .map_err(|_| ComponentHostError::StateUnavailable)?;
        }
        gate.quiescing = true;
        while gate.active != 0 {
            gate = self
                .inner
                .invocation_gate_changed
                .wait(gate)
                .map_err(|_| ComponentHostError::StateUnavailable)?;
        }
        let mut state = match self.inner.state.write() {
            Ok(state) => state,
            Err(_) => {
                gate.quiescing = false;
                self.inner.invocation_gate_changed.notify_all();
                return Err(ComponentHostError::StateUnavailable);
            }
        };
        *state = ComponentState {
            by_extension,
            node_owners,
            registry,
            generation: verified_generation,
        };
        gate.quiescing = false;
        self.inner.invocation_gate_changed.notify_all();
        Ok(())
    }
}

fn empty_component_generation(
    profile_id: &str,
    generation: u64,
    authorization_verifier: PluginAuthorizationVerifier,
) -> VerifiedComponentGeneration {
    VerifiedComponentGeneration {
        profile_id: Arc::from(profile_id),
        generation,
        snapshot_sha256: Arc::from(hex_sha256(&[])),
        authorization_verifier,
        components: Arc::from([]),
    }
}

fn parse_component_manifest(
    bytes: &[u8],
) -> Result<(PluginManifest, Option<ProviderPluginManifestV2>), serde_json::Error> {
    if let Ok(provider_manifest) = serde_json::from_slice::<ProviderPluginManifestV2>(bytes) {
        return Ok((provider_manifest.manifest.clone(), Some(provider_manifest)));
    }
    serde_json::from_slice(bytes).map(|manifest| (manifest, None))
}

fn append_worker_chunks(
    chunks: &mut Vec<WorkerRegistryDeploymentChunk>,
    generation: WorkerRegistryGeneration,
    component_index: u32,
    content: WorkerComponentContent,
    bytes: &[u8],
) -> Result<(), ComponentHostError> {
    for (chunk_index, chunk) in bytes.chunks(MAX_WORKER_COMPONENT_CHUNK_BYTES).enumerate() {
        let chunk_index = u32::try_from(chunk_index)
            .map_err(|error| worker_deployment_error(error.to_string()))?;
        chunks.push(
            WorkerRegistryDeploymentChunk::new(
                generation,
                component_index,
                content,
                chunk_index,
                chunk.to_vec(),
            )
            .map_err(worker_deployment_error)?,
        );
    }
    Ok(())
}

fn worker_deployment_error(error: impl ToString) -> ComponentHostError {
    ComponentHostError::Verification {
        extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
        message: format!("worker deployment mapping failed: {}", error.to_string()),
    }
}

fn valid_worker_plugin_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 1_024
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn component_generation<'a>(
    profile_id: &str,
    generation: u64,
    plugins: impl IntoIterator<Item = &'a InstalledVerifiedPlugin>,
    authorization_sealer: &PluginAuthorizationSealer,
) -> Result<VerifiedComponentGeneration, ComponentHostError> {
    if generation == 0 {
        return Err(ComponentHostError::Verification {
            extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
            message: "component generation must be nonzero".to_owned(),
        });
    }
    let mut components = Vec::new();
    for plugin in plugins {
        let manifest_sha256 = hex_sha256(&plugin.inner.manifest_bytes);
        let component_sha256 = hex_sha256(&plugin.inner.component_bytes);
        if component_sha256 != plugin.inner.binding.signed_digest_sha256() {
            return Err(ComponentHostError::Verification {
                extension_id: Arc::from(plugin.extension_id()),
                message: "verified component bytes no longer match the signed digest".to_owned(),
            });
        }
        let authorization_bytes = plugin
            .inner
            .authorization
            .sealed_bytes(authorization_sealer)
            .map_err(|error| ComponentHostError::Verification {
                extension_id: Arc::from(plugin.extension_id()),
                message: format!("failed to seal worker authorization: {error}"),
            })?;
        let authorization_generation = hex_sha256(&authorization_bytes);
        components.push(VerifiedComponentDeployment {
            extension_id: Arc::from(plugin.extension_id()),
            extension_version: Arc::from(plugin.extension_version()),
            plugin_identifier: Arc::from(plugin.inner.binding.signed_plugin_identifier()),
            plugin_version: Arc::from(plugin.inner.binding.signed_plugin_version()),
            manifest_sha256: Arc::from(manifest_sha256),
            component_sha256: Arc::from(component_sha256),
            authorization_generation: Arc::from(authorization_generation),
            manifest_bytes: plugin.inner.manifest_bytes.clone(),
            authorization_bytes: Arc::from(authorization_bytes),
            component_bytes: plugin.inner.component_bytes.clone(),
        });
    }
    components.sort_by(|left, right| left.extension_id.cmp(&right.extension_id));
    let mut snapshot = Sha256::new();
    hash_field(&mut snapshot, profile_id.as_bytes());
    hash_field(&mut snapshot, &generation.to_le_bytes());
    for component in &components {
        hash_field(&mut snapshot, component.extension_id.as_bytes());
        hash_field(&mut snapshot, component.extension_version.as_bytes());
        hash_field(&mut snapshot, component.plugin_identifier.as_bytes());
        hash_field(&mut snapshot, component.plugin_version.as_bytes());
        hash_field(&mut snapshot, component.manifest_sha256.as_bytes());
        hash_field(&mut snapshot, component.component_sha256.as_bytes());
        hash_field(&mut snapshot, component.authorization_generation.as_bytes());
    }
    Ok(VerifiedComponentGeneration {
        profile_id: Arc::from(profile_id),
        generation,
        snapshot_sha256: Arc::from(format!("{:x}", snapshot.finalize())),
        authorization_verifier: authorization_sealer.verifier().map_err(|error| {
            ComponentHostError::Verification {
                extension_id: Arc::from(SIM_COMPONENT_ADAPTER_ID),
                message: format!("failed to project authorization verifier: {error}"),
            }
        })?,
        components: Arc::from(components),
    })
}

fn hash_field(digest: &mut Sha256, value: &[u8]) {
    digest.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
    digest.update(value);
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl ComponentLifecycleAdapter for ComponentHostRouter {
    fn adapter_id(&self) -> &'static str {
        SIM_COMPONENT_ADAPTER_ID
    }

    fn synchronize(
        &self,
        components: Vec<InstalledComponent>,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'static>> {
        let this = self.clone();
        Box::pin(async move {
            smol::unblock(move || {
                let mut state = this
                    .state
                    .lock()
                    .map_err(|_| ComponentHostError::StateUnavailable)?;
                let generation = state.next_generation;
                state
                    .host
                    .synchronize_components_at_generation(components.clone(), generation)?;
                state.next_generation = state
                    .next_generation
                    .checked_add(1)
                    .ok_or(ComponentHostError::StateUnavailable)?;
                state.extension_store_replay_snapshot = components;
                if !state.bundle_subscribers.is_empty() {
                    let bundle = state.host.execution_registry_bundle()?;
                    state.bundle_subscribers.retain(|subscriber| {
                        // Strict receivers reject older bundle identities. Keep the
                        // newest verified state even when their bounded queue is full.
                        match subscriber.force_send(bundle.clone()) {
                            Ok(_superseded_bundle) => true,
                            Err(_closed) => false,
                        }
                    });
                }
                Ok::<(), ComponentHostError>(())
            })
            .await
            .map_err(|error| error.to_string())
        })
    }
}

#[cfg(test)]
mod latest_registry_delivery_tests {
    use super::*;

    #[test]
    fn saturated_registry_subscription_retains_newest_verified_bundle()
    -> Result<(), Box<dyn std::error::Error>> {
        let host = ComponentHost::new(
            extension_host::ComponentRuntime::no_wasi()?,
            sim_runtime::PluginTrustPolicy::default(),
            sim_runtime::PermissionPolicy::new(
                "00000000-0000-4000-8000-000000000001",
                std::iter::empty(),
            )?,
            ComponentExecutionBoundary::conformance_in_process(Arc::new(
                crate::UnavailablePluginCapabilityServices,
            )),
            ComponentLimits::default(),
            sim_runtime::generated_native_node_registry_projection(None)?,
        )?;
        let router = ComponentHostRouter::with_initial_generation(host, 1)?;
        let receiver = router.subscribe_execution_registry_bundles()?;
        let initial = receiver.try_recv()?;
        for _ in 0..24 {
            smol::block_on(ComponentLifecycleAdapter::synchronize(&router, Vec::new()))
                .map_err(std::io::Error::other)?;
        }
        assert_eq!(receiver.len(), 8);
        let current = router.active_execution_registry_bundle()?;
        let mut delivered = Vec::new();
        loop {
            match receiver.try_recv() {
                Ok(bundle) => delivered.push(bundle),
                Err(async_channel::TryRecvError::Empty) => break,
                Err(async_channel::TryRecvError::Closed) => {
                    return Err(
                        std::io::Error::other("live registry subscription was closed").into(),
                    );
                }
            }
        }
        assert_eq!(delivered.len(), 8);
        let latest = delivered
            .last()
            .ok_or_else(|| std::io::Error::other("latest registry state was dropped"))?;
        assert_eq!(latest.identity_sha256(), current.identity_sha256());
        assert_ne!(initial.identity_sha256(), current.identity_sha256());
        assert_eq!(latest.profile_id(), current.profile_id());
        assert_eq!(latest.worker_deployment().begin().generation().get(), 24);
        assert_eq!(
            delivered
                .iter()
                .filter(|bundle| { bundle.identity_sha256() == current.identity_sha256() })
                .count(),
            1
        );
        drop(receiver);
        smol::block_on(ComponentLifecycleAdapter::synchronize(&router, Vec::new()))
            .map_err(std::io::Error::other)?;
        let state = router
            .state
            .lock()
            .map_err(|_| std::io::Error::other("router state poisoned"))?;
        assert!(state.bundle_subscribers.is_empty());
        Ok(())
    }
}
