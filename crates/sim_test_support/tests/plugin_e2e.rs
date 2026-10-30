use ::fs::{FakeFs, Fs};
use clock::SystemClock;
use extension_host::{
    ComponentLifecycleAdapter, ComponentRuntime, ExtensionIndexEntry, ExtensionManifest,
    ExtensionStore, InstalledComponent,
};
use gpui::BackgroundExecutor;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use sim_api::{
    HostRequestContext, HttpBody, HttpLimits, HttpRequest, NativeApiHost, NativeAutomationBody,
    NativeAutomationResult, NativeCliInvocation, NativeCliOperation, NativeHeadlessPolicy,
    NativeHeadlessService, NativeRuntimeApiHost, NativeRuntimeHttpServices, WebSocketLimits,
    security::{ApiSecurityConfig, ArtifactIdempotencySnapshotStore},
};
use sim_model::NativeModelPayload;
use sim_nodes::NodeRegistry as CatalogNodeRegistry;
use sim_nodes::{
    NativeEffectServiceError, NativeNodeComputeSession, NativeNodeServiceIdentity,
    NativeNodeServices, NativeOutputEffectRequest, NativePreparedEffectKind,
    NativePreparedEffectRequest, NativePreparedEffectService, NativeProviderExecutionIdentity,
};
use sim_plugin_host::{
    CancellationToken, ComponentExecutionBoundary, ComponentHost, ComponentHostError,
    ComponentHostRouter, ComponentLimits, InvocationInputs, LegacyInputSourceProjection,
    LegacyMappingResolver, LegacyNodeReference, LegacyResolution, MappingCandidate, MappingSource,
    MappingTarget, PluginCapabilityServices, PluginError, PluginHost, PluginOutputProposal,
    PluginOutputPublicationAdapter, PrivateWorkerPluginExecutor,
};
use sim_plugin_sdk::{
    ApiRequirement, ApiVersion, ArtifactValue, CachePolicy, CancelReason, CapabilityKind,
    CapabilityQuota, CapabilityRequest, DType, DeterminismPolicy, DeviceId,
    ED25519_SIGNATURE_BYTES, EffectPolicy, InvocationError, ManifestProvenance, ManifestSignature,
    ModelValue, PLUGIN_SIGNATURE_ALGORITHM, PROVIDER_BINDING_API_FEATURE,
    PROVIDER_BINDING_SCHEMA_VERSION, PluginContractError, PluginInvocation, PluginManifest,
    PluginNode, PluginPort, PluginSigningKey, PluginValue, PortCardinality, PortDirection,
    PortPresence, PortSerialization, ProviderBindingClaim, ProviderBindingSet, RouteDeclaration,
    RustNodeInstance, RustSimPlugin, ScalarValue, StreamId, TensorDescriptor, TensorValue,
    TypeRegistry, UiContribution, ValueFamily,
};
use sim_runtime::{
    AuthorizedCredentialPresenceRequest, AuthorizedProviderRequest, Capability, CapabilitySet,
    CredentialPresenceActuator, CredentialScope, DisconnectedExecutionController,
    ExecutionDataSource, ExecutionEventBus, ExecutionPresentationService, ExecutionSnapshotStatus,
    NativeHandleStore, NativeHandleStoreError, NativeHandleStoreGeneration,
    NativeHandleStoreIdentity, NativeHandleType, NativeNodeRegistry, NativeOpaqueHandle,
    NativePrimitive, NativeResolvedPayload, NativeStoredModelPayload, NativeStoredPayload,
    NativeValue, NodeContext, NodeOutcome, OutputCommitter, OutputExecutionScope, PermissionGrant,
    PermissionPolicy, PluginAuthorization, PluginCapabilityBroker, PluginRngPolicy,
    PluginServiceActuatorError, PluginServiceOperationContext, PluginTrustPolicy,
    PluginVerificationKey, ProfileId, ProviderCostAcceptanceIssuer, ProviderCostApproval,
    ProviderCostApprovalAuthority, ProviderCostNonce, ProviderEndpoint, ProviderMode,
    ProviderPolicy, ProviderPriceBound, ProviderRequestActuator, ProviderResultReceiptIssuer,
    ProviderTransportPort, ProviderTransportResponse, ProviderTransportValue, SecretId,
    SecretValue, SharedAssetService, WorkerLaunchConfig, authorize_native_output_committer,
    authorize_native_plugin_asset_broker, native_image_catalog_bindings,
    native_image_registry_projection, open_native_profile_asset_service,
};
use sim_sampler::{NativeConditioningPayload, NativeDiffusionPayload};
use sim_tensor::{
    CpuWorkspaceAuthority, ImageTensor, NativeTensorPayload, NativeTensorRole, RngAlgorithm,
    RngProfileVersion, ScratchReservation, StreamId as TensorStreamId,
};
use sim_test_support::NativeDiffusionFixture;
use sim_types::{AttemptId, HttpMethod, NodeId, PromptId, WorkerId};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use uuid::Uuid;

const KEY_ID: &str = "test.publisher";
const KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";
const TEST_PROFILE_ID: &str = "00000000-0000-0000-0000-000000002100";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyMappingFixture {
    schema_version: u16,
    production_execution: String,
    custom_node: LegacyCustomNodeFixture,
    provider_node: LegacyProviderNodeFixture,
    unresolved_node: LegacyUnresolvedNodeFixture,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyCustomNodeFixture {
    legacy_identifier: String,
    legacy_input_name: String,
    legacy_input_position: u32,
    target_port_id: String,
    legacy_output_index: u32,
    target_output_index: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyProviderNodeFixture {
    legacy_identifier: String,
    provider: String,
    endpoint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyUnresolvedNodeFixture {
    legacy_identifier: String,
    serialized_fields: String,
    serialized_widgets: String,
    serialized_links: String,
    extension_data: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyFrontendExtensionFixture {
    schema_version: u16,
    production_execution: String,
    cases: Vec<LegacyFrontendExtensionCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyFrontendExtensionCase {
    name: String,
    feature_id: Option<String>,
    expected_classification: String,
    payload: String,
}

fn quota() -> CapabilityQuota {
    CapabilityQuota {
        maximum_operations: 16,
        maximum_request_bytes: 4_096,
        maximum_response_bytes: 4_096,
        maximum_total_bytes: 32_768,
        maximum_handles: 8,
        timeout_milliseconds: 5_000,
    }
}

fn port(
    registry: &TypeRegistry,
    id: &str,
    direction: PortDirection,
    source_type: &str,
    cardinality: PortCardinality,
    presence: PortPresence,
) -> Result<PluginPort, Box<dyn Error>> {
    let family = registry.family(registry.resolve(source_type)?)?;
    let serialization = match family {
        ValueFamily::Scalar => PortSerialization::Inline,
        ValueFamily::Tensor | ValueFamily::Model => PortSerialization::Handle,
        ValueFamily::Artifact => PortSerialization::ArtifactReference,
    };
    Ok(PluginPort {
        id: id.to_owned(),
        name: id.to_owned(),
        direction,
        type_id: registry.resolve(source_type)?.clone(),
        cardinality,
        presence,
        hidden: presence == PortPresence::Hidden,
        lazy: cardinality == PortCardinality::List && direction == PortDirection::Input,
        default: None,
        serialization,
        accepted_legacy_names: if id == "scalar-single-in" {
            vec!["legacy_scalar".to_owned()]
        } else {
            Vec::new()
        },
    })
}

fn manifest(component_digest: String) -> Result<PluginManifest, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let mut ports = Vec::new();
    for (family, source_type, presence) in [
        ("scalar", "String", PortPresence::Required),
        ("tensor", "Image", PortPresence::Required),
        ("artifact", "SVG", PortPresence::Optional),
        ("model", "Model", PortPresence::Hidden),
    ] {
        ports.push(port(
            &registry,
            &format!("{family}-single-in"),
            PortDirection::Input,
            source_type,
            PortCardinality::Singular,
            presence,
        )?);
        ports.push(port(
            &registry,
            &format!("{family}-single-out"),
            PortDirection::Output,
            source_type,
            PortCardinality::Singular,
            if family == "artifact" {
                PortPresence::Optional
            } else {
                PortPresence::Required
            },
        )?);
        ports.push(port(
            &registry,
            &format!("{family}-list-in"),
            PortDirection::Input,
            source_type,
            PortCardinality::List,
            PortPresence::Optional,
        )?);
        ports.push(port(
            &registry,
            &format!("{family}-list-out"),
            PortDirection::Output,
            source_type,
            PortCardinality::List,
            PortPresence::Optional,
        )?);
    }
    let capabilities = include_str!("../../sim_plugin_host/tests/fixtures/capabilities")
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (kind, scope) = line
                .split_once('|')
                .ok_or_else(|| format!("invalid capability fixture line `{line}`"))?;
            let kind = match kind {
                "filesystem" => CapabilityKind::Filesystem,
                "network-provider" => CapabilityKind::NetworkProvider,
                "secret" => CapabilityKind::Secret,
                "clock" => CapabilityKind::Clock,
                "randomness" => CapabilityKind::Randomness,
                "model" => CapabilityKind::Model,
                "transactional-output" => CapabilityKind::TransactionalOutput,
                "sanitized-log" => CapabilityKind::SanitizedLog,
                "declarative-ui" => CapabilityKind::DeclarativeUi,
                "route" => CapabilityKind::Route,
                value => return Err(format!("unknown capability fixture kind `{value}`")),
            };
            Ok(CapabilityRequest {
                kind,
                scope: scope.to_owned(),
                quota: quota(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(PluginManifest {
        schema_version: 1,
        identifier: "test.echo-plugin".to_owned(),
        plugin_version: ApiVersion::new(1, 2, 3),
        api: ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 0,
            required_features: vec![
                "capabilities.transactional".to_owned(),
                "handles.revocation".to_owned(),
                "legacy.non-destructive".to_owned(),
                "ports.list".to_owned(),
            ],
        },
        digest_sha256: component_digest,
        signature: ManifestSignature {
            algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
            key_id: KEY_ID.to_owned(),
            value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
        },
        provenance: ManifestProvenance {
            source: "fixture://test.echo-plugin".to_owned(),
            publisher: "Zed test publisher".to_owned(),
            registry: Some("fixture://signed-registry".to_owned()),
        },
        provider_binding: None,
        nodes: vec![PluginNode {
            id: "echo".to_owned(),
            version: ApiVersion::new(1, 0, 0),
            display_name: "Echo".to_owned(),
            category: "test".to_owned(),
            ports,
            determinism: DeterminismPolicy::Deterministic,
            cache: CachePolicy::InputIdentity,
            effects: EffectPolicy::Transactional,
        }],
        capabilities,
        ui: vec![UiContribution {
            id: "panel.demo".to_owned(),
            surface: "node-panel".to_owned(),
            state_schema: "{\"type\":\"object\"}".to_owned(),
        }],
        routes: vec![RouteDeclaration {
            id: "route.demo".to_owned(),
            method: "POST".to_owned(),
            path: "/plugins/test/echo".to_owned(),
            maximum_request_bytes: 4_096,
            maximum_response_bytes: 4_096,
        }],
        legacy_mappings: vec![sim_plugin_sdk::LegacyMapping {
            legacy_identifier: "LegacyEcho".to_owned(),
            node_id: "echo".to_owned(),
            node_version: ApiVersion::new(1, 0, 0),
            legacy_widget_names: vec!["legacy_scalar".to_owned()],
            input_translations: vec![sim_plugin_sdk::LegacyInputTranslation::Rename {
                target_port_id: "scalar-single-in".to_owned(),
                legacy_input_id: "legacy_scalar".to_owned(),
            }],
            output_translations: vec![sim_plugin_sdk::LegacyOutputTranslation {
                target_port_index: 0,
                legacy_output_index: 3,
            }],
        }],
    })
}

fn provider_manifest(component_digest: String) -> Result<PluginManifest, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let input_ports = [
        ("prompt", "String", PortPresence::Required),
        ("model", "COMFY_DYNAMICCOMBO_V3", PortPresence::Required),
        ("seed", "Int", PortPresence::Required),
        ("system_prompt", "String", PortPresence::Optional),
    ]
    .into_iter()
    .map(|(id, source_type, presence)| {
        port(
            &registry,
            id,
            PortDirection::Input,
            source_type,
            PortCardinality::Singular,
            presence,
        )
    })
    .collect::<Result<Vec<_>, _>>()?;
    let output_port = port(
        &registry,
        "output_0",
        PortDirection::Output,
        "String",
        PortCardinality::Singular,
        PortPresence::Required,
    )?;
    let contract_sha256 = sim_runtime::generated_native_node_registry_projection(None)?
        .provider_binding_contract_sha256(
            "OpenRouterLLMNode",
            "zed:sim-provider-transport@1",
            "zed:sim-provider-materializer@1",
        )?
        .ok_or("OpenRouter provider fixture contract is absent")?;
    let mut provider_binding = ProviderBindingSet {
        schema_version: PROVIDER_BINDING_SCHEMA_VERSION,
        implementation_namespace: "zed.sim.provider.openrouter".to_owned(),
        bindings_sha256: "0".repeat(64),
        bindings: vec![ProviderBindingClaim {
            feature_id: "COMFY-NODE-0466".to_owned(),
            node_id: "OpenRouterLLMNode".to_owned(),
            contract_sha256,
            transport_schema: "zed:sim-provider-transport@1".parse()?,
            materializer_schema: "zed:sim-provider-materializer@1".parse()?,
        }],
    };
    provider_binding.bindings_sha256 = provider_binding.canonical_bindings_sha256()?;
    Ok(PluginManifest {
        schema_version: 1,
        identifier: "zed.sim.provider.openrouter".to_owned(),
        plugin_version: ApiVersion::new(1, 0, 0),
        api: ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 0,
            required_features: vec![PROVIDER_BINDING_API_FEATURE.to_owned()],
        },
        digest_sha256: component_digest,
        signature: ManifestSignature {
            algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
            key_id: KEY_ID.to_owned(),
            value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
        },
        provenance: ManifestProvenance {
            source: "fixture://test.provider-plugin".to_owned(),
            publisher: "Zed provider fixture".to_owned(),
            registry: Some("fixture://signed-registry".to_owned()),
        },
        provider_binding: Some(provider_binding),
        nodes: vec![PluginNode {
            id: "OpenRouterLLMNode".to_owned(),
            version: ApiVersion::new(1, 0, 0),
            display_name: "OpenRouter LLM".to_owned(),
            category: "partner/text/OpenRouter".to_owned(),
            ports: input_ports.into_iter().chain([output_port]).collect(),
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        }],
        capabilities: [
            CapabilityKind::NetworkProvider,
            CapabilityKind::ProviderUpload,
            CapabilityKind::ProviderCost,
        ]
        .into_iter()
        .map(|kind| CapabilityRequest {
            kind,
            scope: "fixture|https://fixture.invalid/v1/generate".to_owned(),
            quota: CapabilityQuota {
                maximum_operations: 16,
                maximum_request_bytes: 16 * 1024 * 1024,
                maximum_response_bytes: 64 * 1024 * 1024,
                maximum_total_bytes: 80 * 1024 * 1024,
                maximum_handles: 8,
                timeout_milliseconds: 5_000,
            },
        })
        .collect(),
        ui: Vec::new(),
        routes: Vec::new(),
        legacy_mappings: Vec::new(),
    })
}

fn trust_policy() -> Result<PluginTrustPolicy, Box<dyn Error>> {
    let signing_key = signing_key()?;
    Ok(PluginTrustPolicy::new([PluginVerificationKey::new(
        KEY_ID,
        signing_key.verification_key_bytes()?,
    )?])?)
}

fn signing_key() -> Result<PluginSigningKey, PluginContractError> {
    PluginSigningKey::new(KEY_ID, KEY)
}

fn sign(_trust: &PluginTrustPolicy, manifest: &mut PluginManifest) -> Result<(), Box<dyn Error>> {
    manifest.signature.value = signing_key()?.sign_manifest(manifest)?;
    Ok(())
}

fn authorize(
    trust: &PluginTrustPolicy,
    manifest: &PluginManifest,
) -> Result<PluginAuthorization, Box<dyn Error>> {
    Ok(trust.authorize_manifest(manifest, &permission_policy(manifest)?)?)
}

fn permission_policy(manifest: &PluginManifest) -> Result<PermissionPolicy, Box<dyn Error>> {
    let requested = manifest
        .capabilities
        .iter()
        .map(Capability::from_plugin_request)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PermissionPolicy::new(
        TEST_PROFILE_ID,
        [PermissionGrant::new(
            TEST_PROFILE_ID,
            manifest.identifier.clone(),
            CapabilitySet::new(requested),
            "signed-test-fixture",
        )?],
    )?)
}

fn configured_host(limits: ComponentLimits) -> Result<PluginHost, PluginError> {
    PluginHost::with_configuration(
        limits,
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )
}

fn conformance_component_limits() -> ComponentLimits {
    ComponentLimits {
        epoch_deadline_ticks: 50,
        ..ComponentLimits::default()
    }
}

fn scalar_value(value: &str) -> Result<PluginValue, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    Ok(PluginValue::scalar(
        registry.resolve("String")?.clone(),
        ScalarValue::String(value.to_owned()),
        &registry,
    )?)
}

fn tensor_value(identifier: &str) -> Result<PluginValue, Box<dyn Error>> {
    let stream = identifier.bytes().fold(0_u64, |value, byte| {
        value.wrapping_mul(31).wrapping_add(u64::from(byte))
    });
    let registry = TypeRegistry::built_in()?;
    Ok(PluginValue::tensor(
        registry.resolve("Image")?.clone(),
        TensorValue::new(
            TensorDescriptor::contiguous(
                vec![1, 1, 1, 1],
                DType::F32,
                DeviceId::CPU,
                StreamId::new(stream),
            )?,
            4,
            "1".repeat(64),
        )?,
        &registry,
    )?)
}

fn artifact_value(identifier: &str) -> Result<PluginValue, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    Ok(PluginValue::artifact(
        registry.resolve("SVG")?.clone(),
        ArtifactValue::new("input", identifier, 3, "2".repeat(64))?,
        &registry,
    )?)
}

fn model_value(identifier: &str) -> Result<PluginValue, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    Ok(PluginValue::model(
        registry.resolve("Model")?.clone(),
        ModelValue::new(identifier, "safetensors", "3".repeat(64))?,
        &registry,
    )?)
}

fn invocation_inputs() -> Result<InvocationInputs, Box<dyn Error>> {
    let mut inputs = InvocationInputs::default();
    inputs.set_present("scalar-single-in", vec![scalar_value("scalar")?]);
    inputs.set_present("tensor-single-in", vec![tensor_value("1")?]);
    inputs.set_present("artifact-single-in", vec![artifact_value("artifact.svg")?]);
    inputs.set_present("model-single-in", vec![model_value("model")?]);
    inputs.set_present("scalar-list-in", Vec::new());
    inputs.set_present(
        "tensor-list-in",
        vec![tensor_value("2")?, tensor_value("3")?],
    );
    inputs.set_present("artifact-list-in", Vec::new());
    inputs.set_present("model-list-in", vec![model_value("a")?, model_value("b")?]);
    Ok(inputs)
}

#[derive(Default)]
struct TestPluginServices {
    files: BTreeMap<(String, String), Vec<u8>>,
    provider_responses: BTreeMap<(String, String), Vec<u8>>,
    secret_identifiers: BTreeSet<String>,
    clocks_milliseconds: BTreeMap<String, u64>,
    random_seeds: BTreeMap<String, [u8; 32]>,
    random_counters: Mutex<BTreeMap<String, u64>>,
    models: BTreeMap<String, ModelValue>,
}

impl PluginCapabilityServices for TestPluginServices {
    fn read_asset(
        &self,
        identity: &sim_runtime::AssetIdentity,
        _context: &sim_plugin_host::CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        self.files
            .get(&(
                identity.namespace.locator_type().to_owned(),
                identity.relative_path.to_string_lossy().into_owned(),
            ))
            .cloned()
            .ok_or_else(|| InvocationError::HostFailure("file not found".to_owned()))
    }

    fn call_provider(
        &self,
        provider: &str,
        endpoint: &str,
        _body: &[u8],
        secret_id: Option<&SecretId>,
    ) -> Result<Vec<u8>, InvocationError> {
        if secret_id.is_some_and(|secret| !self.secret_identifiers.contains(secret.as_str())) {
            return Err(InvocationError::HostFailure(
                "provider secret is unavailable".to_owned(),
            ));
        }
        self.provider_responses
            .get(&(provider.to_owned(), endpoint.to_owned()))
            .cloned()
            .ok_or_else(|| InvocationError::HostFailure("provider unavailable".to_owned()))
    }

    fn secret_exists(&self, identifier: &str) -> Result<bool, InvocationError> {
        Ok(self.secret_identifiers.contains(identifier))
    }

    fn clock_milliseconds(&self, clock: &str) -> Result<u64, InvocationError> {
        self.clocks_milliseconds
            .get(clock)
            .copied()
            .ok_or_else(|| InvocationError::HostFailure("clock unavailable".to_owned()))
    }

    fn random_bytes(&self, stream: &str, length: u32) -> Result<Vec<u8>, InvocationError> {
        let seed =
            self.random_seeds.get(stream).copied().ok_or_else(|| {
                InvocationError::HostFailure("random stream unavailable".to_owned())
            })?;
        let length = usize::try_from(length)
            .map_err(|_| InvocationError::HostFailure("random length overflow".to_owned()))?;
        let mut counters = self
            .random_counters
            .lock()
            .map_err(|_| InvocationError::HostFailure("random counter poisoned".to_owned()))?;
        let counter = counters.entry(stream.to_owned()).or_default();
        let mut bytes = Vec::with_capacity(length);
        while bytes.len() < length {
            let mut hasher = Sha256::new();
            hasher.update(b"zed-sim-plugin-random-v1");
            hasher.update(seed);
            hasher.update(counter.to_le_bytes());
            let block = hasher.finalize();
            let remaining = length.saturating_sub(bytes.len());
            bytes.extend_from_slice(&block[..remaining.min(block.len())]);
            *counter = counter.checked_add(1).ok_or_else(|| {
                InvocationError::HostFailure("random counter overflow".to_owned())
            })?;
        }
        Ok(bytes)
    }

    fn open_model(&self, identifier: &str) -> Result<ModelValue, InvocationError> {
        self.models
            .get(identifier)
            .cloned()
            .ok_or_else(|| InvocationError::HostFailure("model unavailable".to_owned()))
    }

    fn sanitize_log(&self, _level: &str, message: &str) -> Result<String, InvocationError> {
        let mut message = message.replace('\0', "");
        for secret in &self.secret_identifiers {
            message = message.replace(secret, "[REDACTED]");
        }
        Ok(message)
    }
}

fn empty_services() -> Arc<dyn PluginCapabilityServices> {
    Arc::new(TestPluginServices::default())
}

fn component_resources() -> Result<Arc<dyn PluginCapabilityServices>, Box<dyn Error>> {
    let mut resources = TestPluginServices::default();
    resources.files.insert(
        ("input".to_owned(), "nested/file.bin".to_owned()),
        b"file".to_vec(),
    );
    resources.provider_responses.insert(
        (
            "demo".to_owned(),
            "https://demo.invalid/v1/generate".to_owned(),
        ),
        b"provider".to_vec(),
    );
    resources
        .secret_identifiers
        .insert("secret.demo".to_owned());
    resources
        .clocks_milliseconds
        .insert("workflow".to_owned(), 1_234);
    resources.random_seeds.insert("sampler".to_owned(), [7; 32]);
    resources.models.insert(
        "zed-asset://model/fixture.json".to_owned(),
        ModelValue::new(
            "zed-asset://model/fixture.json",
            "json-config",
            "4".repeat(64),
        )?,
    );
    Ok(Arc::new(resources))
}

struct WorkerPluginClock {
    now: Mutex<Instant>,
}

impl WorkerPluginClock {
    fn new(now: Instant) -> Self {
        Self {
            now: Mutex::new(now),
        }
    }

    fn advance(&self, duration: Duration) {
        match self.now.lock() {
            Ok(mut now) => *now += duration,
            Err(poisoned) => *poisoned.into_inner() += duration,
        }
    }
}

impl SystemClock for WorkerPluginClock {
    fn utc_now(&self) -> Instant {
        match self.now.lock() {
            Ok(now) => *now,
            Err(poisoned) => *poisoned.into_inner(),
        }
    }
}

#[derive(Default)]
struct WorkerPluginProvider {
    calls: AtomicUsize,
    authorized: Mutex<Vec<(String, String, Option<String>, Option<Vec<u8>>)>>,
}

#[derive(Default)]
struct BlockingWorkerPluginProvider {
    entered: AtomicBool,
}

#[derive(Default)]
struct WorkerLossProvider {
    calls: AtomicUsize,
}

impl ProviderRequestActuator for WorkerLossProvider {
    fn execute(
        &self,
        _request: &AuthorizedProviderRequest,
        _secret: Option<&SecretValue>,
        _body: &[u8],
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Vec<u8>, PluginServiceActuatorError> {
        context
            .check_active()
            .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
        if self.calls.fetch_add(1, Ordering::AcqRel) == 0 {
            std::thread::sleep(Duration::from_secs(6));
        }
        context
            .check_active()
            .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
        Ok(b"provider".to_vec())
    }
}

impl ProviderRequestActuator for BlockingWorkerPluginProvider {
    fn execute(
        &self,
        _request: &AuthorizedProviderRequest,
        _secret: Option<&SecretValue>,
        _body: &[u8],
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Vec<u8>, PluginServiceActuatorError> {
        self.entered.store(true, Ordering::Release);
        loop {
            if let Err(error) = context.check_active() {
                return Err(PluginServiceActuatorError::new(error.to_string()));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl ProviderRequestActuator for WorkerPluginProvider {
    fn execute(
        &self,
        request: &AuthorizedProviderRequest,
        secret: Option<&SecretValue>,
        body: &[u8],
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Vec<u8>, PluginServiceActuatorError> {
        context
            .check_active()
            .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
        if body != b"request" {
            return Err(PluginServiceActuatorError::new(
                "worker plugin provider body changed",
            ));
        }
        let record = (
            request.provider().to_owned(),
            request.endpoint().to_owned(),
            request
                .secret_id()
                .map(|secret_id| secret_id.as_str().to_owned()),
            secret.map(|secret| secret.expose_to(<[u8]>::to_vec)),
        );
        match self.authorized.lock() {
            Ok(mut authorized) => authorized.push(record),
            Err(poisoned) => poisoned.into_inner().push(record),
        }
        self.calls.fetch_add(1, Ordering::AcqRel);
        Ok(b"provider".to_vec())
    }
}

struct ProviderWorldActuator {
    calls: AtomicUsize,
    response: Vec<u8>,
}

impl ProviderRequestActuator for ProviderWorldActuator {
    fn execute(
        &self,
        request: &AuthorizedProviderRequest,
        secret: Option<&SecretValue>,
        body: &[u8],
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Vec<u8>, PluginServiceActuatorError> {
        context
            .check_active()
            .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
        if request.provider() != "fixture"
            || request.endpoint() != "https://fixture.invalid/v1/generate"
            || request.secret_id().is_some()
            || secret.is_some()
            || sim_runtime::ProviderTransportRequest::from_bytes(body).is_err()
        {
            return Err(PluginServiceActuatorError::new(
                "provider-world request identity changed",
            ));
        }
        self.calls.fetch_add(1, Ordering::AcqRel);
        Ok(self.response.clone())
    }
}

#[derive(Default)]
struct WorkerPluginCredentials {
    presence_calls: AtomicUsize,
    read_calls: AtomicUsize,
}

impl CredentialPresenceActuator for WorkerPluginCredentials {
    fn is_present(
        &self,
        request: &AuthorizedCredentialPresenceRequest,
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<bool, PluginServiceActuatorError> {
        context
            .check_active()
            .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
        self.presence_calls.fetch_add(1, Ordering::AcqRel);
        Ok(request.secret_id().as_str() == "secret.demo")
    }

    fn read_for_provider(
        &self,
        request: &AuthorizedCredentialPresenceRequest,
        context: &PluginServiceOperationContext<'_>,
    ) -> Result<Option<SecretValue>, PluginServiceActuatorError> {
        context
            .check_active()
            .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
        self.read_calls.fetch_add(1, Ordering::AcqRel);
        Ok((request.secret_id().as_str() == "secret.demo")
            .then(|| SecretValue::new(b"worker-secret-value".to_vec())))
    }
}

fn worker_plugin_assets() -> Result<(tempfile::TempDir, SharedAssetService), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let assets = open_native_profile_asset_service(TEST_PROFILE_ID, directory.path(), &[])?;
    let input = directory.path().join("input/nested/file.bin");
    fs::create_dir_all(input.parent().ok_or("worker input parent is unavailable")?)?;
    fs::write(input, b"file")?;
    fs::write(
        directory.path().join("model/fixture.json"),
        br#"{"model_type":"fixture","hidden_size":4}"#,
    )?;
    let authorization = authorize_native_plugin_asset_broker(TEST_PROFILE_ID)?;
    assets
        .lock()
        .map_err(|_| "worker plugin asset service lock is unavailable")?
        .scan(&authorization, &CancellationToken::default())?;
    Ok((directory, assets))
}

fn worker_plugin_broker(
    assets: SharedAssetService,
) -> Result<
    (
        PluginCapabilityBroker,
        Arc<WorkerPluginClock>,
        Arc<WorkerPluginProvider>,
        Arc<WorkerPluginCredentials>,
    ),
    Box<dyn Error>,
> {
    worker_plugin_broker_with_provider_mode(assets, ProviderMode::Enabled)
}

fn worker_plugin_broker_with_provider_mode(
    assets: SharedAssetService,
    provider_mode: ProviderMode,
) -> Result<
    (
        PluginCapabilityBroker,
        Arc<WorkerPluginClock>,
        Arc<WorkerPluginProvider>,
        Arc<WorkerPluginCredentials>,
    ),
    Box<dyn Error>,
> {
    let clock = Arc::new(WorkerPluginClock::new(Instant::now()));
    let provider = Arc::new(WorkerPluginProvider::default());
    let credentials = Arc::new(WorkerPluginCredentials::default());
    let broker = PluginCapabilityBroker::new(
        assets,
        sim_model::ModelStore::new(sim_model::ParserLimits::default())?,
        ProviderPolicy::new(
            TEST_PROFILE_ID,
            provider_mode,
            [ProviderEndpoint::new(
                "demo",
                "https://demo.invalid/v1/generate",
            )?],
            [CredentialScope::new(
                TEST_PROFILE_ID,
                "test.echo-plugin",
                "demo",
                SecretId::new("secret.demo")?,
            )?],
        )?,
        provider.clone(),
        credentials.clone(),
        clock.clone(),
        PluginRngPolicy::new(RngProfileVersion::V2, RngAlgorithm::Philox4x32_10, 21_001),
    );
    Ok((broker, clock, provider, credentials))
}

fn extension_entry(
    extension_id: &str,
    extension_version: &str,
) -> Result<ExtensionIndexEntry, Box<dyn Error>> {
    let manifest: ExtensionManifest = serde_json::from_value(json!({
        "id": extension_id,
        "name": "Signed Sim component fixture",
        "version": extension_version,
        "schema_version": 1,
    }))?;
    Ok(ExtensionIndexEntry {
        manifest: Arc::new(manifest),
        dev: false,
    })
}

async fn write_component_pair(
    filesystem: &FakeFs,
    installed_directory: &Path,
    extension_id: &str,
    manifest: &[u8],
    component: &[u8],
) -> Result<(), Box<dyn Error>> {
    let extension_directory = installed_directory.join(extension_id);
    filesystem
        .insert_tree(&extension_directory, json!({}))
        .await;
    filesystem
        .insert_file(
            extension_directory.join(extension_host::SIM_COMPONENT_MANIFEST_FILE),
            manifest.to_vec(),
        )
        .await;
    filesystem
        .insert_file(
            extension_directory.join(extension_host::SIM_COMPONENT_BINARY_FILE),
            component.to_vec(),
        )
        .await;
    Ok(())
}

async fn synchronize_extension_store_with_router(
    filesystem: Arc<dyn Fs>,
    installed_directory: &Path,
    entries: &[ExtensionIndexEntry],
    router: &ComponentHostRouter,
) -> BTreeMap<String, String> {
    let adapters: Vec<Arc<dyn ComponentLifecycleAdapter>> = vec![Arc::new(router.clone())];
    ExtensionStore::synchronize_component_adapters(
        filesystem,
        installed_directory,
        entries,
        &adapters,
    )
    .await
}

async fn exercise_provider_world_private_worker(
    profile_id: ProfileId,
) -> Result<(), Box<dyn Error>> {
    const COMPILED_PLAN_SHA256: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let component = provider_component_fixture()?;
    let component_digest = format!("{:x}", Sha256::digest(&component));
    let mut manifest = provider_manifest(component_digest.clone())?;
    let trust = trust_policy()?;
    sign(&trust, &mut manifest)?;
    let binding_sha256 = manifest
        .provider_binding
        .as_ref()
        .ok_or("provider binding is absent")?
        .bindings_sha256
        .clone();
    let (_asset_directory, assets) = worker_plugin_assets()?;
    let clock = Arc::new(WorkerPluginClock::new(Instant::now()));
    let type_registry = TypeRegistry::built_in()?;
    let output_value = PluginValue::scalar(
        type_registry.resolve("String")?.clone(),
        ScalarValue::String("fixture response".to_owned()),
        &type_registry,
    )?;
    let response = ProviderTransportResponse::checked(
        "OpenRouterLLMNode",
        vec![ProviderTransportPort::checked(
            "output_0",
            true,
            vec![ProviderTransportValue::checked(
                "sim:string@1",
                ValueFamily::Scalar,
                output_value.abi_bytes()?,
            )?],
        )?],
    )?
    .to_bytes()?;
    let provider = Arc::new(ProviderWorldActuator {
        calls: AtomicUsize::new(0),
        response,
    });
    let cost_issuer = Arc::new(ProviderCostAcceptanceIssuer::generate(clock.utc_now())?);
    let cost_authority = Arc::new(ProviderCostApprovalAuthority::new(
        cost_issuer.clone(),
        clock.clone(),
    ));
    let broker = PluginCapabilityBroker::new_with_provider_cost_acceptance(
        assets,
        sim_model::ModelStore::new(sim_model::ParserLimits::default())?,
        ProviderPolicy::new(
            TEST_PROFILE_ID,
            ProviderMode::Enabled,
            [ProviderEndpoint::new(
                "fixture",
                "https://fixture.invalid/v1/generate",
            )?],
            std::iter::empty::<CredentialScope>(),
        )?,
        cost_authority.verifier()?,
        provider.clone(),
        Arc::new(WorkerPluginCredentials::default()),
        clock.clone(),
        PluginRngPolicy::new(RngProfileVersion::V2, RngAlgorithm::Philox4x32_10, 21_078),
    );
    let receipt_issuer = Arc::new(ProviderResultReceiptIssuer::generate(clock.utc_now())?);
    let launch = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_plugin_worker_fixture"),
        profile_id,
        WorkerId(Uuid::from_u128(0x21079)),
        "worker-provider-v1",
        8 * 1024 * 1024 * 1024,
    );
    let boundary = ComponentExecutionBoundary::private_worker(
        PrivateWorkerPluginExecutor::new_with_provider_authorities(
            launch,
            broker,
            TEST_PROFILE_ID,
            receipt_issuer,
            Duration::from_secs(30),
            cost_authority.clone(),
        )?,
    );
    let host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust,
        permission_policy(&manifest)?,
        boundary,
        conformance_component_limits(),
        sim_runtime::generated_native_node_registry_projection(None)?,
    )?;
    let router = ComponentHostRouter::new(host.clone());
    router
        .synchronize(vec![InstalledComponent::checked(
            Arc::from("provider-extension"),
            Arc::from("1.0.0"),
            serde_json::to_vec(&manifest)?.into(),
            component.into(),
        )?])
        .await?;
    let bundle = router.active_execution_registry_bundle()?;
    let registry = bundle.registry();
    let price_badge = registry
        .descriptor("OpenRouterLLMNode")
        .and_then(|descriptor| descriptor.source_schema.as_ref())
        .and_then(|schema| schema.node.price_badge.clone())
        .ok_or("paid provider price badge is absent")?;
    price_badge
        .validate()
        .map_err(|error| format!("generated provider price badge is invalid: {error}"))?;
    sim_runtime::ProviderInvocationIdentity::new(
        TEST_PROFILE_ID,
        TEST_PROFILE_ID,
        "provider-cost-approval",
        COMPILED_PLAN_SHA256,
        "provider-cost-approval",
        "OpenRouterLLMNode",
        0,
        "0".repeat(64),
        manifest.identifier.clone(),
        component_digest.clone(),
        binding_sha256.clone(),
        "fixture",
        "https://fixture.invalid/v1/generate",
    )
    .map_err(|error| format!("provider approval identity is invalid: {error}"))?;
    let cost_approval = ProviderCostApproval::new(
        TEST_PROFILE_ID,
        TEST_PROFILE_ID,
        COMPILED_PLAN_SHA256,
        "OpenRouterLLMNode",
        0,
        manifest.identifier.clone(),
        component_digest,
        binding_sha256,
        "fixture",
        "https://fixture.invalid/v1/generate",
        &price_badge,
        ProviderPriceBound::new("USD", 25_000)?,
        clock.utc_now() + Duration::from_secs(30),
        ProviderCostNonce::new([0x79; 32])?,
    )
    .map_err(|error| format!("provider cost approval construction failed: {error}"))?;
    cost_authority
        .approve(cost_approval)
        .map_err(|error| format!("provider cost approval admission failed: {error}"))?;

    let prompt_id = PromptId(Uuid::from_u128(0x2107_9001));
    let attempt_id = AttemptId(Uuid::from_u128(0x2107_9002));
    let cancellation = CancellationToken::default();
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(attempt_id);
    let context = provider_node_context(
        prompt_id,
        attempt_id,
        NodeId("OpenRouterLLMNode".to_owned()),
        cancellation.clone(),
        store.clone(),
        COMPILED_PLAN_SHA256,
    )?;
    let outcome = registry
        .node("OpenRouterLLMNode")
        .ok_or("provider binding is not executable")?
        .execute(
            context,
            BTreeMap::from([
                (
                    "prompt".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::String("fixture prompt".to_owned()),
                    },
                ),
                (
                    "model".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::String("openai/gpt-5".to_owned()),
                    },
                ),
                (
                    "seed".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(1),
                    },
                ),
            ]),
        )
        .await?;
    let NodeOutcome::Values {
        outputs, effects, ..
    } = outcome
    else {
        return Err("provider node returned a non-value outcome".into());
    };
    assert!(effects.is_empty());
    assert!(matches!(
        outputs.as_slice(),
        [NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }] if value == "fixture response"
    ));
    assert_eq!(provider.calls.load(Ordering::Acquire), 1);

    let replay_generation = NativeHandleStoreGeneration::new()?;
    let replay_store =
        replay_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x2107_9003)));
    let replay = registry
        .node("OpenRouterLLMNode")
        .ok_or("provider binding disappeared")?
        .execute(
            provider_node_context(
                prompt_id,
                AttemptId(Uuid::from_u128(0x2107_9003)),
                NodeId("OpenRouterLLMNode".to_owned()),
                CancellationToken::default(),
                replay_store,
                COMPILED_PLAN_SHA256,
            )?,
            BTreeMap::from([
                (
                    "prompt".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::String("fixture prompt".to_owned()),
                    },
                ),
                (
                    "model".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::String("openai/gpt-5".to_owned()),
                    },
                ),
                (
                    "seed".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(1),
                    },
                ),
            ]),
        )
        .await
        .expect_err("spent cost approval must deny replay before provider actuation");
    assert!(replay.message.contains("cost") || replay.message.contains("denied"));
    assert_eq!(provider.calls.load(Ordering::Acquire), 1);
    Ok(())
}

fn component_inventory_error_contains(
    errors: &BTreeMap<String, String>,
    component_host: &ComponentHost,
    expected: &str,
) -> bool {
    errors.len() == 1
        && errors
            .get(ComponentHostRouter::new(component_host.clone()).adapter_id())
            .is_some_and(|error| error.contains(expected))
}

fn registry_payload_handle(
    store: &dyn NativeHandleStore,
    cancellation: &CancellationToken,
    payload: NativeStoredPayload,
) -> Result<NativeValue, Box<dyn Error>> {
    let handle = store.publish(payload, cancellation)?;
    Ok(NativeValue::Handle { value: handle })
}

fn canonical_image_payload(seed: u8) -> Result<NativeStoredPayload, Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let scratch = workspace_authority.authorize_workspace(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(TensorStreamId::DEFAULT, scratch, &cancellation);
    let value = f32::from(seed) / 255.0;
    let image = ImageTensor::from_f32(&backend, &context, 1, 1, 1, 3, &[value, value, value])?;
    Ok(NativeStoredPayload::Tensor(Arc::new(
        NativeTensorPayload::from_image(NativeTensorRole::Image, image)?,
    )))
}

fn canonical_model_payload() -> Result<NativeStoredPayload, Box<dyn Error>> {
    static MODEL_PAYLOAD: OnceLock<Arc<NativeStoredPayload>> = OnceLock::new();
    if let Some(payload) = MODEL_PAYLOAD.get() {
        return Ok(payload.as_ref().clone());
    }

    let fixture = NativeDiffusionFixture::checked_in();
    let (backend, workspace_authority) =
        CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let scratch = workspace_authority.authorize_workspace(2 * 1024 * 1024 * 1024)?;
    let context = backend.execution_context(TensorStreamId::DEFAULT, scratch, &cancellation);
    let bundle = fixture.load_bundle_with_context(backend, &context)?;
    let model = Arc::new(NativeModelPayload::sd15_model(bundle.model().clone())?);
    let conditioning = Arc::new(NativeConditioningPayload::checked_sd15(
        bundle.model_digest(),
        bundle.model().as_ref(),
        bundle.conditioning().patch_graph().clone(),
        None,
    )?);
    let diffusion = Arc::new(NativeDiffusionPayload::model(model, conditioning)?);
    let payload = Arc::new(NativeStoredPayload::Model(Arc::new(
        NativeStoredModelPayload::native_diffusion(diffusion)?,
    )));
    let _ = MODEL_PAYLOAD.set(payload.clone());
    Ok(payload.as_ref().clone())
}

fn registry_inputs(
    store: &dyn NativeHandleStore,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let model = registry_payload_handle(store, cancellation, canonical_model_payload()?)?;
    Ok(BTreeMap::from([
        (
            "artifact-list-in".to_owned(),
            NativeValue::List { values: Vec::new() },
        ),
        (
            "model-list-in".to_owned(),
            NativeValue::List {
                values: vec![model.clone(), model.clone()],
            },
        ),
        ("model-single-in".to_owned(), model),
        (
            "scalar-list-in".to_owned(),
            NativeValue::List { values: Vec::new() },
        ),
        (
            "scalar-single-in".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::String("scalar".to_owned()),
            },
        ),
        (
            "tensor-list-in".to_owned(),
            NativeValue::List {
                values: vec![
                    registry_payload_handle(store, cancellation, canonical_image_payload(2)?)?,
                    registry_payload_handle(store, cancellation, canonical_image_payload(3)?)?,
                ],
            },
        ),
        (
            "tensor-single-in".to_owned(),
            registry_payload_handle(store, cancellation, canonical_image_payload(1)?)?,
        ),
    ]))
}

#[derive(Debug)]
struct TestPreparedEffectService {
    identity: NativeNodeServiceIdentity,
    ordinal: AtomicUsize,
    prepared: Mutex<BTreeMap<Uuid, NativePreparedEffectRequest>>,
}

impl NativePreparedEffectService for TestPreparedEffectService {
    fn identity(&self) -> &NativeNodeServiceIdentity {
        &self.identity
    }

    fn maximum_output_bytes(&self) -> u64 {
        8 * 1024 * 1024
    }

    fn prepare_output(
        &self,
        request: NativeOutputEffectRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativePreparedEffectRequest, NativeEffectServiceError> {
        cancellation
            .check()
            .map_err(|_| NativeEffectServiceError::Cancelled)?;
        let ordinal = self
            .ordinal
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| NativeEffectServiceError::Rejected)?;
        let mut digest = Sha256::new();
        digest.update(self.identity.service_id().as_bytes());
        digest.update(self.identity.attempt_id().0.as_bytes());
        digest.update(self.identity.node_id().0.as_bytes());
        digest.update(ordinal.to_le_bytes());
        digest.update(request.request_digest_sha256().as_bytes());
        let digest = digest.finalize();
        let mut transaction_bytes = [0_u8; 16];
        transaction_bytes.copy_from_slice(&digest[..16]);
        let transaction_id = Uuid::from_bytes(transaction_bytes);
        let ticket = NativePreparedEffectRequest::checked(
            self.identity.service_id(),
            transaction_id,
            NativePreparedEffectKind::Output,
            request.request_digest_sha256(),
        )
        .map_err(|_| NativeEffectServiceError::Rejected)?;
        let mut prepared = self
            .prepared
            .lock()
            .map_err(|_| NativeEffectServiceError::Rejected)?;
        if prepared.insert(transaction_id, ticket.clone()).is_some() {
            return Err(NativeEffectServiceError::InvalidTicket);
        }
        if cancellation.check().is_err() {
            prepared.remove(&transaction_id);
            return Err(NativeEffectServiceError::Cancelled);
        }
        Ok(ticket)
    }

    fn rollback_prepared(
        &self,
        request: &NativePreparedEffectRequest,
    ) -> Result<(), NativeEffectServiceError> {
        if request.service_id() != self.identity.service_id() {
            return Err(NativeEffectServiceError::InvalidTicket);
        }
        let mut prepared = self
            .prepared
            .lock()
            .map_err(|_| NativeEffectServiceError::Rejected)?;
        match prepared.remove(&request.transaction_id()) {
            Some(stored) if stored == *request => Ok(()),
            Some(stored) => {
                prepared.insert(stored.transaction_id(), stored);
                Err(NativeEffectServiceError::InvalidTicket)
            }
            None => Err(NativeEffectServiceError::InvalidTicket),
        }
    }

    fn rollback_all_prepared(&self) -> Result<(), NativeEffectServiceError> {
        self.prepared
            .lock()
            .map_err(|_| NativeEffectServiceError::Rejected)?
            .clear();
        Ok(())
    }
}

fn plugin_node_context(
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: NodeId,
    cancellation: CancellationToken,
    scratch: ScratchReservation,
    store: Arc<dyn NativeHandleStore>,
) -> Result<NodeContext, Box<dyn Error>> {
    let identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x2100_0000_0000_0000_0000_0000_0000_0001),
        attempt_id,
        node_id.clone(),
    )?;
    let effects = Arc::new(TestPreparedEffectService {
        identity,
        ordinal: AtomicUsize::new(0),
        prepared: Mutex::new(BTreeMap::new()),
    });
    let services = NativeNodeServices::checked(None, Some(effects), None)?;
    Ok(NodeContext::new_with_services(
        prompt_id,
        attempt_id,
        node_id,
        cancellation,
        scratch,
        store,
        services,
    )?)
}

fn provider_node_context(
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: NodeId,
    cancellation: CancellationToken,
    store: Arc<dyn NativeHandleStore>,
    compiled_plan_sha256: &str,
) -> Result<NodeContext, Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let scratch = workspace_authority.authorize_workspace(64 * 1024 * 1024)?;
    let identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x2100_0000_0000_0000_0000_0000_0000_0079),
        attempt_id,
        node_id.clone(),
    )?;
    let compute = NativeNodeComputeSession::checked(
        identity.clone(),
        Arc::new(backend),
        TensorStreamId::DEFAULT,
        &scratch,
    )?;
    let effects = Arc::new(TestPreparedEffectService {
        identity,
        ordinal: AtomicUsize::new(0),
        prepared: Mutex::new(BTreeMap::new()),
    });
    let services = NativeNodeServices::checked(None, Some(effects), Some(compute))?
        .with_provider_execution(NativeProviderExecutionIdentity::checked(
            compiled_plan_sha256,
        )?);
    Ok(NodeContext::new_with_services(
        prompt_id,
        attempt_id,
        node_id,
        cancellation,
        scratch,
        store,
        services,
    )?)
}

fn registry_invocation(
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: NodeId,
    cancellation: CancellationToken,
) -> Result<
    (
        NodeContext,
        BTreeMap<String, NativeValue>,
        NativeHandleStoreGeneration,
        Arc<dyn NativeHandleStore>,
    ),
    Box<dyn Error>,
> {
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(attempt_id);
    let inputs = registry_inputs(store.as_ref(), &cancellation)?;
    let context = plugin_node_context(
        prompt_id,
        attempt_id,
        node_id,
        cancellation,
        zero_scratch()?,
        store.clone(),
    )?;
    Ok((context, inputs, generation, store))
}

#[derive(Debug)]
struct RejectingPublishStore {
    inner: Arc<dyn NativeHandleStore>,
    reject_at: usize,
    publication_count: AtomicUsize,
}

impl NativeHandleStore for RejectingPublishStore {
    fn identity(&self) -> NativeHandleStoreIdentity {
        self.inner.identity()
    }

    fn attempt_id(&self) -> AttemptId {
        self.inner.attempt_id()
    }

    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected_type: &NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
        self.inner.resolve(handle, expected_type, cancellation)
    }

    fn publish(
        &self,
        value: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
        let publication = self.publication_count.fetch_add(1, Ordering::AcqRel) + 1;
        if publication == self.reject_at {
            return Err(NativeHandleStoreError::Rejected(
                "injected plugin output publication failure".to_owned(),
            ));
        }
        self.inner.publish(value, cancellation)
    }

    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeHandleStoreError> {
        self.inner.revoke(handle, cancellation)
    }
}

fn native_handles(values: &[NativeValue]) -> Vec<NativeOpaqueHandle> {
    fn collect(value: &NativeValue, handles: &mut Vec<NativeOpaqueHandle>) {
        match value {
            NativeValue::Handle { value } => handles.push(value.clone()),
            NativeValue::List { values } => {
                for value in values {
                    collect(value, handles);
                }
            }
            NativeValue::Primitive { .. } | NativeValue::PreservedUnknown { .. } => {}
        }
    }

    let mut handles = Vec::new();
    for value in values {
        collect(value, &mut handles);
    }
    handles
}

async fn exercise_native_registry_value_boundary(
    registry: &NativeNodeRegistry,
) -> Result<(), Box<dyn Error>> {
    let artifact_output_index = registry
        .descriptor("echo")
        .ok_or("component registry has no echo descriptor")?
        .outputs
        .iter()
        .position(|output| output.name == "artifact-single-out")
        .ok_or("component registry has no artifact output descriptor")?;
    let node = registry
        .node("echo")
        .ok_or("component registry has no echo binding")?;
    let attempt_id = AttemptId(Uuid::from_u128(0x367));
    let cancellation = CancellationToken::default();
    let (context, inputs, generation, resolver) = registry_invocation(
        PromptId(Uuid::from_u128(0x366)),
        attempt_id,
        NodeId("typed-plugin-roundtrip".to_owned()),
        cancellation.clone(),
    )?;
    let input_handle_count = generation.len();
    let input_values = inputs.values().cloned().collect::<Vec<_>>();
    let input_handles = native_handles(&input_values);
    let input_identifiers = input_handles
        .iter()
        .map(|handle| handle.identifier().to_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(input_handle_count, input_identifiers.len());
    let outcome = node.execute(context, inputs).await?;
    let NodeOutcome::Values { outputs, .. } = outcome else {
        return Err("typed plugin adapter returned a non-value outcome".into());
    };
    assert!(matches!(
        outputs.first(),
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value)
        }) if value == "scalar"
    ));
    assert!(matches!(
        outputs.get(artifact_output_index),
        Some(NativeValue::Primitive {
            value: NativePrimitive::Null
        })
    ));
    let output_handles = native_handles(&outputs);
    assert_eq!(output_handles.len(), input_handles.len());
    assert_eq!(generation.len(), input_handle_count);
    for handle in &output_handles {
        assert!(input_identifiers.contains(handle.identifier()));
        resolver.resolve(handle, handle.handle_type(), &cancellation)?;
    }
    let mut revoked_identifiers = BTreeSet::new();
    for handle in input_handles.iter().rev() {
        if revoked_identifiers.insert(handle.identifier()) {
            resolver.revoke(handle, &cancellation)?;
        }
    }
    assert_eq!(generation.len(), 0);

    let (context, mut inputs, generation, _store) = registry_invocation(
        PromptId(Uuid::from_u128(0x368)),
        AttemptId(Uuid::from_u128(0x369)),
        NodeId("typed-plugin-wrong-type".to_owned()),
        CancellationToken::default(),
    )?;
    let input_handle_count = generation.len();
    let tensor = inputs
        .get("tensor-single-in")
        .cloned()
        .ok_or("tensor fixture is absent")?;
    inputs.insert("model-single-in".to_owned(), tensor);
    let error = node
        .execute(context, inputs)
        .await
        .expect_err("wrong native handle type must fail closed");
    assert!(error.message.contains("does not match"));
    assert_eq!(generation.len(), input_handle_count);

    let attempt_id = AttemptId(Uuid::from_u128(0x36a));
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(attempt_id);
    let valid_inputs = registry_inputs(store.as_ref(), &CancellationToken::default())?;
    let NativeValue::Handle { value: original } = valid_inputs
        .get("tensor-single-in")
        .ok_or("tensor fixture is absent")?
    else {
        return Err("tensor fixture is not a native handle".into());
    };
    let wrong_store_identity = NativeHandleStoreIdentity::new(
        Uuid::from_u128(0x36b),
        original.store_identity().generation_id,
    )?;
    let wrong_generation_identity =
        NativeHandleStoreIdentity::new(original.store_identity().store_id, Uuid::from_u128(0x36c))?;
    let invalid_handles = [
        NativeOpaqueHandle::new(
            original.handle_type().clone(),
            wrong_store_identity,
            original.identifier(),
            original.generation(),
            original.digest_sha256().map(str::to_owned),
        )?,
        NativeOpaqueHandle::new(
            original.handle_type().clone(),
            wrong_generation_identity,
            original.identifier(),
            original.generation(),
            original.digest_sha256().map(str::to_owned),
        )?,
        NativeOpaqueHandle::new(
            original.handle_type().clone(),
            original.store_identity(),
            original.identifier(),
            original.generation(),
            Some("f".repeat(64)),
        )?,
    ];
    let input_handle_count = generation.len();
    for (index, invalid_handle) in invalid_handles.into_iter().enumerate() {
        let mut inputs = valid_inputs.clone();
        inputs.insert(
            "tensor-single-in".to_owned(),
            NativeValue::Handle {
                value: invalid_handle,
            },
        );
        let context = plugin_node_context(
            PromptId(Uuid::from_u128(0x36d)),
            attempt_id,
            NodeId(format!("typed-plugin-invalid-handle-{index}")),
            CancellationToken::default(),
            zero_scratch()?,
            store.clone(),
        )?;
        node.execute(context, inputs)
            .await
            .expect_err("forged native handle must fail closed");
        assert_eq!(generation.len(), input_handle_count);
    }

    let cancellation = CancellationToken::default();
    let attempt_id = AttemptId(Uuid::from_u128(0x370));
    let generation = NativeHandleStoreGeneration::new()?;
    let inner = generation.handle_store_for_attempt(attempt_id);
    let inputs = registry_inputs(inner.as_ref(), &cancellation)?;
    let input_handle_cardinality =
        native_handles(&inputs.values().cloned().collect::<Vec<_>>()).len();
    let input_object_count = generation.len();
    let rejecting_store = Arc::new(RejectingPublishStore {
        inner,
        reject_at: 1,
        publication_count: AtomicUsize::new(0),
    });
    let context = plugin_node_context(
        PromptId(Uuid::from_u128(0x371)),
        attempt_id,
        NodeId("typed-plugin-publication-rollback".to_owned()),
        cancellation,
        zero_scratch()?,
        rejecting_store.clone(),
    )?;
    let NodeOutcome::Values { outputs, .. } = node.execute(context, inputs).await? else {
        return Err("imported plugin handles returned a non-value outcome".into());
    };
    assert_eq!(native_handles(&outputs).len(), input_handle_cardinality);
    assert_eq!(rejecting_store.publication_count.load(Ordering::Acquire), 0);
    assert_eq!(generation.len(), input_object_count);
    Ok(())
}

async fn invoke_registry_binding(
    registry: &NativeNodeRegistry,
) -> Result<NodeOutcome, Box<dyn Error>> {
    let node = registry
        .node("echo")
        .ok_or("component registry has no echo binding")?;
    let (context, inputs, _generation, _store) = registry_invocation(
        PromptId(Uuid::from_u128(1)),
        AttemptId(Uuid::from_u128(2)),
        NodeId("echo-fixture".to_owned()),
        CancellationToken::default(),
    )?;
    Ok(node.execute(context, inputs).await?)
}

fn zero_scratch() -> Result<ScratchReservation, sim_tensor::TensorError> {
    let (_backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1)?;
    workspace_authority.authorize_workspace(0)
}

fn native_values_semantically_equal(left: &NativeValue, right: &NativeValue) -> bool {
    match (left, right) {
        (NativeValue::Primitive { value: left }, NativeValue::Primitive { value: right }) => {
            left == right
        }
        (
            NativeValue::PreservedUnknown {
                type_name: left_type,
                value: left_value,
            },
            NativeValue::PreservedUnknown {
                type_name: right_type,
                value: right_value,
            },
        ) => left_type == right_type && left_value == right_value,
        (NativeValue::List { values: left }, NativeValue::List { values: right }) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| native_values_semantically_equal(left, right))
        }
        (NativeValue::Handle { value: left }, NativeValue::Handle { value: right }) => {
            left.handle_type() == right.handle_type()
                && left.digest_sha256() == right.digest_sha256()
        }
        _ => false,
    }
}

fn native_output_sets_semantically_equal(left: &[NativeValue], right: &[NativeValue]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| native_values_semantically_equal(left, right))
}

async fn invoke_registry_binding_at(
    registry: &NativeNodeRegistry,
    stage: &str,
) -> Result<NodeOutcome, Box<dyn Error>> {
    invoke_registry_binding(registry)
        .await
        .map_err(|error| format!("{stage}: {error}").into())
}

struct EchoPlugin {
    manifest: PluginManifest,
}

struct EchoNode;

impl RustNodeInstance for EchoNode {
    fn invoke(&mut self, invocation: &mut dyn PluginInvocation) -> Result<(), InvocationError> {
        for family in ["scalar", "tensor", "artifact", "model"] {
            for cardinality in ["single", "list"] {
                let input = format!("{family}-{cardinality}-in");
                let output = format!("{family}-{cardinality}-out");
                let state = invocation.input_state(&input)?;
                for index in 0..state.length {
                    let handle = if family == "scalar" {
                        let value = invocation.read_scalar_input(&input, index)?;
                        invocation.create_output_value(value)?
                    } else {
                        invocation.take_input(&input, index)?
                    };
                    invocation.read_handle(handle)?;
                    invocation.push_output(&output, handle)?;
                    if !matches!(
                        invocation.read_handle(handle),
                        Err(InvocationError::RevokedHandle)
                    ) {
                        return Err(InvocationError::PluginFailure(
                            "pushed handle was not revoked".to_owned(),
                        ));
                    }
                }
                invocation.finish_output(&output, state.present)?;
            }
        }
        Ok(())
    }

    fn cancel(&mut self, _reason: CancelReason) {}
}

impl RustSimPlugin for EchoPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn create_node(&self, node_id: &str) -> Result<Box<dyn RustNodeInstance>, PluginContractError> {
        if node_id == "echo" {
            Ok(Box::new(EchoNode))
        } else {
            Err(PluginContractError::UnknownNode(node_id.to_owned()))
        }
    }
}

fn component_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
    let fixture = include_str!("../../sim_plugin_host/tests/fixtures/list_ports");
    let (contract, base64) = fixture
        .split_once("[component-base64]\n")
        .ok_or("component fixture marker is missing")?;
    assert_eq!(
        contract,
        include_str!("../../sim_plugin_host/tests/fixtures/list_ports_component/port_contract.txt")
    );
    let component = decode_base64(base64.trim())?;
    assert_no_wasi_component(&component)?;
    Ok(component)
}

fn provider_component_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
    decode_base64(include_str!("../../sim_plugin_host/tests/fixtures/provider_component").trim())
}

fn assert_no_wasi_component(component: &[u8]) -> Result<(), Box<dyn Error>> {
    for forbidden in [b"wasi:".as_slice(), b"wasi_snapshot_preview1".as_slice()] {
        if component
            .windows(forbidden.len())
            .any(|window| window == forbidden)
        {
            return Err("compiled plugin fixture unexpectedly imports WASI".into());
        }
    }
    let host_interface = b"zed:sim-plugin/host@1.0.0";
    if !component
        .windows(host_interface.len())
        .any(|window| window == host_interface)
    {
        return Err("compiled plugin fixture is missing its declared host import".into());
    }
    Ok(())
}

fn hang_component_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
    decode_base64(include_str!(
        "../../sim_plugin_host/tests/fixtures/hang_component"
    ))
}

fn decode_base64(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let encoded = value
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    if encoded.is_empty() || encoded.len() % 4 != 0 {
        return Err("component fixture has invalid base64 length".into());
    }
    let chunk_count = encoded.len() / 4;
    let mut decoded = Vec::with_capacity(chunk_count.saturating_mul(3));
    for (chunk_index, chunk) in encoded.chunks_exact(4).enumerate() {
        let &[first, second, third, fourth] = chunk else {
            return Err("component fixture base64 chunk is malformed".into());
        };
        let first = decode_base64_character(first)?
            .ok_or("base64 padding appeared in the first position")?;
        let second = decode_base64_character(second)?
            .ok_or("base64 padding appeared in the second position")?;
        let third = decode_base64_character(third)?;
        let fourth = decode_base64_character(fourth)?;
        decoded.push((first << 2) | (second >> 4));
        match (third, fourth) {
            (Some(third), Some(fourth)) => {
                decoded.push((second << 4) | (third >> 2));
                decoded.push((third << 6) | fourth);
            }
            (Some(third), None) if chunk_index + 1 == chunk_count => {
                decoded.push((second << 4) | (third >> 2));
            }
            (None, None) if chunk_index + 1 == chunk_count => {}
            _ => return Err("component fixture has invalid base64 padding".into()),
        }
    }
    Ok(decoded)
}

fn decode_base64_character(value: u8) -> Result<Option<u8>, Box<dyn Error>> {
    let decoded = match value {
        b'A'..=b'Z' => Some(value - b'A'),
        b'a'..=b'z' => Some(value - b'a' + 26),
        b'0'..=b'9' => Some(value - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => None,
        _ => return Err("component fixture contains an invalid base64 character".into()),
    };
    Ok(decoded)
}

fn assert_native_plugin_sources(workspace_root: &Path) -> Result<(), Box<dyn Error>> {
    let paths = [
        "crates/sim_plugin_sdk/Cargo.toml",
        "crates/sim_plugin_sdk/src/sim_plugin_sdk.rs",
        "crates/sim_plugin_sdk/src/type_ids.rs",
        "crates/sim_plugin_host/Cargo.toml",
        "crates/sim_plugin_host/src/sim_plugin_host.rs",
        "crates/sim_plugin_host/src/capabilities.rs",
        "crates/sim_plugin_host/src/legacy_mapping.rs",
    ];
    let forbidden = [
        "std::process::command",
        "smol::process::command",
        "tokio::process",
        "command::new(",
        "python3",
        "python -",
        "node -e",
        "javascriptcore",
    ];
    for relative in paths {
        let source = fs::read_to_string(workspace_root.join(relative))?;
        for marker in forbidden {
            assert!(
                !source.to_ascii_lowercase().contains(marker),
                "production plugin source `{relative}` contains forbidden execution marker `{marker}`"
            );
        }
    }
    Ok(())
}

fn repository_rust_sources(root: &Path) -> Result<Vec<(PathBuf, String)>, Box<dyn Error>> {
    fn visit(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
        let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if sim_test_support::is_apple_double_metadata(&path) {
                continue;
            }
            if path.is_dir() {
                if !matches!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some("target" | ".git" | ".agents" | "projects" | "node_modules")
                ) {
                    visit(&path, files)?;
                }
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut paths = Vec::new();
    visit(root, &mut paths)?;
    paths.sort();
    paths
        .into_iter()
        .map(|path| Ok((path.clone(), fs::read_to_string(path)?)))
        .collect::<Result<Vec<_>, std::io::Error>>()
        .map_err(Into::into)
}

fn source_occurrences(sources: &[(PathBuf, String)], needle: &str) -> Vec<String> {
    let mut occurrences = Vec::new();
    for (path, source) in sources {
        if path.file_name().and_then(|name| name.to_str()) == Some("plugin_e2e.rs") {
            continue;
        }
        for (line_index, line) in source.lines().enumerate() {
            if line.contains(needle) {
                occurrences.push(format!("{}:{}", path.display(), line_index + 1));
            }
        }
    }
    occurrences
}

fn source_digest(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn target_directory(workspace_root: &Path) -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from) {
        Some(target) if target.is_absolute() => target,
        Some(target) => workspace_root.join(target),
        None => workspace_root.join("target"),
    }
}

fn api_catalog_projection_is_exact() -> Result<bool, Box<dyn Error>> {
    let profile_id = ProfileId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_2100));
    let mut presentation = ExecutionPresentationService::new(16)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    let services = Arc::new(NativeRuntimeHttpServices::new(
        profile_id,
        sim_runtime::ExecutionPresentationOwner::ephemeral(presentation),
        Arc::new(DisconnectedExecutionController),
        native_image_registry_projection()?,
    )?);
    let capabilities = services.http_capabilities()?;
    let security_config = ApiSecurityConfig::loopback();
    let state_directory =
        std::env::temp_dir().join(format!("sim-plugin-e2e-api-{}", std::process::id()));
    if state_directory.try_exists()? {
        fs::remove_dir_all(&state_directory)?;
    }
    fs::create_dir(&state_directory)?;
    let host = NativeApiHost::new(
        profile_id.0.to_string(),
        services,
        HttpLimits::default(),
        capabilities,
        WebSocketLimits::default(),
        security_config,
        Arc::new(PermissionPolicy::native_runtime_services(
            profile_id.0.to_string(),
        )?),
        Arc::new(ArtifactIdempotencySnapshotStore::from_directory(
            &state_directory,
            "idempotency.json",
        )?),
    )?;
    let response = host.route_http(
        HttpRequest::new(HttpMethod::Get, "/object_info"),
        HostRequestContext::embedded_loopback(std::net::Ipv4Addr::LOCALHOST.into(), 1)?,
    )?;
    drop(host);
    fs::remove_dir_all(state_directory)?;
    let HttpBody::Json(body) = response.body else {
        return Ok(false);
    };
    let Some(body) = body.as_object() else {
        return Ok(false);
    };
    let bindings = native_image_catalog_bindings()?;
    let catalog = CatalogNodeRegistry::built_in()?;
    if body.len() != catalog.registered().len() + catalog.inactive().len() {
        return Ok(false);
    }
    Ok(bindings.iter().all(|(class_type, binding)| {
        let Some(projected) = body.get(class_type) else {
            return false;
        };
        let Some(schema) = catalog.source_schema(class_type) else {
            return false;
        };
        let Some(python_module) = catalog.source_python_module(class_type) else {
            return false;
        };
        let output_types = schema
            .outputs
            .iter()
            .map(|output| json!(output.source_type_name))
            .collect::<Vec<_>>();
        let output_names = schema
            .outputs
            .iter()
            .map(|output| {
                json!(
                    output
                        .display_name
                        .as_ref()
                        .or(output.source_name.as_ref())
                        .unwrap_or(&output.source_type_name)
                )
            })
            .collect::<Vec<_>>();
        let Ok(source_schema) = serde_json::to_value(schema) else {
            return false;
        };
        let source_v1 = schema.provenance == sim_nodes::NativeSchemaProvenance::SourceV1;
        let deprecated_is_exact = if source_v1 && !schema.presentation.is_deprecated {
            projected.get("deprecated").is_none()
        } else {
            projected["deprecated"] == schema.presentation.is_deprecated
        };
        let experimental_is_exact = if source_v1 && !schema.presentation.is_experimental {
            projected.get("experimental").is_none()
        } else {
            projected["experimental"] == schema.presentation.is_experimental
        };
        let exact = projected["name"] == class_type.as_str()
            && projected["display_name"] == binding.catalog.display_name
            && projected["description"] == binding.native.description
            && projected["python_module"] == python_module
            && projected["category"] == binding.catalog.category
            && projected["output"] == serde_json::Value::Array(output_types)
            && projected["output_name"] == serde_json::Value::Array(output_names)
            && projected["output_node"] == binding.native.output_node
            && deprecated_is_exact
            && experimental_is_exact
            && projected["zed_schema"] == source_schema;
        if !exact {
            eprintln!("{class_type} catalog/API mismatch: {projected:#}");
        }
        exact
    }))
}

async fn exercise_extension_store_component_lifecycle(
    executor: BackgroundExecutor,
    component: &[u8],
    signed_manifest: &PluginManifest,
    trust: &PluginTrustPolicy,
) -> Result<BTreeMap<&'static str, bool>, Box<dyn Error>> {
    let component_host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust.clone(),
        permission_policy(signed_manifest)?,
        ComponentExecutionBoundary::conformance_in_process(component_resources()?),
        conformance_component_limits(),
        native_image_registry_projection()?,
    )?;
    let component_router = ComponentHostRouter::new(component_host.clone());
    let installed_directory = PathBuf::from("/component-host-validation/installed");
    let fake_filesystem = FakeFs::new(executor);
    let filesystem: Arc<dyn Fs> = fake_filesystem.clone();
    write_component_pair(
        &fake_filesystem,
        &installed_directory,
        "test.echo-extension",
        &serde_json::to_vec(signed_manifest)?,
        component,
    )
    .await?;
    let installed_entry = extension_entry("test.echo-extension", "1.2.3")?;
    let errors = synchronize_extension_store_with_router(
        filesystem.clone(),
        &installed_directory,
        std::slice::from_ref(&installed_entry),
        &component_router,
    )
    .await;
    if !errors.is_empty() {
        return Err(format!("component lifecycle install failed: {errors:?}").into());
    }
    let installed_plugin = component_host.installed_plugin("test.echo-extension")?;
    let registry = component_host.registry_snapshot()?;
    let descriptor = registry
        .descriptor("echo")
        .ok_or("component lifecycle did not install the echo descriptor")?;
    assert_eq!(descriptor.implementation_version, "1.0.0");
    assert!(registry.descriptor_is_bound("echo"));
    assert!(registry.node("echo").is_some());
    assert_eq!(
        registry.implementation_namespace("echo"),
        Some(signed_manifest.identifier.as_str())
    );
    assert!(matches!(
        invoke_registry_binding(&registry).await?,
        NodeOutcome::Values { .. }
    ));
    exercise_native_registry_value_boundary(&registry).await?;

    let mut updated_manifest = signed_manifest.clone();
    updated_manifest.provenance.registry = Some("fixture://signed-registry/update".to_owned());
    sign(trust, &mut updated_manifest)?;
    write_component_pair(
        &fake_filesystem,
        &installed_directory,
        "test.echo-extension",
        &serde_json::to_vec(&updated_manifest)?,
        component,
    )
    .await?;
    let errors = synchronize_extension_store_with_router(
        filesystem.clone(),
        &installed_directory,
        std::slice::from_ref(&installed_entry),
        &component_router,
    )
    .await;
    if !errors.is_empty() {
        return Err(format!("component lifecycle update failed: {errors:?}").into());
    }
    assert!(matches!(
        component_host.invoke(
            &installed_plugin,
            "echo",
            invocation_inputs()?,
            CancellationToken::default(),
        ),
        Err(ComponentHostError::Revoked(_))
    ));
    assert!(invoke_registry_binding(&registry).await.is_err());
    let updated_plugin = component_host.installed_plugin("test.echo-extension")?;
    let updated_registry = component_host.registry_snapshot()?;
    assert!(matches!(
        invoke_registry_binding(&updated_registry).await?,
        NodeOutcome::Values { .. }
    ));

    let restarted_host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust.clone(),
        permission_policy(&updated_manifest)?,
        ComponentExecutionBoundary::conformance_in_process(component_resources()?),
        conformance_component_limits(),
        native_image_registry_projection()?,
    )?;
    let restarted_router = ComponentHostRouter::new(restarted_host.clone());
    let errors = synchronize_extension_store_with_router(
        filesystem.clone(),
        &installed_directory,
        std::slice::from_ref(&installed_entry),
        &restarted_router,
    )
    .await;
    if !errors.is_empty() {
        return Err(format!("component lifecycle restart failed: {errors:?}").into());
    }
    assert!(matches!(
        invoke_registry_binding(&restarted_host.registry_snapshot()?).await?,
        NodeOutcome::Values { .. }
    ));

    write_component_pair(
        &fake_filesystem,
        &installed_directory,
        "test.echo-extension",
        br#"{}"#,
        component,
    )
    .await?;
    let errors = synchronize_extension_store_with_router(
        filesystem.clone(),
        &installed_directory,
        std::slice::from_ref(&installed_entry),
        &component_router,
    )
    .await;
    assert!(errors.contains_key(ComponentHostRouter::new(component_host.clone()).adapter_id()));
    assert!(matches!(
        invoke_registry_binding(&component_host.registry_snapshot()?).await?,
        NodeOutcome::Values { .. }
    ));

    let errors = synchronize_extension_store_with_router(
        filesystem.clone(),
        &installed_directory,
        &[],
        &component_router,
    )
    .await;
    if !errors.is_empty() {
        return Err(format!("component lifecycle uninstall failed: {errors:?}").into());
    }
    assert!(matches!(
        component_host.invoke(
            &updated_plugin,
            "echo",
            invocation_inputs()?,
            CancellationToken::default(),
        ),
        Err(ComponentHostError::Revoked(_))
    ));

    let unsafe_identifiers_rejected = {
        let mut rejected = true;
        for unsafe_identifier in ["../escape", "nested/escape", r"nested\escape"] {
            let unsafe_entry = extension_entry(unsafe_identifier, "1.0.0")?;
            let errors = synchronize_extension_store_with_router(
                filesystem.clone(),
                &installed_directory,
                &[unsafe_entry],
                &component_router,
            )
            .await;
            rejected &= component_inventory_error_contains(
                &errors,
                &component_host,
                "one normal path component",
            );
        }
        rejected
    };

    let missing_pair_id = "test.missing-component-pair";
    let missing_pair_directory = installed_directory.join(missing_pair_id);
    fake_filesystem
        .insert_tree(&missing_pair_directory, json!({}))
        .await;
    fake_filesystem
        .insert_file(
            missing_pair_directory.join(extension_host::SIM_COMPONENT_MANIFEST_FILE),
            serde_json::to_vec(signed_manifest)?,
        )
        .await;
    let errors = synchronize_extension_store_with_router(
        filesystem.clone(),
        &installed_directory,
        &[extension_entry(missing_pair_id, "1.0.0")?],
        &component_router,
    )
    .await;
    let missing_pair_rejected = component_inventory_error_contains(
        &errors,
        &component_host,
        "must provide both sim-plugin.json and sim-plugin.wasm",
    );

    let symlink_parent_id = "test.symlink-parent";
    let symlink_parent_source = Path::new("/component-host-validation/outside/symlink-parent");
    write_component_pair(
        &fake_filesystem,
        symlink_parent_source
            .parent()
            .ok_or("symlink source has no parent")?,
        symlink_parent_source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("symlink source has no final component")?,
        &serde_json::to_vec(signed_manifest)?,
        component,
    )
    .await?;
    fake_filesystem
        .insert_symlink(
            installed_directory.join(symlink_parent_id),
            symlink_parent_source.to_path_buf(),
        )
        .await;
    let errors = synchronize_extension_store_with_router(
        filesystem.clone(),
        &installed_directory,
        &[extension_entry(symlink_parent_id, "1.0.0")?],
        &component_router,
    )
    .await;
    let symlink_parent_rejected =
        component_inventory_error_contains(&errors, &component_host, "not a direct real directory");

    let symlink_file_id = "test.symlink-final-file";
    let symlink_file_directory = installed_directory.join(symlink_file_id);
    fake_filesystem
        .insert_tree(&symlink_file_directory, json!({}))
        .await;
    fake_filesystem
        .insert_file(
            symlink_file_directory.join(extension_host::SIM_COMPONENT_MANIFEST_FILE),
            serde_json::to_vec(signed_manifest)?,
        )
        .await;
    let symlink_component_source =
        Path::new("/component-host-validation/outside/component-fixture.wasm");
    fake_filesystem
        .insert_file(symlink_component_source, component.to_vec())
        .await;
    fake_filesystem
        .insert_symlink(
            symlink_file_directory.join(extension_host::SIM_COMPONENT_BINARY_FILE),
            symlink_component_source.to_path_buf(),
        )
        .await;
    let errors = synchronize_extension_store_with_router(
        filesystem.clone(),
        &installed_directory,
        &[extension_entry(symlink_file_id, "1.0.0")?],
        &component_router,
    )
    .await;
    let symlink_final_file_rejected =
        component_inventory_error_contains(&errors, &component_host, "is invalid");

    let oversized_manifest_id = "test.oversized-manifest";
    write_component_pair(
        &fake_filesystem,
        &installed_directory,
        oversized_manifest_id,
        &vec![b'x'; 4 * 1024 * 1024 + 1],
        component,
    )
    .await?;
    let errors = synchronize_extension_store_with_router(
        filesystem,
        &installed_directory,
        &[extension_entry(oversized_manifest_id, "1.0.0")?],
        &component_router,
    )
    .await;
    let oversized_inventory_rejected =
        component_inventory_error_contains(&errors, &component_host, "is oversized");

    Ok(BTreeMap::from([
        ("extension_store_fixed_pair_install", true),
        ("extension_store_registry_actual_node_bound", true),
        ("extension_store_registry_plugin_namespace_exact", true),
        ("extension_store_registry_binding_invocation", true),
        ("extension_store_successful_update", true),
        ("extension_store_old_registry_binding_revoked", true),
        ("extension_store_restart_snapshot_restored", true),
        ("extension_store_failed_update_is_atomic", true),
        ("extension_store_failure_is_visible", true),
        ("extension_store_uninstall_revokes_handles", true),
        (
            "extension_store_rejects_unsafe_identifier",
            unsafe_identifiers_rejected,
        ),
        (
            "extension_store_rejects_missing_fixed_pair",
            missing_pair_rejected,
        ),
        (
            "extension_store_rejects_symlink_parent",
            symlink_parent_rejected,
        ),
        (
            "extension_store_rejects_symlink_final_file",
            symlink_final_file_rejected,
        ),
        (
            "extension_store_rejects_oversized_inventory",
            oversized_inventory_rejected,
        ),
    ]))
}

#[gpui::test(seed = 21027)]
async fn val_worker_plugin_001(executor: BackgroundExecutor) {
    executor.allow_parking();
    let result: Result<(), Box<dyn Error>> = async {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        let profile_id = ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?);
        exercise_provider_world_private_worker(profile_id).await?;
        let component = component_fixture()?;
        let trust = trust_policy()?;
        let mut signed_manifest = manifest(format!("{:x}", Sha256::digest(&component)))?;
        sign(&trust, &mut signed_manifest)?;
        let (asset_directory, assets) = worker_plugin_assets()?;
        let (broker, clock, provider, credentials) = worker_plugin_broker(assets.clone())?;
        let launch = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_plugin_worker_fixture"),
            profile_id,
            WorkerId(Uuid::from_u128(0x21027)),
            "worker-plugin-v1",
            8 * 1024 * 1024 * 1024,
        );
        let boundary = ComponentExecutionBoundary::private_worker(
            PrivateWorkerPluginExecutor::new(launch, broker)?,
        );
        let component_host = ComponentHost::new(
            ComponentRuntime::no_wasi()?,
            trust.clone(),
            permission_policy(&signed_manifest)?,
            boundary,
            conformance_component_limits(),
            native_image_registry_projection()?,
        )?;
        let component_router = ComponentHostRouter::new(component_host.clone());
        let installed_directory = PathBuf::from("/worker-plugin-validation/installed");
        let fake_filesystem = FakeFs::new(executor.clone());
        write_component_pair(
            &fake_filesystem,
            &installed_directory,
            "test.echo-extension",
            &serde_json::to_vec(&signed_manifest)?,
            &component,
        )
        .await?;
        let errors = synchronize_extension_store_with_router(
            fake_filesystem.clone(),
            &installed_directory,
            &[extension_entry("test.echo-extension", "1.2.3")?],
            &component_router,
        )
        .await;
        if !errors.is_empty() {
            return Err(format!("private worker component install failed: {errors:?}").into());
        }

        clock.advance(Duration::from_millis(1_234));
        let registry = component_host.registry_snapshot()?;
        let first = invoke_registry_binding_at(&registry, "initial private-worker invocation").await?;
        let second =
            invoke_registry_binding_at(&registry, "repeated private-worker invocation").await?;
        let (
            NodeOutcome::Values {
                outputs: first_outputs,
                ui: first_ui,
                effects: first_effects,
            },
            NodeOutcome::Values {
                outputs: second_outputs,
                ui: second_ui,
                effects: second_effects,
            },
        ) = (first, second)
        else {
            return Err("private worker plugin returned a non-value outcome".into());
        };
        assert_ne!(first_outputs, second_outputs);
        assert!(native_output_sets_semantically_equal(
            &first_outputs,
            &second_outputs
        ));
        assert_eq!(first_ui, second_ui);
        assert_eq!(first_effects, second_effects);
        assert_eq!(first_effects.len(), 1);
        first_effects[0].validate()?;
        assert_eq!(first_effects[0].kind(), NativePreparedEffectKind::Output);
        assert_eq!(provider.calls.load(Ordering::Acquire), 2);
        assert_eq!(credentials.read_calls.load(Ordering::Acquire), 2);
        assert_eq!(credentials.presence_calls.load(Ordering::Acquire), 2);
        let authorized_provider_calls = match provider.authorized.lock() {
            Ok(calls) => calls.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        assert_eq!(
            authorized_provider_calls,
            vec![
                (
                    "demo".to_owned(),
                    "https://demo.invalid/v1/generate".to_owned(),
                    Some("secret.demo".to_owned()),
                    Some(b"worker-secret-value".to_vec()),
                );
                2
            ]
        );
        let cancelled = CancellationToken::default();
        let (
            cancelled_context,
            cancelled_inputs,
            cancelled_generation,
            _cancelled_store,
        ) = registry_invocation(
            PromptId(Uuid::from_u128(11)),
            AttemptId(Uuid::from_u128(12)),
            NodeId("cancelled-echo-fixture".to_owned()),
            cancelled.clone(),
        )?;
        let staged_input_count = cancelled_generation.len();
        cancelled.cancel();
        let cancelled_error = registry
            .node("echo")
            .ok_or("component registry has no echo binding")?
            .execute(cancelled_context, cancelled_inputs)
            .await
            .expect_err("pre-cancelled private invocation must fail");
        assert!(cancelled_error.message.contains("cancel"));
        assert_eq!(cancelled_generation.len(), staged_input_count);
        assert_eq!(provider.calls.load(Ordering::Acquire), 2);

        let roots = assets
            .lock()
            .map_err(|_| "worker plugin asset service lock is unavailable")?
            .roots()
            .clone();
        let mut committer = OutputCommitter::open(roots)?;
        let output_authorization = authorize_native_output_committer(TEST_PROFILE_ID)?;
        let scope = OutputExecutionScope {
            profile_id,
            prompt_id: PromptId(Uuid::from_u128(1)),
            attempt_id: AttemptId(Uuid::from_u128(2)),
        };
        let output_proposals = vec![PluginOutputProposal {
            identifier: "private-worker-output".to_owned(),
            namespace: "outputs".to_owned(),
            name: "guest.bin".to_owned(),
            bytes: b"guest-output".to_vec(),
        }];
        let receipts = {
            let mut assets = assets
                .lock()
                .map_err(|_| "worker plugin asset service lock is unavailable")?;
            PluginOutputPublicationAdapter.publish(
                &scope,
                &output_proposals,
                &mut committer,
                &mut assets,
                &output_authorization,
                &CancellationToken::default(),
            )?
        };
        assert_eq!(receipts.len(), 1);
        assert_eq!(committer.operations().len(), 1);
        let duplicate = {
            let mut assets = assets
                .lock()
                .map_err(|_| "worker plugin asset service lock is unavailable")?;
            PluginOutputPublicationAdapter.publish(
                &scope,
                &output_proposals,
                &mut committer,
                &mut assets,
                &output_authorization,
                &CancellationToken::default(),
            )
        };
        assert!(duplicate.is_err());
        assert_eq!(committer.committed_receipts_for_scope(&scope)?, receipts);

        let hang_component = hang_component_fixture()?;
        let mut hang_manifest = manifest(format!("{:x}", Sha256::digest(&hang_component)))?;
        sign(&trust, &mut hang_manifest)?;
        write_component_pair(
            &fake_filesystem,
            &installed_directory,
            "test.echo-extension",
            &serde_json::to_vec(&hang_manifest)?,
            &hang_component,
        )
        .await?;
        let errors = synchronize_extension_store_with_router(
            fake_filesystem.clone(),
            &installed_directory,
            &[extension_entry("test.echo-extension", "1.2.3")?],
            &component_router,
        )
        .await;
        let trapped = errors
            .get("zed.sim.component-host.v1")
            .ok_or("fuel-bounded component update unexpectedly succeeded")?;
        assert!(
            trapped.contains("fuel") || trapped.contains("trap"),
            "unexpected private component preflight failure: {trapped}"
        );
        assert!(matches!(
            invoke_registry_binding_at(&registry, "retained component after rejected update").await?,
            NodeOutcome::Values { .. }
        ));

        write_component_pair(
            &fake_filesystem,
            &installed_directory,
            "test.echo-extension",
            &serde_json::to_vec(&signed_manifest)?,
            &component,
        )
        .await?;
        let errors = synchronize_extension_store_with_router(
            fake_filesystem.clone(),
            &installed_directory,
            &[extension_entry("test.echo-extension", "1.2.3")?],
            &component_router,
        )
        .await;
        if !errors.is_empty() {
            return Err(format!("private worker component restore failed: {errors:?}").into());
        }
        assert!(matches!(
            invoke_registry_binding_at(
                &component_host.registry_snapshot()?,
                "restored component invocation",
            )
            .await?,
            NodeOutcome::Values { .. }
        ));
        assert!(invoke_registry_binding(&registry).await.is_err());

        let errors = synchronize_extension_store_with_router(
            fake_filesystem.clone(),
            &installed_directory,
            &[],
            &component_router,
        )
        .await;
        if !errors.is_empty() {
            return Err(format!("private worker component removal failed: {errors:?}").into());
        }
        assert!(component_host.registry_snapshot()?.node("echo").is_none());

        let (denied_broker, denied_clock, denied_provider, denied_credentials) =
            worker_plugin_broker_with_provider_mode(assets.clone(), ProviderMode::Disabled)?;
        let denied_launch = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_plugin_worker_fixture"),
            profile_id,
            WorkerId(Uuid::from_u128(0x21028)),
            "worker-plugin-denial-v1",
            8 * 1024 * 1024 * 1024,
        );
        let denied_host = ComponentHost::new(
            ComponentRuntime::no_wasi()?,
            trust,
            permission_policy(&signed_manifest)?,
            ComponentExecutionBoundary::private_worker(PrivateWorkerPluginExecutor::new(
                denied_launch,
                denied_broker,
            )?),
            conformance_component_limits(),
            native_image_registry_projection()?,
        )?;
        let denied_router = ComponentHostRouter::new(denied_host.clone());
        let denied_directory = PathBuf::from("/worker-plugin-validation/denied");
        write_component_pair(
            &fake_filesystem,
            &denied_directory,
            "test.echo-extension",
            &serde_json::to_vec(&signed_manifest)?,
            &component,
        )
        .await?;
        let errors = synchronize_extension_store_with_router(
            fake_filesystem,
            &denied_directory,
            &[extension_entry("test.echo-extension", "1.2.3")?],
            &denied_router,
        )
        .await;
        if !errors.is_empty() {
            return Err(format!("denied private worker component install failed: {errors:?}").into());
        }
        denied_clock.advance(Duration::from_millis(1_234));
        let denied = invoke_registry_binding(&denied_host.registry_snapshot()?)
            .await
            .expect_err("disabled provider policy must deny the private worker");
        assert!(denied.to_string().contains("denied"));
        assert_eq!(denied_provider.calls.load(Ordering::Acquire), 0);
        assert_eq!(denied_credentials.read_calls.load(Ordering::Acquire), 0);
        assert_eq!(denied_credentials.presence_calls.load(Ordering::Acquire), 0);

        let blocking_clock = Arc::new(WorkerPluginClock::new(Instant::now()));
        let blocking_provider = Arc::new(BlockingWorkerPluginProvider::default());
        let blocking_credentials = Arc::new(WorkerPluginCredentials::default());
        let blocking_broker = PluginCapabilityBroker::new(
            assets.clone(),
            sim_model::ModelStore::new(sim_model::ParserLimits::default())?,
            ProviderPolicy::new(
                TEST_PROFILE_ID,
                ProviderMode::Enabled,
                [ProviderEndpoint::new(
                    "demo",
                    "https://demo.invalid/v1/generate",
                )?],
                [CredentialScope::new(
                    TEST_PROFILE_ID,
                    "test.echo-plugin",
                    "demo",
                    SecretId::new("secret.demo")?,
                )?],
            )?,
            blocking_provider.clone(),
            blocking_credentials.clone(),
            blocking_clock.clone(),
            PluginRngPolicy::new(
                RngProfileVersion::V2,
                RngAlgorithm::Philox4x32_10,
                21_029,
            ),
        );
        let blocking_host = ComponentHost::new(
            ComponentRuntime::no_wasi()?,
            trust_policy()?,
            permission_policy(&signed_manifest)?,
            ComponentExecutionBoundary::private_worker(PrivateWorkerPluginExecutor::new(
                WorkerLaunchConfig::new(
                    env!("CARGO_BIN_EXE_sim_plugin_worker_fixture"),
                    profile_id,
                    WorkerId(Uuid::from_u128(0x21029)),
                    "worker-plugin-cancellation-v1",
                    8 * 1024 * 1024 * 1024,
                ),
                blocking_broker,
            )?),
            conformance_component_limits(),
            native_image_registry_projection()?,
        )?;
        let blocking_router = ComponentHostRouter::new(blocking_host.clone());
        let blocking_filesystem = FakeFs::new(executor.clone());
        let blocking_directory = PathBuf::from("/worker-plugin-validation/cancellation");
        write_component_pair(
            &blocking_filesystem,
            &blocking_directory,
            "test.echo-extension",
            &serde_json::to_vec(&signed_manifest)?,
            &component,
        )
        .await?;
        let errors = synchronize_extension_store_with_router(
            blocking_filesystem,
            &blocking_directory,
            &[extension_entry("test.echo-extension", "1.2.3")?],
            &blocking_router,
        )
        .await;
        if !errors.is_empty() {
            return Err(
                format!("blocking private worker component install failed: {errors:?}").into(),
            );
        }
        blocking_clock.advance(Duration::from_millis(1_234));
        let blocking_registry = blocking_host.registry_snapshot()?;
        let blocking_node = blocking_registry
            .node("echo")
            .ok_or("blocking component registry has no echo binding")?;
        let blocking_cancellation = CancellationToken::default();
        let (blocking_context, blocking_inputs, blocking_generation, _blocking_store) =
            registry_invocation(
                PromptId(Uuid::from_u128(21)),
                AttemptId(Uuid::from_u128(22)),
                NodeId("blocking-echo-fixture".to_owned()),
                blocking_cancellation.clone(),
            )?;
        let staged_input_count = blocking_generation.len();
        let blocking_task = smol::spawn(async move {
            blocking_node
                .execute(blocking_context, blocking_inputs)
                .await
                .map_err(|error| error.to_string())
        });
        for _ in 0..500 {
            if blocking_provider.entered.load(Ordering::Acquire) {
                break;
            }
            executor.timer(Duration::from_millis(10)).await;
        }
        assert!(blocking_provider.entered.load(Ordering::Acquire));
        blocking_cancellation.cancel();
        let blocking_error = blocking_task
            .await
            .expect_err("blocking provider invocation must observe cancellation");
        assert!(
            blocking_error.contains("cancel"),
            "unexpected blocking cancellation error: {blocking_error}"
        );
        assert_eq!(blocking_generation.len(), staged_input_count);
        assert_eq!(blocking_credentials.read_calls.load(Ordering::Acquire), 1);
        assert_eq!(
            blocking_credentials
                .presence_calls
                .load(Ordering::Acquire),
            0
        );

        let loss_provider = Arc::new(WorkerLossProvider::default());
        let loss_credentials = Arc::new(WorkerPluginCredentials::default());
        let loss_clock = Arc::new(WorkerPluginClock::new(Instant::now()));
        let loss_broker = PluginCapabilityBroker::new(
            assets.clone(),
            sim_model::ModelStore::new(sim_model::ParserLimits::default())?,
            ProviderPolicy::new(
                TEST_PROFILE_ID,
                ProviderMode::Enabled,
                [ProviderEndpoint::new(
                    "demo",
                    "https://demo.invalid/v1/generate",
                )?],
                [CredentialScope::new(
                    TEST_PROFILE_ID,
                    "test.echo-plugin",
                    "demo",
                    SecretId::new("secret.demo")?,
                )?],
            )?,
            loss_provider.clone(),
            loss_credentials,
            loss_clock.clone(),
            PluginRngPolicy::new(
                RngProfileVersion::V2,
                RngAlgorithm::Philox4x32_10,
                21_030,
            ),
        );
        let loss_marker = asset_directory.path().join("worker-loss-once.marker");
        let mut loss_launch = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_plugin_worker_fixture"),
            profile_id,
            WorkerId(Uuid::from_u128(0x21030)),
            "worker-plugin-loss-v1",
            8 * 1024 * 1024 * 1024,
        );
        loss_launch.arguments = vec![
            "--exit-after-ms-once".to_owned(),
            "5000".to_owned(),
            "--exit-marker".to_owned(),
            loss_marker.to_string_lossy().into_owned(),
        ];
        let loss_host = ComponentHost::new(
            ComponentRuntime::no_wasi()?,
            trust_policy()?,
            permission_policy(&signed_manifest)?,
            ComponentExecutionBoundary::private_worker(PrivateWorkerPluginExecutor::new(
                loss_launch,
                loss_broker,
            )?),
            conformance_component_limits(),
            native_image_registry_projection()?,
        )?;
        let loss_router = ComponentHostRouter::new(loss_host.clone());
        let loss_filesystem = FakeFs::new(executor.clone());
        let loss_directory = PathBuf::from("/worker-plugin-validation/loss");
        write_component_pair(
            &loss_filesystem,
            &loss_directory,
            "test.echo-extension",
            &serde_json::to_vec(&signed_manifest)?,
            &component,
        )
        .await?;
        let errors = synchronize_extension_store_with_router(
            loss_filesystem,
            &loss_directory,
            &[extension_entry("test.echo-extension", "1.2.3")?],
            &loss_router,
        )
        .await;
        if !errors.is_empty() {
            return Err(format!("worker-loss component install failed: {errors:?}").into());
        }
        loss_clock.advance(Duration::from_millis(1_234));
        let loss_registry = loss_host.registry_snapshot()?;
        let loss_error = invoke_registry_binding(&loss_registry)
            .await
            .expect_err("injected worker loss must abort the active invocation");
        assert!(
            loss_error.to_string().contains("worker"),
            "unexpected worker-loss error: {loss_error}"
        );
        assert_eq!(loss_provider.calls.load(Ordering::Acquire), 1);
        assert!(loss_marker.try_exists()?);
        assert!(matches!(
            invoke_registry_binding_at(
                &loss_host.registry_snapshot()?,
                "post-loss restarted worker invocation",
            )
            .await?,
            NodeOutcome::Values { .. }
        ));
        assert_eq!(loss_provider.calls.load(Ordering::Acquire), 2);

        let shared_component_registry = loss_host.registry_snapshot()?;
        assert!(shared_component_registry.node("echo").is_some());
        let mut headless_presentation = ExecutionPresentationService::new(16)?;
        headless_presentation.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let headless_presentation =
            sim_runtime::ExecutionPresentationOwner::ephemeral(headless_presentation);
        let headless_events = ExecutionEventBus::new(16)?;
        let headless_state_directory = asset_directory.path().join("headless-api-state");
        fs::create_dir(&headless_state_directory)?;
        let headless = NativeHeadlessService::offline(
            headless_presentation,
            move |presentation| {
                    NativeRuntimeApiHost::with_registry(
                        profile_id,
                        presentation,
                        Arc::new(DisconnectedExecutionController),
                        &headless_events,
                        None,
                        shared_component_registry.clone(),
                        HttpLimits::default(),
                        WebSocketLimits::default(),
                        ApiSecurityConfig::loopback(),
                        Arc::new(
                            PermissionPolicy::native_runtime_services(profile_id.0.to_string())
                                .map_err(|error| {
                                    sim_api::NativeApiHostError::Runtime(error.to_string())
                                })?,
                        ),
                        Arc::new(
                            ArtifactIdempotencySnapshotStore::from_directory(
                                &headless_state_directory,
                                "idempotency.json",
                            )
                            .map_err(|error| {
                                sim_api::NativeApiHostError::Runtime(error.to_string())
                            })?,
                        ),
                    )
            },
            NativeHeadlessPolicy::default(),
        )?;
        headless.start()?;
        let catalog = headless.execute_cli(NativeCliInvocation {
            feature_id: "worker-plugin-shared-registry".to_owned(),
            operation: NativeCliOperation::NativeRequest {
                request: HttpRequest::new(HttpMethod::Get, "/object_info"),
                requires_network: false,
            },
            now_epoch_seconds: 1,
        })?;
        let NativeAutomationResult::Native {
            status,
            body: NativeAutomationBody::Json(catalog),
            ..
        } = catalog
        else {
            return Err("headless component catalog did not return JSON".into());
        };
        assert_eq!(status, 200);
        let echo_catalog = catalog
            .get("echo")
            .ok_or("signed component node is absent from the shared API registry")?;
        let signed_node = signed_manifest
            .nodes
            .iter()
            .find(|node| node.id == "echo")
            .ok_or("signed echo node is absent from the fixture manifest")?;
        let signed_output_names = signed_node
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .map(|port| json!(&port.name))
            .collect::<Vec<_>>();
        assert_eq!(echo_catalog["display_name"], json!(&signed_node.display_name));
        assert_eq!(echo_catalog["category"], json!(&signed_node.category));
        assert_eq!(echo_catalog["output_name"], json!(signed_output_names));
        assert_eq!(echo_catalog["python_module"], json!(&signed_manifest.identifier));
        headless.shutdown()?;

        let cases = BTreeMap::from([
            ("path_free_extension_snapshot_deployed", true),
            ("private_worker_process_executed_component", true),
            ("canonical_asset_service_read", true),
            ("canonical_model_store_load", true),
            ("canonical_provider_policy_authorized", true),
            ("canonical_credentials_owner_scoped_secret", true),
            ("canonical_clock_and_rng_services", true),
            ("worker_and_plugin_performed_zero_final_commits", true),
            ("sole_output_committer_published_once", true),
            ("duplicate_publication_rejected", true),
            ("repeated_execution_is_deterministic", true),
            ("pre_dispatch_cancellation_is_effect_free", true),
            ("component_update_redeploys_private_registry", true),
            ("component_trap_is_contained", true),
            ("stale_registry_binding_is_revoked", true),
            ("component_removal_revokes_registry", true),
            ("provider_denial_precedes_credential_and_network_effects", true),
            ("in_flight_provider_cancellation_is_typed_and_effect_free", true),
            ("worker_loss_aborts_active_invocation", true),
            ("next_invocation_restarts_and_redeploys_worker", true),
            ("gpui_component_host_consumes_verified_registry", true),
            ("api_consumes_same_component_registry_snapshot", true),
            ("headless_consumes_same_component_registry_snapshot", true),
            ("signed_component_presentation_is_preserved", true),
        ]);
        let artifact = json!({
            "validation": "VAL-WORKER-PLUGIN-001",
            "validation_id": "VAL-WORKER-PLUGIN-001",
            "component_sha256": format!("{:x}", Sha256::digest(&component)),
            "worker_binary": "sim_plugin_worker_fixture",
            "profile_id": TEST_PROFILE_ID,
            "cases": cases,
            "summary": {
                "passed": cases.len(),
                "failed": 0,
                "skipped": 0,
            },
            "source_digests": {
                "component_host": source_digest(&workspace_root.join("crates/sim_plugin_host/src/component_host.rs"))?,
                "private_worker": source_digest(&workspace_root.join("crates/sim_plugin_host/src/private_worker.rs"))?,
                "worker_runtime": source_digest(&workspace_root.join("crates/sim_worker/src/plugin_runtime.rs"))?,
                "runtime_broker": source_digest(&workspace_root.join("crates/sim_runtime/src/plugin_services.rs"))?,
                "worker_protocol": source_digest(&workspace_root.join("crates/sim_types/src/worker_protocol.rs"))?,
                "native_registry": source_digest(&workspace_root.join("crates/sim_runtime/src/executor.rs"))?,
                "registry_adapter": source_digest(&workspace_root.join("crates/sim_plugin_host/src/registry_adapter.rs"))?,
                "api_projection": source_digest(&workspace_root.join("crates/sim_api/src/services.rs"))?,
            },
            "skipped": [],
        });
        let mut bytes = serde_json::to_vec_pretty(&artifact)?;
        bytes.push(b'\n');
        let artifact_directory = target_directory(workspace_root).join("comfy-parity");
        fs::create_dir_all(&artifact_directory)?;
        fs::write(artifact_directory.join("val-worker-plugin-001.json"), bytes)?;
        Ok(())
    }
    .await;
    result.expect("VAL-WORKER-PLUGIN-001 failed");
}

#[gpui::test(seed = 21001)]
async fn val_plugin_host_001(executor: BackgroundExecutor) {
    executor.allow_parking();
    let result: Result<(), Box<dyn Error>> = async {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        let component = component_fixture()?;
        let trust = trust_policy()?;
        let mut signed_manifest = manifest(format!("{:x}", Sha256::digest(&component)))?;
        sign(&trust, &mut signed_manifest)?;
        let mut cases = exercise_extension_store_component_lifecycle(
            executor,
            &component,
            &signed_manifest,
            &trust,
        )
        .await?;

        let sources = repository_rust_sources(workspace_root)?;
        let extension_source =
            fs::read_to_string(workspace_root.join("crates/extension_host/src/extension_host.rs"))?;
        let extension_runtime_source =
            fs::read_to_string(workspace_root.join("crates/extension_host/src/wasm_host.rs"))?;
        let component_host_source = fs::read_to_string(
            workspace_root.join("crates/sim_plugin_host/src/component_host.rs"),
        )?;
        let plugin_host_source = fs::read_to_string(
            workspace_root.join("crates/sim_plugin_host/src/sim_plugin_host.rs"),
        )?;
        let plugin_host_source =
            sim_test_support::rust_source_before_test_module(&plugin_host_source);
        let registry_adapter_source = fs::read_to_string(
            workspace_root.join("crates/sim_plugin_host/src/registry_adapter.rs"),
        )?;
        let zed_source = fs::read_to_string(workspace_root.join("crates/zed/src/zed.rs"))?;
        let zed_source = sim_test_support::rust_source_before_test_module(&zed_source);
        let main_source = fs::read_to_string(workspace_root.join("crates/zed/src/main.rs"))?;
        let component_runtime_definitions =
            source_occurrences(&sources, "pub struct ComponentRuntime");
        let component_host_definitions = source_occurrences(&sources, "pub struct ComponentHost {");
        let lifecycle_adapter_definitions =
            source_occurrences(&sources, "pub trait ComponentLifecycleAdapter")
                .into_iter()
                .filter(|location| !location.contains("/tests/"))
                .collect::<Vec<_>>();
        let native_registry_definitions =
            source_occurrences(&sources, "pub struct NativeNodeRegistry");
        let zed_initialization = main_source.find("zed::init(cx);");
        let extension_initialization = main_source.find("extension_host::init(");

        cases.extend(BTreeMap::from([
            (
                "component_runtime_is_extension_host_owned",
                component_runtime_definitions.len() == 1
                    && component_runtime_definitions[0]
                        .contains("crates/extension_host/src/wasm_host.rs")
                    && extension_runtime_source.contains("COMPONENT_EPOCH_INTERVAL")
                    && !plugin_host_source.contains("wasmtime::Config::new()")
                    && !plugin_host_source.contains("Engine::new("),
            ),
            (
                "extension_store_is_single_component_lifecycle_owner",
                lifecycle_adapter_definitions.len() == 1
                    && lifecycle_adapter_definitions[0]
                        .contains("crates/extension_host/src/extension_host.rs")
                    && extension_source.contains("pub async fn synchronize_component_adapters(")
                    && extension_source
                        .contains("Self::component_inventory_candidate_from_entries(")
                    && extension_source.contains(
                        "Self::synchronize_component_candidate(candidate, adapters).await",
                    )
                    && extension_source
                        .contains("adapter.synchronize_candidate(candidate.clone()).await")
                    && extension_source
                        .contains("pub async fn canonical_component_inventory_candidate(")
                    && extension_source
                        .contains("let components = Self::load_installed_components("),
            ),
            (
                "production_zed_bootstrap_precedes_extension_store",
                zed_initialization.is_some()
                    && extension_initialization.is_some()
                    && zed_initialization < extension_initialization
                    && zed_source.contains("init_sim_component_host(false, cx)")
                    && zed_source.contains(
                        "let lifecycle_registration = ensure_desktop_component_lifecycle_registration(cx);",
                    )
                    && zed_source.contains(
                        "lifecycle_registration.and_then(|_| init_sim_component_host(false, cx))",
                    )
                    && zed_source.contains(
                        "impl extension_host::ComponentLifecycleAdapter for DeferredDesktopComponentLifecycleAdapter",
                    )
                    && zed_source.contains("register_component_lifecycle_adapter(")
                    && zed_source
                        .matches("register_component_lifecycle_adapter(")
                        .count()
                        == 1
                    && zed_source.contains("SimComponentHostGlobal"),
            ),
            (
                "component_host_is_single_verified_adapter",
                component_host_definitions.len() == 1
                    && component_host_definitions[0]
                        .contains("crates/sim_plugin_host/src/component_host.rs")
                    && component_host_source
                        .contains("impl ComponentLifecycleAdapter for ComponentHostRouter")
                    && !component_host_source
                        .contains("impl ComponentLifecycleAdapter for ComponentHost {"),
            ),
            (
                "native_node_registry_is_single_executable_owner",
                native_registry_definitions.len() == 1
                    && native_registry_definitions[0]
                        .contains("crates/sim_runtime/src/executor.rs")
                    && source_occurrences(&sources, "struct ExecutionRegistry").is_empty()
                    && registry_adapter_source
                        .contains("register_bound_batch_with_presentations(ordinary_bindings)")
                    && registry_adapter_source.contains("activate_provider_binding_set("),
            ),
            (
                "capability_services_are_injected_without_parallel_resources",
                source_occurrences(&sources, "struct HostResources").is_empty()
                    && component_host_source.contains("Arc<dyn PluginCapabilityServices>")
                    && !component_host_source.contains("files: BTreeMap")
                    && !component_host_source.contains("models: BTreeMap")
                    && !component_host_source.contains("secret_identifiers"),
            ),
            (
                "plugin_host_has_no_local_path_security_owner",
                source_occurrences(&sources, "fn validate_relative_path")
                    .iter()
                    .all(|location| !location.contains("crates/sim_plugin_host/"))
                    && source_occurrences(&sources, "fn validate_artifact_path")
                        .iter()
                        .all(|location| !location.contains("crates/sim_plugin_host/")),
            ),
            (
                "component_linker_has_no_wasi_authority",
                plugin_host_source.contains("Linker::<WasmStoreState>::new(self.runtime.engine())")
                    && !plugin_host_source.contains("wasmtime_wasi")
                    && !component_host_source.contains("wasmtime_wasi"),
            ),
            (
                "catalog_runtime_and_api_schema_adapter_is_exact",
                api_catalog_projection_is_exact()?,
            ),
        ]));
        if !cases.values().all(|passed| *passed) {
            return Err(format!("VAL-PLUGIN-HOST-001 cases failed: {cases:#?}").into());
        }

        let fixture_paths = [
            "crates/sim_test_support/parity/catalogs/authoritative-ownership.csv",
            "crates/sim_test_support/parity/catalogs/backend-nodes.csv",
            "crates/sim_test_support/parity/ownership-policy.json",
            "crates/sim_api/src/http.rs",
            "crates/sim_api/src/services.rs",
            "crates/sim_nodes/src/registry_generator.rs",
            "crates/sim_nodes/src/slices/native_image.descriptors.json",
            "crates/sim_nodes/src/slices/native_image.rs",
            "crates/sim_plugin_host/src/capabilities.rs",
            "crates/sim_plugin_host/src/sim_plugin_host.rs",
            "crates/sim_plugin_host/src/component_host.rs",
            "crates/sim_plugin_host/src/legacy_mapping.rs",
            "crates/sim_plugin_host/src/registry_adapter.rs",
            "crates/sim_plugin_host/tests/fixtures/capabilities",
            "crates/sim_plugin_host/tests/fixtures/list_ports",
            "crates/sim_plugin_host/tests/fixtures/list_ports_component/guest.rs",
            "crates/sim_plugin_host/tests/fixtures/list_ports_component/port_contract.txt",
            "crates/sim_plugin_sdk/src/sim_plugin_sdk.rs",
            "crates/sim_plugin_sdk/src/type_ids.rs",
            "crates/sim_plugin_sdk/wit/sim-plugin.wit",
            "crates/sim_runtime/src/executor.rs",
            "crates/sim_runtime/src/native_execution_controller.rs",
            "crates/sim_runtime/src/permissions.rs",
            "crates/sim_runtime/src/prompt_compiler.rs",
            "crates/sim_runtime/src/trust.rs",
            "crates/sim_test_support/Cargo.toml",
            "crates/sim_test_support/tests/plugin_e2e.rs",
            "crates/extension_host/src/extension_host.rs",
            "crates/extension_host/src/wasm_host.rs",
            "crates/zed/src/main.rs",
            "crates/zed/src/zed.rs",
        ];
        let fixture_digests = fixture_paths
            .into_iter()
            .map(|relative| {
                Ok((
                    relative.to_owned(),
                    source_digest(&workspace_root.join(relative))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        let artifact = json!({
            "validation": "VAL-PLUGIN-HOST-001",
            "validation_id": "VAL-PLUGIN-HOST-001",
            "scope": "extension-store-component-host-production-adapter",
            "environment": {
                "operating_system": std::env::consts::OS,
                "architecture": std::env::consts::ARCH,
                "backend": "native-rust-wasmtime-component-model",
                "network_requests": 0,
                "external_processes": 0,
                "wasi_authority": false,
            },
            "fixture_digests": fixture_digests,
            "component_fixture_sha256": format!("{:x}", Sha256::digest(&component)),
            "definition_counts": {
                "component_runtime": component_runtime_definitions.len(),
                "component_host": component_host_definitions.len(),
                "component_lifecycle_adapter": lifecycle_adapter_definitions.len(),
                "native_node_registry": native_registry_definitions.len(),
            },
            "cases": cases,
            "summary": {
                "passed": cases.len(),
                "failed": 0,
                "skipped": 0,
            },
            "skipped": [],
        });
        let mut bytes = serde_json::to_vec_pretty(&artifact)?;
        bytes.push(b'\n');
        let artifact_directory = target_directory(workspace_root).join("comfy-parity");
        fs::create_dir_all(&artifact_directory)?;
        fs::write(artifact_directory.join("val-plugin-host-001.json"), bytes)?;
        Ok(())
    }
    .await;
    result.expect("VAL-PLUGIN-HOST-001 failed");
}

fn exercise_legacy_mapping_fixture(
    manifest: &PluginManifest,
    authorization: &PluginAuthorization,
    trust: &PluginTrustPolicy,
) -> Result<BTreeMap<&'static str, bool>, Box<dyn Error>> {
    let fixture_bytes = include_bytes!("../fixtures/legacy_python_nodes/mapping_cases.json");
    let fixture: LegacyMappingFixture = serde_json::from_slice(fixture_bytes)?;
    if fixture.schema_version != 1 || fixture.production_execution != "native-rust-wasm-only" {
        return Err("legacy mapping fixture has an unsupported production contract".into());
    }

    let reference = LegacyNodeReference::new(
        fixture.custom_node.legacy_identifier.clone(),
        b"fields".to_vec(),
        b"widgets".to_vec(),
        b"links".to_vec(),
        b"extension".to_vec(),
    )?;
    let target = MappingTarget {
        plugin_identifier: manifest.identifier.clone(),
        node_identifier: "echo".to_owned(),
        node_version: ApiVersion::new(1, 0, 0),
    };
    let signed_candidate = MappingCandidate::new(
        target,
        "signed compatibility registry fixture",
        manifest,
        authorization,
    )?;
    let mut signed_resolver = LegacyMappingResolver::default();
    signed_resolver.add_signed_registry(
        fixture.custom_node.legacy_identifier.clone(),
        signed_candidate.clone(),
    )?;
    let signed_resolution = signed_resolver.resolve(&reference)?;
    let LegacyResolution::Projected {
        compatibility,
        provenance,
        rewrite_accepted,
        ..
    } = signed_resolution
    else {
        return Err("signed legacy mapping did not project".into());
    };
    if provenance.source != MappingSource::SignedRegistry || rewrite_accepted {
        return Err("signed legacy mapping provenance was not preserved".into());
    }
    let named_port = compatibility
        .port_by_name(PortDirection::Input, &fixture.custom_node.legacy_input_name)
        .ok_or("legacy input alias did not resolve")?;
    let positioned_port = compatibility
        .port_by_target_position(
            PortDirection::Input,
            fixture.custom_node.legacy_input_position,
        )
        .ok_or("legacy input position did not resolve")?;
    if named_port.target_port_id() != fixture.custom_node.target_port_id
        || positioned_port.target_port_id() != fixture.custom_node.target_port_id
        || named_port.type_id() != positioned_port.type_id()
        || named_port.cardinality() != positioned_port.cardinality()
        || named_port.presence() != positioned_port.presence()
        || named_port.serialization() != positioned_port.serialization()
    {
        return Err("legacy port translation did not preserve the typed manifest contract".into());
    }
    let explicit_input = compatibility
        .inputs()
        .first()
        .ok_or("explicit legacy input translation is absent")?;
    if explicit_input.target().target_port_id() != fixture.custom_node.target_port_id
        || !matches!(
            explicit_input.source(),
            LegacyInputSourceProjection::LegacyInput {
                legacy_input_id,
                legacy_widget_position: Some(position),
            } if legacy_input_id == &fixture.custom_node.legacy_input_name
                && *position == fixture.custom_node.legacy_input_position
        )
    {
        return Err("explicit legacy input translation changed its source mapping".into());
    }
    let explicit_output = compatibility
        .outputs()
        .first()
        .ok_or("explicit legacy output translation is absent")?;
    if explicit_output.target().target_position() != fixture.custom_node.target_output_index
        || explicit_output.legacy_output_index() != fixture.custom_node.legacy_output_index
    {
        return Err("explicit legacy output index translation changed".into());
    }
    let unmentioned_input = compatibility
        .inputs()
        .iter()
        .find(|translation| translation.target().target_port_id() == "tensor-single-in")
        .ok_or("unmentioned legacy input mapping disappeared")?;
    let unmentioned_output = compatibility
        .outputs()
        .iter()
        .find(|translation| translation.target().target_port_id() == "tensor-single-out")
        .ok_or("unmentioned legacy output mapping disappeared")?;
    if compatibility.inputs().len() != 8
        || compatibility.outputs().len() != 8
        || !matches!(
            unmentioned_input.source(),
            LegacyInputSourceProjection::LegacyInput {
                legacy_input_id,
                legacy_widget_position: None,
            } if legacy_input_id == "tensor-single-in"
        )
        || unmentioned_output.target().target_position() != 2
        || unmentioned_output.legacy_output_index() != 2
    {
        return Err("legacy sidecar translations replaced unmentioned default mappings".into());
    }

    let mut user_resolver = LegacyMappingResolver::default();
    user_resolver.set_user_choice(
        fixture.custom_node.legacy_identifier.clone(),
        signed_candidate.clone(),
    )?;
    let LegacyResolution::Projected { provenance, .. } = user_resolver.resolve(&reference)? else {
        return Err("explicit user legacy mapping did not project".into());
    };
    if provenance.source != MappingSource::UserChoice {
        return Err("explicit user legacy mapping lost provenance".into());
    }

    let mut workflow_resolver = LegacyMappingResolver::default();
    workflow_resolver.set_user_choice(
        fixture.custom_node.legacy_identifier.clone(),
        signed_candidate.clone(),
    )?;
    workflow_resolver.set_workflow_pin(
        fixture.custom_node.legacy_identifier.clone(),
        signed_candidate,
    )?;
    let LegacyResolution::Projected { provenance, .. } = workflow_resolver.resolve(&reference)?
    else {
        return Err("workflow-pinned legacy mapping did not project".into());
    };
    if provenance.source != MappingSource::WorkflowPin {
        return Err("workflow-pinned legacy mapping did not win precedence".into());
    }

    let mut provider_manifest = manifest.clone();
    provider_manifest
        .nodes
        .iter_mut()
        .find(|node| node.id == "echo")
        .ok_or("provider fixture target node is absent")?
        .effects = EffectPolicy::Provider;
    sign(trust, &mut provider_manifest)?;
    let provider_authorization = authorize(trust, &provider_manifest)?;
    let provider_candidate = MappingCandidate::new(
        MappingTarget {
            plugin_identifier: provider_manifest.identifier.clone(),
            node_identifier: "echo".to_owned(),
            node_version: ApiVersion::new(1, 0, 0),
        },
        "signed provider compatibility fixture",
        &provider_manifest,
        &provider_authorization,
    )?;
    let mut provider_resolver = LegacyMappingResolver::default();
    provider_resolver.add_signed_registry(
        fixture.provider_node.legacy_identifier.clone(),
        provider_candidate,
    )?;
    let provider_reference = LegacyNodeReference::new(
        fixture.provider_node.legacy_identifier,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )?;
    let LegacyResolution::Projected { compatibility, .. } =
        provider_resolver.resolve(&provider_reference)?
    else {
        return Err("legacy API node did not project to a provider node".into());
    };
    let provider_projection = compatibility
        .provider()
        .ok_or("legacy API node omitted its provider projection")?;
    if !provider_projection.scopes().iter().any(|scope| {
        scope.provider() == fixture.provider_node.provider
            && scope.endpoint() == fixture.provider_node.endpoint
    }) {
        return Err(
            "legacy API node provider scope did not match the permission projection".into(),
        );
    }

    let unresolved_reference = LegacyNodeReference::new(
        fixture.unresolved_node.legacy_identifier,
        fixture.unresolved_node.serialized_fields.into_bytes(),
        fixture.unresolved_node.serialized_widgets.into_bytes(),
        fixture.unresolved_node.serialized_links.into_bytes(),
        fixture.unresolved_node.extension_data.into_bytes(),
    )?;
    let unresolved_resolution = LegacyMappingResolver::default().resolve(&unresolved_reference)?;
    let LegacyResolution::Placeholder {
        original,
        choices,
        reason,
    } = unresolved_resolution
    else {
        return Err("unresolved legacy node did not remain a placeholder".into());
    };
    if original != unresolved_reference
        || !choices.is_empty()
        || !reason.contains("no Rust/WASM mapping")
    {
        return Err("unresolved legacy node payload or diagnostic changed".into());
    }

    let catalog = CatalogNodeRegistry::built_in()?;
    let mut local_nodes = 0_usize;
    let mut api_nodes = 0_usize;
    for descriptor in catalog.registered().values() {
        match descriptor.classification.as_str() {
            "built-in node" => local_nodes = local_nodes.saturating_add(1),
            "API node" => api_nodes = api_nodes.saturating_add(1),
            classification => {
                return Err(format!(
                    "catalog node `{}` has unsupported classification `{classification}`",
                    descriptor.node_identifier
                )
                .into());
            }
        }
        let catalog_reference = LegacyNodeReference::new(
            descriptor.node_identifier.clone(),
            descriptor.feature_id.as_bytes().to_vec(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        let resolution = LegacyMappingResolver::default().resolve(&catalog_reference)?;
        if !matches!(
            resolution,
            LegacyResolution::Placeholder {
                original,
                choices,
                ..
            } if original == catalog_reference && choices.is_empty()
        ) {
            return Err(format!(
                "catalog node `{}` was not preserved as an unresolved native placeholder",
                descriptor.node_identifier
            )
            .into());
        }
    }
    if local_nodes != 565 || api_nodes != 224 || catalog.registered().len() != 789 {
        return Err(format!(
            "legacy node catalog closure changed: local={local_nodes}, api={api_nodes}, total={}",
            catalog.registered().len()
        )
        .into());
    }

    let documented_python_contracts = include_str!(
        "../../../crates/sim_test_support/parity/catalogs/docs-extension-contracts.csv"
    )
    .lines()
    .skip(1)
    .filter(|line| line.contains(",Python V1 legacy,") || line.contains(",Python V3 legacy,"))
    .collect::<Vec<_>>();
    if documented_python_contracts.len() != 24
        || documented_python_contracts
            .iter()
            .any(|line| !line.ends_with(",prohibited"))
    {
        return Err("documented Python extension contracts lost native-only closure".into());
    }

    Ok(BTreeMap::from([
        ("legacy_signed_registry_projection", true),
        ("legacy_typed_name_and_position_translation", true),
        ("legacy_user_and_workflow_precedence", true),
        ("legacy_provider_scope_projection", true),
        ("legacy_unresolved_placeholder_lossless", true),
        ("legacy_registered_node_catalog_closure", true),
        ("legacy_documented_python_contract_closure", true),
    ]))
}

fn exercise_legacy_frontend_extension_fixture()
-> Result<BTreeMap<&'static str, bool>, Box<dyn Error>> {
    let fixture_bytes =
        include_bytes!("../fixtures/legacy_frontend_extensions/compatibility_cases.json");
    let fixture: LegacyFrontendExtensionFixture = serde_json::from_slice(fixture_bytes)?;
    if fixture.schema_version != 1 || fixture.production_execution != "rust-wasm-declarative-only" {
        return Err("legacy frontend extension fixture has an unsupported contract".into());
    }
    let source_catalog =
        include_str!("../../../crates/sim_test_support/parity/catalogs/frontend-extensions.csv");
    let disposition_catalog = include_str!(
        "../../../crates/sim_test_support/parity/catalogs/frontend-extension-dispositions.csv"
    );
    let source_ids = source_catalog
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once(',').map(|(feature_id, _)| feature_id))
        .collect::<BTreeSet<_>>();
    let disposition_lines = disposition_catalog.lines().skip(1).collect::<Vec<_>>();
    if source_ids.len() != 59 || disposition_lines.len() != 59 {
        return Err(format!(
            "frontend extension catalog closure changed: source={}, dispositions={}",
            source_ids.len(),
            disposition_lines.len()
        )
        .into());
    }
    for feature_id in &source_ids {
        let prefix = format!("{feature_id},");
        let matches = disposition_lines
            .iter()
            .filter(|line| line.starts_with(&prefix))
            .collect::<Vec<_>>();
        if matches.len() != 1 || !matches[0].ends_with(",prohibited") {
            return Err(format!(
                "frontend extension `{feature_id}` lacks one native-only disposition"
            )
            .into());
        }
    }
    let expected_cases = BTreeSet::from([
        "legacy_v1_identifier",
        "legacy_v3_identifier",
        "dom_widget",
        "web_directory",
        "commands",
        "menus",
        "settings",
        "routes",
        "serialization_callbacks",
        "explicit_port_mappings",
        "signature_denial",
        "permission_denial",
        "trap",
        "hang",
        "cancellation",
        "resource_exhaustion",
        "unresolved_placeholder",
    ]);
    let actual_cases = fixture
        .cases
        .iter()
        .map(|case| case.name.as_str())
        .collect::<BTreeSet<_>>();
    if actual_cases != expected_cases || fixture.cases.iter().any(|case| case.payload.is_empty()) {
        return Err("legacy frontend extension fixture matrix is incomplete".into());
    }
    for case in &fixture.cases {
        let Some(feature_id) = &case.feature_id else {
            if !matches!(
                case.expected_classification.as_str(),
                "versioned_wit_route" | "host_denial" | "host_isolation"
            ) {
                return Err(
                    format!("host fixture `{}` has unknown classification", case.name).into(),
                );
            }
            continue;
        };
        let prefix = format!("{feature_id},");
        let disposition = disposition_lines
            .iter()
            .find(|line| line.starts_with(&prefix))
            .ok_or_else(|| format!("fixture `{}` references an unknown feature", case.name))?;
        if !disposition.contains(&format!(",{},", case.expected_classification)) {
            return Err(format!(
                "fixture `{}` disagrees with the generated disposition",
                case.name
            )
            .into());
        }
    }
    Ok(BTreeMap::from([
        ("legacy_frontend_catalog_closure", true),
        ("legacy_frontend_fixture_matrix", true),
        ("legacy_frontend_javascript_prohibited", true),
    ]))
}

fn write_report(
    component: &[u8],
    hang_component: &[u8],
    cases: BTreeMap<&str, bool>,
) -> Result<(), Box<dyn Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let target_dir = match std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from) {
        Some(target_dir) if target_dir.is_absolute() => target_dir,
        Some(target_dir) => workspace_root.join(target_dir),
        None => workspace_root.join("target"),
    };
    let artifact_dir = target_dir.join("comfy-parity");
    fs::create_dir_all(&artifact_dir)?;
    let passed = cases.values().filter(|passed| **passed).count();
    let failed = cases.len().saturating_sub(passed);
    if failed != 0 {
        return Err(format!("VAL-E2E-003 cases failed: {cases:#?}").into());
    }
    let report = serde_json::json!({
        "validation": "VAL-E2E-003",
        "fixture_digests": {
            "compiled_wit": format!("{:x}", Sha256::digest(component)),
            "fuel_hang": format!("{:x}", Sha256::digest(hang_component)),
            "legacy_python_nodes": format!("{:x}", Sha256::digest(include_bytes!("../fixtures/legacy_python_nodes/mapping_cases.json"))),
            "legacy_frontend_extensions": format!("{:x}", Sha256::digest(include_bytes!("../fixtures/legacy_frontend_extensions/compatibility_cases.json"))),
            "frontend_extension_dispositions": format!("{:x}", Sha256::digest(include_bytes!("../../../crates/sim_test_support/parity/catalogs/frontend-extension-dispositions.csv"))),
        },
        "compiled_wit_provenance": {
            "guest_source": "crates/sim_plugin_host/tests/fixtures/list_ports_component/guest.rs",
            "target": "wasm32-unknown-unknown",
            "wasi": false,
            "toolchain": {
                "rustc": "1.95.0",
                "wit_bindgen": "0.41.0",
                "wit_component": "0.227.1",
            },
            "source_digests": {
                "cargo_manifest": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_host/tests/fixtures/list_ports_component/Cargo.toml"))),
                "cargo_lock": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_host/tests/fixtures/list_ports_component/Cargo.lock"))),
                "guest": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_host/tests/fixtures/list_ports_component/guest.rs"))),
                "port_contract": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_host/tests/fixtures/list_ports_component/port_contract.txt"))),
                "rust_toolchain": format!("{:x}", Sha256::digest(include_bytes!("../../../rust-toolchain.toml"))),
                "wit": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_sdk/wit/sim-plugin.wit"))),
            },
            "rebuild_commands": [
                "cargo build --manifest-path crates/sim_plugin_host/tests/fixtures/list_ports_component/Cargo.toml --target wasm32-unknown-unknown --release --lib --offline",
                "cargo run --manifest-path crates/sim_plugin_host/tests/fixtures/list_ports_component/Cargo.toml --bin rebuild-fixture --release --offline",
                "cargo run --manifest-path crates/sim_plugin_host/tests/fixtures/list_ports_component/Cargo.toml --bin rebuild-fixture --release --offline -- --check",
            ],
        },
        "fuel_hang_provenance": {
            "guest_source": "crates/sim_plugin_host/tests/fixtures/hang_component_source/guest.rs",
            "target": "wasm32-unknown-unknown",
            "wasi": false,
            "toolchain": {
                "rustc": "1.95.0",
                "wit_bindgen": "0.41.0",
                "wit_component": "0.227.1",
            },
            "source_digests": {
                "cargo_manifest": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_host/tests/fixtures/hang_component_source/Cargo.toml"))),
                "cargo_lock": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_host/tests/fixtures/hang_component_source/Cargo.lock"))),
                "guest": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_host/tests/fixtures/hang_component_source/guest.rs"))),
                "rust_toolchain": format!("{:x}", Sha256::digest(include_bytes!("../../../rust-toolchain.toml"))),
                "wit": format!("{:x}", Sha256::digest(include_bytes!("../../sim_plugin_sdk/wit/sim-plugin.wit"))),
            },
            "rebuild_commands": [
                "cargo build --manifest-path crates/sim_plugin_host/tests/fixtures/hang_component_source/Cargo.toml --target wasm32-unknown-unknown --release --lib --offline",
                "cargo run --manifest-path crates/sim_plugin_host/tests/fixtures/list_ports_component/Cargo.toml --bin rebuild-fixture --release --offline -- --hang",
                "cargo run --manifest-path crates/sim_plugin_host/tests/fixtures/list_ports_component/Cargo.toml --bin rebuild-fixture --release --offline -- --hang --check",
            ],
        },
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-wasmtime",
            "network_access": false,
            "external_processes": false,
        },
        "cases": cases,
        "passed": passed,
        "failed": failed,
        "skipped": [],
    });
    fs::write(
        artifact_dir.join("val-e2e-003.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

#[gpui::test(seed = 21003)]
async fn val_e2e_003(executor: BackgroundExecutor) {
    executor.allow_parking();
    let result: Result<(), Box<dyn Error>> = async {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        assert_native_plugin_sources(workspace_root)?;

        let component = component_fixture()?;
        let component_digest = format!("{:x}", Sha256::digest(&component));
        let trust = trust_policy()?;
        let mut signed_manifest = manifest(component_digest)?;
        sign(&trust, &mut signed_manifest)?;
        let authorization = authorize(&trust, &signed_manifest)?;
        let host = configured_host(conformance_component_limits())?;
        let legacy_mapping_cases =
            exercise_legacy_mapping_fixture(&signed_manifest, &authorization, &trust)?;
        let legacy_frontend_cases = exercise_legacy_frontend_extension_fixture()?;

        let compiled = host.compile_component(&component, &signed_manifest, &authorization)?;
        let invocation = host.begin_invocation(
            &signed_manifest,
            &authorization,
            "echo",
            invocation_inputs()?,
            component_resources()?,
            CancellationToken::default(),
        )?;
        let mut wasm = host.instantiate_component(&compiled, invocation)?;
        assert_eq!(
            wasm.manifest_bytes()?,
            signed_manifest.component_projection()
        );
        let instance = wasm.create_node("echo")?;
        wasm.invoke(instance)?;
        wasm.drop_node(instance)?;
        let wasm_result = wasm.finish()?;

        let lifecycle_cases = exercise_extension_store_component_lifecycle(
            executor,
            &component,
            &signed_manifest,
            &trust,
        )
        .await?;

        let rust_plugin = EchoPlugin {
            manifest: signed_manifest.clone(),
        };
        let rust_result = host.invoke_rust(
            &rust_plugin,
            &authorization,
            "echo",
            invocation_inputs()?,
            empty_services(),
            CancellationToken::default(),
        )?;
        assert_eq!(rust_result.outputs["tensor-list-out"].len(), 2);
        assert_eq!(rust_result.outputs["model-list-out"].len(), 2);
        assert_eq!(wasm_result.outputs, rust_result.outputs);
        assert_eq!(wasm_result.effects.outputs.len(), 1);
        let output_proposal = wasm_result
            .effects
            .outputs
            .first()
            .ok_or("WASM fixture did not propose its transactional output")?;
        assert_eq!(output_proposal.namespace, "outputs");
        assert_eq!(output_proposal.name, "guest.bin");
        assert_eq!(output_proposal.bytes, b"guest-output");
        assert_eq!(
            wasm_result.effects.logs,
            vec!["info: no-WASI echo fixture invoked"]
        );
        assert_eq!(
            wasm_result.effects.ui_state.get("panel.demo"),
            Some(&br#"{"invoked":true}"#.to_vec())
        );
        let route = wasm_result
            .effects
            .routes
            .first()
            .ok_or("WASM fixture did not produce its route response")?;
        assert_eq!(route.route, "route.demo");
        assert_eq!(route.status, 200);
        assert_eq!(route.body, b"guest-route");

        let wasm_cancellation = CancellationToken::default();
        let cancelled_invocation = host.begin_invocation(
            &signed_manifest,
            &authorization,
            "echo",
            invocation_inputs()?,
            component_resources()?,
            wasm_cancellation.clone(),
        )?;
        let mut cancelled_wasm = host.instantiate_component(&compiled, cancelled_invocation)?;
        let cancelled_instance = cancelled_wasm.create_node("echo")?;
        wasm_cancellation.cancel();
        assert!(matches!(
            cancelled_wasm.invoke(cancelled_instance),
            Err(PluginError::Invocation(InvocationError::Cancelled))
        ));

        let mut unsigned = signed_manifest.clone();
        unsigned.signature.value = "f".repeat(ED25519_SIGNATURE_BYTES * 2);
        assert!(matches!(
            trust.authorize_manifest(&unsigned, &permission_policy(&unsigned)?),
            Err(sim_runtime::TrustError::InvalidPluginSignature)
        ));
        assert!(host.validate(&unsigned, &authorization).is_ok());

        let mut changed_manifest = signed_manifest.clone();
        let replacement = if changed_manifest.digest_sha256.starts_with('f') {
            "e"
        } else {
            "f"
        };
        changed_manifest
            .digest_sha256
            .replace_range(..1, replacement);
        assert!(matches!(
            host.validate(&changed_manifest, &authorization),
            Err(PluginError::Trust(
                sim_runtime::TrustError::AuthorizationManifestMismatch
            ))
        ));

        let mut incompatible_major = signed_manifest.clone();
        incompatible_major.api.major = 2;
        sign(&trust, &mut incompatible_major)?;
        let incompatible_authorization = authorize(&trust, &incompatible_major)?;
        assert!(matches!(
            host.validate(&incompatible_major, &incompatible_authorization),
            Err(PluginError::Contract(
                PluginContractError::UnsupportedApi { .. }
            ))
        ));
        let mut missing_feature = signed_manifest.clone();
        missing_feature
            .api
            .required_features
            .push("future.capability".to_owned());
        sign(&trust, &mut missing_feature)?;
        let missing_feature_authorization = authorize(&trust, &missing_feature)?;
        assert!(matches!(
            host.validate(&missing_feature, &missing_feature_authorization),
            Err(PluginError::Contract(PluginContractError::MissingApiFeature(feature)))
                if feature == "future.capability"
        ));

        let mut overprivileged = signed_manifest.clone();
        overprivileged.capabilities.push(CapabilityRequest {
            kind: CapabilityKind::Filesystem,
            scope: "output-root".to_owned(),
            quota: quota(),
        });
        sign(&trust, &mut overprivileged)?;
        assert!(matches!(
            trust.authorize_manifest(&overprivileged, &permission_policy(&signed_manifest)?),
            Err(sim_runtime::TrustError::Permission(
                sim_runtime::PermissionError::Denied(capabilities)
            )) if capabilities == vec![Capability::Asset {
                namespace: "output".to_owned(),
                action: sim_runtime::AssetOperation::Read,
            }]
        ));

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            host.invoke_rust(
                &rust_plugin,
                &authorization,
                "echo",
                invocation_inputs()?,
                empty_services(),
                cancellation,
            ),
            Err(PluginError::Invocation(InvocationError::Cancelled))
        ));

        let limited_host = configured_host(ComponentLimits {
            maximum_component_bytes: component.len().saturating_sub(1),
            ..conformance_component_limits()
        })?;
        assert!(matches!(
            limited_host.compile_component(&component, &signed_manifest, &authorization),
            Err(PluginError::ComponentTooLarge)
        ));

        let reference = LegacyNodeReference::new(
            "LegacyEcho",
            b"fields".to_vec(),
            b"widgets".to_vec(),
            b"links".to_vec(),
            b"extension".to_vec(),
        )?;
        let resolver = LegacyMappingResolver::default();
        let mapping_target = MappingTarget {
            plugin_identifier: "test.echo-plugin".to_owned(),
            node_identifier: "echo".to_owned(),
            node_version: ApiVersion::new(1, 0, 0),
        };
        assert!(
            MappingCandidate::new(
                mapping_target.clone(),
                "sealed signature fixture",
                &unsigned,
                &authorization,
            )
            .is_ok()
        );
        assert!(
            MappingCandidate::new(
                mapping_target,
                "changed semantic fixture",
                &changed_manifest,
                &authorization,
            )
            .is_err()
        );
        assert!(matches!(
            resolver.resolve(&reference)?,
            LegacyResolution::Placeholder { original, choices, .. }
                if original == reference && choices.is_empty()
        ));

        let hang_component = hang_component_fixture()?;
        let mut hang_manifest = manifest(format!("{:x}", Sha256::digest(&hang_component)))?;
        sign(&trust, &mut hang_manifest)?;
        let hang_authorization = authorize(&trust, &hang_manifest)?;
        let hang_host = configured_host(ComponentLimits {
            maximum_fuel: 50_000,
            ..conformance_component_limits()
        })?;
        assert!(matches!(
            hang_host.compile_component(&hang_component, &hang_manifest, &hang_authorization),
            Err(PluginError::WasmTrap(_))
        ));

        let mut cases = BTreeMap::from([
            ("signed_rust_fixture", true),
            ("signed_wit_fixture", true),
            ("wit_guest_create_invoke_drop_finish", true),
            ("wit_guest_port_value_and_capability_imports", true),
            ("wit_guest_no_wasi", true),
            ("wit_guest_cancellation_import", true),
            ("unsigned_rejected", true),
            ("old_new_api_rejected", true),
            ("denied_grant_isolated", true),
            ("cancellation_revokes_invocation", true),
            ("component_bounds_enforced", true),
            ("trap_contained", true),
            ("hang_fuel_contained", true),
            ("legacy_placeholder_preserved", true),
            ("python_javascript_execution_absent", true),
        ]);
        cases.extend(legacy_mapping_cases);
        cases.extend(legacy_frontend_cases);
        cases.extend(lifecycle_cases);
        write_report(&component, &hang_component, cases)?;
        Ok(())
    }
    .await;
    result.expect("VAL-E2E-003 failed");
}

mod bfl_legacy_png_process_probe {
    use super::*;
    use sim_nodes::{
        NativePrimitiveType, NativeValueType, authoritative_provider_namespace_projection,
    };
    use sim_runtime::ProviderTransportRequest;

    const NODE: &str = "FluxProUltraImageNode";
    const NAMESPACE: &str = "zed.sim.provider.bfl";
    const PNG: &[u8] = include_bytes!("../fixtures/providers/bfl/downloaded-image.png");

    fn bfl_probe_manifest(digest: String) -> Result<PluginManifest, Box<dyn Error>> {
        let mut manifest = provider_manifest(digest)?;
        manifest.identifier = NAMESPACE.to_owned();
        manifest.provenance.source = "fixture://signed-bfl-legacy-png-process-probe".to_owned();
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let authority = authoritative_provider_namespace_projection()?;
        let types = TypeRegistry::built_in()?;
        let members = authority
            .namespace_members(NAMESPACE)
            .ok_or("BFL namespace is missing")?;
        assert_eq!(members.len(), 10);
        let mut nodes = Vec::new();
        let mut bindings = Vec::new();
        for identifier in members {
            let descriptor = registry
                .descriptor(identifier)
                .ok_or("BFL descriptor is absent")?;
            let presentation = registry
                .presentation(identifier)
                .ok_or("BFL presentation is absent")?;
            if !descriptor.dynamic_inputs.is_empty() {
                return Err("BFL probe requires explicit dynamic projection before build".into());
            }
            let mut ports = Vec::new();
            for input in &descriptor.inputs {
                let source_type = match input.accepted_types.members() {
                    [NativeValueType::Primitive(NativePrimitiveType::String)] => "STRING",
                    [NativeValueType::Primitive(NativePrimitiveType::Boolean)] => "BOOLEAN",
                    [NativeValueType::Primitive(NativePrimitiveType::Integer)] => "INT",
                    [NativeValueType::Primitive(NativePrimitiveType::Number)] => "FLOAT",
                    [NativeValueType::Handle(handle)]
                        if matches!(handle.type_id.as_str(), "IMAGE" | "MASK") =>
                    {
                        handle.type_id.as_str()
                    }
                    [NativeValueType::NamedPreservedUnknown(source_type)]
                        if matches!(source_type.as_str(), "COMBO" | "COMFY_DYNAMICCOMBO_V3") =>
                    {
                        source_type.as_str()
                    }
                    _ => {
                        return Err(format!(
                            "unprojected canonical BFL input {}:{}",
                            identifier, input.name
                        )
                        .into());
                    }
                };
                let mut projected = port(
                    &types,
                    &input.name,
                    PortDirection::Input,
                    source_type,
                    PortCardinality::Singular,
                    if input.hidden {
                        PortPresence::Hidden
                    } else if input.required {
                        PortPresence::Required
                    } else {
                        PortPresence::Optional
                    },
                )?;
                projected.lazy = input.lazy;
                ports.push(projected);
            }
            for output in &descriptor.outputs {
                let NativeValueType::Handle(handle) = &output.produced_type else {
                    return Err("non-image BFL output requires separate probe".into());
                };
                if handle.type_id != "IMAGE" {
                    return Err("BFL probe cannot fabricate a non-image result".into());
                }
                ports.push(port(
                    &types,
                    &output.name,
                    PortDirection::Output,
                    "IMAGE",
                    if output.is_list {
                        PortCardinality::List
                    } else {
                        PortCardinality::Singular
                    },
                    PortPresence::Required,
                )?);
            }
            nodes.push(PluginNode {
                id: identifier.clone(),
                version: ApiVersion::new(1, 0, 0),
                display_name: presentation.display_name.clone(),
                category: presentation.category.clone(),
                ports,
                determinism: DeterminismPolicy::External,
                cache: CachePolicy::Never,
                effects: EffectPolicy::Provider,
            });
            let contract = authority
                .contract_for_node(identifier)
                .ok_or("BFL Source claim is missing")?;
            bindings.push(ProviderBindingClaim {
                feature_id: contract.feature_id().to_owned(),
                node_id: identifier.clone(),
                contract_sha256: registry
                    .provider_binding_contract_sha256(
                        identifier,
                        "zed:sim-provider-transport@1",
                        "zed:sim-provider-materializer@1",
                    )?
                    .ok_or("BFL binding hash is missing")?,
                transport_schema: "zed:sim-provider-transport@1".parse()?,
                materializer_schema: "zed:sim-provider-materializer@1".parse()?,
            });
        }
        let mut binding = ProviderBindingSet {
            schema_version: PROVIDER_BINDING_SCHEMA_VERSION,
            implementation_namespace: NAMESPACE.to_owned(),
            bindings_sha256: "0".repeat(64),
            bindings,
        };
        binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
        manifest.provider_binding = Some(binding);
        manifest.nodes = nodes;
        Ok(manifest)
    }

    fn bfl_probe_inputs(prompt: &str) -> BTreeMap<String, NativeValue> {
        [
            ("prompt", NativePrimitive::String(prompt.to_owned())),
            ("prompt_upsampling", NativePrimitive::Boolean(false)),
            ("seed", NativePrimitive::Integer(1)),
            ("aspect_ratio", NativePrimitive::String("1:1".to_owned())),
            ("raw", NativePrimitive::Boolean(false)),
            ("image_prompt_strength", NativePrimitive::Number(0.1)),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_owned(), NativeValue::Primitive { value }))
        .collect()
    }

    struct BflPngActuator {
        calls: AtomicUsize,
        response: Vec<u8>,
        cancel_after_request: Option<CancellationToken>,
    }
    impl ProviderRequestActuator for BflPngActuator {
        fn execute(
            &self,
            request: &AuthorizedProviderRequest,
            secret: Option<&SecretValue>,
            body: &[u8],
            context: &PluginServiceOperationContext<'_>,
        ) -> Result<Vec<u8>, PluginServiceActuatorError> {
            context
                .check_active()
                .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?;
            if request.provider() != "fixture"
                || request.endpoint() != "https://fixture.invalid/v1/generate"
                || secret.is_some()
                || request.secret_id().is_some()
                || ProviderTransportRequest::from_bytes(body)
                    .map_err(|error| PluginServiceActuatorError::new(error.to_string()))?
                    .class_type()
                    != NODE
            {
                return Err(PluginServiceActuatorError::new(
                    "scripted BFL development request identity mismatch",
                ));
            }
            self.calls.fetch_add(1, Ordering::AcqRel);
            if let Some(cancellation) = &self.cancel_after_request {
                cancellation.cancel();
            }
            Ok(self.response.clone())
        }
    }
    #[derive(Debug)]
    struct CancelAfterPngPublishStore {
        inner: Arc<dyn NativeHandleStore>,
        cancellation: CancellationToken,
    }
    impl NativeHandleStore for CancelAfterPngPublishStore {
        fn identity(&self) -> NativeHandleStoreIdentity {
            self.inner.identity()
        }
        fn attempt_id(&self) -> AttemptId {
            self.inner.attempt_id()
        }
        fn resolve(
            &self,
            handle: &NativeOpaqueHandle,
            expected: &NativeHandleType,
            cancellation: &CancellationToken,
        ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
            self.inner.resolve(handle, expected, cancellation)
        }
        fn publish(
            &self,
            payload: NativeStoredPayload,
            cancellation: &CancellationToken,
        ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
            let handle = self.inner.publish(payload, cancellation)?;
            self.cancellation.cancel();
            Ok(handle)
        }
        fn revoke(
            &self,
            handle: &NativeOpaqueHandle,
            cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            self.inner.revoke(handle, cancellation)
        }
    }

    fn render_guest_projection(manifest: &PluginManifest) -> Result<String, Box<dyn Error>> {
        let projection = manifest.component_projection();
        if !projection.ui.is_empty()
            || !projection.routes.is_empty()
            || !projection.legacy_mappings.is_empty()
        {
            return Err(
                "development guest requires explicit projection for nonempty UI/routes/mappings"
                    .into(),
            );
        }
        let mut nodes = Vec::new();
        for node in &projection.nodes {
            let mut ports = Vec::new();
            for port in &node.ports {
                if port.default.is_some() {
                    return Err("guest projection must not silently drop a default".into());
                }
                ports.push(format!("types::Port {{ id: {:?}.to_owned(), name: {:?}.to_owned(), direction: types::PortDirection::{:?}, type_id: {:?}.to_owned(), cardinality: types::PortCardinality::{:?}, presence: types::PortPresence::{:?}, hidden: {}, lazy: {}, default: None, serialization: types::PortSerialization::{:?}, accepted_legacy_names: vec!{:?}.into_iter().map(str::to_owned).collect() }}", port.id,port.name,port.direction,port.type_id.to_string(),port.cardinality,port.presence,port.hidden,port.lazy,port.serialization,port.accepted_legacy_names));
            }
            nodes.push(format!("types::Node {{ id: {:?}.to_owned(), version: types::ApiVersion {{ major: {}, minor: {}, patch: {} }}, display_name: {:?}.to_owned(), category: {:?}.to_owned(), ports: vec![{}], determinism: types::DeterminismPolicy::{:?}, cache: types::CachePolicy::{:?}, effects: types::EffectPolicy::{:?} }}",node.id,node.version.major,node.version.minor,node.version.patch,node.display_name,node.category,ports.join(","),node.determinism,node.cache,node.effects));
        }
        let capabilities = projection.capabilities.iter().map(|capability|format!("types::CapabilityRequest {{ kind: types::CapabilityKind::{:?}, scope: {:?}.to_owned(), quota: types::CapabilityQuota {{ maximum_operations: {}, maximum_request_bytes: {}, maximum_response_bytes: {}, maximum_total_bytes: {}, maximum_handles: {}, timeout_milliseconds: {} }} }}",capability.kind,capability.scope,capability.quota.maximum_operations,capability.quota.maximum_request_bytes,capability.quota.maximum_response_bytes,capability.quota.maximum_total_bytes,capability.quota.maximum_handles,capability.quota.timeout_milliseconds)).collect::<Vec<_>>();
        let binding = manifest
            .provider_binding
            .as_ref()
            .ok_or("binding set is absent")?;
        let claims = binding.bindings.iter().map(|claim| format!("types::ProviderBindingClaim {{ feature_id: {:?}.to_owned(), node_id: {:?}.to_owned(), contract_sha256: {:?}.to_owned(), transport_schema: {:?}.to_owned(), materializer_schema: {:?}.to_owned() }}",claim.feature_id,claim.node_id,claim.contract_sha256,claim.transport_schema.to_string(),claim.materializer_schema.to_string())).collect::<Vec<_>>();
        Ok(format!(
            "fn fixture_manifest() -> types::ManifestProjection {{ types::ManifestProjection {{ component_world: {:?}.to_owned(), schema_version: {}, identifier: {:?}.to_owned(), plugin_version: types::ApiVersion {{ major: {}, minor: {}, patch: {} }}, api: types::ApiRequirement {{ major: {}, minimum_minor: {}, maximum_minor: {}, required_features: vec!{:?}.into_iter().map(str::to_owned).collect() }}, nodes: vec![{}], capabilities: vec![{}], ui: Vec::new(), routes: Vec::new(), legacy_mappings: Vec::new() }} }}\nfn fixture_binding_set() -> types::ProviderBindingSet {{ types::ProviderBindingSet {{ schema_version: {}, implementation_namespace: {:?}.to_owned(), bindings_sha256: {:?}.to_owned(), bindings: vec![{}] }} }}\n",
            projection.component_world,
            projection.schema_version,
            projection.identifier,
            projection.plugin_version.major,
            projection.plugin_version.minor,
            projection.plugin_version.patch,
            projection.api.major,
            projection.api.minimum_minor,
            projection.api.maximum_minor,
            projection.api.required_features,
            nodes.join(","),
            capabilities.join(","),
            binding.schema_version,
            binding.implementation_namespace,
            binding.bindings_sha256,
            claims.join(",")
        ))
    }

    #[test]
    #[ignore = "explicit scheduled canonical full-namespace recipe producer; no fixture/gold writes"]
    fn bfl_legacy_png_process_recipe_uses_complete_canonical_namespace()
    -> Result<(), Box<dyn Error>> {
        let manifest = bfl_probe_manifest("0".repeat(64))?;
        let directory = PathBuf::from(std::env::var("SIM_BFL_PNG_PROCESS_RECIPE_DIR")?);
        std::fs::create_dir(&directory)?;
        let projection = render_guest_projection(&manifest)?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("generated_projection.rs"))?;
        std::io::Write::write_all(&mut file, projection.as_bytes())?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("manifest-template.json"))?;
        std::io::Write::write_all(&mut file, &serde_json::to_vec_pretty(&manifest)?)?;
        let bindings = manifest
            .provider_binding
            .as_ref()
            .ok_or("BFL binding set is absent")?;
        assert_eq!(manifest.nodes.len(), 10);
        assert_eq!(bindings.bindings.len(), 10);
        assert!(
            bindings
                .bindings
                .iter()
                .any(|claim| claim.feature_id == "COMFY-NODE-0188" && claim.node_id == NODE)
        );
        Ok(())
    }

    async fn exercise_bfl_png_process(
        profile_id: ProfileId,
        mode: u8,
    ) -> Result<(), Box<dyn Error>> {
        const COMPILED_PLAN_SHA256: &str =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let component = std::fs::read(std::env::var("SIM_BFL_LEGACY_PNG_COMPONENT")?)?;
        let component_digest = format!("{:x}", Sha256::digest(&component));
        let mut manifest = bfl_probe_manifest(component_digest.clone())?;
        let trust = trust_policy()?;
        sign(&trust, &mut manifest)?;
        let binding_sha256 = manifest
            .provider_binding
            .as_ref()
            .ok_or("provider binding is absent")?
            .bindings_sha256
            .clone();
        let (_asset_directory, assets) = worker_plugin_assets()?;
        let clock = Arc::new(WorkerPluginClock::new(Instant::now()));
        let cancellation = CancellationToken::default();
        let scripted_response = match mode {
            5 | 6 | 8 | 9 => {
                let payload = canonical_image_payload(128)?;
                let registry = TypeRegistry::built_in()?;
                let value = sim_runtime::ProviderTransportValue::from_native_payload(
                    registry.resolve("IMAGE")?.to_string(),
                    ValueFamily::Tensor,
                    &payload,
                )?;
                sim_runtime::ProviderTransportResponse::checked(
                    if mode == 6 { "FluxProImageNode" } else { NODE },
                    vec![sim_runtime::ProviderTransportPort::checked(
                        if mode == 8 {
                            "wrong_output"
                        } else {
                            "output_0"
                        },
                        true,
                        vec![value],
                    )?],
                )?
                .to_bytes()?
            }
            7 => b"{not valid JSON".to_vec(),
            1 => b"not a PNG".to_vec(),
            _ => PNG.to_vec(),
        };
        let provider = Arc::new(BflPngActuator {
            calls: AtomicUsize::new(0),
            response: scripted_response,
            cancel_after_request: matches!(mode, 2 | 9).then(|| cancellation.clone()),
        });
        let cost_issuer = Arc::new(ProviderCostAcceptanceIssuer::generate(clock.utc_now())?);
        let cost_authority = Arc::new(ProviderCostApprovalAuthority::new(
            cost_issuer.clone(),
            clock.clone(),
        ));
        let broker = PluginCapabilityBroker::new_with_provider_cost_acceptance(
            assets,
            sim_model::ModelStore::new(sim_model::ParserLimits::default())?,
            ProviderPolicy::new(
                TEST_PROFILE_ID,
                ProviderMode::Enabled,
                [ProviderEndpoint::new(
                    "fixture",
                    "https://fixture.invalid/v1/generate",
                )?],
                std::iter::empty::<CredentialScope>(),
            )?,
            cost_authority.verifier()?,
            provider.clone(),
            Arc::new(WorkerPluginCredentials::default()),
            clock.clone(),
            PluginRngPolicy::new(RngProfileVersion::V2, RngAlgorithm::Philox4x32_10, 21_078),
        );
        let receipt_issuer = Arc::new(ProviderResultReceiptIssuer::generate(clock.utc_now())?);
        let launch = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_plugin_worker_fixture"),
            profile_id,
            WorkerId(Uuid::from_u128(0x21079)),
            "worker-provider-v1",
            8 * 1024 * 1024 * 1024,
        );
        let boundary = ComponentExecutionBoundary::private_worker(
            PrivateWorkerPluginExecutor::new_with_provider_authorities(
                launch,
                broker,
                TEST_PROFILE_ID,
                receipt_issuer,
                Duration::from_secs(30),
                cost_authority.clone(),
            )?,
        );
        let host = ComponentHost::new(
            ComponentRuntime::no_wasi()?,
            trust,
            permission_policy(&manifest)?,
            boundary,
            conformance_component_limits(),
            sim_runtime::generated_native_node_registry_projection(None)?,
        )?;
        let router = ComponentHostRouter::new(host.clone());
        router
            .synchronize(vec![InstalledComponent::checked(
                Arc::from("provider-extension"),
                Arc::from("1.0.0"),
                serde_json::to_vec(&manifest)?.into(),
                component.into(),
            )?])
            .await?;
        let bundle = router.active_execution_registry_bundle()?;
        let registry = bundle.registry();
        let price_badge = registry
            .descriptor(NODE)
            .and_then(|descriptor| descriptor.source_schema.as_ref())
            .and_then(|schema| schema.node.price_badge.clone())
            .ok_or("paid provider price badge is absent")?;
        price_badge
            .validate()
            .map_err(|error| format!("generated provider price badge is invalid: {error}"))?;
        sim_runtime::ProviderInvocationIdentity::new(
            TEST_PROFILE_ID,
            TEST_PROFILE_ID,
            "provider-cost-approval",
            COMPILED_PLAN_SHA256,
            "provider-cost-approval",
            NODE,
            0,
            "0".repeat(64),
            manifest.identifier.clone(),
            component_digest.clone(),
            binding_sha256.clone(),
            "fixture",
            "https://fixture.invalid/v1/generate",
        )
        .map_err(|error| format!("provider approval identity is invalid: {error}"))?;
        let cost_approval = ProviderCostApproval::new(
            TEST_PROFILE_ID,
            TEST_PROFILE_ID,
            COMPILED_PLAN_SHA256,
            NODE,
            0,
            manifest.identifier.clone(),
            component_digest,
            binding_sha256,
            "fixture",
            "https://fixture.invalid/v1/generate",
            &price_badge,
            ProviderPriceBound::new("USD", 1_000_000)?,
            clock.utc_now() + Duration::from_secs(30),
            ProviderCostNonce::new([0x79; 32])?,
        )
        .map_err(|error| format!("provider cost approval construction failed: {error}"))?;
        cost_authority
            .approve(cost_approval)
            .map_err(|error| format!("provider cost approval admission failed: {error}"))?;

        let prompt_id = PromptId(Uuid::from_u128(0x2107_9001));
        let attempt_id = AttemptId(Uuid::from_u128(0x2107_9002));
        let generation = NativeHandleStoreGeneration::new()?;
        let store: Arc<dyn NativeHandleStore> = generation.handle_store_for_attempt(attempt_id);
        let store: Arc<dyn NativeHandleStore> = if mode == 4 {
            Arc::new(CancelAfterPngPublishStore {
                inner: store,
                cancellation: cancellation.clone(),
            })
        } else {
            store
        };
        let context = provider_node_context(
            prompt_id,
            attempt_id,
            NodeId(NODE.to_owned()),
            cancellation.clone(),
            store.clone(),
            COMPILED_PLAN_SHA256,
        )?;
        let outcome = registry
            .node(NODE)
            .ok_or("provider binding is not executable")?
            .execute(
                context.clone(),
                bfl_probe_inputs(if mode == 3 {
                    "fixture-invalid-receipt"
                } else {
                    "fixture prompt"
                }),
            )
            .await;
        if !matches!(mode, 0 | 5) {
            let failure = outcome
                .expect_err("invalid response/contract, forged receipt or cancellation must fail");
            assert!(!failure.message.is_empty());
            if matches!(mode, 6 | 8) {
                assert_eq!(failure.code, "unmaterialized_plugin_payload");
                assert!(failure.message.contains(if mode == 6 {
                    "provider class"
                } else {
                    "provider output port"
                }));
            }
            assert_eq!(provider.calls.load(Ordering::Acquire), 1);
            assert_eq!(generation.len(), 0);
            assert_eq!(context.scratch.in_use_bytes(), 0);
            assert_eq!(
                context
                    .compute_session()?
                    .backend()
                    .memory_snapshot()
                    .current_bytes,
                0
            );
            return Ok(());
        }
        let outcome = outcome?;
        let NodeOutcome::Values {
            outputs, effects, ..
        } = outcome
        else {
            return Err("provider node returned a non-value outcome".into());
        };
        assert!(effects.is_empty());
        let [NativeValue::Handle { value: handle }] = outputs.as_slice() else {
            return Err("scripted PNG did not publish one IMAGE handle".into());
        };
        let resolved = store.resolve(handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
            return Err("PNG handle did not resolve to canonical tensor".into());
        };
        let image = payload
            .image()
            .ok_or("canonical payload is not an IMAGE tensor")?;
        let expected_shape: &[u64] = if mode == 5 {
            &[1, 1, 1, 3]
        } else {
            &[1, 1, 1, 4]
        };
        assert_eq!(image.tensor().descriptor().shape(), expected_shape);
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let values = sim_tensor::generated_native_diffusion::tensor_to_f32(
            compute.backend(),
            image.tensor(),
            &execution,
        )?;
        if mode == 5 {
            assert_eq!(&*values, &[128.0 / 255.0; 3]);
        } else {
            assert_eq!(&*values, &[128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 1.0]);
        }
        drop(values);
        drop(resolved);
        store.revoke(handle, &cancellation)?;
        assert_eq!(generation.len(), 0);
        assert_eq!(
            context
                .compute_session()?
                .backend()
                .memory_snapshot()
                .current_bytes,
            0
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(provider.calls.load(Ordering::Acquire), 1);

        let replay_generation = NativeHandleStoreGeneration::new()?;
        let replay_store =
            replay_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x2107_9003)));
        let replay = registry
            .node(NODE)
            .ok_or("provider binding disappeared")?
            .execute(
                provider_node_context(
                    prompt_id,
                    AttemptId(Uuid::from_u128(0x2107_9003)),
                    NodeId(NODE.to_owned()),
                    CancellationToken::default(),
                    replay_store,
                    COMPILED_PLAN_SHA256,
                )?,
                bfl_probe_inputs("fixture prompt"),
            )
            .await
            .expect_err("spent cost approval must deny replay before provider actuation");
        assert!(replay.message.contains("cost") || replay.message.contains("denied"));
        assert_eq!(provider.calls.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[gpui::test(seed = 43771)]
    #[ignore = "requires coordinator-built distinct signed legacy BFL PNG process probe"]
    async fn bfl_legacy_png_process_settles_original_receipt_before_native_publication(
        executor: BackgroundExecutor,
    ) {
        executor.allow_parking();
        let result: Result<(), Box<dyn Error>> = async {
            exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), 0).await
        }
        .await;
        assert!(
            result.is_ok(),
            "signed legacy PNG process probe failed: {result:?}"
        );
    }
    #[gpui::test(seed = 43772)]
    #[ignore = "requires coordinator-built distinct signed legacy BFL PNG process probe"]
    async fn bfl_legacy_png_process_invalid_response_and_receipt_roll_back(
        executor: BackgroundExecutor,
    ) {
        executor.allow_parking();
        let result: Result<(), Box<dyn Error>> = async {
            for mode in [1, 3] {
                exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), mode)
                    .await?;
            }
            exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), 0).await
        }
        .await;
        assert!(
            result.is_ok(),
            "signed legacy PNG process probe failed: {result:?}"
        );
    }
    #[gpui::test(seed = 43773)]
    #[ignore = "requires coordinator-built distinct signed legacy BFL PNG process probe"]
    async fn bfl_legacy_png_process_late_cancellation_rolls_back_then_recovers(
        executor: BackgroundExecutor,
    ) {
        executor.allow_parking();
        let result: Result<(), Box<dyn Error>> = async {
            for mode in [2, 4] {
                exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), mode)
                    .await?;
            }
            exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), 0).await
        }
        .await;
        assert!(
            result.is_ok(),
            "signed legacy PNG process probe failed: {result:?}"
        );
    }
    #[gpui::test(seed = 43774)]
    #[ignore = "requires coordinator-built distinct signed legacy BFL PNG process probe"]
    async fn bfl_signed_png_probe_manifest_identity_mismatch_is_rejected(
        executor: BackgroundExecutor,
    ) {
        executor.allow_parking();
        let result: Result<(), Box<dyn Error>> = async {
            let component = std::fs::read(std::env::var("SIM_BFL_LEGACY_PNG_COMPONENT")?)?;
            for corrupt_namespace in [true, false] {
                let mut manifest = bfl_probe_manifest(format!("{:x}", Sha256::digest(&component)))?;
                let bindings = manifest
                    .provider_binding
                    .as_mut()
                    .ok_or("BFL bindings are absent")?;
                if corrupt_namespace {
                    manifest.identifier = "zed.sim.provider.openrouter".to_owned();
                    bindings.implementation_namespace = manifest.identifier.clone();
                } else {
                    bindings.bindings.retain(|claim| claim.node_id == NODE);
                    manifest.nodes.retain(|node| node.id == NODE);
                }
                bindings.bindings_sha256 = bindings.canonical_bindings_sha256()?;
                let trust = trust_policy()?;
                sign(&trust, &mut manifest)?;
                let host = ComponentHost::new(
                    ComponentRuntime::no_wasi()?,
                    trust,
                    permission_policy(&manifest)?,
                    ComponentExecutionBoundary::conformance_in_process(component_resources()?),
                    conformance_component_limits(),
                    sim_runtime::generated_native_node_registry_projection(None)?,
                )?;
                let router = ComponentHostRouter::new(host);
                let failure = router
                    .synchronize(vec![InstalledComponent::checked(
                        Arc::from("bfl-namespace-negative"),
                        Arc::from("1.0.0"),
                        serde_json::to_vec(&manifest)?.into(),
                        component.clone().into(),
                    )?])
                    .await;
                assert!(
                    failure.is_err(),
                    "mismatched signed/compiled BFL projection must fail before execution"
                );
            }
            Ok(())
        }
        .await;
        assert!(
            result.is_ok(),
            "signed legacy PNG process probe failed: {result:?}"
        );
    }
    #[gpui::test(seed = 43775)]
    #[ignore = "requires coordinator-built distinct signed legacy BFL process probe"]
    async fn bfl_legacy_structured_image_response_remains_compatible(executor: BackgroundExecutor) {
        executor.allow_parking();
        let result: Result<(), Box<dyn Error>> = async {
            exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), 5).await?;
            exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), 0).await
        }
        .await;
        assert!(result.is_ok(), "{result:?}");
    }

    #[gpui::test(seed = 43776)]
    #[ignore = "requires coordinator-built distinct signed legacy BFL process probe"]
    async fn bfl_legacy_structured_contract_failure_and_cancel_do_not_fallback(
        executor: BackgroundExecutor,
    ) {
        executor.allow_parking();
        let result: Result<(), Box<dyn Error>> = async {
            for mode in [6, 7, 8, 9] {
                exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), mode)
                    .await?;
            }
            exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), 5).await?;
            exercise_bfl_png_process(ProfileId(Uuid::parse_str(TEST_PROFILE_ID)?), 0).await
        }
        .await;
        assert!(result.is_ok(), "{result:?}");
    }
}
