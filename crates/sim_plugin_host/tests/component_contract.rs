use extension_host::{ComponentLifecycleAdapter, ComponentRuntime, InstalledComponent};
use sha2::{Digest, Sha256};
use sim_nodes::{
    NativeHandleKind, NativePortCardinality, NativePrimitiveType, NativeValueType,
    native_plugin_source_type_projection,
};
use sim_plugin_host::{
    AssetPluginCapabilityServices, CancellationToken, CapabilityLimits, CapabilityState,
    ComponentExecutionBoundary, ComponentHost, ComponentHostError, ComponentHostRouter,
    ComponentLimits, InvocationInputs, LegacyInputSourceProjection, LegacyMappingResolver,
    LegacyNodeReference, LegacyResolution, MappingCandidate, MappingSource, MappingTarget,
    PluginCapabilityServices, PluginError, PluginHost, WorkerPluginInvocation,
};
use sim_plugin_sdk::{
    ApiRequirement, ApiVersion, ArtifactValue, CachePolicy, CancelReason, CapabilityCall,
    CapabilityKind, CapabilityQuota, CapabilityRequest, CapabilityResponse, DType,
    DeterminismPolicy, DeviceId, ED25519_SIGNATURE_BYTES, EffectPolicy, InvocationError, Layout,
    MAX_PROVIDER_COST_REQUEST_BYTES, MAX_PROVIDER_COST_RESPONSE_BYTES, ManifestProvenance,
    ManifestSignature, ModelValue, PLUGIN_SIGNATURE_ALGORITHM, PROVIDER_BINDING_API_FEATURE,
    PROVIDER_BINDING_SCHEMA_VERSION, PROVIDER_COMPONENT_WORLD_V2,
    PROVIDER_MANIFEST_SCHEMA_VERSION_V2, PROVIDER_STREAMING_API_FEATURE_V2, PluginContractError,
    PluginInvocation, PluginManifest, PluginNode, PluginPort, PluginSigningKey, PluginValue,
    PortCardinality, PortDirection, PortPresence, PortSerialization, ProviderBindingClaim,
    ProviderBindingSet, ProviderHttpMethodV2, ProviderPluginManifestV2,
    ProviderStreamingContractV2, RouteDeclaration, RustNodeInstance, RustSimPlugin, ScalarValue,
    StreamId, TensorDescriptor, TensorValue, TypeRegistry, UiContribution, ValueFamily,
};
use sim_runtime::{
    AssetError, AssetIdentity, AssetNamespace, Capability, CapabilitySet, PermissionGrant,
    PermissionPolicy, PluginAuthorization, PluginTrustPolicy, PluginVerificationKey,
    ProviderTransportRequest, SecretId, authorize_native_input_reader,
    authorize_native_plugin_asset_broker, open_native_profile_asset_service,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU8, Ordering},
    },
};
use uuid::Uuid;

const KEY_ID: &str = "test.publisher";
const KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";

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

fn conformance_component_limits() -> ComponentLimits {
    ComponentLimits {
        epoch_deadline_ticks: 50,
        ..ComponentLimits::default()
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
    let capability_fixture = include_str!("fixtures/capabilities");
    let capabilities = capability_fixture
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

fn signed_host_and_manifest(
    digest: String,
) -> Result<(PluginHost, PluginManifest, PluginAuthorization), Box<dyn Error>> {
    let mut manifest = manifest(digest)?;
    let authorization = sign_and_authorize(&mut manifest)?;
    let host = PluginHost::with_configuration(
        conformance_component_limits(),
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )?;
    Ok((host, manifest, authorization))
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
    let mut provider_binding = ProviderBindingSet {
        schema_version: PROVIDER_BINDING_SCHEMA_VERSION,
        implementation_namespace: "zed.sim.provider.openrouter".to_owned(),
        bindings_sha256: "0".repeat(64),
        bindings: vec![ProviderBindingClaim {
            feature_id: "COMFY-NODE-0466".to_owned(),
            node_id: "OpenRouterLLMNode".to_owned(),
            contract_sha256: "2c6dbff1b7f89da063de276e6737f58e1dc98a85ad736f620bea96cab9e0a3d5"
                .to_owned(),
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

fn provider_streaming_manifest(
    component_digest: String,
) -> Result<ProviderPluginManifestV2, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let mut binding = ProviderBindingSet {
        schema_version: 1,
        implementation_namespace: "zed.sim.provider.openrouter".to_owned(),
        bindings_sha256: "0".repeat(64),
        bindings: vec![ProviderBindingClaim {
            feature_id: "COMFY-NODE-0466".to_owned(),
            node_id: "OpenRouterLLMNode".to_owned(),
            contract_sha256: "2c6dbff1b7f89da063de276e6737f58e1dc98a85ad736f620bea96cab9e0a3d5"
                .to_owned(),
            transport_schema: "zed:sim-provider-transport@1".parse()?,
            materializer_schema: "zed:sim-provider-materializer@1".parse()?,
        }],
    };
    binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
    assert_eq!(
        binding.bindings_sha256,
        "8b941f79f9560c7ec53819291d84e42d8c68ae2a7a70722143e55e6ea411f741"
    );
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
    let output = port(
        &registry,
        "output_0",
        PortDirection::Output,
        "String",
        PortCardinality::Singular,
        PortPresence::Required,
    )?;
    let mut manifest = PluginManifest {
        schema_version: 1,
        identifier: "zed.sim.provider.openrouter".to_owned(),
        plugin_version: ApiVersion::new(1, 0, 0),
        api: ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 0,
            required_features: vec![
                PROVIDER_BINDING_API_FEATURE.to_owned(),
                PROVIDER_STREAMING_API_FEATURE_V2.to_owned(),
            ],
        },
        digest_sha256: component_digest,
        signature: ManifestSignature {
            algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
            key_id: KEY_ID.to_owned(),
            value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
        },
        provenance: ManifestProvenance {
            source: "fixture://test.provider-streaming-plugin".to_owned(),
            publisher: "Zed provider streaming fixture".to_owned(),
            registry: Some("fixture://signed-registry".to_owned()),
        },
        provider_binding: Some(binding),
        nodes: vec![PluginNode {
            id: "OpenRouterLLMNode".to_owned(),
            version: ApiVersion::new(1, 0, 0),
            display_name: "OpenRouter LLM".to_owned(),
            category: "partner/text/OpenRouter".to_owned(),
            ports: input_ports.into_iter().chain([output]).collect(),
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        }],
        capabilities: vec![
            CapabilityRequest {
                kind: CapabilityKind::NetworkProvider,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 16_384,
                    maximum_response_bytes: 65_536,
                    maximum_total_bytes: 81_920,
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
            CapabilityRequest {
                kind: CapabilityKind::Secret,
                scope: "fixture-secret".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 14,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 15,
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
            CapabilityRequest {
                kind: CapabilityKind::ProviderUpload,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 4_096,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 4_096,
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
            CapabilityRequest {
                kind: CapabilityKind::ProviderCost,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: MAX_PROVIDER_COST_REQUEST_BYTES,
                    maximum_response_bytes: MAX_PROVIDER_COST_RESPONSE_BYTES,
                    maximum_total_bytes: MAX_PROVIDER_COST_REQUEST_BYTES
                        + MAX_PROVIDER_COST_RESPONSE_BYTES,
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
        ],
        ui: Vec::new(),
        routes: Vec::new(),
        legacy_mappings: Vec::new(),
    };
    manifest.signature.value = sign_manifest(&manifest)?;
    let mut provider = ProviderPluginManifestV2 {
        schema_version: PROVIDER_MANIFEST_SCHEMA_VERSION_V2,
        component_world: PROVIDER_COMPONENT_WORLD_V2.to_owned(),
        manifest,
        streaming: ProviderStreamingContractV2 {
            methods: vec![
                ProviderHttpMethodV2::Delete,
                ProviderHttpMethodV2::Get,
                ProviderHttpMethodV2::Head,
                ProviderHttpMethodV2::Options,
                ProviderHttpMethodV2::Patch,
                ProviderHttpMethodV2::Post,
                ProviderHttpMethodV2::Put,
            ],
            maximum_headers: 8,
            maximum_header_bytes: 4096,
            maximum_request_body_bytes: 16384,
            maximum_response_body_bytes: 65536,
            maximum_chunk_bytes: 4096,
            maximum_ndjson_line_bytes: 4096,
            maximum_wait_milliseconds: 1000,
            maximum_uploads: 1,
            maximum_upload_body_bytes: 4096,
            maximum_cost_requests: 1,
            maximum_progress_total: 100,
            uploads: true,
            cost_requests: true,
        },
        signature: ManifestSignature {
            algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
            key_id: KEY_ID.to_owned(),
            value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
        },
    };
    provider.signature.value = signing_key()?.sign_provider_manifest_v2(&provider)?;
    Ok(provider)
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

fn sign_manifest(manifest: &PluginManifest) -> Result<String, PluginContractError> {
    signing_key()?.sign_manifest(manifest)
}

fn sign_and_authorize(
    manifest: &mut PluginManifest,
) -> Result<PluginAuthorization, Box<dyn Error>> {
    sign_and_authorize_for_profile(manifest, "test-profile")
}

fn sign_and_authorize_for_profile(
    manifest: &mut PluginManifest,
    profile_id: &str,
) -> Result<PluginAuthorization, Box<dyn Error>> {
    let trust = trust_policy()?;
    manifest.signature.value = sign_manifest(manifest)?;
    Ok(trust.authorize_manifest(
        manifest,
        &permission_policy_for_profile(profile_id, manifest)?,
    )?)
}

fn permission_policy(manifest: &PluginManifest) -> Result<PermissionPolicy, Box<dyn Error>> {
    permission_policy_for_profile("test-profile", manifest)
}

fn permission_policy_for_profile(
    profile_id: &str,
    manifest: &PluginManifest,
) -> Result<PermissionPolicy, Box<dyn Error>> {
    let requested = manifest
        .capabilities
        .iter()
        .map(Capability::from_plugin_request)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PermissionPolicy::new(
        profile_id,
        [PermissionGrant::new(
            profile_id,
            manifest.identifier.clone(),
            CapabilitySet::new(requested),
            "signed-test-fixture",
        )?],
    )?)
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
    let registry = TypeRegistry::built_in()?;
    let stream = identifier.bytes().fold(0_u64, |value, byte| {
        value.wrapping_mul(31).wrapping_add(u64::from(byte))
    });
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

fn provider_invocation_inputs() -> Result<InvocationInputs, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let mut inputs = InvocationInputs::default();
    for (name, source_type, value) in [
        (
            "prompt",
            "String",
            ScalarValue::String("fixture prompt".to_owned()),
        ),
        (
            "model",
            "COMFY_DYNAMICCOMBO_V3",
            ScalarValue::String("openai/gpt-5".to_owned()),
        ),
        ("seed", "Int", ScalarValue::Integer(1)),
    ] {
        inputs.set_present(
            name,
            vec![PluginValue::scalar(
                registry.resolve(source_type)?.clone(),
                value,
                &registry,
            )?],
        );
    }
    Ok(inputs)
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

fn integer_value(value: i64) -> Result<PluginValue, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    Ok(PluginValue::scalar(
        registry.resolve("Int")?.clone(),
        ScalarValue::Integer(value),
        &registry,
    )?)
}

fn invalid_tensor_value() -> Result<TensorValue, Box<dyn Error>> {
    Ok(TensorValue::new(
        TensorDescriptor::contiguous(
            vec![1, 1, 1, 1],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?,
        4,
        "A".repeat(64),
    )?)
}

fn invocation_inputs(optional_lists_present: bool) -> Result<InvocationInputs, Box<dyn Error>> {
    let mut inputs = InvocationInputs::default();
    inputs.set_present("scalar-single-in", vec![scalar_value("scalar")?]);
    inputs.set_present("tensor-single-in", vec![tensor_value("1")?]);
    inputs.set_present("artifact-single-in", vec![artifact_value("artifact.svg")?]);
    inputs.set_present("model-single-in", vec![model_value("model")?]);
    if optional_lists_present {
        inputs.set_present("scalar-list-in", Vec::new());
        inputs.set_present(
            "tensor-list-in",
            vec![tensor_value("2")?, tensor_value("3")?],
        );
        inputs.set_present("artifact-list-in", Vec::new());
        inputs.set_present("model-list-in", vec![model_value("a")?, model_value("b")?]);
    } else {
        for family in ["scalar", "tensor", "artifact", "model"] {
            inputs.set_absent(format!("{family}-list-in"));
        }
    }
    Ok(inputs)
}

struct EchoPlugin {
    manifest: PluginManifest,
}

struct EchoNode;

struct CancelPlugin {
    manifest: PluginManifest,
    reason: Arc<AtomicU8>,
}

struct CancelNode {
    reason: Arc<AtomicU8>,
}

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

impl RustNodeInstance for CancelNode {
    fn invoke(&mut self, _invocation: &mut dyn PluginInvocation) -> Result<(), InvocationError> {
        Err(InvocationError::Cancelled)
    }

    fn cancel(&mut self, reason: CancelReason) {
        let code = match reason {
            CancelReason::User => 1,
            CancelReason::Timeout => 2,
            CancelReason::HostShutdown => 3,
            CancelReason::CapabilityRevoked => 4,
        };
        self.reason.store(code, Ordering::Release);
    }
}

impl RustSimPlugin for CancelPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn create_node(&self, node_id: &str) -> Result<Box<dyn RustNodeInstance>, PluginContractError> {
        if node_id != "echo" {
            return Err(PluginContractError::UnknownNode(node_id.to_owned()));
        }
        Ok(Box::new(CancelNode {
            reason: self.reason.clone(),
        }))
    }
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
    log_blocker: Option<Arc<LogBlocker>>,
}

#[derive(Default)]
struct LogBlockState {
    entered: bool,
    released: bool,
}

#[derive(Default)]
struct LogBlocker {
    state: Mutex<LogBlockState>,
    changed: Condvar,
}

impl LogBlocker {
    fn block(&self) -> Result<(), InvocationError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| InvocationError::HostFailure("log blocker poisoned".to_owned()))?;
        state.entered = true;
        self.changed.notify_all();
        while !state.released {
            state = self
                .changed
                .wait(state)
                .map_err(|_| InvocationError::HostFailure("log blocker poisoned".to_owned()))?;
        }
        Ok(())
    }

    fn wait_until_entered(&self) -> Result<(), Box<dyn Error>> {
        let mut state = self.state.lock().map_err(|_| "log blocker poisoned")?;
        while !state.entered {
            state = self
                .changed
                .wait(state)
                .map_err(|_| "log blocker poisoned")?;
        }
        Ok(())
    }

    fn release(&self) -> Result<(), Box<dyn Error>> {
        let mut state = self.state.lock().map_err(|_| "log blocker poisoned")?;
        state.released = true;
        self.changed.notify_all();
        Ok(())
    }
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
        if let Some(blocker) = &self.log_blocker {
            blocker.block()?;
        }
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

fn resources() -> Result<Arc<dyn PluginCapabilityServices>, Box<dyn Error>> {
    resources_with_log_blocker(None)
}

fn resources_with_log_blocker(
    log_blocker: Option<Arc<LogBlocker>>,
) -> Result<Arc<dyn PluginCapabilityServices>, Box<dyn Error>> {
    let mut resources = TestPluginServices::default();
    resources.log_blocker = log_blocker;
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
    resources.provider_responses.insert(
        (
            "fixture".to_owned(),
            "https://fixture.invalid/v1/generate".to_owned(),
        ),
        b"host-owned-provider-receipt".to_vec(),
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

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(name: &str) -> Result<Self, Box<dyn Error>> {
        let path = std::env::temp_dir().join(format!("zed-{name}-{}", Uuid::new_v4()));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!(
                "failed to remove test directory {}: {error}",
                self.path.display()
            );
        }
    }
}

fn component_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
    let fixture = include_str!("fixtures/list_ports");
    let (_, base64) = fixture
        .split_once("[component-base64]\n")
        .ok_or("component fixture marker is missing")?;
    decode_base64(base64.trim())
}

fn provider_component_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
    decode_base64(include_str!("fixtures/provider_component").trim())
}

fn provider_streaming_component_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
    decode_base64(include_str!("fixtures/provider_streaming_component").trim())
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

#[test]
fn rust_and_wit_fixtures_project_the_same_ports() -> Result<(), Box<dyn Error>> {
    let (_, manifest, _) = signed_host_and_manifest("0".repeat(64))?;
    let fixture = include_str!("fixtures/list_ports");
    let (port_fixture, _) = fixture
        .split_once("[component-base64]\n")
        .ok_or("component fixture marker is missing")?;
    let mut lines = port_fixture.lines();
    assert_eq!(lines.next(), Some("world=zed:sim-plugin@1.0.0"));
    let projected = manifest.nodes[0]
        .ports
        .iter()
        .map(|port| {
            format!(
                "{}|{}|{}|{}|{}|{}|{}|{}|{}",
                port.id,
                match port.direction {
                    PortDirection::Input => "input",
                    PortDirection::Output => "output",
                },
                port.type_id,
                match port.cardinality {
                    PortCardinality::Singular => "singular",
                    PortCardinality::List => "list",
                },
                match port.presence {
                    PortPresence::Required | PortPresence::Hidden => "required",
                    PortPresence::Optional => "optional",
                },
                if port.hidden { "hidden" } else { "visible" },
                if port.lazy { "lazy" } else { "eager" },
                match port.serialization {
                    PortSerialization::Inline => "inline",
                    PortSerialization::Handle => "handle",
                    PortSerialization::ArtifactReference => "artifact-reference",
                    PortSerialization::OpaquePreserved => "opaque-preserved",
                },
                port.accepted_legacy_names.join(",")
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(lines.collect::<Vec<_>>(), projected);
    let wit = include_str!("../../sim_plugin_sdk/wit/sim-plugin.wit");
    assert!(wit.contains("package zed:sim-plugin@1.0.0;"));
    assert!(wit.contains("variant invocation-error"));
    for operation in [
        "get-input-state",
        "read-scalar-input",
        "take-input",
        "read-handle",
        "create-output-value",
        "push-output",
        "finish-output",
        "filesystem-read",
        "provider-request",
        "output-commit",
        "route-respond",
    ] {
        assert!(wit.contains(operation), "missing WIT operation {operation}");
    }

    let component = component_fixture()?;
    let digest = format!("{:x}", Sha256::digest(&component));
    let (host, component_manifest, authorization) = signed_host_and_manifest(digest)?;
    let compiled = host.compile_component(&component, &component_manifest, &authorization)?;
    assert_eq!(compiled.identifier(), "test.echo-plugin");
    assert_eq!(compiled.digest_sha256(), component_manifest.digest_sha256);
    let expected = host.invoke_rust(
        &EchoPlugin {
            manifest: component_manifest.clone(),
        },
        &authorization,
        "echo",
        invocation_inputs(true)?,
        empty_services(),
        CancellationToken::default(),
    )?;
    let invocation = host.begin_invocation(
        &component_manifest,
        &authorization,
        "echo",
        invocation_inputs(true)?,
        resources()?,
        CancellationToken::default(),
    )?;
    let mut wasm = host.instantiate_component(&compiled, invocation)?;
    assert_eq!(
        wasm.manifest_bytes()?,
        component_manifest.component_projection()
    );
    let instance = wasm.create_node("echo")?;
    wasm.invoke(instance)?;
    wasm.drop_node(instance)?;
    let result = wasm.finish()?;
    assert_eq!(result.outputs, expected.outputs);
    assert_eq!(result.output_presence, expected.output_presence);
    assert_eq!(result.effects.outputs.len(), 1);
    let committed_output = result
        .effects
        .outputs
        .first()
        .ok_or("WASM fixture did not commit its transactional output")?;
    assert_eq!(committed_output.namespace, "outputs");
    assert_eq!(committed_output.name, "guest.bin");
    assert_eq!(committed_output.bytes, b"guest-output");
    assert_eq!(
        result.effects.logs,
        vec!["info: no-WASI echo fixture invoked"]
    );
    assert_eq!(
        result.effects.ui_state.get("panel.demo"),
        Some(&br#"{"invoked":true}"#.to_vec())
    );
    assert_eq!(result.effects.routes.len(), 1);
    assert_eq!(result.effects.routes[0].route, "route.demo");
    assert_eq!(result.effects.routes[0].status, 200);
    assert_eq!(result.effects.routes[0].body, b"guest-route");
    Ok(())
}

#[test]
fn provider_world_preflights_signed_bindings_and_returns_typed_outputs()
-> Result<(), Box<dyn Error>> {
    let component = provider_component_fixture()?;
    let digest = format!("{:x}", Sha256::digest(&component));
    let mut manifest = provider_manifest(digest)?;
    let authorization = sign_and_authorize(&mut manifest)?;
    let host = PluginHost::with_configuration(
        conformance_component_limits(),
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )?;
    let compiled = host.compile_component(&component, &manifest, &authorization)?;
    let invocation = host.begin_invocation(
        &manifest,
        &authorization,
        "OpenRouterLLMNode",
        provider_invocation_inputs()?,
        empty_services(),
        CancellationToken::default(),
    )?;
    let mut instance = host.instantiate_component(&compiled, invocation)?;
    assert_eq!(
        instance.provider_binding_set()?,
        manifest
            .provider_binding
            .clone()
            .ok_or("provider binding disappeared")?
    );
    let request = scalar_value("provider-output")?.abi_bytes()?;
    let result = instance.invoke_provider("OpenRouterLLMNode", &request)?;
    assert!(result.outputs.is_empty());
    assert!(result.output_presence.is_empty());
    assert_eq!(result.receipts(), &[b"provider-fixture-receipt".to_vec()]);
    assert!(result.effects.outputs.is_empty());
    assert!(result.effects.routes.is_empty());
    let encoded = serde_json::to_vec(&result)?;
    let decoded: sim_plugin_host::ProviderInvocationResult = serde_json::from_slice(&encoded)?;
    assert_eq!(decoded, result);

    let invocation = host.begin_invocation(
        &manifest,
        &authorization,
        "OpenRouterLLMNode",
        provider_invocation_inputs()?,
        resources()?,
        CancellationToken::default(),
    )?;
    let instance = host.instantiate_component(&compiled, invocation)?;
    let request = ProviderTransportRequest::checked("OpenRouterLLMNode", Vec::new())?.to_bytes()?;
    let result = instance.invoke_provider("OpenRouterLLMNode", &request)?;
    assert_eq!(
        result.receipts(),
        &[b"host-owned-provider-receipt".to_vec()]
    );
    assert!(result.outputs.is_empty());
    Ok(())
}

#[test]
fn provider_v2_missing_certified_grant_is_denied_before_guest_or_input_exposure()
-> Result<(), Box<dyn Error>> {
    let component = provider_streaming_component_fixture()?;
    let digest = format!("{:x}", Sha256::digest(&component));
    let manifest = provider_streaming_manifest(digest)?;
    assert_eq!(
        manifest
            .manifest
            .capabilities
            .iter()
            .map(|capability| (capability.kind, capability.scope.as_str()))
            .collect::<Vec<_>>(),
        [
            (
                CapabilityKind::NetworkProvider,
                "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream",
            ),
            (CapabilityKind::Secret, "fixture-secret"),
            (
                CapabilityKind::ProviderUpload,
                "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream",
            ),
            (
                CapabilityKind::ProviderCost,
                "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream",
            ),
        ]
    );
    let trust = trust_policy()?;
    let authorization =
        trust.authorize_provider_manifest_v2(&manifest, &permission_policy(&manifest.manifest)?)?;
    let host = PluginHost::with_configuration(
        conformance_component_limits(),
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )?;
    let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
    let mut inputs = InvocationInputs::default();
    inputs.set_present("prompt", vec![scalar_value("fixture")?]);
    let registry = TypeRegistry::built_in()?;
    inputs.set_present(
        "model",
        vec![PluginValue::scalar(
            registry.resolve("COMFY_DYNAMICCOMBO_V3")?.clone(),
            ScalarValue::String("fixture-model".to_owned()),
            &registry,
        )?],
    );
    inputs.set_present("seed", vec![integer_value(425)?]);
    let invocation = host.begin_invocation(
        &manifest.manifest,
        authorization.authorization(),
        "OpenRouterLLMNode",
        inputs,
        empty_services(),
        CancellationToken::default(),
    )?;
    assert!(matches!(
        host.instantiate_component(&compiled, invocation),
        Err(PluginError::ProviderInvocationUnavailable)
    ));

    let guest = include_str!("fixtures/provider_streaming_component_source/guest.rs");
    let invoke = guest
        .split("fn invoke(")
        .nth(1)
        .ok_or("provider-v2 fixture invoke function is missing")?;
    let start = invoke
        .find("provider_streaming_host::start_request")
        .ok_or("provider-v2 fixture start request is missing")?;
    let input = invoke
        .find("invocation_input_host::read_scalar_input")
        .ok_or("provider-v2 fixture input read is missing")?;
    assert!(
        start < input,
        "fixture attempted input access before binding"
    );
    Ok(())
}

#[test]
fn provider_fixture_contract_matches_the_generated_paid_descriptor() -> Result<(), Box<dyn Error>> {
    let registry = sim_runtime::generated_native_node_registry_projection(None)?;
    let contract = registry
        .provider_binding_contract_sha256(
            "OpenRouterLLMNode",
            "zed:sim-provider-transport@1",
            "zed:sim-provider-materializer@1",
        )?
        .ok_or("paid provider contract is absent")?;
    assert_eq!(
        contract,
        "2c6dbff1b7f89da063de276e6737f58e1dc98a85ad736f620bea96cab9e0a3d5"
    );
    let manifest = provider_manifest("0".repeat(64))?;
    assert_eq!(
        manifest
            .provider_binding
            .ok_or("provider binding set is absent")?
            .bindings_sha256,
        "8b941f79f9560c7ec53819291d84e42d8c68ae2a7a70722143e55e6ea411f741"
    );
    Ok(())
}

#[test]
fn legacy_world_rejects_signed_projection_mismatch_during_preflight() -> Result<(), Box<dyn Error>>
{
    let component = component_fixture()?;
    let digest = format!("{:x}", Sha256::digest(&component));
    let mut manifest = manifest(digest)?;
    manifest
        .nodes
        .first_mut()
        .ok_or("legacy fixture node disappeared")?
        .display_name = "Tampered Echo".to_owned();
    let authorization = sign_and_authorize(&mut manifest)?;
    let host = PluginHost::new()?;

    assert!(matches!(
        host.compile_component(&component, &manifest, &authorization),
        Err(PluginError::ManifestProjectionMismatch)
    ));
    Ok(())
}

#[test]
fn provider_world_rejects_binding_mismatch_before_instantiation() -> Result<(), Box<dyn Error>> {
    let component = provider_component_fixture()?;
    let digest = format!("{:x}", Sha256::digest(&component));
    let mut manifest = provider_manifest(digest)?;
    let binding_set = manifest
        .provider_binding
        .as_mut()
        .ok_or("provider binding disappeared")?;
    binding_set
        .bindings
        .first_mut()
        .ok_or("provider binding claim disappeared")?
        .contract_sha256 = "4".repeat(64);
    binding_set.bindings_sha256 = binding_set.canonical_bindings_sha256()?;
    let authorization = sign_and_authorize(&mut manifest)?;
    let host = PluginHost::new()?;
    assert!(matches!(
        host.compile_component(&component, &manifest, &authorization),
        Err(PluginError::ProviderBindingMismatch)
    ));
    Ok(())
}

#[test]
fn provider_world_rejects_malformed_cancelled_and_wrong_class_requests()
-> Result<(), Box<dyn Error>> {
    let component = provider_component_fixture()?;
    let digest = format!("{:x}", Sha256::digest(&component));
    let mut manifest = provider_manifest(digest)?;
    let authorization = sign_and_authorize(&mut manifest)?;
    let host = PluginHost::new()?;
    let compiled = host.compile_component(&component, &manifest, &authorization)?;

    for (class_type, request, cancellation, expected) in [
        (
            "OpenRouterLLMNode",
            b"invalid-provider-receipt".to_vec(),
            CancellationToken::default(),
            "invalid result receipt set",
        ),
        (
            "OpenRouterLLMNode",
            b"guest-authored-output".to_vec(),
            CancellationToken::default(),
            "attempted to author materialized output metadata",
        ),
        (
            "provider.unknown",
            scalar_value("value")?.abi_bytes()?,
            CancellationToken::default(),
            "not declared",
        ),
    ] {
        let invocation = host.begin_invocation(
            &manifest,
            &authorization,
            "OpenRouterLLMNode",
            provider_invocation_inputs()?,
            empty_services(),
            cancellation,
        )?;
        let instance = host.instantiate_component(&compiled, invocation)?;
        let error = instance
            .invoke_provider(class_type, &request)
            .expect_err("invalid provider invocation unexpectedly succeeded");
        assert!(error.to_string().contains(expected), "{error}");
    }

    let cancellation = CancellationToken::default();
    let invocation = host.begin_invocation(
        &manifest,
        &authorization,
        "OpenRouterLLMNode",
        provider_invocation_inputs()?,
        empty_services(),
        cancellation.clone(),
    )?;
    let instance = host.instantiate_component(&compiled, invocation)?;
    cancellation.cancel();
    assert!(matches!(
        instance.invoke_provider("OpenRouterLLMNode", &scalar_value("value")?.abi_bytes()?),
        Err(PluginError::Invocation(InvocationError::Cancelled))
    ));
    Ok(())
}

#[test]
fn port_transfer_handles_all_families_cardinalities_and_presence() -> Result<(), Box<dyn Error>> {
    let (host, manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let plugin = EchoPlugin { manifest };
    let present = host.invoke_rust(
        &plugin,
        &authorization,
        "echo",
        invocation_inputs(true)?,
        resources()?,
        CancellationToken::default(),
    )?;
    assert_eq!(present.outputs["scalar-list-out"].len(), 0);
    assert_eq!(present.outputs["tensor-list-out"].len(), 2);
    assert_eq!(present.outputs["artifact-list-out"].len(), 0);
    assert_eq!(present.outputs["model-list-out"].len(), 2);
    for family in ["scalar", "tensor", "artifact", "model"] {
        assert!(present.output_presence[&format!("{family}-list-out")]);
    }

    let absent = host.invoke_rust(
        &plugin,
        &authorization,
        "echo",
        invocation_inputs(false)?,
        resources()?,
        CancellationToken::default(),
    )?;
    for family in ["scalar", "tensor", "artifact", "model"] {
        assert!(absent.outputs[&format!("{family}-list-out")].is_empty());
        assert!(!absent.output_presence[&format!("{family}-list-out")]);
    }
    Ok(())
}

#[test]
fn rust_cancellation_invokes_the_typed_node_callback() -> Result<(), Box<dyn Error>> {
    let (host, manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let reason = Arc::new(AtomicU8::new(0));
    let plugin = CancelPlugin {
        manifest,
        reason: reason.clone(),
    };
    assert!(matches!(
        host.invoke_rust(
            &plugin,
            &authorization,
            "echo",
            invocation_inputs(false)?,
            resources()?,
            CancellationToken::default(),
        ),
        Err(PluginError::Invocation(InvocationError::Cancelled))
    ));
    assert_eq!(reason.load(Ordering::Acquire), 1);
    Ok(())
}

#[test]
fn invalid_ports_fail_before_invocation() -> Result<(), Box<dyn Error>> {
    let (host, manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let mut missing = invocation_inputs(false)?;
    missing.set_absent("scalar-single-in");
    assert!(matches!(
        host.begin_invocation(
            &manifest,
            &authorization,
            "echo",
            missing,
            resources()?,
            CancellationToken::default(),
        ),
        Err(PluginError::Invocation(InvocationError::MissingRequiredPort(port)))
            if port == "scalar-single-in"
    ));

    let mut wrong_cardinality = invocation_inputs(false)?;
    wrong_cardinality.set_present(
        "scalar-single-in",
        vec![scalar_value("one")?, scalar_value("two")?],
    );
    assert!(matches!(
        host.begin_invocation(
            &manifest,
            &authorization,
            "echo",
            wrong_cardinality,
            resources()?,
            CancellationToken::default(),
        ),
        Err(PluginError::Invocation(InvocationError::InvalidCardinality(port)))
            if port == "scalar-single-in"
    ));

    let mut wrong_family = invocation_inputs(false)?;
    wrong_family.set_present("scalar-single-in", vec![tensor_value("wrong")?]);
    assert!(matches!(
        host.begin_invocation(
            &manifest,
            &authorization,
            "echo",
            wrong_family,
            resources()?,
            CancellationToken::default(),
        ),
        Err(PluginError::Invocation(InvocationError::WrongValueFamily { port, .. }))
            if port == "scalar-single-in"
    ));
    Ok(())
}

#[test]
fn invocation_handles_are_unique_bounded_and_non_transferable() -> Result<(), Box<dyn Error>> {
    let (host, signed_manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let mut first = host.begin_invocation(
        &signed_manifest,
        &authorization,
        "echo",
        invocation_inputs(false)?,
        resources()?,
        CancellationToken::default(),
    )?;
    let mut second = host.begin_invocation(
        &signed_manifest,
        &authorization,
        "echo",
        invocation_inputs(false)?,
        resources()?,
        CancellationToken::default(),
    )?;
    let first_handle = first.take_input("tensor-single-in", 0)?;
    let second_handle = second.take_input("tensor-single-in", 0)?;
    assert_ne!(first_handle.invocation, second_handle.invocation);
    assert!(matches!(
        second.read_handle(first_handle),
        Err(InvocationError::InvalidHandle)
    ));

    let mut bounded_manifest = manifest("0".repeat(64))?;
    let bounded_authorization = sign_and_authorize(&mut bounded_manifest)?;
    let mut limits = conformance_component_limits();
    limits.maximum_value_handles = 1;
    let bounded_host = PluginHost::with_configuration(
        limits,
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )?;
    let mut bounded = bounded_host.begin_invocation(
        &bounded_manifest,
        &bounded_authorization,
        "echo",
        invocation_inputs(false)?,
        resources()?,
        CancellationToken::default(),
    )?;
    let retained = bounded.take_input("tensor-single-in", 0)?;
    assert!(matches!(
        bounded.take_input("model-single-in", 0),
        Err(InvocationError::InvocationQuotaExceeded { limit }) if limit == "invocation-handle"
    ));
    bounded.push_output("tensor-single-out", retained)?;
    bounded.take_input("model-single-in", 0)?;
    Ok(())
}

#[test]
fn every_port_operation_observes_cancellation_and_invocation_quotas() -> Result<(), Box<dyn Error>>
{
    let (host, signed_manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let cancellation = CancellationToken::default();
    let mut cancelled = host.begin_invocation(
        &signed_manifest,
        &authorization,
        "echo",
        invocation_inputs(false)?,
        resources()?,
        cancellation.clone(),
    )?;
    let retained = cancelled.take_input("tensor-single-in", 0)?;
    assert!(cancellation.cancel());
    assert!(matches!(
        cancelled.input_state("scalar-single-in"),
        Err(InvocationError::Cancelled)
    ));
    assert!(matches!(
        cancelled.read_scalar_input("scalar-single-in", 0),
        Err(InvocationError::Cancelled)
    ));
    assert!(matches!(
        cancelled.take_input("artifact-single-in", 0),
        Err(InvocationError::Cancelled)
    ));
    assert!(matches!(
        cancelled.read_handle(retained),
        Err(InvocationError::Cancelled)
    ));
    assert!(matches!(
        cancelled.create_output_value(scalar_value("cancelled")?),
        Err(InvocationError::Cancelled)
    ));
    assert!(matches!(
        cancelled.push_output("tensor-single-out", retained),
        Err(InvocationError::Cancelled)
    ));
    assert!(matches!(
        cancelled.finish_output("tensor-single-out", true),
        Err(InvocationError::Cancelled)
    ));
    assert!(matches!(
        cancelled.check_cancelled(),
        Err(InvocationError::Cancelled)
    ));

    let mut operation_manifest = manifest("0".repeat(64))?;
    let operation_authorization = sign_and_authorize(&mut operation_manifest)?;
    let mut operation_limits = conformance_component_limits();
    operation_limits.maximum_port_operations = 1;
    let operation_host = PluginHost::with_configuration(
        operation_limits,
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )?;
    let operation_bounded = operation_host.begin_invocation(
        &operation_manifest,
        &operation_authorization,
        "echo",
        invocation_inputs(false)?,
        resources()?,
        CancellationToken::default(),
    )?;
    operation_bounded.input_state("scalar-single-in")?;
    assert!(matches!(
        operation_bounded.read_scalar_input("scalar-single-in", 0),
        Err(InvocationError::InvocationQuotaExceeded { limit })
            if limit == "port-operation"
    ));

    let mut response_manifest = manifest("0".repeat(64))?;
    let response_authorization = sign_and_authorize(&mut response_manifest)?;
    let mut response_limits = conformance_component_limits();
    response_limits.maximum_port_response_bytes = 1;
    let response_host = PluginHost::with_configuration(
        response_limits,
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )?;
    let response_bounded = response_host.begin_invocation(
        &response_manifest,
        &response_authorization,
        "echo",
        invocation_inputs(false)?,
        resources()?,
        CancellationToken::default(),
    )?;
    assert!(matches!(
        response_bounded.input_state("scalar-single-in"),
        Err(InvocationError::InvocationQuotaExceeded { limit })
            if limit == "port-response-byte"
    ));
    Ok(())
}

#[test]
fn port_ownership_and_exact_type_failures_preserve_values() -> Result<(), Box<dyn Error>> {
    let (host, signed_manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let mut invocation = host.begin_invocation(
        &signed_manifest,
        &authorization,
        "echo",
        invocation_inputs(false)?,
        resources()?,
        CancellationToken::default(),
    )?;
    assert!(matches!(
        invocation.take_input("scalar-single-in", 0),
        Err(InvocationError::HostFailure(message))
            if message.contains("inline scalar ownership")
    ));
    assert_eq!(
        invocation.read_scalar_input("scalar-single-in", 0)?,
        scalar_value("scalar")?
    );

    let integer = integer_value(7)?;
    let handle = invocation.create_output_value(integer.clone())?;
    assert!(matches!(
        invocation.push_output("scalar-single-out", handle),
        Err(InvocationError::HostFailure(message))
            if message.contains("expects canonical type")
    ));
    assert_eq!(invocation.read_handle(handle)?, &integer);
    Ok(())
}

#[test]
fn aggregate_value_limits_and_value_metadata_fail_closed() -> Result<(), Box<dyn Error>> {
    let mut bounded_manifest = manifest("0".repeat(64))?;
    let bounded_authorization = sign_and_authorize(&mut bounded_manifest)?;
    let mut limits = conformance_component_limits();
    limits.maximum_invocation_value_bytes = 1;
    let bounded_host = PluginHost::with_configuration(
        limits,
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )?;
    assert!(matches!(
        bounded_host.begin_invocation(
            &bounded_manifest,
            &bounded_authorization,
            "echo",
            invocation_inputs(false)?,
            resources()?,
            CancellationToken::default(),
        ),
        Err(PluginError::Invocation(InvocationError::InvocationQuotaExceeded { limit }))
            if limit == "invocation-value-byte"
    ));

    let (host, valid_manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    assert!(invalid_tensor_value().is_err());
    assert!(
        TensorValue::new(
            TensorDescriptor::new_strided(
                vec![1],
                vec![1],
                2,
                DType::F32,
                Layout::Strided,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            11,
            "1".repeat(64),
        )
        .is_err()
    );
    let registry = TypeRegistry::built_in()?;
    assert!(
        PluginValue::scalar(
            registry.resolve("Float")?.clone(),
            ScalarValue::Float(f64::NAN),
            &registry,
        )
        .is_err()
    );

    let mut invalid_artifact = invocation_inputs(false)?;
    invalid_artifact.set_present("artifact-single-in", vec![artifact_value("../escape")?]);
    assert!(matches!(
        host.begin_invocation(
            &valid_manifest,
            &authorization,
            "echo",
            invalid_artifact,
            resources()?,
            CancellationToken::default(),
        ),
        Err(PluginError::Invocation(InvocationError::HostFailure(message)))
            if message.starts_with("invalid plugin artifact")
    ));

    let mut wrong_scalar_type = invocation_inputs(false)?;
    wrong_scalar_type.set_present("scalar-single-in", vec![integer_value(7)?]);
    assert!(matches!(
        host.begin_invocation(
            &valid_manifest,
            &authorization,
            "echo",
            wrong_scalar_type,
            resources()?,
            CancellationToken::default(),
        ),
        Err(PluginError::Invocation(InvocationError::HostFailure(message)))
            if message.contains("expects canonical type")
    ));
    Ok(())
}

#[test]
fn every_capability_is_scoped_bounded_and_transactional() -> Result<(), Box<dyn Error>> {
    let (_, manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let mut state = CapabilityState::new(
        &authorization,
        &manifest,
        resources()?,
        CancellationToken::default(),
    )?;
    assert_eq!(
        state.execute(CapabilityCall::FilesystemRead {
            root: "input-root".to_owned(),
            relative_path: "nested/file.bin".to_owned(),
        })?,
        CapabilityResponse::Bytes(b"file".to_vec())
    );
    assert_eq!(
        state.execute(CapabilityCall::SecretExists {
            identifier: "secret.demo".to_owned(),
        })?,
        CapabilityResponse::Boolean(true)
    );
    assert_eq!(
        state.execute(CapabilityCall::NetworkProvider {
            provider: "demo".to_owned(),
            endpoint: "https://demo.invalid/v1/generate".to_owned(),
            body: b"request".to_vec(),
            secret_id: Some("secret.demo".to_owned()),
        })?,
        CapabilityResponse::Bytes(b"provider".to_vec())
    );
    assert_eq!(
        state.execute(CapabilityCall::ClockNow {
            clock: "workflow".to_owned(),
        })?,
        CapabilityResponse::TimestampMilliseconds(1_234)
    );
    let random = state.execute(CapabilityCall::RandomBytes {
        stream: "sampler".to_owned(),
        length: 48,
    })?;
    assert!(matches!(random, CapabilityResponse::Bytes(bytes) if bytes.len() == 48));
    assert!(matches!(
        state.execute(CapabilityCall::ModelOpen {
            identifier: "zed-asset://model/fixture.json".to_owned(),
        })?,
        CapabilityResponse::Handle(1)
    ));
    let CapabilityResponse::Handle(transaction) = state.execute(CapabilityCall::OutputBegin {
        namespace: "outputs".to_owned(),
        name: "result.bin".to_owned(),
    })?
    else {
        return Err("output begin did not return a handle".into());
    };
    state.execute(CapabilityCall::OutputWrite {
        transaction,
        bytes: b"result".to_vec(),
    })?;
    assert!(matches!(
        state.execute(CapabilityCall::OutputCommit { transaction })?,
        CapabilityResponse::CommittedArtifact(_)
    ));
    state.execute(CapabilityCall::Log {
        level: "info".to_owned(),
        message: "using secret.demo\0".to_owned(),
    })?;
    state.execute(CapabilityCall::UiSet {
        contribution: "panel.demo".to_owned(),
        state: b"state".to_vec(),
    })?;
    state.execute(CapabilityCall::RouteRespond {
        route: "route.demo".to_owned(),
        status: 200,
        body: b"route".to_vec(),
    })?;
    let effects = state.finish()?;
    assert_eq!(effects.outputs.len(), 1);
    assert_eq!(effects.outputs[0].bytes, b"result");
    assert_eq!(effects.logs, ["info: using [REDACTED]"]);
    assert_eq!(effects.ui_state["panel.demo"], b"state");
    assert_eq!(effects.routes[0].body, b"route");
    Ok(())
}

#[test]
fn asset_capability_adapter_maps_wire_roots_and_preserves_canonical_security()
-> Result<(), Box<dyn Error>> {
    let directory = TestDirectory::new("sim-plugin-assets")?;
    let profile_id = "asset-profile";
    let assets = open_native_profile_asset_service(profile_id, directory.path(), &[])?;
    let input_path = directory.path().join("input/nested/file.bin");
    let input_parent = input_path.parent().ok_or("input parent is unavailable")?;
    fs::create_dir_all(input_parent)?;
    fs::write(&input_path, b"canonical-file")?;

    let index_authorization = authorize_native_plugin_asset_broker(profile_id)?;
    assets
        .lock()
        .map_err(|_| "asset service lock is unavailable")?
        .scan_namespaces(
            &[AssetNamespace::Input],
            &index_authorization,
            &CancellationToken::default(),
        )?;

    let input_authorization = authorize_native_input_reader(profile_id)?;
    let services: Arc<dyn PluginCapabilityServices> = Arc::new(AssetPluginCapabilityServices::new(
        assets.clone(),
        input_authorization.clone(),
    )?);
    let mut input_manifest = manifest("0".repeat(64))?;
    let input_plugin_authorization =
        sign_and_authorize_for_profile(&mut input_manifest, profile_id)?;
    let mut input_state = CapabilityState::new(
        &input_plugin_authorization,
        &input_manifest,
        services.clone(),
        CancellationToken::default(),
    )?;
    assert_eq!(
        input_state.execute(CapabilityCall::FilesystemRead {
            root: "input-root".to_owned(),
            relative_path: "nested/file.bin".to_owned(),
        })?,
        CapabilityResponse::Bytes(b"canonical-file".to_vec())
    );
    assert!(matches!(
        input_state.execute(CapabilityCall::FilesystemRead {
            root: "input-root".to_owned(),
            relative_path: "../escape".to_owned(),
        }),
        Err(InvocationError::InvalidCapabilityRequest(_))
    ));

    let output_identity = AssetIdentity::new(profile_id, AssetNamespace::Output, "result.bin")?;
    assert!(matches!(
        assets
            .lock()
            .map_err(|_| "asset service lock is unavailable")?
            .read_verified(
                &output_identity,
                &input_authorization,
                &CancellationToken::default(),
                4_096,
            ),
        Err(AssetError::PermissionDenied {
            namespace: AssetNamespace::Output,
            ..
        })
    ));

    let mut output_manifest = manifest("0".repeat(64))?;
    let filesystem_request = output_manifest
        .capabilities
        .iter_mut()
        .find(|request| request.kind == CapabilityKind::Filesystem)
        .ok_or("filesystem capability is unavailable")?;
    filesystem_request.scope = "output-root".to_owned();
    let output_plugin_authorization =
        sign_and_authorize_for_profile(&mut output_manifest, profile_id)?;
    let mut output_state = CapabilityState::new(
        &output_plugin_authorization,
        &output_manifest,
        services,
        CancellationToken::default(),
    )?;
    assert!(matches!(
        output_state.execute(CapabilityCall::FilesystemRead {
            root: "output-root".to_owned(),
            relative_path: "result.bin".to_owned(),
        }),
        Err(InvocationError::CapabilityDenied {
            kind: CapabilityKind::Filesystem,
            scope,
        }) if scope == "output"
    ));

    let other_profile_authorization = authorize_native_input_reader("other-profile")?;
    assert!(matches!(
        AssetPluginCapabilityServices::new(assets, other_profile_authorization),
        Err(InvocationError::HostFailure(message)) if message.contains("another profile")
    ));
    Ok(())
}

#[test]
fn provider_calls_are_delegated_to_the_canonical_service() -> Result<(), Box<dyn Error>> {
    let (_, manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let provider_call = || CapabilityCall::NetworkProvider {
        provider: "demo".to_owned(),
        endpoint: "https://demo.invalid/v1/generate".to_owned(),
        body: b"request".to_vec(),
        secret_id: Some("secret.demo".to_owned()),
    };
    let mut state = CapabilityState::new(
        &authorization,
        &manifest,
        resources()?,
        CancellationToken::default(),
    )?;
    assert_eq!(
        state.execute(provider_call())?,
        CapabilityResponse::Bytes(b"provider".to_vec())
    );
    Ok(())
}

#[test]
fn denied_quota_cancel_and_lost_transaction_have_no_commit() -> Result<(), Box<dyn Error>> {
    let mut bounded_manifest = manifest("0".repeat(64))?;
    bounded_manifest
        .capabilities
        .retain(|request| request.kind == CapabilityKind::SanitizedLog);
    let bounded_log = bounded_manifest
        .capabilities
        .first_mut()
        .ok_or("bounded log capability is missing")?;
    bounded_log.quota = CapabilityQuota {
        maximum_request_bytes: 4,
        maximum_response_bytes: 1,
        maximum_total_bytes: 4,
        ..quota()
    };
    let bounded_authorization = sign_and_authorize(&mut bounded_manifest)?;
    let mut bounded = CapabilityState::new(
        &bounded_authorization,
        &bounded_manifest,
        empty_services(),
        CancellationToken::default(),
    )?;
    assert!(matches!(
        bounded.execute(CapabilityCall::Log {
            level: "info".to_owned(),
            message: "too long".to_owned(),
        }),
        Err(InvocationError::QuotaExceeded { .. })
    ));

    let mut network_manifest = manifest("0".repeat(64))?;
    network_manifest
        .capabilities
        .retain(|request| request.kind == CapabilityKind::NetworkProvider);
    let network_authorization = sign_and_authorize(&mut network_manifest)?;
    let mut denied_secret = CapabilityState::new(
        &network_authorization,
        &network_manifest,
        resources()?,
        CancellationToken::default(),
    )?;
    assert!(matches!(
        denied_secret.execute(CapabilityCall::NetworkProvider {
            provider: "demo".to_owned(),
            endpoint: "https://demo.invalid/v1/generate".to_owned(),
            body: Vec::new(),
            secret_id: Some("secret.demo".to_owned()),
        }),
        Err(InvocationError::CapabilityDenied {
            kind: CapabilityKind::Secret,
            ..
        })
    ));

    let mut output_manifest = manifest("0".repeat(64))?;
    output_manifest
        .capabilities
        .retain(|request| request.kind == CapabilityKind::TransactionalOutput);
    let output_authorization = sign_and_authorize(&mut output_manifest)?;
    let cancellation = CancellationToken::default();
    let mut cancelled = CapabilityState::new(
        &output_authorization,
        &output_manifest,
        resources()?,
        cancellation.clone(),
    )?;
    cancelled.execute(CapabilityCall::OutputBegin {
        namespace: "outputs".to_owned(),
        name: "cancelled.bin".to_owned(),
    })?;
    cancellation.cancel();
    assert!(matches!(
        cancelled.finish(),
        Err(InvocationError::Cancelled)
    ));

    let mut lost = CapabilityState::new(
        &output_authorization,
        &output_manifest,
        resources()?,
        CancellationToken::default(),
    )?;
    lost.execute(CapabilityCall::OutputBegin {
        namespace: "outputs".to_owned(),
        name: "lost.bin".to_owned(),
    })?;
    assert!(matches!(
        lost.finish(),
        Err(InvocationError::HostFailure(message)) if message.contains("left an output transaction open")
    ));
    Ok(())
}

#[test]
fn versions_signatures_components_and_schema_fail_closed() -> Result<(), Box<dyn Error>> {
    let empty_component = [0, 97, 115, 109, 13, 0, 1, 0];
    let digest = format!("{:x}", Sha256::digest(empty_component));
    let (host, mut signed_manifest, authorization) = signed_host_and_manifest(digest)?;
    assert!(matches!(
        host.compile_component(&empty_component, &signed_manifest, &authorization),
        Err(PluginError::ComponentCompilation(_))
    ));

    let original_signature = signed_manifest.signature.value.clone();
    let replacement = if signed_manifest.signature.value.starts_with('f') {
        "e"
    } else {
        "f"
    };
    signed_manifest
        .signature
        .value
        .replace_range(..1, replacement);
    let requested = signed_manifest
        .capabilities
        .iter()
        .map(Capability::from_plugin_request)
        .collect::<Result<Vec<_>, _>>()?;
    let permissions = PermissionPolicy::new(
        "test-profile",
        [PermissionGrant::new(
            "test-profile",
            signed_manifest.identifier.clone(),
            CapabilitySet::new(requested),
            "signature-owner-test",
        )?],
    )?;
    assert!(matches!(
        trust_policy()?.authorize_manifest(&signed_manifest, &permissions),
        Err(sim_runtime::TrustError::InvalidPluginSignature)
    ));
    assert!(host.validate(&signed_manifest, &authorization).is_ok());

    signed_manifest.signature.value = original_signature;
    let digest_replacement = if signed_manifest.digest_sha256.starts_with('f') {
        "e"
    } else {
        "f"
    };
    signed_manifest
        .digest_sha256
        .replace_range(..1, digest_replacement);
    assert!(matches!(
        host.validate(&signed_manifest, &authorization),
        Err(PluginError::Trust(
            sim_runtime::TrustError::AuthorizationManifestMismatch
        ))
    ));

    let mut malformed = manifest("0".repeat(64))?;
    malformed.nodes.clear();
    assert!(matches!(
        sign_manifest(&malformed),
        Err(PluginContractError::InvalidNodeCount(0))
    ));
    let mut undeclared_ui = manifest("0".repeat(64))?;
    undeclared_ui.ui.clear();
    assert!(matches!(
        sign_manifest(&undeclared_ui),
        Err(PluginContractError::DuplicateOrInvalidCapability)
    ));
    let mut incompatible = manifest("0".repeat(64))?;
    incompatible.api.major = 2;
    let incompatible_authorization = sign_and_authorize(&mut incompatible)?;
    let incompatible_host = PluginHost::with_configuration(
        conformance_component_limits(),
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|feature| (*feature).to_owned()),
    )?;
    assert!(matches!(
        incompatible_host.validate(&incompatible, &incompatible_authorization),
        Err(PluginError::Contract(
            PluginContractError::UnsupportedApi { .. }
        ))
    ));

    let schema = include_str!("../../sim_plugin_sdk/schema/plugin-manifest-v1.schema.json");
    assert!(schema.contains("\"additionalProperties\": false"));
    assert!(schema.contains(PLUGIN_SIGNATURE_ALGORITHM));
    Ok(())
}

#[test]
fn legacy_resolution_preserves_workflow_until_explicit_acceptance() -> Result<(), Box<dyn Error>> {
    let (_, manifest, authorization) = signed_host_and_manifest("0".repeat(64))?;
    let reference = LegacyNodeReference::new(
        "LegacyEcho",
        b"fields".to_vec(),
        b"widgets".to_vec(),
        b"links".to_vec(),
        b"unknown-extension-data".to_vec(),
    )?;
    let candidate = MappingCandidate::new(
        MappingTarget {
            plugin_identifier: "test.echo-plugin".to_owned(),
            node_identifier: "echo".to_owned(),
            node_version: ApiVersion::new(1, 0, 0),
        },
        "signed fixture registry",
        &manifest,
        &authorization,
    )?;
    let mut resolver = LegacyMappingResolver::default();
    resolver.add_signed_registry("LegacyEcho", candidate)?;
    let mut resolution = resolver.resolve(&reference)?;
    let LegacyResolution::Projected {
        original,
        compatibility,
        provenance,
        rewrite_accepted,
        ..
    } = &resolution
    else {
        return Err("expected signed registry projection".into());
    };
    assert_eq!(original, &reference);
    assert_eq!(provenance.source, MappingSource::SignedRegistry);
    assert!(!rewrite_accepted);
    let named_translation = compatibility
        .port_by_name(PortDirection::Input, "legacy_scalar")
        .ok_or("legacy scalar alias was not projected")?;
    let positioned_translation = compatibility
        .port_by_target_position(PortDirection::Input, 0)
        .ok_or("legacy scalar position was not projected")?;
    assert_eq!(named_translation.target_port_id(), "scalar-single-in");
    assert_eq!(positioned_translation.target_port_id(), "scalar-single-in");
    assert_eq!(
        named_translation.type_id(),
        positioned_translation.type_id()
    );
    assert_eq!(named_translation.cardinality(), PortCardinality::Singular);
    assert_eq!(named_translation.presence(), PortPresence::Required);
    assert_eq!(named_translation.serialization(), PortSerialization::Inline);
    let input_translation = compatibility
        .inputs()
        .first()
        .ok_or("explicit legacy input translation is absent")?;
    assert_eq!(
        input_translation.target().target_port_id(),
        "scalar-single-in"
    );
    assert!(matches!(
        input_translation.source(),
        LegacyInputSourceProjection::LegacyInput {
            legacy_input_id,
            legacy_widget_position: Some(0),
        } if legacy_input_id == "legacy_scalar"
    ));
    assert_eq!(compatibility.inputs().len(), 8);
    let unmentioned_input = compatibility
        .inputs()
        .iter()
        .find(|translation| translation.target().target_port_id() == "tensor-single-in")
        .ok_or("unmentioned input did not retain its default legacy mapping")?;
    assert!(matches!(
        unmentioned_input.source(),
        LegacyInputSourceProjection::LegacyInput {
            legacy_input_id,
            legacy_widget_position: None,
        } if legacy_input_id == "tensor-single-in"
    ));
    let output_translation = compatibility
        .outputs()
        .first()
        .ok_or("explicit legacy output translation is absent")?;
    assert_eq!(output_translation.target().target_position(), 0);
    assert_eq!(output_translation.legacy_output_index(), 3);
    assert_eq!(compatibility.outputs().len(), 8);
    let unmentioned_output = compatibility
        .outputs()
        .iter()
        .find(|translation| translation.target().target_port_id() == "tensor-single-out")
        .ok_or("unmentioned output did not retain its default legacy index")?;
    assert_eq!(unmentioned_output.target().target_position(), 2);
    assert_eq!(unmentioned_output.legacy_output_index(), 2);
    assert!(compatibility.provider().is_none());
    let compatibility = compatibility.clone();
    let accepted = resolution
        .accept_rewrite()
        .ok_or("signed registry projection could not be accepted")?;
    assert_eq!(accepted.compatibility, compatibility);
    Ok(())
}

#[test]
fn legacy_provider_projection_uses_signed_manifest_and_canonical_permission_scopes()
-> Result<(), Box<dyn Error>> {
    let mut constant_manifest = manifest("0".repeat(64))?;
    constant_manifest.legacy_mappings[0]
        .legacy_widget_names
        .clear();
    constant_manifest.legacy_mappings[0].input_translations =
        vec![sim_plugin_sdk::LegacyInputTranslation::Constant {
            target_port_id: "scalar-single-in".to_owned(),
            value: ScalarValue::String("fixed-value".to_owned()),
        }];
    let constant_authorization = sign_and_authorize(&mut constant_manifest)?;
    let constant_candidate = MappingCandidate::new(
        MappingTarget {
            plugin_identifier: constant_manifest.identifier.clone(),
            node_identifier: "echo".to_owned(),
            node_version: ApiVersion::new(1, 0, 0),
        },
        "signed constant translation fixture",
        &constant_manifest,
        &constant_authorization,
    )?;
    let constant_compatibility = constant_candidate
        .compatibility_for("LegacyEcho")
        .ok_or("constant mapping compatibility is absent")?;
    let expected_constant_bytes = ScalarValue::String("fixed-value".to_owned()).abi_bytes()?;
    assert!(matches!(
        constant_compatibility
            .inputs()
            .first()
            .map(|translation| translation.source()),
        Some(LegacyInputSourceProjection::Constant { canonical_scalar_bytes })
            if canonical_scalar_bytes == &expected_constant_bytes
    ));

    let mut provider_manifest = manifest("0".repeat(64))?;
    provider_manifest
        .nodes
        .iter_mut()
        .find(|node| node.id == "echo")
        .ok_or("provider fixture node is absent")?
        .effects = EffectPolicy::Provider;
    let provider_authorization = sign_and_authorize(&mut provider_manifest)?;
    let provider_candidate = MappingCandidate::new(
        MappingTarget {
            plugin_identifier: provider_manifest.identifier.clone(),
            node_identifier: "echo".to_owned(),
            node_version: ApiVersion::new(1, 0, 0),
        },
        "signed provider fixture",
        &provider_manifest,
        &provider_authorization,
    )?;
    let provider = provider_candidate
        .compatibility_for("LegacyEcho")
        .ok_or("provider mapping compatibility is absent")?
        .provider()
        .ok_or("provider node omitted provider compatibility")?;
    assert!(provider.scopes().iter().any(|scope| {
        scope.provider() == "demo" && scope.endpoint() == "https://demo.invalid/v1/generate"
    }));

    let mut missing_capability_manifest = provider_manifest;
    missing_capability_manifest
        .capabilities
        .retain(|request| request.kind != CapabilityKind::NetworkProvider);
    let missing_capability_authorization = sign_and_authorize(&mut missing_capability_manifest)?;
    assert!(matches!(
        MappingCandidate::new(
            MappingTarget {
                plugin_identifier: missing_capability_manifest.identifier.clone(),
                node_identifier: "echo".to_owned(),
                node_version: ApiVersion::new(1, 0, 0),
            },
            "missing provider scope fixture",
            &missing_capability_manifest,
            &missing_capability_authorization,
        ),
        Err(sim_plugin_host::LegacyMappingError::ProviderNodeWithoutCapability(node))
            if node == "echo"
    ));
    Ok(())
}

#[test]
fn extension_owned_component_host_updates_registry_atomically_and_revokes_stale_handles()
-> Result<(), Box<dyn Error>> {
    let component = component_fixture()?;
    let mut manifest = manifest(format!("{:x}", Sha256::digest(&component)))?;
    let trust = trust_policy()?;
    manifest.signature.value = sign_manifest(&manifest)?;
    let host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust,
        permission_policy(&manifest)?,
        ComponentExecutionBoundary::conformance_in_process(resources()?),
        conformance_component_limits(),
        sim_runtime::native_image_registry_projection()?,
    )?;
    let router = ComponentHostRouter::new(host.clone());
    let manifest_bytes: Arc<[u8]> = serde_json::to_vec(&manifest)?.into();
    let component_bytes: Arc<[u8]> = component.into();
    let installed = InstalledComponent::checked(
        Arc::from("test-extension"),
        Arc::from("1.2.3"),
        manifest_bytes.clone(),
        component_bytes.clone(),
    )?;
    smol::block_on(router.synchronize(vec![installed]))?;
    let generation = host.verified_generation()?;
    assert_eq!(generation.profile_id(), "test-profile");
    assert_eq!(generation.generation(), 1);
    assert_eq!(generation.components().len(), 1);
    assert_eq!(generation.provider_registry_pin()?, None);
    let deployment = &generation.components()[0];
    assert_eq!(deployment.extension_id(), "test-extension");
    assert_eq!(deployment.extension_version(), "1.2.3");
    assert_eq!(deployment.plugin_identifier(), "test.echo-plugin");
    assert_eq!(deployment.plugin_version(), "1.2.3");
    assert_eq!(deployment.component_sha256(), manifest.digest_sha256);
    assert_eq!(deployment.manifest_bytes(), manifest_bytes.as_ref());
    assert_eq!(deployment.component_bytes(), component_bytes.as_ref());
    assert_eq!(deployment.authorization_generation().len(), 64);
    let worker_deployment = generation.worker_deployment_plan()?;
    assert_eq!(worker_deployment.begin().generation().get(), 1);
    assert_eq!(worker_deployment.begin().components().len(), 1);
    assert_eq!(
        worker_deployment.begin().components()[0]
            .authorization_generation()
            .as_str(),
        deployment.authorization_generation()
    );
    PluginAuthorization::from_sealed_bytes(
        deployment.authorization_bytes(),
        &manifest,
        worker_deployment.authorization_verifier(),
        worker_deployment
            .authorization_verifier()
            .policy_generation(),
        "test-profile",
    )?;
    let worker_invocation = generation.prepare_worker_invocation(
        "test-extension",
        "echo",
        invocation_inputs(true)?,
        1_000,
        1_024,
        conformance_component_limits(),
    )?;
    assert_eq!(worker_invocation.extension_id(), "test-extension");
    assert_eq!(worker_invocation.extension_version(), "1.2.3");
    assert_eq!(worker_invocation.plugin_identifier(), "test.echo-plugin");
    assert_eq!(worker_invocation.plugin_version(), "1.2.3");
    assert_eq!(
        worker_invocation.manifest_digest_sha256().as_str(),
        deployment.manifest_sha256()
    );
    assert_eq!(
        worker_invocation.component_digest_sha256().as_str(),
        deployment.component_sha256()
    );
    assert_eq!(
        WorkerPluginInvocation::from_bytes(&worker_invocation.to_bytes()?)?,
        worker_invocation
    );
    assert!(worker_invocation.provider_request().is_none());
    assert!(
        generation
            .prepare_worker_provider_invocation(
                "test-extension",
                "echo",
                invocation_inputs(true)?,
                b"provider-request".to_vec(),
                1_000,
                1_024,
                conformance_component_limits(),
            )
            .is_err()
    );
    let mut tampered_authorization = deployment.authorization_bytes().to_vec();
    let last = tampered_authorization
        .last_mut()
        .ok_or("sealed authorization was unexpectedly empty")?;
    *last ^= 1;
    assert!(
        PluginAuthorization::from_sealed_bytes(
            &tampered_authorization,
            &manifest,
            worker_deployment.authorization_verifier(),
            worker_deployment
                .authorization_verifier()
                .policy_generation(),
            "test-profile",
        )
        .is_err()
    );
    assert!(!worker_deployment.chunks().is_empty());
    let registry = host.registry_snapshot()?;
    assert_eq!(registry.descriptor_len(), 6);
    assert_eq!(
        registry
            .descriptor("echo")
            .map(|descriptor| descriptor.implementation_version.as_str()),
        Some("1.0.0")
    );
    let descriptor = registry
        .descriptor("echo")
        .ok_or("installed component descriptor is absent")?;
    let manifest_node = manifest.nodes.first().ok_or("manifest node is absent")?;
    for plugin_port in manifest_node
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Input)
    {
        let native_input = descriptor
            .inputs
            .iter()
            .find(|input| input.name == plugin_port.id)
            .ok_or("installed native input is absent")?;
        let canonical =
            native_plugin_source_type_projection(plugin_port.type_id.name())?.value_type()?;
        assert_eq!(
            native_input.accepted_types.members(),
            std::slice::from_ref(&canonical)
        );
        assert_eq!(
            native_input.allows_literal,
            !matches!(canonical, NativeValueType::Handle(_))
        );
    }
    let scalar = descriptor
        .inputs
        .iter()
        .find(|input| input.name == "scalar-single-in")
        .ok_or("scalar input descriptor is absent")?;
    assert_eq!(scalar.cardinality, NativePortCardinality::Scalar);
    assert_eq!(
        scalar.accepted_types.members(),
        &[NativeValueType::Primitive(NativePrimitiveType::String)]
    );
    let tensor = descriptor
        .inputs
        .iter()
        .find(|input| input.name == "tensor-list-in")
        .ok_or("tensor list input descriptor is absent")?;
    assert_eq!(tensor.cardinality, NativePortCardinality::List);
    assert!(tensor.lazy);
    assert!(matches!(
        tensor.accepted_types.members(),
        [NativeValueType::Handle(handle_type)]
            if handle_type.kind == NativeHandleKind::Image && handle_type.type_id == "IMAGE"
    ));
    let artifact = descriptor
        .inputs
        .iter()
        .find(|input| input.name == "artifact-single-in")
        .ok_or("artifact input descriptor is absent")?;
    assert!(matches!(
        artifact.accepted_types.members(),
        [NativeValueType::Handle(handle_type)]
            if handle_type.kind == NativeHandleKind::Artifact && handle_type.type_id == "SVG"
    ));
    let model = descriptor
        .inputs
        .iter()
        .find(|input| input.name == "model-single-in")
        .ok_or("model input descriptor is absent")?;
    assert!(matches!(
        model.accepted_types.members(),
        [NativeValueType::Handle(handle_type)]
            if handle_type.kind == NativeHandleKind::Model && handle_type.type_id == "MODEL"
    ));
    let presentation = registry
        .presentation("echo")
        .ok_or("installed component presentation is absent")?;
    assert_eq!(presentation.display_name, "Echo");
    assert_eq!(presentation.category, "test");
    assert_eq!(
        presentation.output_names,
        manifest
            .nodes
            .first()
            .ok_or("manifest node is absent")?
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .map(|port| port.name.clone())
            .collect::<Vec<_>>()
    );
    assert!(registry.node("echo").is_some());
    assert_eq!(
        registry.implementation_namespace("echo"),
        Some("test.echo-plugin")
    );
    let plugin = host.installed_plugin("test-extension")?;
    assert_eq!(plugin.binding().lifecycle_extension_id(), "test-extension");
    assert_eq!(plugin.binding().lifecycle_extension_version(), "1.2.3");
    assert_eq!(
        plugin.binding().signed_plugin_identifier(),
        "test.echo-plugin"
    );
    assert_eq!(plugin.binding().signed_plugin_version(), "1.2.3");
    assert_eq!(
        plugin.binding().signed_provenance_source(),
        "fixture://test.echo-plugin"
    );
    assert_eq!(plugin.manifest().ui.len(), 1);
    assert_eq!(plugin.manifest().ui[0].id, "panel.demo");
    assert_eq!(plugin.manifest().ui[0].surface, "node-panel");
    assert_eq!(
        plugin.manifest().ui[0].state_schema,
        "{\"type\":\"object\"}"
    );
    let result = host.invoke(
        &plugin,
        "echo",
        invocation_inputs(true)?,
        CancellationToken::default(),
    )?;
    assert_eq!(
        result.outputs["scalar-single-out"],
        vec![scalar_value("scalar")?]
    );

    let invalid_update = InstalledComponent::checked(
        Arc::from("test-extension"),
        Arc::from("1.2.3"),
        Arc::from(br#"{}"#.as_slice()),
        component_bytes.clone(),
    )?;
    assert!(smol::block_on(router.synchronize(vec![invalid_update])).is_err());
    assert!(host.installed_plugin("test-extension").is_ok());

    let invalid_identity = InstalledComponent::checked(
        Arc::from("../escape"),
        Arc::from("1.2.3"),
        manifest_bytes.clone(),
        component_bytes.clone(),
    );
    assert!(matches!(invalid_identity, Err(error) if error.to_string().contains("identifier")));
    let mismatched_version = InstalledComponent::checked(
        Arc::from("test-extension"),
        Arc::from("9.9.9"),
        manifest_bytes,
        component_bytes,
    )?;
    assert!(matches!(
        smol::block_on(router.synchronize(vec![mismatched_version])),
        Err(error) if error.contains("does not match signed plugin version")
    ));
    assert!(host.installed_plugin("test-extension").is_ok());

    smol::block_on(router.synchronize(Vec::new()))?;
    assert!(matches!(
        host.invoke(
            &plugin,
            "echo",
            invocation_inputs(true)?,
            CancellationToken::default(),
        ),
        Err(ComponentHostError::Revoked(extension)) if extension == "test-extension"
    ));
    assert_eq!(host.registry_snapshot()?.descriptor_len(), 5);
    let removed_generation = host.verified_generation()?;
    assert_eq!(removed_generation.generation(), 2);
    assert!(removed_generation.components().is_empty());
    assert!(
        removed_generation
            .worker_deployment_plan()?
            .chunks()
            .is_empty()
    );
    Ok(())
}

#[test]
fn provider_component_activation_publishes_one_exact_registry_bundle_and_rolls_back_rejection()
-> Result<(), Box<dyn Error>> {
    let component = provider_component_fixture()?;
    let mut manifest = provider_manifest(format!("{:x}", Sha256::digest(&component)))?;
    let profile_id = uuid::Uuid::new_v4().to_string();
    manifest.signature.value = sign_manifest(&manifest)?;
    let host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust_policy()?,
        permission_policy_for_profile(&profile_id, &manifest)?,
        ComponentExecutionBoundary::conformance_in_process(resources()?),
        conformance_component_limits(),
        sim_runtime::generated_native_node_registry_projection(None)?,
    )?;
    let router = ComponentHostRouter::new(host);
    let installed = InstalledComponent::checked(
        Arc::from("provider-extension"),
        Arc::from("1.0.0"),
        serde_json::to_vec(&manifest)?.into(),
        component.into(),
    )?;
    smol::block_on(router.synchronize(vec![installed.clone()]))?;

    let bundle = router.active_execution_registry_bundle()?;
    assert_eq!(
        bundle
            .registry()
            .binding_declared_disposition("OpenRouterLLMNode"),
        Some(sim_nodes::NativeNodeBindingDisposition::ProviderRequired)
    );
    assert_eq!(
        bundle.registry().binding_disposition("OpenRouterLLMNode"),
        Some(sim_nodes::NativeNodeBindingDisposition::Executable)
    );
    assert_eq!(
        bundle
            .registry()
            .provider_binding_is_activated("OpenRouterLLMNode"),
        Some(true)
    );
    assert_eq!(
        bundle
            .registry()
            .binding_declared_disposition("OpenRouterLLMNode")
            .ok_or("provider binding is absent")?,
        sim_nodes::NativeNodeBindingDisposition::ProviderRequired
    );
    assert_eq!(
        bundle
            .registry()
            .binding_implementation_namespace("OpenRouterLLMNode"),
        Some("zed.sim.provider.openrouter")
    );
    assert!(bundle.registry().node("OpenRouterLLMNode").is_some());
    let pin = bundle
        .provider_registry()
        .ok_or("provider registry pin is absent")?;
    assert_eq!(
        pin.generation(),
        bundle.worker_deployment().begin().generation().get()
    );
    assert_eq!(pin.binding_digests_sha256().len(), 1);
    let retained_identity = bundle.identity_sha256().to_owned();

    let mut invalid_manifest = manifest;
    invalid_manifest
        .provider_binding
        .as_mut()
        .ok_or("provider binding is absent")?
        .bindings[0]
        .contract_sha256 = "0".repeat(64);
    let provider_binding = invalid_manifest
        .provider_binding
        .as_mut()
        .ok_or("provider binding is absent")?;
    provider_binding.bindings_sha256 = provider_binding.canonical_bindings_sha256()?;
    invalid_manifest.signature.value = sign_manifest(&invalid_manifest)?;
    let invalid = InstalledComponent::checked(
        Arc::from("provider-extension"),
        Arc::from("1.0.0"),
        serde_json::to_vec(&invalid_manifest)?.into(),
        installed.component_bytes().to_vec().into(),
    )?;
    assert!(smol::block_on(router.synchronize(vec![invalid])).is_err());
    assert_eq!(
        router.active_execution_registry_bundle()?.identity_sha256(),
        retained_identity
    );

    smol::block_on(router.synchronize(Vec::new()))?;
    let removed = router.active_execution_registry_bundle()?;
    assert!(removed.provider_registry().is_none());
    assert!(removed.registry().node("OpenRouterLLMNode").is_none());
    assert_eq!(
        removed
            .registry()
            .binding_declared_disposition("OpenRouterLLMNode"),
        Some(sim_nodes::NativeNodeBindingDisposition::ProviderRequired)
    );
    Ok(())
}

#[test]
fn component_update_quiesces_active_invocation_and_revokes_the_previous_generation()
-> Result<(), Box<dyn Error>> {
    let component = component_fixture()?;
    let mut manifest = manifest(format!("{:x}", Sha256::digest(&component)))?;
    let trust = trust_policy()?;
    manifest.signature.value = sign_manifest(&manifest)?;
    let blocker = Arc::new(LogBlocker::default());
    let host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust,
        permission_policy(&manifest)?,
        ComponentExecutionBoundary::conformance_in_process(resources_with_log_blocker(Some(
            blocker.clone(),
        ))?),
        conformance_component_limits(),
        sim_runtime::native_image_registry_projection()?,
    )?;
    let router = ComponentHostRouter::new(host.clone());
    let installed = InstalledComponent::checked(
        Arc::from("test-extension"),
        Arc::from("1.2.3"),
        serde_json::to_vec(&manifest)?.into(),
        component.into(),
    )?;
    smol::block_on(router.synchronize(vec![installed.clone()]))?;
    let previous = host.installed_plugin("test-extension")?;

    let invocation_host = host.clone();
    let invocation_plugin = previous.clone();
    let invocation = std::thread::spawn(move || {
        invocation_host
            .invoke(
                &invocation_plugin,
                "echo",
                invocation_inputs(true).map_err(|error| error.to_string())?,
                CancellationToken::default(),
            )
            .map_err(|error| error.to_string())
    });
    blocker.wait_until_entered()?;

    let update = std::thread::spawn(move || smol::block_on(router.synchronize(vec![installed])));
    blocker.release()?;

    invocation
        .join()
        .map_err(|_| "component invocation thread panicked")??;
    update
        .join()
        .map_err(|_| "component update thread panicked")??;
    assert!(matches!(
        host.invoke(
            &previous,
            "echo",
            invocation_inputs(true)?,
            CancellationToken::default(),
        ),
        Err(ComponentHostError::Revoked(extension)) if extension == "test-extension"
    ));
    let current = host.installed_plugin("test-extension")?;
    host.invoke(
        &current,
        "echo",
        invocation_inputs(true)?,
        CancellationToken::default(),
    )?;
    Ok(())
}

#[test]
fn component_host_router_replaces_profile_services_and_revokes_the_old_generation()
-> Result<(), Box<dyn Error>> {
    let component = component_fixture()?;
    let mut manifest = manifest(format!("{:x}", Sha256::digest(&component)))?;
    manifest.signature.value = sign_manifest(&manifest)?;
    let installed = InstalledComponent::checked(
        Arc::from("test-extension"),
        Arc::from("1.2.3"),
        serde_json::to_vec(&manifest)?.into(),
        component.into(),
    )?;

    let old_host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust_policy()?,
        permission_policy_for_profile("old-profile", &manifest)?,
        ComponentExecutionBoundary::conformance_in_process(resources()?),
        conformance_component_limits(),
        sim_runtime::native_image_registry_projection()?,
    )?;
    let router = ComponentHostRouter::new(old_host.clone());
    smol::block_on(router.synchronize(vec![installed]))?;
    let old_plugin = old_host.installed_plugin("test-extension")?;
    assert_eq!(
        old_plugin.authorization().capabilities().profile_id(),
        "old-profile"
    );

    let rejected_host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust_policy()?,
        PermissionPolicy::new("rejected-profile", std::iter::empty())?,
        ComponentExecutionBoundary::conformance_in_process(resources()?),
        conformance_component_limits(),
        sim_runtime::native_image_registry_projection()?,
    )?;
    assert!(matches!(
        router.replace(rejected_host),
        Err(ComponentHostError::Verification { .. })
    ));
    old_host.invoke(
        &old_plugin,
        "echo",
        invocation_inputs(true)?,
        CancellationToken::default(),
    )?;

    let replacement_host = ComponentHost::new(
        ComponentRuntime::no_wasi()?,
        trust_policy()?,
        permission_policy_for_profile("replacement-profile", &manifest)?,
        ComponentExecutionBoundary::conformance_in_process(resources()?),
        conformance_component_limits(),
        sim_runtime::native_image_registry_projection()?,
    )?;
    router.replace(replacement_host)?;
    assert!(matches!(
        old_host.invoke(
            &old_plugin,
            "echo",
            invocation_inputs(true)?,
            CancellationToken::default(),
        ),
        Err(ComponentHostError::Revoked(extension)) if extension == "test-extension"
    ));

    let current_host = router.current()?;
    let replacement_plugin = current_host.installed_plugin("test-extension")?;
    assert_eq!(
        replacement_plugin
            .authorization()
            .capabilities()
            .profile_id(),
        "replacement-profile"
    );
    current_host.invoke(
        &replacement_plugin,
        "echo",
        invocation_inputs(true)?,
        CancellationToken::default(),
    )?;
    router.replace(current_host.clone())?;
    assert!(current_host.installed_plugin("test-extension").is_ok());

    smol::block_on(router.synchronize(Vec::new()))?;
    assert!(matches!(
        current_host.invoke(
            &replacement_plugin,
            "echo",
            invocation_inputs(true)?,
            CancellationToken::default(),
        ),
        Err(ComponentHostError::Revoked(extension)) if extension == "test-extension"
    ));
    Ok(())
}

#[test]
fn manifest_capability_quotas_cannot_exceed_host_limits() -> Result<(), Box<dyn Error>> {
    let mut signed_manifest = manifest("0".repeat(64))?;
    let request = signed_manifest
        .capabilities
        .first_mut()
        .ok_or("capability fixture is empty")?;
    request.quota.maximum_total_bytes = CapabilityLimits::default()
        .maximum_total_bytes
        .checked_add(1)
        .ok_or("test quota overflow")?;
    request.quota.maximum_response_bytes = request.quota.maximum_total_bytes;
    let authorization = sign_and_authorize(&mut signed_manifest)?;
    let host = PluginHost::new()?;
    assert!(matches!(
        host.validate(&signed_manifest, &authorization),
        Err(PluginError::Invocation(
            InvocationError::InvalidCapabilityRequest(message)
        )) if message.contains("host ceiling")
    ));
    Ok(())
}

#[test]
fn compiled_component_rejects_an_invocation_authorized_for_another_manifest()
-> Result<(), Box<dyn Error>> {
    let component = component_fixture()?;
    let digest = format!("{:x}", Sha256::digest(&component));
    let (host, compiled_manifest, authorization) = signed_host_and_manifest(digest.clone())?;
    let compiled = host.compile_component(&component, &compiled_manifest, &authorization)?;

    let mut other = manifest(digest)?;
    other.identifier = "test.other-plugin".to_owned();
    let trust = trust_policy()?;
    other.signature.value = sign_manifest(&other)?;
    let other_authorization = trust.authorize_manifest(&other, &permission_policy(&other)?)?;
    let invocation = host.begin_invocation(
        &other,
        &other_authorization,
        "echo",
        invocation_inputs(true)?,
        resources()?,
        CancellationToken::default(),
    )?;
    assert!(matches!(
        host.instantiate_component(&compiled, invocation),
        Err(PluginError::InvocationBindingMismatch)
    ));
    Ok(())
}

#[test]
fn no_wasi_linker_rejects_components_with_wasi_imports() -> Result<(), Box<dyn Error>> {
    let component = br#"(component (import "wasi:cli/run@0.2.0" (instance)))"#;
    let mut manifest = manifest(format!("{:x}", Sha256::digest(component)))?;
    let authorization = sign_and_authorize(&mut manifest)?;
    let host = PluginHost::new()?;
    assert!(matches!(
        host.compile_component(component, &manifest, &authorization),
        Err(PluginError::ComponentCompilation(_))
    ));
    Ok(())
}

#[test]
fn val_plugin_001() -> Result<(), Box<dyn Error>> {
    rust_and_wit_fixtures_project_the_same_ports()?;
    port_transfer_handles_all_families_cardinalities_and_presence()?;
    rust_cancellation_invokes_the_typed_node_callback()?;
    invalid_ports_fail_before_invocation()?;
    invocation_handles_are_unique_bounded_and_non_transferable()?;
    every_port_operation_observes_cancellation_and_invocation_quotas()?;
    port_ownership_and_exact_type_failures_preserve_values()?;
    aggregate_value_limits_and_value_metadata_fail_closed()?;
    every_capability_is_scoped_bounded_and_transactional()?;
    asset_capability_adapter_maps_wire_roots_and_preserves_canonical_security()?;
    provider_calls_are_delegated_to_the_canonical_service()?;
    denied_quota_cancel_and_lost_transaction_have_no_commit()?;
    manifest_capability_quotas_cannot_exceed_host_limits()?;
    versions_signatures_components_and_schema_fail_closed()?;
    legacy_resolution_preserves_workflow_until_explicit_acceptance()?;
    legacy_provider_projection_uses_signed_manifest_and_canonical_permission_scopes()?;
    extension_owned_component_host_updates_registry_atomically_and_revokes_stale_handles()?;
    component_update_quiesces_active_invocation_and_revokes_the_previous_generation()?;
    component_host_router_replaces_profile_services_and_revokes_the_old_generation()?;
    compiled_component_rejects_an_invocation_authorized_for_another_manifest()?;
    no_wasi_linker_rejects_components_with_wasi_imports()?;

    let manifest_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_directory
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or("workspace root is unavailable")?;
    let target = match std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from) {
        Some(target) if target.is_absolute() => target,
        Some(target) => workspace_root.join(target),
        None => workspace_root.join("target"),
    };
    let artifact_directory = target.join("sim-parity");
    fs::create_dir_all(&artifact_directory)?;
    let component = component_fixture()?;
    let component_sha256 = format!("{:x}", Sha256::digest(&component));
    let list_ports_fixture_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("fixtures/list_ports"))
    );
    let capability_fixture_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("fixtures/capabilities"))
    );
    let wit_source_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../../sim_plugin_sdk/wit/sim-plugin.wit"))
    );
    let sdk_source_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../../sim_plugin_sdk/src/sim_plugin_sdk.rs"))
    );
    let host_source_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../src/sim_plugin_host.rs"))
    );
    let capability_source_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../src/capabilities.rs"))
    );
    let component_host_source_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../src/component_host.rs"))
    );
    let legacy_mapping_source_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../src/legacy_mapping.rs"))
    );
    let manifest_schema_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!(
            "../../sim_plugin_sdk/schema/plugin-manifest-v1.schema.json"
        ))
    );
    let guest_source_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("fixtures/list_ports_component/guest.rs"))
    );
    let guest_lock_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("fixtures/list_ports_component/Cargo.lock"))
    );
    let guest_rebuilder_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!(
            "fixtures/list_ports_component/rebuild_fixture.rs"
        ))
    );
    let port_contract_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!(
            "fixtures/list_ports_component/port_contract.txt"
        ))
    );
    let report = format!(
        concat!(
            "{{\n",
            "  \"validation\": \"VAL-PLUGIN-001\",\n",
            "  \"world\": \"zed:sim-plugin@1.0.0\",\n",
            "  \"backend\": \"native-rust-wasmtime\",\n",
            "  \"environment\": {{\n",
            "    \"os\": \"{}\",\n",
            "    \"arch\": \"{}\",\n",
            "    \"network_used\": false,\n",
            "    \"external_processes_used\": false\n",
            "  }},\n",
            "  \"digests_sha256\": {{\n",
            "    \"component\": \"{}\",\n",
            "    \"list_ports_fixture\": \"{}\",\n",
            "    \"capability_fixture\": \"{}\",\n",
            "    \"wit_source\": \"{}\",\n",
            "    \"sdk_source\": \"{}\",\n",
            "    \"host_source\": \"{}\",\n",
            "    \"capability_source\": \"{}\",\n",
            "    \"component_host_source\": \"{}\",\n",
            "    \"legacy_mapping_source\": \"{}\",\n",
            "    \"manifest_schema\": \"{}\",\n",
            "    \"guest_source\": \"{}\",\n",
            "    \"guest_lock\": \"{}\",\n",
            "    \"guest_rebuilder\": \"{}\",\n",
            "    \"port_contract\": \"{}\"\n",
            "  }},\n",
            "  \"cases\": {{\n",
            "    \"canonical_type_registry\": true,\n",
            "    \"typed_manifest_projection\": true,\n",
            "    \"successful_wit_invocation\": true,\n",
            "    \"port_presence_cardinality\": true,\n",
            "    \"all_value_families\": true,\n",
            "    \"scalar_read_opaque_take\": true,\n",
            "    \"exact_type_validation\": true,\n",
            "    \"ownership_revocation\": true,\n",
            "    \"cancellation_every_port_call\": true,\n",
            "    \"invocation_port_quotas\": true,\n",
            "    \"handle_quota_nonmutation\": true,\n",
            "    \"invocation_handle_isolation\": true,\n",
            "    \"aggregate_value_quotas\": true,\n",
            "    \"value_metadata_validation\": true,\n",
            "    \"filesystem_capability\": true,\n",
            "    \"provider_secret_isolation\": true,\n",
            "    \"clock_random_model_capabilities\": true,\n",
            "    \"output_proposal_staging\": true,\n",
            "    \"sanitized_log_ui_route\": true,\n",
            "    \"quota_cancel_rollback\": true,\n",
            "    \"host_capability_ceilings\": true,\n",
            "    \"semver_signature_component_boundary\": true,\n",
            "    \"legacy_non_destructive_mapping\": true,\n",
            "    \"legacy_signed_translation_sidecar\": true,\n",
            "    \"legacy_wit_1_0_identity_stability\": true,\n",
            "    \"legacy_provider_scope_projection\": true,\n",
            "    \"extension_lifecycle_atomic_replacement\": true,\n",
            "    \"concurrent_update_revocation_gate\": true,\n",
            "    \"cross_manifest_invocation_rejected\": true,\n",
            "    \"wasi_import_rejected\": true\n",
            "  }},\n",
            "  \"passed\": 30,\n",
            "  \"failed\": 0,\n",
            "  \"skipped\": 0\n",
            "}}\n"
        ),
        std::env::consts::OS,
        std::env::consts::ARCH,
        component_sha256,
        list_ports_fixture_sha256,
        capability_fixture_sha256,
        wit_source_sha256,
        sdk_source_sha256,
        host_source_sha256,
        capability_source_sha256,
        component_host_source_sha256,
        legacy_mapping_source_sha256,
        manifest_schema_sha256,
        guest_source_sha256,
        guest_lock_sha256,
        guest_rebuilder_sha256,
        port_contract_sha256,
    );
    fs::write(artifact_directory.join("val-plugin-001.json"), report)?;
    Ok(())
}
