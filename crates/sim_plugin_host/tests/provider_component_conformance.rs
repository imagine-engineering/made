use serde::Deserialize;
use sha2::{Digest, Sha256};
use sim_plugin_host::{
    CancellationToken, CapabilityServiceContext, ComponentLimits, InvocationInputs,
    InvocationMediaContent, PluginCapabilityServices, PluginError, PluginHost,
    ProviderV2WorkerStreamCall,
};
use sim_plugin_sdk::{
    ApiRequirement, ApiVersion, CachePolicy, CapabilityKind, CapabilityQuota, CapabilityRequest,
    DType, DeterminismPolicy, DeviceId, ED25519_SIGNATURE_BYTES, EffectPolicy, InvocationError,
    ManifestProvenance, ManifestSignature, ModelValue, PLUGIN_SIGNATURE_ALGORITHM,
    PROVIDER_BINDING_API_FEATURE, PROVIDER_BINDING_SCHEMA_VERSION, PROVIDER_COMPONENT_WORLD_V2,
    PROVIDER_MANIFEST_SCHEMA_VERSION_V2, PROVIDER_STREAMING_API_FEATURE_V2, PluginManifest,
    PluginNode, PluginPort, PluginSigningKey, PluginValue, PortCardinality, PortDirection,
    PortPresence, PortSerialization, ProviderBindingClaim, ProviderBindingSet,
    ProviderHttpMethodV2, ProviderPluginManifestV2, ProviderStreamingContractV2, RouteDeclaration,
    ScalarValue, StreamId, TensorDescriptor, TensorValue, TypeRegistry, ValueFamily,
};
use sim_runtime::{
    AssetIdentity, Capability, CapabilitySet, PermissionGrant, PermissionPolicy, PluginTrustPolicy,
    PluginVerificationKey, SecretId, WorkerProviderV2InvocationEnvelope,
};
use sim_types::{
    MAX_WORKER_PROVIDER_UPLOAD_URL_BYTES, WorkerProviderCostRequest, WorkerProviderCostResponse,
    WorkerProviderHeader, WorkerProviderHttpMethod, WorkerProviderInvocationContext,
    WorkerProviderProgress, WorkerProviderRequestChunk, WorkerProviderRequestHead,
    WorkerProviderResponseChunk, WorkerProviderResponseFrame, WorkerProviderResponseFrameEvent,
    WorkerProviderResponseHead, WorkerProviderRetryRequest, WorkerProviderStreamError,
    WorkerProviderStreamHandle, WorkerProviderStreamRequest, WorkerProviderStreamResponse,
    WorkerProviderStreamingContract, WorkerProviderTerminal, WorkerProviderUploadCompletion,
    WorkerProviderUploadCompletionRequest, WorkerProviderUploadRequest, WorkerProviderWaitOutcome,
    WorkerProviderWaitRequest,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};
use uuid::Uuid;

const KEY_ID: &str = "task434.publisher";
const KEY: &[u8; 32] = b"task434-provider-fixture-key-01x";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderFixture {
    component: String,
    malformed_component: String,
    oversized_component: String,
    namespace: String,
    provenance_source: String,
    provenance_publisher: String,
    bindings_sha256: String,
    nodes: Vec<NodeFixture>,
    bindings: Vec<BindingFixture>,
    capabilities: Vec<CapabilityFixture>,
    #[serde(default)]
    present_secrets: Option<Vec<String>>,
    #[serde(default)]
    credential_scenarios: Vec<CredentialScenarioFixture>,
    #[serde(default)]
    exercise_resilience: bool,
    #[serde(default)]
    malformed_trailing_chunk_hex: Option<String>,
    #[serde(default)]
    routes: Vec<RouteFixture>,
    streaming: StreamingFixture,
    invocations: Vec<InvocationFixture>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CredentialScenarioFixture {
    present_secrets: Vec<String>,
    expected_secret_id: Option<String>,
    expected_header: Option<HeaderFixture>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteFixture {
    id: String,
    method: String,
    path: String,
    maximum_request_bytes: u64,
    maximum_response_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeFixture {
    id: String,
    display_name: String,
    category: String,
    ports: Vec<PortFixture>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingFixture {
    feature_id: String,
    node_id: String,
    contract_sha256: String,
    transport_schema: String,
    materializer_schema: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortFixture {
    id: String,
    source_type: String,
    direction: String,
    presence: String,
    cardinality: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CapabilityFixture {
    kind: String,
    scope: String,
    maximum_operations: u64,
    maximum_request_bytes: u64,
    maximum_response_bytes: u64,
    maximum_total_bytes: u64,
    maximum_handles: u32,
    timeout_milliseconds: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamingFixture {
    methods: Vec<String>,
    maximum_headers: u16,
    maximum_header_bytes: u32,
    maximum_request_body_bytes: u64,
    maximum_response_body_bytes: u64,
    maximum_chunk_bytes: u32,
    maximum_ndjson_line_bytes: u32,
    maximum_wait_milliseconds: u64,
    maximum_uploads: u32,
    maximum_upload_body_bytes: u64,
    maximum_cost_requests: u32,
    maximum_progress_total: u64,
    uploads: bool,
    cost_requests: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputFixture {
    port_id: String,
    source_type: String,
    values: Vec<ValueFixture>,
    #[serde(default)]
    media: Vec<MediaFixture>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MediaFixture {
    index: u32,
    #[serde(default)]
    batch_index: u32,
    media_type: String,
    bytes_hex: String,
    content_sha256: String,
}

struct FixtureCapabilityServices {
    present_secrets: BTreeSet<String>,
}

impl PluginCapabilityServices for FixtureCapabilityServices {
    fn read_asset(
        &self,
        _identity: &AssetIdentity,
        _context: &CapabilityServiceContext,
    ) -> Result<Vec<u8>, InvocationError> {
        Err(InvocationError::HostFailure(
            "fixture asset service is unavailable".to_owned(),
        ))
    }

    fn call_provider(
        &self,
        _provider: &str,
        _endpoint: &str,
        _body: &[u8],
        _secret_id: Option<&SecretId>,
    ) -> Result<Vec<u8>, InvocationError> {
        Err(InvocationError::HostFailure(
            "fixture provider service is unavailable".to_owned(),
        ))
    }

    fn secret_exists(&self, identifier: &str) -> Result<bool, InvocationError> {
        Ok(self.present_secrets.contains(identifier))
    }

    fn clock_milliseconds(&self, _clock: &str) -> Result<u64, InvocationError> {
        Err(InvocationError::HostFailure(
            "fixture clock service is unavailable".to_owned(),
        ))
    }

    fn random_bytes(&self, _stream: &str, _length: u32) -> Result<Vec<u8>, InvocationError> {
        Err(InvocationError::HostFailure(
            "fixture random service is unavailable".to_owned(),
        ))
    }

    fn open_model(&self, _identifier: &str) -> Result<ModelValue, InvocationError> {
        Err(InvocationError::HostFailure(
            "fixture model service is unavailable".to_owned(),
        ))
    }

    fn sanitize_log(&self, _level: &str, message: &str) -> Result<String, InvocationError> {
        Ok(message.to_owned())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ValueFixture {
    String {
        value: String,
    },
    Integer {
        value: i64,
    },
    Float {
        value: f64,
    },
    Boolean {
        value: bool,
    },
    Record {
        value: BTreeMap<String, ValueFixture>,
    },
    Tensor {
        shape: Vec<u64>,
        dtype: String,
        device: String,
        stream: u64,
        byte_length: u64,
        digest: String,
    },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InvocationFixture {
    node_id: String,
    inputs: Vec<InputFixture>,
    script: Vec<ScriptFixture>,
    outputs: Vec<OutputFixture>,
    receipt_hex: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderFixture {
    name: String,
    value: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum ScriptFixture {
    Start {
        endpoint: String,
        secret_id: Option<String>,
        method: String,
        headers: Vec<HeaderFixture>,
        declared_body_bytes: Option<u64>,
        response_slot: u32,
    },
    RequestChunk {
        slot: u32,
        sequence: u64,
        input_abi: Option<String>,
        #[serde(default)]
        input_index: usize,
        bytes_hex: Option<String>,
        end: bool,
    },
    CheckCancelled {
        slot: u32,
    },
    StartUpload {
        slot: u32,
        port_id: String,
        media_type: String,
        byte_length: u64,
        content_sha256: String,
        response_slot: u32,
    },
    UploadChunk {
        slot: u32,
        sequence: u64,
        bytes_hex: String,
        end: bool,
    },
    CompleteUpload {
        slot: u32,
        download_url: String,
    },
    Retry {
        slot: u32,
        delay_milliseconds: u64,
        response_slot: u32,
    },
    Cost {
        slot: u32,
        operation_id: String,
        currency: String,
        maximum_microunits: u64,
        accepted: bool,
        approved_microunits: u64,
        receipt_hex: String,
    },
    Progress {
        slot: u32,
        sequence: u64,
        completed: u64,
        total: u64,
        message: Option<String>,
    },
    WaitHead {
        slot: u32,
        after_sequence: Option<u64>,
        timeout_milliseconds: u64,
        sequence: u64,
        status: u16,
        headers: Vec<HeaderFixture>,
    },
    WaitChunk {
        slot: u32,
        after_sequence: Option<u64>,
        timeout_milliseconds: u64,
        sequence: u64,
        bytes_hex: String,
    },
    WaitTerminal {
        slot: u32,
        after_sequence: Option<u64>,
        timeout_milliseconds: u64,
        sequence: u64,
        receipt_hex: String,
    },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputFixture {
    port_id: String,
    source_type: String,
    value: ValueFixture,
}

#[derive(Clone, Copy)]
enum ResponseMode {
    Scripted,
    CancelAtCheck,
    WorkerLostAtWait,
    StopAfterStart,
    OversizedFrame,
    MalformedUploadReceipt,
    NonHttpsUploadReceipt,
    QueryUploadReceipt,
    FragmentUploadReceipt,
    ControlUploadReceipt,
    OversizedUploadReceipt,
}

enum InputFault {
    OversizedString { port_id: String, value: String },
    MissingMedia,
    MalformedMedia,
    SourceDigestMismatch,
    ContentDigestMismatch,
    OversizedMedia,
}

#[test]
fn native_claude_image_reaches_the_compiled_component_upload() -> Result<(), Box<dyn Error>> {
    use sim_nodes::{
        NativeNodeComputeSession, NativeNodeContext, NativeNodeServiceIdentity, NativeNodeServices,
        NativePrimitive, NativeStoredPayload, NativeValue,
    };
    use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
    use sim_tensor::{CpuWorkspaceAuthority, ImageTensor, NativeTensorPayload, NativeTensorRole};
    use sim_types::{AttemptId, NodeId, PromptId};
    use std::collections::BTreeMap;

    let (path, fixture) = discover_fixtures()?
        .into_iter()
        .find(|(_, fixture)| fixture.namespace == "zed.sim.provider.anthropic")
        .ok_or("compiled Anthropic fixture is missing")?;
    let component = read_component(&resolve_fixture_path(&path, &fixture.component)?)?;
    let (host, manifest, authorization) = configured_host(&component, &fixture)?;
    let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
    let mut invocation = fixture
        .invocations
        .first()
        .ok_or("invocation is missing")?
        .clone();
    let node = manifest
        .manifest
        .nodes
        .iter()
        .find(|node| node.id == "ClaudeNode")
        .ok_or("Claude node is missing")?;
    let registry = generated_native_node_registry_projection(None)?;
    let descriptor = registry
        .descriptor("ClaudeNode")
        .ok_or("native Claude descriptor is missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let backend = Arc::new(backend);
    let scratch = authority.authorize_workspace(512 * 1024)?;
    let cancellation = CancellationToken::default();
    let attempt_id = AttemptId(Uuid::from_u128(0x43601));
    let node_id = NodeId::from("native-claude-image");
    let generation = NativeHandleStoreGeneration::with_capacities(4, 1024 * 1024)?;
    let store = generation.handle_store_for_attempt(attempt_id);
    let compute = NativeNodeComputeSession::checked(
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x43602), attempt_id, node_id.clone())?,
        backend.clone(),
        StreamId::DEFAULT,
        &scratch,
    )?;
    let context = NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x43603)),
        attempt_id,
        node_id,
        cancellation.clone(),
        scratch.clone(),
        store.clone(),
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?;
    let execution = context.compute_session()?.execution_context(&context)?;
    let image = ImageTensor::from_f32(&backend, &execution, 1, 1, 1, 4, &[0.0, 0.5, 1.0, 0.25])?;
    let expected_png = context.prepare_provider_image_png(&image, 0, Some(1_568 * 1_568))?;
    let handle = store.publish(
        NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
            NativeTensorRole::Image,
            image,
        )?)),
        &cancellation,
    )?;
    let mut native_inputs = BTreeMap::new();
    for input in &invocation.inputs {
        if input.port_id == "images" {
            let dynamic = descriptor
                .source_schema
                .as_ref()
                .ok_or("native source schema is missing")?
                .dynamic_inputs
                .iter()
                .find(|dynamic| dynamic.input.name == "image")
                .ok_or("native image input template is missing")?;
            native_inputs.insert(
                dynamic
                    .names
                    .first()
                    .ok_or("native image names are missing")?
                    .clone(),
                NativeValue::Handle {
                    value: handle.clone(),
                },
            );
            continue;
        }
        let primitive = match input.values.first().ok_or("scalar input is missing")? {
            ValueFixture::String { value } => NativePrimitive::String(value.clone()),
            ValueFixture::Integer { value } => NativePrimitive::Integer(*value),
            ValueFixture::Float { value } => NativePrimitive::Number(*value),
            ValueFixture::Boolean { value } => NativePrimitive::Boolean(*value),
            ValueFixture::Record { .. } | ValueFixture::Tensor { .. } => {
                return Err("unexpected structured or tensor port".into());
            }
        };
        native_inputs.insert(
            input.port_id.clone(),
            NativeValue::Primitive { value: primitive },
        );
    }
    let mut model_fields = BTreeMap::new();
    for name in ["model", "max_tokens", "temperature", "reasoning_effort"] {
        model_fields.insert(
            name.to_owned(),
            native_inputs.remove(name).ok_or("model field is missing")?,
        );
    }
    native_inputs.insert(
        "model".to_owned(),
        sim_nodes::NativeStructuredValue::checked("COMFY_DYNAMICCOMBO_V3", model_fields)?
            .into_runtime_value()?,
    );
    let mut ambiguous_inputs = native_inputs.clone();
    ambiguous_inputs.insert(
        "max_tokens".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(4096),
        },
    );
    assert!(
        futures::executor::block_on(sim_plugin_host::prepare_native_provider_invocation(
            node,
            descriptor,
            ambiguous_inputs,
            "task434-profile",
            &context,
        ))
        .is_err()
    );
    let mut cancelled_context = context.clone();
    cancelled_context.cancellation = CancellationToken::default();
    assert!(cancelled_context.cancellation.cancel());
    let cancelled =
        futures::executor::block_on(sim_plugin_host::prepare_native_provider_invocation(
            node,
            descriptor,
            native_inputs.clone(),
            "task434-profile",
            &cancelled_context,
        ));
    assert!(matches!(
        cancelled,
        Err(sim_nodes::NativeNodeFailure {
            kind: sim_nodes::NativeNodeFailureKind::Interrupted,
            ..
        })
    ));
    let mut invalid_context = context.clone();
    invalid_context.scratch = authority.authorize_workspace(0)?;
    assert!(
        futures::executor::block_on(sim_plugin_host::prepare_native_provider_invocation(
            node,
            descriptor,
            native_inputs.clone(),
            "task434-profile",
            &invalid_context,
        ))
        .is_err()
    );
    assert_eq!(generation.len(), 1);
    assert_eq!(scratch.in_use_bytes(), 0);
    let prepared =
        futures::executor::block_on(sim_plugin_host::prepare_native_provider_invocation(
            node,
            descriptor,
            native_inputs,
            "task434-profile",
            &context,
        ))?;
    assert!(prepared.request.to_bytes()?.len() < 4096);
    let input_abi = prepared
        .request
        .ports()
        .iter()
        .flat_map(|port| {
            port.values().iter().enumerate().map(move |(index, value)| {
                (port.port_id().to_owned(), index, value.abi_bytes().to_vec())
            })
        })
        .collect();
    let mut script = Vec::new();
    for mut step in invocation.script {
        match &mut step {
            ScriptFixture::StartUpload {
                byte_length,
                content_sha256,
                response_slot,
                ..
            } => {
                *byte_length = expected_png.len() as u64;
                *content_sha256 = encode_hex(&Sha256::digest(&expected_png));
                let slot = *response_slot;
                script.push(step);
                let chunks = expected_png.chunks(16);
                let count = chunks.len();
                for (index, bytes) in chunks.enumerate() {
                    script.push(ScriptFixture::UploadChunk {
                        slot,
                        sequence: index as u64,
                        bytes_hex: encode_hex(bytes),
                        end: index + 1 == count,
                    });
                }
            }
            ScriptFixture::UploadChunk { .. } => {}
            _ => script.push(step),
        }
    }
    invocation.script = script;
    let present_secrets = fixture
        .capabilities
        .iter()
        .filter(|capability| capability.kind == "secret")
        .map(|capability| capability.scope.clone())
        .collect();
    let (pending, requests) = invoke_guest_with_present_secrets(
        &host,
        &compiled,
        &manifest,
        &authorization,
        &fixture,
        &invocation,
        ResponseMode::Scripted,
        None,
        present_secrets,
        Some((prepared.inputs, input_abi)),
    )?;
    assert_eq!(requests.len(), invocation.script.len());
    assert_eq!(pending.result().outputs.len(), invocation.outputs.len());
    let types = TypeRegistry::built_in()?;
    for (output, expected) in pending.result().outputs.iter().zip(&invocation.outputs) {
        assert_eq!(output.port_id, expected.port_id);
        assert_eq!(
            PluginValue::from_abi_bytes(&output.value.abi_bytes, &types)?,
            fixture_value(&types, &expected.source_type, &expected.value)?
        );
    }
    assert_eq!(
        pending.result().receipt,
        decode_hex(&invocation.receipt_hex)?
    );
    assert_eq!(generation.len(), 1);
    assert_eq!(scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn discovered_provider_v2_guests_are_host_validated_instantiated_and_invoked()
-> Result<(), Box<dyn Error>> {
    let fixtures = discover_fixtures()?;
    assert!(
        !fixtures.is_empty(),
        "provider-v2 fixture discovery is empty"
    );
    assert!(
        fixtures.iter().any(|(_, fixture)| {
            fixture.nodes.len() > 1
                && fixture.bindings.len() > 1
                && fixture
                    .nodes
                    .iter()
                    .any(|node| node.ports.iter().any(|port| port.cardinality == "list"))
                && fixture.invocations.iter().any(|invocation| {
                    invocation.outputs.len() > 1
                        && invocation
                            .outputs
                            .iter()
                            .any(|output| matches!(output.value, ValueFixture::Tensor { .. }))
                })
                && !fixture.routes.is_empty()
                && fixture.invocations.iter().any(|invocation| {
                    invocation
                        .inputs
                        .iter()
                        .any(|input| !input.media.is_empty())
                })
        }),
        "provider-v2 fixture discovery lacks a routed multi-node typed media fixture"
    );
    for (path, fixture) in fixtures {
        validate_fixture(&fixture)
            .map_err(|error| format!("{} failed fixture validation: {error}", path.display()))?;
        let component_path = resolve_fixture_path(&path, &fixture.component).map_err(|error| {
            format!(
                "{} failed component path validation: {error}",
                path.display()
            )
        })?;
        let component = read_component(&component_path).map_err(|error| {
            format!(
                "{} failed to read component {}: {error}",
                path.display(),
                component_path.display()
            )
        })?;
        assert!(
            component.len() > 8,
            "placeholder component at {}",
            path.display()
        );
        let (host, manifest, authorization) = configured_host(&component, &fixture)
            .map_err(|error| format!("{} failed host configuration: {error}", path.display()))?;
        let compiled = host
            .compile_provider_component_v2(&component, &manifest, &authorization)
            .map_err(|error| format!("{} failed component compilation: {error}", path.display()))?;
        for invocation in &fixture.invocations {
            let (pending, requests) = invoke_guest(
                &host,
                &compiled,
                &manifest,
                &authorization,
                &fixture,
                invocation,
                ResponseMode::Scripted,
                None,
            )
            .map_err(|error| {
                format!(
                    "{} node {} failed canonical host invocation: {error}",
                    path.display(),
                    invocation.node_id
                )
            })?;
            assert_eq!(requests.len(), invocation.script.len());
            assert_eq!(pending.result().outputs.len(), invocation.outputs.len());
            let registry = TypeRegistry::built_in()?;
            for (output, expected) in pending.result().outputs.iter().zip(&invocation.outputs) {
                assert_eq!(output.port_id, expected.port_id);
                let value = PluginValue::from_abi_bytes(&output.value.abi_bytes, &registry)?;
                assert_eq!(
                    value,
                    fixture_value(&registry, &expected.source_type, &expected.value)?
                );
            }
            assert_eq!(
                pending.result().receipt,
                decode_hex(&invocation.receipt_hex)?
            );
        }
    }
    Ok(())
}

#[test]
fn openrouter_unpriceable_model_options_fail_before_transport() -> Result<(), Box<dyn Error>> {
    let (path, fixture) = discover_fixtures()?
        .into_iter()
        .find(|(path, fixture)| {
            fixture.namespace == "zed.sim.provider.openrouter"
                && path
                    .parent()
                    .is_some_and(|parent| parent.ends_with("openrouter"))
        })
        .ok_or("compiled OpenRouter fixture is missing")?;
    let component = read_component(&resolve_fixture_path(&path, &fixture.component)?)?;
    let (host, manifest, authorization) = configured_host(&component, &fixture)?;
    let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
    for field in [
        (
            "search_context_size",
            ValueFixture::String {
                value: "low".into(),
            },
        ),
        (
            "images",
            ValueFixture::Record {
                value: BTreeMap::from([(
                    "image_1".into(),
                    ValueFixture::String {
                        value: "unpriced".into(),
                    },
                )]),
            },
        ),
    ] {
        let mut invocation = fixture.invocations[0].clone();
        let model = invocation
            .inputs
            .iter_mut()
            .find(|input| input.port_id == "model")
            .ok_or("OpenRouter model port missing")?;
        model.values = vec![ValueFixture::Record {
            value: BTreeMap::from([
                (
                    "model".into(),
                    ValueFixture::String {
                        value: "deepseek/deepseek-v4-flash".into(),
                    },
                ),
                (field.0.into(), field.1),
            ]),
        }];
        let result = invoke_guest(
            &host,
            &compiled,
            &manifest,
            &authorization,
            &fixture,
            &invocation,
            ResponseMode::Scripted,
            None,
        );
        match result {
            Err(PluginError::ProviderStreaming(WorkerProviderStreamError::InvalidCostRequest)) => {}
            Err(error) => return Err(format!("unpriceable input returned {error}").into()),
            Ok(_) => return Err("unpriceable input reached transport".into()),
        }
    }
    Ok(())
}

#[test]
fn openrouter_declined_per_run_estimate_never_reaches_response_transport()
-> Result<(), Box<dyn Error>> {
    let (path, fixture) = discover_fixtures()?
        .into_iter()
        .find(|(path, fixture)| {
            fixture.namespace == "zed.sim.provider.openrouter"
                && path
                    .parent()
                    .is_some_and(|parent| parent.ends_with("openrouter"))
        })
        .ok_or("compiled OpenRouter fixture is missing")?;
    let component = read_component(&resolve_fixture_path(&path, &fixture.component)?)?;
    let (host, manifest, authorization) = configured_host(&component, &fixture)?;
    let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
    let mut invocation = fixture.invocations[0].clone();
    let cost_position = invocation
        .script
        .iter()
        .position(|step| matches!(step, ScriptFixture::Cost { .. }))
        .ok_or("OpenRouter cost step is missing")?;
    invocation.script.truncate(cost_position + 1);
    if let ScriptFixture::Cost {
        accepted,
        approved_microunits,
        receipt_hex,
        ..
    } = &mut invocation.script[cost_position]
    {
        *accepted = false;
        *approved_microunits = 0;
        receipt_hex.clear();
    }
    match invoke_guest(
        &host,
        &compiled,
        &manifest,
        &authorization,
        &fixture,
        &invocation,
        ResponseMode::Scripted,
        None,
    ) {
        Err(PluginError::ProviderStreaming(WorkerProviderStreamError::InvalidCostRequest)) => {}
        Err(error) => return Err(format!("declined OpenRouter estimate returned {error}").into()),
        Ok(_) => return Err("declined OpenRouter estimate reached transport".into()),
    }
    Ok(())
}

#[test]
fn openrouter_unsigned_payload_matches_compiled_component() -> Result<(), Box<dyn Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root = repository.join("crates/sim_test_support/fixtures/providers/openrouter");
    let component = read_component(&root.join("provider_v2_component"))?;
    let payload = fs::read_to_string(root.join("signing_payload.fixture"))?;
    let lines = payload
        .lines()
        .map(|line| {
            line.split_once('=')
                .ok_or("malformed OpenRouter signing field")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let field = |name: &str| {
        lines
            .iter()
            .find_map(|(key, value)| (*key == name).then_some(*value))
    };
    let digest = encode_hex(&Sha256::digest(&component));
    let size = component.len().to_string();
    assert_eq!(field("component_sha256"), Some(digest.as_str()));
    assert_eq!(field("component_bytes"), Some(size.as_str()));
    assert_eq!(
        field("native_contract_sha256"),
        Some("2c6dbff1b7f89da063de276e6737f58e1dc98a85ad736f620bea96cab9e0a3d5")
    );
    assert_eq!(
        field("bindings_sha256"),
        Some("8b941f79f9560c7ec53819291d84e42d8c68ae2a7a70722143e55e6ea411f741")
    );
    assert_eq!(
        field("estimate_policy"),
        Some("explicit-per-run-request-bound-not-guaranteed")
    );
    assert_eq!(field("media_and_search"), Some("reject-unpriceable"));
    assert_eq!(field("billable_attempts"), Some("20"));
    assert!(component.starts_with(b"\0asm"));
    assert!(root.join("component_source/Cargo.lock").is_file());
    Ok(())
}

#[test]
fn compiled_provider_guests_enforce_fixture_credential_selection() -> Result<(), Box<dyn Error>> {
    let mut scenario_count = 0_usize;
    for (path, fixture) in discover_fixtures()? {
        if fixture.credential_scenarios.is_empty() {
            continue;
        }
        let component = read_component(&resolve_fixture_path(&path, &fixture.component)?)?;
        let (host, manifest, authorization) = configured_host(&component, &fixture)?;
        let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
        let invocation = fixture
            .invocations
            .iter()
            .find(|invocation| invocation.inputs.iter().all(|input| input.media.is_empty()))
            .ok_or("credential fixture has no invocation without media")?;
        for scenario in &fixture.credential_scenarios {
            scenario_count = scenario_count.saturating_add(1);
            let mut scenario_invocation = invocation.clone();
            match (&scenario.expected_secret_id, &scenario.expected_header) {
                (Some(expected_secret_id), Some(expected_header)) => {
                    let Some(ScriptFixture::Start {
                        secret_id, headers, ..
                    }) = scenario_invocation.script.first_mut()
                    else {
                        return Err("credential fixture does not begin with start".into());
                    };
                    *secret_id = Some(expected_secret_id.clone());
                    headers.retain(|header| {
                        !header.name.eq_ignore_ascii_case("authorization")
                            && !header.name.eq_ignore_ascii_case("x-api-key")
                    });
                    headers.insert(1, expected_header.clone());
                }
                (None, None) => {}
                _ => return Err("credential fixture expectation is incomplete".into()),
            }
            let result = invoke_guest_with_present_secrets(
                &host,
                &compiled,
                &manifest,
                &authorization,
                &fixture,
                &scenario_invocation,
                ResponseMode::StopAfterStart,
                None,
                scenario.present_secrets.iter().cloned().collect(),
                None,
            );
            match (&scenario.expected_secret_id, result) {
                (
                    Some(_),
                    Err(PluginError::ProviderStreaming(
                        WorkerProviderStreamError::HostFailure
                        | WorkerProviderStreamError::RevokedHandle,
                    )),
                ) => {}
                (
                    None,
                    Err(PluginError::ProviderStreaming(
                        WorkerProviderStreamError::InvalidRequestAuthority,
                    )),
                ) => {}
                (_, Err(error)) => {
                    return Err(format!("credential scenario returned wrong error: {error}").into());
                }
                (_, Ok(_)) => return Err("credential scenario unexpectedly completed".into()),
            }
        }
    }
    assert!(
        scenario_count > 0,
        "no compiled credential scenarios executed"
    );
    Ok(())
}

#[test]
fn compiled_provider_guests_cancel_atomically_and_restart_after_worker_loss()
-> Result<(), Box<dyn Error>> {
    let mut fixture_count = 0_usize;
    for (path, fixture) in discover_fixtures()? {
        if !fixture.exercise_resilience {
            continue;
        }
        fixture_count = fixture_count.saturating_add(1);
        let component = read_component(&resolve_fixture_path(&path, &fixture.component)?)?;
        let (host, manifest, authorization) = configured_host(&component, &fixture)?;
        let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
        let invocation = fixture
            .invocations
            .first()
            .ok_or("resilience fixture has no invocation")?;

        match invoke_guest(
            &host,
            &compiled,
            &manifest,
            &authorization,
            &fixture,
            invocation,
            ResponseMode::CancelAtCheck,
            None,
        ) {
            Err(PluginError::ProviderStreaming(WorkerProviderStreamError::Cancelled)) => {}
            Err(error) => return Err(format!("compiled cancellation returned {error}").into()),
            Ok(_) => return Err("compiled cancellation produced a proposal".into()),
        }

        match invoke_guest(
            &host,
            &compiled,
            &manifest,
            &authorization,
            &fixture,
            invocation,
            ResponseMode::WorkerLostAtWait,
            None,
        ) {
            Err(PluginError::ProviderStreaming(
                WorkerProviderStreamError::HostFailure | WorkerProviderStreamError::RevokedHandle,
            )) => {}
            Err(error) => return Err(format!("compiled worker loss returned {error}").into()),
            Ok(_) => return Err("compiled worker loss produced a proposal".into()),
        }

        let (restarted_host, restarted_manifest, restarted_authorization) =
            configured_host(&component, &fixture)?;
        let restarted_compiled = restarted_host.compile_provider_component_v2(
            &component,
            &restarted_manifest,
            &restarted_authorization,
        )?;
        let (restarted, requests) = invoke_guest(
            &restarted_host,
            &restarted_compiled,
            &restarted_manifest,
            &restarted_authorization,
            &fixture,
            invocation,
            ResponseMode::Scripted,
            None,
        )
        .map_err(|error| format!("compiled restart after worker loss returned {error}"))?;
        assert_eq!(requests.len(), invocation.script.len());
        assert_eq!(restarted.result().outputs.len(), invocation.outputs.len());
    }
    assert!(
        fixture_count > 0,
        "no compiled resilience fixtures executed"
    );
    Ok(())
}

#[test]
fn compiled_provider_guests_do_not_complete_a_valid_prefix_with_malformed_trailing_data()
-> Result<(), Box<dyn Error>> {
    let mut fixture_count = 0_usize;
    for (path, fixture) in discover_fixtures()? {
        let Some(trailing_chunk_hex) = &fixture.malformed_trailing_chunk_hex else {
            continue;
        };
        fixture_count = fixture_count.saturating_add(1);
        let component = read_component(&resolve_fixture_path(&path, &fixture.component)?)?;
        let (host, manifest, authorization) = configured_host(&component, &fixture)?;
        let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
        let mut invocation = fixture
            .invocations
            .iter()
            .find(|invocation| invocation.inputs.iter().all(|input| input.media.is_empty()))
            .cloned()
            .ok_or("malformed-stream fixture has no invocation without media")?;
        let terminal_index = invocation
            .script
            .iter()
            .position(|step| matches!(step, ScriptFixture::WaitTerminal { .. }))
            .ok_or("malformed-stream fixture has no terminal wait")?;
        let (slot, after_sequence, timeout_milliseconds, sequence) =
            match &mut invocation.script[terminal_index] {
                ScriptFixture::WaitTerminal {
                    slot,
                    after_sequence,
                    timeout_milliseconds,
                    sequence,
                    ..
                } => {
                    let inserted_after = *after_sequence;
                    let inserted_sequence = *sequence;
                    *after_sequence = Some(inserted_sequence);
                    *sequence = sequence
                        .checked_add(1)
                        .ok_or("malformed-stream sequence overflow")?;
                    (
                        *slot,
                        inserted_after,
                        *timeout_milliseconds,
                        inserted_sequence,
                    )
                }
                _ => return Err("malformed-stream terminal changed during preparation".into()),
            };
        invocation.script.insert(
            terminal_index,
            ScriptFixture::WaitChunk {
                slot,
                after_sequence,
                timeout_milliseconds,
                sequence,
                bytes_hex: trailing_chunk_hex.clone(),
            },
        );
        invocation
            .script
            .insert(terminal_index + 1, ScriptFixture::CheckCancelled { slot });
        match invoke_guest(
            &host,
            &compiled,
            &manifest,
            &authorization,
            &fixture,
            &invocation,
            ResponseMode::Scripted,
            None,
        ) {
            Err(PluginError::ProviderStreaming(WorkerProviderStreamError::InvalidTerminal)) => {}
            Err(error) => return Err(format!("malformed trailing data returned {error}").into()),
            Ok(_) => return Err("malformed trailing data produced a proposal".into()),
        }
    }
    assert!(
        fixture_count > 0,
        "no compiled malformed trailing-data fixtures executed"
    );
    Ok(())
}

#[test]
fn malformed_components_claims_guest_results_and_host_frames_are_atomic()
-> Result<(), Box<dyn Error>> {
    for (path, fixture) in discover_fixtures()? {
        let component = read_component(&resolve_fixture_path(&path, &fixture.component)?)?;
        let (host, manifest, authorization) = configured_host(&component, &fixture)?;
        for malformed in [&b"not-wasm"[..], &b"\0asm\x0a\0\x01\0"[..]] {
            let malformed_manifest =
                signed_manifest(encode_hex(&Sha256::digest(malformed)), &fixture)?;
            let malformed_authorization = trust_policy()?.authorize_provider_manifest_v2(
                &malformed_manifest,
                &permission_policy(&malformed_manifest.manifest)?,
            )?;
            assert!(matches!(
                host.compile_provider_component_v2(
                    malformed,
                    &malformed_manifest,
                    &malformed_authorization,
                ),
                Err(PluginError::ComponentCompilation(_))
            ));
        }

        let mut wrong_claim = manifest.clone();
        let binding = wrong_claim
            .manifest
            .provider_binding
            .as_mut()
            .ok_or("provider binding missing")?;
        binding.bindings[0].contract_sha256 = "1".repeat(64);
        binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
        resign(&mut wrong_claim)?;
        let wrong_authorization = trust_policy()?.authorize_provider_manifest_v2(
            &wrong_claim,
            &permission_policy(&wrong_claim.manifest)?,
        )?;
        assert!(matches!(
            host.compile_provider_component_v2(&component, &wrong_claim, &wrong_authorization),
            Err(PluginError::ManifestProjectionMismatch)
        ));

        let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
        let invocation = fixture
            .invocations
            .iter()
            .find(|invocation| {
                invocation.inputs.iter().any(|input| {
                    input
                        .values
                        .iter()
                        .any(|value| matches!(value, ValueFixture::String { .. }))
                })
            })
            .ok_or("fixture has no bounded string input")?;
        let oversized_input_port = invocation
            .inputs
            .iter()
            .find(|input| {
                input
                    .values
                    .iter()
                    .any(|value| matches!(value, ValueFixture::String { .. }))
            })
            .map(|input| input.port_id.as_str())
            .ok_or("fixture has no bounded string input")?;
        let oversized_result = invoke_guest(
            &host,
            &compiled,
            &manifest,
            &authorization,
            &fixture,
            invocation,
            ResponseMode::Scripted,
            Some(InputFault::OversizedString {
                port_id: oversized_input_port.to_owned(),
                value: "x".repeat(usize::try_from(fixture.streaming.maximum_chunk_bytes)? + 1),
            }),
        );
        assert!(matches!(
            oversized_result,
            Err(PluginError::ProviderStreaming(
                WorkerProviderStreamError::ChunkLimit
                    | WorkerProviderStreamError::BodyLimit
                    | WorkerProviderStreamError::InvalidInvocationResult
            ))
        ));

        if let Some(media_invocation) = fixture.invocations.iter().find(|invocation| {
            invocation
                .inputs
                .iter()
                .any(|input| !input.media.is_empty())
        }) {
            match invoke_guest(
                &host,
                &compiled,
                &manifest,
                &authorization,
                &fixture,
                media_invocation,
                ResponseMode::Scripted,
                Some(InputFault::MissingMedia),
            ) {
                Err(PluginError::ProviderStreaming(WorkerProviderStreamError::InvalidUpload)) => {}
                Err(error) => {
                    return Err(format!("missing media returned wrong error: {error}").into());
                }
                Ok(_) => return Err("missing media produced a proposal".into()),
            }
            for fault in [
                InputFault::MalformedMedia,
                InputFault::SourceDigestMismatch,
                InputFault::ContentDigestMismatch,
            ] {
                assert!(matches!(
                    invoke_guest(
                        &host,
                        &compiled,
                        &manifest,
                        &authorization,
                        &fixture,
                        media_invocation,
                        ResponseMode::Scripted,
                        Some(fault),
                    ),
                    Err(PluginError::Invocation(InvocationError::HostFailure(_)))
                ));
            }
            assert!(matches!(
                invoke_guest(
                    &host,
                    &compiled,
                    &manifest,
                    &authorization,
                    &fixture,
                    media_invocation,
                    ResponseMode::Scripted,
                    Some(InputFault::OversizedMedia),
                ),
                Err(PluginError::Invocation(
                    InvocationError::InvocationQuotaExceeded { .. }
                ))
            ));
        }

        let malformed_frame = invoke_guest(
            &host,
            &compiled,
            &manifest,
            &authorization,
            &fixture,
            invocation,
            ResponseMode::OversizedFrame,
            None,
        );
        assert!(matches!(
            malformed_frame,
            Err(PluginError::ProviderStreaming(
                WorkerProviderStreamError::ChunkLimit
            ))
        ));

        if invocation
            .script
            .iter()
            .any(|step| matches!(step, ScriptFixture::CompleteUpload { .. }))
        {
            for mode in [
                ResponseMode::MalformedUploadReceipt,
                ResponseMode::NonHttpsUploadReceipt,
                ResponseMode::QueryUploadReceipt,
                ResponseMode::FragmentUploadReceipt,
                ResponseMode::ControlUploadReceipt,
                ResponseMode::OversizedUploadReceipt,
            ] {
                assert!(matches!(
                    invoke_guest(
                        &host,
                        &compiled,
                        &manifest,
                        &authorization,
                        &fixture,
                        invocation,
                        mode,
                        None,
                    ),
                    Err(PluginError::ProviderStreaming(
                        WorkerProviderStreamError::InvalidUpload
                    ))
                ));
            }
        }

        for variant in [&fixture.malformed_component, &fixture.oversized_component] {
            let component = read_component(&resolve_fixture_path(&path, variant)?)?;
            let (host, manifest, authorization) = configured_host(&component, &fixture)?;
            let compiled =
                host.compile_provider_component_v2(&component, &manifest, &authorization)?;
            match invoke_guest(
                &host,
                &compiled,
                &manifest,
                &authorization,
                &fixture,
                invocation,
                ResponseMode::Scripted,
                None,
            ) {
                Err(PluginError::ProviderStreaming(
                    WorkerProviderStreamError::InvalidInvocationResult,
                )) => {}
                Err(error) => {
                    return Err(format!("guest variant returned wrong error: {error}").into());
                }
                Ok(_) => return Err("invalid guest variant produced a proposal".into()),
            }
        }
    }
    Ok(())
}

fn invoke_guest(
    host: &PluginHost,
    compiled: &sim_plugin_host::CompiledPlugin,
    manifest: &ProviderPluginManifestV2,
    authorization: &sim_runtime::ProviderManifestAuthorizationV2,
    fixture: &ProviderFixture,
    invocation_fixture: &InvocationFixture,
    mode: ResponseMode,
    input_fault: Option<InputFault>,
) -> Result<
    (
        sim_plugin_host::ProviderV2WorkerPendingInvocation,
        Vec<WorkerProviderStreamRequest>,
    ),
    PluginError,
> {
    let present_secrets = fixture
        .present_secrets
        .clone()
        .unwrap_or_else(|| {
            fixture
                .capabilities
                .iter()
                .filter(|capability| capability.kind == "secret")
                .map(|capability| capability.scope.clone())
                .collect()
        })
        .into_iter()
        .collect();
    invoke_guest_with_present_secrets(
        host,
        compiled,
        manifest,
        authorization,
        fixture,
        invocation_fixture,
        mode,
        input_fault,
        present_secrets,
        None,
    )
}

fn invoke_guest_with_present_secrets(
    host: &PluginHost,
    compiled: &sim_plugin_host::CompiledPlugin,
    manifest: &ProviderPluginManifestV2,
    authorization: &sim_runtime::ProviderManifestAuthorizationV2,
    fixture: &ProviderFixture,
    invocation_fixture: &InvocationFixture,
    mode: ResponseMode,
    input_fault: Option<InputFault>,
    present_secrets: BTreeSet<String>,
    prepared_inputs: Option<(InvocationInputs, Vec<(String, usize, Vec<u8>)>)>,
) -> Result<
    (
        sim_plugin_host::ProviderV2WorkerPendingInvocation,
        Vec<WorkerProviderStreamRequest>,
    ),
    PluginError,
> {
    let context = WorkerProviderInvocationContext {
        session_id: Uuid::from_u128(0x434),
        session_generation: 1,
        invocation: 1,
        generation: 1,
    };
    let envelope: WorkerProviderV2InvocationEnvelope = serde_json::from_value(serde_json::json!({
        "context": context,
        "streaming_contract": worker_contract(&manifest.streaming),
        "provider_manifest_sha256": encode_hex(&Sha256::digest(manifest.signing_payload()?)),
        "proposal_generation": 1,
        "finalization_nonce": vec![67_u8; 32],
    }))
    .map_err(|_| {
        PluginError::ProviderStreaming(WorkerProviderStreamError::InvalidInvocationResult)
    })?;
    let (inputs, input_abi) = if let Some(prepared) = prepared_inputs {
        prepared
    } else {
        invocation_inputs(fixture, invocation_fixture, input_fault.as_ref()).map_err(|_| {
            PluginError::ProviderStreaming(WorkerProviderStreamError::InvalidInvocationResult)
        })?
    };
    let invocation = host.begin_invocation(
        &manifest.manifest,
        authorization.authorization(),
        &invocation_fixture.node_id,
        inputs,
        Arc::new(FixtureCapabilityServices { present_secrets }),
        CancellationToken::default(),
    )?;
    let (sender, receiver) = async_channel::bounded(1);
    let response_context = context;
    let script = invocation_fixture.script.clone();
    let maximum_chunk_bytes = fixture.streaming.maximum_chunk_bytes;
    let responder = thread::spawn(move || {
        respond_to_guest(
            receiver,
            response_context,
            &script,
            &input_abi,
            maximum_chunk_bytes,
            mode,
        )
    });
    let result = host.invoke_provider_component_v2_for_worker(
        compiled,
        invocation,
        &invocation_fixture.node_id,
        envelope,
        CancellationToken::default(),
        sender,
    );
    let requests = responder
        .join()
        .map_err(|_| PluginError::ProviderStreaming(WorkerProviderStreamError::HostFailure))?;
    if matches!(input_fault, Some(InputFault::MissingMedia)) && !requests.is_empty() {
        return Err(PluginError::ProviderStreaming(
            WorkerProviderStreamError::HostFailure,
        ));
    }
    result.map(|pending| (pending, requests))
}

fn respond_to_guest(
    receiver: async_channel::Receiver<ProviderV2WorkerStreamCall>,
    context: WorkerProviderInvocationContext,
    script: &[ScriptFixture],
    input_abi: &[(String, usize, Vec<u8>)],
    maximum_chunk_bytes: u32,
    mode: ResponseMode,
) -> Vec<WorkerProviderStreamRequest> {
    let mut requests = Vec::new();
    while let Ok(call) = receiver.recv_blocking() {
        let index = requests.len();
        let step = script.get(index).unwrap_or_else(|| {
            panic!(
                "unexpected provider-v2 request {index}: {:?}",
                call.request()
            )
        });
        let (expected, mut response) = scripted_exchange(step, &context, input_abi)
            .unwrap_or_else(|error| panic!("invalid provider-v2 fixture script {index}: {error}"));
        assert_eq!(
            call.request(),
            &expected,
            "provider-v2 request {index} differs"
        );
        if matches!(mode, ResponseMode::CancelAtCheck)
            && matches!(step, ScriptFixture::CheckCancelled { .. })
        {
            requests.push(expected);
            if let Err(error) = call.respond(WorkerProviderStreamResponse::Unit(Err(
                WorkerProviderStreamError::Cancelled,
            ))) {
                panic!("cancelled provider-v2 response failed: {error}");
            }
            break;
        }
        if matches!(mode, ResponseMode::WorkerLostAtWait)
            && matches!(
                step,
                ScriptFixture::WaitHead { .. }
                    | ScriptFixture::WaitChunk { .. }
                    | ScriptFixture::WaitTerminal { .. }
            )
        {
            requests.push(expected);
            break;
        }
        if let WorkerProviderStreamResponse::UploadCompletion(Ok(completion)) = &mut response {
            completion.download_url = match mode {
                ResponseMode::MalformedUploadReceipt => "https://".to_owned(),
                ResponseMode::NonHttpsUploadReceipt => {
                    "http://fixture.invalid/storage/reference.bin".to_owned()
                }
                ResponseMode::QueryUploadReceipt => {
                    "https://fixture.invalid/storage/reference.bin?token=secret".to_owned()
                }
                ResponseMode::FragmentUploadReceipt => {
                    "https://fixture.invalid/storage/reference.bin#fragment".to_owned()
                }
                ResponseMode::ControlUploadReceipt => {
                    "https://fixture.invalid/storage/reference.bin\nnext".to_owned()
                }
                ResponseMode::OversizedUploadReceipt => format!(
                    "https://fixture.invalid/storage/{}",
                    "x".repeat(MAX_WORKER_PROVIDER_UPLOAD_URL_BYTES)
                ),
                ResponseMode::Scripted
                | ResponseMode::CancelAtCheck
                | ResponseMode::WorkerLostAtWait
                | ResponseMode::StopAfterStart
                | ResponseMode::OversizedFrame => completion.download_url.clone(),
            };
        }
        if matches!(mode, ResponseMode::OversizedFrame)
            && matches!(response, WorkerProviderStreamResponse::Wait(_))
            && matches!(
                step,
                ScriptFixture::WaitChunk { .. } | ScriptFixture::WaitTerminal { .. }
            )
        {
            response = WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: primary_handle(step, &context),
                    sequence: wait_response_sequence(step),
                    event: WorkerProviderResponseFrameEvent::Chunk(
                        WorkerProviderResponseChunk::Binary(vec![
                            0;
                            usize::try_from(
                                maximum_chunk_bytes
                            )
                            .unwrap_or(usize::MAX)
                            .saturating_add(1)
                        ]),
                    ),
                },
            )));
            requests.push(expected);
            if let Err(error) = call.respond(response) {
                panic!("oversized provider-v2 frame response failed: {error}");
            }
            break;
        }
        requests.push(expected);
        let stop_after_start = matches!(mode, ResponseMode::StopAfterStart)
            && matches!(step, ScriptFixture::Start { .. });
        if call.respond(response).is_err() || stop_after_start || requests.len() == script.len() {
            break;
        }
    }
    requests
}

fn scripted_exchange(
    step: &ScriptFixture,
    context: &WorkerProviderInvocationContext,
    input_abi: &[(String, usize, Vec<u8>)],
) -> Result<(WorkerProviderStreamRequest, WorkerProviderStreamResponse), Box<dyn Error>> {
    let exchange = match step {
        ScriptFixture::Start {
            endpoint,
            secret_id,
            method,
            headers,
            declared_body_bytes,
            response_slot,
        } => (
            WorkerProviderStreamRequest::StartRequest {
                context: context.clone(),
                head: WorkerProviderRequestHead {
                    endpoint: endpoint.clone(),
                    secret_id: secret_id.clone(),
                    method: worker_method(method)?,
                    headers: worker_headers(headers),
                    declared_body_bytes: *declared_body_bytes,
                },
            },
            WorkerProviderStreamResponse::Stream(Ok(stream_handle(context, *response_slot))),
        ),
        ScriptFixture::RequestChunk {
            slot,
            sequence,
            input_abi: input_port,
            input_index,
            bytes_hex,
            end,
        } => {
            let bytes = match (input_port, bytes_hex) {
                (Some(port), None) => input_abi
                    .iter()
                    .find(|(candidate, index, _)| candidate == port && index == input_index)
                    .map(|(_, _, bytes)| bytes.clone())
                    .ok_or("script input ABI is missing")?,
                (None, Some(bytes)) => decode_hex(bytes)?,
                _ => return Err("request chunk must select exactly one byte source".into()),
            };
            (
                WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                    handle: stream_handle(context, *slot),
                    sequence: *sequence,
                    bytes,
                    end: *end,
                }),
                WorkerProviderStreamResponse::Unit(Ok(())),
            )
        }
        ScriptFixture::CheckCancelled { slot } => (
            WorkerProviderStreamRequest::CheckCancelled(stream_handle(context, *slot)),
            WorkerProviderStreamResponse::Unit(Ok(())),
        ),
        ScriptFixture::StartUpload {
            slot,
            port_id,
            media_type,
            byte_length,
            content_sha256,
            response_slot,
        } => (
            WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                handle: stream_handle(context, *slot),
                port_id: port_id.clone(),
                media_type: media_type.clone(),
                byte_length: *byte_length,
                content_sha256: content_sha256.clone(),
            }),
            WorkerProviderStreamResponse::Stream(Ok(stream_handle(context, *response_slot))),
        ),
        ScriptFixture::UploadChunk {
            slot,
            sequence,
            bytes_hex,
            end,
        } => (
            WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                handle: stream_handle(context, *slot),
                sequence: *sequence,
                bytes: decode_hex(bytes_hex)?,
                end: *end,
            }),
            WorkerProviderStreamResponse::Unit(Ok(())),
        ),
        ScriptFixture::CompleteUpload { slot, download_url } => (
            WorkerProviderStreamRequest::CompleteUpload(WorkerProviderUploadCompletionRequest {
                handle: stream_handle(context, *slot),
            }),
            WorkerProviderStreamResponse::UploadCompletion(Ok(WorkerProviderUploadCompletion {
                download_url: download_url.clone(),
            })),
        ),
        ScriptFixture::Retry {
            slot,
            delay_milliseconds,
            response_slot,
        } => (
            WorkerProviderStreamRequest::RetryRequest(WorkerProviderRetryRequest {
                handle: stream_handle(context, *slot),
                delay_milliseconds: *delay_milliseconds,
            }),
            WorkerProviderStreamResponse::Stream(Ok(stream_handle(context, *response_slot))),
        ),
        ScriptFixture::Cost {
            slot,
            operation_id,
            currency,
            maximum_microunits,
            accepted,
            approved_microunits,
            receipt_hex,
        } => (
            WorkerProviderStreamRequest::RequestCost(WorkerProviderCostRequest {
                handle: stream_handle(context, *slot),
                operation: operation_id.clone(),
                currency: currency.clone(),
                maximum_microunits: *maximum_microunits,
            }),
            WorkerProviderStreamResponse::Cost(Ok(WorkerProviderCostResponse {
                accepted: *accepted,
                approved_microunits: *approved_microunits,
                receipt: decode_hex(receipt_hex)?,
            })),
        ),
        ScriptFixture::Progress {
            slot,
            sequence,
            completed,
            total,
            message,
        } => (
            WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
                handle: stream_handle(context, *slot),
                sequence: *sequence,
                completed: *completed,
                total: *total,
                message: message.clone(),
            }),
            WorkerProviderStreamResponse::Unit(Ok(())),
        ),
        ScriptFixture::WaitHead {
            slot,
            after_sequence,
            timeout_milliseconds,
            sequence,
            status,
            headers,
        } => (
            wait_request(context, *slot, *after_sequence, *timeout_milliseconds),
            WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: stream_handle(context, *slot),
                    sequence: *sequence,
                    event: WorkerProviderResponseFrameEvent::Head(WorkerProviderResponseHead {
                        status: *status,
                        headers: worker_headers(headers),
                    }),
                },
            ))),
        ),
        ScriptFixture::WaitChunk {
            slot,
            after_sequence,
            timeout_milliseconds,
            sequence,
            bytes_hex,
        } => (
            wait_request(context, *slot, *after_sequence, *timeout_milliseconds),
            WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: stream_handle(context, *slot),
                    sequence: *sequence,
                    event: WorkerProviderResponseFrameEvent::Chunk(
                        WorkerProviderResponseChunk::Binary(decode_hex(bytes_hex)?),
                    ),
                },
            ))),
        ),
        ScriptFixture::WaitTerminal {
            slot,
            after_sequence,
            timeout_milliseconds,
            sequence,
            receipt_hex,
        } => (
            wait_request(context, *slot, *after_sequence, *timeout_milliseconds),
            WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: stream_handle(context, *slot),
                    sequence: *sequence,
                    event: WorkerProviderResponseFrameEvent::Terminal(
                        WorkerProviderTerminal::Completed(decode_hex(receipt_hex)?),
                    ),
                },
            ))),
        ),
    };
    Ok(exchange)
}

fn primary_handle(
    step: &ScriptFixture,
    context: &WorkerProviderInvocationContext,
) -> WorkerProviderStreamHandle {
    let slot = match step {
        ScriptFixture::WaitHead { slot, .. }
        | ScriptFixture::WaitChunk { slot, .. }
        | ScriptFixture::WaitTerminal { slot, .. } => *slot,
        _ => 1,
    };
    stream_handle(context, slot)
}

fn wait_response_sequence(step: &ScriptFixture) -> u64 {
    match step {
        ScriptFixture::WaitHead { sequence, .. }
        | ScriptFixture::WaitChunk { sequence, .. }
        | ScriptFixture::WaitTerminal { sequence, .. } => *sequence,
        _ => 0,
    }
}

fn wait_request(
    context: &WorkerProviderInvocationContext,
    slot: u32,
    after_sequence: Option<u64>,
    timeout_milliseconds: u64,
) -> WorkerProviderStreamRequest {
    WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
        handle: stream_handle(context, slot),
        after_sequence,
        timeout_milliseconds,
    })
}

fn stream_handle(
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

fn configured_host(
    component: &[u8],
    fixture: &ProviderFixture,
) -> Result<
    (
        PluginHost,
        ProviderPluginManifestV2,
        sim_runtime::ProviderManifestAuthorizationV2,
    ),
    Box<dyn Error>,
> {
    let manifest = signed_manifest(encode_hex(&Sha256::digest(component)), fixture)?;
    let trust = trust_policy()?;
    let authorization =
        trust.authorize_provider_manifest_v2(&manifest, &permission_policy(&manifest.manifest)?)?;
    let host = PluginHost::with_configuration(
        ComponentLimits {
            epoch_deadline_ticks: 50,
            maximum_value_bytes: 1024 * 1024,
            ..ComponentLimits::default()
        },
        sim_plugin_host::DEFAULT_API_FEATURES
            .iter()
            .map(|value| (*value).to_owned()),
    )?;
    Ok((host, manifest, authorization))
}

fn signed_manifest(
    component_digest: String,
    fixture: &ProviderFixture,
) -> Result<ProviderPluginManifestV2, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let mut binding = ProviderBindingSet {
        schema_version: PROVIDER_BINDING_SCHEMA_VERSION,
        implementation_namespace: fixture.namespace.clone(),
        bindings_sha256: "0".repeat(64),
        bindings: fixture
            .bindings
            .iter()
            .map(|claim| {
                Ok(ProviderBindingClaim {
                    feature_id: claim.feature_id.clone(),
                    node_id: claim.node_id.clone(),
                    contract_sha256: claim.contract_sha256.clone(),
                    transport_schema: claim.transport_schema.parse()?,
                    materializer_schema: claim.materializer_schema.parse()?,
                })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
    };
    binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
    if binding.bindings_sha256 != fixture.bindings_sha256 {
        return Err(format!(
            "fixture binding digest is not canonical: expected {}, actual {}",
            fixture.bindings_sha256, binding.bindings_sha256
        )
        .into());
    }
    let mut inner = PluginManifest {
        schema_version: 1,
        identifier: fixture.namespace.clone(),
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
            source: fixture.provenance_source.clone(),
            publisher: fixture.provenance_publisher.clone(),
            registry: Some("fixture://signed-registry".to_owned()),
        },
        provider_binding: Some(binding),
        nodes: fixture
            .nodes
            .iter()
            .map(|node| {
                Ok(PluginNode {
                    id: node.id.clone(),
                    version: ApiVersion::new(1, 0, 0),
                    display_name: node.display_name.clone(),
                    category: node.category.clone(),
                    ports: node
                        .ports
                        .iter()
                        .map(|port_fixture| port(&registry, port_fixture))
                        .collect::<Result<Vec<_>, _>>()?,
                    determinism: DeterminismPolicy::External,
                    cache: CachePolicy::Never,
                    effects: EffectPolicy::Provider,
                })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
        capabilities: fixture
            .capabilities
            .iter()
            .map(capability)
            .collect::<Result<Vec<_>, _>>()?,
        ui: Vec::new(),
        routes: fixture
            .routes
            .iter()
            .map(|route| RouteDeclaration {
                id: route.id.clone(),
                method: route.method.clone(),
                path: route.path.clone(),
                maximum_request_bytes: route.maximum_request_bytes,
                maximum_response_bytes: route.maximum_response_bytes,
            })
            .collect(),
        legacy_mappings: Vec::new(),
    };
    inner.signature.value = signing_key()?.sign_manifest(&inner)?;
    let mut manifest = ProviderPluginManifestV2 {
        schema_version: PROVIDER_MANIFEST_SCHEMA_VERSION_V2,
        component_world: PROVIDER_COMPONENT_WORLD_V2.to_owned(),
        manifest: inner,
        streaming: streaming_contract(&fixture.streaming)?,
        signature: ManifestSignature {
            algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
            key_id: KEY_ID.to_owned(),
            value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
        },
    };
    manifest.signature.value = signing_key()?.sign_provider_manifest_v2(&manifest)?;
    Ok(manifest)
}

fn resign(manifest: &mut ProviderPluginManifestV2) -> Result<(), Box<dyn Error>> {
    manifest.manifest.signature.value = "0".repeat(ED25519_SIGNATURE_BYTES * 2);
    manifest.manifest.signature.value = signing_key()?.sign_manifest(&manifest.manifest)?;
    manifest.signature.value = "0".repeat(ED25519_SIGNATURE_BYTES * 2);
    manifest.signature.value = signing_key()?.sign_provider_manifest_v2(manifest)?;
    Ok(())
}

fn port(registry: &TypeRegistry, fixture: &PortFixture) -> Result<PluginPort, Box<dyn Error>> {
    let type_id = registry.resolve(&fixture.source_type)?.clone();
    let family = registry.family(&type_id)?;
    Ok(PluginPort {
        id: fixture.id.clone(),
        name: fixture.id.clone(),
        direction: match fixture.direction.as_str() {
            "input" => PortDirection::Input,
            "output" => PortDirection::Output,
            _ => return Err("invalid fixture port direction".into()),
        },
        type_id,
        cardinality: match fixture.cardinality.as_str() {
            "singular" => PortCardinality::Singular,
            "list" => PortCardinality::List,
            _ => return Err("invalid fixture port cardinality".into()),
        },
        presence: match fixture.presence.as_str() {
            "required" => PortPresence::Required,
            "optional" => PortPresence::Optional,
            "hidden" => PortPresence::Hidden,
            _ => return Err("invalid fixture port presence".into()),
        },
        hidden: fixture.presence == "hidden",
        lazy: fixture.cardinality == "list" && fixture.direction == "input",
        default: None,
        serialization: match family {
            ValueFamily::Scalar => PortSerialization::Inline,
            ValueFamily::Tensor | ValueFamily::Model => PortSerialization::Handle,
            ValueFamily::Artifact => PortSerialization::ArtifactReference,
        },
        accepted_legacy_names: Vec::new(),
    })
}

fn capability(fixture: &CapabilityFixture) -> Result<CapabilityRequest, Box<dyn Error>> {
    let kind = match fixture.kind.as_str() {
        "network_provider" => CapabilityKind::NetworkProvider,
        "secret" => CapabilityKind::Secret,
        "provider_upload" => CapabilityKind::ProviderUpload,
        "provider_cost" => CapabilityKind::ProviderCost,
        _ => return Err("invalid fixture capability kind".into()),
    };
    Ok(CapabilityRequest {
        kind,
        scope: fixture.scope.clone(),
        quota: CapabilityQuota {
            maximum_operations: fixture.maximum_operations,
            maximum_request_bytes: fixture.maximum_request_bytes,
            maximum_response_bytes: fixture.maximum_response_bytes,
            maximum_total_bytes: fixture.maximum_total_bytes,
            maximum_handles: fixture.maximum_handles,
            timeout_milliseconds: fixture.timeout_milliseconds,
        },
    })
}

fn streaming_contract(
    fixture: &StreamingFixture,
) -> Result<ProviderStreamingContractV2, Box<dyn Error>> {
    Ok(ProviderStreamingContractV2 {
        methods: fixture
            .methods
            .iter()
            .map(|method| provider_method(method))
            .collect::<Result<Vec<_>, _>>()?,
        maximum_headers: fixture.maximum_headers,
        maximum_header_bytes: fixture.maximum_header_bytes,
        maximum_request_body_bytes: fixture.maximum_request_body_bytes,
        maximum_response_body_bytes: fixture.maximum_response_body_bytes,
        maximum_chunk_bytes: fixture.maximum_chunk_bytes,
        maximum_ndjson_line_bytes: fixture.maximum_ndjson_line_bytes,
        maximum_wait_milliseconds: fixture.maximum_wait_milliseconds,
        maximum_uploads: fixture.maximum_uploads,
        maximum_upload_body_bytes: fixture.maximum_upload_body_bytes,
        maximum_cost_requests: fixture.maximum_cost_requests,
        maximum_progress_total: fixture.maximum_progress_total,
        uploads: fixture.uploads,
        cost_requests: fixture.cost_requests,
    })
}

fn provider_method(method: &str) -> Result<ProviderHttpMethodV2, Box<dyn Error>> {
    Ok(match method {
        "DELETE" => ProviderHttpMethodV2::Delete,
        "GET" => ProviderHttpMethodV2::Get,
        "HEAD" => ProviderHttpMethodV2::Head,
        "OPTIONS" => ProviderHttpMethodV2::Options,
        "PATCH" => ProviderHttpMethodV2::Patch,
        "POST" => ProviderHttpMethodV2::Post,
        "PUT" => ProviderHttpMethodV2::Put,
        _ => return Err("invalid fixture HTTP method".into()),
    })
}

fn worker_method(method: &str) -> Result<WorkerProviderHttpMethod, Box<dyn Error>> {
    Ok(match provider_method(method)? {
        ProviderHttpMethodV2::Delete => WorkerProviderHttpMethod::Delete,
        ProviderHttpMethodV2::Get => WorkerProviderHttpMethod::Get,
        ProviderHttpMethodV2::Head => WorkerProviderHttpMethod::Head,
        ProviderHttpMethodV2::Options => WorkerProviderHttpMethod::Options,
        ProviderHttpMethodV2::Patch => WorkerProviderHttpMethod::Patch,
        ProviderHttpMethodV2::Post => WorkerProviderHttpMethod::Post,
        ProviderHttpMethodV2::Put => WorkerProviderHttpMethod::Put,
    })
}

fn worker_contract(contract: &ProviderStreamingContractV2) -> WorkerProviderStreamingContract {
    WorkerProviderStreamingContract {
        methods: contract
            .methods
            .iter()
            .map(|method| match method {
                ProviderHttpMethodV2::Delete => WorkerProviderHttpMethod::Delete,
                ProviderHttpMethodV2::Get => WorkerProviderHttpMethod::Get,
                ProviderHttpMethodV2::Head => WorkerProviderHttpMethod::Head,
                ProviderHttpMethodV2::Options => WorkerProviderHttpMethod::Options,
                ProviderHttpMethodV2::Patch => WorkerProviderHttpMethod::Patch,
                ProviderHttpMethodV2::Post => WorkerProviderHttpMethod::Post,
                ProviderHttpMethodV2::Put => WorkerProviderHttpMethod::Put,
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

fn invocation_inputs(
    fixture: &ProviderFixture,
    invocation: &InvocationFixture,
    input_fault: Option<&InputFault>,
) -> Result<(InvocationInputs, Vec<(String, usize, Vec<u8>)>), Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let mut inputs = InvocationInputs::default();
    let mut abi = Vec::new();
    for fixture_input in &invocation.inputs {
        let mut values = Vec::with_capacity(fixture_input.values.len());
        for (index, value_fixture) in fixture_input.values.iter().enumerate() {
            let value = if index == 0
                && matches!(input_fault, Some(InputFault::OversizedString { port_id, .. }) if port_id == &fixture_input.port_id)
            {
                PluginValue::scalar(
                    registry.resolve(&fixture_input.source_type)?.clone(),
                    ScalarValue::String(
                        input_fault
                            .and_then(|fault| match fault {
                                InputFault::OversizedString { value, .. } => Some(value.clone()),
                                _ => None,
                            })
                            .ok_or("input override disappeared")?,
                    ),
                    &registry,
                )?
            } else {
                fixture_value(&registry, &fixture_input.source_type, value_fixture)?
            };
            abi.push((fixture_input.port_id.clone(), index, value.abi_bytes()?));
            values.push(value);
        }
        inputs.set_present(&fixture_input.port_id, values);
        for media in &fixture_input.media {
            if matches!(input_fault, Some(InputFault::MissingMedia)) {
                continue;
            }
            let (_, _, source_abi) = abi
                .iter()
                .find(|(port, index, _)| {
                    port == &fixture_input.port_id && *index == media.index as usize
                })
                .ok_or("fixture media references an unknown input value")?;
            let mut bytes = decode_hex(&media.bytes_hex)?;
            if matches!(input_fault, Some(InputFault::OversizedMedia)) {
                bytes = vec![0; 1024 * 1024 + 1];
            }
            let media_type = if matches!(input_fault, Some(InputFault::MalformedMedia)) {
                "not a media type"
            } else {
                &media.media_type
            };
            let source_digest = if matches!(input_fault, Some(InputFault::SourceDigestMismatch)) {
                "0".repeat(64)
            } else {
                encode_hex(&Sha256::digest(source_abi))
            };
            let content_digest = if matches!(input_fault, Some(InputFault::ContentDigestMismatch)) {
                "0".repeat(64)
            } else if matches!(input_fault, Some(InputFault::OversizedMedia)) {
                encode_hex(&Sha256::digest(&bytes))
            } else {
                media.content_sha256.clone()
            };
            inputs.register_media_batch_item(
                &fixture_input.port_id,
                media.index,
                media.batch_index,
                InvocationMediaContent::new(media_type, source_digest, content_digest, bytes),
            )?;
        }
    }
    let node = fixture
        .nodes
        .iter()
        .find(|node| node.id == invocation.node_id)
        .ok_or("invocation node is missing")?;
    for port in &node.ports {
        if port.direction == "input"
            && !invocation
                .inputs
                .iter()
                .any(|input| input.port_id == port.id)
        {
            inputs.set_absent(&port.id);
        }
    }
    Ok((inputs, abi))
}

fn fixture_value(
    registry: &TypeRegistry,
    source_type: &str,
    fixture: &ValueFixture,
) -> Result<PluginValue, Box<dyn Error>> {
    let type_id = registry.resolve(source_type)?.clone();
    Ok(match fixture {
        ValueFixture::String { value } => {
            PluginValue::scalar(type_id, ScalarValue::String(value.clone()), registry)?
        }
        ValueFixture::Integer { value } => {
            PluginValue::scalar(type_id, ScalarValue::Integer(*value), registry)?
        }
        ValueFixture::Float { value } => {
            PluginValue::scalar(type_id, ScalarValue::Float(*value), registry)?
        }
        ValueFixture::Boolean { value } => {
            PluginValue::scalar(type_id, ScalarValue::Boolean(*value), registry)?
        }
        ValueFixture::Record { value } => PluginValue::scalar(
            type_id,
            ScalarValue::Record(
                value
                    .iter()
                    .map(|(name, value)| Ok((name.clone(), fixture_scalar_value(value)?)))
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
            ),
            registry,
        )?,
        ValueFixture::Tensor {
            shape,
            dtype,
            device,
            stream,
            byte_length,
            digest,
        } => {
            let dtype = match dtype.as_str() {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "bf16" => DType::Bf16,
                "u8" => DType::U8,
                _ => return Err("invalid fixture tensor dtype".into()),
            };
            let device = DeviceId::from_source_device(device)?;
            let stream = StreamId::new(*stream);
            let descriptor = TensorDescriptor::contiguous(shape.clone(), dtype, device, stream)?;
            PluginValue::tensor(
                type_id,
                TensorValue::new(descriptor, *byte_length, digest.clone())?,
                registry,
            )?
        }
    })
}

fn fixture_scalar_value(fixture: &ValueFixture) -> Result<ScalarValue, Box<dyn Error>> {
    Ok(match fixture {
        ValueFixture::String { value } => ScalarValue::String(value.clone()),
        ValueFixture::Integer { value } => ScalarValue::Integer(*value),
        ValueFixture::Float { value } => ScalarValue::Float(*value),
        ValueFixture::Boolean { value } => ScalarValue::Boolean(*value),
        ValueFixture::Record { value } => ScalarValue::Record(
            value
                .iter()
                .map(|(name, value)| Ok((name.clone(), fixture_scalar_value(value)?)))
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
        ),
        ValueFixture::Tensor { .. } => {
            return Err("tensor cannot be nested in fixture record".into());
        }
    })
}

fn worker_headers(headers: &[HeaderFixture]) -> Vec<WorkerProviderHeader> {
    headers
        .iter()
        .map(|header| WorkerProviderHeader {
            name: header.name.clone(),
            value: header.value.clone(),
        })
        .collect()
}

fn validate_fixture(fixture: &ProviderFixture) -> Result<(), Box<dyn Error>> {
    if fixture.namespace.len() > 256
        || fixture.nodes.is_empty()
        || fixture.nodes.len() > 64
        || fixture.bindings.is_empty()
        || fixture.bindings.len() > 256
        || fixture.capabilities.is_empty()
        || fixture.capabilities.len() > 16
        || fixture.invocations.is_empty()
        || fixture.invocations.len() > 64
        || fixture.routes.len() > 64
        || fixture.streaming.maximum_chunk_bytes == 0
        || fixture.streaming.maximum_response_body_bytes == 0
    {
        return Err("provider-v2 fixture exceeds its structural bounds".into());
    }
    for route in &fixture.routes {
        if route.id.is_empty()
            || route.id.len() > 256
            || route.path.is_empty()
            || route.path.len() > 4096
            || provider_method(&route.method).is_err()
            || route.maximum_request_bytes == 0
            || route.maximum_response_bytes == 0
        {
            return Err("provider-v2 fixture route exceeds its structural bounds".into());
        }
    }
    for node in &fixture.nodes {
        if node.id.len() > 256 || node.ports.is_empty() || node.ports.len() > 64 {
            return Err("provider-v2 fixture node exceeds its structural bounds".into());
        }
        if fixture
            .bindings
            .iter()
            .all(|binding| binding.node_id != node.id)
            || fixture
                .invocations
                .iter()
                .all(|invocation| invocation.node_id != node.id)
        {
            return Err("provider-v2 fixture node lacks a binding or invocation".into());
        }
    }
    for binding in &fixture.bindings {
        if fixture.nodes.iter().all(|node| node.id != binding.node_id) {
            return Err("provider-v2 fixture binding references an unknown node".into());
        }
    }
    for invocation in &fixture.invocations {
        let node = fixture
            .nodes
            .iter()
            .find(|node| node.id == invocation.node_id)
            .ok_or("provider-v2 fixture invocation references an unknown node")?;
        if invocation.script.is_empty()
            || invocation.script.len() > 256
            || invocation.inputs.len() > 64
            || invocation.outputs.len() > 64
        {
            return Err("provider-v2 invocation exceeds its structural bounds".into());
        }
        let declared_outputs = node
            .ports
            .iter()
            .filter(|port| port.direction == "output")
            .map(|port| port.id.as_str())
            .collect::<Vec<_>>();
        let expected_outputs = invocation
            .outputs
            .iter()
            .map(|output| output.port_id.as_str())
            .collect::<Vec<_>>();
        if expected_outputs != declared_outputs {
            return Err("provider-v2 invocation output set is not exact".into());
        }
        for output in &invocation.outputs {
            let port = node
                .ports
                .iter()
                .find(|port| port.id == output.port_id && port.direction == "output")
                .ok_or("provider-v2 invocation output references an unknown port")?;
            if output.source_type != port.source_type {
                return Err("provider-v2 invocation output does not match its port".into());
            }
        }
        for input in &invocation.inputs {
            let port = node
                .ports
                .iter()
                .find(|port| port.id == input.port_id && port.direction == "input")
                .ok_or("provider-v2 invocation input references an unknown port")?;
            if input.source_type != port.source_type
                || input.values.is_empty()
                || input.values.len() > 256
                || (port.cardinality == "singular" && input.values.len() != 1)
            {
                return Err("provider-v2 invocation input does not match its port".into());
            }
            for media in &input.media {
                if media.index as usize >= input.values.len()
                    || media.media_type.len() > 128
                    || decode_hex(&media.bytes_hex).is_err()
                    || media.content_sha256.len() != 64
                {
                    return Err("provider-v2 invocation media does not match its input".into());
                }
            }
        }
        for port in node
            .ports
            .iter()
            .filter(|port| port.direction == "input" && port.presence == "required")
        {
            if invocation
                .inputs
                .iter()
                .all(|input| input.port_id != port.id)
            {
                return Err("provider-v2 invocation omits a required input".into());
            }
        }
    }
    Ok(())
}

fn trust_policy() -> Result<PluginTrustPolicy, Box<dyn Error>> {
    let signing_key = signing_key()?;
    Ok(PluginTrustPolicy::new([PluginVerificationKey::new(
        KEY_ID,
        signing_key.verification_key_bytes()?,
    )?])?)
}

fn signing_key() -> Result<PluginSigningKey, sim_plugin_sdk::PluginContractError> {
    PluginSigningKey::new(KEY_ID, KEY)
}

fn permission_policy(manifest: &PluginManifest) -> Result<PermissionPolicy, Box<dyn Error>> {
    let capabilities = manifest
        .capabilities
        .iter()
        .map(Capability::from_plugin_request)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PermissionPolicy::new(
        "task434-profile",
        [PermissionGrant::new(
            "task434-profile",
            manifest.identifier.clone(),
            CapabilitySet::new(capabilities),
            "task434-fixture",
        )?],
    )?)
}

fn discover_fixtures() -> Result<Vec<(PathBuf, ProviderFixture)>, Box<dyn Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root = repository.join("crates/sim_test_support/fixtures/providers");
    let provider_filter = env::var_os("SIM_PROVIDER_V2_FIXTURE_FILTER");
    let mut fixture_paths = Vec::new();
    for provider in fs::read_dir(&root)? {
        let provider = provider?;
        if provider.file_type()?.is_dir()
            && provider_filter
                .as_deref()
                .is_none_or(|filter| provider.file_name() == filter)
        {
            for fixture in fs::read_dir(provider.path())? {
                let fixture = fixture?;
                let name = fixture.file_name();
                let name = name.to_string_lossy();
                if fixture.file_type()?.is_file()
                    && name.starts_with("provider_v2")
                    && name.ends_with("_host.fixture")
                {
                    fixture_paths.push(fixture.path());
                }
            }
        }
    }
    fixture_paths.sort();
    fixture_paths
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path)?;
            if bytes.is_empty() || bytes.len() > 1024 * 1024 {
                return Err("provider-v2 host fixture exceeds its byte bound".into());
            }
            let fixture = serde_json::from_slice(&bytes)?;
            Ok((path, fixture))
        })
        .collect()
}

fn resolve_fixture_path(descriptor: &Path, configured: &str) -> Result<PathBuf, Box<dyn Error>> {
    if configured.is_empty() || configured.len() > 4096 || Path::new(configured).is_absolute() {
        return Err("invalid provider-v2 fixture path".into());
    }
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = repository.join(configured);
    let canonical_repository = repository.canonicalize()?;
    let canonical_path = path.canonicalize()?;
    if !canonical_path.starts_with(canonical_repository) {
        return Err(format!("fixture path escaped {}", descriptor.display()).into());
    }
    Ok(canonical_path)
}

fn read_component(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.starts_with(b"\0asm") {
        return Ok(bytes);
    }
    decode_base64(std::str::from_utf8(&bytes)?)
}

fn decode_base64(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let encoded = value
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    if encoded.is_empty() || encoded.len() % 4 != 0 {
        return Err("invalid component base64".into());
    }
    let mut decoded = Vec::with_capacity(encoded.len() / 4 * 3);
    for chunk in encoded.chunks_exact(4) {
        let &[first, second, third, fourth] = chunk else {
            return Err("invalid component base64 chunk".into());
        };
        let first = decode_base64_character(first)?;
        let second = decode_base64_character(second)?;
        let third = decode_base64_character(third)?;
        let fourth = decode_base64_character(fourth)?;
        decoded.push((first << 2) | (second >> 4));
        if chunk[2] != b'=' {
            decoded.push((second << 4) | (third >> 2));
        }
        if chunk[3] != b'=' {
            decoded.push((third << 6) | fourth);
        }
    }
    Ok(decoded)
}

fn decode_base64_character(byte: u8) -> Result<u8, Box<dyn Error>> {
    Ok(match byte {
        b'A'..=b'Z' => byte - b'A',
        b'a'..=b'z' => byte - b'a' + 26,
        b'0'..=b'9' => byte - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        b'=' => 0,
        _ => return Err("invalid component base64 character".into()),
    })
}

fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    if !value.len().is_multiple_of(2) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("invalid fixture hex".into());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair)?;
            Ok(u8::from_str_radix(text, 16)?)
        })
        .collect()
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}
mod hitpaw_image_input_probe {
    use super::*;
    use sim_nodes::{
        NativeNodeComputeSession, NativeNodeContext, NativeNodeServiceIdentity, NativeNodeServices,
        NativePrimitive, NativeStoredPayload, NativeValue,
    };
    use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
    use sim_tensor::{CpuWorkspaceAuthority, ImageTensor, NativeTensorPayload, NativeTensorRole};
    use sim_types::{AttemptId, NodeId, PromptId};

    struct ProbeControls {
        mode: ResponseMode,
        services: Arc<dyn PluginCapabilityServices>,
        cancellation: CancellationToken,
    }

    struct SecretQueryFaultServices {
        fixture: FixtureCapabilityServices,
        identifier: String,
        error: InvocationError,
        calls: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl SecretQueryFaultServices {
        fn record(&self, identifier: &str) -> Result<(), InvocationError> {
            self.calls
                .lock()
                .map_err(|error| InvocationError::HostFailure(error.to_string()))?
                .push(identifier.to_owned());
            Ok(())
        }
    }

    impl PluginCapabilityServices for SecretQueryFaultServices {
        fn read_asset(
            &self,
            identity: &AssetIdentity,
            context: &CapabilityServiceContext,
        ) -> Result<Vec<u8>, InvocationError> {
            self.fixture.read_asset(identity, context)
        }

        fn call_provider(
            &self,
            provider: &str,
            endpoint: &str,
            body: &[u8],
            secret_id: Option<&SecretId>,
        ) -> Result<Vec<u8>, InvocationError> {
            self.fixture
                .call_provider(provider, endpoint, body, secret_id)
        }

        fn secret_exists(&self, identifier: &str) -> Result<bool, InvocationError> {
            self.record(identifier)?;
            if identifier == self.identifier {
                Err(self.error.clone())
            } else {
                self.fixture.secret_exists(identifier)
            }
        }

        fn secret_exists_with_context(
            &self,
            identifier: &str,
            context: &CapabilityServiceContext,
        ) -> Result<bool, InvocationError> {
            context.check_active()?;
            if identifier == self.identifier && self.error == InvocationError::Cancelled {
                self.record(identifier)?;
                if !context.cancellation().cancel() {
                    return Err(InvocationError::HostFailure(
                        "secret query cancellation was already armed".to_owned(),
                    ));
                }
                context.check_active()?;
                return Err(InvocationError::HostFailure(
                    "secret query failed to observe cancellation".to_owned(),
                ));
            }
            let present = self.secret_exists(identifier)?;
            context.check_active()?;
            Ok(present)
        }

        fn clock_milliseconds(&self, clock: &str) -> Result<u64, InvocationError> {
            self.fixture.clock_milliseconds(clock)
        }

        fn random_bytes(&self, stream: &str, length: u32) -> Result<Vec<u8>, InvocationError> {
            self.fixture.random_bytes(stream, length)
        }

        fn open_model(&self, identifier: &str) -> Result<ModelValue, InvocationError> {
            self.fixture.open_model(identifier)
        }

        fn sanitize_log(&self, level: &str, message: &str) -> Result<String, InvocationError> {
            self.fixture.sanitize_log(level, message)
        }
    }

    fn fixture() -> Result<ProviderFixture, Box<dyn Error>> {
        let registry = generated_native_node_registry_projection(None)?;
        let digest = registry
            .provider_binding_contract_sha256(
                "HitPawGeneralImageEnhance",
                "zed:sim-provider-transport@1",
                "zed:sim-provider-materializer@1",
            )?
            .ok_or("HitPaw canonical contract is unavailable")?;
        let mut binding = ProviderBindingSet {
            schema_version: 1,
            implementation_namespace: "zed.sim.provider.hitpaw".to_owned(),
            bindings_sha256: "0".repeat(64),
            bindings: vec![ProviderBindingClaim {
                feature_id: "COMFY-NODE-0226".to_owned(),
                node_id: "HitPawGeneralImageEnhance".to_owned(),
                contract_sha256: digest.clone(),
                transport_schema: "zed:sim-provider-transport@1".parse()?,
                materializer_schema: "zed:sim-provider-materializer@1".parse()?,
            }],
        };
        binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
        Ok(serde_json::from_value(serde_json::json!({
            "component":"crates/sim_test_support/fixtures/providers/hitpaw/provider_v2_image_input_probe_component",
            "malformed_component":"", "oversized_component":"",
            "namespace":"zed.sim.provider.hitpaw",
            "provenance_source":"fixture://hitpaw-image-input-admission-probe",
            "provenance_publisher":"HitPaw image input admission probe",
            "bindings_sha256":binding.bindings_sha256,
            "bindings":[{"feature_id":"COMFY-NODE-0226","node_id":"HitPawGeneralImageEnhance",
                "contract_sha256":digest,"transport_schema":"zed:sim-provider-transport@1",
                "materializer_schema":"zed:sim-provider-materializer@1"}],
            "present_secrets":["sim_org.auth_token"],
            "nodes":[{"id":"HitPawGeneralImageEnhance","display_name":"HitPaw General Image Enhance",
                "category":"partner/image/HitPaw","ports":[
                    {"id":"model","source_type":"COMBO","direction":"input","presence":"required","cardinality":"singular"},
                    {"id":"image","source_type":"IMAGE","direction":"input","presence":"required","cardinality":"singular"},
                    {"id":"upscale_factor","source_type":"COMBO","direction":"input","presence":"required","cardinality":"singular"},
                    {"id":"auto_downscale","source_type":"BOOLEAN","direction":"input","presence":"required","cardinality":"singular"},
                    {"id":"output_0","source_type":"IMAGE","direction":"output","presence":"required","cardinality":"singular"}
                ]}],
            "capabilities":[
                {"kind":"network_provider","scope":"zed.sim.provider.hitpaw|https://api.comfy.org/proxy/hitpaw/api/photo-enhancer","maximum_operations":1,"maximum_request_bytes":1048576,"maximum_response_bytes":16777216,"maximum_total_bytes":17825792,"maximum_handles":1,"timeout_milliseconds":60000},
                {"kind":"secret","scope":"sim_org.auth_token","maximum_operations":1,"maximum_request_bytes":1048576,"maximum_response_bytes":16777216,"maximum_total_bytes":17825792,"maximum_handles":1,"timeout_milliseconds":60000},
                {"kind":"secret","scope":"sim_org.api_key","maximum_operations":1,"maximum_request_bytes":1048576,"maximum_response_bytes":16777216,"maximum_total_bytes":17825792,"maximum_handles":1,"timeout_milliseconds":60000}
            ],
            "routes":[{"id":"COMFY-API-EXT-0049","method":"POST","path":"/proxy/hitpaw/api/photo-enhancer","maximum_request_bytes":1048576,"maximum_response_bytes":16777216}],
            "streaming":{"methods":["POST"],"maximum_headers":8,"maximum_header_bytes":16384,
                "maximum_request_body_bytes":1048576,"maximum_response_body_bytes":16777216,
                "maximum_chunk_bytes":65536,"maximum_ndjson_line_bytes":65536,"maximum_wait_milliseconds":1000,
                "maximum_uploads":0,"maximum_upload_body_bytes":0,"maximum_cost_requests":0,
                "maximum_progress_total":100,"uploads":false,"cost_requests":false},
            "invocations":[{"node_id":"HitPawGeneralImageEnhance","inputs":[],"outputs":[],"receipt_hex":"",
                "script":[{"operation":"start","endpoint":"https://api.comfy.org/proxy/hitpaw/api/photo-enhancer",
                    "secret_id":"sim_org.auth_token","method":"POST","declared_body_bytes":null,
                    "response_slot":1,"headers":[{"name":"Accept","value":"application/json"},
                        {"name":"Authorization","value":"host-capability:auth_token_sim_org"},
                        {"name":"Comfy-Env","value":"host"},{"name":"Comfy-Usage-Source","value":"comfyui-api"},
                        {"name":"Content-Type","value":"application/json"}]},
                    {"operation":"check_cancelled","slot":1}]}]
        }))?)
    }

    #[test]
    fn hitpaw_image_input_probe_binding_recipe_uses_canonical_source_contract()
    -> Result<(), Box<dyn Error>> {
        let fixture = fixture()?;
        let registry = generated_native_node_registry_projection(None)?;
        let descriptor = registry
            .descriptor("HitPawGeneralImageEnhance")
            .ok_or("missing descriptor")?;
        assert_eq!(
            descriptor
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            ["model", "image", "upscale_factor", "auto_downscale"]
        );
        assert_eq!(
            descriptor.outputs.first().ok_or("missing output")?.name,
            "output_0"
        );
        assert_eq!(descriptor.outputs.len(), 1);
        assert_eq!(fixture.namespace, "zed.sim.provider.hitpaw");
        assert_eq!(fixture.bindings.len(), 1);
        assert_eq!(fixture.bindings[0].feature_id, "COMFY-NODE-0226");
        assert_eq!(fixture.bindings[0].node_id, "HitPawGeneralImageEnhance");
        assert_eq!(
            fixture.bindings[0].contract_sha256,
            registry
                .provider_binding_contract_sha256(
                    "HitPawGeneralImageEnhance",
                    "zed:sim-provider-transport@1",
                    "zed:sim-provider-materializer@1",
                )?
                .ok_or("missing canonical contract")?
        );
        let manifest = signed_manifest("0".repeat(64), &fixture)?;
        let binding = manifest
            .manifest
            .provider_binding
            .as_ref()
            .ok_or("missing signed binding")?;
        assert_eq!(binding.implementation_namespace, fixture.namespace);
        assert_eq!(
            binding.bindings_sha256,
            binding.canonical_bindings_sha256()?
        );
        assert_eq!(binding.bindings_sha256, fixture.bindings_sha256);
        assert!(!manifest.streaming.uploads);
        assert!(!manifest.streaming.cost_requests);
        assert_eq!(manifest.streaming.maximum_uploads, 0);
        assert_eq!(manifest.streaming.maximum_upload_body_bytes, 0);
        assert_eq!(manifest.streaming.maximum_cost_requests, 0);
        let image_port = manifest
            .manifest
            .nodes
            .first()
            .ok_or("missing signed node")?
            .ports
            .iter()
            .find(|port| port.id == "image")
            .ok_or("missing image port")?;
        assert_eq!(image_port.serialization, PortSerialization::Handle);
        assert_eq!(image_port.cardinality, PortCardinality::Singular);
        let recipe = serde_json::json!({"node_id":"HitPawGeneralImageEnhance",
            "feature_id":"COMFY-NODE-0226","namespace":fixture.namespace,
            "native_contract_sha256":fixture.bindings[0].contract_sha256,
            "bindings_sha256":fixture.bindings_sha256});
        if let Some(path) = env::var_os("SIM_HITPAW_BINDING_RECIPE_PATH") {
            let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .canonicalize()?;
            let path = PathBuf::from(path);
            if !path.starts_with(repository.join("target")) {
                return Err(
                    "binding recipe must remain under the ignored repository target".into(),
                );
            }
            fs::create_dir_all(path.parent().ok_or("missing recipe directory")?)?;
            fs::write(path, serde_json::to_vec_pretty(&recipe)?)?;
        }
        Ok(())
    }

    fn run_probe(
        host: &PluginHost,
        compiled: &sim_plugin_host::CompiledPlugin,
        manifest: &ProviderPluginManifestV2,
        authorization: &sim_runtime::ProviderManifestAuthorizationV2,
        fixture: &ProviderFixture,
        inputs: InvocationInputs,
        mode: ResponseMode,
    ) -> Result<
        (
            Result<sim_plugin_host::ProviderV2WorkerPendingInvocation, PluginError>,
            Vec<WorkerProviderStreamRequest>,
        ),
        Box<dyn Error>,
    > {
        run_probe_with_controls(
            host,
            compiled,
            manifest,
            authorization,
            fixture,
            inputs,
            ProbeControls {
                mode,
                services: Arc::new(FixtureCapabilityServices {
                    present_secrets: BTreeSet::from(["sim_org.auth_token".to_owned()]),
                }),
                cancellation: CancellationToken::default(),
            },
        )
    }

    fn run_probe_with_controls(
        host: &PluginHost,
        compiled: &sim_plugin_host::CompiledPlugin,
        manifest: &ProviderPluginManifestV2,
        authorization: &sim_runtime::ProviderManifestAuthorizationV2,
        fixture: &ProviderFixture,
        inputs: InvocationInputs,
        controls: ProbeControls,
    ) -> Result<
        (
            Result<sim_plugin_host::ProviderV2WorkerPendingInvocation, PluginError>,
            Vec<WorkerProviderStreamRequest>,
        ),
        Box<dyn Error>,
    > {
        let context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(0x443),
            session_generation: 1,
            invocation: 1,
            generation: 1,
        };
        let envelope: WorkerProviderV2InvocationEnvelope =
            serde_json::from_value(serde_json::json!({
                "context":context,"streaming_contract":worker_contract(&manifest.streaming),
                "provider_manifest_sha256":encode_hex(&Sha256::digest(manifest.signing_payload()?)),
                "proposal_generation":1,"finalization_nonce":vec![67_u8;32]
            }))?;
        let invocation = match host.begin_invocation(
            &manifest.manifest,
            authorization.authorization(),
            "HitPawGeneralImageEnhance",
            inputs,
            controls.services,
            controls.cancellation.clone(),
        ) {
            Ok(invocation) => invocation,
            Err(error) => return Ok((Err(error), Vec::new())),
        };
        let script = fixture
            .invocations
            .first()
            .ok_or("missing script")?
            .script
            .clone();
        let maximum_chunk_bytes = fixture.streaming.maximum_chunk_bytes;
        let (sender, receiver) = async_channel::bounded(1);
        let responder = thread::spawn(move || {
            respond_to_guest(
                receiver,
                context,
                &script,
                &[],
                maximum_chunk_bytes,
                controls.mode,
            )
        });
        let result = host.invoke_provider_component_v2_for_worker(
            compiled,
            invocation,
            "HitPawGeneralImageEnhance",
            envelope,
            controls.cancellation,
            sender,
        );
        let requests = responder
            .join()
            .map_err(|_| "input probe responder panicked")?;
        Ok((result, requests))
    }

    fn native_inputs(
        fixture: &ProviderFixture,
        batch: u64,
    ) -> Result<
        (
            InvocationInputs,
            NativeHandleStoreGeneration,
            NativeNodeContext,
        ),
        Box<dyn Error>,
    > {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(512 * 1024)?;
        let attempt = AttemptId(Uuid::from_u128(0x44300 + u128::from(batch)));
        let node_id = NodeId::from("native-hitpaw-image-input-probe");
        let generation = NativeHandleStoreGeneration::with_capacities(4, 1024 * 1024)?;
        let store = generation.handle_store_for_attempt(attempt);
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x44310 + u128::from(batch)),
                attempt,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x44320)),
            attempt,
            node_id,
            CancellationToken::default(),
            scratch,
            store.clone(),
            NativeNodeServices::checked(None, None, Some(compute))?,
        )?;
        let execution = context.compute_session()?.execution_context(&context)?;
        let pixels = match batch {
            1 => vec![0.0, 0.5, 1.0],
            2 => vec![0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
            _ => return Err("expected batch1 or2".into()),
        };
        let image = ImageTensor::from_f32(&backend, &execution, batch, 1, 1, 3, &pixels)?;
        let expected_png = context.prepare_provider_image_png(&image, 0, None)?;
        let handle = store.publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                NativeTensorRole::Image,
                image,
            )?)),
            &context.cancellation,
        )?;
        let native = BTreeMap::from([
            (
                "model".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String("generative".to_owned()),
                },
            ),
            ("image".to_owned(), NativeValue::Handle { value: handle }),
            (
                "upscale_factor".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(1),
                },
            ),
            (
                "auto_downscale".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Boolean(false),
                },
            ),
        ]);
        let manifest = signed_manifest("0".repeat(64), fixture)?;
        let node = manifest.manifest.nodes.first().ok_or("missing node")?;
        let registry = generated_native_node_registry_projection(None)?;
        let descriptor = registry
            .descriptor("HitPawGeneralImageEnhance")
            .ok_or("missing descriptor")?;
        let prepared = smol::block_on(sim_plugin_host::prepare_native_provider_invocation(
            node,
            descriptor,
            native,
            "profile-a",
            &context,
        ))?;
        let value = serde_json::to_value(&prepared.inputs)?;
        let media = value
            .pointer("/media_batches/image/0/0")
            .ok_or("missing first media")?;
        assert_eq!(
            value
                .pointer("/media_batches/image/0")
                .and_then(serde_json::Value::as_object)
                .ok_or("missing batch")?
                .len(),
            1
        );
        let bytes: Vec<u8> = serde_json::from_value(media["bytes"].clone())?;
        assert_eq!(bytes, expected_png);
        assert_eq!(media["content_sha256"], encode_hex(&Sha256::digest(&bytes)));
        let encoded = prepared
            .request
            .ports()
            .iter()
            .find(|port| port.port_id() == "image")
            .and_then(|port| port.values().first())
            .ok_or("missing source IMAGE value")?;
        assert_eq!(
            media["source_value_sha256"],
            encode_hex(&Sha256::digest(encoded.abi_bytes()))
        );
        let decoded = sim_media::decode_downloaded_image_rgba(
            &bytes,
            sim_media::PngLimits::default(),
            &context.cancellation,
        )?;
        assert_eq!(decoded.pixels_bhwc, vec![0.0, 127.0 / 255.0, 1.0, 1.0]);
        assert_eq!(generation.len(), 1);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok((prepared.inputs, generation, context))
    }

    #[test]
    fn compiled_hitpaw_input_probe_preserves_secret_query_cancellation_and_timeout()
    -> Result<(), Box<dyn Error>> {
        let fixture = fixture()?;
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let component = read_component(&repository.join(&fixture.component))?;
        let (host, manifest, authorization) = configured_host(&component, &fixture)?;
        let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
        let (inputs, generation, context) = native_inputs(&fixture, 2)?;
        let resident = generation.resident_bytes();
        for identifier in ["sim_org.auth_token", "sim_org.api_key"] {
            for error in [InvocationError::Cancelled, InvocationError::TimedOut] {
                let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
                let cancellation = CancellationToken::default();
                let expected = if error == InvocationError::Cancelled {
                    WorkerProviderStreamError::Cancelled
                } else {
                    WorkerProviderStreamError::TimedOut
                };
                let (result, requests) = run_probe_with_controls(
                    &host,
                    &compiled,
                    &manifest,
                    &authorization,
                    &fixture,
                    inputs.clone(),
                    ProbeControls {
                        mode: ResponseMode::Scripted,
                        services: Arc::new(SecretQueryFaultServices {
                            fixture: FixtureCapabilityServices {
                                present_secrets: BTreeSet::new(),
                            },
                            identifier: identifier.to_owned(),
                            error: error.clone(),
                            calls: calls.clone(),
                        }),
                        cancellation: cancellation.clone(),
                    },
                )?;
                assert!(
                    matches!(result, Err(PluginError::ProviderStreaming(actual)) if actual == expected)
                );
                let expected_calls = if identifier == "sim_org.auth_token" {
                    vec!["sim_org.auth_token"]
                } else {
                    vec!["sim_org.auth_token", "sim_org.api_key"]
                };
                assert_eq!(
                    *calls.lock().map_err(|error| error.to_string())?,
                    expected_calls
                );
                assert_eq!(
                    cancellation.is_cancelled(),
                    error == InvocationError::Cancelled
                );
                assert!(requests.is_empty());
                assert_eq!(generation.len(), 1);
                assert_eq!(generation.resident_bytes(), resident);
                assert_eq!(context.scratch.in_use_bytes(), 0);
            }
        }
        Ok(())
    }

    #[test]
    fn native_hitpaw_first_image_reaches_compiled_input_probe_without_body_or_result()
    -> Result<(), Box<dyn Error>> {
        let fixture = fixture()?;
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let component = read_component(&repository.join(&fixture.component))?;
        let (host, manifest, authorization) = configured_host(&component, &fixture)?;
        let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
        for batch in [1, 2] {
            let (inputs, generation, context) = native_inputs(&fixture, batch)?;
            let resident = generation.resident_bytes();
            for mode in [ResponseMode::CancelAtCheck, ResponseMode::Scripted] {
                let (result, requests) = run_probe(
                    &host,
                    &compiled,
                    &manifest,
                    &authorization,
                    &fixture,
                    inputs.clone(),
                    mode,
                )?;
                let expected = if matches!(mode, ResponseMode::CancelAtCheck) {
                    WorkerProviderStreamError::Cancelled
                } else {
                    WorkerProviderStreamError::InvalidContract
                };
                assert!(
                    matches!(result,Err(PluginError::ProviderStreaming(error)) if error==expected)
                );
                let invocation_context = WorkerProviderInvocationContext {
                    session_id: Uuid::from_u128(0x443),
                    session_generation: 1,
                    invocation: 1,
                    generation: 1,
                };
                assert_eq!(
                    requests,
                    vec![
                        WorkerProviderStreamRequest::StartRequest {
                            context: invocation_context.clone(),
                            head: WorkerProviderRequestHead {
                                endpoint: "https://api.comfy.org/proxy/hitpaw/api/photo-enhancer"
                                    .to_owned(),
                                secret_id: Some("sim_org.auth_token".to_owned()),
                                method: WorkerProviderHttpMethod::Post,
                                headers: [
                                    ("Accept", "application/json"),
                                    ("Authorization", "host-capability:auth_token_sim_org"),
                                    ("Comfy-Env", "host"),
                                    ("Comfy-Usage-Source", "comfyui-api"),
                                    ("Content-Type", "application/json"),
                                ]
                                .into_iter()
                                .map(|(name, value)| WorkerProviderHeader {
                                    name: name.to_owned(),
                                    value: value.to_owned(),
                                })
                                .collect(),
                                declared_body_bytes: None,
                            },
                        },
                        WorkerProviderStreamRequest::CheckCancelled(stream_handle(
                            &invocation_context,
                            1
                        )),
                    ]
                );
                assert_eq!(generation.len(), 1);
                assert_eq!(generation.resident_bytes(), resident);
                assert_eq!(context.scratch.in_use_bytes(), 0);
            }
        }
        Ok(())
    }

    #[test]
    fn compiled_hitpaw_input_probe_rejects_missing_and_corrupt_media_before_stream()
    -> Result<(), Box<dyn Error>> {
        let fixture = fixture()?;
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let component = read_component(&repository.join(&fixture.component))?;
        let (host, manifest, authorization) = configured_host(&component, &fixture)?;
        let compiled = host.compile_provider_component_v2(&component, &manifest, &authorization)?;
        let (inputs, generation, context) = native_inputs(&fixture, 2)?;
        let original = serde_json::to_value(inputs)?;
        let resident = generation.resident_bytes();
        for fault in [
            "missing",
            "source_value_sha256",
            "content_sha256",
            "media_type",
            "signature",
        ] {
            let mut value = original.clone();
            if fault == "missing" {
                value
                    .as_object_mut()
                    .ok_or("invalid inputs")?
                    .insert("media_batches".to_owned(), serde_json::json!({}));
            } else {
                let media = value
                    .pointer_mut("/media_batches/image/0/0")
                    .and_then(serde_json::Value::as_object_mut)
                    .ok_or("missing media")?;
                match fault {
                    "source_value_sha256" | "content_sha256" => {
                        media.insert(fault.to_owned(), serde_json::Value::String("f".repeat(64)));
                    }
                    "media_type" => {
                        media.insert(
                            "media_type".to_owned(),
                            serde_json::Value::String("image/jpeg".to_owned()),
                        );
                    }
                    "signature" => {
                        let bytes = vec![1_u8, 2, 3];
                        media.insert("bytes".to_owned(), serde_json::to_value(&bytes)?);
                        media.insert(
                            "content_sha256".to_owned(),
                            serde_json::Value::String(encode_hex(&Sha256::digest(&bytes))),
                        );
                    }
                    _ => return Err("unknown media fault".into()),
                }
            }
            let inputs = serde_json::from_value(value)?;
            let (result, requests) = run_probe(
                &host,
                &compiled,
                &manifest,
                &authorization,
                &fixture,
                inputs,
                ResponseMode::Scripted,
            )?;
            match fault {
                "source_value_sha256" | "content_sha256" => assert!(matches!(
                    result,
                    Err(PluginError::Invocation(InvocationError::HostFailure(message)))
                        if message == "invalid plugin media content digest"
                )),
                _ => assert!(matches!(
                    result,
                    Err(PluginError::ProviderStreaming(
                        WorkerProviderStreamError::InvalidUpload
                    ))
                )),
            }
            assert!(requests.is_empty(), "fault {fault} reached the stream");
            assert_eq!(generation.len(), 1);
            assert_eq!(generation.resident_bytes(), resident);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        Ok(())
    }
}
