use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_media::{
    NativeAudioPayload, NativeMediaTextRenderer, PILLOW_DEFAULT_FONT_SHA256, PngLimits, TextRun,
    encode_png_frame, zero_padded_numeric_tag,
};
use sim_model::model_family::{NativeModelRecipeSourceForm, auraflow_recipe_fixture_state_schema};
use sim_model::{
    BackgroundRemovalFixtureMutation, DepthAnything3FixtureMutation, DepthAnything3FixtureProfile,
    MogeFixtureMutation, MogeFixtureProfile, NATIVE_UPSCALE_ADMITTED_ARCHITECTURE_COUNT,
    NATIVE_UPSCALE_ARCHITECTURE_COUNT, NATIVE_UPSCALE_CONTRACT_SHA256,
    NativeBackgroundRemovalError, NativeBackgroundRemovalResource, NativeDepthAnything3Invocation,
    NativeDepthAnything3ReferenceStrategy, NativeDepthAnything3ResizeMethod,
    NativeDepthAnything3Resource, NativeFrameInterpolationModel, NativeLatentUpscaleCheckpoint,
    NativeLatentUpscaleModelResource, NativeModelPayload, NativeModelPayloadError,
    NativeModelResourceIdentity, NativeModelResourceRole, NativeMogeInvocation, NativeMogeResource,
    NativeSdPoseHeatmapHead, NativeSdPoseModel, NativeSdPoseSd2Denoiser,
    NativeUpscaleContractError, NativeUpscaleModelError, NativeUpscaleModelResource,
    NativeUpscaleStateDictionaryLayout, NativeUpscaleUnavailableReason,
    SdPoseHeatmapHeadConfiguration, SdPoseSd2Configuration, VaeBoundaryKind, VaeKernelProfile,
    VaeLoaderConfiguration, compiled_native_upscale_contract,
    conditioning::{
        ConditioningControlBinding, ConditioningEntry, ConditioningEntryOptions,
        ConditioningIdentity, ConditioningMetadataKey, ConditioningMetadataMap,
        ConditioningMetadataValue, ConditioningSet, ConditioningValue,
    },
    deterministic_reduced_depth_anything_3_checkpoint, deterministic_reduced_moge_checkpoint,
    deterministic_reduced_test_checkpoint,
    generated_native_diffusion::{sd15_latent_format_identity, sd15_model_family_identity},
    mutate_reduced_depth_anything_3_checkpoint, mutate_reduced_moge_checkpoint,
    raft_large_exact_native, reduced_depth_anything_3_checkpoint_parity_for_fixture,
    sdpose_heatmap_head_weight_manifest, sdpose_sd2_weight_manifest,
    select_reduced_depth_anything_3_reference_for_fixture,
};
use sim_nodes::{
    NativeModelResourceRequest, NativePreparedEffectKind, NativeStoredModelPayload,
    NativeStructuredValue, built_in_source_schema,
};
use sim_runtime::{
    AttemptState, CompiledNode, CompiledPlan, InputBinding, NATIVE_IMAGE_REGISTRY_VERSION,
    NativeDiffusionProvider, NativeHandleKind, NativeHandleStoreError, NativeHandleStoreGeneration,
    NativeHandleType, NativeImageExecutor, NativeImageWorkerEvent, NativeImageWorkerPlan,
    NativeInputDescriptor, NativeModelSourceWorkerBinding, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeOpaqueHandle, NativePortCardinality,
    NativePreparedEffectRequest, NativePrimitive, NativePrimitiveType, NativeStoredPayload,
    NativeTypeUnion, NativeValue, NativeValueType, PersistedNativeAudioEvidence,
    PersistedNativeModelResourceEvidence, ProviderTransportResponse, RuntimeSupervisor,
    SupervisorPolicy, WorkerHealth, WorkerLaunchConfig, compile_generated_native_prompt,
    compile_native_diffusion_workflow, execute_reduced_fixture_native_model_resource_for_test,
    generated_native_frontend_descriptors, generated_native_node_registry_projection,
    graph_to_prompt, load_reduced_fixture_native_model_resource_for_test,
    native_image_registry_projection, task407_custom_sampler_inputs_fixture_descriptor_for_test,
    task410_audio_fixture_descriptor_for_test,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    ImageTensor, NativeTensorPayload, NativeTensorRole, RetryRngPolicy, RngStreamAddress, StreamId,
    Tensor, TensorBackend,
    generated_elementwise_or_runtime_operation_21::exp_with_context_exact_native,
    generated_neural_network_functional_01::pixel_shuffle_tensor_with_context_exact_native,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        InterpolateConfiguration, InterpolateMode, bislerp_tensor_with_context_exact_native,
        interpolate_tensor_with_context_exact_native,
        pixel_shuffle_nd_tensor_with_context_exact_native,
    },
};
use sim_test_support::NativeDiffusionFixture;
use sim_types::{
    AttemptId, NodeId, ProfileId, PromptId, WorkerId, WorkerMessage, WorkerModelSourceArtifact,
    WorkerModelSourceArtifactChunk, WorkerModelSourceChunk, WorkerModelSourceClosed,
    WorkerModelSourceError, WorkerModelSourceFormat, WorkerModelSourceManifest,
    WorkerModelSourceNestedStateDisposition, WorkerModelSourceOpened, WorkerModelSourceOperation,
    WorkerModelSourceParserLimits, WorkerModelSourceResponse, WorkerModelSourceTensor,
    WorkerSha256Digest,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;

const PROFILE_ID: ProfileId = ProfileId(Uuid::from_u128(0x3670));
const WORKFLOW_FIXTURE: &[u8] = include_bytes!("../fixtures/native_image/workflow.json");
const SDPOSE_FIXTURE_ARTIFACT: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const UPSCALE_MODEL_RESOURCE_FIXTURE: &str =
    include_str!("../fixtures/models/upscale-model-resource-foundation/contract.json");
const BACKGROUND_REMOVAL_RESOURCE_FIXTURE: &str =
    include_str!("../fixtures/models/background-removal-resource-foundation/manifest.json");
const BACKGROUND_REMOVAL_RESOURCE_ORACLE: &str =
    include_str!("../fixtures/models/background-removal-resource-foundation/oracle.json");
const BACKGROUND_REMOVAL_RESOURCE_PROVENANCE: &str =
    include_str!("../fixtures/models/background-removal-resource-foundation/provenance.json");
const BACKGROUND_REMOVAL_RESOURCE_GENERATOR: &str =
    include_str!("../fixtures/models/background-removal-resource-foundation/generate_oracle.py");
const DEPTH_ANYTHING_3_RESOURCE_FIXTURE: &str =
    include_str!("../fixtures/models/depth-anything-3-resource-foundation/manifest.json");
const DEPTH_ANYTHING_3_RESOURCE_ORACLE: &str = if cfg!(all(
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
)) {
    include_str!(
        "../fixtures/models/depth-anything-3-resource-foundation/oracle-linux-gnu-x86_64.json"
    )
} else {
    include_str!("../fixtures/models/depth-anything-3-resource-foundation/oracle.json")
};
const DEPTH_ANYTHING_3_RESOURCE_PROVENANCE: &str =
    include_str!("../fixtures/models/depth-anything-3-resource-foundation/provenance.json");
const DEPTH_ANYTHING_3_RESOURCE_GENERATOR: &str =
    include_str!("../fixtures/models/depth-anything-3-resource-foundation/generate_oracle.py");
const DEPTH_ANYTHING_3_RESOURCE_SOURCE_GRAPH: &str =
    include_str!("../fixtures/models/depth-anything-3-resource-foundation/source_graph.py");
const MOGE_RESOURCE_FIXTURE: &str =
    include_str!("../fixtures/models/moge-resource-foundation/manifest.json");
const MOGE_RESOURCE_ORACLE: &str =
    include_str!("../fixtures/models/moge-resource-foundation/oracle.json");
const MOGE_RESOURCE_PROVENANCE: &str =
    include_str!("../fixtures/models/moge-resource-foundation/provenance.json");
const MOGE_RESOURCE_GENERATOR: &str =
    include_str!("../fixtures/models/moge-resource-foundation/generate_oracle.py");
const MOGE_RESOURCE_SOURCE_GRAPH: &str =
    include_str!("../fixtures/models/moge-resource-foundation/source_graph.py");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpscaleModelResourceFixture {
    schema_version: u32,
    contract_sha256: String,
    architecture_count: usize,
    admitted_architecture_count: usize,
    cases: Vec<UpscaleModelResourceCase>,
    forbidden_substitutes: Vec<String>,
}

#[test]
fn task415_native_media_text_rendering_survives_owner_reconstruction() -> Result<(), Box<dyn Error>>
{
    let fixture = serde_json::from_str::<Value>(include_str!(
        "../fixtures/media/text-rendering/pillow-12.3.0-oracle.json"
    ))?;
    assert_eq!(
        fixture["font_sha256"],
        Value::String(PILLOW_DEFAULT_FONT_SHA256.to_owned())
    );
    assert_eq!(zero_padded_numeric_tag(1), "01");

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let background = [32_u8, 32, 32, 255].repeat(64 * 24);
    let runs = [TextRun {
        text: "01 café".to_owned(),
        x: 3,
        y: 2,
        font_size: 12,
        color: [240, 80, 40, 255],
    }];
    let render = |renderer: &NativeMediaTextRenderer| -> Result<Vec<u8>, Box<dyn Error>> {
        let image = renderer.render_over_rgba8(64, 24, &background, &runs, &backend, &context)?;
        Ok(image
            .as_f32_slice()?
            .iter()
            .map(|value| (value * 255.0).round() as u8)
            .collect())
    };
    let first = render(&NativeMediaTextRenderer::new()?)?;
    let replacement_worker_owner = NativeMediaTextRenderer::new()?;
    let recovered = render(&replacement_worker_owner)?;
    let expected = include_bytes!("../fixtures/media/text-rendering/pillow-12.3.0-64x24.rgb");
    assert_eq!(first, expected);
    assert_eq!(recovered, expected);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpscaleModelResourceCase {
    layout: String,
    state_keys: Vec<String>,
    architecture_id: String,
    diagnostic: String,
}

fn fixture_raw_bits(value: &serde_json::Value) -> Result<Vec<u32>, Box<dyn Error>> {
    fn collect(value: &serde_json::Value, output: &mut Vec<u32>) -> Result<(), Box<dyn Error>> {
        if let Some(array) = value.as_array() {
            for value in array {
                collect(value, output)?;
            }
            return Ok(());
        }
        output.push(u32::try_from(
            value.as_u64().ok_or("fixture bit is not an integer")?,
        )?);
        Ok(())
    }
    let mut output = Vec::new();
    collect(value, &mut output)?;
    Ok(output)
}

fn require_exact_fixture_bits(
    phase: &str,
    actual: &[u32],
    expected: &[u32],
) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{phase}: actual length {}, expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        return Err(
            format!("{phase}[{index}]: actual {actual:#010x}, expected {expected:#010x}").into(),
        );
    }
    Ok(())
}

fn require_fixture_ulp_bound(
    phase: &str,
    actual: &[u32],
    expected: &[u32],
    maximum_ulp: u32,
) -> Result<u32, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{phase}: actual length {}, expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let ordered = |bits: u32| {
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    };
    let mut observed = 0_u32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !f32::from_bits(actual).is_finite() || !f32::from_bits(expected).is_finite() {
            return Err(format!("{phase}[{index}]: non-finite ULP comparison").into());
        }
        let distance = ordered(actual).abs_diff(ordered(expected));
        observed = observed.max(distance);
        if distance > maximum_ulp {
            return Err(format!(
                "{phase}[{index}]: {distance} ULP exceeds {maximum_ulp}; actual {actual:#010x}, expected {expected:#010x}"
            )
            .into());
        }
    }
    Ok(observed)
}

fn sdpose_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let elements = shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(usize::try_from(*dimension).ok()?)
    });
    let elements = elements.ok_or("SDPose fixture tensor shape overflowed")?;
    let mut values = Vec::new();
    values.try_reserve_exact(elements)?;
    values.resize(elements, 0.0);
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        &values,
        DType::F32,
        backend.device(),
        context,
    )?)
}

fn reduced_sdpose_stored_payload() -> Result<NativeStoredPayload, Box<dyn Error>> {
    sdpose_stored_payload_fixture(
        SdPoseSd2Configuration::reduced_fixture(4, 3, 1, 1, 8, 8)?,
        SdPoseHeatmapHeadConfiguration::reduced_fixture(8, 8, 3)?,
    )
}

fn sdpose_stored_payload_fixture(
    denoiser_configuration: SdPoseSd2Configuration,
    head_configuration: SdPoseHeatmapHeadConfiguration,
) -> Result<NativeStoredPayload, Box<dyn Error>> {
    let workspace_bytes = 64 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(workspace_bytes)?,
        &cancellation,
    );
    let mut denoiser_weights = BTreeMap::new();
    for specification in sdpose_sd2_weight_manifest(&denoiser_configuration)? {
        denoiser_weights.insert(
            specification.key().to_owned(),
            sdpose_tensor(&backend, specification.shape(), &context)?,
        );
    }
    let denoiser = NativeSdPoseSd2Denoiser::from_reduced_fixture(
        denoiser_configuration,
        denoiser_weights,
        &cancellation,
    )?;
    let mut head_weights = BTreeMap::new();
    for specification in sdpose_heatmap_head_weight_manifest(&head_configuration)? {
        head_weights.insert(
            specification.key().to_owned(),
            sdpose_tensor(&backend, specification.shape(), &context)?,
        );
    }
    let head = NativeSdPoseHeatmapHead::from_reduced_fixture(
        head_configuration,
        head_weights,
        &cancellation,
    )?;
    let resource = Arc::new(NativeSdPoseModel::from_reduced_fixture(
        SDPOSE_FIXTURE_ARTIFACT.to_owned(),
        denoiser,
        head,
        &cancellation,
    )?);
    let model = Arc::new(NativeModelPayload::sdpose_model_test_fixture(resource)?);
    Ok(NativeStoredPayload::Model(Arc::new(
        NativeStoredModelPayload::model_resource(model)?,
    )))
}

fn reduced_frame_interpolation_stored_payload() -> Result<NativeStoredPayload, Box<dyn Error>> {
    let workspace_bytes = 16 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(workspace_bytes)?,
        &cancellation,
    );
    let resource = Arc::new(NativeFrameInterpolationModel::reduced_rife_test_fixture(
        &backend, &context,
    )?);
    let model = Arc::new(NativeModelPayload::frame_interpolation(resource)?);
    Ok(NativeStoredPayload::Model(Arc::new(
        NativeStoredModelPayload::model_resource(model)?,
    )))
}

#[test]
fn native_upscale_model_resource_is_closed_and_source_specific() -> Result<(), Box<dyn Error>> {
    let fixture: UpscaleModelResourceFixture =
        serde_json::from_str(UPSCALE_MODEL_RESOURCE_FIXTURE)?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.contract_sha256, NATIVE_UPSCALE_CONTRACT_SHA256);
    assert_eq!(
        fixture.architecture_count,
        NATIVE_UPSCALE_ARCHITECTURE_COUNT
    );
    assert_eq!(
        fixture.admitted_architecture_count,
        NATIVE_UPSCALE_ADMITTED_ARCHITECTURE_COUNT
    );
    assert_eq!(fixture.forbidden_substitutes.len(), 5);

    let cancellation = CancellationToken::default();
    let contract = compiled_native_upscale_contract()?;
    assert_eq!(contract.architectures().len(), fixture.architecture_count);
    assert_eq!(contract.admitted_architecture_count(), 0);
    for architecture in contract.architectures() {
        let error = match NativeUpscaleModelResource::checked(
            NativeUpscaleStateDictionaryLayout::Flat,
            architecture.detection_state_keys.iter(),
            &cancellation,
        ) {
            Err(error) => error,
            Ok(resource) => match resource {},
        };
        let unavailable = error
            .unavailable()
            .ok_or("architecture rejection lost its typed diagnostic")?;
        assert_eq!(unavailable.ordinal(), architecture.ordinal);
        assert_eq!(unavailable.architecture_id(), architecture.architecture_id);
        let expected_reason = if architecture.origin == "main" {
            NativeUpscaleUnavailableReason::MissingIndividualLicense
        } else {
            NativeUpscaleUnavailableReason::ReferenceOnlyExtraArchitecture
        };
        assert_eq!(unavailable.reason(), expected_reason);
        assert_eq!(unavailable.diagnostic(), architecture.license_disposition);
    }

    for fixture_case in fixture.cases {
        let layout = match fixture_case.layout.as_str() {
            "flat" => NativeUpscaleStateDictionaryLayout::Flat,
            "state_dict" => NativeUpscaleStateDictionaryLayout::StateDict,
            _ => return Err("unknown upscale fixture layout".into()),
        };
        let error = match NativeUpscaleModelResource::checked(
            layout,
            fixture_case.state_keys.iter(),
            &cancellation,
        ) {
            Err(error) => error,
            Ok(resource) => match resource {},
        };
        let NativeUpscaleModelError::Unavailable {
            architecture_id,
            reason,
            ..
        } = error
        else {
            return Err("upscale rejection lost its source-specific error".into());
        };
        assert_eq!(architecture_id, fixture_case.architecture_id);
        assert_eq!(reason.diagnostic(), fixture_case.diagnostic);
    }

    assert!(matches!(
        NativeUpscaleModelResource::checked(
            NativeUpscaleStateDictionaryLayout::Flat,
            ["not.an.upscale.architecture"],
            &cancellation,
        ),
        Err(NativeUpscaleModelError::Contract(
            NativeUpscaleContractError::NoArchitectureMatch
        ))
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        NativeUpscaleModelResource::checked(
            NativeUpscaleStateDictionaryLayout::Flat,
            ["body.0.weight", "body.1.weight"],
            &cancelled,
        ),
        Err(NativeUpscaleModelError::Contract(
            NativeUpscaleContractError::Cancelled
        ))
    ));

    let resource_source = include_str!("../../sim_model/src/upscale_model.rs");
    let payload_source = include_str!("../../sim_model/src/native_node_payload.rs");
    for forbidden in [
        "std::fs",
        "pyo3",
        "Python::",
        "image_resize",
        "generic_fallback",
        "model.safetensors",
    ] {
        assert!(!resource_source.contains(forbidden));
    }
    assert!(!payload_source.contains("NativeModelResource::UpscaleModel"));
    Ok(())
}

#[test]
fn native_latent_upscale_model_fixture_oracles_are_complete_and_consumed()
-> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or("sim_test_support has no repository root")?;
    let fixture_path = root.join(
        "crates/sim_test_support/fixtures/models/latent-upscale-model-resource-foundation/manifest.json",
    );
    let fixture: serde_json::Value = serde_json::from_str(&fs::read_to_string(&fixture_path)?)?;
    assert_eq!(
        fixture
            .get("oracle_domain")
            .and_then(serde_json::Value::as_str),
        Some("zed.sim.latent-upscale-independent-source-equations.v1")
    );
    let generator_sha = fixture
        .get("oracle_generator_sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or("latent-upscale generator SHA is missing")?;
    let actual_generator_sha = format!(
        "{:x}",
        Sha256::digest(fs::read(root.join(
            "crates/sim_test_support/src/bin/generate_latent_upscale_model_fixture.rs"
        ),)?)
    );
    assert_eq!(generator_sha, actual_generator_sha);

    let cases = fixture
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or("latent-upscale fixture cases are missing")?;
    let identifiers = cases
        .iter()
        .filter_map(|case| case.get("id").and_then(serde_json::Value::as_str))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        identifiers,
        BTreeSet::from([
            "720-integrated-residual-order",
            "1080-repeat-rms-shortcut-order",
            "pixel-shuffle-dimension-one",
            "pixel-shuffle-dimension-two",
            "pixel-shuffle-dimension-three",
            "rational-blur-center-delta",
            "nearest-exact-half-coordinate",
            "bislerp-edge-cases",
            "ltx-vae-statistics-order",
        ])
    );
    for case in cases {
        let identifier = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or("latent-upscale fixture case lacks an id")?;
        if identifier != "rational-blur-center-delta" {
            assert!(
                case.as_object()
                    .is_some_and(|object| { object.keys().any(|key| key.ends_with("_bits")) }),
                "latent-upscale case {identifier} lacks a raw-bit oracle"
            );
        }
    }
    let expected_720_bits = cases
        .iter()
        .find(|case| {
            case.get("id").and_then(serde_json::Value::as_str)
                == Some("720-integrated-residual-order")
        })
        .and_then(|case| case.get("expected_bits"))
        .and_then(serde_json::Value::as_array)
        .ok_or("720 raw oracle bits are missing")?
        .iter()
        .map(|value| value.as_u64().ok_or("720 raw oracle bit is invalid"))
        .collect::<Result<Vec<_>, _>>()?;
    let workspace_bytes = 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(workspace_bytes)?,
        &cancellation,
    );
    let tensor = |shape: &[u64], values: &[f32]| {
        tensor_from_f32_with_context_exact_native(
            &backend,
            shape,
            values,
            DType::F32,
            backend.device(),
            &context,
        )
    };
    let case = |identifier: &str| -> Result<&serde_json::Value, Box<dyn Error>> {
        cases
            .iter()
            .find(|case| case.get("id").and_then(serde_json::Value::as_str) == Some(identifier))
            .ok_or_else(|| format!("missing latent-upscale case {identifier}").into())
    };
    let tensor_bits = |tensor: &Tensor| -> Result<Vec<u32>, Box<dyn Error>> {
        Ok(sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend,
            tensor,
            &context,
        )?
        .iter()
        .map(|value| value.to_bits())
        .collect())
    };
    let mut identity = vec![0.0_f32; 27];
    identity[22] = 1.0;
    let mut ordered_state = Vec::new();
    for prefix in [
        "in_conv.conv",
        "blocks.0.block.0.conv",
        "blocks.0.block.2.conv",
        "blocks.0.block.4.conv",
        "out_conv.conv",
    ] {
        ordered_state.push((
            format!("{prefix}.weight"),
            tensor(&[1, 1, 3, 3, 3], &identity)?,
        ));
        ordered_state.push((format!("{prefix}.bias"), tensor(&[1], &[0.0])?));
    }
    let ordered_state_720 = ordered_state.clone();
    let resource = NativeLatentUpscaleModelResource::from_checkpoint(
        NativeLatentUpscaleCheckpoint {
            artifact_sha256: "1".repeat(64),
            metadata: BTreeMap::new(),
            ordered_state,
            memory_budget_bytes: workspace_bytes,
        },
        &context,
    )?;
    let input = tensor(&[1, 1, 3, 1, 1], &[-1.0, 0.0, 1.0])?;
    let output = resource.invoke_hunyuan_720p(&backend, &input, &context)?;
    let actual =
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend, &output, &context,
        )?;
    assert_eq!(
        actual
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .collect::<Vec<_>>(),
        expected_720_bits
    );
    let mutated_720_input = resource.invoke_hunyuan_720p(
        &backend,
        &tensor(&[1, 1, 3, 1, 1], &[-1.0, 0.0, 2.0])?,
        &context,
    )?;
    assert_ne!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        tensor_bits(&mutated_720_input)?
    );

    let mut early_branch_state = ordered_state_720.clone();
    let mut late_branch_state = ordered_state_720;
    let mut half_identity = identity.clone();
    half_identity[22] = 0.5;
    for (state, key) in [
        (&mut early_branch_state, "blocks.0.block.0.conv.weight"),
        (&mut late_branch_state, "blocks.0.block.2.conv.weight"),
    ] {
        let (_, weight) = state
            .iter_mut()
            .find(|(candidate, _)| candidate == key)
            .ok_or("720 branch weight is missing")?;
        *weight = tensor(&[1, 1, 3, 3, 3], &half_identity)?;
    }
    let build_720 = |artifact: char, ordered_state| {
        NativeLatentUpscaleModelResource::from_checkpoint(
            NativeLatentUpscaleCheckpoint {
                artifact_sha256: artifact.to_string().repeat(64),
                metadata: BTreeMap::new(),
                ordered_state,
                memory_budget_bytes: workspace_bytes,
            },
            &context,
        )
    };
    let early_branch = build_720('4', early_branch_state)?;
    let late_branch = build_720('5', late_branch_state)?;
    assert_ne!(
        tensor_bits(&early_branch.invoke_hunyuan_720p(&backend, &input, &context)?)?,
        tensor_bits(&late_branch.invoke_hunyuan_720p(&backend, &input, &context)?)?
    );

    let mut state_1080 = Vec::new();
    let conv = |output_channels: usize, input_channels: usize, centers: &[(usize, f32)]| {
        let mut values = vec![0.0_f32; output_channels * input_channels * 27];
        for (index, value) in centers {
            values[*index] = *value;
        }
        values
    };
    state_1080.push((
        "conv_in.conv.weight".to_owned(),
        tensor(&[2, 1, 3, 3, 3], &conv(2, 1, &[(22, 1.0), (49, 2.0)]))?,
    ));
    state_1080.push(("conv_in.conv.bias".to_owned(), tensor(&[2], &[0.0; 2])?));
    let residual_branch_weights = [[0.5_f32, -0.25], [-0.75, 0.5], [0.25, 1.0]];
    for (block, branch) in residual_branch_weights.iter().enumerate() {
        for norm in ["norm1", "norm2"] {
            state_1080.push((
                format!("up.0.block.{block}.{norm}.gamma"),
                tensor(&[2, 1, 1, 1], &[1.0, 1.0])?,
            ));
        }
        state_1080.push((
            format!("up.0.block.{block}.conv1.conv.weight"),
            tensor(&[2, 2, 3, 3, 3], &conv(2, 2, &[(22, 1.0), (103, 1.0)]))?,
        ));
        state_1080.push((
            format!("up.0.block.{block}.conv1.conv.bias"),
            tensor(&[2], &[0.0; 2])?,
        ));
        state_1080.push((
            format!("up.0.block.{block}.conv2.conv.weight"),
            tensor(
                &[2, 2, 3, 3, 3],
                &conv(2, 2, &[(22, branch[0]), (103, branch[1])]),
            )?,
        ));
        state_1080.push((
            format!("up.0.block.{block}.conv2.conv.bias"),
            tensor(&[2], &[0.0; 2])?,
        ));
    }
    state_1080.push((
        "norm_out.gamma".to_owned(),
        tensor(&[2, 1, 1, 1], &[1.0, 1.0])?,
    ));
    state_1080.push((
        "conv_out.conv.weight".to_owned(),
        tensor(&[1, 2, 3, 3, 3], &conv(1, 2, &[(22, 1.0), (49, -1.0)]))?,
    ));
    state_1080.push(("conv_out.conv.bias".to_owned(), tensor(&[1], &[0.0])?));
    let mut mutated_state_1080 = state_1080.clone();
    let resource_1080 = NativeLatentUpscaleModelResource::from_checkpoint(
        NativeLatentUpscaleCheckpoint {
            artifact_sha256: "2".repeat(64),
            metadata: BTreeMap::new(),
            ordered_state: state_1080,
            memory_budget_bytes: workspace_bytes,
        },
        &context,
    )?;
    let output_1080 = resource_1080.invoke_hunyuan_1080p(
        &backend,
        &tensor(&[1, 1, 1, 1, 1], &[1.0])?,
        &context,
    )?;
    assert_eq!(
        tensor_bits(&output_1080)?,
        fixture_raw_bits(
            case("1080-repeat-rms-shortcut-order")?
                .get("expected_bits")
                .ok_or("1080 bits")?
        )?
    );
    let (_, branch_weight) = mutated_state_1080
        .iter_mut()
        .find(|(key, _)| key == "up.0.block.1.conv2.conv.weight")
        .ok_or("ordered 1080 branch weight is missing")?;
    *branch_weight = tensor(&[2, 2, 3, 3, 3], &conv(2, 2, &[(22, -0.5), (103, 0.75)]))?;
    let mutated_resource_1080 = NativeLatentUpscaleModelResource::from_checkpoint(
        NativeLatentUpscaleCheckpoint {
            artifact_sha256: "6".repeat(64),
            metadata: BTreeMap::new(),
            ordered_state: mutated_state_1080,
            memory_budget_bytes: workspace_bytes,
        },
        &context,
    )?;
    assert_ne!(
        tensor_bits(&output_1080)?,
        tensor_bits(&mutated_resource_1080.invoke_hunyuan_1080p(
            &backend,
            &tensor(&[1, 1, 1, 1, 1], &[1.0])?,
            &context,
        )?)?
    );

    let shuffle_one = pixel_shuffle_nd_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 2, 2, 1, 1], &[10.0, 20.0, 11.0, 21.0])?,
        1,
        2,
        &context,
    )?;
    assert_eq!(
        tensor_bits(&shuffle_one)?,
        fixture_raw_bits(
            case("pixel-shuffle-dimension-one")?
                .get("expected_frames_bits")
                .ok_or("shuffle-one bits")?
        )?
    );
    let mutated_shuffle_one = pixel_shuffle_nd_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 2, 2, 1, 1], &[10.0, 20.0, 12.0, 21.0])?,
        1,
        2,
        &context,
    )?;
    assert_ne!(
        tensor_bits(&mutated_shuffle_one)?,
        tensor_bits(&shuffle_one)?
    );
    let shuffle_two = pixel_shuffle_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0])?,
        2,
        &context,
    )?;
    assert_eq!(
        tensor_bits(&shuffle_two)?,
        fixture_raw_bits(
            case("pixel-shuffle-dimension-two")?
                .get("expected_bits")
                .ok_or("shuffle-two bits")?
        )?
    );
    let mutated_shuffle_two = pixel_shuffle_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 4, 1, 1], &[1.0, 2.0, 3.0, 5.0])?,
        2,
        &context,
    )?;
    assert_ne!(
        tensor_bits(&mutated_shuffle_two)?,
        tensor_bits(&shuffle_two)?
    );
    let shuffle_three = pixel_shuffle_nd_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 8, 1, 1, 1], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])?,
        3,
        2,
        &context,
    )?;
    assert_eq!(
        tensor_bits(&shuffle_three)?,
        fixture_raw_bits(
            case("pixel-shuffle-dimension-three")?
                .get("expected_bits")
                .ok_or("shuffle-three bits")?
        )?
    );
    let mutated_shuffle = pixel_shuffle_nd_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 8, 1, 1, 1], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])?,
        3,
        2,
        &context,
    )?;
    assert_ne!(tensor_bits(&mutated_shuffle)?, tensor_bits(&shuffle_three)?);

    let nearest = interpolate_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 1, 1, 2], &[0.0, 10.0])?,
        &InterpolateConfiguration {
            output_size: Some(vec![1, 3]),
            scale_factor: None,
            mode: InterpolateMode::NearestExact,
            align_corners: None,
            recompute_scale_factor: None,
            antialias: false,
        },
        &context,
    )?;
    assert_eq!(
        tensor_bits(&nearest)?,
        fixture_raw_bits(
            case("nearest-exact-half-coordinate")?
                .get("expected_bits")
                .ok_or("nearest bits")?
        )?
    );
    let floor_nearest = interpolate_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 1, 1, 2], &[0.0, 10.0])?,
        &InterpolateConfiguration {
            output_size: Some(vec![1, 3]),
            scale_factor: None,
            mode: InterpolateMode::Nearest,
            align_corners: None,
            recompute_scale_factor: None,
            antialias: false,
        },
        &context,
    )?;
    assert_ne!(tensor_bits(&floor_nearest)?, tensor_bits(&nearest)?);
    let mutated_nearest = interpolate_tensor_with_context_exact_native(
        &backend,
        &tensor(&[1, 1, 1, 2], &[0.0, 11.0])?,
        &InterpolateConfiguration {
            output_size: Some(vec![1, 3]),
            scale_factor: None,
            mode: InterpolateMode::NearestExact,
            align_corners: None,
            recompute_scale_factor: None,
            antialias: false,
        },
        &context,
    )?;
    assert_ne!(tensor_bits(&mutated_nearest)?, tensor_bits(&nearest)?);

    let bislerp_case = case("bislerp-edge-cases")?;
    let pairs = bislerp_case
        .get("pairs")
        .and_then(serde_json::Value::as_array)
        .ok_or("bislerp pairs")?;
    let expected = fixture_raw_bits(bislerp_case.get("expected_bits").ok_or("bislerp bits")?)?;
    let targets = [(4_u64, 1_usize), (3, 1), (3, 1), (8, 3)];
    let changed_ratio_targets = [(3_u64, 1_usize), (4, 1), (4, 1), (4, 1)];
    let mut actual_bislerp = Vec::new();
    for (pair_index, (pair, (target_width, sample))) in pairs.iter().zip(targets).enumerate() {
        let left = pair
            .get("left")
            .and_then(serde_json::Value::as_array)
            .ok_or("bislerp left")?;
        let right = pair
            .get("right")
            .and_then(serde_json::Value::as_array)
            .ok_or("bislerp right")?;
        let values = [
            left[0].as_f64().ok_or("left 0")? as f32,
            right[0].as_f64().ok_or("right 0")? as f32,
            left[1].as_f64().ok_or("left 1")? as f32,
            right[1].as_f64().ok_or("right 1")? as f32,
        ];
        let output = bislerp_tensor_with_context_exact_native(
            &backend,
            &tensor(&[1, 2, 1, 2], &values)?,
            target_width,
            1,
            &context,
        )?;
        let bits = tensor_bits(&output)?;
        actual_bislerp.extend([bits[sample], bits[target_width as usize + sample]]);

        let mut mutated_values = values;
        mutated_values[3] += 0.125;
        let mutated = bislerp_tensor_with_context_exact_native(
            &backend,
            &tensor(&[1, 2, 1, 2], &mutated_values)?,
            target_width,
            1,
            &context,
        )?;
        let mutated_bits = tensor_bits(&mutated)?;
        assert_ne!(
            [bits[sample], bits[target_width as usize + sample]],
            [
                mutated_bits[sample],
                mutated_bits[target_width as usize + sample]
            ],
            "bislerp pair {pair_index} did not discriminate a vector mutation"
        );

        let (changed_width, changed_sample) = changed_ratio_targets[pair_index];
        let changed_ratio = bislerp_tensor_with_context_exact_native(
            &backend,
            &tensor(&[1, 2, 1, 2], &values)?,
            changed_width,
            1,
            &context,
        )?;
        let changed_ratio_bits = tensor_bits(&changed_ratio)?;
        let baseline_sample = [bits[sample], bits[target_width as usize + sample]];
        let changed_sample = [
            changed_ratio_bits[changed_sample],
            changed_ratio_bits[changed_width as usize + changed_sample],
        ];
        if pair_index == 1 {
            assert_eq!(baseline_sample, changed_sample);
        } else {
            assert_ne!(
                baseline_sample, changed_sample,
                "bislerp pair {pair_index} did not discriminate a ratio mutation"
            );
        }
    }
    assert_eq!(actual_bislerp, expected);

    let blur_case = case("rational-blur-center-delta")?;
    let blur_numerator = blur_case
        .get("expected_numerator")
        .and_then(serde_json::Value::as_array)
        .ok_or("blur numerator")?;
    let blur_denominator = blur_case
        .get("normalization_denominator")
        .and_then(serde_json::Value::as_u64)
        .ok_or("blur denominator")? as f32;
    let mut blur_kernel = Vec::new();
    for row in blur_numerator {
        for value in row.as_array().ok_or("blur numerator row")? {
            blur_kernel
                .push(value.as_u64().ok_or("blur numerator value")? as f32 / blur_denominator);
        }
    }
    let blur_axis = [1.0_f32, 4.0, 6.0, 4.0, 1.0];
    let source_kernel = blur_axis
        .iter()
        .flat_map(|vertical| {
            blur_axis
                .iter()
                .map(move |horizontal| vertical * horizontal / 256.0)
        })
        .collect::<Vec<_>>();
    let config = serde_json::json!({
        "_class_name": "LatentUpsampler",
        "in_channels": 1,
        "mid_channels": 32,
        "num_blocks_per_stage": 1,
        "dims": 3,
        "spatial_upsample": true,
        "temporal_upsample": false,
        "spatial_scale": 1.5,
        "rational_resampler": true,
    });
    let zeros = |length: usize| vec![0.0_f32; length];
    let mut rational_state = vec![
        (
            "initial_conv.weight".to_owned(),
            tensor(&[32, 1, 3, 3, 3], &zeros(32 * 27))?,
        ),
        ("initial_conv.bias".to_owned(), tensor(&[32], &zeros(32))?),
        ("initial_norm.weight".to_owned(), tensor(&[32], &[1.0; 32])?),
        ("initial_norm.bias".to_owned(), tensor(&[32], &zeros(32))?),
        (
            "upsampler.conv.weight".to_owned(),
            tensor(&[288, 32, 3, 3], &zeros(288 * 32 * 9))?,
        ),
        (
            "upsampler.conv.bias".to_owned(),
            tensor(&[288], &zeros(288))?,
        ),
        (
            "upsampler.blur_down.kernel".to_owned(),
            tensor(&[1, 1, 5, 5], &source_kernel)?,
        ),
        (
            "final_conv.weight".to_owned(),
            tensor(&[1, 32, 3, 3, 3], &zeros(32 * 27))?,
        ),
        ("final_conv.bias".to_owned(), tensor(&[1], &[0.0])?),
    ];
    for family in ["res_blocks", "post_upsample_res_blocks"] {
        for convolution in ["conv1", "conv2"] {
            rational_state.push((
                format!("{family}.0.{convolution}.weight"),
                tensor(&[32, 32, 3, 3, 3], &zeros(32 * 32 * 27))?,
            ));
            rational_state.push((
                format!("{family}.0.{convolution}.bias"),
                tensor(&[32], &zeros(32))?,
            ));
        }
        for normalization in ["norm1", "norm2"] {
            rational_state.push((
                format!("{family}.0.{normalization}.weight"),
                tensor(&[32], &[1.0; 32])?,
            ));
            rational_state.push((
                format!("{family}.0.{normalization}.bias"),
                tensor(&[32], &zeros(32))?,
            ));
        }
    }
    let mut mutated_rational_state = rational_state.clone();
    let rational_resource = NativeLatentUpscaleModelResource::from_checkpoint(
        NativeLatentUpscaleCheckpoint {
            artifact_sha256: "3".repeat(64),
            metadata: BTreeMap::from([("config".to_owned(), serde_json::to_string(&config)?)]),
            ordered_state: rational_state,
            memory_budget_bytes: workspace_bytes,
        },
        &context,
    )?;
    let mut center_delta = vec![0.0_f32; 25];
    center_delta[12] = 1.0;
    let blurred = rational_resource.rational_blur_test_support(
        &backend,
        &tensor(&[1, 1, 5, 5], &center_delta)?,
        2,
        &context,
    )?;
    let blurred_values =
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend, &blurred, &context,
        )?;
    assert_eq!(blurred_values, blur_kernel);
    let mut mutated_kernel = source_kernel;
    mutated_kernel[12] += 1.0 / 256.0;
    let (_, kernel) = mutated_rational_state
        .iter_mut()
        .find(|(key, _)| key == "upsampler.blur_down.kernel")
        .ok_or("rational blur kernel is missing")?;
    *kernel = tensor(&[1, 1, 5, 5], &mutated_kernel)?;
    let mutated_rational_resource = NativeLatentUpscaleModelResource::from_checkpoint(
        NativeLatentUpscaleCheckpoint {
            artifact_sha256: "7".repeat(64),
            metadata: BTreeMap::from([("config".to_owned(), serde_json::to_string(&config)?)]),
            ordered_state: mutated_rational_state,
            memory_budget_bytes: workspace_bytes,
        },
        &context,
    )?;
    let mutated_blur = mutated_rational_resource.rational_blur_test_support(
        &backend,
        &tensor(&[1, 1, 5, 5], &center_delta)?,
        2,
        &context,
    )?;
    assert_ne!(tensor_bits(&mutated_blur)?, tensor_bits(&blurred)?);

    let ltx_case = case("ltx-vae-statistics-order")?;
    let fixture_floats = |field: &str| -> Result<[f32; 4], Box<dyn Error>> {
        let values = ltx_case
            .get(field)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("missing {field}"))?;
        Ok([
            values[0].as_f64().ok_or("fixture float 0")? as f32,
            values[1].as_f64().ok_or("fixture float 1")? as f32,
            values[2].as_f64().ok_or("fixture float 2")? as f32,
            values[3].as_f64().ok_or("fixture float 3")? as f32,
        ])
    };
    let mut means = [0.0_f32; 128];
    means[..4].copy_from_slice(&fixture_floats("mean")?);
    let mut standard_deviations = [1.0_f32; 128];
    standard_deviations[..4].copy_from_slice(&fixture_floats("standard_deviation")?);
    let mut ltx_input = [0.0_f32; 128];
    ltx_input[..4].copy_from_slice(&fixture_floats("input")?);
    let unnormalized = sim_model::vae_video::ltx_latent_statistics_test_support(
        &backend,
        &tensor(&[1, 128, 1, 1, 1], &ltx_input)?,
        &means,
        &standard_deviations,
        false,
        &context,
    )?;
    assert_eq!(
        &tensor_bits(&unnormalized)?[..4],
        fixture_raw_bits(
            ltx_case
                .get("expected_unnormalized_bits")
                .ok_or("unnormalized bits")?
        )?
        .as_slice()
    );
    let mut model_raw = [0.0_f32; 128];
    model_raw[..4].copy_from_slice(&fixture_floats("model_raw")?);
    let normalized = sim_model::vae_video::ltx_latent_statistics_test_support(
        &backend,
        &tensor(&[1, 128, 1, 1, 1], &model_raw)?,
        &means,
        &standard_deviations,
        true,
        &context,
    )?;
    assert_eq!(
        &tensor_bits(&normalized)?[..4],
        fixture_raw_bits(
            ltx_case
                .get("expected_normalized_bits")
                .ok_or("normalized bits")?
        )?
        .as_slice()
    );
    let mut mutated_means = means;
    mutated_means[0] += 1.0;
    let mean_mutation = sim_model::vae_video::ltx_latent_statistics_test_support(
        &backend,
        &tensor(&[1, 128, 1, 1, 1], &ltx_input)?,
        &mutated_means,
        &standard_deviations,
        false,
        &context,
    )?;
    assert_ne!(
        &tensor_bits(&mean_mutation)?[..4],
        &tensor_bits(&unnormalized)?[..4]
    );

    let mut mutated_standard_deviations = standard_deviations;
    mutated_standard_deviations[0] += 1.0;
    let standard_deviation_mutation = sim_model::vae_video::ltx_latent_statistics_test_support(
        &backend,
        &tensor(&[1, 128, 1, 1, 1], &model_raw)?,
        &means,
        &mutated_standard_deviations,
        true,
        &context,
    )?;
    assert_ne!(
        &tensor_bits(&standard_deviation_mutation)?[..4],
        &tensor_bits(&normalized)?[..4]
    );

    let reversed_order = sim_model::vae_video::ltx_latent_statistics_test_support(
        &backend,
        &tensor(&[1, 128, 1, 1, 1], &ltx_input)?,
        &means,
        &standard_deviations,
        true,
        &context,
    )?;
    assert_ne!(
        &tensor_bits(&reversed_order)?[..4],
        &tensor_bits(&unnormalized)?[..4]
    );
    Ok(())
}

#[test]
fn native_background_removal_fixture_executes_and_discriminates_source_phases()
-> Result<(), Box<dyn Error>> {
    let fixture: serde_json::Value = serde_json::from_str(BACKGROUND_REMOVAL_RESOURCE_FIXTURE)?;
    let oracle: serde_json::Value = serde_json::from_str(BACKGROUND_REMOVAL_RESOURCE_ORACLE)?;
    let provenance: serde_json::Value =
        serde_json::from_str(BACKGROUND_REMOVAL_RESOURCE_PROVENANCE)?;
    assert_eq!(
        fixture
            .get("oracle_domain")
            .and_then(serde_json::Value::as_str),
        Some("zed.sim.background-removal-source-profile.v1")
    );
    assert_eq!(
        fixture
            .get("source_sha256")
            .and_then(serde_json::Value::as_str),
        Some(sim_model::BIREFNET_SOURCE_SHA256)
    );
    assert_eq!(
        oracle.get("format").and_then(serde_json::Value::as_str),
        Some("zed.sim.background-removal-reduced-oracle.v1")
    );
    let generator_sha256 = format!(
        "{:x}",
        Sha256::digest(BACKGROUND_REMOVAL_RESOURCE_GENERATOR.as_bytes())
    );
    let oracle_sha256 = format!(
        "{:x}",
        Sha256::digest(BACKGROUND_REMOVAL_RESOURCE_ORACLE.as_bytes())
    );
    for document in [&fixture, &oracle, &provenance] {
        assert_eq!(
            document
                .get("generator_sha256")
                .and_then(serde_json::Value::as_str),
            Some(generator_sha256.as_str())
        );
    }
    for document in [&fixture, &provenance] {
        assert_eq!(
            document
                .get("oracle_sha256")
                .and_then(serde_json::Value::as_str),
            Some(oracle_sha256.as_str())
        );
    }
    for field in ["generator_command", "platform", "python"] {
        let oracle_value = oracle
            .get(field)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or("background-removal oracle provenance field is missing")?;
        assert_eq!(
            provenance.get(field).and_then(serde_json::Value::as_str),
            Some(oracle_value)
        );
    }
    assert!(
        oracle
            .get("f32_rule")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|value| !value.is_empty())
    );
    let expected_sources = BTreeMap::from([
        (
            "projects/comfy/ComfyUI/comfy_extras/nodes_bg_removal.py",
            sim_model::NODES_BACKGROUND_REMOVAL_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/bg_removal_model.py",
            sim_model::BACKGROUND_REMOVAL_MODEL_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/background_removal/birefnet.py",
            sim_model::BIREFNET_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/background_removal/birefnet.json",
            sim_model::BIREFNET_CONFIG_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/clip_model.py",
            sim_model::CLIP_MODEL_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/ops.py",
            sim_model::SIM_OPS_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/model_management.py",
            sim_model::MODEL_MANAGEMENT_SOURCE_SHA256,
        ),
    ]);
    for document in [&oracle, &provenance] {
        let pinned_sources = document
            .get("pinned_sources")
            .and_then(serde_json::Value::as_object)
            .ok_or("background-removal pinned sources are missing")?;
        assert_eq!(pinned_sources.len(), expected_sources.len());
        for (path, expected_sha256) in &expected_sources {
            assert_eq!(
                pinned_sources
                    .get(*path)
                    .and_then(serde_json::Value::as_str),
                Some(*expected_sha256)
            );
        }
    }
    let input_shape = fixture
        .get("input_shape")
        .and_then(serde_json::Value::as_array)
        .ok_or("background-removal input shape is missing")?
        .iter()
        .map(|value| value.as_u64().ok_or("invalid input-shape dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    let output_shape = fixture
        .get("output_shape")
        .and_then(serde_json::Value::as_array)
        .ok_or("background-removal output shape is missing")?
        .iter()
        .map(|value| value.as_u64().ok_or("invalid output-shape dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(input_shape, vec![1, 3, 5, 4]);
    assert_eq!(output_shape, vec![1, 3, 5]);
    assert_eq!(oracle.get("input_shape"), fixture.get("input_shape"));
    assert_eq!(oracle.get("output_shape"), fixture.get("output_shape"));
    let input = fixture
        .get("input_rgba")
        .and_then(serde_json::Value::as_array)
        .ok_or("background-removal input fixture is missing")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or("background-removal input value is invalid")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_bits = fixture_raw_bits(
        oracle
            .get("output_bits")
            .ok_or("background-removal oracle output is missing")?,
    )?;
    assert_eq!(
        fixture_raw_bits(
            fixture
                .get("baseline_output_bits")
                .ok_or("background-removal output fixture is missing")?,
        )?,
        expected_bits
    );
    let expected_raw_f32_sha256 = oracle
        .get("raw_f32_sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or("background-removal raw-output hash is missing")?;
    for document in [&fixture, &provenance] {
        assert_eq!(
            document
                .get("raw_f32_sha256")
                .and_then(serde_json::Value::as_str),
            Some(expected_raw_f32_sha256)
        );
    }
    let required_mutations = fixture
        .get("required_mutations")
        .and_then(serde_json::Value::as_array)
        .ok_or("background-removal mutation fixture is missing")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or("background-removal mutation is invalid")
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    assert_eq!(
        required_mutations,
        BTreeSet::from([
            "aspp-dilated-branch",
            "aspp-global-pool",
            "deform-mask",
            "deform-offset",
            "relative-position-index",
            "shifted-window-block",
            "unused-decoder-head",
        ])
    );
    let mutation_oracles = oracle
        .get("mutations")
        .and_then(serde_json::Value::as_object)
        .ok_or("background-removal mutation oracles are missing")?;
    assert_eq!(
        mutation_oracles
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        required_mutations
    );
    for (name, mutation_oracle) in mutation_oracles {
        let mutation_bits = fixture_raw_bits(
            mutation_oracle
                .get("output_bits")
                .ok_or("background-removal mutation output is missing")?,
        )?;
        assert_eq!(
            mutation_bits != expected_bits,
            name != "unused-decoder-head",
            "{name} is not a discriminating source-equation oracle"
        );
        let mut mutation_hasher = Sha256::new();
        for bits in &mutation_bits {
            mutation_hasher.update(f32::from_bits(*bits).to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", mutation_hasher.finalize()),
            mutation_oracle
                .get("raw_f32_sha256")
                .and_then(serde_json::Value::as_str)
                .ok_or("background-removal mutation hash is missing")?
        );
    }

    let workspace_bytes = 512 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(workspace_bytes)?,
        &cancellation,
    );
    let image = ImageTensor::from_f32(&backend, &context, 1, 3, 5, 4, &input)?;
    let resource = Arc::new(
        NativeBackgroundRemovalResource::deterministic_reduced_test_fixture(
            &backend,
            &context,
            BackgroundRemovalFixtureMutation::None,
        )?,
    );
    let output = resource.encode_image(&backend, &image, &context)?;
    let output_values =
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend, &output, &context,
        )?;
    let output_bits = output_values
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let mut raw_output_hasher = Sha256::new();
    for value in &output_values {
        raw_output_hasher.update(value.to_le_bytes());
    }
    let actual_raw_f32_sha256 = format!("{:x}", raw_output_hasher.finalize());
    assert_eq!(actual_raw_f32_sha256, expected_raw_f32_sha256);
    assert_eq!(output.descriptor().shape(), output_shape.as_slice());
    assert!(
        output_values
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
    );
    assert_eq!(
        output_bits, expected_bits,
        "background-removal baseline bits were {output_bits:?}"
    );

    let mut alpha_mutation = input.clone();
    for alpha in alpha_mutation.iter_mut().skip(3).step_by(4) {
        *alpha = 1.0 - *alpha;
    }
    let alpha_output = resource.encode_image(
        &backend,
        &ImageTensor::from_f32(&backend, &context, 1, 3, 5, 4, &alpha_mutation)?,
        &context,
    )?;
    assert_eq!(
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend,
            &alpha_output,
            &context,
        )?
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>(),
        output_bits
    );

    let mut rgb_mutation = input.clone();
    rgb_mutation[0] = 0.75;
    let rgb_output = resource.encode_image(
        &backend,
        &ImageTensor::from_f32(&backend, &context, 1, 3, 5, 4, &rgb_mutation)?,
        &context,
    )?;
    let rgb_bits =
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend,
            &rgb_output,
            &context,
        )?
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_ne!(rgb_bits, output_bits);

    let mut ordered_batch = input.clone();
    ordered_batch.extend_from_slice(&rgb_mutation);
    let mut reversed_batch = rgb_mutation.clone();
    reversed_batch.extend_from_slice(&input);
    let ordered = resource.encode_image(
        &backend,
        &ImageTensor::from_f32(&backend, &context, 2, 3, 5, 4, &ordered_batch)?,
        &context,
    )?;
    let reversed = resource.encode_image(
        &backend,
        &ImageTensor::from_f32(&backend, &context, 2, 3, 5, 4, &reversed_batch)?,
        &context,
    )?;
    let ordered_bits =
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend, &ordered, &context,
        )?
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let reversed_bits =
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend, &reversed, &context,
        )?
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(&ordered_bits[..15], output_bits.as_slice());
    assert_eq!(&ordered_bits[15..], rgb_bits.as_slice());
    assert_eq!(&reversed_bits[..15], rgb_bits.as_slice());
    assert_eq!(&reversed_bits[15..], output_bits.as_slice());

    for (mutation, mutation_name, changes_output) in [
        (
            BackgroundRemovalFixtureMutation::ShiftedWindowBlock,
            "shifted-window-block",
            true,
        ),
        (
            BackgroundRemovalFixtureMutation::RelativePositionIndex,
            "relative-position-index",
            true,
        ),
        (
            BackgroundRemovalFixtureMutation::DeformOffset,
            "deform-offset",
            true,
        ),
        (
            BackgroundRemovalFixtureMutation::DeformMask,
            "deform-mask",
            true,
        ),
        (
            BackgroundRemovalFixtureMutation::AsppDilatedBranch,
            "aspp-dilated-branch",
            true,
        ),
        (
            BackgroundRemovalFixtureMutation::AsppGlobalPool,
            "aspp-global-pool",
            true,
        ),
        (
            BackgroundRemovalFixtureMutation::UnusedDecoderHead,
            "unused-decoder-head",
            false,
        ),
    ] {
        let mutated = NativeBackgroundRemovalResource::deterministic_reduced_test_fixture(
            &backend, &context, mutation,
        )?;
        assert_ne!(
            mutated.semantic_digest_sha256(),
            resource.semantic_digest_sha256(),
            "{mutation:?} did not change resource identity"
        );
        let mutated_output = mutated.encode_image(&backend, &image, &context)?;
        let mutated_bits = sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend,
            &mutated_output,
            &context,
        )?
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
        let expected_mutated_bits = fixture_raw_bits(
            mutation_oracles
                .get(mutation_name)
                .and_then(|value| value.get("output_bits"))
                .ok_or("background-removal mutation oracle is missing")?,
        )?;
        assert_eq!(
            mutated_bits, expected_mutated_bits,
            "{mutation:?} diverged from the independent source-equation oracle"
        );
        assert_eq!(
            mutated_bits != output_bits,
            changes_output,
            "{mutation:?} output disposition changed"
        );
    }

    let payload =
        NativeModelPayload::background_removal_test_fixture(resource.clone(), &cancellation)?;
    assert_eq!(
        payload
            .background_removal_resource()
            .ok_or("background-removal resource projection is missing")?
            .semantic_digest_sha256(),
        resource.semantic_digest_sha256()
    );
    assert_eq!(
        payload.resident_parts()?.resident_bytes()?,
        payload.resident_bytes()
    );

    let memory_before_cancellation = backend.memory_snapshot();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        resource.encode_image(
            &backend,
            &image,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(workspace_bytes)?,
                &cancelled,
            ),
        ),
        Err(NativeBackgroundRemovalError::Cancelled)
    ));
    assert!(matches!(
        NativeModelPayload::background_removal_test_fixture(resource, &cancelled),
        Err(NativeModelPayloadError::Tensor(
            sim_tensor::TensorError::Cancelled
        ))
    ));
    assert_eq!(backend.memory_snapshot(), memory_before_cancellation);
    Ok(())
}

fn assert_depth_anything_3_checkpoint_projection(
    oracle: &serde_json::Value,
    profile: &str,
    dtype: &str,
    resource: &NativeDepthAnything3Resource,
    cancellation: &CancellationToken,
) -> Result<(), Box<dyn Error>> {
    let expected = oracle
        .pointer(&format!("/checkpoint_projection/{profile}/{dtype}"))
        .ok_or("DA3 checkpoint projection oracle is missing")?;
    assert_eq!(
        expected.get("ordering").and_then(serde_json::Value::as_str),
        Some("utf8-key-ascending")
    );
    let actual = reduced_depth_anything_3_checkpoint_parity_for_fixture(resource, cancellation)?;
    assert_eq!(
        u64::try_from(actual.states.len())?,
        expected
            .get("key_count")
            .and_then(serde_json::Value::as_u64)
            .ok_or("DA3 checkpoint projection key count is missing")?
    );
    let expected_states = expected
        .get("states")
        .and_then(serde_json::Value::as_array)
        .ok_or("DA3 checkpoint state projection list is missing")?;
    assert_eq!(actual.states.len(), expected_states.len());
    for (actual, expected) in actual.states.iter().zip(expected_states) {
        assert_eq!(
            actual.key,
            expected
                .get("key")
                .and_then(serde_json::Value::as_str)
                .ok_or("DA3 projected checkpoint key is missing")?
        );
        let expected_shape = expected
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or("DA3 projected checkpoint shape is missing")?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .ok_or("DA3 projected checkpoint dimension is invalid")
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(actual.shape, expected_shape);
        assert_eq!(
            actual.source_sha256,
            expected
                .get("source_sha256")
                .and_then(serde_json::Value::as_str)
                .ok_or("DA3 checkpoint state source digest is missing")?
        );
        assert_eq!(
            actual.projected_f32_sha256,
            expected
                .get("projected_f32_sha256")
                .and_then(serde_json::Value::as_str)
                .ok_or("DA3 checkpoint state projected digest is missing")?
        );
    }
    assert_eq!(
        actual.source_sha256,
        expected
            .get("source_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or("DA3 checkpoint source digest is missing")?
    );
    assert_eq!(
        actual.projected_f32_sha256,
        expected
            .get("projected_f32_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or("DA3 projected checkpoint digest is missing")?
    );
    Ok(())
}

#[test]
fn native_depth_anything_3_reduced_resources_execute_and_publish_typed_geometry()
-> Result<(), Box<dyn Error>> {
    let manifest: serde_json::Value = serde_json::from_str(DEPTH_ANYTHING_3_RESOURCE_FIXTURE)?;
    let oracle: serde_json::Value = serde_json::from_str(DEPTH_ANYTHING_3_RESOURCE_ORACLE)?;
    let provenance: serde_json::Value = serde_json::from_str(DEPTH_ANYTHING_3_RESOURCE_PROVENANCE)?;
    assert_eq!(
        manifest
            .get("oracle_domain")
            .and_then(serde_json::Value::as_str),
        oracle.get("format").and_then(serde_json::Value::as_str)
    );
    let generator_sha256 = format!("{:x}", Sha256::digest(DEPTH_ANYTHING_3_RESOURCE_GENERATOR));
    let source_graph_sha256 = format!(
        "{:x}",
        Sha256::digest(DEPTH_ANYTHING_3_RESOURCE_SOURCE_GRAPH)
    );
    let oracle_sha256 = format!("{:x}", Sha256::digest(DEPTH_ANYTHING_3_RESOURCE_ORACLE));
    for document in [&manifest, &provenance] {
        assert_eq!(
            document
                .get("generator_sha256")
                .and_then(serde_json::Value::as_str),
            Some(generator_sha256.as_str())
        );
        assert_eq!(
            (if cfg!(all(
                target_os = "linux",
                target_arch = "x86_64",
                target_env = "gnu"
            )) {
                document.pointer("/oracle_platform_sha256/oracle-linux-gnu-x86_64.json")
            } else {
                document.get("oracle_sha256")
            })
            .and_then(serde_json::Value::as_str),
            Some(oracle_sha256.as_str())
        );
        assert_eq!(
            document
                .get("source_graph_sha256")
                .and_then(serde_json::Value::as_str),
            Some(source_graph_sha256.as_str())
        );
    }
    assert_eq!(
        provenance
            .get("transcendental_boundary")
            .and_then(serde_json::Value::as_str),
        Some(
            "F32 camera atan, DPT positional sin/cos/pow, and canonical exponential oracle bits call the pinned macOS host libc atanf/sinf/cosf/powf/expf through Python ctypes; they are not claimed portable to another libc or platform."
        )
    );
    let expected_sources = BTreeMap::from([
        (
            "projects/comfy/ComfyUI/comfy_extras/nodes_depth_anything_3.py",
            sim_model::NODES_DEPTH_ANYTHING_3_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/ldm/depth_anything_3/model.py",
            sim_model::DEPTH_ANYTHING_3_MODEL_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/ldm/depth_anything_3/preprocess.py",
            sim_model::DEPTH_ANYTHING_3_PREPROCESS_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/ldm/depth_anything_3/dpt.py",
            sim_model::DEPTH_ANYTHING_3_DPT_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/ldm/depth_anything_3/camera.py",
            sim_model::DEPTH_ANYTHING_3_CAMERA_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/ldm/depth_anything_3/ray_pose.py",
            sim_model::DEPTH_ANYTHING_3_RAY_POSE_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/ldm/depth_anything_3/reference_view_selector.py",
            sim_model::DEPTH_ANYTHING_3_REFERENCE_VIEW_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/ldm/depth_anything_3/transform.py",
            sim_model::DEPTH_ANYTHING_3_TRANSFORM_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/image_encoders/dino2.py",
            sim_model::DEPTH_ANYTHING_3_DINO2_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/model_detection.py",
            sim_model::DEPTH_ANYTHING_3_MODEL_DETECTION_SOURCE_SHA256,
        ),
        (
            "projects/comfy/ComfyUI/comfy/text_encoders/bert.py",
            "3f1f32353da95790285a10f452959a871aa949aab15a89b646a95abc6165955c",
        ),
        (
            "projects/comfy/ComfyUI/comfy/ldm/modules/attention.py",
            "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e",
        ),
        (
            "projects/comfy/ComfyUI/comfy/utils.py",
            "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
        ),
        (
            "projects/comfy/ComfyUI/comfy/ops.py",
            "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42",
        ),
        (
            "projects/comfy/ComfyUI/comfy/model_management.py",
            "c2ca243c80a5262ecafe19feb15cec22d4003c16e523b5376f543f0f75acabaa",
        ),
        (
            "projects/comfy/ComfyUI/comfy/supported_models.py",
            "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69",
        ),
        (
            "projects/comfy/ComfyUI/comfy/model_base.py",
            "99dc53baee665eca1a6aea70cfb9ab071d55784dff339b5e919dc14ae4fde8bd",
        ),
    ]);
    for document in [&oracle, &provenance] {
        let pinned = document
            .get("pinned_sources")
            .and_then(serde_json::Value::as_object)
            .ok_or("DA3 pinned sources are missing")?;
        assert_eq!(pinned.len(), expected_sources.len());
        for (path, digest) in &expected_sources {
            assert_eq!(
                pinned.get(*path).and_then(serde_json::Value::as_str),
                Some(*digest)
            );
        }
    }
    assert_ne!(
        oracle
            .pointer("/reference_fixture/saddle_balanced")
            .and_then(serde_json::Value::as_u64),
        oracle
            .pointer("/reference_fixture/saddle_sim_range")
            .and_then(serde_json::Value::as_u64),
        "the independent reference fixture must discriminate both saddle strategies"
    );
    let reference_tokens = fixture_raw_bits(
        oracle
            .pointer("/reference_fixture/token_bits")
            .ok_or("DA3 reference tokens are missing")?,
    )?
    .into_iter()
    .map(f32::from_bits)
    .collect::<Vec<_>>();
    for (name, strategy) in [
        ("first", NativeDepthAnything3ReferenceStrategy::First),
        ("middle", NativeDepthAnything3ReferenceStrategy::Middle),
        (
            "saddle_balanced",
            NativeDepthAnything3ReferenceStrategy::SaddleBalanced,
        ),
        (
            "saddle_sim_range",
            NativeDepthAnything3ReferenceStrategy::SaddleSimRange,
        ),
    ] {
        assert_eq!(
            u64::try_from(select_reduced_depth_anything_3_reference_for_fixture(
                &reference_tokens,
                4,
                strategy,
                &CancellationToken::default(),
            )?)?,
            oracle
                .pointer(&format!("/reference_fixture/{name}"))
                .and_then(serde_json::Value::as_u64)
                .ok_or("DA3 reference index is missing")?,
            "production reference selection diverged for {name}"
        );
    }

    let workspace_bytes = 256 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(workspace_bytes)?,
        &cancellation,
    );
    let tensor_bits = |tensor: &Tensor,
                       context: &ExecutionContext<'_>|
     -> Result<Vec<u32>, Box<dyn Error>> {
        Ok(sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend,
            tensor,
            context,
        )?
        .into_iter()
        .map(f32::to_bits)
        .collect())
    };
    let image_bits = fixture_raw_bits(
        oracle
            .get("input_bits")
            .ok_or("DA3 input oracle is missing")?,
    )?;
    let image_values = image_bits
        .into_iter()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
    let image = ImageTensor::from_f32(&backend, &context, 1, 4, 4, 3, &image_values)?;
    let mut dtype_identities = BTreeSet::new();
    let mut f32_resource = None;
    let mut resources = Vec::new();
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let resource = NativeDepthAnything3Resource::from_reduced_fixture(
            &backend,
            deterministic_reduced_depth_anything_3_checkpoint(
                &backend,
                DepthAnything3FixtureProfile::Dpt,
                dtype,
                workspace_bytes,
                &context,
            )?,
            &context,
        )?;
        let dtype_name = match dtype {
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::F32 => "f32",
            _ => return Err("unexpected DA3 fixture dtype".into()),
        };
        assert_depth_anything_3_checkpoint_projection(
            &oracle,
            "dpt",
            dtype_name,
            &resource,
            &cancellation,
        )?;
        let resource = Arc::new(resource);
        dtype_identities.insert(resource.semantic_digest_sha256().to_owned());
        if dtype == DType::F32 {
            f32_resource = Some(resource.clone());
        }
        resources.push((dtype, resource));
    }
    assert_eq!(dtype_identities.len(), 3);
    let resource = f32_resource.ok_or("F32 DA3 fixture was not retained")?;
    for (dtype, resource) in &resources {
        let dtype_name = match dtype {
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::F32 => "f32",
            _ => return Err("unexpected DA3 execution dtype".into()),
        };
        let geometry = resource.execute(
            &backend,
            NativeDepthAnything3Invocation {
                image: &image,
                views_per_sample: 1,
                process_resolution: 4,
                resize_method: NativeDepthAnything3ResizeMethod::UpperBound,
                reference_strategy: NativeDepthAnything3ReferenceStrategy::First,
                use_ray_pose: false,
                ransac_seed: 17,
                extrinsics: None,
                intrinsics: None,
            },
            &context,
        )?;
        assert_eq!(geometry.depth.descriptor().shape(), &[1, 4, 4]);
        assert!(geometry.confidence.is_none());
        assert_eq!(
            geometry
                .sky
                .as_ref()
                .map(|value| value.descriptor().shape()),
            Some([1, 4, 4].as_slice())
        );
        for (actual, field) in [
            (&geometry.depth, "depth"),
            (
                geometry.sky.as_ref().ok_or("DA3 sky output is missing")?,
                "sky",
            ),
        ] {
            let pointer = if *dtype == DType::F32 {
                format!("/reduced_dpt/{field}/bits")
            } else {
                format!("/reduced_dpt/low_precision/{dtype_name}/{field}/bits")
            };
            assert_eq!(
                tensor_bits(actual, &context)?,
                fixture_raw_bits(
                    oracle
                        .pointer(&pointer)
                        .ok_or_else(|| format!("missing DA3 execution oracle {pointer}"))?,
                )?,
                "{dtype_name} projected execution diverged"
            );
        }
    }
    let nonsquare_values = fixture_raw_bits(
        oracle
            .pointer("/nonsquare_resize_projection/input_bits")
            .ok_or("DA3 non-square input oracle is missing")?,
    )?
    .into_iter()
    .map(f32::from_bits)
    .collect::<Vec<_>>();
    let nonsquare = ImageTensor::from_f32(&backend, &context, 1, 2, 4, 3, &nonsquare_values)?;
    for (name, resize_method) in [
        ("upper_bound", NativeDepthAnything3ResizeMethod::UpperBound),
        ("lower_bound", NativeDepthAnything3ResizeMethod::LowerBound),
    ] {
        let resized_geometry = resource.execute(
            &backend,
            NativeDepthAnything3Invocation {
                image: &nonsquare,
                views_per_sample: 1,
                process_resolution: 6,
                resize_method,
                reference_strategy: NativeDepthAnything3ReferenceStrategy::First,
                use_ray_pose: false,
                ransac_seed: 17,
                extrinsics: None,
                intrinsics: None,
            },
            &context,
        )?;
        assert_eq!(resized_geometry.depth.descriptor().shape(), &[1, 2, 4]);
        for (actual, field) in [
            (&resized_geometry.depth, "depth"),
            (
                resized_geometry
                    .sky
                    .as_ref()
                    .ok_or("DA3 non-square sky output is missing")?,
                "sky",
            ),
        ] {
            let pointer = format!("/nonsquare_resize_projection/cases/{name}/{field}/bits");
            assert_eq!(
                tensor_bits(actual, &context)?,
                fixture_raw_bits(
                    oracle
                        .pointer(&pointer)
                        .ok_or_else(|| format!("missing DA3 resize oracle {pointer}"))?,
                )?,
                "{name} preprocessing/final projection diverged"
            );
        }
    }
    let payload =
        NativeModelPayload::depth_anything_3_test_fixture(resource.clone(), &cancellation)?;
    assert_eq!(
        payload
            .depth_anything_3_resource()
            .map(|value| value.semantic_digest_sha256()),
        Some(resource.semantic_digest_sha256())
    );
    assert_eq!(
        payload.resident_parts()?.resident_bytes()?,
        payload.resident_bytes()
    );

    let multiview_values = fixture_raw_bits(
        oracle
            .get("multiview_input_bits")
            .ok_or("DA3 multiview input oracle is missing")?,
    )?
    .into_iter()
    .map(f32::from_bits)
    .collect::<Vec<_>>();
    let multiview = ImageTensor::from_f32(&backend, &context, 3, 4, 4, 3, &multiview_values)?;
    let dual = Arc::new(NativeDepthAnything3Resource::from_reduced_fixture(
        &backend,
        deterministic_reduced_depth_anything_3_checkpoint(
            &backend,
            DepthAnything3FixtureProfile::DualDpt,
            DType::F32,
            workspace_bytes,
            &context,
        )?,
        &context,
    )?);
    assert_depth_anything_3_checkpoint_projection(&oracle, "dualdpt", "f32", &dual, &cancellation)?;
    let dual_geometry = dual.execute(
        &backend,
        NativeDepthAnything3Invocation {
            image: &multiview,
            views_per_sample: 3,
            process_resolution: 4,
            resize_method: NativeDepthAnything3ResizeMethod::LowerBound,
            reference_strategy: NativeDepthAnything3ReferenceStrategy::SaddleSimRange,
            use_ray_pose: false,
            ransac_seed: 17,
            extrinsics: None,
            intrinsics: None,
        },
        &context,
    )?;
    assert_eq!(dual_geometry.depth.descriptor().shape(), &[3, 4, 4]);
    assert_eq!(
        dual_geometry
            .confidence
            .as_ref()
            .map(|value| value.descriptor().shape()),
        Some([3, 4, 4].as_slice())
    );
    assert!(dual_geometry.sky.is_none());
    assert_eq!(
        dual_geometry
            .extrinsics
            .as_ref()
            .map(|value| value.descriptor().shape()),
        Some([1, 3, 3, 4].as_slice())
    );
    assert_eq!(
        dual_geometry
            .intrinsics
            .as_ref()
            .map(|value| value.descriptor().shape()),
        Some([1, 3, 3, 3].as_slice())
    );
    assert_eq!(
        tensor_bits(&dual_geometry.depth, &context)?,
        fixture_raw_bits(
            oracle
                .pointer("/reduced_dualdpt/depth/bits")
                .ok_or("DA3 DualDPT depth oracle is missing")?,
        )?
    );
    assert_eq!(
        tensor_bits(
            dual_geometry
                .confidence
                .as_ref()
                .ok_or("DA3 DualDPT confidence output is missing")?,
            &context,
        )?,
        fixture_raw_bits(
            oracle
                .pointer("/reduced_dualdpt/confidence/bits")
                .ok_or("DA3 DualDPT confidence oracle is missing")?,
        )?
    );
    for (name, strategy) in [
        ("first", NativeDepthAnything3ReferenceStrategy::First),
        ("middle", NativeDepthAnything3ReferenceStrategy::Middle),
        (
            "saddle_balanced",
            NativeDepthAnything3ReferenceStrategy::SaddleBalanced,
        ),
        (
            "saddle_sim_range",
            NativeDepthAnything3ReferenceStrategy::SaddleSimRange,
        ),
    ] {
        let strategy_geometry = dual.execute(
            &backend,
            NativeDepthAnything3Invocation {
                image: &multiview,
                views_per_sample: 3,
                process_resolution: 4,
                resize_method: NativeDepthAnything3ResizeMethod::LowerBound,
                reference_strategy: strategy,
                use_ray_pose: false,
                ransac_seed: 17,
                extrinsics: None,
                intrinsics: None,
            },
            &context,
        )?;
        for (actual, field) in [
            (&strategy_geometry.depth, "depth"),
            (
                strategy_geometry
                    .confidence
                    .as_ref()
                    .ok_or("DA3 strategy confidence is missing")?,
                "confidence",
            ),
            (
                strategy_geometry
                    .extrinsics
                    .as_ref()
                    .ok_or("DA3 strategy extrinsics are missing")?,
                "camera/extrinsics",
            ),
            (
                strategy_geometry
                    .intrinsics
                    .as_ref()
                    .ok_or("DA3 strategy intrinsics are missing")?,
                "camera/intrinsics",
            ),
        ] {
            let pointer = format!("/reduced_dualdpt/reference_strategies/{name}/{field}/bits");
            assert_eq!(
                tensor_bits(actual, &context)?,
                fixture_raw_bits(
                    oracle
                        .pointer(&pointer)
                        .ok_or_else(|| format!("missing DA3 strategy oracle {pointer}"))?,
                )?,
                "reference strategy {name} did not preserve source reorder/restore"
            );
        }
    }
    assert_eq!(
        tensor_bits(
            dual_geometry
                .extrinsics
                .as_ref()
                .ok_or("DA3 decoded extrinsics are missing")?,
            &context,
        )?,
        fixture_raw_bits(
            oracle
                .pointer("/reduced_dualdpt/camera/extrinsics/bits")
                .ok_or("DA3 decoded extrinsics oracle is missing")?,
        )?
    );
    assert_eq!(
        tensor_bits(
            dual_geometry
                .intrinsics
                .as_ref()
                .ok_or("DA3 decoded intrinsics are missing")?,
            &context,
        )?,
        fixture_raw_bits(
            oracle
                .pointer("/reduced_dualdpt/camera/intrinsics/bits")
                .ok_or("DA3 decoded intrinsics oracle is missing")?,
        )?
    );

    let oracle_tensor = |pointer: &str, shape: &[u64]| -> Result<Tensor, Box<dyn Error>> {
        let values = fixture_raw_bits(
            oracle
                .pointer(pointer)
                .ok_or_else(|| format!("missing DA3 tensor oracle {pointer}"))?,
        )?
        .into_iter()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
        Ok(tensor_from_f32_with_context_exact_native(
            &backend,
            shape,
            &values,
            DType::F32,
            backend.device(),
            &context,
        )?)
    };
    let camera_extrinsics = oracle_tensor("/camera_inputs/extrinsics/bits", &[1, 3, 3, 4])?;
    let camera_intrinsics = oracle_tensor("/camera_inputs/intrinsics/bits", &[1, 3, 3, 3])?;
    let supplied_camera_geometry = dual
        .execute(
            &backend,
            NativeDepthAnything3Invocation {
                image: &multiview,
                views_per_sample: 3,
                process_resolution: 4,
                resize_method: NativeDepthAnything3ResizeMethod::LowerBound,
                reference_strategy: NativeDepthAnything3ReferenceStrategy::SaddleSimRange,
                use_ray_pose: false,
                ransac_seed: 17,
                extrinsics: Some(&camera_extrinsics),
                intrinsics: Some(&camera_intrinsics),
            },
            &context,
        )
        .map_err(|error| -> Box<dyn Error> {
            format!("DA3 supplied-camera execution failed: {error}").into()
        })?;
    for (actual, pointer) in [
        (
            &supplied_camera_geometry.depth,
            "/reduced_dualdpt/supplied_camera_depth/bits",
        ),
        (
            supplied_camera_geometry
                .confidence
                .as_ref()
                .ok_or("DA3 supplied-camera confidence is missing")?,
            "/reduced_dualdpt/supplied_camera_confidence/bits",
        ),
        (
            supplied_camera_geometry
                .extrinsics
                .as_ref()
                .ok_or("DA3 supplied-camera extrinsics are missing")?,
            "/reduced_dualdpt/supplied_camera/extrinsics/bits",
        ),
        (
            supplied_camera_geometry
                .intrinsics
                .as_ref()
                .ok_or("DA3 supplied-camera intrinsics are missing")?,
            "/reduced_dualdpt/supplied_camera/intrinsics/bits",
        ),
    ] {
        assert_eq!(
            tensor_bits(actual, &context)?,
            fixture_raw_bits(
                oracle
                    .pointer(pointer)
                    .ok_or_else(|| format!("missing DA3 supplied-camera oracle {pointer}"))?,
            )?,
            "{pointer}: supplied-camera output diverged"
        );
    }

    let ransac_address = RngStreamAddress::new(
        "task390",
        "fixture",
        "depth-anything-3-ray-pose",
        0,
        "reduced-ray-pose",
        0,
        0,
        RetryRngPolicy::Replay,
    )?;
    let ray_context = ExecutionContext {
        stream: context.stream,
        scratch: context.scratch.clone(),
        rng_phase: Some(&ransac_address),
        cancellation: context.cancellation,
    };
    let (ray_geometry, ray_trace) = dual.execute_with_test_trace(
        &backend,
        NativeDepthAnything3Invocation {
            image: &multiview,
            views_per_sample: 3,
            process_resolution: 4,
            resize_method: NativeDepthAnything3ResizeMethod::LowerBound,
            reference_strategy: NativeDepthAnything3ReferenceStrategy::SaddleSimRange,
            use_ray_pose: true,
            ransac_seed: 17,
            extrinsics: None,
            intrinsics: None,
        },
        &ray_context,
    )?;
    for (actual, pointer) in [
        (
            &ray_trace.raw_ray,
            "/reduced_dualdpt/ray_pose_trace/pre_geometry_ray/bits",
        ),
        (
            &ray_trace.raw_ray_confidence,
            "/reduced_dualdpt/ray_pose_trace/pre_geometry_confidence/bits",
        ),
    ] {
        let document_pointer = pointer
            .strip_suffix("/bits")
            .ok_or("DA3 ray trace pointer has no bits suffix")?;
        let mut expected_shape = oracle
            .pointer(&format!("{document_pointer}/shape"))
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing DA3 ray trace shape {document_pointer}"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| format!("invalid DA3 ray trace shape {document_pointer}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        expected_shape.insert(0, 1);
        assert_eq!(actual.descriptor().shape(), expected_shape);
        require_exact_fixture_bits(
            pointer,
            &tensor_bits(actual, &ray_context)?,
            &fixture_raw_bits(
                oracle
                    .pointer(pointer)
                    .ok_or_else(|| format!("missing DA3 ray trace oracle {pointer}"))?,
            )?,
        )?;
    }
    let raw_ray_values =
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend,
            &ray_trace.raw_ray,
            &ray_context,
        )?;
    let raw_confidence_values =
        sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend,
            &ray_trace.raw_ray_confidence,
            &ray_context,
        )?;
    let admission = oracle
        .pointer("/reduced_dualdpt/ray_pose_trace/admission")
        .ok_or("DA3 ray-pose admission trace is missing")?;
    assert_eq!(
        admission.get("candidate_count").and_then(Value::as_u64),
        Some(76)
    );
    assert_eq!(
        admission
            .pointer("/confidence_ordering/source")
            .and_then(Value::as_str),
        Some("torch.argsort(descending=True, stable=False-default)")
    );
    assert_eq!(
        admission
            .pointer("/confidence_ordering/native_owner")
            .and_then(Value::as_str),
        Some("argsort_with_context_exact_native(descending=true, stable=false)")
    );
    assert_eq!(
        admission
            .pointer("/confidence_ordering/tied_order_pinned")
            .and_then(Value::as_bool),
        Some(false)
    );
    let admission_views = admission
        .get("views")
        .and_then(Value::as_array)
        .ok_or("DA3 ray-pose admission views are missing")?;
    assert_eq!(admission_views.len(), 3);
    for (view, expected) in admission_views.iter().enumerate() {
        let confidence_start = view * 256;
        let confidence_end = confidence_start + 256;
        let confidence = raw_confidence_values
            .get(confidence_start..confidence_end)
            .ok_or("DA3 ray confidence view is unavailable")?;
        assert_eq!(
            expected.get("confidence_count").and_then(Value::as_u64),
            Some(256)
        );
        assert_eq!(
            expected
                .get("confidence_all_finite_positive")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            expected
                .get("confidence_all_bit_distinct")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert!(
            confidence
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        let mut sorted_confidence = confidence.to_vec();
        sorted_confidence.sort_by(f32::total_cmp);
        assert!(
            sorted_confidence
                .windows(2)
                .all(|pair| pair[0].to_bits() != pair[1].to_bits())
        );
        let minimum_ulp_gap = sorted_confidence
            .windows(2)
            .map(|pair| pair[1].to_bits() - pair[0].to_bits())
            .min()
            .ok_or("DA3 ray confidence ULP gap is unavailable")?;
        let minimum_value_gap = sorted_confidence
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .min_by(f32::total_cmp)
            .ok_or("DA3 ray confidence value gap is unavailable")?;
        assert_eq!(
            u64::from(minimum_ulp_gap),
            expected
                .get("minimum_adjacent_ulp_gap")
                .and_then(Value::as_u64)
                .ok_or("DA3 ray confidence ULP gap oracle is missing")?
        );
        assert_eq!(
            u64::from(minimum_value_gap.to_bits()),
            expected
                .get("minimum_adjacent_value_gap_bits")
                .and_then(Value::as_u64)
                .ok_or("DA3 ray confidence value gap oracle is missing")?
        );
        let ray_start = view * 256 * 6;
        let ray_end = ray_start + 256 * 6;
        let ray_z = raw_ray_values
            .get(ray_start..ray_end)
            .ok_or("DA3 ray view is unavailable")?
            .chunks_exact(6)
            .map(|ray| {
                ray.get(2)
                    .copied()
                    .ok_or("DA3 ray z lane is unavailable")
                    .map(f32::abs)
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(ray_z.len(), 256);
        assert_eq!(
            expected.get("ray_z_all_valid").and_then(Value::as_bool),
            Some(true)
        );
        assert!(
            ray_z
                .iter()
                .all(|value| value.is_finite() && *value > 1.0e-4)
        );
        let minimum_ray_z = ray_z
            .into_iter()
            .min_by(f32::total_cmp)
            .ok_or("DA3 ray z minimum is unavailable")?;
        assert_eq!(
            u64::from(minimum_ray_z.to_bits()),
            expected
                .get("minimum_ray_z_abs_bits")
                .and_then(Value::as_u64)
                .ok_or("DA3 ray z minimum oracle is missing")?
        );
    }
    let expected_samples = oracle
        .pointer("/reduced_dualdpt/ray_pose_trace/samples")
        .and_then(Value::as_array)
        .ok_or("DA3 RANSAC samples are missing")?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or("DA3 RANSAC sample row is invalid")?
                .iter()
                .map(|value| value.as_u64().ok_or("DA3 RANSAC sample is invalid"))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(ray_trace.ransac_samples, expected_samples);
    let index_rows_sha256 = |domain: &str, rows: &[Vec<u64>]| {
        let mut digest = Sha256::new();
        digest.update((domain.len() as u64).to_le_bytes());
        digest.update(domain.as_bytes());
        digest.update((rows.len() as u64).to_le_bytes());
        for row in rows {
            digest.update((row.len() as u64).to_le_bytes());
            for value in row {
                digest.update(value.to_le_bytes());
            }
        }
        format!("{:x}", digest.finalize())
    };
    let samples_domain = oracle
        .pointer("/reduced_dualdpt/ray_pose_trace/samples_sha256_domain")
        .and_then(Value::as_str)
        .ok_or("DA3 RANSAC sample SHA domain is missing")?;
    assert_eq!(
        index_rows_sha256(samples_domain, &ray_trace.ransac_samples),
        oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/samples_sha256")
            .and_then(Value::as_str)
            .ok_or("DA3 RANSAC sample SHA is missing")?
    );
    assert_eq!(
        oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/profile_version")
            .and_then(Value::as_u64),
        Some(2)
    );
    assert_eq!(
        oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/algorithm")
            .and_then(Value::as_str),
        Some("mt19937")
    );
    assert_eq!(
        oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/seed")
            .and_then(Value::as_u64),
        Some(17)
    );
    for (pointer, expected) in [
        ("workflow", "task390"),
        ("attempt", "fixture"),
        ("node", "depth-anything-3-ray-pose"),
        ("phase", "reduced-ray-pose"),
        ("retry_policy", "replay"),
        ("device_kind", "cpu"),
    ] {
        assert_eq!(
            oracle
                .pointer(&format!(
                    "/reduced_dualdpt/ray_pose_trace/address/{pointer}"
                ))
                .and_then(Value::as_str),
            Some(expected)
        );
    }
    let expected_views = oracle
        .pointer("/reduced_dualdpt/ray_pose_trace/views")
        .and_then(Value::as_array)
        .ok_or("DA3 RANSAC view traces are missing")?;
    assert_eq!(ray_trace.ransac_views.len(), expected_views.len());
    for (actual, expected) in ray_trace.ransac_views.iter().zip(expected_views) {
        let candidates = expected
            .get("candidate_indices")
            .and_then(Value::as_array)
            .ok_or("DA3 RANSAC candidates are missing")?
            .iter()
            .map(|value| value.as_u64().ok_or("DA3 RANSAC candidate is invalid"))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(actual.candidate_indices, candidates);
        assert_eq!(
            actual.best_iteration,
            expected
                .get("best_iteration")
                .and_then(Value::as_u64)
                .ok_or("DA3 RANSAC best iteration is missing")?
        );
        assert_eq!(
            actual.best_inliers.len() as u64,
            expected
                .get("best_inlier_count")
                .and_then(Value::as_u64)
                .ok_or("DA3 RANSAC inlier count is missing")?
        );
        assert_eq!(
            u64::from(actual.best_score_bits),
            expected
                .get("best_score_bits")
                .and_then(Value::as_u64)
                .ok_or("DA3 RANSAC best score is missing")?
        );
        assert_eq!(
            actual.fallback,
            expected
                .get("fallback")
                .and_then(Value::as_bool)
                .ok_or("DA3 RANSAC fallback disposition is missing")?
        );
        for (phase, actual_values) in [
            (
                "normalized_homography_bits",
                actual.normalized_homography.as_slice(),
            ),
            (
                "homography_post_sign_bits",
                actual.signed_homography.as_slice(),
            ),
            ("rotation_bits", actual.rotation.as_slice()),
            ("lower_bits", actual.lower.as_slice()),
            ("c2w_pre_inverse_bits", actual.c2w_pre_inverse.as_slice()),
        ] {
            let expected_values = fixture_raw_bits(
                expected
                    .get(phase)
                    .ok_or_else(|| format!("DA3 RANSAC {phase} is missing"))?,
            )?
            .into_iter()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
            assert_eq!(actual_values.len(), expected_values.len());
            for (lane, (actual_value, expected_value)) in actual_values
                .iter()
                .copied()
                .zip(expected_values)
                .enumerate()
            {
                let tolerance = 2.5e-3_f32.max(expected_value.abs() * 2.5e-3);
                assert!(
                    (actual_value - expected_value).abs() <= tolerance,
                    "DA3 RANSAC view phase {phase}[{lane}]: {actual_value} != {expected_value}"
                );
            }
        }
        let inlier_domain = expected
            .get("best_inliers_sha256_domain")
            .and_then(Value::as_str)
            .ok_or("DA3 RANSAC inlier SHA domain is missing")?;
        assert_eq!(
            index_rows_sha256(inlier_domain, std::slice::from_ref(&actual.best_inliers)),
            expected
                .get("best_inliers_sha256")
                .and_then(Value::as_str)
                .ok_or("DA3 RANSAC inlier SHA is missing")?
        );
    }
    assert_eq!(
        tensor_bits(&ray_geometry.depth, &ray_context)?,
        fixture_raw_bits(
            oracle
                .pointer("/reduced_dualdpt/ray_depth/bits")
                .ok_or("DA3 ray depth oracle is missing")?,
        )?
    );
    assert_eq!(
        tensor_bits(
            ray_geometry
                .confidence
                .as_ref()
                .ok_or("DA3 ray confidence is missing")?,
            &ray_context,
        )?,
        fixture_raw_bits(
            oracle
                .pointer("/reduced_dualdpt/ray_confidence/bits")
                .ok_or("DA3 ray confidence oracle is missing")?,
        )?
    );
    for (actual, pointer) in [
        (
            ray_geometry
                .extrinsics
                .as_ref()
                .ok_or("DA3 ray extrinsics are missing")?,
            "/reduced_dualdpt/ray_extrinsics/bits",
        ),
        (
            ray_geometry
                .intrinsics
                .as_ref()
                .ok_or("DA3 ray intrinsics are missing")?,
            "/reduced_dualdpt/ray_intrinsics/bits",
        ),
    ] {
        let expected = fixture_raw_bits(
            oracle
                .pointer(pointer)
                .ok_or_else(|| format!("missing DA3 ray geometry oracle {pointer}"))?,
        )?
        .into_iter()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
        let actual = sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
            &backend,
            actual,
            &ray_context,
        )?;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            let tolerance = 2.5e-3_f32.max(expected.abs() * 2.5e-3);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{pointer}: {actual} != {expected}"
            );
        }
    }

    let public_geometry_identity = |geometry: &sim_model::NativeDepthAnything3Geometry,
                                    context: &ExecutionContext<'_>|
     -> Result<String, Box<dyn Error>> {
        let mut hasher = Sha256::new();
        for tensor in [
            Some(&geometry.depth),
            geometry.confidence.as_ref(),
            geometry.sky.as_ref(),
            geometry.extrinsics.as_ref(),
            geometry.intrinsics.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            for value in sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
                &backend,
                tensor,
                context,
            )? {
                hasher.update(value.to_le_bytes());
            }
        }
        Ok(format!("{:x}", hasher.finalize()))
    };
    let mutations = oracle
        .get("full_path_mutations")
        .and_then(serde_json::Value::as_object)
        .ok_or("DA3 full-path mutations are missing")?;
    for (name, mutation) in mutations {
        let execution = mutation
            .get("execution")
            .and_then(serde_json::Value::as_str)
            .ok_or("DA3 mutation execution is missing")?;
        let profile = if execution == "dpt" {
            DepthAnything3FixtureProfile::Dpt
        } else {
            DepthAnything3FixtureProfile::DualDpt
        };
        let mut checkpoint = deterministic_reduced_depth_anything_3_checkpoint(
            &backend,
            profile,
            DType::F32,
            workspace_bytes,
            &context,
        )?;
        mutate_reduced_depth_anything_3_checkpoint(
            &backend,
            &mut checkpoint,
            DepthAnything3FixtureMutation {
                state_key: mutation
                    .get("state_key")
                    .and_then(serde_json::Value::as_str)
                    .ok_or("DA3 mutation state key is missing")?,
                lane: usize::try_from(
                    mutation
                        .get("lane")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("DA3 mutation lane is missing")?,
                )?,
                delta: f32::from_bits(u32::try_from(
                    mutation
                        .get("delta_bits")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("DA3 mutation delta is missing")?,
                )?),
            },
            &context,
        )?;
        let mutated =
            NativeDepthAnything3Resource::from_reduced_fixture(&backend, checkpoint, &context)?;
        let (input, views, camera, use_ray, execution_context) = match execution {
            "dpt" => (&image, 1, None, false, &context),
            "camera" => (
                &multiview,
                3,
                Some((&camera_extrinsics, &camera_intrinsics)),
                false,
                &context,
            ),
            "ray" => (&multiview, 3, None, true, &ray_context),
            "dual" => (&multiview, 3, None, false, &context),
            _ => return Err(format!("unknown DA3 mutation execution {execution}").into()),
        };
        let invocation = NativeDepthAnything3Invocation {
            image: input,
            views_per_sample: views,
            process_resolution: 4,
            resize_method: NativeDepthAnything3ResizeMethod::LowerBound,
            reference_strategy: NativeDepthAnything3ReferenceStrategy::SaddleSimRange,
            use_ray_pose: use_ray,
            ransac_seed: 17,
            extrinsics: camera.map(|camera| camera.0),
            intrinsics: camera.map(|camera| camera.1),
        };
        let (geometry, mutation_trace) = if execution == "ray" {
            let (geometry, trace) =
                mutated.execute_with_test_trace(&backend, invocation, execution_context)?;
            (geometry, Some(trace))
        } else {
            (
                mutated.execute(&backend, invocation, execution_context)?,
                None,
            )
        };
        let actual_identity = public_geometry_identity(&geometry, execution_context)?;
        if execution != "ray" {
            assert_eq!(
                Some(actual_identity.as_str()),
                mutation
                    .get("output_identity_sha256")
                    .and_then(serde_json::Value::as_str),
                "DA3 mutation {name} diverged from the source-equation oracle"
            );
        }
        let changes_output = mutation
            .get("changes_output")
            .and_then(serde_json::Value::as_bool)
            .ok_or("DA3 mutation disposition is missing")?;
        if execution == "ray" {
            assert!(changes_output, "{name}");
            let trace = mutation_trace
                .as_ref()
                .ok_or("DA3 ray mutation trace is unavailable")?;
            let confidence = geometry
                .confidence
                .as_ref()
                .ok_or("DA3 ray mutation confidence is unavailable")?;
            for (phase, actual) in [
                ("depth", &geometry.depth),
                ("confidence", confidence),
                ("raw_ray", &trace.raw_ray),
                ("raw_ray_confidence", &trace.raw_ray_confidence),
            ] {
                require_exact_fixture_bits(
                    &format!("DA3 ray mutation {phase}"),
                    &tensor_bits(actual, execution_context)?,
                    &fixture_raw_bits(
                        mutation
                            .get(phase)
                            .and_then(|value| value.get("bits"))
                            .ok_or("DA3 ray mutation exact phase is missing")?,
                    )?,
                )?;
            }
            assert_ne!(
                tensor_bits(&trace.raw_ray, execution_context)?,
                tensor_bits(&ray_trace.raw_ray, &ray_context)?,
                "DA3 ray mutation must change the production raw ray"
            );
            assert_eq!(
                mutation
                    .get("source_raw_ray_changed")
                    .and_then(serde_json::Value::as_bool),
                Some(true)
            );
            assert_eq!(
                mutation
                    .get("source_geometry_changed")
                    .and_then(serde_json::Value::as_bool),
                Some(true)
            );
            let expected_trace = mutation
                .get("ransac_trace")
                .ok_or("DA3 ray mutation RANSAC trace is missing")?;
            let expected_samples = expected_trace
                .get("samples")
                .and_then(Value::as_array)
                .ok_or("DA3 ray mutation samples are missing")?
                .iter()
                .map(|row| {
                    row.as_array()
                        .ok_or("DA3 ray mutation sample row is invalid")?
                        .iter()
                        .map(|value| value.as_u64().ok_or("DA3 ray mutation sample is invalid"))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(trace.ransac_samples, expected_samples);
            let expected_views = expected_trace
                .get("views")
                .and_then(Value::as_array)
                .ok_or("DA3 ray mutation view traces are missing")?;
            assert_eq!(trace.ransac_views.len(), expected_views.len());
            for (actual, expected) in trace.ransac_views.iter().zip(expected_views) {
                let candidates = expected
                    .get("candidate_indices")
                    .and_then(Value::as_array)
                    .ok_or("DA3 ray mutation candidates are missing")?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .ok_or("DA3 ray mutation candidate is invalid")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                assert_eq!(actual.candidate_indices, candidates);
                assert_eq!(
                    actual.best_iteration,
                    expected
                        .get("best_iteration")
                        .and_then(Value::as_u64)
                        .ok_or("DA3 ray mutation best iteration is missing")?
                );
                assert_eq!(
                    actual.best_inliers.len() as u64,
                    expected
                        .get("best_inlier_count")
                        .and_then(Value::as_u64)
                        .ok_or("DA3 ray mutation inlier count is missing")?
                );
                assert_eq!(
                    u64::from(actual.best_score_bits),
                    expected
                        .get("best_score_bits")
                        .and_then(Value::as_u64)
                        .ok_or("DA3 ray mutation best score is missing")?
                );
                assert_eq!(
                    actual.fallback,
                    expected
                        .get("fallback")
                        .and_then(Value::as_bool)
                        .ok_or("DA3 ray mutation fallback is missing")?
                );
                let inlier_domain = expected
                    .get("best_inliers_sha256_domain")
                    .and_then(Value::as_str)
                    .ok_or("DA3 ray mutation inlier domain is missing")?;
                assert_eq!(
                    index_rows_sha256(inlier_domain, std::slice::from_ref(&actual.best_inliers)),
                    expected
                        .get("best_inliers_sha256")
                        .and_then(Value::as_str)
                        .ok_or("DA3 ray mutation inlier SHA is missing")?
                );
                for (phase, actual_values) in [
                    (
                        "normalized_homography_bits",
                        actual.normalized_homography.as_slice(),
                    ),
                    (
                        "homography_post_sign_bits",
                        actual.signed_homography.as_slice(),
                    ),
                    ("rotation_bits", actual.rotation.as_slice()),
                    ("lower_bits", actual.lower.as_slice()),
                    ("c2w_pre_inverse_bits", actual.c2w_pre_inverse.as_slice()),
                ] {
                    let expected_values = fixture_raw_bits(
                        expected
                            .get(phase)
                            .ok_or("DA3 ray mutation continuous phase is missing")?,
                    )?
                    .into_iter()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>();
                    assert_eq!(actual_values.len(), expected_values.len());
                    for (lane, (actual, expected)) in actual_values
                        .iter()
                        .copied()
                        .zip(expected_values)
                        .enumerate()
                    {
                        let tolerance = 2.5e-3_f32.max(expected.abs() * 2.5e-3);
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "DA3 ray mutation {phase}[{lane}]: {actual} != {expected}"
                        );
                    }
                }
            }
            for (actual, field) in [
                (
                    geometry
                        .extrinsics
                        .as_ref()
                        .ok_or("DA3 mutated ray extrinsics are missing")?,
                    "extrinsics",
                ),
                (
                    geometry
                        .intrinsics
                        .as_ref()
                        .ok_or("DA3 mutated ray intrinsics are missing")?,
                    "intrinsics",
                ),
            ] {
                let actual = sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native(
                    &backend,
                    actual,
                    execution_context,
                )?;
                let expected = fixture_raw_bits(
                    mutation
                        .pointer(&format!("/geometry/{field}/bits"))
                        .ok_or("DA3 ray mutation geometry is missing")?,
                )?
                .into_iter()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
                assert_eq!(actual.len(), expected.len());
                for (lane, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                    let tolerance = 2.5e-3_f32.max(expected.abs() * 2.5e-3);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "DA3 ray mutation {field}[{lane}]: {actual} != {expected}"
                    );
                }
            }
            let baseline_actual_identity = public_geometry_identity(&ray_geometry, &ray_context)?;
            assert_ne!(actual_identity, baseline_actual_identity, "{name}");
        } else {
            let baseline_identity = match execution {
                "dpt" => oracle.pointer("/reduced_dpt/output_identity_sha256"),
                "camera" => {
                    oracle.pointer("/reduced_dualdpt/supplied_camera_output_identity_sha256")
                }
                _ => oracle.pointer("/reduced_dualdpt/output_identity_sha256"),
            }
            .and_then(serde_json::Value::as_str)
            .ok_or("DA3 mutation baseline identity is missing")?;
            assert_eq!(
                actual_identity != baseline_identity,
                changes_output,
                "{name}"
            );
        }
        if !changes_output {
            let baseline_digest = match profile {
                DepthAnything3FixtureProfile::Dpt => resource.semantic_digest_sha256(),
                DepthAnything3FixtureProfile::DualDpt => dual.semantic_digest_sha256(),
            };
            assert_ne!(
                mutated.semantic_digest_sha256(),
                baseline_digest,
                "forward-unused retained state must still change resource identity"
            );
        }
    }
    Ok(())
}

#[test]
fn generated_registry_frontend_compiler_and_worker_dispatch_share_one_path()
-> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    registry.validate_comprehensive_bindings()?;
    let early = native_image_registry_projection()?;
    assert!(
        early
            .descriptors()
            .all(|(class_type, _)| registry.descriptor(class_type).is_some())
    );

    let frontend = generated_native_frontend_descriptors(None)?;
    assert_eq!(frontend.len(), registry.descriptor_len());
    assert!(
        registry
            .descriptors()
            .all(|(class_type, _)| frontend.contains_key(class_type))
    );

    let workflow = sim_runtime::WorkflowFormatDocument::parse(WORKFLOW_FIXTURE)?;
    let submission = graph_to_prompt(&workflow, &frontend, "task367-generated-native")?;
    let mut plan = compile_generated_native_prompt(submission, None)?;
    plan.prompt_id = PromptId(Uuid::from_u128(0x3671));
    assert_eq!(plan.nodes.len(), 5);
    assert_eq!(
        plan.output_nodes,
        vec![NodeId("4".to_owned()), NodeId("5".to_owned())]
    );

    let input_bytes = encode_png_frame(
        &[0.25, 0.5, 0.75],
        1,
        1,
        1,
        3,
        0,
        &BTreeMap::new(),
        PngLimits::default(),
    )?;
    let worker_plan = NativeImageWorkerPlan::new(
        plan.clone(),
        BTreeMap::from([("fixture.png".to_owned(), input_bytes)]),
        true,
        0,
    )?;
    let directory = tempfile::tempdir()?;
    let worker_directory = directory.path().join("worker");
    fs::create_dir(&worker_directory)?;
    let mut launch = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        PROFILE_ID,
        WorkerId(Uuid::from_u128(0x3672)),
        NATIVE_IMAGE_REGISTRY_VERSION,
        1024 * 1024 * 1024,
    );
    launch.working_directory = Some(worker_directory);
    launch.environment = vec![("PATH".to_owned(), String::new())];
    launch.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(60),
        maximum_automatic_restarts: 0,
        restart_backoff: Duration::from_millis(1),
    };
    let mut supervisor = smol::block_on(RuntimeSupervisor::start(launch))
        .map_err(|error| format!("generated registry worker startup: {error}"))?;
    assert_eq!(supervisor.snapshot().health, WorkerHealth::BackendReady);
    smol::block_on(supervisor.execute(
        plan.prompt_id,
        AttemptId(Uuid::from_u128(0x3673)),
        serde_json::to_vec(&worker_plan)?,
    ))
    .map_err(|error| format!("generated registry worker dispatch: {error}"))?;
    let terminal = smol::block_on(await_terminal_worker_event(&supervisor))
        .map_err(|error| format!("generated registry worker completion: {error}"))?;
    let NativeImageWorkerEvent::Completed { result } = terminal else {
        return Err(format!("generated plan did not complete in the worker: {terminal:?}").into());
    };
    assert_eq!(result.report.state, AttemptState::Succeeded);
    assert_eq!(result.executed_node_count, 5);
    smol::block_on(supervisor.shutdown())?;
    Ok(())
}

#[test]
fn portable_values_dynamic_ports_and_attempt_handles_fail_closed() -> Result<(), Box<dyn Error>> {
    let image_type = NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?;
    let union = NativeTypeUnion::new([
        NativeValueType::Primitive(NativePrimitiveType::Integer),
        NativeValueType::Primitive(NativePrimitiveType::String),
        NativeValueType::Handle(image_type.clone()),
    ])?;
    let descriptor = NativeNodeDescriptor {
        schema_version: sim_nodes::NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
        class_type: "Task367PortableProbe".to_owned(),
        implementation_version: "1".to_owned(),
        source_schema: Some(sim_nodes::NativeDescriptorSchemaMetadata::synthetic(
            ["value".to_owned()],
            [sim_nodes::NativeDynamicSchemaMetadata::compatibility(
                "value_{index}",
                1,
                1,
                8,
                sim_nodes::NativeInputSchemaMetadata::compatibility("value", "ANY"),
            )],
            std::iter::empty(),
        )),
        inputs: vec![NativeInputDescriptor {
            name: "value".to_owned(),
            accepted_types: union.clone(),
            required: true,
            hidden: false,
            lazy: true,
            cardinality: NativePortCardinality::Mapped,
            allows_literal: true,
        }],
        dynamic_inputs: vec![sim_runtime::NativeDynamicInputDescriptor {
            name_template: "value_{index}".to_owned(),
            start_index: 1,
            minimum_count: 1,
            maximum_count: 8,
            input: NativeInputDescriptor {
                name: "value".to_owned(),
                accepted_types: union.clone(),
                required: false,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::List,
                allows_literal: true,
            },
        }],
        outputs: Vec::new(),
        output_node: true,
        effect: sim_runtime::NativeEffectClass::WritesArtifact,
        cache: sim_runtime::NativeCachePolicy::Never,
    };
    descriptor.validate()?;

    let scalar = NativeValue::Primitive {
        value: NativePrimitive::Integer(7),
    };
    let list = NativeValue::List {
        values: vec![
            scalar.clone(),
            NativeValue::Primitive {
                value: NativePrimitive::String("seven".to_owned()),
            },
        ],
    };
    assert!(union.accepts(&scalar));
    list.validate()?;
    let restored: NativeValue = serde_json::from_slice(&serde_json::to_vec(&list)?)?;
    assert_eq!(restored, list);

    let first_generation = NativeHandleStoreGeneration::with_capacities(4, 1024)?;
    let attempt_id = AttemptId(Uuid::from_u128(0x3674));
    let first_store = first_generation.handle_store_for_attempt(attempt_id);
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let image_cancellation = CancellationToken::default();
    let image_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &image_cancellation,
    );
    let image = ImageTensor::from_f32(&backend, &image_context, 1, 1, 1, 3, &[0.25, 0.5, 0.75])?;
    let payload = NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
        NativeTensorRole::Image,
        image,
    )?));
    let handle = first_store.publish(payload.clone(), &CancellationToken::default())?;
    let handle_value = NativeValue::Handle {
        value: handle.clone(),
    };
    assert!(union.accepts(&handle_value));
    assert!(
        first_store
            .resolve(&handle, &image_type, &CancellationToken::default())
            .is_ok()
    );

    let recovered_generation = NativeHandleStoreGeneration::with_capacities(4, 1024)?;
    let recovered_store = recovered_generation.handle_store_for_attempt(attempt_id);
    assert!(matches!(
        recovered_store.resolve(&handle, &image_type, &CancellationToken::default()),
        Err(NativeHandleStoreError::WrongStore | NativeHandleStoreError::WrongGeneration)
    ));
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        first_store.publish(payload, &cancellation),
        Err(NativeHandleStoreError::Cancelled)
    ));

    let effect = NativePreparedEffectRequest::checked(
        Uuid::from_u128(0x3674),
        Uuid::from_u128(0x3675),
        NativePreparedEffectKind::Output,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )?;
    effect.validate()?;
    NativeNodeOutcome::Values {
        outputs: vec![list],
        ui: Some(json!({"task": 367})),
        effects: vec![effect],
    }
    .validate()?;
    NativeNodeOutcome::Blocked {
        reason: "provider activation required".to_owned(),
    }
    .validate()?;
    assert!(
        NativeNodeOutcome::Expansion {
            prompt: sim_types::ApiPrompt::default(),
            output_node: NodeId("missing".to_owned()),
        }
        .validate()
        .is_err()
    );
    let failure = NativeNodeFailure {
        code: "task367_failure".to_owned(),
        message: "deterministic failure".to_owned(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    };
    failure.validate()?;
    assert_eq!(
        serde_json::from_slice::<NativeNodeFailure>(&serde_json::to_vec(&failure)?)?,
        failure
    );
    Ok(())
}

#[test]
fn source_structured_values_keep_resolved_handles_typed_across_recovery()
-> Result<(), Box<dyn Error>> {
    let schema = built_in_source_schema("ResizeImageMaskNode")?;
    let resize_type = schema
        .inputs
        .iter()
        .find(|input| input.schema.name == "resize_type")
        .ok_or("ResizeImageMaskNode has no resize_type input")?;
    let match_size = resize_type
        .schema
        .structured_options()?
        .into_iter()
        .find(|option| option.selector == "match size")
        .ok_or("ResizeImageMaskNode has no match size option")?;
    assert!(match_size.fields.iter().any(|field| {
        field.path.as_slice() == ["match"]
            && field.schema.source_type_names.as_slice() == ["IMAGE", "MASK"]
    }));

    let generation = NativeHandleStoreGeneration::with_capacities(4, 1024)?;
    let attempt_id = AttemptId(Uuid::from_u128(0x3760));
    let store = generation.handle_store_for_attempt(attempt_id);
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let image = ImageTensor::from_f32(&backend, &context, 1, 1, 1, 3, &[0.1, 0.2, 0.3])?;
    let payload = NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
        NativeTensorRole::Image,
        image,
    )?));
    let handle = store.publish(payload, &cancellation)?;
    let structured = NativeStructuredValue::checked(
        "COMFY_DYNAMICCOMBO_V3",
        BTreeMap::from([
            (
                "resize_type".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String("match size".to_owned()),
                },
            ),
            (
                "crop".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String("center".to_owned()),
                },
            ),
            (
                "match".to_owned(),
                NativeValue::Handle {
                    value: handle.clone(),
                },
            ),
        ]),
    )?;
    let value = structured.into_native_value();
    let expected = NativeTypeUnion::new([NativeValueType::NamedPreservedUnknown(
        "COMFY_DYNAMICCOMBO_V3".to_owned(),
    )])?;
    assert!(expected.accepts(&value));
    let restored: NativeValue = serde_json::from_slice(&serde_json::to_vec(&value)?)?;
    let restored = NativeStructuredValue::from_native_value(&restored)?
        .ok_or("structured value lost its typed representation")?;
    assert_eq!(
        restored.get("match"),
        Some(&NativeValue::Handle {
            value: handle.clone(),
        })
    );
    let image_type = NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?;
    store.resolve(&handle, &image_type, &cancellation)?;

    let recovered =
        NativeHandleStoreGeneration::with_capacities(4, 1024)?.handle_store_for_attempt(attempt_id);
    assert!(matches!(
        recovered.resolve(&handle, &image_type, &cancellation),
        Err(NativeHandleStoreError::WrongStore | NativeHandleStoreError::WrongGeneration)
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        store.resolve(&handle, &image_type, &cancelled),
        Err(NativeHandleStoreError::Cancelled)
    ));
    Ok(())
}

#[test]
fn task403_conditioning_publication_is_distinct_rollback_safe_and_restart_safe()
-> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(1024 * 1024)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let tensor = tensor_from_f32_with_context_exact_native(
        &backend,
        &[1, 2],
        &[1.0, 2.0],
        DType::F32,
        DeviceId::CPU,
        &context,
    )?;
    let identity = ConditioningIdentity::new(
        "task403-publication",
        sd15_model_family_identity()?,
        sd15_latent_format_identity()?,
    )?;
    let entry = ConditioningEntry::checked(
        "entry",
        ConditioningValue::regular(tensor)?,
        ConditioningEntryOptions::default(),
    )?;
    let base = ConditioningSet::checked(identity, vec![entry], &cancellation)?;
    let metadata = |digest: &str| {
        ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::Control,
            ConditioningMetadataValue::Control(ConditioningControlBinding::checked(digest, false)?),
        )])
    };
    let positive = base.with_entry_metadata(
        vec![metadata(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )?],
        &cancellation,
    )?;
    let negative = base.with_entry_metadata(
        vec![metadata(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )?],
        &cancellation,
    )?;
    assert_ne!(positive.digest(), negative.digest());

    let generation = NativeHandleStoreGeneration::with_capacities(2, 4 * 1024 * 1024)?;
    let attempt = AttemptId(Uuid::from_u128(0x403));
    let store = generation.handle_store_for_attempt(attempt);
    let positive_handle = store.publish(
        NativeStoredPayload::Conditioning(Arc::new(positive.clone())),
        &cancellation,
    )?;
    let negative_handle = store.publish(
        NativeStoredPayload::Conditioning(Arc::new(negative.clone())),
        &cancellation,
    )?;
    assert_ne!(
        positive_handle.digest_sha256(),
        negative_handle.digest_sha256()
    );
    let persisted: NativeOpaqueHandle =
        serde_json::from_slice(&serde_json::to_vec(&positive_handle)?)?;
    let conditioning_type = NativeHandleType::new(NativeHandleKind::Conditioning, "CONDITIONING")?;
    store.resolve(&persisted, &conditioning_type, &cancellation)?;

    assert!(matches!(
        store.publish(
            NativeStoredPayload::Conditioning(Arc::new(base)),
            &cancellation,
        ),
        Err(NativeHandleStoreError::Rejected(_))
    ));
    store.revoke(&negative_handle, &cancellation)?;
    store.revoke(&positive_handle, &cancellation)?;
    assert!(generation.is_empty());

    let recovered = store.publish(
        NativeStoredPayload::Conditioning(Arc::new(positive.clone())),
        &cancellation,
    )?;
    let restarted = generation
        .fresh_generation()?
        .handle_store_for_attempt(attempt);
    assert!(matches!(
        restarted.resolve(&recovered, &conditioning_type, &cancellation),
        Err(NativeHandleStoreError::WrongGeneration)
    ));
    let fresh = restarted.publish(
        NativeStoredPayload::Conditioning(Arc::new(positive)),
        &cancellation,
    )?;
    restarted.resolve(&fresh, &conditioning_type, &cancellation)?;
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let before = generation.len();
    assert!(matches!(
        store.publish(
            NativeStoredPayload::Conditioning(Arc::new(negative)),
            &cancelled,
        ),
        Err(NativeHandleStoreError::Cancelled)
    ));
    assert_eq!(generation.len(), before);
    Ok(())
}

#[test]
fn sdpose_model_resource_handle_is_sealed_alias_aware_and_restart_safe()
-> Result<(), Box<dyn Error>> {
    let payload = reduced_sdpose_stored_payload()?;
    payload.validate()?;
    let handle_type = payload.handle_type()?;
    assert_eq!(handle_type.kind, NativeHandleKind::Model);
    assert_eq!(handle_type.type_id, "MODEL");
    let byte_capacity = payload.resident_bytes()?;
    let generation = NativeHandleStoreGeneration::with_capacities(3, byte_capacity)?;
    let attempt_id = AttemptId(Uuid::from_u128(0x4030));
    let store = generation.handle_store_for_attempt(attempt_id);
    let cancellation = CancellationToken::default();
    let first = store.publish(payload.clone(), &cancellation)?;
    let first_bytes = generation.resident_bytes();
    assert_eq!(first_bytes, byte_capacity);
    let second = store.publish(payload, &cancellation)?;
    assert_eq!(generation.len(), 2);
    assert_eq!(generation.resident_bytes(), first_bytes);

    let resolved = store.resolve(&first, &handle_type, &cancellation)?;
    let NativeStoredPayload::Model(model) = resolved.as_ref() else {
        return Err("SDPose MODEL handle resolved to another stored payload kind".into());
    };
    assert!(model.model_payload().sdpose_model_resource().is_some());
    assert_eq!(Some(model.digest_sha256()), first.digest_sha256());

    let distinct = reduced_sdpose_stored_payload()?;
    assert_eq!(distinct.digest_sha256(), model.digest_sha256());
    assert!(matches!(
        store.publish(distinct, &cancellation),
        Err(NativeHandleStoreError::Rejected(message)) if message.contains("capacity is exhausted")
    ));
    assert_eq!(generation.len(), 2);
    assert_eq!(generation.resident_bytes(), first_bytes);

    let forged = NativeOpaqueHandle::new(
        handle_type.clone(),
        first.store_identity(),
        first.identifier(),
        first.generation(),
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned()),
    )?;
    assert!(matches!(
        store.resolve(&forged, &handle_type, &cancellation),
        Err(NativeHandleStoreError::DigestMismatch)
    ));

    let restarted = NativeHandleStoreGeneration::with_capacities(2, byte_capacity)?
        .handle_store_for_attempt(attempt_id);
    assert!(matches!(
        restarted.resolve(&first, &handle_type, &cancellation),
        Err(NativeHandleStoreError::WrongStore | NativeHandleStoreError::WrongGeneration)
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let before_len = generation.len();
    let before_bytes = generation.resident_bytes();
    assert!(matches!(
        store.publish(reduced_sdpose_stored_payload()?, &cancelled),
        Err(NativeHandleStoreError::Cancelled)
    ));
    assert_eq!(generation.len(), before_len);
    assert_eq!(generation.resident_bytes(), before_bytes);
    assert_ne!(first.identifier(), second.identifier());
    Ok(())
}

#[test]
fn frame_interpolation_resource_handle_is_concrete_alias_aware_and_restart_safe()
-> Result<(), Box<dyn Error>> {
    let payload = reduced_frame_interpolation_stored_payload()?;
    payload.validate()?;
    let handle_type = payload.handle_type()?;
    assert_eq!(handle_type.kind, NativeHandleKind::Model);
    assert_eq!(handle_type.type_id, "INTERP_MODEL");
    let byte_capacity = payload.resident_bytes()?;
    let generation = NativeHandleStoreGeneration::with_capacities(3, byte_capacity)?;
    let attempt_id = AttemptId(Uuid::from_u128(0x4080));
    let store = generation.handle_store_for_attempt(attempt_id);
    let cancellation = CancellationToken::default();
    let first = store.publish(payload.clone(), &cancellation)?;
    let first_bytes = generation.resident_bytes();
    let second = store.publish(payload, &cancellation)?;
    assert_eq!(generation.len(), 2);
    assert_eq!(generation.resident_bytes(), first_bytes);

    let resolved = store.resolve(&first, &handle_type, &cancellation)?;
    let NativeStoredPayload::Model(model) = resolved.as_ref() else {
        return Err("INTERP_MODEL handle resolved to another stored payload kind".into());
    };
    assert!(
        model
            .model_payload()
            .frame_interpolation_resource()
            .is_some()
    );
    assert_eq!(Some(model.digest_sha256()), first.digest_sha256());

    let distinct = reduced_frame_interpolation_stored_payload()?;
    assert_eq!(distinct.digest_sha256(), model.digest_sha256());
    assert!(matches!(
        store.publish(distinct, &cancellation),
        Err(NativeHandleStoreError::Rejected(message)) if message.contains("capacity is exhausted")
    ));
    assert_eq!(generation.len(), 2);
    assert_eq!(generation.resident_bytes(), first_bytes);

    let forged = NativeOpaqueHandle::new(
        handle_type.clone(),
        first.store_identity(),
        first.identifier(),
        first.generation(),
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned()),
    )?;
    assert!(matches!(
        store.resolve(&forged, &handle_type, &cancellation),
        Err(NativeHandleStoreError::DigestMismatch)
    ));
    let restarted = NativeHandleStoreGeneration::with_capacities(2, byte_capacity)?
        .handle_store_for_attempt(attempt_id);
    assert!(matches!(
        restarted.resolve(&first, &handle_type, &cancellation),
        Err(NativeHandleStoreError::WrongStore | NativeHandleStoreError::WrongGeneration)
    ));
    assert_ne!(first.identifier(), second.identifier());
    Ok(())
}

#[test]
fn native_moge_resources_execute_source_oracle_and_publish_typed_geometry()
-> Result<(), Box<dyn Error>> {
    let oracle: Value = serde_json::from_str(MOGE_RESOURCE_ORACLE)?;
    let manifest: Value = serde_json::from_str(MOGE_RESOURCE_FIXTURE)?;
    let provenance: Value = serde_json::from_str(MOGE_RESOURCE_PROVENANCE)?;
    let sha256 = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    assert_eq!(
        manifest["oracle_sha256"],
        sha256(MOGE_RESOURCE_ORACLE.as_bytes())
    );
    assert_eq!(
        manifest["generator_sha256"],
        sha256(MOGE_RESOURCE_GENERATOR.as_bytes())
    );
    assert_eq!(
        manifest["source_graph_sha256"],
        sha256(MOGE_RESOURCE_SOURCE_GRAPH.as_bytes())
    );
    assert_eq!(provenance["oracle_sha256"], manifest["oracle_sha256"]);
    assert_eq!(
        provenance["command"],
        "PYTHONDONTWRITEBYTECODE=1 python3 crates/sim_test_support/fixtures/models/moge-resource-foundation/generate_oracle.py --check"
    );
    let canonical_exp_max_ulp = u32::try_from(
        oracle["numeric_contract"]["canonical_exp_max_ulp"]
            .as_u64()
            .ok_or("MoGe exponential ULP contract is missing")?,
    )?;
    let max_plus_one = u32::try_from(
        oracle["numeric_contract"]["max_plus_one_must_reject"]
            .as_u64()
            .ok_or("MoGe exponential max+1 discriminator is missing")?,
    )?;
    let derived_geometry_distinct_nonzero_ulp_bounds =
        oracle["numeric_contract"]["derived_geometry_distinct_nonzero_ulp_bounds"]
            .as_array()
            .ok_or("MoGe derived geometry ULP contracts are missing")?
            .iter()
            .map(|value| -> Result<u32, Box<dyn Error>> {
                Ok(u32::try_from(value.as_u64().ok_or(
                    "MoGe derived geometry ULP contract is not an integer",
                )?)?)
            })
            .collect::<Result<Vec<u32>, Box<dyn Error>>>()?;
    assert_eq!(
        derived_geometry_distinct_nonzero_ulp_bounds,
        [1, 2, 4, 6, 8, 10, 11, 14, 16, 18]
    );
    for bound in derived_geometry_distinct_nonzero_ulp_bounds {
        let accepted = 0x3f80_0000u32
            .checked_add(bound)
            .ok_or("MoGe admitted ULP witness overflowed")?;
        let rejected = accepted
            .checked_add(1)
            .ok_or("MoGe rejected ULP witness overflowed")?;
        assert_eq!(
            require_fixture_ulp_bound(
                &format!("moge/derived-geometry-{bound}-admitted"),
                &[accepted],
                &[0x3f80_0000],
                bound,
            )?,
            bound
        );
        assert!(
            require_fixture_ulp_bound(
                &format!("moge/derived-geometry-{bound}-plus-one"),
                &[rejected],
                &[0x3f80_0000],
                bound,
            )
            .is_err()
        );
    }
    assert_eq!(max_plus_one, canonical_exp_max_ulp + 1);
    assert!(
        require_fixture_ulp_bound(
            "moge/exp-max-plus-one",
            &[0x3f80_0000 + max_plus_one],
            &[0x3f80_0000],
            canonical_exp_max_ulp,
        )
        .is_err()
    );

    let workspace_bytes = 256 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(workspace_bytes)?,
        &cancellation,
    );
    let exp_probe_inputs =
        fixture_raw_bits(&oracle["numeric_contract"]["canonical_exp_probe_input_bits"])?;
    let exp_probe_expected =
        fixture_raw_bits(&oracle["numeric_contract"]["source_exp_probe_output_bits"])?;
    assert!(!exp_probe_inputs.is_empty());
    assert_eq!(exp_probe_inputs.len(), exp_probe_expected.len());
    let exp_probe_tensor = tensor_from_f32_with_context_exact_native(
        &backend,
        &[u64::try_from(exp_probe_inputs.len())?],
        &exp_probe_inputs
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>(),
        DType::F32,
        sim_tensor::DeviceId::CPU,
        &context,
    )?;
    let exp_probe_actual = tensor_to_f32_with_context_exact_native(
        &backend,
        &exp_with_context_exact_native(&backend, &exp_probe_tensor, &context)?,
        &context,
    )?
    .into_iter()
    .map(f32::to_bits)
    .collect::<Vec<_>>();
    require_fixture_ulp_bound(
        "moge/canonical-exp-admitted-range",
        &exp_probe_actual,
        &exp_probe_expected,
        canonical_exp_max_ulp,
    )?;
    let input_values = fixture_raw_bits(&oracle["input_bits"])?
        .into_iter()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
    let image = ImageTensor::from_f32(&backend, &context, 1, 4, 4, 3, &input_values)?;
    let tensor_bits = |tensor: &Tensor| -> Result<Vec<u32>, Box<dyn Error>> {
        Ok(
            tensor_to_f32_with_context_exact_native(&backend, tensor, &context)?
                .into_iter()
                .map(f32::to_bits)
                .collect(),
        )
    };
    let compare_output = |phase: &str,
                          geometry: &sim_model::NativeMogeGeometry,
                          expected: &Value|
     -> Result<(), Box<dyn Error>> {
        assert_eq!(
            geometry.image().tensor().storage_id(),
            image.tensor().storage_id(),
            "{phase}/source-image"
        );
        for (name, tensor) in [
            ("depth", geometry.depth()),
            ("points", geometry.points()),
            ("intrinsics", geometry.intrinsics()),
            ("mask", geometry.mask()),
        ] {
            let tensor = tensor.ok_or_else(|| format!("{phase}/{name}: missing tensor"))?;
            let expected_shape = expected[format!("{name}_shape")]
                .as_array()
                .ok_or_else(|| format!("{phase}/{name}: missing shape"))?
                .iter()
                .map(|value| value.as_u64().ok_or("fixture shape is not an integer"))
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(
                tensor.descriptor().shape(),
                expected_shape,
                "{phase}/{name}"
            );
            assert_eq!(
                tensor.descriptor().dtype(),
                if name == "mask" {
                    DType::Bool
                } else {
                    DType::F32
                },
                "{phase}/{name} dtype"
            );
            assert_eq!(
                tensor.descriptor().device(),
                sim_tensor::DeviceId::CPU,
                "{phase}/{name} device"
            );
            assert_eq!(
                tensor.descriptor().stream(),
                context.stream,
                "{phase}/{name} stream"
            );
            let actual = tensor_bits(tensor)?;
            assert!(
                actual
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .all(f32::is_finite),
                "{phase}/{name} unexpectedly contains a non-finite value"
            );
            let bound = if name == "mask" {
                None
            } else {
                Some(u32::try_from(
                    expected["output_ulp_bounds"][name]
                        .as_u64()
                        .ok_or_else(|| format!("{phase}/{name}: ULP bound is missing"))?,
                )?)
            };
            let expected = fixture_raw_bits(&expected[format!("{name}_bits")])?;
            if let Some(bound) = bound {
                if bound == 0 {
                    require_exact_fixture_bits(&format!("{phase}/{name}"), &actual, &expected)?;
                } else {
                    // The source bound remains an upper limit when a platform
                    // produces a smaller rounding error than the fixture host.
                    require_fixture_ulp_bound(
                        &format!("{phase}/{name}"),
                        &actual,
                        &expected,
                        bound,
                    )?;
                }
            } else {
                require_exact_fixture_bits(&format!("{phase}/{name}"), &actual, &expected)?;
            }
        }
        if let Some(tensor) = geometry.normal() {
            assert_eq!(tensor.descriptor().dtype(), DType::F32);
            assert_eq!(tensor.descriptor().device(), sim_tensor::DeviceId::CPU);
            assert_eq!(tensor.descriptor().stream(), context.stream);
            assert!(
                tensor_bits(tensor)?
                    .into_iter()
                    .map(f32::from_bits)
                    .all(f32::is_finite),
                "{phase}/normal unexpectedly contains a non-finite value"
            );
        }
        match (geometry.normal(), expected["normal_bits"].is_null()) {
            (None, true) => {}
            (Some(tensor), false) => {
                let bound = u32::try_from(
                    expected["output_ulp_bounds"]["normal"]
                        .as_u64()
                        .ok_or_else(|| format!("{phase}/normal: ULP bound is missing"))?,
                )?;
                let actual = tensor_bits(tensor)?;
                let expected = fixture_raw_bits(&expected["normal_bits"])?;
                if bound == 0 {
                    require_exact_fixture_bits(&format!("{phase}/normal"), &actual, &expected)?;
                } else {
                    require_fixture_ulp_bound(
                        &format!("{phase}/normal"),
                        &actual,
                        &expected,
                        bound,
                    )?;
                }
            }
            _ => return Err(format!("{phase}/normal: option disposition differs").into()),
        }
        Ok(())
    };

    for (profile_name, profile) in [
        ("v1", MogeFixtureProfile::V1),
        ("v2", MogeFixtureProfile::V2),
    ] {
        for (dtype_name, dtype) in [
            ("f32", DType::F32),
            ("f16", DType::F16),
            ("bf16", DType::Bf16),
        ] {
            let resource = Arc::new(NativeMogeResource::from_reduced_fixture(
                &backend,
                deterministic_reduced_moge_checkpoint(
                    &backend,
                    profile,
                    dtype,
                    workspace_bytes,
                    &context,
                )?,
                &context,
            )?);
            resource.validate(&cancellation)?;
            let reconstructed = resource.reconstruct_checkpoint(&cancellation)?;
            let reconstructed_resource =
                NativeMogeResource::from_reduced_fixture(&backend, reconstructed, &context)?;
            assert_eq!(
                reconstructed_resource.semantic_digest_sha256(),
                resource.semantic_digest_sha256()
            );
            let payload = NativeModelPayload::moge_test_fixture(resource.clone(), &cancellation)?;
            payload.validate()?;
            assert_eq!(
                payload.identity().role(),
                NativeModelResourceRole::MogeModel
            );
            assert_eq!(payload.identity().format(), "zed-native-moge-v1");
            assert!(payload.moge_resource().is_some());
            let geometry = resource.execute(
                &backend,
                NativeMogeInvocation {
                    image: &image,
                    resolution_level: 9,
                    fov_x_degrees: Some(60.0),
                    force_projection: true,
                    apply_mask: false,
                    apply_metric_scale: true,
                },
                &context,
            )?;
            compare_output(
                &format!("{profile_name}/{dtype_name}"),
                &geometry,
                &oracle["profiles"][profile_name][dtype_name],
            )?;
        }
    }

    for (mutation_name, mutation) in oracle["mutations"]
        .as_object()
        .ok_or("MoGe mutations are not an object")?
    {
        let profile_name = mutation["profile"]
            .as_str()
            .ok_or("MoGe mutation profile is missing")?;
        let profile = match profile_name {
            "v1" => MogeFixtureProfile::V1,
            "v2" => MogeFixtureProfile::V2,
            _ => return Err("unknown MoGe mutation profile".into()),
        };
        let mut checkpoint = deterministic_reduced_moge_checkpoint(
            &backend,
            profile,
            DType::F32,
            workspace_bytes,
            &context,
        )?;
        mutate_reduced_moge_checkpoint(
            &backend,
            &mut checkpoint,
            MogeFixtureMutation {
                state_key: mutation["state_key"]
                    .as_str()
                    .ok_or("MoGe mutation state key is missing")?,
                lane: usize::try_from(
                    mutation["lane"]
                        .as_u64()
                        .ok_or("MoGe mutation lane is missing")?,
                )?,
                delta: mutation["delta"]
                    .as_f64()
                    .ok_or("MoGe mutation delta is missing")? as f32,
            },
            &context,
        )?;
        let resource = NativeMogeResource::from_reduced_fixture(&backend, checkpoint, &context)?;
        let geometry = resource.execute(
            &backend,
            NativeMogeInvocation {
                image: &image,
                resolution_level: 9,
                fov_x_degrees: Some(60.0),
                force_projection: true,
                apply_mask: false,
                apply_metric_scale: true,
            },
            &context,
        )?;
        compare_output(
            &format!("mutation/{mutation_name}"),
            &geometry,
            &mutation["output"],
        )?;
    }
    Ok(())
}

async fn await_terminal_worker_event(
    supervisor: &RuntimeSupervisor,
) -> Result<NativeImageWorkerEvent, Box<dyn Error>> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("generated native worker dispatch timed out".into());
        }
        let envelope = supervisor.next_event(remaining).await?;
        if let WorkerMessage::Event { event } = envelope.message
            && let Ok(event) = postcard::from_bytes::<NativeImageWorkerEvent>(&event)
            && matches!(
                event,
                NativeImageWorkerEvent::Completed { .. }
                    | NativeImageWorkerEvent::BackendUnavailable { .. }
                    | NativeImageWorkerEvent::Failed { .. }
            )
        {
            return Ok(event);
        }
    }
}

const MODEL_RESOURCE_SERVICE_ORACLE: &[u8] =
    include_bytes!("../fixtures/models/conditioning-auxiliary-resource-foundation/oracle.json");

#[derive(Deserialize)]
struct ModelResourceServiceOracle {
    style: ModelResourceDtypeProfiles,
    photomaker: ModelResourceDtypeProfiles,
    gligen: ModelResourceGligenProfile,
}

#[derive(Deserialize)]
struct ModelResourceDtypeProfiles {
    dtypes: BTreeMap<String, ModelResourceState>,
}

#[derive(Deserialize)]
struct ModelResourceState {
    state: Vec<ModelResourceStateEntry>,
}

#[derive(Deserialize)]
struct ModelResourceGligenProfile {
    state: Vec<ModelResourceStateEntry>,
}

#[derive(Deserialize)]
struct ModelResourceStateEntry {
    key: String,
    shape: Vec<u64>,
}

#[derive(Clone)]
struct ModelResourceFixture {
    manifest: WorkerModelSourceManifest,
    tensors: Vec<ModelResourceTensorFixture>,
    auxiliary_artifacts: BTreeMap<u32, Vec<u8>>,
}

#[derive(Clone)]
enum ModelResourceTensorFixture {
    Zero(u64),
    Bytes(Vec<u8>),
}

fn serve_model_resource_transport(
    host: sim_runtime::NativeModelSourceTransportHost,
    fixtures: Arc<BTreeMap<(String, String), Arc<ModelResourceFixture>>>,
) -> Result<(usize, usize, usize), WorkerModelSourceError> {
    let mut active = BTreeMap::<Uuid, Vec<Arc<ModelResourceFixture>>>::new();
    let mut open_count = 0;
    let mut read_count = 0;
    let mut close_count = 0;
    while let Ok(call) = smol::block_on(host.receive()) {
        let request = call.request().clone();
        let response = match request.operation {
            WorkerModelSourceOperation::Open {
                folder_category,
                source_names,
            } => {
                open_count += 1;
                if request.call_ordinal != 1 || source_names.is_empty() {
                    return Err(WorkerModelSourceError::InvalidOrder);
                }
                let selected = source_names
                    .into_iter()
                    .enumerate()
                    .map(|(source_ordinal, source_name)| {
                        let fixture = fixtures
                            .get(&(folder_category.clone(), source_name))
                            .cloned()
                            .ok_or(WorkerModelSourceError::InvalidSourceSelection)?;
                        let mut fixture = (*fixture).clone();
                        fixture.manifest.source_ordinal = u32::try_from(source_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?;
                        Ok(Arc::new(fixture))
                    })
                    .collect::<Result<Vec<_>, WorkerModelSourceError>>()?;
                let opened_sources = selected
                    .iter()
                    .map(|fixture| fixture.manifest.clone())
                    .collect();
                active.insert(request.context.session_id, selected);
                WorkerModelSourceResponse::Opened(WorkerModelSourceOpened::checked(
                    request.context.session_id,
                    request.call_ordinal,
                    request.context.ordered_source_identity_sha256.clone(),
                    opened_sources,
                )?)
            }
            WorkerModelSourceOperation::Read {
                source_ordinal,
                tensor_ordinal,
                byte_offset,
                byte_length,
            } => {
                read_count += 1;
                let selected = active
                    .get(&request.context.session_id)
                    .ok_or(WorkerModelSourceError::Closed)?;
                let fixture = selected
                    .get(
                        usize::try_from(source_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
                let tensor = fixture
                    .tensors
                    .get(
                        usize::try_from(tensor_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidTensorOrdinal)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidTensorOrdinal)?;
                let start =
                    usize::try_from(byte_offset).map_err(|_| WorkerModelSourceError::RangeLimit)?;
                let end = start
                    .checked_add(
                        usize::try_from(byte_length)
                            .map_err(|_| WorkerModelSourceError::RangeLimit)?,
                    )
                    .ok_or(WorkerModelSourceError::RangeLimit)?;
                let bytes = match tensor {
                    ModelResourceTensorFixture::Zero(tensor_byte_length) => {
                        if end
                            > usize::try_from(*tensor_byte_length)
                                .map_err(|_| WorkerModelSourceError::RangeLimit)?
                        {
                            return Err(WorkerModelSourceError::RangeLimit);
                        }
                        vec![
                            0;
                            usize::try_from(byte_length)
                                .map_err(|_| WorkerModelSourceError::RangeLimit)?
                        ]
                    }
                    ModelResourceTensorFixture::Bytes(bytes) => bytes
                        .get(start..end)
                        .ok_or(WorkerModelSourceError::RangeLimit)?
                        .to_vec(),
                };
                WorkerModelSourceResponse::Chunk(WorkerModelSourceChunk::checked(
                    request.context.session_id,
                    request.call_ordinal,
                    source_ordinal,
                    tensor_ordinal,
                    byte_offset,
                    bytes,
                )?)
            }
            WorkerModelSourceOperation::Close => {
                close_count += 1;
                active
                    .remove(&request.context.session_id)
                    .ok_or(WorkerModelSourceError::Closed)?;
                WorkerModelSourceResponse::Closed(WorkerModelSourceClosed::checked(
                    request.context.session_id,
                    request.call_ordinal,
                )?)
            }
            WorkerModelSourceOperation::ReadArtifact {
                source_ordinal,
                artifact_ordinal,
                artifact_sha256,
                byte_offset,
                byte_length,
            } => {
                read_count += 1;
                let selected = active
                    .get(&request.context.session_id)
                    .ok_or(WorkerModelSourceError::Closed)?;
                let fixture = selected
                    .get(
                        usize::try_from(source_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
                let bytes = fixture
                    .auxiliary_artifacts
                    .get(&artifact_ordinal)
                    .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
                let artifact = fixture
                    .manifest
                    .artifacts
                    .get(
                        usize::try_from(artifact_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
                if artifact.sha256 != artifact_sha256 {
                    return Err(WorkerModelSourceError::DigestMismatch);
                }
                let start =
                    usize::try_from(byte_offset).map_err(|_| WorkerModelSourceError::RangeLimit)?;
                let end = start
                    .checked_add(
                        usize::try_from(byte_length)
                            .map_err(|_| WorkerModelSourceError::RangeLimit)?,
                    )
                    .ok_or(WorkerModelSourceError::RangeLimit)?;
                let chunk = bytes
                    .get(start..end)
                    .ok_or(WorkerModelSourceError::RangeLimit)?
                    .to_vec();
                WorkerModelSourceResponse::ArtifactChunk(WorkerModelSourceArtifactChunk::checked(
                    request.context.session_id,
                    request.call_ordinal,
                    source_ordinal,
                    artifact_ordinal,
                    artifact_sha256,
                    byte_offset,
                    chunk,
                )?)
            }
        };
        let call_id = call.call_id();
        smol::block_on(call.respond(call_id, response))?;
    }
    if active.is_empty() {
        Ok((open_count, read_count, close_count))
    } else {
        Err(WorkerModelSourceError::Closed)
    }
}

fn model_resource_fixture(
    category: &str,
    state: &[ModelResourceStateEntry],
) -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let typed_state = state
        .iter()
        .map(|entry| (entry.key.clone(), entry.shape.clone(), DType::F32))
        .collect::<Vec<_>>();
    model_resource_typed_fixture(category, &typed_state)
}

fn model_resource_typed_fixture(
    category: &str,
    state: &[(String, Vec<u64>, DType)],
) -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let mut tensors = Vec::new();
    let mut tensor_manifests = Vec::new();
    let mut aggregate_tensor_bytes = 0_u64;
    let mut maximum_tensor_bytes = 0_u64;
    for (name, shape, data_type) in state {
        let element_count = shape.iter().try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or("fixture shape overflow")
        })?;
        let byte_length = element_count
            .checked_mul(data_type.byte_width())
            .ok_or("fixture tensor byte count overflow")?;
        let byte_offset = aggregate_tensor_bytes;
        aggregate_tensor_bytes = aggregate_tensor_bytes
            .checked_add(byte_length)
            .ok_or("fixture aggregate tensor byte count overflow")?;
        maximum_tensor_bytes = maximum_tensor_bytes.max(byte_length);
        tensors.push(ModelResourceTensorFixture::Zero(byte_length));
        tensor_manifests.push(WorkerModelSourceTensor {
            name: name.clone(),
            data_type: format!("{data_type:?}"),
            shape: shape.clone(),
            artifact_ordinal: 0,
            byte_offset,
            byte_length,
        });
    }
    let artifact_sha256 = WorkerSha256Digest::new(format!(
        "{:x}",
        Sha256::digest(format!("task400:{category}:artifact"))
    ))?;
    let manifest = WorkerModelSourceManifest {
        source_ordinal: 0,
        model_identity_sha256: WorkerSha256Digest::new(format!(
            "{:x}",
            Sha256::digest(format!("task400:{category}:model"))
        ))?,
        artifacts: vec![WorkerModelSourceArtifact {
            sha256: artifact_sha256,
            byte_size: aggregate_tensor_bytes,
            format: WorkerModelSourceFormat::Safetensors,
            nested_state_disposition: WorkerModelSourceNestedStateDisposition::Flat,
        }],
        tensors: tensor_manifests,
        aggregate_tensor_bytes,
        maximum_read_bytes: maximum_tensor_bytes.max(512 * 1024),
        parser_limits: WorkerModelSourceParserLimits {
            version: 1,
            manifest_bytes: 64 * 1024 * 1024,
            maximum_depth: 64,
            maximum_tensors: 10_000,
            maximum_tensor_bytes: maximum_tensor_bytes.max(512 * 1024),
            maximum_aggregate_tensor_bytes: aggregate_tensor_bytes.max(512 * 1024),
            maximum_name_bytes: 16 * 1024,
            maximum_archive_entries: 10_000,
            maximum_metadata_values: 1_000_000,
        },
    };
    Ok(Arc::new(ModelResourceFixture {
        manifest,
        tensors,
        auxiliary_artifacts: BTreeMap::new(),
    }))
}

fn auraflow_model_resource_fixture(
    source_form: NativeModelRecipeSourceForm,
) -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let state = auraflow_recipe_fixture_state_schema(source_form)?
        .into_iter()
        .map(|(key, shape)| ModelResourceStateEntry { key, shape })
        .collect::<Vec<_>>();
    model_resource_fixture(
        match source_form {
            NativeModelRecipeSourceForm::Checkpoint => "auraflow-checkpoint",
            NativeModelRecipeSourceForm::Standalone => "auraflow-standalone",
        },
        &state,
    )
}

fn model_resource_fixture_with_auxiliary(
    fixture: Arc<ModelResourceFixture>,
    format: WorkerModelSourceFormat,
    bytes: Vec<u8>,
) -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let mut fixture = (*fixture).clone();
    let artifact_ordinal = u32::try_from(fixture.manifest.artifacts.len())?;
    let sha256 = WorkerSha256Digest::new(format!("{:x}", Sha256::digest(&bytes)))?;
    fixture.manifest.artifacts.push(WorkerModelSourceArtifact {
        sha256,
        byte_size: u64::try_from(bytes.len())?,
        format,
        nested_state_disposition: WorkerModelSourceNestedStateDisposition::Flat,
    });
    fixture.auxiliary_artifacts.insert(artifact_ordinal, bytes);
    Ok(Arc::new(fixture))
}

fn clip_recipe_state(prefix: &str) -> Vec<ModelResourceStateEntry> {
    let mut state = vec![
        (
            "text_model.embeddings.token_embedding.weight",
            vec![49_408, 2],
        ),
        (
            "text_model.embeddings.position_embedding.weight",
            vec![77, 2],
        ),
        ("text_model.final_layer_norm.weight", vec![2]),
        ("text_model.final_layer_norm.bias", vec![2]),
        ("text_projection.weight", vec![2, 2]),
    ]
    .into_iter()
    .map(|(key, shape)| ModelResourceStateEntry {
        key: format!("{prefix}{key}"),
        shape,
    })
    .collect::<Vec<_>>();
    for layer in 0..2 {
        for (name, shape) in [
            ("layer_norm1.weight", vec![2]),
            ("layer_norm1.bias", vec![2]),
            ("self_attn.q_proj.weight", vec![2, 2]),
            ("self_attn.q_proj.bias", vec![2]),
            ("self_attn.k_proj.weight", vec![2, 2]),
            ("self_attn.k_proj.bias", vec![2]),
            ("self_attn.v_proj.weight", vec![2, 2]),
            ("self_attn.v_proj.bias", vec![2]),
            ("self_attn.out_proj.weight", vec![2, 2]),
            ("self_attn.out_proj.bias", vec![2]),
            ("layer_norm2.weight", vec![2]),
            ("layer_norm2.bias", vec![2]),
            ("mlp.fc1.weight", vec![2, 2]),
            ("mlp.fc1.bias", vec![2]),
            ("mlp.fc2.weight", vec![2, 2]),
            ("mlp.fc2.bias", vec![2]),
        ] {
            state.push(ModelResourceStateEntry {
                key: format!("{prefix}text_model.encoder.layers.{layer}.{name}"),
                shape,
            });
        }
    }
    state
}

fn clip_json_tokenizer_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or("workspace root is unavailable")?;
    let fixture = root.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1");
    let vocabulary: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.join("vocab.json"))?)?;
    let merges = fs::read_to_string(fixture.join("merges.txt"))?
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with("#version"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    Ok(serde_json::to_vec(&json!({
        "model": { "vocab": vocabulary, "merges": merges }
    }))?)
}

fn clip_model_resource_fixture(
    identity: &str,
    projection_dimension: u64,
    prefix: &str,
) -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let state = clip_recipe_state(prefix);
    let fixture = model_resource_fixture(identity, &state)?;
    let configuration = serde_json::to_vec(&json!({
        "model_type": "clip",
        "hidden_size": 2,
        "projection_dim": projection_dimension,
    }))?;
    let fixture = model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::JsonConfig,
        configuration,
    )?;
    model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::JsonTokenizer,
        clip_json_tokenizer_fixture()?,
    )
}

fn open_clip_recipe_state(prefix: &str) -> Vec<ModelResourceStateEntry> {
    let mut state = vec![
        ("token_embedding.weight", vec![49_408, 2]),
        ("positional_embedding", vec![77, 2]),
        ("ln_final.weight", vec![2]),
        ("ln_final.bias", vec![2]),
        ("text_projection", vec![2, 2]),
    ]
    .into_iter()
    .map(|(key, shape)| ModelResourceStateEntry {
        key: format!("{prefix}{key}"),
        shape,
    })
    .collect::<Vec<_>>();
    for layer in 0..2 {
        for (name, shape) in [
            ("ln_1.weight", vec![2]),
            ("ln_1.bias", vec![2]),
            ("attn.in_proj_weight", vec![6, 2]),
            ("attn.in_proj_bias", vec![6]),
            ("attn.out_proj.weight", vec![2, 2]),
            ("attn.out_proj.bias", vec![2]),
            ("ln_2.weight", vec![2]),
            ("ln_2.bias", vec![2]),
            ("mlp.c_fc.weight", vec![2, 2]),
            ("mlp.c_fc.bias", vec![2]),
            ("mlp.c_proj.weight", vec![2, 2]),
            ("mlp.c_proj.bias", vec![2]),
        ] {
            state.push(ModelResourceStateEntry {
                key: format!("{prefix}transformer.resblocks.{layer}.{name}"),
                shape,
            });
        }
    }
    state
}

fn mixed_sdxl_checkpoint_resource_fixture() -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let mut state = clip_recipe_state("conditioner.embedders.0.transformer.");
    state.extend(open_clip_recipe_state("conditioner.embedders.1.model."));
    let fixture = model_resource_fixture("clip-checkpoint-sdxl", &state)?;
    let fixture = model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::ClipConfiguration { target_ordinal: 0 },
        serde_json::to_vec(&json!({
            "model_type": "clip",
            "hidden_size": 2,
            "projection_dim": 768,
        }))?,
    )?;
    let fixture = model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::ClipTokenizerJson { target_ordinal: 0 },
        clip_json_tokenizer_fixture()?,
    )?;
    let fixture = model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::ClipConfiguration { target_ordinal: 1 },
        serde_json::to_vec(&json!({
            "model_type": "clip",
            "hidden_size": 2,
            "projection_dim": 1_280,
        }))?,
    )?;
    model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::ClipTokenizerJson { target_ordinal: 1 },
        clip_json_tokenizer_fixture()?,
    )
}

fn metadata_backed_ltx_vae_fixture() -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let fixture = model_resource_typed_fixture(
        "vae-ltx-metadata",
        &[(
            "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight".to_owned(),
            vec![512],
            DType::F32,
        )],
    )?;
    let configuration = serde_json::to_string(&json!({
        "vae": {
            "timestep_conditioning": false,
            "causal_decoder": true,
        }
    }))?;
    model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::SafetensorsMetadata {
            source_artifact_ordinal: 0,
        },
        serde_json::to_vec(&BTreeMap::from([("config", configuration)]))?,
    )
}

fn protobuf_varint_fixture(mut value: u64, output: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn sentence_piece_fixture() -> Vec<u8> {
    let mut model = Vec::new();
    for (piece, piece_type) in [("<unk>", 2_u64), ("</s>", 1), ("▁", 1)] {
        let mut record = Vec::new();
        record.push(0x0a);
        protobuf_varint_fixture(piece.len() as u64, &mut record);
        record.extend_from_slice(piece.as_bytes());
        record.push(0x15);
        record.extend_from_slice(&0.0_f32.to_bits().to_le_bytes());
        record.push(0x18);
        protobuf_varint_fixture(piece_type, &mut record);
        model.push(0x0a);
        protobuf_varint_fixture(record.len() as u64, &mut model);
        model.extend(record);
    }
    model
}

fn t5_model_resource_fixture() -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let state = [
        ("shared.weight", vec![3, 4]),
        ("encoder.block.0.layer.0.layer_norm.weight", vec![4]),
        ("encoder.block.0.layer.0.SelfAttention.q.weight", vec![4, 4]),
        ("encoder.block.0.layer.0.SelfAttention.k.weight", vec![4, 4]),
        ("encoder.block.0.layer.0.SelfAttention.v.weight", vec![4, 4]),
        ("encoder.block.0.layer.0.SelfAttention.o.weight", vec![4, 4]),
        (
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            vec![32, 1],
        ),
        ("encoder.block.0.layer.1.layer_norm.weight", vec![4]),
        (
            "encoder.block.0.layer.1.DenseReluDense.wi_0.weight",
            vec![8, 4],
        ),
        (
            "encoder.block.0.layer.1.DenseReluDense.wi_1.weight",
            vec![8, 4],
        ),
        (
            "encoder.block.0.layer.1.DenseReluDense.wo.weight",
            vec![4, 8],
        ),
        ("encoder.final_layer_norm.weight", vec![4]),
    ]
    .into_iter()
    .map(|(key, shape)| ModelResourceStateEntry {
        key: key.to_owned(),
        shape,
    })
    .collect::<Vec<_>>();
    let fixture = model_resource_fixture("clip-t5", &state)?;
    let fixture = model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::JsonConfig,
        serde_json::to_vec(&json!({
            "vocab_size": 3,
            "d_model": 4,
            "num_heads": 1,
            "num_layers": 1,
            "d_ff": 8,
            "d_kv": 4,
            "layer_norm_epsilon": 1.0e-6,
            "feed_forward_proj": "gated-gelu",
            "relative_attention_max_distance": 128,
        }))?,
    )?;
    model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::SentencePiece,
        sentence_piece_fixture(),
    )
}

fn llama_model_resource_fixture() -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let state = [
        ("model.embed_tokens.weight", vec![128_256, 4]),
        ("model.layers.0.input_layernorm.weight", vec![4]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
        ("model.layers.0.post_attention_layernorm.weight", vec![4]),
        ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ("model.norm.weight", vec![4]),
    ]
    .into_iter()
    .map(|(key, shape)| ModelResourceStateEntry {
        key: key.to_owned(),
        shape,
    })
    .collect::<Vec<_>>();
    let fixture = model_resource_fixture("clip-llama", &state)?;
    let fixture = model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::JsonConfig,
        serde_json::to_vec(&json!({
            "vocab_size": 128_256,
            "max_position_embeddings": 2048,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1.0e-5,
            "rope_theta": 10_000.0,
        }))?,
    )?;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or("workspace root is unavailable")?;
    model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::JsonTokenizer,
        fs::read(
            root.join("projects/comfy/ComfyUI/comfy/text_encoders/llama_tokenizer/tokenizer.json"),
        )?,
    )
}

fn clip_vision_model_resource_fixture() -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let mut state = vec![
        (
            "vision_model.embeddings.patch_embedding.weight",
            vec![2, 3, 1, 1],
        ),
        ("vision_model.embeddings.class_embedding", vec![2]),
        (
            "vision_model.embeddings.position_embedding.weight",
            vec![5, 2],
        ),
        ("vision_model.pre_layrnorm.weight", vec![2]),
        ("vision_model.pre_layrnorm.bias", vec![2]),
        ("vision_model.post_layernorm.weight", vec![2]),
        ("vision_model.post_layernorm.bias", vec![2]),
    ]
    .into_iter()
    .map(|(key, shape)| ModelResourceStateEntry {
        key: key.to_owned(),
        shape,
    })
    .collect::<Vec<_>>();
    let prefix = "vision_model.encoder.layers.0";
    for (name, shape) in [
        ("layer_norm1.weight", vec![2]),
        ("layer_norm1.bias", vec![2]),
        ("self_attn.q_proj.weight", vec![2, 2]),
        ("self_attn.q_proj.bias", vec![2]),
        ("self_attn.k_proj.weight", vec![2, 2]),
        ("self_attn.k_proj.bias", vec![2]),
        ("self_attn.v_proj.weight", vec![2, 2]),
        ("self_attn.v_proj.bias", vec![2]),
        ("self_attn.out_proj.weight", vec![2, 2]),
        ("self_attn.out_proj.bias", vec![2]),
        ("layer_norm2.weight", vec![2]),
        ("layer_norm2.bias", vec![2]),
        ("mlp.fc1.weight", vec![2, 2]),
        ("mlp.fc1.bias", vec![2]),
        ("mlp.fc2.weight", vec![2, 2]),
        ("mlp.fc2.bias", vec![2]),
    ] {
        state.push(ModelResourceStateEntry {
            key: format!("{prefix}.{name}"),
            shape,
        });
    }
    let fixture = model_resource_fixture("clip-vision", &state)?;
    model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::JsonConfig,
        serde_json::to_vec(&json!({
            "model_type": "clip",
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "image_size": 2,
            "patch_size": 1,
            "num_channels": 3,
            "max_num_patches": 4,
            "hidden_act": "quick_gelu",
        }))?,
    )
}

fn push_parameter(
    state: &mut Vec<ModelResourceStateEntry>,
    prefix: impl Into<String>,
    weight_shape: Vec<u64>,
    bias_shape: Vec<u64>,
) {
    let prefix = prefix.into();
    state.push(ModelResourceStateEntry {
        key: format!("{prefix}.weight"),
        shape: weight_shape,
    });
    state.push(ModelResourceStateEntry {
        key: format!("{prefix}.bias"),
        shape: bias_shape,
    });
}

fn latent_upscale_model_resource_fixture() -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let mut state = Vec::new();
    push_parameter(&mut state, "initial_conv", vec![32, 4, 3, 3], vec![32]);
    push_parameter(&mut state, "initial_norm", vec![32], vec![32]);
    for family in ["res_blocks", "post_upsample_res_blocks"] {
        for convolution in ["conv1", "conv2"] {
            push_parameter(
                &mut state,
                format!("{family}.0.{convolution}"),
                vec![32, 32, 3, 3],
                vec![32],
            );
        }
        for normalization in ["norm1", "norm2"] {
            push_parameter(
                &mut state,
                format!("{family}.0.{normalization}"),
                vec![32],
                vec![32],
            );
        }
    }
    push_parameter(&mut state, "upsampler.0", vec![128, 32, 3, 3], vec![128]);
    push_parameter(&mut state, "final_conv", vec![4, 32, 3, 3], vec![4]);
    let fixture = model_resource_fixture("latent-upscale", &state)?;
    model_resource_fixture_with_auxiliary(
        fixture,
        WorkerModelSourceFormat::JsonConfig,
        serde_json::to_vec(&json!({
            "_class_name": "LatentUpsampler",
            "in_channels": 4,
            "mid_channels": 32,
            "num_blocks_per_stage": 1,
            "dims": 2,
            "spatial_upsample": true,
            "temporal_upsample": false,
            "spatial_scale": 2.0,
            "rational_resampler": false,
        }))?,
    )
}

fn frame_interpolation_model_resource_fixture() -> Result<Arc<ModelResourceFixture>, Box<dyn Error>>
{
    let mut state = Vec::new();
    push_parameter(&mut state, "encode.cnn0", vec![16, 3, 3, 3], vec![16]);
    push_parameter(&mut state, "encode.cnn1", vec![16, 16, 3, 3], vec![16]);
    push_parameter(&mut state, "encode.cnn2", vec![16, 16, 3, 3], vec![16]);
    push_parameter(&mut state, "encode.cnn3", vec![16, 2, 4, 4], vec![2]);
    for block in 0..5 {
        let input = if block == 0 { 11 } else { 24 };
        push_parameter(
            &mut state,
            format!("blocks.{block}.conv0.0.0"),
            vec![1, input, 3, 3],
            vec![1],
        );
        push_parameter(
            &mut state,
            format!("blocks.{block}.conv0.1.0"),
            vec![2, 1, 3, 3],
            vec![2],
        );
        for residual in 0..8 {
            push_parameter(
                &mut state,
                format!("blocks.{block}.convblock.{residual}.conv"),
                vec![2, 2, 3, 3],
                vec![2],
            );
            state.push(ModelResourceStateEntry {
                key: format!("blocks.{block}.convblock.{residual}.beta"),
                shape: vec![1, 2, 1, 1],
            });
        }
        push_parameter(
            &mut state,
            format!("blocks.{block}.lastconv.0"),
            vec![2, 52, 4, 4],
            vec![52],
        );
    }
    model_resource_fixture("frame-interpolation", &state)
}

fn audio_encoder_model_resource_fixture() -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let width = 768_u64;
    let convolution_width = 512_u64;
    let feed_forward_width = 3_072_u64;
    let mut state = Vec::new();
    for (index, (input_channels, kernel)) in [
        (1, 10),
        (512, 3),
        (512, 3),
        (512, 3),
        (512, 3),
        (512, 2),
        (512, 2),
    ]
    .into_iter()
    .enumerate()
    {
        state.push(ModelResourceStateEntry {
            key: format!("feature_extractor.conv_layers.{index}.conv.weight"),
            shape: vec![convolution_width, input_channels, kernel],
        });
        if index == 0 {
            state.push(ModelResourceStateEntry {
                key: format!("feature_extractor.conv_layers.{index}.layer_norm.weight"),
                shape: vec![convolution_width],
            });
            state.push(ModelResourceStateEntry {
                key: format!("feature_extractor.conv_layers.{index}.layer_norm.bias"),
                shape: vec![convolution_width],
            });
        }
    }
    push_parameter(
        &mut state,
        "feature_projection.layer_norm",
        vec![convolution_width],
        vec![convolution_width],
    );
    push_parameter(
        &mut state,
        "feature_projection.projection",
        vec![width, convolution_width],
        vec![width],
    );
    state.push(ModelResourceStateEntry {
        key: "encoder.pos_conv_embed.conv.parametrizations.weight.original0".to_owned(),
        shape: vec![1, 1, 128],
    });
    state.push(ModelResourceStateEntry {
        key: "encoder.pos_conv_embed.conv.parametrizations.weight.original1".to_owned(),
        shape: vec![width, width / 16, 128],
    });
    state.push(ModelResourceStateEntry {
        key: "encoder.pos_conv_embed.conv.bias".to_owned(),
        shape: vec![width],
    });
    for layer in 0..12 {
        let prefix = format!("encoder.layers.{layer}");
        for projection in ["k_proj", "v_proj", "q_proj", "out_proj"] {
            push_parameter(
                &mut state,
                format!("{prefix}.attention.{projection}"),
                vec![width, width],
                vec![width],
            );
        }
        for normalization in ["layer_norm", "final_layer_norm"] {
            push_parameter(
                &mut state,
                format!("{prefix}.{normalization}"),
                vec![width],
                vec![width],
            );
        }
        push_parameter(
            &mut state,
            format!("{prefix}.feed_forward.intermediate_dense"),
            vec![feed_forward_width, width],
            vec![feed_forward_width],
        );
        push_parameter(
            &mut state,
            format!("{prefix}.feed_forward.output_dense"),
            vec![width, feed_forward_width],
            vec![width],
        );
    }
    push_parameter(&mut state, "encoder.layer_norm", vec![width], vec![width]);
    model_resource_fixture("audio-encoder", &state)
}

fn model_resource_tensor_fixture(
    category: &str,
    ordered_state: &[(String, Tensor)],
) -> Result<Arc<ModelResourceFixture>, Box<dyn Error>> {
    let mut tensors = Vec::new();
    let mut tensor_manifests = Vec::new();
    let mut aggregate_tensor_bytes = 0_u64;
    let mut maximum_tensor_bytes = 0_u64;
    for (name, tensor) in ordered_state {
        let bytes = tensor.contiguous_bytes()?.to_vec();
        let byte_length = u64::try_from(bytes.len())?;
        let byte_offset = aggregate_tensor_bytes;
        aggregate_tensor_bytes = aggregate_tensor_bytes
            .checked_add(byte_length)
            .ok_or("fixture aggregate tensor byte count overflow")?;
        maximum_tensor_bytes = maximum_tensor_bytes.max(byte_length);
        tensor_manifests.push(WorkerModelSourceTensor {
            name: name.clone(),
            data_type: format!("{:?}", tensor.descriptor().dtype()),
            shape: tensor.descriptor().shape().to_vec(),
            artifact_ordinal: 0,
            byte_offset,
            byte_length,
        });
        tensors.push(ModelResourceTensorFixture::Bytes(bytes));
    }
    let maximum_read_bytes = maximum_tensor_bytes.max(512 * 1024);
    Ok(Arc::new(ModelResourceFixture {
        manifest: WorkerModelSourceManifest {
            source_ordinal: 0,
            model_identity_sha256: WorkerSha256Digest::new(format!(
                "{:x}",
                Sha256::digest(format!("task401:{category}:model"))
            ))?,
            artifacts: vec![WorkerModelSourceArtifact {
                sha256: WorkerSha256Digest::new(format!(
                    "{:x}",
                    Sha256::digest(format!("task401:{category}:artifact"))
                ))?,
                byte_size: aggregate_tensor_bytes,
                format: WorkerModelSourceFormat::Safetensors,
                nested_state_disposition: WorkerModelSourceNestedStateDisposition::Flat,
            }],
            tensors: tensor_manifests,
            aggregate_tensor_bytes,
            maximum_read_bytes,
            parser_limits: WorkerModelSourceParserLimits {
                version: 1,
                manifest_bytes: 64 * 1024 * 1024,
                maximum_depth: 64,
                maximum_tensors: 10_000,
                maximum_tensor_bytes: maximum_read_bytes,
                maximum_aggregate_tensor_bytes: aggregate_tensor_bytes.max(512 * 1024),
                maximum_name_bytes: 16 * 1024,
                maximum_archive_entries: 10_000,
                maximum_metadata_values: 1_000_000,
            },
        },
        tensors,
        auxiliary_artifacts: BTreeMap::new(),
    }))
}

#[derive(Debug)]
enum ModelResourceMatrixError {
    Service(sim_nodes::NativeModelResourceServiceError),
    Node(NativeNodeFailure),
}

impl std::fmt::Display for ModelResourceMatrixError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Service(error) => write!(formatter, "{error}"),
            Self::Node(error) => formatter.write_str(&error.message),
        }
    }
}

impl Error for ModelResourceMatrixError {}

fn model_resource_matrix_error_matches(error: &ModelResourceMatrixError, expected: &str) -> bool {
    match (error, expected) {
        (
            ModelResourceMatrixError::Service(
                sim_nodes::NativeModelResourceServiceError::Unavailable,
            ),
            "unavailable",
        ) => true,
        (
            ModelResourceMatrixError::Service(
                sim_nodes::NativeModelResourceServiceError::InvalidRequest,
            ),
            "invalid",
        ) => true,
        (
            ModelResourceMatrixError::Service(
                sim_nodes::NativeModelResourceServiceError::Cancelled,
            ),
            "cancelled",
        ) => true,
        (ModelResourceMatrixError::Node(error), "unavailable") => {
            error.code == "native_model_resource_unavailable"
        }
        (ModelResourceMatrixError::Node(error), "invalid") => {
            error.code == "native_model_resource_invalid"
        }
        (ModelResourceMatrixError::Node(error), "cancelled") => {
            error.code == "native_model_resource_cancelled"
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn load_model_resource_for_matrix<R>(
    exercise_execution_boundary: bool,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: NodeId,
    cancellation: CancellationToken,
    scratch: sim_tensor::ScratchReservation,
    handle_store: Arc<dyn sim_nodes::NativeHandleStore>,
    binding: NativeModelSourceWorkerBinding,
    backend: Arc<CpuBackend>,
    request: NativeModelResourceRequest,
    host_operation: impl FnOnce(
        sim_runtime::NativeModelSourceTransportHost,
    ) -> Result<R, WorkerModelSourceError>
    + Send
    + 'static,
) -> Result<(Result<Arc<NativeModelPayload>, ModelResourceMatrixError>, R), Box<dyn Error>>
where
    R: Send + 'static,
{
    if exercise_execution_boundary {
        let (result, host_result) = execute_reduced_fixture_native_model_resource_for_test(
            prompt_id,
            attempt_id,
            node_id,
            cancellation,
            scratch,
            handle_store,
            binding,
            backend,
            request,
            host_operation,
        )?;
        Ok((
            result
                .map(|(payload, _evidence)| payload)
                .map_err(ModelResourceMatrixError::Node),
            host_result,
        ))
    } else {
        let (result, host_result) = load_reduced_fixture_native_model_resource_for_test(
            prompt_id,
            attempt_id,
            node_id,
            cancellation,
            scratch,
            handle_store,
            binding,
            backend,
            request,
            host_operation,
        )?;
        Ok((
            result.map_err(ModelResourceMatrixError::Service),
            host_result,
        ))
    }
}

fn model_resource_style_loader_plan(
    source_name: &str,
    prompt_id: PromptId,
) -> Result<sim_runtime::CompiledPlan, Box<dyn Error>> {
    let node_id = NodeId("model-resource-loader".to_owned());
    let registry = generated_native_node_registry_projection(None)?;
    let mut descriptor = registry
        .descriptor("StyleModelLoader")
        .cloned()
        .ok_or("generated registry omitted StyleModelLoader")?;
    descriptor.output_node = true;
    Ok(sim_runtime::CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            sim_runtime::CompiledNode {
                id: node_id.clone(),
                class_type: "StyleModelLoader".to_owned(),
                descriptor,
                inputs: BTreeMap::from([(
                    "style_model_name".to_owned(),
                    sim_runtime::InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::String(source_name.to_owned()),
                        },
                    },
                )]),
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    })
}

#[test]
fn audio_encoder_loader_production_combo_retains_full_source_profile_and_cache()
-> Result<(), Box<dyn Error>> {
    let source_name = "full-base.safetensors";
    let fixture = audio_encoder_model_resource_fixture()?;
    let source_bytes = fixture.manifest.aggregate_tensor_bytes;
    assert!(source_bytes > 256 * 1024 * 1024);
    let fixtures = Arc::new(BTreeMap::from([(
        ("audio_encoders".to_owned(), source_name.to_owned()),
        fixture,
    )]));
    let memory_bytes = 2 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_bytes)?;
    let backend = Arc::new(backend);
    let binding =
        NativeModelSourceWorkerBinding::checked_for_test(Uuid::from_u128(0x634_310), 1, 1, 1)?;
    let (executor, host) = NativeImageExecutor::new_with_generated_registry(
        ProfileId(Uuid::from_u128(0x634_311)),
        BTreeMap::new(),
        true,
        backend,
    )?
    .with_model_source_worker(binding)?;
    let generation = executor.handle_store_generation_for_test();
    let projection = generated_native_node_registry_projection(None)?;
    let mut descriptor = projection
        .descriptor("AudioEncoderLoader")
        .cloned()
        .ok_or("AudioEncoderLoader descriptor missing")?;
    descriptor.output_node = true;
    let mut registry = sim_runtime::NativeNodeRegistry::default();
    registry.register_descriptor(descriptor)?;
    let node_id = NodeId("full-audio-loader".to_owned());
    let plan =
        sim_runtime::PromptCompiler::new(&registry).compile(sim_types::PromptSubmission {
            prompt: sim_types::ApiPrompt(BTreeMap::from([(
                node_id.clone(),
                sim_types::PromptNode {
                    class_type: "AudioEncoderLoader".to_owned(),
                    inputs: BTreeMap::from([("audio_encoder_name".to_owned(), json!(source_name))]),
                    unknown: BTreeMap::new(),
                },
            )])),
            prompt_id: Some(PromptId(Uuid::from_u128(0x634_312))),
            client_id: None,
            number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
        })?;
    plan.validate_integrity()?;
    let node = plan.nodes.get(&node_id).ok_or("compiled loader missing")?;
    assert!(
        matches!(node.inputs.get("audio_encoder_name"), Some(InputBinding::Literal {
        value: NativeValue::PreservedUnknown { type_name, value }
    }) if type_name == "COMBO" && value == source_name)
    );
    let host_thread = std::thread::spawn(move || serve_model_resource_transport(host, fixtures));
    let cancellation = CancellationToken::default();
    let first_attempt = AttemptId(Uuid::from_u128(0x634_313));
    let first = executor.execute_blocking(
        &plan,
        first_attempt,
        cancellation.clone(),
        0,
        authority.authorize_workspace(memory_bytes)?,
    )?;
    assert_eq!(
        first.report.state,
        AttemptState::Succeeded,
        "{:?}",
        first.report.error
    );
    assert_eq!(first.report.cache_hits, 0);
    assert_eq!(first.report.ui_outputs.len(), 1);
    let evidence: Vec<PersistedNativeModelResourceEvidence> = serde_json::from_value(
        first
            .report
            .ui_outputs
            .get(&node_id)
            .and_then(|output| output.get("native_model_resource_evidence"))
            .cloned()
            .ok_or("audio loader omitted reconstruction evidence")?,
    )?;
    assert_eq!(evidence.len(), 1);
    let evidence = evidence
        .first()
        .ok_or("audio reconstruction evidence absent")?;
    evidence.validate()?;
    assert_eq!(evidence.node_id, node_id);
    assert!(first.output_proposals.is_empty());
    let handle = match first
        .report
        .outputs
        .get(&node_id)
        .and_then(|values| values.first())
    {
        Some(NativeValue::Handle { value }) => value.clone(),
        _ => return Err("production audio loader did not publish its handle".into()),
    };
    {
        let store = generation.handle_store_for_attempt(first_attempt);
        let resolved = store.resolve(&handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Model(model) = resolved.as_ref() else {
            return Err("audio output is not a canonical model payload".into());
        };
        let audio = model
            .model_payload()
            .audio_encoder_resource()
            .ok_or("audio output omitted concrete backing")?;
        assert_eq!(
            audio.architecture(),
            sim_model::audio_encoder::NativeAudioEncoderArchitecture::Wav2Vec2Base
        );
        let resident = audio
            .resident_tensor_allocations()?
            .iter()
            .try_fold(0_u64, |sum, (_, bytes)| {
                sum.checked_add(*bytes).ok_or("resident byte overflow")
            })?;
        assert!(resident >= source_bytes);
        audio.validate(&cancellation)?;
        assert_eq!(
            &evidence.resource_identity,
            model.model_payload().identity()
        );
        assert!(audio.diagnostic().unexpected_state_keys().is_empty());
    }
    let cached = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x634_314)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(memory_bytes)?,
    )?;
    assert_eq!(cached.report.state, AttemptState::Succeeded);
    assert_eq!(cached.report.cache_hits, 1);
    assert_eq!(cached.report.ui_outputs, first.report.ui_outputs);
    assert!(
        matches!(cached.report.outputs.get(&node_id).and_then(|values| values.first()), Some(NativeValue::Handle { value }) if value == &handle)
    );
    assert!(cached.output_proposals.is_empty());
    drop(cached);
    drop(first);
    drop(executor);
    let counts = host_thread
        .join()
        .map_err(|_| "full audio loader host panicked")??;
    assert!(counts.0 > 0 && counts.1 > 0);
    assert_eq!(counts.0, counts.2);
    assert!(generation.is_empty());
    Ok(())
}

fn model_resource_execution_cache_boundary(
    fixtures: Arc<BTreeMap<(String, String), Arc<ModelResourceFixture>>>,
    backend: Arc<CpuBackend>,
    workspace_authority: &CpuWorkspaceAuthority,
    workspace_bytes: u64,
) -> Result<(), Box<dyn Error>> {
    let first_fixture = fixtures
        .get(&("style_models".to_owned(), "style.safetensors".to_owned()))
        .cloned()
        .ok_or("style cache fixture is missing")?;
    let mut cache_fixtures = (*fixtures).clone();
    cache_fixtures.insert(
        (
            "style_models".to_owned(),
            "style-eviction.safetensors".to_owned(),
        ),
        first_fixture,
    );
    let cache_fixtures = Arc::new(cache_fixtures);
    let binding =
        NativeModelSourceWorkerBinding::checked_for_test(Uuid::from_u128(0x402_200), 1, 1, 1)?;
    let (executor, host) = NativeImageExecutor::new_with_generated_registry(
        ProfileId(Uuid::from_u128(0x402_201)),
        BTreeMap::new(),
        true,
        backend,
    )?
    .with_model_source_worker(binding)?;
    let executor = executor
        .with_reduced_fixture_model_resources()
        .with_cache_capacity_for_test(1)?;
    let generation = executor.handle_store_generation_for_test();
    let host_thread =
        std::thread::spawn(move || serve_model_resource_transport(host, cache_fixtures));
    let first_plan = model_resource_style_loader_plan(
        "style.safetensors",
        PromptId(Uuid::from_u128(0x402_202)),
    )?;
    let first = executor.execute_blocking(
        &first_plan,
        AttemptId(Uuid::from_u128(0x402_203)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(first.report.state, AttemptState::Succeeded);
    assert_eq!(first.report.cache_hits, 0);
    let old_handle = match first
        .report
        .outputs
        .get(&NodeId("model-resource-loader".to_owned()))
        .and_then(|outputs| outputs.first())
    {
        Some(NativeValue::Handle { value }) => value.clone(),
        _ => return Err("style loader did not publish a native handle".into()),
    };
    drop(first);

    let cached = executor.execute_blocking(
        &first_plan,
        AttemptId(Uuid::from_u128(0x402_204)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(cached.report.state, AttemptState::Succeeded);
    assert_eq!(cached.report.cache_hits, 1);
    assert!(matches!(
        cached
            .report
            .outputs
            .get(&NodeId("model-resource-loader".to_owned()))
            .and_then(|outputs| outputs.first()),
        Some(NativeValue::Handle { value }) if value == &old_handle
    ));
    drop(cached);

    let eviction_plan = model_resource_style_loader_plan(
        "style-eviction.safetensors",
        PromptId(Uuid::from_u128(0x402_205)),
    )?;
    let evicted = executor.execute_blocking(
        &eviction_plan,
        AttemptId(Uuid::from_u128(0x402_206)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(evicted.report.state, AttemptState::Succeeded);
    assert_eq!(evicted.report.cache_hits, 0);
    let probe_store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x402_207)));
    assert!(matches!(
        probe_store.resolve(
            &old_handle,
            old_handle.handle_type(),
            &CancellationToken::default(),
        ),
        Err(NativeHandleStoreError::Missing(_))
    ));
    drop(evicted);
    drop(executor);
    let counts = host_thread
        .join()
        .map_err(|_| "model-resource cache host thread panicked")??;
    assert_eq!(counts.0, 7);
    assert!(counts.1 > 0);
    assert_eq!(counts.2, 7);
    Ok(())
}

#[test]
fn native_model_resource_service_recipe_matrix() -> Result<(), Box<dyn Error>> {
    run_native_model_resource_service_recipe_matrix(false)
}

#[test]
fn native_model_resource_execution_matrix() -> Result<(), Box<dyn Error>> {
    run_native_model_resource_service_recipe_matrix(true)
}

#[test]
fn native_model_resource_execution_cache_uses_generated_loader() -> Result<(), Box<dyn Error>> {
    let oracle: ModelResourceServiceOracle = serde_json::from_slice(MODEL_RESOURCE_SERVICE_ORACLE)?;
    let style = oracle
        .style
        .dtypes
        .get("float32")
        .ok_or("style float32 fixture is missing")?;
    let fixtures = Arc::new(BTreeMap::from([(
        ("style_models".to_owned(), "style.safetensors".to_owned()),
        model_resource_fixture("style_models", &style.state)?,
    )]));
    let workspace_bytes = 1024 * 1024 * 1024;
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    model_resource_execution_cache_boundary(
        fixtures,
        Arc::new(backend),
        &workspace_authority,
        workspace_bytes,
    )
}

#[test]
fn native_model_resource_execution_rejects_persisted_identity_before_publication()
-> Result<(), Box<dyn Error>> {
    let oracle: ModelResourceServiceOracle = serde_json::from_slice(MODEL_RESOURCE_SERVICE_ORACLE)?;
    let style = oracle
        .style
        .dtypes
        .get("float32")
        .ok_or("style float32 fixture is missing")?;
    let fixtures = Arc::new(BTreeMap::from([(
        ("style_models".to_owned(), "style.safetensors".to_owned()),
        model_resource_fixture("style_models", &style.state)?,
    )]));
    let workspace_bytes = 1024 * 1024 * 1024;
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let backend = Arc::new(backend);
    let node_id = NodeId("model-resource-loader".to_owned());
    let plan = model_resource_style_loader_plan(
        "style.safetensors",
        PromptId(Uuid::from_u128(0x402_310)),
    )?;

    let (first_executor, first_host) = NativeImageExecutor::new_with_generated_registry(
        ProfileId(Uuid::from_u128(0x402_311)),
        BTreeMap::new(),
        true,
        backend.clone(),
    )?
    .with_model_source_worker(NativeModelSourceWorkerBinding::checked_for_test(
        Uuid::from_u128(0x402_312),
        1,
        1,
        1,
    )?)?;
    let first_executor = first_executor.with_reduced_fixture_model_resources();
    let first_fixtures = fixtures.clone();
    let first_host_thread =
        std::thread::spawn(move || serve_model_resource_transport(first_host, first_fixtures));
    let first = first_executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x402_313)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(first.report.state, AttemptState::Succeeded);
    let mut evidence: Vec<PersistedNativeModelResourceEvidence> = serde_json::from_value(
        first
            .report
            .ui_outputs
            .get(&node_id)
            .and_then(|output| output.get("native_model_resource_evidence"))
            .cloned()
            .ok_or("style execution omitted reconstruction evidence")?,
    )?;
    assert_eq!(evidence.len(), 1);
    drop(first);
    drop(first_executor);
    first_host_thread
        .join()
        .map_err(|_| "initial reconstruction host panicked")??;

    evidence[0].resource_identity = NativeModelResourceIdentity::checked(
        NativeModelResourceRole::StyleModel,
        "persisted-style-identity-mismatch",
        "zed-native-style-model-v1",
        "a".repeat(64),
        "b".repeat(64),
    )?;
    evidence[0].validate()?;
    let binding =
        NativeModelSourceWorkerBinding::checked_for_test(Uuid::from_u128(0x402_314), 2, 2, 2)?
            .with_model_resource_reconstruction_for_test(BTreeMap::from([(node_id, evidence)]))?
            .with_reduced_fixture_model_resources_for_test();
    let (executor, host) = NativeImageExecutor::new_with_generated_registry(
        ProfileId(Uuid::from_u128(0x402_315)),
        BTreeMap::new(),
        true,
        backend,
    )?
    .with_model_source_worker(binding)?;
    let generation = executor.handle_store_generation_for_test();
    let mismatch_fixtures = fixtures;
    let host_thread =
        std::thread::spawn(move || serve_model_resource_transport(host, mismatch_fixtures));
    let mismatch = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x402_316)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(mismatch.report.state, AttemptState::Failed);
    assert!(mismatch.report.outputs.is_empty());
    assert!(mismatch.report.ui_outputs.is_empty());
    assert!(mismatch.output_proposals.is_empty());
    assert!(
        mismatch
            .report
            .error
            .as_deref()
            .is_some_and(|error| error.contains("source changed"))
    );
    assert_eq!(mismatch.report.cache_hits, 0);
    assert_eq!(executor.cache_len_for_test(), 0);
    assert!(generation.is_empty());
    drop(mismatch);
    drop(executor);
    let counts = host_thread
        .join()
        .map_err(|_| "mismatched reconstruction host panicked")??;
    assert_eq!(counts.0, 3);
    assert!(counts.1 > 0);
    assert_eq!(counts.2, 3);
    Ok(())
}

fn run_native_model_resource_service_recipe_matrix(
    exercise_execution_boundary: bool,
) -> Result<(), Box<dyn Error>> {
    let oracle: ModelResourceServiceOracle = serde_json::from_slice(MODEL_RESOURCE_SERVICE_ORACLE)?;
    let style = oracle
        .style
        .dtypes
        .get("float32")
        .ok_or("style float32 fixture is missing")?;
    let photomaker = oracle
        .photomaker
        .dtypes
        .get("float32")
        .ok_or("PhotoMaker float32 fixture is missing")?;
    let malformed_style_state = [ModelResourceStateEntry {
        key: "invalid.weight".to_owned(),
        shape: vec![1],
    }];
    let workspace_bytes = 1024 * 1024 * 1024;
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let fixture_cancellation = CancellationToken::default();
    let fixture_scratch = workspace_authority.authorize_workspace(workspace_bytes)?;
    let fixture_context =
        backend.execution_context(StreamId::DEFAULT, fixture_scratch, &fixture_cancellation);
    let background = deterministic_reduced_test_checkpoint(
        &backend,
        &fixture_context,
        BackgroundRemovalFixtureMutation::None,
    )?;
    let background_fixture =
        model_resource_tensor_fixture("background_removal", &background.ordered_state)?;
    drop(background);
    let da3 = deterministic_reduced_depth_anything_3_checkpoint(
        &backend,
        DepthAnything3FixtureProfile::Dpt,
        DType::F32,
        workspace_bytes,
        &fixture_context,
    )?;
    let da3_fixture = model_resource_tensor_fixture("depth-anything-3", &da3.ordered_state)?;
    drop(da3);
    let moge = deterministic_reduced_moge_checkpoint(
        &backend,
        MogeFixtureProfile::V1,
        DType::F32,
        workspace_bytes,
        &fixture_context,
    )?;
    let moge_fixture = model_resource_tensor_fixture("moge", &moge.ordered_state)?;
    drop(moge);
    let raft = raft_large_exact_native(false, false, &fixture_cancellation)?;
    let raft_state = raft
        .state_schema()
        .iter()
        .map(|specification| {
            (
                specification.name.clone(),
                specification.shape.clone(),
                specification.dtype,
            )
        })
        .collect::<Vec<_>>();
    let raft_fixture = model_resource_typed_fixture("optical-flow", &raft_state)?;
    drop(raft);
    drop(fixture_context);
    let image_vae_fixture = model_resource_typed_fixture(
        "vae-image",
        &[("taesd_decoder.1.weight".to_owned(), vec![1, 4], DType::F32)],
    )?;
    let checkpoint_image_vae_fixture = model_resource_typed_fixture(
        "vae-image-checkpoint",
        &[(
            "first_stage_model.taesd_decoder.1.weight".to_owned(),
            vec![1, 4],
            DType::F32,
        )],
    )?;
    let video_vae_fixture = model_resource_typed_fixture(
        "vae-video",
        &[
            ("decoder.1.weight".to_owned(), vec![1, 32], DType::F16),
            ("decoder.22.bias".to_owned(), vec![12], DType::F16),
        ],
    )?;
    let audio_vae_fixture = model_resource_typed_fixture(
        "vae-audio",
        &[(
            "decoder.layers.1.layers.0.beta".to_owned(),
            vec![1],
            DType::F16,
        )],
    )?;
    let structured_vae_fixture = model_resource_typed_fixture(
        "vae-structured",
        &[(
            "geo_decoder.cross_attn_decoder.ln_1.bias".to_owned(),
            vec![1],
            DType::F16,
        )],
    )?;
    let fixtures = Arc::new(BTreeMap::from([
        (
            ("style_models".to_owned(), "style.safetensors".to_owned()),
            model_resource_fixture("style_models", &style.state)?,
        ),
        (
            ("photomaker".to_owned(), "photo.safetensors".to_owned()),
            model_resource_fixture("photomaker", &photomaker.state)?,
        ),
        (
            ("gligen".to_owned(), "gligen.safetensors".to_owned()),
            model_resource_fixture("gligen", &oracle.gligen.state)?,
        ),
        (
            (
                "style_models".to_owned(),
                "malformed.safetensors".to_owned(),
            ),
            model_resource_fixture("style_models-malformed", &malformed_style_state)?,
        ),
        (
            (
                "background_removal".to_owned(),
                "birefnet.safetensors".to_owned(),
            ),
            background_fixture,
        ),
        (
            (
                "geometry_estimation".to_owned(),
                "da3.safetensors".to_owned(),
            ),
            da3_fixture,
        ),
        (
            (
                "geometry_estimation".to_owned(),
                "moge.safetensors".to_owned(),
            ),
            moge_fixture,
        ),
        (
            (
                "checkpoints".to_owned(),
                "clip-checkpoint.safetensors".to_owned(),
            ),
            mixed_sdxl_checkpoint_resource_fixture()?,
        ),
        (
            ("text_encoders".to_owned(), "clip-l.safetensors".to_owned()),
            clip_model_resource_fixture("clip-l", 768, "")?,
        ),
        (
            ("text_encoders".to_owned(), "clip-g.safetensors".to_owned()),
            clip_model_resource_fixture("clip-g", 1_280, "")?,
        ),
        (
            ("text_encoders".to_owned(), "t5.safetensors".to_owned()),
            t5_model_resource_fixture()?,
        ),
        (
            ("text_encoders".to_owned(), "llama.safetensors".to_owned()),
            llama_model_resource_fixture()?,
        ),
        (
            ("vae".to_owned(), "image.safetensors".to_owned()),
            image_vae_fixture,
        ),
        (
            (
                "checkpoints".to_owned(),
                "image-checkpoint.safetensors".to_owned(),
            ),
            checkpoint_image_vae_fixture,
        ),
        (
            ("vae".to_owned(), "video.safetensors".to_owned()),
            video_vae_fixture,
        ),
        (
            ("vae".to_owned(), "audio.safetensors".to_owned()),
            audio_vae_fixture,
        ),
        (
            ("vae".to_owned(), "structured.safetensors".to_owned()),
            structured_vae_fixture,
        ),
        (
            ("vae".to_owned(), "ltx-metadata.safetensors".to_owned()),
            metadata_backed_ltx_vae_fixture()?,
        ),
        (
            (
                "clip_vision".to_owned(),
                "clip-vision.safetensors".to_owned(),
            ),
            clip_vision_model_resource_fixture()?,
        ),
        (
            (
                "latent_upscale_models".to_owned(),
                "latent.safetensors".to_owned(),
            ),
            latent_upscale_model_resource_fixture()?,
        ),
        (
            (
                "frame_interpolation".to_owned(),
                "rife.safetensors".to_owned(),
            ),
            frame_interpolation_model_resource_fixture()?,
        ),
        (
            ("optical_flow".to_owned(), "raft.safetensors".to_owned()),
            raft_fixture,
        ),
        (
            (
                "audio_encoders".to_owned(),
                "wav2vec.safetensors".to_owned(),
            ),
            audio_encoder_model_resource_fixture()?,
        ),
        (
            (
                "checkpoints".to_owned(),
                "auraflow-checkpoint.safetensors".to_owned(),
            ),
            auraflow_model_resource_fixture(NativeModelRecipeSourceForm::Checkpoint)?,
        ),
        (
            (
                "diffusion_models".to_owned(),
                "auraflow.safetensors".to_owned(),
            ),
            auraflow_model_resource_fixture(NativeModelRecipeSourceForm::Standalone)?,
        ),
    ]));
    let backend = Arc::new(backend);
    let attempt_id = AttemptId(Uuid::from_u128(0x4001));
    let handle_store_generation = NativeHandleStoreGeneration::new()?;
    let handle_store = handle_store_generation.handle_store_for_attempt(attempt_id);
    let binding =
        NativeModelSourceWorkerBinding::checked_for_test(Uuid::from_u128(0x4002), 1, 1, 1)?;
    let mut open_count = 0;
    let mut read_count = 0;
    let mut close_count = 0;
    for (index, (category, source_name)) in [
        ("checkpoints", "auraflow-checkpoint.safetensors"),
        ("diffusion_models", "auraflow.safetensors"),
    ]
    .into_iter()
    .enumerate()
    {
        let fixtures = fixtures.clone();
        let (payload, counts) = load_model_resource_for_matrix(
            exercise_execution_boundary,
            PromptId(Uuid::from_u128(0x40122)),
            attempt_id,
            NodeId(format!("resource-model-{index}")),
            CancellationToken::default(),
            workspace_authority.authorize_workspace(workspace_bytes)?,
            handle_store.clone(),
            binding.clone(),
            backend.clone(),
            NativeModelResourceRequest::checked(
                NativeModelResourceRole::Model,
                category,
                vec![source_name.to_owned()],
            )?,
            move |host| serve_model_resource_transport(host, fixtures),
        )?;
        let payload = payload
            .map_err(|error| format!("{category} MODEL service construction failed: {error}"))?;
        assert_eq!(payload.identity().role(), NativeModelResourceRole::Model);
        let resource = payload
            .native_family_model_resource()
            .ok_or("MODEL payload omitted concrete family backing")?;
        assert_eq!(resource.family_identity()?.identifier(), "AuraFlow");
        open_count += counts.0;
        read_count += counts.1;
        close_count += counts.2;
    }
    for (index, (role, category, source_name)) in [
        (
            NativeModelResourceRole::StyleModel,
            "style_models",
            "style.safetensors",
        ),
        (
            NativeModelResourceRole::Photomaker,
            "photomaker",
            "photo.safetensors",
        ),
        (
            NativeModelResourceRole::Gligen,
            "gligen",
            "gligen.safetensors",
        ),
        (
            NativeModelResourceRole::BackgroundRemoval,
            "background_removal",
            "birefnet.safetensors",
        ),
        (
            NativeModelResourceRole::Da3Model,
            "geometry_estimation",
            "da3.safetensors",
        ),
        (
            NativeModelResourceRole::MogeModel,
            "geometry_estimation",
            "moge.safetensors",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let fixtures = fixtures.clone();
        let (payload, counts) = load_model_resource_for_matrix(
            exercise_execution_boundary,
            PromptId(Uuid::from_u128(0x4003)),
            attempt_id,
            NodeId(format!("resource-{index}")),
            CancellationToken::default(),
            workspace_authority.authorize_workspace(workspace_bytes)?,
            handle_store.clone(),
            binding.clone(),
            backend.clone(),
            NativeModelResourceRequest::checked(role, category, vec![source_name.to_owned()])?,
            move |host| serve_model_resource_transport(host, fixtures),
        )
        .map_err(|error| format!("{role:?} model-resource load failed: {error}"))?;
        let payload = payload
            .map_err(|error| format!("{role:?} model-resource construction failed: {error}"))?;
        assert_eq!(payload.identity().role(), role);
        open_count += counts.0;
        read_count += counts.1;
        close_count += counts.2;
    }
    for (index, (category, source_names, expected_components)) in [
        ("checkpoints", vec!["clip-checkpoint.safetensors"], 2_usize),
        ("text_encoders", vec!["clip-l.safetensors"], 1),
        (
            "text_encoders",
            vec!["clip-l.safetensors", "clip-g.safetensors"],
            2,
        ),
        (
            "text_encoders",
            vec!["clip-l.safetensors", "clip-g.safetensors", "t5.safetensors"],
            3,
        ),
        (
            "text_encoders",
            vec![
                "clip-l.safetensors",
                "clip-g.safetensors",
                "t5.safetensors",
                "llama.safetensors",
            ],
            4,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let fixtures = fixtures.clone();
        let (payload, counts) = load_model_resource_for_matrix(
            exercise_execution_boundary,
            PromptId(Uuid::from_u128(0x40120)),
            attempt_id,
            NodeId(format!("resource-clip-{index}")),
            CancellationToken::default(),
            workspace_authority.authorize_workspace(workspace_bytes)?,
            handle_store.clone(),
            binding.clone(),
            backend.clone(),
            NativeModelResourceRequest::checked(
                NativeModelResourceRole::Clip,
                category,
                source_names.into_iter().map(str::to_owned).collect(),
            )?,
            move |host| serve_model_resource_transport(host, fixtures),
        )?;
        let payload = payload.map_err(|error| {
            format!("{expected_components}-component CLIP service construction failed: {error}")
        })?;
        assert_eq!(payload.identity().role(), NativeModelResourceRole::Clip);
        assert_eq!(
            payload
                .native_clip_resource()
                .ok_or("CLIP payload omitted concrete backing")?
                .components()
                .len(),
            expected_components
        );
        open_count += counts.0;
        read_count += counts.1;
        close_count += counts.2;
    }
    for (index, (role, category, source_name)) in [
        (NativeModelResourceRole::Vae, "vae", "image.safetensors"),
        (
            NativeModelResourceRole::Vae,
            "checkpoints",
            "image-checkpoint.safetensors",
        ),
        (NativeModelResourceRole::Vae, "vae", "video.safetensors"),
        (NativeModelResourceRole::Vae, "vae", "audio.safetensors"),
        (
            NativeModelResourceRole::Vae,
            "vae",
            "structured.safetensors",
        ),
        (
            NativeModelResourceRole::Vae,
            "vae",
            "ltx-metadata.safetensors",
        ),
        (
            NativeModelResourceRole::ClipVision,
            "clip_vision",
            "clip-vision.safetensors",
        ),
        (
            NativeModelResourceRole::LatentUpscaleModel,
            "latent_upscale_models",
            "latent.safetensors",
        ),
        (
            NativeModelResourceRole::FrameInterpolation,
            "frame_interpolation",
            "rife.safetensors",
        ),
        (
            NativeModelResourceRole::OpticalFlow,
            "optical_flow",
            "raft.safetensors",
        ),
        (
            NativeModelResourceRole::AudioEncoder,
            "audio_encoders",
            "wav2vec.safetensors",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let fixtures = fixtures.clone();
        let (payload, counts) = load_model_resource_for_matrix(
            exercise_execution_boundary,
            PromptId(Uuid::from_u128(0x40121)),
            attempt_id,
            NodeId(format!("resource-extra-{index}")),
            CancellationToken::default(),
            workspace_authority.authorize_workspace(workspace_bytes)?,
            handle_store.clone(),
            binding.clone(),
            backend.clone(),
            NativeModelResourceRequest::checked(role, category, vec![source_name.to_owned()])?,
            move |host| serve_model_resource_transport(host, fixtures),
        )?;
        let payload =
            payload.map_err(|error| format!("{role:?} service construction failed: {error}"))?;
        assert_eq!(payload.identity().role(), role);
        match role {
            NativeModelResourceRole::Vae => {
                if source_name == "structured.safetensors" {
                    let vae = payload
                        .structured_vae()
                        .ok_or("structured VAE payload omitted concrete backing")?;
                    assert_eq!(
                        vae.descriptor().boundary().kind(),
                        VaeBoundaryKind::StructuredOutput
                    );
                    assert_eq!(
                        vae.descriptor().identity().profile(),
                        &VaeKernelProfile::HunyuanShapeV1
                    );
                    assert!(payload.vae().is_none());
                } else {
                    let vae = payload
                        .vae()
                        .ok_or("VAE payload omitted concrete backing")?;
                    let (expected_boundary, expected_profile) = match source_name {
                        "image.safetensors" | "image-checkpoint.safetensors" => {
                            (VaeBoundaryKind::Image, VaeKernelProfile::TaesdV1)
                        }
                        "video.safetensors" => {
                            (VaeBoundaryKind::Video, VaeKernelProfile::LightTaeHv15V1)
                        }
                        "audio.safetensors" => (
                            VaeBoundaryKind::Audio,
                            VaeKernelProfile::AudioOobleck44KhzV1,
                        ),
                        "ltx-metadata.safetensors" => (
                            VaeBoundaryKind::Video,
                            VaeKernelProfile::LtxVideoV0 {
                                configuration_sha256: match vae.descriptor().identity().profile() {
                                    VaeKernelProfile::LtxVideoV0 {
                                        configuration_sha256,
                                    } => configuration_sha256.clone(),
                                    actual => {
                                        return Err(format!(
                                            "metadata-backed LTX selected {actual:?}"
                                        )
                                        .into());
                                    }
                                },
                            },
                        ),
                        _ => return Err("unexpected native VAE fixture".into()),
                    };
                    assert_eq!(vae.descriptor().boundary().kind(), expected_boundary);
                    assert_eq!(vae.descriptor().identity().profile(), &expected_profile);
                    if source_name == "ltx-metadata.safetensors" {
                        assert!(matches!(
                            vae.descriptor().identity().loader_configuration(),
                            VaeLoaderConfiguration::LtxVideo {
                                configuration_sha256: Some(_),
                                configuration_json: Some(_),
                            }
                        ));
                    }
                    assert!(payload.structured_vae().is_none());
                }
            }
            NativeModelResourceRole::ClipVision => {
                assert!(payload.clip_vision_resource().is_some());
            }
            NativeModelResourceRole::LatentUpscaleModel => {
                assert!(payload.latent_upscale_model_resource().is_some());
            }
            NativeModelResourceRole::FrameInterpolation => {
                assert!(payload.frame_interpolation_resource().is_some());
            }
            NativeModelResourceRole::OpticalFlow => {
                assert!(payload.optical_flow_resource().is_some());
            }
            NativeModelResourceRole::AudioEncoder => {
                assert!(payload.audio_encoder_resource().is_some());
            }
            _ => return Err("unexpected extra model resource role".into()),
        }
        open_count += counts.0;
        read_count += counts.1;
        close_count += counts.2;
    }
    assert_eq!(handle_store_generation.len(), 0);
    let failure_fixtures = fixtures;
    let (rejected, counts) = load_model_resource_for_matrix(
        exercise_execution_boundary,
        PromptId(Uuid::from_u128(0x4004)),
        attempt_id,
        NodeId("resource-invalid".to_owned()),
        CancellationToken::default(),
        workspace_authority.authorize_workspace(workspace_bytes)?,
        handle_store.clone(),
        binding.clone(),
        backend.clone(),
        NativeModelResourceRequest::checked(
            NativeModelResourceRole::StyleModel,
            "style_models",
            vec!["malformed.safetensors".to_owned()],
        )?,
        move |host| serve_model_resource_transport(host, failure_fixtures),
    )?;
    assert!(rejected.is_err());
    assert_eq!(handle_store_generation.len(), 0);
    open_count += counts.0;
    read_count += counts.1;
    close_count += counts.2;
    assert_eq!(
        open_count,
        if exercise_execution_boundary { 50 } else { 25 }
    );
    assert!(read_count >= 4);
    assert_eq!(
        close_count,
        if exercise_execution_boundary { 50 } else { 25 }
    );

    let unavailable_recipes = [
        (
            "controlnet",
            NativeModelResourceRole::ControlNet,
            "controlnet",
        ),
        (
            "face-detection",
            NativeModelResourceRole::FaceDetection,
            "face_detection",
        ),
        ("hooks", NativeModelResourceRole::Hooks, "hooks"),
        (
            "hook-keyframes",
            NativeModelResourceRole::HookKeyframes,
            "hook_keyframes",
        ),
        (
            "latent-operation",
            NativeModelResourceRole::LatentOperation,
            "latent_operations",
        ),
        ("lora", NativeModelResourceRole::LoraModel, "loras"),
        (
            "model-patch",
            NativeModelResourceRole::ModelPatch,
            "model_patches",
        ),
        (
            "upscale-model",
            NativeModelResourceRole::UpscaleModel,
            "upscale_models",
        ),
    ];
    for (case, role, category) in unavailable_recipes {
        let fixtures = Arc::new(BTreeMap::new());
        let (result, counts) = load_model_resource_for_matrix(
            exercise_execution_boundary,
            PromptId(Uuid::from_u128(0x40110)),
            attempt_id,
            NodeId(format!("resource-unavailable-{case}")),
            CancellationToken::default(),
            workspace_authority.authorize_workspace(workspace_bytes)?,
            handle_store.clone(),
            binding.clone(),
            backend.clone(),
            NativeModelResourceRequest::checked(
                role,
                category,
                vec![format!("{case}.safetensors")],
            )?,
            move |host| serve_model_resource_transport(host, fixtures),
        )?;
        let error = result
            .err()
            .ok_or("unavailable recipe unexpectedly loaded")?;
        assert!(model_resource_matrix_error_matches(&error, "unavailable"));
        assert_eq!(counts, (0, 0, 0));
        assert_eq!(handle_store_generation.len(), 0);
    }

    for (case, request, cancellation, expected) in [
        (
            "wrong-category",
            NativeModelResourceRequest::checked(
                NativeModelResourceRole::StyleModel,
                "checkpoints",
                vec!["style.safetensors".to_owned()],
            )?,
            CancellationToken::default(),
            "invalid",
        ),
        (
            "cancelled",
            NativeModelResourceRequest::checked(
                NativeModelResourceRole::StyleModel,
                "style_models",
                vec!["style.safetensors".to_owned()],
            )?,
            {
                let cancellation = CancellationToken::default();
                cancellation.cancel();
                cancellation
            },
            "cancelled",
        ),
    ] {
        let fixtures = Arc::new(BTreeMap::new());
        let (result, counts) = load_model_resource_for_matrix(
            exercise_execution_boundary,
            PromptId(Uuid::from_u128(0x40110)),
            attempt_id,
            NodeId(format!("resource-{case}")),
            cancellation,
            workspace_authority.authorize_workspace(workspace_bytes)?,
            handle_store.clone(),
            binding.clone(),
            backend.clone(),
            request,
            move |host| serve_model_resource_transport(host, fixtures),
        )?;
        let error = result.err().ok_or("negative recipe unexpectedly loaded")?;
        assert!(model_resource_matrix_error_matches(&error, expected));
        assert_eq!(counts, (0, 0, 0));
        assert_eq!(handle_store_generation.len(), 0);
    }
    Ok(())
}

fn task408_compile_workflow(backends: &[&str]) -> Result<Vec<u8>, Box<dyn Error>> {
    if backends.is_empty() {
        return Err("Task408 compile workflow requires at least one compile node".into());
    }
    let mut workflow: Value =
        serde_json::from_slice(include_bytes!("../fixtures/native_diffusion/workflow.json"))?;
    let nodes = workflow
        .get_mut("nodes")
        .and_then(Value::as_array_mut)
        .ok_or("native diffusion workflow nodes are missing")?;
    let loader = nodes
        .iter_mut()
        .find(|node| node.get("id").and_then(Value::as_str) == Some("1"))
        .ok_or("native diffusion workflow loader is missing")?;
    let model_links = loader
        .get_mut("outputs")
        .and_then(Value::as_array_mut)
        .and_then(|outputs| outputs.first_mut())
        .and_then(|output| output.get_mut("links"))
        .ok_or("native diffusion workflow model links are missing")?;
    *model_links = json!([10]);
    let sampler = nodes
        .iter_mut()
        .find(|node| node.get("id").and_then(Value::as_str) == Some("5"))
        .ok_or("native diffusion workflow sampler is missing")?;
    let sampler_widgets = sampler
        .get_mut("widgets_values")
        .and_then(Value::as_array_mut)
        .ok_or("native diffusion workflow sampler widgets are missing")?;
    *sampler_widgets
        .get_mut(1)
        .ok_or("native diffusion workflow sampler step widget is missing")? = json!(1);

    let compile_count = backends.len();
    for (index, backend) in backends.iter().enumerate() {
        let node_id = (8usize + index).to_string();
        let input_link_id = 10usize + index;
        let output_link_id = if index + 1 == compile_count {
            3usize
        } else {
            11usize + index
        };
        nodes.push(json!({
            "id": node_id,
            "type": "TorchCompileModel",
            "pos": [280 + index * 12, -280],
            "size": [240, 100],
            "inputs": [{"name": "model", "type": "MODEL", "link": input_link_id}],
            "outputs": [{"name": "model", "type": "MODEL", "links": [output_link_id]}],
            "properties": {"widget_input_names": ["backend"]},
            "widgets_values": [backend],
        }));
    }

    let links = workflow
        .get_mut("links")
        .and_then(Value::as_array_mut)
        .ok_or("native diffusion workflow links are missing")?;
    let sampler_model_link = links
        .iter_mut()
        .find(|link| {
            link.as_array()
                .and_then(|fields| fields.first())
                .and_then(Value::as_u64)
                == Some(3)
        })
        .ok_or("native diffusion workflow sampler model link is missing")?;
    let sampler_model_fields = sampler_model_link
        .as_array_mut()
        .ok_or("native diffusion workflow sampler model link is invalid")?;
    let last_compile_id = (7usize + compile_count).to_string();
    *sampler_model_fields
        .get_mut(1)
        .ok_or("native diffusion workflow sampler model source is missing")? =
        Value::String(last_compile_id);
    for index in 0..compile_count {
        let input_link_id = 10usize + index;
        let source_id = if index == 0 {
            "1".to_owned()
        } else {
            (7usize + index).to_string()
        };
        let compile_id = (8usize + index).to_string();
        links.push(json!([input_link_id, source_id, 0, compile_id, 0, "MODEL"]));
    }
    workflow["last_node_id"] = json!(7usize + compile_count);
    workflow["last_link_id"] = json!(9usize + compile_count);
    Ok(serde_json::to_vec(&workflow)?)
}

fn task408_compile_apg_workflow() -> Result<Vec<u8>, Box<dyn Error>> {
    let mut workflow: Value = serde_json::from_slice(&task408_compile_workflow(&["inductor"])?)?;
    let nodes = workflow
        .get_mut("nodes")
        .and_then(Value::as_array_mut)
        .ok_or("Task408 workflow nodes are missing")?;
    let compile = nodes
        .iter_mut()
        .find(|node| node.get("id").and_then(Value::as_str) == Some("8"))
        .ok_or("Task408 compile node is missing")?;
    *compile
        .get_mut("outputs")
        .and_then(Value::as_array_mut)
        .and_then(|outputs| outputs.first_mut())
        .and_then(|output| output.get_mut("links"))
        .ok_or("Task408 compile output links are missing")? = json!([20]);
    nodes.push(json!({
        "id": "9",
        "type": "APG",
        "pos": [520, -280],
        "size": [240, 120],
        "inputs": [{"name": "model", "type": "MODEL", "link": 20}],
        "outputs": [{"name": "model", "type": "MODEL", "links": [3]}],
        "properties": {"widget_input_names": ["eta", "norm_threshold", "momentum"]},
        "widgets_values": [1.0, 5.0, 0.0],
    }));
    let links = workflow
        .get_mut("links")
        .and_then(Value::as_array_mut)
        .ok_or("Task408 workflow links are missing")?;
    let sampler_model_link = links
        .iter_mut()
        .find(|link| {
            link.as_array()
                .and_then(|fields| fields.first())
                .and_then(Value::as_u64)
                == Some(3)
        })
        .and_then(Value::as_array_mut)
        .ok_or("Task408 sampler MODEL link is missing")?;
    *sampler_model_link
        .get_mut(1)
        .ok_or("Task408 sampler MODEL source is missing")? = json!("9");
    links.push(json!([20, "8", 0, "9", 0, "MODEL"]));
    workflow["last_node_id"] = json!(9);
    workflow["last_link_id"] = json!(20);
    Ok(serde_json::to_vec(&workflow)?)
}

fn task408_compile_handle_plan(
    mut descriptor: NativeNodeDescriptor,
    handle: NativeOpaqueHandle,
    prompt_id: PromptId,
) -> Result<CompiledPlan, Box<dyn Error>> {
    descriptor.output_node = true;
    let node_id = NodeId("task408-compile-handle".to_owned());
    let plan = CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            CompiledNode {
                id: node_id.clone(),
                class_type: "TorchCompileModel".to_owned(),
                descriptor,
                inputs: BTreeMap::from([
                    (
                        "model".to_owned(),
                        InputBinding::Literal {
                            value: NativeValue::Handle { value: handle },
                        },
                    ),
                    (
                        "backend".to_owned(),
                        InputBinding::Literal {
                            value: NativeValue::Primitive {
                                value: NativePrimitive::String("inductor".to_owned()),
                            },
                        },
                    ),
                ]),
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    Ok(plan)
}

fn task408_compile_custom_sampler_plan(
    mut plan: CompiledPlan,
    provider: Arc<dyn NativeDiffusionProvider>,
) -> Result<(CompiledPlan, NodeId), Box<dyn Error>> {
    plan.prompt_id = PromptId(Uuid::from_u128(0x408_00c));
    let registry = generated_native_node_registry_projection(Some(provider))?;
    let guider_descriptor = registry
        .descriptor("CFGGuider")
        .cloned()
        .ok_or("Task408 registry omitted CFGGuider")?;
    let custom_descriptor = registry
        .descriptor("SamplerCustomAdvanced")
        .cloned()
        .ok_or("Task408 registry omitted SamplerCustomAdvanced")?;
    let inputs_descriptor = task407_custom_sampler_inputs_fixture_descriptor_for_test()?;
    let inputs_id = NodeId("task408-custom-inputs".to_owned());
    let guider_id = NodeId("task408-cfg-guider".to_owned());
    let custom_id = NodeId("task408-custom-sampler".to_owned());
    plan.nodes.insert(
        inputs_id.clone(),
        CompiledNode {
            id: inputs_id.clone(),
            class_type: "Task407CustomSamplerInputsFixture".to_owned(),
            descriptor: inputs_descriptor,
            inputs: BTreeMap::new(),
            unknown: BTreeMap::new(),
        },
    );
    plan.nodes.insert(
        guider_id.clone(),
        CompiledNode {
            id: guider_id.clone(),
            class_type: "CFGGuider".to_owned(),
            descriptor: guider_descriptor,
            inputs: BTreeMap::from([
                (
                    "model".to_owned(),
                    InputBinding::Link {
                        source: NodeId("8".to_owned()),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "positive".to_owned(),
                    InputBinding::Link {
                        source: NodeId("2".to_owned()),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "negative".to_owned(),
                    InputBinding::Link {
                        source: NodeId("3".to_owned()),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "cfg".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Number(5.5),
                        },
                    },
                ),
            ]),
            unknown: BTreeMap::new(),
        },
    );
    plan.nodes.insert(
        custom_id.clone(),
        CompiledNode {
            id: custom_id.clone(),
            class_type: "SamplerCustomAdvanced".to_owned(),
            descriptor: custom_descriptor,
            inputs: BTreeMap::from([
                (
                    "noise".to_owned(),
                    InputBinding::Link {
                        source: inputs_id.clone(),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "guider".to_owned(),
                    InputBinding::Link {
                        source: guider_id,
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "sampler".to_owned(),
                    InputBinding::Link {
                        source: inputs_id.clone(),
                        output_index: 1,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "sigmas".to_owned(),
                    InputBinding::Link {
                        source: inputs_id,
                        output_index: 2,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "latent_image".to_owned(),
                    InputBinding::Link {
                        source: NodeId("4".to_owned()),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
            ]),
            unknown: BTreeMap::new(),
        },
    );
    let sampler_id = NodeId("5".to_owned());
    plan.nodes.remove(&sampler_id);
    plan.static_required_nodes.remove(&sampler_id);
    let decode_node = plan
        .nodes
        .get_mut(&NodeId("6".to_owned()))
        .ok_or("Task408 plan omitted VAEDecode")?;
    let samples_input = decode_node
        .descriptor
        .inputs
        .iter()
        .find(|input| input.name == "samples")
        .ok_or("Task408 VAEDecode descriptor omitted samples")?;
    let samples_lazy = samples_input.lazy;
    let samples_cardinality = samples_input.cardinality;
    decode_node.inputs.insert(
        "samples".to_owned(),
        InputBinding::Link {
            source: custom_id.clone(),
            output_index: 0,
            lazy: samples_lazy,
            mode: samples_cardinality,
        },
    );
    let mut incoming = plan
        .nodes
        .keys()
        .map(|node_id| (node_id.clone(), 0_usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = BTreeMap::<NodeId, BTreeSet<NodeId>>::new();
    for node in plan.nodes.values() {
        let dependencies = node
            .inputs
            .values()
            .filter_map(|input| match input {
                InputBinding::Link { source, .. } => Some(source),
                InputBinding::Literal { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        incoming.insert(node.id.clone(), dependencies.len());
        for dependency in dependencies {
            outgoing
                .entry(dependency.clone())
                .or_default()
                .insert(node.id.clone());
        }
    }
    let mut ready = incoming
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(node_id, _)| node_id.clone())
        .collect::<BTreeSet<_>>();
    let mut topological_order = Vec::with_capacity(plan.nodes.len());
    while let Some(node_id) = ready.pop_first() {
        topological_order.push(node_id.clone());
        if let Some(dependents) = outgoing.get(&node_id) {
            for dependent in dependents {
                let count = incoming
                    .get_mut(dependent)
                    .ok_or("Task408 topological dependency disappeared")?;
                *count = count
                    .checked_sub(1)
                    .ok_or("Task408 topological dependency underflowed")?;
                if *count == 0 {
                    ready.insert(dependent.clone());
                }
            }
        }
    }
    if topological_order.len() != plan.nodes.len() {
        return Err("Task408 custom sampler plan contains a cycle".into());
    }
    plan.topological_order = topological_order;
    let mut required = BTreeSet::new();
    let mut pending = plan.output_nodes.clone();
    while let Some(node_id) = pending.pop() {
        if !required.insert(node_id.clone()) {
            continue;
        }
        if let Some(node) = plan.nodes.get(&node_id) {
            pending.extend(node.inputs.values().filter_map(|input| match input {
                InputBinding::Link {
                    source,
                    lazy: false,
                    ..
                } => Some(source.clone()),
                InputBinding::Literal { .. } | InputBinding::Link { lazy: true, .. } => None,
            }));
        }
    }
    plan.static_required_nodes = required;
    plan.validate_integrity()
        .map_err(|error| format!("Task408 custom sampler plan integrity failed: {error:?}"))?;
    Ok((plan, custom_id))
}

#[test]
fn task408_compile_custom_sampler_plan_is_integral() -> Result<(), Box<dyn Error>> {
    let provider = Arc::new(NativeDiffusionFixture::checked_in());
    let workflow = task408_compile_workflow(&["inductor"])?;
    let plan = compile_native_diffusion_workflow(&workflow, &BTreeSet::new(), provider.clone())?;
    let provider_interface: Arc<dyn NativeDiffusionProvider> = provider;
    task408_compile_custom_sampler_plan(plan, provider_interface)?;
    Ok(())
}

#[test]
fn task594_compile_node_runtime_presentation_preserves_source_flags() -> Result<(), Box<dyn Error>>
{
    let registry = generated_native_node_registry_projection(Some(Arc::new(
        NativeDiffusionFixture::checked_in(),
    )))?;
    let presentation = registry
        .presentation("TorchCompileModel")
        .ok_or("compile-node runtime presentation missing")?;
    let source = built_in_source_schema("TorchCompileModel")?;
    assert!(source.presentation.is_experimental);
    assert_eq!(
        presentation.is_experimental,
        source.presentation.is_experimental
    );
    assert_eq!(
        presentation.is_deprecated,
        source.presentation.is_deprecated
    );
    let descriptor = registry
        .descriptor("TorchCompileModel")
        .ok_or("compile-node runtime descriptor missing")?;
    descriptor.validate_exact_schema_v2()?;
    Ok(())
}

#[test]
fn task594_compile_node_reconstructs_from_workflow_and_rejects_previous_store_handles()
-> Result<(), Box<dyn Error>> {
    let provider = Arc::new(NativeDiffusionFixture::checked_in());
    let workflow = task408_compile_workflow(&["inductor"])?;
    let persisted: Value = serde_json::from_slice(&workflow)?;
    let restored = serde_json::to_vec(&persisted)?;
    let mut plan =
        compile_native_diffusion_workflow(&restored, &BTreeSet::new(), provider.clone())?;
    let compile_node_id = NodeId("8".to_owned());
    plan.output_nodes = vec![compile_node_id.clone()];
    plan.static_required_nodes = BTreeSet::from([NodeId("1".to_owned()), compile_node_id.clone()]);
    plan.nodes
        .retain(|node_id, _| plan.static_required_nodes.contains(node_id));
    plan.topological_order
        .retain(|node_id| plan.static_required_nodes.contains(node_id));
    // Observe the compiled handle directly so recovery does not require a sampler invocation.
    plan.nodes
        .get_mut(&compile_node_id)
        .ok_or("compile descriptor missing")?
        .descriptor
        .output_node = true;
    plan.validate_integrity()?;
    let workspace_bytes = 2 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let backend = Arc::new(backend);
    let make_executor = || {
        NativeImageExecutor::new_with_diffusion_provider(
            ProfileId(Uuid::from_u128(0x594_001)),
            BTreeMap::new(),
            true,
            backend.clone(),
            provider.clone(),
        )
    };
    let executor = make_executor()?;
    let first = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x594_002)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(
        first.report.state,
        AttemptState::Succeeded,
        "{:?}",
        first.report.error
    );
    let previous_handle = match first
        .report
        .outputs
        .get(&compile_node_id)
        .and_then(|outputs| outputs.first())
    {
        Some(NativeValue::Handle { value }) => value.clone(),
        _ => return Err("compile-only workflow did not publish its MODEL".into()),
    };
    let descriptor = plan
        .nodes
        .get(&compile_node_id)
        .ok_or("compile descriptor missing")?
        .descriptor
        .clone();
    drop(executor);
    let executor = make_executor()?;
    let stale_plan = task408_compile_handle_plan(
        descriptor,
        previous_handle.clone(),
        PromptId(Uuid::from_u128(0x594_003)),
    )?;
    let stale = executor.execute_blocking(
        &stale_plan,
        AttemptId(Uuid::from_u128(0x594_004)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(stale.report.state, AttemptState::Failed);
    assert!(stale.report.outputs.is_empty());
    assert!(stale.report.ui_outputs.is_empty());
    assert!(stale.output_proposals.is_empty());
    let recovered = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x594_005)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(
        recovered.report.state,
        AttemptState::Succeeded,
        "{:?}",
        recovered.report.error
    );
    let recovered_handle = match recovered
        .report
        .outputs
        .get(&compile_node_id)
        .and_then(|outputs| outputs.first())
    {
        Some(NativeValue::Handle { value }) => value,
        _ => return Err("reconstructed workflow did not publish its MODEL".into()),
    };
    assert_ne!(
        recovered_handle.store_identity(),
        previous_handle.store_identity()
    );
    assert_eq!(
        recovered_handle.digest_sha256(),
        previous_handle.digest_sha256()
    );
    assert_eq!(
        recovered.report.ui_outputs.get(&compile_node_id),
        first.report.ui_outputs.get(&compile_node_id)
    );
    Ok(())
}

#[test]
fn task408_torch_compile_model_production_cache_and_atomic_failures() -> Result<(), Box<dyn Error>>
{
    let provider = Arc::new(NativeDiffusionFixture::checked_in());
    let workflow = task408_compile_workflow(&["inductor"])?;
    let plan = compile_native_diffusion_workflow(&workflow, &BTreeSet::new(), provider.clone())?;
    let compile_node_id = NodeId("8".to_owned());
    let workspace_bytes = 2 * 1024 * 1024 * 1024;
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        ProfileId(Uuid::from_u128(0x408_001)),
        BTreeMap::new(),
        true,
        Arc::new(backend),
        provider.clone(),
    )?;

    let first = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x408_002)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(first.report.state, AttemptState::Succeeded);
    let first_ui = first
        .report
        .ui_outputs
        .get(&compile_node_id)
        .ok_or("TorchCompileModel did not publish compile evidence")?;
    assert_eq!(
        first_ui.get("backend").and_then(Value::as_str),
        Some("inductor")
    );
    let policy_sha256 = first_ui
        .get("compile_policy_sha256")
        .and_then(Value::as_str)
        .ok_or("TorchCompileModel omitted policy identity")?;
    assert_eq!(policy_sha256.len(), 64);
    assert_ne!(
        first_ui.get("source_model_sha256"),
        first_ui.get("compiled_model_sha256")
    );
    let sampler_ui = first
        .report
        .ui_outputs
        .get(&NodeId("5".to_owned()))
        .ok_or("KSampler omitted compiled-wrapper invocation evidence")?;
    assert_eq!(
        sampler_ui
            .get("compiled_model_policy_sha256")
            .and_then(Value::as_str),
        Some(policy_sha256)
    );
    assert!(
        sampler_ui
            .get("compiled_model_invocations")
            .and_then(Value::as_u64)
            .is_some_and(|invocations| invocations > 0)
    );
    let compiled_handle = match first
        .report
        .outputs
        .get(&compile_node_id)
        .and_then(|outputs| outputs.first())
    {
        Some(NativeValue::Handle { value }) => value.clone(),
        _ => return Err("TorchCompileModel did not publish a MODEL handle".into()),
    };

    let cached = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x408_003)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(cached.report.state, AttemptState::Succeeded);
    assert!(cached.report.cache_hits > 0);
    assert_eq!(
        cached.report.ui_outputs.get(&compile_node_id),
        Some(first_ui)
    );

    let apg_workflow = task408_compile_apg_workflow()?;
    let apg_plan =
        compile_native_diffusion_workflow(&apg_workflow, &BTreeSet::new(), provider.clone())?;
    let apg = executor.execute_blocking(
        &apg_plan,
        AttemptId(Uuid::from_u128(0x408_00a)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(
        apg.report.state,
        AttemptState::Succeeded,
        "compile→APG→KSampler failed: {:?}",
        apg.report.error
    );
    let apg_sampler_ui = apg
        .report
        .ui_outputs
        .get(&NodeId("5".to_owned()))
        .ok_or("compile→APG→KSampler omitted sampler evidence")?;
    assert!(
        apg_sampler_ui
            .get("compiled_model_invocations")
            .and_then(Value::as_u64)
            .is_some_and(|invocations| invocations > 0)
    );
    assert_eq!(
        apg_sampler_ui
            .get("guidance_modifier_count")
            .and_then(Value::as_u64),
        Some(1)
    );

    let custom_source = task408_compile_workflow(&["inductor"])?;
    let custom_base =
        compile_native_diffusion_workflow(&custom_source, &BTreeSet::new(), provider.clone())?;
    let provider_interface: Arc<dyn NativeDiffusionProvider> = provider.clone();
    let (custom_plan, custom_id) =
        task408_compile_custom_sampler_plan(custom_base, provider_interface)?;
    let custom = executor.execute_blocking(
        &custom_plan,
        AttemptId(Uuid::from_u128(0x408_00b)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(
        custom.report.state,
        AttemptState::Succeeded,
        "compile→CFGGuider→SamplerCustomAdvanced failed: {:?}",
        custom.report.error
    );
    let custom_sampler_ui = custom
        .report
        .ui_outputs
        .get(&custom_id)
        .ok_or("compiled custom sampler omitted invocation evidence")?;
    assert_eq!(
        custom_sampler_ui.get("guider_kind").and_then(Value::as_str),
        Some("cfg")
    );
    assert!(
        custom_sampler_ui
            .get("compiled_model_invocations")
            .and_then(Value::as_u64)
            .is_some_and(|invocations| invocations > 0)
    );

    let invalid_workflow = task408_compile_workflow(&["python"])?;
    assert!(
        compile_native_diffusion_workflow(&invalid_workflow, &BTreeSet::new(), provider.clone(),)
            .is_err()
    );

    let cudagraph_workflow = task408_compile_workflow(&["cudagraphs"])?;
    let cudagraph_plan =
        compile_native_diffusion_workflow(&cudagraph_workflow, &BTreeSet::new(), provider.clone())?;
    let cudagraph = executor.execute_blocking(
        &cudagraph_plan,
        AttemptId(Uuid::from_u128(0x408_004)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(cudagraph.report.state, AttemptState::Failed);
    assert!(cudagraph.report.outputs.is_empty());
    assert!(cudagraph.report.ui_outputs.is_empty());
    assert!(cudagraph.output_proposals.is_empty());
    let cudagraph_error = cudagraph
        .report
        .error
        .as_deref()
        .ok_or("cudagraph rejection omitted its diagnostic")?;
    assert!(
        cudagraph_error.contains("native_model_compile_failed"),
        "unexpected cudagraph rejection: {cudagraph_error}"
    );

    let stale_handle = NativeOpaqueHandle::new(
        compiled_handle.handle_type().clone(),
        compiled_handle.store_identity(),
        compiled_handle.identifier(),
        compiled_handle
            .generation()
            .checked_add(1)
            .ok_or("compiled handle generation overflow")?,
        compiled_handle.digest_sha256().map(str::to_owned),
    )?;
    let compile_descriptor = plan
        .nodes
        .get(&compile_node_id)
        .map(|node| node.descriptor.clone())
        .ok_or("compiled TorchCompileModel descriptor is missing")?;
    let stale_plan = task408_compile_handle_plan(
        compile_descriptor,
        stale_handle,
        PromptId(Uuid::from_u128(0x408_005)),
    )?;
    let stale = executor.execute_blocking(
        &stale_plan,
        AttemptId(Uuid::from_u128(0x408_006)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(stale.report.state, AttemptState::Failed);
    assert!(stale.report.outputs.is_empty());
    assert!(stale.report.ui_outputs.is_empty());
    assert!(stale.output_proposals.is_empty());

    let replacement_workflow = task408_compile_workflow(&["inductor", "inductor"])?;
    let replacement_plan =
        compile_native_diffusion_workflow(&replacement_workflow, &BTreeSet::new(), provider)?;
    let replacement = executor.execute_blocking(
        &replacement_plan,
        AttemptId(Uuid::from_u128(0x408_007)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(
        replacement.report.state,
        AttemptState::Succeeded,
        "sequential compile replacement failed: {:?}",
        replacement.report.error
    );
    let replacement_sampler_ui = replacement
        .report
        .ui_outputs
        .get(&NodeId("5".to_owned()))
        .ok_or("replacement KSampler omitted compiled-wrapper evidence")?;
    assert!(
        replacement_sampler_ui
            .get("compiled_model_invocations")
            .and_then(Value::as_u64)
            .is_some_and(|invocations| invocations > 0)
    );

    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(
        executor
            .execute_blocking(
                &plan,
                AttemptId(Uuid::from_u128(0x408_008)),
                cancellation,
                0,
                workspace_authority.authorize_workspace(workspace_bytes)?,
            )
            .is_err()
    );
    Ok(())
}

fn task410_audio_plan(
    prompt_id: PromptId,
    sample_rate: i64,
) -> Result<CompiledPlan, Box<dyn Error>> {
    let node_id = NodeId("task410-audio".to_owned());
    let descriptor = task410_audio_fixture_descriptor_for_test()?;
    let plan = CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            CompiledNode {
                id: node_id.clone(),
                class_type: descriptor.class_type.clone(),
                descriptor,
                inputs: BTreeMap::from([(
                    "sample_rate".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Integer(sample_rate),
                        },
                    },
                )]),
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    Ok(plan)
}

#[test]
fn task410_zero_sample_audio_is_exact_recoverable_and_atomic() -> Result<(), Box<dyn Error>> {
    fn audio_tensor(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        shape: &[u64],
        values: &[f32],
    ) -> Result<Tensor, Box<dyn Error>> {
        Ok(tensor_from_f32_with_context_exact_native(
            backend,
            shape,
            values,
            DType::F32,
            backend.device(),
            context,
        )?)
    }

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let empty = NativeAudioPayload::checked(audio_tensor(&backend, &context, &[1, 1, 0], &[])?, 1)?;
    assert_eq!(empty.waveform().descriptor().shape(), [1, 1, 0]);
    assert_eq!(empty.waveform().contiguous_bytes()?.len(), 0);

    let one_second = audio_tensor(&backend, &context, &[1, 1, 8], &[0.0; 8])?;
    let requested_start = 16_u64;
    let clamped_start = requested_start.min(one_second.descriptor().shape()[2]);
    let beyond_duration = one_second.narrow_read_only(2, i64::try_from(clamped_start)?, 0)?;
    let sliced = NativeAudioPayload::checked(beyond_duration, 8)?;
    assert_eq!(sliced.waveform().descriptor().shape(), [1, 1, 0]);
    assert_eq!(sliced.waveform().contiguous_bytes()?.len(), 0);

    let maximum_rate =
        NativeAudioPayload::checked(audio_tensor(&backend, &context, &[1, 1, 0], &[])?, 384_000)?;
    let stereo =
        NativeAudioPayload::checked(audio_tensor(&backend, &context, &[1, 2, 0], &[])?, 192_000)?;
    assert_ne!(
        empty.semantic_digest_sha256(),
        maximum_rate.semantic_digest_sha256()
    );
    assert_ne!(
        empty.semantic_digest_sha256(),
        stereo.semantic_digest_sha256()
    );
    assert_eq!(
        empty.resident_parts()?.resident_bytes()?,
        empty.resident_bytes()
    );
    assert_eq!(
        stereo.resident_parts()?.resident_bytes()?,
        stereo.resident_bytes()
    );
    assert!(audio_tensor(&backend, &context, &[1, 1, 1], &[]).is_err());

    let audio_type = NativeHandleType::new(NativeHandleKind::Audio, "AUDIO")?;
    let attempt_id = AttemptId(Uuid::from_u128(0x4100));
    let generation = NativeHandleStoreGeneration::with_capacities(8, 1024 * 1024)?;
    let store = generation.handle_store_for_attempt(attempt_id);
    let handle = store.publish(
        NativeStoredPayload::Audio(Arc::new(empty.clone())),
        &cancellation,
    )?;
    let resolved = store.resolve(&handle, &audio_type, &cancellation)?;
    let NativeStoredPayload::Audio(resolved_audio) = &*resolved else {
        return Err("AUDIO handle resolved to a different payload kind".into());
    };
    assert_eq!(resolved_audio.waveform().descriptor().shape(), [1, 1, 0]);
    assert_eq!(resolved_audio.sample_rate(), 1);

    let list = NativeValue::List {
        values: vec![NativeValue::Handle {
            value: handle.clone(),
        }],
    };
    list.validate()?;
    let persisted_list: NativeValue = serde_json::from_slice(&serde_json::to_vec(&list)?)?;
    assert_eq!(persisted_list, list);

    let evidence =
        PersistedNativeAudioEvidence::checked(NodeId("task410-audio".to_owned()), &empty)?;
    let persisted_evidence: PersistedNativeAudioEvidence =
        serde_json::from_slice(&serde_json::to_vec(&evidence)?)?;
    persisted_evidence.validate()?;
    assert_eq!(persisted_evidence, evidence);
    let persisted_transport =
        ProviderTransportResponse::from_bytes(&persisted_evidence.transport_bytes)?;
    assert_eq!(persisted_transport.ports().len(), 2);
    assert_eq!(persisted_transport.ports()[0].values().len(), 1);
    assert_eq!(persisted_transport.ports()[1].values().len(), 1);

    let executor_workspace_bytes = 1024 * 1024;
    let (executor_backend, executor_authority) =
        CpuWorkspaceAuthority::create_backend(executor_workspace_bytes)?;
    let executor = NativeImageExecutor::new_with_generated_registry(
        ProfileId(Uuid::from_u128(0x4102)),
        BTreeMap::new(),
        true,
        Arc::new(executor_backend),
    )?;
    let plan = task410_audio_plan(PromptId(Uuid::from_u128(0x4103)), 1)?;
    let first_attempt = AttemptId(Uuid::from_u128(0x4104));
    let first = executor.execute_blocking(
        &plan,
        first_attempt,
        CancellationToken::default(),
        0,
        executor_authority.authorize_workspace(executor_workspace_bytes)?,
    )?;
    assert_eq!(first.report.state, AttemptState::Succeeded);
    assert_eq!(first.report.cache_hits, 0);
    let second = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x4105)),
        CancellationToken::default(),
        0,
        executor_authority.authorize_workspace(executor_workspace_bytes)?,
    )?;
    assert_eq!(second.report.state, AttemptState::Succeeded);
    assert_eq!(second.report.cache_hits, 1);
    let cached_outputs = second
        .report
        .outputs
        .get(&NodeId("task410-audio".to_owned()))
        .ok_or("Task410 cache omitted AUDIO outputs")?;
    let [
        NativeValue::Handle {
            value: cached_audio,
        },
        NativeValue::List { values },
    ] = cached_outputs.as_slice()
    else {
        return Err("Task410 cache changed AUDIO/list output types".into());
    };
    let [
        NativeValue::Handle {
            value: cached_list_audio,
        },
    ] = values.as_slice()
    else {
        return Err("Task410 cache changed AUDIO list cardinality".into());
    };
    assert_eq!(cached_audio, cached_list_audio);
    let cached_store = executor
        .handle_store_generation_for_test()
        .handle_store_for_attempt(first_attempt);
    let cached = cached_store.resolve(cached_audio, &audio_type, &cancellation)?;
    let NativeStoredPayload::Audio(cached) = cached.as_ref() else {
        return Err("Task410 cache changed AUDIO payload kind".into());
    };
    assert_eq!(cached.sample_rate(), 1);
    assert_eq!(cached.waveform().descriptor().shape(), [1, 1, 0]);

    let restarted_generation = generation.fresh_generation()?;
    let restarted_store = restarted_generation.handle_store_for_attempt(attempt_id);
    assert!(matches!(
        restarted_store.resolve(&handle, &audio_type, &cancellation),
        Err(NativeHandleStoreError::WrongGeneration)
    ));

    let oom_generation = NativeHandleStoreGeneration::with_capacities(1, 1)?;
    let oom_store = oom_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x4101)));
    assert!(matches!(
        oom_store.publish(
            NativeStoredPayload::Audio(Arc::new(maximum_rate)),
            &cancellation,
        ),
        Err(NativeHandleStoreError::Rejected(message)) if message.contains("capacity")
    ));
    assert!(oom_generation.is_empty());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let before = restarted_generation.len();
    assert!(matches!(
        restarted_store.publish(NativeStoredPayload::Audio(Arc::new(stereo)), &cancelled,),
        Err(NativeHandleStoreError::Cancelled)
    ));
    assert_eq!(restarted_generation.len(), before);
    Ok(())
}

#[test]
fn task595_attention_clip_executes_reconstructs_and_rejects_stale_handles()
-> Result<(), Box<dyn Error>> {
    let memory_limit = 128 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(memory_limit)?,
        &cancellation,
    );
    let resource = sim_model::clip::native_clip_transform_fixture(&backend, &execution)?;
    let payload = NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::model_resource(
        Arc::new(NativeModelPayload::native_clip(Arc::new(resource))?),
    )?));
    let input_digest = payload.digest_sha256();
    let generation = NativeHandleStoreGeneration::with_capacities(16, 64 * 1024 * 1024)?;
    let attempt = AttemptId(Uuid::from_u128(0x59501));
    let store = generation.handle_store_for_attempt(attempt);
    let handle = store.publish(payload.clone(), &cancellation)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("CLIPAttentionMultiply")
        .ok_or("attention node missing")?;
    let make_context = |store: Arc<dyn sim_nodes::NativeHandleStore>,
                        cancellation: CancellationToken|
     -> Result<sim_nodes::NativeNodeContext, Box<dyn Error>> {
        let scratch = authority.authorize_workspace(memory_limit)?;
        let node_id = NodeId("attention".to_owned());
        let identity = sim_nodes::NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x59502),
            attempt,
            node_id.clone(),
        )?;
        let compute = sim_nodes::NativeNodeComputeSession::checked(
            identity,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        Ok(sim_nodes::NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x59503)),
            attempt,
            node_id,
            cancellation,
            scratch,
            store,
            sim_nodes::NativeNodeServices::checked(None, None, Some(compute))?,
        )?)
    };
    let inputs = |handle: NativeOpaqueHandle, scale: f64| {
        let mut inputs =
            BTreeMap::from([("clip".to_owned(), NativeValue::Handle { value: handle })]);
        for name in ["q", "k", "v", "out"] {
            inputs.insert(
                name.to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(scale),
                },
            );
        }
        inputs
    };
    let first_inputs = inputs(handle.clone(), 2.0);
    let context = make_context(store.clone(), cancellation.clone())?;
    assert!(
        node.demanded_lazy_inputs(&context, &first_inputs)?
            .is_empty()
    );
    let dependencies = node.cache_dependencies(&context, &first_inputs)?;
    assert_eq!(
        dependencies.artifact_digests.get("attention-stored-model"),
        Some(&input_digest)
    );
    let result = smol::block_on(node.execute(context, first_inputs.clone()))?;
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = result
    else {
        return Err("attention values absent".into());
    };
    assert!(ui.is_none() && effects.is_empty());
    let Some(NativeValue::Handle { value: output }) = outputs.first() else {
        return Err("attention handle absent".into());
    };
    assert_ne!(output.digest_sha256(), handle.digest_sha256());
    let resolved = store.resolve(output, output.handle_type(), &cancellation)?;
    let NativeStoredPayload::Model(transformed) = resolved.as_ref() else {
        return Err("attention model absent".into());
    };
    let component = transformed
        .model_payload()
        .native_clip_resource()
        .and_then(|resource| resource.components().first())
        .ok_or("attention component absent")?;
    assert_eq!(
        component
            .patch_graph()
            .semantic_operations()
            .iter()
            .filter(|operation| operation.identifier.starts_with("scale-matching:"))
            .count(),
        8
    );
    assert_eq!(
        store
            .resolve(&handle, handle.handle_type(), &cancellation)?
            .as_ref()
            .digest_sha256(),
        input_digest
    );
    let persisted_parameters = serde_json::to_vec(&first_inputs)?;
    let recovered_inputs: BTreeMap<String, NativeValue> =
        serde_json::from_slice(&persisted_parameters)?;
    let recovered_generation = generation.fresh_generation()?;
    let recovered_store = recovered_generation.handle_store_for_attempt(attempt);
    let error = smol::block_on(node.execute(
        make_context(recovered_store.clone(), cancellation.clone())?,
        recovered_inputs,
    ))
    .expect_err("previous generation handles must fail");
    assert_eq!(error.kind, NativeNodeFailureKind::Failure);
    assert!(recovered_generation.is_empty());
    let fresh_handle = recovered_store.publish(payload, &cancellation)?;
    let reconstructed = smol::block_on(node.execute(
        make_context(recovered_store.clone(), cancellation.clone())?,
        inputs(fresh_handle, 2.0),
    ))?;
    let NativeNodeOutcome::Values {
        outputs: recovered, ..
    } = reconstructed
    else {
        return Err("recovered values absent".into());
    };
    let Some(NativeValue::Handle { value: recovered }) = recovered.first() else {
        return Err("recovered handle absent".into());
    };
    assert_eq!(recovered.digest_sha256(), output.digest_sha256());
    assert_ne!(recovered.store_identity(), output.store_identity());
    let before = generation.len();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let error = smol::block_on(node.execute(
        make_context(store.clone(), cancelled)?,
        first_inputs.clone(),
    ))
    .expect_err("cancelled attention must fail");
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(generation.len(), before);
    store.revoke(&handle, &cancellation)?;
    assert!(
        smol::block_on(node.execute(make_context(store, cancellation)?, first_inputs)).is_err()
    );
    Ok(())
}

#[test]
fn task606_sdpose_face_nodes_match_source_and_recover_without_partial_publication()
-> Result<(), Box<dyn Error>> {
    use sim_media::{
        NativePoseFrame, NativePoseKeypoint, NativePoseKeypointPayload, NativePosePerson,
    };
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/image-detection-comfy-node-0607/fixture.json"
    ))?;
    let cases = fixture["face_reference"]["cases"]
        .as_array()
        .ok_or("missing face cases")?;
    assert_eq!(cases.len(), 48);
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("SDPoseFaceBBoxes")
        .ok_or("missing face node")?;
    let (_, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let attempt = AttemptId(Uuid::from_u128(0x60601));
    let generation = NativeHandleStoreGeneration::with_capacities(128, 16 * 1024 * 1024)?;
    let store = generation.handle_store_for_attempt(attempt);
    let cancellation = CancellationToken::default();
    let context = |store: Arc<dyn sim_nodes::NativeHandleStore>,
                   cancellation: CancellationToken|
     -> Result<sim_nodes::NativeNodeContext, Box<dyn Error>> {
        Ok(sim_nodes::NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x60602)),
            attempt,
            NodeId("face".to_owned()),
            cancellation,
            authority.authorize_workspace(16 * 1024 * 1024)?,
            store,
        )?)
    };
    let mut reviewed = Vec::new();
    for case in cases {
        let mut frames = Vec::new();
        for frame in case["frames"].as_array().ok_or("frames absent")? {
            let mut people = Vec::new();
            for person in frame["people"].as_array().ok_or("people absent")? {
                let flat = person["face_keypoints_2d"]
                    .as_array()
                    .ok_or("face absent")?;
                let face = flat
                    .chunks_exact(3)
                    .map(|values| {
                        NativePoseKeypoint::checked(
                            values[0].as_f64().ok_or("x absent")?,
                            values[1].as_f64().ok_or("y absent")?,
                            values[2].as_f64().ok_or("score absent")?,
                        )
                        .map_err(Into::into)
                    })
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
                let point = NativePoseKeypoint::checked(-1.0, -1.0, 0.0)?;
                people.push(NativePosePerson::checked(
                    vec![point; 18],
                    vec![point; 6],
                    face,
                    vec![point; 21],
                    vec![point; 21],
                )?);
            }
            frames.push(NativePoseFrame::checked(
                u32::try_from(frame["canvas_width"].as_u64().ok_or("width absent")?)?,
                u32::try_from(frame["canvas_height"].as_u64().ok_or("height absent")?)?,
                people,
            )?);
        }
        let payload = NativeStoredPayload::PoseKeypoint(Arc::new(
            NativePoseKeypointPayload::checked(frames)?,
        ));
        let digest = payload.digest_sha256();
        let handle = store.publish(payload.clone(), &cancellation)?;
        let inputs = BTreeMap::from([
            (
                "keypoints".to_owned(),
                NativeValue::Handle {
                    value: handle.clone(),
                },
            ),
            (
                "scale".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(case["scale"].as_f64().ok_or("scale absent")?),
                },
            ),
            (
                "force_square".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Boolean(
                        case["force_square"].as_bool().ok_or("square absent")?,
                    ),
                },
            ),
        ]);
        let invocation = context(store.clone(), cancellation.clone())?;
        assert!(node.demanded_lazy_inputs(&invocation, &inputs)?.is_empty());
        assert_eq!(
            node.cache_dependencies(&invocation, &inputs)?
                .artifact_digests
                .get("keypoints"),
            Some(&digest)
        );
        let NativeNodeOutcome::Values {
            outputs,
            effects,
            ui,
        } = smol::block_on(node.execute(invocation, inputs.clone()))?
        else {
            return Err("face output absent".into());
        };
        assert!(effects.is_empty() && ui.is_none());
        let Some(NativeValue::Handle { value: output }) = outputs.first() else {
            return Err("face handle absent".into());
        };
        let resolved = store.resolve(output, output.handle_type(), &cancellation)?;
        let NativeStoredPayload::BoundingBox(boxes) = resolved.as_ref() else {
            return Err("boxes absent".into());
        };
        let actual = boxes.frames().iter().map(|frame| frame.iter().map(|value| json!({"x":value.x() as u64,"y":value.y() as u64,"width":value.width() as u64,"height":value.height() as u64})).collect::<Vec<_>>()).collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(&actual)?,
            case["expected"],
            "{}",
            case["id"]
        );
        assert_eq!(
            store
                .resolve(&handle, handle.handle_type(), &cancellation)?
                .digest_sha256(),
            digest
        );
        reviewed.push(json!({"id":case["id"],"frames":case["frames"],"scale":case["scale"],"force_square":case["force_square"],"actual":actual}));
        if case["id"] == "rectangle-scale-1.5-square-True" {
            let tight = NativeHandleStoreGeneration::with_capacities(1, 16 * 1024 * 1024)?;
            let tight_store = tight.handle_store_for_attempt(attempt);
            let tight_input = tight_store.publish(payload.clone(), &cancellation)?;
            let mut tight_inputs = inputs.clone();
            tight_inputs.insert(
                "keypoints".to_owned(),
                NativeValue::Handle {
                    value: tight_input.clone(),
                },
            );
            assert!(
                smol::block_on(node.execute(
                    context(tight_store.clone(), cancellation.clone())?,
                    tight_inputs
                ))
                .is_err()
            );
            assert_eq!(tight.len(), 1);
            assert_eq!(
                tight_store
                    .resolve(&tight_input, tight_input.handle_type(), &cancellation)?
                    .digest_sha256(),
                digest
            );
        }
        let persisted = serde_json::to_vec(&inputs)?;
        let restored: BTreeMap<String, NativeValue> = serde_json::from_slice(&persisted)?;
        let recovery = generation.fresh_generation()?;
        let recovered_store = recovery.handle_store_for_attempt(attempt);
        assert!(
            smol::block_on(node.execute(
                context(recovered_store.clone(), cancellation.clone())?,
                restored
            ))
            .is_err()
        );
        assert!(recovery.is_empty());
        let fresh = recovered_store.publish(payload, &cancellation)?;
        let mut recovered_inputs = inputs.clone();
        recovered_inputs.insert("keypoints".to_owned(), NativeValue::Handle { value: fresh });
        let NativeNodeOutcome::Values {
            outputs: recovered, ..
        } = smol::block_on(node.execute(
            context(recovered_store, cancellation.clone())?,
            recovered_inputs,
        ))?
        else {
            return Err("recovered face output absent".into());
        };
        let Some(NativeValue::Handle { value: recovered }) = recovered.first() else {
            return Err("recovered face handle absent".into());
        };
        assert_eq!(recovered.digest_sha256(), output.digest_sha256());
        let before = generation.len();
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert_eq!(
            smol::block_on(node.execute(context(store.clone(), cancelled)?, inputs.clone()))
                .expect_err("cancelled face node")
                .kind,
            NativeNodeFailureKind::Interrupted
        );
        assert_eq!(generation.len(), before);
        store.revoke(&handle, &cancellation)?;
        assert!(
            smol::block_on(node.execute(context(store.clone(), cancellation.clone())?, inputs))
                .is_err()
        );
        store.revoke(output, &cancellation)?;
    }
    fs::write(
        std::env::temp_dir().join("sim-sdpose-native-face-review.json"),
        serde_json::to_vec_pretty(&reviewed)?,
    )?;
    Ok(())
}

#[test]
fn task606_sdpose_extractor_engine_reuses_canonical_owner_and_rejects_wrong_model()
-> Result<(), Box<dyn Error>> {
    use sim_media::{NativeBoundingBox, NativeBoundingBoxPayload};
    use sim_sampler::NativeDiffusionPayload;
    let memory_limit = 256 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(memory_limit)?,
        &cancellation,
    );
    let bundle = NativeDiffusionFixture::checked_in()
        .load_bundle_with_context(backend.clone(), &execution)?;
    let vae = NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::native_diffusion(
        Arc::new(NativeDiffusionPayload::vae(Arc::new(
            NativeModelPayload::native_vae(bundle.vae().clone())?,
        ))?),
    )?));
    let image = ImageTensor::from_tensor(tensor_from_f32_with_context_exact_native(
        &backend,
        &[2, 8, 8, 3],
        &vec![0.25; 2 * 8 * 8 * 3],
        DType::F32,
        DeviceId::CPU,
        &execution,
    )?)?;
    let image = NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
        NativeTensorRole::Image,
        image,
    )?));
    let boxes =
        NativeStoredPayload::BoundingBox(Arc::new(NativeBoundingBoxPayload::checked(vec![vec![
            NativeBoundingBox::checked(8.0, 0.0, 1.0, 1.0, None, None)?,
        ]])?));
    let model = reduced_sdpose_stored_payload()?;
    let model_digest = model.digest_sha256();
    let executor = NativeImageExecutor::new_with_generated_registry(
        ProfileId(Uuid::from_u128(0x60610)),
        BTreeMap::new(),
        true,
        backend.clone(),
    )?;
    let attempt = AttemptId(Uuid::from_u128(0x60611));
    let store = executor
        .handle_store_generation_for_test()
        .handle_store_for_attempt(attempt);
    let input_payloads = [model, vae, image, boxes];
    let handles = input_payloads
        .iter()
        .cloned()
        .map(|payload| store.publish(payload, &cancellation))
        .collect::<Result<Vec<_>, _>>()?;
    let mut inputs = ["model", "vae", "image", "bboxes"]
        .into_iter()
        .zip(handles.iter().cloned())
        .map(|(name, value)| {
            (
                name.to_owned(),
                InputBinding::Literal {
                    value: NativeValue::Handle { value },
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    inputs.insert(
        "batch_size".to_owned(),
        InputBinding::Literal {
            value: NativeValue::Primitive {
                value: NativePrimitive::Integer(1),
            },
        },
    );
    let registry = generated_native_node_registry_projection(None)?;
    let mut descriptor = registry
        .descriptor("SDPoseKeypointExtractor")
        .ok_or("extractor descriptor absent")?
        .clone();
    descriptor.output_node = true;
    let node_id = NodeId("extract".to_owned());
    let mut plan = CompiledPlan {
        prompt_id: PromptId(Uuid::from_u128(0x60612)),
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            CompiledNode {
                id: node_id.clone(),
                class_type: "SDPoseKeypointExtractor".to_owned(),
                descriptor,
                inputs,
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id.clone()],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    let result = executor.execute_blocking(
        &plan,
        attempt,
        cancellation.clone(),
        0,
        authority.authorize_workspace(memory_limit)?,
    )?;
    assert_eq!(
        result.report.state,
        AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    let Some(NativeValue::Handle { value: output }) = result
        .report
        .outputs
        .get(&node_id)
        .and_then(|values| values.first())
    else {
        return Err("pose output missing".into());
    };
    let resolved = store.resolve(output, output.handle_type(), &cancellation)?;
    let NativeStoredPayload::PoseKeypoint(pose) = resolved.as_ref() else {
        return Err("wrong pose output".into());
    };
    assert_eq!(pose.frames().len(), 2);
    assert!(pose.frames().iter().all(|frame| frame.canvas_width() == 8
        && frame.canvas_height() == 8
        && frame.people().is_empty()));
    assert_eq!(
        input_payloads
            .first()
            .ok_or("model payload missing")?
            .digest_sha256(),
        model_digest
    );
    let wrong_conditioning = sim_sampler::NativeConditioningPayload::checked_sd15(
        bundle.model_digest(),
        bundle.model().as_ref(),
        bundle.conditioning().patch_graph().clone(),
        None,
    )?;
    let wrong_model = NativeStoredPayload::Model(Arc::new(
        NativeStoredModelPayload::native_diffusion(Arc::new(NativeDiffusionPayload::model(
            Arc::new(NativeModelPayload::sd15_model(bundle.model().clone())?),
            Arc::new(wrong_conditioning),
        )?))?,
    ));
    let attempt = AttemptId(Uuid::from_u128(0x60613));
    let store = executor
        .handle_store_generation_for_test()
        .handle_store_for_attempt(attempt);
    for (index, (name, payload)) in ["model", "vae", "image", "bboxes"]
        .into_iter()
        .zip(input_payloads)
        .enumerate()
    {
        let payload = if index == 0 {
            wrong_model.clone()
        } else {
            payload
        };
        let handle = store.publish(payload, &cancellation)?;
        plan.nodes
            .get_mut(&node_id)
            .ok_or("plan node absent")?
            .inputs
            .insert(
                name.to_owned(),
                InputBinding::Literal {
                    value: NativeValue::Handle { value: handle },
                },
            );
    }
    let result = executor.execute_blocking(
        &plan,
        attempt,
        cancellation.clone(),
        0,
        authority.authorize_workspace(memory_limit)?,
    )?;
    assert_eq!(result.report.state, AttemptState::Failed);
    assert!(result.report.outputs.is_empty());
    Ok(())
}

#[test]
fn task606_sdpose_fixture_executes_timestep_embedding_and_heatmap_head()
-> Result<(), Box<dyn Error>> {
    let memory_limit = 64 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(memory_limit)?,
        &cancellation,
    );
    let NativeStoredPayload::Model(payload) = sdpose_stored_payload_fixture(
        SdPoseSd2Configuration::reduced_fixture(2, 1024, 1, 1, 8, 8)?,
        SdPoseHeatmapHeadConfiguration::reduced_fixture(4, 2, 133)?,
    )?
    else {
        return Err("SDPose model fixture absent".into());
    };
    let model = payload
        .model_payload()
        .sdpose_model_resource()
        .ok_or("SDPose resource absent")?;
    let latent = sdpose_tensor(&backend, &[1, 4, 8, 8], &context)?;
    let (conditioning, adm) =
        sim_model::sdpose::prepare_lotus_sdpose_conditioning(&backend, 1, DType::F32, &context)?;
    let output =
        model
            .denoiser()
            .forward(&backend, &latent, &[999.0], &conditioning, &adm, &context)?;
    let heatmaps = model
        .heatmap_head()
        .forward(&backend, output.feature_640(), &context)?;
    assert_eq!(heatmaps.descriptor().shape(), &[1, 133, 16, 16]);
    assert!(heatmaps.contiguous_bytes()?.iter().all(|value| *value == 0));
    Ok(())
}

#[test]
fn task606_sdpose_keypoint_extractor_runs_canonical_inference() -> Result<(), Box<dyn Error>> {
    sdpose_extraction_fixture(false)
}

#[test]
fn task606_sdpose_keypoint_extractor_failure_boundaries() -> Result<(), Box<dyn Error>> {
    sdpose_extraction_fixture(true)
}

fn sdpose_extraction_fixture(boundaries_only: bool) -> Result<(), Box<dyn Error>> {
    use sim_nodes::{
        NativeNodeComputeSession, NativeNodeContext, NativeNodeServiceIdentity, NativeNodeServices,
    };
    use sim_runtime::NativeRuntimeSdPoseService;
    let memory_limit = 4 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(memory_limit)?,
        &cancellation,
    );
    let model = sdpose_stored_payload_fixture(
        SdPoseSd2Configuration::reduced_fixture(2, 1024, 1, 1, 128, 96)?,
        SdPoseHeatmapHeadConfiguration::reduced_fixture(4, 2, 133)?,
    )?;
    let mut vae_state = Vec::new();
    for specification in sim_model::vae_image::image_vae_source_state_schema(
        &VaeKernelProfile::TaesdV1,
        &VaeLoaderConfiguration::Automatic,
        DType::F32,
    )? {
        let tensor = if specification.name == "vae_scale" {
            tensor_from_f32_with_context_exact_native(
                &backend,
                &[],
                &[1.0],
                DType::F32,
                DeviceId::CPU,
                &execution,
            )?
        } else {
            sdpose_tensor(&backend, &specification.shape, &execution)?
        };
        vae_state.push((specification.name, tensor));
    }
    let sim_model::vae::NativeVaeRecipeResource::Native(vae) =
        sim_model::vae::load_native_vae_recipe(
            NativeModelRecipeSourceForm::Standalone,
            &"6".repeat(64),
            vae_state,
            None,
            &backend,
            &execution,
        )?
    else {
        return Err("image VAE fixture absent".into());
    };
    let vae = NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::native_diffusion(
        Arc::new(sim_sampler::NativeDiffusionPayload::vae(Arc::new(
            NativeModelPayload::native_vae(Arc::new(vae))?,
        ))?),
    )?));
    let image = ImageTensor::from_f32(&backend, &execution, 1, 8, 8, 3, &vec![0.25; 8 * 8 * 3])?;
    let image = NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
        NativeTensorRole::Image,
        image,
    )?));
    let attempt = AttemptId(Uuid::from_u128(0x60620));
    let generation = NativeHandleStoreGeneration::with_capacities(16, 128 * 1024 * 1024)?;
    let store = generation.handle_store_for_attempt(attempt);
    let mut inputs = BTreeMap::new();
    for (name, payload) in [("model", model), ("vae", vae), ("image", image)] {
        inputs.insert(
            name.to_owned(),
            NativeValue::Handle {
                value: store.publish(payload, &cancellation)?,
            },
        );
    }
    inputs.insert(
        "batch_size".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(1),
        },
    );
    let node_id = NodeId("extract".to_owned());
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x60621), attempt, node_id.clone())?;
    let scratch = authority.authorize_workspace(memory_limit)?;
    let compute = NativeNodeComputeSession::checked(
        identity.clone(),
        backend.clone(),
        StreamId::DEFAULT,
        &scratch,
    )?;
    let context = NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x60622)),
        attempt,
        node_id,
        cancellation.clone(),
        scratch,
        store.clone(),
        NativeNodeServices::checked(None, None, Some(compute))?
            .with_sdpose(Arc::new(NativeRuntimeSdPoseService::new(identity)))?,
    )?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("SDPoseKeypointExtractor")
        .ok_or("extractor absent")?;
    let original_count = generation.len();
    let mut cancelled_context = context.clone();
    cancelled_context.cancellation = CancellationToken::default();
    cancelled_context.cancellation.cancel();
    assert_eq!(
        smol::block_on(node.execute(cancelled_context, inputs.clone()))
            .expect_err("cancelled extraction")
            .kind,
        NativeNodeFailureKind::Interrupted
    );
    assert_eq!(generation.len(), original_count);
    let mut invalid_inputs = inputs.clone();
    invalid_inputs.insert(
        "batch_size".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(0),
        },
    );
    assert!(smol::block_on(node.execute(context.clone(), invalid_inputs)).is_err());
    assert_eq!(generation.len(), original_count);
    for size in [-1, 0, 10_001] {
        let mut invalid_inputs = inputs.clone();
        invalid_inputs.insert(
            "batch_size".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(size),
            },
        );
        assert!(node.cache_change_token(&invalid_inputs).is_err());
    }
    let mut upper_bound = inputs.clone();
    upper_bound.insert(
        "batch_size".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(10_000),
        },
    );
    node.cache_change_token(&upper_bound)?;
    let mut list_input = inputs.clone();
    list_input.insert("image".to_owned(), NativeValue::List { values: Vec::new() });
    assert!(node.cache_change_token(&list_input).is_err());
    let missing_service = NativeNodeContext::new(
        context.prompt_id,
        attempt,
        context.node_id.clone(),
        cancellation.clone(),
        authority.authorize_workspace(memory_limit)?,
        store.clone(),
    )?;
    assert_eq!(
        smol::block_on(node.execute(missing_service, inputs.clone()))
            .expect_err("missing SDPose service")
            .code,
        "native_sdpose_unavailable"
    );
    assert_eq!(generation.len(), original_count);
    let stale_identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x60623),
        AttemptId(Uuid::from_u128(0x60624)),
        NodeId("extract".to_owned()),
    )?;
    let stale_context = NativeNodeContext::new_with_services(
        context.prompt_id,
        attempt,
        context.node_id.clone(),
        cancellation.clone(),
        authority.authorize_workspace(memory_limit)?,
        store.clone(),
        NativeNodeServices::default()
            .with_sdpose(Arc::new(NativeRuntimeSdPoseService::new(stale_identity)))?,
    )?;
    assert_eq!(
        smol::block_on(node.execute(stale_context, inputs.clone()))
            .expect_err("stale extraction service")
            .code,
        "native_sdpose_stale_service"
    );
    assert_eq!(generation.len(), original_count);
    #[derive(Debug)]
    struct CancelAfterExtraction(NativeNodeServiceIdentity, bool);
    impl sim_nodes::NativeSdPoseService for CancelAfterExtraction {
        fn identity(&self) -> &NativeNodeServiceIdentity {
            &self.0
        }
        fn extract(
            &self,
            context: &NativeNodeContext,
            _request: &sim_nodes::NativeSdPoseRequest,
        ) -> Result<sim_media::NativePoseKeypointPayload, NativeNodeFailure> {
            context.cancellation.cancel();
            if self.1 {
                return Err(NativeNodeFailure {
                    code: "fixture_failed_after_cancel".to_owned(),
                    message: "Injected failure after cancellation".to_owned(),
                    kind: NativeNodeFailureKind::Failure,
                    retryable: false,
                });
            }
            sim_media::NativePoseKeypointPayload::checked(Vec::new()).map_err(|error| {
                NativeNodeFailure {
                    code: "fixture_pose_invalid".to_owned(),
                    message: error.to_string(),
                    kind: NativeNodeFailureKind::Failure,
                    retryable: false,
                }
            })
        }
    }
    let identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x60626),
        attempt,
        context.node_id.clone(),
    )?;
    for fail in [false, true] {
        let post_cancel_context = NativeNodeContext::new_with_services(
            context.prompt_id,
            attempt,
            context.node_id.clone(),
            CancellationToken::default(),
            authority.authorize_workspace(memory_limit)?,
            store.clone(),
            NativeNodeServices::default()
                .with_sdpose(Arc::new(CancelAfterExtraction(identity.clone(), fail)))?,
        )?;
        assert_eq!(
            smol::block_on(node.execute(post_cancel_context, inputs.clone()))
                .expect_err("cancelled result publication")
                .kind,
            NativeNodeFailureKind::Interrupted
        );
        assert_eq!(generation.len(), original_count);
    }
    let tiny_scratch = authority.authorize_workspace(1)?;
    let identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x60625),
        attempt,
        context.node_id.clone(),
    )?;
    let tiny_compute = NativeNodeComputeSession::checked(
        identity.clone(),
        backend.clone(),
        StreamId::DEFAULT,
        &tiny_scratch,
    )?;
    let tiny_context = NativeNodeContext::new_with_services(
        context.prompt_id,
        attempt,
        context.node_id.clone(),
        cancellation.clone(),
        tiny_scratch,
        store.clone(),
        NativeNodeServices::checked(None, None, Some(tiny_compute))?
            .with_sdpose(Arc::new(NativeRuntimeSdPoseService::new(identity)))?,
    )?;
    assert!(smol::block_on(node.execute(tiny_context, inputs.clone())).is_err());
    assert_eq!(generation.len(), original_count);
    if boundaries_only {
        return Ok(());
    }
    let NativeNodeOutcome::Values {
        outputs,
        effects,
        ui,
    } = smol::block_on(node.execute(context, inputs))?
    else {
        return Err("pose output absent".into());
    };
    assert!(effects.is_empty() && ui.is_none());
    let Some(NativeValue::Handle { value: output }) = outputs.first() else {
        return Err("pose handle absent".into());
    };
    let resolved = store.resolve(output, output.handle_type(), &cancellation)?;
    let NativeStoredPayload::PoseKeypoint(pose) = resolved.as_ref() else {
        return Err("pose payload absent".into());
    };
    assert_eq!(pose.frames().len(), 1);
    let frame = pose.frames().first().ok_or("frame absent")?;
    assert_eq!((frame.canvas_width(), frame.canvas_height()), (8, 8));
    assert_eq!(frame.people().len(), 1);
    let person = frame.people().first().ok_or("person absent")?;
    assert_eq!(
        (
            person.pose().len(),
            person.foot().len(),
            person.face().len(),
            person.hand_left().len(),
            person.hand_right().len()
        ),
        (18, 6, 70, 21, 21)
    );
    for point in person
        .pose()
        .iter()
        .chain(person.foot())
        .chain(person.face())
        .chain(person.hand_left())
        .chain(person.hand_right())
    {
        assert_eq!((point.x(), point.y(), point.score()), (-1.0, -1.0, 0.0));
    }
    assert_eq!(generation.len(), 4);
    Ok(())
}
