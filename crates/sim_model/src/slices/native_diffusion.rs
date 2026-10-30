pub use crate::clip::Sd1Tokenizer;
use crate::clip::{
    ClipError, LoadedSd1Clip, Sd1ClipArtifactProfile, Sd1ClipExecutionBinding, TokenizerIdentity,
};
use crate::{
    ArtifactIndex, ArtifactIndexError, ArtifactRecord, AttentionBackend, AttentionError,
    AttentionFallbackPolicy, AttentionRequest, ClipTextError, ImageVaeError, LatentExtent,
    LatentFormatError, LoadedModel, MappedModelWeights, ModelDetectionRule, ModelFamilyIdentity,
    ModelProbe, ModelStore, ModelStoreError, ModelTokenizerDescriptor, NativeOpsError, NativeVae,
    NativeVisionModelError, PatchGraph, PatchGraphError, PatchGraphIdentity, QuantizationError,
    VaeArchitectureError, VaeArchitectureIdentity, VaeBoundary, VaeDescriptor, VaeError,
    VaeKernelProfile,
    controlnet::ControlResult,
    detect_model_family_rules, empty_latent as canonical_empty_latent,
    generated_sd15_sim_model_0045::LATENT_FORMAT as SD15_LATENT_FORMAT,
    project_latent_preview as canonical_project_latent_preview,
    scaled_dot_product_attention_with_context,
    vae_image::{
        load_image_vae_from_model_store_with_context, sd15_reduced_vae_source_state_schema,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DeviceId, ExecutionContext, Tensor,
    generated_activation_normalization_functional_01::FunctionalError,
    generated_sim_operator_indirection_01::OperatorIndirectionError,
    generated_native_diffusion::{
        NativeDiffusionTensorError, add, add_channel_bias, concat_channels, conv2d, group_norm,
        linear, nearest_upsample_2x, silu, tensor_from_f32, tensor_to_f32,
    },
    generated_neural_network_module_02::NeuralNetworkModulePartTwoError,
    generated_shape_layout_transform_02::ShapeLayoutTransformPartTwoError,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use thiserror::Error;

pub const SD15_FEATURE_ID: &str = "COMFY-MODEL-0117";
pub const SD15_TINY_FIXTURE_ID: &str = "sd15-tiny-v1";
pub const SD15_TINY_WIDTH: usize = 32;
pub const SD15_TOKEN_COUNT: usize = crate::clip::SD1_CONTEXT_LENGTH;
pub const SD15_VOCAB_SIZE: usize = crate::clip::SD1_VOCABULARY_SIZE;

pub fn sd15_model_family_identity() -> Result<ModelFamilyIdentity, NativeDiffusionModelError> {
    ModelFamilyIdentity::new(SD15_FEATURE_ID, "SD15", "sd15-v1")
        .map_err(|_| NativeDiffusionModelError::UnsupportedFamily)
}

pub fn sd15_latent_format_identity()
-> Result<crate::LatentFormatIdentity, NativeDiffusionModelError> {
    crate::LatentFormatIdentity::new(SD15_LATENT_FORMAT.feature_id, SD15_LATENT_FORMAT.identifier)
        .map_err(|error| NativeDiffusionModelError::LatentAdapter(error.to_string()))
}

const SD15_CLIP_SOURCE_PREFIX: &str = "cond_stage_model.transformer.";
const UNET_PREFIX: &str = "model.diffusion_model";
const VAE_PREFIX: &str = "first_stage_model";
const SD15_INPUT_SHAPE: [u64; 4] = [320, 4, 3, 3];
const SD15_TIME_SHAPE: [u64; 2] = [1280, 320];
const SD15_OUTPUT_SHAPE: [u64; 4] = [4, 320, 3, 3];
const SD15_TOKEN_SHAPE: [u64; 2] = [49_408, 768];
const SD15_QUANT_SHAPE: [u64; 4] = [8, 8, 1, 1];
const SD15_DETECTION_RULES: [ModelDetectionRule; 11] = [
    ModelDetectionRule::Metadata {
        key: "model_channels",
        value: "320",
        score: 10,
    },
    ModelDetectionRule::Metadata {
        key: "context_dim",
        value: "768",
        score: 10,
    },
    ModelDetectionRule::Metadata {
        key: "adm_in_channels",
        value: "none",
        score: 10,
    },
    ModelDetectionRule::Metadata {
        key: "use_linear_in_transformer",
        value: "false",
        score: 10,
    },
    ModelDetectionRule::Metadata {
        key: "use_temporal_attention",
        value: "false",
        score: 10,
    },
    ModelDetectionRule::Metadata {
        key: "feature_id",
        value: SD15_FEATURE_ID,
        score: 50,
    },
    ModelDetectionRule::ExactShape {
        key: "model.diffusion_model.input_blocks.0.0.weight",
        shape: &SD15_INPUT_SHAPE,
        score: 100,
    },
    ModelDetectionRule::ExactShape {
        key: "model.diffusion_model.time_embed.0.weight",
        shape: &SD15_TIME_SHAPE,
        score: 100,
    },
    ModelDetectionRule::ExactShape {
        key: "model.diffusion_model.out.2.weight",
        shape: &SD15_OUTPUT_SHAPE,
        score: 100,
    },
    ModelDetectionRule::ExactShape {
        key: "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
        shape: &SD15_TOKEN_SHAPE,
        score: 100,
    },
    ModelDetectionRule::ExactShape {
        key: "first_stage_model.quant_conv.weight",
        shape: &SD15_QUANT_SHAPE,
        score: 100,
    },
];

pub fn empty_sd15_latent(
    backend: &CpuBackend,
    batch: u64,
    width: u64,
    height: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionModelError> {
    canonical_empty_latent(
        &SD15_LATENT_FORMAT,
        backend,
        LatentExtent::TwoDimensional {
            batch,
            width,
            height,
        },
        sim_tensor::DType::F32,
        context.stream,
        context,
    )
    .map_err(|error| map_empty_sd15_latent_error(error, batch, width, height))
}

pub fn sd15_latent_preview(
    backend: &CpuBackend,
    latent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionModelError> {
    canonical_project_latent_preview(&SD15_LATENT_FORMAT, backend, latent, context)
        .map_err(map_sd15_preview_error)
}

fn map_empty_sd15_latent_error(
    error: LatentFormatError,
    batch: u64,
    width: u64,
    height: u64,
) -> NativeDiffusionModelError {
    match error {
        LatentFormatError::ExtentDimensions { .. } | LatentFormatError::InvalidExtent { .. } => {
            NativeDiffusionModelError::InvalidLatentDimensions {
                batch,
                width,
                height,
            }
        }
        LatentFormatError::Tensor(sim_tensor::TensorError::Cancelled) => {
            NativeDiffusionModelError::Cancelled
        }
        LatentFormatError::Tensor(error) => NativeDiffusionModelError::TensorBackend(error),
        error => NativeDiffusionModelError::LatentAdapter(error.to_string()),
    }
}

fn map_sd15_preview_error(error: LatentFormatError) -> NativeDiffusionModelError {
    match error {
        LatentFormatError::InvalidShape {
            expected, actual, ..
        } => NativeDiffusionModelError::InputShape {
            name: "SD15 preview latent",
            expected,
            actual,
        },
        LatentFormatError::Tensor(sim_tensor::TensorError::Cancelled) => {
            NativeDiffusionModelError::Cancelled
        }
        LatentFormatError::Tensor(error) => {
            NativeDiffusionModelError::Tensor(NativeDiffusionTensorError::Tensor(error))
        }
        error => NativeDiffusionModelError::LatentAdapter(error.to_string()),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Sd15DetectorProjection {
    pub feature_id: String,
    pub model_channels: u64,
    pub context_dim: u64,
    pub adm_in_channels: Option<u64>,
    pub use_linear_in_transformer: bool,
    pub use_temporal_attention: bool,
    pub source_shapes: BTreeMap<String, Vec<u64>>,
}

impl Sd15DetectorProjection {
    fn model_probe(&self) -> ModelProbe {
        ModelProbe {
            tensor_shapes: self.source_shapes.clone(),
            metadata: BTreeMap::from([
                ("feature_id".to_owned(), self.feature_id.clone()),
                ("model_channels".to_owned(), self.model_channels.to_string()),
                ("context_dim".to_owned(), self.context_dim.to_string()),
                (
                    "adm_in_channels".to_owned(),
                    self.adm_in_channels
                        .map_or_else(|| "none".to_owned(), |value| value.to_string()),
                ),
                (
                    "use_linear_in_transformer".to_owned(),
                    self.use_linear_in_transformer.to_string(),
                ),
                (
                    "use_temporal_attention".to_owned(),
                    self.use_temporal_attention.to_string(),
                ),
            ]),
        }
    }

    pub fn detect(&self) -> Result<&'static str, NativeDiffusionModelError> {
        let identity = sd15_model_family_identity()?;
        let probe = self.model_probe();
        detect_model_family_rules(identity, &SD15_DETECTION_RULES, &probe)
            .map(|_| SD15_FEATURE_ID)
            .map_err(|_| NativeDiffusionModelError::UnsupportedFamily)
    }
}

#[cfg(feature = "test-support")]
#[derive(Clone, Debug)]
pub struct ReducedFixtureAdmission {
    fixture_id: String,
    detector_transcript_sha256: String,
}

#[cfg(feature = "test-support")]
pub fn admit_reduced_fixture(
    fixture_id: &str,
    detector_transcript_sha256: &str,
) -> Result<ReducedFixtureAdmission, NativeDiffusionModelError> {
    if fixture_id != SD15_TINY_FIXTURE_ID
        || detector_transcript_sha256.len() != 64
        || !detector_transcript_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(NativeDiffusionModelError::InvalidFixtureAdmission);
    }
    Ok(ReducedFixtureAdmission {
        fixture_id: fixture_id.to_owned(),
        detector_transcript_sha256: detector_transcript_sha256.to_owned(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeightSpec {
    pub key: String,
    pub shape: Vec<u64>,
}

pub fn sd15_clip_artifact_profile() -> Result<Sd1ClipArtifactProfile, NativeDiffusionModelError> {
    Sd1ClipArtifactProfile::checked(
        SD15_CLIP_SOURCE_PREFIX,
        SD15_VOCAB_SIZE,
        SD15_TOKEN_COUNT,
        SD15_TINY_WIDTH,
        SD15_TINY_WIDTH * 4,
        2,
        4,
    )
    .map_err(map_sd15_clip_error)
}

pub fn bind_sd15_empty_patch_execution(
    artifact_digest: &str,
) -> Result<(PatchGraph, String), NativeDiffusionModelError> {
    let patch_graph = PatchGraph::checked_semantic(artifact_digest, Vec::new())
        .map_err(|error| NativeDiffusionModelError::Patch(error.to_string()))?;
    let mapped =
        MappedModelWeights::from_parts(artifact_digest.to_owned(), BTreeMap::new(), Vec::new())
            .with_patch_graph_identity(&patch_graph.identity().ordered_digest)
            .map_err(|error| NativeDiffusionModelError::Patch(error.to_string()))?;
    Ok((patch_graph, mapped.cache_identity().to_owned()))
}

pub fn bind_sd15_clip_execution(
    projection: &Sd15DetectorProjection,
    artifact_digest: &str,
    tokenizer_identity: TokenizerIdentity,
) -> Result<(Sd1ClipArtifactProfile, Sd1ClipExecutionBinding), NativeDiffusionModelError> {
    projection.detect()?;
    let family = sd15_model_family_identity()?;
    let patch_graph = PatchGraph::checked_semantic(artifact_digest, Vec::new())
        .map_err(|error| NativeDiffusionModelError::Clip(error.to_string()))?;
    let profile = sd15_clip_artifact_profile()?;
    let binding = profile
        .bind_execution(family, artifact_digest, &patch_graph, tokenizer_identity)
        .map_err(map_sd15_clip_error)?;
    Ok((profile, binding))
}

#[allow(clippy::too_many_arguments)]
pub fn load_sd15_clip_execution(
    store: &ModelStore,
    index: &ArtifactIndex,
    loaded: &LoadedModel,
    projection: &Sd15DetectorProjection,
    tokenizer_identity: TokenizerIdentity,
    backend: Arc<CpuBackend>,
    context: &ExecutionContext<'_>,
) -> Result<LoadedSd1Clip, NativeDiffusionModelError> {
    let (profile, binding) =
        bind_sd15_clip_execution(projection, loaded.identity(), tokenizer_identity)?;
    LoadedSd1Clip::from_model_store(&profile, binding, store, index, loaded, backend, context)
        .map_err(map_sd15_clip_error)
}

pub fn bind_sd15_vae_execution(
    projection: &Sd15DetectorProjection,
    artifact: &ArtifactRecord,
) -> Result<VaeDescriptor, NativeDiffusionModelError> {
    projection.detect()?;
    let patch = PatchGraph::checked_semantic(&artifact.sha256, Vec::new())
        .map_err(|error| NativeDiffusionModelError::Vae(error.to_string()))?
        .identity();
    VaeDescriptor::checked(
        artifact,
        sd15_model_family_identity()?,
        &SD15_LATENT_FORMAT,
        VaeArchitectureIdentity::checked("sim.ldm.models.autoencoder.AutoencoderKL.reduced.v1")
            .map_err(|error| NativeDiffusionModelError::Vae(error.to_string()))?,
        patch,
        DType::F32,
        DeviceId::CPU,
        VaeBoundary::image(3).map_err(|error| NativeDiffusionModelError::Vae(error.to_string()))?,
        VaeKernelProfile::Sd15AutoencoderKlReducedV1,
        [0.0, 1.0],
    )
    .map_err(|error| NativeDiffusionModelError::Vae(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
pub fn load_sd15_vae_execution(
    store: &ModelStore,
    index: &ArtifactIndex,
    loaded: Arc<LoadedModel>,
    artifact: &ArtifactRecord,
    projection: &Sd15DetectorProjection,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, NativeDiffusionModelError> {
    let descriptor = bind_sd15_vae_execution(projection, artifact)?;
    load_image_vae_from_model_store_with_context(
        backend,
        store,
        index,
        loaded,
        descriptor,
        &SD15_LATENT_FORMAT,
        context,
    )
    .map_err(|error| match image_vae_load_failure(&error) {
        Some(class) => classified_load_error(error.to_string(), class),
        None => NativeDiffusionModelError::Vae(error.to_string()),
    })
}

pub fn sd15_tiny_weight_manifest() -> Result<Vec<WeightSpec>, NativeDiffusionModelError> {
    let mut weights = Vec::new();
    weights.extend(
        sd15_clip_artifact_profile()?
            .parameters()
            .iter()
            .map(|parameter| WeightSpec {
                key: parameter.name().to_owned(),
                shape: parameter.shape().to_vec(),
            }),
    );
    add_unet_specs(&mut weights);
    weights.extend(
        sd15_reduced_vae_source_state_schema(DType::F32)
            .into_iter()
            .map(|state| WeightSpec {
                key: format!("{VAE_PREFIX}.{}", state.name),
                shape: state.shape,
            }),
    );
    weights.sort_by(|left, right| left.key.cmp(&right.key));
    let mut names = BTreeSet::new();
    if weights
        .iter()
        .any(|weight| !names.insert(weight.key.clone()))
    {
        return Err(NativeDiffusionModelError::DuplicateWeight);
    }
    Ok(weights)
}

fn parameter(weights: &mut Vec<WeightSpec>, key: impl Into<String>, shape: &[u64]) {
    weights.push(WeightSpec {
        key: key.into(),
        shape: shape.to_vec(),
    });
}

fn linear_parameters(weights: &mut Vec<WeightSpec>, prefix: &str, output: u64, input: u64) {
    parameter(weights, format!("{prefix}.weight"), &[output, input]);
    parameter(weights, format!("{prefix}.bias"), &[output]);
}

fn norm_parameters(weights: &mut Vec<WeightSpec>, prefix: &str, channels: u64) {
    parameter(weights, format!("{prefix}.weight"), &[channels]);
    parameter(weights, format!("{prefix}.bias"), &[channels]);
}

fn conv_parameters(
    weights: &mut Vec<WeightSpec>,
    prefix: &str,
    output: u64,
    input: u64,
    kernel: u64,
) {
    parameter(
        weights,
        format!("{prefix}.weight"),
        &[output, input, kernel, kernel],
    );
    parameter(weights, format!("{prefix}.bias"), &[output]);
}

fn add_unet_specs(weights: &mut Vec<WeightSpec>) {
    conv_parameters(
        weights,
        &format!("{UNET_PREFIX}.input_blocks.0.0"),
        32,
        4,
        3,
    );
    linear_parameters(weights, &format!("{UNET_PREFIX}.time_embed.0"), 128, 32);
    linear_parameters(weights, &format!("{UNET_PREFIX}.time_embed.2"), 128, 128);
    linear_parameters(
        weights,
        &format!("{UNET_PREFIX}.context_projection"),
        32,
        32,
    );
    let channels = [32_u64, 64, 128, 128];
    let mut input = 32;
    for (level, output) in channels.into_iter().enumerate() {
        add_resblock_specs(
            weights,
            &format!("{UNET_PREFIX}.down.{level}.res"),
            input,
            output,
            true,
        );
        if level < 3 {
            conv_parameters(
                weights,
                &format!("{UNET_PREFIX}.down.{level}.downsample"),
                output,
                output,
                3,
            );
        }
        input = output;
    }
    add_resblock_specs(
        weights,
        &format!("{UNET_PREFIX}.mid.block_1"),
        128,
        128,
        true,
    );
    add_attention_specs(weights, &format!("{UNET_PREFIX}.mid.attn_1"), 128, 32);
    add_resblock_specs(
        weights,
        &format!("{UNET_PREFIX}.mid.block_2"),
        128,
        128,
        true,
    );
    for (level, skip) in [128_u64, 128, 64, 32].into_iter().enumerate() {
        let output = [128_u64, 128, 64, 32][level];
        add_resblock_specs(
            weights,
            &format!("{UNET_PREFIX}.up.{level}.res"),
            input + skip,
            output,
            true,
        );
        if level < 3 {
            conv_parameters(
                weights,
                &format!("{UNET_PREFIX}.up.{level}.upsample"),
                output,
                output,
                3,
            );
        }
        input = output;
    }
    norm_parameters(weights, &format!("{UNET_PREFIX}.out.0"), 32);
    conv_parameters(weights, &format!("{UNET_PREFIX}.out.2"), 4, 32, 3);
}

fn add_resblock_specs(
    weights: &mut Vec<WeightSpec>,
    prefix: &str,
    input: u64,
    output: u64,
    time: bool,
) {
    norm_parameters(weights, &format!("{prefix}.norm1"), input);
    conv_parameters(weights, &format!("{prefix}.conv1"), output, input, 3);
    if time {
        linear_parameters(weights, &format!("{prefix}.time_emb_proj"), output, 128);
    }
    norm_parameters(weights, &format!("{prefix}.norm2"), output);
    conv_parameters(weights, &format!("{prefix}.conv2"), output, output, 3);
    if input != output {
        conv_parameters(weights, &format!("{prefix}.nin_shortcut"), output, input, 1);
    }
}

fn add_attention_specs(weights: &mut Vec<WeightSpec>, prefix: &str, channels: u64, context: u64) {
    norm_parameters(weights, &format!("{prefix}.norm"), channels);
    linear_parameters(weights, &format!("{prefix}.q"), channels, channels);
    linear_parameters(weights, &format!("{prefix}.k"), channels, context);
    linear_parameters(weights, &format!("{prefix}.v"), channels, context);
    linear_parameters(weights, &format!("{prefix}.proj_out"), channels, channels);
}

pub fn load_sd15_tokenizer(
    vocabulary_json: &str,
    merges: &str,
) -> Result<Sd1Tokenizer, NativeDiffusionModelError> {
    let descriptor = ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")
        .map_err(|error| NativeDiffusionModelError::Tokenizer(error.to_string()))?;
    Sd1Tokenizer::from_json_and_merges(descriptor, vocabulary_json, merges)
        .map_err(map_sd15_tokenizer_error)
}

pub fn encode_sd15_prompt(
    tokenizer: &Sd1Tokenizer,
    text: &str,
    cancellation: &CancellationToken,
) -> Result<[u32; SD15_TOKEN_COUNT], NativeDiffusionModelError> {
    tokenizer
        .encode_fixed_token_ids(text, cancellation)
        .map_err(map_sd15_tokenizer_error)
}

#[derive(Clone, Copy)]
enum LoadFailureClass {
    Cancelled,
    ResourceExhausted,
}

fn tensor_load_failure(error: &sim_tensor::TensorError) -> Option<LoadFailureClass> {
    match error {
        sim_tensor::TensorError::Cancelled => Some(LoadFailureClass::Cancelled),
        sim_tensor::TensorError::AllocationFailed { .. }
        | sim_tensor::TensorError::ResourceLimitExceeded { .. }
        | sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. } => {
            Some(LoadFailureClass::ResourceExhausted)
        }
        _ => None,
    }
}

fn functional_load_failure(error: &FunctionalError) -> Option<LoadFailureClass> {
    match error {
        FunctionalError::Cancelled => Some(LoadFailureClass::Cancelled),
        FunctionalError::AllocationFailed { .. } => Some(LoadFailureClass::ResourceExhausted),
        FunctionalError::Tensor(error) => tensor_load_failure(error),
        _ => None,
    }
}

fn operator_load_failure(error: &OperatorIndirectionError) -> Option<LoadFailureClass> {
    match error {
        OperatorIndirectionError::Cancelled => Some(LoadFailureClass::Cancelled),
        OperatorIndirectionError::Tensor(error) => tensor_load_failure(error),
        _ => None,
    }
}

fn quantization_load_failure(error: &QuantizationError) -> Option<LoadFailureClass> {
    match error {
        QuantizationError::Cancelled => Some(LoadFailureClass::Cancelled),
        QuantizationError::AllocationFailed { .. }
        | QuantizationError::MaterializationCapacity { .. } => {
            Some(LoadFailureClass::ResourceExhausted)
        }
        _ => None,
    }
}

fn module_part_two_load_failure(
    error: &NeuralNetworkModulePartTwoError,
) -> Option<LoadFailureClass> {
    match error {
        NeuralNetworkModulePartTwoError::Cancelled => Some(LoadFailureClass::Cancelled),
        NeuralNetworkModulePartTwoError::Tensor(error) => tensor_load_failure(error),
        NeuralNetworkModulePartTwoError::Functional(error) => functional_load_failure(error),
        NeuralNetworkModulePartTwoError::Operator(error) => operator_load_failure(error),
        _ => None,
    }
}

fn native_ops_load_failure(error: &NativeOpsError) -> Option<LoadFailureClass> {
    match error {
        NativeOpsError::Cancelled => Some(LoadFailureClass::Cancelled),
        NativeOpsError::Tensor(error) => operator_load_failure(error),
        NativeOpsError::Functional(error) => functional_load_failure(error),
        NativeOpsError::Quantization(error) => quantization_load_failure(error),
        NativeOpsError::Workspace(error) => tensor_load_failure(error),
        NativeOpsError::ModulePartTwo(error) => module_part_two_load_failure(error),
        _ => None,
    }
}

fn attention_load_failure(error: &AttentionError) -> Option<LoadFailureClass> {
    match error {
        AttentionError::Cancelled => Some(LoadFailureClass::Cancelled),
        AttentionError::Tensor(error) => tensor_load_failure(error),
        AttentionError::AllocationFailed { .. } | AttentionError::WorkspaceTooSmall { .. } => {
            Some(LoadFailureClass::ResourceExhausted)
        }
        _ => None,
    }
}

fn artifact_index_load_failure(error: &ArtifactIndexError) -> Option<LoadFailureClass> {
    match error {
        ArtifactIndexError::Cancelled => Some(LoadFailureClass::Cancelled),
        ArtifactIndexError::AllocationFailed(_) => Some(LoadFailureClass::ResourceExhausted),
        _ => None,
    }
}

fn model_store_load_failure(error: &ModelStoreError) -> Option<LoadFailureClass> {
    match error {
        ModelStoreError::Cancelled => Some(LoadFailureClass::Cancelled),
        ModelStoreError::AllocationFailed { .. } => Some(LoadFailureClass::ResourceExhausted),
        ModelStoreError::Index(error) => artifact_index_load_failure(error),
        _ => None,
    }
}

fn native_tensor_load_failure(error: &NativeDiffusionTensorError) -> Option<LoadFailureClass> {
    match error {
        NativeDiffusionTensorError::Tensor(error) => tensor_load_failure(error),
        NativeDiffusionTensorError::Functional(error) => functional_load_failure(error),
        NativeDiffusionTensorError::Operator(error) => operator_load_failure(error),
        _ => None,
    }
}

fn shape_layout_load_failure(error: &ShapeLayoutTransformPartTwoError) -> Option<LoadFailureClass> {
    match error {
        ShapeLayoutTransformPartTwoError::Cancelled => Some(LoadFailureClass::Cancelled),
        ShapeLayoutTransformPartTwoError::Tensor(error) => tensor_load_failure(error),
        _ => None,
    }
}

fn clip_text_load_failure(error: &ClipTextError) -> Option<LoadFailureClass> {
    match error {
        ClipTextError::Tensor(error) => tensor_load_failure(error),
        ClipTextError::Module(error) => native_ops_load_failure(error),
        ClipTextError::Attention(error) => attention_load_failure(error),
        ClipTextError::NativeDiffusion(error) => native_tensor_load_failure(error),
        ClipTextError::ShapeLayout(error) => shape_layout_load_failure(error),
        ClipTextError::Allocation(_) => Some(LoadFailureClass::ResourceExhausted),
        _ => None,
    }
}

fn clip_load_failure(error: &ClipError) -> Option<LoadFailureClass> {
    match error {
        ClipError::Allocation(_) => Some(LoadFailureClass::ResourceExhausted),
        ClipError::Tensor(error) => tensor_load_failure(error),
        ClipError::TensorOperation(error) => native_tensor_load_failure(error),
        ClipError::Attention(error) => attention_load_failure(error),
        ClipError::ModelStore(error) => model_store_load_failure(error),
        ClipError::NativeModule(error) => native_ops_load_failure(error),
        ClipError::TextTransformer(error) => clip_text_load_failure(error),
        _ => None,
    }
}

fn vae_architecture_load_failure(error: &VaeArchitectureError) -> Option<LoadFailureClass> {
    match error {
        VaeArchitectureError::Cancelled(_) => Some(LoadFailureClass::Cancelled),
        VaeArchitectureError::ModelStore(error) => model_store_load_failure(error),
        _ => None,
    }
}

fn vae_load_failure(error: &VaeError) -> Option<LoadFailureClass> {
    match error {
        VaeError::Allocation(_) => Some(LoadFailureClass::ResourceExhausted),
        VaeError::Tensor(error) => tensor_load_failure(error),
        VaeError::NativeTensor(error) => native_tensor_load_failure(error),
        VaeError::Attention(error) => attention_load_failure(error),
        VaeError::ModelStore(error) => model_store_load_failure(error),
        VaeError::NativeOps(error) => native_ops_load_failure(error),
        VaeError::Architecture(error) => vae_architecture_load_failure(error),
        _ => None,
    }
}

fn vision_load_failure(error: &NativeVisionModelError) -> Option<LoadFailureClass> {
    match error {
        NativeVisionModelError::Cancelled => Some(LoadFailureClass::Cancelled),
        NativeVisionModelError::Module(error) => native_ops_load_failure(error),
        NativeVisionModelError::Tensor(error) => operator_load_failure(error),
        NativeVisionModelError::TensorStorage(error) => tensor_load_failure(error),
        NativeVisionModelError::ModelStore(error) => model_store_load_failure(error),
        NativeVisionModelError::Functional(error) => functional_load_failure(error),
        _ => None,
    }
}

fn image_vae_load_failure(error: &ImageVaeError) -> Option<LoadFailureClass> {
    match error {
        ImageVaeError::Vae(error) => vae_load_failure(error),
        ImageVaeError::NativeModule(error) => native_ops_load_failure(error),
        ImageVaeError::VisionState(error) => vision_load_failure(error),
        _ => None,
    }
}

fn patch_load_failure(error: &PatchGraphError) -> Option<LoadFailureClass> {
    match error {
        PatchGraphError::Cancelled(_) => Some(LoadFailureClass::Cancelled),
        PatchGraphError::Tensor(error) => tensor_load_failure(error),
        PatchGraphError::TensorOperation(error) => operator_load_failure(error),
        _ => None,
    }
}

fn classified_load_error(message: String, class: LoadFailureClass) -> NativeDiffusionModelError {
    match class {
        LoadFailureClass::Cancelled => NativeDiffusionModelError::Cancelled,
        LoadFailureClass::ResourceExhausted => {
            NativeDiffusionModelError::ResourceExhausted(message)
        }
    }
}

fn map_sd15_tokenizer_error(error: ClipError) -> NativeDiffusionModelError {
    match clip_load_failure(&error) {
        Some(class) => classified_load_error(error.to_string(), class),
        None => NativeDiffusionModelError::Tokenizer(error.to_string()),
    }
}

fn map_sd15_clip_error(error: ClipError) -> NativeDiffusionModelError {
    if let Some(class) = clip_load_failure(&error) {
        return classified_load_error(error.to_string(), class);
    }
    match error {
        ClipError::Tensor(error) => NativeDiffusionModelError::TensorBackend(error),
        ClipError::TensorOperation(error) => NativeDiffusionModelError::Tensor(error),
        ClipError::Attention(error) => NativeDiffusionModelError::Attention(error),
        ClipError::ModelStore(error) => NativeDiffusionModelError::Store(error),
        error => NativeDiffusionModelError::Clip(error.to_string()),
    }
}

#[derive(Debug)]
pub struct Sd15TinyModel {
    backend: Arc<CpuBackend>,
    weights: BTreeMap<String, Tensor>,
    patch_identity: PatchGraphIdentity,
    patch_execution_digest: String,
}

impl Sd15TinyModel {
    pub fn load_production(
        store: &ModelStore,
        index: &ArtifactIndex,
        loaded: &LoadedModel,
        projection: &Sd15DetectorProjection,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionModelError> {
        check_context(context)?;
        projection.detect()?;
        let patch_graph = PatchGraph::checked_semantic(loaded.identity(), Vec::new())
            .map_err(|error| NativeDiffusionModelError::Patch(error.to_string()))?;
        Self::load_production_with_patch_graph(
            store,
            index,
            loaded,
            projection,
            &patch_graph,
            backend,
            context,
        )
    }

    pub fn load_production_with_patch_graph(
        store: &ModelStore,
        index: &ArtifactIndex,
        loaded: &LoadedModel,
        projection: &Sd15DetectorProjection,
        patch_graph: &PatchGraph,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionModelError> {
        check_context(context)?;
        projection.detect()?;
        Self::load_weights(store, index, loaded, patch_graph, backend, context)
    }

    #[cfg(feature = "test-support")]
    pub fn load_reduced_fixture(
        store: &ModelStore,
        index: &ArtifactIndex,
        loaded: &LoadedModel,
        admission: &ReducedFixtureAdmission,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionModelError> {
        check_context(context)?;
        if admission.fixture_id != SD15_TINY_FIXTURE_ID
            || admission.detector_transcript_sha256.len() != 64
        {
            return Err(NativeDiffusionModelError::InvalidFixtureAdmission);
        }
        let patch_graph = PatchGraph::checked_semantic(loaded.identity(), Vec::new())
            .map_err(|error| NativeDiffusionModelError::Patch(error.to_string()))?;
        Self::load_reduced_fixture_with_patch_graph(
            store,
            index,
            loaded,
            admission,
            &patch_graph,
            backend,
            context,
        )
    }

    #[cfg(feature = "test-support")]
    pub fn load_reduced_fixture_with_patch_graph(
        store: &ModelStore,
        index: &ArtifactIndex,
        loaded: &LoadedModel,
        admission: &ReducedFixtureAdmission,
        patch_graph: &PatchGraph,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionModelError> {
        check_context(context)?;
        if admission.fixture_id != SD15_TINY_FIXTURE_ID
            || admission.detector_transcript_sha256.len() != 64
        {
            return Err(NativeDiffusionModelError::InvalidFixtureAdmission);
        }
        Self::load_weights(store, index, loaded, patch_graph, backend, context)
    }

    fn load_weights(
        store: &ModelStore,
        index: &ArtifactIndex,
        loaded: &LoadedModel,
        patch_graph: &PatchGraph,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionModelError> {
        let mut manifest = Vec::new();
        add_unet_specs(&mut manifest);
        let expected = manifest
            .iter()
            .map(|spec| spec.key.as_str())
            .collect::<BTreeSet<_>>();
        let actual = loaded
            .tensors()
            .keys()
            .filter(|key| key.starts_with(UNET_PREFIX))
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(NativeDiffusionModelError::WeightKeys {
                missing: expected
                    .difference(&actual)
                    .map(|value| (*value).to_owned())
                    .collect(),
                unexpected: actual
                    .difference(&expected)
                    .map(|value| (*value).to_owned())
                    .collect(),
            });
        }
        let mut tensor_bytes = store.read_tensors(
            index,
            loaded,
            manifest.iter().map(|spec| spec.key.as_str()),
            context.cancellation,
        )?;
        let mut weights = BTreeMap::new();
        for spec in manifest {
            check_context(context)?;
            let metadata = loaded
                .tensors()
                .get(&spec.key)
                .ok_or_else(|| NativeDiffusionModelError::MissingWeight(spec.key.clone()))?;
            if metadata.data_type != "F32" || metadata.shape != spec.shape {
                return Err(NativeDiffusionModelError::WeightShape {
                    key: spec.key,
                    expected: spec.shape,
                    actual: metadata.shape.clone(),
                    data_type: metadata.data_type.clone(),
                });
            }
            let bytes = tensor_bytes
                .remove(&metadata.name)
                .ok_or_else(|| NativeDiffusionModelError::MissingWeight(metadata.name.clone()))?;
            if !bytes.len().is_multiple_of(4) {
                return Err(NativeDiffusionModelError::WeightBytes(
                    metadata.name.clone(),
                ));
            }
            let mut values = backend.workspace_vec(context, bytes.len() / 4)?;
            for (index, chunk) in bytes.chunks_exact(4).enumerate() {
                if index.is_multiple_of(64) {
                    check_context(context)?;
                }
                let encoded: [u8; 4] = chunk
                    .try_into()
                    .map_err(|_| NativeDiffusionModelError::WeightBytes(metadata.name.clone()))?;
                values.try_push(f32::from_le_bytes(encoded))?;
            }
            let tensor = tensor_from_f32(&backend, &metadata.shape, &values, context)?;
            weights.insert(metadata.name.clone(), tensor);
        }
        let mapped =
            MappedModelWeights::from_parts(loaded.identity().to_owned(), weights, Vec::new());
        let patched = patch_graph
            .apply(backend.as_ref(), &mapped, context)
            .map_err(|error| match patch_load_failure(&error) {
                Some(class) => classified_load_error(error.to_string(), class),
                None => NativeDiffusionModelError::Patch(error.to_string()),
            })?;
        Ok(Self {
            backend,
            weights: patched.tensors().clone(),
            patch_identity: patch_graph.identity(),
            patch_execution_digest: patched.cache_identity().to_owned(),
        })
    }

    pub fn patch_identity(&self) -> &PatchGraphIdentity {
        &self.patch_identity
    }

    pub fn patch_execution_digest(&self) -> &str {
        &self.patch_execution_digest
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, NativeDiffusionModelError> {
        let mut storages = BTreeSet::new();
        self.weights.values().try_fold(0_u64, |total, tensor| {
            if !storages.insert(tensor.storage_id().get()) {
                return Ok(total);
            }
            total
                .checked_add(tensor.storage_byte_len())
                .ok_or(NativeDiffusionModelError::Overflow(
                    "resident model storage bytes",
                ))
        })
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeDiffusionModelError> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| NativeDiffusionModelError::Overflow("resident model object bytes"))?;
        let entry_bytes = self
            .weights
            .len()
            .checked_mul(std::mem::size_of::<(String, Tensor)>())
            .ok_or(NativeDiffusionModelError::Overflow(
                "resident model entry bytes",
            ))?;
        bytes =
            bytes
                .checked_add(u64::try_from(entry_bytes).map_err(|_| {
                    NativeDiffusionModelError::Overflow("resident model entry bytes")
                })?)
                .ok_or(NativeDiffusionModelError::Overflow(
                    "resident model entry bytes",
                ))?;
        for name in self.weights.keys() {
            bytes =
                bytes
                    .checked_add(u64::try_from(name.capacity()).map_err(|_| {
                        NativeDiffusionModelError::Overflow("resident model key bytes")
                    })?)
                    .ok_or(NativeDiffusionModelError::Overflow(
                        "resident model key bytes",
                    ))?;
        }
        bytes = bytes
            .checked_add(self.patch_identity.owned_resident_bytes().ok_or(
                NativeDiffusionModelError::Overflow("resident model identity bytes"),
            )?)
            .and_then(|bytes| {
                bytes.checked_add(u64::try_from(self.patch_execution_digest.capacity()).ok()?)
            })
            .ok_or(NativeDiffusionModelError::Overflow(
                "resident model identity bytes",
            ))?;
        bytes.checked_add(self.resident_storage_bytes()?).ok_or(
            NativeDiffusionModelError::Overflow("resident model total bytes"),
        )
    }

    pub fn backend(&self) -> &CpuBackend {
        &self.backend
    }

    fn weight(&self, key: &str) -> Result<&Tensor, NativeDiffusionModelError> {
        self.weights
            .get(key)
            .ok_or_else(|| NativeDiffusionModelError::MissingWeight(key.to_owned()))
    }

    pub fn denoise_at_model_time(
        &self,
        latent: &Tensor,
        model_time: f32,
        conditioning: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeDiffusionModelError> {
        self.denoise_at_model_time_with_control(latent, model_time, conditioning, None, context)
    }

    pub fn denoise_at_model_time_with_control(
        &self,
        latent: &Tensor,
        model_time: f32,
        conditioning: &Tensor,
        control: Option<&ControlResult>,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeDiffusionModelError> {
        check_context(context)?;
        require_shape(latent, &[1, 4, 4, 4], "SD15 latent")?;
        require_shape(conditioning, &[1, 77, 32], "SD15 conditioning")?;
        if !model_time.is_finite() || model_time < 0.0 {
            return Err(NativeDiffusionModelError::InvalidModelTime);
        }
        let time_values = timestep_embedding(&self.backend, context, model_time, 32)?;
        let time = tensor_from_f32(&self.backend, &[1, 32], &time_values, context)?;
        drop(time_values);
        let time = self.linear(&time, &format!("{UNET_PREFIX}.time_embed.0"), context)?;
        let time = silu(&self.backend, &time, context)?;
        let time = self.linear(&time, &format!("{UNET_PREFIX}.time_embed.2"), context)?;
        let context_tensor = self.linear(
            conditioning,
            &format!("{UNET_PREFIX}.context_projection"),
            context,
        )?;
        let mut hidden = self.conv(
            latent,
            &format!("{UNET_PREFIX}.input_blocks.0.0"),
            1,
            1,
            context,
        )?;
        let mut input_control = control.map(|result| result.input().iter().rev());
        let mut middle_control = control.map(|result| result.middle().iter().rev());
        let mut output_control = control.map(|result| result.output().iter().rev());
        let mut skips = Vec::new();
        for level in 0..4 {
            hidden = self.resblock(
                &hidden,
                Some(&time),
                &format!("{UNET_PREFIX}.down.{level}.res"),
                context,
            )?;
            hidden = add_next_control(&self.backend, hidden, input_control.as_mut(), context)?;
            skips.push(hidden.clone());
            if level < 3 {
                hidden = self.conv(
                    &hidden,
                    &format!("{UNET_PREFIX}.down.{level}.downsample"),
                    2,
                    1,
                    context,
                )?;
            }
        }
        hidden = self.resblock(
            &hidden,
            Some(&time),
            &format!("{UNET_PREFIX}.mid.block_1"),
            context,
        )?;
        hidden = self.spatial_attention(
            &hidden,
            &context_tensor,
            &format!("{UNET_PREFIX}.mid.attn_1"),
            context,
        )?;
        hidden = self.resblock(
            &hidden,
            Some(&time),
            &format!("{UNET_PREFIX}.mid.block_2"),
            context,
        )?;
        hidden = add_next_control(&self.backend, hidden, middle_control.as_mut(), context)?;
        for level in 0..4 {
            let skip = skips
                .pop()
                .ok_or(NativeDiffusionModelError::Overflow("UNet skip"))?;
            let skip = add_next_control(&self.backend, skip, output_control.as_mut(), context)?;
            hidden = resize_to_match(&self.backend, hidden, &skip, context)?;
            hidden = concat_channels(&self.backend, &hidden, &skip, context)?;
            hidden = self.resblock(
                &hidden,
                Some(&time),
                &format!("{UNET_PREFIX}.up.{level}.res"),
                context,
            )?;
            if level < 3 {
                let next_skip = skips
                    .last()
                    .ok_or(NativeDiffusionModelError::Overflow("UNet next skip"))?;
                if hidden.descriptor().shape()[2] < next_skip.descriptor().shape()[2] {
                    hidden = nearest_upsample_2x(&self.backend, &hidden, context)?;
                }
                hidden = self.conv(
                    &hidden,
                    &format!("{UNET_PREFIX}.up.{level}.upsample"),
                    1,
                    1,
                    context,
                )?;
            }
        }
        reject_excess_control(input_control, "input")?;
        reject_excess_control(middle_control, "middle")?;
        reject_excess_control(output_control, "output")?;
        hidden = self.group_norm(&hidden, &format!("{UNET_PREFIX}.out.0"), context)?;
        hidden = silu(&self.backend, &hidden, context)?;
        self.conv(&hidden, &format!("{UNET_PREFIX}.out.2"), 1, 1, context)
    }

    fn linear(
        &self,
        input: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeDiffusionModelError> {
        Ok(linear(
            &self.backend,
            input,
            self.weight(&format!("{prefix}.weight"))?,
            Some(self.weight(&format!("{prefix}.bias"))?),
            context,
        )?)
    }

    fn group_norm(
        &self,
        input: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeDiffusionModelError> {
        let channels = usize::try_from(input.descriptor().shape()[1])
            .map_err(|_| NativeDiffusionModelError::Overflow("group norm channels"))?;
        let groups = 32.min(channels);
        Ok(group_norm(
            &self.backend,
            input,
            self.weight(&format!("{prefix}.weight"))?,
            self.weight(&format!("{prefix}.bias"))?,
            groups,
            1e-6,
            context,
        )?)
    }

    fn conv(
        &self,
        input: &Tensor,
        prefix: &str,
        stride: usize,
        padding: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeDiffusionModelError> {
        Ok(conv2d(
            &self.backend,
            input,
            self.weight(&format!("{prefix}.weight"))?,
            Some(self.weight(&format!("{prefix}.bias"))?),
            stride,
            padding,
            context,
        )?)
    }

    fn resblock(
        &self,
        input: &Tensor,
        time: Option<&Tensor>,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeDiffusionModelError> {
        let mut hidden = self.group_norm(input, &format!("{prefix}.norm1"), context)?;
        hidden = silu(&self.backend, &hidden, context)?;
        hidden = self.conv(&hidden, &format!("{prefix}.conv1"), 1, 1, context)?;
        if let Some(time) = time {
            let projected = self.linear(time, &format!("{prefix}.time_emb_proj"), context)?;
            let projected_values = tensor_to_f32(&self.backend, &projected, context)?;
            let projected = tensor_from_f32(
                &self.backend,
                &[projected.descriptor().shape()[1]],
                &projected_values,
                context,
            )?;
            hidden = add_channel_bias(&self.backend, &hidden, &projected, context)?;
        }
        hidden = self.group_norm(&hidden, &format!("{prefix}.norm2"), context)?;
        hidden = silu(&self.backend, &hidden, context)?;
        hidden = self.conv(&hidden, &format!("{prefix}.conv2"), 1, 1, context)?;
        let input_channels = input.descriptor().shape()[1];
        let output_channels = hidden.descriptor().shape()[1];
        let residual = if input_channels == output_channels {
            input.clone()
        } else {
            self.conv(input, &format!("{prefix}.nin_shortcut"), 1, 0, context)?
        };
        Ok(add(&self.backend, &residual, &hidden, context)?)
    }

    fn spatial_attention(
        &self,
        input: &Tensor,
        conditioning: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeDiffusionModelError> {
        let normalized = self.group_norm(input, &format!("{prefix}.norm"), context)?;
        let query_tokens = nchw_to_tokens(&self.backend, &normalized, context)?;
        let query = self.linear(&query_tokens, &format!("{prefix}.q"), context)?;
        let key = self.linear(conditioning, &format!("{prefix}.k"), context)?;
        let value = self.linear(conditioning, &format!("{prefix}.v"), context)?;
        let channels = usize::try_from(input.descriptor().shape()[1])
            .map_err(|_| NativeDiffusionModelError::Overflow("attention channels"))?;
        let query_count = usize::try_from(query.descriptor().shape()[1])
            .map_err(|_| NativeDiffusionModelError::Overflow("attention queries"))?;
        let key_count = usize::try_from(key.descriptor().shape()[1])
            .map_err(|_| NativeDiffusionModelError::Overflow("attention keys"))?;
        let heads = 4;
        let head_dimension =
            channels
                .checked_div(heads)
                .ok_or(NativeDiffusionModelError::Overflow(
                    "attention head dimension",
                ))?;
        let query_values = tensor_to_f32(&self.backend, &query, context)?;
        let key_values = tensor_to_f32(&self.backend, &key, context)?;
        let value_values = tensor_to_f32(&self.backend, &value, context)?;
        let outcome = scaled_dot_product_attention_with_context(
            &self.backend,
            AttentionRequest {
                backend: AttentionBackend::PytorchSdp,
                fallback: AttentionFallbackPolicy::AllowExactNative,
                batch: 1,
                query_tokens: query_count,
                key_tokens: key_count,
                heads,
                head_dimension,
                value_dimension: head_dimension,
                scale: None,
                workspace_limit_bytes: key_count * std::mem::size_of::<f32>(),
            },
            &query_values,
            &key_values,
            &value_values,
            None,
            context,
        )?;
        drop(value_values);
        drop(key_values);
        drop(query_values);
        let attention = tensor_from_f32(
            &self.backend,
            &[1, query_count as u64, channels as u64],
            &outcome.values,
            context,
        )?;
        let attention = self.linear(&attention, &format!("{prefix}.proj_out"), context)?;
        let attention = tokens_to_nchw(
            &self.backend,
            &attention,
            input.descriptor().shape()[2],
            input.descriptor().shape()[3],
            context,
        )?;
        Ok(add(&self.backend, input, &attention, context)?)
    }
}

fn require_shape(
    tensor: &Tensor,
    expected: &[u64],
    name: &'static str,
) -> Result<(), NativeDiffusionModelError> {
    if tensor.descriptor().shape() != expected {
        return Err(NativeDiffusionModelError::InputShape {
            name,
            expected: expected.to_vec(),
            actual: tensor.descriptor().shape().to_vec(),
        });
    }
    Ok(())
}

fn check_context(context: &ExecutionContext<'_>) -> Result<(), NativeDiffusionModelError> {
    context.check().map_err(|error| {
        if error == sim_tensor::TensorError::Cancelled {
            NativeDiffusionModelError::Cancelled
        } else {
            NativeDiffusionModelError::TensorBackend(error)
        }
    })
}

fn nchw_to_tokens(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionModelError> {
    let shape = tensor.descriptor().shape();
    if shape.len() != 4 || shape[0] != 1 {
        return Err(NativeDiffusionModelError::InputShape {
            name: "attention NCHW",
            expected: vec![1, 0, 0, 0],
            actual: shape.to_vec(),
        });
    }
    let channels = usize::try_from(shape[1])
        .map_err(|_| NativeDiffusionModelError::Overflow("attention channels"))?;
    let height = usize::try_from(shape[2])
        .map_err(|_| NativeDiffusionModelError::Overflow("attention height"))?;
    let width = usize::try_from(shape[3])
        .map_err(|_| NativeDiffusionModelError::Overflow("attention width"))?;
    let source = tensor_to_f32(backend, tensor, context)?;
    let mut values = backend.workspace_vec(context, source.len())?;
    for _ in 0..source.len() {
        values.try_push(0.0)?;
    }
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                values[(y * width + x) * channels + channel] =
                    source[(channel * height + y) * width + x];
            }
        }
    }
    Ok(tensor_from_f32(
        backend,
        &[1, (height * width) as u64, channels as u64],
        &values,
        context,
    )?)
}

fn tokens_to_nchw(
    backend: &CpuBackend,
    tensor: &Tensor,
    height: u64,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionModelError> {
    let shape = tensor.descriptor().shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != height * width {
        return Err(NativeDiffusionModelError::InputShape {
            name: "attention tokens",
            expected: vec![1, height * width, shape.last().copied().unwrap_or(0)],
            actual: shape.to_vec(),
        });
    }
    let channels = usize::try_from(shape[2])
        .map_err(|_| NativeDiffusionModelError::Overflow("attention channels"))?;
    let height_usize = usize::try_from(height)
        .map_err(|_| NativeDiffusionModelError::Overflow("attention height"))?;
    let width_usize = usize::try_from(width)
        .map_err(|_| NativeDiffusionModelError::Overflow("attention width"))?;
    let source = tensor_to_f32(backend, tensor, context)?;
    let mut values = backend.workspace_vec(context, source.len())?;
    for _ in 0..source.len() {
        values.try_push(0.0)?;
    }
    for y in 0..height_usize {
        for x in 0..width_usize {
            for channel in 0..channels {
                values[(channel * height_usize + y) * width_usize + x] =
                    source[(y * width_usize + x) * channels + channel];
            }
        }
    }
    Ok(tensor_from_f32(
        backend,
        &[1, shape[2], height, width],
        &values,
        context,
    )?)
}

fn resize_to_match(
    backend: &CpuBackend,
    mut input: Tensor,
    target: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionModelError> {
    while input.descriptor().shape()[2] < target.descriptor().shape()[2] {
        input = nearest_upsample_2x(backend, &input, context)?;
    }
    if input.descriptor().shape()[2..] != target.descriptor().shape()[2..] {
        return Err(NativeDiffusionModelError::InputShape {
            name: "UNet skip",
            expected: target.descriptor().shape()[2..].to_vec(),
            actual: input.descriptor().shape()[2..].to_vec(),
        });
    }
    Ok(input)
}

fn add_next_control<'a>(
    backend: &CpuBackend,
    hidden: Tensor,
    controls: Option<&mut std::iter::Rev<std::slice::Iter<'a, Option<Tensor>>>>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionModelError> {
    match controls.and_then(Iterator::next) {
        Some(Some(control)) => add(backend, &hidden, control, context).map_err(Into::into),
        Some(None) | None => Ok(hidden),
    }
}

fn reject_excess_control(
    controls: Option<std::iter::Rev<std::slice::Iter<'_, Option<Tensor>>>>,
    section: &'static str,
) -> Result<(), NativeDiffusionModelError> {
    if controls.is_some_and(|mut controls| controls.next().is_some()) {
        return Err(NativeDiffusionModelError::ExcessControl(section));
    }
    Ok(())
}

fn timestep_embedding(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    timestep: f32,
    width: usize,
) -> Result<CpuWorkspaceVec<f32>, NativeDiffusionModelError> {
    let half = width / 2;
    let mut result = backend.workspace_vec(context, width)?;
    for index in 0..half {
        let frequency = (-10_000_f32.ln() * index as f32 / half as f32).exp();
        result.try_push((timestep * frequency).cos())?;
    }
    for index in 0..half {
        let frequency = (-10_000_f32.ln() * index as f32 / half as f32).exp();
        result.try_push((timestep * frequency).sin())?;
    }
    Ok(result)
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NativeDiffusionModelError {
    #[error("source-shape projection is not a supported SD15 model")]
    UnsupportedFamily,
    #[error("reduced fixture admission is invalid")]
    InvalidFixtureAdmission,
    #[error("native diffusion weight manifest contains duplicate keys")]
    DuplicateWeight,
    #[error("native diffusion model is missing weight {0}")]
    MissingWeight(String),
    #[error(
        "native diffusion model key set differs; missing={missing:?}, unexpected={unexpected:?}"
    )]
    WeightKeys {
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    #[error("native diffusion weight {key} expected F32 {expected:?}, got {data_type} {actual:?}")]
    WeightShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        data_type: String,
    },
    #[error("native diffusion weight {0} has invalid f32 bytes")]
    WeightBytes(String),
    #[error("native diffusion tokenizer error: {0}")]
    Tokenizer(String),
    #[error("canonical native CLIP adapter rejected the request: {0}")]
    Clip(String),
    #[error("canonical native VAE adapter rejected the request: {0}")]
    Vae(String),
    #[error("canonical patch graph rejected the native diffusion model: {0}")]
    Patch(String),
    #[error("native diffusion control result contains excess {0} slots")]
    ExcessControl(&'static str),
    #[error("native diffusion {name} expected shape {expected:?}, got {actual:?}")]
    InputShape {
        name: &'static str,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("native diffusion model time must be finite and non-negative")]
    InvalidModelTime,
    #[error("native diffusion latent adapter rejected the canonical row: {0}")]
    LatentAdapter(String),
    #[error(
        "SD15 latent dimensions require nonzero batch and width/height divisible by eight, got batch={batch}, width={width}, height={height}"
    )]
    InvalidLatentDimensions { batch: u64, width: u64, height: u64 },
    #[error("native diffusion shape or allocation overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("native diffusion model execution was cancelled")]
    Cancelled,
    #[error("native diffusion resource exhausted: {0}")]
    ResourceExhausted(String),
    #[error(transparent)]
    Store(#[from] ModelStoreError),
    #[error(transparent)]
    Tensor(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    TensorBackend(#[from] sim_tensor::TensorError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CancellationToken, StreamId, TensorError};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    fn workspace() -> Result<PathBuf, Box<dyn std::error::Error>> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?
            .to_path_buf())
    }

    fn target_directory(workspace_root: &Path) -> PathBuf {
        match std::env::var_os("CARGO_TARGET_DIR") {
            Some(directory) => {
                let directory = PathBuf::from(directory);
                if directory.is_absolute() {
                    directory
                } else {
                    workspace_root.join(directory)
                }
            }
            None => workspace_root.join("target"),
        }
    }

    fn valid_projection() -> Sd15DetectorProjection {
        let source_shapes = BTreeMap::from([
            (
                "model.diffusion_model.input_blocks.0.0.weight".to_owned(),
                vec![320, 4, 3, 3],
            ),
            (
                "model.diffusion_model.time_embed.0.weight".to_owned(),
                vec![1280, 320],
            ),
            (
                "model.diffusion_model.out.2.weight".to_owned(),
                vec![4, 320, 3, 3],
            ),
            (
                "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
                    .to_owned(),
                vec![49_408, 768],
            ),
            (
                "first_stage_model.quant_conv.weight".to_owned(),
                vec![8, 8, 1, 1],
            ),
        ]);
        Sd15DetectorProjection {
            feature_id: SD15_FEATURE_ID.to_owned(),
            model_channels: 320,
            context_dim: 768,
            adm_in_channels: None,
            use_linear_in_transformer: false,
            use_temporal_attention: false,
            source_shapes,
        }
    }

    fn digest(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
    }

    fn write_artifact(
        filename: &str,
        validation: &str,
        scope: &str,
        fixture_digests: serde_json::Value,
        cases: serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = workspace()?;
        let directory = target_directory(&root).join("sim-parity");
        fs::create_dir_all(&directory)?;
        let passed = cases
            .as_object()
            .ok_or("validation cases must be an object")?
            .len();
        let value = json!({
            "cases": cases,
            "environment": {"architecture": std::env::consts::ARCH, "backend": "native-rust-cpu", "operating_system": std::env::consts::OS},
            "fixture_digests": fixture_digests,
            "remaining_release_gates": ["comfy-parity-native-diffusion-e2e", "comfy-parity-native-compute-breadth-integration", "comfy-parity-final-validation"],
            "scope": scope,
            "skipped": [],
            "summary": {"failed": 0, "passed": passed, "skipped": 0},
            "validation": validation,
            "validation_id": validation,
        });
        let mut bytes = serde_json::to_vec_pretty(&value)?;
        bytes.push(b'\n');
        fs::write(directory.join(filename), bytes)?;
        Ok(())
    }

    #[test]
    fn detector_requires_source_sd15_shapes() -> Result<(), NativeDiffusionModelError> {
        let mut projection = valid_projection();
        assert_eq!(projection.detect()?, SD15_FEATURE_ID);
        projection.model_channels = 32;
        assert_eq!(
            projection.detect(),
            Err(NativeDiffusionModelError::UnsupportedFamily)
        );
        Ok(())
    }

    #[test]
    fn sd15_tokenizer_adapter_preserves_canonical_semantics()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture_root =
            workspace()?.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1");
        let vocabulary = fs::read_to_string(fixture_root.join("vocab.json"))?;
        let merges = fs::read_to_string(fixture_root.join("merges.txt"))?;
        let tokenizer = load_sd15_tokenizer(&vocabulary, &merges)?;
        let cancellation = CancellationToken::default();

        let pinned = encode_sd15_prompt(&tokenizer, "a test", &cancellation)?;
        assert_eq!(&pinned[..4], &[49_406, 320, 1_628, 49_407]);
        let truncated = encode_sd15_prompt(&tokenizer, &"a ".repeat(200), &cancellation)?;
        assert_eq!(truncated.len(), SD15_TOKEN_COUNT);
        assert_eq!(truncated[SD15_TOKEN_COUNT - 1], 49_407);

        let mut duplicate_merges = merges.lines().map(str::to_owned).collect::<Vec<_>>();
        let first_merge = duplicate_merges
            .get(1)
            .ok_or("SD1 merge fixture is missing its first pair")?
            .clone();
        let last_merge = duplicate_merges
            .last_mut()
            .ok_or("SD1 merge fixture is empty")?;
        *last_merge = first_merge;
        let duplicate_merges = duplicate_merges.join("\n");
        assert!(matches!(
            load_sd15_tokenizer(&vocabulary, &duplicate_merges),
            Err(NativeDiffusionModelError::Tokenizer(message))
                if message.contains("duplicate")
        ));
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            encode_sd15_prompt(&tokenizer, "a test", &cancelled),
            Err(NativeDiffusionModelError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn sd15_load_errors_preserve_typed_cancellation_and_capacity_failures() {
        assert!(matches!(
            map_sd15_clip_error(ClipError::TextTransformer(ClipTextError::ShapeLayout(
                ShapeLayoutTransformPartTwoError::Cancelled,
            ))),
            NativeDiffusionModelError::Cancelled
        ));
        assert!(matches!(
            map_sd15_clip_error(ClipError::TextTransformer(ClipTextError::ShapeLayout(
                ShapeLayoutTransformPartTwoError::Tensor(
                    TensorError::WorkspaceAuthorizationExceeded {
                        requested: 64,
                        authorized: 32,
                        in_use: 0,
                    },
                ),
            ))),
            NativeDiffusionModelError::ResourceExhausted(_)
        ));
        assert!(matches!(
            image_vae_load_failure(&ImageVaeError::Vae(VaeError::Tensor(
                TensorError::Cancelled,
            ))),
            Some(LoadFailureClass::Cancelled)
        ));
        assert!(matches!(
            image_vae_load_failure(&ImageVaeError::NativeModule(NativeOpsError::Quantization(
                QuantizationError::Cancelled
            ),)),
            Some(LoadFailureClass::Cancelled)
        ));
        assert!(matches!(
            patch_load_failure(&PatchGraphError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded {
                    requested: 128,
                    authorized: 64,
                    in_use: 0,
                },
            )),
            Some(LoadFailureClass::ResourceExhausted)
        ));
    }

    #[test]
    fn val_model_family_001() -> Result<(), Box<dyn std::error::Error>> {
        let root = workspace()?;
        let fixture_root = root.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1");
        let catalog_path =
            root.join(".agents/specs/comfy-parity/catalogs/native-diffusion-fixture.json");
        let manifest_path = fixture_root.join("state-dict-manifest.json");
        let provenance_path = fixture_root.join("oracle-provenance.json");
        let model_path = fixture_root.join("model.safetensors");
        let projection = valid_projection();
        assert_eq!(projection.detect()?, SD15_FEATURE_ID);
        let mut reduced = projection;
        reduced.model_channels = 32;
        reduced.source_shapes.insert(
            "model.diffusion_model.input_blocks.0.0.weight".to_owned(),
            vec![32, 4, 3, 3],
        );
        assert_eq!(
            reduced.detect(),
            Err(NativeDiffusionModelError::UnsupportedFamily)
        );

        let vocabulary = fs::read_to_string(fixture_root.join("vocab.json"))?;
        let merges = fs::read_to_string(fixture_root.join("merges.txt"))?;
        let tokenizer = load_sd15_tokenizer(&vocabulary, &merges)?;
        let cancellation = CancellationToken::default();
        let positive = encode_sd15_prompt(&tokenizer, "a test", &cancellation)?;
        let negative = encode_sd15_prompt(&tokenizer, "", &cancellation)?;
        assert_eq!(&positive[..4], &[49_406, 320, 1_628, 49_407]);
        assert_eq!(negative[0], 49_406);
        assert!(negative[1..].iter().all(|token| *token == 49_407));

        let manifest: serde_json::Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        let recorded = manifest
            .get("weights")
            .and_then(serde_json::Value::as_array)
            .ok_or("missing weights")?;
        let expected = sd15_tiny_weight_manifest()?;
        assert_eq!(recorded.len(), expected.len());
        for (recorded, expected) in recorded.iter().zip(&expected) {
            assert_eq!(
                recorded.get("key").and_then(serde_json::Value::as_str),
                Some(expected.key.as_str())
            );
            assert_eq!(
                recorded.get("shape"),
                Some(&serde_json::to_value(&expected.shape)?)
            );
        }
        let catalog: serde_json::Value = serde_json::from_slice(&fs::read(&catalog_path)?)?;
        let required_checkpoints = catalog
            .get("required_checkpoints")
            .and_then(serde_json::Value::as_array)
            .ok_or("missing checkpoints")?
            .iter()
            .map(|name| {
                name.as_str()
                    .map(str::to_owned)
                    .ok_or("checkpoint name is not a string")
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let actual_checkpoints = fs::read_dir(&fixture_root)?
            .map(|entry| {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    return Err(std::io::Error::other(format!(
                        "native diffusion fixture contains non-file entry {:?}",
                        entry.path()
                    )));
                }
                entry.file_name().into_string().map_err(|name| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("native diffusion fixture name is not UTF-8: {name:?}"),
                    )
                })
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        assert_eq!(actual_checkpoints, required_checkpoints);

        let provenance: serde_json::Value = serde_json::from_slice(&fs::read(&provenance_path)?)?;
        assert_eq!(
            provenance
                .get("production_dependency")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
        let sources = provenance
            .get("sources")
            .and_then(serde_json::Value::as_object)
            .ok_or("native diffusion provenance has no sources")?;
        for (source, expected_digest) in sources {
            if !source.starts_with("projects/comfy/ComfyUI/")
                || source.split('/').any(|component| component == "..")
            {
                return Err(format!("unsafe native diffusion provenance path {source:?}").into());
            }
            assert_eq!(
                digest(&root.join(source))?,
                expected_digest
                    .as_str()
                    .ok_or("native diffusion source digest is not a string")?,
                "stale native diffusion source provenance for {source}"
            );
        }
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024)?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let authorization = workspace_authority.authorize_workspace(512)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
        assert!(matches!(
            empty_sd15_latent(&backend, 1, 32, 32, &context),
            Err(NativeDiffusionModelError::Cancelled)
        ));
        assert_eq!(authorization.in_use_bytes(), 0);
        let (tiny_backend, tiny_workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(128)?;
        let tiny_cancellation = CancellationToken::default();
        let tiny_context = tiny_backend.execution_context(
            StreamId::DEFAULT,
            tiny_workspace_authority.authorize_workspace(0)?,
            &tiny_cancellation,
        );
        assert!(tensor_from_f32(&tiny_backend, &[1, 4, 4, 4], &[0.0; 64], &tiny_context,).is_err());

        write_artifact(
            "val-model-family-foundation-001.json",
            "VAL-MODEL-FAMILY-FOUNDATION-001",
            "sim-parity-native-diffusion-foundation pinned SD15 detector, tokenizer, topology, checkpoints, cancellation, and OOM slice",
            json!({
                "catalog": digest(&catalog_path)?, "manifest": digest(&manifest_path)?,
                "model": digest(&model_path)?, "provenance": digest(&provenance_path)?,
            }),
            json!({
                "checkpoint_directory_exactly_matches_catalog": true, "complete_topology_manifest": true,
                "detector_rejects_reduced_user_artifact": true, "detector_selects_only_sd15": true,
                "full_sd1_tokenizer_ids": true, "model_allocation_oom_is_typed": true,
                "model_cancellation_is_typed": true, "source_provenance_digests_are_current": true,
                "test_fixture_is_not_a_production_dependency": true,
            }),
        )?;
        Ok(())
    }

    #[test]
    fn sd15_latent_adapter_uses_canonical_format_owner() -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let authorization = workspace_authority.authorize_workspace(4096)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
        assert_eq!(SD15_LATENT_FORMAT.feature_id, "COMFY-MODEL-0045");
        assert_eq!(SD15_LATENT_FORMAT.identifier, "SD15");
        let empty = empty_sd15_latent(&backend, 1, 32, 32, &context)?;
        assert_eq!(empty.descriptor().shape(), &[1, 4, 4, 4]);
        assert!(
            tensor_to_f32(&backend, &empty, &context)?
                .iter()
                .all(|value| *value == 0.0)
        );
        let floored_empty = empty_sd15_latent(&backend, 1, 31, 32, &context)?;
        assert_eq!(floored_empty.descriptor().shape(), &[1, 4, 4, 3]);
        drop(floored_empty);
        let latent = tensor_from_f32(&backend, &[1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let preview_tensor = sd15_latent_preview(&backend, &latent, &context)?;
        let preview = tensor_to_f32(&backend, &preview_tensor, &context)?;
        for channel in 0..3 {
            let expected = (0..4).fold(0.0_f32, |value, latent_channel| {
                ((latent_channel + 1) as f32).mul_add(
                    SD15_LATENT_FORMAT.preview_factors[latent_channel][channel],
                    value,
                )
            });
            assert_eq!(preview[channel], expected);
        }
        assert!(matches!(
            empty_sd15_latent(&backend, 1, 7, 32, &context),
            Err(NativeDiffusionModelError::InvalidLatentDimensions { .. })
        ));
        drop(preview);
        assert_eq!(authorization.in_use_bytes(), 0);
        assert!(authorization.peak_bytes() >= 256);

        let underauthorization = workspace_authority.authorize_workspace(8)?;
        let underauthorized_context =
            backend.execution_context(StreamId::DEFAULT, underauthorization.clone(), &cancellation);
        let underauthorized_preview =
            sd15_latent_preview(&backend, &latent, &underauthorized_context)?;
        assert_eq!(underauthorized_preview.descriptor().shape(), &[1, 3, 1, 1]);
        drop(underauthorized_preview);
        assert_eq!(underauthorization.in_use_bytes(), 0);

        let (foreign_backend, foreign_workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let foreign_authorization = foreign_workspace_authority.authorize_workspace(4096)?;
        let foreign_context = foreign_backend.execution_context(
            StreamId::DEFAULT,
            foreign_authorization.clone(),
            &cancellation,
        );
        assert!(matches!(
            empty_sd15_latent(&backend, 1, 32, 32, &foreign_context),
            Err(NativeDiffusionModelError::TensorBackend(
                TensorError::WorkspaceAuthorizationMismatch { .. }
            ))
        ));
        assert_eq!(foreign_authorization.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_authorization = workspace_authority.authorize_workspace(4096)?;
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            cancelled_authorization.clone(),
            &cancelled,
        );
        assert!(matches!(
            sd15_latent_preview(&backend, &latent, &cancelled_context),
            Err(NativeDiffusionModelError::Cancelled)
        ));
        assert_eq!(cancelled_authorization.in_use_bytes(), 0);
        let root = workspace()?;
        let decoded_path =
            root.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1/vae-decoded.f32le");
        assert_eq!(fs::metadata(&decoded_path)?.len(), 3 * 32 * 32 * 4);
        Ok(())
    }

    #[test]
    fn manifest_is_complete_and_unique() -> Result<(), NativeDiffusionModelError> {
        let manifest = sd15_tiny_weight_manifest()?;
        assert!(manifest.len() > 100);
        assert!(
            manifest
                .iter()
                .any(|weight| weight.key == "model.diffusion_model.out.2.weight")
        );
        assert!(
            manifest
                .iter()
                .any(|weight| weight.key == "first_stage_model.quant_conv.weight")
        );
        assert!(manifest.windows(2).all(|pair| pair[0].key < pair[1].key));
        Ok(())
    }
}
