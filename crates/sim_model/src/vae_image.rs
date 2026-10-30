use crate::vision_models::canonical_vision_model_store_dtype;
use crate::{
    ArtifactIndex, AttentionBackend, AttentionFallbackPolicy, AttentionRequest,
    LatentFormatDefinition, LoadedModel, ModelStore, NativeEfficientNetV2SFeatureSource,
    NativeModule, NativeOpsError, NativeVisionModelError, NativeVisionStateKind,
    NativeVisionStateSpec, VaeDescriptor, VaeError, VaeKernelProfile, VaeLoaderConfiguration,
    efficientnet_v2_s_features_from_module_with_context,
    load_stage_c_efficientnet_feature_module_from_model_store_with_context,
    load_vision_state_from_model_store_with_context,
    load_vision_state_with_sibling_namespaces_from_model_store_with_context,
    scaled_dot_product_attention_with_context,
    vae::{NativeVae, VaeKernelFunctions, VaeModelBinding},
    vae_architecture::ExplicitAutoencoderKlTopology,
};
use sim_tensor::generated_activation_normalization_functional_01::{
    BatchNormTensorDirection, GeluApproximation, batch_norm_tensor_with_context_exact_native,
    channel_layer_norm_tensor_with_context_exact_native,
    channel_standardize_tensor_with_context_exact_native, gelu_scalar_exact_native,
    group_norm_tensor_with_context_exact_native, softmax_tensor_with_context_exact_native,
};
use sim_tensor::generated_neural_network_functional_01::{
    NeuralNetworkFunctionalError, pixel_shuffle_tensor_with_context_exact_native,
    pixel_unshuffle_tensor_with_context_exact_native,
};
use sim_tensor::generated_neural_network_module_02::replication_pad_2d_tensor_with_context_exact_native;
use sim_tensor::generated_sim_operator_indirection_01::{
    ConvolutionGeometry, OperatorIndirectionError, cast_to_with_context_exact_native,
};
use sim_tensor::{
    BinaryOperation, CancellationToken, ConvolutionSpec, CpuBackend, DType, DecodedScalar,
    ExecutionContext, LinearAlgebraOperation, ResizeCrop, ResizeMode, ResizeSpec, Scalar,
    ScalarSide, Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation, ViewAccess,
    generated_native_diffusion::{
        NativeDiffusionTensorError, add as sd15_add, conv2d as sd15_conv2d,
        group_norm as sd15_group_norm_operation, linear,
        nearest_upsample_2x as sd15_nearest_upsample_2x, silu as sd15_silu, tensor_from_f32,
        tensor_to_f32,
    },
};
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;

const PIXEL_SPACE_ARCHITECTURE: &str = "sim.pixel_space_convert.PixelspaceConversionVAE.v1";
const SD15_REDUCED_ARCHITECTURE: &str = "sim.ldm.models.autoencoder.AutoencoderKL.reduced.v1";
const KL_ARCHITECTURE: &str = "sim.ldm.models.autoencoder.AutoencoderKL.v1";
const TEMPORAL_ARCHITECTURE: &str = "sim.ldm.models.autoencoder.AutoencodingEngine.temporal.v1";
const TAESD_ARCHITECTURE: &str = "sim.taesd.TAESD.v1";
const STAGE_A_ARCHITECTURE: &str = "sim.ldm.cascade.stage_a.StageA.v1";
const STAGE_C_ENCODER_ARCHITECTURE: &str = "sim.ldm.cascade.stage_c.StageCEncoder.v1";
const STAGE_C_PREVIEWER_ARCHITECTURE: &str = "sim.ldm.cascade.stage_c.StageCPreviewer.v1";
const STAGE_C_COMBINED_ARCHITECTURE: &str = "sim.ldm.cascade.stage_c.StageCCombined.v1";
const HUNYUAN_IMAGE_ARCHITECTURE: &str = "sim.ldm.hunyuan_video.vae.AutoencodingEngine.image.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeImageVaeArchitecture {
    profile: VaeKernelProfile,
    architecture: String,
    state: Vec<NativeVisionStateSpec>,
    source_names: BTreeMap<String, String>,
    execution_operations: Vec<ImageVaeExecutionOperation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ImageVaeExecutionOperation {
    AveragePool2d(String),
    NearestUpsample2d(String),
    BatchNormLatent,
    DecoderTanh,
}

impl NativeImageVaeArchitecture {
    pub fn profile(&self) -> &VaeKernelProfile {
        &self.profile
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn state_schema(&self) -> &[NativeVisionStateSpec] {
        &self.state
    }
}

#[derive(Debug, Error)]
pub enum ImageVaeError {
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    NativeModule(#[from] NativeOpsError),
    #[error(transparent)]
    VisionState(#[from] NativeVisionModelError),
    #[error("image VAE profile {0:?} is not implemented by the image architecture adapter")]
    UnsupportedProfile(VaeKernelProfile),
    #[error("image VAE state {name} uses unsupported storage dtype {dtype}")]
    UnsupportedStorageDType { name: String, dtype: String },
    #[error("image VAE state is missing required tensor {0}")]
    MissingState(String),
    #[error("image VAE state contains tensor outside the architecture manifest: {0}")]
    UnexpectedState(String),
    #[error("image VAE state tensor {name} has invalid shape {shape:?}: {detail}")]
    InvalidStateShape {
        name: String,
        shape: Vec<u64>,
        detail: &'static str,
    },
    #[error("image VAE state tensor {name} has dtype {actual:?}; expected {expected:?}")]
    InvalidStateDType {
        name: String,
        expected: DType,
        actual: DType,
    },
    #[error("image VAE architecture {expected} does not match descriptor architecture {actual}")]
    ArchitectureMismatch { expected: String, actual: String },
    #[error("image VAE tensor operation failed: {0}")]
    Tensor(String),
}

pub fn inspect_image_vae_architecture(
    descriptor: &VaeDescriptor,
    model: &LoadedModel,
) -> Result<NativeImageVaeArchitecture, ImageVaeError> {
    let profile = descriptor.identity().profile().clone();
    let expected_architecture = match &profile {
        VaeKernelProfile::Sd15AutoencoderKlReducedV1 => SD15_REDUCED_ARCHITECTURE,
        VaeKernelProfile::PixelSpaceV1 => PIXEL_SPACE_ARCHITECTURE,
        VaeKernelProfile::TemporalAutoencodingEngineV1 => TEMPORAL_ARCHITECTURE,
        VaeKernelProfile::TaesdV1 => TAESD_ARCHITECTURE,
        VaeKernelProfile::StableCascadeStageAV1 => STAGE_A_ARCHITECTURE,
        VaeKernelProfile::StableCascadeStageCEncoderV1 => STAGE_C_ENCODER_ARCHITECTURE,
        VaeKernelProfile::StableCascadeStageCPreviewerV1 => STAGE_C_PREVIEWER_ARCHITECTURE,
        VaeKernelProfile::StableCascadeStageCCombinedV1 => STAGE_C_COMBINED_ARCHITECTURE,
        VaeKernelProfile::HunyuanImageV1 => HUNYUAN_IMAGE_ARCHITECTURE,
        VaeKernelProfile::AutoencoderKlV1
        | VaeKernelProfile::AutoencoderKlX4V1
        | VaeKernelProfile::AutoencoderKlBatchNormV1
        | VaeKernelProfile::ExplicitAutoencoderKlV1
        | VaeKernelProfile::AutoencodingEngineV1
        | VaeKernelProfile::AutoencodingEngineX4V1
        | VaeKernelProfile::AutoencodingEngineBatchNormV1 => KL_ARCHITECTURE,
        _ => return Err(ImageVaeError::UnsupportedProfile(profile)),
    };
    let actual_architecture = descriptor.identity().architecture().as_str();
    if actual_architecture != expected_architecture {
        return Err(ImageVaeError::ArchitectureMismatch {
            expected: expected_architecture.to_owned(),
            actual: actual_architecture.to_owned(),
        });
    }

    let source_names = legacy_quantization_source_names(&profile, model)?;
    let dtype_sentinel = state_dtype_sentinel(&profile);
    let source_sentinel = source_names
        .get(dtype_sentinel)
        .map(String::as_str)
        .unwrap_or(dtype_sentinel);
    let metadata = model
        .tensors()
        .get(source_sentinel)
        .ok_or_else(|| ImageVaeError::MissingState(source_sentinel.to_owned()))?;
    let floating_dtype =
        canonical_vision_model_store_dtype(&metadata.data_type).ok_or_else(|| {
            ImageVaeError::UnsupportedStorageDType {
                name: source_sentinel.to_owned(),
                dtype: metadata.data_type.clone(),
            }
        })?;
    if !matches!(floating_dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(ImageVaeError::InvalidStateDType {
            name: source_sentinel.to_owned(),
            expected: DType::F32,
            actual: floating_dtype,
        });
    }
    let state = source_state_manifest(&profile, descriptor, floating_dtype)?;
    let execution_operations = source_execution_operations(&profile, descriptor)?;
    let sibling_namespaces = image_vae_sibling_namespaces(&profile);
    let checkpoint_state = stage_c_checkpoint_manifest(&profile, &state, model.tensors());
    admit_source_manifest(model, &checkpoint_state, &source_names, sibling_namespaces)?;
    Ok(NativeImageVaeArchitecture {
        profile,
        architecture: expected_architecture.to_owned(),
        state,
        source_names,
        execution_operations,
    })
}

fn inspect_image_vae_architecture_from_recipe_state(
    descriptor: &VaeDescriptor,
    state: &BTreeMap<String, Tensor>,
) -> Result<NativeImageVaeArchitecture, ImageVaeError> {
    let profile = descriptor.identity().profile().clone();
    let expected_architecture = match &profile {
        VaeKernelProfile::Sd15AutoencoderKlReducedV1 => SD15_REDUCED_ARCHITECTURE,
        VaeKernelProfile::PixelSpaceV1 => PIXEL_SPACE_ARCHITECTURE,
        VaeKernelProfile::TemporalAutoencodingEngineV1 => TEMPORAL_ARCHITECTURE,
        VaeKernelProfile::TaesdV1 => TAESD_ARCHITECTURE,
        VaeKernelProfile::StableCascadeStageAV1 => STAGE_A_ARCHITECTURE,
        VaeKernelProfile::StableCascadeStageCEncoderV1 => STAGE_C_ENCODER_ARCHITECTURE,
        VaeKernelProfile::StableCascadeStageCPreviewerV1 => STAGE_C_PREVIEWER_ARCHITECTURE,
        VaeKernelProfile::StableCascadeStageCCombinedV1 => STAGE_C_COMBINED_ARCHITECTURE,
        VaeKernelProfile::HunyuanImageV1 => HUNYUAN_IMAGE_ARCHITECTURE,
        VaeKernelProfile::AutoencoderKlV1
        | VaeKernelProfile::AutoencoderKlX4V1
        | VaeKernelProfile::AutoencoderKlBatchNormV1
        | VaeKernelProfile::ExplicitAutoencoderKlV1
        | VaeKernelProfile::AutoencodingEngineV1
        | VaeKernelProfile::AutoencodingEngineX4V1
        | VaeKernelProfile::AutoencodingEngineBatchNormV1 => KL_ARCHITECTURE,
        _ => return Err(ImageVaeError::UnsupportedProfile(profile)),
    };
    if descriptor.identity().architecture().as_str() != expected_architecture {
        return Err(ImageVaeError::ArchitectureMismatch {
            expected: expected_architecture.to_owned(),
            actual: descriptor.identity().architecture().as_str().to_owned(),
        });
    }
    let names = state.keys().map(String::as_str).collect();
    let source_names = legacy_quantization_source_names_from_names(&profile, &names)?;
    let dtype_sentinel = state_dtype_sentinel(&profile);
    let source_sentinel = source_names
        .get(dtype_sentinel)
        .map(String::as_str)
        .unwrap_or(dtype_sentinel);
    let tensor = state
        .get(source_sentinel)
        .ok_or_else(|| ImageVaeError::MissingState(source_sentinel.to_owned()))?;
    let floating_dtype = tensor.descriptor().dtype();
    if !matches!(floating_dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(ImageVaeError::InvalidStateDType {
            name: source_sentinel.to_owned(),
            expected: DType::F32,
            actual: floating_dtype,
        });
    }
    let manifest = source_state_manifest(&profile, descriptor, floating_dtype)?;
    let sibling_namespaces = image_vae_sibling_namespaces(&profile);
    let expected = manifest
        .iter()
        .map(|spec| source_names.get(&spec.name).unwrap_or(&spec.name).as_str())
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(name) = state.keys().find(|name| {
        !expected.contains(name.as_str())
            && !sibling_namespaces
                .iter()
                .any(|prefix| name.starts_with(prefix))
    }) {
        return Err(ImageVaeError::UnexpectedState(name.clone()));
    }
    for spec in &manifest {
        let source_name = source_names.get(&spec.name).unwrap_or(&spec.name);
        let tensor = state
            .get(source_name)
            .ok_or_else(|| ImageVaeError::MissingState(source_name.clone()))?;
        if tensor.descriptor().shape() != spec.shape {
            return Err(ImageVaeError::InvalidStateShape {
                name: source_name.clone(),
                shape: tensor.descriptor().shape().to_vec(),
                detail: "the immutable source topology requires the exact declared dimensions",
            });
        }
        if tensor.descriptor().dtype() != spec.dtype {
            return Err(ImageVaeError::InvalidStateDType {
                name: source_name.clone(),
                expected: spec.dtype,
                actual: tensor.descriptor().dtype(),
            });
        }
    }
    Ok(NativeImageVaeArchitecture {
        profile,
        architecture: expected_architecture.to_owned(),
        state: manifest,
        source_names,
        execution_operations: source_execution_operations(
            descriptor.identity().profile(),
            descriptor,
        )?,
    })
}

fn source_execution_operations(
    profile: &VaeKernelProfile,
    descriptor: &VaeDescriptor,
) -> Result<Vec<ImageVaeExecutionOperation>, ImageVaeError> {
    let configuration =
        innermost_image_loader_configuration(descriptor.identity().loader_configuration());
    let mut operations = Vec::new();
    match configuration {
        VaeLoaderConfiguration::DefaultKl {
            batch_norm_latent: true,
            ..
        } => operations.push(ImageVaeExecutionOperation::BatchNormLatent),
        VaeLoaderConfiguration::ExplicitAutoencoderKl { params_json, .. } => {
            let topology = ExplicitAutoencoderKlTopology::parse(params_json)
                .map_err(|error| ImageVaeError::Tensor(error.to_string()))?;
            if !topology.encoder.resample_with_convolution {
                for level in 0..topology.encoder.channel_multipliers.len().saturating_sub(1) {
                    operations.push(ImageVaeExecutionOperation::AveragePool2d(format!(
                        "encoder.down.{level}.downsample.avg_pool"
                    )));
                }
            }
            if !topology.decoder.resample_with_convolution {
                for level in 1..topology.decoder.channel_multipliers.len() {
                    operations.push(ImageVaeExecutionOperation::NearestUpsample2d(format!(
                        "decoder.up.{level}.upsample.nearest"
                    )));
                }
            }
            if topology.batch_norm_latent {
                operations.push(ImageVaeExecutionOperation::BatchNormLatent);
            }
            if topology.decoder.tanh_output {
                operations.push(ImageVaeExecutionOperation::DecoderTanh);
            }
        }
        _ => {}
    }
    if matches!(
        profile,
        VaeKernelProfile::AutoencoderKlBatchNormV1
            | VaeKernelProfile::AutoencodingEngineBatchNormV1
    ) && !operations.contains(&ImageVaeExecutionOperation::BatchNormLatent)
    {
        operations.push(ImageVaeExecutionOperation::BatchNormLatent);
    }
    Ok(operations)
}

fn state_dtype_sentinel(profile: &VaeKernelProfile) -> &'static str {
    match profile {
        VaeKernelProfile::PixelSpaceV1 => "pixel_space_vae",
        VaeKernelProfile::TaesdV1 => "taesd_encoder.0.weight",
        VaeKernelProfile::StableCascadeStageAV1 => "in_block.1.weight",
        VaeKernelProfile::StableCascadeStageCEncoderV1 => "mapper.0.weight",
        VaeKernelProfile::StableCascadeStageCPreviewerV1 => "blocks.0.weight",
        VaeKernelProfile::StableCascadeStageCCombinedV1 => "encoder.mapper.0.weight",
        _ => "encoder.conv_in.weight",
    }
}

fn image_vae_sibling_namespaces(profile: &VaeKernelProfile) -> &'static [&'static str] {
    match profile {
        VaeKernelProfile::Sd15AutoencoderKlReducedV1 => {
            &["cond_stage_model.", "model.diffusion_model."]
        }
        VaeKernelProfile::StableCascadeStageCEncoderV1 => &["backbone."],
        VaeKernelProfile::StableCascadeStageCCombinedV1 => &["encoder.backbone."],
        _ => &[],
    }
}

fn legacy_quantization_source_names(
    profile: &VaeKernelProfile,
    model: &LoadedModel,
) -> Result<BTreeMap<String, String>, ImageVaeError> {
    let names = model
        .tensors()
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    legacy_quantization_source_names_from_names(profile, &names)
}

fn legacy_quantization_source_names_from_names(
    profile: &VaeKernelProfile,
    names: &std::collections::BTreeSet<&str>,
) -> Result<BTreeMap<String, String>, ImageVaeError> {
    let mut source_names = BTreeMap::new();
    if profile == &VaeKernelProfile::Sd15AutoencoderKlReducedV1 {
        for name in names {
            if let Some(canonical) = name.strip_prefix("first_stage_model.") {
                if source_names
                    .insert(canonical.to_owned(), (*name).to_owned())
                    .is_some()
                {
                    return Err(ImageVaeError::UnexpectedState((*name).to_owned()));
                }
            }
        }
        return Ok(source_names);
    }
    if !matches!(
        profile,
        VaeKernelProfile::AutoencoderKlV1
            | VaeKernelProfile::AutoencoderKlX4V1
            | VaeKernelProfile::AutoencoderKlBatchNormV1
            | VaeKernelProfile::ExplicitAutoencoderKlV1
    ) {
        return Ok(source_names);
    }
    for (legacy, canonical) in [
        ("encoder.quant_conv.", "quant_conv."),
        ("decoder.post_quant_conv.", "post_quant_conv."),
    ] {
        let has_legacy = names.iter().any(|name| name.starts_with(legacy));
        let has_canonical = names.iter().any(|name| name.starts_with(canonical));
        if has_legacy && has_canonical {
            return Err(ImageVaeError::UnexpectedState(format!(
                "{legacy}<duplicate-canonical-prefix>"
            )));
        }
        if has_legacy {
            for suffix in ["weight", "bias"] {
                source_names.insert(format!("{canonical}{suffix}"), format!("{legacy}{suffix}"));
            }
        }
    }
    Ok(source_names)
}

fn admit_source_manifest(
    model: &LoadedModel,
    manifest: &[NativeVisionStateSpec],
    source_names: &BTreeMap<String, String>,
    sibling_namespaces: &[&str],
) -> Result<(), ImageVaeError> {
    admit_source_metadata(model.tensors(), manifest, source_names, sibling_namespaces)
}

fn admit_source_metadata(
    tensors: &BTreeMap<String, crate::formats::TensorMetadata>,
    manifest: &[NativeVisionStateSpec],
    source_names: &BTreeMap<String, String>,
    sibling_namespaces: &[&str],
) -> Result<(), ImageVaeError> {
    let expected = manifest
        .iter()
        .map(|spec| {
            source_names
                .get(&spec.name)
                .map(String::as_str)
                .unwrap_or(&spec.name)
        })
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(name) = tensors.keys().find(|name| {
        !expected.contains(name.as_str())
            && !sibling_namespaces
                .iter()
                .any(|prefix| name.starts_with(prefix))
    }) {
        return Err(ImageVaeError::UnexpectedState(name.clone()));
    }
    for spec in manifest {
        let source_name = source_names.get(&spec.name).unwrap_or(&spec.name);
        let metadata = tensors
            .get(source_name)
            .ok_or_else(|| ImageVaeError::MissingState(source_name.clone()))?;
        if metadata.shape != spec.shape {
            return Err(ImageVaeError::InvalidStateShape {
                name: source_name.clone(),
                shape: metadata.shape.clone(),
                detail: "the immutable source topology requires the exact declared dimensions",
            });
        }
        let actual = canonical_vision_model_store_dtype(&metadata.data_type).ok_or_else(|| {
            ImageVaeError::UnsupportedStorageDType {
                name: source_name.clone(),
                dtype: metadata.data_type.clone(),
            }
        })?;
        if actual != spec.dtype {
            return Err(ImageVaeError::InvalidStateDType {
                name: source_name.clone(),
                expected: spec.dtype,
                actual,
            });
        }
    }
    Ok(())
}

struct SourceStateManifest {
    dtype: DType,
    state: Vec<NativeVisionStateSpec>,
}

impl SourceStateManifest {
    fn new(dtype: DType) -> Self {
        Self {
            dtype,
            state: Vec::new(),
        }
    }

    fn parameter(&mut self, name: impl Into<String>, shape: impl Into<Vec<u64>>) {
        self.state.push(NativeVisionStateSpec {
            name: name.into(),
            shape: shape.into(),
            dtype: self.dtype,
            kind: NativeVisionStateKind::Parameter,
        });
    }

    fn buffer(&mut self, name: impl Into<String>, shape: impl Into<Vec<u64>>, dtype: DType) {
        self.state.push(NativeVisionStateSpec {
            name: name.into(),
            shape: shape.into(),
            dtype,
            kind: NativeVisionStateKind::Buffer,
        });
    }

    fn convolution(&mut self, name: &str, output: u64, input: u64, kernel: &[u64], bias: bool) {
        let mut shape = vec![output, input];
        shape.extend_from_slice(kernel);
        self.parameter(format!("{name}.weight"), shape);
        if bias {
            self.parameter(format!("{name}.bias"), vec![output]);
        }
    }

    fn normalization(&mut self, name: &str, channels: u64) {
        self.parameter(format!("{name}.weight"), vec![channels]);
        self.parameter(format!("{name}.bias"), vec![channels]);
    }

    fn batch_normalization(&mut self, name: &str, channels: u64, affine: bool) {
        if affine {
            self.normalization(name, channels);
        }
        self.buffer(format!("{name}.running_mean"), vec![channels], self.dtype);
        self.buffer(format!("{name}.running_var"), vec![channels], self.dtype);
        self.buffer(
            format!("{name}.num_batches_tracked"),
            Vec::new(),
            DType::I64,
        );
    }

    fn resnet(&mut self, name: &str, input: u64, output: u64) {
        self.normalization(&format!("{name}.norm1"), input);
        self.convolution(&format!("{name}.conv1"), output, input, &[3, 3], true);
        self.normalization(&format!("{name}.norm2"), output);
        self.convolution(&format!("{name}.conv2"), output, output, &[3, 3], true);
        if input != output {
            self.convolution(
                &format!("{name}.nin_shortcut"),
                output,
                input,
                &[1, 1],
                true,
            );
        }
    }

    fn attention(&mut self, name: &str, channels: u64) {
        self.normalization(&format!("{name}.norm"), channels);
        for projection in ["q", "k", "v", "proj_out"] {
            self.convolution(
                &format!("{name}.{projection}"),
                channels,
                channels,
                &[1, 1],
                true,
            );
        }
    }
}

fn source_state_manifest(
    profile: &VaeKernelProfile,
    descriptor: &VaeDescriptor,
    dtype: DType,
) -> Result<Vec<NativeVisionStateSpec>, ImageVaeError> {
    image_vae_source_state_schema(profile, descriptor.identity().loader_configuration(), dtype)
}

pub fn image_vae_source_state_schema(
    profile: &VaeKernelProfile,
    loader_configuration: &VaeLoaderConfiguration,
    dtype: DType,
) -> Result<Vec<NativeVisionStateSpec>, ImageVaeError> {
    let state = match profile {
        VaeKernelProfile::Sd15AutoencoderKlReducedV1 => sd15_reduced_manifest(dtype),
        VaeKernelProfile::PixelSpaceV1 => pixel_space_manifest(dtype),
        VaeKernelProfile::TaesdV1 => {
            let latent_channels = match innermost_image_loader_configuration(loader_configuration) {
                VaeLoaderConfiguration::Taesd {
                    latent_channels, ..
                } => *latent_channels,
                VaeLoaderConfiguration::Automatic => 4,
                _ => return Err(ImageVaeError::UnsupportedProfile(profile.clone())),
            };
            taesd_manifest(latent_channels, dtype)?
        }
        VaeKernelProfile::StableCascadeStageAV1 => stage_a_manifest(dtype),
        VaeKernelProfile::StableCascadeStageCEncoderV1 => stage_c_encoder_manifest("", dtype),
        VaeKernelProfile::StableCascadeStageCPreviewerV1 => stage_c_previewer_manifest("", dtype),
        VaeKernelProfile::StableCascadeStageCCombinedV1 => {
            let mut state = stage_c_encoder_manifest("encoder.", dtype);
            state.extend(stage_c_previewer_manifest("previewer.", dtype));
            state
        }
        VaeKernelProfile::HunyuanImageV1 => hunyuan_image_manifest(dtype),
        VaeKernelProfile::TemporalAutoencodingEngineV1 => temporal_kl_manifest(dtype),
        VaeKernelProfile::AutoencoderKlV1
        | VaeKernelProfile::AutoencoderKlBatchNormV1
        | VaeKernelProfile::AutoencodingEngineV1
        | VaeKernelProfile::AutoencodingEngineBatchNormV1 => {
            let configuration = innermost_image_loader_configuration(loader_configuration);
            let VaeLoaderConfiguration::DefaultKl {
                x4,
                asymmetric_decoder_channels,
                embed_dim,
                ..
            } = configuration
            else {
                return Err(ImageVaeError::UnsupportedProfile(profile.clone()));
            };
            configured_kl_manifest(
                profile,
                *x4,
                asymmetric_decoder_channels.unwrap_or(128),
                *embed_dim,
                dtype,
            )
        }
        VaeKernelProfile::AutoencoderKlX4V1 | VaeKernelProfile::AutoencodingEngineX4V1 => {
            let configuration = innermost_image_loader_configuration(loader_configuration);
            let VaeLoaderConfiguration::DefaultKl {
                asymmetric_decoder_channels,
                embed_dim,
                ..
            } = configuration
            else {
                return Err(ImageVaeError::UnsupportedProfile(profile.clone()));
            };
            configured_kl_manifest(
                profile,
                true,
                asymmetric_decoder_channels.unwrap_or(128),
                *embed_dim,
                dtype,
            )
        }
        VaeKernelProfile::ExplicitAutoencoderKlV1 => {
            explicit_kl_manifest(loader_configuration, dtype)?
        }
        _ => return Err(ImageVaeError::UnsupportedProfile(profile.clone())),
    };
    Ok(state)
}

fn innermost_image_loader_configuration(
    configuration: &VaeLoaderConfiguration,
) -> &VaeLoaderConfiguration {
    match configuration {
        VaeLoaderConfiguration::DiffusersPreconverted { inner, .. } => {
            innermost_image_loader_configuration(inner)
        }
        configuration => configuration,
    }
}

pub(crate) fn sd15_reduced_vae_source_state_schema(dtype: DType) -> Vec<NativeVisionStateSpec> {
    sd15_reduced_manifest(dtype)
}

fn sd15_reduced_manifest(dtype: DType) -> Vec<NativeVisionStateSpec> {
    let mut manifest = SourceStateManifest::new(dtype);
    manifest.convolution("encoder.conv_in", 32, 3, &[3, 3], true);
    manifest.convolution("encoder.conv_out", 8, 32, &[3, 3], true);
    manifest.convolution("quant_conv", 8, 8, &[1, 1], true);
    manifest.convolution("post_quant_conv", 4, 4, &[1, 1], true);
    manifest.convolution("decoder.conv_in", 128, 4, &[3, 3], true);
    manifest.resnet("decoder.mid.block_1", 128, 128);
    manifest.normalization("decoder.mid.attn_1.norm", 128);
    for projection in ["q", "k", "v", "proj_out"] {
        manifest.parameter(
            format!("decoder.mid.attn_1.{projection}.weight"),
            vec![128, 128],
        );
        manifest.parameter(format!("decoder.mid.attn_1.{projection}.bias"), vec![128]);
    }
    manifest.resnet("decoder.mid.block_2", 128, 128);
    let mut input = 128;
    for (level, output) in [128_u64, 128, 64, 32].into_iter().enumerate() {
        manifest.resnet(&format!("decoder.up.{level}.block"), input, output);
        if level < 3 {
            manifest.convolution(
                &format!("decoder.up.{level}.upsample"),
                output,
                output,
                &[3, 3],
                true,
            );
        }
        input = output;
    }
    manifest.normalization("decoder.norm_out", 32);
    manifest.convolution("decoder.conv_out", 3, 32, &[3, 3], true);
    manifest.state
}

fn pixel_space_manifest(dtype: DType) -> Vec<NativeVisionStateSpec> {
    let mut manifest = SourceStateManifest::new(dtype);
    manifest.parameter("pixel_space_vae", Vec::new());
    manifest.state
}

fn taesd_manifest(
    latent_channels: u64,
    dtype: DType,
) -> Result<Vec<NativeVisionStateSpec>, ImageVaeError> {
    if latent_channels == 0 {
        return Err(ImageVaeError::Tensor(
            "TAESD source manifest requires nonzero latent channels".to_owned(),
        ));
    }
    let flux2 = latent_channels == 128;
    let source_latent_channels = if flux2 { 32 } else { latent_channels };
    let mut manifest = SourceStateManifest::new(dtype);
    manifest.convolution("taesd_encoder.0", 64, 3, &[3, 3], true);
    for index in [1_u64, 3, 4, 5, 7, 8, 9, 11, 12, 13] {
        taesd_manifest_block(
            &mut manifest,
            &format!("taesd_encoder.{index}"),
            flux2 && index >= 11,
        );
    }
    for index in [2_u64, 6, 10] {
        manifest.convolution(&format!("taesd_encoder.{index}"), 64, 64, &[3, 3], false);
    }
    manifest.convolution(
        "taesd_encoder.14",
        source_latent_channels,
        64,
        &[3, 3],
        true,
    );
    manifest.convolution("taesd_decoder.1", 64, source_latent_channels, &[3, 3], true);
    for index in [3_u64, 4, 5, 8, 9, 10, 13, 14, 15, 18] {
        taesd_manifest_block(
            &mut manifest,
            &format!("taesd_decoder.{index}"),
            flux2 && index <= 5,
        );
    }
    for index in [7_u64, 12, 17] {
        manifest.convolution(&format!("taesd_decoder.{index}"), 64, 64, &[3, 3], false);
    }
    manifest.convolution("taesd_decoder.19", 3, 64, &[3, 3], true);
    manifest.parameter("vae_scale", Vec::new());
    manifest.parameter("vae_shift", Vec::new());
    Ok(manifest.state)
}

fn taesd_manifest_block(manifest: &mut SourceStateManifest, name: &str, pool: bool) {
    for index in [0_u64, 2, 4] {
        manifest.convolution(&format!("{name}.conv.{index}"), 64, 64, &[3, 3], true);
    }
    if pool {
        manifest.convolution(&format!("{name}.pool.0"), 256, 64, &[1, 1], false);
        manifest.normalization(&format!("{name}.pool.1"), 256);
        manifest.convolution(&format!("{name}.pool.3"), 64, 256, &[1, 1], false);
    }
}

fn stage_a_manifest(dtype: DType) -> Vec<NativeVisionStateSpec> {
    let mut manifest = SourceStateManifest::new(dtype);
    manifest.convolution("in_block.1", 192, 12, &[1, 1], true);
    stage_a_resnet(&mut manifest, "down_blocks.0", 192);
    manifest.convolution("down_blocks.1", 384, 192, &[4, 4], true);
    stage_a_resnet(&mut manifest, "down_blocks.2", 384);
    manifest.convolution("down_blocks.3.0", 4, 384, &[1, 1], false);
    manifest.batch_normalization("down_blocks.3.1", 4, true);
    manifest.parameter("vquantizer.codebook.weight", vec![8192, 4]);
    manifest.convolution("up_blocks.0.0", 384, 4, &[1, 1], true);
    for index in 1..=12 {
        stage_a_resnet(&mut manifest, &format!("up_blocks.{index}"), 384);
    }
    manifest.convolution("up_blocks.13", 384, 192, &[4, 4], true);
    stage_a_resnet(&mut manifest, "up_blocks.14", 192);
    manifest.convolution("out_block.0", 12, 192, &[1, 1], true);
    manifest.state
}

fn stage_a_resnet(manifest: &mut SourceStateManifest, name: &str, channels: u64) {
    manifest.convolution(&format!("{name}.depthwise.1"), channels, 1, &[3, 3], true);
    manifest.parameter(
        format!("{name}.channelwise.0.weight"),
        vec![channels * 4, channels],
    );
    manifest.parameter(format!("{name}.channelwise.0.bias"), vec![channels * 4]);
    manifest.parameter(
        format!("{name}.channelwise.2.weight"),
        vec![channels, channels * 4],
    );
    manifest.parameter(format!("{name}.channelwise.2.bias"), vec![channels]);
    manifest.parameter(format!("{name}.gammas"), vec![6]);
}

fn stage_c_encoder_manifest(prefix: &str, dtype: DType) -> Vec<NativeVisionStateSpec> {
    let mut manifest = SourceStateManifest::new(dtype);
    manifest.convolution(&format!("{prefix}mapper.0"), 16, 1280, &[1, 1], false);
    manifest.batch_normalization(&format!("{prefix}mapper.1"), 16, false);
    manifest.parameter(format!("{prefix}mean"), vec![3]);
    manifest.parameter(format!("{prefix}std"), vec![3]);
    manifest.state
}

fn stage_c_constructor_default(profile: &VaeKernelProfile, name: &str) -> Option<&'static [f32]> {
    let suffix = match profile {
        VaeKernelProfile::StableCascadeStageCEncoderV1 => name,
        VaeKernelProfile::StableCascadeStageCCombinedV1 => name.strip_prefix("encoder.")?,
        _ => return None,
    };
    match suffix {
        "mean" => Some(&[0.485, 0.456, 0.406]),
        "std" => Some(&[0.229, 0.224, 0.225]),
        _ => None,
    }
}

fn stage_c_checkpoint_manifest(
    profile: &VaeKernelProfile,
    manifest: &[NativeVisionStateSpec],
    tensors: &BTreeMap<String, crate::formats::TensorMetadata>,
) -> Vec<NativeVisionStateSpec> {
    manifest
        .iter()
        .filter(|spec| {
            tensors.contains_key(&spec.name)
                || stage_c_constructor_default(profile, &spec.name).is_none()
        })
        .cloned()
        .collect()
}

fn retain_stage_c_constructor_defaults(
    backend: &CpuBackend,
    profile: &VaeKernelProfile,
    manifest: &[NativeVisionStateSpec],
    state: &mut BTreeMap<String, Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<(), ImageVaeError> {
    for spec in manifest {
        context.check().map_err(VaeError::from)?;
        if state.contains_key(&spec.name) {
            continue;
        }
        if let Some(values) = stage_c_constructor_default(profile, &spec.name) {
            // Original load_state_dict(strict=False) preserves these constructor
            // parameters when the verified encoder checkpoint omits them.
            let tensor = tensor_from_f32(backend, &spec.shape, values, context).map_err(
                |error| match error {
                    NativeDiffusionTensorError::Tensor(error) => {
                        ImageVaeError::Vae(VaeError::Tensor(error))
                    }
                    error => ImageVaeError::Tensor(error.to_string()),
                },
            )?;
            let tensor = cast_to_with_context_exact_native(
                backend,
                &tensor,
                spec.dtype,
                backend.device(),
                false,
                false,
                context,
            )
            .map_err(|error| match error {
                OperatorIndirectionError::Tensor(error) => {
                    ImageVaeError::Vae(VaeError::Tensor(error))
                }
                OperatorIndirectionError::Cancelled => {
                    ImageVaeError::Vae(VaeError::Tensor(TensorError::Cancelled))
                }
                error => ImageVaeError::Tensor(error.to_string()),
            })?;
            state.insert(spec.name.clone(), tensor);
        }
    }
    context.check().map_err(VaeError::from)?;
    Ok(())
}

fn stage_c_previewer_manifest(prefix: &str, dtype: DType) -> Vec<NativeVisionStateSpec> {
    let mut manifest = SourceStateManifest::new(dtype);
    let convolutions = [
        (0_u64, 512, 16, 1),
        (3, 512, 512, 3),
        (6, 512, 256, 2),
        (9, 256, 256, 3),
        (12, 256, 128, 2),
        (15, 128, 128, 3),
        (18, 128, 128, 2),
        (21, 128, 128, 3),
        (24, 3, 128, 1),
    ];
    for (index, output, input, kernel) in convolutions {
        manifest.convolution(
            &format!("{prefix}blocks.{index}"),
            output,
            input,
            &[kernel, kernel],
            true,
        );
    }
    for (index, channels) in [
        (2_u64, 512),
        (5, 512),
        (8, 256),
        (11, 256),
        (14, 128),
        (17, 128),
        (20, 128),
        (23, 128),
    ] {
        manifest.batch_normalization(&format!("{prefix}blocks.{index}"), channels, true);
    }
    manifest.state
}

fn kl_manifest(profile: &VaeKernelProfile, x4: bool, dtype: DType) -> Vec<NativeVisionStateSpec> {
    configured_kl_manifest(profile, x4, 128, Some(4), dtype)
}

fn configured_kl_manifest(
    profile: &VaeKernelProfile,
    x4: bool,
    decoder_base_channels: u64,
    embed_dim: Option<u64>,
    dtype: DType,
) -> Vec<NativeVisionStateSpec> {
    let mut manifest = SourceStateManifest::new(dtype);
    let channel_multipliers: &[u64] = if x4 { &[1, 2, 4] } else { &[1, 2, 4, 4] };
    kl_encoder_manifest(
        &mut manifest,
        128,
        channel_multipliers,
        2,
        3,
        4,
        true,
        &[],
        true,
    );
    kl_decoder_manifest(
        &mut manifest,
        decoder_base_channels,
        channel_multipliers,
        2,
        3,
        4,
        &[],
        true,
    );
    if matches!(
        profile,
        VaeKernelProfile::AutoencoderKlV1
            | VaeKernelProfile::AutoencoderKlX4V1
            | VaeKernelProfile::AutoencoderKlBatchNormV1
            | VaeKernelProfile::ExplicitAutoencoderKlV1
    ) {
        let embed_dim = embed_dim.unwrap_or(4);
        manifest.convolution("quant_conv", embed_dim * 2, 8, &[1, 1], true);
        manifest.convolution("post_quant_conv", 4, embed_dim, &[1, 1], true);
    }
    if matches!(
        profile,
        VaeKernelProfile::AutoencoderKlBatchNormV1
            | VaeKernelProfile::AutoencodingEngineBatchNormV1
    ) {
        manifest.batch_normalization("bn", 16, false);
    }
    manifest.state
}

fn explicit_kl_manifest(
    configuration: &VaeLoaderConfiguration,
    dtype: DType,
) -> Result<Vec<NativeVisionStateSpec>, ImageVaeError> {
    configuration
        .digest()
        .map_err(|error| ImageVaeError::Tensor(error.to_string()))?;
    let VaeLoaderConfiguration::ExplicitAutoencoderKl { params_json, .. } =
        innermost_image_loader_configuration(configuration)
    else {
        return Err(ImageVaeError::UnsupportedProfile(
            VaeKernelProfile::ExplicitAutoencoderKlV1,
        ));
    };
    let topology = ExplicitAutoencoderKlTopology::parse(params_json)
        .map_err(|error| ImageVaeError::Tensor(error.to_string()))?;
    let quant_input = topology
        .encoder
        .latent_channels
        .checked_mul(2)
        .ok_or_else(|| ImageVaeError::Tensor("quantization channels overflow".to_owned()))?;
    let quant_output = topology
        .embed_dim
        .checked_mul(2)
        .ok_or_else(|| ImageVaeError::Tensor("quantization channels overflow".to_owned()))?;
    let mut manifest = SourceStateManifest::new(dtype);
    kl_encoder_manifest(
        &mut manifest,
        topology.encoder.base_channels,
        &topology.encoder.channel_multipliers,
        topology.encoder.residual_blocks,
        topology.encoder.boundary_channels,
        topology.encoder.latent_channels,
        true,
        &topology.encoder.attention_levels,
        topology.encoder.resample_with_convolution,
    );
    kl_decoder_manifest(
        &mut manifest,
        topology.decoder.base_channels,
        &topology.decoder.channel_multipliers,
        topology.decoder.residual_blocks,
        topology.decoder.boundary_channels,
        topology.decoder.latent_channels,
        &topology.decoder.attention_levels,
        topology.decoder.resample_with_convolution,
    );
    manifest.convolution("quant_conv", quant_output, quant_input, &[1, 1], true);
    manifest.convolution(
        "post_quant_conv",
        topology.decoder.latent_channels,
        topology.embed_dim,
        &[1, 1],
        true,
    );
    if topology.batch_norm_latent {
        let channels = topology
            .encoder
            .latent_channels
            .checked_mul(4)
            .ok_or_else(|| {
                ImageVaeError::Tensor("batch-normalization channels overflow".to_owned())
            })?;
        manifest.batch_normalization("bn", channels, false);
    }
    Ok(manifest.state)
}

#[allow(clippy::too_many_arguments)]
fn kl_encoder_manifest(
    manifest: &mut SourceStateManifest,
    base_channels: u64,
    multipliers: &[u64],
    residual_blocks: u64,
    input_channels: u64,
    latent_channels: u64,
    double_latent: bool,
    attention_levels: &[usize],
    resample_with_convolution: bool,
) {
    manifest.convolution(
        "encoder.conv_in",
        base_channels,
        input_channels,
        &[3, 3],
        true,
    );
    let mut channels = base_channels;
    for (level, multiplier) in multipliers.iter().copied().enumerate() {
        let output = base_channels * multiplier;
        for block in 0..residual_blocks {
            manifest.resnet(
                &format!("encoder.down.{level}.block.{block}"),
                channels,
                output,
            );
            channels = output;
            if attention_levels.contains(&level) {
                manifest.attention(&format!("encoder.down.{level}.attn.{block}"), channels);
            }
        }
        if level + 1 < multipliers.len() && resample_with_convolution {
            manifest.convolution(
                &format!("encoder.down.{level}.downsample.conv"),
                channels,
                channels,
                &[3, 3],
                true,
            );
        }
    }
    manifest.resnet("encoder.mid.block_1", channels, channels);
    manifest.attention("encoder.mid.attn_1", channels);
    manifest.resnet("encoder.mid.block_2", channels, channels);
    manifest.normalization("encoder.norm_out", channels);
    manifest.convolution(
        "encoder.conv_out",
        if double_latent {
            latent_channels * 2
        } else {
            latent_channels
        },
        channels,
        &[3, 3],
        true,
    );
}

fn kl_decoder_manifest(
    manifest: &mut SourceStateManifest,
    base_channels: u64,
    multipliers: &[u64],
    residual_blocks: u64,
    output_channels: u64,
    latent_channels: u64,
    attention_levels: &[usize],
    resample_with_convolution: bool,
) {
    let mut channels = base_channels * multipliers.last().copied().unwrap_or(1);
    manifest.convolution("decoder.conv_in", channels, latent_channels, &[3, 3], true);
    manifest.resnet("decoder.mid.block_1", channels, channels);
    manifest.attention("decoder.mid.attn_1", channels);
    manifest.resnet("decoder.mid.block_2", channels, channels);
    for (level, multiplier) in multipliers.iter().copied().enumerate().rev() {
        let output = base_channels * multiplier;
        for block in 0..=residual_blocks {
            manifest.resnet(
                &format!("decoder.up.{level}.block.{block}"),
                channels,
                output,
            );
            channels = output;
            if attention_levels.contains(&level) {
                manifest.attention(&format!("decoder.up.{level}.attn.{block}"), channels);
            }
        }
        if level > 0 && resample_with_convolution {
            manifest.convolution(
                &format!("decoder.up.{level}.upsample.conv"),
                channels,
                channels,
                &[3, 3],
                true,
            );
        }
    }
    manifest.normalization("decoder.norm_out", channels);
    manifest.convolution("decoder.conv_out", output_channels, channels, &[3, 3], true);
}

fn temporal_kl_manifest(dtype: DType) -> Vec<NativeVisionStateSpec> {
    let profile = VaeKernelProfile::TemporalAutoencodingEngineV1;
    let mut state = kl_manifest(&profile, false, dtype);
    let mut temporal = SourceStateManifest::new(dtype);
    for (name, channels) in [
        ("decoder.mid.block_1", 512_u64),
        ("decoder.mid.block_2", 512),
    ] {
        temporal_resnet(&mut temporal, name, channels);
    }
    for (level, channels) in [(3_u64, 512_u64), (2, 512), (1, 256), (0, 128)] {
        for block in 0..3 {
            temporal_resnet(
                &mut temporal,
                &format!("decoder.up.{level}.block.{block}"),
                channels,
            );
        }
    }
    temporal.convolution("decoder.conv_out.time_mix_conv", 3, 3, &[3, 1, 1], true);
    state.extend(temporal.state);
    state
}

fn temporal_resnet(manifest: &mut SourceStateManifest, name: &str, channels: u64) {
    manifest.parameter(format!("{name}.mix_factor"), vec![1]);
    manifest.normalization(&format!("{name}.time_stack.in_layers.0"), channels);
    manifest.convolution(
        &format!("{name}.time_stack.in_layers.2"),
        channels,
        channels,
        &[3, 1, 1],
        true,
    );
    manifest.normalization(&format!("{name}.time_stack.out_layers.0"), channels);
    manifest.convolution(
        &format!("{name}.time_stack.out_layers.3"),
        channels,
        channels,
        &[3, 1, 1],
        true,
    );
}

fn hunyuan_image_manifest(dtype: DType) -> Vec<NativeVisionStateSpec> {
    let mut manifest = SourceStateManifest::new(dtype);
    let encoder_channels = [128_u64, 256, 512, 512, 1024, 1024];
    manifest.convolution("encoder.conv_in", 128, 3, &[3, 3], true);
    let mut channels = 128;
    for (level, output) in encoder_channels.into_iter().enumerate() {
        for block in 0..2 {
            manifest.resnet(
                &format!("encoder.down.{level}.block.{block}"),
                channels,
                output,
            );
            channels = output;
        }
        if level < 5 {
            let next = encoder_channels[level + 1];
            manifest.convolution(
                &format!("encoder.down.{level}.downsample.conv"),
                next / 4,
                channels,
                &[3, 3],
                true,
            );
            channels = next;
        }
    }
    manifest.resnet("encoder.mid.block_1", channels, channels);
    manifest.attention("encoder.mid.attn_1", channels);
    manifest.resnet("encoder.mid.block_2", channels, channels);
    manifest.normalization("encoder.norm_out", channels);
    manifest.convolution("encoder.conv_out", 128, channels, &[3, 3], true);

    let decoder_channels = [1024_u64, 1024, 512, 512, 256, 128];
    channels = 1024;
    manifest.convolution("decoder.conv_in", channels, 64, &[3, 3], true);
    manifest.resnet("decoder.mid.block_1", channels, channels);
    manifest.attention("decoder.mid.attn_1", channels);
    manifest.resnet("decoder.mid.block_2", channels, channels);
    for (level, output) in decoder_channels.into_iter().enumerate() {
        for block in 0..3 {
            manifest.resnet(
                &format!("decoder.up.{level}.block.{block}"),
                channels,
                output,
            );
            channels = output;
        }
        if level < 5 {
            let next = decoder_channels[level + 1];
            manifest.convolution(
                &format!("decoder.up.{level}.upsample.conv"),
                next * 4,
                channels,
                &[3, 3],
                true,
            );
            channels = next;
        }
    }
    manifest.normalization("decoder.norm_out", channels);
    manifest.convolution("decoder.conv_out", 3, channels, &[3, 3], true);
    manifest.state
}

pub fn load_image_vae_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: Arc<LoadedModel>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, ImageVaeError> {
    check_image_vae_cancellation(context.cancellation)?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let architecture = inspect_image_vae_architecture(&descriptor, &model)?;
    let feature_source = match architecture.profile() {
        VaeKernelProfile::StableCascadeStageCEncoderV1 => {
            Some(NativeEfficientNetV2SFeatureSource::StableCascadeEncoder)
        }
        VaeKernelProfile::StableCascadeStageCCombinedV1 => {
            Some(NativeEfficientNetV2SFeatureSource::StableCascadeCombined)
        }
        _ => None,
    };
    let mut vision_schema = architecture
        .state_schema()
        .iter()
        .filter(|spec| !is_stage_c_backbone_state(architecture.profile(), &spec.name))
        .cloned()
        .collect::<Vec<_>>();
    vision_schema =
        stage_c_checkpoint_manifest(architecture.profile(), &vision_schema, model.tensors());
    for spec in &mut vision_schema {
        if let Some(source_name) = architecture.source_names.get(&spec.name) {
            spec.name.clone_from(source_name);
        }
    }
    let mut state = match feature_source {
        Some(NativeEfficientNetV2SFeatureSource::StableCascadeEncoder) => {
            load_vision_state_with_sibling_namespaces_from_model_store_with_context(
                backend,
                store,
                index,
                &model,
                &vision_schema,
                &["backbone."],
                context,
            )?
        }
        Some(NativeEfficientNetV2SFeatureSource::StableCascadeCombined) => {
            load_vision_state_with_sibling_namespaces_from_model_store_with_context(
                backend,
                store,
                index,
                &model,
                &vision_schema,
                &["encoder.backbone."],
                context,
            )?
        }
        None if !image_vae_sibling_namespaces(architecture.profile()).is_empty() => {
            load_vision_state_with_sibling_namespaces_from_model_store_with_context(
                backend,
                store,
                index,
                &model,
                &vision_schema,
                image_vae_sibling_namespaces(architecture.profile()),
                context,
            )?
        }
        None => load_vision_state_from_model_store_with_context(
            backend,
            store,
            index,
            &model,
            &vision_schema,
            context,
        )?,
    };
    for (canonical_name, source_name) in &architecture.source_names {
        let tensor = state
            .remove(source_name)
            .ok_or_else(|| ImageVaeError::MissingState(source_name.clone()))?;
        if state.insert(canonical_name.clone(), tensor).is_some() {
            return Err(ImageVaeError::UnexpectedState(canonical_name.clone()));
        }
    }
    retain_stage_c_constructor_defaults(
        backend,
        architecture.profile(),
        architecture.state_schema(),
        &mut state,
        context,
    )?;
    let feature_module = feature_source
        .map(|source| {
            load_stage_c_efficientnet_feature_module_from_model_store_with_context(
                backend, store, index, &model, source, context,
            )
        })
        .transpose()?;
    let module = build_native_module(
        &architecture,
        state,
        feature_module,
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
        context,
    )?;
    check_image_vae_cancellation(context.cancellation)?;
    let binding =
        VaeModelBinding::checked(&descriptor, store, model, module, context.cancellation)?;
    let functions = VaeKernelFunctions::checked(
        descriptor.identity().architecture().clone(),
        image_encode_raw,
        image_decode_raw,
    );
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

pub(crate) fn load_image_vae_from_recipe_state(
    backend: &CpuBackend,
    artifact_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, ImageVaeError> {
    check_image_vae_cancellation(context.cancellation)?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let mut state = BTreeMap::new();
    for (name, tensor) in ordered_state {
        if state.insert(name.clone(), tensor).is_some() {
            return Err(ImageVaeError::UnexpectedState(name));
        }
    }
    let architecture = inspect_image_vae_architecture_from_recipe_state(&descriptor, &state)?;
    if matches!(
        architecture.profile(),
        VaeKernelProfile::StableCascadeStageCEncoderV1
            | VaeKernelProfile::StableCascadeStageCCombinedV1
    ) {
        return Err(ImageVaeError::UnsupportedProfile(
            architecture.profile().clone(),
        ));
    }
    let mut canonical_state = BTreeMap::new();
    for specification in architecture.state_schema() {
        let source_name = architecture
            .source_names
            .get(&specification.name)
            .unwrap_or(&specification.name);
        let tensor = state
            .remove(source_name)
            .ok_or_else(|| ImageVaeError::MissingState(source_name.clone()))?;
        if canonical_state
            .insert(specification.name.clone(), tensor)
            .is_some()
        {
            return Err(ImageVaeError::UnexpectedState(specification.name.clone()));
        }
    }
    let module = build_native_module(
        &architecture,
        canonical_state,
        None,
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
        context,
    )?;
    let binding = VaeModelBinding::checked_recipe(
        &descriptor,
        artifact_sha256,
        module,
        context.cancellation,
    )?;
    let functions = VaeKernelFunctions::checked(
        descriptor.identity().architecture().clone(),
        image_encode_raw,
        image_decode_raw,
    );
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

#[cfg(feature = "test-support")]
pub(crate) fn load_image_vae_from_reduced_recipe_state(
    backend: &CpuBackend,
    artifact_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, ImageVaeError> {
    check_image_vae_cancellation(context.cancellation)?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    if descriptor.identity().profile() != &VaeKernelProfile::TaesdV1
        || descriptor.identity().architecture().as_str() != TAESD_ARCHITECTURE
    {
        return Err(ImageVaeError::UnsupportedProfile(
            descriptor.identity().profile().clone(),
        ));
    }
    if !ordered_state
        .iter()
        .any(|(name, _)| name == "taesd_decoder.1.weight")
    {
        return Err(ImageVaeError::MissingState(
            "taesd_decoder.1.weight".to_owned(),
        ));
    }
    let module = crate::vae::reduced_recipe_state_module(
        "image-vae-reduced-recipe",
        ordered_state,
        context.cancellation,
    )?;
    let binding = VaeModelBinding::checked_recipe(
        &descriptor,
        artifact_sha256,
        module,
        context.cancellation,
    )?;
    let functions = VaeKernelFunctions::checked(
        descriptor.identity().architecture().clone(),
        image_encode_raw,
        image_decode_raw,
    );
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

fn check_image_vae_cancellation(cancellation: &CancellationToken) -> Result<(), ImageVaeError> {
    cancellation
        .check()
        .map_err(TensorError::from)
        .map_err(VaeError::from)
        .map_err(ImageVaeError::from)
}

fn build_native_module(
    architecture: &NativeImageVaeArchitecture,
    mut state: BTreeMap<String, Tensor>,
    feature_module: Option<NativeModule>,
    backend: &CpuBackend,
    execution_dtype: DType,
    execution_device: sim_tensor::DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, ImageVaeError> {
    let mut children = Vec::new();
    if let Some(feature_module) = feature_module {
        children.push(feature_module);
    }
    for operation in &architecture.execution_operations {
        let module = match operation {
            ImageVaeExecutionOperation::AveragePool2d(name) => {
                NativeModule::average_pool_2d(name.clone(), [2, 2], [2, 2])?
            }
            ImageVaeExecutionOperation::NearestUpsample2d(name) => {
                NativeModule::identity(name.clone())?
            }
            ImageVaeExecutionOperation::BatchNormLatent => {
                NativeModule::identity("latent.batch_norm")?
            }
            ImageVaeExecutionOperation::DecoderTanh => NativeModule::identity("decoder.tanh_out")?,
        };
        children.push(module);
    }
    for spec in architecture.state_schema() {
        if is_stage_c_backbone_state(architecture.profile(), &spec.name) {
            continue;
        }
        if !spec.name.ends_with(".weight") || !matches!(spec.shape.len(), 2 | 4 | 5) {
            continue;
        }
        let is_channelwise_linear = spec.shape.len() == 2
            && spec.name.contains("channelwise")
            && !spec.name.starts_with("vquantizer.");
        if spec.shape.len() == 2 && !is_channelwise_linear {
            continue;
        }
        let mut input_channels = usize::try_from(spec.shape[1])
            .map_err(|_| ImageVaeError::Tensor("input channel count overflow".to_owned()))?;
        let mut output_channels = usize::try_from(spec.shape[0])
            .map_err(|_| ImageVaeError::Tensor("output channel count overflow".to_owned()))?;
        let kernel = if is_channelwise_linear {
            vec![1, 1]
        } else {
            spec.shape[2..]
                .iter()
                .copied()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ImageVaeError::Tensor("convolution kernel overflow".to_owned()))?
        };
        let kernel_height = kernel.first().copied().unwrap_or(1);
        let transposed =
            is_transposed_convolution(architecture.profile(), &spec.name, kernel_height);
        let stride = convolution_stride(
            architecture.profile(),
            &spec.name,
            kernel_height,
            transposed,
        );
        if transposed {
            std::mem::swap(&mut input_channels, &mut output_channels);
        }
        let bias_name = spec
            .name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.bias"));
        let has_bias = bias_name
            .as_ref()
            .is_some_and(|name| state.contains_key(name));
        let padding = if spec.name.contains(".depthwise.1.weight")
            || is_source_asymmetric_downsample(architecture.profile(), &spec.name)
        {
            0
        } else if transposed {
            kernel_height.saturating_sub(stride) / 2
        } else {
            kernel_height / 2
        };
        let groups = if spec.name.contains(".depthwise.1.weight") {
            input_channels = output_channels;
            output_channels
        } else {
            1
        };
        let geometry = ConvolutionGeometry::new(
            kernel.len(),
            vec![stride; kernel.len()],
            if spec.shape.len() == 5 {
                kernel.iter().map(|extent| extent / 2).collect()
            } else {
                vec![padding; kernel.len()]
            },
            vec![1; kernel.len()],
            groups,
            transposed,
            vec![0; kernel.len()],
        )
        .map_err(|error| ImageVaeError::Tensor(error.to_string()))?;
        let mut module = NativeModule::convolution(
            spec.name.clone(),
            input_channels,
            output_channels,
            kernel,
            has_bias,
            geometry,
            false,
        )?;
        let mut weight = state
            .remove(&spec.name)
            .ok_or_else(|| ImageVaeError::MissingState(spec.name.clone()))?;
        if is_channelwise_linear {
            weight = reshape_read_only(&weight, vec![spec.shape[0], spec.shape[1], 1, 1])?;
        }
        let bias = bias_name.and_then(|name| state.remove(&name));
        module.load_dense_parameters(weight, bias)?;
        children.push(module);
    }
    for (name, tensor) in state {
        children.push(NativeModule::buffer(name, tensor)?);
    }
    let mut module =
        NativeModule::module_dict(format!("image-vae:{:?}", architecture.profile()), children)?;
    module.materialize_execution_state_with_context(
        backend,
        execution_dtype,
        execution_device,
        context,
    )?;
    Ok(module)
}

fn is_stage_c_backbone_state(profile: &VaeKernelProfile, name: &str) -> bool {
    match profile {
        VaeKernelProfile::StableCascadeStageCEncoderV1 => name.starts_with("backbone."),
        VaeKernelProfile::StableCascadeStageCCombinedV1 => name.starts_with("encoder.backbone."),
        _ => false,
    }
}

fn convolution_stride(
    profile: &VaeKernelProfile,
    name: &str,
    kernel: usize,
    transposed: bool,
) -> usize {
    if transposed || is_source_asymmetric_downsample(profile, name) {
        return 2;
    }
    if profile == &VaeKernelProfile::TaesdV1
        && [
            "taesd_encoder.2.weight",
            "taesd_encoder.6.weight",
            "taesd_encoder.10.weight",
        ]
        .contains(&name)
    {
        return 2;
    }
    if profile == &VaeKernelProfile::StableCascadeStageAV1
        && name.starts_with("down_blocks.")
        && kernel == 4
    {
        return 2;
    }
    1
}

fn is_source_asymmetric_downsample(profile: &VaeKernelProfile, name: &str) -> bool {
    matches!(
        profile,
        VaeKernelProfile::AutoencoderKlV1
            | VaeKernelProfile::AutoencoderKlX4V1
            | VaeKernelProfile::AutoencoderKlBatchNormV1
            | VaeKernelProfile::ExplicitAutoencoderKlV1
            | VaeKernelProfile::AutoencodingEngineV1
            | VaeKernelProfile::AutoencodingEngineX4V1
            | VaeKernelProfile::AutoencodingEngineBatchNormV1
            | VaeKernelProfile::TemporalAutoencodingEngineV1
    ) && name.contains(".downsample.conv.weight")
}

fn is_transposed_convolution(profile: &VaeKernelProfile, name: &str, kernel: usize) -> bool {
    match profile {
        VaeKernelProfile::StableCascadeStageAV1 => {
            name.starts_with("up_blocks.") && name.ends_with(".weight") && kernel == 4
        }
        VaeKernelProfile::StableCascadeStageCPreviewerV1 => {
            ["blocks.6.weight", "blocks.12.weight", "blocks.18.weight"].contains(&name)
        }
        VaeKernelProfile::StableCascadeStageCCombinedV1 => [
            "previewer.blocks.6.weight",
            "previewer.blocks.12.weight",
            "previewer.blocks.18.weight",
        ]
        .contains(&name),
        _ => false,
    }
}

fn image_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if module.layer_name().contains("Sd15AutoencoderKlReducedV1") {
        return Err(VaeError::OperationUnavailable {
            profile: "Sd15AutoencoderKlReducedV1".to_owned(),
            operation: crate::VaeOperation::Encode,
        });
    }
    if module.layer_name().contains("PixelSpaceV1") {
        return pixel_space_encode(backend, input, context);
    }
    if module.layer_name().contains("TaesdV1") {
        return taesd_encode(module, backend, input, latent_definition, context);
    }
    if module.layer_name().contains("StableCascadeStageAV1") {
        return stage_a_encode(module, backend, input, context);
    }
    if module.layer_name().contains("StableCascadeStageC") {
        if module.layer_name().contains("PreviewerV1") {
            return Err(VaeError::OperationUnavailable {
                profile: "StableCascadeStageCPreviewerV1".to_owned(),
                operation: crate::VaeOperation::Encode,
            });
        }
        return stage_c_encode(module, backend, cpu_backend, input, context);
    }
    if module.layer_name().contains("HunyuanImageV1") {
        return hunyuan_image_encode(module, backend, input, latent_definition, context);
    }
    let mut hidden = affine_tensor(backend, input, 2.0, -1.0, context)?;
    hidden = convolution(module, backend, &hidden, "encoder.conv_in.weight", context)?;
    for (level, blocks) in residual_levels(module, "encoder.down.", false) {
        for (block_index, prefix) in blocks.into_iter().enumerate() {
            hidden = resnet_block(module, backend, &hidden, &prefix, context)?;
            let attention = format!("encoder.down.{level}.attn.{block_index}");
            if find_module(module, &format!("{attention}.q.weight")).is_some() {
                hidden = attention_block(module, backend, &hidden, &attention, context)?;
            }
        }
        let downsample = format!("encoder.down.{level}.downsample.conv.weight");
        if find_module(module, &downsample).is_some() {
            hidden = constant_pad_bottom_right(backend, &hidden, context)?;
            hidden = convolution(module, backend, &hidden, &downsample, context)?;
        } else {
            let average_pool = format!("encoder.down.{level}.downsample.avg_pool");
            if find_module(module, &average_pool).is_some() {
                hidden = average_pool_2d(module, cpu_backend, &hidden, &average_pool, context)?;
            }
        }
    }
    if find_module(module, "encoder.mid.block_1.conv1.weight").is_some() {
        hidden = resnet_block(module, backend, &hidden, "encoder.mid.block_1", context)?;
    }
    if find_module(module, "encoder.mid.attn_1.q.weight").is_some() {
        hidden = attention_block(module, backend, &hidden, "encoder.mid.attn_1", context)?;
    }
    if find_module(module, "encoder.mid.block_2.conv1.weight").is_some() {
        hidden = resnet_block(module, backend, &hidden, "encoder.mid.block_2", context)?;
    }
    hidden = group_norm(module, backend, &hidden, "encoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = convolution(module, backend, &hidden, "encoder.conv_out.weight", context)?;
    if find_module(module, "quant_conv.weight").is_some() {
        hidden = convolution(module, backend, &hidden, "quant_conv.weight", context)?;
    }
    let batch_normalized = has_image_operation(module, "latent.batch_norm");
    let mode_channels = if batch_normalized {
        latent_definition.channels / 4
    } else {
        latent_definition.channels
    };
    let mode = mode_half(backend, &hidden, mode_channels, context)?;
    if batch_normalized {
        let packed = pixel_unshuffle(backend, &mode, 2, context)?;
        batch_norm_latent(module, backend, &packed, false, context)
    } else {
        Ok(mode)
    }
}

fn image_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if module.layer_name().contains("Sd15AutoencoderKlReducedV1") {
        let cpu_backend = cpu_backend.ok_or(VaeError::ImageVaeRequiresCpuBackend)?;
        return sd15_reduced_decode(module, cpu_backend, input, context);
    }
    if module.layer_name().contains("PixelSpaceV1") {
        return pixel_space_decode(backend, input, context);
    }
    if module.layer_name().contains("TaesdV1") {
        return taesd_decode(module, backend, input, latent_definition, context);
    }
    if module.layer_name().contains("StableCascadeStageAV1") {
        return stage_a_decode(module, backend, input, context);
    }
    if module.layer_name().contains("StableCascadeStageC") {
        if module.layer_name().contains("EncoderV1") {
            return Err(VaeError::OperationUnavailable {
                profile: "StableCascadeStageCEncoderV1".to_owned(),
                operation: crate::VaeOperation::Decode,
            });
        }
        return stage_c_decode(module, backend, input, context);
    }
    if module.layer_name().contains("HunyuanImageV1") {
        return hunyuan_image_decode(module, backend, input, context);
    }
    let input = if has_image_operation(module, "latent.batch_norm") {
        let denormalized = batch_norm_latent(module, backend, input, true, context)?;
        pixel_shuffle(backend, &denormalized, 2, context)?
    } else {
        input.clone()
    };
    let temporal = module.layer_name().contains("TemporalAutoencodingEngineV1");
    let post_quant = if find_module(module, "post_quant_conv.weight").is_some() {
        "post_quant_conv.weight"
    } else {
        "decoder.post_quant_conv.weight"
    };
    let mut hidden = if find_module(module, post_quant).is_some() {
        convolution(module, backend, &input, post_quant, context)?
    } else {
        input
    };
    hidden = convolution(module, backend, &hidden, "decoder.conv_in.weight", context)?;
    if find_module(module, "decoder.mid.block_1.conv1.weight").is_some() {
        hidden = if temporal {
            temporal_resnet_block(module, backend, &hidden, "decoder.mid.block_1", context)?
        } else {
            resnet_block(module, backend, &hidden, "decoder.mid.block_1", context)?
        };
    }
    if find_module(module, "decoder.mid.attn_1.q.weight").is_some() {
        hidden = attention_block(module, backend, &hidden, "decoder.mid.attn_1", context)?;
    }
    if find_module(module, "decoder.mid.block_2.conv1.weight").is_some() {
        hidden = if temporal {
            temporal_resnet_block(module, backend, &hidden, "decoder.mid.block_2", context)?
        } else {
            resnet_block(module, backend, &hidden, "decoder.mid.block_2", context)?
        };
    }
    for (level, blocks) in residual_levels(module, "decoder.up.", true) {
        for (block_index, prefix) in blocks.into_iter().enumerate() {
            hidden = if temporal {
                temporal_resnet_block(module, backend, &hidden, &prefix, context)?
            } else {
                resnet_block(module, backend, &hidden, &prefix, context)?
            };
            let attention = format!("decoder.up.{level}.attn.{block_index}");
            if find_module(module, &format!("{attention}.q.weight")).is_some() {
                hidden = attention_block(module, backend, &hidden, &attention, context)?;
            }
        }
        let upsample = format!("decoder.up.{level}.upsample.conv.weight");
        if find_module(module, &upsample).is_some() {
            hidden = nearest_upsample_2x(backend, &hidden, context)?;
            hidden = convolution(module, backend, &hidden, &upsample, context)?;
        } else if has_image_operation(module, &format!("decoder.up.{level}.upsample.nearest")) {
            hidden = nearest_upsample_2x(backend, &hidden, context)?;
        }
    }
    hidden = group_norm(module, backend, &hidden, "decoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = convolution(module, backend, &hidden, "decoder.conv_out.weight", context)?;
    if temporal && find_module(module, "decoder.conv_out.time_mix_conv.weight").is_some() {
        let sequence = frames_to_sequence(backend, &hidden, context)?;
        let sequence = convolution(
            module,
            backend,
            &sequence,
            "decoder.conv_out.time_mix_conv.weight",
            context,
        )?;
        hidden = sequence_to_frames(backend, &sequence, context)?;
    }
    if has_image_operation(module, "decoder.tanh_out") {
        hidden = unary_tensor(backend, &hidden, UnaryOperation::HyperbolicTangent, context)?;
    }
    affine_tensor(backend, &hidden, 0.5, 0.5, context)
}

fn sd15_reduced_decode(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    let mut hidden =
        sd15_reduced_convolution(module, backend, input, "post_quant_conv", 1, 0, context)?;
    hidden = sd15_reduced_convolution(module, backend, &hidden, "decoder.conv_in", 1, 1, context)?;
    hidden = sd15_reduced_resblock(module, backend, &hidden, "decoder.mid.block_1", context)?;
    hidden = sd15_reduced_attention(module, backend, &hidden, context)?;
    hidden = sd15_reduced_resblock(module, backend, &hidden, "decoder.mid.block_2", context)?;
    for level in 0..4 {
        hidden = sd15_reduced_resblock(
            module,
            backend,
            &hidden,
            &format!("decoder.up.{level}.block"),
            context,
        )?;
        if level < 3 {
            hidden = sd15_nearest_upsample_2x(backend, &hidden, context)?;
            hidden = sd15_reduced_convolution(
                module,
                backend,
                &hidden,
                &format!("decoder.up.{level}.upsample"),
                1,
                1,
                context,
            )?;
        }
    }
    hidden = sd15_reduced_group_norm(module, backend, &hidden, "decoder.norm_out", context)?;
    hidden = sd15_silu(backend, &hidden, context)?;
    hidden = sd15_reduced_convolution(module, backend, &hidden, "decoder.conv_out", 1, 1, context)?;
    let mut values = tensor_to_f32(backend, &hidden, context)?;
    for value in values.iter_mut() {
        *value = ((value.tanh() + 1.0) * 0.5).clamp(0.0, 1.0);
    }
    tensor_from_f32(backend, hidden.descriptor().shape(), &values, context).map_err(Into::into)
}

fn sd15_reduced_attention(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let prefix = "decoder.mid.attn_1";
    let normalized =
        sd15_reduced_group_norm(module, backend, input, &format!("{prefix}.norm"), context)?;
    let query_tokens = sd15_nchw_to_tokens(backend, &normalized, context)?;
    let context_tokens = sd15_nchw_to_tokens(backend, input, context)?;
    let query = sd15_reduced_linear(
        module,
        backend,
        &query_tokens,
        &format!("{prefix}.q"),
        context,
    )?;
    let key = sd15_reduced_linear(
        module,
        backend,
        &context_tokens,
        &format!("{prefix}.k"),
        context,
    )?;
    let value = sd15_reduced_linear(
        module,
        backend,
        &context_tokens,
        &format!("{prefix}.v"),
        context,
    )?;
    let shape = input.descriptor().shape();
    let channels = usize::try_from(shape[1]).map_err(|_| VaeError::ShapeOverflow)?;
    let query_count =
        usize::try_from(query.descriptor().shape()[1]).map_err(|_| VaeError::ShapeOverflow)?;
    let key_count =
        usize::try_from(key.descriptor().shape()[1]).map_err(|_| VaeError::ShapeOverflow)?;
    let heads = 4;
    let head_dimension = channels.checked_div(heads).ok_or(VaeError::ShapeOverflow)?;
    let query_values = tensor_to_f32(backend, &query, context)?;
    let key_values = tensor_to_f32(backend, &key, context)?;
    let value_values = tensor_to_f32(backend, &value, context)?;
    let outcome = scaled_dot_product_attention_with_context(
        backend,
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
        backend,
        &[
            1,
            u64::try_from(query_count).map_err(|_| VaeError::ShapeOverflow)?,
            shape[1],
        ],
        &outcome.values,
        context,
    )?;
    let attention = sd15_reduced_linear(
        module,
        backend,
        &attention,
        &format!("{prefix}.proj_out"),
        context,
    )?;
    let attention = sd15_tokens_to_nchw(backend, &attention, shape[2], shape[3], context)?;
    sd15_add(backend, input, &attention, context).map_err(Into::into)
}

fn sd15_reduced_resblock(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden =
        sd15_reduced_group_norm(module, backend, input, &format!("{prefix}.norm1"), context)?;
    hidden = sd15_silu(backend, &hidden, context)?;
    hidden = sd15_reduced_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv1"),
        1,
        1,
        context,
    )?;
    hidden = sd15_reduced_group_norm(
        module,
        backend,
        &hidden,
        &format!("{prefix}.norm2"),
        context,
    )?;
    hidden = sd15_silu(backend, &hidden, context)?;
    hidden = sd15_reduced_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2"),
        1,
        1,
        context,
    )?;
    let input_channels = input
        .descriptor()
        .shape()
        .get(1)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let output_channels = hidden
        .descriptor()
        .shape()
        .get(1)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let residual = if input_channels == output_channels {
        input.clone()
    } else {
        sd15_reduced_convolution(
            module,
            backend,
            input,
            &format!("{prefix}.nin_shortcut"),
            1,
            0,
            context,
        )?
    };
    sd15_add(backend, &residual, &hidden, context).map_err(Into::into)
}

fn sd15_reduced_group_norm(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let channels = usize::try_from(
        input
            .descriptor()
            .shape()
            .get(1)
            .copied()
            .ok_or(VaeError::ShapeOverflow)?,
    )
    .map_err(|_| VaeError::ShapeOverflow)?;
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = find_module(module, &weight_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&weight_name))?;
    let bias = find_module(module, &bias_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&bias_name))?;
    sd15_group_norm_operation(
        backend,
        input,
        weight,
        bias,
        32.min(channels),
        1.0e-6,
        context,
    )
    .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn sd15_reduced_convolution(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    stride: usize,
    padding: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let weight_name = format!("{prefix}.weight");
    let convolution =
        find_module(module, &weight_name).ok_or_else(|| missing_module(&weight_name))?;
    let (weight, bias) = convolution.dense_parameters()?;
    sd15_conv2d(backend, input, weight, bias, stride, padding, context).map_err(Into::into)
}

fn sd15_reduced_linear(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = find_module(module, &weight_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&weight_name))?;
    let bias = find_module(module, &bias_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&bias_name))?;
    linear(backend, input, weight, Some(bias), context).map_err(Into::into)
}

fn sd15_nchw_to_tokens(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = tensor.descriptor().shape();
    if shape.len() != 4 || shape[0] != 1 {
        return Err(VaeError::InvalidShape {
            expected: vec![1, 0, 0, 0],
            actual: shape.to_vec(),
        });
    }
    let token_count = shape[2]
        .checked_mul(shape[3])
        .ok_or(VaeError::ShapeOverflow)?;
    let descriptor = tensor.descriptor().permuted_view(&[0, 2, 3, 1])?;
    let channels_last = tensor.view(descriptor, ViewAccess::ReadOnly)?;
    let descriptor = TensorDescriptor::contiguous(
        channels_last.descriptor().shape().to_vec(),
        channels_last.descriptor().dtype(),
        channels_last.descriptor().device(),
        context.stream,
    )?;
    let (channels_last, event) = backend.copy(&channels_last, descriptor, context)?;
    backend.wait_event(event, context)?;
    let descriptor = channels_last
        .descriptor()
        .reshaped_view(vec![1, token_count, shape[1]])?;
    channels_last
        .view(descriptor, ViewAccess::ReadOnly)
        .map_err(Into::into)
}

fn sd15_tokens_to_nchw(
    backend: &CpuBackend,
    tensor: &Tensor,
    height: u64,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = tensor.descriptor().shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != height * width {
        return Err(VaeError::InvalidShape {
            expected: vec![1, height * width, shape.last().copied().unwrap_or(0)],
            actual: shape.to_vec(),
        });
    }
    let descriptor = tensor
        .descriptor()
        .reshaped_view(vec![1, height, width, shape[2]])?;
    let channels_last = tensor.view(descriptor, ViewAccess::ReadOnly)?;
    let descriptor = channels_last.descriptor().permuted_view(&[0, 3, 1, 2])?;
    let channels_first = channels_last.view(descriptor, ViewAccess::ReadOnly)?;
    let descriptor = TensorDescriptor::contiguous(
        channels_first.descriptor().shape().to_vec(),
        channels_first.descriptor().dtype(),
        channels_first.descriptor().device(),
        context.stream,
    )?;
    let (channels_first, event) = backend.copy(&channels_first, descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(channels_first)
}

fn average_pool_2d(
    module: &NativeModule,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let cpu_backend = cpu_backend.ok_or(VaeError::ImageVaeRequiresCpuBackend)?;
    let mut operation = find_module(module, name)
        .cloned()
        .ok_or_else(|| missing_module(name))?;
    operation
        .forward_with_context(cpu_backend, input, context)
        .map_err(VaeError::from)
}

fn has_image_operation(module: &NativeModule, name: &str) -> bool {
    find_module(module, name).is_some()
}

fn pixel_space_encode(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    affine_tensor(backend, input, 2.0, -1.0, context)
}

fn pixel_space_decode(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    affine_tensor(backend, input, 0.5, 0.5, context)
}

fn residual_levels(module: &NativeModule, prefix: &str, reverse: bool) -> Vec<(u64, Vec<String>)> {
    let mut levels = BTreeMap::<u64, BTreeMap<u64, String>>::new();
    for child in module.children() {
        let name = child.layer_name();
        let Some(rest) = name.strip_prefix(prefix) else {
            continue;
        };
        let mut parts = rest.split('.');
        let (Some(level), Some("block"), Some(block), Some("conv1"), Some("weight")) = (
            parts.next().and_then(|value| value.parse::<u64>().ok()),
            parts.next(),
            parts.next().and_then(|value| value.parse::<u64>().ok()),
            parts.next(),
            parts.next(),
        ) else {
            continue;
        };
        levels
            .entry(level)
            .or_default()
            .insert(block, format!("{prefix}{level}.block.{block}"));
    }
    let mut levels = levels
        .into_iter()
        .map(|(level, blocks)| (level, blocks.into_values().collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    if reverse {
        levels.reverse();
    }
    levels
}

fn resnet_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut residual = input.clone();
    for shortcut in ["nin_shortcut", "conv_shortcut"] {
        let name = format!("{prefix}.{shortcut}.weight");
        if find_module(module, &name).is_some() {
            residual = convolution(module, backend, input, &name, context)?;
            break;
        }
    }
    let mut hidden = group_norm(module, backend, input, &format!("{prefix}.norm1"), context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv1.weight"),
        context,
    )?;
    hidden = group_norm(
        module,
        backend,
        &hidden,
        &format!("{prefix}.norm2"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2.weight"),
        context,
    )?;
    add_tensor(backend, &residual, &hidden, context)
}

fn temporal_resnet_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let spatial = resnet_block(module, backend, input, prefix, context)?;
    let mix_factor_name = format!("{prefix}.mix_factor");
    let Some(mix_factor) =
        find_module(module, &mix_factor_name).and_then(NativeModule::registered_buffer)
    else {
        return Ok(spatial);
    };
    let sequence = frames_to_sequence(backend, &spatial, context)?;
    let mut temporal = temporal_group_norm(
        module,
        backend,
        &sequence,
        &format!("{prefix}.time_stack.in_layers.0"),
        context,
    )?;
    temporal = silu_tensor(backend, &temporal, context)?;
    temporal = convolution(
        module,
        backend,
        &temporal,
        &format!("{prefix}.time_stack.in_layers.2.weight"),
        context,
    )?;
    temporal = temporal_group_norm(
        module,
        backend,
        &temporal,
        &format!("{prefix}.time_stack.out_layers.0"),
        context,
    )?;
    temporal = silu_tensor(backend, &temporal, context)?;
    temporal = convolution(
        module,
        backend,
        &temporal,
        &format!("{prefix}.time_stack.out_layers.3.weight"),
        context,
    )?;
    temporal = add_tensor(backend, &sequence, &temporal, context)?;
    let alpha = 1.0 / (1.0 + (-read_real(mix_factor, &[0])?).exp());
    let temporal = affine_tensor(backend, &temporal, alpha as f32, 0.0, context)?;
    let spatial = affine_tensor(backend, &sequence, (1.0 - alpha) as f32, 0.0, context)?;
    let blended = add_tensor(backend, &temporal, &spatial, context)?;
    sequence_to_frames(backend, &blended, context)
}

fn frames_to_sequence(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 {
        return Err(VaeError::ShapeOverflow);
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![1, shape[1], shape[0], shape[2], shape[3]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.allocate(descriptor, context)?;
    backend.wait_event(event, context)?;
    {
        let mut write = output.write()?;
        for frame in 0..shape[0] {
            for channel in 0..shape[1] {
                for y in 0..shape[2] {
                    context.check()?;
                    for x in 0..shape[3] {
                        write
                            .element_bytes_mut(&[0, channel, frame, y, x])?
                            .copy_from_slice(input.element_bytes(&[frame, channel, y, x])?);
                    }
                }
            }
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn sequence_to_frames(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape[0] != 1 {
        return Err(VaeError::ShapeOverflow);
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[2], shape[1], shape[3], shape[4]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.allocate(descriptor, context)?;
    backend.wait_event(event, context)?;
    {
        let mut write = output.write()?;
        for frame in 0..shape[2] {
            for channel in 0..shape[1] {
                for y in 0..shape[3] {
                    context.check()?;
                    for x in 0..shape[4] {
                        write
                            .element_bytes_mut(&[frame, channel, y, x])?
                            .copy_from_slice(input.element_bytes(&[0, channel, frame, y, x])?);
                    }
                }
            }
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn temporal_group_norm(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = find_module(module, &weight_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&weight_name))?;
    let bias = find_module(module, &bias_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&bias_name))?;
    let shape = input.descriptor().shape();
    let channels = shape.get(1).copied().ok_or(VaeError::ShapeOverflow)?;
    if !channels.is_multiple_of(32) {
        return Err(VaeError::ShapeOverflow);
    }
    group_norm_tensor_with_context_exact_native(
        backend,
        input,
        32,
        Some(weight),
        Some(bias),
        1.0e-5,
        context,
    )
    .map_err(NativeOpsError::from)
    .map_err(VaeError::from)
}

fn shape_u64(shape: &[usize]) -> Result<Vec<u64>, VaeError> {
    shape
        .iter()
        .map(|dimension| u64::try_from(*dimension).map_err(|_| VaeError::ShapeOverflow))
        .collect()
}

#[cfg(test)]
fn read_real_linear(input: &Tensor, linear: u64) -> Result<f64, VaeError> {
    match input
        .descriptor()
        .dtype()
        .decode_scalar(input.linear_element_bytes(linear)?)?
    {
        DecodedScalar::Real(value) => Ok(value),
        _ => Err(VaeError::UnsupportedDType(input.descriptor().dtype())),
    }
}

pub(crate) fn attention_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let normalized = group_norm(module, backend, input, &format!("{prefix}.norm"), context)?;
    attention_block_from_normalized(module, backend, input, &normalized, prefix, context)
}

pub(crate) fn attention_block_from_normalized(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    normalized: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let query = convolution(
        module,
        backend,
        normalized,
        &format!("{prefix}.q.weight"),
        context,
    )?;
    let key = convolution(
        module,
        backend,
        normalized,
        &format!("{prefix}.k.weight"),
        context,
    )?;
    let value = convolution(
        module,
        backend,
        normalized,
        &format!("{prefix}.v.weight"),
        context,
    )?;
    let attended = spatial_attention_from_qkv(backend, input, &query, &key, &value, context)?;
    let projected = convolution(
        module,
        backend,
        &attended,
        &format!("{prefix}.proj_out.weight"),
        context,
    )?;
    add_tensor(backend, input, &projected, context)
}

pub(crate) fn spatial_attention_from_qkv(
    backend: &dyn TensorBackend,
    input: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = query.descriptor().shape();
    if !matches!(shape.len(), 3..=5) || shape[1] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let tokens = shape[2..].iter().try_fold(1_u64, |product, extent| {
        product.checked_mul(*extent).ok_or(VaeError::ShapeOverflow)
    })?;
    let query = reshape_permute_read_only(query, vec![shape[0], shape[1], tokens], &[0, 2, 1])?;
    let key = reshape_read_only(key, vec![shape[0], shape[1], tokens])?;
    let score_descriptor = TensorDescriptor::contiguous(
        vec![shape[0], tokens, tokens],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (scores, event) = backend.linear_algebra(
        LinearAlgebraOperation::BatchMatrixMultiply,
        &[query, key],
        score_descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let scores = affine_tensor(
        backend,
        &scores,
        (shape[1] as f64).sqrt().recip() as f32,
        0.0,
        context,
    )?;
    let scores = softmax_last_dimension(backend, &scores, context)?;
    let value = reshape_read_only(value, vec![shape[0], shape[1], tokens])?;
    let scores = reshape_permute_read_only(&scores, vec![shape[0], tokens, tokens], &[0, 2, 1])?;
    let attended_descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[1], tokens],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (attended, event) = backend.linear_algebra(
        LinearAlgebraOperation::BatchMatrixMultiply,
        &[value, scores],
        attended_descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    reshape_read_only(&attended, shape.to_vec())
}

pub(crate) fn softmax_last_dimension(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    softmax_tensor_with_context_exact_native(backend, input, -1, context)
        .map_err(NativeOpsError::from)
        .map_err(VaeError::from)
}

pub(crate) fn reshape_read_only(input: &Tensor, shape: Vec<u64>) -> Result<Tensor, VaeError> {
    let descriptor = input.descriptor().reshaped_view(shape)?;
    Ok(input.view(descriptor, ViewAccess::ReadOnly)?)
}

fn reshape_permute_read_only(
    input: &Tensor,
    shape: Vec<u64>,
    permutation: &[usize],
) -> Result<Tensor, VaeError> {
    let descriptor = input
        .descriptor()
        .reshaped_view(shape)?
        .permuted_view(permutation)?;
    Ok(input.view(descriptor, ViewAccess::ReadOnly)?)
}

pub(crate) fn group_norm(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = find_module(module, &weight_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&weight_name))?;
    let bias = find_module(module, &bias_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&bias_name))?;
    let shape = input.descriptor().shape();
    let channels = shape.get(1).copied().ok_or(VaeError::ShapeOverflow)?;
    if !channels.is_multiple_of(32) {
        return Err(VaeError::ShapeOverflow);
    }
    group_norm_tensor_with_context_exact_native(
        backend,
        input,
        32,
        Some(weight),
        Some(bias),
        1.0e-6,
        context,
    )
    .map_err(NativeOpsError::from)
    .map_err(VaeError::from)
}

fn batch_norm_latent(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    inverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mean = find_module(module, "bn.running_mean")
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module("bn.running_mean"))?;
    let variance = find_module(module, "bn.running_var")
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module("bn.running_var"))?;
    batch_norm_tensor_with_context_exact_native(
        backend,
        input,
        mean,
        variance,
        None,
        None,
        1.0e-4,
        if inverse {
            BatchNormTensorDirection::Denormalize
        } else {
            BatchNormTensorDirection::Normalize
        },
        context,
    )
    .map_err(NativeOpsError::from)
    .map_err(VaeError::from)
}

fn batch_norm_2d(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mean_name = format!("{prefix}.running_mean");
    let variance_name = format!("{prefix}.running_var");
    let mean = find_module(module, &mean_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&mean_name))?;
    let variance = find_module(module, &variance_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&variance_name))?;
    let weight =
        find_module(module, &format!("{prefix}.weight")).and_then(NativeModule::registered_buffer);
    let bias =
        find_module(module, &format!("{prefix}.bias")).and_then(NativeModule::registered_buffer);
    batch_norm_tensor_with_context_exact_native(
        backend,
        input,
        mean,
        variance,
        weight,
        bias,
        1.0e-5,
        BatchNormTensorDirection::Normalize,
        context,
    )
    .map_err(NativeOpsError::from)
    .map_err(VaeError::from)
}

fn read_real(tensor: &Tensor, indices: &[u64]) -> Result<f64, VaeError> {
    match tensor
        .descriptor()
        .dtype()
        .decode_scalar(tensor.element_bytes(indices)?)?
    {
        DecodedScalar::Real(value) => Ok(value),
        value => Err(VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "image VAE normalization requires real tensors, got {value:?}"
        )))),
    }
}

fn write_real(
    write: &mut sim_tensor::TensorWrite<'_>,
    dtype: DType,
    device: sim_tensor::DeviceId,
    indices: &[u64],
    value: f64,
) -> Result<(), VaeError> {
    let bytes = dtype.encode_scalar(
        Scalar::Float(value),
        "sim_model.image_vae.group_norm",
        device,
    )?;
    write.element_bytes_mut(indices)?.copy_from_slice(&bytes);
    Ok(())
}

fn missing_module(name: &str) -> VaeError {
    VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
        "missing image VAE module {name}"
    )))
}

fn map_neural_functional_error(error: NeuralNetworkFunctionalError) -> NativeOpsError {
    match error {
        NeuralNetworkFunctionalError::Cancelled => NativeOpsError::Cancelled,
        error => NativeOpsError::InvalidOwned(error.to_string()),
    }
}

fn taesd_encode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = input.clone();
    for index in 0..=14 {
        context.check()?;
        let top = format!("taesd_encoder.{index}.weight");
        let block = format!("taesd_encoder.{index}");
        if find_module(module, &top).is_some() {
            hidden = convolution(module, backend, &hidden, &top, context)?;
        } else if find_module(module, &format!("{block}.conv.0.weight")).is_some() {
            hidden = taesd_block(module, backend, &hidden, &block, context)?;
        }
    }
    let hidden = scalar_parameter_affine(
        module,
        backend,
        &hidden,
        "vae_scale",
        "vae_shift",
        true,
        context,
    )?;
    if latent_definition.channels == 128 {
        pixel_unshuffle(backend, &hidden, 2, context)
    } else {
        Ok(hidden)
    }
}

fn taesd_decode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let input = if latent_definition.channels == 128 {
        pixel_shuffle(backend, input, 2, context)?
    } else {
        input.clone()
    };
    let mut hidden = scalar_parameter_affine(
        module,
        backend,
        &input,
        "vae_scale",
        "vae_shift",
        false,
        context,
    )?;
    hidden = affine_tensor(backend, &hidden, 1.0 / 3.0, 0.0, context)?;
    hidden = unary_tensor(backend, &hidden, UnaryOperation::HyperbolicTangent, context)?;
    hidden = affine_tensor(backend, &hidden, 3.0, 0.0, context)?;
    for index in 1..=19 {
        context.check()?;
        if [7, 12, 17].contains(&index) {
            hidden = nearest_upsample_2x(backend, &hidden, context)?;
        }
        let top = format!("taesd_decoder.{index}.weight");
        let block = format!("taesd_decoder.{index}");
        if find_module(module, &top).is_some() {
            hidden = convolution(module, backend, &hidden, &top, context)?;
            if index == 1 {
                hidden = relu_tensor(backend, &hidden, context)?;
            }
        } else if find_module(module, &format!("{block}.conv.0.weight")).is_some() {
            hidden = taesd_block(module, backend, &hidden, &block, context)?;
        }
    }
    Ok(hidden)
}

fn taesd_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let input = if find_module(module, &format!("{prefix}.pool.0.weight")).is_some() {
        let mut pooled = convolution(
            module,
            backend,
            input,
            &format!("{prefix}.pool.0.weight"),
            context,
        )?;
        pooled = taesd_pool_group_norm(
            module,
            backend,
            &pooled,
            &format!("{prefix}.pool.1"),
            context,
        )?;
        pooled = relu_tensor(backend, &pooled, context)?;
        pooled = convolution(
            module,
            backend,
            &pooled,
            &format!("{prefix}.pool.3.weight"),
            context,
        )?;
        add_tensor(backend, input, &pooled, context)?
    } else {
        input.clone()
    };
    let mut residual = input.clone();
    let skip = format!("{prefix}.skip.weight");
    if find_module(module, &skip).is_some() {
        residual = convolution(module, backend, &input, &skip, context)?;
    }
    let mut hidden = input;
    for index in [0, 2, 4] {
        hidden = convolution(
            module,
            backend,
            &hidden,
            &format!("{prefix}.conv.{index}.weight"),
            context,
        )?;
        if index != 4 {
            hidden = relu_tensor(backend, &hidden, context)?;
        }
    }
    add_tensor(backend, &hidden, &residual, context)
        .and_then(|output| relu_tensor(backend, &output, context))
}

fn taesd_pool_group_norm(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = find_module(module, &weight_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&weight_name))?;
    let bias = find_module(module, &bias_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&bias_name))?;
    group_norm_tensor_with_context_exact_native(
        backend,
        input,
        4,
        Some(weight),
        Some(bias),
        1.0e-5,
        context,
    )
    .map_err(NativeOpsError::from)
    .map_err(VaeError::from)
}

fn scalar_parameter_affine(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    scale_name: &str,
    shift_name: &str,
    inverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let scale = find_module(module, scale_name).and_then(NativeModule::registered_buffer);
    let shift = find_module(module, shift_name).and_then(NativeModule::registered_buffer);
    let (scale, shift) = match (scale, shift) {
        (None, None) => return copy_tensor(backend, input, context),
        (Some(scale), Some(shift)) => (scale, shift),
        _ => {
            return Err(VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "TAESD scalar state requires both {scale_name} and {shift_name}"
            ))));
        }
    };
    let output_descriptor = input.descriptor().clone();
    if inverse {
        let (divided, event) = backend.binary(
            BinaryOperation::Divide,
            input,
            scale,
            output_descriptor.clone(),
            context,
        )?;
        backend.wait_event(event, context)?;
        let (output, event) = backend.binary(
            BinaryOperation::Add,
            &divided,
            shift,
            output_descriptor,
            context,
        )?;
        backend.wait_event(event, context)?;
        Ok(output)
    } else {
        let (shifted, event) = backend.binary(
            BinaryOperation::Subtract,
            input,
            shift,
            output_descriptor.clone(),
            context,
        )?;
        backend.wait_event(event, context)?;
        let (output, event) = backend.binary(
            BinaryOperation::Multiply,
            &shifted,
            scale,
            output_descriptor,
            context,
        )?;
        backend.wait_event(event, context)?;
        Ok(output)
    }
}

fn stage_a_encode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = pixel_unshuffle(backend, input, 2, context)?;
    hidden = convolution(module, backend, &hidden, "in_block.1.weight", context)?;
    for index in 0..64 {
        let prefix = format!("down_blocks.{index}");
        if find_module(module, &format!("{prefix}.gammas")).is_some() {
            hidden = stage_a_resblock(module, backend, &hidden, &prefix, context)?;
        } else if find_module(module, &format!("{prefix}.weight")).is_some() {
            hidden = convolution(
                module,
                backend,
                &hidden,
                &format!("{prefix}.weight"),
                context,
            )?;
        } else if find_module(module, &format!("{prefix}.0.weight")).is_some() {
            hidden = convolution(
                module,
                backend,
                &hidden,
                &format!("{prefix}.0.weight"),
                context,
            )?;
            if find_module(module, &format!("{prefix}.1.running_mean")).is_some() {
                hidden = batch_norm_2d(module, backend, &hidden, &format!("{prefix}.1"), context)?;
            }
        }
    }
    Ok(hidden)
}

fn stage_a_decode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = input.clone();
    for index in 0..64 {
        let prefix = format!("up_blocks.{index}");
        if find_module(module, &format!("{prefix}.gammas")).is_some() {
            hidden = stage_a_resblock(module, backend, &hidden, &prefix, context)?;
        } else if find_module(module, &format!("{prefix}.weight")).is_some() {
            hidden = convolution(
                module,
                backend,
                &hidden,
                &format!("{prefix}.weight"),
                context,
            )?;
        } else if find_module(module, &format!("{prefix}.0.weight")).is_some() {
            hidden = convolution(
                module,
                backend,
                &hidden,
                &format!("{prefix}.0.weight"),
                context,
            )?;
        }
    }
    hidden = convolution(module, backend, &hidden, "out_block.0.weight", context)?;
    pixel_shuffle(backend, &hidden, 2, context)
}

fn stage_a_resblock(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let gammas = find_module(module, &format!("{prefix}.gammas"))
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&format!("{prefix}.gammas")))?;
    let normalized = layer_norm_channels(backend, input, context)?;
    let mut spatial = affine_tensor(
        backend,
        &normalized,
        (1.0 + read_real(gammas, &[0])?) as f32,
        read_real(gammas, &[1])? as f32,
        context,
    )?;
    spatial = replicate_pad_2d_one(backend, &spatial, context)?;
    spatial = convolution(
        module,
        backend,
        &spatial,
        &format!("{prefix}.depthwise.1.weight"),
        context,
    )?;
    spatial = affine_tensor(
        backend,
        &spatial,
        read_real(gammas, &[2])? as f32,
        0.0,
        context,
    )?;
    let mut hidden = add_tensor(backend, input, &spatial, context)?;

    let normalized = layer_norm_channels(backend, &hidden, context)?;
    let mut channelwise = affine_tensor(
        backend,
        &normalized,
        (1.0 + read_real(gammas, &[3])?) as f32,
        read_real(gammas, &[4])? as f32,
        context,
    )?;
    channelwise = convolution(
        module,
        backend,
        &channelwise,
        &format!("{prefix}.channelwise.0.weight"),
        context,
    )?;
    channelwise = gelu_tensor(backend, &channelwise, context)?;
    channelwise = convolution(
        module,
        backend,
        &channelwise,
        &format!("{prefix}.channelwise.2.weight"),
        context,
    )?;
    channelwise = affine_tensor(
        backend,
        &channelwise,
        read_real(gammas, &[5])? as f32,
        0.0,
        context,
    )?;
    hidden = add_tensor(backend, &hidden, &channelwise, context)?;
    Ok(hidden)
}

fn layer_norm_channels(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    channel_layer_norm_tensor_with_context_exact_native(backend, input, None, None, 1.0e-6, context)
        .map_err(NativeOpsError::from)
        .map_err(VaeError::from)
}

fn replicate_pad_2d_one(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    replication_pad_2d_tensor_with_context_exact_native(backend, input, [1, 1, 1, 1], context)
        .map_err(NativeOpsError::from)
        .map_err(VaeError::from)
}

fn gelu_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    elementwise_real(backend, input, context, |value| {
        f64::from(gelu_scalar_exact_native(
            value as f32,
            GeluApproximation::None,
        ))
    })
}

fn elementwise_real(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
    operation: impl Fn(f64) -> f64,
) -> Result<Tensor, VaeError> {
    let (mut output, event) = backend.allocate(input.descriptor().clone(), context)?;
    backend.wait_event(event, context)?;
    let element_count = input.descriptor().element_count()?;
    {
        let mut write = output.write()?;
        let byte_width = usize::try_from(input.descriptor().dtype().byte_width())
            .map_err(|_| VaeError::ShapeOverflow)?;
        for index in 0..element_count {
            if index % 1024 == 0 {
                context.check()?;
            }
            let value = match input
                .descriptor()
                .dtype()
                .decode_scalar(input.linear_element_bytes(index)?)?
            {
                DecodedScalar::Real(value) => value,
                _ => return Err(VaeError::UnsupportedDType(input.descriptor().dtype())),
            };
            let bytes = input.descriptor().dtype().encode_scalar(
                Scalar::Float(operation(value)),
                "sim_model.image_vae.elementwise",
                input.descriptor().device(),
            )?;
            let start = usize::try_from(index)
                .map_err(|_| VaeError::ShapeOverflow)?
                .checked_mul(byte_width)
                .ok_or(VaeError::ShapeOverflow)?;
            let end = start
                .checked_add(byte_width)
                .ok_or(VaeError::ShapeOverflow)?;
            write
                .bytes_mut()?
                .get_mut(start..end)
                .ok_or(VaeError::ShapeOverflow)?
                .copy_from_slice(&bytes);
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn stage_c_encode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let cpu_backend = cpu_backend.ok_or(VaeError::StageCRequiresCpuBackend)?;
    let prefix = if module.layer_name().contains("Combined") {
        "encoder."
    } else {
        ""
    };
    let input = normalize_stage_c_input(module, backend, input, prefix, context)?;
    let hidden =
        efficientnet_v2_s_features_from_module_with_context(module, cpu_backend, &input, context)
            .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let hidden = convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}mapper.0.weight"),
        context,
    )?;
    batch_norm_2d(
        module,
        backend,
        &hidden,
        &format!("{prefix}mapper.1"),
        context,
    )
}

fn normalize_stage_c_input(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mean_name = format!("{prefix}mean");
    let standard_deviation_name = format!("{prefix}std");
    let mean = find_module(module, &mean_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&mean_name))?;
    let standard_deviation = find_module(module, &standard_deviation_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| missing_module(&standard_deviation_name))?;
    channel_standardize_tensor_with_context_exact_native(
        backend,
        input,
        mean,
        standard_deviation,
        context,
    )
    .map_err(NativeOpsError::from)
    .map_err(VaeError::from)
}

fn stage_c_decode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let prefix = if module.layer_name().contains("Combined") {
        "previewer.blocks."
    } else {
        "blocks."
    };
    let mut hidden = input.clone();
    for convolution_index in [0_u64, 3, 6, 9, 12, 15, 18, 21, 24] {
        let name = format!("{prefix}{convolution_index}.weight");
        hidden = convolution(module, backend, &hidden, &name, context)?;
        if convolution_index != 24 {
            hidden = gelu_tensor(backend, &hidden, context)?;
            hidden = batch_norm_2d(
                module,
                backend,
                &hidden,
                &format!("{prefix}{}", convolution_index + 2),
                context,
            )?;
        }
    }
    Ok(hidden)
}

fn hunyuan_image_encode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let input = affine_tensor(backend, input, 2.0, -1.0, context)?;
    let mut hidden = convolution(module, backend, &input, "encoder.conv_in.weight", context)?;
    for (level, blocks) in residual_levels(module, "encoder.down.", false) {
        for prefix in blocks {
            hidden = resnet_block(module, backend, &hidden, &prefix, context)?;
        }
        let downsample = format!("encoder.down.{level}.downsample.conv.weight");
        if find_module(module, &downsample).is_some() {
            let projected = convolution(module, backend, &hidden, &downsample, context)?;
            let projected = pixel_unshuffle(backend, &projected, 2, context)?;
            let residual = pixel_unshuffle(backend, &hidden, 2, context)?;
            let residual = channel_group_mean(
                backend,
                &residual,
                projected.descriptor().shape()[1],
                context,
            )?;
            hidden = add_tensor(backend, &projected, &residual, context)?;
        }
    }
    hidden = resnet_block(module, backend, &hidden, "encoder.mid.block_1", context)?;
    hidden = attention_block(module, backend, &hidden, "encoder.mid.attn_1", context)?;
    hidden = resnet_block(module, backend, &hidden, "encoder.mid.block_2", context)?;
    let skip = channel_group_mean(
        backend,
        &hidden,
        latent_definition
            .channels
            .checked_mul(2)
            .ok_or(VaeError::ShapeOverflow)?,
        context,
    )?;
    hidden = group_norm(module, backend, &hidden, "encoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = convolution(module, backend, &hidden, "encoder.conv_out.weight", context)?;
    hidden = add_tensor(backend, &hidden, &skip, context)?;
    mode_half(backend, &hidden, latent_definition.channels, context)
}

fn hunyuan_image_decode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = convolution(module, backend, input, "decoder.conv_in.weight", context)?;
    let residual = repeat_channels(backend, input, hidden.descriptor().shape()[1], context)?;
    hidden = add_tensor(backend, &hidden, &residual, context)?;
    hidden = resnet_block(module, backend, &hidden, "decoder.mid.block_1", context)?;
    hidden = attention_block(module, backend, &hidden, "decoder.mid.attn_1", context)?;
    hidden = resnet_block(module, backend, &hidden, "decoder.mid.block_2", context)?;
    for (level, blocks) in residual_levels(module, "decoder.up.", false) {
        for prefix in blocks {
            hidden = resnet_block(module, backend, &hidden, &prefix, context)?;
        }
        let upsample = format!("decoder.up.{level}.upsample.conv.weight");
        if find_module(module, &upsample).is_some() {
            let projected = convolution(module, backend, &hidden, &upsample, context)?;
            let projected = pixel_shuffle(backend, &projected, 2, context)?;
            let target_channels = projected.descriptor().shape()[1]
                .checked_mul(4)
                .ok_or(VaeError::ShapeOverflow)?;
            let residual = repeat_channels(backend, &hidden, target_channels, context)?;
            let residual = pixel_shuffle(backend, &residual, 2, context)?;
            hidden = add_tensor(backend, &projected, &residual, context)?;
        }
    }
    hidden = group_norm(module, backend, &hidden, "decoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = convolution(module, backend, &hidden, "decoder.conv_out.weight", context)?;
    affine_tensor(backend, &hidden, 0.5, 0.5, context)
}

fn channel_group_mean(
    backend: &dyn TensorBackend,
    input: &Tensor,
    target_channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || target_channels == 0 || !shape[1].is_multiple_of(target_channels) {
        return Err(VaeError::ShapeOverflow);
    }
    let group = shape[1] / target_channels;
    let output_shape = vec![shape[0], target_channels, shape[2], shape[3]];
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.allocate(descriptor, context)?;
    backend.wait_event(event, context)?;
    {
        let mut write = output.write()?;
        for batch in 0..shape[0] {
            for channel in 0..target_channels {
                for y in 0..shape[2] {
                    context.check()?;
                    for x in 0..shape[3] {
                        let mut sum = 0.0_f64;
                        for offset in 0..group {
                            sum += read_real(input, &[batch, channel * group + offset, y, x])?;
                        }
                        write_real(
                            &mut write,
                            input.descriptor().dtype(),
                            input.descriptor().device(),
                            &[batch, channel, y, x],
                            sum / group as f64,
                        )?;
                    }
                }
            }
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn repeat_channels(
    backend: &dyn TensorBackend,
    input: &Tensor,
    target_channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[1] == 0 || !target_channels.is_multiple_of(shape[1]) {
        return Err(VaeError::ShapeOverflow);
    }
    let repeat = target_channels / shape[1];
    let output_shape = vec![shape[0], target_channels, shape[2], shape[3]];
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.allocate(descriptor, context)?;
    backend.wait_event(event, context)?;
    {
        let mut write = output.write()?;
        for batch in 0..shape[0] {
            for channel in 0..target_channels {
                context.check()?;
                let source_channel = channel / repeat;
                for y in 0..shape[2] {
                    for x in 0..shape[3] {
                        write
                            .element_bytes_mut(&[batch, channel, y, x])?
                            .copy_from_slice(input.element_bytes(&[
                                batch,
                                source_channel,
                                y,
                                x,
                            ])?);
                    }
                }
            }
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

pub(crate) fn pixel_unshuffle(
    backend: &dyn TensorBackend,
    input: &Tensor,
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    pixel_unshuffle_tensor_with_context_exact_native(backend, input, factor, context)
        .map_err(map_neural_functional_error)
        .map_err(VaeError::from)
}

pub(crate) fn pixel_shuffle(
    backend: &dyn TensorBackend,
    input: &Tensor,
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    pixel_shuffle_tensor_with_context_exact_native(backend, input, factor, context)
        .map_err(map_neural_functional_error)
        .map_err(VaeError::from)
}

pub(crate) fn add_tensor(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (output, event) = backend.binary(
        BinaryOperation::Add,
        left,
        right,
        left.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

pub(crate) fn relu_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    unary_tensor(backend, input, UnaryOperation::Relu, context)
}

pub(crate) fn unary_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    operation: UnaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (output, event) = backend.unary(operation, input, input.descriptor().clone(), context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

pub(crate) fn find_module<'a>(root: &'a NativeModule, name: &str) -> Option<&'a NativeModule> {
    root.children()
        .iter()
        .find(|module| module.layer_name() == name)
}

pub(crate) fn convolution(
    root: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    let module = find_module(root, name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing image VAE module {name}"
        )))
    })?;
    let crate::NativeModuleSpec::Convolution { geometry, .. } = module.spec() else {
        return Err(VaeError::NativeOps(NativeOpsError::Invalid(
            "image VAE convolution manifest contains a non-convolution module",
        )));
    };
    let (weight, bias) = module.dense_parameters()?;
    let input_shape = input.descriptor().shape();
    let weight_shape = weight.descriptor().shape();
    let output_shape = geometry
        .checked_output_shape(
            input_shape,
            weight_shape,
            bias.map(|bias| bias.descriptor().shape()),
        )
        .map_err(NativeOpsError::from)?;
    let output = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let operation = ConvolutionSpec {
        stride: shape_u64(geometry.stride())?,
        padding: shape_u64(geometry.padding())?,
        dilation: shape_u64(geometry.dilation())?,
        groups: u64::try_from(geometry.groups()).map_err(|_| VaeError::ShapeOverflow)?,
        transposed: geometry.transposed(),
        output_padding: shape_u64(geometry.output_padding())?,
    };
    let mut inputs = vec![input.clone(), weight.clone()];
    if let Some(bias) = bias {
        inputs.push(bias.clone());
    }
    let (output, event) = backend.convolution(&operation, &inputs, output, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn mode_half(
    backend: &dyn TensorBackend,
    input: &Tensor,
    channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[1] < channels {
        return Err(VaeError::InvalidShape {
            expected: vec![0, channels, 0, 0],
            actual: shape.to_vec(),
        });
    }
    let narrowed = input.narrow_read_only(1, 0, channels)?;
    let descriptor = TensorDescriptor::contiguous(
        narrowed.descriptor().shape().to_vec(),
        narrowed.descriptor().dtype(),
        narrowed.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.copy(&narrowed, descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

pub(crate) fn nearest_upsample_2x(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 {
        return Err(VaeError::InvalidShape {
            expected: vec![0, 0, 0, 0],
            actual: shape.to_vec(),
        });
    }
    let output_height = shape[2].checked_mul(2).ok_or(VaeError::ShapeOverflow)?;
    let output_width = shape[3].checked_mul(2).ok_or(VaeError::ShapeOverflow)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[1], output_height, output_width],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.resize(
        ResizeSpec {
            width: output_width,
            height: output_height,
            mode: ResizeMode::NearestExact,
            crop: ResizeCrop::Disabled,
            antialias: false,
            align_corners: false,
        },
        input,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

pub(crate) fn affine_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    scale: f32,
    shift: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (scaled, event) = backend.binary_scalar(
        BinaryOperation::Multiply,
        input,
        Scalar::Float(f64::from(scale)),
        ScalarSide::Right,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let (shifted, event) = backend.binary_scalar(
        BinaryOperation::Add,
        &scaled,
        Scalar::Float(f64::from(shift)),
        ScalarSide::Right,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(shifted)
}

pub(crate) fn silu_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (sigmoid, event) = backend.unary(
        UnaryOperation::Sigmoid,
        input,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let (output, event) = backend.binary(
        BinaryOperation::Multiply,
        input,
        &sigmoid,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

pub(crate) fn constant_pad_bottom_right(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (output, event) = backend.constant_pad(
        input,
        &[0, 1, 0, 1],
        Some(DecodedScalar::Signed(0)),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn copy_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (output, event) = backend.copy(input, input.descriptor().clone(), context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::{FileSlice, TensorMetadata};
    use sha2::Digest;
    use sim_tensor::{CpuWorkspaceAuthority, DeviceId, StreamId};
    use sim_types::CancellationToken;
    use std::path::PathBuf;

    #[test]
    fn image_vae_cancellation_remains_typed() {
        let cancellation: CancellationToken = Default::default();
        cancellation.cancel();
        assert!(matches!(
            check_image_vae_cancellation(&cancellation),
            Err(ImageVaeError::Vae(VaeError::Tensor(TensorError::Cancelled)))
        ));
    }

    #[test]
    fn val_vae_001_stage_c_requires_concrete_cpu_backend_before_feature_lookup()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation: CancellationToken = Default::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 3, 1, 1],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (input, event) = backend.upload_f32(descriptor, &[0.0, 0.0, 0.0], &context)?;
        backend.wait_event(event, &context)?;
        let module = NativeModule::identity("image-vae:StableCascadeStageCEncoderV1")?;

        assert!(matches!(
            stage_c_encode(&module, &backend, None, &input, &context),
            Err(VaeError::StageCRequiresCpuBackend)
        ));
        Ok(())
    }

    #[test]
    fn val_vae_001_source_convolution_geometry_distinguishes_named_downsamplers() {
        assert_eq!(
            convolution_stride(
                &VaeKernelProfile::AutoencoderKlV1,
                "encoder.down.0.downsample.conv.weight",
                3,
                false,
            ),
            2
        );
        assert_eq!(
            convolution_stride(
                &VaeKernelProfile::HunyuanImageV1,
                "encoder.down.0.downsample.conv.weight",
                3,
                false,
            ),
            1
        );
        assert_eq!(
            convolution_stride(
                &VaeKernelProfile::TaesdV1,
                "taesd_encoder.6.weight",
                3,
                false,
            ),
            2
        );
        assert!(is_transposed_convolution(
            &VaeKernelProfile::StableCascadeStageCPreviewerV1,
            "blocks.12.weight",
            2,
        ));
    }

    #[test]
    fn val_vae_001_image_state_storage_dtype_delegates_to_canonical_model_materializer() {
        for (source, expected) in [
            ("F32", DType::F32),
            ("Float", DType::F32),
            ("F16", DType::F16),
            ("Half", DType::F16),
            ("BF16", DType::Bf16),
            ("BFloat16", DType::Bf16),
            ("I64", DType::I64),
            ("Long", DType::I64),
        ] {
            assert_eq!(canonical_vision_model_store_dtype(source), Some(expected));
        }
        assert_eq!(canonical_vision_model_store_dtype("F64"), None);
    }

    #[test]
    fn val_vae_001_source_manifests_fix_dimensions_and_state_ownership() {
        let reduced = sd15_reduced_manifest(DType::F32);
        assert_eq!(
            reduced
                .iter()
                .find(|spec| spec.name == "decoder.mid.attn_1.q.weight")
                .map(|spec| (spec.shape.as_slice(), spec.kind)),
            Some((&[128, 128][..], NativeVisionStateKind::Parameter))
        );
        assert_eq!(
            reduced
                .iter()
                .find(|spec| spec.name == "decoder.up.3.block.nin_shortcut.weight")
                .map(|spec| spec.shape.as_slice()),
            Some(&[32, 64, 1, 1][..])
        );

        let pixel = pixel_space_manifest(DType::F32);
        assert_eq!(pixel.len(), 1);
        assert_eq!(pixel[0].shape, Vec::<u64>::new());
        assert_eq!(pixel[0].kind, NativeVisionStateKind::Parameter);

        let stage_a = stage_a_manifest(DType::F16);
        assert_eq!(
            stage_a
                .iter()
                .find(|spec| spec.name == "vquantizer.codebook.weight")
                .map(|spec| (spec.shape.as_slice(), spec.dtype)),
            Some((&[8192, 4][..], DType::F16))
        );
        assert_eq!(
            stage_a
                .iter()
                .find(|spec| spec.name == "down_blocks.3.1.num_batches_tracked")
                .map(|spec| (spec.shape.as_slice(), spec.dtype, spec.kind)),
            Some((&[][..], DType::I64, NativeVisionStateKind::Buffer))
        );

        let standard = kl_manifest(&VaeKernelProfile::AutoencoderKlV1, false, DType::Bf16);
        assert_eq!(
            standard
                .iter()
                .find(|spec| spec.name == "encoder.down.1.block.0.conv1.weight")
                .map(|spec| (spec.shape.as_slice(), spec.dtype)),
            Some((&[256, 128, 3, 3][..], DType::Bf16))
        );

        let asymmetric = configured_kl_manifest(
            &VaeKernelProfile::AutoencoderKlV1,
            false,
            96,
            Some(6),
            DType::F32,
        );
        assert_eq!(
            asymmetric
                .iter()
                .find(|spec| spec.name == "decoder.conv_in.weight")
                .map(|spec| spec.shape.as_slice()),
            Some(&[384, 4, 3, 3][..])
        );
        assert_eq!(
            asymmetric
                .iter()
                .find(|spec| spec.name == "quant_conv.weight")
                .map(|spec| spec.shape.as_slice()),
            Some(&[12, 8, 1, 1][..])
        );
    }

    #[test]
    fn val_vae_001_explicit_manifest_is_derived_only_from_digest_bound_configuration()
    -> Result<(), Box<dyn std::error::Error>> {
        let params_json = serde_json::json!({
            "ddconfig": {
                "batch_norm_latent": true,
                "ch": 64,
                "ch_mult": [1, 2],
                "double_z": true,
                "in_channels": 1,
                "num_res_blocks": 1,
                "out_ch": 2,
                "resolution": 8,
                "attn_resolutions": [4],
                "resamp_with_conv": true,
                "z_channels": 3
            },
            "decoder_ddconfig": {
                "ch": 32,
                "ch_mult": [1, 2],
                "double_z": true,
                "in_channels": 1,
                "num_res_blocks": 1,
                "out_ch": 2,
                "resolution": 8,
                "attn_resolutions": [],
                "resamp_with_conv": true,
                "z_channels": 3
            },
            "embed_dim": 3
        })
        .to_string();
        let configuration = VaeLoaderConfiguration::ExplicitAutoencoderKl {
            params_sha256: format!("{:x}", sha2::Sha256::digest(params_json.as_bytes())),
            params_json,
        };
        let manifest = explicit_kl_manifest(&configuration, DType::F32)?;
        for (name, expected) in [
            ("encoder.conv_in.weight", &[64, 1, 3, 3][..]),
            ("decoder.conv_in.weight", &[64, 3, 3, 3][..]),
            ("quant_conv.weight", &[6, 6, 1, 1][..]),
            ("post_quant_conv.weight", &[3, 3, 1, 1][..]),
            ("bn.running_mean", &[12][..]),
        ] {
            assert_eq!(
                manifest
                    .iter()
                    .find(|spec| spec.name == name)
                    .map(|spec| spec.shape.as_slice()),
                Some(expected),
                "explicit state shape mismatch for {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn val_vae_001_source_manifest_admission_is_atomic_for_shape_dtype_and_membership() {
        let manifest = pixel_space_manifest(DType::F32);
        let source_names = BTreeMap::new();
        let metadata = |name: &str, shape: Vec<u64>, data_type: &str| TensorMetadata {
            name: name.to_owned(),
            data_type: data_type.to_owned(),
            shape,
            storage: FileSlice {
                path: PathBuf::from("fixture.safetensors"),
                offset: 0,
                length: 4,
            },
        };
        let mut tensors = BTreeMap::from([(
            "pixel_space_vae".to_owned(),
            metadata("pixel_space_vae", Vec::new(), "F32"),
        )]);
        assert!(admit_source_metadata(&tensors, &manifest, &source_names, &[]).is_ok());

        tensors.clear();
        assert!(matches!(
            admit_source_metadata(&tensors, &manifest, &source_names, &[]),
            Err(ImageVaeError::MissingState(name)) if name == "pixel_space_vae"
        ));
        tensors.insert(
            "pixel_space_vae".to_owned(),
            metadata("pixel_space_vae", vec![1], "F32"),
        );
        assert!(matches!(
            admit_source_metadata(&tensors, &manifest, &source_names, &[]),
            Err(ImageVaeError::InvalidStateShape { .. })
        ));
        tensors.insert(
            "pixel_space_vae".to_owned(),
            metadata("pixel_space_vae", Vec::new(), "I64"),
        );
        assert!(matches!(
            admit_source_metadata(&tensors, &manifest, &source_names, &[]),
            Err(ImageVaeError::InvalidStateDType { .. })
        ));
        tensors.insert(
            "unexpected".to_owned(),
            metadata("unexpected", Vec::new(), "F32"),
        );
        assert!(matches!(
            admit_source_metadata(&tensors, &manifest, &source_names, &[]),
            Err(ImageVaeError::UnexpectedState(name)) if name == "unexpected"
        ));
    }

    #[test]
    fn stage_c_constructor_defaults_admit_only_absent_known_parameters()
    -> Result<(), Box<dyn std::error::Error>> {
        for (profile, prefix) in [
            (VaeKernelProfile::StableCascadeStageCEncoderV1, ""),
            (VaeKernelProfile::StableCascadeStageCCombinedV1, "encoder."),
        ] {
            let manifest = stage_c_encoder_manifest(prefix, DType::F32);
            let mut metadata = manifest
                .iter()
                .map(|spec| {
                    (
                        spec.name.clone(),
                        TensorMetadata {
                            name: spec.name.clone(),
                            data_type: if spec.dtype == DType::I64 {
                                "I64"
                            } else {
                                "F32"
                            }
                            .to_owned(),
                            shape: spec.shape.clone(),
                            storage: FileSlice {
                                path: PathBuf::from("verified.safetensors"),
                                offset: 0,
                                length: 4,
                            },
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let mean_name = format!("{prefix}mean");
            let standard_deviation_name = format!("{prefix}std");
            metadata.remove(&mean_name);
            metadata.remove(&standard_deviation_name);
            let checkpoint = stage_c_checkpoint_manifest(&profile, &manifest, &metadata);
            assert_eq!(checkpoint.len() + 2, manifest.len());
            admit_source_metadata(&metadata, &checkpoint, &BTreeMap::new(), &[])?;
            let mut invalid = metadata.clone();
            invalid.remove(&format!("{prefix}mapper.0.weight"));
            assert!(matches!(
                admit_source_metadata(
                    &invalid,
                    &stage_c_checkpoint_manifest(&profile, &manifest, &invalid),
                    &BTreeMap::new(),
                    &[]
                ),
                Err(ImageVaeError::MissingState(_))
            ));
            let mut invalid = metadata;
            invalid.insert(
                mean_name.clone(),
                TensorMetadata {
                    name: mean_name,
                    data_type: "F32".to_owned(),
                    shape: vec![2],
                    storage: FileSlice {
                        path: PathBuf::from("verified.safetensors"),
                        offset: 0,
                        length: 8,
                    },
                },
            );
            assert!(matches!(
                admit_source_metadata(
                    &invalid,
                    &stage_c_checkpoint_manifest(&profile, &manifest, &invalid),
                    &BTreeMap::new(),
                    &[]
                ),
                Err(ImageVaeError::InvalidStateShape { .. })
            ));
        }
        assert!(stage_c_constructor_default(&VaeKernelProfile::AutoencoderKlV1, "mean").is_none());
        let cancellation = CancellationToken::default();
        let (backend, authority) = sim_tensor::CpuWorkspaceAuthority::create_backend(1024)?;
        let context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            authority.authorize_workspace(64)?,
            &cancellation,
        );
        let profile = VaeKernelProfile::StableCascadeStageCEncoderV1;
        let manifest = stage_c_encoder_manifest("", DType::F32);
        let mut state = BTreeMap::new();
        retain_stage_c_constructor_defaults(&backend, &profile, &manifest, &mut state, &context)?;
        for (name, expected) in [
            ("mean", &[0.485, 0.456, 0.406][..]),
            ("std", &[0.229, 0.224, 0.225][..]),
        ] {
            let values = tensor_to_f32(
                &backend,
                state.get(name).ok_or("default missing")?,
                &context,
            )?;
            assert_eq!(&*values, expected);
        }
        let override_values = [0.1, 0.2, 0.3];
        let override_tensor = tensor_from_f32(&backend, &[3], &override_values, &context)?;
        state.insert("mean".to_owned(), override_tensor.clone());
        retain_stage_c_constructor_defaults(&backend, &profile, &manifest, &mut state, &context)?;
        assert_eq!(
            state.get("mean").ok_or("override missing")?.storage_id(),
            override_tensor.storage_id()
        );
        let (bounded_backend, bounded_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(8)?;
        let bounded_context = bounded_backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            bounded_authority.authorize_workspace(0)?,
            &cancellation,
        );
        let mut rejected_state = BTreeMap::new();
        assert!(
            retain_stage_c_constructor_defaults(
                &bounded_backend,
                &profile,
                &manifest,
                &mut rejected_state,
                &bounded_context
            )
            .is_err()
        );
        assert!(rejected_state.is_empty());
        assert_eq!(bounded_backend.memory_snapshot().current_bytes, 0);
        let (partial_backend, partial_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(16)?;
        let partial_context = partial_backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            partial_authority.authorize_workspace(0)?,
            &cancellation,
        );
        let mut partial_state = BTreeMap::new();
        assert!(
            retain_stage_c_constructor_defaults(
                &partial_backend,
                &profile,
                &manifest,
                &mut partial_state,
                &partial_context
            )
            .is_err()
        );
        assert!(partial_state.contains_key("mean"));
        assert!(!partial_state.contains_key("std"));
        assert_eq!(partial_backend.memory_snapshot().current_bytes, 16);
        drop(partial_state);
        assert_eq!(partial_backend.memory_snapshot().current_bytes, 0);
        cancellation.cancel();
        assert!(matches!(
            retain_stage_c_constructor_defaults(
                &backend,
                &profile,
                &manifest,
                &mut BTreeMap::new(),
                &context
            ),
            Err(ImageVaeError::Vae(VaeError::Tensor(TensorError::Cancelled)))
        ));
        Ok(())
    }

    #[test]
    fn stage_c_constructor_defaults_retain_manifest_dtype() -> Result<(), Box<dyn std::error::Error>>
    {
        let cancellation = CancellationToken::default();
        let (backend, authority) = sim_tensor::CpuWorkspaceAuthority::create_backend(1024)?;
        let context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            authority.authorize_workspace(64)?,
            &cancellation,
        );
        let profile = VaeKernelProfile::StableCascadeStageCEncoderV1;
        for dtype in [DType::F32, DType::F16, DType::Bf16] {
            let manifest = stage_c_encoder_manifest("", dtype);
            let mut state = BTreeMap::new();
            retain_stage_c_constructor_defaults(
                &backend, &profile, &manifest, &mut state, &context,
            )?;
            assert_eq!(state.len(), 2);
            for name in ["mean", "std"] {
                let tensor = state.get(name).ok_or("constructor parameter missing")?;
                assert_eq!(tensor.descriptor().dtype(), dtype);
                assert_eq!(tensor.descriptor().shape(), &[3]);
            }
        }
        Ok(())
    }

    #[test]
    fn val_vae_001_every_source_topology_manifest_rejects_atomic_metadata_mutations()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut combined_stage_c = stage_c_encoder_manifest("encoder.", DType::F32);
        combined_stage_c.extend(stage_c_previewer_manifest("previewer.", DType::F32));
        let manifests = [
            ("sd15-reduced", sd15_reduced_manifest(DType::F32)),
            ("pixel", pixel_space_manifest(DType::F32)),
            ("taesd", taesd_manifest(4, DType::F32)?),
            ("taesd-flux2", taesd_manifest(128, DType::F32)?),
            ("stage-a", stage_a_manifest(DType::F32)),
            ("stage-c-encoder", stage_c_encoder_manifest("", DType::F32)),
            (
                "stage-c-previewer",
                stage_c_previewer_manifest("", DType::F32),
            ),
            ("stage-c-combined", combined_stage_c),
            ("hunyuan-image", hunyuan_image_manifest(DType::F32)),
            ("temporal", temporal_kl_manifest(DType::F32)),
            (
                "kl",
                kl_manifest(&VaeKernelProfile::AutoencoderKlV1, false, DType::F32),
            ),
            (
                "kl-x4",
                kl_manifest(&VaeKernelProfile::AutoencoderKlX4V1, true, DType::F32),
            ),
            (
                "kl-bn",
                kl_manifest(
                    &VaeKernelProfile::AutoencoderKlBatchNormV1,
                    false,
                    DType::F32,
                ),
            ),
            (
                "engine",
                kl_manifest(&VaeKernelProfile::AutoencodingEngineV1, false, DType::F32),
            ),
            (
                "engine-x4",
                kl_manifest(&VaeKernelProfile::AutoencodingEngineX4V1, true, DType::F32),
            ),
            (
                "engine-bn",
                kl_manifest(
                    &VaeKernelProfile::AutoencodingEngineBatchNormV1,
                    false,
                    DType::F32,
                ),
            ),
        ];
        for (topology, manifest) in manifests {
            assert!(!manifest.is_empty(), "{topology} manifest is empty");
            let first = manifest
                .first()
                .ok_or("source topology manifest is empty")?;
            let metadata_for = |spec: &NativeVisionStateSpec| TensorMetadata {
                name: spec.name.clone(),
                data_type: match spec.dtype {
                    DType::F32 => "F32",
                    DType::I64 => "I64",
                    _ => "UNSUPPORTED",
                }
                .to_owned(),
                shape: spec.shape.clone(),
                storage: FileSlice {
                    path: PathBuf::from(format!("{topology}.safetensors")),
                    offset: 0,
                    length: 4,
                },
            };
            let valid = manifest
                .iter()
                .map(|spec| (spec.name.clone(), metadata_for(spec)))
                .collect::<BTreeMap<_, _>>();
            assert!(
                admit_source_metadata(&valid, &manifest, &BTreeMap::new(), &[]).is_ok(),
                "{topology} valid metadata was rejected"
            );

            let mut missing = valid.clone();
            missing.remove(&first.name);
            assert!(matches!(
                admit_source_metadata(&missing, &manifest, &BTreeMap::new(), &[]),
                Err(ImageVaeError::MissingState(_))
            ));

            let mut extra = valid.clone();
            extra.insert("unexpected".to_owned(), metadata_for(first));
            assert!(matches!(
                admit_source_metadata(&extra, &manifest, &BTreeMap::new(), &[]),
                Err(ImageVaeError::UnexpectedState(_))
            ));

            let dimensional = manifest
                .iter()
                .find(|spec| !spec.shape.is_empty())
                .unwrap_or(first);
            let mut wrong_dimension = valid.clone();
            let entry = wrong_dimension
                .get_mut(&dimensional.name)
                .ok_or("manifest state is missing")?;
            if let Some(last) = entry.shape.last_mut() {
                *last = last.saturating_add(1);
            } else {
                entry.shape.push(1);
            }
            assert!(matches!(
                admit_source_metadata(&wrong_dimension, &manifest, &BTreeMap::new(), &[]),
                Err(ImageVaeError::InvalidStateShape { .. })
            ));

            let mut wrong_dtype = valid;
            wrong_dtype
                .get_mut(&first.name)
                .ok_or("manifest state is missing")?
                .data_type = if first.dtype == DType::I64 {
                "F32".to_owned()
            } else {
                "I64".to_owned()
            };
            assert!(matches!(
                admit_source_metadata(&wrong_dtype, &manifest, &BTreeMap::new(), &[]),
                Err(ImageVaeError::InvalidStateDType { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn val_vae_001_exact_legacy_quantization_prefixes_project_to_canonical_state() {
        let legacy = [
            "encoder.quant_conv.weight",
            "encoder.quant_conv.bias",
            "decoder.post_quant_conv.weight",
            "decoder.post_quant_conv.bias",
        ]
        .into_iter()
        .collect();
        let projection = legacy_quantization_source_names_from_names(
            &VaeKernelProfile::AutoencoderKlV1,
            &legacy,
        );
        assert!(matches!(
            projection,
            Ok(projection)
                if projection.get("quant_conv.weight").map(String::as_str)
                    == Some("encoder.quant_conv.weight")
                    && projection.get("post_quant_conv.bias").map(String::as_str)
                        == Some("decoder.post_quant_conv.bias")
        ));

        let collision = ["quant_conv.weight", "encoder.quant_conv.weight"]
            .into_iter()
            .collect();
        assert!(matches!(
            legacy_quantization_source_names_from_names(
                &VaeKernelProfile::AutoencoderKlV1,
                &collision,
            ),
            Err(ImageVaeError::UnexpectedState(_))
        ));

        let compiled = [
            "first_stage_model.encoder.conv_in.weight",
            "first_stage_model.decoder.conv_out.bias",
            "cond_stage_model.transformer.weight",
            "model.diffusion_model.input.weight",
        ]
        .into_iter()
        .collect();
        let projection = legacy_quantization_source_names_from_names(
            &VaeKernelProfile::Sd15AutoencoderKlReducedV1,
            &compiled,
        )
        .expect("compiled SD15 projection");
        assert_eq!(projection.len(), 2);
        assert_eq!(
            projection.get("encoder.conv_in.weight").map(String::as_str),
            Some("first_stage_model.encoder.conv_in.weight")
        );
        assert_eq!(
            projection.get("decoder.conv_out.bias").map(String::as_str),
            Some("first_stage_model.decoder.conv_out.bias")
        );
    }

    #[test]
    fn val_vae_001_sd15_reduced_encode_is_typed_unavailable()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation: CancellationToken = Default::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 3, 1, 1],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (input, event) = backend.upload_f32(descriptor, &[0.0, 0.0, 0.0], &context)?;
        backend.wait_event(event, &context)?;
        let module = NativeModule::module_dict("image-vae:Sd15AutoencoderKlReducedV1", Vec::new())?;
        assert!(matches!(
            image_encode_raw(
                &module,
                &backend,
                Some(&backend),
                &input,
                &crate::generated_sd15_sim_model_0045::LATENT_FORMAT,
                &context,
            ),
            Err(VaeError::OperationUnavailable {
                operation: crate::VaeOperation::Encode,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn val_vae_001_temporal_frame_sequence_rearrangement_round_trips_exactly()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation: CancellationToken = Default::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let descriptor = TensorDescriptor::contiguous(
            vec![2, 2, 1, 2],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (frames, event) = backend.upload_f32(descriptor, &values, &context)?;
        backend.wait_event(event, &context)?;

        let sequence = frames_to_sequence(&backend, &frames, &context)?;
        assert_eq!(sequence.descriptor().shape(), [1, 2, 2, 1, 2]);
        let round_trip = sequence_to_frames(&backend, &sequence, &context)?;
        assert_eq!(round_trip.contiguous_bytes()?, frames.contiguous_bytes()?);
        Ok(())
    }

    #[test]
    fn val_vae_001_taesd_encode_exposes_raw_image_boundary_without_extra_affine()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation: CancellationToken = Default::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 1, 1, 2],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (input, event) = backend.upload_f32(descriptor, &[0.25, 0.75], &context)?;
        backend.wait_event(event, &context)?;
        let module = NativeModule::identity("image-vae:TaesdV1")?;
        let latent_definition = crate::GENERATED_LATENT_FORMATS
            .iter()
            .find(|definition| definition.identifier == "SD15")
            .ok_or("SD15 latent format is missing")?;

        let encoded = taesd_encode(&module, &backend, &input, latent_definition, &context)?;
        assert_eq!(encoded.contiguous_bytes()?, input.contiguous_bytes()?);
        Ok(())
    }

    #[test]
    fn val_vae_001_pixel_space_boundary_maps_unit_images_to_signed_latents()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation: CancellationToken = Default::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 3, 1, 1],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (input, event) = backend.upload_f32(descriptor, &[0.0, 0.5, 1.0], &context)?;
        backend.wait_event(event, &context)?;

        let latent = pixel_space_encode(&backend, &input, &context)?;
        assert_eq!(read_real_linear(&latent, 0)?, -1.0);
        assert_eq!(read_real_linear(&latent, 1)?, 0.0);
        assert_eq!(read_real_linear(&latent, 2)?, 1.0);
        let decoded = pixel_space_decode(&backend, &latent, &context)?;
        assert_eq!(decoded.contiguous_bytes()?, input.contiguous_bytes()?);
        Ok(())
    }

    #[test]
    fn val_vae_001_kl_downsample_padding_is_bottom_right_only()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation: CancellationToken = Default::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 1, 2, 2],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (input, event) = backend.upload_f32(descriptor, &[1.0, 2.0, 3.0, 4.0], &context)?;
        backend.wait_event(event, &context)?;

        let padded = constant_pad_bottom_right(&backend, &input, &context)?;
        assert_eq!(padded.descriptor().shape(), [1, 1, 3, 3]);
        let expected = [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!(read_real_linear(&padded, u64::try_from(index)?)?, expected);
        }
        Ok(())
    }

    #[test]
    fn val_vae_001_cascade_pixel_rearrangement_round_trips_exactly()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation: CancellationToken = Default::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 4, 2, 2],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let values = (0_u16..16).map(f32::from).collect::<Vec<_>>();
        let (input, event) = backend.upload_f32(descriptor, &values, &context)?;
        backend.wait_event(event, &context)?;

        let shuffled = pixel_shuffle(&backend, &input, 2, &context)?;
        assert_eq!(shuffled.descriptor().shape(), [1, 1, 4, 4]);
        let round_trip = pixel_unshuffle(&backend, &shuffled, 2, &context)?;
        assert_eq!(round_trip.contiguous_bytes()?, input.contiguous_bytes()?);
        Ok(())
    }

    #[test]
    fn val_vae_001_stage_c_normalization_uses_pinned_channel_statistics()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation: CancellationToken = Default::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let channel_descriptor =
            TensorDescriptor::contiguous(vec![3], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        let (mean, event) =
            backend.upload_f32(channel_descriptor.clone(), &[1.0, 2.0, 3.0], &context)?;
        backend.wait_event(event, &context)?;
        let (standard_deviation, event) =
            backend.upload_f32(channel_descriptor, &[2.0, 4.0, 8.0], &context)?;
        backend.wait_event(event, &context)?;
        let module = NativeModule::module_dict(
            "image-vae:StableCascadeStageCEncoderV1",
            vec![
                NativeModule::buffer("mean", mean)?,
                NativeModule::buffer("std", standard_deviation)?,
            ],
        )?;
        let input_descriptor = TensorDescriptor::contiguous(
            vec![1, 3, 1, 1],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (input, event) = backend.upload_f32(input_descriptor, &[3.0, 6.0, 11.0], &context)?;
        backend.wait_event(event, &context)?;

        let normalized = normalize_stage_c_input(&module, &backend, &input, "", &context)?;
        for index in 0..3 {
            assert_eq!(read_real_linear(&normalized, index)?, 1.0);
        }
        Ok(())
    }
}
