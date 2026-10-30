use crate::{
    ArtifactIndex, LatentFormatDefinition, LoadedModel, ModelStore, NativeModule, NativeOpsError,
    NativeVisionModelError, NativeVisionStateKind, NativeVisionStateSpec, VaeDescriptor, VaeError,
    VaeKernelProfile, VaeLoaderConfiguration,
    vae::{NativeVae, VaeKernelFunctions, VaeModelBinding},
    vae_image::{
        add_tensor, affine_tensor, attention_block, attention_block_from_normalized,
        constant_pad_bottom_right, convolution, group_norm, nearest_upsample_2x, pixel_shuffle,
        pixel_unshuffle, relu_tensor, reshape_read_only, silu_tensor, spatial_attention_from_qkv,
        unary_tensor,
    },
    vision_models::canonical_vision_model_store_dtype,
    vision_models::load_projected_vision_state_from_model_store_with_context,
    vision_models::load_vision_state_from_model_store_with_context,
};
use sim_tensor::generated_activation_normalization_functional_01::{
    channel_layer_norm_tensor_with_context_exact_native,
    group_norm_tensor_with_context_exact_native, softmax_tensor_with_context_exact_native,
};
use sim_tensor::generated_neural_network_module_02::replication_pad_2d_tensor_with_context_exact_native;
use sim_tensor::generated_random_number_generation_01::{
    generator_exact_native, randn_like_with_context_exact_native,
};
use sim_tensor::generated_sim_operator_indirection_01::ConvolutionGeometry;
use sim_tensor::generated_sim_operator_indirection_01::tensor_from_f32_with_backend_exact_native;
use sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_backend_exact_native;
use sim_tensor::{
    BinaryOperation, CpuBackend, DType, ExecutionContext, LinearAlgebraOperation,
    ReductionOperation, ReductionSpec, RngAlgorithm, RngProfileVersion, RngTransaction, Scalar,
    ScalarSide, Tensor, TensorBackend, TensorDescriptor, UnaryOperation, ViewAccess,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use thiserror::Error;

const MOCHI_ARCHITECTURE: &str = "sim.ldm.genmo.vae.VideoVAE.v1";
const LTX_ARCHITECTURE: &str = "sim.ldm.lightricks.vae.VideoVAE.v1";
const HUNYUAN_IMAGE_REFINER_ARCHITECTURE: &str =
    "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.image.v1";
const HUNYUAN_VIDEO_REFINER_ARCHITECTURE: &str =
    "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.video.v1";
const COGVIDEOX_ARCHITECTURE: &str = "sim.ldm.cogvideo.vae.AutoencoderKLCogVideoX.v1";
const CAUSAL_3D_ARCHITECTURE: &str = "sim.ldm.models.autoencoder.AutoencoderKL.causal3d.v1";
const COSMOS_ARCHITECTURE: &str = "sim.ldm.cosmos.vae.CausalContinuousVideoTokenizer.v1";
const WAN_21_ARCHITECTURE: &str = "sim.ldm.wan.vae.WanVAE.v1";
const WAN_22_ARCHITECTURE: &str = "sim.ldm.wan.vae2_2.WanVAE.v1";
const TAEHV_ARCHITECTURE: &str = "sim.taesd.taehv.TAEHV.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoVaeSourceCheckpoint {
    pub name: &'static str,
    pub rank: u8,
    pub dimensions: &'static [(usize, u64)],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeVideoVaeArchitecture {
    profile: VaeKernelProfile,
    architecture: &'static str,
    temporal_ratio: u64,
    spatial_ratio: u64,
    checkpoints: &'static [VideoVaeSourceCheckpoint],
    equations: &'static [&'static str],
    storage_dtype: Option<DType>,
}

impl NativeVideoVaeArchitecture {
    pub fn profile(&self) -> &VaeKernelProfile {
        &self.profile
    }

    pub const fn architecture(&self) -> &'static str {
        self.architecture
    }

    pub const fn temporal_ratio(&self) -> u64 {
        self.temporal_ratio
    }

    pub const fn spatial_ratio(&self) -> u64 {
        self.spatial_ratio
    }

    pub const fn state_checkpoints(&self) -> &'static [VideoVaeSourceCheckpoint] {
        self.checkpoints
    }

    pub const fn equation_checkpoints(&self) -> &'static [&'static str] {
        self.equations
    }

    pub const fn storage_dtype(&self) -> Option<DType> {
        self.storage_dtype
    }
}

#[derive(Debug, Error)]
pub enum VideoVaeError {
    #[error(transparent)]
    Cancelled(#[from] sim_types::CancellationError),
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    NativeModule(#[from] NativeOpsError),
    #[error(transparent)]
    VisionState(#[from] NativeVisionModelError),
    #[error("video VAE profile {0:?} is not implemented by the video architecture adapter")]
    UnsupportedProfile(VaeKernelProfile),
    #[error("video VAE architecture {expected} does not match descriptor architecture {actual}")]
    ArchitectureMismatch {
        expected: &'static str,
        actual: String,
    },
    #[error("video VAE state is missing source checkpoint {0}")]
    MissingState(String),
    #[error("video VAE state contains tensor outside the immutable source topology: {0}")]
    UnexpectedState(String),
    #[error("video VAE state checkpoint {name} has invalid shape {shape:?}")]
    InvalidStateShape { name: String, shape: Vec<u64> },
    #[error("video VAE state checkpoint {name} uses unsupported storage dtype {dtype}")]
    UnsupportedStorageDType { name: String, dtype: String },
    #[error(
        "video VAE state mixes floating storage dtypes: expected {expected:?}, got {actual:?} at {name}"
    )]
    MixedStorageDType {
        name: String,
        expected: DType,
        actual: DType,
    },
    #[error("LTX video VAE stochastic decode requires a caller-provided RNG phase")]
    MissingRngPhase,
    #[error("invalid or unsupported LTX video VAE configuration: {0}")]
    InvalidLtxConfiguration(String),
    #[error("invalid Mochi video VAE checkpoint layout: {0}")]
    InvalidMochiCheckpointLayout(String),
}

struct TaeHvTopology {
    source_latent_channels: u64,
    patch_size: u64,
    encoder_temporal_strides: [u64; 3],
    decoder_temporal_strides: [u64; 3],
}

fn taehv_topology(profile: &VaeKernelProfile) -> Result<TaeHvTopology, VideoVaeError> {
    let topology = match profile {
        VaeKernelProfile::TaeHvWan22V1 => TaeHvTopology {
            source_latent_channels: 48,
            patch_size: 2,
            encoder_temporal_strides: [2, 2, 1],
            decoder_temporal_strides: [1, 2, 2],
        },
        VaeKernelProfile::TaeHvLtx2V1 => TaeHvTopology {
            source_latent_channels: 128,
            patch_size: 4,
            encoder_temporal_strides: [2, 2, 2],
            decoder_temporal_strides: [2, 2, 2],
        },
        VaeKernelProfile::LightTaeHv15V1 => TaeHvTopology {
            source_latent_channels: 32,
            patch_size: 2,
            encoder_temporal_strides: [2, 2, 1],
            decoder_temporal_strides: [1, 2, 2],
        },
        VaeKernelProfile::TaeHvHunyuanV1 | VaeKernelProfile::LightTaeWan21V1 => TaeHvTopology {
            source_latent_channels: 16,
            patch_size: 1,
            encoder_temporal_strides: [2, 2, 1],
            decoder_temporal_strides: [1, 2, 2],
        },
        profile => return Err(VideoVaeError::UnsupportedProfile(profile.clone())),
    };
    Ok(topology)
}

struct StateManifest {
    dtype: DType,
    state: Vec<NativeVisionStateSpec>,
}

impl StateManifest {
    fn new(dtype: DType) -> Self {
        Self {
            dtype,
            state: Vec::new(),
        }
    }

    fn convolution(
        &mut self,
        name: &str,
        output_channels: u64,
        input_channels: u64,
        kernel: u64,
        bias: bool,
    ) {
        self.convolution_nd(
            name,
            output_channels,
            input_channels,
            &[kernel, kernel],
            bias,
        );
    }

    fn convolution_nd(
        &mut self,
        name: &str,
        output_channels: u64,
        input_channels: u64,
        kernel: &[u64],
        bias: bool,
    ) {
        let mut shape = vec![output_channels, input_channels];
        shape.extend_from_slice(kernel);
        self.state.push(NativeVisionStateSpec {
            name: format!("{name}.weight"),
            shape,
            dtype: self.dtype,
            kind: NativeVisionStateKind::Parameter,
        });
        if bias {
            self.state.push(NativeVisionStateSpec {
                name: format!("{name}.bias"),
                shape: vec![output_channels],
                dtype: self.dtype,
                kind: NativeVisionStateKind::Parameter,
            });
        }
    }

    fn memory_block(&mut self, name: &str, channels: u64) {
        self.convolution(&format!("{name}.conv.0"), channels, channels * 2, 3, true);
        self.convolution(&format!("{name}.conv.2"), channels, channels, 3, true);
        self.convolution(&format!("{name}.conv.4"), channels, channels, 3, true);
    }

    fn parameter(&mut self, name: impl Into<String>, shape: Vec<u64>) {
        self.state_with_dtype(name, shape, self.dtype, NativeVisionStateKind::Parameter);
    }

    fn buffer(&mut self, name: impl Into<String>, shape: Vec<u64>, dtype: DType) {
        self.state_with_dtype(name, shape, dtype, NativeVisionStateKind::Buffer);
    }

    fn state_with_dtype(
        &mut self,
        name: impl Into<String>,
        shape: Vec<u64>,
        dtype: DType,
        kind: NativeVisionStateKind,
    ) {
        self.state.push(NativeVisionStateSpec {
            name: name.into(),
            shape,
            dtype,
            kind,
        });
    }

    fn linear(&mut self, name: &str, output_features: u64, input_features: u64, bias: bool) {
        self.parameter(
            format!("{name}.weight"),
            vec![output_features, input_features],
        );
        if bias {
            self.parameter(format!("{name}.bias"), vec![output_features]);
        }
    }
}

pub fn video_vae_source_state_schema(
    profile: &VaeKernelProfile,
    dtype: DType,
) -> Result<Vec<NativeVisionStateSpec>, VideoVaeError> {
    if matches!(
        profile,
        VaeKernelProfile::HunyuanImageRefinerV1 | VaeKernelProfile::HunyuanVideoRefinerV1
    ) {
        return Ok(hunyuan_refiner_state_schema(
            matches!(profile, VaeKernelProfile::HunyuanVideoRefinerV1),
            dtype,
        ));
    }
    if profile == &VaeKernelProfile::Causal3dV1 {
        return Ok(causal3d_state_schema(dtype));
    }
    if profile == &VaeKernelProfile::CogVideoXV1 {
        return Ok(cogvideox_state_schema(dtype));
    }
    if profile == &VaeKernelProfile::Wan21V1 {
        return Ok(wan21_state_schema(dtype));
    }
    if profile == &VaeKernelProfile::Wan22V1 {
        return Ok(wan22_state_schema(dtype));
    }
    if profile == &VaeKernelProfile::CosmosV1 {
        return Ok(cosmos_state_schema(dtype));
    }
    if profile == &VaeKernelProfile::MochiV1 {
        return Ok(mochi_state_schema(dtype));
    }
    if matches!(
        profile,
        VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
    ) {
        let configuration = ltx_default_configuration(profile)?;
        return ltx_state_schema(profile, &configuration, dtype);
    }
    let topology = taehv_topology(profile)?;
    let patch_channels = 3_u64
        .checked_mul(topology.patch_size)
        .and_then(|channels| channels.checked_mul(topology.patch_size))
        .ok_or_else(|| VideoVaeError::InvalidStateShape {
            name: "encoder.0.weight".to_owned(),
            shape: Vec::new(),
        })?;
    let mut manifest = StateManifest::new(dtype);
    manifest.convolution("encoder.0", 64, patch_channels, 3, true);
    for (stage, (pool_index, convolution_index, memory_indices)) in [
        (2_u64, 3_u64, [4_u64, 5, 6]),
        (7, 8, [9, 10, 11]),
        (12, 13, [14, 15, 16]),
    ]
    .into_iter()
    .enumerate()
    {
        let stride = topology.encoder_temporal_strides[stage];
        manifest.convolution(
            &format!("encoder.{pool_index}.conv"),
            64,
            64 * stride,
            1,
            false,
        );
        manifest.convolution(&format!("encoder.{convolution_index}"), 64, 64, 3, false);
        for index in memory_indices {
            manifest.memory_block(&format!("encoder.{index}"), 64);
        }
    }
    manifest.convolution("encoder.17", topology.source_latent_channels, 64, 3, true);

    manifest.convolution("decoder.1", 256, topology.source_latent_channels, 3, true);
    for index in [3_u64, 4, 5] {
        manifest.memory_block(&format!("decoder.{index}"), 256);
    }
    for (stage, (grow_index, convolution_index, input_channels, output_channels, memory_indices)) in
        [
            (7_u64, 8_u64, 256_u64, 128_u64, [9_u64, 10, 11]),
            (13, 14, 128, 64, [15, 16, 17]),
        ]
        .into_iter()
        .enumerate()
    {
        let stride = topology.decoder_temporal_strides[stage];
        manifest.convolution(
            &format!("decoder.{grow_index}.conv"),
            input_channels * stride,
            input_channels,
            1,
            false,
        );
        manifest.convolution(
            &format!("decoder.{convolution_index}"),
            output_channels,
            input_channels,
            3,
            false,
        );
        for index in memory_indices {
            manifest.memory_block(&format!("decoder.{index}"), output_channels);
        }
    }
    let final_temporal_stride = topology.decoder_temporal_strides[2];
    manifest.convolution("decoder.19.conv", 64 * final_temporal_stride, 64, 1, false);
    manifest.convolution("decoder.20", 64, 64, 3, false);
    manifest.convolution("decoder.22", patch_channels, 64, 3, true);
    Ok(manifest.state)
}

fn video_vae_source_state_schema_for_descriptor(
    descriptor: &VaeDescriptor,
    dtype: DType,
) -> Result<Vec<NativeVisionStateSpec>, VideoVaeError> {
    let profile = descriptor.identity().profile();
    if matches!(
        profile,
        VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
    ) {
        let configuration =
            ltx_configuration(profile, descriptor.identity().loader_configuration())?;
        ltx_state_schema(profile, &configuration, dtype)
    } else {
        video_vae_source_state_schema(profile, dtype)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LtxBlockKind {
    Residual,
    ResidualChangeChannels,
    CompressTime,
    CompressSpace,
    CompressAll,
    CompressAllChangeChannels,
    CompressAllResidual,
    CompressSpaceResidual,
    CompressTimeResidual,
}

#[derive(Clone, Copy, Debug)]
struct LtxBlock {
    kind: LtxBlockKind,
    layers: u64,
    multiplier: u64,
    residual: bool,
    inject_noise: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LtxSpatialPadding {
    Zeros,
    Reflect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LtxNormLayer {
    Group,
    Pixel,
    Layer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LtxLatentLogVariance {
    PerChannel,
    Uniform,
    Constant,
}

#[derive(Clone, Debug)]
struct LtxConfiguration {
    input_channels: u64,
    output_channels: u64,
    latent_channels: u64,
    encoder_base_channels: u64,
    decoder_base_channels: u64,
    patch_size: u64,
    norm_layer: LtxNormLayer,
    latent_log_variance: LtxLatentLogVariance,
    encoder_blocks: Vec<LtxBlock>,
    decoder_blocks: Vec<LtxBlock>,
    causal_decoder: bool,
    timestep_conditioning: bool,
    decode_noise_scale: f32,
    decode_timestep: f32,
    encoder_spatial_padding: LtxSpatialPadding,
    decoder_spatial_padding: LtxSpatialPadding,
}

const fn ltx_block(kind: LtxBlockKind, layers: u64) -> LtxBlock {
    LtxBlock {
        kind,
        layers,
        multiplier: 1,
        residual: false,
        inject_noise: false,
    }
}

fn ltx_encoder_blocks(profile: &VaeKernelProfile) -> Result<Vec<LtxBlock>, VideoVaeError> {
    let blocks = match profile {
        VaeKernelProfile::LtxVideoV0 { .. } | VaeKernelProfile::LtxVideoV1 { .. } => vec![
            ltx_block(LtxBlockKind::Residual, 4),
            ltx_block(LtxBlockKind::CompressAll, 0),
            LtxBlock {
                multiplier: 2,
                ..ltx_block(LtxBlockKind::ResidualChangeChannels, 1)
            },
            ltx_block(LtxBlockKind::Residual, 3),
            ltx_block(LtxBlockKind::CompressAll, 0),
            LtxBlock {
                multiplier: 2,
                ..ltx_block(LtxBlockKind::ResidualChangeChannels, 1)
            },
            ltx_block(LtxBlockKind::Residual, 3),
            ltx_block(LtxBlockKind::CompressAll, 0),
            ltx_block(LtxBlockKind::Residual, 3),
            ltx_block(LtxBlockKind::Residual, 4),
        ],
        VaeKernelProfile::LtxVideoV2 { .. } => vec![
            ltx_block(LtxBlockKind::Residual, 4),
            LtxBlock {
                multiplier: 2,
                ..ltx_block(LtxBlockKind::CompressSpaceResidual, 0)
            },
            ltx_block(LtxBlockKind::Residual, 6),
            LtxBlock {
                multiplier: 2,
                ..ltx_block(LtxBlockKind::CompressTimeResidual, 0)
            },
            ltx_block(LtxBlockKind::Residual, 6),
            LtxBlock {
                multiplier: 2,
                ..ltx_block(LtxBlockKind::CompressAllResidual, 0)
            },
            ltx_block(LtxBlockKind::Residual, 2),
            LtxBlock {
                multiplier: 2,
                ..ltx_block(LtxBlockKind::CompressAllResidual, 0)
            },
            ltx_block(LtxBlockKind::Residual, 2),
        ],
        profile => return Err(VideoVaeError::UnsupportedProfile(profile.clone())),
    };
    Ok(blocks)
}

fn ltx_decoder_blocks(profile: &VaeKernelProfile) -> Result<Vec<LtxBlock>, VideoVaeError> {
    let blocks = match profile {
        VaeKernelProfile::LtxVideoV0 { .. } => ltx_encoder_blocks(profile)?,
        VaeKernelProfile::LtxVideoV1 { .. } => vec![
            LtxBlock {
                inject_noise: true,
                ..ltx_block(LtxBlockKind::Residual, 5)
            },
            LtxBlock {
                multiplier: 2,
                residual: true,
                ..ltx_block(LtxBlockKind::CompressAll, 0)
            },
            LtxBlock {
                inject_noise: true,
                ..ltx_block(LtxBlockKind::Residual, 6)
            },
            LtxBlock {
                multiplier: 2,
                residual: true,
                ..ltx_block(LtxBlockKind::CompressAll, 0)
            },
            LtxBlock {
                inject_noise: true,
                ..ltx_block(LtxBlockKind::Residual, 7)
            },
            LtxBlock {
                multiplier: 2,
                residual: true,
                ..ltx_block(LtxBlockKind::CompressAll, 0)
            },
            ltx_block(LtxBlockKind::Residual, 8),
        ],
        VaeKernelProfile::LtxVideoV2 { .. } => vec![
            ltx_block(LtxBlockKind::Residual, 5),
            LtxBlock {
                multiplier: 2,
                residual: true,
                ..ltx_block(LtxBlockKind::CompressAll, 0)
            },
            ltx_block(LtxBlockKind::Residual, 5),
            LtxBlock {
                multiplier: 2,
                residual: true,
                ..ltx_block(LtxBlockKind::CompressAll, 0)
            },
            ltx_block(LtxBlockKind::Residual, 5),
            LtxBlock {
                multiplier: 2,
                residual: true,
                ..ltx_block(LtxBlockKind::CompressAll, 0)
            },
            ltx_block(LtxBlockKind::Residual, 5),
        ],
        profile => return Err(VideoVaeError::UnsupportedProfile(profile.clone())),
    };
    Ok(blocks)
}

fn ltx_default_configuration(
    profile: &VaeKernelProfile,
) -> Result<LtxConfiguration, VideoVaeError> {
    let configuration = LtxConfiguration {
        input_channels: 3,
        output_channels: 3,
        latent_channels: 128,
        encoder_base_channels: 128,
        decoder_base_channels: 128,
        patch_size: 4,
        norm_layer: LtxNormLayer::Pixel,
        latent_log_variance: LtxLatentLogVariance::Uniform,
        encoder_blocks: ltx_encoder_blocks(profile)?,
        decoder_blocks: ltx_decoder_blocks(profile)?,
        causal_decoder: false,
        timestep_conditioning: !matches!(profile, VaeKernelProfile::LtxVideoV0 { .. }),
        decode_noise_scale: 0.025,
        decode_timestep: 0.05,
        encoder_spatial_padding: LtxSpatialPadding::Zeros,
        decoder_spatial_padding: LtxSpatialPadding::Reflect,
    };
    ltx_validate_normalization_channels(&configuration)?;
    Ok(configuration)
}

fn ltx_configuration(
    profile: &VaeKernelProfile,
    loader_configuration: &VaeLoaderConfiguration,
) -> Result<LtxConfiguration, VideoVaeError> {
    let VaeLoaderConfiguration::LtxVideo {
        configuration_json, ..
    } = loader_configuration
    else {
        return Err(VideoVaeError::InvalidLtxConfiguration(
            "LTX profiles require the digest-bound LtxVideo loader configuration".to_owned(),
        ));
    };
    let Some(configuration_json) = configuration_json else {
        return ltx_default_configuration(profile);
    };
    let value = serde_json::from_str::<serde_json::Value>(configuration_json)
        .map_err(|error| VideoVaeError::InvalidLtxConfiguration(error.to_string()))?;
    let object = value.as_object().ok_or_else(|| {
        VideoVaeError::InvalidLtxConfiguration("configuration must be a JSON object".to_owned())
    })?;
    if object.get("dims").and_then(serde_json::Value::as_u64) != Some(3) {
        return Err(VideoVaeError::InvalidLtxConfiguration(
            "only the source 3D video topology is valid".to_owned(),
        ));
    }
    let input_channels = ltx_optional_positive_u64(object, "in_channels", 3)?;
    let output_channels = ltx_optional_positive_u64(object, "out_channels", 3)?;
    let latent_channels = ltx_required_positive_u64(object, "latent_channels")?;
    let encoder_base_channels = ltx_optional_positive_u64(object, "encoder_base_channels", 128)?;
    let decoder_base_channels = ltx_optional_positive_u64(object, "decoder_base_channels", 128)?;
    let patch_size = ltx_optional_positive_u64(object, "patch_size", 1)?;
    if !patch_size.is_power_of_two() {
        return Err(VideoVaeError::InvalidLtxConfiguration(
            "patch_size must be a positive power of two".to_owned(),
        ));
    }
    let norm_layer = object
        .get("norm_layer")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("group_norm");
    let norm_layer = match norm_layer {
        "group_norm" => LtxNormLayer::Group,
        "pixel_norm" => LtxNormLayer::Pixel,
        "layer_norm" => LtxNormLayer::Layer,
        other => {
            return Err(VideoVaeError::InvalidLtxConfiguration(format!(
                "unknown norm_layer {other:?}"
            )));
        }
    };
    let double_latent = object
        .get("double_z")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                VideoVaeError::InvalidLtxConfiguration("double_z must be boolean".to_owned())
            })
        })
        .transpose()?
        .unwrap_or(true);
    let latent_log_variance = object
        .get("latent_log_var")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(if double_latent { "per_channel" } else { "none" });
    let latent_log_variance = match latent_log_variance {
        "per_channel" => LtxLatentLogVariance::PerChannel,
        "uniform" => LtxLatentLogVariance::Uniform,
        "constant" => LtxLatentLogVariance::Constant,
        "none" => {
            return Err(VideoVaeError::InvalidLtxConfiguration(
                "latent_log_var=none is incompatible with the source VideoVAE encode chunk and per-channel statistics contract"
                    .to_owned(),
            ));
        }
        other => {
            return Err(VideoVaeError::InvalidLtxConfiguration(format!(
                "unknown latent_log_var {other:?}"
            )));
        }
    };
    let shared_blocks = object.get("blocks");
    let encoder_blocks = ltx_parse_blocks(
        object.get("encoder_blocks").or(shared_blocks),
        "encoder_blocks",
        true,
    )?;
    let decoder_blocks = ltx_parse_blocks(
        object.get("decoder_blocks").or(shared_blocks),
        "decoder_blocks",
        false,
    )?;
    ltx_validate_ratios(&encoder_blocks, &decoder_blocks, patch_size)?;
    let causal_decoder = ltx_optional_bool(object, "causal_decoder", false)?;
    let timestep_conditioning = ltx_optional_bool(object, "timestep_conditioning", false)?;
    let decode_noise_scale = ltx_optional_f32(object, "decode_noise_scale", 0.025)?;
    let decode_timestep = ltx_optional_f32(object, "decode_timestep", 0.05)?;
    let (encoder_spatial_padding, decoder_spatial_padding) =
        if let Some(mode) = object.get("spatial_padding_mode") {
            let mode = ltx_spatial_padding(mode)?;
            (mode, mode)
        } else {
            (LtxSpatialPadding::Zeros, LtxSpatialPadding::Reflect)
        };
    let configuration = LtxConfiguration {
        input_channels,
        output_channels,
        latent_channels,
        encoder_base_channels,
        decoder_base_channels,
        patch_size,
        norm_layer,
        latent_log_variance,
        encoder_blocks,
        decoder_blocks,
        causal_decoder,
        timestep_conditioning,
        decode_noise_scale,
        decode_timestep,
        encoder_spatial_padding,
        decoder_spatial_padding,
    };
    ltx_validate_normalization_channels(&configuration)?;
    Ok(configuration)
}

fn ltx_required_positive_u64(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
) -> Result<u64, VideoVaeError> {
    object
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            VideoVaeError::InvalidLtxConfiguration(format!(
                "{field} must be a positive unsigned integer"
            ))
        })
}

fn ltx_optional_positive_u64(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    default: u64,
) -> Result<u64, VideoVaeError> {
    object
        .get(field)
        .map(|_| ltx_required_positive_u64(object, field))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn ltx_optional_bool(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    default: bool,
) -> Result<bool, VideoVaeError> {
    object
        .get(field)
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                VideoVaeError::InvalidLtxConfiguration(format!("{field} must be boolean"))
            })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn ltx_optional_f32(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    default: f32,
) -> Result<f32, VideoVaeError> {
    let value = object
        .get(field)
        .map(|value| {
            value.as_f64().ok_or_else(|| {
                VideoVaeError::InvalidLtxConfiguration(format!("{field} must be numeric"))
            })
        })
        .transpose()?
        .unwrap_or(f64::from(default));
    let value = value as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(VideoVaeError::InvalidLtxConfiguration(format!(
            "{field} must be finite"
        )))
    }
}

fn ltx_spatial_padding(value: &serde_json::Value) -> Result<LtxSpatialPadding, VideoVaeError> {
    match value.as_str() {
        Some("zeros") => Ok(LtxSpatialPadding::Zeros),
        Some("reflect") => Ok(LtxSpatialPadding::Reflect),
        _ => Err(VideoVaeError::InvalidLtxConfiguration(
            "spatial_padding_mode must be zeros or reflect".to_owned(),
        )),
    }
}

fn ltx_parse_blocks(
    value: Option<&serde_json::Value>,
    field: &'static str,
    encoder: bool,
) -> Result<Vec<LtxBlock>, VideoVaeError> {
    let rows = value.and_then(serde_json::Value::as_array).ok_or_else(|| {
        VideoVaeError::InvalidLtxConfiguration(format!("{field} must be an array"))
    })?;
    if rows.is_empty() || rows.len() > 128 {
        return Err(VideoVaeError::InvalidLtxConfiguration(format!(
            "{field} must contain 1..=128 blocks"
        )));
    }
    rows.iter()
        .enumerate()
        .map(|(index, row)| ltx_parse_block(row, field, index, encoder))
        .collect()
}

fn ltx_parse_block(
    row: &serde_json::Value,
    field: &'static str,
    index: usize,
    encoder: bool,
) -> Result<LtxBlock, VideoVaeError> {
    let pair = row
        .as_array()
        .filter(|pair| pair.len() == 2)
        .ok_or_else(|| {
            VideoVaeError::InvalidLtxConfiguration(format!(
                "{field}[{index}] must be a two-element array"
            ))
        })?;
    let name = pair[0].as_str().ok_or_else(|| {
        VideoVaeError::InvalidLtxConfiguration(format!("{field}[{index}][0] must be a block name"))
    })?;
    let parameters = pair[1].as_object();
    let integer_layers = pair[1].as_u64();
    if parameters.is_none() && integer_layers.is_none() {
        return Err(VideoVaeError::InvalidLtxConfiguration(format!(
            "{field}[{index}][1] must be an unsigned integer or object"
        )));
    }
    let parameter_u64 = |parameter: &'static str, default: u64| {
        parameters
            .and_then(|parameters| parameters.get(parameter))
            .map(|value| {
                value.as_u64().filter(|value| *value > 0).ok_or_else(|| {
                    VideoVaeError::InvalidLtxConfiguration(format!(
                        "{field}[{index}].{parameter} must be positive"
                    ))
                })
            })
            .transpose()
            .map(|value| value.unwrap_or(default))
    };
    let parameter_bool = |parameter: &'static str, default: bool| {
        parameters
            .and_then(|parameters| parameters.get(parameter))
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    VideoVaeError::InvalidLtxConfiguration(format!(
                        "{field}[{index}].{parameter} must be boolean"
                    ))
                })
            })
            .transpose()
            .map(|value| value.unwrap_or(default))
    };
    let kind = match name {
        "res_x" => LtxBlockKind::Residual,
        "res_x_y" => LtxBlockKind::ResidualChangeChannels,
        "compress_time" => LtxBlockKind::CompressTime,
        "compress_space" => LtxBlockKind::CompressSpace,
        "compress_all" => LtxBlockKind::CompressAll,
        "compress_all_x_y" if encoder => LtxBlockKind::CompressAllChangeChannels,
        "compress_all_res" if encoder => LtxBlockKind::CompressAllResidual,
        "compress_space_res" if encoder => LtxBlockKind::CompressSpaceResidual,
        "compress_time_res" if encoder => LtxBlockKind::CompressTimeResidual,
        _ => {
            return Err(VideoVaeError::InvalidLtxConfiguration(format!(
                "{field}[{index}] has unsupported source block {name:?}"
            )));
        }
    };
    let layers = if kind == LtxBlockKind::Residual {
        integer_layers.unwrap_or(parameter_u64("num_layers", 0)?)
    } else {
        integer_layers.unwrap_or(0)
    };
    if kind == LtxBlockKind::Residual && !(1..=128).contains(&layers) {
        return Err(VideoVaeError::InvalidLtxConfiguration(format!(
            "{field}[{index}] residual num_layers must be 1..=128"
        )));
    }
    let default_multiplier = match kind {
        LtxBlockKind::ResidualChangeChannels
        | LtxBlockKind::CompressAllChangeChannels
        | LtxBlockKind::CompressAllResidual
        | LtxBlockKind::CompressSpaceResidual
        | LtxBlockKind::CompressTimeResidual => 2,
        _ => 1,
    };
    Ok(LtxBlock {
        kind,
        layers,
        multiplier: parameter_u64("multiplier", default_multiplier)?,
        residual: parameter_bool("residual", false)?,
        inject_noise: parameter_bool("inject_noise", false)?,
    })
}

fn ltx_block_scale(block: LtxBlock) -> (u64, u64) {
    match block.kind {
        LtxBlockKind::CompressTime | LtxBlockKind::CompressTimeResidual => (2, 1),
        LtxBlockKind::CompressSpace | LtxBlockKind::CompressSpaceResidual => (1, 2),
        LtxBlockKind::CompressAll
        | LtxBlockKind::CompressAllChangeChannels
        | LtxBlockKind::CompressAllResidual => (2, 2),
        LtxBlockKind::Residual | LtxBlockKind::ResidualChangeChannels => (1, 1),
    }
}

fn ltx_validate_ratios(
    encoder_blocks: &[LtxBlock],
    decoder_blocks: &[LtxBlock],
    patch_size: u64,
) -> Result<(), VideoVaeError> {
    let ratio = |blocks: &[LtxBlock]| {
        blocks.iter().try_fold((1_u64, patch_size), |ratio, block| {
            let scale = ltx_block_scale(*block);
            Some((ratio.0.checked_mul(scale.0)?, ratio.1.checked_mul(scale.1)?))
        })
    };
    let encoder = ratio(encoder_blocks).ok_or_else(|| {
        VideoVaeError::InvalidLtxConfiguration("encoder scale ratio overflows".to_owned())
    })?;
    let decoder = ratio(decoder_blocks).ok_or_else(|| {
        VideoVaeError::InvalidLtxConfiguration("decoder scale ratio overflows".to_owned())
    })?;
    if encoder != (8, 32) || decoder != encoder {
        return Err(VideoVaeError::InvalidLtxConfiguration(format!(
            "source wrapper requires matching temporal/spatial ratios 8/32, got encoder {encoder:?} and decoder {decoder:?}"
        )));
    }
    Ok(())
}

fn ltx_validate_normalization_channels(
    configuration: &LtxConfiguration,
) -> Result<(), VideoVaeError> {
    if configuration.norm_layer != LtxNormLayer::Group {
        return Ok(());
    }
    let validate = |channels: u64| {
        if channels.is_multiple_of(32) {
            Ok(())
        } else {
            Err(VideoVaeError::InvalidLtxConfiguration(format!(
                "group_norm channel count {channels} is not divisible by 32"
            )))
        }
    };
    let mut channels = configuration.encoder_base_channels;
    validate(channels)?;
    for block in &configuration.encoder_blocks {
        if matches!(
            block.kind,
            LtxBlockKind::ResidualChangeChannels
                | LtxBlockKind::CompressAllChangeChannels
                | LtxBlockKind::CompressAllResidual
                | LtxBlockKind::CompressSpaceResidual
                | LtxBlockKind::CompressTimeResidual
        ) {
            channels = channels.checked_mul(block.multiplier).ok_or_else(|| {
                VideoVaeError::InvalidLtxConfiguration(
                    "encoder normalization channels overflow".to_owned(),
                )
            })?;
            validate(channels)?;
        }
    }
    channels = configuration.decoder_base_channels;
    validate(channels)?;
    for block in configuration.decoder_blocks.iter().rev() {
        if matches!(
            block.kind,
            LtxBlockKind::ResidualChangeChannels
                | LtxBlockKind::CompressTime
                | LtxBlockKind::CompressSpace
                | LtxBlockKind::CompressAll
        ) {
            channels = channels.checked_mul(block.multiplier).ok_or_else(|| {
                VideoVaeError::InvalidLtxConfiguration(
                    "decoder normalization channels overflow".to_owned(),
                )
            })?;
            validate(channels)?;
        }
    }
    Ok(())
}

fn ltx_causal_convolution_manifest(
    manifest: &mut StateManifest,
    prefix: &str,
    output_channels: u64,
    input_channels: u64,
) {
    manifest.convolution_nd(
        &format!("{prefix}.conv"),
        output_channels,
        input_channels,
        &[3, 3, 3],
        true,
    );
}

fn ltx_timestep_embedding_manifest(
    manifest: &mut StateManifest,
    prefix: &str,
    embedding_channels: u64,
) {
    manifest.linear(
        &format!("{prefix}.timestep_embedder.linear_1"),
        embedding_channels,
        256,
        true,
    );
    manifest.linear(
        &format!("{prefix}.timestep_embedder.linear_2"),
        embedding_channels,
        embedding_channels,
        true,
    );
}

fn ltx_residual_manifest(
    manifest: &mut StateManifest,
    prefix: &str,
    input_channels: u64,
    output_channels: u64,
    inject_noise: bool,
    timestep_conditioning: bool,
    norm_layer: LtxNormLayer,
) {
    ltx_normalization_manifest(
        manifest,
        &format!("{prefix}.norm1"),
        input_channels,
        norm_layer,
    );
    ltx_normalization_manifest(
        manifest,
        &format!("{prefix}.norm2"),
        output_channels,
        norm_layer,
    );
    ltx_causal_convolution_manifest(
        manifest,
        &format!("{prefix}.conv1"),
        output_channels,
        input_channels,
    );
    ltx_causal_convolution_manifest(
        manifest,
        &format!("{prefix}.conv2"),
        output_channels,
        output_channels,
    );
    if input_channels != output_channels {
        manifest.convolution_nd(
            &format!("{prefix}.conv_shortcut"),
            output_channels,
            input_channels,
            &[1, 1, 1],
            true,
        );
        manifest.parameter(format!("{prefix}.norm3.norm.weight"), vec![input_channels]);
        manifest.parameter(format!("{prefix}.norm3.norm.bias"), vec![input_channels]);
    }
    if inject_noise {
        manifest.parameter(
            format!("{prefix}.per_channel_scale1"),
            vec![input_channels, 1, 1],
        );
        manifest.parameter(
            format!("{prefix}.per_channel_scale2"),
            vec![input_channels, 1, 1],
        );
    }
    if timestep_conditioning {
        manifest.parameter(
            format!("{prefix}.scale_shift_table"),
            vec![4, input_channels],
        );
    }
}

fn ltx_mid_block_manifest(
    manifest: &mut StateManifest,
    prefix: &str,
    channels: u64,
    block: LtxBlock,
    timestep_conditioning: bool,
    norm_layer: LtxNormLayer,
) {
    if timestep_conditioning {
        ltx_timestep_embedding_manifest(manifest, &format!("{prefix}.time_embedder"), channels * 4);
    }
    for layer in 0..block.layers {
        ltx_residual_manifest(
            manifest,
            &format!("{prefix}.res_blocks.{layer}"),
            channels,
            channels,
            block.inject_noise,
            timestep_conditioning,
            norm_layer,
        );
    }
}

fn ltx_normalization_manifest(
    manifest: &mut StateManifest,
    prefix: &str,
    channels: u64,
    norm_layer: LtxNormLayer,
) {
    let prefix = if norm_layer == LtxNormLayer::Layer {
        format!("{prefix}.norm")
    } else {
        prefix.to_owned()
    };
    if norm_layer != LtxNormLayer::Pixel {
        manifest.parameter(format!("{prefix}.weight"), vec![channels]);
        manifest.parameter(format!("{prefix}.bias"), vec![channels]);
    }
}

fn ltx_state_schema(
    profile: &VaeKernelProfile,
    configuration: &LtxConfiguration,
    dtype: DType,
) -> Result<Vec<NativeVisionStateSpec>, VideoVaeError> {
    let mut manifest = StateManifest::new(dtype);
    let patch_channels = configuration
        .input_channels
        .checked_mul(configuration.patch_size)
        .and_then(|channels| channels.checked_mul(configuration.patch_size))
        .ok_or_else(|| VideoVaeError::InvalidStateShape {
            name: "encoder.conv_in.conv.weight".to_owned(),
            shape: Vec::new(),
        })?;
    ltx_causal_convolution_manifest(
        &mut manifest,
        "encoder.conv_in",
        configuration.encoder_base_channels,
        patch_channels,
    );
    let mut channels = configuration.encoder_base_channels;
    for (index, block) in configuration.encoder_blocks.iter().copied().enumerate() {
        let prefix = format!("encoder.down_blocks.{index}");
        match block.kind {
            LtxBlockKind::Residual => {
                ltx_mid_block_manifest(
                    &mut manifest,
                    &prefix,
                    channels,
                    block,
                    false,
                    configuration.norm_layer,
                );
            }
            LtxBlockKind::ResidualChangeChannels => {
                let output_channels = channels.checked_mul(block.multiplier).ok_or_else(|| {
                    VideoVaeError::InvalidStateShape {
                        name: prefix.clone(),
                        shape: vec![channels],
                    }
                })?;
                ltx_residual_manifest(
                    &mut manifest,
                    &prefix,
                    channels,
                    output_channels,
                    false,
                    false,
                    configuration.norm_layer,
                );
                channels = output_channels;
            }
            LtxBlockKind::CompressTime
            | LtxBlockKind::CompressSpace
            | LtxBlockKind::CompressAll => {
                ltx_causal_convolution_manifest(&mut manifest, &prefix, channels, channels);
            }
            LtxBlockKind::CompressAllChangeChannels => {
                let output_channels = channels.checked_mul(block.multiplier).ok_or_else(|| {
                    VideoVaeError::InvalidStateShape {
                        name: prefix.clone(),
                        shape: vec![channels],
                    }
                })?;
                ltx_causal_convolution_manifest(&mut manifest, &prefix, output_channels, channels);
                channels = output_channels;
            }
            LtxBlockKind::CompressAllResidual
            | LtxBlockKind::CompressSpaceResidual
            | LtxBlockKind::CompressTimeResidual => {
                let output_channels = channels.checked_mul(block.multiplier).ok_or_else(|| {
                    VideoVaeError::InvalidStateShape {
                        name: prefix.clone(),
                        shape: vec![channels],
                    }
                })?;
                let packed_channels = match block.kind {
                    LtxBlockKind::CompressAllResidual => output_channels / 8,
                    LtxBlockKind::CompressSpaceResidual => output_channels / 4,
                    LtxBlockKind::CompressTimeResidual => output_channels / 2,
                    _ => output_channels,
                };
                ltx_causal_convolution_manifest(
                    &mut manifest,
                    &format!("{prefix}.conv"),
                    packed_channels,
                    channels,
                );
                channels = output_channels;
            }
        }
    }
    ltx_normalization_manifest(
        &mut manifest,
        "encoder.conv_norm_out",
        channels,
        configuration.norm_layer,
    );
    let encoder_output_channels = match configuration.latent_log_variance {
        LtxLatentLogVariance::PerChannel => configuration.latent_channels.checked_mul(2),
        LtxLatentLogVariance::Uniform | LtxLatentLogVariance::Constant => {
            configuration.latent_channels.checked_add(1)
        }
    }
    .ok_or_else(|| VideoVaeError::InvalidStateShape {
        name: "encoder.conv_out.conv.weight".to_owned(),
        shape: vec![configuration.latent_channels],
    })?;
    ltx_causal_convolution_manifest(
        &mut manifest,
        "encoder.conv_out",
        encoder_output_channels,
        channels,
    );

    let decoder_blocks = &configuration.decoder_blocks;
    channels = configuration.decoder_base_channels;
    for block in decoder_blocks.iter().rev() {
        if matches!(
            block.kind,
            LtxBlockKind::ResidualChangeChannels
                | LtxBlockKind::CompressTime
                | LtxBlockKind::CompressSpace
                | LtxBlockKind::CompressAll
        ) {
            channels = channels.checked_mul(block.multiplier).ok_or_else(|| {
                VideoVaeError::InvalidStateShape {
                    name: "decoder.conv_in".to_owned(),
                    shape: vec![channels],
                }
            })?;
        }
    }
    ltx_causal_convolution_manifest(
        &mut manifest,
        "decoder.conv_in",
        channels,
        configuration.latent_channels,
    );
    let timestep_conditioning = configuration.timestep_conditioning;
    for (index, block) in decoder_blocks.iter().copied().rev().enumerate() {
        let prefix = format!("decoder.up_blocks.{index}");
        match block.kind {
            LtxBlockKind::Residual => {
                ltx_mid_block_manifest(
                    &mut manifest,
                    &prefix,
                    channels,
                    block,
                    timestep_conditioning,
                    configuration.norm_layer,
                );
            }
            LtxBlockKind::ResidualChangeChannels => {
                let output_channels = channels / block.multiplier;
                ltx_residual_manifest(
                    &mut manifest,
                    &prefix,
                    channels,
                    output_channels,
                    block.inject_noise,
                    false,
                    configuration.norm_layer,
                );
                channels = output_channels;
            }
            LtxBlockKind::CompressTime
            | LtxBlockKind::CompressSpace
            | LtxBlockKind::CompressAll => {
                let output_channels = channels / block.multiplier;
                let scale = ltx_block_scale(block);
                let packed = scale
                    .0
                    .checked_mul(scale.1)
                    .and_then(|factor| factor.checked_mul(scale.1))
                    .and_then(|factor| channels.checked_mul(factor))
                    .and_then(|packed| packed.checked_div(block.multiplier))
                    .ok_or_else(|| VideoVaeError::InvalidStateShape {
                        name: prefix.clone(),
                        shape: vec![channels],
                    })?;
                ltx_causal_convolution_manifest(
                    &mut manifest,
                    &format!("{prefix}.conv"),
                    packed,
                    channels,
                );
                channels = output_channels;
            }
            _ => return Err(VideoVaeError::UnsupportedProfile(profile.clone())),
        }
    }
    ltx_normalization_manifest(
        &mut manifest,
        "decoder.conv_norm_out",
        channels,
        configuration.norm_layer,
    );
    let output_patch_channels = configuration
        .output_channels
        .checked_mul(configuration.patch_size)
        .and_then(|channels| channels.checked_mul(configuration.patch_size))
        .ok_or_else(|| VideoVaeError::InvalidStateShape {
            name: "decoder.conv_out.conv.weight".to_owned(),
            shape: vec![configuration.output_channels],
        })?;
    ltx_causal_convolution_manifest(
        &mut manifest,
        "decoder.conv_out",
        output_patch_channels,
        channels,
    );
    if timestep_conditioning {
        manifest.state_with_dtype(
            "decoder.timestep_scale_multiplier",
            Vec::new(),
            DType::F32,
            NativeVisionStateKind::Parameter,
        );
        ltx_timestep_embedding_manifest(&mut manifest, "decoder.last_time_embedder", channels * 2);
        manifest.parameter("decoder.last_scale_shift_table", vec![2, channels]);
    }
    manifest.buffer(
        "per_channel_statistics.std-of-means",
        vec![configuration.latent_channels],
        DType::F32,
    );
    manifest.buffer(
        "per_channel_statistics.mean-of-means",
        vec![configuration.latent_channels],
        DType::F32,
    );
    Ok(manifest.state)
}

fn mochi_state_schema(dtype: DType) -> Vec<NativeVisionStateSpec> {
    let mut manifest = StateManifest::new(dtype);
    manifest.linear("encoder.layers.0", 64, 15, true);
    for layer in 1..=3 {
        mochi_residual_manifest(&mut manifest, &format!("encoder.layers.{layer}"), 64, false);
    }
    for (layer, input_channels, output_channels, blocks, temporal_stride) in
        [(4, 64, 128, 3, 1), (5, 128, 256, 4, 2), (6, 256, 384, 6, 3)]
    {
        manifest.convolution_nd(
            &format!("encoder.layers.{layer}.layers.0"),
            output_channels,
            input_channels,
            &[temporal_stride, 2, 2],
            true,
        );
        for block in 0..blocks {
            mochi_residual_manifest(
                &mut manifest,
                &format!("encoder.layers.{layer}.layers.{}", block + 1),
                output_channels,
                true,
            );
        }
    }
    for layer in 7..=9 {
        mochi_residual_manifest(&mut manifest, &format!("encoder.layers.{layer}"), 384, true);
    }
    mochi_normalization_manifest(&mut manifest, "encoder.output_norm", 384);
    manifest.linear("encoder.output_proj", 24, 384, false);

    manifest.convolution_nd("decoder.blocks.0.0", 768, 12, &[1, 1, 1], true);
    for block in 1..=3 {
        mochi_residual_manifest(
            &mut manifest,
            &format!("decoder.blocks.0.{block}"),
            768,
            false,
        );
    }
    for (block, input_channels, output_channels, residual_blocks, temporal_expansion) in [
        (1, 768, 512, 6, 3),
        (2, 512, 256, 4, 2),
        (3, 256, 128, 3, 1),
    ] {
        for residual in 0..residual_blocks {
            mochi_residual_manifest(
                &mut manifest,
                &format!("decoder.blocks.{block}.blocks.{residual}"),
                input_channels,
                false,
            );
        }
        manifest.linear(
            &format!("decoder.blocks.{block}.proj"),
            output_channels * temporal_expansion * 4,
            input_channels,
            true,
        );
    }
    for residual in 0..3 {
        mochi_residual_manifest(
            &mut manifest,
            &format!("decoder.blocks.4.{residual}"),
            128,
            false,
        );
    }
    manifest.linear("decoder.output_proj", 3, 128, true);
    manifest.state
}

fn mochi_state_projection(
    model: &LoadedModel,
    dtype: DType,
) -> Result<Vec<(String, NativeVisionStateSpec)>, VideoVaeError> {
    let tensors = model.tensors();
    let raw_decoder = tensors.contains_key("blocks.2.blocks.3.stack.5.weight");
    let raw_encoder = tensors.contains_key("layers.4.layers.1.attn_block.attn.qkv.weight");
    if raw_decoder && raw_encoder {
        return Err(VideoVaeError::InvalidMochiCheckpointLayout(
            "Mochi checkpoint cannot combine two unprefixed state namespaces".to_owned(),
        ));
    }
    let prefixed_decoder = tensors.contains_key("decoder.blocks.2.blocks.3.stack.5.weight");
    let prefixed_encoder =
        tensors.contains_key("encoder.layers.4.layers.1.attn_block.attn.qkv.weight");
    if !(raw_decoder || raw_encoder || prefixed_decoder || prefixed_encoder) {
        return Err(VideoVaeError::MissingState(
            "complete Mochi encoder or decoder sentinel".to_owned(),
        ));
    }
    Ok(mochi_schema_projection(
        dtype,
        raw_decoder,
        raw_encoder,
        prefixed_decoder,
        prefixed_encoder,
    ))
}

fn mochi_schema_projection(
    dtype: DType,
    raw_decoder: bool,
    raw_encoder: bool,
    prefixed_decoder: bool,
    prefixed_encoder: bool,
) -> Vec<(String, NativeVisionStateSpec)> {
    let mut projection = Vec::new();
    for state in mochi_state_schema(dtype) {
        let (present, source) = if let Some(suffix) = state.name.strip_prefix("decoder.") {
            (
                raw_decoder || prefixed_decoder,
                if raw_decoder {
                    suffix.to_owned()
                } else {
                    state.name.clone()
                },
            )
        } else if let Some(suffix) = state.name.strip_prefix("encoder.") {
            (
                raw_encoder || prefixed_encoder,
                if raw_encoder {
                    suffix.to_owned()
                } else {
                    state.name.clone()
                },
            )
        } else {
            (false, state.name.clone())
        };
        if present {
            projection.push((source, state));
        }
    }
    projection
}

fn inspect_mochi_architecture(
    model: &LoadedModel,
    plan: NativeVideoVaeArchitecture,
) -> Result<NativeVideoVaeArchitecture, VideoVaeError> {
    let sentinel = [
        "decoder.blocks.2.blocks.3.stack.5.weight",
        "blocks.2.blocks.3.stack.5.weight",
        "encoder.layers.4.layers.1.attn_block.attn.qkv.weight",
        "layers.4.layers.1.attn_block.attn.qkv.weight",
    ]
    .iter()
    .find_map(|name| model.tensors().get(*name).map(|metadata| (*name, metadata)))
    .ok_or_else(|| VideoVaeError::MissingState("Mochi dtype sentinel".to_owned()))?;
    let dtype = canonical_vision_model_store_dtype(&sentinel.1.data_type).ok_or_else(|| {
        VideoVaeError::UnsupportedStorageDType {
            name: sentinel.0.to_owned(),
            dtype: sentinel.1.data_type.clone(),
        }
    })?;
    let projection = mochi_state_projection(model, dtype)?;
    let expected = projection
        .iter()
        .map(|(source, _)| source.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(unexpected) = model
        .tensors()
        .keys()
        .find(|name| !expected.contains(name.as_str()))
    {
        return Err(VideoVaeError::UnexpectedState(unexpected.clone()));
    }
    for (source, state) in projection {
        let metadata = model
            .tensors()
            .get(&source)
            .ok_or_else(|| VideoVaeError::MissingState(source.clone()))?;
        if metadata.shape != state.shape {
            return Err(VideoVaeError::InvalidStateShape {
                name: source,
                shape: metadata.shape.clone(),
            });
        }
        let actual_dtype =
            canonical_vision_model_store_dtype(&metadata.data_type).ok_or_else(|| {
                VideoVaeError::UnsupportedStorageDType {
                    name: source.clone(),
                    dtype: metadata.data_type.clone(),
                }
            })?;
        if actual_dtype != state.dtype {
            return Err(VideoVaeError::MixedStorageDType {
                name: source,
                expected: state.dtype,
                actual: actual_dtype,
            });
        }
    }
    Ok(NativeVideoVaeArchitecture {
        storage_dtype: Some(dtype),
        ..plan
    })
}

fn mochi_normalization_manifest(manifest: &mut StateManifest, name: &str, channels: u64) {
    manifest.parameter(format!("{name}.weight"), vec![channels]);
    manifest.parameter(format!("{name}.bias"), vec![channels]);
}

fn mochi_residual_manifest(
    manifest: &mut StateManifest,
    name: &str,
    channels: u64,
    attention: bool,
) {
    mochi_normalization_manifest(manifest, &format!("{name}.stack.0"), channels);
    manifest.convolution_nd(
        &format!("{name}.stack.2"),
        channels,
        channels,
        &[3, 3, 3],
        true,
    );
    mochi_normalization_manifest(manifest, &format!("{name}.stack.3"), channels);
    manifest.convolution_nd(
        &format!("{name}.stack.5"),
        channels,
        channels,
        &[3, 3, 3],
        true,
    );
    if attention {
        mochi_normalization_manifest(manifest, &format!("{name}.attn_block.norm"), channels);
        manifest.linear(
            &format!("{name}.attn_block.attn.qkv"),
            channels * 3,
            channels,
            false,
        );
        manifest.linear(
            &format!("{name}.attn_block.attn.out"),
            channels,
            channels,
            true,
        );
    }
}

fn cosmos_state_schema(dtype: DType) -> Vec<NativeVisionStateSpec> {
    const CHANNELS: u64 = 128;
    const LEVEL_CHANNELS: [u64; 3] = [256, 512, 512];
    let mut manifest = StateManifest::new(dtype);
    cosmos_convolution(
        &mut manifest,
        "encoder.conv_in.0",
        CHANNELS,
        192,
        &[1, 3, 3],
    );
    cosmos_convolution(
        &mut manifest,
        "encoder.conv_in.1",
        CHANNELS,
        CHANNELS,
        &[3, 1, 1],
    );
    let mut input_channels = CHANNELS;
    for (level, output_channels) in LEVEL_CHANNELS.into_iter().enumerate() {
        for block in 0..2 {
            cosmos_residual_manifest(
                &mut manifest,
                &format!("encoder.down.{level}.block.{block}"),
                input_channels,
                output_channels,
            );
            input_channels = output_channels;
        }
        if level == 0 {
            cosmos_hybrid_resample_manifest(
                &mut manifest,
                "encoder.down.0.downsample",
                output_channels,
            );
        }
    }
    cosmos_residual_manifest(&mut manifest, "encoder.mid.block_1", 512, 512);
    cosmos_attention_pair_manifest(&mut manifest, "encoder.mid.attn_1", 512);
    cosmos_residual_manifest(&mut manifest, "encoder.mid.block_2", 512, 512);
    cosmos_normalization_manifest(&mut manifest, "encoder.norm_out", 512);
    cosmos_convolution(&mut manifest, "encoder.conv_out.0", 16, 512, &[1, 3, 3]);
    cosmos_convolution(&mut manifest, "encoder.conv_out.1", 16, 16, &[3, 1, 1]);
    cosmos_convolution(&mut manifest, "quant_conv", 16, 16, &[1, 1, 1]);

    cosmos_convolution(&mut manifest, "post_quant_conv", 16, 16, &[1, 1, 1]);
    cosmos_convolution(&mut manifest, "decoder.conv_in.0", 512, 16, &[1, 3, 3]);
    cosmos_convolution(&mut manifest, "decoder.conv_in.1", 512, 512, &[3, 1, 1]);
    cosmos_residual_manifest(&mut manifest, "decoder.mid.block_1", 512, 512);
    cosmos_attention_pair_manifest(&mut manifest, "decoder.mid.attn_1", 512);
    cosmos_residual_manifest(&mut manifest, "decoder.mid.block_2", 512, 512);
    input_channels = 512;
    for level in (0..3).rev() {
        let output_channels = LEVEL_CHANNELS[level];
        for block in 0..3 {
            cosmos_residual_manifest(
                &mut manifest,
                &format!("decoder.up.{level}.block.{block}"),
                input_channels,
                output_channels,
            );
            input_channels = output_channels;
        }
        if level == 1 {
            cosmos_hybrid_resample_manifest(
                &mut manifest,
                "decoder.up.1.upsample",
                output_channels,
            );
        }
    }
    cosmos_normalization_manifest(&mut manifest, "decoder.norm_out", 256);
    cosmos_convolution(&mut manifest, "decoder.conv_out.0", 192, 256, &[1, 3, 3]);
    cosmos_convolution(&mut manifest, "decoder.conv_out.1", 192, 192, &[3, 1, 1]);
    manifest.parameter("latent_mean", vec![256]);
    manifest.parameter("latent_std", vec![256]);
    manifest.state
}

fn cosmos_convolution(
    manifest: &mut StateManifest,
    name: &str,
    output_channels: u64,
    input_channels: u64,
    kernel: &[u64],
) {
    manifest.convolution_nd(
        &format!("{name}.conv3d"),
        output_channels,
        input_channels,
        kernel,
        true,
    );
}

fn cosmos_normalization_manifest(manifest: &mut StateManifest, name: &str, channels: u64) {
    manifest.parameter(format!("{name}.norm.weight"), vec![channels]);
    manifest.parameter(format!("{name}.norm.bias"), vec![channels]);
}

fn cosmos_residual_manifest(
    manifest: &mut StateManifest,
    name: &str,
    input_channels: u64,
    output_channels: u64,
) {
    cosmos_normalization_manifest(manifest, &format!("{name}.norm1"), input_channels);
    cosmos_convolution(
        manifest,
        &format!("{name}.conv1.0"),
        output_channels,
        input_channels,
        &[1, 3, 3],
    );
    cosmos_convolution(
        manifest,
        &format!("{name}.conv1.1"),
        output_channels,
        output_channels,
        &[3, 1, 1],
    );
    cosmos_normalization_manifest(manifest, &format!("{name}.norm2"), output_channels);
    cosmos_convolution(
        manifest,
        &format!("{name}.conv2.0"),
        output_channels,
        output_channels,
        &[1, 3, 3],
    );
    cosmos_convolution(
        manifest,
        &format!("{name}.conv2.1"),
        output_channels,
        output_channels,
        &[3, 1, 1],
    );
    if input_channels != output_channels {
        cosmos_convolution(
            manifest,
            &format!("{name}.nin_shortcut"),
            output_channels,
            input_channels,
            &[1, 1, 1],
        );
    }
}

fn cosmos_attention_pair_manifest(manifest: &mut StateManifest, name: &str, channels: u64) {
    for index in 0..2 {
        let prefix = format!("{name}.{index}");
        cosmos_normalization_manifest(manifest, &format!("{prefix}.norm"), channels);
        for projection in ["q", "k", "v", "proj_out"] {
            cosmos_convolution(
                manifest,
                &format!("{prefix}.{projection}"),
                channels,
                channels,
                &[1, 1, 1],
            );
        }
    }
}

fn cosmos_hybrid_resample_manifest(manifest: &mut StateManifest, name: &str, channels: u64) {
    cosmos_convolution(
        manifest,
        &format!("{name}.conv1"),
        channels,
        channels,
        &[1, 3, 3],
    );
    cosmos_convolution(
        manifest,
        &format!("{name}.conv2"),
        channels,
        channels,
        &[3, 1, 1],
    );
    cosmos_convolution(
        manifest,
        &format!("{name}.conv3"),
        channels,
        channels,
        &[1, 1, 1],
    );
}

fn wan21_state_schema(dtype: DType) -> Vec<NativeVisionStateSpec> {
    const DIM: u64 = 96;
    const ENCODER_DIMS: [u64; 5] = [DIM, DIM, DIM * 2, DIM * 4, DIM * 4];
    let mut manifest = StateManifest::new(dtype);
    manifest.convolution_nd("encoder.conv1", DIM, 3, &[3, 3, 3], true);
    let mut sequence = 0;
    for level in 0..4 {
        let mut input = ENCODER_DIMS[level];
        let output = ENCODER_DIMS[level + 1];
        for _ in 0..2 {
            wan_residual_manifest(
                &mut manifest,
                &format!("encoder.downsamples.{sequence}"),
                input,
                output,
            );
            input = output;
            sequence += 1;
        }
        if level < 3 {
            wan_resample_manifest(
                &mut manifest,
                &format!("encoder.downsamples.{sequence}"),
                output,
                false,
                level > 0,
            );
            sequence += 1;
        }
    }
    wan_residual_manifest(&mut manifest, "encoder.middle.0", DIM * 4, DIM * 4);
    wan_attention_manifest(&mut manifest, "encoder.middle.1", DIM * 4);
    wan_residual_manifest(&mut manifest, "encoder.middle.2", DIM * 4, DIM * 4);
    wan_rms_manifest(&mut manifest, "encoder.head.0", DIM * 4, 4);
    manifest.convolution_nd("encoder.head.2", 32, DIM * 4, &[3, 3, 3], true);

    manifest.convolution_nd("conv1", 32, 32, &[1, 1, 1], true);
    manifest.convolution_nd("conv2", 16, 16, &[1, 1, 1], true);

    const DECODER_DIMS: [u64; 5] = [DIM * 4, DIM * 4, DIM * 4, DIM * 2, DIM];
    manifest.convolution_nd("decoder.conv1", DIM * 4, 16, &[3, 3, 3], true);
    wan_residual_manifest(&mut manifest, "decoder.middle.0", DIM * 4, DIM * 4);
    wan_attention_manifest(&mut manifest, "decoder.middle.1", DIM * 4);
    wan_residual_manifest(&mut manifest, "decoder.middle.2", DIM * 4, DIM * 4);
    sequence = 0;
    let mut current = DIM * 4;
    for level in 0..4 {
        let output = DECODER_DIMS[level + 1];
        for _ in 0..3 {
            wan_residual_manifest(
                &mut manifest,
                &format!("decoder.upsamples.{sequence}"),
                current,
                output,
            );
            current = output;
            sequence += 1;
        }
        if level < 3 {
            wan_resample_manifest(
                &mut manifest,
                &format!("decoder.upsamples.{sequence}"),
                output,
                true,
                level < 2,
            );
            current = output / 2;
            sequence += 1;
        }
    }
    wan_rms_manifest(&mut manifest, "decoder.head.0", DIM, 4);
    manifest.convolution_nd("decoder.head.2", 3, DIM, &[3, 3, 3], true);
    manifest.state
}

fn wan22_state_schema(dtype: DType) -> Vec<NativeVisionStateSpec> {
    const ENCODER_DIM: u64 = 160;
    const DECODER_DIM: u64 = 256;
    const ENCODER_DIMS: [u64; 5] = [160, 160, 320, 640, 640];
    const DECODER_DIMS: [u64; 5] = [1024, 1024, 1024, 512, 256];
    let mut manifest = StateManifest::new(dtype);
    manifest.convolution_nd("encoder.conv1", ENCODER_DIM, 12, &[3, 3, 3], true);
    for level in 0..4 {
        let mut input = ENCODER_DIMS[level];
        let output = ENCODER_DIMS[level + 1];
        for block in 0..2 {
            wan_residual_manifest(
                &mut manifest,
                &format!("encoder.downsamples.{level}.downsamples.{block}"),
                input,
                output,
            );
            input = output;
        }
        if level < 3 {
            wan22_resample_manifest(
                &mut manifest,
                &format!("encoder.downsamples.{level}.downsamples.2"),
                output,
                false,
                matches!(level, 1 | 2),
            );
        }
    }
    wan_residual_manifest(&mut manifest, "encoder.middle.0", 640, 640);
    wan_attention_manifest(&mut manifest, "encoder.middle.1", 640);
    wan_residual_manifest(&mut manifest, "encoder.middle.2", 640, 640);
    wan_rms_manifest(&mut manifest, "encoder.head.0", 640, 4);
    manifest.convolution_nd("encoder.head.2", 96, 640, &[3, 3, 3], true);

    manifest.convolution_nd("conv1", 96, 96, &[1, 1, 1], true);
    manifest.convolution_nd("conv2", 48, 48, &[1, 1, 1], true);

    manifest.convolution_nd("decoder.conv1", 1024, 48, &[3, 3, 3], true);
    wan_residual_manifest(&mut manifest, "decoder.middle.0", 1024, 1024);
    wan_attention_manifest(&mut manifest, "decoder.middle.1", 1024);
    wan_residual_manifest(&mut manifest, "decoder.middle.2", 1024, 1024);
    let mut input = DECODER_DIMS[0];
    for level in 0..4 {
        let output = DECODER_DIMS[level + 1];
        for block in 0..3 {
            wan_residual_manifest(
                &mut manifest,
                &format!("decoder.upsamples.{level}.upsamples.{block}"),
                input,
                output,
            );
            input = output;
        }
        if level < 3 {
            wan22_resample_manifest(
                &mut manifest,
                &format!("decoder.upsamples.{level}.upsamples.3"),
                output,
                true,
                level < 2,
            );
        }
    }
    wan_rms_manifest(&mut manifest, "decoder.head.0", DECODER_DIM, 4);
    manifest.convolution_nd("decoder.head.2", 12, DECODER_DIM, &[3, 3, 3], true);
    manifest.state
}

fn wan_rms_manifest(manifest: &mut StateManifest, name: &str, channels: u64, rank: usize) {
    let mut shape = vec![channels];
    shape.extend(std::iter::repeat_n(1, rank.saturating_sub(1)));
    manifest.parameter(format!("{name}.gamma"), shape);
}

fn wan_residual_manifest(
    manifest: &mut StateManifest,
    name: &str,
    input_channels: u64,
    output_channels: u64,
) {
    wan_rms_manifest(manifest, &format!("{name}.residual.0"), input_channels, 4);
    manifest.convolution_nd(
        &format!("{name}.residual.2"),
        output_channels,
        input_channels,
        &[3, 3, 3],
        true,
    );
    wan_rms_manifest(manifest, &format!("{name}.residual.3"), output_channels, 4);
    manifest.convolution_nd(
        &format!("{name}.residual.6"),
        output_channels,
        output_channels,
        &[3, 3, 3],
        true,
    );
    if input_channels != output_channels {
        manifest.convolution_nd(
            &format!("{name}.shortcut"),
            output_channels,
            input_channels,
            &[1, 1, 1],
            true,
        );
    }
}

fn wan_attention_manifest(manifest: &mut StateManifest, name: &str, channels: u64) {
    wan_rms_manifest(manifest, &format!("{name}.norm"), channels, 3);
    manifest.convolution(&format!("{name}.to_qkv"), channels * 3, channels, 1, true);
    manifest.convolution(&format!("{name}.proj"), channels, channels, 1, true);
}

fn wan_resample_manifest(
    manifest: &mut StateManifest,
    name: &str,
    channels: u64,
    upsample: bool,
    temporal: bool,
) {
    manifest.convolution(
        &format!("{name}.resample.1"),
        if upsample { channels / 2 } else { channels },
        channels,
        3,
        true,
    );
    if temporal {
        manifest.convolution_nd(
            &format!("{name}.time_conv"),
            if upsample { channels * 2 } else { channels },
            channels,
            &[3, 1, 1],
            true,
        );
    }
}

fn wan22_resample_manifest(
    manifest: &mut StateManifest,
    name: &str,
    channels: u64,
    upsample: bool,
    temporal: bool,
) {
    manifest.convolution(&format!("{name}.resample.1"), channels, channels, 3, true);
    if temporal {
        manifest.convolution_nd(
            &format!("{name}.time_conv"),
            if upsample { channels * 2 } else { channels },
            channels,
            &[3, 1, 1],
            true,
        );
    }
}

fn cogvideox_state_schema(dtype: DType) -> Vec<NativeVisionStateSpec> {
    const ENCODER_CHANNELS: [u64; 4] = [128, 256, 256, 512];
    let mut manifest = StateManifest::new(dtype);
    cog_causal_convolution(&mut manifest, "encoder.conv_in", 128, 3, 3);
    let mut channels = 128;
    for (level, target) in ENCODER_CHANNELS.into_iter().enumerate() {
        for block in 0..3 {
            cog_residual_manifest(
                &mut manifest,
                &format!("encoder.down_blocks.{level}.resnets.{block}"),
                channels,
                target,
                None,
            );
            channels = target;
        }
        if level < 3 {
            manifest.convolution(
                &format!("encoder.down_blocks.{level}.downsamplers.0.conv"),
                channels,
                channels,
                3,
                true,
            );
        }
    }
    for block in 0..2 {
        cog_residual_manifest(
            &mut manifest,
            &format!("encoder.mid_block.resnets.{block}"),
            channels,
            channels,
            None,
        );
    }
    refiner_normalization(&mut manifest, "encoder.norm_out", channels, false);
    cog_causal_convolution(&mut manifest, "encoder.conv_out", 32, channels, 3);

    const DECODER_CHANNELS: [u64; 4] = [512, 256, 256, 128];
    channels = 512;
    cog_causal_convolution(&mut manifest, "decoder.conv_in", channels, 16, 3);
    for block in 0..2 {
        cog_residual_manifest(
            &mut manifest,
            &format!("decoder.mid_block.resnets.{block}"),
            channels,
            channels,
            Some(16),
        );
    }
    for (level, target) in DECODER_CHANNELS.into_iter().enumerate() {
        for block in 0..4 {
            cog_residual_manifest(
                &mut manifest,
                &format!("decoder.up_blocks.{level}.resnets.{block}"),
                channels,
                target,
                Some(16),
            );
            channels = target;
        }
        if level < 3 {
            manifest.convolution(
                &format!("decoder.up_blocks.{level}.upsamplers.0.conv"),
                channels,
                channels,
                3,
                true,
            );
        }
    }
    cog_spatial_norm_manifest(&mut manifest, "decoder.norm_out", channels, 16);
    cog_causal_convolution(&mut manifest, "decoder.conv_out", 3, channels, 3);
    manifest.state
}

fn cog_causal_convolution(
    manifest: &mut StateManifest,
    name: &str,
    output_channels: u64,
    input_channels: u64,
    kernel: u64,
) {
    manifest.convolution_nd(
        &format!("{name}.conv"),
        output_channels,
        input_channels,
        &[kernel, kernel, kernel],
        true,
    );
}

fn cog_spatial_norm_manifest(
    manifest: &mut StateManifest,
    name: &str,
    feature_channels: u64,
    latent_channels: u64,
) {
    refiner_normalization(
        manifest,
        &format!("{name}.norm_layer"),
        feature_channels,
        false,
    );
    cog_causal_convolution(
        manifest,
        &format!("{name}.conv_y"),
        feature_channels,
        latent_channels,
        1,
    );
    cog_causal_convolution(
        manifest,
        &format!("{name}.conv_b"),
        feature_channels,
        latent_channels,
        1,
    );
}

fn cog_residual_manifest(
    manifest: &mut StateManifest,
    name: &str,
    input_channels: u64,
    output_channels: u64,
    spatial_norm_channels: Option<u64>,
) {
    if let Some(latent_channels) = spatial_norm_channels {
        cog_spatial_norm_manifest(
            manifest,
            &format!("{name}.norm1"),
            input_channels,
            latent_channels,
        );
    } else {
        refiner_normalization(manifest, &format!("{name}.norm1"), input_channels, false);
    }
    cog_causal_convolution(
        manifest,
        &format!("{name}.conv1"),
        output_channels,
        input_channels,
        3,
    );
    if let Some(latent_channels) = spatial_norm_channels {
        cog_spatial_norm_manifest(
            manifest,
            &format!("{name}.norm2"),
            output_channels,
            latent_channels,
        );
    } else {
        refiner_normalization(manifest, &format!("{name}.norm2"), output_channels, false);
    }
    cog_causal_convolution(
        manifest,
        &format!("{name}.conv2"),
        output_channels,
        output_channels,
        3,
    );
    if input_channels != output_channels {
        manifest.convolution_nd(
            &format!("{name}.conv_shortcut"),
            output_channels,
            input_channels,
            &[1, 1, 1],
            true,
        );
    }
}

fn causal3d_state_schema(dtype: DType) -> Vec<NativeVisionStateSpec> {
    const CHANNELS: [u64; 4] = [128, 256, 512, 512];
    let mut manifest = StateManifest::new(dtype);
    refiner_convolution(&mut manifest, "encoder.conv_in", 128, 3, 3, true, true);
    let mut channels = 128;
    for (level, target) in CHANNELS.into_iter().enumerate() {
        for block in 0..2 {
            refiner_residual_block(
                &mut manifest,
                &format!("encoder.down.{level}.block.{block}"),
                channels,
                target,
                true,
                false,
            );
            channels = target;
        }
        if level < 3 {
            refiner_convolution(
                &mut manifest,
                &format!("encoder.down.{level}.downsample.conv"),
                channels,
                channels,
                3,
                true,
                true,
            );
        }
    }
    refiner_residual_block(
        &mut manifest,
        "encoder.mid.block_1",
        channels,
        channels,
        true,
        false,
    );
    refiner_attention_block(&mut manifest, "encoder.mid.attn_1", channels, false);
    refiner_residual_block(
        &mut manifest,
        "encoder.mid.block_2",
        channels,
        channels,
        true,
        false,
    );
    refiner_normalization(&mut manifest, "encoder.norm_out", channels, false);
    refiner_convolution(
        &mut manifest,
        "encoder.conv_out",
        8,
        channels,
        3,
        true,
        true,
    );
    manifest.convolution_nd("quant_conv", 8, 8, &[1, 1, 1], true);

    manifest.convolution_nd("post_quant_conv", 4, 4, &[1, 1, 1], true);
    channels = 512;
    refiner_convolution(&mut manifest, "decoder.conv_in", channels, 4, 3, true, true);
    refiner_residual_block(
        &mut manifest,
        "decoder.mid.block_1",
        channels,
        channels,
        true,
        false,
    );
    refiner_attention_block(&mut manifest, "decoder.mid.attn_1", channels, false);
    refiner_residual_block(
        &mut manifest,
        "decoder.mid.block_2",
        channels,
        channels,
        true,
        false,
    );
    for level in (0..4).rev() {
        let target = CHANNELS[level];
        for block in 0..3 {
            refiner_residual_block(
                &mut manifest,
                &format!("decoder.up.{level}.block.{block}"),
                channels,
                target,
                true,
                false,
            );
            channels = target;
        }
        if level > 0 {
            refiner_convolution(
                &mut manifest,
                &format!("decoder.up.{level}.upsample.conv"),
                channels,
                channels,
                3,
                true,
                true,
            );
        }
    }
    refiner_normalization(&mut manifest, "decoder.norm_out", channels, false);
    refiner_convolution(
        &mut manifest,
        "decoder.conv_out",
        3,
        channels,
        3,
        true,
        true,
    );
    manifest.state
}

fn hunyuan_refiner_state_schema(carried: bool, dtype: DType) -> Vec<NativeVisionStateSpec> {
    const CHANNELS: [u64; 5] = [128, 256, 512, 1024, 1024];
    let mut manifest = StateManifest::new(dtype);
    refiner_convolution(&mut manifest, "encoder.conv_in", 128, 3, 3, true, carried);
    let mut channels = 128;
    for (level, target) in CHANNELS.into_iter().enumerate() {
        for block in 0..2 {
            let input = if block == 0 { channels } else { target };
            refiner_residual_block(
                &mut manifest,
                &format!("encoder.down.{level}.block.{block}"),
                input,
                target,
                carried,
                carried,
            );
        }
        channels = target;
        if level < 4 {
            let next = CHANNELS[level + 1];
            let temporal = level >= 2;
            let factor = if temporal { 8 } else { 4 };
            refiner_convolution(
                &mut manifest,
                &format!("encoder.down.{level}.downsample.conv"),
                next / factor,
                channels,
                3,
                true,
                carried,
            );
            channels = next;
        }
    }
    refiner_residual_block(
        &mut manifest,
        "encoder.mid.block_1",
        channels,
        channels,
        carried,
        carried,
    );
    refiner_attention_block(&mut manifest, "encoder.mid.attn_1", channels, carried);
    refiner_residual_block(
        &mut manifest,
        "encoder.mid.block_2",
        channels,
        channels,
        carried,
        carried,
    );
    refiner_normalization(&mut manifest, "encoder.norm_out", channels, carried);
    refiner_convolution(
        &mut manifest,
        "encoder.conv_out",
        64,
        channels,
        3,
        true,
        carried,
    );

    let decoder_channels = [1024_u64, 1024, 512, 256, 128];
    channels = decoder_channels[0];
    refiner_convolution(
        &mut manifest,
        "decoder.conv_in",
        channels,
        32,
        3,
        true,
        carried,
    );
    refiner_residual_block(
        &mut manifest,
        "decoder.mid.block_1",
        channels,
        channels,
        carried,
        carried,
    );
    refiner_attention_block(&mut manifest, "decoder.mid.attn_1", channels, carried);
    refiner_residual_block(
        &mut manifest,
        "decoder.mid.block_2",
        channels,
        channels,
        carried,
        carried,
    );
    for (level, target) in decoder_channels.into_iter().enumerate() {
        for block in 0..3 {
            let input = if block == 0 { channels } else { target };
            refiner_residual_block(
                &mut manifest,
                &format!("decoder.up.{level}.block.{block}"),
                input,
                target,
                carried,
                carried,
            );
        }
        channels = target;
        if level < 4 {
            let next = decoder_channels[level + 1];
            let temporal = level < 2;
            let factor = if temporal { 8 } else { 4 };
            refiner_convolution(
                &mut manifest,
                &format!("decoder.up.{level}.upsample.conv"),
                next * factor,
                channels,
                3,
                true,
                carried,
            );
            channels = next;
        }
    }
    refiner_normalization(&mut manifest, "decoder.norm_out", channels, carried);
    refiner_convolution(
        &mut manifest,
        "decoder.conv_out",
        3,
        channels,
        3,
        true,
        carried,
    );
    manifest.state
}

fn refiner_convolution(
    manifest: &mut StateManifest,
    name: &str,
    output_channels: u64,
    input_channels: u64,
    kernel: u64,
    bias: bool,
    carried: bool,
) {
    let name = if carried {
        format!("{name}.conv")
    } else {
        name.to_owned()
    };
    manifest.convolution_nd(
        &name,
        output_channels,
        input_channels,
        &[kernel, kernel, kernel],
        bias,
    );
}

fn refiner_normalization(manifest: &mut StateManifest, name: &str, channels: u64, rms: bool) {
    if rms {
        manifest.parameter(format!("{name}.gamma"), vec![channels, 1, 1, 1]);
    } else {
        manifest.parameter(format!("{name}.weight"), vec![channels]);
        manifest.parameter(format!("{name}.bias"), vec![channels]);
    }
}

fn refiner_residual_block(
    manifest: &mut StateManifest,
    name: &str,
    input_channels: u64,
    output_channels: u64,
    carried: bool,
    rms: bool,
) {
    refiner_normalization(manifest, &format!("{name}.norm1"), input_channels, rms);
    refiner_convolution(
        manifest,
        &format!("{name}.conv1"),
        output_channels,
        input_channels,
        3,
        true,
        carried,
    );
    refiner_normalization(manifest, &format!("{name}.norm2"), output_channels, rms);
    refiner_convolution(
        manifest,
        &format!("{name}.conv2"),
        output_channels,
        output_channels,
        3,
        true,
        carried,
    );
    if input_channels != output_channels {
        refiner_convolution(
            manifest,
            &format!("{name}.nin_shortcut"),
            output_channels,
            input_channels,
            1,
            true,
            carried,
        );
    }
}

fn refiner_attention_block(manifest: &mut StateManifest, name: &str, channels: u64, rms: bool) {
    refiner_normalization(manifest, &format!("{name}.norm"), channels, rms);
    for projection in ["q", "k", "v", "proj_out"] {
        refiner_convolution(
            manifest,
            &format!("{name}.{projection}"),
            channels,
            channels,
            1,
            true,
            false,
        );
    }
}

const MOCHI_STATE: &[VideoVaeSourceCheckpoint] = &[
    VideoVaeSourceCheckpoint {
        name: "decoder.blocks.2.blocks.3.stack.5.weight",
        rank: 5,
        dimensions: &[],
    },
    VideoVaeSourceCheckpoint {
        name: "encoder.layers.4.layers.1.attn_block.attn.qkv.weight",
        rank: 2,
        dimensions: &[],
    },
];
const LTX_V0_STATE: &[VideoVaeSourceCheckpoint] = &[VideoVaeSourceCheckpoint {
    name: "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
    rank: 5,
    dimensions: &[(0, 512), (1, 512)],
}];
const LTX_V1_V2_STATE: &[VideoVaeSourceCheckpoint] = &[VideoVaeSourceCheckpoint {
    name: "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
    rank: 5,
    dimensions: &[(0, 1024), (1, 1024)],
}];
const HUNYUAN_REFINER_STATE: &[VideoVaeSourceCheckpoint] = &[VideoVaeSourceCheckpoint {
    name: "decoder.conv_in.conv.weight",
    rank: 5,
    dimensions: &[(1, 32)],
}];
const COGVIDEOX_STATE: &[VideoVaeSourceCheckpoint] = &[
    VideoVaeSourceCheckpoint {
        name: "decoder.conv_in.conv.weight",
        rank: 5,
        dimensions: &[],
    },
    VideoVaeSourceCheckpoint {
        name: "decoder.mid_block.resnets.0.norm1.norm_layer.weight",
        rank: 1,
        dimensions: &[],
    },
    VideoVaeSourceCheckpoint {
        name: "encoder.conv_out.conv.weight",
        rank: 5,
        dimensions: &[],
    },
];
const CAUSAL_3D_STATE: &[VideoVaeSourceCheckpoint] = &[
    VideoVaeSourceCheckpoint {
        name: "decoder.conv_in.conv.weight",
        rank: 5,
        dimensions: &[],
    },
    VideoVaeSourceCheckpoint {
        name: "post_quant_conv.weight",
        rank: 5,
        dimensions: &[],
    },
];
const COSMOS_STATE: &[VideoVaeSourceCheckpoint] = &[VideoVaeSourceCheckpoint {
    name: "decoder.unpatcher3d.wavelets",
    rank: 5,
    dimensions: &[],
}];
const WAN_21_STATE: &[VideoVaeSourceCheckpoint] = &[
    VideoVaeSourceCheckpoint {
        name: "decoder.middle.0.residual.0.gamma",
        rank: 1,
        dimensions: &[],
    },
    VideoVaeSourceCheckpoint {
        name: "encoder.conv1.weight",
        rank: 5,
        dimensions: &[],
    },
    VideoVaeSourceCheckpoint {
        name: "decoder.head.2.weight",
        rank: 5,
        dimensions: &[],
    },
];
const WAN_22_STATE: &[VideoVaeSourceCheckpoint] = &[
    VideoVaeSourceCheckpoint {
        name: "decoder.middle.0.residual.0.gamma",
        rank: 1,
        dimensions: &[],
    },
    VideoVaeSourceCheckpoint {
        name: "decoder.upsamples.0.upsamples.0.residual.2.weight",
        rank: 5,
        dimensions: &[],
    },
];
const TAEHV_STATE: &[VideoVaeSourceCheckpoint] = &[
    VideoVaeSourceCheckpoint {
        name: "decoder.1.weight",
        rank: 4,
        dimensions: &[],
    },
    VideoVaeSourceCheckpoint {
        name: "decoder.22.bias",
        rank: 1,
        dimensions: &[],
    },
];

const HUNYUAN_IMAGE_REFINER_STATE: &[VideoVaeSourceCheckpoint] = &[VideoVaeSourceCheckpoint {
    name: "decoder.conv_in.weight",
    rank: 5,
    dimensions: &[(1, 32)],
}];

const MOCHI_EQUATIONS: &[&str] = &[
    "causal_temporal_encode_ceil_t_div_6",
    "causal_temporal_decode_t_mul_6_minus_5",
    "spatial_stride_8",
    "per_channel_latent_affine",
];
const LTX_EQUATIONS: &[&str] = &[
    "causal_temporal_encode_ceil_t_div_8",
    "causal_temporal_decode_t_mul_8_minus_7",
    "spatial_stride_32",
    "caller_addressed_deterministic_decode_rng",
];
const HUNYUAN_REFINER_EQUATIONS: &[&str] = &[
    "causal_temporal_encode_ceil_t_div_4",
    "causal_temporal_decode_t_mul_4_minus_3",
    "spatial_stride_16",
    "first_frame_cache_boundary",
];
const HUNYUAN_IMAGE_REFINER_EQUATIONS: &[&str] = &[
    "single_frame_encode_expand_temporal_4",
    "single_frame_decode_select_last",
    "spatial_stride_16",
    "diagonal_gaussian_mode",
];
const COGVIDEOX_EQUATIONS: &[&str] = &[
    "causal_conv3d_first_frame_replication",
    "rolling_temporal_encode_cache",
    "rolling_temporal_decode_cache",
    "diagonal_gaussian_mode",
];
const CAUSAL_3D_EQUATIONS: &[&str] = &[
    "causal_conv3d_prefix_cache",
    "temporal_compress_4",
    "diagonal_gaussian_mode",
];
const COSMOS_EQUATIONS: &[&str] = &[
    "causal_temporal_encode_ceil_t_div_8",
    "causal_temporal_decode_t_mul_8_minus_7",
    "haar_wavelet_patchify_3d",
    "haar_wavelet_unpatchify_3d",
];
const WAN_EQUATIONS: &[&str] = &[
    "causal_conv3d_two_frame_cache",
    "first_frame_separate_encode",
    "first_frame_separate_decode",
    "causal_temporal_index_4",
];
const TAEHV_EQUATIONS: &[&str] = &[
    "frame_queue_dependency_order",
    "first_frame_zero_temporal_memory",
    "temporal_blend_memory",
    "bounded_frame_work_queue",
];

pub fn video_vae_source_plan(
    profile: &VaeKernelProfile,
) -> Result<NativeVideoVaeArchitecture, VideoVaeError> {
    let (architecture, temporal_ratio, spatial_ratio, checkpoints, equations) = match profile {
        VaeKernelProfile::HunyuanImageRefinerV1 => (
            HUNYUAN_IMAGE_REFINER_ARCHITECTURE,
            4,
            16,
            HUNYUAN_IMAGE_REFINER_STATE,
            HUNYUAN_IMAGE_REFINER_EQUATIONS,
        ),
        VaeKernelProfile::MochiV1 => (MOCHI_ARCHITECTURE, 6, 8, MOCHI_STATE, MOCHI_EQUATIONS),
        VaeKernelProfile::LtxVideoV0 { .. } => {
            (LTX_ARCHITECTURE, 8, 32, LTX_V0_STATE, LTX_EQUATIONS)
        }
        VaeKernelProfile::LtxVideoV1 { .. } | VaeKernelProfile::LtxVideoV2 { .. } => {
            (LTX_ARCHITECTURE, 8, 32, LTX_V1_V2_STATE, LTX_EQUATIONS)
        }
        VaeKernelProfile::HunyuanVideoRefinerV1 => (
            HUNYUAN_VIDEO_REFINER_ARCHITECTURE,
            4,
            16,
            HUNYUAN_REFINER_STATE,
            HUNYUAN_REFINER_EQUATIONS,
        ),
        VaeKernelProfile::CogVideoXV1 => (
            COGVIDEOX_ARCHITECTURE,
            4,
            8,
            COGVIDEOX_STATE,
            COGVIDEOX_EQUATIONS,
        ),
        VaeKernelProfile::Causal3dV1 => (
            CAUSAL_3D_ARCHITECTURE,
            4,
            8,
            CAUSAL_3D_STATE,
            CAUSAL_3D_EQUATIONS,
        ),
        VaeKernelProfile::CosmosV1 => (COSMOS_ARCHITECTURE, 8, 8, COSMOS_STATE, COSMOS_EQUATIONS),
        VaeKernelProfile::Wan21V1 => (WAN_21_ARCHITECTURE, 4, 8, WAN_21_STATE, WAN_EQUATIONS),
        VaeKernelProfile::Wan22V1 => (WAN_22_ARCHITECTURE, 4, 16, WAN_22_STATE, WAN_EQUATIONS),
        VaeKernelProfile::TaeHvWan22V1 => (TAEHV_ARCHITECTURE, 4, 16, TAEHV_STATE, TAEHV_EQUATIONS),
        VaeKernelProfile::TaeHvLtx2V1 => (TAEHV_ARCHITECTURE, 8, 32, TAEHV_STATE, TAEHV_EQUATIONS),
        VaeKernelProfile::LightTaeHv15V1 => {
            (TAEHV_ARCHITECTURE, 4, 16, TAEHV_STATE, TAEHV_EQUATIONS)
        }
        VaeKernelProfile::TaeHvHunyuanV1 | VaeKernelProfile::LightTaeWan21V1 => {
            (TAEHV_ARCHITECTURE, 4, 8, TAEHV_STATE, TAEHV_EQUATIONS)
        }
        profile => return Err(VideoVaeError::UnsupportedProfile(profile.clone())),
    };
    Ok(NativeVideoVaeArchitecture {
        profile: profile.clone(),
        architecture,
        temporal_ratio,
        spatial_ratio,
        checkpoints,
        equations,
        storage_dtype: None,
    })
}

pub fn inspect_video_vae_architecture(
    descriptor: &VaeDescriptor,
    model: &LoadedModel,
) -> Result<NativeVideoVaeArchitecture, VideoVaeError> {
    let plan = video_vae_source_plan(descriptor.identity().profile())?;
    let actual = descriptor.identity().architecture().as_str();
    if actual != plan.architecture() {
        return Err(VideoVaeError::ArchitectureMismatch {
            expected: plan.architecture(),
            actual: actual.to_owned(),
        });
    }
    if descriptor.identity().profile() == &VaeKernelProfile::MochiV1 {
        return inspect_mochi_architecture(model, plan);
    }
    if matches!(
        descriptor.identity().profile(),
        VaeKernelProfile::TaeHvWan22V1
            | VaeKernelProfile::TaeHvLtx2V1
            | VaeKernelProfile::LightTaeHv15V1
            | VaeKernelProfile::TaeHvHunyuanV1
            | VaeKernelProfile::LightTaeWan21V1
            | VaeKernelProfile::HunyuanImageRefinerV1
            | VaeKernelProfile::HunyuanVideoRefinerV1
            | VaeKernelProfile::Causal3dV1
            | VaeKernelProfile::CogVideoXV1
            | VaeKernelProfile::CosmosV1
            | VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
            | VaeKernelProfile::Wan21V1
            | VaeKernelProfile::Wan22V1
    ) {
        let sentinel_name = plan
            .state_checkpoints()
            .first()
            .map(|checkpoint| checkpoint.name)
            .ok_or_else(|| VideoVaeError::MissingState("video VAE dtype sentinel".to_owned()))?;
        let sentinel = model
            .tensors()
            .get(sentinel_name)
            .ok_or_else(|| VideoVaeError::MissingState(sentinel_name.to_owned()))?;
        if descriptor.identity().profile() == &VaeKernelProfile::CosmosV1
            && (sentinel.shape.len() != 5 || sentinel.shape.contains(&0))
        {
            return Err(VideoVaeError::InvalidStateShape {
                name: sentinel_name.to_owned(),
                shape: sentinel.shape.clone(),
            });
        }
        let dtype = canonical_vision_model_store_dtype(&sentinel.data_type).ok_or_else(|| {
            VideoVaeError::UnsupportedStorageDType {
                name: sentinel_name.to_owned(),
                dtype: sentinel.data_type.clone(),
            }
        })?;
        let schema = video_vae_source_state_schema_for_descriptor(descriptor, dtype)?;
        let expected_names = schema
            .iter()
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(unexpected) = model.tensors().keys().find(|name| {
            !expected_names.contains(name.as_str())
                && !(descriptor.identity().profile() == &VaeKernelProfile::CosmosV1
                    && cosmos_compatibility_buffer(name))
        }) {
            return Err(VideoVaeError::UnexpectedState(unexpected.clone()));
        }
        for state in schema {
            let metadata = model
                .tensors()
                .get(&state.name)
                .ok_or_else(|| VideoVaeError::MissingState(state.name.clone()))?;
            if metadata.shape != state.shape {
                return Err(VideoVaeError::InvalidStateShape {
                    name: state.name,
                    shape: metadata.shape.clone(),
                });
            }
            let actual_dtype =
                canonical_vision_model_store_dtype(&metadata.data_type).ok_or_else(|| {
                    VideoVaeError::UnsupportedStorageDType {
                        name: state.name.clone(),
                        dtype: metadata.data_type.clone(),
                    }
                })?;
            if actual_dtype != state.dtype {
                return Err(VideoVaeError::MixedStorageDType {
                    name: state.name,
                    expected: state.dtype,
                    actual: actual_dtype,
                });
            }
        }
        if descriptor.identity().profile() == &VaeKernelProfile::CosmosV1 {
            for (name, metadata) in model
                .tensors()
                .iter()
                .filter(|(name, _)| cosmos_compatibility_buffer(name))
            {
                let (expected_shape, expected_dtype) = cosmos_compatibility_buffer_spec(name)
                    .ok_or_else(|| VideoVaeError::UnexpectedState(name.clone()))?;
                if metadata.shape != expected_shape {
                    return Err(VideoVaeError::InvalidStateShape {
                        name: name.clone(),
                        shape: metadata.shape.clone(),
                    });
                }
                if metadata.data_type != expected_dtype {
                    return Err(VideoVaeError::UnsupportedStorageDType {
                        name: name.clone(),
                        dtype: metadata.data_type.clone(),
                    });
                }
            }
        }
        return Ok(NativeVideoVaeArchitecture {
            storage_dtype: Some(dtype),
            ..plan
        });
    }
    let mut admitted_dtype = None;
    let mut seen = BTreeSet::new();
    for checkpoint in plan.state_checkpoints() {
        if !seen.insert(checkpoint.name) {
            continue;
        }
        let metadata = model
            .tensors()
            .get(checkpoint.name)
            .ok_or_else(|| VideoVaeError::MissingState(checkpoint.name.to_owned()))?;
        if metadata.shape.len() != usize::from(checkpoint.rank)
            || metadata.shape.contains(&0)
            || checkpoint
                .dimensions
                .iter()
                .any(|(axis, expected)| metadata.shape.get(*axis).copied() != Some(*expected))
        {
            return Err(VideoVaeError::InvalidStateShape {
                name: checkpoint.name.to_owned(),
                shape: metadata.shape.clone(),
            });
        }
        let dtype = match metadata.data_type.as_str() {
            "F16" => DType::F16,
            "BF16" => DType::Bf16,
            "F32" => DType::F32,
            _ => {
                return Err(VideoVaeError::UnsupportedStorageDType {
                    name: checkpoint.name.to_owned(),
                    dtype: metadata.data_type.clone(),
                });
            }
        };
        if let Some(expected) = admitted_dtype {
            if expected != dtype {
                return Err(VideoVaeError::MixedStorageDType {
                    name: checkpoint.name.to_owned(),
                    expected,
                    actual: dtype,
                });
            }
        } else {
            admitted_dtype = Some(dtype);
        }
    }
    Ok(NativeVideoVaeArchitecture {
        storage_dtype: admitted_dtype,
        ..plan
    })
}

pub fn load_video_vae_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: Arc<LoadedModel>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, VideoVaeError> {
    context.cancellation.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let architecture = inspect_video_vae_architecture(&descriptor, &model)?;
    if !matches!(
        architecture.profile(),
        VaeKernelProfile::TaeHvWan22V1
            | VaeKernelProfile::TaeHvLtx2V1
            | VaeKernelProfile::LightTaeHv15V1
            | VaeKernelProfile::TaeHvHunyuanV1
            | VaeKernelProfile::LightTaeWan21V1
            | VaeKernelProfile::HunyuanImageRefinerV1
            | VaeKernelProfile::HunyuanVideoRefinerV1
            | VaeKernelProfile::Causal3dV1
            | VaeKernelProfile::CogVideoXV1
            | VaeKernelProfile::CosmosV1
            | VaeKernelProfile::MochiV1
            | VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
            | VaeKernelProfile::Wan21V1
            | VaeKernelProfile::Wan22V1
    ) {
        return Err(VideoVaeError::UnsupportedProfile(
            architecture.profile().clone(),
        ));
    }
    let storage_dtype = architecture.storage_dtype().ok_or_else(|| {
        VideoVaeError::MissingState("video VAE storage dtype sentinel".to_owned())
    })?;
    let schema = video_vae_source_state_schema_for_descriptor(&descriptor, storage_dtype)?;
    let state = if architecture.profile() == &VaeKernelProfile::MochiV1 {
        let projection = mochi_state_projection(&model, storage_dtype)?;
        load_projected_vision_state_from_model_store_with_context(
            backend,
            store,
            index,
            &model,
            &projection,
            context,
        )?
    } else {
        load_vision_state_from_model_store_with_context(
            backend, store, index, &model, &schema, context,
        )?
    };
    let (module, encode, decode) = if matches!(
        architecture.profile(),
        VaeKernelProfile::HunyuanImageRefinerV1 | VaeKernelProfile::HunyuanVideoRefinerV1
    ) {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            hunyuan_refiner_encode_raw as _,
            hunyuan_refiner_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::Causal3dV1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            causal3d_encode_raw as _,
            causal3d_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::CogVideoXV1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            cogvideox_encode_raw as _,
            cogvideox_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::CosmosV1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            cosmos_encode_raw as _,
            cosmos_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::MochiV1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            mochi_encode_raw as _,
            mochi_decode_raw as _,
        )
    } else if matches!(
        architecture.profile(),
        VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
    ) {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            ltx_encode_raw as _,
            ltx_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::Wan21V1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            wan21_encode_raw as _,
            wan21_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::Wan22V1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            wan22_encode_raw as _,
            wan22_decode_raw as _,
        )
    } else {
        (
            build_taehv_module(&architecture, state, backend, &descriptor, context)?,
            taehv_encode_raw as _,
            taehv_decode_raw as _,
        )
    };
    let binding =
        VaeModelBinding::checked(&descriptor, store, model, module, context.cancellation)?;
    let functions =
        VaeKernelFunctions::checked(descriptor.identity().architecture().clone(), encode, decode);
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

pub(crate) fn load_video_vae_from_recipe_state(
    backend: &CpuBackend,
    artifact_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, VideoVaeError> {
    context.cancellation.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let mut state = BTreeMap::new();
    for (name, tensor) in ordered_state {
        if state.insert(name.clone(), tensor).is_some() {
            return Err(VideoVaeError::UnexpectedState(name));
        }
    }
    let storage_dtype = state
        .values()
        .next()
        .map(|tensor| tensor.descriptor().dtype())
        .ok_or_else(|| VideoVaeError::MissingState("video VAE state".to_owned()))?;
    if !matches!(storage_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(VideoVaeError::UnsupportedStorageDType {
            name: "video VAE state".to_owned(),
            dtype: storage_dtype.catalog_name().to_owned(),
        });
    }
    let mut architecture = video_vae_source_plan(descriptor.identity().profile())?;
    if descriptor.identity().architecture().as_str() != architecture.architecture() {
        return Err(VideoVaeError::ArchitectureMismatch {
            expected: architecture.architecture(),
            actual: descriptor.identity().architecture().as_str().to_owned(),
        });
    }
    architecture.storage_dtype = Some(storage_dtype);
    let schema = video_vae_source_state_schema_for_descriptor(&descriptor, storage_dtype)?;
    let expected = schema
        .iter()
        .map(|specification| specification.name.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(name) = state.keys().find(|name| !expected.contains(name.as_str())) {
        return Err(VideoVaeError::UnexpectedState(name.clone()));
    }
    for specification in &schema {
        let tensor = state
            .get(&specification.name)
            .ok_or_else(|| VideoVaeError::MissingState(specification.name.clone()))?;
        if tensor.descriptor().shape() != specification.shape {
            return Err(VideoVaeError::InvalidStateShape {
                name: specification.name.clone(),
                shape: tensor.descriptor().shape().to_vec(),
            });
        }
        if tensor.descriptor().dtype() != specification.dtype {
            return Err(VideoVaeError::MixedStorageDType {
                name: specification.name.clone(),
                expected: specification.dtype,
                actual: tensor.descriptor().dtype(),
            });
        }
    }
    let (module, encode, decode) = if matches!(
        architecture.profile(),
        VaeKernelProfile::HunyuanImageRefinerV1 | VaeKernelProfile::HunyuanVideoRefinerV1
    ) {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            hunyuan_refiner_encode_raw as _,
            hunyuan_refiner_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::Causal3dV1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            causal3d_encode_raw as _,
            causal3d_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::CogVideoXV1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            cogvideox_encode_raw as _,
            cogvideox_decode_raw as _,
        )
    } else if matches!(
        architecture.profile(),
        VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
    ) {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            ltx_encode_raw as _,
            ltx_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::Wan21V1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            wan21_encode_raw as _,
            wan21_decode_raw as _,
        )
    } else if architecture.profile() == &VaeKernelProfile::Wan22V1 {
        (
            build_video_3d_module(&architecture, state, backend, &descriptor, context)?,
            wan22_encode_raw as _,
            wan22_decode_raw as _,
        )
    } else if matches!(
        architecture.profile(),
        VaeKernelProfile::TaeHvWan22V1
            | VaeKernelProfile::TaeHvLtx2V1
            | VaeKernelProfile::LightTaeHv15V1
            | VaeKernelProfile::TaeHvHunyuanV1
            | VaeKernelProfile::LightTaeWan21V1
    ) {
        (
            build_taehv_module(&architecture, state, backend, &descriptor, context)?,
            taehv_encode_raw as _,
            taehv_decode_raw as _,
        )
    } else {
        return Err(VideoVaeError::UnsupportedProfile(
            architecture.profile().clone(),
        ));
    };
    let binding = VaeModelBinding::checked_recipe(
        &descriptor,
        artifact_sha256,
        module,
        context.cancellation,
    )?;
    let functions =
        VaeKernelFunctions::checked(descriptor.identity().architecture().clone(), encode, decode);
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

#[cfg(feature = "test-support")]
pub(crate) fn load_video_vae_from_reduced_recipe_state(
    backend: &CpuBackend,
    artifact_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, VideoVaeError> {
    context.cancellation.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let architecture = video_vae_source_plan(descriptor.identity().profile())?;
    if descriptor.identity().architecture().as_str() != architecture.architecture() {
        return Err(VideoVaeError::ArchitectureMismatch {
            expected: architecture.architecture(),
            actual: descriptor.identity().architecture().as_str().to_owned(),
        });
    }
    let (markers, encode, decode) = match descriptor.identity().profile() {
        VaeKernelProfile::LightTaeHv15V1 => (
            [Some("decoder.22.bias"), Some("decoder.1.weight")],
            taehv_encode_raw as _,
            taehv_decode_raw as _,
        ),
        VaeKernelProfile::LtxVideoV0 { .. }
        | VaeKernelProfile::LtxVideoV1 { .. }
        | VaeKernelProfile::LtxVideoV2 { .. } => (
            [
                Some("decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"),
                None,
            ],
            ltx_encode_raw as _,
            ltx_decode_raw as _,
        ),
        profile => return Err(VideoVaeError::UnsupportedProfile(profile.clone())),
    };
    for marker in markers.into_iter().flatten() {
        if !ordered_state.iter().any(|(name, _)| name == marker) {
            return Err(VideoVaeError::MissingState(marker.to_owned()));
        }
    }
    let module = crate::vae::reduced_recipe_state_module(
        "video-vae-reduced-recipe",
        ordered_state,
        context.cancellation,
    )?;
    let binding = VaeModelBinding::checked_recipe(
        &descriptor,
        artifact_sha256,
        module,
        context.cancellation,
    )?;
    let functions =
        VaeKernelFunctions::checked(descriptor.identity().architecture().clone(), encode, decode);
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

fn build_video_3d_module(
    architecture: &NativeVideoVaeArchitecture,
    mut state: std::collections::BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    descriptor: &VaeDescriptor,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, VideoVaeError> {
    let carried = matches!(
        architecture.profile(),
        VaeKernelProfile::HunyuanVideoRefinerV1
            | VaeKernelProfile::Causal3dV1
            | VaeKernelProfile::CogVideoXV1
            | VaeKernelProfile::CosmosV1
            | VaeKernelProfile::MochiV1
            | VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
            | VaeKernelProfile::Wan21V1
            | VaeKernelProfile::Wan22V1
    );
    let mut children = Vec::new();
    let ltx_configuration = if matches!(
        architecture.profile(),
        VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
    ) {
        Some(ltx_configuration(
            architecture.profile(),
            descriptor.identity().loader_configuration(),
        )?)
    } else {
        None
    };
    for spec in video_vae_source_state_schema_for_descriptor(
        descriptor,
        architecture.storage_dtype().ok_or_else(|| {
            VideoVaeError::MissingState("video VAE storage dtype sentinel".to_owned())
        })?,
    )? {
        if architecture.profile() == &VaeKernelProfile::MochiV1 && !state.contains_key(&spec.name) {
            continue;
        }
        let Some(prefix) = spec.name.strip_suffix(".weight") else {
            if let Some(tensor) = state.remove(&spec.name) {
                children.push(NativeModule::buffer(spec.name, tensor)?);
            } else if !spec.name.ends_with(".bias") {
                return Err(VideoVaeError::MissingState(spec.name));
            }
            continue;
        };
        if spec.shape.len() == 2
            && (architecture.profile() == &VaeKernelProfile::MochiV1
                || matches!(
                    architecture.profile(),
                    VaeKernelProfile::LtxVideoV0 { .. }
                        | VaeKernelProfile::LtxVideoV1 { .. }
                        | VaeKernelProfile::LtxVideoV2 { .. }
                ))
        {
            let bias_name = format!("{prefix}.bias");
            let bias = state.remove(&bias_name);
            let weight = state
                .remove(&spec.name)
                .ok_or_else(|| VideoVaeError::MissingState(spec.name.clone()))?;
            let mut module = NativeModule::linear(
                spec.name,
                usize::try_from(spec.shape[1]).map_err(|_| VaeError::ShapeOverflow)?,
                usize::try_from(spec.shape[0]).map_err(|_| VaeError::ShapeOverflow)?,
                bias.is_some(),
                false,
            )?;
            module.load_dense_parameters(weight, bias)?;
            children.push(module);
            continue;
        }
        if !matches!(spec.shape.len(), 4 | 5) {
            let tensor = state
                .remove(&spec.name)
                .ok_or_else(|| VideoVaeError::MissingState(spec.name.clone()))?;
            children.push(NativeModule::buffer(spec.name, tensor)?);
            continue;
        }
        let kernel = spec.shape[2..]
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| VideoVaeError::InvalidStateShape {
                name: spec.name.clone(),
                shape: spec.shape.clone(),
            })?;
        let causal = spec.shape.len() == 5
            && (matches!(
                architecture.profile(),
                VaeKernelProfile::Wan21V1
                    | VaeKernelProfile::Wan22V1
                    | VaeKernelProfile::CosmosV1
                    | VaeKernelProfile::MochiV1
                    | VaeKernelProfile::LtxVideoV0 { .. }
                    | VaeKernelProfile::LtxVideoV1 { .. }
                    | VaeKernelProfile::LtxVideoV2 { .. }
            ) || (carried && is_carried_convolution_invoked_causally(&spec.name)));
        let stride = if let Some(configuration) = &ltx_configuration {
            ltx_convolution_stride(configuration, &spec.name)?
        } else {
            video_3d_convolution_stride(architecture.profile(), &spec.name)
        };
        let padding = if architecture.profile() == &VaeKernelProfile::CosmosV1
            && spec.name == "encoder.down.0.downsample.conv1.conv3d.weight"
        {
            vec![0; kernel.len()]
        } else if causal
            && matches!(
                architecture.profile(),
                VaeKernelProfile::CogVideoXV1
                    | VaeKernelProfile::Wan21V1
                    | VaeKernelProfile::Wan22V1
                    | VaeKernelProfile::CosmosV1
            )
        {
            vec![0, kernel[1] / 2, kernel[2] / 2]
        } else if causal
            && matches!(
                architecture.profile(),
                VaeKernelProfile::LtxVideoV0 { .. }
                    | VaeKernelProfile::LtxVideoV1 { .. }
                    | VaeKernelProfile::LtxVideoV2 { .. }
            )
        {
            vec![0; kernel.len()]
        } else if causal {
            vec![0; kernel.len()]
        } else if spec.shape.len() == 4
            && (spec.name.contains(".downsamplers.")
                || (matches!(
                    architecture.profile(),
                    VaeKernelProfile::Wan21V1 | VaeKernelProfile::Wan22V1
                ) && spec.name.starts_with("encoder.")))
        {
            vec![0; kernel.len()]
        } else {
            kernel.iter().map(|extent| extent / 2).collect()
        };
        let geometry = ConvolutionGeometry::new(
            kernel.len(),
            stride,
            padding,
            vec![1; kernel.len()],
            1,
            false,
            vec![0; kernel.len()],
        )
        .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
        let bias_name = format!("{prefix}.bias");
        let bias = state.remove(&bias_name);
        let weight = state
            .remove(&spec.name)
            .ok_or_else(|| VideoVaeError::MissingState(spec.name.clone()))?;
        let mut module = NativeModule::convolution(
            spec.name,
            usize::try_from(spec.shape[1]).map_err(|_| VaeError::ShapeOverflow)?,
            usize::try_from(spec.shape[0]).map_err(|_| VaeError::ShapeOverflow)?,
            kernel,
            bias.is_some(),
            geometry,
            false,
        )?;
        module.load_dense_parameters(weight, bias)?;
        children.push(module);
    }
    for (name, tensor) in state {
        children.push(NativeModule::buffer(name, tensor)?);
    }
    if let Some(configuration) = &ltx_configuration {
        children.extend(ltx_configuration_modules(
            configuration,
            backend,
            descriptor.identity().device(),
            context,
        )?);
    }
    let mut module =
        NativeModule::module_dict(format!("video-vae:{:?}", architecture.profile()), children)?;
    module.materialize_execution_state_with_context(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
        context,
    )?;
    Ok(module)
}

fn ltx_convolution_stride(
    configuration: &LtxConfiguration,
    name: &str,
) -> Result<Vec<usize>, VideoVaeError> {
    let Some(index) = name
        .strip_prefix("encoder.down_blocks.")
        .and_then(|suffix| suffix.strip_suffix(".conv.weight"))
        .and_then(|index| index.parse::<usize>().ok())
    else {
        return Ok(vec![1, 1, 1]);
    };
    let block = configuration.encoder_blocks.get(index).ok_or_else(|| {
        VideoVaeError::InvalidLtxConfiguration(format!(
            "encoder convolution refers to missing block {index}"
        ))
    })?;
    let scale = ltx_block_scale(*block);
    Ok(vec![
        usize::try_from(scale.0).map_err(|_| {
            VideoVaeError::InvalidLtxConfiguration("temporal stride overflows".to_owned())
        })?,
        usize::try_from(scale.1).map_err(|_| {
            VideoVaeError::InvalidLtxConfiguration("spatial stride overflows".to_owned())
        })?,
        usize::try_from(scale.1).map_err(|_| {
            VideoVaeError::InvalidLtxConfiguration("spatial stride overflows".to_owned())
        })?,
    ])
}

fn ltx_configuration_modules(
    configuration: &LtxConfiguration,
    backend: &CpuBackend,
    device: sim_tensor::DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<NativeModule>, VideoVaeError> {
    let scalars = [
        1.0,
        configuration.input_channels as f32,
        configuration.output_channels as f32,
        configuration.latent_channels as f32,
        configuration.encoder_base_channels as f32,
        configuration.decoder_base_channels as f32,
        configuration.patch_size as f32,
        if configuration.causal_decoder {
            1.0
        } else {
            0.0
        },
        if configuration.timestep_conditioning {
            1.0
        } else {
            0.0
        },
        configuration.decode_noise_scale,
        configuration.decode_timestep,
        ltx_padding_code(configuration.encoder_spatial_padding),
        ltx_padding_code(configuration.decoder_spatial_padding),
        ltx_norm_code(configuration.norm_layer),
        ltx_latent_log_variance_code(configuration.latent_log_variance),
    ];
    let encoder_blocks = ltx_block_values(&configuration.encoder_blocks);
    let decoder_blocks = ltx_block_values(&configuration.decoder_blocks);
    let mut modules = Vec::with_capacity(3);
    for (name, values) in [
        ("__sim.ltx.configuration", scalars.as_slice()),
        ("__sim.ltx.encoder_blocks", encoder_blocks.as_slice()),
        ("__sim.ltx.decoder_blocks", decoder_blocks.as_slice()),
    ] {
        let tensor = tensor_from_f32_with_backend_exact_native(
            backend,
            &[u64::try_from(values.len()).map_err(|_| VaeError::ShapeOverflow)?],
            values,
            DType::F32,
            device,
            context,
        )
        .map_err(NativeOpsError::from)?;
        modules.push(NativeModule::buffer(name, tensor)?);
    }
    Ok(modules)
}

fn ltx_padding_code(padding: LtxSpatialPadding) -> f32 {
    match padding {
        LtxSpatialPadding::Zeros => 0.0,
        LtxSpatialPadding::Reflect => 1.0,
    }
}

fn ltx_norm_code(norm_layer: LtxNormLayer) -> f32 {
    match norm_layer {
        LtxNormLayer::Group => 0.0,
        LtxNormLayer::Pixel => 1.0,
        LtxNormLayer::Layer => 2.0,
    }
}

fn ltx_latent_log_variance_code(latent_log_variance: LtxLatentLogVariance) -> f32 {
    match latent_log_variance {
        LtxLatentLogVariance::PerChannel => 0.0,
        LtxLatentLogVariance::Uniform => 1.0,
        LtxLatentLogVariance::Constant => 2.0,
    }
}

fn ltx_block_values(blocks: &[LtxBlock]) -> Vec<f32> {
    blocks
        .iter()
        .flat_map(|block| {
            [
                ltx_block_kind_code(block.kind),
                block.layers as f32,
                block.multiplier as f32,
                if block.residual { 1.0 } else { 0.0 },
                if block.inject_noise { 1.0 } else { 0.0 },
            ]
        })
        .collect()
}

fn ltx_block_kind_code(kind: LtxBlockKind) -> f32 {
    match kind {
        LtxBlockKind::Residual => 0.0,
        LtxBlockKind::ResidualChangeChannels => 1.0,
        LtxBlockKind::CompressTime => 2.0,
        LtxBlockKind::CompressSpace => 3.0,
        LtxBlockKind::CompressAll => 4.0,
        LtxBlockKind::CompressAllChangeChannels => 5.0,
        LtxBlockKind::CompressAllResidual => 6.0,
        LtxBlockKind::CompressSpaceResidual => 7.0,
        LtxBlockKind::CompressTimeResidual => 8.0,
    }
}

fn video_3d_convolution_stride(profile: &VaeKernelProfile, name: &str) -> Vec<usize> {
    if matches!(
        profile,
        VaeKernelProfile::Wan21V1 | VaeKernelProfile::Wan22V1
    ) {
        if name.contains(".resample.1.weight") {
            if name.starts_with("encoder.") {
                vec![2, 2]
            } else {
                vec![1, 1]
            }
        } else if name.starts_with("encoder.") && name.ends_with(".time_conv.weight") {
            vec![2, 1, 1]
        } else {
            vec![1; 3]
        }
    } else if profile == &VaeKernelProfile::CosmosV1 {
        if name == "encoder.down.0.downsample.conv1.conv3d.weight" {
            vec![1, 2, 2]
        } else if name == "encoder.down.0.downsample.conv2.conv3d.weight" {
            vec![2, 1, 1]
        } else {
            vec![1, 1, 1]
        }
    } else if profile == &VaeKernelProfile::MochiV1 {
        if name == "encoder.layers.4.layers.0.weight" {
            vec![1, 2, 2]
        } else if name == "encoder.layers.5.layers.0.weight" {
            vec![2, 2, 2]
        } else if name == "encoder.layers.6.layers.0.weight" {
            vec![3, 2, 2]
        } else {
            vec![1, 1, 1]
        }
    } else if matches!(
        profile,
        VaeKernelProfile::LtxVideoV0 { .. }
            | VaeKernelProfile::LtxVideoV1 { .. }
            | VaeKernelProfile::LtxVideoV2 { .. }
    ) {
        if name.starts_with("encoder.down_blocks.")
            && matches!(
                name,
                "encoder.down_blocks.1.conv.weight"
                    | "encoder.down_blocks.4.conv.weight"
                    | "encoder.down_blocks.7.conv.weight"
            )
        {
            vec![2, 2, 2]
        } else {
            vec![1, 1, 1]
        }
    } else if profile == &VaeKernelProfile::CogVideoXV1 {
        if name.contains(".downsamplers.") {
            vec![2, 2]
        } else if name.contains(".upsamplers.") {
            vec![1, 1]
        } else {
            vec![1, 1, 1]
        }
    } else if profile == &VaeKernelProfile::Causal3dV1
        && name.contains(".downsample.conv.conv.weight")
    {
        if name.starts_with("encoder.down.0.") {
            vec![1, 2, 2]
        } else {
            vec![2, 2, 2]
        }
    } else {
        vec![1; 3]
    }
}

fn cosmos_compatibility_buffer(name: &str) -> bool {
    cosmos_compatibility_buffer_spec(name).is_some()
}

fn cosmos_compatibility_buffer_spec(name: &str) -> Option<(Vec<u64>, &'static str)> {
    match name {
        "encoder.patcher3d.wavelets" | "decoder.unpatcher3d.wavelets" => Some((vec![2], "F32")),
        "encoder.patcher3d._arange" | "decoder.unpatcher3d._arange" => Some((vec![2], "I64")),
        "encoder.patcher3d.patch_size_buffer" => Some((vec![1], "I32")),
        _ => None,
    }
}

fn is_carried_convolution_invoked_causally(name: &str) -> bool {
    name.ends_with(".conv.weight")
        && !name.ends_with(".nin_shortcut.conv.weight")
        && ![
            ".q.conv.weight",
            ".k.conv.weight",
            ".v.conv.weight",
            ".proj_out.conv.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn hunyuan_refiner_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let carried = hunyuan_refiner_is_carried(module)?;
    let mut hidden = if carried {
        let frames = input
            .descriptor()
            .shape()
            .get(2)
            .copied()
            .filter(|frames| *frames > 0)
            .ok_or(VaeError::ShapeOverflow)?;
        let usable = frames
            .checked_sub(1)
            .map(|remaining| 1 + remaining / 4 * 4)
            .ok_or(VaeError::ShapeOverflow)?;
        narrow_contiguous(backend, input, 2, 0, usable, context)?
    } else if input.descriptor().shape().get(2) == Some(&1) {
        repeat_temporal(backend, input, 4, context)?
    } else {
        input.clone()
    };
    hidden = refiner_convolution_execute(
        module,
        backend,
        &hidden,
        "encoder.conv_in",
        carried,
        context,
    )?;
    for level in 0..5 {
        for block in 0..2 {
            hidden = refiner_residual_execute(
                module,
                backend,
                &hidden,
                &format!("encoder.down.{level}.block.{block}"),
                carried,
                carried,
                context,
            )?;
        }
        if level < 4 {
            hidden = refiner_downsample(
                module,
                backend,
                &hidden,
                &format!("encoder.down.{level}.downsample.conv"),
                level >= 2,
                carried,
                context,
            )?;
        }
    }
    hidden = refiner_residual_execute(
        module,
        backend,
        &hidden,
        "encoder.mid.block_1",
        carried,
        carried,
        context,
    )?;
    hidden = refiner_attention_execute(
        module,
        backend,
        &hidden,
        "encoder.mid.attn_1",
        carried,
        carried,
        context,
    )?;
    hidden = refiner_residual_execute(
        module,
        backend,
        &hidden,
        "encoder.mid.block_2",
        carried,
        carried,
        context,
    )?;
    let skip = grouped_channel_mean(backend, &hidden, 64, context)?;
    let normalized = refiner_normalize(
        module,
        backend,
        &hidden,
        "encoder.norm_out",
        carried,
        context,
    )?;
    let normalized = silu_tensor(backend, &normalized, context)?;
    let output = refiner_convolution_execute(
        module,
        backend,
        &normalized,
        "encoder.conv_out",
        carried,
        context,
    )?;
    let output = add_tensor(backend, &output, &skip, context)?;
    narrow_contiguous(backend, &output, 1, 0, 32, context)
}

fn hunyuan_refiner_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let carried = hunyuan_refiner_is_carried(module)?;
    let convolution =
        refiner_convolution_execute(module, backend, input, "decoder.conv_in", carried, context)?;
    let residual = repeat_channels_interleave(backend, input, 32, context)?;
    let mut hidden = add_tensor(backend, &convolution, &residual, context)?;
    hidden = refiner_residual_execute(
        module,
        backend,
        &hidden,
        "decoder.mid.block_1",
        carried,
        carried,
        context,
    )?;
    hidden = refiner_attention_execute(
        module,
        backend,
        &hidden,
        "decoder.mid.attn_1",
        carried,
        carried,
        context,
    )?;
    hidden = refiner_residual_execute(
        module,
        backend,
        &hidden,
        "decoder.mid.block_2",
        carried,
        carried,
        context,
    )?;
    for level in 0..5 {
        for block in 0..3 {
            hidden = refiner_residual_execute(
                module,
                backend,
                &hidden,
                &format!("decoder.up.{level}.block.{block}"),
                carried,
                carried,
                context,
            )?;
        }
        if level < 4 {
            hidden = refiner_upsample(
                module,
                backend,
                &hidden,
                &format!("decoder.up.{level}.upsample.conv"),
                level < 2,
                carried,
                context,
            )?;
        }
    }
    hidden = refiner_normalize(
        module,
        backend,
        &hidden,
        "decoder.norm_out",
        carried,
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = refiner_convolution_execute(
        module,
        backend,
        &hidden,
        "decoder.conv_out",
        carried,
        context,
    )?;
    if !carried && input.descriptor().shape().get(2) == Some(&1) {
        let frames = hidden
            .descriptor()
            .shape()
            .get(2)
            .copied()
            .ok_or(VaeError::ShapeOverflow)?;
        return narrow_contiguous(
            backend,
            &hidden,
            2,
            i64::try_from(frames.checked_sub(1).ok_or(VaeError::ShapeOverflow)?)?,
            1,
            context,
        );
    }
    Ok(hidden)
}

fn hunyuan_refiner_is_carried(module: &NativeModule) -> Result<bool, VaeError> {
    let name = module.layer_name();
    if name.contains("HunyuanVideoRefinerV1") {
        Ok(true)
    } else if name.contains("HunyuanImageRefinerV1") {
        Ok(false)
    } else {
        Err(VaeError::KernelProfileMismatch)
    }
}

fn refiner_convolution_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    carried: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let name = if carried {
        format!("{prefix}.conv.weight")
    } else {
        format!("{prefix}.weight")
    };
    let input = if carried {
        causal_replicate_pad_3d(backend, input, context)?
    } else {
        input.clone()
    };
    convolution(module, backend, &input, &name, context)
}

fn refiner_residual_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    carried: bool,
    rms: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shortcut = if carried {
        format!("{prefix}.nin_shortcut.conv.weight")
    } else {
        format!("{prefix}.nin_shortcut.weight")
    };
    let residual = if crate::vae_image::find_module(module, &shortcut).is_some() {
        convolution(module, backend, input, &shortcut, context)?
    } else {
        input.clone()
    };
    let mut hidden = refiner_normalize(
        module,
        backend,
        input,
        &format!("{prefix}.norm1"),
        rms,
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = refiner_convolution_execute(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv1"),
        carried,
        context,
    )?;
    hidden = refiner_normalize(
        module,
        backend,
        &hidden,
        &format!("{prefix}.norm2"),
        rms,
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = refiner_convolution_execute(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2"),
        carried,
        context,
    )?;
    add_tensor(backend, &residual, &hidden, context)
}

fn refiner_attention_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    _carried: bool,
    rms: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if !rms {
        return attention_block(module, backend, input, prefix, context);
    }
    let normalized = rms_channel_norm(module, backend, input, &format!("{prefix}.norm"), context)?;
    attention_block_from_normalized(module, backend, input, &normalized, prefix, context)
}

fn refiner_normalize(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    rms: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if rms {
        rms_channel_norm(module, backend, input, prefix, context)
    } else {
        group_norm(module, backend, input, prefix, context)
    }
}

fn causal3d_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = input
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .filter(|frames| *frames > 0)
        .ok_or(VaeError::ShapeOverflow)?;
    let usable = 1 + frames.saturating_sub(1) / 4 * 4;
    let input = narrow_contiguous(backend, input, 2, 0, usable, context)?;
    let mut hidden =
        refiner_convolution_execute(module, backend, &input, "encoder.conv_in", true, context)?;
    for level in 0..4 {
        for block in 0..2 {
            hidden = refiner_residual_execute(
                module,
                backend,
                &hidden,
                &format!("encoder.down.{level}.block.{block}"),
                true,
                false,
                context,
            )?;
        }
        if level < 3 {
            hidden = refiner_convolution_execute(
                module,
                backend,
                &hidden,
                &format!("encoder.down.{level}.downsample.conv"),
                true,
                context,
            )?;
        }
    }
    hidden = refiner_residual_execute(
        module,
        backend,
        &hidden,
        "encoder.mid.block_1",
        true,
        false,
        context,
    )?;
    hidden = refiner_attention_execute(
        module,
        backend,
        &hidden,
        "encoder.mid.attn_1",
        true,
        false,
        context,
    )?;
    hidden = refiner_residual_execute(
        module,
        backend,
        &hidden,
        "encoder.mid.block_2",
        true,
        false,
        context,
    )?;
    hidden = group_norm(module, backend, &hidden, "encoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden =
        refiner_convolution_execute(module, backend, &hidden, "encoder.conv_out", true, context)?;
    hidden = convolution(module, backend, &hidden, "quant_conv.weight", context)?;
    narrow_contiguous(backend, &hidden, 1, 0, 4, context)
}

fn causal3d_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = convolution(module, backend, input, "post_quant_conv.weight", context)?;
    hidden =
        refiner_convolution_execute(module, backend, &hidden, "decoder.conv_in", true, context)?;
    hidden = refiner_residual_execute(
        module,
        backend,
        &hidden,
        "decoder.mid.block_1",
        true,
        false,
        context,
    )?;
    hidden = refiner_attention_execute(
        module,
        backend,
        &hidden,
        "decoder.mid.attn_1",
        true,
        false,
        context,
    )?;
    hidden = refiner_residual_execute(
        module,
        backend,
        &hidden,
        "decoder.mid.block_2",
        true,
        false,
        context,
    )?;
    for level in (0..4).rev() {
        for block in 0..3 {
            hidden = refiner_residual_execute(
                module,
                backend,
                &hidden,
                &format!("decoder.up.{level}.block.{block}"),
                true,
                false,
                context,
            )?;
        }
        if level > 0 {
            hidden = causal3d_nearest_upsample(backend, &hidden, level <= 2, context)?;
            hidden = refiner_convolution_execute(
                module,
                backend,
                &hidden,
                &format!("decoder.up.{level}.upsample.conv"),
                true,
                context,
            )?;
        }
    }
    hidden = group_norm(module, backend, &hidden, "decoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    refiner_convolution_execute(module, backend, &hidden, "decoder.conv_out", true, context)
}

fn causal3d_nearest_upsample(
    backend: &dyn TensorBackend,
    input: &Tensor,
    temporal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = split_video_frames(backend, input, context)?;
    let mut output = Vec::new();
    for (index, frame) in frames.into_iter().enumerate() {
        let frame = nearest_upsample_2x(backend, &frame, context)?;
        output.push(frame.clone());
        if temporal && index > 0 {
            output.push(frame);
        }
    }
    stack_video_frames(backend, &output, context)
}

fn cogvideox_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden =
        cog_causal_convolution_execute(module, backend, input, "encoder.conv_in", context)?;
    for level in 0..4 {
        for block in 0..3 {
            hidden = cog_residual_execute(
                module,
                backend,
                &hidden,
                None,
                &format!("encoder.down_blocks.{level}.resnets.{block}"),
                context,
            )?;
        }
        if level < 3 {
            hidden = cog_downsample(
                module,
                backend,
                &hidden,
                &format!("encoder.down_blocks.{level}.downsamplers.0.conv.weight"),
                level < 2,
                context,
            )?;
        }
    }
    for block in 0..2 {
        hidden = cog_residual_execute(
            module,
            backend,
            &hidden,
            None,
            &format!("encoder.mid_block.resnets.{block}"),
            context,
        )?;
    }
    hidden = group_norm(module, backend, &hidden, "encoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = cog_causal_convolution_execute(module, backend, &hidden, "encoder.conv_out", context)?;
    narrow_contiguous(backend, &hidden, 1, 0, 16, context)
}

fn cogvideox_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden =
        cog_causal_convolution_execute(module, backend, input, "decoder.conv_in", context)?;
    for block in 0..2 {
        hidden = cog_residual_execute(
            module,
            backend,
            &hidden,
            Some(input),
            &format!("decoder.mid_block.resnets.{block}"),
            context,
        )?;
    }
    for level in 0..4 {
        for block in 0..4 {
            hidden = cog_residual_execute(
                module,
                backend,
                &hidden,
                Some(input),
                &format!("decoder.up_blocks.{level}.resnets.{block}"),
                context,
            )?;
        }
        if level < 3 {
            hidden = cog_upsample(
                module,
                backend,
                &hidden,
                &format!("decoder.up_blocks.{level}.upsamplers.0.conv.weight"),
                level < 2,
                context,
            )?;
        }
    }
    hidden =
        cog_spatial_norm_execute(module, backend, &hidden, input, "decoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    cog_causal_convolution_execute(module, backend, &hidden, "decoder.conv_out", context)
}

fn cog_causal_convolution_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let name = format!("{prefix}.conv.weight");
    let convolution_module = crate::vae_image::find_module(module, &name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing video VAE module {name}"
        )))
    })?;
    let (weight, _) = convolution_module.dense_parameters()?;
    let kernel = weight
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let input = if kernel > 1 {
        first_frame_temporal_pad(backend, input, kernel - 1, context)?
    } else {
        input.clone()
    };
    convolution(module, backend, &input, &name, context)
}

fn cog_residual_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    latent: Option<&Tensor>,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shortcut = format!("{prefix}.conv_shortcut.weight");
    let residual = if crate::vae_image::find_module(module, &shortcut).is_some() {
        convolution(module, backend, input, &shortcut, context)?
    } else {
        input.clone()
    };
    let mut hidden = if let Some(latent) = latent {
        cog_spatial_norm_execute(
            module,
            backend,
            input,
            latent,
            &format!("{prefix}.norm1"),
            context,
        )?
    } else {
        group_norm(module, backend, input, &format!("{prefix}.norm1"), context)?
    };
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = cog_causal_convolution_execute(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv1"),
        context,
    )?;
    hidden = if let Some(latent) = latent {
        cog_spatial_norm_execute(
            module,
            backend,
            &hidden,
            latent,
            &format!("{prefix}.norm2"),
            context,
        )?
    } else {
        group_norm(
            module,
            backend,
            &hidden,
            &format!("{prefix}.norm2"),
            context,
        )?
    };
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = cog_causal_convolution_execute(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2"),
        context,
    )?;
    add_tensor(backend, &residual, &hidden, context)
}

fn cog_spatial_norm_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    features: &Tensor,
    latent: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let normalized = group_norm(
        module,
        backend,
        features,
        &format!("{prefix}.norm_layer"),
        context,
    )?;
    let shape = features.descriptor().shape();
    let latent = cog_interpolate_latent(backend, latent, [shape[2], shape[3], shape[4]], context)?;
    let scale = cog_causal_convolution_execute(
        module,
        backend,
        &latent,
        &format!("{prefix}.conv_y"),
        context,
    )?;
    let bias = cog_causal_convolution_execute(
        module,
        backend,
        &latent,
        &format!("{prefix}.conv_b"),
        context,
    )?;
    let (scaled, event) = backend.binary(
        BinaryOperation::Multiply,
        &normalized,
        &scale,
        normalized.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    add_tensor(backend, &scaled, &bias, context)
}

fn cog_downsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    temporal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let input = if temporal {
        cog_temporal_average_pool(backend, input, context)?
    } else {
        input.clone()
    };
    let frames = split_video_frames(backend, &input, context)?;
    let frames = frames
        .into_iter()
        .map(|frame| {
            let frame = constant_pad_bottom_right(backend, &frame, context)?;
            convolution(module, backend, &frame, name, context)
        })
        .collect::<Result<Vec<_>, VaeError>>()?;
    stack_video_frames(backend, &frames, context)
}

fn cog_temporal_average_pool(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = split_video_frames(backend, input, context)?;
    let odd = !frames.len().is_multiple_of(2);
    let mut output = Vec::new();
    let start = if odd {
        output.push(frames.first().cloned().ok_or(VaeError::ShapeOverflow)?);
        1
    } else {
        0
    };
    for pair in frames[start..].chunks_exact(2) {
        let sum = add_tensor(backend, &pair[0], &pair[1], context)?;
        let (mean, event) = backend.binary_scalar(
            BinaryOperation::Multiply,
            &sum,
            Scalar::Float(0.5),
            ScalarSide::Right,
            sum.descriptor().clone(),
            context,
        )?;
        backend.wait_event(event, context)?;
        output.push(mean);
    }
    stack_video_frames(backend, &output, context)
}

fn cog_upsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    temporal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = split_video_frames(backend, input, context)?;
    let odd_special = temporal && frames.len() > 1 && !frames.len().is_multiple_of(2);
    let repeat_all = temporal && frames.len() > 1 && !odd_special;
    let mut output = Vec::new();
    for (index, frame) in frames.into_iter().enumerate() {
        let frame = nearest_upsample_2x(backend, &frame, context)?;
        let frame = convolution(module, backend, &frame, name, context)?;
        output.push(frame.clone());
        if repeat_all || (odd_special && index > 0) {
            output.push(frame);
        }
    }
    stack_video_frames(backend, &output, context)
}

fn cog_interpolate_latent(
    backend: &dyn TensorBackend,
    input: &Tensor,
    target: [u64; 3],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5
        || target.contains(&0)
        || target[0] < shape[2]
        || target[1] < shape[3]
        || target[2] < shape[4]
        || !target[1].is_multiple_of(shape[3])
        || !target[2].is_multiple_of(shape[4])
    {
        return Err(VaeError::ShapeOverflow);
    }
    let mut frames = split_video_frames(backend, input, context)?;
    while frames
        .first()
        .is_some_and(|frame| frame.descriptor().shape()[2] < target[1])
    {
        frames = frames
            .into_iter()
            .map(|frame| nearest_upsample_2x(backend, &frame, context))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if frames
        .first()
        .is_none_or(|frame| frame.descriptor().shape()[2..] != target[1..])
    {
        return Err(VaeError::ShapeOverflow);
    }
    let source_frames = u64::try_from(frames.len())?;
    let mut temporal = Vec::new();
    if target[0] > 1 && !target[0].is_multiple_of(2) {
        temporal.push(frames.first().cloned().ok_or(VaeError::ShapeOverflow)?);
        let source_rest = source_frames
            .checked_sub(1)
            .ok_or(VaeError::ShapeOverflow)?;
        let target_rest = target[0].checked_sub(1).ok_or(VaeError::ShapeOverflow)?;
        if source_rest == 0 || !target_rest.is_multiple_of(source_rest) {
            return Err(VaeError::ShapeOverflow);
        }
        let repeats = target_rest / source_rest;
        for frame in frames.into_iter().skip(1) {
            temporal.extend(std::iter::repeat_n(frame, usize::try_from(repeats)?));
        }
    } else {
        if !target[0].is_multiple_of(source_frames) {
            return Err(VaeError::ShapeOverflow);
        }
        let repeats = target[0] / source_frames;
        for frame in frames {
            temporal.extend(std::iter::repeat_n(frame, usize::try_from(repeats)?));
        }
    }
    stack_video_frames(backend, &temporal, context)
}

fn first_frame_temporal_pad(
    backend: &dyn TensorBackend,
    input: &Tensor,
    padding: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if padding == 0 {
        return Ok(input.clone());
    }
    let first = narrow_contiguous(backend, input, 2, 0, 1, context)?;
    let mut inputs = Vec::new();
    inputs
        .try_reserve_exact(usize::try_from(padding)?.saturating_add(1))
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    inputs.extend(std::iter::repeat_n(first, usize::try_from(padding)?));
    inputs.push(input.clone());
    concatenate_temporal(backend, &inputs, context)
}

fn cosmos_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = input
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .filter(|frames| *frames > 0)
        .ok_or(VaeError::ShapeOverflow)?;
    let usable = 1 + frames.saturating_sub(1) / 8 * 8;
    let input = narrow_contiguous(backend, input, 2, 0, usable, context)?;
    let mut hidden = cosmos_haar_patchify(backend, &input, context)?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "encoder.conv_in.0.conv3d.weight",
        context,
    )?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "encoder.conv_in.1.conv3d.weight",
        context,
    )?;
    for level in 0..3 {
        for block in 0..2 {
            hidden = cosmos_residual_execute(
                module,
                backend,
                &hidden,
                &format!("encoder.down.{level}.block.{block}"),
                context,
            )?;
        }
        if level == 0 {
            hidden = cosmos_hybrid_downsample(
                module,
                backend,
                &hidden,
                "encoder.down.0.downsample",
                context,
            )?;
        }
    }
    hidden = cosmos_residual_execute(module, backend, &hidden, "encoder.mid.block_1", context)?;
    hidden =
        cosmos_attention_pair_execute(module, backend, &hidden, "encoder.mid.attn_1", context)?;
    hidden = cosmos_residual_execute(module, backend, &hidden, "encoder.mid.block_2", context)?;
    hidden = cosmos_normalize(module, backend, &hidden, "encoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "encoder.conv_out.0.conv3d.weight",
        context,
    )?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "encoder.conv_out.1.conv3d.weight",
        context,
    )?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "quant_conv.conv3d.weight",
        context,
    )?;
    cosmos_latent_affine(module, backend, &hidden, true, context)
}

fn cosmos_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = cosmos_latent_affine(module, backend, input, false, context)?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "post_quant_conv.conv3d.weight",
        context,
    )?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "decoder.conv_in.0.conv3d.weight",
        context,
    )?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "decoder.conv_in.1.conv3d.weight",
        context,
    )?;
    hidden = cosmos_residual_execute(module, backend, &hidden, "decoder.mid.block_1", context)?;
    hidden =
        cosmos_attention_pair_execute(module, backend, &hidden, "decoder.mid.attn_1", context)?;
    hidden = cosmos_residual_execute(module, backend, &hidden, "decoder.mid.block_2", context)?;
    for level in (0..3).rev() {
        for block in 0..3 {
            hidden = cosmos_residual_execute(
                module,
                backend,
                &hidden,
                &format!("decoder.up.{level}.block.{block}"),
                context,
            )?;
        }
        if level == 1 {
            hidden =
                cosmos_hybrid_upsample(module, backend, &hidden, "decoder.up.1.upsample", context)?;
        }
    }
    hidden = cosmos_normalize(module, backend, &hidden, "decoder.norm_out", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "decoder.conv_out.0.conv3d.weight",
        context,
    )?;
    hidden = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        "decoder.conv_out.1.conv3d.weight",
        context,
    )?;
    cosmos_haar_unpatchify(backend, &hidden, context)
}

fn cosmos_causal_convolution(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let convolution_module = crate::vae_image::find_module(module, name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing Cosmos video VAE module {name}"
        )))
    })?;
    let (weight, _) = convolution_module.dense_parameters()?;
    let kernel = weight
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let temporal_padding = if name == "encoder.down.0.downsample.conv2.conv3d.weight" {
        kernel.saturating_sub(2)
    } else {
        kernel.saturating_sub(1)
    };
    let input = first_frame_temporal_pad(backend, input, temporal_padding, context)?;
    convolution(module, backend, &input, name, context)
}

fn cosmos_normalize(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let weight_name = format!("{prefix}.norm.weight");
    let bias_name = format!("{prefix}.norm.bias");
    let weight = crate::vae_image::find_module(module, &weight_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Cosmos normalization buffer {weight_name}"
            )))
        })?;
    let bias = crate::vae_image::find_module(module, &bias_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Cosmos normalization buffer {bias_name}"
            )))
        })?;
    let frames = split_video_frames(backend, input, context)?;
    let frames = frames
        .into_iter()
        .map(|frame| {
            group_norm_tensor_with_context_exact_native(
                backend,
                &frame,
                1,
                Some(weight),
                Some(bias),
                1.0e-6,
                context,
            )
            .map_err(NativeOpsError::from)
            .map_err(VaeError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    stack_video_frames(backend, &frames, context)
}

fn cosmos_residual_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shortcut_name = format!("{prefix}.nin_shortcut.conv3d.weight");
    let residual = if crate::vae_image::find_module(module, &shortcut_name).is_some() {
        cosmos_causal_convolution(module, backend, input, &shortcut_name, context)?
    } else {
        input.clone()
    };
    let mut hidden = cosmos_normalize(module, backend, input, &format!("{prefix}.norm1"), context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    for convolution_index in 0..2 {
        hidden = cosmos_causal_convolution(
            module,
            backend,
            &hidden,
            &format!("{prefix}.conv1.{convolution_index}.conv3d.weight"),
            context,
        )?;
    }
    hidden = cosmos_normalize(
        module,
        backend,
        &hidden,
        &format!("{prefix}.norm2"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    for convolution_index in 0..2 {
        hidden = cosmos_causal_convolution(
            module,
            backend,
            &hidden,
            &format!("{prefix}.conv2.{convolution_index}.conv3d.weight"),
            context,
        )?;
    }
    add_tensor(backend, &residual, &hidden, context)
}

fn cosmos_attention_pair_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let attention_prefix = format!("{prefix}.0");
    let normalized = cosmos_normalize(
        module,
        backend,
        input,
        &format!("{attention_prefix}.norm"),
        context,
    )?;
    let query = cosmos_causal_convolution(
        module,
        backend,
        &normalized,
        &format!("{attention_prefix}.q.conv3d.weight"),
        context,
    )?;
    let key = cosmos_causal_convolution(
        module,
        backend,
        &normalized,
        &format!("{attention_prefix}.k.conv3d.weight"),
        context,
    )?;
    let value = cosmos_causal_convolution(
        module,
        backend,
        &normalized,
        &format!("{attention_prefix}.v.conv3d.weight"),
        context,
    )?;
    let frames = split_video_frames(backend, input, context)?;
    let query_frames = split_video_frames(backend, &query, context)?;
    let key_frames = split_video_frames(backend, &key, context)?;
    let value_frames = split_video_frames(backend, &value, context)?;
    let attended = frames
        .into_iter()
        .zip(query_frames)
        .zip(key_frames)
        .zip(value_frames)
        .map(|(((frame, query), key), value)| {
            let attended =
                spatial_attention_from_qkv(backend, &frame, &query, &key, &value, context)?;
            Ok(attended)
        })
        .collect::<Result<Vec<_>, VaeError>>()?;
    let attended = stack_video_frames(backend, &attended, context)?;
    let projected = cosmos_causal_convolution(
        module,
        backend,
        &attended,
        &format!("{attention_prefix}.proj_out.conv3d.weight"),
        context,
    )?;
    let spatial = add_tensor(backend, input, &projected, context)?;
    cosmos_temporal_attention(module, backend, &spatial, &format!("{prefix}.1"), context)
}

fn cosmos_temporal_attention(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let normalized = cosmos_normalize(module, backend, input, &format!("{prefix}.norm"), context)?;
    let mut projections = Vec::new();
    for projection in ["q", "k", "v"] {
        projections.push(cosmos_causal_convolution(
            module,
            backend,
            &normalized,
            &format!("{prefix}.{projection}.conv3d.weight"),
            context,
        )?);
    }
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape.contains(&0) {
        return Err(VaeError::ShapeOverflow);
    }
    let batch_space = shape[0]
        .checked_mul(shape[3])
        .and_then(|value| value.checked_mul(shape[4]))
        .ok_or(VaeError::ShapeOverflow)?;
    let sequence = |tensor: &Tensor| -> Result<Tensor, VaeError> {
        let permuted = permute_read_only(tensor, &[0, 3, 4, 1, 2])?;
        let contiguous = contiguous_copy(backend, &permuted, context)?;
        reshape_read_only(&contiguous, vec![batch_space, shape[1], shape[2]])
    };
    let query = sequence(&projections[0])?;
    let query = permute_read_only(&query, &[0, 2, 1])?;
    let key = sequence(&projections[1])?;
    let value = sequence(&projections[2])?;
    let mut attended_frames = Vec::new();
    attended_frames
        .try_reserve_exact(usize::try_from(shape[2])?)
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    for frame in 0..shape[2] {
        context.check()?;
        let length = frame.checked_add(1).ok_or(VaeError::ShapeOverflow)?;
        let query_frame = query.narrow_read_only(1, i64::try_from(frame)?, 1)?;
        let key_prefix = key.narrow_read_only(2, 0, length)?;
        let score_descriptor = TensorDescriptor::contiguous(
            vec![batch_space, 1, length],
            input.descriptor().dtype(),
            input.descriptor().device(),
            context.stream,
        )?;
        let (scores, event) = backend.linear_algebra(
            LinearAlgebraOperation::BatchMatrixMultiply,
            &[query_frame, key_prefix],
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
        let scores = softmax_tensor_with_context_exact_native(backend, &scores, -1, context)
            .map_err(NativeOpsError::from)?;
        let scores = permute_read_only(&scores, &[0, 2, 1])?;
        let value_prefix = value.narrow_read_only(2, 0, length)?;
        let output_descriptor = TensorDescriptor::contiguous(
            vec![batch_space, shape[1], 1],
            input.descriptor().dtype(),
            input.descriptor().device(),
            context.stream,
        )?;
        let (attended, event) = backend.linear_algebra(
            LinearAlgebraOperation::BatchMatrixMultiply,
            &[value_prefix, scores],
            output_descriptor,
            context,
        )?;
        backend.wait_event(event, context)?;
        attended_frames.push(attended);
    }
    let attended = concatenate_dimension(backend, &attended_frames, 2, context)?;
    let attended = reshape_read_only(
        &attended,
        vec![shape[0], shape[3], shape[4], shape[1], shape[2]],
    )?;
    let attended = permute_read_only(&attended, &[0, 3, 4, 1, 2])?;
    let attended = contiguous_copy(backend, &attended, context)?;
    let projected = cosmos_causal_convolution(
        module,
        backend,
        &attended,
        &format!("{prefix}.proj_out.conv3d.weight"),
        context,
    )?;
    add_tensor(backend, input, &projected, context)
}

fn cosmos_hybrid_downsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = split_video_frames(backend, input, context)?;
    let padded_frames = frames
        .iter()
        .map(|frame| constant_pad_bottom_right(backend, frame, context))
        .collect::<Result<Vec<_>, _>>()?;
    let padded = stack_video_frames(backend, &padded_frames, context)?;
    let convolution = cosmos_causal_convolution(
        module,
        backend,
        &padded,
        &format!("{prefix}.conv1.conv3d.weight"),
        context,
    )?;
    let average = cosmos_average_pool(backend, &padded, false, context)?;
    let mut hidden = add_tensor(backend, &convolution, &average, context)?;

    let temporal_input = first_frame_temporal_pad(backend, &hidden, 1, context)?;
    let convolution = cosmos_causal_convolution(
        module,
        backend,
        &temporal_input,
        &format!("{prefix}.conv2.conv3d.weight"),
        context,
    )?;
    let average = cosmos_average_pool(backend, &temporal_input, true, context)?;
    hidden = add_tensor(backend, &convolution, &average, context)?;
    cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv3.conv3d.weight"),
        context,
    )
}

fn cosmos_hybrid_upsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = input
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .filter(|frames| *frames > 0)
        .ok_or(VaeError::ShapeOverflow)?;
    let mut hidden = if frames > 1 {
        let repeated = cosmos_repeat_temporal_interleave(backend, input, 2, context)?;
        narrow_contiguous(
            backend,
            &repeated,
            2,
            1,
            frames
                .checked_mul(2)
                .and_then(|value| value.checked_sub(1))
                .ok_or(VaeError::ShapeOverflow)?,
            context,
        )?
    } else {
        input.clone()
    };
    let temporal = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv1.conv3d.weight"),
        context,
    )?;
    hidden = add_tensor(backend, &hidden, &temporal, context)?;
    let spatial = split_video_frames(backend, &hidden, context)?
        .into_iter()
        .map(|frame| nearest_upsample_2x(backend, &frame, context))
        .collect::<Result<Vec<_>, _>>()?;
    hidden = stack_video_frames(backend, &spatial, context)?;
    let convolved = cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2.conv3d.weight"),
        context,
    )?;
    hidden = add_tensor(backend, &hidden, &convolved, context)?;
    cosmos_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv3.conv3d.weight"),
        context,
    )
}

fn cosmos_average_pool(
    backend: &dyn TensorBackend,
    input: &Tensor,
    temporal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5
        || shape.contains(&0)
        || (temporal && !shape[2].is_multiple_of(2))
        || (!temporal && (!shape[3].is_multiple_of(2) || !shape[4].is_multiple_of(2)))
    {
        return Err(VaeError::ShapeOverflow);
    }
    let (grouped, dimensions, output_shape) = if temporal {
        (
            reshape_read_only(
                input,
                vec![shape[0], shape[1], shape[2] / 2, 2, shape[3], shape[4]],
            )?,
            vec![3],
            vec![shape[0], shape[1], shape[2] / 2, shape[3], shape[4]],
        )
    } else {
        (
            reshape_read_only(
                input,
                vec![
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[3] / 2,
                    2,
                    shape[4] / 2,
                    2,
                ],
            )?,
            vec![4, 6],
            vec![shape[0], shape[1], shape[2], shape[3] / 2, shape[4] / 2],
        )
    };
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Mean,
            dimensions,
            keep_dimensions: false,
            accumulation_dtype: Some(input.descriptor().dtype()),
            correction: 0,
        },
        &grouped,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn cosmos_repeat_temporal_interleave(
    backend: &dyn TensorBackend,
    input: &Tensor,
    repeats: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = split_video_frames(backend, input, context)?;
    let mut repeated = Vec::new();
    repeated
        .try_reserve_exact(
            frames
                .len()
                .checked_mul(usize::try_from(repeats)?)
                .ok_or(VaeError::ShapeOverflow)?,
        )
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    for frame in frames {
        repeated.extend(std::iter::repeat_n(frame, usize::try_from(repeats)?));
    }
    stack_video_frames(backend, &repeated, context)
}

fn cosmos_haar_patchify(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape.contains(&0) {
        return Err(VaeError::ShapeOverflow);
    }
    let first = narrow_contiguous(backend, input, 2, 0, 1, context)?;
    let first = repeat_temporal(backend, &first, 4, context)?;
    let input = if shape[2] > 1 {
        let rest = narrow_contiguous(backend, input, 2, 1, shape[2] - 1, context)?;
        concatenate_temporal(backend, &[first, rest], context)?
    } else {
        first
    };
    let first = cosmos_haar_downsample(backend, &input, context)?;
    cosmos_haar_downsample(backend, &first, context)
}

fn cosmos_haar_unpatchify(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let first = cosmos_haar_upsample(backend, input, context)?;
    let output = cosmos_haar_upsample(backend, &first, context)?;
    let frames = output
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    narrow_contiguous(
        backend,
        &output,
        2,
        3,
        frames.checked_sub(3).ok_or(VaeError::ShapeOverflow)?,
        context,
    )
}

fn cosmos_haar_downsample(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5
        || shape.contains(&0)
        || !shape[2].is_multiple_of(2)
        || !shape[3].is_multiple_of(2)
        || !shape[4].is_multiple_of(2)
    {
        return Err(VaeError::ShapeOverflow);
    }
    let grouped = reshape_read_only(
        input,
        vec![
            shape[0],
            shape[1],
            shape[2] / 2,
            2,
            shape[3] / 2,
            2,
            shape[4] / 2,
            2,
        ],
    )?;
    let grouped = permute_read_only(&grouped, &[0, 3, 5, 7, 1, 2, 4, 6])?;
    let grouped = contiguous_copy(backend, &grouped, context)?;
    let grouped = reshape_read_only(
        &grouped,
        vec![
            shape[0],
            shape[1].checked_mul(8).ok_or(VaeError::ShapeOverflow)?,
            shape[2] / 2,
            shape[3] / 2,
            shape[4] / 2,
        ],
    )?;
    let mut components = Vec::new();
    for output_component in 0_u64..8 {
        let mut coefficient = narrow_contiguous(backend, &grouped, 1, 0, shape[1], context)?;
        for input_component in 1_u64..8 {
            let input = narrow_contiguous(
                backend,
                &grouped,
                1,
                i64::try_from(
                    input_component
                        .checked_mul(shape[1])
                        .ok_or(VaeError::ShapeOverflow)?,
                )?,
                shape[1],
                context,
            )?;
            let subtract = (output_component & input_component).count_ones() % 2 == 1;
            coefficient = binary_tensor(
                backend,
                if subtract {
                    BinaryOperation::Subtract
                } else {
                    BinaryOperation::Add
                },
                &coefficient,
                &input,
                context,
            )?;
        }
        components.push(affine_tensor(backend, &coefficient, 0.125, 0.0, context)?);
    }
    concatenate_dimension(backend, &components, 1, context)
}

fn cosmos_haar_upsample(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape.contains(&0) || !shape[1].is_multiple_of(8) {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = shape[1] / 8;
    let mut samples = Vec::new();
    for output_component in 0_u64..8 {
        let mut sample = narrow_contiguous(backend, input, 1, 0, channels, context)?;
        for input_component in 1_u64..8 {
            let coefficient = narrow_contiguous(
                backend,
                input,
                1,
                i64::try_from(
                    input_component
                        .checked_mul(channels)
                        .ok_or(VaeError::ShapeOverflow)?,
                )?,
                channels,
                context,
            )?;
            let subtract = (output_component & input_component).count_ones() % 2 == 1;
            sample = binary_tensor(
                backend,
                if subtract {
                    BinaryOperation::Subtract
                } else {
                    BinaryOperation::Add
                },
                &sample,
                &coefficient,
                context,
            )?;
        }
        samples.push(sample);
    }
    let packed = concatenate_dimension(backend, &samples, 1, context)?;
    let packed = reshape_read_only(
        &packed,
        vec![shape[0], 2, 2, 2, channels, shape[2], shape[3], shape[4]],
    )?;
    let packed = permute_read_only(&packed, &[0, 4, 5, 1, 6, 2, 7, 3])?;
    let packed = contiguous_copy(backend, &packed, context)?;
    reshape_read_only(
        &packed,
        vec![
            shape[0],
            channels,
            shape[2].checked_mul(2).ok_or(VaeError::ShapeOverflow)?,
            shape[3].checked_mul(2).ok_or(VaeError::ShapeOverflow)?,
            shape[4].checked_mul(2).ok_or(VaeError::ShapeOverflow)?,
        ],
    )
}

fn cosmos_latent_affine(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    encode: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mean = cosmos_latent_stat(module, backend, input, "latent_mean", context)?;
    let standard_deviation = cosmos_latent_stat(module, backend, input, "latent_std", context)?;
    if encode {
        let centered = binary_tensor(backend, BinaryOperation::Subtract, input, &mean, context)?;
        let normalized = binary_tensor(
            backend,
            BinaryOperation::Divide,
            &centered,
            &standard_deviation,
            context,
        )?;
        affine_tensor(backend, &normalized, 0.5, 0.0, context)
    } else {
        let scaled = affine_tensor(backend, input, 2.0, 0.0, context)?;
        let scaled = binary_tensor(
            backend,
            BinaryOperation::Multiply,
            &scaled,
            &standard_deviation,
            context,
        )?;
        binary_tensor(backend, BinaryOperation::Add, &scaled, &mean, context)
    }
}

fn cosmos_latent_stat(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape[1] != 16 || shape[2] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let buffer = crate::vae_image::find_module(module, name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Cosmos latent buffer {name}"
            )))
        })?;
    if buffer.descriptor().shape() != [256] {
        return Err(VaeError::ShapeOverflow);
    }
    let buffer = reshape_read_only(buffer, vec![1, 16, 16, 1, 1])?;
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(usize::try_from(shape[2])?)
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    for frame in 0..shape[2] {
        frames.push(narrow_contiguous(
            backend,
            &buffer,
            2,
            i64::try_from(frame % 16)?,
            1,
            context,
        )?);
    }
    concatenate_temporal(backend, &frames, context)
}

fn binary_tensor(
    backend: &dyn TensorBackend,
    operation: BinaryOperation,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (output, event) =
        backend.binary(operation, left, right, left.descriptor().clone(), context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn concatenate_dimension(
    backend: &dyn TensorBackend,
    inputs: &[Tensor],
    dimension: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let first = inputs.first().ok_or(VaeError::ShapeOverflow)?;
    let shape = first.descriptor().shape();
    if dimension >= shape.len() {
        return Err(VaeError::ShapeOverflow);
    }
    let extent = inputs.iter().try_fold(0_u64, |extent, input| {
        let input_shape = input.descriptor().shape();
        if input_shape.len() != shape.len()
            || input_shape
                .iter()
                .enumerate()
                .any(|(axis, value)| axis != dimension && *value != shape[axis])
            || input.descriptor().dtype() != first.descriptor().dtype()
            || input.descriptor().device() != first.descriptor().device()
        {
            return Err(VaeError::ShapeOverflow);
        }
        extent
            .checked_add(input_shape[dimension])
            .ok_or(VaeError::ShapeOverflow)
    })?;
    let mut output_shape = shape.to_vec();
    output_shape[dimension] = extent;
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        first.descriptor().dtype(),
        first.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let mut offset = 0_u64;
    for input in inputs {
        context.check()?;
        let mut offsets = vec![0; shape.len()];
        offsets[dimension] = offset;
        let (updated, event) =
            backend.replace_rectangular_slice(&output, input, &offsets, context)?;
        backend.wait_event(event, context)?;
        output = updated;
        offset = offset
            .checked_add(input.descriptor().shape()[dimension])
            .ok_or(VaeError::ShapeOverflow)?;
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LtxVersion {
    V0,
    V1,
    V2,
}

fn ltx_version(module: &NativeModule) -> Result<LtxVersion, VaeError> {
    let name = module.layer_name();
    if name.contains("LtxVideoV0") {
        Ok(LtxVersion::V0)
    } else if name.contains("LtxVideoV1") {
        Ok(LtxVersion::V1)
    } else if name.contains("LtxVideoV2") {
        Ok(LtxVersion::V2)
    } else {
        Err(VaeError::KernelProfileMismatch)
    }
}

fn ltx_configuration_from_module(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    context: &ExecutionContext<'_>,
) -> Result<LtxConfiguration, VaeError> {
    let values = tensor_to_f32_with_backend_exact_native(
        backend,
        ltx_buffer(module, "__sim.ltx.configuration")?,
        context,
    )
    .map_err(NativeOpsError::from)?;
    if values.len() != 15 || values[0] != 1.0 {
        return Err(VaeError::KernelProfileMismatch);
    }
    let encoder_blocks = ltx_blocks_from_module(module, backend, "encoder", context)?;
    let decoder_blocks = ltx_blocks_from_module(module, backend, "decoder", context)?;
    let configuration = LtxConfiguration {
        input_channels: ltx_configuration_u64(&values, 1)?,
        output_channels: ltx_configuration_u64(&values, 2)?,
        latent_channels: ltx_configuration_u64(&values, 3)?,
        encoder_base_channels: ltx_configuration_u64(&values, 4)?,
        decoder_base_channels: ltx_configuration_u64(&values, 5)?,
        patch_size: ltx_configuration_u64(&values, 6)?,
        norm_layer: match ltx_configuration_nonnegative_u64(&values, 13)? {
            0 => LtxNormLayer::Group,
            1 => LtxNormLayer::Pixel,
            2 => LtxNormLayer::Layer,
            _ => return Err(VaeError::KernelProfileMismatch),
        },
        latent_log_variance: match ltx_configuration_nonnegative_u64(&values, 14)? {
            0 => LtxLatentLogVariance::PerChannel,
            1 => LtxLatentLogVariance::Uniform,
            2 => LtxLatentLogVariance::Constant,
            _ => return Err(VaeError::KernelProfileMismatch),
        },
        encoder_blocks,
        decoder_blocks,
        causal_decoder: ltx_configuration_bool(&values, 7)?,
        timestep_conditioning: ltx_configuration_bool(&values, 8)?,
        decode_noise_scale: ltx_configuration_f32(&values, 9)?,
        decode_timestep: ltx_configuration_f32(&values, 10)?,
        encoder_spatial_padding: ltx_configuration_padding(&values, 11)?,
        decoder_spatial_padding: ltx_configuration_padding(&values, 12)?,
    };
    ltx_validate_ratios(
        &configuration.encoder_blocks,
        &configuration.decoder_blocks,
        configuration.patch_size,
    )
    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    ltx_validate_normalization_channels(&configuration)
        .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    Ok(configuration)
}

fn ltx_blocks_from_module(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    side: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<LtxBlock>, VaeError> {
    let name = format!("__sim.ltx.{side}_blocks");
    let values =
        tensor_to_f32_with_backend_exact_native(backend, ltx_buffer(module, &name)?, context)
            .map_err(NativeOpsError::from)?;
    if values.is_empty() || !values.len().is_multiple_of(5) || values.len() > 128 * 5 {
        return Err(VaeError::KernelProfileMismatch);
    }
    values
        .chunks_exact(5)
        .map(|values| {
            let kind = match ltx_configuration_nonnegative_u64(values, 0)? {
                0 => LtxBlockKind::Residual,
                1 => LtxBlockKind::ResidualChangeChannels,
                2 => LtxBlockKind::CompressTime,
                3 => LtxBlockKind::CompressSpace,
                4 => LtxBlockKind::CompressAll,
                5 => LtxBlockKind::CompressAllChangeChannels,
                6 => LtxBlockKind::CompressAllResidual,
                7 => LtxBlockKind::CompressSpaceResidual,
                8 => LtxBlockKind::CompressTimeResidual,
                _ => return Err(VaeError::KernelProfileMismatch),
            };
            Ok(LtxBlock {
                kind,
                layers: ltx_configuration_nonnegative_u64(values, 1)?,
                multiplier: ltx_configuration_u64(values, 2)?,
                residual: ltx_configuration_bool(values, 3)?,
                inject_noise: ltx_configuration_bool(values, 4)?,
            })
        })
        .collect()
}

fn ltx_configuration_nonnegative_u64(values: &[f32], index: usize) -> Result<u64, VaeError> {
    let value = values
        .get(index)
        .copied()
        .ok_or(VaeError::KernelProfileMismatch)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > u64::MAX as f32 {
        return Err(VaeError::KernelProfileMismatch);
    }
    Ok(value as u64)
}

fn ltx_configuration_u64(values: &[f32], index: usize) -> Result<u64, VaeError> {
    ltx_configuration_nonnegative_u64(values, index).and_then(|value| {
        if value == 0 {
            Err(VaeError::KernelProfileMismatch)
        } else {
            Ok(value)
        }
    })
}

fn ltx_configuration_bool(values: &[f32], index: usize) -> Result<bool, VaeError> {
    match values.get(index).copied() {
        Some(0.0) => Ok(false),
        Some(1.0) => Ok(true),
        _ => Err(VaeError::KernelProfileMismatch),
    }
}

fn ltx_configuration_f32(values: &[f32], index: usize) -> Result<f32, VaeError> {
    values
        .get(index)
        .copied()
        .filter(|value| value.is_finite())
        .ok_or(VaeError::KernelProfileMismatch)
}

fn ltx_configuration_padding(values: &[f32], index: usize) -> Result<LtxSpatialPadding, VaeError> {
    match values.get(index).copied() {
        Some(0.0) => Ok(LtxSpatialPadding::Zeros),
        Some(1.0) => Ok(LtxSpatialPadding::Reflect),
        _ => Err(VaeError::KernelProfileMismatch),
    }
}

fn ltx_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    ltx_version(module)?;
    let configuration = ltx_configuration_from_module(module, backend, context)?;
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape[1] != configuration.input_channels || shape[2] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let usable = 1 + shape[2].saturating_sub(1) / 8 * 8;
    let input = narrow_contiguous(backend, input, 2, 0, usable, context)?;
    let mut hidden = ltx_patchify(backend, &input, configuration.patch_size, context)?;
    hidden = ltx_convolution(
        module,
        backend,
        &hidden,
        "encoder.conv_in.conv.weight",
        true,
        configuration.encoder_spatial_padding == LtxSpatialPadding::Reflect,
        context,
    )?;
    let mut channels = configuration.encoder_base_channels;
    for (index, block) in configuration.encoder_blocks.iter().copied().enumerate() {
        let prefix = format!("encoder.down_blocks.{index}");
        match block.kind {
            LtxBlockKind::Residual => {
                for layer in 0..block.layers {
                    hidden = ltx_residual_execute(
                        module,
                        backend,
                        &hidden,
                        &format!("{prefix}.res_blocks.{layer}"),
                        true,
                        configuration.encoder_spatial_padding == LtxSpatialPadding::Reflect,
                        configuration.norm_layer,
                        None,
                        None,
                        cpu_backend,
                        context,
                    )?;
                }
            }
            LtxBlockKind::ResidualChangeChannels => {
                hidden = ltx_residual_execute(
                    module,
                    backend,
                    &hidden,
                    &prefix,
                    true,
                    configuration.encoder_spatial_padding == LtxSpatialPadding::Reflect,
                    configuration.norm_layer,
                    None,
                    None,
                    cpu_backend,
                    context,
                )?;
                channels = channels
                    .checked_mul(block.multiplier)
                    .ok_or(VaeError::ShapeOverflow)?;
            }
            LtxBlockKind::CompressTime
            | LtxBlockKind::CompressSpace
            | LtxBlockKind::CompressAll
            | LtxBlockKind::CompressAllChangeChannels => {
                hidden = ltx_convolution(
                    module,
                    backend,
                    &hidden,
                    &format!("{prefix}.conv.weight"),
                    true,
                    configuration.encoder_spatial_padding == LtxSpatialPadding::Reflect,
                    context,
                )?;
                if block.kind == LtxBlockKind::CompressAllChangeChannels {
                    channels = channels
                        .checked_mul(block.multiplier)
                        .ok_or(VaeError::ShapeOverflow)?;
                }
            }
            LtxBlockKind::CompressAllResidual
            | LtxBlockKind::CompressSpaceResidual
            | LtxBlockKind::CompressTimeResidual => {
                let temporal_factor = if matches!(
                    block.kind,
                    LtxBlockKind::CompressAllResidual | LtxBlockKind::CompressTimeResidual
                ) {
                    2
                } else {
                    1
                };
                hidden = ltx_residual_downsample(
                    module,
                    backend,
                    &hidden,
                    &format!("{prefix}.conv.conv.weight"),
                    temporal_factor,
                    if matches!(block.kind, LtxBlockKind::CompressTimeResidual) {
                        1
                    } else {
                        2
                    },
                    block.multiplier,
                    configuration.encoder_spatial_padding == LtxSpatialPadding::Reflect,
                    context,
                )?;
                channels = channels
                    .checked_mul(block.multiplier)
                    .ok_or(VaeError::ShapeOverflow)?;
            }
        }
    }
    if hidden.descriptor().shape().get(1) != Some(&channels) {
        return Err(VaeError::ShapeOverflow);
    }
    hidden = ltx_normalize(
        module,
        backend,
        &hidden,
        "encoder.conv_norm_out",
        configuration.norm_layer,
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = ltx_convolution(
        module,
        backend,
        &hidden,
        "encoder.conv_out.conv.weight",
        true,
        configuration.encoder_spatial_padding == LtxSpatialPadding::Reflect,
        context,
    )?;
    let means = narrow_contiguous(
        backend,
        &hidden,
        1,
        0,
        configuration.latent_channels,
        context,
    )?;
    ltx_latent_statistics(module, backend, &means, true, context)
}

fn ltx_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    ltx_version(module)?;
    let configuration = ltx_configuration_from_module(module, backend, context)?;
    if input.descriptor().shape().get(1) != Some(&configuration.latent_channels) {
        return Err(VaeError::ShapeOverflow);
    }
    let stochastic_decode = configuration.timestep_conditioning
        || configuration
            .decoder_blocks
            .iter()
            .any(|block| block.kind == LtxBlockKind::Residual && block.inject_noise);
    let mut rng = if stochastic_decode {
        Some(begin_vae_rng(context)?)
    } else {
        None
    };
    let mut hidden = if configuration.timestep_conditioning {
        let transaction = rng.take().ok_or(VaeError::KernelProfileMismatch)?;
        let cpu_backend = cpu_backend.ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(
                "LTX stochastic decode requires the canonical CPU RNG backend".to_owned(),
            ))
        })?;
        let random = randn_like_with_context_exact_native(cpu_backend, input, transaction, context)
            .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
        rng = Some(random.transaction);
        let random = affine_tensor(
            backend,
            &random.tensor,
            configuration.decode_noise_scale,
            0.0,
            context,
        )?;
        let input = affine_tensor(
            backend,
            input,
            1.0 - configuration.decode_noise_scale,
            0.0,
            context,
        )?;
        add_tensor(backend, &random, &input, context)?
    } else {
        input.clone()
    };
    hidden = ltx_latent_statistics(module, backend, &hidden, false, context)?;
    hidden = ltx_convolution(
        module,
        backend,
        &hidden,
        "decoder.conv_in.conv.weight",
        configuration.causal_decoder,
        configuration.decoder_spatial_padding == LtxSpatialPadding::Reflect,
        context,
    )?;
    let scaled_timestep = if configuration.timestep_conditioning {
        let multiplier = tensor_to_f32_with_backend_exact_native(
            backend,
            ltx_buffer(module, "decoder.timestep_scale_multiplier")?,
            context,
        )
        .map_err(NativeOpsError::from)?;
        Some(
            multiplier
                .first()
                .copied()
                .filter(|value| value.is_finite())
                .ok_or(VaeError::ShapeOverflow)?
                * configuration.decode_timestep,
        )
    } else {
        None
    };
    let mut channels = hidden.descriptor().shape()[1];
    for (index, block) in configuration
        .decoder_blocks
        .iter()
        .copied()
        .rev()
        .enumerate()
    {
        let prefix = format!("decoder.up_blocks.{index}");
        match block.kind {
            LtxBlockKind::Residual => {
                let timestep = if configuration.timestep_conditioning {
                    Some(ltx_timestep_embedding(
                        module,
                        backend,
                        &format!("{prefix}.time_embedder"),
                        channels * 4,
                        scaled_timestep.ok_or(VaeError::ShapeOverflow)?,
                        input.descriptor(),
                        context,
                    )?)
                } else {
                    None
                };
                for layer in 0..block.layers {
                    hidden = ltx_residual_execute(
                        module,
                        backend,
                        &hidden,
                        &format!("{prefix}.res_blocks.{layer}"),
                        configuration.causal_decoder,
                        configuration.decoder_spatial_padding == LtxSpatialPadding::Reflect,
                        configuration.norm_layer,
                        timestep.as_ref(),
                        rng.as_mut(),
                        cpu_backend,
                        context,
                    )?;
                }
            }
            LtxBlockKind::ResidualChangeChannels => {
                hidden = ltx_residual_execute(
                    module,
                    backend,
                    &hidden,
                    &prefix,
                    configuration.causal_decoder,
                    configuration.decoder_spatial_padding == LtxSpatialPadding::Reflect,
                    configuration.norm_layer,
                    None,
                    rng.as_mut(),
                    cpu_backend,
                    context,
                )?;
                channels /= block.multiplier;
            }
            LtxBlockKind::CompressTime
            | LtxBlockKind::CompressSpace
            | LtxBlockKind::CompressAll => {
                let scale = ltx_block_scale(block);
                hidden = ltx_residual_upsample(
                    module,
                    backend,
                    &hidden,
                    &format!("{prefix}.conv.conv.weight"),
                    scale.0,
                    scale.1,
                    block.multiplier,
                    block.residual,
                    configuration.causal_decoder,
                    configuration.decoder_spatial_padding == LtxSpatialPadding::Reflect,
                    context,
                )?;
                channels /= block.multiplier;
            }
            _ => return Err(VaeError::KernelProfileMismatch),
        }
    }
    if hidden.descriptor().shape().get(1) != Some(&channels) {
        return Err(VaeError::ShapeOverflow);
    }
    hidden = ltx_normalize(
        module,
        backend,
        &hidden,
        "decoder.conv_norm_out",
        configuration.norm_layer,
        context,
    )?;
    if configuration.timestep_conditioning {
        let timestep = ltx_timestep_embedding(
            module,
            backend,
            "decoder.last_time_embedder",
            channels * 2,
            scaled_timestep.ok_or(VaeError::ShapeOverflow)?,
            input.descriptor(),
            context,
        )?;
        hidden = ltx_adaptive_norm(
            module,
            backend,
            &hidden,
            &timestep,
            "decoder.last_scale_shift_table",
            2,
            context,
        )?;
    }
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = ltx_convolution(
        module,
        backend,
        &hidden,
        "decoder.conv_out.conv.weight",
        configuration.causal_decoder,
        configuration.decoder_spatial_padding == LtxSpatialPadding::Reflect,
        context,
    )?;
    ltx_unpatchify(backend, &hidden, configuration.patch_size, context)
}

pub(crate) fn begin_vae_rng(context: &ExecutionContext<'_>) -> Result<RngTransaction, VaeError> {
    let address = context.rng_phase.cloned().ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(
            VideoVaeError::MissingRngPhase.to_string(),
        ))
    })?;
    let stream = generator_exact_native(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        0,
        address,
        context.cancellation,
    )
    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    stream
        .begin(None)
        .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()).into())
}

fn ltx_patchify(
    backend: &dyn TensorBackend,
    input: &Tensor,
    patch_size: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5
        || patch_size == 0
        || !shape[3].is_multiple_of(patch_size)
        || !shape[4].is_multiple_of(patch_size)
    {
        return Err(VaeError::ShapeOverflow);
    }
    let reshaped = reshape_read_only(
        input,
        vec![
            shape[0],
            shape[1],
            shape[2],
            shape[3] / patch_size,
            patch_size,
            shape[4] / patch_size,
            patch_size,
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 6, 4, 2, 3, 5])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    reshape_read_only(
        &contiguous,
        vec![
            shape[0],
            shape[1]
                .checked_mul(patch_size)
                .and_then(|channels| channels.checked_mul(patch_size))
                .ok_or(VaeError::ShapeOverflow)?,
            shape[2],
            shape[3] / patch_size,
            shape[4] / patch_size,
        ],
    )
}

fn ltx_unpatchify(
    backend: &dyn TensorBackend,
    input: &Tensor,
    patch_size: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let patch_channels = patch_size
        .checked_mul(patch_size)
        .ok_or(VaeError::ShapeOverflow)?;
    if shape.len() != 5 || patch_size == 0 || !shape[1].is_multiple_of(patch_channels) {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = shape[1] / patch_channels;
    let reshaped = reshape_read_only(
        input,
        vec![
            shape[0], channels, patch_size, patch_size, shape[2], shape[3], shape[4],
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 4, 5, 3, 6, 2])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    reshape_read_only(
        &contiguous,
        vec![
            shape[0],
            channels,
            shape[2],
            shape[3]
                .checked_mul(patch_size)
                .ok_or(VaeError::ShapeOverflow)?,
            shape[4]
                .checked_mul(patch_size)
                .ok_or(VaeError::ShapeOverflow)?,
        ],
    )
}

fn ltx_convolution(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    causal: bool,
    reflect_spatial: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let convolution_module = crate::vae_image::find_module(module, name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing LTX convolution module {name}"
        )))
    })?;
    let (weight, _) = convolution_module.dense_parameters()?;
    let kernel = weight.descriptor().shape();
    if kernel.len() != 5 {
        return Err(VaeError::ShapeOverflow);
    }
    let input = ltx_pad_3d(
        backend,
        input,
        kernel[2].saturating_sub(1),
        kernel[3] / 2,
        kernel[4] / 2,
        causal,
        reflect_spatial,
        context,
    )?;
    convolution(module, backend, &input, name, context)
}

#[allow(clippy::too_many_arguments)]
fn ltx_pad_3d(
    backend: &dyn TensorBackend,
    input: &Tensor,
    temporal_padding: u64,
    height_padding: u64,
    width_padding: u64,
    causal: bool,
    reflect_spatial: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5
        || shape.contains(&0)
        || (reflect_spatial
            && ((height_padding > 0 && shape[3] <= height_padding)
                || (width_padding > 0 && shape[4] <= width_padding)))
    {
        return Err(VaeError::ShapeOverflow);
    }
    let before = if causal {
        temporal_padding
    } else {
        temporal_padding / 2
    };
    let after = temporal_padding.saturating_sub(before);
    let output_shape = vec![
        shape[0],
        shape[1],
        shape[2]
            .checked_add(before)
            .and_then(|value| value.checked_add(after))
            .ok_or(VaeError::ShapeOverflow)?,
        shape[3]
            .checked_add(height_padding * 2)
            .ok_or(VaeError::ShapeOverflow)?,
        shape[4]
            .checked_add(width_padding * 2)
            .ok_or(VaeError::ShapeOverflow)?,
    ];
    if output_shape == shape {
        return Ok(input.clone());
    }
    let descriptor = TensorDescriptor::contiguous(
        output_shape.clone(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    for target_time in 0..output_shape[2] {
        let source_time = target_time.saturating_sub(before).min(shape[2] - 1);
        let frame = input.narrow_read_only(2, i64::try_from(source_time)?, 1)?;
        let (updated, event) = backend.replace_rectangular_slice(
            &output,
            &frame,
            &[0, 0, target_time, height_padding, width_padding],
            context,
        )?;
        backend.wait_event(event, context)?;
        output = updated;
    }
    if !reflect_spatial {
        return Ok(output);
    }
    for padding in 0..height_padding {
        let top_source = height_padding + (height_padding - padding);
        let bottom_source = height_padding + shape[3] - 2 - padding;
        for (source, target) in [
            (top_source, padding),
            (bottom_source, height_padding + shape[3] + padding),
        ] {
            let row = output.narrow_read_only(3, i64::try_from(source)?, 1)?;
            let row = contiguous_copy(backend, &row, context)?;
            let (updated, event) =
                backend.replace_rectangular_slice(&output, &row, &[0, 0, 0, target, 0], context)?;
            backend.wait_event(event, context)?;
            output = updated;
        }
    }
    for padding in 0..width_padding {
        let left_source = width_padding + (width_padding - padding);
        let right_source = width_padding + shape[4] - 2 - padding;
        for (source, target) in [
            (left_source, padding),
            (right_source, width_padding + shape[4] + padding),
        ] {
            let column = output.narrow_read_only(4, i64::try_from(source)?, 1)?;
            let column = contiguous_copy(backend, &column, context)?;
            let (updated, event) = backend.replace_rectangular_slice(
                &output,
                &column,
                &[0, 0, 0, 0, target],
                context,
            )?;
            backend.wait_event(event, context)?;
            output = updated;
        }
    }
    Ok(output)
}

fn ltx_pixel_norm(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape[1] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let squared = binary_tensor(backend, BinaryOperation::Multiply, input, input, context)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], 1, shape[2], shape[3], shape[4]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mean, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Mean,
            dimensions: vec![1],
            keep_dimensions: true,
            accumulation_dtype: Some(input.descriptor().dtype()),
            correction: 0,
        },
        &squared,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let (mean, event) = backend.binary_scalar(
        BinaryOperation::Add,
        &mean,
        Scalar::Float(1.0e-8),
        ScalarSide::Right,
        mean.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let (inverse, event) = backend.unary(
        UnaryOperation::ReciprocalSquareRoot,
        &mean,
        mean.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    binary_tensor(backend, BinaryOperation::Multiply, input, &inverse, context)
}

fn ltx_normalize(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    norm_layer: LtxNormLayer,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    match norm_layer {
        LtxNormLayer::Pixel => ltx_pixel_norm(backend, input, context),
        LtxNormLayer::Group => {
            let weight = ltx_buffer(module, &format!("{prefix}.weight"))?;
            let bias = ltx_buffer(module, &format!("{prefix}.bias"))?;
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
        LtxNormLayer::Layer => {
            let prefix = format!("{prefix}.norm");
            let weight = ltx_buffer(module, &format!("{prefix}.weight"))?;
            let bias = ltx_buffer(module, &format!("{prefix}.bias"))?;
            channel_layer_norm_tensor_with_context_exact_native(
                backend,
                input,
                Some(weight),
                Some(bias),
                1.0e-6,
                context,
            )
            .map_err(NativeOpsError::from)
            .map_err(VaeError::from)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ltx_residual_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    causal: bool,
    reflect_spatial: bool,
    norm_layer: LtxNormLayer,
    timestep: Option<&Tensor>,
    mut rng: Option<&mut RngTransaction>,
    cpu_backend: Option<&CpuBackend>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shortcut_name = format!("{prefix}.conv_shortcut.weight");
    let shortcut = if crate::vae_image::find_module(module, &shortcut_name).is_some() {
        let weight_name = format!("{prefix}.norm3.norm.weight");
        let bias_name = format!("{prefix}.norm3.norm.bias");
        let weight = ltx_buffer(module, &weight_name)?;
        let bias = ltx_buffer(module, &bias_name)?;
        let normalized = channel_layer_norm_tensor_with_context_exact_native(
            backend,
            input,
            Some(weight),
            Some(bias),
            1.0e-6,
            context,
        )
        .map_err(NativeOpsError::from)?;
        convolution(module, backend, &normalized, &shortcut_name, context)?
    } else {
        input.clone()
    };
    let mut hidden = ltx_normalize(
        module,
        backend,
        input,
        &format!("{prefix}.norm1"),
        norm_layer,
        context,
    )?;
    if let Some(timestep) = timestep {
        hidden = ltx_adaptive_norm(
            module,
            backend,
            &hidden,
            timestep,
            &format!("{prefix}.scale_shift_table"),
            4,
            context,
        )?;
    }
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = ltx_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv1.conv.weight"),
        causal,
        reflect_spatial,
        context,
    )?;
    hidden = ltx_spatial_noise(
        module,
        backend,
        &hidden,
        prefix,
        1,
        rng.as_deref_mut(),
        cpu_backend,
        context,
    )?;
    hidden = ltx_normalize(
        module,
        backend,
        &hidden,
        &format!("{prefix}.norm2"),
        norm_layer,
        context,
    )?;
    if let Some(timestep) = timestep {
        hidden = ltx_adaptive_norm_second(backend, &hidden, module, timestep, prefix, context)?;
    }
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = ltx_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2.conv.weight"),
        causal,
        reflect_spatial,
        context,
    )?;
    hidden = ltx_spatial_noise(
        module,
        backend,
        &hidden,
        prefix,
        2,
        rng,
        cpu_backend,
        context,
    )?;
    add_tensor(backend, &hidden, &shortcut, context)
}

fn ltx_residual_downsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    temporal_factor: u64,
    spatial_factor: u64,
    multiplier: u64,
    reflect_spatial: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let input = if temporal_factor == 2 {
        let first = narrow_contiguous(backend, input, 2, 0, 1, context)?;
        concatenate_temporal(backend, &[first, input.clone()], context)?
    } else {
        input.clone()
    };
    let packed = ltx_space_to_depth(backend, &input, temporal_factor, spatial_factor, context)?;
    let output_channels = input.descriptor().shape()[1]
        .checked_mul(multiplier)
        .ok_or(VaeError::ShapeOverflow)?;
    let residual = grouped_channel_mean(backend, &packed, output_channels, context)?;
    let convolved = ltx_convolution(
        module,
        backend,
        &input,
        name,
        true,
        reflect_spatial,
        context,
    )?;
    let convolved = ltx_space_to_depth(
        backend,
        &convolved,
        temporal_factor,
        spatial_factor,
        context,
    )?;
    add_tensor(backend, &convolved, &residual, context)
}

fn ltx_residual_upsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    temporal_factor: u64,
    spatial_factor: u64,
    multiplier: u64,
    residual: bool,
    causal: bool,
    reflect_spatial: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let convolved = ltx_convolution(
        module,
        backend,
        input,
        name,
        causal,
        reflect_spatial,
        context,
    )?;
    let convolved = ltx_depth_to_space(
        backend,
        &convolved,
        temporal_factor,
        spatial_factor,
        context,
    )?;
    let mut output = if temporal_factor == 2 {
        let frames = convolved.descriptor().shape()[2];
        narrow_contiguous(backend, &convolved, 2, 1, frames - 1, context)?
    } else {
        convolved
    };
    if residual {
        let residual =
            ltx_depth_to_space(backend, input, temporal_factor, spatial_factor, context)?;
        let residual = if temporal_factor == 2 {
            let frames = residual.descriptor().shape()[2];
            narrow_contiguous(backend, &residual, 2, 1, frames - 1, context)?
        } else {
            residual
        };
        let volume = temporal_factor
            .checked_mul(spatial_factor)
            .and_then(|factor| factor.checked_mul(spatial_factor))
            .ok_or(VaeError::ShapeOverflow)?;
        if !volume.is_multiple_of(multiplier) {
            return Err(VaeError::ShapeOverflow);
        }
        let residual = ltx_repeat_channels(backend, &residual, volume / multiplier, context)?;
        output = add_tensor(backend, &output, &residual, context)?;
    }
    Ok(output)
}

fn ltx_space_to_depth(
    backend: &dyn TensorBackend,
    input: &Tensor,
    temporal_factor: u64,
    spatial_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5
        || temporal_factor == 0
        || spatial_factor == 0
        || !shape[2].is_multiple_of(temporal_factor)
        || !shape[3].is_multiple_of(spatial_factor)
        || !shape[4].is_multiple_of(spatial_factor)
    {
        return Err(VaeError::ShapeOverflow);
    }
    let reshaped = reshape_read_only(
        input,
        vec![
            shape[0],
            shape[1],
            shape[2] / temporal_factor,
            temporal_factor,
            shape[3] / spatial_factor,
            spatial_factor,
            shape[4] / spatial_factor,
            spatial_factor,
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 3, 5, 7, 2, 4, 6])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    let channels = shape[1]
        .checked_mul(temporal_factor)
        .and_then(|value| value.checked_mul(spatial_factor))
        .and_then(|value| value.checked_mul(spatial_factor))
        .ok_or(VaeError::ShapeOverflow)?;
    reshape_read_only(
        &contiguous,
        vec![
            shape[0],
            channels,
            shape[2] / temporal_factor,
            shape[3] / spatial_factor,
            shape[4] / spatial_factor,
        ],
    )
}

fn ltx_depth_to_space(
    backend: &dyn TensorBackend,
    input: &Tensor,
    temporal_factor: u64,
    spatial_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let factor = temporal_factor
        .checked_mul(spatial_factor)
        .and_then(|value| value.checked_mul(spatial_factor))
        .ok_or(VaeError::ShapeOverflow)?;
    if shape.len() != 5 || factor == 0 || !shape[1].is_multiple_of(factor) {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = shape[1] / factor;
    let reshaped = reshape_read_only(
        input,
        vec![
            shape[0],
            channels,
            temporal_factor,
            spatial_factor,
            spatial_factor,
            shape[2],
            shape[3],
            shape[4],
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 5, 2, 6, 3, 7, 4])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    reshape_read_only(
        &contiguous,
        vec![
            shape[0],
            channels,
            shape[2] * temporal_factor,
            shape[3] * spatial_factor,
            shape[4] * spatial_factor,
        ],
    )
}

fn ltx_repeat_channels(
    backend: &dyn TensorBackend,
    input: &Tensor,
    repeats: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if repeats == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let inputs = std::iter::repeat_n(input.clone(), usize::try_from(repeats)?).collect::<Vec<_>>();
    concatenate_dimension(backend, &inputs, 1, context)
}

pub(crate) fn ltx_latent_statistics(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    normalize: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mean = reshape_read_only(
        ltx_buffer(module, "per_channel_statistics.mean-of-means")?,
        vec![1, 128, 1, 1, 1],
    )?;
    let standard_deviation = reshape_read_only(
        ltx_buffer(module, "per_channel_statistics.std-of-means")?,
        vec![1, 128, 1, 1, 1],
    )?;
    if normalize {
        let centered = binary_tensor(backend, BinaryOperation::Subtract, input, &mean, context)?;
        binary_tensor(
            backend,
            BinaryOperation::Divide,
            &centered,
            &standard_deviation,
            context,
        )
    } else {
        let scaled = binary_tensor(
            backend,
            BinaryOperation::Multiply,
            input,
            &standard_deviation,
            context,
        )?;
        binary_tensor(backend, BinaryOperation::Add, &scaled, &mean, context)
    }
}

#[cfg(feature = "test-support")]
pub fn ltx_latent_statistics_test_support(
    backend: &dyn TensorBackend,
    input: &Tensor,
    mean: &[f32; 128],
    standard_deviation: &[f32; 128],
    normalize: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    let mean = tensor_from_f32_with_backend_exact_native(
        backend,
        &[128],
        mean,
        DType::F32,
        input.descriptor().device(),
        context,
    )
    .map_err(NativeOpsError::from)?;
    let standard_deviation = tensor_from_f32_with_backend_exact_native(
        backend,
        &[128],
        standard_deviation,
        DType::F32,
        input.descriptor().device(),
        context,
    )
    .map_err(NativeOpsError::from)?;
    let module = NativeModule::module_dict(
        "ltx-statistics-test-support",
        vec![
            NativeModule::buffer("per_channel_statistics.mean-of-means", mean)?,
            NativeModule::buffer("per_channel_statistics.std-of-means", standard_deviation)?,
        ],
    )?;
    ltx_latent_statistics(&module, backend, input, normalize, context)
}

fn ltx_buffer<'a>(module: &'a NativeModule, name: &str) -> Result<&'a Tensor, VaeError> {
    crate::vae_image::find_module(module, name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing LTX state {name}"
            )))
        })
}

fn ltx_spatial_noise(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    index: u8,
    rng: Option<&mut RngTransaction>,
    cpu_backend: Option<&CpuBackend>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let name = format!("{prefix}.per_channel_scale{index}");
    let Some(scale) =
        crate::vae_image::find_module(module, &name).and_then(NativeModule::registered_buffer)
    else {
        return Ok(input.clone());
    };
    let transaction = rng.ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(
            VideoVaeError::MissingRngPhase.to_string(),
        ))
    })?;
    let cpu_backend = cpu_backend.ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(
            "LTX spatial noise requires the canonical CPU RNG backend".to_owned(),
        ))
    })?;
    let shape = input.descriptor().shape();
    let descriptor = TensorDescriptor::contiguous(
        vec![1, 1, 1, shape[3], shape[4]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (template, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let random =
        randn_like_with_context_exact_native(cpu_backend, &template, transaction.clone(), context)
            .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    *transaction = random.transaction;
    let scale = reshape_read_only(scale, vec![1, shape[1], 1, 1, 1])?;
    let noise = binary_tensor(
        backend,
        BinaryOperation::Multiply,
        &random.tensor,
        &scale,
        context,
    )?;
    add_tensor(backend, input, &noise, context)
}

fn ltx_timestep_embedding(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    prefix: &str,
    embedding_channels: u64,
    timestep: f32,
    descriptor: &TensorDescriptor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut values = Vec::with_capacity(256);
    for index in 0..128 {
        let exponent = -(index as f32) * 10000.0_f32.ln() / 128.0;
        values.push(timestep * exponent.exp());
    }
    let mut projected = Vec::with_capacity(256);
    projected.extend(values.iter().map(|value| value.cos()));
    projected.extend(values.iter().map(|value| value.sin()));
    let batch = descriptor
        .shape()
        .first()
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let mut batched = Vec::new();
    batched
        .try_reserve_exact(usize::try_from(
            batch.checked_mul(256).ok_or(VaeError::ShapeOverflow)?,
        )?)
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    for _ in 0..batch {
        batched.extend_from_slice(&projected);
    }
    let input = tensor_from_f32_with_backend_exact_native(
        backend,
        &[batch, 256, 1, 1, 1],
        &batched,
        descriptor.dtype(),
        descriptor.device(),
        context,
    )
    .map_err(NativeOpsError::from)?;
    let mut hidden = mochi_linear_channels(
        module,
        backend,
        &input,
        &format!("{prefix}.timestep_embedder.linear_1.weight"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = mochi_linear_channels(
        module,
        backend,
        &hidden,
        &format!("{prefix}.timestep_embedder.linear_2.weight"),
        context,
    )?;
    if hidden.descriptor().shape().get(1) != Some(&embedding_channels) {
        return Err(VaeError::ShapeOverflow);
    }
    Ok(hidden)
}

fn ltx_adaptive_norm(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    timestep: &Tensor,
    table_name: &str,
    parts: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let table = reshape_read_only(
        ltx_buffer(module, table_name)?,
        vec![1, parts, shape[1], 1, 1, 1],
    )?;
    let timestep = reshape_read_only(timestep, vec![shape[0], parts, shape[1], 1, 1, 1])?;
    let values = binary_tensor(backend, BinaryOperation::Add, &table, &timestep, context)?;
    let shift = narrow_contiguous(backend, &values, 1, 0, 1, context)?;
    let shift = reshape_read_only(&shift, vec![shape[0], shape[1], 1, 1, 1])?;
    let scale = narrow_contiguous(backend, &values, 1, 1, 1, context)?;
    let scale = reshape_read_only(&scale, vec![shape[0], shape[1], 1, 1, 1])?;
    let (scale, event) = backend.binary_scalar(
        BinaryOperation::Add,
        &scale,
        Scalar::Float(1.0),
        ScalarSide::Right,
        scale.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let scaled = binary_tensor(backend, BinaryOperation::Multiply, input, &scale, context)?;
    binary_tensor(backend, BinaryOperation::Add, &scaled, &shift, context)
}

fn ltx_adaptive_norm_second(
    backend: &dyn TensorBackend,
    input: &Tensor,
    module: &NativeModule,
    timestep: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let table = reshape_read_only(
        ltx_buffer(module, &format!("{prefix}.scale_shift_table"))?,
        vec![1, 4, shape[1], 1, 1, 1],
    )?;
    let timestep = reshape_read_only(timestep, vec![shape[0], 4, shape[1], 1, 1, 1])?;
    let values = binary_tensor(backend, BinaryOperation::Add, &table, &timestep, context)?;
    let shift = narrow_contiguous(backend, &values, 1, 2, 1, context)?;
    let shift = reshape_read_only(&shift, vec![shape[0], shape[1], 1, 1, 1])?;
    let scale = narrow_contiguous(backend, &values, 1, 3, 1, context)?;
    let scale = reshape_read_only(&scale, vec![shape[0], shape[1], 1, 1, 1])?;
    let (scale, event) = backend.binary_scalar(
        BinaryOperation::Add,
        &scale,
        Scalar::Float(1.0),
        ScalarSide::Right,
        scale.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let scaled = binary_tensor(backend, BinaryOperation::Multiply, input, &scale, context)?;
    binary_tensor(backend, BinaryOperation::Add, &scaled, &shift, context)
}

fn mochi_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = input
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .filter(|frames| *frames > 0)
        .ok_or(VaeError::ShapeOverflow)?;
    let usable = 1 + frames.saturating_sub(1) / 6 * 6;
    let input = narrow_contiguous(backend, input, 2, 0, usable, context)?;
    let mut hidden = mochi_fourier_features(backend, &input, context)?;
    hidden = mochi_linear_channels(module, backend, &hidden, "encoder.layers.0.weight", context)?;
    for layer in 1..=3 {
        hidden = mochi_residual_execute(
            module,
            backend,
            &hidden,
            &format!("encoder.layers.{layer}"),
            false,
            context,
        )?;
    }
    for (layer, blocks) in [(4, 3), (5, 4), (6, 6)] {
        hidden = mochi_causal_convolution(
            module,
            backend,
            &hidden,
            &format!("encoder.layers.{layer}.layers.0.weight"),
            context,
        )?;
        for block in 0..blocks {
            hidden = mochi_residual_execute(
                module,
                backend,
                &hidden,
                &format!("encoder.layers.{layer}.layers.{}", block + 1),
                true,
                context,
            )?;
        }
    }
    for layer in 7..=9 {
        hidden = mochi_residual_execute(
            module,
            backend,
            &hidden,
            &format!("encoder.layers.{layer}"),
            true,
            context,
        )?;
    }
    hidden = mochi_normalize(module, backend, &hidden, "encoder.output_norm", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = mochi_linear_channels(
        module,
        backend,
        &hidden,
        "encoder.output_proj.weight",
        context,
    )?;
    narrow_contiguous(backend, &hidden, 1, 0, 12, context)
}

fn mochi_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = convolution(module, backend, input, "decoder.blocks.0.0.weight", context)?;
    for block in 1..=3 {
        hidden = mochi_residual_execute(
            module,
            backend,
            &hidden,
            &format!("decoder.blocks.0.{block}"),
            false,
            context,
        )?;
    }
    for (block, residual_blocks, temporal_expansion) in [(1, 6, 3), (2, 4, 2), (3, 3, 1)] {
        for residual in 0..residual_blocks {
            hidden = mochi_residual_execute(
                module,
                backend,
                &hidden,
                &format!("decoder.blocks.{block}.blocks.{residual}"),
                false,
                context,
            )?;
        }
        hidden = mochi_linear_channels(
            module,
            backend,
            &hidden,
            &format!("decoder.blocks.{block}.proj.weight"),
            context,
        )?;
        hidden = mochi_depth_to_space_time(backend, &hidden, temporal_expansion, context)?;
    }
    for residual in 0..3 {
        hidden = mochi_residual_execute(
            module,
            backend,
            &hidden,
            &format!("decoder.blocks.4.{residual}"),
            false,
            context,
        )?;
    }
    hidden = silu_tensor(backend, &hidden, context)?;
    mochi_linear_channels(
        module,
        backend,
        &hidden,
        "decoder.output_proj.weight",
        context,
    )
}

fn mochi_linear_channels(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let linear = crate::vae_image::find_module(module, name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing Mochi linear module {name}"
        )))
    })?;
    let (weight, bias) = linear.dense_parameters()?;
    let shape = input.descriptor().shape();
    if shape.len() != 5 {
        return Err(VaeError::ShapeOverflow);
    }
    let channels_last = permute_read_only(input, &[0, 2, 3, 4, 1])?;
    let channels_last = contiguous_copy(backend, &channels_last, context)?;
    let rows = shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != 1)
        .try_fold(1_u64, |rows, (_, extent)| {
            rows.checked_mul(*extent).ok_or(VaeError::ShapeOverflow)
        })?;
    let weight_shape = weight.descriptor().shape();
    if weight_shape.len() != 2 || weight_shape[1] != shape[1] {
        return Err(VaeError::ShapeOverflow);
    }
    let flattened = reshape_read_only(&channels_last, vec![rows, shape[1]])?;
    let transposed_weight = permute_read_only(weight, &[1, 0])?;
    let descriptor = TensorDescriptor::contiguous(
        vec![rows, weight_shape[0]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.linear_algebra(
        LinearAlgebraOperation::MatrixMultiply,
        &[flattened, transposed_weight],
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let output = if let Some(bias) = bias {
        let bias = reshape_read_only(bias, vec![1, weight_shape[0]])?;
        binary_tensor(backend, BinaryOperation::Add, &output, &bias, context)?
    } else {
        output
    };
    let output = reshape_read_only(
        &output,
        vec![shape[0], shape[2], shape[3], shape[4], weight_shape[0]],
    )?;
    let output = permute_read_only(&output, &[0, 4, 1, 2, 3])?;
    contiguous_copy(backend, &output, context)
}

fn mochi_causal_convolution(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let convolution_module = crate::vae_image::find_module(module, name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing Mochi convolution module {name}"
        )))
    })?;
    let (weight, _) = convolution_module.dense_parameters()?;
    let kernel = weight
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let input = first_frame_temporal_pad(backend, input, kernel.saturating_sub(1), context)?;
    let weight_shape = weight.descriptor().shape();
    let input = if weight_shape.get(3) == Some(&3) || weight_shape.get(4) == Some(&3) {
        let frames = split_video_frames(backend, &input, context)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                replication_pad_2d_tensor_with_context_exact_native(
                    backend,
                    &frame,
                    [1, 1, 1, 1],
                    context,
                )
                .map_err(NativeOpsError::from)
                .map_err(VaeError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        stack_video_frames(backend, &frames, context)?
    } else {
        input
    };
    convolution(module, backend, &input, name, context)
}

fn mochi_normalize(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = crate::vae_image::find_module(module, &weight_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Mochi normalization buffer {weight_name}"
            )))
        })?;
    let bias = crate::vae_image::find_module(module, &bias_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Mochi normalization buffer {bias_name}"
            )))
        })?;
    let frames = split_video_frames(backend, input, context)?;
    let frames = frames
        .into_iter()
        .map(|frame| {
            group_norm_tensor_with_context_exact_native(
                backend,
                &frame,
                32,
                Some(weight),
                Some(bias),
                1.0e-5,
                context,
            )
            .map_err(NativeOpsError::from)
            .map_err(VaeError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    stack_video_frames(backend, &frames, context)
}

fn mochi_residual_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    attention: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = mochi_normalize(
        module,
        backend,
        input,
        &format!("{prefix}.stack.0"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = mochi_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.stack.2.weight"),
        context,
    )?;
    hidden = mochi_normalize(
        module,
        backend,
        &hidden,
        &format!("{prefix}.stack.3"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = mochi_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.stack.5.weight"),
        context,
    )?;
    hidden = add_tensor(backend, input, &hidden, context)?;
    if attention {
        mochi_attention_execute(
            module,
            backend,
            &hidden,
            &format!("{prefix}.attn_block"),
            context,
        )
    } else {
        Ok(hidden)
    }
}

fn mochi_attention_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let normalized = mochi_normalize(module, backend, input, &format!("{prefix}.norm"), context)?;
    let qkv = mochi_linear_channels(
        module,
        backend,
        &normalized,
        &format!("{prefix}.attn.qkv.weight"),
        context,
    )?;
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape[1] == 0 || !shape[1].is_multiple_of(32) {
        return Err(VaeError::ShapeOverflow);
    }
    let query = narrow_contiguous(backend, &qkv, 1, 0, shape[1], context)?;
    let key = narrow_contiguous(
        backend,
        &qkv,
        1,
        i64::try_from(shape[1])?,
        shape[1],
        context,
    )?;
    let value = narrow_contiguous(
        backend,
        &qkv,
        1,
        i64::try_from(shape[1].checked_mul(2).ok_or(VaeError::ShapeOverflow)?)?,
        shape[1],
        context,
    )?;
    let attended = if shape[2] == 1 {
        value
    } else {
        mochi_temporal_attention(backend, &query, &key, &value, context)?
    };
    let projected = mochi_linear_channels(
        module,
        backend,
        &attended,
        &format!("{prefix}.attn.out.weight"),
        context,
    )?;
    add_tensor(backend, input, &projected, context)
}

fn mochi_temporal_attention(
    backend: &dyn TensorBackend,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = query.descriptor().shape();
    if shape.len() != 5 || shape[1] == 0 || !shape[1].is_multiple_of(32) {
        return Err(VaeError::ShapeOverflow);
    }
    let heads = shape[1] / 32;
    let batch_space_heads = shape[0]
        .checked_mul(shape[3])
        .and_then(|value| value.checked_mul(shape[4]))
        .and_then(|value| value.checked_mul(heads))
        .ok_or(VaeError::ShapeOverflow)?;
    let prepare = |tensor: &Tensor, normalize: bool| -> Result<Tensor, VaeError> {
        let tensor = reshape_read_only(
            tensor,
            vec![shape[0], heads, 32, shape[2], shape[3], shape[4]],
        )?;
        let tensor = permute_read_only(&tensor, &[0, 4, 5, 1, 2, 3])?;
        let tensor = contiguous_copy(backend, &tensor, context)?;
        let tensor = reshape_read_only(&tensor, vec![batch_space_heads, 32, shape[2]])?;
        if normalize {
            mochi_l2_normalize(backend, &tensor, context)
        } else {
            Ok(tensor)
        }
    };
    let query = prepare(query, true)?;
    let key = prepare(key, true)?;
    let value = prepare(value, false)?;
    let attention_shape = vec![batch_space_heads, 32, 1, shape[2]];
    let query = reshape_read_only(&query, attention_shape.clone())?;
    let key = reshape_read_only(&key, attention_shape.clone())?;
    let value = reshape_read_only(&value, attention_shape)?;
    let attended = spatial_attention_from_qkv(backend, &value, &query, &key, &value, context)?;
    let attended = reshape_read_only(
        &attended,
        vec![shape[0], shape[3], shape[4], heads, 32, shape[2]],
    )?;
    let attended = permute_read_only(&attended, &[0, 3, 4, 5, 1, 2])?;
    let attended = contiguous_copy(backend, &attended, context)?;
    reshape_read_only(
        &attended,
        vec![shape[0], shape[1], shape[2], shape[3], shape[4]],
    )
}

fn mochi_l2_normalize(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let squared = binary_tensor(backend, BinaryOperation::Multiply, input, input, context)?;
    let shape = input.descriptor().shape();
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], 1, shape[2]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (norm, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Sum,
            dimensions: vec![1],
            keep_dimensions: true,
            accumulation_dtype: Some(input.descriptor().dtype()),
            correction: 0,
        },
        &squared,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let norm = unary_tensor(backend, &norm, UnaryOperation::SquareRoot, context)?;
    let (norm, event) = backend.binary_scalar(
        BinaryOperation::Maximum,
        &norm,
        Scalar::Float(1.0e-12),
        ScalarSide::Right,
        norm.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    binary_tensor(backend, BinaryOperation::Divide, input, &norm, context)
}

fn mochi_fourier_features(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape[1] != 3 {
        return Err(VaeError::ShapeOverflow);
    }
    let mut scaled = Vec::new();
    for channel in 0..shape[1] {
        let channel = narrow_contiguous(backend, input, 1, i64::try_from(channel)?, 1, context)?;
        for exponent in [6_i32, 7_i32] {
            scaled.push(affine_tensor(
                backend,
                &channel,
                (2.0_f32).powi(exponent) * std::f32::consts::TAU,
                0.0,
                context,
            )?);
        }
    }
    let scaled = concatenate_dimension(backend, &scaled, 1, context)?;
    let sine = unary_tensor(backend, &scaled, UnaryOperation::Sine, context)?;
    let cosine = unary_tensor(backend, &scaled, UnaryOperation::Cosine, context)?;
    concatenate_dimension(backend, &[input.clone(), sine, cosine], 1, context)
}

fn mochi_depth_to_space_time(
    backend: &dyn TensorBackend,
    input: &Tensor,
    temporal_expansion: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let factor = temporal_expansion
        .checked_mul(4)
        .ok_or(VaeError::ShapeOverflow)?;
    if shape.len() != 5 || temporal_expansion == 0 || !shape[1].is_multiple_of(factor) {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = shape[1] / factor;
    let reshaped = reshape_read_only(
        input,
        vec![
            shape[0],
            channels,
            temporal_expansion,
            2,
            2,
            shape[2],
            shape[3],
            shape[4],
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 5, 2, 6, 3, 7, 4])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    let frames = shape[2]
        .checked_mul(temporal_expansion)
        .ok_or(VaeError::ShapeOverflow)?;
    let output = reshape_read_only(
        &contiguous,
        vec![
            shape[0],
            channels,
            frames,
            shape[3].checked_mul(2).ok_or(VaeError::ShapeOverflow)?,
            shape[4].checked_mul(2).ok_or(VaeError::ShapeOverflow)?,
        ],
    )?;
    if temporal_expansion > 1 {
        narrow_contiguous(
            backend,
            &output,
            2,
            i64::try_from(temporal_expansion - 1)?,
            frames
                .checked_sub(temporal_expansion - 1)
                .ok_or(VaeError::ShapeOverflow)?,
            context,
        )
    } else {
        Ok(output)
    }
}

fn wan21_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = input
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .filter(|frames| *frames > 0)
        .ok_or(VaeError::ShapeOverflow)?;
    let usable = 1 + frames.saturating_sub(1) / 4 * 4;
    let input = narrow_contiguous(backend, input, 2, 0, usable, context)?;
    let mut hidden =
        wan_causal_convolution(module, backend, &input, "encoder.conv1.weight", context)?;
    let mut sequence = 0;
    for level in 0..4 {
        for _ in 0..2 {
            hidden = wan_residual_execute(
                module,
                backend,
                &hidden,
                &format!("encoder.downsamples.{sequence}"),
                context,
            )?;
            sequence += 1;
        }
        if level < 3 {
            hidden = wan_downsample(
                module,
                backend,
                &hidden,
                &format!("encoder.downsamples.{sequence}"),
                level > 0,
                context,
            )?;
            sequence += 1;
        }
    }
    hidden = wan_residual_execute(module, backend, &hidden, "encoder.middle.0", context)?;
    hidden = wan_attention_execute(module, backend, &hidden, "encoder.middle.1", context)?;
    hidden = wan_residual_execute(module, backend, &hidden, "encoder.middle.2", context)?;
    hidden = rms_channel_norm(module, backend, &hidden, "encoder.head.0", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = wan_causal_convolution(module, backend, &hidden, "encoder.head.2.weight", context)?;
    hidden = wan_causal_convolution(module, backend, &hidden, "conv1.weight", context)?;
    narrow_contiguous(backend, &hidden, 1, 0, 16, context)
}

fn wan21_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = wan_causal_convolution(module, backend, input, "conv2.weight", context)?;
    hidden = wan_causal_convolution(module, backend, &hidden, "decoder.conv1.weight", context)?;
    hidden = wan_residual_execute(module, backend, &hidden, "decoder.middle.0", context)?;
    hidden = wan_attention_execute(module, backend, &hidden, "decoder.middle.1", context)?;
    hidden = wan_residual_execute(module, backend, &hidden, "decoder.middle.2", context)?;
    let mut sequence = 0;
    for level in 0..4 {
        for _ in 0..3 {
            hidden = wan_residual_execute(
                module,
                backend,
                &hidden,
                &format!("decoder.upsamples.{sequence}"),
                context,
            )?;
            sequence += 1;
        }
        if level < 3 {
            hidden = wan_upsample(
                module,
                backend,
                &hidden,
                &format!("decoder.upsamples.{sequence}"),
                level < 2,
                context,
            )?;
            sequence += 1;
        }
    }
    hidden = rms_channel_norm(module, backend, &hidden, "decoder.head.0", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    wan_causal_convolution(module, backend, &hidden, "decoder.head.2.weight", context)
}

fn wan22_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = input
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .filter(|frames| *frames > 0)
        .ok_or(VaeError::ShapeOverflow)?;
    let usable = 1 + frames.saturating_sub(1) / 4 * 4;
    let input = narrow_contiguous(backend, input, 2, 0, usable, context)?;
    let input = wan22_patchify(backend, &input, context)?;
    let mut hidden =
        wan_causal_convolution(module, backend, &input, "encoder.conv1.weight", context)?;
    for level in 0..4 {
        hidden = wan22_down_block(
            module,
            backend,
            &hidden,
            &format!("encoder.downsamples.{level}"),
            level < 3,
            matches!(level, 1 | 2),
            context,
        )?;
    }
    hidden = wan_residual_execute(module, backend, &hidden, "encoder.middle.0", context)?;
    hidden = wan_attention_execute(module, backend, &hidden, "encoder.middle.1", context)?;
    hidden = wan_residual_execute(module, backend, &hidden, "encoder.middle.2", context)?;
    hidden = rms_channel_norm(module, backend, &hidden, "encoder.head.0", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = wan_causal_convolution(module, backend, &hidden, "encoder.head.2.weight", context)?;
    hidden = wan_causal_convolution(module, backend, &hidden, "conv1.weight", context)?;
    narrow_contiguous(backend, &hidden, 1, 0, 48, context)
}

fn wan22_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = wan_causal_convolution(module, backend, input, "conv2.weight", context)?;
    hidden = wan_causal_convolution(module, backend, &hidden, "decoder.conv1.weight", context)?;
    hidden = wan_residual_execute(module, backend, &hidden, "decoder.middle.0", context)?;
    hidden = wan_attention_execute(module, backend, &hidden, "decoder.middle.1", context)?;
    hidden = wan_residual_execute(module, backend, &hidden, "decoder.middle.2", context)?;
    for level in 0..4 {
        hidden = wan22_up_block(
            module,
            backend,
            &hidden,
            &format!("decoder.upsamples.{level}"),
            level < 3,
            level < 2,
            context,
        )?;
    }
    hidden = rms_channel_norm(module, backend, &hidden, "decoder.head.0", context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = wan_causal_convolution(module, backend, &hidden, "decoder.head.2.weight", context)?;
    wan22_unpatchify(backend, &hidden, context)
}

fn wan22_down_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    downsample: bool,
    temporal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = input.clone();
    for block in 0..2 {
        hidden = wan_residual_execute(
            module,
            backend,
            &hidden,
            &format!("{prefix}.downsamples.{block}"),
            context,
        )?;
    }
    if downsample {
        hidden = wan_downsample(
            module,
            backend,
            &hidden,
            &format!("{prefix}.downsamples.2"),
            temporal,
            context,
        )?;
    }
    let output_channels = hidden
        .descriptor()
        .shape()
        .get(1)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let shortcut = wan22_avg_shortcut(
        backend,
        input,
        output_channels,
        if temporal { 2 } else { 1 },
        if downsample { 2 } else { 1 },
        context,
    )?;
    add_tensor(backend, &hidden, &shortcut, context)
}

fn wan22_up_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    upsample: bool,
    temporal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = input.clone();
    for block in 0..3 {
        hidden = wan_residual_execute(
            module,
            backend,
            &hidden,
            &format!("{prefix}.upsamples.{block}"),
            context,
        )?;
    }
    if !upsample {
        return Ok(hidden);
    }
    hidden = wan_upsample(
        module,
        backend,
        &hidden,
        &format!("{prefix}.upsamples.3"),
        temporal,
        context,
    )?;
    let output_channels = hidden
        .descriptor()
        .shape()
        .get(1)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let shortcut = wan22_dup_up(
        backend,
        input,
        output_channels,
        if temporal { 2 } else { 1 },
        2,
        temporal,
        context,
    )?;
    add_tensor(backend, &hidden, &shortcut, context)
}

fn wan22_patchify(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || !shape[3].is_multiple_of(2) || !shape[4].is_multiple_of(2) {
        return Err(VaeError::ShapeOverflow);
    }
    let reshaped = reshape_read_only(
        input,
        vec![
            shape[0],
            shape[1],
            shape[2],
            shape[3] / 2,
            2,
            shape[4] / 2,
            2,
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 6, 4, 2, 3, 5])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    let output_channels = shape[1].checked_mul(4).ok_or(VaeError::ShapeOverflow)?;
    reshape_read_only(
        &contiguous,
        vec![
            shape[0],
            output_channels,
            shape[2],
            shape[3] / 2,
            shape[4] / 2,
        ],
    )
}

fn wan22_unpatchify(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || !shape[1].is_multiple_of(4) {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = shape[1] / 4;
    let reshaped = reshape_read_only(
        input,
        vec![shape[0], channels, 2, 2, shape[2], shape[3], shape[4]],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 4, 5, 3, 6, 2])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    let output_height = shape[3].checked_mul(2).ok_or(VaeError::ShapeOverflow)?;
    let output_width = shape[4].checked_mul(2).ok_or(VaeError::ShapeOverflow)?;
    reshape_read_only(
        &contiguous,
        vec![shape[0], channels, shape[2], output_height, output_width],
    )
}

fn wan22_avg_shortcut(
    backend: &dyn TensorBackend,
    input: &Tensor,
    output_channels: u64,
    temporal_factor: u64,
    spatial_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if temporal_factor == 1 {
        return wan22_avg_down(
            backend,
            input,
            output_channels,
            temporal_factor,
            spatial_factor,
            context,
        );
    }
    let frames = input
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let first = narrow_contiguous(backend, input, 2, 0, 1, context)?;
    let first = wan22_avg_down(
        backend,
        &first,
        output_channels,
        temporal_factor,
        spatial_factor,
        context,
    )?;
    if frames == 1 {
        return Ok(first);
    }
    let rest = narrow_contiguous(backend, input, 2, 1, frames - 1, context)?;
    let rest = wan22_avg_down(
        backend,
        &rest,
        output_channels,
        temporal_factor,
        spatial_factor,
        context,
    )?;
    concatenate_temporal(backend, &[first, rest], context)
}

fn wan22_avg_down(
    backend: &dyn TensorBackend,
    input: &Tensor,
    output_channels: u64,
    temporal_factor: u64,
    spatial_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5
        || temporal_factor == 0
        || spatial_factor == 0
        || !shape[3].is_multiple_of(spatial_factor)
        || !shape[4].is_multiple_of(spatial_factor)
    {
        return Err(VaeError::ShapeOverflow);
    }
    let temporal_padding = (temporal_factor - shape[2] % temporal_factor) % temporal_factor;
    let padded = if temporal_padding == 0 {
        input.clone()
    } else {
        let descriptor = TensorDescriptor::contiguous(
            vec![shape[0], shape[1], temporal_padding, shape[3], shape[4]],
            input.descriptor().dtype(),
            input.descriptor().device(),
            context.stream,
        )?;
        let (zeros, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
        backend.wait_event(event, context)?;
        concatenate_temporal(backend, &[input.clone(), zeros], context)?
    };
    let padded_shape = padded.descriptor().shape();
    let reshaped = reshape_read_only(
        &padded,
        vec![
            padded_shape[0],
            padded_shape[1],
            padded_shape[2] / temporal_factor,
            temporal_factor,
            padded_shape[3] / spatial_factor,
            spatial_factor,
            padded_shape[4] / spatial_factor,
            spatial_factor,
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 3, 5, 7, 2, 4, 6])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    let packed_channels = padded_shape[1]
        .checked_mul(temporal_factor)
        .and_then(|channels| channels.checked_mul(spatial_factor))
        .and_then(|channels| channels.checked_mul(spatial_factor))
        .ok_or(VaeError::ShapeOverflow)?;
    let packed = reshape_read_only(
        &contiguous,
        vec![
            padded_shape[0],
            packed_channels,
            padded_shape[2] / temporal_factor,
            padded_shape[3] / spatial_factor,
            padded_shape[4] / spatial_factor,
        ],
    )?;
    grouped_channel_mean(backend, &packed, output_channels, context)
}

fn wan22_dup_up(
    backend: &dyn TensorBackend,
    input: &Tensor,
    output_channels: u64,
    temporal_factor: u64,
    spatial_factor: u64,
    trim_first_chunk: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let factor = temporal_factor
        .checked_mul(spatial_factor)
        .and_then(|value| value.checked_mul(spatial_factor))
        .ok_or(VaeError::ShapeOverflow)?;
    let expanded_channels = output_channels
        .checked_mul(factor)
        .ok_or(VaeError::ShapeOverflow)?;
    if shape.len() != 5
        || temporal_factor == 0
        || spatial_factor == 0
        || !expanded_channels.is_multiple_of(shape[1])
    {
        return Err(VaeError::ShapeOverflow);
    }
    let repeated =
        repeat_channels_interleave(backend, input, expanded_channels / shape[1], context)?;
    let reshaped = reshape_read_only(
        &repeated,
        vec![
            shape[0],
            output_channels,
            temporal_factor,
            spatial_factor,
            spatial_factor,
            shape[2],
            shape[3],
            shape[4],
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 1, 5, 2, 6, 3, 7, 4])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    let output = reshape_read_only(
        &contiguous,
        vec![
            shape[0],
            output_channels,
            shape[2]
                .checked_mul(temporal_factor)
                .ok_or(VaeError::ShapeOverflow)?,
            shape[3]
                .checked_mul(spatial_factor)
                .ok_or(VaeError::ShapeOverflow)?,
            shape[4]
                .checked_mul(spatial_factor)
                .ok_or(VaeError::ShapeOverflow)?,
        ],
    )?;
    if trim_first_chunk && temporal_factor > 1 {
        let frames = output
            .descriptor()
            .shape()
            .get(2)
            .copied()
            .ok_or(VaeError::ShapeOverflow)?;
        narrow_contiguous(
            backend,
            &output,
            2,
            i64::try_from(temporal_factor - 1)?,
            frames
                .checked_sub(temporal_factor - 1)
                .ok_or(VaeError::ShapeOverflow)?,
            context,
        )
    } else {
        Ok(output)
    }
}

fn wan_causal_convolution(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let convolution_module = crate::vae_image::find_module(module, name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing video VAE module {name}"
        )))
    })?;
    let (weight, _) = convolution_module.dense_parameters()?;
    let kernel = weight
        .descriptor()
        .shape()
        .get(2)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    let input = zero_temporal_pad(backend, input, kernel.saturating_sub(1), context)?;
    convolution(module, backend, &input, name, context)
}

fn wan_residual_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shortcut = format!("{prefix}.shortcut.weight");
    let residual = if crate::vae_image::find_module(module, &shortcut).is_some() {
        wan_causal_convolution(module, backend, input, &shortcut, context)?
    } else {
        input.clone()
    };
    let mut hidden = rms_channel_norm(
        module,
        backend,
        input,
        &format!("{prefix}.residual.0"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = wan_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.residual.2.weight"),
        context,
    )?;
    hidden = rms_channel_norm(
        module,
        backend,
        &hidden,
        &format!("{prefix}.residual.3"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = wan_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.residual.6.weight"),
        context,
    )?;
    add_tensor(backend, &residual, &hidden, context)
}

fn wan_attention_execute(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = split_video_frames(backend, input, context)?;
    let frames = frames
        .into_iter()
        .map(|frame| {
            let normalized =
                rms_channel_norm(module, backend, &frame, &format!("{prefix}.norm"), context)?;
            let qkv = convolution(
                module,
                backend,
                &normalized,
                &format!("{prefix}.to_qkv.weight"),
                context,
            )?;
            let channels = frame.descriptor().shape()[1];
            let query = narrow_contiguous(backend, &qkv, 1, 0, channels, context)?;
            let key = narrow_contiguous(
                backend,
                &qkv,
                1,
                i64::try_from(channels)?,
                channels,
                context,
            )?;
            let value = narrow_contiguous(
                backend,
                &qkv,
                1,
                i64::try_from(channels.checked_mul(2).ok_or(VaeError::ShapeOverflow)?)?,
                channels,
                context,
            )?;
            let attended =
                spatial_attention_from_qkv(backend, &frame, &query, &key, &value, context)?;
            let projected = convolution(
                module,
                backend,
                &attended,
                &format!("{prefix}.proj.weight"),
                context,
            )?;
            add_tensor(backend, &frame, &projected, context)
        })
        .collect::<Result<Vec<_>, VaeError>>()?;
    stack_video_frames(backend, &frames, context)
}

fn wan_downsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    temporal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let frames = split_video_frames(backend, input, context)?;
    let frames = frames
        .into_iter()
        .map(|frame| {
            let frame = constant_pad_bottom_right(backend, &frame, context)?;
            convolution(
                module,
                backend,
                &frame,
                &format!("{prefix}.resample.1.weight"),
                context,
            )
        })
        .collect::<Result<Vec<_>, VaeError>>()?;
    let spatial = stack_video_frames(backend, &frames, context)?;
    if !temporal {
        return Ok(spatial);
    }
    let frames = split_video_frames(backend, &spatial, context)?;
    if frames.is_empty() || !frames.len().saturating_sub(1).is_multiple_of(2) {
        return Err(VaeError::ShapeOverflow);
    }
    let mut output = vec![narrow_contiguous(backend, &spatial, 2, 0, 1, context)?];
    for index in (1..frames.len()).step_by(2) {
        let window = stack_video_frames(
            backend,
            &[
                frames[index - 1].clone(),
                frames[index].clone(),
                frames[index + 1].clone(),
            ],
            context,
        )?;
        output.push(convolution(
            module,
            backend,
            &window,
            &format!("{prefix}.time_conv.weight"),
            context,
        )?);
    }
    concatenate_temporal(backend, &output, context)
}

fn wan_upsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    temporal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let input = if temporal {
        let first = narrow_contiguous(backend, input, 2, 0, 1, context)?;
        let frames = input.descriptor().shape()[2];
        if frames == 1 {
            first
        } else {
            let rest = narrow_contiguous(backend, input, 2, 1, frames - 1, context)?;
            let grown = wan_causal_convolution(
                module,
                backend,
                &rest,
                &format!("{prefix}.time_conv.weight"),
                context,
            )?;
            let grown = channel_to_time(backend, &grown, 2, context)?;
            concatenate_temporal(backend, &[first, grown], context)?
        }
    } else {
        input.clone()
    };
    let frames = split_video_frames(backend, &input, context)?;
    let frames = frames
        .into_iter()
        .map(|frame| {
            let frame = nearest_upsample_2x(backend, &frame, context)?;
            convolution(
                module,
                backend,
                &frame,
                &format!("{prefix}.resample.1.weight"),
                context,
            )
        })
        .collect::<Result<Vec<_>, VaeError>>()?;
    stack_video_frames(backend, &frames, context)
}

fn channel_to_time(
    backend: &dyn TensorBackend,
    input: &Tensor,
    factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || factor == 0 || !shape[1].is_multiple_of(factor) {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = shape[1] / factor;
    let frames = split_video_frames(backend, input, context)?;
    let mut output = Vec::new();
    for frame in frames {
        for index in 0..factor {
            output.push(narrow_contiguous(
                backend,
                &frame,
                1,
                i64::try_from(index * channels)?,
                channels,
                context,
            )?);
        }
    }
    stack_video_frames(backend, &output, context)
}

fn zero_temporal_pad(
    backend: &dyn TensorBackend,
    input: &Tensor,
    padding: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if padding == 0 {
        return Ok(input.clone());
    }
    let shape = input.descriptor().shape();
    if shape.len() != 5 {
        return Err(VaeError::ShapeOverflow);
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[1], padding, shape[3], shape[4]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (zeros, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    concatenate_temporal(backend, &[zeros, input.clone()], context)
}

fn rms_channel_norm(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if !matches!(shape.len(), 4 | 5) || shape[1] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let (squared, event) = backend.binary(
        BinaryOperation::Multiply,
        input,
        input,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let mut mean_shape = shape.to_vec();
    mean_shape[1] = 1;
    let mean_descriptor = TensorDescriptor::contiguous(
        mean_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mean, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Mean,
            dimensions: vec![1],
            keep_dimensions: true,
            accumulation_dtype: Some(input.descriptor().dtype()),
            correction: 0,
        },
        &squared,
        mean_descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let minimum = 1.0e-24 / shape[1] as f64;
    let (mean, event) = backend.binary_scalar(
        BinaryOperation::Maximum,
        &mean,
        Scalar::Float(minimum),
        ScalarSide::Right,
        mean.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let (inverse, event) = backend.unary(
        UnaryOperation::ReciprocalSquareRoot,
        &mean,
        mean.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let (normalized, event) = backend.binary(
        BinaryOperation::Multiply,
        input,
        &inverse,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let gamma_name = format!("{prefix}.gamma");
    let gamma = crate::vae_image::find_module(module, &gamma_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing video VAE module {gamma_name}"
            )))
        })?;
    let mut gamma_shape = vec![1];
    gamma_shape.extend_from_slice(gamma.descriptor().shape());
    let gamma = reshape_read_only(gamma, gamma_shape)?;
    let (output, event) = backend.binary(
        BinaryOperation::Multiply,
        &normalized,
        &gamma,
        normalized.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn refiner_downsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    temporal: bool,
    carried: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let convolved = refiner_convolution_execute(module, backend, input, prefix, carried, context)?;
    let temporal_factor = if temporal { 2 } else { 1 };
    if temporal && carried {
        let convolved_first = narrow_contiguous(backend, &convolved, 2, 0, 1, context)?;
        let mut hidden_first = space_to_depth_3d(backend, &convolved_first, 1, context)?;
        hidden_first = repeat_channels_interleave(backend, &hidden_first, 2, context)?;
        let input_first = narrow_contiguous(backend, input, 2, 0, 1, context)?;
        let input_first = space_to_depth_3d(backend, &input_first, 1, context)?;
        let residual_first = grouped_channel_mean(
            backend,
            &input_first,
            hidden_first.descriptor().shape()[1],
            context,
        )?;
        let first = add_tensor(backend, &hidden_first, &residual_first, context)?;
        let frames = input.descriptor().shape()[2];
        if frames == 1 {
            return Ok(first);
        }
        let rest_length = frames.checked_sub(1).ok_or(VaeError::ShapeOverflow)?;
        if !rest_length.is_multiple_of(2) {
            return Err(VaeError::ShapeOverflow);
        }
        let convolved_rest = narrow_contiguous(backend, &convolved, 2, 1, rest_length, context)?;
        let hidden_rest = space_to_depth_3d(backend, &convolved_rest, 2, context)?;
        let input_rest = narrow_contiguous(backend, input, 2, 1, rest_length, context)?;
        let input_rest = space_to_depth_3d(backend, &input_rest, 2, context)?;
        let residual_rest = grouped_channel_mean(
            backend,
            &input_rest,
            hidden_rest.descriptor().shape()[1],
            context,
        )?;
        let rest = add_tensor(backend, &hidden_rest, &residual_rest, context)?;
        return concatenate_temporal(backend, &[first, rest], context);
    }
    let hidden = space_to_depth_3d(backend, &convolved, temporal_factor, context)?;
    let residual = space_to_depth_3d(backend, input, temporal_factor, context)?;
    let residual =
        grouped_channel_mean(backend, &residual, hidden.descriptor().shape()[1], context)?;
    add_tensor(backend, &hidden, &residual, context)
}

fn refiner_upsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    temporal: bool,
    carried: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let convolved = refiner_convolution_execute(module, backend, input, prefix, carried, context)?;
    let temporal_factor = if temporal { 2 } else { 1 };
    let output_channels = convolved.descriptor().shape()[1]
        .checked_div(temporal_factor * 4)
        .ok_or(VaeError::ShapeOverflow)?;
    if output_channels == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let repeat = output_channels
        .checked_mul(temporal_factor * 4)
        .and_then(|channels| channels.checked_div(input.descriptor().shape()[1]))
        .ok_or(VaeError::ShapeOverflow)?;
    if temporal && carried {
        let convolved_first = narrow_contiguous(backend, &convolved, 2, 0, 1, context)?;
        let hidden_first = depth_to_space_3d(backend, &convolved_first, 1, context)?;
        let hidden_first =
            narrow_contiguous(backend, &hidden_first, 1, 0, output_channels, context)?;
        let input_first = narrow_contiguous(backend, input, 2, 0, 1, context)?;
        let input_first = repeat_channels_interleave(backend, &input_first, repeat / 2, context)?;
        let residual_first = depth_to_space_3d(backend, &input_first, 1, context)?;
        let first = add_tensor(backend, &hidden_first, &residual_first, context)?;
        let frames = input.descriptor().shape()[2];
        if frames == 1 {
            return Ok(first);
        }
        let rest_length = frames.checked_sub(1).ok_or(VaeError::ShapeOverflow)?;
        let convolved_rest = narrow_contiguous(backend, &convolved, 2, 1, rest_length, context)?;
        let hidden_rest = depth_to_space_3d(backend, &convolved_rest, 2, context)?;
        let input_rest = narrow_contiguous(backend, input, 2, 1, rest_length, context)?;
        let input_rest = repeat_channels_interleave(backend, &input_rest, repeat, context)?;
        let residual_rest = depth_to_space_3d(backend, &input_rest, 2, context)?;
        let rest = add_tensor(backend, &hidden_rest, &residual_rest, context)?;
        return concatenate_temporal(backend, &[first, rest], context);
    }
    let hidden = depth_to_space_3d(backend, &convolved, temporal_factor, context)?;
    let residual = repeat_channels_interleave(backend, input, repeat, context)?;
    let residual = depth_to_space_3d(backend, &residual, temporal_factor, context)?;
    add_tensor(backend, &hidden, &residual, context)
}

fn space_to_depth_3d(
    backend: &dyn TensorBackend,
    input: &Tensor,
    temporal_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5
        || temporal_factor == 0
        || !shape[2].is_multiple_of(temporal_factor)
        || !shape[3].is_multiple_of(2)
        || !shape[4].is_multiple_of(2)
    {
        return Err(VaeError::ShapeOverflow);
    }
    let reshaped = reshape_read_only(
        input,
        vec![
            shape[0],
            shape[1],
            shape[2] / temporal_factor,
            temporal_factor,
            shape[3] / 2,
            2,
            shape[4] / 2,
            2,
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 3, 5, 7, 1, 2, 4, 6])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    let packed_channels = temporal_factor
        .checked_mul(4)
        .and_then(|factor| factor.checked_mul(shape[1]))
        .ok_or(VaeError::ShapeOverflow)?;
    reshape_read_only(
        &contiguous,
        vec![
            shape[0],
            packed_channels,
            shape[2] / temporal_factor,
            shape[3] / 2,
            shape[4] / 2,
        ],
    )
}

fn depth_to_space_3d(
    backend: &dyn TensorBackend,
    input: &Tensor,
    temporal_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let factor = temporal_factor
        .checked_mul(4)
        .ok_or(VaeError::ShapeOverflow)?;
    if shape.len() != 5 || temporal_factor == 0 || !shape[1].is_multiple_of(factor) {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = shape[1] / factor;
    let reshaped = reshape_read_only(
        input,
        vec![
            shape[0],
            temporal_factor,
            2,
            2,
            channels,
            shape[2],
            shape[3],
            shape[4],
        ],
    )?;
    let permuted = permute_read_only(&reshaped, &[0, 4, 5, 1, 6, 2, 7, 3])?;
    let contiguous = contiguous_copy(backend, &permuted, context)?;
    let frames = shape[2]
        .checked_mul(temporal_factor)
        .ok_or(VaeError::ShapeOverflow)?;
    let height = shape[3].checked_mul(2).ok_or(VaeError::ShapeOverflow)?;
    let width = shape[4].checked_mul(2).ok_or(VaeError::ShapeOverflow)?;
    reshape_read_only(&contiguous, vec![shape[0], channels, frames, height, width])
}

pub(crate) fn grouped_channel_mean(
    backend: &dyn TensorBackend,
    input: &Tensor,
    output_channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if !(3..=5).contains(&shape.len())
        || output_channels == 0
        || !shape[1].is_multiple_of(output_channels)
    {
        return Err(VaeError::ShapeOverflow);
    }
    let mut grouped_shape = vec![shape[0], output_channels, shape[1] / output_channels];
    grouped_shape.extend_from_slice(&shape[2..]);
    let grouped = reshape_read_only(input, grouped_shape)?;
    let mut output_shape = vec![shape[0], output_channels];
    output_shape.extend_from_slice(&shape[2..]);
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Mean,
            dimensions: vec![2],
            keep_dimensions: false,
            accumulation_dtype: Some(input.descriptor().dtype()),
            correction: 0,
        },
        &grouped,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn repeat_temporal(
    backend: &dyn TensorBackend,
    input: &Tensor,
    repeats: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape[2] != 1 || repeats == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[1], repeats, shape[3], shape[4]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    for frame in 0..repeats {
        context.check()?;
        let (updated, event) =
            backend.replace_rectangular_slice(&output, input, &[0, 0, frame, 0, 0], context)?;
        backend.wait_event(event, context)?;
        output = updated;
    }
    Ok(output)
}

pub(crate) fn repeat_channels_interleave(
    backend: &dyn TensorBackend,
    input: &Tensor,
    repeats: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if !(3..=5).contains(&shape.len()) || repeats == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = shape[1]
        .checked_mul(repeats)
        .ok_or(VaeError::ShapeOverflow)?;
    let mut output_shape = vec![shape[0], channels];
    output_shape.extend_from_slice(&shape[2..]);
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    for channel in 0..shape[1] {
        let source = input.narrow_read_only(1, i64::try_from(channel)?, 1)?;
        for repeat in 0..repeats {
            context.check()?;
            let offset = channel
                .checked_mul(repeats)
                .and_then(|value| value.checked_add(repeat))
                .ok_or(VaeError::ShapeOverflow)?;
            let mut offsets = vec![0; shape.len()];
            offsets[1] = offset;
            let (updated, event) =
                backend.replace_rectangular_slice(&output, &source, &offsets, context)?;
            backend.wait_event(event, context)?;
            output = updated;
        }
    }
    Ok(output)
}

fn concatenate_temporal(
    backend: &dyn TensorBackend,
    inputs: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let first = inputs.first().ok_or(VaeError::ShapeOverflow)?;
    let shape = first.descriptor().shape();
    if shape.len() != 5 {
        return Err(VaeError::ShapeOverflow);
    }
    let frames = inputs.iter().try_fold(0_u64, |frames, input| {
        let input_shape = input.descriptor().shape();
        if input_shape.len() != 5
            || input_shape[0..2] != shape[0..2]
            || input_shape[3..5] != shape[3..5]
            || input.descriptor().dtype() != first.descriptor().dtype()
            || input.descriptor().device() != first.descriptor().device()
        {
            return Err(VaeError::ShapeOverflow);
        }
        frames
            .checked_add(input_shape[2])
            .ok_or(VaeError::ShapeOverflow)
    })?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[1], frames, shape[3], shape[4]],
        first.descriptor().dtype(),
        first.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let mut offset = 0_u64;
    for input in inputs {
        context.check()?;
        let (updated, event) =
            backend.replace_rectangular_slice(&output, input, &[0, 0, offset, 0, 0], context)?;
        backend.wait_event(event, context)?;
        output = updated;
        offset = offset
            .checked_add(input.descriptor().shape()[2])
            .ok_or(VaeError::ShapeOverflow)?;
    }
    Ok(output)
}

pub(crate) fn narrow_contiguous(
    backend: &dyn TensorBackend,
    input: &Tensor,
    dimension: usize,
    start: i64,
    length: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let narrowed = input.narrow_read_only(dimension, start, length)?;
    contiguous_copy(backend, &narrowed, context)
}

pub(crate) fn permute_read_only(input: &Tensor, permutation: &[usize]) -> Result<Tensor, VaeError> {
    let descriptor = input.descriptor().permuted_view(permutation)?;
    Ok(input.view(descriptor, ViewAccess::ReadOnly)?)
}

pub(crate) fn contiguous_copy(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.copy(input, descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn causal_replicate_pad_3d(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape.contains(&0) {
        return Err(VaeError::ShapeOverflow);
    }
    let padded_frames = shape[2].checked_add(2).ok_or(VaeError::ShapeOverflow)?;
    let padded_height = shape[3].checked_add(2).ok_or(VaeError::ShapeOverflow)?;
    let padded_width = shape[4].checked_add(2).ok_or(VaeError::ShapeOverflow)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![
            shape[0],
            shape[1],
            padded_frames,
            padded_height,
            padded_width,
        ],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    for target_time in 0..padded_frames {
        let source_time = target_time.saturating_sub(2).min(shape[2] - 1);
        let source_frame = input.narrow_read_only(2, i64::try_from(source_time)?, 1)?;
        let (updated, event) = backend.replace_rectangular_slice(
            &output,
            &source_frame,
            &[0, 0, target_time, 1, 1],
            context,
        )?;
        backend.wait_event(event, context)?;
        output = updated;
    }
    let interior_width = output.narrow_read_only(4, 1, shape[4])?;
    let top = interior_width.narrow_read_only(3, 1, 1)?;
    let bottom = interior_width.narrow_read_only(3, i64::try_from(shape[3])?, 1)?;
    let top = contiguous_copy(backend, &top, context)?;
    let bottom = contiguous_copy(backend, &bottom, context)?;
    for (source, y) in [(&top, 0_u64), (&bottom, padded_height - 1)] {
        let (updated, event) =
            backend.replace_rectangular_slice(&output, source, &[0, 0, 0, y, 1], context)?;
        backend.wait_event(event, context)?;
        output = updated;
    }
    let left = output.narrow_read_only(4, 1, 1)?;
    let right = output.narrow_read_only(4, i64::try_from(shape[4])?, 1)?;
    let left = contiguous_copy(backend, &left, context)?;
    let right = contiguous_copy(backend, &right, context)?;
    for (source, x) in [(&left, 0_u64), (&right, padded_width - 1)] {
        let (updated, event) =
            backend.replace_rectangular_slice(&output, source, &[0, 0, 0, 0, x], context)?;
        backend.wait_event(event, context)?;
        output = updated;
    }
    Ok(output)
}

fn build_taehv_module(
    architecture: &NativeVideoVaeArchitecture,
    mut state: std::collections::BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    descriptor: &VaeDescriptor,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, VideoVaeError> {
    let mut children = Vec::new();
    for spec in video_vae_source_state_schema(
        architecture.profile(),
        architecture.storage_dtype().ok_or_else(|| {
            VideoVaeError::MissingState("video VAE storage dtype sentinel".to_owned())
        })?,
    )? {
        let Some(prefix) = spec.name.strip_suffix(".weight") else {
            continue;
        };
        if spec.shape.len() != 4 {
            return Err(VideoVaeError::InvalidStateShape {
                name: spec.name,
                shape: spec.shape,
            });
        }
        let stride = if matches!(prefix, "encoder.3" | "encoder.8" | "encoder.13") {
            2
        } else {
            1
        };
        let kernel =
            usize::try_from(spec.shape[2]).map_err(|_| VideoVaeError::InvalidStateShape {
                name: spec.name.clone(),
                shape: spec.shape.clone(),
            })?;
        let geometry = ConvolutionGeometry::new(
            2,
            vec![stride, stride],
            vec![kernel / 2, kernel / 2],
            vec![1, 1],
            1,
            false,
            vec![0, 0],
        )
        .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
        let bias_name = format!("{prefix}.bias");
        let bias = state.remove(&bias_name);
        let weight = state
            .remove(&spec.name)
            .ok_or_else(|| VideoVaeError::MissingState(spec.name.clone()))?;
        let mut module = NativeModule::convolution(
            spec.name,
            usize::try_from(spec.shape[1]).map_err(|_| VaeError::ShapeOverflow)?,
            usize::try_from(spec.shape[0]).map_err(|_| VaeError::ShapeOverflow)?,
            vec![kernel, kernel],
            bias.is_some(),
            geometry,
            false,
        )?;
        module.load_dense_parameters(weight, bias)?;
        children.push(module);
    }
    if let Some(name) = state.keys().next() {
        return Err(VideoVaeError::UnexpectedState(name.clone()));
    }
    let mut module =
        NativeModule::module_dict(format!("video-vae:{:?}", architecture.profile()), children)?;
    module.materialize_execution_state_with_context(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
        context,
    )?;
    Ok(module)
}

fn taehv_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let topology = taehv_module_topology(module)?;
    let temporal_padding =
        topology.temporal_ratio - input.descriptor().shape()[2] % topology.temporal_ratio;
    let temporal_padding = if temporal_padding == topology.temporal_ratio {
        0
    } else {
        temporal_padding
    };
    let mut frames = split_video_frames(backend, input, context)?;
    if temporal_padding > 0 {
        let last = frames.last().cloned().ok_or(VaeError::ShapeOverflow)?;
        frames.extend(std::iter::repeat_n(
            last,
            usize::try_from(temporal_padding)?,
        ));
    }
    if topology.patch_size > 1 {
        frames = frames
            .into_iter()
            .map(|frame| pixel_unshuffle(backend, &frame, topology.patch_size, context))
            .collect::<Result<Vec<_>, _>>()?;
    }
    frames = apply_convolution_activation(
        module,
        backend,
        frames,
        "encoder.0.weight",
        topology.leaky,
        context,
    )?;
    for (stage, (pool_index, convolution_index, memory_indices)) in [
        (2_u64, 3_u64, [4_u64, 5, 6]),
        (7, 8, [9, 10, 11]),
        (12, 13, [14, 15, 16]),
    ]
    .into_iter()
    .enumerate()
    {
        frames = temporal_pool(
            module,
            backend,
            frames,
            &format!("encoder.{pool_index}.conv.weight"),
            topology.encoder_temporal_strides[stage],
            context,
        )?;
        frames = apply_convolution(
            module,
            backend,
            frames,
            &format!("encoder.{convolution_index}.weight"),
            context,
        )?;
        for index in memory_indices {
            frames = apply_memory_block(
                module,
                backend,
                frames,
                &format!("encoder.{index}"),
                topology.leaky,
                context,
            )?;
        }
    }
    frames = apply_convolution(module, backend, frames, "encoder.17.weight", context)?;
    stack_video_frames(backend, &frames, context)
}

fn taehv_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    _cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let topology = taehv_module_topology(module)?;
    let mut frames = split_video_frames(backend, input, context)?;
    frames = frames
        .into_iter()
        .map(|frame| clamp_three(backend, &frame, context))
        .collect::<Result<Vec<_>, _>>()?;
    frames = apply_convolution_activation(
        module,
        backend,
        frames,
        "decoder.1.weight",
        topology.leaky,
        context,
    )?;
    for index in [3_u64, 4, 5] {
        frames = apply_memory_block(
            module,
            backend,
            frames,
            &format!("decoder.{index}"),
            topology.leaky,
            context,
        )?;
    }
    for (stage, (grow_index, convolution_index, memory_indices)) in [
        (7_u64, 8_u64, vec![9_u64, 10, 11]),
        (13, 14, vec![15, 16, 17]),
        (19, 20, Vec::new()),
    ]
    .into_iter()
    .enumerate()
    {
        frames = frames
            .into_iter()
            .map(|frame| nearest_upsample_2x(backend, &frame, context))
            .collect::<Result<Vec<_>, _>>()?;
        frames = temporal_grow(
            module,
            backend,
            frames,
            &format!("decoder.{grow_index}.conv.weight"),
            topology.decoder_temporal_strides[stage],
            context,
        )?;
        frames = apply_convolution(
            module,
            backend,
            frames,
            &format!("decoder.{convolution_index}.weight"),
            context,
        )?;
        for index in memory_indices {
            frames = apply_memory_block(
                module,
                backend,
                frames,
                &format!("decoder.{index}"),
                topology.leaky,
                context,
            )?;
        }
    }
    frames = apply_activation(backend, frames, topology.leaky, context)?;
    frames = apply_convolution(module, backend, frames, "decoder.22.weight", context)?;
    if topology.patch_size > 1 {
        frames = frames
            .into_iter()
            .map(|frame| pixel_shuffle(backend, &frame, topology.patch_size, context))
            .collect::<Result<Vec<_>, _>>()?;
    }
    let trim = usize::try_from(topology.temporal_ratio - 1)?;
    if frames.len() <= trim {
        return Err(VaeError::ShapeOverflow);
    }
    let frames = frames.into_iter().skip(trim).collect::<Vec<_>>();
    stack_video_frames(backend, &frames, context)
}

struct ExecutableTaeHvTopology {
    patch_size: u64,
    temporal_ratio: u64,
    encoder_temporal_strides: [u64; 3],
    decoder_temporal_strides: [u64; 3],
    leaky: bool,
}

fn taehv_module_topology(module: &NativeModule) -> Result<ExecutableTaeHvTopology, VaeError> {
    let name = module.layer_name();
    let (profile, leaky) = if name.contains("TaeHvLtx2V1") {
        (VaeKernelProfile::TaeHvLtx2V1, false)
    } else if name.contains("TaeHvWan22V1") {
        (VaeKernelProfile::TaeHvWan22V1, false)
    } else if name.contains("LightTaeHv15V1") {
        (VaeKernelProfile::LightTaeHv15V1, true)
    } else if name.contains("TaeHvHunyuanV1") {
        (VaeKernelProfile::TaeHvHunyuanV1, false)
    } else if name.contains("LightTaeWan21V1") {
        (VaeKernelProfile::LightTaeWan21V1, false)
    } else {
        return Err(VaeError::KernelProfileMismatch);
    };
    let topology = taehv_topology(&profile).map_err(|_| VaeError::KernelProfileMismatch)?;
    Ok(ExecutableTaeHvTopology {
        patch_size: topology.patch_size,
        temporal_ratio: topology.encoder_temporal_strides.iter().product(),
        encoder_temporal_strides: topology.encoder_temporal_strides,
        decoder_temporal_strides: topology.decoder_temporal_strides,
        leaky,
    })
}

fn split_video_frames(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 5 || shape.contains(&0) {
        return Err(VaeError::InvalidShape {
            expected: vec![0, 0, 0, 0, 0],
            actual: shape.to_vec(),
        });
    }
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(usize::try_from(shape[2])?)
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    for frame_index in 0..shape[2] {
        context.check()?;
        let frame = input.narrow_read_only(2, i64::try_from(frame_index)?, 1)?;
        let descriptor = TensorDescriptor::contiguous(
            frame.descriptor().shape().to_vec(),
            frame.descriptor().dtype(),
            frame.descriptor().device(),
            context.stream,
        )?;
        let (frame, event) = backend.copy(&frame, descriptor, context)?;
        backend.wait_event(event, context)?;
        frames.push(reshape_read_only(
            &frame,
            vec![shape[0], shape[1], shape[3], shape[4]],
        )?);
    }
    Ok(frames)
}

fn stack_video_frames(
    backend: &dyn TensorBackend,
    frames: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let first = frames.first().ok_or(VaeError::ShapeOverflow)?;
    let shape = first.descriptor().shape();
    if shape.len() != 4
        || frames
            .iter()
            .any(|frame| frame.descriptor().shape() != shape)
    {
        return Err(VaeError::ShapeOverflow);
    }
    let output_shape = vec![
        shape[0],
        shape[1],
        u64::try_from(frames.len())?,
        shape[2],
        shape[3],
    ];
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        first.descriptor().dtype(),
        first.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    for (index, frame) in frames.iter().enumerate() {
        context.check()?;
        let frame = reshape_read_only(frame, vec![shape[0], shape[1], 1, shape[2], shape[3]])?;
        let (updated, event) = backend.replace_rectangular_slice(
            &output,
            &frame,
            &[0, 0, u64::try_from(index)?, 0, 0],
            context,
        )?;
        backend.wait_event(event, context)?;
        output = updated;
    }
    Ok(output)
}

fn concatenate_channels(
    backend: &dyn TensorBackend,
    inputs: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let first = inputs.first().ok_or(VaeError::ShapeOverflow)?;
    let shape = first.descriptor().shape();
    if shape.len() != 4 {
        return Err(VaeError::ShapeOverflow);
    }
    let channels = inputs
        .iter()
        .try_fold(0_u64, |sum, input| {
            let input_shape = input.descriptor().shape();
            if input_shape.len() != 4
                || input_shape[0] != shape[0]
                || input_shape[2..] != shape[2..]
            {
                return None;
            }
            sum.checked_add(input_shape[1])
        })
        .ok_or(VaeError::ShapeOverflow)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], channels, shape[2], shape[3]],
        first.descriptor().dtype(),
        first.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let mut channel_offset = 0_u64;
    for input in inputs {
        let (updated, event) = backend.replace_rectangular_slice(
            &output,
            input,
            &[0, channel_offset, 0, 0],
            context,
        )?;
        backend.wait_event(event, context)?;
        output = updated;
        channel_offset = channel_offset
            .checked_add(input.descriptor().shape()[1])
            .ok_or(VaeError::ShapeOverflow)?;
    }
    Ok(output)
}

fn apply_convolution(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    frames: Vec<Tensor>,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, VaeError> {
    frames
        .into_iter()
        .map(|frame| convolution(module, backend, &frame, name, context))
        .collect()
}

fn apply_activation(
    backend: &dyn TensorBackend,
    frames: Vec<Tensor>,
    leaky: bool,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, VaeError> {
    frames
        .into_iter()
        .map(|frame| activation(backend, &frame, leaky, context))
        .collect()
}

fn apply_convolution_activation(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    frames: Vec<Tensor>,
    name: &str,
    leaky: bool,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, VaeError> {
    apply_activation(
        backend,
        apply_convolution(module, backend, frames, name, context)?,
        leaky,
        context,
    )
}

fn activation(
    backend: &dyn TensorBackend,
    input: &Tensor,
    leaky: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if !leaky {
        return relu_tensor(backend, input, context);
    }
    let positive = relu_tensor(backend, input, context)?;
    let negative = unary_tensor(backend, input, UnaryOperation::Negate, context)?;
    let negative = relu_tensor(backend, &negative, context)?;
    let (negative, event) = backend.binary_scalar(
        BinaryOperation::Multiply,
        &negative,
        Scalar::Float(-0.2),
        ScalarSide::Right,
        negative.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    add_tensor(backend, &positive, &negative, context)
}

fn apply_memory_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    frames: Vec<Tensor>,
    prefix: &str,
    leaky: bool,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, VaeError> {
    let first = frames.first().ok_or(VaeError::ShapeOverflow)?;
    let descriptor = TensorDescriptor::contiguous(
        first.descriptor().shape().to_vec(),
        first.descriptor().dtype(),
        first.descriptor().device(),
        context.stream,
    )?;
    let (mut past, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(frames.len())
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    for frame in frames {
        context.check()?;
        let input = concatenate_channels(backend, &[frame.clone(), past], context)?;
        let mut hidden = convolution(
            module,
            backend,
            &input,
            &format!("{prefix}.conv.0.weight"),
            context,
        )?;
        hidden = activation(backend, &hidden, leaky, context)?;
        hidden = convolution(
            module,
            backend,
            &hidden,
            &format!("{prefix}.conv.2.weight"),
            context,
        )?;
        hidden = activation(backend, &hidden, leaky, context)?;
        hidden = convolution(
            module,
            backend,
            &hidden,
            &format!("{prefix}.conv.4.weight"),
            context,
        )?;
        hidden = add_tensor(backend, &hidden, &frame, context)?;
        hidden = activation(backend, &hidden, leaky, context)?;
        past = frame;
        output.push(hidden);
    }
    Ok(output)
}

fn temporal_pool(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    frames: Vec<Tensor>,
    name: &str,
    stride: u64,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, VaeError> {
    if stride == 1 {
        return apply_convolution(module, backend, frames, name, context);
    }
    let stride = usize::try_from(stride)?;
    if !frames.len().is_multiple_of(stride) {
        return Err(VaeError::ShapeOverflow);
    }
    frames
        .chunks(stride)
        .map(|chunk| {
            let input = concatenate_channels(backend, chunk, context)?;
            convolution(module, backend, &input, name, context)
        })
        .collect()
}

fn temporal_grow(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    frames: Vec<Tensor>,
    name: &str,
    stride: u64,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, VaeError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(
            frames
                .len()
                .checked_mul(usize::try_from(stride)?)
                .ok_or(VaeError::ShapeOverflow)?,
        )
        .map_err(|error| VaeError::Allocation(error.to_string()))?;
    for frame in frames {
        let grown = convolution(module, backend, &frame, name, context)?;
        let channels = grown.descriptor().shape()[1] / stride;
        if channels == 0 || grown.descriptor().shape()[1] != channels * stride {
            return Err(VaeError::ShapeOverflow);
        }
        for index in 0..stride {
            let narrowed = grown.narrow_read_only(1, i64::try_from(index * channels)?, channels)?;
            let descriptor = TensorDescriptor::contiguous(
                narrowed.descriptor().shape().to_vec(),
                narrowed.descriptor().dtype(),
                narrowed.descriptor().device(),
                context.stream,
            )?;
            let (frame, event) = backend.copy(&narrowed, descriptor, context)?;
            backend.wait_event(event, context)?;
            output.push(frame);
        }
    }
    Ok(output)
}

fn clamp_three(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (scaled, event) = backend.binary_scalar(
        BinaryOperation::Divide,
        input,
        Scalar::Float(3.0),
        ScalarSide::Right,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let bounded = unary_tensor(backend, &scaled, UnaryOperation::HyperbolicTangent, context)?;
    let (output, event) = backend.binary_scalar(
        BinaryOperation::Multiply,
        &bounded,
        Scalar::Float(3.0),
        ScalarSide::Right,
        bounded.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{
        CpuWorkspaceAuthority, DecodedScalar, DeviceId, RetryRngPolicy, RngStreamAddress, StreamId,
    };
    use sim_types::CancellationToken;

    fn upload(
        backend: &CpuBackend,
        shape: &[u64],
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (tensor, event) = backend.upload_f32(descriptor, values, context)?;
        backend.wait_event(event, context)?;
        Ok(tensor)
    }

    fn convolution_module(
        backend: &CpuBackend,
        name: &str,
        output_channels: usize,
        input_channels: usize,
        weights: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<NativeModule, Box<dyn std::error::Error>> {
        let geometry =
            ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, false, vec![0, 0])?;
        let mut module = NativeModule::convolution(
            name,
            input_channels,
            output_channels,
            vec![1, 1],
            false,
            geometry,
            false,
        )?;
        module.load_dense_parameters(
            upload(
                backend,
                &[
                    u64::try_from(output_channels)?,
                    u64::try_from(input_channels)?,
                    1,
                    1,
                ],
                weights,
                context,
            )?,
            None,
        )?;
        Ok(module)
    }

    fn values(tensor: &Tensor) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        (0..tensor.descriptor().element_count()?)
            .map(|index| {
                match tensor
                    .descriptor()
                    .dtype()
                    .decode_scalar(tensor.linear_element_bytes(index)?)?
                {
                    DecodedScalar::Real(value) => Ok(value),
                    _ => Err("expected real tensor value".into()),
                }
            })
            .collect()
    }

    #[test]
    fn val_vae_001_video_frame_split_stack_is_lossless_and_cancellable()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let input = upload(
            &backend,
            &[1, 1, 3, 1, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &context,
        )?;
        let frames = split_video_frames(&backend, &input, &context)?;
        assert_eq!(frames.len(), 3);
        assert_eq!(
            stack_video_frames(&backend, &frames, &context)?.contiguous_bytes()?,
            input.contiguous_bytes()?
        );

        cancellation.cancel();
        assert!(split_video_frames(&backend, &input, &context).is_err());
        Ok(())
    }

    #[test]
    fn val_vae_001_taehv_temporal_pool_and_grow_preserve_source_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let pool = convolution_module(&backend, "pool.weight", 1, 2, &[1.0, 10.0], &context)?;
        let grow = convolution_module(&backend, "grow.weight", 2, 1, &[1.0, 10.0], &context)?;
        let module = NativeModule::module_dict("test", vec![pool, grow])?;
        let frames = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(|value| upload(&backend, &[1, 1, 1, 1], &[value], &context))
            .collect::<Result<Vec<_>, _>>()?;
        let pooled = temporal_pool(&module, &backend, frames, "pool.weight", 2, &context)?;
        assert_eq!(
            values(&stack_video_frames(&backend, &pooled, &context)?)?,
            [21.0, 43.0]
        );
        let grown = temporal_grow(&module, &backend, pooled, "grow.weight", 2, &context)?;
        assert_eq!(
            values(&stack_video_frames(&backend, &grown, &context)?)?,
            [21.0, 210.0, 43.0, 430.0]
        );
        Ok(())
    }

    #[test]
    fn val_vae_001_taehv_memory_uses_zero_then_previous_input_frame()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let modules = vec![
            convolution_module(
                &backend,
                "memory.conv.0.weight",
                1,
                2,
                &[0.0, 1.0],
                &context,
            )?,
            convolution_module(&backend, "memory.conv.2.weight", 1, 1, &[1.0], &context)?,
            convolution_module(&backend, "memory.conv.4.weight", 1, 1, &[1.0], &context)?,
        ];
        let module = NativeModule::module_dict("test", modules)?;
        let frames = [1.0_f32, 2.0, 4.0]
            .into_iter()
            .map(|value| upload(&backend, &[1, 1, 1, 1], &[value], &context))
            .collect::<Result<Vec<_>, _>>()?;
        let output = apply_memory_block(&module, &backend, frames, "memory", false, &context)?;
        assert_eq!(
            values(&stack_video_frames(&backend, &output, &context)?)?,
            [1.0, 3.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn val_vae_001_hunyuan_space_depth_round_trip_preserves_source_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let input = upload(
            &backend,
            &[1, 1, 2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &context,
        )?;
        let packed = space_to_depth_3d(&backend, &input, 2, &context)?;
        assert_eq!(packed.descriptor().shape(), &[1, 8, 1, 1, 1]);
        let restored = depth_to_space_3d(&backend, &packed, 2, &context)?;
        assert_eq!(restored.descriptor().shape(), input.descriptor().shape());
        assert_eq!(restored.contiguous_bytes()?, input.contiguous_bytes()?);
        Ok(())
    }

    #[test]
    fn val_vae_001_hunyuan_carried_convolution_padding_is_causal_and_replicated()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let input = upload(&backend, &[1, 1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let padded = causal_replicate_pad_3d(&backend, &input, &context)?;
        assert_eq!(padded.descriptor().shape(), &[1, 1, 4, 3, 4]);
        let first = [1.0, 1.0, 2.0, 2.0].repeat(3);
        let second = [3.0, 3.0, 4.0, 4.0].repeat(3);
        let expected = [first.clone(), first.clone(), first, second].concat();
        assert_eq!(values(&padded)?, expected);
        Ok(())
    }

    #[test]
    fn val_vae_001_hunyuan_rms_normalization_uses_channel_axis_and_gamma()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let gamma = upload(&backend, &[2, 1, 1, 1], &[2.0, 3.0], &context)?;
        let module =
            NativeModule::module_dict("test", vec![NativeModule::buffer("norm.gamma", gamma)?])?;
        let input = upload(&backend, &[1, 2, 1, 1, 1], &[3.0, 4.0], &context)?;
        let output = values(&rms_channel_norm(
            &module, &backend, &input, "norm", &context,
        )?)?;
        assert!((output[0] - 1.697_056_3).abs() < 1.0e-5);
        assert!((output[1] - 3.394_112_6).abs() < 1.0e-5);
        Ok(())
    }

    #[test]
    fn val_vae_001_causal3d_stride_and_first_frame_upsample_match_source()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            video_3d_convolution_stride(
                &VaeKernelProfile::Causal3dV1,
                "encoder.down.0.downsample.conv.conv.weight"
            ),
            [1, 2, 2]
        );
        assert_eq!(
            video_3d_convolution_stride(
                &VaeKernelProfile::Causal3dV1,
                "encoder.down.1.downsample.conv.conv.weight"
            ),
            [2, 2, 2]
        );

        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let input = upload(&backend, &[1, 1, 3, 1, 1], &[1.0, 2.0, 3.0], &context)?;
        let output = causal3d_nearest_upsample(&backend, &input, true, &context)?;
        assert_eq!(output.descriptor().shape(), &[1, 1, 5, 2, 2]);
        assert_eq!(
            values(&output)?,
            [vec![1.0; 4], vec![2.0; 8], vec![3.0; 8]].concat()
        );
        Ok(())
    }

    #[test]
    fn val_vae_001_cogvideox_first_prefix_and_temporal_pool_match_source()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let input = upload(&backend, &[1, 1, 2, 1, 1], &[1.0, 2.0], &context)?;
        assert_eq!(
            values(&first_frame_temporal_pad(&backend, &input, 2, &context)?)?,
            [1.0, 1.0, 1.0, 2.0]
        );

        let odd = upload(
            &backend,
            &[1, 1, 5, 1, 1],
            &[1.0, 3.0, 5.0, 7.0, 9.0],
            &context,
        )?;
        assert_eq!(
            values(&cog_temporal_average_pool(&backend, &odd, &context)?)?,
            [1.0, 4.0, 8.0]
        );
        let even = upload(&backend, &[1, 1, 4, 1, 1], &[1.0, 3.0, 5.0, 7.0], &context)?;
        assert_eq!(
            values(&cog_temporal_average_pool(&backend, &even, &context)?)?,
            [2.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn val_vae_001_wan21_causal_prefix_is_zero_not_replicated()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let input = upload(&backend, &[1, 1, 2, 1, 1], &[3.0, 4.0], &context)?;
        let padded = zero_temporal_pad(&backend, &input, 2, &context)?;
        assert_eq!(padded.descriptor().shape(), &[1, 1, 4, 1, 1]);
        assert_eq!(values(&padded)?, [0.0, 0.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn val_vae_001_wan22_patch_shortcuts_preserve_source_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(8 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(8 << 20)?,
            &cancellation,
        );
        let image = upload(&backend, &[1, 1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let patched = wan22_patchify(&backend, &image, &context)?;
        assert_eq!(patched.descriptor().shape(), &[1, 4, 1, 1, 1]);
        assert_eq!(values(&patched)?, [1.0, 3.0, 2.0, 4.0]);
        let restored = wan22_unpatchify(&backend, &patched, &context)?;
        assert_eq!(restored.contiguous_bytes()?, image.contiguous_bytes()?);

        let down_input = upload(
            &backend,
            &[1, 1, 3, 2, 2],
            &[2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0],
            &context,
        )?;
        let down = wan22_avg_shortcut(&backend, &down_input, 1, 2, 2, &context)?;
        assert_eq!(down.descriptor().shape(), &[1, 1, 2, 1, 1]);
        assert_eq!(values(&down)?, [1.0, 5.0]);

        let up_input = upload(&backend, &[1, 1, 2, 1, 1], &[2.0, 4.0], &context)?;
        let up = wan22_dup_up(&backend, &up_input, 1, 2, 2, true, &context)?;
        assert_eq!(up.descriptor().shape(), &[1, 1, 3, 2, 2]);
        assert_eq!(values(&up)?, [vec![2.0; 4], vec![4.0; 8]].concat());
        Ok(())
    }

    #[test]
    fn val_vae_001_cosmos_haar_and_pooling_match_source_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(8 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(8 << 20)?,
            &cancellation,
        );
        let cube = upload(
            &backend,
            &[1, 1, 2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &context,
        )?;
        let transformed = cosmos_haar_downsample(&backend, &cube, &context)?;
        assert_eq!(transformed.descriptor().shape(), &[1, 8, 1, 1, 1]);
        assert_eq!(
            values(&transformed)?,
            [4.5, -0.5, -1.0, 0.0, -2.0, 0.0, 0.0, 0.0]
        );
        let restored = cosmos_haar_upsample(&backend, &transformed, &context)?;
        assert_eq!(restored.contiguous_bytes()?, cube.contiguous_bytes()?);

        let image = upload(
            &backend,
            &[1, 1, 1, 4, 4],
            &(1..=16).map(|value| value as f32).collect::<Vec<_>>(),
            &context,
        )?;
        let patched = cosmos_haar_patchify(&backend, &image, &context)?;
        assert_eq!(patched.descriptor().shape(), &[1, 64, 1, 1, 1]);
        let unpatched = cosmos_haar_unpatchify(&backend, &patched, &context)?;
        assert_eq!(unpatched.contiguous_bytes()?, image.contiguous_bytes()?);

        let temporal = cosmos_average_pool(&backend, &cube, true, &context)?;
        assert_eq!(values(&temporal)?, [3.0, 4.0, 5.0, 6.0]);
        let spatial = cosmos_average_pool(&backend, &cube, false, &context)?;
        assert_eq!(values(&spatial)?, [2.5, 6.5]);
        Ok(())
    }

    #[test]
    fn val_vae_001_mochi_fourier_attention_and_expansion_match_source()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(8 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(8 << 20)?,
            &cancellation,
        );
        let zero_image = upload(&backend, &[1, 3, 1, 1, 1], &[0.0; 3], &context)?;
        let fourier = mochi_fourier_features(&backend, &zero_image, &context)?;
        assert_eq!(fourier.descriptor().shape(), &[1, 15, 1, 1, 1]);
        assert_eq!(values(&fourier)?, [vec![0.0; 9], vec![1.0; 6]].concat());

        let packed = upload(
            &backend,
            &[1, 8, 1, 1, 1],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &context,
        )?;
        let expanded = mochi_depth_to_space_time(&backend, &packed, 2, &context)?;
        assert_eq!(expanded.descriptor().shape(), &[1, 1, 1, 2, 2]);
        assert_eq!(values(&expanded)?, [5.0, 6.0, 7.0, 8.0]);

        let query_key = upload(&backend, &[1, 32, 2, 1, 1], &[1.0; 64], &context)?;
        let value = upload(
            &backend,
            &[1, 32, 2, 1, 1],
            &[2.0_f32, 4.0].repeat(32),
            &context,
        )?;
        let attended =
            mochi_temporal_attention(&backend, &query_key, &query_key, &value, &context)?;
        assert_eq!(values(&attended)?, [3.0_f64, 3.0].repeat(32));
        Ok(())
    }

    #[test]
    fn val_vae_001_ltx_patch_and_residual_pack_order_match_source()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(8 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(8 << 20)?,
            &cancellation,
        );
        let image = upload(
            &backend,
            &[1, 1, 1, 4, 4],
            &(1..=16).map(|value| value as f32).collect::<Vec<_>>(),
            &context,
        )?;
        let patched = ltx_patchify(&backend, &image, 4, &context)?;
        assert_eq!(patched.descriptor().shape(), &[1, 16, 1, 1, 1]);
        assert_eq!(
            values(&patched)?,
            [
                1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0,
                16.0,
            ]
        );
        assert_eq!(
            ltx_unpatchify(&backend, &patched, 4, &context)?.contiguous_bytes()?,
            image.contiguous_bytes()?
        );

        let two_channels = upload(
            &backend,
            &[1, 2, 2, 2, 2],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                18.0,
            ],
            &context,
        )?;
        let packed = ltx_space_to_depth(&backend, &two_channels, 2, 2, &context)?;
        assert_eq!(packed.descriptor().shape(), &[1, 16, 1, 1, 1]);
        assert_eq!(
            values(&packed)?,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                18.0,
            ]
        );
        assert_eq!(
            ltx_depth_to_space(&backend, &packed, 2, 2, &context)?.contiguous_bytes()?,
            two_channels.contiguous_bytes()?
        );
        let channel_pair = upload(&backend, &[1, 2, 1, 1, 1], &[1.0, 2.0], &context)?;
        assert_eq!(
            values(&ltx_repeat_channels(&backend, &channel_pair, 3, &context)?)?,
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
        Ok(())
    }

    #[test]
    fn val_vae_001_ltx_noncausal_padding_reflects_space_and_replicates_time()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(8 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(8 << 20)?,
            &cancellation,
        );
        let input = upload(
            &backend,
            &[1, 1, 2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &context,
        )?;
        let padded = ltx_pad_3d(&backend, &input, 2, 1, 1, false, true, &context)?;
        assert_eq!(padded.descriptor().shape(), &[1, 1, 4, 4, 4]);
        let first = [
            4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0,
        ];
        let second = [
            8.0, 7.0, 8.0, 7.0, 6.0, 5.0, 6.0, 5.0, 8.0, 7.0, 8.0, 7.0, 6.0, 5.0, 6.0, 5.0,
        ];
        assert_eq!(
            values(&padded)?,
            [
                first.as_slice(),
                first.as_slice(),
                second.as_slice(),
                second.as_slice()
            ]
            .concat()
        );
        Ok(())
    }

    #[test]
    fn val_vae_001_ltx_rng_is_caller_addressed_deterministic_and_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let base = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        assert!(begin_vae_rng(&base).is_err());
        let address = RngStreamAddress::new(
            "workflow",
            "attempt",
            "vae-decode",
            0,
            "ltx-video-decode:seed-42",
            0,
            0,
            RetryRngPolicy::Replay,
        )?;
        let context = ExecutionContext {
            stream: base.stream,
            scratch: base.scratch.clone(),
            rng_phase: Some(&address),
            cancellation: base.cancellation,
        };
        let mut first = begin_vae_rng(&context)?;
        let mut second = begin_vae_rng(&context)?;
        let first_values = (0..8)
            .map(|_| first.next_u32(&cancellation))
            .collect::<Result<Vec<_>, _>>()?;
        let second_values = (0..8)
            .map(|_| second.next_u32(&cancellation))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(first_values, second_values);
        assert!(first_values.windows(2).any(|pair| pair[0] != pair[1]));
        Ok(())
    }

    #[test]
    fn val_vae_001_ltx_metadata_configuration_drives_source_topology_and_fails_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let configuration_json = serde_json::to_string(&serde_json::json!({
            "dims": 3,
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "encoder_base_channels": 64,
            "decoder_base_channels": 80,
            "patch_size": 4,
            "norm_layer": "pixel_norm",
            "latent_log_var": "uniform",
            "encoder_blocks": [
                ["res_x", 2],
                ["compress_all", {}],
                ["res_x_y", {"multiplier": 2}],
                ["compress_all", {}],
                ["res_x_y", {"multiplier": 2}],
                ["compress_all", {}]
            ],
            "decoder_blocks": [
                ["res_x", {"num_layers": 2, "inject_noise": true}],
                ["compress_all", {"multiplier": 2, "residual": true}],
                ["res_x", 2],
                ["compress_all", {"multiplier": 2, "residual": true}],
                ["res_x", 2],
                ["compress_all", {"multiplier": 2, "residual": true}]
            ],
            "causal_decoder": true,
            "timestep_conditioning": true,
            "decode_noise_scale": 0.125,
            "decode_timestep": 0.25,
            "spatial_padding_mode": "zeros"
        }))?;
        let loader = VaeLoaderConfiguration::LtxVideo {
            configuration_sha256: Some("0".repeat(64)),
            configuration_json: Some(configuration_json.clone()),
        };
        let profile = VaeKernelProfile::LtxVideoV1 {
            configuration_sha256: Some("0".repeat(64)),
        };
        let configuration = ltx_configuration(&profile, &loader)?;
        assert_eq!(configuration.encoder_base_channels, 64);
        assert_eq!(configuration.decoder_base_channels, 80);
        assert_eq!(configuration.decode_noise_scale, 0.125);
        assert_eq!(configuration.decode_timestep, 0.25);
        assert!(configuration.causal_decoder);
        assert_eq!(
            configuration.decoder_spatial_padding,
            LtxSpatialPadding::Zeros
        );
        let schema = ltx_state_schema(&profile, &configuration, DType::F32)?;
        let shape = |name: &str| {
            schema
                .iter()
                .find(|state| state.name == name)
                .map(|state| state.shape.clone())
        };
        assert_eq!(
            shape("encoder.conv_in.conv.weight"),
            Some(vec![64, 48, 3, 3, 3])
        );
        assert_eq!(
            shape("encoder.conv_out.conv.weight"),
            Some(vec![129, 256, 3, 3, 3])
        );
        assert_eq!(
            shape("decoder.conv_in.conv.weight"),
            Some(vec![640, 128, 3, 3, 3])
        );
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let module = NativeModule::module_dict(
            "video-vae:LtxVideoV1",
            ltx_configuration_modules(&configuration, &backend, DeviceId::CPU, &context)?,
        )?;
        let restored = ltx_configuration_from_module(&module, &backend, &context)?;
        assert_eq!(
            restored.encoder_blocks.len(),
            configuration.encoder_blocks.len()
        );
        assert_eq!(
            restored.decoder_blocks.len(),
            configuration.decoder_blocks.len()
        );
        assert_eq!(
            restored.decode_noise_scale,
            configuration.decode_noise_scale
        );
        assert_eq!(restored.decode_timestep, configuration.decode_timestep);
        assert_eq!(restored.causal_decoder, configuration.causal_decoder);

        let mut group_value = serde_json::from_str::<serde_json::Value>(&configuration_json)?;
        group_value["norm_layer"] = serde_json::json!("group_norm");
        group_value["latent_log_var"] = serde_json::json!("per_channel");
        group_value["decoder_base_channels"] = serde_json::json!(96);
        let group_loader = VaeLoaderConfiguration::LtxVideo {
            configuration_sha256: Some("2".repeat(64)),
            configuration_json: Some(serde_json::to_string(&group_value)?),
        };
        let group_configuration = ltx_configuration(&profile, &group_loader)?;
        let group_schema = ltx_state_schema(&profile, &group_configuration, DType::F32)?;
        assert!(
            group_schema
                .iter()
                .any(|state| state.name == "encoder.down_blocks.0.res_blocks.0.norm1.weight")
        );
        assert_eq!(
            group_schema
                .iter()
                .find(|state| state.name == "encoder.conv_out.conv.weight")
                .map(|state| state.shape.clone()),
            Some(vec![256, 256, 3, 3, 3])
        );

        let unsupported_json = serde_json::to_string(&serde_json::json!({
            "dims": 3,
            "latent_channels": 128,
            "blocks": [["compress_all", 1], ["compress_all", 1], ["compress_all", 1]],
            "patch_size": 4,
            "norm_layer": "batch_norm",
            "latent_log_var": "uniform"
        }))?;
        let unsupported = VaeLoaderConfiguration::LtxVideo {
            configuration_sha256: Some("1".repeat(64)),
            configuration_json: Some(unsupported_json),
        };
        assert!(matches!(
            ltx_configuration(&profile, &unsupported),
            Err(VideoVaeError::InvalidLtxConfiguration(detail))
                if detail.contains("norm_layer")
        ));
        Ok(())
    }

    #[test]
    fn val_vae_001_mochi_partial_namespaces_project_to_one_canonical_schema() {
        let raw_decoder = mochi_schema_projection(DType::F16, true, false, false, false);
        assert!(raw_decoder.iter().all(|(source, target)| {
            !source.starts_with("decoder.") && target.name.starts_with("decoder.")
        }));
        assert!(raw_decoder.iter().any(|(source, target)| {
            source == "blocks.2.blocks.3.stack.5.weight"
                && target.name == "decoder.blocks.2.blocks.3.stack.5.weight"
        }));
        assert!(
            raw_decoder
                .iter()
                .all(|(_, target)| !target.name.starts_with("encoder."))
        );

        let full = mochi_schema_projection(DType::F16, false, false, true, true);
        assert!(full.iter().all(|(source, target)| source == &target.name));
        assert!(
            full.iter()
                .any(|(_, target)| target.name.starts_with("encoder."))
        );
        assert!(
            full.iter()
                .any(|(_, target)| target.name.starts_with("decoder."))
        );
    }
}
