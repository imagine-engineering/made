use crate::{
    ArtifactIndex, LatentFormatDefinition, LoadedModel, ModelStore, NativeModule, NativeOpsError,
    NativeVisionModelError, NativeVisionStateKind, NativeVisionStateSpec, PeriodicActivation,
    TensorMetadata, VaeDescriptor, VaeError, VaeKernelProfile, VaeLoaderConfiguration,
    VaeOperation, alias_free_activation_1d_exact_native,
    vae::{NativeVae, VaeKernelFunctions, VaeModelBinding, prepare_pixel_channels},
    vae_image::{
        add_tensor, affine_tensor, convolution as execute_convolution, find_module,
        nearest_upsample_2x, pixel_shuffle, pixel_unshuffle, reshape_read_only, silu_tensor,
        softmax_last_dimension, spatial_attention_from_qkv,
    },
    vae_video::{
        begin_vae_rng, contiguous_copy, grouped_channel_mean, narrow_contiguous, permute_read_only,
        repeat_channels_interleave,
    },
    vision_models::{
        canonical_vision_model_store_dtype, load_vision_state_from_model_store_with_context,
    },
};
use sim_tensor::generated_elementwise_or_runtime_operation_12::stft_with_context_exact_native;
use sim_tensor::generated_elementwise_or_runtime_operation_14::view_as_real_exact_native;
use sim_tensor::generated_external_tensor_kernel_01::{
    NativeMelNormalization, NativeMelScale, NativeMelSpectrogramConfiguration,
    NativeResampleConfiguration, mel_spectrogram_with_context_exact_native,
    resample_with_context_exact_native,
};
use sim_tensor::generated_random_number_generation_01::randn_like_with_context_exact_native;
use sim_tensor::generated_sim_operator_indirection_01::{
    ConvolutionGeometry, ConvolutionPaddingMode, tensor_from_f32_with_backend_exact_native,
    tensor_to_f32_with_backend_exact_native,
};
use sim_tensor::{
    BinaryOperation, CpuBackend, DType, DeviceId, ExecutionContext, LinearAlgebraOperation,
    ReductionOperation, ReductionSpec, Scalar, ScalarSide, Tensor, TensorBackend, TensorDescriptor,
    UnaryOperation,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use thiserror::Error;

const OOBLECK_ARCHITECTURE: &str = "sim.ldm.audio.vae.AudioOobleckVAE.v1";
const MUSIC_DCAE_ARCHITECTURE: &str = "sim.ldm.ace.vae.MusicDCAE.v1";
const MMAUDIO_ARCHITECTURE: &str = "sim.ldm.mmaudio.vae.AudioAutoencoder.v1";
const LTX_AUDIO_ARCHITECTURE: &str = "sim.ldm.lightricks.vae.audio_vae.AudioVAE.v1";
const STABLE_AUDIO_3_ARCHITECTURE: &str = "sim.ldm.audio.vae_sa3.SA3AudioVAE.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioVaeSourceCheckpoint {
    pub name: &'static str,
    pub rank: u8,
    pub dimensions: &'static [(usize, u64)],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAudioVaeArchitecture {
    profile: VaeKernelProfile,
    architecture: &'static str,
    input_sample_rate: u32,
    output_sample_rate: u32,
    sample_ratio_numerator: u64,
    sample_ratio_denominator: u64,
    latent_channels: u64,
    latent_dimensions: u8,
    latent_frequency_bins: Option<u64>,
    checkpoints: &'static [AudioVaeSourceCheckpoint],
    equations: &'static [&'static str],
    storage_dtype: Option<DType>,
    state_schema: Vec<NativeVisionStateSpec>,
    source_names: BTreeMap<String, String>,
}

impl NativeAudioVaeArchitecture {
    pub fn profile(&self) -> &VaeKernelProfile {
        &self.profile
    }

    pub const fn architecture(&self) -> &'static str {
        self.architecture
    }

    pub const fn input_sample_rate(&self) -> u32 {
        self.input_sample_rate
    }

    pub const fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate
    }

    pub const fn sample_ratio(&self) -> (u64, u64) {
        (self.sample_ratio_numerator, self.sample_ratio_denominator)
    }

    pub const fn latent_channels(&self) -> u64 {
        self.latent_channels
    }

    pub const fn latent_dimensions(&self) -> u8 {
        self.latent_dimensions
    }

    pub const fn latent_frequency_bins(&self) -> Option<u64> {
        self.latent_frequency_bins
    }

    pub const fn state_checkpoints(&self) -> &'static [AudioVaeSourceCheckpoint] {
        self.checkpoints
    }

    pub const fn equation_checkpoints(&self) -> &'static [&'static str] {
        self.equations
    }

    pub const fn storage_dtype(&self) -> Option<DType> {
        self.storage_dtype
    }

    pub fn state_schema(&self) -> &[NativeVisionStateSpec] {
        &self.state_schema
    }
}

#[derive(Debug, Error)]
pub enum AudioVaeError {
    #[error(transparent)]
    Cancelled(#[from] sim_types::CancellationError),
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    NativeModule(#[from] NativeOpsError),
    #[error(transparent)]
    VisionState(#[from] NativeVisionModelError),
    #[error("audio VAE profile {0:?} is not implemented by the audio architecture adapter")]
    UnsupportedProfile(VaeKernelProfile),
    #[error("audio VAE architecture {expected} does not match descriptor architecture {actual}")]
    ArchitectureMismatch {
        expected: &'static str,
        actual: String,
    },
    #[error("audio VAE state is missing source checkpoint {0}")]
    MissingState(String),
    #[error("audio VAE source topology contains duplicate checkpoint {0}")]
    DuplicateState(String),
    #[error("audio VAE state checkpoint {name} has invalid shape {shape:?}")]
    InvalidStateShape { name: String, shape: Vec<u64> },
    #[error("audio VAE state checkpoint {name} uses unsupported storage dtype {dtype}")]
    UnsupportedStorageDType { name: String, dtype: String },
    #[error("audio VAE loader produced unexpected state checkpoint {0}")]
    UnexpectedState(String),
    #[error(
        "audio VAE state mixes floating storage dtypes: expected {expected:?}, got {actual:?} at {name}"
    )]
    MixedStorageDType {
        name: String,
        expected: DType,
        actual: DType,
    },
    #[error("audio VAE input rank must be {expected}, got {actual}")]
    InputRank { expected: usize, actual: usize },
    #[error("audio VAE input must contain at least one channel and one sample")]
    EmptyInput,
    #[error("invalid LTX audio loader configuration: {0}")]
    InvalidLtxConfiguration(String),
    #[error("audio VAE shape arithmetic overflowed")]
    ShapeOverflow,
}

const OOBLECK_STATE: &[AudioVaeSourceCheckpoint] = &[
    AudioVaeSourceCheckpoint {
        name: "decoder.layers.1.layers.0.beta",
        rank: 1,
        dimensions: &[],
    },
    AudioVaeSourceCheckpoint {
        name: "encoder.layers.0.parametrizations.weight.original1",
        rank: 3,
        dimensions: &[(0, 128), (1, 2), (2, 7)],
    },
    AudioVaeSourceCheckpoint {
        name: "decoder.layers.7.parametrizations.weight.original1",
        rank: 3,
        dimensions: &[(0, 2), (1, 128), (2, 7)],
    },
];
const MUSIC_DCAE_STATE: &[AudioVaeSourceCheckpoint] = &[
    AudioVaeSourceCheckpoint {
        name: "vocoder.backbone.channel_layers.0.0.bias",
        rank: 1,
        dimensions: &[],
    },
    AudioVaeSourceCheckpoint {
        name: "dcae.encoder.conv_in.weight",
        rank: 4,
        dimensions: &[(1, 2)],
    },
    AudioVaeSourceCheckpoint {
        name: "dcae.decoder.conv_out.weight",
        rank: 4,
        dimensions: &[(0, 2)],
    },
];
const MMAUDIO_STATE: &[AudioVaeSourceCheckpoint] = &[
    AudioVaeSourceCheckpoint {
        name: "vocoder.activation_post.downsample.lowpass.filter",
        rank: 3,
        dimensions: &[],
    },
    AudioVaeSourceCheckpoint {
        name: "vae.encoder.conv_in.weight",
        rank: 3,
        dimensions: &[(1, 80)],
    },
    AudioVaeSourceCheckpoint {
        name: "vae.decoder.conv_out.weight",
        rank: 3,
        dimensions: &[(0, 80)],
    },
];
const LTX_AUDIO_STATE: &[AudioVaeSourceCheckpoint] = &[
    AudioVaeSourceCheckpoint {
        name: "vocoder.resblocks.0.convs1.0.weight",
        rank: 3,
        dimensions: &[],
    },
    AudioVaeSourceCheckpoint {
        name: "autoencoder.encoder.conv_in.weight",
        rank: 4,
        dimensions: &[(1, 2)],
    },
    AudioVaeSourceCheckpoint {
        name: "autoencoder.decoder.conv_out.weight",
        rank: 4,
        dimensions: &[(0, 2)],
    },
];
const STABLE_AUDIO_3_STATE: &[AudioVaeSourceCheckpoint] = &[
    AudioVaeSourceCheckpoint {
        name: "decoder.layers.3.transformers.0.pre_norm.alpha",
        rank: 1,
        dimensions: &[],
    },
    AudioVaeSourceCheckpoint {
        name: "encoder.layers.0.weight",
        rank: 3,
        dimensions: &[(1, 512)],
    },
    AudioVaeSourceCheckpoint {
        name: "decoder.layers.6.weight",
        rank: 3,
        dimensions: &[(0, 512)],
    },
];

const OOBLECK_EQUATIONS: &[&str] = &[
    "weight_normalized_conv1d_residual_stack",
    "snake_beta_alias_free_activation",
    "diagonal_gaussian_reparameterized_sample",
    "replicate_channel_padding",
];
const MUSIC_DCAE_EQUATIONS: &[&str] = &[
    "waveform_resample_to_44100",
    "checkpoint_hann_window_and_mel_filter_bank",
    "log_mel_normalize_minus_11_to_3",
    "dcae_latent_affine_shift_minus_1_9091_scale_0_1786",
    "extra_1d_channel_reshape_16",
    "audio_chunk_multiple_4096",
];
const MMAUDIO_EQUATIONS: &[&str] = &[
    "stereo_mean_then_resample_44100_to_16000",
    "checkpoint_hann_window_and_mel_filter_bank",
    "stft_1024_hop_256_mel_80",
    "per_band_mean_std_normalization",
    "diagonal_gaussian_mode",
    "vocoder_then_resample_16000_to_44100",
];
const LTX_AUDIO_EQUATIONS: &[&str] = &[
    "source_rate_to_configured_sample_rate",
    "stft_configured_hop_and_mel_bins",
    "causal_latent_length_ceil",
    "per_channel_latent_statistics",
    "extra_1d_channel_reshape_16",
    "configured_vocoder_output_rate",
];
const STABLE_AUDIO_3_EQUATIONS: &[&str] = &[
    "zero_pad_to_patch_256",
    "patch_channels_2_times_256",
    "variable_stride_16",
    "softnorm_bottleneck",
    "bounded_transformer_chunks",
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct AudioStateShape {
    name: String,
    shape: Vec<u64>,
}

fn push_weight_normalized_convolution(
    state: &mut Vec<AudioStateShape>,
    prefix: &str,
    weight_shape: Vec<u64>,
    bias_channels: Option<u64>,
) {
    let mut magnitude_shape = vec![1; weight_shape.len()];
    if let Some(output_axis) = weight_shape.first() {
        magnitude_shape[0] = *output_axis;
    }
    state.push(AudioStateShape {
        name: format!("{prefix}.parametrizations.weight.original0"),
        shape: magnitude_shape,
    });
    state.push(AudioStateShape {
        name: format!("{prefix}.parametrizations.weight.original1"),
        shape: weight_shape,
    });
    if let Some(channels) = bias_channels {
        state.push(AudioStateShape {
            name: format!("{prefix}.bias"),
            shape: vec![channels],
        });
    }
}

fn push_snake_beta(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    for parameter in ["alpha", "beta"] {
        state.push(AudioStateShape {
            name: format!("{prefix}.{parameter}"),
            shape: vec![channels],
        });
    }
}

fn push_oobleck_residual_unit(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    push_snake_beta(state, &format!("{prefix}.layers.0"), channels);
    push_weight_normalized_convolution(
        state,
        &format!("{prefix}.layers.1"),
        vec![channels, channels, 7],
        Some(channels),
    );
    push_snake_beta(state, &format!("{prefix}.layers.2"), channels);
    push_weight_normalized_convolution(
        state,
        &format!("{prefix}.layers.3"),
        vec![channels, channels, 1],
        Some(channels),
    );
}

fn oobleck_source_state_shapes(profile: &VaeKernelProfile) -> Option<Vec<AudioStateShape>> {
    let strides = match profile {
        VaeKernelProfile::AudioOobleck44KhzV1 => [2_u64, 4, 4, 8, 8],
        VaeKernelProfile::AudioOobleck48KhzV1 => [2_u64, 4, 4, 6, 10],
        _ => return None,
    };
    let multipliers = [1_u64, 1, 2, 4, 8, 16];
    let base_channels = 128_u64;
    let mut state = Vec::new();

    push_weight_normalized_convolution(
        &mut state,
        "encoder.layers.0",
        vec![base_channels, 2, 7],
        Some(base_channels),
    );
    for block in 0..5 {
        let input_channels = multipliers[block] * base_channels;
        let output_channels = multipliers[block + 1] * base_channels;
        let prefix = format!("encoder.layers.{}", block + 1);
        for residual in 0..3 {
            push_oobleck_residual_unit(
                &mut state,
                &format!("{prefix}.layers.{residual}"),
                input_channels,
            );
        }
        push_snake_beta(&mut state, &format!("{prefix}.layers.3"), input_channels);
        push_weight_normalized_convolution(
            &mut state,
            &format!("{prefix}.layers.4"),
            vec![output_channels, input_channels, 2 * strides[block]],
            Some(output_channels),
        );
    }
    push_snake_beta(&mut state, "encoder.layers.6", 16 * base_channels);
    push_weight_normalized_convolution(
        &mut state,
        "encoder.layers.7",
        vec![128, 16 * base_channels, 3],
        Some(128),
    );

    push_weight_normalized_convolution(
        &mut state,
        "decoder.layers.0",
        vec![16 * base_channels, 64, 7],
        Some(16 * base_channels),
    );
    for block in 0..5 {
        let multiplier_index = 5 - block;
        let input_channels = multipliers[multiplier_index] * base_channels;
        let output_channels = multipliers[multiplier_index - 1] * base_channels;
        let stride = strides[multiplier_index - 1];
        let prefix = format!("decoder.layers.{}", block + 1);
        push_snake_beta(&mut state, &format!("{prefix}.layers.0"), input_channels);
        push_weight_normalized_convolution(
            &mut state,
            &format!("{prefix}.layers.1"),
            vec![input_channels, output_channels, 2 * stride],
            Some(output_channels),
        );
        for residual in 0..3 {
            push_oobleck_residual_unit(
                &mut state,
                &format!("{prefix}.layers.{}", residual + 2),
                output_channels,
            );
        }
    }
    push_snake_beta(&mut state, "decoder.layers.6", base_channels);
    push_weight_normalized_convolution(
        &mut state,
        "decoder.layers.7",
        vec![2, base_channels, 7],
        None,
    );
    Some(state)
}

fn push_dense_convolution(
    state: &mut Vec<AudioStateShape>,
    prefix: &str,
    output_channels: u64,
    input_channels: u64,
    kernel: u64,
    bias: bool,
    transposed: bool,
) {
    state.push(AudioStateShape {
        name: format!("{prefix}.weight"),
        shape: if transposed {
            vec![input_channels, output_channels, kernel]
        } else {
            vec![output_channels, input_channels, kernel]
        },
    });
    if bias {
        state.push(AudioStateShape {
            name: format!("{prefix}.bias"),
            shape: vec![output_channels],
        });
    }
}

fn push_mmaudio_residual_state(
    state: &mut Vec<AudioStateShape>,
    prefix: &str,
    input_channels: u64,
    output_channels: u64,
) {
    push_dense_convolution(
        state,
        &format!("{prefix}.conv1"),
        output_channels,
        input_channels,
        3,
        false,
        false,
    );
    push_dense_convolution(
        state,
        &format!("{prefix}.conv2"),
        output_channels,
        output_channels,
        3,
        false,
        false,
    );
    if input_channels != output_channels {
        push_dense_convolution(
            state,
            &format!("{prefix}.nin_shortcut"),
            output_channels,
            input_channels,
            1,
            false,
            false,
        );
    }
}

fn push_mmaudio_attention_state(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    push_dense_convolution(
        state,
        &format!("{prefix}.qkv"),
        channels * 3,
        channels,
        1,
        false,
        false,
    );
    push_dense_convolution(
        state,
        &format!("{prefix}.proj_out"),
        channels,
        channels,
        1,
        false,
        false,
    );
}

fn push_mmaudio_alias_activation_state(
    state: &mut Vec<AudioStateShape>,
    prefix: &str,
    channels: u64,
) {
    for (suffix, shape) in [
        ("act.alpha", vec![channels]),
        ("act.beta", vec![channels]),
        ("upsample.filter", vec![1, 1, 12]),
        ("downsample.lowpass.filter", vec![1, 1, 12]),
    ] {
        state.push(AudioStateShape {
            name: format!("{prefix}.{suffix}"),
            shape,
        });
    }
}

fn mmaudio_source_state_shapes() -> Vec<AudioStateShape> {
    let mut state = vec![
        AudioStateShape {
            name: "mel_converter.mel_basis".to_owned(),
            shape: vec![80, 513],
        },
        AudioStateShape {
            name: "mel_converter.hann_window".to_owned(),
            shape: vec![1_024],
        },
        AudioStateShape {
            name: "vae.data_mean".to_owned(),
            shape: vec![1, 80, 1],
        },
        AudioStateShape {
            name: "vae.data_std".to_owned(),
            shape: vec![1, 80, 1],
        },
    ];
    push_dense_convolution(&mut state, "vae.encoder.conv_in", 384, 80, 3, false, false);
    let mut channels = 384_u64;
    for (level, output_channels) in [384_u64, 768, 1_536].into_iter().enumerate() {
        for block in 0..2 {
            push_mmaudio_residual_state(
                &mut state,
                &format!("vae.encoder.down.{level}.block.{block}"),
                channels,
                output_channels,
            );
            channels = output_channels;
        }
        if level == 0 {
            push_dense_convolution(
                &mut state,
                "vae.encoder.down.0.downsample.conv1",
                channels,
                channels,
                1,
                false,
                false,
            );
            push_dense_convolution(
                &mut state,
                "vae.encoder.down.0.downsample.conv2",
                channels,
                channels,
                1,
                false,
                false,
            );
        }
    }
    push_mmaudio_residual_state(&mut state, "vae.encoder.mid.block_1", channels, channels);
    push_mmaudio_attention_state(&mut state, "vae.encoder.mid.attn_1", channels);
    push_mmaudio_residual_state(&mut state, "vae.encoder.mid.block_2", channels, channels);
    push_dense_convolution(
        &mut state,
        "vae.encoder.conv_out",
        40,
        channels,
        3,
        false,
        false,
    );
    state.push(AudioStateShape {
        name: "vae.encoder.learnable_gain".to_owned(),
        shape: Vec::new(),
    });

    push_dense_convolution(
        &mut state,
        "vae.decoder.conv_in",
        1_536,
        20,
        3,
        false,
        false,
    );
    channels = 1_536;
    push_mmaudio_residual_state(&mut state, "vae.decoder.mid.block_1", channels, channels);
    push_mmaudio_attention_state(&mut state, "vae.decoder.mid.attn_1", channels);
    push_mmaudio_residual_state(&mut state, "vae.decoder.mid.block_2", channels, channels);
    for (level, output_channels) in [(2, 1_536_u64), (1, 768), (0, 384)] {
        for block in 0..3 {
            push_mmaudio_residual_state(
                &mut state,
                &format!("vae.decoder.up.{level}.block.{block}"),
                channels,
                output_channels,
            );
            channels = output_channels;
        }
        if level == 1 {
            push_dense_convolution(
                &mut state,
                "vae.decoder.up.1.upsample.conv",
                channels,
                channels,
                3,
                false,
                false,
            );
        }
    }
    push_dense_convolution(
        &mut state,
        "vae.decoder.conv_out",
        80,
        channels,
        3,
        false,
        false,
    );
    state.push(AudioStateShape {
        name: "vae.decoder.learnable_gain".to_owned(),
        shape: Vec::new(),
    });

    push_dense_convolution(&mut state, "vocoder.conv_pre", 1_536, 80, 7, true, false);
    let upsample_kernels = [8_u64, 8, 4, 4, 4, 4];
    channels = 1_536;
    for stage in 0..6 {
        let output_channels = channels / 2;
        push_dense_convolution(
            &mut state,
            &format!("vocoder.ups.{stage}.0"),
            output_channels,
            channels,
            upsample_kernels[stage],
            true,
            true,
        );
        channels = output_channels;
        for (kernel_index, kernel) in [3_u64, 7, 11].into_iter().enumerate() {
            let block = stage * 3 + kernel_index;
            for layer in 0..3 {
                push_mmaudio_alias_activation_state(
                    &mut state,
                    &format!("vocoder.resblocks.{block}.activations.{}", layer * 2),
                    channels,
                );
                push_dense_convolution(
                    &mut state,
                    &format!("vocoder.resblocks.{block}.convs1.{layer}"),
                    channels,
                    channels,
                    kernel,
                    true,
                    false,
                );
                push_mmaudio_alias_activation_state(
                    &mut state,
                    &format!("vocoder.resblocks.{block}.activations.{}", layer * 2 + 1),
                    channels,
                );
                push_dense_convolution(
                    &mut state,
                    &format!("vocoder.resblocks.{block}.convs2.{layer}"),
                    channels,
                    channels,
                    kernel,
                    true,
                    false,
                );
            }
        }
    }
    push_mmaudio_alias_activation_state(&mut state, "vocoder.activation_post", channels);
    push_dense_convolution(&mut state, "vocoder.conv_post", 1, channels, 7, true, false);
    state
}

fn push_audio_parameter(
    state: &mut Vec<AudioStateShape>,
    name: impl Into<String>,
    shape: Vec<u64>,
) {
    state.push(AudioStateShape {
        name: name.into(),
        shape,
    });
}

fn push_music_convolution_2d(
    state: &mut Vec<AudioStateShape>,
    prefix: &str,
    output_channels: u64,
    input_channels_per_group: u64,
    kernel: u64,
    bias: bool,
) {
    push_audio_parameter(
        state,
        format!("{prefix}.weight"),
        vec![output_channels, input_channels_per_group, kernel, kernel],
    );
    if bias {
        push_audio_parameter(state, format!("{prefix}.bias"), vec![output_channels]);
    }
}

fn push_music_linear(
    state: &mut Vec<AudioStateShape>,
    prefix: &str,
    output_features: u64,
    input_features: u64,
    bias: bool,
) {
    push_audio_parameter(
        state,
        format!("{prefix}.weight"),
        vec![output_features, input_features],
    );
    if bias {
        push_audio_parameter(state, format!("{prefix}.bias"), vec![output_features]);
    }
}

fn push_music_norm(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    for parameter in ["weight", "bias"] {
        push_audio_parameter(state, format!("{prefix}.{parameter}"), vec![channels]);
    }
}

fn push_music_dcae_residual(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    push_music_convolution_2d(
        state,
        &format!("{prefix}.conv1"),
        channels,
        channels,
        3,
        true,
    );
    push_music_convolution_2d(
        state,
        &format!("{prefix}.conv2"),
        channels,
        channels,
        3,
        false,
    );
    push_music_norm(state, &format!("{prefix}.norm"), channels);
}

fn push_music_efficient_vit(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    let heads = channels / 32;
    for projection in ["to_q", "to_k", "to_v"] {
        push_music_linear(
            state,
            &format!("{prefix}.attn.{projection}"),
            channels,
            channels,
            false,
        );
    }
    push_music_convolution_2d(
        state,
        &format!("{prefix}.attn.to_qkv_multiscale.0.proj_in"),
        channels * 3,
        1,
        5,
        false,
    );
    push_music_convolution_2d(
        state,
        &format!("{prefix}.attn.to_qkv_multiscale.0.proj_out"),
        channels * 3,
        channels / heads,
        1,
        false,
    );
    push_music_linear(
        state,
        &format!("{prefix}.attn.to_out"),
        channels,
        channels * 2,
        false,
    );
    push_music_norm(state, &format!("{prefix}.attn.norm_out"), channels);

    let expanded = channels * 4;
    push_music_convolution_2d(
        state,
        &format!("{prefix}.conv_out.conv_inverted"),
        expanded * 2,
        channels,
        1,
        true,
    );
    push_music_convolution_2d(
        state,
        &format!("{prefix}.conv_out.conv_depth"),
        expanded * 2,
        1,
        3,
        true,
    );
    push_music_convolution_2d(
        state,
        &format!("{prefix}.conv_out.conv_point"),
        channels,
        expanded,
        1,
        false,
    );
    push_music_norm(state, &format!("{prefix}.conv_out.norm"), channels);
}

fn push_music_convnext_block(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    push_dense_convolution(
        state,
        &format!("{prefix}.dwconv"),
        channels,
        1,
        7,
        true,
        false,
    );
    push_music_norm(state, &format!("{prefix}.norm"), channels);
    push_music_linear(
        state,
        &format!("{prefix}.pwconv1"),
        channels * 4,
        channels,
        true,
    );
    push_music_linear(
        state,
        &format!("{prefix}.pwconv2"),
        channels,
        channels * 4,
        true,
    );
    push_audio_parameter(state, format!("{prefix}.gamma"), vec![channels]);
}

fn music_dcae_source_state_shapes() -> Vec<AudioStateShape> {
    let mut state = Vec::new();
    let channels = [128_u64, 256, 512, 1_024];
    let encoder_layers = [2_usize, 2, 3, 3];
    let decoder_layers = [3_usize, 3, 3, 3];

    push_music_convolution_2d(&mut state, "dcae.encoder.conv_in", 128, 2, 3, true);
    for level in 0..4 {
        for block in 0..encoder_layers[level] {
            let prefix = format!("dcae.encoder.down_blocks.{level}.{block}");
            if level == 3 {
                push_music_efficient_vit(&mut state, &prefix, channels[level]);
            } else {
                push_music_dcae_residual(&mut state, &prefix, channels[level]);
            }
        }
        if level < 3 {
            push_music_convolution_2d(
                &mut state,
                &format!(
                    "dcae.encoder.down_blocks.{level}.{}.conv",
                    encoder_layers[level]
                ),
                channels[level + 1],
                channels[level],
                3,
                true,
            );
        }
    }
    push_music_convolution_2d(&mut state, "dcae.encoder.conv_out", 8, 1_024, 3, true);

    push_music_convolution_2d(&mut state, "dcae.decoder.conv_in", 1_024, 8, 3, true);
    for level in 0..4 {
        let mut child = 0_usize;
        if level < 3 {
            push_music_convolution_2d(
                &mut state,
                &format!("dcae.decoder.up_blocks.{level}.{child}.conv"),
                channels[level],
                channels[level + 1],
                3,
                true,
            );
            child += 1;
        }
        for block in 0..decoder_layers[level] {
            let prefix = format!("dcae.decoder.up_blocks.{level}.{}", child + block);
            if level == 3 {
                push_music_efficient_vit(&mut state, &prefix, channels[level]);
            } else {
                push_music_dcae_residual(&mut state, &prefix, channels[level]);
            }
        }
    }
    push_music_norm(&mut state, "dcae.decoder.norm_out", 128);
    push_music_convolution_2d(&mut state, "dcae.decoder.conv_out", 2, 128, 3, true);

    for (level, level_channels) in channels.into_iter().enumerate() {
        if level == 0 {
            push_dense_convolution(
                &mut state,
                "vocoder.backbone.channel_layers.0.0",
                level_channels,
                128,
                7,
                true,
                false,
            );
            push_music_norm(
                &mut state,
                "vocoder.backbone.channel_layers.0.1",
                level_channels,
            );
        } else {
            push_music_norm(
                &mut state,
                &format!("vocoder.backbone.channel_layers.{level}.0"),
                channels[level - 1],
            );
            push_dense_convolution(
                &mut state,
                &format!("vocoder.backbone.channel_layers.{level}.1"),
                level_channels,
                channels[level - 1],
                1,
                true,
                false,
            );
        }
        for block in 0..[3_usize, 3, 9, 3][level] {
            push_music_convnext_block(
                &mut state,
                &format!("vocoder.backbone.stages.{level}.{block}"),
                level_channels,
            );
        }
    }
    push_music_norm(&mut state, "vocoder.backbone.norm", 512);

    push_weight_normalized_convolution(
        &mut state,
        "vocoder.head.conv_pre",
        vec![1_024, 512, 13],
        Some(1_024),
    );
    let rates = [4_u64, 4, 2, 2, 2, 2, 2];
    let kernels = [8_u64, 8, 4, 4, 4, 4, 4];
    let mut input_channels = 1_024_u64;
    for stage in 0..7 {
        let output_channels = input_channels / 2;
        push_weight_normalized_convolution(
            &mut state,
            &format!("vocoder.head.ups.{stage}"),
            vec![input_channels, output_channels, kernels[stage]],
            Some(output_channels),
        );
        for (kernel_index, kernel) in [3_u64, 7, 11, 13].into_iter().enumerate() {
            let block = stage * 4 + kernel_index;
            for layer in 0..3 {
                push_weight_normalized_convolution(
                    &mut state,
                    &format!("vocoder.head.resblocks.{block}.convs1.{layer}"),
                    vec![output_channels, output_channels, kernel],
                    Some(output_channels),
                );
                push_weight_normalized_convolution(
                    &mut state,
                    &format!("vocoder.head.resblocks.{block}.convs2.{layer}"),
                    vec![output_channels, output_channels, kernel],
                    Some(output_channels),
                );
            }
        }
        input_channels = output_channels;
    }
    push_weight_normalized_convolution(
        &mut state,
        "vocoder.head.conv_post",
        vec![1, input_channels, 13],
        Some(1),
    );
    push_audio_parameter(
        &mut state,
        "vocoder.mel_transform.spectrogram.window",
        vec![2_048],
    );
    push_audio_parameter(
        &mut state,
        "vocoder.mel_transform.mel_scale.fb",
        vec![1_025, 128],
    );
    debug_assert_eq!(rates.into_iter().product::<u64>(), 512);
    state
}

fn push_ltx_convolution_2d(
    state: &mut Vec<AudioStateShape>,
    prefix: &str,
    output_channels: u64,
    input_channels: u64,
    kernel: u64,
) {
    push_audio_parameter(
        state,
        format!("{prefix}.weight"),
        vec![output_channels, input_channels, kernel, kernel],
    );
    push_audio_parameter(state, format!("{prefix}.bias"), vec![output_channels]);
}

fn push_ltx_residual_state(
    state: &mut Vec<AudioStateShape>,
    prefix: &str,
    input_channels: u64,
    output_channels: u64,
) {
    push_ltx_convolution_2d(
        state,
        &format!("{prefix}.conv1.conv"),
        output_channels,
        input_channels,
        3,
    );
    push_ltx_convolution_2d(
        state,
        &format!("{prefix}.conv2.conv"),
        output_channels,
        output_channels,
        3,
    );
    if input_channels != output_channels {
        push_ltx_convolution_2d(
            state,
            &format!("{prefix}.nin_shortcut.conv"),
            output_channels,
            input_channels,
            1,
        );
    }
}

fn ltx_audio_source_state_shapes() -> Vec<AudioStateShape> {
    let mut state = Vec::new();
    push_ltx_convolution_2d(&mut state, "autoencoder.encoder.conv_in.conv", 128, 2, 3);
    let mut channels = 128_u64;
    for (level, output_channels) in [128_u64, 256, 512].into_iter().enumerate() {
        for block in 0..2 {
            push_ltx_residual_state(
                &mut state,
                &format!("autoencoder.encoder.down.{level}.block.{block}"),
                channels,
                output_channels,
            );
            channels = output_channels;
        }
        if level != 2 {
            push_ltx_convolution_2d(
                &mut state,
                &format!("autoencoder.encoder.down.{level}.downsample.conv"),
                channels,
                channels,
                3,
            );
        }
    }
    push_ltx_residual_state(
        &mut state,
        "autoencoder.encoder.mid.block_1",
        channels,
        channels,
    );
    push_ltx_residual_state(
        &mut state,
        "autoencoder.encoder.mid.block_2",
        channels,
        channels,
    );
    push_ltx_convolution_2d(
        &mut state,
        "autoencoder.encoder.conv_out.conv",
        16,
        channels,
        3,
    );

    push_ltx_convolution_2d(&mut state, "autoencoder.decoder.conv_in.conv", 512, 8, 3);
    channels = 512;
    push_ltx_residual_state(
        &mut state,
        "autoencoder.decoder.mid.block_1",
        channels,
        channels,
    );
    push_ltx_residual_state(
        &mut state,
        "autoencoder.decoder.mid.block_2",
        channels,
        channels,
    );
    for (level, output_channels) in [(2_usize, 512_u64), (1, 256), (0, 128)] {
        for block in 0..3 {
            push_ltx_residual_state(
                &mut state,
                &format!("autoencoder.decoder.up.{level}.block.{block}"),
                channels,
                output_channels,
            );
            channels = output_channels;
        }
        if level != 0 {
            push_ltx_convolution_2d(
                &mut state,
                &format!("autoencoder.decoder.up.{level}.upsample.conv.conv"),
                channels,
                channels,
                3,
            );
        }
    }
    push_ltx_convolution_2d(
        &mut state,
        "autoencoder.decoder.conv_out.conv",
        2,
        channels,
        3,
    );
    push_audio_parameter(
        &mut state,
        "autoencoder.per_channel_statistics.std-of-means",
        vec![128],
    );
    push_audio_parameter(
        &mut state,
        "autoencoder.per_channel_statistics.mean-of-means",
        vec![128],
    );

    push_dense_convolution(&mut state, "vocoder.conv_pre", 1_024, 128, 7, true, false);
    let mut vocoder_channels = 1_024_u64;
    for (stage, kernel) in [16_u64, 16, 8, 4, 4].into_iter().enumerate() {
        let output_channels = vocoder_channels / 2;
        push_dense_convolution(
            &mut state,
            &format!("vocoder.ups.{stage}"),
            output_channels,
            vocoder_channels,
            kernel,
            true,
            true,
        );
        vocoder_channels = output_channels;
        for (kernel_index, residual_kernel) in [3_u64, 7, 11].into_iter().enumerate() {
            let residual = stage * 3 + kernel_index;
            for layer in 0..3 {
                push_dense_convolution(
                    &mut state,
                    &format!("vocoder.resblocks.{residual}.convs1.{layer}"),
                    vocoder_channels,
                    vocoder_channels,
                    residual_kernel,
                    true,
                    false,
                );
                push_dense_convolution(
                    &mut state,
                    &format!("vocoder.resblocks.{residual}.convs2.{layer}"),
                    vocoder_channels,
                    vocoder_channels,
                    residual_kernel,
                    true,
                    false,
                );
            }
        }
    }
    push_dense_convolution(
        &mut state,
        "vocoder.conv_post",
        2,
        vocoder_channels,
        7,
        true,
        false,
    );
    state
}

fn push_sa3_dynamic_tanh(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    push_audio_parameter(state, format!("{prefix}.alpha"), vec![1]);
    push_audio_parameter(state, format!("{prefix}.gamma"), vec![channels]);
    push_audio_parameter(state, format!("{prefix}.beta"), vec![channels]);
}

fn push_sa3_transformer(state: &mut Vec<AudioStateShape>, prefix: &str, channels: u64) {
    push_sa3_dynamic_tanh(state, &format!("{prefix}.pre_norm"), channels);
    push_music_linear(
        state,
        &format!("{prefix}.self_attn.to_qkv"),
        channels * 5,
        channels,
        false,
    );
    push_music_linear(
        state,
        &format!("{prefix}.self_attn.to_out"),
        channels,
        channels,
        false,
    );
    push_sa3_dynamic_tanh(state, &format!("{prefix}.self_attn.q_norm"), 64);
    push_sa3_dynamic_tanh(state, &format!("{prefix}.self_attn.k_norm"), 64);
    push_sa3_dynamic_tanh(state, &format!("{prefix}.ff_norm"), channels);
    push_music_linear(
        state,
        &format!("{prefix}.ff.ff.0.proj"),
        channels * 6,
        channels,
        true,
    );
    push_music_linear(
        state,
        &format!("{prefix}.ff.ff.2"),
        channels,
        channels * 3,
        true,
    );
    push_audio_parameter(state, format!("{prefix}.rope.inv_freq"), vec![16]);
}

fn stable_audio_3_source_state_shapes(profile: &VaeKernelProfile) -> Option<Vec<AudioStateShape>> {
    let (base_channels, depth, decoder_kernel) = match profile {
        VaeKernelProfile::StableAudio3DeepV1 => (256_u64, 12_usize, 1_u64),
        VaeKernelProfile::StableAudio3ShallowV1 => (128_u64, 6_usize, 3_u64),
        _ => return None,
    };
    let channels = base_channels * 6;
    let mut state = Vec::new();
    push_weight_normalized_convolution(
        &mut state,
        "encoder.layers.0.mapping",
        vec![channels, 512, 1],
        Some(channels),
    );
    push_audio_parameter(
        &mut state,
        "encoder.layers.0.new_tokens",
        vec![1, 1, channels],
    );
    for transformer in 0..depth {
        push_sa3_transformer(
            &mut state,
            &format!("encoder.layers.0.transformers.{transformer}"),
            channels,
        );
    }
    push_music_linear(&mut state, "encoder.layers.2", 256, channels, true);

    push_music_linear(&mut state, "decoder.layers.1", channels, 256, true);
    push_weight_normalized_convolution(
        &mut state,
        "decoder.layers.3.mapping",
        vec![512, channels, decoder_kernel],
        Some(512),
    );
    push_audio_parameter(
        &mut state,
        "decoder.layers.3.new_tokens",
        vec![1, 1, channels],
    );
    for transformer in 0..depth {
        push_sa3_transformer(
            &mut state,
            &format!("decoder.layers.3.transformers.{transformer}"),
            channels,
        );
    }
    push_audio_parameter(&mut state, "bottleneck.scaling_factor", vec![1, 256, 1]);
    push_audio_parameter(&mut state, "bottleneck.bias", vec![1, 256, 1]);
    push_audio_parameter(&mut state, "bottleneck.noise_scaling_factor", vec![1, 0, 1]);
    push_audio_parameter(&mut state, "bottleneck.running_std", vec![1]);
    Some(state)
}

fn audio_source_state_kind(
    profile: &VaeKernelProfile,
    canonical_name: &str,
) -> NativeVisionStateKind {
    let is_buffer = match profile {
        VaeKernelProfile::MmAudio16KhzV1 => {
            matches!(
                canonical_name,
                "mel_converter.mel_basis"
                    | "mel_converter.hann_window"
                    | "vae.data_mean"
                    | "vae.data_std"
            ) || canonical_name.ends_with(".upsample.filter")
                || canonical_name.ends_with(".downsample.lowpass.filter")
        }
        VaeKernelProfile::MusicDcaeV1 => matches!(
            canonical_name,
            "vocoder.mel_transform.spectrogram.window" | "vocoder.mel_transform.mel_scale.fb"
        ),
        VaeKernelProfile::LtxAudioV1 => matches!(
            canonical_name,
            "autoencoder.per_channel_statistics.std-of-means"
                | "autoencoder.per_channel_statistics.mean-of-means"
        ),
        VaeKernelProfile::StableAudio3DeepV1 | VaeKernelProfile::StableAudio3ShallowV1 => {
            canonical_name.ends_with(".rope.inv_freq")
        }
        _ => false,
    };
    if is_buffer {
        NativeVisionStateKind::Buffer
    } else {
        NativeVisionStateKind::Parameter
    }
}

fn audio_storage_dtype(name: &str, source_dtype: &str) -> Result<DType, AudioVaeError> {
    let dtype = canonical_vision_model_store_dtype(source_dtype).ok_or_else(|| {
        AudioVaeError::UnsupportedStorageDType {
            name: name.to_owned(),
            dtype: source_dtype.to_owned(),
        }
    })?;
    if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        return Err(AudioVaeError::UnsupportedStorageDType {
            name: name.to_owned(),
            dtype: source_dtype.to_owned(),
        });
    }
    Ok(dtype)
}

pub fn audio_vae_source_plan(
    profile: &VaeKernelProfile,
) -> Result<NativeAudioVaeArchitecture, AudioVaeError> {
    let (
        architecture,
        input_rate,
        output_rate,
        numerator,
        denominator,
        channels,
        dimensions,
        bins,
        checkpoints,
        equations,
    ) = match profile {
        VaeKernelProfile::AudioOobleck44KhzV1 => (
            OOBLECK_ARCHITECTURE,
            44_100,
            44_100,
            2_048,
            1,
            64,
            1,
            None,
            OOBLECK_STATE,
            OOBLECK_EQUATIONS,
        ),
        VaeKernelProfile::AudioOobleck48KhzV1 => (
            OOBLECK_ARCHITECTURE,
            48_000,
            48_000,
            1_920,
            1,
            64,
            1,
            None,
            OOBLECK_STATE,
            OOBLECK_EQUATIONS,
        ),
        VaeKernelProfile::MusicDcaeV1 => (
            MUSIC_DCAE_ARCHITECTURE,
            44_100,
            44_100,
            4_096,
            1,
            8,
            2,
            Some(16),
            MUSIC_DCAE_STATE,
            MUSIC_DCAE_EQUATIONS,
        ),
        VaeKernelProfile::MmAudio16KhzV1 => (
            MMAUDIO_ARCHITECTURE,
            44_100,
            44_100,
            141_120,
            100,
            20,
            1,
            None,
            MMAUDIO_STATE,
            MMAUDIO_EQUATIONS,
        ),
        VaeKernelProfile::LtxAudioV1 => (
            LTX_AUDIO_ARCHITECTURE,
            44_100,
            16_000,
            1_764,
            1,
            8,
            2,
            Some(16),
            LTX_AUDIO_STATE,
            LTX_AUDIO_EQUATIONS,
        ),
        VaeKernelProfile::StableAudio3DeepV1 | VaeKernelProfile::StableAudio3ShallowV1 => (
            STABLE_AUDIO_3_ARCHITECTURE,
            44_100,
            44_100,
            4_096,
            1,
            256,
            1,
            None,
            STABLE_AUDIO_3_STATE,
            STABLE_AUDIO_3_EQUATIONS,
        ),
        profile => return Err(AudioVaeError::UnsupportedProfile(profile.clone())),
    };
    Ok(NativeAudioVaeArchitecture {
        profile: profile.clone(),
        architecture,
        input_sample_rate: input_rate,
        output_sample_rate: output_rate,
        sample_ratio_numerator: numerator,
        sample_ratio_denominator: denominator,
        latent_channels: channels,
        latent_dimensions: dimensions,
        latent_frequency_bins: bins,
        checkpoints,
        equations,
        storage_dtype: None,
        state_schema: Vec::new(),
        source_names: BTreeMap::new(),
    })
}

fn apply_ltx_configuration(
    mut plan: NativeAudioVaeArchitecture,
    configuration: &VaeLoaderConfiguration,
) -> Result<NativeAudioVaeArchitecture, AudioVaeError> {
    let VaeLoaderConfiguration::LtxAudio {
        latent_channels,
        input_sample_rate,
        output_sample_rate,
        autoencoder_json,
        vocoder_json,
        ..
    } = configuration
    else {
        return Err(AudioVaeError::InvalidLtxConfiguration(
            "LTX Audio requires digest-bound autoencoder and vocoder configuration".to_owned(),
        ));
    };
    validate_ltx_audio_configuration(
        autoencoder_json,
        vocoder_json,
        *latent_channels,
        *input_sample_rate,
        *output_sample_rate,
    )?;
    plan.latent_channels = *latent_channels;
    plan.output_sample_rate = *output_sample_rate;
    Ok(plan)
}

fn ltx_configuration_object<'a>(
    value: &'a serde_json::Value,
    label: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, AudioVaeError> {
    value.as_object().ok_or_else(|| {
        AudioVaeError::InvalidLtxConfiguration(format!("{label} must be a JSON object"))
    })
}

fn ltx_u64(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: u64,
) -> Result<u64, AudioVaeError> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value.as_u64().ok_or_else(|| {
            AudioVaeError::InvalidLtxConfiguration(format!("{key} must be an unsigned integer"))
        }),
    }
}

fn ltx_bool(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: bool,
) -> Result<bool, AudioVaeError> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value.as_bool().ok_or_else(|| {
            AudioVaeError::InvalidLtxConfiguration(format!("{key} must be a boolean"))
        }),
    }
}

fn ltx_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: &'a str,
) -> Result<&'a str, AudioVaeError> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value.as_str().ok_or_else(|| {
            AudioVaeError::InvalidLtxConfiguration(format!("{key} must be a string"))
        }),
    }
}

fn ltx_u64_array(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: &[u64],
) -> Result<Vec<u64>, AudioVaeError> {
    let Some(value) = object.get(key) else {
        return Ok(default.to_vec());
    };
    value
        .as_array()
        .ok_or_else(|| AudioVaeError::InvalidLtxConfiguration(format!("{key} must be an array")))?
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                AudioVaeError::InvalidLtxConfiguration(format!(
                    "{key} entries must be unsigned integers"
                ))
            })
        })
        .collect()
}

fn ltx_u64_matrix(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: &[&[u64]],
) -> Result<Vec<Vec<u64>>, AudioVaeError> {
    let Some(value) = object.get(key) else {
        return Ok(default.iter().map(|row| row.to_vec()).collect());
    };
    value
        .as_array()
        .ok_or_else(|| AudioVaeError::InvalidLtxConfiguration(format!("{key} must be an array")))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| {
                    AudioVaeError::InvalidLtxConfiguration(format!("{key} rows must be arrays"))
                })?
                .iter()
                .map(|value| {
                    value.as_u64().ok_or_else(|| {
                        AudioVaeError::InvalidLtxConfiguration(format!(
                            "{key} entries must be unsigned integers"
                        ))
                    })
                })
                .collect()
        })
        .collect()
}

fn require_ltx_value<T: PartialEq + std::fmt::Debug>(
    label: &str,
    actual: T,
    expected: T,
) -> Result<(), AudioVaeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AudioVaeError::InvalidLtxConfiguration(format!(
            "unsupported {label}: expected {expected:?}, got {actual:?}"
        )))
    }
}

fn validate_ltx_side(
    side: &serde_json::Map<String, serde_json::Value>,
    label: &str,
) -> Result<(), AudioVaeError> {
    require_ltx_value(
        &format!("{label}.double_z"),
        ltx_bool(side, "double_z", true)?,
        true,
    )?;
    require_ltx_value(
        &format!("{label}.mel_bins"),
        ltx_u64(side, "mel_bins", 64)?,
        64,
    )?;
    require_ltx_value(
        &format!("{label}.z_channels"),
        ltx_u64(side, "z_channels", 8)?,
        8,
    )?;
    require_ltx_value(
        &format!("{label}.resolution"),
        ltx_u64(side, "resolution", 256)?,
        256,
    )?;
    require_ltx_value(
        &format!("{label}.in_channels"),
        ltx_u64(side, "in_channels", 2)?,
        2,
    )?;
    require_ltx_value(&format!("{label}.out_ch"), ltx_u64(side, "out_ch", 2)?, 2)?;
    require_ltx_value(&format!("{label}.ch"), ltx_u64(side, "ch", 128)?, 128)?;
    require_ltx_value(
        &format!("{label}.ch_mult"),
        ltx_u64_array(side, "ch_mult", &[1, 2, 4])?,
        vec![1, 2, 4],
    )?;
    require_ltx_value(
        &format!("{label}.num_res_blocks"),
        ltx_u64(side, "num_res_blocks", 2)?,
        2,
    )?;
    require_ltx_value(
        &format!("{label}.attn_resolutions"),
        ltx_u64_array(side, "attn_resolutions", &[])?,
        Vec::new(),
    )?;
    require_ltx_value(
        &format!("{label}.mid_block_add_attention"),
        ltx_bool(side, "mid_block_add_attention", false)?,
        false,
    )?;
    require_ltx_value(
        &format!("{label}.norm_type"),
        ltx_string(side, "norm_type", "pixel")?,
        "pixel",
    )?;
    require_ltx_value(
        &format!("{label}.causality_axis"),
        ltx_string(side, "causality_axis", "height")?,
        "height",
    )?;
    require_ltx_value(
        &format!("{label}.resamp_with_conv"),
        ltx_bool(side, "resamp_with_conv", true)?,
        true,
    )?;
    require_ltx_value(
        &format!("{label}.attn_type"),
        ltx_string(side, "attn_type", "vanilla")?,
        "vanilla",
    )?;
    if label == "decoder" {
        require_ltx_value(
            "decoder.give_pre_end",
            ltx_bool(side, "give_pre_end", false)?,
            false,
        )?;
        require_ltx_value(
            "decoder.tanh_out",
            ltx_bool(side, "tanh_out", false)?,
            false,
        )?;
    }
    let dropout = match side.get("dropout") {
        None => 0.0,
        Some(value) => value.as_f64().ok_or_else(|| {
            AudioVaeError::InvalidLtxConfiguration(format!("{label}.dropout must be numeric"))
        })?,
    };
    require_ltx_value(&format!("{label}.dropout"), dropout, 0.0)
}

fn validate_ltx_audio_configuration(
    autoencoder_json: &str,
    vocoder_json: &str,
    latent_channels: u64,
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> Result<(), AudioVaeError> {
    let autoencoder: serde_json::Value = serde_json::from_str(autoencoder_json)
        .map_err(|error| AudioVaeError::InvalidLtxConfiguration(error.to_string()))?;
    let autoencoder = ltx_configuration_object(&autoencoder, "audio_vae")?;
    let params = autoencoder
        .get("model")
        .and_then(serde_json::Value::as_object)
        .and_then(|model| model.get("params"))
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            AudioVaeError::InvalidLtxConfiguration(
                "audio_vae.model.params must be an object".to_owned(),
            )
        })?;
    require_ltx_value("latent_channels", latent_channels, 8)?;
    require_ltx_value("input_sample_rate", input_sample_rate, 16_000)?;
    require_ltx_value("output_sample_rate", output_sample_rate, 16_000)?;
    require_ltx_value(
        "sampling_rate",
        ltx_u64(params, "sampling_rate", 16_000)?,
        16_000,
    )?;
    let encoder = params
        .get("encoder")
        .or_else(|| params.get("ddconfig"))
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            AudioVaeError::InvalidLtxConfiguration(
                "audio_vae encoder or ddconfig must be an object".to_owned(),
            )
        })?;
    let decoder = params
        .get("decoder")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(encoder);
    validate_ltx_side(encoder, "encoder")?;
    validate_ltx_side(decoder, "decoder")?;
    let stft = autoencoder
        .get("preprocessing")
        .and_then(serde_json::Value::as_object)
        .and_then(|preprocessing| preprocessing.get("stft"))
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            AudioVaeError::InvalidLtxConfiguration(
                "audio_vae.preprocessing.stft must be an object".to_owned(),
            )
        })?;
    require_ltx_value(
        "stft.filter_length",
        ltx_u64(stft, "filter_length", 1_024)?,
        1_024,
    )?;
    require_ltx_value("stft.hop_length", ltx_u64(stft, "hop_length", 160)?, 160)?;

    let vocoder: serde_json::Value = serde_json::from_str(vocoder_json)
        .map_err(|error| AudioVaeError::InvalidLtxConfiguration(error.to_string()))?;
    let vocoder = ltx_configuration_object(&vocoder, "vocoder")?;
    if vocoder.contains_key("bwe") || vocoder.contains_key("vocoder") {
        return Err(AudioVaeError::InvalidLtxConfiguration(
            "bandwidth-extension LTX vocoders require a distinct supported configuration"
                .to_owned(),
        ));
    }
    require_ltx_value(
        "vocoder.resblock_kernel_sizes",
        ltx_u64_array(vocoder, "resblock_kernel_sizes", &[3, 7, 11])?,
        vec![3, 7, 11],
    )?;
    require_ltx_value(
        "vocoder.upsample_rates",
        ltx_u64_array(vocoder, "upsample_rates", &[5, 4, 2, 2, 2])?,
        vec![5, 4, 2, 2, 2],
    )?;
    require_ltx_value(
        "vocoder.upsample_kernel_sizes",
        ltx_u64_array(vocoder, "upsample_kernel_sizes", &[16, 16, 8, 4, 4])?,
        vec![16, 16, 8, 4, 4],
    )?;
    require_ltx_value(
        "vocoder.upsample_initial_channel",
        ltx_u64(vocoder, "upsample_initial_channel", 1_024)?,
        1_024,
    )?;
    require_ltx_value(
        "vocoder.resblock_dilation_sizes",
        ltx_u64_matrix(
            vocoder,
            "resblock_dilation_sizes",
            &[&[1, 3, 5], &[1, 3, 5], &[1, 3, 5]],
        )?,
        vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
    )?;
    require_ltx_value("vocoder.stereo", ltx_bool(vocoder, "stereo", true)?, true)?;
    require_ltx_value(
        "vocoder.resblock",
        ltx_string(vocoder, "resblock", "1")?,
        "1",
    )?;
    require_ltx_value(
        "vocoder.activation",
        ltx_string(vocoder, "activation", "snake")?,
        "snake",
    )?;
    require_ltx_value(
        "vocoder.use_bias_at_final",
        ltx_bool(vocoder, "use_bias_at_final", true)?,
        true,
    )?;
    require_ltx_value(
        "vocoder.use_tanh_at_final",
        ltx_bool(vocoder, "use_tanh_at_final", true)?,
        true,
    )?;
    require_ltx_value(
        "vocoder.apply_final_activation",
        ltx_bool(vocoder, "apply_final_activation", true)?,
        true,
    )?;
    match vocoder.get("output_sample_rate") {
        None | Some(serde_json::Value::Null) => Ok(()),
        Some(value) => require_ltx_value(
            "vocoder.output_sample_rate",
            value.as_u64().ok_or_else(|| {
                AudioVaeError::InvalidLtxConfiguration(
                    "vocoder.output_sample_rate must be an unsigned integer or null".to_owned(),
                )
            })?,
            16_000,
        ),
    }
}

pub fn inspect_audio_vae_architecture(
    descriptor: &VaeDescriptor,
    model: &LoadedModel,
) -> Result<NativeAudioVaeArchitecture, AudioVaeError> {
    inspect_audio_vae_architecture_from_tensors(
        descriptor.identity().profile(),
        descriptor.identity().loader_configuration(),
        descriptor.identity().architecture().as_str(),
        model.tensors(),
    )
}

fn inspect_audio_vae_architecture_from_tensors(
    profile: &VaeKernelProfile,
    loader_configuration: &VaeLoaderConfiguration,
    actual_architecture: &str,
    tensors: &BTreeMap<String, impl AudioVaeStateMetadata>,
) -> Result<NativeAudioVaeArchitecture, AudioVaeError> {
    inspect_audio_vae_architecture_from_state(
        profile,
        loader_configuration,
        actual_architecture,
        tensors,
    )
}

trait AudioVaeStateMetadata {
    fn shape(&self) -> &[u64];
    fn data_type(&self) -> &str;
}

impl AudioVaeStateMetadata for TensorMetadata {
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn data_type(&self) -> &str {
        &self.data_type
    }
}

impl AudioVaeStateMetadata for Tensor {
    fn shape(&self) -> &[u64] {
        self.descriptor().shape()
    }

    fn data_type(&self) -> &str {
        self.descriptor().dtype().catalog_name()
    }
}

fn inspect_audio_vae_architecture_from_state<T: AudioVaeStateMetadata>(
    profile: &VaeKernelProfile,
    loader_configuration: &VaeLoaderConfiguration,
    actual_architecture: &str,
    tensors: &BTreeMap<String, T>,
) -> Result<NativeAudioVaeArchitecture, AudioVaeError> {
    let mut plan = audio_vae_source_plan(profile)?;
    if plan.profile == VaeKernelProfile::LtxAudioV1 {
        plan = apply_ltx_configuration(plan, loader_configuration)?;
    }
    if actual_architecture != plan.architecture() {
        return Err(AudioVaeError::ArchitectureMismatch {
            expected: plan.architecture(),
            actual: actual_architecture.to_owned(),
        });
    }
    let mut storage_dtype = None;
    let mut names = BTreeSet::new();
    if let Some(expected_state) = oobleck_source_state_shapes(plan.profile()) {
        for expected in expected_state {
            if !names.insert(expected.name.clone()) {
                return Err(AudioVaeError::DuplicateState(expected.name));
            }
            let source_name = if tensors.contains_key(&expected.name) {
                expected.name.clone()
            } else if let Some(prefix) = expected
                .name
                .strip_suffix(".parametrizations.weight.original0")
            {
                format!("{prefix}.weight_g")
            } else if let Some(prefix) = expected
                .name
                .strip_suffix(".parametrizations.weight.original1")
            {
                format!("{prefix}.weight_v")
            } else {
                expected.name.clone()
            };
            let metadata = tensors
                .get(&source_name)
                .ok_or_else(|| AudioVaeError::MissingState(expected.name.clone()))?;
            if metadata.shape() != expected.shape {
                return Err(AudioVaeError::InvalidStateShape {
                    name: expected.name,
                    shape: metadata.shape().to_vec(),
                });
            }
            let dtype = audio_storage_dtype(&expected.name, metadata.data_type())?;
            if let Some(expected_dtype) = storage_dtype {
                if expected_dtype != dtype {
                    return Err(AudioVaeError::MixedStorageDType {
                        name: expected.name,
                        expected: expected_dtype,
                        actual: dtype,
                    });
                }
            } else {
                storage_dtype = Some(dtype);
            }
            plan.state_schema.push(NativeVisionStateSpec {
                name: source_name.clone(),
                shape: expected.shape,
                dtype,
                kind: NativeVisionStateKind::Parameter,
            });
            if source_name != expected.name {
                plan.source_names.insert(expected.name, source_name);
            }
        }
        let admitted_source_names = plan
            .state_schema
            .iter()
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(name) = tensors
            .keys()
            .find(|name| !admitted_source_names.contains(name.as_str()))
        {
            return Err(AudioVaeError::UnexpectedState(name.clone()));
        }
        plan.storage_dtype = storage_dtype;
        return Ok(plan);
    }
    let exact_state = match plan.profile() {
        VaeKernelProfile::MusicDcaeV1 => Some(music_dcae_source_state_shapes()),
        VaeKernelProfile::MmAudio16KhzV1 => Some(mmaudio_source_state_shapes()),
        VaeKernelProfile::LtxAudioV1 => Some(ltx_audio_source_state_shapes()),
        VaeKernelProfile::StableAudio3DeepV1 | VaeKernelProfile::StableAudio3ShallowV1 => {
            stable_audio_3_source_state_shapes(plan.profile())
        }
        _ => None,
    };
    if let Some(exact_state) = exact_state {
        for expected in exact_state {
            if !names.insert(expected.name.clone()) {
                return Err(AudioVaeError::DuplicateState(expected.name));
            }
            let source_name = if tensors.contains_key(&expected.name) {
                expected.name.clone()
            } else if let Some(suffix) = expected.name.strip_prefix("autoencoder.") {
                let legacy = format!("audio_vae.{suffix}");
                if tensors.contains_key(&legacy) {
                    legacy
                } else {
                    expected.name.clone()
                }
            } else if let Some(prefix) = expected
                .name
                .strip_suffix(".parametrizations.weight.original0")
            {
                format!("{prefix}.weight_g")
            } else if let Some(prefix) = expected
                .name
                .strip_suffix(".parametrizations.weight.original1")
            {
                let legacy = format!("{prefix}.weight_v");
                if tensors.contains_key(&legacy) {
                    legacy
                } else {
                    format!("{prefix}.weight")
                }
            } else {
                expected.name.clone()
            };
            let metadata = tensors
                .get(&source_name)
                .ok_or_else(|| AudioVaeError::MissingState(expected.name.clone()))?;
            if metadata.shape() != expected.shape {
                return Err(AudioVaeError::InvalidStateShape {
                    name: expected.name,
                    shape: metadata.shape().to_vec(),
                });
            }
            let dtype = audio_storage_dtype(&expected.name, metadata.data_type())?;
            if let Some(expected_dtype) = storage_dtype {
                if expected_dtype != dtype {
                    return Err(AudioVaeError::MixedStorageDType {
                        name: expected.name,
                        expected: expected_dtype,
                        actual: dtype,
                    });
                }
            } else {
                storage_dtype = Some(dtype);
            }
            plan.state_schema.push(NativeVisionStateSpec {
                name: source_name.clone(),
                shape: expected.shape,
                dtype,
                kind: audio_source_state_kind(plan.profile(), &expected.name),
            });
            if source_name != expected.name {
                plan.source_names.insert(expected.name, source_name);
            }
        }
        let admitted_source_names = plan
            .state_schema
            .iter()
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(name) = tensors
            .keys()
            .find(|name| !admitted_source_names.contains(name.as_str()))
        {
            return Err(AudioVaeError::UnexpectedState(name.clone()));
        }
        plan.storage_dtype = storage_dtype;
        return Ok(plan);
    }
    for checkpoint in plan.state_checkpoints() {
        if !names.insert(checkpoint.name.to_owned()) {
            continue;
        }
        let metadata = tensors
            .get(checkpoint.name)
            .ok_or_else(|| AudioVaeError::MissingState(checkpoint.name.to_owned()))?;
        if metadata.shape().len() != usize::from(checkpoint.rank)
            || metadata.shape().contains(&0)
            || checkpoint
                .dimensions
                .iter()
                .any(|(axis, expected)| metadata.shape().get(*axis) != Some(expected))
        {
            return Err(AudioVaeError::InvalidStateShape {
                name: checkpoint.name.to_owned(),
                shape: metadata.shape().to_vec(),
            });
        }
        let dtype = audio_storage_dtype(checkpoint.name, metadata.data_type())?;
        if let Some(expected) = storage_dtype {
            if expected != dtype {
                return Err(AudioVaeError::MixedStorageDType {
                    name: checkpoint.name.to_owned(),
                    expected,
                    actual: dtype,
                });
            }
        } else {
            storage_dtype = Some(dtype);
        }
        plan.state_schema.push(NativeVisionStateSpec {
            name: checkpoint.name.to_owned(),
            shape: metadata.shape().to_vec(),
            dtype,
            kind: NativeVisionStateKind::Parameter,
        });
    }
    for (name, metadata) in tensors {
        if names.contains(name) {
            continue;
        }
        if metadata.shape().contains(&0) {
            return Err(AudioVaeError::InvalidStateShape {
                name: name.clone(),
                shape: metadata.shape().to_vec(),
            });
        }
        let dtype = audio_storage_dtype(name, metadata.data_type())?;
        if let Some(expected) = storage_dtype {
            if expected != dtype {
                return Err(AudioVaeError::MixedStorageDType {
                    name: name.clone(),
                    expected,
                    actual: dtype,
                });
            }
        } else {
            storage_dtype = Some(dtype);
        }
        names.insert(name.clone());
        plan.state_schema.push(NativeVisionStateSpec {
            name: name.clone(),
            shape: metadata.shape().to_vec(),
            dtype,
            kind: NativeVisionStateKind::Parameter,
        });
    }
    plan.state_schema
        .sort_by(|left, right| left.name.cmp(&right.name));
    plan.storage_dtype = storage_dtype;
    Ok(plan)
}

pub fn audio_vae_source_state_schema(
    descriptor: &VaeDescriptor,
    model: &LoadedModel,
) -> Result<Vec<NativeVisionStateSpec>, AudioVaeError> {
    Ok(inspect_audio_vae_architecture(descriptor, model)?.state_schema)
}

pub fn load_audio_vae_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: Arc<LoadedModel>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, AudioVaeError> {
    context.cancellation.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let architecture = inspect_audio_vae_architecture(&descriptor, &model)?;
    let mut state = load_vision_state_from_model_store_with_context(
        backend,
        store,
        index,
        &model,
        architecture.state_schema(),
        context,
    )?;
    for (canonical_name, source_name) in &architecture.source_names {
        let tensor = state
            .remove(source_name)
            .ok_or_else(|| AudioVaeError::MissingState(source_name.clone()))?;
        if state.insert(canonical_name.clone(), tensor).is_some() {
            return Err(AudioVaeError::UnexpectedState(canonical_name.clone()));
        }
    }
    let module = build_audio_module(
        &architecture,
        state,
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
        context,
    )?;
    let binding =
        VaeModelBinding::checked(&descriptor, store, model, module, context.cancellation)?;
    let functions = VaeKernelFunctions::checked(
        descriptor.identity().architecture().clone(),
        audio_encode_raw,
        audio_decode_raw,
    );
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

pub(crate) fn load_audio_vae_from_recipe_state(
    backend: &CpuBackend,
    artifact_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, AudioVaeError> {
    context.cancellation.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let mut state = BTreeMap::new();
    for (name, tensor) in ordered_state {
        if state.insert(name.clone(), tensor).is_some() {
            return Err(AudioVaeError::DuplicateState(name));
        }
    }
    let architecture = inspect_audio_vae_architecture_from_state(
        descriptor.identity().profile(),
        descriptor.identity().loader_configuration(),
        descriptor.identity().architecture().as_str(),
        &state,
    )?;
    for (canonical_name, source_name) in &architecture.source_names {
        let tensor = state
            .remove(source_name)
            .ok_or_else(|| AudioVaeError::MissingState(source_name.clone()))?;
        if state.insert(canonical_name.clone(), tensor).is_some() {
            return Err(AudioVaeError::UnexpectedState(canonical_name.clone()));
        }
    }
    let module = build_audio_module(
        &architecture,
        state,
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
        audio_encode_raw,
        audio_decode_raw,
    );
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

#[cfg(feature = "test-support")]
pub(crate) fn load_audio_vae_from_reduced_recipe_state(
    backend: &CpuBackend,
    artifact_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeVae, AudioVaeError> {
    context.cancellation.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    if descriptor.identity().profile() != &VaeKernelProfile::AudioOobleck44KhzV1
        || descriptor.identity().architecture().as_str() != OOBLECK_ARCHITECTURE
    {
        return Err(AudioVaeError::UnsupportedProfile(
            descriptor.identity().profile().clone(),
        ));
    }
    if !ordered_state
        .iter()
        .any(|(name, _)| name == "decoder.layers.1.layers.0.beta")
    {
        return Err(AudioVaeError::MissingState(
            "decoder.layers.1.layers.0.beta".to_owned(),
        ));
    }
    let module = crate::vae::reduced_recipe_state_module(
        "audio-vae-reduced-recipe",
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
        audio_encode_raw,
        audio_decode_raw,
    );
    Ok(NativeVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        functions,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn push_oobleck_convolution(
    children: &mut Vec<NativeModule>,
    state: &mut BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    name: &str,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    transposed: bool,
    bias: bool,
    context: &ExecutionContext<'_>,
) -> Result<(), AudioVaeError> {
    let geometry = ConvolutionGeometry::new(
        1,
        vec![stride],
        vec![padding],
        vec![dilation],
        1,
        transposed,
        vec![0],
    )
    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    let mut module = NativeModule::convolution(
        name,
        input_channels,
        output_channels,
        vec![kernel],
        bias,
        geometry,
        false,
    )?;
    let magnitude_name = format!("{name}.parametrizations.weight.original0");
    let direction_name = format!("{name}.parametrizations.weight.original1");
    let magnitude = state
        .remove(&magnitude_name)
        .ok_or_else(|| AudioVaeError::MissingState(magnitude_name))?;
    let direction = state
        .remove(&direction_name)
        .ok_or_else(|| AudioVaeError::MissingState(direction_name))?;
    let bias_tensor = if bias {
        let bias_name = format!("{name}.bias");
        Some(
            state
                .remove(&bias_name)
                .ok_or_else(|| AudioVaeError::MissingState(bias_name))?,
        )
    } else {
        None
    };
    module.load_weight_norm_parameters_with_context_exact_native(
        backend,
        magnitude,
        direction,
        bias_tensor,
        Some(0),
        context,
    )?;
    children.push(module);
    Ok(())
}

fn push_oobleck_activation_buffers(
    children: &mut Vec<NativeModule>,
    state: &mut BTreeMap<String, Tensor>,
    name: &str,
) -> Result<(), AudioVaeError> {
    for parameter in ["alpha", "beta"] {
        let parameter_name = format!("{name}.{parameter}");
        let tensor = state
            .remove(&parameter_name)
            .ok_or_else(|| AudioVaeError::MissingState(parameter_name.clone()))?;
        children.push(NativeModule::buffer(parameter_name, tensor)?);
    }
    Ok(())
}

fn push_oobleck_residual_modules(
    children: &mut Vec<NativeModule>,
    state: &mut BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    prefix: &str,
    channels: usize,
    dilation: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), AudioVaeError> {
    push_oobleck_activation_buffers(children, state, &format!("{prefix}.layers.0"))?;
    push_oobleck_convolution(
        children,
        state,
        backend,
        &format!("{prefix}.layers.1"),
        channels,
        channels,
        7,
        1,
        dilation * 3,
        dilation,
        false,
        true,
        context,
    )?;
    push_oobleck_activation_buffers(children, state, &format!("{prefix}.layers.2"))?;
    push_oobleck_convolution(
        children,
        state,
        backend,
        &format!("{prefix}.layers.3"),
        channels,
        channels,
        1,
        1,
        0,
        1,
        false,
        true,
        context,
    )
}

fn build_oobleck_module(
    profile: &VaeKernelProfile,
    mut state: BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    execution_dtype: DType,
    execution_device: sim_tensor::DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, AudioVaeError> {
    let strides = match profile {
        VaeKernelProfile::AudioOobleck44KhzV1 => [2_usize, 4, 4, 8, 8],
        VaeKernelProfile::AudioOobleck48KhzV1 => [2_usize, 4, 4, 6, 10],
        _ => return Err(AudioVaeError::UnsupportedProfile(profile.clone())),
    };
    let multipliers = [1_usize, 1, 2, 4, 8, 16];
    let base_channels = 128_usize;
    let mut children = Vec::new();
    push_oobleck_convolution(
        &mut children,
        &mut state,
        backend,
        "encoder.layers.0",
        2,
        base_channels,
        7,
        1,
        3,
        1,
        false,
        true,
        context,
    )?;
    for block in 0..5 {
        let input_channels = multipliers[block] * base_channels;
        let output_channels = multipliers[block + 1] * base_channels;
        let prefix = format!("encoder.layers.{}", block + 1);
        for (residual, dilation) in [1_usize, 3, 9].into_iter().enumerate() {
            push_oobleck_residual_modules(
                &mut children,
                &mut state,
                backend,
                &format!("{prefix}.layers.{residual}"),
                input_channels,
                dilation,
                context,
            )?;
        }
        push_oobleck_activation_buffers(&mut children, &mut state, &format!("{prefix}.layers.3"))?;
        let stride = strides[block];
        push_oobleck_convolution(
            &mut children,
            &mut state,
            backend,
            &format!("{prefix}.layers.4"),
            input_channels,
            output_channels,
            2 * stride,
            stride,
            stride.div_ceil(2),
            1,
            false,
            true,
            context,
        )?;
    }
    push_oobleck_activation_buffers(&mut children, &mut state, "encoder.layers.6")?;
    push_oobleck_convolution(
        &mut children,
        &mut state,
        backend,
        "encoder.layers.7",
        16 * base_channels,
        128,
        3,
        1,
        1,
        1,
        false,
        true,
        context,
    )?;

    push_oobleck_convolution(
        &mut children,
        &mut state,
        backend,
        "decoder.layers.0",
        64,
        16 * base_channels,
        7,
        1,
        3,
        1,
        false,
        true,
        context,
    )?;
    for block in 0..5 {
        let multiplier_index = 5 - block;
        let input_channels = multipliers[multiplier_index] * base_channels;
        let output_channels = multipliers[multiplier_index - 1] * base_channels;
        let stride = strides[multiplier_index - 1];
        let prefix = format!("decoder.layers.{}", block + 1);
        push_oobleck_activation_buffers(&mut children, &mut state, &format!("{prefix}.layers.0"))?;
        push_oobleck_convolution(
            &mut children,
            &mut state,
            backend,
            &format!("{prefix}.layers.1"),
            input_channels,
            output_channels,
            2 * stride,
            stride,
            stride.div_ceil(2),
            1,
            true,
            true,
            context,
        )?;
        for (residual, dilation) in [1_usize, 3, 9].into_iter().enumerate() {
            push_oobleck_residual_modules(
                &mut children,
                &mut state,
                backend,
                &format!("{prefix}.layers.{}", residual + 2),
                output_channels,
                dilation,
                context,
            )?;
        }
    }
    push_oobleck_activation_buffers(&mut children, &mut state, "decoder.layers.6")?;
    push_oobleck_convolution(
        &mut children,
        &mut state,
        backend,
        "decoder.layers.7",
        base_channels,
        2,
        7,
        1,
        3,
        1,
        false,
        false,
        context,
    )?;
    if let Some((name, _)) = state.into_iter().next() {
        return Err(AudioVaeError::UnexpectedState(name));
    }
    let mut module = NativeModule::module_dict(format!("audio-vae:{profile:?}"), children)?;
    module.materialize_execution_state_with_context(
        backend,
        execution_dtype,
        execution_device,
        context,
    )?;
    Ok(module)
}

fn mmaudio_convolution_geometry(
    name: &str,
    kernel: usize,
) -> Result<ConvolutionGeometry, AudioVaeError> {
    let mut stride = 1_usize;
    let mut dilation = 1_usize;
    let mut transposed = false;
    if let Some(rest) = name.strip_prefix("vocoder.ups.") {
        let stage = rest
            .split('.')
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
        stride = *[4_usize, 4, 2, 2, 2, 2]
            .get(stage)
            .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
        transposed = true;
    } else if name.contains(".resblocks.") && name.contains(".convs1.") {
        let index = name
            .split(".convs1.")
            .nth(1)
            .and_then(|rest| rest.split('.').next())
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
        dilation = *[1_usize, 3, 5]
            .get(index)
            .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
    }
    let padding = if transposed {
        kernel.saturating_sub(stride) / 2
    } else {
        kernel
            .checked_mul(dilation)
            .and_then(|value| value.checked_sub(dilation))
            .ok_or(AudioVaeError::ShapeOverflow)?
            / 2
    };
    ConvolutionGeometry::new(
        1,
        vec![stride],
        vec![padding],
        vec![dilation],
        1,
        transposed,
        vec![0],
    )
    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()).into())
}

fn build_mmaudio_module(
    profile: &VaeKernelProfile,
    mut state: BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    execution_dtype: DType,
    execution_device: sim_tensor::DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, AudioVaeError> {
    let convolution_names = state
        .iter()
        .filter(|(name, tensor)| name.ends_with(".weight") && tensor.descriptor().rank() == 3)
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut children = Vec::new();
    for weight_name in convolution_names {
        let weight = state
            .remove(&weight_name)
            .ok_or_else(|| AudioVaeError::MissingState(weight_name.clone()))?;
        let shape = weight.descriptor().shape();
        let kernel = usize::try_from(shape[2]).map_err(|_| AudioVaeError::ShapeOverflow)?;
        let geometry = mmaudio_convolution_geometry(&weight_name, kernel)?;
        let (input_channels, output_channels) = if geometry.transposed() {
            (
                usize::try_from(shape[0]).map_err(|_| AudioVaeError::ShapeOverflow)?,
                usize::try_from(shape[1]).map_err(|_| AudioVaeError::ShapeOverflow)?,
            )
        } else {
            (
                usize::try_from(shape[1]).map_err(|_| AudioVaeError::ShapeOverflow)?,
                usize::try_from(shape[0]).map_err(|_| AudioVaeError::ShapeOverflow)?,
            )
        };
        let prefix = weight_name
            .strip_suffix(".weight")
            .ok_or_else(|| AudioVaeError::UnexpectedState(weight_name.clone()))?;
        let bias_name = format!("{prefix}.bias");
        let bias = state.remove(&bias_name);
        let mut convolution = NativeModule::convolution(
            weight_name,
            input_channels,
            output_channels,
            vec![kernel],
            bias.is_some(),
            geometry,
            false,
        )?;
        convolution.load_dense_parameters(weight, bias)?;
        children.push(convolution);
    }
    children.extend(
        state
            .into_iter()
            .map(|(name, tensor)| NativeModule::buffer(name, tensor))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let mut module = NativeModule::module_dict(format!("audio-vae:{profile:?}"), children)?;
    module.materialize_execution_state_with_context(
        backend,
        execution_dtype,
        execution_device,
        context,
    )?;
    Ok(module)
}

fn music_convolution_geometry(
    name: &str,
    shape: &[u64],
) -> Result<(usize, usize, ConvolutionGeometry), AudioVaeError> {
    let rank = shape.len();
    if !matches!(rank, 3 | 4) {
        return Err(AudioVaeError::UnexpectedState(name.to_owned()));
    }
    let mut stride = vec![1_usize; rank - 2];
    let mut dilation = vec![1_usize; rank - 2];
    let mut transposed = false;
    let mut groups = 1_usize;
    let mut padding_mode = ConvolutionPaddingMode::Zeros;

    if rank == 3 {
        if let Some(rest) = name.strip_prefix("vocoder.head.ups.") {
            let stage = rest
                .split('.')
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
            stride[0] = *[4_usize, 4, 2, 2, 2, 2, 2]
                .get(stage)
                .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
            transposed = true;
        } else if name.contains("vocoder.head.resblocks.") && name.contains(".convs1.") {
            let layer = name
                .split(".convs1.")
                .nth(1)
                .and_then(|rest| rest.split('.').next())
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
            dilation[0] = *[1_usize, 3, 5]
                .get(layer)
                .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
        }
        if name.contains(".dwconv.") || name.ends_with(".dwconv") {
            groups = usize::try_from(shape[0]).map_err(|_| AudioVaeError::ShapeOverflow)?;
        }
        if name == "vocoder.backbone.channel_layers.0.0" {
            padding_mode = ConvolutionPaddingMode::Replicate;
        }
    } else {
        if name.starts_with("dcae.encoder.down_blocks.") && name.ends_with(".conv") {
            stride.fill(2);
        }
        if name.contains(".proj_in") || name.contains(".conv_depth") {
            groups = usize::try_from(shape[0]).map_err(|_| AudioVaeError::ShapeOverflow)?;
        } else if name.contains(".proj_out") {
            let output_channels =
                usize::try_from(shape[0]).map_err(|_| AudioVaeError::ShapeOverflow)?;
            groups = output_channels
                .checked_div(32)
                .ok_or(AudioVaeError::ShapeOverflow)?;
        }
    }
    let kernel = shape
        .get(2..)
        .ok_or(AudioVaeError::ShapeOverflow)?
        .iter()
        .map(|value| usize::try_from(*value).map_err(|_| AudioVaeError::ShapeOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let padding = kernel
        .iter()
        .zip(&dilation)
        .zip(&stride)
        .map(|((&kernel, &dilation), &stride)| {
            if transposed {
                kernel.saturating_sub(stride) / 2
            } else {
                kernel.saturating_mul(dilation).saturating_sub(dilation) / 2
            }
        })
        .collect::<Vec<_>>();
    let padding = if name == "vocoder.backbone.channel_layers.0.0" {
        vec![0]
    } else {
        padding
    };
    let geometry = ConvolutionGeometry::new_with_padding_mode(
        rank - 2,
        stride,
        padding,
        dilation,
        groups,
        transposed,
        vec![0; rank - 2],
        padding_mode,
    )
    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    let (input_channels, output_channels) = if transposed {
        (
            usize::try_from(shape[0]).map_err(|_| AudioVaeError::ShapeOverflow)?,
            usize::try_from(shape[1])
                .map_err(|_| AudioVaeError::ShapeOverflow)?
                .checked_mul(groups)
                .ok_or(AudioVaeError::ShapeOverflow)?,
        )
    } else {
        (
            usize::try_from(shape[1])
                .map_err(|_| AudioVaeError::ShapeOverflow)?
                .checked_mul(groups)
                .ok_or(AudioVaeError::ShapeOverflow)?,
            usize::try_from(shape[0]).map_err(|_| AudioVaeError::ShapeOverflow)?,
        )
    };
    Ok((input_channels, output_channels, geometry))
}

fn ltx_convolution_geometry(
    name: &str,
    shape: &[u64],
) -> Result<(usize, usize, ConvolutionGeometry), AudioVaeError> {
    let rank = shape.len();
    if !matches!(rank, 3 | 4) {
        return Err(AudioVaeError::UnexpectedState(name.to_owned()));
    }
    let mut stride = vec![1_usize; rank - 2];
    let mut dilation = vec![1_usize; rank - 2];
    let mut transposed = false;
    if rank == 4 {
        if name.contains(".downsample.conv") {
            stride.fill(2);
        }
    } else if let Some(rest) = name.strip_prefix("vocoder.ups.") {
        let stage = rest
            .split('.')
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
        stride[0] = *[5_usize, 4, 2, 2, 2]
            .get(stage)
            .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
        transposed = true;
    } else if name.contains("vocoder.resblocks.") && name.contains(".convs1.") {
        let layer = name
            .split(".convs1.")
            .nth(1)
            .and_then(|rest| rest.split('.').next())
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
        dilation[0] = *[1_usize, 3, 5]
            .get(layer)
            .ok_or_else(|| AudioVaeError::UnexpectedState(name.to_owned()))?;
    }
    let kernel = shape[2..]
        .iter()
        .map(|extent| usize::try_from(*extent).map_err(|_| AudioVaeError::ShapeOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let padding = kernel
        .iter()
        .zip(&stride)
        .zip(&dilation)
        .map(|((&kernel, &stride), &dilation)| {
            if rank == 4 {
                0
            } else if transposed {
                kernel.saturating_sub(stride) / 2
            } else {
                kernel.saturating_mul(dilation).saturating_sub(dilation) / 2
            }
        })
        .collect::<Vec<_>>();
    let geometry = ConvolutionGeometry::new(
        rank - 2,
        stride,
        padding,
        dilation,
        1,
        transposed,
        vec![0; rank - 2],
    )
    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    let (input_channels, output_channels) = if transposed {
        (shape[0], shape[1])
    } else {
        (shape[1], shape[0])
    };
    Ok((
        usize::try_from(input_channels).map_err(|_| AudioVaeError::ShapeOverflow)?,
        usize::try_from(output_channels).map_err(|_| AudioVaeError::ShapeOverflow)?,
        geometry,
    ))
}

fn dense_audio_convolution_geometry(
    profile: &VaeKernelProfile,
    name: &str,
    shape: &[u64],
) -> Result<(usize, usize, ConvolutionGeometry), AudioVaeError> {
    if profile == &VaeKernelProfile::LtxAudioV1 {
        ltx_convolution_geometry(name, shape)
    } else {
        music_convolution_geometry(name, shape)
    }
}

fn build_dense_audio_module(
    profile: &VaeKernelProfile,
    mut state: BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    execution_dtype: DType,
    execution_device: sim_tensor::DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, AudioVaeError> {
    let weight_norm_names = state
        .keys()
        .filter_map(|name| {
            name.strip_suffix(".parametrizations.weight.original1")
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    let mut children = Vec::new();
    for prefix in weight_norm_names {
        let magnitude_name = format!("{prefix}.parametrizations.weight.original0");
        let direction_name = format!("{prefix}.parametrizations.weight.original1");
        let magnitude = state
            .remove(&magnitude_name)
            .ok_or_else(|| AudioVaeError::MissingState(magnitude_name))?;
        let direction = state
            .remove(&direction_name)
            .ok_or_else(|| AudioVaeError::MissingState(direction_name))?;
        let shape = direction.descriptor().shape();
        let (input_channels, output_channels, geometry) =
            dense_audio_convolution_geometry(profile, &prefix, shape)?;
        let kernel = shape
            .get(2..)
            .ok_or(AudioVaeError::ShapeOverflow)?
            .iter()
            .map(|value| usize::try_from(*value).map_err(|_| AudioVaeError::ShapeOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let bias = state.remove(&format!("{prefix}.bias"));
        let mut module = NativeModule::convolution(
            prefix,
            input_channels,
            output_channels,
            kernel,
            bias.is_some(),
            geometry,
            false,
        )?;
        module.load_weight_norm_parameters_with_context_exact_native(
            backend,
            magnitude,
            direction,
            bias,
            Some(0),
            context,
        )?;
        children.push(module);
    }

    let dense_weight_names = state
        .iter()
        .filter(|(name, tensor)| {
            name.ends_with(".weight") && matches!(tensor.descriptor().rank(), 2..=4)
        })
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    for weight_name in dense_weight_names {
        let weight = state
            .remove(&weight_name)
            .ok_or_else(|| AudioVaeError::MissingState(weight_name.clone()))?;
        let shape = weight.descriptor().shape();
        let prefix = weight_name
            .strip_suffix(".weight")
            .ok_or_else(|| AudioVaeError::UnexpectedState(weight_name.clone()))?;
        let bias = state.remove(&format!("{prefix}.bias"));
        let mut module = if shape.len() == 2 {
            NativeModule::linear(
                weight_name,
                usize::try_from(shape[1]).map_err(|_| AudioVaeError::ShapeOverflow)?,
                usize::try_from(shape[0]).map_err(|_| AudioVaeError::ShapeOverflow)?,
                bias.is_some(),
                false,
            )?
        } else {
            let (input_channels, output_channels, geometry) =
                dense_audio_convolution_geometry(profile, prefix, shape)?;
            let kernel = shape[2..]
                .iter()
                .map(|value| usize::try_from(*value).map_err(|_| AudioVaeError::ShapeOverflow))
                .collect::<Result<Vec<_>, _>>()?;
            NativeModule::convolution(
                weight_name,
                input_channels,
                output_channels,
                kernel,
                bias.is_some(),
                geometry,
                false,
            )?
        };
        module.load_dense_parameters(weight, bias)?;
        children.push(module);
    }
    children.extend(
        state
            .into_iter()
            .map(|(name, tensor)| NativeModule::buffer(name, tensor))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let mut module = NativeModule::module_dict(format!("audio-vae:{profile:?}"), children)?;
    module.materialize_execution_state_with_context(
        backend,
        execution_dtype,
        execution_device,
        context,
    )?;
    Ok(module)
}

fn build_audio_module(
    architecture: &NativeAudioVaeArchitecture,
    state: BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    execution_dtype: DType,
    execution_device: sim_tensor::DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, AudioVaeError> {
    if matches!(
        architecture.profile(),
        VaeKernelProfile::AudioOobleck44KhzV1 | VaeKernelProfile::AudioOobleck48KhzV1
    ) {
        return build_oobleck_module(
            architecture.profile(),
            state,
            backend,
            execution_dtype,
            execution_device,
            context,
        );
    }
    if architecture.profile() == &VaeKernelProfile::MmAudio16KhzV1 {
        return build_mmaudio_module(
            architecture.profile(),
            state,
            backend,
            execution_dtype,
            execution_device,
            context,
        );
    }
    if matches!(
        architecture.profile(),
        VaeKernelProfile::MusicDcaeV1
            | VaeKernelProfile::LtxAudioV1
            | VaeKernelProfile::StableAudio3DeepV1
            | VaeKernelProfile::StableAudio3ShallowV1
    ) {
        return build_dense_audio_module(
            architecture.profile(),
            state,
            backend,
            execution_dtype,
            execution_device,
            context,
        );
    }
    let children = state
        .into_iter()
        .map(|(name, tensor)| NativeModule::buffer(name, tensor))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NativeModule::module_dict(
        format!("audio-vae:{:?}", architecture.profile()),
        children,
    )?)
}

#[cfg(test)]
fn checked_ceil_ratio(value: usize, numerator: u64, denominator: u64) -> Result<usize, VaeError> {
    let value = u64::try_from(value).map_err(|_| VaeError::ShapeOverflow)?;
    let scaled = value
        .checked_mul(denominator)
        .and_then(|value| value.checked_add(numerator - 1))
        .ok_or(VaeError::ShapeOverflow)?;
    usize::try_from(scaled / numerator).map_err(|_| VaeError::ShapeOverflow)
}

#[cfg(test)]
fn checked_expand_ratio(value: usize, numerator: u64, denominator: u64) -> Result<usize, VaeError> {
    let value = u64::try_from(value).map_err(|_| VaeError::ShapeOverflow)?;
    usize::try_from(
        value
            .checked_mul(numerator)
            .ok_or(VaeError::ShapeOverflow)?
            / denominator,
    )
    .map_err(|_| VaeError::ShapeOverflow)
}

fn unary_audio_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    operation: UnaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (output, event) = backend.unary(operation, input, input.descriptor().clone(), context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn binary_audio_tensor(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    operation: BinaryOperation,
    output: TensorDescriptor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (output, event) = backend.binary(operation, left, right, output, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn scalar_audio_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    operation: BinaryOperation,
    scalar: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let (output, event) = backend.binary_scalar(
        operation,
        input,
        Scalar::Float(scalar),
        ScalarSide::Right,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn clamp_audio_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    minimum: f64,
    maximum: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let minimum = scalar_audio_tensor(backend, input, BinaryOperation::Maximum, minimum, context)?;
    scalar_audio_tensor(
        backend,
        &minimum,
        BinaryOperation::Minimum,
        maximum,
        context,
    )
}

fn mmaudio_channel_normalize(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape().to_vec();
    if shape.len() < 2 || shape[1] == 0 {
        return Err(VaeError::InvalidShape {
            expected: vec![0, 1, 0],
            actual: shape.to_vec(),
        });
    }
    let squared = binary_audio_tensor(
        backend,
        input,
        input,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let mut norm_shape = shape.to_vec();
    norm_shape[1] = 1;
    let descriptor = TensorDescriptor::contiguous(
        norm_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (sum, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Sum,
            dimensions: vec![1],
            keep_dimensions: true,
            accumulation_dtype: Some(DType::F32),
            correction: 0,
        },
        &squared,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let norm = unary_audio_tensor(backend, &sum, UnaryOperation::SquareRoot, context)?;
    let epsilon = 1.0e-4_f64 / (shape[1] as f64).sqrt();
    let norm = scalar_audio_tensor(backend, &norm, BinaryOperation::Add, epsilon, context)?;
    binary_audio_tensor(
        backend,
        input,
        &norm,
        BinaryOperation::Divide,
        input.descriptor().clone(),
        context,
    )
}

fn mmaudio_nonlinearity(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let activated = silu_tensor(backend, input, context)?;
    affine_tensor(backend, &activated, 1.0 / 0.596, 0.0, context)
}

fn mmaudio_mp_sum(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let denominator = (0.7_f64.powi(2) + 0.3_f64.powi(2)).sqrt();
    let left = scalar_audio_tensor(
        backend,
        left,
        BinaryOperation::Multiply,
        0.7 / denominator,
        context,
    )?;
    let right = scalar_audio_tensor(
        backend,
        right,
        BinaryOperation::Multiply,
        0.3 / denominator,
        context,
    )?;
    add_tensor(backend, &left, &right, context)
}

fn mmaudio_residual_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let normalized = mmaudio_channel_normalize(backend, input, context)?;
    let hidden = mmaudio_nonlinearity(backend, &normalized, context)?;
    let hidden = execute_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv1.weight"),
        context,
    )?;
    let hidden = mmaudio_nonlinearity(backend, &hidden, context)?;
    let hidden = execute_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2.weight"),
        context,
    )?;
    let shortcut_name = format!("{prefix}.nin_shortcut.weight");
    let shortcut = if find_module(module, &shortcut_name).is_some() {
        execute_convolution(module, backend, &normalized, &shortcut_name, context)?
    } else {
        normalized
    };
    mmaudio_mp_sum(backend, &shortcut, &hidden, context)
}

fn mmaudio_attention(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let qkv = execute_convolution(
        module,
        backend,
        input,
        &format!("{prefix}.qkv.weight"),
        context,
    )?;
    let shape = input.descriptor().shape();
    if shape.len() != 3 || qkv.descriptor().shape() != [shape[0], shape[1] * 3, shape[2]] {
        return Err(VaeError::InvalidShape {
            expected: vec![shape[0], shape[1] * 3, shape[2]],
            actual: qkv.descriptor().shape().to_vec(),
        });
    }
    let qkv = reshape_read_only(&qkv, vec![shape[0], shape[1], 3, shape[2]])?;
    let qkv = mmaudio_channel_normalize(backend, &qkv, context)?;
    let split = |index: i64| -> Result<Tensor, VaeError> {
        let tensor = qkv.narrow_read_only(2, index, 1)?;
        reshape_read_only(&tensor, shape.to_vec())
    };
    let query = split(0)?;
    let key = split(1)?;
    let value = split(2)?;
    let attended = spatial_attention_from_qkv(backend, input, &query, &key, &value, context)?;
    let projected = execute_convolution(
        module,
        backend,
        &attended,
        &format!("{prefix}.proj_out.weight"),
        context,
    )?;
    mmaudio_mp_sum(backend, input, &projected, context)
}

fn mmaudio_average_pool_2(
    cpu_backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[2] < 2 {
        return Err(VaeError::InvalidShape {
            expected: vec![0, 0, 2],
            actual: shape.to_vec(),
        });
    }
    let batch = usize::try_from(shape[0]).map_err(|_| VaeError::ShapeOverflow)?;
    let channels = usize::try_from(shape[1]).map_err(|_| VaeError::ShapeOverflow)?;
    let samples = usize::try_from(shape[2]).map_err(|_| VaeError::ShapeOverflow)?;
    let output_samples = samples / 2;
    let values = tensor_to_f32_with_backend_exact_native(cpu_backend, input, context)
        .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let mut output = vec![0.0_f32; batch * channels * output_samples];
    for batch_index in 0..batch {
        for channel in 0..channels {
            for output_index in 0..output_samples {
                if output_index % 1_024 == 0 {
                    context.check()?;
                }
                let source = (batch_index * channels + channel) * samples + output_index * 2;
                let destination =
                    (batch_index * channels + channel) * output_samples + output_index;
                output[destination] = (values[source] + values[source + 1]) * 0.5;
            }
        }
    }
    tensor_from_f32_with_backend_exact_native(
        cpu_backend,
        &[shape[0], shape[1], output_samples as u64],
        &output,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))
}

fn audio_buffer<'a>(module: &'a NativeModule, name: &str) -> Result<&'a Tensor, VaeError> {
    find_module(module, name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing audio checkpoint buffer {name}"
            )))
        })
}

fn mmaudio_apply_statistics(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    inverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] != 80 {
        return Err(VaeError::InvalidShape {
            expected: vec![shape.first().copied().unwrap_or(0), 80, 0],
            actual: shape.to_vec(),
        });
    }
    let reshape_statistic = |name: &str| -> Result<Tensor, VaeError> {
        let statistic = audio_buffer(module, name)?;
        match statistic.descriptor().shape() {
            [80] => reshape_read_only(statistic, vec![1, 80, 1]),
            [1, 80, 1] => Ok(statistic.clone()),
            actual => Err(VaeError::InvalidShape {
                expected: vec![1, 80, 1],
                actual: actual.to_vec(),
            }),
        }
    };
    let mean = reshape_statistic("vae.data_mean")?;
    let standard_deviation = reshape_statistic("vae.data_std")?;
    if inverse {
        let scaled = binary_audio_tensor(
            backend,
            input,
            &standard_deviation,
            BinaryOperation::Multiply,
            input.descriptor().clone(),
            context,
        )?;
        binary_audio_tensor(
            backend,
            &scaled,
            &mean,
            BinaryOperation::Add,
            input.descriptor().clone(),
            context,
        )
    } else {
        let centered = binary_audio_tensor(
            backend,
            input,
            &mean,
            BinaryOperation::Subtract,
            input.descriptor().clone(),
            context,
        )?;
        binary_audio_tensor(
            backend,
            &centered,
            &standard_deviation,
            BinaryOperation::Divide,
            input.descriptor().clone(),
            context,
        )
    }
}

fn mmaudio_apply_gain(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let gain = audio_buffer(module, name)?;
    if !matches!(gain.descriptor().shape(), [] | [1]) {
        return Err(VaeError::InvalidShape {
            expected: Vec::new(),
            actual: gain.descriptor().shape().to_vec(),
        });
    }
    let gain = scalar_audio_tensor(backend, gain, BinaryOperation::Add, 1.0, context)?;
    binary_audio_tensor(
        backend,
        input,
        &gain,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )
}

fn mmaudio_vae_encode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: &CpuBackend,
    mel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mel = mmaudio_apply_statistics(module, backend, mel, false, context)?;
    let mut hidden =
        execute_convolution(module, backend, &mel, "vae.encoder.conv_in.weight", context)?;
    for level in 0..3 {
        for block in 0..2 {
            hidden = mmaudio_residual_block(
                module,
                backend,
                &hidden,
                &format!("vae.encoder.down.{level}.block.{block}"),
                context,
            )?;
            hidden = clamp_audio_tensor(backend, &hidden, -256.0, 256.0, context)?;
        }
        if level == 0 {
            hidden = execute_convolution(
                module,
                backend,
                &hidden,
                "vae.encoder.down.0.downsample.conv1.weight",
                context,
            )?;
            hidden = mmaudio_average_pool_2(cpu_backend, &hidden, context)?;
            hidden = execute_convolution(
                module,
                backend,
                &hidden,
                "vae.encoder.down.0.downsample.conv2.weight",
                context,
            )?;
        }
    }
    hidden = mmaudio_residual_block(module, backend, &hidden, "vae.encoder.mid.block_1", context)?;
    hidden = mmaudio_attention(module, backend, &hidden, "vae.encoder.mid.attn_1", context)?;
    hidden = mmaudio_residual_block(module, backend, &hidden, "vae.encoder.mid.block_2", context)?;
    hidden = clamp_audio_tensor(backend, &hidden, -256.0, 256.0, context)?;
    hidden = mmaudio_nonlinearity(backend, &hidden, context)?;
    hidden = execute_convolution(
        module,
        backend,
        &hidden,
        "vae.encoder.conv_out.weight",
        context,
    )?;
    hidden = mmaudio_apply_gain(
        module,
        backend,
        &hidden,
        "vae.encoder.learnable_gain",
        context,
    )?;
    let shape = hidden.descriptor().shape();
    if shape.len() != 3 || shape[1] != 40 {
        return Err(VaeError::InvalidShape {
            expected: vec![shape.first().copied().unwrap_or(0), 40, 0],
            actual: shape.to_vec(),
        });
    }
    let mean = hidden.narrow_read_only(1, 0, 20)?;
    let descriptor = TensorDescriptor::contiguous(
        mean.descriptor().shape().to_vec(),
        mean.descriptor().dtype(),
        mean.descriptor().device(),
        context.stream,
    )?;
    let (mean, event) = backend.copy(&mean, descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(mean)
}

fn mmaudio_nearest_upsample_2(
    cpu_backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[2] == 0 {
        return Err(VaeError::InvalidShape {
            expected: vec![0, 0, 1],
            actual: shape.to_vec(),
        });
    }
    let values = tensor_to_f32_with_backend_exact_native(cpu_backend, input, context)
        .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let mut output = Vec::with_capacity(values.len() * 2);
    for (index, value) in values.into_iter().enumerate() {
        if index % 1_024 == 0 {
            context.check()?;
        }
        output.extend([value, value]);
    }
    tensor_from_f32_with_backend_exact_native(
        cpu_backend,
        &[shape[0], shape[1], shape[2] * 2],
        &output,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))
}

fn mmaudio_vae_decode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: &CpuBackend,
    latent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = execute_convolution(
        module,
        backend,
        latent,
        "vae.decoder.conv_in.weight",
        context,
    )?;
    hidden = mmaudio_residual_block(module, backend, &hidden, "vae.decoder.mid.block_1", context)?;
    hidden = mmaudio_attention(module, backend, &hidden, "vae.decoder.mid.attn_1", context)?;
    hidden = mmaudio_residual_block(module, backend, &hidden, "vae.decoder.mid.block_2", context)?;
    hidden = clamp_audio_tensor(backend, &hidden, -256.0, 256.0, context)?;
    for level in (0..3).rev() {
        for block in 0..3 {
            hidden = mmaudio_residual_block(
                module,
                backend,
                &hidden,
                &format!("vae.decoder.up.{level}.block.{block}"),
                context,
            )?;
            hidden = clamp_audio_tensor(backend, &hidden, -256.0, 256.0, context)?;
        }
        if level == 1 {
            hidden = mmaudio_nearest_upsample_2(cpu_backend, &hidden, context)?;
            hidden = execute_convolution(
                module,
                backend,
                &hidden,
                "vae.decoder.up.1.upsample.conv.weight",
                context,
            )?;
        }
    }
    hidden = mmaudio_nonlinearity(backend, &hidden, context)?;
    hidden = execute_convolution(
        module,
        backend,
        &hidden,
        "vae.decoder.conv_out.weight",
        context,
    )?;
    hidden = mmaudio_apply_gain(
        module,
        backend,
        &hidden,
        "vae.decoder.learnable_gain",
        context,
    )?;
    mmaudio_apply_statistics(module, backend, &hidden, true, context)
}

fn mmaudio_alias_free_activation(
    module: &NativeModule,
    cpu_backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let alpha = tensor_to_f32_with_backend_exact_native(
        cpu_backend,
        audio_buffer(module, &format!("{prefix}.act.alpha"))?,
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let beta = tensor_to_f32_with_backend_exact_native(
        cpu_backend,
        audio_buffer(module, &format!("{prefix}.act.beta"))?,
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let activation = alias_free_activation_1d_exact_native(
        PeriodicActivation::SnakeBeta {
            alpha,
            beta,
            logscale: true,
        },
        2,
        2,
        12,
        12,
        context.cancellation,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    activation
        .forward_with_context(cpu_backend, input, context)
        .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))
}

fn mmaudio_bigvgan_residual_block(
    module: &NativeModule,
    cpu_backend: &CpuBackend,
    input: &Tensor,
    block: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = input.clone();
    for layer in 0..3 {
        let activated = mmaudio_alias_free_activation(
            module,
            cpu_backend,
            &hidden,
            &format!("vocoder.resblocks.{block}.activations.{}", layer * 2),
            context,
        )?;
        let convolved = execute_convolution(
            module,
            cpu_backend,
            &activated,
            &format!("vocoder.resblocks.{block}.convs1.{layer}.weight"),
            context,
        )?;
        let activated = mmaudio_alias_free_activation(
            module,
            cpu_backend,
            &convolved,
            &format!("vocoder.resblocks.{block}.activations.{}", layer * 2 + 1),
            context,
        )?;
        let convolved = execute_convolution(
            module,
            cpu_backend,
            &activated,
            &format!("vocoder.resblocks.{block}.convs2.{layer}.weight"),
            context,
        )?;
        hidden = add_tensor(cpu_backend, &hidden, &convolved, context)?;
    }
    Ok(hidden)
}

fn mmaudio_vocoder(
    module: &NativeModule,
    cpu_backend: &CpuBackend,
    mel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden =
        execute_convolution(module, cpu_backend, mel, "vocoder.conv_pre.weight", context)?;
    for stage in 0..6 {
        hidden = execute_convolution(
            module,
            cpu_backend,
            &hidden,
            &format!("vocoder.ups.{stage}.0.weight"),
            context,
        )?;
        let mut aggregate = None;
        for kernel in 0..3 {
            let block = mmaudio_bigvgan_residual_block(
                module,
                cpu_backend,
                &hidden,
                stage * 3 + kernel,
                context,
            )?;
            aggregate = Some(match aggregate {
                Some(current) => add_tensor(cpu_backend, &current, &block, context)?,
                None => block,
            });
        }
        hidden = scalar_audio_tensor(
            cpu_backend,
            &aggregate.ok_or(VaeError::ShapeOverflow)?,
            BinaryOperation::Divide,
            3.0,
            context,
        )?;
    }
    hidden = mmaudio_alias_free_activation(
        module,
        cpu_backend,
        &hidden,
        "vocoder.activation_post",
        context,
    )?;
    hidden = execute_convolution(
        module,
        cpu_backend,
        &hidden,
        "vocoder.conv_post.weight",
        context,
    )?;
    unary_audio_tensor(
        cpu_backend,
        &hidden,
        UnaryOperation::HyperbolicTangent,
        context,
    )
}

fn mmaudio_decode_waveform(
    module: &NativeModule,
    cpu_backend: &CpuBackend,
    latent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mel = mmaudio_vae_decode(module, cpu_backend, cpu_backend, latent, context)?;
    let waveform = mmaudio_vocoder(module, cpu_backend, &mel, context)?;
    let waveform = resample_with_context_exact_native(
        cpu_backend,
        &waveform,
        NativeResampleConfiguration::torchaudio_default(16_000, 44_100),
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    prepare_pixel_channels(
        cpu_backend,
        &waveform,
        2,
        &VaeKernelProfile::MmAudio16KhzV1,
        context,
    )
}

fn audio_reflect_pad_last(
    cpu_backend: &CpuBackend,
    input: &Tensor,
    padding: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let Some(&sample_extent) = shape.last() else {
        return Err(VaeError::InvalidShape {
            expected: vec![padding as u64 + 1],
            actual: shape.to_vec(),
        });
    };
    if sample_extent <= padding as u64 {
        return Err(VaeError::InvalidShape {
            expected: vec![padding as u64 + 1],
            actual: shape.to_vec(),
        });
    }
    let samples = usize::try_from(sample_extent).map_err(|_| VaeError::ShapeOverflow)?;
    let leading = shape[..shape.len() - 1]
        .iter()
        .try_fold(1_u64, |product, extent| product.checked_mul(*extent))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(VaeError::ShapeOverflow)?;
    let output_samples = samples
        .checked_add(padding * 2)
        .ok_or(VaeError::ShapeOverflow)?;
    let values = tensor_to_f32_with_backend_exact_native(cpu_backend, input, context)
        .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let mut output = vec![0.0_f32; leading * output_samples];
    for leading_index in 0..leading {
        for output_index in 0..output_samples {
            if output_index % 1_024 == 0 {
                context.check()?;
            }
            let source_index = if output_index < padding {
                padding - output_index
            } else if output_index >= padding + samples {
                2 * samples + padding - output_index - 2
            } else {
                output_index - padding
            };
            output[leading_index * output_samples + output_index] =
                values[leading_index * samples + source_index];
        }
    }
    let mut output_shape = shape.to_vec();
    *output_shape.last_mut().ok_or(VaeError::ShapeOverflow)? = output_samples as u64;
    tensor_from_f32_with_backend_exact_native(
        cpu_backend,
        &output_shape,
        &output,
        DType::F32,
        sim_tensor::DeviceId::CPU,
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))
}

#[allow(clippy::too_many_arguments)]
fn loaded_mel_spectrogram(
    cpu_backend: &CpuBackend,
    waveform: &Tensor,
    window: &Tensor,
    mel_basis: &Tensor,
    n_fft: u64,
    hop_length: u64,
    magnitude_epsilon: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let waveform_shape = waveform.descriptor().shape();
    let frequency_bins = n_fft / 2 + 1;
    let basis_shape = mel_basis.descriptor().shape();
    if waveform_shape.len() < 2
        || window.descriptor().shape() != [n_fft]
        || basis_shape.len() != 2
        || basis_shape[1] != frequency_bins
    {
        return Err(VaeError::ShapeOverflow);
    }
    let flattened_batch = waveform_shape[..waveform_shape.len() - 1]
        .iter()
        .try_fold(1_u64, |product, extent| {
            product.checked_mul(*extent).ok_or(VaeError::ShapeOverflow)
        })?;
    let waveform = contiguous_copy(cpu_backend, waveform, context)?;
    let waveform = reshape_read_only(
        &waveform,
        vec![
            flattened_batch,
            *waveform_shape.last().ok_or(VaeError::ShapeOverflow)?,
        ],
    )?;
    let n_fft = usize::try_from(n_fft)?;
    let spectrum = stft_with_context_exact_native(
        cpu_backend,
        &waveform,
        n_fft,
        Some(usize::try_from(hop_length)?),
        Some(n_fft),
        Some(window),
        false,
        false,
        true,
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let spectrum_shape = spectrum.descriptor().shape();
    if spectrum_shape.len() != 3
        || spectrum_shape[0] != flattened_batch
        || spectrum_shape[1] != frequency_bins
    {
        return Err(VaeError::ShapeOverflow);
    }
    let components = view_as_real_exact_native(&spectrum, context.cancellation)
        .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let squared = binary_audio_tensor(
        cpu_backend,
        &components,
        &components,
        BinaryOperation::Multiply,
        components.descriptor().clone(),
        context,
    )?;
    let sum_descriptor = TensorDescriptor::contiguous(
        vec![flattened_batch, frequency_bins, spectrum_shape[2], 1],
        DType::F32,
        DeviceId::CPU,
        context.stream,
    )?;
    let (squared, event) = cpu_backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Sum,
            dimensions: vec![3],
            keep_dimensions: true,
            accumulation_dtype: Some(DType::F32),
            correction: 0,
        },
        &squared,
        sum_descriptor,
        context,
    )?;
    cpu_backend.wait_event(event, context)?;
    let squared = scalar_audio_tensor(
        cpu_backend,
        &squared,
        BinaryOperation::Add,
        magnitude_epsilon,
        context,
    )?;
    let magnitude = unary_audio_tensor(cpu_backend, &squared, UnaryOperation::SquareRoot, context)?;
    let magnitude = reshape_read_only(
        &magnitude,
        vec![flattened_batch, frequency_bins, spectrum_shape[2]],
    )?;
    let basis_descriptor = TensorDescriptor::contiguous(
        vec![flattened_batch, basis_shape[0], frequency_bins],
        DType::F32,
        DeviceId::CPU,
        context.stream,
    )?;
    let (zeros, event) = cpu_backend.fill(Scalar::Float(0.0), basis_descriptor, context)?;
    cpu_backend.wait_event(event, context)?;
    let basis = binary_audio_tensor(
        cpu_backend,
        &zeros,
        mel_basis,
        BinaryOperation::Add,
        zeros.descriptor().clone(),
        context,
    )?;
    let mel_descriptor = TensorDescriptor::contiguous(
        vec![flattened_batch, basis_shape[0], spectrum_shape[2]],
        DType::F32,
        DeviceId::CPU,
        context.stream,
    )?;
    let (mel, event) = cpu_backend.linear_algebra(
        LinearAlgebraOperation::BatchMatrixMultiply,
        &[basis, magnitude],
        mel_descriptor,
        context,
    )?;
    cpu_backend.wait_event(event, context)?;
    let mut output_shape = waveform_shape[..waveform_shape.len() - 1].to_vec();
    output_shape.extend([basis_shape[0], spectrum_shape[2]]);
    reshape_read_only(&mel, output_shape)
}

fn mmaudio_waveform_to_mel(
    module: &NativeModule,
    cpu_backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] == 0 || shape[2] == 0 {
        return Err(VaeError::InvalidShape {
            expected: vec![0, 2, 1],
            actual: shape.to_vec(),
        });
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[2]],
        DType::F32,
        sim_tensor::DeviceId::CPU,
        context.stream,
    )?;
    let (mono, event) = cpu_backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Mean,
            dimensions: vec![1],
            keep_dimensions: false,
            accumulation_dtype: Some(DType::F32),
            correction: 0,
        },
        input,
        descriptor,
        context,
    )?;
    cpu_backend.wait_event(event, context)?;
    let resampled = resample_with_context_exact_native(
        cpu_backend,
        &mono,
        NativeResampleConfiguration::torchaudio_default(44_100, 16_000),
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let resampled = clamp_audio_tensor(cpu_backend, &resampled, -1.0, 1.0, context)?;
    let padded = audio_reflect_pad_last(cpu_backend, &resampled, 384, context)?;
    let window = audio_buffer(module, "mel_converter.hann_window")?;
    let mel_basis = audio_buffer(module, "mel_converter.mel_basis")?;
    if window.descriptor().shape() != [1_024] || mel_basis.descriptor().shape() != [80, 513] {
        return Err(VaeError::ShapeOverflow);
    }
    let mel = loaded_mel_spectrogram(
        cpu_backend,
        &padded,
        window,
        mel_basis,
        1_024,
        256,
        1.0e-9,
        context,
    )?;
    let mel = scalar_audio_tensor(cpu_backend, &mel, BinaryOperation::Maximum, 1.0e-5, context)?;
    let mel = unary_audio_tensor(cpu_backend, &mel, UnaryOperation::NaturalLogarithm, context)?;
    scalar_audio_tensor(
        cpu_backend,
        &mel,
        BinaryOperation::Divide,
        10.0_f64.ln(),
        context,
    )
}

fn music_waveform_to_mel_image(
    module: &NativeModule,
    cpu_backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] != 2 || shape[2] == 0 {
        return Err(VaeError::InvalidShape {
            expected: vec![shape.first().copied().unwrap_or(0), 2, 1],
            actual: shape.to_vec(),
        });
    }
    let remainder = shape[2] % 4_096;
    let padded = if remainder == 0 {
        input.clone()
    } else {
        let right = 4_096 - remainder;
        let (padded, event) = cpu_backend.constant_pad(
            input,
            &[0, i64::try_from(right)?, 0, 0, 0, 0],
            Some(sim_tensor::DecodedScalar::Real(0.0)),
            context,
        )?;
        cpu_backend.wait_event(event, context)?;
        padded
    };
    let padded = audio_reflect_pad_last(cpu_backend, &padded, 768, context)?;
    let window = audio_buffer(module, "vocoder.mel_transform.spectrogram.window")?;
    let mel_basis = audio_buffer(module, "vocoder.mel_transform.mel_scale.fb")?;
    if window.descriptor().shape() != [2_048] || mel_basis.descriptor().shape() != [1_025, 128] {
        return Err(VaeError::ShapeOverflow);
    }
    let mel_basis = permute_read_only(mel_basis, &[1, 0])?;
    let mel = loaded_mel_spectrogram(
        cpu_backend,
        &padded,
        window,
        &mel_basis,
        2_048,
        512,
        1.0e-6,
        context,
    )?;
    let mel = scalar_audio_tensor(cpu_backend, &mel, BinaryOperation::Maximum, 1.0e-5, context)?;
    let mel = unary_audio_tensor(cpu_backend, &mel, UnaryOperation::NaturalLogarithm, context)?;
    affine_tensor(cpu_backend, &mel, 1.0 / 7.0, 4.0 / 7.0, context)
}

fn concatenate_audio_dimension(
    backend: &dyn TensorBackend,
    inputs: &[Tensor],
    dimension: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let first = inputs.first().ok_or(VaeError::ShapeOverflow)?;
    let rank = first.descriptor().rank();
    if dimension >= rank {
        return Err(VaeError::ShapeOverflow);
    }
    let mut output_shape = first.descriptor().shape().to_vec();
    output_shape[dimension] = 0;
    for input in inputs {
        let shape = input.descriptor().shape();
        if shape.len() != rank
            || input.descriptor().dtype() != first.descriptor().dtype()
            || input.descriptor().device() != first.descriptor().device()
            || shape
                .iter()
                .enumerate()
                .any(|(axis, extent)| axis != dimension && *extent != output_shape[axis])
        {
            return Err(VaeError::ShapeOverflow);
        }
        output_shape[dimension] = output_shape[dimension]
            .checked_add(shape[dimension])
            .ok_or(VaeError::ShapeOverflow)?;
    }
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
        let mut offsets = vec![0_u64; rank];
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

fn music_linear_channels_last(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let linear = find_module(module, name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing MusicDCAE linear module {name}"
        )))
    })?;
    let (weight, bias) = linear.dense_parameters()?;
    let shape = input.descriptor().shape();
    let weight_shape = weight.descriptor().shape();
    if !matches!(shape.len(), 3 | 4) || weight_shape.len() != 2 || weight_shape[1] != shape[1] {
        return Err(VaeError::ShapeOverflow);
    }
    let permutation: &[usize] = if shape.len() == 3 {
        &[0, 2, 1]
    } else {
        &[0, 2, 3, 1]
    };
    let channels_last = permute_read_only(input, permutation)?;
    let channels_last = contiguous_copy(backend, &channels_last, context)?;
    let rows = shape[2..].iter().try_fold(shape[0], |rows, extent| {
        rows.checked_mul(*extent).ok_or(VaeError::ShapeOverflow)
    })?;
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
        binary_audio_tensor(
            backend,
            &output,
            &bias,
            BinaryOperation::Add,
            output.descriptor().clone(),
            context,
        )?
    } else {
        output
    };
    let mut output_shape = vec![shape[0]];
    output_shape.extend_from_slice(&shape[2..]);
    output_shape.push(weight_shape[0]);
    let output = reshape_read_only(&output, output_shape)?;
    let output_permutation: &[usize] = if shape.len() == 3 {
        &[0, 2, 1]
    } else {
        &[0, 3, 1, 2]
    };
    let output = permute_read_only(&output, output_permutation)?;
    contiguous_copy(backend, &output, context)
}

fn music_rms_norm(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[1] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let squared = binary_audio_tensor(
        backend,
        input,
        input,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], 1, shape[2], shape[3]],
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
    let mean = scalar_audio_tensor(backend, &mean, BinaryOperation::Add, 1.0e-5, context)?;
    let inverse = unary_audio_tensor(
        backend,
        &mean,
        UnaryOperation::ReciprocalSquareRoot,
        context,
    )?;
    let normalized = binary_audio_tensor(
        backend,
        input,
        &inverse,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let parameter = |suffix: &str| -> Result<Tensor, VaeError> {
        let name = format!("{prefix}.{suffix}");
        let tensor = find_module(module, &name)
            .and_then(NativeModule::registered_buffer)
            .ok_or_else(|| {
                VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                    "missing MusicDCAE normalization parameter {name}"
                )))
            })?;
        reshape_read_only(tensor, vec![1, shape[1], 1, 1])
    };
    let weight = parameter("weight")?;
    let bias = parameter("bias")?;
    let scaled = binary_audio_tensor(
        backend,
        &normalized,
        &weight,
        BinaryOperation::Multiply,
        normalized.descriptor().clone(),
        context,
    )?;
    binary_audio_tensor(
        backend,
        &scaled,
        &bias,
        BinaryOperation::Add,
        scaled.descriptor().clone(),
        context,
    )
}

fn music_dcae_residual(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = execute_convolution(
        module,
        backend,
        input,
        &format!("{prefix}.conv1.weight"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = execute_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2.weight"),
        context,
    )?;
    hidden = music_rms_norm(module, backend, &hidden, &format!("{prefix}.norm"), context)?;
    add_tensor(backend, &hidden, input, context)
}

fn music_dcae_downsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    output_channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let hidden = execute_convolution(module, backend, input, name, context)?;
    let residual = pixel_unshuffle(backend, input, 2, context)?;
    let residual = grouped_channel_mean(backend, &residual, output_channels, context)?;
    add_tensor(backend, &hidden, &residual, context)
}

fn music_dcae_upsample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    output_channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let upsampled = nearest_upsample_2x(backend, input, context)?;
    let hidden = execute_convolution(module, backend, &upsampled, name, context)?;
    let input_channels = input.descriptor().shape()[1];
    let repeats = output_channels
        .checked_mul(4)
        .and_then(|value| value.checked_div(input_channels))
        .filter(|value| *value > 0)
        .ok_or(VaeError::ShapeOverflow)?;
    let residual = repeat_channels_interleave(backend, input, repeats, context)?;
    let residual = pixel_shuffle(backend, &residual, 2, context)?;
    add_tensor(backend, &hidden, &residual, context)
}

fn music_batched_matrix_multiply(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    output_shape: Vec<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let left = contiguous_copy(backend, left, context)?;
    let right = contiguous_copy(backend, right, context)?;
    let left_shape = left.descriptor().shape();
    let right_shape = right.descriptor().shape();
    if left_shape.len() < 3
        || left_shape.len() != right_shape.len()
        || left_shape[..left_shape.len() - 2] != right_shape[..right_shape.len() - 2]
        || left_shape[left_shape.len() - 1] != right_shape[right_shape.len() - 2]
    {
        return Err(VaeError::ShapeOverflow);
    }
    let batch = left_shape[..left_shape.len() - 2]
        .iter()
        .try_fold(1_u64, |product, extent| {
            product.checked_mul(*extent).ok_or(VaeError::ShapeOverflow)
        })?;
    let rows = left_shape[left_shape.len() - 2];
    let contracted = left_shape[left_shape.len() - 1];
    let columns = right_shape[right_shape.len() - 1];
    let left = reshape_read_only(&left, vec![batch, rows, contracted])?;
    let right = reshape_read_only(&right, vec![batch, contracted, columns])?;
    let descriptor = TensorDescriptor::contiguous(
        vec![batch, rows, columns],
        left.descriptor().dtype(),
        left.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.linear_algebra(
        LinearAlgebraOperation::BatchMatrixMultiply,
        &[left, right],
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    reshape_read_only(&output, output_shape)
}

fn music_sana_attention(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape().to_vec();
    if shape.len() != 4 || !shape[1].is_multiple_of(32) {
        return Err(VaeError::ShapeOverflow);
    }
    let mut projected = Vec::with_capacity(3);
    for name in ["to_q", "to_k", "to_v"] {
        projected.push(music_linear_channels_last(
            module,
            backend,
            input,
            &format!("{prefix}.{name}.weight"),
            context,
        )?);
    }
    let qkv = concatenate_audio_dimension(backend, &projected, 1, context)?;
    let mut multiscale = execute_convolution(
        module,
        backend,
        &qkv,
        &format!("{prefix}.to_qkv_multiscale.0.proj_in.weight"),
        context,
    )?;
    multiscale = execute_convolution(
        module,
        backend,
        &multiscale,
        &format!("{prefix}.to_qkv_multiscale.0.proj_out.weight"),
        context,
    )?;
    let qkv = concatenate_audio_dimension(backend, &[qkv, multiscale], 1, context)?;
    let heads = shape[1]
        .checked_div(32)
        .and_then(|value| value.checked_mul(2))
        .ok_or(VaeError::ShapeOverflow)?;
    let tokens = shape[2]
        .checked_mul(shape[3])
        .ok_or(VaeError::ShapeOverflow)?;
    let qkv = reshape_read_only(&qkv, vec![shape[0], heads, 96, tokens])?;
    let query = narrow_contiguous(backend, &qkv, 2, 0, 32, context)?;
    let key = narrow_contiguous(backend, &qkv, 2, 32, 32, context)?;
    let value = narrow_contiguous(backend, &qkv, 2, 64, 32, context)?;
    let query = unary_audio_tensor(backend, &query, UnaryOperation::Relu, context)?;
    let key = unary_audio_tensor(backend, &key, UnaryOperation::Relu, context)?;
    let attended = if tokens > 32 {
        let descriptor = TensorDescriptor::contiguous(
            vec![shape[0], heads, 1, tokens],
            input.descriptor().dtype(),
            input.descriptor().device(),
            context.stream,
        )?;
        let (ones, event) = backend.fill(Scalar::Float(1.0), descriptor, context)?;
        backend.wait_event(event, context)?;
        let value = concatenate_audio_dimension(backend, &[value, ones], 2, context)?;
        let key_transposed = permute_read_only(&key, &[0, 1, 3, 2])?;
        let scores = music_batched_matrix_multiply(
            backend,
            &value,
            &key_transposed,
            vec![shape[0], heads, 33, 32],
            context,
        )?;
        let hidden = music_batched_matrix_multiply(
            backend,
            &scores,
            &query,
            vec![shape[0], heads, 33, tokens],
            context,
        )?;
        let numerator = narrow_contiguous(backend, &hidden, 2, 0, 32, context)?;
        let denominator = narrow_contiguous(backend, &hidden, 2, 32, 1, context)?;
        let denominator = scalar_audio_tensor(
            backend,
            &denominator,
            BinaryOperation::Add,
            1.0e-15,
            context,
        )?;
        binary_audio_tensor(
            backend,
            &numerator,
            &denominator,
            BinaryOperation::Divide,
            numerator.descriptor().clone(),
            context,
        )?
    } else {
        let key_transposed = permute_read_only(&key, &[0, 1, 3, 2])?;
        let scores = music_batched_matrix_multiply(
            backend,
            &key_transposed,
            &query,
            vec![shape[0], heads, tokens, tokens],
            context,
        )?;
        let denominator_descriptor = TensorDescriptor::contiguous(
            vec![shape[0], heads, 1, tokens],
            input.descriptor().dtype(),
            input.descriptor().device(),
            context.stream,
        )?;
        let (denominator, event) = backend.reduction(
            &ReductionSpec {
                operation: ReductionOperation::Sum,
                dimensions: vec![2],
                keep_dimensions: true,
                accumulation_dtype: Some(input.descriptor().dtype()),
                correction: 0,
            },
            &scores,
            denominator_descriptor,
            context,
        )?;
        backend.wait_event(event, context)?;
        let denominator = scalar_audio_tensor(
            backend,
            &denominator,
            BinaryOperation::Add,
            1.0e-15,
            context,
        )?;
        let scores = binary_audio_tensor(
            backend,
            &scores,
            &denominator,
            BinaryOperation::Divide,
            scores.descriptor().clone(),
            context,
        )?;
        music_batched_matrix_multiply(
            backend,
            &value,
            &scores,
            vec![shape[0], heads, 32, tokens],
            context,
        )?
    };
    let attended = reshape_read_only(&attended, vec![shape[0], shape[1] * 2, shape[2], shape[3]])?;
    let mut attended = music_linear_channels_last(
        module,
        backend,
        &attended,
        &format!("{prefix}.to_out.weight"),
        context,
    )?;
    attended = music_rms_norm(
        module,
        backend,
        &attended,
        &format!("{prefix}.norm_out"),
        context,
    )?;
    add_tensor(backend, &attended, input, context)
}

fn music_glumb(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = execute_convolution(
        module,
        backend,
        input,
        &format!("{prefix}.conv_inverted.weight"),
        context,
    )?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = execute_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv_depth.weight"),
        context,
    )?;
    let channels = hidden.descriptor().shape()[1] / 2;
    let values = narrow_contiguous(backend, &hidden, 1, 0, channels, context)?;
    let gate = narrow_contiguous(
        backend,
        &hidden,
        1,
        i64::try_from(channels)?,
        channels,
        context,
    )?;
    let gate = silu_tensor(backend, &gate, context)?;
    hidden = binary_audio_tensor(
        backend,
        &values,
        &gate,
        BinaryOperation::Multiply,
        values.descriptor().clone(),
        context,
    )?;
    hidden = execute_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv_point.weight"),
        context,
    )?;
    hidden = music_rms_norm(module, backend, &hidden, &format!("{prefix}.norm"), context)?;
    add_tensor(backend, &hidden, input, context)
}

fn music_efficient_vit(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let hidden = music_sana_attention(module, backend, input, &format!("{prefix}.attn"), context)?;
    music_glumb(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv_out"),
        context,
    )
}

fn music_dcae_encode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = execute_convolution(
        module,
        backend,
        input,
        "dcae.encoder.conv_in.weight",
        context,
    )?;
    for level in 0..4 {
        let layers = [2_usize, 2, 3, 3][level];
        for block in 0..layers {
            let prefix = format!("dcae.encoder.down_blocks.{level}.{block}");
            hidden = if level == 3 {
                music_efficient_vit(module, backend, &hidden, &prefix, context)?
            } else {
                music_dcae_residual(module, backend, &hidden, &prefix, context)?
            };
        }
        if level < 3 {
            hidden = music_dcae_downsample(
                module,
                backend,
                &hidden,
                &format!("dcae.encoder.down_blocks.{level}.{layers}.conv.weight"),
                [256_u64, 512, 1_024][level],
                context,
            )?;
        }
    }
    hidden = execute_convolution(
        module,
        backend,
        &hidden,
        "dcae.encoder.conv_out.weight",
        context,
    )?;
    affine_tensor(backend, &hidden, 0.1786, 1.9091 * 0.1786, context)
}

fn music_dcae_decode_mel(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let input = affine_tensor(backend, input, 1.0 / 0.1786, -1.9091, context)?;
    let input_channels = input.descriptor().shape()[1];
    if input_channels == 0 || !1_024_u64.is_multiple_of(input_channels) {
        return Err(VaeError::ShapeOverflow);
    }
    let mut hidden = execute_convolution(
        module,
        backend,
        &input,
        "dcae.decoder.conv_in.weight",
        context,
    )?;
    let residual = repeat_channels_interleave(backend, &input, 1_024 / input_channels, context)?;
    hidden = add_tensor(backend, &hidden, &residual, context)?;
    for level in (0..4).rev() {
        let mut child = 0_usize;
        if level < 3 {
            hidden = music_dcae_upsample(
                module,
                backend,
                &hidden,
                &format!("dcae.decoder.up_blocks.{level}.0.conv.weight"),
                [128_u64, 256, 512][level],
                context,
            )?;
            child = 1;
        }
        for block in 0..3 {
            let prefix = format!("dcae.decoder.up_blocks.{level}.{}", child + block);
            hidden = if level == 3 {
                music_efficient_vit(module, backend, &hidden, &prefix, context)?
            } else {
                music_dcae_residual(module, backend, &hidden, &prefix, context)?
            };
        }
    }
    hidden = music_rms_norm(module, backend, &hidden, "dcae.decoder.norm_out", context)?;
    hidden = unary_audio_tensor(backend, &hidden, UnaryOperation::Relu, context)?;
    hidden = execute_convolution(
        module,
        backend,
        &hidden,
        "dcae.decoder.conv_out.weight",
        context,
    )?;
    affine_tensor(backend, &hidden, 7.0, -11.0, context)
}

fn music_layer_norm_1d(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let reduced_descriptor = TensorDescriptor::contiguous(
        vec![shape[0], 1, shape[2]],
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
        input,
        reduced_descriptor.clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let centered = binary_audio_tensor(
        backend,
        input,
        &mean,
        BinaryOperation::Subtract,
        input.descriptor().clone(),
        context,
    )?;
    let squared = binary_audio_tensor(
        backend,
        &centered,
        &centered,
        BinaryOperation::Multiply,
        centered.descriptor().clone(),
        context,
    )?;
    let (variance, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Mean,
            dimensions: vec![1],
            keep_dimensions: true,
            accumulation_dtype: Some(input.descriptor().dtype()),
            correction: 0,
        },
        &squared,
        reduced_descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let variance = scalar_audio_tensor(backend, &variance, BinaryOperation::Add, 1.0e-6, context)?;
    let inverse = unary_audio_tensor(
        backend,
        &variance,
        UnaryOperation::ReciprocalSquareRoot,
        context,
    )?;
    let normalized = binary_audio_tensor(
        backend,
        &centered,
        &inverse,
        BinaryOperation::Multiply,
        centered.descriptor().clone(),
        context,
    )?;
    let parameter = |suffix: &str| -> Result<Tensor, VaeError> {
        let name = format!("{prefix}.{suffix}");
        let tensor = find_module(module, &name)
            .and_then(NativeModule::registered_buffer)
            .ok_or_else(|| {
                VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                    "missing ACE vocoder normalization parameter {name}"
                )))
            })?;
        reshape_read_only(tensor, vec![1, shape[1], 1])
    };
    let weight = parameter("weight")?;
    let bias = parameter("bias")?;
    let scaled = binary_audio_tensor(
        backend,
        &normalized,
        &weight,
        BinaryOperation::Multiply,
        normalized.descriptor().clone(),
        context,
    )?;
    binary_audio_tensor(
        backend,
        &scaled,
        &bias,
        BinaryOperation::Add,
        scaled.descriptor().clone(),
        context,
    )
}

fn music_gelu(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let argument = scalar_audio_tensor(
        backend,
        input,
        BinaryOperation::Divide,
        std::f64::consts::SQRT_2,
        context,
    )?;
    let absolute = unary_audio_tensor(backend, &argument, UnaryOperation::Absolute, context)?;
    let denominator = scalar_audio_tensor(
        backend,
        &absolute,
        BinaryOperation::Multiply,
        0.327_591_1,
        context,
    )?;
    let denominator =
        scalar_audio_tensor(backend, &denominator, BinaryOperation::Add, 1.0, context)?;
    let t = unary_audio_tensor(backend, &denominator, UnaryOperation::Reciprocal, context)?;
    let mut polynomial =
        scalar_audio_tensor(backend, &t, BinaryOperation::Multiply, 1.061_405_4, context)?;
    for coefficient in [-1.453_152_1, 1.421_413_8, -0.284_496_72, 0.254_829_6] {
        polynomial = scalar_audio_tensor(
            backend,
            &polynomial,
            BinaryOperation::Add,
            coefficient,
            context,
        )?;
        polynomial = binary_audio_tensor(
            backend,
            &polynomial,
            &t,
            BinaryOperation::Multiply,
            polynomial.descriptor().clone(),
            context,
        )?;
    }
    let squared = binary_audio_tensor(
        backend,
        &argument,
        &argument,
        BinaryOperation::Multiply,
        argument.descriptor().clone(),
        context,
    )?;
    let negative_squared = unary_audio_tensor(backend, &squared, UnaryOperation::Negate, context)?;
    let exponential = unary_audio_tensor(
        backend,
        &negative_squared,
        UnaryOperation::Exponential,
        context,
    )?;
    let tail = binary_audio_tensor(
        backend,
        &polynomial,
        &exponential,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let erf_absolute =
        unary_audio_tensor(backend, &tail, UnaryOperation::InvertUnitInterval, context)?;
    let sign = unary_audio_tensor(backend, &argument, UnaryOperation::Signum, context)?;
    let erf = binary_audio_tensor(
        backend,
        &erf_absolute,
        &sign,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let one_plus = scalar_audio_tensor(backend, &erf, BinaryOperation::Add, 1.0, context)?;
    let product = binary_audio_tensor(
        backend,
        input,
        &one_plus,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    scalar_audio_tensor(backend, &product, BinaryOperation::Multiply, 0.5, context)
}

fn music_replicate_pad_1d(
    backend: &dyn TensorBackend,
    input: &Tensor,
    padding: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[2] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let output_length = shape[2]
        .checked_add(padding.checked_mul(2).ok_or(VaeError::ShapeOverflow)?)
        .ok_or(VaeError::ShapeOverflow)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[1], output_length],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let first = narrow_contiguous(backend, input, 2, 0, 1, context)?;
    let last = narrow_contiguous(backend, input, 2, i64::try_from(shape[2] - 1)?, 1, context)?;
    for index in 0..padding {
        let (updated, event) =
            backend.replace_rectangular_slice(&output, &first, &[0, 0, index], context)?;
        backend.wait_event(event, context)?;
        output = updated;
        let destination = padding
            .checked_add(shape[2])
            .and_then(|value| value.checked_add(index))
            .ok_or(VaeError::ShapeOverflow)?;
        let (updated, event) =
            backend.replace_rectangular_slice(&output, &last, &[0, 0, destination], context)?;
        backend.wait_event(event, context)?;
        output = updated;
    }
    let (output, event) =
        backend.replace_rectangular_slice(&output, input, &[0, 0, padding], context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn music_convnext_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = execute_convolution(
        module,
        backend,
        input,
        &format!("{prefix}.dwconv.weight"),
        context,
    )?;
    hidden = music_layer_norm_1d(module, backend, &hidden, &format!("{prefix}.norm"), context)?;
    hidden = music_linear_channels_last(
        module,
        backend,
        &hidden,
        &format!("{prefix}.pwconv1.weight"),
        context,
    )?;
    hidden = music_gelu(backend, &hidden, context)?;
    hidden = music_linear_channels_last(
        module,
        backend,
        &hidden,
        &format!("{prefix}.pwconv2.weight"),
        context,
    )?;
    let gamma_name = format!("{prefix}.gamma");
    let gamma = find_module(module, &gamma_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing ACE vocoder layer scale {gamma_name}"
            )))
        })?;
    let gamma = reshape_read_only(gamma, vec![1, hidden.descriptor().shape()[1], 1])?;
    hidden = binary_audio_tensor(
        backend,
        &hidden,
        &gamma,
        BinaryOperation::Multiply,
        hidden.descriptor().clone(),
        context,
    )?;
    add_tensor(backend, input, &hidden, context)
}

fn music_vocoder_residual(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = input.clone();
    for layer in 0..3 {
        let residual = hidden.clone();
        hidden = silu_tensor(backend, &hidden, context)?;
        hidden = execute_convolution(
            module,
            backend,
            &hidden,
            &format!("{prefix}.convs1.{layer}"),
            context,
        )?;
        hidden = silu_tensor(backend, &hidden, context)?;
        hidden = execute_convolution(
            module,
            backend,
            &hidden,
            &format!("{prefix}.convs2.{layer}"),
            context,
        )?;
        hidden = add_tensor(backend, &hidden, &residual, context)?;
    }
    Ok(hidden)
}

fn music_vocoder_decode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    mel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = mel.descriptor().shape();
    if shape.len() != 4 || shape[1] != 2 || shape[2] != 128 {
        return Err(VaeError::ShapeOverflow);
    }
    let flattened = reshape_read_only(
        mel,
        vec![
            shape[0]
                .checked_mul(shape[1])
                .ok_or(VaeError::ShapeOverflow)?,
            shape[2],
            shape[3],
        ],
    )?;
    let padded = music_replicate_pad_1d(backend, &flattened, 3, context)?;
    let mut hidden = execute_convolution(
        module,
        backend,
        &padded,
        "vocoder.backbone.channel_layers.0.0.weight",
        context,
    )?;
    hidden = music_layer_norm_1d(
        module,
        backend,
        &hidden,
        "vocoder.backbone.channel_layers.0.1",
        context,
    )?;
    for stage in 0..4 {
        if stage > 0 {
            hidden = music_layer_norm_1d(
                module,
                backend,
                &hidden,
                &format!("vocoder.backbone.channel_layers.{stage}.0"),
                context,
            )?;
            hidden = execute_convolution(
                module,
                backend,
                &hidden,
                &format!("vocoder.backbone.channel_layers.{stage}.1.weight"),
                context,
            )?;
        }
        for block in 0..[3_usize, 3, 9, 3][stage] {
            hidden = music_convnext_block(
                module,
                backend,
                &hidden,
                &format!("vocoder.backbone.stages.{stage}.{block}"),
                context,
            )?;
        }
    }
    hidden = music_layer_norm_1d(module, backend, &hidden, "vocoder.backbone.norm", context)?;
    hidden = execute_convolution(module, backend, &hidden, "vocoder.head.conv_pre", context)?;
    for stage in 0..7 {
        hidden = silu_tensor(backend, &hidden, context)?;
        hidden = execute_convolution(
            module,
            backend,
            &hidden,
            &format!("vocoder.head.ups.{stage}"),
            context,
        )?;
        let mut sum = None;
        for kernel in 0..4 {
            let block = music_vocoder_residual(
                module,
                backend,
                &hidden,
                &format!("vocoder.head.resblocks.{}", stage * 4 + kernel),
                context,
            )?;
            sum = Some(if let Some(sum) = sum {
                add_tensor(backend, &sum, &block, context)?
            } else {
                block
            });
        }
        hidden = scalar_audio_tensor(
            backend,
            &sum.ok_or(VaeError::ShapeOverflow)?,
            BinaryOperation::Divide,
            4.0,
            context,
        )?;
    }
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = execute_convolution(module, backend, &hidden, "vocoder.head.conv_post", context)?;
    hidden = unary_audio_tensor(backend, &hidden, UnaryOperation::HyperbolicTangent, context)?;
    let output_length = hidden.descriptor().shape()[2];
    reshape_read_only(&hidden, vec![shape[0], shape[1], output_length])
}

fn sa3_linear_last(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let linear = find_module(module, name).ok_or_else(|| {
        VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
            "missing Stable Audio 3 linear module {name}"
        )))
    })?;
    let (weight, bias) = linear.dense_parameters()?;
    let shape = input.descriptor().shape();
    let weight_shape = weight.descriptor().shape();
    if shape.is_empty() || weight_shape.len() != 2 || shape.last() != Some(&weight_shape[1]) {
        return Err(VaeError::ShapeOverflow);
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1_u64, |rows, extent| {
            rows.checked_mul(*extent).ok_or(VaeError::ShapeOverflow)
        })?;
    let flattened = reshape_read_only(input, vec![rows, weight_shape[1]])?;
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
        binary_audio_tensor(
            backend,
            &output,
            &bias,
            BinaryOperation::Add,
            output.descriptor().clone(),
            context,
        )?
    } else {
        output
    };
    let mut output_shape = shape.to_vec();
    *output_shape.last_mut().ok_or(VaeError::ShapeOverflow)? = weight_shape[0];
    reshape_read_only(&output, output_shape)
}

fn sa3_dynamic_tanh(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let channels = *input
        .descriptor()
        .shape()
        .last()
        .ok_or(VaeError::ShapeOverflow)?;
    let buffer = |suffix: &str| -> Result<&Tensor, VaeError> {
        let name = format!("{prefix}.{suffix}");
        find_module(module, &name)
            .and_then(NativeModule::registered_buffer)
            .ok_or_else(|| {
                VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                    "missing Stable Audio 3 dynamic-tanh parameter {name}"
                )))
            })
    };
    let alpha = buffer("alpha")?;
    let gamma = reshape_read_only(buffer("gamma")?, vec![1, 1, channels])?;
    let beta = reshape_read_only(buffer("beta")?, vec![1, 1, channels])?;
    let scaled = binary_audio_tensor(
        backend,
        input,
        alpha,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let activated =
        unary_audio_tensor(backend, &scaled, UnaryOperation::HyperbolicTangent, context)?;
    let activated = binary_audio_tensor(
        backend,
        &activated,
        &gamma,
        BinaryOperation::Multiply,
        activated.descriptor().clone(),
        context,
    )?;
    binary_audio_tensor(
        backend,
        &activated,
        &beta,
        BinaryOperation::Add,
        activated.descriptor().clone(),
        context,
    )
}

fn sa3_apply_rotary(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[3] < 32 {
        return Err(VaeError::ShapeOverflow);
    }
    let inv = find_module(module, name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Stable Audio 3 rotary buffer {name}"
            )))
        })?;
    if inv.descriptor().shape() != [16] {
        return Err(VaeError::ShapeOverflow);
    }
    let positions = (0..shape[2]).map(|value| value as f32).collect::<Vec<_>>();
    let positions = tensor_from_f32_with_backend_exact_native(
        backend,
        &[shape[2], 1],
        &positions,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )
    .map_err(NativeOpsError::from)?;
    let inv = reshape_read_only(inv, vec![1, 16])?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[2], 16],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (frequencies, event) = backend.linear_algebra(
        LinearAlgebraOperation::MatrixMultiply,
        &[positions, inv],
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let frequencies =
        concatenate_audio_dimension(backend, &[frequencies.clone(), frequencies], 1, context)?;
    let cosine = unary_audio_tensor(backend, &frequencies, UnaryOperation::Cosine, context)?;
    let sine = unary_audio_tensor(backend, &frequencies, UnaryOperation::Sine, context)?;
    let cosine = reshape_read_only(&cosine, vec![1, 1, shape[2], 32])?;
    let sine = reshape_read_only(&sine, vec![1, 1, shape[2], 32])?;
    let rotating = narrow_contiguous(backend, input, 3, 0, 32, context)?;
    let first = narrow_contiguous(backend, &rotating, 3, 0, 16, context)?;
    let second = narrow_contiguous(backend, &rotating, 3, 16, 16, context)?;
    let negative_second = unary_audio_tensor(backend, &second, UnaryOperation::Negate, context)?;
    let rotated_half = concatenate_audio_dimension(backend, &[negative_second, first], 3, context)?;
    let cosine_part = binary_audio_tensor(
        backend,
        &rotating,
        &cosine,
        BinaryOperation::Multiply,
        rotating.descriptor().clone(),
        context,
    )?;
    let sine_part = binary_audio_tensor(
        backend,
        &rotated_half,
        &sine,
        BinaryOperation::Multiply,
        rotated_half.descriptor().clone(),
        context,
    )?;
    let rotating = add_tensor(backend, &cosine_part, &sine_part, context)?;
    if shape[3] == 32 {
        return Ok(rotating);
    }
    let pass = narrow_contiguous(backend, input, 3, 32, shape[3] - 32, context)?;
    concatenate_audio_dimension(backend, &[rotating, pass], 3, context)
}

fn sa3_attention_once(
    backend: &dyn TensorBackend,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    window: Option<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = query.descriptor().shape();
    if shape.len() != 4
        || key.descriptor().shape() != shape
        || value.descriptor().shape() != shape
        || shape.contains(&0)
    {
        return Err(VaeError::ShapeOverflow);
    }
    if let Some(window) = window {
        return sa3_sliding_window_attention(backend, query, key, value, window, context);
    }
    let key = permute_read_only(key, &[0, 1, 3, 2])?;
    let scores = music_batched_matrix_multiply(
        backend,
        query,
        &key,
        vec![shape[0], shape[1], shape[2], shape[2]],
        context,
    )?;
    let scores = scalar_audio_tensor(
        backend,
        &scores,
        BinaryOperation::Multiply,
        (shape[3] as f64).sqrt().recip(),
        context,
    )?;
    let scores = softmax_last_dimension(backend, &scores, context)?;
    music_batched_matrix_multiply(
        backend,
        &scores,
        value,
        vec![shape[0], shape[1], shape[2], shape[3]],
        context,
    )
}

const SA3_ATTENTION_QUERY_TILE: u64 = 64;

fn sa3_attention_tile_geometry(
    sequence: u64,
    window: u64,
    query_start: u64,
) -> Result<(u64, u64, u64), VaeError> {
    if sequence == 0 || query_start >= sequence {
        return Err(VaeError::ShapeOverflow);
    }
    let query_length = SA3_ATTENTION_QUERY_TILE.min(sequence - query_start);
    let key_start = query_start.saturating_sub(window);
    let key_end = query_start
        .checked_add(query_length)
        .and_then(|value| value.checked_add(window))
        .ok_or(VaeError::ShapeOverflow)?
        .min(sequence);
    Ok((query_length, key_start, key_end - key_start))
}

fn sa3_sliding_window_attention(
    backend: &dyn TensorBackend,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    window: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = query.descriptor().shape();
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        query.descriptor().dtype(),
        query.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let masked_value = query.descriptor().dtype().floating_point_info()?.minimum() / 4.0;
    let scale = (shape[3] as f64).sqrt().recip();
    let mut query_start = 0_u64;
    while query_start < shape[2] {
        context.check()?;
        let (query_length, key_start, key_length) =
            sa3_attention_tile_geometry(shape[2], window, query_start)?;
        let query_tile = narrow_contiguous(
            backend,
            query,
            2,
            i64::try_from(query_start)?,
            query_length,
            context,
        )?;
        let key_tile = narrow_contiguous(
            backend,
            key,
            2,
            i64::try_from(key_start)?,
            key_length,
            context,
        )?;
        let value_tile = narrow_contiguous(
            backend,
            value,
            2,
            i64::try_from(key_start)?,
            key_length,
            context,
        )?;
        let key_tile = permute_read_only(&key_tile, &[0, 1, 3, 2])?;
        let scores = music_batched_matrix_multiply(
            backend,
            &query_tile,
            &key_tile,
            vec![shape[0], shape[1], query_length, key_length],
            context,
        )?;
        let scores =
            scalar_audio_tensor(backend, &scores, BinaryOperation::Multiply, scale, context)?;
        let mask_elements = query_length
            .checked_mul(key_length)
            .ok_or(VaeError::ShapeOverflow)?;
        let mut mask = Vec::new();
        mask.try_reserve_exact(usize::try_from(mask_elements)?)
            .map_err(|error| VaeError::Allocation(error.to_string()))?;
        for query_offset in 0..query_length {
            let query_position = query_start
                .checked_add(query_offset)
                .ok_or(VaeError::ShapeOverflow)?;
            for key_offset in 0..key_length {
                let key_position = key_start
                    .checked_add(key_offset)
                    .ok_or(VaeError::ShapeOverflow)?;
                mask.push(if query_position.abs_diff(key_position) > window {
                    masked_value as f32
                } else {
                    0.0
                });
            }
        }
        let mask = tensor_from_f32_with_backend_exact_native(
            backend,
            &[1, 1, query_length, key_length],
            &mask,
            scores.descriptor().dtype(),
            scores.descriptor().device(),
            context,
        )
        .map_err(NativeOpsError::from)?;
        let scores = binary_audio_tensor(
            backend,
            &scores,
            &mask,
            BinaryOperation::Add,
            scores.descriptor().clone(),
            context,
        )?;
        let scores = softmax_last_dimension(backend, &scores, context)?;
        let attended = music_batched_matrix_multiply(
            backend,
            &scores,
            &value_tile,
            vec![shape[0], shape[1], query_length, shape[3]],
            context,
        )?;
        let (updated, event) = backend.replace_rectangular_slice(
            &output,
            &attended,
            &[0, 0, query_start, 0],
            context,
        )?;
        backend.wait_event(event, context)?;
        output = updated;
        query_start = query_start
            .checked_add(query_length)
            .ok_or(VaeError::ShapeOverflow)?;
    }
    Ok(output)
}

fn sa3_attention(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    window: Option<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || !shape[2].is_multiple_of(64) {
        return Err(VaeError::ShapeOverflow);
    }
    let qkv = sa3_linear_last(
        module,
        backend,
        input,
        &format!("{prefix}.to_qkv.weight"),
        context,
    )?;
    let channels = shape[2];
    let heads = channels / 64;
    let head = |offset: u64| -> Result<Tensor, VaeError> {
        let tensor = narrow_contiguous(
            backend,
            &qkv,
            2,
            i64::try_from(
                offset
                    .checked_mul(channels)
                    .ok_or(VaeError::ShapeOverflow)?,
            )?,
            channels,
            context,
        )?;
        let tensor = reshape_read_only(&tensor, vec![shape[0], shape[1], heads, 64])?;
        let tensor = permute_read_only(&tensor, &[0, 2, 1, 3])?;
        contiguous_copy(backend, &tensor, context)
    };
    let mut query = head(0)?;
    let mut key = head(1)?;
    let value = head(2)?;
    let mut query_difference = head(3)?;
    let mut key_difference = head(4)?;
    query = sa3_dynamic_tanh(
        module,
        backend,
        &query,
        &format!("{prefix}.q_norm"),
        context,
    )?;
    query_difference = sa3_dynamic_tanh(
        module,
        backend,
        &query_difference,
        &format!("{prefix}.q_norm"),
        context,
    )?;
    key = sa3_dynamic_tanh(module, backend, &key, &format!("{prefix}.k_norm"), context)?;
    key_difference = sa3_dynamic_tanh(
        module,
        backend,
        &key_difference,
        &format!("{prefix}.k_norm"),
        context,
    )?;
    let rope = format!("{prefix}.rope.inv_freq");
    query = sa3_apply_rotary(module, backend, &query, &rope, context)?;
    query_difference = sa3_apply_rotary(module, backend, &query_difference, &rope, context)?;
    key = sa3_apply_rotary(module, backend, &key, &rope, context)?;
    key_difference = sa3_apply_rotary(module, backend, &key_difference, &rope, context)?;
    let primary = sa3_attention_once(backend, &query, &key, &value, window, context)?;
    let difference = sa3_attention_once(
        backend,
        &query_difference,
        &key_difference,
        &value,
        window,
        context,
    )?;
    let attended = binary_audio_tensor(
        backend,
        &primary,
        &difference,
        BinaryOperation::Subtract,
        primary.descriptor().clone(),
        context,
    )?;
    let attended = permute_read_only(&attended, &[0, 2, 1, 3])?;
    let attended = contiguous_copy(backend, &attended, context)?;
    let attended = reshape_read_only(&attended, vec![shape[0], shape[1], channels])?;
    sa3_linear_last(
        module,
        backend,
        &attended,
        &format!("{prefix}.to_out.weight"),
        context,
    )
}

fn sa3_transformer(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    window: Option<u64>,
    sinusoidal: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let normalized = sa3_dynamic_tanh(
        module,
        backend,
        input,
        &format!("{prefix}.pre_norm"),
        context,
    )?;
    let attention = sa3_attention(
        module,
        backend,
        &normalized,
        &format!("{prefix}.self_attn"),
        window,
        context,
    )?;
    let mut hidden = add_tensor(backend, input, &attention, context)?;
    let normalized = sa3_dynamic_tanh(
        module,
        backend,
        &hidden,
        &format!("{prefix}.ff_norm"),
        context,
    )?;
    let projected = sa3_linear_last(
        module,
        backend,
        &normalized,
        &format!("{prefix}.ff.ff.0.proj.weight"),
        context,
    )?;
    let channels = projected.descriptor().shape()[2] / 2;
    let values = narrow_contiguous(backend, &projected, 2, 0, channels, context)?;
    let gate = narrow_contiguous(
        backend,
        &projected,
        2,
        i64::try_from(channels)?,
        channels,
        context,
    )?;
    let gate = if sinusoidal {
        let gate = scalar_audio_tensor(
            backend,
            &gate,
            BinaryOperation::Multiply,
            std::f64::consts::PI,
            context,
        )?;
        unary_audio_tensor(backend, &gate, UnaryOperation::Sine, context)?
    } else {
        silu_tensor(backend, &gate, context)?
    };
    let feed_forward = binary_audio_tensor(
        backend,
        &values,
        &gate,
        BinaryOperation::Multiply,
        values.descriptor().clone(),
        context,
    )?;
    let feed_forward = sa3_linear_last(
        module,
        backend,
        &feed_forward,
        &format!("{prefix}.ff.ff.2.weight"),
        context,
    )?;
    hidden = add_tensor(backend, &hidden, &feed_forward, context)?;
    Ok(hidden)
}

#[derive(Clone, Copy)]
struct Sa3Profile {
    channels: u64,
    depth: usize,
    chunk_size: u64,
    midpoint_shift: bool,
    sliding_window: Option<u64>,
    sinusoidal_blocks: usize,
}

fn sa3_profile(module: &NativeModule) -> Result<Sa3Profile, VaeError> {
    if module.layer_name().contains("StableAudio3DeepV1") {
        Ok(Sa3Profile {
            channels: 1_536,
            depth: 12,
            chunk_size: 128,
            midpoint_shift: false,
            sliding_window: Some(17),
            sinusoidal_blocks: 8,
        })
    } else if module.layer_name().contains("StableAudio3ShallowV1") {
        Ok(Sa3Profile {
            channels: 768,
            depth: 6,
            chunk_size: 32,
            midpoint_shift: true,
            sliding_window: None,
            sinusoidal_blocks: 0,
        })
    } else {
        Err(VaeError::KernelProfileMismatch)
    }
}

fn sa3_run_transformers(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    mut input: Tensor,
    prefix: &str,
    profile: Sa3Profile,
    stride: u64,
    decoder: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape().to_vec();
    if let Some(window) = profile.sliding_window {
        for layer in 0..profile.depth {
            input = sa3_transformer(
                module,
                backend,
                &input,
                &format!("{prefix}.transformers.{layer}"),
                Some(window),
                decoder && profile.depth - layer <= profile.sinusoidal_blocks,
                context,
            )?;
        }
        return Ok(input);
    }
    let effective_chunk = profile
        .chunk_size
        .checked_add(profile.chunk_size / stride)
        .ok_or(VaeError::ShapeOverflow)?;
    if !shape[1].is_multiple_of(effective_chunk) {
        return Err(VaeError::ShapeOverflow);
    }
    input = reshape_read_only(
        &input,
        vec![
            shape[0]
                .checked_mul(shape[1] / effective_chunk)
                .ok_or(VaeError::ShapeOverflow)?,
            effective_chunk,
            shape[2],
        ],
    )?;
    let split = if profile.midpoint_shift {
        profile.depth / 2
    } else {
        profile.depth
    };
    for layer in 0..split {
        input = sa3_transformer(
            module,
            backend,
            &input,
            &format!("{prefix}.transformers.{layer}"),
            None,
            decoder && profile.depth - layer <= profile.sinusoidal_blocks,
            context,
        )?;
    }
    if !profile.midpoint_shift {
        return reshape_read_only(&input, shape);
    }
    input = reshape_read_only(&input, shape.clone())?;
    let shift = effective_chunk / 2;
    let first = narrow_contiguous(backend, &input, 1, 0, shift, context)?;
    let last = narrow_contiguous(
        backend,
        &input,
        1,
        i64::try_from(shape[1] - shift)?,
        shift,
        context,
    )?;
    input = concatenate_audio_dimension(backend, &[first, input, last], 1, context)?;
    let shifted_shape = input.descriptor().shape().to_vec();
    input = reshape_read_only(
        &input,
        vec![
            shifted_shape[0]
                .checked_mul(shifted_shape[1] / effective_chunk)
                .ok_or(VaeError::ShapeOverflow)?,
            effective_chunk,
            shifted_shape[2],
        ],
    )?;
    for layer in split..profile.depth {
        input = sa3_transformer(
            module,
            backend,
            &input,
            &format!("{prefix}.transformers.{layer}"),
            None,
            decoder && profile.depth - layer <= profile.sinusoidal_blocks,
            context,
        )?;
    }
    input = reshape_read_only(&input, shifted_shape)?;
    narrow_contiguous(backend, &input, 1, i64::try_from(shift)?, shape[1], context)
}

fn sa3_zero_pad_last_to_multiple(
    backend: &dyn TensorBackend,
    input: &Tensor,
    multiple: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || multiple == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let padding = (multiple - shape[2] % multiple) % multiple;
    if padding == 0 {
        return Ok(input.clone());
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[1], padding],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (zeros, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    concatenate_audio_dimension(backend, &[input.clone(), zeros], 2, context)
}

fn sa3_append_new_tokens(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    output_tokens: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let token = find_module(module, name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Stable Audio 3 new-token parameter {name}"
            )))
        })?;
    if shape.len() != 3 || token.descriptor().shape() != [1, 1, shape[2]] {
        return Err(VaeError::ShapeOverflow);
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], output_tokens, shape[2]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (zeros, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let tokens = binary_audio_tensor(
        backend,
        &zeros,
        token,
        BinaryOperation::Add,
        zeros.descriptor().clone(),
        context,
    )?;
    concatenate_audio_dimension(backend, &[input.clone(), tokens], 1, context)
}

fn sa3_pretransform_encode(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] != 2 {
        return Err(VaeError::ShapeOverflow);
    }
    let input = sa3_zero_pad_last_to_multiple(backend, input, 256, context)?;
    let length = input.descriptor().shape()[2] / 256;
    let patched = reshape_read_only(&input, vec![shape[0], 2, length, 256])?;
    let patched = permute_read_only(&patched, &[0, 1, 3, 2])?;
    let patched = contiguous_copy(backend, &patched, context)?;
    reshape_read_only(&patched, vec![shape[0], 512, length])
}

fn sa3_pretransform_decode(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] != 512 {
        return Err(VaeError::ShapeOverflow);
    }
    let unpacked = reshape_read_only(input, vec![shape[0], 2, 256, shape[2]])?;
    let unpacked = permute_read_only(&unpacked, &[0, 1, 3, 2])?;
    let unpacked = contiguous_copy(backend, &unpacked, context)?;
    reshape_read_only(
        &unpacked,
        vec![
            shape[0],
            2,
            shape[2].checked_mul(256).ok_or(VaeError::ShapeOverflow)?,
        ],
    )
}

fn sa3_encoder_resample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    profile: Sa3Profile,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let pad_multiple = if profile.sliding_window.is_some() {
        16
    } else {
        profile.chunk_size
    };
    let input = sa3_zero_pad_last_to_multiple(backend, input, pad_multiple, context)?;
    let mapped = execute_convolution(module, backend, &input, "encoder.layers.0.mapping", context)?;
    let shape = mapped.descriptor().shape();
    let segments = shape[2] / 16;
    let mapped = permute_read_only(&mapped, &[0, 2, 1])?;
    let mapped = contiguous_copy(backend, &mapped, context)?;
    let mapped = reshape_read_only(
        &mapped,
        vec![
            shape[0]
                .checked_mul(segments)
                .ok_or(VaeError::ShapeOverflow)?,
            16,
            shape[1],
        ],
    )?;
    let mapped = sa3_append_new_tokens(
        module,
        backend,
        &mapped,
        "encoder.layers.0.new_tokens",
        1,
        context,
    )?;
    let mapped = reshape_read_only(&mapped, vec![shape[0], segments * 17, shape[1]])?;
    let mapped = sa3_run_transformers(
        module,
        backend,
        mapped,
        "encoder.layers.0",
        profile,
        16,
        false,
        context,
    )?;
    let mapped = reshape_read_only(
        &mapped,
        vec![
            shape[0]
                .checked_mul(segments)
                .ok_or(VaeError::ShapeOverflow)?,
            17,
            shape[1],
        ],
    )?;
    let output = narrow_contiguous(backend, &mapped, 1, 16, 1, context)?;
    let output = reshape_read_only(&output, vec![shape[0], segments, shape[1]])?;
    let output = sa3_linear_last(module, backend, &output, "encoder.layers.2.weight", context)?;
    let output = permute_read_only(&output, &[0, 2, 1])?;
    contiguous_copy(backend, &output, context)
}

fn sa3_decoder_resample(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    profile: Sa3Profile,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    let input = permute_read_only(input, &[0, 2, 1])?;
    let input = contiguous_copy(backend, &input, context)?;
    let input = sa3_linear_last(module, backend, &input, "decoder.layers.1.weight", context)?;
    let input = permute_read_only(&input, &[0, 2, 1])?;
    let input = contiguous_copy(backend, &input, context)?;
    let pad_multiple = if profile.sliding_window.is_some() {
        1
    } else {
        profile.chunk_size / 16
    };
    let input = sa3_zero_pad_last_to_multiple(backend, &input, pad_multiple, context)?;
    let steps = input.descriptor().shape()[2];
    let input = permute_read_only(&input, &[0, 2, 1])?;
    let input = contiguous_copy(backend, &input, context)?;
    let input = reshape_read_only(
        &input,
        vec![
            shape[0].checked_mul(steps).ok_or(VaeError::ShapeOverflow)?,
            1,
            profile.channels,
        ],
    )?;
    let input = sa3_append_new_tokens(
        module,
        backend,
        &input,
        "decoder.layers.3.new_tokens",
        16,
        context,
    )?;
    let input = reshape_read_only(&input, vec![shape[0], steps * 17, profile.channels])?;
    let input = sa3_run_transformers(
        module,
        backend,
        input,
        "decoder.layers.3",
        profile,
        16,
        true,
        context,
    )?;
    let input = reshape_read_only(
        &input,
        vec![
            shape[0].checked_mul(steps).ok_or(VaeError::ShapeOverflow)?,
            17,
            profile.channels,
        ],
    )?;
    let input = narrow_contiguous(backend, &input, 1, 1, 16, context)?;
    let input = reshape_read_only(&input, vec![shape[0], steps * 16, profile.channels])?;
    let input = permute_read_only(&input, &[0, 2, 1])?;
    let input = contiguous_copy(backend, &input, context)?;
    execute_convolution(module, backend, &input, "decoder.layers.3.mapping", context)
}

fn sa3_bottleneck_parameter<'a>(
    module: &'a NativeModule,
    name: &str,
) -> Result<&'a Tensor, VaeError> {
    find_module(module, name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Stable Audio 3 bottleneck parameter {name}"
            )))
        })
}

fn sa3_encode_tensor(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let profile = sa3_profile(module)?;
    let patched = sa3_pretransform_encode(backend, input, context)?;
    let latent = sa3_encoder_resample(module, backend, &patched, profile, context)?;
    let scaling = sa3_bottleneck_parameter(module, "bottleneck.scaling_factor")?;
    let bias = sa3_bottleneck_parameter(module, "bottleneck.bias")?;
    let running = sa3_bottleneck_parameter(module, "bottleneck.running_std")?;
    let latent = binary_audio_tensor(
        backend,
        &latent,
        scaling,
        BinaryOperation::Multiply,
        latent.descriptor().clone(),
        context,
    )?;
    let latent = binary_audio_tensor(
        backend,
        &latent,
        bias,
        BinaryOperation::Add,
        latent.descriptor().clone(),
        context,
    )?;
    binary_audio_tensor(
        backend,
        &latent,
        running,
        BinaryOperation::Divide,
        latent.descriptor().clone(),
        context,
    )
}

fn sa3_decode_tensor(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let profile = sa3_profile(module)?;
    let running = sa3_bottleneck_parameter(module, "bottleneck.running_std")?;
    let mut latent = binary_audio_tensor(
        backend,
        input,
        running,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let cpu_backend = cpu_backend.ok_or(VaeError::AudioVaeRequiresCpuBackend)?;
    let random = randn_like_with_context_exact_native(
        cpu_backend,
        &latent,
        begin_vae_rng(context)?,
        context,
    )
    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    let noise_scale =
        scalar_audio_tensor(backend, running, BinaryOperation::Multiply, 1.0e-3, context)?;
    let noise = binary_audio_tensor(
        backend,
        &random.tensor,
        &noise_scale,
        BinaryOperation::Multiply,
        latent.descriptor().clone(),
        context,
    )?;
    latent = add_tensor(backend, &latent, &noise, context)?;
    let decoded = sa3_decoder_resample(module, backend, &latent, profile, context)?;
    sa3_pretransform_decode(backend, &decoded, context)
}

fn ltx_zero_pad_2d(
    backend: &dyn TensorBackend,
    input: &Tensor,
    left: u64,
    right: u64,
    top: u64,
    bottom: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape.contains(&0) {
        return Err(VaeError::ShapeOverflow);
    }
    let output_shape = vec![
        shape[0],
        shape[1],
        shape[2]
            .checked_add(top)
            .and_then(|value| value.checked_add(bottom))
            .ok_or(VaeError::ShapeOverflow)?,
        shape[3]
            .checked_add(left)
            .and_then(|value| value.checked_add(right))
            .ok_or(VaeError::ShapeOverflow)?,
    ];
    let descriptor = TensorDescriptor::contiguous(
        output_shape,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    backend.wait_event(event, context)?;
    let (output, event) =
        backend.replace_rectangular_slice(&output, input, &[0, 0, top, left], context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn ltx_causal_convolution(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let input = if name.contains("nin_shortcut") {
        input.clone()
    } else if name.contains(".downsample.") {
        ltx_zero_pad_2d(backend, input, 0, 1, 2, 0, context)?
    } else {
        ltx_zero_pad_2d(backend, input, 1, 1, 2, 0, context)?
    };
    execute_convolution(module, backend, &input, name, context)
}

fn ltx_pixel_norm(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[1] == 0 {
        return Err(VaeError::ShapeOverflow);
    }
    let squared = binary_audio_tensor(
        backend,
        input,
        input,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], 1, shape[2], shape[3]],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mean, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Mean,
            dimensions: vec![1],
            keep_dimensions: true,
            accumulation_dtype: Some(DType::F32),
            correction: 0,
        },
        &squared,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let mean = scalar_audio_tensor(backend, &mean, BinaryOperation::Add, 1.0e-6, context)?;
    let norm = unary_audio_tensor(backend, &mean, UnaryOperation::SquareRoot, context)?;
    binary_audio_tensor(
        backend,
        input,
        &norm,
        BinaryOperation::Divide,
        input.descriptor().clone(),
        context,
    )
}

fn ltx_residual_block(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let normalized = ltx_pixel_norm(backend, input, context)?;
    let activated = silu_tensor(backend, &normalized, context)?;
    let hidden = ltx_causal_convolution(
        module,
        backend,
        &activated,
        &format!("{prefix}.conv1.conv.weight"),
        context,
    )?;
    let hidden = ltx_pixel_norm(backend, &hidden, context)?;
    let hidden = silu_tensor(backend, &hidden, context)?;
    let hidden = ltx_causal_convolution(
        module,
        backend,
        &hidden,
        &format!("{prefix}.conv2.conv.weight"),
        context,
    )?;
    let shortcut_name = format!("{prefix}.nin_shortcut.conv.weight");
    let shortcut = if find_module(module, &shortcut_name).is_some() {
        ltx_causal_convolution(module, backend, input, &shortcut_name, context)?
    } else {
        input.clone()
    };
    add_tensor(backend, &shortcut, &hidden, context)
}

fn ltx_normalize_latent(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    inverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[1] != 8 || shape[3] != 16 {
        return Err(VaeError::InvalidShape {
            expected: vec![shape.first().copied().unwrap_or(0), 8, 0, 16],
            actual: shape.to_vec(),
        });
    }
    let flattened = permute_read_only(input, &[0, 2, 1, 3])?;
    let flattened = contiguous_copy(backend, &flattened, context)?;
    let flattened = reshape_read_only(&flattened, vec![shape[0], shape[2], 128])?;
    let statistic = |name: &str| -> Result<Tensor, VaeError> {
        let tensor = audio_buffer(module, name)?;
        if tensor.descriptor().shape() != [128] {
            return Err(VaeError::ShapeOverflow);
        }
        reshape_read_only(tensor, vec![1, 1, 128])
    };
    let mean = statistic("autoencoder.per_channel_statistics.mean-of-means")?;
    let standard_deviation = statistic("autoencoder.per_channel_statistics.std-of-means")?;
    let normalized = if inverse {
        let scaled = binary_audio_tensor(
            backend,
            &flattened,
            &standard_deviation,
            BinaryOperation::Multiply,
            flattened.descriptor().clone(),
            context,
        )?;
        binary_audio_tensor(
            backend,
            &scaled,
            &mean,
            BinaryOperation::Add,
            flattened.descriptor().clone(),
            context,
        )?
    } else {
        let centered = binary_audio_tensor(
            backend,
            &flattened,
            &mean,
            BinaryOperation::Subtract,
            flattened.descriptor().clone(),
            context,
        )?;
        binary_audio_tensor(
            backend,
            &centered,
            &standard_deviation,
            BinaryOperation::Divide,
            flattened.descriptor().clone(),
            context,
        )?
    };
    let normalized = reshape_read_only(&normalized, vec![shape[0], shape[2], 8, 16])?;
    let normalized = permute_read_only(&normalized, &[0, 2, 1, 3])?;
    contiguous_copy(backend, &normalized, context)
}

fn ltx_waveform_to_mel(
    cpu_backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] != 2 || shape[2] <= 1_411 {
        return Err(VaeError::InvalidShape {
            expected: vec![shape.first().copied().unwrap_or(0), 2, 1_412],
            actual: shape.to_vec(),
        });
    }
    let resampled = resample_with_context_exact_native(
        cpu_backend,
        input,
        NativeResampleConfiguration::torchaudio_default(44_100, 16_000),
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let mel = mel_spectrogram_with_context_exact_native(
        cpu_backend,
        &resampled,
        NativeMelSpectrogramConfiguration {
            sample_rate: 16_000,
            n_fft: 1_024,
            win_length: Some(1_024),
            hop_length: Some(160),
            f_min: 0.0,
            f_max: Some(8_000.0),
            n_mels: 64,
            power: 1.0,
            center: true,
            normalized: false,
            mel_scale: NativeMelScale::Slaney,
            mel_normalization: NativeMelNormalization::Slaney,
        },
        context,
    )
    .map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))?;
    let mel = scalar_audio_tensor(cpu_backend, &mel, BinaryOperation::Maximum, 1.0e-5, context)?;
    let mel = unary_audio_tensor(cpu_backend, &mel, UnaryOperation::NaturalLogarithm, context)?;
    let mel = permute_read_only(&mel, &[0, 1, 3, 2])?;
    contiguous_copy(cpu_backend, &mel, context)
}

fn ltx_autoencoder_encode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = ltx_causal_convolution(
        module,
        backend,
        input,
        "autoencoder.encoder.conv_in.conv.weight",
        context,
    )?;
    for level in 0..3 {
        for block in 0..2 {
            hidden = ltx_residual_block(
                module,
                backend,
                &hidden,
                &format!("autoencoder.encoder.down.{level}.block.{block}"),
                context,
            )?;
        }
        if level != 2 {
            hidden = ltx_causal_convolution(
                module,
                backend,
                &hidden,
                &format!("autoencoder.encoder.down.{level}.downsample.conv.weight"),
                context,
            )?;
        }
    }
    hidden = ltx_residual_block(
        module,
        backend,
        &hidden,
        "autoencoder.encoder.mid.block_1",
        context,
    )?;
    hidden = ltx_residual_block(
        module,
        backend,
        &hidden,
        "autoencoder.encoder.mid.block_2",
        context,
    )?;
    hidden = ltx_pixel_norm(backend, &hidden, context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    hidden = ltx_causal_convolution(
        module,
        backend,
        &hidden,
        "autoencoder.encoder.conv_out.conv.weight",
        context,
    )?;
    let hidden = narrow_contiguous(backend, &hidden, 1, 0, 8, context)?;
    ltx_normalize_latent(module, backend, &hidden, false, context)
}

fn ltx_autoencoder_decode(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = ltx_normalize_latent(module, backend, input, true, context)?;
    hidden = ltx_causal_convolution(
        module,
        backend,
        &hidden,
        "autoencoder.decoder.conv_in.conv.weight",
        context,
    )?;
    hidden = ltx_residual_block(
        module,
        backend,
        &hidden,
        "autoencoder.decoder.mid.block_1",
        context,
    )?;
    hidden = ltx_residual_block(
        module,
        backend,
        &hidden,
        "autoencoder.decoder.mid.block_2",
        context,
    )?;
    for level in (0..3).rev() {
        for block in 0..3 {
            hidden = ltx_residual_block(
                module,
                backend,
                &hidden,
                &format!("autoencoder.decoder.up.{level}.block.{block}"),
                context,
            )?;
        }
        if level != 0 {
            hidden = nearest_upsample_2x(backend, &hidden, context)?;
            hidden = ltx_causal_convolution(
                module,
                backend,
                &hidden,
                &format!("autoencoder.decoder.up.{level}.upsample.conv.conv.weight"),
                context,
            )?;
            let time = hidden.descriptor().shape()[2];
            hidden = narrow_contiguous(backend, &hidden, 2, 1, time - 1, context)?;
        }
    }
    hidden = ltx_pixel_norm(backend, &hidden, context)?;
    hidden = silu_tensor(backend, &hidden, context)?;
    ltx_causal_convolution(
        module,
        backend,
        &hidden,
        "autoencoder.decoder.conv_out.conv.weight",
        context,
    )
}

fn ltx_leaky_relu(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let positive = unary_audio_tensor(backend, input, UnaryOperation::Relu, context)?;
    let negative = unary_audio_tensor(backend, input, UnaryOperation::Negate, context)?;
    let negative = unary_audio_tensor(backend, &negative, UnaryOperation::Relu, context)?;
    let negative =
        scalar_audio_tensor(backend, &negative, BinaryOperation::Multiply, -0.1, context)?;
    add_tensor(backend, &positive, &negative, context)
}

fn ltx_vocoder_residual(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    block: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mut hidden = input.clone();
    for layer in 0..3 {
        let activated = ltx_leaky_relu(backend, &hidden, context)?;
        let convolved = execute_convolution(
            module,
            backend,
            &activated,
            &format!("vocoder.resblocks.{block}.convs1.{layer}.weight"),
            context,
        )?;
        let activated = ltx_leaky_relu(backend, &convolved, context)?;
        let convolved = execute_convolution(
            module,
            backend,
            &activated,
            &format!("vocoder.resblocks.{block}.convs2.{layer}.weight"),
            context,
        )?;
        hidden = add_tensor(backend, &hidden, &convolved, context)?;
    }
    Ok(hidden)
}

fn ltx_vocoder(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    mel: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = mel.descriptor().shape();
    if shape.len() != 4 || shape[1] != 2 || shape[3] != 64 {
        return Err(VaeError::ShapeOverflow);
    }
    let mel = permute_read_only(mel, &[0, 1, 3, 2])?;
    let mel = contiguous_copy(backend, &mel, context)?;
    let mel = reshape_read_only(&mel, vec![shape[0], 128, shape[2]])?;
    let mut hidden =
        execute_convolution(module, backend, &mel, "vocoder.conv_pre.weight", context)?;
    for stage in 0..5 {
        hidden = ltx_leaky_relu(backend, &hidden, context)?;
        hidden = execute_convolution(
            module,
            backend,
            &hidden,
            &format!("vocoder.ups.{stage}.weight"),
            context,
        )?;
        let mut aggregate = None;
        for kernel in 0..3 {
            let residual =
                ltx_vocoder_residual(module, backend, &hidden, stage * 3 + kernel, context)?;
            aggregate = Some(match aggregate {
                None => residual,
                Some(current) => add_tensor(backend, &current, &residual, context)?,
            });
        }
        hidden = scalar_audio_tensor(
            backend,
            &aggregate.ok_or(VaeError::ShapeOverflow)?,
            BinaryOperation::Divide,
            3.0,
            context,
        )?;
    }
    hidden = ltx_leaky_relu(backend, &hidden, context)?;
    hidden = execute_convolution(
        module,
        backend,
        &hidden,
        "vocoder.conv_post.weight",
        context,
    )?;
    unary_audio_tensor(backend, &hidden, UnaryOperation::HyperbolicTangent, context)
}

fn ltx_encode_tensor(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mel = ltx_waveform_to_mel(cpu_backend, input, context)?;
    ltx_autoencoder_encode(module, backend, &mel, context)
}

fn ltx_decode_tensor(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let mel = ltx_autoencoder_decode(module, backend, input, context)?;
    ltx_vocoder(module, backend, &mel, context)
}

fn oobleck_snake_beta(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    let shape = input.descriptor().shape();
    if shape.len() != 3 {
        return Err(VaeError::InvalidShape {
            expected: vec![0, 0, 0],
            actual: shape.to_vec(),
        });
    }
    let alpha_name = format!("{name}.alpha");
    let beta_name = format!("{name}.beta");
    let alpha = find_module(module, &alpha_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Oobleck activation parameter {alpha_name}"
            )))
        })?;
    let beta = find_module(module, &beta_name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| {
            VaeError::NativeOps(NativeOpsError::InvalidOwned(format!(
                "missing Oobleck activation parameter {beta_name}"
            )))
        })?;
    if alpha.descriptor().shape() != [shape[1]] || beta.descriptor().shape() != [shape[1]] {
        return Err(VaeError::InvalidShape {
            expected: vec![shape[1]],
            actual: alpha.descriptor().shape().to_vec(),
        });
    }
    let alpha = unary_audio_tensor(backend, alpha, UnaryOperation::Exponential, context)?;
    let beta = unary_audio_tensor(backend, beta, UnaryOperation::Exponential, context)?;
    let alpha = reshape_read_only(&alpha, vec![1, shape[1], 1])?;
    let beta = reshape_read_only(&beta, vec![1, shape[1], 1])?;
    let phase = binary_audio_tensor(
        backend,
        input,
        &alpha,
        BinaryOperation::Multiply,
        input.descriptor().clone(),
        context,
    )?;
    let sine = unary_audio_tensor(backend, &phase, UnaryOperation::Sine, context)?;
    let squared = binary_audio_tensor(
        backend,
        &sine,
        &sine,
        BinaryOperation::Multiply,
        sine.descriptor().clone(),
        context,
    )?;
    let beta = scalar_audio_tensor(backend, &beta, BinaryOperation::Add, 1.0e-9, context)?;
    let periodic = binary_audio_tensor(
        backend,
        &squared,
        &beta,
        BinaryOperation::Divide,
        squared.descriptor().clone(),
        context,
    )?;
    add_tensor(backend, input, &periodic, context)
}

fn oobleck_residual_unit(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let activated = oobleck_snake_beta(
        module,
        backend,
        input,
        &format!("{prefix}.layers.0"),
        context,
    )?;
    let convolved = execute_convolution(
        module,
        backend,
        &activated,
        &format!("{prefix}.layers.1"),
        context,
    )?;
    let activated = oobleck_snake_beta(
        module,
        backend,
        &convolved,
        &format!("{prefix}.layers.2"),
        context,
    )?;
    let convolved = execute_convolution(
        module,
        backend,
        &activated,
        &format!("{prefix}.layers.3"),
        context,
    )?;
    add_tensor(backend, &convolved, input, context)
}

fn audio_softplus(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let positive = unary_audio_tensor(backend, input, UnaryOperation::Relu, context)?;
    let absolute = unary_audio_tensor(backend, input, UnaryOperation::Absolute, context)?;
    let negative = unary_audio_tensor(backend, &absolute, UnaryOperation::Negate, context)?;
    let exponential = unary_audio_tensor(backend, &negative, UnaryOperation::Exponential, context)?;
    let one_plus = scalar_audio_tensor(backend, &exponential, BinaryOperation::Add, 1.0, context)?;
    let logarithm = unary_audio_tensor(
        backend,
        &one_plus,
        UnaryOperation::NaturalLogarithm,
        context,
    )?;
    add_tensor(backend, &positive, &logarithm, context)
}

fn oobleck_reparameterized_sample(
    backend: &dyn TensorBackend,
    cpu_backend: &CpuBackend,
    mean: &Tensor,
    scale: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    if mean.descriptor().shape() != scale.descriptor().shape() {
        return Err(VaeError::InvalidShape {
            expected: mean.descriptor().shape().to_vec(),
            actual: scale.descriptor().shape().to_vec(),
        });
    }
    let mean = contiguous_copy(backend, mean, context)?;
    let scale = contiguous_copy(backend, scale, context)?;
    let standard_deviation = audio_softplus(backend, &scale, context)?;
    let standard_deviation = scalar_audio_tensor(
        backend,
        &standard_deviation,
        BinaryOperation::Add,
        1.0e-4,
        context,
    )?;
    let random =
        randn_like_with_context_exact_native(cpu_backend, &mean, begin_vae_rng(context)?, context)
            .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    let noise = binary_audio_tensor(
        backend,
        &random.tensor,
        &standard_deviation,
        BinaryOperation::Multiply,
        mean.descriptor().clone(),
        context,
    )?;
    add_tensor(backend, &mean, &noise, context)
}

fn oobleck_encode_tensor(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] != 2 || shape[2] == 0 {
        return Err(VaeError::InvalidShape {
            expected: vec![shape.first().copied().unwrap_or(0), 2, 0],
            actual: shape.to_vec(),
        });
    }
    let mut hidden = execute_convolution(module, backend, input, "encoder.layers.0", context)?;
    for block in 1..=5 {
        for residual in 0..3 {
            hidden = oobleck_residual_unit(
                module,
                backend,
                &hidden,
                &format!("encoder.layers.{block}.layers.{residual}"),
                context,
            )?;
        }
        hidden = oobleck_snake_beta(
            module,
            backend,
            &hidden,
            &format!("encoder.layers.{block}.layers.3"),
            context,
        )?;
        hidden = execute_convolution(
            module,
            backend,
            &hidden,
            &format!("encoder.layers.{block}.layers.4"),
            context,
        )?;
    }
    hidden = oobleck_snake_beta(module, backend, &hidden, "encoder.layers.6", context)?;
    hidden = execute_convolution(module, backend, &hidden, "encoder.layers.7", context)?;
    let hidden_shape = hidden.descriptor().shape();
    if hidden_shape.len() != 3 || hidden_shape[1] != 128 {
        return Err(VaeError::InvalidShape {
            expected: vec![shape[0], 128, 0],
            actual: hidden_shape.to_vec(),
        });
    }
    let mean = hidden.narrow_read_only(1, 0, 64)?;
    let scale = hidden.narrow_read_only(1, 64, 64)?;
    oobleck_reparameterized_sample(backend, cpu_backend, &mean, &scale, context)
}

fn oobleck_decode_tensor(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 3 || shape[1] != 64 || shape[2] == 0 {
        return Err(VaeError::InvalidShape {
            expected: vec![shape.first().copied().unwrap_or(0), 64, 0],
            actual: shape.to_vec(),
        });
    }
    let mut hidden = execute_convolution(module, backend, input, "decoder.layers.0", context)?;
    for block in 1..=5 {
        hidden = oobleck_snake_beta(
            module,
            backend,
            &hidden,
            &format!("decoder.layers.{block}.layers.0"),
            context,
        )?;
        hidden = execute_convolution(
            module,
            backend,
            &hidden,
            &format!("decoder.layers.{block}.layers.1"),
            context,
        )?;
        for residual in 0..3 {
            hidden = oobleck_residual_unit(
                module,
                backend,
                &hidden,
                &format!("decoder.layers.{block}.layers.{}", residual + 2),
                context,
            )?;
        }
    }
    hidden = oobleck_snake_beta(module, backend, &hidden, "decoder.layers.6", context)?;
    execute_convolution(module, backend, &hidden, "decoder.layers.7", context)
}

fn audio_encode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    audio_encode_tensor(module, backend, cpu_backend, input, context)
}

fn audio_encode_tensor(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    if module.layer_name().contains("AudioOobleck44KhzV1")
        || module.layer_name().contains("AudioOobleck48KhzV1")
    {
        let cpu_backend = cpu_backend.ok_or(VaeError::AudioVaeRequiresCpuBackend)?;
        return oobleck_encode_tensor(module, backend, cpu_backend, input, context);
    }
    if module.layer_name().contains("MmAudio16KhzV1") {
        let cpu_backend = cpu_backend.ok_or(VaeError::AudioVaeRequiresCpuBackend)?;
        let mel = mmaudio_waveform_to_mel(module, cpu_backend, input, context)?;
        return mmaudio_vae_encode(module, backend, cpu_backend, &mel, context);
    }
    if module.layer_name().contains("MusicDcaeV1") {
        let cpu_backend = cpu_backend.ok_or(VaeError::AudioVaeRequiresCpuBackend)?;
        let mel = music_waveform_to_mel_image(module, cpu_backend, input, context)?;
        return music_dcae_encode(module, backend, &mel, context);
    }
    if module.layer_name().contains("LtxAudioV1") {
        let cpu_backend = cpu_backend.ok_or(VaeError::AudioVaeRequiresCpuBackend)?;
        return ltx_encode_tensor(module, backend, cpu_backend, input, context);
    }
    if module.layer_name().contains("StableAudio3DeepV1")
        || module.layer_name().contains("StableAudio3ShallowV1")
    {
        return sa3_encode_tensor(module, backend, input, context);
    }
    Err(VaeError::OperationUnavailable {
        profile: module.layer_name().to_owned(),
        operation: VaeOperation::Encode,
    })
}

fn audio_decode_raw(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    _latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    audio_decode_tensor(module, backend, cpu_backend, input, context)
}

fn audio_decode_tensor(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    if module.layer_name().contains("AudioOobleck44KhzV1")
        || module.layer_name().contains("AudioOobleck48KhzV1")
    {
        return oobleck_decode_tensor(module, backend, input, context);
    }
    if module.layer_name().contains("MmAudio16KhzV1") {
        let cpu_backend = cpu_backend.ok_or(VaeError::AudioVaeRequiresCpuBackend)?;
        return mmaudio_decode_waveform(module, cpu_backend, input, context);
    }
    if module.layer_name().contains("MusicDcaeV1") {
        let mel = music_dcae_decode_mel(module, backend, input, context)?;
        return music_vocoder_decode(module, backend, &mel, context);
    }
    if module.layer_name().contains("LtxAudioV1") {
        return ltx_decode_tensor(module, backend, input, context);
    }
    if module.layer_name().contains("StableAudio3DeepV1")
        || module.layer_name().contains("StableAudio3ShallowV1")
    {
        return sa3_decode_tensor(module, backend, cpu_backend, input, context);
    }
    Err(VaeError::OperationUnavailable {
        profile: module.layer_name().to_owned(),
        operation: VaeOperation::Decode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{
        CpuWorkspaceAuthority, DeviceId, RetryRngPolicy, RngStreamAddress, StreamId,
        TensorDescriptor, TensorError,
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

    #[test]
    fn source_plans_preserve_audio_rates_layouts_and_ratios() -> Result<(), AudioVaeError> {
        for (profile, rates, ratio, channels, dimensions, bins) in [
            (
                VaeKernelProfile::AudioOobleck44KhzV1,
                (44_100, 44_100),
                (2_048, 1),
                64,
                1,
                None,
            ),
            (
                VaeKernelProfile::AudioOobleck48KhzV1,
                (48_000, 48_000),
                (1_920, 1),
                64,
                1,
                None,
            ),
            (
                VaeKernelProfile::MusicDcaeV1,
                (44_100, 44_100),
                (4_096, 1),
                8,
                2,
                Some(16),
            ),
            (
                VaeKernelProfile::MmAudio16KhzV1,
                (44_100, 44_100),
                (141_120, 100),
                20,
                1,
                None,
            ),
            (
                VaeKernelProfile::LtxAudioV1,
                (44_100, 16_000),
                (1_764, 1),
                8,
                2,
                Some(16),
            ),
            (
                VaeKernelProfile::StableAudio3DeepV1,
                (44_100, 44_100),
                (4_096, 1),
                256,
                1,
                None,
            ),
            (
                VaeKernelProfile::StableAudio3ShallowV1,
                (44_100, 44_100),
                (4_096, 1),
                256,
                1,
                None,
            ),
        ] {
            let plan = audio_vae_source_plan(&profile)?;
            assert_eq!((plan.input_sample_rate(), plan.output_sample_rate()), rates);
            assert_eq!(plan.sample_ratio(), ratio);
            assert_eq!(plan.latent_channels(), channels);
            assert_eq!(plan.latent_dimensions(), dimensions);
            assert_eq!(plan.latent_frequency_bins(), bins);
            assert!(!plan.state_checkpoints().is_empty());
            assert!(!plan.equation_checkpoints().is_empty());
        }
        Ok(())
    }

    #[test]
    fn source_state_kinds_match_registered_buffers_exactly() {
        for (profile, name) in [
            (VaeKernelProfile::MmAudio16KhzV1, "mel_converter.mel_basis"),
            (
                VaeKernelProfile::MmAudio16KhzV1,
                "mel_converter.hann_window",
            ),
            (VaeKernelProfile::MmAudio16KhzV1, "vae.data_mean"),
            (VaeKernelProfile::MmAudio16KhzV1, "vae.data_std"),
            (
                VaeKernelProfile::MmAudio16KhzV1,
                "vocoder.activation_post.downsample.lowpass.filter",
            ),
            (
                VaeKernelProfile::MusicDcaeV1,
                "vocoder.mel_transform.spectrogram.window",
            ),
            (
                VaeKernelProfile::MusicDcaeV1,
                "vocoder.mel_transform.mel_scale.fb",
            ),
            (
                VaeKernelProfile::LtxAudioV1,
                "autoencoder.per_channel_statistics.std-of-means",
            ),
            (
                VaeKernelProfile::StableAudio3DeepV1,
                "encoder.layers.0.transformers.0.rope.inv_freq",
            ),
        ] {
            assert_eq!(
                audio_source_state_kind(&profile, name),
                NativeVisionStateKind::Buffer,
                "{profile:?} {name}"
            );
        }
        for (profile, name) in [
            (
                VaeKernelProfile::MmAudio16KhzV1,
                "vae.encoder.conv_in.weight",
            ),
            (VaeKernelProfile::MusicDcaeV1, "dcae.encoder.conv_in.weight"),
            (
                VaeKernelProfile::LtxAudioV1,
                "autoencoder.encoder.conv_in.weight",
            ),
            (
                VaeKernelProfile::StableAudio3ShallowV1,
                "encoder.layers.0.transformers.0.attn.to_q.weight",
            ),
        ] {
            assert_eq!(
                audio_source_state_kind(&profile, name),
                NativeVisionStateKind::Parameter,
                "{profile:?} {name}"
            );
        }
    }

    #[test]
    fn every_audio_profile_admits_only_its_complete_exact_state_schema()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::formats::FileSlice;
        use std::path::PathBuf;

        fn metadata(
            state: Vec<AudioStateShape>,
        ) -> Result<BTreeMap<String, TensorMetadata>, AudioVaeError> {
            state
                .into_iter()
                .enumerate()
                .map(|(index, state)| {
                    let length = state
                        .shape
                        .iter()
                        .try_fold(4_u64, |bytes, extent| bytes.checked_mul(*extent))
                        .ok_or(AudioVaeError::ShapeOverflow)?;
                    Ok((
                        state.name.clone(),
                        TensorMetadata {
                            name: state.name,
                            data_type: "F32".to_owned(),
                            shape: state.shape,
                            storage: FileSlice {
                                path: PathBuf::from("metadata-only-audio-vae.safetensors"),
                                offset: u64::try_from(index)
                                    .map_err(|_| AudioVaeError::ShapeOverflow)?,
                                length,
                            },
                        },
                    ))
                })
                .collect()
        }

        let ltx_autoencoder = r#"{
            "model":{"params":{"ddconfig":{
                "double_z":true,"mel_bins":64,"z_channels":8,"resolution":256,
                "in_channels":2,"out_ch":2,"ch":128,"ch_mult":[1,2,4],
                "num_res_blocks":2,"attn_resolutions":[],"dropout":0.0,
                "mid_block_add_attention":false,"norm_type":"pixel",
                "causality_axis":"height"
            },"sampling_rate":16000}},
            "preprocessing":{"stft":{"filter_length":1024,"hop_length":160}}
        }"#;
        let ltx_vocoder = r#"{"upsample_rates":[5,4,2,2,2]}"#;
        let ltx_configuration = VaeLoaderConfiguration::LtxAudio {
            autoencoder_sha256: "0".repeat(64),
            autoencoder_json: ltx_autoencoder.to_owned(),
            vocoder_sha256: "1".repeat(64),
            vocoder_json: ltx_vocoder.to_owned(),
            latent_channels: 8,
            input_sample_rate: 16_000,
            output_sample_rate: 16_000,
        };
        let automatic = VaeLoaderConfiguration::Automatic;
        let cases = vec![
            (
                VaeKernelProfile::AudioOobleck44KhzV1,
                OOBLECK_ARCHITECTURE,
                &automatic,
                oobleck_source_state_shapes(&VaeKernelProfile::AudioOobleck44KhzV1)
                    .ok_or(AudioVaeError::ShapeOverflow)?,
            ),
            (
                VaeKernelProfile::AudioOobleck48KhzV1,
                OOBLECK_ARCHITECTURE,
                &automatic,
                oobleck_source_state_shapes(&VaeKernelProfile::AudioOobleck48KhzV1)
                    .ok_or(AudioVaeError::ShapeOverflow)?,
            ),
            (
                VaeKernelProfile::MusicDcaeV1,
                MUSIC_DCAE_ARCHITECTURE,
                &automatic,
                music_dcae_source_state_shapes(),
            ),
            (
                VaeKernelProfile::MmAudio16KhzV1,
                MMAUDIO_ARCHITECTURE,
                &automatic,
                mmaudio_source_state_shapes(),
            ),
            (
                VaeKernelProfile::LtxAudioV1,
                LTX_AUDIO_ARCHITECTURE,
                &ltx_configuration,
                ltx_audio_source_state_shapes(),
            ),
            (
                VaeKernelProfile::StableAudio3DeepV1,
                STABLE_AUDIO_3_ARCHITECTURE,
                &automatic,
                stable_audio_3_source_state_shapes(&VaeKernelProfile::StableAudio3DeepV1)
                    .ok_or(AudioVaeError::ShapeOverflow)?,
            ),
            (
                VaeKernelProfile::StableAudio3ShallowV1,
                STABLE_AUDIO_3_ARCHITECTURE,
                &automatic,
                stable_audio_3_source_state_shapes(&VaeKernelProfile::StableAudio3ShallowV1)
                    .ok_or(AudioVaeError::ShapeOverflow)?,
            ),
        ];
        for (profile, architecture, configuration, state) in cases {
            let expected_len = state.len();
            let tensors = metadata(state)?;
            let admitted = inspect_audio_vae_architecture_from_tensors(
                &profile,
                configuration,
                architecture,
                &tensors,
            )?;
            assert_eq!(admitted.state_schema().len(), expected_len, "{profile:?}");
            assert_eq!(admitted.storage_dtype(), Some(DType::F32));

            let mut missing = tensors.clone();
            let missing_name = missing
                .keys()
                .next()
                .cloned()
                .ok_or(AudioVaeError::ShapeOverflow)?;
            missing.remove(&missing_name);
            assert!(matches!(
                inspect_audio_vae_architecture_from_tensors(
                    &profile,
                    configuration,
                    architecture,
                    &missing,
                ),
                Err(AudioVaeError::MissingState(_))
            ));

            let mut invalid_shape = tensors.clone();
            let invalid_shape_name = invalid_shape
                .keys()
                .next()
                .cloned()
                .ok_or(AudioVaeError::ShapeOverflow)?;
            invalid_shape
                .get_mut(&invalid_shape_name)
                .ok_or(AudioVaeError::ShapeOverflow)?
                .shape
                .push(1);
            assert!(matches!(
                inspect_audio_vae_architecture_from_tensors(
                    &profile,
                    configuration,
                    architecture,
                    &invalid_shape,
                ),
                Err(AudioVaeError::InvalidStateShape { .. })
            ));

            let mut integer_state = tensors.clone();
            let integer_name = integer_state
                .keys()
                .next()
                .cloned()
                .ok_or(AudioVaeError::ShapeOverflow)?;
            integer_state
                .get_mut(&integer_name)
                .ok_or(AudioVaeError::ShapeOverflow)?
                .data_type = "I64".to_owned();
            assert!(matches!(
                inspect_audio_vae_architecture_from_tensors(
                    &profile,
                    configuration,
                    architecture,
                    &integer_state,
                ),
                Err(AudioVaeError::UnsupportedStorageDType { .. })
            ));

            let mut unexpected = tensors;
            unexpected.insert(
                "parallel.audio.owner.weight".to_owned(),
                TensorMetadata {
                    name: "parallel.audio.owner.weight".to_owned(),
                    data_type: "F32".to_owned(),
                    shape: vec![1],
                    storage: FileSlice {
                        path: PathBuf::from("metadata-only-audio-vae.safetensors"),
                        offset: 0,
                        length: 4,
                    },
                },
            );
            assert!(matches!(
                inspect_audio_vae_architecture_from_tensors(
                    &profile,
                    configuration,
                    architecture,
                    &unexpected,
                ),
                Err(AudioVaeError::UnexpectedState(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn ratio_arithmetic_is_checked_and_source_exact() -> Result<(), VaeError> {
        assert_eq!(checked_ceil_ratio(44_100, 141_120, 100)?, 32);
        assert_eq!(checked_expand_ratio(32, 141_120, 100)?, 45_158);
        assert_eq!(checked_ceil_ratio(4_097, 4_096, 1)?, 2);
        assert_eq!(checked_expand_ratio(2, 4_096, 1)?, 8_192);
        Ok(())
    }

    #[test]
    fn oobleck_manifest_covers_complete_learned_topology() {
        for profile in [
            VaeKernelProfile::AudioOobleck44KhzV1,
            VaeKernelProfile::AudioOobleck48KhzV1,
        ] {
            let state = oobleck_source_state_shapes(&profile).expect("Oobleck manifest");
            let names = state
                .iter()
                .map(|state| state.name.as_str())
                .collect::<BTreeSet<_>>();
            assert_eq!(names.len(), state.len());
            assert_eq!(state.len(), 365);
            assert!(names.contains("encoder.layers.0.parametrizations.weight.original0"));
            assert!(names.contains("encoder.layers.7.parametrizations.weight.original1"));
            assert!(names.contains("decoder.layers.1.layers.1.parametrizations.weight.original1"));
            assert!(names.contains("decoder.layers.7.parametrizations.weight.original1"));
        }
        let forty_four = oobleck_source_state_shapes(&VaeKernelProfile::AudioOobleck44KhzV1)
            .expect("44.1 kHz manifest");
        let forty_eight = oobleck_source_state_shapes(&VaeKernelProfile::AudioOobleck48KhzV1)
            .expect("48 kHz manifest");
        let forty_four_stride = forty_four
            .iter()
            .find(|state| {
                state.name == "decoder.layers.2.layers.1.parametrizations.weight.original1"
            })
            .expect("44.1 kHz stride checkpoint");
        let forty_eight_stride = forty_eight
            .iter()
            .find(|state| {
                state.name == "decoder.layers.2.layers.1.parametrizations.weight.original1"
            })
            .expect("48 kHz stride checkpoint");
        assert_eq!(forty_four_stride.shape.last(), Some(&16));
        assert_eq!(forty_eight_stride.shape.last(), Some(&12));
    }

    #[test]
    fn oobleck_reduced_equation_uses_loaded_weight_norm_and_snake_parameters()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(8 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(8 << 20)?,
            &cancellation,
        );
        let mut state = BTreeMap::new();
        for activation in ["tiny.layers.0", "tiny.layers.2"] {
            state.insert(
                format!("{activation}.alpha"),
                upload(&backend, &[1], &[0.0], &context)?,
            );
            state.insert(
                format!("{activation}.beta"),
                upload(&backend, &[1], &[0.0], &context)?,
            );
        }
        state.insert(
            "tiny.layers.1.parametrizations.weight.original0".to_owned(),
            upload(&backend, &[1, 1, 1], &[1.0], &context)?,
        );
        state.insert(
            "tiny.layers.1.parametrizations.weight.original1".to_owned(),
            upload(
                &backend,
                &[1, 1, 7],
                &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                &context,
            )?,
        );
        state.insert(
            "tiny.layers.1.bias".to_owned(),
            upload(&backend, &[1], &[0.0], &context)?,
        );
        state.insert(
            "tiny.layers.3.parametrizations.weight.original0".to_owned(),
            upload(&backend, &[1, 1, 1], &[1.0], &context)?,
        );
        state.insert(
            "tiny.layers.3.parametrizations.weight.original1".to_owned(),
            upload(&backend, &[1, 1, 1], &[1.0], &context)?,
        );
        state.insert(
            "tiny.layers.3.bias".to_owned(),
            upload(&backend, &[1], &[0.0], &context)?,
        );
        let mut children = Vec::new();
        push_oobleck_activation_buffers(&mut children, &mut state, "tiny.layers.0")?;
        push_oobleck_convolution(
            &mut children,
            &mut state,
            &backend,
            "tiny.layers.1",
            1,
            1,
            7,
            1,
            3,
            1,
            false,
            true,
            &context,
        )?;
        push_oobleck_activation_buffers(&mut children, &mut state, "tiny.layers.2")?;
        push_oobleck_convolution(
            &mut children,
            &mut state,
            &backend,
            "tiny.layers.3",
            1,
            1,
            1,
            1,
            0,
            1,
            false,
            true,
            &context,
        )?;
        assert!(state.is_empty());
        let mut module = NativeModule::module_dict("tiny-oobleck", children)?;
        module.materialize_execution_state_with_context(
            &backend,
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let source = [0.0_f32, 0.25, -0.5, 1.0];
        let input = upload(&backend, &[1, 1, 4], &source, &context)?;
        let output = oobleck_residual_unit(&module, &backend, &input, "tiny", &context)?;
        let actual = tensor_to_f32_with_backend_exact_native(&backend, &output, &context)?;
        for (actual, source) in actual.into_iter().zip(source) {
            let first = source + source.sin().powi(2);
            let second = first + first.sin().powi(2);
            let expected = second + source;
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn oobleck_reparameterization_is_caller_addressed_exact_and_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let base = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let mean_values = [0.0_f32, 1.0, -2.0, 3.0];
        let mean = upload(&backend, &[1, 1, 4], &mean_values, &base)?;
        let scale = upload(&backend, &[1, 1, 4], &[0.0; 4], &base)?;
        assert!(oobleck_reparameterized_sample(&backend, &backend, &mean, &scale, &base).is_err());

        let address = RngStreamAddress::new(
            "workflow",
            "attempt",
            "vae-encode",
            0,
            "oobleck:seed-42",
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
        let first = oobleck_reparameterized_sample(&backend, &backend, &mean, &scale, &context)?;
        let second = oobleck_reparameterized_sample(&backend, &backend, &mean, &scale, &context)?;
        let first_values = tensor_to_f32_with_backend_exact_native(&backend, &first, &context)?;
        assert_eq!(
            first_values,
            tensor_to_f32_with_backend_exact_native(&backend, &second, &context)?
        );

        let random = randn_like_with_context_exact_native(
            &backend,
            &mean,
            begin_vae_rng(&context)?,
            &context,
        )?;
        let random_values =
            tensor_to_f32_with_backend_exact_native(&backend, &random.tensor, &context)?;
        let standard_deviation = 2.0_f32.ln() + 1.0e-4;
        for ((actual, mean), random) in first_values.iter().zip(mean_values).zip(random_values) {
            let expected = mean + random * standard_deviation;
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }

        let different_address = RngStreamAddress::new(
            "workflow",
            "attempt",
            "vae-encode",
            0,
            "oobleck:seed-43",
            0,
            0,
            RetryRngPolicy::Replay,
        )?;
        let different_context = ExecutionContext {
            stream: base.stream,
            scratch: base.scratch.clone(),
            rng_phase: Some(&different_address),
            cancellation: base.cancellation,
        };
        let different =
            oobleck_reparameterized_sample(&backend, &backend, &mean, &scale, &different_context)?;
        assert_ne!(
            first_values,
            tensor_to_f32_with_backend_exact_native(&backend, &different, &different_context)?
        );
        Ok(())
    }

    #[test]
    fn mmaudio_preprocess_is_source_rate_exact_and_deterministic()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(64 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(64 << 20)?,
            &cancellation,
        );
        let mel_module = |scale: f32| -> Result<NativeModule, Box<dyn std::error::Error>> {
            let mut basis = vec![0.0_f32; 80 * 513];
            for mel in 0..80_usize {
                basis[mel * 513 + 1 + mel * 5] = scale;
            }
            let window = (0..1_024)
                .map(|index| 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / 1_024.0).cos())
                .collect::<Vec<_>>();
            Ok(NativeModule::module_dict(
                "audio-vae:MmAudio16KhzV1",
                vec![
                    NativeModule::buffer(
                        "mel_converter.mel_basis",
                        upload(&backend, &[80, 513], &basis, &context)?,
                    )?,
                    NativeModule::buffer(
                        "mel_converter.hann_window",
                        upload(&backend, &[1_024], &window, &context)?,
                    )?,
                ],
            )?)
        };
        let module = mel_module(1.0)?;
        let samples = 4_410_usize;
        let mut values = Vec::with_capacity(samples * 2);
        for channel in 0..2 {
            for sample in 0..samples {
                let phase = sample as f32 * 440.0 * std::f32::consts::TAU / 44_100.0;
                values.push(phase.sin() * if channel == 0 { 0.5 } else { 0.25 });
            }
        }
        let input = upload(&backend, &[1, 2, samples as u64], &values, &context)?;
        let first = mmaudio_waveform_to_mel(&module, &backend, &input, &context)?;
        let second = mmaudio_waveform_to_mel(&module, &backend, &input, &context)?;
        assert_eq!(first.descriptor().shape(), &[1, 80, 6]);
        assert_eq!(first.contiguous_bytes()?, second.contiguous_bytes()?);
        let changed = mmaudio_waveform_to_mel(&mel_module(0.5)?, &backend, &input, &context)?;
        assert_ne!(first.contiguous_bytes()?, changed.contiguous_bytes()?);
        let values = tensor_to_f32_with_backend_exact_native(&backend, &first, &context)?;
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn music_dcae_preprocess_pads_to_4096_and_emits_128_bin_stereo_mel()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(64 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(64 << 20)?,
            &cancellation,
        );
        let mel_module = |scale: f32| -> Result<NativeModule, Box<dyn std::error::Error>> {
            let mut basis = vec![0.0_f32; 1_025 * 128];
            for mel in 0..128_usize {
                basis[(1 + mel * 7) * 128 + mel] = scale;
            }
            let window = (0..2_048)
                .map(|index| 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / 2_048.0).cos())
                .collect::<Vec<_>>();
            Ok(NativeModule::module_dict(
                "audio-vae:MusicDcaeV1",
                vec![
                    NativeModule::buffer(
                        "vocoder.mel_transform.mel_scale.fb",
                        upload(&backend, &[1_025, 128], &basis, &context)?,
                    )?,
                    NativeModule::buffer(
                        "vocoder.mel_transform.spectrogram.window",
                        upload(&backend, &[2_048], &window, &context)?,
                    )?,
                ],
            )?)
        };
        let module = mel_module(1.0)?;
        let samples = 4_095_usize;
        let values = (0..samples * 2)
            .map(|index| ((index % samples) as f32 * 0.013).sin() * 0.25)
            .collect::<Vec<_>>();
        let input = upload(&backend, &[1, 2, samples as u64], &values, &context)?;
        let first = music_waveform_to_mel_image(&module, &backend, &input, &context)?;
        let second = music_waveform_to_mel_image(&module, &backend, &input, &context)?;
        assert_eq!(first.descriptor().shape(), &[1, 2, 128, 8]);
        assert_eq!(first.contiguous_bytes()?, second.contiguous_bytes()?);
        let changed = music_waveform_to_mel_image(&mel_module(0.5)?, &backend, &input, &context)?;
        assert_ne!(first.contiguous_bytes()?, changed.contiguous_bytes()?);
        let values = tensor_to_f32_with_backend_exact_native(&backend, &first, &context)?;
        assert!(values.iter().all(|value| value.is_finite()));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn mmaudio_convolution_geometry_matches_vae_and_bigvgan_source() -> Result<(), AudioVaeError> {
        let regular = mmaudio_convolution_geometry("vae.encoder.conv_in.weight", 3)?;
        assert_eq!(regular.stride(), &[1]);
        assert_eq!(regular.padding(), &[1]);
        assert_eq!(regular.dilation(), &[1]);
        assert!(!regular.transposed());

        let dilated = mmaudio_convolution_geometry("vocoder.resblocks.0.convs1.2.weight", 11)?;
        assert_eq!(dilated.dilation(), &[5]);
        assert_eq!(dilated.padding(), &[25]);

        let transposed = mmaudio_convolution_geometry("vocoder.ups.1.0.weight", 8)?;
        assert!(transposed.transposed());
        assert_eq!(transposed.stride(), &[4]);
        assert_eq!(transposed.padding(), &[2]);
        Ok(())
    }

    #[test]
    fn mmaudio_manifest_covers_complete_vae_and_bigvgan_state() {
        let state = mmaudio_source_state_shapes();
        let names = state
            .iter()
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(state.len(), 727);
        assert_eq!(names.len(), state.len());
        for required in [
            "mel_converter.mel_basis",
            "vae.encoder.mid.attn_1.qkv.weight",
            "vae.decoder.up.1.upsample.conv.weight",
            "vocoder.ups.5.0.weight",
            "vocoder.resblocks.17.activations.5.act.beta",
            "vocoder.activation_post.downsample.lowpass.filter",
            "vocoder.conv_post.weight",
        ] {
            assert!(names.contains(required), "missing {required}");
        }
    }

    #[test]
    fn music_dcae_manifest_covers_complete_dcae_and_vocoder_topology() {
        let state = music_dcae_source_state_shapes();
        let names = state
            .iter()
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), state.len());
        assert_eq!(state.len(), 905);
        for checkpoint in [
            "dcae.encoder.conv_in.weight",
            "dcae.encoder.down_blocks.3.2.attn.to_q.weight",
            "dcae.encoder.down_blocks.3.2.conv_out.conv_depth.weight",
            "dcae.decoder.up_blocks.0.0.conv.weight",
            "dcae.decoder.up_blocks.3.2.attn.to_out.weight",
            "vocoder.backbone.channel_layers.0.0.weight",
            "vocoder.backbone.stages.2.8.gamma",
            "vocoder.head.conv_pre.parametrizations.weight.original1",
            "vocoder.head.ups.6.parametrizations.weight.original1",
            "vocoder.head.resblocks.27.convs2.2.parametrizations.weight.original1",
            "vocoder.mel_transform.spectrogram.window",
            "vocoder.mel_transform.mel_scale.fb",
        ] {
            assert!(names.contains(checkpoint), "missing {checkpoint}");
        }
    }

    #[test]
    fn music_gelu_matches_source_erf_profile() -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let input = upload(&backend, &[1, 1, 3], &[-1.0, 0.0, 1.0], &context)?;
        let output = music_gelu(&backend, &input, &context)?;
        let actual = tensor_to_f32_with_backend_exact_native(&backend, &output, &context)?;
        for (actual, expected) in actual.into_iter().zip([-0.158_655_26, 0.0, 0.841_344_7]) {
            assert!((actual - expected).abs() < 2.0e-6, "{actual} != {expected}");
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn music_batched_matrix_multiply_preserves_batch_and_head_semantics()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let left = upload(
            &backend,
            &[1, 2, 2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, -1.0, 2.0, 1.0, 0.0],
            &context,
        )?;
        let right = upload(
            &backend,
            &[1, 2, 3, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0, 3.0],
            &context,
        )?;
        let output =
            music_batched_matrix_multiply(&backend, &left, &right, vec![1, 2, 2, 2], &context)?;
        let actual = tensor_to_f32_with_backend_exact_native(&backend, &output, &context)?;
        assert_eq!(actual, vec![4.0, 5.0, 10.0, 11.0, 2.0, -2.0, 5.0, 2.0]);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn mmaudio_reduced_residual_uses_loaded_convolutions_and_source_magnitude_preservation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(2 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(2 << 20)?,
            &cancellation,
        );
        let mut state = BTreeMap::new();
        for name in ["tiny.conv1.weight", "tiny.conv2.weight"] {
            state.insert(
                name.to_owned(),
                upload(&backend, &[1, 1, 3], &[0.0, 1.0, 0.0], &context)?,
            );
        }
        let module = build_mmaudio_module(
            &VaeKernelProfile::MmAudio16KhzV1,
            state,
            &backend,
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let source = [-2.0_f32, -0.5, 0.25, 3.0];
        let input = upload(&backend, &[1, 1, 4], &source, &context)?;
        let output = mmaudio_residual_block(&module, &backend, &input, "tiny", &context)?;
        let actual = tensor_to_f32_with_backend_exact_native(&backend, &output, &context)?;
        let denominator = (0.7_f32.powi(2) + 0.3_f32.powi(2)).sqrt();
        for (actual, source) in actual.into_iter().zip(source) {
            let normalized = source / (source.abs() + 1.0e-4);
            let first = normalized / (1.0 + (-normalized).exp()) / 0.596;
            let second = first / (1.0 + (-first).exp()) / 0.596;
            let expected = (0.7 * normalized + 0.3 * second) / denominator;
            assert!((actual - expected).abs() < 2.0e-5, "{actual} != {expected}");
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn music_reduced_residual_uses_loaded_convolutions_and_rms_parameters()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(2 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(2 << 20)?,
            &cancellation,
        );
        let mut state = BTreeMap::new();
        for name in ["tiny.conv1.weight", "tiny.conv2.weight"] {
            state.insert(
                name.to_owned(),
                upload(&backend, &[1, 1, 1, 1], &[1.0], &context)?,
            );
        }
        state.insert(
            "tiny.norm.weight".to_owned(),
            upload(&backend, &[1], &[1.5], &context)?,
        );
        state.insert(
            "tiny.norm.bias".to_owned(),
            upload(&backend, &[1], &[-0.25], &context)?,
        );
        let module = build_dense_audio_module(
            &VaeKernelProfile::MusicDcaeV1,
            state,
            &backend,
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let source = [-1.0_f32, -0.25, 0.5, 2.0];
        let input = upload(&backend, &[1, 1, 2, 2], &source, &context)?;
        let output = music_dcae_residual(&module, &backend, &input, "tiny", &context)?;
        let actual = tensor_to_f32_with_backend_exact_native(&backend, &output, &context)?;
        for (actual, source) in actual.into_iter().zip(source) {
            let hidden = source / (1.0 + (-source).exp());
            let normalized = hidden / (hidden * hidden + 1.0e-5).sqrt();
            let expected = source + normalized * 1.5 - 0.25;
            assert!((actual - expected).abs() < 2.0e-5, "{actual} != {expected}");
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn ltx_audio_configuration_and_topology_are_exact_and_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let autoencoder = r#"{
            "model":{"params":{"ddconfig":{
                "double_z":true,"mel_bins":64,"z_channels":8,"resolution":256,
                "in_channels":2,"out_ch":2,"ch":128,"ch_mult":[1,2,4],
                "num_res_blocks":2,"attn_resolutions":[],"dropout":0.0,
                "mid_block_add_attention":false,"norm_type":"pixel",
                "causality_axis":"height"
            },"sampling_rate":16000}},
            "preprocessing":{"stft":{"filter_length":1024,"hop_length":160}}
        }"#;
        let vocoder = r#"{"upsample_rates":[5,4,2,2,2]}"#;
        validate_ltx_audio_configuration(autoencoder, vocoder, 8, 16_000, 16_000)?;
        assert!(
            validate_ltx_audio_configuration(autoencoder, vocoder, 16, 16_000, 16_000).is_err()
        );
        assert!(
            validate_ltx_audio_configuration(
                autoencoder,
                r#"{"vocoder":{},"bwe":{}}"#,
                8,
                16_000,
                16_000,
            )
            .is_err()
        );
        for unsupported in [
            r#"{"upsample_rates":[5,4,2,2,2],"resblock_dilation_sizes":[[1,3],[1,3,5],[1,3,5]]}"#,
            r#"{"upsample_rates":[5,4,2,2,2],"apply_final_activation":false}"#,
            r#"{"upsample_rates":[5,4,2,2,2],"use_tanh_at_final":false}"#,
            r#"{"upsample_rates":[5,4,2,2,2],"output_sample_rate":24000}"#,
        ] {
            assert!(
                validate_ltx_audio_configuration(autoencoder, unsupported, 8, 16_000, 16_000,)
                    .is_err(),
                "unsupported LTX vocoder configuration was admitted: {unsupported}"
            );
        }

        let state = ltx_audio_source_state_shapes();
        let names = state
            .iter()
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(state.len(), 296);
        assert_eq!(names.len(), state.len());
        for checkpoint in [
            "autoencoder.encoder.conv_in.conv.weight",
            "autoencoder.encoder.down.1.block.0.nin_shortcut.conv.weight",
            "autoencoder.encoder.down.1.downsample.conv.weight",
            "autoencoder.decoder.up.2.upsample.conv.conv.weight",
            "autoencoder.per_channel_statistics.std-of-means",
            "vocoder.conv_pre.weight",
            "vocoder.ups.4.weight",
            "vocoder.resblocks.14.convs2.2.weight",
            "vocoder.conv_post.bias",
        ] {
            assert!(names.contains(checkpoint), "missing {checkpoint}");
        }
        Ok(())
    }

    #[test]
    fn ltx_audio_preprocess_is_source_rate_exact_and_deterministic()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(64 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(64 << 20)?,
            &cancellation,
        );
        let samples = 44_100_usize;
        let values = (0..samples * 2)
            .map(|index| {
                let sample = index % samples;
                (sample as f32 * 330.0 * std::f32::consts::TAU / 44_100.0).sin() * 0.25
            })
            .collect::<Vec<_>>();
        let input = upload(&backend, &[1, 2, samples as u64], &values, &context)?;
        let first = ltx_waveform_to_mel(&backend, &input, &context)?;
        let second = ltx_waveform_to_mel(&backend, &input, &context)?;
        assert_eq!(first.descriptor().shape(), &[1, 2, 101, 64]);
        assert_eq!(first.contiguous_bytes()?, second.contiguous_bytes()?);
        let values = tensor_to_f32_with_backend_exact_native(&backend, &first, &context)?;
        assert!(values.iter().all(|value| value.is_finite()));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn ltx_audio_latent_statistics_round_trip_patch_space() -> Result<(), Box<dyn std::error::Error>>
    {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(4 << 20)?,
            &cancellation,
        );
        let mean = upload(
            &backend,
            &[128],
            &(0..128)
                .map(|index| index as f32 * 0.001)
                .collect::<Vec<_>>(),
            &context,
        )?;
        let standard_deviation = upload(
            &backend,
            &[128],
            &(0..128)
                .map(|index| 1.0 + index as f32 * 0.002)
                .collect::<Vec<_>>(),
            &context,
        )?;
        let module = NativeModule::module_dict(
            "ltx-statistics",
            vec![
                NativeModule::buffer("autoencoder.per_channel_statistics.mean-of-means", mean)?,
                NativeModule::buffer(
                    "autoencoder.per_channel_statistics.std-of-means",
                    standard_deviation,
                )?,
            ],
        )?;
        let values = (0..8 * 3 * 16)
            .map(|index| index as f32 * 0.003 - 0.5)
            .collect::<Vec<_>>();
        let input = upload(&backend, &[1, 8, 3, 16], &values, &context)?;
        let normalized = ltx_normalize_latent(&module, &backend, &input, false, &context)?;
        let restored = ltx_normalize_latent(&module, &backend, &normalized, true, &context)?;
        let restored = tensor_to_f32_with_backend_exact_native(&backend, &restored, &context)?;
        for (actual, expected) in restored.into_iter().zip(values) {
            assert!((actual - expected).abs() < 2.0e-6, "{actual} != {expected}");
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn ltx_reduced_residual_uses_loaded_causal_convolutions_and_pixel_norm()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(2 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(2 << 20)?,
            &cancellation,
        );
        let mut causal_identity = vec![0.0_f32; 9];
        causal_identity[7] = 1.0;
        let mut state = BTreeMap::new();
        for name in ["tiny.conv1.conv.weight", "tiny.conv2.conv.weight"] {
            state.insert(
                name.to_owned(),
                upload(&backend, &[1, 1, 3, 3], &causal_identity, &context)?,
            );
        }
        let module = build_dense_audio_module(
            &VaeKernelProfile::LtxAudioV1,
            state,
            &backend,
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let source = [-2.0_f32, -0.5, 0.25, 3.0];
        let input = upload(&backend, &[1, 1, 2, 2], &source, &context)?;
        let output = ltx_residual_block(&module, &backend, &input, "tiny", &context)?;
        let actual = tensor_to_f32_with_backend_exact_native(&backend, &output, &context)?;
        for (actual, source) in actual.into_iter().zip(source) {
            let first_norm = source / (source * source + 1.0e-6).sqrt();
            let first = first_norm / (1.0 + (-first_norm).exp());
            let second_norm = first / (first * first + 1.0e-6).sqrt();
            let second = second_norm / (1.0 + (-second_norm).exp());
            let expected = source + second;
            assert!((actual - expected).abs() < 2.0e-5, "{actual} != {expected}");
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn stable_audio_3_manifests_cover_both_exact_transformer_profiles() {
        for (profile, expected, last_transformer, channels, decoder_kernel) in [
            (VaeKernelProfile::StableAudio3DeepV1, 472, 11, 1_536, 1),
            (VaeKernelProfile::StableAudio3ShallowV1, 244, 5, 768, 3),
        ] {
            let state = stable_audio_3_source_state_shapes(&profile).expect("SA3 manifest");
            let names = state
                .iter()
                .map(|state| state.name.as_str())
                .collect::<BTreeSet<_>>();
            assert_eq!(names.len(), state.len());
            assert_eq!(state.len(), expected);
            for checkpoint in [
                "encoder.layers.0.mapping.parametrizations.weight.original1".to_owned(),
                format!("encoder.layers.0.transformers.{last_transformer}.self_attn.to_out.weight"),
                "encoder.layers.2.weight".to_owned(),
                "decoder.layers.1.weight".to_owned(),
                format!("decoder.layers.3.transformers.{last_transformer}.ff.ff.0.proj.weight"),
                "bottleneck.running_std".to_owned(),
            ] {
                assert!(names.contains(checkpoint.as_str()), "missing {checkpoint}");
            }
            let mapping = state
                .iter()
                .find(|state| {
                    state.name == "decoder.layers.3.mapping.parametrizations.weight.original1"
                })
                .expect("decoder mapping");
            assert_eq!(mapping.shape, vec![512, channels, decoder_kernel]);
        }
    }

    #[test]
    fn stable_audio_3_reduced_dynamic_tanh_uses_loaded_affine_parameters()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let module = NativeModule::module_dict(
            "audio-vae:StableAudio3ShallowV1",
            vec![
                NativeModule::buffer("tiny.alpha", upload(&backend, &[1], &[0.5], &context)?)?,
                NativeModule::buffer(
                    "tiny.gamma",
                    upload(&backend, &[2], &[2.0, -1.5], &context)?,
                )?,
                NativeModule::buffer(
                    "tiny.beta",
                    upload(&backend, &[2], &[0.25, -0.5], &context)?,
                )?,
            ],
        )?;
        let source = [-2.0_f32, -1.0, 0.5, 3.0];
        let input = upload(&backend, &[1, 2, 2], &source, &context)?;
        let output = sa3_dynamic_tanh(&module, &backend, &input, "tiny", &context)?;
        let actual = tensor_to_f32_with_backend_exact_native(&backend, &output, &context)?;
        for (index, (actual, source)) in actual.into_iter().zip(source).enumerate() {
            let channel = index % 2;
            let gamma = [2.0_f32, -1.5][channel];
            let beta = [0.25_f32, -0.5][channel];
            let expected = (source * 0.5).tanh() * gamma + beta;
            assert!((actual - expected).abs() < 2.0e-6, "{actual} != {expected}");
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn stable_audio_3_patch_pretransform_round_trips_padded_waveform()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(2 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(2 << 20)?,
            &cancellation,
        );
        let values = (0..514)
            .map(|index| index as f32 / 514.0)
            .collect::<Vec<_>>();
        let input = upload(&backend, &[1, 2, 257], &values, &context)?;
        let patched = sa3_pretransform_encode(&backend, &input, &context)?;
        assert_eq!(patched.descriptor().shape(), &[1, 512, 2]);
        let decoded = sa3_pretransform_decode(&backend, &patched, &context)?;
        assert_eq!(decoded.descriptor().shape(), &[1, 2, 512]);
        let decoded = tensor_to_f32_with_backend_exact_native(&backend, &decoded, &context)?;
        assert_eq!(&decoded[..257], &values[..257]);
        assert!(decoded[257..512].iter().all(|value| *value == 0.0));
        assert_eq!(&decoded[512..512 + 257], &values[257..]);
        assert!(decoded[512 + 257..].iter().all(|value| *value == 0.0));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn stable_audio_3_sliding_attention_is_window_exact_and_allocation_bounded()
    -> Result<(), Box<dyn std::error::Error>> {
        let sequence = 1_000_000_u64;
        let window = 17_u64;
        let mut query_start = 0_u64;
        while query_start < sequence {
            let (query_length, _key_start, key_length) =
                sa3_attention_tile_geometry(sequence, window, query_start)?;
            assert!(query_length <= SA3_ATTENTION_QUERY_TILE);
            assert!(key_length <= SA3_ATTENTION_QUERY_TILE + window * 2);
            assert!(query_length * key_length <= 64 * 98);
            query_start += query_length;
        }

        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(2 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(2 << 20)?,
            &cancellation,
        );
        let query_values = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5, 0.25, -0.5];
        let key_values = [0.5_f32, 0.0, 0.0, 0.5, 0.5, 0.5, -0.5, 0.25, 0.1, -0.2];
        let value_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let query = upload(&backend, &[1, 1, 5, 2], &query_values, &context)?;
        let key = upload(&backend, &[1, 1, 5, 2], &key_values, &context)?;
        let value = upload(&backend, &[1, 1, 5, 2], &value_values, &context)?;
        let output = sa3_attention_once(&backend, &query, &key, &value, Some(1), &context)?;
        let actual = tensor_to_f32_with_backend_exact_native(&backend, &output, &context)?;
        let scale = 2.0_f32.sqrt().recip();
        let mut expected = Vec::with_capacity(10);
        for query_index in 0..5_usize {
            let start = query_index.saturating_sub(1);
            let end = (query_index + 2).min(5);
            let query = &query_values[query_index * 2..query_index * 2 + 2];
            let scores = (start..end)
                .map(|key_index| {
                    let key = &key_values[key_index * 2..key_index * 2 + 2];
                    (query[0] * key[0] + query[1] * key[1]) * scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = exponentials.iter().sum::<f32>();
            for channel in 0..2_usize {
                expected.push(
                    (start..end)
                        .zip(exponentials.iter())
                        .map(|(key_index, exponential)| {
                            exponential * value_values[key_index * 2 + channel] / denominator
                        })
                        .sum::<f32>(),
                );
            }
        }
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-5, "{actual} != {expected}");
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn missing_audio_state_fails_closed_instead_of_publishing_shape_facades()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(32 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(32 << 20)?,
            &cancellation,
        );
        for (name, samples, latent_shape) in [
            ("audio-vae:MusicDcaeV1", 4_096, vec![1, 8, 1, 16]),
            ("audio-vae:MmAudio16KhzV1", 44_100, vec![1, 20, 32]),
            ("audio-vae:LtxAudioV1", 4_096, vec![1, 8, 1, 16]),
            ("audio-vae:StableAudio3DeepV1", 4_096, vec![1, 256, 1]),
            ("audio-vae:StableAudio3ShallowV1", 4_096, vec![1, 256, 1]),
        ] {
            let module = NativeModule::module_dict(name, Vec::new())?;
            let values = (0..samples * 2)
                .map(|index| (index % 97) as f32 / 97.0)
                .collect::<Vec<_>>();
            let input = upload(&backend, &[1, 2, samples as u64], &values, &context)?;
            assert!(
                audio_encode_tensor(&module, &backend, Some(&backend), &input, &context).is_err()
            );
            let latent = upload(
                &backend,
                &latent_shape,
                &vec![0.0; latent_shape.iter().product::<u64>() as usize],
                &context,
            )?;
            assert!(
                audio_decode_tensor(&module, &backend, Some(&backend), &latent, &context).is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn native_audio_equations_observe_structural_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let input = upload(&backend, &[1, 2, 4_096], &vec![0.0; 8_192], &context)?;
        let module = NativeModule::module_dict("audio-vae:MusicDcaeV1", Vec::new())?;
        cancellation.cancel();
        assert!(audio_encode_tensor(&module, &backend, Some(&backend), &input, &context).is_err());
        Ok(())
    }

    #[test]
    fn audio_workspace_oom_is_typed_atomic_and_caller_retry_is_deterministic()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let normal_scratch = workspace.authorize_workspace(1 << 20)?;
        let normal_context =
            backend.execution_context(StreamId::DEFAULT, normal_scratch.clone(), &cancellation);
        let input = upload(
            &backend,
            &[1, 2, 257],
            &vec![0.25; 2 * 257],
            &normal_context,
        )?;
        let baseline = sa3_zero_pad_last_to_multiple(&backend, &input, 256, &normal_context)?;
        let baseline_bytes = baseline.contiguous_bytes()?;
        assert_eq!(normal_scratch.in_use_bytes(), 0);

        let insufficient_scratch = workspace.authorize_workspace(1)?;
        let insufficient_context = backend.execution_context(
            StreamId::DEFAULT,
            insufficient_scratch.clone(),
            &cancellation,
        );
        assert!(matches!(
            sa3_zero_pad_last_to_multiple(&backend, &input, 256, &insufficient_context),
            Err(VaeError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(insufficient_scratch.in_use_bytes(), 0);

        let retry = sa3_zero_pad_last_to_multiple(&backend, &input, 256, &normal_context)?;
        assert_eq!(retry.contiguous_bytes()?, baseline_bytes);
        assert_eq!(normal_scratch.in_use_bytes(), 0);
        Ok(())
    }
}
