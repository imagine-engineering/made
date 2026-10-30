use serde::Deserialize;
use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, Layout,
    MemoryFormatReference, NativeLatentBundle, NativeLatentBundleError, NativeLatentMetadata,
    NativeLatentSamples, StorageId, StreamId, Tensor, TensorError,
    generated_activation_normalization_functional_01::{
        group_norm_tensor_with_context_exact_native, normalize_with_context_exact_native,
        silu_tensor_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_03::real_add_with_context_exact_native,
    generated_indexing_masking_01::narrow_function_exact_native,
    generated_neural_network_functional_01::pixel_shuffle_tensor_with_context_exact_native,
    generated_shape_layout_transform_01::{
        RepeatInterleaveSpec, tensor_repeat_interleave_with_context_exact_native,
    },
    generated_shape_layout_transform_02::tensor_reshape_with_context_exact_native,
    generated_shape_layout_transform_03::{
        FunctionalPadMode, functional_pad_with_context_exact_native, tensor_permute_exact_native,
    },
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        ConvolutionConfiguration, InterpolateConfiguration, InterpolateMode,
        bislerp_tensor_with_context_exact_native, conv_2d_tensor_with_context_exact_native,
        conv_3d_tensor_with_context_exact_native, interpolate_tensor_with_context_exact_native,
        pixel_shuffle_nd_tensor_with_context_exact_native,
    },
    generated_storage_dtype_device_01::contiguous_with_context_exact_native,
};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::{LtxLatentStatisticsDirection, NativeVae};

pub const NODES_HUNYUAN_SOURCE_SHA256: &str =
    "028abde8d150dfe890ae987d8f87b0df439a3e9d068d7c63400d7f442dc4b7d7";
pub const HUNYUAN_UPSAMPLER_SOURCE_SHA256: &str =
    "f667649504185a70023cd6f51033d904d08da6455bc198d613a7e1faa802ce75";
pub const LTX_UPSAMPLER_SOURCE_SHA256: &str =
    "04d36045252d475daf4dfa058e77b5327616595e66012da8c81fce254e962207";

const HUNYUAN_720_MARKER: &str = "blocks.0.block.0.conv.weight";
const HUNYUAN_1080_MARKER: &str = "up.0.block.0.conv1.conv.weight";
const LTX_MARKER: &str = "post_upsample_res_blocks.0.conv2.bias";
const MAX_STATE_TENSORS: usize = 8_192;
const MAX_STATE_KEY_BYTES: usize = 1_024;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeLatentUpscaleArchitecture {
    Hunyuan720p,
    Hunyuan1080p,
    Ltx,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HunyuanLatentUpscaleMode {
    NearestExact,
    Bilinear,
    Area,
    Bicubic,
    Bislerp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HunyuanLatentUpscaleCrop {
    Disabled,
    Center,
}

impl NativeLatentUpscaleArchitecture {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Hunyuan720p => "hunyuan-video-1.5-720p",
            Self::Hunyuan1080p => "hunyuan-video-1.5-1080p",
            Self::Ltx => "ltx-latent-upsampler",
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeLatentUpscaleCheckpoint {
    pub artifact_sha256: String,
    pub metadata: BTreeMap<String, String>,
    pub ordered_state: Vec<(String, Tensor)>,
    pub memory_budget_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct NativeLtxUpscaleConfiguration {
    pub in_channels: usize,
    pub mid_channels: usize,
    pub num_blocks_per_stage: usize,
    pub dims: usize,
    pub spatial_upsample: bool,
    pub temporal_upsample: bool,
    pub spatial_scale_milli: u32,
    pub rational_resampler: bool,
}

#[derive(Deserialize)]
#[serde(default)]
struct NativeLtxUpscaleConfigurationWire {
    #[serde(rename = "_class_name")]
    _class_name: Option<String>,
    in_channels: usize,
    mid_channels: usize,
    num_blocks_per_stage: usize,
    dims: usize,
    spatial_upsample: bool,
    temporal_upsample: bool,
    spatial_scale: f64,
    rational_resampler: bool,
}

impl Default for NativeLtxUpscaleConfigurationWire {
    fn default() -> Self {
        let config = NativeLtxUpscaleConfiguration::default();
        Self {
            _class_name: None,
            in_channels: config.in_channels,
            mid_channels: config.mid_channels,
            num_blocks_per_stage: config.num_blocks_per_stage,
            dims: config.dims,
            spatial_upsample: config.spatial_upsample,
            temporal_upsample: config.temporal_upsample,
            spatial_scale: f64::from(config.spatial_scale_milli) / 1_000.0,
            rational_resampler: config.rational_resampler,
        }
    }
}

impl Default for NativeLtxUpscaleConfiguration {
    fn default() -> Self {
        Self {
            in_channels: 4,
            mid_channels: 128,
            num_blocks_per_stage: 4,
            dims: 2,
            spatial_upsample: true,
            temporal_upsample: false,
            spatial_scale_milli: 2_000,
            rational_resampler: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NativeLatentUpscaleProfile {
    Hunyuan720p {
        in_channels: usize,
        out_channels: usize,
        hidden_channels: usize,
        blocks: usize,
    },
    Hunyuan1080p {
        z_channels: usize,
        out_channels: usize,
        block_out_channels: Vec<usize>,
    },
    Ltx(NativeLtxUpscaleConfiguration),
}

#[derive(Clone, Debug)]
pub struct NativeLatentUpscaleModelResource {
    architecture: NativeLatentUpscaleArchitecture,
    profile: NativeLatentUpscaleProfile,
    artifact_sha256: String,
    state: BTreeMap<String, Tensor>,
    stream: StreamId,
    memory_budget_bytes: u64,
    resident_bytes: u64,
    semantic_digest_sha256: String,
}

impl NativeLatentUpscaleModelResource {
    pub fn from_checkpoint(
        checkpoint: NativeLatentUpscaleCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentUpscaleModelError> {
        context.check()?;
        validate_sha256(&checkpoint.artifact_sha256)?;
        if checkpoint.ordered_state.is_empty()
            || checkpoint.ordered_state.len() > MAX_STATE_TENSORS
            || checkpoint.memory_budget_bytes == 0
        {
            return Err(NativeLatentUpscaleModelError::InvalidCheckpoint(
                "state cardinality or memory budget is invalid".to_owned(),
            ));
        }
        let mut state = BTreeMap::new();
        for (index, (key, tensor)) in checkpoint.ordered_state.into_iter().enumerate() {
            if index.is_multiple_of(32) {
                context.check()?;
            }
            validate_state_key(&key)?;
            if state.insert(key.clone(), tensor).is_some() {
                return Err(NativeLatentUpscaleModelError::DuplicateStateKey(key));
            }
        }
        let architecture = if state.contains_key(HUNYUAN_720_MARKER) {
            NativeLatentUpscaleArchitecture::Hunyuan720p
        } else if state.contains_key(HUNYUAN_1080_MARKER) {
            NativeLatentUpscaleArchitecture::Hunyuan1080p
        } else if state.contains_key(LTX_MARKER) {
            NativeLatentUpscaleArchitecture::Ltx
        } else {
            return Err(NativeLatentUpscaleModelError::UnsupportedArchitecture);
        };
        if architecture == NativeLatentUpscaleArchitecture::Hunyuan1080p {
            state = normalize_1080_shortcuts(state, context.cancellation)?;
        }
        let profile = infer_profile(architecture, &state, &checkpoint.metadata)?;
        let expected = expected_state_shapes(&profile)?;
        validate_strict_state(&state, &expected, context)?;
        let semantic_digest_sha256 = semantic_digest(
            architecture,
            &profile,
            &checkpoint.artifact_sha256,
            &state,
            context.cancellation,
        )?;
        let resident_bytes = resident_state_bytes(&state, context.cancellation)?
            .checked_add(resident_owned_bytes(
                &checkpoint.artifact_sha256,
                &semantic_digest_sha256,
                &state,
            )?)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        if resident_bytes > checkpoint.memory_budget_bytes {
            return Err(NativeLatentUpscaleModelError::OutOfMemory {
                required: resident_bytes,
                budget: checkpoint.memory_budget_bytes,
            });
        }
        context.check()?;
        Ok(Self {
            architecture,
            profile,
            artifact_sha256: checkpoint.artifact_sha256,
            state,
            stream: context.stream,
            memory_budget_bytes: checkpoint.memory_budget_bytes,
            resident_bytes,
            semantic_digest_sha256,
        })
    }

    pub const fn architecture(&self) -> NativeLatentUpscaleArchitecture {
        self.architecture
    }

    pub const fn identifier(&self) -> &'static str {
        self.architecture.identifier()
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub const fn memory_budget_bytes(&self) -> u64 {
        self.memory_budget_bytes
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, NativeLatentUpscaleModelError> {
        resident_owned_bytes(
            &self.artifact_sha256,
            &self.semantic_digest_sha256,
            &self.state,
        )
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(StorageId, u64)>, NativeLatentUpscaleModelError> {
        let mut seen = BTreeSet::new();
        let mut allocations = Vec::new();
        allocations
            .try_reserve_exact(self.state.len())
            .map_err(|_| NativeLatentUpscaleModelError::Allocation)?;
        for tensor in self.state.values() {
            if seen.insert(tensor.storage_id().get()) {
                allocations.push((tensor.storage_id(), tensor.storage_byte_len()));
            }
        }
        Ok(allocations)
    }

    pub fn state(&self) -> &BTreeMap<String, Tensor> {
        &self.state
    }

    pub fn validate(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeLatentUpscaleModelError> {
        cancellation.check()?;
        validate_sha256(&self.artifact_sha256)?;
        let expected = expected_state_shapes(&self.profile)?;
        validate_strict_state_retained(&self.state, &expected, Some(self.stream), cancellation)?;
        if resident_state_bytes(&self.state, cancellation)?
            .checked_add(self.resident_owned_bytes()?)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?
            != self.resident_bytes
            || semantic_digest(
                self.architecture,
                &self.profile,
                &self.artifact_sha256,
                &self.state,
                cancellation,
            )? != self.semantic_digest_sha256
        {
            return Err(NativeLatentUpscaleModelError::SemanticStateChanged);
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn invoke_hunyuan_720p(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeLatentUpscaleModelError> {
        self.normalize_execution_result(
            self.invoke_hunyuan_720p_inner(backend, input, context),
            context.cancellation,
        )
    }

    fn invoke_hunyuan_720p_inner(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeLatentUpscaleModelError> {
        context.check()?;
        let NativeLatentUpscaleProfile::Hunyuan720p {
            in_channels,
            out_channels,
            hidden_channels,
            blocks,
        } = &self.profile
        else {
            return Err(NativeLatentUpscaleModelError::CrossVariantInvocation);
        };
        self.authorize_execution_shape(
            input,
            *in_channels,
            (*hidden_channels).max(*out_channels),
            1,
            1,
            8,
        )?;
        validate_input_tensor(input, *in_channels, context)?;
        let mut hidden = causal_video_convolution(
            backend,
            input,
            self.state_tensor("in_conv.conv.weight")?,
            self.state_tensor("in_conv.conv.bias")?,
            context,
        )?;
        for block in 0..*blocks {
            context.check()?;
            let residual = hidden.clone();
            hidden = causal_video_convolution(
                backend,
                &hidden,
                self.state_tensor(&format!("blocks.{block}.block.0.conv.weight"))?,
                self.state_tensor(&format!("blocks.{block}.block.0.conv.bias"))?,
                context,
            )?;
            hidden = silu_tensor_with_context_exact_native(backend, &hidden, context)
                .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
            hidden = causal_video_convolution(
                backend,
                &hidden,
                self.state_tensor(&format!("blocks.{block}.block.2.conv.weight"))?,
                self.state_tensor(&format!("blocks.{block}.block.2.conv.bias"))?,
                context,
            )?;
            hidden = silu_tensor_with_context_exact_native(backend, &hidden, context)
                .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
            hidden = causal_video_convolution(
                backend,
                &hidden,
                self.state_tensor(&format!("blocks.{block}.block.4.conv.weight"))?,
                self.state_tensor(&format!("blocks.{block}.block.4.conv.bias"))?,
                context,
            )?;
            hidden = real_add_with_context_exact_native(backend, &residual, &hidden, context)
                .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
        }
        let output = causal_video_convolution(
            backend,
            &hidden,
            self.state_tensor("out_conv.conv.weight")?,
            self.state_tensor("out_conv.conv.bias")?,
            context,
        )?;
        context.check()?;
        Ok(output)
    }

    pub fn invoke_hunyuan_1080p(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeLatentUpscaleModelError> {
        self.normalize_execution_result(
            self.invoke_hunyuan_1080p_inner(backend, input, context),
            context.cancellation,
        )
    }

    fn invoke_hunyuan_1080p_inner(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeLatentUpscaleModelError> {
        context.check()?;
        let NativeLatentUpscaleProfile::Hunyuan1080p {
            z_channels,
            out_channels,
            block_out_channels,
        } = &self.profile
        else {
            return Err(NativeLatentUpscaleModelError::CrossVariantInvocation);
        };
        let peak_channels = block_out_channels
            .iter()
            .copied()
            .max()
            .unwrap_or(*out_channels)
            .max(*out_channels);
        self.authorize_execution_shape(input, *z_channels, peak_channels, 1, 1, 10)?;
        validate_input_tensor(input, *z_channels, context)?;
        let convolved = causal_video_convolution(
            backend,
            input,
            self.state_tensor("conv_in.conv.weight")?,
            self.state_tensor("conv_in.conv.bias")?,
            context,
        )?;
        let repeat = block_out_channels[0]
            .checked_div(*z_channels)
            .filter(|repeat| *repeat > 0 && repeat * *z_channels == block_out_channels[0])
            .ok_or_else(|| {
                NativeLatentUpscaleModelError::InvalidInput(
                    "1080p first width must be a multiple of latent channels".to_owned(),
                )
            })?;
        let repeated = tensor_repeat_interleave_with_context_exact_native(
            backend,
            input,
            RepeatInterleaveSpec::Scalar(u64_from(repeat)?),
            Some(1),
            Some(u64_from(block_out_channels[0])?),
            context,
        )
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
        let mut hidden =
            real_add_with_context_exact_native(backend, &convolved, &repeated, context)
                .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
        let mut input_channels = block_out_channels[0];
        for (stage, output_channels) in block_out_channels.iter().copied().enumerate() {
            for block in 0..3 {
                context.check()?;
                let prefix = format!("up.{stage}.block.{block}");
                let residual = if input_channels == output_channels {
                    hidden.clone()
                } else {
                    causal_video_convolution(
                        backend,
                        &hidden,
                        self.state_tensor(&format!("{prefix}.nin_shortcut.conv.weight"))?,
                        self.state_tensor(&format!("{prefix}.nin_shortcut.conv.bias"))?,
                        context,
                    )?
                };
                hidden = hunyuan_rms_norm(
                    backend,
                    &hidden,
                    self.state_tensor(&format!("{prefix}.norm1.gamma"))?,
                    context,
                )?;
                hidden = silu_tensor_with_context_exact_native(backend, &hidden, context)
                    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
                hidden = causal_video_convolution(
                    backend,
                    &hidden,
                    self.state_tensor(&format!("{prefix}.conv1.conv.weight"))?,
                    self.state_tensor(&format!("{prefix}.conv1.conv.bias"))?,
                    context,
                )?;
                hidden = hunyuan_rms_norm(
                    backend,
                    &hidden,
                    self.state_tensor(&format!("{prefix}.norm2.gamma"))?,
                    context,
                )?;
                hidden = silu_tensor_with_context_exact_native(backend, &hidden, context)
                    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
                hidden = causal_video_convolution(
                    backend,
                    &hidden,
                    self.state_tensor(&format!("{prefix}.conv2.conv.weight"))?,
                    self.state_tensor(&format!("{prefix}.conv2.conv.bias"))?,
                    context,
                )?;
                hidden = real_add_with_context_exact_native(backend, &residual, &hidden, context)
                    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
                input_channels = output_channels;
            }
        }
        hidden = hunyuan_rms_norm(
            backend,
            &hidden,
            self.state_tensor("norm_out.gamma")?,
            context,
        )?;
        hidden = silu_tensor_with_context_exact_native(backend, &hidden, context)
            .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
        causal_video_convolution(
            backend,
            &hidden,
            self.state_tensor("conv_out.conv.weight")?,
            self.state_tensor("conv_out.conv.bias")?,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke_hunyuan_bundle(
        &self,
        backend: &CpuBackend,
        input: &NativeLatentBundle,
        width: u32,
        height: u32,
        mode: HunyuanLatentUpscaleMode,
        crop: HunyuanLatentUpscaleCrop,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentBundle, NativeLatentUpscaleModelError> {
        self.normalize_execution_result(
            self.invoke_hunyuan_bundle_inner(backend, input, width, height, mode, crop, context),
            context.cancellation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn invoke_hunyuan_bundle_inner(
        &self,
        backend: &CpuBackend,
        input: &NativeLatentBundle,
        width: u32,
        height: u32,
        mode: HunyuanLatentUpscaleMode,
        crop: HunyuanLatentUpscaleCrop,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentBundle, NativeLatentUpscaleModelError> {
        context.check()?;
        if width > 16_384 || height > 16_384 {
            return Err(NativeLatentUpscaleModelError::InvalidInput(
                "Hunyuan dimensions exceed 16384".to_owned(),
            ));
        }
        if width == 0 && height == 0 {
            return Ok(input.clone());
        }
        let NativeLatentSamples::Tensor(samples) = input.samples() else {
            return Err(NativeLatentUpscaleModelError::InvalidInput(
                "Hunyuan requires a single tensor latent".to_owned(),
            ));
        };
        let [_, _, _, source_height, source_width] = samples.descriptor().shape() else {
            return Err(NativeLatentUpscaleModelError::InvalidInput(
                "Hunyuan latent rank".to_owned(),
            ));
        };
        let (width, height) =
            resolve_hunyuan_dimensions(width, height, *source_width, *source_height)?;
        let target_width = u64::from(width / 16);
        let target_height = u64::from(height / 16);
        self.authorize_resize_execution(samples, target_width, target_height, mode)?;
        let resized = common_upscale_video(
            backend,
            samples,
            target_width,
            target_height,
            mode,
            crop,
            context,
        )?;
        let output = match self.architecture {
            NativeLatentUpscaleArchitecture::Hunyuan720p => {
                self.invoke_hunyuan_720p(backend, &resized, context)?
            }
            NativeLatentUpscaleArchitecture::Hunyuan1080p => {
                self.invoke_hunyuan_1080p(backend, &resized, context)?
            }
            NativeLatentUpscaleArchitecture::Ltx => {
                return Err(NativeLatentUpscaleModelError::CrossVariantInvocation);
            }
        };
        context.check()?;
        NativeLatentBundle::single(output, None, None, NativeLatentMetadata::default(), context)
            .map_err(NativeLatentUpscaleModelError::Latent)
    }

    pub fn invoke_ltx(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeLatentUpscaleModelError> {
        self.normalize_execution_result(
            self.invoke_ltx_inner(backend, input, context),
            context.cancellation,
        )
    }

    fn invoke_ltx_inner(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeLatentUpscaleModelError> {
        context.check()?;
        let NativeLatentUpscaleProfile::Ltx(config) = &self.profile else {
            return Err(NativeLatentUpscaleModelError::CrossVariantInvocation);
        };
        let peak_channels = config
            .mid_channels
            .checked_mul(if config.spatial_upsample && config.temporal_upsample {
                8
            } else if config.spatial_upsample {
                4
            } else {
                2
            })
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        self.authorize_execution_shape(
            input,
            config.in_channels,
            peak_channels,
            if config.temporal_upsample { 2 } else { 1 },
            if config.spatial_upsample { 2 } else { 1 },
            12,
        )?;
        validate_input_tensor(input, config.in_channels, context)?;
        let [batch, _, frames, _, _] = input.descriptor().shape() else {
            return Err(NativeLatentUpscaleModelError::InvalidInput(
                "LTX rank".to_owned(),
            ));
        };
        let mut hidden = if config.dims == 2 {
            fold_video_frames(backend, input, context)?
        } else {
            input.clone()
        };
        hidden = ltx_convolution(
            backend,
            &hidden,
            self.state_tensor("initial_conv.weight")?,
            self.state_tensor("initial_conv.bias")?,
            config.dims,
            1,
            context,
        )?;
        hidden = group_norm_tensor_with_context_exact_native(
            backend,
            &hidden,
            32,
            Some(self.state_tensor("initial_norm.weight")?),
            Some(self.state_tensor("initial_norm.bias")?),
            1.0e-5,
            context,
        )
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
        hidden = silu_tensor_with_context_exact_native(backend, &hidden, context)
            .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
        for block in 0..config.num_blocks_per_stage {
            hidden = ltx_residual_block(
                self,
                backend,
                &hidden,
                "res_blocks",
                block,
                config.dims,
                context,
            )?;
        }
        hidden = if config.spatial_upsample && config.temporal_upsample {
            let convolved = ltx_convolution(
                backend,
                &hidden,
                self.state_tensor("upsampler.0.weight")?,
                self.state_tensor("upsampler.0.bias")?,
                3,
                1,
                context,
            )?;
            let shuffled = pixel_shuffle_nd_tensor_with_context_exact_native(
                backend, &convolved, 3, 2, context,
            )
            .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
            narrow_function_exact_native(
                &shuffled,
                2,
                1,
                shuffled.descriptor().shape()[2] - 1,
                context.cancellation,
            )
            .map_err(NativeLatentUpscaleModelError::tensor_operation)?
        } else if config.spatial_upsample && config.rational_resampler {
            ltx_rational_resample(self, backend, &hidden, config, context)?
        } else if config.spatial_upsample {
            let was_video = config.dims == 3;
            let folded = if was_video {
                fold_video_frames(backend, &hidden, context)?
            } else {
                hidden
            };
            let convolved = ltx_convolution(
                backend,
                &folded,
                self.state_tensor("upsampler.0.weight")?,
                self.state_tensor("upsampler.0.bias")?,
                2,
                1,
                context,
            )?;
            let shuffled =
                pixel_shuffle_tensor_with_context_exact_native(backend, &convolved, 2, context)
                    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
            if was_video {
                unfold_video_frames(backend, &shuffled, *batch, *frames, context)?
            } else {
                shuffled
            }
        } else {
            let convolved = ltx_convolution(
                backend,
                &hidden,
                self.state_tensor("upsampler.0.weight")?,
                self.state_tensor("upsampler.0.bias")?,
                3,
                1,
                context,
            )?;
            let shuffled = pixel_shuffle_nd_tensor_with_context_exact_native(
                backend, &convolved, 1, 2, context,
            )
            .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
            narrow_function_exact_native(
                &shuffled,
                2,
                1,
                shuffled.descriptor().shape()[2] - 1,
                context.cancellation,
            )
            .map_err(NativeLatentUpscaleModelError::tensor_operation)?
        };
        for block in 0..config.num_blocks_per_stage {
            hidden = ltx_residual_block(
                self,
                backend,
                &hidden,
                "post_upsample_res_blocks",
                block,
                config.dims,
                context,
            )?;
        }
        hidden = ltx_convolution(
            backend,
            &hidden,
            self.state_tensor("final_conv.weight")?,
            self.state_tensor("final_conv.bias")?,
            config.dims,
            1,
            context,
        )?;
        if config.dims == 2 {
            unfold_video_frames(backend, &hidden, *batch, *frames, context)
        } else {
            context.check()?;
            Ok(hidden)
        }
    }

    pub fn invoke_ltx_bundle(
        &self,
        backend: &CpuBackend,
        input: &NativeLatentBundle,
        vae: &NativeVae,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentBundle, NativeLatentUpscaleModelError> {
        self.normalize_execution_result(
            self.invoke_ltx_bundle_inner(backend, input, vae, context),
            context.cancellation,
        )
    }

    #[cfg(feature = "test-support")]
    pub fn rational_blur_test_support(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        stride: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeLatentUpscaleModelError> {
        ltx_depthwise_blur_downsample(self, backend, input, stride, context)
    }

    fn invoke_ltx_bundle_inner(
        &self,
        backend: &CpuBackend,
        input: &NativeLatentBundle,
        vae: &NativeVae,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentBundle, NativeLatentUpscaleModelError> {
        context.check()?;
        let NativeLatentUpscaleProfile::Ltx(config) = &self.profile else {
            return Err(NativeLatentUpscaleModelError::CrossVariantInvocation);
        };
        if config.in_channels != 128 {
            return Err(NativeLatentUpscaleModelError::InvalidInput(
                "LTX VAE invocation requires 128 model channels".to_owned(),
            ));
        }
        let NativeLatentSamples::Tensor(samples) = input.samples() else {
            return Err(NativeLatentUpscaleModelError::InvalidInput(
                "LTX requires a single tensor latent".to_owned(),
            ));
        };
        self.authorize_execution_shape(
            samples,
            config.in_channels,
            config.mid_channels.max(128),
            2,
            2,
            16,
        )?;
        let unnormalized = vae
            .apply_ltx_latent_statistics(
                backend,
                samples,
                LtxLatentStatisticsDirection::Unnormalize,
                context,
            )
            .map_err(NativeLatentUpscaleModelError::vae)?;
        let upscaled = self.invoke_ltx(backend, &unnormalized, context)?;
        let normalized = vae
            .apply_ltx_latent_statistics(
                backend,
                &upscaled,
                LtxLatentStatisticsDirection::Normalize,
                context,
            )
            .map_err(NativeLatentUpscaleModelError::vae)?;
        context.check()?;
        input
            .replaced_samples_mask_and_indices_source_dictionary(
                NativeLatentSamples::Tensor(normalized),
                None,
                input.batch_indices().map(ToOwned::to_owned),
                context,
            )
            .map_err(NativeLatentUpscaleModelError::Latent)
    }

    fn state_tensor(&self, key: &str) -> Result<&Tensor, NativeLatentUpscaleModelError> {
        self.state
            .get(key)
            .ok_or_else(|| NativeLatentUpscaleModelError::MissingState(key.to_owned()))
    }

    fn authorize_execution_shape(
        &self,
        input: &Tensor,
        configured_input_channels: usize,
        phase_channels: usize,
        temporal_scale: usize,
        spatial_scale: usize,
        simultaneous_tensors: usize,
    ) -> Result<(), NativeLatentUpscaleModelError> {
        let [batch, actual_input_channels, frames, height, width] = input.descriptor().shape()
        else {
            return Err(NativeLatentUpscaleModelError::InvalidInput(
                "execution memory planning requires rank five".to_owned(),
            ));
        };
        let retained_input_channels =
            usize_from(*actual_input_channels)?.max(configured_input_channels);
        let planned_channels = retained_input_channels
            .checked_add(phase_channels)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        let peak_elements = [
            usize_from(*batch)?,
            planned_channels,
            usize_from(*frames)?
                .checked_mul(temporal_scale)
                .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
            usize_from(*height)?
                .checked_mul(spatial_scale)
                .and_then(|value| value.checked_add(2))
                .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
            usize_from(*width)?
                .checked_mul(spatial_scale)
                .and_then(|value| value.checked_add(2))
                .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
        ];
        let execution_bytes = u64_from(checked_product(&peak_elements)?)?
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_mul(u64_from(simultaneous_tensors).ok()?))
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        self.authorize_execution_bytes(execution_bytes)
    }

    fn authorize_resize_execution(
        &self,
        input: &Tensor,
        target_width: u64,
        target_height: u64,
        mode: HunyuanLatentUpscaleMode,
    ) -> Result<(), NativeLatentUpscaleModelError> {
        let [batch, channels, frames, source_height, source_width] = input.descriptor().shape()
        else {
            return Err(NativeLatentUpscaleModelError::InvalidInput(
                "resize memory planning requires rank five".to_owned(),
            ));
        };
        let prefix = batch
            .checked_mul(*channels)
            .and_then(|value| value.checked_mul(*frames))
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        let source = source_height
            .checked_mul(*source_width)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        let target = target_height
            .checked_mul(target_width)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        let horizontal = if mode == HunyuanLatentUpscaleMode::Bislerp {
            source_height
                .checked_mul(target_width)
                .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?
        } else {
            0
        };
        let execution_bytes = prefix
            .checked_mul(
                source
                    .checked_add(target)
                    .and_then(|value| value.checked_add(horizontal))
                    .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
            )
            .and_then(|elements| elements.checked_mul(4))
            .and_then(|bytes| bytes.checked_mul(3))
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        self.authorize_execution_bytes(execution_bytes)
    }

    fn authorize_execution_bytes(
        &self,
        execution_bytes: u64,
    ) -> Result<(), NativeLatentUpscaleModelError> {
        let required = self
            .resident_bytes
            .checked_add(execution_bytes)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        if required > self.memory_budget_bytes {
            return Err(NativeLatentUpscaleModelError::OutOfMemory {
                required,
                budget: self.memory_budget_bytes,
            });
        }
        Ok(())
    }

    fn normalize_execution_result<T>(
        &self,
        result: Result<T, NativeLatentUpscaleModelError>,
        cancellation: &CancellationToken,
    ) -> Result<T, NativeLatentUpscaleModelError> {
        match result {
            Err(_) if cancellation.is_cancelled() => Err(NativeLatentUpscaleModelError::Cancelled),
            Err(NativeLatentUpscaleModelError::TensorOperation(error))
                if error_chain_is_resource_exhaustion(error.as_ref()) =>
            {
                Err(NativeLatentUpscaleModelError::Allocation)
            }
            Err(NativeLatentUpscaleModelError::Vae(error))
                if error_chain_is_resource_exhaustion(error.as_ref()) =>
            {
                Err(NativeLatentUpscaleModelError::Allocation)
            }
            result => result,
        }
    }
}

fn error_chain_is_resource_exhaustion(mut error: &(dyn std::error::Error + 'static)) -> bool {
    loop {
        if let Some(error) = error.downcast_ref::<TensorError>() {
            return matches!(
                error,
                TensorError::AllocationFailed { .. }
                    | TensorError::ResourceLimitExceeded { .. }
                    | TensorError::WorkspaceAuthorizationExceeded { .. }
            );
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

fn validate_input_tensor(
    input: &Tensor,
    channels: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeLatentUpscaleModelError> {
    context.check()?;
    let expected_channels = u64_from(channels)?;
    if !matches!(input.descriptor().shape(), [batch, actual_channels, frames, height, width]
        if *batch > 0 && *actual_channels == expected_channels && *frames > 0 && *height > 0 && *width > 0)
    {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "input must be a nonempty rank-five tensor with the configured channels".to_owned(),
        ));
    }
    if input.descriptor().dtype() != DType::F32
        || input.descriptor().device() != DeviceId::CPU
        || input.descriptor().stream() != context.stream
        || !input.descriptor().is_contiguous()?
    {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "input must be contiguous CPU F32 on the execution stream".to_owned(),
        ));
    }
    Ok(())
}

fn causal_video_convolution(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    context.check()?;
    let kernel_time = weight.descriptor().shape().get(2).copied().ok_or_else(|| {
        NativeLatentUpscaleModelError::InvalidInput("causal convolution weight rank".to_owned())
    })?;
    let spatial_padding = weight.descriptor().shape().get(3).copied().ok_or_else(|| {
        NativeLatentUpscaleModelError::InvalidInput("causal convolution weight rank".to_owned())
    })? / 2;
    let temporal_padding = kernel_time.saturating_sub(1);
    let padding = [
        i64::try_from(spatial_padding).map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?,
        i64::try_from(spatial_padding).map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?,
        i64::try_from(spatial_padding).map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?,
        i64::try_from(spatial_padding).map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?,
        i64::try_from(temporal_padding)
            .map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?,
        0,
    ];
    let padded = functional_pad_with_context_exact_native(
        backend,
        input,
        &padding,
        FunctionalPadMode::Replicate,
        None,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let configuration = ConvolutionConfiguration {
        stride: vec![1, 1, 1],
        padding: vec![0, 0, 0],
        dilation: vec![1, 1, 1],
        groups: 1,
        output_padding: vec![0, 0, 0],
    };
    conv_3d_tensor_with_context_exact_native(
        backend,
        &padded,
        weight,
        Some(bias),
        &configuration,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

fn hunyuan_rms_norm(
    backend: &CpuBackend,
    input: &Tensor,
    gamma: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    context.check()?;
    let [batch, channels, frames, height, width] = input.descriptor().shape() else {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "RMS normalization requires rank five".to_owned(),
        ));
    };
    if gamma.descriptor().shape() != [*channels, 1, 1, 1] {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "RMS gamma shape does not match channels".to_owned(),
        ));
    }
    let values = tensor_to_f32_with_context_exact_native(backend, input, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let gamma = tensor_to_f32_with_context_exact_native(backend, gamma, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let batch = usize_from(*batch)?;
    let channels = usize_from(*channels)?;
    let frames = usize_from(*frames)?;
    let height = usize_from(*height)?;
    let width = usize_from(*width)?;
    let shape = [batch, channels, frames, height, width];
    let normalized = normalize_with_context_exact_native(
        backend,
        &values,
        &shape,
        2.0,
        &[1],
        1.0e-12,
        DeviceId::CPU,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let spatial = checked_product(&[frames, height, width])?;
    let mut output = fallible_zeroed_f32(values.len())?;
    let scale = (channels as f32).sqrt();
    for batch_index in 0..batch {
        for spatial_index in 0..spatial {
            if spatial_index.is_multiple_of(64) {
                context.check()?;
            }
            for channel in 0..channels {
                let index = (batch_index * channels + channel) * spatial + spatial_index;
                output[index] = normalized[index] * scale * gamma[channel];
            }
        }
    }
    tensor_from_f32_with_context_exact_native(
        backend,
        input.descriptor().shape(),
        &output,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

fn ltx_residual_block(
    resource: &NativeLatentUpscaleModelResource,
    backend: &CpuBackend,
    input: &Tensor,
    family: &str,
    block: usize,
    dimensions: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    let prefix = format!("{family}.{block}");
    let residual = input.clone();
    let mut hidden = ltx_convolution(
        backend,
        input,
        resource.state_tensor(&format!("{prefix}.conv1.weight"))?,
        resource.state_tensor(&format!("{prefix}.conv1.bias"))?,
        dimensions,
        1,
        context,
    )?;
    hidden = group_norm_tensor_with_context_exact_native(
        backend,
        &hidden,
        32,
        Some(resource.state_tensor(&format!("{prefix}.norm1.weight"))?),
        Some(resource.state_tensor(&format!("{prefix}.norm1.bias"))?),
        1.0e-5,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    hidden = silu_tensor_with_context_exact_native(backend, &hidden, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    hidden = ltx_convolution(
        backend,
        &hidden,
        resource.state_tensor(&format!("{prefix}.conv2.weight"))?,
        resource.state_tensor(&format!("{prefix}.conv2.bias"))?,
        dimensions,
        1,
        context,
    )?;
    hidden = group_norm_tensor_with_context_exact_native(
        backend,
        &hidden,
        32,
        Some(resource.state_tensor(&format!("{prefix}.norm2.weight"))?),
        Some(resource.state_tensor(&format!("{prefix}.norm2.bias"))?),
        1.0e-5,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    hidden = real_add_with_context_exact_native(backend, &hidden, &residual, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    silu_tensor_with_context_exact_native(backend, &hidden, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

fn ltx_convolution(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    dimensions: usize,
    groups: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    let configuration = ConvolutionConfiguration {
        stride: vec![1; dimensions],
        padding: vec![1; dimensions],
        dilation: vec![1; dimensions],
        groups,
        output_padding: vec![0; dimensions],
    };
    match dimensions {
        2 => conv_2d_tensor_with_context_exact_native(
            backend,
            input,
            weight,
            Some(bias),
            &configuration,
            context,
        ),
        3 => conv_3d_tensor_with_context_exact_native(
            backend,
            input,
            weight,
            Some(bias),
            &configuration,
            context,
        ),
        _ => {
            return Err(NativeLatentUpscaleModelError::InvalidConfiguration(
                "convolution dimensions".to_owned(),
            ));
        }
    }
    .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

fn fold_video_frames(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    let [batch, channels, frames, height, width] = input.descriptor().shape() else {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "video fold rank".to_owned(),
        ));
    };
    let permuted = tensor_permute_exact_native(input, &[0, 2, 1, 3, 4], context.cancellation)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let contiguous = contiguous_with_context_exact_native(
        backend,
        &permuted,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let batch_frames = batch
        .checked_mul(*frames)
        .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
    tensor_reshape_with_context_exact_native(
        backend,
        &contiguous,
        &[
            i64_from(batch_frames)?,
            i64_from(*channels)?,
            i64_from(*height)?,
            i64_from(*width)?,
        ],
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

fn unfold_video_frames(
    backend: &CpuBackend,
    input: &Tensor,
    batch: u64,
    frames: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    let [batch_frames, channels, height, width] = input.descriptor().shape() else {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "video unfold rank".to_owned(),
        ));
    };
    if *batch_frames
        != batch
            .checked_mul(frames)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?
    {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "video unfold batch".to_owned(),
        ));
    }
    let reshaped = tensor_reshape_with_context_exact_native(
        backend,
        input,
        &[
            i64_from(batch)?,
            i64_from(frames)?,
            i64_from(*channels)?,
            i64_from(*height)?,
            i64_from(*width)?,
        ],
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let permuted = tensor_permute_exact_native(&reshaped, &[0, 2, 1, 3, 4], context.cancellation)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    contiguous_with_context_exact_native(
        backend,
        &permuted,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

#[cfg(test)]
#[allow(dead_code)]
fn pixel_shuffle_nd_direct(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: usize,
    factor: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    let [batch, input_channels, frames, height, width] = input.descriptor().shape() else {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "PixelShuffleND rank".to_owned(),
        ));
    };
    let divisor = factor
        .pow(u32::try_from(dimensions).map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?);
    let input_channels = usize_from(*input_channels)?;
    let channels = input_channels
        .checked_div(divisor)
        .filter(|channels| channels.checked_mul(divisor) == Some(input_channels))
        .ok_or_else(|| {
            NativeLatentUpscaleModelError::InvalidInput("PixelShuffleND channels".to_owned())
        })?;
    let batch = usize_from(*batch)?;
    let frames = usize_from(*frames)?;
    let height = usize_from(*height)?;
    let width = usize_from(*width)?;
    let values = tensor_to_f32_with_context_exact_native(backend, input, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let output_frames = if dimensions == 1 || dimensions == 3 {
        frames
            .checked_mul(factor)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?
    } else {
        frames
    };
    let output_height = if dimensions == 3 {
        height
            .checked_mul(factor)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?
    } else {
        height
    };
    let output_width = if dimensions == 3 {
        width
            .checked_mul(factor)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?
    } else {
        width
    };
    let output_count =
        checked_product(&[batch, channels, output_frames, output_height, output_width])?;
    let mut output = fallible_zeroed_f32(output_count)?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            for frame in 0..frames {
                for y in 0..height {
                    for x in 0..width {
                        if x.is_multiple_of(64) {
                            context.check()?;
                        }
                        for temporal in 0..if dimensions == 1 || dimensions == 3 {
                            factor
                        } else {
                            1
                        } {
                            for vertical in 0..if dimensions == 3 { factor } else { 1 } {
                                for horizontal in 0..if dimensions == 3 { factor } else { 1 } {
                                    let subchannel = ((channel
                                        * if dimensions == 1 || dimensions == 3 {
                                            factor
                                        } else {
                                            1
                                        }
                                        + temporal)
                                        * if dimensions == 3 { factor } else { 1 }
                                        + vertical)
                                        * if dimensions == 3 { factor } else { 1 }
                                        + horizontal;
                                    let source = ((((batch_index * input_channels + subchannel)
                                        * frames
                                        + frame)
                                        * height
                                        + y)
                                        * width)
                                        + x;
                                    let destination = ((((batch_index * channels + channel)
                                        * output_frames
                                        + frame
                                            * if dimensions == 1 || dimensions == 3 {
                                                factor
                                            } else {
                                                1
                                            }
                                        + temporal)
                                        * output_height
                                        + y * if dimensions == 3 { factor } else { 1 }
                                        + vertical)
                                        * output_width)
                                        + x * if dimensions == 3 { factor } else { 1 }
                                        + horizontal;
                                    output[destination] = values[source];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64_from(batch)?,
            u64_from(channels)?,
            u64_from(output_frames)?,
            u64_from(output_height)?,
            u64_from(output_width)?,
        ],
        &output,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

fn ltx_rational_resample(
    resource: &NativeLatentUpscaleModelResource,
    backend: &CpuBackend,
    input: &Tensor,
    config: &NativeLtxUpscaleConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    let [batch, _, frames, _, _] = input.descriptor().shape() else {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "rational resampler rank".to_owned(),
        ));
    };
    let folded = fold_video_frames(backend, input, context)?;
    let (numerator, denominator) = rational_scale(config.spatial_scale_milli)?;
    let convolved = ltx_convolution(
        backend,
        &folded,
        resource.state_tensor("upsampler.conv.weight")?,
        resource.state_tensor("upsampler.conv.bias")?,
        2,
        1,
        context,
    )?;
    let shuffled = pixel_shuffle_tensor_with_context_exact_native(
        backend,
        &convolved,
        u64_from(numerator)?,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let blurred =
        ltx_depthwise_blur_downsample(resource, backend, &shuffled, denominator, context)?;
    unfold_video_frames(backend, &blurred, *batch, *frames, context)
}

fn ltx_depthwise_blur_downsample(
    resource: &NativeLatentUpscaleModelResource,
    backend: &CpuBackend,
    input: &Tensor,
    stride: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    if stride == 1 {
        return Ok(input.clone());
    }
    let channels = usize_from(
        *input
            .descriptor()
            .shape()
            .get(1)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
    )?;
    let kernel = resource.state_tensor("upsampler.blur_down.kernel")?;
    let kernel_values = tensor_to_f32_with_context_exact_native(backend, kernel, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(
            channels
                .checked_mul(kernel_values.len())
                .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
        )
        .map_err(|_| NativeLatentUpscaleModelError::Allocation)?;
    for _ in 0..channels {
        expanded.extend_from_slice(&kernel_values);
    }
    let weight = tensor_from_f32_with_context_exact_native(
        backend,
        &[u64_from(channels)?, 1, 5, 5],
        &expanded,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let configuration = ConvolutionConfiguration {
        stride: vec![stride, stride],
        padding: vec![2, 2],
        dilation: vec![1, 1],
        groups: channels,
        output_padding: vec![0, 0],
    };
    conv_2d_tensor_with_context_exact_native(backend, input, &weight, None, &configuration, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

fn resolve_hunyuan_dimensions(
    width: u32,
    height: u32,
    source_width: u64,
    source_height: u64,
) -> Result<(u32, u32), NativeLatentUpscaleModelError> {
    let mut width = width;
    let mut height = height;
    if width == 0 {
        height = height.max(64);
        let resolved =
            (source_width as f64 * f64::from(height) / source_height as f64).round_ties_even();
        width = u32::try_from(resolved as u64)
            .map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?
            .max(64);
    } else if height == 0 {
        width = width.max(64);
        let resolved =
            (source_height as f64 * f64::from(width) / source_width as f64).round_ties_even();
        height = u32::try_from(resolved as u64)
            .map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?
            .max(64);
    } else {
        width = width.max(64);
        height = height.max(64);
    }
    Ok((width, height))
}

fn common_upscale_video(
    backend: &CpuBackend,
    input: &Tensor,
    target_width: u64,
    target_height: u64,
    mode: HunyuanLatentUpscaleMode,
    crop: HunyuanLatentUpscaleCrop,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    if target_width == 0 || target_height == 0 {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "Hunyuan target dimensions floor below one latent cell".to_owned(),
        ));
    }
    let [batch, _, frames, source_height, source_width] = input.descriptor().shape() else {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "common upscale rank".to_owned(),
        ));
    };
    let cropped = if crop == HunyuanLatentUpscaleCrop::Center {
        let old_aspect = *source_width as f64 / *source_height as f64;
        let new_aspect = target_width as f64 / target_height as f64;
        let (x, y) = if old_aspect > new_aspect {
            (
                (((source_width - 1) as f64 - *source_width as f64 * (new_aspect / old_aspect)
                    + 1.0)
                    / 2.0)
                    .round_ties_even() as u64,
                0,
            )
        } else if old_aspect < new_aspect {
            (
                0,
                (((source_height - 1) as f64 - *source_height as f64 * (old_aspect / new_aspect)
                    + 1.0)
                    / 2.0)
                    .round_ties_even() as u64,
            )
        } else {
            (0, 0)
        };
        let height = source_height
            .checked_sub(
                y.checked_mul(2)
                    .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
            )
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        let width = source_width
            .checked_sub(
                x.checked_mul(2)
                    .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
            )
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
        let narrowed =
            narrow_function_exact_native(input, 3, i64_from(y)?, height, context.cancellation)
                .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
        narrow_function_exact_native(&narrowed, 4, i64_from(x)?, width, context.cancellation)
            .map_err(NativeLatentUpscaleModelError::tensor_operation)?
    } else {
        input.clone()
    };
    let folded = fold_video_frames(backend, &cropped, context)?;
    let resized = if mode == HunyuanLatentUpscaleMode::Bislerp {
        bislerp_tensor_with_context_exact_native(
            backend,
            &folded,
            target_width,
            target_height,
            context,
        )
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?
    } else {
        let mode = match mode {
            HunyuanLatentUpscaleMode::NearestExact => InterpolateMode::NearestExact,
            HunyuanLatentUpscaleMode::Bilinear => InterpolateMode::Bilinear,
            HunyuanLatentUpscaleMode::Area => InterpolateMode::Area,
            HunyuanLatentUpscaleMode::Bicubic => InterpolateMode::Bicubic,
            HunyuanLatentUpscaleMode::Bislerp => {
                return Err(NativeLatentUpscaleModelError::InvalidInput(
                    "bislerp must use its source-exact path".to_owned(),
                ));
            }
        };
        interpolate_tensor_with_context_exact_native(
            backend,
            &folded,
            &InterpolateConfiguration {
                output_size: Some(vec![usize_from(target_height)?, usize_from(target_width)?]),
                scale_factor: None,
                mode,
                align_corners: None,
                recompute_scale_factor: None,
                antialias: false,
            },
            context,
        )
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?
    };
    unfold_video_frames(backend, &resized, *batch, *frames, context)
}

#[cfg(test)]
#[allow(dead_code)]
fn bislerp_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    target_width: u64,
    target_height: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentUpscaleModelError> {
    let [batch, channels, height, width] = input.descriptor().shape() else {
        return Err(NativeLatentUpscaleModelError::InvalidInput(
            "bislerp rank".to_owned(),
        ));
    };
    let batch = usize_from(*batch)?;
    let channels = usize_from(*channels)?;
    let height = usize_from(*height)?;
    let width = usize_from(*width)?;
    let target_width = usize_from(target_width)?;
    let target_height = usize_from(target_height)?;
    let source = tensor_to_f32_with_context_exact_native(backend, input, context)
        .map_err(NativeLatentUpscaleModelError::tensor_operation)?;
    let mut horizontal =
        fallible_zeroed_f32(checked_product(&[batch, channels, height, target_width])?)?;
    let mut left = fallible_zeroed_f32(channels)?;
    let mut right = fallible_zeroed_f32(channels)?;
    let mut mixed = fallible_zeroed_f32(channels)?;
    for batch_index in 0..batch {
        for y in 0..height {
            for output_x in 0..target_width {
                if output_x.is_multiple_of(64) {
                    context.check()?;
                }
                let coordinate = bilinear_coordinate(width, target_width, output_x);
                let x1 = coordinate.floor().max(0.0) as usize;
                let x2 = (x1 + 1).min(width - 1);
                let ratio = coordinate - x1 as f32;
                for channel in 0..channels {
                    left[channel] =
                        source[((batch_index * channels + channel) * height + y) * width + x1];
                    right[channel] =
                        source[((batch_index * channels + channel) * height + y) * width + x2];
                }
                source_slerp(&left, &right, ratio, &mut mixed);
                for channel in 0..channels {
                    horizontal[((batch_index * channels + channel) * height + y) * target_width
                        + output_x] = mixed[channel];
                }
            }
        }
    }
    let mut output = fallible_zeroed_f32(checked_product(&[
        batch,
        channels,
        target_height,
        target_width,
    ])?)?;
    for batch_index in 0..batch {
        for output_y in 0..target_height {
            let coordinate = bilinear_coordinate(height, target_height, output_y);
            let y1 = coordinate.floor().max(0.0) as usize;
            let y2 = (y1 + 1).min(height - 1);
            let ratio = coordinate - y1 as f32;
            for x in 0..target_width {
                if x.is_multiple_of(64) {
                    context.check()?;
                }
                for channel in 0..channels {
                    left[channel] = horizontal
                        [((batch_index * channels + channel) * height + y1) * target_width + x];
                    right[channel] = horizontal
                        [((batch_index * channels + channel) * height + y2) * target_width + x];
                }
                source_slerp(&left, &right, ratio, &mut mixed);
                for channel in 0..channels {
                    output[((batch_index * channels + channel) * target_height + output_y)
                        * target_width
                        + x] = mixed[channel];
                }
            }
        }
    }
    tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64_from(batch)?,
            u64_from(channels)?,
            u64_from(target_height)?,
            u64_from(target_width)?,
        ],
        &output,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(NativeLatentUpscaleModelError::tensor_operation)
}

#[cfg(test)]
fn bilinear_coordinate(source: usize, target: usize, index: usize) -> f32 {
    (((index as f32 + 0.5) * source as f32 / target as f32) - 0.5)
        .clamp(0.0, source.saturating_sub(1) as f32)
}

#[cfg(test)]
fn source_slerp(left: &[f32], right: &[f32], ratio: f32, output: &mut [f32]) {
    let left_norm = left
        .iter()
        .fold(0.0_f32, |sum, value| value.mul_add(*value, sum))
        .sqrt();
    let right_norm = right
        .iter()
        .fold(0.0_f32, |sum, value| value.mul_add(*value, sum))
        .sqrt();
    let dot = left.iter().zip(right).fold(0.0_f32, |sum, (left, right)| {
        let left = if left_norm == 0.0 {
            0.0
        } else {
            *left / left_norm
        };
        let right = if right_norm == 0.0 {
            0.0
        } else {
            *right / right_norm
        };
        left.mul_add(right, sum)
    });
    if dot > 1.0 - 1.0e-5 {
        output.copy_from_slice(left);
        return;
    }
    if dot < 1.0e-5 - 1.0 {
        for ((output, left), right) in output.iter_mut().zip(left).zip(right) {
            *output = *left * (1.0 - ratio) + *right * ratio;
        }
        return;
    }
    let omega = dot.acos();
    let sine = omega.sin();
    let length = left_norm * (1.0 - ratio) + right_norm * ratio;
    for ((output, left), right) in output.iter_mut().zip(left).zip(right) {
        let left = if left_norm == 0.0 {
            0.0
        } else {
            *left / left_norm
        };
        let right = if right_norm == 0.0 {
            0.0
        } else {
            *right / right_norm
        };
        let direction =
            (((1.0 - ratio) * omega).sin() / sine) * left + ((ratio * omega).sin() / sine) * right;
        *output = direction * length;
    }
}

fn normalize_1080_shortcuts(
    state: BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, Tensor>, NativeLatentUpscaleModelError> {
    let mut normalized = BTreeMap::new();
    for (index, (key, tensor)) in state.into_iter().enumerate() {
        if index.is_multiple_of(32) {
            cancellation.check()?;
        }
        let destination = if key.contains(".nin_shortcut.conv.") {
            key
        } else {
            key.replacen(".nin_shortcut.", ".nin_shortcut.conv.", 1)
        };
        if normalized.insert(destination.clone(), tensor).is_some() {
            return Err(NativeLatentUpscaleModelError::NormalizedStateCollision(
                destination,
            ));
        }
    }
    Ok(normalized)
}

fn infer_profile(
    architecture: NativeLatentUpscaleArchitecture,
    state: &BTreeMap<String, Tensor>,
    metadata: &BTreeMap<String, String>,
) -> Result<NativeLatentUpscaleProfile, NativeLatentUpscaleModelError> {
    match architecture {
        NativeLatentUpscaleArchitecture::Hunyuan720p => {
            let input = required_shape(state, "in_conv.conv.weight")?;
            let output = required_shape(state, "out_conv.conv.weight")?;
            require_kernel_shape("in_conv.conv.weight", input, 5, &[3, 3, 3])?;
            require_kernel_shape("out_conv.conv.weight", output, 5, &[3, 3, 3])?;
            let hidden_channels = usize_from(input[0])?;
            let in_channels = usize_from(input[1])?;
            let out_channels = usize_from(output[0])?;
            if output[1] != input[0] {
                return Err(NativeLatentUpscaleModelError::InvalidCheckpoint(
                    "720p input and output hidden channels disagree".to_owned(),
                ));
            }
            let blocks = contiguous_count(state, "blocks.", ".block.0.conv.weight")?;
            if blocks == 0 {
                return Err(NativeLatentUpscaleModelError::InvalidCheckpoint(
                    "720p checkpoint has no residual blocks".to_owned(),
                ));
            }
            Ok(NativeLatentUpscaleProfile::Hunyuan720p {
                in_channels,
                out_channels,
                hidden_channels,
                blocks,
            })
        }
        NativeLatentUpscaleArchitecture::Hunyuan1080p => {
            let input = required_shape(state, "conv_in.conv.weight")?;
            let output = required_shape(state, "conv_out.conv.weight")?;
            require_kernel_shape("conv_in.conv.weight", input, 5, &[3, 3, 3])?;
            require_kernel_shape("conv_out.conv.weight", output, 5, &[3, 3, 3])?;
            let z_channels = usize_from(input[1])?;
            let out_channels = usize_from(output[0])?;
            let stage_count = contiguous_count(state, "up.", ".block.0.conv1.conv.weight")?;
            if stage_count == 0 {
                return Err(NativeLatentUpscaleModelError::InvalidCheckpoint(
                    "1080p checkpoint has no stages".to_owned(),
                ));
            }
            let mut block_out_channels = Vec::new();
            block_out_channels
                .try_reserve_exact(stage_count)
                .map_err(|_| NativeLatentUpscaleModelError::Allocation)?;
            for stage in 0..stage_count {
                let shape =
                    required_shape(state, &format!("up.{stage}.block.0.conv1.conv.weight"))?;
                require_kernel_shape("1080p stage conv1", shape, 5, &[3, 3, 3])?;
                block_out_channels.push(usize_from(shape[0])?);
            }
            if input[0]
                != u64::try_from(block_out_channels[0])
                    .map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?
                || output[1]
                    != u64::try_from(
                        *block_out_channels
                            .last()
                            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
                    )
                    .map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?
            {
                return Err(NativeLatentUpscaleModelError::InvalidCheckpoint(
                    "1080p stage widths disagree with input or output convolution".to_owned(),
                ));
            }
            Ok(NativeLatentUpscaleProfile::Hunyuan1080p {
                z_channels,
                out_channels,
                block_out_channels,
            })
        }
        NativeLatentUpscaleArchitecture::Ltx => {
            let encoded = metadata.get("config").ok_or_else(|| {
                NativeLatentUpscaleModelError::InvalidConfiguration(
                    "LTX metadata is missing config".to_owned(),
                )
            })?;
            let value: serde_json::Value = serde_json::from_str(encoded).map_err(|error| {
                NativeLatentUpscaleModelError::InvalidConfiguration(error.to_string())
            })?;
            if !value.is_object() {
                return Err(NativeLatentUpscaleModelError::InvalidConfiguration(
                    "LTX config must be an object".to_owned(),
                ));
            }
            let wire: NativeLtxUpscaleConfigurationWire =
                serde_json::from_value(value).map_err(|error| {
                    NativeLatentUpscaleModelError::InvalidConfiguration(error.to_string())
                })?;
            let scaled = if wire.rational_resampler {
                let scaled = wire.spatial_scale * 1_000.0;
                if !scaled.is_finite()
                    || scaled.fract() != 0.0
                    || !(0.0..=f64::from(u32::MAX)).contains(&scaled)
                {
                    return Err(NativeLatentUpscaleModelError::InvalidConfiguration(
                        "LTX rational spatial scale is not canonical".to_owned(),
                    ));
                }
                scaled as u32
            } else {
                2_000
            };
            let config = NativeLtxUpscaleConfiguration {
                in_channels: wire.in_channels,
                mid_channels: wire.mid_channels,
                num_blocks_per_stage: wire.num_blocks_per_stage,
                dims: wire.dims,
                spatial_upsample: wire.spatial_upsample,
                temporal_upsample: wire.temporal_upsample,
                spatial_scale_milli: scaled,
                rational_resampler: wire.rational_resampler,
            };
            validate_ltx_configuration(&config)?;
            Ok(NativeLatentUpscaleProfile::Ltx(config))
        }
    }
}

fn expected_state_shapes(
    profile: &NativeLatentUpscaleProfile,
) -> Result<BTreeMap<String, Vec<u64>>, NativeLatentUpscaleModelError> {
    let mut expected = BTreeMap::new();
    match profile {
        NativeLatentUpscaleProfile::Hunyuan720p {
            in_channels,
            out_channels,
            hidden_channels,
            blocks,
        } => {
            add_conv(
                &mut expected,
                "in_conv.conv",
                *hidden_channels,
                *in_channels,
                &[3, 3, 3],
            )?;
            for block in 0..*blocks {
                for layer in [0, 2, 4] {
                    add_conv(
                        &mut expected,
                        &format!("blocks.{block}.block.{layer}.conv"),
                        *hidden_channels,
                        *hidden_channels,
                        &[3, 3, 3],
                    )?;
                }
            }
            add_conv(
                &mut expected,
                "out_conv.conv",
                *out_channels,
                *hidden_channels,
                &[3, 3, 3],
            )?;
        }
        NativeLatentUpscaleProfile::Hunyuan1080p {
            z_channels,
            out_channels,
            block_out_channels,
        } => {
            add_conv(
                &mut expected,
                "conv_in.conv",
                block_out_channels[0],
                *z_channels,
                &[3, 3, 3],
            )?;
            let mut input_channels = block_out_channels[0];
            for (stage, output_channels) in block_out_channels.iter().copied().enumerate() {
                for block in 0..3 {
                    let block_input = if block == 0 {
                        input_channels
                    } else {
                        output_channels
                    };
                    add_shape(
                        &mut expected,
                        format!("up.{stage}.block.{block}.norm1.gamma"),
                        &[block_input, 1, 1, 1],
                    )?;
                    add_conv(
                        &mut expected,
                        &format!("up.{stage}.block.{block}.conv1.conv"),
                        output_channels,
                        block_input,
                        &[3, 3, 3],
                    )?;
                    add_shape(
                        &mut expected,
                        format!("up.{stage}.block.{block}.norm2.gamma"),
                        &[output_channels, 1, 1, 1],
                    )?;
                    add_conv(
                        &mut expected,
                        &format!("up.{stage}.block.{block}.conv2.conv"),
                        output_channels,
                        output_channels,
                        &[3, 3, 3],
                    )?;
                    if block_input != output_channels {
                        add_conv(
                            &mut expected,
                            &format!("up.{stage}.block.{block}.nin_shortcut.conv"),
                            output_channels,
                            block_input,
                            &[1, 1, 1],
                        )?;
                    }
                }
                input_channels = output_channels;
            }
            add_shape(
                &mut expected,
                "norm_out.gamma".to_owned(),
                &[input_channels, 1, 1, 1],
            )?;
            add_conv(
                &mut expected,
                "conv_out.conv",
                *out_channels,
                input_channels,
                &[3, 3, 3],
            )?;
        }
        NativeLatentUpscaleProfile::Ltx(config) => {
            let kernel = if config.dims == 2 {
                vec![3, 3]
            } else {
                vec![3, 3, 3]
            };
            add_conv(
                &mut expected,
                "initial_conv",
                config.mid_channels,
                config.in_channels,
                &kernel,
            )?;
            add_group_norm(&mut expected, "initial_norm", config.mid_channels)?;
            for family in ["res_blocks", "post_upsample_res_blocks"] {
                for block in 0..config.num_blocks_per_stage {
                    add_conv(
                        &mut expected,
                        &format!("{family}.{block}.conv1"),
                        config.mid_channels,
                        config.mid_channels,
                        &kernel,
                    )?;
                    add_group_norm(
                        &mut expected,
                        &format!("{family}.{block}.norm1"),
                        config.mid_channels,
                    )?;
                    add_conv(
                        &mut expected,
                        &format!("{family}.{block}.conv2"),
                        config.mid_channels,
                        config.mid_channels,
                        &kernel,
                    )?;
                    add_group_norm(
                        &mut expected,
                        &format!("{family}.{block}.norm2"),
                        config.mid_channels,
                    )?;
                }
            }
            if config.spatial_upsample && config.temporal_upsample {
                add_conv(
                    &mut expected,
                    "upsampler.0",
                    config
                        .mid_channels
                        .checked_mul(8)
                        .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
                    config.mid_channels,
                    &[3, 3, 3],
                )?;
            } else if config.spatial_upsample && config.rational_resampler {
                let (numerator, _) = rational_scale(config.spatial_scale_milli)?;
                let output_channels = config
                    .mid_channels
                    .checked_mul(numerator)
                    .and_then(|channels| channels.checked_mul(numerator))
                    .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
                add_conv(
                    &mut expected,
                    "upsampler.conv",
                    output_channels,
                    config.mid_channels,
                    &[3, 3],
                )?;
                add_shape(
                    &mut expected,
                    "upsampler.blur_down.kernel".to_owned(),
                    &[1, 1, 5, 5],
                )?;
            } else if config.spatial_upsample {
                add_conv(
                    &mut expected,
                    "upsampler.0",
                    config
                        .mid_channels
                        .checked_mul(4)
                        .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
                    config.mid_channels,
                    &[3, 3],
                )?;
            } else {
                add_conv(
                    &mut expected,
                    "upsampler.0",
                    config
                        .mid_channels
                        .checked_mul(2)
                        .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?,
                    config.mid_channels,
                    &[3, 3, 3],
                )?;
            }
            add_conv(
                &mut expected,
                "final_conv",
                config.in_channels,
                config.mid_channels,
                &kernel,
            )?;
        }
    }
    Ok(expected)
}

fn validate_ltx_configuration(
    config: &NativeLtxUpscaleConfiguration,
) -> Result<(), NativeLatentUpscaleModelError> {
    if config.in_channels == 0
        || config.in_channels > 16_384
        || config.mid_channels == 0
        || config.mid_channels > 16_384
        || !config.mid_channels.is_multiple_of(32)
        || config.num_blocks_per_stage == 0
        || config.num_blocks_per_stage > 1_024
        || !matches!(config.dims, 2 | 3)
        || (!config.spatial_upsample && !config.temporal_upsample)
        || (config.dims == 2
            && (!config.spatial_upsample || config.temporal_upsample || config.rational_resampler))
        || (config.rational_resampler && (!config.spatial_upsample || config.temporal_upsample))
    {
        return Err(NativeLatentUpscaleModelError::InvalidConfiguration(
            "LTX dimensions, channels, blocks, or upsampling mode are invalid".to_owned(),
        ));
    }
    if config.rational_resampler {
        rational_scale(config.spatial_scale_milli)?;
    }
    Ok(())
}

fn rational_scale(scale_milli: u32) -> Result<(usize, usize), NativeLatentUpscaleModelError> {
    match scale_milli {
        750 => Ok((3, 4)),
        1_500 => Ok((3, 2)),
        2_000 => Ok((2, 1)),
        4_000 => Ok((4, 1)),
        _ => Err(NativeLatentUpscaleModelError::InvalidConfiguration(
            "unsupported LTX rational spatial scale".to_owned(),
        )),
    }
}

fn add_conv(
    expected: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    output: usize,
    input: usize,
    kernel: &[usize],
) -> Result<(), NativeLatentUpscaleModelError> {
    let mut shape = Vec::with_capacity(kernel.len() + 2);
    shape.push(u64_from(output)?);
    shape.push(u64_from(input)?);
    shape.extend(
        kernel
            .iter()
            .copied()
            .map(u64_from)
            .collect::<Result<Vec<_>, _>>()?,
    );
    expected.insert(format!("{prefix}.weight"), shape);
    expected.insert(format!("{prefix}.bias"), vec![u64_from(output)?]);
    Ok(())
}

fn add_group_norm(
    expected: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    channels: usize,
) -> Result<(), NativeLatentUpscaleModelError> {
    let shape = vec![u64_from(channels)?];
    expected.insert(format!("{prefix}.weight"), shape.clone());
    expected.insert(format!("{prefix}.bias"), shape);
    Ok(())
}

fn add_shape(
    expected: &mut BTreeMap<String, Vec<u64>>,
    key: String,
    shape: &[usize],
) -> Result<(), NativeLatentUpscaleModelError> {
    expected.insert(
        key,
        shape
            .iter()
            .copied()
            .map(u64_from)
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(())
}

fn validate_strict_state(
    state: &BTreeMap<String, Tensor>,
    expected: &BTreeMap<String, Vec<u64>>,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeLatentUpscaleModelError> {
    validate_strict_state_retained(state, expected, Some(context.stream), context.cancellation)
}

fn validate_strict_state_retained(
    state: &BTreeMap<String, Tensor>,
    expected: &BTreeMap<String, Vec<u64>>,
    expected_stream: Option<StreamId>,
    cancellation: &CancellationToken,
) -> Result<(), NativeLatentUpscaleModelError> {
    if state.len() != expected.len() {
        let missing = expected.keys().find(|key| !state.contains_key(*key));
        let unexpected = state.keys().find(|key| !expected.contains_key(*key));
        return Err(match (missing, unexpected) {
            (Some(key), _) => NativeLatentUpscaleModelError::MissingState(key.clone()),
            (_, Some(key)) => NativeLatentUpscaleModelError::UnexpectedState(key.clone()),
            _ => NativeLatentUpscaleModelError::InvalidCheckpoint(
                "state cardinality changed".to_owned(),
            ),
        });
    }
    for (index, (key, shape)) in expected.iter().enumerate() {
        if index.is_multiple_of(32) {
            cancellation.check()?;
        }
        let tensor = state
            .get(key)
            .ok_or_else(|| NativeLatentUpscaleModelError::MissingState(key.clone()))?;
        if tensor.descriptor().shape() != shape {
            return Err(NativeLatentUpscaleModelError::StateShape {
                key: key.clone(),
                expected: shape.clone(),
                actual: tensor.descriptor().shape().to_vec(),
            });
        }
        if tensor.descriptor().dtype() != DType::F32
            || tensor.descriptor().device() != DeviceId::CPU
            || expected_stream.is_some_and(|stream| tensor.descriptor().stream() != stream)
            || !tensor.descriptor().is_contiguous()?
        {
            return Err(NativeLatentUpscaleModelError::StatePlacement(key.clone()));
        }
    }
    cancellation.check()?;
    Ok(())
}

fn resident_state_bytes(
    state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<u64, NativeLatentUpscaleModelError> {
    let mut seen = BTreeMap::<u64, u64>::new();
    let mut bytes = 0_u64;
    for (index, tensor) in state.values().enumerate() {
        if index.is_multiple_of(32) {
            cancellation.check()?;
        }
        let storage_bytes = tensor.storage_byte_len();
        match seen.get(&tensor.storage_id().get()) {
            Some(previous) if *previous != storage_bytes => {
                return Err(NativeLatentUpscaleModelError::SemanticStateChanged);
            }
            Some(_) => {}
            None => {
                seen.insert(tensor.storage_id().get(), storage_bytes);
                bytes = bytes
                    .checked_add(storage_bytes)
                    .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
            }
        }
    }
    Ok(bytes)
}

fn resident_owned_bytes(
    artifact: &String,
    digest: &String,
    state: &BTreeMap<String, Tensor>,
) -> Result<u64, NativeLatentUpscaleModelError> {
    let mut bytes = u64::try_from(std::mem::size_of::<NativeLatentUpscaleModelResource>())
        .map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?;
    for capacity in [artifact.capacity(), digest.capacity()] {
        bytes = bytes
            .checked_add(u64_from(capacity)?)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
    }
    for key in state.keys() {
        bytes = bytes
            .checked_add(u64_from(key.capacity())?)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)?;
    }
    Ok(bytes)
}

fn semantic_digest(
    architecture: NativeLatentUpscaleArchitecture,
    profile: &NativeLatentUpscaleProfile,
    artifact: &str,
    state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, NativeLatentUpscaleModelError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.native-latent-upscale-model.v1");
    hasher.update([match architecture {
        NativeLatentUpscaleArchitecture::Hunyuan720p => 1,
        NativeLatentUpscaleArchitecture::Hunyuan1080p => 2,
        NativeLatentUpscaleArchitecture::Ltx => 3,
    }]);
    hash_field(&mut hasher, artifact.as_bytes())?;
    hash_profile(&mut hasher, profile)?;
    for (index, (key, tensor)) in state.iter().enumerate() {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        hash_field(&mut hasher, key.as_bytes())?;
        for dimension in tensor.descriptor().shape() {
            hasher.update(dimension.to_le_bytes());
        }
        let bytes = tensor.contiguous_bytes()?;
        hasher.update(
            u64::try_from(bytes.len())
                .map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?
                .to_le_bytes(),
        );
        for chunk in bytes.chunks(64 * 1_024) {
            cancellation.check()?;
            hasher.update(chunk);
        }
    }
    cancellation.check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_profile(
    hasher: &mut Sha256,
    profile: &NativeLatentUpscaleProfile,
) -> Result<(), NativeLatentUpscaleModelError> {
    match profile {
        NativeLatentUpscaleProfile::Hunyuan720p {
            in_channels,
            out_channels,
            hidden_channels,
            blocks,
        } => {
            hasher.update([1]);
            for value in [*in_channels, *out_channels, *hidden_channels, *blocks] {
                hasher.update(u64_from(value)?.to_le_bytes());
            }
        }
        NativeLatentUpscaleProfile::Hunyuan1080p {
            z_channels,
            out_channels,
            block_out_channels,
        } => {
            hasher.update([2]);
            hasher.update(u64_from(*z_channels)?.to_le_bytes());
            hasher.update(u64_from(*out_channels)?.to_le_bytes());
            hasher.update(u64_from(block_out_channels.len())?.to_le_bytes());
            for channels in block_out_channels {
                hasher.update(u64_from(*channels)?.to_le_bytes());
            }
        }
        NativeLatentUpscaleProfile::Ltx(config) => {
            hasher.update([3]);
            for value in [
                config.in_channels,
                config.mid_channels,
                config.num_blocks_per_stage,
                config.dims,
            ] {
                hasher.update(u64_from(value)?.to_le_bytes());
            }
            hasher.update([
                u8::from(config.spatial_upsample),
                u8::from(config.temporal_upsample),
                u8::from(config.rational_resampler),
            ]);
            hasher.update(config.spatial_scale_milli.to_le_bytes());
        }
    }
    Ok(())
}

fn contiguous_count(
    state: &BTreeMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Result<usize, NativeLatentUpscaleModelError> {
    let count = state
        .keys()
        .filter_map(|key| {
            key.strip_prefix(prefix)?
                .strip_suffix(suffix)?
                .parse::<usize>()
                .ok()
        })
        .collect::<BTreeSet<_>>();
    if count.iter().copied().eq(0..count.len()) {
        Ok(count.len())
    } else {
        Err(NativeLatentUpscaleModelError::InvalidCheckpoint(
            "state indices must be contiguous".to_owned(),
        ))
    }
}

fn required_shape<'a>(
    state: &'a BTreeMap<String, Tensor>,
    key: &str,
) -> Result<&'a [u64], NativeLatentUpscaleModelError> {
    state
        .get(key)
        .map(|tensor| tensor.descriptor().shape())
        .ok_or_else(|| NativeLatentUpscaleModelError::MissingState(key.to_owned()))
}

fn require_kernel_shape(
    key: &str,
    shape: &[u64],
    rank: usize,
    kernel: &[u64],
) -> Result<(), NativeLatentUpscaleModelError> {
    if shape.len() != rank || shape.get(2..) != Some(kernel) || shape[0] == 0 || shape[1] == 0 {
        return Err(NativeLatentUpscaleModelError::InvalidCheckpoint(format!(
            "{key} has an invalid convolution shape"
        )));
    }
    Ok(())
}

fn validate_state_key(key: &str) -> Result<(), NativeLatentUpscaleModelError> {
    if key.is_empty()
        || key.len() > MAX_STATE_KEY_BYTES
        || key.trim() != key
        || key.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(NativeLatentUpscaleModelError::InvalidStateKey);
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), NativeLatentUpscaleModelError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(NativeLatentUpscaleModelError::InvalidArtifactDigest);
    }
    Ok(())
}

fn hash_field(hasher: &mut Sha256, field: &[u8]) -> Result<(), NativeLatentUpscaleModelError> {
    hasher.update(
        u64::try_from(field.len())
            .map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(field);
    Ok(())
}

fn checked_product(values: &[usize]) -> Result<usize, NativeLatentUpscaleModelError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(NativeLatentUpscaleModelError::ShapeOverflow)
    })
}

fn fallible_zeroed_f32(length: usize) -> Result<Vec<f32>, NativeLatentUpscaleModelError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| NativeLatentUpscaleModelError::Allocation)?;
    values.resize(length, 0.0);
    Ok(values)
}

fn usize_from(value: u64) -> Result<usize, NativeLatentUpscaleModelError> {
    usize::try_from(value).map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)
}
fn u64_from(value: usize) -> Result<u64, NativeLatentUpscaleModelError> {
    u64::try_from(value).map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)
}
fn i64_from(value: u64) -> Result<i64, NativeLatentUpscaleModelError> {
    i64::try_from(value).map_err(|_| NativeLatentUpscaleModelError::ShapeOverflow)
}

#[derive(Debug, Error)]
pub enum NativeLatentUpscaleModelError {
    #[error("latent upscale operation was cancelled")]
    Cancelled,
    #[error("latent upscale checkpoint is invalid: {0}")]
    InvalidCheckpoint(String),
    #[error("latent upscale checkpoint has an invalid state key")]
    InvalidStateKey,
    #[error("latent upscale checkpoint has duplicate state key {0}")]
    DuplicateStateKey(String),
    #[error("latent upscale checkpoint normalization collides at {0}")]
    NormalizedStateCollision(String),
    #[error("latent upscale checkpoint architecture is unsupported")]
    UnsupportedArchitecture,
    #[error("latent upscale configuration is invalid: {0}")]
    InvalidConfiguration(String),
    #[error("latent upscale state is missing {0}")]
    MissingState(String),
    #[error("latent upscale state contains unexpected {0}")]
    UnexpectedState(String),
    #[error("latent upscale state {key} shape differs: expected {expected:?}, got {actual:?}")]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("latent upscale state {0} must be contiguous CPU F32")]
    StatePlacement(String),
    #[error("latent upscale artifact digest is invalid")]
    InvalidArtifactDigest,
    #[error("latent upscale state requires {required} bytes but budget is {budget}")]
    OutOfMemory { required: u64, budget: u64 },
    #[error("latent upscale semantic state changed")]
    SemanticStateChanged,
    #[error("latent upscale shape or byte accounting overflowed")]
    ShapeOverflow,
    #[error("latent upscale allocation failed")]
    Allocation,
    #[error("latent upscale invocation does not match the retained model variant")]
    CrossVariantInvocation,
    #[error("latent upscale input is invalid: {0}")]
    InvalidInput(String),
    #[error("latent upscale tensor operation failed: {0}")]
    TensorOperation(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("latent upscale VAE statistics failed: {0}")]
    Vae(#[source] Box<crate::VaeError>),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Latent(#[from] NativeLatentBundleError),
}

impl NativeLatentUpscaleModelError {
    fn tensor_operation(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::TensorOperation(Box::new(error))
    }

    fn vae(error: crate::VaeError) -> Self {
        Self::Vae(Box::new(error))
    }
}

impl From<sim_types::CancellationError> for NativeLatentUpscaleModelError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{
        CpuWorkspaceAuthority, StreamId, TensorBackend,
        generated_sim_operator_indirection_01::{
            tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
        },
    };
    use std::error::Error;

    const MEMORY_LIMIT: u64 = 32 * 1024 * 1024;

    fn context<'a>(
        authority: &CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
        Ok(ExecutionContext {
            stream: StreamId::default(),
            scratch: authority.authorize_workspace(MEMORY_LIMIT)?,
            rng_phase: None,
            cancellation,
        })
    }

    fn tensor(
        backend: &CpuBackend,
        shape: &[u64],
        values: &[f32],
        context: &ExecutionContext<'_>,
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

    fn identity_conv3d(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn Error>> {
        let mut values = vec![0.0_f32; 27];
        values[22] = 1.0;
        tensor(backend, &[1, 1, 3, 3, 3], &values, context)
    }

    fn hunyuan_720_checkpoint(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentUpscaleCheckpoint, Box<dyn Error>> {
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
                identity_conv3d(backend, context)?,
            ));
            ordered_state.push((
                format!("{prefix}.bias"),
                tensor(backend, &[1], &[0.0], context)?,
            ));
        }
        Ok(NativeLatentUpscaleCheckpoint {
            artifact_sha256: "1".repeat(64),
            metadata: BTreeMap::new(),
            ordered_state,
            memory_budget_bytes: MEMORY_LIMIT,
        })
    }

    fn hunyuan_720_wide_input_checkpoint(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        in_channels: usize,
    ) -> Result<NativeLatentUpscaleCheckpoint, Box<dyn Error>> {
        let mut ordered_state = Vec::new();
        let mut input_weight = vec![0.0_f32; in_channels * 27];
        for channel in 0..in_channels {
            input_weight[channel * 27 + 22] = 1.0 / in_channels as f32;
        }
        ordered_state.push((
            "in_conv.conv.weight".to_owned(),
            tensor(
                backend,
                &[1, u64_from(in_channels)?, 3, 3, 3],
                &input_weight,
                context,
            )?,
        ));
        ordered_state.push((
            "in_conv.conv.bias".to_owned(),
            tensor(backend, &[1], &[0.0], context)?,
        ));
        for prefix in [
            "blocks.0.block.0.conv",
            "blocks.0.block.2.conv",
            "blocks.0.block.4.conv",
            "out_conv.conv",
        ] {
            ordered_state.push((
                format!("{prefix}.weight"),
                identity_conv3d(backend, context)?,
            ));
            ordered_state.push((
                format!("{prefix}.bias"),
                tensor(backend, &[1], &[0.0], context)?,
            ));
        }
        Ok(NativeLatentUpscaleCheckpoint {
            artifact_sha256: "5".repeat(64),
            metadata: BTreeMap::new(),
            ordered_state,
            memory_budget_bytes: MEMORY_LIMIT,
        })
    }

    fn hunyuan_1080_checkpoint(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentUpscaleCheckpoint, Box<dyn Error>> {
        let profile = NativeLatentUpscaleProfile::Hunyuan1080p {
            z_channels: 1,
            out_channels: 1,
            block_out_channels: vec![2, 1],
        };
        let expected = expected_state_shapes(&profile)?;
        let mut ordered_state = Vec::new();
        for (key, shape) in expected {
            let raw_key = key.replacen("nin_shortcut.conv", "nin_shortcut", 1);
            let count = shape
                .iter()
                .try_fold(1_usize, |count, dimension| {
                    count.checked_mul(usize::try_from(*dimension).ok()?)
                })
                .ok_or("shape overflow")?;
            let mut values = vec![0.0_f32; count];
            if key.ends_with(".gamma") {
                values.fill(1.0);
            }
            if key == "conv_in.conv.weight" {
                values[22] = 1.0;
                values[27 + 22] = 2.0;
            } else if key == "up.1.block.0.nin_shortcut.conv.weight" {
                values.copy_from_slice(&[2.0, -1.0]);
            } else if key == "conv_out.conv.weight" {
                values[22] = 1.0;
            }
            ordered_state.push((raw_key, tensor(backend, &shape, &values, context)?));
        }
        Ok(NativeLatentUpscaleCheckpoint {
            artifact_sha256: "2".repeat(64),
            metadata: BTreeMap::new(),
            ordered_state,
            memory_budget_bytes: MEMORY_LIMIT,
        })
    }

    fn hunyuan_1080_order_sensitive_checkpoint(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentUpscaleCheckpoint, Box<dyn Error>> {
        let profile = NativeLatentUpscaleProfile::Hunyuan1080p {
            z_channels: 1,
            out_channels: 1,
            block_out_channels: vec![2],
        };
        let branch_weights = [[0.5_f32, -0.25], [-0.75, 0.5], [0.25, 1.0]];
        let expected = expected_state_shapes(&profile)?;
        let mut ordered_state = Vec::new();
        for (key, shape) in expected {
            let count = shape
                .iter()
                .try_fold(1_usize, |count, dimension| {
                    count.checked_mul(usize::try_from(*dimension).ok()?)
                })
                .ok_or("shape overflow")?;
            let mut values = vec![0.0_f32; count];
            if key.ends_with(".gamma") {
                values.fill(1.0);
            }
            if key == "conv_in.conv.weight" {
                *values.get_mut(22).ok_or("conv-in first channel")? = 1.0;
                *values.get_mut(49).ok_or("conv-in second channel")? = 2.0;
            } else if key == "conv_out.conv.weight" {
                *values.get_mut(22).ok_or("conv-out first channel")? = 1.0;
                *values.get_mut(49).ok_or("conv-out second channel")? = -1.0;
            }
            for (block, branch) in branch_weights.iter().enumerate() {
                if key == format!("up.0.block.{block}.conv1.conv.weight") {
                    *values.get_mut(22).ok_or("conv1 first diagonal")? = 1.0;
                    *values.get_mut(103).ok_or("conv1 second diagonal")? = 1.0;
                } else if key == format!("up.0.block.{block}.conv2.conv.weight") {
                    *values.get_mut(22).ok_or("conv2 first diagonal")? = branch[0];
                    *values.get_mut(103).ok_or("conv2 second diagonal")? = branch[1];
                }
            }
            ordered_state.push((key, tensor(backend, &shape, &values, context)?));
        }
        Ok(NativeLatentUpscaleCheckpoint {
            artifact_sha256: "4".repeat(64),
            metadata: BTreeMap::new(),
            ordered_state,
            memory_budget_bytes: MEMORY_LIMIT,
        })
    }

    fn ltx_checkpoint(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        config: NativeLtxUpscaleConfiguration,
    ) -> Result<NativeLatentUpscaleCheckpoint, Box<dyn Error>> {
        let expected = expected_state_shapes(&NativeLatentUpscaleProfile::Ltx(config.clone()))?;
        let mut ordered_state = Vec::new();
        for (key, shape) in expected {
            let count = shape
                .iter()
                .try_fold(1_usize, |count, dimension| {
                    count.checked_mul(usize::try_from(*dimension).ok()?)
                })
                .ok_or("shape overflow")?;
            let mut values = vec![0.0_f32; count];
            if key.ends_with("norm.weight")
                || key.contains(".norm1.weight")
                || key.contains(".norm2.weight")
            {
                values.fill(1.0);
            }
            if key == "upsampler.0.bias" {
                values.fill(1.0);
            } else if key == "final_conv.weight" {
                values[4] = 1.0;
            }
            ordered_state.push((key, tensor(backend, &shape, &values, context)?));
        }
        let config_json = serde_json::json!({
            "_class_name": "LatentUpsampler",
            "in_channels": config.in_channels,
            "mid_channels": config.mid_channels,
            "num_blocks_per_stage": config.num_blocks_per_stage,
            "dims": config.dims,
            "spatial_upsample": config.spatial_upsample,
            "temporal_upsample": config.temporal_upsample,
            "spatial_scale": f64::from(config.spatial_scale_milli) / 1_000.0,
            "rational_resampler": config.rational_resampler,
        });
        Ok(NativeLatentUpscaleCheckpoint {
            artifact_sha256: "3".repeat(64),
            metadata: BTreeMap::from([("config".to_owned(), serde_json::to_string(&config_json)?)]),
            ordered_state,
            memory_budget_bytes: MEMORY_LIMIT,
        })
    }

    #[test]
    fn latent_upscale_model_720_loader_and_raw_graph_are_source_exact() -> Result<(), Box<dyn Error>>
    {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;
        let resource = NativeLatentUpscaleModelResource::from_checkpoint(
            hunyuan_720_checkpoint(&backend, &context)?,
            &context,
        )?;
        let payload =
            crate::NativeModelPayload::latent_upscale_model(std::sync::Arc::new(resource.clone()))?;
        assert_eq!(
            payload.identity().role(),
            crate::NativeModelResourceRole::LatentUpscaleModel
        );
        assert_eq!(
            payload
                .latent_upscale_model_resource()
                .map(|resource| resource.identifier()),
            Some(resource.identifier())
        );
        payload.validate()?;
        assert_eq!(
            resource.architecture(),
            NativeLatentUpscaleArchitecture::Hunyuan720p
        );
        resource.validate(&cancellation)?;

        let input = tensor(&backend, &[1, 1, 3, 1, 1], &[-1.0, 0.0, 1.0], &context)?;
        let output = resource.invoke_hunyuan_720p(&backend, &input, &context)?;
        let values = tensor_to_f32_with_context_exact_native(&backend, &output, &context)?;
        let expected = [-1.116_496_6_f32, 0.0, 1.493_492];
        for (actual, expected) in values.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2.0e-6,
                "{actual} != {expected}"
            );
        }
        assert_eq!(output.descriptor().shape(), &[1, 1, 3, 1, 1]);
        Ok(())
    }

    #[test]
    fn latent_upscale_model_causal_video_convolution_replicates_only_the_past()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;
        let input = tensor(&backend, &[1, 1, 3, 1, 1], &[1.0, 2.0, 4.0], &context)?;
        let mut kernel = vec![0.0_f32; 27];
        for (index, value) in [(4, 1.0), (13, 10.0), (22, 100.0)] {
            let destination = kernel
                .get_mut(index)
                .ok_or("causal kernel index is invalid")?;
            *destination = value;
        }
        let weight = tensor(&backend, &[1, 1, 3, 3, 3], &kernel, &context)?;
        let bias = tensor(&backend, &[1], &[0.0], &context)?;
        let output = causal_video_convolution(&backend, &input, &weight, &bias, &context)?;
        assert_eq!(output.descriptor().shape(), &[1, 1, 3, 1, 1]);
        assert_eq!(
            tensor_to_f32_with_context_exact_native(&backend, &output, &context)?,
            vec![111.0, 211.0, 421.0]
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn latent_upscale_model_execution_budget_fails_before_tensor_publication()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;
        let baseline = NativeLatentUpscaleModelResource::from_checkpoint(
            hunyuan_720_checkpoint(&backend, &context)?,
            &context,
        )?;
        let mut constrained = hunyuan_720_checkpoint(&backend, &context)?;
        constrained.memory_budget_bytes = baseline.resident_bytes();
        let constrained = NativeLatentUpscaleModelResource::from_checkpoint(constrained, &context)?;
        let input = tensor(&backend, &[1, 1, 3, 1, 1], &[1.0, 2.0, 4.0], &context)?;
        let input_version = input.mutation_version();
        assert!(matches!(
            constrained.invoke_hunyuan_720p(&backend, &input, &context),
            Err(NativeLatentUpscaleModelError::OutOfMemory { budget, .. })
                if budget == constrained.resident_bytes()
        ));
        assert_eq!(input.mutation_version(), input_version);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn latent_upscale_model_execution_budget_accounts_for_wide_inputs_and_phase_channels()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;

        let baseline_720 = NativeLatentUpscaleModelResource::from_checkpoint(
            hunyuan_720_wide_input_checkpoint(&backend, &context, 64)?,
            &context,
        )?;
        let mut constrained_720 = hunyuan_720_wide_input_checkpoint(&backend, &context, 64)?;
        constrained_720.memory_budget_bytes = baseline_720
            .resident_bytes()
            .checked_add(4_096)
            .ok_or("720 budget overflow")?;
        let constrained_720 =
            NativeLatentUpscaleModelResource::from_checkpoint(constrained_720, &context)?;
        let wide_720 = tensor(&backend, &[1, 64, 1, 1, 1], &[0.0; 64], &context)?;
        let wide_720_version = wide_720.mutation_version();
        assert!(matches!(
            constrained_720.invoke_hunyuan_720p(&backend, &wide_720, &context),
            Err(NativeLatentUpscaleModelError::OutOfMemory { .. })
        ));
        assert_eq!(wide_720.mutation_version(), wide_720_version);

        let config = NativeLtxUpscaleConfiguration {
            in_channels: 256,
            mid_channels: 32,
            num_blocks_per_stage: 1,
            dims: 3,
            spatial_upsample: true,
            temporal_upsample: false,
            spatial_scale_milli: 2_000,
            rational_resampler: false,
        };
        let baseline_ltx = NativeLatentUpscaleModelResource::from_checkpoint(
            ltx_checkpoint(&backend, &context, config.clone())?,
            &context,
        )?;
        let mut constrained_ltx = ltx_checkpoint(&backend, &context, config)?;
        constrained_ltx.memory_budget_bytes = baseline_ltx
            .resident_bytes()
            .checked_add(150_000)
            .ok_or("LTX budget overflow")?;
        let constrained_ltx =
            NativeLatentUpscaleModelResource::from_checkpoint(constrained_ltx, &context)?;
        let wide_ltx = tensor(&backend, &[1, 256, 1, 1, 1], &[0.0; 256], &context)?;
        let wide_ltx_version = wide_ltx.mutation_version();
        assert!(matches!(
            constrained_ltx.invoke_ltx(&backend, &wide_ltx, &context),
            Err(NativeLatentUpscaleModelError::OutOfMemory { .. })
        ));
        assert_eq!(wide_ltx.mutation_version(), wide_ltx_version);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn latent_upscale_model_marker_precedence_and_strict_state_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;
        let mut checkpoint = hunyuan_720_checkpoint(&backend, &context)?;
        checkpoint.ordered_state.push((
            HUNYUAN_1080_MARKER.to_owned(),
            identity_conv3d(&backend, &context)?,
        ));
        let error = NativeLatentUpscaleModelResource::from_checkpoint(checkpoint, &context)
            .expect_err("the earlier 720 marker must select 720 then reject lower-profile state");
        assert!(
            matches!(error, NativeLatentUpscaleModelError::UnexpectedState(key) if key == HUNYUAN_1080_MARKER)
        );

        let mut checkpoint = hunyuan_720_checkpoint(&backend, &context)?;
        checkpoint.ordered_state.pop();
        let error = NativeLatentUpscaleModelResource::from_checkpoint(checkpoint, &context)
            .expect_err("missing strict state must fail");
        assert!(matches!(
            error,
            NativeLatentUpscaleModelError::MissingState(_)
        ));
        Ok(())
    }

    #[test]
    fn latent_upscale_model_1080_normalization_and_graph_are_source_exact()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;
        let resource = NativeLatentUpscaleModelResource::from_checkpoint(
            hunyuan_1080_checkpoint(&backend, &context)?,
            &context,
        )?;
        assert_eq!(
            resource.architecture(),
            NativeLatentUpscaleArchitecture::Hunyuan1080p
        );
        assert!(
            resource
                .state()
                .contains_key("up.1.block.0.nin_shortcut.conv.weight")
        );
        assert!(
            !resource
                .state()
                .contains_key("up.1.block.0.nin_shortcut.weight")
        );
        let input = tensor(&backend, &[1, 1, 1, 1, 1], &[1.0], &context)?;
        let output = resource.invoke_hunyuan_1080p(&backend, &input, &context)?;
        let values = tensor_to_f32_with_context_exact_native(&backend, &output, &context)?;
        assert_eq!(values.len(), 1);
        assert!((values[0] - 0.731_058_6).abs() <= 2.0e-6, "{}", values[0]);

        let mut collision = hunyuan_1080_checkpoint(&backend, &context)?;
        collision.ordered_state.push((
            "up.1.block.0.nin_shortcut.conv.weight".to_owned(),
            tensor(&backend, &[1, 2, 1, 1, 1], &[2.0, -1.0], &context)?,
        ));
        assert!(matches!(
            NativeLatentUpscaleModelResource::from_checkpoint(collision, &context),
            Err(NativeLatentUpscaleModelError::NormalizedStateCollision(key))
                if key == "up.1.block.0.nin_shortcut.conv.weight"
        ));
        Ok(())
    }

    #[test]
    fn latent_upscale_model_1080_executes_all_three_residual_blocks_in_order()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;
        let resource = NativeLatentUpscaleModelResource::from_checkpoint(
            hunyuan_1080_order_sensitive_checkpoint(&backend, &context)?,
            &context,
        )?;
        let input = tensor(&backend, &[1, 1, 1, 1, 1], &[1.0], &context)?;
        let output = resource.invoke_hunyuan_1080p(&backend, &input, &context)?;
        let values = tensor_to_f32_with_context_exact_native(&backend, &output, &context)?;
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].to_bits(), 3_206_739_852);
        Ok(())
    }

    #[test]
    fn latent_upscale_model_ltx_dims_two_executes_exact_spatial_shuffle()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;
        let config = NativeLtxUpscaleConfiguration {
            in_channels: 1,
            mid_channels: 32,
            num_blocks_per_stage: 1,
            dims: 2,
            spatial_upsample: true,
            temporal_upsample: false,
            spatial_scale_milli: 2_000,
            rational_resampler: false,
        };
        let resource = NativeLatentUpscaleModelResource::from_checkpoint(
            ltx_checkpoint(&backend, &context, config)?,
            &context,
        )?;
        assert_eq!(
            resource.architecture(),
            NativeLatentUpscaleArchitecture::Ltx
        );
        let input = tensor(&backend, &[1, 1, 2, 2, 2], &[0.0; 8], &context)?;
        let output = resource.invoke_ltx(&backend, &input, &context)?;
        assert_eq!(output.descriptor().shape(), &[1, 1, 2, 4, 4]);
        let values = tensor_to_f32_with_context_exact_native(&backend, &output, &context)?;
        assert!(
            values
                .iter()
                .all(|value| (*value - 0.731_058_6).abs() <= 2.0e-5)
        );
        Ok(())
    }

    #[test]
    fn latent_upscale_model_hunyuan_bundle_aliases_zero_and_drops_nonzero_fields()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation)?;
        let resource = NativeLatentUpscaleModelResource::from_checkpoint(
            hunyuan_720_checkpoint(&backend, &context)?,
            &context,
        )?;
        let samples = tensor(&backend, &[1, 1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let mask = tensor(&backend, &[1, 1, 1, 2, 2], &[1.0; 4], &context)?;
        let metadata = NativeLatentMetadata::checked(None, Some(48_000), Some(8), Some(4))?;
        let input =
            NativeLatentBundle::single(samples, Some(mask), Some(vec![9]), metadata, &context)?;
        let alias = resource.invoke_hunyuan_bundle(
            &backend,
            &input,
            0,
            0,
            HunyuanLatentUpscaleMode::Bilinear,
            HunyuanLatentUpscaleCrop::Disabled,
            &context,
        )?;
        let (
            NativeLatentSamples::Tensor(alias_samples),
            NativeLatentSamples::Tensor(input_samples),
        ) = (alias.samples(), input.samples())
        else {
            return Err("unexpected nested samples".into());
        };
        assert_eq!(alias_samples.storage_id(), input_samples.storage_id());
        assert_eq!(alias.projection(), input.projection());

        let output = resource.invoke_hunyuan_bundle(
            &backend,
            &input,
            64,
            64,
            HunyuanLatentUpscaleMode::NearestExact,
            HunyuanLatentUpscaleCrop::Center,
            &context,
        )?;
        assert!(output.noise_mask().is_none());
        assert!(output.batch_indices().is_none());
        assert_eq!(output.metadata(), &NativeLatentMetadata::default());
        let NativeLatentSamples::Tensor(output_samples) = output.samples() else {
            return Err("unexpected nested samples".into());
        };
        assert_eq!(output_samples.descriptor().shape(), &[1, 1, 1, 4, 4]);
        assert_ne!(output_samples.storage_id(), input_samples.storage_id());
        Ok(())
    }
}
