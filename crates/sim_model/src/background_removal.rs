use sha2::{Digest, Sha256};
#[cfg(any(test, feature = "test-support"))]
use sim_tensor::TensorDescriptor;
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, ImageTensor, Scalar,
    StorageId, StreamId, Tensor, TensorError,
    generated_activation_normalization_functional_01::{
        FunctionalError, GeluApproximation, batch_norm_with_context_exact_native,
        gelu_with_context_exact_native, layer_norm_with_context_exact_native,
        relu_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, ElementwiseRuntimePartThreeError, sigmoid_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_05::{
        ElementwiseRuntimePartFiveError, div_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_06::{
        ElementwiseRuntimePartSixError, round_method_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_16::{
        ElementwiseRuntimePartSixteenError, add_method_with_context_exact_native,
        mul_method_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_17::{
        ElementwiseRuntimePartSeventeenError, clip_with_context_exact_native,
        roll_with_context_exact_native,
    },
    generated_external_tensor_kernel_02::{
        ExternalTensorKernelPartTwoError, NativeDeformConv2dConfiguration,
        deform_conv2d_with_context_exact_native,
    },
    generated_indexing_masking_01::{IndexingMaskingPartOneError, narrow_method_exact_native},
    generated_neural_network_functional_01::{
        NeuralNetworkFunctionalError, linear_with_context_exact_native,
    },
    generated_neural_network_module_01::{
        NeuralNetworkModuleError, adaptive_average_pool_2d_with_context_exact_native,
    },
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, torch_cat_with_context_exact_native,
    },
    generated_shape_layout_transform_03::{
        FunctionalPadMode, ShapeLayoutTransformPartThreeError,
        functional_pad_with_context_exact_native, tensor_permute_exact_native,
        tensor_squeeze_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, tensor_from_f32_with_context_exact_native,
        tensor_to_f32_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        ConvolutionConfiguration, InterpolateConfiguration, InterpolateMode,
        SpatialFunctionalKernelError, conv_2d_tensor_with_context_exact_native,
        interpolate_tensor_with_context_exact_native,
    },
};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::attention::{
    AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionMask, AttentionMaskShape,
    AttentionRequest, scaled_dot_product_attention_with_context,
};

pub const NODES_BACKGROUND_REMOVAL_SOURCE_SHA256: &str =
    "c2cf4b42f10cfb1bb057b60a8745fb96a2462e1f1a2bd275e00795bb3f758cce";
pub const BACKGROUND_REMOVAL_MODEL_SOURCE_SHA256: &str =
    "c4f6f7beea512c759849efa07f03f09044ea76fe5c71fb7afff31e4886e4daa7";
pub const BIREFNET_SOURCE_SHA256: &str =
    "00a083bd9a619943a7fdd1d8f827dae7734a5031ced3c37893f25ee925c670b1";
pub const BIREFNET_CONFIG_SOURCE_SHA256: &str =
    "50dd9639fa207a823437370b46d32a56b3f00eb1bef3bd225fe87eeeb8f255d2";
pub const CLIP_MODEL_SOURCE_SHA256: &str =
    "08be993d86c3b494b58305fb868638b4b525bbe40abead89e9c94da021716845";
pub const SIM_OPS_SOURCE_SHA256: &str =
    "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42";
pub const MODEL_MANAGEMENT_SOURCE_SHA256: &str =
    "c2ca243c80a5262ecafe19feb15cec22d4003c16e523b5376f543f0f75acabaa";

pub const BIREFNET_MARKER: &str = "bb.layers.1.blocks.0.attn.relative_position_index";
const MAX_STATE_TENSORS: usize = 8_192;
const MAX_STATE_KEY_BYTES: usize = 1_024;

#[derive(Clone, Debug)]
pub struct NativeBackgroundRemovalCheckpoint {
    pub artifact_sha256: String,
    pub ordered_state: Vec<(String, Tensor)>,
    pub memory_budget_bytes: u64,
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub enum BackgroundRemovalFixtureMutation {
    None,
    ShiftedWindowBlock,
    RelativePositionIndex,
    DeformOffset,
    DeformMask,
    AsppDilatedBranch,
    AsppGlobalPool,
    UnusedDecoderHead,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BiRefNetProfile {
    image_size: usize,
    embed_dim: usize,
    depths: [usize; 4],
    heads: [usize; 4],
    window_size: usize,
    decoder_inter_channels: usize,
    aspp_channels: usize,
    gdt_channels: usize,
    source_exact: bool,
}

impl BiRefNetProfile {
    fn source() -> Self {
        Self {
            image_size: 1_024,
            embed_dim: 192,
            depths: [2, 2, 18, 2],
            heads: [6, 12, 24, 48],
            window_size: 12,
            decoder_inter_channels: 64,
            aspp_channels: 256,
            gdt_channels: 16,
            source_exact: true,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    fn reduced() -> Self {
        Self {
            image_size: 8,
            embed_dim: 4,
            depths: [2, 1, 1, 1],
            heads: [1, 1, 2, 4],
            window_size: 2,
            decoder_inter_channels: 2,
            aspp_channels: 2,
            gdt_channels: 2,
            source_exact: false,
        }
    }

    fn channels(&self) -> Result<[usize; 4], NativeBackgroundRemovalError> {
        let twice = self
            .embed_dim
            .checked_mul(2)
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
        Ok([
            twice
                .checked_mul(8)
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?,
            twice
                .checked_mul(4)
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?,
            twice
                .checked_mul(2)
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?,
            twice,
        ])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StateKind {
    Float,
    RelativePositionIndex { maximum: u64 },
    BatchCounter,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StateSpecification {
    key: String,
    shape: Vec<u64>,
    kind: StateKind,
}

#[derive(Clone, Debug)]
pub struct NativeBackgroundRemovalResource {
    profile: BiRefNetProfile,
    artifact_sha256: String,
    state: BTreeMap<String, Tensor>,
    stream: StreamId,
    memory_budget_bytes: u64,
    resident_bytes: u64,
    semantic_digest_sha256: String,
}

impl NativeBackgroundRemovalResource {
    pub fn from_checkpoint(
        checkpoint: NativeBackgroundRemovalCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeBackgroundRemovalError> {
        Self::checked(checkpoint, BiRefNetProfile::source(), context)
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn from_reduced_fixture(
        checkpoint: NativeBackgroundRemovalCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeBackgroundRemovalError> {
        Self::checked(checkpoint, BiRefNetProfile::reduced(), context)
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn deterministic_reduced_test_fixture(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        mutation: BackgroundRemovalFixtureMutation,
    ) -> Result<Self, NativeBackgroundRemovalError> {
        let checkpoint = deterministic_reduced_test_checkpoint(backend, context, mutation)?;
        Self::from_reduced_fixture(checkpoint, context)
    }

    fn checked(
        checkpoint: NativeBackgroundRemovalCheckpoint,
        profile: BiRefNetProfile,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeBackgroundRemovalError> {
        context.check()?;
        validate_sha256(&checkpoint.artifact_sha256)?;
        if checkpoint.ordered_state.is_empty()
            || checkpoint.ordered_state.len() > MAX_STATE_TENSORS
            || checkpoint.memory_budget_bytes == 0
        {
            return Err(NativeBackgroundRemovalError::InvalidCheckpoint(
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
                return Err(NativeBackgroundRemovalError::DuplicateStateKey(key));
            }
        }
        if !state.contains_key(BIREFNET_MARKER) {
            return Err(NativeBackgroundRemovalError::UnsupportedArchitecture);
        }
        let specifications = birefnet_state_manifest(&profile)?;
        validate_strict_state(&state, &specifications, context)?;
        let stream = state
            .values()
            .next()
            .ok_or(NativeBackgroundRemovalError::UnsupportedArchitecture)?
            .descriptor()
            .stream();
        let semantic_digest_sha256 = semantic_digest(
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
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
        if resident_bytes > checkpoint.memory_budget_bytes {
            return Err(NativeBackgroundRemovalError::OutOfMemory {
                required: resident_bytes,
                budget: checkpoint.memory_budget_bytes,
            });
        }
        context.check()?;
        Ok(Self {
            profile,
            artifact_sha256: checkpoint.artifact_sha256,
            state,
            stream,
            memory_budget_bytes: checkpoint.memory_budget_bytes,
            resident_bytes,
            semantic_digest_sha256,
        })
    }

    pub fn identifier(&self) -> &'static str {
        "birefnet"
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn is_source_exact_profile(&self) -> bool {
        self.profile.source_exact
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, NativeBackgroundRemovalError> {
        resident_owned_bytes(
            &self.artifact_sha256,
            &self.semantic_digest_sha256,
            &self.state,
        )
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(StorageId, u64)>, NativeBackgroundRemovalError> {
        resident_tensor_allocations(&self.state, &CancellationToken::default())
    }

    pub fn validate(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeBackgroundRemovalError> {
        cancellation.check()?;
        if self.artifact_sha256.len() != 64
            || self.semantic_digest_sha256.len() != 64
            || self.state.is_empty()
            || self.memory_budget_bytes == 0
        {
            return Err(NativeBackgroundRemovalError::SemanticStateChanged);
        }
        let specifications = birefnet_state_manifest(&self.profile)?;
        validate_strict_state_without_backend(
            &self.state,
            &specifications,
            self.stream,
            cancellation,
        )?;
        let digest = semantic_digest(
            &self.profile,
            &self.artifact_sha256,
            &self.state,
            cancellation,
        )?;
        if digest != self.semantic_digest_sha256
            || resident_state_bytes(&self.state, cancellation)?
                .checked_add(self.resident_owned_bytes()?)
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?
                != self.resident_bytes
        {
            return Err(NativeBackgroundRemovalError::SemanticStateChanged);
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn encode_image(
        &self,
        backend: &CpuBackend,
        image: &ImageTensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeBackgroundRemovalError> {
        self.normalize_execution_result(self.encode_image_inner(backend, image, context), context)
    }

    fn encode_image_inner(
        &self,
        backend: &CpuBackend,
        image: &ImageTensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeBackgroundRemovalError> {
        context.check()?;
        if context.stream != self.stream {
            return Err(NativeBackgroundRemovalError::InvalidInput(
                "image execution stream does not match retained state".to_owned(),
            ));
        }
        let (batch, height, width, channels) = image.dimensions()?;
        if batch == 0 || height == 0 || width == 0 || !matches!(channels, 3 | 4) {
            return Err(NativeBackgroundRemovalError::InvalidInput(
                "input must be nonempty RGB or RGBA IMAGE".to_owned(),
            ));
        }
        let required = self.invocation_memory_upper_bound(batch, height, width)?;
        if required > self.memory_budget_bytes {
            return Err(NativeBackgroundRemovalError::OutOfMemory {
                required,
                budget: self.memory_budget_bytes,
            });
        }
        let preprocessed = preprocess_image(backend, image, self.profile.image_size, context)?;
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(
                usize::try_from(batch).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            )
            .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
        for batch_index in 0..batch {
            context.check()?;
            logits.push(birefnet_single(
                self,
                backend,
                &preprocessed,
                usize::try_from(batch_index)
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                context,
            )?);
        }
        project_masks(backend, &logits, height, width, context)
    }

    fn state_tensor(&self, key: &str) -> Result<&Tensor, NativeBackgroundRemovalError> {
        self.state
            .get(key)
            .ok_or_else(|| NativeBackgroundRemovalError::MissingState(key.to_owned()))
    }

    fn invocation_memory_upper_bound(
        &self,
        batch: u64,
        height: u64,
        width: u64,
    ) -> Result<u64, NativeBackgroundRemovalError> {
        let image = checked_bytes(&[batch, height, width, 4])?;
        let square = u64::try_from(self.profile.image_size)
            .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
        let preprocessed = checked_bytes(&[batch, 3, square, square])?;
        let accumulated_logits = checked_bytes(&[batch, 1, square, square])?;
        let embed = u64::try_from(self.profile.embed_dim)
            .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
        let patch = square.div_ceil(4);
        let backbone = checked_bytes(&[1, embed * 16, patch, patch])?
            .checked_mul(2)
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
        let window = u64::try_from(self.profile.window_size)
            .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
        let window_tokens = window
            .checked_mul(window)
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
        let mut attention = 0_u64;
        for layer in 0..4 {
            let divisor = 1_u64 << layer;
            let stage = patch.div_ceil(divisor);
            let windows = stage.div_ceil(window);
            attention = attention.max(checked_bytes(&[
                windows * windows,
                u64::try_from(self.profile.heads[layer])
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                window_tokens,
                window_tokens,
            ])?);
        }
        let aspp = checked_bytes(&[
            5,
            u64::try_from(self.profile.aspp_channels)
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            patch,
            patch,
        ])?;
        self.resident_bytes
            .checked_add(image)
            .and_then(|value| value.checked_add(preprocessed.checked_mul(3)?))
            .and_then(|value| value.checked_add(accumulated_logits.checked_mul(2)?))
            .and_then(|value| value.checked_add(backbone.checked_mul(3)?))
            .and_then(|value| value.checked_add(attention.checked_mul(2)?))
            .and_then(|value| value.checked_add(aspp.checked_mul(3)?))
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)
    }

    fn normalize_execution_result<T>(
        &self,
        result: Result<T, NativeBackgroundRemovalError>,
        context: &ExecutionContext<'_>,
    ) -> Result<T, NativeBackgroundRemovalError> {
        match result {
            Err(_) if context.cancellation.is_cancelled() => {
                Err(NativeBackgroundRemovalError::Cancelled)
            }
            other => other,
        }
    }
}

#[derive(Debug, Error)]
pub enum NativeBackgroundRemovalError {
    #[error("background-removal execution was cancelled")]
    Cancelled,
    #[error("background-removal checkpoint is invalid: {0}")]
    InvalidCheckpoint(String),
    #[error("background-removal input is invalid: {0}")]
    InvalidInput(String),
    #[error("background-removal checkpoint has duplicate state key {0}")]
    DuplicateStateKey(String),
    #[error("background-removal checkpoint has no supported architecture marker")]
    UnsupportedArchitecture,
    #[error("background-removal checkpoint is missing state key {0}")]
    MissingState(String),
    #[error("background-removal checkpoint has unexpected state key {0}")]
    UnexpectedState(String),
    #[error(
        "background-removal state {key} expected {expected:?} {expected_dtype:?}, got {actual:?} {actual_dtype:?}"
    )]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        expected_dtype: DType,
        actual_dtype: DType,
    },
    #[error("background-removal relative-position index {key} is out of range")]
    RelativePositionIndex { key: String },
    #[error("background-removal semantic state changed")]
    SemanticStateChanged,
    #[error("background-removal memory requirement {required} exceeds budget {budget}")]
    OutOfMemory { required: u64, budget: u64 },
    #[error("background-removal shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("background-removal allocation failed")]
    Allocation,
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Attention(AttentionError),
    #[error(transparent)]
    Operator(OperatorIndirectionError),
    #[error(transparent)]
    Functional(FunctionalError),
    #[error(transparent)]
    NeuralFunctional(NeuralNetworkFunctionalError),
    #[error(transparent)]
    NeuralModule(NeuralNetworkModuleError),
    #[error(transparent)]
    Spatial(SpatialFunctionalKernelError),
    #[error(transparent)]
    External(ExternalTensorKernelPartTwoError),
    #[error(transparent)]
    ElementwiseThree(ElementwiseRuntimePartThreeError),
    #[error(transparent)]
    ElementwiseSix(ElementwiseRuntimePartSixError),
    #[error(transparent)]
    ElementwiseSixteen(ElementwiseRuntimePartSixteenError),
    #[error(transparent)]
    ElementwiseSeventeen(ElementwiseRuntimePartSeventeenError),
    #[error(transparent)]
    ElementwiseFive(ElementwiseRuntimePartFiveError),
    #[error(transparent)]
    Indexing(IndexingMaskingPartOneError),
    #[error(transparent)]
    ShapeLayoutTwo(ShapeLayoutTransformPartTwoError),
    #[error(transparent)]
    ShapeLayoutThree(ShapeLayoutTransformPartThreeError),
}

impl From<sim_types::CancellationError> for NativeBackgroundRemovalError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

fn birefnet_state_manifest(
    profile: &BiRefNetProfile,
) -> Result<Vec<StateSpecification>, NativeBackgroundRemovalError> {
    let mut specifications = Vec::new();
    add_conv(
        &mut specifications,
        "bb.patch_embed.proj",
        profile.embed_dim,
        3,
        4,
        true,
    )?;
    add_affine(
        &mut specifications,
        "bb.patch_embed.norm",
        profile.embed_dim,
    )?;
    let relative_positions = (2 * profile.window_size - 1)
        .checked_mul(2 * profile.window_size - 1)
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    let window_tokens = profile
        .window_size
        .checked_mul(profile.window_size)
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    for layer in 0..4 {
        let channels = profile
            .embed_dim
            .checked_mul(1 << layer)
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
        for block in 0..profile.depths[layer] {
            let prefix = format!("bb.layers.{layer}.blocks.{block}");
            add_affine(&mut specifications, &format!("{prefix}.norm1"), channels)?;
            add_state(
                &mut specifications,
                format!("{prefix}.attn.relative_position_bias_table"),
                &[relative_positions, profile.heads[layer]],
                StateKind::Float,
            )?;
            add_state(
                &mut specifications,
                format!("{prefix}.attn.relative_position_index"),
                &[window_tokens, window_tokens],
                StateKind::RelativePositionIndex {
                    maximum: u64::try_from(relative_positions)
                        .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                },
            )?;
            add_linear(
                &mut specifications,
                &format!("{prefix}.attn.qkv"),
                channels * 3,
                channels,
                true,
            )?;
            add_linear(
                &mut specifications,
                &format!("{prefix}.attn.proj"),
                channels,
                channels,
                true,
            )?;
            add_affine(&mut specifications, &format!("{prefix}.norm2"), channels)?;
            add_linear(
                &mut specifications,
                &format!("{prefix}.mlp.fc1"),
                channels * 4,
                channels,
                true,
            )?;
            add_linear(
                &mut specifications,
                &format!("{prefix}.mlp.fc2"),
                channels,
                channels * 4,
                true,
            )?;
        }
        if layer < 3 {
            add_linear(
                &mut specifications,
                &format!("bb.layers.{layer}.downsample.reduction"),
                channels * 2,
                channels * 4,
                false,
            )?;
            add_affine(
                &mut specifications,
                &format!("bb.layers.{layer}.downsample.norm"),
                channels * 4,
            )?;
        }
        add_affine(&mut specifications, &format!("bb.norm{layer}"), channels)?;
    }
    add_decoder_manifest(&mut specifications, profile)?;
    Ok(specifications)
}

fn add_decoder_manifest(
    specifications: &mut Vec<StateSpecification>,
    profile: &BiRefNetProfile,
) -> Result<(), NativeBackgroundRemovalError> {
    let channels = profile.channels()?;
    add_basic_decoder_block(
        specifications,
        "squeeze_module.0",
        channels[0]
            .checked_add(channels[1] + channels[2] + channels[3])
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?,
        channels[0],
        profile,
    )?;
    let stage_spatial = backbone_stage_spatial(profile.image_size)?;
    let patch_inputs = [
        split_patch_channels(profile.image_size, stage_spatial[3])?,
        split_patch_channels(profile.image_size, stage_spatial[2])?,
        split_patch_channels(profile.image_size, stage_spatial[1])?,
        split_patch_channels(profile.image_size, stage_spatial[0])?,
        3,
    ];
    if profile.source_exact && patch_inputs != [3_072, 768, 192, 48, 3] {
        return Err(NativeBackgroundRemovalError::InvalidCheckpoint(
            "source split-patch geometry changed".to_owned(),
        ));
    }
    let patch_outputs = [
        channels[0] / 8,
        channels[0] / 8,
        channels[1] / 8,
        channels[2] / 8,
        channels[3] / 8,
    ];
    for index in 0..5 {
        add_simple_convs(
            specifications,
            &format!("decoder.ipt_blk{}", 5 - index),
            patch_inputs[index],
            patch_outputs[index],
            profile.decoder_inter_channels,
        )?;
    }
    add_basic_decoder_block(
        specifications,
        "decoder.decoder_block4",
        channels[0] + patch_outputs[0],
        channels[1],
        profile,
    )?;
    add_basic_decoder_block(
        specifications,
        "decoder.decoder_block3",
        channels[1] + patch_outputs[1],
        channels[2],
        profile,
    )?;
    add_basic_decoder_block(
        specifications,
        "decoder.decoder_block2",
        channels[2] + patch_outputs[2],
        channels[3],
        profile,
    )?;
    add_basic_decoder_block(
        specifications,
        "decoder.decoder_block1",
        channels[3] + patch_outputs[3],
        channels[3] / 2,
        profile,
    )?;
    add_conv(
        specifications,
        "decoder.conv_out1.0",
        1,
        channels[3] / 2 + patch_outputs[4],
        1,
        true,
    )?;
    add_conv(
        specifications,
        "decoder.lateral_block4.conv",
        channels[1],
        channels[1],
        1,
        true,
    )?;
    add_conv(
        specifications,
        "decoder.lateral_block3.conv",
        channels[2],
        channels[2],
        1,
        true,
    )?;
    add_conv(
        specifications,
        "decoder.lateral_block2.conv",
        channels[3],
        channels[3],
        1,
        true,
    )?;
    for (suffix, input_channels) in [(4, channels[1]), (3, channels[2]), (2, channels[3])] {
        add_conv(
            specifications,
            &format!("decoder.conv_ms_spvn_{suffix}"),
            1,
            input_channels,
            1,
            true,
        )?;
    }
    for (suffix, input_channels) in [
        (4, channels[0] / 2),
        (3, channels[1] / 2),
        (2, channels[2] / 2),
    ] {
        add_conv(
            specifications,
            &format!("decoder.gdt_convs_{suffix}.0"),
            profile.gdt_channels,
            input_channels,
            3,
            true,
        )?;
        add_batch_norm(
            specifications,
            &format!("decoder.gdt_convs_{suffix}.1"),
            profile.gdt_channels,
        )?;
        add_conv(
            specifications,
            &format!("decoder.gdt_convs_pred_{suffix}.0"),
            1,
            profile.gdt_channels,
            1,
            true,
        )?;
        add_conv(
            specifications,
            &format!("decoder.gdt_convs_attn_{suffix}.0"),
            1,
            profile.gdt_channels,
            1,
            true,
        )?;
    }
    Ok(())
}

fn add_basic_decoder_block(
    specifications: &mut Vec<StateSpecification>,
    prefix: &str,
    input_channels: usize,
    output_channels: usize,
    profile: &BiRefNetProfile,
) -> Result<(), NativeBackgroundRemovalError> {
    let inter = profile.decoder_inter_channels;
    add_conv(
        specifications,
        &format!("{prefix}.conv_in"),
        inter,
        input_channels,
        3,
        true,
    )?;
    add_batch_norm(specifications, &format!("{prefix}.bn_in"), inter)?;
    for (branch, kernel) in std::iter::once(("aspp1".to_owned(), 1)).chain(
        [1_usize, 3, 7]
            .into_iter()
            .enumerate()
            .map(|(index, kernel)| (format!("aspp_deforms.{index}"), kernel)),
    ) {
        let branch = format!("{prefix}.dec_att.{branch}");
        add_conv(
            specifications,
            &format!("{branch}.atrous_conv.offset_conv"),
            kernel * kernel * 2,
            inter,
            kernel,
            true,
        )?;
        add_conv(
            specifications,
            &format!("{branch}.atrous_conv.modulator_conv"),
            kernel * kernel,
            inter,
            kernel,
            true,
        )?;
        add_conv(
            specifications,
            &format!("{branch}.atrous_conv.regular_conv"),
            profile.aspp_channels,
            inter,
            kernel,
            false,
        )?;
        add_batch_norm(
            specifications,
            &format!("{branch}.bn"),
            profile.aspp_channels,
        )?;
    }
    add_conv(
        specifications,
        &format!("{prefix}.dec_att.global_avg_pool.1"),
        profile.aspp_channels,
        inter,
        1,
        false,
    )?;
    add_batch_norm(
        specifications,
        &format!("{prefix}.dec_att.global_avg_pool.2"),
        profile.aspp_channels,
    )?;
    add_conv(
        specifications,
        &format!("{prefix}.dec_att.conv1"),
        output_channels.min(inter),
        profile.aspp_channels * 5,
        1,
        false,
    )?;
    add_batch_norm(
        specifications,
        &format!("{prefix}.dec_att.bn1"),
        output_channels.min(inter),
    )?;
    add_conv(
        specifications,
        &format!("{prefix}.conv_out"),
        output_channels,
        output_channels.min(inter),
        3,
        true,
    )?;
    add_batch_norm(specifications, &format!("{prefix}.bn_out"), output_channels)?;
    Ok(())
}

fn add_simple_convs(
    specifications: &mut Vec<StateSpecification>,
    prefix: &str,
    input_channels: usize,
    output_channels: usize,
    inter_channels: usize,
) -> Result<(), NativeBackgroundRemovalError> {
    add_conv(
        specifications,
        &format!("{prefix}.conv1"),
        inter_channels,
        input_channels,
        3,
        true,
    )?;
    add_conv(
        specifications,
        &format!("{prefix}.conv_out"),
        output_channels,
        inter_channels,
        3,
        true,
    )
}

fn add_conv(
    specifications: &mut Vec<StateSpecification>,
    prefix: &str,
    output_channels: usize,
    input_channels: usize,
    kernel: usize,
    bias: bool,
) -> Result<(), NativeBackgroundRemovalError> {
    add_state(
        specifications,
        format!("{prefix}.weight"),
        &[output_channels, input_channels, kernel, kernel],
        StateKind::Float,
    )?;
    if bias {
        add_state(
            specifications,
            format!("{prefix}.bias"),
            &[output_channels],
            StateKind::Float,
        )?;
    }
    Ok(())
}

fn add_linear(
    specifications: &mut Vec<StateSpecification>,
    prefix: &str,
    output_features: usize,
    input_features: usize,
    bias: bool,
) -> Result<(), NativeBackgroundRemovalError> {
    add_state(
        specifications,
        format!("{prefix}.weight"),
        &[output_features, input_features],
        StateKind::Float,
    )?;
    if bias {
        add_state(
            specifications,
            format!("{prefix}.bias"),
            &[output_features],
            StateKind::Float,
        )?;
    }
    Ok(())
}

fn add_affine(
    specifications: &mut Vec<StateSpecification>,
    prefix: &str,
    features: usize,
) -> Result<(), NativeBackgroundRemovalError> {
    add_state(
        specifications,
        format!("{prefix}.weight"),
        &[features],
        StateKind::Float,
    )?;
    add_state(
        specifications,
        format!("{prefix}.bias"),
        &[features],
        StateKind::Float,
    )
}

fn add_batch_norm(
    specifications: &mut Vec<StateSpecification>,
    prefix: &str,
    channels: usize,
) -> Result<(), NativeBackgroundRemovalError> {
    add_affine(specifications, prefix, channels)?;
    add_state(
        specifications,
        format!("{prefix}.running_mean"),
        &[channels],
        StateKind::Float,
    )?;
    add_state(
        specifications,
        format!("{prefix}.running_var"),
        &[channels],
        StateKind::Float,
    )?;
    add_state(
        specifications,
        format!("{prefix}.num_batches_tracked"),
        &[],
        StateKind::BatchCounter,
    )
}

fn add_state(
    specifications: &mut Vec<StateSpecification>,
    key: String,
    shape: &[usize],
    kind: StateKind,
) -> Result<(), NativeBackgroundRemovalError> {
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(shape.len())
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for value in shape {
        converted
            .push(u64::try_from(*value).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?);
    }
    specifications.push(StateSpecification {
        key,
        shape: converted,
        kind,
    });
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn deterministic_reduced_test_checkpoint(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    mutation: BackgroundRemovalFixtureMutation,
) -> Result<NativeBackgroundRemovalCheckpoint, NativeBackgroundRemovalError> {
    let profile = BiRefNetProfile::reduced();
    let specifications = birefnet_state_manifest(&profile)?;
    let float_mutation = match mutation {
        BackgroundRemovalFixtureMutation::None
        | BackgroundRemovalFixtureMutation::RelativePositionIndex => None,
        BackgroundRemovalFixtureMutation::ShiftedWindowBlock => {
            Some(("bb.layers.0.blocks.1.attn.proj.bias", 0, 1.0))
        }
        BackgroundRemovalFixtureMutation::DeformOffset => Some((
            "decoder.decoder_block1.dec_att.aspp_deforms.1.atrous_conv.offset_conv.weight",
            139,
            -100.0,
        )),
        BackgroundRemovalFixtureMutation::DeformMask => Some((
            "decoder.decoder_block1.dec_att.aspp_deforms.1.atrous_conv.modulator_conv.weight",
            85,
            100.0,
        )),
        BackgroundRemovalFixtureMutation::AsppDilatedBranch => Some((
            "decoder.decoder_block1.dec_att.aspp_deforms.2.atrous_conv.regular_conv.weight",
            73,
            10.0,
        )),
        BackgroundRemovalFixtureMutation::AsppGlobalPool => Some((
            "decoder.decoder_block1.dec_att.global_avg_pool.1.weight",
            1,
            0.125,
        )),
        BackgroundRemovalFixtureMutation::UnusedDecoderHead => {
            Some(("decoder.conv_ms_spvn_4.weight", 0, 0.125))
        }
    };
    let mut ordered_state = Vec::new();
    ordered_state
        .try_reserve_exact(specifications.len())
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for (state_index, specification) in specifications.into_iter().enumerate() {
        if state_index.is_multiple_of(32) {
            context.check()?;
        }
        let value_count = specification
            .shape
            .iter()
            .try_fold(1_usize, |count, dimension| {
                count
                    .checked_mul(usize_from(*dimension)?)
                    .ok_or(NativeBackgroundRemovalError::ShapeOverflow)
            })?;
        let tensor = match specification.kind {
            StateKind::Float => {
                let mut values = fallible_filled(value_count, 0.0)?;
                fill_fixture_float_state(&specification.key, &mut values);
                if let Some((mutation_key, mutation_index, mutation_delta)) = float_mutation
                    && mutation_key == specification.key
                {
                    let selected = values
                        .get_mut(mutation_index)
                        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
                    *selected += mutation_delta;
                }
                tensor_from_f32_with_context_exact_native(
                    backend,
                    &specification.shape,
                    &values,
                    DType::F32,
                    DeviceId::CPU,
                    context,
                )
                .map_err(tensor_operation)?
            }
            StateKind::RelativePositionIndex { maximum } => {
                let mut values = relative_position_indices(profile.window_size)?;
                if mutation == BackgroundRemovalFixtureMutation::RelativePositionIndex
                    && specification.key == "bb.layers.0.blocks.0.attn.relative_position_index"
                {
                    let first = values
                        .first_mut()
                        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
                    *first = (*first + 3)
                        % i64::try_from(maximum)
                            .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
                }
                upload_i64_tensor(backend, &specification.shape, &values, context)?
            }
            StateKind::BatchCounter => {
                upload_i64_tensor(backend, &specification.shape, &[0], context)?
            }
        };
        ordered_state.push((specification.key, tensor));
    }
    Ok(NativeBackgroundRemovalCheckpoint {
        artifact_sha256: BIREFNET_SOURCE_SHA256.to_owned(),
        ordered_state,
        memory_budget_bytes: 512 * 1024 * 1024,
    })
}

#[cfg(any(test, feature = "test-support"))]
fn fill_fixture_float_state(key: &str, values: &mut [f32]) {
    if key.ends_with("running_var")
        || (key.ends_with(".weight") && (key.contains(".norm") || key.contains(".bn")))
    {
        values.fill(1.0);
        return;
    }
    if key == "decoder.decoder_block1.dec_att.bn1.bias" {
        values.fill(0.1);
        return;
    }
    if key.ends_with(".bias") || key.ends_with("running_mean") {
        return;
    }
    let seed = key.bytes().fold(2_166_136_261_u32, |hash, byte| {
        hash.wrapping_mul(16_777_619) ^ u32::from(byte)
    });
    for (index, value) in values.iter_mut().enumerate() {
        let lane = seed.wrapping_add((index as u32).wrapping_mul(2_654_435_761));
        let scale = if key.starts_with("decoder.decoder_block1.")
            || key.starts_with("bb.layers.0.blocks.0.attn.")
        {
            0.1
        } else {
            0.001
        };
        *value = ((lane % 17) as f32 - 8.0) * scale;
    }
}

#[cfg(any(test, feature = "test-support"))]
fn relative_position_indices(window: usize) -> Result<Vec<i64>, NativeBackgroundRemovalError> {
    let tokens = window
        .checked_mul(window)
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    let width = window
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(tokens * tokens)
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for query in 0..tokens {
        for key in 0..tokens {
            let relative_y =
                (query / window) as isize - (key / window) as isize + (window - 1) as isize;
            let relative_x =
                (query % window) as isize - (key % window) as isize + (window - 1) as isize;
            let index = relative_y
                .checked_mul(width as isize)
                .and_then(|value| value.checked_add(relative_x))
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
            indices.push(
                i64::try_from(index).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            );
        }
    }
    Ok(indices)
}

#[cfg(any(test, feature = "test-support"))]
fn upload_i64_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(std::mem::size_of_val(values))
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(8_192) {
            context.check()?;
        }
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn validate_strict_state(
    state: &BTreeMap<String, Tensor>,
    specifications: &[StateSpecification],
    context: &ExecutionContext<'_>,
) -> Result<(), NativeBackgroundRemovalError> {
    let stream = state
        .values()
        .next()
        .ok_or(NativeBackgroundRemovalError::UnsupportedArchitecture)?
        .descriptor()
        .stream();
    if stream != context.stream {
        return Err(NativeBackgroundRemovalError::InvalidCheckpoint(
            "retained state stream does not match admission context".to_owned(),
        ));
    }
    validate_strict_state_without_backend(state, specifications, stream, context.cancellation)
}

fn validate_strict_state_without_backend(
    state: &BTreeMap<String, Tensor>,
    specifications: &[StateSpecification],
    stream: StreamId,
    cancellation: &CancellationToken,
) -> Result<(), NativeBackgroundRemovalError> {
    cancellation.check()?;
    let expected = specifications
        .iter()
        .map(|specification| specification.key.as_str())
        .collect::<BTreeSet<_>>();
    for key in expected.iter() {
        if !state.contains_key(*key) {
            return Err(NativeBackgroundRemovalError::MissingState(
                (*key).to_owned(),
            ));
        }
    }
    for key in state.keys() {
        if !expected.contains(key.as_str()) {
            return Err(NativeBackgroundRemovalError::UnexpectedState(key.clone()));
        }
    }
    for (index, specification) in specifications.iter().enumerate() {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        let tensor = state
            .get(&specification.key)
            .ok_or_else(|| NativeBackgroundRemovalError::MissingState(specification.key.clone()))?;
        let descriptor = tensor.descriptor();
        let expected_dtype = match specification.kind {
            StateKind::Float => DType::F32,
            StateKind::RelativePositionIndex { .. } | StateKind::BatchCounter => DType::I64,
        };
        if descriptor.shape() != specification.shape
            || descriptor.dtype() != expected_dtype
            || descriptor.device() != DeviceId::CPU
            || descriptor.stream() != stream
            || !descriptor.is_contiguous()?
        {
            return Err(NativeBackgroundRemovalError::StateShape {
                key: specification.key.clone(),
                expected: specification.shape.clone(),
                actual: descriptor.shape().to_vec(),
                expected_dtype,
                actual_dtype: descriptor.dtype(),
            });
        }
        match specification.kind {
            StateKind::Float => validate_finite_tensor(tensor, cancellation)?,
            StateKind::RelativePositionIndex { maximum } => {
                validate_relative_position_index(
                    tensor,
                    maximum,
                    &specification.key,
                    cancellation,
                )?;
            }
            StateKind::BatchCounter => validate_nonnegative_i64(tensor, &specification.key)?,
        }
    }
    cancellation.check()?;
    Ok(())
}

fn validate_finite_tensor(
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeBackgroundRemovalError> {
    let bytes = tensor.contiguous_bytes()?;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(NativeBackgroundRemovalError::InvalidCheckpoint(
                "state contains a non-finite value".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_relative_position_index(
    tensor: &Tensor,
    maximum: u64,
    key: &str,
    cancellation: &CancellationToken,
) -> Result<(), NativeBackgroundRemovalError> {
    for (index, chunk) in tensor.contiguous_bytes()?.chunks_exact(8).enumerate() {
        if index.is_multiple_of(8_192) {
            cancellation.check()?;
        }
        let value = i64::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        );
        if value < 0 || u64::try_from(value).map_or(true, |value| value >= maximum) {
            return Err(NativeBackgroundRemovalError::RelativePositionIndex {
                key: key.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_nonnegative_i64(
    tensor: &Tensor,
    key: &str,
) -> Result<(), NativeBackgroundRemovalError> {
    let bytes = tensor.contiguous_bytes()?;
    if bytes.len() != 8
        || i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        ) < 0
    {
        return Err(NativeBackgroundRemovalError::InvalidCheckpoint(format!(
            "{key} is invalid"
        )));
    }
    Ok(())
}

fn semantic_digest(
    profile: &BiRefNetProfile,
    artifact_sha256: &str,
    state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, NativeBackgroundRemovalError> {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, b"zed.sim.background-removal.birefnet.v1")?;
    hash_field(&mut hasher, artifact_sha256.as_bytes())?;
    for value in [
        profile.image_size,
        profile.embed_dim,
        profile.window_size,
        profile.decoder_inter_channels,
        profile.aspp_channels,
        profile.gdt_channels,
    ] {
        hash_field(
            &mut hasher,
            &u64::try_from(value)
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?
                .to_le_bytes(),
        )?;
    }
    for value in profile.depths.into_iter().chain(profile.heads) {
        hash_field(
            &mut hasher,
            &u64::try_from(value)
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?
                .to_le_bytes(),
        )?;
    }
    for (index, (key, tensor)) in state.iter().enumerate() {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        hash_field(&mut hasher, key.as_bytes())?;
        hash_field(
            &mut hasher,
            &[match tensor.descriptor().dtype() {
                DType::F32 => 1,
                DType::I64 => 2,
                _ => return Err(NativeBackgroundRemovalError::SemanticStateChanged),
            }],
        )?;
        for dimension in tensor.descriptor().shape() {
            hash_field(&mut hasher, &dimension.to_le_bytes())?;
        }
        for chunk in tensor.contiguous_bytes()?.chunks(64 * 1_024) {
            cancellation.check()?;
            hasher.update(chunk);
        }
    }
    cancellation.check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn resident_state_bytes(
    state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<u64, NativeBackgroundRemovalError> {
    resident_tensor_allocations(state, cancellation)?
        .into_iter()
        .try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(bytes)
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)
        })
}

fn resident_tensor_allocations(
    state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<Vec<(StorageId, u64)>, NativeBackgroundRemovalError> {
    let mut unique = BTreeMap::new();
    for (index, tensor) in state.values().enumerate() {
        if index.is_multiple_of(32) {
            cancellation.check()?;
        }
        unique
            .entry(tensor.storage_id().get())
            .or_insert((tensor.storage_id(), tensor.storage_byte_len()));
    }
    Ok(unique.into_values().collect())
}

fn resident_owned_bytes(
    artifact_sha256: &String,
    semantic_digest_sha256: &String,
    state: &BTreeMap<String, Tensor>,
) -> Result<u64, NativeBackgroundRemovalError> {
    let mut total = u64::try_from(std::mem::size_of::<NativeBackgroundRemovalResource>())
        .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
    total = total
        .checked_add(
            u64::try_from(artifact_sha256.capacity())
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        )
        .and_then(|value| value.checked_add(u64::try_from(semantic_digest_sha256.capacity()).ok()?))
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    for key in state.keys() {
        total = total
            .checked_add(
                u64::try_from(key.capacity())
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            )
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    }
    Ok(total)
}

fn validate_state_key(key: &str) -> Result<(), NativeBackgroundRemovalError> {
    if key.is_empty()
        || key.len() > MAX_STATE_KEY_BYTES
        || key.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(NativeBackgroundRemovalError::InvalidCheckpoint(
            "state key is invalid".to_owned(),
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), NativeBackgroundRemovalError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NativeBackgroundRemovalError::InvalidCheckpoint(
            "artifact SHA-256 is invalid".to_owned(),
        ));
    }
    Ok(())
}

fn hash_field(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), NativeBackgroundRemovalError> {
    hasher.update(
        u64::try_from(bytes.len())
            .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(bytes);
    Ok(())
}

fn checked_bytes(shape: &[u64]) -> Result<u64, NativeBackgroundRemovalError> {
    shape.iter().try_fold(4_u64, |total, value| {
        total
            .checked_mul(*value)
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)
    })
}

fn backbone_stage_spatial(image_size: usize) -> Result<[usize; 4], NativeBackgroundRemovalError> {
    let stage0 = image_size.div_ceil(4);
    if stage0 == 0 {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    }
    let stage1 = stage0.div_ceil(2);
    let stage2 = stage1.div_ceil(2);
    let stage3 = stage2.div_ceil(2);
    Ok([stage0, stage1, stage2, stage3])
}

fn split_patch_channels(
    image_size: usize,
    patch_extent: usize,
) -> Result<usize, NativeBackgroundRemovalError> {
    if patch_extent == 0 {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    }
    image_size
        .div_ceil(patch_extent)
        .checked_mul(image_size.div_ceil(patch_extent))
        .and_then(|value| value.checked_mul(3))
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)
}

fn preprocess_image(
    backend: &CpuBackend,
    image: &ImageTensor,
    size: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    context.check()?;
    let (_, height, width, channels) = image.dimensions()?;
    let rgb = if channels == 4 {
        narrow_method_exact_native(image.tensor(), 3, 0, 3, context.cancellation)
            .map_err(tensor_operation)?
    } else {
        image.tensor().clone()
    };
    let channel_count =
        usize::try_from(channels).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
    for (pixel_index, pixel) in image
        .as_f32_slice()?
        .chunks_exact(channel_count)
        .enumerate()
    {
        if pixel_index.is_multiple_of(4_096) {
            context.check()?;
        }
        if pixel
            .get(..3)
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(NativeBackgroundRemovalError::InvalidInput(
                "input IMAGE contains a non-finite channel".to_owned(),
            ));
        }
    }
    let input = tensor_permute_exact_native(&rgb, &[0, 3, 1, 2], context.cancellation)
        .map_err(tensor_operation)?;
    let resized = if height
        == u64::try_from(size).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?
        && width == u64::try_from(size).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?
    {
        input
    } else {
        interpolate_tensor_with_context_exact_native(
            backend,
            &input,
            &InterpolateConfiguration {
                output_size: Some(vec![size, size]),
                scale_factor: None,
                mode: InterpolateMode::Bicubic,
                align_corners: None,
                recompute_scale_factor: None,
                antialias: true,
            },
            context,
        )
        .map_err(tensor_operation)?
    };
    let scaled = multiply_scalar(backend, &resized, 255.0, context)?;
    let clipped = clip_with_context_exact_native(backend, &scaled, Some(0.0), Some(255.0), context)
        .map_err(tensor_operation)?;
    let rounded = round_method_with_context_exact_native(backend, &clipped, 0, context)
        .map_err(tensor_operation)?;
    div_with_context_exact_native(
        backend,
        &rounded,
        ElementwiseOperand::Scalar(Scalar::Float(255.0)),
        context,
    )
    .map_err(tensor_operation)
}

fn birefnet_single(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    preprocessed: &Tensor,
    batch_index: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    context.check()?;
    let [batch, channels, height, width] = preprocessed.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::InvalidInput(
            "preprocessed image rank changed".to_owned(),
        ));
    };
    if *channels != 3 || batch_index >= usize_from(*batch)? {
        return Err(NativeBackgroundRemovalError::InvalidInput(
            "serial image index is invalid".to_owned(),
        ));
    }
    let image = narrow_method_exact_native(
        preprocessed,
        0,
        i64::try_from(batch_index).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        1,
        context.cancellation,
    )
    .map_err(tensor_operation)?;
    let full = swin_backbone(resource, backend, &image, context)?;
    let half_input = resize_tensor(
        backend,
        &image,
        usize_from(*height)? / 2,
        usize_from(*width)? / 2,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    let half = swin_backbone(resource, backend, &half_input, context)?;
    let mut features = Vec::new();
    features
        .try_reserve_exact(4)
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for index in 0..4 {
        let full_feature = full
            .get(index)
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
        let [_, _, feature_height, feature_width] = full_feature.descriptor().shape() else {
            return Err(NativeBackgroundRemovalError::ShapeOverflow);
        };
        let resized_half = resize_tensor(
            backend,
            half.get(index)
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?,
            usize_from(*feature_height)?,
            usize_from(*feature_width)?,
            InterpolateMode::Bilinear,
            Some(true),
            false,
            context,
        )?;
        features.push(concat_channels(
            backend,
            &[full_feature, &resized_half],
            context,
        )?);
    }
    let x1 = features.remove(0);
    let x2 = features.remove(0);
    let x3 = features.remove(0);
    let x4 = features.remove(0);
    let [_, _, x4_height, x4_width] = x4.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    };
    let x1_context = resize_tensor(
        backend,
        &x1,
        usize_from(*x4_height)?,
        usize_from(*x4_width)?,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    let x2_context = resize_tensor(
        backend,
        &x2,
        usize_from(*x4_height)?,
        usize_from(*x4_width)?,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    let x3_context = resize_tensor(
        backend,
        &x3,
        usize_from(*x4_height)?,
        usize_from(*x4_width)?,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    let squeezed_input = concat_channels(
        backend,
        &[&x1_context, &x2_context, &x3_context, &x4],
        context,
    )?;
    let squeezed = basic_decoder_block(
        resource,
        backend,
        &squeezed_input,
        "squeeze_module.0",
        context,
    )?;
    decode_birefnet(resource, backend, &image, &x1, &x2, &x3, &squeezed, context)
}

fn swin_backbone(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, NativeBackgroundRemovalError> {
    let profile = &resource.profile;
    let projected = conv2d(
        resource,
        backend,
        input,
        "bb.patch_embed.proj",
        4,
        0,
        context,
    )?;
    let [batch, channels, initial_height, initial_width] = projected.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    };
    if *batch != 1 || usize_from(*channels)? != profile.embed_dim {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    }
    let mut height = *initial_height;
    let mut width = *initial_width;
    let mut tokens = nchw_to_tokens(backend, &projected, context)?;
    tokens = layer_norm_values(
        resource,
        backend,
        &tokens,
        &[
            1,
            usize_from(height)? * usize_from(width)?,
            profile.embed_dim,
        ],
        "bb.patch_embed.norm",
        context,
    )?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(4)
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for layer in 0..4 {
        let stage_channels = profile
            .embed_dim
            .checked_mul(1 << layer)
            .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
        for block in 0..profile.depths[layer] {
            tokens = swin_block(
                resource,
                backend,
                &tokens,
                usize_from(height)?,
                usize_from(width)?,
                layer,
                block,
                context,
            )?;
        }
        let normalized = layer_norm_values(
            resource,
            backend,
            &tokens,
            &[1, usize_from(height)? * usize_from(width)?, stage_channels],
            &format!("bb.norm{layer}"),
            context,
        )?;
        outputs.push(tokens_to_nchw(
            backend,
            &normalized,
            1,
            usize_from(height)?,
            usize_from(width)?,
            stage_channels,
            context,
        )?);
        if layer < 3 {
            let merged = patch_merge(
                resource,
                backend,
                &tokens,
                usize_from(height)?,
                usize_from(width)?,
                stage_channels,
                layer,
                context,
            )?;
            tokens = merged.0;
            height =
                u64::try_from(merged.1).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
            width =
                u64::try_from(merged.2).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
        }
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
fn swin_block(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &[f32],
    height: usize,
    width: usize,
    layer: usize,
    block: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeBackgroundRemovalError> {
    context.check()?;
    let channels = resource.profile.embed_dim << layer;
    let heads = resource.profile.heads[layer];
    let window = resource.profile.window_size;
    let shift = if block.is_multiple_of(2) {
        0
    } else {
        window / 2
    };
    let prefix = format!("bb.layers.{layer}.blocks.{block}");
    let normalized = layer_norm_values(
        resource,
        backend,
        input,
        &[1, height * width, channels],
        &format!("{prefix}.norm1"),
        context,
    )?;
    let padded_height = height.div_ceil(window) * window;
    let padded_width = width.div_ceil(window) * window;
    let normalized = tokens_to_nchw(backend, &normalized, 1, height, width, channels, context)?;
    let padded = if padded_height != height || padded_width != width {
        functional_pad_with_context_exact_native(
            backend,
            &normalized,
            &[
                0,
                i64::try_from(padded_width - width)
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                0,
                i64::try_from(padded_height - height)
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            ],
            FunctionalPadMode::Constant,
            None,
            context,
        )
        .map_err(tensor_operation)?
    } else {
        normalized
    };
    let shifted = if shift == 0 {
        padded
    } else {
        roll_with_context_exact_native(
            backend,
            &padded,
            &[-i64::try_from(shift).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?; 2],
            Some(&[2, 3]),
            context,
        )
        .map_err(tensor_operation)?
    };
    let padded = nchw_to_tokens(backend, &shifted, context)?;
    let windows_y = padded_height / window;
    let windows_x = padded_width / window;
    let window_count = windows_y * windows_x;
    let tokens = window * window;
    let mut window_values = fallible_filled(window_count * tokens * channels, 0.0)?;
    for window_y in 0..windows_y {
        for window_x in 0..windows_x {
            context.check()?;
            let window_index = window_y * windows_x + window_x;
            for local_y in 0..window {
                for local_x in 0..window {
                    let token = local_y * window + local_x;
                    let y = window_y * window + local_y;
                    let x = window_x * window + local_x;
                    for channel in 0..channels {
                        window_values[(window_index * tokens + token) * channels + channel] =
                            padded[(y * padded_width + x) * channels + channel];
                    }
                }
            }
        }
    }
    let qkv = linear_values(
        resource,
        backend,
        &window_values,
        &[window_count, tokens, channels],
        &format!("{prefix}.attn.qkv"),
        context,
    )?;
    let head_dimension = channels / heads;
    let query_scale = (head_dimension as f64).powf(-0.5) as f32;
    if !query_scale.is_finite() || query_scale <= 0.0 {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    }
    let mut query = fallible_filled(window_count * tokens * heads * head_dimension, 0.0)?;
    let mut key = fallible_filled(query.len(), 0.0)?;
    let mut value = fallible_filled(query.len(), 0.0)?;
    for batch in 0..window_count {
        context.check()?;
        for token in 0..tokens {
            for head in 0..heads {
                for dimension in 0..head_dimension {
                    let destination =
                        ((batch * tokens + token) * heads + head) * head_dimension + dimension;
                    let base = (batch * tokens + token) * (3 * channels)
                        + head * head_dimension
                        + dimension;
                    query[destination] = qkv[base] * query_scale;
                    key[destination] = qkv[base + channels];
                    value[destination] = qkv[base + 2 * channels];
                }
            }
        }
    }
    let (relative_bias, shifted_mask) = window_attention_mask(
        resource,
        &prefix,
        padded_height,
        padded_width,
        window_count,
        heads,
        tokens,
        shift,
        context.cancellation,
    )?;
    let attention = scaled_dot_product_attention_with_context(
        backend,
        AttentionRequest {
            backend: AttentionBackend::PytorchSdp,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch: window_count,
            query_tokens: tokens,
            key_tokens: tokens,
            heads,
            head_dimension,
            value_dimension: head_dimension,
            scale: Some(1.0),
            workspace_limit_bytes: tokens
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?,
        },
        &query,
        &key,
        &value,
        Some(AttentionMask::OrderedAdditive {
            first_values: &relative_bias,
            second_values: &shifted_mask,
            shape: AttentionMaskShape::BatchHeadQueryByKey,
        }),
        context,
    )
    .map_err(tensor_operation)?
    .values;
    let projected = linear_values(
        resource,
        backend,
        &attention,
        &[window_count, tokens, channels],
        &format!("{prefix}.attn.proj"),
        context,
    )?;
    let mut reversed = fallible_filled(padded_height * padded_width * channels, 0.0)?;
    for window_y in 0..windows_y {
        for window_x in 0..windows_x {
            context.check()?;
            let window_index = window_y * windows_x + window_x;
            for local_y in 0..window {
                for local_x in 0..window {
                    let token = local_y * window + local_x;
                    let shifted_y = window_y * window + local_y;
                    let shifted_x = window_x * window + local_x;
                    for channel in 0..channels {
                        reversed[(shifted_y * padded_width + shifted_x) * channels + channel] =
                            projected[(window_index * tokens + token) * channels + channel];
                    }
                }
            }
        }
    }
    let reversed = tokens_to_nchw(
        backend,
        &reversed,
        1,
        padded_height,
        padded_width,
        channels,
        context,
    )?;
    let reversed = if shift == 0 {
        reversed
    } else {
        roll_with_context_exact_native(
            backend,
            &reversed,
            &[i64::try_from(shift).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?; 2],
            Some(&[2, 3]),
            context,
        )
        .map_err(tensor_operation)?
    };
    let reversed = narrow_method_exact_native(
        &reversed,
        2,
        0,
        u64::try_from(height).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        context.cancellation,
    )
    .map_err(tensor_operation)?;
    let reversed = narrow_method_exact_native(
        &reversed,
        3,
        0,
        u64::try_from(width).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        context.cancellation,
    )
    .map_err(tensor_operation)?;
    let input_tensor = tokens_to_nchw(backend, input, 1, height, width, channels, context)?;
    let residual_tensor = add_tensors(backend, &input_tensor, &reversed, context)?;
    let residual = nchw_to_tokens(backend, &residual_tensor, context)?;
    let normalized = layer_norm_values(
        resource,
        backend,
        &residual,
        &[1, height * width, channels],
        &format!("{prefix}.norm2"),
        context,
    )?;
    let hidden = linear_values(
        resource,
        backend,
        &normalized,
        &[1, height * width, channels],
        &format!("{prefix}.mlp.fc1"),
        context,
    )?;
    let hidden = gelu_with_context_exact_native(
        backend,
        &hidden,
        GeluApproximation::None,
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)?;
    let hidden = linear_values(
        resource,
        backend,
        &hidden,
        &[1, height * width, channels * 4],
        &format!("{prefix}.mlp.fc2"),
        context,
    )?;
    let residual = tokens_to_nchw(backend, &residual, 1, height, width, channels, context)?;
    let hidden = tokens_to_nchw(backend, &hidden, 1, height, width, channels, context)?;
    let output = add_tensors(backend, &residual, &hidden, context)?;
    nchw_to_tokens(backend, &output, context)
}

#[allow(clippy::too_many_arguments)]
fn window_attention_mask(
    resource: &NativeBackgroundRemovalResource,
    prefix: &str,
    padded_height: usize,
    padded_width: usize,
    window_count: usize,
    heads: usize,
    tokens: usize,
    shift: usize,
    cancellation: &CancellationToken,
) -> Result<(Vec<f32>, Vec<f32>), NativeBackgroundRemovalError> {
    let bias = tensor_f32(
        resource.state_tensor(&format!("{prefix}.attn.relative_position_bias_table"))?,
        cancellation,
    )?;
    let indices = tensor_i64(
        resource.state_tensor(&format!("{prefix}.attn.relative_position_index"))?,
        cancellation,
    )?;
    let window = resource.profile.window_size;
    let windows_x = padded_width / window;
    let mask_length = window_count
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(tokens))
        .and_then(|value| value.checked_mul(tokens))
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    let mut relative_bias = fallible_filled(mask_length, 0.0)?;
    let mut shifted_mask = fallible_filled(mask_length, 0.0)?;
    for window_index in 0..window_count {
        let window_y = window_index / windows_x;
        let window_x = window_index % windows_x;
        for query in 0..tokens {
            cancellation.check()?;
            let query_y = window_y * window + query / window;
            let query_x = window_x * window + query % window;
            let query_region = shift_region(query_y, padded_height, window, shift) * 3
                + shift_region(query_x, padded_width, window, shift);
            for key in 0..tokens {
                let key_y = window_y * window + key / window;
                let key_x = window_x * window + key % window;
                let key_region = shift_region(key_y, padded_height, window, shift) * 3
                    + shift_region(key_x, padded_width, window, shift);
                let position = usize::try_from(
                    *indices
                        .get(query * tokens + key)
                        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?,
                )
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?;
                for head in 0..heads {
                    let destination =
                        (((window_index * heads + head) * tokens + query) * tokens) + key;
                    relative_bias[destination] = bias[position * heads + head];
                    shifted_mask[destination] = if shift > 0 && query_region != key_region {
                        -100.0
                    } else {
                        0.0
                    };
                }
            }
        }
    }
    cancellation.check()?;
    Ok((relative_bias, shifted_mask))
}

fn shift_region(position: usize, extent: usize, window: usize, shift: usize) -> usize {
    if shift == 0 || position < extent.saturating_sub(window) {
        0
    } else if position < extent.saturating_sub(shift) {
        1
    } else {
        2
    }
}

#[allow(clippy::too_many_arguments)]
fn patch_merge(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &[f32],
    height: usize,
    width: usize,
    channels: usize,
    layer: usize,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<f32>, usize, usize), NativeBackgroundRemovalError> {
    let output_height = height.div_ceil(2);
    let output_width = width.div_ceil(2);
    let mut merged = fallible_filled(output_height * output_width * channels * 4, 0.0)?;
    for output_y in 0..output_height {
        for output_x in 0..output_width {
            context.check()?;
            for (part, (delta_y, delta_x)) in
                [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate()
            {
                let source_y = output_y * 2 + delta_y;
                let source_x = output_x * 2 + delta_x;
                if source_y < height && source_x < width {
                    for channel in 0..channels {
                        merged[((output_y * output_width + output_x) * channels * 4)
                            + part * channels
                            + channel] = input[(source_y * width + source_x) * channels + channel];
                    }
                }
            }
        }
    }
    let prefix = format!("bb.layers.{layer}.downsample");
    let normalized = layer_norm_values(
        resource,
        backend,
        &merged,
        &[1, output_height * output_width, channels * 4],
        &format!("{prefix}.norm"),
        context,
    )?;
    let output = linear_values(
        resource,
        backend,
        &normalized,
        &[1, output_height * output_width, channels * 4],
        &format!("{prefix}.reduction"),
        context,
    )?;
    Ok((output, output_height, output_width))
}

fn conv2d(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    stride: usize,
    padding: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let configuration = ConvolutionConfiguration {
        stride: vec![stride, stride],
        padding: vec![padding, padding],
        dilation: vec![1, 1],
        groups: 1,
        output_padding: vec![0, 0],
    };
    conv_2d_tensor_with_context_exact_native(
        backend,
        input,
        resource.state_tensor(&format!("{prefix}.weight"))?,
        resource.state.get(&format!("{prefix}.bias")),
        &configuration,
        context,
    )
    .map_err(tensor_operation)
}

fn resize_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    height: usize,
    width: usize,
    mode: InterpolateMode,
    align_corners: Option<bool>,
    antialias: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    if height == 0 || width == 0 {
        return Err(NativeBackgroundRemovalError::InvalidInput(
            "resize target must be nonzero".to_owned(),
        ));
    }
    interpolate_tensor_with_context_exact_native(
        backend,
        input,
        &InterpolateConfiguration {
            output_size: Some(vec![height, width]),
            scale_factor: None,
            mode,
            align_corners,
            recompute_scale_factor: None,
            antialias,
        },
        context,
    )
    .map_err(tensor_operation)
}

fn tensor_f32(
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NativeBackgroundRemovalError> {
    cancellation.check()?;
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(NativeBackgroundRemovalError::SemanticStateChanged);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(tensor.contiguous_bytes()?.len() / 4)
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for (index, chunk) in tensor.contiguous_bytes()?.chunks_exact(4).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        values.push(f32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        ));
    }
    Ok(values)
}

fn tensor_i64(
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<i64>, NativeBackgroundRemovalError> {
    cancellation.check()?;
    if tensor.descriptor().dtype() != DType::I64 {
        return Err(NativeBackgroundRemovalError::SemanticStateChanged);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(tensor.contiguous_bytes()?.len() / 8)
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for (index, chunk) in tensor.contiguous_bytes()?.chunks_exact(8).enumerate() {
        if index.is_multiple_of(8_192) {
            cancellation.check()?;
        }
        values.push(i64::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        ));
    }
    Ok(values)
}

fn layer_norm_values(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeBackgroundRemovalError> {
    let normalized = *shape
        .last()
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    let weight = tensor_f32(
        resource.state_tensor(&format!("{prefix}.weight"))?,
        context.cancellation,
    )?;
    let bias = tensor_f32(
        resource.state_tensor(&format!("{prefix}.bias"))?,
        context.cancellation,
    )?;
    layer_norm_with_context_exact_native(
        backend,
        input,
        shape,
        &[normalized],
        Some(&weight),
        Some(&bias),
        1.0e-5,
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)
}

fn linear_values(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeBackgroundRemovalError> {
    let weight_tensor = resource.state_tensor(&format!("{prefix}.weight"))?;
    let weight = tensor_f32(weight_tensor, context.cancellation)?;
    let weight_shape = weight_tensor
        .descriptor()
        .shape()
        .iter()
        .map(|value| usize_from(*value))
        .collect::<Result<Vec<_>, _>>()?;
    let bias = resource
        .state
        .get(&format!("{prefix}.bias"))
        .map(|tensor| tensor_f32(tensor, context.cancellation))
        .transpose()?;
    Ok(linear_with_context_exact_native(
        backend,
        input,
        input_shape,
        &weight,
        &weight_shape,
        bias.as_deref(),
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)?
    .values)
}

fn nchw_to_tokens(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeBackgroundRemovalError> {
    let [_, _, _, _] = tensor.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    };
    let permuted = tensor_permute_exact_native(tensor, &[0, 2, 3, 1], context.cancellation)
        .map_err(tensor_operation)?;
    tensor_to_f32_with_context_exact_native(backend, &permuted, context).map_err(tensor_operation)
}

#[allow(clippy::too_many_arguments)]
fn tokens_to_nchw(
    backend: &CpuBackend,
    values: &[f32],
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let nhwc = tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64::try_from(batch).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            u64::try_from(height).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            u64::try_from(width).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            u64::try_from(channels).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
        ],
        values,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)?;
    tensor_permute_exact_native(&nhwc, &[0, 3, 1, 2], context.cancellation)
        .map_err(tensor_operation)
}

fn concat_channels(
    backend: &CpuBackend,
    tensors: &[&Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    if tensors.is_empty() {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    }
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(tensors.len())
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for tensor in tensors {
        owned.push((*tensor).clone());
    }
    torch_cat_with_context_exact_native(backend, &owned, 1, context).map_err(tensor_operation)
}

fn add_tensors(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    add_method_with_context_exact_native(
        backend,
        left,
        ElementwiseOperand::Tensor(right),
        1.0,
        context,
    )
    .map_err(tensor_operation)
}

fn multiply_tensors(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    mul_method_with_context_exact_native(backend, left, right, context).map_err(tensor_operation)
}

fn usize_from(value: u64) -> Result<usize, NativeBackgroundRemovalError> {
    usize::try_from(value).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)
}

fn fallible_filled(length: usize, value: f32) -> Result<Vec<f32>, NativeBackgroundRemovalError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    output.resize(length, value);
    Ok(output)
}

fn batch_norm_eval(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let shape = input
        .descriptor()
        .shape()
        .iter()
        .map(|value| usize_from(*value))
        .collect::<Result<Vec<_>, _>>()?;
    let input_values = tensor_to_f32_with_context_exact_native(backend, input, context)
        .map_err(tensor_operation)?;
    let mut running_mean = tensor_f32(
        resource.state_tensor(&format!("{prefix}.running_mean"))?,
        context.cancellation,
    )?;
    let mut running_variance = tensor_f32(
        resource.state_tensor(&format!("{prefix}.running_var"))?,
        context.cancellation,
    )?;
    let weight = tensor_f32(
        resource.state_tensor(&format!("{prefix}.weight"))?,
        context.cancellation,
    )?;
    let bias = tensor_f32(
        resource.state_tensor(&format!("{prefix}.bias"))?,
        context.cancellation,
    )?;
    let values = batch_norm_with_context_exact_native(
        backend,
        &input_values,
        &shape,
        Some(&mut running_mean),
        Some(&mut running_variance),
        Some(&weight),
        Some(&bias),
        false,
        0.1,
        1.0e-5,
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)?;
    tensor_from_f32_with_context_exact_native(
        backend,
        input.descriptor().shape(),
        &values,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)
}

fn relu_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let input_values = tensor_to_f32_with_context_exact_native(backend, input, context)
        .map_err(tensor_operation)?;
    let values = relu_with_context_exact_native(backend, &input_values, DeviceId::CPU, context)
        .map_err(tensor_operation)?;
    tensor_from_f32_with_context_exact_native(
        backend,
        input.descriptor().shape(),
        &values,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)
}

fn sigmoid_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let sigmoid =
        sigmoid_with_context_exact_native(backend, input, context).map_err(tensor_operation)?;
    if scale == 1.0 {
        Ok(sigmoid)
    } else {
        multiply_scalar(backend, &sigmoid, scale, context)
    }
}

fn multiply_scalar(
    backend: &CpuBackend,
    input: &Tensor,
    scalar: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let scalar = tensor_from_f32_with_context_exact_native(
        backend,
        &[1],
        &[scalar],
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)?;
    mul_method_with_context_exact_native(backend, input, &scalar, context).map_err(tensor_operation)
}

fn basic_decoder_block(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let hidden = conv2d(
        resource,
        backend,
        input,
        &format!("{prefix}.conv_in"),
        1,
        1,
        context,
    )?;
    let hidden = batch_norm_eval(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.bn_in"),
        context,
    )?;
    let hidden = relu_tensor(backend, &hidden, context)?;
    let hidden = aspp_deformable(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.dec_att"),
        context,
    )?;
    let hidden = conv2d(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.conv_out"),
        1,
        1,
        context,
    )?;
    batch_norm_eval(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.bn_out"),
        context,
    )
}

fn aspp_deformable(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let mut branches = Vec::new();
    branches
        .try_reserve_exact(5)
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    branches.push(deformable_aspp_branch(
        resource,
        backend,
        input,
        &format!("{prefix}.aspp1"),
        1,
        context,
    )?);
    for (index, kernel) in [1_usize, 3, 7].into_iter().enumerate() {
        branches.push(deformable_aspp_branch(
            resource,
            backend,
            input,
            &format!("{prefix}.aspp_deforms.{index}"),
            kernel,
            context,
        )?);
    }
    let input_shape = input
        .descriptor()
        .shape()
        .iter()
        .map(|value| usize_from(*value))
        .collect::<Result<Vec<_>, _>>()?;
    let input_values = tensor_to_f32_with_context_exact_native(backend, input, context)
        .map_err(tensor_operation)?;
    let pooled = adaptive_average_pool_2d_with_context_exact_native(
        backend,
        &input_values,
        &input_shape,
        [1, 1],
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)?;
    let pooled = tensor_from_f32_with_context_exact_native(
        backend,
        &pooled
            .shape
            .iter()
            .map(|value| {
                u64::try_from(*value).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?,
        &pooled.values,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(tensor_operation)?;
    let pooled = conv2d(
        resource,
        backend,
        &pooled,
        &format!("{prefix}.global_avg_pool.1"),
        1,
        0,
        context,
    )?;
    let pooled = batch_norm_eval(
        resource,
        backend,
        &pooled,
        &format!("{prefix}.global_avg_pool.2"),
        context,
    )?;
    let pooled = relu_tensor(backend, &pooled, context)?;
    let [_, _, height, width] = input.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    };
    branches.push(resize_tensor(
        backend,
        &pooled,
        usize_from(*height)?,
        usize_from(*width)?,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?);
    let branch_refs = branches.iter().collect::<Vec<_>>();
    let hidden = concat_channels(backend, &branch_refs, context)?;
    let hidden = conv2d(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.conv1"),
        1,
        0,
        context,
    )?;
    let hidden = batch_norm_eval(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.bn1"),
        context,
    )?;
    relu_tensor(backend, &hidden, context)
}

fn deformable_aspp_branch(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    kernel: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let padding = kernel / 2;
    let offset = conv2d(
        resource,
        backend,
        input,
        &format!("{prefix}.atrous_conv.offset_conv"),
        1,
        padding,
        context,
    )?;
    let modulator = conv2d(
        resource,
        backend,
        input,
        &format!("{prefix}.atrous_conv.modulator_conv"),
        1,
        padding,
        context,
    )?;
    let modulator = sigmoid_tensor(backend, &modulator, 2.0, context)?;
    let hidden = deform_conv2d_with_context_exact_native(
        backend,
        input,
        &offset,
        resource.state_tensor(&format!("{prefix}.atrous_conv.regular_conv.weight"))?,
        None,
        NativeDeformConv2dConfiguration {
            stride: [1, 1],
            padding: [
                u64::try_from(padding).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                u64::try_from(padding).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            ],
            dilation: [1, 1],
        },
        Some(&modulator),
        context,
    )
    .map_err(tensor_operation)?;
    let hidden = batch_norm_eval(resource, backend, &hidden, &format!("{prefix}.bn"), context)?;
    relu_tensor(backend, &hidden, context)
}

fn simple_convs(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let hidden = conv2d(
        resource,
        backend,
        input,
        &format!("{prefix}.conv1"),
        1,
        1,
        context,
    )?;
    conv2d(
        resource,
        backend,
        &hidden,
        &format!("{prefix}.conv_out"),
        1,
        1,
        context,
    )
}

fn split_image_patches(
    backend: &CpuBackend,
    image: &Tensor,
    target_height: usize,
    target_width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let [batch, channels, height, width] = image.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    };
    if *batch != 1 || *channels != 3 {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    }
    let height = usize_from(*height)?;
    let width = usize_from(*width)?;
    if !height.is_multiple_of(target_height) || !width.is_multiple_of(target_width) {
        return Err(NativeBackgroundRemovalError::InvalidInput(
            "split-patch geometry must divide the preprocessed image".to_owned(),
        ));
    }
    let rows = height / target_height;
    let columns = width / target_width;
    let patch_count = rows
        .checked_mul(columns)
        .ok_or(NativeBackgroundRemovalError::ShapeOverflow)?;
    let mut patches = Vec::new();
    patches
        .try_reserve_exact(patch_count)
        .map_err(|_| NativeBackgroundRemovalError::Allocation)?;
    for column in 0..columns {
        for row in 0..rows {
            context.check()?;
            let patch = narrow_method_exact_native(
                image,
                2,
                i64::try_from(row * target_height)
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                u64::try_from(target_height)
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                context.cancellation,
            )
            .map_err(tensor_operation)?;
            let patch = narrow_method_exact_native(
                &patch,
                3,
                i64::try_from(column * target_width)
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                u64::try_from(target_width)
                    .map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                context.cancellation,
            )
            .map_err(tensor_operation)?;
            patches.push(patch);
        }
    }
    torch_cat_with_context_exact_native(backend, &patches, 1, context).map_err(tensor_operation)
}

#[allow(clippy::too_many_arguments)]
fn decode_birefnet(
    resource: &NativeBackgroundRemovalResource,
    backend: &CpuBackend,
    image: &Tensor,
    x1: &Tensor,
    x2: &Tensor,
    x3: &Tensor,
    x4: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    let mut current = x4.clone();
    let stages = [
        (4_usize, x3, "decoder.lateral_block4.conv"),
        (3_usize, x2, "decoder.lateral_block3.conv"),
        (2_usize, x1, "decoder.lateral_block2.conv"),
    ];
    for (stage, lateral, lateral_prefix) in stages {
        let [_, _, height, width] = current.descriptor().shape() else {
            return Err(NativeBackgroundRemovalError::ShapeOverflow);
        };
        let patches = split_image_patches(
            backend,
            image,
            usize_from(*height)?,
            usize_from(*width)?,
            context,
        )?;
        let patches = resize_tensor(
            backend,
            &patches,
            usize_from(*height)?,
            usize_from(*width)?,
            InterpolateMode::Bilinear,
            Some(true),
            false,
            context,
        )?;
        let patches = simple_convs(
            resource,
            backend,
            &patches,
            &format!("decoder.ipt_blk{}", stage + 1),
            context,
        )?;
        current = concat_channels(backend, &[&current, &patches], context)?;
        current = basic_decoder_block(
            resource,
            backend,
            &current,
            &format!("decoder.decoder_block{stage}"),
            context,
        )?;
        let gdt = conv2d(
            resource,
            backend,
            &current,
            &format!("decoder.gdt_convs_{stage}.0"),
            1,
            1,
            context,
        )?;
        let gdt = batch_norm_eval(
            resource,
            backend,
            &gdt,
            &format!("decoder.gdt_convs_{stage}.1"),
            context,
        )?;
        let gdt = relu_tensor(backend, &gdt, context)?;
        let attention = conv2d(
            resource,
            backend,
            &gdt,
            &format!("decoder.gdt_convs_attn_{stage}.0"),
            1,
            0,
            context,
        )?;
        let attention = sigmoid_tensor(backend, &attention, 1.0, context)?;
        current = multiply_tensors(backend, &current, &attention, context)?;
        let [_, _, target_height, target_width] = lateral.descriptor().shape() else {
            return Err(NativeBackgroundRemovalError::ShapeOverflow);
        };
        current = resize_tensor(
            backend,
            &current,
            usize_from(*target_height)?,
            usize_from(*target_width)?,
            InterpolateMode::Bilinear,
            Some(true),
            false,
            context,
        )?;
        let lateral = conv2d(resource, backend, lateral, lateral_prefix, 1, 0, context)?;
        current = add_tensors(backend, &current, &lateral, context)?;
    }
    let [_, _, height, width] = current.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    };
    let patches = split_image_patches(
        backend,
        image,
        usize_from(*height)?,
        usize_from(*width)?,
        context,
    )?;
    let patches = resize_tensor(
        backend,
        &patches,
        usize_from(*height)?,
        usize_from(*width)?,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    let patches = simple_convs(resource, backend, &patches, "decoder.ipt_blk2", context)?;
    current = concat_channels(backend, &[&current, &patches], context)?;
    current = basic_decoder_block(
        resource,
        backend,
        &current,
        "decoder.decoder_block1",
        context,
    )?;
    let [_, _, image_height, image_width] = image.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::ShapeOverflow);
    };
    current = resize_tensor(
        backend,
        &current,
        usize_from(*image_height)?,
        usize_from(*image_width)?,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    let patches = split_image_patches(
        backend,
        image,
        usize_from(*image_height)?,
        usize_from(*image_width)?,
        context,
    )?;
    let patches = resize_tensor(
        backend,
        &patches,
        usize_from(*image_height)?,
        usize_from(*image_width)?,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    let patches = simple_convs(resource, backend, &patches, "decoder.ipt_blk1", context)?;
    current = concat_channels(backend, &[&current, &patches], context)?;
    conv2d(
        resource,
        backend,
        &current,
        "decoder.conv_out1.0",
        1,
        0,
        context,
    )
}

fn project_masks(
    backend: &CpuBackend,
    logits: &[Tensor],
    height: u64,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeBackgroundRemovalError> {
    context.check()?;
    let first = logits.first().ok_or_else(|| {
        NativeBackgroundRemovalError::InvalidInput("empty serial logit batch".to_owned())
    })?;
    let [one, channel, _, _] = first.descriptor().shape() else {
        return Err(NativeBackgroundRemovalError::InvalidInput(
            "BiRefNet logits must have rank four".to_owned(),
        ));
    };
    if *one != 1 || *channel != 1 {
        return Err(NativeBackgroundRemovalError::InvalidInput(
            "BiRefNet logits must have one batch and one channel per serial invocation".to_owned(),
        ));
    }
    for logit in logits {
        context.check()?;
        if logit.descriptor().shape() != first.descriptor().shape() {
            return Err(NativeBackgroundRemovalError::InvalidInput(
                "serial BiRefNet logits changed shape".to_owned(),
            ));
        }
    }
    let batched = torch_cat_with_context_exact_native(backend, logits, 0, context)
        .map_err(tensor_operation)?;
    let projected = interpolate_tensor_with_context_exact_native(
        backend,
        &batched,
        &InterpolateConfiguration {
            output_size: Some(vec![
                usize::try_from(height).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
                usize::try_from(width).map_err(|_| NativeBackgroundRemovalError::ShapeOverflow)?,
            ]),
            scale_factor: None,
            mode: InterpolateMode::Bicubic,
            align_corners: None,
            recompute_scale_factor: None,
            antialias: false,
        },
        context,
    )
    .map_err(tensor_operation)?;
    let masks = sigmoid_with_context_exact_native(backend, &projected, context)
        .map_err(tensor_operation)?;
    tensor_squeeze_exact_native(&masks, Some(&[1]), context.cancellation).map_err(tensor_operation)
}

trait IntoBackgroundRemovalError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError;
}

fn tensor_operation(error: impl IntoBackgroundRemovalError) -> NativeBackgroundRemovalError {
    error.into_background_removal_error()
}

impl IntoBackgroundRemovalError for OperatorIndirectionError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::Operator(error),
        }
    }
}

impl IntoBackgroundRemovalError for FunctionalError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::Functional(error),
        }
    }
}

impl IntoBackgroundRemovalError for NeuralNetworkFunctionalError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            Self::Operator(error) => error.into_background_removal_error(),
            error => NativeBackgroundRemovalError::NeuralFunctional(error),
        }
    }
}

impl IntoBackgroundRemovalError for NeuralNetworkModuleError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            Self::Operator(error) => error.into_background_removal_error(),
            Self::Functional(error) => error.into_background_removal_error(),
            error => NativeBackgroundRemovalError::NeuralModule(error),
        }
    }
}

impl IntoBackgroundRemovalError for SpatialFunctionalKernelError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::Spatial(error),
        }
    }
}

impl IntoBackgroundRemovalError for ExternalTensorKernelPartTwoError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            Self::Convolution(error) => error.into_background_removal_error(),
            error => NativeBackgroundRemovalError::External(error),
        }
    }
}

impl IntoBackgroundRemovalError for AttentionError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::Attention(error),
        }
    }
}

impl IntoBackgroundRemovalError for ElementwiseRuntimePartThreeError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::ElementwiseThree(error),
        }
    }
}

impl IntoBackgroundRemovalError for ElementwiseRuntimePartSixError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::ElementwiseSix(error),
        }
    }
}

impl IntoBackgroundRemovalError for ElementwiseRuntimePartSixteenError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::ElementwiseSixteen(error),
        }
    }
}

impl IntoBackgroundRemovalError for ElementwiseRuntimePartSeventeenError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::ElementwiseSeventeen(error),
        }
    }
}

impl IntoBackgroundRemovalError for ElementwiseRuntimePartFiveError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::ElementwiseFive(error),
        }
    }
}

impl IntoBackgroundRemovalError for IndexingMaskingPartOneError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::Indexing(error),
        }
    }
}

impl IntoBackgroundRemovalError for ShapeLayoutTransformPartTwoError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::ShapeLayoutTwo(error),
        }
    }
}

impl IntoBackgroundRemovalError for ShapeLayoutTransformPartThreeError {
    fn into_background_removal_error(self) -> NativeBackgroundRemovalError {
        match self {
            Self::Cancelled => NativeBackgroundRemovalError::Cancelled,
            Self::Tensor(error) => NativeBackgroundRemovalError::Tensor(error),
            error => NativeBackgroundRemovalError::ShapeLayoutThree(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::CpuWorkspaceAuthority;
    use sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native;

    #[test]
    fn reduced_birefnet_baseline_bits() -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 512 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let input = [
            0.0,
            0.1,
            0.2,
            0.3,
            0.25,
            0.5,
            0.75,
            0.0,
            1.0,
            0.0,
            0.5,
            1.0,
            0.5019608,
            0.5,
            0.49803922,
            0.75,
            0.003921569,
            0.99607843,
            0.2,
            0.25,
            0.1,
            0.2,
            0.3,
            0.4,
            0.4,
            0.3,
            0.2,
            0.1,
            0.8,
            0.6,
            0.4,
            0.2,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
            0.7,
            0.5,
            0.3,
            0.05,
            0.15,
            0.25,
            0.35,
            0.35,
            0.45,
            0.55,
            0.65,
            0.65,
            0.55,
            0.45,
            0.35,
            0.95,
            0.85,
            0.75,
            0.65,
            0.125,
            0.375,
            0.625,
            0.875,
        ];
        let image = ImageTensor::from_f32(&backend, &context, 1, 3, 5, 4, &input)?;
        let resource = NativeBackgroundRemovalResource::deterministic_reduced_test_fixture(
            &backend,
            &context,
            BackgroundRemovalFixtureMutation::None,
        )?;
        let output = resource.encode_image(&backend, &image, &context)?;
        let bits = tensor_to_f32_with_context_exact_native(&backend, &output, &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            bits,
            vec![
                1_056_965_453,
                1_056_964_747,
                1_056_963_438,
                1_056_962_004,
                1_056_960_561,
                1_056_968_878,
                1_056_968_192,
                1_056_967_504,
                1_056_966_818,
                1_056_966_114,
                1_056_972_310,
                1_056_971_649,
                1_056_970_985,
                1_056_970_323,
                1_056_969_648,
            ]
        );
        assert_eq!(output.descriptor().shape(), &[1, 3, 5]);
        Ok(())
    }

    #[test]
    fn preprocessing_uses_ties_even_division_and_discards_alpha()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let rgba = ImageTensor::from_f32(
            &backend,
            &context,
            1,
            1,
            1,
            4,
            &[0.5 / 255.0, 1.5 / 255.0, 3.0 / 255.0, 0.25],
        )?;
        let mut alpha_changed = rgba.as_f32_slice()?.to_vec();
        alpha_changed[3] = f32::NAN;
        let alpha_changed = ImageTensor::from_f32(&backend, &context, 1, 1, 1, 4, &alpha_changed)?;
        let processed = preprocess_image(&backend, &rgba, 1, &context)?;
        let processed_alpha_changed = preprocess_image(&backend, &alpha_changed, 1, &context)?;
        let bits = tensor_to_f32_with_context_exact_native(&backend, &processed, &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(bits, vec![0, 1_006_665_857, 1_010_876_609]);
        assert_eq!(
            bits,
            tensor_to_f32_with_context_exact_native(&backend, &processed_alpha_changed, &context,)?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_ne!(
            bits[2],
            (3.0_f32 * (1.0_f32 / 255.0_f32)).to_bits(),
            "source division was replaced by reciprocal multiplication"
        );
        let mut rgb_nonfinite = rgba.as_f32_slice()?.to_vec();
        rgb_nonfinite[0] = f32::INFINITY;
        let rgb_nonfinite = ImageTensor::from_f32(&backend, &context, 1, 1, 1, 4, &rgb_nonfinite)?;
        assert!(matches!(
            preprocess_image(&backend, &rgb_nonfinite, 1, &context),
            Err(NativeBackgroundRemovalError::InvalidInput(_))
        ));
        Ok(())
    }

    #[test]
    fn mask_projection_resizes_logits_before_sigmoid() -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let logits = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 1, 2, 2],
            &[-4.0, 0.0, 1.0, 3.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let source_order = project_masks(&backend, std::slice::from_ref(&logits), 3, 3, &context)?;
        let sigmoid_first = sigmoid_with_context_exact_native(&backend, &logits, &context)?;
        let resized_after_sigmoid = interpolate_tensor_with_context_exact_native(
            &backend,
            &sigmoid_first,
            &InterpolateConfiguration {
                output_size: Some(vec![3, 3]),
                scale_factor: None,
                mode: InterpolateMode::Bicubic,
                align_corners: None,
                recompute_scale_factor: None,
                antialias: false,
            },
            &context,
        )?;
        let resized_after_sigmoid =
            tensor_squeeze_exact_native(&resized_after_sigmoid, Some(&[1]), &cancellation)?;
        assert_ne!(
            tensor_to_f32_with_context_exact_native(&backend, &source_order, &context)?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            tensor_to_f32_with_context_exact_native(&backend, &resized_after_sigmoid, &context,)?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "sigmoid-before-resize must remain distinguishable from source projection order"
        );
        Ok(())
    }

    #[test]
    fn reduced_admission_rejects_state_placement_budget_and_cancellation_atomically()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 512 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let checkpoint = || {
            deterministic_reduced_test_checkpoint(
                &backend,
                &context,
                BackgroundRemovalFixtureMutation::None,
            )
        };

        let mut unsupported = checkpoint()?;
        unsupported
            .ordered_state
            .retain(|(key, _)| key != BIREFNET_MARKER);
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(unsupported, &context),
            Err(NativeBackgroundRemovalError::UnsupportedArchitecture)
        ));

        let mut missing = checkpoint()?;
        missing.ordered_state.remove(0);
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(missing, &context),
            Err(NativeBackgroundRemovalError::MissingState(_))
        ));

        let mut duplicate = checkpoint()?;
        let duplicate_entry = duplicate
            .ordered_state
            .first()
            .ok_or("reduced checkpoint is empty")?
            .clone();
        duplicate.ordered_state.push(duplicate_entry);
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(duplicate, &context),
            Err(NativeBackgroundRemovalError::DuplicateStateKey(_))
        ));

        let current_bytes_before_unexpected = backend.memory_snapshot().current_bytes;
        let mut unexpected = checkpoint()?;
        let unexpected_tensor = unexpected
            .ordered_state
            .first()
            .ok_or("reduced checkpoint is empty")?
            .1
            .clone();
        unexpected
            .ordered_state
            .push(("unexpected.weight".to_owned(), unexpected_tensor));
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(unexpected, &context),
            Err(NativeBackgroundRemovalError::UnexpectedState(_))
        ));
        assert_eq!(
            backend.memory_snapshot().current_bytes,
            current_bytes_before_unexpected
        );

        let current_bytes_before_nonfinite = backend.memory_snapshot().current_bytes;
        let mut nonfinite = checkpoint()?;
        let (_, learned_weight) = nonfinite
            .ordered_state
            .iter_mut()
            .find(|(key, _)| key == "bb.patch_embed.proj.weight")
            .ok_or("patch-embedding weight is missing")?;
        let learned_shape = learned_weight.descriptor().shape().to_vec();
        let learned_count = learned_shape.iter().try_fold(1_usize, |count, dimension| {
            count
                .checked_mul(usize_from(*dimension)?)
                .ok_or(NativeBackgroundRemovalError::ShapeOverflow)
        })?;
        let mut nonfinite_values = vec![0.0; learned_count];
        nonfinite_values[0] = f32::NAN;
        *learned_weight = tensor_from_f32_with_context_exact_native(
            &backend,
            &learned_shape,
            &nonfinite_values,
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(nonfinite, &context),
            Err(NativeBackgroundRemovalError::InvalidCheckpoint(_))
        ));
        assert_eq!(
            backend.memory_snapshot().current_bytes,
            current_bytes_before_nonfinite
        );

        let mut shape_invalid = checkpoint()?;
        let (_, marker) = shape_invalid
            .ordered_state
            .iter_mut()
            .find(|(key, _)| key == BIREFNET_MARKER)
            .ok_or("relative-position marker is missing")?;
        *marker = upload_i64_tensor(&backend, &[1], &[0], &context)?;
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(shape_invalid, &context),
            Err(NativeBackgroundRemovalError::StateShape { .. })
        ));

        let mut index_invalid = checkpoint()?;
        let (_, marker) = index_invalid
            .ordered_state
            .iter_mut()
            .find(|(key, _)| key == BIREFNET_MARKER)
            .ok_or("relative-position marker is missing")?;
        let mut indices = relative_position_indices(2)?;
        indices[0] = 9;
        *marker = upload_i64_tensor(&backend, &[4, 4], &indices, &context)?;
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(index_invalid, &context),
            Err(NativeBackgroundRemovalError::RelativePositionIndex { .. })
        ));

        let foreign_cancellation = CancellationToken::default();
        let foreign_context = backend.execution_context(
            StreamId::new(7),
            authority.authorize_workspace(workspace_bytes)?,
            &foreign_cancellation,
        );
        let foreign = deterministic_reduced_test_checkpoint(
            &backend,
            &foreign_context,
            BackgroundRemovalFixtureMutation::None,
        )?;
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(foreign, &context),
            Err(NativeBackgroundRemovalError::InvalidCheckpoint(_))
        ));

        let mut admission_oom = checkpoint()?;
        admission_oom.memory_budget_bytes = 1;
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(admission_oom, &context),
            Err(NativeBackgroundRemovalError::OutOfMemory { .. })
        ));

        let admitted =
            NativeBackgroundRemovalResource::from_reduced_fixture(checkpoint()?, &context)?;
        let mut invocation_oom = checkpoint()?;
        invocation_oom.memory_budget_bytes = admitted.resident_bytes();
        drop(admitted);
        let invocation_oom =
            NativeBackgroundRemovalResource::from_reduced_fixture(invocation_oom, &context)?;
        let image = ImageTensor::from_f32(&backend, &context, 1, 1, 1, 3, &[0.0; 3])?;
        assert!(matches!(
            invocation_oom.encode_image(&backend, &image, &context),
            Err(NativeBackgroundRemovalError::OutOfMemory { .. })
        ));
        let invalid_channels = ImageTensor::from_f32(&backend, &context, 1, 1, 1, 1, &[0.0])?;
        assert!(matches!(
            invocation_oom.encode_image(&backend, &invalid_channels, &context),
            Err(NativeBackgroundRemovalError::InvalidInput(_))
        ));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancelled,
        );
        assert!(matches!(
            NativeBackgroundRemovalResource::from_reduced_fixture(
                checkpoint()?,
                &cancelled_context,
            ),
            Err(NativeBackgroundRemovalError::Cancelled
                | NativeBackgroundRemovalError::Tensor(TensorError::Cancelled))
        ));

        let distinct_checkpoint = checkpoint()?;
        let distinct_resource = NativeBackgroundRemovalResource::from_reduced_fixture(
            distinct_checkpoint.clone(),
            &context,
        )?;
        let mut aliased_checkpoint = distinct_checkpoint;
        let mut aliased_pair = None;
        'outer: for first in 0..aliased_checkpoint.ordered_state.len() {
            for second in first + 1..aliased_checkpoint.ordered_state.len() {
                let first_tensor = &aliased_checkpoint.ordered_state[first].1;
                let second_tensor = &aliased_checkpoint.ordered_state[second].1;
                if first_tensor.descriptor().dtype() == DType::F32
                    && first_tensor.descriptor().shape() == second_tensor.descriptor().shape()
                    && first_tensor.contiguous_bytes()? != second_tensor.contiguous_bytes()?
                {
                    aliased_pair = Some((first, second));
                    break 'outer;
                }
            }
        }
        let (first, second) = aliased_pair.ok_or("no distinct same-shaped F32 state pair")?;
        let shared_tensor = aliased_checkpoint.ordered_state[first].1.clone();
        let removed_storage_bytes = aliased_checkpoint.ordered_state[second]
            .1
            .storage_byte_len();
        let shared_storage_id = shared_tensor.storage_id();
        aliased_checkpoint.ordered_state[second].1 = shared_tensor;
        let aliased_resource =
            NativeBackgroundRemovalResource::from_reduced_fixture(aliased_checkpoint, &context)?;
        let allocations = aliased_resource.resident_tensor_allocations()?;
        assert_eq!(
            allocations
                .iter()
                .filter(|(storage_id, _)| *storage_id == shared_storage_id)
                .count(),
            1
        );
        assert_eq!(
            distinct_resource
                .resident_bytes()
                .checked_sub(aliased_resource.resident_bytes())
                .ok_or("aliased residency exceeded distinct residency")?,
            removed_storage_bytes
        );
        assert_ne!(
            distinct_resource.semantic_digest_sha256(),
            aliased_resource.semantic_digest_sha256()
        );
        Ok(())
    }
}
