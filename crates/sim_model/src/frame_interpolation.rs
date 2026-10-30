use crate::native_ops::{
    NativeModule, disable_weight_init_convolution_exact_native, tensor_from_f32,
};
use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, Layout,
    MemoryFormatReference, Scalar, StorageId, StreamId, Tensor, TensorError,
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, real_add_with_context_exact_native,
        real_lerp_tensor_weight_with_context_exact_native, real_multiply_with_context_exact_native,
        sigmoid_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_09::clamp_with_context_exact_native,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_neural_network_functional_01::pixel_shuffle_tensor_with_context_exact_native,
    generated_shape_layout_transform_01::{
        tensor_expand_exact_native, torch_unsqueeze_exact_native,
    },
    generated_shape_layout_transform_02::torch_cat_with_context_exact_native,
    generated_shape_layout_transform_03::{
        FunctionalPadMode, functional_pad_with_context_exact_native, tensor_permute_exact_native,
    },
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionPaddingMode, cast_to_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        AveragePoolConfiguration, GridPaddingMode, GridSampleConfiguration, GridSampleMode,
        InterpolateConfiguration, InterpolateMode,
        average_pool_2d_tensor_with_context_exact_native,
        grid_sample_tensor_with_context_exact_native, interpolate_tensor_with_context_exact_native,
    },
    generated_storage_dtype_device_01::{
        clone_with_context_exact_native, contiguous_with_context_exact_native,
    },
};
use sim_types::CancellationError;
use std::collections::{BTreeMap, HashMap};
use std::error::Error as StdError;
use thiserror::Error;

const FILM_MARKER: &str = "extract.extract_sublevels.convs.0.0.conv.weight";
const FRAME_INTERPOLATION_SOURCE_SHA256: &str =
    "038762ff4e248c91e168685796f590a2e5aa0dc3b3c2922aa5f9d936b1fff369";
const FILM_SOURCE_SHA256: &str = "e4efa6666846cecb5dc83cb4668410b37b6c4ffae6b08e48b74184bc037c4ab1";
const RIFE_SOURCE_SHA256: &str = "854b808a425d01a82df2395cb925d7a5dab86669c62485f95fb790736ced11a3";
const MAXIMUM_INTERPOLATION_MULTIPLIER: u64 = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameInterpolationProfile {
    Film,
    Rife {
        head_channels: u64,
        block_channels: [u64; 5],
    },
}

impl FrameInterpolationProfile {
    pub const fn alignment(&self) -> u64 {
        match self {
            Self::Film => 1,
            Self::Rife { .. } => 64,
        }
    }

    pub const fn identifier(&self) -> &'static str {
        match self {
            Self::Film => "film",
            Self::Rife { .. } => "rife",
        }
    }
}

#[derive(Debug, Error)]
pub enum FrameInterpolationError {
    #[error("frame interpolation checkpoint format is unrecognized")]
    Unrecognized,
    #[error("frame interpolation checkpoint contains colliding normalized key `{0}`")]
    KeyCollision(String),
    #[error(
        "frame interpolation checkpoint tensor `{key}` has shape {actual:?}, expected {expected:?}"
    )]
    Shape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("frame interpolation checkpoint tensor set is incomplete or contains unexpected state")]
    StateMismatch,
    #[error("frame interpolation checkpoint tensors do not share supported CPU placement")]
    Placement,
    #[error("frame interpolation artifact digest is invalid")]
    ArtifactDigest,
    #[error("frame interpolation accounting overflow")]
    Overflow,
    #[error("frame interpolation invocation is invalid: {0}")]
    InvalidInvocation(&'static str),
    #[error("frame interpolation tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("frame interpolation operation was cancelled")]
    Cancelled,
    #[error("frame interpolation execution exhausted a bounded resource: {0}")]
    ResourceExhausted(String),
    #[error("frame interpolation execution failed: {0}")]
    Execution(String),
    #[cfg(any(test, feature = "test-support"))]
    #[error("frame interpolation test fixture failed: {0}")]
    TestFixture(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameInterpolationInvocationPlan {
    frame_count: u64,
    multiplier: u64,
    output_frame_count: u64,
    output_element_count: u64,
    total_interpolation_steps: u64,
    padded_height: u64,
    padded_width: u64,
    padding_bottom: u64,
    padding_right: u64,
    timesteps: Vec<f32>,
}

impl FrameInterpolationInvocationPlan {
    pub fn checked(
        profile: &FrameInterpolationProfile,
        frame_count: u64,
        multiplier: u64,
        height: u64,
        width: u64,
        cancellation: &CancellationToken,
    ) -> Result<Self, FrameInterpolationError> {
        cancellation.check()?;
        if height == 0 || width == 0 {
            return Err(FrameInterpolationError::InvalidInvocation(
                "frame extent is zero",
            ));
        }
        if multiplier > MAXIMUM_INTERPOLATION_MULTIPLIER {
            return Err(FrameInterpolationError::InvalidInvocation(
                "multiplier exceeds the source schema maximum",
            ));
        }
        let bypass = frame_count < 2 || multiplier < 2;
        let interpolation_count = if bypass { 0 } else { multiplier - 1 };
        let pair_count = frame_count.saturating_sub(1);
        let total_interpolation_steps = pair_count
            .checked_mul(interpolation_count)
            .ok_or(FrameInterpolationError::Overflow)?;
        let output_frame_count = if bypass {
            frame_count
        } else {
            pair_count
                .checked_mul(multiplier)
                .and_then(|count| count.checked_add(1))
                .ok_or(FrameInterpolationError::Overflow)?
        };
        let output_element_count = output_frame_count
            .checked_mul(3)
            .and_then(|count| count.checked_mul(height))
            .and_then(|count| count.checked_mul(width))
            .ok_or(FrameInterpolationError::Overflow)?;
        let alignment = profile.alignment();
        let padding_bottom = (alignment - height % alignment) % alignment;
        let padding_right = (alignment - width % alignment) % alignment;
        if alignment > 1 && (padding_bottom >= height || padding_right >= width) {
            return Err(FrameInterpolationError::InvalidInvocation(
                "reflection padding is not representable",
            ));
        }
        let padded_height = height
            .checked_add(padding_bottom)
            .ok_or(FrameInterpolationError::Overflow)?;
        let padded_width = width
            .checked_add(padding_right)
            .ok_or(FrameInterpolationError::Overflow)?;
        let timestep_capacity =
            usize::try_from(interpolation_count).map_err(|_| FrameInterpolationError::Overflow)?;
        let mut timesteps = Vec::new();
        timesteps
            .try_reserve_exact(timestep_capacity)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        for index in 1..=interpolation_count {
            cancellation.check()?;
            let index = u16::try_from(index).map_err(|_| FrameInterpolationError::Overflow)?;
            let multiplier =
                u16::try_from(multiplier).map_err(|_| FrameInterpolationError::Overflow)?;
            timesteps.push(f32::from(index) / f32::from(multiplier));
        }
        Ok(Self {
            frame_count,
            multiplier,
            output_frame_count,
            output_element_count,
            total_interpolation_steps,
            padded_height,
            padded_width,
            padding_bottom,
            padding_right,
            timesteps,
        })
    }

    pub const fn is_bypass(&self) -> bool {
        self.frame_count < 2 || self.multiplier < 2
    }
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }
    pub const fn multiplier(&self) -> u64 {
        self.multiplier
    }
    pub const fn output_frame_count(&self) -> u64 {
        self.output_frame_count
    }
    pub const fn output_element_count(&self) -> u64 {
        self.output_element_count
    }
    pub const fn total_interpolation_steps(&self) -> u64 {
        self.total_interpolation_steps
    }
    pub const fn padded_height(&self) -> u64 {
        self.padded_height
    }
    pub const fn padded_width(&self) -> u64 {
        self.padded_width
    }
    pub const fn padding_bottom(&self) -> u64 {
        self.padding_bottom
    }
    pub const fn padding_right(&self) -> u64 {
        self.padding_right
    }
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameInterpolationFallbackState {
    multi_timestep_enabled: bool,
    single_timestep_batch: u64,
}

impl FrameInterpolationFallbackState {
    pub fn for_plan(
        plan: &FrameInterpolationInvocationPlan,
        multi_timestep_available: bool,
    ) -> Result<Self, FrameInterpolationError> {
        if plan.is_bypass() {
            return Ok(Self {
                multi_timestep_enabled: false,
                single_timestep_batch: 0,
            });
        }
        Ok(Self {
            multi_timestep_enabled: multi_timestep_available,
            single_timestep_batch: plan.multiplier - 1,
        })
    }

    pub const fn multi_timestep_enabled(&self) -> bool {
        self.multi_timestep_enabled
    }
    pub const fn single_timestep_batch(&self) -> u64 {
        self.single_timestep_batch
    }
    pub fn record_multi_timestep_oom(&mut self) {
        self.multi_timestep_enabled = false;
    }
    pub fn record_single_timestep_oom(&mut self) -> Result<(), FrameInterpolationError> {
        if self.single_timestep_batch <= 1 {
            return Err(FrameInterpolationError::InvalidInvocation(
                "single-timestep batch one exhausted memory",
            ));
        }
        self.single_timestep_batch = (self.single_timestep_batch / 2).max(1);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FrameInterpolationSequenceAttempt<'a> {
    MultiTimestep(&'a [f32]),
    SingleTimestepBatch(&'a [f32]),
}

fn execute_frame_interpolation_sequence_fallback<T>(
    profile: &FrameInterpolationProfile,
    timesteps: &[f32],
    fallback: &mut FrameInterpolationFallbackState,
    cancellation: &CancellationToken,
    mut execute: impl FnMut(FrameInterpolationSequenceAttempt<'_>) -> Result<T, FrameInterpolationError>,
) -> Result<Vec<T>, FrameInterpolationError> {
    cancellation.check()?;
    if timesteps.is_empty() || fallback.single_timestep_batch() == 0 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "frame interpolation fallback requires at least one timestep",
        ));
    }

    if fallback.multi_timestep_enabled() {
        match execute(FrameInterpolationSequenceAttempt::MultiTimestep(timesteps)) {
            Ok(output) => {
                cancellation.check()?;
                return Ok(vec![output]);
            }
            Err(FrameInterpolationError::ResourceExhausted(_)) => {
                cancellation.check()?;
                fallback.record_multi_timestep_oom();
            }
            Err(error) => return Err(error),
        }
    }

    if matches!(profile, FrameInterpolationProfile::Film) && timesteps.len() > 1 {
        return Err(FrameInterpolationError::Execution(
            "FILM source fallback cannot scalarize a multi-timestep batch".into(),
        ));
    }

    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(timesteps.len())
        .map_err(|_| FrameInterpolationError::Overflow)?;
    let mut offset = 0_usize;
    while offset < timesteps.len() {
        cancellation.check()?;
        let remaining = timesteps
            .len()
            .checked_sub(offset)
            .ok_or(FrameInterpolationError::Overflow)?;
        let batch = usize::try_from(fallback.single_timestep_batch())
            .map_err(|_| FrameInterpolationError::Overflow)?
            .min(remaining);
        let end = offset
            .checked_add(batch)
            .ok_or(FrameInterpolationError::Overflow)?;
        let batch_timesteps = timesteps
            .get(offset..end)
            .ok_or(FrameInterpolationError::StateMismatch)?;
        match execute(FrameInterpolationSequenceAttempt::SingleTimestepBatch(
            batch_timesteps,
        )) {
            Ok(output) => {
                cancellation.check()?;
                outputs.push(output);
                offset = end;
            }
            Err(error @ FrameInterpolationError::ResourceExhausted(_)) => {
                cancellation.check()?;
                if fallback.single_timestep_batch() == 1 {
                    return Err(error);
                }
                fallback.record_single_timestep_oom()?;
            }
            Err(error) => return Err(error),
        }
    }
    cancellation.check()?;
    Ok(outputs)
}

impl From<CancellationError> for FrameInterpolationError {
    fn from(_: CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone)]
pub struct NativeFrameInterpolationModel {
    profile: FrameInterpolationProfile,
    artifact_sha256: String,
    weights: BTreeMap<String, Tensor>,
    dtype: DType,
    stream: StreamId,
    semantic_state_digest_sha256: String,
}

impl NativeFrameInterpolationModel {
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn reduced_rife_test_fixture(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, FrameInterpolationError> {
        let manifest = rife_manifest(2, [2; 5])?;
        let mut weights = BTreeMap::new();
        for (key, shape) in manifest {
            context.cancellation.check()?;
            let count = shape.iter().try_fold(1_usize, |total, dimension| {
                total.checked_mul(usize::try_from(*dimension).ok()?)
            });
            let count = count.ok_or(FrameInterpolationError::Overflow)?;
            let value = if key.ends_with(".beta") { 1.0 } else { 0.0 };
            let tensor = tensor_from_f32(
                backend,
                &shape,
                &vec![value; count],
                DType::F32,
                DeviceId::CPU,
                context,
            )
            .map_err(|error| FrameInterpolationError::TestFixture(error.to_string()))?;
            weights.insert(key, tensor);
        }
        Self::from_checkpoint("f".repeat(64), weights, context.cancellation)
    }

    pub fn from_checkpoint(
        artifact_sha256: impl Into<String>,
        weights: BTreeMap<String, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<Self, FrameInterpolationError> {
        cancellation.check()?;
        let artifact_sha256 = artifact_sha256.into();
        if !valid_sha256(&artifact_sha256) {
            return Err(FrameInterpolationError::ArtifactDigest);
        }
        let (profile, weights) = normalize_and_detect(weights, cancellation)?;
        let manifest = weight_manifest(&profile)?;
        if weights.len() != manifest.len() {
            return Err(FrameInterpolationError::StateMismatch);
        }
        let first = weights
            .values()
            .next()
            .ok_or(FrameInterpolationError::StateMismatch)?;
        let dtype = first.descriptor().dtype();
        let stream = first.descriptor().stream();
        if !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Err(FrameInterpolationError::Placement);
        }
        for (key, expected) in &manifest {
            cancellation.check()?;
            let tensor = weights
                .get(key)
                .ok_or(FrameInterpolationError::StateMismatch)?;
            if tensor.descriptor().shape() != expected {
                return Err(FrameInterpolationError::Shape {
                    key: key.clone(),
                    expected: expected.clone(),
                    actual: tensor.descriptor().shape().to_vec(),
                });
            }
            if tensor.descriptor().dtype() != dtype
                || tensor.descriptor().device() != DeviceId::CPU
                || tensor.descriptor().stream() != stream
                || !tensor.descriptor().is_contiguous()?
            {
                return Err(FrameInterpolationError::Placement);
            }
        }
        let semantic_state_digest_sha256 =
            semantic_digest(&profile, &artifact_sha256, &weights, cancellation)?;
        Ok(Self {
            profile,
            artifact_sha256,
            weights,
            dtype,
            stream,
            semantic_state_digest_sha256,
        })
    }

    pub fn interpolate_sequence(
        &self,
        backend: &CpuBackend,
        images: &Tensor,
        multiplier: u64,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        match self.profile {
            FrameInterpolationProfile::Film => {
                self.interpolate_film_sequence(backend, images, multiplier, context)
            }
            FrameInterpolationProfile::Rife { .. } => {
                self.interpolate_rife_sequence(backend, images, multiplier, context)
            }
        }
    }

    pub fn profile(&self) -> &FrameInterpolationProfile {
        &self.profile
    }
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn stream(&self) -> StreamId {
        self.stream
    }
    pub fn semantic_state_digest_sha256(&self) -> &str {
        &self.semantic_state_digest_sha256
    }
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    pub fn validate(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), FrameInterpolationError> {
        cancellation.check()?;
        if !valid_sha256(&self.artifact_sha256) {
            return Err(FrameInterpolationError::ArtifactDigest);
        }
        if !matches!(self.dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Err(FrameInterpolationError::Placement);
        }
        let manifest = weight_manifest(&self.profile)?;
        if manifest.len() != self.weights.len() {
            return Err(FrameInterpolationError::StateMismatch);
        }
        for (key, expected) in manifest {
            cancellation.check()?;
            let tensor = self
                .weights
                .get(&key)
                .ok_or(FrameInterpolationError::StateMismatch)?;
            if tensor.descriptor().shape() != expected {
                return Err(FrameInterpolationError::Shape {
                    key,
                    expected,
                    actual: tensor.descriptor().shape().to_vec(),
                });
            }
            if tensor.descriptor().dtype() != self.dtype
                || tensor.descriptor().device() != DeviceId::CPU
                || tensor.descriptor().stream() != self.stream
                || !tensor.descriptor().is_contiguous()?
            {
                return Err(FrameInterpolationError::Placement);
            }
        }
        if semantic_digest(
            &self.profile,
            &self.artifact_sha256,
            &self.weights,
            cancellation,
        )? != self.semantic_state_digest_sha256
        {
            return Err(FrameInterpolationError::StateMismatch);
        }
        Ok(())
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(StorageId, u64)>, FrameInterpolationError> {
        let mut allocations = HashMap::new();
        for tensor in self.weights.values() {
            let storage = tensor.storage_id();
            let bytes = tensor.storage_byte_len();
            if let Some(existing) = allocations.insert(storage, bytes)
                && existing != bytes
            {
                return Err(FrameInterpolationError::StateMismatch);
            }
        }
        let mut allocations = allocations.into_iter().collect::<Vec<_>>();
        allocations.sort_unstable_by_key(|(storage, _)| storage.get());
        Ok(allocations)
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, FrameInterpolationError> {
        let keys = self.weights.keys().try_fold(0usize, |total, key| {
            total
                .checked_add(key.capacity())
                .ok_or(FrameInterpolationError::Overflow)
        })?;
        let entries = self
            .weights
            .len()
            .checked_mul(std::mem::size_of::<(String, Tensor)>())
            .ok_or(FrameInterpolationError::Overflow)?;
        let bytes = std::mem::size_of::<Self>()
            .checked_add(keys)
            .and_then(|value| value.checked_add(entries))
            .and_then(|value| value.checked_add(self.artifact_sha256.capacity()))
            .and_then(|value| value.checked_add(self.semantic_state_digest_sha256.capacity()))
            .ok_or(FrameInterpolationError::Overflow)?;
        u64::try_from(bytes).map_err(|_| FrameInterpolationError::Overflow)
    }

    pub fn resident_bytes(&self) -> Result<u64, FrameInterpolationError> {
        self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(FrameInterpolationError::Overflow)
            },
        )
    }

    pub fn film_subtree_features(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        pooling_levels: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, FrameInterpolationError> {
        context.cancellation.check()?;
        if self.profile != FrameInterpolationProfile::Film {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM subtree execution requires a FILM checkpoint",
            ));
        }
        let descriptor = input.descriptor();
        let shape = descriptor.shape();
        if shape.len() != 4
            || shape.first() == Some(&0)
            || shape.get(1) != Some(&3)
            || shape.get(2) == Some(&0)
            || shape.get(3) == Some(&0)
        {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM subtree input must be nonempty NCHW RGB",
            ));
        }
        if descriptor.dtype() != self.dtype
            || descriptor.device() != DeviceId::CPU
            || descriptor.stream() != self.stream
            || descriptor.stream() != context.stream
            || !descriptor.is_contiguous()?
        {
            return Err(FrameInterpolationError::Placement);
        }
        film_subtree_features_from_weights(
            backend,
            input,
            pooling_levels,
            64,
            4,
            &self.weights,
            context,
        )
    }

    pub fn film_feature_pyramid(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, FrameInterpolationError> {
        context.cancellation.check()?;
        if self.profile != FrameInterpolationProfile::Film {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM feature extraction requires a FILM checkpoint",
            ));
        }
        let image_pyramid =
            film_image_pyramid_with_context_exact_native(backend, input, 7, context)?;
        self.film_feature_pyramid_from_images(backend, &image_pyramid, context)
    }

    fn film_feature_pyramid_from_images(
        &self,
        backend: &CpuBackend,
        image_pyramid: &[Tensor],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, FrameInterpolationError> {
        if image_pyramid.len() != 7 {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM feature extraction requires seven image levels",
            ));
        }
        let mut sub_pyramids = Vec::new();
        sub_pyramids
            .try_reserve_exact(image_pyramid.len())
            .map_err(|_| FrameInterpolationError::Overflow)?;
        for (index, image) in image_pyramid.iter().enumerate() {
            context.cancellation.check()?;
            let pooling_levels = image_pyramid.len().saturating_sub(index).min(4);
            sub_pyramids.push(self.film_subtree_features(
                backend,
                image,
                pooling_levels,
                context,
            )?);
        }
        compose_film_feature_pyramid(backend, sub_pyramids, 4, context)
    }

    pub fn film_residual_flow_pyramid(
        &self,
        backend: &CpuBackend,
        first_features: &[Tensor],
        second_features: &[Tensor],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, FrameInterpolationError> {
        context.cancellation.check()?;
        if self.profile != FrameInterpolationProfile::Film {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM flow prediction requires a FILM checkpoint",
            ));
        }
        validate_film_feature_pyramids(
            first_features,
            second_features,
            self.dtype,
            self.stream,
            context,
        )?;
        let deepest = 6_usize;
        let mut flow = film_flow_estimator_from_weights(
            backend,
            "predict_flow._predictor",
            first_features
                .get(deepest)
                .ok_or(FrameInterpolationError::StateMismatch)?,
            second_features
                .get(deepest)
                .ok_or(FrameInterpolationError::StateMismatch)?,
            &self.weights,
            context,
        )?;
        let mut residuals = Vec::new();
        residuals
            .try_reserve_exact(7)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        residuals.push(flow.clone());
        for index in [5_usize, 4, 3] {
            context.cancellation.check()?;
            flow = film_upsample_double_to_residual(
                backend,
                &flow,
                first_features
                    .get(index)
                    .ok_or(FrameInterpolationError::StateMismatch)?,
                context,
            )?;
            let warped = film_warp_with_context_exact_native(
                backend,
                second_features
                    .get(index)
                    .ok_or(FrameInterpolationError::StateMismatch)?,
                &flow,
                context,
            )?;
            let residual = film_flow_estimator_from_weights(
                backend,
                "predict_flow._predictor",
                first_features
                    .get(index)
                    .ok_or(FrameInterpolationError::StateMismatch)?,
                &warped,
                &self.weights,
                context,
            )?;
            flow = execution_result(
                real_add_with_context_exact_native(backend, &flow, &residual, context),
                context,
            )?;
            residuals.push(residual);
        }
        for (index, predictor) in [
            (2_usize, "predict_flow._predictors.0"),
            (1_usize, "predict_flow._predictors.1"),
            (0_usize, "predict_flow._predictors.2"),
        ] {
            context.cancellation.check()?;
            flow = film_upsample_double_to_residual(
                backend,
                &flow,
                first_features
                    .get(index)
                    .ok_or(FrameInterpolationError::StateMismatch)?,
                context,
            )?;
            let warped = film_warp_with_context_exact_native(
                backend,
                second_features
                    .get(index)
                    .ok_or(FrameInterpolationError::StateMismatch)?,
                &flow,
                context,
            )?;
            let residual = film_flow_estimator_from_weights(
                backend,
                predictor,
                first_features
                    .get(index)
                    .ok_or(FrameInterpolationError::StateMismatch)?,
                &warped,
                &self.weights,
                context,
            )?;
            flow = execution_result(
                real_add_with_context_exact_native(backend, &flow, &residual, context),
                context,
            )?;
            residuals.push(residual);
        }
        residuals.reverse();
        context.cancellation.check()?;
        Ok(residuals)
    }

    pub fn film_fuse_pyramid(
        &self,
        backend: &CpuBackend,
        pyramid: &[Tensor],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        context.cancellation.check()?;
        if self.profile != FrameInterpolationProfile::Film {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM fusion requires a FILM checkpoint",
            ));
        }
        for level in pyramid {
            let descriptor = level.descriptor();
            if descriptor.dtype() != self.dtype
                || descriptor.device() != DeviceId::CPU
                || descriptor.stream() != self.stream
                || descriptor.stream() != context.stream
                || !descriptor.is_contiguous()?
            {
                return Err(FrameInterpolationError::Placement);
            }
        }
        film_fusion_from_weights(backend, pyramid, &self.weights, context)
    }

    pub fn interpolate_film_pair_multi_timestep(
        &self,
        backend: &CpuBackend,
        first: &Tensor,
        second: &Tensor,
        timesteps: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        context.cancellation.check()?;
        if self.profile != FrameInterpolationProfile::Film {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM pair execution requires a FILM checkpoint",
            ));
        }
        if timesteps.is_empty()
            || timesteps.len()
                > usize::try_from(MAXIMUM_INTERPOLATION_MULTIPLIER - 1)
                    .map_err(|_| FrameInterpolationError::Overflow)?
            || timesteps
                .iter()
                .any(|timestep| !timestep.is_finite() || !(0.0..=1.0).contains(timestep))
        {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM timesteps must be a nonempty bounded finite unit interval",
            ));
        }
        let first_descriptor = first.descriptor();
        let second_descriptor = second.descriptor();
        let shape = first_descriptor.shape();
        if shape.len() != 4
            || shape.first() != Some(&1)
            || shape.get(1) != Some(&3)
            || shape.get(2) == Some(&0)
            || shape.get(3) == Some(&0)
            || second_descriptor.shape() != shape
        {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM pair inputs must be matching batch-one NCHW RGB tensors",
            ));
        }
        for descriptor in [first_descriptor, second_descriptor] {
            if descriptor.dtype() != self.dtype
                || descriptor.device() != DeviceId::CPU
                || descriptor.stream() != self.stream
                || descriptor.stream() != context.stream
                || !descriptor.is_contiguous()?
            {
                return Err(FrameInterpolationError::Placement);
            }
        }

        let first_images =
            film_image_pyramid_with_context_exact_native(backend, first, 7, context)?;
        let first_features =
            self.film_feature_pyramid_from_images(backend, &first_images, context)?;
        let second_images =
            film_image_pyramid_with_context_exact_native(backend, second, 7, context)?;
        let second_features =
            self.film_feature_pyramid_from_images(backend, &second_images, context)?;
        self.film_pair_multi_timestep_from_pyramids(
            backend,
            &first_images,
            &first_features,
            &second_images,
            &second_features,
            timesteps,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn film_pair_multi_timestep_from_pyramids(
        &self,
        backend: &CpuBackend,
        first_images: &[Tensor],
        first_features: &[Tensor],
        second_images: &[Tensor],
        second_features: &[Tensor],
        timesteps: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        context.cancellation.check()?;
        if first_images.len() != 7
            || first_features.len() != 7
            || second_images.len() != 7
            || second_features.len() != 7
        {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM pair synthesis requires seven endpoint pyramid levels",
            ));
        }
        let forward_residuals =
            self.film_residual_flow_pyramid(backend, first_features, second_features, context)?;
        let backward_residuals =
            self.film_residual_flow_pyramid(backend, second_features, first_features, context)?;
        let forward_flows = film_flow_pyramid_synthesis_with_context_exact_native(
            backend,
            &forward_residuals,
            context,
        )?;
        let backward_flows = film_flow_pyramid_synthesis_with_context_exact_native(
            backend,
            &backward_residuals,
            context,
        )?;
        let first_warp_targets = film_concatenate_pyramids_with_context_exact_native(
            backend,
            first_images
                .get(..5)
                .ok_or(FrameInterpolationError::StateMismatch)?,
            first_features
                .get(..5)
                .ok_or(FrameInterpolationError::StateMismatch)?,
            context,
        )?;
        let second_warp_targets = film_concatenate_pyramids_with_context_exact_native(
            backend,
            second_images
                .get(..5)
                .ok_or(FrameInterpolationError::StateMismatch)?,
            second_features
                .get(..5)
                .ok_or(FrameInterpolationError::StateMismatch)?,
            context,
        )?;
        film_synthesize_timesteps_from_pyramids(
            backend,
            &first_warp_targets,
            &second_warp_targets,
            forward_flows
                .get(..5)
                .ok_or(FrameInterpolationError::StateMismatch)?,
            backward_flows
                .get(..5)
                .ok_or(FrameInterpolationError::StateMismatch)?,
            timesteps,
            &self.weights,
            self.dtype,
            context,
        )
    }

    pub fn interpolate_film_sequence(
        &self,
        backend: &CpuBackend,
        images: &Tensor,
        multiplier: u64,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        context.cancellation.check()?;
        if self.profile != FrameInterpolationProfile::Film {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM sequence execution requires a FILM checkpoint",
            ));
        }
        let descriptor = images.descriptor();
        let shape = descriptor.shape();
        if shape.len() != 4
            || shape.get(1) == Some(&0)
            || shape.get(2) == Some(&0)
            || shape.get(3) != Some(&3)
        {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM sequence input must be BHWC RGB",
            ));
        }
        if descriptor.dtype() != self.dtype
            || descriptor.device() != DeviceId::CPU
            || descriptor.stream() != self.stream
            || descriptor.stream() != context.stream
        {
            return Err(FrameInterpolationError::Placement);
        }
        let frame_count = *shape.first().ok_or(FrameInterpolationError::Overflow)?;
        let height = *shape.get(1).ok_or(FrameInterpolationError::Overflow)?;
        let width = *shape.get(2).ok_or(FrameInterpolationError::Overflow)?;
        let plan = FrameInterpolationInvocationPlan::checked(
            &self.profile,
            frame_count,
            multiplier,
            height,
            width,
            context.cancellation,
        )?;
        if plan.is_bypass() {
            context.cancellation.check()?;
            return Ok(images.clone());
        }
        let mut fallback = FrameInterpolationFallbackState::for_plan(&plan, true)?;

        let pair_count = frame_count.saturating_sub(1);
        let output_tensor_capacity = pair_count
            .checked_mul(2)
            .and_then(|count| count.checked_add(1))
            .and_then(|count| usize::try_from(count).ok())
            .ok_or(FrameInterpolationError::Overflow)?;
        let mut output_tensors = Vec::new();
        output_tensors
            .try_reserve_exact(output_tensor_capacity)
            .map_err(|_| FrameInterpolationError::Overflow)?;

        let first_frame = rife_sequence_frame(backend, images, 0, context)?;
        output_tensors.push(first_frame.clone());
        let mut first_images =
            film_image_pyramid_with_context_exact_native(backend, &first_frame, 7, context)?;
        let mut first_features =
            self.film_feature_pyramid_from_images(backend, &first_images, context)?;
        for pair in 0..pair_count {
            context.cancellation.check()?;
            let second_frame = rife_sequence_frame(backend, images, pair + 1, context)?;
            let second_images =
                film_image_pyramid_with_context_exact_native(backend, &second_frame, 7, context)?;
            let second_features =
                self.film_feature_pyramid_from_images(backend, &second_images, context)?;
            let midpoints = execute_frame_interpolation_sequence_fallback(
                &self.profile,
                plan.timesteps(),
                &mut fallback,
                context.cancellation,
                |attempt| {
                    let timesteps = match attempt {
                        FrameInterpolationSequenceAttempt::MultiTimestep(timesteps)
                        | FrameInterpolationSequenceAttempt::SingleTimestepBatch(timesteps) => {
                            timesteps
                        }
                    };
                    self.film_pair_multi_timestep_from_pyramids(
                        backend,
                        &first_images,
                        &first_features,
                        &second_images,
                        &second_features,
                        timesteps,
                        context,
                    )
                },
            )?;
            output_tensors.extend(midpoints);
            output_tensors.push(second_frame);
            first_images = second_images;
            first_features = second_features;
        }
        if output_tensors.len() != output_tensor_capacity {
            return Err(FrameInterpolationError::StateMismatch);
        }
        film_finalize_sequence_output(backend, &output_tensors, plan.output_frame_count(), context)
    }

    pub fn interpolate_rife_pair(
        &self,
        backend: &CpuBackend,
        first: &Tensor,
        second: &Tensor,
        timestep: f32,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        context.cancellation.check()?;
        let FrameInterpolationProfile::Rife { head_channels, .. } = self.profile else {
            return Err(FrameInterpolationError::InvalidInvocation(
                "RIFE execution requires a RIFE checkpoint",
            ));
        };
        if !timestep.is_finite() || !(0.0..=1.0).contains(&timestep) {
            return Err(FrameInterpolationError::InvalidInvocation(
                "RIFE timestep must be finite and between zero and one",
            ));
        }
        validate_rife_image(first, self.dtype, self.stream, context)?;
        validate_rife_image(second, self.dtype, self.stream, context)?;
        if first.descriptor().shape() != second.descriptor().shape() {
            return Err(FrameInterpolationError::InvalidInvocation(
                "RIFE input frame shapes do not match",
            ));
        }
        let shape = first.descriptor().shape();
        let height = *shape.get(2).ok_or(FrameInterpolationError::Overflow)?;
        let width = *shape.get(3).ok_or(FrameInterpolationError::Overflow)?;
        if !height.is_multiple_of(64) || !width.is_multiple_of(64) {
            return Err(FrameInterpolationError::InvalidInvocation(
                "RIFE pair execution requires dimensions padded to multiples of 64",
            ));
        }
        let base_grid = rife_base_grid(backend, height, width, context)?;
        let first_features = self.rife_head(backend, first, head_channels, context)?;
        let second_features = self.rife_head(backend, second, head_channels, context)?;
        self.interpolate_rife_pair_with_features(
            backend,
            first,
            second,
            &first_features,
            &second_features,
            &base_grid,
            &[timestep],
            height,
            width,
            context,
        )
    }

    pub fn interpolate_rife_sequence(
        &self,
        backend: &CpuBackend,
        images: &Tensor,
        multiplier: u64,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        context.cancellation.check()?;
        let FrameInterpolationProfile::Rife { head_channels, .. } = self.profile else {
            return Err(FrameInterpolationError::InvalidInvocation(
                "RIFE sequence execution requires a RIFE checkpoint",
            ));
        };
        let descriptor = images.descriptor();
        let shape = descriptor.shape();
        if shape.len() != 4
            || shape.get(1) == Some(&0)
            || shape.get(2) == Some(&0)
            || shape.get(3) != Some(&3)
        {
            return Err(FrameInterpolationError::InvalidInvocation(
                "RIFE sequence input must be BHWC RGB",
            ));
        }
        if descriptor.dtype() != self.dtype
            || descriptor.device() != DeviceId::CPU
            || descriptor.stream() != self.stream
            || descriptor.stream() != context.stream
        {
            return Err(FrameInterpolationError::Placement);
        }
        let frame_count = *shape.first().ok_or(FrameInterpolationError::Overflow)?;
        let height = *shape.get(1).ok_or(FrameInterpolationError::Overflow)?;
        let width = *shape.get(2).ok_or(FrameInterpolationError::Overflow)?;
        let plan = FrameInterpolationInvocationPlan::checked(
            &self.profile,
            frame_count,
            multiplier,
            height,
            width,
            context.cancellation,
        )?;
        if plan.is_bypass() {
            context.cancellation.check()?;
            return Ok(images.clone());
        }
        let mut fallback = FrameInterpolationFallbackState::for_plan(&plan, false)?;

        let output_capacity = usize::try_from(plan.output_frame_count())
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let mut output_frames = Vec::new();
        output_frames
            .try_reserve_exact(output_capacity)
            .map_err(|_| FrameInterpolationError::Overflow)?;

        let first_unpadded = rife_sequence_frame(backend, images, 0, context)?;
        output_frames.push(first_unpadded.clone());
        let mut emitted_frames = 1_u64;
        let mut first = pad_rife_sequence_frame(backend, &first_unpadded, &plan, context)?;
        let base_grid =
            rife_base_grid(backend, plan.padded_height(), plan.padded_width(), context)?;
        let mut first_features = self.rife_head(backend, &first, head_channels, context)?;

        for pair in 0..frame_count.saturating_sub(1) {
            context.cancellation.check()?;
            let second_unpadded = rife_sequence_frame(backend, images, pair + 1, context)?;
            let second = pad_rife_sequence_frame(backend, &second_unpadded, &plan, context)?;
            let second_features = self.rife_head(backend, &second, head_channels, context)?;
            let midpoints = execute_frame_interpolation_sequence_fallback(
                &self.profile,
                plan.timesteps(),
                &mut fallback,
                context.cancellation,
                |attempt| {
                    let timesteps = match attempt {
                        FrameInterpolationSequenceAttempt::MultiTimestep(timesteps)
                        | FrameInterpolationSequenceAttempt::SingleTimestepBatch(timesteps) => {
                            timesteps
                        }
                    };
                    self.interpolate_rife_pair_with_features(
                        backend,
                        &first,
                        &second,
                        &first_features,
                        &second_features,
                        &base_grid,
                        timesteps,
                        plan.padded_height(),
                        plan.padded_width(),
                        context,
                    )
                },
            )?;
            for midpoint in midpoints {
                let midpoint_batch = *midpoint
                    .descriptor()
                    .shape()
                    .first()
                    .ok_or(FrameInterpolationError::StateMismatch)?;
                emitted_frames = emitted_frames
                    .checked_add(midpoint_batch)
                    .ok_or(FrameInterpolationError::Overflow)?;
                output_frames.push(crop_rife_sequence_frame(
                    backend, &midpoint, height, width, context,
                )?);
            }
            output_frames.push(second_unpadded);
            emitted_frames = emitted_frames
                .checked_add(1)
                .ok_or(FrameInterpolationError::Overflow)?;
            first = second;
            first_features = second_features;
        }

        if emitted_frames != plan.output_frame_count() {
            return Err(FrameInterpolationError::StateMismatch);
        }
        let output = execution_result(
            torch_cat_with_context_exact_native(backend, &output_frames, 0, context),
            context,
        )?;
        let output = execution_result(
            tensor_permute_exact_native(&output, &[0, 2, 3, 1], context.cancellation),
            context,
        )?;
        let output = execution_result(
            contiguous_with_context_exact_native(
                backend,
                &output,
                MemoryFormatReference::Layout(Layout::Contiguous),
                context,
            ),
            context,
        )?;
        let output = execution_result(
            clamp_with_context_exact_native(
                backend,
                &output,
                Some(Scalar::Float(0.0)),
                Some(Scalar::Float(1.0)),
                context,
            ),
            context,
        )?;
        context.cancellation.check()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn interpolate_rife_pair_with_features(
        &self,
        backend: &CpuBackend,
        first: &Tensor,
        second: &Tensor,
        first_features: &Tensor,
        second_features: &Tensor,
        base_grid: &Tensor,
        timesteps: &[f32],
        height: u64,
        width: u64,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        context.cancellation.check()?;
        let source_batch = *first
            .descriptor()
            .shape()
            .first()
            .ok_or(FrameInterpolationError::Overflow)?;
        let batch = if timesteps.len() == 1 {
            source_batch
        } else {
            if source_batch != 1 {
                return Err(FrameInterpolationError::InvalidInvocation(
                    "RIFE timestep batches require batch-one endpoints",
                ));
            }
            u64::try_from(timesteps.len()).map_err(|_| FrameInterpolationError::Overflow)?
        };
        let first = expand_rife_batch(first, batch, context)?;
        let second = expand_rife_batch(second, batch, context)?;
        let first_features = expand_rife_batch(first_features, batch, context)?;
        let second_features = expand_rife_batch(second_features, batch, context)?;
        let timestep_tensor = rife_timestep_tensor(
            backend, timesteps, batch, height, width, self.dtype, context,
        )?;
        let mut flow: Option<Tensor> = None;
        let mut mask: Option<Tensor> = None;
        let mut features: Option<Tensor> = None;
        let mut warped_first = first.clone();
        let mut warped_second = second.clone();
        for (block, scale) in [16_usize, 8, 4, 2, 1].into_iter().enumerate() {
            context.cancellation.check()?;
            let block_input = if let (Some(flow), Some(mask), Some(features)) =
                (flow.as_ref(), mask.as_ref(), features.as_ref())
            {
                let first_feature_flow = contiguous_narrow(backend, flow, 1, 0, 2, context)?;
                let second_feature_flow = contiguous_narrow(backend, flow, 1, 2, 2, context)?;
                let warped_first_features = warp_rife(
                    backend,
                    &first_features,
                    &first_feature_flow,
                    base_grid,
                    context,
                )?;
                let warped_second_features = warp_rife(
                    backend,
                    &second_features,
                    &second_feature_flow,
                    base_grid,
                    context,
                )?;
                execution_result(
                    torch_cat_with_context_exact_native(
                        backend,
                        &[
                            warped_first.clone(),
                            warped_second.clone(),
                            warped_first_features,
                            warped_second_features,
                            timestep_tensor.clone(),
                            mask.clone(),
                            features.clone(),
                        ],
                        1,
                        context,
                    ),
                    context,
                )?
            } else {
                execution_result(
                    torch_cat_with_context_exact_native(
                        backend,
                        &[
                            first.clone(),
                            second.clone(),
                            first_features.clone(),
                            second_features.clone(),
                            timestep_tensor.clone(),
                        ],
                        1,
                        context,
                    ),
                    context,
                )?
            };
            let (flow_delta, next_mask, next_features) = self.rife_block(
                backend,
                block,
                scale,
                &block_input,
                flow.as_ref(),
                height,
                width,
                context,
            )?;
            flow = Some(match flow {
                Some(ref current) => execution_result(
                    real_add_with_context_exact_native(backend, current, &flow_delta, context),
                    context,
                )?,
                None => flow_delta,
            });
            mask = Some(next_mask);
            features = Some(next_features);
            let flow = flow
                .as_ref()
                .ok_or(FrameInterpolationError::StateMismatch)?;
            let first_flow = contiguous_narrow(backend, flow, 1, 0, 2, context)?;
            let second_flow = contiguous_narrow(backend, flow, 1, 2, 2, context)?;
            warped_first = warp_rife(backend, &first, &first_flow, base_grid, context)?;
            warped_second = warp_rife(backend, &second, &second_flow, base_grid, context)?;
        }
        let mask = mask.ok_or(FrameInterpolationError::StateMismatch)?;
        let weight = execution_result(
            sigmoid_with_context_exact_native(backend, &mask, context),
            context,
        )?;
        let output = execution_result(
            real_lerp_tensor_weight_with_context_exact_native(
                backend,
                &warped_second,
                &warped_first,
                &weight,
                context,
            ),
            context,
        )?;
        context.cancellation.check()?;
        Ok(output)
    }

    fn rife_head(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        head_channels: u64,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        let mut output = self.convolution(
            backend,
            "encode.cnn0",
            input,
            3,
            16,
            3,
            2,
            1,
            false,
            context,
        )?;
        output = leaky_relu(backend, &output, context)?;
        output = self.convolution(
            backend,
            "encode.cnn1",
            &output,
            16,
            16,
            3,
            1,
            1,
            false,
            context,
        )?;
        output = leaky_relu(backend, &output, context)?;
        output = self.convolution(
            backend,
            "encode.cnn2",
            &output,
            16,
            16,
            3,
            1,
            1,
            false,
            context,
        )?;
        output = leaky_relu(backend, &output, context)?;
        self.convolution(
            backend,
            "encode.cnn3",
            &output,
            16,
            usize::try_from(head_channels).map_err(|_| FrameInterpolationError::Overflow)?,
            4,
            2,
            1,
            true,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rife_block(
        &self,
        backend: &CpuBackend,
        block: usize,
        scale: usize,
        input: &Tensor,
        flow: Option<&Tensor>,
        output_height: u64,
        output_width: u64,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, Tensor, Tensor), FrameInterpolationError> {
        let profile_channels = match &self.profile {
            FrameInterpolationProfile::Rife { block_channels, .. } => block_channels,
            FrameInterpolationProfile::Film => return Err(FrameInterpolationError::StateMismatch),
        };
        let channels = usize::try_from(
            *profile_channels
                .get(block)
                .ok_or(FrameInterpolationError::StateMismatch)?,
        )
        .map_err(|_| FrameInterpolationError::Overflow)?;
        let down_height =
            usize::try_from(output_height).map_err(|_| FrameInterpolationError::Overflow)? / scale;
        let down_width =
            usize::try_from(output_width).map_err(|_| FrameInterpolationError::Overflow)? / scale;
        let mut output = interpolate_bilinear(backend, input, down_height, down_width, context)?;
        if let Some(flow) = flow {
            let down_flow = interpolate_bilinear(backend, flow, down_height, down_width, context)?;
            let down_flow = execution_result(
                real_multiply_with_context_exact_native(
                    backend,
                    &down_flow,
                    ElementwiseOperand::Scalar(Scalar::Float(1.0 / scale as f64)),
                    context,
                ),
                context,
            )?;
            output = execution_result(
                torch_cat_with_context_exact_native(backend, &[output, down_flow], 1, context),
                context,
            )?;
        }
        output = self.convolution(
            backend,
            &format!("blocks.{block}.conv0.0.0"),
            &output,
            usize::try_from(output.descriptor().shape()[1])
                .map_err(|_| FrameInterpolationError::Overflow)?,
            channels / 2,
            3,
            2,
            1,
            false,
            context,
        )?;
        output = leaky_relu(backend, &output, context)?;
        output = self.convolution(
            backend,
            &format!("blocks.{block}.conv0.1.0"),
            &output,
            channels / 2,
            channels,
            3,
            2,
            1,
            false,
            context,
        )?;
        output = leaky_relu(backend, &output, context)?;
        for residual in 0..8 {
            context.cancellation.check()?;
            let prefix = format!("blocks.{block}.convblock.{residual}");
            let convolved = self.convolution(
                backend,
                &format!("{prefix}.conv"),
                &output,
                channels,
                channels,
                3,
                1,
                1,
                false,
                context,
            )?;
            let beta = self
                .weights
                .get(&format!("{prefix}.beta"))
                .ok_or(FrameInterpolationError::StateMismatch)?;
            let residual = execution_result(
                real_multiply_with_context_exact_native(
                    backend,
                    &convolved,
                    ElementwiseOperand::Tensor(beta),
                    context,
                ),
                context,
            )?;
            output = execution_result(
                real_add_with_context_exact_native(backend, &output, &residual, context),
                context,
            )?;
            output = leaky_relu(backend, &output, context)?;
        }
        output = self.convolution(
            backend,
            &format!("blocks.{block}.lastconv.0"),
            &output,
            channels,
            52,
            4,
            2,
            1,
            true,
            context,
        )?;
        output = execution_result(
            pixel_shuffle_tensor_with_context_exact_native(backend, &output, 2, context),
            context,
        )?;
        output = interpolate_bilinear(
            backend,
            &output,
            usize::try_from(output_height).map_err(|_| FrameInterpolationError::Overflow)?,
            usize::try_from(output_width).map_err(|_| FrameInterpolationError::Overflow)?,
            context,
        )?;
        let flow = contiguous_narrow(backend, &output, 1, 0, 4, context)?;
        let flow = execution_result(
            real_multiply_with_context_exact_native(
                backend,
                &flow,
                ElementwiseOperand::Scalar(Scalar::Float(scale as f64)),
                context,
            ),
            context,
        )?;
        let mask = contiguous_narrow(backend, &output, 1, 4, 1, context)?;
        let features = contiguous_narrow(backend, &output, 1, 5, 8, context)?;
        Ok((flow, mask, features))
    }

    #[allow(clippy::too_many_arguments)]
    fn convolution(
        &self,
        backend: &CpuBackend,
        prefix: &str,
        input: &Tensor,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        transposed: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, FrameInterpolationError> {
        let geometry = execution_result(
            ConvolutionGeometry::new_with_padding_mode(
                2,
                vec![stride; 2],
                vec![padding; 2],
                vec![1; 2],
                1,
                transposed,
                vec![0; 2],
                ConvolutionPaddingMode::Zeros,
            ),
            context,
        )?;
        let mut module = execution_result(
            disable_weight_init_convolution_exact_native(
                prefix,
                input_channels,
                output_channels,
                vec![kernel; 2],
                true,
                geometry,
            ),
            context,
        )?;
        let weight = self
            .weights
            .get(&format!("{prefix}.weight"))
            .ok_or(FrameInterpolationError::StateMismatch)?
            .clone();
        let bias = self
            .weights
            .get(&format!("{prefix}.bias"))
            .ok_or(FrameInterpolationError::StateMismatch)?
            .clone();
        execution_result(module.load_dense_parameters(weight, Some(bias)), context)?;
        execution_result(
            module.forward_dense_inference_with_context(backend, input, context),
            context,
        )
    }
}

fn execution_result<T, E: StdError + 'static>(
    result: Result<T, E>,
    context: &ExecutionContext<'_>,
) -> Result<T, FrameInterpolationError> {
    result.map_err(|error| {
        if context.cancellation.is_cancelled() {
            FrameInterpolationError::Cancelled
        } else if error_chain_is_resource_exhaustion(&error) {
            FrameInterpolationError::ResourceExhausted(error.to_string())
        } else {
            FrameInterpolationError::Execution(error.to_string())
        }
    })
}

fn error_chain_is_resource_exhaustion(error: &(dyn StdError + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if error
            .downcast_ref::<TensorError>()
            .is_some_and(tensor_error_is_resource_exhaustion)
        {
            return true;
        }
        current = error.source();
    }
    false
}

fn tensor_error_is_resource_exhaustion(error: &TensorError) -> bool {
    matches!(
        error,
        TensorError::AllocationFailed { .. }
            | TensorError::ResourceLimitExceeded { .. }
            | TensorError::WorkspaceAuthorizationExceeded { .. }
    )
}

fn validate_rife_image(
    image: &Tensor,
    dtype: DType,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<(), FrameInterpolationError> {
    context.cancellation.check()?;
    let descriptor = image.descriptor();
    let shape = descriptor.shape();
    if shape.len() != 4
        || shape.first() == Some(&0)
        || shape.get(1) != Some(&3)
        || shape.get(2) == Some(&0)
        || shape.get(3) == Some(&0)
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "RIFE inputs must be nonempty NCHW RGB tensors",
        ));
    }
    if descriptor.dtype() != dtype
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != stream
        || descriptor.stream() != context.stream
        || !descriptor.is_contiguous()?
    {
        return Err(FrameInterpolationError::Placement);
    }
    Ok(())
}

fn expand_rife_batch(
    input: &Tensor,
    batch: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    let shape = input.descriptor().shape();
    let current_batch = *shape.first().ok_or(FrameInterpolationError::Overflow)?;
    if current_batch == batch {
        return Ok(input.clone());
    }
    if current_batch != 1 || batch == 0 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "RIFE batch expansion requires a batch-one tensor",
        ));
    }
    let mut expanded_shape = shape
        .iter()
        .map(|dimension| i64::try_from(*dimension).map_err(|_| FrameInterpolationError::Overflow))
        .collect::<Result<Vec<_>, _>>()?;
    let expanded_batch = i64::try_from(batch).map_err(|_| FrameInterpolationError::Overflow)?;
    let first_dimension = expanded_shape
        .first_mut()
        .ok_or(FrameInterpolationError::Overflow)?;
    *first_dimension = expanded_batch;
    execution_result(
        tensor_expand_exact_native(input, &expanded_shape, context.cancellation),
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn rife_timestep_tensor(
    backend: &CpuBackend,
    timesteps: &[f32],
    batch: u64,
    height: u64,
    width: u64,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    let timestep_count =
        u64::try_from(timesteps.len()).map_err(|_| FrameInterpolationError::Overflow)?;
    if timesteps.is_empty() || (timestep_count != 1 && timestep_count != batch) {
        return Err(FrameInterpolationError::InvalidInvocation(
            "RIFE timestep batch does not match the endpoint batch",
        ));
    }
    let plane_elements = height
        .checked_mul(width)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(FrameInterpolationError::Overflow)?;
    let element_count = batch
        .checked_mul(height)
        .and_then(|count| count.checked_mul(width))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(FrameInterpolationError::Overflow)?;
    let mut values = execution_result(backend.workspace_vec(context, element_count), context)?;
    for batch_index in 0..batch {
        context.cancellation.check()?;
        let timestep_index = if timesteps.len() == 1 {
            0
        } else {
            usize::try_from(batch_index).map_err(|_| FrameInterpolationError::Overflow)?
        };
        let timestep = *timesteps
            .get(timestep_index)
            .ok_or(FrameInterpolationError::StateMismatch)?;
        if !timestep.is_finite() || !(0.0..=1.0).contains(&timestep) {
            return Err(FrameInterpolationError::InvalidInvocation(
                "RIFE timestep must be finite and between zero and one",
            ));
        }
        for element in 0..plane_elements {
            if element.is_multiple_of(64) {
                context.cancellation.check()?;
            }
            execution_result(values.try_push(timestep), context)?;
        }
    }
    execution_result(
        tensor_from_f32(
            backend,
            &[batch, 1, height, width],
            &values,
            dtype,
            DeviceId::CPU,
            context,
        ),
        context,
    )
}

fn constant_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    value: f32,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let count = shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension).ok()?)
    });
    let count = count.ok_or(FrameInterpolationError::Overflow)?;
    let mut values = execution_result(backend.workspace_vec(context, count), context)?;
    for index in 0..count {
        if index.is_multiple_of(64) {
            context.cancellation.check()?;
        }
        execution_result(values.try_push(value), context)?;
    }
    execution_result(
        tensor_from_f32(backend, shape, &values, dtype, DeviceId::CPU, context),
        context,
    )
}

fn contiguous_narrow(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    start: i64,
    length: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let view = execution_result(
        narrow_method_exact_native(input, dimension, start, length, context.cancellation),
        context,
    )?;
    execution_result(
        contiguous_with_context_exact_native(
            backend,
            &view,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        ),
        context,
    )
}

fn film_finalize_sequence_output(
    backend: &CpuBackend,
    output_tensors: &[Tensor],
    output_frame_count: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    if output_tensors.is_empty() || output_frame_count == 0 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM sequence output is empty",
        ));
    }
    let output = execution_result(
        torch_cat_with_context_exact_native(backend, output_tensors, 0, context),
        context,
    )?;
    if output.descriptor().shape().first() != Some(&output_frame_count) {
        return Err(FrameInterpolationError::StateMismatch);
    }
    let output = execution_result(
        tensor_permute_exact_native(&output, &[0, 2, 3, 1], context.cancellation),
        context,
    )?;
    let output = execution_result(
        clone_with_context_exact_native(
            backend,
            &output,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        ),
        context,
    )?;
    let output = execution_result(
        clamp_with_context_exact_native(
            backend,
            &output,
            Some(Scalar::Float(0.0)),
            Some(Scalar::Float(1.0)),
            context,
        ),
        context,
    )?;
    context.cancellation.check()?;
    Ok(output)
}

fn rife_sequence_frame(
    backend: &CpuBackend,
    images: &Tensor,
    index: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let index = i64::try_from(index).map_err(|_| FrameInterpolationError::Overflow)?;
    let frame = execution_result(
        narrow_method_exact_native(images, 0, index, 1, context.cancellation),
        context,
    )?;
    let frame = execution_result(
        tensor_permute_exact_native(&frame, &[0, 3, 1, 2], context.cancellation),
        context,
    )?;
    execution_result(
        contiguous_with_context_exact_native(
            backend,
            &frame,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        ),
        context,
    )
}

fn pad_rife_sequence_frame(
    backend: &CpuBackend,
    frame: &Tensor,
    plan: &FrameInterpolationInvocationPlan,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    if plan.padding_bottom() == 0 && plan.padding_right() == 0 {
        context.cancellation.check()?;
        return Ok(frame.clone());
    }
    let padding_right =
        i64::try_from(plan.padding_right()).map_err(|_| FrameInterpolationError::Overflow)?;
    let padding_bottom =
        i64::try_from(plan.padding_bottom()).map_err(|_| FrameInterpolationError::Overflow)?;
    execution_result(
        functional_pad_with_context_exact_native(
            backend,
            frame,
            &[0, padding_right, 0, padding_bottom],
            FunctionalPadMode::Reflect,
            None,
            context,
        ),
        context,
    )
}

fn crop_rife_sequence_frame(
    backend: &CpuBackend,
    frame: &Tensor,
    height: u64,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let frame = execution_result(
        narrow_method_exact_native(frame, 2, 0, height, context.cancellation),
        context,
    )?;
    let frame = execution_result(
        narrow_method_exact_native(&frame, 3, 0, width, context.cancellation),
        context,
    )?;
    execution_result(
        contiguous_with_context_exact_native(
            backend,
            &frame,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        ),
        context,
    )
}

fn leaky_relu(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let module = execution_result(NativeModule::leaky_relu("rife.leaky_relu", 0.2), context)?;
    execution_result(
        module.forward_dense_inference_with_context(backend, input, context),
        context,
    )
}

fn interpolate_bilinear(
    backend: &CpuBackend,
    input: &Tensor,
    height: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    if height == 0 || width == 0 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "RIFE interpolation extent is zero",
        ));
    }
    execution_result(
        interpolate_tensor_with_context_exact_native(
            backend,
            input,
            &InterpolateConfiguration {
                output_size: Some(vec![height, width]),
                scale_factor: None,
                mode: InterpolateMode::Bilinear,
                align_corners: Some(false),
                recompute_scale_factor: None,
                antialias: false,
            },
            context,
        ),
        context,
    )
}

fn warp_rife(
    backend: &CpuBackend,
    input: &Tensor,
    flow: &Tensor,
    base_grid: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    if flow.descriptor().shape().len() != 4
        || flow.descriptor().shape().first() != input.descriptor().shape().first()
        || flow.descriptor().shape().get(1) != Some(&2)
        || flow.descriptor().shape().get(2) != input.descriptor().shape().get(2)
        || flow.descriptor().shape().get(3) != input.descriptor().shape().get(3)
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "RIFE flow shape does not match the warped tensor",
        ));
    }
    let height = *input
        .descriptor()
        .shape()
        .get(2)
        .ok_or(FrameInterpolationError::Overflow)?;
    let width = *input
        .descriptor()
        .shape()
        .get(3)
        .ok_or(FrameInterpolationError::Overflow)?;
    if height <= 1 || width <= 1 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "RIFE warp dimensions must exceed one",
        ));
    }
    let flow = execution_result(
        cast_to_with_context_exact_native(
            backend,
            flow,
            DType::F32,
            DeviceId::CPU,
            false,
            false,
            context,
        ),
        context,
    )?;
    let horizontal = contiguous_narrow(backend, &flow, 1, 0, 1, context)?;
    let horizontal = execution_result(
        real_multiply_with_context_exact_native(
            backend,
            &horizontal,
            ElementwiseOperand::Scalar(Scalar::Float(2.0 / (width - 1) as f64)),
            context,
        ),
        context,
    )?;
    let vertical = contiguous_narrow(backend, &flow, 1, 1, 1, context)?;
    let vertical = execution_result(
        real_multiply_with_context_exact_native(
            backend,
            &vertical,
            ElementwiseOperand::Scalar(Scalar::Float(2.0 / (height - 1) as f64)),
            context,
        ),
        context,
    )?;
    let normalized_flow = execution_result(
        torch_cat_with_context_exact_native(backend, &[horizontal, vertical], 1, context),
        context,
    )?;
    let grid = execution_result(
        real_add_with_context_exact_native(backend, base_grid, &normalized_flow, context),
        context,
    )?;
    let grid = execution_result(
        tensor_permute_exact_native(&grid, &[0, 2, 3, 1], context.cancellation),
        context,
    )?;
    let grid = execution_result(
        contiguous_with_context_exact_native(
            backend,
            &grid,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        ),
        context,
    )?;
    execution_result(
        grid_sample_tensor_with_context_exact_native(
            backend,
            input,
            &grid,
            GridSampleConfiguration {
                mode: GridSampleMode::Bilinear,
                padding_mode: GridPaddingMode::Border,
                align_corners: true,
            },
            context,
        ),
        context,
    )
}

pub fn film_warp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    flow: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    let input_descriptor = input.descriptor();
    let flow_descriptor = flow.descriptor();
    let input_shape = input_descriptor.shape();
    let flow_shape = flow_descriptor.shape();
    if input_shape.len() != 4
        || input_shape.first() == Some(&0)
        || input_shape.get(1) == Some(&0)
        || input_shape.get(2) == Some(&0)
        || input_shape.get(3) == Some(&0)
        || flow_shape.len() != 4
        || flow_shape.first() != input_shape.first()
        || flow_shape.get(1) != Some(&2)
        || flow_shape.get(2) != input_shape.get(2)
        || flow_shape.get(3) != input_shape.get(3)
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM flow shape does not match the warped tensor",
        ));
    }
    if !matches!(
        input_descriptor.dtype(),
        DType::F16 | DType::Bf16 | DType::F32
    ) || flow_descriptor.dtype() != input_descriptor.dtype()
        || input_descriptor.device() != DeviceId::CPU
        || flow_descriptor.device() != DeviceId::CPU
        || input_descriptor.stream() != context.stream
        || flow_descriptor.stream() != context.stream
        || !input_descriptor.is_contiguous()?
        || !flow_descriptor.is_contiguous()?
    {
        return Err(FrameInterpolationError::Placement);
    }
    let height = *input_shape
        .get(2)
        .ok_or(FrameInterpolationError::Overflow)?;
    let width = *input_shape
        .get(3)
        .ok_or(FrameInterpolationError::Overflow)?;
    let flow = execution_result(
        cast_to_with_context_exact_native(
            backend,
            flow,
            DType::F32,
            DeviceId::CPU,
            false,
            false,
            context,
        ),
        context,
    )?;
    let horizontal = contiguous_narrow(backend, &flow, 1, 0, 1, context)?;
    let horizontal = execution_result(
        real_multiply_with_context_exact_native(
            backend,
            &horizontal,
            ElementwiseOperand::Scalar(Scalar::Float(2.0 / width as f64)),
            context,
        ),
        context,
    )?;
    let vertical = contiguous_narrow(backend, &flow, 1, 1, 1, context)?;
    let vertical = execution_result(
        real_multiply_with_context_exact_native(
            backend,
            &vertical,
            ElementwiseOperand::Scalar(Scalar::Float(2.0 / height as f64)),
            context,
        ),
        context,
    )?;
    let normalized_flow = execution_result(
        torch_cat_with_context_exact_native(backend, &[horizontal, vertical], 1, context),
        context,
    )?;
    let base_grid = film_base_grid(backend, height, width, context)?;
    let grid = execution_result(
        real_add_with_context_exact_native(backend, &base_grid, &normalized_flow, context),
        context,
    )?;
    let grid = execution_result(
        tensor_permute_exact_native(&grid, &[0, 2, 3, 1], context.cancellation),
        context,
    )?;
    let grid = execution_result(
        contiguous_with_context_exact_native(
            backend,
            &grid,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        ),
        context,
    )?;
    execution_result(
        grid_sample_tensor_with_context_exact_native(
            backend,
            input,
            &grid,
            GridSampleConfiguration {
                mode: GridSampleMode::Bilinear,
                padding_mode: GridPaddingMode::Border,
                align_corners: false,
            },
            context,
        ),
        context,
    )
}

pub fn film_image_pyramid_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    levels: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, FrameInterpolationError> {
    context.cancellation.check()?;
    let shape = input.descriptor().shape();
    if shape.len() != 4
        || shape.first() == Some(&0)
        || shape.get(1) == Some(&0)
        || shape.get(2) == Some(&0)
        || shape.get(3) == Some(&0)
        || !(1..=7).contains(&levels)
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM image pyramid input or level count is invalid",
        ));
    }
    let required_extent = 1_u64
        .checked_shl(
            u32::try_from(levels.saturating_sub(1))
                .map_err(|_| FrameInterpolationError::Overflow)?,
        )
        .ok_or(FrameInterpolationError::Overflow)?;
    if shape.get(2).copied().unwrap_or_default() < required_extent
        || shape.get(3).copied().unwrap_or_default() < required_extent
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM image pyramid extent cannot support every level",
        ));
    }

    let configuration = film_average_pool_configuration();
    let mut pyramid = Vec::new();
    pyramid
        .try_reserve_exact(levels)
        .map_err(|_| FrameInterpolationError::Overflow)?;
    let mut image = input.clone();
    pyramid.push(image.clone());
    for _ in 1..levels {
        context.cancellation.check()?;
        image = execution_result(
            average_pool_2d_tensor_with_context_exact_native(
                backend,
                &image,
                &configuration,
                context,
            ),
            context,
        )?;
        pyramid.push(image.clone());
    }
    context.cancellation.check()?;
    Ok(pyramid)
}

pub fn film_flow_pyramid_synthesis_with_context_exact_native(
    backend: &CpuBackend,
    residual_pyramid: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, FrameInterpolationError> {
    context.cancellation.check()?;
    let finest = residual_pyramid
        .first()
        .ok_or(FrameInterpolationError::InvalidInvocation(
            "FILM residual flow pyramid is empty",
        ))?;
    if residual_pyramid.len() > 7 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM residual flow pyramid exceeds the source depth",
        ));
    }
    let descriptor = finest.descriptor();
    let shape = descriptor.shape();
    if shape.len() != 4
        || shape.first() == Some(&0)
        || shape.get(1) != Some(&2)
        || shape.get(2) == Some(&0)
        || shape.get(3) == Some(&0)
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != context.stream
        || !descriptor.is_contiguous()?
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM residual flows must be nonempty contiguous NCHW two-channel tensors",
        ));
    }
    let mut flow = residual_pyramid
        .last()
        .ok_or(FrameInterpolationError::StateMismatch)?
        .clone();
    let mut flow_pyramid = Vec::new();
    flow_pyramid
        .try_reserve_exact(residual_pyramid.len())
        .map_err(|_| FrameInterpolationError::Overflow)?;
    flow_pyramid.push(flow.clone());
    for residual in residual_pyramid.iter().rev().skip(1) {
        context.cancellation.check()?;
        let residual_descriptor = residual.descriptor();
        let residual_shape = residual_descriptor.shape();
        if residual_shape.len() != 4
            || residual_shape.first() != shape.first()
            || residual_shape.get(1) != Some(&2)
            || residual_descriptor.dtype() != descriptor.dtype()
            || residual_descriptor.device() != descriptor.device()
            || residual_descriptor.stream() != descriptor.stream()
            || !residual_descriptor.is_contiguous()?
        {
            return Err(FrameInterpolationError::Placement);
        }
        flow = film_upsample_double_to_residual(backend, &flow, residual, context)?;
        flow = execution_result(
            real_add_with_context_exact_native(backend, &flow, residual, context),
            context,
        )?;
        flow_pyramid.push(flow.clone());
    }
    flow_pyramid.reverse();
    context.cancellation.check()?;
    Ok(flow_pyramid)
}

pub fn film_concatenate_pyramids_with_context_exact_native(
    backend: &CpuBackend,
    first: &[Tensor],
    second: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, FrameInterpolationError> {
    context.cancellation.check()?;
    if first.is_empty() || first.len() != second.len() || first.len() > 7 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM pyramid concatenation requires equal nonempty bounded inputs",
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(first.len())
        .map_err(|_| FrameInterpolationError::Overflow)?;
    for (first, second) in first.iter().zip(second) {
        context.cancellation.check()?;
        output.push(execution_result(
            torch_cat_with_context_exact_native(
                backend,
                &[first.clone(), second.clone()],
                1,
                context,
            ),
            context,
        )?);
    }
    context.cancellation.check()?;
    Ok(output)
}

pub fn film_multiply_pyramid_with_context_exact_native(
    backend: &CpuBackend,
    pyramid: &[Tensor],
    scalar: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, FrameInterpolationError> {
    context.cancellation.check()?;
    if pyramid.is_empty() || pyramid.len() > 7 || scalar.descriptor().shape().len() != 1 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM pyramid multiplication inputs are invalid",
        ));
    }
    let mut broadcast = scalar.clone();
    for dimension in 1..=3_i64 {
        broadcast = execution_result(
            torch_unsqueeze_exact_native(&broadcast, dimension, context.cancellation),
            context,
        )?;
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(pyramid.len())
        .map_err(|_| FrameInterpolationError::Overflow)?;
    for image in pyramid {
        context.cancellation.check()?;
        if image.descriptor().shape().first() != scalar.descriptor().shape().first()
            || image.descriptor().dtype() != scalar.descriptor().dtype()
            || image.descriptor().device() != scalar.descriptor().device()
            || image.descriptor().stream() != scalar.descriptor().stream()
            || image.descriptor().stream() != context.stream
        {
            return Err(FrameInterpolationError::Placement);
        }
        output.push(execution_result(
            real_multiply_with_context_exact_native(
                backend,
                image,
                ElementwiseOperand::Tensor(&broadcast),
                context,
            ),
            context,
        )?);
    }
    context.cancellation.check()?;
    Ok(output)
}

pub fn film_warp_pyramid_with_context_exact_native(
    backend: &CpuBackend,
    feature_pyramid: &[Tensor],
    flow_pyramid: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, FrameInterpolationError> {
    context.cancellation.check()?;
    if feature_pyramid.is_empty()
        || feature_pyramid.len() != flow_pyramid.len()
        || feature_pyramid.len() > 7
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM pyramid warp requires equal nonempty bounded inputs",
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(feature_pyramid.len())
        .map_err(|_| FrameInterpolationError::Overflow)?;
    for (features, flow) in feature_pyramid.iter().zip(flow_pyramid) {
        context.cancellation.check()?;
        output.push(film_warp_with_context_exact_native(
            backend, features, flow, context,
        )?);
    }
    context.cancellation.check()?;
    Ok(output)
}

fn film_fusion_from_weights(
    backend: &CpuBackend,
    pyramid: &[Tensor],
    weights: &BTreeMap<String, Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    if pyramid.len() != 5 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM fusion requires five pyramid levels",
        ));
    }
    let reference = pyramid
        .first()
        .ok_or(FrameInterpolationError::StateMismatch)?
        .descriptor();
    let reference_shape = reference.shape();
    if reference_shape.len() != 4
        || reference_shape.first() == Some(&0)
        || reference_shape.get(1) == Some(&0)
        || reference_shape.get(2) == Some(&0)
        || reference_shape.get(3) == Some(&0)
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM fusion levels must be nonempty NCHW tensors",
        ));
    }
    for level in pyramid.iter().skip(1) {
        let descriptor = level.descriptor();
        let shape = descriptor.shape();
        if shape.len() != 4
            || shape.first() != reference_shape.first()
            || shape.get(1) == Some(&0)
            || shape.get(2) == Some(&0)
            || shape.get(3) == Some(&0)
            || descriptor.dtype() != reference.dtype()
            || descriptor.device() != reference.device()
            || descriptor.stream() != reference.stream()
            || !descriptor.is_contiguous()?
        {
            return Err(FrameInterpolationError::Placement);
        }
    }
    if reference.device() != DeviceId::CPU || reference.stream() != context.stream {
        return Err(FrameInterpolationError::Placement);
    }

    let mut net = pyramid
        .last()
        .ok_or(FrameInterpolationError::StateMismatch)?
        .clone();
    for convolution in 0..4_usize {
        context.cancellation.check()?;
        let level_index = 3_usize
            .checked_sub(convolution)
            .ok_or(FrameInterpolationError::Overflow)?;
        let level = pyramid
            .get(level_index)
            .ok_or(FrameInterpolationError::StateMismatch)?;
        let shape = level.descriptor().shape();
        let height = usize::try_from(*shape.get(2).ok_or(FrameInterpolationError::StateMismatch)?)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let width = usize::try_from(*shape.get(3).ok_or(FrameInterpolationError::StateMismatch)?)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        net = film_interpolate_nearest(backend, &net, height, width, context)?;
        net = film_convolution_from_weights(
            backend,
            &format!("fuse.convs.{convolution}.0.conv"),
            &net,
            false,
            weights,
            context,
        )?;
        net = execution_result(
            torch_cat_with_context_exact_native(backend, &[level.clone(), net], 1, context),
            context,
        )?;
        net = film_convolution_from_weights(
            backend,
            &format!("fuse.convs.{convolution}.1.conv"),
            &net,
            true,
            weights,
            context,
        )?;
        net = film_convolution_from_weights(
            backend,
            &format!("fuse.convs.{convolution}.2.conv"),
            &net,
            true,
            weights,
            context,
        )?;
    }
    let output =
        film_convolution_from_weights(backend, "fuse.output_conv", &net, false, weights, context)?;
    if output.descriptor().shape().get(1) != Some(&3) {
        return Err(FrameInterpolationError::StateMismatch);
    }
    context.cancellation.check()?;
    Ok(output)
}

fn film_interpolate_nearest(
    backend: &CpuBackend,
    input: &Tensor,
    height: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    if height == 0 || width == 0 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM fusion interpolation extent is zero",
        ));
    }
    execution_result(
        interpolate_tensor_with_context_exact_native(
            backend,
            input,
            &InterpolateConfiguration {
                output_size: Some(vec![height, width]),
                scale_factor: None,
                mode: InterpolateMode::Nearest,
                align_corners: None,
                recompute_scale_factor: None,
                antialias: false,
            },
            context,
        ),
        context,
    )
}

fn film_convolution_from_weights(
    backend: &CpuBackend,
    prefix: &str,
    input: &Tensor,
    activation: bool,
    weights: &BTreeMap<String, Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .ok_or(FrameInterpolationError::StateMismatch)?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .ok_or(FrameInterpolationError::StateMismatch)?;
    film_conv_2d_with_context_exact_native(backend, input, weight, bias, activation, context)
}

#[allow(clippy::too_many_arguments)]
fn film_synthesize_timesteps_from_pyramids(
    backend: &CpuBackend,
    first_warp_targets: &[Tensor],
    second_warp_targets: &[Tensor],
    forward_flows: &[Tensor],
    backward_flows: &[Tensor],
    timesteps: &[f32],
    weights: &BTreeMap<String, Tensor>,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    if first_warp_targets.len() != 5
        || second_warp_targets.len() != 5
        || forward_flows.len() != 5
        || backward_flows.len() != 5
        || timesteps.is_empty()
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM timestep synthesis requires five aligned levels",
        ));
    }
    let mut results = Vec::new();
    results
        .try_reserve_exact(timesteps.len())
        .map_err(|_| FrameInterpolationError::Overflow)?;
    for &timestep in timesteps {
        context.cancellation.check()?;
        if !timestep.is_finite() || !(0.0..=1.0).contains(&timestep) {
            return Err(FrameInterpolationError::InvalidInvocation(
                "FILM timestep is outside the finite unit interval",
            ));
        }
        let timestep_tensor = constant_tensor(backend, &[1], timestep, dtype, context)?;
        let inverse_timestep_tensor =
            constant_tensor(backend, &[1], 1.0_f32 - timestep, dtype, context)?;
        let backward_scaled = film_multiply_pyramid_with_context_exact_native(
            backend,
            backward_flows,
            &timestep_tensor,
            context,
        )?;
        let forward_scaled = film_multiply_pyramid_with_context_exact_native(
            backend,
            forward_flows,
            &inverse_timestep_tensor,
            context,
        )?;
        let forward_warped = film_warp_pyramid_with_context_exact_native(
            backend,
            first_warp_targets,
            &backward_scaled,
            context,
        )?;
        let backward_warped = film_warp_pyramid_with_context_exact_native(
            backend,
            second_warp_targets,
            &forward_scaled,
            context,
        )?;
        let warped = film_concatenate_pyramids_with_context_exact_native(
            backend,
            &forward_warped,
            &backward_warped,
            context,
        )?;
        let flows = film_concatenate_pyramids_with_context_exact_native(
            backend,
            &backward_scaled,
            &forward_scaled,
            context,
        )?;
        let aligned =
            film_concatenate_pyramids_with_context_exact_native(backend, &warped, &flows, context)?;
        results.push(film_fusion_from_weights(
            backend, &aligned, weights, context,
        )?);
    }
    let output = execution_result(
        torch_cat_with_context_exact_native(backend, &results, 0, context),
        context,
    )?;
    context.cancellation.check()?;
    Ok(output)
}

fn film_upsample_double_to_residual(
    backend: &CpuBackend,
    flow: &Tensor,
    residual: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let shape = residual.descriptor().shape();
    let height = usize::try_from(*shape.get(2).ok_or(FrameInterpolationError::StateMismatch)?)
        .map_err(|_| FrameInterpolationError::Overflow)?;
    let width = usize::try_from(*shape.get(3).ok_or(FrameInterpolationError::StateMismatch)?)
        .map_err(|_| FrameInterpolationError::Overflow)?;
    let flow = interpolate_bilinear(backend, flow, height, width, context)?;
    execution_result(
        real_multiply_with_context_exact_native(
            backend,
            &flow,
            ElementwiseOperand::Scalar(Scalar::Float(2.0)),
            context,
        ),
        context,
    )
}

fn validate_film_feature_pyramids(
    first_features: &[Tensor],
    second_features: &[Tensor],
    dtype: DType,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<(), FrameInterpolationError> {
    context.cancellation.check()?;
    let expected_channels = [64_u64, 192, 448, 960, 960, 960, 960];
    if first_features.len() != expected_channels.len()
        || second_features.len() != expected_channels.len()
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM flow prediction requires exact seven-level feature pyramids",
        ));
    }
    for (index, ((first_feature, second_feature), channels)) in first_features
        .iter()
        .zip(second_features)
        .zip(expected_channels)
        .enumerate()
    {
        context.cancellation.check()?;
        let first_descriptor = first_feature.descriptor();
        let second_descriptor = second_feature.descriptor();
        let shape = first_descriptor.shape();
        if shape.len() != 4
            || shape.first() == Some(&0)
            || shape.get(1) != Some(&channels)
            || shape.get(2) == Some(&0)
            || shape.get(3) == Some(&0)
            || second_descriptor.shape() != shape
            || first_descriptor.dtype() != dtype
            || second_descriptor.dtype() != dtype
            || first_descriptor.device() != DeviceId::CPU
            || second_descriptor.device() != DeviceId::CPU
            || first_descriptor.stream() != stream
            || second_descriptor.stream() != stream
            || stream != context.stream
            || !first_descriptor.is_contiguous()?
            || !second_descriptor.is_contiguous()?
        {
            return Err(FrameInterpolationError::Placement);
        }
        if index > 0 {
            let prior = first_features
                .get(index - 1)
                .ok_or(FrameInterpolationError::StateMismatch)?
                .descriptor();
            if prior.shape().get(2).copied().unwrap_or_default() / 2
                != shape.get(2).copied().unwrap_or_default()
                || prior.shape().get(3).copied().unwrap_or_default() / 2
                    != shape.get(3).copied().unwrap_or_default()
            {
                return Err(FrameInterpolationError::InvalidInvocation(
                    "FILM feature pyramid extents are not source-compatible",
                ));
            }
        }
    }
    Ok(())
}

fn film_flow_estimator_from_weights(
    backend: &CpuBackend,
    prefix: &str,
    first: &Tensor,
    second: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    let mut output = execution_result(
        torch_cat_with_context_exact_native(backend, &[first.clone(), second.clone()], 1, context),
        context,
    )?;
    for convolution in 0..5_usize {
        context.cancellation.check()?;
        let parameter = format!("{prefix}._convs.{convolution}.conv");
        let weight = weights
            .get(&format!("{parameter}.weight"))
            .ok_or(FrameInterpolationError::StateMismatch)?;
        let bias = weights
            .get(&format!("{parameter}.bias"))
            .ok_or(FrameInterpolationError::StateMismatch)?;
        output = film_conv_2d_with_context_exact_native(
            backend,
            &output,
            weight,
            bias,
            convolution < 4,
            context,
        )?;
    }
    if output.descriptor().shape().get(1) != Some(&2) {
        return Err(FrameInterpolationError::StateMismatch);
    }
    Ok(output)
}

fn film_average_pool_configuration() -> AveragePoolConfiguration {
    AveragePoolConfiguration {
        kernel_size: vec![2, 2],
        stride: Some(vec![2, 2]),
        padding: vec![0, 0],
        ceil_mode: false,
        count_include_pad: true,
        divisor_override: None,
    }
}

fn compose_film_feature_pyramid(
    backend: &CpuBackend,
    sub_pyramids: Vec<Vec<Tensor>>,
    sublevels: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, FrameInterpolationError> {
    context.cancellation.check()?;
    if sub_pyramids.is_empty()
        || sub_pyramids.len() > 7
        || !(1..=4).contains(&sublevels)
        || sub_pyramids
            .iter()
            .any(|pyramid| pyramid.len() != sublevels)
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM feature sub-pyramids are invalid",
        ));
    }
    let mut sub_pyramids = sub_pyramids.into_iter().map(Some).collect::<Vec<_>>();
    let mut feature_pyramid = Vec::new();
    feature_pyramid
        .try_reserve_exact(sub_pyramids.len())
        .map_err(|_| FrameInterpolationError::Overflow)?;
    for index in 0..sub_pyramids.len() {
        context.cancellation.check()?;
        let input_count = index
            .checked_add(1)
            .ok_or(FrameInterpolationError::Overflow)?
            .min(sublevels);
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(input_count)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        inputs.push(
            sub_pyramids
                .get(index)
                .and_then(Option::as_ref)
                .and_then(|pyramid| pyramid.first())
                .ok_or(FrameInterpolationError::StateMismatch)?
                .clone(),
        );
        for level in 1..input_count {
            context.cancellation.check()?;
            let source_index = index
                .checked_sub(level)
                .ok_or(FrameInterpolationError::Overflow)?;
            inputs.push(
                sub_pyramids
                    .get(source_index)
                    .and_then(Option::as_ref)
                    .and_then(|pyramid| pyramid.get(level))
                    .ok_or(FrameInterpolationError::StateMismatch)?
                    .clone(),
            );
        }
        let features = if inputs.len() == 1 {
            inputs.pop().ok_or(FrameInterpolationError::StateMismatch)?
        } else {
            execution_result(
                torch_cat_with_context_exact_native(backend, &inputs, 1, context),
                context,
            )?
        };
        feature_pyramid.push(features);
        if index >= sublevels.saturating_sub(1) {
            let released_index = index
                .checked_sub(sublevels.saturating_sub(1))
                .ok_or(FrameInterpolationError::Overflow)?;
            let released = sub_pyramids
                .get_mut(released_index)
                .ok_or(FrameInterpolationError::StateMismatch)?;
            if released.take().is_none() {
                return Err(FrameInterpolationError::StateMismatch);
            }
        }
    }
    context.cancellation.check()?;
    Ok(feature_pyramid)
}

#[allow(clippy::too_many_arguments)]
fn film_subtree_features_from_weights(
    backend: &CpuBackend,
    input: &Tensor,
    pooling_levels: usize,
    base_channels: usize,
    sublevels: usize,
    weights: &BTreeMap<String, Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, FrameInterpolationError> {
    context.cancellation.check()?;
    if base_channels == 0
        || !(1..=4).contains(&sublevels)
        || !(1..=sublevels).contains(&pooling_levels)
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM subtree configuration is invalid",
        ));
    }
    let mut pyramid = Vec::new();
    pyramid
        .try_reserve_exact(sublevels)
        .map_err(|_| FrameInterpolationError::Overflow)?;
    let mut head = input.clone();
    for level in 0..sublevels {
        context.cancellation.check()?;
        let prefix = format!("extract.extract_sublevels.convs.{level}");
        let first_weight = weights
            .get(&format!("{prefix}.0.conv.weight"))
            .ok_or(FrameInterpolationError::StateMismatch)?;
        let first_bias = weights
            .get(&format!("{prefix}.0.conv.bias"))
            .ok_or(FrameInterpolationError::StateMismatch)?;
        head = film_conv_2d_with_context_exact_native(
            backend,
            &head,
            first_weight,
            first_bias,
            true,
            context,
        )?;
        let second_weight = weights
            .get(&format!("{prefix}.1.conv.weight"))
            .ok_or(FrameInterpolationError::StateMismatch)?;
        let second_bias = weights
            .get(&format!("{prefix}.1.conv.bias"))
            .ok_or(FrameInterpolationError::StateMismatch)?;
        head = film_conv_2d_with_context_exact_native(
            backend,
            &head,
            second_weight,
            second_bias,
            true,
            context,
        )?;
        let output_channels = base_channels
            .checked_shl(u32::try_from(level).map_err(|_| FrameInterpolationError::Overflow)?)
            .ok_or(FrameInterpolationError::Overflow)?;
        if head.descriptor().shape().get(1)
            != Some(&u64::try_from(output_channels).map_err(|_| FrameInterpolationError::Overflow)?)
        {
            return Err(FrameInterpolationError::StateMismatch);
        }
        pyramid.push(head.clone());
        if level < pooling_levels.saturating_sub(1) {
            head = execution_result(
                average_pool_2d_tensor_with_context_exact_native(
                    backend,
                    &head,
                    &film_average_pool_configuration(),
                    context,
                ),
                context,
            )?;
        }
    }
    context.cancellation.check()?;
    Ok(pyramid)
}

pub fn film_conv_2d_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    activation: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    context.cancellation.check()?;
    let input_shape = input.descriptor().shape();
    let weight_shape = weight.descriptor().shape();
    let bias_shape = bias.descriptor().shape();
    if input_shape.len() != 4
        || input_shape.first() == Some(&0)
        || input_shape.get(1) == Some(&0)
        || input_shape.get(2) == Some(&0)
        || input_shape.get(3) == Some(&0)
        || weight_shape.len() != 4
        || weight_shape.first() == Some(&0)
        || weight_shape.get(1) != input_shape.get(1)
        || weight_shape.get(2) != weight_shape.get(3)
        || !matches!(weight_shape.get(2), Some(1..=3))
        || bias_shape.len() != 1
        || bias_shape.first() != weight_shape.first()
    {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM convolution tensor shapes are invalid",
        ));
    }
    let input_channels = usize::try_from(
        *input_shape
            .get(1)
            .ok_or(FrameInterpolationError::Overflow)?,
    )
    .map_err(|_| FrameInterpolationError::Overflow)?;
    let output_channels = usize::try_from(
        *weight_shape
            .first()
            .ok_or(FrameInterpolationError::Overflow)?,
    )
    .map_err(|_| FrameInterpolationError::Overflow)?;
    let kernel = usize::try_from(
        *weight_shape
            .get(2)
            .ok_or(FrameInterpolationError::Overflow)?,
    )
    .map_err(|_| FrameInterpolationError::Overflow)?;
    let input = if kernel.is_multiple_of(2) {
        execution_result(
            functional_pad_with_context_exact_native(
                backend,
                input,
                &[0, 1, 0, 1],
                FunctionalPadMode::Constant,
                None,
                context,
            ),
            context,
        )?
    } else {
        input.clone()
    };
    let geometry = execution_result(
        ConvolutionGeometry::new_with_padding_mode(
            2,
            vec![1; 2],
            vec![
                if kernel.is_multiple_of(2) {
                    0
                } else {
                    kernel / 2
                };
                2
            ],
            vec![1; 2],
            1,
            false,
            vec![0; 2],
            ConvolutionPaddingMode::Zeros,
        ),
        context,
    )?;
    let mut module = execution_result(
        disable_weight_init_convolution_exact_native(
            "film.conv2d",
            input_channels,
            output_channels,
            vec![kernel; 2],
            true,
            geometry,
        ),
        context,
    )?;
    execution_result(
        module.load_dense_parameters(weight.clone(), Some(bias.clone())),
        context,
    )?;
    let output = execution_result(
        module.forward_dense_inference_with_context(backend, &input, context),
        context,
    )?;
    if activation {
        leaky_relu(backend, &output, context)
    } else {
        Ok(output)
    }
}

fn film_base_grid(
    backend: &CpuBackend,
    height: u64,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let height_usize = usize::try_from(height).map_err(|_| FrameInterpolationError::Overflow)?;
    let width_usize = usize::try_from(width).map_err(|_| FrameInterpolationError::Overflow)?;
    if height_usize == 0 || width_usize == 0 {
        return Err(FrameInterpolationError::InvalidInvocation(
            "FILM warp dimensions must be nonzero",
        ));
    }
    let plane = height_usize
        .checked_mul(width_usize)
        .ok_or(FrameInterpolationError::Overflow)?;
    let mut values = execution_result(
        backend.workspace_vec(
            context,
            plane
                .checked_mul(2)
                .ok_or(FrameInterpolationError::Overflow)?,
        ),
        context,
    )?;
    for y in 0..height_usize {
        for x in 0..width_usize {
            let index = y
                .checked_mul(width_usize)
                .and_then(|value| value.checked_add(x))
                .ok_or(FrameInterpolationError::Overflow)?;
            if index.is_multiple_of(64) {
                context.cancellation.check()?;
            }
            execution_result(
                values.try_push(-1.0 + (2.0 * x as f32 + 1.0) / width_usize as f32),
                context,
            )?;
        }
    }
    for y in 0..height_usize {
        for x in 0..width_usize {
            let index = y
                .checked_mul(width_usize)
                .and_then(|value| value.checked_add(x))
                .ok_or(FrameInterpolationError::Overflow)?;
            if index.is_multiple_of(64) {
                context.cancellation.check()?;
            }
            execution_result(
                values.try_push(-1.0 + (2.0 * y as f32 + 1.0) / height_usize as f32),
                context,
            )?;
        }
    }
    execution_result(
        tensor_from_f32(
            backend,
            &[1, 2, height, width],
            &values,
            DType::F32,
            DeviceId::CPU,
            context,
        ),
        context,
    )
}

fn rife_base_grid(
    backend: &CpuBackend,
    height: u64,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FrameInterpolationError> {
    let height_usize = usize::try_from(height).map_err(|_| FrameInterpolationError::Overflow)?;
    let width_usize = usize::try_from(width).map_err(|_| FrameInterpolationError::Overflow)?;
    let plane = height_usize
        .checked_mul(width_usize)
        .ok_or(FrameInterpolationError::Overflow)?;
    let mut values = execution_result(
        backend.workspace_vec(
            context,
            plane
                .checked_mul(2)
                .ok_or(FrameInterpolationError::Overflow)?,
        ),
        context,
    )?;
    for y in 0..height_usize {
        for x in 0..width_usize {
            let index = y
                .checked_mul(width_usize)
                .and_then(|value| value.checked_add(x))
                .ok_or(FrameInterpolationError::Overflow)?;
            if index.is_multiple_of(64) {
                context.cancellation.check()?;
            }
            execution_result(
                values.try_push(-1.0 + 2.0 * x as f32 / (width_usize - 1) as f32),
                context,
            )?;
        }
    }
    for y in 0..height_usize {
        for x in 0..width_usize {
            let index = y
                .checked_mul(width_usize)
                .and_then(|value| value.checked_add(x))
                .ok_or(FrameInterpolationError::Overflow)?;
            if index.is_multiple_of(64) {
                context.cancellation.check()?;
            }
            execution_result(
                values.try_push(-1.0 + 2.0 * y as f32 / (height_usize - 1) as f32),
                context,
            )?;
        }
    }
    execution_result(
        tensor_from_f32(
            backend,
            &[1, 2, height, width],
            &values,
            DType::F32,
            DeviceId::CPU,
            context,
        ),
        context,
    )
}

fn normalize_and_detect(
    weights: BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<(FrameInterpolationProfile, BTreeMap<String, Tensor>), FrameInterpolationError> {
    if weights.contains_key(FILM_MARKER) {
        return Ok((FrameInterpolationProfile::Film, weights));
    }
    let mut normalized = BTreeMap::new();
    for (key, tensor) in weights {
        cancellation.check()?;
        let key = key.strip_prefix("module.").unwrap_or(&key).to_owned();
        let key = key.strip_prefix("flownet.").unwrap_or(&key).to_owned();
        if key.starts_with("teacher.") || key.starts_with("caltime.") {
            continue;
        }
        let key = (0..5)
            .find_map(|index| {
                let prefix = format!("block{index}.");
                key.strip_prefix(&prefix)
                    .map(|suffix| format!("blocks.{index}.{suffix}"))
            })
            .unwrap_or(key);
        if normalized.insert(key.clone(), tensor).is_some() {
            return Err(FrameInterpolationError::KeyCollision(key));
        }
    }
    let head = normalized
        .get("encode.cnn3.weight")
        .ok_or(FrameInterpolationError::Unrecognized)?;
    let head_channels = *head
        .descriptor()
        .shape()
        .get(1)
        .ok_or(FrameInterpolationError::Unrecognized)?;
    if head.descriptor().shape().len() != 4 || head_channels == 0 {
        return Err(FrameInterpolationError::Unrecognized);
    }
    let mut block_channels = [0; 5];
    for (index, channel) in block_channels.iter_mut().enumerate() {
        let key = format!("blocks.{index}.conv0.1.0.weight");
        let tensor = normalized
            .get(&key)
            .ok_or(FrameInterpolationError::Unrecognized)?;
        *channel = *tensor
            .descriptor()
            .shape()
            .first()
            .ok_or(FrameInterpolationError::Unrecognized)?;
        if tensor.descriptor().shape().len() != 4 || *channel == 0 || !(*channel).is_multiple_of(2)
        {
            return Err(FrameInterpolationError::Unrecognized);
        }
    }
    Ok((
        FrameInterpolationProfile::Rife {
            head_channels,
            block_channels,
        },
        normalized,
    ))
}

fn weight_manifest(
    profile: &FrameInterpolationProfile,
) -> Result<BTreeMap<String, Vec<u64>>, FrameInterpolationError> {
    match profile {
        FrameInterpolationProfile::Film => film_manifest(),
        FrameInterpolationProfile::Rife {
            head_channels,
            block_channels,
        } => rife_manifest(*head_channels, *block_channels),
    }
}

fn insert_parameter(
    manifest: &mut BTreeMap<String, Vec<u64>>,
    prefix: String,
    weight: Vec<u64>,
    bias: u64,
) {
    manifest.insert(format!("{prefix}.weight"), weight);
    manifest.insert(format!("{prefix}.bias"), vec![bias]);
}

fn rife_manifest(
    head_channels: u64,
    channels: [u64; 5],
) -> Result<BTreeMap<String, Vec<u64>>, FrameInterpolationError> {
    let mut manifest = BTreeMap::new();
    insert_parameter(&mut manifest, "encode.cnn0".into(), vec![16, 3, 3, 3], 16);
    insert_parameter(&mut manifest, "encode.cnn1".into(), vec![16, 16, 3, 3], 16);
    insert_parameter(&mut manifest, "encode.cnn2".into(), vec![16, 16, 3, 3], 16);
    insert_parameter(
        &mut manifest,
        "encode.cnn3".into(),
        vec![16, head_channels, 4, 4],
        head_channels,
    );
    for (index, channel) in channels.into_iter().enumerate() {
        let input = if index == 0 {
            7_u64
                .checked_add(
                    head_channels
                        .checked_mul(2)
                        .ok_or(FrameInterpolationError::Overflow)?,
                )
                .ok_or(FrameInterpolationError::Overflow)?
        } else {
            20_u64
                .checked_add(
                    head_channels
                        .checked_mul(2)
                        .ok_or(FrameInterpolationError::Overflow)?,
                )
                .ok_or(FrameInterpolationError::Overflow)?
        };
        insert_parameter(
            &mut manifest,
            format!("blocks.{index}.conv0.0.0"),
            vec![channel / 2, input, 3, 3],
            channel / 2,
        );
        insert_parameter(
            &mut manifest,
            format!("blocks.{index}.conv0.1.0"),
            vec![channel, channel / 2, 3, 3],
            channel,
        );
        for residual in 0..8 {
            insert_parameter(
                &mut manifest,
                format!("blocks.{index}.convblock.{residual}.conv"),
                vec![channel, channel, 3, 3],
                channel,
            );
            manifest.insert(
                format!("blocks.{index}.convblock.{residual}.beta"),
                vec![1, channel, 1, 1],
            );
        }
        insert_parameter(
            &mut manifest,
            format!("blocks.{index}.lastconv.0"),
            vec![channel, 52, 4, 4],
            52,
        );
    }
    Ok(manifest)
}

fn film_manifest() -> Result<BTreeMap<String, Vec<u64>>, FrameInterpolationError> {
    let mut manifest = BTreeMap::new();
    let mut input = 3;
    for level in 0..4 {
        let output = 64_u64
            .checked_shl(u32::try_from(level).map_err(|_| FrameInterpolationError::Overflow)?)
            .ok_or(FrameInterpolationError::Overflow)?;
        insert_parameter(
            &mut manifest,
            format!("extract.extract_sublevels.convs.{level}.0.conv"),
            vec![output, input, 3, 3],
            output,
        );
        insert_parameter(
            &mut manifest,
            format!("extract.extract_sublevels.convs.{level}.1.conv"),
            vec![output, output, 3, 3],
            output,
        );
        input = output;
    }
    let predictors = [
        ("_predictor", 1920, 256),
        ("_predictors.0", 896, 128),
        ("_predictors.1", 384, 64),
        ("_predictors.2", 128, 32),
    ];
    for (name, input, filter) in predictors {
        for convolution in 0..3 {
            insert_parameter(
                &mut manifest,
                format!("predict_flow.{name}._convs.{convolution}.conv"),
                vec![filter, if convolution == 0 { input } else { filter }, 3, 3],
                filter,
            );
        }
        insert_parameter(
            &mut manifest,
            format!("predict_flow.{name}._convs.3.conv"),
            vec![filter / 2, filter, 1, 1],
            filter / 2,
        );
        insert_parameter(
            &mut manifest,
            format!("predict_flow.{name}._convs.4.conv"),
            vec![2, filter / 2, 1, 1],
            2,
        );
    }
    insert_parameter(
        &mut manifest,
        "fuse.output_conv".into(),
        vec![3, 64, 1, 1],
        3,
    );
    for (index, input, joined, output) in [
        (0, 1930, 2442, 512),
        (1, 512, 1162, 256),
        (2, 256, 522, 128),
        (3, 128, 202, 64),
    ] {
        insert_parameter(
            &mut manifest,
            format!("fuse.convs.{index}.0.conv"),
            vec![output, input, 2, 2],
            output,
        );
        insert_parameter(
            &mut manifest,
            format!("fuse.convs.{index}.1.conv"),
            vec![output, joined, 3, 3],
            output,
        );
        insert_parameter(
            &mut manifest,
            format!("fuse.convs.{index}.2.conv"),
            vec![output, output, 3, 3],
            output,
        );
    }
    Ok(manifest)
}

fn semantic_digest(
    profile: &FrameInterpolationProfile,
    artifact: &str,
    weights: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, FrameInterpolationError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.frame-interpolation-model.v1");
    hasher.update(FRAME_INTERPOLATION_SOURCE_SHA256.as_bytes());
    match profile {
        FrameInterpolationProfile::Film => hasher.update(FILM_SOURCE_SHA256.as_bytes()),
        FrameInterpolationProfile::Rife {
            head_channels,
            block_channels,
        } => {
            hasher.update(RIFE_SOURCE_SHA256.as_bytes());
            hasher.update(head_channels.to_le_bytes());
            for channel in block_channels {
                hasher.update(channel.to_le_bytes());
            }
        }
    }
    hasher.update(artifact.as_bytes());
    hasher.update([
        match weights
            .values()
            .next()
            .map(|tensor| tensor.descriptor().dtype())
        {
            Some(DType::F16) => 0,
            Some(DType::Bf16) => 1,
            Some(DType::F32) => 2,
            _ => return Err(FrameInterpolationError::Placement),
        },
    ]);
    for (key, tensor) in weights {
        cancellation.check()?;
        hasher.update(
            u64::try_from(key.len())
                .map_err(|_| FrameInterpolationError::Overflow)?
                .to_le_bytes(),
        );
        hasher.update(key.as_bytes());
        for dimension in tensor.descriptor().shape() {
            hasher.update(dimension.to_le_bytes());
        }
        hasher.update(tensor.contiguous_bytes()?);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reduced_film_fusion_weights(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        pyramid_channels: usize,
        initial_channels: &[usize],
        joined_channels: &[usize],
    ) -> Result<BTreeMap<String, Tensor>, FrameInterpolationError> {
        let tensor = |shape: &[u64], values: &[f32]| {
            tensor_from_f32(backend, shape, values, DType::F32, DeviceId::CPU, context)
                .map_err(|error| FrameInterpolationError::Execution(error.to_string()))
        };
        let joined_channel_count = pyramid_channels
            .checked_add(1)
            .ok_or(FrameInterpolationError::Overflow)?;
        let joined_channel_count_u64 =
            u64::try_from(joined_channel_count).map_err(|_| FrameInterpolationError::Overflow)?;
        let mut weights = BTreeMap::new();
        for convolution in 0..4_usize {
            let first_input_channels = if convolution == 0 {
                pyramid_channels
            } else {
                1
            };
            let mut first_weight = vec![
                0.0_f32;
                first_input_channels
                    .checked_mul(4)
                    .ok_or(FrameInterpolationError::Overflow)?
            ];
            let selected = if convolution == 0 {
                initial_channels
            } else {
                &[0_usize]
            };
            for &channel in selected {
                let index = channel
                    .checked_mul(4)
                    .ok_or(FrameInterpolationError::Overflow)?;
                *first_weight
                    .get_mut(index)
                    .ok_or(FrameInterpolationError::StateMismatch)? = 1.0;
            }
            weights.insert(
                format!("fuse.convs.{convolution}.0.conv.weight"),
                tensor(
                    &[
                        1,
                        u64::try_from(first_input_channels)
                            .map_err(|_| FrameInterpolationError::Overflow)?,
                        2,
                        2,
                    ],
                    &first_weight,
                )?,
            );
            weights.insert(
                format!("fuse.convs.{convolution}.0.conv.bias"),
                tensor(&[1], &[0.0])?,
            );

            let mut joined_weight = vec![
                0.0_f32;
                joined_channel_count
                    .checked_mul(9)
                    .ok_or(FrameInterpolationError::Overflow)?
            ];
            for &channel in joined_channels {
                let index = channel
                    .checked_mul(9)
                    .and_then(|index| index.checked_add(4))
                    .ok_or(FrameInterpolationError::Overflow)?;
                *joined_weight
                    .get_mut(index)
                    .ok_or(FrameInterpolationError::StateMismatch)? = 1.0;
            }
            weights.insert(
                format!("fuse.convs.{convolution}.1.conv.weight"),
                tensor(&[1, joined_channel_count_u64, 3, 3], &joined_weight)?,
            );
            weights.insert(
                format!("fuse.convs.{convolution}.1.conv.bias"),
                tensor(&[1], &[0.0])?,
            );

            let mut final_weight = vec![0.0_f32; 9];
            *final_weight
                .get_mut(4)
                .ok_or(FrameInterpolationError::StateMismatch)? = 1.0;
            weights.insert(
                format!("fuse.convs.{convolution}.2.conv.weight"),
                tensor(&[1, 1, 3, 3], &final_weight)?,
            );
            weights.insert(
                format!("fuse.convs.{convolution}.2.conv.bias"),
                tensor(&[1], &[0.0])?,
            );
        }
        weights.insert(
            "fuse.output_conv.weight".into(),
            tensor(&[3, 1, 1, 1], &[1.0, 2.0, 3.0])?,
        );
        weights.insert(
            "fuse.output_conv.bias".into(),
            tensor(&[3], &[0.0, 0.0, 0.0])?,
        );
        Ok(weights)
    }

    #[test]
    fn film_fusion_executes_nearest_coarse_to_fine_convolution_schedule()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(2 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let tensor = |shape: &[u64], values: &[f32]| {
            tensor_from_f32(&backend, shape, values, DType::F32, DeviceId::CPU, &context)
                .map_err(|error| FrameInterpolationError::Execution(error.to_string()))
        };
        let pyramid = [1.0_f32, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .zip([16_u64, 8, 4, 2, 1])
            .map(|(value, extent)| {
                let element_count = usize::try_from(extent)
                    .ok()
                    .and_then(|extent| extent.checked_mul(extent))
                    .ok_or(FrameInterpolationError::Overflow)?;
                tensor(&[1, 1, extent, extent], &vec![value; element_count])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_bytes = pyramid
            .iter()
            .map(Tensor::contiguous_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let weights = reduced_film_fusion_weights(&backend, &context, 1, &[0], &[0, 1])?;

        let output = film_fusion_from_weights(&backend, &pyramid, &weights, &context)?;
        assert_eq!(output.descriptor().shape(), &[1, 3, 16, 16]);
        for (linear, expected) in [(0_u64, 15.0_f32), (256, 30.0), (512, 45.0)] {
            let actual = match DType::F32.decode_scalar(output.linear_element_bytes(linear)?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert_eq!(actual, expected);
        }
        assert_ne!(
            output.storage_id(),
            pyramid
                .first()
                .ok_or(FrameInterpolationError::StateMismatch)?
                .storage_id()
        );
        for (level, bytes) in pyramid.iter().zip(source_bytes) {
            assert_eq!(level.contiguous_bytes()?, bytes);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(2 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            film_fusion_from_weights(&backend, &pyramid, &weights, &cancelled_context),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_multi_timestep_synthesis_reuses_flows_and_orders_outputs()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let tensor = |shape: &[u64], values: &[f32]| {
            tensor_from_f32(&backend, shape, values, DType::F32, DeviceId::CPU, &context)
                .map_err(|error| FrameInterpolationError::Execution(error.to_string()))
        };
        let first_warp_targets = (0..5)
            .map(|_| tensor(&[1, 1, 1, 1], &[10.0]))
            .collect::<Result<Vec<_>, _>>()?;
        let second_warp_targets = (0..5)
            .map(|_| tensor(&[1, 1, 1, 1], &[20.0]))
            .collect::<Result<Vec<_>, _>>()?;
        let forward_flows = (0..5)
            .map(|_| tensor(&[1, 2, 1, 1], &[4.0, 0.0]))
            .collect::<Result<Vec<_>, _>>()?;
        let backward_flows = (0..5)
            .map(|_| tensor(&[1, 2, 1, 1], &[2.0, 0.0]))
            .collect::<Result<Vec<_>, _>>()?;
        let source_bytes = first_warp_targets
            .iter()
            .chain(&second_warp_targets)
            .chain(&forward_flows)
            .chain(&backward_flows)
            .map(Tensor::contiguous_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let weights = reduced_film_fusion_weights(&backend, &context, 6, &[2, 4], &[6])?;

        let output = film_synthesize_timesteps_from_pyramids(
            &backend,
            &first_warp_targets,
            &second_warp_targets,
            &forward_flows,
            &backward_flows,
            &[0.25, 0.75],
            &weights,
            DType::F32,
            &context,
        )?;
        assert_eq!(output.descriptor().shape(), &[2, 3, 1, 1]);
        for (linear, expected) in [3.5_f32, 7.0, 10.5, 2.5, 5.0, 7.5].into_iter().enumerate() {
            let linear = u64::try_from(linear).map_err(|_| FrameInterpolationError::Overflow)?;
            let actual = match DType::F32.decode_scalar(output.linear_element_bytes(linear)?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert_eq!(actual, expected);
        }
        assert_ne!(
            output.storage_id(),
            first_warp_targets
                .first()
                .ok_or(FrameInterpolationError::StateMismatch)?
                .storage_id()
        );
        for (source, bytes) in first_warp_targets
            .iter()
            .chain(&second_warp_targets)
            .chain(&forward_flows)
            .chain(&backward_flows)
            .zip(source_bytes)
        {
            assert_eq!(source.contiguous_bytes()?, bytes);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            film_synthesize_timesteps_from_pyramids(
                &backend,
                &first_warp_targets,
                &second_warp_targets,
                &forward_flows,
                &backward_flows,
                &[0.25, 0.75],
                &weights,
                DType::F32,
                &cancelled_context,
            ),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_sequence_finalization_preserves_endpoints_midpoints_and_failure_atomicity()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let tensor = |batch: u64, values: &[f32]| {
            tensor_from_f32(
                &backend,
                &[batch, 3, 1, 1],
                values,
                DType::F32,
                DeviceId::CPU,
                &context,
            )
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))
        };
        let tensors = vec![
            tensor(1, &[0.0, 0.01, 0.02])?,
            tensor(2, &[0.1, 0.11, 0.12, 0.2, 0.21, 0.22])?,
            tensor(1, &[0.3, 0.31, 0.32])?,
            tensor(2, &[0.4, 0.41, 0.42, 0.5, 0.51, 0.52])?,
            tensor(1, &[0.6, 0.61, 0.62])?,
        ];
        let source_bytes = tensors
            .iter()
            .map(Tensor::contiguous_bytes)
            .collect::<Result<Vec<_>, _>>()?;

        let output = film_finalize_sequence_output(&backend, &tensors, 7, &context)?;
        assert_eq!(output.descriptor().shape(), &[7, 1, 1, 3]);
        for (linear, expected) in [
            0.0_f32, 0.01, 0.02, 0.1, 0.11, 0.12, 0.2, 0.21, 0.22, 0.3, 0.31, 0.32, 0.4, 0.41,
            0.42, 0.5, 0.51, 0.52, 0.6, 0.61, 0.62,
        ]
        .into_iter()
        .enumerate()
        {
            let linear = u64::try_from(linear).map_err(|_| FrameInterpolationError::Overflow)?;
            let actual = match DType::F32.decode_scalar(output.linear_element_bytes(linear)?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert!((actual - expected).abs() <= f32::EPSILON);
        }
        for (source, bytes) in tensors.iter().zip(source_bytes) {
            assert_eq!(source.contiguous_bytes()?, bytes);
        }
        assert!(matches!(
            film_finalize_sequence_output(&backend, &tensors, 6, &context),
            Err(FrameInterpolationError::StateMismatch)
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            film_finalize_sequence_output(&backend, &tensors, 7, &cancelled_context),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn frame_interpolation_classifies_typed_resource_exhaustion_without_message_matching()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext};

        let (_, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        for error in [
            TensorError::AllocationFailed {
                requested: 64,
                reason: "fixture".into(),
            },
            TensorError::ResourceLimitExceeded {
                resource: "fixture",
                limit: 1,
            },
            TensorError::WorkspaceAuthorizationExceeded {
                requested: 64,
                authorized: 32,
                in_use: 0,
            },
        ] {
            assert!(matches!(
                execution_result::<(), _>(Err(error), &context),
                Err(FrameInterpolationError::ResourceExhausted(_))
            ));
        }
        assert!(matches!(
            execution_result::<(), _>(Err(TensorError::ShapeOverflow), &context),
            Err(FrameInterpolationError::Execution(_))
        ));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            execution_result::<(), _>(
                Err(TensorError::WorkspaceAuthorizationExceeded {
                    requested: 64,
                    authorized: 32,
                    in_use: 0,
                }),
                &cancelled_context,
            ),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_pyramid_algebra_delegates_concat_broadcast_multiply_and_warp()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let make_tensor = |value: f32, channels: u64| {
            tensor_from_f32(
                &backend,
                &[1, channels, 1, 1],
                &vec![
                    value;
                    usize::try_from(channels).map_err(|_| FrameInterpolationError::Overflow)?
                ],
                DType::F32,
                DeviceId::CPU,
                &context,
            )
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))
        };
        let first = vec![make_tensor(1.0, 1)?, make_tensor(3.0, 1)?];
        let second = vec![make_tensor(2.0, 1)?, make_tensor(4.0, 1)?];
        let first_bytes = first
            .iter()
            .map(Tensor::contiguous_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let concatenated = film_concatenate_pyramids_with_context_exact_native(
            &backend, &first, &second, &context,
        )?;
        assert_eq!(concatenated[0].descriptor().shape(), &[1, 2, 1, 1]);
        for (linear, expected) in [1.0_f32, 2.0].into_iter().enumerate() {
            let actual = match DType::F32.decode_scalar(concatenated[0].linear_element_bytes(
                u64::try_from(linear).map_err(|_| FrameInterpolationError::Overflow)?,
            )?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert_eq!(actual, expected);
        }
        let scalar = tensor_from_f32(&backend, &[1], &[0.5], DType::F32, DeviceId::CPU, &context)
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let multiplied =
            film_multiply_pyramid_with_context_exact_native(&backend, &first, &scalar, &context)?;
        for (index, expected) in [0.5_f32, 1.5].into_iter().enumerate() {
            let actual =
                match DType::F32.decode_scalar(multiplied[index].linear_element_bytes(0)?)? {
                    DecodedScalar::Real(value) => value as f32,
                    _ => return Err(FrameInterpolationError::StateMismatch),
                };
            assert_eq!(actual, expected);
        }
        let flows = vec![make_tensor(0.0, 2)?, make_tensor(0.0, 2)?];
        let warped =
            film_warp_pyramid_with_context_exact_native(&backend, &first, &flows, &context)?;
        for (index, expected) in [1.0_f32, 3.0].into_iter().enumerate() {
            let actual = match DType::F32.decode_scalar(warped[index].linear_element_bytes(0)?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert_eq!(actual, expected);
            assert_ne!(warped[index].storage_id(), first[index].storage_id());
            assert_eq!(first[index].contiguous_bytes()?, first_bytes[index]);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            film_warp_pyramid_with_context_exact_native(
                &backend,
                &first,
                &flows,
                &cancelled_context,
            ),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_flow_estimator_executes_five_source_convolutions() -> Result<(), FrameInterpolationError>
    {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let prefix = "predict_flow.reduced";
        let shapes = [
            ([2_u64, 2, 3, 3], 2_usize),
            ([2, 2, 3, 3], 2),
            ([2, 2, 3, 3], 2),
            ([1, 2, 1, 1], 1),
            ([2, 1, 1, 1], 2),
        ];
        let mut weights = BTreeMap::new();
        for (convolution, (shape, output_channels)) in shapes.into_iter().enumerate() {
            let count = shape.iter().try_fold(1_usize, |count, extent| {
                count
                    .checked_mul(
                        usize::try_from(*extent).map_err(|_| FrameInterpolationError::Overflow)?,
                    )
                    .ok_or(FrameInterpolationError::Overflow)
            })?;
            let mut values = vec![0.0_f32; count];
            match convolution {
                0 => {
                    values[4] = 1.0;
                    values[13] = 1.0;
                    values[22] = 1.0;
                    values[31] = 1.0;
                }
                1 | 2 => {
                    values[4] = 1.0;
                    values[31] = 1.0;
                }
                3 => {
                    values[0] = 1.0;
                    values[1] = 1.0;
                }
                4 => {
                    values[0] = 1.0;
                    values[1] = 2.0;
                }
                _ => return Err(FrameInterpolationError::StateMismatch),
            }
            let parameter = format!("{prefix}._convs.{convolution}.conv");
            weights.insert(
                format!("{parameter}.weight"),
                tensor_from_f32(
                    &backend,
                    &shape,
                    &values,
                    DType::F32,
                    DeviceId::CPU,
                    &context,
                )
                .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?,
            );
            weights.insert(
                format!("{parameter}.bias"),
                tensor_from_f32(
                    &backend,
                    &[u64::try_from(output_channels)
                        .map_err(|_| FrameInterpolationError::Overflow)?],
                    &vec![0.0; output_channels],
                    DType::F32,
                    DeviceId::CPU,
                    &context,
                )
                .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?,
            );
        }
        let first = tensor_from_f32(
            &backend,
            &[1, 1, 1, 1],
            &[1.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let second = tensor_from_f32(
            &backend,
            &[1, 1, 1, 1],
            &[2.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let first_bytes = first.contiguous_bytes()?;
        let second_bytes = second.contiguous_bytes()?;
        let output = film_flow_estimator_from_weights(
            &backend, prefix, &first, &second, &weights, &context,
        )?;
        assert_eq!(output.descriptor().shape(), &[1, 2, 1, 1]);
        for (linear, expected) in [6.0_f32, 12.0].into_iter().enumerate() {
            let actual = match DType::F32.decode_scalar(output.linear_element_bytes(
                u64::try_from(linear).map_err(|_| FrameInterpolationError::Overflow)?,
            )?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert_eq!(actual, expected);
        }
        assert_ne!(output.storage_id(), first.storage_id());
        assert_ne!(output.storage_id(), second.storage_id());
        assert_eq!(first.contiguous_bytes()?, first_bytes);
        assert_eq!(second.contiguous_bytes()?, second_bytes);
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let rife = NativeFrameInterpolationModel::reduced_rife_test_fixture(&backend, &context)?;
        assert!(matches!(
            rife.film_residual_flow_pyramid(&backend, &[], &[], &context),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));
        Ok(())
    }

    #[test]
    fn film_flow_pyramid_synthesis_upsamples_doubles_adds_and_reverses()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let residuals = [
            ([1, 2, 4, 4], 100.0_f32),
            ([1, 2, 2, 2], 10.0_f32),
            ([1, 2, 1, 1], 1.0_f32),
        ]
        .into_iter()
        .map(|(shape, value)| {
            let count = shape.iter().try_fold(1_usize, |count, extent| {
                count
                    .checked_mul(
                        usize::try_from(*extent).map_err(|_| FrameInterpolationError::Overflow)?,
                    )
                    .ok_or(FrameInterpolationError::Overflow)
            })?;
            tensor_from_f32(
                &backend,
                &shape,
                &vec![value; count],
                DType::F32,
                DeviceId::CPU,
                &context,
            )
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
        let source_bytes = residuals
            .iter()
            .map(Tensor::contiguous_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let flows =
            film_flow_pyramid_synthesis_with_context_exact_native(&backend, &residuals, &context)?;
        assert_eq!(flows.len(), 3);
        for (index, (shape, expected)) in [
            ([1, 2, 4, 4], 124.0_f32),
            ([1, 2, 2, 2], 12.0_f32),
            ([1, 2, 1, 1], 1.0_f32),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(flows[index].descriptor().shape(), &shape);
            let actual = match DType::F32.decode_scalar(flows[index].linear_element_bytes(0)?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert_eq!(actual, expected);
        }
        assert_eq!(flows[2].storage_id(), residuals[2].storage_id());
        assert_ne!(flows[0].storage_id(), residuals[0].storage_id());
        for (residual, expected) in residuals.iter().zip(source_bytes) {
            assert_eq!(residual.contiguous_bytes()?, expected);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            film_flow_pyramid_synthesis_with_context_exact_native(
                &backend,
                &residuals,
                &cancelled_context,
            ),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_feature_pyramid_concatenates_source_diagonals_and_releases_inputs()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let mut sub_pyramids = Vec::new();
        for pyramid_index in 0..4_usize {
            let mut pyramid = Vec::new();
            for level in 0..3_usize {
                let value = pyramid_index
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(level))
                    .ok_or(FrameInterpolationError::Overflow)? as f32;
                pyramid.push(
                    tensor_from_f32(
                        &backend,
                        &[1, 1, 1, 1],
                        &[value],
                        DType::F32,
                        DeviceId::CPU,
                        &context,
                    )
                    .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?,
                );
            }
            sub_pyramids.push(pyramid);
        }
        let source = sub_pyramids.clone();
        let source_bytes = source
            .iter()
            .flat_map(|pyramid| pyramid.iter())
            .map(Tensor::contiguous_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let features = compose_film_feature_pyramid(&backend, sub_pyramids, 3, &context)?;
        assert_eq!(features.len(), 4);
        let expected = [
            vec![0.0_f32],
            vec![10.0_f32, 1.0],
            vec![20.0_f32, 11.0, 2.0],
            vec![30.0_f32, 21.0, 12.0],
        ];
        for (index, expected_values) in expected.iter().enumerate() {
            let feature = features
                .get(index)
                .ok_or(FrameInterpolationError::StateMismatch)?;
            assert_eq!(
                feature.descriptor().shape(),
                &[
                    1,
                    u64::try_from(expected_values.len())
                        .map_err(|_| FrameInterpolationError::Overflow)?,
                    1,
                    1
                ]
            );
            for (linear, expected_value) in expected_values.iter().enumerate() {
                let actual = match DType::F32.decode_scalar(feature.linear_element_bytes(
                    u64::try_from(linear).map_err(|_| FrameInterpolationError::Overflow)?,
                )?)? {
                    DecodedScalar::Real(value) => value as f32,
                    _ => return Err(FrameInterpolationError::StateMismatch),
                };
                assert_eq!(actual, *expected_value);
            }
        }
        assert_eq!(features[0].storage_id(), source[0][0].storage_id());
        for (index, feature) in features.iter().enumerate().skip(1) {
            assert!(
                source[index]
                    .iter()
                    .all(|tensor| tensor.storage_id() != feature.storage_id())
            );
        }
        for (tensor, expected_bytes) in source
            .iter()
            .flat_map(|pyramid| pyramid.iter())
            .zip(source_bytes)
        {
            assert_eq!(tensor.contiguous_bytes()?, expected_bytes);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let (constrained_backend, constrained_authority) = CpuWorkspaceAuthority::create_backend(1)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let constrained_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: constrained_authority
                .authorize_workspace(1)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        assert!(matches!(
            compose_film_feature_pyramid(
                &constrained_backend,
                source.clone(),
                3,
                &constrained_context,
            ),
            Err(FrameInterpolationError::ResourceExhausted(_))
        ));
        assert_eq!(constrained_context.scratch.in_use_bytes(), 0);

        let rife = NativeFrameInterpolationModel::reduced_rife_test_fixture(&backend, &context)?;
        assert!(matches!(
            rife.film_feature_pyramid(&backend, &source[0][0], &context),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            compose_film_feature_pyramid(&backend, source, 3, &cancelled_context,),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_subtree_executes_two_convolutions_and_conditional_pooling()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let mut weights = BTreeMap::new();
        let mut input_channels = 1_usize;
        for level in 0..4 {
            let output_channels = 1_usize
                .checked_shl(u32::try_from(level).map_err(|_| FrameInterpolationError::Overflow)?)
                .ok_or(FrameInterpolationError::Overflow)?;
            let prefix = format!("extract.extract_sublevels.convs.{level}");
            for (convolution, channels, scale) in [
                (0_usize, input_channels, 2.0_f32),
                (1_usize, output_channels, 3.0_f32),
            ] {
                let count = output_channels
                    .checked_mul(channels)
                    .and_then(|value| value.checked_mul(9))
                    .ok_or(FrameInterpolationError::Overflow)?;
                let mut values = vec![0.0_f32; count];
                for output_channel in 0..output_channels {
                    let index = output_channel
                        .checked_mul(channels)
                        .and_then(|value| value.checked_mul(9))
                        .and_then(|value| value.checked_add(4))
                        .ok_or(FrameInterpolationError::Overflow)?;
                    let value = values
                        .get_mut(index)
                        .ok_or(FrameInterpolationError::StateMismatch)?;
                    *value = scale;
                }
                weights.insert(
                    format!("{prefix}.{convolution}.conv.weight"),
                    tensor_from_f32(
                        &backend,
                        &[
                            u64::try_from(output_channels)
                                .map_err(|_| FrameInterpolationError::Overflow)?,
                            u64::try_from(channels)
                                .map_err(|_| FrameInterpolationError::Overflow)?,
                            3,
                            3,
                        ],
                        &values,
                        DType::F32,
                        DeviceId::CPU,
                        &context,
                    )
                    .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?,
                );
                weights.insert(
                    format!("{prefix}.{convolution}.conv.bias"),
                    tensor_from_f32(
                        &backend,
                        &[u64::try_from(output_channels)
                            .map_err(|_| FrameInterpolationError::Overflow)?],
                        &vec![0.0; output_channels],
                        DType::F32,
                        DeviceId::CPU,
                        &context,
                    )
                    .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?,
                );
            }
            input_channels = output_channels;
        }
        let input = tensor_from_f32(
            &backend,
            &[1, 1, 4, 4],
            &(1_u8..=16).map(f32::from).collect::<Vec<_>>(),
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let input_bytes = input.contiguous_bytes()?;
        let weight_bytes = weights
            .iter()
            .map(|(key, tensor)| Ok((key.clone(), tensor.contiguous_bytes()?)))
            .collect::<Result<BTreeMap<_, _>, FrameInterpolationError>>()?;
        let pyramid =
            film_subtree_features_from_weights(&backend, &input, 3, 1, 4, &weights, &context)?;
        assert_eq!(pyramid.len(), 4);
        assert_eq!(pyramid[0].descriptor().shape(), &[1, 1, 4, 4]);
        assert_eq!(pyramid[1].descriptor().shape(), &[1, 2, 2, 2]);
        assert_eq!(pyramid[2].descriptor().shape(), &[1, 4, 1, 1]);
        assert_eq!(pyramid[3].descriptor().shape(), &[1, 8, 1, 1]);
        for (level, expected) in [6.0_f32, 126.0, 1836.0, 11016.0].into_iter().enumerate() {
            let actual = match DType::F32.decode_scalar(pyramid[level].linear_element_bytes(0)?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert!((actual - expected).abs() <= 1.0e-4);
            assert_ne!(pyramid[level].storage_id(), input.storage_id());
        }
        assert_eq!(input.contiguous_bytes()?, input_bytes);
        for (key, expected) in &weight_bytes {
            assert_eq!(
                weights
                    .get(key)
                    .ok_or(FrameInterpolationError::StateMismatch)?
                    .contiguous_bytes()?,
                *expected
            );
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let constrained_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(63)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        assert!(matches!(
            film_subtree_features_from_weights(
                &backend,
                &input,
                3,
                1,
                4,
                &weights,
                &constrained_context,
            ),
            Err(FrameInterpolationError::ResourceExhausted(_))
        ));
        assert_eq!(constrained_context.scratch.in_use_bytes(), 0);
        assert_eq!(input.contiguous_bytes()?, input_bytes);

        let rife = NativeFrameInterpolationModel::reduced_rife_test_fixture(&backend, &context)?;
        assert!(matches!(
            rife.film_subtree_features(&backend, &input, 3, &context),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            film_subtree_features_from_weights(
                &backend,
                &input,
                3,
                1,
                4,
                &weights,
                &cancelled_context,
            ),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_image_pyramid_repeats_exact_pooling_and_is_failure_atomic()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let values = (1_u8..=16).map(f32::from).collect::<Vec<_>>();
        for dtype in [DType::F16, DType::Bf16, DType::F32] {
            let input = tensor_from_f32(
                &backend,
                &[1, 1, 4, 4],
                &values,
                dtype,
                DeviceId::CPU,
                &context,
            )
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
            let input_bytes = input.contiguous_bytes()?;
            let pyramid =
                film_image_pyramid_with_context_exact_native(&backend, &input, 3, &context)?;
            assert_eq!(pyramid.len(), 3);
            assert_eq!(pyramid[0].storage_id(), input.storage_id());
            assert_eq!(pyramid[0].descriptor().shape(), &[1, 1, 4, 4]);
            assert_eq!(pyramid[1].descriptor().shape(), &[1, 1, 2, 2]);
            assert_eq!(pyramid[2].descriptor().shape(), &[1, 1, 1, 1]);
            assert_ne!(pyramid[1].storage_id(), input.storage_id());
            assert_ne!(pyramid[2].storage_id(), input.storage_id());
            assert_ne!(pyramid[1].storage_id(), pyramid[2].storage_id());
            for level in &pyramid {
                assert_eq!(level.descriptor().dtype(), dtype);
            }
            let expected_levels: [&[f32]; 2] = [&[3.5, 5.5, 11.5, 13.5], &[8.5]];
            for (level, expected) in pyramid.iter().skip(1).zip(expected_levels) {
                for (index, expected) in expected.iter().copied().enumerate() {
                    let index =
                        u64::try_from(index).map_err(|_| FrameInterpolationError::Overflow)?;
                    let actual = match dtype.decode_scalar(level.linear_element_bytes(index)?)? {
                        DecodedScalar::Real(value) => value as f32,
                        _ => return Err(FrameInterpolationError::StateMismatch),
                    };
                    assert!((actual - expected).abs() <= 0.01);
                }
            }
            assert_eq!(input.contiguous_bytes()?, input_bytes);
            assert_eq!(context.scratch.in_use_bytes(), 0);

            let constrained_context = ExecutionContext {
                stream: StreamId::DEFAULT,
                scratch: authority
                    .authorize_workspace(63)
                    .map_err(|_| FrameInterpolationError::Overflow)?,
                rng_phase: None,
                cancellation: &cancellation,
            };
            let constrained_error = film_image_pyramid_with_context_exact_native(
                &backend,
                &input,
                3,
                &constrained_context,
            )
            .expect_err("the constrained FILM image pyramid must fail");
            assert!(
                matches!(
                    constrained_error,
                    FrameInterpolationError::ResourceExhausted(_)
                ),
                "unexpected constrained FILM image-pyramid error: {constrained_error:?}"
            );
            assert_eq!(constrained_context.scratch.in_use_bytes(), 0);
            assert_eq!(input.contiguous_bytes()?, input_bytes);
        }

        let input = tensor_from_f32(
            &backend,
            &[1, 1, 4, 4],
            &values,
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        assert!(matches!(
            film_image_pyramid_with_context_exact_native(&backend, &input, 0, &context),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));
        assert!(matches!(
            film_image_pyramid_with_context_exact_native(&backend, &input, 4, &context),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));
        let production_input = tensor_from_f32(
            &backend,
            &[1, 1, 64, 64],
            &vec![0.0; 64 * 64],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let production_pyramid =
            film_image_pyramid_with_context_exact_native(&backend, &production_input, 7, &context)?;
        assert_eq!(production_pyramid.len(), 7);
        assert_eq!(production_pyramid[6].descriptor().shape(), &[1, 1, 1, 1]);
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            film_image_pyramid_with_context_exact_native(&backend, &input, 3, &cancelled_context,),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_conv_uses_source_padding_activation_and_failure_atomicity()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        for dtype in [DType::F16, DType::Bf16, DType::F32] {
            let input = tensor_from_f32(
                &backend,
                &[1, 1, 2, 2],
                &[1.0, 2.0, 3.0, 4.0],
                dtype,
                DeviceId::CPU,
                &context,
            )
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
            let weight = tensor_from_f32(
                &backend,
                &[1, 1, 2, 2],
                &[1.0; 4],
                dtype,
                DeviceId::CPU,
                &context,
            )
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
            let bias = tensor_from_f32(&backend, &[1], &[0.0], dtype, DeviceId::CPU, &context)
                .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
            let input_bytes = input.contiguous_bytes()?;
            let weight_bytes = weight.contiguous_bytes()?;
            let bias_bytes = bias.contiguous_bytes()?;
            let output = film_conv_2d_with_context_exact_native(
                &backend, &input, &weight, &bias, false, &context,
            )?;
            assert_eq!(output.descriptor().shape(), &[1, 1, 2, 2]);
            assert_eq!(output.descriptor().dtype(), dtype);
            assert_ne!(output.storage_id(), input.storage_id());
            assert_ne!(output.storage_id(), weight.storage_id());
            assert_ne!(output.storage_id(), bias.storage_id());
            for (index, expected) in [10.0_f32, 6.0, 7.0, 4.0].into_iter().enumerate() {
                let index = u64::try_from(index).map_err(|_| FrameInterpolationError::Overflow)?;
                let actual = match dtype.decode_scalar(output.linear_element_bytes(index)?)? {
                    DecodedScalar::Real(value) => value as f32,
                    _ => return Err(FrameInterpolationError::StateMismatch),
                };
                assert!((actual - expected).abs() <= 0.01);
            }
            assert_eq!(input.contiguous_bytes()?, input_bytes);
            assert_eq!(weight.contiguous_bytes()?, weight_bytes);
            assert_eq!(bias.contiguous_bytes()?, bias_bytes);
            assert_eq!(context.scratch.in_use_bytes(), 0);

            let constrained_context = ExecutionContext {
                stream: StreamId::DEFAULT,
                scratch: authority
                    .authorize_workspace(35)
                    .map_err(|_| FrameInterpolationError::Overflow)?,
                rng_phase: None,
                cancellation: &cancellation,
            };
            let constrained_error = film_conv_2d_with_context_exact_native(
                &backend,
                &input,
                &weight,
                &bias,
                false,
                &constrained_context,
            )
            .expect_err("the constrained FILM convolution must fail");
            assert!(
                matches!(
                    constrained_error,
                    FrameInterpolationError::ResourceExhausted(_)
                ),
                "unexpected constrained FILM convolution error: {constrained_error:?}"
            );
            assert_eq!(constrained_context.scratch.in_use_bytes(), 0);
            assert_eq!(input.contiguous_bytes()?, input_bytes);
            assert_eq!(weight.contiguous_bytes()?, weight_bytes);
            assert_eq!(bias.contiguous_bytes()?, bias_bytes);
        }

        let input = tensor_from_f32(
            &backend,
            &[1, 1, 2, 2],
            &[-2.0, 1.0, 3.0, 4.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let weight = tensor_from_f32(
            &backend,
            &[1, 1, 3, 3],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let bias = tensor_from_f32(&backend, &[1], &[0.0], DType::F32, DeviceId::CPU, &context)
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let output = film_conv_2d_with_context_exact_native(
            &backend, &input, &weight, &bias, true, &context,
        )?;
        let first = match DType::F32.decode_scalar(output.linear_element_bytes(0)?)? {
            DecodedScalar::Real(value) => value as f32,
            _ => return Err(FrameInterpolationError::StateMismatch),
        };
        assert!((first + 0.4).abs() <= 1.0e-6);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            film_conv_2d_with_context_exact_native(
                &backend,
                &input,
                &weight,
                &bias,
                true,
                &cancelled_context,
            ),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_warp_uses_pixel_centers_and_is_failure_atomic() -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        for dtype in [DType::F16, DType::Bf16, DType::F32] {
            let input = tensor_from_f32(
                &backend,
                &[1, 1, 2, 2],
                &[1.0, 3.0, 5.0, 7.0],
                dtype,
                DeviceId::CPU,
                &context,
            )
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
            let flow = tensor_from_f32(
                &backend,
                &[1, 2, 2, 2],
                &[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                dtype,
                DeviceId::CPU,
                &context,
            )
            .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
            let input_bytes = input.contiguous_bytes()?;
            let flow_bytes = flow.contiguous_bytes()?;
            let output = film_warp_with_context_exact_native(&backend, &input, &flow, &context)?;
            assert_eq!(output.descriptor().shape(), &[1, 1, 2, 2]);
            assert_eq!(output.descriptor().dtype(), dtype);
            assert_eq!(output.descriptor().device(), DeviceId::CPU);
            assert_eq!(output.descriptor().stream(), StreamId::DEFAULT);
            assert_ne!(output.storage_id(), input.storage_id());
            assert_ne!(output.storage_id(), flow.storage_id());
            for (index, expected) in [2.0_f32, 3.0, 6.0, 7.0].into_iter().enumerate() {
                let index = u64::try_from(index).map_err(|_| FrameInterpolationError::Overflow)?;
                let actual = match dtype.decode_scalar(output.linear_element_bytes(index)?)? {
                    DecodedScalar::Real(value) => value as f32,
                    _ => return Err(FrameInterpolationError::StateMismatch),
                };
                assert!((actual - expected).abs() <= 0.01);
            }
            assert_eq!(input.contiguous_bytes()?, input_bytes);
            assert_eq!(flow.contiguous_bytes()?, flow_bytes);
            assert_eq!(context.scratch.in_use_bytes(), 0);

            let constrained_context = ExecutionContext {
                stream: StreamId::DEFAULT,
                scratch: authority
                    .authorize_workspace(47)
                    .map_err(|_| FrameInterpolationError::Overflow)?,
                rng_phase: None,
                cancellation: &cancellation,
            };
            assert!(matches!(
                film_warp_with_context_exact_native(&backend, &input, &flow, &constrained_context),
                Err(FrameInterpolationError::ResourceExhausted(_))
            ));
            assert_eq!(constrained_context.scratch.in_use_bytes(), 0);
            assert_eq!(input.contiguous_bytes()?, input_bytes);
            assert_eq!(flow.contiguous_bytes()?, flow_bytes);
        }

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        let input = tensor_from_f32(
            &backend,
            &[1, 1, 1, 1],
            &[1.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let flow = tensor_from_f32(
            &backend,
            &[1, 2, 1, 1],
            &[0.0, 0.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        assert!(matches!(
            film_warp_with_context_exact_native(&backend, &input, &flow, &cancelled_context),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn invocation_plans_order_midpoints_align_rife_and_persist_oom_downgrades()
    -> Result<(), FrameInterpolationError> {
        let cancellation = CancellationToken::default();
        let rife = FrameInterpolationProfile::Rife {
            head_channels: 4,
            block_channels: [192, 128, 96, 64, 32],
        };
        let plan = FrameInterpolationInvocationPlan::checked(&rife, 3, 4, 65, 66, &cancellation)?;
        assert!(!plan.is_bypass());
        assert_eq!(plan.output_frame_count(), 9);
        assert_eq!(plan.output_element_count(), 115_830);
        assert_eq!(plan.total_interpolation_steps(), 6);
        assert_eq!((plan.padded_height(), plan.padded_width()), (128, 128));
        assert_eq!((plan.padding_bottom(), plan.padding_right()), (63, 62));
        assert_eq!(plan.timesteps(), &[0.25, 0.5, 0.75]);

        let mut fallback = FrameInterpolationFallbackState::for_plan(&plan, true)?;
        assert!(fallback.multi_timestep_enabled());
        assert_eq!(fallback.single_timestep_batch(), 3);
        fallback.record_multi_timestep_oom();
        assert!(!fallback.multi_timestep_enabled());
        fallback.record_single_timestep_oom()?;
        assert_eq!(fallback.single_timestep_batch(), 1);
        assert!(matches!(
            fallback.record_single_timestep_oom(),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));

        let film = FrameInterpolationInvocationPlan::checked(
            &FrameInterpolationProfile::Film,
            5,
            2,
            65,
            66,
            &cancellation,
        )?;
        assert_eq!((film.padded_height(), film.padded_width()), (65, 66));
        assert_eq!(film.output_frame_count(), 9);
        assert_eq!(film.timesteps(), &[0.5]);
        Ok(())
    }

    #[test]
    fn frame_interpolation_sequence_fallback_preserves_rife_halving_and_terminal_exhaustion()
    -> Result<(), FrameInterpolationError> {
        let cancellation = CancellationToken::default();
        let profile = FrameInterpolationProfile::Rife {
            head_channels: 2,
            block_channels: [2; 5],
        };
        let plan =
            FrameInterpolationInvocationPlan::checked(&profile, 3, 5, 64, 64, &cancellation)?;
        let mut fallback = FrameInterpolationFallbackState::for_plan(&plan, false)?;
        let mut attempts = Vec::new();
        let outputs = execute_frame_interpolation_sequence_fallback(
            &profile,
            plan.timesteps(),
            &mut fallback,
            &cancellation,
            |attempt| {
                let FrameInterpolationSequenceAttempt::SingleTimestepBatch(timesteps) = attempt
                else {
                    return Err(FrameInterpolationError::StateMismatch);
                };
                attempts.push(timesteps.to_vec());
                match timesteps.len() {
                    4 => Err(FrameInterpolationError::ResourceExhausted(
                        "batch four".into(),
                    )),
                    2 => Err(FrameInterpolationError::ResourceExhausted(
                        "batch two".into(),
                    )),
                    1 => Ok(timesteps.to_vec()),
                    _ => Err(FrameInterpolationError::StateMismatch),
                }
            },
        )?;
        assert_eq!(fallback.single_timestep_batch(), 1);
        assert_eq!(
            attempts.iter().map(Vec::len).collect::<Vec<_>>(),
            [4, 2, 1, 1, 1, 1]
        );
        assert_eq!(outputs.concat(), plan.timesteps());

        attempts.clear();
        let second_pair = execute_frame_interpolation_sequence_fallback(
            &profile,
            plan.timesteps(),
            &mut fallback,
            &cancellation,
            |attempt| {
                let FrameInterpolationSequenceAttempt::SingleTimestepBatch(timesteps) = attempt
                else {
                    return Err(FrameInterpolationError::StateMismatch);
                };
                attempts.push(timesteps.to_vec());
                Ok(timesteps.to_vec())
            },
        )?;
        assert_eq!(
            attempts.iter().map(Vec::len).collect::<Vec<_>>(),
            [1, 1, 1, 1]
        );
        assert_eq!(second_pair.concat(), plan.timesteps());

        let terminal_plan =
            FrameInterpolationInvocationPlan::checked(&profile, 2, 2, 64, 64, &cancellation)?;
        let mut terminal = FrameInterpolationFallbackState::for_plan(&terminal_plan, false)?;
        let error = execute_frame_interpolation_sequence_fallback::<()>(
            &profile,
            terminal_plan.timesteps(),
            &mut terminal,
            &cancellation,
            |_| {
                Err(FrameInterpolationError::ResourceExhausted(
                    "terminal".into(),
                ))
            },
        )
        .expect_err("batch-one exhaustion must remain typed");
        assert!(matches!(
            error,
            FrameInterpolationError::ResourceExhausted(message) if message == "terminal"
        ));
        Ok(())
    }

    #[test]
    fn frame_interpolation_sequence_fallback_preserves_film_scalarization_failure()
    -> Result<(), FrameInterpolationError> {
        let cancellation = CancellationToken::default();
        let profile = FrameInterpolationProfile::Film;
        let scalar_plan =
            FrameInterpolationInvocationPlan::checked(&profile, 2, 2, 8, 8, &cancellation)?;
        let mut scalar_fallback = FrameInterpolationFallbackState::for_plan(&scalar_plan, true)?;
        let mut scalar_attempts = Vec::new();
        let scalar = execute_frame_interpolation_sequence_fallback(
            &profile,
            scalar_plan.timesteps(),
            &mut scalar_fallback,
            &cancellation,
            |attempt| {
                scalar_attempts.push(match attempt {
                    FrameInterpolationSequenceAttempt::MultiTimestep(_) => "multi",
                    FrameInterpolationSequenceAttempt::SingleTimestepBatch(_) => "single",
                });
                if scalar_attempts.len() == 1 {
                    Err(FrameInterpolationError::ResourceExhausted("multi".into()))
                } else {
                    Ok(())
                }
            },
        )?;
        assert_eq!(scalar, [()]);
        assert_eq!(scalar_attempts, ["multi", "single"]);
        assert!(!scalar_fallback.multi_timestep_enabled());

        let batched_plan =
            FrameInterpolationInvocationPlan::checked(&profile, 2, 4, 8, 8, &cancellation)?;
        let mut batched_fallback = FrameInterpolationFallbackState::for_plan(&batched_plan, true)?;
        let mut calls = 0_usize;
        let error = execute_frame_interpolation_sequence_fallback::<()>(
            &profile,
            batched_plan.timesteps(),
            &mut batched_fallback,
            &cancellation,
            |_| {
                calls = calls.saturating_add(1);
                Err(FrameInterpolationError::ResourceExhausted("multi".into()))
            },
        )
        .expect_err("the pinned FILM fallback cannot scalarize more than one timestep");
        assert_eq!(calls, 1);
        assert!(matches!(error, FrameInterpolationError::Execution(_)));
        Ok(())
    }

    #[test]
    fn frame_interpolation_sequence_fallback_prioritizes_cancellation_and_never_retries_execution()
    -> Result<(), FrameInterpolationError> {
        let profile = FrameInterpolationProfile::Rife {
            head_channels: 2,
            block_channels: [2; 5],
        };
        let cancellation = CancellationToken::default();
        let plan =
            FrameInterpolationInvocationPlan::checked(&profile, 2, 4, 64, 64, &cancellation)?;
        let mut cancelled_fallback = FrameInterpolationFallbackState::for_plan(&plan, false)?;
        let mut cancelled_calls = 0_usize;
        let error = execute_frame_interpolation_sequence_fallback::<()>(
            &profile,
            plan.timesteps(),
            &mut cancelled_fallback,
            &cancellation,
            |_| {
                cancelled_calls = cancelled_calls.saturating_add(1);
                cancellation.cancel();
                Err(FrameInterpolationError::ResourceExhausted(
                    "cancelled".into(),
                ))
            },
        )
        .expect_err("cancellation must dominate a retryable exhaustion");
        assert_eq!(cancelled_calls, 1);
        assert!(matches!(error, FrameInterpolationError::Cancelled));

        let ordinary_cancellation = CancellationToken::default();
        let ordinary_plan = FrameInterpolationInvocationPlan::checked(
            &profile,
            2,
            4,
            64,
            64,
            &ordinary_cancellation,
        )?;
        let mut ordinary_fallback =
            FrameInterpolationFallbackState::for_plan(&ordinary_plan, false)?;
        let mut ordinary_calls = 0_usize;
        let error = execute_frame_interpolation_sequence_fallback::<()>(
            &profile,
            ordinary_plan.timesteps(),
            &mut ordinary_fallback,
            &ordinary_cancellation,
            |_| {
                ordinary_calls = ordinary_calls.saturating_add(1);
                Err(FrameInterpolationError::Execution("ordinary".into()))
            },
        )
        .expect_err("ordinary execution errors must not retry");
        assert_eq!(ordinary_calls, 1);
        assert!(matches!(error, FrameInterpolationError::Execution(_)));
        Ok(())
    }

    #[test]
    fn rife_timestep_batches_preserve_source_order() -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let tensor =
            rife_timestep_tensor(&backend, &[0.25, 0.5, 0.75], 3, 2, 2, DType::F32, &context)?;
        assert_eq!(tensor.descriptor().shape(), &[3, 1, 2, 2]);
        for (linear, expected) in [
            0.25_f32, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75,
        ]
        .into_iter()
        .enumerate()
        {
            let linear = u64::try_from(linear).map_err(|_| FrameInterpolationError::Overflow)?;
            let actual = match DType::F32.decode_scalar(tensor.linear_element_bytes(linear)?)? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(FrameInterpolationError::StateMismatch),
            };
            assert_eq!(actual, expected);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn invocation_plans_bypass_and_reject_unbounded_or_unreflectable_requests()
    -> Result<(), FrameInterpolationError> {
        let cancellation = CancellationToken::default();
        let rife = FrameInterpolationProfile::Rife {
            head_channels: 2,
            block_channels: [2; 5],
        };
        let bypass = FrameInterpolationInvocationPlan::checked(&rife, 1, 2, 64, 64, &cancellation)?;
        assert!(bypass.is_bypass());
        assert_eq!(bypass.output_frame_count(), 1);
        assert!(bypass.timesteps().is_empty());
        assert_eq!(
            FrameInterpolationFallbackState::for_plan(&bypass, true)?,
            FrameInterpolationFallbackState {
                multi_timestep_enabled: false,
                single_timestep_batch: 0,
            }
        );
        assert!(matches!(
            FrameInterpolationInvocationPlan::checked(&rife, 2, 17, 64, 64, &cancellation),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));
        assert!(matches!(
            FrameInterpolationInvocationPlan::checked(&rife, 2, 2, 1, 1, &cancellation),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));
        assert!(matches!(
            FrameInterpolationInvocationPlan::checked(
                &FrameInterpolationProfile::Film,
                2,
                16,
                u64::MAX,
                u64::MAX,
                &cancellation
            ),
            Err(FrameInterpolationError::Overflow)
        ));
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            FrameInterpolationInvocationPlan::checked(&rife, 2, 2, 64, 64, &cancelled),
            Err(FrameInterpolationError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn manifests_close_exact_source_tensor_counts() -> Result<(), FrameInterpolationError> {
        assert_eq!(film_manifest()?.len(), 82);
        assert_eq!(rife_manifest(4, [192, 128, 96, 64, 32])?.len(), 158);
        Ok(())
    }

    #[test]
    fn rife_normalization_is_sequential_filtered_and_collision_safe()
    -> Result<(), FrameInterpolationError> {
        use crate::native_ops::tensor_from_f32;
        use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext};

        let manifest = rife_manifest(4, [2, 2, 2, 2, 2])?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let scalar = tensor_from_f32(&backend, &[1], &[1.0], DType::F32, DeviceId::CPU, &context)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let mut state = BTreeMap::new();
        for key in manifest.keys() {
            state.insert(format!("module.flownet.{key}"), scalar.clone());
        }
        let head = tensor_from_f32(
            &backend,
            &[16, 4, 1, 1],
            &vec![1.0; 64],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|_| FrameInterpolationError::Overflow)?;
        state.insert("module.flownet.encode.cnn3.weight".into(), head);
        for index in 0..5 {
            let detector = tensor_from_f32(
                &backend,
                &[2, 1, 1, 1],
                &[1.0, 1.0],
                DType::F32,
                DeviceId::CPU,
                &context,
            )
            .map_err(|_| FrameInterpolationError::Overflow)?;
            state.insert(
                format!("module.flownet.blocks.{index}.conv0.1.0.weight"),
                detector,
            );
        }
        state.insert("teacher.discarded".into(), scalar);
        let (profile, normalized) = normalize_and_detect(state, &cancellation)?;
        assert_eq!(
            profile,
            FrameInterpolationProfile::Rife {
                head_channels: 4,
                block_channels: [2; 5]
            }
        );
        assert!(!normalized.keys().any(|key| key.starts_with("teacher.")));
        Ok(())
    }

    #[test]
    fn reduced_rife_checkpoint_is_strict_content_bound_and_alias_aware()
    -> Result<(), FrameInterpolationError> {
        use crate::native_ops::tensor_from_f32;
        use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(4 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let manifest = rife_manifest(2, [2; 5])?;
        let mut state = BTreeMap::new();
        let mut tensors_by_shape: BTreeMap<Vec<u64>, Tensor> = BTreeMap::new();
        for (key, shape) in &manifest {
            let tensor = if let Some(tensor) = tensors_by_shape.get(shape) {
                tensor.clone()
            } else {
                let count = shape.iter().try_fold(1_usize, |total, dimension| {
                    total.checked_mul(usize::try_from(*dimension).ok()?)
                });
                let count = count.ok_or(FrameInterpolationError::Overflow)?;
                let tensor = tensor_from_f32(
                    &backend,
                    shape,
                    &vec![0.25; count],
                    DType::F32,
                    DeviceId::CPU,
                    &context,
                )
                .map_err(|_| FrameInterpolationError::Overflow)?;
                tensors_by_shape.insert(shape.clone(), tensor.clone());
                tensor
            };
            state.insert(key.clone(), tensor);
        }
        let model = NativeFrameInterpolationModel::from_checkpoint(
            "a".repeat(64),
            state.clone(),
            &cancellation,
        )?;
        assert_eq!(
            model.profile(),
            &FrameInterpolationProfile::Rife {
                head_channels: 2,
                block_channels: [2; 5],
            }
        );
        assert_eq!(model.weight_count(), 158);
        assert_eq!(model.profile().alignment(), 64);
        assert_eq!(model.semantic_state_digest_sha256().len(), 64);
        assert!(model.resident_tensor_allocations()?.len() < 158);
        assert!(
            model.resident_owned_bytes()?
                > std::mem::size_of::<NativeFrameInterpolationModel>() as u64
        );
        assert!(model.resident_bytes()? > model.resident_owned_bytes()?);
        let payload =
            crate::NativeModelPayload::frame_interpolation(std::sync::Arc::new(model.clone()))
                .map_err(|_| FrameInterpolationError::StateMismatch)?;
        assert_eq!(
            payload.identity().role(),
            crate::NativeModelResourceRole::FrameInterpolation
        );
        assert_eq!(
            payload.identity().identifier(),
            "native-frame-interpolation-rife"
        );
        assert_eq!(
            payload.identity().format(),
            "zed-native-frame-interpolation-v1"
        );
        assert!(payload.frame_interpolation_resource().is_some());
        assert!(payload.model().is_none());
        assert!(payload.clip().is_none());
        assert!(payload.vae().is_none());
        let parts = payload
            .resident_parts()
            .map_err(|_| FrameInterpolationError::StateMismatch)?;
        assert!(parts.backing_allocations().iter().any(|allocation| {
            allocation.kind() == crate::NativeModelBackingKind::NativeFrameInterpolationModel
        }));
        payload
            .validate()
            .map_err(|_| FrameInterpolationError::StateMismatch)?;

        let mut missing = state.clone();
        missing.remove("encode.cnn0.weight");
        assert!(matches!(
            NativeFrameInterpolationModel::from_checkpoint("a".repeat(64), missing, &cancellation),
            Err(FrameInterpolationError::StateMismatch)
        ));

        let changed_shape = vec![16, 3, 3, 3];
        let changed = tensor_from_f32(
            &backend,
            &changed_shape,
            &vec![0.5; 16 * 3 * 3 * 3],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|_| FrameInterpolationError::Overflow)?;
        state.insert("encode.cnn0.weight".into(), changed);
        let changed =
            NativeFrameInterpolationModel::from_checkpoint("a".repeat(64), state, &cancellation)?;
        assert_ne!(
            model.semantic_state_digest_sha256(),
            changed.semantic_state_digest_sha256()
        );

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            NativeFrameInterpolationModel::from_checkpoint(
                "a".repeat(64),
                BTreeMap::new(),
                &cancelled
            ),
            Err(FrameInterpolationError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn reduced_rife_forward_executes_the_retained_graph_and_is_failure_atomic()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(512 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(256 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let model = NativeFrameInterpolationModel::reduced_rife_test_fixture(&backend, &context)?;
        let element_count = 3 * 64 * 64;
        let first = tensor_from_f32(
            &backend,
            &[1, 3, 64, 64],
            &vec![0.0; element_count],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let second = tensor_from_f32(
            &backend,
            &[1, 3, 64, 64],
            &vec![1.0; element_count],
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let digest = model.semantic_state_digest_sha256().to_owned();
        let allocations = model.resident_tensor_allocations()?;
        let output = model.interpolate_rife_pair(&backend, &first, &second, 0.5, &context)?;
        assert_eq!(output.descriptor().shape(), &[1, 3, 64, 64]);
        assert_eq!(output.descriptor().dtype(), DType::F32);
        assert_ne!(output.storage_id(), first.storage_id());
        assert_ne!(output.storage_id(), second.storage_id());
        for encoded in output.contiguous_bytes()?.chunks_exact(4) {
            let encoded: [u8; 4] = encoded
                .try_into()
                .map_err(|_| FrameInterpolationError::StateMismatch)?;
            assert_eq!(f32::from_ne_bytes(encoded), 0.5);
        }
        assert_eq!(model.semantic_state_digest_sha256(), digest);
        assert_eq!(model.resident_tensor_allocations()?, allocations);
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let constrained_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1024)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        assert!(matches!(
            model.interpolate_rife_pair(&backend, &first, &second, 0.5, &constrained_context),
            Err(FrameInterpolationError::ResourceExhausted(_))
        ));
        assert_eq!(constrained_context.scratch.in_use_bytes(), 0);
        assert_eq!(model.semantic_state_digest_sha256(), digest);
        assert_eq!(model.resident_tensor_allocations()?, allocations);
        assert!(matches!(
            model.interpolate_rife_pair(&backend, &first, &second, f32::NAN, &context),
            Err(FrameInterpolationError::InvalidInvocation(_))
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(256 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            model.interpolate_rife_pair(&backend, &first, &second, 0.5, &cancelled_context),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        assert_eq!(model.semantic_state_digest_sha256(), digest);
        assert_eq!(model.resident_tensor_allocations()?, allocations);
        Ok(())
    }

    #[test]
    fn reduced_rife_sequence_preserves_endpoints_padding_order_and_owner_state()
    -> Result<(), FrameInterpolationError> {
        use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(512 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(256 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let model = NativeFrameInterpolationModel::reduced_rife_test_fixture(&backend, &context)?;
        let frame_elements = 64 * 63 * 3;
        let mut values = vec![0.0; frame_elements];
        values.extend(std::iter::repeat_n(1.0, frame_elements));
        let images = tensor_from_f32(
            &backend,
            &[2, 64, 63, 3],
            &values,
            DType::F32,
            DeviceId::CPU,
            &context,
        )
        .map_err(|error| FrameInterpolationError::Execution(error.to_string()))?;
        let digest = model.semantic_state_digest_sha256().to_owned();
        let allocations = model.resident_tensor_allocations()?;
        let output = model.interpolate_rife_sequence(&backend, &images, 2, &context)?;
        assert_eq!(output.descriptor().shape(), &[3, 64, 63, 3]);
        assert_eq!(output.descriptor().dtype(), DType::F32);
        assert_ne!(output.storage_id(), images.storage_id());
        let bytes = output.contiguous_bytes()?;
        for (frame, expected) in [0.0_f32, 0.5, 1.0].into_iter().enumerate() {
            let start = frame
                .checked_mul(frame_elements)
                .and_then(|offset| offset.checked_mul(4))
                .ok_or(FrameInterpolationError::Overflow)?;
            let end = start
                .checked_add(
                    frame_elements
                        .checked_mul(4)
                        .ok_or(FrameInterpolationError::Overflow)?,
                )
                .ok_or(FrameInterpolationError::Overflow)?;
            let frame_bytes = bytes
                .get(start..end)
                .ok_or(FrameInterpolationError::StateMismatch)?;
            for encoded in frame_bytes.chunks_exact(4) {
                let encoded: [u8; 4] = encoded
                    .try_into()
                    .map_err(|_| FrameInterpolationError::StateMismatch)?;
                assert_eq!(f32::from_ne_bytes(encoded), expected);
            }
        }
        assert_eq!(model.semantic_state_digest_sha256(), digest);
        assert_eq!(model.resident_tensor_allocations()?, allocations);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        let expected_input_bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(images.contiguous_bytes()?, expected_input_bytes);

        let batched_output = model.interpolate_rife_sequence(&backend, &images, 4, &context)?;
        assert_eq!(batched_output.descriptor().shape(), &[5, 64, 63, 3]);
        let batched_bytes = batched_output.contiguous_bytes()?;
        for (frame, expected) in [0.0_f32, 0.5, 0.5, 0.5, 1.0].into_iter().enumerate() {
            let start = frame
                .checked_mul(frame_elements)
                .and_then(|offset| offset.checked_mul(4))
                .ok_or(FrameInterpolationError::Overflow)?;
            let end = start
                .checked_add(
                    frame_elements
                        .checked_mul(4)
                        .ok_or(FrameInterpolationError::Overflow)?,
                )
                .ok_or(FrameInterpolationError::Overflow)?;
            let frame_bytes = batched_bytes
                .get(start..end)
                .ok_or(FrameInterpolationError::StateMismatch)?;
            for encoded in frame_bytes.chunks_exact(4) {
                let encoded: [u8; 4] = encoded
                    .try_into()
                    .map_err(|_| FrameInterpolationError::StateMismatch)?;
                assert_eq!(f32::from_ne_bytes(encoded), expected);
            }
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(images.contiguous_bytes()?, expected_input_bytes);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(256 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            model.interpolate_rife_sequence(&backend, &images, 2, &cancelled_context),
            Err(FrameInterpolationError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn film_precedence_and_rife_normalization_collisions_fail_closed()
    -> Result<(), FrameInterpolationError> {
        use crate::native_ops::tensor_from_f32;
        use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext};

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)
            .map_err(|_| FrameInterpolationError::Overflow)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority
                .authorize_workspace(1 << 20)
                .map_err(|_| FrameInterpolationError::Overflow)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let scalar = tensor_from_f32(&backend, &[1], &[1.0], DType::F32, DeviceId::CPU, &context)
            .map_err(|_| FrameInterpolationError::Overflow)?;

        let mut film = BTreeMap::new();
        film.insert(FILM_MARKER.into(), scalar.clone());
        film.insert("encode.cnn3.weight".into(), scalar.clone());
        let (profile, _) = normalize_and_detect(film, &cancellation)?;
        assert_eq!(profile, FrameInterpolationProfile::Film);

        let mut collision = BTreeMap::new();
        collision.insert("module.flownet.block0.duplicate".into(), scalar.clone());
        collision.insert("module.flownet.blocks.0.duplicate".into(), scalar);
        assert!(matches!(
            normalize_and_detect(collision, &cancellation),
            Err(FrameInterpolationError::KeyCollision(key)) if key == "blocks.0.duplicate"
        ));
        Ok(())
    }
}
