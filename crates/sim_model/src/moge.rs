use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, ImageTensor, Layout,
    StorageId, StreamId, Tensor, TensorError,
    generated_activation_normalization_functional_01::{
        FunctionalError, group_norm_with_context_exact_native, relu_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_16::{
        ElementwiseRuntimePartSixteenError, add_method_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_21::{
        ElementwiseRuntimePartTwentyOneError, exp_with_context_exact_native,
    },
    generated_linear_algebra_01::LinearAlgebraPartOneError,
    generated_neural_network_functional_01::{
        NeuralNetworkFunctionalError, linear_with_context_exact_native,
    },
    generated_shape_layout_transform_03::{
        FunctionalPadMode, ShapeLayoutTransformPartThreeError,
        functional_pad_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        ConvolutionConfiguration, InterpolateConfiguration, InterpolateMode,
        SpatialFunctionalKernelError, conv_2d_tensor_with_context_exact_native,
        conv_transpose_2d_tensor_with_context_exact_native,
        interpolate_tensor_with_context_exact_native,
    },
    generated_tensor_creation_01::{
        TensorCreationPartOneError, linspace_with_context_exact_native,
    },
};
use std::{collections::BTreeMap, mem};
use thiserror::Error;

use crate::dino2::{
    NativeDino2Backbone, NativeDino2Configuration, NativeDino2Error, NativeDino2Feature,
};

pub const NODES_MOGE_SOURCE_SHA256: &str =
    "160f48e4b6bb1e34617f9de78380758ef5d04caa7c8ea7768ce31b98fccee265";
pub const MOGE_MODEL_SOURCE_SHA256: &str =
    "68ee3db2ff7eb96c8a90234b182559129c5c094374128e5ba99baed7caf0cb3c";
pub const MOGE_MODULES_SOURCE_SHA256: &str =
    "3655abdce2de058624bd4ea2f02757ab42bd811d9afab0fef824fe480afdb2a6";
pub const MOGE_GEOMETRY_SOURCE_SHA256: &str =
    "db8e2da75f13028a98067c517d6495fd9818b2878b44599beaa281ff7fde397c";
pub const MOGE_DINO2_SOURCE_SHA256: &str =
    "1dec8c1d6104c268e593cea20302d925f637266edce2a6e4dfa142af8a00d579";

const MAX_STATE_TENSORS: usize = 16_384;
const MAX_STATE_KEY_BYTES: usize = 1_024;
const DIGEST_CHUNK_BYTES: usize = 64 * 1_024;
const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGE_STANDARD_DEVIATION: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeMogeVersion {
    V1,
    V2,
}

#[derive(Clone, Debug)]
pub struct NativeMogeCheckpoint {
    pub artifact_sha256: String,
    pub ordered_state: Vec<(String, Tensor)>,
    pub memory_budget_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct NativeMogeNestedCheckpoint {
    pub artifact_sha256: String,
    pub model: Vec<(String, Tensor)>,
    pub model_config: serde_json::Map<String, serde_json::Value>,
    pub memory_budget_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct NativeMogeInvocation<'a> {
    pub image: &'a ImageTensor,
    pub resolution_level: u8,
    pub fov_x_degrees: Option<f32>,
    pub force_projection: bool,
    pub apply_mask: bool,
    pub apply_metric_scale: bool,
}

#[derive(Clone, Debug)]
pub struct NativeMogeGeometry {
    image: ImageTensor,
    points: Option<Tensor>,
    depth: Option<Tensor>,
    intrinsics: Option<Tensor>,
    mask: Option<Tensor>,
    normal: Option<Tensor>,
}

impl NativeMogeGeometry {
    pub fn image(&self) -> &ImageTensor {
        &self.image
    }

    pub fn points(&self) -> Option<&Tensor> {
        self.points.as_ref()
    }

    pub fn depth(&self) -> Option<&Tensor> {
        self.depth.as_ref()
    }

    pub fn intrinsics(&self) -> Option<&Tensor> {
        self.intrinsics.as_ref()
    }

    pub fn mask(&self) -> Option<&Tensor> {
        self.mask.as_ref()
    }

    pub fn normal(&self) -> Option<&Tensor> {
        self.normal.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeMogeConfiguration {
    version: NativeMogeVersion,
    dino_prefix: &'static str,
    hidden: usize,
    layer_count: usize,
    attention_heads: usize,
    patch: usize,
    image: usize,
    output_layers: [usize; 4],
    has_normal: bool,
    source_exact: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StateSpecification {
    key: String,
    shape: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct NativeMogeResource {
    configuration: NativeMogeConfiguration,
    artifact_sha256: String,
    source_state: BTreeMap<String, Tensor>,
    execution_state: BTreeMap<String, Tensor>,
    source_dtype: DType,
    stream: StreamId,
    memory_budget_bytes: u64,
    resident_bytes: u64,
    semantic_digest_sha256: String,
}

#[derive(Debug, Error)]
pub enum NativeMogeError {
    #[error("MoGe execution was cancelled")]
    Cancelled,
    #[error("MoGe architecture is unsupported or ambiguous")]
    UnsupportedArchitecture,
    #[error("MoGe checkpoint is invalid: {0}")]
    InvalidCheckpoint(String),
    #[error("MoGe state contains duplicate or colliding key {0}")]
    DuplicateStateKey(String),
    #[error("MoGe state is missing key {0}")]
    MissingState(String),
    #[error("MoGe state is unexpected: {0}")]
    UnexpectedState(String),
    #[error("MoGe state {key} expected {expected:?}, got {actual:?} {actual_dtype:?}")]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        actual_dtype: DType,
    },
    #[error("MoGe retained semantic state changed")]
    SemanticStateChanged,
    #[error("MoGe image is invalid: {0}")]
    InvalidImage(String),
    #[error("MoGe geometry controls are invalid")]
    InvalidControls,
    #[error("MoGe geometry projection is invalid")]
    InvalidGeometryProjection,
    #[error("MoGe shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("MoGe allocation failed")]
    Allocation,
    #[error("MoGe memory requirement {required} exceeds budget {budget}")]
    OutOfMemory { required: u64, budget: u64 },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error(transparent)]
    Elementwise(#[from] ElementwiseRuntimePartSixteenError),
    #[error(transparent)]
    Exponential(#[from] ElementwiseRuntimePartTwentyOneError),
    #[error(transparent)]
    Angle(#[from] sim_tensor::generated_elementwise_or_runtime_operation_04::ElementwiseRuntimePartFourError),
    #[error(transparent)]
    Tangent(#[from] sim_tensor::generated_elementwise_or_runtime_operation_15::ElementwiseRuntimePartFifteenError),
    #[error(transparent)]
    Neural(#[from] NeuralNetworkFunctionalError),
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraPartOneError),
    #[error(transparent)]
    Reduction(#[from] sim_tensor::generated_reduction_01::ReductionPartOneError),
    #[error(transparent)]
    ShapeLayout(#[from] ShapeLayoutTransformPartThreeError),
    #[error(transparent)]
    Spatial(#[from] SpatialFunctionalKernelError),
    #[error(transparent)]
    TensorCreation(#[from] TensorCreationPartOneError),
    #[error("MoGe DINOv2 owner failed: {0}")]
    Dino(String),
}

impl From<sim_types::CancellationError> for NativeMogeError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<NativeDino2Error> for NativeMogeError {
    fn from(error: NativeDino2Error) -> Self {
        match error {
            NativeDino2Error::Cancelled => Self::Cancelled,
            NativeDino2Error::Tensor(error) => Self::Tensor(error),
            error => Self::Dino(error.to_string()),
        }
    }
}

impl NativeMogeResource {
    fn execution_tensor(&self, key: &str) -> Result<&Tensor, NativeMogeError> {
        self.execution_state
            .get(key)
            .ok_or_else(|| NativeMogeError::MissingState(key.to_owned()))
    }

    pub fn from_checkpoint(
        backend: &CpuBackend,
        checkpoint: NativeMogeCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeMogeError> {
        Self::checked(backend, checkpoint, true, context)
    }

    pub fn from_nested_checkpoint(
        backend: &CpuBackend,
        checkpoint: NativeMogeNestedCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeMogeError> {
        context.check()?;
        Self::checked(
            backend,
            canonical_flat_checkpoint(checkpoint),
            true,
            context,
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn from_reduced_fixture(
        backend: &CpuBackend,
        checkpoint: NativeMogeCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeMogeError> {
        Self::checked(backend, checkpoint, false, context)
    }

    fn checked(
        backend: &CpuBackend,
        checkpoint: NativeMogeCheckpoint,
        source_exact: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeMogeError> {
        context.check()?;
        validate_sha256(&checkpoint.artifact_sha256)?;
        if checkpoint.ordered_state.is_empty()
            || checkpoint.ordered_state.len() > MAX_STATE_TENSORS
            || checkpoint.memory_budget_bytes == 0
        {
            return Err(NativeMogeError::InvalidCheckpoint(
                "state cardinality or memory budget is invalid".to_owned(),
            ));
        }
        raw_ordered_state_preflight(
            &checkpoint.artifact_sha256,
            &checkpoint.ordered_state,
            checkpoint.memory_budget_bytes,
            context,
        )?;
        let source_state = normalize_state(backend, checkpoint.ordered_state, context)?;
        let configuration = detect_configuration(&source_state, source_exact)?;
        let backbone = dino_backbone(configuration)?;
        let specifications = state_manifest(configuration, &source_state)?;
        let source_dtype = validate_source_state(
            &source_state,
            &specifications,
            context.stream,
            context.cancellation,
        )?;
        let required = projected_resident_preflight(
            &checkpoint.artifact_sha256,
            &source_state,
            context.cancellation,
        )?;
        if required > checkpoint.memory_budget_bytes {
            return Err(NativeMogeError::OutOfMemory {
                required,
                budget: checkpoint.memory_budget_bytes,
            });
        }
        let mut execution_state = BTreeMap::new();
        for (index, (key, tensor)) in source_state.iter().enumerate() {
            if index.is_multiple_of(16) {
                context.check()?;
            }
            let projected = if backbone.owns_state_key(key) {
                backbone.project_state_tensor(backend, key, tensor, context)?
            } else {
                cast_to_with_context_exact_native(
                    backend,
                    tensor,
                    DType::F32,
                    DeviceId::CPU,
                    false,
                    true,
                    context,
                )?
            };
            validate_finite_tensor(key, &projected, context.cancellation)?;
            execution_state.insert(key.clone(), projected);
        }
        let semantic_digest_sha256 = semantic_digest(
            configuration,
            &checkpoint.artifact_sha256,
            source_dtype,
            &source_state,
            &execution_state,
            context.cancellation,
        )?;
        let resident_bytes =
            resident_tensor_bytes([&source_state, &execution_state], context.cancellation)?
                .checked_add(resident_owned_bytes(
                    &checkpoint.artifact_sha256,
                    &semantic_digest_sha256,
                    &source_state,
                    &execution_state,
                )?)
                .ok_or(NativeMogeError::ShapeOverflow)?;
        if resident_bytes > checkpoint.memory_budget_bytes {
            return Err(NativeMogeError::OutOfMemory {
                required: resident_bytes,
                budget: checkpoint.memory_budget_bytes,
            });
        }
        context.check()?;
        Ok(Self {
            configuration,
            artifact_sha256: checkpoint.artifact_sha256,
            source_state,
            execution_state,
            source_dtype,
            stream: context.stream,
            memory_budget_bytes: checkpoint.memory_budget_bytes,
            resident_bytes,
            semantic_digest_sha256,
        })
    }

    pub const fn version(&self) -> NativeMogeVersion {
        self.configuration.version
    }

    pub fn identifier(&self) -> &'static str {
        match self.configuration.version {
            NativeMogeVersion::V1 => "moge-v1",
            NativeMogeVersion::V2 => "moge-v2",
        }
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub const fn source_dtype(&self) -> DType {
        self.source_dtype
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub const fn is_source_exact_profile(&self) -> bool {
        self.configuration.source_exact
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, NativeMogeError> {
        resident_owned_bytes(
            &self.artifact_sha256,
            &self.semantic_digest_sha256,
            &self.source_state,
            &self.execution_state,
        )
    }

    pub fn resident_tensor_allocations(&self) -> Result<Vec<(StorageId, u64)>, NativeMogeError> {
        resident_tensor_allocations(
            [&self.source_state, &self.execution_state],
            &CancellationToken::default(),
        )
    }

    pub fn reconstruct_checkpoint(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NativeMogeCheckpoint, NativeMogeError> {
        self.validate(cancellation)?;
        let mut ordered_state = Vec::new();
        ordered_state
            .try_reserve_exact(self.source_state.len())
            .map_err(|_| NativeMogeError::Allocation)?;
        for (index, (key, tensor)) in self.source_state.iter().enumerate() {
            if index.is_multiple_of(32) {
                cancellation.check()?;
            }
            ordered_state.push((key.clone(), tensor.clone()));
        }
        Ok(NativeMogeCheckpoint {
            artifact_sha256: self.artifact_sha256.clone(),
            ordered_state,
            memory_budget_bytes: self.memory_budget_bytes,
        })
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), NativeMogeError> {
        cancellation.check()?;
        let specifications = state_manifest(self.configuration, &self.source_state)?;
        let dtype = validate_source_state(
            &self.source_state,
            &specifications,
            self.stream,
            cancellation,
        )?;
        if dtype != self.source_dtype || self.execution_state.len() != self.source_state.len() {
            return Err(NativeMogeError::SemanticStateChanged);
        }
        for (key, tensor) in &self.execution_state {
            if tensor.descriptor().dtype() != DType::F32
                || tensor.descriptor().device() != DeviceId::CPU
                || tensor.descriptor().stream() != self.stream
            {
                return Err(NativeMogeError::SemanticStateChanged);
            }
            validate_finite_tensor(key, tensor, cancellation)?;
        }
        let digest = semantic_digest(
            self.configuration,
            &self.artifact_sha256,
            self.source_dtype,
            &self.source_state,
            &self.execution_state,
            cancellation,
        )?;
        let resident =
            resident_tensor_bytes([&self.source_state, &self.execution_state], cancellation)?
                .checked_add(self.resident_owned_bytes()?)
                .ok_or(NativeMogeError::ShapeOverflow)?;
        if digest != self.semantic_digest_sha256 || resident != self.resident_bytes {
            return Err(NativeMogeError::SemanticStateChanged);
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn execute(
        &self,
        backend: &CpuBackend,
        invocation: NativeMogeInvocation<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeMogeGeometry, NativeMogeError> {
        context.check()?;
        self.validate(context.cancellation)?;
        if invocation.resolution_level > 9
            || invocation
                .fov_x_degrees
                .is_some_and(|value| !value.is_finite() || !(0.0..=170.0).contains(&value))
        {
            return Err(NativeMogeError::InvalidControls);
        }
        let (batch, height, width, channels) = invocation.image.dimensions()?;
        if invocation.image.tensor().descriptor().stream() != context.stream {
            return Err(NativeMogeError::InvalidImage(
                "image is on a foreign execution stream".to_owned(),
            ));
        }
        if batch == 0 || height == 0 || width == 0 || !matches!(channels, 3 | 4) {
            return Err(NativeMogeError::InvalidImage(
                "expected non-empty RGB or RGBA BHWC input".to_owned(),
            ));
        }
        let (batch, height, width, channels) = (
            usize_from(batch)?,
            usize_from(height)?,
            usize_from(width)?,
            usize_from(channels)?,
        );
        let values = invocation.image.as_f32_slice()?;
        for (index, pixel) in values.chunks_exact(channels).enumerate() {
            if index.is_multiple_of(16_384) {
                context.check()?;
            }
            if pixel[..3]
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            {
                return Err(NativeMogeError::InvalidImage(
                    "RGB values must be finite and in [0, 1]".to_owned(),
                ));
            }
        }
        preflight_execution_memory(self, batch, height, width, invocation.resolution_level)?;
        let (target_height, target_width, head_height, head_width) = preprocessing_dimensions(
            self.configuration,
            height,
            width,
            invocation.resolution_level,
        )?;
        let prepared = prepare_image(
            backend,
            self,
            values,
            batch,
            height,
            width,
            channels,
            target_height,
            target_width,
            head_height,
            head_width,
            context,
        )?;
        let execution = dino_backbone(self.configuration)?.bind_parent_preflighted(
            &self.execution_state,
            self.memory_budget_bytes,
            self.resident_bytes,
        );
        let features = execution.get_intermediate_layers(backend, &prepared, batch, context)?;
        let class_feature = features
            .last()
            .cloned()
            .ok_or(NativeMogeError::InvalidGeometryProjection)?;
        let raw = execute_moge_head(
            backend,
            self,
            &features,
            &class_feature,
            batch,
            target_height / self.configuration.patch,
            target_width / self.configuration.patch,
            head_height,
            head_width,
            height,
            width,
            context,
        )?;
        project_geometry(
            backend,
            invocation,
            self.configuration.version,
            raw,
            batch,
            height,
            width,
            context,
        )
    }
}

fn canonical_flat_checkpoint(checkpoint: NativeMogeNestedCheckpoint) -> NativeMogeCheckpoint {
    let NativeMogeNestedCheckpoint {
        artifact_sha256,
        model,
        model_config: _,
        memory_budget_bytes,
    } = checkpoint;
    NativeMogeCheckpoint {
        artifact_sha256,
        ordered_state: model,
        memory_budget_bytes,
    }
}

fn raw_ordered_state_preflight(
    artifact_sha256: &str,
    ordered_state: &[(String, Tensor)],
    budget: u64,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeMogeError> {
    let mut raw_bytes = 0_u64;
    let mut largest_fused_conversion = 0_u64;
    for (index, (key, tensor)) in ordered_state.iter().enumerate() {
        if index.is_multiple_of(16) {
            context.check()?;
        }
        validate_state_key(key)?;
        let descriptor = tensor.descriptor();
        if descriptor.device() != DeviceId::CPU || descriptor.stream() != context.stream {
            return Err(NativeMogeError::InvalidCheckpoint(format!(
                "state {key} has invalid placement"
            )));
        }
        let element_bytes = match descriptor.dtype() {
            DType::F16 | DType::Bf16 => 2_u64,
            DType::F32 => 4_u64,
            dtype => {
                return Err(NativeMogeError::StateShape {
                    key: key.clone(),
                    expected: descriptor.shape().to_vec(),
                    actual: descriptor.shape().to_vec(),
                    actual_dtype: dtype,
                });
            }
        };
        let elements = descriptor.element_count()?;
        raw_bytes = raw_bytes
            .checked_add(
                elements
                    .checked_mul(element_bytes)
                    .ok_or(NativeMogeError::ShapeOverflow)?,
            )
            .ok_or(NativeMogeError::ShapeOverflow)?;
        if key.ends_with(".attn.qkv.weight") || key.ends_with(".attn.qkv.bias") {
            largest_fused_conversion = largest_fused_conversion.max(
                elements
                    .checked_mul(4)
                    .ok_or(NativeMogeError::ShapeOverflow)?,
            );
        }
    }
    let required = raw_bytes
        .checked_mul(2)
        .and_then(|value| value.checked_add(largest_fused_conversion))
        .and_then(|value| value.checked_add(u64::try_from(artifact_sha256.len()).ok()?))
        .ok_or(NativeMogeError::ShapeOverflow)?;
    if required > budget {
        return Err(NativeMogeError::OutOfMemory { required, budget });
    }
    context.check()?;
    Ok(())
}

#[derive(Clone)]
struct RawMogeOutput {
    points: Vec<f32>,
    mask: Vec<f32>,
    normal: Option<Vec<f32>>,
    metric_scale: Option<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn execute_moge_head(
    backend: &CpuBackend,
    resource: &NativeMogeResource,
    features: &[NativeDino2Feature],
    class_feature: &NativeDino2Feature,
    batch: usize,
    patch_height: usize,
    patch_width: usize,
    head_height: usize,
    head_width: usize,
    output_height: usize,
    output_width: usize,
    context: &ExecutionContext<'_>,
) -> Result<RawMogeOutput, NativeMogeError> {
    if features.len() != 4
        || features.iter().any(|feature| {
            feature.patches != patch_height * patch_width
                || feature.channels != resource.configuration.hidden
        })
    {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    match resource.configuration.version {
        NativeMogeVersion::V1 => execute_head_v1(
            backend,
            resource,
            features,
            batch,
            patch_height,
            patch_width,
            head_height,
            head_width,
            output_height,
            output_width,
            context,
        ),
        NativeMogeVersion::V2 => execute_head_v2(
            backend,
            resource,
            features,
            class_feature,
            batch,
            patch_height,
            patch_width,
            output_height,
            output_width,
            context,
        ),
    }
}

fn shape4(tensor: &Tensor) -> Result<(usize, usize, usize, usize), NativeMogeError> {
    match tensor.descriptor().shape() {
        [batch, channels, height, width] => Ok((
            usize_from(*batch)?,
            usize_from(*channels)?,
            usize_from(*height)?,
            usize_from(*width)?,
        )),
        _ => Err(NativeMogeError::InvalidGeometryProjection),
    }
}

fn feature_nchw(
    backend: &CpuBackend,
    feature: &NativeDino2Feature,
    batch: usize,
    height: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    if feature.patches != height * width
        || feature.patch_values.len() != batch * feature.patches * feature.channels
    {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    let mut values = filled_f32(feature.patch_values.len(), 0.0)?;
    for batch_index in 0..batch {
        for patch in 0..feature.patches {
            if patch.is_multiple_of(4_096) {
                context.check()?;
            }
            for channel in 0..feature.channels {
                values[((batch_index * feature.channels + channel) * height) * width + patch] =
                    feature.patch_values
                        [(batch_index * feature.patches + patch) * feature.channels + channel];
            }
        }
    }
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64_from(batch)?,
            u64_from(feature.channels)?,
            u64_from(height)?,
            u64_from(width)?,
        ],
        &values,
        DType::F32,
        DeviceId::CPU,
        context,
    )?)
}

fn convolution(
    backend: &CpuBackend,
    resource: &NativeMogeResource,
    input: &Tensor,
    prefix: &str,
    transposed: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    let weight = resource.execution_tensor(&format!("{prefix}.weight"))?;
    let bias = resource.execution_state.get(&format!("{prefix}.bias"));
    let shape = weight.descriptor().shape();
    if shape.len() != 4 || shape[2] != shape[3] {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    let kernel = usize_from(shape[2])?;
    let input = if !transposed && kernel > 1 {
        functional_pad_with_context_exact_native(
            backend,
            input,
            &[1, 1, 1, 1],
            FunctionalPadMode::Replicate,
            None,
            context,
        )?
    } else {
        input.clone()
    };
    let configuration = ConvolutionConfiguration {
        stride: vec![if transposed { 2 } else { 1 }; 2],
        padding: vec![0, 0],
        dilation: vec![1, 1],
        groups: 1,
        output_padding: vec![0, 0],
    };
    Ok(if transposed {
        conv_transpose_2d_tensor_with_context_exact_native(
            backend,
            &input,
            weight,
            bias,
            &configuration,
            context,
        )?
    } else {
        conv_2d_tensor_with_context_exact_native(
            backend,
            &input,
            weight,
            bias,
            &configuration,
            context,
        )?
    })
}

fn resize_nchw(
    backend: &CpuBackend,
    input: &Tensor,
    height: usize,
    width: usize,
    antialias: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    Ok(interpolate_tensor_with_context_exact_native(
        backend,
        input,
        &InterpolateConfiguration {
            output_size: Some(vec![height, width]),
            scale_factor: None,
            mode: InterpolateMode::Bilinear,
            align_corners: Some(false),
            recompute_scale_factor: None,
            antialias,
        },
        context,
    )?)
}

fn relu_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    let values = tensor_to_f32_with_context_exact_native(backend, input, context)?;
    let values = relu_with_context_exact_native(backend, &values, DeviceId::CPU, context)?;
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        input.descriptor().shape(),
        &values,
        DType::F32,
        DeviceId::CPU,
        context,
    )?)
}

fn add_tensor(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    Ok(add_method_with_context_exact_native(
        backend,
        left,
        ElementwiseOperand::Tensor(right),
        1.0,
        context,
    )?)
}

fn concat_view_plane(
    backend: &CpuBackend,
    input: &Tensor,
    aspect: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    let (batch, channels, height, width) = shape4(input)?;
    let source = tensor_to_f32_with_context_exact_native(backend, input, context)?;
    let mut output = filled_f32(batch * (channels + 2) * height * width, 0.0)?;
    let diagonal = (1.0 + aspect * aspect).sqrt();
    let span_x = aspect / diagonal;
    let span_y = 1.0 / diagonal;
    let horizontal = tensor_to_f32_with_context_exact_native(
        backend,
        &linspace_with_context_exact_native(
            backend,
            -span_x * (width - 1) as f64 / width as f64,
            span_x * (width - 1) as f64 / width as f64,
            u64_from(width)?,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            context,
        )?,
        context,
    )?;
    let vertical = tensor_to_f32_with_context_exact_native(
        backend,
        &linspace_with_context_exact_native(
            backend,
            -span_y * (height - 1) as f64 / height as f64,
            span_y * (height - 1) as f64 / height as f64,
            u64_from(height)?,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            context,
        )?,
        context,
    )?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            let source_start = (batch_index * channels + channel) * height * width;
            let output_start = (batch_index * (channels + 2) + channel) * height * width;
            output[output_start..output_start + height * width]
                .copy_from_slice(&source[source_start..source_start + height * width]);
        }
        for y in 0..height {
            for x in 0..width {
                let pixel = y * width + x;
                let u = horizontal[x];
                let v = vertical[y];
                output[(batch_index * (channels + 2) + channels) * height * width + pixel] = u;
                output[(batch_index * (channels + 2) + channels + 1) * height * width + pixel] = v;
            }
        }
    }
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64_from(batch)?,
            u64_from(channels + 2)?,
            u64_from(height)?,
            u64_from(width)?,
        ],
        &output,
        DType::F32,
        DeviceId::CPU,
        context,
    )?)
}

fn view_plane_tensor(
    backend: &CpuBackend,
    batch: usize,
    height: usize,
    width: usize,
    aspect: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    let empty = tensor_from_f32_with_context_exact_native(
        backend,
        &[u64_from(batch)?, 1, u64_from(height)?, u64_from(width)?],
        &filled_f32(batch * height * width, 0.0)?,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let with_grid = concat_view_plane(backend, &empty, aspect, context)?;
    let values = tensor_to_f32_with_context_exact_native(backend, &with_grid, context)?;
    let mut grid = filled_f32(batch * 2 * height * width, 0.0)?;
    for batch_index in 0..batch {
        for channel in 0..2 {
            let source = (batch_index * 3 + channel + 1) * height * width;
            let destination = (batch_index * 2 + channel) * height * width;
            grid[destination..destination + height * width]
                .copy_from_slice(&values[source..source + height * width]);
        }
    }
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        &[u64_from(batch)?, 2, u64_from(height)?, u64_from(width)?],
        &grid,
        DType::F32,
        DeviceId::CPU,
        context,
    )?)
}

fn group_norm_tensor(
    backend: &CpuBackend,
    resource: &NativeMogeResource,
    input: &Tensor,
    prefix: &str,
    groups: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    let (batch, channels, height, width) = shape4(input)?;
    let values = tensor_to_f32_with_context_exact_native(backend, input, context)?;
    let weight = tensor_to_f32_with_context_exact_native(
        backend,
        resource.execution_tensor(&format!("{prefix}.weight"))?,
        context,
    )?;
    let bias = tensor_to_f32_with_context_exact_native(
        backend,
        resource.execution_tensor(&format!("{prefix}.bias"))?,
        context,
    )?;
    let output = group_norm_with_context_exact_native(
        backend,
        &values,
        &[batch, channels, height, width],
        groups,
        Some(&weight),
        Some(&bias),
        1.0e-5,
        DeviceId::CPU,
        context,
    )?;
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        input.descriptor().shape(),
        &output,
        DType::F32,
        DeviceId::CPU,
        context,
    )?)
}

fn residual_conv_block(
    backend: &CpuBackend,
    resource: &NativeMogeResource,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    let channels = shape4(input)?.1;
    let mut output = if resource
        .execution_state
        .contains_key(&format!("{prefix}.layers.0.weight"))
    {
        group_norm_tensor(
            backend,
            resource,
            input,
            &format!("{prefix}.layers.0"),
            1,
            context,
        )?
    } else {
        input.clone()
    };
    output = relu_tensor(backend, &output, context)?;
    output = convolution(
        backend,
        resource,
        &output,
        &format!("{prefix}.layers.2"),
        false,
        context,
    )?;
    if resource
        .execution_state
        .contains_key(&format!("{prefix}.layers.3.weight"))
    {
        let hidden = shape4(&output)?.1;
        output = group_norm_tensor(
            backend,
            resource,
            &output,
            &format!("{prefix}.layers.3"),
            (hidden / 32).max(1),
            context,
        )?;
    }
    output = relu_tensor(backend, &output, context)?;
    output = convolution(
        backend,
        resource,
        &output,
        &format!("{prefix}.layers.5"),
        false,
        context,
    )?;
    if shape4(&output)?.1 != channels {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    add_tensor(backend, &output, input, context)
}

fn conv_stack(
    backend: &CpuBackend,
    resource: &NativeMogeResource,
    prefix: &str,
    inputs: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, NativeMogeError> {
    let mut outputs = Vec::new();
    let mut current: Option<Tensor> = None;
    for (level, input) in inputs.iter().enumerate() {
        context.check()?;
        let input_prefix = format!("{prefix}.input_blocks.{level}");
        let feature = if resource
            .execution_state
            .contains_key(&format!("{input_prefix}.weight"))
        {
            convolution(backend, resource, input, &input_prefix, false, context)?
        } else {
            input.clone()
        };
        current = Some(match current.take() {
            None => feature,
            Some(current) => add_tensor(backend, &current, &feature, context)?,
        });
        let mut value = current
            .take()
            .ok_or(NativeMogeError::InvalidGeometryProjection)?;
        let mut residual = 0;
        while resource.execution_state.contains_key(&format!(
            "{prefix}.res_blocks.{level}.{residual}.layers.2.weight"
        )) {
            value = residual_conv_block(
                backend,
                resource,
                &value,
                &format!("{prefix}.res_blocks.{level}.{residual}"),
                context,
            )?;
            residual += 1;
        }
        let output_prefix = format!("{prefix}.output_blocks.{level}");
        outputs.push(
            if resource
                .execution_state
                .contains_key(&format!("{output_prefix}.weight"))
            {
                convolution(backend, resource, &value, &output_prefix, false, context)?
            } else {
                value.clone()
            },
        );
        if level + 1 < inputs.len() {
            let resampler = format!("{prefix}.resamplers.{level}");
            value = if resource
                .execution_state
                .contains_key(&format!("{resampler}.0.weight"))
            {
                convolution(
                    backend,
                    resource,
                    &value,
                    &format!("{resampler}.0"),
                    true,
                    context,
                )?
            } else {
                let (_, _, height, width) = shape4(&value)?;
                resize_nchw(backend, &value, height * 2, width * 2, false, context)?
            };
            value = convolution(
                backend,
                resource,
                &value,
                &format!("{resampler}.1"),
                false,
                context,
            )?;
        }
        current = Some(value);
    }
    Ok(outputs)
}

fn raw_from_nchw(
    backend: &CpuBackend,
    points: &Tensor,
    mask: &Tensor,
    normal: Option<&Tensor>,
    sigmoid_mask: bool,
    metric_scale: Option<Vec<f32>>,
    context: &ExecutionContext<'_>,
) -> Result<RawMogeOutput, NativeMogeError> {
    let (batch, point_channels, height, width) = shape4(points)?;
    let (mask_batch, mask_channels, mask_height, mask_width) = shape4(mask)?;
    if point_channels != 3
        || (mask_batch, mask_channels, mask_height, mask_width) != (batch, 1, height, width)
    {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    let point_values = tensor_to_f32_with_context_exact_native(backend, points, context)?;
    let mask_values = tensor_to_f32_with_context_exact_native(backend, mask, context)?;
    let point_exponentials = tensor_to_f32_with_context_exact_native(
        backend,
        &exp_with_context_exact_native(backend, points, context)?,
        context,
    )?;
    let mask_exponentials = if sigmoid_mask {
        let negative_mask = tensor_from_f32_with_context_exact_native(
            backend,
            mask.descriptor().shape(),
            &mask_values.iter().map(|value| -*value).collect::<Vec<_>>(),
            DType::F32,
            DeviceId::CPU,
            context,
        )?;
        Some(tensor_to_f32_with_context_exact_native(
            backend,
            &exp_with_context_exact_native(backend, &negative_mask, context)?,
            context,
        )?)
    } else {
        None
    };
    let normal_values = normal
        .map(|tensor| tensor_to_f32_with_context_exact_native(backend, tensor, context))
        .transpose()?;
    let pixels = batch * height * width;
    let mut output_points = filled_f32(pixels * 3, 0.0)?;
    let mut output_mask = filled_f32(pixels, 0.0)?;
    let mut output_normal = normal_values
        .as_ref()
        .map(|_| filled_f32(pixels * 3, 0.0))
        .transpose()?;
    for batch_index in 0..batch {
        for pixel in 0..height * width {
            if pixel.is_multiple_of(4_096) {
                context.check()?;
            }
            let output = batch_index * height * width + pixel;
            let z = point_exponentials[(batch_index * 3 + 2) * height * width + pixel];
            if !z.is_finite() {
                return Err(NativeMogeError::InvalidGeometryProjection);
            }
            output_points[output * 3] =
                point_values[(batch_index * 3) * height * width + pixel] * z;
            output_points[output * 3 + 1] =
                point_values[(batch_index * 3 + 1) * height * width + pixel] * z;
            output_points[output * 3 + 2] = z;
            let mask = mask_values[output];
            output_mask[output] = mask_exponentials
                .as_ref()
                .map_or(mask, |exponentials| 1.0 / (1.0 + exponentials[output]));
            if output_points[output * 3..output * 3 + 3]
                .iter()
                .any(|value| !value.is_finite())
                || !output_mask[output].is_finite()
            {
                return Err(NativeMogeError::InvalidGeometryProjection);
            }
            if let (Some(source), Some(destination)) = (&normal_values, output_normal.as_mut()) {
                let x = source[(batch_index * 3) * height * width + pixel];
                let y = source[(batch_index * 3 + 1) * height * width + pixel];
                let z = source[(batch_index * 3 + 2) * height * width + pixel];
                let length = x.mul_add(x, y.mul_add(y, z * z)).sqrt().max(1.0e-12);
                destination[output * 3] = x / length;
                destination[output * 3 + 1] = y / length;
                destination[output * 3 + 2] = z / length;
                if destination[output * 3..output * 3 + 3]
                    .iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(NativeMogeError::InvalidGeometryProjection);
                }
            }
        }
    }
    Ok(RawMogeOutput {
        points: output_points,
        mask: output_mask,
        normal: output_normal,
        metric_scale,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_head_v1(
    backend: &CpuBackend,
    resource: &NativeMogeResource,
    features: &[NativeDino2Feature],
    batch: usize,
    patch_height: usize,
    patch_width: usize,
    head_height: usize,
    head_width: usize,
    output_height: usize,
    output_width: usize,
    context: &ExecutionContext<'_>,
) -> Result<RawMogeOutput, NativeMogeError> {
    let mut projected: Option<Tensor> = None;
    for (index, feature) in features.iter().enumerate() {
        let feature = feature_nchw(backend, feature, batch, patch_height, patch_width, context)?;
        let feature = convolution(
            backend,
            resource,
            &feature,
            &format!("head.projects.{index}"),
            false,
            context,
        )?;
        projected = Some(match projected.take() {
            None => feature,
            Some(projected) => add_tensor(backend, &projected, &feature, context)?,
        });
    }
    let mut value = projected.ok_or(NativeMogeError::InvalidGeometryProjection)?;
    let aspect = head_width as f64 / head_height as f64;
    let mut level = 0;
    while resource
        .execution_state
        .contains_key(&format!("head.upsample_blocks.{level}.0.0.weight"))
    {
        value = concat_view_plane(backend, &value, aspect, context)?;
        value = convolution(
            backend,
            resource,
            &value,
            &format!("head.upsample_blocks.{level}.0.0"),
            true,
            context,
        )?;
        value = convolution(
            backend,
            resource,
            &value,
            &format!("head.upsample_blocks.{level}.0.1"),
            false,
            context,
        )?;
        let mut residual = 1;
        while resource.execution_state.contains_key(&format!(
            "head.upsample_blocks.{level}.{residual}.layers.2.weight"
        )) {
            value = residual_conv_block(
                backend,
                resource,
                &value,
                &format!("head.upsample_blocks.{level}.{residual}"),
                context,
            )?;
            residual += 1;
        }
        level += 1;
    }
    value = resize_nchw(backend, &value, head_height, head_width, false, context)?;
    value = concat_view_plane(backend, &value, aspect, context)?;
    let output = |index: usize| -> Result<Tensor, NativeMogeError> {
        let value = convolution(
            backend,
            resource,
            &value,
            &format!("head.output_block.{index}.0"),
            false,
            context,
        )?;
        let value = relu_tensor(backend, &value, context)?;
        convolution(
            backend,
            resource,
            &value,
            &format!("head.output_block.{index}.2"),
            false,
            context,
        )
    };
    let points = resize_nchw(
        backend,
        &output(0)?,
        output_height,
        output_width,
        false,
        context,
    )?;
    let mask = resize_nchw(
        backend,
        &output(1)?,
        output_height,
        output_width,
        false,
        context,
    )?;
    raw_from_nchw(backend, &points, &mask, None, false, None, context)
}

#[allow(clippy::too_many_arguments)]
fn execute_head_v2(
    backend: &CpuBackend,
    resource: &NativeMogeResource,
    features: &[NativeDino2Feature],
    class_feature: &NativeDino2Feature,
    batch: usize,
    patch_height: usize,
    patch_width: usize,
    output_height: usize,
    output_width: usize,
    context: &ExecutionContext<'_>,
) -> Result<RawMogeOutput, NativeMogeError> {
    let mut projected: Option<Tensor> = None;
    for (index, feature) in features.iter().enumerate() {
        let feature = feature_nchw(backend, feature, batch, patch_height, patch_width, context)?;
        let feature = convolution(
            backend,
            resource,
            &feature,
            &format!("encoder.output_projections.{index}"),
            false,
            context,
        )?;
        projected = Some(match projected.take() {
            None => feature,
            Some(projected) => add_tensor(backend, &projected, &feature, context)?,
        });
    }
    let aspect = output_width as f64 / output_height as f64;
    let mut levels = Vec::new();
    levels.push(concat_view_plane(
        backend,
        &projected.ok_or(NativeMogeError::InvalidGeometryProjection)?,
        aspect,
        context,
    )?);
    for level in 1..5 {
        let height = patch_height << level;
        let width = patch_width << level;
        levels.push(view_plane_tensor(
            backend, batch, height, width, aspect, context,
        )?);
    }
    let neck = conv_stack(backend, resource, "neck", &levels, context)?;
    let points = conv_stack(backend, resource, "points_head", &neck, context)?
        .pop()
        .ok_or(NativeMogeError::InvalidGeometryProjection)?;
    let mask = conv_stack(backend, resource, "mask_head", &neck, context)?
        .pop()
        .ok_or(NativeMogeError::InvalidGeometryProjection)?;
    let normal = resource
        .configuration
        .has_normal
        .then(|| {
            conv_stack(backend, resource, "normal_head", &neck, context)?
                .pop()
                .ok_or(NativeMogeError::InvalidGeometryProjection)
        })
        .transpose()?;
    let points = resize_nchw(
        backend,
        &points,
        output_height,
        output_width,
        false,
        context,
    )?;
    let mask = resize_nchw(backend, &mask, output_height, output_width, false, context)?;
    let normal = normal
        .map(|normal| {
            resize_nchw(
                backend,
                &normal,
                output_height,
                output_width,
                false,
                context,
            )
        })
        .transpose()?;
    if class_feature.camera_values.len() != batch * resource.configuration.hidden {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    let mut scale = class_feature.camera_values.clone();
    let mut input_channels = resource.configuration.hidden;
    let mut layer = 0_usize;
    loop {
        let weight_key = format!("scale_head.{layer}.weight");
        let bias_key = format!("scale_head.{layer}.bias");
        let weight = resource.execution_tensor(&weight_key)?;
        let weight_shape = weight
            .descriptor()
            .shape()
            .iter()
            .map(|value| usize_from(*value))
            .collect::<Result<Vec<_>, _>>()?;
        scale = linear_with_context_exact_native(
            backend,
            &scale,
            &[batch, input_channels],
            &tensor_to_f32_with_context_exact_native(backend, weight, context)?,
            &weight_shape,
            Some(&tensor_to_f32_with_context_exact_native(
                backend,
                resource.execution_tensor(&bias_key)?,
                context,
            )?),
            DeviceId::CPU,
            context,
        )?
        .values;
        let next_layer = layer.checked_add(2).ok_or(NativeMogeError::ShapeOverflow)?;
        if !resource
            .execution_state
            .contains_key(&format!("scale_head.{next_layer}.weight"))
        {
            break;
        }
        scale = relu_with_context_exact_native(backend, &scale, DeviceId::CPU, context)?;
        input_channels = scale
            .len()
            .checked_div(batch)
            .filter(|channels| *channels > 0)
            .ok_or(NativeMogeError::InvalidGeometryProjection)?;
        layer = next_layer;
    }
    if scale.len() != batch {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    let scale_tensor = tensor_from_f32_with_context_exact_native(
        backend,
        &[u64_from(batch)?],
        &scale,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let scale = tensor_to_f32_with_context_exact_native(
        backend,
        &exp_with_context_exact_native(backend, &scale_tensor, context)?,
        context,
    )?;
    if scale
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    raw_from_nchw(
        backend,
        &points,
        &mask,
        normal.as_ref(),
        true,
        Some(scale),
        context,
    )
}

fn project_geometry(
    backend: &CpuBackend,
    invocation: NativeMogeInvocation<'_>,
    invocation_version: NativeMogeVersion,
    mut raw: RawMogeOutput,
    batch: usize,
    height: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<NativeMogeGeometry, NativeMogeError> {
    let pixels_per_batch = height
        .checked_mul(width)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let pixels = batch
        .checked_mul(pixels_per_batch)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let point_values = pixels
        .checked_mul(3)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    if raw.points.len() != point_values
        || raw.mask.len() != pixels
        || raw
            .normal
            .as_ref()
            .is_some_and(|normal| normal.len() != point_values)
        || raw
            .metric_scale
            .as_ref()
            .is_some_and(|scale| scale.len() != batch)
        || raw.points.iter().any(|value| !value.is_finite())
        || raw.mask.iter().any(|value| !value.is_finite())
        || raw
            .normal
            .as_ref()
            .is_some_and(|normal| normal.iter().any(|value| !value.is_finite()))
        || raw.metric_scale.as_ref().is_some_and(|scale| {
            scale
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        })
    {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    let mut depth = filled_f32(pixels, 0.0)?;
    let mut intrinsics = filled_f32(
        batch.checked_mul(9).ok_or(NativeMogeError::ShapeOverflow)?,
        0.0,
    )?;
    let mut mask = filled_f32(depth.len(), 0.0)?;
    for batch_index in 0..batch {
        context.check()?;
        let point_range =
            batch_index * pixels_per_batch * 3..(batch_index + 1) * pixels_per_batch * 3;
        let mask_range = batch_index * pixels_per_batch..(batch_index + 1) * pixels_per_batch;
        let (mut focal, mut shift) = recover_focal_shift(
            backend,
            &raw.points[point_range.clone()],
            &raw.mask[mask_range.clone()],
            width,
            height,
            invocation.fov_x_degrees.filter(|value| *value > 0.0),
            context,
        )?;
        let aspect = width as f64 / height as f64;
        let diagonal = (1.0 + aspect * aspect).sqrt();
        if !focal.is_finite() || focal <= 0.0 {
            (focal, shift) = recover_focal_shift(
                backend,
                &raw.points[point_range.clone()],
                &raw.mask[mask_range.clone()],
                width,
                height,
                Some(60.0),
                context,
            )?;
        }
        let f_diagonal = f64::from(focal) * 0.5 * diagonal;
        let fx = (f_diagonal / aspect) as f32;
        let fy = f_diagonal as f32;
        let intrinsics_index = batch_index * 9;
        intrinsics[intrinsics_index..intrinsics_index + 9]
            .copy_from_slice(&[fx, 0.0, 0.5, 0.0, fy, 0.5, 0.0, 0.0, 1.0]);
        let metric_scale = raw
            .metric_scale
            .as_ref()
            .and_then(|values| values.get(batch_index))
            .copied()
            .unwrap_or(1.0);
        for pixel in 0..pixels_per_batch {
            if pixel.is_multiple_of(4_096) {
                context.check()?;
            }
            let output_index = batch_index * pixels_per_batch + pixel;
            let point_index = output_index * 3;
            let projected_depth = raw.points[point_index + 2] + shift;
            raw.points[point_index + 2] = projected_depth;
            let valid = if invocation_version == NativeMogeVersion::V2 {
                raw.mask[output_index] > 0.5 && projected_depth > 0.0
            } else {
                raw.mask[output_index] > 0.5
            };
            mask[output_index] = if valid { 1.0 } else { 0.0 };
            depth[output_index] = projected_depth
                * if invocation.apply_metric_scale {
                    metric_scale
                } else {
                    1.0
                };
            if invocation.force_projection {
                let y = pixel / width;
                let x = pixel % width;
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                raw.points[point_index] = (u - 0.5) / fx * depth[output_index];
                raw.points[point_index + 1] = (v - 0.5) / fy * depth[output_index];
                raw.points[point_index + 2] = depth[output_index];
            } else if invocation.apply_metric_scale {
                for lane in &mut raw.points[point_index..point_index + 3] {
                    *lane *= metric_scale;
                }
            }
            if invocation.apply_mask && !valid {
                raw.points[point_index..point_index + 3].fill(f32::INFINITY);
                depth[output_index] = f32::INFINITY;
                if let Some(normal) = raw.normal.as_mut() {
                    normal[point_index..point_index + 3].fill(0.0);
                }
            }
        }
    }
    if intrinsics.iter().any(|value| !value.is_finite()) {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    for output_index in 0..depth.len() {
        let point_index = output_index
            .checked_mul(3)
            .ok_or(NativeMogeError::ShapeOverflow)?;
        let point = &raw.points[point_index..point_index + 3];
        let intentionally_masked = invocation.apply_mask && mask[output_index] == 0.0;
        if intentionally_masked {
            if depth[output_index] != f32::INFINITY
                || point.iter().any(|value| *value != f32::INFINITY)
            {
                return Err(NativeMogeError::InvalidGeometryProjection);
            }
        } else if !depth[output_index].is_finite() || point.iter().any(|value| !value.is_finite()) {
            return Err(NativeMogeError::InvalidGeometryProjection);
        }
        if let Some(normal) = raw.normal.as_ref()
            && normal[point_index..point_index + 3]
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(NativeMogeError::InvalidGeometryProjection);
        }
    }
    let shape = [u64_from(batch)?, u64_from(height)?, u64_from(width)?];
    let points = tensor_from_f32_with_context_exact_native(
        backend,
        &[shape[0], shape[1], shape[2], 3],
        &raw.points,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let depth = tensor_from_f32_with_context_exact_native(
        backend,
        &shape,
        &depth,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let intrinsics = tensor_from_f32_with_context_exact_native(
        backend,
        &[shape[0], 3, 3],
        &intrinsics,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let mask = tensor_from_f32_with_context_exact_native(
        backend,
        &shape,
        &mask,
        DType::Bool,
        DeviceId::CPU,
        context,
    )?;
    let normal = raw
        .normal
        .map(|values| {
            tensor_from_f32_with_context_exact_native(
                backend,
                &[shape[0], shape[1], shape[2], 3],
                &values,
                DType::F32,
                DeviceId::CPU,
                context,
            )
        })
        .transpose()?;
    publish_geometry(
        invocation.image.clone(),
        points,
        depth,
        intrinsics,
        mask,
        normal,
        context,
    )
}

fn publish_geometry(
    image: ImageTensor,
    points: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    mask: Tensor,
    normal: Option<Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<NativeMogeGeometry, NativeMogeError> {
    context.check()?;
    Ok(NativeMogeGeometry {
        image,
        points: Some(points),
        depth: Some(depth),
        intrinsics: Some(intrinsics),
        mask: Some(mask),
        normal,
    })
}

fn focal_from_fov_degrees(
    backend: &CpuBackend,
    degrees: f32,
    aspect: f64,
    context: &ExecutionContext<'_>,
) -> Result<f32, NativeMogeError> {
    use sim_tensor::{
        BinaryOperation, Scalar, ScalarSide, TensorBackend, UnaryOperation,
        generated_elementwise_or_runtime_operation_04::deg2rad_with_context_exact_native,
        generated_elementwise_or_runtime_operation_15::tan_with_context_exact_native,
    };
    context.check()?;
    let half_degrees = tensor_from_f32_with_context_exact_native(
        backend,
        &[1],
        &[degrees / 2.0],
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let radians = deg2rad_with_context_exact_native(backend, &half_degrees, context)?;
    let tangent = tan_with_context_exact_native(backend, &radians, context)?;
    let descriptor = tangent.descriptor().clone();
    let reciprocal = backend
        .unary(
            UnaryOperation::Reciprocal,
            &tangent,
            descriptor.clone(),
            context,
        )?
        .0;
    let focal = backend
        .binary_scalar(
            BinaryOperation::Multiply,
            &reciprocal,
            Scalar::Float(aspect / (1.0 + aspect * aspect).sqrt()),
            ScalarSide::Right,
            descriptor,
            context,
        )?
        .0;
    tensor_to_f32_with_context_exact_native(backend, &focal, context)?
        .first()
        .copied()
        .ok_or(NativeMogeError::InvalidGeometryProjection)
}

fn recover_focal_shift(
    backend: &CpuBackend,
    points: &[f32],
    mask: &[f32],
    width: usize,
    height: usize,
    forced_fov_degrees: Option<f32>,
    context: &ExecutionContext<'_>,
) -> Result<(f32, f32), NativeMogeError> {
    use sim_tensor::{
        generated_linear_algebra_01::minpack_scalar_least_squares_with_context_exact_native,
        generated_reduction_01::{
            numpy_contiguous_sum_f32_with_context_exact_native,
            numpy_contiguous_sum_f64_with_context_exact_native,
        },
        generated_tensor_creation_01::linspace_f32_value,
    };
    context.check()?;
    let pixels = width
        .checked_mul(height)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let point_count = pixels
        .checked_mul(3)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    if width == 0 || height == 0 || mask.len() != pixels || points.len() != point_count {
        return Err(NativeMogeError::InvalidGeometryProjection);
    }
    let aspect = width as f64 / height as f64;
    let diagonal = (1.0 + aspect * aspect).sqrt();
    let forced_focal = forced_fov_degrees
        .map(|degrees| focal_from_fov_degrees(backend, degrees, aspect, context))
        .transpose()?;
    let span_x = aspect / diagonal;
    let span_y = 1.0 / diagonal;
    let start_x = (-span_x * (width - 1) as f64 / width as f64) as f32;
    let end_x = (span_x * (width - 1) as f64 / width as f64) as f32;
    let start_y = (-span_y * (height - 1) as f64 / height as f64) as f32;
    let end_y = (span_y * (height - 1) as f64 / height as f64) as f32;
    let mut samples = backend.workspace_vec(context, 64 * 64)?;
    for output_y in 0usize..64 {
        let source_y = output_y
            .checked_mul(height)
            .ok_or(NativeMogeError::ShapeOverflow)?
            / 64;
        for output_x in 0usize..64 {
            let source_x = output_x
                .checked_mul(width)
                .ok_or(NativeMogeError::ShapeOverflow)?
                / 64;
            let pixel = source_y * width + source_x;
            if mask[pixel] > 0.5 {
                samples.try_push((
                    pixel,
                    [
                        linspace_f32_value(start_x, end_x, width, source_x)?,
                        linspace_f32_value(start_y, end_y, height, source_y)?,
                    ],
                ))?;
            }
        }
        context.check()?;
    }
    if samples.len() < 2 {
        return Ok((forced_focal.unwrap_or(1.0), 0.0));
    }
    let residual_count = samples
        .len()
        .checked_mul(2)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let solved = {
        let mut projected = backend.workspace_vec(context, residual_count)?;
        let mut products = backend.workspace_vec(context, residual_count)?;
        for _ in 0..residual_count {
            projected.try_push(0.0_f64)?;
            products.try_push(0.0_f64)?;
        }
        minpack_scalar_least_squares_with_context_exact_native::<NativeMogeError>(
            backend,
            residual_count,
            |shift, residual| {
                for (sample_index, &(pixel, uv)) in samples.iter().enumerate() {
                    if sample_index.is_multiple_of(256) {
                        context.check()?;
                    }
                    let z = f64::from(points[pixel * 3 + 2]) + shift;
                    for (axis, coordinate) in uv.into_iter().enumerate() {
                        let index = sample_index * 2 + axis;
                        projected[index] = f64::from(points[pixel * 3 + axis]) / z;
                        products[index] = projected[index] * f64::from(coordinate);
                    }
                }
                let focal = match forced_focal {
                    Some(focal) => f64::from(focal),
                    None => {
                        let numerator =
                            numpy_contiguous_sum_f64_with_context_exact_native(&products, context)?;
                        for (square, value) in products.iter_mut().zip(projected.iter()) {
                            *square = value * value;
                        }
                        numerator
                            / numpy_contiguous_sum_f64_with_context_exact_native(
                                &products, context,
                            )?
                    }
                };
                for (sample_index, &(_, uv)) in samples.iter().enumerate() {
                    for (axis, coordinate) in uv.into_iter().enumerate() {
                        let index = sample_index * 2 + axis;
                        residual[index] = focal * projected[index] - f64::from(coordinate);
                    }
                }
                Ok(())
            },
            context,
        )?
    };
    // SciPy passes a float64 array during solving, but the original helper's
    // final Python scalar is weakly promoted against its float32 point array.
    let focal = match forced_focal {
        Some(focal) => focal,
        None => {
            let mut cross = backend.workspace_vec(context, residual_count)?;
            let mut square = backend.workspace_vec(context, residual_count)?;
            let shift = solved.value as f32;
            for (sample_index, &(pixel, uv)) in samples.iter().enumerate() {
                if sample_index.is_multiple_of(256) {
                    context.check()?;
                }
                let z = points[pixel * 3 + 2] + shift;
                for (axis, coordinate) in uv.into_iter().enumerate() {
                    let projected = points[pixel * 3 + axis] / z;
                    cross.try_push(projected * coordinate)?;
                    square.try_push(projected * projected)?;
                }
            }
            numpy_contiguous_sum_f32_with_context_exact_native(&cross, context)?
                / numpy_contiguous_sum_f32_with_context_exact_native(&square, context)?
        }
    };
    context.check()?;
    Ok((focal, solved.value as f32))
}

fn preprocessing_dimensions(
    configuration: NativeMogeConfiguration,
    height: usize,
    width: usize,
    resolution_level: u8,
) -> Result<(usize, usize, usize, usize), NativeMogeError> {
    if !configuration.source_exact {
        return Ok((
            configuration.image,
            configuration.image,
            configuration.image,
            configuration.image,
        ));
    }
    let fraction = f64::from(resolution_level) / 9.0;
    let (minimum, maximum) = match configuration.version {
        NativeMogeVersion::V1 => (1_200.0, 2_500.0),
        NativeMogeVersion::V2 => (1_200.0, 3_600.0),
    };
    let tokens = minimum + fraction * (maximum - minimum);
    let aspect = width as f64 / height as f64;
    match configuration.version {
        NativeMogeVersion::V1 => {
            let patch_area = configuration
                .patch
                .checked_mul(configuration.patch)
                .ok_or(NativeMogeError::ShapeOverflow)?;
            let resize = (tokens * patch_area as f64 / (height as f64 * width as f64)).sqrt();
            let resized_height = (height as f64 * resize) as usize;
            let resized_width = (width as f64 * resize) as usize;
            if resized_height < configuration.patch || resized_width < configuration.patch {
                return Err(NativeMogeError::InvalidImage(
                    "image aspect ratio produces an empty patch grid".to_owned(),
                ));
            }
            Ok((
                resized_height / configuration.patch * configuration.patch,
                resized_width / configuration.patch * configuration.patch,
                resized_height,
                resized_width,
            ))
        }
        NativeMogeVersion::V2 => {
            let rows = (tokens / aspect).sqrt().round_ties_even() as usize;
            let columns = (tokens * aspect).sqrt().round_ties_even() as usize;
            if rows == 0 || columns == 0 {
                return Err(NativeMogeError::InvalidImage(
                    "image aspect ratio produces an empty patch grid".to_owned(),
                ));
            }
            let target_height = rows
                .checked_mul(configuration.patch)
                .ok_or(NativeMogeError::ShapeOverflow)?;
            let target_width = columns
                .checked_mul(configuration.patch)
                .ok_or(NativeMogeError::ShapeOverflow)?;
            Ok((target_height, target_width, height, width))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_image(
    backend: &CpuBackend,
    resource: &NativeMogeResource,
    values: &[f32],
    batch: usize,
    source_height: usize,
    source_width: usize,
    channels: usize,
    target_height: usize,
    target_width: usize,
    head_height: usize,
    head_width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeMogeError> {
    let source_length = batch
        .checked_mul(3)
        .and_then(|value| value.checked_mul(source_height))
        .and_then(|value| value.checked_mul(source_width))
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let mut source = filled_f32(source_length, 0.0)?;
    for batch_index in 0..batch {
        for y in 0..source_height {
            for x in 0..source_width {
                let destination = (batch_index * source_height + y) * source_width + x;
                if destination.is_multiple_of(4_096) {
                    context.check()?;
                }
                for channel in 0..3 {
                    source[(batch_index * 3 + channel) * source_height * source_width
                        + y * source_width
                        + x] = values[((batch_index * source_height + y) * source_width + x)
                        * channels
                        + channel];
                }
            }
        }
    }
    let mut image = tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64_from(batch)?,
            3,
            u64_from(source_height)?,
            u64_from(source_width)?,
        ],
        &source,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let (mut normalized_height, mut normalized_width) = match resource.configuration.version {
        NativeMogeVersion::V1 if resource.configuration.source_exact => {
            image = interpolate_tensor_with_context_exact_native(
                backend,
                &image,
                &InterpolateConfiguration {
                    output_size: Some(vec![head_height, head_width]),
                    scale_factor: None,
                    mode: InterpolateMode::Bicubic,
                    align_corners: Some(false),
                    recompute_scale_factor: None,
                    antialias: true,
                },
                context,
            )?;
            (head_height, head_width)
        }
        _ => (source_height, source_width),
    };
    if resource.configuration.version == NativeMogeVersion::V2
        && (normalized_height, normalized_width) != (target_height, target_width)
    {
        image = resize_nchw(backend, &image, target_height, target_width, true, context)?;
        normalized_height = target_height;
        normalized_width = target_width;
    }
    let (mean, deviation) = if resource.configuration.source_exact {
        let prefix = if resource.configuration.version == NativeMogeVersion::V1 {
            ""
        } else {
            "encoder."
        };
        (
            tensor_to_f32_with_context_exact_native(
                backend,
                resource.execution_tensor(&format!("{prefix}image_mean"))?,
                context,
            )?,
            tensor_to_f32_with_context_exact_native(
                backend,
                resource.execution_tensor(&format!("{prefix}image_std"))?,
                context,
            )?,
        )
    } else {
        (IMAGE_MEAN.to_vec(), IMAGE_STANDARD_DEVIATION.to_vec())
    };
    if mean.len() != 3 || deviation.len() != 3 {
        return Err(NativeMogeError::InvalidCheckpoint(
            "image normalization buffers are invalid".to_owned(),
        ));
    }
    let mut normalized = tensor_to_f32_with_context_exact_native(backend, &image, context)?;
    for batch_index in 0..batch {
        for channel in 0..3 {
            for pixel in 0..normalized_height * normalized_width {
                let index =
                    (batch_index * 3 + channel) * normalized_height * normalized_width + pixel;
                normalized[index] = (normalized[index] - mean[channel]) / deviation[channel];
            }
        }
    }
    image = tensor_from_f32_with_context_exact_native(
        backend,
        &[
            u64_from(batch)?,
            3,
            u64_from(normalized_height)?,
            u64_from(normalized_width)?,
        ],
        &normalized,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    if resource.configuration.version == NativeMogeVersion::V1
        && (normalized_height, normalized_width) != (target_height, target_width)
    {
        image = resize_nchw(backend, &image, target_height, target_width, true, context)?;
    }
    Ok(image)
}

fn dino_backbone(
    configuration: NativeMogeConfiguration,
) -> Result<NativeDino2Backbone, NativeMogeError> {
    Ok(NativeDino2Backbone::new(NativeDino2Configuration {
        prefix: configuration.dino_prefix,
        hidden: configuration.hidden,
        layer_count: configuration.layer_count,
        attention_heads: configuration.attention_heads,
        patch: configuration.patch,
        image: configuration.image,
        qknorm_start: None,
        alternate_attention_start: None,
        rope_start: None,
        concatenate_camera_token: false,
        use_mask_token: true,
        swiglu: false,
        output_layers: configuration.output_layers,
    })?)
}

fn detect_configuration(
    state: &BTreeMap<String, Tensor>,
    source_exact: bool,
) -> Result<NativeMogeConfiguration, NativeMogeError> {
    let has_v1 = state.keys().any(|key| key.starts_with("head."));
    let has_v2 = state
        .keys()
        .any(|key| key.starts_with("encoder.output_projections."));
    if has_v1 == has_v2 {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    let version = if has_v2 {
        NativeMogeVersion::V2
    } else {
        NativeMogeVersion::V1
    };
    let dino_prefix = "native.backbone";
    let cls_key = format!("{dino_prefix}.embeddings.cls_token");
    let cls = state
        .get(&cls_key)
        .ok_or_else(|| NativeMogeError::MissingState(cls_key.clone()))?;
    let hidden = match cls.descriptor().shape() {
        [1, 1, hidden] => usize_from(*hidden)?,
        shape => {
            return Err(NativeMogeError::StateShape {
                key: cls_key,
                expected: vec![1, 1, 1],
                actual: shape.to_vec(),
                actual_dtype: cls.descriptor().dtype(),
            });
        }
    };
    let layer_prefix = format!("{dino_prefix}.encoder.layer.");
    let layer_count = state
        .keys()
        .filter_map(|key| {
            key.strip_prefix(&layer_prefix)
                .and_then(|suffix| suffix.split('.').next())
                .and_then(|index| index.parse::<usize>().ok())
        })
        .max()
        .and_then(|value| value.checked_add(1))
        .ok_or(NativeMogeError::UnsupportedArchitecture)?;
    let projection_key = format!("{dino_prefix}.embeddings.patch_embeddings.projection.weight");
    let projection = state
        .get(&projection_key)
        .ok_or_else(|| NativeMogeError::MissingState(projection_key.clone()))?;
    let patch = match projection.descriptor().shape() {
        [output, 3, patch_height, patch_width]
            if *output == u64_from(hidden)? && patch_height == patch_width =>
        {
            usize_from(*patch_height)?
        }
        _ => return Err(NativeMogeError::UnsupportedArchitecture),
    };
    let positions_key = format!("{dino_prefix}.embeddings.position_embeddings");
    let positions = state
        .get(&positions_key)
        .ok_or_else(|| NativeMogeError::MissingState(positions_key.clone()))?;
    let position_tokens = match positions.descriptor().shape() {
        [1, tokens, channels] if *channels == u64_from(hidden)? && *tokens > 1 => {
            usize_from(*tokens - 1)?
        }
        _ => return Err(NativeMogeError::UnsupportedArchitecture),
    };
    let side = integer_square_root(position_tokens);
    if side * side != position_tokens || layer_count < 4 {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    let image = side
        .checked_mul(patch)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    if source_exact && (patch != 14 || image != 518) {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    let attention_heads = if source_exact {
        if hidden < 64 || !hidden.is_multiple_of(64) {
            return Err(NativeMogeError::UnsupportedArchitecture);
        }
        hidden / 64
    } else {
        1
    };
    let output_layers = match version {
        NativeMogeVersion::V1 => [
            layer_count - 4,
            layer_count - 3,
            layer_count - 2,
            layer_count - 1,
        ],
        NativeMogeVersion::V2 => [
            layer_count / 4 - 1,
            layer_count / 2 - 1,
            layer_count * 3 / 4 - 1,
            layer_count - 1,
        ],
    };
    let has_normal = state.keys().any(|key| key.starts_with("normal_head."));
    if version == NativeMogeVersion::V1 && has_normal {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    let required: &[&str] = match version {
        NativeMogeVersion::V1 => &[
            "head.projects.0.weight",
            "head.output_block.0.2.weight",
            "head.output_block.1.2.weight",
            "head.upsample_blocks.0.0.0.weight",
        ],
        NativeMogeVersion::V2 => &[
            "encoder.output_projections.0.weight",
            "neck.input_blocks.0.weight",
            "points_head.input_blocks.0.weight",
            "mask_head.input_blocks.0.weight",
            "scale_head.0.weight",
        ],
    };
    for key in required {
        if !state.contains_key(*key) {
            return Err(NativeMogeError::MissingState((*key).to_owned()));
        }
    }
    Ok(NativeMogeConfiguration {
        version,
        dino_prefix,
        hidden,
        layer_count,
        attention_heads,
        patch,
        image,
        output_layers,
        has_normal,
        source_exact,
    })
}

fn state_manifest(
    configuration: NativeMogeConfiguration,
    state: &BTreeMap<String, Tensor>,
) -> Result<Vec<StateSpecification>, NativeMogeError> {
    let backbone = dino_backbone(configuration)?;
    let mut manifest = backbone
        .state_manifest()?
        .into_iter()
        .map(|specification| StateSpecification {
            key: specification.key,
            shape: specification.shape,
        })
        .collect::<Vec<_>>();
    if !configuration.source_exact {
        add_reduced_head_manifest_version(configuration.version, &mut manifest);
        manifest.sort_by(|left, right| left.key.cmp(&right.key));
        return Ok(manifest);
    }
    match configuration.version {
        NativeMogeVersion::V1 => add_source_v1_manifest(configuration, state, &mut manifest)?,
        NativeMogeVersion::V2 => add_source_v2_manifest(configuration, state, &mut manifest)?,
    }
    manifest.sort_by(|left, right| left.key.cmp(&right.key));
    Ok(manifest)
}

fn state_shape<'a>(
    state: &'a BTreeMap<String, Tensor>,
    key: &str,
) -> Result<&'a [u64], NativeMogeError> {
    state
        .get(key)
        .map(|tensor| tensor.descriptor().shape())
        .ok_or_else(|| NativeMogeError::MissingState(key.to_owned()))
}

fn indexed_count(
    state: &BTreeMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Result<usize, NativeMogeError> {
    let mut indexes = state
        .keys()
        .filter_map(|key| {
            key.strip_prefix(prefix)
                .and_then(|tail| tail.strip_suffix(suffix))
                .and_then(|index| index.parse::<usize>().ok())
        })
        .collect::<Vec<_>>();
    indexes.sort_unstable();
    indexes.dedup();
    if indexes.is_empty() || indexes.iter().copied().ne(0..indexes.len()) {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    Ok(indexes.len())
}

fn one_based_indexed_count(
    state: &BTreeMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Result<usize, NativeMogeError> {
    let mut indexes = state
        .keys()
        .filter_map(|key| {
            key.strip_prefix(prefix)
                .and_then(|tail| tail.strip_suffix(suffix))
                .and_then(|index| index.parse::<usize>().ok())
        })
        .collect::<Vec<_>>();
    indexes.sort_unstable();
    indexes.dedup();
    if indexes.is_empty() || indexes.iter().copied().ne(1..=indexes.len()) {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    Ok(indexes.len())
}

fn add_source_v1_manifest(
    configuration: NativeMogeConfiguration,
    state: &BTreeMap<String, Tensor>,
    manifest: &mut Vec<StateSpecification>,
) -> Result<(), NativeMogeError> {
    add_reduced_specification(manifest, "image_mean".to_owned(), &[1, 3, 1, 1]);
    add_reduced_specification(manifest, "image_std".to_owned(), &[1, 3, 1, 1]);
    for index in 0..4 {
        add_reduced_convolution(
            manifest,
            format!("head.projects.{index}"),
            u64_from(configuration.hidden)?,
            512,
            1,
            false,
        );
    }
    let levels = indexed_count(state, "head.upsample_blocks.", ".0.0.weight")?;
    let residuals = one_based_indexed_count(state, "head.upsample_blocks.0.", ".layers.2.weight")?;
    let mut input_channels = 512_u64;
    let mut final_channels = None;
    let mut hidden_multiplier = None;
    for level in 0..levels {
        let weight_key = format!("head.upsample_blocks.{level}.0.0.weight");
        let output_channels = match state_shape(state, &weight_key)? {
            [input, output, 2, 2] if *input == input_channels + 2 && *output > 0 => *output,
            _ => return Err(NativeMogeError::UnsupportedArchitecture),
        };
        add_reduced_convolution(
            manifest,
            format!("head.upsample_blocks.{level}.0.0"),
            input_channels + 2,
            output_channels,
            2,
            true,
        );
        add_reduced_convolution(
            manifest,
            format!("head.upsample_blocks.{level}.0.1"),
            output_channels,
            output_channels,
            3,
            false,
        );
        let level_residuals = one_based_indexed_count(
            state,
            &format!("head.upsample_blocks.{level}."),
            ".layers.2.weight",
        )?;
        if level_residuals != residuals {
            return Err(NativeMogeError::UnsupportedArchitecture);
        }
        for residual in 1..=residuals {
            let prefix = format!("head.upsample_blocks.{level}.{residual}");
            let hidden = match state_shape(state, &format!("{prefix}.layers.2.weight"))? {
                [hidden, input, 3, 3] if *input == output_channels && *hidden > 0 => *hidden,
                _ => return Err(NativeMogeError::UnsupportedArchitecture),
            };
            if !hidden.is_multiple_of(output_channels) {
                return Err(NativeMogeError::UnsupportedArchitecture);
            }
            let multiplier = hidden / output_channels;
            if hidden_multiplier
                .replace(multiplier)
                .is_some_and(|value| value != multiplier)
            {
                return Err(NativeMogeError::UnsupportedArchitecture);
            }
            add_source_residual(manifest, prefix, output_channels, hidden);
        }
        input_channels = output_channels;
        final_channels = Some(output_channels);
    }
    let final_channels = final_channels.ok_or(NativeMogeError::UnsupportedArchitecture)?;
    for (index, output_channels) in [(0, 3), (1, 1)] {
        add_reduced_convolution(
            manifest,
            format!("head.output_block.{index}.0"),
            final_channels + 2,
            32,
            3,
            false,
        );
        add_reduced_convolution(
            manifest,
            format!("head.output_block.{index}.2"),
            32,
            output_channels,
            1,
            false,
        );
    }
    Ok(())
}

fn add_source_residual(
    manifest: &mut Vec<StateSpecification>,
    prefix: String,
    channels: u64,
    hidden: u64,
) {
    add_reduced_specification(manifest, format!("{prefix}.layers.0.weight"), &[channels]);
    add_reduced_specification(manifest, format!("{prefix}.layers.0.bias"), &[channels]);
    add_reduced_convolution(
        manifest,
        format!("{prefix}.layers.2"),
        channels,
        hidden,
        3,
        false,
    );
    add_reduced_specification(manifest, format!("{prefix}.layers.3.weight"), &[hidden]);
    add_reduced_specification(manifest, format!("{prefix}.layers.3.bias"), &[hidden]);
    add_reduced_convolution(
        manifest,
        format!("{prefix}.layers.5"),
        hidden,
        channels,
        3,
        false,
    );
}

fn add_source_conv_stack_manifest(
    prefix: &str,
    state: &BTreeMap<String, Tensor>,
    manifest: &mut Vec<StateSpecification>,
    expected_inputs: Option<&[u64]>,
) -> Result<Vec<u64>, NativeMogeError> {
    let levels = indexed_count(state, &format!("{prefix}.input_blocks."), ".weight")?;
    if expected_inputs.is_some_and(|inputs| inputs.len() != levels) {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    let has_norm = state.contains_key(&format!("{prefix}.res_blocks.0.0.layers.0.weight"));
    let mut residual_channels = Vec::with_capacity(levels);
    let mut output_channels = Vec::with_capacity(levels);
    let mut residual_counts = Vec::with_capacity(levels);
    for level in 0..levels {
        let (residual, input) =
            match state_shape(state, &format!("{prefix}.input_blocks.{level}.weight"))? {
                [output, input, 1, 1] if *output > 0 && *input > 0 => (*output, *input),
                _ => return Err(NativeMogeError::UnsupportedArchitecture),
            };
        if let Some(inputs) = expected_inputs {
            if inputs[level] != input {
                return Err(NativeMogeError::UnsupportedArchitecture);
            }
        } else if (level == 0 && input < 3) || (level > 0 && input != 2) {
            return Err(NativeMogeError::UnsupportedArchitecture);
        }
        add_reduced_convolution(
            manifest,
            format!("{prefix}.input_blocks.{level}"),
            input,
            residual,
            1,
            false,
        );
        let blocks = indexed_count(
            state,
            &format!("{prefix}.res_blocks.{level}."),
            ".layers.2.weight",
        )?;
        residual_counts.push(blocks);
        for block in 0..blocks {
            let block_prefix = format!("{prefix}.res_blocks.{level}.{block}");
            let hidden = match state_shape(state, &format!("{block_prefix}.layers.2.weight"))? {
                [hidden, input, 3, 3] if *input == residual && *hidden > 0 => *hidden,
                _ => return Err(NativeMogeError::UnsupportedArchitecture),
            };
            if has_norm {
                add_source_residual(manifest, block_prefix, residual, hidden);
            } else {
                add_reduced_convolution(
                    manifest,
                    format!("{block_prefix}.layers.2"),
                    residual,
                    hidden,
                    3,
                    false,
                );
                add_reduced_convolution(
                    manifest,
                    format!("{block_prefix}.layers.5"),
                    hidden,
                    residual,
                    3,
                    false,
                );
            }
        }
        let output_key = format!("{prefix}.output_blocks.{level}.weight");
        let output = if state.contains_key(&output_key) {
            let output = match state_shape(state, &output_key)? {
                [output, input, 1, 1] if *input == residual && *output > 0 => *output,
                _ => return Err(NativeMogeError::UnsupportedArchitecture),
            };
            add_reduced_convolution(
                manifest,
                format!("{prefix}.output_blocks.{level}"),
                residual,
                output,
                1,
                false,
            );
            output
        } else {
            residual
        };
        residual_channels.push(residual);
        output_channels.push(output);
    }
    if residual_counts
        .windows(2)
        .any(|counts| counts[0] != counts[1])
    {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    for level in 0..levels.saturating_sub(1) {
        let transposed_prefix = format!("{prefix}.resamplers.{level}.0");
        let convolution_input = if state.contains_key(&format!("{transposed_prefix}.weight")) {
            add_reduced_convolution(
                manifest,
                transposed_prefix,
                residual_channels[level],
                residual_channels[level + 1],
                2,
                true,
            );
            residual_channels[level + 1]
        } else {
            residual_channels[level]
        };
        add_reduced_convolution(
            manifest,
            format!("{prefix}.resamplers.{level}.1"),
            convolution_input,
            residual_channels[level + 1],
            3,
            false,
        );
    }
    Ok(output_channels)
}

fn add_source_v2_manifest(
    configuration: NativeMogeConfiguration,
    state: &BTreeMap<String, Tensor>,
    manifest: &mut Vec<StateSpecification>,
) -> Result<(), NativeMogeError> {
    add_reduced_specification(manifest, "encoder.image_mean".to_owned(), &[1, 3, 1, 1]);
    add_reduced_specification(manifest, "encoder.image_std".to_owned(), &[1, 3, 1, 1]);
    let projection_channels = match state_shape(state, "encoder.output_projections.0.weight")? {
        [output, input, 1, 1] if *input == u64_from(configuration.hidden)? && *output > 0 => {
            *output
        }
        _ => return Err(NativeMogeError::UnsupportedArchitecture),
    };
    for index in 0..4 {
        add_reduced_convolution(
            manifest,
            format!("encoder.output_projections.{index}"),
            u64_from(configuration.hidden)?,
            projection_channels,
            1,
            false,
        );
    }
    let neck_levels = indexed_count(state, "neck.input_blocks.", ".weight")?;
    let mut neck_inputs = vec![2_u64; neck_levels];
    let first = neck_inputs
        .first_mut()
        .ok_or(NativeMogeError::UnsupportedArchitecture)?;
    *first = projection_channels + 2;
    let neck_outputs = add_source_conv_stack_manifest("neck", state, manifest, Some(&neck_inputs))?;
    add_source_conv_stack_manifest("points_head", state, manifest, Some(&neck_outputs))?;
    add_source_conv_stack_manifest("mask_head", state, manifest, Some(&neck_outputs))?;
    if configuration.has_normal {
        add_source_conv_stack_manifest("normal_head", state, manifest, Some(&neck_outputs))?;
    }
    let mut scale_layers = 0_usize;
    while state.contains_key(&format!("scale_head.{}.weight", scale_layers * 2)) {
        scale_layers += 1;
    }
    if scale_layers == 0 {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    let mut input = u64_from(configuration.hidden)?;
    for layer in 0..scale_layers {
        let sequential_index = layer * 2;
        let output = match state_shape(state, &format!("scale_head.{sequential_index}.weight"))? {
            [output, actual_input] if *actual_input == input && *output > 0 => *output,
            _ => return Err(NativeMogeError::UnsupportedArchitecture),
        };
        add_reduced_specification(
            manifest,
            format!("scale_head.{sequential_index}.weight"),
            &[output, input],
        );
        add_reduced_specification(
            manifest,
            format!("scale_head.{sequential_index}.bias"),
            &[output],
        );
        input = output;
    }
    if input != 1 {
        return Err(NativeMogeError::UnsupportedArchitecture);
    }
    Ok(())
}

fn normalize_state(
    backend: &CpuBackend,
    ordered_state: Vec<(String, Tensor)>,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, NativeMogeError> {
    let raw_v2 = ordered_state
        .iter()
        .any(|(key, _)| strip_model_prefix(key).starts_with("encoder.backbone."));
    let source_dino_prefix = if raw_v2 {
        "encoder.backbone."
    } else {
        "backbone."
    };
    let mut normalized = BTreeMap::new();
    for (index, (key, tensor)) in ordered_state.into_iter().enumerate() {
        if index.is_multiple_of(16) {
            context.check()?;
        }
        validate_state_key(&key)?;
        if tensor.descriptor().stream() != context.stream {
            return Err(NativeMogeError::InvalidCheckpoint(format!(
                "state {key} is on a foreign execution stream"
            )));
        }
        let remapped = remap_dino_key(
            strip_model_prefix(&key),
            source_dino_prefix,
            "native.backbone.",
        );
        if remapped.ends_with(".attn.qkv.weight") || remapped.ends_with(".attn.qkv.bias") {
            split_qkv_state(backend, &remapped, tensor, &mut normalized, context)?;
        } else {
            insert_unique(&mut normalized, remapped, tensor)?;
        }
    }
    Ok(normalized)
}

fn strip_model_prefix(key: &str) -> &str {
    key.strip_prefix("model.").unwrap_or(key)
}

fn remap_dino_key(key: &str, source_prefix: &str, target_prefix: &str) -> String {
    let Some(relative) = key.strip_prefix(source_prefix) else {
        return key.to_owned();
    };
    let top = match relative {
        "patch_embed.proj.weight" => Some("embeddings.patch_embeddings.projection.weight"),
        "patch_embed.proj.bias" => Some("embeddings.patch_embeddings.projection.bias"),
        "cls_token" => Some("embeddings.cls_token"),
        "pos_embed" => Some("embeddings.position_embeddings"),
        "register_tokens" => Some("embeddings.register_tokens"),
        "mask_token" => Some("embeddings.mask_token"),
        "norm.weight" => Some("layernorm.weight"),
        "norm.bias" => Some("layernorm.bias"),
        _ => None,
    };
    if let Some(top) = top {
        return format!("{target_prefix}{top}");
    }
    let Some(block) = relative.strip_prefix("blocks.") else {
        return key.to_owned();
    };
    let Some((layer, suffix)) = block.split_once('.') else {
        return key.to_owned();
    };
    let suffix = suffix
        .replace("ls1.gamma", "layer_scale1.lambda1")
        .replace("ls2.gamma", "layer_scale2.lambda1")
        .replace("attn.proj.", "attention.output.dense.")
        .replace("mlp.w12.", "mlp.weights_in.")
        .replace("mlp.w3.", "mlp.weights_out.");
    format!("{target_prefix}encoder.layer.{layer}.{suffix}")
}

fn split_qkv_state(
    backend: &CpuBackend,
    key: &str,
    tensor: Tensor,
    output: &mut BTreeMap<String, Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeMogeError> {
    let descriptor = tensor.descriptor().clone();
    let first = descriptor
        .shape()
        .first()
        .copied()
        .ok_or(NativeMogeError::ShapeOverflow)?;
    if !first.is_multiple_of(3) {
        return Err(NativeMogeError::InvalidCheckpoint(format!(
            "fused QKV state {key} has an indivisible leading dimension"
        )));
    }
    let values = tensor_to_f32_with_context_exact_native(backend, &tensor, context)?;
    let rows = usize_from(first / 3)?;
    let trailing = descriptor.shape()[1..]
        .iter()
        .try_fold(1_usize, |total, dimension| {
            total
                .checked_mul(usize_from(*dimension)?)
                .ok_or(NativeMogeError::ShapeOverflow)
        })?;
    let chunk = rows
        .checked_mul(trailing)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let total = chunk.checked_mul(3).ok_or(NativeMogeError::ShapeOverflow)?;
    if values.len() != total {
        return Err(NativeMogeError::InvalidCheckpoint(format!(
            "fused QKV state {key} has invalid storage"
        )));
    }
    let mut shape = descriptor.shape().to_vec();
    *shape.first_mut().ok_or(NativeMogeError::ShapeOverflow)? = first / 3;
    let (base, tail) = key
        .strip_suffix(".attn.qkv.weight")
        .map(|value| (value, "weight"))
        .or_else(|| {
            key.strip_suffix(".attn.qkv.bias")
                .map(|value| (value, "bias"))
        })
        .ok_or_else(|| NativeMogeError::UnexpectedState(key.to_owned()))?;
    for (index, name) in ["query", "key", "value"].into_iter().enumerate() {
        let start = index
            .checked_mul(chunk)
            .ok_or(NativeMogeError::ShapeOverflow)?;
        let end = start
            .checked_add(chunk)
            .ok_or(NativeMogeError::ShapeOverflow)?;
        let values = values
            .get(start..end)
            .ok_or(NativeMogeError::ShapeOverflow)?;
        let projected = tensor_from_f32_with_context_exact_native(
            backend,
            &shape,
            values,
            descriptor.dtype(),
            descriptor.device(),
            context,
        )?;
        insert_unique(
            output,
            format!(
                "{}.attention.attention.{name}.{tail}",
                base.replace(".blocks.", ".encoder.layer.")
            ),
            projected,
        )?;
    }
    Ok(())
}

fn insert_unique(
    output: &mut BTreeMap<String, Tensor>,
    key: String,
    tensor: Tensor,
) -> Result<(), NativeMogeError> {
    if output.insert(key.clone(), tensor).is_some() {
        return Err(NativeMogeError::DuplicateStateKey(key));
    }
    Ok(())
}

fn validate_source_state(
    state: &BTreeMap<String, Tensor>,
    specifications: &[StateSpecification],
    stream: StreamId,
    cancellation: &CancellationToken,
) -> Result<DType, NativeMogeError> {
    if state.len() != specifications.len() {
        return Err(NativeMogeError::InvalidCheckpoint(
            "strict state cardinality changed".to_owned(),
        ));
    }
    let mut dtype = None;
    for (index, specification) in specifications.iter().enumerate() {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        let tensor = state
            .get(&specification.key)
            .ok_or_else(|| NativeMogeError::MissingState(specification.key.clone()))?;
        let descriptor = tensor.descriptor();
        if descriptor.shape() != specification.shape
            || !matches!(descriptor.dtype(), DType::F16 | DType::Bf16 | DType::F32)
            || descriptor.device() != DeviceId::CPU
            || descriptor.stream() != stream
        {
            return Err(NativeMogeError::StateShape {
                key: specification.key.clone(),
                expected: specification.shape.clone(),
                actual: descriptor.shape().to_vec(),
                actual_dtype: descriptor.dtype(),
            });
        }
        if dtype
            .replace(descriptor.dtype())
            .is_some_and(|value| value != descriptor.dtype())
        {
            return Err(NativeMogeError::InvalidCheckpoint(
                "mixed checkpoint dtypes are unsupported".to_owned(),
            ));
        }
    }
    dtype.ok_or(NativeMogeError::UnsupportedArchitecture)
}

fn preflight_execution_memory(
    resource: &NativeMogeResource,
    batch: usize,
    height: usize,
    width: usize,
    resolution_level: u8,
) -> Result<(), NativeMogeError> {
    let (target_height, target_width, head_height, head_width) =
        preprocessing_dimensions(resource.configuration, height, width, resolution_level)?;
    let input_elements = batch
        .checked_mul(3)
        .and_then(|value| {
            value.checked_mul(
                height
                    .checked_mul(width)?
                    .checked_add(head_height.checked_mul(head_width)?)?
                    .checked_add(target_height.checked_mul(target_width)?)?,
            )
        })
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let input = input_elements
        .checked_mul(4)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let tokens = batch
        .checked_mul(target_height / resource.configuration.patch)
        .and_then(|value| value.checked_mul(target_width / resource.configuration.patch))
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let dino = tokens
        .checked_mul(resource.configuration.hidden)
        .and_then(|value| value.checked_mul(32))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let max_channels = resource
        .execution_state
        .values()
        .flat_map(|tensor| tensor.descriptor().shape().iter().copied())
        .max()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let head_spatial = match resource.configuration.version {
        NativeMogeVersion::V1 => head_height
            .checked_mul(head_width)
            .ok_or(NativeMogeError::ShapeOverflow)?,
        NativeMogeVersion::V2 => (target_height / resource.configuration.patch)
            .checked_mul(16)
            .and_then(|value| {
                value.checked_mul((target_width / resource.configuration.patch).checked_mul(16)?)
            })
            .ok_or(NativeMogeError::ShapeOverflow)?,
    };
    let head = batch
        .checked_mul(head_spatial)
        .and_then(|value| value.checked_mul(max_channels))
        .and_then(|value| value.checked_mul(12))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let output_pixels = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let output = output_pixels
        .checked_mul(if resource.configuration.has_normal {
            8
        } else {
            5
        })
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_add(output_pixels))
        .and_then(|value| value.checked_add(batch.checked_mul(9)?.checked_mul(4)?))
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let required = resource
        .resident_bytes
        .checked_add(u64_from(input)?)
        .and_then(|value| value.checked_add(u64_from(dino).ok()?))
        .and_then(|value| value.checked_add(u64_from(head).ok()?))
        .and_then(|value| value.checked_add(u64_from(output).ok()?))
        .ok_or(NativeMogeError::ShapeOverflow)?;
    if required > resource.memory_budget_bytes {
        return Err(NativeMogeError::OutOfMemory {
            required,
            budget: resource.memory_budget_bytes,
        });
    }
    Ok(())
}

fn projected_resident_preflight(
    artifact: &str,
    source_state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<u64, NativeMogeError> {
    let source = resident_tensor_bytes([source_state], cancellation)?;
    let projected = source_state.values().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(
                tensor
                    .descriptor()
                    .element_count()?
                    .checked_mul(4)
                    .ok_or(NativeMogeError::ShapeOverflow)?,
            )
            .ok_or(NativeMogeError::ShapeOverflow)
    })?;
    source
        .checked_add(projected)
        .and_then(|value| value.checked_add(u64::try_from(artifact.len()).ok()?))
        .ok_or(NativeMogeError::ShapeOverflow)
}

fn semantic_digest(
    configuration: NativeMogeConfiguration,
    artifact: &str,
    source_dtype: DType,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, NativeMogeError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.moge-resource.v1\0");
    hasher.update([match configuration.version {
        NativeMogeVersion::V1 => 1,
        NativeMogeVersion::V2 => 2,
    }]);
    hasher.update(artifact.as_bytes());
    hasher.update(source_dtype.catalog_name().as_bytes());
    for (index, (key, source)) in source_state.iter().enumerate() {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        let projected = execution_state
            .get(key)
            .ok_or_else(|| NativeMogeError::MissingState(key.clone()))?;
        hash_tensor(&mut hasher, key, source, cancellation)?;
        hash_tensor(&mut hasher, key, projected, cancellation)?;
    }
    cancellation.check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_tensor(
    hasher: &mut Sha256,
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeMogeError> {
    hasher.update(u64_from(key.len())?.to_le_bytes());
    hasher.update(key.as_bytes());
    hasher.update(tensor.descriptor().dtype().catalog_name().as_bytes());
    for dimension in tensor.descriptor().shape() {
        hasher.update(dimension.to_le_bytes());
    }
    for chunk in tensor.contiguous_bytes()?.chunks(DIGEST_CHUNK_BYTES) {
        cancellation.check()?;
        hasher.update(chunk);
    }
    Ok(())
}

fn validate_finite_tensor(
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeMogeError> {
    let bytes = tensor.contiguous_bytes()?;
    if !bytes.len().is_multiple_of(4) {
        return Err(NativeMogeError::InvalidCheckpoint(format!(
            "state {key} has invalid F32 storage"
        )));
    }
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        let value = f32::from_ne_bytes(
            chunk
                .try_into()
                .map_err(|_| NativeMogeError::SemanticStateChanged)?,
        );
        if !value.is_finite() {
            return Err(NativeMogeError::InvalidCheckpoint(format!(
                "state {key} contains a non-finite value"
            )));
        }
    }
    Ok(())
}

fn resident_tensor_bytes<'a>(
    maps: impl IntoIterator<Item = &'a BTreeMap<String, Tensor>>,
    cancellation: &CancellationToken,
) -> Result<u64, NativeMogeError> {
    resident_tensor_allocations(maps, cancellation)?
        .into_iter()
        .try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(bytes)
                .ok_or(NativeMogeError::ShapeOverflow)
        })
}

fn resident_tensor_allocations<'a>(
    maps: impl IntoIterator<Item = &'a BTreeMap<String, Tensor>>,
    cancellation: &CancellationToken,
) -> Result<Vec<(StorageId, u64)>, NativeMogeError> {
    let mut allocations = Vec::new();
    for map in maps {
        for (index, tensor) in map.values().enumerate() {
            if index.is_multiple_of(32) {
                cancellation.check()?;
            }
            let storage_id = tensor.storage_id();
            let bytes = tensor.storage_byte_len();
            if let Some((_, existing)) = allocations
                .iter()
                .find(|(existing_id, _)| *existing_id == storage_id)
            {
                if *existing != bytes {
                    return Err(NativeMogeError::SemanticStateChanged);
                }
            } else {
                allocations.push((storage_id, bytes));
            }
        }
    }
    Ok(allocations)
}

fn resident_owned_bytes(
    artifact: &str,
    digest: &str,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
) -> Result<u64, NativeMogeError> {
    let base = u64::try_from(mem::size_of::<NativeMogeResource>())
        .map_err(|_| NativeMogeError::ShapeOverflow)?;
    [artifact.len(), digest.len()]
        .into_iter()
        .chain(source_state.keys().map(String::capacity))
        .chain(execution_state.keys().map(String::capacity))
        .try_fold(base, |total, capacity| {
            total
                .checked_add(u64_from(capacity)?)
                .ok_or(NativeMogeError::ShapeOverflow)
        })
}

fn validate_sha256(value: &str) -> Result<(), NativeMogeError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NativeMogeError::InvalidCheckpoint(
            "artifact identity must be a SHA-256 digest".to_owned(),
        ));
    }
    Ok(())
}

fn validate_state_key(key: &str) -> Result<(), NativeMogeError> {
    if key.is_empty()
        || key.len() > MAX_STATE_KEY_BYTES
        || key.chars().any(char::is_control)
        || key.starts_with('.')
        || key.ends_with('.')
    {
        return Err(NativeMogeError::InvalidCheckpoint(
            "state key is invalid".to_owned(),
        ));
    }
    Ok(())
}

fn integer_square_root(value: usize) -> usize {
    (value as f64).sqrt() as usize
}

fn filled_f32(length: usize, value: f32) -> Result<Vec<f32>, NativeMogeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| NativeMogeError::Allocation)?;
    values.resize(length, value);
    Ok(values)
}

fn usize_from(value: u64) -> Result<usize, NativeMogeError> {
    usize::try_from(value).map_err(|_| NativeMogeError::ShapeOverflow)
}

fn u64_from(value: usize) -> Result<u64, NativeMogeError> {
    u64::try_from(value).map_err(|_| NativeMogeError::ShapeOverflow)
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MogeFixtureProfile {
    V1,
    V2,
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug)]
pub struct MogeFixtureMutation<'a> {
    pub state_key: &'a str,
    pub lane: usize,
    pub delta: f32,
}

#[cfg(any(test, feature = "test-support"))]
pub fn deterministic_reduced_moge_checkpoint(
    backend: &CpuBackend,
    profile: MogeFixtureProfile,
    source_dtype: DType,
    memory_budget_bytes: u64,
    context: &ExecutionContext<'_>,
) -> Result<NativeMogeCheckpoint, NativeMogeError> {
    if !matches!(source_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(NativeMogeError::InvalidCheckpoint(
            "fixture dtype must be F16, BF16, or F32".to_owned(),
        ));
    }
    let configuration = reduced_configuration(profile);
    let mut specifications = dino_backbone(configuration)?
        .state_manifest()?
        .into_iter()
        .map(|specification| StateSpecification {
            key: specification.key,
            shape: specification.shape,
        })
        .collect::<Vec<_>>();
    let dino_state_count = specifications.len();
    add_reduced_head_manifest_version(configuration.version, &mut specifications);
    let mut ordered_state = Vec::new();
    ordered_state
        .try_reserve_exact(specifications.len())
        .map_err(|_| NativeMogeError::Allocation)?;
    for (state_index, specification) in specifications.into_iter().enumerate() {
        context.check()?;
        let elements = specification
            .shape
            .iter()
            .try_fold(1_usize, |total, dimension| {
                total
                    .checked_mul(usize_from(*dimension)?)
                    .ok_or(NativeMogeError::ShapeOverflow)
            })?;
        let mut values = filled_f32(elements, 0.0)?;
        let fixture_state_index = if state_index < dino_state_count {
            reduced_dino_fixture_state_index(&specification.key, state_index)?
        } else {
            state_index
        };
        let norm_weight = specification.key.ends_with(".weight")
            && (specification.key.contains("norm")
                || specification.key.contains("layernorm")
                || specification.key.contains(".layers.0.")
                || specification.key.contains(".layers.3."));
        let layer_scale = specification.key.ends_with(".lambda1");
        for (value_index, value) in values.iter_mut().enumerate() {
            if value_index.is_multiple_of(16_384) {
                context.check()?;
            }
            *value = if norm_weight {
                1.0
            } else if layer_scale {
                0.125
            } else if specification.key.ends_with(".bias") {
                0.0
            } else {
                let lane = ((fixture_state_index * 17 + value_index * 13) % 29) as f32 - 14.0;
                lane * 0.0025
            };
        }
        if specification.key == "native.backbone.embeddings.mask_token" && values.len() == 4 {
            values.copy_from_slice(&[-0.035, -0.0025, 0.03, -0.01]);
        }
        ordered_state.push((
            specification.key,
            tensor_from_f32_with_context_exact_native(
                backend,
                &specification.shape,
                &values,
                source_dtype,
                DeviceId::CPU,
                context,
            )?,
        ));
    }
    Ok(NativeMogeCheckpoint {
        artifact_sha256: match profile {
            MogeFixtureProfile::V1 => {
                "5bd9ce811f140c17a16652703493ee99b0ed0e9f066a960a7181075e0babaf92"
            }
            MogeFixtureProfile::V2 => {
                "52a4211cc705221fc340dfa47bf103c40e0f3f4a4671e95168589b4941c9ee9f"
            }
        }
        .to_owned(),
        ordered_state,
        memory_budget_bytes,
    })
}

#[cfg(any(test, feature = "test-support"))]
fn reduced_dino_fixture_state_index(
    key: &str,
    ordinary_index: usize,
) -> Result<usize, NativeMogeError> {
    if key == "native.backbone.embeddings.mask_token" {
        return Ok(ordinary_index);
    }
    if key.starts_with("native.backbone.layernorm.") {
        return ordinary_index
            .checked_sub(1)
            .and_then(|index| index.checked_add(12))
            .ok_or(NativeMogeError::ShapeOverflow);
    }
    let Some(layer_suffix) = key.strip_prefix("native.backbone.encoder.layer.") else {
        return Ok(ordinary_index);
    };
    let (layer, suffix) = layer_suffix
        .split_once('.')
        .ok_or(NativeMogeError::ShapeOverflow)?;
    let layer = layer
        .parse::<usize>()
        .map_err(|_| NativeMogeError::ShapeOverflow)?;
    let skipped_before_layer = layer.saturating_sub(1).saturating_mul(4);
    let skipped_in_layer = usize::from(
        layer >= 1
            && !suffix.starts_with("norm1.")
            && !suffix.starts_with("attention.attention.")
            && !suffix.starts_with("attention.output.dense."),
    ) * 4;
    ordinary_index
        .checked_sub(1)
        .and_then(|index| index.checked_add(skipped_before_layer))
        .and_then(|index| index.checked_add(skipped_in_layer))
        .ok_or(NativeMogeError::ShapeOverflow)
}

#[cfg(any(test, feature = "test-support"))]
pub fn mutate_reduced_moge_checkpoint(
    backend: &CpuBackend,
    checkpoint: &mut NativeMogeCheckpoint,
    mutation: MogeFixtureMutation<'_>,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeMogeError> {
    let (_, tensor) = checkpoint
        .ordered_state
        .iter_mut()
        .find(|(key, _)| key == mutation.state_key)
        .ok_or_else(|| NativeMogeError::MissingState(mutation.state_key.to_owned()))?;
    let descriptor = tensor.descriptor().clone();
    let mut values = tensor_to_f32_with_context_exact_native(backend, tensor, context)?;
    let value = values
        .get_mut(mutation.lane)
        .ok_or(NativeMogeError::ShapeOverflow)?;
    *value += mutation.delta;
    *tensor = tensor_from_f32_with_context_exact_native(
        backend,
        descriptor.shape(),
        &values,
        descriptor.dtype(),
        descriptor.device(),
        context,
    )?;
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
fn reduced_configuration(profile: MogeFixtureProfile) -> NativeMogeConfiguration {
    NativeMogeConfiguration {
        version: match profile {
            MogeFixtureProfile::V1 => NativeMogeVersion::V1,
            MogeFixtureProfile::V2 => NativeMogeVersion::V2,
        },
        dino_prefix: "native.backbone",
        hidden: 4,
        layer_count: 4,
        attention_heads: 1,
        patch: 2,
        image: 4,
        output_layers: [0, 1, 2, 3],
        has_normal: profile == MogeFixtureProfile::V2,
        source_exact: false,
    }
}

fn add_reduced_head_manifest_version(
    version: NativeMogeVersion,
    states: &mut Vec<StateSpecification>,
) {
    match version {
        NativeMogeVersion::V1 => {
            for index in 0..4 {
                add_reduced_convolution(states, format!("head.projects.{index}"), 4, 4, 1, false);
            }
            add_reduced_convolution(
                states,
                "head.upsample_blocks.0.0.0".to_owned(),
                6,
                4,
                2,
                true,
            );
            add_reduced_convolution(
                states,
                "head.upsample_blocks.0.0.1".to_owned(),
                4,
                4,
                3,
                false,
            );
            add_reduced_residual(states, "head.upsample_blocks.0.1".to_owned(), 4);
            for (index, output) in [(0, 3), (1, 1)] {
                add_reduced_convolution(
                    states,
                    format!("head.output_block.{index}.0"),
                    6,
                    4,
                    3,
                    false,
                );
                add_reduced_convolution(
                    states,
                    format!("head.output_block.{index}.2"),
                    4,
                    output,
                    1,
                    false,
                );
            }
        }
        NativeMogeVersion::V2 => {
            for index in 0..4 {
                add_reduced_convolution(
                    states,
                    format!("encoder.output_projections.{index}"),
                    4,
                    4,
                    1,
                    false,
                );
            }
            for (prefix, final_output) in [
                ("neck", Some(4)),
                ("points_head", Some(3)),
                ("mask_head", Some(1)),
                ("normal_head", Some(3)),
            ] {
                for level in 0..5 {
                    let input = if prefix == "neck" {
                        if level == 0 { 6 } else { 2 }
                    } else {
                        4
                    };
                    add_reduced_convolution(
                        states,
                        format!("{prefix}.input_blocks.{level}"),
                        input,
                        4,
                        1,
                        false,
                    );
                    add_reduced_residual(states, format!("{prefix}.res_blocks.{level}.0"), 4);
                    if level == 4 {
                        add_reduced_convolution(
                            states,
                            format!("{prefix}.output_blocks.{level}"),
                            4,
                            final_output.unwrap_or(4),
                            1,
                            false,
                        );
                    }
                    if level < 4 {
                        add_reduced_convolution(
                            states,
                            format!("{prefix}.resamplers.{level}.1"),
                            4,
                            4,
                            3,
                            false,
                        );
                    }
                }
            }
            add_reduced_specification(states, "scale_head.0.weight".to_owned(), &[4, 4]);
            add_reduced_specification(states, "scale_head.0.bias".to_owned(), &[4]);
            add_reduced_specification(states, "scale_head.2.weight".to_owned(), &[1, 4]);
            add_reduced_specification(states, "scale_head.2.bias".to_owned(), &[1]);
        }
    }
}

fn add_reduced_specification(states: &mut Vec<StateSpecification>, key: String, shape: &[u64]) {
    states.push(StateSpecification {
        key,
        shape: shape.to_vec(),
    });
}

fn add_reduced_convolution(
    states: &mut Vec<StateSpecification>,
    prefix: String,
    input: u64,
    output: u64,
    kernel: u64,
    transposed: bool,
) {
    let shape = if transposed {
        vec![input, output, kernel, kernel]
    } else {
        vec![output, input, kernel, kernel]
    };
    add_reduced_specification(states, format!("{prefix}.weight"), &shape);
    add_reduced_specification(states, format!("{prefix}.bias"), &[output]);
}

fn add_reduced_residual(states: &mut Vec<StateSpecification>, prefix: String, channels: u64) {
    add_reduced_specification(states, format!("{prefix}.layers.0.weight"), &[channels]);
    add_reduced_specification(states, format!("{prefix}.layers.0.bias"), &[channels]);
    add_reduced_convolution(
        states,
        format!("{prefix}.layers.2"),
        channels,
        channels,
        3,
        false,
    );
    add_reduced_specification(states, format!("{prefix}.layers.3.weight"), &[channels]);
    add_reduced_specification(states, format!("{prefix}.layers.3.bias"), &[channels]);
    add_reduced_convolution(
        states,
        format!("{prefix}.layers.5"),
        channels,
        channels,
        3,
        false,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, StreamId};

    #[test]
    fn source_f32_fov_observations_use_canonical_owners_and_release_workspace()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(bytes)?;
        let cancellation = CancellationToken::default();
        let authorization = authority.authorize_workspace(bytes)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
        for (width, height, degrees, expected) in [
            (8u32, 8u32, 0x3dcccccdu32, 0x444a9234u32),
            (8u32, 8u32, 0x3f333333u32, 0x42e781cdu32),
            (8u32, 8u32, 0x3f800000u32, 0x42a20d85u32),
            (8u32, 8u32, 0x418a6666u32, 0x4094bd24u32),
            (8u32, 8u32, 0x41f00000u32, 0x4028e4b2u32),
            (8u32, 8u32, 0x42340000u32, 0x3fda8279u32),
            (8u32, 8u32, 0x42700000u32, 0x3f9cc471u32),
            (8u32, 8u32, 0x42b3cccdu32, 0x3f3555e7u32),
            (8u32, 8u32, 0x42b40000u32, 0x3f3504f3u32),
            (8u32, 8u32, 0x42f00000u32, 0x3ed105ebu32),
            (8u32, 8u32, 0x4329e666u32, 0x3d7ff0f9u32),
            (8u32, 8u32, 0x432a0000u32, 0x3d7d64f7u32),
            (17u32, 9u32, 0x3dcccccdu32, 0x447d2fc0u32),
            (17u32, 9u32, 0x3f333333u32, 0x4310ad1fu32),
            (17u32, 9u32, 0x3f800000u32, 0x42ca8b4cu32),
            (17u32, 9u32, 0x418a6666u32, 0x40b9e748u32),
            (17u32, 9u32, 0x41f00000u32, 0x40531807u32),
            (17u32, 9u32, 0x42340000u32, 0x40088dc4u32),
            (17u32, 9u32, 0x42700000u32, 0x3fc3f021u32),
            (17u32, 9u32, 0x42b3cccdu32, 0x3f62a51bu32),
            (17u32, 9u32, 0x42b40000u32, 0x3f623fedu32),
            (17u32, 9u32, 0x42f00000u32, 0x3f02a016u32),
            (17u32, 9u32, 0x4329e666u32, 0x3d9ff22eu32),
            (17u32, 9u32, 0x432a0000u32, 0x3d9e5ab8u32),
            (3u32, 5u32, 0x3dcccccdu32, 0x44136463u32),
            (3u32, 5u32, 0x3f333333u32, 0x42a87234u32),
            (3u32, 5u32, 0x3f800000u32, 0x426bd24cu32),
            (3u32, 5u32, 0x418a6666u32, 0x40587261u32),
            (3u32, 5u32, 0x41f00000u32, 0x3ff5c6a2u32),
            (3u32, 5u32, 0x42340000u32, 0x3f9efd37u32),
            (3u32, 5u32, 0x42700000u32, 0x3f642147u32),
            (3u32, 5u32, 0x42b3cccdu32, 0x3f03f0e5u32),
            (3u32, 5u32, 0x42b40000u32, 0x3f03b5feu32),
            (3u32, 5u32, 0x42f00000u32, 0x3e98162fu32),
            (3u32, 5u32, 0x4329e666u32, 0x3d3a3982u32),
            (3u32, 5u32, 0x432a0000u32, 0x3d385f1au32),
        ] {
            let actual = focal_from_fov_degrees(
                &backend,
                f32::from_bits(degrees),
                f64::from(width) / f64::from(height),
                &context,
            )?;
            assert_eq!(
                actual.to_bits(),
                expected,
                "{width}x{height}, degrees bits {degrees:08x}"
            );
            assert_eq!(authorization.in_use_bytes(), 0);
        }
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let authorization = authority.authorize_workspace(0)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancelled);
        assert!(focal_from_fov_degrees(&backend, 60.0, 1.0, &context).is_err());
        assert_eq!(authorization.peak_bytes(), 0);
        assert_eq!(authorization.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn reduced_moge_versions_execute_through_the_shared_dino_owner()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = 256 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &cancellation,
        );
        let values = (0..48)
            .map(|index| (index as f32 + 1.0) / 64.0)
            .collect::<Vec<_>>();
        let image = ImageTensor::from_f32(&backend, &context, 1, 4, 4, 3, &values)?;
        for profile in [MogeFixtureProfile::V1, MogeFixtureProfile::V2] {
            let resource = NativeMogeResource::from_reduced_fixture(
                &backend,
                deterministic_reduced_moge_checkpoint(
                    &backend,
                    profile,
                    DType::F32,
                    bytes,
                    &context,
                )?,
                &context,
            )?;
            let geometry = resource.execute(
                &backend,
                NativeMogeInvocation {
                    image: &image,
                    resolution_level: 9,
                    fov_x_degrees: None,
                    force_projection: true,
                    apply_mask: false,
                    apply_metric_scale: true,
                },
                &context,
            )?;
            assert_eq!(
                geometry.points().map(|value| value.descriptor().shape()),
                Some([1, 4, 4, 3].as_slice())
            );
            assert_eq!(
                geometry.depth().map(|value| value.descriptor().shape()),
                Some([1, 4, 4].as_slice())
            );
            assert_eq!(
                geometry.normal().is_some(),
                profile == MogeFixtureProfile::V2
            );
            resource.validate(&cancellation)?;
            assert_eq!(
                resource
                    .reconstruct_checkpoint(&cancellation)?
                    .ordered_state
                    .len(),
                resource.source_state.len()
            );
            let foreign_context = backend.execution_context(
                StreamId::new(7),
                authority.authorize_workspace(bytes)?,
                &cancellation,
            );
            let foreign_image =
                ImageTensor::from_f32(&backend, &foreign_context, 1, 4, 4, 3, &values)?;
            assert!(matches!(
                resource.execute(
                    &backend,
                    NativeMogeInvocation {
                        image: &foreign_image,
                        resolution_level: 9,
                        fov_x_degrees: None,
                        force_projection: true,
                        apply_mask: false,
                        apply_metric_scale: true,
                    },
                    &context,
                ),
                Err(NativeMogeError::InvalidImage(message)) if message.contains("foreign execution stream")
            ));
        }
        Ok(())
    }

    #[test]
    fn geometry_projection_preserves_shift_scale_and_boolean_mask_contract()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = 16 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &cancellation,
        );
        let image = ImageTensor::from_f32(&backend, &context, 1, 2, 2, 3, &[0.5; 12])?;
        let raw = RawMogeOutput {
            points: vec![
                -0.25, -0.25, 1.0, 0.25, -0.25, 1.0, -0.25, 0.25, 1.0, 0.25, 0.25, 1.0,
            ],
            mask: vec![1.0; 4],
            normal: None,
            metric_scale: Some(vec![2.0]),
        };
        let invocation = |fov_x_degrees, apply_metric_scale| NativeMogeInvocation {
            image: &image,
            resolution_level: 9,
            fov_x_degrees,
            force_projection: false,
            apply_mask: false,
            apply_metric_scale,
        };
        let unscaled = project_geometry(
            &backend,
            invocation(None, false),
            NativeMogeVersion::V1,
            raw.clone(),
            1,
            2,
            2,
            &context,
        )?;
        let zero_fov = project_geometry(
            &backend,
            invocation(Some(0.0), false),
            NativeMogeVersion::V1,
            raw.clone(),
            1,
            2,
            2,
            &context,
        )?;
        let scaled = project_geometry(
            &backend,
            invocation(None, true),
            NativeMogeVersion::V1,
            raw,
            1,
            2,
            2,
            &context,
        )?;
        let values = |tensor: Option<&Tensor>| {
            tensor_to_f32_with_context_exact_native(
                &backend,
                tensor.ok_or(NativeMogeError::InvalidGeometryProjection)?,
                &context,
            )
            .map_err(NativeMogeError::from)
        };
        let unscaled_points = values(unscaled.points())?;
        let zero_fov_points = values(zero_fov.points())?;
        let scaled_points = values(scaled.points())?;
        assert_eq!(zero_fov_points, unscaled_points);
        for (scaled, unscaled) in scaled_points.iter().zip(&unscaled_points) {
            assert_eq!(*scaled, *unscaled * 2.0);
        }
        assert_eq!(
            scaled.mask().map(|tensor| tensor.descriptor().dtype()),
            Some(DType::Bool)
        );
        assert_eq!(values(unscaled.depth())?, vec![unscaled_points[2]; 4]);

        let signed_depth = RawMogeOutput {
            points: vec![0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            mask: vec![1.0; 4],
            normal: None,
            metric_scale: None,
        };
        let masked_invocation = NativeMogeInvocation {
            image: &image,
            resolution_level: 9,
            fov_x_degrees: Some(60.0),
            force_projection: false,
            apply_mask: true,
            apply_metric_scale: false,
        };
        let v1_masked = project_geometry(
            &backend,
            masked_invocation.clone(),
            NativeMogeVersion::V1,
            signed_depth.clone(),
            1,
            2,
            2,
            &context,
        )?;
        let v2_masked = project_geometry(
            &backend,
            masked_invocation,
            NativeMogeVersion::V2,
            signed_depth,
            1,
            2,
            2,
            &context,
        )?;
        assert!(values(v1_masked.points())?.into_iter().all(f32::is_finite));
        assert!(
            values(v2_masked.points())?[..3]
                .iter()
                .all(|value| value.is_infinite())
        );
        assert_eq!(values(v1_masked.mask())?, vec![1.0; 4]);
        assert_eq!(values(v2_masked.mask())?, vec![0.0, 1.0, 1.0, 1.0]);

        let late_cancellation = CancellationToken::default();
        let late_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &late_cancellation,
        );
        assert!(late_cancellation.cancel());
        let unpublished = publish_geometry(
            image.clone(),
            v1_masked
                .points()
                .cloned()
                .ok_or(NativeMogeError::InvalidGeometryProjection)?,
            v1_masked
                .depth()
                .cloned()
                .ok_or(NativeMogeError::InvalidGeometryProjection)?,
            v1_masked
                .intrinsics()
                .cloned()
                .ok_or(NativeMogeError::InvalidGeometryProjection)?,
            v1_masked
                .mask()
                .cloned()
                .ok_or(NativeMogeError::InvalidGeometryProjection)?,
            None,
            &late_context,
        );
        assert!(matches!(
            unpublished,
            Err(NativeMogeError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn normal_projection_uses_the_source_epsilon_boundary() -> Result<(), Box<dyn std::error::Error>>
    {
        let bytes = 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &cancellation,
        );
        let points = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 3, 1, 3],
            &[0.0; 9],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let mask = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 1, 1, 3],
            &[1.0; 3],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let normal = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 3, 1, 3],
            &[0.0, 5.0e-13, 2.0e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let raw = raw_from_nchw(
            &backend,
            &points,
            &mask,
            Some(&normal),
            false,
            None,
            &context,
        )?;
        assert_eq!(
            raw.normal,
            Some(vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0])
        );
        let overflowing_points = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 3, 1, 1],
            &[0.0, 0.0, 1000.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let one_mask = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 1, 1, 1],
            &[1.0],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        assert!(matches!(
            raw_from_nchw(
                &backend,
                &overflowing_points,
                &one_mask,
                None,
                false,
                None,
                &context,
            ),
            Err(NativeMogeError::InvalidGeometryProjection)
        ));
        Ok(())
    }

    #[test]
    fn checkpoint_and_execution_memory_boundaries_fail_before_work()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = 256 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &cancellation,
        );
        let checkpoint = deterministic_reduced_moge_checkpoint(
            &backend,
            MogeFixtureProfile::V2,
            DType::F16,
            bytes,
            &context,
        )?;
        let mut constrained = checkpoint.clone();
        constrained.memory_budget_bytes = 1;
        let required =
            match NativeMogeResource::from_reduced_fixture(&backend, constrained, &context) {
                Err(NativeMogeError::OutOfMemory {
                    required,
                    budget: 1,
                }) => required,
                result => {
                    return Err(
                        format!("unexpected constrained checkpoint result: {result:?}").into(),
                    );
                }
            };
        assert!(required > 1);
        let mut resource =
            NativeMogeResource::from_reduced_fixture(&backend, checkpoint, &context)?;
        resource.memory_budget_bytes = 1;
        let execution_required = match preflight_execution_memory(&resource, 1, 4, 4, 9) {
            Err(NativeMogeError::OutOfMemory {
                required,
                budget: 1,
            }) => required,
            result => {
                return Err(format!("unexpected execution preflight result: {result:?}").into());
            }
        };
        resource.memory_budget_bytes = execution_required - 1;
        assert!(matches!(
            preflight_execution_memory(&resource, 1, 4, 4, 9),
            Err(NativeMogeError::OutOfMemory { .. })
        ));
        resource.memory_budget_bytes = execution_required;
        preflight_execution_memory(&resource, 1, 4, 4, 9)?;

        let fused_qkv = tensor_from_f32_with_context_exact_native(
            &backend,
            &[6, 2],
            &[0.0; 12],
            DType::F16,
            DeviceId::CPU,
            &context,
        )?;
        let fused_state = vec![("encoder.blocks.0.attn.qkv.weight".to_owned(), fused_qkv)];
        let artifact = "b".repeat(64);
        let fused_required = match raw_ordered_state_preflight(&artifact, &fused_state, 1, &context)
        {
            Err(NativeMogeError::OutOfMemory {
                required,
                budget: 1,
            }) => required,
            result => {
                return Err(format!("unexpected fused-QKV preflight result: {result:?}").into());
            }
        };
        assert!(matches!(
            raw_ordered_state_preflight(&artifact, &fused_state, fused_required - 1, &context,),
            Err(NativeMogeError::OutOfMemory { .. })
        ));
        raw_ordered_state_preflight(&artifact, &fused_state, fused_required, &context)?;
        Ok(())
    }

    #[test]
    fn checkpoint_admission_rejects_malformed_state_atomically()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = 256 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &cancellation,
        );
        let checkpoint = deterministic_reduced_moge_checkpoint(
            &backend,
            MogeFixtureProfile::V2,
            DType::F32,
            bytes,
            &context,
        )?;

        let mut missing = checkpoint.clone();
        missing.ordered_state.pop();
        assert!(NativeMogeResource::from_reduced_fixture(&backend, missing, &context).is_err());

        let mut unexpected = checkpoint.clone();
        let first_state = checkpoint
            .ordered_state
            .first()
            .ok_or(NativeMogeError::UnsupportedArchitecture)?;
        unexpected
            .ordered_state
            .push(("unexpected.weight".to_owned(), first_state.1.clone()));
        assert!(matches!(
            NativeMogeResource::from_reduced_fixture(&backend, unexpected, &context),
            Err(NativeMogeError::InvalidCheckpoint(message))
                if message.contains("cardinality")
        ));

        let mut duplicate = checkpoint.clone();
        duplicate.ordered_state.push(first_state.clone());
        assert!(matches!(
            NativeMogeResource::from_reduced_fixture(&backend, duplicate, &context),
            Err(NativeMogeError::DuplicateStateKey(_))
        ));

        let replace_first = |checkpoint: &mut NativeMogeCheckpoint,
                             dtype: DType,
                             make_nan: bool|
         -> Result<(), NativeMogeError> {
            let (_, tensor) = checkpoint
                .ordered_state
                .first_mut()
                .ok_or(NativeMogeError::UnsupportedArchitecture)?;
            let descriptor = tensor.descriptor().clone();
            let mut values = tensor_to_f32_with_context_exact_native(&backend, tensor, &context)?;
            if make_nan {
                *values
                    .first_mut()
                    .ok_or(NativeMogeError::UnsupportedArchitecture)? = f32::NAN;
            }
            *tensor = tensor_from_f32_with_context_exact_native(
                &backend,
                descriptor.shape(),
                &values,
                dtype,
                DeviceId::CPU,
                &context,
            )?;
            Ok(())
        };
        let mut mixed = checkpoint.clone();
        replace_first(&mut mixed, DType::F16, false)?;
        assert!(matches!(
            NativeMogeResource::from_reduced_fixture(&backend, mixed, &context),
            Err(NativeMogeError::InvalidCheckpoint(message))
                if message.contains("mixed checkpoint dtypes")
        ));

        let mut non_finite = checkpoint.clone();
        replace_first(&mut non_finite, DType::F32, true)?;
        assert!(matches!(
            NativeMogeResource::from_reduced_fixture(&backend, non_finite, &context),
            Err(NativeMogeError::InvalidCheckpoint(message))
                if message.contains("non-finite")
        ));

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &cancelled,
        );
        assert!(matches!(
            NativeMogeResource::from_reduced_fixture(&backend, checkpoint, &cancelled_context,),
            Err(NativeMogeError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn nested_checkpoint_configuration_is_presence_only_and_sha_is_lowercase()
    -> Result<(), Box<dyn std::error::Error>> {
        let checkpoint = canonical_flat_checkpoint(NativeMogeNestedCheckpoint {
            artifact_sha256: "a".repeat(64),
            model: Vec::new(),
            model_config: serde_json::Map::from_iter([(
                String::new(),
                serde_json::Value::String("ignored by the pinned source".to_owned()),
            )]),
            memory_budget_bytes: 7,
        });
        assert_eq!(checkpoint.artifact_sha256, "a".repeat(64));
        assert!(checkpoint.ordered_state.is_empty());
        assert_eq!(checkpoint.memory_budget_bytes, 7);
        validate_sha256(&checkpoint.artifact_sha256)?;
        assert!(validate_sha256(&"A".repeat(64)).is_err());
        Ok(())
    }

    #[test]
    fn production_preprocessing_uses_the_versioned_source_choreography()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = 256 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &cancellation,
        );
        let values = (0..45)
            .map(|index| (index as f32 + 1.0) / 64.0)
            .collect::<Vec<_>>();
        let mut nchw = vec![0.0; values.len()];
        for y in 0..3 {
            for x in 0..5 {
                for channel in 0..3 {
                    nchw[(channel * 3 + y) * 5 + x] = values[(y * 5 + x) * 3 + channel];
                }
            }
        }
        let source = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 3, 3, 5],
            &nchw,
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        for profile in [MogeFixtureProfile::V1, MogeFixtureProfile::V2] {
            let mut resource = NativeMogeResource::from_reduced_fixture(
                &backend,
                deterministic_reduced_moge_checkpoint(
                    &backend,
                    profile,
                    DType::F32,
                    bytes,
                    &context,
                )?,
                &context,
            )?;
            resource.configuration.source_exact = true;
            let prefix = if profile == MogeFixtureProfile::V1 {
                ""
            } else {
                "encoder."
            };
            for (name, buffer) in [
                ("image_mean", IMAGE_MEAN.as_slice()),
                ("image_std", IMAGE_STANDARD_DEVIATION.as_slice()),
            ] {
                resource.execution_state.insert(
                    format!("{prefix}{name}"),
                    tensor_from_f32_with_context_exact_native(
                        &backend,
                        &[1, 3, 1, 1],
                        buffer,
                        DType::F32,
                        DeviceId::CPU,
                        &context,
                    )?,
                );
            }
            let actual = prepare_image(
                &backend, &resource, &values, 1, 3, 5, 3, 4, 6, 5, 7, &context,
            )?;
            let (mut expected, normalized_height, normalized_width) =
                if profile == MogeFixtureProfile::V1 {
                    (
                        interpolate_tensor_with_context_exact_native(
                            &backend,
                            &source,
                            &InterpolateConfiguration {
                                output_size: Some(vec![5, 7]),
                                scale_factor: None,
                                mode: InterpolateMode::Bicubic,
                                align_corners: Some(false),
                                recompute_scale_factor: None,
                                antialias: true,
                            },
                            &context,
                        )?,
                        5,
                        7,
                    )
                } else {
                    (resize_nchw(&backend, &source, 4, 6, true, &context)?, 4, 6)
                };
            let mut expected_values =
                tensor_to_f32_with_context_exact_native(&backend, &expected, &context)?;
            for channel in 0..3 {
                for pixel in 0..normalized_height * normalized_width {
                    let index = channel * normalized_height * normalized_width + pixel;
                    expected_values[index] = (expected_values[index] - IMAGE_MEAN[channel])
                        / IMAGE_STANDARD_DEVIATION[channel];
                }
            }
            expected = tensor_from_f32_with_context_exact_native(
                &backend,
                &[
                    1,
                    3,
                    u64_from(normalized_height)?,
                    u64_from(normalized_width)?,
                ],
                &expected_values,
                DType::F32,
                DeviceId::CPU,
                &context,
            )?;
            if profile == MogeFixtureProfile::V1 {
                expected = resize_nchw(&backend, &expected, 4, 6, true, &context)?;
            }
            assert_eq!(actual.descriptor(), expected.descriptor());
            assert_eq!(actual.contiguous_bytes()?, expected.contiguous_bytes()?);
        }
        for profile in [MogeFixtureProfile::V1, MogeFixtureProfile::V2] {
            let mut configuration = reduced_configuration(profile);
            configuration.source_exact = true;
            assert!(matches!(
                preprocessing_dimensions(configuration, 1, usize::MAX, 9),
                Err(NativeMogeError::InvalidImage(message))
                    if message.contains("empty patch grid")
            ));
        }
        Ok(())
    }

    #[test]
    fn source_lm_geometry_discriminators_are_within_the_declared_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../sim_test_support/fixtures/models/moge-resource-foundation/oracle.json"
        ))?;
        let expected = &oracle["geometry_discriminators"];
        let tolerance = expected["absolute_tolerance"]
            .as_f64()
            .ok_or("missing MoGe LM tolerance")? as f32;
        let max_plus_one = expected["max_plus_one"]
            .as_f64()
            .ok_or("missing MoGe LM max+1")? as f32;
        assert!(max_plus_one > tolerance);
        let source_solver = expected["auto_original_scipy_receipt"]
            .as_array()
            .and_then(|receipts| receipts.first())
            .ok_or("missing original SciPy camera receipt")?;
        assert!(
            source_solver["nfev"]
                .as_u64()
                .ok_or("missing original SciPy evaluations")?
                > 0
        );
        let bytes = 16 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            &cancellation,
        );
        let width = 8_usize;
        let height = 8_usize;
        let diagonal = (width as f64).hypot(height as f64);
        let mut points = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let pixel = y * width + x;
                let z = 1.0 + pixel as f64 * 0.01;
                let u = ((x as f64 + 0.5) * 2.0 - width as f64) / diagonal;
                let v = ((y as f64 + 0.5) * 2.0 - height as f64) / diagonal;
                points.extend([
                    (u * (z + 0.25) / 0.8) as f32,
                    (v * (z + 0.25) / 0.8) as f32,
                    z as f32,
                ]);
            }
        }
        let confidence = vec![1.0; width * height];
        let (focal, shift) = recover_focal_shift(
            &backend,
            &points,
            &confidence,
            width,
            height,
            None,
            &context,
        )?;
        let expected_focal = expected["auto_focal"]
            .as_f64()
            .ok_or("missing auto focal")? as f32;
        let expected_shift = expected["auto_shift"]
            .as_f64()
            .ok_or("missing auto shift")? as f32;
        assert!((focal - expected_focal).abs() <= tolerance);
        assert!((shift - expected_shift).abs() <= tolerance);
        assert!((focal - (expected_focal + max_plus_one)).abs() > tolerance);
        let recovery_grid_points = vec![0.0; 64 * 64 * 3];
        let sparse = [vec![1.0], vec![0.0; 64 * 64 - 1]].concat();
        let sparse_recovery = recover_focal_shift(
            &backend,
            &recovery_grid_points,
            &sparse,
            64,
            64,
            None,
            &context,
        )?;
        assert_eq!(
            sparse_recovery,
            (
                expected["less_than_two_focal"]
                    .as_f64()
                    .ok_or("missing sparse focal disposition")? as f32,
                expected["less_than_two_shift"]
                    .as_f64()
                    .ok_or("missing sparse shift disposition")? as f32,
            )
        );
        let negative = points
            .iter()
            .enumerate()
            .map(|(index, value)| if index % 3 == 2 { *value } else { -*value })
            .collect::<Vec<_>>();
        let (invalid_focal, _) = recover_focal_shift(
            &backend,
            &negative,
            &confidence,
            width,
            height,
            None,
            &context,
        )?;
        assert!(invalid_focal <= 0.0);
        let (fallback_focal, fallback_shift) = recover_focal_shift(
            &backend,
            &negative,
            &confidence,
            width,
            height,
            Some(60.0),
            &context,
        )?;
        let expected_fallback_focal = expected["fallback_focal"]
            .as_f64()
            .ok_or("missing fallback focal")?;
        assert!(
            (f64::from(fallback_focal) - expected_fallback_focal).abs() <= f64::from(tolerance)
        );
        let expected_fallback_shift = expected["fallback_shift"]
            .as_f64()
            .ok_or("missing fallback shift")?;
        let fallback_relative_tolerance = expected["fallback_relative_tolerance"]
            .as_f64()
            .ok_or("missing fallback relative tolerance")?;
        let fallback_relative_max_plus_one = expected["fallback_relative_max_plus_one"]
            .as_f64()
            .ok_or("missing fallback relative max+1")?;
        assert!(fallback_relative_max_plus_one > fallback_relative_tolerance);
        let fallback_bound =
            f64::from(tolerance).max(expected_fallback_shift.abs() * fallback_relative_tolerance);
        let fallback_difference = (f64::from(fallback_shift) - expected_fallback_shift).abs();
        assert!(
            fallback_difference <= fallback_bound,
            "fallback bounded-LM shift {fallback_shift} differs from the finite-difference source oracle {expected_fallback_shift} by {fallback_difference}, exceeding {fallback_bound}"
        );
        // The absolute floor dominates this source value, so the rejection
        // control must exceed both unchanged tolerance components.
        let outside_fallback = expected_fallback_shift
            + f64::from(max_plus_one)
                .max(expected_fallback_shift.abs() * fallback_relative_max_plus_one);
        assert!(
            (outside_fallback - expected_fallback_shift).abs() > fallback_bound,
            "fallback relative max+1 discriminator did not exceed {fallback_bound}"
        );
        Ok(())
    }
}
