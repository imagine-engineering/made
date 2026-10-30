use crate::{LayerQuantizationV1, QuantizationKind, QuantizationMetadataV1, QuantizedMatrix};
use crate::{
    QuantLinearExecution, QuantLinearLayout, QuantLinearOptions, QuantLinearWeight,
    quant_linear_forward_exact_native,
};
use sha2::{Digest, Sha256};
pub use sim_tensor::generated_activation_normalization_functional_01::GeluApproximation;
pub use sim_tensor::generated_neural_network_functional_01::EmbeddingOptions;
pub use sim_tensor::generated_neural_network_module_01::{LossReduction, UpsampleMode};
use sim_tensor::{
    BackendCapabilityMatrix, BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec,
    DType, DecodedScalar, DeviceId, ExecutionContext, Layout, LinearAlgebraOperation,
    OperationSupport, ReductionOperation, ResizeMode, StorageId, StreamId, Tensor, TensorBackend,
    TensorError, UnaryOperation,
    generated_activation_normalization_functional_01::{
        FunctionalError, group_norm_with_context_exact_native,
        layer_norm_with_context_exact_native, leaky_relu_tensor_with_context_exact_native,
        silu_tensor_with_context_exact_native,
    },
    generated_neural_network_module_01::{
        NeuralNetworkModuleError, average_pool_2d_with_context_exact_native,
        prelu_with_context_exact_native, silu_module_with_context_exact_native,
        smooth_l1_loss_with_context_exact_native, softmax_module_with_context_exact_native,
        tanh_module_with_context_exact_native, upsample_with_context_exact_native,
    },
    generated_neural_network_module_02::{
        BATCH_NORM_1D_OPERATION_ID, BATCH_NORM_2D_OPERATION_ID, NeuralNetworkModulePartTwoError,
        adaptive_average_pool_2d_module_with_context_exact_native,
        average_pool_3d_module_with_context_exact_native,
        batch_norm_module_with_context_exact_native, embedding_module_with_context_exact_native,
        huber_loss_with_context_exact_native, instance_norm_2d_tensor_with_context_exact_native,
        instance_norm_2d_with_context_exact_native, leaky_relu_module_with_context_exact_native,
        multihead_attention_projected_with_context_exact_native,
        replication_pad_2d_with_context_exact_native,
    },
    generated_neural_network_module_03::{
        NeuralNetworkModulePartThreeError, gelu_module_with_context_exact_native,
        l1_loss_with_context_exact_native, max_pool_2d_with_context_exact_native,
        pixel_shuffle_module_with_context_exact_native,
        pixel_unshuffle_module_with_context_exact_native, relu_6_with_context_exact_native,
        relu_module_with_context_exact_native, zero_pad_2d_with_context_exact_native,
    },
    generated_neural_network_module_04::{
        NeuralNetworkModulePartFourError, average_pool_1d_with_context_exact_native,
        dropout_with_context_exact_native, elu_module_with_context_exact_native,
        identity_with_context_exact_native, mse_loss_with_context_exact_native,
        sigmoid_module_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionPaddingMode, OperatorIndirectionError, TensorValues,
        cast_to_with_backend_exact_native, cast_to_with_context_exact_native,
        convolution_with_context_exact_native, linear_with_context_exact_native,
        tensor_from_f32_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        ConvolutionConfiguration, SpatialFunctionalKernelError,
        conv_2d_tensor_with_context_exact_native,
        conv_transpose_2d_tensor_with_context_exact_native,
    },
    rng::RngTransaction,
};
use sim_types::DeviceKind;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NativeOpsError {
    #[error(transparent)]
    Tensor(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error(transparent)]
    Quantization(#[from] crate::QuantizationError),
    #[error("{0}")]
    Workspace(
        #[from]
        #[source]
        TensorError,
    ),
    #[error(transparent)]
    Module(NeuralNetworkModuleError),
    #[error(transparent)]
    ModulePartTwo(NeuralNetworkModulePartTwoError),
    #[error("native module parameter is invalid: {0}")]
    Invalid(&'static str),
    #[error("native module parameter is invalid: {0}")]
    InvalidOwned(String),
    #[error("native module parameters have not been loaded")]
    ParametersNotLoaded,
    #[error("native module cast lease {generation} has already been completed")]
    LeaseAlreadyCompleted { generation: u64 },
    #[error("native module generation overflowed")]
    GenerationOverflow,
    #[error("native module resident byte accounting overflowed")]
    ResidentBytesOverflow,
    #[error("native module operation is unavailable on device {device:?}")]
    UnsupportedDevice { device: DeviceId },
    #[error(
        "native execution requested device {requested:?}, but the selected backend owns {backend:?}"
    )]
    BackendTargetMismatch {
        requested: DeviceId,
        backend: DeviceId,
    },
    #[error("native backend {backend:?} advertised a stale capability matrix for {capabilities:?}")]
    StaleBackendCapabilities {
        backend: DeviceId,
        capabilities: DeviceId,
    },
    #[error(
        "native execution requested stream {requested:?}, but the caller context owns {context:?}"
    )]
    ExecutionStreamMismatch {
        requested: StreamId,
        context: StreamId,
    },
    #[error(
        "native execution requirement uses dtype {requirement:?}, but the requested target uses {requested:?}"
    )]
    ExecutionDTypeMismatch {
        requested: DType,
        requirement: DType,
    },
    #[error(
        "native execution requirement uses layout {requirement:?}, but the requested target uses {requested:?}"
    )]
    ExecutionLayoutMismatch {
        requested: Layout,
        requirement: Layout,
    },
    #[error("native module operation was cancelled")]
    Cancelled,
}

impl From<SpatialFunctionalKernelError> for NativeOpsError {
    fn from(error: SpatialFunctionalKernelError) -> Self {
        match error {
            SpatialFunctionalKernelError::Cancelled => Self::Cancelled,
            SpatialFunctionalKernelError::Tensor(error) => Self::Workspace(error),
            error => Self::InvalidOwned(error.to_string()),
        }
    }
}

impl From<NeuralNetworkModulePartThreeError> for NativeOpsError {
    fn from(error: NeuralNetworkModulePartThreeError) -> Self {
        match error {
            NeuralNetworkModulePartThreeError::Cancelled => Self::Cancelled,
            error => Self::InvalidOwned(error.to_string()),
        }
    }
}

impl From<NeuralNetworkModulePartFourError> for NativeOpsError {
    fn from(error: NeuralNetworkModulePartFourError) -> Self {
        match error {
            NeuralNetworkModulePartFourError::Cancelled => Self::Cancelled,
            error => Self::InvalidOwned(error.to_string()),
        }
    }
}

impl From<sim_types::CancellationError> for NativeOpsError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<NeuralNetworkModuleError> for NativeOpsError {
    fn from(error: NeuralNetworkModuleError) -> Self {
        match error {
            NeuralNetworkModuleError::Cancelled => Self::Cancelled,
            error => Self::Module(error),
        }
    }
}

impl From<NeuralNetworkModulePartTwoError> for NativeOpsError {
    fn from(error: NeuralNetworkModulePartTwoError) -> Self {
        match error {
            NeuralNetworkModulePartTwoError::Cancelled => Self::Cancelled,
            error => Self::ModulePartTwo(error),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NativeModuleSpec {
    Container,
    AveragePool1d {
        kernel_size: usize,
        stride: usize,
    },
    AveragePool2d {
        kernel_size: [usize; 2],
        stride: [usize; 2],
    },
    AdaptiveAveragePool2d {
        output_size: [usize; 2],
    },
    AveragePool3d {
        kernel_size: [usize; 3],
        stride: [usize; 3],
    },
    BatchNorm {
        dimensions: usize,
        features: usize,
        epsilon: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
        training: bool,
    },
    Buffer,
    Dropout {
        probability: f32,
        training: bool,
    },
    Elu {
        alpha: f32,
    },
    Linear {
        input_features: usize,
        output_features: usize,
        bias: bool,
    },
    Convolution {
        input_channels: usize,
        output_channels: usize,
        kernel_shape: Vec<usize>,
        bias: bool,
        geometry: ConvolutionGeometry,
    },
    Gelu {
        approximation: GeluApproximation,
    },
    LayerNorm {
        normalized_shape: Vec<usize>,
        epsilon: f32,
        elementwise_affine: bool,
        bias: bool,
    },
    GroupNorm {
        groups: usize,
        channels: usize,
        epsilon: f32,
        affine: bool,
    },
    Embedding {
        embeddings: usize,
        dimensions: usize,
        options: EmbeddingOptions,
    },
    HuberLoss {
        delta: f32,
        reduction: LossReduction,
    },
    Identity,
    L1Loss {
        reduction: LossReduction,
    },
    InstanceNorm2d {
        features: usize,
        epsilon: f32,
        affine: bool,
    },
    LeakyRelu {
        negative_slope: f32,
    },
    MultiheadAttention {
        embed_dimension: usize,
        heads: usize,
    },
    MaxPool2d {
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    },
    MseLoss {
        reduction: LossReduction,
    },
    ModuleDict,
    ModuleList,
    PRelu {
        num_parameters: usize,
    },
    ReplicationPad2d {
        padding: [usize; 4],
    },
    PixelShuffle {
        factor: u64,
    },
    PixelUnshuffle {
        factor: u64,
    },
    Relu,
    Relu6,
    Sequential,
    Silu,
    Sigmoid,
    SmoothL1Loss {
        beta: f32,
        reduction: LossReduction,
    },
    Softmax {
        dimension: isize,
    },
    Tanh,
    Upsample {
        scale_factor: [f64; 2],
        mode: UpsampleMode,
        align_corners: Option<bool>,
    },
    ZeroPad2d {
        padding: [usize; 4],
    },
}

#[derive(Clone, Debug)]
enum NativeWeight {
    Dense(Tensor),
    Quantized(QuantizedMatrix),
    WeightNorm(NativeWeightNorm),
    SpectralNorm(NativeSpectralNorm),
}

#[derive(Clone, Debug)]
struct NativeWeightNorm {
    magnitude: Tensor,
    direction: Tensor,
    dimension: Option<usize>,
}

#[derive(Clone, Debug)]
struct NativeSpectralNormConfig {
    dimension: usize,
    power_iterations: usize,
    epsilon: f32,
}

#[derive(Clone, Debug)]
struct NativeSpectralNorm {
    original: Tensor,
    config: NativeSpectralNormConfig,
    left: Option<Vec<f32>>,
    right: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct NativeNormalizationState {
    running_mean: Vec<f32>,
    running_variance: Vec<f32>,
}

#[derive(Clone, Debug)]
struct PrefetchedParameters {
    weight: Tensor,
    bias: Option<Tensor>,
    requantized_weight: Option<QuantizedMatrix>,
    dtype: DType,
    bias_dtype: DType,
    device: DeviceId,
    next_weight: Option<NativeWeight>,
}

fn tensor_resident_bytes(tensor: &Tensor, storages: &mut HashSet<StorageId>) -> u64 {
    if storages.insert(tensor.storage_id()) {
        tensor.storage_byte_len()
    } else {
        0
    }
}

fn quantized_resident_bytes(matrix: &QuantizedMatrix) -> Result<u64, NativeOpsError> {
    matrix
        .resident_storage_bytes()
        .map_err(|_| NativeOpsError::ResidentBytesOverflow)
}

fn native_weight_resident_bytes(
    weight: &NativeWeight,
    tensor_storages: &mut HashSet<StorageId>,
) -> Result<u64, NativeOpsError> {
    match weight {
        NativeWeight::Dense(tensor) => Ok(tensor_resident_bytes(tensor, tensor_storages)),
        NativeWeight::Quantized(matrix) => quantized_resident_bytes(matrix),
        NativeWeight::WeightNorm(weight) => {
            tensor_resident_bytes(&weight.magnitude, tensor_storages)
                .checked_add(tensor_resident_bytes(&weight.direction, tensor_storages))
                .ok_or(NativeOpsError::ResidentBytesOverflow)
        }
        NativeWeight::SpectralNorm(weight) => {
            let vectors = weight
                .left
                .as_ref()
                .map_or(0, Vec::capacity)
                .checked_add(weight.right.as_ref().map_or(0, Vec::capacity))
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            tensor_resident_bytes(&weight.original, tensor_storages)
                .checked_add(
                    u64::try_from(vectors).map_err(|_| NativeOpsError::ResidentBytesOverflow)?,
                )
                .ok_or(NativeOpsError::ResidentBytesOverflow)
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConvolutionAutopad {
    #[default]
    Disabled,
    CausalZero,
}

#[derive(Clone, Debug)]
pub struct NativeModule {
    layer_name: String,
    spec: NativeModuleSpec,
    manual_cast: bool,
    weight: Option<NativeWeight>,
    weight_norm_dimension: Option<Option<usize>>,
    spectral_norm_config: Option<NativeSpectralNormConfig>,
    bias: Option<Tensor>,
    generation: u64,
    prefetched: Option<PrefetchedParameters>,
    children: Vec<NativeModule>,
    registered_buffer: Option<Tensor>,
    normalization_state: Option<NativeNormalizationState>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NativeExecutionRequirements {
    supports: Vec<OperationSupport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeModelExecutionRole {
    Model,
    Clip,
    Vae,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeModelExecutionAdmission {
    role: NativeModelExecutionRole,
    execution_profile: String,
    resource_identity_sha256: String,
    target_device: DeviceId,
    dtype: DType,
    layout: Layout,
    stream: StreamId,
    transfer_bytes: u64,
    cast_bytes: u64,
    resident_bytes: u64,
    semantic_digest_sha256: String,
}

impl NativeModelExecutionAdmission {
    pub const fn role(&self) -> NativeModelExecutionRole {
        self.role
    }

    pub fn execution_profile(&self) -> &str {
        &self.execution_profile
    }

    pub fn resource_identity_sha256(&self) -> &str {
        &self.resource_identity_sha256
    }

    pub const fn target_device(&self) -> DeviceId {
        self.target_device
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn layout(&self) -> Layout {
        self.layout
    }

    pub const fn stream(&self) -> StreamId {
        self.stream
    }

    pub const fn transfer_bytes(&self) -> u64 {
        self.transfer_bytes
    }

    pub const fn cast_bytes(&self) -> u64 {
        self.cast_bytes
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }
}

fn parameter_materialization_requirements(dtype: DType) -> NativeExecutionRequirements {
    let mut requirements = NativeExecutionRequirements::new();
    requirements.append_tensor_io(dtype);
    requirements
}

impl NativeExecutionRequirements {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, support: OperationSupport) {
        if !self.supports.contains(&support) {
            self.supports.push(support);
        }
    }

    pub fn extend(&mut self, supports: impl IntoIterator<Item = OperationSupport>) {
        for support in supports {
            self.insert(support);
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = OperationSupport> + '_ {
        self.supports.iter().copied()
    }

    fn require_matrix_support(
        &self,
        capabilities: &BackendCapabilityMatrix,
    ) -> Result<(), NativeOpsError> {
        for support in self.iter() {
            capabilities.require("zed.sim_model.native_module.execute", support)?;
            if !capabilities.is_deterministic(support) {
                return Err(TensorError::UnsupportedCapability {
                    operation: "zed.sim_model.native_module.execute".to_owned(),
                    device: capabilities.device(),
                    reason: format!(
                        "primitive {:?}, role {:?}, dtype {:?}, layout {:?} is not certified deterministic",
                        support.primitive(),
                        support.role(),
                        support.dtype(),
                        support.layout()
                    ),
                }
                .into());
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn admit_backend_target(
        &self,
        backend: &dyn TensorBackend,
        device: DeviceId,
        dtype: DType,
        layout: Layout,
        stream: StreamId,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        context.cancellation.check()?;
        if context.stream != stream {
            return Err(NativeOpsError::ExecutionStreamMismatch {
                requested: stream,
                context: context.stream,
            });
        }
        if backend.device() != device {
            return Err(NativeOpsError::BackendTargetMismatch {
                requested: device,
                backend: backend.device(),
            });
        }
        let capabilities = backend.capabilities();
        if capabilities.device() != backend.device() {
            return Err(NativeOpsError::StaleBackendCapabilities {
                backend: backend.device(),
                capabilities: capabilities.device(),
            });
        }
        for support in self.iter() {
            if let Some(requirement) = support.dtype()
                && requirement != dtype
            {
                return Err(NativeOpsError::ExecutionDTypeMismatch {
                    requested: dtype,
                    requirement,
                });
            }
            if let Some(requirement) = support.layout()
                && requirement != layout
            {
                return Err(NativeOpsError::ExecutionLayoutMismatch {
                    requested: layout,
                    requirement,
                });
            }
        }
        self.require_matrix_support(capabilities)?;
        capabilities.require(
            "zed.sim_model.native_module.execute.record_event",
            OperationSupport::record_event(),
        )?;
        capabilities.require(
            "zed.sim_model.native_module.execute.wait_event",
            OperationSupport::wait_event(),
        )?;
        for support in [
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ] {
            if !capabilities.is_deterministic(support) {
                return Err(TensorError::UnsupportedCapability {
                    operation: "zed.sim_model.native_module.execute.event".to_owned(),
                    device: capabilities.device(),
                    reason: format!(
                        "event primitive {:?} is not certified deterministic",
                        support.primitive()
                    ),
                }
                .into());
            }
        }
        context.cancellation.check()?;
        Ok(())
    }

    pub(crate) fn append_tensor_io(&mut self, dtype: DType) {
        self.extend([
            OperationSupport::allocation(dtype, Layout::Contiguous),
            OperationSupport::copy_input(dtype, Layout::Contiguous),
            OperationSupport::copy_output(dtype, Layout::Contiguous),
        ]);
    }

    pub(crate) fn append_unary(&mut self, operation: UnaryOperation, dtype: DType) {
        self.extend([
            OperationSupport::unary_input(operation, dtype, Layout::Contiguous),
            OperationSupport::unary_output(operation, dtype, Layout::Contiguous),
        ]);
    }

    pub(crate) fn append_binary(&mut self, operation: BinaryOperation, dtype: DType) {
        self.extend([
            OperationSupport::binary_input(operation, dtype, Layout::Contiguous),
            OperationSupport::binary_output(operation, dtype, Layout::Contiguous),
        ]);
    }

    pub(crate) fn append_reduction(&mut self, operation: ReductionOperation, dtype: DType) {
        self.extend([
            OperationSupport::reduction_input(operation, dtype, Layout::Contiguous),
            OperationSupport::reduction_output(operation, dtype, Layout::Contiguous),
        ]);
    }

    pub(crate) fn append_linear_algebra(
        &mut self,
        operation: LinearAlgebraOperation,
        dtype: DType,
    ) {
        self.extend([
            OperationSupport::linear_algebra_input(operation, dtype, Layout::Contiguous),
            OperationSupport::linear_algebra_output(operation, dtype, Layout::Contiguous),
        ]);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn admit_native_model_resource_execution<F>(
    role: NativeModelExecutionRole,
    execution_profile: &str,
    resource_identity_sha256: &str,
    requirements: &NativeExecutionRequirements,
    resident_accounting: F,
    backend: &dyn TensorBackend,
    target_device: DeviceId,
    dtype: DType,
    layout: Layout,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<NativeModelExecutionAdmission, NativeOpsError>
where
    F: FnOnce() -> Result<(Vec<Tensor>, u64), NativeOpsError>,
{
    if execution_profile.is_empty() || execution_profile.len() > 1_024 {
        return Err(NativeOpsError::Invalid(
            "native model execution profile must be non-empty and bounded",
        ));
    }
    if resource_identity_sha256.len() != 64
        || !resource_identity_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(NativeOpsError::Invalid(
            "native model resource identity must be a lowercase SHA-256 digest",
        ));
    }
    requirements.admit_backend_target(backend, target_device, dtype, layout, stream, context)?;

    let (resident_tensors, structural_resident_bytes) = resident_accounting()?;
    let mut resident_storages = HashSet::new();
    let mut resident_bytes = structural_resident_bytes;
    let mut transfer_bytes = 0_u64;
    let mut cast_bytes = 0_u64;
    for tensor in &resident_tensors {
        context.cancellation.check()?;
        if resident_storages.insert(tensor.storage_id()) {
            resident_bytes = resident_bytes
                .checked_add(tensor.storage_byte_len())
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            if tensor.descriptor().device() != target_device {
                transfer_bytes = transfer_bytes
                    .checked_add(tensor.descriptor().byte_len()?)
                    .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            }
            if tensor.descriptor().dtype() != dtype {
                cast_bytes = cast_bytes
                    .checked_add(
                        tensor
                            .descriptor()
                            .element_count()?
                            .checked_mul(dtype.byte_width())
                            .ok_or(NativeOpsError::ResidentBytesOverflow)?,
                    )
                    .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            }
        }
    }
    context.cancellation.check()?;

    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.native-model-execution-admission.v1");
    hasher.update([match role {
        NativeModelExecutionRole::Model => 0,
        NativeModelExecutionRole::Clip => 1,
        NativeModelExecutionRole::Vae => 2,
    }]);
    hasher.update((execution_profile.len() as u64).to_be_bytes());
    hasher.update(execution_profile.as_bytes());
    hasher.update(resource_identity_sha256.as_bytes());
    hasher.update([device_kind_tag(target_device.kind())]);
    hasher.update(target_device.ordinal().to_be_bytes());
    hasher.update(dtype.catalog_name().as_bytes());
    hasher.update(format!("{layout:?}").as_bytes());
    hasher.update(stream.get().to_be_bytes());
    hasher.update(transfer_bytes.to_be_bytes());
    hasher.update(cast_bytes.to_be_bytes());
    hasher.update(resident_bytes.to_be_bytes());
    let encoded_requirements = serde_json::to_vec(&requirements.supports)
        .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
    hasher.update((encoded_requirements.len() as u64).to_be_bytes());
    hasher.update(&encoded_requirements);
    let semantic_digest_sha256 = format!("{:x}", hasher.finalize());

    Ok(NativeModelExecutionAdmission {
        role,
        execution_profile: execution_profile.to_owned(),
        resource_identity_sha256: resource_identity_sha256.to_owned(),
        target_device,
        dtype,
        layout,
        stream,
        transfer_bytes,
        cast_bytes,
        resident_bytes,
        semantic_digest_sha256,
    })
}

const fn device_kind_tag(kind: DeviceKind) -> u8 {
    match kind {
        DeviceKind::Cpu => 0,
        DeviceKind::Cuda => 1,
        DeviceKind::Rocm => 2,
        DeviceKind::Metal => 3,
        DeviceKind::DirectMl => 4,
        DeviceKind::Xpu => 5,
        DeviceKind::Npu => 6,
        DeviceKind::Mlu => 7,
        DeviceKind::CoreX => 8,
    }
}

pub struct RngAwareModuleForward {
    pub output: Tensor,
    pub transaction: RngTransaction,
}

impl NativeModule {
    pub fn keyed_state(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<BTreeMap<String, Tensor>, NativeOpsError> {
        cancellation.check()?;
        let mut state = BTreeMap::new();
        self.append_keyed_state("", &mut state, cancellation)?;
        Ok(state)
    }

    fn append_keyed_state(
        &self,
        parent: &str,
        state: &mut BTreeMap<String, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeOpsError> {
        cancellation.check()?;
        let path = match (parent.is_empty(), self.layer_name.is_empty()) {
            (true, _) => self.layer_name.clone(),
            (_, true) => parent.to_owned(),
            (false, false) if self.layer_name.starts_with(parent) => self.layer_name.clone(),
            (false, false) => format!("{parent}.{}", self.layer_name),
        };
        let insert = |state: &mut BTreeMap<String, Tensor>, name: String, tensor: &Tensor| {
            if state.insert(name.clone(), tensor.clone()).is_some() {
                Err(NativeOpsError::InvalidOwned(format!(
                    "native module state repeats {name}"
                )))
            } else {
                Ok(())
            }
        };
        if let Some(weight) = &self.weight {
            match weight {
                NativeWeight::Dense(tensor) => {
                    insert(state, format!("{path}.weight"), tensor)?;
                }
                NativeWeight::WeightNorm(weight) => {
                    insert(state, format!("{path}.weight_g"), &weight.magnitude)?;
                    insert(state, format!("{path}.weight_v"), &weight.direction)?;
                }
                NativeWeight::SpectralNorm(weight) => {
                    insert(state, format!("{path}.weight_orig"), &weight.original)?;
                }
                NativeWeight::Quantized(_) => {
                    return Err(NativeOpsError::InvalidOwned(format!(
                        "quantized module state {path} cannot be exported as an exact tensor"
                    )));
                }
            }
        }
        if let Some(bias) = &self.bias {
            insert(state, format!("{path}.bias"), bias)?;
        }
        if let Some(buffer) = &self.registered_buffer {
            insert(state, path.clone(), buffer)?;
        }
        if self.normalization_state.is_some() {
            return Err(NativeOpsError::InvalidOwned(format!(
                "normalization state {path} cannot be exported without its source tensors"
            )));
        }
        for child in &self.children {
            child.append_keyed_state(&path, state, cancellation)?;
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn container(layer_name: impl Into<String>) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Container, false)
    }

    pub fn linear(
        layer_name: impl Into<String>,
        input_features: usize,
        output_features: usize,
        bias: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if input_features == 0 || output_features == 0 {
            return Err(NativeOpsError::Invalid(
                "linear feature dimensions must be nonzero",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::Linear {
                input_features,
                output_features,
                bias,
            },
            manual_cast,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn convolution(
        layer_name: impl Into<String>,
        input_channels: usize,
        output_channels: usize,
        kernel_shape: Vec<usize>,
        bias: bool,
        geometry: ConvolutionGeometry,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if input_channels == 0
            || output_channels == 0
            || kernel_shape.len() != geometry.spatial_dimensions()
            || kernel_shape.contains(&0)
            || !input_channels.is_multiple_of(geometry.groups())
            || !output_channels.is_multiple_of(geometry.groups())
        {
            return Err(NativeOpsError::Invalid(
                "convolution channel, kernel, or group configuration is invalid",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::Convolution {
                input_channels,
                output_channels,
                kernel_shape,
                bias,
                geometry,
            },
            manual_cast,
        )
    }

    pub fn layer_norm(
        layer_name: impl Into<String>,
        normalized_shape: Vec<usize>,
        epsilon: f32,
        elementwise_affine: bool,
        bias: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if normalized_shape.is_empty()
            || normalized_shape.contains(&0)
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || (bias && !elementwise_affine)
        {
            return Err(NativeOpsError::Invalid(
                "layer-normalization shape, epsilon, or affine configuration is invalid",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::LayerNorm {
                normalized_shape,
                epsilon,
                elementwise_affine,
                bias,
            },
            manual_cast,
        )
    }

    pub fn group_norm(
        layer_name: impl Into<String>,
        groups: usize,
        channels: usize,
        epsilon: f32,
        affine: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if groups == 0
            || channels == 0
            || !channels.is_multiple_of(groups)
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(NativeOpsError::Invalid(
                "group-normalization channels, groups, or epsilon are invalid",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::GroupNorm {
                groups,
                channels,
                epsilon,
                affine,
            },
            manual_cast,
        )
    }

    fn new(
        layer_name: impl Into<String>,
        spec: NativeModuleSpec,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        Ok(Self {
            layer_name: checked_layer_name(layer_name.into())?,
            spec,
            manual_cast,
            weight: None,
            weight_norm_dimension: None,
            spectral_norm_config: None,
            bias: None,
            generation: 0,
            prefetched: None,
            children: Vec::new(),
            registered_buffer: None,
            normalization_state: None,
        })
    }

    pub fn average_pool_2d(
        layer_name: impl Into<String>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
    ) -> Result<Self, NativeOpsError> {
        if kernel_size.contains(&0) || stride.contains(&0) {
            return Err(NativeOpsError::Invalid(
                "average-pool kernel and stride dimensions must be nonzero",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::AveragePool2d {
                kernel_size,
                stride,
            },
            false,
        )
    }

    pub fn average_pool_1d(
        layer_name: impl Into<String>,
        kernel_size: usize,
        stride: usize,
    ) -> Result<Self, NativeOpsError> {
        if kernel_size == 0 || stride == 0 {
            return Err(NativeOpsError::Invalid(
                "average-pool-1d kernel and stride must be nonzero",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::AveragePool1d {
                kernel_size,
                stride,
            },
            false,
        )
    }

    pub fn adaptive_average_pool_2d(
        layer_name: impl Into<String>,
        output_size: [usize; 2],
    ) -> Result<Self, NativeOpsError> {
        if output_size.contains(&0) {
            return Err(NativeOpsError::Invalid(
                "adaptive-average-pool output dimensions must be nonzero",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::AdaptiveAveragePool2d { output_size },
            false,
        )
    }

    pub fn average_pool_3d(
        layer_name: impl Into<String>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
    ) -> Result<Self, NativeOpsError> {
        if kernel_size.contains(&0) || stride.contains(&0) {
            return Err(NativeOpsError::Invalid(
                "average-pool-3d kernel and stride dimensions must be nonzero",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::AveragePool3d {
                kernel_size,
                stride,
            },
            false,
        )
    }

    pub fn batch_norm_1d(
        layer_name: impl Into<String>,
        features: usize,
        epsilon: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        Self::batch_norm(
            layer_name,
            1,
            features,
            epsilon,
            momentum,
            affine,
            track_running_stats,
            manual_cast,
        )
    }

    pub fn batch_norm_2d(
        layer_name: impl Into<String>,
        features: usize,
        epsilon: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        Self::batch_norm(
            layer_name,
            2,
            features,
            epsilon,
            momentum,
            affine,
            track_running_stats,
            manual_cast,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_norm(
        layer_name: impl Into<String>,
        dimensions: usize,
        features: usize,
        epsilon: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if !matches!(dimensions, 1 | 2)
            || features == 0
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || !momentum.is_finite()
            || !(0.0..=1.0).contains(&momentum)
        {
            return Err(NativeOpsError::Invalid(
                "batch-normalization dimensions, features, epsilon, or momentum are invalid",
            ));
        }
        let mut module = Self::new(
            layer_name,
            NativeModuleSpec::BatchNorm {
                dimensions,
                features,
                epsilon,
                momentum,
                affine,
                track_running_stats,
                training: true,
            },
            manual_cast,
        )?;
        if track_running_stats {
            module.normalization_state = Some(NativeNormalizationState {
                running_mean: vec![0.0; features],
                running_variance: vec![1.0; features],
            });
        }
        Ok(module)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_2d(
        layer_name: impl Into<String>,
        input_channels: usize,
        output_channels: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
        bias: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        let geometry = ConvolutionGeometry::new(
            2,
            stride.to_vec(),
            padding.to_vec(),
            dilation.to_vec(),
            groups,
            false,
            vec![0, 0],
        )?;
        Self::convolution(
            layer_name,
            input_channels,
            output_channels,
            kernel_size.to_vec(),
            bias,
            geometry,
            manual_cast,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_3d(
        layer_name: impl Into<String>,
        input_channels: usize,
        output_channels: usize,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
        bias: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        let geometry = ConvolutionGeometry::new(
            3,
            stride.to_vec(),
            padding.to_vec(),
            dilation.to_vec(),
            groups,
            false,
            vec![0, 0, 0],
        )?;
        Self::convolution(
            layer_name,
            input_channels,
            output_channels,
            kernel_size.to_vec(),
            bias,
            geometry,
            manual_cast,
        )
    }

    pub fn buffer(layer_name: impl Into<String>, tensor: Tensor) -> Result<Self, NativeOpsError> {
        let mut module = Self::new(layer_name, NativeModuleSpec::Buffer, false)?;
        module.registered_buffer = Some(tensor);
        Ok(module)
    }

    pub fn embedding(
        layer_name: impl Into<String>,
        embeddings: usize,
        dimensions: usize,
        options: EmbeddingOptions,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if embeddings == 0
            || dimensions == 0
            || options.sparse
            || !options.norm_type.is_finite()
            || options.norm_type < 1.0
            || options
                .max_norm
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(NativeOpsError::Invalid(
                "embedding dimensions or options are invalid for the native dense owner",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::Embedding {
                embeddings,
                dimensions,
                options,
            },
            manual_cast,
        )
    }

    pub fn dropout(
        layer_name: impl Into<String>,
        probability: f32,
    ) -> Result<Self, NativeOpsError> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(NativeOpsError::Invalid(
                "dropout probability must be finite and in the inclusive range zero to one",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::Dropout {
                probability,
                training: true,
            },
            false,
        )
    }

    pub fn elu(layer_name: impl Into<String>, alpha: f32) -> Result<Self, NativeOpsError> {
        if !alpha.is_finite() {
            return Err(NativeOpsError::Invalid("ELU alpha must be finite"));
        }
        Self::new(layer_name, NativeModuleSpec::Elu { alpha }, false)
    }

    pub fn gelu(
        layer_name: impl Into<String>,
        approximation: GeluApproximation,
    ) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Gelu { approximation }, false)
    }

    pub fn huber_loss(
        layer_name: impl Into<String>,
        delta: f32,
        reduction: LossReduction,
    ) -> Result<Self, NativeOpsError> {
        if !delta.is_finite() || delta <= 0.0 {
            return Err(NativeOpsError::Invalid(
                "Huber loss delta must be finite and positive",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::HuberLoss { delta, reduction },
            false,
        )
    }

    pub fn l1_loss(
        layer_name: impl Into<String>,
        reduction: LossReduction,
    ) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::L1Loss { reduction }, false)
    }

    pub fn identity(layer_name: impl Into<String>) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Identity, false)
    }

    pub fn mse_loss(
        layer_name: impl Into<String>,
        reduction: LossReduction,
    ) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::MseLoss { reduction }, false)
    }

    pub fn instance_norm_2d(
        layer_name: impl Into<String>,
        features: usize,
        epsilon: f32,
        affine: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if features == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(NativeOpsError::Invalid(
                "instance-normalization features or epsilon are invalid",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::InstanceNorm2d {
                features,
                epsilon,
                affine,
            },
            manual_cast,
        )
    }

    pub fn leaky_relu(
        layer_name: impl Into<String>,
        negative_slope: f32,
    ) -> Result<Self, NativeOpsError> {
        if !negative_slope.is_finite() {
            return Err(NativeOpsError::Invalid(
                "leaky-ReLU negative slope must be finite",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::LeakyRelu { negative_slope },
            false,
        )
    }

    pub fn multihead_attention(
        layer_name: impl Into<String>,
        embed_dimension: usize,
        heads: usize,
        bias: bool,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if heads == 0 || embed_dimension == 0 || !embed_dimension.is_multiple_of(heads) {
            return Err(NativeOpsError::Invalid(
                "multihead-attention embedding dimension must be nonzero and divisible by heads",
            ));
        }
        let mut module = Self::new(
            layer_name,
            NativeModuleSpec::MultiheadAttention {
                embed_dimension,
                heads,
            },
            false,
        )?;
        module.children = ["q_proj", "k_proj", "v_proj", "out_proj"]
            .into_iter()
            .map(|name| Self::linear(name, embed_dimension, embed_dimension, bias, manual_cast))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(module)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool_2d(
        layer_name: impl Into<String>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> Result<Self, NativeOpsError> {
        if kernel_size.contains(&0) || stride.contains(&0) || dilation.contains(&0) {
            return Err(NativeOpsError::Invalid(
                "max-pool kernel, stride, and dilation dimensions must be nonzero",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::MaxPool2d {
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            },
            false,
        )
    }

    pub fn module_dict(
        layer_name: impl Into<String>,
        children: Vec<NativeModule>,
    ) -> Result<Self, NativeOpsError> {
        let mut names = BTreeSet::new();
        if children
            .iter()
            .any(|child| !names.insert(child.layer_name.clone()))
        {
            return Err(NativeOpsError::Invalid(
                "module-dict child layer names must be unique",
            ));
        }
        let mut module = Self::new(layer_name, NativeModuleSpec::ModuleDict, false)?;
        module.children = children;
        Ok(module)
    }

    pub fn module_list(
        layer_name: impl Into<String>,
        children: Vec<NativeModule>,
    ) -> Result<Self, NativeOpsError> {
        let mut module = Self::new(layer_name, NativeModuleSpec::ModuleList, false)?;
        module.children = children;
        Ok(module)
    }

    pub fn replication_pad_2d(
        layer_name: impl Into<String>,
        padding: [usize; 4],
    ) -> Result<Self, NativeOpsError> {
        Self::new(
            layer_name,
            NativeModuleSpec::ReplicationPad2d { padding },
            false,
        )
    }

    pub fn pixel_shuffle(
        layer_name: impl Into<String>,
        factor: u64,
    ) -> Result<Self, NativeOpsError> {
        if factor == 0 {
            return Err(NativeOpsError::Invalid(
                "pixel-shuffle factor must be nonzero",
            ));
        }
        Self::new(layer_name, NativeModuleSpec::PixelShuffle { factor }, false)
    }

    pub fn pixel_unshuffle(
        layer_name: impl Into<String>,
        factor: u64,
    ) -> Result<Self, NativeOpsError> {
        if factor == 0 {
            return Err(NativeOpsError::Invalid(
                "pixel-unshuffle factor must be nonzero",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::PixelUnshuffle { factor },
            false,
        )
    }

    pub fn prelu(
        layer_name: impl Into<String>,
        num_parameters: usize,
        manual_cast: bool,
    ) -> Result<Self, NativeOpsError> {
        if num_parameters == 0 {
            return Err(NativeOpsError::Invalid(
                "PReLU parameter count must be nonzero",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::PRelu { num_parameters },
            manual_cast,
        )
    }

    pub fn sequential(
        layer_name: impl Into<String>,
        children: Vec<NativeModule>,
    ) -> Result<Self, NativeOpsError> {
        let mut names = BTreeSet::new();
        if children
            .iter()
            .any(|child| !names.insert(child.layer_name.clone()))
        {
            return Err(NativeOpsError::Invalid(
                "sequential child layer names must be unique",
            ));
        }
        let mut module = Self::new(layer_name, NativeModuleSpec::Sequential, false)?;
        module.children = children;
        Ok(module)
    }

    pub fn relu(layer_name: impl Into<String>) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Relu, false)
    }

    pub fn relu_6(layer_name: impl Into<String>) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Relu6, false)
    }

    pub fn silu(layer_name: impl Into<String>) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Silu, false)
    }

    pub fn sigmoid(layer_name: impl Into<String>) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Sigmoid, false)
    }

    pub fn smooth_l1_loss(
        layer_name: impl Into<String>,
        beta: f32,
        reduction: LossReduction,
    ) -> Result<Self, NativeOpsError> {
        if !beta.is_finite() || beta < 0.0 {
            return Err(NativeOpsError::Invalid(
                "smooth-L1 beta must be finite and nonnegative",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::SmoothL1Loss { beta, reduction },
            false,
        )
    }

    pub fn softmax(
        layer_name: impl Into<String>,
        dimension: isize,
    ) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Softmax { dimension }, false)
    }

    pub fn tanh(layer_name: impl Into<String>) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::Tanh, false)
    }

    pub fn upsample(
        layer_name: impl Into<String>,
        scale_factor: [f64; 2],
        mode: UpsampleMode,
        align_corners: Option<bool>,
    ) -> Result<Self, NativeOpsError> {
        if scale_factor
            .iter()
            .any(|factor| !factor.is_finite() || *factor <= 0.0)
        {
            return Err(NativeOpsError::Invalid(
                "upsample scale factors must be finite and positive",
            ));
        }
        if mode == UpsampleMode::Nearest && align_corners == Some(true) {
            return Err(NativeOpsError::Invalid(
                "align_corners is invalid for nearest-neighbor upsampling",
            ));
        }
        Self::new(
            layer_name,
            NativeModuleSpec::Upsample {
                scale_factor,
                mode,
                align_corners,
            },
            false,
        )
    }

    pub fn zero_pad_2d(
        layer_name: impl Into<String>,
        padding: [usize; 4],
    ) -> Result<Self, NativeOpsError> {
        Self::new(layer_name, NativeModuleSpec::ZeroPad2d { padding }, false)
    }

    pub fn registered_buffer(&self) -> Option<&Tensor> {
        self.registered_buffer.as_ref()
    }

    pub(crate) fn dense_parameters(&self) -> Result<(&Tensor, Option<&Tensor>), NativeOpsError> {
        let weight = match self
            .weight
            .as_ref()
            .ok_or(NativeOpsError::ParametersNotLoaded)?
        {
            NativeWeight::Dense(weight) => weight,
            _ => {
                return Err(NativeOpsError::Invalid(
                    "the focused architecture adapter requires dense module parameters",
                ));
            }
        };
        Ok((weight, self.bias.as_ref()))
    }

    pub(crate) fn materialize_execution_state_with_context(
        &mut self,
        backend: &CpuBackend,
        dtype: DType,
        device: DeviceId,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        parameter_materialization_requirements(dtype).admit_backend_target(
            backend,
            device,
            dtype,
            Layout::Contiguous,
            context.stream,
            context,
        )?;
        if self.weight.is_some() {
            let prepared = self.prepare_parameters(
                backend,
                None,
                Some(dtype),
                Some(device),
                Some(dtype),
                Some(dtype),
                false,
                context,
            )?;
            self.weight = Some(NativeWeight::Dense(prepared.weight));
            self.bias = prepared.bias;
            self.prefetched = None;
            self.generation = self.next_generation()?;
        }
        if let Some(buffer) = &self.registered_buffer
            && matches!(
                buffer.descriptor().dtype(),
                DType::F16 | DType::Bf16 | DType::F32
            )
        {
            self.registered_buffer = Some(cast_to(
                backend, buffer, dtype, device, false, false, context,
            )?);
            self.generation = self.next_generation()?;
        }
        for child in &mut self.children {
            child.materialize_execution_state_with_context(backend, dtype, device, context)?;
        }
        context.cancellation.check()?;
        Ok(())
    }

    pub fn children(&self) -> &[NativeModule] {
        &self.children
    }

    pub fn child_mut(&mut self, layer_name: &str) -> Option<&mut NativeModule> {
        self.children
            .iter_mut()
            .find(|child| child.layer_name == layer_name)
    }

    pub fn child_at(&self, index: usize) -> Option<&NativeModule> {
        self.children.get(index)
    }

    pub fn child_at_mut(&mut self, index: usize) -> Option<&mut NativeModule> {
        self.children.get_mut(index)
    }

    pub fn set_training(&mut self, training: bool) {
        match &mut self.spec {
            NativeModuleSpec::BatchNorm {
                training: current, ..
            }
            | NativeModuleSpec::Dropout {
                training: current, ..
            } => *current = training,
            _ => {}
        }
        for child in &mut self.children {
            child.set_training(training);
        }
    }

    pub fn running_statistics(&self) -> Option<(&[f32], &[f32])> {
        self.normalization_state.as_ref().map(|state| {
            (
                state.running_mean.as_slice(),
                state.running_variance.as_slice(),
            )
        })
    }

    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }

    pub fn spec(&self) -> &NativeModuleSpec {
        &self.spec
    }

    pub const fn manual_cast(&self) -> bool {
        self.manual_cast
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn has_execution_state(&self) -> bool {
        self.weight.is_some()
            || self.bias.is_some()
            || self.registered_buffer.is_some()
            || self.normalization_state.is_some()
            || self.children.iter().any(Self::has_execution_state)
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, NativeOpsError> {
        let mut tensor_storages = HashSet::new();
        self.collect_resident_storage_bytes(&mut tensor_storages)
    }

    pub fn resident_tensor_allocations(&self) -> Vec<(StorageId, u64)> {
        let mut allocations = Vec::new();
        self.collect_resident_tensor_allocations(&mut allocations);
        allocations
    }

    fn collect_resident_tensor_allocations(&self, allocations: &mut Vec<(StorageId, u64)>) {
        fn insert_tensor(allocations: &mut Vec<(StorageId, u64)>, tensor: &Tensor) {
            let storage_id = tensor.storage_id();
            if !allocations
                .iter()
                .any(|(existing, _)| *existing == storage_id)
            {
                allocations.push((storage_id, tensor.storage_byte_len()));
            }
        }

        fn insert_weight(allocations: &mut Vec<(StorageId, u64)>, weight: &NativeWeight) {
            match weight {
                NativeWeight::Dense(tensor) => insert_tensor(allocations, tensor),
                NativeWeight::WeightNorm(weight) => {
                    insert_tensor(allocations, &weight.magnitude);
                    insert_tensor(allocations, &weight.direction);
                }
                NativeWeight::SpectralNorm(weight) => {
                    insert_tensor(allocations, &weight.original);
                }
                NativeWeight::Quantized(_) => {}
            }
        }

        if let Some(weight) = &self.weight {
            insert_weight(allocations, weight);
        }
        if let Some(bias) = &self.bias {
            insert_tensor(allocations, bias);
        }
        if let Some(buffer) = &self.registered_buffer {
            insert_tensor(allocations, buffer);
        }
        if let Some(prefetched) = &self.prefetched {
            insert_tensor(allocations, &prefetched.weight);
            if let Some(bias) = &prefetched.bias {
                insert_tensor(allocations, bias);
            }
            if let Some(weight) = &prefetched.next_weight {
                insert_weight(allocations, weight);
            }
        }
        for child in &self.children {
            child.collect_resident_tensor_allocations(allocations);
        }
    }

    fn collect_resident_storage_bytes(
        &self,
        tensor_storages: &mut HashSet<StorageId>,
    ) -> Result<u64, NativeOpsError> {
        let mut total = 0_u64;
        if let Some(weight) = &self.weight {
            total = total
                .checked_add(native_weight_resident_bytes(weight, tensor_storages)?)
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
        }
        if let Some(bias) = &self.bias {
            total = total
                .checked_add(tensor_resident_bytes(bias, tensor_storages))
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
        }
        if let Some(buffer) = &self.registered_buffer {
            total = total
                .checked_add(tensor_resident_bytes(buffer, tensor_storages))
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
        }
        if let Some(prefetched) = &self.prefetched {
            total = total
                .checked_add(tensor_resident_bytes(&prefetched.weight, tensor_storages))
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            if let Some(bias) = &prefetched.bias {
                total = total
                    .checked_add(tensor_resident_bytes(bias, tensor_storages))
                    .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            }
            if let Some(weight) = &prefetched.requantized_weight {
                total = total
                    .checked_add(quantized_resident_bytes(weight)?)
                    .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            }
            if let Some(weight) = &prefetched.next_weight {
                total = total
                    .checked_add(native_weight_resident_bytes(weight, tensor_storages)?)
                    .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            }
        }
        if let Some(normalization) = &self.normalization_state {
            let values = normalization
                .running_mean
                .capacity()
                .checked_add(normalization.running_variance.capacity())
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
            total = total
                .checked_add(
                    u64::try_from(values).map_err(|_| NativeOpsError::ResidentBytesOverflow)?,
                )
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
        }
        for child in &self.children {
            total = total
                .checked_add(child.collect_resident_storage_bytes(tensor_storages)?)
                .ok_or(NativeOpsError::ResidentBytesOverflow)?;
        }
        Ok(total)
    }

    pub fn execution_requirements(&self, dtype: DType) -> NativeExecutionRequirements {
        let mut requirements = NativeExecutionRequirements::new();
        self.append_execution_requirements(dtype, &mut requirements);
        requirements
    }

    fn append_execution_requirements(
        &self,
        dtype: DType,
        requirements: &mut NativeExecutionRequirements,
    ) {
        let tensor_io = !matches!(
            self.spec,
            NativeModuleSpec::Container
                | NativeModuleSpec::ModuleDict
                | NativeModuleSpec::ModuleList
                | NativeModuleSpec::Sequential
        );
        if tensor_io {
            requirements.append_tensor_io(dtype);
        }
        match self.spec {
            NativeModuleSpec::Linear { .. } => requirements
                .append_linear_algebra(LinearAlgebraOperation::BatchMatrixMultiply, dtype),
            NativeModuleSpec::Convolution { .. } => requirements.extend([
                OperationSupport::convolution_input(dtype, Layout::Contiguous),
                OperationSupport::convolution_output(dtype, Layout::Contiguous),
            ]),
            NativeModuleSpec::LayerNorm { .. }
            | NativeModuleSpec::GroupNorm { .. }
            | NativeModuleSpec::BatchNorm { .. }
            | NativeModuleSpec::InstanceNorm2d { .. } => {
                requirements.append_reduction(ReductionOperation::Mean, dtype);
                requirements.append_reduction(ReductionOperation::Variance, dtype);
                for operation in [
                    BinaryOperation::Add,
                    BinaryOperation::Subtract,
                    BinaryOperation::Multiply,
                    BinaryOperation::Divide,
                ] {
                    requirements.append_binary(operation, dtype);
                }
                requirements.append_unary(UnaryOperation::SquareRoot, dtype);
            }
            NativeModuleSpec::Gelu { approximation } => {
                requirements.append_binary(BinaryOperation::Multiply, dtype);
                requirements.append_binary(BinaryOperation::Add, dtype);
                if approximation == GeluApproximation::Tanh {
                    requirements.append_unary(UnaryOperation::HyperbolicTangent, dtype);
                }
            }
            NativeModuleSpec::Elu { .. } => {
                requirements.append_unary(UnaryOperation::Exponential, dtype);
            }
            NativeModuleSpec::LeakyRelu { .. }
            | NativeModuleSpec::PRelu { .. }
            | NativeModuleSpec::Relu
            | NativeModuleSpec::Relu6 => {
                requirements.append_unary(UnaryOperation::Relu, dtype);
            }
            NativeModuleSpec::Sigmoid => {
                requirements.append_unary(UnaryOperation::Sigmoid, dtype);
            }
            // The canonical SiLU owner decodes the admitted tensor into its bounded f32
            // workspace, evaluates the exact scalar map, and re-encodes the requested dtype.
            // It does not dispatch the backend unary or binary primitives, so advertising
            // those as target requirements would reject otherwise executable low-precision
            // CPU modules.
            NativeModuleSpec::Silu => {}
            NativeModuleSpec::Tanh => {
                requirements.append_unary(UnaryOperation::HyperbolicTangent, dtype);
            }
            NativeModuleSpec::Softmax { .. } => {
                requirements.append_unary(UnaryOperation::Exponential, dtype);
                requirements.append_reduction(ReductionOperation::Sum, dtype);
                requirements.append_binary(BinaryOperation::Divide, dtype);
            }
            NativeModuleSpec::AdaptiveAveragePool2d { .. }
            | NativeModuleSpec::AveragePool1d { .. }
            | NativeModuleSpec::AveragePool2d { .. }
            | NativeModuleSpec::AveragePool3d { .. } => {
                requirements.append_reduction(ReductionOperation::Mean, dtype);
            }
            NativeModuleSpec::MaxPool2d { .. } => {
                requirements.append_reduction(ReductionOperation::Maximum, dtype);
            }
            NativeModuleSpec::Upsample { mode, .. } => {
                let mode = match mode {
                    UpsampleMode::Nearest => ResizeMode::NearestExact,
                    UpsampleMode::Bilinear => ResizeMode::Bilinear,
                };
                requirements.extend([
                    OperationSupport::resize_input(mode, dtype, Layout::Contiguous),
                    OperationSupport::resize_output(mode, dtype, Layout::Contiguous),
                ]);
            }
            NativeModuleSpec::HuberLoss { reduction, .. }
            | NativeModuleSpec::SmoothL1Loss { reduction, .. }
            | NativeModuleSpec::L1Loss { reduction }
            | NativeModuleSpec::MseLoss { reduction } => {
                requirements.append_binary(BinaryOperation::Subtract, dtype);
                requirements.append_unary(UnaryOperation::Absolute, dtype);
                if reduction != LossReduction::None {
                    requirements.append_reduction(
                        if reduction == LossReduction::Mean {
                            ReductionOperation::Mean
                        } else {
                            ReductionOperation::Sum
                        },
                        dtype,
                    );
                }
            }
            NativeModuleSpec::MultiheadAttention { .. } => {
                requirements
                    .append_linear_algebra(LinearAlgebraOperation::BatchMatrixMultiply, dtype);
                requirements.append_unary(UnaryOperation::Exponential, dtype);
                requirements.append_reduction(ReductionOperation::Sum, dtype);
                requirements.append_binary(BinaryOperation::Divide, dtype);
            }
            NativeModuleSpec::Dropout { .. } => {
                requirements.append_binary(BinaryOperation::Multiply, dtype);
            }
            NativeModuleSpec::Container
            | NativeModuleSpec::Buffer
            | NativeModuleSpec::Embedding { .. }
            | NativeModuleSpec::Identity
            | NativeModuleSpec::ModuleDict
            | NativeModuleSpec::ModuleList
            | NativeModuleSpec::PixelShuffle { .. }
            | NativeModuleSpec::PixelUnshuffle { .. }
            | NativeModuleSpec::ReplicationPad2d { .. }
            | NativeModuleSpec::Sequential
            | NativeModuleSpec::ZeroPad2d { .. } => {}
        }
        for child in &self.children {
            child.append_execution_requirements(dtype, requirements);
        }
    }

    pub fn semantic_state_digest(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeOpsError> {
        cancellation.check()?;
        let mut digest = NativeModuleStateDigest::new(cancellation);
        digest.module(self)?;
        cancellation.check()?;
        Ok(format!("{:x}", digest.finish()))
    }

    pub fn prefetched_dtype_device(&self) -> Option<(DType, DeviceId)> {
        self.prefetched
            .as_ref()
            .map(|parameters| (parameters.dtype, parameters.device))
    }

    pub fn load_dense_parameters(
        &mut self,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<(), NativeOpsError> {
        if self.weight_norm_dimension.is_some() {
            return Err(NativeOpsError::Invalid(
                "a weight-normalized module must load magnitude and direction parameters",
            ));
        }
        self.validate_parameter_shapes(weight.descriptor().shape(), bias.as_ref())?;
        if bias
            .as_ref()
            .is_some_and(|bias| bias.descriptor().dtype() != weight.descriptor().dtype())
        {
            return Err(NativeOpsError::Invalid(
                "dense weight and bias dtypes must match at load time",
            ));
        }
        let next_generation = self.next_generation()?;
        self.weight = Some(match self.spectral_norm_config.clone() {
            Some(config) => NativeWeight::SpectralNorm(NativeSpectralNorm {
                original: weight,
                config,
                left: None,
                right: None,
            }),
            None => NativeWeight::Dense(weight),
        });
        self.bias = bias;
        self.prefetched = None;
        self.generation = next_generation;
        Ok(())
    }

    pub fn load_quantized_linear_parameters(
        &mut self,
        weight: QuantizedMatrix,
        bias: Option<Tensor>,
    ) -> Result<(), NativeOpsError> {
        if self.weight_norm_dimension.is_some() || self.spectral_norm_config.is_some() {
            return Err(NativeOpsError::Invalid(
                "weight or spectral normalization is incompatible with quantized parameter storage",
            ));
        }
        let NativeModuleSpec::Linear {
            input_features,
            output_features,
            bias: expects_bias,
        } = self.spec
        else {
            return Err(NativeOpsError::Invalid(
                "quantized storage is currently defined only for linear modules",
            ));
        };
        if weight.rows() != output_features || weight.columns() != input_features {
            return Err(NativeOpsError::Invalid(
                "quantized linear weight shape does not match the module",
            ));
        }
        validate_bias_shape(expects_bias, output_features, bias.as_ref())?;
        let next_generation = self.next_generation()?;
        self.weight = Some(NativeWeight::Quantized(weight));
        self.bias = bias;
        self.prefetched = None;
        self.generation = next_generation;
        Ok(())
    }

    pub fn load_weight_norm_parameters_with_context_exact_native(
        &mut self,
        backend: &CpuBackend,
        magnitude: Tensor,
        direction: Tensor,
        bias: Option<Tensor>,
        dimension: Option<usize>,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        self.load_weight_norm_parameters_impl(
            backend, magnitude, direction, bias, dimension, context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_weight_norm_parameters_impl(
        &mut self,
        backend: &CpuBackend,
        magnitude: Tensor,
        direction: Tensor,
        bias: Option<Tensor>,
        dimension: Option<usize>,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        if self.spectral_norm_config.is_some() {
            return Err(NativeOpsError::Invalid(
                "weight normalization cannot replace spectral normalization",
            ));
        }
        if self
            .weight_norm_dimension
            .is_some_and(|registered| registered != dimension)
        {
            return Err(NativeOpsError::Invalid(
                "loaded weight-normalization dimension does not match the registered parametrization",
            ));
        }
        self.validate_parameter_shapes(direction.descriptor().shape(), bias.as_ref())?;
        if magnitude.descriptor().dtype() != direction.descriptor().dtype()
            || magnitude.descriptor().device() != direction.descriptor().device()
            || bias.as_ref().is_some_and(|bias| {
                bias.descriptor().dtype() != direction.descriptor().dtype()
                    || bias.descriptor().device() != direction.descriptor().device()
            })
        {
            return Err(NativeOpsError::Invalid(
                "weight-normalization parameters must share dtype and device",
            ));
        }
        validate_weight_norm_magnitude_shape(
            magnitude.descriptor().shape(),
            direction.descriptor().shape(),
            dimension,
        )?;
        let parametrization = NativeWeightNorm {
            magnitude,
            direction,
            dimension,
        };
        materialize_weight_norm(backend, &parametrization, context)?;
        let next_generation = self.next_generation()?;
        context.cancellation.check()?;
        self.weight = Some(NativeWeight::WeightNorm(parametrization));
        self.weight_norm_dimension = Some(dimension);
        self.bias = bias;
        self.prefetched = None;
        self.generation = next_generation;
        Ok(())
    }

    pub fn has_weight_parametrization(&self) -> bool {
        self.weight_norm_dimension.is_some() || self.spectral_norm_config.is_some()
    }

    pub fn has_spectral_parametrization(&self) -> bool {
        self.spectral_norm_config.is_some()
    }

    pub fn register_weight_norm_exact_native(
        &mut self,
        name: &str,
        dimension: Option<usize>,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeOpsError> {
        cancellation.check()?;
        if name != "weight" {
            return Err(NativeOpsError::Invalid(
                "native modules support weight normalization only for weight",
            ));
        }
        if self.weight_norm_dimension.is_some() {
            return Err(NativeOpsError::Invalid(
                "module weight already has a parametrization",
            ));
        }
        if self.spectral_norm_config.is_some() {
            return Err(NativeOpsError::Invalid(
                "module weight already has spectral normalization",
            ));
        }
        if self.weight.is_some() {
            return Err(NativeOpsError::Invalid(
                "weight normalization must be registered before native parameters are loaded",
            ));
        }
        let rank = self.parameter_shape("weight")?.len();
        if dimension.is_some_and(|dimension| dimension >= rank) {
            return Err(NativeOpsError::Invalid(
                "weight-normalization dimension is outside the weight rank",
            ));
        }
        let next_generation = self.next_generation()?;
        cancellation.check()?;
        self.weight_norm_dimension = Some(dimension);
        self.prefetched = None;
        self.generation = next_generation;
        Ok(())
    }

    pub fn register_spectral_norm_exact_native(
        &mut self,
        name: &str,
        power_iterations: usize,
        epsilon: f32,
        dimension: Option<usize>,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeOpsError> {
        cancellation.check()?;
        if name != "weight" {
            return Err(NativeOpsError::Invalid(
                "native modules support spectral normalization only for weight",
            ));
        }
        if power_iterations == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(NativeOpsError::Invalid(
                "spectral normalization requires positive power iterations and epsilon",
            ));
        }
        if self.weight_norm_dimension.is_some() || self.spectral_norm_config.is_some() {
            return Err(NativeOpsError::Invalid(
                "module weight already has a parametrization",
            ));
        }
        if self.weight.is_some() {
            return Err(NativeOpsError::Invalid(
                "spectral normalization must be registered before native parameters are loaded",
            ));
        }
        let weight_shape = self.parameter_shape("weight")?;
        let dimension = dimension.unwrap_or_else(|| match &self.spec {
            NativeModuleSpec::Convolution { geometry, .. } if geometry.transposed() => 1,
            _ => 0,
        });
        if dimension >= weight_shape.len() {
            return Err(NativeOpsError::Invalid(
                "spectral-normalization dimension is outside the weight rank",
            ));
        }
        let next_generation = self.next_generation()?;
        cancellation.check()?;
        self.spectral_norm_config = Some(NativeSpectralNormConfig {
            dimension,
            power_iterations,
            epsilon,
        });
        self.prefetched = None;
        self.generation = next_generation;
        Ok(())
    }

    pub fn remove_parametrizations_with_context_exact_native(
        &mut self,
        backend: &CpuBackend,
        name: &str,
        leave_parametrized: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        context.cancellation.check()?;
        self.remove_parametrizations_impl(backend, name, leave_parametrized, context)
    }

    fn remove_parametrizations_impl(
        &mut self,
        backend: &CpuBackend,
        name: &str,
        leave_parametrized: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        if name != "weight" {
            return Err(NativeOpsError::Invalid(
                "native modules support parametrization removal only for weight",
            ));
        }
        if !leave_parametrized {
            return Err(NativeOpsError::Invalid(
                "weight norm has multiple original tensors and cannot be restored as one weight",
            ));
        }
        let NativeWeight::WeightNorm(parametrization) = self
            .weight
            .as_ref()
            .ok_or(NativeOpsError::ParametersNotLoaded)?
        else {
            return Err(NativeOpsError::Invalid("module weight is not parametrized"));
        };
        let materialized = materialize_weight_norm(backend, parametrization, context)?;
        let next_generation = self.next_generation()?;
        context.cancellation.check()?;
        self.weight = Some(NativeWeight::Dense(materialized));
        self.weight_norm_dimension = None;
        self.prefetched = None;
        self.generation = next_generation;
        Ok(())
    }

    pub fn zero_init_parameter_with_context_exact_native(
        &mut self,
        backend: &CpuBackend,
        name: &str,
        fallback_dtype: DType,
        fallback_device: DeviceId,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        self.zero_init_parameter_impl(backend, name, fallback_dtype, fallback_device, context)
    }

    fn zero_init_parameter_impl(
        &mut self,
        backend: &CpuBackend,
        name: &str,
        fallback_dtype: DType,
        fallback_device: DeviceId,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        context.cancellation.check()?;
        let shape = self.parameter_shape(name)?;
        let (dtype, device) = match name {
            "weight" => match self.weight.as_ref() {
                Some(NativeWeight::Dense(parameter)) => (
                    parameter.descriptor().dtype(),
                    parameter.descriptor().device(),
                ),
                Some(NativeWeight::Quantized(_)) => {
                    return Err(NativeOpsError::Invalid(
                        "quantized weights cannot be replaced by zero initialization",
                    ));
                }
                Some(NativeWeight::WeightNorm(parametrization)) => (
                    parametrization.direction.descriptor().dtype(),
                    parametrization.direction.descriptor().device(),
                ),
                Some(NativeWeight::SpectralNorm(_)) => {
                    return Err(NativeOpsError::Invalid(
                        "spectral-normalized weights cannot be replaced by zero initialization",
                    ));
                }
                None => (fallback_dtype, fallback_device),
            },
            "bias" => self
                .bias
                .as_ref()
                .map_or((fallback_dtype, fallback_device), |parameter| {
                    (
                        parameter.descriptor().dtype(),
                        parameter.descriptor().device(),
                    )
                }),
            _ => {
                return Err(NativeOpsError::Invalid(
                    "only weight and bias parameters can be zero initialized",
                ));
            }
        };
        let value_count = checked_product(&shape, "zero-initialized parameter shape")?;
        let values = temporary_filled(backend, context, value_count, 0.0_f32)?;
        let tensor = tensor_from_f32(
            backend,
            &shape_to_u64(&shape)?,
            &values,
            dtype,
            device,
            context,
        )?;
        let next_generation = self.next_generation()?;
        context.cancellation.check()?;
        match name {
            "weight" => self.weight = Some(NativeWeight::Dense(tensor)),
            "bias" => self.bias = Some(tensor),
            _ => {
                return Err(NativeOpsError::Invalid(
                    "only weight and bias parameters can be zero initialized",
                ));
            }
        }
        self.prefetched = None;
        self.generation = next_generation;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn cast_bias_weight_with_context_exact_native(
        &mut self,
        backend: &CpuBackend,
        input: Option<&Tensor>,
        dtype: Option<DType>,
        device: Option<DeviceId>,
        bias_dtype: Option<DType>,
        offloadable: bool,
        compute_dtype: Option<DType>,
        want_requant: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<CastedParameters, NativeOpsError> {
        self.cast_bias_weight_impl(
            backend,
            input,
            dtype,
            device,
            bias_dtype,
            offloadable,
            compute_dtype,
            want_requant,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cast_bias_weight_impl(
        &mut self,
        backend: &CpuBackend,
        input: Option<&Tensor>,
        dtype: Option<DType>,
        device: Option<DeviceId>,
        bias_dtype: Option<DType>,
        offloadable: bool,
        compute_dtype: Option<DType>,
        want_requant: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<CastedParameters, NativeOpsError> {
        let mut prepared = self.prepare_parameters(
            backend,
            input,
            dtype,
            device,
            bias_dtype,
            compute_dtype,
            want_requant,
            context,
        )?;
        context.cancellation.check()?;
        let next_generation = self.next_generation()?;
        if let Some(next_weight) = prepared.next_weight.take() {
            self.weight = Some(next_weight);
            self.prefetched = None;
        }
        self.generation = next_generation;
        Ok(CastedParameters {
            weight: prepared.weight,
            bias: prepared.bias,
            requantized_weight: prepared.requantized_weight,
            lease: WeightCastLease {
                generation: self.generation,
                offloadable,
                completed: false,
            },
        })
    }

    pub fn forward_with_context(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeOpsError> {
        self.forward_with_autopad_impl(backend, input, ConvolutionAutopad::Disabled, context)
    }

    pub fn forward_quantized_autograd_with_context(
        &self,
        backend: &dyn TensorBackend,
        input: &Tensor,
        layout: QuantLinearLayout,
        input_scale: crate::QuantLinearScale,
        weight_requires_grad: bool,
        fp8_backward: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<QuantLinearExecution, crate::QuantLinearError> {
        if !matches!(self.spec, NativeModuleSpec::Linear { .. }) {
            return Err(NativeOpsError::Invalid(
                "quantized autograd is defined only for linear modules",
            )
            .into());
        }
        context.check()?;
        let weight = match self
            .weight
            .as_ref()
            .ok_or(NativeOpsError::ParametersNotLoaded)?
        {
            NativeWeight::Dense(weight) => {
                QuantLinearWeight::Dense(cast_to_with_backend_exact_native(
                    backend,
                    weight,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    false,
                    self.manual_cast,
                    context,
                )?)
            }
            NativeWeight::Quantized(matrix) => QuantLinearWeight::CatalogQuantized(matrix.clone()),
            NativeWeight::WeightNorm(_) | NativeWeight::SpectralNorm(_) => {
                return Err(NativeOpsError::Invalid(
                    "quantized autograd requires materialized linear parameters",
                )
                .into());
            }
        };
        let bias = self
            .bias
            .as_ref()
            .map(|bias| {
                cast_to_with_backend_exact_native(
                    backend,
                    bias,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    false,
                    self.manual_cast,
                    context,
                )
            })
            .transpose()?;
        quant_linear_forward_exact_native(
            backend,
            input,
            weight,
            bias.as_ref(),
            QuantLinearOptions {
                layout: Some(layout),
                input_scale,
                compute_dtype: input.descriptor().dtype(),
                weight_requires_grad,
                fp8_backward,
            },
            context,
        )
    }

    pub fn forward_with_rng_with_context(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        transaction: RngTransaction,
        context: &ExecutionContext<'_>,
    ) -> Result<RngAwareModuleForward, NativeOpsError> {
        context.cancellation.check()?;
        match self.spec.clone() {
            NativeModuleSpec::Dropout {
                probability,
                training,
            } => {
                let values = tensor_to_f32(backend, input, context)?;
                let result = dropout_with_context_exact_native(
                    &values,
                    probability,
                    training,
                    transaction,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    input.descriptor().shape(),
                    &result.values,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                let output = self.complete_forward(output, context)?;
                Ok(RngAwareModuleForward {
                    output,
                    transaction: result.transaction,
                })
            }
            NativeModuleSpec::Sequential => {
                let mut staged_children = self.children.clone();
                let mut output = input.clone();
                let mut transaction = transaction;
                for child in &mut staged_children {
                    context.cancellation.check()?;
                    let result = child.forward_with_rng_with_context(
                        backend,
                        &output,
                        transaction,
                        context,
                    )?;
                    output = result.output;
                    transaction = result.transaction;
                }
                context.cancellation.check()?;
                let next_generation = self.next_generation()?;
                self.children = staged_children;
                self.generation = next_generation;
                Ok(RngAwareModuleForward {
                    output,
                    transaction,
                })
            }
            _ => {
                let output = self.forward_with_context(backend, input, context)?;
                Ok(RngAwareModuleForward {
                    output,
                    transaction,
                })
            }
        }
    }

    pub fn forward_attention_with_context(
        &mut self,
        backend: &CpuBackend,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeOpsError> {
        context.cancellation.check()?;
        let NativeModuleSpec::MultiheadAttention {
            embed_dimension,
            heads,
        } = self.spec
        else {
            return Err(NativeOpsError::Invalid(
                "only a multihead-attention module accepts query, key, and value tensors",
            ));
        };
        if query.descriptor().dtype() != key.descriptor().dtype()
            || query.descriptor().dtype() != value.descriptor().dtype()
            || query.descriptor().device() != key.descriptor().device()
            || query.descriptor().device() != value.descriptor().device()
        {
            return Err(NativeOpsError::Invalid(
                "multihead-attention inputs must share dtype and device",
            ));
        }
        let mut staged_children = self.children.clone();
        if staged_children.len() != 4 {
            return Err(NativeOpsError::Invalid(
                "multihead-attention module must own four projection children",
            ));
        }
        let projected_query = staged_children
            .get_mut(0)
            .ok_or(NativeOpsError::Invalid("missing query projection"))?
            .forward_with_context(backend, query, context)?;
        let projected_key = staged_children
            .get_mut(1)
            .ok_or(NativeOpsError::Invalid("missing key projection"))?
            .forward_with_context(backend, key, context)?;
        let projected_value = staged_children
            .get_mut(2)
            .ok_or(NativeOpsError::Invalid("missing value projection"))?
            .forward_with_context(backend, value, context)?;
        let query_shape = shape_to_usize(projected_query.descriptor().shape())?;
        let key_shape = shape_to_usize(projected_key.descriptor().shape())?;
        let value_shape = shape_to_usize(projected_value.descriptor().shape())?;
        if query_shape.last() != Some(&embed_dimension)
            || key_shape.last() != Some(&embed_dimension)
            || value_shape.last() != Some(&embed_dimension)
        {
            return Err(NativeOpsError::Invalid(
                "multihead-attention projection width does not match the module",
            ));
        }
        let projected = multihead_attention_projected_with_context_exact_native(
            backend,
            &tensor_to_f32(backend, &projected_query, context)?,
            &query_shape,
            &tensor_to_f32(backend, &projected_key, context)?,
            &key_shape,
            &tensor_to_f32(backend, &projected_value, context)?,
            &value_shape,
            heads,
            context,
        )?;
        let projected = tensor_from_f32(
            backend,
            &shape_to_u64(&projected.shape)?,
            &projected.values,
            query.descriptor().dtype(),
            query.descriptor().device(),
            context,
        )?;
        let output = staged_children
            .get_mut(3)
            .ok_or(NativeOpsError::Invalid("missing output projection"))?
            .forward_with_context(backend, &projected, context)?;
        context.cancellation.check()?;
        let next_generation = self.next_generation()?;
        self.children = staged_children;
        self.generation = next_generation;
        Ok(output)
    }

    pub fn forward_if_dense_weight_is_zero_with_context(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, NativeOpsError> {
        self.forward_if_dense_weight_is_zero_impl(backend, input, context)
    }

    fn forward_if_dense_weight_is_zero_impl(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, NativeOpsError> {
        context.cancellation.check()?;
        self.validate_parameter_presence()?;
        let Some(NativeWeight::Dense(weight)) = &self.weight else {
            return Ok(None);
        };
        if !tensor_to_f32(backend, weight, context)?
            .iter()
            .all(|value| *value == 0.0)
        {
            return Ok(None);
        }
        if !tensor_to_f32(backend, input, context)?
            .iter()
            .all(|value| value.is_finite())
        {
            return Ok(None);
        }
        let input_shape = shape_to_usize(input.descriptor().shape())?;
        let (output_shape, output_channels) = match &self.spec {
            NativeModuleSpec::Linear {
                input_features,
                output_features,
                ..
            } => {
                if input_shape.last() != Some(input_features) {
                    return Err(NativeOpsError::Invalid(
                        "linear input feature dimension does not match the module",
                    ));
                }
                let mut output_shape = input_shape;
                let Some(last) = output_shape.last_mut() else {
                    return Err(NativeOpsError::Invalid(
                        "linear input rank must be positive",
                    ));
                };
                *last = *output_features;
                (output_shape, *output_features)
            }
            NativeModuleSpec::Convolution {
                input_channels,
                output_channels,
                kernel_shape,
                geometry,
                ..
            } => {
                if input_shape.len() != kernel_shape.len() + 2
                    || input_shape.get(1) != Some(input_channels)
                {
                    return Err(NativeOpsError::Invalid(
                        "convolution input shape does not match the module",
                    ));
                }
                let mut output_shape = vec![input_shape[0], *output_channels];
                for spatial in 0..kernel_shape.len() {
                    let padded =
                        input_shape[spatial + 2]
                            .checked_add(geometry.padding()[spatial].checked_mul(2).ok_or(
                                NativeOpsError::Invalid("convolution output shape overflow"),
                            )?)
                            .ok_or(NativeOpsError::Invalid("convolution output shape overflow"))?;
                    let receptive = geometry.dilation()[spatial]
                        .checked_mul(kernel_shape[spatial].saturating_sub(1))
                        .and_then(|value| value.checked_add(1))
                        .ok_or(NativeOpsError::Invalid("convolution output shape overflow"))?;
                    if padded < receptive {
                        return Err(NativeOpsError::Invalid(
                            "convolution kernel exceeds padded input",
                        ));
                    }
                    output_shape.push((padded - receptive) / geometry.stride()[spatial] + 1);
                }
                (output_shape, *output_channels)
            }
            _ => return Ok(None),
        };
        let output_count = checked_product(&output_shape, "zero-weight output shape")?;
        let bias_values = self
            .bias
            .as_ref()
            .map(|bias| tensor_to_f32(backend, bias, context))
            .transpose()?;
        let mut output = temporary_filled(backend, context, output_count, 0.0_f32)?;
        if let Some(bias_values) = bias_values {
            if bias_values.len() != output_channels {
                return Err(NativeOpsError::Invalid(
                    "module bias shape does not match its output channels",
                ));
            }
            if output_shape.len() == 2 {
                for row in output.chunks_exact_mut(output_channels) {
                    context.cancellation.check()?;
                    row.copy_from_slice(&bias_values);
                }
            } else {
                let spatial = checked_product(
                    output_shape.get(2..).ok_or(NativeOpsError::Invalid(
                        "convolution output rank must be at least three",
                    ))?,
                    "zero-weight convolution spatial shape",
                )?;
                for batch in 0..output_shape[0] {
                    for (channel, bias) in bias_values.iter().copied().enumerate() {
                        context.cancellation.check()?;
                        let start = (batch * output_channels + channel)
                            .checked_mul(spatial)
                            .ok_or(NativeOpsError::Invalid(
                                "zero-weight module output offset overflow",
                            ))?;
                        output[start..start + spatial].fill(bias);
                    }
                }
            }
        }
        let output_shape = output_shape
            .iter()
            .map(|dimension| {
                u64::try_from(*dimension)
                    .map_err(|_| NativeOpsError::Invalid("module output shape overflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        tensor_from_f32(
            backend,
            &output_shape,
            &output,
            input.descriptor().dtype(),
            input.descriptor().device(),
            context,
        )
        .map(Some)
    }

    pub fn forward_dense_inference_with_context(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeOpsError> {
        context.cancellation.check()?;
        if !matches!(
            input.descriptor().dtype(),
            DType::F16 | DType::Bf16 | DType::F32
        ) {
            return Err(NativeOpsError::Invalid(
                "immutable dense inference requires a supported floating-point dtype",
            ));
        }
        if input.descriptor().device() != backend.device() {
            return Err(NativeOpsError::BackendTargetMismatch {
                requested: input.descriptor().device(),
                backend: backend.device(),
            });
        }
        if input.descriptor().stream() != context.stream {
            return Err(NativeOpsError::ExecutionStreamMismatch {
                requested: input.descriptor().stream(),
                context: context.stream,
            });
        }
        if !input.descriptor().is_contiguous()? {
            return Err(NativeOpsError::Invalid(
                "immutable dense inference requires contiguous input",
            ));
        }
        let output_dtype = input.descriptor().dtype();
        let output_device = input.descriptor().device();
        let input_shape = shape_to_usize(input.descriptor().shape())?;
        match &self.spec {
            NativeModuleSpec::Convolution {
                input_channels,
                kernel_shape,
                geometry,
                ..
            } => {
                if geometry.spatial_dimensions() != 2 {
                    return Err(NativeOpsError::Invalid(
                        "immutable dense spatial inference supports only Conv2d modules",
                    ));
                }
                if geometry.padding_mode() != ConvolutionPaddingMode::Zeros {
                    return Err(NativeOpsError::Invalid(
                        "immutable dense spatial inference requires zero padding",
                    ));
                }
                if input_shape.len() != 4 || input_shape.get(1) != Some(input_channels) {
                    return Err(NativeOpsError::Invalid(
                        "convolution input shape does not match the module",
                    ));
                }
                if kernel_shape.len() != 2 {
                    return Err(NativeOpsError::Invalid(
                        "convolution kernel rank does not match Conv2d",
                    ));
                }
                let (weight, bias) = self.dense_inference_parameters(input)?;
                let configuration = ConvolutionConfiguration {
                    stride: geometry.stride().to_vec(),
                    padding: geometry.padding().to_vec(),
                    dilation: geometry.dilation().to_vec(),
                    groups: geometry.groups(),
                    output_padding: geometry.output_padding().to_vec(),
                };
                let output = if geometry.transposed() {
                    conv_transpose_2d_tensor_with_context_exact_native(
                        backend,
                        input,
                        weight,
                        bias,
                        &configuration,
                        context,
                    )?
                } else {
                    conv_2d_tensor_with_context_exact_native(
                        backend,
                        input,
                        weight,
                        bias,
                        &configuration,
                        context,
                    )?
                };
                context.check()?;
                return Ok(output);
            }
            NativeModuleSpec::InstanceNorm2d {
                features,
                epsilon,
                affine,
            } => {
                if *affine {
                    return Err(NativeOpsError::Invalid(
                        "immutable dense inference supports only parameterless InstanceNorm2d",
                    ));
                }
                if input_shape.len() != 4 || input_shape.get(1) != Some(features) {
                    return Err(NativeOpsError::Invalid(
                        "instance-normalization input rank or channels do not match the module",
                    ));
                }
                self.validate_parameter_presence()?;
                let output = instance_norm_2d_tensor_with_context_exact_native(
                    backend,
                    input,
                    f64::from(*epsilon),
                    context,
                )?;
                context.check()?;
                return Ok(output);
            }
            NativeModuleSpec::Silu => {
                self.validate_parameter_presence()?;
                let output = silu_tensor_with_context_exact_native(backend, input, context)?;
                context.check()?;
                return Ok(output);
            }
            NativeModuleSpec::LeakyRelu { negative_slope } => {
                self.validate_parameter_presence()?;
                let output = leaky_relu_tensor_with_context_exact_native(
                    backend,
                    input,
                    *negative_slope,
                    context,
                )?;
                context.check()?;
                return Ok(output);
            }
            _ => {}
        }
        self.execution_requirements(DType::F32)
            .admit_backend_target(
                backend,
                input.descriptor().device(),
                DType::F32,
                Layout::Contiguous,
                input.descriptor().stream(),
                context,
            )?;
        let input_values = tensor_to_f32(backend, input, context)?;
        let result = match &self.spec {
            NativeModuleSpec::Linear {
                input_features,
                output_features,
                ..
            } => {
                let (weight, bias) = self.dense_inference_parameters(input)?;
                let weight_values = tensor_to_f32(backend, weight, context)?;
                let bias_values = bias
                    .map(|bias| tensor_to_f32(backend, bias, context))
                    .transpose()?;
                linear_with_context_exact_native(
                    &input_values,
                    &input_shape,
                    &weight_values,
                    &[*output_features, *input_features],
                    bias_values.as_deref(),
                    output_device,
                    context,
                )?
            }
            NativeModuleSpec::LayerNorm {
                normalized_shape,
                epsilon,
                elementwise_affine,
                ..
            } => {
                let (weight_values, bias_values) = if *elementwise_affine {
                    let (weight, bias) = self.dense_inference_parameters(input)?;
                    (
                        Some(tensor_to_f32(backend, weight, context)?),
                        bias.map(|bias| tensor_to_f32(backend, bias, context))
                            .transpose()?,
                    )
                } else {
                    self.validate_parameter_presence()?;
                    (None, None)
                };
                TensorValues {
                    values: layer_norm_with_context_exact_native(
                        backend,
                        &input_values,
                        &input_shape,
                        normalized_shape,
                        weight_values.as_deref(),
                        bias_values.as_deref(),
                        *epsilon,
                        output_device,
                        context,
                    )?,
                    shape: input_shape,
                }
            }
            NativeModuleSpec::GroupNorm {
                groups,
                channels,
                epsilon,
                affine,
            } => {
                if input_shape.len() < 2 || input_shape.get(1) != Some(channels) {
                    return Err(NativeOpsError::Invalid(
                        "group-normalization input channels do not match the module",
                    ));
                }
                let (weight_values, bias_values) = if *affine {
                    let (weight, bias) = self.dense_inference_parameters(input)?;
                    (
                        Some(tensor_to_f32(backend, weight, context)?),
                        bias.map(|bias| tensor_to_f32(backend, bias, context))
                            .transpose()?,
                    )
                } else {
                    self.validate_parameter_presence()?;
                    (None, None)
                };
                TensorValues {
                    values: group_norm_with_context_exact_native(
                        backend,
                        &input_values,
                        &input_shape,
                        *groups,
                        weight_values.as_deref(),
                        bias_values.as_deref(),
                        *epsilon,
                        output_device,
                        context,
                    )?,
                    shape: input_shape,
                }
            }
            NativeModuleSpec::Gelu { approximation } => TensorValues {
                values: gelu_module_with_context_exact_native(
                    backend,
                    &input_values,
                    *approximation,
                    output_device,
                    context,
                )?,
                shape: input_shape,
            },
            NativeModuleSpec::Convolution { .. }
            | NativeModuleSpec::InstanceNorm2d { .. }
            | NativeModuleSpec::Silu => {
                return Err(NativeOpsError::Invalid(
                    "bounded dense spatial module dispatch was not handled",
                ));
            }
            _ => {
                return Err(NativeOpsError::Invalid(
                    "module does not support immutable dense inference",
                ));
            }
        };
        let output = tensor_from_f32(
            backend,
            &shape_to_u64(&result.shape)?,
            &result.values,
            output_dtype,
            output_device,
            context,
        )?;
        context.cancellation.check()?;
        Ok(output)
    }

    fn dense_inference_parameters(
        &self,
        input: &Tensor,
    ) -> Result<(&Tensor, Option<&Tensor>), NativeOpsError> {
        self.validate_parameter_presence()?;
        let (weight, bias) = self.dense_parameters()?;
        for parameter in [Some(weight), bias].into_iter().flatten() {
            if parameter.descriptor().dtype() != input.descriptor().dtype()
                || parameter.descriptor().device() != input.descriptor().device()
                || parameter.descriptor().stream() != input.descriptor().stream()
            {
                return Err(NativeOpsError::Invalid(
                    "immutable dense inference parameters must match input dtype, device, and stream",
                ));
            }
        }
        Ok((weight, bias))
    }

    pub fn forward_with_autopad_with_context(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        autopad: ConvolutionAutopad,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeOpsError> {
        self.forward_with_autopad_impl(backend, input, autopad, context)
    }

    fn forward_with_autopad_impl(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        autopad: ConvolutionAutopad,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeOpsError> {
        self.execution_requirements(input.descriptor().dtype())
            .admit_backend_target(
                backend,
                input.descriptor().device(),
                input.descriptor().dtype(),
                Layout::Contiguous,
                input.descriptor().stream(),
                context,
            )?;
        if matches!(self.spec, NativeModuleSpec::Container) {
            return Err(NativeOpsError::Invalid(
                "a base module container has no forward implementation",
            ));
        }
        if autopad != ConvolutionAutopad::Disabled
            && !matches!(self.spec, NativeModuleSpec::Convolution { .. })
        {
            return Err(NativeOpsError::Invalid(
                "causal autopad is valid only for convolution modules",
            ));
        }
        match self.spec.clone() {
            NativeModuleSpec::Buffer => {
                return Err(NativeOpsError::Invalid(
                    "a registered buffer is state and has no forward implementation",
                ));
            }
            NativeModuleSpec::Sequential => {
                let mut staged_children = self.children.clone();
                let mut output = input.clone();
                for child in &mut staged_children {
                    context.cancellation.check()?;
                    output = child.forward_with_context(backend, &output, context)?;
                }
                context.cancellation.check()?;
                let next_generation = self.next_generation()?;
                self.children = staged_children;
                self.generation = next_generation;
                return Ok(output);
            }
            NativeModuleSpec::AveragePool1d {
                kernel_size,
                stride,
            } => {
                let values = tensor_to_f32(backend, input, context)?;
                let shape = shape_to_usize(input.descriptor().shape())?;
                let result = average_pool_1d_with_context_exact_native(
                    backend,
                    &values,
                    &shape,
                    kernel_size,
                    stride,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    &shape_to_u64(&result.shape)?,
                    &result.values,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::AveragePool2d {
                kernel_size,
                stride,
            } => {
                let values = tensor_to_f32(backend, input, context)?;
                let shape = shape_to_usize(input.descriptor().shape())?;
                let result = average_pool_2d_with_context_exact_native(
                    backend,
                    &values,
                    &shape,
                    kernel_size,
                    stride,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    &shape_to_u64(&result.shape)?,
                    &result.values,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Gelu { approximation } => {
                let values = tensor_to_f32(backend, input, context)?;
                let result = gelu_module_with_context_exact_native(
                    backend,
                    &values,
                    approximation,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    input.descriptor().shape(),
                    &result,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Dropout {
                probability: _,
                training,
            } => {
                if training {
                    return Err(NativeOpsError::Invalid(
                        "training dropout requires forward_with_rng_with_context",
                    ));
                }
                let output = identity_with_context_exact_native(input, context)?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Elu { alpha } => {
                let values = tensor_to_f32(backend, input, context)?;
                let result = elu_module_with_context_exact_native(
                    backend,
                    &values,
                    alpha,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    input.descriptor().shape(),
                    &result,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Identity => {
                let output = identity_with_context_exact_native(input, context)?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::MseLoss { .. } => {
                return Err(NativeOpsError::Invalid(
                    "MSE loss requires forward_loss_with_context and a target tensor",
                ));
            }
            NativeModuleSpec::AdaptiveAveragePool2d { output_size } => {
                let values = tensor_to_f32(backend, input, context)?;
                let shape = shape_to_usize(input.descriptor().shape())?;
                let result = adaptive_average_pool_2d_module_with_context_exact_native(
                    backend,
                    &values,
                    &shape,
                    output_size,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    &shape_to_u64(&result.shape)?,
                    &result.values,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::AveragePool3d {
                kernel_size,
                stride,
            } => {
                let values = tensor_to_f32(backend, input, context)?;
                let shape = shape_to_usize(input.descriptor().shape())?;
                let result = average_pool_3d_module_with_context_exact_native(
                    backend,
                    &values,
                    &shape,
                    kernel_size,
                    stride,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    &shape_to_u64(&result.shape)?,
                    &result.values,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Embedding { options, .. } => {
                let mut prepared = self.prepare_parameters(
                    backend,
                    None,
                    Some(DType::F32),
                    Some(input.descriptor().device()),
                    None,
                    Some(DType::F32),
                    false,
                    context,
                )?;
                if options.max_norm.is_some()
                    && !matches!(self.weight, Some(NativeWeight::Dense(_)))
                {
                    return Err(NativeOpsError::Invalid(
                        "embedding max-norm mutation requires an ordinary dense weight",
                    ));
                }
                let output = embedding_module_with_context_exact_native(
                    backend,
                    input,
                    &mut prepared.weight,
                    options,
                    context,
                )?;
                context.cancellation.check()?;
                let next_generation = self.next_generation()?;
                if options.max_norm.is_some() {
                    self.weight = Some(NativeWeight::Dense(prepared.weight));
                    self.prefetched = None;
                } else if let Some(next_weight) = prepared.next_weight.take() {
                    self.weight = Some(next_weight);
                    self.prefetched = None;
                }
                self.generation = next_generation;
                return Ok(output);
            }
            NativeModuleSpec::LeakyRelu { negative_slope } => {
                let values = tensor_to_f32(backend, input, context)?;
                let result = leaky_relu_module_with_context_exact_native(
                    backend,
                    &values,
                    negative_slope,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    input.descriptor().shape(),
                    &result,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::MaxPool2d {
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            } => {
                let values = tensor_to_f32(backend, input, context)?;
                let shape = shape_to_usize(input.descriptor().shape())?;
                let result = max_pool_2d_with_context_exact_native(
                    &values,
                    &shape,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    &shape_to_u64(&result.shape)?,
                    &result.values,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::ModuleDict | NativeModuleSpec::ModuleList => {
                return Err(NativeOpsError::Invalid(
                    "module collections own child lifecycle but do not define forward order",
                ));
            }
            NativeModuleSpec::PixelShuffle { factor } => {
                let output = pixel_shuffle_module_with_context_exact_native(
                    backend, input, factor, context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::PixelUnshuffle { factor } => {
                let output = pixel_unshuffle_module_with_context_exact_native(
                    backend, input, factor, context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::ReplicationPad2d { padding } => {
                let values = tensor_to_f32(backend, input, context)?;
                let shape = shape_to_usize(input.descriptor().shape())?;
                let result = replication_pad_2d_with_context_exact_native(
                    &values,
                    &shape,
                    padding,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    &shape_to_u64(&result.shape)?,
                    &result.values,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Relu | NativeModuleSpec::Relu6 => {
                let values = tensor_to_f32(backend, input, context)?;
                let result = if matches!(self.spec, NativeModuleSpec::Relu6) {
                    relu_6_with_context_exact_native(
                        backend,
                        &values,
                        input.descriptor().device(),
                        context,
                    )?
                } else {
                    relu_module_with_context_exact_native(
                        backend,
                        &values,
                        input.descriptor().device(),
                        context,
                    )?
                };
                let output = tensor_from_f32(
                    backend,
                    input.descriptor().shape(),
                    &result,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Silu => {
                let values = tensor_to_f32(backend, input, context)?;
                let result = silu_module_with_context_exact_native(
                    backend,
                    &values,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    input.descriptor().shape(),
                    &result,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Sigmoid => {
                let output = sigmoid_module_with_context_exact_native(backend, input, context)?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Softmax { dimension } => {
                let values = tensor_to_f32(backend, input, context)?;
                let shape = shape_to_usize(input.descriptor().shape())?;
                let result = softmax_module_with_context_exact_native(
                    backend,
                    &values,
                    &shape,
                    dimension,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    input.descriptor().shape(),
                    &result,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Tanh => {
                let output = tanh_module_with_context_exact_native(backend, input, context)?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::Upsample {
                scale_factor,
                mode,
                align_corners,
            } => {
                let shape = input.descriptor().shape();
                if shape.len() < 2 {
                    return Err(NativeOpsError::Invalid(
                        "upsample input must have at least two spatial dimensions",
                    ));
                }
                let output_height =
                    scaled_dimension(shape[shape.len() - 2], scale_factor[0], "upsample height")?;
                let output_width =
                    scaled_dimension(shape[shape.len() - 1], scale_factor[1], "upsample width")?;
                let output = upsample_with_context_exact_native(
                    backend,
                    input,
                    output_height,
                    output_width,
                    mode,
                    align_corners,
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::ZeroPad2d { padding } => {
                let values = tensor_to_f32(backend, input, context)?;
                let shape = shape_to_usize(input.descriptor().shape())?;
                let result = zero_pad_2d_with_context_exact_native(
                    &values,
                    &shape,
                    padding,
                    input.descriptor().device(),
                    context,
                )?;
                let output = tensor_from_f32(
                    backend,
                    &shape_to_u64(&result.shape)?,
                    &result.values,
                    input.descriptor().dtype(),
                    input.descriptor().device(),
                    context,
                )?;
                return self.complete_forward(output, context);
            }
            NativeModuleSpec::SmoothL1Loss { .. }
            | NativeModuleSpec::HuberLoss { .. }
            | NativeModuleSpec::L1Loss { .. } => {
                return Err(NativeOpsError::Invalid(
                    "loss modules require both input and target tensors",
                ));
            }
            NativeModuleSpec::MultiheadAttention { .. } => {
                return Err(NativeOpsError::Invalid(
                    "multihead attention requires query, key, and value tensors",
                ));
            }
            NativeModuleSpec::Container
            | NativeModuleSpec::Linear { .. }
            | NativeModuleSpec::Convolution { .. }
            | NativeModuleSpec::BatchNorm { .. }
            | NativeModuleSpec::LayerNorm { .. }
            | NativeModuleSpec::GroupNorm { .. }
            | NativeModuleSpec::InstanceNorm2d { .. }
            | NativeModuleSpec::PRelu { .. } => {}
        }
        let output_dtype = input.descriptor().dtype();
        let output_device = input.descriptor().device();
        let input_values = tensor_to_f32(backend, input, context)?;
        let input_shape = shape_to_usize(input.descriptor().shape())?;
        if let NativeModuleSpec::GroupNorm { channels, .. } = &self.spec
            && (input_shape.len() < 2 || input_shape.get(1) != Some(channels))
        {
            return Err(NativeOpsError::Invalid(
                "group-normalization input channels do not match the module",
            ));
        }
        if let NativeModuleSpec::InstanceNorm2d { features, .. } = &self.spec
            && (input_shape.len() != 4 || input_shape.get(1) != Some(features))
        {
            return Err(NativeOpsError::Invalid(
                "instance-normalization input rank or channels do not match the module",
            ));
        }
        if let NativeModuleSpec::BatchNorm {
            dimensions,
            features,
            epsilon,
            momentum,
            affine,
            track_running_stats,
            training,
        } = self.spec.clone()
        {
            let valid_rank = match dimensions {
                1 => matches!(input_shape.len(), 2 | 3),
                2 => input_shape.len() == 4,
                _ => false,
            };
            if !valid_rank || input_shape.get(1) != Some(&features) {
                return Err(NativeOpsError::Invalid(
                    "batch-normalization input rank or channels do not match the module",
                ));
            }
            let mut prepared = affine
                .then(|| {
                    self.prepare_parameters(
                        backend,
                        Some(input),
                        Some(output_dtype),
                        Some(output_device),
                        Some(output_dtype),
                        Some(DType::F32),
                        false,
                        context,
                    )
                })
                .transpose()?;
            let weight_values = prepared
                .as_ref()
                .map(|parameters| tensor_to_f32(backend, &parameters.weight, context))
                .transpose()?;
            let bias_values = prepared
                .as_ref()
                .and_then(|parameters| parameters.bias.as_ref())
                .map(|bias| tensor_to_f32(backend, bias, context))
                .transpose()?;
            let mut staged_state = self.normalization_state.clone();
            let use_batch_statistics = training || !track_running_stats;
            let operation = if dimensions == 1 {
                BATCH_NORM_1D_OPERATION_ID
            } else {
                BATCH_NORM_2D_OPERATION_ID
            };
            let values = match staged_state.as_mut() {
                Some(NativeNormalizationState {
                    running_mean,
                    running_variance,
                }) => batch_norm_module_with_context_exact_native(
                    backend,
                    &input_values,
                    &input_shape,
                    input_shape.len(),
                    Some(running_mean),
                    Some(running_variance),
                    weight_values.as_deref(),
                    bias_values.as_deref(),
                    use_batch_statistics,
                    momentum,
                    epsilon,
                    operation,
                    DeviceId::CPU,
                    context,
                )?,
                None => batch_norm_module_with_context_exact_native(
                    backend,
                    &input_values,
                    &input_shape,
                    input_shape.len(),
                    None,
                    None,
                    weight_values.as_deref(),
                    bias_values.as_deref(),
                    true,
                    momentum,
                    epsilon,
                    operation,
                    DeviceId::CPU,
                    context,
                )?,
            };
            let output = tensor_from_f32(
                backend,
                input.descriptor().shape(),
                &values,
                output_dtype,
                output_device,
                context,
            )?;
            context.cancellation.check()?;
            let next_generation = self.next_generation()?;
            if let Some(next_weight) = prepared
                .as_mut()
                .and_then(|parameters| parameters.next_weight.take())
            {
                self.weight = Some(next_weight);
                self.prefetched = None;
            }
            self.normalization_state = staged_state;
            self.generation = next_generation;
            return Ok(output);
        }
        if let NativeModuleSpec::LayerNorm {
            normalized_shape,
            epsilon,
            elementwise_affine: false,
            ..
        } = &self.spec
        {
            let values = layer_norm_with_context_exact_native(
                backend,
                &input_values,
                &input_shape,
                normalized_shape,
                None,
                None,
                *epsilon,
                DeviceId::CPU,
                context,
            )?;
            return tensor_from_f32(
                backend,
                input.descriptor().shape(),
                &values,
                output_dtype,
                output_device,
                context,
            );
        }
        if let NativeModuleSpec::GroupNorm {
            groups,
            epsilon,
            affine: false,
            ..
        } = &self.spec
        {
            let values = group_norm_with_context_exact_native(
                backend,
                &input_values,
                &input_shape,
                *groups,
                None,
                None,
                *epsilon,
                DeviceId::CPU,
                context,
            )?;
            return tensor_from_f32(
                backend,
                input.descriptor().shape(),
                &values,
                output_dtype,
                output_device,
                context,
            );
        }
        if let NativeModuleSpec::InstanceNorm2d {
            features,
            epsilon,
            affine: false,
        } = &self.spec
        {
            if input_shape.len() != 4 || input_shape.get(1) != Some(features) {
                return Err(NativeOpsError::Invalid(
                    "instance-normalization input rank or channels do not match the module",
                ));
            }
            let values = instance_norm_2d_with_context_exact_native(
                backend,
                &input_values,
                &input_shape,
                None,
                None,
                *epsilon,
                DeviceId::CPU,
                context,
            )?;
            let output = tensor_from_f32(
                backend,
                input.descriptor().shape(),
                &values,
                output_dtype,
                output_device,
                context,
            )?;
            return self.complete_forward(output, context);
        }
        let mut prepared = self.prepare_parameters(
            backend,
            Some(input),
            Some(output_dtype),
            Some(output_device),
            Some(output_dtype),
            Some(DType::F32),
            false,
            context,
        )?;
        let weight_values = tensor_to_f32(backend, &prepared.weight, context)?;
        let bias_values = prepared
            .bias
            .as_ref()
            .map(|bias| tensor_to_f32(backend, bias, context))
            .transpose()?;
        let result = match &self.spec {
            NativeModuleSpec::Container => {
                return Err(NativeOpsError::Invalid(
                    "a base module container has no forward implementation",
                ));
            }
            NativeModuleSpec::Linear {
                input_features,
                output_features,
                ..
            } => linear_with_context_exact_native(
                &input_values,
                &input_shape,
                &weight_values,
                &[*output_features, *input_features],
                bias_values.as_deref(),
                DeviceId::CPU,
                context,
            )?,
            NativeModuleSpec::Convolution {
                input_channels,
                output_channels,
                kernel_shape,
                geometry,
                ..
            } => {
                let mut weight_shape = if geometry.transposed() {
                    vec![*input_channels, output_channels / geometry.groups()]
                } else {
                    vec![*output_channels, input_channels / geometry.groups()]
                };
                weight_shape.extend_from_slice(kernel_shape);
                let projected_weight = if autopad == ConvolutionAutopad::CausalZero {
                    if geometry.spatial_dimensions() != 3 || geometry.transposed() {
                        return Err(NativeOpsError::Invalid(
                            "causal-zero autopad requires ordinary three-dimensional convolution",
                        ));
                    }
                    let input_depth = *input_shape.get(2).ok_or(NativeOpsError::Invalid(
                        "causal convolution input must have NCDHW rank",
                    ))?;
                    Some(crop_causal_weight(
                        backend,
                        &weight_values,
                        &weight_shape,
                        input_depth,
                        context,
                    )?)
                } else {
                    None
                };
                let (kernel_values, kernel_shape) = match &projected_weight {
                    Some((values, shape)) => (&values[..], shape.as_slice()),
                    None => (&weight_values[..], weight_shape.as_slice()),
                };
                convolution_with_context_exact_native(
                    &input_values,
                    &input_shape,
                    kernel_values,
                    kernel_shape,
                    bias_values.as_deref(),
                    geometry,
                    DeviceId::CPU,
                    context,
                )?
            }
            NativeModuleSpec::LayerNorm {
                normalized_shape,
                epsilon,
                elementwise_affine,
                ..
            } => TensorValues {
                values: layer_norm_with_context_exact_native(
                    backend,
                    &input_values,
                    &input_shape,
                    normalized_shape,
                    elementwise_affine.then_some(&weight_values[..]),
                    bias_values.as_deref(),
                    *epsilon,
                    DeviceId::CPU,
                    context,
                )?,
                shape: input_shape,
            },
            NativeModuleSpec::GroupNorm {
                groups,
                epsilon,
                affine,
                ..
            } => TensorValues {
                values: group_norm_with_context_exact_native(
                    backend,
                    &input_values,
                    &input_shape,
                    *groups,
                    affine.then_some(&weight_values[..]),
                    bias_values.as_deref(),
                    *epsilon,
                    DeviceId::CPU,
                    context,
                )?,
                shape: input_shape,
            },
            NativeModuleSpec::InstanceNorm2d {
                epsilon, affine, ..
            } => TensorValues {
                values: instance_norm_2d_with_context_exact_native(
                    backend,
                    &input_values,
                    &input_shape,
                    affine.then_some(&weight_values[..]),
                    bias_values.as_deref(),
                    *epsilon,
                    DeviceId::CPU,
                    context,
                )?,
                shape: input_shape,
            },
            NativeModuleSpec::PRelu { .. } => TensorValues {
                values: prelu_with_context_exact_native(
                    backend,
                    &input_values,
                    &input_shape,
                    &weight_values,
                    DeviceId::CPU,
                    context,
                )?,
                shape: input_shape,
            },
            NativeModuleSpec::AdaptiveAveragePool2d { .. }
            | NativeModuleSpec::AveragePool1d { .. }
            | NativeModuleSpec::AveragePool2d { .. }
            | NativeModuleSpec::AveragePool3d { .. }
            | NativeModuleSpec::BatchNorm { .. }
            | NativeModuleSpec::Buffer
            | NativeModuleSpec::Dropout { .. }
            | NativeModuleSpec::Elu { .. }
            | NativeModuleSpec::Embedding { .. }
            | NativeModuleSpec::Gelu { .. }
            | NativeModuleSpec::HuberLoss { .. }
            | NativeModuleSpec::Identity
            | NativeModuleSpec::L1Loss { .. }
            | NativeModuleSpec::LeakyRelu { .. }
            | NativeModuleSpec::MaxPool2d { .. }
            | NativeModuleSpec::ModuleDict
            | NativeModuleSpec::ModuleList
            | NativeModuleSpec::MseLoss { .. }
            | NativeModuleSpec::MultiheadAttention { .. }
            | NativeModuleSpec::PixelShuffle { .. }
            | NativeModuleSpec::PixelUnshuffle { .. }
            | NativeModuleSpec::ReplicationPad2d { .. }
            | NativeModuleSpec::Relu
            | NativeModuleSpec::Relu6
            | NativeModuleSpec::Sequential
            | NativeModuleSpec::Sigmoid
            | NativeModuleSpec::Silu
            | NativeModuleSpec::SmoothL1Loss { .. }
            | NativeModuleSpec::Softmax { .. }
            | NativeModuleSpec::Tanh
            | NativeModuleSpec::Upsample { .. }
            | NativeModuleSpec::ZeroPad2d { .. } => {
                return Err(NativeOpsError::Invalid(
                    "parameter-free module reached parameter dispatch",
                ));
            }
        };
        let output = tensor_from_f32(
            backend,
            &shape_to_u64(&result.shape)?,
            &result.values,
            output_dtype,
            output_device,
            context,
        )?;
        context.cancellation.check()?;
        let next_generation = self.next_generation()?;
        if let Some(next_weight) = prepared.next_weight.take() {
            self.weight = Some(next_weight);
            self.prefetched = None;
        }
        self.generation = next_generation;
        Ok(output)
    }

    fn complete_forward(
        &mut self,
        output: Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeOpsError> {
        context.cancellation.check()?;
        self.generation = self.next_generation()?;
        Ok(output)
    }

    pub fn forward_loss_with_context(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        target: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeOpsError> {
        context.cancellation.check()?;
        let (parameter, reduction, loss_kind) = match self.spec {
            NativeModuleSpec::SmoothL1Loss { beta, reduction } => (beta, reduction, 0_u8),
            NativeModuleSpec::HuberLoss { delta, reduction } => (delta, reduction, 1_u8),
            NativeModuleSpec::L1Loss { reduction } => (0.0, reduction, 2_u8),
            NativeModuleSpec::MseLoss { reduction } => (0.0, reduction, 3_u8),
            _ => {
                return Err(NativeOpsError::Invalid(
                    "only L1, MSE, smooth-L1, and Huber modules accept a target tensor",
                ));
            }
        };
        if input.descriptor() != target.descriptor() {
            return Err(NativeOpsError::Invalid(
                "loss input and target descriptors must match",
            ));
        }
        let input_values = tensor_to_f32(backend, input, context)?;
        let target_values = tensor_to_f32(backend, target, context)?;
        let values = match loss_kind {
            1 => huber_loss_with_context_exact_native(
                backend,
                &input_values,
                &target_values,
                parameter,
                reduction,
                input.descriptor().device(),
                context,
            )?,
            2 => l1_loss_with_context_exact_native(
                backend,
                &input_values,
                &target_values,
                reduction,
                input.descriptor().device(),
                context,
            )?,
            3 => mse_loss_with_context_exact_native(
                &input_values,
                &target_values,
                reduction,
                input.descriptor().device(),
                context,
            )?,
            _ => smooth_l1_loss_with_context_exact_native(
                backend,
                &input_values,
                &target_values,
                parameter,
                reduction,
                input.descriptor().device(),
                context,
            )?,
        };
        let shape = if reduction == LossReduction::None {
            input.descriptor().shape().to_vec()
        } else {
            Vec::new()
        };
        let output = tensor_from_f32(
            backend,
            &shape,
            &values,
            input.descriptor().dtype(),
            input.descriptor().device(),
            context,
        )?;
        self.complete_forward(output, context)
    }

    fn prepare_parameters(
        &self,
        backend: &CpuBackend,
        input: Option<&Tensor>,
        dtype: Option<DType>,
        device: Option<DeviceId>,
        bias_dtype: Option<DType>,
        compute_dtype: Option<DType>,
        want_requant: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<PrefetchedParameters, NativeOpsError> {
        context.cancellation.check()?;
        self.validate_parameter_presence()?;
        let weight = self
            .weight
            .as_ref()
            .ok_or(NativeOpsError::ParametersNotLoaded)?;
        let inferred_dtype = input.map(|input| input.descriptor().dtype());
        let inferred_device = input.map(|input| input.descriptor().device());
        let target_dtype = compute_dtype
            .or(dtype)
            .or(inferred_dtype)
            .unwrap_or(DType::F32);
        let target_device = device.or(inferred_device).unwrap_or(DeviceId::CPU);
        let target_bias_dtype = bias_dtype
            .or(dtype)
            .or(inferred_dtype)
            .unwrap_or(target_dtype);
        if !matches!(weight, NativeWeight::SpectralNorm(_))
            && let Some(prefetched) = &self.prefetched
            && prefetched.dtype == target_dtype
            && prefetched.bias_dtype == target_bias_dtype
            && prefetched.device == target_device
            && (!want_requant || prefetched.requantized_weight.is_some())
        {
            return Ok(prefetched.clone());
        }
        let requantized_weight = match weight {
            NativeWeight::Quantized(weight) if want_requant => Some(weight.clone()),
            _ => None,
        };
        let (weight, next_weight) = match weight {
            NativeWeight::Dense(weight) => (
                cast_to(
                    backend,
                    weight,
                    target_dtype,
                    target_device,
                    false,
                    self.manual_cast,
                    context,
                )?,
                None,
            ),
            NativeWeight::Quantized(weight) => {
                let materialization = weight.materialize(backend, context)?;
                (
                    tensor_from_f32(
                        backend,
                        &[
                            u64::try_from(weight.rows())
                                .map_err(|_| NativeOpsError::Invalid("quantized row overflow"))?,
                            u64::try_from(weight.columns()).map_err(|_| {
                                NativeOpsError::Invalid("quantized column overflow")
                            })?,
                        ],
                        materialization.values(),
                        target_dtype,
                        target_device,
                        context,
                    )?,
                    None,
                )
            }
            NativeWeight::WeightNorm(parametrization) => {
                let weight = materialize_weight_norm(backend, parametrization, context)?;
                (
                    cast_to(
                        backend,
                        &weight,
                        target_dtype,
                        target_device,
                        false,
                        self.manual_cast,
                        context,
                    )?,
                    None,
                )
            }
            NativeWeight::SpectralNorm(parametrization) => {
                let (weight, next_parametrization) =
                    materialize_spectral_norm(backend, parametrization, context)?;
                (
                    cast_to(
                        backend,
                        &weight,
                        target_dtype,
                        target_device,
                        false,
                        self.manual_cast,
                        context,
                    )?,
                    Some(NativeWeight::SpectralNorm(next_parametrization)),
                )
            }
        };
        let bias = self
            .bias
            .as_ref()
            .map(|bias| {
                cast_to(
                    backend,
                    bias,
                    target_bias_dtype,
                    target_device,
                    false,
                    self.manual_cast,
                    context,
                )
            })
            .transpose()?;
        Ok(PrefetchedParameters {
            weight,
            bias,
            requantized_weight,
            dtype: target_dtype,
            bias_dtype: target_bias_dtype,
            device: target_device,
            next_weight,
        })
    }

    fn validate_parameter_shapes(
        &self,
        weight_shape: &[u64],
        bias: Option<&Tensor>,
    ) -> Result<(), NativeOpsError> {
        let expected_weight = match &self.spec {
            NativeModuleSpec::Container => {
                return Err(NativeOpsError::Invalid(
                    "a base module container has no weight or bias parameters",
                ));
            }
            NativeModuleSpec::Linear {
                input_features,
                output_features,
                bias: expects_bias,
            } => {
                validate_bias_shape(*expects_bias, *output_features, bias)?;
                vec![*output_features, *input_features]
            }
            NativeModuleSpec::Convolution {
                input_channels,
                output_channels,
                kernel_shape,
                bias: expects_bias,
                geometry,
            } => {
                validate_bias_shape(*expects_bias, *output_channels, bias)?;
                let mut shape = if geometry.transposed() {
                    vec![*input_channels, output_channels / geometry.groups()]
                } else {
                    vec![*output_channels, input_channels / geometry.groups()]
                };
                shape.extend_from_slice(kernel_shape);
                shape
            }
            NativeModuleSpec::LayerNorm {
                normalized_shape,
                elementwise_affine,
                bias: expects_bias,
                ..
            } => {
                validate_parameter_shape(*expects_bias, normalized_shape, bias)?;
                if !elementwise_affine {
                    return Err(NativeOpsError::Invalid(
                        "non-affine layer normalization does not load a weight",
                    ));
                }
                normalized_shape.clone()
            }
            NativeModuleSpec::GroupNorm {
                channels, affine, ..
            } => {
                validate_parameter_shape(*affine, &[*channels], bias)?;
                if !affine {
                    return Err(NativeOpsError::Invalid(
                        "non-affine group normalization does not load a weight",
                    ));
                }
                vec![*channels]
            }
            NativeModuleSpec::BatchNorm {
                features, affine, ..
            } => {
                validate_parameter_shape(*affine, &[*features], bias)?;
                if !affine {
                    return Err(NativeOpsError::Invalid(
                        "non-affine batch normalization does not load a weight",
                    ));
                }
                vec![*features]
            }
            NativeModuleSpec::Embedding {
                embeddings,
                dimensions,
                ..
            } => {
                validate_parameter_shape(false, &[*embeddings, *dimensions], bias)?;
                vec![*embeddings, *dimensions]
            }
            NativeModuleSpec::InstanceNorm2d {
                features, affine, ..
            } => {
                validate_parameter_shape(*affine, &[*features], bias)?;
                if !affine {
                    return Err(NativeOpsError::Invalid(
                        "non-affine instance normalization does not load a weight",
                    ));
                }
                vec![*features]
            }
            NativeModuleSpec::PRelu { num_parameters } => {
                validate_parameter_shape(false, &[*num_parameters], bias)?;
                vec![*num_parameters]
            }
            NativeModuleSpec::AdaptiveAveragePool2d { .. }
            | NativeModuleSpec::AveragePool1d { .. }
            | NativeModuleSpec::AveragePool2d { .. }
            | NativeModuleSpec::AveragePool3d { .. }
            | NativeModuleSpec::Buffer
            | NativeModuleSpec::Dropout { .. }
            | NativeModuleSpec::Elu { .. }
            | NativeModuleSpec::Gelu { .. }
            | NativeModuleSpec::HuberLoss { .. }
            | NativeModuleSpec::Identity
            | NativeModuleSpec::L1Loss { .. }
            | NativeModuleSpec::LeakyRelu { .. }
            | NativeModuleSpec::MaxPool2d { .. }
            | NativeModuleSpec::ModuleDict
            | NativeModuleSpec::ModuleList
            | NativeModuleSpec::MseLoss { .. }
            | NativeModuleSpec::MultiheadAttention { .. }
            | NativeModuleSpec::PixelShuffle { .. }
            | NativeModuleSpec::PixelUnshuffle { .. }
            | NativeModuleSpec::ReplicationPad2d { .. }
            | NativeModuleSpec::Relu
            | NativeModuleSpec::Relu6
            | NativeModuleSpec::Sequential
            | NativeModuleSpec::Sigmoid
            | NativeModuleSpec::Silu
            | NativeModuleSpec::SmoothL1Loss { .. }
            | NativeModuleSpec::Softmax { .. }
            | NativeModuleSpec::Tanh
            | NativeModuleSpec::Upsample { .. }
            | NativeModuleSpec::ZeroPad2d { .. } => {
                return Err(NativeOpsError::Invalid(
                    "parameter-free modules do not load weight or bias parameters",
                ));
            }
        };
        if weight_shape
            != expected_weight
                .iter()
                .map(|value| u64::try_from(*value))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| NativeOpsError::Invalid("weight shape overflow"))?
        {
            return Err(NativeOpsError::Invalid(
                "loaded weight shape does not match module configuration",
            ));
        }
        Ok(())
    }

    fn validate_parameter_presence(&self) -> Result<(), NativeOpsError> {
        let (expects_weight, expects_bias) = match &self.spec {
            NativeModuleSpec::Container => (false, false),
            NativeModuleSpec::Linear { bias, .. } | NativeModuleSpec::Convolution { bias, .. } => {
                (true, *bias)
            }
            NativeModuleSpec::LayerNorm {
                elementwise_affine,
                bias,
                ..
            } => (*elementwise_affine, *bias),
            NativeModuleSpec::GroupNorm { affine, .. } => (*affine, *affine),
            NativeModuleSpec::BatchNorm { affine, .. }
            | NativeModuleSpec::InstanceNorm2d { affine, .. } => (*affine, *affine),
            NativeModuleSpec::Embedding { .. } => (true, false),
            NativeModuleSpec::PRelu { .. } => (true, false),
            NativeModuleSpec::AdaptiveAveragePool2d { .. }
            | NativeModuleSpec::AveragePool1d { .. }
            | NativeModuleSpec::AveragePool2d { .. }
            | NativeModuleSpec::AveragePool3d { .. }
            | NativeModuleSpec::Buffer
            | NativeModuleSpec::Dropout { .. }
            | NativeModuleSpec::Elu { .. }
            | NativeModuleSpec::Gelu { .. }
            | NativeModuleSpec::HuberLoss { .. }
            | NativeModuleSpec::Identity
            | NativeModuleSpec::L1Loss { .. }
            | NativeModuleSpec::LeakyRelu { .. }
            | NativeModuleSpec::MaxPool2d { .. }
            | NativeModuleSpec::ModuleDict
            | NativeModuleSpec::ModuleList
            | NativeModuleSpec::MseLoss { .. }
            | NativeModuleSpec::MultiheadAttention { .. }
            | NativeModuleSpec::PixelShuffle { .. }
            | NativeModuleSpec::PixelUnshuffle { .. }
            | NativeModuleSpec::ReplicationPad2d { .. }
            | NativeModuleSpec::Relu
            | NativeModuleSpec::Relu6
            | NativeModuleSpec::Sequential
            | NativeModuleSpec::Sigmoid
            | NativeModuleSpec::Silu
            | NativeModuleSpec::SmoothL1Loss { .. }
            | NativeModuleSpec::Softmax { .. }
            | NativeModuleSpec::Tanh
            | NativeModuleSpec::Upsample { .. }
            | NativeModuleSpec::ZeroPad2d { .. } => (false, false),
        };
        if expects_weight != self.weight.is_some() || expects_bias != self.bias.is_some() {
            Err(NativeOpsError::ParametersNotLoaded)
        } else {
            Ok(())
        }
    }

    fn parameter_shape(&self, name: &str) -> Result<Vec<usize>, NativeOpsError> {
        match (&self.spec, name) {
            (
                NativeModuleSpec::Linear {
                    input_features,
                    output_features,
                    ..
                },
                "weight",
            ) => Ok(vec![*output_features, *input_features]),
            (
                NativeModuleSpec::Linear {
                    output_features,
                    bias: true,
                    ..
                },
                "bias",
            ) => Ok(vec![*output_features]),
            (
                NativeModuleSpec::Convolution {
                    input_channels,
                    output_channels,
                    kernel_shape,
                    geometry,
                    ..
                },
                "weight",
            ) => {
                let mut shape = if geometry.transposed() {
                    vec![*input_channels, output_channels / geometry.groups()]
                } else {
                    vec![*output_channels, input_channels / geometry.groups()]
                };
                shape.extend_from_slice(kernel_shape);
                Ok(shape)
            }
            (
                NativeModuleSpec::Convolution {
                    output_channels,
                    bias: true,
                    ..
                },
                "bias",
            ) => Ok(vec![*output_channels]),
            (
                NativeModuleSpec::LayerNorm {
                    normalized_shape,
                    elementwise_affine: true,
                    ..
                },
                "weight",
            ) => Ok(normalized_shape.clone()),
            (
                NativeModuleSpec::LayerNorm {
                    normalized_shape,
                    bias: true,
                    ..
                },
                "bias",
            ) => Ok(normalized_shape.clone()),
            (
                NativeModuleSpec::GroupNorm {
                    channels,
                    affine: true,
                    ..
                },
                "weight" | "bias",
            ) => Ok(vec![*channels]),
            (
                NativeModuleSpec::BatchNorm {
                    features,
                    affine: true,
                    ..
                }
                | NativeModuleSpec::InstanceNorm2d {
                    features,
                    affine: true,
                    ..
                },
                "weight" | "bias",
            ) => Ok(vec![*features]),
            (
                NativeModuleSpec::Embedding {
                    embeddings,
                    dimensions,
                    ..
                },
                "weight",
            ) => Ok(vec![*embeddings, *dimensions]),
            (NativeModuleSpec::PRelu { num_parameters }, "weight") => Ok(vec![*num_parameters]),
            (_, "weight" | "bias") => Err(NativeOpsError::Invalid(
                "module configuration does not contain the requested parameter",
            )),
            _ => Err(NativeOpsError::Invalid(
                "only weight and bias parameters can be zero initialized",
            )),
        }
    }

    fn next_generation(&self) -> Result<u64, NativeOpsError> {
        self.generation
            .checked_add(1)
            .ok_or(NativeOpsError::GenerationOverflow)
    }
}

struct NativeModuleStateDigest<'a> {
    hasher: Sha256,
    cancellation: &'a CancellationToken,
}

impl<'a> NativeModuleStateDigest<'a> {
    fn new(cancellation: &'a CancellationToken) -> Self {
        let mut digest = Self {
            hasher: Sha256::new(),
            cancellation,
        };
        digest.bytes(b"zed.native-module.semantic-state.v1");
        digest
    }

    fn finish(self) -> impl std::fmt::LowerHex {
        self.hasher.finalize()
    }

    fn module(&mut self, module: &NativeModule) -> Result<(), NativeOpsError> {
        self.cancellation.check()?;
        self.bytes(b"module");
        self.bytes(module.layer_name.as_bytes());
        self.spec(&module.spec)?;
        self.boolean(module.manual_cast);
        self.weight_norm_registration(module.weight_norm_dimension)?;
        self.spectral_config(module.spectral_norm_config.as_ref())?;
        self.weight(module.weight.as_ref())?;
        self.tensor_option(module.bias.as_ref())?;
        self.tensor_option(module.registered_buffer.as_ref())?;
        match module.normalization_state.as_ref() {
            Some(state) => {
                self.bytes(b"normalization_state");
                self.f32_slice(&state.running_mean)?;
                self.f32_slice(&state.running_variance)?;
            }
            None => self.bytes(b"no_normalization_state"),
        }
        self.usize(module.children.len())?;
        for child in &module.children {
            self.module(child)?;
        }
        Ok(())
    }

    fn spec(&mut self, spec: &NativeModuleSpec) -> Result<(), NativeOpsError> {
        match spec {
            NativeModuleSpec::Container => self.bytes(b"container"),
            NativeModuleSpec::AveragePool1d {
                kernel_size,
                stride,
            } => {
                self.bytes(b"average_pool_1d");
                self.usize(*kernel_size)?;
                self.usize(*stride)?;
            }
            NativeModuleSpec::AveragePool2d {
                kernel_size,
                stride,
            } => {
                self.bytes(b"average_pool_2d");
                self.usize_slice(kernel_size)?;
                self.usize_slice(stride)?;
            }
            NativeModuleSpec::AdaptiveAveragePool2d { output_size } => {
                self.bytes(b"adaptive_average_pool_2d");
                self.usize_slice(output_size)?;
            }
            NativeModuleSpec::AveragePool3d {
                kernel_size,
                stride,
            } => {
                self.bytes(b"average_pool_3d");
                self.usize_slice(kernel_size)?;
                self.usize_slice(stride)?;
            }
            NativeModuleSpec::BatchNorm {
                dimensions,
                features,
                epsilon,
                momentum,
                affine,
                track_running_stats,
                training,
            } => {
                self.bytes(b"batch_norm");
                self.usize(*dimensions)?;
                self.usize(*features)?;
                self.f32(*epsilon);
                self.f32(*momentum);
                self.boolean(*affine);
                self.boolean(*track_running_stats);
                self.boolean(*training);
            }
            NativeModuleSpec::Buffer => self.bytes(b"buffer"),
            NativeModuleSpec::Dropout {
                probability,
                training,
            } => {
                self.bytes(b"dropout");
                self.f32(*probability);
                self.boolean(*training);
            }
            NativeModuleSpec::Elu { alpha } => {
                self.bytes(b"elu");
                self.f32(*alpha);
            }
            NativeModuleSpec::Linear {
                input_features,
                output_features,
                bias,
            } => {
                self.bytes(b"linear");
                self.usize(*input_features)?;
                self.usize(*output_features)?;
                self.boolean(*bias);
            }
            NativeModuleSpec::Convolution {
                input_channels,
                output_channels,
                kernel_shape,
                bias,
                geometry,
            } => {
                self.bytes(b"convolution");
                self.usize(*input_channels)?;
                self.usize(*output_channels)?;
                self.usize_slice(kernel_shape)?;
                self.boolean(*bias);
                self.usize(geometry.spatial_dimensions())?;
                self.usize_slice(geometry.stride())?;
                self.usize_slice(geometry.padding())?;
                self.usize_slice(geometry.dilation())?;
                self.usize(geometry.groups())?;
                self.boolean(geometry.transposed());
                self.usize_slice(geometry.output_padding())?;
                self.convolution_padding_mode(geometry.padding_mode());
            }
            NativeModuleSpec::Gelu { approximation } => {
                self.bytes(b"gelu");
                self.gelu_approximation(*approximation);
            }
            NativeModuleSpec::LayerNorm {
                normalized_shape,
                epsilon,
                elementwise_affine,
                bias,
            } => {
                self.bytes(b"layer_norm");
                self.usize_slice(normalized_shape)?;
                self.f32(*epsilon);
                self.boolean(*elementwise_affine);
                self.boolean(*bias);
            }
            NativeModuleSpec::GroupNorm {
                groups,
                channels,
                epsilon,
                affine,
            } => {
                self.bytes(b"group_norm");
                self.usize(*groups)?;
                self.usize(*channels)?;
                self.f32(*epsilon);
                self.boolean(*affine);
            }
            NativeModuleSpec::Embedding {
                embeddings,
                dimensions,
                options,
            } => {
                self.bytes(b"embedding");
                self.usize(*embeddings)?;
                self.usize(*dimensions)?;
                match options.padding_index {
                    Some(value) => {
                        self.bytes(b"padding_index");
                        self.bytes(&value.to_le_bytes());
                    }
                    None => self.bytes(b"no_padding_index"),
                }
                match options.max_norm {
                    Some(value) => {
                        self.bytes(b"max_norm");
                        self.f32(value);
                    }
                    None => self.bytes(b"no_max_norm"),
                }
                self.f32(options.norm_type);
                self.boolean(options.scale_gradient_by_frequency);
                self.boolean(options.sparse);
            }
            NativeModuleSpec::HuberLoss { delta, reduction } => {
                self.bytes(b"huber_loss");
                self.f32(*delta);
                self.loss_reduction(*reduction);
            }
            NativeModuleSpec::Identity => self.bytes(b"identity"),
            NativeModuleSpec::L1Loss { reduction } => {
                self.bytes(b"l1_loss");
                self.loss_reduction(*reduction);
            }
            NativeModuleSpec::InstanceNorm2d {
                features,
                epsilon,
                affine,
            } => {
                self.bytes(b"instance_norm_2d");
                self.usize(*features)?;
                self.f32(*epsilon);
                self.boolean(*affine);
            }
            NativeModuleSpec::LeakyRelu { negative_slope } => {
                self.bytes(b"leaky_relu");
                self.f32(*negative_slope);
            }
            NativeModuleSpec::MultiheadAttention {
                embed_dimension,
                heads,
            } => {
                self.bytes(b"multihead_attention");
                self.usize(*embed_dimension)?;
                self.usize(*heads)?;
            }
            NativeModuleSpec::MaxPool2d {
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            } => {
                self.bytes(b"max_pool_2d");
                self.usize_slice(kernel_size)?;
                self.usize_slice(stride)?;
                self.usize_slice(padding)?;
                self.usize_slice(dilation)?;
                self.boolean(*ceil_mode);
            }
            NativeModuleSpec::MseLoss { reduction } => {
                self.bytes(b"mse_loss");
                self.loss_reduction(*reduction);
            }
            NativeModuleSpec::ModuleDict => self.bytes(b"module_dict"),
            NativeModuleSpec::ModuleList => self.bytes(b"module_list"),
            NativeModuleSpec::PRelu { num_parameters } => {
                self.bytes(b"prelu");
                self.usize(*num_parameters)?;
            }
            NativeModuleSpec::ReplicationPad2d { padding } => {
                self.bytes(b"replication_pad_2d");
                self.usize_slice(padding)?;
            }
            NativeModuleSpec::PixelShuffle { factor } => {
                self.bytes(b"pixel_shuffle");
                self.bytes(&factor.to_le_bytes());
            }
            NativeModuleSpec::PixelUnshuffle { factor } => {
                self.bytes(b"pixel_unshuffle");
                self.bytes(&factor.to_le_bytes());
            }
            NativeModuleSpec::Relu => self.bytes(b"relu"),
            NativeModuleSpec::Relu6 => self.bytes(b"relu_6"),
            NativeModuleSpec::Sequential => self.bytes(b"sequential"),
            NativeModuleSpec::Silu => self.bytes(b"silu"),
            NativeModuleSpec::Sigmoid => self.bytes(b"sigmoid"),
            NativeModuleSpec::SmoothL1Loss { beta, reduction } => {
                self.bytes(b"smooth_l1_loss");
                self.f32(*beta);
                self.loss_reduction(*reduction);
            }
            NativeModuleSpec::Softmax { dimension } => {
                self.bytes(b"softmax");
                let dimension = i64::try_from(*dimension)
                    .map_err(|_| NativeOpsError::Invalid("softmax dimension overflow"))?;
                self.bytes(&dimension.to_le_bytes());
            }
            NativeModuleSpec::Tanh => self.bytes(b"tanh"),
            NativeModuleSpec::Upsample {
                scale_factor,
                mode,
                align_corners,
            } => {
                self.bytes(b"upsample");
                for value in scale_factor {
                    self.bytes(&value.to_bits().to_le_bytes());
                }
                self.upsample_mode(*mode);
                match align_corners {
                    Some(value) => {
                        self.bytes(b"align_corners");
                        self.boolean(*value);
                    }
                    None => self.bytes(b"no_align_corners"),
                }
            }
            NativeModuleSpec::ZeroPad2d { padding } => {
                self.bytes(b"zero_pad_2d");
                self.usize_slice(padding)?;
            }
        }
        Ok(())
    }

    fn weight(&mut self, weight: Option<&NativeWeight>) -> Result<(), NativeOpsError> {
        match weight {
            Some(NativeWeight::Dense(tensor)) => {
                self.bytes(b"dense_weight");
                self.tensor(tensor)?;
            }
            Some(NativeWeight::Quantized(matrix)) => {
                self.bytes(b"quantized_weight");
                self.usize(matrix.rows())?;
                self.usize(matrix.columns())?;
                self.dtype(matrix.original_dtype());
                self.quantization_kind(matrix.kind());
                self.bytes(matrix.content_identity().as_bytes());
            }
            Some(NativeWeight::WeightNorm(weight_norm)) => {
                self.bytes(b"weight_norm");
                self.tensor(&weight_norm.magnitude)?;
                self.tensor(&weight_norm.direction)?;
                self.optional_usize(weight_norm.dimension)?;
            }
            Some(NativeWeight::SpectralNorm(spectral_norm)) => {
                self.bytes(b"spectral_norm");
                self.tensor(&spectral_norm.original)?;
                self.spectral_config(Some(&spectral_norm.config))?;
                self.optional_f32_slice(spectral_norm.left.as_deref())?;
                self.optional_f32_slice(spectral_norm.right.as_deref())?;
            }
            None => self.bytes(b"no_weight"),
        }
        Ok(())
    }

    fn tensor_option(&mut self, tensor: Option<&Tensor>) -> Result<(), NativeOpsError> {
        match tensor {
            Some(tensor) => {
                self.bytes(b"tensor");
                self.tensor(tensor)?;
            }
            None => self.bytes(b"no_tensor"),
        }
        Ok(())
    }

    fn tensor(&mut self, tensor: &Tensor) -> Result<(), NativeOpsError> {
        let descriptor = tensor.descriptor();
        self.bytes(b"tensor_state");
        self.u64_slice(descriptor.shape());
        self.i64_slice(descriptor.strides());
        self.bytes(&descriptor.offset_elements().to_le_bytes());
        self.dtype(descriptor.dtype());
        self.layout(descriptor.layout());
        self.device(descriptor.device());
        self.bytes(&descriptor.stream().get().to_le_bytes());
        let element_count = descriptor.element_count()?;
        self.bytes(&element_count.to_le_bytes());
        for index in 0..element_count {
            if index.is_multiple_of(1_024) {
                self.cancellation.check()?;
            }
            self.bytes(tensor.linear_element_bytes(index)?);
        }
        self.cancellation.check()?;
        Ok(())
    }

    fn weight_norm_registration(
        &mut self,
        registration: Option<Option<usize>>,
    ) -> Result<(), NativeOpsError> {
        match registration {
            Some(dimension) => {
                self.bytes(b"weight_norm_registration");
                self.optional_usize(dimension)?;
            }
            None => self.bytes(b"no_weight_norm_registration"),
        }
        Ok(())
    }

    fn spectral_config(
        &mut self,
        config: Option<&NativeSpectralNormConfig>,
    ) -> Result<(), NativeOpsError> {
        match config {
            Some(config) => {
                self.bytes(b"spectral_norm_config");
                self.usize(config.dimension)?;
                self.usize(config.power_iterations)?;
                self.f32(config.epsilon);
            }
            None => self.bytes(b"no_spectral_norm_config"),
        }
        Ok(())
    }

    fn optional_usize(&mut self, value: Option<usize>) -> Result<(), NativeOpsError> {
        match value {
            Some(value) => {
                self.bytes(b"some_usize");
                self.usize(value)?;
            }
            None => self.bytes(b"no_usize"),
        }
        Ok(())
    }

    fn optional_f32_slice(&mut self, values: Option<&[f32]>) -> Result<(), NativeOpsError> {
        match values {
            Some(values) => {
                self.bytes(b"some_f32_slice");
                self.f32_slice(values)?;
            }
            None => self.bytes(b"no_f32_slice"),
        }
        Ok(())
    }

    fn f32_slice(&mut self, values: &[f32]) -> Result<(), NativeOpsError> {
        self.usize(values.len())?;
        for value in values {
            self.f32(*value);
        }
        Ok(())
    }

    fn usize_slice(&mut self, values: &[usize]) -> Result<(), NativeOpsError> {
        self.usize(values.len())?;
        for value in values {
            self.usize(*value)?;
        }
        Ok(())
    }

    fn u64_slice(&mut self, values: &[u64]) {
        self.bytes(
            &u64::try_from(values.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for value in values {
            self.bytes(&value.to_le_bytes());
        }
    }

    fn i64_slice(&mut self, values: &[i64]) {
        self.bytes(
            &u64::try_from(values.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for value in values {
            self.bytes(&value.to_le_bytes());
        }
    }

    fn usize(&mut self, value: usize) -> Result<(), NativeOpsError> {
        let value = u64::try_from(value)
            .map_err(|_| NativeOpsError::Invalid("module state dimension overflow"))?;
        self.bytes(&value.to_le_bytes());
        Ok(())
    }

    fn f32(&mut self, value: f32) {
        self.bytes(&value.to_bits().to_le_bytes());
    }

    fn boolean(&mut self, value: bool) {
        self.bytes(&[u8::from(value)]);
    }

    fn dtype(&mut self, dtype: DType) {
        let tag = match dtype {
            DType::F64 => 0,
            DType::F32 => 1,
            DType::F16 => 2,
            DType::Bf16 => 3,
            DType::I64 => 4,
            DType::I32 => 5,
            DType::I16 => 6,
            DType::I8 => 7,
            DType::U64 => 8,
            DType::U32 => 9,
            DType::U16 => 10,
            DType::U8 => 11,
            DType::Bool => 12,
            DType::Complex64 => 13,
            DType::Complex128 => 14,
            DType::Float8E4m3Fn => 15,
            DType::Float8E5m2 => 16,
            DType::Float8E4m3Fnuz => 17,
            DType::Float8E5m2Fnuz => 18,
            DType::Float8E8m0Fnu => 19,
        };
        self.bytes(&[tag]);
    }

    fn layout(&mut self, layout: Layout) {
        let tag = match layout {
            Layout::Contiguous => 0,
            Layout::ChannelsLast => 1,
            Layout::ChannelsLast3d => 2,
            Layout::Strided => 3,
        };
        self.bytes(&[tag]);
    }

    fn device(&mut self, device: DeviceId) {
        let tag = match device.kind() {
            DeviceKind::Cpu => 0,
            DeviceKind::Cuda => 1,
            DeviceKind::Rocm => 2,
            DeviceKind::Metal => 3,
            DeviceKind::DirectMl => 4,
            DeviceKind::Xpu => 5,
            DeviceKind::Npu => 6,
            DeviceKind::Mlu => 7,
            DeviceKind::CoreX => 8,
        };
        self.bytes(&[tag]);
        self.bytes(&device.ordinal().to_le_bytes());
    }

    fn quantization_kind(&mut self, kind: QuantizationKind) {
        let tag = match kind {
            QuantizationKind::Int8Tensorwise => 0,
            QuantizationKind::MxFp8 => 1,
            QuantizationKind::NvFp4 => 2,
            QuantizationKind::MixedPerLayerV1 => 3,
        };
        self.bytes(&[tag]);
    }

    fn convolution_padding_mode(&mut self, mode: ConvolutionPaddingMode) {
        let tag = match mode {
            ConvolutionPaddingMode::Zeros => 0,
            ConvolutionPaddingMode::Reflect => 1,
            ConvolutionPaddingMode::Replicate => 2,
            ConvolutionPaddingMode::Circular => 3,
        };
        self.bytes(&[tag]);
    }

    fn gelu_approximation(&mut self, approximation: GeluApproximation) {
        let tag = match approximation {
            GeluApproximation::None => 0,
            GeluApproximation::Tanh => 1,
        };
        self.bytes(&[tag]);
    }

    fn loss_reduction(&mut self, reduction: LossReduction) {
        let tag = match reduction {
            LossReduction::None => 0,
            LossReduction::Sum => 1,
            LossReduction::Mean => 2,
        };
        self.bytes(&[tag]);
    }

    fn upsample_mode(&mut self, mode: UpsampleMode) {
        let tag = match mode {
            UpsampleMode::Nearest => 0,
            UpsampleMode::Bilinear => 1,
        };
        self.bytes(&[tag]);
    }

    fn bytes(&mut self, value: &[u8]) {
        self.hasher
            .update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
        self.hasher.update(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, StreamId, TensorDescriptor};

    struct TestBackend {
        backend: CpuBackend,
        workspace: sim_tensor::BackendWorkspaceAuthority,
    }

    fn backend() -> Result<TestBackend, NativeOpsError> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        Ok(TestBackend { backend, workspace })
    }

    fn tensor(
        test_backend: &TestBackend,
        shape: &[u64],
        values: &[f32],
    ) -> Result<Tensor, NativeOpsError> {
        let cancellation = CancellationToken::default();
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let context = test_backend.backend.execution_context(
            StreamId::DEFAULT,
            test_backend.workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        Ok(test_backend
            .backend
            .upload_f32(descriptor, values, &context)?
            .0)
    }

    fn dense(
        test_backend: &TestBackend,
        weight: &[f32],
        bias: &[f32],
        manual_cast: bool,
    ) -> Result<NativeModule, NativeOpsError> {
        let mut module = NativeModule::linear("projection", 2, 2, true, manual_cast)?;
        module.load_dense_parameters(
            tensor(test_backend, &[2, 2], weight)?,
            Some(tensor(test_backend, &[2], bias)?),
        )?;
        Ok(module)
    }

    fn state_digest(module: &NativeModule) -> Result<String, NativeOpsError> {
        module.semantic_state_digest(&CancellationToken::default())
    }

    #[test]
    fn recursive_execution_requirements_use_the_canonical_capability_owner()
    -> Result<(), NativeOpsError> {
        let module = NativeModule::sequential(
            "encoder",
            vec![
                NativeModule::linear("projection", 2, 2, true, false)?,
                NativeModule::layer_norm("normalization", vec![2], 1e-5, true, true, false)?,
            ],
        )?;
        let requirements = module.execution_requirements(DType::F32);
        assert!(requirements.iter().any(|support| {
            support
                == OperationSupport::linear_algebra_input(
                    LinearAlgebraOperation::BatchMatrixMultiply,
                    DType::F32,
                    Layout::Contiguous,
                )
        }));
        assert!(requirements.iter().any(|support| {
            support
                == OperationSupport::reduction_input(
                    ReductionOperation::Variance,
                    DType::F32,
                    Layout::Contiguous,
                )
        }));
        requirements.require_matrix_support(&CpuBackend::capability_matrix())?;
        let operations = pick_operations_exact_native(
            DType::F32,
            None,
            Some(DeviceId::CPU),
            false,
            false,
            None,
            &CancellationToken::default(),
        )?;
        operations.require_matrix_support(
            std::slice::from_ref(&module),
            &CpuBackend::capability_matrix(),
        )?;

        let under_provisioned = BackendCapabilityMatrix::new(
            DeviceId::CPU,
            vec![OperationSupport::allocation(DType::F32, Layout::Contiguous)],
            Vec::new(),
        )?;
        assert!(matches!(
            requirements.require_matrix_support(&under_provisioned),
            Err(NativeOpsError::Workspace(
                TensorError::UnsupportedCapability { .. }
            ))
        ));
        assert!(matches!(
            module
                .execution_requirements(DType::F16)
                .require_matrix_support(&CpuBackend::capability_matrix()),
            Err(NativeOpsError::Workspace(
                TensorError::UnsupportedCapability { .. }
            ))
        ));
        Ok(())
    }

    #[test]
    fn equivalent_dense_state_is_stable_and_weight_bias_or_config_changes_are_distinct()
    -> Result<(), NativeOpsError> {
        let test_backend = backend()?;
        let first = dense(&test_backend, &[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5], false)?;
        let equivalent = dense(&test_backend, &[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5], false)?;
        let changed_weight = dense(&test_backend, &[1.0, 2.0, 3.0, 5.0], &[0.5, -0.5], false)?;
        let changed_bias = dense(&test_backend, &[1.0, 2.0, 3.0, 4.0], &[0.5, 0.5], false)?;
        let changed_config = dense(&test_backend, &[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5], true)?;

        let digest = state_digest(&first)?;
        assert_eq!(digest, state_digest(&equivalent)?);
        assert_ne!(digest, state_digest(&changed_weight)?);
        assert_ne!(digest, state_digest(&changed_bias)?);
        assert_ne!(digest, state_digest(&changed_config)?);
        Ok(())
    }

    #[test]
    fn resident_storage_bytes_count_shared_tensor_backing_once() -> Result<(), NativeOpsError> {
        let test_backend = backend()?;
        let shared = tensor(&test_backend, &[2], &[1.0, 2.0])?;
        let independent = tensor(&test_backend, &[2], &[1.0, 2.0])?;
        let aliased = NativeModule::sequential(
            "root",
            vec![
                NativeModule::buffer("first", shared.clone())?,
                NativeModule::buffer("second", shared.clone())?,
            ],
        )?;
        let distinct = NativeModule::sequential(
            "root",
            vec![
                NativeModule::buffer("first", shared.clone())?,
                NativeModule::buffer("second", independent)?,
            ],
        )?;
        assert_eq!(aliased.resident_storage_bytes()?, shared.storage_byte_len());
        assert_eq!(aliased.resident_tensor_allocations().len(), 1);
        assert_eq!(
            distinct.resident_storage_bytes()?,
            shared
                .storage_byte_len()
                .checked_mul(2)
                .ok_or(NativeOpsError::ResidentBytesOverflow)?
        );
        assert_eq!(distinct.resident_tensor_allocations().len(), 2);
        assert_eq!(state_digest(&aliased)?, state_digest(&distinct)?);
        Ok(())
    }

    #[test]
    fn quantized_raw_storage_and_ordered_child_state_are_distinct() -> Result<(), NativeOpsError> {
        let test_backend = backend()?;
        let cancellation = CancellationToken::default();
        let first_weight = crate::quantize_matrix(
            QuantizationKind::Int8Tensorwise,
            DType::F32,
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
            &cancellation,
        )?;
        let second_weight = crate::quantize_matrix(
            QuantizationKind::Int8Tensorwise,
            DType::F32,
            &[1.0, 2.0, 3.0, 5.0],
            2,
            2,
            &cancellation,
        )?;
        let mut first = NativeModule::linear("quantized", 2, 2, false, false)?;
        first.load_quantized_linear_parameters(first_weight, None)?;
        let mut second = NativeModule::linear("quantized", 2, 2, false, false)?;
        second.load_quantized_linear_parameters(second_weight, None)?;
        assert_ne!(state_digest(&first)?, state_digest(&second)?);

        let encoded_first = crate::quantize_matrix(
            QuantizationKind::Int8Tensorwise,
            DType::F32,
            &[1.0, 0.100_00],
            1,
            2,
            &cancellation,
        )?;
        let encoded_second = crate::quantize_matrix(
            QuantizationKind::Int8Tensorwise,
            DType::F32,
            &[1.0, 0.100_01],
            1,
            2,
            &cancellation,
        )?;
        assert_eq!(
            encoded_first.raw_storage_digest(&cancellation)?,
            encoded_second.raw_storage_digest(&cancellation)?
        );
        assert_ne!(
            encoded_first.content_identity(),
            encoded_second.content_identity()
        );
        let mut source_first = NativeModule::linear("quantized", 2, 1, false, false)?;
        source_first.load_quantized_linear_parameters(encoded_first, None)?;
        let mut source_second = NativeModule::linear("quantized", 2, 1, false, false)?;
        source_second.load_quantized_linear_parameters(encoded_second, None)?;
        assert_ne!(state_digest(&source_first)?, state_digest(&source_second)?);

        let relu = NativeModule::sequential("root", vec![NativeModule::relu("activation")?])?;
        let silu = NativeModule::sequential("root", vec![NativeModule::silu("activation")?])?;
        assert_ne!(state_digest(&relu)?, state_digest(&silu)?);

        let ordered = NativeModule::sequential(
            "root",
            vec![NativeModule::relu("first")?, NativeModule::silu("second")?],
        )?;
        let reversed = NativeModule::sequential(
            "root",
            vec![NativeModule::silu("second")?, NativeModule::relu("first")?],
        )?;
        assert_ne!(state_digest(&ordered)?, state_digest(&reversed)?);

        let buffer = NativeModule::buffer("buffer", tensor(&test_backend, &[2], &[1.0, 2.0])?)?;
        let changed_buffer =
            NativeModule::buffer("buffer", tensor(&test_backend, &[2], &[1.0, 3.0])?)?;
        assert_ne!(state_digest(&buffer)?, state_digest(&changed_buffer)?);
        Ok(())
    }

    #[test]
    fn prefetch_cache_and_generation_do_not_change_semantic_state() -> Result<(), NativeOpsError> {
        let test_backend = backend()?;
        let module = dense(&test_backend, &[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5], true)?;
        let digest = state_digest(&module)?;
        let cancellation = CancellationToken::default();
        let context = test_backend.backend.execution_context(
            StreamId::DEFAULT,
            test_backend.workspace.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let mut modules = [module];
        cast_modules_with_vbar_with_context_exact_native(
            &mut modules,
            &test_backend.backend,
            DType::F32,
            DeviceId::CPU,
            DType::F32,
            false,
            &context,
        )?;
        assert_eq!(digest, state_digest(&modules[0])?);
        Ok(())
    }

    #[test]
    fn execution_state_recurses_and_does_not_treat_empty_structure_as_loaded()
    -> Result<(), NativeOpsError> {
        let test_backend = backend()?;
        let empty = NativeModule::container("empty")?;
        let unloaded = NativeModule::linear("unloaded", 2, 2, false, false)?;
        assert!(!empty.has_execution_state());
        assert!(!unloaded.has_execution_state());

        let loaded = dense(&test_backend, &[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5], false)?;
        assert!(loaded.has_execution_state());
        let parent =
            NativeModule::sequential("parent", vec![NativeModule::relu("activation")?, loaded])?;
        assert!(parent.has_execution_state());

        let buffer = NativeModule::buffer("buffer", tensor(&test_backend, &[1], &[1.0])?)?;
        assert!(buffer.has_execution_state());
        let batch_norm = NativeModule::batch_norm_1d("norm", 2, 1.0e-5, 0.1, false, true, false)?;
        assert!(batch_norm.has_execution_state());
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightCastLease {
    generation: u64,
    offloadable: bool,
    completed: bool,
}

impl WeightCastLease {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn offloadable(&self) -> bool {
        self.offloadable
    }

    pub const fn is_completed(&self) -> bool {
        self.completed
    }
}

#[derive(Clone, Debug)]
pub struct CastedParameters {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub requantized_weight: Option<QuantizedMatrix>,
    lease: WeightCastLease,
}

impl CastedParameters {
    pub fn lease(&self) -> &WeightCastLease {
        &self.lease
    }

    pub fn finish(&mut self, cancellation: &CancellationToken) -> Result<(), NativeOpsError> {
        cancellation.check()?;
        if self.lease.completed {
            return Err(NativeOpsError::LeaseAlreadyCompleted {
                generation: self.lease.generation,
            });
        }
        self.lease.completed = true;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrefetchReceipt {
    pub module_generations: Vec<(String, u64)>,
}

#[allow(clippy::too_many_arguments)]
pub fn cast_modules_with_vbar_with_context_exact_native(
    modules: &mut [NativeModule],
    backend: &CpuBackend,
    dtype: DType,
    device: DeviceId,
    bias_dtype: DType,
    _non_blocking: bool,
    context: &ExecutionContext<'_>,
) -> Result<PrefetchReceipt, NativeOpsError> {
    cast_modules_with_vbar_impl(modules, backend, dtype, device, bias_dtype, context)
}

fn cast_modules_with_vbar_impl(
    modules: &mut [NativeModule],
    backend: &CpuBackend,
    dtype: DType,
    device: DeviceId,
    bias_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<PrefetchReceipt, NativeOpsError> {
    parameter_materialization_requirements(dtype).admit_backend_target(
        backend,
        device,
        dtype,
        Layout::Contiguous,
        context.stream,
        context,
    )?;
    if bias_dtype != dtype {
        parameter_materialization_requirements(bias_dtype).admit_backend_target(
            backend,
            device,
            bias_dtype,
            Layout::Contiguous,
            context.stream,
            context,
        )?;
    }
    let mut staged = temporary_vec(backend, context, modules.len())?;
    for module in modules.iter() {
        context.cancellation.check()?;
        staged.try_push(Some(module.prepare_parameters(
            backend,
            None,
            Some(dtype),
            Some(device),
            Some(bias_dtype),
            None,
            false,
            context,
        )?))?;
    }
    let mut next_generations = temporary_vec(backend, context, modules.len())?;
    for module in modules.iter() {
        next_generations.try_push(module.next_generation()?)?;
    }
    context.cancellation.check()?;
    let mut module_generations = Vec::new();
    module_generations
        .try_reserve_exact(modules.len())
        .map_err(|_| NativeOpsError::Invalid("prefetch receipt is too large"))?;
    for (index, module) in modules.iter_mut().enumerate() {
        let mut prepared = staged
            .get_mut(index)
            .and_then(Option::take)
            .ok_or(NativeOpsError::Invalid("prefetch staging entry is missing"))?;
        let next_generation = *next_generations.get(index).ok_or(NativeOpsError::Invalid(
            "prefetch generation entry is missing",
        ))?;
        if let Some(next_weight) = prepared.next_weight.take() {
            module.weight = Some(next_weight);
        }
        module.generation = next_generation;
        module.prefetched = Some(prepared);
        module_generations.push((module.layer_name.clone(), module.generation));
    }
    Ok(PrefetchReceipt { module_generations })
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeOperationSet {
    quantization: Option<QuantizationMetadataV1>,
    weight_dtype: DType,
    compute_dtype: DType,
    manual_cast: bool,
    mixed_precision: bool,
    full_precision_matrix_multiply: bool,
    disabled: BTreeSet<QuantizationKind>,
}

impl NativeOperationSet {
    pub fn execution_requirements(&self, modules: &[NativeModule]) -> NativeExecutionRequirements {
        let mut requirements = NativeExecutionRequirements::new();
        for module in modules {
            requirements.extend(module.execution_requirements(self.compute_dtype).iter());
        }
        requirements
    }

    #[cfg(test)]
    fn require_matrix_support(
        &self,
        modules: &[NativeModule],
        capabilities: &BackendCapabilityMatrix,
    ) -> Result<(), NativeOpsError> {
        self.execution_requirements(modules)
            .require_matrix_support(capabilities)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn admit_backend_target(
        &self,
        modules: &[NativeModule],
        backend: &dyn TensorBackend,
        device: DeviceId,
        layout: Layout,
        stream: StreamId,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeOpsError> {
        self.execution_requirements(modules).admit_backend_target(
            backend,
            device,
            self.compute_dtype,
            layout,
            stream,
            context,
        )
    }

    pub fn average_pool_2d(
        &self,
        layer_name: impl Into<String>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
    ) -> Result<NativeModule, NativeOpsError> {
        NativeModule::average_pool_2d(layer_name, kernel_size, stride)
    }

    pub fn linear(
        &self,
        layer_name: impl Into<String>,
        input_features: usize,
        output_features: usize,
        bias: bool,
    ) -> Result<NativeModule, NativeOpsError> {
        NativeModule::linear(
            layer_name,
            input_features,
            output_features,
            bias,
            self.manual_cast,
        )
    }

    pub fn quantization_for_layer(&self, layer_name: &str) -> Option<&LayerQuantizationV1> {
        self.quantization
            .as_ref()
            .and_then(|metadata| metadata.layers.get(layer_name))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        layer_name: impl Into<String>,
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        padding_mode: ConvolutionPaddingMode,
    ) -> Result<NativeModule, NativeOpsError> {
        let geometry = ConvolutionGeometry::new_with_padding_mode(
            1,
            vec![stride],
            vec![padding],
            vec![dilation],
            groups,
            false,
            vec![0],
            padding_mode,
        )?;
        NativeModule::convolution(
            layer_name,
            input_channels,
            output_channels,
            vec![kernel_size],
            bias,
            geometry,
            self.manual_cast,
        )
    }

    pub fn layer_norm(
        &self,
        layer_name: impl Into<String>,
        normalized_shape: Vec<usize>,
        epsilon: f32,
        elementwise_affine: bool,
        bias: bool,
    ) -> Result<NativeModule, NativeOpsError> {
        NativeModule::layer_norm(
            layer_name,
            normalized_shape,
            epsilon,
            elementwise_affine,
            bias,
            self.manual_cast,
        )
    }

    pub fn group_norm(
        &self,
        layer_name: impl Into<String>,
        groups: usize,
        channels: usize,
        epsilon: f32,
        affine: bool,
    ) -> Result<NativeModule, NativeOpsError> {
        NativeModule::group_norm(
            layer_name,
            groups,
            channels,
            epsilon,
            affine,
            self.manual_cast,
        )
    }

    pub fn prelu(
        &self,
        layer_name: impl Into<String>,
        num_parameters: usize,
    ) -> Result<NativeModule, NativeOpsError> {
        NativeModule::prelu(layer_name, num_parameters, self.manual_cast)
    }

    pub const fn compute_dtype(&self) -> DType {
        self.compute_dtype
    }

    pub const fn weight_dtype(&self) -> DType {
        self.weight_dtype
    }

    pub const fn manual_cast(&self) -> bool {
        self.manual_cast
    }

    pub const fn is_mixed_precision(&self) -> bool {
        self.mixed_precision
    }

    pub const fn full_precision_matrix_multiply(&self) -> bool {
        self.full_precision_matrix_multiply
    }

    pub fn is_native_quantization_enabled(&self, kind: QuantizationKind) -> bool {
        !self.disabled.contains(&kind)
    }
}

pub fn mixed_precision_ops_exact_native(
    quant_config: BTreeMap<String, LayerQuantizationV1>,
    compute_dtype: DType,
    full_precision_matrix_multiply: bool,
    disabled: BTreeSet<QuantizationKind>,
) -> Result<NativeOperationSet, NativeOpsError> {
    if !matches!(compute_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(NativeOpsError::Invalid(
            "mixed-precision compute dtype must be float16, bfloat16, or float32",
        ));
    }
    let quantization = if quant_config.is_empty() {
        None
    } else {
        let metadata = QuantizationMetadataV1 {
            version: 1,
            layers: quant_config,
        };
        metadata.validate()?;
        Some(metadata)
    };
    Ok(NativeOperationSet {
        quantization,
        weight_dtype: compute_dtype,
        compute_dtype,
        manual_cast: true,
        mixed_precision: true,
        full_precision_matrix_multiply,
        disabled,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn pick_operations_exact_native(
    weight_dtype: DType,
    compute_dtype: Option<DType>,
    load_device: Option<DeviceId>,
    _disable_fast_fp8: bool,
    _fp8_optimizations: bool,
    quant_config: Option<BTreeMap<String, LayerQuantizationV1>>,
    cancellation: &CancellationToken,
) -> Result<NativeOperationSet, NativeOpsError> {
    cancellation.check()?;
    let load_device = load_device.unwrap_or(DeviceId::CPU);
    if load_device.kind() != DeviceKind::Cpu || load_device.ordinal() != 0 {
        return Err(NativeOpsError::UnsupportedDevice {
            device: load_device,
        });
    }
    let operations = if let Some(quant_config) = quant_config
        && !quant_config.is_empty()
    {
        mixed_precision_ops_exact_native(
            quant_config,
            compute_dtype.unwrap_or(DType::Bf16),
            false,
            BTreeSet::new(),
        )?
    } else {
        let compute_dtype = compute_dtype.unwrap_or(weight_dtype);
        NativeOperationSet {
            quantization: None,
            weight_dtype,
            compute_dtype,
            manual_cast: weight_dtype != compute_dtype,
            mixed_precision: false,
            full_precision_matrix_multiply: false,
            disabled: BTreeSet::new(),
        }
    };
    cancellation.check()?;
    Ok(operations)
}

pub fn disable_weight_init_linear_exact_native(
    layer_name: impl Into<String>,
    input_features: usize,
    output_features: usize,
    bias: bool,
) -> Result<NativeModule, NativeOpsError> {
    NativeModule::linear(layer_name, input_features, output_features, bias, false)
}

pub fn manual_cast_linear_exact_native(
    layer_name: impl Into<String>,
    input_features: usize,
    output_features: usize,
    bias: bool,
) -> Result<NativeModule, NativeOpsError> {
    NativeModule::linear(layer_name, input_features, output_features, bias, true)
}

#[allow(clippy::too_many_arguments)]
pub fn disable_weight_init_conv1d_exact_native(
    layer_name: impl Into<String>,
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
    padding_mode: ConvolutionPaddingMode,
) -> Result<NativeModule, NativeOpsError> {
    let geometry = ConvolutionGeometry::new_with_padding_mode(
        1,
        vec![stride],
        vec![padding],
        vec![dilation],
        groups,
        false,
        vec![0],
        padding_mode,
    )?;
    NativeModule::convolution(
        layer_name,
        input_channels,
        output_channels,
        vec![kernel_size],
        bias,
        geometry,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn disable_weight_init_convolution_exact_native(
    layer_name: impl Into<String>,
    input_channels: usize,
    output_channels: usize,
    kernel_shape: Vec<usize>,
    bias: bool,
    geometry: ConvolutionGeometry,
) -> Result<NativeModule, NativeOpsError> {
    NativeModule::convolution(
        layer_name,
        input_channels,
        output_channels,
        kernel_shape,
        bias,
        geometry,
        false,
    )
}

pub fn manual_cast_layer_norm_exact_native(
    layer_name: impl Into<String>,
    normalized_shape: Vec<usize>,
    epsilon: f32,
    elementwise_affine: bool,
    bias: bool,
) -> Result<NativeModule, NativeOpsError> {
    NativeModule::layer_norm(
        layer_name,
        normalized_shape,
        epsilon,
        elementwise_affine,
        bias,
        true,
    )
}

pub fn disable_weight_init_layer_norm_exact_native(
    layer_name: impl Into<String>,
    normalized_shape: Vec<usize>,
    epsilon: f32,
    elementwise_affine: bool,
    bias: bool,
) -> Result<NativeModule, NativeOpsError> {
    NativeModule::layer_norm(
        layer_name,
        normalized_shape,
        epsilon,
        elementwise_affine,
        bias,
        false,
    )
}

pub fn disable_weight_init_group_norm_exact_native(
    layer_name: impl Into<String>,
    groups: usize,
    channels: usize,
    epsilon: f32,
    affine: bool,
) -> Result<NativeModule, NativeOpsError> {
    NativeModule::group_norm(layer_name, groups, channels, epsilon, affine, false)
}

pub fn module_init_exact_native(
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::container("torch.nn.Module")?;
    cancellation.check()?;
    Ok(module)
}

pub fn module_exact_native(
    layer_name: impl Into<String>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::container(layer_name)?;
    cancellation.check()?;
    Ok(module)
}

pub fn average_pool_1d_module_exact_native(
    layer_name: impl Into<String>,
    kernel_size: usize,
    stride: Option<usize>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module =
        NativeModule::average_pool_1d(layer_name, kernel_size, stride.unwrap_or(kernel_size))?;
    cancellation.check()?;
    Ok(module)
}

pub fn average_pool_2d_module_exact_native(
    layer_name: impl Into<String>,
    kernel_size: [usize; 2],
    stride: Option<[usize; 2]>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module =
        NativeModule::average_pool_2d(layer_name, kernel_size, stride.unwrap_or(kernel_size))?;
    cancellation.check()?;
    Ok(module)
}

pub fn adaptive_average_pool_2d_module_exact_native(
    layer_name: impl Into<String>,
    output_size: [usize; 2],
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::adaptive_average_pool_2d(layer_name, output_size)?;
    cancellation.check()?;
    Ok(module)
}

pub fn average_pool_3d_module_exact_native(
    layer_name: impl Into<String>,
    kernel_size: [usize; 3],
    stride: Option<[usize; 3]>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module =
        NativeModule::average_pool_3d(layer_name, kernel_size, stride.unwrap_or(kernel_size))?;
    cancellation.check()?;
    Ok(module)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_1d_module_exact_native(
    layer_name: impl Into<String>,
    features: usize,
    epsilon: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::batch_norm_1d(
        layer_name,
        features,
        epsilon,
        momentum,
        affine,
        track_running_stats,
        false,
    )?;
    cancellation.check()?;
    Ok(module)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_2d_module_exact_native(
    layer_name: impl Into<String>,
    features: usize,
    epsilon: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::batch_norm_2d(
        layer_name,
        features,
        epsilon,
        momentum,
        affine,
        track_running_stats,
        false,
    )?;
    cancellation.check()?;
    Ok(module)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_2d_module_exact_native(
    layer_name: impl Into<String>,
    input_channels: usize,
    output_channels: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    bias: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::conv_2d(
        layer_name,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        false,
    )?;
    cancellation.check()?;
    Ok(module)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_3d_module_exact_native(
    layer_name: impl Into<String>,
    input_channels: usize,
    output_channels: usize,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    bias: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::conv_3d(
        layer_name,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        false,
    )?;
    cancellation.check()?;
    Ok(module)
}

pub fn embedding_module_exact_native(
    layer_name: impl Into<String>,
    embeddings: usize,
    dimensions: usize,
    options: EmbeddingOptions,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::embedding(layer_name, embeddings, dimensions, options, false)?;
    cancellation.check()?;
    Ok(module)
}

pub fn dropout_module_exact_native(
    layer_name: impl Into<String>,
    probability: f32,
    inplace: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    if inplace {
        return Err(NativeOpsError::Invalid(
            "native modules reject in-place dropout because graph-visible alias mutation is forbidden",
        ));
    }
    let module = NativeModule::dropout(layer_name, probability)?;
    cancellation.check()?;
    Ok(module)
}

pub fn elu_module_exact_native(
    layer_name: impl Into<String>,
    alpha: f32,
    inplace: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    if inplace {
        return Err(NativeOpsError::Invalid(
            "native modules reject in-place ELU because graph-visible alias mutation is forbidden",
        ));
    }
    let module = NativeModule::elu(layer_name, alpha)?;
    cancellation.check()?;
    Ok(module)
}

pub fn gelu_module_exact_native(
    layer_name: impl Into<String>,
    approximation: GeluApproximation,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::gelu(layer_name, approximation)?;
    cancellation.check()?;
    Ok(module)
}

pub fn huber_loss_module_exact_native(
    layer_name: impl Into<String>,
    delta: f32,
    reduction: LossReduction,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::huber_loss(layer_name, delta, reduction)?;
    cancellation.check()?;
    Ok(module)
}

pub fn l1_loss_module_exact_native(
    layer_name: impl Into<String>,
    reduction: LossReduction,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::l1_loss(layer_name, reduction)?;
    cancellation.check()?;
    Ok(module)
}

pub fn identity_module_exact_native(
    layer_name: impl Into<String>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::identity(layer_name)?;
    cancellation.check()?;
    Ok(module)
}

pub fn mse_loss_module_exact_native(
    layer_name: impl Into<String>,
    reduction: LossReduction,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::mse_loss(layer_name, reduction)?;
    cancellation.check()?;
    Ok(module)
}

pub fn instance_norm_2d_module_exact_native(
    layer_name: impl Into<String>,
    features: usize,
    epsilon: f32,
    affine: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::instance_norm_2d(layer_name, features, epsilon, affine, false)?;
    cancellation.check()?;
    Ok(module)
}

pub fn leaky_relu_module_exact_native(
    layer_name: impl Into<String>,
    negative_slope: f32,
    inplace: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    if inplace {
        return Err(NativeOpsError::Invalid(
            "native modules reject in-place leaky-ReLU because graph-visible alias mutation is forbidden",
        ));
    }
    let module = NativeModule::leaky_relu(layer_name, negative_slope)?;
    cancellation.check()?;
    Ok(module)
}

pub fn linear_module_exact_native(
    layer_name: impl Into<String>,
    input_features: usize,
    output_features: usize,
    bias: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::linear(layer_name, input_features, output_features, bias, false)?;
    cancellation.check()?;
    Ok(module)
}

pub fn multihead_attention_module_exact_native(
    layer_name: impl Into<String>,
    embed_dimension: usize,
    heads: usize,
    bias: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module =
        NativeModule::multihead_attention(layer_name, embed_dimension, heads, bias, false)?;
    cancellation.check()?;
    Ok(module)
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_2d_module_exact_native(
    layer_name: impl Into<String>,
    kernel_size: [usize; 2],
    stride: Option<[usize; 2]>,
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::max_pool_2d(
        layer_name,
        kernel_size,
        stride.unwrap_or(kernel_size),
        padding,
        dilation,
        ceil_mode,
    )?;
    cancellation.check()?;
    Ok(module)
}

pub fn module_dict_exact_native(
    layer_name: impl Into<String>,
    children: Vec<NativeModule>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::module_dict(layer_name, children)?;
    cancellation.check()?;
    Ok(module)
}

pub fn module_list_exact_native(
    layer_name: impl Into<String>,
    children: Vec<NativeModule>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::module_list(layer_name, children)?;
    cancellation.check()?;
    Ok(module)
}

pub fn pixel_shuffle_module_exact_native(
    layer_name: impl Into<String>,
    factor: u64,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::pixel_shuffle(layer_name, factor)?;
    cancellation.check()?;
    Ok(module)
}

pub fn pixel_unshuffle_module_exact_native(
    layer_name: impl Into<String>,
    factor: u64,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::pixel_unshuffle(layer_name, factor)?;
    cancellation.check()?;
    Ok(module)
}

pub fn replication_pad_2d_module_exact_native(
    layer_name: impl Into<String>,
    padding: [usize; 4],
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::replication_pad_2d(layer_name, padding)?;
    cancellation.check()?;
    Ok(module)
}

pub fn buffer_module_exact_native(
    layer_name: impl Into<String>,
    tensor: Tensor,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::buffer(layer_name, tensor)?;
    cancellation.check()?;
    Ok(module)
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_module_exact_native(
    layer_name: impl Into<String>,
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let geometry = ConvolutionGeometry::new_with_padding_mode(
        1,
        vec![stride],
        vec![padding],
        vec![dilation],
        groups,
        false,
        vec![0],
        ConvolutionPaddingMode::Zeros,
    )?;
    let module = NativeModule::convolution(
        layer_name,
        input_channels,
        output_channels,
        vec![kernel_size],
        bias,
        geometry,
        false,
    )?;
    cancellation.check()?;
    Ok(module)
}

pub fn group_norm_module_exact_native(
    layer_name: impl Into<String>,
    groups: usize,
    channels: usize,
    epsilon: f32,
    affine: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::group_norm(layer_name, groups, channels, epsilon, affine, false)?;
    cancellation.check()?;
    Ok(module)
}

pub fn layer_norm_module_exact_native(
    layer_name: impl Into<String>,
    normalized_shape: Vec<usize>,
    epsilon: f32,
    elementwise_affine: bool,
    bias: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::layer_norm(
        layer_name,
        normalized_shape,
        epsilon,
        elementwise_affine,
        bias,
        false,
    )?;
    cancellation.check()?;
    Ok(module)
}

pub fn prelu_module_exact_native(
    layer_name: impl Into<String>,
    num_parameters: usize,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::prelu(layer_name, num_parameters, false)?;
    cancellation.check()?;
    Ok(module)
}

pub fn sequential_module_exact_native(
    layer_name: impl Into<String>,
    children: Vec<NativeModule>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::sequential(layer_name, children)?;
    cancellation.check()?;
    Ok(module)
}

pub fn relu_module_exact_native(
    layer_name: impl Into<String>,
    inplace: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    if inplace {
        return Err(NativeOpsError::Invalid(
            "native modules reject in-place ReLU because graph-visible alias mutation is forbidden",
        ));
    }
    let module = NativeModule::relu(layer_name)?;
    cancellation.check()?;
    Ok(module)
}

pub fn relu_6_module_exact_native(
    layer_name: impl Into<String>,
    inplace: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    if inplace {
        return Err(NativeOpsError::Invalid(
            "native modules reject in-place ReLU6 because graph-visible alias mutation is forbidden",
        ));
    }
    let module = NativeModule::relu_6(layer_name)?;
    cancellation.check()?;
    Ok(module)
}

pub fn silu_module_exact_native(
    layer_name: impl Into<String>,
    inplace: bool,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    if inplace {
        return Err(NativeOpsError::Invalid(
            "native modules reject in-place SiLU because graph-visible alias mutation is forbidden",
        ));
    }
    let module = NativeModule::silu(layer_name)?;
    cancellation.check()?;
    Ok(module)
}

pub fn sigmoid_module_exact_native(
    layer_name: impl Into<String>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::sigmoid(layer_name)?;
    cancellation.check()?;
    Ok(module)
}

pub fn smooth_l1_loss_module_exact_native(
    layer_name: impl Into<String>,
    beta: f32,
    reduction: LossReduction,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::smooth_l1_loss(layer_name, beta, reduction)?;
    cancellation.check()?;
    Ok(module)
}

pub fn softmax_module_exact_native(
    layer_name: impl Into<String>,
    dimension: isize,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::softmax(layer_name, dimension)?;
    cancellation.check()?;
    Ok(module)
}

pub fn tanh_module_exact_native(
    layer_name: impl Into<String>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::tanh(layer_name)?;
    cancellation.check()?;
    Ok(module)
}

pub fn upsample_module_exact_native(
    layer_name: impl Into<String>,
    scale_factor: [f64; 2],
    mode: UpsampleMode,
    align_corners: Option<bool>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::upsample(layer_name, scale_factor, mode, align_corners)?;
    cancellation.check()?;
    Ok(module)
}

pub fn zero_pad_2d_module_exact_native(
    layer_name: impl Into<String>,
    padding: [usize; 4],
    cancellation: &CancellationToken,
) -> Result<NativeModule, NativeOpsError> {
    cancellation.check()?;
    let module = NativeModule::zero_pad_2d(layer_name, padding)?;
    cancellation.check()?;
    Ok(module)
}

pub fn remove_parametrizations_with_context_exact_native(
    backend: &CpuBackend,
    module: &mut NativeModule,
    name: &str,
    leave_parametrized: bool,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeOpsError> {
    context.cancellation.check()?;
    module.remove_parametrizations_with_context_exact_native(
        backend,
        name,
        leave_parametrized,
        context,
    )
}

pub fn weight_norm_exact_native<'a>(
    module: &'a mut NativeModule,
    name: &str,
    dimension: Option<usize>,
    cancellation: &CancellationToken,
) -> Result<&'a mut NativeModule, NativeOpsError> {
    cancellation.check()?;
    module.register_weight_norm_exact_native(name, dimension, cancellation)?;
    Ok(module)
}

pub fn spectral_norm_exact_native<'a>(
    module: &'a mut NativeModule,
    name: &str,
    power_iterations: usize,
    epsilon: f32,
    dimension: Option<usize>,
    cancellation: &CancellationToken,
) -> Result<&'a mut NativeModule, NativeOpsError> {
    cancellation.check()?;
    module.register_spectral_norm_exact_native(
        name,
        power_iterations,
        epsilon,
        dimension,
        cancellation,
    )?;
    Ok(module)
}

fn checked_layer_name(layer_name: String) -> Result<String, NativeOpsError> {
    if layer_name.is_empty() || layer_name.len() > 1024 || layer_name.contains('\0') {
        Err(NativeOpsError::Invalid(
            "layer names must contain 1..=1024 non-NUL bytes",
        ))
    } else {
        Ok(layer_name)
    }
}

fn validate_weight_norm_magnitude_shape(
    magnitude_shape: &[u64],
    direction_shape: &[u64],
    dimension: Option<usize>,
) -> Result<(), NativeOpsError> {
    if direction_shape.is_empty() || direction_shape.contains(&0) {
        return Err(NativeOpsError::Invalid(
            "weight-normalization direction shape must be nonempty",
        ));
    }
    match dimension {
        Some(dimension) => {
            let Some(&dimension_size) = direction_shape.get(dimension) else {
                return Err(NativeOpsError::Invalid(
                    "weight-normalization dimension is out of range",
                ));
            };
            if magnitude_shape.len() != direction_shape.len()
                || magnitude_shape.iter().enumerate().any(|(index, size)| {
                    *size
                        != if index == dimension {
                            dimension_size
                        } else {
                            1
                        }
                })
            {
                return Err(NativeOpsError::Invalid(
                    "weight-normalization magnitude shape is invalid",
                ));
            }
        }
        None if !magnitude_shape.is_empty() && magnitude_shape != [1] => {
            return Err(NativeOpsError::Invalid(
                "dimensionless weight-normalization magnitude must be scalar",
            ));
        }
        None => {}
    }
    Ok(())
}

fn materialize_weight_norm(
    backend: &CpuBackend,
    parametrization: &NativeWeightNorm,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeOpsError> {
    context.cancellation.check()?;
    let direction_shape = shape_to_usize(parametrization.direction.descriptor().shape())?;
    validate_weight_norm_magnitude_shape(
        parametrization.magnitude.descriptor().shape(),
        parametrization.direction.descriptor().shape(),
        parametrization.dimension,
    )?;
    let direction = tensor_to_f32(backend, &parametrization.direction, context)?;
    let magnitude = tensor_to_f32(backend, &parametrization.magnitude, context)?;
    let group_count = parametrization
        .dimension
        .map_or(1, |dimension| direction_shape[dimension]);
    if magnitude.len() != group_count {
        return Err(NativeOpsError::Invalid(
            "weight-normalization magnitude value count is invalid",
        ));
    }
    let mut squared_norms = temporary_filled(backend, context, group_count, 0.0_f64)?;
    let inner_stride = match parametrization.dimension {
        Some(dimension) => checked_product(
            direction_shape
                .get(dimension + 1..)
                .ok_or(NativeOpsError::Invalid(
                    "weight-normalization dimension is out of range",
                ))?,
            "weight-normalization stride overflow",
        )?,
        None => 1,
    };
    for (index, value) in direction.iter().copied().enumerate() {
        if index % 1_024 == 0 {
            context.cancellation.check()?;
        }
        let group = parametrization
            .dimension
            .map_or(0, |_| (index / inner_stride) % group_count);
        squared_norms[group] += f64::from(value) * f64::from(value);
    }
    for squared_norm in squared_norms.iter_mut() {
        *squared_norm = squared_norm.sqrt();
    }
    let mut materialized = temporary_vec(backend, context, direction.len())?;
    for (index, value) in direction.iter().copied().enumerate() {
        if index % 1_024 == 0 {
            context.cancellation.check()?;
        }
        let group = parametrization
            .dimension
            .map_or(0, |_| (index / inner_stride) % group_count);
        materialized.try_push(
            (f64::from(value) * f64::from(magnitude[group]) / squared_norms[group]) as f32,
        )?;
    }
    context.cancellation.check()?;
    tensor_from_f32(
        backend,
        parametrization.direction.descriptor().shape(),
        &materialized,
        parametrization.direction.descriptor().dtype(),
        parametrization.direction.descriptor().device(),
        context,
    )
}

fn materialize_spectral_norm(
    backend: &CpuBackend,
    parametrization: &NativeSpectralNorm,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, NativeSpectralNorm), NativeOpsError> {
    context.cancellation.check()?;
    let shape = shape_to_usize(parametrization.original.descriptor().shape())?;
    let rows = *shape
        .get(parametrization.config.dimension)
        .ok_or(NativeOpsError::Invalid(
            "spectral-normalization dimension is outside the weight rank",
        ))?;
    let inner_stride = checked_product(
        shape
            .get(parametrization.config.dimension + 1..)
            .ok_or(NativeOpsError::Invalid(
                "spectral-normalization dimension is outside the weight rank",
            ))?,
        "spectral-normalization inner stride",
    )?;
    let outer = checked_product(
        shape
            .get(..parametrization.config.dimension)
            .ok_or(NativeOpsError::Invalid(
                "spectral-normalization dimension is outside the weight rank",
            ))?,
        "spectral-normalization outer size",
    )?;
    let columns = outer
        .checked_mul(inner_stride)
        .ok_or(NativeOpsError::Invalid(
            "spectral-normalization matrix shape overflowed",
        ))?;
    if rows == 0 || columns == 0 {
        return Err(NativeOpsError::Invalid(
            "spectral-normalization weight dimensions must be nonzero",
        ));
    }
    let values = tensor_to_f32(backend, &parametrization.original, context)?;
    let matrix_len = rows.checked_mul(columns).ok_or(NativeOpsError::Invalid(
        "spectral-normalization matrix size overflowed",
    ))?;
    if values.len() != matrix_len {
        return Err(NativeOpsError::Invalid(
            "spectral-normalization weight value count is invalid",
        ));
    }
    let mut matrix = temporary_filled(backend, context, matrix_len, 0.0_f32)?;
    let row_block = rows
        .checked_mul(inner_stride)
        .ok_or(NativeOpsError::Invalid(
            "spectral-normalization row block overflowed",
        ))?;
    for (linear, value) in values.iter().copied().enumerate() {
        if linear % 1_024 == 0 {
            context.cancellation.check()?;
        }
        let row = (linear / inner_stride) % rows;
        let outer_index = linear / row_block;
        let inner_index = linear % inner_stride;
        let column = outer_index
            .checked_mul(inner_stride)
            .and_then(|value| value.checked_add(inner_index))
            .ok_or(NativeOpsError::Invalid(
                "spectral-normalization column overflowed",
            ))?;
        let matrix_index = row
            .checked_mul(columns)
            .and_then(|value| value.checked_add(column))
            .ok_or(NativeOpsError::Invalid(
                "spectral-normalization matrix index overflowed",
            ))?;
        *matrix.get_mut(matrix_index).ok_or(NativeOpsError::Invalid(
            "spectral-normalization matrix index is invalid",
        ))? = value;
    }
    let mut left = temporary_vec(backend, context, rows)?;
    if let Some(previous) = &parametrization.left {
        for value in previous.iter().copied() {
            left.try_push(value)?;
        }
    } else {
        let initial = (rows as f32).sqrt().recip();
        for _ in 0..rows {
            left.try_push(initial)?;
        }
    }
    if left.len() != rows {
        return Err(NativeOpsError::Invalid(
            "spectral-normalization left vector length is invalid",
        ));
    }
    let mut right = temporary_vec(backend, context, columns)?;
    if let Some(previous) = &parametrization.right {
        for value in previous.iter().copied() {
            right.try_push(value)?;
        }
    } else {
        let initial = (columns as f32).sqrt().recip();
        for _ in 0..columns {
            right.try_push(initial)?;
        }
    }
    if right.len() != columns {
        return Err(NativeOpsError::Invalid(
            "spectral-normalization right vector length is invalid",
        ));
    }
    for _ in 0..parametrization.config.power_iterations {
        context.cancellation.check()?;
        right.fill(0.0);
        for row in 0..rows {
            let left_value = *left.get(row).ok_or(NativeOpsError::Invalid(
                "spectral-normalization left vector index is invalid",
            ))?;
            for column in 0..columns {
                let matrix_value =
                    *matrix
                        .get(row * columns + column)
                        .ok_or(NativeOpsError::Invalid(
                            "spectral-normalization matrix index is invalid",
                        ))?;
                let slot = right.get_mut(column).ok_or(NativeOpsError::Invalid(
                    "spectral-normalization right vector index is invalid",
                ))?;
                *slot += matrix_value * left_value;
            }
        }
        normalize_spectral_vector(&mut right, parametrization.config.epsilon)?;
        left.fill(0.0);
        for row in 0..rows {
            let slot = left.get_mut(row).ok_or(NativeOpsError::Invalid(
                "spectral-normalization left vector index is invalid",
            ))?;
            for column in 0..columns {
                *slot += *matrix
                    .get(row * columns + column)
                    .ok_or(NativeOpsError::Invalid(
                        "spectral-normalization matrix index is invalid",
                    ))?
                    * *right.get(column).ok_or(NativeOpsError::Invalid(
                        "spectral-normalization right vector index is invalid",
                    ))?;
            }
        }
        normalize_spectral_vector(&mut left, parametrization.config.epsilon)?;
    }
    context.cancellation.check()?;
    let mut sigma = 0.0_f64;
    for row in 0..rows {
        for column in 0..columns {
            sigma += f64::from(*left.get(row).ok_or(NativeOpsError::Invalid(
                "spectral-normalization left vector index is invalid",
            ))?) * f64::from(*matrix.get(row * columns + column).ok_or(
                NativeOpsError::Invalid("spectral-normalization matrix index is invalid"),
            )?) * f64::from(*right.get(column).ok_or(NativeOpsError::Invalid(
                "spectral-normalization right vector index is invalid",
            ))?);
        }
    }
    if !sigma.is_finite() || sigma.abs() <= f64::from(parametrization.config.epsilon) {
        return Err(NativeOpsError::Invalid(
            "spectral-normalization dominant singular value is invalid",
        ));
    }
    let mut normalized = temporary_vec(backend, context, values.len())?;
    for value in values.iter() {
        normalized.try_push((f64::from(*value) / sigma) as f32)?;
    }
    context.cancellation.check()?;
    let materialized = tensor_from_f32(
        backend,
        parametrization.original.descriptor().shape(),
        &normalized,
        parametrization.original.descriptor().dtype(),
        parametrization.original.descriptor().device(),
        context,
    )?;
    let mut next = parametrization.clone();
    next.left = Some(left.to_vec());
    next.right = Some(right.to_vec());
    Ok((materialized, next))
}

fn normalize_spectral_vector(values: &mut [f32], epsilon: f32) -> Result<(), NativeOpsError> {
    let squared_norm = values.iter().try_fold(0.0_f64, |sum, value| {
        let square = f64::from(*value) * f64::from(*value);
        let next = sum + square;
        next.is_finite()
            .then_some(next)
            .ok_or(NativeOpsError::Invalid(
                "spectral-normalization vector norm is not finite",
            ))
    })?;
    let norm = squared_norm.sqrt();
    if norm <= f64::from(epsilon) {
        return Err(NativeOpsError::Invalid(
            "spectral-normalization vector norm is zero",
        ));
    }
    for value in values {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

fn validate_bias_shape(
    expects_bias: bool,
    output_width: usize,
    bias: Option<&Tensor>,
) -> Result<(), NativeOpsError> {
    match (expects_bias, bias) {
        (true, Some(bias))
            if bias.descriptor().shape()
                == [u64::try_from(output_width)
                    .map_err(|_| NativeOpsError::Invalid("bias width overflow"))?] =>
        {
            Ok(())
        }
        (false, None) => Ok(()),
        (true, None) => Err(NativeOpsError::Invalid("module requires a bias parameter")),
        (false, Some(_)) => Err(NativeOpsError::Invalid("module does not accept a bias")),
        (true, Some(_)) => Err(NativeOpsError::Invalid(
            "bias shape does not match module output width",
        )),
    }
}

fn validate_parameter_shape(
    expected: bool,
    shape: &[usize],
    parameter: Option<&Tensor>,
) -> Result<(), NativeOpsError> {
    match (expected, parameter) {
        (true, Some(parameter)) => {
            let expected_shape = shape
                .iter()
                .map(|value| {
                    u64::try_from(*value)
                        .map_err(|_| NativeOpsError::Invalid("parameter shape overflow"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if parameter.descriptor().shape() == expected_shape {
                Ok(())
            } else {
                Err(NativeOpsError::Invalid(
                    "parameter shape does not match module configuration",
                ))
            }
        }
        (false, None) => Ok(()),
        (true, None) => Err(NativeOpsError::Invalid("module requires this parameter")),
        (false, Some(_)) => Err(NativeOpsError::Invalid(
            "module configuration does not accept this parameter",
        )),
    }
}

fn crop_causal_weight(
    backend: &CpuBackend,
    weight: &[f32],
    weight_shape: &[usize],
    input_depth: usize,
    context: &ExecutionContext<'_>,
) -> Result<(CpuWorkspaceVec<f32>, Vec<usize>), NativeOpsError> {
    if weight_shape.len() != 5 || input_depth == 0 {
        return Err(NativeOpsError::Invalid(
            "causal convolution requires nonempty OIDHW weight and input depth",
        ));
    }
    let retained_depth = weight_shape[2].min(input_depth);
    let plane = weight_shape[3]
        .checked_mul(weight_shape[4])
        .ok_or(NativeOpsError::Invalid("causal convolution plane overflow"))?;
    let original_block = weight_shape[2]
        .checked_mul(plane)
        .ok_or(NativeOpsError::Invalid("causal convolution block overflow"))?;
    let retained_block = retained_depth
        .checked_mul(plane)
        .ok_or(NativeOpsError::Invalid("causal convolution crop overflow"))?;
    let leading = weight_shape[0]
        .checked_mul(weight_shape[1])
        .ok_or(NativeOpsError::Invalid(
            "causal convolution channels overflow",
        ))?;
    if weight.len()
        != leading
            .checked_mul(original_block)
            .ok_or(NativeOpsError::Invalid(
                "causal convolution weight overflow",
            ))?
    {
        return Err(NativeOpsError::Invalid(
            "causal convolution weight value count mismatch",
        ));
    }
    let skipped = original_block - retained_block;
    let cropped_count = leading
        .checked_mul(retained_block)
        .ok_or(NativeOpsError::Invalid(
            "causal convolution output overflow",
        ))?;
    let mut cropped = temporary_vec(backend, context, cropped_count)?;
    for block in weight.chunks_exact(original_block) {
        for value in block[skipped..].iter().copied() {
            cropped.try_push(value)?;
        }
    }
    let mut shape = weight_shape.to_vec();
    shape[2] = retained_depth;
    Ok((cropped, shape))
}

fn shape_to_usize(shape: &[u64]) -> Result<Vec<usize>, NativeOpsError> {
    shape
        .iter()
        .map(|value| {
            usize::try_from(*value).map_err(|_| NativeOpsError::Invalid("tensor shape overflow"))
        })
        .collect()
}

fn shape_to_u64(shape: &[usize]) -> Result<Vec<u64>, NativeOpsError> {
    shape
        .iter()
        .map(|value| {
            u64::try_from(*value).map_err(|_| NativeOpsError::Invalid("tensor shape overflow"))
        })
        .collect()
}

fn checked_product(values: &[usize], name: &'static str) -> Result<usize, NativeOpsError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(NativeOpsError::Invalid(name))
    })
}

fn scaled_dimension(input: u64, scale: f64, subject: &'static str) -> Result<u64, NativeOpsError> {
    let output = (input as f64 * scale).floor();
    if !output.is_finite() || output < 1.0 || output > u64::MAX as f64 {
        return Err(NativeOpsError::Invalid(subject));
    }
    Ok(output as u64)
}

pub(crate) fn tensor_to_f32(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, NativeOpsError> {
    context.cancellation.check()?;
    let count = usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| NativeOpsError::Invalid("tensor value count overflow"))?;
    let mut values = temporary_vec(backend, context, count)?;
    for linear_index in 0..count {
        if linear_index.is_multiple_of(1_024) {
            context.cancellation.check()?;
        }
        let linear_index = u64::try_from(linear_index)
            .map_err(|_| NativeOpsError::Invalid("tensor linear index overflow"))?;
        let value = match input
            .descriptor()
            .dtype()
            .decode_scalar(input.linear_element_bytes(linear_index)?)?
        {
            DecodedScalar::Boolean(value) => f32::from(u8::from(value)),
            DecodedScalar::Signed(value) => value as f32,
            DecodedScalar::Unsigned(value) => value as f32,
            DecodedScalar::Real(value) => value as f32,
            DecodedScalar::Complex { .. } => {
                return Err(NativeOpsError::Invalid(
                    "complex module parameters cannot be converted to f32",
                ));
            }
        };
        values.try_push(value)?;
    }
    Ok(values)
}

pub(crate) fn tensor_from_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeOpsError> {
    tensor_from_f32_with_context_exact_native(backend, shape, values, dtype, device, context)
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn cast_to(
    backend: &CpuBackend,
    input: &Tensor,
    dtype: DType,
    device: DeviceId,
    non_blocking: bool,
    copy: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeOpsError> {
    cast_to_with_context_exact_native(backend, input, dtype, device, non_blocking, copy, context)
        .map_err(Into::into)
}

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
) -> Result<CpuWorkspaceVec<T>, NativeOpsError> {
    backend.workspace_vec(context, capacity).map_err(Into::into)
}

fn temporary_filled<T: Clone>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    length: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, NativeOpsError> {
    let mut values = temporary_vec(backend, context, length)?;
    for _ in 0..length {
        values.try_push(value.clone())?;
    }
    Ok(values)
}
