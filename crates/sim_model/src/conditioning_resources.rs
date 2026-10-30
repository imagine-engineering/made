use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId, ExecutionContext, Scalar,
    StorageId, StreamId, Tensor, TensorBackend, TensorError,
    generated_activation_normalization_functional_01::{
        FunctionalError, GeluApproximation, gelu_with_context_exact_native,
        layer_norm_with_context_exact_native, silu_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_02::tanh_with_context_exact_native,
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, ElementwiseRuntimePartThreeError, real_add_with_context_exact_native,
        real_multiply_with_context_exact_native, sigmoid_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_09::{
        ElementwiseRuntimePartNineError, full_like_with_context_exact_native,
    },
    generated_indexing_masking_01::{
        IndexingMaskingPartOneError, NonzeroOutput, gather_method_with_context_exact_native,
        narrow_method_exact_native, nonzero_with_context_exact_native,
        scatter_method_with_context_exact_native,
    },
    generated_linear_algebra_01::{LinearAlgebraPartOneError, matmul_with_context_exact_native},
    generated_neural_network_functional_01::{
        NeuralNetworkFunctionalError, linear_with_context_exact_native,
    },
    generated_reduction_02::tensor_sum_with_context_exact_native,
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, tensor_repeat_with_context_exact_native,
        torch_cat_with_context_exact_native, torch_reshape_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use std::{
    collections::{BTreeMap, HashSet},
    mem,
};
use thiserror::Error;

use crate::{
    attention::{
        AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionRequest,
        scaled_dot_product_attention_tensor_with_context,
    },
    clip_vision::{
        ClipVisionActivation, ClipVisionConfiguration, ClipVisionError, ClipVisionIntermediate,
        ClipVisionLayerWeights, ClipVisionModelType, ClipVisionOutput, ClipVisionWeights,
        NativeClipVision,
    },
    model_family::{
        ModelDetectionRule, ModelFamilyIdentity, ModelProbe, detect_model_family_rules,
    },
};

pub const STYLE_MODEL_NODES_SOURCE_SHA256: &str =
    "b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b";
pub const STYLE_MODEL_SD_SOURCE_SHA256: &str =
    "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42";
pub const STYLE_MODEL_OPS_SOURCE_SHA256: &str =
    "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42";
pub const STYLE_ADAPTER_SOURCE_SHA256: &str =
    "efc52cc85f941e11b509c0339e8950a6680d031b09bec36736d8834b9ccfa1af";
pub const FLUX_REDUX_SOURCE_SHA256: &str =
    "3b7abf43e15fc7b9613a64e701635f943351fa47f5036d87427ea672f93c7952";
pub const PHOTOMAKER_SOURCE_SHA256: &str =
    "c7e86d4684d2884eda20250bd7122e31a8aaf7bfa003245ad0e0431a9ef1957e";
pub const PHOTOMAKER_CLIP_VISION_SOURCE_SHA256: &str =
    "8e3cc5d5d257b52d120885ba7427ff3fdf56129a485fdadac3b6215ae2c67b20";
pub const GLIGEN_SOURCE_SHA256: &str =
    "87c4297809a1a0a7727e3623cef0930463080dfedc068eb97c5e19bc6e155d0c";
pub const GLIGEN_ATTENTION_SOURCE_SHA256: &str =
    "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e";
pub const GLIGEN_SAMPLERS_SOURCE_SHA256: &str =
    "d882256ae9baa1d23f1367ab2ec3b021fdc15fe39ce4cb49ea2c1ee10026a649";
pub const GLIGEN_OPENAIMODEL_SOURCE_SHA256: &str =
    "9d27fb036cab8a262ef3d866a643f7fdc40994022616f1b8be14b7d919f57f96";

const MAX_STATE_KEY_BYTES: usize = 1_024;
const DIGEST_CHUNK_BYTES: usize = 64 * 1_024;
const STYLE_LAYER_COUNT: usize = 3;
const STYLE_SOURCE_WIDTH: usize = 1_024;
const STYLE_SOURCE_CONTEXT_WIDTH: usize = 768;
const STYLE_SOURCE_HEADS: usize = 8;
const STYLE_SOURCE_TOKENS: usize = 8;
const REDUX_SOURCE_INPUT_WIDTH: usize = 1_152;
const REDUX_SOURCE_HIDDEN_WIDTH: usize = 12_288;
const REDUX_SOURCE_OUTPUT_WIDTH: usize = 4_096;
const REDUCED_STYLE_WIDTH: usize = 8;
const REDUCED_STYLE_CONTEXT_WIDTH: usize = 6;
const REDUCED_STYLE_HEADS: usize = 2;
const REDUCED_STYLE_TOKENS: usize = 2;
const REDUCED_REDUX_INPUT_WIDTH: usize = 4;
const REDUCED_REDUX_HIDDEN_WIDTH: usize = 12;
const REDUCED_REDUX_OUTPUT_WIDTH: usize = 6;
const LAYER_NORM_EPSILON: f32 = 1.0e-5;
const QUICK_GELU_SCALE: f32 = 1.702;
const PHOTOMAKER_CLIP_STATE_COUNT: usize = 392;
const PHOTOMAKER_STATE_COUNT: usize = 407;
const PHOTOMAKER_SOURCE_HIDDEN: usize = 1_024;
const PHOTOMAKER_SOURCE_INTERMEDIATE: usize = 4_096;
const PHOTOMAKER_SOURCE_HEADS: usize = 16;
const PHOTOMAKER_SOURCE_LAYERS: usize = 24;
const PHOTOMAKER_SOURCE_IMAGE: usize = 224;
const PHOTOMAKER_SOURCE_PATCH: usize = 14;
const PHOTOMAKER_SOURCE_PROJECTION: usize = 768;
const PHOTOMAKER_SOURCE_EXTRA_PROJECTION: usize = 1_280;
const GLIGEN_POSITION_STATE_COUNT: usize = 8;
const GLIGEN_FUSER_STATE_COUNT: usize = 17;
const GLIGEN_MAX_OBJECTS: usize = 30;
const GLIGEN_POSITION_WIDTH: usize = 64;
const GLIGEN_POSITION_HIDDEN: usize = 512;
const GLIGEN_FOURIER_FREQUENCIES: usize = 8;
const GLIGEN_FOURIER_TEMPERATURE: f32 = 100.0;
#[cfg(any(test, feature = "test-support"))]
const PHOTOMAKER_REDUCED_HIDDEN: usize = 4;
#[cfg(any(test, feature = "test-support"))]
const PHOTOMAKER_REDUCED_INTERMEDIATE: usize = 8;
#[cfg(any(test, feature = "test-support"))]
const PHOTOMAKER_REDUCED_HEADS: usize = 2;
#[cfg(any(test, feature = "test-support"))]
const PHOTOMAKER_REDUCED_IMAGE: usize = 4;
#[cfg(any(test, feature = "test-support"))]
const PHOTOMAKER_REDUCED_PATCH: usize = 2;
#[cfg(any(test, feature = "test-support"))]
const PHOTOMAKER_REDUCED_PROJECTION: usize = 3;
#[cfg(any(test, feature = "test-support"))]
const PHOTOMAKER_REDUCED_EXTRA_PROJECTION: usize = 5;

const STYLE_DETECTION_RULES: [ModelDetectionRule; 1] = [ModelDetectionRule::KeyPresent {
    key: "style_embedding",
    score: 1,
}];
const REDUX_DETECTION_RULES: [ModelDetectionRule; 1] = [ModelDetectionRule::KeyPresent {
    key: "redux_down.weight",
    score: 1,
}];

#[derive(Clone, Debug)]
pub struct NativeStyleModelCheckpoint {
    pub artifact_sha256: String,
    pub ordered_state: Vec<(String, Tensor)>,
    pub memory_budget_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeStyleModelArchitecture {
    StyleAdapter,
    FluxRedux,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeStyleModelConfiguration {
    architecture: NativeStyleModelArchitecture,
    input_width: usize,
    hidden_width: usize,
    output_width: usize,
    heads: usize,
    layers: usize,
    output_tokens: usize,
    source_exact: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StateSpecification {
    key: String,
    shape: Vec<u64>,
}

#[derive(Clone, Debug)]
struct NativeStyleAttentionProjection {
    query_weight: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    query_bias: Tensor,
    key_bias: Tensor,
    value_bias: Tensor,
}

#[derive(Debug)]
pub struct NativeStyleModelResource {
    configuration: NativeStyleModelConfiguration,
    artifact_sha256: String,
    source_state: BTreeMap<String, Tensor>,
    execution_state: BTreeMap<String, Tensor>,
    style_attention: Box<[NativeStyleAttentionProjection]>,
    source_dtype: DType,
    stream: StreamId,
    memory_budget_bytes: u64,
    resident_bytes: u64,
    semantic_digest_sha256: String,
}

#[derive(Debug, Error)]
pub enum NativeStyleModelError {
    #[error("style-model execution was cancelled")]
    Cancelled,
    #[error("style-model architecture is unsupported")]
    UnsupportedArchitecture,
    #[error("style-model checkpoint is invalid: {0}")]
    InvalidCheckpoint(String),
    #[error("style-model state contains duplicate key {0}")]
    DuplicateStateKey(String),
    #[error("style-model state is missing key {0}")]
    MissingState(String),
    #[error("style-model state is unexpected: {0}")]
    UnexpectedState(String),
    #[error("style-model state {key} expected {expected:?}, got {actual:?} {actual_dtype:?}")]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        actual_dtype: DType,
    },
    #[error("style-model retained semantic state changed")]
    SemanticStateChanged,
    #[error("style-model CLIP-vision input is invalid: {0}")]
    InvalidInput(String),
    #[error("style-model shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("style-model allocation failed")]
    Allocation,
    #[error("style-model memory requirement {required} exceeds budget {budget}")]
    OutOfMemory { required: u64, budget: u64 },
    #[error("style-model canonical owner failed: {0}")]
    Canonical(String),
    #[error(transparent)]
    ClipVision(ClipVisionError),
    #[error(transparent)]
    Tensor(TensorError),
}

impl From<sim_types::CancellationError> for NativeStyleModelError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<TensorError> for NativeStyleModelError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled,
            error => Self::Tensor(error),
        }
    }
}

impl From<ClipVisionError> for NativeStyleModelError {
    fn from(error: ClipVisionError) -> Self {
        match error {
            ClipVisionError::Cancelled | ClipVisionError::Tensor(TensorError::Cancelled) => {
                Self::Cancelled
            }
            error => Self::ClipVision(error),
        }
    }
}

impl NativeStyleModelResource {
    pub fn from_checkpoint(
        backend: &CpuBackend,
        checkpoint: NativeStyleModelCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeStyleModelError> {
        Self::checked(backend, checkpoint, true, context)
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn from_reduced_fixture(
        backend: &CpuBackend,
        checkpoint: NativeStyleModelCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeStyleModelError> {
        Self::checked(backend, checkpoint, false, context)
    }

    fn checked(
        backend: &CpuBackend,
        checkpoint: NativeStyleModelCheckpoint,
        source_exact: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeStyleModelError> {
        context.check()?;
        validate_sha256(&checkpoint.artifact_sha256)?;
        if checkpoint.memory_budget_bytes == 0 || checkpoint.ordered_state.is_empty() {
            return Err(NativeStyleModelError::InvalidCheckpoint(
                "state cardinality or memory budget is invalid".to_owned(),
            ));
        }
        raw_checkpoint_preflight(
            &checkpoint.artifact_sha256,
            checkpoint.artifact_sha256.capacity(),
            &checkpoint.ordered_state,
            checkpoint.ordered_state.capacity(),
            checkpoint.memory_budget_bytes,
        )?;
        let configuration = detect_configuration(&checkpoint.ordered_state, source_exact)?;
        let specifications = state_manifest(configuration)?;
        validate_ordered_keys(&checkpoint.ordered_state, &specifications)?;
        let source_dtype = validate_source_state(
            &checkpoint.ordered_state,
            &specifications,
            context.stream,
            context.cancellation,
        )?;
        let construction_peak = construction_memory_preflight(
            &checkpoint.artifact_sha256,
            checkpoint.artifact_sha256.capacity(),
            &checkpoint.ordered_state,
            &specifications,
            configuration,
        )?;
        if construction_peak > checkpoint.memory_budget_bytes {
            return Err(NativeStyleModelError::OutOfMemory {
                required: construction_peak,
                budget: checkpoint.memory_budget_bytes,
            });
        }

        let source_state = checkpoint
            .ordered_state
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let mut execution_state = BTreeMap::new();
        for (index, specification) in specifications.iter().enumerate() {
            if index.is_multiple_of(8) {
                context.check()?;
            }
            let source = source_state
                .get(&specification.key)
                .ok_or_else(|| NativeStyleModelError::MissingState(specification.key.clone()))?;
            let projected = cast_to_with_context_exact_native(
                backend,
                source,
                DType::F32,
                DeviceId::CPU,
                false,
                true,
                context,
            )
            .map_err(canonical)?;
            validate_finite_tensor(&specification.key, &projected, context.cancellation)?;
            execution_state.insert(specification.key.clone(), projected);
        }
        validate_distinct_projection_storage(&source_state, &execution_state)?;
        let style_attention =
            build_style_attention(backend, configuration, &execution_state, context)?;
        let semantic_digest_sha256 = semantic_digest(
            configuration,
            &checkpoint.artifact_sha256,
            source_dtype,
            &source_state,
            &execution_state,
            &specifications,
            context.cancellation,
        )?;
        let resident_bytes =
            resident_tensor_bytes([&source_state, &execution_state], context.cancellation)?
                .checked_add(resident_owned_bytes(
                    &checkpoint.artifact_sha256,
                    checkpoint.artifact_sha256.capacity(),
                    &semantic_digest_sha256,
                    semantic_digest_sha256.capacity(),
                    &source_state,
                    &execution_state,
                    style_attention.len(),
                )?)
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
        if resident_bytes > checkpoint.memory_budget_bytes {
            return Err(NativeStyleModelError::OutOfMemory {
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
            style_attention,
            source_dtype,
            stream: context.stream,
            memory_budget_bytes: checkpoint.memory_budget_bytes,
            resident_bytes,
            semantic_digest_sha256,
        })
    }

    pub fn identifier(&self) -> &'static str {
        match self.configuration.architecture {
            NativeStyleModelArchitecture::StyleAdapter => "style-adapter",
            NativeStyleModelArchitecture::FluxRedux => "flux-redux",
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

    pub fn resident_owned_bytes(&self) -> Result<u64, NativeStyleModelError> {
        resident_owned_bytes(
            &self.artifact_sha256,
            self.artifact_sha256.capacity(),
            &self.semantic_digest_sha256,
            self.semantic_digest_sha256.capacity(),
            &self.source_state,
            &self.execution_state,
            self.style_attention.len(),
        )
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(StorageId, u64)>, NativeStyleModelError> {
        resident_tensor_allocations(
            [&self.source_state, &self.execution_state],
            &CancellationToken::default(),
        )
    }

    pub fn reconstruct_checkpoint(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NativeStyleModelCheckpoint, NativeStyleModelError> {
        self.validate(cancellation)?;
        let specifications = state_manifest(self.configuration)?;
        let mut ordered_state = Vec::new();
        ordered_state
            .try_reserve_exact(specifications.len())
            .map_err(|_| NativeStyleModelError::Allocation)?;
        for (index, specification) in specifications.iter().enumerate() {
            if index.is_multiple_of(8) {
                cancellation.check()?;
            }
            ordered_state.push((
                specification.key.clone(),
                self.source_state
                    .get(&specification.key)
                    .ok_or_else(|| NativeStyleModelError::MissingState(specification.key.clone()))?
                    .clone(),
            ));
        }
        cancellation.check()?;
        Ok(NativeStyleModelCheckpoint {
            artifact_sha256: self.artifact_sha256.clone(),
            ordered_state,
            memory_budget_bytes: self.memory_budget_bytes,
        })
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), NativeStyleModelError> {
        cancellation.check()?;
        let specifications = state_manifest(self.configuration)?;
        let source_order = specifications
            .iter()
            .map(|specification| {
                self.source_state
                    .get(&specification.key)
                    .cloned()
                    .map(|tensor| (specification.key.clone(), tensor))
                    .ok_or_else(|| NativeStyleModelError::MissingState(specification.key.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dtype =
            validate_source_state(&source_order, &specifications, self.stream, cancellation)?;
        if dtype != self.source_dtype
            || self.source_state.len() != specifications.len()
            || self.execution_state.len() != specifications.len()
        {
            return Err(NativeStyleModelError::SemanticStateChanged);
        }
        for specification in &specifications {
            let projected = self
                .execution_state
                .get(&specification.key)
                .ok_or_else(|| NativeStyleModelError::MissingState(specification.key.clone()))?;
            if projected.descriptor().shape() != specification.shape
                || projected.descriptor().dtype() != DType::F32
                || projected.descriptor().device() != DeviceId::CPU
                || projected.descriptor().stream() != self.stream
                || !projected.descriptor().is_contiguous()?
            {
                return Err(NativeStyleModelError::SemanticStateChanged);
            }
            validate_finite_tensor(&specification.key, projected, cancellation)?;
        }
        validate_distinct_projection_storage(&self.source_state, &self.execution_state)?;
        validate_attention_views(self, cancellation)?;
        let digest = semantic_digest(
            self.configuration,
            &self.artifact_sha256,
            self.source_dtype,
            &self.source_state,
            &self.execution_state,
            &specifications,
            cancellation,
        )?;
        let resident =
            resident_tensor_bytes([&self.source_state, &self.execution_state], cancellation)?
                .checked_add(self.resident_owned_bytes()?)
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
        if digest != self.semantic_digest_sha256 || resident != self.resident_bytes {
            return Err(NativeStyleModelError::SemanticStateChanged);
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn get_cond(
        &self,
        backend: &CpuBackend,
        input: &ClipVisionOutput,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeStyleModelError> {
        context.check()?;
        self.validate(context.cancellation)?;
        input.validate()?;
        let hidden = &input.last_hidden_state;
        let [batch, tokens, width] = hidden.descriptor().shape() else {
            return Err(NativeStyleModelError::InvalidInput(
                "last_hidden_state must have rank three".to_owned(),
            ));
        };
        if *batch == 0 || *tokens == 0 || *width != u64_from(self.configuration.input_width)? {
            return Err(NativeStyleModelError::InvalidInput(
                "last_hidden_state has the wrong shape".to_owned(),
            ));
        }
        if hidden.descriptor().dtype() != DType::F32
            || hidden.descriptor().device() != DeviceId::CPU
            || hidden.descriptor().stream() != context.stream
            || !hidden.descriptor().is_contiguous()?
        {
            return Err(NativeStyleModelError::InvalidInput(
                "last_hidden_state must be contiguous F32 CPU state on the execution stream"
                    .to_owned(),
            ));
        }
        let batch = usize_from(*batch)?;
        let tokens = usize_from(*tokens)?;
        preflight_invocation_memory(self, batch, tokens)?;
        let input_values =
            tensor_to_f32_with_context_exact_native(backend, hidden, context).map_err(canonical)?;
        if input_values.iter().any(|value| !value.is_finite()) {
            return Err(NativeStyleModelError::InvalidInput(
                "last_hidden_state contains a non-finite value".to_owned(),
            ));
        }
        let output = match self.configuration.architecture {
            NativeStyleModelArchitecture::StyleAdapter => {
                self.execute_style_adapter(backend, hidden, batch, tokens, context)?
            }
            NativeStyleModelArchitecture::FluxRedux => {
                self.execute_flux_redux(backend, &input_values, batch, tokens, context)?
            }
        };
        context.check()?;
        Ok(output)
    }

    fn execute_style_adapter(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        batch: usize,
        input_tokens: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeStyleModelError> {
        let width = self.configuration.input_width;
        let style_embedding = self.execution_tensor("style_embedding")?;
        let repeated = tensor_repeat_with_context_exact_native(
            backend,
            style_embedding,
            &[i64_from(batch)?, 1, 1],
            context,
        )
        .map_err(canonical)?;
        let zeros = full_like_with_context_exact_native(
            backend,
            &repeated,
            Scalar::Float(0.0),
            Some(DType::F32),
            context,
        )
        .map_err(canonical)?;
        let repeated = real_add_with_context_exact_native(backend, &repeated, &zeros, context)
            .map_err(canonical)?;
        let concatenated =
            torch_cat_with_context_exact_native(backend, &[input.clone(), repeated], 1, context)
                .map_err(canonical)?;
        let sequence = input_tokens
            .checked_add(self.configuration.output_tokens)
            .ok_or(NativeStyleModelError::ShapeOverflow)?;
        let mut values = tensor_to_f32_with_context_exact_native(backend, &concatenated, context)
            .map_err(canonical)?;
        values = self.layer_norm(
            backend,
            &values,
            &[batch, sequence, width],
            "ln_pre.weight",
            "ln_pre.bias",
            context,
        )?;

        for layer in 0..self.configuration.layers {
            context.check()?;
            let normalized = self.layer_norm(
                backend,
                &values,
                &[batch, sequence, width],
                &format!("transformer_layes.{layer}.ln_1.weight"),
                &format!("transformer_layes.{layer}.ln_1.bias"),
                context,
            )?;
            let attention =
                self.execute_attention(backend, layer, &normalized, batch, sequence, context)?;
            let projected = self.linear_state(
                backend,
                &attention,
                &[batch, sequence, width],
                &format!("transformer_layes.{layer}.attn.out_proj.weight"),
                Some(&format!("transformer_layes.{layer}.attn.out_proj.bias")),
                context,
            )?;
            add_in_place(&mut values, &projected, context.cancellation)?;

            let normalized = self.layer_norm(
                backend,
                &values,
                &[batch, sequence, width],
                &format!("transformer_layes.{layer}.ln_2.weight"),
                &format!("transformer_layes.{layer}.ln_2.bias"),
                context,
            )?;
            let expanded = self.linear_state(
                backend,
                &normalized,
                &[batch, sequence, width],
                &format!("transformer_layes.{layer}.mlp.c_fc.weight"),
                Some(&format!("transformer_layes.{layer}.mlp.c_fc.bias")),
                context,
            )?;
            let activated = quick_gelu(
                backend,
                &expanded,
                &[
                    batch,
                    sequence,
                    width
                        .checked_mul(4)
                        .ok_or(NativeStyleModelError::ShapeOverflow)?,
                ],
                context,
            )?;
            let projected = self.linear_state(
                backend,
                &activated,
                &[
                    batch,
                    sequence,
                    width
                        .checked_mul(4)
                        .ok_or(NativeStyleModelError::ShapeOverflow)?,
                ],
                &format!("transformer_layes.{layer}.mlp.c_proj.weight"),
                Some(&format!("transformer_layes.{layer}.mlp.c_proj.bias")),
                context,
            )?;
            add_in_place(&mut values, &projected, context.cancellation)?;
            context.check()?;
        }

        let selected_count = batch
            .checked_mul(self.configuration.output_tokens)
            .and_then(|value| value.checked_mul(width))
            .ok_or(NativeStyleModelError::ShapeOverflow)?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(selected_count)
            .map_err(|_| NativeStyleModelError::Allocation)?;
        for batch_index in 0..batch {
            context.check()?;
            let start = batch_index
                .checked_mul(sequence)
                .and_then(|value| value.checked_add(input_tokens))
                .and_then(|value| value.checked_mul(width))
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
            let end = start
                .checked_add(
                    self.configuration
                        .output_tokens
                        .checked_mul(width)
                        .ok_or(NativeStyleModelError::ShapeOverflow)?,
                )
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
            selected.extend_from_slice(
                values
                    .get(start..end)
                    .ok_or(NativeStyleModelError::ShapeOverflow)?,
            );
        }
        let selected = self.layer_norm(
            backend,
            &selected,
            &[batch, self.configuration.output_tokens, width],
            "ln_post.weight",
            "ln_post.bias",
            context,
        )?;
        let selected = tensor_from_f32_with_context_exact_native(
            backend,
            &[
                u64_from(batch)?,
                u64_from(self.configuration.output_tokens)?,
                u64_from(width)?,
            ],
            &selected,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(canonical)?;
        let output = matmul_with_context_exact_native(
            backend,
            &selected,
            self.execution_tensor("proj")?,
            context,
        )
        .map_err(canonical)?;
        context.check()?;
        Ok(output)
    }

    fn execute_flux_redux(
        &self,
        backend: &CpuBackend,
        input: &[f32],
        batch: usize,
        tokens: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeStyleModelError> {
        let expanded = self.linear_state(
            backend,
            input,
            &[batch, tokens, self.configuration.input_width],
            "redux_up.weight",
            Some("redux_up.bias"),
            context,
        )?;
        let activated = silu_with_context_exact_native(backend, &expanded, DeviceId::CPU, context)
            .map_err(canonical)?;
        let output = self.linear_state(
            backend,
            &activated,
            &[batch, tokens, self.configuration.hidden_width],
            "redux_down.weight",
            Some("redux_down.bias"),
            context,
        )?;
        let output = tensor_from_f32_with_context_exact_native(
            backend,
            &[
                u64_from(batch)?,
                u64_from(tokens)?,
                u64_from(self.configuration.output_width)?,
            ],
            &output,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(canonical)?;
        context.check()?;
        Ok(output)
    }

    fn execute_attention(
        &self,
        backend: &CpuBackend,
        layer: usize,
        input: &[f32],
        batch: usize,
        tokens: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, NativeStyleModelError> {
        let projection = self
            .style_attention
            .get(layer)
            .ok_or(NativeStyleModelError::SemanticStateChanged)?;
        let width = self.configuration.input_width;
        let query = linear_tensor(
            backend,
            input,
            &[batch, tokens, width],
            &projection.query_weight,
            Some(&projection.query_bias),
            context,
        )?;
        let key = linear_tensor(
            backend,
            input,
            &[batch, tokens, width],
            &projection.key_weight,
            Some(&projection.key_bias),
            context,
        )?;
        let value = linear_tensor(
            backend,
            input,
            &[batch, tokens, width],
            &projection.value_weight,
            Some(&projection.value_bias),
            context,
        )?;
        let head_dimension = width
            .checked_div(self.configuration.heads)
            .filter(|dimension| *dimension > 0)
            .ok_or(NativeStyleModelError::ShapeOverflow)?;
        let attention_workspace_bytes = u64_from(batch)?
            .checked_mul(u64_from(self.configuration.heads)?)
            .and_then(|value| value.checked_mul(u64_from(tokens).ok()?))
            .and_then(|value| value.checked_mul(u64_from(tokens).ok()?))
            .and_then(|value| value.checked_mul(8))
            .ok_or(NativeStyleModelError::ShapeOverflow)?;
        let shape = [
            u64_from(batch)?,
            u64_from(tokens)?,
            u64_from(self.configuration.heads)?,
            u64_from(head_dimension)?,
        ];
        let query = tensor_from_f32_with_context_exact_native(
            backend,
            &shape,
            &query,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(canonical)?;
        let key = tensor_from_f32_with_context_exact_native(
            backend,
            &shape,
            &key,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(canonical)?;
        let value = tensor_from_f32_with_context_exact_native(
            backend,
            &shape,
            &value,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(canonical)?;
        let output = scaled_dot_product_attention_tensor_with_context(
            backend,
            AttentionRequest {
                backend: AttentionBackend::PytorchSdp,
                fallback: AttentionFallbackPolicy::AllowExactNative,
                batch,
                query_tokens: tokens,
                key_tokens: tokens,
                heads: self.configuration.heads,
                head_dimension,
                value_dimension: head_dimension,
                scale: None,
                workspace_limit_bytes: usize::try_from(attention_workspace_bytes)
                    .map_err(|_| NativeStyleModelError::ShapeOverflow)?,
            },
            &query,
            &key,
            &value,
            None,
            context,
        )
        .map_err(attention_error)?;
        tensor_to_f32_with_context_exact_native(backend, &output, context).map_err(canonical)
    }

    fn layer_norm(
        &self,
        backend: &CpuBackend,
        input: &[f32],
        shape: &[usize],
        weight_key: &str,
        bias_key: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, NativeStyleModelError> {
        let weight = tensor_to_f32_with_context_exact_native(
            backend,
            self.execution_tensor(weight_key)?,
            context,
        )
        .map_err(canonical)?;
        let bias = tensor_to_f32_with_context_exact_native(
            backend,
            self.execution_tensor(bias_key)?,
            context,
        )
        .map_err(canonical)?;
        layer_norm_with_context_exact_native(
            backend,
            input,
            shape,
            &[self.configuration.input_width],
            Some(&weight),
            Some(&bias),
            LAYER_NORM_EPSILON,
            DeviceId::CPU,
            context,
        )
        .map_err(canonical)
    }

    fn linear_state(
        &self,
        backend: &CpuBackend,
        input: &[f32],
        input_shape: &[usize],
        weight_key: &str,
        bias_key: Option<&str>,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, NativeStyleModelError> {
        let weight = self.execution_tensor(weight_key)?;
        let bias = bias_key.map(|key| self.execution_tensor(key)).transpose()?;
        linear_tensor(backend, input, input_shape, weight, bias, context)
    }

    fn execution_tensor(&self, key: &str) -> Result<&Tensor, NativeStyleModelError> {
        self.execution_state
            .get(key)
            .ok_or_else(|| NativeStyleModelError::MissingState(key.to_owned()))
    }
}

fn detect_configuration(
    ordered_state: &[(String, Tensor)],
    source_exact: bool,
) -> Result<NativeStyleModelConfiguration, NativeStyleModelError> {
    let mut tensor_shapes = BTreeMap::new();
    for (key, tensor) in ordered_state {
        if tensor_shapes
            .insert(key.clone(), tensor.descriptor().shape().to_vec())
            .is_some()
        {
            return Err(NativeStyleModelError::DuplicateStateKey(key.clone()));
        }
    }
    let probe = ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    };
    if probe.tensor_shapes.contains_key("style_embedding") {
        detect_model_family_rules(
            ModelFamilyIdentity::new("COMFY-MODEL-0393", "style-adapter", "1")
                .map_err(canonical)?,
            &STYLE_DETECTION_RULES,
            &probe,
        )
        .map_err(canonical)?;
        Ok(if source_exact {
            style_configuration(
                STYLE_SOURCE_WIDTH,
                STYLE_SOURCE_CONTEXT_WIDTH,
                STYLE_SOURCE_HEADS,
                STYLE_SOURCE_TOKENS,
                true,
            )
        } else {
            style_configuration(
                REDUCED_STYLE_WIDTH,
                REDUCED_STYLE_CONTEXT_WIDTH,
                REDUCED_STYLE_HEADS,
                REDUCED_STYLE_TOKENS,
                false,
            )
        })
    } else if probe.tensor_shapes.contains_key("redux_down.weight") {
        detect_model_family_rules(
            ModelFamilyIdentity::new("COMFY-MODEL-0393", "flux-redux", "1").map_err(canonical)?,
            &REDUX_DETECTION_RULES,
            &probe,
        )
        .map_err(canonical)?;
        Ok(if source_exact {
            redux_configuration(
                REDUX_SOURCE_INPUT_WIDTH,
                REDUX_SOURCE_HIDDEN_WIDTH,
                REDUX_SOURCE_OUTPUT_WIDTH,
                true,
            )
        } else {
            redux_configuration(
                REDUCED_REDUX_INPUT_WIDTH,
                REDUCED_REDUX_HIDDEN_WIDTH,
                REDUCED_REDUX_OUTPUT_WIDTH,
                false,
            )
        })
    } else {
        Err(NativeStyleModelError::UnsupportedArchitecture)
    }
}

fn raw_checkpoint_preflight(
    artifact: &str,
    artifact_capacity: usize,
    ordered_state: &[(String, Tensor)],
    ordered_state_capacity: usize,
    memory_budget_bytes: u64,
) -> Result<(), NativeStyleModelError> {
    if !matches!(ordered_state.len(), 4 | 42) {
        return Err(NativeStyleModelError::UnexpectedState(format!(
            "expected 4 or 42 entries, got {}",
            ordered_state.len()
        )));
    }
    let key_bytes = ordered_state.iter().try_fold(0_u64, |total, (key, _)| {
        if key.is_empty() || key.len() > MAX_STATE_KEY_BYTES {
            return Err(NativeStyleModelError::InvalidCheckpoint(
                "state key length is invalid".to_owned(),
            ));
        }
        total
            .checked_add(u64_from(key.capacity())?)
            .ok_or(NativeStyleModelError::ShapeOverflow)
    })?;
    for (index, (key, _)) in ordered_state.iter().enumerate() {
        if ordered_state
            .iter()
            .skip(index + 1)
            .any(|(candidate, _)| candidate == key)
        {
            return Err(NativeStyleModelError::DuplicateStateKey(key.clone()));
        }
    }
    let tensor_bytes = ordered_state.iter().try_fold(0_u64, |total, (_, tensor)| {
        total
            .checked_add(tensor.storage_byte_len())
            .ok_or(NativeStyleModelError::ShapeOverflow)
    })?;
    let tuple_bytes = u64_from(ordered_state_capacity)?
        .checked_mul(
            u64::try_from(mem::size_of::<(String, Tensor)>())
                .map_err(|_| NativeStyleModelError::ShapeOverflow)?,
        )
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    let manifest_bytes = u64_from(ordered_state.len())?
        .checked_mul(
            u64::try_from(mem::size_of::<StateSpecification>())
                .map_err(|_| NativeStyleModelError::ShapeOverflow)?
                .checked_add(16)
                .ok_or(NativeStyleModelError::ShapeOverflow)?,
        )
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    let detector_and_manifest_keys = key_bytes
        .checked_mul(2)
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    let required = tensor_bytes
        .checked_add(u64_from(artifact_capacity.max(artifact.len()))?)
        .and_then(|value| value.checked_add(tuple_bytes))
        .and_then(|value| value.checked_add(detector_and_manifest_keys))
        .and_then(|value| value.checked_add(manifest_bytes))
        .and_then(|value| value.checked_add(map_node_estimate(ordered_state.len()).ok()?))
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    if required > memory_budget_bytes {
        return Err(NativeStyleModelError::OutOfMemory {
            required,
            budget: memory_budget_bytes,
        });
    }
    Ok(())
}

const fn style_configuration(
    width: usize,
    context_width: usize,
    heads: usize,
    output_tokens: usize,
    source_exact: bool,
) -> NativeStyleModelConfiguration {
    NativeStyleModelConfiguration {
        architecture: NativeStyleModelArchitecture::StyleAdapter,
        input_width: width,
        hidden_width: width * 4,
        output_width: context_width,
        heads,
        layers: STYLE_LAYER_COUNT,
        output_tokens,
        source_exact,
    }
}

const fn redux_configuration(
    input_width: usize,
    hidden_width: usize,
    output_width: usize,
    source_exact: bool,
) -> NativeStyleModelConfiguration {
    NativeStyleModelConfiguration {
        architecture: NativeStyleModelArchitecture::FluxRedux,
        input_width,
        hidden_width,
        output_width,
        heads: 0,
        layers: 0,
        output_tokens: 0,
        source_exact,
    }
}

fn state_manifest(
    configuration: NativeStyleModelConfiguration,
) -> Result<Vec<StateSpecification>, NativeStyleModelError> {
    let mut specifications = Vec::new();
    let mut push = |key: String, shape: Vec<u64>| {
        specifications.push(StateSpecification { key, shape });
    };
    match configuration.architecture {
        NativeStyleModelArchitecture::StyleAdapter => {
            let width = u64_from(configuration.input_width)?;
            let hidden = u64_from(configuration.hidden_width)?;
            push(
                "style_embedding".to_owned(),
                vec![1, u64_from(configuration.output_tokens)?, width],
            );
            push(
                "proj".to_owned(),
                vec![width, u64_from(configuration.output_width)?],
            );
            for layer in 0..configuration.layers {
                let prefix = format!("transformer_layes.{layer}");
                push(
                    format!("{prefix}.attn.in_proj_weight"),
                    vec![width * 3, width],
                );
                push(format!("{prefix}.attn.in_proj_bias"), vec![width * 3]);
                push(format!("{prefix}.attn.out_proj.weight"), vec![width, width]);
                push(format!("{prefix}.attn.out_proj.bias"), vec![width]);
                push(format!("{prefix}.ln_1.weight"), vec![width]);
                push(format!("{prefix}.ln_1.bias"), vec![width]);
                push(format!("{prefix}.mlp.c_fc.weight"), vec![hidden, width]);
                push(format!("{prefix}.mlp.c_fc.bias"), vec![hidden]);
                push(format!("{prefix}.mlp.c_proj.weight"), vec![width, hidden]);
                push(format!("{prefix}.mlp.c_proj.bias"), vec![width]);
                push(format!("{prefix}.ln_2.weight"), vec![width]);
                push(format!("{prefix}.ln_2.bias"), vec![width]);
            }
            push("ln_post.weight".to_owned(), vec![width]);
            push("ln_post.bias".to_owned(), vec![width]);
            push("ln_pre.weight".to_owned(), vec![width]);
            push("ln_pre.bias".to_owned(), vec![width]);
        }
        NativeStyleModelArchitecture::FluxRedux => {
            let input = u64_from(configuration.input_width)?;
            let hidden = u64_from(configuration.hidden_width)?;
            let output = u64_from(configuration.output_width)?;
            push("redux_up.weight".to_owned(), vec![hidden, input]);
            push("redux_up.bias".to_owned(), vec![hidden]);
            push("redux_down.weight".to_owned(), vec![output, hidden]);
            push("redux_down.bias".to_owned(), vec![output]);
        }
    }
    Ok(specifications)
}

fn validate_ordered_keys(
    ordered_state: &[(String, Tensor)],
    specifications: &[StateSpecification],
) -> Result<(), NativeStyleModelError> {
    if ordered_state.len() != specifications.len() {
        return Err(NativeStyleModelError::UnexpectedState(format!(
            "expected {} entries, got {}",
            specifications.len(),
            ordered_state.len()
        )));
    }
    for ((actual, _), expected) in ordered_state.iter().zip(specifications) {
        if actual != &expected.key {
            return Err(NativeStyleModelError::UnexpectedState(format!(
                "expected {}, got {actual}",
                expected.key
            )));
        }
    }
    Ok(())
}

fn validate_source_state(
    ordered_state: &[(String, Tensor)],
    specifications: &[StateSpecification],
    stream: StreamId,
    cancellation: &CancellationToken,
) -> Result<DType, NativeStyleModelError> {
    validate_ordered_keys(ordered_state, specifications)?;
    let mut source_dtype = None;
    let mut storage_ids = HashSet::new();
    for (index, ((key, tensor), specification)) in
        ordered_state.iter().zip(specifications).enumerate()
    {
        if index.is_multiple_of(8) {
            cancellation.check()?;
        }
        if key.len() > MAX_STATE_KEY_BYTES || key.is_empty() {
            return Err(NativeStyleModelError::InvalidCheckpoint(
                "state key length is invalid".to_owned(),
            ));
        }
        let descriptor = tensor.descriptor();
        if descriptor.shape() != specification.shape
            || !matches!(descriptor.dtype(), DType::F32 | DType::F16 | DType::Bf16)
        {
            return Err(NativeStyleModelError::StateShape {
                key: key.clone(),
                expected: specification.shape.clone(),
                actual: descriptor.shape().to_vec(),
                actual_dtype: descriptor.dtype(),
            });
        }
        if descriptor.device() != DeviceId::CPU
            || descriptor.stream() != stream
            || !descriptor.is_contiguous()?
        {
            return Err(NativeStyleModelError::InvalidCheckpoint(format!(
                "state {key} must be contiguous CPU storage on the construction stream"
            )));
        }
        if source_dtype
            .replace(descriptor.dtype())
            .is_some_and(|dtype| dtype != descriptor.dtype())
        {
            return Err(NativeStyleModelError::InvalidCheckpoint(
                "all source tensors must use one storage dtype".to_owned(),
            ));
        }
        if !storage_ids.insert(tensor.storage_id()) {
            return Err(NativeStyleModelError::InvalidCheckpoint(format!(
                "state {key} aliases another source tensor"
            )));
        }
        validate_finite_source_tensor(key, tensor, cancellation)?;
    }
    source_dtype
        .ok_or_else(|| NativeStyleModelError::InvalidCheckpoint("state is empty".to_owned()))
}

fn validate_finite_source_tensor(
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeStyleModelError> {
    let element_count = usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| NativeStyleModelError::ShapeOverflow)?;
    for index in 0..element_count {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        match tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(u64_from(index)?)?)?
        {
            DecodedScalar::Real(value) if value.is_finite() => {}
            _ => {
                return Err(NativeStyleModelError::InvalidCheckpoint(format!(
                    "state {key} contains a non-finite or non-real value"
                )));
            }
        }
    }
    Ok(())
}

fn validate_distinct_projection_storage(
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
) -> Result<(), NativeStyleModelError> {
    let source_storage_ids = source_state
        .values()
        .map(Tensor::storage_id)
        .collect::<HashSet<_>>();
    let mut execution_storage_ids = HashSet::new();
    for tensor in execution_state.values() {
        if source_storage_ids.contains(&tensor.storage_id())
            || !execution_storage_ids.insert(tensor.storage_id())
        {
            return Err(NativeStyleModelError::SemanticStateChanged);
        }
    }
    Ok(())
}

fn build_style_attention(
    backend: &CpuBackend,
    configuration: NativeStyleModelConfiguration,
    execution_state: &BTreeMap<String, Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Box<[NativeStyleAttentionProjection]>, NativeStyleModelError> {
    if configuration.architecture != NativeStyleModelArchitecture::StyleAdapter {
        return Ok(Vec::new().into_boxed_slice());
    }
    let width = u64_from(configuration.input_width)?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(configuration.layers)
        .map_err(|_| NativeStyleModelError::Allocation)?;
    for layer in 0..configuration.layers {
        context.check()?;
        let prefix = format!("transformer_layes.{layer}.attn");
        let weight = execution_state
            .get(&format!("{prefix}.in_proj_weight"))
            .ok_or_else(|| {
                NativeStyleModelError::MissingState(format!("{prefix}.in_proj_weight"))
            })?;
        let bias = execution_state
            .get(&format!("{prefix}.in_proj_bias"))
            .ok_or_else(|| NativeStyleModelError::MissingState(format!("{prefix}.in_proj_bias")))?;
        let weight_parts = split_three(weight, width, context.cancellation)?;
        let bias_parts = split_three(bias, width, context.cancellation)?;
        require_exact_fused_reconstruction(backend, weight, &weight_parts, context)?;
        require_exact_fused_reconstruction(backend, bias, &bias_parts, context)?;
        layers.push(NativeStyleAttentionProjection {
            query_weight: weight_parts[0].clone(),
            key_weight: weight_parts[1].clone(),
            value_weight: weight_parts[2].clone(),
            query_bias: bias_parts[0].clone(),
            key_bias: bias_parts[1].clone(),
            value_bias: bias_parts[2].clone(),
        });
    }
    Ok(layers.into_boxed_slice())
}

fn split_three(
    tensor: &Tensor,
    width: u64,
    cancellation: &CancellationToken,
) -> Result<[Tensor; 3], NativeStyleModelError> {
    Ok([
        narrow_method_exact_native(tensor, 0, 0, width, cancellation).map_err(canonical)?,
        narrow_method_exact_native(
            tensor,
            0,
            i64::try_from(width).map_err(|_| NativeStyleModelError::ShapeOverflow)?,
            width,
            cancellation,
        )
        .map_err(canonical)?,
        narrow_method_exact_native(
            tensor,
            0,
            i64::try_from(
                width
                    .checked_mul(2)
                    .ok_or(NativeStyleModelError::ShapeOverflow)?,
            )
            .map_err(|_| NativeStyleModelError::ShapeOverflow)?,
            width,
            cancellation,
        )
        .map_err(canonical)?,
    ])
}

fn require_exact_fused_reconstruction(
    backend: &CpuBackend,
    fused: &Tensor,
    parts: &[Tensor; 3],
    context: &ExecutionContext<'_>,
) -> Result<(), NativeStyleModelError> {
    let reconstructed =
        torch_cat_with_context_exact_native(backend, parts, 0, context).map_err(canonical)?;
    if reconstructed.descriptor().shape() != fused.descriptor().shape()
        || reconstructed.contiguous_bytes()? != fused.contiguous_bytes()?
    {
        return Err(NativeStyleModelError::SemanticStateChanged);
    }
    Ok(())
}

fn validate_attention_views(
    resource: &NativeStyleModelResource,
    cancellation: &CancellationToken,
) -> Result<(), NativeStyleModelError> {
    if resource.configuration.architecture != NativeStyleModelArchitecture::StyleAdapter {
        if !resource.style_attention.is_empty() {
            return Err(NativeStyleModelError::SemanticStateChanged);
        }
        return Ok(());
    }
    if resource.style_attention.len() != resource.configuration.layers {
        return Err(NativeStyleModelError::SemanticStateChanged);
    }
    let width = u64_from(resource.configuration.input_width)?;
    for (layer, projection) in resource.style_attention.iter().enumerate() {
        cancellation.check()?;
        let prefix = format!("transformer_layes.{layer}.attn");
        let fused_weight = resource.execution_tensor(&format!("{prefix}.in_proj_weight"))?;
        let fused_bias = resource.execution_tensor(&format!("{prefix}.in_proj_bias"))?;
        for tensor in [
            &projection.query_weight,
            &projection.key_weight,
            &projection.value_weight,
        ] {
            if tensor.storage_id() != fused_weight.storage_id()
                || tensor.descriptor().shape() != [width, width]
            {
                return Err(NativeStyleModelError::SemanticStateChanged);
            }
        }
        for tensor in [
            &projection.query_bias,
            &projection.key_bias,
            &projection.value_bias,
        ] {
            if tensor.storage_id() != fused_bias.storage_id()
                || tensor.descriptor().shape() != [width]
            {
                return Err(NativeStyleModelError::SemanticStateChanged);
            }
        }
    }
    Ok(())
}

fn linear_tensor(
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &Tensor,
    bias: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeStyleModelError> {
    let weight_values =
        tensor_to_f32_with_context_exact_native(backend, weight, context).map_err(canonical)?;
    let bias_values = bias
        .map(|bias| tensor_to_f32_with_context_exact_native(backend, bias, context))
        .transpose()
        .map_err(canonical)?;
    let weight_shape = weight
        .descriptor()
        .shape()
        .iter()
        .map(|dimension| usize_from(*dimension))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(linear_with_context_exact_native(
        backend,
        input,
        input_shape,
        &weight_values,
        &weight_shape,
        bias_values.as_deref(),
        DeviceId::CPU,
        context,
    )
    .map_err(canonical)?
    .values)
}

fn quick_gelu(
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeStyleModelError> {
    let mut scaled = Vec::new();
    scaled
        .try_reserve_exact(input.len())
        .map_err(|_| NativeStyleModelError::Allocation)?;
    for (index, value) in input.iter().copied().enumerate() {
        if index.is_multiple_of(16_384) {
            context.check()?;
        }
        scaled.push(QUICK_GELU_SCALE * value);
    }
    let shape = shape
        .iter()
        .map(|dimension| u64_from(*dimension))
        .collect::<Result<Vec<_>, _>>()?;
    let scaled = tensor_from_f32_with_context_exact_native(
        backend,
        &shape,
        &scaled,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(canonical)?;
    let sigmoid =
        sigmoid_with_context_exact_native(backend, &scaled, context).map_err(canonical)?;
    let sigmoid =
        tensor_to_f32_with_context_exact_native(backend, &sigmoid, context).map_err(canonical)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| NativeStyleModelError::Allocation)?;
    for (index, (value, sigmoid)) in input.iter().zip(sigmoid).enumerate() {
        if index.is_multiple_of(16_384) {
            context.check()?;
        }
        output.push(*value * sigmoid);
    }
    Ok(output)
}

fn add_in_place(
    left: &mut [f32],
    right: &[f32],
    cancellation: &CancellationToken,
) -> Result<(), NativeStyleModelError> {
    if left.len() != right.len() {
        return Err(NativeStyleModelError::ShapeOverflow);
    }
    for (index, (left, right)) in left.iter_mut().zip(right).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        *left += *right;
    }
    cancellation.check()?;
    Ok(())
}

fn construction_memory_preflight(
    artifact: &str,
    artifact_capacity: usize,
    ordered_state: &[(String, Tensor)],
    specifications: &[StateSpecification],
    configuration: NativeStyleModelConfiguration,
) -> Result<u64, NativeStyleModelError> {
    let source_bytes = ordered_state.iter().try_fold(0_u64, |total, (_, tensor)| {
        total
            .checked_add(tensor.storage_byte_len())
            .ok_or(NativeStyleModelError::ShapeOverflow)
    })?;
    let projected_bytes = specifications
        .iter()
        .try_fold(0_u64, |total, specification| {
            total
                .checked_add(checked_tensor_bytes(&specification.shape, DType::F32)?)
                .ok_or(NativeStyleModelError::ShapeOverflow)
        })?;
    let maximum_projected = specifications
        .iter()
        .map(|specification| checked_tensor_bytes(&specification.shape, DType::F32))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let maximum_fused = if configuration.architecture == NativeStyleModelArchitecture::StyleAdapter
    {
        u64_from(configuration.input_width)?
            .checked_mul(u64_from(configuration.input_width)?)
            .and_then(|value| value.checked_mul(3))
            .and_then(|value| value.checked_mul(4))
            .ok_or(NativeStyleModelError::ShapeOverflow)?
    } else {
        0
    };
    let owned = construction_owned_estimate(
        artifact,
        artifact_capacity,
        ordered_state,
        specifications,
        configuration,
    )?;
    source_bytes
        .checked_add(projected_bytes)
        .and_then(|value| value.checked_add(maximum_projected.max(maximum_fused)))
        .and_then(|value| value.checked_add(owned))
        .ok_or(NativeStyleModelError::ShapeOverflow)
}

fn construction_owned_estimate(
    artifact: &str,
    artifact_capacity: usize,
    ordered_state: &[(String, Tensor)],
    specifications: &[StateSpecification],
    configuration: NativeStyleModelConfiguration,
) -> Result<u64, NativeStyleModelError> {
    let base = u64::try_from(mem::size_of::<NativeStyleModelResource>())
        .map_err(|_| NativeStyleModelError::ShapeOverflow)?;
    let source_key_bytes = ordered_state.iter().try_fold(0_u64, |total, (key, _)| {
        total
            .checked_add(u64_from(key.capacity())?)
            .ok_or(NativeStyleModelError::ShapeOverflow)
    })?;
    let projected_key_bytes = specifications
        .iter()
        .try_fold(0_u64, |total, specification| {
            total
                .checked_add(u64_from(specification.key.capacity())?)
                .and_then(|value| {
                    value.checked_add(
                        u64_from(specification.shape.capacity())
                            .ok()?
                            .checked_mul(8)?,
                    )
                })
                .ok_or(NativeStyleModelError::ShapeOverflow)
        })?;
    let temporary_container_bytes = u64_from(ordered_state.len())?
        .checked_mul(
            u64::try_from(mem::size_of::<(String, Tensor)>())
                .map_err(|_| NativeStyleModelError::ShapeOverflow)?,
        )
        .and_then(|value| {
            value.checked_add(
                u64_from(specifications.len())
                    .ok()?
                    .checked_mul(u64::try_from(mem::size_of::<StateSpecification>()).ok()?)?,
            )
        })
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    base.checked_add(u64_from(artifact_capacity.max(artifact.len()))?)
        .and_then(|value| value.checked_add(source_key_bytes))
        .and_then(|value| value.checked_add(projected_key_bytes))
        .and_then(|value| value.checked_add(temporary_container_bytes))
        .and_then(|value| value.checked_add(map_node_estimate(specifications.len()).ok()?))
        .ok_or(NativeStyleModelError::ShapeOverflow)?
        .checked_add(
            u64_from(configuration.layers)?
                .checked_mul(
                    u64::try_from(mem::size_of::<NativeStyleAttentionProjection>())
                        .map_err(|_| NativeStyleModelError::ShapeOverflow)?,
                )
                .ok_or(NativeStyleModelError::ShapeOverflow)?,
        )
        .ok_or(NativeStyleModelError::ShapeOverflow)
}

fn preflight_invocation_memory(
    resource: &NativeStyleModelResource,
    batch: usize,
    tokens: usize,
) -> Result<(), NativeStyleModelError> {
    let scalar_bytes = 4_u64;
    let input_elements = u64_from(batch)?
        .checked_mul(u64_from(tokens)?)
        .and_then(|value| value.checked_mul(u64_from(resource.configuration.input_width).ok()?))
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    let input_bytes = input_elements
        .checked_mul(scalar_bytes)
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    let phase_bytes = match resource.configuration.architecture {
        NativeStyleModelArchitecture::StyleAdapter => {
            let sequence = tokens
                .checked_add(resource.configuration.output_tokens)
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
            let hidden_elements = u64_from(batch)?
                .checked_mul(u64_from(sequence)?)
                .and_then(|value| {
                    value.checked_mul(u64_from(resource.configuration.input_width).ok()?)
                })
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
            let hidden_bytes = hidden_elements
                .checked_mul(scalar_bytes)
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
            let score_bytes = u64_from(batch)?
                .checked_mul(u64_from(resource.configuration.heads)?)
                .and_then(|value| value.checked_mul(u64_from(sequence).ok()?))
                .and_then(|value| value.checked_mul(u64_from(sequence).ok()?))
                .and_then(|value| value.checked_mul(scalar_bytes))
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
            hidden_bytes
                .checked_mul(24)
                .and_then(|value| value.checked_add(score_bytes.checked_mul(2)?))
                .and_then(|value| value.checked_add(input_bytes))
                .ok_or(NativeStyleModelError::ShapeOverflow)?
        }
        NativeStyleModelArchitecture::FluxRedux => {
            let hidden_bytes = u64_from(batch)?
                .checked_mul(u64_from(tokens)?)
                .and_then(|value| {
                    value.checked_mul(u64_from(resource.configuration.hidden_width).ok()?)
                })
                .and_then(|value| value.checked_mul(scalar_bytes))
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
            let output_bytes = u64_from(batch)?
                .checked_mul(u64_from(tokens)?)
                .and_then(|value| {
                    value.checked_mul(u64_from(resource.configuration.output_width).ok()?)
                })
                .and_then(|value| value.checked_mul(scalar_bytes))
                .ok_or(NativeStyleModelError::ShapeOverflow)?;
            input_bytes
                .checked_mul(2)
                .and_then(|value| value.checked_add(hidden_bytes.checked_mul(3)?))
                .and_then(|value| value.checked_add(output_bytes))
                .ok_or(NativeStyleModelError::ShapeOverflow)?
        }
    };
    let decoded_weight_bytes = resource
        .execution_state
        .values()
        .map(Tensor::storage_byte_len)
        .max()
        .unwrap_or(0)
        .checked_mul(2)
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    let required = resource
        .resident_bytes
        .checked_add(phase_bytes)
        .and_then(|value| value.checked_add(decoded_weight_bytes))
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    if required > resource.memory_budget_bytes {
        return Err(NativeStyleModelError::OutOfMemory {
            required,
            budget: resource.memory_budget_bytes,
        });
    }
    Ok(())
}

fn semantic_digest(
    configuration: NativeStyleModelConfiguration,
    artifact: &str,
    source_dtype: DType,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
    specifications: &[StateSpecification],
    cancellation: &CancellationToken,
) -> Result<String, NativeStyleModelError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.style-model-resource.v1\0");
    hasher.update([match configuration.architecture {
        NativeStyleModelArchitecture::StyleAdapter => 1,
        NativeStyleModelArchitecture::FluxRedux => 2,
    }]);
    for value in [
        configuration.input_width,
        configuration.hidden_width,
        configuration.output_width,
        configuration.heads,
        configuration.layers,
        configuration.output_tokens,
    ] {
        hasher.update(u64_from(value)?.to_le_bytes());
    }
    hasher.update([u8::from(configuration.source_exact)]);
    hasher.update(artifact.as_bytes());
    hasher.update(source_dtype.catalog_name().as_bytes());
    for (index, specification) in specifications.iter().enumerate() {
        if index.is_multiple_of(8) {
            cancellation.check()?;
        }
        hash_tensor(
            &mut hasher,
            &specification.key,
            source_state
                .get(&specification.key)
                .ok_or_else(|| NativeStyleModelError::MissingState(specification.key.clone()))?,
            cancellation,
        )?;
        hash_tensor(
            &mut hasher,
            &specification.key,
            execution_state
                .get(&specification.key)
                .ok_or_else(|| NativeStyleModelError::MissingState(specification.key.clone()))?,
            cancellation,
        )?;
    }
    cancellation.check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_tensor(
    hasher: &mut Sha256,
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeStyleModelError> {
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
) -> Result<(), NativeStyleModelError> {
    let bytes = tensor.contiguous_bytes()?;
    if !bytes.len().is_multiple_of(mem::size_of::<f32>()) {
        return Err(NativeStyleModelError::InvalidCheckpoint(format!(
            "state {key} has invalid F32 storage"
        )));
    }
    for (index, chunk) in bytes.chunks_exact(mem::size_of::<f32>()).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        let encoded: [u8; 4] = chunk
            .try_into()
            .map_err(|_| NativeStyleModelError::SemanticStateChanged)?;
        if !f32::from_ne_bytes(encoded).is_finite() {
            return Err(NativeStyleModelError::InvalidCheckpoint(format!(
                "state {key} contains a non-finite value"
            )));
        }
    }
    Ok(())
}

fn resident_tensor_bytes<'a>(
    maps: impl IntoIterator<Item = &'a BTreeMap<String, Tensor>>,
    cancellation: &CancellationToken,
) -> Result<u64, NativeStyleModelError> {
    resident_tensor_allocations(maps, cancellation)?
        .into_iter()
        .try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(bytes)
                .ok_or(NativeStyleModelError::ShapeOverflow)
        })
}

fn resident_tensor_allocations<'a>(
    maps: impl IntoIterator<Item = &'a BTreeMap<String, Tensor>>,
    cancellation: &CancellationToken,
) -> Result<Vec<(StorageId, u64)>, NativeStyleModelError> {
    let mut allocations = Vec::new();
    for map in maps {
        for (index, tensor) in map.values().enumerate() {
            if index.is_multiple_of(16) {
                cancellation.check()?;
            }
            let storage_id = tensor.storage_id();
            let bytes = tensor.storage_byte_len();
            if let Some((_, existing)) = allocations
                .iter()
                .find(|(existing_id, _)| *existing_id == storage_id)
            {
                if *existing != bytes {
                    return Err(NativeStyleModelError::SemanticStateChanged);
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
    artifact_capacity: usize,
    digest: &str,
    digest_capacity: usize,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
    attention_layers: usize,
) -> Result<u64, NativeStyleModelError> {
    let base = u64::try_from(mem::size_of::<NativeStyleModelResource>())
        .map_err(|_| NativeStyleModelError::ShapeOverflow)?;
    let keys =
        source_state
            .keys()
            .chain(execution_state.keys())
            .try_fold(0_u64, |total, key| {
                total
                    .checked_add(u64_from(key.capacity())?)
                    .ok_or(NativeStyleModelError::ShapeOverflow)
            })?;
    base.checked_add(u64_from(artifact_capacity.max(artifact.len()))?)
        .and_then(|value| value.checked_add(u64_from(digest_capacity.max(digest.len())).ok()?))
        .and_then(|value| value.checked_add(keys))
        .and_then(|value| value.checked_add(map_node_estimate(source_state.len()).ok()?))
        .and_then(|value| {
            value.checked_add(u64_from(attention_layers).ok()?.checked_mul(
                u64::try_from(mem::size_of::<NativeStyleAttentionProjection>()).ok()?,
            )?)
        })
        .ok_or(NativeStyleModelError::ShapeOverflow)
}

fn map_node_estimate(entries: usize) -> Result<u64, NativeStyleModelError> {
    let per_entry = u64::try_from(mem::size_of::<(String, Tensor)>())
        .map_err(|_| NativeStyleModelError::ShapeOverflow)?
        .checked_add(64)
        .ok_or(NativeStyleModelError::ShapeOverflow)?;
    u64_from(entries)?
        .checked_mul(per_entry)
        .and_then(|value| value.checked_mul(2))
        .ok_or(NativeStyleModelError::ShapeOverflow)
}

fn checked_tensor_bytes(shape: &[u64], dtype: DType) -> Result<u64, NativeStyleModelError> {
    shape
        .iter()
        .try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(NativeStyleModelError::ShapeOverflow)
        })?
        .checked_mul(dtype.byte_width())
        .ok_or(NativeStyleModelError::ShapeOverflow)
}

fn validate_sha256(value: &str) -> Result<(), NativeStyleModelError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NativeStyleModelError::InvalidCheckpoint(
            "artifact identity must be canonical lowercase SHA-256".to_owned(),
        ));
    }
    Ok(())
}

fn attention_error(error: AttentionError) -> NativeStyleModelError {
    match error {
        AttentionError::Cancelled | AttentionError::Tensor(TensorError::Cancelled) => {
            NativeStyleModelError::Cancelled
        }
        error => canonical(error),
    }
}

fn canonical(error: impl std::error::Error + 'static) -> NativeStyleModelError {
    let mut source: &(dyn std::error::Error + 'static) = &error;
    loop {
        if is_cancellation_error(source) {
            return NativeStyleModelError::Cancelled;
        }
        let Some(next) = source.source() else {
            break;
        };
        source = next;
    }
    NativeStyleModelError::Canonical(error.to_string())
}

fn is_cancellation_error(error: &(dyn std::error::Error + 'static)) -> bool {
    error
        .downcast_ref::<TensorError>()
        .is_some_and(|error| matches!(error, TensorError::Cancelled))
        || error
            .downcast_ref::<sim_types::CancellationError>()
            .is_some()
        || error
            .downcast_ref::<OperatorIndirectionError>()
            .is_some_and(|error| matches!(error, OperatorIndirectionError::Cancelled))
        || error
            .downcast_ref::<ElementwiseRuntimePartThreeError>()
            .is_some_and(|error| matches!(error, ElementwiseRuntimePartThreeError::Cancelled))
        || error
            .downcast_ref::<ElementwiseRuntimePartNineError>()
            .is_some_and(|error| matches!(error, ElementwiseRuntimePartNineError::Cancelled))
        || error
            .downcast_ref::<IndexingMaskingPartOneError>()
            .is_some_and(|error| matches!(error, IndexingMaskingPartOneError::Cancelled))
        || error
            .downcast_ref::<LinearAlgebraPartOneError>()
            .is_some_and(|error| matches!(error, LinearAlgebraPartOneError::Cancelled))
        || error
            .downcast_ref::<NeuralNetworkFunctionalError>()
            .is_some_and(|error| matches!(error, NeuralNetworkFunctionalError::Cancelled))
        || error
            .downcast_ref::<ShapeLayoutTransformPartTwoError>()
            .is_some_and(|error| matches!(error, ShapeLayoutTransformPartTwoError::Cancelled))
        || error
            .downcast_ref::<FunctionalError>()
            .is_some_and(|error| matches!(error, FunctionalError::Cancelled))
        || error
            .downcast_ref::<AttentionError>()
            .is_some_and(|error| matches!(error, AttentionError::Cancelled))
}

fn u64_from(value: usize) -> Result<u64, NativeStyleModelError> {
    u64::try_from(value).map_err(|_| NativeStyleModelError::ShapeOverflow)
}

fn usize_from(value: u64) -> Result<usize, NativeStyleModelError> {
    usize::try_from(value).map_err(|_| NativeStyleModelError::ShapeOverflow)
}

fn i64_from(value: usize) -> Result<i64, NativeStyleModelError> {
    i64::try_from(value).map_err(|_| NativeStyleModelError::ShapeOverflow)
}

#[derive(Clone, Debug)]
pub enum NativePhotoMakerCheckpointEntry {
    Tensor {
        key: String,
        tensor: Tensor,
    },
    Mapping {
        key: String,
        ordered_state: Vec<(String, Tensor)>,
    },
}

#[derive(Clone, Debug)]
pub struct NativePhotoMakerCheckpoint {
    pub artifact_sha256: String,
    pub ordered_entries: Vec<NativePhotoMakerCheckpointEntry>,
    pub memory_budget_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativePhotoMakerWrapper {
    Flat,
    IdEncoder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativePhotoMakerConfiguration {
    hidden: usize,
    intermediate: usize,
    heads: usize,
    layers: usize,
    image_size: usize,
    patch_size: usize,
    projection: usize,
    extra_projection: usize,
    prompt_width: usize,
    source_exact: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativePhotoMakerPhase {
    Entry,
    Wrapper,
    Schema,
    SourceValidation,
    Projection,
    ClipVision,
    SemanticDigest,
    Validation,
    Return,
    InvocationAdmission,
    ClipForward,
    ExtraProjection,
    MaskIndices,
    Gather,
    Mlp1LayerNorm,
    Mlp1Linear1,
    Mlp1Gelu,
    Mlp1Linear2,
    PromptResidual,
    Mlp2LayerNorm,
    Mlp2Linear1,
    Mlp2Gelu,
    Mlp2Linear2,
    Mlp2Residual,
    FinalLayerNorm,
    Scatter,
    InvocationReturn,
}

#[derive(Debug)]
pub struct NativePhotoMakerResource {
    configuration: NativePhotoMakerConfiguration,
    wrapper: NativePhotoMakerWrapper,
    artifact_sha256: String,
    source_state: BTreeMap<String, Tensor>,
    execution_state: BTreeMap<String, Tensor>,
    clip_vision: NativeClipVision,
    source_dtype: DType,
    stream: StreamId,
    memory_budget_bytes: u64,
    resident_bytes: u64,
    semantic_digest_sha256: String,
}

#[derive(Debug, Error)]
pub enum NativePhotoMakerError {
    #[error("PhotoMaker execution was cancelled")]
    Cancelled,
    #[error("PhotoMaker checkpoint is invalid: {0}")]
    InvalidCheckpoint(String),
    #[error("PhotoMaker state is missing key {0}")]
    MissingState(String),
    #[error("PhotoMaker state is unexpected: {0}")]
    UnexpectedState(String),
    #[error("PhotoMaker state {key} expected {expected:?}, got {actual:?} {actual_dtype:?}")]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        actual_dtype: DType,
    },
    #[error("PhotoMaker input is invalid: {0}")]
    InvalidInput(String),
    #[error("PhotoMaker retained semantic state changed")]
    SemanticStateChanged,
    #[error("PhotoMaker shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("PhotoMaker allocation failed")]
    Allocation,
    #[error("PhotoMaker memory requirement {required} exceeds budget {budget}")]
    OutOfMemory { required: u64, budget: u64 },
    #[error("PhotoMaker canonical owner failed: {0}")]
    Canonical(String),
    #[error(transparent)]
    ClipVision(ClipVisionError),
    #[error(transparent)]
    Tensor(TensorError),
}

impl From<sim_types::CancellationError> for NativePhotoMakerError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<TensorError> for NativePhotoMakerError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled,
            error => Self::Tensor(error),
        }
    }
}

impl From<ClipVisionError> for NativePhotoMakerError {
    fn from(error: ClipVisionError) -> Self {
        match error {
            ClipVisionError::Cancelled | ClipVisionError::Tensor(TensorError::Cancelled) => {
                Self::Cancelled
            }
            error => Self::ClipVision(error),
        }
    }
}

impl NativePhotoMakerResource {
    pub fn from_checkpoint(
        backend: &CpuBackend,
        checkpoint: NativePhotoMakerCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativePhotoMakerError> {
        Self::checked(
            backend,
            checkpoint,
            photo_source_configuration(),
            context,
            &mut |_, _| {},
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn from_reduced_fixture(
        backend: &CpuBackend,
        checkpoint: NativePhotoMakerCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativePhotoMakerError> {
        Self::checked(
            backend,
            checkpoint,
            photo_reduced_configuration(),
            context,
            &mut |_, _| {},
        )
    }

    fn checked(
        backend: &CpuBackend,
        checkpoint: NativePhotoMakerCheckpoint,
        configuration: NativePhotoMakerConfiguration,
        context: &ExecutionContext<'_>,
        phase_hook: &mut impl FnMut(NativePhotoMakerPhase, &CancellationToken),
    ) -> Result<Self, NativePhotoMakerError> {
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Entry,
            phase_hook,
        )?;
        photo_validate_sha256(&checkpoint.artifact_sha256)?;
        if checkpoint.memory_budget_bytes == 0 || checkpoint.ordered_entries.is_empty() {
            return Err(NativePhotoMakerError::InvalidCheckpoint(
                "state cardinality or memory budget is invalid".to_owned(),
            ));
        }
        photo_raw_checkpoint_preflight(&checkpoint)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Wrapper,
            phase_hook,
        )?;
        let (wrapper, ordered_state) = photo_normalize_checkpoint(checkpoint.ordered_entries)?;
        let specifications = photo_state_manifest(configuration)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Schema,
            phase_hook,
        )?;
        photo_validate_ordered_keys(&ordered_state, &specifications)?;
        let source_dtype = photo_validate_source_state(
            &ordered_state,
            &specifications,
            context.stream,
            context.cancellation,
        )?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::SourceValidation,
            phase_hook,
        )?;
        let construction_peak = photo_construction_peak(
            &checkpoint.artifact_sha256,
            checkpoint.artifact_sha256.capacity(),
            &ordered_state,
            &specifications,
        )?;
        if construction_peak > checkpoint.memory_budget_bytes {
            return Err(NativePhotoMakerError::OutOfMemory {
                required: construction_peak,
                budget: checkpoint.memory_budget_bytes,
            });
        }

        let source_state = ordered_state.into_iter().collect::<BTreeMap<_, _>>();
        let mut projected_state = BTreeMap::new();
        for (index, specification) in specifications.iter().enumerate() {
            if index.is_multiple_of(8) {
                photo_checkpoint(
                    context.cancellation,
                    NativePhotoMakerPhase::Projection,
                    phase_hook,
                )?;
            }
            let source = source_state
                .get(&specification.key)
                .ok_or_else(|| NativePhotoMakerError::MissingState(specification.key.clone()))?;
            let projected = cast_to_with_context_exact_native(
                backend,
                source,
                DType::F32,
                DeviceId::CPU,
                false,
                true,
                context,
            )
            .map_err(photo_canonical)?;
            photo_validate_finite_tensor(&specification.key, &projected, context.cancellation)?;
            projected_state.insert(specification.key.clone(), projected);
        }
        photo_validate_distinct_storage(&source_state, &projected_state)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::ClipVision,
            phase_hook,
        )?;
        let clip_vision = NativeClipVision::new_with_cancellation(
            photo_clip_configuration(configuration),
            photo_clip_weights(&projected_state, configuration)?,
            context.cancellation,
        )?;
        photo_require_clip_storage_from_execution(&clip_vision, &projected_state)?;
        let mut execution_state = BTreeMap::new();
        for specification in specifications.iter().skip(PHOTOMAKER_CLIP_STATE_COUNT) {
            execution_state.insert(
                specification.key.clone(),
                projected_state.remove(&specification.key).ok_or_else(|| {
                    NativePhotoMakerError::MissingState(specification.key.clone())
                })?,
            );
        }
        photo_validate_resource_storage(&source_state, &execution_state, &clip_vision)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::SemanticDigest,
            phase_hook,
        )?;
        let semantic_digest_sha256 = photo_semantic_digest(
            configuration,
            wrapper,
            &checkpoint.artifact_sha256,
            source_dtype,
            &source_state,
            &execution_state,
            clip_vision.semantic_digest_sha256(),
            &specifications,
            context.cancellation,
        )?;
        let owned_bytes = photo_resident_owned_bytes(
            &checkpoint.artifact_sha256,
            checkpoint.artifact_sha256.capacity(),
            &semantic_digest_sha256,
            semantic_digest_sha256.capacity(),
            &source_state,
            &execution_state,
            &clip_vision,
        )?;
        let resident_bytes = photo_resource_tensor_allocations(
            &source_state,
            &execution_state,
            &clip_vision,
            context.cancellation,
        )?
        .into_iter()
        .try_fold(owned_bytes, |total, (_, bytes)| {
            total
                .checked_add(bytes)
                .ok_or(NativePhotoMakerError::ShapeOverflow)
        })?;
        if resident_bytes > checkpoint.memory_budget_bytes {
            return Err(NativePhotoMakerError::OutOfMemory {
                required: resident_bytes,
                budget: checkpoint.memory_budget_bytes,
            });
        }
        let resource = Self {
            configuration,
            wrapper,
            artifact_sha256: checkpoint.artifact_sha256,
            source_state,
            execution_state,
            clip_vision,
            source_dtype,
            stream: context.stream,
            memory_budget_bytes: checkpoint.memory_budget_bytes,
            resident_bytes,
            semantic_digest_sha256,
        };
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Validation,
            phase_hook,
        )?;
        resource.validate(context.cancellation)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Return,
            phase_hook,
        )?;
        Ok(resource)
    }

    pub const fn identifier(&self) -> &'static str {
        "photomaker-id-encoder"
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

    pub fn resident_owned_bytes(&self) -> Result<u64, NativePhotoMakerError> {
        photo_resident_owned_bytes(
            &self.artifact_sha256,
            self.artifact_sha256.capacity(),
            &self.semantic_digest_sha256,
            self.semantic_digest_sha256.capacity(),
            &self.source_state,
            &self.execution_state,
            &self.clip_vision,
        )
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(StorageId, u64)>, NativePhotoMakerError> {
        photo_resource_tensor_allocations(
            &self.source_state,
            &self.execution_state,
            &self.clip_vision,
            &CancellationToken::default(),
        )
    }

    pub fn reconstruct_checkpoint(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NativePhotoMakerCheckpoint, NativePhotoMakerError> {
        self.validate(cancellation)?;
        self.clip_vision.reconstruct(cancellation)?;
        let specifications = photo_state_manifest(self.configuration)?;
        let mut ordered_state = Vec::new();
        ordered_state
            .try_reserve_exact(specifications.len())
            .map_err(|_| NativePhotoMakerError::Allocation)?;
        for (index, specification) in specifications.iter().enumerate() {
            if index.is_multiple_of(8) {
                cancellation.check()?;
            }
            ordered_state.push((
                specification.key.clone(),
                self.source_state
                    .get(&specification.key)
                    .ok_or_else(|| NativePhotoMakerError::MissingState(specification.key.clone()))?
                    .clone(),
            ));
        }
        let ordered_entries = match self.wrapper {
            NativePhotoMakerWrapper::Flat => ordered_state
                .into_iter()
                .map(|(key, tensor)| NativePhotoMakerCheckpointEntry::Tensor { key, tensor })
                .collect(),
            NativePhotoMakerWrapper::IdEncoder => {
                vec![NativePhotoMakerCheckpointEntry::Mapping {
                    key: "id_encoder".to_owned(),
                    ordered_state,
                }]
            }
        };
        cancellation.check()?;
        Ok(NativePhotoMakerCheckpoint {
            artifact_sha256: self.artifact_sha256.clone(),
            ordered_entries,
            memory_budget_bytes: self.memory_budget_bytes,
        })
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), NativePhotoMakerError> {
        cancellation.check()?;
        let specifications = photo_state_manifest(self.configuration)?;
        let source_order = specifications
            .iter()
            .map(|specification| {
                self.source_state
                    .get(&specification.key)
                    .cloned()
                    .map(|tensor| (specification.key.clone(), tensor))
                    .ok_or_else(|| NativePhotoMakerError::MissingState(specification.key.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_dtype =
            photo_validate_source_state(&source_order, &specifications, self.stream, cancellation)?;
        if source_dtype != self.source_dtype
            || self.source_state.len() != PHOTOMAKER_STATE_COUNT
            || self.execution_state.len() != PHOTOMAKER_STATE_COUNT - PHOTOMAKER_CLIP_STATE_COUNT
        {
            return Err(NativePhotoMakerError::SemanticStateChanged);
        }
        for specification in specifications.iter().skip(PHOTOMAKER_CLIP_STATE_COUNT) {
            let tensor = self
                .execution_state
                .get(&specification.key)
                .ok_or_else(|| NativePhotoMakerError::MissingState(specification.key.clone()))?;
            if tensor.descriptor().shape() != specification.shape
                || tensor.descriptor().dtype() != DType::F32
                || tensor.descriptor().device() != DeviceId::CPU
                || tensor.descriptor().stream() != self.stream
                || !tensor.descriptor().is_contiguous()?
            {
                return Err(NativePhotoMakerError::SemanticStateChanged);
            }
            photo_validate_finite_tensor(&specification.key, tensor, cancellation)?;
        }
        photo_validate_resource_storage(
            &self.source_state,
            &self.execution_state,
            &self.clip_vision,
        )?;
        self.clip_vision.validate(cancellation)?;
        let reconstructed_clip = self.clip_vision.reconstruct(cancellation)?;
        if reconstructed_clip.semantic_identity() != self.clip_vision.semantic_identity()
            || reconstructed_clip.resident_parts()? != self.clip_vision.resident_parts()?
        {
            return Err(NativePhotoMakerError::SemanticStateChanged);
        }
        let digest = photo_semantic_digest(
            self.configuration,
            self.wrapper,
            &self.artifact_sha256,
            self.source_dtype,
            &self.source_state,
            &self.execution_state,
            self.clip_vision.semantic_digest_sha256(),
            &specifications,
            cancellation,
        )?;
        let resident = self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(NativePhotoMakerError::ShapeOverflow)
            },
        )?;
        if digest != self.semantic_digest_sha256 || resident != self.resident_bytes {
            return Err(NativePhotoMakerError::SemanticStateChanged);
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn fuse_conditioning(
        &self,
        backend: &CpuBackend,
        image: &Tensor,
        prompt: &Tensor,
        class_tokens_mask: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativePhotoMakerError> {
        self.fuse_conditioning_with_phase_hook(
            backend,
            image,
            prompt,
            class_tokens_mask,
            context,
            &mut |_, _| {},
        )
    }

    fn fuse_conditioning_with_phase_hook(
        &self,
        backend: &CpuBackend,
        image: &Tensor,
        prompt: &Tensor,
        class_tokens_mask: &Tensor,
        context: &ExecutionContext<'_>,
        phase_hook: &mut impl FnMut(NativePhotoMakerPhase, &CancellationToken),
    ) -> Result<Tensor, NativePhotoMakerError> {
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::InvocationAdmission,
            phase_hook,
        )?;
        self.validate(context.cancellation)?;
        photo_validate_invocation(
            self.configuration,
            image,
            prompt,
            class_tokens_mask,
            context,
        )?;
        let image_shape = image.descriptor().shape();
        let prompt_shape = prompt.descriptor().shape();
        let image_count = pm_usize(image_shape[1])?;
        let sequence = pm_usize(prompt_shape[1])?;
        photo_invocation_peak(self, image_count, sequence)?;
        let selected = photo_mask_true_count(backend, class_tokens_mask, context)?;
        if selected != image_count {
            return Err(NativePhotoMakerError::InvalidInput(format!(
                "mask contains {selected} true entries for {image_count} images"
            )));
        }

        let flattened_image = torch_reshape_with_context_exact_native(
            backend,
            image,
            &[
                pm_i64(image_shape[1])?,
                3,
                pm_i64(image_shape[3])?,
                pm_i64(image_shape[4])?,
            ],
            context,
        )
        .map_err(photo_canonical)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::ClipForward,
            phase_hook,
        )?;
        let mut clip_vision = self.clip_vision.clone();
        let checked = clip_vision.forward_checked(
            backend,
            &flattened_image,
            ClipVisionIntermediate::None,
            context,
        )?;
        let first_projection = checked.output.image_embeds;
        let pooled_hidden = checked.pooled_hidden;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::ExtraProjection,
            phase_hook,
        )?;
        let pooled_values =
            tensor_to_f32_with_context_exact_native(backend, &pooled_hidden, context)
                .map_err(photo_canonical)?;
        let second_projection = photo_linear(
            backend,
            &pooled_values,
            &[image_count, self.configuration.hidden],
            self.photo_tensor("visual_projection_2.weight")?,
            None,
            context,
        )?;
        let second_projection = tensor_from_f32_with_context_exact_native(
            backend,
            &[
                pm_u64(image_count)?,
                pm_u64(self.configuration.extra_projection)?,
            ],
            &second_projection,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(photo_canonical)?;
        let identity_embeddings = torch_cat_with_context_exact_native(
            backend,
            &[first_projection, second_projection],
            1,
            context,
        )
        .map_err(photo_canonical)?;

        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::MaskIndices,
            phase_hook,
        )?;
        let positions = photo_mask_positions(backend, class_tokens_mask, context)?;
        if positions.len() != image_count {
            return Err(NativePhotoMakerError::InvalidInput(format!(
                "mask contains {} true entries for {image_count} images",
                positions.len()
            )));
        }
        let index = photo_row_index_tensor(
            backend,
            &positions,
            self.configuration.prompt_width,
            context,
        )?;
        let flattened_prompt = torch_reshape_with_context_exact_native(
            backend,
            prompt,
            &[pm_i64(sequence)?, pm_i64(self.configuration.prompt_width)?],
            context,
        )
        .map_err(photo_canonical)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Gather,
            phase_hook,
        )?;
        let prompt_rows =
            gather_method_with_context_exact_native(backend, &flattened_prompt, 0, &index, context)
                .map_err(photo_canonical)?;
        let identity_values =
            tensor_to_f32_with_context_exact_native(backend, &identity_embeddings, context)
                .map_err(photo_canonical)?;
        let prompt_values = tensor_to_f32_with_context_exact_native(backend, &prompt_rows, context)
            .map_err(photo_canonical)?;
        let mut stacked = Vec::new();
        stacked
            .try_reserve_exact(
                image_count
                    .checked_mul(self.configuration.prompt_width)
                    .and_then(|value| value.checked_mul(2))
                    .ok_or(NativePhotoMakerError::ShapeOverflow)?,
            )
            .map_err(|_| NativePhotoMakerError::Allocation)?;
        for row in 0..image_count {
            context.check()?;
            let start = row
                .checked_mul(self.configuration.prompt_width)
                .ok_or(NativePhotoMakerError::ShapeOverflow)?;
            let end = start
                .checked_add(self.configuration.prompt_width)
                .ok_or(NativePhotoMakerError::ShapeOverflow)?;
            stacked.extend_from_slice(
                prompt_values
                    .get(start..end)
                    .ok_or(NativePhotoMakerError::ShapeOverflow)?,
            );
            stacked.extend_from_slice(
                identity_values
                    .get(start..end)
                    .ok_or(NativePhotoMakerError::ShapeOverflow)?,
            );
        }

        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp1LayerNorm,
            phase_hook,
        )?;
        let stacked_width = self
            .configuration
            .prompt_width
            .checked_mul(2)
            .ok_or(NativePhotoMakerError::ShapeOverflow)?;
        let normalized = self.photo_layer_norm(
            backend,
            &stacked,
            &[image_count, stacked_width],
            "fuse_module.mlp1.layernorm.weight",
            "fuse_module.mlp1.layernorm.bias",
            context,
        )?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp1Linear1,
            phase_hook,
        )?;
        let expanded = self.photo_linear_state(
            backend,
            &normalized,
            &[image_count, stacked_width],
            "fuse_module.mlp1.fc1.weight",
            Some("fuse_module.mlp1.fc1.bias"),
            context,
        )?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp1Gelu,
            phase_hook,
        )?;
        let activated = gelu_with_context_exact_native(
            backend,
            &expanded,
            GeluApproximation::None,
            DeviceId::CPU,
            context,
        )
        .map_err(photo_canonical)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp1Linear2,
            phase_hook,
        )?;
        let mut fused = self.photo_linear_state(
            backend,
            &activated,
            &[image_count, self.configuration.prompt_width],
            "fuse_module.mlp1.fc2.weight",
            Some("fuse_module.mlp1.fc2.bias"),
            context,
        )?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::PromptResidual,
            phase_hook,
        )?;
        photo_add_in_place(&mut fused, &prompt_values, context.cancellation)?;

        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp2LayerNorm,
            phase_hook,
        )?;
        let normalized = self.photo_layer_norm(
            backend,
            &fused,
            &[image_count, self.configuration.prompt_width],
            "fuse_module.mlp2.layernorm.weight",
            "fuse_module.mlp2.layernorm.bias",
            context,
        )?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp2Linear1,
            phase_hook,
        )?;
        let expanded = self.photo_linear_state(
            backend,
            &normalized,
            &[image_count, self.configuration.prompt_width],
            "fuse_module.mlp2.fc1.weight",
            Some("fuse_module.mlp2.fc1.bias"),
            context,
        )?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp2Gelu,
            phase_hook,
        )?;
        let activated = gelu_with_context_exact_native(
            backend,
            &expanded,
            GeluApproximation::None,
            DeviceId::CPU,
            context,
        )
        .map_err(photo_canonical)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp2Linear2,
            phase_hook,
        )?;
        let residual = self.photo_linear_state(
            backend,
            &activated,
            &[image_count, self.configuration.prompt_width],
            "fuse_module.mlp2.fc2.weight",
            Some("fuse_module.mlp2.fc2.bias"),
            context,
        )?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Mlp2Residual,
            phase_hook,
        )?;
        photo_add_in_place(&mut fused, &residual, context.cancellation)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::FinalLayerNorm,
            phase_hook,
        )?;
        let fused = self.photo_layer_norm(
            backend,
            &fused,
            &[image_count, self.configuration.prompt_width],
            "fuse_module.layer_norm.weight",
            "fuse_module.layer_norm.bias",
            context,
        )?;
        let fused = tensor_from_f32_with_context_exact_native(
            backend,
            &[
                pm_u64(image_count)?,
                pm_u64(self.configuration.prompt_width)?,
            ],
            &fused,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(photo_canonical)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::Scatter,
            phase_hook,
        )?;
        let scattered = scatter_method_with_context_exact_native(
            backend,
            &flattened_prompt,
            0,
            &index,
            &fused,
            context,
        )
        .map_err(photo_canonical)?;
        let output = torch_reshape_with_context_exact_native(
            backend,
            &scattered,
            &[
                1,
                pm_i64(sequence)?,
                pm_i64(self.configuration.prompt_width)?,
            ],
            context,
        )
        .map_err(photo_canonical)?;
        photo_require_finite_invocation("output", &output, context.cancellation)?;
        photo_checkpoint(
            context.cancellation,
            NativePhotoMakerPhase::InvocationReturn,
            phase_hook,
        )?;
        Ok(output)
    }

    fn photo_tensor(&self, key: &str) -> Result<&Tensor, NativePhotoMakerError> {
        self.execution_state
            .get(key)
            .ok_or_else(|| NativePhotoMakerError::MissingState(key.to_owned()))
    }

    fn photo_linear_state(
        &self,
        backend: &CpuBackend,
        input: &[f32],
        input_shape: &[usize],
        weight: &str,
        bias: Option<&str>,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, NativePhotoMakerError> {
        photo_linear(
            backend,
            input,
            input_shape,
            self.photo_tensor(weight)?,
            bias.map(|key| self.photo_tensor(key)).transpose()?,
            context,
        )
    }

    fn photo_layer_norm(
        &self,
        backend: &CpuBackend,
        input: &[f32],
        shape: &[usize],
        weight: &str,
        bias: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, NativePhotoMakerError> {
        let width = *shape.last().ok_or(NativePhotoMakerError::ShapeOverflow)?;
        let weight =
            tensor_to_f32_with_context_exact_native(backend, self.photo_tensor(weight)?, context)
                .map_err(photo_canonical)?;
        let bias =
            tensor_to_f32_with_context_exact_native(backend, self.photo_tensor(bias)?, context)
                .map_err(photo_canonical)?;
        layer_norm_with_context_exact_native(
            backend,
            input,
            shape,
            &[width],
            Some(&weight),
            Some(&bias),
            LAYER_NORM_EPSILON,
            DeviceId::CPU,
            context,
        )
        .map_err(photo_canonical)
    }
}

fn photo_source_configuration() -> NativePhotoMakerConfiguration {
    NativePhotoMakerConfiguration {
        hidden: PHOTOMAKER_SOURCE_HIDDEN,
        intermediate: PHOTOMAKER_SOURCE_INTERMEDIATE,
        heads: PHOTOMAKER_SOURCE_HEADS,
        layers: PHOTOMAKER_SOURCE_LAYERS,
        image_size: PHOTOMAKER_SOURCE_IMAGE,
        patch_size: PHOTOMAKER_SOURCE_PATCH,
        projection: PHOTOMAKER_SOURCE_PROJECTION,
        extra_projection: PHOTOMAKER_SOURCE_EXTRA_PROJECTION,
        prompt_width: PHOTOMAKER_SOURCE_PROJECTION + PHOTOMAKER_SOURCE_EXTRA_PROJECTION,
        source_exact: true,
    }
}

#[cfg(any(test, feature = "test-support"))]
fn photo_reduced_configuration() -> NativePhotoMakerConfiguration {
    NativePhotoMakerConfiguration {
        hidden: PHOTOMAKER_REDUCED_HIDDEN,
        intermediate: PHOTOMAKER_REDUCED_INTERMEDIATE,
        heads: PHOTOMAKER_REDUCED_HEADS,
        layers: PHOTOMAKER_SOURCE_LAYERS,
        image_size: PHOTOMAKER_REDUCED_IMAGE,
        patch_size: PHOTOMAKER_REDUCED_PATCH,
        projection: PHOTOMAKER_REDUCED_PROJECTION,
        extra_projection: PHOTOMAKER_REDUCED_EXTRA_PROJECTION,
        prompt_width: PHOTOMAKER_REDUCED_PROJECTION + PHOTOMAKER_REDUCED_EXTRA_PROJECTION,
        source_exact: false,
    }
}

fn photo_checkpoint(
    cancellation: &CancellationToken,
    phase: NativePhotoMakerPhase,
    phase_hook: &mut impl FnMut(NativePhotoMakerPhase, &CancellationToken),
) -> Result<(), NativePhotoMakerError> {
    phase_hook(phase, cancellation);
    cancellation.check()?;
    Ok(())
}

fn photo_raw_checkpoint_preflight(
    checkpoint: &NativePhotoMakerCheckpoint,
) -> Result<(), NativePhotoMakerError> {
    if checkpoint
        .ordered_entries
        .iter()
        .all(|entry| matches!(entry, NativePhotoMakerCheckpointEntry::Tensor { .. }))
    {
        return photo_raw_state_preflight(
            checkpoint,
            checkpoint.ordered_entries.len(),
            checkpoint
                .ordered_entries
                .iter()
                .filter_map(|entry| match entry {
                    NativePhotoMakerCheckpointEntry::Tensor { key, tensor } => Some((key, tensor)),
                    NativePhotoMakerCheckpointEntry::Mapping { .. } => None,
                }),
        );
    }
    let [NativePhotoMakerCheckpointEntry::Mapping { key, ordered_state }] =
        checkpoint.ordered_entries.as_slice()
    else {
        return Err(NativePhotoMakerError::InvalidCheckpoint(
            "checkpoint outer mapping must contain only id_encoder".to_owned(),
        ));
    };
    if key != "id_encoder" {
        return Err(NativePhotoMakerError::InvalidCheckpoint(
            "checkpoint nested mapping must be named id_encoder".to_owned(),
        ));
    }
    photo_raw_state_preflight(
        checkpoint,
        ordered_state.len(),
        ordered_state.iter().map(|(key, tensor)| (key, tensor)),
    )
}

fn photo_raw_state_preflight<'a>(
    checkpoint: &NativePhotoMakerCheckpoint,
    state_count: usize,
    state: impl Iterator<Item = (&'a String, &'a Tensor)>,
) -> Result<(), NativePhotoMakerError> {
    if state_count != PHOTOMAKER_STATE_COUNT {
        return Err(NativePhotoMakerError::UnexpectedState(format!(
            "expected {PHOTOMAKER_STATE_COUNT} entries, got {}",
            state_count
        )));
    }
    let mut source_bytes = 0_u64;
    let mut projected_bytes = 0_u64;
    let mut maximum_scratch = 0_u64;
    let mut key_bytes = 0_u64;
    for (key, tensor) in state {
        if key.is_empty() || key.len() > MAX_STATE_KEY_BYTES {
            return Err(NativePhotoMakerError::InvalidCheckpoint(
                "state key length is invalid".to_owned(),
            ));
        }
        let source = tensor.storage_byte_len();
        let projected = tensor
            .descriptor()
            .element_count()?
            .checked_mul(DType::F32.byte_width())
            .ok_or(NativePhotoMakerError::ShapeOverflow)?;
        source_bytes = source_bytes
            .checked_add(source)
            .ok_or(NativePhotoMakerError::ShapeOverflow)?;
        projected_bytes = projected_bytes
            .checked_add(projected)
            .ok_or(NativePhotoMakerError::ShapeOverflow)?;
        maximum_scratch = maximum_scratch.max(source.max(projected));
        key_bytes = key_bytes
            .checked_add(pm_u64(key.capacity())?)
            .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    }
    let container_bytes = pm_u64(PHOTOMAKER_STATE_COUNT)?
        .checked_mul(
            u64::try_from(mem::size_of::<(String, Tensor)>())
                .map_err(|_| NativePhotoMakerError::ShapeOverflow)?,
        )
        .and_then(|bytes| bytes.checked_mul(3))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let required = source_bytes
        .checked_add(projected_bytes)
        .and_then(|bytes| bytes.checked_add(maximum_scratch.checked_mul(2)?))
        .and_then(|bytes| bytes.checked_add(key_bytes.checked_mul(3)?))
        .and_then(|bytes| bytes.checked_add(container_bytes))
        .and_then(|bytes| {
            bytes.checked_add(u64::try_from(mem::size_of::<NativePhotoMakerResource>()).ok()?)
        })
        .and_then(|bytes| bytes.checked_add(pm_u64(checkpoint.artifact_sha256.capacity()).ok()?))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    if required > checkpoint.memory_budget_bytes {
        return Err(NativePhotoMakerError::OutOfMemory {
            required,
            budget: checkpoint.memory_budget_bytes,
        });
    }
    Ok(())
}

fn photo_normalize_checkpoint(
    entries: Vec<NativePhotoMakerCheckpointEntry>,
) -> Result<(NativePhotoMakerWrapper, Vec<(String, Tensor)>), NativePhotoMakerError> {
    if entries
        .iter()
        .all(|entry| matches!(entry, NativePhotoMakerCheckpointEntry::Tensor { .. }))
    {
        let state = entries
            .into_iter()
            .map(|entry| match entry {
                NativePhotoMakerCheckpointEntry::Tensor { key, tensor } => Ok((key, tensor)),
                NativePhotoMakerCheckpointEntry::Mapping { .. } => {
                    Err(NativePhotoMakerError::InvalidCheckpoint(
                        "flat checkpoint contains a mapping".to_owned(),
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((NativePhotoMakerWrapper::Flat, state));
    }
    let [NativePhotoMakerCheckpointEntry::Mapping { key, ordered_state }] = entries.as_slice()
    else {
        return Err(NativePhotoMakerError::InvalidCheckpoint(
            "checkpoint outer mapping must contain only id_encoder".to_owned(),
        ));
    };
    if key != "id_encoder" {
        return Err(NativePhotoMakerError::InvalidCheckpoint(
            "checkpoint nested mapping must be named id_encoder".to_owned(),
        ));
    }
    Ok((NativePhotoMakerWrapper::IdEncoder, ordered_state.clone()))
}

fn photo_state_manifest(
    configuration: NativePhotoMakerConfiguration,
) -> Result<Vec<StateSpecification>, NativePhotoMakerError> {
    let hidden = pm_u64(configuration.hidden)?;
    let intermediate = pm_u64(configuration.intermediate)?;
    let positions = configuration
        .image_size
        .checked_div(configuration.patch_size)
        .and_then(|side| side.checked_mul(side))
        .and_then(|patches| patches.checked_add(1))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let mut specifications = Vec::new();
    specifications
        .try_reserve_exact(PHOTOMAKER_STATE_COUNT)
        .map_err(|_| NativePhotoMakerError::Allocation)?;
    let mut push = |key: String, shape: Vec<u64>| {
        specifications.push(StateSpecification { key, shape });
    };
    push(
        "vision_model.embeddings.class_embedding".to_owned(),
        vec![hidden],
    );
    push(
        "vision_model.embeddings.patch_embedding.weight".to_owned(),
        vec![
            hidden,
            3,
            pm_u64(configuration.patch_size)?,
            pm_u64(configuration.patch_size)?,
        ],
    );
    push(
        "vision_model.embeddings.position_embedding.weight".to_owned(),
        vec![pm_u64(positions)?, hidden],
    );
    push("vision_model.pre_layrnorm.weight".to_owned(), vec![hidden]);
    push("vision_model.pre_layrnorm.bias".to_owned(), vec![hidden]);
    for layer in 0..configuration.layers {
        let prefix = format!("vision_model.encoder.layers.{layer}");
        for (suffix, shape) in [
            ("layer_norm1.weight", vec![hidden]),
            ("layer_norm1.bias", vec![hidden]),
            ("self_attn.q_proj.weight", vec![hidden, hidden]),
            ("self_attn.q_proj.bias", vec![hidden]),
            ("self_attn.k_proj.weight", vec![hidden, hidden]),
            ("self_attn.k_proj.bias", vec![hidden]),
            ("self_attn.v_proj.weight", vec![hidden, hidden]),
            ("self_attn.v_proj.bias", vec![hidden]),
            ("self_attn.out_proj.weight", vec![hidden, hidden]),
            ("self_attn.out_proj.bias", vec![hidden]),
            ("layer_norm2.weight", vec![hidden]),
            ("layer_norm2.bias", vec![hidden]),
            ("mlp.fc1.weight", vec![intermediate, hidden]),
            ("mlp.fc1.bias", vec![intermediate]),
            ("mlp.fc2.weight", vec![hidden, intermediate]),
            ("mlp.fc2.bias", vec![hidden]),
        ] {
            push(format!("{prefix}.{suffix}"), shape);
        }
    }
    push(
        "vision_model.post_layernorm.weight".to_owned(),
        vec![hidden],
    );
    push("vision_model.post_layernorm.bias".to_owned(), vec![hidden]);
    push(
        "visual_projection.weight".to_owned(),
        vec![pm_u64(configuration.projection)?, hidden],
    );
    push(
        "visual_projection_2.weight".to_owned(),
        vec![pm_u64(configuration.extra_projection)?, hidden],
    );
    let prompt = pm_u64(configuration.prompt_width)?;
    let stacked = prompt
        .checked_mul(2)
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    for prefix in ["fuse_module.mlp1", "fuse_module.mlp2"] {
        let input = if prefix.ends_with("mlp1") {
            stacked
        } else {
            prompt
        };
        push(format!("{prefix}.layernorm.weight"), vec![input]);
        push(format!("{prefix}.layernorm.bias"), vec![input]);
        push(format!("{prefix}.fc1.weight"), vec![prompt, input]);
        push(format!("{prefix}.fc1.bias"), vec![prompt]);
        push(format!("{prefix}.fc2.weight"), vec![prompt, prompt]);
        push(format!("{prefix}.fc2.bias"), vec![prompt]);
    }
    push("fuse_module.layer_norm.weight".to_owned(), vec![prompt]);
    push("fuse_module.layer_norm.bias".to_owned(), vec![prompt]);
    if specifications.len() != PHOTOMAKER_STATE_COUNT {
        return Err(NativePhotoMakerError::ShapeOverflow);
    }
    Ok(specifications)
}

fn photo_validate_ordered_keys(
    state: &[(String, Tensor)],
    specifications: &[StateSpecification],
) -> Result<(), NativePhotoMakerError> {
    if state.len() != specifications.len() {
        return Err(NativePhotoMakerError::UnexpectedState(format!(
            "expected {} entries, got {}",
            specifications.len(),
            state.len()
        )));
    }
    let mut seen = HashSet::new();
    for ((actual, _), expected) in state.iter().zip(specifications) {
        if !seen.insert(actual) {
            return Err(NativePhotoMakerError::UnexpectedState(format!(
                "duplicate key {actual}"
            )));
        }
        if actual != &expected.key {
            return Err(NativePhotoMakerError::UnexpectedState(format!(
                "expected {}, got {actual}",
                expected.key
            )));
        }
    }
    Ok(())
}

fn photo_validate_source_state(
    state: &[(String, Tensor)],
    specifications: &[StateSpecification],
    stream: StreamId,
    cancellation: &CancellationToken,
) -> Result<DType, NativePhotoMakerError> {
    photo_validate_ordered_keys(state, specifications)?;
    let mut dtype = None;
    let mut storages = HashSet::new();
    for (index, ((key, tensor), specification)) in state.iter().zip(specifications).enumerate() {
        if index.is_multiple_of(8) {
            cancellation.check()?;
        }
        if key.is_empty() || key.len() > MAX_STATE_KEY_BYTES {
            return Err(NativePhotoMakerError::InvalidCheckpoint(
                "state key length is invalid".to_owned(),
            ));
        }
        let descriptor = tensor.descriptor();
        if descriptor.shape() != specification.shape
            || !matches!(descriptor.dtype(), DType::F32 | DType::F16 | DType::Bf16)
        {
            return Err(NativePhotoMakerError::StateShape {
                key: key.clone(),
                expected: specification.shape.clone(),
                actual: descriptor.shape().to_vec(),
                actual_dtype: descriptor.dtype(),
            });
        }
        if descriptor.device() != DeviceId::CPU
            || descriptor.stream() != stream
            || !descriptor.is_contiguous()?
        {
            return Err(NativePhotoMakerError::InvalidCheckpoint(format!(
                "state {key} must be contiguous CPU storage on the construction stream"
            )));
        }
        if dtype
            .replace(descriptor.dtype())
            .is_some_and(|previous| previous != descriptor.dtype())
        {
            return Err(NativePhotoMakerError::InvalidCheckpoint(
                "all source tensors must use one storage dtype".to_owned(),
            ));
        }
        if !storages.insert(tensor.storage_id()) {
            return Err(NativePhotoMakerError::InvalidCheckpoint(format!(
                "state {key} aliases another source tensor"
            )));
        }
        photo_validate_finite_source_tensor(key, tensor, cancellation)?;
    }
    dtype.ok_or_else(|| NativePhotoMakerError::InvalidCheckpoint("state is empty".to_owned()))
}

fn photo_validate_finite_source_tensor(
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativePhotoMakerError> {
    let count = pm_usize(tensor.descriptor().element_count()?)?;
    for index in 0..count {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        match tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(pm_u64(index)?)?)?
        {
            DecodedScalar::Real(value) if value.is_finite() => {}
            _ => {
                return Err(NativePhotoMakerError::InvalidCheckpoint(format!(
                    "state {key} contains a non-finite or non-real value"
                )));
            }
        }
    }
    cancellation.check()?;
    Ok(())
}

fn photo_validate_finite_tensor(
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativePhotoMakerError> {
    let bytes = tensor.contiguous_bytes()?;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        let raw: [u8; 4] = chunk
            .try_into()
            .map_err(|_| NativePhotoMakerError::SemanticStateChanged)?;
        if !f32::from_ne_bytes(raw).is_finite() {
            return Err(NativePhotoMakerError::InvalidCheckpoint(format!(
                "state {key} contains a non-finite value"
            )));
        }
    }
    cancellation.check()?;
    Ok(())
}

fn photo_validate_distinct_storage(
    source: &BTreeMap<String, Tensor>,
    execution: &BTreeMap<String, Tensor>,
) -> Result<(), NativePhotoMakerError> {
    let source_ids = source
        .values()
        .map(Tensor::storage_id)
        .collect::<HashSet<_>>();
    let mut execution_ids = HashSet::new();
    for tensor in execution.values() {
        if source_ids.contains(&tensor.storage_id()) || !execution_ids.insert(tensor.storage_id()) {
            return Err(NativePhotoMakerError::SemanticStateChanged);
        }
    }
    Ok(())
}

fn photo_clip_configuration(
    configuration: NativePhotoMakerConfiguration,
) -> ClipVisionConfiguration {
    ClipVisionConfiguration {
        model_type: ClipVisionModelType::Clip,
        dtype: DType::F32,
        device: DeviceId::CPU,
        hidden_size: configuration.hidden,
        intermediate_size: configuration.intermediate,
        attention_heads: configuration.heads,
        layer_count: configuration.layers,
        image_size: configuration.image_size,
        patch_size: configuration.patch_size,
        num_channels: 3,
        max_num_patches: 0,
        activation: ClipVisionActivation::QuickGelu,
        projection_dimension: Some(configuration.projection),
        llava_projection_dimension: None,
    }
}

fn photo_clip_weights(
    state: &BTreeMap<String, Tensor>,
    configuration: NativePhotoMakerConfiguration,
) -> Result<ClipVisionWeights, NativePhotoMakerError> {
    let take = |key: &str| {
        state
            .get(key)
            .cloned()
            .ok_or_else(|| NativePhotoMakerError::MissingState(key.to_owned()))
    };
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(configuration.layers)
        .map_err(|_| NativePhotoMakerError::Allocation)?;
    for layer in 0..configuration.layers {
        let prefix = format!("vision_model.encoder.layers.{layer}");
        layers.push(ClipVisionLayerWeights {
            layer_norm_1_weight: take(&format!("{prefix}.layer_norm1.weight"))?,
            layer_norm_1_bias: take(&format!("{prefix}.layer_norm1.bias"))?,
            query_weight: take(&format!("{prefix}.self_attn.q_proj.weight"))?,
            query_bias: take(&format!("{prefix}.self_attn.q_proj.bias"))?,
            key_weight: take(&format!("{prefix}.self_attn.k_proj.weight"))?,
            key_bias: take(&format!("{prefix}.self_attn.k_proj.bias"))?,
            value_weight: take(&format!("{prefix}.self_attn.v_proj.weight"))?,
            value_bias: take(&format!("{prefix}.self_attn.v_proj.bias"))?,
            output_weight: take(&format!("{prefix}.self_attn.out_proj.weight"))?,
            output_bias: take(&format!("{prefix}.self_attn.out_proj.bias"))?,
            layer_norm_2_weight: take(&format!("{prefix}.layer_norm2.weight"))?,
            layer_norm_2_bias: take(&format!("{prefix}.layer_norm2.bias"))?,
            feed_forward_1_weight: take(&format!("{prefix}.mlp.fc1.weight"))?,
            feed_forward_1_bias: take(&format!("{prefix}.mlp.fc1.bias"))?,
            feed_forward_2_weight: take(&format!("{prefix}.mlp.fc2.weight"))?,
            feed_forward_2_bias: take(&format!("{prefix}.mlp.fc2.bias"))?,
        });
    }
    Ok(ClipVisionWeights {
        patch_embedding_weight: take("vision_model.embeddings.patch_embedding.weight")?,
        patch_embedding_bias: None,
        class_embedding: Some(take("vision_model.embeddings.class_embedding")?),
        position_embedding: take("vision_model.embeddings.position_embedding.weight")?,
        pre_layer_norm_weight: Some(take("vision_model.pre_layrnorm.weight")?),
        pre_layer_norm_bias: Some(take("vision_model.pre_layrnorm.bias")?),
        layers,
        post_layer_norm_weight: take("vision_model.post_layernorm.weight")?,
        post_layer_norm_bias: take("vision_model.post_layernorm.bias")?,
        visual_projection_weight: Some(take("visual_projection.weight")?),
        llava_linear_1_weight: None,
        llava_linear_1_bias: None,
        llava_linear_2_weight: None,
        llava_linear_2_bias: None,
    })
}

fn photo_require_clip_storage_from_execution(
    clip: &NativeClipVision,
    execution: &BTreeMap<String, Tensor>,
) -> Result<(), NativePhotoMakerError> {
    let execution_ids = execution
        .values()
        .map(Tensor::storage_id)
        .collect::<HashSet<_>>();
    let parts = clip.resident_parts()?;
    if parts.tensor_allocations().len() != PHOTOMAKER_CLIP_STATE_COUNT
        || parts
            .tensor_allocations()
            .iter()
            .any(|allocation| !execution_ids.contains(&allocation.storage_id()))
    {
        return Err(NativePhotoMakerError::SemanticStateChanged);
    }
    Ok(())
}

fn photo_validate_resource_storage(
    source: &BTreeMap<String, Tensor>,
    execution: &BTreeMap<String, Tensor>,
    clip: &NativeClipVision,
) -> Result<(), NativePhotoMakerError> {
    photo_validate_distinct_storage(source, execution)?;
    let mut retained = source
        .values()
        .chain(execution.values())
        .map(Tensor::storage_id)
        .collect::<HashSet<_>>();
    let parts = clip.resident_parts()?;
    if parts.tensor_allocations().len() != PHOTOMAKER_CLIP_STATE_COUNT {
        return Err(NativePhotoMakerError::SemanticStateChanged);
    }
    for allocation in parts.tensor_allocations() {
        if !retained.insert(allocation.storage_id()) {
            return Err(NativePhotoMakerError::SemanticStateChanged);
        }
    }
    Ok(())
}

fn photo_validate_invocation(
    configuration: NativePhotoMakerConfiguration,
    image: &Tensor,
    prompt: &Tensor,
    mask: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(), NativePhotoMakerError> {
    let image_shape = image.descriptor().shape();
    let prompt_shape = prompt.descriptor().shape();
    let mask_shape = mask.descriptor().shape();
    if image_shape.len() != 5
        || image_shape[0] != 1
        || image_shape[1] == 0
        || image_shape[2] != 3
        || image_shape[3] != pm_u64(configuration.image_size)?
        || image_shape[4] != pm_u64(configuration.image_size)?
    {
        return Err(NativePhotoMakerError::InvalidInput(
            "image must have [1, N, 3, image_size, image_size] shape with N greater than zero"
                .to_owned(),
        ));
    }
    if prompt_shape.len() != 3
        || prompt_shape[0] != 1
        || prompt_shape[1] == 0
        || prompt_shape[2] != pm_u64(configuration.prompt_width)?
    {
        return Err(NativePhotoMakerError::InvalidInput(
            "prompt must have [1, S, projection_width] shape".to_owned(),
        ));
    }
    if mask_shape != [1, prompt_shape[1]] {
        return Err(NativePhotoMakerError::InvalidInput(
            "class token mask must have [1, S] shape".to_owned(),
        ));
    }
    for (name, tensor, dtype) in [
        ("image", image, DType::F32),
        ("prompt", prompt, DType::F32),
        ("class token mask", mask, DType::Bool),
    ] {
        let descriptor = tensor.descriptor();
        if descriptor.dtype() != dtype
            || descriptor.device() != DeviceId::CPU
            || descriptor.stream() != context.stream
            || !descriptor.is_contiguous()?
        {
            return Err(NativePhotoMakerError::InvalidInput(format!(
                "{name} must be contiguous {dtype:?} CPU state on the execution stream"
            )));
        }
    }
    photo_require_finite_invocation("image", image, context.cancellation)?;
    photo_require_finite_invocation("prompt", prompt, context.cancellation)?;
    Ok(())
}

fn photo_mask_true_count(
    backend: &CpuBackend,
    mask: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<usize, NativePhotoMakerError> {
    let sum = tensor_sum_with_context_exact_native(backend, mask, None, false, None, context)
        .map_err(photo_canonical)?;
    match sum
        .descriptor()
        .dtype()
        .decode_scalar(sum.linear_element_bytes(0)?)?
    {
        DecodedScalar::Signed(value) => {
            usize::try_from(value).map_err(|_| NativePhotoMakerError::ShapeOverflow)
        }
        DecodedScalar::Unsigned(value) => pm_usize(value),
        _ => Err(NativePhotoMakerError::SemanticStateChanged),
    }
}

fn photo_require_finite_invocation(
    name: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativePhotoMakerError> {
    let bytes = tensor.contiguous_bytes()?;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        let raw: [u8; 4] = chunk
            .try_into()
            .map_err(|_| NativePhotoMakerError::InvalidInput(name.to_owned()))?;
        if !f32::from_ne_bytes(raw).is_finite() {
            return Err(NativePhotoMakerError::InvalidInput(format!(
                "{name} contains a non-finite value"
            )));
        }
    }
    cancellation.check()?;
    Ok(())
}

fn photo_mask_positions(
    backend: &CpuBackend,
    mask: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<usize>, NativePhotoMakerError> {
    let matrix = match nonzero_with_context_exact_native(backend, mask, false, context)
        .map_err(photo_canonical)?
    {
        NonzeroOutput::Matrix(matrix) => matrix,
        NonzeroOutput::Tuple(_) => {
            return Err(NativePhotoMakerError::SemanticStateChanged);
        }
    };
    let [rows, columns] = matrix.descriptor().shape() else {
        return Err(NativePhotoMakerError::SemanticStateChanged);
    };
    if *columns != 2 {
        return Err(NativePhotoMakerError::SemanticStateChanged);
    }
    let rows = pm_usize(*rows)?;
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(rows)
        .map_err(|_| NativePhotoMakerError::Allocation)?;
    for row in 0..rows {
        context.check()?;
        let byte_index = row
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(NativePhotoMakerError::ShapeOverflow)?;
        let bytes = matrix.linear_element_bytes(pm_u64(byte_index)?)?;
        let raw: [u8; 8] = bytes
            .try_into()
            .map_err(|_| NativePhotoMakerError::SemanticStateChanged)?;
        let position = i64::from_ne_bytes(raw);
        positions
            .push(usize::try_from(position).map_err(|_| NativePhotoMakerError::ShapeOverflow)?);
    }
    Ok(positions)
}

fn photo_row_index_tensor(
    backend: &CpuBackend,
    positions: &[usize],
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativePhotoMakerError> {
    let count = positions
        .len()
        .checked_mul(width)
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            count
                .checked_mul(mem::size_of::<i64>())
                .ok_or(NativePhotoMakerError::ShapeOverflow)?,
        )
        .map_err(|_| NativePhotoMakerError::Allocation)?;
    for (row, position) in positions.iter().copied().enumerate() {
        context.check()?;
        let position = i64::try_from(position).map_err(|_| NativePhotoMakerError::ShapeOverflow)?;
        for _ in 0..width {
            bytes.extend_from_slice(&position.to_ne_bytes());
        }
        if row.is_multiple_of(64) {
            context.check()?;
        }
    }
    let descriptor = sim_tensor::TensorDescriptor::contiguous(
        vec![pm_u64(positions.len())?, pm_u64(width)?],
        DType::I64,
        DeviceId::CPU,
        context.stream,
    )?;
    let (tensor, event) = backend.upload_bytes(descriptor, &bytes, context)?;
    backend.wait_event(event, context)?;
    Ok(tensor)
}

fn photo_linear(
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &Tensor,
    bias: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativePhotoMakerError> {
    let weight_values = tensor_to_f32_with_context_exact_native(backend, weight, context)
        .map_err(photo_canonical)?;
    let bias_values = bias
        .map(|bias| tensor_to_f32_with_context_exact_native(backend, bias, context))
        .transpose()
        .map_err(photo_canonical)?;
    let weight_shape = weight
        .descriptor()
        .shape()
        .iter()
        .map(|dimension| pm_usize(*dimension))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(linear_with_context_exact_native(
        backend,
        input,
        input_shape,
        &weight_values,
        &weight_shape,
        bias_values.as_deref(),
        DeviceId::CPU,
        context,
    )
    .map_err(photo_canonical)?
    .values)
}

fn photo_add_in_place(
    left: &mut [f32],
    right: &[f32],
    cancellation: &CancellationToken,
) -> Result<(), NativePhotoMakerError> {
    if left.len() != right.len() {
        return Err(NativePhotoMakerError::ShapeOverflow);
    }
    for (index, (left, right)) in left.iter_mut().zip(right).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        *left += *right;
    }
    cancellation.check()?;
    Ok(())
}

fn photo_construction_peak(
    artifact: &str,
    artifact_capacity: usize,
    ordered_state: &[(String, Tensor)],
    specifications: &[StateSpecification],
) -> Result<u64, NativePhotoMakerError> {
    let source_bytes = ordered_state.iter().try_fold(0_u64, |total, (_, tensor)| {
        total
            .checked_add(tensor.storage_byte_len())
            .ok_or(NativePhotoMakerError::ShapeOverflow)
    })?;
    let projected_bytes = specifications
        .iter()
        .try_fold(0_u64, |total, specification| {
            total
                .checked_add(photo_tensor_bytes(&specification.shape, DType::F32)?)
                .ok_or(NativePhotoMakerError::ShapeOverflow)
        })?;
    let maximum_projection_scratch = specifications
        .iter()
        .map(|specification| photo_tensor_bytes(&specification.shape, DType::F32))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0)
        .checked_mul(2)
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let key_bytes = ordered_state.iter().try_fold(0_u64, |total, (key, _)| {
        total
            .checked_add(pm_u64(key.capacity())?)
            .ok_or(NativePhotoMakerError::ShapeOverflow)
    })?;
    let container_bytes = pm_u64(ordered_state.len())?
        .checked_mul(
            u64::try_from(mem::size_of::<(String, Tensor)>())
                .map_err(|_| NativePhotoMakerError::ShapeOverflow)?,
        )
        .and_then(|bytes| bytes.checked_mul(3))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    source_bytes
        .checked_add(projected_bytes)
        .and_then(|bytes| bytes.checked_add(maximum_projection_scratch))
        .and_then(|bytes| bytes.checked_add(key_bytes.checked_mul(3)?))
        .and_then(|bytes| bytes.checked_add(container_bytes))
        .and_then(|bytes| {
            bytes.checked_add(u64::try_from(mem::size_of::<NativePhotoMakerResource>()).ok()?)
        })
        .and_then(|bytes| bytes.checked_add(pm_u64(artifact_capacity.max(artifact.len())).ok()?))
        .ok_or(NativePhotoMakerError::ShapeOverflow)
}

fn photo_invocation_peak(
    resource: &NativePhotoMakerResource,
    image_count: usize,
    sequence: usize,
) -> Result<(), NativePhotoMakerError> {
    let configuration = resource.configuration;
    let positions = configuration
        .image_size
        .checked_div(configuration.patch_size)
        .and_then(|side| side.checked_mul(side))
        .and_then(|patches| patches.checked_add(1))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let scalar_bytes = 4_u64;
    let image_bytes = pm_u64(image_count)?
        .checked_mul(3)
        .and_then(|count| count.checked_mul(pm_u64(configuration.image_size).ok()?))
        .and_then(|count| count.checked_mul(pm_u64(configuration.image_size).ok()?))
        .and_then(|count| count.checked_mul(scalar_bytes))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let hidden_bytes = pm_u64(image_count)?
        .checked_mul(pm_u64(positions)?)
        .and_then(|count| count.checked_mul(pm_u64(configuration.hidden).ok()?))
        .and_then(|count| count.checked_mul(scalar_bytes))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let attention_bytes = pm_u64(image_count)?
        .checked_mul(pm_u64(configuration.heads)?)
        .and_then(|count| count.checked_mul(pm_u64(positions).ok()?))
        .and_then(|count| count.checked_mul(pm_u64(positions).ok()?))
        .and_then(|count| count.checked_mul(scalar_bytes))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let prompt_bytes = pm_u64(sequence)?
        .checked_mul(pm_u64(configuration.prompt_width)?)
        .and_then(|count| count.checked_mul(scalar_bytes))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let selected_bytes = pm_u64(image_count)?
        .checked_mul(pm_u64(configuration.prompt_width)?)
        .and_then(|count| count.checked_mul(scalar_bytes))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let decoded_weight_bytes = resource
        .execution_state
        .values()
        .map(Tensor::storage_byte_len)
        .max()
        .unwrap_or(0)
        .checked_mul(2)
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let phase_bytes = image_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(hidden_bytes.checked_mul(32)?))
        .and_then(|bytes| bytes.checked_add(attention_bytes.checked_mul(2)?))
        .and_then(|bytes| bytes.checked_add(prompt_bytes.checked_mul(3)?))
        .and_then(|bytes| bytes.checked_add(selected_bytes.checked_mul(24)?))
        .and_then(|bytes| bytes.checked_add(decoded_weight_bytes))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let required = resource
        .resident_bytes
        .checked_add(phase_bytes)
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    if required > resource.memory_budget_bytes {
        return Err(NativePhotoMakerError::OutOfMemory {
            required,
            budget: resource.memory_budget_bytes,
        });
    }
    Ok(())
}

fn photo_semantic_digest(
    configuration: NativePhotoMakerConfiguration,
    wrapper: NativePhotoMakerWrapper,
    artifact: &str,
    source_dtype: DType,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
    clip_digest: &str,
    specifications: &[StateSpecification],
    cancellation: &CancellationToken,
) -> Result<String, NativePhotoMakerError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.photomaker-resource.v1\0");
    hasher.update([match wrapper {
        NativePhotoMakerWrapper::Flat => 1,
        NativePhotoMakerWrapper::IdEncoder => 2,
    }]);
    for value in [
        configuration.hidden,
        configuration.intermediate,
        configuration.heads,
        configuration.layers,
        configuration.image_size,
        configuration.patch_size,
        configuration.projection,
        configuration.extra_projection,
        configuration.prompt_width,
    ] {
        hasher.update(pm_u64(value)?.to_le_bytes());
    }
    hasher.update([u8::from(configuration.source_exact)]);
    hasher.update(artifact.as_bytes());
    hasher.update(source_dtype.catalog_name().as_bytes());
    hasher.update(clip_digest.as_bytes());
    for (index, specification) in specifications.iter().enumerate() {
        if index.is_multiple_of(8) {
            cancellation.check()?;
        }
        photo_hash_tensor(
            &mut hasher,
            &specification.key,
            source_state
                .get(&specification.key)
                .ok_or_else(|| NativePhotoMakerError::MissingState(specification.key.clone()))?,
            cancellation,
        )?;
        if index >= PHOTOMAKER_CLIP_STATE_COUNT {
            photo_hash_tensor(
                &mut hasher,
                &specification.key,
                execution_state.get(&specification.key).ok_or_else(|| {
                    NativePhotoMakerError::MissingState(specification.key.clone())
                })?,
                cancellation,
            )?;
        }
    }
    cancellation.check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn photo_hash_tensor(
    hasher: &mut Sha256,
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativePhotoMakerError> {
    hasher.update(pm_u64(key.len())?.to_le_bytes());
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

fn photo_resident_owned_bytes(
    artifact: &str,
    artifact_capacity: usize,
    digest: &str,
    digest_capacity: usize,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
    clip: &NativeClipVision,
) -> Result<u64, NativePhotoMakerError> {
    let base = u64::try_from(mem::size_of::<NativePhotoMakerResource>())
        .map_err(|_| NativePhotoMakerError::ShapeOverflow)?;
    let keys =
        source_state
            .keys()
            .chain(execution_state.keys())
            .try_fold(0_u64, |total, key| {
                total
                    .checked_add(pm_u64(key.capacity())?)
                    .ok_or(NativePhotoMakerError::ShapeOverflow)
            })?;
    let map_nodes = pm_u64(source_state.len())?
        .checked_mul(
            u64::try_from(mem::size_of::<(String, Tensor)>())
                .map_err(|_| NativePhotoMakerError::ShapeOverflow)?
                .checked_add(64)
                .ok_or(NativePhotoMakerError::ShapeOverflow)?,
        )
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or(NativePhotoMakerError::ShapeOverflow)?;
    let clip_owned_bytes = clip.resident_parts()?.owned_bytes();
    base.checked_add(pm_u64(artifact_capacity.max(artifact.len()))?)
        .and_then(|bytes| bytes.checked_add(pm_u64(digest_capacity.max(digest.len())).ok()?))
        .and_then(|bytes| bytes.checked_add(keys))
        .and_then(|bytes| bytes.checked_add(map_nodes))
        .and_then(|bytes| bytes.checked_add(clip_owned_bytes))
        .ok_or(NativePhotoMakerError::ShapeOverflow)
}

fn photo_resident_tensor_allocations<'a>(
    maps: impl IntoIterator<Item = &'a BTreeMap<String, Tensor>>,
    cancellation: &CancellationToken,
) -> Result<Vec<(StorageId, u64)>, NativePhotoMakerError> {
    let mut allocations = Vec::new();
    for map in maps {
        for (index, tensor) in map.values().enumerate() {
            if index.is_multiple_of(16) {
                cancellation.check()?;
            }
            let storage_id = tensor.storage_id();
            let bytes = tensor.storage_byte_len();
            if let Some((_, existing)) = allocations
                .iter()
                .find(|(existing_id, _)| *existing_id == storage_id)
            {
                if *existing != bytes {
                    return Err(NativePhotoMakerError::SemanticStateChanged);
                }
            } else {
                allocations.push((storage_id, bytes));
            }
        }
    }
    cancellation.check()?;
    Ok(allocations)
}

fn photo_resource_tensor_allocations(
    source: &BTreeMap<String, Tensor>,
    execution: &BTreeMap<String, Tensor>,
    clip: &NativeClipVision,
    cancellation: &CancellationToken,
) -> Result<Vec<(StorageId, u64)>, NativePhotoMakerError> {
    let mut allocations = photo_resident_tensor_allocations([source, execution], cancellation)?;
    let clip_parts = clip.resident_parts()?;
    if clip_parts.tensor_allocations().len() != PHOTOMAKER_CLIP_STATE_COUNT {
        return Err(NativePhotoMakerError::SemanticStateChanged);
    }
    for (index, allocation) in clip_parts.tensor_allocations().iter().enumerate() {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        if allocations
            .iter()
            .any(|(storage_id, _)| *storage_id == allocation.storage_id())
        {
            return Err(NativePhotoMakerError::SemanticStateChanged);
        }
        allocations.push((allocation.storage_id(), allocation.resident_bytes()));
    }
    cancellation.check()?;
    Ok(allocations)
}

fn photo_tensor_bytes(shape: &[u64], dtype: DType) -> Result<u64, NativePhotoMakerError> {
    shape
        .iter()
        .try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(NativePhotoMakerError::ShapeOverflow)
        })?
        .checked_mul(dtype.byte_width())
        .ok_or(NativePhotoMakerError::ShapeOverflow)
}

fn photo_validate_sha256(value: &str) -> Result<(), NativePhotoMakerError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NativePhotoMakerError::InvalidCheckpoint(
            "artifact identity must be canonical lowercase SHA-256".to_owned(),
        ));
    }
    Ok(())
}

fn photo_canonical(error: impl std::error::Error + 'static) -> NativePhotoMakerError {
    let mut source: &(dyn std::error::Error + 'static) = &error;
    loop {
        if is_cancellation_error(source) {
            return NativePhotoMakerError::Cancelled;
        }
        let Some(next) = source.source() else {
            break;
        };
        source = next;
    }
    NativePhotoMakerError::Canonical(error.to_string())
}

fn pm_u64(value: usize) -> Result<u64, NativePhotoMakerError> {
    u64::try_from(value).map_err(|_| NativePhotoMakerError::ShapeOverflow)
}

fn pm_usize(value: u64) -> Result<usize, NativePhotoMakerError> {
    usize::try_from(value).map_err(|_| NativePhotoMakerError::ShapeOverflow)
}

fn pm_i64(value: impl TryInto<i64>) -> Result<i64, NativePhotoMakerError> {
    value
        .try_into()
        .map_err(|_| NativePhotoMakerError::ShapeOverflow)
}

#[derive(Clone, Debug)]
pub struct NativeGligenCheckpoint {
    pub artifact_sha256: String,
    pub ordered_state: Vec<(String, Tensor)>,
    pub memory_budget_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeGligenRegion {
    InputBlock,
    MiddleBlock,
    OutputBlock,
}

impl NativeGligenRegion {
    const fn source_name(self) -> &'static str {
        match self {
            Self::InputBlock => "input_blocks",
            Self::MiddleBlock => "middle_block",
            Self::OutputBlock => "output_blocks",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "input_blocks" => Some(Self::InputBlock),
            "middle_block" => Some(Self::MiddleBlock),
            "output_blocks" => Some(Self::OutputBlock),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGligenFuserLocation {
    region: NativeGligenRegion,
    block_index: u8,
    namespace: String,
    transformer_index: usize,
    query_dimension: usize,
    key_dimension: usize,
    heads: usize,
    head_dimension: usize,
}

impl NativeGligenFuserLocation {
    pub const fn region(&self) -> NativeGligenRegion {
        self.region
    }

    pub const fn block_index(&self) -> u8 {
        self.block_index
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub const fn transformer_index(&self) -> usize {
        self.transformer_index
    }

    pub const fn query_dimension(&self) -> usize {
        self.query_dimension
    }

    pub const fn key_dimension(&self) -> usize {
        self.key_dimension
    }

    pub const fn heads(&self) -> usize {
        self.heads
    }

    pub const fn head_dimension(&self) -> usize {
        self.head_dimension
    }
}

#[derive(Clone, Debug)]
pub struct NativeGligenPositionParameter {
    pub embedding: Tensor,
    pub height: f32,
    pub width: f32,
    pub y: f32,
    pub x: f32,
}

#[derive(Debug)]
pub struct NativeGligenPreparedPositions {
    resource_semantic_digest_sha256: String,
    batch: usize,
    device: DeviceId,
    stream: StreamId,
    objects: Tensor,
}

impl NativeGligenPreparedPositions {
    pub const fn batch(&self) -> usize {
        self.batch
    }

    pub const fn device(&self) -> DeviceId {
        self.device
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn objects(&self) -> &Tensor {
        &self.objects
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeGligenPhase {
    Entry,
    Discovery,
    Schema,
    SourceValidation,
    Projection,
    SemanticDigest,
    Validation,
    Return,
    PositionAdmission,
    Fourier,
    NullProjection,
    PositionLinearOne,
    PositionSiluOne,
    PositionLinearTwo,
    PositionSiluTwo,
    PositionLinearThree,
    PositionReturn,
    ApplyAdmission,
    ContextProjection,
    Concatenate,
    AttentionLayerNorm,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionSdp,
    AttentionOutput,
    AttentionResidual,
    DenseLayerNorm,
    GegluProjection,
    GegluActivation,
    DenseOutput,
    DenseResidual,
    ApplyReturn,
}

#[derive(Debug)]
pub struct NativeGligenResource {
    artifact_sha256: String,
    source_state: BTreeMap<String, Tensor>,
    execution_state: BTreeMap<String, Tensor>,
    fusers: Box<[NativeGligenFuserLocation]>,
    key_dimension: usize,
    source_dtype: DType,
    stream: StreamId,
    memory_budget_bytes: u64,
    resident_bytes: u64,
    semantic_digest_sha256: String,
    source_exact: bool,
}

#[derive(Debug, Error)]
pub enum NativeGligenError {
    #[error("GLIGEN execution was cancelled")]
    Cancelled,
    #[error("GLIGEN checkpoint is invalid: {0}")]
    InvalidCheckpoint(String),
    #[error("GLIGEN state is missing key {0}")]
    MissingState(String),
    #[error("GLIGEN state is unexpected: {0}")]
    UnexpectedState(String),
    #[error("GLIGEN state {key} expected {expected:?}, got {actual:?} {actual_dtype:?}")]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        actual_dtype: DType,
    },
    #[error("GLIGEN input is invalid: {0}")]
    InvalidInput(String),
    #[error("GLIGEN retained semantic state changed")]
    SemanticStateChanged,
    #[error("GLIGEN shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("GLIGEN allocation failed")]
    Allocation,
    #[error("GLIGEN memory requirement {required} exceeds budget {budget}")]
    OutOfMemory { required: u64, budget: u64 },
    #[error("GLIGEN canonical owner failed: {0}")]
    Canonical(String),
    #[error(transparent)]
    Tensor(TensorError),
}

impl From<sim_types::CancellationError> for NativeGligenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<TensorError> for NativeGligenError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled,
            error => Self::Tensor(error),
        }
    }
}

impl NativeGligenResource {
    pub fn from_checkpoint(
        backend: &CpuBackend,
        checkpoint: NativeGligenCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeGligenError> {
        Self::checked(backend, checkpoint, true, context, &mut |_, _| {})
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn from_reduced_fixture(
        backend: &CpuBackend,
        checkpoint: NativeGligenCheckpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeGligenError> {
        Self::checked(backend, checkpoint, false, context, &mut |_, _| {})
    }

    fn checked(
        backend: &CpuBackend,
        checkpoint: NativeGligenCheckpoint,
        source_exact: bool,
        context: &ExecutionContext<'_>,
        phase_hook: &mut impl FnMut(NativeGligenPhase, &CancellationToken),
    ) -> Result<Self, NativeGligenError> {
        gligen_checkpoint(context.cancellation, NativeGligenPhase::Entry, phase_hook)?;
        gligen_validate_sha256(&checkpoint.artifact_sha256)?;
        if checkpoint.memory_budget_bytes == 0 {
            return Err(NativeGligenError::InvalidCheckpoint(
                "memory budget must be nonzero".to_owned(),
            ));
        }
        gligen_raw_checkpoint_preflight(&checkpoint)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::Discovery,
            phase_hook,
        )?;
        let (fusers, key_dimension, specifications) =
            gligen_discover_state(&checkpoint.ordered_state)?;
        gligen_checkpoint(context.cancellation, NativeGligenPhase::Schema, phase_hook)?;
        gligen_validate_ordered_keys(&checkpoint.ordered_state, &specifications)?;
        let source_dtype = gligen_validate_source_state(
            &checkpoint.ordered_state,
            &specifications,
            context.stream,
            context.cancellation,
        )?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::SourceValidation,
            phase_hook,
        )?;
        let construction_peak = gligen_construction_peak(
            &checkpoint.artifact_sha256,
            checkpoint.artifact_sha256.capacity(),
            &checkpoint.ordered_state,
            &specifications,
            &fusers,
        )?;
        if construction_peak > checkpoint.memory_budget_bytes {
            return Err(NativeGligenError::OutOfMemory {
                required: construction_peak,
                budget: checkpoint.memory_budget_bytes,
            });
        }
        let source_state = checkpoint
            .ordered_state
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let mut execution_state = BTreeMap::new();
        for (index, specification) in specifications.iter().enumerate() {
            if index.is_multiple_of(8) {
                gligen_checkpoint(
                    context.cancellation,
                    NativeGligenPhase::Projection,
                    phase_hook,
                )?;
            }
            let source = source_state
                .get(&specification.key)
                .ok_or_else(|| NativeGligenError::MissingState(specification.key.clone()))?;
            let projected = cast_to_with_context_exact_native(
                backend,
                source,
                DType::F32,
                DeviceId::CPU,
                false,
                true,
                context,
            )
            .map_err(gligen_canonical)?;
            gligen_validate_finite_tensor(&specification.key, &projected, context.cancellation)?;
            execution_state.insert(specification.key.clone(), projected);
        }
        gligen_validate_distinct_storage(&source_state, &execution_state)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::SemanticDigest,
            phase_hook,
        )?;
        let semantic_digest_sha256 = gligen_semantic_digest(
            source_exact,
            &checkpoint.artifact_sha256,
            source_dtype,
            key_dimension,
            &fusers,
            &source_state,
            &execution_state,
            &specifications,
            context.cancellation,
        )?;
        let owned_bytes = gligen_resident_owned_bytes(
            &checkpoint.artifact_sha256,
            checkpoint.artifact_sha256.capacity(),
            &semantic_digest_sha256,
            semantic_digest_sha256.capacity(),
            &source_state,
            &execution_state,
            &fusers,
        )?;
        let resident_bytes = gligen_resident_tensor_allocations(
            [&source_state, &execution_state],
            context.cancellation,
        )?
        .into_iter()
        .try_fold(owned_bytes, |total, (_, bytes)| {
            total
                .checked_add(bytes)
                .ok_or(NativeGligenError::ShapeOverflow)
        })?;
        if resident_bytes > checkpoint.memory_budget_bytes {
            return Err(NativeGligenError::OutOfMemory {
                required: resident_bytes,
                budget: checkpoint.memory_budget_bytes,
            });
        }
        let resource = Self {
            artifact_sha256: checkpoint.artifact_sha256,
            source_state,
            execution_state,
            fusers: fusers.into_boxed_slice(),
            key_dimension,
            source_dtype,
            stream: context.stream,
            memory_budget_bytes: checkpoint.memory_budget_bytes,
            resident_bytes,
            semantic_digest_sha256,
            source_exact,
        };
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::Validation,
            phase_hook,
        )?;
        resource.validate(context.cancellation)?;
        gligen_checkpoint(context.cancellation, NativeGligenPhase::Return, phase_hook)?;
        Ok(resource)
    }

    pub const fn identifier(&self) -> &'static str {
        "gligen"
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
        self.source_exact
    }

    pub const fn key_dimension(&self) -> usize {
        self.key_dimension
    }

    pub fn fuser_locations(&self) -> &[NativeGligenFuserLocation] {
        &self.fusers
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, NativeGligenError> {
        gligen_resident_owned_bytes(
            &self.artifact_sha256,
            self.artifact_sha256.capacity(),
            &self.semantic_digest_sha256,
            self.semantic_digest_sha256.capacity(),
            &self.source_state,
            &self.execution_state,
            &self.fusers,
        )
    }

    pub fn resident_tensor_allocations(&self) -> Result<Vec<(StorageId, u64)>, NativeGligenError> {
        gligen_resident_tensor_allocations(
            [&self.source_state, &self.execution_state],
            &CancellationToken::default(),
        )
    }

    pub fn reconstruct_checkpoint(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NativeGligenCheckpoint, NativeGligenError> {
        self.validate(cancellation)?;
        let specifications = gligen_state_manifest(&self.fusers, self.key_dimension)?;
        let mut ordered_state = Vec::new();
        ordered_state
            .try_reserve_exact(specifications.len())
            .map_err(|_| NativeGligenError::Allocation)?;
        for (index, specification) in specifications.iter().enumerate() {
            if index.is_multiple_of(8) {
                cancellation.check()?;
            }
            ordered_state.push((
                specification.key.clone(),
                self.source_state
                    .get(&specification.key)
                    .ok_or_else(|| NativeGligenError::MissingState(specification.key.clone()))?
                    .clone(),
            ));
        }
        cancellation.check()?;
        Ok(NativeGligenCheckpoint {
            artifact_sha256: self.artifact_sha256.clone(),
            ordered_state,
            memory_budget_bytes: self.memory_budget_bytes,
        })
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), NativeGligenError> {
        cancellation.check()?;
        if self.fusers.is_empty() {
            return Err(NativeGligenError::SemanticStateChanged);
        }
        let specifications = gligen_state_manifest(&self.fusers, self.key_dimension)?;
        let ordered_source = specifications
            .iter()
            .map(|specification| {
                self.source_state
                    .get(&specification.key)
                    .cloned()
                    .map(|tensor| (specification.key.clone(), tensor))
                    .ok_or_else(|| NativeGligenError::MissingState(specification.key.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dtype = gligen_validate_source_state(
            &ordered_source,
            &specifications,
            self.stream,
            cancellation,
        )?;
        if dtype != self.source_dtype || self.execution_state.len() != specifications.len() {
            return Err(NativeGligenError::SemanticStateChanged);
        }
        for specification in &specifications {
            let tensor = self
                .execution_state
                .get(&specification.key)
                .ok_or_else(|| NativeGligenError::MissingState(specification.key.clone()))?;
            if tensor.descriptor().shape() != specification.shape
                || tensor.descriptor().dtype() != DType::F32
                || tensor.descriptor().device() != DeviceId::CPU
                || tensor.descriptor().stream() != self.stream
                || !tensor.descriptor().is_contiguous()?
            {
                return Err(NativeGligenError::SemanticStateChanged);
            }
            gligen_validate_finite_tensor(&specification.key, tensor, cancellation)?;
        }
        gligen_validate_distinct_storage(&self.source_state, &self.execution_state)?;
        let digest = gligen_semantic_digest(
            self.source_exact,
            &self.artifact_sha256,
            self.source_dtype,
            self.key_dimension,
            &self.fusers,
            &self.source_state,
            &self.execution_state,
            &specifications,
            cancellation,
        )?;
        if digest != self.semantic_digest_sha256 {
            return Err(NativeGligenError::SemanticStateChanged);
        }
        let resident = gligen_resident_tensor_allocations(
            [&self.source_state, &self.execution_state],
            cancellation,
        )?
        .into_iter()
        .try_fold(self.resident_owned_bytes()?, |total, (_, bytes)| {
            total
                .checked_add(bytes)
                .ok_or(NativeGligenError::ShapeOverflow)
        })?;
        if resident != self.resident_bytes || resident > self.memory_budget_bytes {
            return Err(NativeGligenError::SemanticStateChanged);
        }
        cancellation.check()?;
        Ok(())
    }

    fn tensor(&self, key: &str) -> Result<&Tensor, NativeGligenError> {
        self.execution_state
            .get(key)
            .ok_or_else(|| NativeGligenError::MissingState(key.to_owned()))
    }
}

const GLIGEN_POSITION_KEYS: [&str; GLIGEN_POSITION_STATE_COUNT] = [
    "position_net.null_positive_feature",
    "position_net.null_position_feature",
    "position_net.linears.0.weight",
    "position_net.linears.0.bias",
    "position_net.linears.2.weight",
    "position_net.linears.2.bias",
    "position_net.linears.4.weight",
    "position_net.linears.4.bias",
];

const GLIGEN_FUSER_SUFFIXES: [&str; GLIGEN_FUSER_STATE_COUNT] = [
    "alpha_attn",
    "alpha_dense",
    "linear.weight",
    "linear.bias",
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "attn.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
];

fn gligen_checkpoint(
    cancellation: &CancellationToken,
    phase: NativeGligenPhase,
    phase_hook: &mut impl FnMut(NativeGligenPhase, &CancellationToken),
) -> Result<(), NativeGligenError> {
    phase_hook(phase, cancellation);
    cancellation.check()?;
    Ok(())
}

fn gligen_raw_checkpoint_preflight(
    checkpoint: &NativeGligenCheckpoint,
) -> Result<(), NativeGligenError> {
    let count = checkpoint.ordered_state.len();
    if count < GLIGEN_POSITION_STATE_COUNT + GLIGEN_FUSER_STATE_COUNT
        || count > GLIGEN_POSITION_STATE_COUNT + 60 * GLIGEN_FUSER_STATE_COUNT
        || !(count - GLIGEN_POSITION_STATE_COUNT).is_multiple_of(GLIGEN_FUSER_STATE_COUNT)
    {
        return Err(NativeGligenError::UnexpectedState(format!(
            "state count {count} is not 8 + 17*fuser_count for 1..=60 fusers"
        )));
    }
    let mut source_bytes = 0_u64;
    let mut projected_bytes = 0_u64;
    let mut maximum_scratch = 0_u64;
    let mut key_bytes = 0_u64;
    for (key, tensor) in &checkpoint.ordered_state {
        if key.is_empty() || key.len() > MAX_STATE_KEY_BYTES {
            return Err(NativeGligenError::InvalidCheckpoint(
                "state key length is invalid".to_owned(),
            ));
        }
        let source = tensor.storage_byte_len();
        let projected = tensor
            .descriptor()
            .element_count()?
            .checked_mul(DType::F32.byte_width())
            .ok_or(NativeGligenError::ShapeOverflow)?;
        source_bytes = source_bytes
            .checked_add(source)
            .ok_or(NativeGligenError::ShapeOverflow)?;
        projected_bytes = projected_bytes
            .checked_add(projected)
            .ok_or(NativeGligenError::ShapeOverflow)?;
        maximum_scratch = maximum_scratch.max(source.max(projected));
        key_bytes = key_bytes
            .checked_add(gligen_u64(key.capacity())?)
            .ok_or(NativeGligenError::ShapeOverflow)?;
    }
    let container_bytes = gligen_u64(count)?
        .checked_mul(
            u64::try_from(mem::size_of::<(String, Tensor)>())
                .map_err(|_| NativeGligenError::ShapeOverflow)?,
        )
        .and_then(|bytes| bytes.checked_mul(3))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let required = source_bytes
        .checked_add(projected_bytes)
        .and_then(|bytes| bytes.checked_add(maximum_scratch.checked_mul(2)?))
        .and_then(|bytes| bytes.checked_add(key_bytes.checked_mul(3)?))
        .and_then(|bytes| bytes.checked_add(container_bytes))
        .and_then(|bytes| {
            bytes.checked_add(u64::try_from(mem::size_of::<NativeGligenResource>()).ok()?)
        })
        .and_then(|bytes| {
            bytes.checked_add(gligen_u64(checkpoint.artifact_sha256.capacity()).ok()?)
        })
        .ok_or(NativeGligenError::ShapeOverflow)?;
    if required > checkpoint.memory_budget_bytes {
        return Err(NativeGligenError::OutOfMemory {
            required,
            budget: checkpoint.memory_budget_bytes,
        });
    }
    Ok(())
}

fn gligen_discover_state(
    state: &[(String, Tensor)],
) -> Result<
    (
        Vec<NativeGligenFuserLocation>,
        usize,
        Vec<StateSpecification>,
    ),
    NativeGligenError,
> {
    for (index, expected) in GLIGEN_POSITION_KEYS.iter().enumerate() {
        let Some((actual, _)) = state.get(index) else {
            return Err(NativeGligenError::MissingState((*expected).to_owned()));
        };
        if actual != expected {
            return Err(NativeGligenError::UnexpectedState(format!(
                "expected {expected}, got {actual}"
            )));
        }
    }
    let key_dimension = state
        .first()
        .and_then(|(_, tensor)| tensor.descriptor().shape().first().copied())
        .ok_or_else(|| {
            NativeGligenError::InvalidCheckpoint(
                "position_net.null_positive_feature must be rank one".to_owned(),
            )
        })
        .and_then(gligen_usize)?;
    if key_dimension == 0 {
        return Err(NativeGligenError::InvalidCheckpoint(
            "GLIGEN key dimension must be nonzero".to_owned(),
        ));
    }

    let mut discovered = BTreeMap::<(NativeGligenRegion, u8), (String, HashSet<String>)>::new();
    for (key, _) in state.iter().skip(GLIGEN_POSITION_STATE_COUNT) {
        let Some((namespace, suffix)) = key.split_once(".fuser.") else {
            return Err(NativeGligenError::UnexpectedState(format!(
                "unrelated GLIGEN state {key}"
            )));
        };
        if suffix.is_empty() || suffix.contains(".fuser.") {
            return Err(NativeGligenError::UnexpectedState(format!(
                "invalid fuser key {key}"
            )));
        }
        let mut components = namespace.split('.');
        let region_name = components.next().unwrap_or_default();
        let index_name = components.next().unwrap_or_default();
        if components.next().is_none() {
            return Err(NativeGligenError::UnexpectedState(format!(
                "fuser namespace {namespace} is not structurally anchored"
            )));
        }
        let region = NativeGligenRegion::parse(region_name).ok_or_else(|| {
            NativeGligenError::UnexpectedState(format!("unknown GLIGEN region {region_name}"))
        })?;
        let block_index = index_name.parse::<u8>().map_err(|_| {
            NativeGligenError::UnexpectedState(format!("invalid GLIGEN block index {index_name}"))
        })?;
        if block_index >= 20 {
            return Err(NativeGligenError::UnexpectedState(format!(
                "GLIGEN block index {block_index} is outside 0..19"
            )));
        }
        let (existing_namespace, suffixes) = discovered
            .entry((region, block_index))
            .or_insert_with(|| (namespace.to_owned(), HashSet::new()));
        if existing_namespace != namespace {
            return Err(NativeGligenError::UnexpectedState(format!(
                "multiple fuser namespaces occupy {}.{block_index}",
                region.source_name()
            )));
        }
        if !GLIGEN_FUSER_SUFFIXES.contains(&suffix) {
            return Err(NativeGligenError::UnexpectedState(format!(
                "unknown fuser suffix {suffix}"
            )));
        }
        if !suffixes.insert(suffix.to_owned()) {
            return Err(NativeGligenError::UnexpectedState(format!(
                "duplicate fuser suffix {suffix} at {namespace}"
            )));
        }
    }
    if discovered.is_empty() {
        return Err(NativeGligenError::InvalidCheckpoint(
            "GLIGEN requires at least one fuser".to_owned(),
        ));
    }
    for ((region, block_index), (_, suffixes)) in &discovered {
        if suffixes.len() != GLIGEN_FUSER_STATE_COUNT
            || GLIGEN_FUSER_SUFFIXES
                .iter()
                .any(|suffix| !suffixes.contains(*suffix))
        {
            return Err(NativeGligenError::UnexpectedState(format!(
                "partial fuser state at {}.{block_index}",
                region.source_name()
            )));
        }
    }

    let state_by_key = state
        .iter()
        .map(|(key, tensor)| (key.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let mut fusers = Vec::new();
    fusers
        .try_reserve_exact(discovered.len())
        .map_err(|_| NativeGligenError::Allocation)?;
    for region in [
        NativeGligenRegion::InputBlock,
        NativeGligenRegion::MiddleBlock,
        NativeGligenRegion::OutputBlock,
    ] {
        for block_index in 0_u8..20 {
            let Some((namespace, _)) = discovered.get(&(region, block_index)) else {
                continue;
            };
            let linear_key = format!("{namespace}.fuser.linear.weight");
            let shape = state_by_key
                .get(linear_key.as_str())
                .ok_or_else(|| NativeGligenError::MissingState(linear_key.clone()))?
                .descriptor()
                .shape();
            let [query_dimension, observed_key_dimension] = shape else {
                return Err(NativeGligenError::InvalidCheckpoint(format!(
                    "state {linear_key} must be rank two"
                )));
            };
            let query_dimension = gligen_usize(*query_dimension)?;
            let observed_key_dimension = gligen_usize(*observed_key_dimension)?;
            if query_dimension == 0 || observed_key_dimension != key_dimension {
                return Err(NativeGligenError::InvalidCheckpoint(format!(
                    "state {linear_key} has inconsistent key/query dimensions"
                )));
            }
            let (heads, head_dimension) = if key_dimension == 768 {
                if !query_dimension.is_multiple_of(8) {
                    return Err(NativeGligenError::InvalidCheckpoint(format!(
                        "query dimension {query_dimension} is not divisible by 8"
                    )));
                }
                (8, query_dimension / 8)
            } else {
                if !query_dimension.is_multiple_of(64) {
                    return Err(NativeGligenError::InvalidCheckpoint(format!(
                        "query dimension {query_dimension} is not divisible by d_head 64"
                    )));
                }
                (query_dimension / 64, 64)
            };
            fusers.push(NativeGligenFuserLocation {
                region,
                block_index,
                namespace: namespace.clone(),
                transformer_index: fusers.len(),
                query_dimension,
                key_dimension,
                heads,
                head_dimension,
            });
        }
    }
    let specifications = gligen_state_manifest(&fusers, key_dimension)?;
    Ok((fusers, key_dimension, specifications))
}

fn gligen_state_manifest(
    fusers: &[NativeGligenFuserLocation],
    key_dimension: usize,
) -> Result<Vec<StateSpecification>, NativeGligenError> {
    let key = gligen_u64(key_dimension)?;
    let input = key
        .checked_add(gligen_u64(GLIGEN_POSITION_WIDTH)?)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let hidden = gligen_u64(GLIGEN_POSITION_HIDDEN)?;
    let mut specifications = Vec::new();
    specifications
        .try_reserve_exact(
            GLIGEN_POSITION_STATE_COUNT
                .checked_add(
                    fusers
                        .len()
                        .checked_mul(GLIGEN_FUSER_STATE_COUNT)
                        .ok_or(NativeGligenError::ShapeOverflow)?,
                )
                .ok_or(NativeGligenError::ShapeOverflow)?,
        )
        .map_err(|_| NativeGligenError::Allocation)?;
    for (key_name, shape) in [
        (GLIGEN_POSITION_KEYS[0], vec![key]),
        (
            GLIGEN_POSITION_KEYS[1],
            vec![gligen_u64(GLIGEN_POSITION_WIDTH)?],
        ),
        (GLIGEN_POSITION_KEYS[2], vec![hidden, input]),
        (GLIGEN_POSITION_KEYS[3], vec![hidden]),
        (GLIGEN_POSITION_KEYS[4], vec![hidden, hidden]),
        (GLIGEN_POSITION_KEYS[5], vec![hidden]),
        (GLIGEN_POSITION_KEYS[6], vec![key, hidden]),
        (GLIGEN_POSITION_KEYS[7], vec![key]),
    ] {
        specifications.push(StateSpecification {
            key: key_name.to_owned(),
            shape,
        });
    }
    for fuser in fusers {
        if fuser.key_dimension != key_dimension {
            return Err(NativeGligenError::SemanticStateChanged);
        }
        let query = gligen_u64(fuser.query_dimension)?;
        let doubled_query = query
            .checked_mul(2)
            .ok_or(NativeGligenError::ShapeOverflow)?;
        let prefix = format!("{}.fuser", fuser.namespace);
        for (suffix, shape) in [
            ("alpha_attn", vec![]),
            ("alpha_dense", vec![]),
            ("linear.weight", vec![query, key]),
            ("linear.bias", vec![query]),
            ("attn.to_q.weight", vec![query, query]),
            ("attn.to_k.weight", vec![query, query]),
            ("attn.to_v.weight", vec![query, query]),
            ("attn.to_out.0.weight", vec![query, query]),
            ("attn.to_out.0.bias", vec![query]),
            ("ff.net.0.proj.weight", vec![doubled_query, query]),
            ("ff.net.0.proj.bias", vec![doubled_query]),
            ("ff.net.2.weight", vec![query, query]),
            ("ff.net.2.bias", vec![query]),
            ("norm1.weight", vec![query]),
            ("norm1.bias", vec![query]),
            ("norm2.weight", vec![query]),
            ("norm2.bias", vec![query]),
        ] {
            specifications.push(StateSpecification {
                key: format!("{prefix}.{suffix}"),
                shape,
            });
        }
    }
    Ok(specifications)
}

fn gligen_validate_ordered_keys(
    state: &[(String, Tensor)],
    specifications: &[StateSpecification],
) -> Result<(), NativeGligenError> {
    if state.len() != specifications.len() {
        return Err(NativeGligenError::UnexpectedState(format!(
            "expected {} entries, got {}",
            specifications.len(),
            state.len()
        )));
    }
    let mut seen = HashSet::new();
    for ((actual, _), expected) in state.iter().zip(specifications) {
        if !seen.insert(actual) {
            return Err(NativeGligenError::UnexpectedState(format!(
                "duplicate state key {actual}"
            )));
        }
        if actual != &expected.key {
            return Err(NativeGligenError::UnexpectedState(format!(
                "expected {}, got {actual}",
                expected.key
            )));
        }
    }
    Ok(())
}

fn gligen_validate_source_state(
    state: &[(String, Tensor)],
    specifications: &[StateSpecification],
    stream: StreamId,
    cancellation: &CancellationToken,
) -> Result<DType, NativeGligenError> {
    gligen_validate_ordered_keys(state, specifications)?;
    let mut dtype = None;
    let mut storages = HashSet::new();
    for (index, ((key, tensor), specification)) in state.iter().zip(specifications).enumerate() {
        if index.is_multiple_of(8) {
            cancellation.check()?;
        }
        let descriptor = tensor.descriptor();
        if descriptor.shape() != specification.shape
            || !matches!(descriptor.dtype(), DType::F32 | DType::F16 | DType::Bf16)
        {
            return Err(NativeGligenError::StateShape {
                key: key.clone(),
                expected: specification.shape.clone(),
                actual: descriptor.shape().to_vec(),
                actual_dtype: descriptor.dtype(),
            });
        }
        if descriptor.device() != DeviceId::CPU
            || descriptor.stream() != stream
            || !descriptor.is_contiguous()?
        {
            return Err(NativeGligenError::InvalidCheckpoint(format!(
                "state {key} must be contiguous CPU storage on the construction stream"
            )));
        }
        if dtype
            .replace(descriptor.dtype())
            .is_some_and(|previous| previous != descriptor.dtype())
        {
            return Err(NativeGligenError::InvalidCheckpoint(
                "all source tensors must use one storage dtype".to_owned(),
            ));
        }
        if !storages.insert(tensor.storage_id()) {
            return Err(NativeGligenError::InvalidCheckpoint(format!(
                "state {key} aliases another source tensor"
            )));
        }
        gligen_validate_finite_source_tensor(key, tensor, cancellation)?;
    }
    dtype.ok_or_else(|| NativeGligenError::InvalidCheckpoint("state is empty".to_owned()))
}

fn gligen_validate_finite_source_tensor(
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeGligenError> {
    let count = gligen_usize(tensor.descriptor().element_count()?)?;
    for index in 0..count {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        match tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(gligen_u64(index)?)?)?
        {
            DecodedScalar::Real(value) if value.is_finite() => {}
            _ => {
                return Err(NativeGligenError::InvalidCheckpoint(format!(
                    "state {key} contains a non-finite or non-real value"
                )));
            }
        }
    }
    cancellation.check()?;
    Ok(())
}

fn gligen_validate_finite_tensor(
    name: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeGligenError> {
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(NativeGligenError::InvalidInput(format!(
            "{name} must use F32 execution storage"
        )));
    }
    for (index, chunk) in tensor.contiguous_bytes()?.chunks_exact(4).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        let raw: [u8; 4] = chunk
            .try_into()
            .map_err(|_| NativeGligenError::SemanticStateChanged)?;
        if !f32::from_ne_bytes(raw).is_finite() {
            return Err(NativeGligenError::InvalidInput(format!(
                "{name} contains a non-finite value"
            )));
        }
    }
    cancellation.check()?;
    Ok(())
}

fn gligen_validate_distinct_storage(
    source: &BTreeMap<String, Tensor>,
    execution: &BTreeMap<String, Tensor>,
) -> Result<(), NativeGligenError> {
    let source_ids = source
        .values()
        .map(Tensor::storage_id)
        .collect::<HashSet<_>>();
    let mut execution_ids = HashSet::new();
    for tensor in execution.values() {
        if source_ids.contains(&tensor.storage_id()) || !execution_ids.insert(tensor.storage_id()) {
            return Err(NativeGligenError::SemanticStateChanged);
        }
    }
    Ok(())
}

fn gligen_tensor_values(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeGligenError> {
    tensor_to_f32_with_context_exact_native(backend, tensor, context).map_err(gligen_canonical)
}

fn gligen_linear_values(
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &Tensor,
    bias: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeGligenError> {
    let weight_values = gligen_tensor_values(backend, weight, context)?;
    let bias_values = bias
        .map(|bias| gligen_tensor_values(backend, bias, context))
        .transpose()?;
    let weight_shape = weight
        .descriptor()
        .shape()
        .iter()
        .map(|dimension| gligen_usize(*dimension))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(linear_with_context_exact_native(
        backend,
        input,
        input_shape,
        &weight_values,
        &weight_shape,
        bias_values.as_deref(),
        DeviceId::CPU,
        context,
    )
    .map_err(gligen_canonical)?
    .values)
}

fn gligen_linear_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeGligenError> {
    let input_shape = input
        .descriptor()
        .shape()
        .iter()
        .map(|dimension| gligen_usize(*dimension))
        .collect::<Result<Vec<_>, _>>()?;
    let input_values = gligen_tensor_values(backend, input, context)?;
    let output_values =
        gligen_linear_values(backend, &input_values, &input_shape, weight, bias, context)?;
    let output_width = weight
        .descriptor()
        .shape()
        .first()
        .copied()
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let mut output_shape = input.descriptor().shape().to_vec();
    let Some(last) = output_shape.last_mut() else {
        return Err(NativeGligenError::ShapeOverflow);
    };
    *last = output_width;
    tensor_from_f32_with_context_exact_native(
        backend,
        &output_shape,
        &output_values,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(gligen_canonical)
}

fn gligen_layer_norm_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeGligenError> {
    let shape = input
        .descriptor()
        .shape()
        .iter()
        .map(|dimension| gligen_usize(*dimension))
        .collect::<Result<Vec<_>, _>>()?;
    let width = *shape.last().ok_or(NativeGligenError::ShapeOverflow)?;
    let input_values = gligen_tensor_values(backend, input, context)?;
    let weight_values = gligen_tensor_values(backend, weight, context)?;
    let bias_values = gligen_tensor_values(backend, bias, context)?;
    let output = layer_norm_with_context_exact_native(
        backend,
        &input_values,
        &shape,
        &[width],
        Some(&weight_values),
        Some(&bias_values),
        LAYER_NORM_EPSILON,
        DeviceId::CPU,
        context,
    )
    .map_err(gligen_canonical)?;
    tensor_from_f32_with_context_exact_native(
        backend,
        input.descriptor().shape(),
        &output,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(gligen_canonical)
}

fn gligen_tanh_scalar(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<f32, NativeGligenError> {
    if !input.descriptor().shape().is_empty() {
        return Err(NativeGligenError::SemanticStateChanged);
    }
    let value =
        tanh_with_context_exact_native(backend, input, context).map_err(gligen_canonical)?;
    let bytes = value.linear_element_bytes(0)?;
    match value.descriptor().dtype().decode_scalar(bytes)? {
        DecodedScalar::Real(value) if value.is_finite() => Ok(value as f32),
        _ => Err(NativeGligenError::SemanticStateChanged),
    }
}

fn gligen_attention_workspace_bytes(
    batch: usize,
    heads: usize,
    tokens: usize,
) -> Result<usize, NativeGligenError> {
    batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(tokens))
        .and_then(|value| value.checked_mul(tokens))
        .and_then(|value| value.checked_mul(8))
        .ok_or(NativeGligenError::ShapeOverflow)
}

fn gligen_construction_peak(
    artifact: &str,
    artifact_capacity: usize,
    ordered_state: &[(String, Tensor)],
    specifications: &[StateSpecification],
    fusers: &[NativeGligenFuserLocation],
) -> Result<u64, NativeGligenError> {
    let source_bytes = ordered_state.iter().try_fold(0_u64, |total, (_, tensor)| {
        total
            .checked_add(tensor.storage_byte_len())
            .ok_or(NativeGligenError::ShapeOverflow)
    })?;
    let projected_bytes = specifications
        .iter()
        .try_fold(0_u64, |total, specification| {
            total
                .checked_add(gligen_tensor_bytes(&specification.shape, DType::F32)?)
                .ok_or(NativeGligenError::ShapeOverflow)
        })?;
    let maximum_scratch = specifications
        .iter()
        .map(|specification| gligen_tensor_bytes(&specification.shape, DType::F32))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0)
        .checked_mul(2)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let key_bytes = ordered_state.iter().try_fold(0_u64, |total, (key, _)| {
        total
            .checked_add(gligen_u64(key.capacity())?)
            .ok_or(NativeGligenError::ShapeOverflow)
    })?;
    let fuser_owned = fusers.iter().try_fold(0_u64, |total, fuser| {
        total
            .checked_add(gligen_u64(fuser.namespace.capacity())?)
            .and_then(|bytes| {
                bytes.checked_add(u64::try_from(mem::size_of::<NativeGligenFuserLocation>()).ok()?)
            })
            .ok_or(NativeGligenError::ShapeOverflow)
    })?;
    let container_bytes = gligen_u64(ordered_state.len())?
        .checked_mul(
            u64::try_from(mem::size_of::<(String, Tensor)>())
                .map_err(|_| NativeGligenError::ShapeOverflow)?,
        )
        .and_then(|bytes| bytes.checked_mul(3))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    source_bytes
        .checked_add(projected_bytes)
        .and_then(|bytes| bytes.checked_add(maximum_scratch))
        .and_then(|bytes| bytes.checked_add(key_bytes.checked_mul(3)?))
        .and_then(|bytes| bytes.checked_add(fuser_owned))
        .and_then(|bytes| bytes.checked_add(container_bytes))
        .and_then(|bytes| {
            bytes.checked_add(u64::try_from(mem::size_of::<NativeGligenResource>()).ok()?)
        })
        .and_then(|bytes| {
            bytes.checked_add(gligen_u64(artifact_capacity.max(artifact.len())).ok()?)
        })
        .ok_or(NativeGligenError::ShapeOverflow)
}

fn gligen_prepare_peak(
    resource: &NativeGligenResource,
    batch: usize,
    position_count: usize,
) -> Result<(), NativeGligenError> {
    let rows = batch
        .checked_mul(GLIGEN_MAX_OBJECTS)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let input_width = resource
        .key_dimension
        .checked_add(GLIGEN_POSITION_WIDTH)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let scalar_bytes = 4_u64;
    let positions = gligen_u64(rows)?
        .checked_mul(gligen_u64(input_width)?)
        .and_then(|value| value.checked_mul(scalar_bytes))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let hidden = gligen_u64(rows)?
        .checked_mul(gligen_u64(GLIGEN_POSITION_HIDDEN)?)
        .and_then(|value| value.checked_mul(scalar_bytes))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let objects = gligen_u64(rows)?
        .checked_mul(gligen_u64(resource.key_dimension)?)
        .and_then(|value| value.checked_mul(scalar_bytes))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let decoded_embedding = gligen_u64(position_count)?
        .checked_mul(gligen_u64(resource.key_dimension)?)
        .and_then(|value| value.checked_mul(scalar_bytes))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let decoded_weight = resource
        .execution_state
        .values()
        .map(Tensor::storage_byte_len)
        .max()
        .unwrap_or(0)
        .checked_mul(2)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let phase = positions
        .checked_mul(3)
        .and_then(|value| value.checked_add(hidden.checked_mul(4)?))
        .and_then(|value| value.checked_add(objects.checked_mul(2)?))
        .and_then(|value| value.checked_add(decoded_embedding))
        .and_then(|value| value.checked_add(decoded_weight))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    gligen_require_budget(resource, phase)
}

fn gligen_apply_peak(
    resource: &NativeGligenResource,
    fuser: &NativeGligenFuserLocation,
    batch: usize,
    visual_tokens: usize,
) -> Result<(), NativeGligenError> {
    let total_tokens = visual_tokens
        .checked_add(GLIGEN_MAX_OBJECTS)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let scalar_bytes = 4_u64;
    let visual = gligen_u64(batch)?
        .checked_mul(gligen_u64(visual_tokens)?)
        .and_then(|value| value.checked_mul(gligen_u64(fuser.query_dimension).ok()?))
        .and_then(|value| value.checked_mul(scalar_bytes))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let combined = gligen_u64(batch)?
        .checked_mul(gligen_u64(total_tokens)?)
        .and_then(|value| value.checked_mul(gligen_u64(fuser.query_dimension).ok()?))
        .and_then(|value| value.checked_mul(scalar_bytes))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let attention = gligen_u64(batch)?
        .checked_mul(gligen_u64(fuser.heads)?)
        .and_then(|value| value.checked_mul(gligen_u64(total_tokens).ok()?))
        .and_then(|value| value.checked_mul(gligen_u64(total_tokens).ok()?))
        .and_then(|value| value.checked_mul(8))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let geglu = visual
        .checked_mul(2)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let decoded_weight = resource
        .execution_state
        .values()
        .map(Tensor::storage_byte_len)
        .max()
        .unwrap_or(0)
        .checked_mul(2)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let phase = combined
        .checked_mul(12)
        .and_then(|value| value.checked_add(attention.checked_mul(2)?))
        .and_then(|value| value.checked_add(visual.checked_mul(8)?))
        .and_then(|value| value.checked_add(geglu.checked_mul(2)?))
        .and_then(|value| value.checked_add(decoded_weight))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    gligen_require_budget(resource, phase)
}

fn gligen_require_budget(
    resource: &NativeGligenResource,
    phase_bytes: u64,
) -> Result<(), NativeGligenError> {
    let required = resource
        .resident_bytes
        .checked_add(phase_bytes)
        .ok_or(NativeGligenError::ShapeOverflow)?;
    if required > resource.memory_budget_bytes {
        return Err(NativeGligenError::OutOfMemory {
            required,
            budget: resource.memory_budget_bytes,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gligen_semantic_digest(
    source_exact: bool,
    artifact: &str,
    source_dtype: DType,
    key_dimension: usize,
    fusers: &[NativeGligenFuserLocation],
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
    specifications: &[StateSpecification],
    cancellation: &CancellationToken,
) -> Result<String, NativeGligenError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.gligen-resource.v1\0");
    hasher.update([u8::from(source_exact)]);
    hasher.update(artifact.as_bytes());
    hasher.update(source_dtype.catalog_name().as_bytes());
    hasher.update(gligen_u64(key_dimension)?.to_le_bytes());
    hasher.update(gligen_u64(fusers.len())?.to_le_bytes());
    for fuser in fusers {
        cancellation.check()?;
        hasher.update([match fuser.region {
            NativeGligenRegion::InputBlock => 1,
            NativeGligenRegion::MiddleBlock => 2,
            NativeGligenRegion::OutputBlock => 3,
        }]);
        hasher.update([fuser.block_index]);
        hasher.update(gligen_u64(fuser.transformer_index)?.to_le_bytes());
        hasher.update(gligen_u64(fuser.namespace.len())?.to_le_bytes());
        hasher.update(fuser.namespace.as_bytes());
        for value in [
            fuser.query_dimension,
            fuser.key_dimension,
            fuser.heads,
            fuser.head_dimension,
        ] {
            hasher.update(gligen_u64(value)?.to_le_bytes());
        }
    }
    for (index, specification) in specifications.iter().enumerate() {
        if index.is_multiple_of(8) {
            cancellation.check()?;
        }
        gligen_hash_tensor(
            &mut hasher,
            &specification.key,
            source_state
                .get(&specification.key)
                .ok_or_else(|| NativeGligenError::MissingState(specification.key.clone()))?,
            cancellation,
        )?;
        gligen_hash_tensor(
            &mut hasher,
            &specification.key,
            execution_state
                .get(&specification.key)
                .ok_or_else(|| NativeGligenError::MissingState(specification.key.clone()))?,
            cancellation,
        )?;
    }
    cancellation.check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn gligen_hash_tensor(
    hasher: &mut Sha256,
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeGligenError> {
    hasher.update(gligen_u64(key.len())?.to_le_bytes());
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

fn gligen_resident_owned_bytes(
    artifact: &str,
    artifact_capacity: usize,
    digest: &str,
    digest_capacity: usize,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
    fusers: &[NativeGligenFuserLocation],
) -> Result<u64, NativeGligenError> {
    let base = u64::try_from(mem::size_of::<NativeGligenResource>())
        .map_err(|_| NativeGligenError::ShapeOverflow)?;
    let keys =
        source_state
            .keys()
            .chain(execution_state.keys())
            .try_fold(0_u64, |total, key| {
                total
                    .checked_add(gligen_u64(key.capacity())?)
                    .ok_or(NativeGligenError::ShapeOverflow)
            })?;
    let map_nodes = gligen_u64(source_state.len())?
        .checked_mul(
            u64::try_from(mem::size_of::<(String, Tensor)>())
                .map_err(|_| NativeGligenError::ShapeOverflow)?
                .checked_add(64)
                .ok_or(NativeGligenError::ShapeOverflow)?,
        )
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or(NativeGligenError::ShapeOverflow)?;
    let fuser_owned = fusers.iter().try_fold(0_u64, |total, fuser| {
        total
            .checked_add(gligen_u64(fuser.namespace.capacity())?)
            .and_then(|bytes| {
                bytes.checked_add(u64::try_from(mem::size_of::<NativeGligenFuserLocation>()).ok()?)
            })
            .ok_or(NativeGligenError::ShapeOverflow)
    })?;
    base.checked_add(gligen_u64(artifact_capacity.max(artifact.len()))?)
        .and_then(|bytes| bytes.checked_add(gligen_u64(digest_capacity.max(digest.len())).ok()?))
        .and_then(|bytes| bytes.checked_add(keys))
        .and_then(|bytes| bytes.checked_add(map_nodes))
        .and_then(|bytes| bytes.checked_add(fuser_owned))
        .ok_or(NativeGligenError::ShapeOverflow)
}

fn gligen_resident_tensor_allocations<'a>(
    maps: impl IntoIterator<Item = &'a BTreeMap<String, Tensor>>,
    cancellation: &CancellationToken,
) -> Result<Vec<(StorageId, u64)>, NativeGligenError> {
    let mut allocations = Vec::new();
    for map in maps {
        for (index, tensor) in map.values().enumerate() {
            if index.is_multiple_of(16) {
                cancellation.check()?;
            }
            let storage_id = tensor.storage_id();
            let bytes = tensor.storage_byte_len();
            if let Some((_, existing)) = allocations
                .iter()
                .find(|(existing_id, _)| *existing_id == storage_id)
            {
                if *existing != bytes {
                    return Err(NativeGligenError::SemanticStateChanged);
                }
            } else {
                allocations.push((storage_id, bytes));
            }
        }
    }
    cancellation.check()?;
    Ok(allocations)
}

fn gligen_tensor_bytes(shape: &[u64], dtype: DType) -> Result<u64, NativeGligenError> {
    shape
        .iter()
        .try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(NativeGligenError::ShapeOverflow)
        })?
        .checked_mul(dtype.byte_width())
        .ok_or(NativeGligenError::ShapeOverflow)
}

fn gligen_validate_sha256(value: &str) -> Result<(), NativeGligenError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NativeGligenError::InvalidCheckpoint(
            "artifact identity must be canonical lowercase SHA-256".to_owned(),
        ));
    }
    Ok(())
}

fn gligen_canonical(error: impl std::error::Error + 'static) -> NativeGligenError {
    let mut source: &(dyn std::error::Error + 'static) = &error;
    loop {
        if is_cancellation_error(source) {
            return NativeGligenError::Cancelled;
        }
        let Some(next) = source.source() else {
            break;
        };
        source = next;
    }
    NativeGligenError::Canonical(error.to_string())
}

fn gligen_u64(value: usize) -> Result<u64, NativeGligenError> {
    u64::try_from(value).map_err(|_| NativeGligenError::ShapeOverflow)
}

fn gligen_usize(value: u64) -> Result<usize, NativeGligenError> {
    usize::try_from(value).map_err(|_| NativeGligenError::ShapeOverflow)
}

fn gligen_i64(value: impl TryInto<i64>) -> Result<i64, NativeGligenError> {
    value
        .try_into()
        .map_err(|_| NativeGligenError::ShapeOverflow)
}

impl NativeGligenResource {
    pub fn prepare_positions(
        &self,
        backend: &CpuBackend,
        latent_shape: [u64; 4],
        positions: &[NativeGligenPositionParameter],
        context: &ExecutionContext<'_>,
    ) -> Result<NativeGligenPreparedPositions, NativeGligenError> {
        self.prepare_positions_checked(backend, latent_shape, positions, context, &mut |_, _| {})
    }

    fn prepare_positions_checked(
        &self,
        backend: &CpuBackend,
        latent_shape: [u64; 4],
        positions: &[NativeGligenPositionParameter],
        context: &ExecutionContext<'_>,
        phase_hook: &mut impl FnMut(NativeGligenPhase, &CancellationToken),
    ) -> Result<NativeGligenPreparedPositions, NativeGligenError> {
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::PositionAdmission,
            phase_hook,
        )?;
        self.validate(context.cancellation)?;
        let [batch, channels, latent_height, latent_width] = latent_shape;
        if batch == 0 || channels == 0 || latent_height == 0 || latent_width == 0 {
            return Err(NativeGligenError::InvalidInput(
                "latent shape must be nonzero [B,C,H,W]".to_owned(),
            ));
        }
        if positions.len() > GLIGEN_MAX_OBJECTS {
            return Err(NativeGligenError::InvalidInput(format!(
                "GLIGEN accepts at most {GLIGEN_MAX_OBJECTS} position entries"
            )));
        }
        let batch = gligen_usize(batch)?;
        gligen_prepare_peak(self, batch, positions.len())?;
        for position in positions {
            let descriptor = position.embedding.descriptor();
            if descriptor.shape() != [1, gligen_u64(self.key_dimension)?]
                || descriptor.dtype() != DType::F32
                || descriptor.device() != DeviceId::CPU
                || descriptor.stream() != context.stream
                || !descriptor.is_contiguous()?
            {
                return Err(NativeGligenError::InvalidInput(format!(
                    "position embedding must be contiguous CPU F32 [1, {}] state on the execution stream",
                    self.key_dimension
                )));
            }
            if ![position.height, position.width, position.y, position.x]
                .into_iter()
                .all(f32::is_finite)
            {
                return Err(NativeGligenError::InvalidInput(
                    "position coordinates must be finite".to_owned(),
                ));
            }
            gligen_validate_finite_tensor(
                "position embedding",
                &position.embedding,
                context.cancellation,
            )?;
        }

        gligen_checkpoint(context.cancellation, NativeGligenPhase::Fourier, phase_hook)?;
        let mut boxes = vec![[0.0_f32; 4]; GLIGEN_MAX_OBJECTS];
        let mut embedding_rows = vec![vec![0.0_f32; self.key_dimension]; GLIGEN_MAX_OBJECTS];
        for (index, position) in positions.iter().enumerate() {
            context.check()?;
            boxes[index] = [
                position.x / latent_width as f32,
                position.y / latent_height as f32,
                (position.x + position.width) / latent_width as f32,
                (position.y + position.height) / latent_height as f32,
            ];
            embedding_rows[index] =
                tensor_to_f32_with_context_exact_native(backend, &position.embedding, context)
                    .map_err(gligen_canonical)?;
        }
        let mut fourier_rows = Vec::new();
        fourier_rows
            .try_reserve_exact(GLIGEN_MAX_OBJECTS)
            .map_err(|_| NativeGligenError::Allocation)?;
        for (index, box_values) in boxes.iter().enumerate() {
            if index.is_multiple_of(8) {
                context.check()?;
            }
            let mut row = Vec::new();
            row.try_reserve_exact(GLIGEN_POSITION_WIDTH)
                .map_err(|_| NativeGligenError::Allocation)?;
            for frequency_index in 0..GLIGEN_FOURIER_FREQUENCIES {
                let frequency = GLIGEN_FOURIER_TEMPERATURE
                    .powf(frequency_index as f32 / GLIGEN_FOURIER_FREQUENCIES as f32);
                for value in box_values {
                    row.push((frequency * value).sin());
                }
                for value in box_values {
                    row.push((frequency * value).cos());
                }
            }
            fourier_rows.push(row);
        }

        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::NullProjection,
            phase_hook,
        )?;
        let null_positive = gligen_tensor_values(
            backend,
            self.tensor("position_net.null_positive_feature")?,
            context,
        )?;
        let null_position = gligen_tensor_values(
            backend,
            self.tensor("position_net.null_position_feature")?,
            context,
        )?;
        let input_width = self
            .key_dimension
            .checked_add(GLIGEN_POSITION_WIDTH)
            .ok_or(NativeGligenError::ShapeOverflow)?;
        let row_count = batch
            .checked_mul(GLIGEN_MAX_OBJECTS)
            .ok_or(NativeGligenError::ShapeOverflow)?;
        let mut input = Vec::new();
        input
            .try_reserve_exact(
                row_count
                    .checked_mul(input_width)
                    .ok_or(NativeGligenError::ShapeOverflow)?,
            )
            .map_err(|_| NativeGligenError::Allocation)?;
        for batch_index in 0..batch {
            for object_index in 0..GLIGEN_MAX_OBJECTS {
                if (batch_index * GLIGEN_MAX_OBJECTS + object_index).is_multiple_of(8) {
                    context.check()?;
                }
                if object_index < positions.len() {
                    input.extend_from_slice(&embedding_rows[object_index]);
                    input.extend_from_slice(&fourier_rows[object_index]);
                } else {
                    input.extend_from_slice(&null_positive);
                    input.extend_from_slice(&null_position);
                }
            }
        }

        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::PositionLinearOne,
            phase_hook,
        )?;
        let hidden = gligen_linear_values(
            backend,
            &input,
            &[row_count, input_width],
            self.tensor("position_net.linears.0.weight")?,
            Some(self.tensor("position_net.linears.0.bias")?),
            context,
        )?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::PositionSiluOne,
            phase_hook,
        )?;
        let hidden = silu_with_context_exact_native(backend, &hidden, DeviceId::CPU, context)
            .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::PositionLinearTwo,
            phase_hook,
        )?;
        let hidden = gligen_linear_values(
            backend,
            &hidden,
            &[row_count, GLIGEN_POSITION_HIDDEN],
            self.tensor("position_net.linears.2.weight")?,
            Some(self.tensor("position_net.linears.2.bias")?),
            context,
        )?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::PositionSiluTwo,
            phase_hook,
        )?;
        let hidden = silu_with_context_exact_native(backend, &hidden, DeviceId::CPU, context)
            .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::PositionLinearThree,
            phase_hook,
        )?;
        let objects = gligen_linear_values(
            backend,
            &hidden,
            &[row_count, GLIGEN_POSITION_HIDDEN],
            self.tensor("position_net.linears.4.weight")?,
            Some(self.tensor("position_net.linears.4.bias")?),
            context,
        )?;
        let objects = tensor_from_f32_with_context_exact_native(
            backend,
            &[
                gligen_u64(batch)?,
                gligen_u64(GLIGEN_MAX_OBJECTS)?,
                gligen_u64(self.key_dimension)?,
            ],
            &objects,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::PositionReturn,
            phase_hook,
        )?;
        Ok(NativeGligenPreparedPositions {
            resource_semantic_digest_sha256: self.semantic_digest_sha256.clone(),
            batch,
            device: DeviceId::CPU,
            stream: context.stream,
            objects,
        })
    }

    pub fn apply_fuser(
        &self,
        backend: &CpuBackend,
        transformer_index: usize,
        visual: &Tensor,
        prepared: &NativeGligenPreparedPositions,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeGligenError> {
        self.apply_fuser_checked(
            backend,
            transformer_index,
            visual,
            prepared,
            context,
            &mut |_, _| {},
        )
    }

    fn apply_fuser_checked(
        &self,
        backend: &CpuBackend,
        transformer_index: usize,
        visual: &Tensor,
        prepared: &NativeGligenPreparedPositions,
        context: &ExecutionContext<'_>,
        phase_hook: &mut impl FnMut(NativeGligenPhase, &CancellationToken),
    ) -> Result<Tensor, NativeGligenError> {
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::ApplyAdmission,
            phase_hook,
        )?;
        self.validate(context.cancellation)?;
        let fuser = self.fusers.get(transformer_index).ok_or_else(|| {
            NativeGligenError::InvalidInput(format!(
                "transformer_index {transformer_index} is out of range"
            ))
        })?;
        let [batch, visual_tokens, width] = visual.descriptor().shape() else {
            return Err(NativeGligenError::InvalidInput(
                "visual input must have [B,V,query_dim] shape".to_owned(),
            ));
        };
        if *batch == 0
            || *visual_tokens == 0
            || *width != gligen_u64(fuser.query_dimension)?
            || visual.descriptor().dtype() != DType::F32
            || visual.descriptor().device() != DeviceId::CPU
            || visual.descriptor().stream() != context.stream
            || !visual.descriptor().is_contiguous()?
        {
            return Err(NativeGligenError::InvalidInput(format!(
                "visual input must be contiguous CPU F32 [B,V,{}] state on the execution stream",
                fuser.query_dimension
            )));
        }
        let batch = gligen_usize(*batch)?;
        let visual_tokens = gligen_usize(*visual_tokens)?;
        if prepared.resource_semantic_digest_sha256 != self.semantic_digest_sha256
            || prepared.batch != batch
            || prepared.device != DeviceId::CPU
            || prepared.stream != context.stream
            || prepared.objects.descriptor().shape()
                != [
                    gligen_u64(batch)?,
                    gligen_u64(GLIGEN_MAX_OBJECTS)?,
                    gligen_u64(self.key_dimension)?,
                ]
            || prepared.objects.descriptor().dtype() != DType::F32
            || prepared.objects.descriptor().device() != DeviceId::CPU
            || prepared.objects.descriptor().stream() != context.stream
            || !prepared.objects.descriptor().is_contiguous()?
        {
            return Err(NativeGligenError::InvalidInput(
                "prepared positions do not belong to this resource, batch, device, and stream"
                    .to_owned(),
            ));
        }
        gligen_validate_finite_tensor("visual input", visual, context.cancellation)?;
        gligen_apply_peak(self, fuser, batch, visual_tokens)?;
        let prefix = format!("{}.fuser", fuser.namespace);

        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::ContextProjection,
            phase_hook,
        )?;
        let objects = gligen_linear_tensor(
            backend,
            &prepared.objects,
            self.tensor(&format!("{prefix}.linear.weight"))?,
            Some(self.tensor(&format!("{prefix}.linear.bias"))?),
            context,
        )?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::Concatenate,
            phase_hook,
        )?;
        let combined =
            torch_cat_with_context_exact_native(backend, &[visual.clone(), objects], 1, context)
                .map_err(gligen_canonical)?;
        let total_tokens = visual_tokens
            .checked_add(GLIGEN_MAX_OBJECTS)
            .ok_or(NativeGligenError::ShapeOverflow)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::AttentionLayerNorm,
            phase_hook,
        )?;
        let normalized = gligen_layer_norm_tensor(
            backend,
            &combined,
            self.tensor(&format!("{prefix}.norm1.weight"))?,
            self.tensor(&format!("{prefix}.norm1.bias"))?,
            context,
        )?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::AttentionQuery,
            phase_hook,
        )?;
        let query = gligen_linear_tensor(
            backend,
            &normalized,
            self.tensor(&format!("{prefix}.attn.to_q.weight"))?,
            None,
            context,
        )?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::AttentionKey,
            phase_hook,
        )?;
        let key = gligen_linear_tensor(
            backend,
            &normalized,
            self.tensor(&format!("{prefix}.attn.to_k.weight"))?,
            None,
            context,
        )?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::AttentionValue,
            phase_hook,
        )?;
        let value = gligen_linear_tensor(
            backend,
            &normalized,
            self.tensor(&format!("{prefix}.attn.to_v.weight"))?,
            None,
            context,
        )?;
        let attention_shape = [
            gligen_i64(batch)?,
            gligen_i64(total_tokens)?,
            gligen_i64(fuser.heads)?,
            gligen_i64(fuser.head_dimension)?,
        ];
        let query =
            torch_reshape_with_context_exact_native(backend, &query, &attention_shape, context)
                .map_err(gligen_canonical)?;
        let key = torch_reshape_with_context_exact_native(backend, &key, &attention_shape, context)
            .map_err(gligen_canonical)?;
        let value =
            torch_reshape_with_context_exact_native(backend, &value, &attention_shape, context)
                .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::AttentionSdp,
            phase_hook,
        )?;
        let workspace_limit_bytes =
            gligen_attention_workspace_bytes(batch, fuser.heads, total_tokens)?;
        let attention = scaled_dot_product_attention_tensor_with_context(
            backend,
            AttentionRequest {
                backend: AttentionBackend::PytorchSdp,
                fallback: AttentionFallbackPolicy::AllowExactNative,
                batch,
                query_tokens: total_tokens,
                key_tokens: total_tokens,
                heads: fuser.heads,
                head_dimension: fuser.head_dimension,
                value_dimension: fuser.head_dimension,
                scale: None,
                workspace_limit_bytes,
            },
            &query,
            &key,
            &value,
            None,
            context,
        )
        .map_err(gligen_canonical)?;
        let attention = torch_reshape_with_context_exact_native(
            backend,
            &attention,
            &[
                gligen_i64(batch)?,
                gligen_i64(total_tokens)?,
                gligen_i64(fuser.query_dimension)?,
            ],
            context,
        )
        .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::AttentionOutput,
            phase_hook,
        )?;
        let attention = gligen_linear_tensor(
            backend,
            &attention,
            self.tensor(&format!("{prefix}.attn.to_out.0.weight"))?,
            Some(self.tensor(&format!("{prefix}.attn.to_out.0.bias"))?),
            context,
        )?;
        let attention = narrow_method_exact_native(
            &attention,
            1,
            0,
            gligen_u64(visual_tokens)?,
            context.cancellation,
        )
        .map_err(gligen_canonical)?;
        let alpha_attention = gligen_tanh_scalar(
            backend,
            self.tensor(&format!("{prefix}.alpha_attn"))?,
            context,
        )?;
        let attention = real_multiply_with_context_exact_native(
            backend,
            &attention,
            ElementwiseOperand::Scalar(Scalar::Float(f64::from(alpha_attention))),
            context,
        )
        .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::AttentionResidual,
            phase_hook,
        )?;
        let mut output = real_add_with_context_exact_native(backend, visual, &attention, context)
            .map_err(gligen_canonical)?;

        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::DenseLayerNorm,
            phase_hook,
        )?;
        let normalized = gligen_layer_norm_tensor(
            backend,
            &output,
            self.tensor(&format!("{prefix}.norm2.weight"))?,
            self.tensor(&format!("{prefix}.norm2.bias"))?,
            context,
        )?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::GegluProjection,
            phase_hook,
        )?;
        let projected = gligen_linear_tensor(
            backend,
            &normalized,
            self.tensor(&format!("{prefix}.ff.net.0.proj.weight"))?,
            Some(self.tensor(&format!("{prefix}.ff.net.0.proj.bias"))?),
            context,
        )?;
        let value = narrow_method_exact_native(
            &projected,
            2,
            0,
            gligen_u64(fuser.query_dimension)?,
            context.cancellation,
        )
        .map_err(gligen_canonical)?;
        let gate = narrow_method_exact_native(
            &projected,
            2,
            gligen_i64(fuser.query_dimension)?,
            gligen_u64(fuser.query_dimension)?,
            context.cancellation,
        )
        .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::GegluActivation,
            phase_hook,
        )?;
        let gate_values = gligen_tensor_values(backend, &gate, context)?;
        let gate_values = gelu_with_context_exact_native(
            backend,
            &gate_values,
            GeluApproximation::None,
            DeviceId::CPU,
            context,
        )
        .map_err(gligen_canonical)?;
        let gate = tensor_from_f32_with_context_exact_native(
            backend,
            value.descriptor().shape(),
            &gate_values,
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(gligen_canonical)?;
        let dense = sim_tensor::generated_elementwise_or_runtime_operation_09::mul_with_context_exact_native(
            backend,
            &value,
            &gate,
            context,
        )
        .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::DenseOutput,
            phase_hook,
        )?;
        let dense = gligen_linear_tensor(
            backend,
            &dense,
            self.tensor(&format!("{prefix}.ff.net.2.weight"))?,
            Some(self.tensor(&format!("{prefix}.ff.net.2.bias"))?),
            context,
        )?;
        let alpha_dense = gligen_tanh_scalar(
            backend,
            self.tensor(&format!("{prefix}.alpha_dense"))?,
            context,
        )?;
        let dense = real_multiply_with_context_exact_native(
            backend,
            &dense,
            ElementwiseOperand::Scalar(Scalar::Float(f64::from(alpha_dense))),
            context,
        )
        .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::DenseResidual,
            phase_hook,
        )?;
        output = real_add_with_context_exact_native(backend, &output, &dense, context)
            .map_err(gligen_canonical)?;
        gligen_checkpoint(
            context.cancellation,
            NativeGligenPhase::ApplyReturn,
            phase_hook,
        )?;
        gligen_validate_finite_tensor("fuser output", &output, context.cancellation)?;
        Ok(output)
    }
}

#[cfg(test)]
mod photomaker_tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, TensorDescriptor};

    const TEST_MEMORY: u64 = 64 * 1024 * 1024;

    fn fixture_checkpoint(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativePhotoMakerCheckpoint, NativePhotoMakerError> {
        let configuration = photo_reduced_configuration();
        let specifications = photo_state_manifest(configuration)?;
        let mut ordered_entries = Vec::new();
        ordered_entries
            .try_reserve_exact(specifications.len())
            .map_err(|_| NativePhotoMakerError::Allocation)?;
        for (state_index, specification) in specifications.into_iter().enumerate() {
            context.check()?;
            let count = pm_usize(specification.shape.iter().try_fold(
                1_u64,
                |count, dimension| {
                    count
                        .checked_mul(*dimension)
                        .ok_or(NativePhotoMakerError::ShapeOverflow)
                },
            )?)?;
            let values = (0..count)
                .map(|index| {
                    if specification.key.ends_with("layernorm.weight")
                        || specification.key.ends_with("layrnorm.weight")
                        || specification.key.ends_with("layer_norm1.weight")
                        || specification.key.ends_with("layer_norm2.weight")
                        || specification.key.ends_with("post_layernorm.weight")
                    {
                        1.0
                    } else {
                        ((state_index + index) % 7) as f32 * 0.001
                    }
                })
                .collect::<Vec<_>>();
            let tensor = tensor_from_f32_with_context_exact_native(
                backend,
                &specification.shape,
                &values,
                DType::F32,
                DeviceId::CPU,
                context,
            )
            .map_err(photo_canonical)?;
            ordered_entries.push(NativePhotoMakerCheckpointEntry::Tensor {
                key: specification.key,
                tensor,
            });
        }
        Ok(NativePhotoMakerCheckpoint {
            artifact_sha256: format!("{:x}", Sha256::digest(b"task395-phase-fixture")),
            ordered_entries,
            memory_budget_bytes: TEST_MEMORY,
        })
    }

    fn invocation_inputs(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<(Tensor, Tensor, Tensor), NativePhotoMakerError> {
        let image = tensor_from_f32_with_context_exact_native(
            backend,
            &[1, 2, 3, 4, 4],
            &[0.03125; 96],
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(photo_canonical)?;
        let prompt = tensor_from_f32_with_context_exact_native(
            backend,
            &[1, 4, 8],
            &[0.0625; 32],
            DType::F32,
            DeviceId::CPU,
            context,
        )
        .map_err(photo_canonical)?;
        let descriptor =
            TensorDescriptor::contiguous(vec![1, 4], DType::Bool, DeviceId::CPU, context.stream)?;
        let (mask, event) = backend.upload_bytes(descriptor, &[0, 1, 0, 1], context)?;
        backend.wait_event(event, context)?;
        Ok((image, prompt, mask))
    }

    #[test]
    fn photomaker_cancellation_phases_do_not_publish() -> Result<(), NativePhotoMakerError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY)?;
        let construction_phases = [
            NativePhotoMakerPhase::Entry,
            NativePhotoMakerPhase::Wrapper,
            NativePhotoMakerPhase::Schema,
            NativePhotoMakerPhase::SourceValidation,
            NativePhotoMakerPhase::Projection,
            NativePhotoMakerPhase::ClipVision,
            NativePhotoMakerPhase::SemanticDigest,
            NativePhotoMakerPhase::Validation,
            NativePhotoMakerPhase::Return,
        ];
        let setup_cancellation = CancellationToken::default();
        let setup_workspace = authority.authorize_workspace(TEST_MEMORY)?;
        let setup_context =
            backend.execution_context(StreamId::DEFAULT, setup_workspace, &setup_cancellation);
        let checkpoint = fixture_checkpoint(&backend, &setup_context)?;
        for target in construction_phases {
            let cancellation = CancellationToken::default();
            let workspace = authority.authorize_workspace(TEST_MEMORY)?;
            let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
            let result = NativePhotoMakerResource::checked(
                &backend,
                checkpoint.clone(),
                photo_reduced_configuration(),
                &context,
                &mut |phase, cancellation| {
                    if phase == target {
                        cancellation.cancel();
                    }
                },
            );
            assert!(matches!(result, Err(NativePhotoMakerError::Cancelled)));
        }

        let resource =
            NativePhotoMakerResource::from_reduced_fixture(&backend, checkpoint, &setup_context)?;
        let (image, prompt, mask) = invocation_inputs(&backend, &setup_context)?;
        let invocation_phases = [
            NativePhotoMakerPhase::InvocationAdmission,
            NativePhotoMakerPhase::ClipForward,
            NativePhotoMakerPhase::ExtraProjection,
            NativePhotoMakerPhase::MaskIndices,
            NativePhotoMakerPhase::Gather,
            NativePhotoMakerPhase::Mlp1LayerNorm,
            NativePhotoMakerPhase::Mlp1Linear1,
            NativePhotoMakerPhase::Mlp1Gelu,
            NativePhotoMakerPhase::Mlp1Linear2,
            NativePhotoMakerPhase::PromptResidual,
            NativePhotoMakerPhase::Mlp2LayerNorm,
            NativePhotoMakerPhase::Mlp2Linear1,
            NativePhotoMakerPhase::Mlp2Gelu,
            NativePhotoMakerPhase::Mlp2Linear2,
            NativePhotoMakerPhase::Mlp2Residual,
            NativePhotoMakerPhase::FinalLayerNorm,
            NativePhotoMakerPhase::Scatter,
            NativePhotoMakerPhase::InvocationReturn,
        ];
        for target in invocation_phases {
            let cancellation = CancellationToken::default();
            let workspace = authority.authorize_workspace(TEST_MEMORY)?;
            let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
            let result = resource.fuse_conditioning_with_phase_hook(
                &backend,
                &image,
                &prompt,
                &mask,
                &context,
                &mut |phase, cancellation| {
                    if phase == target {
                        cancellation.cancel();
                    }
                },
            );
            assert!(matches!(result, Err(NativePhotoMakerError::Cancelled)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod gligen_tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, TensorDescriptor};

    const TEST_MEMORY: u64 = 64 * 1024 * 1024;

    fn fixture_checkpoint(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeGligenCheckpoint, NativeGligenError> {
        let fuser = NativeGligenFuserLocation {
            region: NativeGligenRegion::InputBlock,
            block_index: 2,
            namespace: "input_blocks.2.transformer_blocks.0".to_owned(),
            transformer_index: 0,
            query_dimension: 64,
            key_dimension: 4,
            heads: 1,
            head_dimension: 64,
        };
        let specifications = gligen_state_manifest(&[fuser], 4)?;
        let mut ordered_state = Vec::new();
        for specification in specifications {
            let descriptor = TensorDescriptor::contiguous(
                specification.shape,
                DType::F32,
                DeviceId::CPU,
                context.stream,
            )?;
            let count = usize::try_from(descriptor.element_count()?)
                .map_err(|_| NativeGligenError::ShapeOverflow)?;
            let byte_count = count
                .checked_mul(4)
                .ok_or(NativeGligenError::ShapeOverflow)?;
            let bytes = vec![0_u8; byte_count];
            let (tensor, event) = backend.upload_bytes(descriptor, &bytes, context)?;
            backend.wait_event(event, context)?;
            ordered_state.push((specification.key, tensor));
        }
        Ok(NativeGligenCheckpoint {
            artifact_sha256: format!("{:x}", Sha256::digest("task396:phase-fixture")),
            ordered_state,
            memory_budget_bytes: TEST_MEMORY,
        })
    }

    #[test]
    fn gligen_resource_cancellation_phases_do_not_publish() -> Result<(), NativeGligenError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY)?;
        let setup_cancellation = CancellationToken::default();
        let setup_workspace = authority.authorize_workspace(TEST_MEMORY)?;
        let setup_context = backend.execution_context(
            StreamId::DEFAULT,
            setup_workspace.clone(),
            &setup_cancellation,
        );
        let checkpoint = fixture_checkpoint(&backend, &setup_context)?;
        for target in [
            NativeGligenPhase::Entry,
            NativeGligenPhase::Discovery,
            NativeGligenPhase::Schema,
            NativeGligenPhase::SourceValidation,
            NativeGligenPhase::Projection,
            NativeGligenPhase::SemanticDigest,
            NativeGligenPhase::Validation,
            NativeGligenPhase::Return,
        ] {
            let cancellation = CancellationToken::default();
            let workspace = authority.authorize_workspace(TEST_MEMORY)?;
            let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
            let result = NativeGligenResource::checked(
                &backend,
                checkpoint.clone(),
                false,
                &context,
                &mut |phase, cancellation| {
                    if phase == target {
                        cancellation.cancel();
                    }
                },
            );
            assert!(matches!(result, Err(NativeGligenError::Cancelled)));
        }

        let resource =
            NativeGligenResource::from_reduced_fixture(&backend, checkpoint, &setup_context)?;
        let embedding = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 4],
            &[0.0; 4],
            DType::F32,
            DeviceId::CPU,
            &setup_context,
        )
        .map_err(gligen_canonical)?;
        let positions = [NativeGligenPositionParameter {
            embedding,
            height: 1.0,
            width: 1.0,
            y: 0.0,
            x: 0.0,
        }];
        for target in [
            NativeGligenPhase::PositionAdmission,
            NativeGligenPhase::Fourier,
            NativeGligenPhase::NullProjection,
            NativeGligenPhase::PositionLinearOne,
            NativeGligenPhase::PositionSiluOne,
            NativeGligenPhase::PositionLinearTwo,
            NativeGligenPhase::PositionSiluTwo,
            NativeGligenPhase::PositionLinearThree,
            NativeGligenPhase::PositionReturn,
        ] {
            let cancellation = CancellationToken::default();
            let workspace = authority.authorize_workspace(TEST_MEMORY)?;
            let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
            let result = resource.prepare_positions_checked(
                &backend,
                [1, 4, 8, 8],
                &positions,
                &context,
                &mut |phase, cancellation| {
                    if phase == target {
                        cancellation.cancel();
                    }
                },
            );
            assert!(matches!(result, Err(NativeGligenError::Cancelled)));
        }

        let prepared =
            resource.prepare_positions(&backend, [1, 4, 8, 8], &positions, &setup_context)?;
        let visual = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 2, 64],
            &[0.0; 128],
            DType::F32,
            DeviceId::CPU,
            &setup_context,
        )
        .map_err(gligen_canonical)?;
        for target in [
            NativeGligenPhase::ApplyAdmission,
            NativeGligenPhase::ContextProjection,
            NativeGligenPhase::Concatenate,
            NativeGligenPhase::AttentionLayerNorm,
            NativeGligenPhase::AttentionQuery,
            NativeGligenPhase::AttentionKey,
            NativeGligenPhase::AttentionValue,
            NativeGligenPhase::AttentionSdp,
            NativeGligenPhase::AttentionOutput,
            NativeGligenPhase::AttentionResidual,
            NativeGligenPhase::DenseLayerNorm,
            NativeGligenPhase::GegluProjection,
            NativeGligenPhase::GegluActivation,
            NativeGligenPhase::DenseOutput,
            NativeGligenPhase::DenseResidual,
            NativeGligenPhase::ApplyReturn,
        ] {
            let cancellation = CancellationToken::default();
            let workspace = authority.authorize_workspace(TEST_MEMORY)?;
            let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
            let result = resource.apply_fuser_checked(
                &backend,
                0,
                &visual,
                &prepared,
                &context,
                &mut |phase, cancellation| {
                    if phase == target {
                        cancellation.cancel();
                    }
                },
            );
            assert!(matches!(result, Err(NativeGligenError::Cancelled)));
        }
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            resource.reconstruct_checkpoint(&cancelled),
            Err(NativeGligenError::Cancelled)
        ));
        assert_eq!(setup_workspace.in_use_bytes(), 0);
        Ok(())
    }
}
