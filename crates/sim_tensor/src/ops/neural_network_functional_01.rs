use crate::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId, ExecutionContext, Layout,
    Tensor, TensorBackend, TensorDescriptor, UnaryOperation, ViewAccess,
    cpu_backend::{apply_unary_scalar, binary_broadcast_shape},
    generated_accelerated_attention_kernel_01::{
        AttentionKernelRequest, AttentionMask, AttentionVjp,
    },
    generated_activation_normalization_functional_01::FunctionalError,
    generated_sim_operator_indirection_01::{
        LinearVjp, OperatorIndirectionError, TensorValues,
        linear_jvp_with_context_exact_native as canonical_linear_jvp,
        linear_vjp_with_context_exact_native as canonical_linear_vjp,
        linear_with_context_exact_native as canonical_linear,
        scaled_dot_product_attention_jvp_with_context_exact_native as canonical_attention_jvp_with_context,
        scaled_dot_product_attention_vjp_with_context_exact_native as canonical_attention_vjp_with_context,
        scaled_dot_product_attention_with_context_exact_native as canonical_attention_with_context,
    },
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseRuntimePartThreeError,
        sigmoid_jvp_with_context_exact_native as canonical_sigmoid_jvp_with_context,
        sigmoid_vjp_with_context_exact_native as canonical_sigmoid_vjp_with_context,
        sigmoid_with_context_exact_native as canonical_sigmoid_with_context,
    },
    generated_elementwise_or_runtime_operation_08::{
        ElementwiseRuntimePartEightError,
        index_select_jvp_with_context_exact_native as canonical_index_select_jvp_with_context,
        index_select_vjp_with_context_exact_native as canonical_index_select_vjp_with_context,
        index_select_with_context_exact_native as canonical_index_select_with_context,
    },
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError, NativeRearrangePlan,
        rearrange_tensor_with_context_exact_native_for_operation as canonical_rearrange_with_context,
    },
};
use sim_types::CancellationError;
use thiserror::Error;

pub const COSINE_SIMILARITY_OPERATION_ID: &str = "COMFY-TENSOR-OP-D1F3FBA7CEDA";
pub const EMBEDDING_OPERATION_ID: &str = "COMFY-TENSOR-OP-47FEBAD36339";
pub const FOLD_OPERATION_ID: &str = "COMFY-TENSOR-OP-3D194029352B";
pub const GLU_OPERATION_ID: &str = "COMFY-TENSOR-OP-BEE0237BCAD2";
pub const LINEAR_OPERATION_ID: &str = "COMFY-TENSOR-OP-D1BCDE69E795";
pub const ONE_HOT_OPERATION_ID: &str = "COMFY-TENSOR-OP-512C2ADE983A";
pub const PIXEL_SHUFFLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-56789F9ED09B";
pub const PIXEL_UNSHUFFLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-5D882CF77D16";
pub const SCALED_DOT_PRODUCT_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-C6DD68C579DD";
pub const SIGMOID_OPERATION_ID: &str = "COMFY-TENSOR-OP-5B08F2022527";
pub const SOFTPLUS_OPERATION_ID: &str = "COMFY-TENSOR-OP-13DF18F5F426";
pub const UNFOLD_OPERATION_ID: &str = "COMFY-TENSOR-OP-87C10166BCF5";

#[derive(Debug, Error)]
pub enum NeuralNetworkFunctionalError {
    #[error(transparent)]
    Tensor(#[from] crate::TensorError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Elementwise(#[from] ElementwiseRuntimePartThreeError),
    #[error(transparent)]
    Rearrangement(#[from] ExternalTensorKernelPartOneError),
    #[error(transparent)]
    Normalization(#[from] FunctionalError),
    #[error(transparent)]
    IndexSelect(#[from] ElementwiseRuntimePartEightError),
    #[error("neural-network functional execution was cancelled")]
    Cancelled,
    #[error("operation {operation} is unavailable for device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<CancellationError> for NeuralNetworkFunctionalError {
    fn from(_: CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionalTensor {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IntegerTensor {
    pub values: Vec<i64>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CosineSimilarityVjp {
    pub input_one: Vec<f32>,
    pub input_two: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddingOptions {
    pub padding_index: Option<i64>,
    pub max_norm: Option<f32>,
    pub norm_type: f32,
    pub scale_gradient_by_frequency: bool,
    pub sparse: bool,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            padding_index: None,
            max_norm: None,
            norm_type: 2.0,
            scale_gradient_by_frequency: false,
            sparse: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpatialParameters2d {
    pub kernel_size: [usize; 2],
    pub dilation: [usize; 2],
    pub padding: [usize; 2],
    pub stride: [usize; 2],
}

impl SpatialParameters2d {
    pub fn new(kernel_size: [usize; 2]) -> Self {
        Self {
            kernel_size,
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cosine_similarity_with_context_exact_native(
    _backend: &CpuBackend,
    input_one: &[f32],
    input_one_shape: &[usize],
    input_two: &[f32],
    input_two_shape: &[usize],
    dimension: i64,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    require_cpu(COSINE_SIMILARITY_OPERATION_ID, device)?;
    validate_cosine_inputs(
        input_one,
        input_one_shape,
        input_two,
        input_two_shape,
        epsilon,
    )?;
    let shape = broadcast_shape(input_one_shape, input_two_shape)?;
    let axis = normalize_axis(dimension, shape.len(), COSINE_SIMILARITY_OPERATION_ID)?;
    let output_shape = reduced_shape(&shape, axis);
    let mut output = reserved_zeros(element_count(&output_shape)?, "cosine output")?;
    for (linear, value) in output.iter_mut().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let (dot, one_norm, two_norm) = cosine_group_statistics(
            input_one,
            input_one_shape,
            input_two,
            input_two_shape,
            &shape,
            axis,
            &output_indices,
        )?;
        *value = dot / (one_norm.max(epsilon) * two_norm.max(epsilon));
    }
    context.cancellation.check()?;
    Ok(FunctionalTensor {
        values: output,
        shape: output_shape,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn cosine_similarity_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input_one: &[f32],
    input_one_shape: &[usize],
    input_two: &[f32],
    input_two_shape: &[usize],
    dimension: i64,
    epsilon: f32,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<CosineSimilarityVjp, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    require_cpu(COSINE_SIMILARITY_OPERATION_ID, device)?;
    validate_cosine_inputs(
        input_one,
        input_one_shape,
        input_two,
        input_two_shape,
        epsilon,
    )?;
    let shape = broadcast_shape(input_one_shape, input_two_shape)?;
    let axis = normalize_axis(dimension, shape.len(), COSINE_SIMILARITY_OPERATION_ID)?;
    let output_shape = reduced_shape(&shape, axis);
    require_count(
        output_gradient.len(),
        &output_shape,
        COSINE_SIMILARITY_OPERATION_ID,
        "output gradient",
    )?;
    let mut input_one_gradient =
        reserved_zeros(input_one.len(), "cosine first input gradient")?;
    let mut input_two_gradient =
        reserved_zeros(input_two.len(), "cosine second input gradient")?;
    for (linear, upstream) in output_gradient.iter().copied().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let (dot, one_norm, two_norm) = cosine_group_statistics(
            input_one,
            input_one_shape,
            input_two,
            input_two_shape,
            &shape,
            axis,
            &output_indices,
        )?;
        let one_denominator = one_norm.max(epsilon);
        let two_denominator = two_norm.max(epsilon);
        for reduced in 0..shape[axis] {
            let indices = insert_axis(&output_indices, axis, reduced);
            let one_index = broadcast_linear_index(&indices, &shape, input_one_shape)?;
            let two_index = broadcast_linear_index(&indices, &shape, input_two_shape)?;
            let one = input_one[one_index];
            let two = input_two[two_index];
            let one_derivative = two / (one_denominator * two_denominator)
                - if one_norm > epsilon {
                    dot * one / (one_norm.powi(3) * two_denominator)
                } else {
                    0.0
                };
            let two_derivative = one / (one_denominator * two_denominator)
                - if two_norm > epsilon {
                    dot * two / (two_norm.powi(3) * one_denominator)
                } else {
                    0.0
                };
            input_one_gradient[one_index] += upstream * one_derivative;
            input_two_gradient[two_index] += upstream * two_derivative;
        }
    }
    context.cancellation.check()?;
    Ok(CosineSimilarityVjp {
        input_one: input_one_gradient,
        input_two: input_two_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn cosine_similarity_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    input_one: &[f32],
    input_one_tangent: &[f32],
    input_one_shape: &[usize],
    input_two: &[f32],
    input_two_tangent: &[f32],
    input_two_shape: &[usize],
    dimension: i64,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    require_same_length(
        input_one,
        input_one_tangent,
        COSINE_SIMILARITY_OPERATION_ID,
        "first tangent",
    )?;
    require_same_length(
        input_two,
        input_two_tangent,
        COSINE_SIMILARITY_OPERATION_ID,
        "second tangent",
    )?;
    context.cancellation.check()?;
    require_cpu(COSINE_SIMILARITY_OPERATION_ID, device)?;
    validate_cosine_inputs(
        input_one,
        input_one_shape,
        input_two,
        input_two_shape,
        epsilon,
    )?;
    let shape = broadcast_shape(input_one_shape, input_two_shape)?;
    let axis = normalize_axis(dimension, shape.len(), COSINE_SIMILARITY_OPERATION_ID)?;
    let output_shape = reduced_shape(&shape, axis);
    let mut output = reserved_zeros(element_count(&output_shape)?, "cosine tangent")?;
    for (linear, output_value) in output.iter_mut().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let (dot, one_norm, two_norm) = cosine_group_statistics(
            input_one,
            input_one_shape,
            input_two,
            input_two_shape,
            &shape,
            axis,
            &output_indices,
        )?;
        let one_denominator = one_norm.max(epsilon);
        let two_denominator = two_norm.max(epsilon);
        let mut tangent = 0.0;
        for reduced in 0..shape[axis] {
            let indices = insert_axis(&output_indices, axis, reduced);
            let one_index = broadcast_linear_index(&indices, &shape, input_one_shape)?;
            let two_index = broadcast_linear_index(&indices, &shape, input_two_shape)?;
            let one = input_one[one_index];
            let two = input_two[two_index];
            let one_derivative = two / (one_denominator * two_denominator)
                - if one_norm > epsilon {
                    dot * one / (one_norm.powi(3) * two_denominator)
                } else {
                    0.0
                };
            let two_derivative = one / (one_denominator * two_denominator)
                - if two_norm > epsilon {
                    dot * two / (two_norm.powi(3) * one_denominator)
                } else {
                    0.0
                };
            tangent += one_derivative * input_one_tangent[one_index]
                + two_derivative * input_two_tangent[two_index];
        }
        *output_value = tangent;
    }
    Ok(FunctionalTensor {
        values: output,
        shape: output_shape,
    })
}

pub fn embedding_with_context_exact_native(
    backend: &CpuBackend,
    indices: &Tensor,
    weight: &mut Tensor,
    options: EmbeddingOptions,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    require_tensor_cpu(indices, EMBEDDING_OPERATION_ID)?;
    require_tensor_cpu(weight, EMBEDDING_OPERATION_ID)?;
    if indices.descriptor().dtype() != DType::I64 {
        return invalid(EMBEDDING_OPERATION_ID, "indices must use I64 dtype");
    }
    if weight.descriptor().dtype() != DType::F32 || weight.descriptor().rank() != 2 {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "weight must be a rank-two F32 tensor",
        );
    }
    if indices.descriptor().stream() != weight.descriptor().stream() {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "indices and weight must share one stream",
        );
    }
    let rows = usize::try_from(weight.descriptor().shape()[0])
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding rows"))?;
    validate_embedding_options(options, rows)?;
    let flattened_indices = flatten_tensor_with_context(backend, indices, context)?;
    let raw_indices = tensor_i64_values_with_context(backend, &flattened_indices, context)?;
    let mut normalized_indices = backend.workspace_vec(context, raw_indices.len())?;
    for index in raw_indices.iter().copied() {
        normalized_indices.try_push(normalize_embedding_index(index, rows)?)?;
    }
    drop(raw_indices);
    let mut selected_weight = fresh_contiguous_copy_with_context(backend, weight, context)?;
    if let Some(max_norm) = options.max_norm {
        renormalize_embedding_tensor_with_context(
            backend,
            context,
            &mut selected_weight,
            &normalized_indices,
            max_norm,
            options.norm_type,
        )?;
    }
    drop(normalized_indices);
    let selected = canonical_index_select_with_context(
        backend,
        &selected_weight,
        0,
        &flattened_indices,
        context,
    )?;
    let output = reshape_embedding_output(&selected, indices.descriptor().shape())?;
    context.cancellation.check()?;
    if options.max_norm.is_some() {
        *weight = selected_weight;
    }
    Ok(output)
}

pub fn embedding_vjp_with_context_exact_native(
    backend: &CpuBackend,
    indices: &Tensor,
    weight: &Tensor,
    options: EmbeddingOptions,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    require_tensor_cpu(indices, EMBEDDING_OPERATION_ID)?;
    require_tensor_cpu(weight, EMBEDDING_OPERATION_ID)?;
    require_tensor_cpu(output_gradient, EMBEDDING_OPERATION_ID)?;
    if weight.descriptor().dtype() != DType::F32 || weight.descriptor().rank() != 2 {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "weight must be a rank-two F32 tensor",
        );
    }
    let rows = usize::try_from(weight.descriptor().shape()[0])
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding rows"))?;
    validate_embedding_options(options, rows)?;
    let flattened_indices = flatten_tensor_with_context(backend, indices, context)?;
    let raw_indices = tensor_i64_values_with_context(backend, &flattened_indices, context)?;
    let mut normalized = backend.workspace_vec(context, raw_indices.len())?;
    for index in raw_indices.iter().copied() {
        normalized.try_push(normalize_embedding_index(index, rows)?)?;
    }
    drop(raw_indices);
    let mut frequencies = workspace_filled(backend, context, rows, 0_usize)?;
    for row in normalized.iter().copied() {
        frequencies[row] = frequencies[row].saturating_add(1);
    }
    let padding = normalized_padding(options.padding_index, rows)?;
    let flattened_gradient = flatten_embedding_gradient_with_context(
        backend,
        output_gradient,
        normalized.len(),
        weight.descriptor().shape()[1],
        context,
    )?;
    let mut scaled_values = tensor_f32_values_with_context(backend, &flattened_gradient, context)?;
    let width = usize::try_from(weight.descriptor().shape()[1])
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding width"))?;
    let positions = normalized.len();
    for (position, row) in normalized.iter().copied().enumerate() {
        check_periodically(position, context.cancellation)?;
        let scale = if Some(row) == padding {
            0.0
        } else if options.scale_gradient_by_frequency {
            1.0 / frequencies[row] as f32
        } else {
            1.0
        };
        for value in &mut scaled_values[position * width..(position + 1) * width] {
            *value *= scale;
        }
    }
    drop(frequencies);
    drop(normalized);
    let scaled_gradient = upload_f32_tensor_with_context(
        backend,
        &[
            u64::try_from(positions).map_err(|_| {
                NeuralNetworkFunctionalError::ShapeOverflow("embedding positions")
            })?,
            weight.descriptor().shape()[1],
        ],
        weight.descriptor().stream(),
        &scaled_values,
        context,
    )?;
    drop(scaled_values);
    Ok(canonical_index_select_vjp_with_context(
        backend,
        weight,
        0,
        &flattened_indices,
        &scaled_gradient,
        context,
    )?)
}

pub fn embedding_jvp_with_context_exact_native(
    backend: &CpuBackend,
    indices: &Tensor,
    weight: &Tensor,
    weight_tangent: &Tensor,
    options: EmbeddingOptions,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    if weight.descriptor().shape() != weight_tangent.descriptor().shape()
        || weight.descriptor().dtype() != weight_tangent.descriptor().dtype()
        || weight.descriptor().stream() != weight_tangent.descriptor().stream()
    {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "weight tangent must match weight geometry",
        );
    }
    let rows = usize::try_from(weight.descriptor().shape()[0])
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding rows"))?;
    validate_embedding_options(options, rows)?;
    let flattened_indices = flatten_tensor_with_context(backend, indices, context)?;
    let selected = canonical_index_select_jvp_with_context(
        backend,
        weight,
        weight_tangent,
        0,
        &flattened_indices,
        context,
    )?;
    reshape_embedding_output(&selected, indices.descriptor().shape())
}

fn unfold_exact_native(
    input: &[f32],
    input_shape: [usize; 4],
    parameters: SpatialParameters2d,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    cancellation.check()?;
    require_cpu(UNFOLD_OPERATION_ID, device)?;
    require_count(input.len(), &input_shape, UNFOLD_OPERATION_ID, "input")?;
    let [output_height, output_width] = checked_spatial_output(
        [input_shape[2], input_shape[3]],
        parameters,
        UNFOLD_OPERATION_ID,
    )?;
    let channels_per_column = input_shape[1]
        .checked_mul(parameters.kernel_size[0])
        .and_then(|value| value.checked_mul(parameters.kernel_size[1]))
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
            "unfold channels",
        ))?;
    let columns = output_height.checked_mul(output_width).ok_or(
        NeuralNetworkFunctionalError::ShapeOverflow("unfold columns"),
    )?;
    let shape = vec![input_shape[0], channels_per_column, columns];
    let mut output = reserved_zeros(element_count(&shape)?, "unfold output")?;
    for batch in 0..input_shape[0] {
        for channel in 0..input_shape[1] {
            for kernel_row in 0..parameters.kernel_size[0] {
                for kernel_column in 0..parameters.kernel_size[1] {
                    let output_channel = (channel * parameters.kernel_size[0] + kernel_row)
                        * parameters.kernel_size[1]
                        + kernel_column;
                    for output_row in 0..output_height {
                        for output_column in 0..output_width {
                            let column = output_row * output_width + output_column;
                            let output_index =
                                (batch * channels_per_column + output_channel) * columns + column;
                            check_periodically(output_index, cancellation)?;
                            if let Some([input_row, input_column]) = spatial_input_position(
                                output_row,
                                output_column,
                                kernel_row,
                                kernel_column,
                                parameters,
                            ) {
                                if input_row < input_shape[2] && input_column < input_shape[3] {
                                    let input_index = ((batch * input_shape[1] + channel)
                                        * input_shape[2]
                                        + input_row)
                                        * input_shape[3]
                                        + input_column;
                                    output[output_index] = input[input_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    cancellation.check()?;
    Ok(FunctionalTensor {
        values: output,
        shape,
    })
}

pub fn unfold_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: [usize; 4],
    parameters: SpatialParameters2d,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    unfold_exact_native(
        input,
        input_shape,
        parameters,
        device,
        context.cancellation,
    )
}

pub fn unfold_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &[f32],
    input_shape: [usize; 4],
    parameters: SpatialParameters2d,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    require_cpu(UNFOLD_OPERATION_ID, device)?;
    let [output_height, output_width] = checked_spatial_output(
        [input_shape[2], input_shape[3]],
        parameters,
        UNFOLD_OPERATION_ID,
    )?;
    let input_channels = input_shape[1]
        .checked_mul(parameters.kernel_size[0])
        .and_then(|value| value.checked_mul(parameters.kernel_size[1]))
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
            "unfold VJP channels",
        ))?;
    let column_count = output_height
        .checked_mul(output_width)
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
            "unfold VJP columns",
        ))?;
    require_count(
        output_gradient.len(),
        &[input_shape[0], input_channels, column_count],
        UNFOLD_OPERATION_ID,
        "output gradient",
    )?;
    fold_with_context_exact_native(
        backend,
        output_gradient,
        [input_shape[0], input_channels, column_count],
        [input_shape[2], input_shape[3]],
        parameters,
        device,
        context,
    )
}

pub fn unfold_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &[f32],
    input_shape: [usize; 4],
    parameters: SpatialParameters2d,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    unfold_with_context_exact_native(
        backend,
        input_tangent,
        input_shape,
        parameters,
        device,
        context,
    )
}

fn fold_exact_native(
    input: &[f32],
    input_shape: [usize; 3],
    output_size: [usize; 2],
    parameters: SpatialParameters2d,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    cancellation.check()?;
    require_cpu(FOLD_OPERATION_ID, device)?;
    require_count(input.len(), &input_shape, FOLD_OPERATION_ID, "input")?;
    if output_size.contains(&0) {
        return invalid(FOLD_OPERATION_ID, "output size must be positive");
    }
    let [output_height, output_width] =
        checked_spatial_output(output_size, parameters, FOLD_OPERATION_ID)?;
    let columns = output_height
        .checked_mul(output_width)
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow("fold columns"))?;
    let kernel_area = parameters.kernel_size[0]
        .checked_mul(parameters.kernel_size[1])
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
            "fold kernel area",
        ))?;
    if input_shape[2] != columns {
        return invalid(
            FOLD_OPERATION_ID,
            "input block count does not match output geometry",
        );
    }
    if kernel_area == 0 || !input_shape[1].is_multiple_of(kernel_area) {
        return invalid(
            FOLD_OPERATION_ID,
            "input channel dimension is not divisible by kernel area",
        );
    }
    let batch = input_shape[0];
    let channels = input_shape[1] / kernel_area;
    let shape = vec![batch, channels, output_size[0], output_size[1]];
    let mut output = reserved_zeros(element_count(&shape)?, "fold output")?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            for kernel_row in 0..parameters.kernel_size[0] {
                for kernel_column in 0..parameters.kernel_size[1] {
                    let input_channel = (channel * parameters.kernel_size[0] + kernel_row)
                        * parameters.kernel_size[1]
                        + kernel_column;
                    for output_row in 0..output_height {
                        for output_column in 0..output_width {
                            let column = output_row * output_width + output_column;
                            let input_index =
                                (batch_index * input_shape[1] + input_channel) * columns + column;
                            check_periodically(input_index, cancellation)?;
                            if let Some([row, column]) = spatial_input_position(
                                output_row,
                                output_column,
                                kernel_row,
                                kernel_column,
                                parameters,
                            ) {
                                if row < output_size[0] && column < output_size[1] {
                                    output[((batch_index * channels + channel)
                                        * output_size[0]
                                        + row)
                                        * output_size[1]
                                        + column] += input[input_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    cancellation.check()?;
    Ok(FunctionalTensor {
        values: output,
        shape,
    })
}

pub fn fold_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: [usize; 3],
    output_size: [usize; 2],
    parameters: SpatialParameters2d,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    fold_exact_native(
        input,
        input_shape,
        output_size,
        parameters,
        device,
        context.cancellation,
    )
}

pub fn fold_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &[f32],
    input_shape: [usize; 3],
    output_size: [usize; 2],
    parameters: SpatialParameters2d,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    require_cpu(FOLD_OPERATION_ID, device)?;
    let kernel_area = parameters.kernel_size[0]
        .checked_mul(parameters.kernel_size[1])
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
            "fold VJP kernel area",
        ))?;
    if kernel_area == 0 || !input_shape[1].is_multiple_of(kernel_area) {
        return invalid(
            FOLD_OPERATION_ID,
            "input channels must be divisible by the kernel area",
        );
    }
    let channels = input_shape[1] / kernel_area;
    unfold_with_context_exact_native(
        backend,
        output_gradient,
        [input_shape[0], channels, output_size[0], output_size[1]],
        parameters,
        device,
        context,
    )
}

pub fn fold_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &[f32],
    input_shape: [usize; 3],
    output_size: [usize; 2],
    parameters: SpatialParameters2d,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    fold_with_context_exact_native(
        backend,
        input_tangent,
        input_shape,
        output_size,
        parameters,
        device,
        context,
    )
}

fn glu_exact_native(
    input: &[f32],
    input_shape: &[usize],
    dimension: i64,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    cancellation.check()?;
    require_cpu(GLU_OPERATION_ID, device)?;
    require_count(input.len(), input_shape, GLU_OPERATION_ID, "input")?;
    let axis = normalize_axis(dimension, input_shape.len(), GLU_OPERATION_ID)?;
    if !input_shape[axis].is_multiple_of(2) {
        return invalid(GLU_OPERATION_ID, "selected dimension must have even length");
    }
    let mut shape = input_shape.to_vec();
    shape[axis] /= 2;
    let mut output = reserved_zeros(element_count(&shape)?, "GLU output")?;
    for (linear, value) in output.iter_mut().enumerate() {
        check_periodically(linear, cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let first = ravel_index(&indices, input_shape)?;
        let mut second_indices = indices;
        second_indices[axis] += shape[axis];
        let second = ravel_index(&second_indices, input_shape)?;
        *value = input[first] * sigmoid_scalar(input[second]);
    }
    cancellation.check()?;
    Ok(FunctionalTensor {
        values: output,
        shape,
    })
}

pub fn glu_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    dimension: i64,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    glu_exact_native(input, input_shape, dimension, device, context.cancellation)
}

pub fn glu_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    dimension: i64,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    require_cpu(GLU_OPERATION_ID, device)?;
    require_count(input.len(), input_shape, GLU_OPERATION_ID, "input")?;
    let axis = normalize_axis(dimension, input_shape.len(), GLU_OPERATION_ID)?;
    if !input_shape[axis].is_multiple_of(2) {
        return invalid(GLU_OPERATION_ID, "selected dimension must have even length");
    }
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] /= 2;
    require_count(
        output_gradient.len(),
        &output_shape,
        GLU_OPERATION_ID,
        "output gradient",
    )?;
    let mut gradient = reserved_zeros(input.len(), "GLU gradient")?;
    for (linear, &upstream) in output_gradient.iter().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &output_shape)?;
        let first = ravel_index(&indices, input_shape)?;
        let mut second_indices = indices;
        second_indices[axis] += output_shape[axis];
        let second = ravel_index(&second_indices, input_shape)?;
        let sigmoid = sigmoid_scalar(input[second]);
        gradient[first] = upstream * sigmoid;
        gradient[second] = upstream * input[first] * sigmoid * (1.0 - sigmoid);
    }
    context.cancellation.check()?;
    Ok(gradient)
}

pub fn glu_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    dimension: i64,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<FunctionalTensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    require_same_length(input, input_tangent, GLU_OPERATION_ID, "input tangent")?;
    require_cpu(GLU_OPERATION_ID, device)?;
    require_count(input.len(), input_shape, GLU_OPERATION_ID, "input")?;
    let axis = normalize_axis(dimension, input_shape.len(), GLU_OPERATION_ID)?;
    if !input_shape[axis].is_multiple_of(2) {
        return invalid(GLU_OPERATION_ID, "selected dimension must have even length");
    }
    let mut shape = input_shape.to_vec();
    shape[axis] /= 2;
    let mut output = reserved_zeros(element_count(&shape)?, "GLU tangent")?;
    for (linear, value) in output.iter_mut().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let first = ravel_index(&indices, input_shape)?;
        let mut second_indices = indices;
        second_indices[axis] += shape[axis];
        let second = ravel_index(&second_indices, input_shape)?;
        let sigmoid = sigmoid_scalar(input[second]);
        *value = input_tangent[first] * sigmoid
            + input[first] * sigmoid * (1.0 - sigmoid) * input_tangent[second];
    }
    Ok(FunctionalTensor {
        values: output,
        shape,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn linear_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_linear(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        device,
        context,
    )?)
}

fn one_hot_exact_native(
    indices: &[i64],
    indices_shape: &[usize],
    number_of_classes: i64,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<IntegerTensor, NeuralNetworkFunctionalError> {
    cancellation.check()?;
    require_cpu(ONE_HOT_OPERATION_ID, device)?;
    require_count(
        indices.len(),
        indices_shape,
        ONE_HOT_OPERATION_ID,
        "indices",
    )?;
    if indices.iter().any(|&index| index < 0) {
        return invalid(ONE_HOT_OPERATION_ID, "class indices must be non-negative");
    }
    let classes = if number_of_classes == -1 {
        indices
            .iter()
            .copied()
            .max()
            .ok_or_else(|| {
                invalid_error(
                    ONE_HOT_OPERATION_ID,
                    "cannot infer classes from an empty tensor",
                )
            })?
            .checked_add(1)
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
                "one-hot classes",
            ))?
    } else if number_of_classes > 0 {
        number_of_classes
    } else {
        return invalid(
            ONE_HOT_OPERATION_ID,
            "number_of_classes must be -1 or positive",
        );
    };
    if indices.iter().any(|&index| index >= classes) {
        return invalid(
            ONE_HOT_OPERATION_ID,
            "class index is outside number_of_classes",
        );
    }
    let classes = usize::try_from(classes)
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("one-hot classes"))?;
    let mut shape = indices_shape.to_vec();
    shape.push(classes);
    let mut values = vec![0_i64; element_count(&shape)?];
    for (position, &index) in indices.iter().enumerate() {
        check_periodically(position, cancellation)?;
        let index = usize::try_from(index)
            .map_err(|_| invalid_error(ONE_HOT_OPERATION_ID, "negative class index"))?;
        values[position * classes + index] = 1;
    }
    cancellation.check()?;
    Ok(IntegerTensor { values, shape })
}

#[allow(clippy::too_many_arguments)]
pub fn linear_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<LinearVjp, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_linear_vjp(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        output_gradient,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn linear_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_linear_jvp(
        input,
        input_tangent,
        input_shape,
        weight,
        weight_tangent,
        weight_shape,
        bias,
        bias_tangent,
        device,
        context,
    )?)
}

pub fn one_hot_with_context_exact_native(
    _backend: &CpuBackend,
    indices: &[i64],
    indices_shape: &[usize],
    number_of_classes: i64,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<IntegerTensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    one_hot_exact_native(
        indices,
        indices_shape,
        number_of_classes,
        device,
        context.cancellation,
    )
}

pub fn pixel_shuffle_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    upscale_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    pixel_rearrange_with_context(backend, input, upscale_factor, true, context)
}

pub fn pixel_shuffle_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    upscale_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    pixel_rearrange_with_context(backend, input, upscale_factor, true, context)
}

pub fn pixel_shuffle_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    upscale_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    let expected = pixel_rearranged_shape(input_shape, upscale_factor, true)?;
    if output_gradient.descriptor().shape() != expected {
        return invalid(
            PIXEL_SHUFFLE_OPERATION_ID,
            "output gradient shape does not match pixel-shuffle output",
        );
    }
    pixel_rearrange_with_context(backend, output_gradient, upscale_factor, false, context)
}

pub fn pixel_shuffle_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    upscale_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    pixel_shuffle_with_context_exact_native(backend, input_tangent, upscale_factor, context)
}

fn softplus_exact_native(
    input: &[f32],
    beta: f32,
    threshold: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    cancellation.check()?;
    validate_softplus(beta, threshold, device)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("softplus output"))?;
    for (index, &value) in input.iter().enumerate() {
        check_periodically(index, cancellation)?;
        output.push(softplus_scalar(value, beta, threshold));
    }
    cancellation.check()?;
    Ok(output)
}

fn softplus_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    beta: f32,
    threshold: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    cancellation.check()?;
    validate_softplus(beta, threshold, device)?;
    require_same_length(
        input,
        output_gradient,
        SOFTPLUS_OPERATION_ID,
        "output gradient",
    )?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("softplus gradient"))?;
    for (index, (&value, &gradient)) in input.iter().zip(output_gradient).enumerate() {
        check_periodically(index, cancellation)?;
        let scaled = beta * value;
        output.push(
            gradient
                * if scaled > threshold {
                    1.0
                } else {
                    sigmoid_scalar(scaled)
                },
        );
    }
    cancellation.check()?;
    Ok(output)
}

pub fn pixel_unshuffle_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    downscale_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    pixel_rearrange_with_context(backend, input, downscale_factor, false, context)
}

pub fn pixel_unshuffle_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    downscale_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    pixel_rearrange_with_context(backend, input, downscale_factor, false, context)
}

pub fn pixel_unshuffle_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    downscale_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    let expected = pixel_rearranged_shape(input_shape, downscale_factor, false)?;
    if output_gradient.descriptor().shape() != expected {
        return invalid(
            PIXEL_UNSHUFFLE_OPERATION_ID,
            "output gradient shape does not match pixel-unshuffle output",
        );
    }
    pixel_rearrange_with_context(backend, output_gradient, downscale_factor, true, context)
}

pub fn pixel_unshuffle_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    downscale_factor: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    pixel_unshuffle_with_context_exact_native(backend, input_tangent, downscale_factor, context)
}

#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_with_context_exact_native(
    backend: &CpuBackend,
    request: AttentionKernelRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_attention_with_context(
        backend, request, query, key, value, mask, context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_vjp_with_context_exact_native(
    backend: &CpuBackend,
    request: AttentionKernelRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    output_gradient: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<AttentionVjp, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_attention_vjp_with_context(
        backend,
        request,
        query,
        key,
        value,
        mask,
        output_gradient,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_jvp_with_context_exact_native(
    backend: &CpuBackend,
    request: AttentionKernelRequest,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<AttentionMask<'_>>,
    query_tangent: &[f32],
    key_tangent: &[f32],
    value_tangent: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_attention_jvp_with_context(
        backend,
        request,
        query,
        key,
        value,
        mask,
        query_tangent,
        key_tangent,
        value_tangent,
        context,
    )?)
}

pub fn sigmoid_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_sigmoid_with_context(backend, input, context)?)
}

pub fn sigmoid_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_sigmoid_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn sigmoid_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    Ok(canonical_sigmoid_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn softplus_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    beta: f32,
    threshold: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    softplus_exact_native(input, beta, threshold, device, context.cancellation)
}

pub fn softplus_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    beta: f32,
    threshold: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    softplus_vjp_exact_native(
        input,
        output_gradient,
        beta,
        threshold,
        device,
        context.cancellation,
    )
}

pub fn softplus_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    beta: f32,
    threshold: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    softplus_vjp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        beta,
        threshold,
        device,
        context,
    )
}

fn pixel_rearranged_shape(
    shape: &[u64],
    factor: u64,
    shuffle: bool,
) -> Result<Vec<u64>, NeuralNetworkFunctionalError> {
    if shape.len() < 3 || factor == 0 {
        return invalid(
            if shuffle {
                PIXEL_SHUFFLE_OPERATION_ID
            } else {
                PIXEL_UNSHUFFLE_OPERATION_ID
            },
            "input rank must be at least three and factor positive",
        );
    }
    let channel_axis = shape.len() - 3;
    let factor_squared =
        factor
            .checked_mul(factor)
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
                "pixel rearrangement factor",
            ))?;
    let mut output = shape.to_vec();
    if shuffle {
        if !shape[channel_axis].is_multiple_of(factor_squared) {
            return invalid(
                PIXEL_SHUFFLE_OPERATION_ID,
                "channel count must be divisible by factor squared",
            );
        }
        output[channel_axis] /= factor_squared;
        output[channel_axis + 1] = output[channel_axis + 1]
            .checked_mul(factor)
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
                "pixel-shuffle height",
            ))?;
        output[channel_axis + 2] = output[channel_axis + 2]
            .checked_mul(factor)
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
                "pixel-shuffle width",
            ))?;
    } else {
        if !shape[channel_axis + 1].is_multiple_of(factor)
            || !shape[channel_axis + 2].is_multiple_of(factor)
        {
            return invalid(
                PIXEL_UNSHUFFLE_OPERATION_ID,
                "spatial dimensions must be divisible by factor",
            );
        }
        output[channel_axis] = output[channel_axis]
            .checked_mul(factor_squared)
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
                "pixel-unshuffle channels",
            ))?;
        output[channel_axis + 1] /= factor;
        output[channel_axis + 2] /= factor;
    }
    Ok(output)
}

fn pixel_rearrange_with_context(
    backend: &dyn TensorBackend,
    input: &Tensor,
    factor: u64,
    shuffle: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    let operation = if shuffle {
        PIXEL_SHUFFLE_OPERATION_ID
    } else {
        PIXEL_UNSHUFFLE_OPERATION_ID
    };
    require_tensor_backend_input(backend, input, operation, context)?;
    let plan = pixel_rearrange_plan(input.descriptor().shape(), factor, shuffle, operation)?;
    Ok(canonical_rearrange_with_context(
        backend, input, &plan, operation, context,
    )?)
}

fn pixel_rearrange_plan(
    input_shape: &[u64],
    factor: u64,
    shuffle: bool,
    operation: &'static str,
) -> Result<NativeRearrangePlan, NeuralNetworkFunctionalError> {
    let output_shape = pixel_rearranged_shape(input_shape, factor, shuffle)?;
    let channel_axis = input_shape.len() - 3;
    let mut lengths = input_shape[..channel_axis].to_vec();
    let mut input_groups = (0..channel_axis)
        .map(|axis| vec![axis])
        .collect::<Vec<_>>();
    let mut output_groups = input_groups.clone();
    let base = lengths.len();
    if shuffle {
        lengths.extend_from_slice(&[
            output_shape[channel_axis],
            factor,
            factor,
            input_shape[channel_axis + 1],
            input_shape[channel_axis + 2],
        ]);
        input_groups.extend([
            vec![base, base + 1, base + 2],
            vec![base + 3],
            vec![base + 4],
        ]);
        output_groups.extend([
            vec![base],
            vec![base + 3, base + 1],
            vec![base + 4, base + 2],
        ]);
    } else {
        lengths.extend_from_slice(&[
            input_shape[channel_axis],
            output_shape[channel_axis + 1],
            factor,
            output_shape[channel_axis + 2],
            factor,
        ]);
        input_groups.extend([
            vec![base],
            vec![base + 1, base + 2],
            vec![base + 3, base + 4],
        ]);
        output_groups.extend([
            vec![base, base + 2, base + 4],
            vec![base + 1],
            vec![base + 3],
        ]);
    }
    Ok(NativeRearrangePlan::from_atomic_axes(
        operation,
        input_shape.to_vec(),
        lengths,
        input_groups,
        output_groups,
    )?)
}

fn require_tensor_backend_input(
    backend: &dyn TensorBackend,
    input: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), NeuralNetworkFunctionalError> {
    if input.descriptor().device() != backend.device() {
        return Err(crate::TensorError::DeviceMismatch {
            expected: backend.device(),
            actual: input.descriptor().device(),
        }
        .into());
    }
    if input.descriptor().stream() != context.stream {
        return Err(crate::TensorError::StreamMismatch {
            expected: context.stream,
            actual: input.descriptor().stream(),
        }
        .into());
    }
    if input.descriptor().shape().len() < 3 {
        return invalid(operation, "input rank must be at least three");
    }
    Ok(())
}

fn validate_cosine_inputs(
    input_one: &[f32],
    input_one_shape: &[usize],
    input_two: &[f32],
    input_two_shape: &[usize],
    epsilon: f32,
) -> Result<(), NeuralNetworkFunctionalError> {
    require_count(
        input_one.len(),
        input_one_shape,
        COSINE_SIMILARITY_OPERATION_ID,
        "first input",
    )?;
    require_count(
        input_two.len(),
        input_two_shape,
        COSINE_SIMILARITY_OPERATION_ID,
        "second input",
    )?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return invalid(
            COSINE_SIMILARITY_OPERATION_ID,
            "epsilon must be finite and positive",
        );
    }
    Ok(())
}

fn validate_embedding_options(
    options: EmbeddingOptions,
    rows: usize,
) -> Result<(), NeuralNetworkFunctionalError> {
    if options.sparse {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "sparse gradients are not a certified native backend",
        );
    }
    normalized_padding(options.padding_index, rows)?;
    if options
        .max_norm
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "max_norm must be finite and positive",
        );
    }
    if !options.norm_type.is_finite() || options.norm_type <= 0.0 {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "norm_type must be finite and positive",
        );
    }
    Ok(())
}

fn normalize_embedding_index(
    index: i64,
    rows: usize,
) -> Result<usize, NeuralNetworkFunctionalError> {
    let index = usize::try_from(index)
        .map_err(|_| invalid_error(EMBEDDING_OPERATION_ID, "indices must be non-negative"))?;
    if index >= rows {
        return invalid(EMBEDDING_OPERATION_ID, "index is outside the weight table");
    }
    Ok(index)
}

fn normalized_padding(
    padding: Option<i64>,
    rows: usize,
) -> Result<Option<usize>, NeuralNetworkFunctionalError> {
    padding
        .map(|index| {
            let rows = i64::try_from(rows)
                .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding rows"))?;
            let normalized = if index < 0 {
                index
                    .checked_add(rows)
                    .ok_or(NeuralNetworkFunctionalError::ShapeOverflow("padding index"))?
            } else {
                index
            };
            if normalized < 0 || normalized >= rows {
                return invalid(
                    EMBEDDING_OPERATION_ID,
                    "padding_index is outside the weight table",
                );
            }
            usize::try_from(normalized)
                .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("padding index"))
        })
        .transpose()
}

fn renormalize_embedding_tensor_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    weight: &mut Tensor,
    indices: &[usize],
    max_norm: f32,
    norm_type: f32,
) -> Result<(), NeuralNetworkFunctionalError> {
    let rows = usize::try_from(weight.descriptor().shape()[0])
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding rows"))?;
    let width = usize::try_from(weight.descriptor().shape()[1])
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding width"))?;
    let mut seen = workspace_filled(backend, context, rows, false)?;
    let mut changed = workspace_filled(backend, context, rows, false)?;
    let replacement_count = rows
        .checked_mul(width)
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
            "embedding replacements",
        ))?;
    let mut replacements = workspace_filled(backend, context, replacement_count, 0.0_f32)?;
    for (position, row) in indices.iter().copied().enumerate() {
        check_periodically(position, context.cancellation)?;
        if seen[row] {
            continue;
        }
        seen[row] = true;
        let row_u64 = u64::try_from(row)
            .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding row"))?;
        let mut sum = 0.0_f32;
        for column in 0..width {
            let column_u64 = u64::try_from(column)
                .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding column"))?;
            let DecodedScalar::Real(value) = DType::F32
                .decode_scalar(weight.element_bytes(&[row_u64, column_u64])?)?
            else {
                return invalid(
                    EMBEDDING_OPERATION_ID,
                    "F32 weight decoded as a non-real scalar",
                );
            };
            sum += (value as f32).abs().powf(norm_type);
        }
        let norm = sum.powf(1.0 / norm_type);
        if norm > max_norm {
            changed[row] = true;
            let scale = max_norm / (norm + 1.0e-7);
            for column in 0..width {
                let column_u64 = u64::try_from(column).map_err(|_| {
                    NeuralNetworkFunctionalError::ShapeOverflow("embedding column")
                })?;
                let DecodedScalar::Real(value) = DType::F32
                    .decode_scalar(weight.element_bytes(&[row_u64, column_u64])?)?
                else {
                    return invalid(
                        EMBEDDING_OPERATION_ID,
                        "F32 weight decoded as a non-real scalar",
                    );
                };
                replacements[row * width + column] = value as f32 * scale;
            }
        }
    }
    context.cancellation.check()?;
    let mut write = weight.write()?;
    for row in 0..rows {
        if !changed[row] {
            continue;
        }
        for column in 0..width {
            let bytes = DType::F32.encode_decoded_scalar(
                DecodedScalar::Real(f64::from(replacements[row * width + column])),
                EMBEDDING_OPERATION_ID,
                DeviceId::CPU,
            )?;
            write
                .element_bytes_mut(&[
                    u64::try_from(row).map_err(|_| {
                        NeuralNetworkFunctionalError::ShapeOverflow("embedding row")
                    })?,
                    u64::try_from(column).map_err(|_| {
                        NeuralNetworkFunctionalError::ShapeOverflow("embedding column")
                    })?,
                ])?
                .copy_from_slice(&bytes);
        }
    }
    Ok(())
}

fn require_tensor_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), NeuralNetworkFunctionalError> {
    require_cpu(operation, tensor.descriptor().device())
}

fn fresh_contiguous_copy_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    Ok(backend.copy(input, descriptor, context)?.0)
}

fn flatten_tensor_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    let count = input.descriptor().element_count()?;
    if input.descriptor().is_contiguous()? {
        let descriptor = TensorDescriptor::new_strided(
            vec![count],
            vec![1],
            input.descriptor().offset_elements(),
            input.descriptor().dtype(),
            Layout::Strided,
            input.descriptor().device(),
            input.descriptor().stream(),
        )?;
        return Ok(input.view(descriptor, ViewAccess::ReadOnly)?);
    }
    let count_usize = usize::try_from(count)
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("flattened element count"))?;
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("flattened element width"))?;
    let capacity = count_usize
        .checked_mul(width)
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
            "flattened bytes",
        ))?;
    let mut bytes = backend.workspace_vec(context, capacity)?;
    for linear in 0..count_usize {
        check_periodically(linear, context.cancellation)?;
        let linear = u64::try_from(linear)
            .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("flattened index"))?;
        for value in input.linear_element_bytes(linear)?.iter().copied() {
            bytes.try_push(value)?;
        }
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![count],
        input.descriptor().dtype(),
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn tensor_i64_values_with_context(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<i64>, NeuralNetworkFunctionalError> {
    let count = usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("I64 value count"))?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let linear = u64::try_from(linear)
            .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("I64 index"))?;
        let DecodedScalar::Signed(value) =
            DType::I64.decode_scalar(tensor.linear_element_bytes(linear)?)?
        else {
            return invalid(
                EMBEDDING_OPERATION_ID,
                "I64 tensor decoded as a non-integer scalar",
            );
        };
        values.try_push(value)?;
    }
    Ok(values)
}

fn tensor_f32_values_with_context(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, NeuralNetworkFunctionalError> {
    let count = usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("F32 value count"))?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let linear = u64::try_from(linear)
            .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("F32 index"))?;
        let DecodedScalar::Real(value) =
            DType::F32.decode_scalar(tensor.linear_element_bytes(linear)?)?
        else {
            return invalid(
                EMBEDDING_OPERATION_ID,
                "F32 tensor decoded as a non-real scalar",
            );
        };
        values.try_push(value as f32)?;
    }
    Ok(values)
}

fn reshape_embedding_output(
    selected: &Tensor,
    index_shape: &[u64],
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    let mut shape = index_shape.to_vec();
    shape.push(selected.descriptor().shape()[1]);
    let contiguous = TensorDescriptor::contiguous(
        shape.clone(),
        selected.descriptor().dtype(),
        selected.descriptor().device(),
        selected.descriptor().stream(),
    )?;
    let descriptor = TensorDescriptor::new_strided(
        shape,
        contiguous.strides().to_vec(),
        selected.descriptor().offset_elements(),
        selected.descriptor().dtype(),
        Layout::Strided,
        selected.descriptor().device(),
        selected.descriptor().stream(),
    )?;
    Ok(selected.view(descriptor, ViewAccess::ReadOnly)?)
}

fn flatten_embedding_gradient_with_context(
    backend: &CpuBackend,
    gradient: &Tensor,
    positions: usize,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    if gradient.descriptor().dtype() != DType::F32 {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "embedding gradient must use F32 dtype",
        );
    }
    let expected = u64::try_from(positions)
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding positions"))?
        .checked_mul(width)
        .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
            "embedding gradient count",
        ))?;
    if gradient.descriptor().element_count()? != expected {
        return invalid(
            EMBEDDING_OPERATION_ID,
            "embedding gradient shape does not match output",
        );
    }
    let flattened = flatten_tensor_with_context(backend, gradient, context)?;
    let shape = vec![
        u64::try_from(positions)
            .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("embedding positions"))?,
        width,
    ];
    let contiguous = TensorDescriptor::contiguous(
        shape.clone(),
        DType::F32,
        gradient.descriptor().device(),
        gradient.descriptor().stream(),
    )?;
    let descriptor = TensorDescriptor::new_strided(
        shape,
        contiguous.strides().to_vec(),
        flattened.descriptor().offset_elements(),
        DType::F32,
        Layout::Strided,
        gradient.descriptor().device(),
        gradient.descriptor().stream(),
    )?;
    Ok(flattened.view(descriptor, ViewAccess::ReadOnly)?)
}

fn upload_f32_tensor_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: crate::StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkFunctionalError> {
    context.cancellation.check()?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, NeuralNetworkFunctionalError> {
    let mut output = backend.workspace_vec(context, count)?;
    for _ in 0..count {
        output.try_push(value)?;
    }
    Ok(output)
}

fn checked_spatial_output(
    input: [usize; 2],
    parameters: SpatialParameters2d,
    operation: &'static str,
) -> Result<[usize; 2], NeuralNetworkFunctionalError> {
    if parameters.kernel_size.contains(&0)
        || parameters.dilation.contains(&0)
        || parameters.stride.contains(&0)
    {
        return invalid(
            operation,
            "kernel size, dilation, and stride must be positive",
        );
    }
    let mut output = [0; 2];
    for axis in 0..2 {
        let receptive = parameters.dilation[axis]
            .checked_mul(parameters.kernel_size[axis] - 1)
            .and_then(|value| value.checked_add(1))
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
                "spatial receptive field",
            ))?;
        let padded = input[axis]
            .checked_add(parameters.padding[axis].checked_mul(2).ok_or(
                NeuralNetworkFunctionalError::ShapeOverflow("spatial padding"),
            )?)
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow(
                "padded spatial size",
            ))?;
        if padded < receptive {
            return invalid(operation, "kernel is larger than the padded input");
        }
        output[axis] = (padded - receptive) / parameters.stride[axis] + 1;
    }
    Ok(output)
}

fn spatial_input_position(
    output_row: usize,
    output_column: usize,
    kernel_row: usize,
    kernel_column: usize,
    parameters: SpatialParameters2d,
) -> Option<[usize; 2]> {
    let row = output_row
        .checked_mul(parameters.stride[0])?
        .checked_add(kernel_row.checked_mul(parameters.dilation[0])?)?;
    let column = output_column
        .checked_mul(parameters.stride[1])?
        .checked_add(kernel_column.checked_mul(parameters.dilation[1])?)?;
    Some([
        row.checked_sub(parameters.padding[0])?,
        column.checked_sub(parameters.padding[1])?,
    ])
}

fn validate_softplus(
    beta: f32,
    threshold: f32,
    device: DeviceId,
) -> Result<(), NeuralNetworkFunctionalError> {
    require_cpu(SOFTPLUS_OPERATION_ID, device)?;
    if !beta.is_finite() || beta <= 0.0 || !threshold.is_finite() {
        return invalid(
            SOFTPLUS_OPERATION_ID,
            "beta must be finite and positive and threshold finite",
        );
    }
    Ok(())
}

fn softplus_scalar(value: f32, beta: f32, threshold: f32) -> f32 {
    let scaled = beta * value;
    if scaled > threshold {
        value
    } else {
        scaled.exp().ln_1p() / beta
    }
}

fn sigmoid_scalar(value: f32) -> f32 {
    apply_unary_scalar(UnaryOperation::Sigmoid, value)
}

fn broadcast_shape(
    left: &[usize],
    right: &[usize],
) -> Result<Vec<usize>, NeuralNetworkFunctionalError> {
    let left = left
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("broadcast shape"))?;
    let right = right
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("broadcast shape"))?;
    binary_broadcast_shape(&left, &right)?
        .into_iter()
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("broadcast shape"))
        })
        .collect()
}

fn broadcast_linear_index(
    indices: &[usize],
    broadcast_shape: &[usize],
    source_shape: &[usize],
) -> Result<usize, NeuralNetworkFunctionalError> {
    let offset = broadcast_shape
        .len()
        .checked_sub(source_shape.len())
        .ok_or_else(|| {
            invalid_error(
                COSINE_SIMILARITY_OPERATION_ID,
                "source rank exceeds broadcast rank",
            )
        })?;
    let source_indices = source_shape
        .iter()
        .enumerate()
        .map(|(axis, &length)| {
            if length == 1 {
                0
            } else {
                indices[offset + axis]
            }
        })
        .collect::<Vec<_>>();
    ravel_index(&source_indices, source_shape)
}

#[allow(clippy::too_many_arguments)]
fn cosine_group_statistics(
    input_one: &[f32],
    input_one_shape: &[usize],
    input_two: &[f32],
    input_two_shape: &[usize],
    broadcast_shape: &[usize],
    axis: usize,
    output_indices: &[usize],
) -> Result<(f32, f32, f32), NeuralNetworkFunctionalError> {
    let mut dot = 0.0_f32;
    let mut one_squared = 0.0_f32;
    let mut two_squared = 0.0_f32;
    for reduced in 0..broadcast_shape[axis] {
        let indices = insert_axis(output_indices, axis, reduced);
        let one = input_one[broadcast_linear_index(
            &indices,
            broadcast_shape,
            input_one_shape,
        )?];
        let two = input_two[broadcast_linear_index(
            &indices,
            broadcast_shape,
            input_two_shape,
        )?];
        dot = one.mul_add(two, dot);
        one_squared = one.mul_add(one, one_squared);
        two_squared = two.mul_add(two, two_squared);
    }
    Ok((dot, one_squared.sqrt(), two_squared.sqrt()))
}

fn reduced_shape(shape: &[usize], axis: usize) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (index != axis).then_some(value))
        .collect()
}

fn insert_axis(indices: &[usize], axis: usize, value: usize) -> Vec<usize> {
    let mut output = Vec::with_capacity(indices.len() + 1);
    output.extend_from_slice(&indices[..axis]);
    output.push(value);
    output.extend_from_slice(&indices[axis..]);
    output
}

fn normalize_axis(
    dimension: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, NeuralNetworkFunctionalError> {
    if rank == 0 {
        return invalid(operation, "operation requires a non-scalar input");
    }
    let rank =
        i64::try_from(rank).map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("rank"))?;
    let normalized = if dimension < 0 {
        dimension
            .checked_add(rank)
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow("axis"))?
    } else {
        dimension
    };
    if normalized < 0 || normalized >= rank {
        return invalid(operation, "dimension is outside the input rank");
    }
    usize::try_from(normalized).map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow("axis"))
}

fn require_cpu(
    operation: &'static str,
    device: DeviceId,
) -> Result<(), NeuralNetworkFunctionalError> {
    if device != DeviceId::CPU {
        return Err(NeuralNetworkFunctionalError::UnsupportedDevice { operation, device });
    }
    Ok(())
}

fn require_count(
    actual: usize,
    shape: &[usize],
    operation: &'static str,
    name: &'static str,
) -> Result<(), NeuralNetworkFunctionalError> {
    let expected = element_count(shape)?;
    if actual != expected {
        return invalid(
            operation,
            format!("{name} expected {expected} values, got {actual}"),
        );
    }
    Ok(())
}

fn require_same_length(
    left: &[f32],
    right: &[f32],
    operation: &'static str,
    name: &'static str,
) -> Result<(), NeuralNetworkFunctionalError> {
    if left.len() != right.len() {
        return invalid(
            operation,
            format!("{name} expected {} values, got {}", left.len(), right.len()),
        );
    }
    Ok(())
}

fn element_count(shape: &[usize]) -> Result<usize, NeuralNetworkFunctionalError> {
    shape.iter().try_fold(1_usize, |product, &length| {
        product
            .checked_mul(length)
            .ok_or(NeuralNetworkFunctionalError::ShapeOverflow("element count"))
    })
}

fn ravel_index(indices: &[usize], shape: &[usize]) -> Result<usize, NeuralNetworkFunctionalError> {
    if indices.len() != shape.len() {
        return invalid(COSINE_SIMILARITY_OPERATION_ID, "index rank mismatch");
    }
    indices
        .iter()
        .zip(shape)
        .try_fold(0_usize, |linear, (&index, &length)| {
            if index >= length {
                return invalid(COSINE_SIMILARITY_OPERATION_ID, "index is outside shape");
            }
            linear
                .checked_mul(length)
                .and_then(|value| value.checked_add(index))
                .ok_or(NeuralNetworkFunctionalError::ShapeOverflow("linear index"))
        })
}

fn unravel_index(
    mut linear: usize,
    shape: &[usize],
) -> Result<Vec<usize>, NeuralNetworkFunctionalError> {
    let mut indices = vec![0; shape.len()];
    for (index, &length) in indices.iter_mut().zip(shape).rev() {
        if length == 0 {
            return invalid(
                COSINE_SIMILARITY_OPERATION_ID,
                "cannot index an empty dimension",
            );
        }
        *index = linear % length;
        linear /= length;
    }
    Ok(indices)
}

fn reserved_zeros(
    length: usize,
    subject: &'static str,
) -> Result<Vec<f32>, NeuralNetworkFunctionalError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| NeuralNetworkFunctionalError::ShapeOverflow(subject))?;
    values.resize(length, 0.0);
    Ok(values)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), NeuralNetworkFunctionalError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, NeuralNetworkFunctionalError> {
    Err(invalid_error(operation, reason))
}

fn invalid_error(
    operation: &'static str,
    reason: impl Into<String>,
) -> NeuralNetworkFunctionalError {
    NeuralNetworkFunctionalError::Invalid {
        operation,
        reason: reason.into(),
    }
}
