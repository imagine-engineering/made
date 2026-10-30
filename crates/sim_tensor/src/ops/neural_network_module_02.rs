use crate::{
    CpuBackend, DeviceId, ExecutionContext, Tensor, TensorBackend, TensorDescriptor, TensorError,
    generated_accelerated_attention_kernel_01::{
        AttentionKernelKind, AttentionKernelRequest, AttentionLayout, AttentionShape, AttentionVjp,
    },
    generated_activation_normalization_functional_01::{
        AffineVjp, FunctionalError,
        batch_norm_jvp_with_context_exact_native as canonical_batch_norm_jvp,
        batch_norm_vjp_with_context_exact_native as canonical_batch_norm_vjp,
        batch_norm_with_context_exact_native as canonical_batch_norm,
        group_norm_jvp_with_context_exact_native as canonical_group_norm_jvp,
        group_norm_tensor_with_context_exact_native as canonical_group_norm_tensor,
        group_norm_vjp_with_context_exact_native as canonical_group_norm_vjp,
        group_norm_with_context_exact_native as canonical_group_norm,
        leaky_relu_jvp_with_context_exact_native as canonical_leaky_relu_jvp,
        leaky_relu_vjp_with_context_exact_native as canonical_leaky_relu_vjp,
        leaky_relu_with_context_exact_native as canonical_leaky_relu,
    },
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionPaddingMode, ConvolutionVjp, LinearVjp,
        OperatorIndirectionError, TensorValues,
        convolution_jvp_with_context_exact_native as canonical_convolution_jvp,
        convolution_vjp_with_context_exact_native as canonical_convolution_vjp,
        convolution_with_context_exact_native as canonical_convolution,
        linear_jvp_with_context_exact_native as canonical_linear_jvp,
        linear_vjp_with_context_exact_native as canonical_linear_vjp,
        linear_with_context_exact_native as canonical_linear, map_padded_coordinate,
    },
    generated_elementwise_or_runtime_operation_08::ElementwiseRuntimePartEightError,
    generated_neural_network_functional_01::{
        EmbeddingOptions, NeuralNetworkFunctionalError,
        embedding_jvp_with_context_exact_native as canonical_embedding_jvp,
        embedding_vjp_with_context_exact_native as canonical_embedding_vjp,
        embedding_with_context_exact_native as canonical_embedding,
        scaled_dot_product_attention_jvp_with_context_exact_native as canonical_attention_jvp,
        scaled_dot_product_attention_vjp_with_context_exact_native as canonical_attention_vjp,
        scaled_dot_product_attention_with_context_exact_native as canonical_attention,
    },
    generated_neural_network_module_01::{
        AveragePool2dVjp, LossReduction, NeuralNetworkModuleError,
        adaptive_average_pool_2d_jvp_with_context_exact_native as canonical_adaptive_pool_jvp,
        adaptive_average_pool_2d_vjp_with_context_exact_native as canonical_adaptive_pool_vjp,
        adaptive_average_pool_2d_with_context_exact_native as canonical_adaptive_pool,
        average_pool_3d_jvp_with_context_exact_native as canonical_average_pool_3d_jvp,
        average_pool_3d_vjp_with_context_exact_native as canonical_average_pool_3d_vjp,
        average_pool_3d_with_context_exact_native as canonical_average_pool_3d,
        smooth_l1_loss_jvp_with_context_exact_native as canonical_smooth_l1_jvp,
        smooth_l1_loss_vjp_with_context_exact_native as canonical_smooth_l1_vjp,
        smooth_l1_loss_with_context_exact_native as canonical_smooth_l1,
    },
};
use thiserror::Error;

pub const ADAPTIVE_AVG_POOL_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-6A8710D48911";
pub const AVG_POOL_3D_OPERATION_ID: &str = "COMFY-TENSOR-OP-835CE6AE654F";
pub const BATCH_NORM_1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-7346A72D7D61";
pub const BATCH_NORM_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-72C64A93D406";
pub const CONV_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-700E959DB8E4";
pub const EMBEDDING_OPERATION_ID: &str = "COMFY-TENSOR-OP-690D2FADC241";
pub const HUBER_LOSS_OPERATION_ID: &str = "COMFY-TENSOR-OP-5B8CE1451811";
pub const INSTANCE_NORM_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-69EEDCD403B3";
pub const LEAKY_RELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-853B6CAD39A0";
pub const LINEAR_OPERATION_ID: &str = "COMFY-TENSOR-OP-707C3B6AD0B4";
pub const MULTIHEAD_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-679C11D943DB";
pub const REPLICATION_PAD_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-7BEE9C744BDC";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NeuralNetworkModulePartTwoError {
    #[error(transparent)]
    Tensor(TensorError),
    #[error(transparent)]
    Module(NeuralNetworkModuleError),
    #[error(transparent)]
    Functional(FunctionalError),
    #[error(transparent)]
    Operator(OperatorIndirectionError),
    #[error("canonical neural-network functional operation failed: {0}")]
    NeuralFunctional(String),
    #[error("neural-network module part-two operation was cancelled")]
    Cancelled,
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<TensorError> for NeuralNetworkModulePartTwoError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled,
            error => Self::Tensor(error),
        }
    }
}

impl From<sim_types::CancellationError> for NeuralNetworkModulePartTwoError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<NeuralNetworkModuleError> for NeuralNetworkModulePartTwoError {
    fn from(error: NeuralNetworkModuleError) -> Self {
        match error {
            NeuralNetworkModuleError::Cancelled => Self::Cancelled,
            error => Self::Module(error),
        }
    }
}

impl From<FunctionalError> for NeuralNetworkModulePartTwoError {
    fn from(error: FunctionalError) -> Self {
        match error {
            FunctionalError::Cancelled => Self::Cancelled,
            error => Self::Functional(error),
        }
    }
}

impl From<OperatorIndirectionError> for NeuralNetworkModulePartTwoError {
    fn from(error: OperatorIndirectionError) -> Self {
        match error {
            OperatorIndirectionError::Cancelled => Self::Cancelled,
            error => Self::Operator(error),
        }
    }
}

impl From<NeuralNetworkFunctionalError> for NeuralNetworkModulePartTwoError {
    fn from(error: NeuralNetworkFunctionalError) -> Self {
        match error {
            NeuralNetworkFunctionalError::Cancelled => Self::Cancelled,
            NeuralNetworkFunctionalError::Tensor(error) => error.into(),
            NeuralNetworkFunctionalError::Operator(error) => error.into(),
            NeuralNetworkFunctionalError::Normalization(error) => error.into(),
            NeuralNetworkFunctionalError::IndexSelect(
                ElementwiseRuntimePartEightError::Cancelled,
            ) => Self::Cancelled,
            NeuralNetworkFunctionalError::IndexSelect(
                ElementwiseRuntimePartEightError::Tensor(error),
            ) => error.into(),
            error => Self::NeuralFunctional(error.to_string()),
        }
    }
}

pub fn adaptive_average_pool_2d_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    output_size: [usize; 2],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_adaptive_pool(
        backend, input, input_shape, output_size, device, context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn adaptive_average_pool_2d_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    output_size: [usize; 2],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AveragePool2dVjp, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_adaptive_pool_vjp(
        backend,
        input,
        input_shape,
        output_size,
        output_gradient,
        device,
        context,
    )?)
}

pub fn adaptive_average_pool_2d_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &[f32],
    input_shape: &[usize],
    output_size: [usize; 2],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_adaptive_pool_jvp(
        backend,
        input_tangent,
        input_shape,
        output_size,
        device,
        context,
    )?)
}

pub fn average_pool_3d_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 3],
    stride: [usize; 3],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_average_pool_3d(
        backend,
        input,
        input_shape,
        kernel_size,
        stride,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_3d_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 3],
    stride: [usize; 3],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AveragePool2dVjp, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_average_pool_3d_vjp(
        backend,
        input,
        input_shape,
        kernel_size,
        stride,
        output_gradient,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_3d_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 3],
    stride: [usize; 3],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_average_pool_3d_jvp(
        backend,
        input_tangent,
        input_shape,
        kernel_size,
        stride,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    expected_rank: usize,
    running_mean: Option<&mut [f32]>,
    running_variance: Option<&mut [f32]>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    training: bool,
    momentum: f32,
    epsilon: f32,
    operation: &'static str,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    require_rank(shape, expected_rank, operation)?;
    Ok(canonical_batch_norm(
        backend,
        input,
        shape,
        running_mean,
        running_variance,
        weight,
        bias,
        training,
        momentum,
        epsilon,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    expected_rank: usize,
    running_mean: Option<&[f32]>,
    running_variance: Option<&[f32]>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    training: bool,
    epsilon: f32,
    operation: &'static str,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AffineVjp, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    require_rank(shape, expected_rank, operation)?;
    Ok(canonical_batch_norm_vjp(
        backend,
        input,
        output_gradient,
        shape,
        running_mean,
        running_variance,
        weight,
        bias,
        training,
        epsilon,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    expected_rank: usize,
    running_mean: Option<&[f32]>,
    running_variance: Option<&[f32]>,
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    training: bool,
    epsilon: f32,
    operation: &'static str,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    require_rank(shape, expected_rank, operation)?;
    Ok(canonical_batch_norm_jvp(
        backend,
        input,
        input_tangent,
        shape,
        running_mean,
        running_variance,
        weight,
        weight_tangent,
        bias_tangent,
        training,
        epsilon,
        device,
        context,
    )?)
}

pub fn conv_2d_geometry(
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<ConvolutionGeometry, NeuralNetworkModulePartTwoError> {
    Ok(ConvolutionGeometry::new(
        2,
        stride.to_vec(),
        padding.to_vec(),
        dilation.to_vec(),
        groups,
        false,
        vec![0, 0],
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_2d_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_convolution(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        &conv_2d_geometry(stride, padding, dilation, groups)?,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_2d_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<ConvolutionVjp, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_convolution_vjp(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        output_gradient,
        &conv_2d_geometry(stride, padding, dilation, groups)?,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_2d_jvp_with_context_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_convolution_jvp(
        input,
        input_tangent,
        input_shape,
        weight,
        weight_tangent,
        weight_shape,
        bias,
        bias_tangent,
        &conv_2d_geometry(stride, padding, dilation, groups)?,
        device,
        context,
    )?)
}

pub fn embedding_module_with_context_exact_native(
    backend: &CpuBackend,
    indices: &Tensor,
    weight: &mut Tensor,
    options: EmbeddingOptions,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_embedding(
        backend, indices, weight, options, context,
    )?)
}

pub fn embedding_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    indices: &Tensor,
    weight: &Tensor,
    options: EmbeddingOptions,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_embedding_vjp(
        backend,
        indices,
        weight,
        options,
        output_gradient,
        context,
    )?)
}

pub fn embedding_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    indices: &Tensor,
    weight: &Tensor,
    weight_tangent: &Tensor,
    options: EmbeddingOptions,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_embedding_jvp(
        backend,
        indices,
        weight,
        weight_tangent,
        options,
        context,
    )?)
}

pub fn huber_loss_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    target: &[f32],
    delta: f32,
    reduction: LossReduction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    validate_delta(delta)?;
    let mut output = canonical_smooth_l1(
        backend, input, target, delta, reduction, device, context,
    )?;
    scale_in_place(&mut output, delta, context)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn huber_loss_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    target: &[f32],
    delta: f32,
    reduction: LossReduction,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    validate_delta(delta)?;
    let mut gradient = canonical_smooth_l1_vjp(
        backend,
        input,
        target,
        delta,
        reduction,
        output_gradient,
        device,
        context,
    )?;
    scale_in_place(&mut gradient, delta, context)?;
    Ok(gradient)
}

#[allow(clippy::too_many_arguments)]
pub fn huber_loss_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    target: &[f32],
    target_tangent: &[f32],
    delta: f32,
    reduction: LossReduction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    validate_delta(delta)?;
    let mut tangent = canonical_smooth_l1_jvp(
        backend,
        input,
        input_tangent,
        target,
        target_tangent,
        delta,
        reduction,
        device,
        context,
    )?;
    scale_in_place(&mut tangent, delta, context)?;
    Ok(tangent)
}

#[allow(clippy::too_many_arguments)]
pub fn instance_norm_2d_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    require_rank(shape, 4, INSTANCE_NORM_2D_OPERATION_ID)?;
    let channels = channel_count(shape, INSTANCE_NORM_2D_OPERATION_ID)?;
    Ok(canonical_group_norm(
        backend, input, shape, channels, weight, bias, epsilon, device, context,
    )?)
}

pub fn instance_norm_2d_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    epsilon: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartTwoError> {
    context.check()?;
    let shape = input.descriptor().shape();
    if shape.len() != 4 {
        return Err(NeuralNetworkModulePartTwoError::Invalid {
            operation: INSTANCE_NORM_2D_OPERATION_ID,
            reason: format!("expected rank 4, got {}", shape.len()),
        });
    }
    let channels = shape.get(1).copied().ok_or(
        NeuralNetworkModulePartTwoError::Invalid {
            operation: INSTANCE_NORM_2D_OPERATION_ID,
            reason: "missing channel dimension".to_owned(),
        },
    )?;
    Ok(canonical_group_norm_tensor(
        backend,
        input,
        channels,
        None,
        None,
        epsilon,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn instance_norm_2d_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AffineVjp, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    require_rank(shape, 4, INSTANCE_NORM_2D_OPERATION_ID)?;
    let channels = channel_count(shape, INSTANCE_NORM_2D_OPERATION_ID)?;
    Ok(canonical_group_norm_vjp(
        backend,
        input,
        output_gradient,
        shape,
        channels,
        weight,
        bias,
        epsilon,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn instance_norm_2d_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    require_rank(shape, 4, INSTANCE_NORM_2D_OPERATION_ID)?;
    let channels = channel_count(shape, INSTANCE_NORM_2D_OPERATION_ID)?;
    Ok(canonical_group_norm_jvp(
        backend,
        input,
        input_tangent,
        shape,
        channels,
        weight,
        weight_tangent,
        bias_tangent,
        epsilon,
        device,
        context,
    )?)
}

pub fn leaky_relu_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    negative_slope: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_leaky_relu(
        backend,
        input,
        negative_slope,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn leaky_relu_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    negative_slope: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_leaky_relu_vjp(
        backend,
        input,
        output_gradient,
        negative_slope,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn leaky_relu_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    negative_slope: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    Ok(canonical_leaky_relu_jvp(
        backend,
        input,
        input_tangent,
        negative_slope,
        device,
        context,
    )?)
}

pub fn linear_module_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
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

#[allow(clippy::too_many_arguments)]
pub fn linear_module_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<LinearVjp, NeuralNetworkModulePartTwoError> {
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
pub fn linear_module_jvp_with_context_exact_native(
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
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
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

#[derive(Clone, Debug, PartialEq)]
pub struct MultiheadAttentionVjp {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn multihead_attention_projected_with_context_exact_native(
    backend: &CpuBackend,
    query: &[f32],
    query_shape: &[usize],
    key: &[f32],
    key_shape: &[usize],
    value: &[f32],
    value_shape: &[usize],
    heads: usize,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    let geometry = MultiheadGeometry::new(
        query,
        query_shape,
        key,
        key_shape,
        value,
        value_shape,
        heads,
    )?;
    let query = geometry.to_nhd(query, geometry.query_tokens)?;
    let key = geometry.to_nhd(key, geometry.key_tokens)?;
    let value = geometry.to_nhd(value, geometry.key_tokens)?;
    let output = canonical_attention(
        backend,
        geometry.request(),
        &query,
        &key,
        &value,
        None,
        context,
    )?;
    Ok(TensorValues {
        values: geometry.from_nhd(&output, geometry.query_tokens)?,
        shape: query_shape.to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn multihead_attention_projected_vjp_with_context_exact_native(
    backend: &CpuBackend,
    query: &[f32],
    query_shape: &[usize],
    key: &[f32],
    key_shape: &[usize],
    value: &[f32],
    value_shape: &[usize],
    heads: usize,
    output_gradient: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<MultiheadAttentionVjp, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    let geometry = MultiheadGeometry::new(
        query,
        query_shape,
        key,
        key_shape,
        value,
        value_shape,
        heads,
    )?;
    require_length(
        output_gradient.len(),
        query.len(),
        MULTIHEAD_ATTENTION_OPERATION_ID,
        "output gradient",
    )?;
    let query_nhd = geometry.to_nhd(query, geometry.query_tokens)?;
    let key_nhd = geometry.to_nhd(key, geometry.key_tokens)?;
    let value_nhd = geometry.to_nhd(value, geometry.key_tokens)?;
    let output_gradient = geometry.to_nhd(output_gradient, geometry.query_tokens)?;
    let AttentionVjp { query, key, value } = canonical_attention_vjp(
        backend,
        geometry.request(),
        &query_nhd,
        &key_nhd,
        &value_nhd,
        None,
        &output_gradient,
        context,
    )?;
    Ok(MultiheadAttentionVjp {
        query: geometry.from_nhd(&query, geometry.query_tokens)?,
        key: geometry.from_nhd(&key, geometry.key_tokens)?,
        value: geometry.from_nhd(&value, geometry.key_tokens)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn multihead_attention_projected_jvp_with_context_exact_native(
    backend: &CpuBackend,
    query: &[f32],
    query_tangent: &[f32],
    query_shape: &[usize],
    key: &[f32],
    key_tangent: &[f32],
    key_shape: &[usize],
    value: &[f32],
    value_tangent: &[f32],
    value_shape: &[usize],
    heads: usize,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    let geometry = MultiheadGeometry::new(
        query,
        query_shape,
        key,
        key_shape,
        value,
        value_shape,
        heads,
    )?;
    for (values, tangent, name) in [
        (query, query_tangent, "query tangent"),
        (key, key_tangent, "key tangent"),
        (value, value_tangent, "value tangent"),
    ] {
        require_length(
            tangent.len(),
            values.len(),
            MULTIHEAD_ATTENTION_OPERATION_ID,
            name,
        )?;
    }
    let output = canonical_attention_jvp(
        backend,
        geometry.request(),
        &geometry.to_nhd(query, geometry.query_tokens)?,
        &geometry.to_nhd(key, geometry.key_tokens)?,
        &geometry.to_nhd(value, geometry.key_tokens)?,
        None,
        &geometry.to_nhd(query_tangent, geometry.query_tokens)?,
        &geometry.to_nhd(key_tangent, geometry.key_tokens)?,
        &geometry.to_nhd(value_tangent, geometry.key_tokens)?,
        context,
    )?;
    Ok(TensorValues {
        values: geometry.from_nhd(&output, geometry.query_tokens)?,
        shape: query_shape.to_vec(),
    })
}

pub fn replication_pad_2d_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    padding: [usize; 4],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    require_cpu(device, REPLICATION_PAD_2D_OPERATION_ID)?;
    let geometry = Pad2dGeometry::new(
        input, input_shape, padding, REPLICATION_PAD_2D_OPERATION_ID,
        ConvolutionPaddingMode::Replicate,
    )?;
    let mut output = vec![0.0; geometry.output_count()?];
    geometry.for_each_mapping(context, |input_index, output_index| {
        let input_index = input_index.ok_or_else(|| {
            invalid_error(REPLICATION_PAD_2D_OPERATION_ID, "replicate mapping")
        })?;
        output[output_index] = input[input_index];
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: geometry.output_shape,
    })
}

pub fn replication_pad_2d_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    padding: [u64; 4],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModulePartTwoError> {
    context.check()?;
    if input.descriptor().device() != backend.device() {
        return Err(TensorError::DeviceMismatch {
            expected: backend.device(),
            actual: input.descriptor().device(),
        }
        .into());
    }
    if input.descriptor().stream() != context.stream {
        return Err(TensorError::StreamMismatch {
            expected: context.stream,
            actual: input.descriptor().stream(),
        }
        .into());
    }
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[2] == 0 || shape[3] == 0 {
        return Err(NeuralNetworkModulePartTwoError::Invalid {
            operation: REPLICATION_PAD_2D_OPERATION_ID,
            reason: "input must be non-empty NCHW".to_owned(),
        });
    }
    let output_shape = vec![
        shape[0],
        shape[1],
        shape[2]
            .checked_add(padding[2])
            .and_then(|extent| extent.checked_add(padding[3]))
            .ok_or(NeuralNetworkModulePartTwoError::ShapeOverflow(
                "replication padding height",
            ))?,
        shape[3]
            .checked_add(padding[0])
            .and_then(|extent| extent.checked_add(padding[1]))
            .ok_or(NeuralNetworkModulePartTwoError::ShapeOverflow(
                "replication padding width",
            ))?,
    ];
    let descriptor = TensorDescriptor::contiguous(
        output_shape.clone(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mut output, event) = backend.allocate(descriptor, context)?;
    backend.wait_event(event, context)?;
    {
        let mut write = output.write()?;
        for batch in 0..output_shape[0] {
            for channel in 0..output_shape[1] {
                for output_y in 0..output_shape[2] {
                    context.check()?;
                    let input_y = output_y.saturating_sub(padding[2]).min(shape[2] - 1);
                    for output_x in 0..output_shape[3] {
                        let input_x = output_x.saturating_sub(padding[0]).min(shape[3] - 1);
                        write
                            .element_bytes_mut(&[batch, channel, output_y, output_x])?
                            .copy_from_slice(input.element_bytes(&[
                                batch, channel, input_y, input_x,
                            ])?);
                    }
                }
            }
        }
    }
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    context.check()?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn replication_pad_2d_vjp_with_context_exact_native(
    input: &[f32],
    input_shape: &[usize],
    padding: [usize; 4],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
    context.cancellation.check()?;
    require_cpu(device, REPLICATION_PAD_2D_OPERATION_ID)?;
    let geometry = Pad2dGeometry::new(
        input, input_shape, padding, REPLICATION_PAD_2D_OPERATION_ID,
        ConvolutionPaddingMode::Replicate,
    )?;
    require_length(
        output_gradient.len(),
        geometry.output_count()?,
        REPLICATION_PAD_2D_OPERATION_ID,
        "output gradient",
    )?;
    let mut input_gradient = vec![0.0; input.len()];
    geometry.for_each_mapping(context, |input_index, output_index| {
        let input_index = input_index.ok_or_else(|| {
            invalid_error(REPLICATION_PAD_2D_OPERATION_ID, "replicate mapping")
        })?;
        input_gradient[input_index] += output_gradient[output_index];
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(input_gradient)
}

pub fn replication_pad_2d_jvp_with_context_exact_native(
    input_tangent: &[f32],
    input_shape: &[usize],
    padding: [usize; 4],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModulePartTwoError> {
    replication_pad_2d_with_context_exact_native(
        input_tangent,
        input_shape,
        padding,
        device,
        context,
    )
}

struct MultiheadGeometry {
    query_tokens: usize,
    key_tokens: usize,
    batch: usize,
    embed: usize,
    heads: usize,
    head_dimension: usize,
    unbatched: bool,
}

impl MultiheadGeometry {
    #[allow(clippy::too_many_arguments)]
    fn new(
        query: &[f32],
        query_shape: &[usize],
        key: &[f32],
        key_shape: &[usize],
        value: &[f32],
        value_shape: &[usize],
        heads: usize,
    ) -> Result<Self, NeuralNetworkModulePartTwoError> {
        let (query_tokens, batch, embed, unbatched) = sequence_shape(query_shape)?;
        let (key_tokens, key_batch, key_embed, key_unbatched) = sequence_shape(key_shape)?;
        let (value_tokens, value_batch, value_embed, value_unbatched) =
            sequence_shape(value_shape)?;
        if heads == 0
            || embed == 0
            || !embed.is_multiple_of(heads)
            || batch != key_batch
            || batch != value_batch
            || embed != key_embed
            || embed != value_embed
            || key_tokens != value_tokens
            || unbatched != key_unbatched
            || unbatched != value_unbatched
        {
            return invalid(
                MULTIHEAD_ATTENTION_OPERATION_ID,
                "query, key, value, batch, embedding, or head geometry is incompatible",
            );
        }
        require_length(
            query.len(),
            checked_product(query_shape, "multihead query shape")?,
            MULTIHEAD_ATTENTION_OPERATION_ID,
            "query",
        )?;
        require_length(
            key.len(),
            checked_product(key_shape, "multihead key shape")?,
            MULTIHEAD_ATTENTION_OPERATION_ID,
            "key",
        )?;
        require_length(
            value.len(),
            checked_product(value_shape, "multihead value shape")?,
            MULTIHEAD_ATTENTION_OPERATION_ID,
            "value",
        )?;
        Ok(Self {
            query_tokens,
            key_tokens,
            batch,
            embed,
            heads,
            head_dimension: embed / heads,
            unbatched,
        })
    }

    fn request(&self) -> AttentionKernelRequest {
        AttentionKernelRequest {
            kind: AttentionKernelKind::ReferenceSdp,
            device: DeviceId::CPU,
            layout: AttentionLayout::Nhd,
            shape: AttentionShape {
                batch: self.batch,
                query_tokens: self.query_tokens,
                key_tokens: self.key_tokens,
                heads: self.heads,
                head_dimension: self.head_dimension,
                value_dimension: self.head_dimension,
            },
            scale: None,
            causal: false,
            dropout_probability: 0.0,
        }
    }

    fn to_nhd(
        &self,
        input: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
        let mut output = vec![0.0; input.len()];
        for batch in 0..self.batch {
            for token in 0..tokens {
                for head in 0..self.heads {
                    for dimension in 0..self.head_dimension {
                        let feature = head * self.head_dimension + dimension;
                        let input_index = if self.unbatched {
                            token * self.embed + feature
                        } else {
                            (token * self.batch + batch) * self.embed + feature
                        };
                        let output_index =
                            ((batch * tokens + token) * self.heads + head) * self.head_dimension
                                + dimension;
                        output[output_index] = input[input_index];
                    }
                }
            }
        }
        Ok(output)
    }

    fn from_nhd(
        &self,
        input: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, NeuralNetworkModulePartTwoError> {
        let expected = self
            .batch
            .checked_mul(tokens)
            .and_then(|value| value.checked_mul(self.embed))
            .ok_or(NeuralNetworkModulePartTwoError::ShapeOverflow(
                "multihead output shape",
            ))?;
        require_length(
            input.len(),
            expected,
            MULTIHEAD_ATTENTION_OPERATION_ID,
            "canonical attention output",
        )?;
        let mut output = vec![0.0; input.len()];
        for batch in 0..self.batch {
            for token in 0..tokens {
                for head in 0..self.heads {
                    for dimension in 0..self.head_dimension {
                        let feature = head * self.head_dimension + dimension;
                        let input_index =
                            ((batch * tokens + token) * self.heads + head) * self.head_dimension
                                + dimension;
                        let output_index = if self.unbatched {
                            token * self.embed + feature
                        } else {
                            (token * self.batch + batch) * self.embed + feature
                        };
                        output[output_index] = input[input_index];
                    }
                }
            }
        }
        Ok(output)
    }
}

pub(crate) struct Pad2dGeometry {
    outer: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    padding: [usize; 4],
    output_shape: Vec<usize>,
    mode: ConvolutionPaddingMode,
}

impl Pad2dGeometry {
    pub(crate) fn new(
        input: &[f32],
        input_shape: &[usize],
        padding: [usize; 4],
        operation: &'static str,
        mode: ConvolutionPaddingMode,
    ) -> Result<Self, NeuralNetworkModulePartTwoError> {
        let [left, right, top, bottom] = padding;
        let (outer_shape, input_height, input_width) = match input_shape {
            [_, height, width] => (&input_shape[..1], *height, *width),
            [_, _, height, width] => (&input_shape[..2], *height, *width),
            _ => {
                return invalid(
                    operation,
                    "two-dimensional padding expects CHW or NCHW input",
                );
            }
        };
        if input_height == 0 || input_width == 0 {
            return invalid(
                operation,
                "two-dimensional padding requires nonempty spatial dimensions",
            );
        }
        require_length(
            input.len(),
            checked_product(input_shape, "replication-pad input shape")?,
            operation,
            "input",
        )?;
        let output_height = input_height
            .checked_add(top)
            .and_then(|value| value.checked_add(bottom))
            .ok_or(NeuralNetworkModulePartTwoError::ShapeOverflow(
                "replication-pad height",
            ))?;
        let output_width = input_width
            .checked_add(left)
            .and_then(|value| value.checked_add(right))
            .ok_or(NeuralNetworkModulePartTwoError::ShapeOverflow(
                "replication-pad width",
            ))?;
        let mut output_shape = outer_shape.to_vec();
        output_shape.extend_from_slice(&[output_height, output_width]);
        Ok(Self {
            outer: checked_product(outer_shape, "replication-pad outer shape")?,
            input_height,
            input_width,
            output_height,
            output_width,
            padding,
            output_shape,
            mode,
        })
    }

    pub(crate) fn output_count(&self) -> Result<usize, NeuralNetworkModulePartTwoError> {
        checked_product(&self.output_shape, "replication-pad output shape")
    }

    pub(crate) fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    pub(crate) fn for_each_mapping(
        &self,
        context: &ExecutionContext<'_>,
        mut visit: impl FnMut(Option<usize>, usize) -> Result<(), NeuralNetworkModulePartTwoError>,
    ) -> Result<(), NeuralNetworkModulePartTwoError> {
        let input_plane = self
            .input_height
            .checked_mul(self.input_width)
            .ok_or(NeuralNetworkModulePartTwoError::ShapeOverflow(
                "replication-pad input plane",
            ))?;
        let output_plane = self
            .output_height
            .checked_mul(self.output_width)
            .ok_or(NeuralNetworkModulePartTwoError::ShapeOverflow(
                "replication-pad output plane",
            ))?;
        for outer in 0..self.outer {
            for output_y in 0..self.output_height {
                let input_y = map_padded_coordinate(
                    output_y,
                    self.padding[2],
                    self.input_height,
                    self.mode,
                )?;
                for output_x in 0..self.output_width {
                    let output_index = outer * output_plane + output_y * self.output_width + output_x;
                    if output_index.is_multiple_of(1_024) {
                        context.cancellation.check()?;
                    }
                    let input_x = map_padded_coordinate(
                        output_x,
                        self.padding[0],
                        self.input_width,
                        self.mode,
                    )?;
                    let input_index = match (input_y, input_x) {
                        (Some(input_y), Some(input_x)) => Some(
                            outer * input_plane + input_y * self.input_width + input_x,
                        ),
                        _ => None,
                    };
                    visit(input_index, output_index)?;
                }
            }
        }
        Ok(())
    }
}

fn sequence_shape(
    shape: &[usize],
) -> Result<(usize, usize, usize, bool), NeuralNetworkModulePartTwoError> {
    match shape {
        [tokens, embed] => Ok((*tokens, 1, *embed, true)),
        [tokens, batch, embed] => Ok((*tokens, *batch, *embed, false)),
        _ => invalid(
            MULTIHEAD_ATTENTION_OPERATION_ID,
            "multihead attention expects sequence-major LE or LNE tensors",
        ),
    }
}

fn require_rank(
    shape: &[usize],
    expected: usize,
    operation: &'static str,
) -> Result<(), NeuralNetworkModulePartTwoError> {
    if shape.len() != expected {
        return invalid(operation, format!("expected rank {expected}"));
    }
    Ok(())
}

fn channel_count(
    shape: &[usize],
    operation: &'static str,
) -> Result<usize, NeuralNetworkModulePartTwoError> {
    shape
        .get(1)
        .copied()
        .filter(|channels| *channels != 0)
        .ok_or_else(|| invalid_error(operation, "channel dimension must be nonzero"))
}

fn validate_delta(delta: f32) -> Result<(), NeuralNetworkModulePartTwoError> {
    if !delta.is_finite() || delta <= 0.0 {
        return invalid(HUBER_LOSS_OPERATION_ID, "delta must be finite and positive");
    }
    Ok(())
}

fn scale_in_place(
    values: &mut [f32],
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<(), NeuralNetworkModulePartTwoError> {
    for (index, value) in values.iter_mut().enumerate() {
        if index.is_multiple_of(1_024) {
            context.cancellation.check()?;
        }
        *value *= scale;
    }
    context.cancellation.check()?;
    Ok(())
}

fn require_cpu(
    device: DeviceId,
    operation: &'static str,
) -> Result<(), NeuralNetworkModulePartTwoError> {
    if device != DeviceId::CPU {
        return invalid(operation, format!("unsupported device {device:?}"));
    }
    Ok(())
}

fn require_length(
    actual: usize,
    expected: usize,
    operation: &'static str,
    name: &'static str,
) -> Result<(), NeuralNetworkModulePartTwoError> {
    if actual != expected {
        return invalid(
            operation,
            format!("{name} expected {expected} values, got {actual}"),
        );
    }
    Ok(())
}

fn checked_product(
    shape: &[usize],
    name: &'static str,
) -> Result<usize, NeuralNetworkModulePartTwoError> {
    shape.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or(NeuralNetworkModulePartTwoError::ShapeOverflow(name))
    })
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, NeuralNetworkModulePartTwoError> {
    Err(invalid_error(operation, reason))
}

fn invalid_error(
    operation: &'static str,
    reason: impl Into<String>,
) -> NeuralNetworkModulePartTwoError {
    NeuralNetworkModulePartTwoError::Invalid {
        operation,
        reason: reason.into(),
    }
}
