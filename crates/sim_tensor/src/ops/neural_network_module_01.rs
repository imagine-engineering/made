use crate::{
    CpuBackend, DeviceId, ExecutionContext, ResizeCrop, ResizeMode, ResizeSpec, Tensor,
    generated_activation_normalization_functional_01::{
        AffineVjp, FunctionalError,
        group_norm_jvp_with_context_exact_native as canonical_group_norm_jvp,
        group_norm_vjp_with_context_exact_native as canonical_group_norm_vjp,
        group_norm_with_context_exact_native as canonical_group_norm,
        layer_norm_jvp_with_context_exact_native as canonical_layer_norm_jvp,
        layer_norm_vjp_with_context_exact_native as canonical_layer_norm_vjp,
        layer_norm_with_context_exact_native as canonical_layer_norm,
        silu_jvp_with_context_exact_native as canonical_silu_jvp,
        silu_vjp_with_context_exact_native as canonical_silu_vjp,
        silu_with_context_exact_native as canonical_silu,
        softmax_jvp_with_context_exact_native as canonical_softmax_jvp,
        softmax_vjp_with_context_exact_native as canonical_softmax_vjp,
        softmax_with_context_exact_native as canonical_softmax,
    },
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionPaddingMode, ConvolutionVjp,
        OperatorIndirectionError, TensorValues,
        convolution_jvp_with_context_exact_native as canonical_convolution_jvp,
        convolution_vjp_with_context_exact_native as canonical_convolution_vjp,
        convolution_with_context_exact_native as canonical_convolution,
    },
    generated_elementwise_or_runtime_operation_02::{
        ElementwiseRuntimePartTwoError,
        tanh_jvp_with_context_exact_native as canonical_tanh_jvp,
        tanh_vjp_with_context_exact_native as canonical_tanh_vjp,
        tanh_with_context_exact_native as canonical_tanh,
    },
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError,
        resize_with_coordinate_transform_with_context_exact_native as canonical_resize,
    },
};
use thiserror::Error;

pub const AVG_POOL_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-378824FF198F";
pub const ADAPTIVE_AVG_POOL_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-6A8710D48911";
pub const AVG_POOL_3D_OPERATION_ID: &str = "COMFY-TENSOR-OP-835CE6AE654F";
pub const BUFFER_OPERATION_ID: &str = "COMFY-TENSOR-OP-2B09E55DCFFA";
pub const CONV_1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-5B3DCD30FC9C";
pub const GROUP_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-4029AD87847D";
pub const LAYER_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-36234382A497";
pub const PRELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-5A7C1AC03892";
pub const SEQUENTIAL_OPERATION_ID: &str = "COMFY-TENSOR-OP-163D5E4B774E";
pub const SILU_OPERATION_ID: &str = "COMFY-TENSOR-OP-46DAF04A6B91";
pub const SMOOTH_L1_LOSS_OPERATION_ID: &str = "COMFY-TENSOR-OP-0E92B0EA1CF0";
pub const SOFTMAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-0E602E58360A";
pub const TANH_OPERATION_ID: &str = "COMFY-TENSOR-OP-1FDC96D7E7C2";
pub const UPSAMPLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-3F14ADC9E576";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NeuralNetworkModuleError {
    #[error(transparent)]
    Tensor(#[from] crate::TensorError),
    #[error(transparent)]
    Operator(OperatorIndirectionError),
    #[error(transparent)]
    Functional(FunctionalError),
    #[error(transparent)]
    Elementwise(ElementwiseRuntimePartTwoError),
    #[error("canonical resize operation failed: {0}")]
    Resize(String),
    #[error("neural-network module operation was cancelled")]
    Cancelled,
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("operation {operation} is unavailable for device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for NeuralNetworkModuleError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<OperatorIndirectionError> for NeuralNetworkModuleError {
    fn from(error: OperatorIndirectionError) -> Self {
        match error {
            OperatorIndirectionError::Cancelled => Self::Cancelled,
            error => Self::Operator(error),
        }
    }
}

impl From<FunctionalError> for NeuralNetworkModuleError {
    fn from(error: FunctionalError) -> Self {
        match error {
            FunctionalError::Cancelled => Self::Cancelled,
            error => Self::Functional(error),
        }
    }
}

impl From<ElementwiseRuntimePartTwoError> for NeuralNetworkModuleError {
    fn from(error: ElementwiseRuntimePartTwoError) -> Self {
        match error {
            ElementwiseRuntimePartTwoError::Cancelled => Self::Cancelled,
            error => Self::Elementwise(error),
        }
    }
}

impl From<ExternalTensorKernelPartOneError> for NeuralNetworkModuleError {
    fn from(error: ExternalTensorKernelPartOneError) -> Self {
        match error {
            ExternalTensorKernelPartOneError::Cancelled => Self::Cancelled,
            error => Self::Resize(error.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AveragePool2dVjp {
    pub input: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PReluVjp {
    pub input: Vec<f32>,
    pub weight: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LossReduction {
    None,
    Sum,
    #[default]
    Mean,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpsampleMode {
    Nearest,
    Bilinear,
}

pub fn average_pool_2d_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(AVG_POOL_2D_OPERATION_ID, device)?;
    let geometry = AveragePoolGeometry::new(
        input,
        input_shape,
        &kernel_size,
        &stride,
        AVG_POOL_2D_OPERATION_ID,
    )?;
    let mut output = zeroed(geometry.output_count()?, "average-pool output")?;
    geometry.for_each_connection(context, |input_index, output_index, scale| {
        output[output_index] = input[input_index].mul_add(scale, output[output_index]);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: geometry.output_shape,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_2d_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AveragePool2dVjp, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(AVG_POOL_2D_OPERATION_ID, device)?;
    let geometry = AveragePoolGeometry::new(
        input,
        input_shape,
        &kernel_size,
        &stride,
        AVG_POOL_2D_OPERATION_ID,
    )?;
    require_length(
        output_gradient.len(),
        geometry.output_count()?,
        AVG_POOL_2D_OPERATION_ID,
        "output gradient",
    )?;
    let mut input_gradient = zeroed(input.len(), "average-pool input gradient")?;
    geometry.for_each_connection(context, |input_index, output_index, scale| {
        input_gradient[input_index] =
            output_gradient[output_index].mul_add(scale, input_gradient[input_index]);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(AveragePool2dVjp {
        input: input_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_2d_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModuleError> {
    average_pool_2d_with_context_exact_native(
        backend,
        input_tangent,
        input_shape,
        kernel_size,
        stride,
        device,
        context,
    )
}

pub fn average_pool_3d_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 3],
    stride: [usize; 3],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(AVG_POOL_3D_OPERATION_ID, device)?;
    let geometry = AveragePoolGeometry::new(
        input,
        input_shape,
        &kernel_size,
        &stride,
        AVG_POOL_3D_OPERATION_ID,
    )?;
    let mut output = zeroed(geometry.output_count()?, "average-pool-3d output")?;
    geometry.for_each_connection(context, |input_index, output_index, scale| {
        output[output_index] = input[input_index].mul_add(scale, output[output_index]);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: geometry.output_shape,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_3d_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 3],
    stride: [usize; 3],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AveragePool2dVjp, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(AVG_POOL_3D_OPERATION_ID, device)?;
    let geometry = AveragePoolGeometry::new(
        input,
        input_shape,
        &kernel_size,
        &stride,
        AVG_POOL_3D_OPERATION_ID,
    )?;
    require_length(
        output_gradient.len(),
        geometry.output_count()?,
        AVG_POOL_3D_OPERATION_ID,
        "output gradient",
    )?;
    let mut input_gradient = zeroed(input.len(), "average-pool-3d input gradient")?;
    geometry.for_each_connection(context, |input_index, output_index, scale| {
        input_gradient[input_index] =
            output_gradient[output_index].mul_add(scale, input_gradient[input_index]);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(AveragePool2dVjp {
        input: input_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn average_pool_3d_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &[f32],
    input_shape: &[usize],
    kernel_size: [usize; 3],
    stride: [usize; 3],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModuleError> {
    average_pool_3d_with_context_exact_native(
        backend,
        input_tangent,
        input_shape,
        kernel_size,
        stride,
        device,
        context,
    )
}

pub fn adaptive_average_pool_2d_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    output_size: [usize; 2],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(ADAPTIVE_AVG_POOL_2D_OPERATION_ID, device)?;
    let geometry = AdaptiveAveragePool2dGeometry::new(input, input_shape, output_size)?;
    let mut output = zeroed(geometry.output_count()?, "adaptive-average-pool output")?;
    geometry.for_each_connection(context, |input_index, output_index, scale| {
        output[output_index] = input[input_index].mul_add(scale, output[output_index]);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(TensorValues {
        values: output,
        shape: geometry.output_shape,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn adaptive_average_pool_2d_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    output_size: [usize; 2],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AveragePool2dVjp, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(ADAPTIVE_AVG_POOL_2D_OPERATION_ID, device)?;
    let geometry = AdaptiveAveragePool2dGeometry::new(input, input_shape, output_size)?;
    require_length(
        output_gradient.len(),
        geometry.output_count()?,
        ADAPTIVE_AVG_POOL_2D_OPERATION_ID,
        "output gradient",
    )?;
    let mut input_gradient = zeroed(input.len(), "adaptive-average-pool input gradient")?;
    geometry.for_each_connection(context, |input_index, output_index, scale| {
        input_gradient[input_index] =
            output_gradient[output_index].mul_add(scale, input_gradient[input_index]);
        Ok(())
    })?;
    context.cancellation.check()?;
    Ok(AveragePool2dVjp {
        input: input_gradient,
    })
}

pub fn adaptive_average_pool_2d_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &[f32],
    input_shape: &[usize],
    output_size: [usize; 2],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModuleError> {
    adaptive_average_pool_2d_with_context_exact_native(
        backend,
        input_tangent,
        input_shape,
        output_size,
        device,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    let geometry = conv1d_geometry(stride, padding, dilation, groups)?;
    Ok(canonical_convolution(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        &geometry,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    output_gradient: &[f32],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<ConvolutionVjp, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    let geometry = conv1d_geometry(stride, padding, dilation, groups)?;
    Ok(canonical_convolution_vjp(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        output_gradient,
        &geometry,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<TensorValues, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    let geometry = conv1d_geometry(stride, padding, dilation, groups)?;
    Ok(canonical_convolution_jvp(
        input,
        input_tangent,
        input_shape,
        weight,
        weight_tangent,
        weight_shape,
        bias,
        bias_tangent,
        &geometry,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    groups: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_group_norm(
        backend, input, shape, groups, weight, bias, epsilon, device, context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    groups: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AffineVjp, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_group_norm_vjp(
        backend,
        input,
        output_gradient,
        shape,
        groups,
        weight,
        bias,
        epsilon,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    groups: usize,
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_group_norm_jvp(
        backend,
        input,
        input_tangent,
        shape,
        groups,
        weight,
        weight_tangent,
        bias_tangent,
        epsilon,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_layer_norm(
        backend,
        input,
        shape,
        normalized_shape,
        weight,
        bias,
        epsilon,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AffineVjp, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_layer_norm_vjp(
        backend,
        input,
        output_gradient,
        shape,
        normalized_shape,
        weight,
        bias,
        epsilon,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_layer_norm_jvp(
        backend,
        input,
        input_tangent,
        shape,
        normalized_shape,
        weight,
        weight_tangent,
        bias_tangent,
        epsilon,
        device,
        context,
    )?)
}

pub fn prelu_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(PRELU_OPERATION_ID, device)?;
    validate_prelu(input, input_shape, weight)?;
    let mut output = reserved(input.len(), "PReLU output")?;
    for (index, input) in input.iter().copied().enumerate() {
        check_periodically(index, context)?;
        let weight = weight[prelu_weight_index(index, input_shape, weight.len())?];
        output.push(if input > 0.0 { input } else { input * weight });
    }
    context.cancellation.check()?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn prelu_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<PReluVjp, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(PRELU_OPERATION_ID, device)?;
    validate_prelu(input, input_shape, weight)?;
    require_length(
        output_gradient.len(),
        input.len(),
        PRELU_OPERATION_ID,
        "output gradient",
    )?;
    let mut input_gradient = zeroed(input.len(), "PReLU input gradient")?;
    let mut weight_gradient = zeroed(weight.len(), "PReLU weight gradient")?;
    for (index, (input, gradient)) in input
        .iter()
        .copied()
        .zip(output_gradient.iter().copied())
        .enumerate()
    {
        check_periodically(index, context)?;
        let weight_index = prelu_weight_index(index, input_shape, weight.len())?;
        if input > 0.0 {
            input_gradient[index] = gradient;
        } else {
            input_gradient[index] = gradient * weight[weight_index];
            weight_gradient[weight_index] =
                input.mul_add(gradient, weight_gradient[weight_index]);
        }
    }
    context.cancellation.check()?;
    Ok(PReluVjp {
        input: input_gradient,
        weight: weight_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prelu_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_tangent: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(PRELU_OPERATION_ID, device)?;
    validate_prelu(input, input_shape, weight)?;
    require_length(
        input_tangent.len(),
        input.len(),
        PRELU_OPERATION_ID,
        "input tangent",
    )?;
    require_length(
        weight_tangent.len(),
        weight.len(),
        PRELU_OPERATION_ID,
        "weight tangent",
    )?;
    let mut output = reserved(input.len(), "PReLU tangent")?;
    for (index, (input, tangent)) in input
        .iter()
        .copied()
        .zip(input_tangent.iter().copied())
        .enumerate()
    {
        check_periodically(index, context)?;
        let weight_index = prelu_weight_index(index, input_shape, weight.len())?;
        output.push(if input > 0.0 {
            tangent
        } else {
            weight[weight_index].mul_add(tangent, input * weight_tangent[weight_index])
        });
    }
    context.cancellation.check()?;
    Ok(output)
}

pub fn silu_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_silu(backend, input, device, context)?)
}

pub fn silu_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_silu_vjp(
        backend,
        input,
        output_gradient,
        device,
        context,
    )?)
}

pub fn silu_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_silu_jvp(
        backend,
        input,
        input_tangent,
        device,
        context,
    )?)
}

pub fn smooth_l1_loss_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    target: &[f32],
    beta: f32,
    reduction: LossReduction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(SMOOTH_L1_LOSS_OPERATION_ID, device)?;
    validate_smooth_l1(input, target, beta)?;
    let mut losses = reserved(input.len(), "smooth-L1 output")?;
    for (index, (input, target)) in input.iter().zip(target).enumerate() {
        check_periodically(index, context)?;
        let difference = input - target;
        let absolute = difference.abs();
        losses.push(if beta == 0.0 || absolute >= beta {
            absolute - 0.5 * beta
        } else {
            0.5 * difference * difference / beta
        });
    }
    context.cancellation.check()?;
    reduce_loss(losses, reduction)
}

#[allow(clippy::too_many_arguments)]
pub fn smooth_l1_loss_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    target: &[f32],
    beta: f32,
    reduction: LossReduction,
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_cpu(SMOOTH_L1_LOSS_OPERATION_ID, device)?;
    validate_smooth_l1(input, target, beta)?;
    let expected_gradient = if reduction == LossReduction::None {
        input.len()
    } else {
        1
    };
    require_length(
        output_gradient.len(),
        expected_gradient,
        SMOOTH_L1_LOSS_OPERATION_ID,
        "output gradient",
    )?;
    let mean_scale = if reduction == LossReduction::Mean && !input.is_empty() {
        1.0 / input.len() as f32
    } else {
        1.0
    };
    let mut gradient = reserved(input.len(), "smooth-L1 input gradient")?;
    for (index, (input, target)) in input.iter().zip(target).enumerate() {
        check_periodically(index, context)?;
        let difference = input - target;
        let absolute = difference.abs();
        let derivative = if beta != 0.0 && absolute < beta {
            difference / beta
        } else {
            difference.signum()
        };
        let upstream = if reduction == LossReduction::None {
            output_gradient[index]
        } else {
            output_gradient[0]
        };
        gradient.push(derivative * upstream * mean_scale);
    }
    context.cancellation.check()?;
    Ok(gradient)
}

#[allow(clippy::too_many_arguments)]
pub fn smooth_l1_loss_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    target: &[f32],
    target_tangent: &[f32],
    beta: f32,
    reduction: LossReduction,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    require_length(
        input_tangent.len(),
        input.len(),
        SMOOTH_L1_LOSS_OPERATION_ID,
        "input tangent",
    )?;
    require_length(
        target_tangent.len(),
        target.len(),
        SMOOTH_L1_LOSS_OPERATION_ID,
        "target tangent",
    )?;
    let upstream = if reduction == LossReduction::None {
        vec![1.0; input.len()]
    } else {
        vec![1.0]
    };
    let derivative = smooth_l1_loss_vjp_with_context_exact_native(
        backend,
        input,
        target,
        beta,
        reduction,
        &upstream,
        device,
        context,
    )?;
    let mut tangent = reserved(input.len(), "smooth-L1 tangent")?;
    for (index, derivative) in derivative.iter().copied().enumerate() {
        check_periodically(index, context)?;
        tangent.push(derivative * (input_tangent[index] - target_tangent[index]));
    }
    match reduction {
        LossReduction::None => Ok(tangent),
        LossReduction::Sum | LossReduction::Mean => {
            Ok(vec![tangent.iter().copied().sum()])
        }
    }
}

pub fn softmax_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_softmax(
        backend, input, shape, dimension, device, context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn softmax_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_softmax_vjp(
        backend,
        input,
        output_gradient,
        shape,
        dimension,
        device,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn softmax_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_softmax_jvp(
        backend,
        input,
        input_tangent,
        shape,
        dimension,
        device,
        context,
    )?)
}

pub fn tanh_module_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_tanh(backend, input, context)?)
}

pub fn tanh_module_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_tanh_vjp(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn tanh_module_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    Ok(canonical_tanh_jvp(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn upsample_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_height: u64,
    output_width: u64,
    mode: UpsampleMode,
    align_corners: Option<bool>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    match (mode, align_corners) {
        (UpsampleMode::Nearest, None | Some(false)) => {}
        (UpsampleMode::Nearest, Some(true)) => {
            return invalid(
                UPSAMPLE_OPERATION_ID,
                "align_corners is invalid for nearest-neighbor upsampling",
            );
        }
        (UpsampleMode::Bilinear, None | Some(false) | Some(true)) => {}
    }
    let resize_mode = match mode {
        UpsampleMode::Nearest => ResizeMode::NearestExact,
        UpsampleMode::Bilinear => ResizeMode::Bilinear,
    };
    Ok(canonical_resize(
        backend,
        input,
        output_height,
        output_width,
        resize_mode,
        false,
        align_corners.unwrap_or(false),
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn upsample_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    output_height: u64,
    output_width: u64,
    mode: UpsampleMode,
    align_corners: Option<bool>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModuleError> {
    context.cancellation.check()?;
    if input.descriptor().rank() != 4 || output_gradient.descriptor().rank() != 4 {
        return invalid(
            UPSAMPLE_OPERATION_ID,
            "upsample VJP expects rank-four NCHW input and output gradient",
        );
    }
    if mode == UpsampleMode::Nearest && align_corners == Some(true) {
        return invalid(
            UPSAMPLE_OPERATION_ID,
            "align_corners is invalid for nearest-neighbor upsampling",
        );
    }
    let resize_mode = match mode {
        UpsampleMode::Nearest => ResizeMode::NearestExact,
        UpsampleMode::Bilinear => ResizeMode::Bilinear,
    };
    Ok(backend.resize_vjp(
        ResizeSpec {
            width: output_width,
            height: output_height,
            mode: resize_mode,
            crop: ResizeCrop::Disabled,
            antialias: false,
            align_corners: align_corners.unwrap_or(false),
        },
        input,
        output_gradient,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn upsample_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    output_height: u64,
    output_width: u64,
    mode: UpsampleMode,
    align_corners: Option<bool>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NeuralNetworkModuleError> {
    upsample_with_context_exact_native(
        backend,
        input_tangent,
        output_height,
        output_width,
        mode,
        align_corners,
        context,
    )
}

fn conv1d_geometry(
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvolutionGeometry, NeuralNetworkModuleError> {
    Ok(ConvolutionGeometry::new_with_padding_mode(
        1,
        vec![stride],
        vec![padding],
        vec![dilation],
        groups,
        false,
        vec![0],
        ConvolutionPaddingMode::Zeros,
    )?)
}

pub(crate) struct AveragePoolGeometry {
    operation: &'static str,
    batch: usize,
    channels: usize,
    input_spatial: Vec<usize>,
    output_spatial: Vec<usize>,
    output_shape: Vec<usize>,
    kernel_size: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
}

impl AveragePoolGeometry {
    pub(crate) fn new(
        input: &[f32],
        input_shape: &[usize],
        kernel_size: &[usize],
        stride: &[usize],
        operation: &'static str,
    ) -> Result<Self, NeuralNetworkModuleError> {
        Self::new_extended(
            input,
            input_shape,
            kernel_size,
            stride,
            &vec![0; kernel_size.len()],
            &vec![1; kernel_size.len()],
            false,
            operation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_extended(
        input: &[f32],
        input_shape: &[usize],
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        ceil_mode: bool,
        operation: &'static str,
    ) -> Result<Self, NeuralNetworkModuleError> {
        if kernel_size.is_empty() || kernel_size.len() != stride.len() {
            return invalid(operation, "pooling kernel and stride ranks must match");
        }
        if padding.len() != kernel_size.len() || dilation.len() != kernel_size.len() {
            return invalid(operation, "pooling padding and dilation ranks must match");
        }
        let spatial_dimensions = kernel_size.len();
        let (batch, channels, spatial_start, batched) = match input_shape.len() {
            rank if rank == spatial_dimensions + 1 => (1, input_shape[0], 1, false),
            rank if rank == spatial_dimensions + 2 => (input_shape[0], input_shape[1], 2, true),
            _ => return invalid(operation, "average pooling input rank is invalid"),
        };
        if kernel_size.contains(&0) || stride.contains(&0) || dilation.contains(&0) {
            return invalid(operation, "kernel, stride, and dilation dimensions must be nonzero");
        }
        let input_spatial = input_shape[spatial_start..].to_vec();
        require_length(
            input.len(),
            checked_product(input_shape, "average-pool input shape")?,
            operation,
            "input",
        )?;
        let mut output_spatial = Vec::with_capacity(spatial_dimensions);
        for dimension in 0..spatial_dimensions {
            let effective_kernel = dilation[dimension]
                .checked_mul(kernel_size[dimension] - 1)
                .and_then(|value| value.checked_add(1))
                .ok_or(NeuralNetworkModuleError::ShapeOverflow("pooling effective kernel"))?;
            let padded_input = input_spatial[dimension]
                .checked_add(padding[dimension].checked_mul(2).ok_or(
                    NeuralNetworkModuleError::ShapeOverflow("pooling padding"),
                )?)
                .ok_or(NeuralNetworkModuleError::ShapeOverflow("pooling padded input"))?;
            if padded_input < effective_kernel {
                return invalid(operation, "pooling kernel exceeds the padded input");
            }
            let numerator = padded_input - effective_kernel;
            let rounded = if ceil_mode {
                numerator
                    .checked_add(stride[dimension] - 1)
                    .ok_or(NeuralNetworkModuleError::ShapeOverflow("pooling ceil output"))?
            } else {
                numerator
            };
            let mut output = rounded / stride[dimension] + 1;
            let admissible_start = input_spatial[dimension]
                .checked_add(padding[dimension])
                .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                    "pooling admissible start",
                ))?;
            if output > 0
                && (output - 1)
                    .checked_mul(stride[dimension])
                    .is_some_and(|start| start >= admissible_start)
            {
                output -= 1;
            }
            output_spatial.push(output);
        }
        let mut output_shape = if batched {
            vec![batch, channels]
        } else {
            vec![channels]
        };
        output_shape.extend_from_slice(&output_spatial);
        Ok(Self {
            operation,
            batch,
            channels,
            input_spatial,
            output_spatial,
            output_shape,
            kernel_size: kernel_size.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
            dilation: dilation.to_vec(),
        })
    }

    pub(crate) fn output_count(&self) -> Result<usize, NeuralNetworkModuleError> {
        checked_product(&self.output_shape, "average-pool output shape")
    }

    pub(crate) fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    pub(crate) fn for_each_connection(
        &self,
        context: &ExecutionContext<'_>,
        mut visit: impl FnMut(usize, usize, f32) -> Result<(), NeuralNetworkModuleError>,
    ) -> Result<(), NeuralNetworkModuleError> {
        self.for_each_connection_with_divisor(context, true, None, |input, output, scale| {
            visit(input, output, scale)
        })
    }

    pub(crate) fn for_each_connection_with_divisor(
        &self,
        context: &ExecutionContext<'_>,
        count_include_pad: bool,
        divisor_override: Option<usize>,
        mut visit: impl FnMut(usize, usize, f32) -> Result<(), NeuralNetworkModuleError>,
    ) -> Result<(), NeuralNetworkModuleError> {
        if divisor_override == Some(0) {
            return invalid(
                self.operation,
                "average-pool divisor override must be nonzero",
            );
        }
        let input_spatial_count = checked_product(&self.input_spatial, "pool input spatial")?;
        let output_spatial_count = checked_product(&self.output_spatial, "pool output spatial")?;
        let kernel_count = checked_product(&self.kernel_size, "average-pool kernel")?;
        for batch in 0..self.batch {
            for channel in 0..self.channels {
                let plane = batch * self.channels + channel;
                for output_linear in 0..output_spatial_count {
                    let output_index = plane * output_spatial_count + output_linear;
                    check_periodically(output_index, context)?;
                    let divisor = match divisor_override {
                        Some(divisor) => divisor,
                        None => {
                            let mut count = 0usize;
                            for kernel_linear in 0..kernel_count {
                                let counted = if count_include_pad {
                                    self.connection_is_within_explicit_padding(
                                        output_linear,
                                        kernel_linear,
                                    )?
                                } else {
                                    self.connection_input_index(
                                        plane,
                                        input_spatial_count,
                                        output_linear,
                                        kernel_linear,
                                    )?
                                    .is_some()
                                };
                                if counted {
                                    count = count.checked_add(1).ok_or(
                                        NeuralNetworkModuleError::ShapeOverflow(
                                            "average-pool divisor",
                                        ),
                                    )?;
                                }
                            }
                            count
                        }
                    };
                    if divisor == 0 {
                        return invalid(
                            self.operation,
                            "average-pool output window has no values",
                        );
                    }
                    let scale = (divisor as f32).recip();
                    for kernel_linear in 0..kernel_count {
                        let Some(input_index) = self.connection_input_index(
                            plane,
                            input_spatial_count,
                            output_linear,
                            kernel_linear,
                        )? else {
                            continue;
                        };
                        visit(input_index, output_index, scale)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn connection_is_within_explicit_padding(
        &self,
        output_linear: usize,
        kernel_linear: usize,
    ) -> Result<bool, NeuralNetworkModuleError> {
        let mut remaining_output = output_linear;
        let mut remaining_kernel = kernel_linear;
        for dimension in (0..self.input_spatial.len()).rev() {
            let output_extent = self.output_spatial.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool output extent"),
            )?;
            let kernel_extent = self.kernel_size.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool kernel extent"),
            )?;
            let output_coordinate = remaining_output % output_extent;
            remaining_output /= output_extent;
            let kernel_coordinate = remaining_kernel % kernel_extent;
            remaining_kernel /= kernel_extent;
            let stride = self.stride.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool stride"),
            )?;
            let dilation = self.dilation.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool dilation"),
            )?;
            let padding = self.padding.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool padding"),
            )?;
            let padded_coordinate = output_coordinate
                .checked_mul(stride)
                .and_then(|value| {
                    kernel_coordinate
                        .checked_mul(dilation)
                        .and_then(|kernel| value.checked_add(kernel))
                })
                .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                    "average-pool coordinate",
                ))?;
            let padded_extent = self
                .input_spatial
                .get(dimension)
                .copied()
                .and_then(|extent| padding.checked_mul(2).and_then(|pad| extent.checked_add(pad)))
                .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                    "average-pool padded extent",
                ))?;
            if padded_coordinate >= padded_extent {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn connection_input_index(
        &self,
        plane: usize,
        input_spatial_count: usize,
        output_linear: usize,
        kernel_linear: usize,
    ) -> Result<Option<usize>, NeuralNetworkModuleError> {
        let mut remaining_output = output_linear;
        let mut remaining_kernel = kernel_linear;
        let mut input_spatial_index = 0usize;
        let mut input_stride = 1usize;
        for dimension in (0..self.input_spatial.len()).rev() {
            let output_extent = self.output_spatial.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool output extent"),
            )?;
            let kernel_extent = self.kernel_size.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool kernel extent"),
            )?;
            let output_coordinate = remaining_output % output_extent;
            remaining_output /= output_extent;
            let kernel_coordinate = remaining_kernel % kernel_extent;
            remaining_kernel /= kernel_extent;
            let stride = self.stride.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool stride"),
            )?;
            let dilation = self.dilation.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool dilation"),
            )?;
            let padding = self.padding.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool padding"),
            )?;
            let input_extent = self.input_spatial.get(dimension).copied().ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool input extent"),
            )?;
            let padded_coordinate = output_coordinate
                .checked_mul(stride)
                .and_then(|value| {
                    kernel_coordinate
                        .checked_mul(dilation)
                        .and_then(|kernel| value.checked_add(kernel))
                })
                .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                    "average-pool coordinate",
                ))?;
            if padded_coordinate < padding {
                return Ok(None);
            }
            let input_coordinate = padded_coordinate - padding;
            if input_coordinate >= input_extent {
                return Ok(None);
            }
            input_spatial_index = input_spatial_index
                .checked_add(input_coordinate.checked_mul(input_stride).ok_or(
                    NeuralNetworkModuleError::ShapeOverflow("average-pool input index"),
                )?)
                .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                    "average-pool input index",
                ))?;
            input_stride = input_stride.checked_mul(input_extent).ok_or(
                NeuralNetworkModuleError::ShapeOverflow("average-pool input stride"),
            )?;
        }
        plane
            .checked_mul(input_spatial_count)
            .and_then(|value| value.checked_add(input_spatial_index))
            .map(Some)
            .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                "average-pool input index",
            ))
    }
}

struct AdaptiveAveragePool2dGeometry {
    batch: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    output_shape: Vec<usize>,
}

impl AdaptiveAveragePool2dGeometry {
    fn new(
        input: &[f32],
        input_shape: &[usize],
        output_size: [usize; 2],
    ) -> Result<Self, NeuralNetworkModuleError> {
        if output_size.contains(&0) {
            return invalid(
                ADAPTIVE_AVG_POOL_2D_OPERATION_ID,
                "adaptive output dimensions must be nonzero",
            );
        }
        let (batch, channels, input_height, input_width, batched) = match input_shape {
            [channels, input_height, input_width] => {
                (1, *channels, *input_height, *input_width, false)
            }
            [batch, channels, input_height, input_width] => {
                (*batch, *channels, *input_height, *input_width, true)
            }
            _ => {
                return invalid(
                    ADAPTIVE_AVG_POOL_2D_OPERATION_ID,
                    "adaptive average pooling expects CHW or NCHW input",
                );
            }
        };
        if input_height == 0 || input_width == 0 {
            return invalid(
                ADAPTIVE_AVG_POOL_2D_OPERATION_ID,
                "adaptive average pooling requires nonempty spatial dimensions",
            );
        }
        require_length(
            input.len(),
            checked_product(input_shape, "adaptive-average-pool input shape")?,
            ADAPTIVE_AVG_POOL_2D_OPERATION_ID,
            "input",
        )?;
        let output_shape = if batched {
            vec![batch, channels, output_size[0], output_size[1]]
        } else {
            vec![channels, output_size[0], output_size[1]]
        };
        Ok(Self {
            batch,
            channels,
            input_height,
            input_width,
            output_height: output_size[0],
            output_width: output_size[1],
            output_shape,
        })
    }

    fn output_count(&self) -> Result<usize, NeuralNetworkModuleError> {
        checked_product(&self.output_shape, "adaptive-average-pool output shape")
    }

    fn for_each_connection(
        &self,
        context: &ExecutionContext<'_>,
        mut visit: impl FnMut(usize, usize, f32) -> Result<(), NeuralNetworkModuleError>,
    ) -> Result<(), NeuralNetworkModuleError> {
        for batch in 0..self.batch {
            for channel in 0..self.channels {
                for output_y in 0..self.output_height {
                    let start_y = output_y
                        .checked_mul(self.input_height)
                        .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                            "adaptive-average-pool height bin",
                        ))?
                        / self.output_height;
                    let end_y = (output_y + 1)
                        .checked_mul(self.input_height)
                        .and_then(|value| value.checked_add(self.output_height - 1))
                        .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                            "adaptive-average-pool height bin",
                        ))?
                        / self.output_height;
                    for output_x in 0..self.output_width {
                        let start_x = output_x
                            .checked_mul(self.input_width)
                            .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                                "adaptive-average-pool width bin",
                            ))?
                            / self.output_width;
                        let end_x = (output_x + 1)
                            .checked_mul(self.input_width)
                            .and_then(|value| value.checked_add(self.output_width - 1))
                            .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                                "adaptive-average-pool width bin",
                            ))?
                            / self.output_width;
                        let window = (end_y - start_y)
                            .checked_mul(end_x - start_x)
                            .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                                "adaptive-average-pool window",
                            ))?;
                        let scale = (window as f32).recip();
                        let plane = batch
                            .checked_mul(self.channels)
                            .and_then(|value| value.checked_add(channel))
                            .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                                "adaptive-average-pool output index",
                            ))?;
                        let output_index = plane
                            .checked_mul(self.output_height)
                            .and_then(|value| value.checked_add(output_y))
                            .and_then(|value| value.checked_mul(self.output_width))
                            .and_then(|value| value.checked_add(output_x))
                            .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                                "adaptive-average-pool output index",
                            ))?;
                        check_periodically(output_index, context)?;
                        for input_y in start_y..end_y {
                            for input_x in start_x..end_x {
                                let input_index = plane
                                    .checked_mul(self.input_height)
                                    .and_then(|value| value.checked_add(input_y))
                                    .and_then(|value| value.checked_mul(self.input_width))
                                    .and_then(|value| value.checked_add(input_x))
                                    .ok_or(NeuralNetworkModuleError::ShapeOverflow(
                                        "adaptive-average-pool input index",
                                    ))?;
                                visit(input_index, output_index, scale)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_prelu(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
) -> Result<(), NeuralNetworkModuleError> {
    require_length(
        input.len(),
        checked_product(input_shape, "PReLU input shape")?,
        PRELU_OPERATION_ID,
        "input",
    )?;
    if weight.is_empty()
        || (weight.len() != 1
            && (input_shape.len() < 2 || input_shape.get(1) != Some(&weight.len())))
    {
        return invalid(
            PRELU_OPERATION_ID,
            "weight must be scalar or match input channel dimension one",
        );
    }
    Ok(())
}

fn prelu_weight_index(
    linear: usize,
    shape: &[usize],
    weight_count: usize,
) -> Result<usize, NeuralNetworkModuleError> {
    if weight_count == 1 {
        return Ok(0);
    }
    let spatial = checked_product(
        shape.get(2..).ok_or_else(|| invalid_error(PRELU_OPERATION_ID, "missing channel axis"))?,
        "PReLU spatial shape",
    )?;
    if spatial == 0 {
        return invalid(PRELU_OPERATION_ID, "zero-sized channel plane");
    }
    Ok((linear / spatial) % weight_count)
}

fn validate_smooth_l1(
    input: &[f32],
    target: &[f32],
    beta: f32,
) -> Result<(), NeuralNetworkModuleError> {
    if input.len() != target.len() {
        return invalid(
            SMOOTH_L1_LOSS_OPERATION_ID,
            "input and target must have identical element counts",
        );
    }
    if !beta.is_finite() || beta < 0.0 {
        return invalid(
            SMOOTH_L1_LOSS_OPERATION_ID,
            "beta must be finite and nonnegative",
        );
    }
    Ok(())
}

fn reduce_loss(
    values: Vec<f32>,
    reduction: LossReduction,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    match reduction {
        LossReduction::None => Ok(values),
        LossReduction::Sum => Ok(vec![values.iter().copied().sum()]),
        LossReduction::Mean if values.is_empty() => Ok(vec![f32::NAN]),
        LossReduction::Mean => Ok(vec![values.iter().copied().sum::<f32>() / values.len() as f32]),
    }
}

fn require_cpu(
    operation: &'static str,
    device: DeviceId,
) -> Result<(), NeuralNetworkModuleError> {
    if device != DeviceId::CPU {
        return Err(NeuralNetworkModuleError::UnsupportedDevice { operation, device });
    }
    Ok(())
}

fn checked_product(
    shape: &[usize],
    subject: &'static str,
) -> Result<usize, NeuralNetworkModuleError> {
    shape.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or(NeuralNetworkModuleError::ShapeOverflow(subject))
    })
}

fn require_length(
    actual: usize,
    expected: usize,
    operation: &'static str,
    subject: &'static str,
) -> Result<(), NeuralNetworkModuleError> {
    if actual != expected {
        return invalid(
            operation,
            format!("{subject} has {actual} elements, expected {expected}"),
        );
    }
    Ok(())
}

fn reserved<T>(
    length: usize,
    subject: &'static str,
) -> Result<Vec<T>, NeuralNetworkModuleError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| NeuralNetworkModuleError::ShapeOverflow(subject))?;
    Ok(values)
}

fn zeroed(
    length: usize,
    subject: &'static str,
) -> Result<Vec<f32>, NeuralNetworkModuleError> {
    let mut values = reserved(length, subject)?;
    values.resize(length, 0.0);
    Ok(values)
}

fn check_periodically(
    index: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), NeuralNetworkModuleError> {
    if index.is_multiple_of(256) {
        context.cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, NeuralNetworkModuleError> {
    Err(invalid_error(operation, reason))
}

fn invalid_error(
    operation: &'static str,
    reason: impl Into<String>,
) -> NeuralNetworkModuleError {
    NeuralNetworkModuleError::Invalid {
        operation,
        reason: reason.into(),
    }
}
