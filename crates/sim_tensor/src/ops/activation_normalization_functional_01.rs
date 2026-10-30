use crate::{
    CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId, ExecutionContext, NumericClass,
    Scalar, Tensor, TensorBackend, TensorDescriptor, TensorError, TensorWrite, UnaryOperation,
    cpu_backend::apply_unary_scalar,
};
use sim_types::CancellationToken;
use thiserror::Error;

pub const BATCH_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-62E4C5B6AD0A";
pub const GELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-AED18ABCFD2B";
pub const GROUP_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-E5601CAF0B90";
pub const LAYER_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-B8FC03A4DCA1";
pub const LEAKY_RELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-8F7DF3AF61B9";
pub const NORMALIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-198CE0C4C23A";
pub const RELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-0538873A73B1";
pub const RMS_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-56C3DFEB2F05";
pub const SILU_OPERATION_ID: &str = "COMFY-TENSOR-OP-8D65C6573263";
pub const SOFTMAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-0F14CB65D5B7";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeluApproximation {
    None,
    Tanh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchNormTensorDirection {
    Normalize,
    Denormalize,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum FunctionalError {
    #[error(transparent)]
    Tensor(TensorError),
    #[error("functional tensor shape overflowed")]
    ShapeOverflow,
    #[error("functional tensor rank must be at least {minimum}, got {actual}")]
    Rank { minimum: usize, actual: usize },
    #[error("functional tensor expected {expected} values, got {actual}")]
    ValueCount { expected: usize, actual: usize },
    #[error("functional parameter {name} expected {expected} values, got {actual}")]
    ParameterValueCount {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("functional dimension {dimension} is invalid for rank {rank}")]
    InvalidDimension { dimension: isize, rank: usize },
    #[error("functional dimensions contain duplicate axis {axis}")]
    DuplicateDimension { axis: usize },
    #[error("functional normalization requires at least one dimension")]
    EmptyDimensions,
    #[error("functional normalized shape {normalized:?} does not match input shape {input:?}")]
    NormalizedShape {
        normalized: Vec<usize>,
        input: Vec<usize>,
    },
    #[error("functional group count {groups} must be nonzero and divide {channels} channels")]
    InvalidGroups { groups: usize, channels: usize },
    #[error("functional normalization group must contain more than one value")]
    InsufficientGroupValues,
    #[error("functional epsilon must be finite and greater than zero")]
    InvalidEpsilon,
    #[error("functional momentum must be finite and in the inclusive range zero to one")]
    InvalidMomentum,
    #[error("functional norm order must be finite and at least one")]
    InvalidNormOrder,
    #[error("functional negative slope must be finite")]
    InvalidNegativeSlope,
    #[error("functional ELU alpha must be finite")]
    InvalidEluAlpha,
    #[error(
        "functional batch normalization running mean and variance must be both present or both absent"
    )]
    UnpairedRunningStatistics,
    #[error("functional batch normalization evaluation requires running statistics")]
    MissingRunningStatistics,
    #[error("functional kernel has no certified adapter for device {device:?}")]
    UnsupportedDevice { device: DeviceId },
    #[error("functional allocation for {name} failed")]
    AllocationFailed { name: &'static str },
    #[error("functional operation was cancelled")]
    Cancelled,
}

impl From<sim_types::CancellationError> for FunctionalError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<TensorError> for FunctionalError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled,
            error => Self::Tensor(error),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineVjp {
    pub input: Vec<f32>,
    pub weight: Option<Vec<f32>>,
    pub bias: Option<Vec<f32>>,
}

fn relu_exact_native(
    input: &[f32],
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    elementwise_forward(input, device, cancellation, |value| {
        apply_unary_scalar(UnaryOperation::Relu, value)
    })
}

pub fn relu_with_context_exact_native_in_place(
    backend: &CpuBackend,
    input: &mut [f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<(), FunctionalError> {
    elementwise_in_place_with_context(backend, input, device, context, |value| {
        apply_unary_scalar(UnaryOperation::Relu, value)
    })
}

fn relu_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    elementwise_gradient(input, output_gradient, device, cancellation, |value| {
        f32::from(value > 0.0)
    })
}

fn leaky_relu_exact_native(
    input: &[f32],
    negative_slope: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_negative_slope(negative_slope)?;
    elementwise_forward(input, device, cancellation, |value| {
        leaky_relu_scalar(value, negative_slope)
    })
}

fn leaky_relu_scalar(value: f32, negative_slope: f32) -> f32 {
    if value >= 0.0 {
        value
    } else {
        value * negative_slope
    }
}

pub fn leaky_relu_with_context_exact_native_in_place(
    backend: &CpuBackend,
    input: &mut [f32],
    negative_slope: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<(), FunctionalError> {
    validate_negative_slope(negative_slope)?;
    elementwise_in_place_with_context(backend, input, device, context, |value| {
        if value >= 0.0 {
            value
        } else {
            value * negative_slope
        }
    })
}

fn leaky_relu_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    negative_slope: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_negative_slope(negative_slope)?;
    elementwise_gradient(input, output_gradient, device, cancellation, |value| {
        if value >= 0.0 { 1.0 } else { negative_slope }
    })
}

fn elu_exact_native(
    input: &[f32],
    alpha: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    if !alpha.is_finite() {
        return Err(FunctionalError::InvalidEluAlpha);
    }
    elementwise_forward(input, device, cancellation, |value| {
        if value > 0.0 {
            value
        } else {
            alpha * value.exp_m1()
        }
    })
}

fn elu_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    alpha: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    if !alpha.is_finite() {
        return Err(FunctionalError::InvalidEluAlpha);
    }
    elementwise_gradient(input, output_gradient, device, cancellation, |value| {
        if value > 0.0 {
            1.0
        } else {
            alpha * value.exp()
        }
    })
}

fn silu_exact_native(
    input: &[f32],
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    elementwise_forward(input, device, cancellation, silu_scalar)
}

pub fn silu_with_context_exact_native_in_place(
    backend: &CpuBackend,
    input: &mut [f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<(), FunctionalError> {
    elementwise_in_place_with_context(backend, input, device, context, silu_scalar)
}

fn silu_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    elementwise_gradient(input, output_gradient, device, cancellation, |value| {
        let sigmoid = apply_unary_scalar(UnaryOperation::Sigmoid, value);
        sigmoid * (1.0 + value * (1.0 - sigmoid))
    })
}

fn gelu_exact_native(
    input: &[f32],
    approximation: GeluApproximation,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    elementwise_forward(input, device, cancellation, |value| {
        gelu_scalar_exact_native(value, approximation)
    })
}

fn gelu_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    approximation: GeluApproximation,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    elementwise_gradient(input, output_gradient, device, cancellation, |value| {
        gelu_derivative(value, approximation)
    })
}

fn softmax_exact_native(
    input: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    cancellation.check()?;
    let mut output = zeroed(input.len(), "softmax output")?;
    softmax_into(input, &mut output, shape, dimension, cancellation)?;
    Ok(output)
}

fn softmax_into(
    input: &[f32],
    output: &mut [f32],
    shape: &[usize],
    dimension: isize,
    cancellation: &CancellationToken,
) -> Result<(), FunctionalError> {
    validate_gradient(input, output)?;
    let plan = AxisPlan::new(shape, dimension)?;
    for outer in 0..plan.outer {
        for inner in 0..plan.inner {
            cancellation.check()?;
            let mut maximum = f32::NEG_INFINITY;
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                let value = read(input, index)?;
                if value.is_nan() {
                    maximum = f32::NAN;
                    break;
                }
                maximum = maximum.max(value);
            }
            let mut sum = 0.0_f32;
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                let exponential = (read(input, index)? - maximum).exp();
                write(output, index, exponential)?;
                sum += exponential;
            }
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                let value = read(&output, index)? / sum;
                write(output, index, value)?;
            }
        }
    }
    Ok(())
}

fn log_softmax_exact_native(
    input: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    let plan = AxisPlan::new(shape, dimension)?;
    cancellation.check()?;
    let mut output = zeroed(input.len(), "log softmax output")?;
    for outer in 0..plan.outer {
        for inner in 0..plan.inner {
            cancellation.check()?;
            let mut maximum = f32::NEG_INFINITY;
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                let value = read(input, index)?;
                if value.is_nan() {
                    maximum = f32::NAN;
                    break;
                }
                maximum = maximum.max(value);
            }
            let mut exponential_sum = 0.0_f64;
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                exponential_sum += f64::from(read(input, index)? - maximum).exp();
            }
            let logarithm = exponential_sum.ln() as f32;
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                write(
                    &mut output,
                    index,
                    read(input, index)? - maximum - logarithm,
                )?;
            }
        }
    }
    Ok(output)
}

fn log_softmax_vjp_exact_native(
    output: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    log_softmax_linearized(output, output_gradient, shape, dimension, cancellation)
}

fn log_softmax_jvp_exact_native(
    output: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    log_softmax_jvp_linearized(output, input_tangent, shape, dimension, cancellation)
}

#[allow(clippy::too_many_arguments)]
pub fn normalize_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    norm_order: f32,
    dimensions: &[isize],
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    normalize_impl(
        input, shape, norm_order, dimensions, epsilon, device, backend, context,
    )
}

#[allow(clippy::too_many_arguments)]
fn normalize_impl(
    input: &[f32],
    shape: &[usize],
    norm_order: f32,
    dimensions: &[isize],
    epsilon: f32,
    device: DeviceId,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_epsilon(epsilon)?;
    if !norm_order.is_finite() || norm_order < 1.0 {
        return Err(FunctionalError::InvalidNormOrder);
    }
    let plan = ReductionPlan::new(shape, dimensions)?;
    context.cancellation.check()?;
    let mut sums = temporary_zeroed(backend, context, plan.groups)?;
    for (index, value) in input.iter().copied().enumerate() {
        check_periodically(index, context.cancellation)?;
        let group = plan.group(index)?;
        let destination = sums.get_mut(group).ok_or(FunctionalError::ShapeOverflow)?;
        *destination += value.abs().powf(norm_order);
    }
    for value in sums.iter_mut() {
        *value = value.powf(norm_order.recip()).max(epsilon);
    }
    let mut output = zeroed(input.len(), "normalization output")?;
    for (index, (value, destination)) in input.iter().zip(&mut output).enumerate() {
        check_periodically(index, context.cancellation)?;
        let denominator = read(&sums, plan.group(index)?)?;
        *destination = *value / denominator;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn normalize_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    norm_order: f32,
    dimensions: &[isize],
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    normalize_vjp_impl(
        input,
        output_gradient,
        shape,
        norm_order,
        dimensions,
        epsilon,
        device,
        backend,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn normalize_vjp_impl(
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    norm_order: f32,
    dimensions: &[isize],
    epsilon: f32,
    device: DeviceId,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_gradient(input, output_gradient)?;
    validate_epsilon(epsilon)?;
    if !norm_order.is_finite() || norm_order < 1.0 {
        return Err(FunctionalError::InvalidNormOrder);
    }
    let plan = ReductionPlan::new(shape, dimensions)?;
    let (norms, dot_products) = normalization_reductions(
        input,
        output_gradient,
        &plan,
        norm_order,
        backend,
        context,
        context.cancellation,
    )?;
    let mut result = zeroed(input.len(), "normalization VJP")?;
    for index in 0..input.len() {
        check_periodically(index, context.cancellation)?;
        let group = plan.group(index)?;
        let norm = read(&norms, group)?;
        let gradient = read(output_gradient, index)?;
        if norm <= epsilon {
            write(&mut result, index, gradient / epsilon)?;
            continue;
        }
        let value = read(input, index)?;
        let norm_gradient =
            value.signum() * value.abs().powf(norm_order - 1.0) / norm.powf(norm_order - 1.0);
        let result_value =
            gradient / norm - norm_gradient * read(&dot_products, group)? / (norm * norm);
        write(&mut result, index, result_value)?;
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub fn normalize_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    norm_order: f32,
    dimensions: &[isize],
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    normalize_jvp_impl(
        input,
        input_tangent,
        shape,
        norm_order,
        dimensions,
        epsilon,
        device,
        backend,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn normalize_jvp_impl(
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    norm_order: f32,
    dimensions: &[isize],
    epsilon: f32,
    device: DeviceId,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_gradient(input, input_tangent)?;
    validate_epsilon(epsilon)?;
    if !norm_order.is_finite() || norm_order < 1.0 {
        return Err(FunctionalError::InvalidNormOrder);
    }
    let plan = ReductionPlan::new(shape, dimensions)?;
    let mut norm_powers = temporary_zeroed(backend, context, plan.groups)?;
    let mut norm_tangents = temporary_zeroed(backend, context, plan.groups)?;
    for index in 0..input.len() {
        check_periodically(index, context.cancellation)?;
        let group = plan.group(index)?;
        let value = read(input, index)?;
        let tangent = read(input_tangent, index)?;
        *norm_powers
            .get_mut(group)
            .ok_or(FunctionalError::ShapeOverflow)? += value.abs().powf(norm_order);
        *norm_tangents
            .get_mut(group)
            .ok_or(FunctionalError::ShapeOverflow)? +=
            value.signum() * value.abs().powf(norm_order - 1.0) * tangent;
    }
    for group in 0..plan.groups {
        let norm = read(&norm_powers, group)?.powf(norm_order.recip());
        write(&mut norm_powers, group, norm)?;
        if norm > epsilon {
            let tangent = read(&norm_tangents, group)? / norm.powf(norm_order - 1.0);
            write(&mut norm_tangents, group, tangent)?;
        } else {
            write(&mut norm_tangents, group, 0.0)?;
        }
    }
    let mut output = zeroed(input.len(), "normalization JVP output")?;
    for index in 0..input.len() {
        check_periodically(index, context.cancellation)?;
        let group = plan.group(index)?;
        let norm = read(&norm_powers, group)?;
        let denominator = norm.max(epsilon);
        let value = read(input, index)?;
        let tangent = read(input_tangent, index)?;
        let denominator_tangent = read(&norm_tangents, group)?;
        write(
            &mut output,
            index,
            tangent / denominator - value * denominator_tangent / (denominator * denominator),
        )?;
    }
    Ok(output)
}

fn layer_norm_exact_native(
    input: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_epsilon(epsilon)?;
    let group_size = validate_trailing_shape(shape, normalized_shape)?;
    validate_optional_parameter(weight, group_size, "layer norm weight")?;
    validate_optional_parameter(bias, group_size, "layer norm bias")?;
    cancellation.check()?;
    let mut output = zeroed(input.len(), "layer norm output")?;
    for (group, (source, destination)) in input
        .chunks_exact(group_size)
        .zip(output.chunks_exact_mut(group_size))
        .enumerate()
    {
        check_periodically(group, cancellation)?;
        let (mean, inverse_standard_deviation) = mean_and_inverse(source, epsilon)?;
        for component in 0..group_size {
            let normalized = (read(source, component)? - mean) * inverse_standard_deviation;
            let scale = optional_read(weight, component, 1.0)?;
            let offset = optional_read(bias, component, 0.0)?;
            write(destination, component, normalized * scale + offset)?;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<AffineVjp, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_gradient(input, output_gradient)?;
    validate_epsilon(epsilon)?;
    let group_size = validate_trailing_shape(shape, normalized_shape)?;
    validate_optional_parameter(weight, group_size, "layer norm weight")?;
    validate_optional_parameter(bias, group_size, "layer norm bias")?;
    let mut input_gradient = zeroed(input.len(), "layer norm input VJP")?;
    let mut weight_gradient = weight
        .map(|_| zeroed(group_size, "layer norm weight VJP"))
        .transpose()?;
    let mut bias_gradient = bias
        .map(|_| zeroed(group_size, "layer norm bias VJP"))
        .transpose()?;
    for (group, ((source, gradient), destination)) in input
        .chunks_exact(group_size)
        .zip(output_gradient.chunks_exact(group_size))
        .zip(input_gradient.chunks_exact_mut(group_size))
        .enumerate()
    {
        check_periodically(group, cancellation)?;
        let (mean, inverse) = mean_and_inverse(source, epsilon)?;
        let mut scaled_sum = 0.0_f32;
        let mut scaled_normalized_sum = 0.0_f32;
        for component in 0..group_size {
            let normalized = (read(source, component)? - mean) * inverse;
            let upstream = read(gradient, component)?;
            let scaled = upstream * optional_read(weight, component, 1.0)?;
            scaled_sum += scaled;
            scaled_normalized_sum += scaled * normalized;
            add_optional(&mut weight_gradient, component, upstream * normalized)?;
            add_optional(&mut bias_gradient, component, upstream)?;
        }
        let count = group_size as f32;
        for component in 0..group_size {
            let normalized = (read(source, component)? - mean) * inverse;
            let scaled = read(gradient, component)? * optional_read(weight, component, 1.0)?;
            let value = inverse
                * (scaled - scaled_sum / count - normalized * scaled_normalized_sum / count);
            write(destination, component, value)?;
        }
    }
    Ok(AffineVjp {
        input: input_gradient,
        weight: weight_gradient,
        bias: bias_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_jvp_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_gradient(input, input_tangent)?;
    validate_epsilon(epsilon)?;
    let group_size = validate_trailing_shape(shape, normalized_shape)?;
    validate_optional_parameter(weight, group_size, "layer norm weight")?;
    validate_optional_parameter(weight_tangent, group_size, "layer norm weight tangent")?;
    validate_optional_parameter(bias_tangent, group_size, "layer norm bias tangent")?;
    let mut output = zeroed(input.len(), "layer norm JVP")?;
    for (group, ((source, tangent), destination)) in input
        .chunks_exact(group_size)
        .zip(input_tangent.chunks_exact(group_size))
        .zip(output.chunks_exact_mut(group_size))
        .enumerate()
    {
        check_periodically(group, cancellation)?;
        let (mean, inverse) = mean_and_inverse(source, epsilon)?;
        let tangent_mean =
            tangent.iter().map(|value| f64::from(*value)).sum::<f64>() / group_size as f64;
        let covariance = source
            .iter()
            .zip(tangent)
            .map(|(value, tangent)| (f64::from(*value) - f64::from(mean)) * f64::from(*tangent))
            .sum::<f64>()
            / group_size as f64;
        for component in 0..group_size {
            let centered = read(source, component)? - mean;
            let normalized = centered * inverse;
            let normalized_tangent = (read(tangent, component)? - tangent_mean as f32) * inverse
                - centered * inverse.powi(3) * covariance as f32;
            let value = normalized_tangent * optional_read(weight, component, 1.0)?
                + normalized * optional_read(weight_tangent, component, 0.0)?
                + optional_read(bias_tangent, component, 0.0)?;
            write(destination, component, value)?;
        }
    }
    Ok(output)
}

fn rms_norm_exact_native(
    input: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    epsilon: Option<f32>,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    let epsilon = epsilon.unwrap_or(f32::EPSILON);
    validate_epsilon(epsilon)?;
    let group_size = validate_trailing_shape(shape, normalized_shape)?;
    validate_optional_parameter(weight, group_size, "RMS norm weight")?;
    let mut output = zeroed(input.len(), "RMS norm output")?;
    for (group, (source, destination)) in input
        .chunks_exact(group_size)
        .zip(output.chunks_exact_mut(group_size))
        .enumerate()
    {
        check_periodically(group, cancellation)?;
        let inverse = root_mean_square_inverse(source, epsilon)?;
        for component in 0..group_size {
            write(
                destination,
                component,
                read(source, component)? * inverse * optional_read(weight, component, 1.0)?,
            )?;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn rms_norm_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    epsilon: Option<f32>,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<AffineVjp, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_gradient(input, output_gradient)?;
    let epsilon = epsilon.unwrap_or(f32::EPSILON);
    validate_epsilon(epsilon)?;
    let group_size = validate_trailing_shape(shape, normalized_shape)?;
    validate_optional_parameter(weight, group_size, "RMS norm weight")?;
    let mut input_gradient = zeroed(input.len(), "RMS norm input VJP")?;
    let mut weight_gradient = weight
        .map(|_| zeroed(group_size, "RMS norm weight VJP"))
        .transpose()?;
    for (group, ((source, gradient), destination)) in input
        .chunks_exact(group_size)
        .zip(output_gradient.chunks_exact(group_size))
        .zip(input_gradient.chunks_exact_mut(group_size))
        .enumerate()
    {
        check_periodically(group, cancellation)?;
        let inverse = root_mean_square_inverse(source, epsilon)?;
        let scaled_dot = source.iter().zip(gradient).enumerate().try_fold(
            0.0_f32,
            |sum, (component, (value, gradient))| {
                Ok::<_, FunctionalError>(
                    sum + value * gradient * optional_read(weight, component, 1.0)?,
                )
            },
        )?;
        let count = group_size as f32;
        for component in 0..group_size {
            let value = read(source, component)?;
            let upstream = read(gradient, component)?;
            let normalized = value * inverse;
            let input_value = upstream * optional_read(weight, component, 1.0)? * inverse
                - value * inverse.powi(3) * scaled_dot / count;
            write(destination, component, input_value)?;
            add_optional(&mut weight_gradient, component, upstream * normalized)?;
        }
    }
    Ok(AffineVjp {
        input: input_gradient,
        weight: weight_gradient,
        bias: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn rms_norm_jvp_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    epsilon: Option<f32>,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_gradient(input, input_tangent)?;
    let epsilon = epsilon.unwrap_or(f32::EPSILON);
    validate_epsilon(epsilon)?;
    let group_size = validate_trailing_shape(shape, normalized_shape)?;
    validate_optional_parameter(weight, group_size, "RMS norm weight")?;
    validate_optional_parameter(weight_tangent, group_size, "RMS norm weight tangent")?;
    let mut output = zeroed(input.len(), "RMS norm JVP")?;
    for (group, ((source, tangent), destination)) in input
        .chunks_exact(group_size)
        .zip(input_tangent.chunks_exact(group_size))
        .zip(output.chunks_exact_mut(group_size))
        .enumerate()
    {
        check_periodically(group, cancellation)?;
        let inverse = root_mean_square_inverse(source, epsilon)?;
        let dot = source
            .iter()
            .zip(tangent)
            .map(|(value, tangent)| value * tangent)
            .sum::<f32>()
            / group_size as f32;
        for component in 0..group_size {
            let value = read(source, component)?;
            let normalized = value * inverse;
            let normalized_tangent =
                read(tangent, component)? * inverse - value * inverse.powi(3) * dot;
            write(
                destination,
                component,
                normalized_tangent * optional_read(weight, component, 1.0)?
                    + normalized * optional_read(weight_tangent, component, 0.0)?,
            )?;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn group_norm_exact_native(
    input: &[f32],
    shape: &[usize],
    groups: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    let plan = ChannelNormPlan::new(input, shape, groups, epsilon, device)?;
    validate_optional_parameter(weight, plan.channels, "group norm weight")?;
    validate_optional_parameter(bias, plan.channels, "group norm bias")?;
    let mut output = zeroed(input.len(), "group norm output")?;
    for batch in 0..plan.batch {
        for group in 0..plan.groups {
            cancellation.check()?;
            let (mean, inverse) = plan.group_statistics(input, batch, group)?;
            plan.for_group(batch, group, |index, channel| {
                let normalized = (read(input, index)? - mean) * inverse;
                write(
                    &mut output,
                    index,
                    normalized * optional_read(weight, channel, 1.0)?
                        + optional_read(bias, channel, 0.0)?,
                )
            })?;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn group_norm_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    groups: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<AffineVjp, FunctionalError> {
    validate_gradient(input, output_gradient)?;
    let plan = ChannelNormPlan::new(input, shape, groups, epsilon, device)?;
    validate_optional_parameter(weight, plan.channels, "group norm weight")?;
    validate_optional_parameter(bias, plan.channels, "group norm bias")?;
    let mut input_gradient = zeroed(input.len(), "group norm input VJP")?;
    let mut weight_gradient = weight
        .map(|_| zeroed(plan.channels, "group norm weight VJP"))
        .transpose()?;
    let mut bias_gradient = bias
        .map(|_| zeroed(plan.channels, "group norm bias VJP"))
        .transpose()?;
    for batch in 0..plan.batch {
        for group in 0..plan.groups {
            cancellation.check()?;
            let (mean, inverse) = plan.group_statistics(input, batch, group)?;
            let mut scaled_sum = 0.0_f32;
            let mut scaled_normalized_sum = 0.0_f32;
            plan.for_group(batch, group, |index, channel| {
                let normalized = (read(input, index)? - mean) * inverse;
                let upstream = read(output_gradient, index)?;
                let scaled = upstream * optional_read(weight, channel, 1.0)?;
                scaled_sum += scaled;
                scaled_normalized_sum += scaled * normalized;
                add_optional(&mut weight_gradient, channel, upstream * normalized)?;
                add_optional(&mut bias_gradient, channel, upstream)
            })?;
            let count = plan.group_values as f32;
            plan.for_group(batch, group, |index, channel| {
                let normalized = (read(input, index)? - mean) * inverse;
                let scaled = read(output_gradient, index)? * optional_read(weight, channel, 1.0)?;
                write(
                    &mut input_gradient,
                    index,
                    inverse
                        * (scaled
                            - scaled_sum / count
                            - normalized * scaled_normalized_sum / count),
                )
            })?;
        }
    }
    Ok(AffineVjp {
        input: input_gradient,
        weight: weight_gradient,
        bias: bias_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
fn group_norm_jvp_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    groups: usize,
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_gradient(input, input_tangent)?;
    let plan = ChannelNormPlan::new(input, shape, groups, epsilon, device)?;
    validate_optional_parameter(weight, plan.channels, "group norm weight")?;
    validate_optional_parameter(weight_tangent, plan.channels, "group norm weight tangent")?;
    validate_optional_parameter(bias_tangent, plan.channels, "group norm bias tangent")?;
    let mut output = zeroed(input.len(), "group norm JVP")?;
    for batch in 0..plan.batch {
        for group in 0..plan.groups {
            cancellation.check()?;
            let (mean, inverse) = plan.group_statistics(input, batch, group)?;
            let mut tangent_mean = 0.0_f64;
            let mut covariance = 0.0_f64;
            plan.for_group(batch, group, |index, _| {
                tangent_mean += f64::from(read(input_tangent, index)?);
                covariance +=
                    f64::from(read(input, index)? - mean) * f64::from(read(input_tangent, index)?);
                Ok(())
            })?;
            tangent_mean /= plan.group_values as f64;
            covariance /= plan.group_values as f64;
            plan.for_group(batch, group, |index, channel| {
                let centered = read(input, index)? - mean;
                let normalized = centered * inverse;
                let normalized_tangent = (read(input_tangent, index)? - tangent_mean as f32)
                    * inverse
                    - centered * inverse.powi(3) * covariance as f32;
                write(
                    &mut output,
                    index,
                    normalized_tangent * optional_read(weight, channel, 1.0)?
                        + normalized * optional_read(weight_tangent, channel, 0.0)?
                        + optional_read(bias_tangent, channel, 0.0)?,
                )
            })?;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    running_mean: Option<&mut [f32]>,
    running_variance: Option<&mut [f32]>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    training: bool,
    momentum: f32,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    batch_norm_impl(
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
        backend,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn batch_norm_impl(
    input: &[f32],
    shape: &[usize],
    running_mean: Option<&mut [f32]>,
    running_variance: Option<&mut [f32]>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    training: bool,
    momentum: f32,
    epsilon: f32,
    device: DeviceId,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    let plan = BatchNormPlan::new(input, shape, training, momentum, epsilon, device)?;
    validate_optional_parameter(weight, plan.channels, "batch norm weight")?;
    validate_optional_parameter(bias, plan.channels, "batch norm bias")?;
    let (running_mean, running_variance) =
        validate_running_statistics(running_mean, running_variance, plan.channels, training)?;
    context.cancellation.check()?;
    let mut output = zeroed(input.len(), "batch norm output")?;
    let mut next_mean = running_mean
        .as_ref()
        .map(|values| temporary_copy(backend, context, values))
        .transpose()?;
    let mut next_variance = running_variance
        .as_ref()
        .map(|values| temporary_copy(backend, context, values))
        .transpose()?;
    for channel in 0..plan.channels {
        context.cancellation.check()?;
        let (mean, biased_variance) = if training {
            plan.channel_statistics(input, channel)?
        } else {
            (
                optional_read(running_mean.as_deref(), channel, 0.0)?,
                optional_read(running_variance.as_deref(), channel, 0.0)?,
            )
        };
        let inverse = (biased_variance + epsilon).sqrt().recip();
        plan.for_channel(channel, |index| {
            let normalized = (read(input, index)? - mean) * inverse;
            write(
                &mut output,
                index,
                normalized * optional_read(weight, channel, 1.0)?
                    + optional_read(bias, channel, 0.0)?,
            )
        })?;
        if training {
            if let Some(values) = &mut next_mean {
                let current = read(values, channel)?;
                write(
                    values,
                    channel,
                    (1.0 - momentum) * current + momentum * mean,
                )?;
            }
            if let Some(values) = &mut next_variance {
                let current = read(values, channel)?;
                let unbiased = biased_variance * plan.values_per_channel as f32
                    / (plan.values_per_channel - 1) as f32;
                write(
                    values,
                    channel,
                    (1.0 - momentum) * current + momentum * unbiased,
                )?;
            }
        }
    }
    context.cancellation.check()?;
    if let (Some(destination), Some(values)) = (running_mean, next_mean) {
        destination.copy_from_slice(&values);
    }
    if let (Some(destination), Some(values)) = (running_variance, next_variance) {
        destination.copy_from_slice(&values);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn batch_norm_vjp_exact_native(
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    running_mean: Option<&[f32]>,
    running_variance: Option<&[f32]>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    training: bool,
    epsilon: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<AffineVjp, FunctionalError> {
    validate_gradient(input, output_gradient)?;
    let plan = BatchNormPlan::new(input, shape, training, 0.0, epsilon, device)?;
    validate_optional_parameter(weight, plan.channels, "batch norm weight")?;
    validate_optional_parameter(bias, plan.channels, "batch norm bias")?;
    validate_running_statistics_ref(running_mean, running_variance, plan.channels, training)?;
    let mut input_gradient = zeroed(input.len(), "batch norm input VJP")?;
    let mut weight_gradient = weight
        .map(|_| zeroed(plan.channels, "batch norm weight VJP"))
        .transpose()?;
    let mut bias_gradient = bias
        .map(|_| zeroed(plan.channels, "batch norm bias VJP"))
        .transpose()?;
    for channel in 0..plan.channels {
        cancellation.check()?;
        let (mean, variance) = if training {
            plan.channel_statistics(input, channel)?
        } else {
            (
                optional_read(running_mean, channel, 0.0)?,
                optional_read(running_variance, channel, 0.0)?,
            )
        };
        let inverse = (variance + epsilon).sqrt().recip();
        let mut scaled_sum = 0.0_f32;
        let mut scaled_normalized_sum = 0.0_f32;
        plan.for_channel(channel, |index| {
            let upstream = read(output_gradient, index)?;
            let normalized = (read(input, index)? - mean) * inverse;
            let scaled = upstream * optional_read(weight, channel, 1.0)?;
            scaled_sum += scaled;
            scaled_normalized_sum += scaled * normalized;
            add_optional(&mut weight_gradient, channel, upstream * normalized)?;
            add_optional(&mut bias_gradient, channel, upstream)
        })?;
        let count = plan.values_per_channel as f32;
        plan.for_channel(channel, |index| {
            let scaled = read(output_gradient, index)? * optional_read(weight, channel, 1.0)?;
            let value = if training {
                let normalized = (read(input, index)? - mean) * inverse;
                inverse * (scaled - scaled_sum / count - normalized * scaled_normalized_sum / count)
            } else {
                scaled * inverse
            };
            write(&mut input_gradient, index, value)
        })?;
    }
    Ok(AffineVjp {
        input: input_gradient,
        weight: weight_gradient,
        bias: bias_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
fn batch_norm_jvp_exact_native(
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    running_mean: Option<&[f32]>,
    running_variance: Option<&[f32]>,
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    training: bool,
    epsilon: f32,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_gradient(input, input_tangent)?;
    let plan = BatchNormPlan::new(input, shape, training, 0.0, epsilon, device)?;
    validate_optional_parameter(weight, plan.channels, "batch norm weight")?;
    validate_optional_parameter(weight_tangent, plan.channels, "batch norm weight tangent")?;
    validate_optional_parameter(bias_tangent, plan.channels, "batch norm bias tangent")?;
    validate_running_statistics_ref(running_mean, running_variance, plan.channels, training)?;
    let mut output = zeroed(input.len(), "batch norm JVP")?;
    for channel in 0..plan.channels {
        cancellation.check()?;
        let (mean, variance) = if training {
            plan.channel_statistics(input, channel)?
        } else {
            (
                optional_read(running_mean, channel, 0.0)?,
                optional_read(running_variance, channel, 0.0)?,
            )
        };
        let inverse = (variance + epsilon).sqrt().recip();
        let mut tangent_mean = 0.0_f64;
        let mut covariance = 0.0_f64;
        if training {
            plan.for_channel(channel, |index| {
                tangent_mean += f64::from(read(input_tangent, index)?);
                covariance +=
                    f64::from(read(input, index)? - mean) * f64::from(read(input_tangent, index)?);
                Ok(())
            })?;
            tangent_mean /= plan.values_per_channel as f64;
            covariance /= plan.values_per_channel as f64;
        }
        plan.for_channel(channel, |index| {
            let centered = read(input, index)? - mean;
            let normalized = centered * inverse;
            let normalized_tangent = if training {
                (read(input_tangent, index)? - tangent_mean as f32) * inverse
                    - centered * inverse.powi(3) * covariance as f32
            } else {
                read(input_tangent, index)? * inverse
            };
            write(
                &mut output,
                index,
                normalized_tangent * optional_read(weight, channel, 1.0)?
                    + normalized * optional_read(weight_tangent, channel, 0.0)?
                    + optional_read(bias_tangent, channel, 0.0)?,
            )
        })?;
    }
    Ok(output)
}

pub fn relu_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    relu_exact_native(input, device, context.cancellation)
}

pub fn relu_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    relu_vjp_exact_native(input, output_gradient, device, context.cancellation)
}

pub fn relu_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    relu_vjp_with_context_exact_native(backend, input, input_tangent, device, context)
}

pub fn leaky_relu_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    negative_slope: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    leaky_relu_exact_native(input, negative_slope, device, context.cancellation)
}

pub fn leaky_relu_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    negative_slope: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    validate_negative_slope(negative_slope)?;
    validate_tensor_input(backend, input, LEAKY_RELU_OPERATION_ID, context)?;
    let mut output = allocate_tensor_output(backend, input, context)?;
    {
        let mut write = output.write()?;
        let element_count = input.descriptor().element_count()?;
        for linear in 0..element_count {
            context.check()?;
            let value = read_tensor_real_linear(input, linear)? as f32;
            let activated = leaky_relu_scalar(value, negative_slope);
            write_tensor_real_linear(
                &mut write,
                input.descriptor().dtype(),
                input.descriptor().device(),
                linear,
                f64::from(activated),
                LEAKY_RELU_OPERATION_ID,
            )?;
        }
    }
    finish_tensor_output(backend, output, context)
}

#[allow(clippy::too_many_arguments)]
pub fn leaky_relu_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    negative_slope: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    leaky_relu_vjp_exact_native(
        input,
        output_gradient,
        negative_slope,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn leaky_relu_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    negative_slope: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    leaky_relu_vjp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        negative_slope,
        device,
        context,
    )
}

pub fn elu_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    alpha: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.cancellation.check()?;
    elu_exact_native(input, alpha, device, context.cancellation)
}

#[allow(clippy::too_many_arguments)]
pub fn elu_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    alpha: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.cancellation.check()?;
    elu_vjp_exact_native(input, output_gradient, alpha, device, context.cancellation)
}

#[allow(clippy::too_many_arguments)]
pub fn elu_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    alpha: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    elu_vjp_with_context_exact_native(backend, input, input_tangent, alpha, device, context)
}

pub fn silu_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    silu_exact_native(input, device, context.cancellation)
}

pub fn silu_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    validate_tensor_input(backend, input, SILU_OPERATION_ID, context)?;
    let mut output = allocate_tensor_output(backend, input, context)?;
    {
        let mut write = output.write()?;
        let element_count = input.descriptor().element_count()?;
        for linear in 0..element_count {
            context.check()?;
            let value = read_tensor_real_linear(input, linear)? as f32;
            let activated = f64::from(silu_scalar(value));
            write_tensor_real_linear(
                &mut write,
                input.descriptor().dtype(),
                input.descriptor().device(),
                linear,
                activated,
                SILU_OPERATION_ID,
            )?;
        }
    }
    finish_tensor_output(backend, output, context)
}

pub fn silu_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    silu_vjp_exact_native(input, output_gradient, device, context.cancellation)
}

pub fn silu_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    silu_vjp_with_context_exact_native(backend, input, input_tangent, device, context)
}

pub fn gelu_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    approximation: GeluApproximation,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    gelu_exact_native(input, approximation, device, context.cancellation)
}

#[allow(clippy::too_many_arguments)]
pub fn gelu_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    approximation: GeluApproximation,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    gelu_vjp_exact_native(
        input,
        output_gradient,
        approximation,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gelu_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    approximation: GeluApproximation,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    gelu_vjp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        approximation,
        device,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn softmax_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    softmax_exact_native(input, shape, dimension, device, context.cancellation)
}

pub fn softmax_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    dimension: isize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    validate_tensor_input(backend, input, SOFTMAX_OPERATION_ID, context)?;
    let shape = input.descriptor().shape();
    let axis = normalize_tensor_axis(dimension, shape.len())?;
    let width = shape[axis];
    if width == 0 {
        return Err(FunctionalError::EmptyDimensions);
    }
    let inner = checked_u64_product(&shape[axis + 1..], "softmax inner dimensions")?;
    let outer = checked_u64_product(&shape[..axis], "softmax outer dimensions")?;
    let mut output = allocate_tensor_output(backend, input, context)?;
    {
        let mut write = output.write()?;
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                context.check()?;
                let base = outer_index
                    .checked_mul(width)
                    .and_then(|value| value.checked_mul(inner))
                    .and_then(|value| value.checked_add(inner_index))
                    .ok_or(FunctionalError::ShapeOverflow)?;
                let mut maximum = f64::NEG_INFINITY;
                for column in 0..width {
                    let index = base
                        .checked_add(column.checked_mul(inner).ok_or(FunctionalError::ShapeOverflow)?)
                        .ok_or(FunctionalError::ShapeOverflow)?;
                    maximum = maximum.max(read_tensor_real_linear(input, index)?);
                }
                let mut denominator = 0.0_f64;
                for column in 0..width {
                    let index = base
                        .checked_add(column.checked_mul(inner).ok_or(FunctionalError::ShapeOverflow)?)
                        .ok_or(FunctionalError::ShapeOverflow)?;
                    denominator += (read_tensor_real_linear(input, index)? - maximum).exp();
                }
                if !denominator.is_finite() || denominator == 0.0 {
                    return Err(FunctionalError::Tensor(TensorError::InvalidNumeric {
                        reason: "softmax denominator is not finite and positive".to_owned(),
                    }));
                }
                for column in 0..width {
                    let index = base
                        .checked_add(column.checked_mul(inner).ok_or(FunctionalError::ShapeOverflow)?)
                        .ok_or(FunctionalError::ShapeOverflow)?;
                    let probability =
                        (read_tensor_real_linear(input, index)? - maximum).exp() / denominator;
                    write_tensor_real_linear(
                        &mut write,
                        input.descriptor().dtype(),
                        input.descriptor().device(),
                        index,
                        probability,
                        SOFTMAX_OPERATION_ID,
                    )?;
                }
            }
        }
    }
    finish_tensor_output(backend, output, context)
}

#[allow(clippy::too_many_arguments)]
pub fn softmax_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    validate_shape(input, shape)?;
    validate_gradient(input, output_gradient)?;
    context.check()?;
    let mut output = temporary_zeroed(backend, context, input.len())?;
    softmax_into(input, &mut output, shape, dimension, context.cancellation)?;
    softmax_linearized(
        &output,
        output_gradient,
        shape,
        dimension,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn softmax_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    softmax_vjp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        shape,
        dimension,
        device,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn log_softmax_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    log_softmax_exact_native(input, shape, dimension, device, context.cancellation)
}

#[allow(clippy::too_many_arguments)]
pub fn log_softmax_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    output: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    log_softmax_vjp_exact_native(
        output,
        output_gradient,
        shape,
        dimension,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn log_softmax_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    output: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    dimension: isize,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    log_softmax_jvp_exact_native(
        output,
        input_tangent,
        shape,
        dimension,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    layer_norm_exact_native(
        input,
        shape,
        normalized_shape,
        weight,
        bias,
        epsilon,
        device,
        context.cancellation,
    )
}

pub fn channel_layer_norm_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    epsilon: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    validate_tensor_input(backend, input, LAYER_NORM_OPERATION_ID, context)?;
    validate_epsilon_f64(epsilon)?;
    let shape = input.descriptor().shape();
    if shape.len() < 2 || shape[1] == 0 {
        return Err(FunctionalError::Rank {
            minimum: 2,
            actual: shape.len(),
        });
    }
    validate_channel_parameters(backend, input, weight, bias, shape[1], context)?;
    let spatial = checked_u64_product(&shape[2..], "channel layer norm spatial dimensions")?;
    let mut output = allocate_tensor_output(backend, input, context)?;
    {
        let mut write = output.write()?;
        for batch in 0..shape[0] {
            for spatial_index in 0..spatial {
                context.check()?;
                let mut sum = 0.0_f64;
                let mut square_sum = 0.0_f64;
                for channel in 0..shape[1] {
                    let index = channel_spatial_index(shape[1], spatial, batch, channel, spatial_index)?;
                    let value = read_tensor_real_linear(input, index)?;
                    sum += value;
                    square_sum = value.mul_add(value, square_sum);
                }
                let count = shape[1] as f64;
                let mean = sum / count;
                let reciprocal_standard_deviation =
                    (square_sum / count - mean * mean).max(0.0).mul_add(1.0, epsilon).sqrt().recip();
                for channel in 0..shape[1] {
                    let index = channel_spatial_index(shape[1], spatial, batch, channel, spatial_index)?;
                    let scale = weight
                        .map(|tensor| read_tensor_real_linear(tensor, channel))
                        .transpose()?
                        .unwrap_or(1.0);
                    let shift = bias
                        .map(|tensor| read_tensor_real_linear(tensor, channel))
                        .transpose()?
                        .unwrap_or(0.0);
                    let normalized = (read_tensor_real_linear(input, index)? - mean)
                        .mul_add(reciprocal_standard_deviation * scale, shift);
                    write_tensor_real_linear(
                        &mut write,
                        input.descriptor().dtype(),
                        input.descriptor().device(),
                        index,
                        normalized,
                        LAYER_NORM_OPERATION_ID,
                    )?;
                }
            }
        }
    }
    finish_tensor_output(backend, output, context)
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AffineVjp, FunctionalError> {
    context.check()?;
    layer_norm_vjp_exact_native(
        input,
        output_gradient,
        shape,
        normalized_shape,
        weight,
        bias,
        epsilon,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_jvp_with_context_exact_native(
    _backend: &CpuBackend,
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
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    layer_norm_jvp_exact_native(
        input,
        input_tangent,
        shape,
        normalized_shape,
        weight,
        weight_tangent,
        bias_tangent,
        epsilon,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    epsilon: Option<f32>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    rms_norm_exact_native(
        input,
        shape,
        normalized_shape,
        weight,
        epsilon,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    epsilon: Option<f32>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AffineVjp, FunctionalError> {
    context.check()?;
    rms_norm_vjp_exact_native(
        input,
        output_gradient,
        shape,
        normalized_shape,
        weight,
        epsilon,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    normalized_shape: &[usize],
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    epsilon: Option<f32>,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    rms_norm_jvp_exact_native(
        input,
        input_tangent,
        shape,
        normalized_shape,
        weight,
        weight_tangent,
        epsilon,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    groups: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    group_norm_exact_native(
        input,
        shape,
        groups,
        weight,
        bias,
        epsilon,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    groups: u64,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    epsilon: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    validate_tensor_input(backend, input, GROUP_NORM_OPERATION_ID, context)?;
    validate_epsilon_f64(epsilon)?;
    let shape = input.descriptor().shape();
    if shape.len() < 2 || shape[1] == 0 || groups == 0 || !shape[1].is_multiple_of(groups) {
        return Err(FunctionalError::InvalidGroups {
            groups: usize::try_from(groups).unwrap_or(usize::MAX),
            channels: usize::try_from(shape.get(1).copied().unwrap_or(0)).unwrap_or(usize::MAX),
        });
    }
    validate_channel_parameters(backend, input, weight, bias, shape[1], context)?;
    let spatial = checked_u64_product(&shape[2..], "group norm spatial dimensions")?;
    let channels_per_group = shape[1] / groups;
    let values_per_group = channels_per_group
        .checked_mul(spatial)
        .ok_or(FunctionalError::ShapeOverflow)?;
    if values_per_group <= 1 {
        return Err(FunctionalError::InsufficientGroupValues);
    }
    let mut output = allocate_tensor_output(backend, input, context)?;
    {
        let mut write = output.write()?;
        for batch in 0..shape[0] {
            for group in 0..groups {
                context.check()?;
                let first_channel = group
                    .checked_mul(channels_per_group)
                    .ok_or(FunctionalError::ShapeOverflow)?;
                let mut sum = 0.0_f64;
                let mut square_sum = 0.0_f64;
                for channel_offset in 0..channels_per_group {
                    let channel = first_channel + channel_offset;
                    for spatial_index in 0..spatial {
                        let index = channel_spatial_index(
                            shape[1], spatial, batch, channel, spatial_index,
                        )?;
                        let value = read_tensor_real_linear(input, index)?;
                        sum += value;
                        square_sum = value.mul_add(value, square_sum);
                    }
                }
                let count = values_per_group as f64;
                let mean = sum / count;
                let reciprocal_standard_deviation =
                    (square_sum / count - mean * mean).max(0.0).mul_add(1.0, epsilon).sqrt().recip();
                for channel_offset in 0..channels_per_group {
                    let channel = first_channel + channel_offset;
                    let scale = weight
                        .map(|tensor| read_tensor_real_linear(tensor, channel))
                        .transpose()?
                        .unwrap_or(1.0);
                    let shift = bias
                        .map(|tensor| read_tensor_real_linear(tensor, channel))
                        .transpose()?
                        .unwrap_or(0.0);
                    for spatial_index in 0..spatial {
                        let index = channel_spatial_index(
                            shape[1], spatial, batch, channel, spatial_index,
                        )?;
                        let normalized = (read_tensor_real_linear(input, index)? - mean)
                            .mul_add(reciprocal_standard_deviation * scale, shift);
                        write_tensor_real_linear(
                            &mut write,
                            input.descriptor().dtype(),
                            input.descriptor().device(),
                            index,
                            normalized,
                            GROUP_NORM_OPERATION_ID,
                        )?;
                    }
                }
            }
        }
    }
    finish_tensor_output(backend, output, context)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    running_mean: &Tensor,
    running_variance: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    epsilon: f64,
    direction: BatchNormTensorDirection,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    validate_tensor_input(backend, input, BATCH_NORM_OPERATION_ID, context)?;
    validate_epsilon_f64(epsilon)?;
    let shape = input.descriptor().shape();
    if shape.len() < 2 || shape[1] == 0 {
        return Err(FunctionalError::Rank {
            minimum: 2,
            actual: shape.len(),
        });
    }
    validate_channel_parameters(
        backend,
        input,
        Some(running_mean),
        Some(running_variance),
        shape[1],
        context,
    )?;
    validate_channel_parameters(backend, input, weight, bias, shape[1], context)?;
    let spatial = checked_u64_product(&shape[2..], "batch norm spatial dimensions")?;
    let mut output = allocate_tensor_output(backend, input, context)?;
    {
        let mut write = output.write()?;
        for batch in 0..shape[0] {
            for channel in 0..shape[1] {
                context.check()?;
                let mean = read_tensor_real_linear(running_mean, channel)?;
                let standard_deviation =
                    (read_tensor_real_linear(running_variance, channel)? + epsilon).sqrt();
                if !standard_deviation.is_finite() || standard_deviation == 0.0 {
                    return Err(FunctionalError::Tensor(TensorError::InvalidNumeric {
                        reason: "batch norm standard deviation is not finite and positive".to_owned(),
                    }));
                }
                let scale = weight
                    .map(|tensor| read_tensor_real_linear(tensor, channel))
                    .transpose()?
                    .unwrap_or(1.0);
                let shift = bias
                    .map(|tensor| read_tensor_real_linear(tensor, channel))
                    .transpose()?
                    .unwrap_or(0.0);
                if direction == BatchNormTensorDirection::Denormalize && scale == 0.0 {
                    return Err(FunctionalError::Tensor(TensorError::InvalidNumeric {
                        reason: "batch norm denormalization scale must be nonzero".to_owned(),
                    }));
                }
                for spatial_index in 0..spatial {
                    let index = channel_spatial_index(
                        shape[1], spatial, batch, channel, spatial_index,
                    )?;
                    let value = read_tensor_real_linear(input, index)?;
                    let transformed = match direction {
                        BatchNormTensorDirection::Normalize => {
                            ((value - mean) / standard_deviation).mul_add(scale, shift)
                        }
                        BatchNormTensorDirection::Denormalize => {
                            ((value - shift) / scale).mul_add(standard_deviation, mean)
                        }
                    };
                    write_tensor_real_linear(
                        &mut write,
                        input.descriptor().dtype(),
                        input.descriptor().device(),
                        index,
                        transformed,
                        BATCH_NORM_OPERATION_ID,
                    )?;
                }
            }
        }
    }
    finish_tensor_output(backend, output, context)
}

pub fn channel_standardize_tensor_with_context_exact_native(
    backend: &dyn TensorBackend,
    input: &Tensor,
    mean: &Tensor,
    standard_deviation: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    validate_tensor_input(backend, input, NORMALIZE_OPERATION_ID, context)?;
    let shape = input.descriptor().shape();
    if shape.len() < 2 || shape[1] == 0 {
        return Err(FunctionalError::Rank {
            minimum: 2,
            actual: shape.len(),
        });
    }
    validate_channel_parameters(
        backend,
        input,
        Some(mean),
        Some(standard_deviation),
        shape[1],
        context,
    )?;
    let spatial = checked_u64_product(&shape[2..], "channel standardization spatial dimensions")?;
    let mut output = allocate_tensor_output(backend, input, context)?;
    {
        let mut write = output.write()?;
        for batch in 0..shape[0] {
            for channel in 0..shape[1] {
                context.check()?;
                let channel_mean = read_tensor_real_linear(mean, channel)?;
                let channel_standard_deviation =
                    read_tensor_real_linear(standard_deviation, channel)?;
                if !channel_standard_deviation.is_finite() || channel_standard_deviation == 0.0 {
                    return Err(FunctionalError::Tensor(TensorError::InvalidNumeric {
                        reason: format!(
                            "channel standard deviation {channel} must be finite and nonzero"
                        ),
                    }));
                }
                for spatial_index in 0..spatial {
                    let index = channel_spatial_index(
                        shape[1], spatial, batch, channel, spatial_index,
                    )?;
                    let standardized = (read_tensor_real_linear(input, index)? - channel_mean)
                        / channel_standard_deviation;
                    write_tensor_real_linear(
                        &mut write,
                        input.descriptor().dtype(),
                        input.descriptor().device(),
                        index,
                        standardized,
                        NORMALIZE_OPERATION_ID,
                    )?;
                }
            }
        }
    }
    finish_tensor_output(backend, output, context)
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    groups: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AffineVjp, FunctionalError> {
    context.check()?;
    group_norm_vjp_exact_native(
        input,
        output_gradient,
        shape,
        groups,
        weight,
        bias,
        epsilon,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_jvp_with_context_exact_native(
    _backend: &CpuBackend,
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
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    group_norm_jvp_exact_native(
        input,
        input_tangent,
        shape,
        groups,
        weight,
        weight_tangent,
        bias_tangent,
        epsilon,
        device,
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_vjp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    output_gradient: &[f32],
    shape: &[usize],
    running_mean: Option<&[f32]>,
    running_variance: Option<&[f32]>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    training: bool,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<AffineVjp, FunctionalError> {
    context.check()?;
    batch_norm_vjp_exact_native(
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
        context.cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_jvp_with_context_exact_native(
    _backend: &CpuBackend,
    input: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    running_mean: Option<&[f32]>,
    running_variance: Option<&[f32]>,
    weight: Option<&[f32]>,
    weight_tangent: Option<&[f32]>,
    bias_tangent: Option<&[f32]>,
    training: bool,
    epsilon: f32,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, FunctionalError> {
    context.check()?;
    batch_norm_jvp_exact_native(
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
        context.cancellation,
    )
}

struct ChannelNormPlan {
    batch: usize,
    channels: usize,
    spatial: usize,
    groups: usize,
    channels_per_group: usize,
    group_values: usize,
    epsilon: f32,
}

impl ChannelNormPlan {
    fn new(
        input: &[f32],
        shape: &[usize],
        groups: usize,
        epsilon: f32,
        device: DeviceId,
    ) -> Result<Self, FunctionalError> {
        validate_device(device)?;
        validate_shape(input, shape)?;
        validate_epsilon(epsilon)?;
        if shape.len() < 2 {
            return Err(FunctionalError::Rank {
                minimum: 2,
                actual: shape.len(),
            });
        }
        let batch = shape[0];
        let channels = shape[1];
        if groups == 0 || channels == 0 || !channels.is_multiple_of(groups) {
            return Err(FunctionalError::InvalidGroups { groups, channels });
        }
        let spatial = checked_product(&shape[2..])?;
        let channels_per_group = channels / groups;
        let group_values = channels_per_group
            .checked_mul(spatial)
            .ok_or(FunctionalError::ShapeOverflow)?;
        if batch == 0 || group_values <= 1 {
            return Err(FunctionalError::InsufficientGroupValues);
        }
        Ok(Self {
            batch,
            channels,
            spatial,
            groups,
            channels_per_group,
            group_values,
            epsilon,
        })
    }

    fn index(
        &self,
        batch: usize,
        channel: usize,
        spatial: usize,
    ) -> Result<usize, FunctionalError> {
        batch
            .checked_mul(self.channels)
            .and_then(|value| value.checked_add(channel))
            .and_then(|value| value.checked_mul(self.spatial))
            .and_then(|value| value.checked_add(spatial))
            .ok_or(FunctionalError::ShapeOverflow)
    }

    fn for_group(
        &self,
        batch: usize,
        group: usize,
        mut function: impl FnMut(usize, usize) -> Result<(), FunctionalError>,
    ) -> Result<(), FunctionalError> {
        let channel_start = group
            .checked_mul(self.channels_per_group)
            .ok_or(FunctionalError::ShapeOverflow)?;
        for channel in channel_start..channel_start + self.channels_per_group {
            for spatial in 0..self.spatial {
                function(self.index(batch, channel, spatial)?, channel)?;
            }
        }
        Ok(())
    }

    fn group_statistics(
        &self,
        input: &[f32],
        batch: usize,
        group: usize,
    ) -> Result<(f32, f32), FunctionalError> {
        let mut sum = 0.0_f64;
        self.for_group(batch, group, |index, _| {
            sum += f64::from(read(input, index)?);
            Ok(())
        })?;
        let mean = (sum / self.group_values as f64) as f32;
        let mut square_sum = 0.0_f64;
        self.for_group(batch, group, |index, _| {
            let difference = f64::from(read(input, index)?) - f64::from(mean);
            square_sum += difference * difference;
            Ok(())
        })?;
        let variance = square_sum / self.group_values as f64;
        Ok((
            mean,
            (variance + f64::from(self.epsilon)).sqrt().recip() as f32,
        ))
    }
}

struct BatchNormPlan {
    batch: usize,
    channels: usize,
    spatial: usize,
    values_per_channel: usize,
}

impl BatchNormPlan {
    fn new(
        input: &[f32],
        shape: &[usize],
        training: bool,
        momentum: f32,
        epsilon: f32,
        device: DeviceId,
    ) -> Result<Self, FunctionalError> {
        validate_device(device)?;
        validate_shape(input, shape)?;
        validate_epsilon(epsilon)?;
        if !momentum.is_finite() || !(0.0..=1.0).contains(&momentum) {
            return Err(FunctionalError::InvalidMomentum);
        }
        if shape.len() < 2 {
            return Err(FunctionalError::Rank {
                minimum: 2,
                actual: shape.len(),
            });
        }
        let batch = shape[0];
        let channels = shape[1];
        let spatial = checked_product(&shape[2..])?;
        let values_per_channel = batch
            .checked_mul(spatial)
            .ok_or(FunctionalError::ShapeOverflow)?;
        if channels == 0 || (training && values_per_channel <= 1) {
            return Err(FunctionalError::InsufficientGroupValues);
        }
        Ok(Self {
            batch,
            channels,
            spatial,
            values_per_channel,
        })
    }

    fn index(
        &self,
        batch: usize,
        channel: usize,
        spatial: usize,
    ) -> Result<usize, FunctionalError> {
        batch
            .checked_mul(self.channels)
            .and_then(|value| value.checked_add(channel))
            .and_then(|value| value.checked_mul(self.spatial))
            .and_then(|value| value.checked_add(spatial))
            .ok_or(FunctionalError::ShapeOverflow)
    }

    fn for_channel(
        &self,
        channel: usize,
        mut function: impl FnMut(usize) -> Result<(), FunctionalError>,
    ) -> Result<(), FunctionalError> {
        for batch in 0..self.batch {
            for spatial in 0..self.spatial {
                function(self.index(batch, channel, spatial)?)?;
            }
        }
        Ok(())
    }

    fn channel_statistics(
        &self,
        input: &[f32],
        channel: usize,
    ) -> Result<(f32, f32), FunctionalError> {
        let mut sum = 0.0_f64;
        self.for_channel(channel, |index| {
            sum += f64::from(read(input, index)?);
            Ok(())
        })?;
        let mean = (sum / self.values_per_channel as f64) as f32;
        let mut square_sum = 0.0_f64;
        self.for_channel(channel, |index| {
            let difference = f64::from(read(input, index)?) - f64::from(mean);
            square_sum += difference * difference;
            Ok(())
        })?;
        Ok((mean, (square_sum / self.values_per_channel as f64) as f32))
    }
}

fn validate_running_statistics<'a>(
    running_mean: Option<&'a mut [f32]>,
    running_variance: Option<&'a mut [f32]>,
    channels: usize,
    training: bool,
) -> Result<(Option<&'a mut [f32]>, Option<&'a mut [f32]>), FunctionalError> {
    match (running_mean, running_variance) {
        (Some(mean), Some(variance)) => {
            validate_parameter(mean, channels, "batch norm running mean")?;
            validate_parameter(variance, channels, "batch norm running variance")?;
            Ok((Some(mean), Some(variance)))
        }
        (None, None) if training => Ok((None, None)),
        (None, None) => Err(FunctionalError::MissingRunningStatistics),
        _ => Err(FunctionalError::UnpairedRunningStatistics),
    }
}

fn validate_running_statistics_ref<'a>(
    running_mean: Option<&'a [f32]>,
    running_variance: Option<&'a [f32]>,
    channels: usize,
    training: bool,
) -> Result<(Option<&'a [f32]>, Option<&'a [f32]>), FunctionalError> {
    match (running_mean, running_variance) {
        (Some(mean), Some(variance)) => {
            validate_parameter(mean, channels, "batch norm running mean")?;
            validate_parameter(variance, channels, "batch norm running variance")?;
            Ok((Some(mean), Some(variance)))
        }
        (None, None) if training => Ok((None, None)),
        (None, None) => Err(FunctionalError::MissingRunningStatistics),
        _ => Err(FunctionalError::UnpairedRunningStatistics),
    }
}

fn validate_trailing_shape(
    shape: &[usize],
    normalized_shape: &[usize],
) -> Result<usize, FunctionalError> {
    if normalized_shape.is_empty() {
        return Err(FunctionalError::EmptyDimensions);
    }
    if normalized_shape.len() > shape.len()
        || shape[shape.len() - normalized_shape.len()..] != *normalized_shape
    {
        return Err(FunctionalError::NormalizedShape {
            normalized: copy_usize(normalized_shape, "normalized shape error")?,
            input: copy_usize(shape, "input shape error")?,
        });
    }
    let group_size = checked_product(normalized_shape)?;
    if group_size == 0 {
        return Err(FunctionalError::InsufficientGroupValues);
    }
    Ok(group_size)
}

fn validate_parameter(
    parameter: &[f32],
    expected: usize,
    name: &'static str,
) -> Result<(), FunctionalError> {
    if parameter.len() != expected {
        return Err(FunctionalError::ParameterValueCount {
            name,
            expected,
            actual: parameter.len(),
        });
    }
    Ok(())
}

fn validate_optional_parameter(
    parameter: Option<&[f32]>,
    expected: usize,
    name: &'static str,
) -> Result<(), FunctionalError> {
    if let Some(parameter) = parameter {
        validate_parameter(parameter, expected, name)?;
    }
    Ok(())
}

fn optional_read(
    values: Option<&[f32]>,
    index: usize,
    default: f32,
) -> Result<f32, FunctionalError> {
    values.map_or(Ok(default), |values| read(values, index))
}

fn add_optional(
    values: &mut Option<Vec<f32>>,
    index: usize,
    value: f32,
) -> Result<(), FunctionalError> {
    if let Some(values) = values {
        let destination = values
            .get_mut(index)
            .ok_or(FunctionalError::ShapeOverflow)?;
        *destination += value;
    }
    Ok(())
}

fn mean_and_inverse(source: &[f32], epsilon: f32) -> Result<(f32, f32), FunctionalError> {
    if source.is_empty() {
        return Err(FunctionalError::InsufficientGroupValues);
    }
    let mean =
        (source.iter().map(|value| f64::from(*value)).sum::<f64>() / source.len() as f64) as f32;
    let variance = source
        .iter()
        .map(|value| {
            let difference = f64::from(*value) - f64::from(mean);
            difference * difference
        })
        .sum::<f64>()
        / source.len() as f64;
    Ok((mean, (variance + f64::from(epsilon)).sqrt().recip() as f32))
}

fn root_mean_square_inverse(source: &[f32], epsilon: f32) -> Result<f32, FunctionalError> {
    if source.is_empty() {
        return Err(FunctionalError::InsufficientGroupValues);
    }
    let mean_square = source
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>()
        / source.len() as f64;
    Ok((mean_square + f64::from(epsilon)).sqrt().recip() as f32)
}

fn silu_scalar(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub fn gelu_scalar_exact_native(value: f32, approximation: GeluApproximation) -> f32 {
    match approximation {
        GeluApproximation::None => {
            0.5 * value * (1.0 + erf_approximation(value * std::f32::consts::FRAC_1_SQRT_2))
        }
        GeluApproximation::Tanh => {
            let inner =
                (2.0 / std::f32::consts::PI).sqrt() * (value + 0.044_715 * value * value * value);
            0.5 * value * (1.0 + inner.tanh())
        }
    }
}

fn gelu_derivative(value: f32, approximation: GeluApproximation) -> f32 {
    match approximation {
        GeluApproximation::None => {
            let normal_cdf =
                0.5 * (1.0 + erf_approximation(value * std::f32::consts::FRAC_1_SQRT_2));
            normal_cdf + value * (-0.5 * value * value).exp() / (2.0 * std::f32::consts::PI).sqrt()
        }
        GeluApproximation::Tanh => {
            let coefficient = (2.0 / std::f32::consts::PI).sqrt();
            let inner = coefficient * (value + 0.044_715 * value * value * value);
            let tanh = inner.tanh();
            0.5 * (1.0 + tanh)
                + 0.5
                    * value
                    * (1.0 - tanh * tanh)
                    * coefficient
                    * (1.0 + 3.0 * 0.044_715 * value * value)
        }
    }
}

fn erf_approximation(value: f32) -> f32 {
    if value.is_nan() {
        return f32::NAN;
    }
    if value.is_infinite() {
        return value.signum();
    }
    let sign = value.signum();
    let absolute = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * absolute);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - polynomial * (-absolute * absolute).exp())
}

fn elementwise_forward(
    input: &[f32],
    device: DeviceId,
    cancellation: &CancellationToken,
    function: impl Fn(f32) -> f32,
) -> Result<Vec<f32>, FunctionalError> {
    validate_device(device)?;
    cancellation.check()?;
    let mut output = zeroed(input.len(), "elementwise output")?;
    for (index, (value, destination)) in input.iter().zip(&mut output).enumerate() {
        check_periodically(index, cancellation)?;
        *destination = function(*value);
    }
    Ok(output)
}

fn elementwise_in_place_with_context(
    backend: &CpuBackend,
    input: &mut [f32],
    device: DeviceId,
    context: &ExecutionContext<'_>,
    function: impl Fn(f32) -> f32,
) -> Result<(), FunctionalError> {
    validate_device(device)?;
    context.check()?;
    let mut output = temporary_zeroed(backend, context, input.len())?;
    for (index, (value, destination)) in input.iter().zip(&mut *output).enumerate() {
        check_periodically(index, context.cancellation)?;
        *destination = function(*value);
    }
    context.check()?;
    input.copy_from_slice(&output);
    Ok(())
}

fn elementwise_gradient(
    input: &[f32],
    gradient: &[f32],
    device: DeviceId,
    cancellation: &CancellationToken,
    derivative: impl Fn(f32) -> f32,
) -> Result<Vec<f32>, FunctionalError> {
    validate_gradient(input, gradient)?;
    validate_device(device)?;
    cancellation.check()?;
    let mut output = zeroed(input.len(), "elementwise gradient")?;
    for (index, ((value, gradient), destination)) in
        input.iter().zip(gradient).zip(&mut output).enumerate()
    {
        check_periodically(index, cancellation)?;
        *destination = derivative(*value) * gradient;
    }
    Ok(output)
}

fn softmax_linearized(
    output: &[f32],
    direction: &[f32],
    shape: &[usize],
    dimension: isize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_gradient(output, direction)?;
    let plan = AxisPlan::new(shape, dimension)?;
    let mut result = zeroed(output.len(), "softmax gradient")?;
    for outer in 0..plan.outer {
        for inner in 0..plan.inner {
            cancellation.check()?;
            let mut dot_product = 0.0_f32;
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                dot_product += read(output, index)? * read(direction, index)?;
            }
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                let value = read(output, index)? * (read(direction, index)? - dot_product);
                write(&mut result, index, value)?;
            }
        }
    }
    Ok(result)
}

fn log_softmax_linearized(
    output: &[f32],
    direction: &[f32],
    shape: &[usize],
    dimension: isize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_gradient(output, direction)?;
    validate_shape(output, shape)?;
    let plan = AxisPlan::new(shape, dimension)?;
    let mut result = zeroed(output.len(), "log softmax gradient")?;
    for outer in 0..plan.outer {
        for inner in 0..plan.inner {
            cancellation.check()?;
            let mut sum = 0.0_f32;
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                sum += read(direction, index)?;
            }
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                let value = read(direction, index)? - read(output, index)?.exp() * sum;
                write(&mut result, index, value)?;
            }
        }
    }
    Ok(result)
}

fn log_softmax_jvp_linearized(
    output: &[f32],
    input_tangent: &[f32],
    shape: &[usize],
    dimension: isize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, FunctionalError> {
    validate_gradient(output, input_tangent)?;
    validate_shape(output, shape)?;
    let plan = AxisPlan::new(shape, dimension)?;
    let mut result = zeroed(output.len(), "log softmax tangent")?;
    for outer in 0..plan.outer {
        for inner in 0..plan.inner {
            cancellation.check()?;
            let mut softmax_dot_tangent = 0.0_f32;
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                softmax_dot_tangent += read(output, index)?.exp() * read(input_tangent, index)?;
            }
            for position in 0..plan.width {
                let index = plan.index(outer, position, inner)?;
                write(
                    &mut result,
                    index,
                    read(input_tangent, index)? - softmax_dot_tangent,
                )?;
            }
        }
    }
    Ok(result)
}

fn normalization_reductions(
    input: &[f32],
    gradient: &[f32],
    plan: &ReductionPlan,
    norm_order: f32,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    cancellation: &CancellationToken,
) -> Result<(CpuWorkspaceVec<f32>, CpuWorkspaceVec<f32>), FunctionalError> {
    let mut powers = temporary_zeroed(backend, context, plan.groups)?;
    let mut dot_products = temporary_zeroed(backend, context, plan.groups)?;
    for index in 0..input.len() {
        check_periodically(index, cancellation)?;
        let group = plan.group(index)?;
        let value = read(input, index)?;
        *powers
            .get_mut(group)
            .ok_or(FunctionalError::ShapeOverflow)? += value.abs().powf(norm_order);
        *dot_products
            .get_mut(group)
            .ok_or(FunctionalError::ShapeOverflow)? += value * read(gradient, index)?;
    }
    for value in powers.iter_mut() {
        *value = value.powf(norm_order.recip());
    }
    Ok((powers, dot_products))
}

struct AxisPlan {
    outer: usize,
    width: usize,
    inner: usize,
}

impl AxisPlan {
    fn new(shape: &[usize], dimension: isize) -> Result<Self, FunctionalError> {
        if shape.is_empty() {
            return Err(FunctionalError::Rank {
                minimum: 1,
                actual: 0,
            });
        }
        let axis = normalize_dimension(dimension, shape.len())?;
        Ok(Self {
            outer: checked_product(&shape[..axis])?,
            width: shape[axis],
            inner: checked_product(&shape[axis + 1..])?,
        })
    }

    fn index(&self, outer: usize, position: usize, inner: usize) -> Result<usize, FunctionalError> {
        outer
            .checked_mul(self.width)
            .and_then(|value| value.checked_add(position))
            .and_then(|value| value.checked_mul(self.inner))
            .and_then(|value| value.checked_add(inner))
            .ok_or(FunctionalError::ShapeOverflow)
    }
}

struct ReductionPlan<'a> {
    shape: &'a [usize],
    dimensions: &'a [isize],
    groups: usize,
}

impl<'a> ReductionPlan<'a> {
    fn new(shape: &'a [usize], dimensions: &'a [isize]) -> Result<Self, FunctionalError> {
        if shape.is_empty() {
            return Err(FunctionalError::Rank {
                minimum: 1,
                actual: 0,
            });
        }
        if dimensions.is_empty() {
            return Err(FunctionalError::EmptyDimensions);
        }
        for (position, dimension) in dimensions.iter().enumerate() {
            let axis = normalize_dimension(*dimension, shape.len())?;
            if dimensions[..position]
                .iter()
                .try_fold(false, |duplicate, prior| {
                    Ok::<_, FunctionalError>(
                        duplicate || normalize_dimension(*prior, shape.len())? == axis,
                    )
                })?
            {
                return Err(FunctionalError::DuplicateDimension { axis });
            }
        }
        let groups = shape
            .iter()
            .enumerate()
            .try_fold(1_usize, |product, (axis, dimension)| {
                if dimensions.iter().try_fold(false, |reduced, value| {
                    Ok::<_, FunctionalError>(
                        reduced || normalize_dimension(*value, shape.len())? == axis,
                    )
                })? {
                    Ok(product)
                } else {
                    product
                        .checked_mul(*dimension)
                        .ok_or(FunctionalError::ShapeOverflow)
                }
            })?;
        Ok(Self {
            shape,
            dimensions,
            groups,
        })
    }

    fn group(&self, mut flat_index: usize) -> Result<usize, FunctionalError> {
        let mut group = 0_usize;
        let mut group_stride = 1_usize;
        for (axis, dimension) in self.shape.iter().enumerate().rev() {
            if *dimension == 0 {
                return Err(FunctionalError::ShapeOverflow);
            }
            let coordinate = flat_index % dimension;
            flat_index /= dimension;
            let reduced = self.dimensions.iter().try_fold(false, |reduced, value| {
                Ok::<_, FunctionalError>(
                    reduced || normalize_dimension(*value, self.shape.len())? == axis,
                )
            })?;
            if !reduced {
                group = group
                    .checked_add(
                        coordinate
                            .checked_mul(group_stride)
                            .ok_or(FunctionalError::ShapeOverflow)?,
                    )
                    .ok_or(FunctionalError::ShapeOverflow)?;
                group_stride = group_stride
                    .checked_mul(*dimension)
                    .ok_or(FunctionalError::ShapeOverflow)?;
            }
        }
        Ok(group)
    }
}

fn validate_shape(input: &[f32], shape: &[usize]) -> Result<(), FunctionalError> {
    let expected = checked_product(shape)?;
    if input.len() != expected {
        return Err(FunctionalError::ValueCount {
            expected,
            actual: input.len(),
        });
    }
    Ok(())
}

fn validate_gradient(input: &[f32], gradient: &[f32]) -> Result<(), FunctionalError> {
    if input.len() != gradient.len() {
        return Err(FunctionalError::ValueCount {
            expected: input.len(),
            actual: gradient.len(),
        });
    }
    Ok(())
}

fn validate_tensor_input(
    backend: &dyn TensorBackend,
    input: &Tensor,
    _operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), FunctionalError> {
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
    if input.descriptor().dtype().class() != NumericClass::FloatingPoint {
        return Err(TensorError::InvalidNumeric {
            reason: format!(
                "normalization requires a floating-point tensor, got {}",
                input.descriptor().dtype().catalog_name()
            ),
        }
        .into());
    }
    Ok(())
}

fn validate_channel_parameters(
    backend: &dyn TensorBackend,
    input: &Tensor,
    first: Option<&Tensor>,
    second: Option<&Tensor>,
    channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<(), FunctionalError> {
    for parameter in [first, second].into_iter().flatten() {
        validate_tensor_input(backend, parameter, "normalization parameter", context)?;
        if parameter.descriptor().dtype() != input.descriptor().dtype() {
            return Err(TensorError::DTypeMismatch {
                expected: input.descriptor().dtype(),
                actual: parameter.descriptor().dtype(),
            }
            .into());
        }
        if parameter.descriptor().shape() != [channels] {
            return Err(FunctionalError::ParameterValueCount {
                name: "channel parameter",
                expected: usize::try_from(channels).map_err(|_| FunctionalError::ShapeOverflow)?,
                actual: usize::try_from(parameter.descriptor().element_count()?)
                    .map_err(|_| FunctionalError::ShapeOverflow)?,
            });
        }
    }
    Ok(())
}

fn allocate_tensor_output(
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.allocate(descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn finish_tensor_output(
    backend: &dyn TensorBackend,
    output: Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, FunctionalError> {
    let event = backend.record_event(context)?;
    backend.wait_event(event, context)?;
    context.check()?;
    Ok(output)
}

fn read_tensor_real_linear(input: &Tensor, linear: u64) -> Result<f64, FunctionalError> {
    match input
        .descriptor()
        .dtype()
        .decode_scalar(input.linear_element_bytes(linear)?)?
    {
        DecodedScalar::Real(value) => Ok(value),
        value => Err(TensorError::InvalidNumeric {
            reason: format!("normalization requires real tensor values, got {value:?}"),
        }
        .into()),
    }
}

fn write_tensor_real_linear(
    write: &mut TensorWrite<'_>,
    dtype: DType,
    device: DeviceId,
    linear: u64,
    value: f64,
    operation: &'static str,
) -> Result<(), FunctionalError> {
    let byte_width = usize::try_from(dtype.byte_width()).map_err(|_| FunctionalError::ShapeOverflow)?;
    let start = usize::try_from(linear)
        .map_err(|_| FunctionalError::ShapeOverflow)?
        .checked_mul(byte_width)
        .ok_or(FunctionalError::ShapeOverflow)?;
    let end = start
        .checked_add(byte_width)
        .ok_or(FunctionalError::ShapeOverflow)?;
    let encoded = dtype.encode_scalar(Scalar::Float(value), operation, device)?;
    write
        .bytes_mut()?
        .get_mut(start..end)
        .ok_or(FunctionalError::ShapeOverflow)?
        .copy_from_slice(&encoded);
    Ok(())
}

fn checked_u64_product(values: &[u64], _name: &'static str) -> Result<u64, FunctionalError> {
    values.iter().try_fold(1_u64, |product, value| {
        product.checked_mul(*value).ok_or(FunctionalError::ShapeOverflow)
    })
}

fn channel_spatial_index(
    channels: u64,
    spatial: u64,
    batch: u64,
    channel: u64,
    spatial_index: u64,
) -> Result<u64, FunctionalError> {
    batch
        .checked_mul(channels)
        .and_then(|value| value.checked_add(channel))
        .and_then(|value| value.checked_mul(spatial))
        .and_then(|value| value.checked_add(spatial_index))
        .ok_or(FunctionalError::ShapeOverflow)
}

fn normalize_tensor_axis(dimension: isize, rank: usize) -> Result<usize, FunctionalError> {
    normalize_dimension(dimension, rank)
}

fn validate_epsilon_f64(epsilon: f64) -> Result<(), FunctionalError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(FunctionalError::InvalidEpsilon);
    }
    Ok(())
}

fn validate_device(device: DeviceId) -> Result<(), FunctionalError> {
    if device != DeviceId::CPU {
        return Err(FunctionalError::UnsupportedDevice { device });
    }
    Ok(())
}

fn validate_epsilon(epsilon: f32) -> Result<(), FunctionalError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(FunctionalError::InvalidEpsilon);
    }
    Ok(())
}

fn validate_negative_slope(negative_slope: f32) -> Result<(), FunctionalError> {
    if !negative_slope.is_finite() {
        return Err(FunctionalError::InvalidNegativeSlope);
    }
    Ok(())
}

fn normalize_dimension(dimension: isize, rank: usize) -> Result<usize, FunctionalError> {
    let rank_isize = isize::try_from(rank).map_err(|_| FunctionalError::ShapeOverflow)?;
    let normalized = if dimension < 0 {
        rank_isize
            .checked_add(dimension)
            .ok_or(FunctionalError::InvalidDimension { dimension, rank })?
    } else {
        dimension
    };
    if normalized < 0 || normalized >= rank_isize {
        return Err(FunctionalError::InvalidDimension { dimension, rank });
    }
    usize::try_from(normalized).map_err(|_| FunctionalError::InvalidDimension { dimension, rank })
}

fn checked_product(values: &[usize]) -> Result<usize, FunctionalError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(FunctionalError::ShapeOverflow)
    })
}

fn zeroed(length: usize, name: &'static str) -> Result<Vec<f32>, FunctionalError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| FunctionalError::AllocationFailed { name })?;
    values.resize(length, 0.0);
    Ok(values)
}

fn temporary_zeroed(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    length: usize,
) -> Result<CpuWorkspaceVec<f32>, FunctionalError> {
    let mut values = backend.workspace_vec(context, length)?;
    for _ in 0..length {
        values.try_push(0.0)?;
    }
    Ok(values)
}

fn temporary_copy(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    values: &[f32],
) -> Result<CpuWorkspaceVec<f32>, FunctionalError> {
    let mut output = backend.workspace_vec(context, values.len())?;
    for value in values {
        output.try_push(*value)?;
    }
    Ok(output)
}

fn copy_usize(values: &[usize], name: &'static str) -> Result<Vec<usize>, FunctionalError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| FunctionalError::AllocationFailed { name })?;
    output.extend_from_slice(values);
    Ok(output)
}

fn read(values: &[f32], index: usize) -> Result<f32, FunctionalError> {
    values
        .get(index)
        .copied()
        .ok_or(FunctionalError::ShapeOverflow)
}

fn write(values: &mut [f32], index: usize, value: f32) -> Result<(), FunctionalError> {
    let destination = values
        .get_mut(index)
        .ok_or(FunctionalError::ShapeOverflow)?;
    *destination = value;
    Ok(())
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), FunctionalError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}
