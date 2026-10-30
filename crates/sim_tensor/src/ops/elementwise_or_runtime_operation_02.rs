use crate::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar,
    DeviceId, ExecutionContext, GradientMode, Layout, Scalar, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError, UnaryOperation,
};
use thiserror::Error;

pub const CEIL_OPERATION_ID: &str = "COMFY-TENSOR-OP-1599F5E140D0";
pub const TANH_OPERATION_ID: &str = "COMFY-TENSOR-OP-10FC4A6ED9AA";
pub const ACOS_OPERATION_ID: &str = "COMFY-TENSOR-OP-0FB8594194A8";
pub const CUDNN_AVAILABLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-14815FE141B4";
pub const EQ_OPERATION_ID: &str = "COMFY-TENSOR-OP-14218933001D";
pub const INFERENCE_MODE_OPERATION_ID: &str = "COMFY-TENSOR-OP-160E75523010";
pub const JIT_SCRIPTING_OPERATION_ID: &str = "COMFY-TENSOR-OP-190DE0F94657";
pub const LOG_OPERATION_ID: &str = "COMFY-TENSOR-OP-1912E4160DE1";
pub const NPU_MEMORY_STATS_OPERATION_ID: &str = "COMFY-TENSOR-OP-147180FA6AF4";
pub const ADAMW_OPERATION_ID: &str = "COMFY-TENSOR-OP-1602683BB161";
pub const POLAR_OPERATION_ID: &str = "COMFY-TENSOR-OP-10A0FC173128";
pub const VIEW_AS_COMPLEX_OPERATION_ID: &str = "COMFY-TENSOR-OP-11C887BB4214";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ElementwiseRuntimePartTwoError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("elementwise/runtime part-two operation was cancelled")]
    Cancelled,
    #[error("elementwise/runtime part-two input is invalid: {0}")]
    Invalid(&'static str),
    #[error("operation {operation} is unavailable for device {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} does not support dtype {dtype:?}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartTwoError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn ceil_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    map_real_preserving_dtype(backend, input, CEIL_OPERATION_ID, f64::ceil, context)
}

pub fn ceil_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    zeros_like_f32(backend, input, CEIL_OPERATION_ID, context)
}

pub fn ceil_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    ceil_vjp_with_context_exact_native(backend, input, context)
}

pub fn tanh_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    unary_float_preserving_dtype(
        backend,
        input,
        TANH_OPERATION_ID,
        UnaryOperation::HyperbolicTangent,
        context,
    )
}

pub fn tanh_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    elementwise_f32_binary(
        backend,
        input,
        output_gradient,
        TANH_OPERATION_ID,
        |value, gradient| {
            let output = value.tanh();
            gradient * (1.0 - output * output)
        },
        context,
    )
}

pub fn tanh_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    tanh_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn acos_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    map_f32(backend, input, ACOS_OPERATION_ID, f32::acos, context)
}

pub fn acos_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    elementwise_f32_binary(
        backend,
        input,
        output_gradient,
        ACOS_OPERATION_ID,
        |value, gradient| -gradient / (1.0 - value * value).sqrt(),
        context,
    )
}

pub fn acos_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    acos_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn cudnn_is_available_exact_native(
    _capabilities: &BackendCapabilityMatrix,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartTwoError> {
    cancellation.check()?;
    cancellation.check()?;
    Ok(false)
}

pub fn equal_scalar_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: Scalar,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    require_cpu(input, EQ_OPERATION_ID)?;
    let encoded =
        input
            .descriptor()
            .dtype()
            .encode_scalar(other, EQ_OPERATION_ID, DeviceId::CPU)?;
    let other = input.descriptor().dtype().decode_scalar(&encoded)?;
    let element_count = element_count(input, "equality output")?;
    let mut bytes = temporary_vec(backend, context, element_count, "equality output")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        bytes.try_push(u8::from(decoded_equal(value, other)))?;
    }
    upload_bytes(
        backend,
        input.descriptor().shape(),
        DType::Bool,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn is_inference_mode_enabled_exact_native(
    mode: GradientMode,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartTwoError> {
    cancellation.check()?;
    let enabled = mode == GradientMode::Inference;
    cancellation.check()?;
    Ok(enabled)
}

pub fn jit_is_scripting_exact_native(
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartTwoError> {
    cancellation.check()?;
    Ok(false)
}

pub fn log_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    unary_f32(
        backend,
        input,
        LOG_OPERATION_ID,
        UnaryOperation::NaturalLogarithm,
        context,
    )
}

pub fn log_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    elementwise_f32_binary(
        backend,
        input,
        output_gradient,
        LOG_OPERATION_ID,
        |value, gradient| gradient / value,
        context,
    )
}

pub fn log_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    log_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_with_context_exact_native(
    backend: &CpuBackend,
    parameters: &mut [Tensor],
    gradients: &[Tensor],
    exponential_averages: &mut [Tensor],
    exponential_average_squares: &mut [Tensor],
    maximum_exponential_average_squares: &mut [Tensor],
    steps: &[u64],
    amsgrad: bool,
    beta1: f32,
    beta2: f32,
    learning_rate: f32,
    weight_decay: f32,
    epsilon: f32,
    maximize: bool,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    if parameters.len() != gradients.len()
        || parameters.len() != exponential_averages.len()
        || parameters.len() != exponential_average_squares.len()
        || parameters.len() != steps.len()
        || (amsgrad && parameters.len() != maximum_exponential_average_squares.len())
        || (!amsgrad && !maximum_exponential_average_squares.is_empty())
    {
        return Err(ElementwiseRuntimePartTwoError::Invalid(
            "AdamW parallel state lists must have matching lengths",
        ));
    }
    if !(0.0..1.0).contains(&beta1)
        || !(0.0..1.0).contains(&beta2)
        || !learning_rate.is_finite()
        || learning_rate < 0.0
        || !weight_decay.is_finite()
        || weight_decay < 0.0
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err(ElementwiseRuntimePartTwoError::Invalid(
            "invalid AdamW configuration",
        ));
    }
    let mut staged = reserved_vec(parameters.len(), "AdamW staged parameter list")?;
    for index in 0..parameters.len() {
        context.check()?;
        let maximum = amsgrad.then(|| &maximum_exponential_average_squares[index]);
        staged.push(stage_adamw_parameter(
            backend,
            &parameters[index],
            &gradients[index],
            &exponential_averages[index],
            &exponential_average_squares[index],
            maximum,
            steps[index],
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            epsilon,
            maximize,
            context,
        )?);
    }
    context.check()?;
    for (index, next) in staged.into_iter().enumerate() {
        parameters[index].commit_in_place(next.parameter)?;
        exponential_averages[index].commit_in_place(next.exponential_average)?;
        exponential_average_squares[index].commit_in_place(next.exponential_average_square)?;
        if let Some(maximum) = next.maximum_exponential_average_square {
            maximum_exponential_average_squares[index].commit_in_place(maximum)?;
        }
    }
    Ok(())
}

struct StagedAdamWParameter {
    parameter: Tensor,
    exponential_average: Tensor,
    exponential_average_square: Tensor,
    maximum_exponential_average_square: Option<Tensor>,
}

#[allow(clippy::too_many_arguments)]
fn stage_adamw_parameter(
    backend: &CpuBackend,
    parameter: &Tensor,
    gradient: &Tensor,
    exponential_average: &Tensor,
    exponential_average_square: &Tensor,
    maximum_exponential_average_square: Option<&Tensor>,
    step: u64,
    beta1: f32,
    beta2: f32,
    learning_rate: f32,
    weight_decay: f32,
    epsilon: f32,
    maximize: bool,
    context: &ExecutionContext<'_>,
) -> Result<StagedAdamWParameter, ElementwiseRuntimePartTwoError> {
    if step == 0 {
        return Err(ElementwiseRuntimePartTwoError::Invalid(
            "AdamW steps must be positive",
        ));
    }
    require_same_f32(parameter, gradient, ADAMW_OPERATION_ID)?;
    require_same_f32(parameter, exponential_average, ADAMW_OPERATION_ID)?;
    require_same_f32(parameter, exponential_average_square, ADAMW_OPERATION_ID)?;
    if let Some(maximum) = maximum_exponential_average_square {
        require_same_f32(parameter, maximum, ADAMW_OPERATION_ID)?;
    }
    let parameters = tensor_f32(backend, parameter, context)?;
    let gradients = tensor_f32(backend, gradient, context)?;
    let averages = tensor_f32(backend, exponential_average, context)?;
    let average_squares = tensor_f32(backend, exponential_average_square, context)?;
    let maximum_squares = maximum_exponential_average_square
        .map(|value| tensor_f32(backend, value, context))
        .transpose()?;
    let mut next_parameters =
        temporary_vec(backend, context, parameters.len(), "AdamW parameters")?;
    let mut next_averages = temporary_vec(backend, context, parameters.len(), "AdamW averages")?;
    let mut next_average_squares =
        temporary_vec(backend, context, parameters.len(), "AdamW average squares")?;
    let mut next_maximum_squares = maximum_squares
        .as_ref()
        .map(|_| {
            temporary_vec(
                backend,
                context,
                parameters.len(),
                "AdamW maximum average squares",
            )
        })
        .transpose()?;
    let step_i32 = i32::try_from(step)
        .map_err(|_| ElementwiseRuntimePartTwoError::ShapeOverflow("AdamW step"))?;
    let bias_correction1 = 1.0 - beta1.powi(step_i32);
    let bias_correction2 = 1.0 - beta2.powi(step_i32);
    for index in 0..parameters.len() {
        check_periodically(index, context.cancellation)?;
        let gradient = if maximize {
            -gradients[index]
        } else {
            gradients[index]
        };
        let average = beta1 * averages[index] + (1.0 - beta1) * gradient;
        let average_square = beta2 * average_squares[index] + (1.0 - beta2) * gradient * gradient;
        let denominator_square = if let (Some(previous), Some(next)) =
            (maximum_squares.as_ref(), next_maximum_squares.as_mut())
        {
            let maximum = previous[index].max(average_square);
            next.try_push(maximum)?;
            maximum
        } else {
            average_square
        };
        let denominator = (denominator_square.sqrt() / bias_correction2.sqrt()) + epsilon;
        let decayed = parameters[index] * (1.0 - learning_rate * weight_decay);
        next_parameters
            .try_push(decayed - (learning_rate / bias_correction1) * average / denominator)?;
        next_averages.try_push(average)?;
        next_average_squares.try_push(average_square)?;
    }
    let shape = parameter.descriptor().shape();
    Ok(StagedAdamWParameter {
        parameter: tensor_from_f32(backend, shape, &next_parameters, context)?,
        exponential_average: tensor_from_f32(backend, shape, &next_averages, context)?,
        exponential_average_square: tensor_from_f32(
            backend,
            shape,
            &next_average_squares,
            context,
        )?,
        maximum_exponential_average_square: next_maximum_squares
            .as_deref()
            .map(|values| tensor_from_f32(backend, shape, values, context))
            .transpose()?,
    })
}

pub fn polar_with_context_exact_native(
    backend: &CpuBackend,
    magnitude: &Tensor,
    angle: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    context.cancellation.check()?;
    require_same_f32(magnitude, angle, POLAR_OPERATION_ID)?;
    let magnitudes = tensor_f32(backend, magnitude, context)?;
    let angles = tensor_f32(backend, angle, context)?;
    let byte_len =
        magnitudes
            .len()
            .checked_mul(8)
            .ok_or(ElementwiseRuntimePartTwoError::ShapeOverflow(
                "polar output",
            ))?;
    let mut bytes = temporary_vec(backend, context, byte_len, "polar output")?;
    for (index, (magnitude, angle)) in magnitudes.iter().zip(angles.iter()).enumerate() {
        check_periodically(index, context.cancellation)?;
        temporary_extend(&mut bytes, &(*magnitude * angle.cos()).to_ne_bytes())?;
        temporary_extend(&mut bytes, &(*magnitude * angle.sin()).to_ne_bytes())?;
    }
    upload_bytes(
        backend,
        magnitude.descriptor().shape(),
        DType::Complex64,
        magnitude.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn view_as_complex_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    cancellation.check()?;
    require_cpu(input, VIEW_AS_COMPLEX_OPERATION_ID)?;
    if input.descriptor().dtype() != DType::F32
        || input.descriptor().layout() != Layout::Contiguous
        || input.descriptor().shape().last().copied() != Some(2)
    {
        return Err(ElementwiseRuntimePartTwoError::Invalid(
            "view_as_complex requires contiguous f32 input with trailing dimension two",
        ));
    }
    let mut shape = input.descriptor().shape().to_vec();
    shape.pop();
    let descriptor = TensorDescriptor::contiguous(
        shape,
        DType::Complex64,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let output = input.reinterpret_contiguous_read_only(descriptor)?;
    cancellation.check()?;
    Ok(output)
}

pub fn view_as_complex_vjp_exact_native(
    output_gradient: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    cancellation.check()?;
    require_cpu(output_gradient, VIEW_AS_COMPLEX_OPERATION_ID)?;
    if output_gradient.descriptor().dtype() != DType::Complex64
        || output_gradient.descriptor().layout() != Layout::Contiguous
    {
        return Err(ElementwiseRuntimePartTwoError::Invalid(
            "view_as_complex VJP requires contiguous complex64 output gradient",
        ));
    }
    let mut shape = output_gradient.descriptor().shape().to_vec();
    shape.push(2);
    let descriptor = TensorDescriptor::contiguous(
        shape,
        DType::F32,
        output_gradient.descriptor().device(),
        output_gradient.descriptor().stream(),
    )?;
    let input_gradient = output_gradient.reinterpret_contiguous_read_only(descriptor)?;
    cancellation.check()?;
    Ok(input_gradient)
}

pub fn view_as_complex_jvp_exact_native(
    input_tangent: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    view_as_complex_exact_native(input_tangent, cancellation)
}

fn unary_float_preserving_dtype(
    backend: &CpuBackend,
    input: &Tensor,
    operation_id: &'static str,
    operation: UnaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    require_cpu(input, operation_id)?;
    let dtype = input.descriptor().dtype();
    if !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(ElementwiseRuntimePartTwoError::UnsupportedDType {
            operation: operation_id,
            dtype,
        });
    }
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        dtype,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    let (output, _) = backend.unary(operation, input, descriptor, context)?;
    context.check()?;
    Ok(output)
}

fn unary_f32(
    backend: &CpuBackend,
    input: &Tensor,
    operation_id: &'static str,
    operation: UnaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    require_f32(input, operation_id)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    let (output, _) = backend.unary(operation, input, descriptor, context)?;
    context.check()?;
    Ok(output)
}

fn map_f32(
    backend: &CpuBackend,
    input: &Tensor,
    operation_id: &'static str,
    operation: impl Fn(f32) -> f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    require_f32(input, operation_id)?;
    let values = tensor_f32(backend, input, context)?;
    let mut output = temporary_vec(backend, context, values.len(), "f32 unary output")?;
    for (index, value) in values.iter().enumerate() {
        check_periodically(index, context.cancellation)?;
        output.try_push(operation(*value))?;
    }
    tensor_from_f32(backend, input.descriptor().shape(), &output, context)
}

fn map_real_preserving_dtype(
    backend: &CpuBackend,
    input: &Tensor,
    operation_id: &'static str,
    operation: impl Fn(f64) -> f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    require_cpu(input, operation_id)?;
    let element_count = element_count(input, "real unary output")?;
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| ElementwiseRuntimePartTwoError::ShapeOverflow("real unary width"))?;
    let byte_len =
        element_count
            .checked_mul(width)
            .ok_or(ElementwiseRuntimePartTwoError::ShapeOverflow(
                "real unary output",
            ))?;
    let mut bytes = temporary_vec(backend, context, byte_len, "real unary output")?;
    for linear_index in 0..element_count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        match input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?
        {
            DecodedScalar::Real(value) => temporary_extend(
                &mut bytes,
                &input.descriptor().dtype().encode_scalar(
                    Scalar::Float(operation(value)),
                    operation_id,
                    DeviceId::CPU,
                )?,
            )?,
            DecodedScalar::Boolean(_) | DecodedScalar::Signed(_) | DecodedScalar::Unsigned(_) => {
                temporary_extend(&mut bytes, input.element_bytes(&indices)?)?;
            }
            DecodedScalar::Complex { .. } => {
                return Err(ElementwiseRuntimePartTwoError::UnsupportedDType {
                    operation: operation_id,
                    dtype: input.descriptor().dtype(),
                });
            }
        }
    }
    upload_bytes(
        backend,
        input.descriptor().shape(),
        input.descriptor().dtype(),
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

fn elementwise_f32_binary(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    operation_id: &'static str,
    operation: impl Fn(f32, f32) -> f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    require_same_f32(left, right, operation_id)?;
    let left_values = tensor_f32(backend, left, context)?;
    let right_values = tensor_f32(backend, right, context)?;
    let mut values = temporary_vec(backend, context, left_values.len(), "binary f32 output")?;
    for (index, (left, right)) in left_values.iter().zip(right_values.iter()).enumerate() {
        check_periodically(index, context.cancellation)?;
        values.try_push(operation(*left, *right))?;
    }
    tensor_from_f32(backend, left.descriptor().shape(), &values, context)
}

fn zeros_like_f32(
    backend: &CpuBackend,
    input: &Tensor,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    require_f32(input, operation_id)?;
    let count = element_count(input, "zero gradient")?;
    let mut values = temporary_vec(backend, context, count, "zero gradient")?;
    for _ in 0..count {
        values.try_push(0.0)?;
    }
    tensor_from_f32(backend, input.descriptor().shape(), &values, context)
}

fn tensor_f32(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<TemporaryVec<f32>, ElementwiseRuntimePartTwoError> {
    require_f32(input, "SIM-TENSOR-CONVERSION-F32")?;
    let count = element_count(input, "decoded f32 values")?;
    let mut values = temporary_vec(backend, context, count, "decoded f32 values")?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let encoded: [u8; 4] = input
            .element_bytes(&indices)?
            .try_into()
            .map_err(|_| ElementwiseRuntimePartTwoError::Invalid("unaligned f32 tensor bytes"))?;
        values.try_push(f32::from_ne_bytes(encoded))?;
    }
    Ok(values)
}

fn tensor_from_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_bytes(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwoError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    let (output, _) = backend.upload_bytes(descriptor, bytes, context)?;
    context.check()?;
    Ok(output)
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwoError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartTwoError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}

fn require_f32(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwoError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartTwoError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}

fn require_same_f32(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwoError> {
    require_f32(left, operation)?;
    require_f32(right, operation)?;
    if left.descriptor().shape() != right.descriptor().shape()
        || left.descriptor().stream() != right.descriptor().stream()
    {
        return Err(ElementwiseRuntimePartTwoError::Invalid(
            "tensor descriptors must share shape and stream",
        ));
    }
    Ok(())
}

fn element_count(
    input: &Tensor,
    context: &'static str,
) -> Result<usize, ElementwiseRuntimePartTwoError> {
    usize::try_from(input.descriptor().element_count()?)
        .map_err(|_| ElementwiseRuntimePartTwoError::ShapeOverflow(context))
}

fn unravel_index(
    mut linear_index: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartTwoError> {
    let mut indices = vec![0; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartTwoError::ShapeOverflow("tensor index"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartTwoError::Invalid(
                "cannot index an empty tensor",
            ));
        }
        indices[axis] = u64::try_from(linear_index % dimension)
            .map_err(|_| ElementwiseRuntimePartTwoError::ShapeOverflow("tensor index"))?;
        linear_index /= dimension;
    }
    Ok(indices)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartTwoError> {
    if index & 1023 == 0 {
        cancellation.check()?;
    }
    Ok(())
}

fn decoded_equal(left: DecodedScalar, right: DecodedScalar) -> bool {
    match (left, right) {
        (DecodedScalar::Boolean(left), DecodedScalar::Boolean(right)) => left == right,
        (DecodedScalar::Signed(left), DecodedScalar::Signed(right)) => left == right,
        (DecodedScalar::Unsigned(left), DecodedScalar::Unsigned(right)) => left == right,
        (DecodedScalar::Real(left), DecodedScalar::Real(right)) => left == right,
        (
            DecodedScalar::Complex {
                real: left_real,
                imaginary: left_imaginary,
            },
            DecodedScalar::Complex {
                real: right_real,
                imaginary: right_imaginary,
            },
        ) => left_real == right_real && left_imaginary == right_imaginary,
        _ => false,
    }
}

fn reserved_vec<T>(
    capacity: usize,
    context: &'static str,
) -> Result<Vec<T>, ElementwiseRuntimePartTwoError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| ElementwiseRuntimePartTwoError::ShapeOverflow(context))?;
    Ok(values)
}

type TemporaryVec<T> = CpuWorkspaceVec<T>;

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    _allocation: &'static str,
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartTwoError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn temporary_extend(
    values: &mut TemporaryVec<u8>,
    extension: &[u8],
) -> Result<(), ElementwiseRuntimePartTwoError> {
    for value in extension {
        values.try_push(*value)?;
    }
    Ok(())
}
