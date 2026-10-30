use crate::{
    BackendCapabilityMatrix, BinaryOperation, CancellationToken, CpuBackend, DType, DecodedScalar,
    DeviceId, DeviceKind, ExecutionContext, GradientMode, Scalar, ScalarSide, StreamId, Tensor,
    TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
    cpu_backend::{CpuWorkspaceVec, binary_broadcast_shape, broadcast_indices},
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_05::{
        ElementwiseRuntimePartFiveError,
        zero_in_place_with_context_exact_native as canonical_zero_in_place_with_context,
    },
    generated_elementwise_or_runtime_operation_06::{
        ElementwiseRuntimePartSixError, round_method_jvp_with_context_exact_native,
        round_method_vjp_with_context_exact_native, round_method_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_09::{
        ElementwiseRuntimePartNineError, NativeAdamW, clamp_jvp_with_context_exact_native,
        clamp_vjp_with_context_exact_native, clamp_with_context_exact_native,
    },
};
use thiserror::Error;

type TemporaryVec<T> = CpuWorkspaceVec<T>;

pub const CLAMP_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-7E962991728B";
pub const TOLIST_OPERATION_ID: &str = "COMFY-TENSOR-OP-7E09C5749B60";
pub const ENABLE_GRAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-7F86521B5D09";
pub const ISCLOSE_OPERATION_ID: &str = "COMFY-TENSOR-OP-7A0F5559B701";
pub const MAXIMUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-7DB0B0EC6483";
pub const ZEROS_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-82BC07D67AFD";
pub const NPU_EMPTY_CACHE_OPERATION_ID: &str = "COMFY-TENSOR-OP-791664AD5273";
pub const NPU_DEVICE_NAME_OPERATION_ID: &str = "COMFY-TENSOR-OP-7EA8F732F7A7";
pub const ADAM_CONSTRUCTOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-790CDE1EBF17";
pub const ROUND_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-8162B4C00596";
pub const SIGN_OPERATION_ID: &str = "COMFY-TENSOR-OP-82310D1230AF";
pub const XPU_MEMORY_STATS_OPERATION_ID: &str = "COMFY-TENSOR-OP-80B937845579";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartElevenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartFive(#[from] ElementwiseRuntimePartFiveError),
    #[error(transparent)]
    PartSix(#[from] ElementwiseRuntimePartSixError),
    #[error(transparent)]
    PartNine(#[from] ElementwiseRuntimePartNineError),
    #[error("elementwise/runtime part-eleven operation was cancelled")]
    Cancelled,
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
    #[error("elementwise/runtime part-eleven input is invalid: {0}")]
    Invalid(String),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartElevenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TensorListValue {
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    Real(f64),
    Complex { real: f64, imaginary: f64 },
    List(Vec<Self>),
}

#[derive(Clone, Debug)]
pub struct MaximumVjp {
    pub input: Tensor,
    pub other: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct NativeAdam {
    inner: NativeAdamW,
}

impl NativeAdam {
    pub fn new_with_context_exact_native(
        backend: &CpuBackend,
        parameters: &[Tensor],
        learning_rate: f32,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ElementwiseRuntimePartElevenError> {
        context.cancellation.check()?;
        Ok(Self {
            inner: NativeAdamW::new_with_context_exact_native(
                backend,
                parameters,
                learning_rate,
                0.9,
                0.999,
                1.0e-8,
                0.0,
                false,
                false,
                context,
            )?,
        })
    }

    pub fn step_with_context_exact_native(
        &mut self,
        backend: &CpuBackend,
        parameters: &mut [Tensor],
        gradients: &[Tensor],
        context: &ExecutionContext<'_>,
    ) -> Result<(), ElementwiseRuntimePartElevenError> {
        context.cancellation.check()?;
        self.inner
            .step_with_context_exact_native(backend, parameters, gradients, context)?;
        Ok(())
    }

    pub fn steps(&self) -> &[u64] {
        self.inner.steps()
    }
}

pub fn clamp_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<Scalar>,
    maximum: Option<Scalar>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    Ok(clamp_with_context_exact_native(
        backend, input, minimum, maximum, context,
    )?)
}

pub fn clamp_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<f32>,
    maximum: Option<f32>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    Ok(clamp_vjp_with_context_exact_native(
        backend,
        input,
        minimum,
        maximum,
        output_gradient,
        context,
    )?)
}

pub fn clamp_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<f32>,
    maximum: Option<f32>,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    Ok(clamp_jvp_with_context_exact_native(
        backend,
        input,
        minimum,
        maximum,
        input_tangent,
        context,
    )?)
}

pub fn tolist_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<TensorListValue, ElementwiseRuntimePartElevenError> {
    cancellation.check()?;
    require_cpu(input, TOLIST_OPERATION_ID)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(input.descriptor().rank())
        .map_err(|_| ElementwiseRuntimePartElevenError::ShapeOverflow("tolist indices"))?;
    build_list(input, 0, &mut indices, cancellation)
}

pub fn enable_grad_exact_native(
    cancellation: &CancellationToken,
) -> Result<GradientMode, ElementwiseRuntimePartElevenError> {
    cancellation.check()?;
    Ok(GradientMode::Enabled)
}

pub fn isclose_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    relative_tolerance: f32,
    absolute_tolerance: f32,
    equal_nan: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    isclose_impl(
        backend,
        input,
        other,
        relative_tolerance,
        absolute_tolerance,
        equal_nan,
        context,
    )
}

fn isclose_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    relative_tolerance: f32,
    absolute_tolerance: f32,
    equal_nan: bool,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    require_same_f32_stream(input, other, ISCLOSE_OPERATION_ID)?;
    if !relative_tolerance.is_finite()
        || relative_tolerance < 0.0
        || !absolute_tolerance.is_finite()
        || absolute_tolerance < 0.0
    {
        return Err(ElementwiseRuntimePartElevenError::Invalid(
            "isclose tolerances must be finite and nonnegative".to_owned(),
        ));
    }
    let shape = binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?;
    let count = element_count(&shape)?;
    let mut bytes = temporary_vec(backend, execution, count)?;
    for linear in 0..count {
        check_periodically(linear, execution.cancellation)?;
        let output_indices = unravel_index(linear, &shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let other_indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
        let left = read_f32(input, &input_indices)?;
        let right = read_f32(other, &other_indices)?;
        let close = if left == right {
            true
        } else if left.is_nan() || right.is_nan() {
            equal_nan && left.is_nan() && right.is_nan()
        } else {
            (left - right).abs() <= absolute_tolerance + relative_tolerance * right.abs()
        };
        bytes.try_push(u8::from(close))?;
    }
    upload_bytes_mode(
        backend,
        &shape,
        DType::Bool,
        input.descriptor().stream(),
        &bytes,
        execution,
    )
}

pub fn maximum_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, MAXIMUM_OPERATION_ID)?;
    let descriptor = match other {
        ElementwiseOperand::Tensor(other) => {
            require_same_f32_stream(input, other, MAXIMUM_OPERATION_ID)?;
            TensorDescriptor::contiguous(
                binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?,
                DType::F32,
                DeviceId::CPU,
                input.descriptor().stream(),
            )?
        }
        ElementwiseOperand::Scalar(_) => TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            DType::F32,
            DeviceId::CPU,
            input.descriptor().stream(),
        )?,
    };
    Ok(match other {
        ElementwiseOperand::Tensor(other) => {
            backend
                .binary(BinaryOperation::Maximum, input, other, descriptor, context)?
                .0
        }
        ElementwiseOperand::Scalar(value) => {
            backend
                .binary_scalar(
                    BinaryOperation::Maximum,
                    input,
                    value,
                    ScalarSide::Right,
                    descriptor,
                    context,
                )?
                .0
        }
    })
}

pub fn maximum_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<MaximumVjp, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    maximum_vjp_impl(backend, input, other, output_gradient, context)
}

fn maximum_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<MaximumVjp, ElementwiseRuntimePartElevenError> {
    require_f32_cpu(input, MAXIMUM_OPERATION_ID)?;
    require_f32_cpu(output_gradient, MAXIMUM_OPERATION_ID)?;
    let other_tensor = match other {
        ElementwiseOperand::Tensor(other) => Some(other),
        ElementwiseOperand::Scalar(_) => None,
    };
    let output_shape = match other_tensor {
        Some(other) => {
            require_same_f32_stream(input, other, MAXIMUM_OPERATION_ID)?;
            binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?
        }
        None => input.descriptor().shape().to_vec(),
    };
    require_shape(output_gradient, &output_shape, MAXIMUM_OPERATION_ID)?;
    let mut input_gradient = temporary_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    let mut other_gradient = match other_tensor {
        Some(tensor) => Some(temporary_filled(
            backend,
            context,
            element_count(tensor.descriptor().shape())?,
            0.0_f32,
        )?),
        None => None,
    };
    for linear in 0..element_count(&output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let input_linear = ravel_index(&input_indices, input.descriptor().shape())?;
        let left = read_f32(input, &input_indices)?;
        let gradient = read_f32(output_gradient, &output_indices)?;
        let (right, right_linear) = match other {
            ElementwiseOperand::Tensor(other) => {
                let indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
                (
                    read_f32(other, &indices)?,
                    Some(ravel_index(&indices, other.descriptor().shape())?),
                )
            }
            ElementwiseOperand::Scalar(value) => (scalar_f32(value)?, None),
        };
        let (left_scale, right_scale) = extremum_scales(left, right);
        input_gradient[input_linear] += gradient * left_scale;
        if let (Some(position), Some(values)) = (right_linear, other_gradient.as_mut()) {
            values[position] += gradient * right_scale;
        }
    }
    Ok(MaximumVjp {
        input: upload_f32_mode(
            backend,
            input.descriptor().shape(),
            input.descriptor().stream(),
            &input_gradient,
            context,
        )?,
        other: match (other_tensor, other_gradient) {
            (Some(other), Some(values)) => Some(upload_f32_mode(
                backend,
                other.descriptor().shape(),
                other.descriptor().stream(),
                &values,
                context,
            )?),
            _ => None,
        },
    })
}

pub fn maximum_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    maximum_jvp_impl(backend, input, other, input_tangent, other_tangent, context)
}

fn maximum_jvp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    require_same_f32_stream(input, input_tangent, MAXIMUM_OPERATION_ID)?;
    require_shape(
        input_tangent,
        input.descriptor().shape(),
        MAXIMUM_OPERATION_ID,
    )?;
    let output_shape = match other {
        ElementwiseOperand::Tensor(other) => {
            require_same_f32_stream(input, other, MAXIMUM_OPERATION_ID)?;
            let tangent = other_tangent.ok_or_else(|| {
                ElementwiseRuntimePartElevenError::Invalid(
                    "tensor maximum requires an other tangent".to_owned(),
                )
            })?;
            require_same_f32_stream(other, tangent, MAXIMUM_OPERATION_ID)?;
            require_shape(tangent, other.descriptor().shape(), MAXIMUM_OPERATION_ID)?;
            binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?
        }
        ElementwiseOperand::Scalar(_) => {
            if other_tangent.is_some() {
                return Err(ElementwiseRuntimePartElevenError::Invalid(
                    "scalar maximum does not accept an other tangent".to_owned(),
                ));
            }
            input.descriptor().shape().to_vec()
        }
    };
    let mut values = temporary_vec(backend, context, element_count(&output_shape)?)?;
    for linear in 0..element_count(&output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let left = read_f32(input, &input_indices)?;
        let left_tangent = read_f32(input_tangent, &input_indices)?;
        let (right, right_tangent) = match other {
            ElementwiseOperand::Tensor(other) => {
                let indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
                (
                    read_f32(other, &indices)?,
                    read_f32(
                        other_tangent.ok_or_else(|| {
                            ElementwiseRuntimePartElevenError::Invalid(
                                "tensor maximum requires an other tangent".to_owned(),
                            )
                        })?,
                        &indices,
                    )?,
                )
            }
            ElementwiseOperand::Scalar(value) => (scalar_f32(value)?, 0.0),
        };
        let (left_scale, right_scale) = extremum_scales(left, right);
        values.try_push(left_scale * left_tangent + right_scale * right_tangent)?;
    }
    upload_f32_mode(
        backend,
        &output_shape,
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn zeros_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    canonical_zero_in_place_with_context(backend, input, context)?;
    Ok(())
}

pub fn npu_get_device_name_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<String, ElementwiseRuntimePartElevenError> {
    cancellation.check()?;
    Ok(crate::native_device_name_exact(
        capabilities,
        device,
        DeviceKind::Npu,
        NPU_DEVICE_NAME_OPERATION_ID,
        cancellation,
    )?)
}

pub fn round_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    Ok(round_method_with_context_exact_native(
        backend, input, 0, context,
    )?)
}

pub fn round_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    Ok(round_method_vjp_with_context_exact_native(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn round_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    Ok(round_method_jvp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn sign_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, SIGN_OPERATION_ID)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .unary(UnaryOperation::Signum, input, descriptor, context)?
        .0)
}

pub fn sign_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    sign_vjp_impl(backend, input, output_gradient, context)
}

fn sign_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    require_same_f32_stream(input, output_gradient, SIGN_OPERATION_ID)?;
    require_shape(
        output_gradient,
        input.descriptor().shape(),
        SIGN_OPERATION_ID,
    )?;
    let values = temporary_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    upload_f32_mode(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn sign_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    context.cancellation.check()?;
    sign_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

fn build_list(
    input: &Tensor,
    dimension: usize,
    indices: &mut Vec<u64>,
    cancellation: &CancellationToken,
) -> Result<TensorListValue, ElementwiseRuntimePartElevenError> {
    cancellation.check()?;
    if dimension == input.descriptor().rank() {
        return Ok(decoded_list_value(
            input
                .descriptor()
                .dtype()
                .decode_scalar(input.element_bytes(indices)?)?,
        ));
    }
    let length = usize::try_from(input.descriptor().shape()[dimension])
        .map_err(|_| ElementwiseRuntimePartElevenError::ShapeOverflow("tolist dimension"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ElementwiseRuntimePartElevenError::ShapeOverflow("tolist values"))?;
    for index in 0..length {
        check_periodically(index, cancellation)?;
        indices.push(
            u64::try_from(index)
                .map_err(|_| ElementwiseRuntimePartElevenError::ShapeOverflow("tolist index"))?,
        );
        let value = build_list(input, dimension + 1, indices, cancellation)?;
        indices.pop();
        values.push(value);
    }
    Ok(TensorListValue::List(values))
}

fn decoded_list_value(value: DecodedScalar) -> TensorListValue {
    match value {
        DecodedScalar::Boolean(value) => TensorListValue::Boolean(value),
        DecodedScalar::Signed(value) => TensorListValue::Signed(value),
        DecodedScalar::Unsigned(value) => TensorListValue::Unsigned(value),
        DecodedScalar::Real(value) => TensorListValue::Real(value),
        DecodedScalar::Complex { real, imaginary } => TensorListValue::Complex { real, imaginary },
    }
}

fn extremum_scales(left: f32, right: f32) -> (f32, f32) {
    if left > right {
        (1.0, 0.0)
    } else if left < right {
        (0.0, 1.0)
    } else if left == right {
        (0.5, 0.5)
    } else {
        (0.0, 0.0)
    }
}

fn scalar_f32(value: Scalar) -> Result<f32, ElementwiseRuntimePartElevenError> {
    match value {
        Scalar::Float(value) if !value.is_finite() => Ok(value as f32),
        Scalar::Float(value) if (value as f32).is_finite() => Ok(value as f32),
        Scalar::Signed(value) => Ok(value as f32),
        Scalar::Unsigned(value) => Ok(value as f32),
        Scalar::Boolean(value) => Ok(f32::from(u8::from(value))),
        Scalar::Float(_) => Err(ElementwiseRuntimePartElevenError::Invalid(
            "maximum scalar is outside f32 range".to_owned(),
        )),
    }
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartElevenError> {
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ElementwiseRuntimePartElevenError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        });
    }
    Ok(())
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartElevenError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimePartElevenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_same_f32_stream(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartElevenError> {
    require_f32_cpu(input, operation)?;
    require_f32_cpu(other, operation)?;
    if input.descriptor().stream() != other.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: input.descriptor().stream(),
            actual: other.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn require_shape(
    input: &Tensor,
    shape: &[u64],
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartElevenError> {
    if input.descriptor().shape() != shape {
        return Err(ElementwiseRuntimePartElevenError::Invalid(format!(
            "{operation} expected shape {shape:?}, got {:?}",
            input.descriptor().shape()
        )));
    }
    Ok(())
}

fn read_f32(input: &Tensor, indices: &[u64]) -> Result<f32, TensorError> {
    let bytes = input.element_bytes(indices)?;
    let array: [u8; 4] = bytes.try_into().map_err(|_| TensorError::Faulted {
        reason: format!("f32 element exposed {} bytes instead of 4", bytes.len()),
    })?;
    Ok(f32::from_ne_bytes(array))
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartElevenError> {
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(ElementwiseRuntimePartElevenError::ShapeOverflow(
                "element count",
            ))
    })?;
    usize::try_from(count)
        .map_err(|_| ElementwiseRuntimePartElevenError::ShapeOverflow("element count"))
}

fn unravel_index(
    linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartElevenError> {
    let mut remainder = u64::try_from(linear)
        .map_err(|_| ElementwiseRuntimePartElevenError::ShapeOverflow("logical index"))?;
    let mut indices = vec![0; shape.len()];
    for (slot, dimension) in indices.iter_mut().zip(shape).rev() {
        if *dimension == 0 {
            return Err(ElementwiseRuntimePartElevenError::Invalid(
                "cannot unravel an empty dimension".to_owned(),
            ));
        }
        *slot = remainder % *dimension;
        remainder /= *dimension;
    }
    Ok(indices)
}

fn ravel_index(indices: &[u64], shape: &[u64]) -> Result<usize, ElementwiseRuntimePartElevenError> {
    let linear = indices
        .iter()
        .zip(shape)
        .try_fold(0_u64, |linear, (index, dimension)| {
            linear
                .checked_mul(*dimension)
                .and_then(|value| value.checked_add(*index))
                .ok_or(ElementwiseRuntimePartElevenError::ShapeOverflow(
                    "logical offset",
                ))
        })?;
    usize::try_from(linear)
        .map_err(|_| ElementwiseRuntimePartElevenError::ShapeOverflow("logical offset"))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartElevenError> {
    if index & 0x3ff == 0 {
        cancellation.check()?;
    }
    Ok(())
}

fn upload_bytes_mode(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    bytes: &[u8],
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    Ok(backend.upload_bytes(descriptor, bytes, execution)?.0)
}

fn upload_f32_mode(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartElevenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, execution)?.0)
}

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartElevenError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn temporary_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    value: T,
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartElevenError> {
    let mut values = temporary_vec(backend, context, capacity)?;
    for _ in 0..capacity {
        values.try_push(value)?;
    }
    Ok(values)
}
