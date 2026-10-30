use crate::{
    BackendCapabilityMatrix, BinaryOperation, CachedAllocationOwner, CancellationToken, CpuBackend,
    DType, DecodedScalar, DeviceId, ExecutionContext, Scalar, ScalarSide, Tensor, TensorBackend,
    TensorDescriptor, TensorError, UnaryOperation,
    cpu_backend::{CpuWorkspaceVec, binary_broadcast_shape, broadcast_indices},
    generated_sim_operator_indirection_01::OperatorIndirectionError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const DIV_OPERATION_ID: &str = "COMFY-TENSOR-OP-365E27719CFD";
pub const ITEM_OPERATION_ID: &str = "COMFY-TENSOR-OP-3D0519DB53BD";
pub const SIN_OPERATION_ID: &str = "COMFY-TENSOR-OP-332E7E59DC10";
pub const SQRT_OPERATION_ID: &str = "COMFY-TENSOR-OP-3D09997B7D21";
pub const ZERO_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-33CCFBAAA7B3";
pub const BITWISE_LEFT_SHIFT_OPERATION_ID: &str = "COMFY-TENSOR-OP-384A8C6954B8";
pub const COUNT_NONZERO_OPERATION_ID: &str = "COMFY-TENSOR-OP-3ADC7A3998E4";
pub const CUDA_GET_ALLOCATOR_BACKEND_OPERATION_ID: &str = "COMFY-TENSOR-OP-40FEFA2DEAC6";
pub const CUDA_SET_DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-3A3C79159CBC";
pub const MINIMUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-388D285AB0F7";
pub const CONSTANT_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-37266D0A196F";
pub const XPU_GET_DEVICE_NAME_OPERATION_ID: &str = "COMFY-TENSOR-OP-3A641CA3FC0F";

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ElementwiseRuntimePartFiveError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error("elementwise/runtime part-five operation was cancelled")]
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
    #[error("elementwise/runtime part-five input is invalid: {0}")]
    Invalid(&'static str),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartFiveError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
pub struct DivideVjp {
    pub input: Tensor,
    pub other: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct MinimumVjp {
    pub input: Tensor,
    pub other: Option<Tensor>,
}

fn workspace_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
) -> Result<CpuWorkspaceVec<T>, TensorError> {
    backend.workspace_vec(context, capacity)
}

pub fn div_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<DivideVjp, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    div_vjp_impl(backend, input, other, output_gradient, context)
}

pub fn div_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    div_jvp_impl(backend, input, other, input_tangent, other_tangent, context)
}

pub fn bitwise_left_shift_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: BitwiseShiftOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    bitwise_left_shift_impl(backend, input, other, context)
}

pub fn minimum_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<MinimumVjp, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    minimum_vjp_impl(backend, input, other, output_gradient, context)
}

pub fn minimum_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    minimum_jvp_impl(backend, input, other, input_tangent, other_tangent, context)
}

pub fn div_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    require_f32_cpu(input, DIV_OPERATION_ID)?;
    match other {
        ElementwiseOperand::Tensor(other) => {
            require_same_f32_stream(input, other, DIV_OPERATION_ID)?;
            let shape =
                binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?;
            let descriptor = contiguous_output(input, shape, DType::F32)?;
            Ok(backend
                .binary(BinaryOperation::Divide, input, other, descriptor, context)?
                .0)
        }
        ElementwiseOperand::Scalar(other) => {
            let descriptor =
                contiguous_output(input, input.descriptor().shape().to_vec(), DType::F32)?;
            Ok(backend
                .binary_scalar(
                    BinaryOperation::Divide,
                    input,
                    other,
                    ScalarSide::Right,
                    descriptor,
                    context,
                )?
                .0)
        }
    }
}

fn div_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<DivideVjp, ElementwiseRuntimePartFiveError> {
    require_f32_cpu(input, DIV_OPERATION_ID)?;
    require_f32_cpu(output_gradient, DIV_OPERATION_ID)?;
    let (other_tensor, output_shape) = match other {
        ElementwiseOperand::Tensor(other) => {
            require_same_f32_stream(input, other, DIV_OPERATION_ID)?;
            (
                Some(other),
                binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?,
            )
        }
        ElementwiseOperand::Scalar(_) => (None, input.descriptor().shape().to_vec()),
    };
    require_shape(output_gradient, &output_shape, DIV_OPERATION_ID)?;
    let input_count = element_count(input.descriptor().shape())?;
    let mut input_gradient = workspace_vec::<f32>(backend, context, input_count)?;
    for _ in 0..input_count {
        input_gradient.try_push(0.0)?;
    }
    let mut other_gradient = other_tensor
        .map(|tensor| element_count(tensor.descriptor().shape()))
        .transpose()?
        .map(|count| {
            let mut values = workspace_vec::<f32>(backend, context, count)?;
            for _ in 0..count {
                values.try_push(0.0)?;
            }
            Ok::<_, TensorError>(values)
        })
        .transpose()?;
    let output_count = element_count(&output_shape)?;
    for linear_index in 0..output_count {
        check_periodically(linear_index, context.cancellation)?;
        let output_indices = unravel_index(linear_index, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let input_linear = ravel_index(&input_indices, input.descriptor().shape())?;
        let input_value = f32_value(input, &input_indices)?;
        let gradient = f32_value(output_gradient, &output_indices)?;
        let other_value = match other {
            ElementwiseOperand::Tensor(other) => {
                let indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
                f32_value(other, &indices)?
            }
            ElementwiseOperand::Scalar(value) => scalar_f32(value, DIV_OPERATION_ID)?,
        };
        input_gradient[input_linear] += gradient / other_value;
        if let (Some(other), Some(other_gradient)) = (other_tensor, other_gradient.as_mut()) {
            let other_indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
            let other_linear = ravel_index(&other_indices, other.descriptor().shape())?;
            other_gradient[other_linear] += -gradient * input_value / (other_value * other_value);
        }
    }
    context.check()?;
    Ok(DivideVjp {
        input: upload_f32(
            backend,
            input.descriptor().shape(),
            &input_gradient,
            context,
        )?,
        other: match (other_tensor, other_gradient) {
            (Some(other), Some(values)) => Some(upload_f32(
                backend,
                other.descriptor().shape(),
                &values,
                context,
            )?),
            _ => None,
        },
    })
}

fn div_jvp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    require_same_f32_stream(input, input_tangent, DIV_OPERATION_ID)?;
    require_shape(input_tangent, input.descriptor().shape(), DIV_OPERATION_ID)?;
    let output_shape = match other {
        ElementwiseOperand::Tensor(other) => {
            require_same_f32_stream(input, other, DIV_OPERATION_ID)?;
            let tangent = other_tangent.ok_or(ElementwiseRuntimePartFiveError::Invalid(
                "tensor divisor requires a matching tangent",
            ))?;
            require_same_f32_stream(other, tangent, DIV_OPERATION_ID)?;
            require_shape(tangent, other.descriptor().shape(), DIV_OPERATION_ID)?;
            binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?
        }
        ElementwiseOperand::Scalar(_) => {
            if other_tangent.is_some() {
                return Err(ElementwiseRuntimePartFiveError::Invalid(
                    "scalar divisor does not accept a tensor tangent",
                ));
            }
            input.descriptor().shape().to_vec()
        }
    };
    let count = element_count(&output_shape)?;
    let mut values = workspace_vec::<f32>(backend, context, count)?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let output_indices = unravel_index(linear_index, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let input_value = f32_value(input, &input_indices)?;
        let input_tangent_value = f32_value(input_tangent, &input_indices)?;
        let (other_value, other_tangent_value) = match other {
            ElementwiseOperand::Tensor(other) => {
                let other_indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
                (
                    f32_value(other, &other_indices)?,
                    f32_value(
                        other_tangent.ok_or(ElementwiseRuntimePartFiveError::Invalid(
                            "tensor divisor requires a matching tangent",
                        ))?,
                        &other_indices,
                    )?,
                )
            }
            ElementwiseOperand::Scalar(other) => (scalar_f32(other, DIV_OPERATION_ID)?, 0.0),
        };
        values.try_push(
            (input_tangent_value * other_value - input_value * other_tangent_value)
                / (other_value * other_value),
        )?;
    }
    upload_f32(backend, &output_shape, &values, context)
}

pub fn item_with_context_exact_native(
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<DecodedScalar, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    require_cpu(input, ITEM_OPERATION_ID)?;
    if element_count(input.descriptor().shape())? != 1 {
        return Err(ElementwiseRuntimePartFiveError::Invalid(
            "item requires exactly one logical element",
        ));
    }
    let indices = vec![0; input.descriptor().rank()];
    let value = input
        .descriptor()
        .dtype()
        .decode_scalar(input.element_bytes(&indices)?)?;
    context.check()?;
    Ok(value)
}

pub fn sin_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    unary_exact_native(
        backend,
        input,
        UnaryOperation::Sine,
        SIN_OPERATION_ID,
        context,
    )
}

pub fn sin_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    unary_gradient(
        backend,
        input,
        output_gradient,
        SIN_OPERATION_ID,
        |value, gradient| value.cos() * gradient,
        context,
    )
}

pub fn sin_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    sin_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn sqrt_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    unary_exact_native(
        backend,
        input,
        UnaryOperation::SquareRoot,
        SQRT_OPERATION_ID,
        context,
    )
}

pub fn sqrt_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    unary_gradient(
        backend,
        input,
        output_gradient,
        SQRT_OPERATION_ID,
        |value, gradient| gradient / (2.0 * value.sqrt()),
        context,
    )
}

pub fn sqrt_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    sqrt_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn zero_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    fill_in_place(
        backend,
        input,
        Scalar::Signed(0),
        ZERO_IN_PLACE_OPERATION_ID,
        context,
    )
}

#[derive(Clone, Copy, Debug)]
pub enum BitwiseShiftOperand<'a> {
    Tensor(&'a Tensor),
    Scalar(u32),
}

fn bitwise_left_shift_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: BitwiseShiftOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    require_cpu(input, BITWISE_LEFT_SHIFT_OPERATION_ID)?;
    let dtype = input.descriptor().dtype();
    let width = integer_width(dtype).ok_or(ElementwiseRuntimePartFiveError::UnsupportedDType {
        operation: BITWISE_LEFT_SHIFT_OPERATION_ID,
        dtype,
    })?;
    let output_shape = match other {
        BitwiseShiftOperand::Tensor(other) => {
            require_cpu(other, BITWISE_LEFT_SHIFT_OPERATION_ID)?;
            if integer_width(other.descriptor().dtype()).is_none() {
                return Err(ElementwiseRuntimePartFiveError::UnsupportedDType {
                    operation: BITWISE_LEFT_SHIFT_OPERATION_ID,
                    dtype: other.descriptor().dtype(),
                });
            }
            binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?
        }
        BitwiseShiftOperand::Scalar(_) => input.descriptor().shape().to_vec(),
    };
    let count = element_count(&output_shape)?;
    let byte_width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartFiveError::ShapeOverflow("bitwise output"))?;
    let byte_count =
        count
            .checked_mul(byte_width)
            .ok_or(ElementwiseRuntimePartFiveError::ShapeOverflow(
                "bitwise output",
            ))?;
    let mut bytes = workspace_vec::<u8>(backend, context, byte_count)?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let output_indices = unravel_index(linear_index, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&input_indices)?)?;
        let shift = match other {
            BitwiseShiftOperand::Tensor(other) => {
                let indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
                decoded_shift(
                    other
                        .descriptor()
                        .dtype()
                        .decode_scalar(other.element_bytes(&indices)?)?,
                )?
            }
            BitwiseShiftOperand::Scalar(shift) => shift,
        };
        if shift >= width {
            return Err(ElementwiseRuntimePartFiveError::Invalid(
                "left shift must be smaller than the input dtype width",
            ));
        }
        let encoded = shifted_scalar(dtype, value, shift)?;
        for byte in dtype.encode_scalar(encoded, BITWISE_LEFT_SHIFT_OPERATION_ID, DeviceId::CPU)? {
            bytes.try_push(byte)?;
        }
    }
    upload_bytes(backend, &output_shape, dtype, input, &bytes, context)
}

pub fn count_nonzero_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    require_cpu(input, COUNT_NONZERO_OPERATION_ID)?;
    let count = element_count(input.descriptor().shape())?;
    let mut nonzero = 0_u64;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let indices = unravel_index(linear_index, input.descriptor().shape())?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        if value.is_nonzero() {
            nonzero =
                nonzero
                    .checked_add(1)
                    .ok_or(ElementwiseRuntimePartFiveError::ShapeOverflow(
                        "nonzero count",
                    ))?;
        }
    }
    let nonzero = i64::try_from(nonzero)
        .map_err(|_| ElementwiseRuntimePartFiveError::ShapeOverflow("nonzero count"))?;
    let descriptor = TensorDescriptor::contiguous(
        Vec::new(),
        DType::I64,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .fill(Scalar::Signed(nonzero), descriptor, context)?
        .0)
}

pub fn cuda_get_allocator_backend_exact_native(
    owner: &dyn CachedAllocationOwner,
    cancellation: &CancellationToken,
) -> Result<&'static str, ElementwiseRuntimePartFiveError> {
    cancellation.check()?;
    let device = owner.cache_device();
    if !matches!(device.kind(), DeviceKind::Cuda | DeviceKind::Rocm) {
        return Err(ElementwiseRuntimePartFiveError::UnsupportedDevice {
            operation: CUDA_GET_ALLOCATOR_BACKEND_OPERATION_ID,
            device,
        });
    }
    let name = owner.allocator_backend_name();
    if name.is_empty() || name.len() > 128 || name.contains('\0') {
        return Err(ElementwiseRuntimePartFiveError::Invalid(
            "allocator backend name must contain 1..=128 non-NUL bytes",
        ));
    }
    cancellation.check()?;
    Ok(name)
}

pub fn cuda_set_device_exact_native<'a>(
    available: &'a [BackendCapabilityMatrix],
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<&'a BackendCapabilityMatrix, ElementwiseRuntimePartFiveError> {
    cancellation.check()?;
    Ok(crate::native_select_device_exact(
        available,
        device,
        &[DeviceKind::Cuda, DeviceKind::Rocm],
        CUDA_SET_DEVICE_OPERATION_ID,
        cancellation,
    )?)
}

pub fn minimum_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    require_f32_cpu(input, MINIMUM_OPERATION_ID)?;
    match other {
        ElementwiseOperand::Tensor(other) => {
            require_same_f32_stream(input, other, MINIMUM_OPERATION_ID)?;
            let shape =
                binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?;
            let descriptor = contiguous_output(input, shape, DType::F32)?;
            Ok(backend
                .binary(BinaryOperation::Minimum, input, other, descriptor, context)?
                .0)
        }
        ElementwiseOperand::Scalar(other) => {
            let descriptor =
                contiguous_output(input, input.descriptor().shape().to_vec(), DType::F32)?;
            Ok(backend
                .binary_scalar(
                    BinaryOperation::Minimum,
                    input,
                    other,
                    ScalarSide::Right,
                    descriptor,
                    context,
                )?
                .0)
        }
    }
}

fn minimum_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<MinimumVjp, ElementwiseRuntimePartFiveError> {
    require_f32_cpu(input, MINIMUM_OPERATION_ID)?;
    require_f32_cpu(output_gradient, MINIMUM_OPERATION_ID)?;
    let (other_tensor, output_shape) = match other {
        ElementwiseOperand::Tensor(other) => {
            require_same_f32_stream(input, other, MINIMUM_OPERATION_ID)?;
            (
                Some(other),
                binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?,
            )
        }
        ElementwiseOperand::Scalar(_) => (None, input.descriptor().shape().to_vec()),
    };
    require_shape(output_gradient, &output_shape, MINIMUM_OPERATION_ID)?;
    let input_count = element_count(input.descriptor().shape())?;
    let mut input_gradient = workspace_vec::<f32>(backend, context, input_count)?;
    for _ in 0..input_count {
        input_gradient.try_push(0.0)?;
    }
    let mut other_gradient = other_tensor
        .map(|tensor| element_count(tensor.descriptor().shape()))
        .transpose()?
        .map(|count| {
            let mut values = workspace_vec::<f32>(backend, context, count)?;
            for _ in 0..count {
                values.try_push(0.0)?;
            }
            Ok::<_, TensorError>(values)
        })
        .transpose()?;
    for linear_index in 0..element_count(&output_shape)? {
        check_periodically(linear_index, context.cancellation)?;
        let output_indices = unravel_index(linear_index, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let input_linear = ravel_index(&input_indices, input.descriptor().shape())?;
        let input_value = f32_value(input, &input_indices)?;
        let gradient = f32_value(output_gradient, &output_indices)?;
        let (other_value, other_position) = match other {
            ElementwiseOperand::Tensor(other) => {
                let indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
                let linear = ravel_index(&indices, other.descriptor().shape())?;
                (f32_value(other, &indices)?, Some(linear))
            }
            ElementwiseOperand::Scalar(value) => (scalar_f32(value, MINIMUM_OPERATION_ID)?, None),
        };
        let (input_scale, other_scale) = if input_value < other_value {
            (1.0, 0.0)
        } else if input_value > other_value {
            (0.0, 1.0)
        } else if input_value == other_value {
            (0.5, 0.5)
        } else {
            (0.0, 0.0)
        };
        input_gradient[input_linear] += gradient * input_scale;
        if let (Some(position), Some(other_gradient)) = (other_position, other_gradient.as_mut()) {
            other_gradient[position] += gradient * other_scale;
        }
    }
    context.check()?;
    Ok(MinimumVjp {
        input: upload_f32(
            backend,
            input.descriptor().shape(),
            &input_gradient,
            context,
        )?,
        other: match (other_tensor, other_gradient) {
            (Some(other), Some(values)) => Some(upload_f32(
                backend,
                other.descriptor().shape(),
                &values,
                context,
            )?),
            _ => None,
        },
    })
}

fn minimum_jvp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    require_same_f32_stream(input, input_tangent, MINIMUM_OPERATION_ID)?;
    require_shape(
        input_tangent,
        input.descriptor().shape(),
        MINIMUM_OPERATION_ID,
    )?;
    let output_shape = match other {
        ElementwiseOperand::Tensor(other) => {
            require_same_f32_stream(input, other, MINIMUM_OPERATION_ID)?;
            let tangent = other_tangent.ok_or(ElementwiseRuntimePartFiveError::Invalid(
                "tensor minimum operand requires a matching tangent",
            ))?;
            require_same_f32_stream(other, tangent, MINIMUM_OPERATION_ID)?;
            require_shape(tangent, other.descriptor().shape(), MINIMUM_OPERATION_ID)?;
            binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?
        }
        ElementwiseOperand::Scalar(_) => {
            if other_tangent.is_some() {
                return Err(ElementwiseRuntimePartFiveError::Invalid(
                    "scalar minimum operand does not accept a tensor tangent",
                ));
            }
            input.descriptor().shape().to_vec()
        }
    };
    let count = element_count(&output_shape)?;
    let mut values = workspace_vec::<f32>(backend, context, count)?;
    for linear_index in 0..count {
        check_periodically(linear_index, context.cancellation)?;
        let output_indices = unravel_index(linear_index, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let input_value = f32_value(input, &input_indices)?;
        let input_tangent_value = f32_value(input_tangent, &input_indices)?;
        let (other_value, other_tangent_value) = match other {
            ElementwiseOperand::Tensor(other) => {
                let other_indices = broadcast_indices(&output_indices, other.descriptor().shape())?;
                (
                    f32_value(other, &other_indices)?,
                    f32_value(
                        other_tangent.ok_or(ElementwiseRuntimePartFiveError::Invalid(
                            "tensor minimum operand requires a matching tangent",
                        ))?,
                        &other_indices,
                    )?,
                )
            }
            ElementwiseOperand::Scalar(other) => (scalar_f32(other, MINIMUM_OPERATION_ID)?, 0.0),
        };
        values.try_push(if input_value < other_value {
            input_tangent_value
        } else if input_value > other_value {
            other_tangent_value
        } else if input_value == other_value {
            0.5 * (input_tangent_value + other_tangent_value)
        } else {
            0.0
        })?;
    }
    upload_f32(backend, &output_shape, &values, context)
}

pub fn constant_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    value: Scalar,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartFiveError> {
    context.cancellation.check()?;
    fill_in_place(
        backend,
        input,
        value,
        CONSTANT_IN_PLACE_OPERATION_ID,
        context,
    )
}

pub fn xpu_get_device_name_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<String, ElementwiseRuntimePartFiveError> {
    cancellation.check()?;
    Ok(crate::native_device_name_exact(
        capabilities,
        device,
        DeviceKind::Xpu,
        XPU_GET_DEVICE_NAME_OPERATION_ID,
        cancellation,
    )?)
}

fn unary_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    operation: UnaryOperation,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    require_f32_cpu(input, operation_id)?;
    let descriptor = contiguous_output(input, input.descriptor().shape().to_vec(), DType::F32)?;
    Ok(backend.unary(operation, input, descriptor, context)?.0)
}

fn unary_gradient(
    backend: &CpuBackend,
    input: &Tensor,
    gradient: &Tensor,
    operation: &'static str,
    derivative: impl Fn(f32, f32) -> f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    require_same_f32_stream(input, gradient, operation)?;
    require_shape(gradient, input.descriptor().shape(), operation)?;
    let count = element_count(input.descriptor().shape())?;
    let mut values = workspace_vec::<f32>(backend, context, count)?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        let indices = unravel_index(index, input.descriptor().shape())?;
        values.try_push(derivative(
            f32_value(input, &indices)?,
            f32_value(gradient, &indices)?,
        ))?;
    }
    upload_f32(backend, input.descriptor().shape(), &values, context)
}

fn fill_in_place(
    backend: &CpuBackend,
    input: &mut Tensor,
    value: Scalar,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartFiveError> {
    require_cpu(input, operation)?;
    context.check()?;
    input
        .descriptor()
        .dtype()
        .encode_scalar(value, operation, DeviceId::CPU)?;
    let descriptor = input.descriptor().clone();
    let (staged, _) = backend.fill(value, descriptor, context)?;
    context.check()?;
    input.commit_in_place(staged)?;
    Ok(())
}

fn contiguous_output(
    input: &Tensor,
    shape: Vec<u64>,
    dtype: DType,
) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, input.descriptor().stream())
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFiveError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartFiveError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFiveError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartFiveError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}

fn require_same_f32_stream(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFiveError> {
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
    tensor: &Tensor,
    expected: &[u64],
    _operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFiveError> {
    if tensor.descriptor().shape() != expected {
        return Err(ElementwiseRuntimePartFiveError::Invalid(
            "gradient shape does not match the operation output",
        ));
    }
    Ok(())
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_bytes(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    reference: &Tensor,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFiveError> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        dtype,
        DeviceId::CPU,
        reference.descriptor().stream(),
    )?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn scalar_f32(
    value: Scalar,
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartFiveError> {
    let encoded = DType::F32.encode_scalar(value, operation, DeviceId::CPU)?;
    match DType::F32.decode_scalar(&encoded)? {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(ElementwiseRuntimePartFiveError::Invalid(
            "f32 scalar decoding produced a non-real value",
        )),
    }
}

fn f32_value(tensor: &Tensor, indices: &[u64]) -> Result<f32, ElementwiseRuntimePartFiveError> {
    match DType::F32.decode_scalar(tensor.element_bytes(indices)?)? {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(ElementwiseRuntimePartFiveError::Invalid(
            "f32 tensor decoding produced a non-real value",
        )),
    }
}

fn integer_width(dtype: DType) -> Option<u32> {
    match dtype {
        DType::I8 | DType::U8 => Some(8),
        DType::I16 | DType::U16 => Some(16),
        DType::I32 | DType::U32 => Some(32),
        DType::I64 | DType::U64 => Some(64),
        _ => None,
    }
}

fn decoded_shift(value: DecodedScalar) -> Result<u32, ElementwiseRuntimePartFiveError> {
    match value {
        DecodedScalar::Signed(value) => u32::try_from(value).map_err(|_| {
            ElementwiseRuntimePartFiveError::Invalid("left shift must be non-negative")
        }),
        DecodedScalar::Unsigned(value) => u32::try_from(value)
            .map_err(|_| ElementwiseRuntimePartFiveError::Invalid("left shift does not fit u32")),
        _ => Err(ElementwiseRuntimePartFiveError::Invalid(
            "left shift operand must be an integer",
        )),
    }
}

fn shifted_scalar(
    dtype: DType,
    value: DecodedScalar,
    shift: u32,
) -> Result<Scalar, ElementwiseRuntimePartFiveError> {
    let width = integer_width(dtype).ok_or(ElementwiseRuntimePartFiveError::UnsupportedDType {
        operation: BITWISE_LEFT_SHIFT_OPERATION_ID,
        dtype,
    })?;
    let mask = (1_u128 << width) - 1;
    let bits = match value {
        DecodedScalar::Signed(value) => (value as i128 as u128) & mask,
        DecodedScalar::Unsigned(value) => u128::from(value) & mask,
        _ => {
            return Err(ElementwiseRuntimePartFiveError::Invalid(
                "left shift input must be an integer",
            ));
        }
    };
    let shifted = (bits << shift) & mask;
    if matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
        let sign_bit = 1_u128 << (width - 1);
        let signed = if shifted & sign_bit == 0 {
            i128::try_from(shifted).map_err(|_| {
                ElementwiseRuntimePartFiveError::Invalid("shifted value exceeds i128")
            })?
        } else {
            i128::try_from(shifted).map_err(|_| {
                ElementwiseRuntimePartFiveError::Invalid("shifted value exceeds i128")
            })? - (1_i128 << width)
        };
        Ok(Scalar::Signed(i64::try_from(signed).map_err(|_| {
            ElementwiseRuntimePartFiveError::Invalid("shifted value exceeds i64")
        })?))
    } else {
        Ok(Scalar::Unsigned(u64::try_from(shifted).map_err(|_| {
            ElementwiseRuntimePartFiveError::Invalid("shifted value exceeds u64")
        })?))
    }
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartFiveError> {
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(ElementwiseRuntimePartFiveError::ShapeOverflow(
                "logical element count",
            ))
    })?;
    usize::try_from(count)
        .map_err(|_| ElementwiseRuntimePartFiveError::ShapeOverflow("logical element count"))
}

fn unravel_index(
    mut linear_index: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartFiveError> {
    let mut indices = vec![0; shape.len()];
    for dimension_index in (0..shape.len()).rev() {
        let dimension = usize::try_from(shape[dimension_index]).map_err(|_| {
            ElementwiseRuntimePartFiveError::ShapeOverflow("logical tensor dimension")
        })?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartFiveError::Invalid(
                "zero-sized shape has no logical indices",
            ));
        }
        indices[dimension_index] = u64::try_from(linear_index % dimension)
            .map_err(|_| ElementwiseRuntimePartFiveError::ShapeOverflow("logical tensor index"))?;
        linear_index /= dimension;
    }
    Ok(indices)
}

fn ravel_index(indices: &[u64], shape: &[u64]) -> Result<usize, ElementwiseRuntimePartFiveError> {
    if indices.len() != shape.len() {
        return Err(ElementwiseRuntimePartFiveError::Invalid(
            "logical index rank does not match shape",
        ));
    }
    let mut linear = 0_u64;
    for (&index, &dimension) in indices.iter().zip(shape) {
        if index >= dimension {
            return Err(ElementwiseRuntimePartFiveError::Invalid(
                "logical index exceeds shape",
            ));
        }
        linear = linear
            .checked_mul(dimension)
            .and_then(|value| value.checked_add(index))
            .ok_or(ElementwiseRuntimePartFiveError::ShapeOverflow(
                "logical linear index",
            ))?;
    }
    usize::try_from(linear)
        .map_err(|_| ElementwiseRuntimePartFiveError::ShapeOverflow("logical linear index"))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartFiveError> {
    if index.is_multiple_of(1_024) {
        cancellation.check()?;
    }
    Ok(())
}
