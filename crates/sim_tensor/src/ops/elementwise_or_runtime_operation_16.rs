use crate::{
    BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar,
    DeviceId, ExecutionContext, Scalar, ScalarSide, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError, UnaryOperation,
    cpu_backend::{binary_broadcast_shape, broadcast_indices},
    generated_sim_operator_indirection_01::OperatorIndirectionError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_08::{
        ElementwiseRuntimePartEightError,
        square_jvp_with_context_exact_native as canonical_square_jvp_with_context,
        square_vjp_with_context_exact_native as canonical_square_vjp_with_context,
        square_with_context_exact_native as canonical_square_with_context,
    },
    generated_elementwise_or_runtime_operation_09::{
        ElementwiseRuntimePartNineError, NativeBitwiseOperation,
        bitwise_binary_with_context_exact_native as canonical_bitwise_binary_with_context,
        mul_jvp_with_context_exact_native as canonical_mul_jvp_with_context,
        mul_vjp_with_context_exact_native as canonical_mul_vjp_with_context,
        mul_with_context_exact_native as canonical_mul_with_context,
    },
};
use thiserror::Error;

pub const ADD_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-B74E6E64A97F";
pub const MUL_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-B82E0C11E45D";
pub const SQUARE_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-B40398C137FF";
pub const ATAN_OPERATION_ID: &str = "COMFY-TENSOR-OP-B4F8F3B2B2E6";
pub const BITWISE_AND_OPERATION_ID: &str = "COMFY-TENSOR-OP-B4B7266D14A9";
pub const CUDART_OPERATION_ID: &str = "COMFY-TENSOR-OP-B153098D5C48";
pub const IS_TENSOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-B1A79A94DDE6";
pub const KAISER_WINDOW_OPERATION_ID: &str = "COMFY-TENSOR-OP-B296530D4BB3";
pub const LOGADDEXP_OPERATION_ID: &str = "COMFY-TENSOR-OP-B7955A0A7AC9";
pub const MLU_EMPTY_CACHE_OPERATION_ID: &str = "COMFY-TENSOR-OP-B2699A727A6C";
pub const ADD_SAFE_GLOBALS_OPERATION_ID: &str = "COMFY-TENSOR-OP-B30CBD7D8727";
pub const TILE_OPERATION_ID: &str = "COMFY-TENSOR-OP-B088976A05AB";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartSixteenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Cast(#[from] OperatorIndirectionError),
    #[error(transparent)]
    PartEight(#[from] ElementwiseRuntimePartEightError),
    #[error(transparent)]
    PartNine(#[from] ElementwiseRuntimePartNineError),
    #[error("elementwise/runtime part-sixteen execution was cancelled")]
    Cancelled,
    #[error("operation {operation} requires CPU ordinal zero, got {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} does not support dtype {dtype:?}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartSixteenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
pub struct AddGradients {
    pub input: Tensor,
    pub other: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct LogAddExpGradients {
    pub left: Tensor,
    pub right: Tensor,
}

type TemporaryVec<T> = CpuWorkspaceVec<T>;

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    _label: &'static str,
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartSixteenError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

pub fn add_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, ADD_METHOD_OPERATION_ID)?;
    match other {
        ElementwiseOperand::Tensor(other) => {
            require_binary_f32(input, other, ADD_METHOD_OPERATION_ID)?;
            let shape =
                binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?;
            let scaled_descriptor = TensorDescriptor::contiguous(
                other.descriptor().shape().to_vec(),
                DType::F32,
                DeviceId::CPU,
                input.descriptor().stream(),
            )?;
            let scaled = backend
                .binary_scalar(
                    BinaryOperation::Multiply,
                    other,
                    Scalar::Float(f64::from(alpha)),
                    ScalarSide::Right,
                    scaled_descriptor,
                    context,
                )?
                .0;
            binary_forward(
                backend,
                BinaryOperation::Add,
                input,
                &scaled,
                shape,
                context,
            )
        }
        ElementwiseOperand::Scalar(value) => {
            let scalar = scalar_as_f64(value) * f64::from(alpha);
            let descriptor = TensorDescriptor::contiguous(
                input.descriptor().shape().to_vec(),
                DType::F32,
                DeviceId::CPU,
                input.descriptor().stream(),
            )?;
            Ok(backend
                .binary_scalar(
                    BinaryOperation::Add,
                    input,
                    Scalar::Float(scalar),
                    ScalarSide::Right,
                    descriptor,
                    context,
                )?
                .0)
        }
    }
}

fn add_method_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    alpha: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<AddGradients, ElementwiseRuntimePartSixteenError> {
    require_f32_cpu(input, ADD_METHOD_OPERATION_ID)?;
    require_f32_cpu(output_gradient, ADD_METHOD_OPERATION_ID)?;
    let output_shape = match other {
        ElementwiseOperand::Tensor(other) => {
            require_binary_f32(input, other, ADD_METHOD_OPERATION_ID)?;
            binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?
        }
        ElementwiseOperand::Scalar(_) => input.descriptor().shape().to_vec(),
    };
    require_shape_stream(
        output_gradient,
        &output_shape,
        input.descriptor().stream(),
        ADD_METHOD_OPERATION_ID,
    )?;
    let input_gradient = reduce_broadcast_f32(
        backend,
        output_gradient,
        input.descriptor().shape(),
        1.0,
        ADD_METHOD_OPERATION_ID,
        context,
    )?;
    let other_gradient = match other {
        ElementwiseOperand::Tensor(other) => Some(reduce_broadcast_f32(
            backend,
            output_gradient,
            other.descriptor().shape(),
            alpha,
            ADD_METHOD_OPERATION_ID,
            context,
        )?),
        ElementwiseOperand::Scalar(_) => None,
    };
    Ok(AddGradients {
        input: input_gradient,
        other: other_gradient,
    })
}

pub fn add_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: ElementwiseOperand<'_>,
    alpha: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<AddGradients, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    add_method_vjp_impl(backend, input, other, alpha, output_gradient, context)
}

pub fn add_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    other_tangent: Option<&Tensor>,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    match other_tangent {
        Some(other_tangent) => add_method_with_context_exact_native(
            backend,
            input_tangent,
            ElementwiseOperand::Tensor(other_tangent),
            alpha,
            context,
        ),
        None => add_method_with_context_exact_native(
            backend,
            input_tangent,
            ElementwiseOperand::Scalar(Scalar::Float(0.0)),
            alpha,
            context,
        ),
    }
}

pub fn mul_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    Ok(canonical_mul_with_context(backend, input, other, context)?)
}

pub fn mul_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<
    crate::generated_elementwise_or_runtime_operation_09::BinaryGradients,
    ElementwiseRuntimePartSixteenError,
> {
    context.cancellation.check()?;
    Ok(canonical_mul_vjp_with_context(
        backend,
        input,
        other,
        output_gradient,
        context,
    )?)
}

pub fn mul_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    Ok(canonical_mul_jvp_with_context(
        backend,
        input,
        other,
        input_tangent,
        other_tangent,
        context,
    )?)
}

pub fn square_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    Ok(canonical_square_with_context(backend, input, context)?)
}

pub fn square_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    Ok(canonical_square_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn square_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    Ok(canonical_square_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn atan_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    unary_forward(
        backend,
        UnaryOperation::ArcTangent,
        input,
        ATAN_OPERATION_ID,
        context,
    )
}

pub fn atan_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    unary_derivative(
        backend,
        input,
        output_gradient,
        ATAN_OPERATION_ID,
        |value, gradient| gradient / (1.0 + value * value),
        context,
    )
}

pub fn atan_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    unary_derivative(
        backend,
        input,
        input_tangent,
        ATAN_OPERATION_ID,
        |value, gradient| gradient / (1.0 + value * value),
        context,
    )
}

pub fn bitwise_and_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    Ok(canonical_bitwise_binary_with_context(
        backend,
        left,
        right,
        NativeBitwiseOperation::And,
        BITWISE_AND_OPERATION_ID,
        context,
    )?)
}

pub fn is_tensor_exact_native(
    input: Option<&Tensor>,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartSixteenError> {
    cancellation.check()?;
    Ok(input.is_some())
}

pub fn kaiser_window_with_context_exact_native(
    backend: &CpuBackend,
    length: u64,
    periodic: bool,
    beta: f64,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    kaiser_window_impl(
        backend,
        length,
        periodic,
        beta,
        dtype,
        context.stream,
        context,
    )
}

fn kaiser_window_impl(
    backend: &CpuBackend,
    length: u64,
    periodic: bool,
    beta: f64,
    dtype: DType,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(ElementwiseRuntimePartSixteenError::UnsupportedDType {
            operation: KAISER_WINDOW_OPERATION_ID,
            dtype,
        });
    }
    if !beta.is_finite() || beta < 0.0 {
        return invalid(
            KAISER_WINDOW_OPERATION_ID,
            "beta must be finite and non-negative",
        );
    }
    let count = usize::try_from(length)
        .map_err(|_| ElementwiseRuntimePartSixteenError::ShapeOverflow("Kaiser length"))?;
    let denominator = if periodic {
        length
    } else {
        length.saturating_sub(1)
    };
    let normalization = modified_bessel_i0(beta);
    let width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartSixteenError::ShapeOverflow("Kaiser dtype width"))?;
    let byte_count =
        count
            .checked_mul(width)
            .ok_or(ElementwiseRuntimePartSixteenError::ShapeOverflow(
                "Kaiser result bytes",
            ))?;
    let mut bytes = temporary_vec(backend, context, byte_count, "Kaiser result bytes")?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        let value = if length <= 1 {
            1.0
        } else {
            let ratio = 2.0 * index as f64 / denominator as f64 - 1.0;
            modified_bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / normalization
        };
        for byte in dtype.encode_scalar(
            Scalar::Float(value),
            KAISER_WINDOW_OPERATION_ID,
            DeviceId::CPU,
        )? {
            bytes.try_push(byte)?;
        }
    }
    upload_bytes(backend, &[length], dtype, stream, &bytes, context)
}

pub fn logaddexp_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    require_binary_f32(left, right, LOGADDEXP_OPERATION_ID)?;
    let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    binary_forward(
        backend,
        BinaryOperation::LogAddExp,
        left,
        right,
        shape,
        context,
    )
}

pub fn logaddexp_vjp_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<LogAddExpGradients, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    logaddexp_vjp_impl(backend, left, right, output_gradient, context)
}

fn logaddexp_vjp_impl(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<LogAddExpGradients, ElementwiseRuntimePartSixteenError> {
    require_binary_f32(left, right, LOGADDEXP_OPERATION_ID)?;
    require_f32_cpu(output_gradient, LOGADDEXP_OPERATION_ID)?;
    let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    require_shape_stream(
        output_gradient,
        &shape,
        left.descriptor().stream(),
        LOGADDEXP_OPERATION_ID,
    )?;
    let mut left_values = zero_f32_values(
        backend,
        context,
        left.descriptor().shape(),
        "logaddexp left gradient",
    )?;
    let mut right_values = zero_f32_values(
        backend,
        context,
        right.descriptor().shape(),
        "logaddexp right gradient",
    )?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let left_indices = broadcast_indices(&indices, left.descriptor().shape())?;
        let right_indices = broadcast_indices(&indices, right.descriptor().shape())?;
        let left_value = read_f32(left, &left_indices, LOGADDEXP_OPERATION_ID)?;
        let right_value = read_f32(right, &right_indices, LOGADDEXP_OPERATION_ID)?;
        let gradient = read_f32(output_gradient, &indices, LOGADDEXP_OPERATION_ID)?;
        let (left_weight, right_weight) = logaddexp_weights(left_value, right_value);
        let left_linear = ravel_index(&left_indices, left.descriptor().shape())?;
        let right_linear = ravel_index(&right_indices, right.descriptor().shape())?;
        left_values[left_linear] += gradient * left_weight;
        right_values[right_linear] += gradient * right_weight;
    }
    Ok(LogAddExpGradients {
        left: upload_f32(
            backend,
            left.descriptor().shape(),
            left.descriptor().stream(),
            &left_values,
            context,
        )?,
        right: upload_f32(
            backend,
            right.descriptor().shape(),
            right.descriptor().stream(),
            &right_values,
            context,
        )?,
    })
}

pub fn logaddexp_jvp_with_context_exact_native(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    left_tangent: &Tensor,
    right_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    logaddexp_jvp_impl(backend, left, right, left_tangent, right_tangent, context)
}

fn logaddexp_jvp_impl(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    left_tangent: &Tensor,
    right_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    require_binary_f32(left, right, LOGADDEXP_OPERATION_ID)?;
    require_binary_f32(left_tangent, right_tangent, LOGADDEXP_OPERATION_ID)?;
    require_shape_stream(
        left_tangent,
        left.descriptor().shape(),
        left.descriptor().stream(),
        LOGADDEXP_OPERATION_ID,
    )?;
    require_shape_stream(
        right_tangent,
        right.descriptor().shape(),
        right.descriptor().stream(),
        LOGADDEXP_OPERATION_ID,
    )?;
    let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    let mut values = temporary_vec(backend, context, element_count(&shape)?, "logaddexp JVP")?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let left_indices = broadcast_indices(&indices, left.descriptor().shape())?;
        let right_indices = broadcast_indices(&indices, right.descriptor().shape())?;
        let (left_weight, right_weight) = logaddexp_weights(
            read_f32(left, &left_indices, LOGADDEXP_OPERATION_ID)?,
            read_f32(right, &right_indices, LOGADDEXP_OPERATION_ID)?,
        );
        values.try_push(
            left_weight * read_f32(left_tangent, &left_indices, LOGADDEXP_OPERATION_ID)?
                + right_weight * read_f32(right_tangent, &right_indices, LOGADDEXP_OPERATION_ID)?,
        )?;
    }
    upload_f32(
        backend,
        &shape,
        left.descriptor().stream(),
        &values,
        context,
    )
}

pub fn tile_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    tile_impl(backend, input, repeats, context)
}

fn tile_impl(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    require_cpu(input, TILE_OPERATION_ID)?;
    if repeats.iter().any(|repeat| *repeat < 0) {
        return invalid(TILE_OPERATION_ID, "repeat counts must be non-negative");
    }
    let rank = input.descriptor().rank().max(repeats.len());
    let input_padding = rank.saturating_sub(input.descriptor().rank());
    let repeat_padding = rank.saturating_sub(repeats.len());
    let mut input_shape = vec![1_u64; input_padding];
    input_shape.extend_from_slice(input.descriptor().shape());
    let mut normalized_repeats = vec![1_u64; repeat_padding];
    for repeat in repeats {
        normalized_repeats.push(u64::try_from(*repeat).map_err(|_| {
            ElementwiseRuntimePartSixteenError::ShapeOverflow("tile repeat conversion")
        })?);
    }
    let output_shape =
        input_shape
            .iter()
            .zip(&normalized_repeats)
            .map(|(dimension, repeat)| {
                dimension.checked_mul(*repeat).ok_or(
                    ElementwiseRuntimePartSixteenError::ShapeOverflow("tile output shape"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
    let count = element_count(&output_shape)?;
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| ElementwiseRuntimePartSixteenError::ShapeOverflow("tile dtype width"))?;
    let byte_count =
        count
            .checked_mul(width)
            .ok_or(ElementwiseRuntimePartSixteenError::ShapeOverflow(
                "tile result bytes",
            ))?;
    let mut bytes = temporary_vec(backend, context, byte_count, "tile result bytes")?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let padded_indices = output_indices
            .iter()
            .zip(&input_shape)
            .map(|(index, dimension)| {
                if *dimension == 0 {
                    0
                } else {
                    index % dimension
                }
            })
            .collect::<Vec<_>>();
        let source_indices = padded_indices.get(input_padding..).ok_or(
            ElementwiseRuntimePartSixteenError::ShapeOverflow("tile source indices"),
        )?;
        for byte in input.element_bytes(source_indices)? {
            bytes.try_push(*byte)?;
        }
    }
    upload_bytes(
        backend,
        &output_shape,
        input.descriptor().dtype(),
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn tile_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: &[i64],
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    tile_vjp_impl(backend, input, repeats, output_gradient, context)
}

fn tile_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: &[i64],
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    require_f32_cpu(input, TILE_OPERATION_ID)?;
    require_f32_cpu(output_gradient, TILE_OPERATION_ID)?;
    let tiled = tile_impl(backend, input, repeats, context)?;
    require_shape_stream(
        output_gradient,
        tiled.descriptor().shape(),
        input.descriptor().stream(),
        TILE_OPERATION_ID,
    )?;
    let rank = tiled.descriptor().rank();
    let input_padding = rank.saturating_sub(input.descriptor().rank());
    let mut input_shape = vec![1_u64; input_padding];
    input_shape.extend_from_slice(input.descriptor().shape());
    let mut values = zero_f32_values(
        backend,
        context,
        input.descriptor().shape(),
        "tile gradient",
    )?;
    for linear in 0..element_count(tiled.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, tiled.descriptor().shape())?;
        let padded_indices = output_indices
            .iter()
            .zip(&input_shape)
            .map(|(index, dimension)| {
                if *dimension == 0 {
                    0
                } else {
                    index % dimension
                }
            })
            .collect::<Vec<_>>();
        let source_indices = padded_indices.get(input_padding..).ok_or(
            ElementwiseRuntimePartSixteenError::ShapeOverflow("tile gradient indices"),
        )?;
        let source_linear = ravel_index(source_indices, input.descriptor().shape())?;
        values[source_linear] += read_f32(output_gradient, &output_indices, TILE_OPERATION_ID)?;
    }
    upload_f32(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn tile_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    repeats: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    context.cancellation.check()?;
    require_shape_stream(
        input_tangent,
        input.descriptor().shape(),
        input.descriptor().stream(),
        TILE_OPERATION_ID,
    )?;
    tile_with_context_exact_native(backend, input_tangent, repeats, context)
}

fn unary_forward(
    backend: &CpuBackend,
    operation: UnaryOperation,
    input: &Tensor,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    require_f32_cpu(input, operation_id)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.unary(operation, input, descriptor, context)?.0)
}

fn binary_forward(
    backend: &CpuBackend,
    operation: BinaryOperation,
    left: &Tensor,
    right: &Tensor,
    shape: Vec<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, left.descriptor().stream())?;
    Ok(backend
        .binary(operation, left, right, descriptor, context)?
        .0)
}

fn unary_derivative(
    backend: &CpuBackend,
    input: &Tensor,
    gradient: &Tensor,
    operation: &'static str,
    derivative: impl Fn(f32, f32) -> f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    require_f32_cpu(input, operation)?;
    require_f32_cpu(gradient, operation)?;
    require_shape_stream(
        gradient,
        input.descriptor().shape(),
        input.descriptor().stream(),
        operation,
    )?;
    let mut values = temporary_vec(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        "unary derivative",
    )?;
    for linear in 0..element_count(input.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        values.try_push(derivative(
            read_f32(input, &indices, operation)?,
            read_f32(gradient, &indices, operation)?,
        ))?;
    }
    upload_f32(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

fn reduce_broadcast_f32(
    backend: &CpuBackend,
    source: &Tensor,
    target_shape: &[u64],
    scale: f32,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    let mut values = zero_f32_values(backend, context, target_shape, "broadcast gradient")?;
    for linear in 0..element_count(source.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, source.descriptor().shape())?;
        let target_indices = broadcast_indices(&indices, target_shape)?;
        let target_linear = ravel_index(&target_indices, target_shape)?;
        values[target_linear] += read_f32(source, &indices, operation)? * scale;
    }
    upload_f32(
        backend,
        target_shape,
        source.descriptor().stream(),
        &values,
        context,
    )
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSixteenError> {
    if tensor.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartSixteenError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        })
    }
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSixteenError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartSixteenError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        })
    }
}

fn require_binary_f32(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSixteenError> {
    require_f32_cpu(left, operation)?;
    require_f32_cpu(right, operation)?;
    require_same_stream(left, right, operation)
}

fn require_same_stream(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSixteenError> {
    if left.descriptor().stream() == right.descriptor().stream() {
        Ok(())
    } else {
        invalid(operation, "input tensors must use the same stream")
    }
}

fn require_shape_stream(
    tensor: &Tensor,
    shape: &[u64],
    stream: StreamId,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSixteenError> {
    if tensor.descriptor().shape() != shape {
        return invalid(operation, "tensor shape does not match the required shape");
    }
    if tensor.descriptor().stream() != stream {
        return invalid(
            operation,
            "tensor stream does not match the required stream",
        );
    }
    Ok(())
}

fn scalar_as_f64(scalar: Scalar) -> f64 {
    match scalar {
        Scalar::Boolean(value) => f64::from(u8::from(value)),
        Scalar::Signed(value) => value as f64,
        Scalar::Unsigned(value) => value as f64,
        Scalar::Float(value) => value,
    }
}

fn read_f32(
    tensor: &Tensor,
    indices: &[u64],
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartSixteenError> {
    match tensor
        .descriptor()
        .dtype()
        .decode_scalar(tensor.element_bytes(indices)?)?
    {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(ElementwiseRuntimePartSixteenError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        }),
    }
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_bytes(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSixteenError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartSixteenError> {
    if index.is_multiple_of(64) {
        cancellation.check()?;
    }
    Ok(())
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartSixteenError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(usize::try_from(*dimension).map_err(|_| {
                ElementwiseRuntimePartSixteenError::ShapeOverflow("dimension conversion")
            })?)
            .ok_or(ElementwiseRuntimePartSixteenError::ShapeOverflow(
                "element count",
            ))
    })
}

fn zero_f32_values(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: &[u64],
    label: &'static str,
) -> Result<TemporaryVec<f32>, ElementwiseRuntimePartSixteenError> {
    let count = element_count(shape)?;
    let mut values = temporary_vec(backend, context, count, label)?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        values.try_push(0.0)?;
    }
    Ok(values)
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartSixteenError> {
    let mut indices = vec![0_u64; shape.len()];
    for dimension in (0..shape.len()).rev() {
        let size = usize::try_from(shape[dimension])
            .map_err(|_| ElementwiseRuntimePartSixteenError::ShapeOverflow("index dimension"))?;
        if size == 0 {
            return invalid(TILE_OPERATION_ID, "cannot index an empty tensor");
        }
        indices[dimension] = u64::try_from(linear % size)
            .map_err(|_| ElementwiseRuntimePartSixteenError::ShapeOverflow("index conversion"))?;
        linear /= size;
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
) -> Result<usize, ElementwiseRuntimePartSixteenError> {
    if indices.len() != shape.len() {
        return invalid(TILE_OPERATION_ID, "index rank does not match shape rank");
    }
    indices
        .iter()
        .zip(shape)
        .try_fold(0_usize, |linear, (index, dimension)| {
            let dimension = usize::try_from(*dimension).map_err(|_| {
                ElementwiseRuntimePartSixteenError::ShapeOverflow("ravel dimension")
            })?;
            let index = usize::try_from(*index)
                .map_err(|_| ElementwiseRuntimePartSixteenError::ShapeOverflow("ravel index"))?;
            linear
                .checked_mul(dimension)
                .and_then(|linear| linear.checked_add(index))
                .ok_or(ElementwiseRuntimePartSixteenError::ShapeOverflow(
                    "ravel index",
                ))
        })
}

fn modified_bessel_i0(value: f64) -> f64 {
    let absolute = value.abs();
    if absolute < 3.75 {
        let ratio = value / 3.75;
        let squared = ratio * ratio;
        1.0 + squared
            * (3.515_622_9
                + squared
                    * (3.089_942_4
                        + squared
                            * (1.206_749_2
                                + squared
                                    * (0.265_973_2
                                        + squared * (0.036_076_8 + squared * 0.004_581_3)))))
    } else {
        let ratio = 3.75 / absolute;
        (absolute.exp() / absolute.sqrt())
            * (0.398_942_28
                + ratio
                    * (0.013_285_92
                        + ratio
                            * (0.002_253_19
                                + ratio
                                    * (-0.001_575_65
                                        + ratio
                                            * (0.009_162_81
                                                + ratio
                                                    * (-0.020_577_06
                                                        + ratio
                                                            * (0.026_355_37
                                                                + ratio
                                                                    * (-0.016_476_33
                                                                        + ratio
                                                                            * 0.003_923_77))))))))
    }
}

fn logaddexp_weights(left: f32, right: f32) -> (f32, f32) {
    if left.is_nan() || right.is_nan() {
        return (f32::NAN, f32::NAN);
    }
    if left == right && left.is_infinite() {
        return (0.5, 0.5);
    }
    let left_weight = 1.0 / (1.0 + (right - left).exp());
    (left_weight, 1.0 - left_weight)
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, ElementwiseRuntimePartSixteenError> {
    Err(ElementwiseRuntimePartSixteenError::Invalid {
        operation,
        reason: reason.into(),
    })
}
