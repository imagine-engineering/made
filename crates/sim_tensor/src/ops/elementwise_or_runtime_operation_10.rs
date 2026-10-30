use std::cmp::Ordering;

use crate::cpu_backend::CpuWorkspaceVec;
use crate::{
    AutocastPolicy, BinaryOperation, CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId,
    ExecutionContext, IntegerInfo, NumericClass, Scalar, ScalarSide, StreamId, Tensor,
    TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
    generated_elementwise_or_runtime_operation_06::{
        ElementwiseRuntimePartSixError, unique_flat_with_context_exact_native,
    },
};
use thiserror::Error;

pub use crate::generated_elementwise_or_runtime_operation_06::UniqueResult;

pub const CUMSUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-75127DF334F2";
pub const SIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-77C67CAAC4AD";
pub const FMOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-78FD9BB26FAF";
pub const HANN_WINDOW_OPERATION_ID: &str = "COMFY-TENSOR-OP-73EF9076727A";
pub const IINFO_OPERATION_ID: &str = "COMFY-TENSOR-OP-75F89A81FD21";
pub const AUTOCAST_CACHE_OPERATION_ID: &str = "COMFY-TENSOR-OP-6F8F8AE14084";
pub const LOG2_OPERATION_ID: &str = "COMFY-TENSOR-OP-6D6C617423EA";
pub const MLU_DEVICE_COUNT_OPERATION_ID: &str = "COMFY-TENSOR-OP-73E8932FDF3A";
pub const PAIR_OPERATION_ID: &str = "COMFY-TENSOR-OP-6E17F49E5F14";
pub const UNIQUE_OPERATION_ID: &str = "COMFY-TENSOR-OP-706EE92A3AD0";
pub const UNIQUE_CONSECUTIVE_OPERATION_ID: &str = "COMFY-TENSOR-OP-78CD1B8EFCEC";
pub const XPU_EMPTY_CACHE_OPERATION_ID: &str = "COMFY-TENSOR-OP-6DEA145A655F";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartTenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartSix(#[from] ElementwiseRuntimePartSixError),
    #[error("elementwise/runtime part-ten operation was cancelled")]
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
    #[error("elementwise/runtime part-ten input is invalid: {0}")]
    Invalid(String),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartTenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorSize {
    Shape(Vec<u64>),
    Dimension(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairInput {
    Scalar(u64),
    Pair([u64; 2]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeCumulativeOperation {
    Product,
    Sum,
}

pub fn cumsum_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    cumulative_with_context_exact_native(
        backend,
        input,
        dimension,
        dtype,
        NativeCumulativeOperation::Sum,
        CUMSUM_OPERATION_ID,
        context,
    )
}

pub fn cumulative_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    dtype: Option<DType>,
    cumulative_operation: NativeCumulativeOperation,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    require_cpu(input, operation)?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let output_dtype = dtype.unwrap_or_else(|| cumsum_default_dtype(input.descriptor().dtype()));
    if output_dtype == DType::Bool || output_dtype.is_float8() {
        return Err(ElementwiseRuntimePartTenError::UnsupportedDType {
            operation,
            dtype: output_dtype,
        });
    }
    let shape = input.descriptor().shape();
    let state_count = element_count_excluding(shape, axis)?;
    let initial = match cumulative_operation {
        NativeCumulativeOperation::Product => one_decoded(output_dtype, operation)?,
        NativeCumulativeOperation::Sum => zero_decoded(output_dtype, operation)?,
    };
    let mut accumulators = workspace_filled(backend, context, state_count, initial)?;
    let count = element_count(shape)?;
    let width = usize::try_from(output_dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("cumsum dtype width"))?;
    let byte_count =
        count
            .checked_mul(width)
            .ok_or(ElementwiseRuntimePartTenError::ShapeOverflow(
                "cumsum bytes",
            ))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, shape)?;
        let state = linear_index_excluding(&indices, shape, axis)?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        accumulators[state] = combine_decoded(
            accumulators[state],
            value,
            output_dtype,
            cumulative_operation,
            operation,
        )?;
        temporary_extend(
            &mut bytes,
            &output_dtype.encode_decoded_scalar(accumulators[state], operation, DeviceId::CPU)?,
        )?;
    }
    upload_bytes_with_context(
        backend,
        shape,
        output_dtype,
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

pub fn cumsum_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    cumulative_f32_with_context(backend, output_gradient, dimension, true, context)
}

pub fn cumsum_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    cumulative_f32_with_context(backend, input_tangent, dimension, false, context)
}

pub fn size_exact_native(
    input: &Tensor,
    dimension: Option<i64>,
    cancellation: &CancellationToken,
) -> Result<TensorSize, ElementwiseRuntimePartTenError> {
    cancellation.check()?;
    match dimension {
        Some(dimension) => {
            let axis = normalize_axis(dimension, input.descriptor().rank())?;
            Ok(TensorSize::Dimension(input.descriptor().shape()[axis]))
        }
        None => Ok(TensorSize::Shape(input.descriptor().shape().to_vec())),
    }
}

pub fn fmod_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    divisor: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, FMOD_OPERATION_ID)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .binary_scalar(
            BinaryOperation::FloatingRemainder,
            input,
            Scalar::Float(f64::from(divisor)),
            ScalarSide::Right,
            descriptor,
            context,
        )?
        .0)
}

pub fn fmod_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    require_same_f32(input, output_gradient, FMOD_OPERATION_ID)?;
    copy_tensor_with_context(backend, output_gradient, context)
}

pub fn fmod_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    fmod_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn hann_window_with_context_exact_native(
    backend: &CpuBackend,
    window_length: usize,
    periodic: bool,
    dtype: DType,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(ElementwiseRuntimePartTenError::UnsupportedDType {
            operation: HANN_WINDOW_OPERATION_ID,
            dtype,
        });
    }
    let width = usize::try_from(dtype.byte_width())
        .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("Hann dtype width"))?;
    let byte_count = window_length
        .checked_mul(width)
        .ok_or(ElementwiseRuntimePartTenError::ShapeOverflow("Hann bytes"))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for index in 0..window_length {
        check_periodically(index, context.cancellation)?;
        let value = if window_length <= 1 {
            1.0
        } else {
            let denominator = if periodic {
                window_length
            } else {
                window_length - 1
            };
            0.5 * (1.0 - (std::f64::consts::TAU * index as f64 / denominator as f64).cos())
        };
        temporary_extend(
            &mut bytes,
            &dtype.encode_scalar(
                Scalar::Float(value),
                HANN_WINDOW_OPERATION_ID,
                DeviceId::CPU,
            )?,
        )?;
    }
    upload_bytes_with_context(
        backend,
        &[u64::try_from(window_length)
            .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("Hann shape"))?],
        dtype,
        stream,
        &bytes,
        context,
    )
}

pub fn iinfo_exact_native(
    dtype: DType,
    cancellation: &CancellationToken,
) -> Result<IntegerInfo, ElementwiseRuntimePartTenError> {
    cancellation.check()?;
    Ok(dtype.integer_info()?)
}

pub fn is_autocast_cache_enabled_exact_native(
    policy: &AutocastPolicy,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartTenError> {
    cancellation.check()?;
    Ok(policy.cache_enabled())
}

pub fn log2_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, LOG2_OPERATION_ID)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .unary(UnaryOperation::LogarithmBaseTwo, input, descriptor, context)?
        .0)
}

pub fn log2_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    map_binary_f32_with_context(
        backend,
        input,
        output_gradient,
        LOG2_OPERATION_ID,
        context,
        |value, gradient| gradient / (value * std::f32::consts::LN_2),
    )
}

pub fn log2_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    log2_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn pair_exact_native(
    input: PairInput,
    cancellation: &CancellationToken,
) -> Result<[u64; 2], ElementwiseRuntimePartTenError> {
    cancellation.check()?;
    Ok(match input {
        PairInput::Scalar(value) => [value, value],
        PairInput::Pair(value) => value,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn unique_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    _sorted: bool,
    return_inverse: bool,
    return_counts: bool,
    dimension: Option<i64>,
    context: &ExecutionContext<'_>,
) -> Result<UniqueResult, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    unique_impl_with_context(
        backend,
        input,
        return_inverse,
        return_counts,
        dimension,
        false,
        context,
    )
}

pub fn unique_consecutive_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    return_inverse: bool,
    return_counts: bool,
    dimension: Option<i64>,
    context: &ExecutionContext<'_>,
) -> Result<UniqueResult, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    unique_impl_with_context(
        backend,
        input,
        return_inverse,
        return_counts,
        dimension,
        true,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn unique_impl_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    return_inverse: bool,
    return_counts: bool,
    dimension: Option<i64>,
    consecutive: bool,
    context: &ExecutionContext<'_>,
) -> Result<UniqueResult, ElementwiseRuntimePartTenError> {
    context.cancellation.check()?;
    let operation = if consecutive {
        UNIQUE_CONSECUTIVE_OPERATION_ID
    } else {
        UNIQUE_OPERATION_ID
    };
    if dimension.is_none() && !consecutive {
        return Ok(unique_flat_with_context_exact_native(
            backend,
            input,
            return_inverse,
            return_counts,
            operation,
            context,
        )?);
    }
    require_cpu(input, operation)?;
    if input.descriptor().dtype().class() == NumericClass::Complex {
        return Err(ElementwiseRuntimePartTenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        });
    }
    let axis = dimension
        .map(|dimension| normalize_axis(dimension, input.descriptor().rank()))
        .transpose()?;
    let item_count = match axis {
        Some(axis) => usize::try_from(input.descriptor().shape()[axis])
            .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("unique axis"))?,
        None => element_count(input.descriptor().shape())?,
    };
    let values_per_item = match axis {
        Some(axis) => element_count_excluding(input.descriptor().shape(), axis)?,
        None => 1,
    };
    let decoded_count = item_count.checked_mul(values_per_item).ok_or(
        ElementwiseRuntimePartTenError::ShapeOverflow("unique items"),
    )?;
    let mut decoded_items = backend.workspace_vec::<DecodedScalar>(context, decoded_count)?;
    for source_index in 0..item_count {
        check_periodically(source_index, context.cancellation)?;
        for item_index in 0..values_per_item {
            let indices =
                unique_source_indices(input.descriptor().shape(), axis, source_index, item_index)?;
            decoded_items.try_push(
                input
                    .descriptor()
                    .dtype()
                    .decode_scalar(input.element_bytes(&indices)?)?,
            )?;
        }
    }
    let mut order = backend.workspace_vec::<usize>(context, item_count)?;
    for index in 0..item_count {
        order.try_push(index)?;
    }
    if !consecutive {
        order.sort_by(|left, right| {
            let left_start = *left * values_per_item;
            let right_start = *right * values_per_item;
            compare_unique_value_slices(
                &decoded_items[left_start..left_start + values_per_item],
                &decoded_items[right_start..right_start + values_per_item],
            )
            .then_with(|| left.cmp(right))
        });
    }
    let mut representatives = backend.workspace_vec::<usize>(context, item_count)?;
    let mut counts = backend.workspace_vec::<u64>(context, item_count)?;
    let mut inverse = workspace_filled(backend, context, item_count, 0_u64)?;
    for item_index in order.iter().copied() {
        let same_as_previous = representatives
            .last()
            .is_some_and(|representative: &usize| {
                let representative_start = *representative * values_per_item;
                let item_start = item_index * values_per_item;
                equal_unique_value_slices(
                    &decoded_items[representative_start..representative_start + values_per_item],
                    &decoded_items[item_start..item_start + values_per_item],
                )
            });
        if same_as_previous {
            let count = counts.last_mut().ok_or_else(|| {
                ElementwiseRuntimePartTenError::Invalid("unique count state is empty".to_owned())
            })?;
            *count = count
                .checked_add(1)
                .ok_or(ElementwiseRuntimePartTenError::ShapeOverflow(
                    "unique count",
                ))?;
        } else {
            representatives.try_push(item_index)?;
            counts.try_push(1)?;
        }
        let group = representatives.len().checked_sub(1).ok_or(
            ElementwiseRuntimePartTenError::ShapeOverflow("unique inverse"),
        )?;
        inverse[item_index] = u64::try_from(group)
            .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("unique inverse"))?;
    }
    drop(order);
    let values = build_unique_values_with_context(backend, input, axis, &representatives, context)?;
    let inverse_indices = if return_inverse {
        let shape = match axis {
            Some(_) => vec![u64::try_from(item_count).map_err(|_| {
                ElementwiseRuntimePartTenError::ShapeOverflow("unique inverse shape")
            })?],
            None => input.descriptor().shape().to_vec(),
        };
        Some(upload_i64_with_context(
            backend,
            &shape,
            input.descriptor().stream(),
            &inverse,
            context,
        )?)
    } else {
        None
    };
    let counts = if return_counts {
        Some(upload_i64_with_context(
            backend,
            &[u64::try_from(counts.len()).map_err(|_| {
                ElementwiseRuntimePartTenError::ShapeOverflow("unique counts shape")
            })?],
            input.descriptor().stream(),
            &counts,
            context,
        )?)
    } else {
        None
    };
    Ok(UniqueResult {
        values,
        inverse_indices,
        counts,
    })
}

fn compare_unique_value_slices(left: &[DecodedScalar], right: &[DecodedScalar]) -> Ordering {
    for (left, right) in left.iter().zip(right) {
        let ordering = compare_decoded(*left, *right);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn equal_unique_value_slices(left: &[DecodedScalar], right: &[DecodedScalar]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| equal_decoded(*left, *right))
}

fn compare_decoded(left: DecodedScalar, right: DecodedScalar) -> Ordering {
    match (left, right) {
        (DecodedScalar::Boolean(left), DecodedScalar::Boolean(right)) => left.cmp(&right),
        (DecodedScalar::Signed(left), DecodedScalar::Signed(right)) => left.cmp(&right),
        (DecodedScalar::Unsigned(left), DecodedScalar::Unsigned(right)) => left.cmp(&right),
        (DecodedScalar::Real(left), DecodedScalar::Real(right)) => left.total_cmp(&right),
        _ => decoded_kind(left).cmp(&decoded_kind(right)),
    }
}

fn equal_decoded(left: DecodedScalar, right: DecodedScalar) -> bool {
    match (left, right) {
        (DecodedScalar::Boolean(left), DecodedScalar::Boolean(right)) => left == right,
        (DecodedScalar::Signed(left), DecodedScalar::Signed(right)) => left == right,
        (DecodedScalar::Unsigned(left), DecodedScalar::Unsigned(right)) => left == right,
        (DecodedScalar::Real(left), DecodedScalar::Real(right)) => left == right,
        _ => false,
    }
}

fn decoded_kind(value: DecodedScalar) -> u8 {
    match value {
        DecodedScalar::Boolean(_) => 0,
        DecodedScalar::Signed(_) => 1,
        DecodedScalar::Unsigned(_) => 2,
        DecodedScalar::Real(_) => 3,
        DecodedScalar::Complex { .. } => 4,
    }
}

fn build_unique_values_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    axis: Option<usize>,
    representatives: &[usize],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    let output_shape =
        match axis {
            Some(axis) => {
                let mut shape = input.descriptor().shape().to_vec();
                shape[axis] = u64::try_from(representatives.len()).map_err(|_| {
                    ElementwiseRuntimePartTenError::ShapeOverflow("unique values shape")
                })?;
                shape
            }
            None => vec![u64::try_from(representatives.len()).map_err(|_| {
                ElementwiseRuntimePartTenError::ShapeOverflow("unique values shape")
            })?],
        };
    let descriptor = TensorDescriptor::contiguous(
        output_shape.clone(),
        input.descriptor().dtype(),
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    let (mut output, _) = backend.allocate(descriptor, context)?;
    let count = element_count(&output_shape)?;
    let mut write = output.write()?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let source_indices = match axis {
            Some(axis) => {
                let representative = usize::try_from(output_indices[axis]).map_err(|_| {
                    ElementwiseRuntimePartTenError::ShapeOverflow("unique representative")
                })?;
                let source_index = *representatives.get(representative).ok_or_else(|| {
                    ElementwiseRuntimePartTenError::Invalid(
                        "unique representative is missing".to_owned(),
                    )
                })?;
                let mut indices = output_indices.clone();
                indices[axis] = u64::try_from(source_index).map_err(|_| {
                    ElementwiseRuntimePartTenError::ShapeOverflow("unique source index")
                })?;
                indices
            }
            None => {
                let representative = usize::try_from(output_indices[0]).map_err(|_| {
                    ElementwiseRuntimePartTenError::ShapeOverflow("unique representative")
                })?;
                let source_index = *representatives.get(representative).ok_or_else(|| {
                    ElementwiseRuntimePartTenError::Invalid(
                        "unique representative is missing".to_owned(),
                    )
                })?;
                unravel_index(source_index, input.descriptor().shape())?
            }
        };
        write
            .element_bytes_mut(&output_indices)?
            .copy_from_slice(input.element_bytes(&source_indices)?);
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

fn cumulative_f32_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    reverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    require_f32_cpu(input, CUMSUM_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let shape = input.descriptor().shape();
    let state_count = element_count_excluding(shape, axis)?;
    let mut accumulators = workspace_filled(backend, context, state_count, 0.0_f32)?;
    let count = element_count(shape)?;
    let mut output = workspace_filled(backend, context, count, 0.0_f32)?;
    for iteration in 0..count {
        check_periodically(iteration, context.cancellation)?;
        let linear = if reverse {
            count - iteration - 1
        } else {
            iteration
        };
        let indices = unravel_index(linear, shape)?;
        let state = linear_index_excluding(&indices, shape, axis)?;
        accumulators[state] += read_f32(input, &indices)?;
        output[linear] = accumulators[state];
    }
    upload_f32_with_context(
        backend,
        shape,
        input.descriptor().stream(),
        &output,
        context,
    )
}

fn map_binary_f32_with_context(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
    function: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    require_same_f32(left, right, operation)?;
    let count = element_count(left.descriptor().shape())?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, left.descriptor().shape())?;
        values.try_push(function(
            read_f32(left, &indices)?,
            read_f32(right, &indices)?,
        ))?;
    }
    upload_f32_with_context(
        backend,
        left.descriptor().shape(),
        left.descriptor().stream(),
        &values,
        context,
    )
}

fn copy_tensor_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        input.descriptor().dtype(),
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.copy(input, descriptor, context)?.0)
}

fn cumsum_default_dtype(dtype: DType) -> DType {
    match dtype.class() {
        NumericClass::Boolean | NumericClass::SignedInteger | NumericClass::UnsignedInteger => {
            DType::I64
        }
        NumericClass::FloatingPoint | NumericClass::Complex => dtype,
    }
}

fn zero_decoded(
    dtype: DType,
    operation: &'static str,
) -> Result<DecodedScalar, ElementwiseRuntimePartTenError> {
    Ok(match dtype.class() {
        NumericClass::Boolean => {
            return Err(ElementwiseRuntimePartTenError::UnsupportedDType { operation, dtype });
        }
        NumericClass::SignedInteger => DecodedScalar::Signed(0),
        NumericClass::UnsignedInteger => DecodedScalar::Unsigned(0),
        NumericClass::FloatingPoint => DecodedScalar::Real(0.0),
        NumericClass::Complex => DecodedScalar::Complex {
            real: 0.0,
            imaginary: 0.0,
        },
    })
}

fn one_decoded(
    dtype: DType,
    operation: &'static str,
) -> Result<DecodedScalar, ElementwiseRuntimePartTenError> {
    Ok(match dtype.class() {
        NumericClass::Boolean => {
            return Err(ElementwiseRuntimePartTenError::UnsupportedDType { operation, dtype });
        }
        NumericClass::SignedInteger => DecodedScalar::Signed(1),
        NumericClass::UnsignedInteger => DecodedScalar::Unsigned(1),
        NumericClass::FloatingPoint => DecodedScalar::Real(1.0),
        NumericClass::Complex => DecodedScalar::Complex {
            real: 1.0,
            imaginary: 0.0,
        },
    })
}

fn combine_decoded(
    accumulator: DecodedScalar,
    value: DecodedScalar,
    output_dtype: DType,
    cumulative_operation: NativeCumulativeOperation,
    operation: &'static str,
) -> Result<DecodedScalar, ElementwiseRuntimePartTenError> {
    match cumulative_operation {
        NativeCumulativeOperation::Sum => add_decoded(accumulator, value, output_dtype),
        NativeCumulativeOperation::Product => {
            multiply_decoded(accumulator, value, output_dtype, operation)
        }
    }
}

fn add_decoded(
    accumulator: DecodedScalar,
    value: DecodedScalar,
    output_dtype: DType,
) -> Result<DecodedScalar, ElementwiseRuntimePartTenError> {
    match accumulator {
        DecodedScalar::Signed(accumulator) => {
            let value = match value {
                DecodedScalar::Boolean(value) => i64::from(value),
                DecodedScalar::Signed(value) => value,
                DecodedScalar::Unsigned(value) => i64::try_from(value).map_err(|_| {
                    ElementwiseRuntimePartTenError::Invalid(
                        "cumsum unsigned input exceeds int64 accumulation".to_owned(),
                    )
                })?,
                _ => return incompatible_accumulator(output_dtype),
            };
            Ok(DecodedScalar::Signed(normalize_signed_accumulator(
                accumulator.wrapping_add(value),
                output_dtype,
            )?))
        }
        DecodedScalar::Unsigned(accumulator) => {
            let value = match value {
                DecodedScalar::Boolean(value) => u64::from(value),
                DecodedScalar::Signed(value) => u64::try_from(value).map_err(|_| {
                    ElementwiseRuntimePartTenError::Invalid(
                        "cumsum negative input cannot use unsigned accumulation".to_owned(),
                    )
                })?,
                DecodedScalar::Unsigned(value) => value,
                _ => return incompatible_accumulator(output_dtype),
            };
            Ok(DecodedScalar::Unsigned(normalize_unsigned_accumulator(
                accumulator.wrapping_add(value),
                output_dtype,
            )?))
        }
        DecodedScalar::Real(accumulator) => {
            let value = match value {
                DecodedScalar::Boolean(value) => f64::from(u8::from(value)),
                DecodedScalar::Signed(value) => value as f64,
                DecodedScalar::Unsigned(value) => value as f64,
                DecodedScalar::Real(value) => value,
                _ => return incompatible_accumulator(output_dtype),
            };
            Ok(DecodedScalar::Real(accumulator + value))
        }
        DecodedScalar::Complex {
            real: accumulator_real,
            imaginary: accumulator_imaginary,
        } => {
            let (real, imaginary) = match value {
                DecodedScalar::Boolean(value) => (f64::from(u8::from(value)), 0.0),
                DecodedScalar::Signed(value) => (value as f64, 0.0),
                DecodedScalar::Unsigned(value) => (value as f64, 0.0),
                DecodedScalar::Real(value) => (value, 0.0),
                DecodedScalar::Complex { real, imaginary } => (real, imaginary),
            };
            Ok(DecodedScalar::Complex {
                real: accumulator_real + real,
                imaginary: accumulator_imaginary + imaginary,
            })
        }
        DecodedScalar::Boolean(_) => incompatible_accumulator(output_dtype),
    }
}

fn multiply_decoded(
    accumulator: DecodedScalar,
    value: DecodedScalar,
    output_dtype: DType,
    operation: &'static str,
) -> Result<DecodedScalar, ElementwiseRuntimePartTenError> {
    match accumulator {
        DecodedScalar::Signed(accumulator) => {
            let value = match value {
                DecodedScalar::Boolean(value) => i64::from(value),
                DecodedScalar::Signed(value) => value,
                DecodedScalar::Unsigned(value) => i64::try_from(value).map_err(|_| {
                    ElementwiseRuntimePartTenError::Invalid(format!(
                        "{operation} unsigned input exceeds int64 accumulation"
                    ))
                })?,
                _ => return incompatible_accumulator(output_dtype),
            };
            Ok(DecodedScalar::Signed(normalize_signed_accumulator(
                accumulator.wrapping_mul(value),
                output_dtype,
            )?))
        }
        DecodedScalar::Unsigned(accumulator) => {
            let value = match value {
                DecodedScalar::Boolean(value) => u64::from(value),
                DecodedScalar::Signed(value) => u64::try_from(value).map_err(|_| {
                    ElementwiseRuntimePartTenError::Invalid(format!(
                        "{operation} negative input cannot use unsigned accumulation"
                    ))
                })?,
                DecodedScalar::Unsigned(value) => value,
                _ => return incompatible_accumulator(output_dtype),
            };
            Ok(DecodedScalar::Unsigned(normalize_unsigned_accumulator(
                accumulator.wrapping_mul(value),
                output_dtype,
            )?))
        }
        DecodedScalar::Real(accumulator) => {
            let value = match value {
                DecodedScalar::Boolean(value) => f64::from(u8::from(value)),
                DecodedScalar::Signed(value) => value as f64,
                DecodedScalar::Unsigned(value) => value as f64,
                DecodedScalar::Real(value) => value,
                _ => return incompatible_accumulator(output_dtype),
            };
            Ok(DecodedScalar::Real(accumulator * value))
        }
        DecodedScalar::Complex {
            real: accumulator_real,
            imaginary: accumulator_imaginary,
        } => {
            let (real, imaginary) = match value {
                DecodedScalar::Boolean(value) => (f64::from(u8::from(value)), 0.0),
                DecodedScalar::Signed(value) => (value as f64, 0.0),
                DecodedScalar::Unsigned(value) => (value as f64, 0.0),
                DecodedScalar::Real(value) => (value, 0.0),
                DecodedScalar::Complex { real, imaginary } => (real, imaginary),
            };
            Ok(DecodedScalar::Complex {
                real: accumulator_real * real - accumulator_imaginary * imaginary,
                imaginary: accumulator_real * imaginary + accumulator_imaginary * real,
            })
        }
        DecodedScalar::Boolean(_) => incompatible_accumulator(output_dtype),
    }
}

fn normalize_signed_accumulator(
    value: i64,
    dtype: DType,
) -> Result<i64, ElementwiseRuntimePartTenError> {
    match dtype {
        DType::I8 => Ok(i64::from(value as i8)),
        DType::I16 => Ok(i64::from(value as i16)),
        DType::I32 => Ok(i64::from(value as i32)),
        DType::I64 => Ok(value),
        _ => incompatible_accumulator(dtype),
    }
}

fn normalize_unsigned_accumulator(
    value: u64,
    dtype: DType,
) -> Result<u64, ElementwiseRuntimePartTenError> {
    match dtype {
        DType::U8 => Ok(u64::from(value as u8)),
        DType::U16 => Ok(u64::from(value as u16)),
        DType::U32 => Ok(u64::from(value as u32)),
        DType::U64 => Ok(value),
        _ => incompatible_accumulator(dtype),
    }
}

fn incompatible_accumulator<T>(dtype: DType) -> Result<T, ElementwiseRuntimePartTenError> {
    Err(ElementwiseRuntimePartTenError::Invalid(format!(
        "cumsum input cannot be accumulated into {dtype:?}"
    )))
}

fn upload_bytes_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn temporary_extend(
    values: &mut CpuWorkspaceVec<u8>,
    extension: &[u8],
) -> Result<(), ElementwiseRuntimePartTenError> {
    for value in extension {
        values.try_push(*value)?;
    }
    Ok(())
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, ElementwiseRuntimePartTenError> {
    let mut values = backend.workspace_vec(context, capacity)?;
    for _ in 0..capacity {
        values.try_push(value)?;
    }
    Ok(values)
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_i64_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTenError> {
    let byte_count =
        values
            .len()
            .checked_mul(8)
            .ok_or(ElementwiseRuntimePartTenError::ShapeOverflow(
                "int64 output",
            ))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for value in values {
        temporary_extend(
            &mut bytes,
            &i64::try_from(*value)
                .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("int64 output"))?
                .to_ne_bytes(),
        )?;
    }
    upload_bytes_with_context(backend, shape, DType::I64, stream, &bytes, context)
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTenError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartTenError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTenError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartTenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}

fn require_same_f32(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTenError> {
    require_f32_cpu(left, operation)?;
    require_f32_cpu(right, operation)?;
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(ElementwiseRuntimePartTenError::Invalid(
            "tensor shapes must match".to_owned(),
        ));
    }
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: left.descriptor().stream(),
            actual: right.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn normalize_axis(dimension: i64, rank: usize) -> Result<usize, ElementwiseRuntimePartTenError> {
    if rank == 0 {
        return Err(ElementwiseRuntimePartTenError::Invalid(
            "dimension is invalid for a scalar tensor".to_owned(),
        ));
    }
    let rank = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("tensor rank"))?;
    let dimension = if dimension < 0 {
        dimension
            .checked_add(rank)
            .ok_or(ElementwiseRuntimePartTenError::ShapeOverflow("dimension"))?
    } else {
        dimension
    };
    if !(0..rank).contains(&dimension) {
        return Err(ElementwiseRuntimePartTenError::Invalid(format!(
            "dimension {dimension} is outside rank {rank}"
        )));
    }
    usize::try_from(dimension)
        .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("dimension"))
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartTenError> {
    usize::try_from(
        shape
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
            .ok_or(ElementwiseRuntimePartTenError::ShapeOverflow(
                "element count",
            ))?,
    )
    .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("element count"))
}

fn element_count_excluding(
    shape: &[u64],
    axis: usize,
) -> Result<usize, ElementwiseRuntimePartTenError> {
    usize::try_from(
        shape
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != axis)
            .try_fold(1_u64, |count, (_, dimension)| count.checked_mul(*dimension))
            .ok_or(ElementwiseRuntimePartTenError::ShapeOverflow(
                "scan state count",
            ))?,
    )
    .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("scan state count"))
}

fn unique_source_indices(
    shape: &[u64],
    axis: Option<usize>,
    source_index: usize,
    item_index: usize,
) -> Result<Vec<u64>, ElementwiseRuntimePartTenError> {
    match axis {
        None => unravel_index(source_index, shape),
        Some(axis) => {
            let mut slice_shape = shape.to_vec();
            slice_shape.remove(axis);
            let mut indices = unravel_index(item_index, &slice_shape)?;
            indices.insert(
                axis,
                u64::try_from(source_index).map_err(|_| {
                    ElementwiseRuntimePartTenError::ShapeOverflow("unique source index")
                })?,
            );
            Ok(indices)
        }
    }
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartTenError> {
    let mut indices = vec![0; shape.len()];
    for (slot, dimension) in indices.iter_mut().zip(shape).rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("tensor index"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartTenError::Invalid(
                "cannot index an empty tensor".to_owned(),
            ));
        }
        *slot = u64::try_from(linear % dimension)
            .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("tensor index"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn linear_index_excluding(
    indices: &[u64],
    shape: &[u64],
    axis: usize,
) -> Result<usize, ElementwiseRuntimePartTenError> {
    usize::try_from(
        indices
            .iter()
            .zip(shape)
            .enumerate()
            .filter(|(index, _)| *index != axis)
            .try_fold(0_u64, |linear, (_, (index, dimension))| {
                linear
                    .checked_mul(*dimension)
                    .and_then(|value| value.checked_add(*index))
            })
            .ok_or(ElementwiseRuntimePartTenError::ShapeOverflow(
                "scan state index",
            ))?,
    )
    .map_err(|_| ElementwiseRuntimePartTenError::ShapeOverflow("scan state index"))
}

fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, ElementwiseRuntimePartTenError> {
    let bytes: [u8; 4] = tensor.element_bytes(indices)?.try_into().map_err(|_| {
        ElementwiseRuntimePartTenError::Invalid(
            "f32 tensor element has an invalid byte width".to_owned(),
        )
    })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartTenError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}
