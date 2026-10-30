use crate::{
    AutocastPolicy, BackendCapabilityMatrix, BinaryOperation, CancellationToken, CpuBackend,
    CpuWorkspaceVec, DType, DecodedScalar, DeviceId, ExecutionContext, Layout, ScalarSide, Tensor,
    TensorBackend, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_01::{
        ElementwiseRuntimeError as ElementwiseRuntimePartOneError,
        abs_jvp_with_context_exact_native as canonical_abs_jvp,
        abs_vjp_with_context_exact_native as canonical_abs_vjp,
        abs_with_context_exact_native as canonical_abs,
    },
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_08::{
        ElementwiseRuntimePartEightError,
        concatenate_jvp_with_context_exact_native as canonical_concatenate_jvp,
        concatenate_vjp_with_context_exact_native as canonical_concatenate_vjp,
        concatenate_with_context_exact_native as canonical_concatenate,
    },
    generated_elementwise_or_runtime_operation_09::{
        ElementwiseRuntimePartNineError, mul_with_context_exact_native as canonical_mul,
    },
};
use sim_types::DeviceKind;
use std::cmp::Ordering;
use thiserror::Error;

pub const DETACH_OPERATION_ID: &str = "COMFY-TENSOR-OP-A2B7298E8EB4";
pub const MUL_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-A496777C1987";
pub const ABS_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-A3A6638578F7";
pub const ARGSORT_OPERATION_ID: &str = "COMFY-TENSOR-OP-A46ED7068064";
pub const AUTOCAST_OPERATION_ID: &str = "COMFY-TENSOR-OP-A59D885AD4F9";
pub const CUDNN_VERSION_OPERATION_ID: &str = "COMFY-TENSOR-OP-9E2C8B750099";
pub const CONCAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-9CD229514F61";
pub const ISPOSINF_OPERATION_ID: &str = "COMFY-TENSOR-OP-9D504472EFE5";
pub const VIEW_AS_REAL_OPERATION_ID: &str = "COMFY-TENSOR-OP-A67546895304";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartFourteenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartOne(#[from] ElementwiseRuntimePartOneError),
    #[error(transparent)]
    PartEight(#[from] ElementwiseRuntimePartEightError),
    #[error(transparent)]
    PartNine(#[from] ElementwiseRuntimePartNineError),
    #[error("elementwise/runtime part-fourteen operation was cancelled")]
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
    #[error("elementwise/runtime part-fourteen input is invalid: {0}")]
    Invalid(&'static str),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartFourteenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn detach_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    cancellation.check()?;
    let detached = input.detached_alias()?;
    cancellation.check()?;
    Ok(detached)
}

pub fn detach_vjp_exact_native(
    cancellation: &CancellationToken,
) -> Result<Option<Tensor>, ElementwiseRuntimePartFourteenError> {
    cancellation.check()?;
    let gradient = None;
    cancellation.check()?;
    Ok(gradient)
}

pub fn detach_jvp_exact_native(
    cancellation: &CancellationToken,
) -> Result<Option<Tensor>, ElementwiseRuntimePartFourteenError> {
    detach_vjp_exact_native(cancellation)
}

pub fn mul_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    other: ElementwiseOperand<'_>,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    require_cpu(input, MUL_IN_PLACE_OPERATION_ID)?;
    let staged = match other {
        ElementwiseOperand::Tensor(other) => canonical_mul(backend, input, other, context)?,
        ElementwiseOperand::Scalar(scalar) => {
            let descriptor = TensorDescriptor::contiguous(
                input.descriptor().shape().to_vec(),
                input.descriptor().dtype(),
                input.descriptor().device(),
                input.descriptor().stream(),
            )?;
            backend
                .binary_scalar(
                    BinaryOperation::Multiply,
                    input,
                    scalar,
                    ScalarSide::Right,
                    descriptor,
                    context,
                )?
                .0
        }
    };
    if staged.descriptor().shape() != input.descriptor().shape() {
        return Err(ElementwiseRuntimePartFourteenError::Invalid(
            "mul_ broadcast output must match the receiver shape",
        ));
    }
    context.check()?;
    input.commit_in_place(staged)?;
    Ok(())
}

pub fn abs_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    Ok(canonical_abs(backend, input, context)?)
}

pub fn abs_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    Ok(canonical_abs_vjp(backend, input, output_gradient, context)?)
}

pub fn abs_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    Ok(canonical_abs_jvp(backend, input, input_tangent, context)?)
}

pub fn argsort_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    descending: bool,
    stable: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    require_cpu(input, ARGSORT_OPERATION_ID)?;
    if matches!(
        input.descriptor().dtype(),
        DType::Complex64 | DType::Complex128
    ) {
        return Err(ElementwiseRuntimePartFourteenError::UnsupportedDType {
            operation: ARGSORT_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        });
    }
    let shape = input.descriptor().shape();
    let axis = normalize_axis(dimension, shape.len())?;
    let axis_length = usize::try_from(shape[axis])
        .map_err(|_| ElementwiseRuntimePartFourteenError::ShapeOverflow("argsort axis"))?;
    let outer = element_count(&shape[..axis])?;
    let inner = element_count(&shape[axis + 1..])?;
    let count = element_count(shape)?;
    let mut indices = workspace_filled(backend, context, count, 0_i64)?;
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            context.check()?;
            let mut order = backend.workspace_vec(context, axis_length)?;
            for axis_index in 0..axis_length {
                let linear = ((outer_index * axis_length) + axis_index) * inner + inner_index;
                let value = input
                    .descriptor()
                    .dtype()
                    .decode_scalar(input.element_bytes(&unravel_index(linear, shape)?)?)?;
                order.try_push((axis_index, value))?;
            }
            let compare = |left: &(usize, DecodedScalar), right: &(usize, DecodedScalar)| {
                let ordering = compare_scalars(left.1, right.1);
                let ordering = if descending {
                    ordering.reverse()
                } else {
                    ordering
                };
                if ordering == Ordering::Equal && stable {
                    left.0.cmp(&right.0)
                } else {
                    ordering
                }
            };
            if stable {
                order.sort_by(compare);
            } else {
                order.sort_unstable_by(compare);
            }
            for (output_axis_index, &(source_axis_index, _)) in order.iter().enumerate() {
                let linear =
                    ((outer_index * axis_length) + output_axis_index) * inner + inner_index;
                indices[linear] = i64::try_from(source_axis_index).map_err(|_| {
                    ElementwiseRuntimePartFourteenError::ShapeOverflow("argsort index")
                })?;
            }
        }
    }
    upload_i64(backend, shape, &indices, context)
}

pub fn autocast_exact_native(
    device_type: DeviceKind,
    dtype: Option<DType>,
    enabled: bool,
    cache_enabled: Option<bool>,
    cancellation: &CancellationToken,
) -> Result<AutocastPolicy, ElementwiseRuntimePartFourteenError> {
    cancellation.check()?;
    let dtype = dtype.unwrap_or(if device_type == DeviceKind::Cpu {
        DType::Bf16
    } else {
        DType::F16
    });
    let policy = AutocastPolicy::new(enabled, dtype, cache_enabled.unwrap_or(true))?;
    cancellation.check()?;
    Ok(policy)
}

pub fn cudnn_version_exact_native(
    _capabilities: &BackendCapabilityMatrix,
    cancellation: &CancellationToken,
) -> Result<Option<u64>, ElementwiseRuntimePartFourteenError> {
    cancellation.check()?;
    let version = None;
    cancellation.check()?;
    Ok(version)
}

pub fn concat_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    Ok(canonical_concatenate(backend, inputs, dimension, context)?)
}

pub fn concat_vjp_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    dimension: i64,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    Ok(canonical_concatenate_vjp(
        backend,
        inputs,
        dimension,
        output_gradient,
        context,
    )?)
}

pub fn concat_jvp_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    input_tangents: &[Tensor],
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    Ok(canonical_concatenate_jvp(
        backend,
        inputs,
        input_tangents,
        dimension,
        context,
    )?)
}

pub fn isposinf_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    context.cancellation.check()?;
    require_cpu(input, ISPOSINF_OPERATION_ID)?;
    let shape = input.descriptor().shape();
    let count = element_count(shape)?;
    let mut values = workspace_filled(backend, context, count, 0_u8)?;
    for (linear, output) in values.iter_mut().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let value = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&unravel_index(linear, shape)?)?)?;
        *output = u8::from(matches!(value, DecodedScalar::Real(value) if value == f64::INFINITY));
    }
    upload_bytes(backend, shape, DType::Bool, &values, context)
}

pub fn view_as_real_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    cancellation.check()?;
    require_cpu(input, VIEW_AS_REAL_OPERATION_ID)?;
    let dtype = match input.descriptor().dtype() {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        dtype => {
            return Err(ElementwiseRuntimePartFourteenError::UnsupportedDType {
                operation: VIEW_AS_REAL_OPERATION_ID,
                dtype,
            });
        }
    };
    let mut shape = input.descriptor().shape().to_vec();
    shape.push(2);
    let mut strides = Vec::new();
    strides
        .try_reserve_exact(input.descriptor().strides().len() + 1)
        .map_err(|_| ElementwiseRuntimePartFourteenError::ShapeOverflow("view_as_real strides"))?;
    for stride in input.descriptor().strides() {
        strides.push(stride.checked_mul(2).ok_or(
            ElementwiseRuntimePartFourteenError::ShapeOverflow("view_as_real stride"),
        )?);
    }
    strides.push(1);
    let offset = input.descriptor().offset_elements().checked_mul(2).ok_or(
        ElementwiseRuntimePartFourteenError::ShapeOverflow("view_as_real offset"),
    )?;
    let layout = if input.descriptor().layout() == Layout::Contiguous {
        Layout::Contiguous
    } else {
        Layout::Strided
    };
    let descriptor = TensorDescriptor::new_strided(
        shape,
        strides,
        offset,
        dtype,
        layout,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let output = input.reinterpret_read_only(descriptor)?;
    cancellation.check()?;
    Ok(output)
}

pub fn view_as_real_vjp_exact_native(
    output_gradient: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    cancellation.check()?;
    require_cpu(output_gradient, VIEW_AS_REAL_OPERATION_ID)?;
    if output_gradient.descriptor().shape().last().copied() != Some(2)
        || output_gradient.descriptor().strides().last().copied() != Some(1)
    {
        return Err(ElementwiseRuntimePartFourteenError::Invalid(
            "view_as_real gradient requires contiguous input with trailing dimension two",
        ));
    }
    let dtype = match output_gradient.descriptor().dtype() {
        DType::F32 => DType::Complex64,
        DType::F64 => DType::Complex128,
        dtype => {
            return Err(ElementwiseRuntimePartFourteenError::UnsupportedDType {
                operation: VIEW_AS_REAL_OPERATION_ID,
                dtype,
            });
        }
    };
    let mut shape = output_gradient.descriptor().shape().to_vec();
    shape.pop();
    let mut strides = output_gradient.descriptor().strides().to_vec();
    strides.pop();
    for stride in &mut strides {
        if *stride % 2 != 0 {
            return Err(ElementwiseRuntimePartFourteenError::Invalid(
                "view_as_real gradient strides must be complex aligned",
            ));
        }
        *stride /= 2;
    }
    let offset = output_gradient.descriptor().offset_elements();
    if !offset.is_multiple_of(2) {
        return Err(ElementwiseRuntimePartFourteenError::Invalid(
            "view_as_real gradient offset must be complex aligned",
        ));
    }
    let layout = if output_gradient.descriptor().layout() == Layout::Contiguous {
        Layout::Contiguous
    } else {
        Layout::Strided
    };
    let descriptor = TensorDescriptor::new_strided(
        shape,
        strides,
        offset / 2,
        dtype,
        layout,
        output_gradient.descriptor().device(),
        output_gradient.descriptor().stream(),
    )?;
    let gradient = output_gradient.reinterpret_read_only(descriptor)?;
    cancellation.check()?;
    Ok(gradient)
}

pub fn view_as_real_jvp_exact_native(
    input_tangent: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    view_as_real_exact_native(input_tangent, cancellation)
}

fn compare_scalars(left: DecodedScalar, right: DecodedScalar) -> Ordering {
    match (left, right) {
        (DecodedScalar::Boolean(left), DecodedScalar::Boolean(right)) => left.cmp(&right),
        (DecodedScalar::Signed(left), DecodedScalar::Signed(right)) => left.cmp(&right),
        (DecodedScalar::Unsigned(left), DecodedScalar::Unsigned(right)) => left.cmp(&right),
        (DecodedScalar::Real(left), DecodedScalar::Real(right)) => {
            match (left.is_nan(), right.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
            }
        }
        _ => Ordering::Equal,
    }
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartFourteenError> {
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ElementwiseRuntimePartFourteenError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        });
    }
    Ok(())
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, ElementwiseRuntimePartFourteenError> {
    if rank == 0 {
        return Err(ElementwiseRuntimePartFourteenError::Invalid(
            "operation requires a tensor dimension",
        ));
    }
    let rank = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartFourteenError::ShapeOverflow("rank"))?;
    let axis = if axis < 0 { rank + axis } else { axis };
    if !(0..rank).contains(&axis) {
        return Err(ElementwiseRuntimePartFourteenError::Invalid(
            "dimension is outside the tensor rank",
        ));
    }
    usize::try_from(axis).map_err(|_| ElementwiseRuntimePartFourteenError::ShapeOverflow("axis"))
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartFourteenError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(
                usize::try_from(*dimension).map_err(|_| {
                    ElementwiseRuntimePartFourteenError::ShapeOverflow("element count")
                })?,
            )
            .ok_or(ElementwiseRuntimePartFourteenError::ShapeOverflow(
                "element count",
            ))
    })
}

fn unravel_index(
    linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartFourteenError> {
    let mut remaining = linear;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(shape.len())
        .map_err(|_| ElementwiseRuntimePartFourteenError::ShapeOverflow("index"))?;
    indices.resize(shape.len(), 0_u64);
    for (axis, dimension) in shape.iter().enumerate().rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartFourteenError::ShapeOverflow("index"))?;
        if dimension == 0 {
            return Ok(indices);
        }
        indices[axis] = u64::try_from(remaining % dimension)
            .map_err(|_| ElementwiseRuntimePartFourteenError::ShapeOverflow("index"))?;
        remaining /= dimension;
    }
    Ok(indices)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartFourteenError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}

fn upload_i64(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    let byte_count =
        values
            .len()
            .checked_mul(8)
            .ok_or(ElementwiseRuntimePartFourteenError::ShapeOverflow(
                "i64 output",
            ))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for value in values {
        for byte in value.to_ne_bytes() {
            bytes.try_push(byte)?;
        }
    }
    upload_bytes(backend, shape, DType::I64, &bytes, context)
}

fn upload_bytes(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartFourteenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, context.stream)?;
    let (mut output, _) = backend.allocate(descriptor, context)?;
    {
        let mut write = output.write()?;
        write.bytes_mut()?.copy_from_slice(bytes);
    }
    context.check()?;
    Ok(output)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, ElementwiseRuntimePartFourteenError> {
    let mut values = backend.workspace_vec(context, count)?;
    for _ in 0..count {
        values.try_push(value)?;
    }
    Ok(values)
}
