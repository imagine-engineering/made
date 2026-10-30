use crate::{
    AutogradError, AutogradTape, BackendCapabilityMatrix, CancellationToken, CpuBackend,
    CpuWorkspaceVec, DType, DeviceId, ExecutionContext, LeafId, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError, UnaryOperation,
    generated_elementwise_or_runtime_operation_09::{
        ElementwiseRuntimePartNineError, clamp_jvp_with_context_exact_native,
        clamp_vjp_with_context_exact_native, clamp_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const REQUIRES_GRAD_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-BDAC082E9091";
pub const ATANH_OPERATION_ID: &str = "COMFY-TENSOR-OP-BA7930023140";
pub const CLIP_OPERATION_ID: &str = "COMFY-TENSOR-OP-BB442559BFF4";
pub const CUDA_IPC_COLLECT_OPERATION_ID: &str = "COMFY-TENSOR-OP-BBE4FD70D20E";
pub const CUDA_MEM_GET_INFO_OPERATION_ID: &str = "COMFY-TENSOR-OP-BF7BF3AA74D7";
pub const SDPA_KERNEL_OPERATION_ID: &str = "COMFY-TENSOR-OP-BB1114038F65";
pub const SPECTRAL_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-B96B1B025618";
pub const ROLL_OPERATION_ID: &str = "COMFY-TENSOR-OP-BD0C27F1B551";
pub const TENSOR_SPLIT_OPERATION_ID: &str = "COMFY-TENSOR-OP-BE1F415B5A74";
pub const XPU_IS_AVAILABLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-BF0B50BCC3B4";
pub const XPU_SYNCHRONIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-B91A910A5AF9";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorSplitSpec {
    Sections(u64),
    Size(u64),
    Sizes(Vec<u64>),
    Indices(Vec<i64>),
}

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartSeventeenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    PartNine(#[from] ElementwiseRuntimePartNineError),
    #[error("elementwise/runtime part-seventeen execution was cancelled")]
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

impl From<sim_types::CancellationError> for ElementwiseRuntimePartSeventeenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

type TemporaryVec<T> = CpuWorkspaceVec<T>;

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    _label: &'static str,
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartSeventeenError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

pub fn requires_grad_method_exact_native(
    tape: &mut AutogradTape,
    input: &Tensor,
    leaf: Option<LeafId>,
    requires_grad: bool,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    cancellation.check()?;
    tape.set_requires_grad(input, leaf, requires_grad, cancellation)?;
    Ok(input.clone())
}

pub fn atanh_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.cancellation.check()?;
    unary_forward(
        backend,
        UnaryOperation::ArcHyperbolicTangent,
        input,
        ATANH_OPERATION_ID,
        context,
    )
}

pub fn atanh_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.cancellation.check()?;
    unary_derivative(
        backend,
        input,
        output_gradient,
        ATANH_OPERATION_ID,
        |value, gradient| gradient / (1.0 - value * value),
        context,
    )
}

pub fn atanh_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.cancellation.check()?;
    atanh_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn clip_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<f32>,
    maximum: Option<f32>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.cancellation.check()?;
    Ok(clamp_with_context_exact_native(
        backend,
        input,
        minimum.map(|v| crate::Scalar::Float(f64::from(v))),
        maximum.map(|v| crate::Scalar::Float(f64::from(v))),
        context,
    )?)
}

pub fn clip_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<f32>,
    maximum: Option<f32>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
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

pub fn clip_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    minimum: Option<f32>,
    maximum: Option<f32>,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
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

pub fn roll_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    shifts: &[i64],
    dimensions: Option<&[i64]>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.cancellation.check()?;
    roll_impl(backend, input, shifts, dimensions, false, context)
}

pub fn roll_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    shifts: &[i64],
    dimensions: Option<&[i64]>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.cancellation.check()?;
    roll_impl(backend, output_gradient, shifts, dimensions, true, context)
}

pub fn roll_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    shifts: &[i64],
    dimensions: Option<&[i64]>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.cancellation.check()?;
    roll_with_context_exact_native(backend, input_tangent, shifts, dimensions, context)
}

pub fn tensor_split_exact_native(
    input: &Tensor,
    specification: &TensorSplitSpec,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ElementwiseRuntimePartSeventeenError> {
    cancellation.check()?;
    require_cpu(input, TENSOR_SPLIT_OPERATION_ID)?;
    let axis = normalize_axis(
        dimension,
        input.descriptor().rank(),
        TENSOR_SPLIT_OPERATION_ID,
    )?;
    let axis_size = *input.descriptor().shape().get(axis).ok_or(
        ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split axis"),
    )?;
    let segments = split_segments(axis_size, specification)?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(segments.len())
        .map_err(|_| ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split outputs"))?;
    for (index, (start, end)) in segments.into_iter().enumerate() {
        check_periodically(index, cancellation)?;
        let length =
            end.checked_sub(start)
                .ok_or(ElementwiseRuntimePartSeventeenError::ShapeOverflow(
                    "tensor split length",
                ))?;
        let start = i64::try_from(start).map_err(|_| {
            ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split start")
        })?;
        outputs.push(input.narrow_read_only(axis, start, length)?);
    }
    cancellation.check()?;
    Ok(outputs)
}

pub fn tensor_split_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradients: &[Tensor],
    specification: &TensorSplitSpec,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.cancellation.check()?;
    tensor_split_vjp_impl(
        backend,
        input,
        output_gradients,
        specification,
        dimension,
        context,
    )
}

fn tensor_split_vjp_impl(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradients: &[Tensor],
    specification: &TensorSplitSpec,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    require_f32_cpu(input, TENSOR_SPLIT_OPERATION_ID)?;
    let axis = normalize_axis(
        dimension,
        input.descriptor().rank(),
        TENSOR_SPLIT_OPERATION_ID,
    )?;
    let axis_size = *input.descriptor().shape().get(axis).ok_or(
        ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split VJP axis"),
    )?;
    let segments = split_segments(axis_size, specification)?;
    if output_gradients.len() != segments.len() {
        return invalid(
            TENSOR_SPLIT_OPERATION_ID,
            "tensor split gradient count does not match the output count",
        );
    }
    let count = element_count(input.descriptor().shape())?;
    let mut values = temporary_vec(backend, context, count, "tensor split VJP values")?;
    for index in 0..count {
        check_periodically(index, context.cancellation)?;
        values.try_push(0.0)?;
    }
    for (output_index, (gradient, (start, end))) in
        output_gradients.iter().zip(segments).enumerate()
    {
        check_periodically(output_index, context.cancellation)?;
        require_f32_cpu(gradient, TENSOR_SPLIT_OPERATION_ID)?;
        let mut expected_shape = input.descriptor().shape().to_vec();
        *expected_shape.get_mut(axis).ok_or(
            ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split VJP shape"),
        )? = end.saturating_sub(start);
        if gradient.descriptor().shape() != expected_shape
            || gradient.descriptor().stream() != input.descriptor().stream()
        {
            return invalid(
                TENSOR_SPLIT_OPERATION_ID,
                "tensor split output gradient descriptor is incompatible",
            );
        }
        for linear in 0..element_count(&expected_shape)? {
            check_periodically(linear, context.cancellation)?;
            let local_indices = unravel_index(linear, &expected_shape)?;
            let mut input_indices = local_indices.clone();
            let local_axis = *local_indices.get(axis).ok_or(
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split VJP local axis"),
            )?;
            *input_indices.get_mut(axis).ok_or(
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split VJP input axis"),
            )? = start.checked_add(local_axis).ok_or(
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split VJP input axis"),
            )?;
            let destination = ravel_index(&input_indices, input.descriptor().shape())?;
            *values.get_mut(destination).ok_or(
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split VJP destination"),
            )? += read_f32(gradient, &local_indices, TENSOR_SPLIT_OPERATION_ID)?;
        }
    }
    upload_f32(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn tensor_split_jvp_exact_native(
    input_tangent: &Tensor,
    specification: &TensorSplitSpec,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ElementwiseRuntimePartSeventeenError> {
    cancellation.check()?;
    tensor_split_exact_native(input_tangent, specification, dimension, cancellation)
}

pub fn xpu_synchronize_exact_native(
    backend: &dyn TensorBackend,
    capabilities: &BackendCapabilityMatrix,
    execution: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartSeventeenError> {
    execution.cancellation.check()?;
    Ok(crate::synchronize_device_exact_native(
        backend,
        capabilities,
        &[DeviceKind::Xpu],
        XPU_SYNCHRONIZE_OPERATION_ID,
        execution,
    )?)
}

fn roll_impl(
    backend: &CpuBackend,
    input: &Tensor,
    shifts: &[i64],
    dimensions: Option<&[i64]>,
    reverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    require_cpu(input, ROLL_OPERATION_ID)?;
    if shifts.is_empty() {
        return invalid(ROLL_OPERATION_ID, "roll requires at least one shift");
    }
    let shape = input.descriptor().shape();
    let count = element_count(shape)?;
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll dtype width"))?;
    let byte_count =
        count
            .checked_mul(width)
            .ok_or(ElementwiseRuntimePartSeventeenError::ShapeOverflow(
                "roll output bytes",
            ))?;
    let mut bytes = temporary_vec(backend, context, byte_count, "roll output bytes")?;
    match dimensions {
        None => {
            if shifts.len() != 1 {
                return invalid(
                    ROLL_OPERATION_ID,
                    "flattened roll requires exactly one shift",
                );
            }
            let count_i128 = i128::try_from(count).map_err(|_| {
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll element count")
            })?;
            for linear in 0..count {
                check_periodically(linear, context.cancellation)?;
                let source_linear = if count == 0 {
                    linear
                } else {
                    let shift = i128::from(shifts[0]) * if reverse { -1 } else { 1 };
                    let linear = i128::try_from(linear).map_err(|_| {
                        ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll index")
                    })?;
                    usize::try_from((linear - shift).rem_euclid(count_i128)).map_err(|_| {
                        ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll index")
                    })?
                };
                let source_indices = unravel_index(source_linear, shape)?;
                for byte in input.element_bytes(&source_indices)? {
                    bytes.try_push(*byte)?;
                }
            }
        }
        Some(dimensions) => {
            if shifts.len() != dimensions.len() {
                return invalid(
                    ROLL_OPERATION_ID,
                    "roll shifts and dimensions must have matching lengths",
                );
            }
            let mut normalized_shifts = vec![0_i128; shape.len()];
            for (shift, dimension) in shifts.iter().zip(dimensions) {
                let axis = normalize_axis(*dimension, shape.len(), ROLL_OPERATION_ID)?;
                let size = i128::from(*shape.get(axis).ok_or(
                    ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll axis"),
                )?);
                if size != 0 {
                    let signed_shift = i128::from(*shift) * if reverse { -1 } else { 1 };
                    let slot = normalized_shifts.get_mut(axis).ok_or(
                        ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll shift axis"),
                    )?;
                    *slot = (*slot + signed_shift).rem_euclid(size);
                }
            }
            for linear in 0..count {
                check_periodically(linear, context.cancellation)?;
                let output_indices = unravel_index(linear, shape)?;
                let mut source_indices = Vec::new();
                source_indices.try_reserve_exact(shape.len()).map_err(|_| {
                    ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll indices")
                })?;
                for (axis, (&index, &size)) in output_indices.iter().zip(shape).enumerate() {
                    let shift = *normalized_shifts.get(axis).ok_or(
                        ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll shift"),
                    )?;
                    let source = if size == 0 {
                        0
                    } else {
                        u64::try_from((i128::from(index) - shift).rem_euclid(i128::from(size)))
                            .map_err(|_| {
                                ElementwiseRuntimePartSeventeenError::ShapeOverflow("roll index")
                            })?
                    };
                    source_indices.push(source);
                }
                for byte in input.element_bytes(&source_indices)? {
                    bytes.try_push(*byte)?;
                }
            }
        }
    }
    upload_bytes(
        backend,
        shape,
        input.descriptor().dtype(),
        input.descriptor().stream(),
        &bytes,
        context,
    )
}

fn split_segments(
    size: u64,
    specification: &TensorSplitSpec,
) -> Result<Vec<(u64, u64)>, ElementwiseRuntimePartSeventeenError> {
    match specification {
        TensorSplitSpec::Sections(sections) => {
            if *sections == 0 {
                return invalid(
                    TENSOR_SPLIT_OPERATION_ID,
                    "tensor split sections must be nonzero",
                );
            }
            let section_count = usize::try_from(*sections).map_err(|_| {
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split sections")
            })?;
            let quotient = size / *sections;
            let remainder = size % *sections;
            let mut start = 0_u64;
            let mut segments = Vec::new();
            segments.try_reserve_exact(section_count).map_err(|_| {
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split segments")
            })?;
            for section in 0..*sections {
                let length = quotient + u64::from(section < remainder);
                let end = start.checked_add(length).ok_or(
                    ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split boundary"),
                )?;
                segments.push((start, end));
                start = end;
            }
            Ok(segments)
        }
        TensorSplitSpec::Size(split_size) => {
            if *split_size == 0 {
                return invalid(
                    TENSOR_SPLIT_OPERATION_ID,
                    "tensor split size must be nonzero",
                );
            }
            if size == 0 {
                return Ok(vec![(0, 0)]);
            }
            let segment_count = size
                .checked_add(split_size.saturating_sub(1))
                .ok_or(ElementwiseRuntimePartSeventeenError::ShapeOverflow(
                    "tensor split segment count",
                ))?
                / split_size;
            let segment_count = usize::try_from(segment_count).map_err(|_| {
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split segment count")
            })?;
            let mut segments = Vec::new();
            segments.try_reserve_exact(segment_count).map_err(|_| {
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split segments")
            })?;
            let mut start = 0_u64;
            while start < size {
                let end = start.saturating_add(*split_size).min(size);
                segments.push((start, end));
                start = end;
            }
            Ok(segments)
        }
        TensorSplitSpec::Sizes(sizes) => {
            if sizes.is_empty() {
                return invalid(
                    TENSOR_SPLIT_OPERATION_ID,
                    "tensor split sizes cannot be empty",
                );
            }
            let mut segments = Vec::new();
            segments.try_reserve_exact(sizes.len()).map_err(|_| {
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split segments")
            })?;
            let mut start = 0_u64;
            for length in sizes {
                let end = start.checked_add(*length).ok_or(
                    ElementwiseRuntimePartSeventeenError::ShapeOverflow(
                        "tensor split boundary",
                    ),
                )?;
                if end > size {
                    return invalid(
                        TENSOR_SPLIT_OPERATION_ID,
                        "tensor split sizes exceed the selected dimension",
                    );
                }
                segments.push((start, end));
                start = end;
            }
            if start != size {
                return invalid(
                    TENSOR_SPLIT_OPERATION_ID,
                    "tensor split sizes must sum to the selected dimension",
                );
            }
            Ok(segments)
        }
        TensorSplitSpec::Indices(indices) => {
            let mut segments = Vec::new();
            segments
                .try_reserve_exact(indices.len().saturating_add(1))
                .map_err(|_| {
                    ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split segments")
                })?;
            let mut start = 0_u64;
            for index in indices {
                let normalized = normalize_split_index(*index, size)?;
                segments.push((start, normalized.max(start)));
                start = normalized;
            }
            segments.push((start, size.max(start)));
            Ok(segments)
        }
    }
}

fn normalize_split_index(
    index: i64,
    size: u64,
) -> Result<u64, ElementwiseRuntimePartSeventeenError> {
    let size = i128::from(size);
    let index = i128::from(index);
    let normalized = if index < 0 { size + index } else { index }.clamp(0, size);
    u64::try_from(normalized)
        .map_err(|_| ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor split index"))
}

fn unary_forward(
    backend: &CpuBackend,
    operation: UnaryOperation,
    input: &Tensor,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    require_f32_cpu(input, operation_id)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.unary(operation, input, descriptor, context)?.0)
}

fn unary_derivative(
    backend: &CpuBackend,
    input: &Tensor,
    tangent: &Tensor,
    operation: &'static str,
    derivative: impl Fn(f32, f32) -> f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    require_f32_cpu(input, operation)?;
    require_f32_cpu(tangent, operation)?;
    if input.descriptor().shape() != tangent.descriptor().shape()
        || input.descriptor().stream() != tangent.descriptor().stream()
    {
        return invalid(
            operation,
            "derivative tensors must have matching descriptors",
        );
    }
    let count = element_count(input.descriptor().shape())?;
    let mut values = temporary_vec(backend, context, count, "unary derivative values")?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        values.try_push(derivative(
            read_f32(input, &indices, operation)?,
            read_f32(tangent, &indices, operation)?,
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

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
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
) -> Result<Tensor, ElementwiseRuntimePartSeventeenError> {
    context.check()?;
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    let (mut output, _) = backend.allocate(descriptor, context)?;
    let mut write = output.write()?;
    let destination = write.bytes_mut()?;
    if destination.len() != bytes.len() {
        return Err(ElementwiseRuntimePartSeventeenError::Tensor(
            TensorError::StorageLength {
                expected: u64::try_from(destination.len()).map_err(|_| {
                    ElementwiseRuntimePartSeventeenError::ShapeOverflow("upload bytes")
                })?,
                actual: u64::try_from(bytes.len()).map_err(|_| {
                    ElementwiseRuntimePartSeventeenError::ShapeOverflow("upload bytes")
                })?,
            },
        ));
    }
    destination.copy_from_slice(bytes);
    drop(write);
    context.check()?;
    Ok(output)
}

fn read_f32(
    tensor: &Tensor,
    indices: &[u64],
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartSeventeenError> {
    let bytes: [u8; 4] = tensor.element_bytes(indices)?.try_into().map_err(|_| {
        ElementwiseRuntimePartSeventeenError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        }
    })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSeventeenError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        Err(ElementwiseRuntimePartSeventeenError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        })
    } else {
        Ok(())
    }
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSeventeenError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimePartSeventeenError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn normalize_axis(
    dimension: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ElementwiseRuntimePartSeventeenError> {
    let rank_i64 = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor rank"))?;
    let normalized = if dimension < 0 {
        rank_i64.checked_add(dimension)
    } else {
        Some(dimension)
    }
    .filter(|dimension| *dimension >= 0 && *dimension < rank_i64)
    .ok_or_else(|| ElementwiseRuntimePartSeventeenError::Invalid {
        operation,
        reason: format!("dimension {dimension} is outside rank {rank}"),
    })?;
    usize::try_from(normalized)
        .map_err(|_| ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor axis"))
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartSeventeenError> {
    let count =
        shape.iter().try_fold(1_u64, |count, dimension| {
            count.checked_mul(*dimension).ok_or(
                ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor element count"),
            )
        })?;
    usize::try_from(count)
        .map_err(|_| ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor element count"))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartSeventeenError> {
    let mut indices = vec![0_u64; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dimension = usize::try_from(*shape.get(axis).ok_or(
            ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor index axis"),
        )?)
        .map_err(|_| ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor index axis"))?;
        if dimension == 0 {
            return invalid(ROLL_OPERATION_ID, "cannot unravel an empty tensor");
        }
        *indices
            .get_mut(axis)
            .ok_or(ElementwiseRuntimePartSeventeenError::ShapeOverflow(
                "tensor index axis",
            ))? = u64::try_from(linear % dimension).map_err(|_| {
            ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor index value")
        })?;
        linear /= dimension;
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
) -> Result<usize, ElementwiseRuntimePartSeventeenError> {
    if indices.len() != shape.len() {
        return Err(ElementwiseRuntimePartSeventeenError::ShapeOverflow(
            "tensor index rank",
        ));
    }
    let mut linear = 0_u64;
    for (&index, &dimension) in indices.iter().zip(shape) {
        if index >= dimension {
            return Err(ElementwiseRuntimePartSeventeenError::ShapeOverflow(
                "tensor index bounds",
            ));
        }
        linear = linear
            .checked_mul(dimension)
            .and_then(|value| value.checked_add(index))
            .ok_or(ElementwiseRuntimePartSeventeenError::ShapeOverflow(
                "tensor linear index",
            ))?;
    }
    usize::try_from(linear)
        .map_err(|_| ElementwiseRuntimePartSeventeenError::ShapeOverflow("tensor linear index"))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartSeventeenError> {
    if index.is_multiple_of(1_024) {
        cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, ElementwiseRuntimePartSeventeenError> {
    Err(ElementwiseRuntimePartSeventeenError::Invalid {
        operation,
        reason: reason.into(),
    })
}
