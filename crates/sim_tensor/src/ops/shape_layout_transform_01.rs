use std::collections::{BTreeMap, BTreeSet};

use crate::{
    CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId, ExecutionContext, Layout,
    Tensor, TensorBackend, TensorDescriptor, TensorError, ViewAccess,
    generated_elementwise_or_runtime_operation_17::{
        ElementwiseRuntimePartSeventeenError, TensorSplitSpec, tensor_split_exact_native,
        tensor_split_jvp_exact_native, tensor_split_vjp_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_20::{
        ElementwiseRuntimePartTwentyError, broadcast_tensor_jvp_exact_native,
        broadcast_tensor_vjp_with_context_exact_native, broadcast_view_to_shape_for_operation,
    },
    generated_external_tensor_kernel_02::{
        ExternalTensorKernelPartTwoError,
        einops_repeat_jvp_with_context_exact_native_for_operation,
        einops_repeat_vjp_with_context_exact_native_for_operation,
        einops_repeat_with_context_exact_native_for_operation,
    },
};
use thiserror::Error;

pub const EINOPS_REPEAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-71DB8F99EAAC";
pub const TENSOR_CHUNK_OPERATION_ID: &str = "COMFY-TENSOR-OP-5A4B8BBBFD81";
pub const TENSOR_EXPAND_OPERATION_ID: &str = "COMFY-TENSOR-OP-3D13DA91C9F3";
pub const TENSOR_EXPAND_AS_OPERATION_ID: &str = "COMFY-TENSOR-OP-25362A66A957";
pub const TENSOR_FLATTEN_OPERATION_ID: &str = "COMFY-TENSOR-OP-67D2FDD707E0";
pub const TENSOR_MOVEDIM_OPERATION_ID: &str = "COMFY-TENSOR-OP-73D179A8CEB9";
pub const TENSOR_REPEAT_INTERLEAVE_OPERATION_ID: &str = "COMFY-TENSOR-OP-3E6301EB6AA6";
pub const TENSOR_UNSQUEEZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-17E7C823A86F";
pub const TENSOR_VIEW_OPERATION_ID: &str = "COMFY-TENSOR-OP-5380FDF9E668";
pub const TORCH_CHUNK_OPERATION_ID: &str = "COMFY-TENSOR-OP-47B154B1D223";
pub const TORCH_REPEAT_INTERLEAVE_OPERATION_ID: &str = "COMFY-TENSOR-OP-0C2E0712DA68";
pub const TORCH_UNSQUEEZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-3E9A0E130935";

#[derive(Clone, Copy, Debug)]
pub enum RepeatInterleaveSpec<'a> {
    Scalar(u64),
    PerElement(&'a [u64]),
    Tensor(&'a Tensor),
}

#[derive(Clone, Copy, Debug)]
pub enum TensorViewSpec<'a> {
    Shape(&'a [i64]),
    DType(DType),
}

#[derive(Debug, Error)]
pub enum ShapeLayoutTransformPartOneError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Split(#[from] ElementwiseRuntimePartSeventeenError),
    #[error(transparent)]
    Broadcast(#[from] ElementwiseRuntimePartTwentyError),
    #[error(transparent)]
    Einops(#[from] ExternalTensorKernelPartTwoError),
    #[error("shape/layout-transform part-one execution was cancelled")]
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
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed for operation {operation} while computing {subject}")]
    ShapeOverflow {
        operation: &'static str,
        subject: &'static str,
    },
}

impl From<sim_types::CancellationError> for ShapeLayoutTransformPartOneError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn einops_repeat_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    Ok(einops_repeat_with_context_exact_native_for_operation(
        backend,
        input,
        pattern,
        axis_lengths,
        EINOPS_REPEAT_OPERATION_ID,
        context,
    )?)
}

pub fn einops_repeat_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    Ok(einops_repeat_vjp_with_context_exact_native_for_operation(
        backend,
        output_gradient,
        input_shape,
        pattern,
        axis_lengths,
        EINOPS_REPEAT_OPERATION_ID,
        context,
    )?)
}

pub fn einops_repeat_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    pattern: &str,
    axis_lengths: &BTreeMap<String, u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    Ok(einops_repeat_jvp_with_context_exact_native_for_operation(
        backend,
        input_tangent,
        pattern,
        axis_lengths,
        EINOPS_REPEAT_OPERATION_ID,
        context,
    )?)
}

pub fn tensor_chunk_exact_native(
    input: &Tensor,
    chunks: u64,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartOneError> {
    chunk_for_operation(input, chunks, dimension, TENSOR_CHUNK_OPERATION_ID, cancellation)
}

pub fn torch_chunk_exact_native(
    input: &Tensor,
    chunks: u64,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartOneError> {
    chunk_for_operation(input, chunks, dimension, TORCH_CHUNK_OPERATION_ID, cancellation)
}

pub fn chunk_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradients: &[Tensor],
    chunks: u64,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    let specification = chunk_specification(input, chunks, dimension, TENSOR_CHUNK_OPERATION_ID)?;
    Ok(tensor_split_vjp_with_context_exact_native(
        backend,
        input,
        output_gradients,
        &specification,
        dimension,
        context,
    )?)
}

pub fn chunk_jvp_exact_native(
    input_tangent: &Tensor,
    chunks: u64,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    let specification =
        chunk_specification(input_tangent, chunks, dimension, TENSOR_CHUNK_OPERATION_ID)?;
    Ok(tensor_split_jvp_exact_native(
        input_tangent,
        &specification,
        dimension,
        cancellation,
    )?)
}

pub fn tensor_expand_exact_native(
    input: &Tensor,
    sizes: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    expand_for_operation(input, sizes, TENSOR_EXPAND_OPERATION_ID, cancellation)
}

pub fn tensor_expand_as_exact_native(
    input: &Tensor,
    other: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    if input.descriptor().device() != other.descriptor().device()
        || input.descriptor().stream() != other.descriptor().stream()
    {
        return Err(ShapeLayoutTransformPartOneError::UnsupportedDevice {
            operation: TENSOR_EXPAND_AS_OPERATION_ID,
            device: other.descriptor().device(),
        });
    }
    let sizes = other
        .descriptor()
        .shape()
        .iter()
        .map(|dimension| {
            i64::try_from(*dimension).map_err(|_| overflow(TENSOR_EXPAND_AS_OPERATION_ID, "shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    expand_for_operation(input, &sizes, TENSOR_EXPAND_AS_OPERATION_ID, cancellation)
}

pub fn expand_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    Ok(broadcast_tensor_vjp_with_context_exact_native(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn expand_jvp_exact_native(
    input_tangent: &Tensor,
    output_shape: &[u64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    Ok(broadcast_tensor_jvp_exact_native(
        input_tangent,
        output_shape,
        cancellation,
    )?)
}

pub fn tensor_flatten_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    start_dimension: i64,
    end_dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    flatten_with_context_exact_native_for_operation(
        backend,
        input,
        start_dimension,
        end_dimension,
        TENSOR_FLATTEN_OPERATION_ID,
        context,
    )
}

pub(crate) fn flatten_with_context_exact_native_for_operation(
    backend: &CpuBackend,
    input: &Tensor,
    start_dimension: i64,
    end_dimension: i64,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    let target_shape = flatten_shape(
        input.descriptor().shape(),
        start_dimension,
        end_dimension,
        operation,
    )?;
    reshape_to_shape_with_context_for_operation(backend, input, target_shape, operation, context)
}

pub fn flatten_vjp_exact_native(
    output_gradient: &Tensor,
    input_shape: &[u64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    view_with_shape_for_operation(
        output_gradient,
        &shape_as_i64(input_shape, TENSOR_FLATTEN_OPERATION_ID)?,
        TENSOR_FLATTEN_OPERATION_ID,
        cancellation,
    )
}

pub fn tensor_movedim_exact_native(
    input: &Tensor,
    source: &[i64],
    destination: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    permute_moved_dimensions(
        input,
        source,
        destination,
        TENSOR_MOVEDIM_OPERATION_ID,
        cancellation,
    )
}

pub fn movedim_vjp_exact_native(
    output_gradient: &Tensor,
    source: &[i64],
    destination: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    permute_moved_dimensions(
        output_gradient,
        destination,
        source,
        TENSOR_MOVEDIM_OPERATION_ID,
        cancellation,
    )
}

pub fn movedim_jvp_exact_native(
    input_tangent: &Tensor,
    source: &[i64],
    destination: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    tensor_movedim_exact_native(input_tangent, source, destination, cancellation)
}

pub fn tensor_repeat_interleave_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: RepeatInterleaveSpec<'_>,
    dimension: Option<i64>,
    output_size: Option<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    repeat_interleave_forward(
        backend,
        input,
        repeats,
        dimension,
        output_size,
        TENSOR_REPEAT_INTERLEAVE_OPERATION_ID,
        context,
    )
}

pub fn torch_repeat_interleave_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: RepeatInterleaveSpec<'_>,
    dimension: Option<i64>,
    output_size: Option<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    repeat_interleave_forward(
        backend,
        input,
        repeats,
        dimension,
        output_size,
        TORCH_REPEAT_INTERLEAVE_OPERATION_ID,
        context,
    )
}

pub fn repeat_interleave_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: RepeatInterleaveSpec<'_>,
    dimension: Option<i64>,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    require_f32_cpu(input, TENSOR_REPEAT_INTERLEAVE_OPERATION_ID)?;
    require_f32_cpu(output_gradient, TENSOR_REPEAT_INTERLEAVE_OPERATION_ID)?;
    let plan = repeat_interleave_plan(
        input,
        repeats,
        dimension,
        None,
        TENSOR_REPEAT_INTERLEAVE_OPERATION_ID,
    )?;
    if output_gradient.descriptor().shape() != plan.output_shape
        || output_gradient.descriptor().stream() != input.descriptor().stream()
    {
        return invalid(
            TENSOR_REPEAT_INTERLEAVE_OPERATION_ID,
            "repeat-interleave gradient descriptor does not match the output",
        );
    }
    let count = element_count(input.descriptor().shape(), TENSOR_REPEAT_INTERLEAVE_OPERATION_ID)?;
    let mut values = backend.workspace_vec(context, count)?;
    for _ in 0..count {
        values.try_push(0.0_f32)?;
    }
    for (output_linear, source_linear) in plan.source_linear_indices.iter().enumerate() {
        check_periodically(output_linear, context.cancellation)?;
        let source = values.get_mut(*source_linear).ok_or_else(|| {
            invalid_error(
                TENSOR_REPEAT_INTERLEAVE_OPERATION_ID,
                "repeat-interleave gradient source is outside input",
            )
        })?;
        let bytes: [u8; 4] = output_gradient
            .linear_element_bytes(u64::try_from(output_linear).map_err(|_| {
                overflow(TENSOR_REPEAT_INTERLEAVE_OPERATION_ID, "gradient index")
            })?)?
            .try_into()
            .map_err(|_| {
                invalid_error(
                    TENSOR_REPEAT_INTERLEAVE_OPERATION_ID,
                    "F32 gradient has invalid element width",
                )
            })?;
        *source += f32::from_ne_bytes(bytes);
    }
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.upload_f32(descriptor, &values, context)?.0)
}

pub fn repeat_interleave_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    repeats: RepeatInterleaveSpec<'_>,
    dimension: Option<i64>,
    output_size: Option<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    tensor_repeat_interleave_with_context_exact_native(
        backend,
        input_tangent,
        repeats,
        dimension,
        output_size,
        context,
    )
}

pub fn tensor_unsqueeze_exact_native(
    input: &Tensor,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    unsqueeze_for_operation(input, dimension, TENSOR_UNSQUEEZE_OPERATION_ID, cancellation)
}

pub fn torch_unsqueeze_exact_native(
    input: &Tensor,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    unsqueeze_for_operation(input, dimension, TORCH_UNSQUEEZE_OPERATION_ID, cancellation)
}

pub fn unsqueeze_vjp_exact_native(
    output_gradient: &Tensor,
    input_shape: &[u64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    view_with_shape_for_operation(
        output_gradient,
        &shape_as_i64(input_shape, TENSOR_UNSQUEEZE_OPERATION_ID)?,
        TENSOR_UNSQUEEZE_OPERATION_ID,
        cancellation,
    )
}

pub fn tensor_view_exact_native(
    input: &Tensor,
    specification: TensorViewSpec<'_>,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    match specification {
        TensorViewSpec::Shape(shape) => {
            view_with_shape_for_operation(input, shape, TENSOR_VIEW_OPERATION_ID, cancellation)
        }
        TensorViewSpec::DType(dtype) => {
            cancellation.check()?;
            let descriptor = input.descriptor().reinterpreted_dtype_view(dtype).map_err(|error| {
                if matches!(error, TensorError::NonContiguousAccess) {
                    invalid_error(
                        TENSOR_VIEW_OPERATION_ID,
                        "dtype view requires a last-axis stride of one and aligned shape, strides, and storage offset",
                    )
                } else {
                    ShapeLayoutTransformPartOneError::Tensor(error)
                }
            })?;
            let output = input.reinterpret_read_only(descriptor)?;
            cancellation.check()?;
            Ok(output)
        }
    }
}

pub fn view_vjp_exact_native(
    output_gradient: &Tensor,
    input_shape: &[u64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    view_with_shape_for_operation(
        output_gradient,
        &shape_as_i64(input_shape, TENSOR_VIEW_OPERATION_ID)?,
        TENSOR_VIEW_OPERATION_ID,
        cancellation,
    )
}

struct RepeatInterleavePlan {
    output_shape: Vec<u64>,
    source_linear_indices: Vec<usize>,
}

fn chunk_for_operation(
    input: &Tensor,
    chunks: u64,
    dimension: i64,
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    let specification = chunk_specification(input, chunks, dimension, operation)?;
    Ok(tensor_split_exact_native(
        input,
        &specification,
        dimension,
        cancellation,
    )?)
}

fn chunk_specification(
    input: &Tensor,
    chunks: u64,
    dimension: i64,
    operation: &'static str,
) -> Result<TensorSplitSpec, ShapeLayoutTransformPartOneError> {
    if chunks == 0 {
        return invalid(operation, "chunk count must be greater than zero");
    }
    let axis = normalize_axis(dimension, input.descriptor().rank(), operation)?;
    let size = input.descriptor().shape()[axis];
    if size == 0 {
        return Ok(TensorSplitSpec::Sections(chunks));
    }
    let chunk_size = size
        .checked_add(chunks - 1)
        .ok_or_else(|| overflow(operation, "chunk size"))?
        / chunks;
    let output_count = size
        .checked_add(chunk_size - 1)
        .ok_or_else(|| overflow(operation, "chunk output count"))?
        / chunk_size;
    let mut indices = Vec::new();
    let index_count = usize::try_from(output_count.saturating_sub(1))
        .map_err(|_| overflow(operation, "chunk split-index count"))?;
    indices
        .try_reserve_exact(index_count)
        .map_err(|_| overflow(operation, "chunk split-index allocation"))?;
    for output in 1..output_count {
        indices.push(i64::try_from(output.checked_mul(chunk_size).ok_or_else(|| {
            overflow(operation, "chunk split index")
        })?)
        .map_err(|_| overflow(operation, "chunk split index"))?);
    }
    Ok(TensorSplitSpec::Indices(indices))
}

fn expand_for_operation(
    input: &Tensor,
    sizes: &[i64],
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    if sizes.len() < input.descriptor().rank() {
        return invalid(operation, "expanded rank cannot be smaller than input rank");
    }
    let rank_offset = sizes.len() - input.descriptor().rank();
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(sizes.len())
        .map_err(|_| overflow(operation, "expanded shape allocation"))?;
    for (axis, size) in sizes.iter().enumerate() {
        if *size == -1 {
            if axis < rank_offset {
                return invalid(operation, "-1 cannot define a new leading expanded axis");
            }
            shape.push(input.descriptor().shape()[axis - rank_offset]);
        } else if *size < 0 {
            return invalid(operation, "expanded dimensions must be non-negative or -1");
        } else {
            shape.push(u64::try_from(*size).map_err(|_| overflow(operation, "expanded shape"))?);
        }
    }
    let output = broadcast_view_to_shape_for_operation(input, &shape, operation)?;
    cancellation.check()?;
    Ok(output)
}

fn flatten_shape(
    shape: &[u64],
    start_dimension: i64,
    end_dimension: i64,
    operation: &'static str,
) -> Result<Vec<u64>, ShapeLayoutTransformPartOneError> {
    if shape.is_empty() {
        if !matches!(start_dimension, 0 | -1) || !matches!(end_dimension, 0 | -1) {
            return invalid(operation, "scalar flatten dimensions must address axis zero");
        }
        return Ok(vec![1]);
    }
    let start = normalize_axis(start_dimension, shape.len(), operation)?;
    let end = normalize_axis(end_dimension, shape.len(), operation)?;
    if start > end {
        return invalid(operation, "flatten start dimension exceeds end dimension");
    }
    let merged = shape[start..=end].iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| overflow(operation, "flattened extent"))
    })?;
    let mut output = Vec::new();
    output.extend_from_slice(&shape[..start]);
    output.push(merged);
    output.extend_from_slice(&shape[end + 1..]);
    Ok(output)
}

pub(crate) fn permute_moved_dimensions(
    input: &Tensor,
    source: &[i64],
    destination: &[i64],
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    if source.len() != destination.len() {
        return invalid(operation, "movedim source and destination lengths differ");
    }
    let rank = input.descriptor().rank();
    let sources = source
        .iter()
        .map(|axis| normalize_axis(*axis, rank, operation))
        .collect::<Result<Vec<_>, _>>()?;
    let destinations = destination
        .iter()
        .map(|axis| normalize_axis(*axis, rank, operation))
        .collect::<Result<Vec<_>, _>>()?;
    if sources.iter().copied().collect::<BTreeSet<_>>().len() != sources.len()
        || destinations.iter().copied().collect::<BTreeSet<_>>().len() != destinations.len()
    {
        return invalid(operation, "movedim axes must be unique");
    }
    let source_set = sources.iter().copied().collect::<BTreeSet<_>>();
    let mut permutation = (0..rank)
        .filter(|axis| !source_set.contains(axis))
        .collect::<Vec<_>>();
    let mut insertions = destinations
        .iter()
        .copied()
        .zip(sources.iter().copied())
        .collect::<Vec<_>>();
    insertions.sort_by_key(|(destination, _)| *destination);
    for (destination, source) in insertions {
        permutation.insert(destination, source);
    }
    let descriptor = input.descriptor().permuted_view(&permutation)?;
    let output = input.view(descriptor, ViewAccess::ReadOnly)?;
    cancellation.check()?;
    Ok(output)
}

fn repeat_interleave_forward(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: RepeatInterleaveSpec<'_>,
    dimension: Option<i64>,
    output_size: Option<u64>,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    require_cpu(input, operation)?;
    let plan = repeat_interleave_plan(input, repeats, dimension, output_size, operation)?;
    let width = usize::try_from(input.descriptor().dtype().byte_width())
        .map_err(|_| overflow(operation, "repeat-interleave dtype width"))?;
    let capacity = plan
        .source_linear_indices
        .len()
        .checked_mul(width)
        .ok_or_else(|| overflow(operation, "repeat-interleave bytes"))?;
    let mut bytes = backend.workspace_vec(context, capacity)?;
    for (output_linear, source_linear) in plan.source_linear_indices.iter().enumerate() {
        check_periodically(output_linear, context.cancellation)?;
        for byte in input
            .linear_element_bytes(
                u64::try_from(*source_linear)
                    .map_err(|_| overflow(operation, "repeat-interleave source"))?,
            )?
            .iter()
            .copied()
        {
            bytes.try_push(byte)?;
        }
    }
    let descriptor = TensorDescriptor::contiguous(
        plan.output_shape,
        input.descriptor().dtype(),
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn repeat_interleave_plan(
    input: &Tensor,
    repeats: RepeatInterleaveSpec<'_>,
    dimension: Option<i64>,
    output_size: Option<u64>,
    operation: &'static str,
) -> Result<RepeatInterleavePlan, ShapeLayoutTransformPartOneError> {
    let input_shape = input.descriptor().shape();
    let (logical_shape, axis) = match dimension {
        Some(dimension) => (input_shape.to_vec(), normalize_axis(dimension, input_shape.len(), operation)?),
        None => (vec![u64::try_from(element_count(input_shape, operation)?)
            .map_err(|_| overflow(operation, "flattened repeat-interleave shape"))?], 0),
    };
    let axis_size = logical_shape[axis];
    let repeat_counts = match repeats {
        RepeatInterleaveSpec::Scalar(repeat) => {
            let count = usize::try_from(axis_size)
                .map_err(|_| overflow(operation, "repeat-interleave axis"))?;
            vec![repeat; count]
        }
        RepeatInterleaveSpec::PerElement(values) => {
            if values.len()
                != usize::try_from(axis_size)
                    .map_err(|_| overflow(operation, "repeat-interleave axis"))?
            {
                return invalid(operation, "per-element repeats length must match the selected axis");
            }
            values.to_vec()
        }
        RepeatInterleaveSpec::Tensor(repeats) => {
            repeat_counts_from_tensor(input, repeats, axis_size, operation)?
        }
    };
    let repeated_size = repeat_counts.iter().try_fold(0_u64, |sum, repeat| {
        sum.checked_add(*repeat)
            .ok_or_else(|| overflow(operation, "repeat-interleave output axis"))
    })?;
    if output_size.is_some_and(|expected| expected != repeated_size) {
        return invalid(operation, "output_size does not match the sum of repeats");
    }
    let mut output_shape = logical_shape.clone();
    output_shape[axis] = repeated_size;
    let output_count = element_count(&output_shape, operation)?;
    let mut source_linear_indices = Vec::new();
    source_linear_indices
        .try_reserve_exact(output_count)
        .map_err(|_| overflow(operation, "repeat-interleave source allocation"))?;
    let mut expanded_axis = Vec::new();
    expanded_axis
        .try_reserve_exact(
            usize::try_from(repeated_size)
                .map_err(|_| overflow(operation, "repeat-interleave axis allocation"))?,
        )
        .map_err(|_| overflow(operation, "repeat-interleave axis allocation"))?;
    for (source, repeat) in repeat_counts.iter().enumerate() {
        for _ in 0..*repeat {
            expanded_axis.push(u64::try_from(source)
                .map_err(|_| overflow(operation, "repeat-interleave axis index"))?);
        }
    }
    for output_linear in 0..output_count {
        let mut indices = unravel_index(output_linear, &output_shape, operation)?;
        let expanded = usize::try_from(indices[axis])
            .map_err(|_| overflow(operation, "repeat-interleave output coordinate"))?;
        indices[axis] = *expanded_axis.get(expanded).ok_or_else(|| {
            invalid_error(operation, "repeat-interleave output coordinate is outside axis")
        })?;
        source_linear_indices.push(ravel_index(&indices, &logical_shape, operation)?);
    }
    Ok(RepeatInterleavePlan {
        output_shape,
        source_linear_indices,
    })
}

fn repeat_counts_from_tensor(
    input: &Tensor,
    repeats: &Tensor,
    axis_size: u64,
    operation: &'static str,
) -> Result<Vec<u64>, ShapeLayoutTransformPartOneError> {
    if repeats.descriptor().device() != DeviceId::CPU {
        return Err(ShapeLayoutTransformPartOneError::UnsupportedDevice {
            operation,
            device: repeats.descriptor().device(),
        });
    }
    if repeats.descriptor().stream() != input.descriptor().stream() {
        return invalid(operation, "repeat counts must use the input tensor stream");
    }
    if repeats.descriptor().rank() > 1 {
        return invalid(operation, "repeat counts tensor must be scalar or one-dimensional");
    }
    if !matches!(repeats.descriptor().dtype(), DType::I64 | DType::I32) {
        return Err(ShapeLayoutTransformPartOneError::UnsupportedDType {
            operation,
            dtype: repeats.descriptor().dtype(),
        });
    }

    let count = usize::try_from(repeats.descriptor().element_count()?)
        .map_err(|_| overflow(operation, "repeat counts length"))?;
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(count)
        .map_err(|_| overflow(operation, "repeat counts allocation"))?;
    for linear in 0..count {
        let bytes = repeats.linear_element_bytes(
            u64::try_from(linear).map_err(|_| overflow(operation, "repeat count index"))?,
        )?;
        let repeat = match repeats.descriptor().dtype().decode_scalar(bytes)? {
            DecodedScalar::Signed(value) if value >= 0 => u64::try_from(value)
                .map_err(|_| overflow(operation, "signed repeat count"))?,
            DecodedScalar::Signed(_) => {
                return invalid(operation, "repeat counts cannot be negative");
            }
            _ => {
                return Err(ShapeLayoutTransformPartOneError::UnsupportedDType {
                    operation,
                    dtype: repeats.descriptor().dtype(),
                });
            }
        };
        decoded.push(repeat);
    }

    if repeats.descriptor().rank() == 0 {
        let repeat = decoded
            .first()
            .copied()
            .ok_or_else(|| invalid_error(operation, "scalar repeat count is missing"))?;
        return Ok(vec![
            repeat;
            usize::try_from(axis_size)
                .map_err(|_| overflow(operation, "repeat-interleave axis"))?
        ]);
    }
    if decoded.len()
        != usize::try_from(axis_size)
            .map_err(|_| overflow(operation, "repeat-interleave axis"))?
    {
        return invalid(
            operation,
            "repeat counts tensor length must match the selected axis",
        );
    }
    Ok(decoded)
}

pub(crate) fn unsqueeze_for_operation(
    input: &Tensor,
    dimension: i64,
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    let axis = normalize_insertion_axis(dimension, input.descriptor().rank(), operation)?;
    let stride = if axis >= input.descriptor().rank() {
        1
    } else {
        i128::from(input.descriptor().strides()[axis])
            .checked_mul(i128::from(input.descriptor().shape()[axis]))
            .and_then(|value| i64::try_from(value).ok())
            .ok_or_else(|| overflow(operation, "unsqueeze stride"))?
    };
    let mut shape = input.descriptor().shape().to_vec();
    let mut strides = input.descriptor().strides().to_vec();
    shape.insert(axis, 1);
    strides.insert(axis, stride);
    let descriptor = TensorDescriptor::new_strided(
        shape,
        strides,
        input.descriptor().offset_elements(),
        input.descriptor().dtype(),
        Layout::Strided,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let output = input.view(descriptor, ViewAccess::ReadOnly)?;
    cancellation.check()?;
    Ok(output)
}

pub(crate) fn view_with_shape_for_operation(
    input: &Tensor,
    shape: &[i64],
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    cancellation.check()?;
    let shape = infer_shape(input.descriptor().element_count()?, shape, operation)?;
    let descriptor = input.descriptor().reshaped_view(shape).map_err(|error| {
        if matches!(error, TensorError::NonContiguousAccess) {
            invalid_error(operation, "requested view is incompatible with input strides")
        } else {
            ShapeLayoutTransformPartOneError::Tensor(error)
        }
    })?;
    let output = input.view(descriptor, ViewAccess::ReadOnly)?;
    cancellation.check()?;
    Ok(output)
}

pub(crate) fn reshape_with_context_for_operation(
    backend: &CpuBackend,
    input: &Tensor,
    shape: &[i64],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    context.cancellation.check()?;
    let shape = infer_shape(input.descriptor().element_count()?, shape, operation)?;
    reshape_to_shape_with_context_for_operation(backend, input, shape, operation, context)
}

fn reshape_to_shape_with_context_for_operation(
    backend: &CpuBackend,
    input: &Tensor,
    target_shape: Vec<u64>,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartOneError> {
    match input.descriptor().reshaped_view(target_shape.clone()) {
        Ok(descriptor) => Ok(input.view(descriptor, ViewAccess::ReadOnly)?),
        Err(TensorError::NonContiguousAccess) => {
            require_cpu(input, operation)?;
            let descriptor = TensorDescriptor::contiguous(
                input.descriptor().shape().to_vec(),
                input.descriptor().dtype(),
                DeviceId::CPU,
                input.descriptor().stream(),
            )?;
            let contiguous = backend.copy(input, descriptor, context)?.0;
            let descriptor = contiguous.descriptor().reshaped_view(target_shape)?;
            Ok(contiguous.view(descriptor, ViewAccess::ReadOnly)?)
        }
        Err(error) => Err(error.into()),
    }
}

fn infer_shape(
    element_count: u64,
    shape: &[i64],
    operation: &'static str,
) -> Result<Vec<u64>, ShapeLayoutTransformPartOneError> {
    let mut inferred = None;
    let mut known = 1_u64;
    let mut output = Vec::new();
    output
        .try_reserve_exact(shape.len())
        .map_err(|_| overflow(operation, "view shape allocation"))?;
    for (axis, dimension) in shape.iter().enumerate() {
        if *dimension == -1 {
            if inferred.replace(axis).is_some() {
                return invalid(operation, "only one view dimension may be inferred");
            }
            output.push(1);
        } else if *dimension < 0 {
            return invalid(operation, "view dimensions must be non-negative or -1");
        } else {
            let dimension =
                u64::try_from(*dimension).map_err(|_| overflow(operation, "view shape"))?;
            known = known
                .checked_mul(dimension)
                .ok_or_else(|| overflow(operation, "view element count"))?;
            output.push(dimension);
        }
    }
    if let Some(axis) = inferred {
        if known == 0 || !element_count.is_multiple_of(known) {
            return invalid(operation, "inferred view dimension is ambiguous or non-integral");
        }
        output[axis] = element_count / known;
    } else if known != element_count {
        return invalid(operation, "view shape changes the tensor element count");
    }
    Ok(output)
}

fn shape_as_i64(
    shape: &[u64],
    operation: &'static str,
) -> Result<Vec<i64>, ShapeLayoutTransformPartOneError> {
    shape
        .iter()
        .map(|dimension| i64::try_from(*dimension).map_err(|_| overflow(operation, "shape")))
        .collect()
}

fn normalize_axis(
    axis: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ShapeLayoutTransformPartOneError> {
    if rank == 0 {
        return invalid(operation, "operation requires a tensor axis");
    }
    let rank_i64 = i64::try_from(rank).map_err(|_| overflow(operation, "rank"))?;
    let normalized = if axis < 0 {
        rank_i64
            .checked_add(axis)
            .ok_or_else(|| overflow(operation, "axis"))?
    } else {
        axis
    };
    if normalized < 0 || normalized >= rank_i64 {
        return invalid(operation, format!("axis {axis} is outside rank {rank}"));
    }
    usize::try_from(normalized).map_err(|_| overflow(operation, "axis"))
}

fn normalize_insertion_axis(
    axis: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ShapeLayoutTransformPartOneError> {
    let output_rank = rank
        .checked_add(1)
        .ok_or_else(|| overflow(operation, "output rank"))?;
    let output_rank_i64 =
        i64::try_from(output_rank).map_err(|_| overflow(operation, "output rank"))?;
    let normalized = if axis < 0 {
        output_rank_i64
            .checked_add(axis)
            .ok_or_else(|| overflow(operation, "insertion axis"))?
    } else {
        axis
    };
    if normalized < 0 || normalized >= output_rank_i64 {
        return invalid(operation, format!("axis {axis} is outside insertion rank {output_rank}"));
    }
    usize::try_from(normalized).map_err(|_| overflow(operation, "insertion axis"))
}

fn element_count(
    shape: &[u64],
    operation: &'static str,
) -> Result<usize, ShapeLayoutTransformPartOneError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(
                usize::try_from(*dimension)
                    .map_err(|_| overflow(operation, "dimension conversion"))?,
            )
            .ok_or_else(|| overflow(operation, "element count"))
    })
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
    operation: &'static str,
) -> Result<Vec<u64>, ShapeLayoutTransformPartOneError> {
    let mut indices = vec![0_u64; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dimension = usize::try_from(shape[axis])
            .map_err(|_| overflow(operation, "index dimension"))?;
        if dimension == 0 {
            return invalid(operation, "cannot index an empty tensor");
        }
        indices[axis] = u64::try_from(linear % dimension)
            .map_err(|_| overflow(operation, "index coordinate"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn ravel_index(
    indices: &[u64],
    shape: &[u64],
    operation: &'static str,
) -> Result<usize, ShapeLayoutTransformPartOneError> {
    if indices.len() != shape.len() {
        return invalid(operation, "index rank does not match shape rank");
    }
    let mut linear = 0_u64;
    for (index, dimension) in indices.iter().zip(shape) {
        if index >= dimension {
            return invalid(operation, "index is outside tensor shape");
        }
        linear = linear
            .checked_mul(*dimension)
            .and_then(|value| value.checked_add(*index))
            .ok_or_else(|| overflow(operation, "linear index"))?;
    }
    usize::try_from(linear).map_err(|_| overflow(operation, "linear index"))
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartOneError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(ShapeLayoutTransformPartOneError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        });
    }
    Ok(())
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartOneError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(ShapeLayoutTransformPartOneError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ShapeLayoutTransformPartOneError> {
    if index.is_multiple_of(64) {
        cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, ShapeLayoutTransformPartOneError> {
    Err(invalid_error(operation, reason))
}

fn invalid_error(
    operation: &'static str,
    reason: impl Into<String>,
) -> ShapeLayoutTransformPartOneError {
    ShapeLayoutTransformPartOneError::Invalid {
        operation,
        reason: reason.into(),
    }
}

fn overflow(
    operation: &'static str,
    subject: &'static str,
) -> ShapeLayoutTransformPartOneError {
    ShapeLayoutTransformPartOneError::ShapeOverflow { operation, subject }
}

#[cfg(test)]
mod validation_tests {
    use std::collections::BTreeMap;

    #[test]
    fn writes_task_validation_artifacts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture_digests = BTreeMap::from([
            ("COMFY-TENSOR-OP-0C2E0712DA68", "84b24f30f0b0b1aefb368e128a16ffcd880ba5d587e4d161f9742c57a2b39b3a"),
            ("COMFY-TENSOR-OP-17E7C823A86F", "f2de2de77480b9fa9d349674270e7be579887f884ceaf9ed1fe8d930fda7c8d6"),
            ("COMFY-TENSOR-OP-25362A66A957", "5d0f64b9b9d00a2da59ecb9b43afe263caf476071306d9b5c97ea8627b70e65f"),
            ("COMFY-TENSOR-OP-3D13DA91C9F3", "bc81a0a28a435a4a28d9e7edf5276cf0fecc6bcee16a75d8d34a07071cdcd47b"),
            ("COMFY-TENSOR-OP-3E6301EB6AA6", "862704fe9c791394baf13575131b58885b8245774e55e9c03216d2bdb7e55704"),
            ("COMFY-TENSOR-OP-3E9A0E130935", "62255ce6f5ec3e7770d4f702ecbbb59c7d6476038a5193d41c6db6726ae29406"),
            ("COMFY-TENSOR-OP-47B154B1D223", "4079d4531c598041052a01daad18d4b0475ead0fdd89d82edb73169711245357"),
            ("COMFY-TENSOR-OP-5380FDF9E668", "3b1130a82a1dd45075bc3fb5a6a5af096b9d5c9267e680869fecfc406d8e47dc"),
            ("COMFY-TENSOR-OP-5A4B8BBBFD81", "e993b80c337aa65e218a602bde50dc8270997de18c0f50e4cb20729f0856b29e"),
            ("COMFY-TENSOR-OP-67D2FDD707E0", "bc520ef02e159deb24e772bda7642e9cba3988c1364293ca073761b66e44666b"),
            ("COMFY-TENSOR-OP-71DB8F99EAAC", "158c87a3850b220fc65402bab95cd18db4c7c69e0728f176843edf2015b3e4cb"),
            ("COMFY-TENSOR-OP-73D179A8CEB9", "b611d45a63bb8c764957543880888bdcdd27ee2566b5606f4e3e09e6f9b0688d"),
        ]);
        let cases = fixture_digests
            .keys()
            .map(|operation| (*operation, true))
            .collect::<BTreeMap<_, _>>();
        crate::validation_artifacts::write(
            "val-tensor-shape-layout-transform-01.json",
            "VAL-TENSOR-001",
            "Task 86 canonical shape/dtype views, split, broadcast, einops, and scalar/slice/tensor-count repeated-axis adapters",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        crate::validation_artifacts::write(
            "val-autograd-shape-layout-transform-01.json",
            "VAL-AUTOGRAD-001",
            "Task 86 analytical shape-transform VJP and JVP contracts",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        Ok(())
    }
}
