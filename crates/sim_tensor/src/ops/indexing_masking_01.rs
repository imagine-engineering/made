use crate::{
    CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId, ExecutionContext, Layout,
    StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError, ViewAccess,
    cpu_backend::{binary_broadcast_shape, broadcast_indices},
    generated_elementwise_or_runtime_operation_07::{
        ElementwiseRuntimePartSevenError, argwhere_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_20::{
        ElementwiseRuntimePartTwentyError, broadcast_tensor_vjp_with_context_exact_native,
    },
    promote_types,
};
use thiserror::Error;

pub const GATHER_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-650A7E36398C";
pub const INDEX_ADD_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-787E82C83CB5";
pub const MASKED_FILL_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-923E7CBA8F2A";
pub const NARROW_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-006E05C5DAAF";
pub const SCATTER_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-3710A378E57B";
pub const SCATTER_IN_PLACE_OPERATION_ID: &str = "COMFY-TENSOR-OP-6CEB132BD4F8";
pub const GATHER_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-301932E71E58";
pub const NARROW_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-A29830647789";
pub const NONZERO_OPERATION_ID: &str = "COMFY-TENSOR-OP-3885D52BE05C";
pub const SCATTER_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-2CC6738611B8";
pub const WHERE_OPERATION_ID: &str = "COMFY-TENSOR-OP-40CEC38A1D1F";

#[derive(Debug, Error)]
pub enum IndexingMaskingPartOneError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Argwhere(#[from] ElementwiseRuntimePartSevenError),
    #[error(transparent)]
    BroadcastVjp(#[from] ElementwiseRuntimePartTwentyError),
    #[error("indexing/masking part-one operation was cancelled")]
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
    #[error("indexing/masking part-one input is invalid: {0}")]
    Invalid(String),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for IndexingMaskingPartOneError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
struct GatherScatterPlan {
    axis: usize,
    traversal_shape: Vec<u64>,
}

#[derive(Debug)]
pub struct IndexAddGradients {
    pub input: Tensor,
    pub source: Tensor,
}

#[derive(Debug)]
pub struct ScatterGradients {
    pub input: Tensor,
    pub source: Tensor,
}

#[derive(Debug)]
pub struct WhereGradients {
    pub input: Tensor,
    pub other: Tensor,
}

#[derive(Debug)]
pub enum NonzeroOutput {
    Matrix(Tensor),
    Tuple(Vec<Tensor>),
}

pub fn gather_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    gather_with_context_exact_native(backend, input, dimension, index, context)
}

pub fn gather_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    gather_with_context_exact_native(backend, input, dimension, index, context)
}

fn gather_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    let plan = GatherScatterPlan::for_gather(input, dimension, index)?;
    let mut output = allocate_with_context(
        backend,
        &plan.traversal_shape,
        input.descriptor().dtype(),
        input.descriptor().stream(),
        context,
    )?;
    let mut write = output.write()?;
    for linear in 0..element_count(&plan.traversal_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &plan.traversal_shape)?;
        let mut input_indices = output_indices.clone();
        input_indices[plan.axis] = decode_index(
            index,
            &output_indices,
            input.descriptor().shape()[plan.axis],
        )?;
        write
            .element_bytes_mut(&output_indices)?
            .copy_from_slice(input.element_bytes(&input_indices)?);
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

pub fn gather_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_f32(input, GATHER_FUNCTION_OPERATION_ID)?;
    let plan = GatherScatterPlan::for_gather(input, dimension, index)?;
    require_f32_shape_stream(
        output_gradient,
        &plan.traversal_shape,
        input.descriptor().stream(),
        GATHER_FUNCTION_OPERATION_ID,
    )?;
    let mut values = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    for linear in 0..element_count(&plan.traversal_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &plan.traversal_shape)?;
        let mut input_indices = output_indices.clone();
        input_indices[plan.axis] = decode_index(
            index,
            &output_indices,
            input.descriptor().shape()[plan.axis],
        )?;
        let destination = ravel_index(&input_indices, input.descriptor().shape())?;
        values[destination] += read_f32(output_gradient, &output_indices)?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn gather_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    index: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_same_shape_dtype_stream(input, input_tangent, GATHER_FUNCTION_OPERATION_ID)?;
    gather_with_context_exact_native(backend, input_tangent, dimension, index, context)
}

pub fn index_add_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    dimension: i64,
    index: &Tensor,
    source: &Tensor,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<(), IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_f32(input, INDEX_ADD_IN_PLACE_OPERATION_ID)?;
    require_f32(source, INDEX_ADD_IN_PLACE_OPERATION_ID)?;
    require_index(
        index,
        input.descriptor().stream(),
        INDEX_ADD_IN_PLACE_OPERATION_ID,
    )?;
    if index.descriptor().rank() != 1 || source.descriptor().rank() != input.descriptor().rank() {
        return invalid("index_add_ requires a rank-one index and equal input/source ranks");
    }
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    for current in 0..input.descriptor().rank() {
        let expected = if current == axis {
            index.descriptor().shape()[0]
        } else {
            input.descriptor().shape()[current]
        };
        if source.descriptor().shape()[current] != expected {
            return invalid("index_add_ source shape does not match the indexed input shape");
        }
    }
    require_same_stream(input, source)?;
    let selected = decode_index_vector_with_context(
        backend,
        index,
        input.descriptor().shape()[axis],
        context,
    )?;
    let mut values = tensor_f32_with_context(backend, input, context)?;
    for linear in 0..element_count(source.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let source_indices = unravel_index(linear, source.descriptor().shape())?;
        let position = usize::try_from(source_indices[axis])
            .map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("index_add position"))?;
        let mut destination_indices = source_indices.clone();
        destination_indices[axis] = *selected.get(position).ok_or_else(|| {
            IndexingMaskingPartOneError::Invalid("index_add_ index position is missing".to_owned())
        })?;
        let destination = ravel_index(&destination_indices, input.descriptor().shape())?;
        values[destination] += alpha * read_f32(source, &source_indices)?;
    }
    let staged = upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )?;
    context.cancellation.check()?;
    input.commit_in_place(staged)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn index_add_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    source: &Tensor,
    alpha: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<IndexAddGradients, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_f32_shape_stream(
        output_gradient,
        input.descriptor().shape(),
        input.descriptor().stream(),
        INDEX_ADD_IN_PLACE_OPERATION_ID,
    )?;
    let mut input_probe = input.clone();
    index_add_in_place_with_context_exact_native(
        backend,
        &mut input_probe,
        dimension,
        index,
        source,
        alpha,
        context,
    )?;
    let source_gradient =
        gather_by_vector_with_context(backend, output_gradient, dimension, index, alpha, context)?;
    Ok(IndexAddGradients {
        input: copy_tensor_with_context(backend, output_gradient, context)?,
        source: source_gradient,
    })
}

pub fn index_add_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    source_tangent: &Tensor,
    dimension: i64,
    index: &Tensor,
    alpha: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    let mut output = copy_tensor_with_context(backend, input_tangent, context)?;
    index_add_in_place_with_context_exact_native(
        backend,
        &mut output,
        dimension,
        index,
        source_tangent,
        alpha,
        context,
    )?;
    Ok(output)
}

pub fn masked_fill_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    mask: &Tensor,
    value: crate::Scalar,
    context: &ExecutionContext<'_>,
) -> Result<(), IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_cpu(input, MASKED_FILL_IN_PLACE_OPERATION_ID)?;
    require_boolean_mask(
        mask,
        input.descriptor().stream(),
        MASKED_FILL_IN_PLACE_OPERATION_ID,
    )?;
    let shape = binary_broadcast_shape(input.descriptor().shape(), mask.descriptor().shape())?;
    if shape != input.descriptor().shape() {
        return invalid("masked_fill_ mask must broadcast to the receiver shape");
    }
    let encoded = input.descriptor().dtype().encode_scalar(
        value,
        MASKED_FILL_IN_PLACE_OPERATION_ID,
        input.descriptor().device(),
    )?;
    let mut staged = copy_tensor_with_context(backend, input, context)?;
    {
        let mut write = staged.write()?;
        for linear in 0..element_count(input.descriptor().shape())? {
            check_periodically(linear, context.cancellation)?;
            let indices = unravel_index(linear, input.descriptor().shape())?;
            let mask_indices = broadcast_indices(&indices, mask.descriptor().shape())?;
            if mask_value(mask, &mask_indices)? {
                write.element_bytes_mut(&indices)?.copy_from_slice(&encoded);
            }
        }
    }
    context.cancellation.check()?;
    input.commit_in_place(staged)?;
    Ok(())
}

pub fn masked_fill_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mask: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_f32(input, MASKED_FILL_IN_PLACE_OPERATION_ID)?;
    require_f32_shape_stream(
        output_gradient,
        input.descriptor().shape(),
        input.descriptor().stream(),
        MASKED_FILL_IN_PLACE_OPERATION_ID,
    )?;
    select_gradient_with_context(backend, input, mask, output_gradient, false, context)
}

pub fn masked_fill_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mask: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_f32(input, MASKED_FILL_IN_PLACE_OPERATION_ID)?;
    require_same_shape_dtype_stream(input, input_tangent, MASKED_FILL_IN_PLACE_OPERATION_ID)?;
    select_gradient_with_context(backend, input, mask, input_tangent, false, context)
}

pub fn narrow_method_exact_native(
    input: &Tensor,
    dimension: i64,
    start: i64,
    length: u64,
    cancellation: &CancellationToken,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    narrow_exact_native(input, dimension, start, length, cancellation)
}

pub fn narrow_function_exact_native(
    input: &Tensor,
    dimension: i64,
    start: i64,
    length: u64,
    cancellation: &CancellationToken,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    narrow_exact_native(input, dimension, start, length, cancellation)
}

fn narrow_exact_native(
    input: &Tensor,
    dimension: i64,
    start: i64,
    length: u64,
    cancellation: &CancellationToken,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    cancellation.check()?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let output = input.narrow_read_only(axis, start, length)?;
    cancellation.check()?;
    Ok(output)
}

pub fn narrow_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    start: i64,
    length: u64,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_f32(input, NARROW_FUNCTION_OPERATION_ID)?;
    let view = narrow_exact_native(input, dimension, start, length, context.cancellation)?;
    require_f32_shape_stream(
        output_gradient,
        view.descriptor().shape(),
        input.descriptor().stream(),
        NARROW_FUNCTION_OPERATION_ID,
    )?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let normalized_start = if start < 0 {
        i128::from(input.descriptor().shape()[axis]) + i128::from(start)
    } else {
        i128::from(start)
    };
    let normalized_start = u64::try_from(normalized_start)
        .map_err(|_| IndexingMaskingPartOneError::Invalid("invalid narrow start".to_owned()))?;
    let mut values = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    for linear in 0..element_count(output_gradient.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let gradient_indices = unravel_index(linear, output_gradient.descriptor().shape())?;
        let mut input_indices = gradient_indices.clone();
        input_indices[axis] = input_indices[axis].checked_add(normalized_start).ok_or(
            IndexingMaskingPartOneError::ShapeOverflow("narrow VJP index"),
        )?;
        values[ravel_index(&input_indices, input.descriptor().shape())?] =
            read_f32(output_gradient, &gradient_indices)?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn narrow_jvp_exact_native(
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    start: i64,
    length: u64,
    cancellation: &CancellationToken,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    cancellation.check()?;
    require_same_shape_dtype_stream(input, input_tangent, NARROW_FUNCTION_OPERATION_ID)?;
    narrow_exact_native(input_tangent, dimension, start, length, cancellation)
}

pub fn scatter_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    source: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    scatter_with_context_exact_native(backend, input, dimension, index, source, context)
}

pub fn scatter_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    source: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    scatter_with_context_exact_native(backend, input, dimension, index, source, context)
}

fn scatter_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    source: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    let plan = GatherScatterPlan::for_scatter(input, dimension, index, source)?;
    let mut output = copy_tensor_with_context(backend, input, context)?;
    {
        let mut write = output.write()?;
        for linear in 0..element_count(&plan.traversal_shape)? {
            check_periodically(linear, context.cancellation)?;
            let source_indices = unravel_index(linear, &plan.traversal_shape)?;
            let mut destination_indices = source_indices.clone();
            destination_indices[plan.axis] = decode_index(
                index,
                &source_indices,
                input.descriptor().shape()[plan.axis],
            )?;
            write
                .element_bytes_mut(&destination_indices)?
                .copy_from_slice(source.element_bytes(&source_indices)?);
        }
    }
    context.cancellation.check()?;
    Ok(output)
}

pub fn scatter_in_place_with_context_exact_native(
    backend: &CpuBackend,
    input: &mut Tensor,
    dimension: i64,
    index: &Tensor,
    source: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(), IndexingMaskingPartOneError> {
    let staged =
        scatter_with_context_exact_native(backend, input, dimension, index, source, context)?;
    context.cancellation.check()?;
    input.commit_in_place(staged)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn scatter_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    source: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<ScatterGradients, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_f32(input, SCATTER_FUNCTION_OPERATION_ID)?;
    require_f32(source, SCATTER_FUNCTION_OPERATION_ID)?;
    require_f32_shape_stream(
        output_gradient,
        input.descriptor().shape(),
        input.descriptor().stream(),
        SCATTER_FUNCTION_OPERATION_ID,
    )?;
    let plan = GatherScatterPlan::for_scatter(input, dimension, index, source)?;
    let mut input_values = tensor_f32_with_context(backend, output_gradient, context)?;
    let mut source_values = workspace_filled(
        backend,
        context,
        element_count(source.descriptor().shape())?,
        0.0_f32,
    )?;
    for linear in 0..element_count(&plan.traversal_shape)? {
        check_periodically(linear, context.cancellation)?;
        let source_indices = unravel_index(linear, &plan.traversal_shape)?;
        let mut destination_indices = source_indices.clone();
        destination_indices[plan.axis] = decode_index(
            index,
            &source_indices,
            input.descriptor().shape()[plan.axis],
        )?;
        let destination = ravel_index(&destination_indices, input.descriptor().shape())?;
        input_values[destination] = 0.0;
        source_values[ravel_index(&source_indices, source.descriptor().shape())?] =
            read_f32(output_gradient, &destination_indices)?;
    }
    let input_gradient = upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &input_values,
        context,
    )?;
    drop(input_values);
    let source_gradient = upload_f32_with_context(
        backend,
        source.descriptor().shape(),
        source.descriptor().stream(),
        &source_values,
        context,
    )?;
    Ok(ScatterGradients {
        input: input_gradient,
        source: source_gradient,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn scatter_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    index: &Tensor,
    source: &Tensor,
    source_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_same_shape_dtype_stream(input, input_tangent, SCATTER_FUNCTION_OPERATION_ID)?;
    require_same_shape_dtype_stream(source, source_tangent, SCATTER_FUNCTION_OPERATION_ID)?;
    scatter_with_context_exact_native(
        backend,
        input_tangent,
        dimension,
        index,
        source_tangent,
        context,
    )
}

pub fn nonzero_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    as_tuple: bool,
    context: &ExecutionContext<'_>,
) -> Result<NonzeroOutput, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    let matrix = argwhere_with_context_exact_native(backend, input, context)
        .map_err(map_argwhere_error)?;
    if !as_tuple {
        return Ok(NonzeroOutput::Matrix(matrix));
    }
    let rank = input.descriptor().rank();
    let rows = matrix.descriptor().shape()[0];
    let stride = i64::try_from(rank)
        .map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("nonzero tuple stride"))?;
    let mut columns = Vec::new();
    columns
        .try_reserve_exact(rank)
        .map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("nonzero tuple"))?;
    for column in 0..rank {
        check_periodically(column, context.cancellation)?;
        let descriptor = TensorDescriptor::new_strided(
            vec![rows],
            vec![stride],
            u64::try_from(column)
                .map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("nonzero column"))?,
            DType::I64,
            Layout::Strided,
            matrix.descriptor().device(),
            matrix.descriptor().stream(),
        )?;
        columns.push(matrix.view(descriptor, ViewAccess::ReadOnly)?);
    }
    context.cancellation.check()?;
    Ok(NonzeroOutput::Tuple(columns))
}

pub fn where_with_context_exact_native(
    backend: &CpuBackend,
    condition: &Tensor,
    input: &Tensor,
    other: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_boolean_mask(condition, input.descriptor().stream(), WHERE_OPERATION_ID)?;
    require_cpu(input, WHERE_OPERATION_ID)?;
    require_cpu(other, WHERE_OPERATION_ID)?;
    require_same_stream(input, other)?;
    let branch_shape =
        binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?;
    let output_shape = binary_broadcast_shape(&branch_shape, condition.descriptor().shape())?;
    let output_dtype = promote_types(input.descriptor().dtype(), other.descriptor().dtype())?;
    let mut output = allocate_with_context(
        backend,
        &output_shape,
        output_dtype,
        input.descriptor().stream(),
        context,
    )?;
    let mut write = output.write()?;
    for linear in 0..element_count(&output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let condition_indices = broadcast_indices(&output_indices, condition.descriptor().shape())?;
        let branch = if mask_value(condition, &condition_indices)? {
            input
        } else {
            other
        };
        let branch_indices = broadcast_indices(&output_indices, branch.descriptor().shape())?;
        let decoded = branch
            .descriptor()
            .dtype()
            .decode_scalar(branch.element_bytes(&branch_indices)?)?;
        let encoded =
            output_dtype.encode_decoded_scalar(decoded, WHERE_OPERATION_ID, DeviceId::CPU)?;
        write
            .element_bytes_mut(&output_indices)?
            .copy_from_slice(&encoded);
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

pub fn where_nonzero_with_context_exact_native(
    backend: &CpuBackend,
    condition: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<NonzeroOutput, IndexingMaskingPartOneError> {
    nonzero_with_context_exact_native(backend, condition, true, context)
}

pub fn where_vjp_with_context_exact_native(
    backend: &CpuBackend,
    condition: &Tensor,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<WhereGradients, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_f32(input, WHERE_OPERATION_ID)?;
    require_f32(other, WHERE_OPERATION_ID)?;
    let output_shape = binary_broadcast_shape(
        &binary_broadcast_shape(input.descriptor().shape(), other.descriptor().shape())?,
        condition.descriptor().shape(),
    )?;
    require_f32_shape_stream(
        output_gradient,
        &output_shape,
        input.descriptor().stream(),
        WHERE_OPERATION_ID,
    )?;
    let input_masked =
        masked_output_gradient_with_context(backend, condition, output_gradient, true, context)?;
    let other_masked =
        masked_output_gradient_with_context(backend, condition, output_gradient, false, context)?;
    Ok(WhereGradients {
        input: broadcast_tensor_vjp_with_context_exact_native(
            backend,
            input,
            &input_masked,
            context,
        )
        .map_err(map_broadcast_vjp_error)?,
        other: broadcast_tensor_vjp_with_context_exact_native(
            backend,
            other,
            &other_masked,
            context,
        )
        .map_err(map_broadcast_vjp_error)?,
    })
}

pub fn where_jvp_with_context_exact_native(
    backend: &CpuBackend,
    condition: &Tensor,
    input: &Tensor,
    input_tangent: &Tensor,
    other: &Tensor,
    other_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    context.cancellation.check()?;
    require_same_shape_dtype_stream(input, input_tangent, WHERE_OPERATION_ID)?;
    require_same_shape_dtype_stream(other, other_tangent, WHERE_OPERATION_ID)?;
    where_with_context_exact_native(backend, condition, input_tangent, other_tangent, context)
}

impl GatherScatterPlan {
    fn for_gather(
        input: &Tensor,
        dimension: i64,
        index: &Tensor,
    ) -> Result<Self, IndexingMaskingPartOneError> {
        require_cpu(input, GATHER_FUNCTION_OPERATION_ID)?;
        require_index_i64(
            index,
            input.descriptor().stream(),
            GATHER_FUNCTION_OPERATION_ID,
        )?;
        if input.descriptor().rank() != index.descriptor().rank() {
            return invalid("gather input and index ranks must match");
        }
        let axis = normalize_axis(dimension, input.descriptor().rank())?;
        for current in 0..input.descriptor().rank() {
            if current != axis
                && index.descriptor().shape()[current] > input.descriptor().shape()[current]
            {
                return invalid("gather index exceeds a non-gather input dimension");
            }
        }
        Ok(Self {
            axis,
            traversal_shape: index.descriptor().shape().to_vec(),
        })
    }

    fn for_scatter(
        input: &Tensor,
        dimension: i64,
        index: &Tensor,
        source: &Tensor,
    ) -> Result<Self, IndexingMaskingPartOneError> {
        require_cpu(input, SCATTER_FUNCTION_OPERATION_ID)?;
        require_cpu(source, SCATTER_FUNCTION_OPERATION_ID)?;
        require_index_i64(
            index,
            input.descriptor().stream(),
            SCATTER_FUNCTION_OPERATION_ID,
        )?;
        require_same_stream(input, source)?;
        if input.descriptor().dtype() != source.descriptor().dtype() {
            return Err(TensorError::DTypeMismatch {
                expected: input.descriptor().dtype(),
                actual: source.descriptor().dtype(),
            }
            .into());
        }
        if input.descriptor().rank() != index.descriptor().rank()
            || input.descriptor().rank() != source.descriptor().rank()
        {
            return invalid("scatter input, index, and source ranks must match");
        }
        let axis = normalize_axis(dimension, input.descriptor().rank())?;
        for current in 0..input.descriptor().rank() {
            if index.descriptor().shape()[current] > source.descriptor().shape()[current]
                || (current != axis
                    && index.descriptor().shape()[current] > input.descriptor().shape()[current])
            {
                return invalid("scatter index shape exceeds source or receiver bounds");
            }
        }
        Ok(Self {
            axis,
            traversal_shape: index.descriptor().shape().to_vec(),
        })
    }
}

fn gather_by_vector_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    index: &Tensor,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let selected = decode_index_vector_with_context(
        backend,
        index,
        input.descriptor().shape()[axis],
        context,
    )?;
    let mut shape = input.descriptor().shape().to_vec();
    shape[axis] = index.descriptor().shape()[0];
    let mut values = backend.workspace_vec(context, element_count(&shape)?)?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let mut indices = unravel_index(linear, &shape)?;
        let position = usize::try_from(indices[axis])
            .map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("index_add VJP"))?;
        indices[axis] = selected[position];
        values.try_push(scale * read_f32(input, &indices)?)?;
    }
    upload_f32_with_context(
        backend,
        &shape,
        input.descriptor().stream(),
        &values,
        context,
    )
}

fn select_gradient_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    mask: &Tensor,
    gradient: &Tensor,
    keep_true: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    require_boolean_mask(
        mask,
        input.descriptor().stream(),
        MASKED_FILL_IN_PLACE_OPERATION_ID,
    )?;
    let shape = binary_broadcast_shape(input.descriptor().shape(), mask.descriptor().shape())?;
    if shape != input.descriptor().shape() {
        return invalid("mask does not broadcast exactly to the receiver shape");
    }
    let mut values = backend.workspace_vec(context, element_count(input.descriptor().shape())?)?;
    for linear in 0..element_count(input.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        let mask_indices = broadcast_indices(&indices, mask.descriptor().shape())?;
        let selected = mask_value(mask, &mask_indices)? == keep_true;
        values.try_push(if selected {
            read_f32(gradient, &indices)?
        } else {
            0.0
        })?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

fn masked_output_gradient_with_context(
    backend: &CpuBackend,
    condition: &Tensor,
    gradient: &Tensor,
    keep_true: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    require_boolean_mask(
        condition,
        gradient.descriptor().stream(),
        WHERE_OPERATION_ID,
    )?;
    let mut values =
        backend.workspace_vec(context, element_count(gradient.descriptor().shape())?)?;
    for linear in 0..element_count(gradient.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, gradient.descriptor().shape())?;
        let condition_indices = broadcast_indices(&indices, condition.descriptor().shape())?;
        values.try_push(if mask_value(condition, &condition_indices)? == keep_true {
            read_f32(gradient, &indices)?
        } else {
            0.0
        })?;
    }
    upload_f32_with_context(
        backend,
        gradient.descriptor().shape(),
        gradient.descriptor().stream(),
        &values,
        context,
    )
}

fn copy_tensor_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    let mut output = allocate_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().dtype(),
        input.descriptor().stream(),
        context,
    )?;
    let mut write = output.write()?;
    for linear in 0..element_count(input.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        write
            .element_bytes_mut(&indices)?
            .copy_from_slice(input.element_bytes(&indices)?);
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

fn allocate_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    Ok(backend.allocate(descriptor, context)?.0)
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartOneError> {
    if values.len() != element_count(shape)? {
        return invalid("F32 upload length does not match its shape");
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn tensor_f32_with_context(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<crate::CpuWorkspaceVec<f32>, IndexingMaskingPartOneError> {
    require_f32(tensor, INDEX_ADD_IN_PLACE_OPERATION_ID)?;
    let mut values = backend.workspace_vec(context, element_count(tensor.descriptor().shape())?)?;
    for linear in 0..element_count(tensor.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        values.try_push(read_f32(
            tensor,
            &unravel_index(linear, tensor.descriptor().shape())?,
        )?)?;
    }
    Ok(values)
}

fn decode_index_vector_with_context(
    backend: &CpuBackend,
    index: &Tensor,
    bound: u64,
    context: &ExecutionContext<'_>,
) -> Result<crate::CpuWorkspaceVec<u64>, IndexingMaskingPartOneError> {
    let mut values = backend.workspace_vec(context, element_count(index.descriptor().shape())?)?;
    for linear in 0..element_count(index.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        values.try_push(decode_index(
            index,
            &[u64::try_from(linear)
                .map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("index vector"))?],
            bound,
        )?)?;
    }
    Ok(values)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    value: T,
) -> Result<crate::CpuWorkspaceVec<T>, IndexingMaskingPartOneError> {
    let mut values = backend.workspace_vec(context, capacity)?;
    for _ in 0..capacity {
        values.try_push(value)?;
    }
    Ok(values)
}

fn decode_index(
    index: &Tensor,
    indices: &[u64],
    bound: u64,
) -> Result<u64, IndexingMaskingPartOneError> {
    match index
        .descriptor()
        .dtype()
        .decode_scalar(index.element_bytes(indices)?)?
    {
        DecodedScalar::Signed(value) if value >= 0 => {
            let value = u64::try_from(value)
                .map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("index value"))?;
            if value < bound {
                Ok(value)
            } else {
                invalid("index value is outside the selected dimension")
            }
        }
        _ => invalid("indices must contain nonnegative in-range integers"),
    }
}

fn mask_value(mask: &Tensor, indices: &[u64]) -> Result<bool, IndexingMaskingPartOneError> {
    match mask
        .descriptor()
        .dtype()
        .decode_scalar(mask.element_bytes(indices)?)?
    {
        DecodedScalar::Boolean(value) => Ok(value),
        _ => Err(IndexingMaskingPartOneError::UnsupportedDType {
            operation: WHERE_OPERATION_ID,
            dtype: mask.descriptor().dtype(),
        }),
    }
}

fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, IndexingMaskingPartOneError> {
    match tensor
        .descriptor()
        .dtype()
        .decode_scalar(tensor.element_bytes(indices)?)?
    {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(IndexingMaskingPartOneError::UnsupportedDType {
            operation: WHERE_OPERATION_ID,
            dtype: tensor.descriptor().dtype(),
        }),
    }
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), IndexingMaskingPartOneError> {
    if tensor.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(IndexingMaskingPartOneError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        })
    }
}

fn require_f32(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), IndexingMaskingPartOneError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(IndexingMaskingPartOneError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        })
    }
}

fn require_index(
    index: &Tensor,
    stream: StreamId,
    operation: &'static str,
) -> Result<(), IndexingMaskingPartOneError> {
    require_cpu(index, operation)?;
    if !matches!(index.descriptor().dtype(), DType::I32 | DType::I64) {
        return Err(IndexingMaskingPartOneError::UnsupportedDType {
            operation,
            dtype: index.descriptor().dtype(),
        });
    }
    if index.descriptor().stream() != stream {
        return Err(TensorError::StreamMismatch {
            expected: stream,
            actual: index.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn require_index_i64(
    index: &Tensor,
    stream: StreamId,
    operation: &'static str,
) -> Result<(), IndexingMaskingPartOneError> {
    require_cpu(index, operation)?;
    if index.descriptor().dtype() != DType::I64 {
        return Err(IndexingMaskingPartOneError::UnsupportedDType {
            operation,
            dtype: index.descriptor().dtype(),
        });
    }
    if index.descriptor().stream() != stream {
        return Err(TensorError::StreamMismatch {
            expected: stream,
            actual: index.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn require_boolean_mask(
    mask: &Tensor,
    stream: StreamId,
    operation: &'static str,
) -> Result<(), IndexingMaskingPartOneError> {
    require_cpu(mask, operation)?;
    if mask.descriptor().dtype() != DType::Bool {
        return Err(IndexingMaskingPartOneError::UnsupportedDType {
            operation,
            dtype: mask.descriptor().dtype(),
        });
    }
    if mask.descriptor().stream() != stream {
        return Err(TensorError::StreamMismatch {
            expected: stream,
            actual: mask.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn require_same_stream(left: &Tensor, right: &Tensor) -> Result<(), IndexingMaskingPartOneError> {
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: left.descriptor().stream(),
            actual: right.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn require_same_shape_dtype_stream(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), IndexingMaskingPartOneError> {
    require_cpu(input, operation)?;
    require_cpu(other, operation)?;
    require_same_stream(input, other)?;
    if input.descriptor().shape() != other.descriptor().shape() {
        return invalid("tensor shapes must match exactly");
    }
    if input.descriptor().dtype() != other.descriptor().dtype() {
        return Err(TensorError::DTypeMismatch {
            expected: input.descriptor().dtype(),
            actual: other.descriptor().dtype(),
        }
        .into());
    }
    Ok(())
}

fn require_f32_shape_stream(
    tensor: &Tensor,
    shape: &[u64],
    stream: StreamId,
    operation: &'static str,
) -> Result<(), IndexingMaskingPartOneError> {
    require_f32(tensor, operation)?;
    if tensor.descriptor().shape() != shape {
        return invalid("gradient shape does not match the operation output");
    }
    if tensor.descriptor().stream() != stream {
        return Err(TensorError::StreamMismatch {
            expected: stream,
            actual: tensor.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, IndexingMaskingPartOneError> {
    let rank_i64 =
        i64::try_from(rank).map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("rank"))?;
    let normalized = if axis < 0 {
        rank_i64
            .checked_add(axis)
            .ok_or(IndexingMaskingPartOneError::ShapeOverflow("axis"))?
    } else {
        axis
    };
    if normalized < 0 || normalized >= rank_i64 {
        return invalid("dimension is outside the tensor rank");
    }
    usize::try_from(normalized).map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("axis"))
}

fn element_count(shape: &[u64]) -> Result<usize, IndexingMaskingPartOneError> {
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(IndexingMaskingPartOneError::ShapeOverflow("element count"))
    })?;
    usize::try_from(count).map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("element count"))
}

fn unravel_index(linear: usize, shape: &[u64]) -> Result<Vec<u64>, IndexingMaskingPartOneError> {
    let mut remainder = u64::try_from(linear)
        .map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("linear index"))?;
    let mut indices = vec![0; shape.len()];
    for (index, dimension) in indices.iter_mut().zip(shape).rev() {
        if *dimension == 0 {
            return invalid("cannot unravel an index through an empty dimension");
        }
        *index = remainder % dimension;
        remainder /= dimension;
    }
    Ok(indices)
}

fn ravel_index(indices: &[u64], shape: &[u64]) -> Result<usize, IndexingMaskingPartOneError> {
    if indices.len() != shape.len() {
        return invalid("index rank does not match tensor rank");
    }
    let linear = indices
        .iter()
        .zip(shape)
        .try_fold(0_u64, |linear, (index, dimension)| {
            if index >= dimension {
                return Err(IndexingMaskingPartOneError::Invalid(
                    "index is outside the tensor shape".to_owned(),
                ));
            }
            linear
                .checked_mul(*dimension)
                .and_then(|value| value.checked_add(*index))
                .ok_or(IndexingMaskingPartOneError::ShapeOverflow("linear index"))
        })?;
    usize::try_from(linear).map_err(|_| IndexingMaskingPartOneError::ShapeOverflow("linear index"))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), IndexingMaskingPartOneError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}

fn map_argwhere_error(error: ElementwiseRuntimePartSevenError) -> IndexingMaskingPartOneError {
    match error {
        ElementwiseRuntimePartSevenError::Cancelled => IndexingMaskingPartOneError::Cancelled,
        error => IndexingMaskingPartOneError::Argwhere(error),
    }
}

fn map_broadcast_vjp_error(
    error: ElementwiseRuntimePartTwentyError,
) -> IndexingMaskingPartOneError {
    match error {
        ElementwiseRuntimePartTwentyError::Cancelled => IndexingMaskingPartOneError::Cancelled,
        error => IndexingMaskingPartOneError::BroadcastVjp(error),
    }
}

fn invalid<T>(message: &str) -> Result<T, IndexingMaskingPartOneError> {
    Err(IndexingMaskingPartOneError::Invalid(message.to_owned()))
}
