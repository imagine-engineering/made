use crate::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, Tensor,
    TensorDescriptor, TensorError, ViewAccess,
    generated_sim_operator_indirection_01::OperatorIndirectionError,
    generated_elementwise_or_runtime_operation_08::{
        ElementwiseRuntimePartEightError, concatenate_jvp_with_context_exact_native,
        concatenate_vjp_with_context_exact_native, concatenate_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_16::{
        ElementwiseRuntimePartSixteenError, tile_jvp_with_context_exact_native,
        tile_vjp_with_context_exact_native, tile_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_17::{
        TensorSplitSpec, tensor_split_exact_native, tensor_split_jvp_exact_native,
        tensor_split_vjp_with_context_exact_native,
    },
    generated_shape_layout_transform_01::{
        flatten_with_context_exact_native_for_operation, permute_moved_dimensions,
        reshape_with_context_for_operation, unsqueeze_for_operation,
        view_with_shape_for_operation,
    },
};
use thiserror::Error;

pub const TENSOR_REPEAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-AED8D6D11C0F";
pub const TENSOR_RESHAPE_OPERATION_ID: &str = "COMFY-TENSOR-OP-CBDBBB4DF32F";
pub const TENSOR_UNBIND_OPERATION_ID: &str = "COMFY-TENSOR-OP-CA47A2C1F7CF";
pub const TENSOR_VIEW_AS_OPERATION_ID: &str = "COMFY-TENSOR-OP-868899CC3FD0";
pub const TORCH_CAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-A125CEF23550";
pub const TORCH_FLATTEN_OPERATION_ID: &str = "COMFY-TENSOR-OP-B4740B5E4B4D";
pub const TORCH_MOVEDIM_OPERATION_ID: &str = "COMFY-TENSOR-OP-A4C280408637";
pub const TORCH_PERMUTE_OPERATION_ID: &str = "COMFY-TENSOR-OP-890C284C875F";
pub const TORCH_RESHAPE_OPERATION_ID: &str = "COMFY-TENSOR-OP-B918E3498A97";
pub const TORCH_SPLIT_OPERATION_ID: &str = "COMFY-TENSOR-OP-D4BF78DE58FC";
pub const TORCH_STACK_OPERATION_ID: &str = "COMFY-TENSOR-OP-C99C862C91E1";
pub const TORCH_UNBIND_OPERATION_ID: &str = "COMFY-TENSOR-OP-75596B7E1112";

#[derive(Clone, Copy, Debug)]
pub enum TorchSplitSpec<'a> {
    Size(u64),
    Sizes(&'a [u64]),
}

#[derive(Debug, Error)]
pub enum ShapeLayoutTransformPartTwoError {
    #[error("{0}")]
    Tensor(#[from] #[source] TensorError),
    #[error("shape/layout-transform part-two execution was cancelled")]
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
    #[error("operation {operation} failed in its canonical owner: {reason}")]
    CanonicalOwner {
        operation: &'static str,
        reason: String,
    },
}

impl From<sim_types::CancellationError> for ShapeLayoutTransformPartTwoError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn tensor_repeat_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    repeat_for_operation(backend, input, repeats, TENSOR_REPEAT_OPERATION_ID, context)
}

pub fn repeat_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: &[i64],
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    validate_repeat(input, repeats, TENSOR_REPEAT_OPERATION_ID)?;
    require_f32_cpu(input, TENSOR_REPEAT_OPERATION_ID)?;
    require_f32_cpu(output_gradient, TENSOR_REPEAT_OPERATION_ID)?;
    tile_vjp_with_context_exact_native(backend, input, repeats, output_gradient, context)
        .map_err(|error| part_sixteen_error(TENSOR_REPEAT_OPERATION_ID, error))
}

pub fn repeat_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    repeats: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    validate_repeat(input, repeats, TENSOR_REPEAT_OPERATION_ID)?;
    tile_jvp_with_context_exact_native(backend, input, input_tangent, repeats, context)
        .map_err(|error| part_sixteen_error(TENSOR_REPEAT_OPERATION_ID, error))
}

pub fn tensor_reshape_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    shape: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    reshape_for_operation(backend, input, shape, TENSOR_RESHAPE_OPERATION_ID, context)
}

pub fn torch_reshape_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    shape: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    reshape_for_operation(backend, input, shape, TORCH_RESHAPE_OPERATION_ID, context)
}

pub fn reshape_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    let shape = shape_as_i64(input_shape, operation)?;
    reshape_for_operation(backend, output_gradient, &shape, operation, context)
}

pub fn reshape_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    shape: &[i64],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    reshape_for_operation(backend, input_tangent, shape, operation, context)
}

pub fn tensor_view_as_exact_native(
    input: &Tensor,
    other: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    require_same_stream(input, other, TENSOR_VIEW_AS_OPERATION_ID)?;
    let shape = shape_as_i64(other.descriptor().shape(), TENSOR_VIEW_AS_OPERATION_ID)?;
    view_with_shape_for_operation(input, &shape, TENSOR_VIEW_AS_OPERATION_ID, cancellation)
        .map_err(|error| canonical_error(TENSOR_VIEW_AS_OPERATION_ID, error))
}

pub fn view_as_vjp_exact_native(
    output_gradient: &Tensor,
    input_shape: &[u64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    let shape = shape_as_i64(input_shape, TENSOR_VIEW_AS_OPERATION_ID)?;
    view_with_shape_for_operation(
        output_gradient,
        &shape,
        TENSOR_VIEW_AS_OPERATION_ID,
        cancellation,
    )
    .map_err(|error| canonical_error(TENSOR_VIEW_AS_OPERATION_ID, error))
}

pub fn view_as_jvp_exact_native(
    input_tangent: &Tensor,
    other: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    tensor_view_as_exact_native(input_tangent, other, cancellation)
}

pub fn tensor_unbind_exact_native(
    input: &Tensor,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartTwoError> {
    unbind_for_operation(input, dimension, TENSOR_UNBIND_OPERATION_ID, cancellation)
}

pub fn torch_unbind_exact_native(
    input: &Tensor,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartTwoError> {
    unbind_for_operation(input, dimension, TORCH_UNBIND_OPERATION_ID, cancellation)
}

pub fn unbind_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradients: &[Tensor],
    dimension: i64,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    stack_for_operation(backend, output_gradients, dimension, operation, context)
}

pub fn unbind_jvp_exact_native(
    input_tangent: &Tensor,
    dimension: i64,
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartTwoError> {
    unbind_for_operation(input_tangent, dimension, operation, cancellation)
}

pub fn torch_cat_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    require_nonempty_compatible(inputs, TORCH_CAT_OPERATION_ID)?;
    concatenate_with_context_exact_native(backend, inputs, dimension, context)
        .map_err(|error| part_eight_error(TORCH_CAT_OPERATION_ID, error))
}

pub fn cat_vjp_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    dimension: i64,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    require_nonempty_compatible(inputs, TORCH_CAT_OPERATION_ID)?;
    concatenate_vjp_with_context_exact_native(
        backend,
        inputs,
        dimension,
        output_gradient,
        context,
    )
    .map_err(|error| part_eight_error(TORCH_CAT_OPERATION_ID, error))
}

pub fn cat_jvp_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    input_tangents: &[Tensor],
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    require_nonempty_compatible(inputs, TORCH_CAT_OPERATION_ID)?;
    concatenate_jvp_with_context_exact_native(
        backend,
        inputs,
        input_tangents,
        dimension,
        context,
    )
    .map_err(|error| part_eight_error(TORCH_CAT_OPERATION_ID, error))
}

pub fn torch_flatten_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    start_dimension: i64,
    end_dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    flatten_with_context_exact_native_for_operation(
        backend,
        input,
        start_dimension,
        end_dimension,
        TORCH_FLATTEN_OPERATION_ID,
        context,
    )
    .map_err(|error| canonical_error(TORCH_FLATTEN_OPERATION_ID, error))
}

pub fn flatten_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    input_shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    reshape_vjp_with_context_exact_native(
        backend,
        output_gradient,
        input_shape,
        TORCH_FLATTEN_OPERATION_ID,
        context,
    )
}

pub fn flatten_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    start_dimension: i64,
    end_dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    torch_flatten_with_context_exact_native(
        backend,
        input_tangent,
        start_dimension,
        end_dimension,
        context,
    )
}

pub fn torch_movedim_exact_native(
    input: &Tensor,
    source: &[i64],
    destination: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    permute_moved_dimensions(
        input,
        source,
        destination,
        TORCH_MOVEDIM_OPERATION_ID,
        cancellation,
    )
    .map_err(|error| canonical_error(TORCH_MOVEDIM_OPERATION_ID, error))
}

pub fn movedim_vjp_exact_native(
    output_gradient: &Tensor,
    source: &[i64],
    destination: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    permute_moved_dimensions(
        output_gradient,
        destination,
        source,
        TORCH_MOVEDIM_OPERATION_ID,
        cancellation,
    )
    .map_err(|error| canonical_error(TORCH_MOVEDIM_OPERATION_ID, error))
}

pub fn movedim_jvp_exact_native(
    input_tangent: &Tensor,
    source: &[i64],
    destination: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    torch_movedim_exact_native(input_tangent, source, destination, cancellation)
}

pub fn torch_permute_exact_native(
    input: &Tensor,
    dimensions: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    permute_for_operation(input, dimensions, TORCH_PERMUTE_OPERATION_ID, cancellation)
}

pub fn permute_vjp_exact_native(
    output_gradient: &Tensor,
    dimensions: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    permute_vjp_for_operation(
        output_gradient,
        dimensions,
        TORCH_PERMUTE_OPERATION_ID,
        cancellation,
    )
}

pub(crate) fn permute_vjp_for_operation(
    output_gradient: &Tensor,
    dimensions: &[i64],
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    let permutation = normalized_permutation(
        dimensions,
        output_gradient.descriptor().rank(),
        operation,
    )?;
    let mut inverse = vec![0_i64; permutation.len()];
    for (destination, source) in permutation.into_iter().enumerate() {
        *inverse
            .get_mut(source)
            .ok_or_else(|| overflow(operation, "inverse permutation"))? =
            i64::try_from(destination)
                .map_err(|_| overflow(operation, "inverse permutation"))?;
    }
    permute_for_operation(output_gradient, &inverse, operation, cancellation)
}

pub fn permute_jvp_exact_native(
    input_tangent: &Tensor,
    dimensions: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    torch_permute_exact_native(input_tangent, dimensions, cancellation)
}

pub fn torch_split_exact_native(
    input: &Tensor,
    specification: TorchSplitSpec<'_>,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    let specification = canonical_split_spec(input, specification, dimension)?;
    tensor_split_exact_native(input, &specification, dimension, cancellation)
        .map_err(|error| canonical_error(TORCH_SPLIT_OPERATION_ID, error))
}

pub fn split_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradients: &[Tensor],
    specification: TorchSplitSpec<'_>,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    let specification = canonical_split_spec(input, specification, dimension)?;
    tensor_split_vjp_with_context_exact_native(
        backend,
        input,
        output_gradients,
        &specification,
        dimension,
        context,
    )
    .map_err(|error| canonical_error(TORCH_SPLIT_OPERATION_ID, error))
}

pub fn split_jvp_exact_native(
    input_tangent: &Tensor,
    specification: TorchSplitSpec<'_>,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    let specification = canonical_split_spec(input_tangent, specification, dimension)?;
    tensor_split_jvp_exact_native(input_tangent, &specification, dimension, cancellation)
        .map_err(|error| canonical_error(TORCH_SPLIT_OPERATION_ID, error))
}

pub fn torch_stack_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    stack_for_operation(backend, inputs, dimension, TORCH_STACK_OPERATION_ID, context)
}

pub fn stack_vjp_exact_native(
    output_gradient: &Tensor,
    dimension: i64,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartTwoError> {
    unbind_for_operation(
        output_gradient,
        dimension,
        TORCH_STACK_OPERATION_ID,
        cancellation,
    )
}

pub fn stack_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangents: &[Tensor],
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    stack_for_operation(
        backend,
        input_tangents,
        dimension,
        TORCH_STACK_OPERATION_ID,
        context,
    )
}

fn repeat_for_operation(
    backend: &CpuBackend,
    input: &Tensor,
    repeats: &[i64],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    validate_repeat(input, repeats, operation)?;
    tile_with_context_exact_native(backend, input, repeats, context)
        .map_err(|error| part_sixteen_error(operation, error))
}

fn validate_repeat(
    input: &Tensor,
    repeats: &[i64],
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartTwoError> {
    require_cpu(input, operation)?;
    if repeats.len() < input.descriptor().rank() {
        return invalid(operation, "repeat dimensions cannot be shorter than the input rank");
    }
    if repeats.iter().any(|repeat| *repeat < 0) {
        return invalid(operation, "repeat counts must be non-negative");
    }
    Ok(())
}

fn reshape_for_operation(
    backend: &CpuBackend,
    input: &Tensor,
    shape: &[i64],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    reshape_with_context_for_operation(backend, input, shape, operation, context)
        .map_err(|error| canonical_error(operation, error))
}

pub(crate) fn permute_for_operation(
    input: &Tensor,
    dimensions: &[i64],
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    let permutation = normalized_permutation(dimensions, input.descriptor().rank(), operation)?;
    let descriptor = input.descriptor().permuted_view(&permutation)?;
    let output = input.view(descriptor, ViewAccess::ReadOnly)?;
    cancellation.check()?;
    Ok(output)
}

fn normalized_permutation(
    dimensions: &[i64],
    rank: usize,
    operation: &'static str,
) -> Result<Vec<usize>, ShapeLayoutTransformPartTwoError> {
    if dimensions.len() != rank {
        return invalid(operation, "permutation length must match the input rank");
    }
    let mut seen = vec![false; rank];
    let mut permutation = Vec::new();
    permutation
        .try_reserve_exact(rank)
        .map_err(|_| overflow(operation, "permutation"))?;
    for &dimension in dimensions {
        let axis = normalize_axis(dimension, rank, operation)?;
        let was_seen = seen
            .get_mut(axis)
            .ok_or_else(|| overflow(operation, "permutation axis"))?;
        if *was_seen {
            return invalid(operation, "permutation dimensions must be unique");
        }
        *was_seen = true;
        permutation.push(axis);
    }
    Ok(permutation)
}

fn unbind_for_operation(
    input: &Tensor,
    dimension: i64,
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ShapeLayoutTransformPartTwoError> {
    cancellation.check()?;
    require_cpu(input, operation)?;
    let axis = normalize_axis(dimension, input.descriptor().rank(), operation)?;
    let count = usize::try_from(input.descriptor().shape()[axis])
        .map_err(|_| overflow(operation, "unbind output count"))?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(count)
        .map_err(|_| overflow(operation, "unbind outputs"))?;
    for index in 0..count {
        if index.is_multiple_of(64) {
            cancellation.check()?;
        }
        let start = i64::try_from(index).map_err(|_| overflow(operation, "unbind index"))?;
        let narrowed = input.narrow_read_only(axis, start, 1)?;
        let mut shape = narrowed.descriptor().shape().to_vec();
        let mut strides = narrowed.descriptor().strides().to_vec();
        shape.remove(axis);
        strides.remove(axis);
        let descriptor = TensorDescriptor::new_strided(
            shape,
            strides,
            narrowed.descriptor().offset_elements(),
            narrowed.descriptor().dtype(),
            crate::Layout::Strided,
            narrowed.descriptor().device(),
            narrowed.descriptor().stream(),
        )?;
        outputs.push(narrowed.view(descriptor, ViewAccess::ReadOnly)?);
    }
    cancellation.check()?;
    Ok(outputs)
}

fn stack_for_operation(
    backend: &CpuBackend,
    inputs: &[Tensor],
    dimension: i64,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ShapeLayoutTransformPartTwoError> {
    context.cancellation.check()?;
    let first = inputs
        .first()
        .ok_or_else(|| invalid_error(operation, "stack input list is empty"))?;
    require_nonempty_compatible(inputs, operation)?;
    let axis = normalize_insertion_axis(dimension, first.descriptor().rank(), operation)?;
    let axis = i64::try_from(axis).map_err(|_| overflow(operation, "stack axis"))?;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(inputs.len())
        .map_err(|_| overflow(operation, "stack views"))?;
    for input in inputs {
        expanded.push(
            unsqueeze_for_operation(input, axis, operation, context.cancellation)
                .map_err(|error| canonical_error(operation, error))?,
        );
    }
    concatenate_with_context_exact_native(backend, &expanded, axis, context)
        .map_err(|error| part_eight_error(operation, error))
}

fn part_eight_error(
    operation: &'static str,
    error: ElementwiseRuntimePartEightError,
) -> ShapeLayoutTransformPartTwoError {
    match error {
        ElementwiseRuntimePartEightError::Cancelled => ShapeLayoutTransformPartTwoError::Cancelled,
        ElementwiseRuntimePartEightError::Tensor(error) => error.into(),
        error => canonical_error(operation, error),
    }
}

fn part_sixteen_error(
    operation: &'static str,
    error: ElementwiseRuntimePartSixteenError,
) -> ShapeLayoutTransformPartTwoError {
    match error {
        ElementwiseRuntimePartSixteenError::Cancelled => {
            ShapeLayoutTransformPartTwoError::Cancelled
        }
        ElementwiseRuntimePartSixteenError::Tensor(error) => error.into(),
        ElementwiseRuntimePartSixteenError::Cast(OperatorIndirectionError::Cancelled) => {
            ShapeLayoutTransformPartTwoError::Cancelled
        }
        ElementwiseRuntimePartSixteenError::Cast(OperatorIndirectionError::Tensor(error)) => {
            error.into()
        }
        ElementwiseRuntimePartSixteenError::PartEight(error) => {
            part_eight_error(operation, error)
        }
        error => canonical_error(operation, error),
    }
}

fn canonical_split_spec(
    input: &Tensor,
    specification: TorchSplitSpec<'_>,
    dimension: i64,
) -> Result<TensorSplitSpec, ShapeLayoutTransformPartTwoError> {
    require_cpu(input, TORCH_SPLIT_OPERATION_ID)?;
    let _axis = normalize_axis(
        dimension,
        input.descriptor().rank(),
        TORCH_SPLIT_OPERATION_ID,
    )?;
    match specification {
        TorchSplitSpec::Size(size) => {
            if size == 0 {
                return invalid(TORCH_SPLIT_OPERATION_ID, "split size must be nonzero");
            }
            Ok(TensorSplitSpec::Size(size))
        }
        TorchSplitSpec::Sizes(sizes) => {
            Ok(TensorSplitSpec::Sizes(sizes.to_vec()))
        }
    }
}

fn require_nonempty_compatible(
    inputs: &[Tensor],
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartTwoError> {
    let first = inputs
        .first()
        .ok_or_else(|| invalid_error(operation, "input tensor list is empty"))?;
    require_cpu(first, operation)?;
    for input in &inputs[1..] {
        require_cpu(input, operation)?;
        if input.descriptor().dtype() != first.descriptor().dtype() {
            return invalid(operation, "input tensors must have the same dtype");
        }
        require_same_stream(first, input, operation)?;
    }
    Ok(())
}

fn require_same_stream(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartTwoError> {
    if left.descriptor().stream() != right.descriptor().stream() {
        return invalid(operation, "input tensors must use the same stream");
    }
    Ok(())
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartTwoError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(ShapeLayoutTransformPartTwoError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        });
    }
    Ok(())
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ShapeLayoutTransformPartTwoError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(ShapeLayoutTransformPartTwoError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn normalize_axis(
    axis: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ShapeLayoutTransformPartTwoError> {
    if rank == 0 {
        return invalid(operation, "operation requires a tensor axis");
    }
    let rank_i64 = i64::try_from(rank).map_err(|_| overflow(operation, "rank"))?;
    let axis = if axis < 0 {
        rank_i64
            .checked_add(axis)
            .ok_or_else(|| overflow(operation, "axis"))?
    } else {
        axis
    };
    if axis < 0 || axis >= rank_i64 {
        return invalid(operation, "axis is outside the input rank");
    }
    usize::try_from(axis).map_err(|_| overflow(operation, "axis"))
}

fn normalize_insertion_axis(
    axis: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ShapeLayoutTransformPartTwoError> {
    let output_rank = rank
        .checked_add(1)
        .ok_or_else(|| overflow(operation, "output rank"))?;
    let output_rank_i64 =
        i64::try_from(output_rank).map_err(|_| overflow(operation, "output rank"))?;
    let axis = if axis < 0 {
        output_rank_i64
            .checked_add(axis)
            .ok_or_else(|| overflow(operation, "axis"))?
    } else {
        axis
    };
    if axis < 0 || axis >= output_rank_i64 {
        return invalid(operation, "axis is outside the output rank");
    }
    usize::try_from(axis).map_err(|_| overflow(operation, "axis"))
}

fn shape_as_i64(
    shape: &[u64],
    operation: &'static str,
) -> Result<Vec<i64>, ShapeLayoutTransformPartTwoError> {
    shape
        .iter()
        .map(|dimension| i64::try_from(*dimension).map_err(|_| overflow(operation, "shape")))
        .collect()
}

fn canonical_error(
    operation: &'static str,
    error: impl std::fmt::Display,
) -> ShapeLayoutTransformPartTwoError {
    ShapeLayoutTransformPartTwoError::CanonicalOwner {
        operation,
        reason: error.to_string(),
    }
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, ShapeLayoutTransformPartTwoError> {
    Err(invalid_error(operation, reason))
}

fn invalid_error(
    operation: &'static str,
    reason: impl Into<String>,
) -> ShapeLayoutTransformPartTwoError {
    ShapeLayoutTransformPartTwoError::Invalid {
        operation,
        reason: reason.into(),
    }
}

fn overflow(
    operation: &'static str,
    subject: &'static str,
) -> ShapeLayoutTransformPartTwoError {
    ShapeLayoutTransformPartTwoError::ShapeOverflow { operation, subject }
}

#[cfg(test)]
mod validation_tests {
    use super::{
        ShapeLayoutTransformPartTwoError, TORCH_CAT_OPERATION_ID, part_eight_error,
        part_sixteen_error,
    };
    use crate::{
        TensorError,
        generated_sim_operator_indirection_01::OperatorIndirectionError,
        generated_elementwise_or_runtime_operation_08::ElementwiseRuntimePartEightError,
        generated_elementwise_or_runtime_operation_16::ElementwiseRuntimePartSixteenError,
    };
    use std::collections::BTreeMap;

    #[test]
    fn canonical_owner_wrappers_preserve_typed_cancellation_and_resource_exhaustion() {
        assert!(matches!(
            part_eight_error(
                TORCH_CAT_OPERATION_ID,
                ElementwiseRuntimePartEightError::Cancelled,
            ),
            ShapeLayoutTransformPartTwoError::Cancelled
        ));
        assert!(matches!(
            part_eight_error(
                TORCH_CAT_OPERATION_ID,
                ElementwiseRuntimePartEightError::Tensor(
                    TensorError::WorkspaceAuthorizationExceeded {
                        requested: 64,
                        authorized: 32,
                        in_use: 0,
                    },
                ),
            ),
            ShapeLayoutTransformPartTwoError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded {
                    requested: 64,
                    authorized: 32,
                    in_use: 0,
                }
            )
        ));
        assert!(matches!(
            part_sixteen_error(
                TORCH_CAT_OPERATION_ID,
                ElementwiseRuntimePartSixteenError::Cast(OperatorIndirectionError::Cancelled),
            ),
            ShapeLayoutTransformPartTwoError::Cancelled
        ));
        assert!(matches!(
            part_sixteen_error(
                TORCH_CAT_OPERATION_ID,
                ElementwiseRuntimePartSixteenError::PartEight(
                    ElementwiseRuntimePartEightError::Cancelled,
                ),
            ),
            ShapeLayoutTransformPartTwoError::Cancelled
        ));
    }

    #[test]
    fn writes_task_validation_artifacts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture_digests = BTreeMap::from([
            ("COMFY-TENSOR-OP-75596B7E1112", "f86f3a262eff663771e0766fb11d5a02b480ad574b353ce3fcf32babb9f13edb"),
            ("COMFY-TENSOR-OP-868899CC3FD0", "031cba3eb98e8da4fd82176910c01e9bb5dcae794d59976f5763fbd1a1a2d941"),
            ("COMFY-TENSOR-OP-890C284C875F", "f16cd46a6665a6da29e16e2109db77f29cd0bb114123e25a0637ccecf22b8d8d"),
            ("COMFY-TENSOR-OP-A125CEF23550", "18c6a4677e3332fdc3509c30d4f3bde4164881568533eeeaba241f218af994db"),
            ("COMFY-TENSOR-OP-A4C280408637", "ac743fb42475867075042b54c9d248349aa07e3df2741401ec896d6d61698e11"),
            ("COMFY-TENSOR-OP-AED8D6D11C0F", "c2a33148205f49e4ad9608d7bd9d4ea34b7402c99d2d67fe520cb9eca1930852"),
            ("COMFY-TENSOR-OP-B4740B5E4B4D", "352292893c556b5df417967000327a9d452a4d48b97a2e66207dbffc64b780ef"),
            ("COMFY-TENSOR-OP-B918E3498A97", "209f82fd077985725368979cd5982241fd93141a79f185add0753923c7220124"),
            ("COMFY-TENSOR-OP-C99C862C91E1", "6810e1290933b571162d6b1dd5f9bb579cc3f69c13e8b30bf735b3c5f4cc822e"),
            ("COMFY-TENSOR-OP-CA47A2C1F7CF", "2ef1d16844a032acfba410e7520f578b70d697f1ff91e4c62dcf1a7b749b6c33"),
            ("COMFY-TENSOR-OP-CBDBBB4DF32F", "0e113c1361a9dfc2d32e39a718ce78757c8095288eadc2fd61c97c2be0b9d83c"),
            ("COMFY-TENSOR-OP-D4BF78DE58FC", "da2261d2889e735caac4d8d15479091c6d256f3710fb81541f54235562702a9a"),
        ]);
        let cases = fixture_digests
            .keys()
            .map(|operation| (*operation, true))
            .collect::<BTreeMap<_, _>>();
        crate::validation_artifacts::write(
            "val-tensor-shape-layout-transform-02.json",
            "VAL-TENSOR-001",
            "Task 87 exact shape facades over canonical descriptor, split, concatenate, tile, and Task 86 owners",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        crate::validation_artifacts::write(
            "val-autograd-shape-layout-transform-02.json",
            "VAL-AUTOGRAD-001",
            "Task 87 analytical composition VJP and JVP contracts",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        Ok(())
    }
}
