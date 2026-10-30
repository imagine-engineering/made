use crate::{
    BackendCapabilityMatrix, BinaryOperation, CancellationToken, CpuBackend, DType, DeviceId,
    ExecutionContext, Layout, NativeStream, NativeStreamRegistry, Tensor, TensorBackend,
    TensorDescriptor, TensorError, ViewAccess,
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_13::{
        ElementwiseRuntimePartThirteenError,
        softmax_function_jvp_with_context_exact_native as canonical_softmax_jvp_with_context,
        softmax_function_vjp_with_context_exact_native as canonical_softmax_vjp_with_context,
        softmax_function_with_context_exact_native as canonical_softmax_with_context,
    },
    generated_elementwise_or_runtime_operation_15::{
        ElementwiseRuntimePartFifteenError,
        flip_dimensions_with_context_exact_native as canonical_flip_with_context,
    },
    generated_elementwise_or_runtime_operation_19::{
        ElementwiseRuntimePartNineteenError, cuda_stream_exact_native as canonical_cuda_stream,
    },
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const ACTIVATION_1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-DDAAD49116D0";
pub const INT_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-D5D333C89A34";
pub const SOFTMAX_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-E430CCED2202";
pub const MEM_EFFICIENT_SDP_OPERATION_ID: &str = "COMFY-TENSOR-OP-D54F52B18FB1";
pub const BROADCAST_TENSORS_OPERATION_ID: &str = "COMFY-TENSOR-OP-E644686B4E0F";
pub const CROSS_OPERATION_ID: &str = "COMFY-TENSOR-OP-D6F57272FC58";
pub const CUDA_STREAM_OPERATION_ID: &str = "COMFY-TENSOR-OP-E237E236E06A";
pub const CUDA_SYNCHRONIZE_OPERATION_ID: &str = "COMFY-TENSOR-OP-D718FC279D3E";
pub const FLIP_OPERATION_ID: &str = "COMFY-TENSOR-OP-E07BBEBA226B";
pub const MODULE_INIT_OPERATION_ID: &str = "COMFY-TENSOR-OP-D54AF27B4D70";
pub const SWAPAXES_OPERATION_ID: &str = "COMFY-TENSOR-OP-E0C529F06769";
pub const DIRECTML_DEVICE_NAME_OPERATION_ID: &str = "COMFY-TENSOR-OP-E01C0CE81BB1";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartTwentyError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Cast(Box<OperatorIndirectionError>),
    #[error(transparent)]
    PartThirteen(Box<ElementwiseRuntimePartThirteenError>),
    #[error(transparent)]
    PartFifteen(Box<ElementwiseRuntimePartFifteenError>),
    #[error(transparent)]
    PartNineteen(Box<ElementwiseRuntimePartNineteenError>),
    #[error("elementwise/runtime part-twenty execution was cancelled")]
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
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<OperatorIndirectionError> for ElementwiseRuntimePartTwentyError {
    fn from(error: OperatorIndirectionError) -> Self {
        Self::Cast(Box::new(error))
    }
}

impl From<ElementwiseRuntimePartThirteenError> for ElementwiseRuntimePartTwentyError {
    fn from(error: ElementwiseRuntimePartThirteenError) -> Self {
        Self::PartThirteen(Box::new(error))
    }
}

impl From<ElementwiseRuntimePartFifteenError> for ElementwiseRuntimePartTwentyError {
    fn from(error: ElementwiseRuntimePartFifteenError) -> Self {
        Self::PartFifteen(Box::new(error))
    }
}

impl From<ElementwiseRuntimePartNineteenError> for ElementwiseRuntimePartTwentyError {
    fn from(error: ElementwiseRuntimePartNineteenError) -> Self {
        Self::PartNineteen(Box::new(error))
    }
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartTwentyError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn int_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    Ok(cast_to_with_context_exact_native(
        backend,
        input,
        DType::I32,
        input.descriptor().device(),
        false,
        false,
        context,
    )?)
}

pub fn softmax_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    Ok(canonical_softmax_with_context(
        backend, input, dimension, context,
    )?)
}

pub fn softmax_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    Ok(canonical_softmax_vjp_with_context(
        backend,
        input,
        output_gradient,
        dimension,
        context,
    )?)
}

pub fn softmax_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    Ok(canonical_softmax_jvp_with_context(
        backend,
        input,
        input_tangent,
        dimension,
        context,
    )?)
}

pub fn broadcast_tensors_exact_native(
    inputs: &[Tensor],
    cancellation: &CancellationToken,
) -> Result<Vec<Tensor>, ElementwiseRuntimePartTwentyError> {
    cancellation.check()?;
    if inputs.is_empty() {
        return invalid(
            BROADCAST_TENSORS_OPERATION_ID,
            "broadcast_tensors requires at least one tensor",
        );
    }
    let device = inputs[0].descriptor().device();
    let stream = inputs[0].descriptor().stream();
    let mut output_shape = Vec::new();
    for input in inputs {
        if input.descriptor().device() != device || input.descriptor().stream() != stream {
            return Err(ElementwiseRuntimePartTwentyError::UnsupportedDevice {
                operation: BROADCAST_TENSORS_OPERATION_ID,
                device: input.descriptor().device(),
            });
        }
        output_shape = crate::binary_broadcast_shape(&output_shape, input.descriptor().shape())?;
    }
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(inputs.len())
        .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("broadcast outputs"))?;
    for (index, input) in inputs.iter().enumerate() {
        check_periodically(index, cancellation)?;
        outputs.push(broadcast_view_to_shape(input, &output_shape)?);
    }
    cancellation.check()?;
    Ok(outputs)
}

pub fn broadcast_tensor_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    require_f32_cpu(input, BROADCAST_TENSORS_OPERATION_ID)?;
    require_f32_cpu(output_gradient, BROADCAST_TENSORS_OPERATION_ID)?;
    context.check()?;
    if input.descriptor().stream() != output_gradient.descriptor().stream() {
        return Err(ElementwiseRuntimePartTwentyError::UnsupportedDevice {
            operation: BROADCAST_TENSORS_OPERATION_ID,
            device: output_gradient.descriptor().device(),
        });
    }
    let output_shape = output_gradient.descriptor().shape();
    let expected_shape = crate::binary_broadcast_shape(input.descriptor().shape(), output_shape)?;
    if expected_shape != output_shape {
        return invalid(
            BROADCAST_TENSORS_OPERATION_ID,
            "output gradient shape must be the broadcast output shape",
        );
    }
    let mut values = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    let rank_offset = output_shape
        .len()
        .checked_sub(input.descriptor().rank())
        .ok_or(ElementwiseRuntimePartTwentyError::ShapeOverflow(
            "broadcast VJP rank",
        ))?;
    for linear in 0..element_count(output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, output_shape)?;
        let mut input_indices = Vec::new();
        input_indices
            .try_reserve_exact(input.descriptor().rank())
            .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("broadcast VJP index"))?;
        for input_axis in 0..input.descriptor().rank() {
            let output_axis = rank_offset + input_axis;
            input_indices.push(if input.descriptor().shape()[input_axis] == 1 {
                0
            } else {
                output_indices[output_axis]
            });
        }
        let destination = ravel_index(&input_indices, input.descriptor().shape())?;
        values[destination] += read_f32(
            output_gradient,
            &output_indices,
            BROADCAST_TENSORS_OPERATION_ID,
        )?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn broadcast_tensor_jvp_exact_native(
    input_tangent: &Tensor,
    output_shape: &[u64],
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    cancellation.check()?;
    let expected = crate::binary_broadcast_shape(input_tangent.descriptor().shape(), output_shape)?;
    if expected != output_shape {
        return invalid(
            BROADCAST_TENSORS_OPERATION_ID,
            "requested tangent shape is not the exact broadcast output shape",
        );
    }
    let output = broadcast_view_to_shape(input_tangent, output_shape)?;
    cancellation.check()?;
    Ok(output)
}

pub fn cross_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    dimension: Option<i64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    cross_impl_with_context(backend, input, other, dimension, context)
}

pub fn cross_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    dimension: Option<i64>,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Tensor), ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    require_matching_f32(input, output_gradient, CROSS_OPERATION_ID)?;
    let input_gradient =
        cross_impl_with_context(backend, other, output_gradient, dimension, context)?;
    let other_gradient =
        cross_impl_with_context(backend, output_gradient, input, dimension, context)?;
    Ok((input_gradient, other_gradient))
}

pub fn cross_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    dimension: Option<i64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    require_matching_f32(input, input_tangent, CROSS_OPERATION_ID)?;
    require_matching_f32(other, other_tangent, CROSS_OPERATION_ID)?;
    let left = cross_impl_with_context(backend, input_tangent, other, dimension, context)?;
    let right = cross_impl_with_context(backend, input, other_tangent, dimension, context)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .binary(BinaryOperation::Add, &left, &right, descriptor, context)?
        .0)
}

pub fn cuda_stream_exact_native(
    registry: &NativeStreamRegistry,
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    priority: i32,
    cancellation: &CancellationToken,
) -> Result<NativeStream, ElementwiseRuntimePartTwentyError> {
    cancellation.check()?;
    Ok(canonical_cuda_stream(
        registry,
        capabilities,
        device,
        priority,
        cancellation,
    )?)
}

pub fn cuda_synchronize_exact_native(
    backend: &dyn TensorBackend,
    capabilities: &BackendCapabilityMatrix,
    execution: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartTwentyError> {
    execution.cancellation.check()?;
    Ok(crate::synchronize_device_exact_native(
        backend,
        capabilities,
        &[DeviceKind::Cuda, DeviceKind::Rocm],
        CUDA_SYNCHRONIZE_OPERATION_ID,
        execution,
    )?)
}

pub fn flip_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    Ok(canonical_flip_with_context(
        backend,
        input,
        dimensions,
        FLIP_OPERATION_ID,
        context,
    )?)
}

pub fn flip_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    flip_with_context_exact_native(backend, output_gradient, dimensions, context)
}

pub fn flip_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.cancellation.check()?;
    flip_with_context_exact_native(backend, input_tangent, dimensions, context)
}

pub fn swapaxes_exact_native(
    input: &Tensor,
    axis_zero: i64,
    axis_one: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    cancellation.check()?;
    let first = normalize_axis(axis_zero, input.descriptor().rank(), SWAPAXES_OPERATION_ID)?;
    let second = normalize_axis(axis_one, input.descriptor().rank(), SWAPAXES_OPERATION_ID)?;
    let mut shape = input.descriptor().shape().to_vec();
    let mut strides = input.descriptor().strides().to_vec();
    shape.swap(first, second);
    strides.swap(first, second);
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

pub fn swapaxes_vjp_exact_native(
    output_gradient: &Tensor,
    axis_zero: i64,
    axis_one: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    swapaxes_exact_native(output_gradient, axis_zero, axis_one, cancellation)
}

pub fn swapaxes_jvp_exact_native(
    input_tangent: &Tensor,
    axis_zero: i64,
    axis_one: i64,
    cancellation: &CancellationToken,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    swapaxes_exact_native(input_tangent, axis_zero, axis_one, cancellation)
}

pub fn directml_device_name_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<String, ElementwiseRuntimePartTwentyError> {
    cancellation.check()?;
    Ok(crate::native_device_name_exact(
        capabilities,
        device,
        DeviceKind::DirectMl,
        DIRECTML_DEVICE_NAME_OPERATION_ID,
        cancellation,
    )?)
}

fn cross_impl_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    dimension: Option<i64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    require_matching_f32(input, other, CROSS_OPERATION_ID)?;
    let axis = match dimension {
        Some(dimension) => {
            normalize_axis(dimension, input.descriptor().rank(), CROSS_OPERATION_ID)?
        }
        None => input
            .descriptor()
            .shape()
            .iter()
            .position(|dimension| *dimension == 3)
            .ok_or_else(|| ElementwiseRuntimePartTwentyError::Invalid {
                operation: CROSS_OPERATION_ID,
                reason: "cross without dim requires a dimension of length three".to_owned(),
            })?,
    };
    if input.descriptor().shape()[axis] != 3 {
        return invalid(CROSS_OPERATION_ID, "cross dimension must have length three");
    }
    let shape = input.descriptor().shape();
    let mut values = backend.workspace_vec(context, element_count(shape)?)?;
    for linear in 0..element_count(shape)? {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, shape)?;
        let component = usize::try_from(indices[axis])
            .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("cross component"))?;
        let left_component = (component + 1) % 3;
        let right_component = (component + 2) % 3;
        let mut left_indices = indices.clone();
        let mut right_indices = indices;
        left_indices[axis] = u64::try_from(left_component)
            .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("cross index"))?;
        right_indices[axis] = u64::try_from(right_component)
            .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("cross index"))?;
        values.try_push(
            read_f32(input, &left_indices, CROSS_OPERATION_ID)?
                * read_f32(other, &right_indices, CROSS_OPERATION_ID)?
                - read_f32(input, &right_indices, CROSS_OPERATION_ID)?
                    * read_f32(other, &left_indices, CROSS_OPERATION_ID)?,
        )?;
    }
    upload_f32_with_context(
        backend,
        shape,
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub(crate) fn broadcast_view_to_shape_for_operation(
    input: &Tensor,
    output_shape: &[u64],
    operation: &'static str,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    let rank_offset = output_shape
        .len()
        .checked_sub(input.descriptor().rank())
        .ok_or(ElementwiseRuntimePartTwentyError::ShapeOverflow(
            "broadcast rank offset",
        ))?;
    let mut strides = vec![0_i64; output_shape.len()];
    for output_axis in rank_offset..output_shape.len() {
        let input_axis = output_axis - rank_offset;
        let input_dimension = input.descriptor().shape()[input_axis];
        let output_dimension = output_shape[output_axis];
        strides[output_axis] = if input_dimension == output_dimension {
            input.descriptor().strides()[input_axis]
        } else if input_dimension == 1 {
            0
        } else {
            return invalid(operation, "input shapes are not broadcast compatible");
        };
    }
    let descriptor = TensorDescriptor::new_strided(
        output_shape.to_vec(),
        strides,
        input.descriptor().offset_elements(),
        input.descriptor().dtype(),
        Layout::Strided,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    Ok(input.view(descriptor, ViewAccess::ReadOnly)?)
}

fn broadcast_view_to_shape(
    input: &Tensor,
    output_shape: &[u64],
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    broadcast_view_to_shape_for_operation(input, output_shape, BROADCAST_TENSORS_OPERATION_ID)
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyError> {
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ElementwiseRuntimePartTwentyError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        });
    }
    if input.descriptor().dtype() != DType::F32 {
        return Err(ElementwiseRuntimePartTwentyError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_matching_f32(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyError> {
    require_f32_cpu(input, operation)?;
    require_f32_cpu(other, operation)?;
    if input.descriptor().shape() != other.descriptor().shape()
        || input.descriptor().stream() != other.descriptor().stream()
    {
        return invalid(operation, "tensor shape and stream must match");
    }
    Ok(())
}

fn read_f32(
    input: &Tensor,
    indices: &[u64],
    operation: &'static str,
) -> Result<f32, ElementwiseRuntimePartTwentyError> {
    let bytes: [u8; 4] = input.element_bytes(indices)?.try_into().map_err(|_| {
        ElementwiseRuntimePartTwentyError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        }
    })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: crate::StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyError> {
    context.check()?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    value: T,
) -> Result<crate::CpuWorkspaceVec<T>, ElementwiseRuntimePartTwentyError> {
    let mut values = backend.workspace_vec(context, capacity)?;
    for _ in 0..capacity {
        values.try_push(value)?;
    }
    Ok(values)
}

fn normalize_axis(
    axis: i64,
    rank: usize,
    operation: &'static str,
) -> Result<usize, ElementwiseRuntimePartTwentyError> {
    if rank == 0 {
        return invalid(operation, "axis is invalid for a scalar tensor");
    }
    let rank_i64 = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("tensor rank"))?;
    let normalized = if axis < 0 {
        axis.checked_add(rank_i64)
            .ok_or(ElementwiseRuntimePartTwentyError::ShapeOverflow(
                "normalized axis",
            ))?
    } else {
        axis
    };
    if normalized < 0 || normalized >= rank_i64 {
        return invalid(operation, "axis is out of range");
    }
    usize::try_from(normalized)
        .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("normalized axis"))
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartTwentyError> {
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(ElementwiseRuntimePartTwentyError::ShapeOverflow(
                "element count",
            ))
    })?;
    usize::try_from(count)
        .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("element count"))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartTwentyError> {
    let mut indices = vec![0_u64; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        if *dimension == 0 {
            return invalid(
                BROADCAST_TENSORS_OPERATION_ID,
                "cannot index an empty dimension",
            );
        }
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("index dimension"))?;
        indices[axis] = u64::try_from(linear % dimension)
            .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("logical index"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn ravel_index(indices: &[u64], shape: &[u64]) -> Result<usize, ElementwiseRuntimePartTwentyError> {
    indices
        .iter()
        .zip(shape)
        .try_fold(0_usize, |linear, (index, dimension)| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("ravel dimension"))?;
            let index = usize::try_from(*index)
                .map_err(|_| ElementwiseRuntimePartTwentyError::ShapeOverflow("ravel index"))?;
            linear
                .checked_mul(dimension)
                .and_then(|value| value.checked_add(index))
                .ok_or(ElementwiseRuntimePartTwentyError::ShapeOverflow(
                    "ravel index",
                ))
        })
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartTwentyError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, ElementwiseRuntimePartTwentyError> {
    Err(ElementwiseRuntimePartTwentyError::Invalid {
        operation,
        reason: reason.into(),
    })
}
