use crate::cpu_backend::CpuWorkspaceVec;
use crate::{
    AutogradError, AutogradTape, BackendCapabilityMatrix, CancellationToken, CpuBackend, DType,
    DecodedScalar, DeviceId, ExecutionContext, GradientReducer, LeafId, OutputSlot, StreamId,
    Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
    generated_elementwise_or_runtime_operation_02::ElementwiseRuntimePartTwoError,
    generated_elementwise_or_runtime_operation_03::ElementwiseRuntimePartThreeError,
};
use sim_types::DeviceKind;
use thiserror::Error;

type TemporaryVec<T> = CpuWorkspaceVec<T>;

pub const BYTE_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-5DFBC70338A1";
pub const LOG_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-5ED49DCA2F78";
pub const AUTOGRAD_GRAD_OPERATION_ID: &str = "COMFY-TENSOR-OP-5CEC7CF2D62D";
pub const MPS_IS_AVAILABLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-5BF1DE9DA499";
pub const CONCATENATE_OPERATION_ID: &str = "COMFY-TENSOR-OP-5C52F193416C";
pub const COS_OPERATION_ID: &str = "COMFY-TENSOR-OP-5C00AB949613";
pub const EXPM1_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-5D7C103AB024";
pub const INDEX_SELECT_OPERATION_ID: &str = "COMFY-TENSOR-OP-5AB4376A79B5";
pub const MEDIAN_OPERATION_ID: &str = "COMFY-TENSOR-OP-5BA79209BB02";
pub const MLU_CURRENT_DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-5EECCD4F0130";
pub const ROT90_OPERATION_ID: &str = "COMFY-TENSOR-OP-5CDFF9F97B6F";
pub const SQUARE_OPERATION_ID: &str = "COMFY-TENSOR-OP-60A72EC2F5DD";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartEightError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    PartTwo(#[from] ElementwiseRuntimePartTwoError),
    #[error(transparent)]
    PartThree(#[from] ElementwiseRuntimePartThreeError),
    #[error("elementwise/runtime part-eight operation was cancelled")]
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
    #[error("elementwise/runtime part-eight input is invalid: {0}")]
    Invalid(String),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartEightError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn byte_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_cpu(input, BYTE_METHOD_OPERATION_ID)?;
    let shape = input.descriptor().shape();
    let count = element_count(shape)?;
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::U8,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    let (mut output, _) = backend.allocate(descriptor, context)?;
    let mut write = output.write()?;
    let output_bytes = write.bytes_mut()?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, shape)?;
        let decoded = input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?;
        output_bytes[linear] = scalar_to_byte(input.descriptor().dtype(), decoded)?;
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

pub fn log_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    Ok(
        crate::generated_elementwise_or_runtime_operation_02::log_with_context_exact_native(
            backend, input, context,
        )?,
    )
}

pub fn log_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    Ok(
        crate::generated_elementwise_or_runtime_operation_02::log_vjp_with_context_exact_native(
            backend,
            input,
            output_gradient,
            context,
        )?,
    )
}

pub fn log_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    Ok(
        crate::generated_elementwise_or_runtime_operation_02::log_jvp_with_context_exact_native(
            backend,
            input,
            input_tangent,
            context,
        )?,
    )
}

pub fn autograd_grad_exact_native(
    tape: &mut AutogradTape,
    seeds: Vec<(OutputSlot, Tensor)>,
    inputs: &[LeafId],
    reducer: &dyn GradientReducer,
    allow_unused: bool,
    cancellation: &CancellationToken,
) -> Result<Vec<Option<Tensor>>, ElementwiseRuntimePartEightError> {
    cancellation.check()?;
    if inputs.is_empty() {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "autograd grad requires at least one input leaf".to_owned(),
        ));
    }
    let mut gradients = tape.backward(seeds, reducer, cancellation)?;
    let mut ordered = Vec::new();
    ordered
        .try_reserve_exact(inputs.len())
        .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("autograd grad output"))?;
    for (index, input) in inputs.iter().enumerate() {
        check_periodically(index, cancellation)?;
        let gradient = gradients.remove(input);
        if gradient.is_none() && !allow_unused {
            return Err(ElementwiseRuntimePartEightError::Invalid(format!(
                "autograd input leaf {:?} was unused",
                input.as_str()
            )));
        }
        ordered.push(gradient);
    }
    cancellation.check()?;
    Ok(ordered)
}

pub fn mps_is_available_exact_native(
    capabilities: &[BackendCapabilityMatrix],
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartEightError> {
    cancellation.check()?;
    for (index, capability) in capabilities.iter().enumerate() {
        check_periodically(index, cancellation)?;
        if capability.device().kind() == DeviceKind::Metal {
            cancellation.check()?;
            return Ok(true);
        }
    }
    cancellation.check()?;
    Ok(false)
}

pub fn concatenate_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    let first = inputs.first().ok_or_else(|| {
        ElementwiseRuntimePartEightError::Invalid("concatenate input list is empty".to_owned())
    })?;
    require_cpu(first, CONCATENATE_OPERATION_ID)?;
    let rank = first.descriptor().rank();
    let axis = normalize_axis(dimension, rank)?;
    let mut output_shape = first.descriptor().shape().to_vec();
    output_shape[axis] = 0;
    for (input_index, input) in inputs.iter().enumerate() {
        check_periodically(input_index, context.cancellation)?;
        require_compatible_tensor(first, input, CONCATENATE_OPERATION_ID)?;
        if input.descriptor().rank() != rank {
            return Err(ElementwiseRuntimePartEightError::Invalid(
                "concatenate inputs must have matching ranks".to_owned(),
            ));
        }
        for (current_axis, (actual, expected)) in input
            .descriptor()
            .shape()
            .iter()
            .zip(first.descriptor().shape())
            .enumerate()
        {
            if current_axis != axis && actual != expected {
                return Err(ElementwiseRuntimePartEightError::Invalid(
                    "concatenate non-axis dimensions must match".to_owned(),
                ));
            }
        }
        output_shape[axis] = output_shape[axis]
            .checked_add(input.descriptor().shape()[axis])
            .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
                "concatenate axis",
            ))?;
    }
    let mut output = allocate_like_with_context(backend, first, output_shape, context)?;
    let mut write = output.write()?;
    let mut axis_offset = 0_u64;
    for input in inputs {
        let count = element_count(input.descriptor().shape())?;
        for linear in 0..count {
            check_periodically(linear, context.cancellation)?;
            let input_indices = unravel_index(linear, input.descriptor().shape())?;
            let mut output_indices = input_indices.clone();
            output_indices[axis] = output_indices[axis].checked_add(axis_offset).ok_or(
                ElementwiseRuntimePartEightError::ShapeOverflow("concatenate output index"),
            )?;
            write
                .element_bytes_mut(&output_indices)?
                .copy_from_slice(input.element_bytes(&input_indices)?);
        }
        axis_offset = axis_offset
            .checked_add(input.descriptor().shape()[axis])
            .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
                "concatenate axis offset",
            ))?;
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

pub fn concatenate_vjp_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    dimension: i64,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Tensor>, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    let expected = concatenate_with_context_exact_native(backend, inputs, dimension, context)?;
    require_same_descriptor_shape_dtype_stream(
        &expected,
        output_gradient,
        CONCATENATE_OPERATION_ID,
    )?;
    let axis = normalize_axis(dimension, output_gradient.descriptor().rank())?;
    let mut gradients = Vec::new();
    gradients
        .try_reserve_exact(inputs.len())
        .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("concatenate VJP"))?;
    let mut axis_offset = 0_u64;
    for input in inputs {
        gradients.push(copy_axis_range_with_context(
            backend,
            output_gradient,
            input.descriptor().shape(),
            axis,
            axis_offset,
            context,
        )?);
        axis_offset = axis_offset
            .checked_add(input.descriptor().shape()[axis])
            .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
                "concatenate VJP offset",
            ))?;
    }
    Ok(gradients)
}

pub fn concatenate_jvp_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    input_tangents: &[Tensor],
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    if inputs.len() != input_tangents.len() {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "concatenate primals and tangents must have matching lengths".to_owned(),
        ));
    }
    for (input, tangent) in inputs.iter().zip(input_tangents) {
        require_same_descriptor_shape_dtype_stream(input, tangent, CONCATENATE_OPERATION_ID)?;
    }
    concatenate_with_context_exact_native(backend, input_tangents, dimension, context)
}

pub fn cos_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    unary_f32_with_context(
        backend,
        input,
        UnaryOperation::Cosine,
        COS_OPERATION_ID,
        context,
    )
}

pub fn cos_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    map_same_shape_f32_with_context(
        backend,
        input,
        output_gradient,
        COS_OPERATION_ID,
        context,
        |value, gradient| -value.sin() * gradient,
    )
}

pub fn cos_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    cos_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

pub fn expm1_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    Ok(
        crate::generated_elementwise_or_runtime_operation_03::expm1_with_context_exact_native(
            backend, input, context,
        )?,
    )
}

pub fn expm1_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    Ok(
        crate::generated_elementwise_or_runtime_operation_03::expm1_vjp_with_context_exact_native(
            backend,
            input,
            output_gradient,
            context,
        )?,
    )
}

pub fn expm1_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    Ok(
        crate::generated_elementwise_or_runtime_operation_03::expm1_jvp_with_context_exact_native(
            backend,
            input,
            input_tangent,
            context,
        )?,
    )
}

pub fn index_select_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    indices: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_cpu(input, INDEX_SELECT_OPERATION_ID)?;
    require_cpu(indices, INDEX_SELECT_OPERATION_ID)?;
    if indices.descriptor().dtype() != DType::I64 {
        return Err(ElementwiseRuntimePartEightError::UnsupportedDType {
            operation: INDEX_SELECT_OPERATION_ID,
            dtype: indices.descriptor().dtype(),
        });
    }
    if indices.descriptor().rank() != 1 {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "index_select indices must be one-dimensional".to_owned(),
        ));
    }
    if input.descriptor().stream() != indices.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: input.descriptor().stream(),
            actual: indices.descriptor().stream(),
        }
        .into());
    }
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let mut output_shape = input.descriptor().shape().to_vec();
    output_shape[axis] = indices.descriptor().shape()[0];
    let selected =
        decode_indices_with_context(backend, indices, input.descriptor().shape()[axis], context)?;
    let mut output = allocate_like_with_context(backend, input, output_shape.clone(), context)?;
    let mut write = output.write()?;
    for linear in 0..element_count(&output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let select_position = usize::try_from(output_indices[axis]).map_err(|_| {
            ElementwiseRuntimePartEightError::ShapeOverflow("index_select position")
        })?;
        let selected_index = *selected.get(select_position).ok_or_else(|| {
            ElementwiseRuntimePartEightError::Invalid(
                "index_select position is outside the index vector".to_owned(),
            )
        })?;
        let mut input_indices = output_indices.clone();
        input_indices[axis] = selected_index;
        write
            .element_bytes_mut(&output_indices)?
            .copy_from_slice(input.element_bytes(&input_indices)?);
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

pub fn index_select_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    indices: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_f32_cpu(input, INDEX_SELECT_OPERATION_ID)?;
    let selected_output =
        index_select_with_context_exact_native(backend, input, dimension, indices, context)?;
    require_same_descriptor_shape_dtype_stream(
        &selected_output,
        output_gradient,
        INDEX_SELECT_OPERATION_ID,
    )?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let selected =
        decode_indices_with_context(backend, indices, input.descriptor().shape()[axis], context)?;
    let mut gradient_values = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    for linear in 0..element_count(output_gradient.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, output_gradient.descriptor().shape())?;
        let select_position = usize::try_from(output_indices[axis]).map_err(|_| {
            ElementwiseRuntimePartEightError::ShapeOverflow("index_select VJP position")
        })?;
        let mut input_indices = output_indices.clone();
        input_indices[axis] = selected[select_position];
        let input_linear = linear_index(&input_indices, input.descriptor().shape())?;
        gradient_values[input_linear] += read_f32(output_gradient, &output_indices)?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &gradient_values,
        context,
    )
}

pub fn index_select_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    indices: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_same_descriptor_shape_dtype_stream(input, input_tangent, INDEX_SELECT_OPERATION_ID)?;
    index_select_with_context_exact_native(backend, input_tangent, dimension, indices, context)
}

#[derive(Debug)]
pub struct MedianDimensionResult {
    pub values: Tensor,
    pub indices: Tensor,
}

pub fn median_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    keep_dimension: bool,
    context: &ExecutionContext<'_>,
) -> Result<MedianDimensionResult, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_f32_cpu(input, MEDIAN_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let axis_length = input.descriptor().shape()[axis];
    if axis_length == 0 {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "median cannot reduce an empty dimension".to_owned(),
        ));
    }
    let output_shape = reduced_shape(input.descriptor().shape(), axis, keep_dimension);
    let output_count = element_count(&output_shape)?;
    let mut values = backend.workspace_vec(context, output_count)?;
    let mut indices = backend.workspace_vec(context, output_count)?;
    for output_linear in 0..output_count {
        check_periodically(output_linear, context.cancellation)?;
        let output_indices = unravel_index(output_linear, &output_shape)?;
        let base_indices = expand_reduced_indices(&output_indices, axis, keep_dimension)?;
        let (value, selected) =
            median_slice_with_context(backend, input, &base_indices, axis, axis_length, context)?;
        values.try_push(value)?;
        indices.try_push(i64::try_from(selected).map_err(|_| {
            ElementwiseRuntimePartEightError::ShapeOverflow("median selected index")
        })?)?;
    }
    Ok(MedianDimensionResult {
        values: upload_f32_with_context(
            backend,
            &output_shape,
            input.descriptor().stream(),
            &values,
            context,
        )?,
        indices: upload_i64_with_context(
            backend,
            &output_shape,
            input.descriptor().stream(),
            &indices,
            context,
        )?,
    })
}

pub fn median_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    keep_dimension: bool,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    let median =
        median_with_context_exact_native(backend, input, dimension, keep_dimension, context)?;
    require_same_descriptor_shape_dtype_stream(
        &median.values,
        output_gradient,
        MEDIAN_OPERATION_ID,
    )?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let mut values = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    for linear in 0..element_count(output_gradient.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, output_gradient.descriptor().shape())?;
        let mut input_indices = expand_reduced_indices(&output_indices, axis, keep_dimension)?;
        input_indices[axis] = read_i64(&median.indices, &output_indices)?
            .try_into()
            .map_err(|_| {
                ElementwiseRuntimePartEightError::Invalid(
                    "median gradient index is negative".to_owned(),
                )
            })?;
        let input_linear = linear_index(&input_indices, input.descriptor().shape())?;
        values[input_linear] += read_f32(output_gradient, &output_indices)?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn median_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    dimension: i64,
    keep_dimension: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_same_descriptor_shape_dtype_stream(input, input_tangent, MEDIAN_OPERATION_ID)?;
    let median =
        median_with_context_exact_native(backend, input, dimension, keep_dimension, context)?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let shape = median.values.descriptor().shape().to_vec();
    let mut values = backend.workspace_vec(context, element_count(&shape)?)?;
    for linear in 0..element_count(&shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &shape)?;
        let mut input_indices = expand_reduced_indices(&output_indices, axis, keep_dimension)?;
        input_indices[axis] = read_i64(&median.indices, &output_indices)?
            .try_into()
            .map_err(|_| {
                ElementwiseRuntimePartEightError::Invalid(
                    "median tangent index is negative".to_owned(),
                )
            })?;
        values.try_push(read_f32(input_tangent, &input_indices)?)?;
    }
    upload_f32_with_context(
        backend,
        &shape,
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn mlu_current_device_exact_native(
    capabilities: &BackendCapabilityMatrix,
    cancellation: &CancellationToken,
) -> Result<u32, ElementwiseRuntimePartEightError> {
    cancellation.check()?;
    let device = capabilities.device();
    if device.kind() != DeviceKind::Mlu {
        return Err(ElementwiseRuntimePartEightError::UnsupportedDevice {
            operation: MLU_CURRENT_DEVICE_OPERATION_ID,
            device,
        });
    }
    cancellation.check()?;
    Ok(device.ordinal())
}

pub fn rot90_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    quarter_turns: i64,
    dimensions: [i64; 2],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_cpu(input, ROT90_OPERATION_ID)?;
    let first_axis = normalize_axis(dimensions[0], input.descriptor().rank())?;
    let second_axis = normalize_axis(dimensions[1], input.descriptor().rank())?;
    if first_axis == second_axis {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "rot90 dimensions must be distinct".to_owned(),
        ));
    }
    let turns = quarter_turns.rem_euclid(4) as u8;
    let mut output_shape = input.descriptor().shape().to_vec();
    if turns % 2 == 1 {
        output_shape.swap(first_axis, second_axis);
    }
    let mut output = allocate_like_with_context(backend, input, output_shape.clone(), context)?;
    let first_length = input.descriptor().shape()[first_axis];
    let second_length = input.descriptor().shape()[second_axis];
    let mut write = output.write()?;
    for linear in 0..element_count(&output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let mut input_indices = output_indices.clone();
        match turns {
            0 => {}
            1 => {
                input_indices[first_axis] = output_indices[second_axis];
                input_indices[second_axis] = second_length
                    .checked_sub(1)
                    .and_then(|last| last.checked_sub(output_indices[first_axis]))
                    .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
                        "rot90 index",
                    ))?;
            }
            2 => {
                input_indices[first_axis] = first_length
                    .checked_sub(1)
                    .and_then(|last| last.checked_sub(output_indices[first_axis]))
                    .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
                        "rot90 index",
                    ))?;
                input_indices[second_axis] = second_length
                    .checked_sub(1)
                    .and_then(|last| last.checked_sub(output_indices[second_axis]))
                    .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
                        "rot90 index",
                    ))?;
            }
            3 => {
                input_indices[first_axis] = first_length
                    .checked_sub(1)
                    .and_then(|last| last.checked_sub(output_indices[second_axis]))
                    .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
                        "rot90 index",
                    ))?;
                input_indices[second_axis] = output_indices[first_axis];
            }
            _ => {
                return Err(ElementwiseRuntimePartEightError::Invalid(
                    "rot90 normalized turn is invalid".to_owned(),
                ));
            }
        }
        write
            .element_bytes_mut(&output_indices)?
            .copy_from_slice(input.element_bytes(&input_indices)?);
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

pub fn rot90_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    quarter_turns: i64,
    dimensions: [i64; 2],
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    let output =
        rot90_with_context_exact_native(backend, input, quarter_turns, dimensions, context)?;
    require_same_descriptor_shape_dtype_stream(&output, output_gradient, ROT90_OPERATION_ID)?;
    rot90_with_context_exact_native(
        backend,
        output_gradient,
        -quarter_turns,
        dimensions,
        context,
    )
}

pub fn rot90_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    quarter_turns: i64,
    dimensions: [i64; 2],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_same_descriptor_shape_dtype_stream(input, input_tangent, ROT90_OPERATION_ID)?;
    rot90_with_context_exact_native(backend, input_tangent, quarter_turns, dimensions, context)
}

pub fn square_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    require_f32_cpu(input, SQUARE_OPERATION_ID)?;
    map_f32_with_context(backend, input, SQUARE_OPERATION_ID, context, |value| {
        value * value
    })
}

pub fn square_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    map_same_shape_f32_with_context(
        backend,
        input,
        output_gradient,
        SQUARE_OPERATION_ID,
        context,
        |value, gradient| 2.0 * value * gradient,
    )
}

pub fn square_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    context.cancellation.check()?;
    square_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

fn scalar_to_byte(
    dtype: DType,
    value: DecodedScalar,
) -> Result<u8, ElementwiseRuntimePartEightError> {
    Ok(match value {
        DecodedScalar::Boolean(value) => u8::from(value),
        DecodedScalar::Signed(value) => value.rem_euclid(256) as u8,
        DecodedScalar::Unsigned(value) => (value % 256) as u8,
        DecodedScalar::Real(value) => value.trunc().rem_euclid(256.0) as u8,
        DecodedScalar::Complex { .. } => {
            return Err(ElementwiseRuntimePartEightError::UnsupportedDType {
                operation: BYTE_METHOD_OPERATION_ID,
                dtype,
            });
        }
    })
}

fn unary_f32_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    operation: UnaryOperation,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    require_f32_cpu(input, operation_id)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.unary(operation, input, descriptor, context)?.0)
}

fn map_f32_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
    operation: impl Fn(f32) -> f32,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    require_f32_cpu(input, operation_id)?;
    let count = element_count(input.descriptor().shape())?;
    let mut values = temporary_vec(backend, context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        values.try_push(operation(read_f32(
            input,
            &unravel_index(linear, input.descriptor().shape())?,
        )?))?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

fn map_same_shape_f32_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
    operation: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    require_f32_cpu(input, operation_id)?;
    require_f32_cpu(other, operation_id)?;
    require_same_descriptor_shape_dtype_stream(input, other, operation_id)?;
    let count = element_count(input.descriptor().shape())?;
    let mut values = temporary_vec(backend, context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        values.try_push(operation(
            read_f32(input, &indices)?,
            read_f32(other, &indices)?,
        ))?;
    }
    upload_f32_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &values,
        context,
    )
}

fn allocate_like_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    shape: Vec<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    let descriptor = TensorDescriptor::contiguous(
        shape,
        input.descriptor().dtype(),
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend.allocate(descriptor, context)?.0)
}

fn copy_axis_range_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    output_shape: &[u64],
    axis: usize,
    axis_offset: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    let mut output = allocate_like_with_context(backend, input, output_shape.to_vec(), context)?;
    let mut write = output.write()?;
    for linear in 0..element_count(output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, output_shape)?;
        let mut input_indices = output_indices.clone();
        input_indices[axis] = input_indices[axis].checked_add(axis_offset).ok_or(
            ElementwiseRuntimePartEightError::ShapeOverflow("axis range index"),
        )?;
        write
            .element_bytes_mut(&output_indices)?
            .copy_from_slice(input.element_bytes(&input_indices)?);
    }
    drop(write);
    context.cancellation.check()?;
    Ok(output)
}

fn decode_indices_with_context(
    backend: &CpuBackend,
    indices: &Tensor,
    input_axis_length: u64,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<u64>, ElementwiseRuntimePartEightError> {
    let count = element_count(indices.descriptor().shape())?;
    let mut decoded = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let value = read_i64(
            indices,
            &[u64::try_from(linear).map_err(|_| {
                ElementwiseRuntimePartEightError::ShapeOverflow("index vector position")
            })?],
        )?;
        let value = u64::try_from(value).map_err(|_| {
            ElementwiseRuntimePartEightError::Invalid(
                "index_select indices must be nonnegative".to_owned(),
            )
        })?;
        if value >= input_axis_length {
            return Err(ElementwiseRuntimePartEightError::Invalid(
                "index_select index is outside the selected dimension".to_owned(),
            ));
        }
        decoded.try_push(value)?;
    }
    Ok(decoded)
}

fn median_slice_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    base_indices: &[u64],
    axis: usize,
    axis_length: u64,
    context: &ExecutionContext<'_>,
) -> Result<(f32, u64), ElementwiseRuntimePartEightError> {
    let length = usize::try_from(axis_length)
        .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("median axis"))?;
    let mut candidates = backend.workspace_vec(context, length)?;
    for position in 0..length {
        check_periodically(position, context.cancellation)?;
        let mut indices = base_indices.to_vec();
        indices[axis] = u64::try_from(position)
            .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("median index"))?;
        let value = read_f32(input, &indices)?;
        if value.is_nan() {
            return Ok((value, indices[axis]));
        }
        candidates.try_push((value, indices[axis]))?;
    }
    context.cancellation.check()?;
    candidates.sort_unstable_by(|(left_value, left_index), (right_value, right_index)| {
        left_value
            .total_cmp(right_value)
            .then(left_index.cmp(right_index))
    });
    context.cancellation.check()?;
    candidates.get((length - 1) / 2).copied().ok_or_else(|| {
        ElementwiseRuntimePartEightError::Invalid("median slice is empty".to_owned())
    })
}

fn reduced_shape(shape: &[u64], axis: usize, keep_dimension: bool) -> Vec<u64> {
    if keep_dimension {
        let mut reduced = shape.to_vec();
        reduced[axis] = 1;
        reduced
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(index, dimension)| (index != axis).then_some(*dimension))
            .collect()
    }
}

fn expand_reduced_indices(
    indices: &[u64],
    axis: usize,
    keep_dimension: bool,
) -> Result<Vec<u64>, ElementwiseRuntimePartEightError> {
    if keep_dimension {
        let mut expanded = indices.to_vec();
        let slot = expanded.get_mut(axis).ok_or_else(|| {
            ElementwiseRuntimePartEightError::Invalid(
                "reduced index does not contain its kept dimension".to_owned(),
            )
        })?;
        *slot = 0;
        Ok(expanded)
    } else {
        let mut expanded = Vec::new();
        expanded
            .try_reserve_exact(indices.len() + 1)
            .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("expanded index"))?;
        let mut source = 0;
        for target in 0..=indices.len() {
            if target == axis {
                expanded.push(0);
            } else {
                expanded.push(*indices.get(source).ok_or_else(|| {
                    ElementwiseRuntimePartEightError::Invalid(
                        "reduced index is missing a dimension".to_owned(),
                    )
                })?);
                source += 1;
            }
        }
        Ok(expanded)
    }
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    if values.len() != element_count(shape)? {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "f32 upload length does not match its shape".to_owned(),
        ));
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
) -> Result<TemporaryVec<T>, ElementwiseRuntimePartEightError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn upload_i64_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartEightError> {
    if values.len() != element_count(shape)? {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "i64 upload length does not match its shape".to_owned(),
        ));
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, stream)?;
    let (mut tensor, _) = backend.allocate(descriptor, context)?;
    let mut write = tensor.write()?;
    for (index, (chunk, value)) in write
        .bytes_mut()?
        .chunks_exact_mut(8)
        .zip(values)
        .enumerate()
    {
        check_periodically(index, context.cancellation)?;
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
    drop(write);
    context.cancellation.check()?;
    Ok(tensor)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, ElementwiseRuntimePartEightError> {
    let mut values = backend.workspace_vec(context, capacity)?;
    for _ in 0..capacity {
        values.try_push(value)?;
    }
    Ok(values)
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartEightError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartEightError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartEightError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartEightError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}

fn require_compatible_tensor(
    first: &Tensor,
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartEightError> {
    require_cpu(input, operation)?;
    if first.descriptor().dtype() != input.descriptor().dtype() {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "tensor dtypes must match".to_owned(),
        ));
    }
    if first.descriptor().stream() != input.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: first.descriptor().stream(),
            actual: input.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn require_same_descriptor_shape_dtype_stream(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartEightError> {
    require_compatible_tensor(input, other, operation)?;
    if input.descriptor().shape() != other.descriptor().shape() {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "tensor shapes must match".to_owned(),
        ));
    }
    Ok(())
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, ElementwiseRuntimePartEightError> {
    if rank == 0 {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "operation requires a non-scalar tensor".to_owned(),
        ));
    }
    let rank = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("tensor rank"))?;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if !(0..rank).contains(&normalized) {
        return Err(ElementwiseRuntimePartEightError::Invalid(
            "dimension is outside the tensor rank".to_owned(),
        ));
    }
    usize::try_from(normalized)
        .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("tensor axis"))
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartEightError> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
            "element count",
        ))?;
    usize::try_from(count)
        .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("element count"))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartEightError> {
    let mut indices = vec![0; shape.len()];
    for (slot, dimension) in indices.iter_mut().zip(shape).rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("tensor index"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartEightError::Invalid(
                "cannot index an empty tensor".to_owned(),
            ));
        }
        *slot = u64::try_from(linear % dimension)
            .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("tensor index"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn linear_index(indices: &[u64], shape: &[u64]) -> Result<usize, ElementwiseRuntimePartEightError> {
    let mut linear = 0_u64;
    for (index, dimension) in indices.iter().zip(shape) {
        linear = linear
            .checked_mul(*dimension)
            .and_then(|value| value.checked_add(*index))
            .ok_or(ElementwiseRuntimePartEightError::ShapeOverflow(
                "linear index",
            ))?;
    }
    usize::try_from(linear)
        .map_err(|_| ElementwiseRuntimePartEightError::ShapeOverflow("linear index"))
}

fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, ElementwiseRuntimePartEightError> {
    let bytes: [u8; 4] = tensor.element_bytes(indices)?.try_into().map_err(|_| {
        ElementwiseRuntimePartEightError::Invalid("f32 element width is invalid".to_owned())
    })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn read_i64(tensor: &Tensor, indices: &[u64]) -> Result<i64, ElementwiseRuntimePartEightError> {
    let bytes: [u8; 8] = tensor.element_bytes(indices)?.try_into().map_err(|_| {
        ElementwiseRuntimePartEightError::Invalid("i64 element width is invalid".to_owned())
    })?;
    Ok(i64::from_ne_bytes(bytes))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartEightError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}
