use crate::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DeviceId,
    ExecutionContext, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError,
    UnaryOperation,
    cpu_backend::{binary_broadcast_shape, broadcast_indices},
    generated_elementwise_or_runtime_operation_02::{
        ElementwiseRuntimePartTwoError,
        tanh_jvp_with_context_exact_native as canonical_tanh_jvp_with_context,
        tanh_vjp_with_context_exact_native as canonical_tanh_vjp_with_context,
        tanh_with_context_exact_native as canonical_tanh_with_context,
    },
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseRuntimePartThreeError,
        expm1_jvp_with_context_exact_native as canonical_expm1_jvp_with_context,
        expm1_vjp_with_context_exact_native as canonical_expm1_vjp_with_context,
        expm1_with_context_exact_native as canonical_expm1_with_context,
    },
};
use sim_types::DeviceKind;
use thiserror::Error;

pub const EXPM1_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-5A1598AB1BFB";
pub const ADDCDIV_OPERATION_ID: &str = "COMFY-TENSOR-OP-5668EBF27561";
pub const ARGWHERE_OPERATION_ID: &str = "COMFY-TENSOR-OP-59C70700F28E";
pub const LOG1P_OPERATION_ID: &str = "COMFY-TENSOR-OP-56E8CFEB8E84";
pub const WEIGHT_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-58AE3CA27BFE";
pub const SGD_OPERATION_ID: &str = "COMFY-TENSOR-OP-594BD684E5EF";
pub const OUTER_OPERATION_ID: &str = "COMFY-TENSOR-OP-59EBFDE56C4F";
pub const RSQRT_OPERATION_ID: &str = "COMFY-TENSOR-OP-54E28780B32B";
pub const SET_PRINTOPTIONS_OPERATION_ID: &str = "COMFY-TENSOR-OP-5547BE508AEE";
pub const TANH_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-59AD8FFF431A";
pub const XPU_CURRENT_STREAM_OPERATION_ID: &str = "COMFY-TENSOR-OP-576587FE2EAF";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartSevenError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartTwo(#[from] ElementwiseRuntimePartTwoError),
    #[error(transparent)]
    PartThree(#[from] ElementwiseRuntimePartThreeError),
    #[error("elementwise/runtime part-seven operation was cancelled")]
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
    #[error("elementwise/runtime part-seven input is invalid: {0}")]
    Invalid(&'static str),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartSevenError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn expm1_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    Ok(canonical_expm1_with_context(backend, input, context)?)
}

pub fn expm1_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    Ok(canonical_expm1_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn expm1_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    Ok(canonical_expm1_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn addcdiv_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    tensor1: &Tensor,
    tensor2: &Tensor,
    value: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_f32_triplet(input, tensor1, tensor2, ADDCDIV_OPERATION_ID)?;
    map_broadcast_three_with_context(
        backend,
        input,
        tensor1,
        tensor2,
        context,
        |base, left, right| base + value * left / right,
    )
}

#[derive(Debug)]
pub struct AddcdivGradients {
    pub input: Tensor,
    pub tensor1: Tensor,
    pub tensor2: Tensor,
}

pub fn addcdiv_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    tensor1: &Tensor,
    tensor2: &Tensor,
    value: f32,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<AddcdivGradients, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_f32_triplet(input, tensor1, tensor2, ADDCDIV_OPERATION_ID)?;
    require_matching_stream_dtype(input, output_gradient, ADDCDIV_OPERATION_ID)?;
    let output_shape = broadcast_shape_three(
        input.descriptor().shape(),
        tensor1.descriptor().shape(),
        tensor2.descriptor().shape(),
    )?;
    if output_gradient.descriptor().shape() != output_shape {
        return Err(ElementwiseRuntimePartSevenError::Invalid(
            "addcdiv output gradient shape must match the broadcast output",
        ));
    }
    let input_count = element_count(input.descriptor().shape())?;
    let tensor1_count = element_count(tensor1.descriptor().shape())?;
    let tensor2_count = element_count(tensor2.descriptor().shape())?;
    let mut input_gradient = zeroed_workspace(backend, context, input_count)?;
    let mut tensor1_gradient = zeroed_workspace(backend, context, tensor1_count)?;
    let mut tensor2_gradient = zeroed_workspace(backend, context, tensor2_count)?;
    for linear in 0..element_count(&output_shape)? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, &output_shape)?;
        let input_indices = broadcast_indices(&output_indices, input.descriptor().shape())?;
        let tensor1_indices = broadcast_indices(&output_indices, tensor1.descriptor().shape())?;
        let tensor2_indices = broadcast_indices(&output_indices, tensor2.descriptor().shape())?;
        let gradient = read_f32(output_gradient, &output_indices)?;
        let left = read_f32(tensor1, &tensor1_indices)?;
        let right = read_f32(tensor2, &tensor2_indices)?;
        input_gradient[linear_index(&input_indices, input.descriptor().shape())?] += gradient;
        tensor1_gradient[linear_index(&tensor1_indices, tensor1.descriptor().shape())?] +=
            gradient * value / right;
        tensor2_gradient[linear_index(&tensor2_indices, tensor2.descriptor().shape())?] +=
            -gradient * value * left / (right * right);
    }
    Ok(AddcdivGradients {
        input: upload_f32_with_context(
            backend,
            input.descriptor().shape(),
            input.descriptor().stream(),
            &input_gradient,
            context,
        )?,
        tensor1: upload_f32_with_context(
            backend,
            tensor1.descriptor().shape(),
            tensor1.descriptor().stream(),
            &tensor1_gradient,
            context,
        )?,
        tensor2: upload_f32_with_context(
            backend,
            tensor2.descriptor().shape(),
            tensor2.descriptor().stream(),
            &tensor2_gradient,
            context,
        )?,
    })
}

pub fn addcdiv_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    tensor1: &Tensor,
    tensor2: &Tensor,
    value: f32,
    input_tangent: &Tensor,
    tensor1_tangent: &Tensor,
    tensor2_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_f32_triplet(input, tensor1, tensor2, ADDCDIV_OPERATION_ID)?;
    require_matching_f32(input, input_tangent, ADDCDIV_OPERATION_ID)?;
    require_matching_f32(tensor1, tensor1_tangent, ADDCDIV_OPERATION_ID)?;
    require_matching_f32(tensor2, tensor2_tangent, ADDCDIV_OPERATION_ID)?;
    let shape = broadcast_shape_three(
        input.descriptor().shape(),
        tensor1.descriptor().shape(),
        tensor2.descriptor().shape(),
    )?;
    let count = element_count(&shape)?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        let input_indices = broadcast_indices(&indices, input.descriptor().shape())?;
        let tensor1_indices = broadcast_indices(&indices, tensor1.descriptor().shape())?;
        let tensor2_indices = broadcast_indices(&indices, tensor2.descriptor().shape())?;
        let left = read_f32(tensor1, &tensor1_indices)?;
        let right = read_f32(tensor2, &tensor2_indices)?;
        let input_tangent = read_f32(input_tangent, &input_indices)?;
        let tensor1_tangent = read_f32(tensor1_tangent, &tensor1_indices)?;
        let tensor2_tangent = read_f32(tensor2_tangent, &tensor2_indices)?;
        values.try_push(
            input_tangent
                + value * (tensor1_tangent / right - left * tensor2_tangent / (right * right)),
        )?;
    }
    upload_f32_with_context(
        backend,
        &shape,
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn argwhere_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_cpu(input, ARGWHERE_OPERATION_ID)?;
    let rank = input.descriptor().shape().len();
    let element_count = element_count(input.descriptor().shape())?;
    let coordinate_capacity =
        element_count
            .checked_mul(rank)
            .ok_or(ElementwiseRuntimePartSevenError::ShapeOverflow(
                "argwhere coordinates",
            ))?;
    let mut coordinates = backend.workspace_vec::<i64>(context, coordinate_capacity)?;
    let mut matching_rows = 0_usize;
    for linear in 0..element_count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, input.descriptor().shape())?;
        if input
            .descriptor()
            .dtype()
            .decode_scalar(input.element_bytes(&indices)?)?
            .is_nonzero()
        {
            matching_rows = matching_rows.checked_add(1).ok_or(
                ElementwiseRuntimePartSevenError::ShapeOverflow("argwhere rows"),
            )?;
            for index in indices {
                coordinates.try_push(i64::try_from(index).map_err(|_| {
                    ElementwiseRuntimePartSevenError::ShapeOverflow("argwhere coordinate")
                })?)?;
            }
        }
    }
    upload_i64_with_context(
        backend,
        &[
            u64::try_from(matching_rows)
                .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("argwhere rows"))?,
            u64::try_from(rank)
                .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("argwhere rank"))?,
        ],
        input.descriptor().stream(),
        &coordinates,
        context,
    )
}

pub fn log1p_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, LOG1P_OPERATION_ID)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .unary(UnaryOperation::Log1p, input, descriptor, context)?
        .0)
}

pub fn log1p_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    map_same_shape_binary_with_context(
        backend,
        input,
        output_gradient,
        LOG1P_OPERATION_ID,
        context,
        |value, gradient| gradient / (1.0 + value),
    )
}

pub fn log1p_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    log1p_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

#[derive(Clone, Debug)]
pub struct NativeSgd {
    learning_rate: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    maximize: bool,
    momentum_buffers: Vec<Option<Tensor>>,
}

impl NativeSgd {
    #[allow(clippy::too_many_arguments)]
    pub fn new_exact_native(
        parameter_count: usize,
        learning_rate: f32,
        momentum: f32,
        dampening: f32,
        weight_decay: f32,
        nesterov: bool,
        maximize: bool,
        cancellation: &CancellationToken,
    ) -> Result<Self, ElementwiseRuntimePartSevenError> {
        cancellation.check()?;
        if parameter_count == 0
            || !learning_rate.is_finite()
            || learning_rate < 0.0
            || !momentum.is_finite()
            || momentum < 0.0
            || !dampening.is_finite()
            || dampening < 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
            || (nesterov && (momentum <= 0.0 || dampening != 0.0))
        {
            return Err(ElementwiseRuntimePartSevenError::Invalid(
                "invalid native SGD configuration",
            ));
        }
        Ok(Self {
            learning_rate,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            maximize,
            momentum_buffers: vec![None; parameter_count],
        })
    }

    pub fn step_with_context_exact_native(
        &mut self,
        backend: &CpuBackend,
        parameters: &mut [Tensor],
        gradients: &[Tensor],
        context: &ExecutionContext<'_>,
    ) -> Result<(), ElementwiseRuntimePartSevenError> {
        context.cancellation.check()?;
        if parameters.len() != gradients.len() || parameters.len() != self.momentum_buffers.len() {
            return Err(ElementwiseRuntimePartSevenError::Invalid(
                "SGD parameter, gradient, and state lengths must match",
            ));
        }
        let mut staged_parameters = backend.workspace_vec::<Tensor>(context, parameters.len())?;
        let mut staged_buffers =
            backend.workspace_vec::<Option<Tensor>>(context, parameters.len())?;
        for index in 0..parameters.len() {
            context.cancellation.check()?;
            require_matching_f32(&parameters[index], &gradients[index], SGD_OPERATION_ID)?;
            let count = element_count(parameters[index].descriptor().shape())?;
            let mut next_parameters = backend.workspace_vec::<f32>(context, count)?;
            let mut next_buffer = if self.momentum > 0.0 {
                Some(backend.workspace_vec::<f32>(context, count)?)
            } else {
                None
            };
            if let Some(previous) = self.momentum_buffers[index].as_ref() {
                require_matching_f32(&parameters[index], previous, SGD_OPERATION_ID)?;
            }
            for element in 0..count {
                check_periodically(element, context.cancellation)?;
                let indices = unravel_index(element, parameters[index].descriptor().shape())?;
                let parameter = read_f32(&parameters[index], &indices)?;
                let gradient = read_f32(&gradients[index], &indices)?;
                let mut direction = if self.maximize { -gradient } else { gradient };
                direction += self.weight_decay * parameter;
                if let Some(buffer_values) = next_buffer.as_mut() {
                    let buffer = if let Some(previous) = self.momentum_buffers[index].as_ref() {
                        self.momentum * read_f32(previous, &indices)?
                            + (1.0 - self.dampening) * direction
                    } else {
                        direction
                    };
                    buffer_values.try_push(buffer)?;
                    direction = if self.nesterov {
                        direction + self.momentum * buffer
                    } else {
                        buffer
                    };
                }
                next_parameters.try_push(parameter - self.learning_rate * direction)?;
            }
            staged_parameters.try_push(upload_f32_with_context(
                backend,
                parameters[index].descriptor().shape(),
                parameters[index].descriptor().stream(),
                &next_parameters,
                context,
            )?)?;
            staged_buffers.try_push(
                next_buffer
                    .as_ref()
                    .map(|values| {
                        upload_f32_with_context(
                            backend,
                            parameters[index].descriptor().shape(),
                            parameters[index].descriptor().stream(),
                            values,
                            context,
                        )
                    })
                    .transpose()?,
            )?;
        }
        context.cancellation.check()?;
        for (parameter, staged) in parameters.iter_mut().zip(staged_parameters.iter()) {
            parameter.commit_in_place(staged.clone())?;
        }
        self.momentum_buffers.clear();
        self.momentum_buffers.extend(staged_buffers.iter().cloned());
        Ok(())
    }
}

pub fn outer_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_matching_stream_dtype(input, other, OUTER_OPERATION_ID)?;
    if input.descriptor().shape().len() != 1 || other.descriptor().shape().len() != 1 {
        return Err(ElementwiseRuntimePartSevenError::Invalid(
            "outer requires two one-dimensional tensors",
        ));
    }
    let left_count = element_count(input.descriptor().shape())?;
    let right_count = element_count(other.descriptor().shape())?;
    let count = left_count.checked_mul(right_count).ok_or(
        ElementwiseRuntimePartSevenError::ShapeOverflow("outer output"),
    )?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for left_index in 0..left_count {
        for right_index in 0..right_count {
            let linear = left_index
                .checked_mul(right_count)
                .and_then(|value| value.checked_add(right_index))
                .ok_or(ElementwiseRuntimePartSevenError::ShapeOverflow(
                    "outer index",
                ))?;
            check_periodically(linear, context.cancellation)?;
            values.try_push(
                read_f32(
                    input,
                    &[u64::try_from(left_index).map_err(|_| {
                        ElementwiseRuntimePartSevenError::ShapeOverflow("outer left index")
                    })?],
                )? * read_f32(
                    other,
                    &[u64::try_from(right_index).map_err(|_| {
                        ElementwiseRuntimePartSevenError::ShapeOverflow("outer right index")
                    })?],
                )?,
            )?;
        }
    }
    upload_f32_with_context(
        backend,
        &[
            u64::try_from(left_count)
                .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("outer left"))?,
            u64::try_from(right_count)
                .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("outer right"))?,
        ],
        input.descriptor().stream(),
        &values,
        context,
    )
}

#[derive(Debug)]
pub struct OuterGradients {
    pub input: Tensor,
    pub other: Tensor,
}

pub fn outer_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<OuterGradients, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_matching_stream_dtype(input, other, OUTER_OPERATION_ID)?;
    require_matching_stream_dtype(input, output_gradient, OUTER_OPERATION_ID)?;
    let left_count = element_count(input.descriptor().shape())?;
    let right_count = element_count(other.descriptor().shape())?;
    if input.descriptor().shape().len() != 1
        || other.descriptor().shape().len() != 1
        || output_gradient.descriptor().shape()
            != [
                u64::try_from(left_count)
                    .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("outer left"))?,
                u64::try_from(right_count)
                    .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("outer right"))?,
            ]
    {
        return Err(ElementwiseRuntimePartSevenError::Invalid(
            "outer VJP shapes are invalid",
        ));
    }
    let mut left_gradient = zeroed_workspace(backend, context, left_count)?;
    let mut right_gradient = zeroed_workspace(backend, context, right_count)?;
    for left_index in 0..left_count {
        for right_index in 0..right_count {
            let linear = left_index
                .checked_mul(right_count)
                .and_then(|value| value.checked_add(right_index))
                .ok_or(ElementwiseRuntimePartSevenError::ShapeOverflow(
                    "outer VJP index",
                ))?;
            check_periodically(linear, context.cancellation)?;
            let left_u64 = u64::try_from(left_index)
                .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("outer left index"))?;
            let right_u64 = u64::try_from(right_index).map_err(|_| {
                ElementwiseRuntimePartSevenError::ShapeOverflow("outer right index")
            })?;
            let gradient = read_f32(output_gradient, &[left_u64, right_u64])?;
            left_gradient[left_index] += gradient * read_f32(other, &[right_u64])?;
            right_gradient[right_index] += gradient * read_f32(input, &[left_u64])?;
        }
    }
    Ok(OuterGradients {
        input: upload_f32_with_context(
            backend,
            input.descriptor().shape(),
            input.descriptor().stream(),
            &left_gradient,
            context,
        )?,
        other: upload_f32_with_context(
            backend,
            other.descriptor().shape(),
            other.descriptor().stream(),
            &right_gradient,
            context,
        )?,
    })
}

pub fn outer_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_matching_stream_dtype(input, other, OUTER_OPERATION_ID)?;
    require_matching_f32(input, input_tangent, OUTER_OPERATION_ID)?;
    require_matching_f32(other, other_tangent, OUTER_OPERATION_ID)?;
    let left_count = element_count(input.descriptor().shape())?;
    let right_count = element_count(other.descriptor().shape())?;
    let count = left_count.checked_mul(right_count).ok_or(
        ElementwiseRuntimePartSevenError::ShapeOverflow("outer JVP output"),
    )?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for left_index in 0..left_count {
        for right_index in 0..right_count {
            let linear = left_index
                .checked_mul(right_count)
                .and_then(|value| value.checked_add(right_index))
                .ok_or(ElementwiseRuntimePartSevenError::ShapeOverflow(
                    "outer JVP index",
                ))?;
            check_periodically(linear, context.cancellation)?;
            let left_u64 = u64::try_from(left_index)
                .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("outer left index"))?;
            let right_u64 = u64::try_from(right_index).map_err(|_| {
                ElementwiseRuntimePartSevenError::ShapeOverflow("outer right index")
            })?;
            values.try_push(
                read_f32(input_tangent, &[left_u64])? * read_f32(other, &[right_u64])?
                    + read_f32(input, &[left_u64])? * read_f32(other_tangent, &[right_u64])?,
            )?;
        }
    }
    upload_f32_with_context(
        backend,
        &[
            u64::try_from(left_count)
                .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("outer left"))?,
            u64::try_from(right_count)
                .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("outer right"))?,
        ],
        input.descriptor().stream(),
        &values,
        context,
    )
}

pub fn rsqrt_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    require_f32_cpu(input, RSQRT_OPERATION_ID)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .unary(
            UnaryOperation::ReciprocalSquareRoot,
            input,
            descriptor,
            context,
        )?
        .0)
}

pub fn rsqrt_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    map_same_shape_binary_with_context(
        backend,
        input,
        output_gradient,
        RSQRT_OPERATION_ID,
        context,
        |value, gradient| -0.5 * gradient / (value * value.sqrt()),
    )
}

pub fn rsqrt_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    rsqrt_vjp_with_context_exact_native(backend, input, input_tangent, context)
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorPrintOptions {
    pub precision: usize,
    pub threshold: usize,
    pub edge_items: usize,
    pub line_width: usize,
    pub scientific_mode: Option<bool>,
}

impl Default for TensorPrintOptions {
    fn default() -> Self {
        Self {
            precision: 4,
            threshold: 1_000,
            edge_items: 3,
            line_width: 80,
            scientific_mode: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorPrintOptionsUpdate {
    pub precision: Option<usize>,
    pub threshold: Option<usize>,
    pub edge_items: Option<usize>,
    pub line_width: Option<usize>,
    pub scientific_mode: Option<Option<bool>>,
}

pub fn set_printoptions_exact_native(
    current: &TensorPrintOptions,
    update: Option<TensorPrintOptionsUpdate>,
    cancellation: &CancellationToken,
) -> Result<TensorPrintOptions, ElementwiseRuntimePartSevenError> {
    cancellation.check()?;
    let Some(update) = update else {
        return Ok(TensorPrintOptions::default());
    };
    let next = TensorPrintOptions {
        precision: update.precision.unwrap_or(current.precision),
        threshold: update.threshold.unwrap_or(current.threshold),
        edge_items: update.edge_items.unwrap_or(current.edge_items),
        line_width: update.line_width.unwrap_or(current.line_width),
        scientific_mode: update.scientific_mode.unwrap_or(current.scientific_mode),
    };
    if next.precision > 64
        || next.threshold > 10_000_000
        || next.edge_items > 1_000_000
        || next.line_width == 0
        || next.line_width > 16_384
    {
        return Err(ElementwiseRuntimePartSevenError::Invalid(
            "tensor print options exceed native bounds",
        ));
    }
    cancellation.check()?;
    Ok(next)
}

pub fn tanh_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    Ok(canonical_tanh_with_context(backend, input, context)?)
}

pub fn tanh_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    Ok(canonical_tanh_vjp_with_context(
        backend,
        input,
        output_gradient,
        context,
    )?)
}

pub fn tanh_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    context.cancellation.check()?;
    Ok(canonical_tanh_jvp_with_context(
        backend,
        input,
        input_tangent,
        context,
    )?)
}

pub fn xpu_current_stream_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    execution: &ExecutionContext<'_>,
) -> Result<StreamId, ElementwiseRuntimePartSevenError> {
    execution.cancellation.check()?;
    if device.kind() != DeviceKind::Xpu || capabilities.device() != device {
        return Err(ElementwiseRuntimePartSevenError::UnsupportedDevice {
            operation: XPU_CURRENT_STREAM_OPERATION_ID,
            device,
        });
    }
    execution.cancellation.check()?;
    Ok(execution.stream)
}

fn map_same_shape_binary_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    operation_id: &'static str,
    context: &ExecutionContext<'_>,
    operation: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    require_matching_f32(input, other, operation_id)?;
    let count = element_count(input.descriptor().shape())?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
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

fn map_broadcast_three_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    tensor1: &Tensor,
    tensor2: &Tensor,
    context: &ExecutionContext<'_>,
    operation: impl Fn(f32, f32, f32) -> f32,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    let shape = broadcast_shape_three(
        input.descriptor().shape(),
        tensor1.descriptor().shape(),
        tensor2.descriptor().shape(),
    )?;
    let count = element_count(&shape)?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &shape)?;
        values.try_push(operation(
            read_f32(
                input,
                &broadcast_indices(&indices, input.descriptor().shape())?,
            )?,
            read_f32(
                tensor1,
                &broadcast_indices(&indices, tensor1.descriptor().shape())?,
            )?,
            read_f32(
                tensor2,
                &broadcast_indices(&indices, tensor2.descriptor().shape())?,
            )?,
        ))?;
    }
    upload_f32_with_context(
        backend,
        &shape,
        input.descriptor().stream(),
        &values,
        context,
    )
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn zeroed_workspace(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
) -> Result<CpuWorkspaceVec<f32>, ElementwiseRuntimePartSevenError> {
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for _ in 0..count {
        values.try_push(0.0)?;
    }
    Ok(values)
}

fn upload_i64_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartSevenError> {
    if values.len() != element_count(shape)? {
        return Err(ElementwiseRuntimePartSevenError::Invalid(
            "i64 upload length does not match its shape",
        ));
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, stream)?;
    let (mut tensor, _) = backend.allocate(descriptor, context)?;
    let mut write = tensor.write()?;
    let bytes = write.bytes_mut()?;
    for (index, (chunk, value)) in bytes.chunks_exact_mut(8).zip(values).enumerate() {
        check_periodically(index, context.cancellation)?;
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
    drop(write);
    context.check()?;
    Ok(tensor)
}

fn require_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSevenError> {
    if input.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartSevenError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        })
    }
}

fn require_f32_cpu(
    input: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSevenError> {
    require_cpu(input, operation)?;
    if input.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartSevenError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        })
    }
}

fn require_matching_f32(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSevenError> {
    require_matching_stream_dtype(input, other, operation)?;
    if input.descriptor().shape() != other.descriptor().shape() {
        return Err(ElementwiseRuntimePartSevenError::Invalid(
            "tensor shapes must match",
        ));
    }
    Ok(())
}

fn require_matching_stream_dtype(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSevenError> {
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

fn require_f32_triplet(
    input: &Tensor,
    tensor1: &Tensor,
    tensor2: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartSevenError> {
    require_matching_stream_dtype(input, tensor1, operation)?;
    require_matching_stream_dtype(input, tensor2, operation)
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartSevenError> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension));
    usize::try_from(count.ok_or(ElementwiseRuntimePartSevenError::ShapeOverflow(
        "element count",
    ))?)
    .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("element count"))
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartSevenError> {
    let mut indices = vec![0; shape.len()];
    for (slot, dimension) in indices.iter_mut().zip(shape).rev() {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("tensor index"))?;
        if dimension == 0 {
            return Err(ElementwiseRuntimePartSevenError::Invalid(
                "cannot index an empty tensor",
            ));
        }
        *slot = u64::try_from(linear % dimension)
            .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("tensor index"))?;
        linear /= dimension;
    }
    Ok(indices)
}

fn linear_index(indices: &[u64], shape: &[u64]) -> Result<usize, ElementwiseRuntimePartSevenError> {
    let mut linear = 0_u64;
    for (index, dimension) in indices.iter().zip(shape) {
        linear = linear
            .checked_mul(*dimension)
            .and_then(|value| value.checked_add(*index))
            .ok_or(ElementwiseRuntimePartSevenError::ShapeOverflow(
                "linear index",
            ))?;
    }
    usize::try_from(linear)
        .map_err(|_| ElementwiseRuntimePartSevenError::ShapeOverflow("linear index"))
}

fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, ElementwiseRuntimePartSevenError> {
    let bytes: [u8; 4] = tensor
        .element_bytes(indices)?
        .try_into()
        .map_err(|_| ElementwiseRuntimePartSevenError::Invalid("f32 element width"))?;
    Ok(f32::from_ne_bytes(bytes))
}

fn broadcast_shape_three(
    first: &[u64],
    second: &[u64],
    third: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartSevenError> {
    let shape = binary_broadcast_shape(first, second)?;
    Ok(binary_broadcast_shape(&shape, third)?)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartSevenError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}
