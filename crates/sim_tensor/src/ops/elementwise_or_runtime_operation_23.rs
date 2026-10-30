use crate::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, StreamId, Tensor,
    TensorBackend, TensorDescriptor, TensorError,
};
use thiserror::Error;

pub const CARTESIAN_PROD_OPERATION_ID: &str = "COMFY-TENSOR-OP-FEFD7C671451";
pub const RMSPROP_OPERATION_ID: &str = "COMFY-TENSOR-OP-FCDA841034ED";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartTwentyThreeError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("elementwise/runtime part-twenty-three execution was cancelled")]
    Cancelled,
    #[error("operation {operation} does not support device {device:?}")]
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
        reason: &'static str,
    },
    #[error("operation {operation} overflowed while computing {subject}")]
    ShapeOverflow {
        operation: &'static str,
        subject: &'static str,
    },
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartTwentyThreeError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

pub fn cartesian_prod_with_context_exact_native(
    backend: &CpuBackend,
    inputs: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyThreeError> {
    context.cancellation.check()?;
    let first = inputs
        .first()
        .ok_or(ElementwiseRuntimePartTwentyThreeError::Invalid {
            operation: CARTESIAN_PROD_OPERATION_ID,
            reason: "cartesian_prod expects at least one tensor",
        })?;
    require_cpu(first, CARTESIAN_PROD_OPERATION_ID)?;
    if first.descriptor().rank() != 1 {
        return Err(ElementwiseRuntimePartTwentyThreeError::Invalid {
            operation: CARTESIAN_PROD_OPERATION_ID,
            reason: "cartesian_prod expects one-dimensional tensors",
        });
    }
    let dtype = first.descriptor().dtype();
    let stream = first.descriptor().stream();
    let mut row_count = 1_u64;
    for (index, input) in inputs.iter().enumerate() {
        check_periodically(index, context.cancellation)?;
        require_cpu(input, CARTESIAN_PROD_OPERATION_ID)?;
        if input.descriptor().rank() != 1 {
            return Err(ElementwiseRuntimePartTwentyThreeError::Invalid {
                operation: CARTESIAN_PROD_OPERATION_ID,
                reason: "cartesian_prod expects one-dimensional tensors",
            });
        }
        if input.descriptor().dtype() != dtype {
            return Err(ElementwiseRuntimePartTwentyThreeError::Invalid {
                operation: CARTESIAN_PROD_OPERATION_ID,
                reason: "cartesian_prod tensors must have the same dtype",
            });
        }
        if input.descriptor().stream() != stream {
            return Err(TensorError::StreamMismatch {
                expected: stream,
                actual: input.descriptor().stream(),
            }
            .into());
        }
        row_count = row_count.checked_mul(input.descriptor().shape()[0]).ok_or(
            ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
                operation: CARTESIAN_PROD_OPERATION_ID,
                subject: "cartesian product rows",
            },
        )?;
    }
    let column_count = u64::try_from(inputs.len()).map_err(|_| {
        ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
            operation: CARTESIAN_PROD_OPERATION_ID,
            subject: "cartesian product columns",
        }
    })?;
    let byte_count = row_count
        .checked_mul(column_count)
        .and_then(|count| count.checked_mul(dtype.byte_width()))
        .ok_or(ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
            operation: CARTESIAN_PROD_OPERATION_ID,
            subject: "cartesian product bytes",
        })?;
    let byte_count = usize::try_from(byte_count).map_err(|_| {
        ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
            operation: CARTESIAN_PROD_OPERATION_ID,
            subject: "cartesian product host allocation",
        }
    })?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    let mut suffix_products = vec![1_u64; inputs.len()];
    let mut suffix = 1_u64;
    for index in (0..inputs.len()).rev() {
        suffix_products[index] = suffix;
        suffix = suffix
            .checked_mul(inputs[index].descriptor().shape()[0])
            .ok_or(ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
                operation: CARTESIAN_PROD_OPERATION_ID,
                subject: "cartesian product indexing",
            })?;
    }
    for row in 0..row_count {
        check_periodically_u64(row, context.cancellation)?;
        for (input, suffix_product) in inputs.iter().zip(&suffix_products) {
            let length = input.descriptor().shape()[0];
            let input_index = (row / suffix_product) % length;
            for byte in input.element_bytes(&[input_index])? {
                bytes.try_push(*byte)?;
            }
        }
    }
    let shape = if inputs.len() == 1 {
        vec![row_count]
    } else {
        vec![row_count, column_count]
    };
    upload_bytes_with_context(backend, &shape, dtype, stream, &bytes, context)
}

#[derive(Clone, Debug)]
pub struct NativeRmsprop {
    learning_rate: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    maximize: bool,
    steps: Vec<u64>,
    square_averages: Vec<Tensor>,
    momentum_buffers: Vec<Tensor>,
    gradient_averages: Vec<Tensor>,
}

impl NativeRmsprop {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_context_exact_native(
        backend: &CpuBackend,
        parameters: &[Tensor],
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
        weight_decay: f32,
        momentum: f32,
        centered: bool,
        maximize: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ElementwiseRuntimePartTwentyThreeError> {
        context.cancellation.check()?;
        if parameters.is_empty()
            || !learning_rate.is_finite()
            || learning_rate < 0.0
            || !alpha.is_finite()
            || alpha < 0.0
            || !epsilon.is_finite()
            || epsilon < 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
            || !momentum.is_finite()
            || momentum < 0.0
        {
            return Err(ElementwiseRuntimePartTwentyThreeError::Invalid {
                operation: RMSPROP_OPERATION_ID,
                reason: "invalid native RMSprop configuration",
            });
        }
        let mut square_averages = reserved_tensor_state(parameters.len(), "square averages")?;
        let mut momentum_buffers = if momentum > 0.0 {
            reserved_tensor_state(parameters.len(), "momentum buffers")?
        } else {
            Vec::new()
        };
        let mut gradient_averages = if centered {
            reserved_tensor_state(parameters.len(), "gradient averages")?
        } else {
            Vec::new()
        };
        for (index, parameter) in parameters.iter().enumerate() {
            check_periodically(index, context.cancellation)?;
            require_f32_cpu(parameter, RMSPROP_OPERATION_ID)?;
            square_averages.push(zero_like_with_context(backend, parameter, context)?);
            if momentum > 0.0 {
                momentum_buffers.push(zero_like_with_context(backend, parameter, context)?);
            }
            if centered {
                gradient_averages.push(zero_like_with_context(backend, parameter, context)?);
            }
        }
        Ok(Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay,
            momentum,
            centered,
            maximize,
            steps: vec![0; parameters.len()],
            square_averages,
            momentum_buffers,
            gradient_averages,
        })
    }
    pub fn step_with_context_exact_native(
        &mut self,
        backend: &CpuBackend,
        parameters: &mut [Tensor],
        gradients: &[Tensor],
        context: &ExecutionContext<'_>,
    ) -> Result<(), ElementwiseRuntimePartTwentyThreeError> {
        context.cancellation.check()?;
        if parameters.len() != gradients.len()
            || parameters.len() != self.steps.len()
            || parameters.len() != self.square_averages.len()
            || (self.momentum > 0.0 && parameters.len() != self.momentum_buffers.len())
            || (self.momentum == 0.0 && !self.momentum_buffers.is_empty())
            || (self.centered && parameters.len() != self.gradient_averages.len())
            || (!self.centered && !self.gradient_averages.is_empty())
        {
            return Err(ElementwiseRuntimePartTwentyThreeError::Invalid {
                operation: RMSPROP_OPERATION_ID,
                reason: "RMSprop parameter, gradient, and state lengths must match",
            });
        }
        let mut next_steps = backend.workspace_vec(context, self.steps.len())?;
        for step in &self.steps {
            next_steps.try_push(step.checked_add(1).ok_or(
                ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
                    operation: RMSPROP_OPERATION_ID,
                    subject: "RMSprop step",
                },
            )?)?;
        }
        let mut staged_parameters = reserved_tensor_state(parameters.len(), "parameters")?;
        let mut staged_square_averages =
            reserved_tensor_state(parameters.len(), "square averages")?;
        let mut staged_momentum_buffers = if self.momentum > 0.0 {
            reserved_tensor_state(parameters.len(), "momentum buffers")?
        } else {
            Vec::new()
        };
        let mut staged_gradient_averages = if self.centered {
            reserved_tensor_state(parameters.len(), "gradient averages")?
        } else {
            Vec::new()
        };

        for index in 0..parameters.len() {
            context.cancellation.check()?;
            require_matching_f32(&parameters[index], &gradients[index], RMSPROP_OPERATION_ID)?;
            require_matching_f32(
                &parameters[index],
                &self.square_averages[index],
                RMSPROP_OPERATION_ID,
            )?;
            if self.momentum > 0.0 {
                require_matching_f32(
                    &parameters[index],
                    &self.momentum_buffers[index],
                    RMSPROP_OPERATION_ID,
                )?;
            }
            if self.centered {
                require_matching_f32(
                    &parameters[index],
                    &self.gradient_averages[index],
                    RMSPROP_OPERATION_ID,
                )?;
            }
            let parameter_values = tensor_f32_with_context(backend, &parameters[index], context)?;
            let gradient_values = tensor_f32_with_context(backend, &gradients[index], context)?;
            let square_average_values =
                tensor_f32_with_context(backend, &self.square_averages[index], context)?;
            let momentum_values = if self.momentum > 0.0 {
                Some(tensor_f32_with_context(
                    backend,
                    &self.momentum_buffers[index],
                    context,
                )?)
            } else {
                None
            };
            let gradient_average_values = if self.centered {
                Some(tensor_f32_with_context(
                    backend,
                    &self.gradient_averages[index],
                    context,
                )?)
            } else {
                None
            };
            let count = parameter_values.len();
            let mut next_parameter_values = backend.workspace_vec(context, count)?;
            let mut next_square_average_values = backend.workspace_vec(context, count)?;
            let mut next_momentum_values = if self.momentum > 0.0 {
                Some(backend.workspace_vec(context, count)?)
            } else {
                None
            };
            let mut next_gradient_average_values = if self.centered {
                Some(backend.workspace_vec(context, count)?)
            } else {
                None
            };
            for element in 0..count {
                check_periodically(element, context.cancellation)?;
                let parameter = parameter_values[element];
                let mut gradient = if self.maximize {
                    -gradient_values[element]
                } else {
                    gradient_values[element]
                };
                gradient += self.weight_decay * parameter;
                let square_average = self.alpha * square_average_values[element]
                    + (1.0 - self.alpha) * gradient * gradient;
                next_square_average_values.try_push(square_average)?;
                let denominator_square = match (
                    gradient_average_values.as_ref(),
                    next_gradient_average_values.as_mut(),
                ) {
                    (Some(previous), Some(next)) => {
                        let average =
                            self.alpha * previous[element] + (1.0 - self.alpha) * gradient;
                        next.try_push(average)?;
                        square_average - average * average
                    }
                    _ => square_average,
                };
                let normalized_gradient = gradient / (denominator_square.sqrt() + self.epsilon);
                let update = match (momentum_values.as_ref(), next_momentum_values.as_mut()) {
                    (Some(previous), Some(next)) => {
                        let momentum = self.momentum * previous[element] + normalized_gradient;
                        next.try_push(momentum)?;
                        momentum
                    }
                    _ => normalized_gradient,
                };
                next_parameter_values.try_push(parameter - self.learning_rate * update)?;
            }
            let shape = parameters[index].descriptor().shape();
            let stream = parameters[index].descriptor().stream();
            staged_parameters.push(upload_f32_with_context(
                backend,
                shape,
                stream,
                &next_parameter_values,
                context,
            )?);
            staged_square_averages.push(upload_f32_with_context(
                backend,
                shape,
                stream,
                &next_square_average_values,
                context,
            )?);
            if let Some(values) = next_momentum_values {
                staged_momentum_buffers.push(upload_f32_with_context(
                    backend, shape, stream, &values, context,
                )?);
            }
            if let Some(values) = next_gradient_average_values {
                staged_gradient_averages.push(upload_f32_with_context(
                    backend, shape, stream, &values, context,
                )?);
            }
        }
        context.cancellation.check()?;
        for (parameter, staged) in parameters.iter_mut().zip(staged_parameters) {
            parameter.commit_in_place(staged)?;
        }
        for (square_average, staged) in self
            .square_averages
            .iter_mut()
            .zip(staged_square_averages)
        {
            square_average.commit_in_place(staged)?;
        }
        for (momentum_buffer, staged) in self
            .momentum_buffers
            .iter_mut()
            .zip(staged_momentum_buffers)
        {
            momentum_buffer.commit_in_place(staged)?;
        }
        for (gradient_average, staged) in self
            .gradient_averages
            .iter_mut()
            .zip(staged_gradient_averages)
        {
            gradient_average.commit_in_place(staged)?;
        }
        self.steps = next_steps.iter().copied().collect();
        Ok(())
    }

    pub fn steps(&self) -> &[u64] {
        &self.steps
    }

    pub fn square_averages(&self) -> &[Tensor] {
        &self.square_averages
    }

    pub fn momentum_buffers(&self) -> &[Tensor] {
        &self.momentum_buffers
    }

    pub fn gradient_averages(&self) -> &[Tensor] {
        &self.gradient_averages
    }
}

fn reserved_tensor_state(
    length: usize,
    subject: &'static str,
) -> Result<Vec<Tensor>, ElementwiseRuntimePartTwentyThreeError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| {
        ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
            operation: RMSPROP_OPERATION_ID,
            subject,
        }
    })?;
    Ok(values)
}

fn zero_like_with_context(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyThreeError> {
    context.cancellation.check()?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .fill(crate::Scalar::Float(0.0), descriptor, context)?
        .0)
}

fn tensor_f32_with_context(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<crate::CpuWorkspaceVec<f32>, ElementwiseRuntimePartTwentyThreeError> {
    require_f32_cpu(tensor, RMSPROP_OPERATION_ID)?;
    let count = usize::try_from(tensor.descriptor().element_count()?).map_err(|_| {
        ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
            operation: RMSPROP_OPERATION_ID,
            subject: "RMSprop tensor decode",
        }
    })?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, tensor.descriptor().shape())?;
        let encoded: [u8; 4] = tensor.element_bytes(&indices)?.try_into().map_err(|_| {
            ElementwiseRuntimePartTwentyThreeError::Invalid {
                operation: RMSPROP_OPERATION_ID,
                reason: "F32 tensor element has an invalid byte width",
            }
        })?;
        values.try_push(f32::from_ne_bytes(encoded))?;
    }
    Ok(values)
}

fn upload_f32_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyThreeError> {
    context.cancellation.check()?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_bytes_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    stream: StreamId,
    bytes: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwentyThreeError> {
    context.cancellation.check()?;
    let descriptor = TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, stream)?;
    Ok(backend.upload_bytes(descriptor, bytes, context)?.0)
}

fn require_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyThreeError> {
    if tensor.descriptor().device() == DeviceId::CPU {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartTwentyThreeError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        })
    }
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyThreeError> {
    require_cpu(tensor, operation)?;
    if tensor.descriptor().dtype() == DType::F32 {
        Ok(())
    } else {
        Err(ElementwiseRuntimePartTwentyThreeError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        })
    }
}

fn require_matching_f32(
    parameter: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwentyThreeError> {
    require_f32_cpu(parameter, operation)?;
    require_f32_cpu(other, operation)?;
    if parameter.descriptor().shape() != other.descriptor().shape() {
        return Err(ElementwiseRuntimePartTwentyThreeError::Invalid {
            operation,
            reason: "RMSprop parameter, gradient, and state shapes must match",
        });
    }
    if parameter.descriptor().stream() != other.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: parameter.descriptor().stream(),
            actual: other.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn unravel_index(
    linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartTwentyThreeError> {
    let mut remaining = u64::try_from(linear).map_err(|_| {
        ElementwiseRuntimePartTwentyThreeError::ShapeOverflow {
            operation: RMSPROP_OPERATION_ID,
            subject: "logical tensor index",
        }
    })?;
    let mut indices = vec![0_u64; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dimension = shape[axis];
        if dimension == 0 {
            continue;
        }
        indices[axis] = remaining % dimension;
        remaining /= dimension;
    }
    Ok(indices)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartTwentyThreeError> {
    if index.is_multiple_of(1024) {
        cancellation.check()?;
    }
    Ok(())
}

fn check_periodically_u64(
    index: u64,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartTwentyThreeError> {
    if index.is_multiple_of(1024) {
        cancellation.check()?;
    }
    Ok(())
}
