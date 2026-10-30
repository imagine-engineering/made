use crate::{
    CpuBackend, CpuWorkspaceVec, DType, DeviceId, ExecutionContext, Tensor, TensorDescriptor,
    TensorError,
    generated_activation_normalization_functional_01::{
        FunctionalError, group_norm_with_context_exact_native,
        layer_norm_with_context_exact_native, silu_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, OperatorIndirectionError, convolution_with_context_exact_native,
        linear_with_context_exact_native,
    },
};
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NativeDiffusionTensorError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error("native diffusion tensor `{name}` requires shape {expected:?}, got {actual:?}")]
    Shape {
        name: &'static str,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("native diffusion tensor arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("native diffusion tensor parameter is invalid: {0}")]
    Invalid(&'static str),
}

pub fn tensor_from_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
    Ok(tensor)
}

pub fn tensor_to_f32(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, NativeDiffusionTensorError> {
    context.check()?;
    let descriptor = tensor.descriptor();
    if descriptor.dtype() != DType::F32 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: descriptor.dtype(),
        }
        .into());
    }
    if descriptor.device() != DeviceId::CPU {
        return Err(TensorError::NonHostDevice {
            device: descriptor.device(),
        }
        .into());
    }
    let bytes = tensor.contiguous_bytes()?;
    let width = std::mem::size_of::<f32>();
    if !bytes.len().is_multiple_of(width) {
        return Err(NativeDiffusionTensorError::Invalid(
            "unaligned f32 tensor bytes",
        ));
    }
    let mut values = backend.workspace_vec(context, bytes.len() / width)?;
    for (index, chunk) in bytes.chunks_exact(width).enumerate() {
        check_periodically(index, context)?;
        let encoded: [u8; 4] = chunk
            .try_into()
            .map_err(|_| NativeDiffusionTensorError::Invalid("unaligned f32 tensor bytes"))?;
        values.try_push(f32::from_ne_bytes(encoded))?;
    }
    Ok(values)
}

pub fn add(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    require_same_shape("add", left, right)?;
    let left_values = tensor_to_f32(backend, left, context)?;
    let right_values = tensor_to_f32(backend, right, context)?;
    let mut output = backend.workspace_vec(context, left_values.len())?;
    for (index, (left, right)) in left_values.iter().zip(right_values.iter()).enumerate() {
        check_periodically(index, context)?;
        output.try_push(left + right)?;
    }
    tensor_from_f32(backend, left.descriptor().shape(), &output, context)
}

pub fn add_channel_bias(
    backend: &CpuBackend,
    input: &Tensor,
    bias: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let shape = require_nchw("channel bias input", input)?;
    if bias.descriptor().shape() != [shape[1]] {
        return Err(NativeDiffusionTensorError::Shape {
            name: "channel bias",
            expected: vec![shape[1]],
            actual: bias.descriptor().shape().to_vec(),
        });
    }
    let channels = to_usize(shape[1], "channel bias channels")?;
    let spatial = to_usize(
        shape[2]
            .checked_mul(shape[3])
            .ok_or(NativeDiffusionTensorError::Overflow("channel bias spatial"))?,
        "channel bias spatial",
    )?;
    let values = tensor_to_f32(backend, input, context)?;
    let biases = tensor_to_f32(backend, bias, context)?;
    let mut output = values;
    for (index, value) in output.iter_mut().enumerate() {
        check_periodically(index, context)?;
        let channel = index / spatial % channels;
        *value += biases[channel];
    }
    tensor_from_f32(backend, &shape, &output, context)
}

pub fn linear(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let input_values = tensor_to_f32(backend, input, context)?;
    let weight_values = tensor_to_f32(backend, weight, context)?;
    let bias_values = bias
        .map(|bias| tensor_to_f32(backend, bias, context))
        .transpose()?;
    let input_shape = shape_to_usize(input.descriptor().shape(), "linear input shape")?;
    let weight_shape = shape_to_usize(weight.descriptor().shape(), "linear weight shape")?;
    let result = linear_with_context_exact_native(
        &input_values,
        &input_shape,
        &weight_values,
        &weight_shape,
        bias_values.as_deref(),
        DeviceId::CPU,
        context,
    )?;
    let output_shape = result
        .shape
        .iter()
        .map(|value| {
            u64::try_from(*value)
                .map_err(|_| NativeDiffusionTensorError::Overflow("linear output shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    tensor_from_f32(backend, &output_shape, &result.values, context)
}

pub fn layer_norm(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let width = *input
        .descriptor()
        .shape()
        .last()
        .ok_or(NativeDiffusionTensorError::Invalid("layer norm input rank"))?;
    for (name, parameter) in [("layer norm weight", weight), ("layer norm bias", bias)] {
        if parameter.descriptor().shape() != [width] {
            return Err(NativeDiffusionTensorError::Shape {
                name,
                expected: vec![width],
                actual: parameter.descriptor().shape().to_vec(),
            });
        }
    }
    let values = tensor_to_f32(backend, input, context)?;
    let weights = tensor_to_f32(backend, weight, context)?;
    let biases = tensor_to_f32(backend, bias, context)?;
    let shape = shape_to_usize(input.descriptor().shape(), "layer norm shape")?;
    let width = to_usize(width, "layer norm width")?;
    let output = layer_norm_with_context_exact_native(
        backend,
        &values,
        &shape,
        &[width],
        Some(&weights),
        Some(&biases),
        epsilon,
        DeviceId::CPU,
        context,
    )?;
    tensor_from_f32(backend, input.descriptor().shape(), &output, context)
}

pub fn group_norm(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    groups: usize,
    epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let shape = require_nchw("group norm", input)?;
    if weight.descriptor().shape() != [shape[1]] || bias.descriptor().shape() != [shape[1]] {
        return Err(NativeDiffusionTensorError::Shape {
            name: "group norm parameters",
            expected: vec![shape[1]],
            actual: weight.descriptor().shape().to_vec(),
        });
    }
    let values = tensor_to_f32(backend, input, context)?;
    let weights = tensor_to_f32(backend, weight, context)?;
    let biases = tensor_to_f32(backend, bias, context)?;
    let functional_shape = shape_to_usize(&shape, "group norm shape")?;
    let output = group_norm_with_context_exact_native(
        backend,
        &values,
        &functional_shape,
        groups,
        Some(&weights),
        Some(&biases),
        epsilon,
        DeviceId::CPU,
        context,
    )?;
    tensor_from_f32(backend, &shape, &output, context)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let input_values = tensor_to_f32(backend, input, context)?;
    let weight_values = tensor_to_f32(backend, weight, context)?;
    let bias_values = bias
        .map(|bias| tensor_to_f32(backend, bias, context))
        .transpose()?;
    let input_shape = shape_to_usize(input.descriptor().shape(), "convolution input shape")?;
    let weight_shape = shape_to_usize(weight.descriptor().shape(), "convolution weight shape")?;
    let geometry = ConvolutionGeometry::new(
        2,
        vec![stride; 2],
        vec![padding; 2],
        vec![1; 2],
        1,
        false,
        vec![0; 2],
    )?;
    let result = convolution_with_context_exact_native(
        &input_values,
        &input_shape,
        &weight_values,
        &weight_shape,
        bias_values.as_deref(),
        &geometry,
        DeviceId::CPU,
        context,
    )?;
    let output_shape = result
        .shape
        .iter()
        .map(|value| {
            u64::try_from(*value)
                .map_err(|_| NativeDiffusionTensorError::Overflow("convolution output shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    tensor_from_f32(backend, &output_shape, &result.values, context)
}

pub fn silu(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let values = tensor_to_f32(backend, input, context)?;
    let output = silu_with_context_exact_native(backend, &values, DeviceId::CPU, context)?;
    tensor_from_f32(backend, input.descriptor().shape(), &output, context)
}

pub fn quick_gelu(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    map_values(backend, input, context, |value| {
        value / (1.0 + (-1.702 * value).exp())
    })
}

pub fn nearest_upsample_2x(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let shape = require_nchw("nearest upsample", input)?;
    let batch = to_usize(shape[0], "upsample batch")?;
    let channels = to_usize(shape[1], "upsample channels")?;
    let height = to_usize(shape[2], "upsample height")?;
    let width = to_usize(shape[3], "upsample width")?;
    let output_height = height
        .checked_mul(2)
        .ok_or(NativeDiffusionTensorError::Overflow("upsample height"))?;
    let output_width = width
        .checked_mul(2)
        .ok_or(NativeDiffusionTensorError::Overflow("upsample width"))?;
    let values = tensor_to_f32(backend, input, context)?;
    let output_count = checked_product(
        &[batch, channels, output_height, output_width],
        "upsample output",
    )?;
    let mut output = workspace_zeros(backend, context, output_count)?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            for output_y in 0..output_height {
                check_periodically(output_y, context)?;
                for output_x in 0..output_width {
                    let source = flat4(
                        [batch_index, channel, output_y / 2, output_x / 2],
                        [batch, channels, height, width],
                    )?;
                    let destination = flat4(
                        [batch_index, channel, output_y, output_x],
                        [batch, channels, output_height, output_width],
                    )?;
                    output[destination] = values[source];
                }
            }
        }
    }
    tensor_from_f32(
        backend,
        &[
            shape[0],
            shape[1],
            shape[2]
                .checked_mul(2)
                .ok_or(NativeDiffusionTensorError::Overflow("upsample shape"))?,
            shape[3]
                .checked_mul(2)
                .ok_or(NativeDiffusionTensorError::Overflow("upsample shape"))?,
        ],
        &output,
        context,
    )
}

pub fn concat_channels(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let left_shape = require_nchw("concat left", left)?;
    let right_shape = require_nchw("concat right", right)?;
    if left_shape[0] != right_shape[0]
        || left_shape[2] != right_shape[2]
        || left_shape[3] != right_shape[3]
    {
        return Err(NativeDiffusionTensorError::Shape {
            name: "concat right",
            expected: vec![left_shape[0], right_shape[1], left_shape[2], left_shape[3]],
            actual: right_shape.to_vec(),
        });
    }
    let batch = to_usize(left_shape[0], "concat batch")?;
    let left_channels = to_usize(left_shape[1], "concat left channels")?;
    let right_channels = to_usize(right_shape[1], "concat right channels")?;
    let spatial = to_usize(
        left_shape[2]
            .checked_mul(left_shape[3])
            .ok_or(NativeDiffusionTensorError::Overflow("concat spatial"))?,
        "concat spatial",
    )?;
    let left_values = tensor_to_f32(backend, left, context)?;
    let right_values = tensor_to_f32(backend, right, context)?;
    let output_channels = left_channels
        .checked_add(right_channels)
        .ok_or(NativeDiffusionTensorError::Overflow("concat channels"))?;
    let output_count = checked_product(&[batch, output_channels, spatial], "concat output")?;
    let mut output = workspace_zeros(backend, context, output_count)?;
    for batch_index in 0..batch {
        context.check()?;
        let output_start = batch_index * output_channels * spatial;
        let left_start = batch_index * left_channels * spatial;
        let right_start = batch_index * right_channels * spatial;
        let output_left = output
            .get_mut(output_start..output_start + left_channels * spatial)
            .ok_or(NativeDiffusionTensorError::Overflow(
                "concat left destination",
            ))?;
        output_left.copy_from_slice(
            left_values
                .get(left_start..left_start + left_channels * spatial)
                .ok_or(NativeDiffusionTensorError::Overflow("concat left source"))?,
        );
        let output_right = output
            .get_mut(
                output_start + left_channels * spatial..output_start + output_channels * spatial,
            )
            .ok_or(NativeDiffusionTensorError::Overflow(
                "concat right destination",
            ))?;
        output_right.copy_from_slice(
            right_values
                .get(right_start..right_start + right_channels * spatial)
                .ok_or(NativeDiffusionTensorError::Overflow("concat right source"))?,
        );
    }
    tensor_from_f32(
        backend,
        &[
            left_shape[0],
            u64::try_from(output_channels)
                .map_err(|_| NativeDiffusionTensorError::Overflow("concat output channels"))?,
            left_shape[2],
            left_shape[3],
        ],
        &output,
        context,
    )
}

fn map_values(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
    operation: impl Fn(f32) -> f32,
) -> Result<Tensor, NativeDiffusionTensorError> {
    let values = tensor_to_f32(backend, input, context)?;
    let mut output = backend.workspace_vec(context, values.len())?;
    for (index, value) in values.iter().copied().enumerate() {
        check_periodically(index, context)?;
        let value = operation(value);
        if !value.is_finite() {
            return Err(TensorError::InvalidNumeric {
                reason: "native diffusion elementwise operation produced non-finite output"
                    .to_owned(),
            }
            .into());
        }
        output.try_push(value)?;
    }
    tensor_from_f32(backend, input.descriptor().shape(), &output, context)
}

fn require_nchw(
    name: &'static str,
    tensor: &Tensor,
) -> Result<[u64; 4], NativeDiffusionTensorError> {
    tensor
        .descriptor()
        .shape()
        .try_into()
        .map_err(|_| NativeDiffusionTensorError::Shape {
            name,
            expected: vec![0, 0, 0, 0],
            actual: tensor.descriptor().shape().to_vec(),
        })
}

fn require_same_shape(
    name: &'static str,
    left: &Tensor,
    right: &Tensor,
) -> Result<(), NativeDiffusionTensorError> {
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(NativeDiffusionTensorError::Shape {
            name,
            expected: left.descriptor().shape().to_vec(),
            actual: right.descriptor().shape().to_vec(),
        });
    }
    Ok(())
}

fn to_usize(value: u64, context: &'static str) -> Result<usize, NativeDiffusionTensorError> {
    usize::try_from(value).map_err(|_| NativeDiffusionTensorError::Overflow(context))
}

fn shape_to_usize(
    shape: &[u64],
    context: &'static str,
) -> Result<Vec<usize>, NativeDiffusionTensorError> {
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(shape.len())
        .map_err(|_| NativeDiffusionTensorError::Overflow(context))?;
    for dimension in shape {
        converted.push(to_usize(*dimension, context)?);
    }
    Ok(converted)
}

fn checked_product(
    values: &[usize],
    context: &'static str,
) -> Result<usize, NativeDiffusionTensorError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(NativeDiffusionTensorError::Overflow(context))
    })
}

fn workspace_zeros(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
) -> Result<CpuWorkspaceVec<f32>, NativeDiffusionTensorError> {
    let mut values = backend.workspace_vec(context, count)?;
    for index in 0..count {
        check_periodically(index, context)?;
        values.try_push(0.0)?;
    }
    Ok(values)
}

fn flat4(index: [usize; 4], shape: [usize; 4]) -> Result<usize, NativeDiffusionTensorError> {
    if index.iter().zip(shape).any(|(index, size)| *index >= size) {
        return Err(NativeDiffusionTensorError::Overflow(
            "four-dimensional index",
        ));
    }
    index[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_add(index[1]))
        .and_then(|value| value.checked_mul(shape[2]))
        .and_then(|value| value.checked_add(index[2]))
        .and_then(|value| value.checked_mul(shape[3]))
        .and_then(|value| value.checked_add(index[3]))
        .ok_or(NativeDiffusionTensorError::Overflow(
            "four-dimensional index",
        ))
}

fn check_periodically(
    index: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeDiffusionTensorError> {
    if index.is_multiple_of(256) {
        context.check()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancellationToken, CpuWorkspaceAuthority, StreamId};

    #[test]
    fn linear_and_convolution_use_canonical_cpu_tensors() -> Result<(), Box<dyn std::error::Error>>
    {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let authorization = authority.authorize_workspace(1024)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
        let input = tensor_from_f32(&backend, &[1, 2], &[2.0, 3.0], &context)?;
        let weight = tensor_from_f32(&backend, &[1, 2], &[4.0, 5.0], &context)?;
        let bias = tensor_from_f32(&backend, &[1], &[1.0], &context)?;
        let output = linear(&backend, &input, &weight, Some(&bias), &context)?;
        let values = tensor_to_f32(&backend, &output, &context)?;
        assert_eq!(&*values, &[24.0]);
        drop(values);

        let image = tensor_from_f32(&backend, &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let kernel = tensor_from_f32(&backend, &[1, 1, 1, 1], &[2.0], &context)?;
        let output = conv2d(&backend, &image, &kernel, None, 1, 0, &context)?;
        let values = tensor_to_f32(&backend, &output, &context)?;
        assert_eq!(&*values, &[2.0, 4.0, 6.0, 8.0]);
        drop(values);
        assert_eq!(authorization.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn workspace_accounting_is_exact_and_converges_after_native_operation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let cancellation = CancellationToken::default();
        let authorization = authority.authorize_workspace(1024)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
        let left = tensor_from_f32(&backend, &[2], &[1.0, 2.0], &context)?;
        let right = tensor_from_f32(&backend, &[2], &[3.0, 4.0], &context)?;
        let output = add(&backend, &left, &right, &context)?;

        assert_eq!(authorization.peak_bytes(), 24);
        assert_eq!(authorization.in_use_bytes(), 0);
        let values = tensor_to_f32(&backend, &output, &context)?;
        assert_eq!(&*values, &[4.0, 6.0]);
        assert_eq!(authorization.in_use_bytes(), 8);
        drop(values);
        assert_eq!(authorization.in_use_bytes(), 0);
        drop(output);
        drop(right);
        drop(left);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn underauthorization_cancellation_fault_backend_mismatch_and_oom_are_typed()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let live_cancellation = CancellationToken::default();
        let upload_authorization = authority.authorize_workspace(0)?;
        let upload_context =
            backend.execution_context(StreamId::DEFAULT, upload_authorization, &live_cancellation);
        let tensor = tensor_from_f32(&backend, &[1], &[1.0], &upload_context)?;

        let insufficient = authority.authorize_workspace(3)?;
        let insufficient_context =
            backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &live_cancellation);
        assert!(matches!(
            tensor_to_f32(&backend, &tensor, &insufficient_context),
            Err(NativeDiffusionTensorError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded {
                    requested: 4,
                    authorized: 3,
                    in_use: 0,
                }
            ))
        ));
        assert_eq!(insufficient.in_use_bytes(), 0);

        let fault_authorization = authority.authorize_workspace(8)?;
        let fault_context = backend.execution_context(
            StreamId::DEFAULT,
            fault_authorization.clone(),
            &live_cancellation,
        );
        let non_finite = tensor_from_f32(&backend, &[1], &[f32::INFINITY], &fault_context)?;
        assert!(matches!(
            quick_gelu(&backend, &non_finite, &fault_context),
            Err(NativeDiffusionTensorError::Tensor(
                TensorError::InvalidNumeric { .. }
            ))
        ));
        assert_eq!(fault_authorization.peak_bytes(), 8);
        assert_eq!(fault_authorization.in_use_bytes(), 0);
        drop(non_finite);

        let (other_backend, other_authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let other_authorization = other_authority.authorize_workspace(4)?;
        let mismatched_context = other_backend.execution_context(
            StreamId::DEFAULT,
            other_authorization,
            &live_cancellation,
        );
        assert!(matches!(
            tensor_to_f32(&backend, &tensor, &mismatched_context),
            Err(NativeDiffusionTensorError::Tensor(
                TensorError::WorkspaceAuthorizationMismatch { .. }
            ))
        ));

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let authorization = authority.authorize_workspace(4)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
        assert!(matches!(
            tensor_to_f32(&backend, &tensor, &context),
            Err(NativeDiffusionTensorError::Tensor(TensorError::Cancelled))
        ));
        assert_eq!(authorization.in_use_bytes(), 0);

        let cancellation = CancellationToken::default();
        let (constrained_backend, constrained_authority) =
            CpuWorkspaceAuthority::create_backend(16)?;
        let authorization = constrained_authority.authorize_workspace(0)?;
        let context =
            constrained_backend.execution_context(StreamId::DEFAULT, authorization, &cancellation);
        assert!(matches!(
            tensor_from_f32(&constrained_backend, &[8], &[0.0; 8], &context),
            Err(NativeDiffusionTensorError::Tensor(
                TensorError::AllocationFailed { .. }
            ))
        ));
        drop(tensor);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }
}
