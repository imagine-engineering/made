use std::{cmp::Ordering, f32::consts::TAU};

use crate::{
    AutocastPolicy, BackendCapabilityMatrix, CancellationToken, CpuBackend, CpuWorkspaceVec, DType,
    DecodedScalar, DeviceId, DeviceKind, ExecutionContext, NumericClass, StreamId, Tensor,
    TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_01::ElementwiseRuntimeError,
    generated_elementwise_or_runtime_operation_09::{
        BinaryGradients, ElementwiseRuntimePartNineError, pow_jvp_with_context_exact_native,
        pow_vjp_with_context_exact_native, pow_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_10::{
        ElementwiseRuntimePartTenError, cumsum_jvp_with_context_exact_native,
        cumsum_vjp_with_context_exact_native, cumsum_with_context_exact_native,
    },
};
use thiserror::Error;

pub const IS_CONTIGUOUS_OPERATION_ID: &str = "COMFY-TENSOR-OP-8A892AD7A3C2";
pub const NUMEL_OPERATION_ID: &str = "COMFY-TENSOR-OP-8AE4F174E7A1";
pub const POW_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-8E0C873ABBAD";
pub const ENABLE_MATH_SDP_OPERATION_ID: &str = "COMFY-TENSOR-OP-8C351B65C789";
pub const CUDA_CURRENT_STREAM_OPERATION_ID: &str = "COMFY-TENSOR-OP-861EE6173859";
pub const CUMSUM_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-88FE050115A9";
pub const AUTOCAST_GPU_DTYPE_OPERATION_ID: &str = "COMFY-TENSOR-OP-868F72C2BE67";
pub const IS_FLOATING_POINT_OPERATION_ID: &str = "COMFY-TENSOR-OP-8B5439D32B8F";
pub const FAN_IN_OUT_OPERATION_ID: &str = "COMFY-TENSOR-OP-8E5582D70F18";
pub const STFT_OPERATION_ID: &str = "COMFY-TENSOR-OP-8C29B75AEA2A";
pub const TOPK_OPERATION_ID: &str = "COMFY-TENSOR-OP-8DF974B2A77C";
pub const TRIL_OPERATION_ID: &str = "COMFY-TENSOR-OP-874C83BCB8C5";

#[derive(Debug, Error)]
pub enum ElementwiseRuntimePartTwelveError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    PartOne(#[from] ElementwiseRuntimeError),
    #[error(transparent)]
    PartNine(#[from] ElementwiseRuntimePartNineError),
    #[error(transparent)]
    PartTen(#[from] ElementwiseRuntimePartTenError),
    #[error("elementwise/runtime part-twelve operation was cancelled")]
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
    #[error("elementwise/runtime part-twelve input is invalid: {0}")]
    Invalid(String),
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
}

impl From<sim_types::CancellationError> for ElementwiseRuntimePartTwelveError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FanInAndFanOut {
    pub fan_in: u64,
    pub fan_out: u64,
}

#[derive(Debug)]
pub struct TopKResult {
    pub values: Tensor,
    pub indices: Tensor,
}

pub fn is_contiguous_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartTwelveError> {
    cancellation.check()?;
    Ok(input.descriptor().is_contiguous()?)
}

pub fn numel_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<u64, ElementwiseRuntimePartTwelveError> {
    cancellation.check()?;
    Ok(input.descriptor().element_count()?)
}

pub fn tensor_pow_with_context_exact_native(
    backend: &CpuBackend,
    base: &Tensor,
    exponent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    Ok(pow_with_context_exact_native(
        backend, base, exponent, context,
    )?)
}

pub fn tensor_pow_vjp_with_context_exact_native(
    backend: &CpuBackend,
    base: &Tensor,
    exponent: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<BinaryGradients, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    Ok(pow_vjp_with_context_exact_native(
        backend,
        base,
        exponent,
        output_gradient,
        context,
    )?)
}

pub fn tensor_pow_jvp_with_context_exact_native(
    backend: &CpuBackend,
    base: &Tensor,
    exponent: &Tensor,
    base_tangent: &Tensor,
    exponent_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    Ok(pow_jvp_with_context_exact_native(
        backend,
        base,
        exponent,
        base_tangent,
        exponent_tangent,
        context,
    )?)
}

pub fn cuda_current_stream_exact_native(
    capabilities: &BackendCapabilityMatrix,
    device: DeviceId,
    execution: &ExecutionContext<'_>,
) -> Result<StreamId, ElementwiseRuntimePartTwelveError> {
    execution.cancellation.check()?;
    if device.kind() != DeviceKind::Cuda || capabilities.device() != device {
        return Err(ElementwiseRuntimePartTwelveError::UnsupportedDevice {
            operation: CUDA_CURRENT_STREAM_OPERATION_ID,
            device,
        });
    }
    Ok(execution.stream)
}

pub fn cumsum_function_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: i64,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    Ok(cumsum_with_context_exact_native(
        backend, input, dimension, dtype, context,
    )?)
}

pub fn cumsum_function_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    Ok(cumsum_vjp_with_context_exact_native(
        backend,
        output_gradient,
        dimension,
        context,
    )?)
}

pub fn cumsum_function_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    Ok(cumsum_jvp_with_context_exact_native(
        backend,
        input_tangent,
        dimension,
        context,
    )?)
}

pub fn get_autocast_gpu_dtype_exact_native(
    policy: &AutocastPolicy,
    cancellation: &CancellationToken,
) -> Result<DType, ElementwiseRuntimePartTwelveError> {
    cancellation.check()?;
    Ok(policy.dtype())
}

pub fn is_floating_point_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<bool, ElementwiseRuntimePartTwelveError> {
    cancellation.check()?;
    Ok(input.descriptor().dtype().class() == NumericClass::FloatingPoint)
}

pub fn calculate_fan_in_and_fan_out_exact_native(
    input: &Tensor,
    cancellation: &CancellationToken,
) -> Result<FanInAndFanOut, ElementwiseRuntimePartTwelveError> {
    cancellation.check()?;
    let shape = input.descriptor().shape();
    if shape.len() < 2 {
        return Err(ElementwiseRuntimePartTwelveError::Invalid(
            "fan-in/fan-out requires a tensor with at least two dimensions".to_owned(),
        ));
    }
    let receptive_field = shape[2..]
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension));
    let receptive_field = receptive_field.ok_or(
        ElementwiseRuntimePartTwelveError::ShapeOverflow("fan receptive field"),
    )?;
    Ok(FanInAndFanOut {
        fan_in: shape[1]
            .checked_mul(receptive_field)
            .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow("fan in"))?,
        fan_out: shape[0]
            .checked_mul(receptive_field)
            .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow("fan out"))?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn stft_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Option<&Tensor>,
    center: bool,
    normalized: bool,
    onesided: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    let configuration = StftConfiguration::new(
        input, n_fft, hop_length, win_length, window, center, normalized, onesided,
    )?;
    let values = stft_values(backend, input, window, &configuration, context)?;
    upload_complex64(backend, &configuration.output_shape, &values, context)
}

#[allow(clippy::too_many_arguments)]
pub fn stft_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Option<&Tensor>,
    center: bool,
    normalized: bool,
    onesided: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    stft_with_context_exact_native(
        backend,
        input_tangent,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        normalized,
        onesided,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn stft_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Option<&Tensor>,
    center: bool,
    normalized: bool,
    onesided: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    let configuration = StftConfiguration::new(
        input, n_fft, hop_length, win_length, window, center, normalized, onesided,
    )?;
    require_cpu_dtype(output_gradient, DType::Complex64, STFT_OPERATION_ID)?;
    if output_gradient.descriptor().shape() != configuration.output_shape {
        return Err(ElementwiseRuntimePartTwelveError::Invalid(
            "STFT output gradient shape does not match the forward output".to_owned(),
        ));
    }
    let input_length = configuration.input_length;
    let gradient_count = configuration
        .batch_count
        .checked_mul(input_length)
        .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow("STFT VJP"))?;
    let mut gradients = workspace_filled(backend, context, gradient_count, 0.0_f32)?;
    let scale = if normalized {
        1.0 / (n_fft as f32).sqrt()
    } else {
        1.0
    };
    for batch in 0..configuration.batch_count {
        for frame in 0..configuration.frame_count {
            for frequency in 0..configuration.frequency_count {
                check_periodically(frequency, context.cancellation)?;
                let output_indices = configuration.output_indices(batch, frequency, frame)?;
                let (gradient_real, gradient_imaginary) =
                    read_complex64(output_gradient, &output_indices)?;
                for sample in 0..n_fft {
                    let Some(source) = configuration.source_index(frame, sample)? else {
                        continue;
                    };
                    let window_value = configuration.window_value(window, sample)?;
                    if window_value == 0.0 {
                        continue;
                    }
                    let angle = TAU * (frequency * sample) as f32 / n_fft as f32;
                    let contribution = scale
                        * window_value
                        * (gradient_real * angle.cos() - gradient_imaginary * angle.sin());
                    let offset = batch
                        .checked_mul(input_length)
                        .and_then(|value| value.checked_add(source))
                        .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow("STFT VJP"))?;
                    let slot = gradients.get_mut(offset).ok_or(
                        ElementwiseRuntimePartTwelveError::ShapeOverflow("STFT VJP slot"),
                    )?;
                    *slot += contribution;
                }
            }
        }
    }
    upload_f32(backend, input.descriptor().shape(), &gradients, context)
}

pub fn topk_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    k: usize,
    dimension: i64,
    largest: bool,
    sorted: bool,
    context: &ExecutionContext<'_>,
) -> Result<TopKResult, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    require_cpu_dtype(input, DType::F32, TOPK_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let axis_length = usize::try_from(input.descriptor().shape()[axis])
        .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("topk axis"))?;
    if k > axis_length {
        return Err(ElementwiseRuntimePartTwelveError::Invalid(format!(
            "topk k={k} exceeds dimension length {axis_length}"
        )));
    }
    let mut output_shape = input.descriptor().shape().to_vec();
    output_shape[axis] =
        u64::try_from(k).map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("topk k"))?;
    let output_count = element_count(&output_shape)?;
    let slice_count = element_count_excluding(input.descriptor().shape(), axis)?;
    let mut output_values = workspace_filled(backend, context, output_count, 0.0_f32)?;
    let mut output_indices = workspace_filled(backend, context, output_count, 0_i64)?;
    for slice in 0..slice_count {
        check_periodically(slice, context.cancellation)?;
        let base_indices = unravel_excluding(slice, input.descriptor().shape(), axis)?;
        let mut candidates = backend.workspace_vec(context, axis_length)?;
        for index in 0..axis_length {
            let mut indices = base_indices.clone();
            indices[axis] = u64::try_from(index)
                .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("topk index"))?;
            candidates.try_push((read_f32(input, &indices)?, index))?;
        }
        candidates.sort_by(|left, right| compare_topk(*left, *right, largest));
        if !sorted {
            candidates[..k].sort_by_key(|candidate| candidate.1);
        }
        for (output_axis_index, &(value, input_axis_index)) in candidates[..k].iter().enumerate() {
            let mut indices = base_indices.clone();
            indices[axis] = u64::try_from(output_axis_index).map_err(|_| {
                ElementwiseRuntimePartTwelveError::ShapeOverflow("topk output index")
            })?;
            let linear = ravel_index(&indices, &output_shape)?;
            output_values[linear] = value;
            output_indices[linear] = i64::try_from(input_axis_index).map_err(|_| {
                ElementwiseRuntimePartTwelveError::ShapeOverflow("topk index dtype")
            })?;
        }
    }
    Ok(TopKResult {
        values: upload_f32(backend, &output_shape, &output_values, context)?,
        indices: upload_i64(backend, &output_shape, &output_indices, context)?,
    })
}

pub fn topk_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    indices: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    require_cpu_dtype(input, DType::F32, TOPK_OPERATION_ID)?;
    require_cpu_dtype(indices, DType::I64, TOPK_OPERATION_ID)?;
    require_cpu_dtype(output_gradient, DType::F32, TOPK_OPERATION_ID)?;
    if indices.descriptor().shape() != output_gradient.descriptor().shape()
        || indices.descriptor().rank() != input.descriptor().rank()
    {
        return Err(ElementwiseRuntimePartTwelveError::Invalid(
            "topk derivative tensors have incompatible shapes".to_owned(),
        ));
    }
    let axis = normalize_axis(dimension, input.descriptor().rank())?;
    let mut values = workspace_filled(
        backend,
        context,
        element_count(input.descriptor().shape())?,
        0.0_f32,
    )?;
    for linear in 0..element_count(indices.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let output_indices = unravel_index(linear, indices.descriptor().shape())?;
        let selected = read_i64(indices, &output_indices)?;
        let selected = usize::try_from(selected).map_err(|_| {
            ElementwiseRuntimePartTwelveError::Invalid("topk index is negative".to_owned())
        })?;
        let axis_length = usize::try_from(input.descriptor().shape()[axis])
            .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("topk axis"))?;
        if selected >= axis_length {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "topk index is outside the input dimension".to_owned(),
            ));
        }
        let gradient = read_f32(output_gradient, &output_indices)?;
        let mut input_indices = output_indices;
        input_indices[axis] = u64::try_from(selected)
            .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("topk VJP index"))?;
        let input_linear = ravel_index(&input_indices, input.descriptor().shape())?;
        values[input_linear] += gradient;
    }
    upload_f32(backend, input.descriptor().shape(), &values, context)
}

pub fn topk_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    indices: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    require_cpu_dtype(input_tangent, DType::F32, TOPK_OPERATION_ID)?;
    require_cpu_dtype(indices, DType::I64, TOPK_OPERATION_ID)?;
    let axis = normalize_axis(dimension, input_tangent.descriptor().rank())?;
    let mut values =
        backend.workspace_vec(context, element_count(indices.descriptor().shape())?)?;
    for linear in 0..element_count(indices.descriptor().shape())? {
        check_periodically(linear, context.cancellation)?;
        let mut output_indices = unravel_index(linear, indices.descriptor().shape())?;
        let selected = read_i64(indices, &output_indices)?;
        let selected = u64::try_from(selected).map_err(|_| {
            ElementwiseRuntimePartTwelveError::Invalid("topk index is negative".to_owned())
        })?;
        if selected >= input_tangent.descriptor().shape()[axis] {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "topk index is outside the tangent dimension".to_owned(),
            ));
        }
        output_indices[axis] = selected;
        values.try_push(read_f32(input_tangent, &output_indices)?)?;
    }
    upload_f32(backend, indices.descriptor().shape(), &values, context)
}

pub fn tril_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    diagonal: isize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    Ok(crate::generated_elementwise_or_runtime_operation_01::triangular_mask_with_context_exact_native(
        backend, input, diagonal, false, context,
    )?)
}

pub fn tril_vjp_with_context_exact_native(
    backend: &CpuBackend,
    output_gradient: &Tensor,
    diagonal: isize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    tril_with_context_exact_native(backend, output_gradient, diagonal, context)
}

pub fn tril_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input_tangent: &Tensor,
    diagonal: isize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    context.cancellation.check()?;
    tril_with_context_exact_native(backend, input_tangent, diagonal, context)
}

#[derive(Clone, Debug)]
struct StftConfiguration {
    batch_count: usize,
    input_length: usize,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    center_padding: usize,
    frame_count: usize,
    frequency_count: usize,
    normalized: bool,
    output_shape: Vec<u64>,
}

impl StftConfiguration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        input: &Tensor,
        n_fft: usize,
        hop_length: Option<usize>,
        win_length: Option<usize>,
        window: Option<&Tensor>,
        center: bool,
        normalized: bool,
        onesided: bool,
    ) -> Result<Self, ElementwiseRuntimePartTwelveError> {
        require_cpu_dtype(input, DType::F32, STFT_OPERATION_ID)?;
        if !matches!(input.descriptor().rank(), 1 | 2) {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "STFT input rank must be one or two".to_owned(),
            ));
        }
        if n_fft == 0 {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "STFT n_fft must be positive".to_owned(),
            ));
        }
        let hop_length = hop_length.unwrap_or(n_fft / 4);
        if hop_length == 0 {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "STFT hop length must be positive".to_owned(),
            ));
        }
        let win_length = win_length.unwrap_or(n_fft);
        if win_length == 0 || win_length > n_fft {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "STFT window length must be in 1..=n_fft".to_owned(),
            ));
        }
        if let Some(window) = window {
            require_cpu_dtype(window, DType::F32, STFT_OPERATION_ID)?;
            if window.descriptor().shape() != [win_length as u64] {
                return Err(ElementwiseRuntimePartTwelveError::Invalid(
                    "STFT window must be one-dimensional and match win_length".to_owned(),
                ));
            }
        }
        let input_length =
            usize::try_from(*input.descriptor().shape().last().ok_or_else(|| {
                ElementwiseRuntimePartTwelveError::Invalid(
                    "STFT input has no time dimension".to_owned(),
                )
            })?)
            .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("STFT input length"))?;
        let center_padding = if center { n_fft / 2 } else { 0 };
        if center && (input_length <= center_padding || input_length < 2) {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "STFT reflect padding must be smaller than the input length".to_owned(),
            ));
        }
        let padded_length = input_length
            .checked_add(center_padding.checked_mul(2).ok_or(
                ElementwiseRuntimePartTwelveError::ShapeOverflow("STFT padding"),
            )?)
            .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow(
                "STFT padded input",
            ))?;
        if padded_length < n_fft {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "STFT input is shorter than n_fft".to_owned(),
            ));
        }
        let frame_count = 1 + (padded_length - n_fft) / hop_length;
        let frequency_count = if onesided { n_fft / 2 + 1 } else { n_fft };
        let batch_count = if input.descriptor().rank() == 2 {
            usize::try_from(input.descriptor().shape()[0])
                .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("STFT batch"))?
        } else {
            1
        };
        let mut output_shape = if input.descriptor().rank() == 2 {
            vec![batch_count as u64]
        } else {
            Vec::new()
        };
        output_shape.extend([frequency_count as u64, frame_count as u64]);
        Ok(Self {
            batch_count,
            input_length,
            n_fft,
            hop_length,
            win_length,
            center_padding,
            frame_count,
            frequency_count,
            normalized,
            output_shape,
        })
    }

    fn source_index(
        &self,
        frame: usize,
        sample: usize,
    ) -> Result<Option<usize>, ElementwiseRuntimePartTwelveError> {
        let padded = frame
            .checked_mul(self.hop_length)
            .and_then(|value| value.checked_add(sample))
            .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow(
                "STFT source index",
            ))?;
        if self.center_padding == 0 {
            return Ok(Some(padded));
        }
        if padded < self.center_padding {
            return Ok(Some(self.center_padding - padded));
        }
        let shifted = padded - self.center_padding;
        if shifted < self.input_length {
            return Ok(Some(shifted));
        }
        let maximum = self
            .input_length
            .checked_mul(2)
            .and_then(|value| value.checked_sub(2))
            .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow(
                "STFT reflection",
            ))?;
        Ok(Some(maximum.checked_sub(shifted).ok_or_else(|| {
            ElementwiseRuntimePartTwelveError::Invalid(
                "STFT reflection exceeds the supported center padding".to_owned(),
            )
        })?))
    }

    fn window_value(
        &self,
        window: Option<&Tensor>,
        sample: usize,
    ) -> Result<f32, ElementwiseRuntimePartTwelveError> {
        let offset = (self.n_fft - self.win_length) / 2;
        if sample < offset || sample >= offset + self.win_length {
            return Ok(0.0);
        }
        match window {
            Some(window) => read_f32(window, &[(sample - offset) as u64]),
            None => Ok(1.0),
        }
    }

    fn input_indices(&self, batch: usize, source: usize) -> Vec<u64> {
        if self.batch_count == 1 && self.output_shape.len() == 2 {
            vec![source as u64]
        } else {
            vec![batch as u64, source as u64]
        }
    }

    fn output_indices(
        &self,
        batch: usize,
        frequency: usize,
        frame: usize,
    ) -> Result<Vec<u64>, ElementwiseRuntimePartTwelveError> {
        if batch >= self.batch_count
            || frequency >= self.frequency_count
            || frame >= self.frame_count
        {
            return Err(ElementwiseRuntimePartTwelveError::ShapeOverflow(
                "STFT output indices",
            ));
        }
        Ok(if self.output_shape.len() == 2 {
            vec![frequency as u64, frame as u64]
        } else {
            vec![batch as u64, frequency as u64, frame as u64]
        })
    }
}

fn stft_values(
    backend: &CpuBackend,
    input: &Tensor,
    window: Option<&Tensor>,
    configuration: &StftConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<(f32, f32)>, ElementwiseRuntimePartTwelveError> {
    let output_count = element_count(&configuration.output_shape)?;
    let mut values = workspace_filled(backend, context, output_count, (0.0_f32, 0.0_f32))?;
    let scale = if configuration.normalized {
        1.0 / (configuration.n_fft as f32).sqrt()
    } else {
        1.0
    };
    let mut frame_values =
        workspace_filled(backend, context, configuration.n_fft, (0.0_f32, 0.0_f32))?;
    for batch in 0..configuration.batch_count {
        for frame in 0..configuration.frame_count {
            for (sample, slot) in frame_values.iter_mut().enumerate() {
                check_periodically(sample, context.cancellation)?;
                let source = configuration.source_index(frame, sample)?.ok_or(
                    ElementwiseRuntimePartTwelveError::ShapeOverflow("STFT source"),
                )?;
                let value = read_f32(input, &configuration.input_indices(batch, source))?
                    * configuration.window_value(window, sample)?;
                *slot = (value, 0.0);
            }
            complex_fft_in_place(backend, &mut frame_values, false, context)?;
            for (frequency, &(real, imaginary)) in frame_values
                .iter()
                .take(configuration.frequency_count)
                .enumerate()
            {
                let indices = configuration.output_indices(batch, frequency, frame)?;
                let linear = ravel_index(&indices, &configuration.output_shape)?;
                values[linear] = (real * scale, imaginary * scale);
            }
        }
    }
    Ok(values)
}

pub(crate) trait FftScalar:
    Copy
    + Send
    + Sync
    + 'static
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::MulAssign
{
    const ZERO: Self;
    const TAU: Self;
    fn from_f64(value: f64) -> Self;
    fn as_f64(self) -> f64;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
}

impl FftScalar for f32 {
    const ZERO: Self = 0.0;
    const TAU: Self = std::f32::consts::TAU;
    fn from_f64(value: f64) -> Self {
        value as Self
    }
    fn as_f64(self) -> f64 {
        f64::from(self)
    }
    fn cos(self) -> Self {
        f32::cos(self)
    }
    fn sin(self) -> Self {
        f32::sin(self)
    }
}

impl FftScalar for f64 {
    const ZERO: Self = 0.0;
    const TAU: Self = std::f64::consts::TAU;
    fn from_f64(value: f64) -> Self {
        value
    }
    fn as_f64(self) -> f64 {
        self
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
}

pub(crate) fn complex_fft_in_place<Scalar: FftScalar>(
    backend: &CpuBackend,
    values: &mut [(Scalar, Scalar)],
    inverse: bool,
    context: &ExecutionContext<'_>,
) -> Result<(), ElementwiseRuntimePartTwelveError> {
    let direction = Scalar::from_f64(if inverse { 1.0 } else { -1.0 });
    if values.len().is_power_of_two() {
        let mut target = 0_usize;
        for source in 1..values.len() {
            let mut bit = values.len() >> 1;
            while target & bit != 0 {
                target ^= bit;
                bit >>= 1;
            }
            target ^= bit;
            if source < target {
                values.swap(source, target);
            }
        }
        let mut width = 2;
        while width <= values.len() {
            context.check()?;
            let half = width / 2;
            let angle = direction * Scalar::TAU / Scalar::from_f64(width as f64);
            for start in (0..values.len()).step_by(width) {
                for offset in 0..half {
                    let phase = angle * Scalar::from_f64(offset as f64);
                    let twiddle = (phase.cos(), phase.sin());
                    let even = values[start + offset];
                    let odd = multiply_complex(values[start + offset + half], twiddle);
                    values[start + offset] = (even.0 + odd.0, even.1 + odd.1);
                    values[start + offset + half] = (even.0 - odd.0, even.1 - odd.1);
                }
            }
            if width == values.len() {
                break;
            }
            width =
                width
                    .checked_mul(2)
                    .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow(
                        "FFT stage",
                    ))?;
        }
        return Ok(());
    }
    let mut input = backend.workspace_vec(context, values.len())?;
    for value in values.iter().copied() {
        input.try_push(value)?;
    }
    for (frequency, output) in values.iter_mut().enumerate() {
        check_periodically(frequency, context.cancellation)?;
        let mut sum = (Scalar::ZERO, Scalar::ZERO);
        for (sample, value) in input.iter().enumerate() {
            let angle = direction
                * Scalar::TAU
                * Scalar::from_f64(frequency as f64)
                * Scalar::from_f64(sample as f64)
                / Scalar::from_f64(input.len() as f64);
            let term = multiply_complex(*value, (angle.cos(), angle.sin()));
            sum.0 += term.0;
            sum.1 += term.1;
        }
        *output = sum;
    }
    Ok(())
}

fn multiply_complex<Scalar: FftScalar>(
    left: (Scalar, Scalar),
    right: (Scalar, Scalar),
) -> (Scalar, Scalar) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}

fn compare_topk(left: (f32, usize), right: (f32, usize), largest: bool) -> Ordering {
    let value_order = match (left.0.is_nan(), right.0.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left.0.total_cmp(&right.0),
    };
    let order = if largest {
        value_order.reverse()
    } else {
        value_order
    };
    order.then_with(|| left.1.cmp(&right.1))
}

fn normalize_axis(dimension: i64, rank: usize) -> Result<usize, ElementwiseRuntimePartTwelveError> {
    if rank == 0 {
        return Err(ElementwiseRuntimePartTwelveError::Invalid(
            "operation requires a tensor with at least one dimension".to_owned(),
        ));
    }
    let rank_i64 = i64::try_from(rank)
        .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("rank"))?;
    let normalized = if dimension < 0 {
        dimension.checked_add(rank_i64)
    } else {
        Some(dimension)
    }
    .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow("axis"))?;
    if normalized < 0 || normalized >= rank_i64 {
        return Err(ElementwiseRuntimePartTwelveError::Invalid(format!(
            "dimension {dimension} is outside rank {rank}"
        )));
    }
    usize::try_from(normalized)
        .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("axis"))
}

fn require_cpu_dtype(
    input: &Tensor,
    dtype: DType,
    operation: &'static str,
) -> Result<(), ElementwiseRuntimePartTwelveError> {
    if input.descriptor().device() != DeviceId::CPU {
        return Err(ElementwiseRuntimePartTwelveError::UnsupportedDevice {
            operation,
            device: input.descriptor().device(),
        });
    }
    if input.descriptor().dtype() != dtype {
        return Err(ElementwiseRuntimePartTwelveError::UnsupportedDType {
            operation,
            dtype: input.descriptor().dtype(),
        });
    }
    Ok(())
}

fn read_f32(input: &Tensor, indices: &[u64]) -> Result<f32, ElementwiseRuntimePartTwelveError> {
    match input
        .descriptor()
        .dtype()
        .decode_scalar(input.element_bytes(indices)?)?
    {
        DecodedScalar::Real(value) => Ok(value as f32),
        _ => Err(ElementwiseRuntimePartTwelveError::UnsupportedDType {
            operation: STFT_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        }),
    }
}

fn read_i64(input: &Tensor, indices: &[u64]) -> Result<i64, ElementwiseRuntimePartTwelveError> {
    match input
        .descriptor()
        .dtype()
        .decode_scalar(input.element_bytes(indices)?)?
    {
        DecodedScalar::Signed(value) => Ok(value),
        _ => Err(ElementwiseRuntimePartTwelveError::UnsupportedDType {
            operation: TOPK_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        }),
    }
}

fn read_complex64(
    input: &Tensor,
    indices: &[u64],
) -> Result<(f32, f32), ElementwiseRuntimePartTwelveError> {
    match input
        .descriptor()
        .dtype()
        .decode_scalar(input.element_bytes(indices)?)?
    {
        DecodedScalar::Complex { real, imaginary } => Ok((real as f32, imaginary as f32)),
        _ => Err(ElementwiseRuntimePartTwelveError::UnsupportedDType {
            operation: STFT_OPERATION_ID,
            dtype: input.descriptor().dtype(),
        }),
    }
}

fn element_count(shape: &[u64]) -> Result<usize, ElementwiseRuntimePartTwelveError> {
    shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow(
            "element count",
        ))
}

fn element_count_excluding(
    shape: &[u64],
    excluded: usize,
) -> Result<usize, ElementwiseRuntimePartTwelveError> {
    let mut dimensions = shape.to_vec();
    dimensions[excluded] = 1;
    element_count(&dimensions)
}

fn unravel_index(
    mut linear: usize,
    shape: &[u64],
) -> Result<Vec<u64>, ElementwiseRuntimePartTwelveError> {
    let mut indices = vec![0_u64; shape.len()];
    for dimension in (0..shape.len()).rev() {
        let size = usize::try_from(shape[dimension])
            .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("unravel dimension"))?;
        if size == 0 {
            return Ok(indices);
        }
        indices[dimension] = u64::try_from(linear % size)
            .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("unravel index"))?;
        linear /= size;
    }
    Ok(indices)
}

fn unravel_excluding(
    linear: usize,
    shape: &[u64],
    excluded: usize,
) -> Result<Vec<u64>, ElementwiseRuntimePartTwelveError> {
    let mut dimensions = shape.to_vec();
    dimensions[excluded] = 1;
    unravel_index(linear, &dimensions)
}

fn ravel_index(indices: &[u64], shape: &[u64]) -> Result<usize, ElementwiseRuntimePartTwelveError> {
    if indices.len() != shape.len() {
        return Err(ElementwiseRuntimePartTwelveError::Invalid(
            "index rank does not match shape rank".to_owned(),
        ));
    }
    let mut linear = 0_u64;
    for (&index, &dimension) in indices.iter().zip(shape) {
        if index >= dimension {
            return Err(ElementwiseRuntimePartTwelveError::Invalid(
                "index is outside shape".to_owned(),
            ));
        }
        linear = linear
            .checked_mul(dimension)
            .and_then(|value| value.checked_add(index))
            .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow(
                "ravel index",
            ))?;
    }
    usize::try_from(linear)
        .map_err(|_| ElementwiseRuntimePartTwelveError::ShapeOverflow("ravel index"))
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), ElementwiseRuntimePartTwelveError> {
    if index & 0x3ff == 0 {
        cancellation.check()?;
    }
    Ok(())
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_i64(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    let byte_count =
        values
            .len()
            .checked_mul(8)
            .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow(
                "i64 upload",
            ))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for value in values {
        for byte in value.to_ne_bytes() {
            bytes.try_push(byte)?;
        }
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn upload_complex64(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[(f32, f32)],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ElementwiseRuntimePartTwelveError> {
    let byte_count =
        values
            .len()
            .checked_mul(8)
            .ok_or(ElementwiseRuntimePartTwelveError::ShapeOverflow(
                "complex upload",
            ))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for (real, imaginary) in values {
        for byte in real
            .to_ne_bytes()
            .into_iter()
            .chain(imaginary.to_ne_bytes())
        {
            bytes.try_push(byte)?;
        }
    }
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::Complex64,
        DeviceId::CPU,
        context.stream,
    )?;
    Ok(backend.upload_bytes(descriptor, &bytes, context)?.0)
}

fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, ElementwiseRuntimePartTwelveError> {
    let mut values = backend.workspace_vec(context, count)?;
    for _ in 0..count {
        values.try_push(value)?;
    }
    Ok(values)
}
