use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DeviceId, ExecutionContext, Tensor,
    TensorDescriptor, TensorError,
};
use thiserror::Error;

use crate::{NativeModule, NativeOpsError};

#[derive(Clone, Debug, PartialEq)]
pub enum PeriodicActivation {
    Snake {
        alpha: Vec<f32>,
        logscale: bool,
    },
    SnakeBeta {
        alpha: Vec<f32>,
        beta: Vec<f32>,
        logscale: bool,
    },
}

impl PeriodicActivation {
    fn channels(&self) -> usize {
        match self {
            Self::Snake { alpha, .. } | Self::SnakeBeta { alpha, .. } => alpha.len(),
        }
    }

    fn apply(&self, channel: usize, value: f32) -> Result<f32, AliasFreeActivationError> {
        let (alpha, beta, logscale) = match self {
            Self::Snake { alpha, logscale } => (
                *alpha.get(channel).ok_or(AliasFreeActivationError::Invalid(
                    "activation channel is outside alpha parameters",
                ))?,
                *alpha.get(channel).ok_or(AliasFreeActivationError::Invalid(
                    "activation channel is outside alpha parameters",
                ))?,
                *logscale,
            ),
            Self::SnakeBeta {
                alpha,
                beta,
                logscale,
            } => (
                *alpha.get(channel).ok_or(AliasFreeActivationError::Invalid(
                    "activation channel is outside alpha parameters",
                ))?,
                *beta.get(channel).ok_or(AliasFreeActivationError::Invalid(
                    "activation channel is outside beta parameters",
                ))?,
                *logscale,
            ),
        };
        let alpha = if logscale { alpha.exp() } else { alpha };
        let beta = if logscale { beta.exp() } else { beta };
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(AliasFreeActivationError::Invalid(
                "periodic activation parameters must remain finite",
            ));
        }
        let periodic = (value * alpha).sin();
        Ok(value + periodic * periodic / (beta + 1e-9))
    }

    fn validate(&self) -> Result<(), AliasFreeActivationError> {
        if self.channels() == 0 {
            return Err(AliasFreeActivationError::Invalid(
                "periodic activation requires at least one channel",
            ));
        }
        match self {
            Self::Snake { alpha, .. } => {
                if alpha.iter().any(|value| !value.is_finite()) {
                    return Err(AliasFreeActivationError::Invalid(
                        "Snake alpha parameters must be finite",
                    ));
                }
            }
            Self::SnakeBeta { alpha, beta, .. } => {
                if alpha.len() != beta.len()
                    || alpha.iter().chain(beta).any(|value| !value.is_finite())
                {
                    return Err(AliasFreeActivationError::Invalid(
                        "SnakeBeta alpha and beta parameters must be finite and shape matched",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AliasFreeActivationError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Module(#[from] NativeOpsError),
    #[error("alias-free activation configuration is invalid: {0}")]
    Invalid(&'static str),
    #[error("alias-free activation was cancelled")]
    Cancelled,
    #[error("alias-free activation shape arithmetic overflowed")]
    ShapeOverflow,
}

impl From<sim_types::CancellationError> for AliasFreeActivationError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
pub struct NativeAliasFreeActivation1d {
    base: NativeModule,
    activation: PeriodicActivation,
    up_ratio: usize,
    down_ratio: usize,
    up_kernel: Vec<f32>,
    down_kernel: Vec<f32>,
}

impl NativeAliasFreeActivation1d {
    pub fn base(&self) -> &NativeModule {
        &self.base
    }

    pub fn forward_with_context(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, AliasFreeActivationError> {
        self.forward_impl(backend, input, context)
    }

    fn forward_impl(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, AliasFreeActivationError> {
        context.cancellation.check()?;
        if input.descriptor().device() != DeviceId::CPU
            || input.descriptor().dtype() != DType::F32
            || input.descriptor().rank() != 3
        {
            return Err(AliasFreeActivationError::Invalid(
                "input must be a rank-three CPU F32 tensor",
            ));
        }
        let batch = usize::try_from(input.descriptor().shape()[0])
            .map_err(|_| AliasFreeActivationError::ShapeOverflow)?;
        let channels = usize::try_from(input.descriptor().shape()[1])
            .map_err(|_| AliasFreeActivationError::ShapeOverflow)?;
        let samples = usize::try_from(input.descriptor().shape()[2])
            .map_err(|_| AliasFreeActivationError::ShapeOverflow)?;
        if channels != self.activation.channels() || samples == 0 {
            return Err(AliasFreeActivationError::Invalid(
                "input channels must match activation parameters and time must be nonempty",
            ));
        }
        let input_values = logical_f32(backend, input, context)?;
        let upsampled = upsample(
            backend,
            &input_values,
            batch,
            channels,
            samples,
            self.up_ratio,
            &self.up_kernel,
            context,
        )?;
        drop(input_values);
        let upsampled_samples =
            upsample_output_samples(samples, self.up_ratio, self.up_kernel.len())?;
        let mut upsampled = upsampled;
        for (linear, value) in upsampled.iter_mut().enumerate() {
            check_periodically(linear, context.cancellation)?;
            let channel = (linear / upsampled_samples) % channels;
            *value = self.activation.apply(channel, *value)?;
        }
        let downsampled = downsample(
            backend,
            &upsampled,
            batch,
            channels,
            upsampled_samples,
            self.down_ratio,
            &self.down_kernel,
            context,
        )?;
        let output_samples =
            downsample_output_samples(upsampled_samples, self.down_ratio, self.down_kernel.len())?;
        let output_shape = vec![
            u64::try_from(batch).map_err(|_| AliasFreeActivationError::ShapeOverflow)?,
            u64::try_from(channels).map_err(|_| AliasFreeActivationError::ShapeOverflow)?,
            u64::try_from(output_samples).map_err(|_| AliasFreeActivationError::ShapeOverflow)?,
        ];
        let descriptor = TensorDescriptor::contiguous(
            output_shape,
            DType::F32,
            DeviceId::CPU,
            input.descriptor().stream(),
        )?;
        drop(upsampled);
        context.cancellation.check()?;
        Ok(backend.upload_f32(descriptor, &downsampled, context)?.0)
    }
}

pub fn alias_free_activation_1d_exact_native(
    activation: PeriodicActivation,
    up_ratio: usize,
    down_ratio: usize,
    up_kernel_size: usize,
    down_kernel_size: usize,
    cancellation: &CancellationToken,
) -> Result<NativeAliasFreeActivation1d, AliasFreeActivationError> {
    cancellation.check()?;
    activation.validate()?;
    if up_ratio == 0 || down_ratio == 0 || up_kernel_size == 0 || down_kernel_size == 0 {
        return Err(AliasFreeActivationError::Invalid(
            "sample ratios and kernel sizes must be nonzero",
        ));
    }
    let base = NativeModule::container("alias_free_torch.Activation1d")?;
    let up_kernel =
        kaiser_sinc_filter(0.5 / up_ratio as f64, 0.6 / up_ratio as f64, up_kernel_size)?;
    let down_kernel = kaiser_sinc_filter(
        0.5 / down_ratio as f64,
        0.6 / down_ratio as f64,
        down_kernel_size,
    )?;
    cancellation.check()?;
    Ok(NativeAliasFreeActivation1d {
        base,
        activation,
        up_ratio,
        down_ratio,
        up_kernel,
        down_kernel,
    })
}

fn kaiser_sinc_filter(
    cutoff: f64,
    half_width: f64,
    kernel_size: usize,
) -> Result<Vec<f32>, AliasFreeActivationError> {
    let half_size = kernel_size / 2;
    let delta_frequency = 4.0 * half_width;
    let attenuation =
        2.285 * (half_size.saturating_sub(1) as f64) * std::f64::consts::PI * delta_frequency
            + 7.95;
    let beta = if attenuation > 50.0 {
        0.1102 * (attenuation - 8.7)
    } else if attenuation >= 21.0 {
        0.5842 * (attenuation - 21.0).powf(0.4) + 0.07886 * (attenuation - 21.0)
    } else {
        0.0
    };
    let denominator = modified_bessel_zero(beta);
    let mut filter = Vec::new();
    filter
        .try_reserve_exact(kernel_size)
        .map_err(|_| AliasFreeActivationError::ShapeOverflow)?;
    for index in 0..kernel_size {
        let normalized = if kernel_size <= 1 {
            0.0
        } else {
            2.0 * index as f64 / (kernel_size - 1) as f64 - 1.0
        };
        let window = modified_bessel_zero(beta * (1.0 - normalized * normalized).max(0.0).sqrt())
            / denominator;
        let time = if kernel_size.is_multiple_of(2) {
            index as f64 - half_size as f64 + 0.5
        } else {
            index as f64 - half_size as f64
        };
        let argument = 2.0 * cutoff * time;
        let sinc = if argument == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * argument).sin() / (std::f64::consts::PI * argument)
        };
        filter.push((2.0 * cutoff * window * sinc) as f32);
    }
    let sum = filter.iter().map(|value| f64::from(*value)).sum::<f64>();
    if !sum.is_finite() || sum == 0.0 {
        return Err(AliasFreeActivationError::Invalid(
            "Kaiser-sinc filter normalization is invalid",
        ));
    }
    for value in &mut filter {
        *value = (f64::from(*value) / sum) as f32;
    }
    Ok(filter)
}

fn modified_bessel_zero(value: f64) -> f64 {
    let squared = value * value / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for order in 1..=32 {
        term *= squared / (order * order) as f64;
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    sum
}

fn upsample(
    backend: &CpuBackend,
    input: &[f32],
    batch: usize,
    channels: usize,
    samples: usize,
    ratio: usize,
    kernel: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<TemporaryVec<f32>, AliasFreeActivationError> {
    let pad = kernel
        .len()
        .checked_div(ratio)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AliasFreeActivationError::Invalid(
            "upsample kernel must be at least twice the ratio",
        ))?;
    let padded_samples = samples
        .checked_add(
            pad.checked_mul(2)
                .ok_or(AliasFreeActivationError::ShapeOverflow)?,
        )
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    let raw_samples = padded_samples
        .checked_sub(1)
        .and_then(|value| value.checked_mul(ratio))
        .and_then(|value| value.checked_add(kernel.len()))
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    let crop_left = pad
        .checked_mul(ratio)
        .and_then(|value| value.checked_add((kernel.len() - ratio) / 2))
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    let output_samples = upsample_output_samples(samples, ratio, kernel.len())?;
    let output_len = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_samples))
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    let mut output = temporary_filled(backend, context, output_len, 0.0)?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            let mut raw = temporary_filled(backend, context, raw_samples, 0.0)?;
            for padded_index in 0..padded_samples {
                check_periodically(padded_index, context.cancellation)?;
                let source = padded_index.saturating_sub(pad).min(samples - 1);
                let input_index = ((batch_index * channels + channel) * samples) + source;
                let value = *input
                    .get(input_index)
                    .ok_or(AliasFreeActivationError::ShapeOverflow)?;
                let start = padded_index
                    .checked_mul(ratio)
                    .ok_or(AliasFreeActivationError::ShapeOverflow)?;
                for (kernel_index, coefficient) in kernel.iter().enumerate() {
                    let destination = start
                        .checked_add(kernel_index)
                        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
                    *raw.get_mut(destination)
                        .ok_or(AliasFreeActivationError::ShapeOverflow)? +=
                        ratio as f32 * value * coefficient;
                }
            }
            let output_start = (batch_index * channels + channel) * output_samples;
            let source = raw
                .get(crop_left..crop_left + output_samples)
                .ok_or(AliasFreeActivationError::ShapeOverflow)?;
            output
                .get_mut(output_start..output_start + output_samples)
                .ok_or(AliasFreeActivationError::ShapeOverflow)?
                .copy_from_slice(source);
        }
    }
    context.cancellation.check()?;
    Ok(output)
}

fn downsample(
    backend: &CpuBackend,
    input: &[f32],
    batch: usize,
    channels: usize,
    samples: usize,
    ratio: usize,
    kernel: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<TemporaryVec<f32>, AliasFreeActivationError> {
    let even = usize::from(kernel.len().is_multiple_of(2));
    let pad_left = kernel.len() / 2 - even;
    let output_samples = downsample_output_samples(samples, ratio, kernel.len())?;
    let output_len = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_samples))
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    let mut output = temporary_filled(backend, context, output_len, 0.0)?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            for output_index in 0..output_samples {
                check_periodically(output_index, context.cancellation)?;
                let mut sum = 0.0_f32;
                let start = output_index
                    .checked_mul(ratio)
                    .ok_or(AliasFreeActivationError::ShapeOverflow)?;
                for (kernel_index, coefficient) in kernel.iter().enumerate() {
                    let padded_index = start
                        .checked_add(kernel_index)
                        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
                    let source = padded_index.saturating_sub(pad_left).min(samples - 1);
                    let input_index = ((batch_index * channels + channel) * samples) + source;
                    sum += input
                        .get(input_index)
                        .copied()
                        .ok_or(AliasFreeActivationError::ShapeOverflow)?
                        * coefficient;
                }
                let destination =
                    ((batch_index * channels + channel) * output_samples) + output_index;
                *output
                    .get_mut(destination)
                    .ok_or(AliasFreeActivationError::ShapeOverflow)? = sum;
            }
        }
    }
    context.cancellation.check()?;
    Ok(output)
}

fn upsample_output_samples(
    samples: usize,
    ratio: usize,
    kernel_size: usize,
) -> Result<usize, AliasFreeActivationError> {
    let pad = kernel_size
        .checked_div(ratio)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AliasFreeActivationError::Invalid(
            "upsample kernel must be at least twice the ratio",
        ))?;
    let padded_samples = samples
        .checked_add(
            pad.checked_mul(2)
                .ok_or(AliasFreeActivationError::ShapeOverflow)?,
        )
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    let raw_samples = padded_samples
        .checked_sub(1)
        .and_then(|value| value.checked_mul(ratio))
        .and_then(|value| value.checked_add(kernel_size))
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    let crop_left = pad
        .checked_mul(ratio)
        .and_then(|value| value.checked_add((kernel_size - ratio) / 2))
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    let crop_right = pad
        .checked_mul(ratio)
        .and_then(|value| value.checked_add((kernel_size - ratio).div_ceil(2)))
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    raw_samples
        .checked_sub(crop_left)
        .and_then(|value| value.checked_sub(crop_right))
        .ok_or(AliasFreeActivationError::ShapeOverflow)
}

fn downsample_output_samples(
    samples: usize,
    ratio: usize,
    kernel_size: usize,
) -> Result<usize, AliasFreeActivationError> {
    let even = usize::from(kernel_size.is_multiple_of(2));
    let padded_samples = samples
        .checked_add(kernel_size / 2 - even)
        .and_then(|value| value.checked_add(kernel_size / 2))
        .ok_or(AliasFreeActivationError::ShapeOverflow)?;
    padded_samples
        .checked_sub(kernel_size)
        .and_then(|value| value.checked_div(ratio))
        .and_then(|value| value.checked_add(1))
        .ok_or(AliasFreeActivationError::ShapeOverflow)
}

fn logical_f32(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<TemporaryVec<f32>, AliasFreeActivationError> {
    let shape = input.descriptor().shape();
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(AliasFreeActivationError::ShapeOverflow)
    })?;
    let count = usize::try_from(count).map_err(|_| AliasFreeActivationError::ShapeOverflow)?;
    let mut values = temporary_vec(backend, context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let linear = u64::try_from(linear).map_err(|_| AliasFreeActivationError::ShapeOverflow)?;
        let bytes: [u8; 4] = input
            .linear_element_bytes(linear)?
            .try_into()
            .map_err(|_| AliasFreeActivationError::Invalid("input dtype must be F32"))?;
        values.try_push(f32::from_ne_bytes(bytes))?;
    }
    Ok(values)
}

type TemporaryVec<T> = CpuWorkspaceVec<T>;

fn temporary_vec<T>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
) -> Result<TemporaryVec<T>, AliasFreeActivationError> {
    Ok(backend.workspace_vec(context, capacity)?)
}

fn temporary_filled<T: Clone>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    length: usize,
    value: T,
) -> Result<TemporaryVec<T>, AliasFreeActivationError> {
    let mut values = temporary_vec(backend, context, length)?;
    for _ in 0..length {
        values.try_push(value.clone())?;
    }
    Ok(values)
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), AliasFreeActivationError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}
