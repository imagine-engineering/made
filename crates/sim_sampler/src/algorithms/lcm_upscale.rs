use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProgress, SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    CpuBackend, ExecutionContext, ResizeCrop, ResizeMode, ResizeSpec, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorBackend,
    TensorDescriptor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const LCM_UPSCALE_SAMPLER_ID: &str = "lcm_upscale";
pub const LCM_UPSCALE_FEATURE_ID: &str = "COMFY-MODEL-0213";
pub const LCM_UPSCALE_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: LCM_UPSCALE_SAMPLER_ID,
    feature_id: LCM_UPSCALE_FEATURE_ID,
    source_ordinal: 45,
    aliases: &[],
    implementation_module: "algorithms/lcm_upscale",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LcmUpscaleMethod {
    Bislerp,
    NearestExact,
    Bilinear,
    Area,
    Bicubic,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LcmUpscaleOptions {
    total_upscale: f32,
    upscale_steps: Option<u16>,
    method: LcmUpscaleMethod,
}

impl LcmUpscaleOptions {
    pub fn new(
        total_upscale: f32,
        upscale_steps: Option<u16>,
        method: LcmUpscaleMethod,
    ) -> Result<Self, LcmUpscaleError> {
        if !total_upscale.is_finite() || !(0.1..=20.0).contains(&total_upscale) {
            return Err(LcmUpscaleError::InvalidScale(total_upscale));
        }
        if upscale_steps.is_some_and(|steps| steps > 1_000) {
            return Err(LcmUpscaleError::InvalidUpscaleSteps(upscale_steps));
        }
        Ok(Self {
            total_upscale,
            upscale_steps,
            method,
        })
    }

    pub fn source_defaults() -> Self {
        Self {
            total_upscale: 2.0,
            upscale_steps: None,
            method: LcmUpscaleMethod::Bislerp,
        }
    }

    pub const fn total_upscale(self) -> f32 {
        self.total_upscale
    }

    pub const fn upscale_steps(self) -> Option<u16> {
        self.upscale_steps
    }

    pub const fn method(self) -> LcmUpscaleMethod {
        self.method
    }
}

impl Default for LcmUpscaleOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Debug, Error)]
pub enum LcmUpscaleError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error("LCM-upscale requires sampler identity `lcm_upscale`, got {0:?}")]
    WrongSampler(String),
    #[error("LCM-upscale total scale must be finite and in [0.1, 20], got {0}")]
    InvalidScale(f32),
    #[error("LCM-upscale step count must be at most 1000, got {0:?}")]
    InvalidUpscaleSteps(Option<u16>),
    #[error("LCM-upscale requires a contiguous CPU f32 NCHW tensor")]
    InvalidTensor,
    #[error("LCM-upscale denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("LCM-upscale denoiser output descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("LCM-upscale output dimension overflowed")]
    DimensionOverflow,
    #[error("LCM-upscale produced a non-finite value at step {step}, element {element}")]
    NonFinite { step: usize, element: usize },
    #[error("LCM-upscale allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_lcm_upscale<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: LcmUpscaleOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), LcmUpscaleError>
where
    CallbackError: Display,
{
    context.check()?;
    let options =
        LcmUpscaleOptions::new(options.total_upscale, options.upscale_steps, options.method)?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        plan.profile(),
    )?;
    if plan.sampler().as_str() != LCM_UPSCALE_SAMPLER_ID {
        return Err(LcmUpscaleError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    validate_tensor(&initial)?;
    let shape = initial.descriptor().shape();
    let original_height = *shape.get(2).ok_or(LcmUpscaleError::InvalidTensor)?;
    let original_width = *shape.get(3).ok_or(LcmUpscaleError::InvalidTensor)?;
    let transition_count = sigmas.len().saturating_sub(1);
    let requested_points = match options.upscale_steps {
        Some(steps) => usize::from(steps)
            .checked_add(1)
            .ok_or(LcmUpscaleError::DimensionOverflow)?
            .min(sigmas.len().saturating_add(1)),
        None => (sigmas.len() / 2 + 1).max(2),
    };
    let upscale_count = requested_points.saturating_sub(1);

    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| LcmUpscaleError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan.clone(), owned_sigmas, initial)?;
    let mut noise = noise_request.open_transaction(
        LCM_UPSCALE_NOISE_CONTRACT_ID,
        i128::from(plan.seed()),
        RngSeedTransform::Add(0),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: sim_tensor::DeviceId::CPU,
        },
        None,
        context.cancellation,
    )?;
    let noise_before = noise.checkpoint();

    for (step, transition) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = transition[0];
        let next_sigma = transition[1];
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| LcmUpscaleError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(LcmUpscaleError::DenoiserContract { step });
        }
        let observed = session.observe_step(
            &current,
            denoised.clone(),
            context.cancellation,
            |progress, latent, output| callback(progress, latent, output),
        )?;
        let mut next = denoised;
        if step < upscale_count {
            let ratio =
                1.0 + (options.total_upscale - 1.0) * (step + 1) as f32 / upscale_count as f32;
            let width = rounded_dimension(original_width, ratio)?;
            let height = rounded_dimension(original_height, ratio)?;
            next = resize(backend, &next, width, height, options.method, context)?;
        }
        if next_sigma > 0.0 {
            let count = usize::try_from(next.descriptor().element_count()?)
                .map_err(|_| LcmUpscaleError::DimensionOverflow)?;
            let generated = noise.draw_normal(count, context.cancellation)?;
            let values = tensor_to_f32(backend, &next, context)?;
            let mut noisy = backend.workspace_vec::<f32>(context, count)?;
            for (element, (value, random)) in values
                .iter()
                .copied()
                .zip(generated)
                .enumerate()
            {
                let value = value + next_sigma * random as f32;
                if !value.is_finite() {
                    return Err(LcmUpscaleError::NonFinite { step, element });
                }
                noisy.try_push(value)?;
            }
            next = tensor_from_f32(backend, next.descriptor().shape(), &noisy, context)?;
        }
        observed.commit_shape_transition(next, context.cancellation)?;
    }
    if session.next_step() != transition_count {
        return Err(LcmUpscaleError::DimensionOverflow);
    }
    let trace = session.finish()?;
    let noise_after = noise.commit();
    Ok((trace, noise_before, noise_after))
}

fn rounded_dimension(original: u64, ratio: f32) -> Result<u64, LcmUpscaleError> {
    let value = (original as f64 * f64::from(ratio)).round_ties_even();
    if !value.is_finite() || value < 1.0 || value > u64::MAX as f64 {
        return Err(LcmUpscaleError::DimensionOverflow);
    }
    Ok(value as u64)
}

fn validate_tensor(tensor: &Tensor) -> Result<(), LcmUpscaleError> {
    let descriptor = tensor.descriptor();
    if descriptor.shape().len() != 4
        || descriptor.dtype() != sim_tensor::DType::F32
        || descriptor.device() != sim_tensor::DeviceId::CPU
        || !descriptor.is_contiguous()?
    {
        return Err(LcmUpscaleError::InvalidTensor);
    }
    Ok(())
}

fn resize(
    backend: &CpuBackend,
    input: &Tensor,
    width: u64,
    height: u64,
    method: LcmUpscaleMethod,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LcmUpscaleError> {
    let mode = match method {
        LcmUpscaleMethod::Bislerp => return bislerp(backend, input, width, height, context),
        LcmUpscaleMethod::NearestExact => ResizeMode::NearestExact,
        LcmUpscaleMethod::Bilinear => ResizeMode::Bilinear,
        LcmUpscaleMethod::Area => ResizeMode::Area,
        LcmUpscaleMethod::Bicubic => ResizeMode::Bicubic,
    };
    let output = TensorDescriptor::contiguous(
        vec![
            input.descriptor().shape()[0],
            input.descriptor().shape()[1],
            height,
            width,
        ],
        input.descriptor().dtype(),
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    backend
        .resize(
            ResizeSpec {
                width,
                height,
                mode,
                crop: ResizeCrop::Disabled,
                antialias: false,
                align_corners: false,
            },
            input,
            output,
            context,
        )
        .map(|(tensor, _)| tensor)
        .map_err(LcmUpscaleError::Tensor)
}

fn bislerp(
    backend: &CpuBackend,
    input: &Tensor,
    width: u64,
    height: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LcmUpscaleError> {
    let shape = input.descriptor().shape();
    let [batch_count, channel_count, input_height, input_width] = shape else {
        return Err(LcmUpscaleError::InvalidTensor);
    };
    let batches = usize::try_from(*batch_count).map_err(|_| LcmUpscaleError::DimensionOverflow)?;
    let channels =
        usize::try_from(*channel_count).map_err(|_| LcmUpscaleError::DimensionOverflow)?;
    let input_height =
        usize::try_from(*input_height).map_err(|_| LcmUpscaleError::DimensionOverflow)?;
    let input_width =
        usize::try_from(*input_width).map_err(|_| LcmUpscaleError::DimensionOverflow)?;
    let output_height = usize::try_from(height).map_err(|_| LcmUpscaleError::DimensionOverflow)?;
    let output_width = usize::try_from(width).map_err(|_| LcmUpscaleError::DimensionOverflow)?;
    let input_values = tensor_to_f32(backend, input, context)?;
    let horizontal_count = batches
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(input_height))
        .and_then(|value| value.checked_mul(output_width))
        .ok_or(LcmUpscaleError::DimensionOverflow)?;
    let mut horizontal = backend.workspace_vec::<f32>(context, horizontal_count)?;
    for _ in 0..horizontal_count {
        horizontal.try_push(0.0)?;
    }
    for batch in 0..batches {
        for row in 0..input_height {
            for output_column in 0..output_width {
                context.check()?;
                let (first, second, ratio) =
                    bilinear_coordinate(input_width, output_width, output_column)?;
                slerp_channels(
                    &input_values,
                    &mut horizontal,
                    batch,
                    row,
                    first,
                    second,
                    ratio,
                    channels,
                    input_height,
                    input_width,
                    input_height,
                    output_width,
                    output_column,
                )?;
            }
        }
    }
    let output_count = batches
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_height))
        .and_then(|value| value.checked_mul(output_width))
        .ok_or(LcmUpscaleError::DimensionOverflow)?;
    let mut output = backend.workspace_vec::<f32>(context, output_count)?;
    for _ in 0..output_count {
        output.try_push(0.0)?;
    }
    for batch in 0..batches {
        for output_row in 0..output_height {
            let (first, second, ratio) =
                bilinear_coordinate(input_height, output_height, output_row)?;
            for column in 0..output_width {
                context.check()?;
                slerp_rows(
                    &horizontal,
                    &mut output,
                    batch,
                    first,
                    second,
                    column,
                    output_row,
                    ratio,
                    channels,
                    input_height,
                    output_height,
                    output_width,
                )?;
            }
        }
    }
    tensor_from_f32(
        backend,
        &[*batch_count, *channel_count, height, width],
        &output,
        context,
    )
    .map_err(LcmUpscaleError::TensorKernel)
}

fn bilinear_coordinate(
    input_length: usize,
    output_length: usize,
    output_index: usize,
) -> Result<(usize, usize, f32), LcmUpscaleError> {
    if input_length == 0 || output_length == 0 || output_index >= output_length {
        return Err(LcmUpscaleError::InvalidTensor);
    }
    let coordinate = ((output_index as f64 + 0.5) * input_length as f64 / output_length as f64
        - 0.5)
        .clamp(0.0, input_length.saturating_sub(1) as f64);
    let first = coordinate.floor() as usize;
    let second = first.saturating_add(1).min(input_length.saturating_sub(1));
    Ok((first, second, (coordinate - first as f64) as f32))
}

#[allow(clippy::too_many_arguments)]
fn slerp_channels(
    source: &[f32],
    target: &mut [f32],
    batch: usize,
    source_row: usize,
    first_column: usize,
    second_column: usize,
    ratio: f32,
    channels: usize,
    source_height: usize,
    source_width: usize,
    target_height: usize,
    target_width: usize,
    target_column: usize,
) -> Result<(), LcmUpscaleError> {
    let target_row = source_row.min(target_height.saturating_sub(1));
    let mut first = Vec::new();
    let mut second = Vec::new();
    first
        .try_reserve_exact(channels)
        .map_err(|_| LcmUpscaleError::OutOfMemory("bislerp channels"))?;
    second
        .try_reserve_exact(channels)
        .map_err(|_| LcmUpscaleError::OutOfMemory("bislerp channels"))?;
    for channel in 0..channels {
        first.push(
            source[index(
                batch,
                channel,
                source_row,
                first_column,
                channels,
                source_height,
                source_width,
            )?],
        );
        second.push(
            source[index(
                batch,
                channel,
                source_row,
                second_column,
                channels,
                source_height,
                source_width,
            )?],
        );
    }
    let values = spherical_interpolate(&first, &second, ratio)?;
    for (channel, value) in values.into_iter().enumerate() {
        let target_index = index(
            batch,
            channel,
            target_row,
            target_column,
            channels,
            target_height,
            target_width,
        )?;
        target[target_index] = value;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn slerp_rows(
    source: &[f32],
    target: &mut [f32],
    batch: usize,
    first_row: usize,
    second_row: usize,
    column: usize,
    target_row: usize,
    ratio: f32,
    channels: usize,
    source_height: usize,
    target_height: usize,
    width: usize,
) -> Result<(), LcmUpscaleError> {
    let mut first = Vec::new();
    let mut second = Vec::new();
    first
        .try_reserve_exact(channels)
        .map_err(|_| LcmUpscaleError::OutOfMemory("bislerp channels"))?;
    second
        .try_reserve_exact(channels)
        .map_err(|_| LcmUpscaleError::OutOfMemory("bislerp channels"))?;
    for channel in 0..channels {
        first.push(
            source[index(
                batch,
                channel,
                first_row,
                column,
                channels,
                source_height,
                width,
            )?],
        );
        second.push(
            source[index(
                batch,
                channel,
                second_row,
                column,
                channels,
                source_height,
                width,
            )?],
        );
    }
    let values = spherical_interpolate(&first, &second, ratio)?;
    for (channel, value) in values.into_iter().enumerate() {
        target[index(
            batch,
            channel,
            target_row,
            column,
            channels,
            target_height,
            width,
        )?] = value;
    }
    Ok(())
}

fn spherical_interpolate(
    first: &[f32],
    second: &[f32],
    ratio: f32,
) -> Result<Vec<f32>, LcmUpscaleError> {
    let first_norm = first.iter().map(|value| value * value).sum::<f32>().sqrt();
    let second_norm = second.iter().map(|value| value * value).sum::<f32>().sqrt();
    let dot = first
        .iter()
        .zip(second)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let normalized_dot = if first_norm == 0.0 || second_norm == 0.0 {
        0.0
    } else {
        dot / (first_norm * second_norm)
    };
    let mut output = Vec::new();
    output
        .try_reserve_exact(first.len())
        .map_err(|_| LcmUpscaleError::OutOfMemory("bislerp output"))?;
    if normalized_dot > 1.0 - 1.0e-5 {
        output.extend_from_slice(first);
        return Ok(output);
    }
    if normalized_dot < 1.0e-5 - 1.0 {
        output.extend(
            first
                .iter()
                .zip(second)
                .map(|(left, right)| left * (1.0 - ratio) + right * ratio),
        );
        return Ok(output);
    }
    let omega = normalized_dot.acos();
    let sine = omega.sin();
    let scale = first_norm * (1.0 - ratio) + second_norm * ratio;
    let first_weight = ((1.0 - ratio) * omega).sin() / sine;
    let second_weight = (ratio * omega).sin() / sine;
    output.extend(first.iter().zip(second).map(|(left, right)| {
        scale
            * (first_weight * left / first_norm.max(f32::MIN_POSITIVE)
                + second_weight * right / second_norm.max(f32::MIN_POSITIVE))
    }));
    Ok(output)
}

fn index(
    batch: usize,
    channel: usize,
    row: usize,
    column: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<usize, LcmUpscaleError> {
    batch
        .checked_mul(channels)
        .and_then(|value| value.checked_add(channel))
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_add(row))
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_add(column))
        .ok_or(LcmUpscaleError::DimensionOverflow)
}
