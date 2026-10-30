use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProfileError, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    BackendCapabilityMatrix, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const LCM_SAMPLER_ID: &str = "lcm";
pub const LCM_FEATURE_ID: &str = "COMFY-MODEL-0191";
pub const LCM_SOURCE_ORDINAL: u16 = 26;
pub const LCM_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: LCM_SAMPLER_ID,
    feature_id: LCM_FEATURE_ID,
    source_ordinal: LCM_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/lcm_comfy_model_0191",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LcmOptions {
    pub noise_scale_start: f32,
    pub noise_scale_end: Option<f32>,
    pub noise_clip_standard_deviations: f32,
}

impl Default for LcmOptions {
    fn default() -> Self {
        Self {
            noise_scale_start: 1.0,
            noise_scale_end: None,
            noise_clip_standard_deviations: 0.0,
        }
    }
}

#[derive(Debug, Error)]
pub enum LcmError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    SamplingProfile(#[from] SamplingProfileError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error("LCM requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("LCM option {name} must be finite, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("LCM denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("LCM denoiser descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("LCM noise clipping requires at least two elements at step {step}")]
    NoiseClipCardinality { step: usize },
    #[error("LCM produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("native LCM noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
    #[error("LCM allocation failed for {0}")]
    OutOfMemory(&'static str),
}

pub fn lcm_rng_profile(device: DeviceId) -> (RngSeedTransform, RngGenerationPlacement) {
    if device == DeviceId::CPU {
        (
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: device,
            },
        )
    } else {
        (
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(device),
        )
    }
}

pub fn validate_lcm_generation_device(device: DeviceId) -> Result<(), LcmError> {
    if device == DeviceId::CPU {
        return Ok(());
    }
    BackendCapabilityMatrix::for_native_device(device).map_err(|error| {
        LcmError::DeviceUnavailable {
            device,
            reason: error.reason().to_owned(),
        }
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sample_lcm<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: LcmOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), LcmError>
where
    CallbackError: Display,
{
    context.check()?;
    validate_options(options)?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        profile.identity(),
    )?;
    if plan.sampler().as_str() != LCM_SAMPLER_ID {
        return Err(LcmError::WrongSampler {
            expected: LCM_SAMPLER_ID,
            actual: plan.sampler().as_str().to_owned(),
        });
    }

    let seed = plan.seed();
    let device = initial.descriptor().device();
    validate_lcm_generation_device(device)?;
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| LcmError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let (seed_transform, generation_placement) = lcm_rng_profile(device);
    let mut noise_transaction = noise_request.open_transaction(
        LCM_NOISE_CONTRACT_ID,
        i128::from(seed),
        seed_transform,
        generation_placement,
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();
    let total_steps = sigmas.len().saturating_sub(1).max(1);
    let noise_scale_end = options
        .noise_scale_end
        .unwrap_or(options.noise_scale_start);

    for (step, sigma_pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = sigma_pair
            .first()
            .copied()
            .ok_or(SamplingError::Overflow("LCM source sigma lookup"))?;
        let next_sigma = sigma_pair
            .get(1)
            .copied()
            .ok_or(SamplingError::Overflow("LCM target sigma lookup"))?;
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| LcmError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(LcmError::DenoiserContract { step });
        }
        let denoised_for_equation = denoised.clone();
        let observed = session.observe_step(
            &current,
            denoised,
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;
        let denoised_values = tensor_to_f32(backend, &denoised_for_equation, context)?;
        validate_finite_values(&denoised_values, step, "denoiser output")?;

        let next = if next_sigma > 0.0 {
            let count = usize::try_from(current.descriptor().element_count()?)
                .map_err(|_| TensorError::ShapeOverflow)?;
            let raw_noise = noise_transaction.draw_normal(count, context.cancellation)?;
            let interpolation = if total_steps > 1 {
                step as f32 / (total_steps - 1) as f32
            } else {
                0.0
            };
            let noise_scale = options.noise_scale_start
                + (noise_scale_end - options.noise_scale_start) * interpolation;
            let mut noise_values = prepare_noise(
                backend,
                raw_noise,
                noise_scale,
                options.noise_clip_standard_deviations,
                step,
                context,
            )?;
            profile.scale_initial_noise_in_place(
                &mut noise_values,
                &denoised_values,
                next_sigma,
                false,
            )?;
            validate_finite_values(&noise_values, step, "noise-scaled latent")?;
            tensor_from_f32(
                backend,
                current.descriptor().shape(),
                &noise_values,
                context,
            )?
        } else {
            denoised_for_equation
        };
        observed.commit(next, context.cancellation)?;
    }

    context.check()?;
    let sampling = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok((sampling, noise_before, noise_after))
}

fn validate_options(options: LcmOptions) -> Result<(), LcmError> {
    for (name, value) in [
        ("s_noise", options.noise_scale_start),
        (
            "s_noise_end",
            options
                .noise_scale_end
                .unwrap_or(options.noise_scale_start),
        ),
        (
            "noise_clip_std",
            options.noise_clip_standard_deviations,
        ),
    ] {
        if !value.is_finite() {
            return Err(LcmError::InvalidOption { name, value });
        }
    }
    Ok(())
}

fn prepare_noise(
    backend: &CpuBackend,
    raw_noise: Vec<f64>,
    noise_scale: f32,
    noise_clip_standard_deviations: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, LcmError> {
    let mut noise = backend.workspace_vec::<f32>(context, raw_noise.len())?;
    for (element, value) in raw_noise.into_iter().enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = value as f32;
        checked_finite(value, step, "raw noise", element)?;
        noise.try_push(value)?;
    }

    if noise_clip_standard_deviations > 0.0 {
        if noise.len() < 2 {
            return Err(LcmError::NoiseClipCardinality { step });
        }
        let count = noise.len() as f32;
        let mean = noise.iter().copied().sum::<f32>() / count;
        let squared_deviation = noise
            .iter()
            .copied()
            .map(|value| {
                let deviation = value - mean;
                deviation * deviation
            })
            .sum::<f32>();
        let standard_deviation = (squared_deviation / (count - 1.0)).sqrt();
        checked_finite(standard_deviation, step, "noise standard deviation", 0)?;
        let clip = noise_clip_standard_deviations * standard_deviation;
        checked_finite(clip, step, "noise clip", 0)?;
        for (element, value) in noise.iter_mut().enumerate() {
            if element.is_multiple_of(256) {
                context.check()?;
            }
            *value = value.clamp(-clip, clip);
        }
    }

    if noise_scale != 1.0 {
        for (element, value) in noise.iter_mut().enumerate() {
            if element.is_multiple_of(256) {
                context.check()?;
            }
            *value *= noise_scale;
            checked_finite(*value, step, "scaled noise", element)?;
        }
    }
    Ok(noise)
}

fn validate_finite_values(
    values: &[f32],
    step: usize,
    stage: &'static str,
) -> Result<(), LcmError> {
    for (element, value) in values.iter().copied().enumerate() {
        checked_finite(value, step, stage, element)?;
    }
    Ok(())
}

fn checked_finite(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<(), LcmError> {
    if !value.is_finite() {
        return Err(LcmError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(())
}
