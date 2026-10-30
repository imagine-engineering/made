use crate::{
    CfgPpDenoiserOutput, CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry,
    SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    generated_native_diffusion::validate_euler_noise_generation_device,
    standard_ancestral_step, validate_cfg_pp_denoiser_output,
};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const RES_MULTISTEP_SAMPLER_ID: &str = "res_multistep";
pub const RES_MULTISTEP_FEATURE_ID: &str = "COMFY-MODEL-0193";
pub const RES_MULTISTEP_SOURCE_ORDINAL: u16 = 30;
pub const RES_MULTISTEP_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: RES_MULTISTEP_SAMPLER_ID,
    feature_id: RES_MULTISTEP_FEATURE_ID,
    source_ordinal: RES_MULTISTEP_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/res_multistep_comfy_model_0193",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResMultistepOptions {
    noise_scale: f32,
}

impl ResMultistepOptions {
    pub fn new(noise_scale: f32) -> Result<Self, ResMultistepSamplerError> {
        if !noise_scale.is_finite() {
            return Err(ResMultistepSamplerError::InvalidOption {
                name: "s_noise",
                value: noise_scale,
            });
        }
        Ok(Self { noise_scale })
    }

    pub const fn source_defaults() -> Self {
        Self { noise_scale: 1.0 }
    }

    pub const fn noise_scale(self) -> f32 {
        self.noise_scale
    }
}

impl Default for ResMultistepOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResMultistepFamilyOptions {
    eta: f32,
    noise_scale: f32,
    cfg_pp: bool,
}

impl ResMultistepFamilyOptions {
    pub fn new(
        eta: f32,
        noise_scale: f32,
        cfg_pp: bool,
    ) -> Result<Self, ResMultistepSamplerError> {
        for (name, value) in [("eta", eta), ("s_noise", noise_scale)] {
            if !value.is_finite() {
                return Err(ResMultistepSamplerError::InvalidOption { name, value });
            }
        }
        Ok(Self {
            eta,
            noise_scale,
            cfg_pp,
        })
    }

    pub const fn eta(self) -> f32 {
        self.eta
    }

    pub const fn noise_scale(self) -> f32 {
        self.noise_scale
    }

    pub const fn cfg_pp(self) -> bool {
        self.cfg_pp
    }
}

#[derive(Debug, Error)]
pub enum ResMultistepSamplerError {
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
    #[error("RES multistep family requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("RES multistep option {name} must be finite, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("RES multistep denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("RES multistep denoiser output descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("RES multistep coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("RES multistep produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("RES multistep history is unavailable at step {step}")]
    MissingHistory { step: usize },
    #[error("RES multistep allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("native RES multistep noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
}

#[allow(clippy::too_many_arguments)]
pub fn sample_res_multistep<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: ResMultistepOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), ResMultistepSamplerError>
where
    CallbackError: Display,
{
    let family_options =
        ResMultistepFamilyOptions::new(0.0, options.noise_scale(), false)?;
    sample_res_multistep_family(
        backend,
        plan,
        profile,
        RES_MULTISTEP_SAMPLER_ID,
        initial,
        sigmas,
        noise_request,
        family_options,
        context,
        |current, sigma, step| {
            let denoised = denoiser(current, sigma, step)?;
            Ok(CfgPpDenoiserOutput {
                unconditional_denoised: denoised.clone(),
                denoised,
            })
        },
        |progress, current, denoised| callback(progress, current, denoised),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sample_res_multistep_family<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    expected_sampler: &'static str,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: ResMultistepFamilyOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<CfgPpDenoiserOutput, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), ResMultistepSamplerError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        profile.identity(),
    )?;
    if plan.sampler().as_str() != expected_sampler {
        return Err(ResMultistepSamplerError::WrongSampler {
            expected: expected_sampler,
            actual: plan.sampler().as_str().to_owned(),
        });
    }

    let effective_noise_scale = profile.scale_sampler_noise(options.noise_scale())?;
    let generation_device = initial.descriptor().device();
    validate_euler_noise_generation_device(generation_device).map_err(|error| {
        ResMultistepSamplerError::DeviceUnavailable {
            device: generation_device,
            reason: error.to_string(),
        }
    })?;
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| ResMultistepSamplerError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let seed = plan.seed();
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let (seed_transform, generation_placement) = default_noise_profile(generation_device);
    let mut noise_transaction = noise_request.open_transaction(
        RES_MULTISTEP_NOISE_CONTRACT_ID,
        i128::from(seed),
        seed_transform,
        generation_placement,
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();
    let mut previous_denoised: Option<Tensor> = None;
    let mut previous_sigma_down: Option<f32> = None;

    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = checked_positive(
            step,
            "sigma",
            pair.first()
                .copied()
                .ok_or(SamplingError::Overflow("RES multistep source sigma lookup"))?,
        )?;
        let next_sigma = checked_nonnegative(
            step,
            "next sigma",
            pair.get(1)
                .copied()
                .ok_or(SamplingError::Overflow("RES multistep target sigma lookup"))?,
        )?;
        let current = session.current().clone();
        let output = denoiser(&current, sigma, step)
            .map_err(|reason| ResMultistepSamplerError::Denoiser { step, reason })?;
        if validate_cfg_pp_denoiser_output(&current, &output).is_err() {
            return Err(ResMultistepSamplerError::DenoiserContract { step });
        }
        let (sigma_down, sigma_up) = standard_ancestral_step(sigma, next_sigma, options.eta())?;
        let current_values = tensor_to_f32(backend, &current, context)?;
        let denoised_values = tensor_to_f32(backend, &output.denoised, context)?;
        let unconditional_values = tensor_to_f32(backend, &output.unconditional_denoised, context)?;
        validate_finite(&current_values, step, "latent")?;
        validate_finite(&denoised_values, step, "denoiser")?;
        validate_finite(&unconditional_values, step, "unconditional denoiser")?;
        let observed = session.observe_step(
            &current,
            output.denoised.clone(),
            context.cancellation,
            |progress, current, denoised| callback(progress, current, denoised),
        )?;

        let deterministic = if sigma_down == 0.0 || previous_denoised.is_none() {
            euler_step(
                backend,
                &current,
                &output.denoised,
                &output.unconditional_denoised,
                sigma,
                sigma_down,
                options.cfg_pp(),
                step,
                context,
            )?
        } else {
            multistep(
                backend,
                &current,
                &output.denoised,
                &output.unconditional_denoised,
                previous_denoised
                    .as_ref()
                    .ok_or(ResMultistepSamplerError::MissingHistory { step })?,
                sigma,
                previous_sigma_down
                    .ok_or(ResMultistepSamplerError::MissingHistory { step })?,
                sigma_down,
                *sigmas
                    .get(step.saturating_sub(1))
                    .ok_or(ResMultistepSamplerError::MissingHistory { step })?,
                options.cfg_pp(),
                step,
                context,
            )?
        };

        let next = if next_sigma > 0.0 {
            add_source_noise(
                backend,
                &deterministic,
                effective_noise_scale,
                sigma_up,
                step,
                &mut noise_transaction,
                context,
            )?
        } else {
            deterministic
        };
        observed.commit(next, context.cancellation)?;
        previous_denoised = Some(if options.cfg_pp() {
            output.unconditional_denoised
        } else {
            output.denoised
        });
        previous_sigma_down = Some(sigma_down);
    }

    let trace = session.finish()?;
    Ok((trace, (noise_before, noise_transaction.commit())))
}

fn default_noise_profile(device: DeviceId) -> (RngSeedTransform, RngGenerationPlacement) {
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

#[allow(clippy::too_many_arguments)]
fn multistep(
    backend: &CpuBackend,
    current: &Tensor,
    denoised: &Tensor,
    unconditional_denoised: &Tensor,
    previous_denoised: &Tensor,
    sigma: f32,
    previous_sigma_down: f32,
    next_sigma: f32,
    previous_sigma: f32,
    cfg_pp: bool,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ResMultistepSamplerError> {
    let time = checked_coefficient(step, "time", -sigma.ln())?;
    let old_time = checked_coefficient(step, "old time", -previous_sigma_down.ln())?;
    let next_time = checked_coefficient(step, "next time", -next_sigma.ln())?;
    let previous_time = checked_coefficient(step, "previous time", -previous_sigma.ln())?;
    let step_size = checked_positive(step, "step size", next_time - time)?;
    let c2 = checked_nonzero(step, "c2", (previous_time - old_time) / step_size)?;
    let negative_step = -step_size;
    let phi1 = nan_to_num(negative_step.exp_m1() / negative_step);
    let phi2 = nan_to_num((phi1 - 1.0) / negative_step);
    let first_weight = nan_to_num(phi1 - phi2 / c2);
    let second_weight = nan_to_num(phi2 / c2);
    let latent_scale = checked_coefficient(step, "latent scale", (-step_size).exp())?;
    let current_values = tensor_to_f32(backend, current, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let unconditional_values = tensor_to_f32(backend, unconditional_denoised, context)?;
    let previous_values = tensor_to_f32(backend, previous_denoised, context)?;
    let mut values = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, (((current, denoised), unconditional), previous)) in current_values
        .iter()
        .zip(denoised_values.iter())
        .zip(unconditional_values.iter())
        .zip(previous_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let equation_denoised = if cfg_pp { unconditional } else { denoised };
        let denoised_mix = first_weight * equation_denoised + second_weight * previous;
        let equation_current = if cfg_pp {
            current + (denoised - unconditional)
        } else {
            *current
        };
        let value = latent_scale * equation_current + step_size * denoised_mix;
        checked_element(value, step, "multistep latent", element)?;
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &values,
        context,
    )?)
}

fn euler_step(
    backend: &CpuBackend,
    current: &Tensor,
    denoised: &Tensor,
    unconditional_denoised: &Tensor,
    sigma: f32,
    sigma_down: f32,
    cfg_pp: bool,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ResMultistepSamplerError> {
    let current_values = tensor_to_f32(backend, current, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let unconditional_values = tensor_to_f32(backend, unconditional_denoised, context)?;
    let delta = sigma_down - sigma;
    let mut values = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, ((current, denoised), unconditional)) in current_values
        .iter()
        .zip(denoised_values.iter())
        .zip(unconditional_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let derivative_source = if cfg_pp { unconditional } else { denoised };
        let derivative = (current - derivative_source) / sigma;
        let value = if cfg_pp {
            denoised + derivative * sigma_down
        } else {
            current + derivative * delta
        };
        checked_element(value, step, "Euler latent", element)?;
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &values,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn add_source_noise(
    backend: &CpuBackend,
    deterministic: &Tensor,
    noise_scale: f32,
    sigma_up: f32,
    step: usize,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ResMultistepSamplerError> {
    let count = usize::try_from(deterministic.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let noise = transaction.draw_normal(count, context.cancellation)?;
    let deterministic_values = tensor_to_f32(backend, deterministic, context)?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for (element, (deterministic, noise)) in
        deterministic_values.iter().zip(noise).enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = deterministic + (noise as f32 * noise_scale) * sigma_up;
        checked_element(value, step, "noise update", element)?;
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        deterministic.descriptor().shape(),
        &values,
        context,
    )?)
}

fn nan_to_num(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else if value == f32::INFINITY {
        f32::MAX
    } else if value == f32::NEG_INFINITY {
        f32::MIN
    } else {
        value
    }
}

fn checked_positive(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, ResMultistepSamplerError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(ResMultistepSamplerError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_nonnegative(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, ResMultistepSamplerError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value < 0.0 {
        return Err(ResMultistepSamplerError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_nonzero(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, ResMultistepSamplerError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value == 0.0 {
        return Err(ResMultistepSamplerError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_coefficient(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, ResMultistepSamplerError> {
    if !value.is_finite() {
        return Err(ResMultistepSamplerError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn validate_finite(
    values: &[f32],
    step: usize,
    stage: &'static str,
) -> Result<(), ResMultistepSamplerError> {
    for (element, value) in values.iter().enumerate() {
        checked_element(*value, step, stage, element)?;
    }
    Ok(())
}

fn checked_element(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<(), ResMultistepSamplerError> {
    if !value.is_finite() {
        return Err(ResMultistepSamplerError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(())
}
