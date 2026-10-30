use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProfileError, SamplingProgress, SamplingSession, SamplingTrace,
    SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    BackendCapabilityMatrix, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const ER_SDE_SAMPLER_ID: &str = "er_sde";
pub const ER_SDE_FEATURE_ID: &str = "COMFY-MODEL-0178";
pub const ER_SDE_SOURCE_ORDINAL: u16 = 36;
pub const ER_SDE_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

const INTEGRATION_POINT_COUNT: usize = 200;
const INTEGRATION_POINT_COUNT_F32: f32 = 200.0;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: ER_SDE_SAMPLER_ID,
    feature_id: ER_SDE_FEATURE_ID,
    source_ordinal: ER_SDE_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/er_sde_comfy_model_0178",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErSdeOptions {
    pub noise_scale: f32,
    pub max_stage: usize,
}

impl Default for ErSdeOptions {
    fn default() -> Self {
        Self {
            noise_scale: 1.0,
            max_stage: 3,
        }
    }
}

#[derive(Debug, Error)]
pub enum ErSdeError {
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
    #[error("ER-SDE requires sampler identity `er_sde`, got {0:?}")]
    WrongSampler(String),
    #[error("ER-SDE denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("ER-SDE denoiser descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("ER-SDE history {history} is unavailable at step {step}")]
    MissingHistory { step: usize, history: &'static str },
    #[error("ER-SDE coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("ER-SDE produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("ER-SDE allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("ER-SDE index overflow for {0}")]
    Overflow(&'static str),
    #[error("ER-SDE noise scaler failed at step {step} for input {input}: {reason}")]
    NoiseScaler {
        step: usize,
        input: f32,
        reason: String,
    },
    #[error("native ER-SDE noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
}

pub fn er_sde_rng_profile(device: DeviceId) -> (RngSeedTransform, RngGenerationPlacement) {
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

pub fn validate_er_sde_generation_device(device: DeviceId) -> Result<(), ErSdeError> {
    BackendCapabilityMatrix::for_native_device(device).map_err(|error| {
        ErSdeError::DeviceUnavailable {
            device,
            reason: error.reason().to_owned(),
        }
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sample_er_sde<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: ErSdeOptions,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), ErSdeError>
where
    CallbackError: Display,
{
    sample_er_sde_with_noise_scaler(
        backend,
        plan,
        profile,
        initial,
        sigmas,
        noise_request,
        options,
        context,
        denoiser,
        callback,
        |value| Ok(value * (value.powf(0.3).exp() + 10.0)),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sample_er_sde_with_noise_scaler<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: ErSdeOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
    mut noise_scaler: impl FnMut(f32) -> Result<f32, String>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), ErSdeError>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != ER_SDE_SAMPLER_ID {
        return Err(ErSdeError::WrongSampler(plan.sampler().as_str().to_owned()));
    }

    let effective_noise_scale = profile.scale_sampler_noise(options.noise_scale)?;
    let seed = plan.seed();
    let device = initial.descriptor().device();
    validate_er_sde_generation_device(device)?;
    let mut adjusted_sigmas = Vec::new();
    adjusted_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| ErSdeError::OutOfMemory("adjusted sigma schedule"))?;
    adjusted_sigmas.extend_from_slice(sigmas);
    profile.adjust_first_sigma_for_snr(&mut adjusted_sigmas)?;
    let er_lambdas = er_lambdas(profile, &adjusted_sigmas)?;
    let mut session = SamplingSession::new(plan, adjusted_sigmas.clone(), initial)?;
    let (seed_transform, generation_placement) = er_sde_rng_profile(device);
    let mut noise_transaction = noise_request.open_transaction(
        ER_SDE_NOISE_CONTRACT_ID,
        i128::from(seed),
        seed_transform,
        generation_placement,
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();
    let mut old_denoised: Option<Vec<f32>> = None;
    let mut old_denoised_derivative: Option<Vec<f32>> = None;

    for (step, sigma_pair) in adjusted_sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = sigma_pair[0];
        let next_sigma = sigma_pair[1];
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| ErSdeError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(ErSdeError::DenoiserContract { step });
        }
        let observed = session.observe_step(
            &current,
            denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;
        let denoised_values = tensor_to_f32(backend, &denoised, context)?;
        validate_finite(&denoised_values, step, "denoiser")?;

        let next = if next_sigma == 0.0 {
            denoised.clone()
        } else {
            let current_values = tensor_to_f32(backend, &current, context)?;
            validate_finite(&current_values, step, "latent")?;
            let lambda_source = *er_lambdas
                .get(step)
                .ok_or(ErSdeError::Overflow("source ER lambda lookup"))?;
            let next_index = step
                .checked_add(1)
                .ok_or(ErSdeError::Overflow("target ER lambda index"))?;
            let lambda_target = *er_lambdas
                .get(next_index)
                .ok_or(ErSdeError::Overflow("target ER lambda lookup"))?;
            let alpha_source = checked_positive(
                step,
                "source alpha",
                sigma / checked_positive(step, "source ER lambda", lambda_source)?,
            )?;
            let alpha_target = checked_positive(
                step,
                "target alpha",
                next_sigma / checked_positive(step, "target ER lambda", lambda_target)?,
            )?;
            let scaler_source = evaluate_noise_scaler(&mut noise_scaler, lambda_source, step)?;
            let scaler_target = evaluate_noise_scaler(&mut noise_scaler, lambda_target, step)?;
            let scaler_ratio =
                checked_nonzero(step, "noise scaler ratio", scaler_target / scaler_source)?;
            let alpha_ratio = checked_positive(step, "alpha ratio", alpha_target / alpha_source)?;
            let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
            for (element, (current_value, denoised_value)) in current_values
                .iter()
                .copied()
                .zip(denoised_values.iter().copied())
                .enumerate()
            {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                let value = alpha_ratio * scaler_ratio * current_value
                    + alpha_target * (1.0 - scaler_ratio) * denoised_value;
                checked_value(value, step, "stage one", element)?;
                next_values.try_push(value)?;
            }

            let stage_used = options.max_stage.min(
                step.checked_add(1)
                    .ok_or(ErSdeError::Overflow("stage number"))?,
            );
            let mut denoised_derivative = None;
            if stage_used >= 2 {
                let delta_lambda =
                    checked_nonzero(step, "ER lambda delta", lambda_target - lambda_source)?;
                let lambda_step_size = checked_positive(
                    step,
                    "integration step size",
                    -delta_lambda / INTEGRATION_POINT_COUNT_F32,
                )?;
                let (integration_s, integration_s_u) = integrate_noise_scaler(
                    lambda_source,
                    lambda_target,
                    lambda_step_size,
                    stage_used >= 3,
                    step,
                    context,
                    &mut noise_scaler,
                )?;
                let previous_denoised =
                    old_denoised.as_deref().ok_or(ErSdeError::MissingHistory {
                        step,
                        history: "previous denoiser",
                    })?;
                let previous_lambda_index = step
                    .checked_sub(1)
                    .ok_or(ErSdeError::Overflow("previous ER lambda index"))?;
                let previous_lambda = *er_lambdas
                    .get(previous_lambda_index)
                    .ok_or(ErSdeError::Overflow("previous ER lambda lookup"))?;
                let derivative_denominator = checked_nonzero(
                    step,
                    "denoiser derivative denominator",
                    lambda_source - previous_lambda,
                )?;
                let derivative = differences(
                    &denoised_values,
                    previous_denoised,
                    derivative_denominator,
                    step,
                    "stage two derivative",
                    context,
                )?;
                let stage_two_coefficient = checked_finite(
                    step,
                    "stage two coefficient",
                    alpha_target * (delta_lambda + integration_s * scaler_target),
                )?;
                add_scaled(
                    &mut next_values,
                    &derivative,
                    stage_two_coefficient,
                    step,
                    "stage two",
                    context,
                )?;

                if stage_used >= 3 {
                    let previous_derivative =
                        old_denoised_derivative
                            .as_deref()
                            .ok_or(ErSdeError::MissingHistory {
                                step,
                                history: "previous denoiser derivative",
                            })?;
                    let previous_two_index = step
                        .checked_sub(2)
                        .ok_or(ErSdeError::Overflow("two-step ER lambda index"))?;
                    let previous_two_lambda = *er_lambdas
                        .get(previous_two_index)
                        .ok_or(ErSdeError::Overflow("two-step ER lambda lookup"))?;
                    let derivative_u_denominator = checked_nonzero(
                        step,
                        "stage three derivative denominator",
                        (lambda_source - previous_two_lambda) / 2.0,
                    )?;
                    let derivative_u = differences(
                        &derivative,
                        previous_derivative,
                        derivative_u_denominator,
                        step,
                        "stage three derivative",
                        context,
                    )?;
                    let stage_three_coefficient = checked_finite(
                        step,
                        "stage three coefficient",
                        alpha_target
                            * (delta_lambda.powi(2) / 2.0 + integration_s_u * scaler_target),
                    )?;
                    add_scaled(
                        &mut next_values,
                        &derivative_u,
                        stage_three_coefficient,
                        step,
                        "stage three",
                        context,
                    )?;
                }
                denoised_derivative = Some(derivative);
            }

            if effective_noise_scale > 0.0 {
                let noise =
                    noise_transaction.draw_normal(next_values.len(), context.cancellation)?;
                let radicand = lambda_target.powi(2) - lambda_source.powi(2) * scaler_ratio.powi(2);
                let root = radicand.sqrt();
                let root = if root.is_nan() {
                    0.0
                } else {
                    checked_finite(step, "stochastic radicand root", root)?
                };
                let coefficient = checked_finite(
                    step,
                    "stochastic coefficient",
                    alpha_target * effective_noise_scale * root,
                )?;
                for (element, (value, noise_value)) in next_values.iter_mut().zip(noise).enumerate()
                {
                    if element.is_multiple_of(256) {
                        context.check()?;
                    }
                    *value += coefficient * noise_value as f32;
                    checked_value(*value, step, "stochastic update", element)?;
                }
            }
            if denoised_derivative.is_some() {
                old_denoised_derivative = denoised_derivative;
            }
            tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?
        };

        old_denoised = Some(copy_values(&denoised_values, "denoiser history")?);
        observed.commit(next, context.cancellation)?;
    }

    let trace = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok((trace, noise_before, noise_after))
}

fn er_lambdas(profile: &impl SamplingProfile, sigmas: &[f32]) -> Result<Vec<f32>, ErSdeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(sigmas.len())
        .map_err(|_| ErSdeError::OutOfMemory("ER lambda schedule"))?;
    for (step, sigma) in sigmas.iter().copied().enumerate() {
        let value = if sigma == 0.0 {
            0.0
        } else {
            checked_positive(step, "ER lambda", (-profile.half_log_snr(sigma)?).exp())?
        };
        values.push(value);
    }
    Ok(values)
}

fn evaluate_noise_scaler(
    noise_scaler: &mut impl FnMut(f32) -> Result<f32, String>,
    value: f32,
    step: usize,
) -> Result<f32, ErSdeError> {
    let scaled = noise_scaler(value).map_err(|reason| ErSdeError::NoiseScaler {
        step,
        input: value,
        reason,
    })?;
    checked_nonzero(step, "ER noise scaler", scaled)
}

fn integrate_noise_scaler(
    lambda_source: f32,
    lambda_target: f32,
    lambda_step_size: f32,
    stage_three: bool,
    step: usize,
    context: &ExecutionContext<'_>,
    noise_scaler: &mut impl FnMut(f32) -> Result<f32, String>,
) -> Result<(f32, f32), ErSdeError> {
    let mut inverse_sum = 0.0_f32;
    let mut weighted_sum = 0.0_f32;
    for point in 0..INTEGRATION_POINT_COUNT {
        if point.is_multiple_of(32) {
            context.check()?;
        }
        let position = checked_positive(
            step,
            "integration position",
            lambda_target + point as f32 * lambda_step_size,
        )?;
        let scaled = evaluate_noise_scaler(noise_scaler, position, step)?;
        inverse_sum = checked_finite(step, "integration inverse sum", inverse_sum + 1.0 / scaled)?;
        if stage_three {
            weighted_sum = checked_finite(
                step,
                "integration weighted sum",
                weighted_sum + (position - lambda_source) / scaled,
            )?;
        }
    }
    Ok((
        checked_finite(step, "integration s", inverse_sum * lambda_step_size)?,
        checked_finite(step, "integration s u", weighted_sum * lambda_step_size)?,
    ))
}

fn differences(
    current: &[f32],
    previous: &[f32],
    denominator: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, ErSdeError> {
    if current.len() != previous.len() {
        return Err(ErSdeError::MissingHistory {
            step,
            history: "shape-compatible derivative",
        });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(current.len())
        .map_err(|_| ErSdeError::OutOfMemory(stage))?;
    for (element, (current, previous)) in current
        .iter()
        .copied()
        .zip(previous.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        values.push(checked_value(
            (current - previous) / denominator,
            step,
            stage,
            element,
        )?);
    }
    Ok(values)
}

fn add_scaled(
    destination: &mut [f32],
    values: &[f32],
    coefficient: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), ErSdeError> {
    if destination.len() != values.len() {
        return Err(ErSdeError::MissingHistory {
            step,
            history: "shape-compatible correction",
        });
    }
    for (element, (destination, value)) in destination.iter_mut().zip(values).enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        *destination += coefficient * value;
        checked_value(*destination, step, stage, element)?;
    }
    Ok(())
}

fn copy_values(values: &[f32], allocation: &'static str) -> Result<Vec<f32>, ErSdeError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(values.len())
        .map_err(|_| ErSdeError::OutOfMemory(allocation))?;
    copy.extend_from_slice(values);
    Ok(copy)
}

fn validate_finite(values: &[f32], step: usize, stage: &'static str) -> Result<(), ErSdeError> {
    for (element, value) in values.iter().copied().enumerate() {
        checked_value(value, step, stage, element)?;
    }
    Ok(())
}

fn checked_value(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<f32, ErSdeError> {
    if !value.is_finite() {
        return Err(ErSdeError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(value)
}

fn checked_finite(step: usize, coefficient: &'static str, value: f32) -> Result<f32, ErSdeError> {
    if !value.is_finite() {
        return Err(ErSdeError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_positive(step: usize, coefficient: &'static str, value: f32) -> Result<f32, ErSdeError> {
    let value = checked_finite(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(ErSdeError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_nonzero(step: usize, coefficient: &'static str, value: f32) -> Result<f32, ErSdeError> {
    let value = checked_finite(step, coefficient, value)?;
    if value == 0.0 {
        return Err(ErSdeError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
