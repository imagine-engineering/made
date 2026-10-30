use crate::{
    CfgPpDenoiserOutput, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProgress, SamplingSession, SamplingTrace, SchedulerError,
    SchedulerRegistry, validate_cfg_pp_denoiser_output,
};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const GRADIENT_ESTIMATION_SAMPLER_ID: &str = "gradient_estimation";
pub const GRADIENT_ESTIMATION_FEATURE_ID: &str = "COMFY-MODEL-0185";
pub const GRADIENT_ESTIMATION_SOURCE_ORDINAL: u16 = 34;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: GRADIENT_ESTIMATION_SAMPLER_ID,
    feature_id: GRADIENT_ESTIMATION_FEATURE_ID,
    source_ordinal: GRADIENT_ESTIMATION_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/gradient_estimation_comfy_model_0185",
    stochastic: false,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientEstimationOptions {
    pub gamma: f32,
}

impl GradientEstimationOptions {
    pub const fn source_defaults() -> Self {
        Self { gamma: 2.0 }
    }
}

impl Default for GradientEstimationOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Debug, Error)]
pub enum GradientEstimationError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("gradient estimation requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("gradient estimation option ge_gamma must be finite, got {0}")]
    InvalidGamma(f32),
    #[error("gradient estimation denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("gradient estimation {output} descriptor changed at step {step}")]
    DenoiserContract { step: usize, output: &'static str },
    #[error(
        "gradient estimation produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("gradient estimation allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("gradient estimation previous derivative is missing at step {0}")]
    MissingPreviousDerivative(usize),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_gradient_estimation<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    options: GradientEstimationOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, GradientEstimationError>
where
    CallbackError: Display,
{
    sample_gradient_estimation_family(
        backend,
        plan,
        GRADIENT_ESTIMATION_SAMPLER_ID,
        profile,
        initial,
        sigmas,
        options,
        false,
        context,
        move |latent, sigma, step| {
            let denoised = denoiser(latent, sigma, step)?;
            Ok(CfgPpDenoiserOutput {
                unconditional_denoised: denoised.clone(),
                denoised,
            })
        },
        callback,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_gradient_estimation_family<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_sampler: &'static str,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    options: GradientEstimationOptions,
    cfg_pp: bool,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<CfgPpDenoiserOutput, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, GradientEstimationError>
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
    if plan.sampler().as_str() != expected_sampler {
        return Err(GradientEstimationError::WrongSampler {
            expected: expected_sampler,
            actual: plan.sampler().as_str().to_owned(),
        });
    }

    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| GradientEstimationError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let mut previous_derivative = None;

    for (step, sigma_pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = sigma_pair[0];
        let next_sigma = sigma_pair[1];
        let delta_sigma = next_sigma - sigma;
        let current = session.current().clone();
        let output = denoiser(&current, sigma, step)
            .map_err(|reason| GradientEstimationError::Denoiser { step, reason })?;
        validate_cfg_pp_denoiser_output(&current, &output).map_err(|error| {
            GradientEstimationError::DenoiserContract {
                step,
                output: error.output,
            }
        })?;

        let current_values = tensor_to_f32(backend, &current, context)?;
        let guided_values = tensor_to_f32(backend, &output.denoised, context)?;
        validate_values(&current_values, step, "latent")?;
        validate_values(&guided_values, step, "guided denoiser")?;
        let unconditional_values = if cfg_pp {
            let values = tensor_to_f32(backend, &output.unconditional_denoised, context)?;
            validate_values(&values, step, "unconditional denoiser")?;
            Some(values)
        } else {
            None
        };
        let derivative_source_values: &[f32] = if let Some(values) = &unconditional_values {
            values
        } else {
            &guided_values
        };
        let mut derivative_values = backend.workspace_vec::<f32>(context, current_values.len())?;
        for (element, (current_value, denoised_value)) in current_values
            .iter()
            .copied()
            .zip(derivative_source_values.iter().copied())
            .enumerate()
        {
            if element.is_multiple_of(256) {
                context.check()?;
            }
            let derivative = (current_value - denoised_value) / sigma;
            checked_value(derivative, step, "derivative", element)?;
            derivative_values.try_push(derivative)?;
        }
        let derivative = tensor_from_f32(
            backend,
            current.descriptor().shape(),
            &derivative_values,
            context,
        )?;

        let observed = session.observe_step(
            &current,
            output.denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;
        let next = if next_sigma == 0.0 {
            output.denoised
        } else {
            gradient_estimation_update(
                backend,
                &current,
                &output.denoised,
                &derivative,
                previous_derivative.as_ref(),
                next_sigma,
                delta_sigma,
                options.gamma,
                cfg_pp,
                step,
                context,
            )?
        };
        observed.commit(next, context.cancellation)?;
        previous_derivative = Some(derivative);
    }

    context.check()?;
    Ok(session.finish()?)
}

#[allow(clippy::too_many_arguments)]
fn gradient_estimation_update(
    backend: &CpuBackend,
    current: &Tensor,
    guided: &Tensor,
    derivative: &Tensor,
    previous_derivative: Option<&Tensor>,
    next_sigma: f32,
    delta_sigma: f32,
    gamma: f32,
    cfg_pp: bool,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GradientEstimationError> {
    let current_values = tensor_to_f32(backend, current, context)?;
    let guided_values = tensor_to_f32(backend, guided, context)?;
    let derivative_values = tensor_to_f32(backend, derivative, context)?;
    let previous_values = match previous_derivative {
        Some(previous) => Some(tensor_to_f32(backend, previous, context)?),
        None if step == 0 => None,
        None => return Err(GradientEstimationError::MissingPreviousDerivative(step)),
    };
    let mut output_values = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, ((current_value, guided_value), derivative_value)) in current_values
        .iter()
        .copied()
        .zip(guided_values.iter().copied())
        .zip(derivative_values.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let mut value = if cfg_pp {
            guided_value + derivative_value * next_sigma
        } else {
            current_value + derivative_value * delta_sigma
        };
        if let Some(previous_values) = previous_values.as_ref() {
            let previous = previous_values.get(element).copied().ok_or(
                GradientEstimationError::NonFinite {
                    step,
                    stage: "previous derivative length",
                    element,
                },
            )?;
            let correction = (gamma - 1.0) * (derivative_value - previous) * delta_sigma;
            value += correction;
        }
        checked_value(value, step, "updated latent", element)?;
        output_values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &output_values,
        context,
    )?)
}

fn validate_options(options: GradientEstimationOptions) -> Result<(), GradientEstimationError> {
    if !options.gamma.is_finite() {
        return Err(GradientEstimationError::InvalidGamma(options.gamma));
    }
    Ok(())
}

fn validate_values(
    values: &[f32],
    step: usize,
    stage: &'static str,
) -> Result<(), GradientEstimationError> {
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
) -> Result<f32, GradientEstimationError> {
    if !value.is_finite() {
        return Err(GradientEstimationError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(value)
}
