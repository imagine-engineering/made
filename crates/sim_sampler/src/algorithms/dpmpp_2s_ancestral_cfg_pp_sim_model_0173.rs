use crate::{
    CfgPpDenoiserOutput, CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry,
    SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry, standard_ancestral_step,
    validate_cfg_pp_denoiser_output,
};
use sim_tensor::{
    CpuBackend, DeviceId, ExecutionContext, RngCheckpoint, RngCompatibilityError,
    RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_2S_ANCESTRAL_CFG_PP_SAMPLER_ID: &str = "dpmpp_2s_ancestral_cfg_pp";
pub const DPMPP_2S_ANCESTRAL_CFG_PP_FEATURE_ID: &str = "COMFY-MODEL-0173";
pub const DPMPP_2S_ANCESTRAL_CFG_PP_SOURCE_ORDINAL: u16 = 14;
pub const DPMPP_2S_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_2S_ANCESTRAL_CFG_PP_SAMPLER_ID,
    feature_id: DPMPP_2S_ANCESTRAL_CFG_PP_FEATURE_ID,
    source_ordinal: DPMPP_2S_ANCESTRAL_CFG_PP_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_2s_ancestral_cfg_pp_comfy_model_0173",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dpmpp2sAncestralCfgPpDenoiserStage {
    Primary,
    Midpoint,
}

pub type Dpmpp2sAncestralCfgPpDenoiserOutput = CfgPpDenoiserOutput;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dpmpp2sAncestralCfgPpOptions {
    pub eta: f32,
    pub noise_scale: f32,
}

impl Default for Dpmpp2sAncestralCfgPpOptions {
    fn default() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
        }
    }
}

#[derive(Debug, Error)]
pub enum Dpmpp2sAncestralCfgPpError {
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
    #[error(
        "DPM-Solver++(2S) ancestral CFG++ requires sampler identity `dpmpp_2s_ancestral_cfg_pp`, got {0:?}"
    )]
    WrongSampler(String),
    #[error("DPM-Solver++(2S) ancestral CFG++ option {name} must be finite, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error(
        "DPM-Solver++(2S) ancestral CFG++ denoiser failed at step {step} during {stage:?}: {reason}"
    )]
    Denoiser {
        step: usize,
        stage: Dpmpp2sAncestralCfgPpDenoiserStage,
        reason: String,
    },
    #[error(
        "DPM-Solver++(2S) ancestral CFG++ {output} descriptor changed at step {step} during {stage:?}"
    )]
    DenoiserContract {
        step: usize,
        stage: Dpmpp2sAncestralCfgPpDenoiserStage,
        output: &'static str,
    },
    #[error(
        "DPM-Solver++(2S) ancestral CFG++ coefficient {coefficient} is invalid at step {step}: {value}"
    )]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error(
        "DPM-Solver++(2S) ancestral CFG++ produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM-Solver++(2S) ancestral CFG++ allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_2s_ancestral_cfg_pp<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: Dpmpp2sAncestralCfgPpOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(
        &Tensor,
        f32,
        usize,
        Dpmpp2sAncestralCfgPpDenoiserStage,
    ) -> Result<Dpmpp2sAncestralCfgPpDenoiserOutput, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), Dpmpp2sAncestralCfgPpError>
where
    CallbackError: Display,
{
    context.check()?;
    validate_options(options)?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != DPMPP_2S_ANCESTRAL_CFG_PP_SAMPLER_ID {
        return Err(Dpmpp2sAncestralCfgPpError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let seed = plan.seed();
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| Dpmpp2sAncestralCfgPpError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let effective_noise_scale = profile.scale_sampler_noise(options.noise_scale)?;
    let mut noise_transaction = noise_request.open_transaction(
        DPMPP_2S_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();

    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = pair[0];
        let next_sigma = pair[1];
        let current = session.current().clone();
        let primary = call_denoiser(
            &mut denoiser,
            &current,
            sigma,
            step,
            Dpmpp2sAncestralCfgPpDenoiserStage::Primary,
        )?;
        validate_output_contract(
            &current,
            &primary,
            step,
            Dpmpp2sAncestralCfgPpDenoiserStage::Primary,
        )?;
        let (sigma_down, sigma_up) =
            standard_ancestral_step(sigma, next_sigma, options.eta).map_err(|_| {
                Dpmpp2sAncestralCfgPpError::InvalidCoefficient {
                    step,
                    coefficient: "ancestral step",
                    value: f32::NAN,
                }
            })?;
        let observed = session.observe_step(
            &current,
            primary.denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;

        let current_values = tensor_to_f32(backend, &current, context)?;
        let denoised_values = tensor_to_f32(backend, &primary.denoised, context)?;
        let unconditional_values =
            tensor_to_f32(backend, &primary.unconditional_denoised, context)?;
        validate_finite(&current_values, step, "latent")?;
        validate_finite(&denoised_values, step, "guided denoiser")?;
        validate_finite(&unconditional_values, step, "unconditional denoiser")?;

        let mut next_values = if sigma_down == 0.0 {
            terminal_update(
                backend,
                &current_values,
                &denoised_values,
                &unconditional_values,
                sigma,
                sigma_down,
                step,
                context,
            )?
        } else {
            let time = checked_finite(step, "time", -sigma.ln())?;
            let next_time = checked_finite(step, "next time", -sigma_down.ln())?;
            let step_size = checked_positive(step, "step size", next_time - time)?;
            let midpoint_time = checked_finite(step, "midpoint time", time + 0.5 * step_size)?;
            let midpoint_sigma = checked_positive(step, "midpoint sigma", (-midpoint_time).exp())?;
            let midpoint_values = second_order_input(
                backend,
                &current_values,
                &denoised_values,
                &unconditional_values,
                sigma,
                midpoint_sigma,
                -(-0.5 * step_size).exp_m1(),
                step,
                "midpoint latent",
                context,
            )?;
            let midpoint = tensor_from_f32(
                backend,
                current.descriptor().shape(),
                &midpoint_values,
                context,
            )?;
            let secondary = call_denoiser(
                &mut denoiser,
                &midpoint,
                midpoint_sigma,
                step,
                Dpmpp2sAncestralCfgPpDenoiserStage::Midpoint,
            )?;
            validate_output_contract(
                &current,
                &secondary,
                step,
                Dpmpp2sAncestralCfgPpDenoiserStage::Midpoint,
            )?;
            let secondary_values = tensor_to_f32(backend, &secondary.denoised, context)?;
            let secondary_unconditional_values =
                tensor_to_f32(backend, &secondary.unconditional_denoised, context)?;
            validate_finite(&secondary_values, step, "midpoint guided denoiser")?;
            validate_finite(
                &secondary_unconditional_values,
                step,
                "midpoint unconditional denoiser",
            )?;
            second_order_output(
                backend,
                &current_values,
                &denoised_values,
                &unconditional_values,
                &secondary_values,
                sigma,
                sigma_down,
                -(-step_size).exp_m1(),
                step,
                context,
            )?
        };

        if next_sigma > 0.0 {
            let noise = noise_transaction.draw_normal(next_values.len(), context.cancellation)?;
            let scale = effective_noise_scale * sigma_up;
            for (element, (value, noise)) in next_values.iter_mut().zip(noise).enumerate() {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                *value += noise as f32 * scale;
                checked_value(*value, step, "stochastic CFG++ update", element)?;
            }
        }
        let next = tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?;
        observed.commit(next, context.cancellation)?;
    }

    let trace = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok((trace, noise_before, noise_after))
}

fn validate_options(
    options: Dpmpp2sAncestralCfgPpOptions,
) -> Result<(), Dpmpp2sAncestralCfgPpError> {
    for (name, value) in [("eta", options.eta), ("noise scale", options.noise_scale)] {
        if !value.is_finite() {
            return Err(Dpmpp2sAncestralCfgPpError::InvalidOption { name, value });
        }
    }
    Ok(())
}

fn call_denoiser(
    denoiser: &mut impl FnMut(
        &Tensor,
        f32,
        usize,
        Dpmpp2sAncestralCfgPpDenoiserStage,
    ) -> Result<Dpmpp2sAncestralCfgPpDenoiserOutput, String>,
    latent: &Tensor,
    sigma: f32,
    step: usize,
    stage: Dpmpp2sAncestralCfgPpDenoiserStage,
) -> Result<Dpmpp2sAncestralCfgPpDenoiserOutput, Dpmpp2sAncestralCfgPpError> {
    denoiser(latent, sigma, step, stage).map_err(|reason| Dpmpp2sAncestralCfgPpError::Denoiser {
        step,
        stage,
        reason,
    })
}

fn validate_output_contract(
    current: &Tensor,
    output: &Dpmpp2sAncestralCfgPpDenoiserOutput,
    step: usize,
    stage: Dpmpp2sAncestralCfgPpDenoiserStage,
) -> Result<(), Dpmpp2sAncestralCfgPpError> {
    validate_cfg_pp_denoiser_output(current, output).map_err(|error| {
        Dpmpp2sAncestralCfgPpError::DenoiserContract {
            step,
            stage,
            output: error.output,
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn terminal_update(
    backend: &CpuBackend,
    current: &[f32],
    denoised: &[f32],
    unconditional: &[f32],
    sigma: f32,
    sigma_down: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, Dpmpp2sAncestralCfgPpError> {
    let mut values = backend.workspace_vec::<f32>(context, current.len())?;
    for (element, ((current, denoised), unconditional)) in current
        .iter()
        .copied()
        .zip(denoised.iter().copied())
        .zip(unconditional.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let derivative = (current - unconditional) / sigma;
        let value = denoised + derivative * sigma_down;
        checked_value(value, step, "terminal CFG++ update", element)?;
        values.try_push(value)?;
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn second_order_input(
    backend: &CpuBackend,
    current: &[f32],
    denoised: &[f32],
    unconditional: &[f32],
    sigma: f32,
    target_sigma: f32,
    denoised_coefficient: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, Dpmpp2sAncestralCfgPpError> {
    let ratio = checked_nonnegative(step, "sigma ratio", target_sigma / sigma)?;
    let mut values = backend.workspace_vec::<f32>(context, current.len())?;
    for (element, ((current, denoised), unconditional)) in current
        .iter()
        .copied()
        .zip(denoised.iter().copied())
        .zip(unconditional.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let cfg_adjusted = current + (denoised - unconditional);
        let value = ratio * cfg_adjusted + denoised_coefficient * denoised;
        checked_value(value, step, stage, element)?;
        values.try_push(value)?;
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn second_order_output(
    backend: &CpuBackend,
    current: &[f32],
    denoised: &[f32],
    unconditional: &[f32],
    secondary_denoised: &[f32],
    sigma: f32,
    sigma_down: f32,
    secondary_coefficient: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, Dpmpp2sAncestralCfgPpError> {
    let ratio = checked_nonnegative(step, "down sigma ratio", sigma_down / sigma)?;
    let mut values = backend.workspace_vec::<f32>(context, current.len())?;
    for (element, (((current, denoised), unconditional), secondary_denoised)) in current
        .iter()
        .copied()
        .zip(denoised.iter().copied())
        .zip(unconditional.iter().copied())
        .zip(secondary_denoised.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let cfg_adjusted = current + (denoised - unconditional);
        let value = ratio * cfg_adjusted + secondary_coefficient * secondary_denoised;
        checked_value(value, step, "second-order CFG++ update", element)?;
        values.try_push(value)?;
    }
    Ok(values)
}

fn validate_finite(
    values: &[f32],
    step: usize,
    stage: &'static str,
) -> Result<(), Dpmpp2sAncestralCfgPpError> {
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
) -> Result<f32, Dpmpp2sAncestralCfgPpError> {
    if !value.is_finite() {
        return Err(Dpmpp2sAncestralCfgPpError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(value)
}

fn checked_finite(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, Dpmpp2sAncestralCfgPpError> {
    if !value.is_finite() {
        return Err(Dpmpp2sAncestralCfgPpError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_positive(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, Dpmpp2sAncestralCfgPpError> {
    let value = checked_finite(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(Dpmpp2sAncestralCfgPpError::InvalidCoefficient {
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
) -> Result<f32, Dpmpp2sAncestralCfgPpError> {
    let value = checked_finite(step, coefficient, value)?;
    if value < 0.0 {
        return Err(Dpmpp2sAncestralCfgPpError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
