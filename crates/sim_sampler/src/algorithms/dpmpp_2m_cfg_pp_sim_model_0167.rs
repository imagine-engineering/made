use crate::{
    CfgPpDenoiserOutput, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProgress, SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    validate_cfg_pp_denoiser_output,
};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_2M_CFG_PP_SAMPLER_ID: &str = "dpmpp_2m_cfg_pp";
pub const DPMPP_2M_CFG_PP_FEATURE_ID: &str = "COMFY-MODEL-0167";
pub const DPMPP_2M_CFG_PP_SOURCE_ORDINAL: u16 = 18;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_2M_CFG_PP_SAMPLER_ID,
    feature_id: DPMPP_2M_CFG_PP_FEATURE_ID,
    source_ordinal: DPMPP_2M_CFG_PP_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_2m_cfg_pp_comfy_model_0167",
    stochastic: false,
};

pub type Dpmpp2mCfgPpDenoiserOutput = CfgPpDenoiserOutput;

#[derive(Debug, Error)]
pub enum Dpmpp2mCfgPpSamplerError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("DPM-Solver++(2M) CFG++ requires sampler identity `dpmpp_2m_cfg_pp`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM-Solver++(2M) CFG++ denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("DPM-Solver++(2M) CFG++ {output} descriptor changed at step {step}")]
    DenoiserContract { step: usize, output: &'static str },
    #[error(
        "DPM-Solver++(2M) CFG++ produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM-Solver++(2M) CFG++ coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("DPM-Solver++(2M) CFG++ arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("DPM-Solver++(2M) CFG++ allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_2m_cfg_pp<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Dpmpp2mCfgPpDenoiserOutput, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, Dpmpp2mCfgPpSamplerError>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, expected_profile)?;
    if plan.sampler().as_str() != DPMPP_2M_CFG_PP_SAMPLER_ID {
        return Err(Dpmpp2mCfgPpSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let step_count = usize::try_from(plan.steps())
        .map_err(|_| Dpmpp2mCfgPpSamplerError::Overflow("sampling step count"))?;
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| Dpmpp2mCfgPpSamplerError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let mut previous_unconditional: Option<Tensor> = None;

    for step in 0..step_count {
        context.check()?;
        let sigma = *sigmas
            .get(step)
            .ok_or(Dpmpp2mCfgPpSamplerError::Overflow("current sigma lookup"))?;
        let next_sigma = *sigmas
            .get(step + 1)
            .ok_or(Dpmpp2mCfgPpSamplerError::Overflow("next sigma lookup"))?;
        let current = session.current().clone();
        let output = denoiser(&current, sigma, step)
            .map_err(|reason| Dpmpp2mCfgPpSamplerError::Denoiser { step, reason })?;
        validate_cfg_pp_denoiser_output(&current, &output).map_err(|error| {
            Dpmpp2mCfgPpSamplerError::DenoiserContract {
                step,
                output: error.output,
            }
        })?;
        let Dpmpp2mCfgPpDenoiserOutput {
            denoised,
            unconditional_denoised,
        } = output;

        let observed = session.observe_step(
            &current,
            denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;
        let current_values = tensor_to_f32(backend, &current, context)?;
        let denoised_values = tensor_to_f32(backend, &denoised, context)?;
        let unconditional_values = tensor_to_f32(backend, &unconditional_denoised, context)?;
        validate_finite(&current_values, step, "latent")?;
        validate_finite(&denoised_values, step, "guided denoiser")?;
        validate_finite(&unconditional_values, step, "unconditional denoiser")?;

        let time = checked_coefficient(step, "time", -sigma.ln())?;
        let (step_size, latent_ratio) = if next_sigma == 0.0 {
            (None, 0.0)
        } else {
            let next_time = checked_coefficient(step, "next time", -next_sigma.ln())?;
            let step_size = checked_positive(step, "step size", next_time - time)?;
            let latent_ratio = checked_nonnegative(step, "latent ratio", (-step_size).exp())?;
            (Some(step_size), latent_ratio)
        };
        let unconditional_weight =
            checked_coefficient(step, "unconditional weight", -latent_ratio)?;

        let (history_weight, history_values) = if next_sigma != 0.0 {
            if let Some(previous_unconditional) = previous_unconditional.as_ref() {
                let previous_sigma = *sigmas
                    .get(
                        step.checked_sub(1)
                            .ok_or(Dpmpp2mCfgPpSamplerError::Overflow("previous sigma index"))?,
                    )
                    .ok_or(Dpmpp2mCfgPpSamplerError::Overflow("previous sigma lookup"))?;
                let previous_time =
                    checked_coefficient(step, "previous time", -previous_sigma.ln())?;
                let previous_step =
                    checked_positive(step, "previous step size", time - previous_time)?;
                let current_step = step_size
                    .ok_or(Dpmpp2mCfgPpSamplerError::Overflow("non-terminal step size"))?;
                let ratio = checked_positive(step, "step ratio", previous_step / current_step)?;
                let history_weight = checked_coefficient(
                    step,
                    "history weight",
                    -(-current_step).exp_m1() * (1.0 / (2.0 * ratio)),
                )?;

                let previous_values = tensor_to_f32(backend, previous_unconditional, context)?;
                validate_finite(&previous_values, step, "previous unconditional denoiser")?;
                let mut history_values =
                    backend.workspace_vec::<f32>(context, denoised_values.len())?;
                for (element, (denoised_value, previous_value)) in denoised_values
                    .iter()
                    .copied()
                    .zip(previous_values.iter().copied())
                    .enumerate()
                {
                    if element.is_multiple_of(256) {
                        context.check()?;
                    }
                    let value = denoised_value - previous_value;
                    if !value.is_finite() {
                        return Err(Dpmpp2mCfgPpSamplerError::NonFinite {
                            step,
                            stage: "guided/unconditional history delta",
                            element,
                        });
                    }
                    history_values.try_push(value)?;
                }
                (history_weight, Some(history_values))
            } else {
                (0.0, None)
            }
        } else {
            (0.0, None)
        };
        let mut denoised_mix_values =
            backend.workspace_vec::<f32>(context, unconditional_values.len())?;
        for (element, unconditional_value) in unconditional_values.iter().copied().enumerate() {
            if element.is_multiple_of(256) {
                context.check()?;
            }
            let unconditional_term = unconditional_weight * unconditional_value;
            let value = history_values
                .as_ref()
                .and_then(|values| values.get(element))
                .copied()
                .map(|history| unconditional_term + history_weight * history)
                .unwrap_or(unconditional_term);
            if !value.is_finite() {
                return Err(Dpmpp2mCfgPpSamplerError::NonFinite {
                    step,
                    stage: "denoised mix",
                    element,
                });
            }
            denoised_mix_values.try_push(value)?;
        }

        let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
        for (element, ((denoised_value, mix_value), current_value)) in denoised_values
            .iter()
            .copied()
            .zip(denoised_mix_values.iter().copied())
            .zip(current_values.iter().copied())
            .enumerate()
        {
            if element.is_multiple_of(256) {
                context.check()?;
            }
            let value = (denoised_value + mix_value) + latent_ratio * current_value;
            if !value.is_finite() {
                return Err(Dpmpp2mCfgPpSamplerError::NonFinite {
                    step,
                    stage: "next latent",
                    element,
                });
            }
            next_values.try_push(value)?;
        }
        let next = tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?;

        observed.commit(next, context.cancellation)?;
        previous_unconditional = Some(unconditional_denoised);
    }

    Ok(session.finish()?)
}

fn validate_finite(
    values: &[f32],
    step: usize,
    stage: &'static str,
) -> Result<(), Dpmpp2mCfgPpSamplerError> {
    for (element, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(Dpmpp2mCfgPpSamplerError::NonFinite {
                step,
                stage,
                element,
            });
        }
    }
    Ok(())
}

fn checked_coefficient(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, Dpmpp2mCfgPpSamplerError> {
    if !value.is_finite() {
        return Err(Dpmpp2mCfgPpSamplerError::InvalidCoefficient {
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
) -> Result<f32, Dpmpp2mCfgPpSamplerError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(Dpmpp2mCfgPpSamplerError::InvalidCoefficient {
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
) -> Result<f32, Dpmpp2mCfgPpSamplerError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value < 0.0 {
        return Err(Dpmpp2mCfgPpSamplerError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
