use crate::{
    SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_2M_SAMPLER_ID: &str = "dpmpp_2m";
pub const DPMPP_2M_FEATURE_ID: &str = "COMFY-MODEL-0166";
pub const DPMPP_2M_SOURCE_ORDINAL: u16 = 17;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_2M_SAMPLER_ID,
    feature_id: DPMPP_2M_FEATURE_ID,
    source_ordinal: DPMPP_2M_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_2m_comfy_model_0166",
    stochastic: false,
};

#[derive(Debug, Error)]
pub enum Dpmpp2mSamplerError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("DPM-Solver++(2M) requires sampler identity `dpmpp_2m`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM-Solver++(2M) denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("DPM-Solver++(2M) denoiser descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error(
        "DPM-Solver++(2M) produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM-Solver++(2M) coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("DPM-Solver++(2M) arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("DPM-Solver++(2M) allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_2m<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, Dpmpp2mSamplerError>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, expected_profile)?;
    if plan.sampler().as_str() != DPMPP_2M_SAMPLER_ID {
        return Err(Dpmpp2mSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let step_count = usize::try_from(plan.steps())
        .map_err(|_| Dpmpp2mSamplerError::Overflow("sampling step count"))?;
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| Dpmpp2mSamplerError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let mut previous_denoised: Option<Tensor> = None;

    for step in 0..step_count {
        context.check()?;
        let sigma = *sigmas
            .get(step)
            .ok_or(Dpmpp2mSamplerError::Overflow("current sigma lookup"))?;
        let next_sigma = *sigmas
            .get(step + 1)
            .ok_or(Dpmpp2mSamplerError::Overflow("next sigma lookup"))?;
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| Dpmpp2mSamplerError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(Dpmpp2mSamplerError::DenoiserContract { step });
        }

        let observed = session.observe_step(
            &current,
            denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;
        let current_values = tensor_to_f32(backend, &current, context)?;
        let denoised_values = tensor_to_f32(backend, &denoised, context)?;
        validate_finite(&current_values, step, "latent")?;
        validate_finite(&denoised_values, step, "denoiser")?;
        let time = checked_coefficient(step, "time", -sigma.ln())?;

        let next = if next_sigma == 0.0 {
            denoised.clone()
        } else {
            let next_time_value = checked_coefficient(step, "next time", -next_sigma.ln())?;
            let step_size_value = checked_positive(step, "step size", next_time_value - time)?;
            let latent_ratio_value = checked_coefficient(
                step,
                "latent ratio",
                (-next_time_value).exp() / (-time).exp(),
            )?;
            if latent_ratio_value < 0.0 {
                return Err(Dpmpp2mSamplerError::InvalidCoefficient {
                    step,
                    coefficient: "latent ratio",
                    value: latent_ratio_value,
                });
            }

            let (current_weight, previous_weight) = if previous_denoised.is_some() {
                    let previous_sigma = *sigmas
                        .get(
                            step.checked_sub(1)
                                .ok_or(Dpmpp2mSamplerError::Overflow("previous sigma index"))?,
                        )
                        .ok_or(Dpmpp2mSamplerError::Overflow("previous sigma lookup"))?;
                    let previous_time =
                        checked_coefficient(step, "previous time", -previous_sigma.ln())?;
                    let previous_step_size_value =
                        checked_positive(step, "previous step size", time - previous_time)?;
                    let step_ratio_value = checked_positive(
                        step,
                        "step ratio",
                        previous_step_size_value / step_size_value,
                    )?;
                    let inverse_double_ratio = checked_coefficient(
                        step,
                        "inverse double step ratio",
                        1.0 / (2.0 * step_ratio_value),
                    )?;
                    (
                        checked_coefficient(
                            step,
                            "current denoised weight",
                            1.0 + inverse_double_ratio,
                        )?,
                        checked_coefficient(
                            step,
                            "previous denoised weight",
                            -inverse_double_ratio,
                        )?,
                    )
                } else {
                    (1.0, 0.0)
                };

            let previous_values = if let Some(previous) = previous_denoised.as_ref() {
                Some(tensor_to_f32(backend, previous, context)?)
            } else {
                None
            };
            let mut transformed_values =
                backend.workspace_vec::<f32>(context, denoised_values.len())?;
            for (element, denoised_value) in denoised_values.iter().copied().enumerate() {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                let previous_value = previous_values
                    .as_ref()
                    .and_then(|values| values.get(element))
                    .copied()
                    .unwrap_or(0.0);
                let transformed =
                    current_weight * denoised_value + previous_weight * previous_value;
                if !transformed.is_finite() {
                    return Err(Dpmpp2mSamplerError::NonFinite {
                        step,
                        stage: "transformed denoiser",
                        element,
                    });
                }
                transformed_values.try_push(transformed)?;
            }
            let denoised_scale = (-step_size_value).exp_m1();
            let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
            for (element, (current_value, transformed_value)) in current_values
                .iter()
                .copied()
                .zip(transformed_values.iter().copied())
                .enumerate()
            {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                let next_value =
                    latent_ratio_value * current_value - denoised_scale * transformed_value;
                if !next_value.is_finite() {
                    return Err(Dpmpp2mSamplerError::NonFinite {
                        step,
                        stage: "next latent",
                        element,
                    });
                }
                next_values.try_push(next_value)?;
            }
            tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?
        };

        observed.commit(next, context.cancellation)?;
        previous_denoised = Some(denoised);
    }

    session.finish().map_err(Dpmpp2mSamplerError::from)
}

fn validate_finite(
    values: &[f32],
    step: usize,
    stage: &'static str,
) -> Result<(), Dpmpp2mSamplerError> {
    for (element, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(Dpmpp2mSamplerError::NonFinite {
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
) -> Result<f32, Dpmpp2mSamplerError> {
    if !value.is_finite() {
        return Err(Dpmpp2mSamplerError::InvalidCoefficient {
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
) -> Result<f32, Dpmpp2mSamplerError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(Dpmpp2mSamplerError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
