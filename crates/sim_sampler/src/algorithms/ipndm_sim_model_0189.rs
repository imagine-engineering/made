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

pub const IPNDM_SAMPLER_ID: &str = "ipndm";
pub const IPNDM_FEATURE_ID: &str = "COMFY-MODEL-0189";
pub const IPNDM_SOURCE_ORDINAL: u16 = 27;
pub const IPNDM_MAX_ORDER: usize = 4;
pub const IPNDM_HISTORY_CAPACITY: usize = IPNDM_MAX_ORDER - 1;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: IPNDM_SAMPLER_ID,
    feature_id: IPNDM_FEATURE_ID,
    source_ordinal: IPNDM_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/ipndm_comfy_model_0189",
    stochastic: false,
};

pub fn sample_ipndm<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, IpndmSamplerError>
where
    CallbackError: Display,
{
    check_cancelled(context, 0)?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != IPNDM_SAMPLER_ID {
        return Err(IpndmSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let step_count = sigmas.len().saturating_sub(1);
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("IPNDM sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let callback_latent = initial.clone();
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let mut derivative_history = Vec::new();
    derivative_history
        .try_reserve_exact(IPNDM_HISTORY_CAPACITY)
        .map_err(|_| IpndmSamplerError::OutOfMemory("derivative history"))?;

    for (step, pair) in sigmas.windows(2).enumerate() {
        check_cancelled(context, step)?;
        let sigma = pair
            .first()
            .copied()
            .ok_or(IpndmSamplerError::Overflow("current sigma lookup"))?;
        let next_sigma = pair
            .get(1)
            .copied()
            .ok_or(IpndmSamplerError::Overflow("next sigma lookup"))?;
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| IpndmSamplerError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(IpndmSamplerError::DenoiserContract { step });
        }

        let observed = session
            .observe_step(
                &callback_latent,
                denoised.clone(),
                context.cancellation,
                |progress, current, denoised| callback(progress, current, denoised),
            )
            .map_err(|error| map_sampling_error(error, step))?;

        let current_values = tensor_to_f32(backend, &current, context)?;
        let denoised_values = tensor_to_f32(backend, &denoised, context)?;
        let mut derivative_values = backend.workspace_vec::<f32>(context, current_values.len())?;
        for (element, (current_value, denoised_value)) in current_values
            .iter()
            .zip(denoised_values.iter())
            .enumerate()
        {
            if element.is_multiple_of(256) {
                check_cancelled(context, step)?;
            }
            checked_value(*denoised_value, step, "denoiser", element)?;
            let derivative = (*current_value - *denoised_value) / sigma;
            checked_value(derivative, step, "derivative", element)?;
            derivative_values.try_push(derivative)?;
        }
        let derivative = tensor_from_f32(
            backend,
            current.descriptor().shape(),
            &derivative_values,
            context,
        )?;

        let order = IPNDM_MAX_ORDER.min(step + 1);
        let next = if next_sigma == 0.0 {
            denoised
        } else {
            let previous_one = if order >= 2 {
                Some(history_values(
                    backend,
                    &derivative_history,
                    1,
                    step,
                    order,
                    context,
                )?)
            } else {
                None
            };
            let previous_two = if order >= 3 {
                Some(history_values(
                    backend,
                    &derivative_history,
                    2,
                    step,
                    order,
                    context,
                )?)
            } else {
                None
            };
            let previous_three = if order >= 4 {
                Some(history_values(
                    backend,
                    &derivative_history,
                    3,
                    step,
                    order,
                    context,
                )?)
            } else {
                None
            };
            let delta = next_sigma - sigma;
            let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
            for (element, (current_value, derivative_value)) in current_values
                .iter()
                .zip(derivative_values.iter())
                .enumerate()
            {
                if element.is_multiple_of(256) {
                    check_cancelled(context, step)?;
                }
                let effective_derivative = match order {
                    1 => *derivative_value,
                    2 => {
                        (3.0 * *derivative_value
                            - history_element(previous_one.as_deref(), element, step, order)?)
                            / 2.0
                    }
                    3 => {
                        (23.0 * *derivative_value
                            - 16.0
                                * history_element(previous_one.as_deref(), element, step, order)?
                            + 5.0 * history_element(previous_two.as_deref(), element, step, order)?)
                            / 12.0
                    }
                    4 => {
                        (55.0 * *derivative_value
                            - 59.0
                                * history_element(previous_one.as_deref(), element, step, order)?
                            + 37.0
                                * history_element(previous_two.as_deref(), element, step, order)?
                            - 9.0
                                * history_element(previous_three.as_deref(), element, step, order)?)
                            / 24.0
                    }
                    _ => {
                        return Err(IpndmSamplerError::Overflow("Adams-Bashforth order"));
                    }
                };
                checked_value(effective_derivative, step, "effective derivative", element)?;
                let next_value = *current_value + delta * effective_derivative;
                checked_value(next_value, step, "latent update", element)?;
                next_values.try_push(next_value)?;
            }
            tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?
        };

        observed
            .commit(next, context.cancellation)
            .map_err(|error| map_sampling_error(error, step))?;
        update_history(&mut derivative_history, derivative, step)?;
    }

    if session.next_step() != step_count {
        return Err(IpndmSamplerError::Overflow("sampling step traversal"));
    }
    session.finish().map_err(IpndmSamplerError::Sampling)
}

fn history_values(
    backend: &CpuBackend,
    history: &[Tensor],
    distance: usize,
    step: usize,
    order: usize,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, IpndmSamplerError> {
    let index = history
        .len()
        .checked_sub(distance)
        .ok_or(IpndmSamplerError::MissingHistory { step, order })?;
    let tensor = history
        .get(index)
        .ok_or(IpndmSamplerError::MissingHistory { step, order })?;
    tensor_to_f32(backend, tensor, context).map_err(IpndmSamplerError::TensorKernel)
}

fn history_element(
    values: Option<&[f32]>,
    element: usize,
    step: usize,
    order: usize,
) -> Result<f32, IpndmSamplerError> {
    values
        .and_then(|values| values.get(element))
        .copied()
        .ok_or(IpndmSamplerError::MissingHistory { step, order })
}

fn update_history(
    history: &mut Vec<Tensor>,
    derivative: Tensor,
    step: usize,
) -> Result<(), IpndmSamplerError> {
    if history.len() == IPNDM_HISTORY_CAPACITY {
        history.rotate_left(1);
        let newest = history
            .last_mut()
            .ok_or(IpndmSamplerError::MissingHistory {
                step,
                order: IPNDM_MAX_ORDER,
            })?;
        *newest = derivative;
    } else {
        history.push(derivative);
    }
    Ok(())
}

fn checked_value(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<(), IpndmSamplerError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IpndmSamplerError::NonFinite {
            step,
            stage,
            element,
        })
    }
}

fn check_cancelled(context: &ExecutionContext<'_>, step: usize) -> Result<(), IpndmSamplerError> {
    context
        .cancellation
        .check()
        .map_err(|_| IpndmSamplerError::Cancelled { step })
}

fn map_sampling_error(error: SamplingError, step: usize) -> IpndmSamplerError {
    match error {
        SamplingError::Cancelled => IpndmSamplerError::Cancelled { step },
        error => IpndmSamplerError::Sampling(error),
    }
}

#[derive(Debug, Error)]
pub enum IpndmSamplerError {
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error("IPNDM requires sampler identity `ipndm`, got {0:?}")]
    WrongSampler(String),
    #[error("IPNDM denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("IPNDM denoiser output descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("IPNDM produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("IPNDM derivative history is unavailable at step {step} for order {order}")]
    MissingHistory { step: usize, order: usize },
    #[error("IPNDM allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("IPNDM arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("IPNDM sampling was cancelled at step {step}")]
    Cancelled { step: usize },
}
