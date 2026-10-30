use crate::{
    CfgPpDenoiserOutput, SamplerRegistry, SamplingError, SamplingPlan, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const EULER_CFG_PP_ALTERNATIVE_SAMPLER_ID: &str = "euler_cfg_pp";

#[derive(Debug, Error)]
pub enum EulerCfgPpAlternativeError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("alternative Euler CFG++ requires sampler identity `euler_cfg_pp`, got {0:?}")]
    WrongSampler(String),
    #[error("alternative Euler CFG++ denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("alternative Euler CFG++ denoiser output descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("alternative Euler CFG++ produced a non-finite value at step {step}, element {element}")]
    NonFinite { step: usize, element: usize },
    #[error("alternative Euler CFG++ allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_euler_cfg_pp_alternative<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<CfgPpDenoiserOutput, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, EulerCfgPpAlternativeError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        plan.profile(),
    )?;
    if plan.sampler().as_str() != EULER_CFG_PP_ALTERNATIVE_SAMPLER_ID {
        return Err(EulerCfgPpAlternativeError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| EulerCfgPpAlternativeError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;

    for (step, transition) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = transition[0];
        let next_sigma = transition[1];
        let current = session.current().clone();
        let output = denoiser(&current, sigma, step)
            .map_err(|reason| EulerCfgPpAlternativeError::Denoiser { step, reason })?;
        if current.descriptor() != output.denoised.descriptor()
            || current.descriptor() != output.unconditional_denoised.descriptor()
        {
            return Err(EulerCfgPpAlternativeError::DenoiserContract { step });
        }
        let observed = session.observe_step(
            &current,
            output.denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;
        let current_values = tensor_to_f32(backend, &current, context)?;
        let guided_values = tensor_to_f32(backend, &output.denoised, context)?;
        let unconditional_values =
            tensor_to_f32(backend, &output.unconditional_denoised, context)?;
        let time_delta = next_sigma - sigma;
        let mut values = backend.workspace_vec::<f32>(context, current_values.len())?;
        for (element, ((current_value, guided), unconditional)) in current_values
            .iter()
            .copied()
            .zip(guided_values.iter().copied())
            .zip(unconditional_values.iter().copied())
            .enumerate()
        {
            if element.is_multiple_of(256) {
                context.check()?;
            }
            let derivative = (current_value - guided + unconditional - guided) / sigma;
            let value = current_value + derivative * time_delta;
            if !value.is_finite() {
                return Err(EulerCfgPpAlternativeError::NonFinite { step, element });
            }
            values.try_push(value)?;
        }
        let next = tensor_from_f32(
            backend,
            current.descriptor().shape(),
            &values,
            context,
        )?;
        observed.commit(next, context.cancellation)?;
    }
    Ok(session.finish()?)
}
