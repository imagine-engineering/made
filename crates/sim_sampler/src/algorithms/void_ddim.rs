use crate::{
    SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    CpuBackend, DType, ExecutionContext, Tensor, TensorBackend, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const VOID_DDIM_SAMPLER_ID: &str = "void_ddim";
pub const VOID_DDIM_FEATURE_ID: &str = "COMFY-MODEL-0212";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: VOID_DDIM_SAMPLER_ID,
    feature_id: VOID_DDIM_FEATURE_ID,
    source_ordinal: 44,
    aliases: &[],
    implementation_module: "algorithms/void_ddim",
    stochastic: false,
};

#[derive(Debug, Error)]
pub enum VoidDdimError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("VOID DDIM requires sampler identity `void_ddim`, got {0:?}")]
    WrongSampler(String),
    #[error("VOID DDIM denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("VOID DDIM denoiser output descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("VOID DDIM produced a non-finite {stage} at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("VOID DDIM allocation failed for {0}")]
    OutOfMemory(&'static str),
}

pub fn sample_void_ddim<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    initial_noise: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, VoidDdimError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        plan.profile(),
    )?;
    if plan.sampler().as_str() != VOID_DDIM_SAMPLER_ID {
        return Err(VoidDdimError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    let initial_noise = if initial_noise.descriptor().dtype() == DType::F32 {
        initial_noise
    } else {
        backend
            .cast_tensor(&initial_noise, DType::F32, false, false, context)?
            .0
    };
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| VoidDdimError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial_noise)?;

    for (step, transition) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = transition[0];
        let next_sigma = transition[1];
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| VoidDdimError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(VoidDdimError::DenoiserContract { step });
        }
        let observed = session.observe_step(
            &current,
            denoised.clone(),
            context.cancellation,
            |progress, latent, output| callback(progress, latent, output),
        )?;
        let next = if next_sigma == 0.0 {
            denoised
        } else {
            let alpha = 1.0 / (1.0 + sigma * sigma);
            let next_alpha = 1.0 / (1.0 + next_sigma * next_sigma);
            let alpha_root = alpha.sqrt();
            let epsilon_root = (1.0 - alpha).sqrt();
            let next_alpha_root = next_alpha.sqrt();
            let next_epsilon_root = (1.0 - next_alpha).sqrt();
            let current_values = tensor_to_f32(backend, &current, context)?;
            let denoised_values = tensor_to_f32(backend, &denoised, context)?;
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
                let predicted_epsilon =
                    (current_value - alpha_root * denoised_value) / epsilon_root;
                let value = next_alpha_root * denoised_value
                    + next_epsilon_root * predicted_epsilon;
                if !value.is_finite() {
                    return Err(VoidDdimError::NonFinite {
                        step,
                        stage: "alpha-space update",
                        element,
                    });
                }
                next_values.try_push(value)?;
            }
            tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?
        };
        observed.commit(next, context.cancellation)?;
    }
    session.finish().map_err(VoidDdimError::Sampling)
}
