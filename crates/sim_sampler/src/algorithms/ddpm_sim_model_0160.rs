use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingSession, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    CpuBackend, DeviceId, ExecutionContext, RngCheckpoint, RngCompatibilityError,
    RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use thiserror::Error;

pub const DDPM_SAMPLER_ID: &str = "ddpm";
pub const DDPM_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DDPM_SAMPLER_ID,
    feature_id: "COMFY-MODEL-0160",
    source_ordinal: 25,
    aliases: &[],
    implementation_module: "algorithms/ddpm_comfy_model_0160",
    stochastic: true,
};

#[derive(Debug, Error)]
pub enum DdpmSamplerError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("DDPM requires sampler identity `ddpm`, got {0:?}")]
    WrongSampler(String),
    #[error("DDPM denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("DDPM produced a non-finite value at step {step}, element {element}")]
    NonFinite { step: usize, element: usize },
}

pub fn sample_ddpm<E>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&crate::SamplingProgress, &Tensor, &Tensor) -> Result<(), E>,
) -> Result<(crate::SamplingTrace, RngCheckpoint, RngCheckpoint), DdpmSamplerError>
where
    E: std::fmt::Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        plan.profile(),
    )?;
    if plan.sampler().as_str() != DDPM_SAMPLER_ID {
        return Err(DdpmSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    let seed = plan.seed();
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("DDPM sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let mut noise_transaction = noise_request.open_transaction(
        DDPM_NOISE_CONTRACT_ID,
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
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| DdpmSamplerError::Denoiser { step, reason })?;
        let noise = if next_sigma > 0.0 {
            let count = usize::try_from(current.descriptor().element_count()?)
                .map_err(|_| TensorError::ShapeOverflow)?;
            let normal = noise_transaction.draw_normal(count, context.cancellation)?;
            let mut values = backend.workspace_vec::<f32>(context, count)?;
            for (element, value) in normal.into_iter().enumerate() {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                values.try_push(value as f32)?;
            }
            Some(tensor_from_f32(
                backend,
                current.descriptor().shape(),
                &values,
                context,
            )?)
        } else {
            None
        };
        let next = ddpm_step(
            backend,
            &current,
            &denoised,
            noise.as_ref(),
            sigma,
            next_sigma,
            step,
            context,
        )?;
        session.commit_step(
            denoised,
            next,
            context.cancellation,
            |progress, denoised, _| callback(progress, &current, denoised),
        )?;
    }
    let sampling = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok((sampling, noise_before, noise_after))
}

pub fn ddpm_step(
    backend: &CpuBackend,
    current: &Tensor,
    denoised: &Tensor,
    stochastic_noise: Option<&Tensor>,
    sigma: f32,
    next_sigma: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, DdpmSamplerError> {
    let current_values = tensor_to_f32(backend, current, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let noise_values = stochastic_noise
        .map(|noise| tensor_to_f32(backend, noise, context))
        .transpose()?;

    let alpha_cumprod = 1.0 / (sigma * sigma + 1.0);
    let alpha_cumprod_previous = 1.0 / (next_sigma * next_sigma + 1.0);
    let alpha = alpha_cumprod / alpha_cumprod_previous;
    let inverse_alpha_root = (1.0 / alpha).sqrt();
    let normalized_scale = (1.0 + sigma * sigma).sqrt();
    let epsilon_scale = (1.0 - alpha) / (1.0 - alpha_cumprod).sqrt();
    let stochastic_scale = if next_sigma > 0.0 {
        Some(
            ((1.0 - alpha) * (1.0 - alpha_cumprod_previous)
                / (1.0 - alpha_cumprod))
                .sqrt(),
        )
    } else {
        None
    };
    let output_scale = if next_sigma != 0.0 {
        (1.0 + next_sigma * next_sigma).sqrt()
    } else {
        1.0
    };

    let mut output = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, (current_value, denoised_value)) in current_values
        .iter()
        .zip(denoised_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let epsilon = (current_value - denoised_value) / sigma;
        let normalized = current_value / normalized_scale;
        let mut value = inverse_alpha_root * (normalized - epsilon_scale * epsilon);
        if let Some(stochastic_scale) = stochastic_scale {
            let noise_value = noise_values
                .as_ref()
                .and_then(|values| values.get(element))
                .copied()
                .ok_or_else(|| SamplingError::TensorContract {
                    role: "DDPM stochastic noise",
                    expected: format!("{:?}", current.descriptor()),
                    actual: stochastic_noise
                        .map(|noise| format!("{:?}", noise.descriptor()))
                        .unwrap_or_else(|| "missing".to_owned()),
                })?;
            value += stochastic_scale * noise_value;
        }
        value *= output_scale;
        if !value.is_finite() {
            return Err(DdpmSamplerError::NonFinite { step, element });
        }
        output.try_push(value)?;
    }
    tensor_from_f32(backend, current.descriptor().shape(), &output, context)
        .map_err(DdpmSamplerError::TensorKernel)
}
