use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProgress, SamplingTrace, SchedulerError, SchedulerRegistry,
    generated_dpmpp_3m_sde_sim_model_0174::{
        DPMPP_3M_SDE_SAMPLER_ID, Dpmpp3mSdeError, Dpmpp3mSdeOptions,
        sample_dpmpp_3m_sde_with_generation_placement,
    },
};
use sim_tensor::{
    BackendCapabilityMatrix, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngGenerationPlacement, Tensor, TensorError,
};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_3M_SDE_GPU_SAMPLER_ID: &str = "dpmpp_3m_sde_gpu";
pub const DPMPP_3M_SDE_GPU_FEATURE_ID: &str = "COMFY-MODEL-0175";
pub const DPMPP_3M_SDE_GPU_SOURCE_ORDINAL: u16 = 24;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_3M_SDE_GPU_SAMPLER_ID,
    feature_id: DPMPP_3M_SDE_GPU_FEATURE_ID,
    source_ordinal: DPMPP_3M_SDE_GPU_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_3m_sde_gpu_comfy_model_0175",
    stochastic: true,
};

#[derive(Debug, Error)]
pub enum Dpmpp3mSdeGpuError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    EquationFamily(#[from] Dpmpp3mSdeError),
    #[error("DPM-Solver++(3M) SDE GPU requires sampler identity `dpmpp_3m_sde_gpu`, got {0:?}")]
    WrongSampler(String),
    #[error("native Brownian generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
}

pub fn validate_dpmpp_3m_sde_gpu_generation_device(
    device: DeviceId,
) -> Result<(), Dpmpp3mSdeGpuError> {
    BackendCapabilityMatrix::for_native_device(device).map_err(|error| {
        Dpmpp3mSdeGpuError::DeviceUnavailable {
            device,
            reason: error.reason().to_owned(),
        }
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_3m_sde_gpu<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    eta: f32,
    noise_scale: f32,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), Dpmpp3mSdeGpuError>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != DPMPP_3M_SDE_GPU_SAMPLER_ID {
        return Err(Dpmpp3mSdeGpuError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let device = initial.descriptor().device();
    if sigmas.len() > 1 {
        validate_dpmpp_3m_sde_gpu_generation_device(device)?;
    }
    let equation_plan = SamplingPlan::new(
        DPMPP_3M_SDE_SAMPLER_ID,
        plan.scheduler().as_str(),
        plan.profile().clone(),
        plan.seed(),
        plan.steps(),
        plan.guidance(),
        plan.denoise(),
    )?;
    let family_options = if sigmas.len() <= 1 {
        Dpmpp3mSdeOptions::source_defaults()
    } else {
        Dpmpp3mSdeOptions::new(eta, noise_scale)?
    };
    Ok(sample_dpmpp_3m_sde_with_generation_placement(
        backend,
        equation_plan,
        profile,
        initial,
        sigmas,
        family_options,
        noise_request,
        RngGenerationPlacement::Native(device),
        context,
        denoiser,
        callback,
    )?)
}
