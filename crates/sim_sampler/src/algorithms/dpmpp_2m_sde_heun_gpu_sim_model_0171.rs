use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProgress, SamplingTrace, SchedulerError, SchedulerRegistry,
    generated_dpmpp_2m_sde_gpu_sim_model_0169::{
        DPMPP_2M_SDE_GPU_SAMPLER_ID, Dpmpp2mSdeGpuError, Dpmpp2mSdeGpuOptions,
        Dpmpp2mSdeSolverType, sample_dpmpp_2m_sde_gpu,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, RngCheckpoint, Tensor, TensorError};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_2M_SDE_HEUN_GPU_SAMPLER_ID: &str = "dpmpp_2m_sde_heun_gpu";
pub const DPMPP_2M_SDE_HEUN_GPU_FEATURE_ID: &str = "COMFY-MODEL-0171";
pub const DPMPP_2M_SDE_HEUN_GPU_SOURCE_ORDINAL: u16 = 22;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_2M_SDE_HEUN_GPU_SAMPLER_ID,
    feature_id: DPMPP_2M_SDE_HEUN_GPU_FEATURE_ID,
    source_ordinal: DPMPP_2M_SDE_HEUN_GPU_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_2m_sde_heun_gpu_comfy_model_0171",
    stochastic: true,
};

#[derive(Debug, Error)]
pub enum Dpmpp2mSdeHeunGpuError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    EquationFamily(#[from] Dpmpp2mSdeGpuError),
    #[error(
        "DPM-Solver++(2M) SDE Heun GPU requires sampler identity `dpmpp_2m_sde_heun_gpu`, got {0:?}"
    )]
    WrongSampler(String),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_2m_sde_heun_gpu<CallbackError>(
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
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), Dpmpp2mSdeHeunGpuError>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != DPMPP_2M_SDE_HEUN_GPU_SAMPLER_ID {
        return Err(Dpmpp2mSdeHeunGpuError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let equation_plan = SamplingPlan::new(
        DPMPP_2M_SDE_GPU_SAMPLER_ID,
        plan.scheduler().as_str(),
        plan.profile().clone(),
        plan.seed(),
        plan.steps(),
        plan.guidance(),
        plan.denoise(),
    )?;
    Ok(sample_dpmpp_2m_sde_gpu(
        backend,
        equation_plan,
        profile,
        initial,
        sigmas,
        noise_request,
        Dpmpp2mSdeGpuOptions {
            eta,
            noise_scale,
            solver_type: Dpmpp2mSdeSolverType::Heun,
        },
        context,
        denoiser,
        callback,
    )?)
}
