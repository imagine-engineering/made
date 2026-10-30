use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProgress, SamplingTrace, SchedulerError, SchedulerRegistry,
    generated_dpmpp_2m_sde_sim_model_0168::{
        DPMPP_2M_SDE_SAMPLER_ID, Dpmpp2mSdeOptions, Dpmpp2mSdeSamplerError, Dpmpp2mSdeSolverType,
        sample_dpmpp_2m_sde,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, RngCheckpoint, Tensor, TensorError};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_2M_SDE_HEUN_SAMPLER_ID: &str = "dpmpp_2m_sde_heun";
pub const DPMPP_2M_SDE_HEUN_FEATURE_ID: &str = "COMFY-MODEL-0170";
pub const DPMPP_2M_SDE_HEUN_SOURCE_ORDINAL: u16 = 21;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_2M_SDE_HEUN_SAMPLER_ID,
    feature_id: DPMPP_2M_SDE_HEUN_FEATURE_ID,
    source_ordinal: DPMPP_2M_SDE_HEUN_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_2m_sde_heun_comfy_model_0170",
    stochastic: true,
};

#[derive(Debug, Error)]
pub enum Dpmpp2mSdeHeunError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    EquationFamily(#[from] Dpmpp2mSdeSamplerError),
    #[error("DPM-Solver++(2M) SDE Heun requires sampler identity `dpmpp_2m_sde_heun`, got {0:?}")]
    WrongSampler(String),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_2m_sde_heun<CallbackError>(
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
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), Dpmpp2mSdeHeunError>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != DPMPP_2M_SDE_HEUN_SAMPLER_ID {
        return Err(Dpmpp2mSdeHeunError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let equation_plan = SamplingPlan::new(
        DPMPP_2M_SDE_SAMPLER_ID,
        plan.scheduler().as_str(),
        plan.profile().clone(),
        plan.seed(),
        plan.steps(),
        plan.guidance(),
        plan.denoise(),
    )?;
    let family_options = if sigmas.len() <= 1 {
        Dpmpp2mSdeOptions::new_with_solver_type(1.0, 1.0, Dpmpp2mSdeSolverType::Heun)?
    } else {
        Dpmpp2mSdeOptions::new_with_solver_type(eta, noise_scale, Dpmpp2mSdeSolverType::Heun)?
    };
    Ok(sample_dpmpp_2m_sde(
        backend,
        equation_plan,
        profile,
        initial,
        sigmas,
        family_options,
        noise_request,
        context,
        denoiser,
        callback,
    )?)
}
