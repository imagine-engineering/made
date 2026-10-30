use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProgress, SamplingTrace, SchedulerError, SchedulerRegistry,
    generated_dpmpp_2m_sde_sim_model_0168::{
        DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID, DPMPP_2M_SDE_SAMPLER_ID, Dpmpp2mSdeOptions,
        Dpmpp2mSdeSamplerError, sample_dpmpp_2m_sde_with_generation_placement,
    },
};
pub use crate::generated_dpmpp_2m_sde_sim_model_0168::Dpmpp2mSdeSolverType;
use sim_tensor::{
    BackendCapabilityMatrix, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngGenerationPlacement, Tensor, TensorError,
};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_2M_SDE_GPU_SAMPLER_ID: &str = "dpmpp_2m_sde_gpu";
pub const DPMPP_2M_SDE_GPU_FEATURE_ID: &str = "COMFY-MODEL-0169";
pub const DPMPP_2M_SDE_GPU_SOURCE_ORDINAL: u16 = 20;
pub const DPMPP_2M_SDE_GPU_BROWNIAN_CONTRACT_ID: &str = DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_2M_SDE_GPU_SAMPLER_ID,
    feature_id: DPMPP_2M_SDE_GPU_FEATURE_ID,
    source_ordinal: DPMPP_2M_SDE_GPU_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_2m_sde_gpu_comfy_model_0169",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dpmpp2mSdeGpuOptions {
    pub eta: f32,
    pub noise_scale: f32,
    pub solver_type: Dpmpp2mSdeSolverType,
}

impl Dpmpp2mSdeGpuOptions {
    fn into_family_options(self) -> Result<Dpmpp2mSdeOptions, Dpmpp2mSdeSamplerError> {
        Dpmpp2mSdeOptions::new_with_solver_type(self.eta, self.noise_scale, self.solver_type)
    }
}

impl Default for Dpmpp2mSdeGpuOptions {
    fn default() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
            solver_type: Dpmpp2mSdeSolverType::Midpoint,
        }
    }
}

#[derive(Debug, Error)]
pub enum Dpmpp2mSdeGpuError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    EquationFamily(#[from] Dpmpp2mSdeSamplerError),
    #[error("DPM-Solver++(2M) SDE GPU requires sampler identity `dpmpp_2m_sde_gpu`, got {0:?}")]
    WrongSampler(String),
    #[error("native Brownian generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
}

pub fn validate_dpmpp_2m_sde_gpu_generation_device(
    device: DeviceId,
) -> Result<(), Dpmpp2mSdeGpuError> {
    BackendCapabilityMatrix::for_native_device(device).map_err(|error| {
        Dpmpp2mSdeGpuError::DeviceUnavailable {
            device,
            reason: error.reason().to_owned(),
        }
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_2m_sde_gpu<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: Dpmpp2mSdeGpuOptions,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), Dpmpp2mSdeGpuError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        profile.identity(),
    )?;
    if plan.sampler().as_str() != DPMPP_2M_SDE_GPU_SAMPLER_ID {
        return Err(Dpmpp2mSdeGpuError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let device = initial.descriptor().device();
    if sigmas.len() > 1 {
        validate_dpmpp_2m_sde_gpu_generation_device(device)?;
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
        Dpmpp2mSdeOptions::source_defaults()
    } else {
        options.into_family_options()?
    };
    Ok(sample_dpmpp_2m_sde_with_generation_placement(
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
