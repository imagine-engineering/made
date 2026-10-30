use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplingError, SamplingPlan, SamplingProfile,
    SamplingProfileError, SamplingProgress, SamplingTrace, SchedulerError,
    generated_exp_heun_2_x0_sim_model_0183::ExpHeun2X0Error,
    generated_seeds_2_sim_model_0199::{
        Seeds2DenoiserStage, Seeds2Error, Seeds2Options, Seeds2SolverType,
        sample_seeds_2_stochastic_family, seeds_2_rng_profile, validate_seeds_2_generation_device,
    },
};
use sim_tensor::{
    CpuBackend, DeviceId, ExecutionContext, RngCheckpoint, RngCompatibilityError,
    RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::NativeDiffusionTensorError,
};
use std::fmt::Display;
use thiserror::Error;

pub use crate::generated_exp_heun_2_x0_sim_model_0183::{
    ExpHeun2X0DenoiserStage, ExpHeun2X0SolverType,
};

pub const EXP_HEUN_2_X0_SDE_SAMPLER_ID: &str = "exp_heun_2_x0_sde";
pub const EXP_HEUN_2_X0_SDE_FEATURE_ID: &str = "COMFY-MODEL-0184";
pub const EXP_HEUN_2_X0_SDE_SOURCE_ORDINAL: u16 = 7;
pub const EXP_HEUN_2_X0_SDE_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: EXP_HEUN_2_X0_SDE_SAMPLER_ID,
    feature_id: EXP_HEUN_2_X0_SDE_FEATURE_ID,
    source_ordinal: EXP_HEUN_2_X0_SDE_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/exp_heun_2_x0_sde_comfy_model_0184",
    stochastic: true,
};

pub fn exp_heun_2_x0_sde_rng_profile(
    device: DeviceId,
) -> (RngSeedTransform, RngGenerationPlacement) {
    seeds_2_rng_profile(device)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpHeun2X0SdeOptions {
    pub eta: f32,
    pub noise_scale: f32,
    pub solver_type: ExpHeun2X0SolverType,
}

impl Default for ExpHeun2X0SdeOptions {
    fn default() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
            solver_type: ExpHeun2X0SolverType::Phi2,
        }
    }
}

#[derive(Debug, Error)]
pub enum ExpHeun2X0SdeError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    SamplingProfile(#[from] SamplingProfileError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error(transparent)]
    Family(#[from] ExpHeun2X0Error),
    #[error(
        "stochastic exponential Heun 2 x0 requires sampler identity `exp_heun_2_x0_sde`, got {0:?}"
    )]
    WrongSampler(String),
    #[error("stochastic exponential Heun 2 x0 option {name} must be finite, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error(
        "stochastic exponential Heun 2 x0 denoiser failed at step {step} during {stage:?}: {reason}"
    )]
    Denoiser {
        step: usize,
        stage: ExpHeun2X0DenoiserStage,
        reason: String,
    },
    #[error(
        "stochastic exponential Heun 2 x0 denoiser descriptor changed at step {step} during {stage:?}"
    )]
    DenoiserContract {
        step: usize,
        stage: ExpHeun2X0DenoiserStage,
    },
    #[error(
        "stochastic exponential Heun 2 x0 produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("native exponential-Heun noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
    #[error(transparent)]
    Seeds2Family(Seeds2Error),
}

pub fn validate_exp_heun_2_x0_sde_generation_device(
    device: DeviceId,
) -> Result<(), ExpHeun2X0SdeError> {
    validate_seeds_2_generation_device(device).map_err(map_family_error)
}

#[allow(clippy::too_many_arguments)]
pub fn sample_exp_heun_2_x0_sde<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: ExpHeun2X0SdeOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, ExpHeun2X0DenoiserStage) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), ExpHeun2X0SdeError>
where
    CallbackError: Display,
{
    sample_seeds_2_stochastic_family(
        backend,
        plan,
        profile,
        initial,
        sigmas,
        EXP_HEUN_2_X0_SDE_SAMPLER_ID,
        EXP_HEUN_2_X0_SDE_NOISE_CONTRACT_ID,
        noise_request,
        Seeds2Options {
            eta: options.eta,
            noise_scale: options.noise_scale,
            intermediate_step_ratio: 1.0,
            solver_type: map_solver_type(options.solver_type),
        },
        context,
        |input, sigma, step, stage| denoiser(input, sigma, step, map_denoiser_stage(stage)),
        callback,
    )
    .map_err(map_family_error)
}

const fn map_solver_type(solver_type: ExpHeun2X0SolverType) -> Seeds2SolverType {
    match solver_type {
        ExpHeun2X0SolverType::Phi1 => Seeds2SolverType::Phi1,
        ExpHeun2X0SolverType::Phi2 => Seeds2SolverType::Phi2,
    }
}

const fn map_denoiser_stage(stage: Seeds2DenoiserStage) -> ExpHeun2X0DenoiserStage {
    match stage {
        Seeds2DenoiserStage::Primary => ExpHeun2X0DenoiserStage::Primary,
        Seeds2DenoiserStage::Intermediate => ExpHeun2X0DenoiserStage::Corrector,
    }
}

fn map_family_error(error: Seeds2Error) -> ExpHeun2X0SdeError {
    match error {
        Seeds2Error::Tensor(error) => ExpHeun2X0SdeError::Tensor(error),
        Seeds2Error::TensorKernel(error) => ExpHeun2X0SdeError::TensorKernel(error),
        Seeds2Error::Sampling(error) => ExpHeun2X0SdeError::Sampling(error),
        Seeds2Error::SamplingProfile(error) => ExpHeun2X0SdeError::SamplingProfile(error),
        Seeds2Error::Scheduler(error) => ExpHeun2X0SdeError::Scheduler(error),
        Seeds2Error::RngCompatibility(error) => ExpHeun2X0SdeError::RngCompatibility(error),
        Seeds2Error::WrongSampler { actual, .. } => ExpHeun2X0SdeError::WrongSampler(actual),
        Seeds2Error::InvalidOption { name, value } => {
            ExpHeun2X0SdeError::InvalidOption { name, value }
        }
        Seeds2Error::Denoiser {
            step,
            stage,
            reason,
        } => ExpHeun2X0SdeError::Denoiser {
            step,
            stage: map_denoiser_stage(stage),
            reason,
        },
        Seeds2Error::DenoiserContract { step, stage } => ExpHeun2X0SdeError::DenoiserContract {
            step,
            stage: map_denoiser_stage(stage),
        },
        Seeds2Error::InvalidCoefficient {
            step,
            coefficient,
            value,
        } => ExpHeun2X0SdeError::Family(ExpHeun2X0Error::InvalidCoefficient {
            step,
            coefficient,
            value,
        }),
        Seeds2Error::NonFinite {
            step,
            stage,
            element,
        } => ExpHeun2X0SdeError::NonFinite {
            step,
            stage,
            element,
        },
        Seeds2Error::DeviceUnavailable { device, reason } => {
            ExpHeun2X0SdeError::DeviceUnavailable { device, reason }
        }
        error => ExpHeun2X0SdeError::Seeds2Family(error),
    }
}
