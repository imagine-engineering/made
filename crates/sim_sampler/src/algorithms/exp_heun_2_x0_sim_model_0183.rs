use crate::{
    SamplerDefinition, SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError,
    SamplingProgress, SamplingTrace, SchedulerError,
    generated_seeds_2_sim_model_0199::{
        Seeds2DenoiserStage, Seeds2Error, Seeds2Options, Seeds2SolverType,
        sample_seeds_2_deterministic_family,
    },
};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::NativeDiffusionTensorError,
};
use std::fmt::Display;
use thiserror::Error;

pub const EXP_HEUN_2_X0_SAMPLER_ID: &str = "exp_heun_2_x0";
pub const EXP_HEUN_2_X0_FEATURE_ID: &str = "COMFY-MODEL-0183";
pub const EXP_HEUN_2_X0_SOURCE_ORDINAL: u16 = 6;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: EXP_HEUN_2_X0_SAMPLER_ID,
    feature_id: EXP_HEUN_2_X0_FEATURE_ID,
    source_ordinal: EXP_HEUN_2_X0_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/exp_heun_2_x0_comfy_model_0183",
    stochastic: false,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExpHeun2X0SolverType {
    Phi1,
    #[default]
    Phi2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpHeun2X0DenoiserStage {
    Primary,
    Corrector,
}

#[derive(Debug, Error)]
pub enum ExpHeun2X0Error {
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
    #[error("exponential Heun 2 x0 requires sampler identity `exp_heun_2_x0`, got {0:?}")]
    WrongSampler(String),
    #[error("exponential Heun 2 x0 denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: ExpHeun2X0DenoiserStage,
        reason: String,
    },
    #[error("exponential Heun 2 x0 denoiser descriptor changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: ExpHeun2X0DenoiserStage,
    },
    #[error("exponential Heun 2 x0 coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error(
        "exponential Heun 2 x0 produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error(transparent)]
    Family(#[from] Seeds2Error),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_exp_heun_2_x0<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    solver_type: ExpHeun2X0SolverType,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, ExpHeun2X0DenoiserStage) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, ExpHeun2X0Error>
where
    CallbackError: Display,
{
    let options = Seeds2Options {
        eta: 0.0,
        noise_scale: 0.0,
        intermediate_step_ratio: 1.0,
        solver_type: map_solver_type(solver_type),
    };
    sample_seeds_2_deterministic_family(
        backend,
        plan,
        profile,
        initial,
        sigmas,
        EXP_HEUN_2_X0_SAMPLER_ID,
        options,
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

fn map_family_error(error: Seeds2Error) -> ExpHeun2X0Error {
    match error {
        Seeds2Error::Tensor(error) => ExpHeun2X0Error::Tensor(error),
        Seeds2Error::TensorKernel(error) => ExpHeun2X0Error::TensorKernel(error),
        Seeds2Error::Sampling(error) => ExpHeun2X0Error::Sampling(error),
        Seeds2Error::SamplingProfile(error) => ExpHeun2X0Error::SamplingProfile(error),
        Seeds2Error::Scheduler(error) => ExpHeun2X0Error::Scheduler(error),
        Seeds2Error::WrongSampler { actual, .. } => ExpHeun2X0Error::WrongSampler(actual),
        Seeds2Error::Denoiser {
            step,
            stage,
            reason,
        } => ExpHeun2X0Error::Denoiser {
            step,
            stage: map_denoiser_stage(stage),
            reason,
        },
        Seeds2Error::DenoiserContract { step, stage } => ExpHeun2X0Error::DenoiserContract {
            step,
            stage: map_denoiser_stage(stage),
        },
        Seeds2Error::InvalidCoefficient {
            step,
            coefficient,
            value,
        } => ExpHeun2X0Error::InvalidCoefficient {
            step,
            coefficient,
            value,
        },
        Seeds2Error::NonFinite {
            step,
            stage,
            element,
        } => ExpHeun2X0Error::NonFinite {
            step,
            stage,
            element,
        },
        error => ExpHeun2X0Error::Family(error),
    }
}
