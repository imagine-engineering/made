use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplingPlan, SamplingProfile, SamplingProgress,
    SamplingTrace,
    generated_sa_solver_sim_model_0197::{
        SaSolverError, SaSolverEvaluation, SaSolverFamilyOptions, SaSolverOptions,
        sample_sa_solver_family, source_default_tau_interval,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, RngCheckpoint, Tensor};
use std::fmt::Display;

pub const SA_SOLVER_PECE_SAMPLER_ID: &str = "sa_solver_pece";
pub const SA_SOLVER_PECE_FEATURE_ID: &str = "COMFY-MODEL-0198";
pub const SA_SOLVER_PECE_SOURCE_ORDINAL: u16 = 40;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: SA_SOLVER_PECE_SAMPLER_ID,
    feature_id: SA_SOLVER_PECE_FEATURE_ID,
    source_ordinal: SA_SOLVER_PECE_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/sa_solver_pece_comfy_model_0198",
    stochastic: true,
};

#[allow(clippy::too_many_arguments)]
pub fn sample_sa_solver_pece<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: SaSolverOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, SaSolverEvaluation) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), SaSolverError>
where
    CallbackError: Display,
{
    if sigmas.len() <= 1 {
        return sample_sa_solver_family(
            backend,
            plan,
            profile,
            SA_SOLVER_PECE_SAMPLER_ID,
            initial,
            sigmas,
            noise_request,
            SaSolverFamilyOptions::new(options, true),
            context,
            |_sigma, _step| Ok(0.0),
            move |input, sigma, step, evaluation| denoiser(input, sigma, step, evaluation),
            callback,
        );
    }
    let (start_sigma, end_sigma) = source_default_tau_interval(profile)?;
    sample_sa_solver_family(
        backend,
        plan,
        profile,
        SA_SOLVER_PECE_SAMPLER_ID,
        initial,
        sigmas,
        noise_request,
        SaSolverFamilyOptions::new(options, true),
        context,
        move |sigma, _step| {
            Ok(if start_sigma >= sigma && sigma >= end_sigma {
                1.0
            } else {
                0.0
            })
        },
        move |input, sigma, step, evaluation| denoiser(input, sigma, step, evaluation),
        callback,
    )
}
