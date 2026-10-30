use crate::{
    CfgPpDenoiserOutput, SamplerDefinition, SamplingPlan, SamplingProfile, SamplingProgress,
    SamplingTrace,
    generated_gradient_estimation_sim_model_0185::{
        GradientEstimationError, GradientEstimationOptions, sample_gradient_estimation_family,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, Tensor};
use std::fmt::Display;

pub const GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID: &str = "gradient_estimation_cfg_pp";
pub const GRADIENT_ESTIMATION_CFG_PP_FEATURE_ID: &str = "COMFY-MODEL-0186";
pub const GRADIENT_ESTIMATION_CFG_PP_SOURCE_ORDINAL: u16 = 35;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID,
    feature_id: GRADIENT_ESTIMATION_CFG_PP_FEATURE_ID,
    source_ordinal: GRADIENT_ESTIMATION_CFG_PP_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/gradient_estimation_cfg_pp_comfy_model_0186",
    stochastic: false,
};

pub type GradientEstimationCfgPpDenoiserOutput = CfgPpDenoiserOutput;

#[allow(clippy::too_many_arguments)]
pub fn sample_gradient_estimation_cfg_pp<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    options: GradientEstimationOptions,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(
        &Tensor,
        f32,
        usize,
    ) -> Result<GradientEstimationCfgPpDenoiserOutput, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, GradientEstimationError>
where
    CallbackError: Display,
{
    sample_gradient_estimation_family(
        backend,
        plan,
        GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID,
        profile,
        initial,
        sigmas,
        options,
        true,
        context,
        denoiser,
        callback,
    )
}
