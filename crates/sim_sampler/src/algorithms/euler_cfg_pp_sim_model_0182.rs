use crate::{
    CfgPpDenoiserOutput, CompatibilityNoiseRequest, SamplerDefinition, SamplingPlan,
    SamplingProfile, SamplingProgress, SamplingTrace,
    generated_euler_ancestral_cfg_pp_sim_model_0181::{
        EulerAncestralCfgPpError, EulerAncestralCfgPpOptions, sample_euler_cfg_pp_family,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, RngCheckpoint, Tensor};
use std::fmt::Display;

pub const EULER_CFG_PP_SAMPLER_ID: &str = "euler_cfg_pp";
pub const EULER_CFG_PP_FEATURE_ID: &str = "COMFY-MODEL-0182";
pub const EULER_CFG_PP_SOURCE_ORDINAL: u16 = 1;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: EULER_CFG_PP_SAMPLER_ID,
    feature_id: EULER_CFG_PP_FEATURE_ID,
    source_ordinal: EULER_CFG_PP_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/euler_cfg_pp_comfy_model_0182",
    stochastic: false,
};

pub type EulerCfgPpDenoiserOutput = CfgPpDenoiserOutput;

#[allow(clippy::too_many_arguments)]
pub fn sample_euler_cfg_pp<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<EulerCfgPpDenoiserOutput, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), EulerAncestralCfgPpError>
where
    CallbackError: Display,
{
    sample_euler_cfg_pp_family(
        backend,
        plan,
        EULER_CFG_PP_SAMPLER_ID,
        profile,
        initial,
        sigmas,
        noise_request,
        EulerAncestralCfgPpOptions {
            eta: 0.0,
            noise_scale: 0.0,
        },
        context,
        denoiser,
        callback,
    )
}
