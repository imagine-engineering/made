use crate::{
    CfgPpDenoiserOutput, CompatibilityNoiseRequest, SamplerDefinition, SamplingPlan,
    SamplingProfile, SamplingProgress, SamplingTrace,
    generated_res_multistep_sim_model_0193::{
        ResMultistepFamilyOptions, ResMultistepOptions, ResMultistepSamplerError,
        sample_res_multistep_family,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, RngCheckpoint, Tensor};
use std::fmt::Display;

pub const RES_MULTISTEP_CFG_PP_SAMPLER_ID: &str = "res_multistep_cfg_pp";
pub const RES_MULTISTEP_CFG_PP_FEATURE_ID: &str = "COMFY-MODEL-0196";
pub const RES_MULTISTEP_CFG_PP_SOURCE_ORDINAL: u16 = 31;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: RES_MULTISTEP_CFG_PP_SAMPLER_ID,
    feature_id: RES_MULTISTEP_CFG_PP_FEATURE_ID,
    source_ordinal: RES_MULTISTEP_CFG_PP_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/res_multistep_cfg_pp_comfy_model_0196",
    stochastic: true,
};

pub type ResMultistepCfgPpOptions = ResMultistepOptions;
pub type ResMultistepCfgPpDenoiserOutput = CfgPpDenoiserOutput;

#[allow(clippy::too_many_arguments)]
pub fn sample_res_multistep_cfg_pp<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: ResMultistepCfgPpOptions,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<ResMultistepCfgPpDenoiserOutput, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), ResMultistepSamplerError>
where
    CallbackError: Display,
{
    let family_options = ResMultistepFamilyOptions::new(0.0, options.noise_scale(), true)?;
    sample_res_multistep_family(
        backend,
        plan,
        profile,
        RES_MULTISTEP_CFG_PP_SAMPLER_ID,
        initial,
        sigmas,
        noise_request,
        family_options,
        context,
        denoiser,
        callback,
    )
}
