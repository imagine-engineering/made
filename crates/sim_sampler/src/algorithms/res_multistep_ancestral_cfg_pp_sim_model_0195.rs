use crate::{
    CfgPpDenoiserOutput, CompatibilityNoiseRequest, SamplerDefinition, SamplingPlan,
    SamplingProfile, SamplingProgress, SamplingTrace,
    generated_res_multistep_sim_model_0193::{
        RES_MULTISTEP_NOISE_CONTRACT_ID, ResMultistepFamilyOptions, ResMultistepSamplerError,
        sample_res_multistep_family,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, RngCheckpoint, Tensor};
use std::fmt::Display;

pub const RES_MULTISTEP_ANCESTRAL_CFG_PP_SAMPLER_ID: &str = "res_multistep_ancestral_cfg_pp";
pub const RES_MULTISTEP_ANCESTRAL_CFG_PP_FEATURE_ID: &str = "COMFY-MODEL-0195";
pub const RES_MULTISTEP_ANCESTRAL_CFG_PP_SOURCE_ORDINAL: u16 = 33;
pub const RES_MULTISTEP_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID: &str = RES_MULTISTEP_NOISE_CONTRACT_ID;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: RES_MULTISTEP_ANCESTRAL_CFG_PP_SAMPLER_ID,
    feature_id: RES_MULTISTEP_ANCESTRAL_CFG_PP_FEATURE_ID,
    source_ordinal: RES_MULTISTEP_ANCESTRAL_CFG_PP_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/res_multistep_ancestral_cfg_pp_comfy_model_0195",
    stochastic: true,
};

pub type ResMultistepAncestralCfgPpDenoiserOutput = CfgPpDenoiserOutput;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResMultistepAncestralCfgPpOptions {
    eta: f32,
    noise_scale: f32,
}

impl ResMultistepAncestralCfgPpOptions {
    pub fn new(eta: f32, noise_scale: f32) -> Result<Self, ResMultistepSamplerError> {
        ResMultistepFamilyOptions::new(eta, noise_scale, true)?;
        Ok(Self { eta, noise_scale })
    }

    pub const fn source_defaults() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
        }
    }

    pub const fn eta(self) -> f32 {
        self.eta
    }

    pub const fn noise_scale(self) -> f32 {
        self.noise_scale
    }

    fn family(self) -> Result<ResMultistepFamilyOptions, ResMultistepSamplerError> {
        ResMultistepFamilyOptions::new(self.eta, self.noise_scale, true)
    }
}

impl Default for ResMultistepAncestralCfgPpOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sample_res_multistep_ancestral_cfg_pp<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: ResMultistepAncestralCfgPpOptions,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(
        &Tensor,
        f32,
        usize,
    ) -> Result<ResMultistepAncestralCfgPpDenoiserOutput, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), ResMultistepSamplerError>
where
    CallbackError: Display,
{
    sample_res_multistep_family(
        backend,
        plan,
        profile,
        RES_MULTISTEP_ANCESTRAL_CFG_PP_SAMPLER_ID,
        initial,
        sigmas,
        noise_request,
        options.family()?,
        context,
        denoiser,
        callback,
    )
}
