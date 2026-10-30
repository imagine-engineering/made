use crate::{
    SamplerDefinition, SamplingPlan, SamplingProfileIdentity, SamplingProgress, SamplingTrace,
    generated_uni_pc_sim_model_0201::{
        UniPcVariant, sample_uni_pc_variant,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, Tensor};
use std::fmt::Display;

pub use crate::generated_uni_pc_sim_model_0201::{UniPcDenoiserStage, UniPcError};

pub const UNI_PC_BH2_SAMPLER_ID: &str = "uni_pc_bh2";
pub const UNI_PC_BH2_FEATURE_ID: &str = "COMFY-MODEL-0202";
pub const UNI_PC_BH2_SOURCE_ORDINAL: u16 = 43;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: UNI_PC_BH2_SAMPLER_ID,
    feature_id: UNI_PC_BH2_FEATURE_ID,
    source_ordinal: UNI_PC_BH2_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/uni_pc_bh2_comfy_model_0202",
    stochastic: false,
};

#[allow(clippy::too_many_arguments)]
pub fn sample_uni_pc_bh2<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize, UniPcDenoiserStage) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, UniPcError>
where
    CallbackError: Display,
{
    sample_uni_pc_variant(
        backend,
        plan,
        expected_profile,
        initial,
        sigmas,
        UNI_PC_BH2_SAMPLER_ID,
        UniPcVariant::Bh2,
        context,
        denoiser,
        callback,
    )
}
