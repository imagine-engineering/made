use crate::{
    CompatibilityNoiseRequest, EULER_FOUNDATION_DEFINITION, SamplerDefinition, SamplingPlan,
    SamplingProfileIdentity, SamplingProgress, SamplingTrace,
    generated_native_diffusion::{
        EulerOptions, NativeDiffusionSamplerError, sample_euler_with_options,
    },
};
use sim_tensor::{CpuBackend, ExecutionContext, RngCheckpoint, Tensor};
use std::fmt::Display;

pub const EULER_FEATURE_ID: &str = EULER_FOUNDATION_DEFINITION.feature_id;
pub const EULER_SOURCE_ORDINAL: u16 = EULER_FOUNDATION_DEFINITION.source_ordinal;
pub const DEFINITION: SamplerDefinition = EULER_FOUNDATION_DEFINITION;

#[allow(clippy::too_many_arguments)]
pub fn sample_euler_sim_model_0179<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    options: EulerOptions,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<
    (SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>),
    NativeDiffusionSamplerError,
>
where
    CallbackError: Display,
{
    sample_euler_with_options(
        backend,
        plan,
        expected_profile,
        initial,
        sigmas,
        options,
        noise_request,
        context,
        denoiser,
        callback,
    )
}
