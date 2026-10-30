use crate::{
    SamplingProfile, SchedulerDefinition, SchedulerError, SchedulerRegistry, SchedulerRequest,
    normal_schedule_with_mode,
};
use sim_tensor::{CpuBackend, ExecutionContext};

pub const SGM_UNIFORM_SCHEDULER_ID: &str = "sgm_uniform";
pub const SGM_UNIFORM_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0210";
pub const SGM_UNIFORM_SCHEDULER_SOURCE_ORDINAL: u16 = 1;

pub const DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: SGM_UNIFORM_SCHEDULER_ID,
    feature_id: SGM_UNIFORM_SCHEDULER_FEATURE_ID,
    source_ordinal: SGM_UNIFORM_SCHEDULER_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "schedulers/sgm_uniform_comfy_model_0210",
};

pub fn sgm_uniform_schedule<P: SamplingProfile>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &P,
    request: &SchedulerRequest,
) -> Result<Vec<f32>, SchedulerError> {
    normal_schedule_with_mode(
        backend,
        context,
        registry,
        profile,
        request,
        SGM_UNIFORM_SCHEDULER_ID,
        true,
    )
}
