use crate::{
    NORMAL_FOUNDATION_DEFINITION, SamplingProfile, SchedulerDefinition, SchedulerError,
    SchedulerRegistry, SchedulerRequest,
};
use sim_tensor::{CpuBackend, ExecutionContext};

pub const NORMAL_SCHEDULER_SOURCE_ORDINAL: u16 = NORMAL_FOUNDATION_DEFINITION.source_ordinal;

pub const DEFINITION: SchedulerDefinition = NORMAL_FOUNDATION_DEFINITION;

pub fn normal_schedule_adapter(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &impl SamplingProfile,
    request: &SchedulerRequest,
) -> Result<Vec<f32>, SchedulerError> {
    crate::normal_schedule(backend, context, registry, profile, request)
}
