use crate::{
    SamplingProfile, SchedulerDefinition, SchedulerError, SchedulerRegistry, SchedulerRequest,
    build_scheduler_schedule_with_capacity,
};
use sim_tensor::{CpuBackend, ExecutionContext};

pub const DDIM_UNIFORM_SCHEDULER_ID: &str = "ddim_uniform";
pub const DDIM_UNIFORM_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0204";

pub const DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: DDIM_UNIFORM_SCHEDULER_ID,
    feature_id: DDIM_UNIFORM_SCHEDULER_FEATURE_ID,
    source_ordinal: 4,
    aliases: &[],
    implementation_module: "schedulers/ddim_uniform_comfy_model_0204",
};

pub fn ddim_uniform_schedule<P: SamplingProfile>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &P,
    request: &SchedulerRequest,
) -> Result<Vec<f32>, SchedulerError> {
    build_scheduler_schedule_with_capacity(
        backend,
        context,
        registry,
        profile,
        request,
        DDIM_UNIFORM_SCHEDULER_ID,
        2,
        |effective_steps, profile, context, full| {
            let second_sigma = profile.sigma_at_index(1)?;
            let second_sigma_is_zero = second_sigma.abs() <= 0.00001;
            let selection_steps = if second_sigma_is_zero {
                effective_steps
                    .checked_add(1)
                    .ok_or(SchedulerError::StepOverflow)?
            } else {
                full.try_push(0.0)?;
                effective_steps
            };
            let stride = (profile.sigma_count() / selection_steps).max(1);
            let mut source_index = 1_usize;
            while source_index < profile.sigma_count() {
                if source_index.is_multiple_of(256) {
                    context
                        .cancellation
                        .check()
                        .map_err(|_| SchedulerError::Cancelled)?;
                }
                full.try_push(profile.sigma_at_index(source_index)?)?;
                source_index = source_index
                    .checked_add(stride)
                    .ok_or(SchedulerError::StepOverflow)?;
            }
            full.reverse();
            Ok(())
        },
    )
}
