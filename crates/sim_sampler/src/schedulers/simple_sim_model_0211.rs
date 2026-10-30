use crate::{
    SamplingProfile, SchedulerDefinition, SchedulerError, SchedulerRegistry, SchedulerRequest,
    build_scheduler_schedule,
};
use sim_tensor::{CpuBackend, ExecutionContext};

pub const SIMPLE_SCHEDULER_SOURCE_ORDINAL: u16 = 0;

pub const DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: crate::SIMPLE_SCHEDULER_ID,
    feature_id: crate::SIMPLE_SCHEDULER_FEATURE_ID,
    source_ordinal: SIMPLE_SCHEDULER_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "schedulers/simple_comfy_model_0211",
};

pub fn simple_schedule<P: SamplingProfile>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &P,
    request: &SchedulerRequest,
) -> Result<Vec<f32>, SchedulerError> {
    build_scheduler_schedule(
        backend,
        context,
        registry,
        profile,
        request,
        crate::SIMPLE_SCHEDULER_ID,
        |effective_steps, profile, context, sigmas| {
            let stride = profile.sigma_count() as f64 / effective_steps as f64;
            for index in 0..effective_steps {
                if index.is_multiple_of(256) {
                    context
                        .cancellation
                        .check()
                        .map_err(|_| SchedulerError::Cancelled)?;
                }
                let offset = (index as f64 * stride).trunc() as usize;
                let source_index = profile
                    .sigma_count()
                    .checked_sub(offset.checked_add(1).ok_or(SchedulerError::StepOverflow)?)
                    .ok_or(SchedulerError::StepOverflow)?;
                sigmas.try_push(profile.sigma_at_index(source_index)?)?;
            }
            sigmas.try_push(0.0)?;
            Ok(())
        },
    )
}
