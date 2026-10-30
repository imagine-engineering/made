use crate::{
    SamplingProfile, SchedulerDefinition, SchedulerError, SchedulerRegistry, SchedulerRequest,
    build_scheduler_schedule,
};
use sim_tensor::{CpuBackend, CpuWorkspaceVec, ExecutionContext};

pub const KL_OPTIMAL_SCHEDULER_ID: &str = "kl_optimal";
pub const KL_OPTIMAL_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0207";
pub const KL_OPTIMAL_SCHEDULER_SOURCE_ORDINAL: u16 = 8;

pub const DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: KL_OPTIMAL_SCHEDULER_ID,
    feature_id: KL_OPTIMAL_SCHEDULER_FEATURE_ID,
    source_ordinal: KL_OPTIMAL_SCHEDULER_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "schedulers/kl_optimal_comfy_model_0207",
};

pub fn kl_optimal_schedule<P: SamplingProfile>(
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
        KL_OPTIMAL_SCHEDULER_ID,
        kl_optimal_equation,
    )
}

fn kl_optimal_equation<P: SamplingProfile>(
    effective_steps: usize,
    profile: &P,
    context: &ExecutionContext<'_>,
    sigmas: &mut CpuWorkspaceVec<f32>,
) -> Result<(), SchedulerError> {
    let minimum_angle = profile.sigma_min().atan();
    let maximum_angle = profile.sigma_max().atan();
    for index in 0..effective_steps {
        if index.is_multiple_of(256) {
            context
                .cancellation
                .check()
                .map_err(|_| SchedulerError::Cancelled)?;
        }
        let fraction = index as f32 / (effective_steps - 1) as f32;
        let angle = fraction * minimum_angle + (1.0 - fraction) * maximum_angle;
        sigmas.try_push(angle.tan())?;
    }
    sigmas.try_push(0.0)?;
    Ok(())
}
