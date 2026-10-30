use crate::{
    SamplingProfile, SchedulerDefinition, SchedulerError, SchedulerRegistry, SchedulerRequest,
    build_scheduler_schedule,
};
use sim_tensor::generated_tensor_creation_01::linspace_f32_value;
use sim_tensor::{CpuBackend, CpuWorkspaceVec, ExecutionContext};

pub const EXPONENTIAL_SCHEDULER_ID: &str = "exponential";
pub const EXPONENTIAL_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0205";
pub const EXPONENTIAL_SCHEDULER_SOURCE_ORDINAL: u16 = 3;

pub const DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: EXPONENTIAL_SCHEDULER_ID,
    feature_id: EXPONENTIAL_SCHEDULER_FEATURE_ID,
    source_ordinal: EXPONENTIAL_SCHEDULER_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "schedulers/exponential_comfy_model_0205",
};

pub fn exponential_schedule<P: SamplingProfile>(
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
        EXPONENTIAL_SCHEDULER_ID,
        exponential_equation,
    )
}

fn exponential_equation<P: SamplingProfile>(
    effective_steps: usize,
    profile: &P,
    context: &ExecutionContext<'_>,
    sigmas: &mut CpuWorkspaceVec<f32>,
) -> Result<(), SchedulerError> {
    write_exponential_sigmas(
        effective_steps,
        f64::from(profile.sigma_min()),
        f64::from(profile.sigma_max()),
        context,
        sigmas,
    )
}

pub fn exponential_sigmas(
    backend: &CpuBackend,
    steps: u64,
    minimum: f64,
    maximum: f64,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    context.check()?;
    let steps = crate::scheduler::validate_parameterized_sigma_bounds(steps, minimum, maximum)?;
    if minimum == 0.0 || maximum == 0.0 {
        return Err(SchedulerError::InvalidParameters(
            "exponential sigma bounds must be positive",
        ));
    }
    crate::scheduler::build_parameterized_sigmas(backend, steps, context, |values| {
        write_exponential_sigmas(steps, minimum, maximum, context, values)
    })
}

fn write_exponential_sigmas(
    steps: usize,
    minimum: f64,
    maximum: f64,
    context: &ExecutionContext<'_>,
    sigmas: &mut CpuWorkspaceVec<f32>,
) -> Result<(), SchedulerError> {
    let maximum_log_sigma = maximum.ln() as f32;
    let minimum_log_sigma = minimum.ln() as f32;
    if steps == 1 {
        sigmas.try_push(maximum_log_sigma.exp())?;
        sigmas.try_push(0.0)?;
        return Ok(());
    }
    if !maximum_log_sigma.is_finite() || !minimum_log_sigma.is_finite() {
        return Err(SchedulerError::NonFiniteSigma {
            index: 0,
            value: f32::NAN,
        });
    }
    for index in 0..steps {
        if index.is_multiple_of(256) {
            context
                .cancellation
                .check()
                .map_err(|_| SchedulerError::Cancelled)?;
        }
        let log_sigma = linspace_f32_value(maximum_log_sigma, minimum_log_sigma, steps, index)?;
        sigmas.try_push(log_sigma.exp())?;
    }
    sigmas.try_push(0.0)?;
    Ok(())
}
