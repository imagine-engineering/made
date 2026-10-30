use crate::{
    SamplingProfile, SchedulerDefinition, SchedulerError, SchedulerRegistry, SchedulerRequest,
    build_scheduler_schedule,
};
use sim_tensor::{CpuBackend, CpuWorkspaceVec, ExecutionContext};

pub const LINEAR_QUADRATIC_SCHEDULER_ID: &str = "linear_quadratic";
pub const LINEAR_QUADRATIC_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0208";
pub const LINEAR_QUADRATIC_SCHEDULER_SOURCE_ORDINAL: u16 = 7;

pub const DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: LINEAR_QUADRATIC_SCHEDULER_ID,
    feature_id: LINEAR_QUADRATIC_SCHEDULER_FEATURE_ID,
    source_ordinal: LINEAR_QUADRATIC_SCHEDULER_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "schedulers/linear_quadratic_comfy_model_0208",
};

const THRESHOLD_NOISE: f64 = 0.025;

pub fn linear_quadratic_schedule<P: SamplingProfile>(
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
        LINEAR_QUADRATIC_SCHEDULER_ID,
        linear_quadratic_equation,
    )
}

fn linear_quadratic_equation<P: SamplingProfile>(
    effective_steps: usize,
    profile: &P,
    context: &ExecutionContext<'_>,
    sigmas: &mut CpuWorkspaceVec<f32>,
) -> Result<(), SchedulerError> {
    let sigma_maximum = profile.sigma_max();
    if effective_steps == 1 {
        sigmas.try_push(sigma_maximum)?;
        sigmas.try_push(0.0)?;
        return Ok(());
    }

    let linear_steps = effective_steps / 2;
    let quadratic_steps = effective_steps - linear_steps;
    let linear_steps_squared = linear_steps
        .checked_mul(linear_steps)
        .ok_or(SchedulerError::StepOverflow)?;
    let quadratic_steps_squared = quadratic_steps
        .checked_mul(quadratic_steps)
        .ok_or(SchedulerError::StepOverflow)?;
    let quadratic_denominator = linear_steps
        .checked_mul(quadratic_steps_squared)
        .ok_or(SchedulerError::StepOverflow)?;
    let linear_steps_float = linear_steps as f64;
    let effective_steps_float = effective_steps as f64;
    let quadratic_steps_squared_float = quadratic_steps_squared as f64;
    let threshold_noise_step_difference =
        linear_steps_float - THRESHOLD_NOISE * effective_steps_float;
    let quadratic_coefficient =
        threshold_noise_step_difference / quadratic_denominator as f64;
    let linear_coefficient = THRESHOLD_NOISE / linear_steps_float
        - 2.0 * threshold_noise_step_difference / quadratic_steps_squared_float;
    let constant = quadratic_coefficient * linear_steps_squared as f64;

    for index in 0..linear_steps {
        check_cancellation(index, context)?;
        let normalized_sigma =
            1.0 - index as f64 * THRESHOLD_NOISE / linear_steps_float;
        sigmas.try_push(normalized_sigma as f32 * sigma_maximum)?;
    }
    for index in linear_steps..effective_steps {
        check_cancellation(index, context)?;
        let index_float = index as f64;
        let schedule_value = quadratic_coefficient * index_float.powi(2)
            + linear_coefficient * index_float
            + constant;
        sigmas.try_push((1.0 - schedule_value) as f32 * sigma_maximum)?;
    }
    sigmas.try_push(0.0)?;
    Ok(())
}

fn check_cancellation(
    index: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), SchedulerError> {
    if index.is_multiple_of(256) {
        context
            .cancellation
            .check()
            .map_err(|_| SchedulerError::Cancelled)?;
    }
    Ok(())
}
