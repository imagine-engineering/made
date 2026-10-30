use crate::{
    SamplingProfile, SchedulerDefinition, SchedulerError, SchedulerRegistry, SchedulerRequest,
    build_scheduler_schedule,
};
use sim_tensor::generated_tensor_creation_01::linspace_f32_value;
use sim_tensor::{CpuBackend, CpuWorkspaceVec, ExecutionContext};

pub const KARRAS_SCHEDULER_ID: &str = "karras";
pub const KARRAS_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0206";
pub const KARRAS_SCHEDULER_SOURCE_ORDINAL: u16 = 2;

pub const DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: KARRAS_SCHEDULER_ID,
    feature_id: KARRAS_SCHEDULER_FEATURE_ID,
    source_ordinal: KARRAS_SCHEDULER_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "schedulers/karras_comfy_model_0206",
};

const RHO: f64 = 7.0;

pub fn karras_schedule<P: SamplingProfile>(
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
        KARRAS_SCHEDULER_ID,
        karras_equation,
    )
}

fn karras_equation<P: SamplingProfile>(
    effective_steps: usize,
    profile: &P,
    context: &ExecutionContext<'_>,
    sigmas: &mut CpuWorkspaceVec<f32>,
) -> Result<(), SchedulerError> {
    write_karras_sigmas(
        effective_steps,
        f64::from(profile.sigma_min()),
        f64::from(profile.sigma_max()),
        RHO,
        context,
        sigmas,
    )
}

pub fn karras_sigmas(
    backend: &CpuBackend,
    steps: u64,
    minimum: f64,
    maximum: f64,
    rho: f64,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    context.check()?;
    let steps = crate::scheduler::validate_parameterized_sigma_bounds(steps, minimum, maximum)?;
    if !rho.is_finite() || rho <= 0.0 || rho > 100.0 {
        return Err(SchedulerError::InvalidParameters(
            "Karras rho must be positive and at most 100",
        ));
    }
    crate::scheduler::build_parameterized_sigmas(backend, steps, context, |values| {
        write_karras_sigmas(steps, minimum, maximum, rho, context, values)
    })
}

fn write_karras_sigmas(
    steps: usize,
    minimum: f64,
    maximum: f64,
    rho: f64,
    context: &ExecutionContext<'_>,
    sigmas: &mut CpuWorkspaceVec<f32>,
) -> Result<(), SchedulerError> {
    let minimum_inverse_rho = minimum.powf(1.0 / rho);
    let maximum_inverse_rho = maximum.powf(1.0 / rho);
    let difference = (minimum_inverse_rho - maximum_inverse_rho) as f32;
    let maximum_inverse_rho = maximum_inverse_rho as f32;
    for index in 0..steps {
        if index.is_multiple_of(256) {
            context
                .cancellation
                .check()
                .map_err(|_| SchedulerError::Cancelled)?;
        }
        let ramp = linspace_f32_value(0.0, 1.0, steps, index)?;
        let inverse_rho = maximum_inverse_rho + ramp * difference;
        sigmas.try_push(inverse_rho.powf(rho as f32))?;
    }
    sigmas.try_push(0.0)?;
    Ok(())
}
