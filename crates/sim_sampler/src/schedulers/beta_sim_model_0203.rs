use crate::{
    SamplingProfile, SchedulerDefinition, SchedulerError, SchedulerRegistry, SchedulerRequest,
    build_scheduler_schedule,
};
use sim_tensor::{CpuBackend, ExecutionContext};

pub const BETA_SCHEDULER_ID: &str = "beta";
pub const BETA_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0203";
pub const BETA_SCHEDULER_SOURCE_ORDINAL: u16 = 5;

pub const DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: BETA_SCHEDULER_ID,
    feature_id: BETA_SCHEDULER_FEATURE_ID,
    source_ordinal: BETA_SCHEDULER_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "schedulers/beta_comfy_model_0203",
};

const BETA_SHAPE: f64 = 0.6;
const LOG_BETA_NORMALIZATION: f64 = 0.881_841_806_141_785_9;
const INVERSE_CDF_ITERATIONS: usize = 80;
const SERIES_ITERATIONS: usize = 256;

pub fn beta_schedule(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &impl SamplingProfile,
    request: &SchedulerRequest,
) -> Result<Vec<f32>, SchedulerError> {
    build_scheduler_schedule(
        backend,
        context,
        registry,
        profile,
        request,
        BETA_SCHEDULER_ID,
        |effective_steps, profile, context, schedule| {
            let total_timesteps = profile
                .sigma_count()
                .checked_sub(1)
                .ok_or_else(|| {
                    SchedulerError::InvalidProfile(
                        "profile must contain at least two sigmas".to_owned(),
                    )
                })?;
            let total_timesteps_f64 = total_timesteps as f64;
            let mut last_timestep = None;
            for index in 0..effective_steps {
                if index.is_multiple_of(256) {
                    context
                        .cancellation
                        .check()
                        .map_err(|_| SchedulerError::Cancelled)?;
                }
                let probability = 1.0 - index as f64 / effective_steps as f64;
                let timestep = (beta_point_six_quantile(probability) * total_timesteps_f64)
                    .round_ties_even() as usize;
                if last_timestep != Some(timestep) {
                    schedule.try_push(profile.sigma_at_model_time(timestep as f32)?)?;
                }
                last_timestep = Some(timestep);
            }
            schedule.try_push(0.0)?;
            Ok(())
        },
    )
}

fn beta_point_six_quantile(probability: f64) -> f64 {
    if probability <= 0.0 {
        return 0.0;
    }
    if probability >= 1.0 {
        return 1.0;
    }
    if probability == 0.5 {
        return 0.5;
    }
    if probability > 0.5 {
        return 1.0 - beta_point_six_quantile(1.0 - probability);
    }

    let mut lower = 0.0;
    let mut upper = 0.5;
    for _ in 0..INVERSE_CDF_ITERATIONS {
        let midpoint = (lower + upper) * 0.5;
        if regularized_beta_lower_half(midpoint) < probability {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    (lower + upper) * 0.5
}

fn regularized_beta_lower_half(value: f64) -> f64 {
    if value <= 0.0 {
        return 0.0;
    }

    let mut term = 1.0;
    let mut sum = 1.0;
    for index in 1..=SERIES_ITERATIONS {
        let previous = (index - 1) as f64;
        let index = index as f64;
        term *= (BETA_SHAPE + previous) * (1.0 - BETA_SHAPE + previous) * value
            / ((BETA_SHAPE + 1.0 + previous) * index);
        sum += term;
        if term.abs() <= sum.abs() * f64::EPSILON {
            break;
        }
    }
    (BETA_SHAPE * value.ln() - BETA_SHAPE.ln() - LOG_BETA_NORMALIZATION).exp() * sum
}
