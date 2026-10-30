use crate::{
    SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerRegistry,
};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use thiserror::Error;

pub const DEIS_SAMPLER_ID: &str = "deis";
pub const DEIS_SAMPLER_FEATURE_ID: &str = "COMFY-MODEL-0161";
pub const DEIS_MAX_ORDER: usize = 3;
pub const DEIS_TABULATION_POINTS: usize = 10_000;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DEIS_SAMPLER_ID,
    feature_id: DEIS_SAMPLER_FEATURE_ID,
    source_ordinal: 29,
    aliases: &[],
    implementation_module: "algorithms/deis_comfy_model_0161",
    stochastic: false,
};

const DEIS_EPSILON_S: f32 = 1.0e-3;
const DEIS_SIGMA_MIN: f32 = 0.002;
const DEIS_SIGMA_MAX: f32 = 80.0;

#[derive(Clone, Copy, Debug, PartialEq)]
struct DeisStepCoefficients {
    values: [f32; DEIS_MAX_ORDER],
    order: usize,
}

impl DeisStepCoefficients {
    fn values(&self) -> &[f32] {
        match self.values.get(..self.order) {
            Some(values) => values,
            None => &[],
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum DeisSamplerError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error("DEIS requires at least one sampling step")]
    ZeroSteps,
    #[error("DEIS cannot execute sampling plan for {actual:?}")]
    SamplerIdentity { actual: String },
    #[error("DEIS denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("DEIS denoiser output descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("DEIS produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DEIS coefficient history is unavailable at step {step} for order {order}")]
    MissingHistory { step: usize, order: usize },
    #[error("DEIS allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("DEIS arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("DEIS sampling was cancelled at step {step}")]
    Cancelled { step: usize },
}

fn deis_tabulated_coefficients(
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Vec<DeisStepCoefficients>, DeisSamplerError> {
    if sigmas.len() < 2 {
        return Err(DeisSamplerError::ZeroSteps);
    }
    let mut times = Vec::new();
    times
        .try_reserve_exact(sigmas.len())
        .map_err(|_| DeisSamplerError::OutOfMemory("DEIS time schedule"))?;
    let (beta_zero, beta_one, beta_delta) = deis_beta_parameters();
    for (step, sigma) in sigmas.iter().copied().enumerate() {
        context
            .cancellation
            .check()
            .map_err(|_| DeisSamplerError::Cancelled { step })?;
        let time = edm_sigma_to_time(sigma, beta_zero, beta_delta);
        if !time.is_finite() {
            return Err(DeisSamplerError::NonFinite {
                step,
                stage: "time conversion",
                element: 0,
            });
        }
        times.push(time);
    }

    let step_count = sigmas.len() - 1;
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(step_count)
        .map_err(|_| DeisSamplerError::OutOfMemory("DEIS coefficient schedule"))?;
    for step in 0..step_count {
        context
            .cancellation
            .check()
            .map_err(|_| DeisSamplerError::Cancelled { step })?;
        let order = DEIS_MAX_ORDER.min(step + 1);
        let next_sigma = *sigmas
            .get(step + 1)
            .ok_or(DeisSamplerError::Overflow("next sigma lookup"))?;
        if order == 1 || next_sigma <= 0.0 {
            coefficients.push(DeisStepCoefficients {
                values: [0.0; DEIS_MAX_ORDER],
                order: 0,
            });
            continue;
        }
        coefficients.push(tabulate_step_coefficients(
            &times, step, order, beta_zero, beta_one, context,
        )?);
    }
    Ok(coefficients)
}

pub fn sample_deis(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), String>,
) -> Result<SamplingTrace, DeisSamplerError> {
    if sigmas.len() < 2 {
        return Err(DeisSamplerError::ZeroSteps);
    }
    if plan.sampler().as_str() != DEIS_SAMPLER_ID {
        return Err(DeisSamplerError::SamplerIdentity {
            actual: plan.sampler().as_str().to_owned(),
        });
    }
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()
        .map_err(|error| SamplingError::Scheduler(error.to_string()))?;
    plan.validate(&samplers, &schedulers, expected_profile)?;
    let steps = plan.steps();
    let owned_sigmas = copy_sigmas(sigmas)?;
    let callback_latent = initial.clone();
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let coefficients = deis_tabulated_coefficients(sigmas, context)?;
    let mut derivative_history: [Option<Tensor>; DEIS_MAX_ORDER - 1] = [None, None];

    for step in 0..usize::try_from(steps)
        .map_err(|_| DeisSamplerError::Overflow("sampling step traversal"))?
    {
        context
            .cancellation
            .check()
            .map_err(|_| DeisSamplerError::Cancelled { step })?;
        let sigma = *sigmas
            .get(step)
            .ok_or(DeisSamplerError::Overflow("current sigma lookup"))?;
        let next_sigma = *sigmas
            .get(step + 1)
            .ok_or(DeisSamplerError::Overflow("next sigma lookup"))?;
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| DeisSamplerError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(DeisSamplerError::DenoiserContract { step });
        }
        let current_values = tensor_to_f32(backend, &current, context)?;
        let denoised_values = tensor_to_f32(backend, &denoised, context)?;
        let mut derivative_values = backend.workspace_vec::<f32>(context, current_values.len())?;
        for (element, (current_value, denoised_value)) in current_values
            .iter()
            .zip(denoised_values.iter())
            .enumerate()
        {
            if element.is_multiple_of(256) {
                context
                    .cancellation
                    .check()
                    .map_err(|_| DeisSamplerError::Cancelled { step })?;
            }
            if !denoised_value.is_finite() {
                return Err(DeisSamplerError::NonFinite {
                    step,
                    stage: "denoiser",
                    element,
                });
            }
            let derivative = (current_value - denoised_value) / sigma;
            if !derivative.is_finite() {
                return Err(DeisSamplerError::NonFinite {
                    step,
                    stage: "derivative",
                    element,
                });
            }
            derivative_values.try_push(derivative)?;
        }
        let derivative = tensor_from_f32(
            backend,
            current.descriptor().shape(),
            &derivative_values,
            context,
        )?;

        let order = if next_sigma <= 0.0 {
            1
        } else {
            DEIS_MAX_ORDER.min(step + 1)
        };
        let previous_one_values = if order >= 2 {
            let previous = derivative_history
                .iter()
                .rev()
                .flatten()
                .next()
                .ok_or(DeisSamplerError::MissingHistory { step, order })?;
            Some(tensor_to_f32(backend, previous, context)?)
        } else {
            None
        };
        let previous_two_values = if order >= 3 {
            let previous = derivative_history
                .iter()
                .rev()
                .flatten()
                .nth(1)
                .ok_or(DeisSamplerError::MissingHistory { step, order })?;
            Some(tensor_to_f32(backend, previous, context)?)
        } else {
            None
        };
        let step_coefficients = coefficients
            .get(step)
            .ok_or(DeisSamplerError::Overflow("coefficient lookup"))?;
        let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
        for element in 0..current_values.len() {
            if element.is_multiple_of(256) {
                context
                    .cancellation
                    .check()
                    .map_err(|_| DeisSamplerError::Cancelled { step })?;
            }
            let current_value = *current_values
                .get(element)
                .ok_or(DeisSamplerError::Overflow("current element lookup"))?;
            let derivative_value = *derivative_values
                .get(element)
                .ok_or(DeisSamplerError::Overflow("derivative element lookup"))?;
            let mut next_value = if order == 1 {
                current_value + (next_sigma - sigma) * derivative_value
            } else {
                let coefficient = *step_coefficients
                    .values()
                    .first()
                    .ok_or(DeisSamplerError::MissingHistory { step, order })?;
                current_value + coefficient * derivative_value
            };
            if let Some(previous_values) = previous_one_values.as_ref() {
                let coefficient = *step_coefficients
                    .values()
                    .get(1)
                    .ok_or(DeisSamplerError::MissingHistory { step, order })?;
                let previous = *previous_values
                    .get(element)
                    .ok_or(DeisSamplerError::Overflow("first history element lookup"))?;
                next_value += coefficient * previous;
            }
            if let Some(previous_values) = previous_two_values.as_ref() {
                let coefficient = *step_coefficients
                    .values()
                    .get(2)
                    .ok_or(DeisSamplerError::MissingHistory { step, order })?;
                let previous = *previous_values
                    .get(element)
                    .ok_or(DeisSamplerError::Overflow("second history element lookup"))?;
                next_value += coefficient * previous;
            }
            if !next_value.is_finite() {
                return Err(DeisSamplerError::NonFinite {
                    step,
                    stage: "latent update",
                    element,
                });
            }
            next_values.try_push(next_value)?;
        }
        let next = tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?;
        session
            .commit_step(
                denoised,
                next,
                context.cancellation,
                |progress, denoised, _next| callback(progress, &callback_latent, denoised),
            )
            .map_err(|error| match error {
                SamplingError::Cancelled => DeisSamplerError::Cancelled { step },
                error => DeisSamplerError::Sampling(error),
            })?;

        if derivative_history.get(1).is_some_and(Option::is_some) {
            derivative_history.rotate_left(1);
            let slot = derivative_history
                .last_mut()
                .ok_or(DeisSamplerError::Overflow("derivative history rotation"))?;
            *slot = Some(derivative);
        } else if derivative_history.get(0).is_some_and(Option::is_some) {
            let slot = derivative_history
                .get_mut(1)
                .ok_or(DeisSamplerError::Overflow("second derivative history slot"))?;
            *slot = Some(derivative);
        } else {
            let slot = derivative_history
                .first_mut()
                .ok_or(DeisSamplerError::Overflow("first derivative history slot"))?;
            *slot = Some(derivative);
        }
    }
    session.finish().map_err(DeisSamplerError::Sampling)
}

fn copy_sigmas(sigmas: &[f32]) -> Result<Vec<f32>, DeisSamplerError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(sigmas.len())
        .map_err(|_| DeisSamplerError::OutOfMemory("DEIS sigma schedule"))?;
    owned.extend_from_slice(sigmas);
    Ok(owned)
}

fn deis_beta_parameters() -> (f32, f32, f32) {
    let minimum_log = (DEIS_SIGMA_MIN * DEIS_SIGMA_MIN + 1.0).ln();
    let maximum_log = (DEIS_SIGMA_MAX * DEIS_SIGMA_MAX + 1.0).ln();
    let beta_delta = 2.0 * (minimum_log / DEIS_EPSILON_S - maximum_log) / (DEIS_EPSILON_S - 1.0);
    let beta_zero = maximum_log - 0.5 * beta_delta;
    (beta_zero, beta_delta + beta_zero, beta_delta)
}

fn edm_sigma_to_time(sigma: f32, beta_zero: f32, beta_delta: f32) -> f32 {
    let radicand = beta_zero * beta_zero + 2.0 * beta_delta * (sigma * sigma + 1.0).ln();
    (radicand.sqrt() - beta_zero) / beta_delta
}

fn tabulate_step_coefficients(
    times: &[f32],
    step: usize,
    order: usize,
    beta_zero: f32,
    beta_one: f32,
    context: &ExecutionContext<'_>,
) -> Result<DeisStepCoefficients, DeisSamplerError> {
    let current = *times
        .get(step)
        .ok_or(DeisSamplerError::Overflow("coefficient current time"))?;
    let next = *times
        .get(step + 1)
        .ok_or(DeisSamplerError::Overflow("coefficient next time"))?;
    let denominator = (DEIS_TABULATION_POINTS - 1) as f32;
    let delta_tau = (next - current) / DEIS_TABULATION_POINTS as f32;
    let mut values = [0.0_f32; DEIS_MAX_ORDER];
    for coefficient_index in 0..order {
        let previous_index = step
            .checked_sub(coefficient_index)
            .ok_or(DeisSamplerError::MissingHistory { step, order })?;
        let previous_time = *times
            .get(previous_index)
            .ok_or(DeisSamplerError::MissingHistory { step, order })?;
        let mut sum = 0.0_f32;
        for point in 0..DEIS_TABULATION_POINTS {
            if point.is_multiple_of(256) {
                context
                    .cancellation
                    .check()
                    .map_err(|_| DeisSamplerError::Cancelled { step })?;
            }
            let tau = current + (next - current) * point as f32 / denominator;
            let mut polynomial = 1.0_f32;
            for other_index in 0..order {
                if other_index == coefficient_index {
                    continue;
                }
                let other_previous_index = step
                    .checked_sub(other_index)
                    .ok_or(DeisSamplerError::MissingHistory { step, order })?;
                let other_time = *times
                    .get(other_previous_index)
                    .ok_or(DeisSamplerError::MissingHistory { step, order })?;
                polynomial *= (tau - other_time) / (previous_time - other_time);
            }
            let alpha = (-0.5 * tau * tau * (beta_one - beta_zero) - tau * beta_zero).exp();
            let derivative_log_alpha = -tau * (beta_one - beta_zero) - beta_zero;
            let integrand = -0.5 * derivative_log_alpha / (alpha * (1.0 - alpha)).sqrt();
            sum += integrand * polynomial;
        }
        let coefficient = sum * delta_tau;
        if !coefficient.is_finite() {
            return Err(DeisSamplerError::NonFinite {
                step,
                stage: "tabulated coefficient",
                element: coefficient_index,
            });
        }
        if let Some(slot) = values.get_mut(coefficient_index) {
            *slot = coefficient;
        } else {
            return Err(DeisSamplerError::Overflow("coefficient output index"));
        }
    }
    Ok(DeisStepCoefficients { values, order })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CancellationToken, CpuWorkspaceAuthority, StreamId};
    use serde_json::Value;

    #[test]
    fn tabulated_coefficients_match_every_fixture_intermediate()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/samplers/deis_comfy_model_0161/trajectory.json"
        )))?;
        let sigmas = fixture
            .get("sigmas")
            .and_then(Value::as_array)
            .ok_or("DEIS fixture sigmas are unavailable")?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|value| value as f32)
                    .ok_or("DEIS fixture sigma is not numeric")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected = fixture
            .get("coefficients")
            .and_then(Value::as_array)
            .ok_or("DEIS fixture coefficients are unavailable")?;
        let tolerance = fixture
            .get("tolerance")
            .and_then(Value::as_f64)
            .ok_or("DEIS fixture tolerance is unavailable")? as f32;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let actual = deis_tabulated_coefficients(&sigmas, &context)?;
        assert_eq!(actual.len(), expected.len());
        for (step, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let expected = expected
                .as_array()
                .ok_or("DEIS fixture coefficient row is not an array")?;
            assert_eq!(actual.values().len(), expected.len(), "step {step}");
            for (index, (actual, expected)) in actual.values().iter().zip(expected).enumerate() {
                let expected = expected
                    .as_f64()
                    .ok_or("DEIS fixture coefficient is not numeric")?
                    as f32;
                assert!(
                    (*actual - expected).abs() <= tolerance,
                    "step {step} coefficient {index}: expected {expected}, got {actual}"
                );
            }
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }
}
