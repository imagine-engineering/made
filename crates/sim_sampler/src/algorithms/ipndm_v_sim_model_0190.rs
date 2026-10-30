use crate::{
    SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const IPNDM_V_SAMPLER_ID: &str = "ipndm_v";
pub const IPNDM_V_FEATURE_ID: &str = "COMFY-MODEL-0190";
pub const IPNDM_V_SOURCE_ORDINAL: u16 = 28;
pub const IPNDM_V_MAX_ORDER: usize = 4;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: IPNDM_V_SAMPLER_ID,
    feature_id: IPNDM_V_FEATURE_ID,
    source_ordinal: IPNDM_V_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/ipndm_v_comfy_model_0190",
    stochastic: false,
};

#[derive(Debug, Error)]
pub enum IpndmVError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("IPNDM-V requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("IPNDM-V denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("IPNDM-V denoiser descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("IPNDM-V history is unavailable at step {step} for order {order}")]
    MissingHistory { step: usize, order: usize },
    #[error("IPNDM-V coefficient equation is singular at step {step}: {role}")]
    SingularCoefficient { step: usize, role: &'static str },
    #[error("IPNDM-V produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("IPNDM-V allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_ipndm_v<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, IpndmVError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != IPNDM_V_SAMPLER_ID {
        return Err(IpndmVError::WrongSampler {
            expected: IPNDM_V_SAMPLER_ID,
            actual: plan.sampler().as_str().to_owned(),
        });
    }

    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| IpndmVError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let callback_latent = initial.clone();
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let mut derivative_history = Vec::new();
    derivative_history
        .try_reserve_exact(IPNDM_V_MAX_ORDER - 1)
        .map_err(|_| IpndmVError::OutOfMemory("derivative history"))?;

    for (step, sigma_pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = sigma_pair
            .first()
            .copied()
            .ok_or(IpndmVError::SingularCoefficient {
                step,
                role: "current sigma",
            })?;
        let next_sigma = sigma_pair
            .get(1)
            .copied()
            .ok_or(IpndmVError::SingularCoefficient {
                step,
                role: "next sigma",
            })?;
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| IpndmVError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(IpndmVError::DenoiserContract { step });
        }
        let denoised_for_equation = denoised.clone();
        let observed = session.observe_step(
            &callback_latent,
            denoised,
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;

        let current_values = tensor_to_f32(backend, &current, context)?;
        let denoised_values = tensor_to_f32(backend, &denoised_for_equation, context)?;
        let mut derivative_values = backend.workspace_vec::<f32>(context, current_values.len())?;
        for (element, (current_value, denoised_value)) in current_values
            .iter()
            .copied()
            .zip(denoised_values.iter().copied())
            .enumerate()
        {
            if element.is_multiple_of(256) {
                context.check()?;
            }
            checked_value(denoised_value, step, "denoiser", element)?;
            let derivative = (current_value - denoised_value) / sigma;
            checked_value(derivative, step, "derivative", element)?;
            derivative_values.try_push(derivative)?;
        }
        let derivative = tensor_from_f32(
            backend,
            current.descriptor().shape(),
            &derivative_values,
            context,
        )?;

        let order = IPNDM_V_MAX_ORDER.min(step + 1);
        let next = if next_sigma == 0.0 {
            denoised_for_equation
        } else {
            let coefficients = variable_step_coefficients(sigmas, step, order)?;
            let required_history = order.saturating_sub(1);
            if derivative_history.len() < required_history {
                return Err(IpndmVError::MissingHistory { step, order });
            }
            let mut history_values = Vec::new();
            history_values
                .try_reserve_exact(required_history)
                .map_err(|_| IpndmVError::OutOfMemory("decoded derivative history"))?;
            for previous in derivative_history.iter().rev().take(required_history) {
                history_values.push(tensor_to_f32(backend, previous, context)?);
            }
            let delta_sigma = next_sigma - sigma;
            let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
            for element in 0..current_values.len() {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                let current_value = current_values
                    .get(element)
                    .copied()
                    .ok_or(IpndmVError::MissingHistory { step, order })?;
                let current_derivative = derivative_values
                    .get(element)
                    .copied()
                    .ok_or(IpndmVError::MissingHistory { step, order })?;
                let first_coefficient = coefficients
                    .first()
                    .copied()
                    .ok_or(IpndmVError::MissingHistory { step, order })?;
                let mut weighted_derivative = first_coefficient * current_derivative;
                for (history_index, values) in history_values.iter().enumerate() {
                    let coefficient = coefficients
                        .get(history_index + 1)
                        .copied()
                        .ok_or(IpndmVError::MissingHistory { step, order })?;
                    let value = values
                        .get(element)
                        .copied()
                        .ok_or(IpndmVError::MissingHistory { step, order })?;
                    weighted_derivative += coefficient * value;
                }
                let next_value = current_value + delta_sigma * weighted_derivative;
                checked_value(next_value, step, "latent update", element)?;
                next_values.try_push(next_value)?;
            }
            tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?
        };

        observed.commit(next, context.cancellation)?;
        if derivative_history.len() == IPNDM_V_MAX_ORDER - 1 {
            derivative_history.remove(0);
        }
        derivative_history.push(derivative);
    }

    context.check()?;
    Ok(session.finish()?)
}

fn variable_step_coefficients(
    sigmas: &[f32],
    step: usize,
    order: usize,
) -> Result<[f32; IPNDM_V_MAX_ORDER], IpndmVError> {
    let mut coefficients = [0.0; IPNDM_V_MAX_ORDER];
    if order == 1 {
        coefficients[0] = 1.0;
        return Ok(coefficients);
    }
    let sigma = schedule_value(sigmas, step, step, "current sigma")?;
    let next_sigma = schedule_value(sigmas, step + 1, step, "next sigma")?;
    let previous_sigma = schedule_value(sigmas, step - 1, step, "previous sigma")?;
    let current_delta = next_sigma - sigma;
    let previous_delta = sigma - previous_sigma;
    let current_ratio = checked_div(current_delta, previous_delta, step, "h_n / h_n_1")?;
    coefficients[0] = (2.0 + current_ratio) / 2.0;
    coefficients[1] = -current_ratio / 2.0;
    if order == 2 {
        validate_coefficients(&coefficients[..order], step)?;
        return Ok(coefficients);
    }

    let previous_two_sigma = schedule_value(sigmas, step - 2, step, "second previous sigma")?;
    let previous_two_delta = previous_sigma - previous_two_sigma;
    let current_plus_previous = current_delta + previous_delta;
    let previous_plus_two = previous_delta + previous_two_delta;
    let temp =
        (1.0 - checked_div(
            current_delta,
            3.0 * current_plus_previous,
            step,
            "third-order current interval",
        )? * checked_div(
            current_delta * current_plus_previous,
            previous_delta * previous_plus_two,
            step,
            "third-order interval product",
        )?) / 2.0;
    let previous_ratio = checked_div(previous_delta, previous_two_delta, step, "h_n_1 / h_n_2")?;
    coefficients[0] += temp;
    coefficients[1] -= (1.0 + previous_ratio) * temp;
    coefficients[2] = temp * previous_ratio;
    if order == 3 {
        validate_coefficients(&coefficients[..order], step)?;
        return Ok(coefficients);
    }

    let previous_three_sigma = schedule_value(sigmas, step - 3, step, "third previous sigma")?;
    let previous_three_delta = previous_two_sigma - previous_three_sigma;
    let current_previous_two = current_plus_previous + previous_two_delta;
    let previous_two_three = previous_plus_two + previous_three_delta;
    let temp_two_left =
        (1.0 - checked_div(
            current_delta,
            3.0 * current_plus_previous,
            step,
            "fourth-order first interval",
        )?) / 2.0
            + (1.0
                - checked_div(
                    current_delta,
                    2.0 * current_plus_previous,
                    step,
                    "fourth-order second interval",
                )?)
                * checked_div(
                    current_delta,
                    6.0 * current_previous_two,
                    step,
                    "fourth-order third interval",
                )?;
    let temp_two_right = checked_div(
        current_delta * current_plus_previous * current_previous_two,
        previous_delta * previous_plus_two * previous_two_three,
        step,
        "fourth-order interval product",
    )?;
    let temp_two = temp_two_left * temp_two_right;
    let cross_ratio = checked_div(
        previous_delta * previous_plus_two,
        previous_two_delta * (previous_two_delta + previous_three_delta),
        step,
        "fourth-order cross ratio",
    )?;
    let previous_three_ratio = checked_div(
        previous_two_delta,
        previous_three_delta,
        step,
        "h_n_2 / h_n_3",
    )?;
    coefficients[0] += temp_two;
    coefficients[1] -= (1.0 + previous_ratio + cross_ratio) * temp_two;
    coefficients[2] += (previous_ratio + cross_ratio) * (1.0 + previous_three_ratio) * temp_two;
    coefficients[3] = -temp_two * cross_ratio * previous_ratio;
    validate_coefficients(&coefficients[..order], step)?;
    Ok(coefficients)
}

fn schedule_value(
    sigmas: &[f32],
    index: usize,
    step: usize,
    role: &'static str,
) -> Result<f32, IpndmVError> {
    sigmas
        .get(index)
        .copied()
        .ok_or(IpndmVError::SingularCoefficient { step, role })
}

fn checked_div(
    numerator: f32,
    denominator: f32,
    step: usize,
    role: &'static str,
) -> Result<f32, IpndmVError> {
    if denominator == 0.0 || !denominator.is_finite() {
        return Err(IpndmVError::SingularCoefficient { step, role });
    }
    let value = numerator / denominator;
    if !value.is_finite() {
        return Err(IpndmVError::SingularCoefficient { step, role });
    }
    Ok(value)
}

fn validate_coefficients(coefficients: &[f32], step: usize) -> Result<(), IpndmVError> {
    for (element, value) in coefficients.iter().copied().enumerate() {
        checked_value(value, step, "coefficient", element)?;
    }
    Ok(())
}

fn checked_value(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<(), IpndmVError> {
    if !value.is_finite() {
        return Err(IpndmVError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(())
}
