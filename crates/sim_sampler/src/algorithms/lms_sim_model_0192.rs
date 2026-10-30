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

pub const LMS_SAMPLER_ID: &str = "lms";
pub const LMS_FEATURE_ID: &str = "COMFY-MODEL-0192";
pub const LMS_SOURCE_ORDINAL: u16 = 10;
pub const LMS_MAX_ORDER: usize = 4;
pub const LMS_MAX_CONFIGURED_ORDER: usize = 100;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: LMS_SAMPLER_ID,
    feature_id: LMS_FEATURE_ID,
    source_ordinal: LMS_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/lms_comfy_model_0192",
    stochastic: false,
};

#[allow(clippy::too_many_arguments)]
pub fn sample_lms<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, LmsSamplerError>
where
    CallbackError: Display,
{
    sample_lms_with_order(
        backend,
        plan,
        expected_profile,
        initial,
        sigmas,
        LMS_MAX_ORDER,
        context,
        denoiser,
        callback,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sample_lms_with_order<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    configured_order: usize,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, LmsSamplerError>
where
    CallbackError: Display,
{
    check_cancelled(context, 0)?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != LMS_SAMPLER_ID {
        return Err(LmsSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    if !(1..=LMS_MAX_CONFIGURED_ORDER).contains(&configured_order) {
        return Err(LmsSamplerError::ConfiguredOrder(configured_order));
    }

    let step_count = sigmas.len().saturating_sub(1);
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| LmsSamplerError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let mut derivative_history = Vec::new();
    derivative_history
        .try_reserve_exact(configured_order)
        .map_err(|_| LmsSamplerError::OutOfMemory("derivative history"))?;

    for (step, sigma_pair) in sigmas.windows(2).enumerate() {
        check_cancelled(context, step)?;
        let sigma = sigma_pair
            .first()
            .copied()
            .ok_or(LmsSamplerError::ArithmeticOverflow("current sigma lookup"))?;
        let next_sigma = sigma_pair
            .get(1)
            .copied()
            .ok_or(LmsSamplerError::ArithmeticOverflow("next sigma lookup"))?;
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| LmsSamplerError::Denoiser { step, reason })?;
        if current.descriptor() != denoised.descriptor() {
            return Err(LmsSamplerError::DenoiserContract { step });
        }

        let current_values = tensor_to_f32(backend, &current, context)?;
        let denoised_values = tensor_to_f32(backend, &denoised, context)?;
        let mut derivative_values = backend.workspace_vec::<f32>(context, current_values.len())?;
        for (element, (current_value, denoised_value)) in current_values
            .iter()
            .copied()
            .zip(denoised_values.iter().copied())
            .enumerate()
        {
            if element.is_multiple_of(256) {
                check_cancelled(context, step)?;
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
        update_history(&mut derivative_history, derivative, step, configured_order)?;

        let observed = session
            .observe_step(
                &current,
                denoised.clone(),
                context.cancellation,
                |progress, callback_latent, callback_denoised| {
                    callback(progress, callback_latent, callback_denoised)
                },
            )
            .map_err(|error| map_sampling_error(error, step))?;

        let next = if next_sigma == 0.0 {
            denoised
        } else {
            let order = configured_order.min(step + 1);
            let mut coefficients = Vec::new();
            coefficients
                .try_reserve_exact(order)
                .map_err(|_| LmsSamplerError::OutOfMemory("step coefficients"))?;
            for coefficient_index in 0..order {
                coefficients.push(linear_multistep_coefficient(
                    order,
                    sigmas,
                    step,
                    coefficient_index,
                )?);
            }

            if derivative_history.len() < order {
                return Err(LmsSamplerError::MissingHistory { step, order });
            }
            let mut decoded_history = Vec::new();
            decoded_history
                .try_reserve_exact(order)
                .map_err(|_| LmsSamplerError::OutOfMemory("decoded derivative history"))?;
            for derivative in derivative_history.iter().rev().take(order) {
                decoded_history.push(tensor_to_f32(backend, derivative, context)?);
            }

            let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
            for (element, current_value) in current_values.iter().copied().enumerate() {
                if element.is_multiple_of(256) {
                    check_cancelled(context, step)?;
                }
                let mut update = 0.0_f32;
                for (coefficient, values) in coefficients.iter().zip(decoded_history.iter()) {
                    let derivative_value = values
                        .get(element)
                        .copied()
                        .ok_or(LmsSamplerError::MissingHistory { step, order })?;
                    update += *coefficient * derivative_value;
                }
                checked_value(update, step, "multistep update", element)?;
                let next_value = current_value + update;
                checked_value(next_value, step, "latent update", element)?;
                next_values.try_push(next_value)?;
            }
            tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?
        };

        observed
            .commit(next, context.cancellation)
            .map_err(|error| map_sampling_error(error, step))?;
    }

    if session.next_step() != step_count {
        return Err(LmsSamplerError::ArithmeticOverflow(
            "sampling step traversal",
        ));
    }
    session.finish().map_err(LmsSamplerError::Sampling)
}

pub fn linear_multistep_coefficient(
    order: usize,
    sigmas: &[f32],
    step: usize,
    coefficient_index: usize,
) -> Result<f32, LmsSamplerError> {
    if order == 0
        || order > LMS_MAX_CONFIGURED_ORDER
        || coefficient_index >= order
        || order.saturating_sub(1) > step
    {
        return Err(LmsSamplerError::InvalidOrder {
            step,
            order,
            coefficient_index,
        });
    }
    let lower = schedule_value(sigmas, step, step, "integration lower bound")?;
    let upper = schedule_value(sigmas, step + 1, step, "integration upper bound")?;
    let interval = upper - lower;
    let basis_index = step
        .checked_sub(coefficient_index)
        .ok_or(LmsSamplerError::InvalidOrder {
            step,
            order,
            coefficient_index,
        })?;
    let basis_sigma = schedule_value(sigmas, basis_index, step, "basis sigma")?;
    let mut polynomial = [0.0_f64; LMS_MAX_CONFIGURED_ORDER + 1];
    polynomial[0] = 1.0;
    let mut degree = 0_usize;

    for history_index in 0..order {
        if history_index == coefficient_index {
            continue;
        }
        let node_index = step
            .checked_sub(history_index)
            .ok_or(LmsSamplerError::InvalidOrder {
                step,
                order,
                coefficient_index,
            })?;
        let node_sigma = schedule_value(sigmas, node_index, step, "history sigma")?;
        let denominator = basis_sigma - node_sigma;
        if denominator == 0.0 || !denominator.is_finite() {
            return Err(LmsSamplerError::SingularCoefficient {
                step,
                order,
                coefficient_index,
            });
        }
        let mut next_polynomial = [0.0_f64; LMS_MAX_CONFIGURED_ORDER + 1];
        for power in 0..=degree {
            let value =
                polynomial
                    .get(power)
                    .copied()
                    .ok_or(LmsSamplerError::ArithmeticOverflow(
                        "Lagrange polynomial lookup",
                    ))?;
            let constant_slot =
                next_polynomial
                    .get_mut(power)
                    .ok_or(LmsSamplerError::ArithmeticOverflow(
                        "Lagrange constant coefficient",
                    ))?;
            // Integrate at tau = lower + interval * u to avoid cancellation
            // between large absolute sigma powers, especially at high orders.
            *constant_slot += value * (lower - node_sigma) / denominator;
            let linear_slot =
                next_polynomial
                    .get_mut(power + 1)
                    .ok_or(LmsSamplerError::ArithmeticOverflow(
                        "Lagrange linear coefficient",
                    ))?;
            *linear_slot += value * interval / denominator;
        }
        degree = degree
            .checked_add(1)
            .ok_or(LmsSamplerError::ArithmeticOverflow(
                "Lagrange polynomial degree",
            ))?;
        polynomial = next_polynomial;
    }

    let mut integral = 0.0_f64;
    for (power, value) in polynomial.iter().copied().take(degree + 1).enumerate() {
        let exponent = i32::try_from(power + 1)
            .map_err(|_| LmsSamplerError::ArithmeticOverflow("integration exponent"))?;
        integral += value / f64::from(exponent);
    }
    let coefficient = (interval * integral) as f32;
    if coefficient.is_finite() {
        Ok(coefficient)
    } else {
        Err(LmsSamplerError::NonFiniteCoefficient {
            step,
            order,
            coefficient_index,
        })
    }
}

fn schedule_value(
    sigmas: &[f32],
    index: usize,
    step: usize,
    role: &'static str,
) -> Result<f64, LmsSamplerError> {
    let value = sigmas
        .get(index)
        .copied()
        .ok_or(LmsSamplerError::ScheduleLookup { step, role })?;
    if value.is_finite() {
        Ok(f64::from(value))
    } else {
        Err(LmsSamplerError::NonFiniteCoefficient {
            step,
            order: 0,
            coefficient_index: 0,
        })
    }
}

fn update_history(
    history: &mut Vec<Tensor>,
    derivative: Tensor,
    step: usize,
    maximum_order: usize,
) -> Result<(), LmsSamplerError> {
    if history.len() == maximum_order {
        history.rotate_left(1);
        let newest = history.last_mut().ok_or(LmsSamplerError::MissingHistory {
            step,
            order: maximum_order,
        })?;
        *newest = derivative;
    } else {
        history.push(derivative);
    }
    Ok(())
}

fn checked_value(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<(), LmsSamplerError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(LmsSamplerError::NonFinite {
            step,
            stage,
            element,
        })
    }
}

fn check_cancelled(context: &ExecutionContext<'_>, step: usize) -> Result<(), LmsSamplerError> {
    context
        .cancellation
        .check()
        .map_err(|_| LmsSamplerError::Cancelled { step })
}

fn map_sampling_error(error: SamplingError, step: usize) -> LmsSamplerError {
    match error {
        SamplingError::Cancelled => LmsSamplerError::Cancelled { step },
        error => LmsSamplerError::Sampling(error),
    }
}

#[derive(Debug, Error)]
pub enum LmsSamplerError {
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error("LMS requires sampler identity `lms`, got {0:?}")]
    WrongSampler(String),
    #[error("LMS configured order must be in 1..=100, got {0}")]
    ConfiguredOrder(usize),
    #[error("LMS denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("LMS denoiser output descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error("LMS order {order} with coefficient {coefficient_index} is invalid at step {step}")]
    InvalidOrder {
        step: usize,
        order: usize,
        coefficient_index: usize,
    },
    #[error("LMS derivative history is unavailable at step {step} for order {order}")]
    MissingHistory { step: usize, order: usize },
    #[error("LMS coefficient {coefficient_index} is singular at step {step} for order {order}")]
    SingularCoefficient {
        step: usize,
        order: usize,
        coefficient_index: usize,
    },
    #[error("LMS coefficient {coefficient_index} is non-finite at step {step} for order {order}")]
    NonFiniteCoefficient {
        step: usize,
        order: usize,
        coefficient_index: usize,
    },
    #[error("LMS sigma schedule is unavailable at step {step}: {role}")]
    ScheduleLookup { step: usize, role: &'static str },
    #[error("LMS produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("LMS allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("LMS arithmetic overflowed while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("LMS sampling was cancelled at step {step}")]
    Cancelled { step: usize },
}
