use crate::{
    SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    exponential_integrator_phi_one,
};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const UNI_PC_SAMPLER_ID: &str = "uni_pc";
pub const UNI_PC_FEATURE_ID: &str = "COMFY-MODEL-0201";
pub const UNI_PC_SOURCE_ORDINAL: u16 = 42;
pub const UNI_PC_MAX_ORDER: usize = 3;
pub const UNI_PC_TERMINAL_SIGMA: f32 = 0.001;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: UNI_PC_SAMPLER_ID,
    feature_id: UNI_PC_FEATURE_ID,
    source_ordinal: UNI_PC_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/uni_pc_comfy_model_0201",
    stochastic: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UniPcDenoiserStage {
    Initial,
    Corrector,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub(crate) enum UniPcVariant {
    Bh1,
    Bh2,
}

#[derive(Clone)]
struct ModelHistory {
    sigma: f32,
    denoised: Tensor,
}

#[derive(Debug, Error)]
pub enum UniPcError {
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error("UniPC requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("UniPC denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: UniPcDenoiserStage,
        reason: String,
    },
    #[error("UniPC denoiser descriptor changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: UniPcDenoiserStage,
    },
    #[error("UniPC coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("UniPC history is unavailable at step {step} for order {order}")]
    MissingHistory { step: usize, order: usize },
    #[error("UniPC coefficient system is singular at step {step} for order {order}")]
    SingularSystem { step: usize, order: usize },
    #[error("UniPC produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("UniPC arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("UniPC sampling was cancelled at step {step}")]
    Cancelled { step: usize },
}

pub fn sample_uni_pc<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize, UniPcDenoiserStage) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, UniPcError>
where
    CallbackError: Display,
{
    sample_uni_pc_variant(
        backend,
        plan,
        expected_profile,
        initial,
        sigmas,
        UNI_PC_SAMPLER_ID,
        UniPcVariant::Bh1,
        context,
        denoiser,
        callback,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_uni_pc_variant<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    expected_sampler: &'static str,
    variant: UniPcVariant,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, UniPcDenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, UniPcError>
where
    CallbackError: Display,
{
    check_cancelled(context, 0)?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != expected_sampler {
        return Err(UniPcError::WrongSampler {
            expected: expected_sampler,
            actual: plan.sampler().as_str().to_owned(),
        });
    }

    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("UniPC sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let step_count =
        usize::try_from(plan.steps()).map_err(|_| UniPcError::Overflow("sampling step count"))?;
    let first_sigma = sigmas
        .first()
        .copied()
        .ok_or(SamplingError::ScheduleLength {
            expected: step_count.saturating_add(1),
            actual: 0,
        })?;
    let last_sigma = sigmas
        .last()
        .copied()
        .ok_or(SamplingError::ScheduleLength {
            expected: step_count.saturating_add(1),
            actual: 0,
        })?;
    let terminal_sigma = if last_sigma == 0.0 {
        UNI_PC_TERMINAL_SIGMA
    } else {
        last_sigma
    };

    let normalized_initial = scale_tensor(
        backend,
        &initial,
        checked_finite(
            0,
            "initial inverse sigma scale",
            1.0 / sigma_scale(first_sigma, 0)?,
        )?,
        0,
        "initial normalization",
        context,
    )?;
    if step_count == 0 {
        let output = scale_tensor(
            backend,
            &normalized_initial,
            checked_finite(
                0,
                "terminal inverse alpha",
                1.0 / sigma_alpha(terminal_sigma, 0)?,
            )?,
            0,
            "terminal scaling",
            context,
        )?;
        return SamplingSession::new(plan, owned_sigmas, output)?
            .finish()
            .map_err(Into::into);
    }

    let order = UNI_PC_MAX_ORDER.min(sigmas.len().saturating_sub(2));
    let mut session = SamplingSession::new(plan, owned_sigmas, normalized_initial.clone())?;
    let mut state = normalized_initial;
    let mut history = Vec::new();
    history
        .try_reserve_exact(UNI_PC_MAX_ORDER)
        .map_err(|_| SamplingError::OutOfMemory("UniPC model history"))?;

    for step_index in 0..step_count {
        check_cancelled(context, step_index)?;
        if step_index == 0 {
            let denoised = evaluate_model(
                backend,
                &state,
                first_sigma,
                0,
                UniPcDenoiserStage::Initial,
                context,
                &mut denoiser,
            )?;
            history.push(ModelHistory {
                sigma: first_sigma,
                denoised,
            });
        } else if step_index < order {
            let target_sigma = adjusted_sigma(sigmas, step_index)?;
            let (next, model) = multistep_update(
                backend,
                &state,
                &history,
                target_sigma,
                step_index,
                step_index,
                true,
                variant,
                context,
                &mut denoiser,
            )?;
            state = next;
            history.push(ModelHistory {
                sigma: target_sigma,
                denoised: model.ok_or(UniPcError::MissingHistory {
                    step: step_index,
                    order: step_index,
                })?,
            });
        } else {
            let last_outer_step = step_index == step_count.saturating_sub(1);
            let final_target = step_index
                .checked_add(usize::from(last_outer_step))
                .ok_or(UniPcError::Overflow("final target index"))?;
            for target_index in step_index..=final_target {
                let target_sigma = adjusted_sigma(sigmas, target_index)?;
                let step_order = order.min(
                    step_count
                        .checked_add(1)
                        .and_then(|value| value.checked_sub(target_index))
                        .ok_or(UniPcError::Overflow("lower-order final step"))?,
                );
                let use_corrector = target_index < step_count;
                let (next, model) = multistep_update(
                    backend,
                    &state,
                    &history,
                    target_sigma,
                    target_index,
                    step_order,
                    use_corrector,
                    variant,
                    context,
                    &mut denoiser,
                )?;
                state = next;
                shift_history(&mut history, target_sigma, model, step_index, order)?;
            }
        }

        let callback_denoised = history
            .last()
            .ok_or(UniPcError::MissingHistory {
                step: step_index,
                order,
            })?
            .denoised
            .clone();
        let observed = session
            .observe_step(
                &state,
                callback_denoised,
                context.cancellation,
                |progress, latent, denoised| callback(progress, latent, denoised),
            )
            .map_err(|error| map_sampling_error(error, step_index))?;
        let committed = if step_index == step_count.saturating_sub(1) {
            scale_tensor(
                backend,
                &state,
                checked_finite(
                    step_index,
                    "terminal inverse alpha",
                    1.0 / sigma_alpha(terminal_sigma, step_index)?,
                )?,
                step_index,
                "terminal scaling",
                context,
            )?
        } else {
            state.clone()
        };
        observed
            .commit(committed, context.cancellation)
            .map_err(|error| map_sampling_error(error, step_index))?;
    }

    session.finish().map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn multistep_update(
    backend: &CpuBackend,
    state: &Tensor,
    history: &[ModelHistory],
    target_sigma: f32,
    step: usize,
    order: usize,
    use_corrector: bool,
    variant: UniPcVariant,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(&Tensor, f32, usize, UniPcDenoiserStage) -> Result<Tensor, String>,
) -> Result<(Tensor, Option<Tensor>), UniPcError> {
    if order == 0 || order > UNI_PC_MAX_ORDER || history.len() < order {
        return Err(UniPcError::MissingHistory { step, order });
    }
    check_cancelled(context, step)?;
    let previous = history
        .last()
        .ok_or(UniPcError::MissingHistory { step, order })?;
    let previous_lambda = sigma_lambda(previous.sigma, step, "previous sigma")?;
    let target_lambda = sigma_lambda(target_sigma, step, "target sigma")?;
    let h = checked_positive(step, "half-log-SNR step", target_lambda - previous_lambda)?;
    let negative_h = -h;
    let phi_one = checked_finite(
        step,
        "h phi one",
        exponential_integrator_phi_one(negative_h),
    )?;
    let b_h = match variant {
        UniPcVariant::Bh1 => negative_h,
        UniPcVariant::Bh2 => phi_one,
    };
    let b_h = checked_nonzero(step, "B(h)", b_h)?;
    let alpha_target = sigma_alpha(target_sigma, step)?;
    let standard_deviation_target = sigma_standard_deviation(target_sigma, step)?;
    let standard_deviation_previous = sigma_standard_deviation(previous.sigma, step)?;
    let state_values = tensor_to_f32(backend, state, context)?;
    let previous_values = tensor_to_f32(backend, &previous.denoised, context)?;
    validate_values(&state_values, step, "latent")?;
    validate_values(&previous_values, step, "previous denoiser")?;

    let mut rks = [0.0_f32; UNI_PC_MAX_ORDER];
    let mut older_values = Vec::new();
    older_values
        .try_reserve_exact(order.saturating_sub(1))
        .map_err(|_| SamplingError::OutOfMemory("UniPC divided differences"))?;
    for history_distance in 1..order {
        let history_index = history
            .len()
            .checked_sub(history_distance + 1)
            .ok_or(UniPcError::MissingHistory { step, order })?;
        let older = history
            .get(history_index)
            .ok_or(UniPcError::MissingHistory { step, order })?;
        let older_lambda = sigma_lambda(older.sigma, step, "history sigma")?;
        rks[history_distance - 1] =
            checked_nonzero(step, "history ratio", (older_lambda - previous_lambda) / h)?;
        older_values.push(tensor_to_f32(backend, &older.denoised, context)?);
    }
    rks[order - 1] = 1.0;

    let b = bh_system_rhs(negative_h, b_h, order, step)?;
    let mut predictor_weights = [0.0_f32; UNI_PC_MAX_ORDER];
    if order == 2 {
        predictor_weights[0] = 0.5;
    } else if order > 2 {
        predictor_weights[..order - 1].copy_from_slice(
            &solve_vandermonde(&rks[..order - 1], &b[..order - 1], step)?[..order - 1],
        );
    }
    let corrector_weights = if use_corrector {
        if order == 1 {
            let mut weights = [0.0_f32; UNI_PC_MAX_ORDER];
            weights[0] = 0.5;
            weights
        } else {
            solve_vandermonde(&rks[..order], &b[..order], step)?
        }
    } else {
        [0.0_f32; UNI_PC_MAX_ORDER]
    };

    let standard_deviation_ratio = checked_finite(
        step,
        "standard-deviation ratio",
        standard_deviation_target / standard_deviation_previous,
    )?;
    let mut base_values = backend.workspace_vec::<f32>(context, state_values.len())?;
    let mut predictor_values = backend.workspace_vec::<f32>(context, state_values.len())?;
    for (element, (state_value, previous_value)) in state_values
        .iter()
        .copied()
        .zip(previous_values.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            check_cancelled(context, step)?;
        }
        let base = standard_deviation_ratio * state_value - alpha_target * phi_one * previous_value;
        checked_element(base, step, "base update", element)?;
        let predictor_residual = divided_difference_sum(
            &older_values,
            &previous_values,
            &rks,
            &predictor_weights,
            order.saturating_sub(1),
            element,
            step,
        )?;
        let predictor = base - alpha_target * b_h * predictor_residual;
        checked_element(predictor, step, "predictor", element)?;
        base_values.try_push(base)?;
        predictor_values.try_push(predictor)?;
    }
    let predictor = tensor_from_f32(
        backend,
        state.descriptor().shape(),
        &predictor_values,
        context,
    )?;
    if !use_corrector {
        return Ok((predictor, None));
    }

    let model = evaluate_model(
        backend,
        &predictor,
        target_sigma,
        step,
        UniPcDenoiserStage::Corrector,
        context,
        denoiser,
    )?;
    let model_values = tensor_to_f32(backend, &model, context)?;
    validate_values(&model_values, step, "corrector denoiser")?;
    let mut corrected_values = backend.workspace_vec::<f32>(context, state_values.len())?;
    for (element, ((base, model_value), previous_value)) in base_values
        .iter()
        .copied()
        .zip(model_values.iter().copied())
        .zip(previous_values.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            check_cancelled(context, step)?;
        }
        let history_residual = divided_difference_sum(
            &older_values,
            &previous_values,
            &rks,
            &corrector_weights,
            order.saturating_sub(1),
            element,
            step,
        )?;
        let correction =
            history_residual + corrector_weights[order - 1] * (model_value - previous_value);
        let corrected = base - alpha_target * b_h * correction;
        checked_element(corrected, step, "corrector", element)?;
        corrected_values.try_push(corrected)?;
    }
    let corrected = tensor_from_f32(
        backend,
        state.descriptor().shape(),
        &corrected_values,
        context,
    )?;
    Ok((corrected, Some(model)))
}

fn bh_system_rhs(
    negative_h: f32,
    b_h: f32,
    order: usize,
    step: usize,
) -> Result<[f32; UNI_PC_MAX_ORDER], UniPcError> {
    let mut values = [0.0_f32; UNI_PC_MAX_ORDER];
    let mut h_phi = exponential_integrator_phi_one(negative_h) / negative_h - 1.0;
    let mut factorial = 1.0_f32;
    for index in 0..order {
        values[index] = checked_finite(step, "B(h) system value", h_phi * factorial / b_h)?;
        factorial *= (index + 2) as f32;
        h_phi = h_phi / negative_h - 1.0 / factorial;
    }
    Ok(values)
}

fn solve_vandermonde(
    nodes: &[f32],
    right_hand_side: &[f32],
    step: usize,
) -> Result<[f32; UNI_PC_MAX_ORDER], UniPcError> {
    let size = nodes.len();
    if size == 0 || size > UNI_PC_MAX_ORDER || right_hand_side.len() != size {
        return Err(UniPcError::SingularSystem { step, order: size });
    }
    let mut matrix = [[0.0_f32; UNI_PC_MAX_ORDER + 1]; UNI_PC_MAX_ORDER];
    for row in 0..size {
        for column in 0..size {
            matrix[row][column] = nodes[column].powi(row as i32);
        }
        matrix[row][size] = right_hand_side[row];
    }
    for pivot in 0..size {
        let mut pivot_row = pivot;
        for candidate in pivot + 1..size {
            if matrix[candidate][pivot].abs() > matrix[pivot_row][pivot].abs() {
                pivot_row = candidate;
            }
        }
        if matrix[pivot_row][pivot] == 0.0 || !matrix[pivot_row][pivot].is_finite() {
            return Err(UniPcError::SingularSystem { step, order: size });
        }
        matrix.swap(pivot, pivot_row);
        let divisor = matrix[pivot][pivot];
        for column in pivot..=size {
            matrix[pivot][column] /= divisor;
        }
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..=size {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
        }
    }
    let mut solution = [0.0_f32; UNI_PC_MAX_ORDER];
    for index in 0..size {
        solution[index] = checked_finite(step, "B(h) system solution", matrix[index][size])?;
    }
    Ok(solution)
}

fn divided_difference_sum(
    older_values: &[sim_tensor::CpuWorkspaceVec<f32>],
    previous_values: &[f32],
    rks: &[f32; UNI_PC_MAX_ORDER],
    weights: &[f32; UNI_PC_MAX_ORDER],
    count: usize,
    element: usize,
    step: usize,
) -> Result<f32, UniPcError> {
    let previous = previous_values
        .get(element)
        .copied()
        .ok_or(UniPcError::MissingHistory {
            step,
            order: count + 1,
        })?;
    let mut sum = 0.0_f32;
    for index in 0..count {
        let older = older_values
            .get(index)
            .and_then(|values| values.get(element))
            .copied()
            .ok_or(UniPcError::MissingHistory {
                step,
                order: count + 1,
            })?;
        sum += weights[index] * (older - previous) / rks[index];
    }
    checked_finite(step, "divided-difference sum", sum)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_model(
    backend: &CpuBackend,
    state: &Tensor,
    sigma: f32,
    step: usize,
    stage: UniPcDenoiserStage,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(&Tensor, f32, usize, UniPcDenoiserStage) -> Result<Tensor, String>,
) -> Result<Tensor, UniPcError> {
    let physical = scale_tensor(
        backend,
        state,
        sigma_scale(sigma, step)?,
        step,
        "model input scaling",
        context,
    )?;
    let output =
        denoiser(&physical, sigma, step, stage).map_err(|reason| UniPcError::Denoiser {
            step,
            stage,
            reason,
        })?;
    check_cancelled(context, step)?;
    if physical.descriptor() != output.descriptor() {
        return Err(UniPcError::DenoiserContract { step, stage });
    }
    validate_values(&tensor_to_f32(backend, &output, context)?, step, "denoiser")?;
    Ok(output)
}

fn scale_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    scale: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, UniPcError> {
    let values = tensor_to_f32(backend, tensor, context)?;
    let mut scaled = backend.workspace_vec::<f32>(context, values.len())?;
    for (element, value) in values.iter().copied().enumerate() {
        if element.is_multiple_of(256) {
            check_cancelled(context, step)?;
        }
        let value = value * scale;
        checked_element(value, step, stage, element)?;
        scaled.try_push(value)?;
    }
    tensor_from_f32(backend, tensor.descriptor().shape(), &scaled, context).map_err(Into::into)
}

fn shift_history(
    history: &mut Vec<ModelHistory>,
    sigma: f32,
    model: Option<Tensor>,
    step: usize,
    order: usize,
) -> Result<(), UniPcError> {
    if history.len() != order || order == 0 {
        return Err(UniPcError::MissingHistory { step, order });
    }
    let retained_model = history
        .last()
        .ok_or(UniPcError::MissingHistory { step, order })?
        .denoised
        .clone();
    history.rotate_left(1);
    let newest = history
        .last_mut()
        .ok_or(UniPcError::MissingHistory { step, order })?;
    newest.sigma = sigma;
    newest.denoised = model.unwrap_or(retained_model);
    Ok(())
}

fn adjusted_sigma(sigmas: &[f32], index: usize) -> Result<f32, UniPcError> {
    let sigma = sigmas
        .get(index)
        .copied()
        .ok_or(UniPcError::Overflow("sigma lookup"))?;
    if index == sigmas.len().saturating_sub(1) && sigma == 0.0 {
        Ok(UNI_PC_TERMINAL_SIGMA)
    } else {
        Ok(sigma)
    }
}

fn sigma_scale(sigma: f32, step: usize) -> Result<f32, UniPcError> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient: "sigma scale",
            value: sigma,
        });
    }
    let scale = (1.0 + sigma * sigma).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient: "sigma scale",
            value: scale,
        });
    }
    Ok(scale)
}

fn sigma_alpha(sigma: f32, step: usize) -> Result<f32, UniPcError> {
    let log_alpha = sigma_log_alpha(sigma, step)?;
    let alpha = log_alpha.exp();
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient: "marginal alpha",
            value: alpha,
        });
    }
    Ok(alpha)
}

fn sigma_standard_deviation(sigma: f32, step: usize) -> Result<f32, UniPcError> {
    let log_alpha = sigma_log_alpha(sigma, step)?;
    let value = (1.0 - (2.0 * log_alpha).exp()).sqrt();
    if !value.is_finite() || value <= 0.0 {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient: "marginal standard deviation",
            value,
        });
    }
    Ok(value)
}

fn sigma_lambda(sigma: f32, step: usize, coefficient: &'static str) -> Result<f32, UniPcError> {
    let sigma = checked_positive(step, coefficient, sigma)?;
    let log_alpha = sigma_log_alpha(sigma, step)?;
    let variance = 1.0 - (2.0 * log_alpha).exp();
    let lambda = log_alpha - 0.5 * variance.ln();
    checked_finite(step, coefficient, lambda)
}

fn sigma_log_alpha(sigma: f32, step: usize) -> Result<f32, UniPcError> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient: "marginal log alpha",
            value: sigma,
        });
    }
    let denominator = sigma * sigma + 1.0;
    let log_alpha = 0.5 * (1.0 / denominator).ln();
    if !log_alpha.is_finite() {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient: "marginal log alpha",
            value: log_alpha,
        });
    }
    Ok(log_alpha)
}

fn validate_values(values: &[f32], step: usize, stage: &'static str) -> Result<(), UniPcError> {
    for (element, value) in values.iter().copied().enumerate() {
        checked_element(value, step, stage, element)?;
    }
    Ok(())
}

fn checked_element(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<f32, UniPcError> {
    if !value.is_finite() {
        return Err(UniPcError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(value)
}

fn checked_finite(step: usize, coefficient: &'static str, value: f32) -> Result<f32, UniPcError> {
    if !value.is_finite() {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_positive(step: usize, coefficient: &'static str, value: f32) -> Result<f32, UniPcError> {
    let value = checked_finite(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_nonzero(step: usize, coefficient: &'static str, value: f32) -> Result<f32, UniPcError> {
    let value = checked_finite(step, coefficient, value)?;
    if value == 0.0 {
        return Err(UniPcError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn check_cancelled(context: &ExecutionContext<'_>, step: usize) -> Result<(), UniPcError> {
    context
        .cancellation
        .check()
        .map_err(|_| UniPcError::Cancelled { step })
}

fn map_sampling_error(error: SamplingError, step: usize) -> UniPcError {
    match error {
        SamplingError::Cancelled => UniPcError::Cancelled { step },
        error => UniPcError::Sampling(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CancellationToken, CpuWorkspaceAuthority, StreamId};

    #[test]
    fn bh_variants_share_the_system_and_select_only_b_h() -> Result<(), UniPcError> {
        let negative_h = -0.75_f32;
        let phi_one = exponential_integrator_phi_one(negative_h);
        let bh1 = bh_system_rhs(negative_h, negative_h, 3, 0)?;
        let bh2 = bh_system_rhs(negative_h, phi_one, 3, 0)?;
        assert!(bh1 != bh2);
        assert_eq!(UniPcVariant::Bh1, UniPcVariant::Bh1);
        assert_eq!(UniPcVariant::Bh2, UniPcVariant::Bh2);
        Ok(())
    }

    #[test]
    fn bh2_uses_the_shared_traversal_with_source_variant_output()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let profile = crate::SamplingProfileIdentity::new("uni-pc-bh-family-test")?;
        let sigmas = [2.0_f32, 1.5, 1.0, 0.7, 0.4, 0.0];
        let trace = sample_uni_pc_variant(
            &backend,
            SamplingPlan::new(
                UNI_PC_SAMPLER_ID,
                "normal",
                profile.clone(),
                99,
                5,
                1.0,
                1.0,
            )?,
            &profile,
            tensor_from_f32(&backend, &[2], &[0.75, -1.0], &context)?,
            &sigmas,
            UNI_PC_SAMPLER_ID,
            UniPcVariant::Bh2,
            &context,
            |input, sigma, _step, _stage| {
                let input =
                    tensor_to_f32(&backend, input, &context).map_err(|error| error.to_string())?;
                let first = input.first().copied().ok_or("missing first input")?;
                let second = input.get(1).copied().ok_or("missing second input")?;
                let values = [
                    0.2 * first + 0.05 * sigma + 0.1,
                    0.2 * second + 0.05 * sigma - 0.15,
                ];
                tensor_from_f32(&backend, &[2], &values, &context)
                    .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        )?;
        let terminal = tensor_to_f32(
            &backend,
            trace.latents.last().ok_or("missing terminal latent")?,
            &context,
        )?;
        for (actual, expected) in terminal.iter().zip([0.18711297_f32, -0.20507306]) {
            assert!((*actual - expected).abs() <= 2.0e-5);
        }
        Ok(())
    }
}
