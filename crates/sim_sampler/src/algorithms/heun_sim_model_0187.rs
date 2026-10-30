use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProgress, SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    generated_native_diffusion::{
        EULER_CHURN_NOISE_CONTRACT_ID, EulerOptions, NativeDiffusionSamplerError, advance_euler,
        observe_euler_denoised, validate_euler_noise_generation_device,
    },
};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, ExecutionContext, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const HEUN_SAMPLER_ID: &str = "heun";
pub const HEUN_FEATURE_ID: &str = "COMFY-MODEL-0187";
pub const HEUN_SOURCE_ORDINAL: u16 = 4;
pub const HEUN_CHURN_NOISE_CONTRACT_ID: &str = EULER_CHURN_NOISE_CONTRACT_ID;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: HEUN_SAMPLER_ID,
    feature_id: HEUN_FEATURE_ID,
    source_ordinal: HEUN_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/heun_comfy_model_0187",
    stochastic: true,
};

pub type HeunOptions = EulerOptions;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeunDenoiserStage {
    Primary,
    Correction,
}

#[allow(clippy::too_many_arguments)]
pub fn sample_heun<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    options: HeunOptions,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, HeunDenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), HeunSamplerError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != HEUN_SAMPLER_ID {
        return Err(HeunSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let step_count = sigmas.len().saturating_sub(1);
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("Heun sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let seed = plan.seed();
    let generation_device = initial.descriptor().device();
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;

    let churn_enabled = sigmas
        .iter()
        .take(step_count)
        .copied()
        .any(|sigma| gamma(options, sigma, step_count) > 0.0);
    let mut noise_transaction = if churn_enabled {
        validate_euler_noise_generation_device(generation_device)?;
        Some(noise_request.open_transaction(
            HEUN_CHURN_NOISE_CONTRACT_ID,
            i128::from(seed),
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(generation_device),
            None,
            context.cancellation,
        )?)
    } else {
        None
    };
    let noise_before = noise_transaction
        .as_ref()
        .map(CompatibilityRngTransaction::checkpoint);

    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = pair
            .first()
            .copied()
            .ok_or(SamplingError::Overflow("Heun current sigma lookup"))?;
        let next_sigma = pair
            .get(1)
            .copied()
            .ok_or(SamplingError::Overflow("Heun next sigma lookup"))?;
        let step_gamma = gamma(options, sigma, step_count);
        let sigma_hat = sigma * (step_gamma + 1.0);
        if !sigma_hat.is_finite() || sigma_hat <= 0.0 {
            return Err(HeunSamplerError::NonFinite {
                step,
                stage: "sigma hat",
                element: 0,
            });
        }

        let current = session.current().clone();
        let churned = if step_gamma > 0.0 {
            let transaction = noise_transaction
                .as_mut()
                .ok_or(HeunSamplerError::MissingNoiseTransaction { step })?;
            apply_churn(
                backend,
                &current,
                sigma,
                sigma_hat,
                options.s_noise(),
                step,
                transaction,
                context,
            )?
        } else {
            current
        };
        let primary_denoised = denoiser(&churned, sigma_hat, step, HeunDenoiserStage::Primary)
            .map_err(|reason| HeunSamplerError::Denoiser {
                step,
                stage: HeunDenoiserStage::Primary,
                reason,
            })?;
        validate_denoiser_contract(
            &churned,
            &primary_denoised,
            step,
            HeunDenoiserStage::Primary,
        )?;
        validate_finite_tensor(
            backend,
            &primary_denoised,
            step,
            "primary denoiser",
            context,
        )?;
        let primary_derivative = derivative(
            backend,
            &churned,
            &primary_denoised,
            sigma_hat,
            step,
            "primary derivative",
            context,
        )?;
        let observed = observe_euler_denoised(
            &mut session,
            &churned,
            primary_denoised,
            sigma_hat,
            context,
            &mut callback,
        )?;

        let delta = next_sigma - sigma_hat;
        let next = if next_sigma == 0.0 {
            advance_euler(backend, &churned, &primary_derivative, delta, step, context)?
        } else {
            let predicted =
                advance_euler(backend, &churned, &primary_derivative, delta, step, context)?;
            let correction_denoised =
                denoiser(&predicted, next_sigma, step, HeunDenoiserStage::Correction).map_err(
                    |reason| HeunSamplerError::Denoiser {
                        step,
                        stage: HeunDenoiserStage::Correction,
                        reason,
                    },
                )?;
            validate_denoiser_contract(
                &predicted,
                &correction_denoised,
                step,
                HeunDenoiserStage::Correction,
            )?;
            validate_finite_tensor(
                backend,
                &correction_denoised,
                step,
                "correction denoiser",
                context,
            )?;
            let correction_derivative = derivative(
                backend,
                &predicted,
                &correction_denoised,
                next_sigma,
                step,
                "correction derivative",
                context,
            )?;
            let average_derivative = average_derivatives(
                backend,
                &primary_derivative,
                &correction_derivative,
                step,
                context,
            )?;
            advance_euler(backend, &churned, &average_derivative, delta, step, context)?
        };
        observed.commit(next, context.cancellation)?;
    }

    let sampling = session.finish()?;
    let checkpoints = match (noise_before, noise_transaction) {
        (Some(before), Some(transaction)) => Some((before, transaction.commit())),
        (None, None) => None,
        _ => {
            return Err(HeunSamplerError::MissingNoiseTransaction { step: step_count });
        }
    };
    Ok((sampling, checkpoints))
}

fn gamma(options: HeunOptions, sigma: f32, steps: usize) -> f32 {
    if options.s_churn() > 0.0 && options.s_tmin() <= sigma && sigma <= options.s_tmax() {
        (options.s_churn() / steps as f32).min(2.0_f32.sqrt() - 1.0)
    } else {
        0.0
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_churn(
    backend: &CpuBackend,
    current: &Tensor,
    sigma: f32,
    sigma_hat: f32,
    noise_scale: f32,
    step: usize,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, HeunSamplerError> {
    let count = usize::try_from(current.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let normal = transaction.draw_normal(count, context.cancellation)?;
    let current_values = tensor_to_f32(backend, current, context)?;
    let perturbation_scale = (sigma_hat * sigma_hat - sigma * sigma).sqrt();
    let mut churned_values = backend.workspace_vec::<f32>(context, count)?;
    for (element, (current_value, noise_value)) in current_values.iter().zip(normal).enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let noise_value = noise_value as f32 * noise_scale;
        let churned = noise_value.mul_add(perturbation_scale, *current_value);
        checked_value(noise_value, step, "noise", element)?;
        checked_value(churned, step, "churn", element)?;
        churned_values.try_push(churned)?;
    }
    tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &churned_values,
        context,
    )
    .map_err(HeunSamplerError::TensorKernel)
}

#[allow(clippy::too_many_arguments)]
fn derivative(
    backend: &CpuBackend,
    input: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, HeunSamplerError> {
    let input_values = tensor_to_f32(backend, input, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let mut derivative_values = backend.workspace_vec::<f32>(context, input_values.len())?;
    for (element, (input_value, denoised_value)) in
        input_values.iter().zip(denoised_values.iter()).enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = (input_value - denoised_value) / sigma;
        checked_value(value, step, stage, element)?;
        derivative_values.try_push(value)?;
    }
    tensor_from_f32(
        backend,
        input.descriptor().shape(),
        &derivative_values,
        context,
    )
    .map_err(HeunSamplerError::TensorKernel)
}

fn average_derivatives(
    backend: &CpuBackend,
    primary: &Tensor,
    correction: &Tensor,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, HeunSamplerError> {
    let primary_values = tensor_to_f32(backend, primary, context)?;
    let correction_values = tensor_to_f32(backend, correction, context)?;
    let mut average_values = backend.workspace_vec::<f32>(context, primary_values.len())?;
    for (element, (primary_value, correction_value)) in primary_values
        .iter()
        .zip(correction_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = (primary_value + correction_value) * 0.5;
        checked_value(value, step, "average derivative", element)?;
        average_values.try_push(value)?;
    }
    tensor_from_f32(
        backend,
        primary.descriptor().shape(),
        &average_values,
        context,
    )
    .map_err(HeunSamplerError::TensorKernel)
}

fn validate_denoiser_contract(
    input: &Tensor,
    denoised: &Tensor,
    step: usize,
    stage: HeunDenoiserStage,
) -> Result<(), HeunSamplerError> {
    if input.descriptor() != denoised.descriptor() {
        return Err(HeunSamplerError::DenoiserContract { step, stage });
    }
    Ok(())
}

fn validate_finite_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), HeunSamplerError> {
    for (element, value) in tensor_to_f32(backend, tensor, context)?.iter().enumerate() {
        checked_value(*value, step, stage, element)?;
    }
    Ok(())
}

fn checked_value(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<(), HeunSamplerError> {
    if !value.is_finite() {
        return Err(HeunSamplerError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum HeunSamplerError {
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error(transparent)]
    EulerFoundation(#[from] NativeDiffusionSamplerError),
    #[error("Heun requires sampler identity `heun`, got {0:?}")]
    WrongSampler(String),
    #[error("Heun denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: HeunDenoiserStage,
        reason: String,
    },
    #[error("Heun denoiser output descriptor changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: HeunDenoiserStage,
    },
    #[error("Heun produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("Heun churn at step {step} has no canonical RNG transaction")]
    MissingNoiseTransaction { step: usize },
}
