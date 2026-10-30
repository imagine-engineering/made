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

pub const HEUNPP2_SAMPLER_ID: &str = "heunpp2";
pub const HEUNPP2_FEATURE_ID: &str = "COMFY-MODEL-0188";
pub const HEUNPP2_SOURCE_ORDINAL: u16 = 5;
pub const HEUNPP2_NOISE_CONTRACT_ID: &str = EULER_CHURN_NOISE_CONTRACT_ID;

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: HEUNPP2_SAMPLER_ID,
    feature_id: HEUNPP2_FEATURE_ID,
    source_ordinal: HEUNPP2_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/heunpp2_comfy_model_0188",
    stochastic: true,
};

pub type HeunPp2Options = EulerOptions;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeunPp2DenoiserStage {
    Primary,
    Correction,
    Lookahead,
}

#[allow(clippy::too_many_arguments)]
pub fn sample_heunpp2<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    options: HeunPp2Options,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, HeunPp2DenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), HeunPp2SamplerError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != HEUNPP2_SAMPLER_ID {
        return Err(HeunPp2SamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let step_count = sigmas.len().saturating_sub(1);
    let initial_sigma = sigmas
        .first()
        .copied()
        .ok_or(SamplingError::ScheduleLength {
            expected: 2,
            actual: 0,
        })?;
    let terminal_sigma = sigmas
        .last()
        .copied()
        .ok_or(SamplingError::ScheduleLength {
            expected: 2,
            actual: 0,
        })?;
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("Heun++2 sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let seed = plan.seed();
    let generation_device = initial.descriptor().device();
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;

    validate_euler_noise_generation_device(generation_device)?;
    let mut noise_transaction = noise_request.open_transaction(
        HEUNPP2_NOISE_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::TorchSigned64,
        RngGenerationPlacement::Native(generation_device),
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();

    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = pair
            .first()
            .copied()
            .ok_or(SamplingError::Overflow("Heun++2 current sigma lookup"))?;
        let next_sigma = pair
            .get(1)
            .copied()
            .ok_or(SamplingError::Overflow("Heun++2 next sigma lookup"))?;
        let step_gamma = gamma(options, sigma, step_count);
        let sigma_hat = sigma * (step_gamma + 1.0);

        let current = session.current().clone();
        let churned = draw_and_apply_churn(
            backend,
            &current,
            sigma,
            sigma_hat,
            step_gamma,
            options.s_noise(),
            step,
            &mut noise_transaction,
            context,
        )?;
        checked_value(sigma_hat, step, "sigma hat", 0)?;
        if sigma_hat <= 0.0 {
            return Err(HeunPp2SamplerError::InvalidSigmaHat { step, sigma_hat });
        }

        let primary_denoised = evaluate_denoiser(
            backend,
            &churned,
            sigma_hat,
            step,
            HeunPp2DenoiserStage::Primary,
            context,
            &mut denoiser,
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
        let next = if next_sigma == terminal_sigma {
            advance_euler(backend, &churned, &primary_derivative, delta, step, context)?
        } else {
            let lookahead_sigma = sigmas
                .get(step + 2)
                .copied()
                .ok_or(HeunPp2SamplerError::MissingLookaheadSigma { step })?;
            let correction_input =
                advance_euler(backend, &churned, &primary_derivative, delta, step, context)?;
            let correction_denoised = evaluate_denoiser(
                backend,
                &correction_input,
                next_sigma,
                step,
                HeunPp2DenoiserStage::Correction,
                context,
                &mut denoiser,
            )?;
            let correction_derivative = derivative(
                backend,
                &correction_input,
                &correction_denoised,
                next_sigma,
                step,
                "correction derivative",
                context,
            )?;

            let weighted_derivative = if lookahead_sigma == terminal_sigma {
                weighted_heun_derivative(
                    backend,
                    &primary_derivative,
                    &correction_derivative,
                    initial_sigma,
                    next_sigma,
                    step,
                    context,
                )?
            } else {
                let lookahead_input = advance_euler(
                    backend,
                    &correction_input,
                    &correction_derivative,
                    lookahead_sigma - next_sigma,
                    step,
                    context,
                )?;
                let lookahead_denoised = evaluate_denoiser(
                    backend,
                    &lookahead_input,
                    lookahead_sigma,
                    step,
                    HeunPp2DenoiserStage::Lookahead,
                    context,
                    &mut denoiser,
                )?;
                let lookahead_derivative = derivative(
                    backend,
                    &lookahead_input,
                    &lookahead_denoised,
                    lookahead_sigma,
                    step,
                    "lookahead derivative",
                    context,
                )?;
                weighted_heunpp_derivative(
                    backend,
                    &primary_derivative,
                    &correction_derivative,
                    &lookahead_derivative,
                    initial_sigma,
                    next_sigma,
                    lookahead_sigma,
                    step,
                    context,
                )?
            };
            advance_euler(
                backend,
                &churned,
                &weighted_derivative,
                delta,
                step,
                context,
            )?
        };
        observed.commit(next, context.cancellation)?;
    }

    let sampling = session.finish()?;
    Ok((sampling, (noise_before, noise_transaction.commit())))
}

fn gamma(options: HeunPp2Options, sigma: f32, steps: usize) -> f32 {
    if options.s_tmin() <= sigma && sigma <= options.s_tmax() {
        (options.s_churn() / steps as f32).min(2.0_f32.sqrt() - 1.0)
    } else {
        0.0
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_and_apply_churn(
    backend: &CpuBackend,
    current: &Tensor,
    sigma: f32,
    sigma_hat: f32,
    gamma: f32,
    noise_scale: f32,
    step: usize,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, HeunPp2SamplerError> {
    let count = usize::try_from(current.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let normal = transaction.draw_normal(count, context.cancellation)?;
    let mut scaled_noise = backend.workspace_vec::<f32>(context, count)?;
    for (element, noise_value) in normal.into_iter().enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let noise_value = noise_value as f32 * noise_scale;
        checked_value(noise_value, step, "noise", element)?;
        scaled_noise.try_push(noise_value)?;
    }
    if gamma <= 0.0 {
        return Ok(current.clone());
    }

    let perturbation_scale = (sigma_hat * sigma_hat - sigma * sigma).sqrt();
    checked_value(perturbation_scale, step, "churn scale", 0)?;
    let current_values = tensor_to_f32(backend, current, context)?;
    let mut churned_values = backend.workspace_vec::<f32>(context, count)?;
    for (element, (current_value, noise_value)) in
        current_values.iter().zip(scaled_noise.iter()).enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let churned = noise_value.mul_add(perturbation_scale, *current_value);
        checked_value(churned, step, "churn", element)?;
        churned_values.try_push(churned)?;
    }
    tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &churned_values,
        context,
    )
    .map_err(HeunPp2SamplerError::TensorKernel)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_denoiser(
    backend: &CpuBackend,
    input: &Tensor,
    sigma: f32,
    step: usize,
    stage: HeunPp2DenoiserStage,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(&Tensor, f32, usize, HeunPp2DenoiserStage) -> Result<Tensor, String>,
) -> Result<Tensor, HeunPp2SamplerError> {
    let denoised =
        denoiser(input, sigma, step, stage).map_err(|reason| HeunPp2SamplerError::Denoiser {
            step,
            stage,
            reason,
        })?;
    if input.descriptor() != denoised.descriptor() {
        return Err(HeunPp2SamplerError::DenoiserContract { step, stage });
    }
    for (element, value) in tensor_to_f32(backend, &denoised, context)?
        .iter()
        .enumerate()
    {
        checked_value(*value, step, "denoiser", element)?;
    }
    Ok(denoised)
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
) -> Result<Tensor, HeunPp2SamplerError> {
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
    .map_err(HeunPp2SamplerError::TensorKernel)
}

#[allow(clippy::too_many_arguments)]
fn weighted_heun_derivative(
    backend: &CpuBackend,
    primary: &Tensor,
    correction: &Tensor,
    initial_sigma: f32,
    next_sigma: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, HeunPp2SamplerError> {
    let correction_weight = next_sigma / (2.0 * initial_sigma);
    let primary_weight = 1.0 - correction_weight;
    combine_derivatives(
        backend,
        primary,
        correction,
        None,
        [primary_weight, correction_weight, 0.0],
        step,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn weighted_heunpp_derivative(
    backend: &CpuBackend,
    primary: &Tensor,
    correction: &Tensor,
    lookahead: &Tensor,
    initial_sigma: f32,
    next_sigma: f32,
    lookahead_sigma: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, HeunPp2SamplerError> {
    let denominator = 3.0 * initial_sigma;
    let correction_weight = next_sigma / denominator;
    let lookahead_weight = lookahead_sigma / denominator;
    let primary_weight = 1.0 - correction_weight - lookahead_weight;
    combine_derivatives(
        backend,
        primary,
        correction,
        Some(lookahead),
        [primary_weight, correction_weight, lookahead_weight],
        step,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn combine_derivatives(
    backend: &CpuBackend,
    primary: &Tensor,
    correction: &Tensor,
    lookahead: Option<&Tensor>,
    weights: [f32; 3],
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, HeunPp2SamplerError> {
    let primary_values = tensor_to_f32(backend, primary, context)?;
    let correction_values = tensor_to_f32(backend, correction, context)?;
    let lookahead_values = lookahead
        .map(|value| tensor_to_f32(backend, value, context))
        .transpose()?;
    let mut combined_values = backend.workspace_vec::<f32>(context, primary_values.len())?;
    for (element, (primary_value, correction_value)) in primary_values
        .iter()
        .zip(correction_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let mut value = primary_value.mul_add(weights[0], correction_value * weights[1]);
        if let Some(lookahead_values) = &lookahead_values {
            let lookahead_value = lookahead_values
                .get(element)
                .ok_or(TensorError::ShapeOverflow)?;
            value = lookahead_value.mul_add(weights[2], value);
        }
        checked_value(value, step, "weighted derivative", element)?;
        combined_values.try_push(value)?;
    }
    tensor_from_f32(
        backend,
        primary.descriptor().shape(),
        &combined_values,
        context,
    )
    .map_err(HeunPp2SamplerError::TensorKernel)
}

fn checked_value(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<(), HeunPp2SamplerError> {
    if !value.is_finite() {
        return Err(HeunPp2SamplerError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum HeunPp2SamplerError {
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
    #[error("Heun++2 requires sampler identity `heunpp2`, got {0:?}")]
    WrongSampler(String),
    #[error("Heun++2 denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: HeunPp2DenoiserStage,
        reason: String,
    },
    #[error("Heun++2 denoiser output descriptor changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: HeunPp2DenoiserStage,
    },
    #[error("Heun++2 produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("Heun++2 produced invalid sigma hat {sigma_hat} at step {step}")]
    InvalidSigmaHat { step: usize, sigma_hat: f32 },
    #[error("Heun++2 schedule has no lookahead sigma at step {step}")]
    MissingLookaheadSigma { step: usize },
}
