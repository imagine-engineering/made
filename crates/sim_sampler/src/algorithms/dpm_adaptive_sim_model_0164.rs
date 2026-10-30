use crate::generated_dpm_solver::{
    DpmSolverEquationError, DpmSolverEvaluation, DpmSolverStage, DpmSolverStepError,
    dpm_solver_first_intermediate, dpm_solver_first_order, dpm_solver_second_order,
    dpm_solver_third_order,
};
use crate::{
    AdaptiveSamplingAttempt, AdaptiveSamplingProgress, AdaptiveSamplingSession,
    AdaptiveSamplingTrace, CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry,
    SamplingError, SamplingPlan, SchedulerError, SchedulerRegistry, standard_ancestral_step,
};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, CpuWorkspaceVec, DeviceId, ExecutionContext,
    RngCheckpoint, RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor,
    TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use thiserror::Error;

pub const DPM_ADAPTIVE_SAMPLER_ID: &str = "dpm_adaptive";
pub const DPM_ADAPTIVE_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPM_ADAPTIVE_SAMPLER_ID,
    feature_id: "COMFY-MODEL-0164",
    source_ordinal: 12,
    aliases: &[],
    implementation_module: "algorithms/dpm_adaptive_comfy_model_0164",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpmAdaptiveOptions {
    pub order: u8,
    pub relative_tolerance: f32,
    pub absolute_tolerance: f32,
    pub initial_step_size: f32,
    pub proportional_coefficient: f64,
    pub integral_coefficient: f64,
    pub derivative_coefficient: f64,
    pub acceptance_safety: f64,
    pub eta: f32,
    pub noise_scale: f32,
    pub attempt_limit: u32,
}

impl Default for DpmAdaptiveOptions {
    fn default() -> Self {
        Self {
            order: 3,
            relative_tolerance: 0.05,
            absolute_tolerance: 0.0078,
            initial_step_size: 0.05,
            proportional_coefficient: 0.0,
            integral_coefficient: 1.0,
            derivative_coefficient: 0.0,
            acceptance_safety: 0.81,
            eta: 0.0,
            noise_scale: 1.0,
            attempt_limit: 1_024,
        }
    }
}

impl DpmAdaptiveOptions {
    fn validate(self) -> Result<Self, DpmAdaptiveSamplerError> {
        if !matches!(self.order, 2 | 3) {
            return Err(DpmAdaptiveSamplerError::InvalidOrder(self.order));
        }
        for (name, value) in [
            ("relative tolerance", self.relative_tolerance),
            ("absolute tolerance", self.absolute_tolerance),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(DpmAdaptiveSamplerError::InvalidOption {
                    name,
                    value: f64::from(value),
                });
            }
        }
        if self.relative_tolerance == 0.0 && self.absolute_tolerance == 0.0 {
            return Err(DpmAdaptiveSamplerError::InvalidOption {
                name: "zero error tolerances cannot produce a finite normalized error",
                value: 0.0,
            });
        }
        validate_positive("initial step size", self.initial_step_size.abs())?;
        validate_finite("proportional coefficient", self.proportional_coefficient)?;
        validate_finite("integral coefficient", self.integral_coefficient)?;
        validate_finite("derivative coefficient", self.derivative_coefficient)?;
        if !self.acceptance_safety.is_finite() || self.acceptance_safety < 0.0 {
            return Err(DpmAdaptiveSamplerError::InvalidOption {
                name: "acceptance safety",
                value: self.acceptance_safety,
            });
        }
        if !self.eta.is_finite() {
            return Err(DpmAdaptiveSamplerError::InvalidOption {
                name: "eta",
                value: f64::from(self.eta),
            });
        }
        validate_finite("noise scale", f64::from(self.noise_scale))?;
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DpmAdaptiveEvaluationStage {
    Base,
    FirstIntermediate,
    SecondIntermediate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DpmAdaptiveEvaluation {
    pub attempt: u32,
    pub stage: DpmAdaptiveEvaluationStage,
}

#[derive(Clone, Debug)]
pub struct DpmAdaptiveResult {
    pub sampling: Option<AdaptiveSamplingTrace>,
    pub output: Tensor,
    pub noise_before: Option<RngCheckpoint>,
    pub noise_after: Option<RngCheckpoint>,
}

#[derive(Debug, Error)]
pub enum DpmAdaptiveSamplerError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error("DPM adaptive requires sampler identity `dpm_adaptive`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM adaptive order must be 2 or 3, got {0}")]
    InvalidOrder(u8),
    #[error("DPM adaptive {name} is invalid: {value}")]
    InvalidOption { name: &'static str, value: f64 },
    #[error("DPM adaptive requires a finite positive sigma endpoint, got {0}")]
    InvalidSigma(f32),
    #[error("DPM adaptive sigma schedule contains no positive terminal sigma")]
    MissingTerminalSigma,
    #[error("DPM adaptive denoiser failed during attempt {attempt}, stage {stage:?}: {reason}")]
    Denoiser {
        attempt: u32,
        stage: DpmAdaptiveEvaluationStage,
        reason: String,
    },
    #[error("DPM adaptive denoiser descriptor changed during attempt {attempt}, stage {stage:?}")]
    DenoiserContract {
        attempt: u32,
        stage: DpmAdaptiveEvaluationStage,
    },
    #[error(
        "DPM adaptive produced a non-finite {stage} value during attempt {attempt}, element {element}"
    )]
    NonFinite {
        attempt: u32,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM adaptive arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("DPM adaptive allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpm_adaptive<E>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    options: DpmAdaptiveOptions,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, DpmAdaptiveEvaluation) -> Result<Tensor, String>,
    mut callback: impl FnMut(&AdaptiveSamplingProgress, &Tensor, &Tensor) -> Result<(), E>,
) -> Result<DpmAdaptiveResult, DpmAdaptiveSamplerError>
where
    E: std::fmt::Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != DPM_ADAPTIVE_SAMPLER_ID {
        return Err(DpmAdaptiveSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    let options = options.validate()?;
    if sigmas.len() <= 1 {
        return Ok(DpmAdaptiveResult {
            output: initial,
            sampling: None,
            noise_before: None,
            noise_after: None,
        });
    }
    let initial_sigma = *sigmas
        .first()
        .ok_or(DpmAdaptiveSamplerError::MissingTerminalSigma)?;
    validate_sigma(initial_sigma)?;
    let last_sigma = *sigmas
        .last()
        .ok_or(DpmAdaptiveSamplerError::MissingTerminalSigma)?;
    let terminal_sigma = if last_sigma == 0.0 {
        *sigmas
            .get(sigmas.len().saturating_sub(2))
            .ok_or(DpmAdaptiveSamplerError::MissingTerminalSigma)?
    } else {
        last_sigma
    };
    validate_sigma(terminal_sigma)?;
    let evaluation_count = u32::from(options.order);
    let mut session = AdaptiveSamplingSession::new(
        plan.clone(),
        initial_sigma,
        terminal_sigma,
        initial,
        options.attempt_limit,
        evaluation_count,
    )?;
    let mut noise_transaction = noise_request.open_transaction(
        DPM_ADAPTIVE_NOISE_CONTRACT_ID,
        i128::from(plan.seed()),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();
    let mut controller = PidStepSizeController::new(options);
    let mut current_time = -initial_sigma.ln();
    let terminal_time = -terminal_sigma.ln();
    let mut previous_low = tensor_to_f32(backend, session.current(), context)?;

    while !session.is_complete() {
        context.check()?;
        let attempt = session.next_attempt(context.cancellation)?;
        let proposed_time = (current_time + controller.step_size() as f32).min(terminal_time);
        let proposed_sigma = (-proposed_time).exp();
        validate_sigma(proposed_sigma)?;
        let (solver_time, stochastic_scale) =
            ancestral_solver_target(current_time, proposed_time, terminal_time, options.eta)?;
        let current = session.current().clone();
        let current_values = tensor_to_f32(backend, &current, context)?;
        let base = evaluate(
            backend,
            &current,
            current_time,
            attempt,
            DpmAdaptiveEvaluationStage::Base,
            context,
            &mut denoiser,
        )?;
        let mut evaluation_tensors = Vec::new();
        evaluation_tensors
            .try_reserve_exact(usize::from(options.order))
            .map_err(|_| DpmAdaptiveSamplerError::OutOfMemory("denoiser evaluations"))?;
        evaluation_tensors.push(base.denoised.clone());

        let (proposed_low, proposed_high) = if options.order == 2 {
            let low_values = dpm_solver_first_order(
                backend,
                &current_values,
                &base.epsilon,
                current_time,
                solver_time,
                context,
            )
            .map_err(|error| map_solver_equation_error(error, attempt))?;
            let first = dpm_solver_first_intermediate(
                backend,
                &current,
                &current_values,
                &base.epsilon,
                current_time,
                solver_time,
                0.5,
                context,
                &mut |input, time, _evaluation| {
                    evaluate(
                        backend,
                        input,
                        time,
                        attempt,
                        DpmAdaptiveEvaluationStage::FirstIntermediate,
                        context,
                        &mut denoiser,
                    )
                },
            )
            .map_err(|error| map_solver_step_error(error, attempt))?;
            evaluation_tensors.push(first.evaluation.denoised.clone());
            let high_values = dpm_solver_second_order(
                backend,
                &current_values,
                &base.epsilon,
                &first.evaluation.epsilon,
                current_time,
                solver_time,
                0.5,
                context,
            )
            .map_err(|error| map_solver_equation_error(error, attempt))?;
            (
                tensor_from_f32(backend, current.descriptor().shape(), &low_values, context)?,
                tensor_from_f32(backend, current.descriptor().shape(), &high_values, context)?,
            )
        } else {
            let first = dpm_solver_first_intermediate(
                backend,
                &current,
                &current_values,
                &base.epsilon,
                current_time,
                solver_time,
                1.0 / 3.0,
                context,
                &mut |input, time, _evaluation| {
                    evaluate(
                        backend,
                        input,
                        time,
                        attempt,
                        DpmAdaptiveEvaluationStage::FirstIntermediate,
                        context,
                        &mut denoiser,
                    )
                },
            )
            .map_err(|error| map_solver_step_error(error, attempt))?;
            evaluation_tensors.push(first.evaluation.denoised.clone());
            let low_values = dpm_solver_second_order(
                backend,
                &current_values,
                &base.epsilon,
                &first.evaluation.epsilon,
                current_time,
                solver_time,
                1.0 / 3.0,
                context,
            )
            .map_err(|error| map_solver_equation_error(error, attempt))?;
            let third = dpm_solver_third_order(
                backend,
                &current,
                &current_values,
                &base.epsilon,
                &first.evaluation.epsilon,
                current_time,
                solver_time,
                context,
                &mut |input, time, _evaluation| {
                    evaluate(
                        backend,
                        input,
                        time,
                        attempt,
                        DpmAdaptiveEvaluationStage::SecondIntermediate,
                        context,
                        &mut denoiser,
                    )
                },
            )
            .map_err(|error| map_solver_step_error(error, attempt))?;
            evaluation_tensors.push(third.second_evaluation.denoised);
            (
                tensor_from_f32(backend, current.descriptor().shape(), &low_values, context)?,
                tensor_from_f32(
                    backend,
                    current.descriptor().shape(),
                    &third.values,
                    context,
                )?,
            )
        };
        let low_values = tensor_to_f32(backend, &proposed_low, context)?;
        let high_values = tensor_to_f32(backend, &proposed_high, context)?;
        let error = normalized_error(
            &low_values,
            &high_values,
            &previous_low,
            options,
            attempt,
            context,
        )?;
        let accepted = controller.propose_step(error);
        let mut stochastic_noise = None;
        let accepted_next = if accepted {
            previous_low = copy_values(backend, &low_values, context)?;
            let noise = draw_noise(backend, &current, &mut noise_transaction, context)?;
            let next = add_scaled_noise(
                backend,
                &proposed_high,
                &noise,
                stochastic_scale * options.noise_scale,
                attempt,
                context,
            )?;
            stochastic_noise = Some(noise);
            Some(next)
        } else {
            None
        };
        session.commit_attempt(
            AdaptiveSamplingAttempt {
                proposed_sigma,
                base_denoised: base.denoised,
                evaluations: evaluation_tensors,
                proposed_low,
                proposed_high,
                stochastic_noise,
                accepted_next,
                error,
                next_step_size: controller.step_size() as f32,
            },
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;
        if accepted {
            current_time = proposed_time;
        }
    }
    let output = session.current().clone();
    let sampling = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok(DpmAdaptiveResult {
        sampling: Some(sampling),
        output,
        noise_before: Some(noise_before),
        noise_after: Some(noise_after),
    })
}

fn evaluate(
    backend: &CpuBackend,
    input: &Tensor,
    time: f32,
    attempt: u32,
    stage: DpmAdaptiveEvaluationStage,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(&Tensor, f32, DpmAdaptiveEvaluation) -> Result<Tensor, String>,
) -> Result<DpmSolverEvaluation, DpmAdaptiveSamplerError> {
    context.check()?;
    let sigma = (-time).exp();
    validate_sigma(sigma)?;
    let denoised =
        denoiser(input, sigma, DpmAdaptiveEvaluation { attempt, stage }).map_err(|reason| {
            DpmAdaptiveSamplerError::Denoiser {
                attempt,
                stage,
                reason,
            }
        })?;
    if input.descriptor() != denoised.descriptor() {
        return Err(DpmAdaptiveSamplerError::DenoiserContract { attempt, stage });
    }
    let input_values = tensor_to_f32(backend, input, context)?;
    let denoised_values = tensor_to_f32(backend, &denoised, context)?;
    let mut epsilon = backend.workspace_vec(context, input_values.len())?;
    for (element, (input, denoised)) in input_values.iter().zip(denoised_values.iter()).enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = (input - denoised) / sigma;
        ensure_finite(value, attempt, "epsilon", element)?;
        epsilon.try_push(value)?;
    }
    Ok(DpmSolverEvaluation { denoised, epsilon })
}

fn map_solver_step_error(
    error: DpmSolverStepError<DpmAdaptiveSamplerError>,
    attempt: u32,
) -> DpmAdaptiveSamplerError {
    match error {
        DpmSolverStepError::Equation(error) => map_solver_equation_error(error, attempt),
        DpmSolverStepError::Evaluation(error) => error,
    }
}

fn map_solver_equation_error(
    error: DpmSolverEquationError,
    attempt: u32,
) -> DpmAdaptiveSamplerError {
    match error {
        DpmSolverEquationError::Tensor(error) => DpmAdaptiveSamplerError::Tensor(error),
        DpmSolverEquationError::TensorKernel(error) => DpmAdaptiveSamplerError::TensorKernel(error),
        DpmSolverEquationError::Shape(stage) => {
            DpmAdaptiveSamplerError::Overflow(adaptive_solver_stage(stage))
        }
        DpmSolverEquationError::NonFinite { stage, element } => {
            DpmAdaptiveSamplerError::NonFinite {
                attempt,
                stage: adaptive_solver_stage(stage),
                element,
            }
        }
    }
}

fn adaptive_solver_stage(stage: DpmSolverStage) -> &'static str {
    match stage {
        DpmSolverStage::FirstOrder => "first-order proposal",
        DpmSolverStage::FirstIntermediate => "first intermediate",
        DpmSolverStage::SecondDifference => "second-order epsilon",
        DpmSolverStage::SecondOrder => "second-order proposal",
        DpmSolverStage::ThirdFirstDifference => "third-order first epsilon",
        DpmSolverStage::ThirdIntermediate => "third-order intermediate",
        DpmSolverStage::ThirdSecondDifference => "third-order second epsilon",
        DpmSolverStage::ThirdOrder => "third-order proposal",
    }
}

fn normalized_error(
    low: &[f32],
    high: &[f32],
    previous_low: &[f32],
    options: DpmAdaptiveOptions,
    attempt: u32,
    context: &ExecutionContext<'_>,
) -> Result<f32, DpmAdaptiveSamplerError> {
    if low.len() != high.len() || low.len() != previous_low.len() || low.is_empty() {
        return Err(DpmAdaptiveSamplerError::Overflow("adaptive error shape"));
    }
    let mut sum = 0.0_f32;
    for (element, ((low, high), previous)) in low
        .iter()
        .zip(high.iter())
        .zip(previous_low.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let scale = options
            .absolute_tolerance
            .max(options.relative_tolerance * low.abs().max(previous.abs()));
        let normalized = (low - high) / scale;
        ensure_finite(normalized, attempt, "normalized error", element)?;
        sum += normalized * normalized;
    }
    let count = u32::try_from(low.len())
        .map_err(|_| DpmAdaptiveSamplerError::Overflow("error element count"))?;
    let error = sum.sqrt() / (count as f32).sqrt();
    ensure_finite(error, attempt, "error norm", 0)?;
    Ok(error)
}

fn ancestral_solver_target(
    current_time: f32,
    proposed_time: f32,
    terminal_time: f32,
    eta: f32,
) -> Result<(f32, f32), DpmAdaptiveSamplerError> {
    if eta == 0.0 {
        return Ok((proposed_time, 0.0));
    }
    let sigma_from = (-current_time).exp();
    let sigma_to = (-proposed_time).exp();
    let (sigma_down, _) = standard_ancestral_step(sigma_from, sigma_to, eta).map_err(|_| {
        DpmAdaptiveSamplerError::InvalidOption {
            name: "ancestral target",
            value: f64::from(proposed_time),
        }
    })?;
    let solver_time = if sigma_down == 0.0 {
        terminal_time
    } else {
        validate_sigma(sigma_down)?;
        (-sigma_down.ln()).min(terminal_time)
    };
    let stochastic_scale = (sigma_to.powi(2) - (-solver_time).exp().powi(2))
        .max(0.0)
        .sqrt();
    if !solver_time.is_finite() || !stochastic_scale.is_finite() {
        return Err(DpmAdaptiveSamplerError::InvalidOption {
            name: "ancestral target",
            value: f64::from(solver_time),
        });
    }
    Ok((solver_time, stochastic_scale))
}

fn draw_noise(
    backend: &CpuBackend,
    template: &Tensor,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, DpmAdaptiveSamplerError> {
    let count = usize::try_from(template.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let values = transaction.draw_normal(count, context.cancellation)?;
    let mut output = backend.workspace_vec::<f32>(context, count)?;
    for (index, value) in values.into_iter().enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        output.try_push(value as f32)?;
    }
    Ok(tensor_from_f32(
        backend,
        template.descriptor().shape(),
        &output,
        context,
    )?)
}

fn add_scaled_noise(
    backend: &CpuBackend,
    proposed: &Tensor,
    noise: &Tensor,
    scale: f32,
    attempt: u32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, DpmAdaptiveSamplerError> {
    let proposed_values = tensor_to_f32(backend, proposed, context)?;
    let noise_values = tensor_to_f32(backend, noise, context)?;
    let values = combine(
        backend,
        &proposed_values,
        &[(&noise_values, scale)],
        attempt,
        "stochastic update",
        context,
    )?;
    Ok(tensor_from_f32(
        backend,
        proposed.descriptor().shape(),
        &values,
        context,
    )?)
}

fn combine(
    backend: &CpuBackend,
    base: &[f32],
    terms: &[(&[f32], f32)],
    attempt: u32,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, DpmAdaptiveSamplerError> {
    if terms.iter().any(|(values, _)| values.len() != base.len()) {
        return Err(DpmAdaptiveSamplerError::Overflow(stage));
    }
    let mut output = backend.workspace_vec(context, base.len())?;
    for element in 0..base.len() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let mut value = *base
            .get(element)
            .ok_or(DpmAdaptiveSamplerError::Overflow(stage))?;
        for (values, coefficient) in terms {
            let term = values
                .get(element)
                .ok_or(DpmAdaptiveSamplerError::Overflow(stage))?;
            value += coefficient * term;
        }
        ensure_finite(value, attempt, stage, element)?;
        output.try_push(value)?;
    }
    Ok(output)
}

fn copy_values(
    backend: &CpuBackend,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, DpmAdaptiveSamplerError> {
    let mut output = backend.workspace_vec(context, values.len())?;
    for (index, value) in values.iter().copied().enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        output.try_push(value)?;
    }
    Ok(output)
}

struct PidStepSizeController {
    step_size: f64,
    first: f64,
    second: f64,
    third: f64,
    has_errors: bool,
    first_exponent: f64,
    second_exponent: f64,
    third_exponent: f64,
    acceptance_safety: f64,
}

impl PidStepSizeController {
    fn new(options: DpmAdaptiveOptions) -> Self {
        let order = if options.eta == 0.0 {
            f64::from(options.order)
        } else {
            1.5
        };
        Self {
            step_size: f64::from(options.initial_step_size.abs()),
            first: 0.0,
            second: 0.0,
            third: 0.0,
            has_errors: false,
            first_exponent: (options.proportional_coefficient
                + options.integral_coefficient
                + options.derivative_coefficient)
                / order,
            second_exponent: -(options.proportional_coefficient
                + 2.0 * options.derivative_coefficient)
                / order,
            third_exponent: options.derivative_coefficient / order,
            acceptance_safety: options.acceptance_safety,
        }
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn propose_step(&mut self, error: f32) -> bool {
        let inverse_error = 1.0 / (f64::from(error) + 1.0e-8);
        if !self.has_errors {
            self.first = inverse_error;
            self.second = inverse_error;
            self.third = inverse_error;
            self.has_errors = true;
        }
        self.first = inverse_error;
        let raw_factor = self.first.powf(self.first_exponent)
            * self.second.powf(self.second_exponent)
            * self.third.powf(self.third_exponent);
        let factor = 1.0 + (raw_factor - 1.0).atan();
        let accepted = factor >= self.acceptance_safety;
        if accepted {
            self.third = self.second;
            self.second = self.first;
        }
        self.step_size *= factor;
        accepted
    }
}

fn validate_sigma(sigma: f32) -> Result<(), DpmAdaptiveSamplerError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(DpmAdaptiveSamplerError::InvalidSigma(sigma));
    }
    Ok(())
}

fn validate_positive(name: &'static str, value: f32) -> Result<(), DpmAdaptiveSamplerError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(DpmAdaptiveSamplerError::InvalidOption {
            name,
            value: f64::from(value),
        });
    }
    Ok(())
}

fn validate_finite(name: &'static str, value: f64) -> Result<(), DpmAdaptiveSamplerError> {
    if !value.is_finite() {
        return Err(DpmAdaptiveSamplerError::InvalidOption { name, value });
    }
    Ok(())
}

fn ensure_finite(
    value: f32,
    attempt: u32,
    stage: &'static str,
    element: usize,
) -> Result<(), DpmAdaptiveSamplerError> {
    if !value.is_finite() {
        return Err(DpmAdaptiveSamplerError::NonFinite {
            attempt,
            stage,
            element,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ancestral_solver_target;

    #[test]
    fn high_eta_zero_sigma_down_uses_terminal_time() -> Result<(), Box<dyn std::error::Error>> {
        let current_time = -2.0_f32.ln();
        let proposed_time = -1.0_f32.ln();
        let terminal_time = -0.5_f32.ln();
        let (solver_time, stochastic_scale) =
            ancestral_solver_target(current_time, proposed_time, terminal_time, 100.0)?;
        assert_eq!(solver_time.to_bits(), terminal_time.to_bits());
        assert!((stochastic_scale - 0.75_f32.sqrt()).abs() <= 1.0e-6);
        Ok(())
    }
}
