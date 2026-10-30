use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProfileError, SamplingProgress, SamplingSession, SamplingTrace,
    SchedulerError, SchedulerRegistry,
    generated_native_diffusion::validate_euler_noise_generation_device,
};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const SA_SOLVER_SAMPLER_ID: &str = "sa_solver";
pub const SA_SOLVER_FEATURE_ID: &str = "COMFY-MODEL-0197";
pub const SA_SOLVER_SOURCE_ORDINAL: u16 = 39;
pub const SA_SOLVER_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: SA_SOLVER_SAMPLER_ID,
    feature_id: SA_SOLVER_FEATURE_ID,
    source_ordinal: SA_SOLVER_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/sa_solver_comfy_model_0197",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaSolverOptions {
    noise_scale: f32,
    predictor_order: usize,
    corrector_order: usize,
    simple_order_2: bool,
}

impl SaSolverOptions {
    pub fn new(
        noise_scale: f32,
        predictor_order: usize,
        corrector_order: usize,
        simple_order_2: bool,
    ) -> Result<Self, SaSolverError> {
        if !noise_scale.is_finite() {
            return Err(SaSolverError::InvalidOption {
                name: "s_noise",
                value: noise_scale.to_string(),
            });
        }
        if predictor_order == 0 {
            return Err(SaSolverError::InvalidOption {
                name: "predictor_order",
                value: predictor_order.to_string(),
            });
        }
        Ok(Self {
            noise_scale,
            predictor_order,
            corrector_order,
            simple_order_2,
        })
    }

    pub const fn source_defaults() -> Self {
        Self {
            noise_scale: 1.0,
            predictor_order: 3,
            corrector_order: 4,
            simple_order_2: false,
        }
    }

    pub const fn noise_scale(self) -> f32 {
        self.noise_scale
    }

    pub const fn predictor_order(self) -> usize {
        self.predictor_order
    }

    pub const fn corrector_order(self) -> usize {
        self.corrector_order
    }

    pub const fn simple_order_2(self) -> bool {
        self.simple_order_2
    }
}

impl Default for SaSolverOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaSolverEvaluation {
    Predictor,
    Corrected,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaSolverFamilyOptions {
    solver: SaSolverOptions,
    use_pece: bool,
}

impl SaSolverFamilyOptions {
    pub const fn new(solver: SaSolverOptions, use_pece: bool) -> Self {
        Self { solver, use_pece }
    }

    pub const fn solver(self) -> SaSolverOptions {
        self.solver
    }

    pub const fn use_pece(self) -> bool {
        self.use_pece
    }
}

#[derive(Debug, Error)]
pub enum SaSolverError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    SamplingProfile(#[from] SamplingProfileError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error("SA-Solver family requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("SA-Solver option {name} is invalid: {value}")]
    InvalidOption { name: &'static str, value: String },
    #[error("SA-Solver denoiser failed at step {step} during {evaluation:?}: {reason}")]
    Denoiser {
        step: usize,
        evaluation: SaSolverEvaluation,
        reason: String,
    },
    #[error("SA-Solver denoiser descriptor changed at step {step} during {evaluation:?}")]
    DenoiserContract {
        step: usize,
        evaluation: SaSolverEvaluation,
    },
    #[error("SA-Solver tau function failed at step {step}: {reason}")]
    TauFunction { step: usize, reason: String },
    #[error("SA-Solver coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("SA-Solver coefficient system is singular at step {step}, order {order}")]
    SingularSystem { step: usize, order: usize },
    #[error("SA-Solver produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("SA-Solver history is unavailable at step {step}")]
    MissingHistory { step: usize },
    #[error("SA-Solver allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("SA-Solver index arithmetic overflowed for {0}")]
    Overflow(&'static str),
    #[error("native SA-Solver noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
}

#[allow(clippy::too_many_arguments)]
pub fn sample_sa_solver<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: SaSolverOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), SaSolverError>
where
    CallbackError: Display,
{
    if sigmas.len() <= 1 {
        return sample_sa_solver_family(
            backend,
            plan,
            profile,
            SA_SOLVER_SAMPLER_ID,
            initial,
            sigmas,
            noise_request,
            SaSolverFamilyOptions::new(options, false),
            context,
            |_sigma, _step| Ok(0.0),
            |input, sigma, step, _evaluation| denoiser(input, sigma, step),
            callback,
        );
    }
    let (start_sigma, end_sigma) = source_default_tau_interval(profile)?;
    sample_sa_solver_family(
        backend,
        plan,
        profile,
        SA_SOLVER_SAMPLER_ID,
        initial,
        sigmas,
        noise_request,
        SaSolverFamilyOptions::new(options, false),
        context,
        move |sigma, _step| {
            Ok(if start_sigma >= sigma && sigma >= end_sigma {
                1.0
            } else {
                0.0
            })
        },
        |input, sigma, step, _evaluation| denoiser(input, sigma, step),
        callback,
    )
}

pub fn source_default_tau_interval(
    profile: &impl SamplingProfile,
) -> Result<(f32, f32), SaSolverError> {
    let final_index = profile
        .sigma_count()
        .checked_sub(1)
        .ok_or(SaSolverError::InvalidOption {
            name: "sampling_profile",
            value: "empty sigma grid".to_owned(),
        })?;
    let final_time = final_index as f32;
    let start_sigma = profile.sigma_at_model_time(final_time * 0.8)?;
    let end_sigma = profile.sigma_at_model_time(final_time * 0.2)?;
    if !start_sigma.is_finite()
        || !end_sigma.is_finite()
        || start_sigma < end_sigma
        || end_sigma < 0.0
    {
        return Err(SaSolverError::InvalidOption {
            name: "tau_interval",
            value: format!("{start_sigma}..{end_sigma}"),
        });
    }
    Ok((start_sigma, end_sigma))
}

#[allow(clippy::too_many_arguments)]
pub fn sample_sa_solver_family<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    expected_sampler: &'static str,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    family_options: SaSolverFamilyOptions,
    context: &ExecutionContext<'_>,
    mut tau_function: impl FnMut(f32, usize) -> Result<f32, String>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, SaSolverEvaluation) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), SaSolverError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        profile.identity(),
    )?;
    if plan.sampler().as_str() != expected_sampler {
        return Err(SaSolverError::WrongSampler {
            expected: expected_sampler,
            actual: plan.sampler().as_str().to_owned(),
        });
    }

    if sigmas.len() <= 1 {
        let seed = plan.seed();
        let generation_device = initial.descriptor().device();
        let (seed_transform, generation_placement) = noise_profile(generation_device);
        let noise_transaction = noise_request.open_transaction(
            SA_SOLVER_NOISE_CONTRACT_ID,
            i128::from(seed),
            seed_transform,
            generation_placement,
            None,
            context.cancellation,
        )?;
        let noise_before = noise_transaction.checkpoint();
        let mut schedule = Vec::new();
        schedule
            .try_reserve_exact(sigmas.len())
            .map_err(|_| SaSolverError::OutOfMemory("short sigma schedule"))?;
        schedule.extend_from_slice(sigmas);
        let denoiser_evaluations = Vec::new();
        let mut latents = Vec::new();
        latents
            .try_reserve_exact(1)
            .map_err(|_| SaSolverError::OutOfMemory("short latent trace"))?;
        latents.push(initial);
        return Ok((
            SamplingTrace {
                sigmas: schedule,
                denoiser_evaluations,
                latents,
            },
            (noise_before, noise_transaction.commit()),
        ));
    }

    let options = family_options.solver();
    let effective_noise_scale = profile.scale_sampler_noise(options.noise_scale())?;
    let generation_device = initial.descriptor().device();
    validate_euler_noise_generation_device(generation_device).map_err(|error| {
        SaSolverError::DeviceUnavailable {
            device: generation_device,
            reason: error.to_string(),
        }
    })?;
    let mut adjusted_sigmas = Vec::new();
    adjusted_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SaSolverError::OutOfMemory("adjusted sigma schedule"))?;
    adjusted_sigmas.extend_from_slice(sigmas);
    profile.adjust_first_sigma_for_snr(&mut adjusted_sigmas)?;
    let lambdas = half_log_snr_schedule(profile, &adjusted_sigmas)?;
    let seed = plan.seed();
    let mut session = SamplingSession::new(plan, adjusted_sigmas.clone(), initial.clone())?;
    let (seed_transform, generation_placement) = noise_profile(generation_device);
    let mut noise_transaction = noise_request.open_transaction(
        SA_SOLVER_NOISE_CONTRACT_ID,
        i128::from(seed),
        seed_transform,
        generation_placement,
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();
    let max_used_order = options.predictor_order().max(options.corrector_order());
    let lower_order_to_end = adjusted_sigmas.last().copied() == Some(0.0);
    let mut prediction_history = Vec::new();
    prediction_history
        .try_reserve_exact(max_used_order.min(adjusted_sigmas.len()))
        .map_err(|_| SaSolverError::OutOfMemory("prediction history"))?;
    let mut corrected_state = initial;
    let mut previous_step_size = 0.0_f32;
    let mut previous_tau = 0.0_f32;
    let mut previous_noise: Option<Tensor> = None;

    for (step, sigma_pair) in adjusted_sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = checked_positive(step, "sigma", sigma_pair[0])?;
        let next_sigma = checked_nonnegative(step, "next sigma", sigma_pair[1])?;
        let predicted_state = session.current().clone();
        let mut denoised = evaluate_denoiser(
            &mut denoiser,
            &predicted_state,
            sigma,
            step,
            SaSolverEvaluation::Predictor,
        )?;
        let observed = session.observe_step(
            &predicted_state,
            denoised.clone(),
            context.cancellation,
            |progress, current, output| callback(progress, current, output),
        )?;
        prediction_history.push(denoised.clone());
        if prediction_history.len() > max_used_order {
            prediction_history.remove(0);
        }

        let remaining_predictor_order = adjusted_sigmas
            .len()
            .checked_sub(2)
            .and_then(|value| value.checked_sub(step))
            .ok_or(SaSolverError::Overflow("remaining predictor order"))?;
        let remaining_corrector_order = adjusted_sigmas
            .len()
            .checked_sub(1)
            .and_then(|value| value.checked_sub(step))
            .ok_or(SaSolverError::Overflow("remaining corrector order"))?;
        let mut predictor_order_used = options.predictor_order().min(prediction_history.len());
        let mut corrector_order_used =
            if step == 0 || (next_sigma == 0.0 && !family_options.use_pece()) {
                0
            } else {
                options.corrector_order().min(prediction_history.len())
            };
        if lower_order_to_end {
            predictor_order_used = predictor_order_used.min(remaining_predictor_order);
            corrector_order_used = corrector_order_used.min(remaining_corrector_order);
        }

        let corrected = if corrector_order_used == 0 {
            predicted_state.clone()
        } else {
            let lambda_start_index = step
                .checked_sub(corrector_order_used - 1)
                .ok_or(SaSolverError::MissingHistory { step })?;
            let lambda_source_index = step
                .checked_sub(1)
                .ok_or(SaSolverError::MissingHistory { step })?;
            let coefficients = stochastic_adams_coefficients(
                sigma,
                lambdas
                    .get(lambda_start_index..=step)
                    .ok_or(SaSolverError::MissingHistory { step })?,
                schedule_value(
                    &lambdas,
                    lambda_source_index,
                    step,
                    "corrector source lambda",
                )?,
                schedule_value(&lambdas, step, step, "corrector target lambda")?,
                previous_tau,
                options.simple_order_2(),
                true,
                step,
            )?;
            let correction = weighted_history(
                backend,
                &prediction_history,
                corrector_order_used,
                &coefficients,
                step,
                context,
            )?;
            let state_scale = checked_coefficient(
                step,
                "corrector state scale",
                sigma / adjusted_sigmas[step - 1]
                    * (-(previous_tau * previous_tau) * previous_step_size).exp(),
            )?;
            let mut corrected = combine_state(
                backend,
                &corrected_state,
                state_scale,
                &correction,
                step,
                "corrector",
                context,
            )?;
            if previous_tau > 0.0 && effective_noise_scale > 0.0 {
                corrected = add_tensor(
                    backend,
                    &corrected,
                    previous_noise
                        .as_ref()
                        .ok_or(SaSolverError::MissingHistory { step })?,
                    step,
                    "corrector noise",
                    context,
                )?;
            }
            corrected
        };

        if family_options.use_pece() && corrector_order_used > 0 {
            denoised = evaluate_denoiser(
                &mut denoiser,
                &corrected,
                sigma,
                step,
                SaSolverEvaluation::Corrected,
            )?;
            let history_slot = prediction_history
                .last_mut()
                .ok_or(SaSolverError::MissingHistory { step })?;
            *history_slot = denoised.clone();
        }

        let next_predicted = if next_sigma == 0.0 {
            previous_noise = None;
            let terminal_values = tensor_to_f32(backend, &denoised, context)?;
            for (element, value) in terminal_values.iter().copied().enumerate() {
                checked_element(value, step, "terminal denoiser", element)?;
            }
            denoised
        } else {
            if predictor_order_used == 0 {
                return Err(SaSolverError::MissingHistory { step });
            }
            let tau = tau_function(next_sigma, step)
                .map_err(|reason| SaSolverError::TauFunction { step, reason })?;
            let tau = checked_nonnegative(step, "tau", tau)?;
            let lambda_start_index = step
                .checked_sub(predictor_order_used - 1)
                .ok_or(SaSolverError::MissingHistory { step })?;
            let lambda_source = schedule_value(&lambdas, step, step, "predictor source lambda")?;
            let lambda_target =
                schedule_value(&lambdas, step + 1, step, "predictor target lambda")?;
            let coefficients = stochastic_adams_coefficients(
                next_sigma,
                lambdas
                    .get(lambda_start_index..=step)
                    .ok_or(SaSolverError::MissingHistory { step })?,
                lambda_source,
                lambda_target,
                tau,
                options.simple_order_2(),
                false,
                step,
            )?;
            let prediction = weighted_history(
                backend,
                &prediction_history,
                predictor_order_used,
                &coefficients,
                step,
                context,
            )?;
            let step_size =
                checked_positive(step, "predictor step size", lambda_target - lambda_source)?;
            let state_scale = checked_coefficient(
                step,
                "predictor state scale",
                next_sigma / sigma * (-(tau * tau) * step_size).exp(),
            )?;
            let mut predicted = combine_state(
                backend,
                &corrected,
                state_scale,
                &prediction,
                step,
                "predictor",
                context,
            )?;
            previous_noise = if tau > 0.0 && effective_noise_scale > 0.0 {
                let radicand = checked_nonnegative(
                    step,
                    "noise variance",
                    -(-2.0 * tau * tau * step_size).exp_m1(),
                )?;
                let noise = source_noise(
                    backend,
                    &predicted,
                    next_sigma * radicand.sqrt() * effective_noise_scale,
                    step,
                    &mut noise_transaction,
                    context,
                )?;
                predicted = add_tensor(
                    backend,
                    &predicted,
                    &noise,
                    step,
                    "predictor noise",
                    context,
                )?;
                Some(noise)
            } else {
                None
            };
            previous_step_size = step_size;
            previous_tau = tau;
            predicted
        };
        corrected_state = corrected;
        observed.commit(next_predicted, context.cancellation)?;
    }

    let trace = session.finish()?;
    Ok((trace, (noise_before, noise_transaction.commit())))
}

#[allow(clippy::too_many_arguments)]
pub fn stochastic_adams_coefficients(
    sigma_next: f32,
    lambdas: &[f32],
    lambda_source: f32,
    lambda_target: f32,
    tau: f32,
    simple_order_2: bool,
    corrector: bool,
    step: usize,
) -> Result<Vec<f32>, SaSolverError> {
    let order = lambdas.len();
    if order == 0 {
        return Err(SaSolverError::MissingHistory { step });
    }
    let sigma_next = checked_nonnegative(step, "coefficient sigma", sigma_next)?;
    let lambda_source = checked_coefficient(step, "coefficient source lambda", lambda_source)?;
    let lambda_target = checked_coefficient(step, "coefficient target lambda", lambda_target)?;
    let tau = checked_nonnegative(step, "coefficient tau", tau)?;
    for lambda in lambdas.iter().copied() {
        checked_coefficient(step, "interpolation lambda", lambda)?;
    }
    let tau_multiplier = 1.0_f64 + f64::from(tau).powi(2);
    let step_size = f64::from(lambda_target - lambda_source);
    let alpha_target = f64::from(sigma_next) * f64::from(lambda_target).exp();
    if simple_order_2 && order == 2 {
        let first = alpha_target * (0.5 * tau_multiplier * step_size);
        let total = alpha_target * -(-step_size * tau_multiplier).exp_m1();
        let second = if corrector {
            total - first
        } else {
            let denominator = f64::from(lambdas[0] - lambda_source);
            if denominator == 0.0 || !denominator.is_finite() {
                return Err(SaSolverError::SingularSystem { step, order });
            }
            alpha_target * (0.5 * tau_multiplier * step_size.powi(2)) / denominator
        };
        let first = if corrector { first } else { total - second };
        return coefficients_to_f32([second, first], step);
    }

    let exponential = (-tau_multiplier * step_size).exp();
    let mut integrals = Vec::new();
    integrals
        .try_reserve_exact(order)
        .map_err(|_| SaSolverError::OutOfMemory("exponential coefficients"))?;
    for power in 0..order {
        let exponent =
            i32::try_from(power).map_err(|_| SaSolverError::Overflow("coefficient power"))?;
        let product = f64::from(lambda_target).powi(exponent)
            - f64::from(lambda_source).powi(exponent) * exponential;
        let integral = if power == 0 {
            product
        } else {
            product - power as f64 / tau_multiplier * integrals[power - 1]
        };
        if !integral.is_finite() {
            return Err(SaSolverError::InvalidCoefficient {
                step,
                coefficient: "exponential integral",
                value: integral as f32,
            });
        }
        integrals.push(integral);
    }
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(order)
        .map_err(|_| SaSolverError::OutOfMemory("Vandermonde matrix"))?;
    for power in 0..order {
        let exponent =
            i32::try_from(power).map_err(|_| SaSolverError::Overflow("Vandermonde power"))?;
        let mut row = Vec::new();
        row.try_reserve_exact(order)
            .map_err(|_| SaSolverError::OutOfMemory("Vandermonde row"))?;
        for lambda in lambdas {
            row.push(f64::from(*lambda).powi(exponent));
        }
        matrix.push(row);
    }
    let solution = solve_linear_system(matrix, integrals, step)?;
    coefficients_to_f32(solution.into_iter().map(|value| value * alpha_target), step)
}

fn coefficients_to_f32(
    values: impl IntoIterator<Item = f64>,
    step: usize,
) -> Result<Vec<f32>, SaSolverError> {
    let mut output = Vec::new();
    for value in values {
        let value = value as f32;
        if !value.is_finite() {
            return Err(SaSolverError::InvalidCoefficient {
                step,
                coefficient: "Adams weight",
                value,
            });
        }
        output.push(value);
    }
    Ok(output)
}

fn solve_linear_system(
    mut matrix: Vec<Vec<f64>>,
    mut right: Vec<f64>,
    step: usize,
) -> Result<Vec<f64>, SaSolverError> {
    let order = right.len();
    for pivot in 0..order {
        let pivot_row = (pivot..order)
            .max_by(|left, right_index| {
                matrix[*left][pivot]
                    .abs()
                    .total_cmp(&matrix[*right_index][pivot].abs())
            })
            .ok_or(SaSolverError::SingularSystem { step, order })?;
        if matrix[pivot_row][pivot] == 0.0 || !matrix[pivot_row][pivot].is_finite() {
            return Err(SaSolverError::SingularSystem { step, order });
        }
        matrix.swap(pivot, pivot_row);
        right.swap(pivot, pivot_row);
        for row in pivot + 1..order {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for column in pivot..order {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            right[row] -= factor * right[pivot];
        }
    }
    let mut output = vec![0.0_f64; order];
    for row in (0..order).rev() {
        let mut value = right[row];
        for column in row + 1..order {
            value -= matrix[row][column] * output[column];
        }
        let diagonal = matrix[row][row];
        if diagonal == 0.0 || !diagonal.is_finite() {
            return Err(SaSolverError::SingularSystem { step, order });
        }
        output[row] = value / diagonal;
        if !output[row].is_finite() {
            return Err(SaSolverError::SingularSystem { step, order });
        }
    }
    Ok(output)
}

fn half_log_snr_schedule(
    profile: &impl SamplingProfile,
    sigmas: &[f32],
) -> Result<Vec<f32>, SaSolverError> {
    let mut lambdas = Vec::new();
    lambdas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SaSolverError::OutOfMemory("half-log-SNR schedule"))?;
    for sigma in sigmas {
        lambdas.push(if *sigma == 0.0 {
            f32::INFINITY
        } else {
            profile.half_log_snr(*sigma)?
        });
    }
    Ok(lambdas)
}

fn evaluate_denoiser(
    denoiser: &mut impl FnMut(&Tensor, f32, usize, SaSolverEvaluation) -> Result<Tensor, String>,
    input: &Tensor,
    sigma: f32,
    step: usize,
    evaluation: SaSolverEvaluation,
) -> Result<Tensor, SaSolverError> {
    let output =
        denoiser(input, sigma, step, evaluation).map_err(|reason| SaSolverError::Denoiser {
            step,
            evaluation,
            reason,
        })?;
    if input.descriptor() != output.descriptor() {
        return Err(SaSolverError::DenoiserContract { step, evaluation });
    }
    Ok(output)
}

fn weighted_history(
    backend: &CpuBackend,
    history: &[Tensor],
    order: usize,
    coefficients: &[f32],
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SaSolverError> {
    if coefficients.len() != order || history.len() < order {
        return Err(SaSolverError::MissingHistory { step });
    }
    let selected = &history[history.len() - order..];
    let descriptor = selected
        .first()
        .ok_or(SaSolverError::MissingHistory { step })?
        .descriptor();
    let count =
        usize::try_from(descriptor.element_count()?).map_err(|_| TensorError::ShapeOverflow)?;
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(order)
        .map_err(|_| SaSolverError::OutOfMemory("decoded prediction history"))?;
    for tensor in selected {
        decoded.push(tensor_to_f32(backend, tensor, context)?);
    }
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for element in 0..count {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let mut value = 0.0_f32;
        for (coefficient, prediction) in coefficients.iter().zip(decoded.iter()) {
            value += *coefficient
                * prediction
                    .get(element)
                    .copied()
                    .ok_or(SaSolverError::MissingHistory { step })?;
        }
        checked_element(value, step, "weighted prediction", element)?;
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        descriptor.shape(),
        &values,
        context,
    )?)
}

fn combine_state(
    backend: &CpuBackend,
    state: &Tensor,
    state_scale: f32,
    residual: &Tensor,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SaSolverError> {
    let state_values = tensor_to_f32(backend, state, context)?;
    let residual_values = tensor_to_f32(backend, residual, context)?;
    let mut values = backend.workspace_vec::<f32>(context, state_values.len())?;
    for (element, (state, residual)) in state_values.iter().zip(residual_values.iter()).enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = state_scale.mul_add(*state, *residual);
        checked_element(value, step, stage, element)?;
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        state.descriptor().shape(),
        &values,
        context,
    )?)
}

fn add_tensor(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SaSolverError> {
    if left.descriptor() != right.descriptor() {
        return Err(SaSolverError::DenoiserContract {
            step,
            evaluation: SaSolverEvaluation::Predictor,
        });
    }
    let left_values = tensor_to_f32(backend, left, context)?;
    let right_values = tensor_to_f32(backend, right, context)?;
    let mut values = backend.workspace_vec::<f32>(context, left_values.len())?;
    for (element, (left, right)) in left_values.iter().zip(right_values.iter()).enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = left + right;
        checked_element(value, step, stage, element)?;
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        left.descriptor().shape(),
        &values,
        context,
    )?)
}

fn source_noise(
    backend: &CpuBackend,
    template: &Tensor,
    scale: f32,
    step: usize,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SaSolverError> {
    let count = usize::try_from(template.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let draws = transaction.draw_normal(count, context.cancellation)?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for (element, draw) in draws.into_iter().enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = draw as f32 * scale;
        checked_element(value, step, "source noise", element)?;
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        template.descriptor().shape(),
        &values,
        context,
    )?)
}

fn noise_profile(device: DeviceId) -> (RngSeedTransform, RngGenerationPlacement) {
    if device == DeviceId::CPU {
        (
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: device,
            },
        )
    } else {
        (
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(device),
        )
    }
}

fn schedule_value(
    values: &[f32],
    index: usize,
    step: usize,
    coefficient: &'static str,
) -> Result<f32, SaSolverError> {
    checked_coefficient(
        step,
        coefficient,
        values
            .get(index)
            .copied()
            .ok_or(SaSolverError::MissingHistory { step })?,
    )
}

fn checked_positive(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, SaSolverError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(SaSolverError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_nonnegative(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, SaSolverError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value < 0.0 {
        return Err(SaSolverError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_coefficient(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, SaSolverError> {
    if !value.is_finite() {
        return Err(SaSolverError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_element(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<(), SaSolverError> {
    if !value.is_finite() {
        return Err(SaSolverError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(())
}
