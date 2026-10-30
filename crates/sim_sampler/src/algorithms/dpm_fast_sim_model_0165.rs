use crate::generated_dpm_solver::{
    DpmSolverEquationError, DpmSolverEvaluation, DpmSolverStage, DpmSolverStepError,
    dpm_solver_first_intermediate, dpm_solver_first_order, dpm_solver_second_order,
    dpm_solver_third_order,
};
use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProgress, SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    standard_ancestral_step,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceVec, DeviceId, ExecutionContext, RngCheckpoint, RngCompatibilityError,
    RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use thiserror::Error;

pub const DPM_FAST_SAMPLER_ID: &str = "dpm_fast";
pub const DPM_FAST_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPM_FAST_SAMPLER_ID,
    feature_id: "COMFY-MODEL-0165",
    source_ordinal: 11,
    aliases: &[],
    implementation_module: "algorithms/dpm_fast_comfy_model_0165",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpmFastOptions {
    pub eta: f32,
    pub noise_scale: f32,
}

impl Default for DpmFastOptions {
    fn default() -> Self {
        Self {
            eta: 0.0,
            noise_scale: 1.0,
        }
    }
}

impl DpmFastOptions {
    pub fn new(eta: f32, noise_scale: f32) -> Result<Self, DpmFastSamplerError> {
        let options = Self { eta, noise_scale };
        options.validate()?;
        Ok(options)
    }

    fn validate(self) -> Result<(), DpmFastSamplerError> {
        if !self.eta.is_finite() || self.eta < 0.0 {
            return Err(DpmFastSamplerError::InvalidOption {
                name: "eta",
                value: self.eta,
            });
        }
        if !self.noise_scale.is_finite() {
            return Err(DpmFastSamplerError::InvalidOption {
                name: "noise scale",
                value: self.noise_scale,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum DpmFastSamplerError {
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
    #[error("DPM fast requires sampler identity `dpm_fast`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM fast {name} is invalid: {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("DPM fast requires finite positive sigma endpoints, got {maximum} -> {minimum}")]
    InvalidEndpoints { maximum: f32, minimum: f32 },
    #[error("DPM fast cannot apply eta {eta} while sampling in reverse")]
    ReverseEta { eta: f32 },
    #[error("DPM fast denoiser failed at interval {interval}, evaluation {evaluation}: {reason}")]
    Denoiser {
        interval: u32,
        evaluation: u8,
        reason: String,
    },
    #[error("DPM fast denoiser descriptor changed at interval {interval}, evaluation {evaluation}")]
    DenoiserContract { interval: u32, evaluation: u8 },
    #[error(
        "DPM fast produced a non-finite {stage} value at interval {interval}, element {element}"
    )]
    NonFinite {
        interval: u32,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM fast arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("DPM fast allocation failed for {0}")]
    OutOfMemory(&'static str),
}

pub fn sample_dpm_fast<E>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    options: DpmFastOptions,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, u32, u8) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), E>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), DpmFastSamplerError>
where
    E: std::fmt::Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != DPM_FAST_SAMPLER_ID {
        return Err(DpmFastSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    options.validate()?;
    if sigmas.len() <= 1 {
        let mut traced_sigmas = Vec::new();
        traced_sigmas
            .try_reserve_exact(sigmas.len())
            .map_err(|_| DpmFastSamplerError::OutOfMemory("short sigma trace"))?;
        traced_sigmas.extend_from_slice(sigmas);
        let mut latents = Vec::new();
        latents
            .try_reserve_exact(1)
            .map_err(|_| DpmFastSamplerError::OutOfMemory("short latent trace"))?;
        latents.push(initial);
        return Ok((
            SamplingTrace {
                sigmas: traced_sigmas,
                denoiser_evaluations: Vec::new(),
                latents,
            },
            None,
        ));
    }

    let mut caller_sigmas = Vec::new();
    caller_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| DpmFastSamplerError::OutOfMemory("caller sigma schedule"))?;
    caller_sigmas.extend_from_slice(sigmas);
    let caller_validation = SamplingSession::new(plan.clone(), caller_sigmas, initial.clone())?;
    drop(caller_validation);

    let maximum = *sigmas
        .first()
        .ok_or(DpmFastSamplerError::Overflow("maximum sigma"))?;
    let last = *sigmas
        .last()
        .ok_or(DpmFastSamplerError::Overflow("minimum sigma"))?;
    let minimum = if last == 0.0 {
        *sigmas
            .get(sigmas.len().saturating_sub(2))
            .ok_or(DpmFastSamplerError::Overflow("positive minimum sigma"))?
    } else {
        last
    };
    if !maximum.is_finite() || !minimum.is_finite() || maximum <= 0.0 || minimum <= 0.0 {
        return Err(DpmFastSamplerError::InvalidEndpoints { maximum, minimum });
    }

    let nfe = plan.steps();
    let group_count = nfe
        .checked_div(3)
        .and_then(|value| value.checked_add(1))
        .ok_or(DpmFastSamplerError::Overflow("solver group count"))?;
    let orders = solver_orders(nfe, group_count)?;
    let times = linear_times(-maximum.ln(), -minimum.ln(), group_count)?;
    let mut internal_sigmas = Vec::new();
    internal_sigmas
        .try_reserve_exact(times.len())
        .map_err(|_| DpmFastSamplerError::OutOfMemory("internal sigma schedule"))?;
    for time in times.iter().copied() {
        internal_sigmas.push((-time).exp());
    }
    let internal_steps = u32::try_from(orders.len())
        .map_err(|_| DpmFastSamplerError::Overflow("internal solver steps"))?;
    let internal_plan = SamplingPlan::new(
        DPM_FAST_SAMPLER_ID,
        plan.scheduler().as_str(),
        plan.profile().clone(),
        plan.seed(),
        internal_steps,
        plan.guidance(),
        plan.denoise(),
    )?;
    let mut session = SamplingSession::new(internal_plan, internal_sigmas, initial)?;
    let mut noise_transaction = noise_request.open_transaction(
        DPM_FAST_NOISE_CONTRACT_ID,
        i128::from(plan.seed()),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();
    let terminal_time = -minimum.ln();

    for (index, order) in orders.iter().copied().enumerate() {
        context.check()?;
        let interval =
            u32::try_from(index).map_err(|_| DpmFastSamplerError::Overflow("solver interval"))?;
        let time = *times
            .get(index)
            .ok_or(DpmFastSamplerError::Overflow("interval time"))?;
        let next_time = *times
            .get(index + 1)
            .ok_or(DpmFastSamplerError::Overflow("next interval time"))?;
        let current = session.current().clone();
        let base = evaluate(backend, &current, time, interval, 0, context, &mut denoiser)?;
        let observed = session.observe_step(
            &current,
            base.denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| {
                callback(
                    &SamplingProgress {
                        step: progress.step,
                        total_steps: nfe,
                        sigma: progress.sigma,
                        sigma_hat: progress.sigma_hat,
                        next_sigma: progress.next_sigma,
                    },
                    latent,
                    denoised,
                )
            },
        )?;
        let (solver_next_time, sigma_up) =
            ancestral_target(time, next_time, terminal_time, options.eta)?;
        let current_values = tensor_to_f32(backend, &current, context)?;
        let solved = solve_interval(
            backend,
            &current,
            &current_values,
            &base.epsilon,
            time,
            solver_next_time,
            order,
            interval,
            context,
            &mut denoiser,
        )?;
        let count = current_values.len();
        let normal = noise_transaction.draw_normal(count, context.cancellation)?;
        let mut noise = backend.workspace_vec::<f32>(context, count)?;
        for (element, value) in normal.into_iter().enumerate() {
            if element.is_multiple_of(256) {
                context.check()?;
            }
            noise.try_push(value as f32)?;
        }
        let next_values = combine(
            backend,
            &solved,
            &[(&noise, sigma_up * options.noise_scale)],
            interval,
            "stochastic update",
            context,
        )?;
        let next = tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?;
        observed.commit(next, context.cancellation)?;
    }

    let sampling = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok((sampling, Some((noise_before, noise_after))))
}

fn evaluate(
    backend: &CpuBackend,
    input: &Tensor,
    time: f32,
    interval: u32,
    evaluation: u8,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(&Tensor, f32, u32, u8) -> Result<Tensor, String>,
) -> Result<DpmSolverEvaluation, DpmFastSamplerError> {
    context.check()?;
    let sigma = (-time).exp();
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(DpmFastSamplerError::InvalidEndpoints {
            maximum: sigma,
            minimum: sigma,
        });
    }
    let denoised = denoiser(input, sigma, interval, evaluation).map_err(|reason| {
        DpmFastSamplerError::Denoiser {
            interval,
            evaluation,
            reason,
        }
    })?;
    if input.descriptor() != denoised.descriptor() {
        return Err(DpmFastSamplerError::DenoiserContract {
            interval,
            evaluation,
        });
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
        ensure_finite(value, interval, "epsilon", element)?;
        epsilon.try_push(value)?;
    }
    Ok(DpmSolverEvaluation { denoised, epsilon })
}

#[allow(clippy::too_many_arguments)]
fn solve_interval(
    backend: &CpuBackend,
    template: &Tensor,
    current: &[f32],
    epsilon: &[f32],
    time: f32,
    next_time: f32,
    order: u8,
    interval: u32,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(&Tensor, f32, u32, u8) -> Result<Tensor, String>,
) -> Result<CpuWorkspaceVec<f32>, DpmFastSamplerError> {
    if order == 1 {
        return dpm_solver_first_order(backend, current, epsilon, time, next_time, context)
            .map_err(|error| map_solver_equation_error(error, interval));
    }

    let first_ratio = if order == 2 { 0.5 } else { 1.0 / 3.0 };
    let first = dpm_solver_first_intermediate(
        backend,
        template,
        current,
        epsilon,
        time,
        next_time,
        first_ratio,
        context,
        &mut |input, evaluation_time, evaluation| {
            evaluate(
                backend,
                input,
                evaluation_time,
                interval,
                evaluation,
                context,
                denoiser,
            )
        },
    )
    .map_err(|error| map_solver_step_error(error, interval))?;
    if order == 2 {
        return dpm_solver_second_order(
            backend,
            current,
            epsilon,
            &first.evaluation.epsilon,
            time,
            next_time,
            first_ratio,
            context,
        )
        .map_err(|error| map_solver_equation_error(error, interval));
    }

    dpm_solver_third_order(
        backend,
        template,
        current,
        epsilon,
        &first.evaluation.epsilon,
        time,
        next_time,
        context,
        &mut |input, evaluation_time, evaluation| {
            evaluate(
                backend,
                input,
                evaluation_time,
                interval,
                evaluation,
                context,
                denoiser,
            )
        },
    )
    .map(|solution| solution.values)
    .map_err(|error| map_solver_step_error(error, interval))
}

fn map_solver_step_error(
    error: DpmSolverStepError<DpmFastSamplerError>,
    interval: u32,
) -> DpmFastSamplerError {
    match error {
        DpmSolverStepError::Equation(error) => map_solver_equation_error(error, interval),
        DpmSolverStepError::Evaluation(error) => error,
    }
}

fn map_solver_equation_error(error: DpmSolverEquationError, interval: u32) -> DpmFastSamplerError {
    match error {
        DpmSolverEquationError::Tensor(error) => DpmFastSamplerError::Tensor(error),
        DpmSolverEquationError::TensorKernel(error) => DpmFastSamplerError::TensorKernel(error),
        DpmSolverEquationError::Shape(stage) => {
            DpmFastSamplerError::Overflow(fast_solver_stage(stage))
        }
        DpmSolverEquationError::NonFinite { stage, element } => DpmFastSamplerError::NonFinite {
            interval,
            stage: fast_solver_stage(stage),
            element,
        },
    }
}

fn fast_solver_stage(stage: DpmSolverStage) -> &'static str {
    match stage {
        DpmSolverStage::FirstOrder => "first-order solution",
        DpmSolverStage::FirstIntermediate => "first intermediate",
        DpmSolverStage::SecondDifference | DpmSolverStage::ThirdFirstDifference => {
            "first epsilon difference"
        }
        DpmSolverStage::SecondOrder => "second-order solution",
        DpmSolverStage::ThirdIntermediate => "second intermediate",
        DpmSolverStage::ThirdSecondDifference => "second epsilon difference",
        DpmSolverStage::ThirdOrder => "third-order solution",
    }
}

fn solver_orders(nfe: u32, group_count: u32) -> Result<Vec<u8>, DpmFastSamplerError> {
    let capacity = usize::try_from(group_count)
        .map_err(|_| DpmFastSamplerError::Overflow("solver-order capacity"))?;
    let mut orders = Vec::new();
    orders
        .try_reserve_exact(capacity)
        .map_err(|_| DpmFastSamplerError::OutOfMemory("solver orders"))?;
    let remainder = nfe % 3;
    let third_order_count = if remainder == 0 {
        group_count.saturating_sub(2)
    } else {
        group_count.saturating_sub(1)
    };
    for _ in 0..third_order_count {
        orders.push(3);
    }
    if remainder == 0 {
        orders.push(2);
        orders.push(1);
    } else {
        orders.push(
            u8::try_from(remainder)
                .map_err(|_| DpmFastSamplerError::Overflow("final solver order"))?,
        );
    }
    if orders.len() != capacity {
        return Err(DpmFastSamplerError::Overflow("solver-order count"));
    }
    Ok(orders)
}

fn linear_times(start: f32, end: f32, groups: u32) -> Result<Vec<f32>, DpmFastSamplerError> {
    if !start.is_finite() || !end.is_finite() {
        return Err(DpmFastSamplerError::InvalidEndpoints {
            maximum: (-start).exp(),
            minimum: (-end).exp(),
        });
    }
    let count = usize::try_from(groups)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or(DpmFastSamplerError::Overflow("time-grid length"))?;
    let mut times = Vec::new();
    times
        .try_reserve_exact(count)
        .map_err(|_| DpmFastSamplerError::OutOfMemory("time grid"))?;
    let step = (end - start) / groups as f32;
    for index in 0..groups {
        times.push((index as f32).mul_add(step, start));
    }
    times.push(end);
    Ok(times)
}

fn ancestral_target(
    time: f32,
    next_time: f32,
    terminal_time: f32,
    eta: f32,
) -> Result<(f32, f32), DpmFastSamplerError> {
    if eta == 0.0 {
        return Ok((next_time, 0.0));
    }
    if next_time <= time {
        return Err(DpmFastSamplerError::ReverseEta { eta });
    }
    let sigma_from = (-time).exp();
    let sigma_to = (-next_time).exp();
    let (sigma_down, _) = standard_ancestral_step(sigma_from, sigma_to, eta).map_err(|_| {
        DpmFastSamplerError::InvalidOption {
            name: "ancestral target",
            value: next_time,
        }
    })?;
    let solver_time = if sigma_down == 0.0 {
        terminal_time
    } else {
        (-sigma_down.ln()).min(terminal_time)
    };
    let solver_sigma = (-solver_time).exp();
    let stochastic_scale = (sigma_to.powi(2) - solver_sigma.powi(2)).max(0.0).sqrt();
    if !solver_time.is_finite() || !stochastic_scale.is_finite() {
        return Err(DpmFastSamplerError::InvalidOption {
            name: "ancestral target",
            value: solver_time,
        });
    }
    Ok((solver_time, stochastic_scale))
}

fn combine(
    backend: &CpuBackend,
    base: &[f32],
    terms: &[(&[f32], f32)],
    interval: u32,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, DpmFastSamplerError> {
    if terms.iter().any(|(values, _)| values.len() != base.len()) {
        return Err(DpmFastSamplerError::Overflow(stage));
    }
    let mut output = backend.workspace_vec(context, base.len())?;
    for element in 0..base.len() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let mut value = *base
            .get(element)
            .ok_or(DpmFastSamplerError::Overflow(stage))?;
        for (values, coefficient) in terms {
            let term = values
                .get(element)
                .ok_or(DpmFastSamplerError::Overflow(stage))?;
            value += coefficient * term;
        }
        ensure_finite(value, interval, stage, element)?;
        output.try_push(value)?;
    }
    Ok(output)
}

fn ensure_finite(
    value: f32,
    interval: u32,
    stage: &'static str,
    element: usize,
) -> Result<(), DpmFastSamplerError> {
    if !value.is_finite() {
        return Err(DpmFastSamplerError::NonFinite {
            interval,
            stage,
            element,
        });
    }
    Ok(())
}
