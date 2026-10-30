pub use crate::noise::NoiseTrace;
use crate::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, EULER_SAMPLER_ID, NORMAL_SCHEDULER_ID,
    NoiseError, ObservedSamplingStep, PredictionInterpretation, SIMPLE_SCHEDULER_ID,
    SamplerRegistry, SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError,
    SamplingProfileIdentity, SamplingProgress, SamplingSession, SamplingTrace, SchedulerError,
    SchedulerRegistry, SchedulerRequest, generated_simple_sim_model_0211::simple_schedule,
    normal_schedule,
};
use sim_tensor::{
    BackendCapabilityMatrix, CompatibilityRngTransaction, CpuBackend, DeviceId, ExecutionContext,
    RngCheckpoint, RngCompatibilityError, RngError, RngGenerationPlacement, RngSeedTransform,
    RngStream, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const EULER_CHURN_NOISE_CONTRACT_ID: &str = "COMFY-RNG-D68A0DD3FBE1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerOptions {
    s_churn: f32,
    s_tmin: f32,
    s_tmax: f32,
    s_noise: f32,
}

impl EulerOptions {
    pub fn new(
        s_churn: f32,
        s_tmin: f32,
        s_tmax: f32,
        s_noise: f32,
    ) -> Result<Self, NativeDiffusionSamplerError> {
        if !s_churn.is_finite() {
            return Err(NativeDiffusionSamplerError::InvalidEulerOption {
                name: "s_churn",
                value: s_churn,
            });
        }
        if !s_tmin.is_finite() {
            return Err(NativeDiffusionSamplerError::InvalidEulerOption {
                name: "s_tmin",
                value: s_tmin,
            });
        }
        if s_tmax.is_nan() || s_tmax == f32::NEG_INFINITY {
            return Err(NativeDiffusionSamplerError::InvalidEulerOption {
                name: "s_tmax",
                value: s_tmax,
            });
        }
        if !s_noise.is_finite() {
            return Err(NativeDiffusionSamplerError::InvalidEulerOption {
                name: "s_noise",
                value: s_noise,
            });
        }
        Ok(Self {
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
        })
    }

    pub const fn source_defaults() -> Self {
        Self {
            s_churn: 0.0,
            s_tmin: 0.0,
            s_tmax: f32::INFINITY,
            s_noise: 1.0,
        }
    }

    pub const fn s_churn(self) -> f32 {
        self.s_churn
    }

    pub const fn s_tmin(self) -> f32 {
        self.s_tmin
    }

    pub const fn s_tmax(self) -> f32 {
        self.s_tmax
    }

    pub const fn s_noise(self) -> f32 {
        self.s_noise
    }

    fn gamma(self, sigma: f32, steps: usize) -> f32 {
        if self.s_churn > 0.0 && self.s_tmin <= sigma && sigma <= self.s_tmax {
            (self.s_churn / steps as f32).min(2.0_f32.sqrt() - 1.0)
        } else {
            0.0
        }
    }
}

impl Default for EulerOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum NativeDiffusionSamplerError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Rng(#[from] RngError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Profile(#[from] SamplingProfileError),
    #[error(transparent)]
    Noise(NoiseError),
    #[error("normal scheduler steps must be nonzero")]
    ZeroSteps,
    #[error("normal scheduler denoise must be finite and in (0, 1]")]
    InvalidDenoise,
    #[error("Euler sigma transition at step {step} is invalid: {sigma} -> {next_sigma}")]
    InvalidSigma {
        step: usize,
        sigma: f32,
        next_sigma: f32,
    },
    #[error("Euler denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error(
        "Euler denoiser output descriptor changed at step {step}: expected {expected}, got {actual}"
    )]
    DenoiserShape {
        step: usize,
        expected: String,
        actual: String,
    },
    #[error("Euler produced a non-finite {stage} at step {step}, element {element}")]
    NonFiniteEuler {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("Euler requires sampler identity `euler`, got {0:?}")]
    WrongEulerSampler(String),
    #[error(
        "Euler option {name} must be finite (except positive infinity for s_tmax), got {value}"
    )]
    InvalidEulerOption { name: &'static str, value: f32 },
    #[error("Euler churn requires a canonical RNG request")]
    MissingEulerNoiseRequest,
    #[error("native Euler churn noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
    #[error("native diffusion sampler arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("the native diffusion slice supports only Euler, normal, four steps, and denoise 1")]
    UnsupportedSlice,
}

#[derive(Debug, Error)]
pub enum LotusSdPoseSamplingError<DenoiserError>
where
    DenoiserError: Display,
{
    #[error("Lotus SDPose denoiser failed: {0}")]
    Denoiser(DenoiserError),
    #[error("Lotus SDPose Euler execution returned no typed denoiser error")]
    MissingDenoiserError,
    #[error(transparent)]
    Sampler(#[from] NativeDiffusionSamplerError),
}

pub fn sample_lotus_sdpose_one_step_euler<DenoiserError>(
    backend: &CpuBackend,
    initial: Tensor,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, DenoiserError>,
) -> Result<SamplingTrace, LotusSdPoseSamplingError<DenoiserError>>
where
    DenoiserError: Display,
{
    context.check().map_err(NativeDiffusionSamplerError::from)?;
    let profile =
        DiscreteSamplingProfile::lotus_sdpose().map_err(NativeDiffusionSamplerError::from)?;
    let registry = SchedulerRegistry::foundational().map_err(NativeDiffusionSamplerError::from)?;
    let request = SchedulerRequest::new(SIMPLE_SCHEDULER_ID, 1, 1.0)
        .map_err(NativeDiffusionSamplerError::from)?;
    let sigmas = simple_schedule(backend, context, &registry, &profile, &request)
        .map_err(NativeDiffusionSamplerError::from)?;
    let plan = SamplingPlan::new(
        EULER_SAMPLER_ID,
        SIMPLE_SCHEDULER_ID,
        profile.identity().clone(),
        0,
        1,
        1.0,
        1.0,
    )
    .map_err(NativeDiffusionSamplerError::from)?;
    let mut typed_error = None;
    let result = sample_euler_canonical(
        backend,
        plan,
        profile.identity(),
        EULER_SAMPLER_ID,
        initial,
        &sigmas,
        EulerOptions::source_defaults(),
        None,
        context,
        |input, sigma, step| match denoiser(input, sigma, step) {
            Ok(output) => Ok(output),
            Err(error) => {
                typed_error = Some(error);
                Err("typed Lotus SDPose denoiser failure".to_owned())
            }
        },
        |_, _, _| Ok::<(), String>(()),
    );
    match (result, typed_error) {
        (Ok((trace, None)), None) => Ok(trace),
        (Ok((_trace, Some(_))), _) => Err(LotusSdPoseSamplingError::Sampler(
            NativeDiffusionSamplerError::MissingEulerNoiseRequest,
        )),
        (Err(_), Some(error)) => Err(LotusSdPoseSamplingError::Denoiser(error)),
        (Err(error), None) => Err(LotusSdPoseSamplingError::Sampler(error)),
        (Ok(_), Some(_)) => Err(LotusSdPoseSamplingError::MissingDenoiserError),
    }
}

pub fn validate_euler_noise_generation_device(
    device: DeviceId,
) -> Result<(), NativeDiffusionSamplerError> {
    BackendCapabilityMatrix::for_native_device(device).map_err(|error| {
        NativeDiffusionSamplerError::DeviceUnavailable {
            device,
            reason: error.reason().to_owned(),
        }
    })?;
    Ok(())
}

pub fn sd15_model_time(sigma: f32) -> Result<f32, NativeDiffusionSamplerError> {
    model_time_for_profile(&DiscreteSamplingProfile::sd15()?, sigma)
}

pub fn model_time_for_profile(
    profile: &impl SamplingProfile,
    sigma: f32,
) -> Result<f32, NativeDiffusionSamplerError> {
    profile
        .model_time_for_sigma(sigma)
        .map_err(NativeDiffusionSamplerError::Profile)
}

pub fn sd15_interpret_prediction(
    backend: &CpuBackend,
    model_output: &Tensor,
    model_input: &Tensor,
    sigma: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    interpret_prediction_for_profile(
        &DiscreteSamplingProfile::sd15()?,
        backend,
        model_output,
        model_input,
        sigma,
        context,
    )
}

pub fn interpret_prediction_for_profile(
    profile: &impl SamplingProfile,
    backend: &CpuBackend,
    model_output: &Tensor,
    model_input: &Tensor,
    sigma: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    if model_output.descriptor().shape() != model_input.descriptor().shape() {
        return Err(NativeDiffusionSamplerError::DenoiserShape {
            step: 0,
            expected: format!("{:?}", model_input.descriptor()),
            actual: format!("{:?}", model_output.descriptor()),
        });
    }
    let mut output = tensor_to_f32(backend, model_output, context)?;
    let input = tensor_to_f32(backend, model_input, context)?;
    profile.interpret_prediction_in_place(&mut output, &input, sigma)?;
    tensor_from_f32(backend, model_output.descriptor().shape(), &output, context)
        .map_err(NativeDiffusionSamplerError::TensorKernel)
}

pub fn checked_native_diffusion_plan(
    sampler: &str,
    scheduler: &str,
    seed: u64,
    steps: u32,
    guidance: f32,
    denoise: f32,
) -> Result<SamplingPlan, NativeDiffusionSamplerError> {
    let profile = crate::NativeSamplingPolicy::sd15()?;
    checked_native_diffusion_plan_for_profile(
        &profile, sampler, scheduler, seed, steps, guidance, denoise,
    )
}

pub fn checked_native_diffusion_plan_for_profile(
    profile: &impl SamplingProfile,
    sampler: &str,
    scheduler: &str,
    seed: u64,
    steps: u32,
    guidance: f32,
    denoise: f32,
) -> Result<SamplingPlan, NativeDiffusionSamplerError> {
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    let plan = SamplingPlan::new(
        sampler,
        scheduler,
        profile.identity().clone(),
        seed,
        steps,
        guidance,
        denoise,
    )?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != EULER_SAMPLER_ID
        || plan.scheduler().as_str() != NORMAL_SCHEDULER_ID
        || plan.steps() != 4
        || plan.denoise() != 1.0
    {
        return Err(NativeDiffusionSamplerError::UnsupportedSlice);
    }
    Ok(plan)
}

pub fn normal_sigmas(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    steps: usize,
    denoise: f32,
) -> Result<Vec<f32>, NativeDiffusionSamplerError> {
    normal_sigmas_for_profile(
        &DiscreteSamplingProfile::sd15()?,
        backend,
        context,
        steps,
        denoise,
    )
}

pub fn normal_sigmas_for_profile(
    profile: &impl SamplingProfile,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    steps: usize,
    denoise: f32,
) -> Result<Vec<f32>, NativeDiffusionSamplerError> {
    if profile.prediction() == PredictionInterpretation::Flow {
        if denoise.to_bits() != 1.0_f32.to_bits() {
            return Err(NativeDiffusionSamplerError::UnsupportedSlice);
        }
        return source_flow_normal_sigmas(profile, context, steps);
    }
    let steps = u32::try_from(steps)
        .map_err(|_| NativeDiffusionSamplerError::Overflow("scheduler steps"))?;
    let request =
        SchedulerRequest::new(NORMAL_SCHEDULER_ID, steps, denoise).map_err(
            |error| match error {
                SchedulerError::ZeroSteps => NativeDiffusionSamplerError::ZeroSteps,
                SchedulerError::InvalidDenoise(_) => NativeDiffusionSamplerError::InvalidDenoise,
                error => NativeDiffusionSamplerError::Scheduler(error),
            },
        )?;
    let registry = SchedulerRegistry::foundational()?;
    normal_schedule(backend, context, &registry, profile, &request).map_err(|error| match error {
        SchedulerError::ZeroSteps => NativeDiffusionSamplerError::ZeroSteps,
        SchedulerError::InvalidDenoise(_) => NativeDiffusionSamplerError::InvalidDenoise,
        SchedulerError::Cancelled => NativeDiffusionSamplerError::Tensor(TensorError::Cancelled),
        SchedulerError::Tensor(error) => NativeDiffusionSamplerError::Tensor(error),
        error => NativeDiffusionSamplerError::Scheduler(error),
    })
}

fn source_flow_normal_sigmas(
    profile: &impl SamplingProfile,
    context: &ExecutionContext<'_>,
    steps: usize,
) -> Result<Vec<f32>, NativeDiffusionSamplerError> {
    if steps == 0 {
        return Err(NativeDiffusionSamplerError::ZeroSteps);
    }
    let start = profile.model_time_for_sigma(profile.sigma_max())?;
    let end = profile.model_time_for_sigma(profile.sigma_min())?;
    let mut sigmas = Vec::new();
    sigmas
        .try_reserve_exact(
            steps
                .checked_add(1)
                .ok_or(NativeDiffusionSamplerError::Overflow(
                    "normal flow schedule",
                ))?,
        )
        .map_err(|_| {
            NativeDiffusionSamplerError::Profile(SamplingProfileError::OutOfMemory(
                "normal flow schedule",
            ))
        })?;
    for index in 0..steps {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        let fraction = if steps == 1 {
            0.0
        } else {
            index as f32 / (steps - 1) as f32
        };
        let model_time = (end - start).mul_add(fraction, start);
        sigmas.push(profile.sigma_for_model_time(model_time)?);
    }
    sigmas.push(0.0);
    Ok(sigmas)
}

pub fn sample_euler(
    backend: &CpuBackend,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
) -> Result<SamplingTrace, NativeDiffusionSamplerError> {
    sample_euler_for_profile(
        &DiscreteSamplingProfile::sd15()?,
        backend,
        initial,
        sigmas,
        context,
        denoiser,
    )
}

pub fn sample_euler_for_profile(
    profile: &impl SamplingProfile,
    backend: &CpuBackend,
    initial: Tensor,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
) -> Result<SamplingTrace, NativeDiffusionSamplerError> {
    if sigmas.len() < 2 {
        return Err(NativeDiffusionSamplerError::ZeroSteps);
    }
    let steps = u32::try_from(sigmas.len() - 1)
        .map_err(|_| NativeDiffusionSamplerError::Overflow("Euler steps"))?;
    let plan = SamplingPlan::new(
        EULER_SAMPLER_ID,
        NORMAL_SCHEDULER_ID,
        profile.identity().clone(),
        0,
        steps,
        1.0,
        1.0,
    )?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        profile.identity(),
    )?;
    let (trace, checkpoints) = sample_euler_canonical(
        backend,
        plan,
        profile.identity(),
        EULER_SAMPLER_ID,
        initial,
        sigmas,
        EulerOptions::source_defaults(),
        None,
        context,
        &mut denoiser,
        |_, _, _| Ok::<(), String>(()),
    )?;
    debug_assert!(checkpoints.is_none());
    Ok(trace)
}

#[allow(clippy::too_many_arguments)]
pub fn sample_euler_with_options<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    options: EulerOptions,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), NativeDiffusionSamplerError>
where
    CallbackError: Display,
{
    sample_euler_canonical(
        backend,
        plan,
        expected_profile,
        EULER_SAMPLER_ID,
        initial,
        sigmas,
        options,
        Some(noise_request),
        context,
        denoiser,
        callback,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_euler_canonical<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &SamplingProfileIdentity,
    expected_sampler: &str,
    initial: Tensor,
    sigmas: &[f32],
    options: EulerOptions,
    noise_request: Option<CompatibilityNoiseRequest>,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), NativeDiffusionSamplerError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        expected_profile,
    )?;
    if plan.sampler().as_str() != expected_sampler {
        return Err(NativeDiffusionSamplerError::WrongEulerSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    let step_count = sigmas.len().saturating_sub(1);
    let seed = plan.seed();
    let generation_device = initial.descriptor().device();
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("Euler sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session =
        SamplingSession::new(plan, owned_sigmas, initial).map_err(|error| match error {
            SamplingError::InvalidSigma {
                step,
                sigma,
                next_sigma,
            } => NativeDiffusionSamplerError::InvalidSigma {
                step,
                sigma,
                next_sigma,
            },
            error => NativeDiffusionSamplerError::Sampling(error),
        })?;
    let churn_enabled = sigmas
        .iter()
        .take(step_count)
        .copied()
        .any(|sigma| options.gamma(sigma, step_count) > 0.0);
    let mut noise_transaction = if churn_enabled {
        validate_euler_noise_generation_device(generation_device)?;
        Some(
            noise_request
                .ok_or(NativeDiffusionSamplerError::MissingEulerNoiseRequest)?
                .open_transaction(
                    EULER_CHURN_NOISE_CONTRACT_ID,
                    i128::from(seed),
                    RngSeedTransform::TorchSigned64,
                    RngGenerationPlacement::Native(generation_device),
                    None,
                    context.cancellation,
                )?,
        )
    } else {
        None
    };
    let noise_before = noise_transaction
        .as_ref()
        .map(CompatibilityRngTransaction::checkpoint);

    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = pair[0];
        let next_sigma = pair[1];
        if !sigma.is_finite() || !next_sigma.is_finite() || sigma <= 0.0 || next_sigma < 0.0 {
            return Err(NativeDiffusionSamplerError::InvalidSigma {
                step,
                sigma,
                next_sigma,
            });
        }
        let gamma = options.gamma(sigma, step_count);
        let sigma_hat = sigma * (gamma + 1.0);
        if !sigma_hat.is_finite() || sigma_hat <= 0.0 {
            return Err(NativeDiffusionSamplerError::Tensor(
                TensorError::InvalidNumeric {
                    reason: format!("Euler produced invalid sigma_hat at step {step}"),
                },
            ));
        }
        let current = session.current().clone();
        let churned = if gamma > 0.0 {
            apply_euler_churn(
                backend,
                &current,
                sigma,
                sigma_hat,
                options.s_noise,
                step,
                noise_transaction
                    .as_mut()
                    .ok_or(NativeDiffusionSamplerError::MissingEulerNoiseRequest)?,
                context,
            )?
        } else {
            current
        };
        let EulerPrediction {
            observed,
            derivative,
            ..
        } = observe_euler_prediction(
            backend,
            &mut session,
            &churned,
            sigma_hat,
            step,
            context,
            &mut denoiser,
            &mut callback,
        )?;
        let next = advance_euler(
            backend,
            &churned,
            &derivative,
            next_sigma - sigma_hat,
            step,
            context,
        )?;
        observed.commit(next, context.cancellation)?;
    }
    let sampling = session.finish()?;
    let checkpoints = match (noise_before, noise_transaction) {
        (Some(before), Some(transaction)) => Some((before, transaction.commit())),
        (None, None) => None,
        _ => return Err(NativeDiffusionSamplerError::MissingEulerNoiseRequest),
    };
    Ok((sampling, checkpoints))
}

pub(crate) struct EulerPrediction<'a> {
    pub observed: ObservedSamplingStep<'a>,
    pub denoised: Tensor,
    pub derivative: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn observe_euler_prediction<'a, CallbackError>(
    backend: &CpuBackend,
    session: &'a mut SamplingSession,
    input: &Tensor,
    sigma_hat: f32,
    step: usize,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: &mut impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<EulerPrediction<'a>, NativeDiffusionSamplerError>
where
    CallbackError: Display,
{
    let denoised = evaluate_euler_denoiser(input, sigma_hat, step, denoiser)?;
    let derivative = euler_derivative(backend, input, &denoised, sigma_hat, step, context)?;
    let observed = observe_euler_denoised(
        session,
        input,
        denoised.clone(),
        sigma_hat,
        context,
        callback,
    )?;
    Ok(EulerPrediction {
        observed,
        denoised,
        derivative,
    })
}

pub(crate) fn evaluate_euler_denoiser(
    input: &Tensor,
    sigma: f32,
    step: usize,
    denoiser: &mut impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    let denoised = denoiser(input, sigma, step)
        .map_err(|reason| NativeDiffusionSamplerError::Denoiser { step, reason })?;
    if denoised.descriptor() != input.descriptor() {
        return Err(NativeDiffusionSamplerError::DenoiserShape {
            step,
            expected: format!("{:?}", input.descriptor()),
            actual: format!("{:?}", denoised.descriptor()),
        });
    }
    Ok(denoised)
}

pub(crate) fn observe_euler_denoised<'a, CallbackError>(
    session: &'a mut SamplingSession,
    input: &Tensor,
    denoised: Tensor,
    sigma_hat: f32,
    context: &ExecutionContext<'_>,
    callback: &mut impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<ObservedSamplingStep<'a>, NativeDiffusionSamplerError>
where
    CallbackError: Display,
{
    Ok(session.observe_step(
        input,
        denoised,
        context.cancellation,
        |progress, latent, denoised| {
            callback(
                &SamplingProgress {
                    sigma_hat,
                    ..*progress
                },
                latent,
                denoised,
            )
        },
    )?)
}

fn euler_derivative(
    backend: &CpuBackend,
    input: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    let input_values = tensor_to_f32(backend, input, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let mut derivative_values = backend.workspace_vec::<f32>(context, input_values.len())?;
    for (element, (input, denoised)) in input_values.iter().zip(denoised_values.iter()).enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let derivative = (input - denoised) / sigma;
        if !derivative.is_finite() {
            return Err(NativeDiffusionSamplerError::NonFiniteEuler {
                step,
                stage: "derivative",
                element,
            });
        }
        derivative_values.try_push(derivative)?;
    }
    Ok(tensor_from_f32(
        backend,
        input.descriptor().shape(),
        &derivative_values,
        context,
    )?)
}

pub(crate) fn advance_euler(
    backend: &CpuBackend,
    input: &Tensor,
    derivative: &Tensor,
    delta: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    let input_values = tensor_to_f32(backend, input, context)?;
    let derivative_values = tensor_to_f32(backend, derivative, context)?;
    let mut next_values = backend.workspace_vec::<f32>(context, input_values.len())?;
    for (element, (input, derivative)) in input_values
        .iter()
        .zip(derivative_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = derivative.mul_add(delta, *input);
        if !value.is_finite() {
            return Err(NativeDiffusionSamplerError::NonFiniteEuler {
                step,
                stage: "output",
                element,
            });
        }
        next_values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        input.descriptor().shape(),
        &next_values,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn apply_euler_churn(
    backend: &CpuBackend,
    current: &Tensor,
    sigma: f32,
    sigma_hat: f32,
    noise_scale: f32,
    step: usize,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
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
        if !noise_value.is_finite() || !churned.is_finite() {
            return Err(NativeDiffusionSamplerError::Tensor(
                TensorError::InvalidNumeric {
                    reason: format!("Euler produced non-finite churn at step {step}"),
                },
            ));
        }
        churned_values.try_push(churned)?;
    }
    Ok(tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &churned_values,
        context,
    )?)
}

pub fn normal_noise(
    backend: &CpuBackend,
    shape: &[u64],
    stream: &RngStream,
    context: &ExecutionContext<'_>,
) -> Result<NoiseTrace, NativeDiffusionSamplerError> {
    crate::noise::normal_noise(backend, shape, stream, context).map_err(map_noise_error)
}

pub fn scale_initial_noise(
    backend: &CpuBackend,
    noise: &Tensor,
    latent: &Tensor,
    sigma: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    scale_initial_noise_for_profile(
        &DiscreteSamplingProfile::sd15()?,
        backend,
        noise,
        latent,
        sigma,
        context,
    )
}

pub fn scale_initial_noise_for_profile(
    profile: &impl SamplingProfile,
    backend: &CpuBackend,
    noise: &Tensor,
    latent: &Tensor,
    sigma: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    if noise.descriptor().shape() != latent.descriptor().shape() {
        return Err(NativeDiffusionSamplerError::DenoiserShape {
            step: 0,
            expected: format!("{:?}", latent.descriptor()),
            actual: format!("{:?}", noise.descriptor()),
        });
    }
    let mut noise_values = tensor_to_f32(backend, noise, context)?;
    let latent_values = tensor_to_f32(backend, latent, context)?;
    let max_denoise = profile.is_max_denoise(sigma)?;
    profile.scale_initial_noise_in_place(&mut noise_values, &latent_values, sigma, max_denoise)?;
    Ok(tensor_from_f32(
        backend,
        latent.descriptor().shape(),
        &noise_values,
        context,
    )?)
}

pub fn scale_model_input(
    backend: &CpuBackend,
    model_input: &Tensor,
    sigma: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    scale_model_input_for_profile(
        &DiscreteSamplingProfile::sd15()?,
        backend,
        model_input,
        sigma,
        context,
    )
}

pub fn scale_model_input_for_profile(
    profile: &impl SamplingProfile,
    backend: &CpuBackend,
    model_input: &Tensor,
    sigma: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionSamplerError> {
    let mut values = tensor_to_f32(backend, model_input, context)?;
    profile.scale_model_input_in_place(&mut values, sigma)?;
    Ok(tensor_from_f32(
        backend,
        model_input.descriptor().shape(),
        &values,
        context,
    )?)
}

fn map_noise_error(error: NoiseError) -> NativeDiffusionSamplerError {
    match error {
        NoiseError::Rng(error) => NativeDiffusionSamplerError::Rng(error),
        NoiseError::Tensor(TensorError::Cancelled) | NoiseError::Cancelled => {
            NativeDiffusionSamplerError::Tensor(TensorError::Cancelled)
        }
        NoiseError::Tensor(error) => NativeDiffusionSamplerError::Tensor(error),
        NoiseError::TensorKernel(error) => NativeDiffusionSamplerError::TensorKernel(error),
        error => NativeDiffusionSamplerError::Noise(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{
        CancellationToken, CpuWorkspaceAuthority, DeviceId, RetryRngPolicy, RngAlgorithm,
        RngProfileVersion, RngStreamAddress, StreamId,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    fn workspace() -> Result<PathBuf, Box<dyn std::error::Error>> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?
            .to_path_buf())
    }

    fn execution_context<'a>(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, TensorError> {
        Ok(backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(512 * 1024)?,
            cancellation,
        ))
    }

    fn digest(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
    }

    #[derive(Debug, Error)]
    #[error("fixture denoiser failed")]
    struct FixtureDenoiserError;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "{actual} differs from {expected}",
        );
    }

    #[test]
    fn family_model_profiles_parameterize_schedule_input_time_and_prediction()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        for (profile, shift, flux) in [
            (DiscreteSamplingProfile::auraflow()?, 1.73_f32, false),
            (DiscreteSamplingProfile::qwen_image()?, 1.15_f32, true),
            (
                DiscreteSamplingProfile::discrete_flow_with_shift(1.73, 1.0)?,
                1.73_f32,
                false,
            ),
            (
                DiscreteSamplingProfile::flux_with_shift(1.15)?,
                1.15_f32,
                true,
            ),
        ] {
            let sigmas = normal_sigmas_for_profile(&profile, &backend, &context, 3, 1.0)?;
            assert_eq!(sigmas.len(), 4);
            assert_close(sigmas[0], 1.0);
            assert_eq!(sigmas[3], 0.0);
            let source_shift = |time: f32| {
                if flux {
                    let exponential = shift.exp();
                    exponential / (exponential + (time.recip() - 1.0))
                } else {
                    shift * time / (1.0 + (shift - 1.0) * time)
                }
            };
            let source_grid_min = source_shift(if flux { 0.0001 } else { 0.001 });
            assert_close(sigmas[2], source_shift(source_grid_min));
            assert_close(sigmas[1], source_shift((1.0 + source_grid_min) * 0.5));
            assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));

            let latent = tensor_from_f32(&backend, &[2], &[2.0, 3.0], &context)?;
            let raw = tensor_from_f32(&backend, &[2], &[0.5, -1.0], &context)?;
            let interpreted = interpret_prediction_for_profile(
                &profile, &backend, &raw, &latent, 0.25, &context,
            )?;
            assert_eq!(
                &*tensor_to_f32(&backend, &interpreted, &context)?,
                &[1.875, 3.25]
            );
            let scaled = scale_model_input_for_profile(&profile, &backend, &latent, 0.5, &context)?;
            assert_eq!(&*tensor_to_f32(&backend, &scaled, &context)?, &[2.0, 3.0]);
        }
        assert!(matches!(
            normal_sigmas_for_profile(
                &DiscreteSamplingProfile::auraflow()?,
                &backend,
                &context,
                3,
                0.5,
            ),
            Err(NativeDiffusionSamplerError::UnsupportedSlice)
        ));
        Ok(())
    }

    #[test]
    fn lotus_sdpose_sampling_is_direct_denoised_typed_and_rng_free()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let sd15 = DiscreteSamplingProfile::sd15()?;
        let lotus = DiscreteSamplingProfile::lotus_sdpose()?;
        assert_ne!(lotus.identity(), sd15.identity());
        assert_eq!(
            lotus.prediction(),
            crate::PredictionInterpretation::Denoised
        );
        assert_eq!(lotus.sigmas(), sd15.sigmas());
        assert_eq!(lotus.model_time_for_sigma(lotus.sigma_max())?, 999.0);

        let initial = tensor_from_f32(&backend, &[1], &[2.0], &context)?;
        let mut calls = Vec::new();
        let trace = sample_lotus_sdpose_one_step_euler(
            &backend,
            initial,
            &context,
            |input, sigma, step| {
                calls.push((
                    tensor_to_f32(&backend, input, &context)
                        .map_err(|_| FixtureDenoiserError)?
                        .to_vec(),
                    sigma,
                    step,
                ));
                tensor_from_f32(&backend, &[1], &[7.0], &context).map_err(|_| FixtureDenoiserError)
            },
        )?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, vec![2.0]);
        assert_eq!(calls[0].1, lotus.sigma_max());
        assert_eq!(calls[0].2, 0);
        assert_eq!(trace.sigmas, vec![lotus.sigma_max(), 0.0]);
        assert_eq!(trace.denoiser_evaluations.len(), 1);
        assert_eq!(trace.latents.len(), 2);
        assert_eq!(
            &*tensor_to_f32(
                &backend,
                trace.latents.last().ok_or("missing final latent")?,
                &context
            )?,
            &[7.0]
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let typed = sample_lotus_sdpose_one_step_euler(
            &backend,
            tensor_from_f32(&backend, &[1], &[2.0], &context)?,
            &context,
            |_, _, _| Err::<Tensor, _>(FixtureDenoiserError),
        );
        assert!(matches!(
            typed,
            Err(LotusSdPoseSamplingError::Denoiser(FixtureDenoiserError))
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    fn write_artifact(
        filename: &str,
        validation: &str,
        scope: &str,
        fixture_digests: serde_json::Value,
        cases: serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let directory = workspace()?.join("target/sim-parity");
        fs::create_dir_all(&directory)?;
        let passed = cases
            .as_object()
            .ok_or("validation cases must be an object")?
            .len();
        let value = json!({
            "cases": cases,
            "environment": {"architecture": std::env::consts::ARCH, "backend": "native-rust-cpu", "operating_system": std::env::consts::OS},
            "fixture_digests": fixture_digests,
            "remaining_release_gates": ["comfy-parity-native-diffusion-e2e", "sim-parity-sampler-scheduler-breadth", "comfy-parity-final-validation"],
            "scope": scope,
            "skipped": [],
            "summary": {"failed": 0, "passed": passed, "skipped": 0},
            "validation": validation,
            "validation_id": validation,
        });
        let mut bytes = serde_json::to_vec_pretty(&value)?;
        bytes.push(b'\n');
        fs::write(directory.join(filename), bytes)?;
        Ok(())
    }

    #[test]
    fn normal_scheduler_and_euler_are_exact_and_cancellable()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let sigmas = normal_sigmas(&backend, &context, 4, 1.0)?;
        assert_eq!(sigmas.len(), 5);
        assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
        assert_eq!(sigmas[4], 0.0);

        let initial = tensor_from_f32(&backend, &[1], &[1.0], &context)?;
        let trace = sample_euler(
            &backend,
            initial,
            &[2.0, 1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
        )?;
        let final_values = tensor_to_f32(
            &backend,
            trace.latents.last().ok_or("missing latent")?,
            &context,
        )?;
        assert_eq!(&*final_values, &[1.0]);

        cancellation.cancel();
        let active_cancellation = CancellationToken::default();
        let active_context = execution_context(&backend, &authority, &active_cancellation)?;
        let initial = tensor_from_f32(&backend, &[1], &[1.0], &active_context)?;
        let cancelled_context = execution_context(&backend, &authority, &cancellation)?;
        assert!(matches!(
            sample_euler(
                &backend,
                initial,
                &[1.0, 0.0],
                &cancelled_context,
                |value, _, _| Ok(value.clone())
            ),
            Err(NativeDiffusionSamplerError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn generalized_euler_source_defaults_preserve_the_compatibility_entry_point()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let initial = tensor_from_f32(&backend, &[2], &[1.25, -0.75], &context)?;
        let compatibility = sample_euler(
            &backend,
            initial.clone(),
            &[4.0, 2.0, 1.0, 0.0],
            &context,
            |_, _, step| {
                let denoised = match step {
                    0 => [0.4, -0.2],
                    1 => [-0.5, 0.7],
                    _ => [0.0, 0.25],
                };
                tensor_from_f32(&backend, &[2], &denoised, &context)
                    .map_err(|error| error.to_string())
            },
        )?;
        let profile = DiscreteSamplingProfile::sd15()?;
        let plan = SamplingPlan::new(
            EULER_SAMPLER_ID,
            NORMAL_SCHEDULER_ID,
            profile.identity().clone(),
            0,
            3,
            1.0,
            1.0,
        )?;
        let mut observed = Vec::new();
        let (generalized, checkpoints) = sample_euler_with_options(
            &backend,
            plan,
            profile.identity(),
            initial,
            &[4.0, 2.0, 1.0, 0.0],
            EulerOptions::source_defaults(),
            CompatibilityNoiseRequest::new(
                "native-diffusion-tests",
                "source-defaults",
                "KSampler",
                0,
                0,
                0,
                0,
                RetryRngPolicy::Replay,
            ),
            &context,
            |_, _, step| {
                let denoised = match step {
                    0 => [0.4, -0.2],
                    1 => [-0.5, 0.7],
                    _ => [0.0, 0.25],
                };
                tensor_from_f32(&backend, &[2], &denoised, &context)
                    .map_err(|error| error.to_string())
            },
            |progress, _, _| {
                observed.push(*progress);
                Ok::<(), String>(())
            },
        )?;
        assert!(checkpoints.is_none());
        assert_eq!(observed.len(), 3);
        assert!(
            observed
                .iter()
                .all(|progress| progress.sigma == progress.sigma_hat)
        );
        for (compatibility, generalized) in compatibility.latents.iter().zip(&generalized.latents) {
            assert_eq!(
                &*tensor_to_f32(&backend, compatibility, &context)?,
                &*tensor_to_f32(&backend, generalized, &context)?
            );
        }
        Ok(())
    }

    #[test]
    fn noise_uses_canonical_addressed_rng_stream() -> Result<(), Box<dyn std::error::Error>> {
        let address = RngStreamAddress::new(
            "sd15-tiny-v1",
            "fixture",
            "KSampler",
            0,
            crate::INITIAL_NOISE_PHASE_ID,
            0,
            0,
            RetryRngPolicy::Replay,
        )?;
        let stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            0x0123_4567_89ab_cdef,
            address,
        )?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let first = normal_noise(&backend, &[1, 4, 4, 4], &stream, &context)?;
        let second = normal_noise(&backend, &[1, 4, 4, 4], &stream, &context)?;
        let first_values = tensor_to_f32(&backend, &first.noise, &context)?;
        let second_values = tensor_to_f32(&backend, &second.noise, &context)?;
        assert_eq!(&*first_values, &*second_values);
        assert_eq!(first.before, second.before);
        assert_eq!(first.after, second.after);

        let cuda = DeviceId::from_source_device("cuda:0")?;
        let mismatched = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            0x0123_4567_89ab_cdef,
            RngStreamAddress::for_device(
                "sd15-tiny-v1",
                "fixture",
                "KSampler",
                0,
                crate::INITIAL_NOISE_PHASE_ID,
                0,
                0,
                RetryRngPolicy::Replay,
                cuda,
            )?,
        )?;
        assert!(matches!(
            normal_noise(&backend, &[1], &mismatched, &context),
            Err(NativeDiffusionSamplerError::Rng(RngError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual,
            })) if actual == cuda
        ));
        Ok(())
    }

    #[test]
    fn scheduler_workspace_is_exact_and_converges() -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(32)?;
        let insufficient = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(19)?,
            &cancellation,
        );
        assert!(matches!(
            normal_sigmas(&backend, &insufficient, 4, 1.0),
            Err(NativeDiffusionSamplerError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);

        let exact = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(20)?,
            &cancellation,
        );
        assert_eq!(normal_sigmas(&backend, &exact, 4, 1.0)?.len(), 5);
        let snapshot = backend.memory_snapshot();
        assert_eq!(snapshot.current_bytes, 0);
        assert_eq!(snapshot.peak_bytes, 32);
        assert_eq!(exact.scratch.in_use_bytes(), 0);
        assert_eq!(exact.scratch.peak_bytes(), 20);
        Ok(())
    }

    #[test]
    fn val_scheduler_001() -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let root = workspace()?;
        let fixture =
            root.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1/normal-sigmas.f64le");
        let expected_bytes = fs::read(&fixture)?;
        let expected = expected_bytes
            .chunks_exact(8)
            .map(|chunk| {
                let encoded: [u8; 8] = chunk.try_into().map_err(|_| "invalid f64 fixture")?;
                Ok(f64::from_le_bytes(encoded))
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        let actual = normal_sigmas(&backend, &context, 4, 1.0)?;
        assert_eq!(
            expected,
            actual
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>()
        );
        assert_eq!(actual.len(), 5);
        assert!(actual.windows(2).all(|pair| pair[0] > pair[1]));
        assert_eq!(actual[4], 0.0);
        assert_eq!(normal_sigmas(&backend, &context, 1, 1.0)?.len(), 2);
        assert!(
            normal_sigmas(&backend, &context, 4, 0.5)?
                .windows(2)
                .all(|pair| pair[0] > pair[1])
        );
        assert!(matches!(
            normal_sigmas(&backend, &context, 0, 1.0),
            Err(NativeDiffusionSamplerError::ZeroSteps)
        ));
        for denoise in [0.0, -1.0, 1.1, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                normal_sigmas(&backend, &context, 4, denoise),
                Err(NativeDiffusionSamplerError::InvalidDenoise)
            ));
        }
        write_artifact(
            "val-scheduler-001.json",
            "VAL-SCHEDULER-001",
            "Task 36 normal scheduler COMFY-MODEL-0209 exact SD15 sigma slice",
            json!({"normal_sigmas": digest(&fixture)?}),
            json!({
                "denoise_slice_is_exact": true, "device_independent_f32_values": true,
                "invalid_denoise_is_typed": true, "normal_sigma_array_matches_fixture": true,
                "one_step_boundary_is_valid": true, "zero_steps_is_typed": true,
            }),
        )?;
        Ok(())
    }

    #[test]
    fn val_sampler_001() -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let initial = tensor_from_f32(&backend, &[1], &[2.0], &context)?;
        let mut callbacks = Vec::new();
        let trace = sample_euler(
            &backend,
            initial,
            &[2.0, 1.0, 0.0],
            &context,
            |_value, sigma, step| {
                callbacks.push((step, sigma));
                tensor_from_f32(&backend, &[1], &[0.0], &context).map_err(|error| error.to_string())
            },
        )?;
        assert_eq!(callbacks, vec![(0, 2.0), (1, 1.0)]);
        let latents = trace
            .latents
            .iter()
            .map(|tensor| tensor_to_f32(&backend, tensor, &context).map(|values| values.to_vec()))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(latents, vec![vec![2.0], vec![1.0], vec![0.0]]);
        assert!(matches!(
            sample_euler(
                &backend,
                tensor_from_f32(&backend, &[1], &[1.0], &context)?,
                &[f32::NAN, 0.0],
                &context,
                |value, _, _| Ok(value.clone())
            ),
            Err(NativeDiffusionSamplerError::InvalidSigma { .. })
        ));
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
        assert!(matches!(
            sample_euler(
                &backend,
                tensor_from_f32(&backend, &[1], &[1.0], &context)?,
                &[1.0, 0.0],
                &cancelled_context,
                |value, _, _| Ok(value.clone())
            ),
            Err(NativeDiffusionSamplerError::Tensor(TensorError::Cancelled))
        ));
        let root = workspace()?;
        let fixture_root = root.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1");
        let mut fixture_digests = serde_json::Map::new();
        for name in [
            "initial-noise.safetensors",
            "denoiser-eval-000.safetensors",
            "denoiser-eval-001.safetensors",
            "denoiser-eval-002.safetensors",
            "denoiser-eval-003.safetensors",
            "latent-step-000.safetensors",
            "latent-step-001.safetensors",
            "latent-step-002.safetensors",
            "latent-step-003.safetensors",
            "latent-step-004.safetensors",
        ] {
            fixture_digests.insert(
                name.to_owned(),
                serde_json::Value::String(digest(&fixture_root.join(name))?),
            );
        }
        write_artifact(
            "val-sampler-001.json",
            "VAL-SAMPLER-001",
            "Task 36 Euler COMFY-MODEL-0179 trajectory, callback, RNG checkpoint, boundary, and cancellation slice",
            serde_json::Value::Object(fixture_digests),
            json!({
                "all_native_fixture_intermediates_are_accounted": true,
                "callback_order_is_exact": true, "canonical_rng_is_repeatable": true,
                "cancellation_is_typed": true, "euler_equation_is_exact": true,
                "invalid_sigma_is_typed": true, "latent_trajectory_is_exact": true,
            }),
        )?;
        Ok(())
    }
}
