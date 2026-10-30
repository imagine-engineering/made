use crate::{
    CompatibilityNoiseRequest, PredictionInterpretation, SamplerDefinition, SamplerRegistry,
    SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    generated_native_diffusion::{
        EulerPrediction, NativeDiffusionSamplerError, advance_euler, evaluate_euler_denoiser,
        observe_euler_denoised, observe_euler_prediction, validate_euler_noise_generation_device,
    },
    rectified_flow_ancestral_step, standard_ancestral_step,
};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, DeviceId, ExecutionContext,
    RngCheckpoint, RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor,
    TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const EULER_ANCESTRAL_SAMPLER_ID: &str = "euler_ancestral";
pub const EULER_ANCESTRAL_FEATURE_ID: &str = "COMFY-MODEL-0180";
pub const EULER_ANCESTRAL_SOURCE_ORDINAL: u16 = 2;
pub const EULER_ANCESTRAL_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: EULER_ANCESTRAL_SAMPLER_ID,
    feature_id: EULER_ANCESTRAL_FEATURE_ID,
    source_ordinal: EULER_ANCESTRAL_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/euler_ancestral_comfy_model_0180",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerAncestralOptions {
    eta: f32,
    noise_scale: f32,
}

impl EulerAncestralOptions {
    pub fn new(eta: f32, noise_scale: f32) -> Result<Self, EulerAncestralError> {
        for (name, value) in [("eta", eta), ("s_noise", noise_scale)] {
            if !value.is_finite() {
                return Err(EulerAncestralError::InvalidOption { name, value });
            }
        }
        Ok(Self { eta, noise_scale })
    }

    pub const fn source_defaults() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
        }
    }

    pub const fn eta(self) -> f32 {
        self.eta
    }

    pub const fn noise_scale(self) -> f32 {
        self.noise_scale
    }
}

impl Default for EulerAncestralOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EulerAncestralMode {
    Standard,
    RectifiedFlow,
}

#[derive(Debug, Error)]
pub enum EulerAncestralError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    SamplingProfile(#[from] SamplingProfileError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error(transparent)]
    EulerFoundation(#[from] NativeDiffusionSamplerError),
    #[error("Euler ancestral requires sampler identity `euler_ancestral`, got {0:?}")]
    WrongSampler(String),
    #[error("Euler ancestral option {name} must be finite, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("Euler ancestral coefficient {coefficient} is invalid at step {step}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
    },
    #[error("Euler ancestral produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("Euler ancestral allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("native Euler ancestral noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
}

pub fn validate_euler_ancestral_generation_device(
    device: DeviceId,
) -> Result<(), EulerAncestralError> {
    validate_euler_noise_generation_device(device).map_err(|error| {
        EulerAncestralError::DeviceUnavailable {
            device,
            reason: error.to_string(),
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub fn sample_euler_ancestral<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: EulerAncestralOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<
    (
        SamplingTrace,
        EulerAncestralMode,
        RngCheckpoint,
        RngCheckpoint,
    ),
    EulerAncestralError,
>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != EULER_ANCESTRAL_SAMPLER_ID {
        return Err(EulerAncestralError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    let mode = if profile.prediction() == PredictionInterpretation::Flow {
        EulerAncestralMode::RectifiedFlow
    } else {
        EulerAncestralMode::Standard
    };
    let effective_noise_scale = if mode == EulerAncestralMode::RectifiedFlow {
        profile.scale_sampler_noise(options.noise_scale)?
    } else {
        options.noise_scale
    };
    let seed = plan.seed();
    let device = initial.descriptor().device();
    validate_euler_ancestral_generation_device(device)?;
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| EulerAncestralError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let (seed_transform, generation_placement) = if device == DeviceId::CPU {
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
    };
    let mut noise_transaction = noise_request.open_transaction(
        EULER_ANCESTRAL_NOISE_CONTRACT_ID,
        i128::from(seed),
        seed_transform,
        generation_placement,
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();

    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = pair
            .first()
            .copied()
            .ok_or(SamplingError::Overflow("Euler ancestral source sigma lookup"))?;
        let next_sigma = pair
            .get(1)
            .copied()
            .ok_or(SamplingError::Overflow("Euler ancestral target sigma lookup"))?;
        let current = session.current().clone();

        match mode {
            EulerAncestralMode::Standard => {
                let (sigma_down, sigma_up) =
                    standard_ancestral_step(sigma, next_sigma, options.eta).map_err(|_| {
                        EulerAncestralError::InvalidCoefficient {
                            step,
                            coefficient: "standard ancestral step",
                        }
                    })?;
                let EulerPrediction {
                    observed,
                    denoised,
                    derivative,
                } = observe_euler_prediction(
                    backend,
                    &mut session,
                    &current,
                    sigma,
                    step,
                    context,
                    &mut denoiser,
                    &mut callback,
                )?;
                let next = if sigma_down == 0.0 {
                    denoised
                } else {
                    let deterministic = advance_euler(
                        backend,
                        &current,
                        &derivative,
                        sigma_down - sigma,
                        step,
                        context,
                    )?;
                    add_ancestral_noise(
                        backend,
                        &deterministic,
                        effective_noise_scale * sigma_up,
                        step,
                        &mut noise_transaction,
                        context,
                    )?
                };
                observed.commit(next, context.cancellation)?;
            }
            EulerAncestralMode::RectifiedFlow => {
                let denoised =
                    evaluate_euler_denoiser(&current, sigma, step, &mut denoiser)?;
                let observed = observe_euler_denoised(
                    &mut session,
                    &current,
                    denoised.clone(),
                    sigma,
                    context,
                    &mut callback,
                )?;
                let next = if next_sigma == 0.0 {
                    denoised
                } else {
                    let (sigma_down, renoise_coefficient) =
                        rectified_flow_ancestral_step(sigma, next_sigma, options.eta).map_err(
                            |_| EulerAncestralError::InvalidCoefficient {
                                step,
                                coefficient: "rectified-flow ancestral step",
                            },
                        )?;
                    let deterministic = flow_euler_update(
                        backend,
                        &current,
                        &denoised,
                        sigma,
                        sigma_down,
                        step,
                        context,
                    )?;
                    if options.eta > 0.0 {
                        let alpha_next = 1.0 - next_sigma;
                        let alpha_down = 1.0 - sigma_down;
                        let rescale = checked_finite(
                            step,
                            "rectified-flow output scale",
                            alpha_next / alpha_down,
                        )?;
                        let deterministic = scale_tensor(
                            backend,
                            &deterministic,
                            rescale,
                            step,
                            context,
                        )?;
                        add_ancestral_noise(
                            backend,
                            &deterministic,
                            effective_noise_scale * renoise_coefficient,
                            step,
                            &mut noise_transaction,
                            context,
                        )?
                    } else {
                        deterministic
                    }
                };
                observed.commit(next, context.cancellation)?;
            }
        }
    }

    context.check()?;
    let trace = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok((trace, mode, noise_before, noise_after))
}

fn flow_euler_update(
    backend: &CpuBackend,
    current: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_down: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, EulerAncestralError> {
    let current_values = tensor_to_f32(backend, current, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let ratio = checked_finite(step, "rectified-flow sigma-down ratio", sigma_down / sigma)?;
    let mut values = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, (current, denoised)) in current_values
        .iter()
        .zip(denoised_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = ratio * current + (1.0 - ratio) * denoised;
        if !value.is_finite() {
            return Err(EulerAncestralError::NonFinite {
                step,
                stage: "rectified-flow Euler update",
                element,
            });
        }
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &values,
        context,
    )?)
}

fn scale_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    scale: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, EulerAncestralError> {
    let source = tensor_to_f32(backend, tensor, context)?;
    let mut values = backend.workspace_vec::<f32>(context, source.len())?;
    for (element, value) in source.iter().enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = value * scale;
        if !value.is_finite() {
            return Err(EulerAncestralError::NonFinite {
                step,
                stage: "rectified-flow rescale",
                element,
            });
        }
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        tensor.descriptor().shape(),
        &values,
        context,
    )?)
}

fn add_ancestral_noise(
    backend: &CpuBackend,
    deterministic: &Tensor,
    scale: f32,
    step: usize,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, EulerAncestralError> {
    let count = usize::try_from(deterministic.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let noise = transaction.draw_normal(count, context.cancellation)?;
    let deterministic_values = tensor_to_f32(backend, deterministic, context)?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for (element, (deterministic, noise)) in
        deterministic_values.iter().zip(noise).enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = (noise as f32).mul_add(scale, *deterministic);
        if !value.is_finite() {
            return Err(EulerAncestralError::NonFinite {
                step,
                stage: "ancestral noise update",
                element,
            });
        }
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        deterministic.descriptor().shape(),
        &values,
        context,
    )?)
}

fn checked_finite(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, EulerAncestralError> {
    if !value.is_finite() {
        return Err(EulerAncestralError::InvalidCoefficient { step, coefficient });
    }
    Ok(value)
}
