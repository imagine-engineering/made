use crate::{
    CompatibilityNoiseRequest, PredictionInterpretation, SamplerDefinition, SamplerRegistry,
    SamplingError, SamplingPlan, SamplingProfile, SamplingProgress, SamplingSession,
    SchedulerError, SchedulerRegistry, rectified_flow_ancestral_step, standard_ancestral_step,
};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DPM_2_ANCESTRAL_SAMPLER_ID: &str = "dpm_2_ancestral";
pub const DPM_2_ANCESTRAL_FEATURE_ID: &str = "COMFY-MODEL-0163";
pub const DPM_2_ANCESTRAL_SOURCE_ORDINAL: u16 = 9;
pub const DPM_2_ANCESTRAL_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPM_2_ANCESTRAL_SAMPLER_ID,
    feature_id: DPM_2_ANCESTRAL_FEATURE_ID,
    source_ordinal: DPM_2_ANCESTRAL_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpm_2_ancestral_comfy_model_0163",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dpm2AncestralOptions {
    pub eta: f32,
    pub noise_scale: f32,
    pub flow_noise_scale: f32,
}

impl Default for Dpm2AncestralOptions {
    fn default() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
            flow_noise_scale: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dpm2AncestralMode {
    Standard,
    RectifiedFlow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dpm2AncestralDenoiserStage {
    Primary,
    Midpoint,
}

#[derive(Debug, Error)]
pub enum Dpm2AncestralError {
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
    #[error("DPM-Solver-2 ancestral requires sampler identity `dpm_2_ancestral`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM-Solver-2 ancestral option {name} must be finite, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("DPM-Solver-2 ancestral denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: Dpm2AncestralDenoiserStage,
        reason: String,
    },
    #[error("DPM-Solver-2 ancestral denoiser contract changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: Dpm2AncestralDenoiserStage,
    },
    #[error("DPM-Solver-2 ancestral coefficient {coefficient} is invalid at step {step}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
    },
    #[error(
        "DPM-Solver-2 ancestral produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM-Solver-2 ancestral allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpm_2_ancestral<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: Dpm2AncestralOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, Dpm2AncestralDenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<
    (
        crate::SamplingTrace,
        Dpm2AncestralMode,
        RngCheckpoint,
        RngCheckpoint,
    ),
    Dpm2AncestralError,
>
where
    CallbackError: Display,
{
    context.check()?;
    validate_options(options)?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != DPM_2_ANCESTRAL_SAMPLER_ID {
        return Err(Dpm2AncestralError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let mode = if profile.prediction() == PredictionInterpretation::Flow {
        Dpm2AncestralMode::RectifiedFlow
    } else {
        Dpm2AncestralMode::Standard
    };
    let seed = plan.seed();
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| Dpm2AncestralError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let mut noise_transaction = noise_request.open_transaction(
        DPM_2_ANCESTRAL_NOISE_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();
    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = pair
            .first()
            .copied()
            .ok_or(SamplingError::Overflow("current sigma lookup"))?;
        let next_sigma = pair
            .get(1)
            .copied()
            .ok_or(SamplingError::Overflow("next sigma lookup"))?;
        let current = session.current().clone();
        let primary_denoised = denoiser(&current, sigma, step, Dpm2AncestralDenoiserStage::Primary)
            .map_err(|reason| Dpm2AncestralError::Denoiser {
                step,
                stage: Dpm2AncestralDenoiserStage::Primary,
                reason,
            })?;
        validate_denoiser_contract(
            &current,
            &primary_denoised,
            step,
            Dpm2AncestralDenoiserStage::Primary,
        )?;
        let (sigma_down, stochastic_sigma) = match mode {
            Dpm2AncestralMode::Standard => {
                standard_ancestral_step(sigma, next_sigma, options.eta).map_err(|_| {
                    Dpm2AncestralError::InvalidCoefficient {
                        step,
                        coefficient: "standard ancestral step",
                    }
                })?
            }
            Dpm2AncestralMode::RectifiedFlow => {
                rectified_flow_ancestral_step(sigma, next_sigma, options.eta).map_err(|_| {
                    Dpm2AncestralError::InvalidCoefficient {
                        step,
                        coefficient: "rectified-flow ancestral step",
                    }
                })?
            }
        };
        let observation = session.observe_step(
            &current,
            primary_denoised.clone(),
            context.cancellation,
            |progress, current, denoised| callback(progress, current, denoised),
        )?;
        let primary_derivative = derivative(
            backend,
            &current,
            &primary_denoised,
            sigma,
            step,
            "primary derivative",
            context,
        )?;

        let next = if sigma_down == 0.0 {
            euler_update(
                backend,
                &current,
                &primary_derivative,
                sigma_down - sigma,
                step,
                "terminal Euler update",
                context,
            )?
        } else {
            let log_sigma = sigma.ln();
            let sigma_mid = (log_sigma + (sigma_down.ln() - log_sigma) * 0.5).exp();
            if !sigma_mid.is_finite() || sigma_mid <= 0.0 {
                return Err(Dpm2AncestralError::InvalidCoefficient {
                    step,
                    coefficient: "sigma_mid",
                });
            }
            let midpoint_latent = euler_update(
                backend,
                &current,
                &primary_derivative,
                sigma_mid - sigma,
                step,
                "midpoint update",
                context,
            )?;
            let midpoint_denoised = denoiser(
                &midpoint_latent,
                sigma_mid,
                step,
                Dpm2AncestralDenoiserStage::Midpoint,
            )
            .map_err(|reason| Dpm2AncestralError::Denoiser {
                step,
                stage: Dpm2AncestralDenoiserStage::Midpoint,
                reason,
            })?;
            validate_denoiser_contract(
                &current,
                &midpoint_denoised,
                step,
                Dpm2AncestralDenoiserStage::Midpoint,
            )?;
            let midpoint_derivative = derivative(
                backend,
                &midpoint_latent,
                &midpoint_denoised,
                sigma_mid,
                step,
                "midpoint derivative",
                context,
            )?;
            let deterministic = euler_update(
                backend,
                &current,
                &midpoint_derivative,
                sigma_down - sigma,
                step,
                "second-order update",
                context,
            )?;
            let stochastic_noise =
                draw_noise(backend, &current, &mut noise_transaction, step, context)?;
            let next = add_stochastic_update(
                backend,
                &deterministic,
                &stochastic_noise,
                next_sigma,
                sigma_down,
                stochastic_sigma,
                mode,
                options,
                step,
                context,
            )?;
            next
        };

        observation.commit(next, context.cancellation)?;
    }

    let sampling = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok((sampling, mode, noise_before, noise_after))
}

fn validate_options(options: Dpm2AncestralOptions) -> Result<(), Dpm2AncestralError> {
    for (name, value) in [
        ("eta", options.eta),
        ("noise_scale", options.noise_scale),
        ("flow_noise_scale", options.flow_noise_scale),
    ] {
        if !value.is_finite() {
            return Err(Dpm2AncestralError::InvalidOption { name, value });
        }
    }
    Ok(())
}

fn validate_denoiser_contract(
    current: &Tensor,
    denoised: &Tensor,
    step: usize,
    stage: Dpm2AncestralDenoiserStage,
) -> Result<(), Dpm2AncestralError> {
    if current.descriptor() != denoised.descriptor() {
        return Err(Dpm2AncestralError::DenoiserContract { step, stage });
    }
    Ok(())
}

fn derivative(
    backend: &CpuBackend,
    input: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpm2AncestralError> {
    let input_values = tensor_to_f32(backend, input, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let mut values = backend.workspace_vec::<f32>(context, input_values.len())?;
    for (element, (input, denoised)) in input_values.iter().zip(denoised_values.iter()).enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = (input - denoised) / sigma;
        if !value.is_finite() {
            return Err(Dpm2AncestralError::NonFinite {
                step,
                stage,
                element,
            });
        }
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        input.descriptor().shape(),
        &values,
        context,
    )?)
}

fn euler_update(
    backend: &CpuBackend,
    current: &Tensor,
    derivative: &Tensor,
    delta: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpm2AncestralError> {
    let current_values = tensor_to_f32(backend, current, context)?;
    let derivative_values = tensor_to_f32(backend, derivative, context)?;
    let mut values = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, (current, derivative)) in current_values
        .iter()
        .zip(derivative_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = current + derivative * delta;
        if !value.is_finite() {
            return Err(Dpm2AncestralError::NonFinite {
                step,
                stage,
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

fn draw_noise(
    backend: &CpuBackend,
    current: &Tensor,
    transaction: &mut CompatibilityRngTransaction,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpm2AncestralError> {
    let count = usize::try_from(current.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let normal = transaction.draw_normal(count, context.cancellation)?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for (element, value) in normal.into_iter().enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = value as f32;
        if !value.is_finite() {
            return Err(Dpm2AncestralError::NonFinite {
                step,
                stage: "stochastic noise",
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

#[allow(clippy::too_many_arguments)]
fn add_stochastic_update(
    backend: &CpuBackend,
    deterministic: &Tensor,
    noise: &Tensor,
    next_sigma: f32,
    sigma_down: f32,
    stochastic_sigma: f32,
    mode: Dpm2AncestralMode,
    options: Dpm2AncestralOptions,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpm2AncestralError> {
    let deterministic_values = tensor_to_f32(backend, deterministic, context)?;
    let noise_values = tensor_to_f32(backend, noise, context)?;
    let deterministic_scale = match mode {
        Dpm2AncestralMode::Standard => 1.0,
        Dpm2AncestralMode::RectifiedFlow => {
            let denominator = 1.0 - sigma_down;
            if denominator == 0.0 {
                return Err(Dpm2AncestralError::InvalidCoefficient {
                    step,
                    coefficient: "rectified-flow output scale",
                });
            }
            (1.0 - next_sigma) / denominator
        }
    };
    let source_noise_scale = if mode == Dpm2AncestralMode::RectifiedFlow {
        options.noise_scale * options.flow_noise_scale
    } else {
        options.noise_scale
    };
    let mut values = backend.workspace_vec::<f32>(context, deterministic_values.len())?;
    for (element, (deterministic, noise)) in deterministic_values
        .iter()
        .zip(noise_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let stochastic = (noise * source_noise_scale) * stochastic_sigma;
        let value = deterministic_scale * deterministic + stochastic;
        if !value.is_finite() {
            return Err(Dpm2AncestralError::NonFinite {
                step,
                stage: "stochastic update",
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
