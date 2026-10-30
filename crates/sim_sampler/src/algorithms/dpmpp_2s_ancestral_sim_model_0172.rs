use crate::{
    CompatibilityNoiseRequest, PredictionInterpretation, SamplerDefinition, SamplerRegistry,
    SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    rectified_flow_ancestral_step, standard_ancestral_step,
};
use sim_tensor::{
    CpuBackend, DeviceId, ExecutionContext, RngCheckpoint, RngCompatibilityError,
    RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_2S_ANCESTRAL_SAMPLER_ID: &str = "dpmpp_2s_ancestral";
pub const DPMPP_2S_ANCESTRAL_FEATURE_ID: &str = "COMFY-MODEL-0172";
pub const DPMPP_2S_ANCESTRAL_SOURCE_ORDINAL: u16 = 13;
pub const DPMPP_2S_ANCESTRAL_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_2S_ANCESTRAL_SAMPLER_ID,
    feature_id: DPMPP_2S_ANCESTRAL_FEATURE_ID,
    source_ordinal: DPMPP_2S_ANCESTRAL_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_2s_ancestral_comfy_model_0172",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dpmpp2sAncestralOptions {
    pub eta: f32,
    pub noise_scale: f32,
}

impl Default for Dpmpp2sAncestralOptions {
    fn default() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dpmpp2sAncestralMode {
    Standard,
    RectifiedFlow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dpmpp2sAncestralDenoiserStage {
    Primary,
    SecondOrder,
}

#[derive(Debug, Error)]
pub enum Dpmpp2sAncestralError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Profile(#[from] SamplingProfileError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error("DPM-Solver++(2S) ancestral requires sampler identity `dpmpp_2s_ancestral`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM-Solver++(2S) ancestral option {name} must be finite and nonnegative, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("DPM-Solver++(2S) ancestral denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: Dpmpp2sAncestralDenoiserStage,
        reason: String,
    },
    #[error("DPM-Solver++(2S) ancestral denoiser contract changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: Dpmpp2sAncestralDenoiserStage,
    },
    #[error(
        "DPM-Solver++(2S) ancestral coefficient {coefficient} is invalid at step {step}: {value}"
    )]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error(
        "DPM-Solver++(2S) ancestral produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM-Solver++(2S) ancestral allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_2s_ancestral<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: Dpmpp2sAncestralOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(
        &Tensor,
        f32,
        usize,
        Dpmpp2sAncestralDenoiserStage,
    ) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<
    (
        SamplingTrace,
        Dpmpp2sAncestralMode,
        RngCheckpoint,
        RngCheckpoint,
    ),
    Dpmpp2sAncestralError,
>
where
    CallbackError: Display,
{
    context.check()?;
    validate_options(options)?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != DPMPP_2S_ANCESTRAL_SAMPLER_ID {
        return Err(Dpmpp2sAncestralError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }

    let mode = if profile.prediction() == PredictionInterpretation::Flow {
        Dpmpp2sAncestralMode::RectifiedFlow
    } else {
        Dpmpp2sAncestralMode::Standard
    };
    let mut adjusted_sigmas = Vec::new();
    adjusted_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| Dpmpp2sAncestralError::OutOfMemory("adjusted sigma schedule"))?;
    adjusted_sigmas.extend_from_slice(sigmas);
    profile.adjust_first_sigma_for_snr(&mut adjusted_sigmas)?;
    let device = initial.descriptor().device();
    if device != DeviceId::CPU {
        return Err(Dpmpp2sAncestralError::Tensor(TensorError::DeviceMismatch {
            expected: DeviceId::CPU,
            actual: device,
        }));
    }
    let seed = plan.seed();
    let mut session = SamplingSession::new(plan, adjusted_sigmas.clone(), initial)?;
    let effective_noise_scale = match mode {
        Dpmpp2sAncestralMode::Standard => options.noise_scale,
        Dpmpp2sAncestralMode::RectifiedFlow => profile.scale_sampler_noise(options.noise_scale)?,
    };
    let mut noise_transaction = noise_request.open_transaction(
        DPMPP_2S_ANCESTRAL_NOISE_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: device,
        },
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();

    for (step, pair) in adjusted_sigmas.windows(2).enumerate() {
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
        let primary_denoised = denoiser(
            &current,
            sigma,
            step,
            Dpmpp2sAncestralDenoiserStage::Primary,
        )
        .map_err(|reason| Dpmpp2sAncestralError::Denoiser {
            step,
            stage: Dpmpp2sAncestralDenoiserStage::Primary,
            reason,
        })?;
        validate_denoiser_contract(
            &current,
            &primary_denoised,
            step,
            Dpmpp2sAncestralDenoiserStage::Primary,
        )?;
        let ancestral_coefficients = match mode {
            Dpmpp2sAncestralMode::Standard => {
                let (sigma_down, sigma_up) =
                    standard_ancestral_step(sigma, next_sigma, options.eta).map_err(|_| {
                        Dpmpp2sAncestralError::InvalidCoefficient {
                            step,
                            coefficient: "standard ancestral step",
                            value: f32::NAN,
                        }
                    })?;
                (sigma_down, sigma_up)
            }
            Dpmpp2sAncestralMode::RectifiedFlow => rectified_flow_ancestral_step(
                sigma,
                next_sigma,
                options.eta,
            )
            .map_err(|_| Dpmpp2sAncestralError::InvalidCoefficient {
                step,
                coefficient: "rectified-flow ancestral step",
                value: f32::NAN,
            })?,
        };
        let observed = session.observe_step(
            &current,
            primary_denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;

        let (mut deterministic, stochastic_coefficient, should_draw_noise) = match mode {
            Dpmpp2sAncestralMode::Standard => {
                let (sigma_down, sigma_up) = ancestral_coefficients;
                let deterministic = if sigma_down == 0.0 {
                    terminal_euler_update(
                        backend,
                        &current,
                        &primary_denoised,
                        sigma,
                        sigma_down,
                        step,
                        context,
                    )?
                } else {
                    standard_second_order_update(
                        backend,
                        &current,
                        &primary_denoised,
                        sigma,
                        sigma_down,
                        step,
                        context,
                        &mut denoiser,
                    )?
                };
                (deterministic, sigma_up, next_sigma > 0.0)
            }
            Dpmpp2sAncestralMode::RectifiedFlow => {
                let (sigma_down, renoise_coefficient) = ancestral_coefficients;
                let mut deterministic = if next_sigma == 0.0 {
                    terminal_euler_update(
                        backend,
                        &current,
                        &primary_denoised,
                        sigma,
                        sigma_down,
                        step,
                        context,
                    )?
                } else {
                    flow_second_order_update(
                        backend,
                        &current,
                        &primary_denoised,
                        sigma,
                        sigma_down,
                        step,
                        context,
                        &mut denoiser,
                    )?
                };
                if next_sigma > 0.0 && options.eta > 0.0 {
                    let alpha_next = 1.0 - next_sigma;
                    let alpha_down = 1.0 - sigma_down;
                    let output_scale = checked_finite(
                        step,
                        "rectified-flow output scale",
                        alpha_next / alpha_down,
                    )?;
                    deterministic = scale_tensor(
                        backend,
                        &deterministic,
                        output_scale,
                        step,
                        "rectified-flow deterministic rescale",
                        context,
                    )?;
                }
                (
                    deterministic,
                    renoise_coefficient,
                    next_sigma > 0.0 && options.eta > 0.0,
                )
            }
        };

        if should_draw_noise {
            let count = usize::try_from(deterministic.descriptor().element_count()?)
                .map_err(|_| TensorError::ShapeOverflow)?;
            let noise = noise_transaction.draw_normal(count, context.cancellation)?;
            deterministic = add_noise(
                backend,
                &deterministic,
                &noise,
                effective_noise_scale * stochastic_coefficient,
                step,
                context,
            )?;
        }
        observed.commit(deterministic, context.cancellation)?;
    }

    context.check()?;
    let trace = session.finish()?;
    let noise_after = noise_transaction.commit();
    Ok((trace, mode, noise_before, noise_after))
}

fn validate_options(options: Dpmpp2sAncestralOptions) -> Result<(), Dpmpp2sAncestralError> {
    for (name, value) in [("eta", options.eta), ("noise scale", options.noise_scale)] {
        if !value.is_finite() {
            return Err(Dpmpp2sAncestralError::InvalidOption { name, value });
        }
    }
    Ok(())
}

fn validate_denoiser_contract(
    current: &Tensor,
    denoised: &Tensor,
    step: usize,
    stage: Dpmpp2sAncestralDenoiserStage,
) -> Result<(), Dpmpp2sAncestralError> {
    if current.descriptor() != denoised.descriptor() {
        return Err(Dpmpp2sAncestralError::DenoiserContract { step, stage });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn standard_second_order_update(
    backend: &CpuBackend,
    current: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_down: f32,
    step: usize,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(
        &Tensor,
        f32,
        usize,
        Dpmpp2sAncestralDenoiserStage,
    ) -> Result<Tensor, String>,
) -> Result<Tensor, Dpmpp2sAncestralError> {
    let current_time = checked_finite(step, "current time", -sigma.ln())?;
    let next_time = checked_finite(step, "down time", -sigma_down.ln())?;
    let step_size = checked_positive(step, "step size", next_time - current_time)?;
    let second_sigma = checked_positive(
        step,
        "second-order sigma",
        (-(current_time + 0.5 * step_size)).exp(),
    )?;
    let second_input = combine_two(
        backend,
        current,
        second_sigma / sigma,
        denoised,
        -(-0.5 * step_size).exp_m1(),
        step,
        "standard second-order input",
        context,
    )?;
    let second_denoised = denoiser(
        &second_input,
        second_sigma,
        step,
        Dpmpp2sAncestralDenoiserStage::SecondOrder,
    )
    .map_err(|reason| Dpmpp2sAncestralError::Denoiser {
        step,
        stage: Dpmpp2sAncestralDenoiserStage::SecondOrder,
        reason,
    })?;
    validate_denoiser_contract(
        current,
        &second_denoised,
        step,
        Dpmpp2sAncestralDenoiserStage::SecondOrder,
    )?;
    combine_two(
        backend,
        current,
        sigma_down / sigma,
        &second_denoised,
        -(-step_size).exp_m1(),
        step,
        "standard second-order output",
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn flow_second_order_update(
    backend: &CpuBackend,
    current: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_down: f32,
    step: usize,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(
        &Tensor,
        f32,
        usize,
        Dpmpp2sAncestralDenoiserStage,
    ) -> Result<Tensor, String>,
) -> Result<Tensor, Dpmpp2sAncestralError> {
    let second_sigma = if sigma == 1.0 {
        0.9999
    } else {
        let current_lambda = checked_finite(
            step,
            "rectified-flow current lambda",
            ((1.0 - sigma) / sigma).ln(),
        )?;
        let down_lambda = checked_finite(
            step,
            "rectified-flow down lambda",
            ((1.0 - sigma_down) / sigma_down).ln(),
        )?;
        let middle_lambda = current_lambda + 0.5 * (down_lambda - current_lambda);
        checked_positive(
            step,
            "rectified-flow second-order sigma",
            1.0 / (middle_lambda.exp() + 1.0),
        )?
    };
    let second_ratio = checked_nonnegative(
        step,
        "rectified-flow second-order ratio",
        second_sigma / sigma,
    )?;
    let second_input = combine_two(
        backend,
        current,
        second_ratio,
        denoised,
        1.0 - second_ratio,
        step,
        "rectified-flow second-order input",
        context,
    )?;
    let second_denoised = denoiser(
        &second_input,
        second_sigma,
        step,
        Dpmpp2sAncestralDenoiserStage::SecondOrder,
    )
    .map_err(|reason| Dpmpp2sAncestralError::Denoiser {
        step,
        stage: Dpmpp2sAncestralDenoiserStage::SecondOrder,
        reason,
    })?;
    validate_denoiser_contract(
        current,
        &second_denoised,
        step,
        Dpmpp2sAncestralDenoiserStage::SecondOrder,
    )?;
    let down_ratio = checked_nonnegative(step, "rectified-flow down ratio", sigma_down / sigma)?;
    combine_two(
        backend,
        current,
        down_ratio,
        &second_denoised,
        1.0 - down_ratio,
        step,
        "rectified-flow second-order output",
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn terminal_euler_update(
    backend: &CpuBackend,
    current: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_down: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpmpp2sAncestralError> {
    let derivative_scale = (sigma_down - sigma) / sigma;
    combine_two(
        backend,
        current,
        1.0 + derivative_scale,
        denoised,
        -derivative_scale,
        step,
        "terminal Euler output",
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn combine_two(
    backend: &CpuBackend,
    left: &Tensor,
    left_scale: f32,
    right: &Tensor,
    right_scale: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpmpp2sAncestralError> {
    if left.descriptor() != right.descriptor() {
        return Err(Dpmpp2sAncestralError::DenoiserContract {
            step,
            stage: Dpmpp2sAncestralDenoiserStage::SecondOrder,
        });
    }
    let left_values = tensor_to_f32(backend, left, context)?;
    let right_values = tensor_to_f32(backend, right, context)?;
    let mut values = backend.workspace_vec::<f32>(context, left_values.len())?;
    for (element, (left, right)) in left_values
        .iter()
        .copied()
        .zip(right_values.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = left_scale * left + right_scale * right;
        if !value.is_finite() {
            return Err(Dpmpp2sAncestralError::NonFinite {
                step,
                stage,
                element,
            });
        }
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        left.descriptor().shape(),
        &values,
        context,
    )?)
}

fn scale_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    scale: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpmpp2sAncestralError> {
    let input = tensor_to_f32(backend, tensor, context)?;
    let mut values = backend.workspace_vec::<f32>(context, input.len())?;
    for (element, value) in input.iter().copied().enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = scale * value;
        if !value.is_finite() {
            return Err(Dpmpp2sAncestralError::NonFinite {
                step,
                stage,
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

fn add_noise(
    backend: &CpuBackend,
    deterministic: &Tensor,
    noise: &[f64],
    scale: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpmpp2sAncestralError> {
    let deterministic_values = tensor_to_f32(backend, deterministic, context)?;
    if deterministic_values.len() != noise.len() {
        return Err(Dpmpp2sAncestralError::Tensor(TensorError::ShapeOverflow));
    }
    let mut values = backend.workspace_vec::<f32>(context, deterministic_values.len())?;
    for (element, (deterministic, noise)) in deterministic_values
        .iter()
        .copied()
        .zip(noise.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = deterministic + (noise as f32) * scale;
        if !value.is_finite() {
            return Err(Dpmpp2sAncestralError::NonFinite {
                step,
                stage: "stochastic output",
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
) -> Result<f32, Dpmpp2sAncestralError> {
    if !value.is_finite() {
        return Err(Dpmpp2sAncestralError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_positive(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, Dpmpp2sAncestralError> {
    let value = checked_finite(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(Dpmpp2sAncestralError::InvalidCoefficient {
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
) -> Result<f32, Dpmpp2sAncestralError> {
    let value = checked_finite(step, coefficient, value)?;
    if value < 0.0 {
        return Err(Dpmpp2sAncestralError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
