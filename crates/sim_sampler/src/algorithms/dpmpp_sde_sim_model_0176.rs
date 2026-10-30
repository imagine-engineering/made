use crate::{
    BrownianNoiseIntervalAddress, CompatibilityNoiseRequest, NoiseError, SamplerDefinition,
    SamplerRegistry, SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError,
    SamplingProgress, SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    standard_ancestral_step,
};
use sim_tensor::{
    BrownianTree, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint, RngCompatibilityError,
    RngError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_SDE_SAMPLER_ID: &str = "dpmpp_sde";
pub const DPMPP_SDE_FEATURE_ID: &str = "COMFY-MODEL-0176";
pub const DPMPP_SDE_SOURCE_ORDINAL: u16 = 15;
pub const DPMPP_SDE_BROWNIAN_CONTRACT_ID: &str = "COMFY-RNG-DED616CC3432";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_SDE_SAMPLER_ID,
    feature_id: DPMPP_SDE_FEATURE_ID,
    source_ordinal: DPMPP_SDE_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_sde_comfy_model_0176",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpmppSdeOptions {
    pub eta: f32,
    pub noise_scale: f32,
    pub r: f32,
}

impl Default for DpmppSdeOptions {
    fn default() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
            r: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DpmppSdeDenoiserStage {
    Primary,
    Intermediate,
}

#[derive(Debug, Error)]
pub enum DpmppSdeError {
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
    Noise(#[from] NoiseError),
    #[error(transparent)]
    Rng(#[from] RngError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error("DPM-Solver++ SDE requires sampler identity `dpmpp_sde`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM-Solver++ SDE option {name} is invalid: {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("DPM-Solver++ SDE denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: DpmppSdeDenoiserStage,
        reason: String,
    },
    #[error("DPM-Solver++ SDE denoiser contract changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: DpmppSdeDenoiserStage,
    },
    #[error("DPM-Solver++ SDE coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error(
        "DPM-Solver++ SDE produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM-Solver++ SDE generation placement outputs {actual:?}, expected {expected:?}")]
    PlacementOutputMismatch {
        expected: DeviceId,
        actual: DeviceId,
    },
    #[error("DPM-Solver++ SDE arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("DPM-Solver++ SDE allocation failed for {0}")]
    OutOfMemory(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_sde<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    options: DpmppSdeOptions,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize, DpmppSdeDenoiserStage) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), DpmppSdeError>
where
    CallbackError: Display,
{
    let output_device = initial.descriptor().device();
    sample_dpmpp_sde_with_generation_placement(
        backend,
        plan,
        profile,
        initial,
        sigmas,
        options,
        noise_request,
        RngGenerationPlacement::CpuSeededTransfer { output_device },
        context,
        denoiser,
        callback,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_dpmpp_sde_with_generation_placement<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    options: DpmppSdeOptions,
    noise_request: CompatibilityNoiseRequest,
    generation_placement: RngGenerationPlacement,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, DpmppSdeDenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), DpmppSdeError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        profile.identity(),
    )?;
    if plan.sampler().as_str() != DPMPP_SDE_SAMPLER_ID {
        return Err(DpmppSdeError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    if sigmas.len() <= 1 {
        return short_sampling(sigmas, initial);
    }
    validate_options(options)?;
    validate_dpmpp_sde_generation_placement(initial.descriptor().device(), generation_placement)?;

    let (brownian_minimum, brownian_maximum) = brownian_bounds(sigmas)?;
    let mut adjusted_sigmas = Vec::new();
    adjusted_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| DpmppSdeError::OutOfMemory("adjusted sigma schedule"))?;
    adjusted_sigmas.extend_from_slice(sigmas);
    profile.adjust_first_sigma_for_snr(&mut adjusted_sigmas)?;
    let effective_noise_scale = profile.scale_sampler_noise(options.noise_scale)?;
    let seed = plan.seed();
    let mut session = SamplingSession::new(plan, adjusted_sigmas.clone(), initial)?;
    let element_count = usize::try_from(session.current().descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let mut transaction = noise_request.open_transaction(
        DPMPP_SDE_BROWNIAN_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::TorchSigned64,
        generation_placement,
        None,
        context.cancellation,
    )?;
    let noise_before = transaction.checkpoint();
    let mut initial_brownian = Vec::new();
    initial_brownian
        .try_reserve_exact(element_count)
        .map_err(|_| DpmppSdeError::OutOfMemory("Brownian initial value"))?;
    initial_brownian.resize(element_count, 0.0);
    let mut brownian_tree = transaction.brownian_tree(
        f64::from(brownian_minimum),
        initial_brownian,
        f64::from(brownian_maximum),
        context.cancellation,
    )?;
    let stochastic = options.eta > 0.0 && effective_noise_scale > 0.0;

    for (step, pair) in adjusted_sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = pair
            .first()
            .copied()
            .ok_or(DpmppSdeError::Overflow("current sigma lookup"))?;
        let next_sigma = pair
            .get(1)
            .copied()
            .ok_or(DpmppSdeError::Overflow("next sigma lookup"))?;
        let current = session.current().clone();
        let primary =
            denoiser(&current, sigma, step, DpmppSdeDenoiserStage::Primary).map_err(|reason| {
                DpmppSdeError::Denoiser {
                    step,
                    stage: DpmppSdeDenoiserStage::Primary,
                    reason,
                }
            })?;
        validate_denoiser(&current, &primary, step, DpmppSdeDenoiserStage::Primary)?;
        let observed = session.observe_step(
            &current,
            primary.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;

        let next = if next_sigma == 0.0 {
            validate_finite(backend, &primary, step, "terminal denoiser", context)?;
            primary.clone()
        } else {
            let lambda_source =
                checked_finite(step, "source half-log-SNR", profile.half_log_snr(sigma)?)?;
            let lambda_target = checked_finite(
                step,
                "target half-log-SNR",
                profile.half_log_snr(next_sigma)?,
            )?;
            let step_size = checked_positive(step, "step size", lambda_target - lambda_source)?;
            let intermediate_lambda = checked_finite(
                step,
                "intermediate half-log-SNR",
                lambda_source + options.r * step_size,
            )?;
            let combination_factor =
                checked_finite(step, "combination factor", 1.0 / (2.0 * options.r))?;
            let intermediate_sigma = checked_positive(
                step,
                "intermediate sigma",
                profile.sigma_from_half_log_snr(intermediate_lambda)?,
            )?;
            let alpha_source = checked_positive(step, "source alpha", sigma * lambda_source.exp())?;
            let alpha_intermediate = checked_positive(
                step,
                "intermediate alpha",
                intermediate_sigma * intermediate_lambda.exp(),
            )?;
            let alpha_target =
                checked_positive(step, "target alpha", next_sigma * lambda_target.exp())?;

            let (first_down, first_up) = checked_standard_ancestral_step(
                (-lambda_source).exp(),
                (-intermediate_lambda).exp(),
                options.eta,
                step,
                "first ancestral step",
            )?;
            let first_adjusted_lambda =
                checked_finite(step, "first adjusted half-log-SNR", -first_down.ln())?;
            let first_adjusted_step = checked_positive(
                step,
                "first adjusted step",
                first_adjusted_lambda - lambda_source,
            )?;
            let first_latent_weight = checked_finite(
                step,
                "first latent weight",
                alpha_intermediate / alpha_source * (-first_adjusted_step).exp(),
            )?;
            let first_denoised_weight = checked_finite(
                step,
                "first denoised weight",
                -alpha_intermediate * (-first_adjusted_step).exp_m1(),
            )?;
            let mut intermediate_input = combine_two(
                backend,
                &current,
                first_latent_weight,
                &primary,
                first_denoised_weight,
                step,
                "first deterministic update",
                context,
            )?;
            if stochastic {
                let first_noise = normalized_brownian_increment(
                    &mut brownian_tree,
                    sigma,
                    intermediate_sigma,
                    step,
                    context,
                )?;
                intermediate_input = add_noise(
                    backend,
                    &intermediate_input,
                    &first_noise,
                    alpha_intermediate * effective_noise_scale * first_up,
                    step,
                    "first stochastic update",
                    context,
                )?;
            }
            let intermediate = denoiser(
                &intermediate_input,
                intermediate_sigma,
                step,
                DpmppSdeDenoiserStage::Intermediate,
            )
            .map_err(|reason| DpmppSdeError::Denoiser {
                step,
                stage: DpmppSdeDenoiserStage::Intermediate,
                reason,
            })?;
            validate_denoiser(
                &current,
                &intermediate,
                step,
                DpmppSdeDenoiserStage::Intermediate,
            )?;

            let (second_down, second_up) = checked_standard_ancestral_step(
                (-lambda_source).exp(),
                (-lambda_target).exp(),
                options.eta,
                step,
                "second ancestral step",
            )?;
            let second_adjusted_lambda =
                checked_finite(step, "second adjusted half-log-SNR", -second_down.ln())?;
            let second_adjusted_step = checked_positive(
                step,
                "second adjusted step",
                second_adjusted_lambda - lambda_source,
            )?;
            let combined_denoised = combine_two(
                backend,
                &primary,
                1.0 - combination_factor,
                &intermediate,
                combination_factor,
                step,
                "combined denoiser",
                context,
            )?;
            let second_latent_weight = checked_finite(
                step,
                "second latent weight",
                alpha_target / alpha_source * (-second_adjusted_step).exp(),
            )?;
            let second_denoised_weight = checked_finite(
                step,
                "second denoised weight",
                -alpha_target * (-second_adjusted_step).exp_m1(),
            )?;
            let mut next = combine_two(
                backend,
                &current,
                second_latent_weight,
                &combined_denoised,
                second_denoised_weight,
                step,
                "second deterministic update",
                context,
            )?;
            if stochastic {
                let second_noise = normalized_brownian_increment(
                    &mut brownian_tree,
                    sigma,
                    next_sigma,
                    step,
                    context,
                )?;
                next = add_noise(
                    backend,
                    &next,
                    &second_noise,
                    alpha_target * effective_noise_scale * second_up,
                    step,
                    "second stochastic update",
                    context,
                )?;
            }
            next
        };
        observed.commit(next, context.cancellation)?;
    }

    context.check()?;
    let trace = session.finish()?;
    let noise_after = transaction.commit();
    Ok((trace, Some((noise_before, noise_after))))
}

pub(crate) fn validate_dpmpp_sde_generation_placement(
    output_device: DeviceId,
    placement: RngGenerationPlacement,
) -> Result<(), DpmppSdeError> {
    if placement.output_device() != output_device {
        return Err(DpmppSdeError::PlacementOutputMismatch {
            expected: output_device,
            actual: placement.output_device(),
        });
    }
    Ok(())
}

fn short_sampling(
    sigmas: &[f32],
    initial: Tensor,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), DpmppSdeError> {
    let mut traced_sigmas = Vec::new();
    traced_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| DpmppSdeError::OutOfMemory("short sigma trace"))?;
    traced_sigmas.extend_from_slice(sigmas);
    let mut latents = Vec::new();
    latents
        .try_reserve_exact(1)
        .map_err(|_| DpmppSdeError::OutOfMemory("short latent trace"))?;
    latents.push(initial);
    Ok((
        SamplingTrace {
            sigmas: traced_sigmas,
            denoiser_evaluations: Vec::new(),
            latents,
        },
        None,
    ))
}

fn validate_options(options: DpmppSdeOptions) -> Result<(), DpmppSdeError> {
    for (name, value) in [
        ("eta", options.eta),
        ("noise scale", options.noise_scale),
        ("r", options.r),
    ] {
        if !value.is_finite() {
            return Err(DpmppSdeError::InvalidOption { name, value });
        }
    }
    Ok(())
}

fn brownian_bounds(sigmas: &[f32]) -> Result<(f32, f32), DpmppSdeError> {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for sigma in sigmas.iter().copied().filter(|sigma| *sigma > 0.0) {
        let sigma = checked_positive(0, "Brownian sigma bound", sigma)?;
        minimum = minimum.min(sigma);
        maximum = maximum.max(sigma);
    }
    if !minimum.is_finite() || !maximum.is_finite() || minimum >= maximum {
        return Err(DpmppSdeError::InvalidCoefficient {
            step: 0,
            coefficient: "Brownian sigma range",
            value: maximum - minimum,
        });
    }
    Ok((minimum, maximum))
}

fn checked_standard_ancestral_step(
    sigma_from: f32,
    sigma_to: f32,
    eta: f32,
    step: usize,
    coefficient: &'static str,
) -> Result<(f32, f32), DpmppSdeError> {
    let (sigma_down, sigma_up) = standard_ancestral_step(sigma_from, sigma_to, eta).map_err(
        |_| DpmppSdeError::InvalidCoefficient {
            step,
            coefficient,
            value: sigma_from,
        },
    )?;
    Ok((
        checked_positive(step, coefficient, sigma_down)?,
        checked_nonnegative(step, coefficient, sigma_up)?,
    ))
}

fn normalized_brownian_increment(
    tree: &mut BrownianTree,
    start: f32,
    end: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, DpmppSdeError> {
    let address = BrownianNoiseIntervalAddress::new(
        start,
        end,
        u32::try_from(step).map_err(|_| DpmppSdeError::Overflow("Brownian interval step"))?,
    )?;
    let (lower, upper) = address.canonical_interval();
    let normalization = f64::from(upper - lower).sqrt();
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(DpmppSdeError::InvalidCoefficient {
            step,
            coefficient: "Brownian normalization",
            value: normalization as f32,
        });
    }
    let sign = if address.reverse { -1.0_f64 } else { 1.0 };
    let increment = tree.increment(f64::from(lower), f64::from(upper), context.cancellation)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(increment.len())
        .map_err(|_| DpmppSdeError::OutOfMemory("Brownian increment"))?;
    for (element, value) in increment.into_iter().enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = (sign * value / normalization) as f32;
        if !value.is_finite() {
            return Err(DpmppSdeError::NonFinite {
                step,
                stage: "Brownian increment",
                element,
            });
        }
        values.push(value);
    }
    Ok(values)
}

fn validate_denoiser(
    current: &Tensor,
    denoised: &Tensor,
    step: usize,
    stage: DpmppSdeDenoiserStage,
) -> Result<(), DpmppSdeError> {
    if current.descriptor() != denoised.descriptor() {
        return Err(DpmppSdeError::DenoiserContract { step, stage });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn combine_two(
    backend: &CpuBackend,
    left: &Tensor,
    left_weight: f32,
    right: &Tensor,
    right_weight: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, DpmppSdeError> {
    if left.descriptor() != right.descriptor() {
        return Err(DpmppSdeError::DenoiserContract {
            step,
            stage: DpmppSdeDenoiserStage::Intermediate,
        });
    }
    let left_values = tensor_to_f32(backend, left, context)?;
    let right_values = tensor_to_f32(backend, right, context)?;
    let mut output = backend.workspace_vec::<f32>(context, left_values.len())?;
    for (element, (left, right)) in left_values
        .iter()
        .copied()
        .zip(right_values.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = left_weight * left + right_weight * right;
        if !value.is_finite() {
            return Err(DpmppSdeError::NonFinite {
                step,
                stage,
                element,
            });
        }
        output.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        left.descriptor().shape(),
        &output,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn add_noise(
    backend: &CpuBackend,
    deterministic: &Tensor,
    noise: &[f32],
    scale: f32,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, DpmppSdeError> {
    let deterministic_values = tensor_to_f32(backend, deterministic, context)?;
    if deterministic_values.len() != noise.len() {
        return Err(DpmppSdeError::Overflow("Brownian element count"));
    }
    let mut output = backend.workspace_vec::<f32>(context, deterministic_values.len())?;
    for (element, (deterministic, noise)) in deterministic_values
        .iter()
        .copied()
        .zip(noise.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = deterministic + scale * noise;
        if !value.is_finite() {
            return Err(DpmppSdeError::NonFinite {
                step,
                stage,
                element,
            });
        }
        output.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        deterministic.descriptor().shape(),
        &output,
        context,
    )?)
}

fn validate_finite(
    backend: &CpuBackend,
    tensor: &Tensor,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), DpmppSdeError> {
    for (element, value) in tensor_to_f32(backend, tensor, context)?.iter().enumerate() {
        if !value.is_finite() {
            return Err(DpmppSdeError::NonFinite {
                step,
                stage,
                element,
            });
        }
    }
    Ok(())
}

fn checked_finite(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, DpmppSdeError> {
    if !value.is_finite() {
        return Err(DpmppSdeError::InvalidCoefficient {
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
) -> Result<f32, DpmppSdeError> {
    let value = checked_finite(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(DpmppSdeError::InvalidCoefficient {
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
) -> Result<f32, DpmppSdeError> {
    let value = checked_finite(step, coefficient, value)?;
    if value < 0.0 {
        return Err(DpmppSdeError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
