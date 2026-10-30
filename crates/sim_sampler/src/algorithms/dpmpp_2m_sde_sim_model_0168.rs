use crate::{
    BrownianNoiseIntervalAddress, CompatibilityNoiseRequest, NoiseError, SamplerDefinition,
    SamplerRegistry, SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError,
    SamplingProgress, SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    BrownianTree, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint, RngCompatibilityError,
    RngError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DPMPP_2M_SDE_SAMPLER_ID: &str = "dpmpp_2m_sde";
pub const DPMPP_2M_SDE_FEATURE_ID: &str = "COMFY-MODEL-0168";
pub const DPMPP_2M_SDE_SOURCE_ORDINAL: u16 = 19;
pub const DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID: &str = "COMFY-RNG-DED616CC3432";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPMPP_2M_SDE_SAMPLER_ID,
    feature_id: DPMPP_2M_SDE_FEATURE_ID,
    source_ordinal: DPMPP_2M_SDE_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpmpp_2m_sde_comfy_model_0168",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Dpmpp2mSdeSolverType {
    Heun,
    #[default]
    Midpoint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dpmpp2mSdeOptions {
    eta: f32,
    noise_scale: f32,
    solver_type: Dpmpp2mSdeSolverType,
}

impl Dpmpp2mSdeOptions {
    pub fn new(eta: f32, noise_scale: f32) -> Result<Self, Dpmpp2mSdeSamplerError> {
        Self::new_with_solver_type(eta, noise_scale, Dpmpp2mSdeSolverType::Midpoint)
    }

    pub fn new_with_solver_type(
        eta: f32,
        noise_scale: f32,
        solver_type: Dpmpp2mSdeSolverType,
    ) -> Result<Self, Dpmpp2mSdeSamplerError> {
        let options = Self {
            eta,
            noise_scale,
            solver_type,
        };
        options.validate()?;
        Ok(options)
    }

    pub fn source_defaults() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
            solver_type: Dpmpp2mSdeSolverType::Midpoint,
        }
    }

    pub fn eta(self) -> f32 {
        self.eta
    }

    pub fn noise_scale(self) -> f32 {
        self.noise_scale
    }

    pub fn solver_type(self) -> Dpmpp2mSdeSolverType {
        self.solver_type
    }

    fn validate(self) -> Result<(), Dpmpp2mSdeSamplerError> {
        for (name, value) in [("eta", self.eta), ("s_noise", self.noise_scale)] {
            if !value.is_finite() {
                return Err(Dpmpp2mSdeSamplerError::InvalidOption { name, value });
            }
        }
        Ok(())
    }
}

impl Default for Dpmpp2mSdeOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Debug, Error)]
pub enum Dpmpp2mSdeSamplerError {
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
    #[error("DPM-Solver++(2M) SDE requires sampler identity `dpmpp_2m_sde`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM-Solver++(2M) SDE option {name} is invalid: {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("DPM-Solver++(2M) SDE denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("DPM-Solver++(2M) SDE denoiser descriptor changed at step {step}")]
    DenoiserContract { step: usize },
    #[error(
        "DPM-Solver++(2M) SDE produced a non-finite {stage} value at step {step}, element {element}"
    )]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM-Solver++(2M) SDE coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("DPM-Solver++(2M) SDE arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("DPM-Solver++(2M) SDE allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("DPM-Solver++(2M) SDE has no Brownian tree for stochastic step {step}")]
    MissingBrownianTree { step: usize },
    #[error("DPM-Solver++(2M) SDE generation placement outputs {actual:?}, expected {expected:?}")]
    PlacementOutputMismatch {
        expected: DeviceId,
        actual: DeviceId,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpmpp_2m_sde<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    options: Dpmpp2mSdeOptions,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), Dpmpp2mSdeSamplerError>
where
    CallbackError: Display,
{
    let output_device = initial.descriptor().device();
    sample_dpmpp_2m_sde_with_generation_placement(
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
pub(crate) fn sample_dpmpp_2m_sde_with_generation_placement<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    options: Dpmpp2mSdeOptions,
    noise_request: CompatibilityNoiseRequest,
    generation_placement: RngGenerationPlacement,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), Dpmpp2mSdeSamplerError>
where
    CallbackError: Display,
{
    context.check()?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        profile.identity(),
    )?;
    if plan.sampler().as_str() != DPMPP_2M_SDE_SAMPLER_ID {
        return Err(Dpmpp2mSdeSamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    if sigmas.len() <= 1 {
        return short_trace(sigmas, initial);
    }
    options.validate()?;
    validate_dpmpp_2m_sde_generation_placement(
        initial.descriptor().device(),
        generation_placement,
    )?;

    let seed = plan.seed();
    let step_count = usize::try_from(plan.steps())
        .map_err(|_| Dpmpp2mSdeSamplerError::Overflow("sampling step count"))?;
    let (brownian_minimum, brownian_maximum) = brownian_bounds(sigmas)?;
    let mut adjusted_sigmas = Vec::new();
    adjusted_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| Dpmpp2mSdeSamplerError::OutOfMemory("adjusted sigma schedule"))?;
    adjusted_sigmas.extend_from_slice(sigmas);
    profile.adjust_first_sigma_for_snr(&mut adjusted_sigmas)?;
    let effective_noise_scale = profile.scale_sampler_noise(options.noise_scale)?;
    let draw_noise = options.eta > 0.0
        && effective_noise_scale > 0.0
        && adjusted_sigmas.windows(2).any(|pair| {
            pair.first().is_some_and(|sigma| *sigma > 0.0)
                && pair.get(1).is_some_and(|sigma| *sigma > 0.0)
        });
    let mut session = SamplingSession::new(plan, adjusted_sigmas.clone(), initial)?;
    let element_count = usize::try_from(session.current().descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let mut noise_transaction = noise_request.open_transaction(
        DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::TorchSigned64,
        generation_placement,
        None,
        context.cancellation,
    )?;
    let noise_before = noise_transaction.checkpoint();
    let mut initial_brownian = Vec::new();
    initial_brownian
        .try_reserve_exact(element_count)
        .map_err(|_| Dpmpp2mSdeSamplerError::OutOfMemory("Brownian initial value"))?;
    initial_brownian.resize(element_count, 0.0);
    let mut brownian_tree = noise_transaction.brownian_tree(
        f64::from(brownian_minimum),
        initial_brownian,
        f64::from(brownian_maximum),
        context.cancellation,
    )?;
    let mut previous_denoised: Option<Tensor> = None;
    let mut previous_step_size: Option<f32> = None;

    for step in 0..step_count {
        context.check()?;
        let sigma = *adjusted_sigmas
            .get(step)
            .ok_or(Dpmpp2mSdeSamplerError::Overflow("current sigma lookup"))?;
        let next_index = step
            .checked_add(1)
            .ok_or(Dpmpp2mSdeSamplerError::Overflow("next sigma index"))?;
        let next_sigma = *adjusted_sigmas
            .get(next_index)
            .ok_or(Dpmpp2mSdeSamplerError::Overflow("next sigma lookup"))?;
        let current = session.current().clone();
        let denoised = denoiser(&current, sigma, step)
            .map_err(|reason| Dpmpp2mSdeSamplerError::Denoiser { step, reason })?;
        validate_denoiser(&current, &denoised, step)?;
        let observed = session.observe_step(
            &current,
            denoised.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;

        let next = if next_sigma == 0.0 {
            validate_finite(backend, &denoised, step, "terminal denoiser", context)?;
            denoised.clone()
        } else {
            let lambda_source =
                checked_coefficient(step, "source half-log-SNR", profile.half_log_snr(sigma)?)?;
            let lambda_target = checked_coefficient(
                step,
                "target half-log-SNR",
                profile.half_log_snr(next_sigma)?,
            )?;
            let step_size = checked_positive(step, "step size", lambda_target - lambda_source)?;
            let eta_step_size =
                checked_coefficient(step, "eta step size", step_size * (options.eta + 1.0))?;
            let alpha_target =
                checked_positive(step, "target alpha", next_sigma * lambda_target.exp())?;
            let latent_weight = checked_coefficient(
                step,
                "latent weight",
                next_sigma / sigma * (-step_size * options.eta).exp(),
            )?;
            let denoised_weight = checked_coefficient(
                step,
                "denoised weight",
                alpha_target * -(-eta_step_size).exp_m1(),
            )?;
            let correction_weight = match (previous_denoised.as_ref(), previous_step_size) {
                (Some(_), Some(previous_step_size)) => {
                    let ratio =
                        checked_positive(step, "step ratio", previous_step_size / step_size)?;
                    match options.solver_type {
                        Dpmpp2mSdeSolverType::Heun => checked_coefficient(
                            step,
                            "Heun correction weight",
                            alpha_target * (-(-eta_step_size).exp_m1() / -eta_step_size + 1.0)
                                / ratio,
                        )?,
                        Dpmpp2mSdeSolverType::Midpoint => checked_coefficient(
                            step,
                            "midpoint correction weight",
                            0.5 * denoised_weight / ratio,
                        )?,
                    }
                }
                (None, None) => 0.0,
                _ => {
                    return Err(Dpmpp2mSdeSamplerError::InvalidCoefficient {
                        step,
                        coefficient: "previous step state",
                        value: f32::NAN,
                    });
                }
            };
            let stochastic_scale = if draw_noise {
                checked_coefficient(
                    step,
                    "stochastic scale",
                    next_sigma
                        * (-(-2.0 * step_size * options.eta).exp_m1()).sqrt()
                        * effective_noise_scale,
                )?
            } else {
                0.0
            };
            let brownian_noise = if draw_noise {
                Some(normalized_brownian_increment(
                    &mut brownian_tree,
                    sigma,
                    next_sigma,
                    step,
                    context,
                )?)
            } else {
                None
            };
            let current_values = tensor_to_f32(backend, &current, context)?;
            let denoised_values = tensor_to_f32(backend, &denoised, context)?;
            let previous_values = previous_denoised
                .as_ref()
                .map(|previous| tensor_to_f32(backend, previous, context))
                .transpose()?;
            let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
            for (element, (current_value, denoised_value)) in current_values
                .iter()
                .copied()
                .zip(denoised_values.iter().copied())
                .enumerate()
            {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                let mut next_value =
                    latent_weight * current_value + denoised_weight * denoised_value;
                if let Some(previous_values) = previous_values.as_ref() {
                    let previous_value = previous_values.get(element).copied().ok_or(
                        Dpmpp2mSdeSamplerError::Overflow("previous denoiser element lookup"),
                    )?;
                    next_value += correction_weight * (denoised_value - previous_value);
                }
                if let Some(noise) = brownian_noise.as_ref() {
                    let noise_value =
                        noise
                            .get(element)
                            .copied()
                            .ok_or(Dpmpp2mSdeSamplerError::Overflow(
                                "Brownian noise element lookup",
                            ))?;
                    next_value += noise_value * stochastic_scale;
                }
                if !next_value.is_finite() {
                    return Err(Dpmpp2mSdeSamplerError::NonFinite {
                        step,
                        stage: "next latent",
                        element,
                    });
                }
                next_values.try_push(next_value)?;
            }
            previous_step_size = Some(step_size);
            tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)?
        };
        observed.commit(next, context.cancellation)?;
        previous_denoised = Some(denoised);
    }

    Ok((
        session.finish()?,
        Some((noise_before, noise_transaction.commit())),
    ))
}

pub(crate) fn validate_dpmpp_2m_sde_generation_placement(
    output_device: DeviceId,
    placement: RngGenerationPlacement,
) -> Result<(), Dpmpp2mSdeSamplerError> {
    if placement.output_device() != output_device {
        return Err(Dpmpp2mSdeSamplerError::PlacementOutputMismatch {
            expected: output_device,
            actual: placement.output_device(),
        });
    }
    Ok(())
}

fn short_trace(
    sigmas: &[f32],
    initial: Tensor,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), Dpmpp2mSdeSamplerError> {
    let mut traced_sigmas = Vec::new();
    traced_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| Dpmpp2mSdeSamplerError::OutOfMemory("short sigma trace"))?;
    traced_sigmas.extend_from_slice(sigmas);
    let mut latents = Vec::new();
    latents
        .try_reserve_exact(1)
        .map_err(|_| Dpmpp2mSdeSamplerError::OutOfMemory("short latent trace"))?;
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

fn brownian_bounds(sigmas: &[f32]) -> Result<(f32, f32), Dpmpp2mSdeSamplerError> {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for sigma in sigmas.iter().copied().filter(|sigma| *sigma > 0.0) {
        if !sigma.is_finite() {
            return Err(Dpmpp2mSdeSamplerError::InvalidCoefficient {
                step: 0,
                coefficient: "Brownian sigma bound",
                value: sigma,
            });
        }
        minimum = minimum.min(sigma);
        maximum = maximum.max(sigma);
    }
    if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
        return Err(Dpmpp2mSdeSamplerError::InvalidCoefficient {
            step: 0,
            coefficient: "Brownian sigma bounds",
            value: minimum,
        });
    }
    Ok((minimum, maximum))
}

fn normalized_brownian_increment(
    tree: &mut BrownianTree,
    sigma: f32,
    next_sigma: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, Dpmpp2mSdeSamplerError> {
    let step = u32::try_from(step)
        .map_err(|_| Dpmpp2mSdeSamplerError::Overflow("Brownian interval step"))?;
    let address = BrownianNoiseIntervalAddress::new(sigma, next_sigma, step)?;
    let (lower, upper) = address.canonical_interval();
    let increment = tree.increment(f64::from(lower), f64::from(upper), context.cancellation)?;
    let normalization = f64::from(upper - lower).sqrt();
    let sign = if address.reverse { -1.0 } else { 1.0 };
    let mut normalized = Vec::new();
    normalized
        .try_reserve_exact(increment.len())
        .map_err(|_| Dpmpp2mSdeSamplerError::OutOfMemory("normalized Brownian increment"))?;
    for (element, increment) in increment.into_iter().enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let value = (increment * sign / normalization) as f32;
        if !value.is_finite() {
            return Err(Dpmpp2mSdeSamplerError::NonFinite {
                step: usize::try_from(step)
                    .map_err(|_| Dpmpp2mSdeSamplerError::Overflow("Brownian step index"))?,
                stage: "Brownian noise",
                element,
            });
        }
        normalized.push(value);
    }
    Ok(normalized)
}

fn validate_denoiser(
    input: &Tensor,
    denoised: &Tensor,
    step: usize,
) -> Result<(), Dpmpp2mSdeSamplerError> {
    if input.descriptor() != denoised.descriptor() {
        return Err(Dpmpp2mSdeSamplerError::DenoiserContract { step });
    }
    Ok(())
}

fn validate_finite(
    backend: &CpuBackend,
    tensor: &Tensor,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), Dpmpp2mSdeSamplerError> {
    for (element, value) in tensor_to_f32(backend, tensor, context)?.iter().enumerate() {
        if !value.is_finite() {
            return Err(Dpmpp2mSdeSamplerError::NonFinite {
                step,
                stage,
                element,
            });
        }
    }
    Ok(())
}

fn checked_coefficient(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, Dpmpp2mSdeSamplerError> {
    if !value.is_finite() {
        return Err(Dpmpp2mSdeSamplerError::InvalidCoefficient {
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
) -> Result<f32, Dpmpp2mSdeSamplerError> {
    let value = checked_coefficient(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(Dpmpp2mSdeSamplerError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
