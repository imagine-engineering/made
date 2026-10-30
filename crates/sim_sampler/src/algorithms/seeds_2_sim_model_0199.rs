use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProfileError, SamplingProgress, SamplingSession, SamplingTrace,
    SchedulerError, SchedulerRegistry, exponential_integrator_phi_one,
    exponential_integrator_phi_two,
    generated_native_diffusion::validate_euler_noise_generation_device,
};
use sim_tensor::{
    CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const SEEDS_2_SAMPLER_ID: &str = "seeds_2";
pub const SEEDS_2_FEATURE_ID: &str = "COMFY-MODEL-0199";
pub const SEEDS_2_SOURCE_ORDINAL: u16 = 37;
pub const SEEDS_2_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: SEEDS_2_SAMPLER_ID,
    feature_id: SEEDS_2_FEATURE_ID,
    source_ordinal: SEEDS_2_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/seeds_2_comfy_model_0199",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Seeds2SolverType {
    #[default]
    Phi1,
    Phi2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Seeds2DenoiserStage {
    Primary,
    Intermediate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seeds2Options {
    pub eta: f32,
    pub noise_scale: f32,
    pub intermediate_step_ratio: f32,
    pub solver_type: Seeds2SolverType,
}

impl Default for Seeds2Options {
    fn default() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
            intermediate_step_ratio: 0.5,
            solver_type: Seeds2SolverType::Phi1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Seeds2Coefficients {
    intermediate_sigma: f32,
    predictor_latent_weight: f32,
    predictor_denoised_weight: f32,
    output_latent_weight: f32,
    output_alpha: f32,
    primary_weight: f32,
    intermediate_weight: f32,
    intermediate_noise_root: f32,
    first_segment_noise_weight: f32,
    second_segment_noise_weight: f32,
}

#[derive(Debug, Error)]
pub enum Seeds2Error {
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
    #[error("SEEDS-2 family requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("SEEDS-2 option {name} is invalid: {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("SEEDS-2 denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: Seeds2DenoiserStage,
        reason: String,
    },
    #[error("SEEDS-2 denoiser descriptor changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: Seeds2DenoiserStage,
    },
    #[error("SEEDS-2 coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("SEEDS-2 produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("native SEEDS-2 noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
    #[error("stochastic SEEDS-2 execution requires an authoritative compatibility RNG transaction")]
    MissingNoiseTransaction,
}

enum Seeds2NoiseMode {
    Deterministic,
    Compatibility {
        request: CompatibilityNoiseRequest,
        contract_id: &'static str,
    },
}

struct Seeds2FamilyOutput {
    trace: SamplingTrace,
    noise_checkpoints: Option<(RngCheckpoint, RngCheckpoint)>,
}

pub fn seeds_2_rng_profile(device: DeviceId) -> (RngSeedTransform, RngGenerationPlacement) {
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

pub fn validate_seeds_2_generation_device(device: DeviceId) -> Result<(), Seeds2Error> {
    if device == DeviceId::CPU {
        return Ok(());
    }
    validate_euler_noise_generation_device(device).map_err(|error| {
        Seeds2Error::DeviceUnavailable {
            device,
            reason: error.to_string(),
        }
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sample_seeds_2<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: Seeds2Options,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, Seeds2DenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), Seeds2Error>
where
    CallbackError: Display,
{
    let output = sample_seeds_2_family(
        backend,
        plan,
        profile,
        initial,
        sigmas,
        SEEDS_2_SAMPLER_ID,
        Seeds2NoiseMode::Compatibility {
            request: noise_request,
            contract_id: SEEDS_2_NOISE_CONTRACT_ID,
        },
        options,
        context,
        &mut denoiser,
        &mut callback,
    )?;
    let (noise_before, noise_after) = output
        .noise_checkpoints
        .ok_or(Seeds2Error::MissingNoiseTransaction)?;
    Ok((output.trace, noise_before, noise_after))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_seeds_2_deterministic_family<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    expected_sampler: &'static str,
    options: Seeds2Options,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, Seeds2DenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<SamplingTrace, Seeds2Error>
where
    CallbackError: Display,
{
    let output = sample_seeds_2_family(
        backend,
        plan,
        profile,
        initial,
        sigmas,
        expected_sampler,
        Seeds2NoiseMode::Deterministic,
        options,
        context,
        &mut denoiser,
        &mut callback,
    )?;
    Ok(output.trace)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_seeds_2_stochastic_family<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    expected_sampler: &'static str,
    noise_contract_id: &'static str,
    noise_request: CompatibilityNoiseRequest,
    options: Seeds2Options,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, Seeds2DenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), Seeds2Error>
where
    CallbackError: Display,
{
    let output = sample_seeds_2_family(
        backend,
        plan,
        profile,
        initial,
        sigmas,
        expected_sampler,
        Seeds2NoiseMode::Compatibility {
            request: noise_request,
            contract_id: noise_contract_id,
        },
        options,
        context,
        &mut denoiser,
        &mut callback,
    )?;
    let (noise_before, noise_after) = output
        .noise_checkpoints
        .ok_or(Seeds2Error::MissingNoiseTransaction)?;
    Ok((output.trace, noise_before, noise_after))
}

#[allow(clippy::too_many_arguments)]
fn sample_seeds_2_family<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    expected_sampler: &'static str,
    noise_mode: Seeds2NoiseMode,
    options: Seeds2Options,
    context: &ExecutionContext<'_>,
    denoiser: &mut impl FnMut(&Tensor, f32, usize, Seeds2DenoiserStage) -> Result<Tensor, String>,
    callback: &mut impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<Seeds2FamilyOutput, Seeds2Error>
where
    CallbackError: Display,
{
    context.check()?;
    validate_options(options)?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != expected_sampler {
        return Err(Seeds2Error::WrongSampler {
            expected: expected_sampler,
            actual: plan.sampler().as_str().to_owned(),
        });
    }

    let effective_noise_scale = profile.scale_sampler_noise(options.noise_scale)?;
    let seed = plan.seed();
    let device = initial.descriptor().device();
    if matches!(&noise_mode, Seeds2NoiseMode::Compatibility { .. }) {
        validate_seeds_2_generation_device(device)?;
    }
    let mut adjusted_sigmas = Vec::new();
    adjusted_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("adjusted sigma schedule"))?;
    adjusted_sigmas.extend_from_slice(sigmas);
    profile.adjust_first_sigma_for_snr(&mut adjusted_sigmas)?;
    let mut session = SamplingSession::new(plan, adjusted_sigmas.clone(), initial)?;
    let inject_noise = options.eta > 0.0 && effective_noise_scale > 0.0;
    let mut noise_transaction = match noise_mode {
        Seeds2NoiseMode::Deterministic if inject_noise => {
            return Err(Seeds2Error::MissingNoiseTransaction);
        }
        Seeds2NoiseMode::Deterministic => None,
        Seeds2NoiseMode::Compatibility {
            request,
            contract_id,
        } => {
            let (seed_transform, generation_placement) = seeds_2_rng_profile(device);
            Some(request.open_transaction(
                contract_id,
                i128::from(seed),
                seed_transform,
                generation_placement,
                None,
                context.cancellation,
            )?)
        }
    };
    let noise_before = noise_transaction
        .as_ref()
        .map(sim_tensor::CompatibilityRngTransaction::checkpoint);

    for (step, sigma_pair) in adjusted_sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = sigma_pair[0];
        let next_sigma = sigma_pair[1];
        let current = session.current().clone();
        let primary = call_denoiser(
            denoiser,
            &current,
            sigma,
            step,
            Seeds2DenoiserStage::Primary,
        )?;
        validate_descriptor(&current, &primary, step, Seeds2DenoiserStage::Primary)?;
        let observed = session.observe_step(
            &current,
            primary.clone(),
            context.cancellation,
            |progress, latent, denoised| callback(progress, latent, denoised),
        )?;

        let next = if next_sigma == 0.0 {
            validate_tensor(backend, &primary, step, "terminal denoiser", context)?;
            primary
        } else {
            let coefficients =
                coefficients(profile, sigma, next_sigma, options, inject_noise, step)?;
            let current_values = tensor_to_f32(backend, &current, context)?;
            let primary_values = tensor_to_f32(backend, &primary, context)?;
            validate_values(&current_values, step, "latent")?;
            validate_values(&primary_values, step, "primary denoiser")?;
            let count = current_values.len();
            let first_noise = if inject_noise {
                Some(
                    noise_transaction
                        .as_mut()
                        .ok_or(Seeds2Error::MissingNoiseTransaction)?
                        .draw_normal(count, context.cancellation)?,
                )
            } else {
                None
            };
            let predictor_noise_scale = if inject_noise {
                checked_nonnegative(
                    step,
                    "predictor noise scale",
                    coefficients.intermediate_noise_root
                        * coefficients.intermediate_sigma
                        * effective_noise_scale,
                )?
            } else {
                0.0
            };
            let mut predictor_values = backend.workspace_vec::<f32>(context, count)?;
            for (element, (current_value, primary_value)) in current_values
                .iter()
                .copied()
                .zip(primary_values.iter().copied())
                .enumerate()
            {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                let noise = first_noise
                    .as_ref()
                    .and_then(|values| values.get(element))
                    .copied()
                    .unwrap_or(0.0) as f32;
                let value = coefficients.predictor_latent_weight * current_value
                    + coefficients.predictor_denoised_weight * primary_value
                    + predictor_noise_scale * noise;
                checked_value(value, step, "predictor", element)?;
                predictor_values.try_push(value)?;
            }
            let predictor = tensor_from_f32(
                backend,
                current.descriptor().shape(),
                &predictor_values,
                context,
            )?;
            context.check()?;
            let intermediate = call_denoiser(
                denoiser,
                &predictor,
                coefficients.intermediate_sigma,
                step,
                Seeds2DenoiserStage::Intermediate,
            )?;
            validate_descriptor(
                &current,
                &intermediate,
                step,
                Seeds2DenoiserStage::Intermediate,
            )?;
            let intermediate_values = tensor_to_f32(backend, &intermediate, context)?;
            validate_values(&intermediate_values, step, "intermediate denoiser")?;
            let second_noise = if inject_noise {
                Some(
                    noise_transaction
                        .as_mut()
                        .ok_or(Seeds2Error::MissingNoiseTransaction)?
                        .draw_normal(count, context.cancellation)?,
                )
            } else {
                None
            };
            let output_noise_scale = if inject_noise {
                checked_nonnegative(
                    step,
                    "output noise scale",
                    next_sigma * effective_noise_scale,
                )?
            } else {
                0.0
            };
            let mut output_values = backend.workspace_vec::<f32>(context, count)?;
            for (element, ((current_value, primary_value), intermediate_value)) in current_values
                .iter()
                .copied()
                .zip(primary_values.iter().copied())
                .zip(intermediate_values.iter().copied())
                .enumerate()
            {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                let first = first_noise
                    .as_ref()
                    .and_then(|values| values.get(element))
                    .copied()
                    .unwrap_or(0.0) as f32;
                let second = second_noise
                    .as_ref()
                    .and_then(|values| values.get(element))
                    .copied()
                    .unwrap_or(0.0) as f32;
                let sde_noise = coefficients.intermediate_noise_root
                    * first
                    * coefficients.first_segment_noise_weight
                    + coefficients.second_segment_noise_weight * second;
                let value = coefficients.output_latent_weight * current_value
                    - coefficients.output_alpha
                        * (coefficients.primary_weight * primary_value
                            + coefficients.intermediate_weight * intermediate_value)
                    + sde_noise * output_noise_scale;
                checked_value(value, step, "output", element)?;
                output_values.try_push(value)?;
            }
            tensor_from_f32(
                backend,
                current.descriptor().shape(),
                &output_values,
                context,
            )?
        };
        observed.commit(next, context.cancellation)?;
    }

    let trace = session.finish()?;
    let noise_after = noise_transaction.map(sim_tensor::CompatibilityRngTransaction::commit);
    let noise_checkpoints = match (noise_before, noise_after) {
        (Some(before), Some(after)) => Some((before, after)),
        (None, None) => None,
        _ => return Err(Seeds2Error::MissingNoiseTransaction),
    };
    Ok(Seeds2FamilyOutput {
        trace,
        noise_checkpoints,
    })
}

fn coefficients(
    profile: &impl SamplingProfile,
    sigma: f32,
    next_sigma: f32,
    options: Seeds2Options,
    inject_noise: bool,
    step: usize,
) -> Result<Seeds2Coefficients, Seeds2Error> {
    let lambda_source = checked_finite(step, "source half-log-SNR", profile.half_log_snr(sigma)?)?;
    let lambda_target = checked_finite(
        step,
        "target half-log-SNR",
        profile.half_log_snr(next_sigma)?,
    )?;
    let step_size = checked_positive(
        step,
        "half-log-SNR step size",
        lambda_target - lambda_source,
    )?;
    let eta_step_size = checked_finite(step, "eta step size", step_size * (options.eta + 1.0))?;
    let intermediate_lambda = checked_finite(
        step,
        "intermediate half-log-SNR",
        lambda_source + (lambda_target - lambda_source) * options.intermediate_step_ratio,
    )?;
    let intermediate_sigma = checked_positive(
        step,
        "intermediate sigma",
        profile.sigma_from_half_log_snr(intermediate_lambda)?,
    )?;
    let intermediate_alpha = checked_positive(
        step,
        "intermediate alpha",
        intermediate_sigma * intermediate_lambda.exp(),
    )?;
    let output_alpha = checked_positive(step, "target alpha", next_sigma * lambda_target.exp())?;
    let intermediate_phi_argument = -options.intermediate_step_ratio * eta_step_size;
    let intermediate_phi_one = checked_finite(
        step,
        "intermediate phi one",
        exponential_integrator_phi_one(intermediate_phi_argument),
    )?;
    let phi_argument = -eta_step_size;
    let phi_one = checked_finite(
        step,
        "phi one",
        exponential_integrator_phi_one(phi_argument),
    )?;
    let (primary_weight, intermediate_weight) = match options.solver_type {
        Seeds2SolverType::Phi1 => {
            let factor = checked_finite(
                step,
                "phi one interpolation factor",
                1.0 / (2.0 * options.intermediate_step_ratio),
            )?;
            (phi_one * (1.0 - factor), phi_one * factor)
        }
        Seeds2SolverType::Phi2 => {
            let phi_two = checked_finite(
                step,
                "phi two",
                exponential_integrator_phi_two(phi_argument),
            )?;
            let second = checked_finite(
                step,
                "phi two intermediate weight",
                phi_two / options.intermediate_step_ratio,
            )?;
            (phi_one - second, second)
        }
    };
    let segment_factor = (options.intermediate_step_ratio - 1.0) * step_size * options.eta;
    let (intermediate_noise_root, first_segment_noise_weight, second_segment_noise_weight) =
        if inject_noise {
            (
                checked_nonnegative(
                    step,
                    "intermediate noise root",
                    (-(-2.0 * options.intermediate_step_ratio * step_size * options.eta).exp_m1())
                        .sqrt(),
                )?,
                checked_nonnegative(step, "first segment noise weight", segment_factor.exp())?,
                checked_nonnegative(
                    step,
                    "second segment noise weight",
                    (-(2.0 * segment_factor).exp_m1()).sqrt(),
                )?,
            )
        } else {
            (0.0, 0.0, 0.0)
        };
    Ok(Seeds2Coefficients {
        intermediate_sigma,
        predictor_latent_weight: checked_positive(
            step,
            "predictor latent weight",
            intermediate_sigma / sigma
                * (-options.intermediate_step_ratio * step_size * options.eta).exp(),
        )?,
        predictor_denoised_weight: checked_finite(
            step,
            "predictor denoised weight",
            -intermediate_alpha * intermediate_phi_one,
        )?,
        output_latent_weight: checked_positive(
            step,
            "output latent weight",
            next_sigma / sigma * (-step_size * options.eta).exp(),
        )?,
        output_alpha,
        primary_weight: checked_finite(step, "primary weight", primary_weight)?,
        intermediate_weight: checked_finite(step, "intermediate weight", intermediate_weight)?,
        intermediate_noise_root,
        first_segment_noise_weight,
        second_segment_noise_weight,
    })
}

fn validate_options(options: Seeds2Options) -> Result<(), Seeds2Error> {
    for (name, value) in [
        ("eta", options.eta),
        ("noise scale", options.noise_scale),
        ("intermediate step ratio", options.intermediate_step_ratio),
    ] {
        if !value.is_finite() {
            return Err(Seeds2Error::InvalidOption { name, value });
        }
    }
    if options.intermediate_step_ratio == 0.0 {
        return Err(Seeds2Error::InvalidOption {
            name: "intermediate step ratio",
            value: options.intermediate_step_ratio,
        });
    }
    Ok(())
}

fn call_denoiser(
    denoiser: &mut impl FnMut(&Tensor, f32, usize, Seeds2DenoiserStage) -> Result<Tensor, String>,
    latent: &Tensor,
    sigma: f32,
    step: usize,
    stage: Seeds2DenoiserStage,
) -> Result<Tensor, Seeds2Error> {
    denoiser(latent, sigma, step, stage).map_err(|reason| Seeds2Error::Denoiser {
        step,
        stage,
        reason,
    })
}

fn validate_descriptor(
    current: &Tensor,
    denoised: &Tensor,
    step: usize,
    stage: Seeds2DenoiserStage,
) -> Result<(), Seeds2Error> {
    if current.descriptor() != denoised.descriptor() {
        return Err(Seeds2Error::DenoiserContract { step, stage });
    }
    Ok(())
}

fn validate_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), Seeds2Error> {
    validate_values(&tensor_to_f32(backend, tensor, context)?, step, stage)
}

fn validate_values(values: &[f32], step: usize, stage: &'static str) -> Result<(), Seeds2Error> {
    for (element, value) in values.iter().copied().enumerate() {
        checked_value(value, step, stage, element)?;
    }
    Ok(())
}

fn checked_value(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<f32, Seeds2Error> {
    if !value.is_finite() {
        return Err(Seeds2Error::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(value)
}

fn checked_finite(step: usize, coefficient: &'static str, value: f32) -> Result<f32, Seeds2Error> {
    if !value.is_finite() {
        return Err(Seeds2Error::InvalidCoefficient {
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
) -> Result<f32, Seeds2Error> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Seeds2Error::InvalidCoefficient {
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
) -> Result<f32, Seeds2Error> {
    if !value.is_finite() || value < 0.0 {
        return Err(Seeds2Error::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
