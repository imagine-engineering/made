use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProfileError, SamplingProgress, SamplingSession, SamplingTrace,
    SchedulerError, SchedulerRegistry, exponential_integrator_phi_one,
    exponential_integrator_phi_two,
    generated_native_diffusion::validate_euler_noise_generation_device,
};
use sim_tensor::{
    CpuBackend, DeviceId, ExecutionContext, RngCheckpoint, RngCompatibilityError,
    RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const SEEDS_3_SAMPLER_ID: &str = "seeds_3";
pub const SEEDS_3_FEATURE_ID: &str = "COMFY-MODEL-0200";
pub const SEEDS_3_SOURCE_ORDINAL: u16 = 38;
pub const SEEDS_3_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: SEEDS_3_SAMPLER_ID,
    feature_id: SEEDS_3_FEATURE_ID,
    source_ordinal: SEEDS_3_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/seeds_3_comfy_model_0200",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Seeds3DenoiserStage {
    Primary,
    StageTwo,
    StageThree,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seeds3Options {
    eta: f32,
    noise_scale: f32,
    first_stage: f32,
    second_stage: f32,
}

impl Seeds3Options {
    pub fn new(
        eta: f32,
        noise_scale: f32,
        first_stage: f32,
        second_stage: f32,
    ) -> Result<Self, Seeds3Error> {
        for (name, value) in [
            ("eta", eta),
            ("s_noise", noise_scale),
            ("r_1", first_stage),
            ("r_2", second_stage),
        ] {
            if !value.is_finite() {
                return Err(Seeds3Error::InvalidOption { name, value });
            }
        }
        if first_stage <= 0.0 || first_stage > second_stage || second_stage > 1.0 {
            return Err(Seeds3Error::InvalidStageFractions {
                first: first_stage,
                second: second_stage,
            });
        }
        Ok(Self {
            eta,
            noise_scale,
            first_stage,
            second_stage,
        })
    }

    pub fn source_defaults() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
            first_stage: 1.0 / 3.0,
            second_stage: 2.0 / 3.0,
        }
    }

    pub const fn eta(self) -> f32 {
        self.eta
    }

    pub const fn noise_scale(self) -> f32 {
        self.noise_scale
    }

    pub const fn first_stage(self) -> f32 {
        self.first_stage
    }

    pub const fn second_stage(self) -> f32 {
        self.second_stage
    }
}

impl Default for Seeds3Options {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Debug, Error)]
pub enum Seeds3Error {
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
    #[error("SEEDS-3 requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("SEEDS-3 option {name} must be finite, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("SEEDS-3 stage fractions require 0 < r_1 <= r_2 <= 1, got {first}, {second}")]
    InvalidStageFractions { first: f32, second: f32 },
    #[error("SEEDS-3 denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: Seeds3DenoiserStage,
        reason: String,
    },
    #[error("SEEDS-3 denoiser descriptor changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: Seeds3DenoiserStage,
    },
    #[error("SEEDS-3 coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("SEEDS-3 produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("native SEEDS-3 noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
}

#[derive(Clone, Copy, Debug)]
struct Seeds3Coefficients {
    first_sigma: f32,
    second_sigma: f32,
    first_alpha: f32,
    second_alpha: f32,
    target_alpha: f32,
    first_latent_weight: f32,
    first_denoised_weight: f32,
    second_latent_weight: f32,
    second_primary_weight: f32,
    second_stage_weight: f32,
    output_latent_weight: f32,
    output_primary_weight: f32,
    output_stage_weight: f32,
    first_noise_root: f32,
    second_segment_old: f32,
    second_segment_new: f32,
    output_segment_old: f32,
    output_segment_new: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn sample_seeds_3<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: Seeds3Options,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, Seeds3DenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, (RngCheckpoint, RngCheckpoint)), Seeds3Error>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    if plan.sampler().as_str() != SEEDS_3_SAMPLER_ID {
        return Err(Seeds3Error::WrongSampler {
            expected: SEEDS_3_SAMPLER_ID,
            actual: plan.sampler().as_str().to_owned(),
        });
    }
    let effective_noise_scale = profile.scale_sampler_noise(options.noise_scale())?;
    let device = initial.descriptor().device();
    validate_euler_noise_generation_device(device).map_err(|error| {
        Seeds3Error::DeviceUnavailable {
            device,
            reason: error.to_string(),
        }
    })?;
    let mut adjusted_sigmas = Vec::new();
    adjusted_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("SEEDS-3 adjusted sigma schedule"))?;
    adjusted_sigmas.extend_from_slice(sigmas);
    profile.adjust_first_sigma_for_snr(&mut adjusted_sigmas)?;
    let seed = plan.seed();
    let mut session = SamplingSession::new(plan, adjusted_sigmas.clone(), initial)?;
    let (seed_transform, placement) = noise_profile(device);
    let mut noise_transaction = noise_request.open_transaction(
        SEEDS_3_NOISE_CONTRACT_ID,
        i128::from(seed),
        seed_transform,
        placement,
        None,
        context.cancellation,
    )?;
    let before = noise_transaction.checkpoint();
    let inject_noise = options.eta() > 0.0 && effective_noise_scale > 0.0;

    for (step, sigma_pair) in adjusted_sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = checked_positive(step, "sigma", sigma_pair[0])?;
        let next_sigma = checked_nonnegative(step, "next sigma", sigma_pair[1])?;
        let current = session.current().clone();
        let primary = call_denoiser(
            &mut denoiser,
            &current,
            sigma,
            step,
            Seeds3DenoiserStage::Primary,
        )?;
        validate_descriptor(&current, &primary, step, Seeds3DenoiserStage::Primary)?;
        let observed = session.observe_step(
            &current,
            primary.clone(),
            context.cancellation,
            |progress, current, denoised| callback(progress, current, denoised),
        )?;

        let next = if next_sigma == 0.0 {
            validate_values(
                &tensor_to_f32(backend, &primary, context)?,
                step,
                "terminal denoiser",
            )?;
            primary
        } else {
            let coefficients =
                coefficients(profile, sigma, next_sigma, options, inject_noise, step)?;
            let current_values = tensor_to_f32(backend, &current, context)?;
            let primary_values = tensor_to_f32(backend, &primary, context)?;
            validate_values(&current_values, step, "latent")?;
            validate_values(&primary_values, step, "primary denoiser")?;
            let count = current_values.len();
            let first_noise = draw_noise(inject_noise, count, &mut noise_transaction, context)?;
            let mut accumulated_noise = backend.workspace_vec::<f32>(context, count)?;
            let mut stage_two_input_values = backend.workspace_vec::<f32>(context, count)?;
            for (element, (current_value, primary_value)) in current_values
                .iter()
                .copied()
                .zip(primary_values.iter().copied())
                .enumerate()
            {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                let noise = noise_value(first_noise.as_deref(), element);
                let accumulated = coefficients.first_noise_root * noise;
                let value = coefficients.first_latent_weight * current_value
                    - coefficients.first_alpha * coefficients.first_denoised_weight * primary_value
                    + accumulated * coefficients.first_sigma * effective_noise_scale;
                checked_element(value, step, "stage-two input", element)?;
                accumulated_noise.try_push(accumulated)?;
                stage_two_input_values.try_push(value)?;
            }
            let stage_two_input = tensor_from_f32(
                backend,
                current.descriptor().shape(),
                &stage_two_input_values,
                context,
            )?;
            let stage_two = call_denoiser(
                &mut denoiser,
                &stage_two_input,
                coefficients.first_sigma,
                step,
                Seeds3DenoiserStage::StageTwo,
            )?;
            validate_descriptor(&current, &stage_two, step, Seeds3DenoiserStage::StageTwo)?;
            let stage_two_values = tensor_to_f32(backend, &stage_two, context)?;
            validate_values(&stage_two_values, step, "stage-two denoiser")?;
            let second_noise = draw_noise(inject_noise, count, &mut noise_transaction, context)?;
            let mut stage_three_input_values = backend.workspace_vec::<f32>(context, count)?;
            for (element, (((current_value, primary_value), stage_two_value), accumulated)) in
                current_values
                    .iter()
                    .copied()
                    .zip(primary_values.iter().copied())
                    .zip(stage_two_values.iter().copied())
                    .zip(accumulated_noise.iter_mut())
                    .enumerate()
            {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                *accumulated = *accumulated * coefficients.second_segment_old
                    + coefficients.second_segment_new
                        * noise_value(second_noise.as_deref(), element);
                let value = coefficients.second_latent_weight * current_value
                    - coefficients.second_alpha
                        * (coefficients.second_primary_weight * primary_value
                            + coefficients.second_stage_weight * stage_two_value)
                    + *accumulated * coefficients.second_sigma * effective_noise_scale;
                checked_element(value, step, "stage-three input", element)?;
                stage_three_input_values.try_push(value)?;
            }
            let stage_three_input = tensor_from_f32(
                backend,
                current.descriptor().shape(),
                &stage_three_input_values,
                context,
            )?;
            let stage_three = call_denoiser(
                &mut denoiser,
                &stage_three_input,
                coefficients.second_sigma,
                step,
                Seeds3DenoiserStage::StageThree,
            )?;
            validate_descriptor(
                &current,
                &stage_three,
                step,
                Seeds3DenoiserStage::StageThree,
            )?;
            let stage_three_values = tensor_to_f32(backend, &stage_three, context)?;
            validate_values(&stage_three_values, step, "stage-three denoiser")?;
            let third_noise = draw_noise(inject_noise, count, &mut noise_transaction, context)?;
            let mut output_values = backend.workspace_vec::<f32>(context, count)?;
            for (element, (((current_value, primary_value), stage_three_value), accumulated)) in
                current_values
                    .iter()
                    .copied()
                    .zip(primary_values.iter().copied())
                    .zip(stage_three_values.iter().copied())
                    .zip(accumulated_noise.iter_mut())
                    .enumerate()
            {
                if element.is_multiple_of(256) {
                    context.check()?;
                }
                *accumulated = *accumulated * coefficients.output_segment_old
                    + coefficients.output_segment_new
                        * noise_value(third_noise.as_deref(), element);
                let value = coefficients.output_latent_weight * current_value
                    - coefficients.target_alpha
                        * (coefficients.output_primary_weight * primary_value
                            + coefficients.output_stage_weight * stage_three_value)
                    + *accumulated * next_sigma * effective_noise_scale;
                checked_element(value, step, "output", element)?;
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
    Ok((session.finish()?, (before, noise_transaction.commit())))
}

fn coefficients(
    profile: &impl SamplingProfile,
    sigma: f32,
    next_sigma: f32,
    options: Seeds3Options,
    inject_noise: bool,
    step: usize,
) -> Result<Seeds3Coefficients, Seeds3Error> {
    let lambda_source = checked_finite(step, "source half-log-SNR", profile.half_log_snr(sigma)?)?;
    let lambda_target = checked_finite(
        step,
        "target half-log-SNR",
        profile.half_log_snr(next_sigma)?,
    )?;
    let step_size = checked_positive(step, "half-log-SNR step", lambda_target - lambda_source)?;
    let eta_step = checked_nonzero(step, "eta step", step_size * (options.eta() + 1.0))?;
    let first_lambda = lambda_source + step_size * options.first_stage();
    let second_lambda = lambda_source + step_size * options.second_stage();
    let first_sigma = checked_positive(
        step,
        "first intermediate sigma",
        profile.sigma_from_half_log_snr(first_lambda)?,
    )?;
    let second_sigma = checked_positive(
        step,
        "second intermediate sigma",
        profile.sigma_from_half_log_snr(second_lambda)?,
    )?;
    let first_alpha = checked_positive(step, "first alpha", first_sigma * first_lambda.exp())?;
    let second_alpha = checked_positive(step, "second alpha", second_sigma * second_lambda.exp())?;
    let target_alpha = checked_positive(step, "target alpha", next_sigma * lambda_target.exp())?;
    let first_phi = exponential_integrator_phi_one(-options.first_stage() * eta_step);
    let second_phi_one = exponential_integrator_phi_one(-options.second_stage() * eta_step);
    let second_stage_weight = options.second_stage() / options.first_stage()
        * checked_phi_two(-options.second_stage() * eta_step, step)?;
    let output_stage_weight = checked_phi_two(-eta_step, step)? / options.second_stage();
    let first_factor = (options.first_stage() - options.second_stage()) * step_size * options.eta();
    let output_factor = (options.second_stage() - 1.0) * step_size * options.eta();
    let (
        first_noise_root,
        second_segment_old,
        second_segment_new,
        output_segment_old,
        output_segment_new,
    ) = if inject_noise {
        (
            noise_root(
                -2.0 * options.first_stage() * step_size * options.eta(),
                step,
                "first noise root",
            )?,
            checked_nonnegative(step, "second old-noise weight", first_factor.exp())?,
            noise_root(2.0 * first_factor, step, "second new-noise root")?,
            checked_nonnegative(step, "output old-noise weight", output_factor.exp())?,
            noise_root(2.0 * output_factor, step, "output new-noise root")?,
        )
    } else {
        (0.0, 1.0, 0.0, 1.0, 0.0)
    };
    Ok(Seeds3Coefficients {
        first_sigma,
        second_sigma,
        first_alpha,
        second_alpha,
        target_alpha,
        first_latent_weight: checked_positive(
            step,
            "first latent weight",
            first_sigma / sigma * (-options.first_stage() * step_size * options.eta()).exp(),
        )?,
        first_denoised_weight: checked_finite(step, "first denoised weight", first_phi)?,
        second_latent_weight: checked_positive(
            step,
            "second latent weight",
            second_sigma / sigma * (-options.second_stage() * step_size * options.eta()).exp(),
        )?,
        second_primary_weight: checked_finite(
            step,
            "second primary weight",
            second_phi_one - second_stage_weight,
        )?,
        second_stage_weight: checked_finite(step, "second stage weight", second_stage_weight)?,
        output_latent_weight: checked_positive(
            step,
            "output latent weight",
            next_sigma / sigma * (-step_size * options.eta()).exp(),
        )?,
        output_primary_weight: checked_finite(
            step,
            "output primary weight",
            exponential_integrator_phi_one(-eta_step) - output_stage_weight,
        )?,
        output_stage_weight: checked_finite(step, "output stage weight", output_stage_weight)?,
        first_noise_root,
        second_segment_old,
        second_segment_new,
        output_segment_old,
        output_segment_new,
    })
}

fn checked_phi_two(value: f32, step: usize) -> Result<f32, Seeds3Error> {
    let value = checked_nonzero(step, "phi-two argument", value)?;
    checked_finite(step, "phi two", exponential_integrator_phi_two(value))
}

fn noise_root(value: f32, step: usize, name: &'static str) -> Result<f32, Seeds3Error> {
    checked_nonnegative(step, name, (-value.exp_m1()).sqrt())
}

fn draw_noise(
    inject_noise: bool,
    count: usize,
    transaction: &mut sim_tensor::CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Option<Vec<f64>>, Seeds3Error> {
    if inject_noise {
        Ok(Some(transaction.draw_normal(count, context.cancellation)?))
    } else {
        Ok(None)
    }
}

fn noise_value(noise: Option<&[f64]>, element: usize) -> f32 {
    noise
        .and_then(|values| values.get(element))
        .copied()
        .unwrap_or(0.0) as f32
}

fn call_denoiser(
    denoiser: &mut impl FnMut(&Tensor, f32, usize, Seeds3DenoiserStage) -> Result<Tensor, String>,
    latent: &Tensor,
    sigma: f32,
    step: usize,
    stage: Seeds3DenoiserStage,
) -> Result<Tensor, Seeds3Error> {
    denoiser(latent, sigma, step, stage).map_err(|reason| Seeds3Error::Denoiser {
        step,
        stage,
        reason,
    })
}

fn validate_descriptor(
    current: &Tensor,
    denoised: &Tensor,
    step: usize,
    stage: Seeds3DenoiserStage,
) -> Result<(), Seeds3Error> {
    if current.descriptor() != denoised.descriptor() {
        return Err(Seeds3Error::DenoiserContract { step, stage });
    }
    Ok(())
}

fn validate_values(values: &[f32], step: usize, stage: &'static str) -> Result<(), Seeds3Error> {
    for (element, value) in values.iter().copied().enumerate() {
        checked_element(value, step, stage, element)?;
    }
    Ok(())
}

fn checked_element(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<f32, Seeds3Error> {
    if !value.is_finite() {
        return Err(Seeds3Error::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(value)
}

fn checked_finite(step: usize, coefficient: &'static str, value: f32) -> Result<f32, Seeds3Error> {
    if !value.is_finite() {
        return Err(Seeds3Error::InvalidCoefficient {
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
) -> Result<f32, Seeds3Error> {
    let value = checked_finite(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(Seeds3Error::InvalidCoefficient {
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
) -> Result<f32, Seeds3Error> {
    let value = checked_finite(step, coefficient, value)?;
    if value < 0.0 {
        return Err(Seeds3Error::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}

fn checked_nonzero(step: usize, coefficient: &'static str, value: f32) -> Result<f32, Seeds3Error> {
    let value = checked_finite(step, coefficient, value)?;
    if value == 0.0 {
        return Err(Seeds3Error::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
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
