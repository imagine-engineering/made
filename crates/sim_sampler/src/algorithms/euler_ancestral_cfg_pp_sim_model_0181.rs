use crate::{
    CfgPpDenoiserOutput, CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry,
    SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError, SamplingProgress,
    SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
    generated_native_diffusion::{
        NativeDiffusionSamplerError, observe_euler_denoised,
        validate_euler_noise_generation_device,
    },
    standard_ancestral_step, validate_cfg_pp_denoiser_output,
};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, DeviceId, ExecutionContext,
    RngCheckpoint, RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor,
    TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const EULER_ANCESTRAL_CFG_PP_SAMPLER_ID: &str = "euler_ancestral_cfg_pp";
pub const EULER_ANCESTRAL_CFG_PP_FEATURE_ID: &str = "COMFY-MODEL-0181";
pub const EULER_ANCESTRAL_CFG_PP_SOURCE_ORDINAL: u16 = 3;
pub const EULER_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: EULER_ANCESTRAL_CFG_PP_SAMPLER_ID,
    feature_id: EULER_ANCESTRAL_CFG_PP_FEATURE_ID,
    source_ordinal: EULER_ANCESTRAL_CFG_PP_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/euler_ancestral_cfg_pp_comfy_model_0181",
    stochastic: true,
};

pub type EulerAncestralCfgPpDenoiserOutput = CfgPpDenoiserOutput;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerAncestralCfgPpOptions {
    pub eta: f32,
    pub noise_scale: f32,
}

impl EulerAncestralCfgPpOptions {
    pub const fn source_defaults() -> Self {
        Self {
            eta: 1.0,
            noise_scale: 1.0,
        }
    }
}

impl Default for EulerAncestralCfgPpOptions {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Debug, Error)]
pub enum EulerAncestralCfgPpError {
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
    #[error(transparent)]
    EulerFoundation(#[from] NativeDiffusionSamplerError),
    #[error("Euler ancestral CFG++ requires sampler identity {expected:?}, got {actual:?}")]
    WrongSampler {
        expected: &'static str,
        actual: String,
    },
    #[error("Euler ancestral CFG++ option {name} must be finite, got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("Euler ancestral CFG++ denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("Euler ancestral CFG++ {output} descriptor changed at step {step}")]
    DenoiserContract {
        step: usize,
        output: &'static str,
    },
    #[error("Euler ancestral CFG++ coefficient {coefficient} is invalid at step {step}: {value}")]
    InvalidCoefficient {
        step: usize,
        coefficient: &'static str,
        value: f32,
    },
    #[error("Euler ancestral CFG++ produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("Euler ancestral CFG++ allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("native Euler ancestral CFG++ noise generation on {device:?} is unavailable: {reason}")]
    DeviceUnavailable { device: DeviceId, reason: String },
}

#[allow(clippy::too_many_arguments)]
pub fn sample_euler_ancestral_cfg_pp<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: EulerAncestralCfgPpOptions,
    context: &ExecutionContext<'_>,
    denoiser: impl FnMut(
        &Tensor,
        f32,
        usize,
    ) -> Result<EulerAncestralCfgPpDenoiserOutput, String>,
    callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), EulerAncestralCfgPpError>
where
    CallbackError: Display,
{
    sample_euler_cfg_pp_family(
        backend,
        plan,
        EULER_ANCESTRAL_CFG_PP_SAMPLER_ID,
        profile,
        initial,
        sigmas,
        noise_request,
        options,
        context,
        denoiser,
        callback,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_euler_cfg_pp_family<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_sampler: &'static str,
    profile: &impl SamplingProfile,
    initial: Tensor,
    sigmas: &[f32],
    noise_request: CompatibilityNoiseRequest,
    options: EulerAncestralCfgPpOptions,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(
        &Tensor,
        f32,
        usize,
    ) -> Result<EulerAncestralCfgPpDenoiserOutput, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, RngCheckpoint, RngCheckpoint), EulerAncestralCfgPpError>
where
    CallbackError: Display,
{
    context.check()?;
    validate_options(options)?;
    plan.validate(
        &SamplerRegistry::foundational()?,
        &SchedulerRegistry::foundational()?,
        profile.identity(),
    )?;
    if plan.sampler().as_str() != expected_sampler {
        return Err(EulerAncestralCfgPpError::WrongSampler {
            expected: expected_sampler,
            actual: plan.sampler().as_str().to_owned(),
        });
    }

    let device = initial.descriptor().device();
    validate_generation_device(device)?;
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| EulerAncestralCfgPpError::OutOfMemory("sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let seed = plan.seed();
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;
    let effective_noise_scale = if options.noise_scale > 0.0 {
        profile.scale_sampler_noise(options.noise_scale)?
    } else {
        options.noise_scale
    };
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
    let mut transaction = noise_request.open_transaction(
        EULER_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
        i128::from(seed),
        seed_transform,
        generation_placement,
        None,
        context.cancellation,
    )?;
    let before = transaction.checkpoint();

    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = pair[0];
        let next_sigma = pair[1];
        let current = session.current().clone();
        let output = denoiser(&current, sigma, step)
            .map_err(|reason| EulerAncestralCfgPpError::Denoiser { step, reason })?;
        validate_output_contract(&current, &output, step)?;
        let observed = observe_euler_denoised(
            &mut session,
            &current,
            output.denoised.clone(),
            sigma,
            context,
            &mut callback,
        )?;
        let next = if next_sigma == 0.0 {
            validate_tensor_finite(
                backend,
                &output.denoised,
                step,
                "terminal guided denoiser",
                context,
            )?;
            output.denoised
        } else {
            cfg_pp_step(
                backend,
                &current,
                &output,
                profile,
                sigma,
                next_sigma,
                options,
                effective_noise_scale,
                step,
                &mut transaction,
                context,
            )?
        };
        observed.commit(next, context.cancellation)?;
    }

    context.check()?;
    let trace = session.finish()?;
    let after = transaction.commit();
    Ok((trace, before, after))
}

fn validate_options(options: EulerAncestralCfgPpOptions) -> Result<(), EulerAncestralCfgPpError> {
    for (name, value) in [("eta", options.eta), ("s_noise", options.noise_scale)] {
        if !value.is_finite() {
            return Err(EulerAncestralCfgPpError::InvalidOption { name, value });
        }
    }
    Ok(())
}

fn validate_generation_device(device: DeviceId) -> Result<(), EulerAncestralCfgPpError> {
    validate_euler_noise_generation_device(device).map_err(|error| {
        EulerAncestralCfgPpError::DeviceUnavailable {
            device,
            reason: error.to_string(),
        }
    })
}

fn validate_output_contract(
    current: &Tensor,
    output: &EulerAncestralCfgPpDenoiserOutput,
    step: usize,
) -> Result<(), EulerAncestralCfgPpError> {
    validate_cfg_pp_denoiser_output(current, output).map_err(|error| {
        EulerAncestralCfgPpError::DenoiserContract {
            step,
            output: error.output,
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn cfg_pp_step(
    backend: &CpuBackend,
    current: &Tensor,
    output: &EulerAncestralCfgPpDenoiserOutput,
    profile: &impl SamplingProfile,
    sigma: f32,
    next_sigma: f32,
    options: EulerAncestralCfgPpOptions,
    effective_noise_scale: f32,
    step: usize,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, EulerAncestralCfgPpError> {
    let lambda_source = checked_finite(step, "source half-log-SNR", profile.half_log_snr(sigma)?)?;
    let lambda_target = checked_finite(
        step,
        "target half-log-SNR",
        profile.half_log_snr(next_sigma)?,
    )?;
    let alpha_source = checked_positive(step, "source alpha", sigma * lambda_source.exp())?;
    let alpha_target = checked_positive(step, "target alpha", next_sigma * lambda_target.exp())?;
    let sigma_source = checked_nonnegative(step, "source ancestral sigma", sigma / alpha_source)?;
    let sigma_target = checked_nonnegative(
        step,
        "target ancestral sigma",
        next_sigma / alpha_target,
    )?;
    let (sigma_down, sigma_up) =
        standard_ancestral_step(sigma_source, sigma_target, options.eta).map_err(|_| {
            EulerAncestralCfgPpError::InvalidCoefficient {
                step,
                coefficient: "standard ancestral step",
                value: f32::NAN,
            }
        })?;
    let sigma_down = checked_nonnegative(
        step,
        "scaled ancestral sigma down",
        alpha_target * sigma_down,
    )?;
    let current_values = tensor_to_f32(backend, current, context)?;
    let guided_values = tensor_to_f32(backend, &output.denoised, context)?;
    let unconditional_values = tensor_to_f32(backend, &output.unconditional_denoised, context)?;
    let draw_noise = options.eta > 0.0 && effective_noise_scale > 0.0;
    let noise = if draw_noise {
        Some(transaction.draw_normal(current_values.len(), context.cancellation)?)
    } else {
        None
    };
    let noise_scale = checked_finite(
        step,
        "ancestral noise scale",
        alpha_target * effective_noise_scale * sigma_up,
    )?;
    let mut values = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, ((current, guided), unconditional)) in current_values
        .iter()
        .copied()
        .zip(guided_values.iter().copied())
        .zip(unconditional_values.iter().copied())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let derivative = (current - alpha_source * unconditional) / sigma;
        let mut value = alpha_target * guided + sigma_down * derivative;
        if let Some(noise) = noise.as_ref() {
            let noise = noise.get(element).copied().ok_or(
                EulerAncestralCfgPpError::InvalidCoefficient {
                    step,
                    coefficient: "noise element count",
                    value: f32::NAN,
                },
            )?;
            value = (noise as f32).mul_add(noise_scale, value);
        }
        checked_value(value, step, "CFG++ Euler update", element)?;
        values.try_push(value)?;
    }
    Ok(tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &values,
        context,
    )?)
}

fn validate_tensor_finite(
    backend: &CpuBackend,
    tensor: &Tensor,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), EulerAncestralCfgPpError> {
    for (element, value) in tensor_to_f32(backend, tensor, context)?.iter().copied().enumerate() {
        checked_value(value, step, stage, element)?;
    }
    Ok(())
}

fn checked_value(
    value: f32,
    step: usize,
    stage: &'static str,
    element: usize,
) -> Result<f32, EulerAncestralCfgPpError> {
    if !value.is_finite() {
        return Err(EulerAncestralCfgPpError::NonFinite {
            step,
            stage,
            element,
        });
    }
    Ok(value)
}

fn checked_finite(
    step: usize,
    coefficient: &'static str,
    value: f32,
) -> Result<f32, EulerAncestralCfgPpError> {
    if !value.is_finite() {
        return Err(EulerAncestralCfgPpError::InvalidCoefficient {
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
) -> Result<f32, EulerAncestralCfgPpError> {
    let value = checked_finite(step, coefficient, value)?;
    if value <= 0.0 {
        return Err(EulerAncestralCfgPpError::InvalidCoefficient {
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
) -> Result<f32, EulerAncestralCfgPpError> {
    let value = checked_finite(step, coefficient, value)?;
    if value < 0.0 {
        return Err(EulerAncestralCfgPpError::InvalidCoefficient {
            step,
            coefficient,
            value,
        });
    }
    Ok(value)
}
