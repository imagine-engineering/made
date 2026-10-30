use crate::{
    CompatibilityNoiseRequest, SamplerDefinition, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProgress, SamplingSession, SamplingTrace, SchedulerError, SchedulerRegistry,
};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, DeviceId, ExecutionContext, RngCheckpoint,
    RngCompatibilityError, RngGenerationPlacement, RngSeedTransform, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DPM_2_SAMPLER_ID: &str = "dpm_2";
pub const DPM_2_FEATURE_ID: &str = "COMFY-MODEL-0162";
pub const DPM_2_SOURCE_ORDINAL: u16 = 8;
pub const DPM_2_CHURN_NOISE_CONTRACT_ID: &str = "COMFY-RNG-D68A0DD3FBE1";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DPM_2_SAMPLER_ID,
    feature_id: DPM_2_FEATURE_ID,
    source_ordinal: DPM_2_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/dpm_2_comfy_model_0162",
    stochastic: true,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dpm2Options {
    s_churn: f32,
    s_tmin: f32,
    s_tmax: f32,
    s_noise: f32,
}

impl Dpm2Options {
    pub fn new(
        s_churn: f32,
        s_tmin: f32,
        s_tmax: f32,
        s_noise: f32,
    ) -> Result<Self, Dpm2SamplerError> {
        if !s_churn.is_finite() {
            return Err(Dpm2SamplerError::InvalidOption {
                name: "s_churn",
                value: s_churn,
            });
        }
        if !s_tmin.is_finite() {
            return Err(Dpm2SamplerError::InvalidOption {
                name: "s_tmin",
                value: s_tmin,
            });
        }
        if s_tmax.is_nan() || s_tmax == f32::NEG_INFINITY {
            return Err(Dpm2SamplerError::InvalidOption {
                name: "s_tmax",
                value: s_tmax,
            });
        }
        if !s_noise.is_finite() {
            return Err(Dpm2SamplerError::InvalidOption {
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

    pub fn source_defaults() -> Self {
        Self {
            s_churn: 0.0,
            s_tmin: 0.0,
            s_tmax: f32::INFINITY,
            s_noise: 1.0,
        }
    }

    pub fn s_churn(self) -> f32 {
        self.s_churn
    }

    pub fn s_tmin(self) -> f32 {
        self.s_tmin
    }

    pub fn s_tmax(self) -> f32 {
        self.s_tmax
    }

    pub fn s_noise(self) -> f32 {
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

impl Default for Dpm2Options {
    fn default() -> Self {
        Self::source_defaults()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dpm2DenoiserStage {
    Primary,
    Midpoint,
}

#[allow(clippy::too_many_arguments)]
pub fn sample_dpm_2<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    expected_profile: &crate::SamplingProfileIdentity,
    initial: Tensor,
    sigmas: &[f32],
    options: Dpm2Options,
    noise_request: CompatibilityNoiseRequest,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, f32, usize, Dpm2DenoiserStage) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(SamplingTrace, Option<(RngCheckpoint, RngCheckpoint)>), Dpm2SamplerError>
where
    CallbackError: Display,
{
    context.check()?;
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    plan.validate(&samplers, &schedulers, expected_profile)?;
    if plan.sampler().as_str() != DPM_2_SAMPLER_ID {
        return Err(Dpm2SamplerError::WrongSampler(
            plan.sampler().as_str().to_owned(),
        ));
    }
    let step_count = sigmas.len().saturating_sub(1);
    let mut owned_sigmas = Vec::new();
    owned_sigmas
        .try_reserve_exact(sigmas.len())
        .map_err(|_| SamplingError::OutOfMemory("DPM2 sigma schedule"))?;
    owned_sigmas.extend_from_slice(sigmas);
    let seed = plan.seed();
    let mut session = SamplingSession::new(plan, owned_sigmas, initial)?;

    let churn_enabled = sigmas
        .iter()
        .take(step_count)
        .copied()
        .any(|sigma| options.gamma(sigma, step_count) > 0.0);
    let mut noise_transaction = if churn_enabled {
        Some(noise_request.open_transaction(
            DPM_2_CHURN_NOISE_CONTRACT_ID,
            i128::from(seed),
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(DeviceId::CPU),
            None,
            context.cancellation,
        )?)
    } else {
        None
    };
    let noise_before = noise_transaction
        .as_ref()
        .map(CompatibilityRngTransaction::checkpoint);

    for (step, pair) in sigmas.windows(2).enumerate() {
        context.check()?;
        let sigma = *pair
            .first()
            .ok_or(SamplingError::Overflow("DPM2 current sigma lookup"))?;
        let next_sigma = *pair
            .get(1)
            .ok_or(SamplingError::Overflow("DPM2 next sigma lookup"))?;
        let gamma = options.gamma(sigma, step_count);
        let sigma_hat = sigma * (gamma + 1.0);
        if !sigma_hat.is_finite() || sigma_hat <= 0.0 {
            return Err(Dpm2SamplerError::NonFinite {
                step,
                stage: "sigma hat",
                element: 0,
            });
        }

        let current = session.current().clone();
        let churned = if gamma > 0.0 {
            let transaction = noise_transaction
                .as_mut()
                .ok_or(Dpm2SamplerError::MissingNoiseTransaction { step })?;
            apply_churn(
                backend,
                &current,
                sigma,
                sigma_hat,
                options.s_noise,
                step,
                transaction,
                context,
            )?
        } else {
            current
        };
        let denoised =
            denoiser(&churned, sigma_hat, step, Dpm2DenoiserStage::Primary).map_err(|reason| {
                Dpm2SamplerError::Denoiser {
                    step,
                    stage: Dpm2DenoiserStage::Primary,
                    reason,
                }
            })?;
        validate_denoiser_contract(&churned, &denoised, step, Dpm2DenoiserStage::Primary)?;
        validate_finite_tensor(backend, &denoised, step, "primary denoiser", context)?;
        let primary_derivative =
            derivative(backend, &churned, &denoised, sigma_hat, step, context)?;

        let observed = session.observe_step(
            &churned,
            denoised.clone(),
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
        )?;

        let next = if next_sigma == 0.0 {
            let next = advance(
                backend,
                &churned,
                &primary_derivative,
                next_sigma - sigma_hat,
                step,
                context,
            )?;
            next
        } else {
            let sigma_mid = (sigma_hat.ln() + (next_sigma.ln() - sigma_hat.ln()) * 0.5).exp();
            if !sigma_mid.is_finite() || sigma_mid <= 0.0 {
                return Err(Dpm2SamplerError::NonFinite {
                    step,
                    stage: "midpoint sigma",
                    element: 0,
                });
            }
            let midpoint_input = advance(
                backend,
                &churned,
                &primary_derivative,
                sigma_mid - sigma_hat,
                step,
                context,
            )?;
            let midpoint_denoised = denoiser(
                &midpoint_input,
                sigma_mid,
                step,
                Dpm2DenoiserStage::Midpoint,
            )
            .map_err(|reason| Dpm2SamplerError::Denoiser {
                step,
                stage: Dpm2DenoiserStage::Midpoint,
                reason,
            })?;
            validate_denoiser_contract(
                &midpoint_input,
                &midpoint_denoised,
                step,
                Dpm2DenoiserStage::Midpoint,
            )?;
            validate_finite_tensor(
                backend,
                &midpoint_denoised,
                step,
                "midpoint denoiser",
                context,
            )?;
            let midpoint_derivative = derivative(
                backend,
                &midpoint_input,
                &midpoint_denoised,
                sigma_mid,
                step,
                context,
            )?;
            let next = advance(
                backend,
                &churned,
                &midpoint_derivative,
                next_sigma - sigma_hat,
                step,
                context,
            )?;
            next
        };

        observed.commit(next, context.cancellation)?;
    }

    let sampling = session.finish()?;
    let checkpoints = match (noise_before, noise_transaction) {
        (Some(before), Some(transaction)) => Some((before, transaction.commit())),
        (None, None) => None,
        _ => return Err(Dpm2SamplerError::MissingNoiseTransaction { step: step_count }),
    };
    Ok((sampling, checkpoints))
}

fn apply_churn(
    backend: &CpuBackend,
    current: &Tensor,
    sigma: f32,
    sigma_hat: f32,
    noise_scale: f32,
    step: usize,
    transaction: &mut CompatibilityRngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpm2SamplerError> {
    let count = usize::try_from(current.descriptor().element_count()?)
        .map_err(|_| TensorError::ShapeOverflow)?;
    let normal = transaction.draw_normal(count, context.cancellation)?;
    let mut churned_values = backend.workspace_vec::<f32>(context, count)?;
    let current_values = tensor_to_f32(backend, current, context)?;
    let perturbation_scale = (sigma_hat * sigma_hat - sigma * sigma).sqrt();
    for (element, (current_value, noise_value)) in current_values.iter().zip(normal).enumerate() {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let noise_value = noise_value as f32 * noise_scale;
        let churned_value = noise_value.mul_add(perturbation_scale, *current_value);
        if !noise_value.is_finite() || !churned_value.is_finite() {
            return Err(Dpm2SamplerError::NonFinite {
                step,
                stage: "churn",
                element,
            });
        }
        churned_values.try_push(churned_value)?;
    }
    tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &churned_values,
        context,
    )
    .map_err(Dpm2SamplerError::TensorKernel)
}

fn derivative(
    backend: &CpuBackend,
    current: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpm2SamplerError> {
    let current_values = tensor_to_f32(backend, current, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let mut derivative_values = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, (current_value, denoised_value)) in current_values
        .iter()
        .zip(denoised_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let derivative = (current_value - denoised_value) / sigma;
        if !derivative.is_finite() {
            return Err(Dpm2SamplerError::NonFinite {
                step,
                stage: "derivative",
                element,
            });
        }
        derivative_values.try_push(derivative)?;
    }
    tensor_from_f32(
        backend,
        current.descriptor().shape(),
        &derivative_values,
        context,
    )
    .map_err(Dpm2SamplerError::TensorKernel)
}

fn advance(
    backend: &CpuBackend,
    current: &Tensor,
    derivative: &Tensor,
    delta: f32,
    step: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Dpm2SamplerError> {
    let current_values = tensor_to_f32(backend, current, context)?;
    let derivative_values = tensor_to_f32(backend, derivative, context)?;
    let mut next_values = backend.workspace_vec::<f32>(context, current_values.len())?;
    for (element, (current_value, derivative_value)) in current_values
        .iter()
        .zip(derivative_values.iter())
        .enumerate()
    {
        if element.is_multiple_of(256) {
            context.check()?;
        }
        let next = derivative_value.mul_add(delta, *current_value);
        if !next.is_finite() {
            return Err(Dpm2SamplerError::NonFinite {
                step,
                stage: "latent update",
                element,
            });
        }
        next_values.try_push(next)?;
    }
    tensor_from_f32(backend, current.descriptor().shape(), &next_values, context)
        .map_err(Dpm2SamplerError::TensorKernel)
}

fn validate_denoiser_contract(
    input: &Tensor,
    denoised: &Tensor,
    step: usize,
    stage: Dpm2DenoiserStage,
) -> Result<(), Dpm2SamplerError> {
    if input.descriptor() != denoised.descriptor() {
        return Err(Dpm2SamplerError::DenoiserContract { step, stage });
    }
    Ok(())
}

fn validate_finite_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    step: usize,
    stage: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), Dpm2SamplerError> {
    for (element, value) in tensor_to_f32(backend, tensor, context)?.iter().enumerate() {
        if !value.is_finite() {
            return Err(Dpm2SamplerError::NonFinite {
                step,
                stage,
                element,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum Dpm2SamplerError {
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    RngCompatibility(#[from] RngCompatibilityError),
    #[error("DPM2 requires sampler identity `dpm_2`, got {0:?}")]
    WrongSampler(String),
    #[error("DPM2 option {name} must be finite (except positive infinity for s_tmax), got {value}")]
    InvalidOption { name: &'static str, value: f32 },
    #[error("DPM2 denoiser failed at step {step} during {stage:?}: {reason}")]
    Denoiser {
        step: usize,
        stage: Dpm2DenoiserStage,
        reason: String,
    },
    #[error("DPM2 denoiser output descriptor changed at step {step} during {stage:?}")]
    DenoiserContract {
        step: usize,
        stage: Dpm2DenoiserStage,
    },
    #[error("DPM2 produced a non-finite {stage} value at step {step}, element {element}")]
    NonFinite {
        step: usize,
        stage: &'static str,
        element: usize,
    },
    #[error("DPM2 churn at step {step} has no canonical RNG transaction")]
    MissingNoiseTransaction { step: usize },
}
