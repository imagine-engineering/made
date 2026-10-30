use crate::{
    CompatibilityNoiseRequest, NoiseTrace, SamplerDefinition, SamplingError, SamplingPlan,
    SamplingProgress, SchedulerError,
    generated_native_diffusion::{
        EulerOptions, NativeDiffusionSamplerError, sample_euler_canonical,
    },
};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, RngCompatibilityError, RngGenerationPlacement,
    RngSeedTransform, Tensor, TensorDescriptor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32},
};
use std::fmt::Display;
use thiserror::Error;

pub const DDIM_SAMPLER_ID: &str = "ddim";
pub const DDIM_FEATURE_ID: &str = "COMFY-MODEL-0159";
pub const DDIM_SOURCE_ORDINAL: u16 = 41;
pub const DDIM_INPAINT_NOISE_CONTRACT_ID: &str = "COMFY-RNG-B35F0F617BFA";

pub const DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: DDIM_SAMPLER_ID,
    feature_id: DDIM_FEATURE_ID,
    source_ordinal: DDIM_SOURCE_ORDINAL,
    aliases: &[],
    implementation_module: "algorithms/ddim_comfy_model_0159",
    stochastic: true,
};

pub fn ddim_inpaint_replacement_noise(
    backend: &CpuBackend,
    shape: &[u64],
    request: CompatibilityNoiseRequest,
    base_seed: i128,
    context: &ExecutionContext<'_>,
) -> Result<NoiseTrace, DdimError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    let count =
        usize::try_from(descriptor.element_count()?).map_err(|_| TensorError::ShapeOverflow)?;
    let mut transaction = request.open_transaction(
        DDIM_INPAINT_NOISE_CONTRACT_ID,
        base_seed,
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        context.cancellation,
    )?;
    let before = transaction.checkpoint();
    let normal = transaction.draw_normal(count, context.cancellation)?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    for (index, value) in normal.into_iter().enumerate() {
        if index.is_multiple_of(256) {
            context
                .cancellation
                .check()
                .map_err(|_| DdimError::Cancelled)?;
        }
        values.try_push(value as f32)?;
    }
    let noise = tensor_from_f32(backend, shape, &values, context)?;
    let after = transaction.commit();
    Ok(NoiseTrace {
        noise,
        before,
        after,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn sample_ddim<CallbackError>(
    backend: &CpuBackend,
    plan: SamplingPlan,
    initial: Tensor,
    sigmas: &[f32],
    inpaint_replacement_noise: NoiseTrace,
    context: &ExecutionContext<'_>,
    mut denoiser: impl FnMut(&Tensor, &Tensor, f32, usize) -> Result<Tensor, String>,
    mut callback: impl FnMut(&SamplingProgress, &Tensor, &Tensor) -> Result<(), CallbackError>,
) -> Result<(crate::SamplingTrace, NoiseTrace), DdimError>
where
    CallbackError: Display,
{
    if initial.descriptor() != inpaint_replacement_noise.noise.descriptor() {
        return Err(DdimError::ReplacementNoiseContract {
            expected: format!("{:?}", initial.descriptor()),
            actual: format!("{:?}", inpaint_replacement_noise.noise.descriptor()),
        });
    }
    let expected_profile = plan.profile().clone();
    let replacement_noise = &inpaint_replacement_noise.noise;
    let (sampling, checkpoints) = sample_euler_canonical(
        backend,
        plan,
        &expected_profile,
        DDIM_SAMPLER_ID,
        initial,
        sigmas,
        EulerOptions::source_defaults(),
        None,
        context,
        |latent, sigma, step| denoiser(latent, replacement_noise, sigma, step),
        |progress, latent, denoised| callback(progress, denoised, latent),
    )
    .map_err(map_euler_error)?;
    if checkpoints.is_some() {
        return Err(DdimError::UnexpectedEulerNoiseCheckpoint);
    }
    Ok((sampling, inpaint_replacement_noise))
}

fn map_euler_error(error: NativeDiffusionSamplerError) -> DdimError {
    match error {
        NativeDiffusionSamplerError::Tensor(TensorError::Cancelled) => DdimError::Cancelled,
        NativeDiffusionSamplerError::Tensor(error) => DdimError::Tensor(error),
        NativeDiffusionSamplerError::TensorKernel(error) => DdimError::TensorKernel(error),
        NativeDiffusionSamplerError::RngCompatibility(error) => DdimError::RngCompatibility(error),
        NativeDiffusionSamplerError::Sampling(error) => DdimError::Sampling(error),
        NativeDiffusionSamplerError::Scheduler(error) => DdimError::Scheduler(error),
        NativeDiffusionSamplerError::Denoiser { step, reason } => {
            DdimError::Denoiser { step, reason }
        }
        NativeDiffusionSamplerError::DenoiserShape {
            step,
            expected,
            actual,
        } => DdimError::DenoiserContract {
            step,
            expected,
            actual,
        },
        NativeDiffusionSamplerError::WrongEulerSampler(identity) => {
            DdimError::WrongSampler(identity)
        }
        NativeDiffusionSamplerError::NonFiniteEuler { step, element, .. } => {
            DdimError::NonFiniteOutput {
                step,
                index: element,
            }
        }
        NativeDiffusionSamplerError::InvalidSigma {
            step,
            sigma,
            next_sigma,
        } => DdimError::Sampling(SamplingError::InvalidSigma {
            step,
            sigma,
            next_sigma,
        }),
        error => DdimError::EulerFoundation(error),
    }
}

#[derive(Debug, Error)]
pub enum DdimError {
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
    #[error(transparent)]
    EulerFoundation(NativeDiffusionSamplerError),
    #[error("DDIM sampling was cancelled")]
    Cancelled,
    #[error("DDIM requires sampler identity `ddim`, got {0:?}")]
    WrongSampler(String),
    #[error("DDIM replacement-noise contract mismatch: expected {expected}, got {actual}")]
    ReplacementNoiseContract { expected: String, actual: String },
    #[error("DDIM denoiser failed at step {step}: {reason}")]
    Denoiser { step: usize, reason: String },
    #[error("DDIM denoiser contract mismatch at step {step}: expected {expected}, got {actual}")]
    DenoiserContract {
        step: usize,
        expected: String,
        actual: String,
    },
    #[error("DDIM produced a non-finite latent at step {step}, element {index}")]
    NonFiniteOutput { step: usize, index: usize },
    #[error("DDIM's deterministic Euler adapter unexpectedly produced a churn RNG checkpoint")]
    UnexpectedEulerNoiseCheckpoint,
}
