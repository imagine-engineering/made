use crate::validate_identifier;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use sim_model::{NativeModelPayload, NativeModelResourceRole};
use sim_tensor::{
    AutogradTape, CpuBackend, ExecutionContext, ScalarAutogradValue, Tensor,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::sync::{Arc, OnceLock};
use thiserror::Error;

const SD15_TIMESTEPS: usize = 1_000;
const SD15_LINEAR_START: f64 = 0.00085;
const SD15_LINEAR_END: f64 = 0.012;
pub const SD15_SAMPLING_PROFILE_ID: &str = "sd15-discrete-epsilon-v1";
pub const LOTUS_SDPOSE_SAMPLING_PROFILE_ID: &str = "lotus-sdpose-discrete-denoised-v1";
pub const AURAFLOW_SAMPLING_PROFILE_ID: &str = "auraflow-discrete-flow-shift-1.73-v1";
pub const QWEN_IMAGE_SAMPLING_PROFILE_ID: &str = "qwen-image-flux-shift-1.15-v1";
const AURAFLOW_TIMESTEPS: usize = 1_000;
const AURAFLOW_SHIFT: f32 = 1.73;
const AURAFLOW_MULTIPLIER: f32 = 1.0;
const QWEN_IMAGE_TIMESTEPS: usize = 10_000;
const QWEN_IMAGE_SHIFT: f32 = 1.15;

pub fn exponential_integrator_phi_one(value: f32) -> f32 {
    value.exp_m1()
}

pub fn exponential_integrator_phi_two(value: f32) -> f32 {
    (value.exp_m1() - value) / value
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct SamplingProfileIdentity(String);

impl SamplingProfileIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, SamplingProfileError> {
        let value = value.into();
        validate_identifier("sampling profile", &value)
            .map_err(|_| SamplingProfileError::InvalidIdentity(value.clone()))?;
        Ok(Self(value))
    }

    pub fn sd15() -> Self {
        Self(SD15_SAMPLING_PROFILE_ID.to_owned())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SamplingProfileIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionInterpretation {
    Epsilon,
    VPrediction,
    Flow,
    Denoised,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplingSnrMode {
    Standard,
    ConstantFlow { shift: f32 },
}

pub trait SamplingProfile {
    fn identity(&self) -> &SamplingProfileIdentity;
    fn prediction(&self) -> PredictionInterpretation;
    fn sigma_count(&self) -> usize;
    fn sigma_at_index(&self, index: usize) -> Result<f32, SamplingProfileError> {
        if index >= self.sigma_count() {
            return Err(SamplingProfileError::GridIndex(index));
        }
        self.sigma_at_model_time(index as f32)
    }
    fn sigma_at_model_time(&self, model_time: f32) -> Result<f32, SamplingProfileError>;
    fn model_time_for_sigma(&self, sigma: f32) -> Result<f32, SamplingProfileError>;
    fn sigma_for_model_time(&self, model_time: f32) -> Result<f32, SamplingProfileError> {
        self.sigma_at_model_time(model_time)
    }
    fn sampling_percent_for_sigma(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        let maximum = self.sigma_count().saturating_sub(1) as f32;
        if maximum == 0.0 {
            return Err(SamplingProfileError::EmptyGrid);
        }
        Ok((1.0 - self.model_time_for_sigma(sigma)? / maximum).clamp(0.0, 1.0))
    }
    fn sigma_for_sampling_percent(&self, percent: f32) -> Result<f32, SamplingProfileError> {
        validate_sampling_percent(percent)?;
        if percent == 0.0 {
            return Ok(999_999_999.9);
        }
        if percent == 1.0 {
            return Ok(0.0);
        }
        self.sigma_at_model_time((1.0 - percent) * self.sigma_count().saturating_sub(1) as f32)
    }
    fn sigma_min(&self) -> f32;
    fn sigma_max(&self) -> f32;
    fn half_log_snr(&self, sigma: f32) -> Result<f32, SamplingProfileError>;
    fn sigma_from_half_log_snr(&self, half_log_snr: f32) -> Result<f32, SamplingProfileError>;
    fn adjust_first_sigma_for_snr(&self, sigmas: &mut [f32]) -> Result<(), SamplingProfileError>;
    fn scale_sampler_noise(&self, sampler_noise_scale: f32) -> Result<f32, SamplingProfileError>;
    fn is_max_denoise(&self, sigma: f32) -> Result<bool, SamplingProfileError> {
        validate_sigma(sigma, true)?;
        let maximum = self.sigma_max();
        let tolerance = 1.0e-5 * maximum.abs().max(sigma.abs());
        Ok(sigma > maximum || (sigma - maximum).abs() <= tolerance)
    }
    fn scale_initial_noise_in_place(
        &self,
        noise: &mut [f32],
        latent: &[f32],
        sigma: f32,
        max_denoise: bool,
    ) -> Result<(), SamplingProfileError> {
        validate_equal_lengths(noise, latent, "initial noise and latent")?;
        validate_sigma(sigma, false)?;
        if self.prediction() == PredictionInterpretation::Flow {
            let noise_scale = self.scale_sampler_noise(1.0)?;
            for (index, (noise_value, latent_value)) in
                noise.iter_mut().zip(latent.iter()).enumerate()
            {
                let value =
                    (sigma * noise_scale).mul_add(*noise_value, (1.0 - sigma) * *latent_value);
                if !value.is_finite() {
                    return Err(SamplingProfileError::NonFiniteOutput {
                        operation: "initial flow-noise scaling",
                        index,
                    });
                }
                *noise_value = value;
            }
            return Ok(());
        }
        let scale = if max_denoise {
            (1.0 + sigma * sigma).sqrt()
        } else {
            sigma
        };
        for (index, (noise_value, latent_value)) in noise.iter_mut().zip(latent.iter()).enumerate()
        {
            let value = noise_value.mul_add(scale, *latent_value);
            if !value.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "initial noise scaling",
                    index,
                });
            }
            *noise_value = value;
        }
        Ok(())
    }
    fn inverse_noise_scaling_in_place(
        &self,
        _sigma: f32,
        _latent: &mut [f32],
    ) -> Result<(), SamplingProfileError> {
        Ok(())
    }
    fn scale_model_input_in_place(
        &self,
        values: &mut [f32],
        sigma: f32,
    ) -> Result<(), SamplingProfileError> {
        validate_sigma(sigma, false)?;
        if self.prediction() == PredictionInterpretation::Flow {
            return Ok(());
        }
        let divisor = (sigma * sigma + 1.0).sqrt();
        for (index, value) in values.iter_mut().enumerate() {
            *value /= divisor;
            if !value.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "model input scaling",
                    index,
                });
            }
        }
        Ok(())
    }
    fn interpret_prediction_in_place(
        &self,
        model_output: &mut [f32],
        model_input: &[f32],
        sigma: f32,
    ) -> Result<(), SamplingProfileError> {
        validate_equal_lengths(model_output, model_input, "prediction and model input")?;
        validate_sigma(sigma, false)?;
        for (index, (output, input)) in model_output.iter_mut().zip(model_input.iter()).enumerate()
        {
            let denoised = match self.prediction() {
                PredictionInterpretation::Epsilon | PredictionInterpretation::Flow => {
                    output.mul_add(-sigma, *input)
                }
                PredictionInterpretation::VPrediction => {
                    let denominator = (sigma * sigma + 1.0).sqrt();
                    *input / (sigma * sigma + 1.0) - *output * sigma / denominator
                }
                PredictionInterpretation::Denoised => *output,
            };
            if !denoised.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "prediction interpretation",
                    index,
                });
            }
            *output = denoised;
        }
        Ok(())
    }
}

pub fn standard_ancestral_step(
    sigma_from: f32,
    sigma_to: f32,
    eta: f32,
) -> Result<(f32, f32), SamplingProfileError> {
    if eta == 0.0 {
        return Ok((sigma_to, 0.0));
    }
    if !sigma_from.is_finite()
        || sigma_from <= 0.0
        || !sigma_to.is_finite()
        || sigma_to < 0.0
        || sigma_to > sigma_from
        || !eta.is_finite()
    {
        return Err(SamplingProfileError::InvalidAncestralStep {
            sigma_from,
            sigma_to,
            eta,
        });
    }
    let sigma_from_squared = sigma_from * sigma_from;
    let sigma_to_squared = sigma_to * sigma_to;
    let radicand = sigma_to_squared * (sigma_from_squared - sigma_to_squared) / sigma_from_squared;
    let sigma_up = sigma_to.min(eta * radicand.sqrt());
    let sigma_down = (sigma_to_squared - sigma_up * sigma_up).sqrt();
    if !sigma_down.is_finite() || sigma_down < 0.0 || !sigma_up.is_finite() {
        return Err(SamplingProfileError::InvalidAncestralStep {
            sigma_from,
            sigma_to,
            eta,
        });
    }
    Ok((sigma_down, sigma_up))
}

pub fn rectified_flow_ancestral_step(
    sigma_from: f32,
    sigma_to: f32,
    eta: f32,
) -> Result<(f32, f32), SamplingProfileError> {
    if !sigma_from.is_finite()
        || sigma_from <= 0.0
        || !sigma_to.is_finite()
        || sigma_to < 0.0
        || sigma_to > sigma_from
        || !eta.is_finite()
    {
        return Err(SamplingProfileError::InvalidAncestralStep {
            sigma_from,
            sigma_to,
            eta,
        });
    }

    let downstep_ratio = 1.0 + (sigma_to / sigma_from - 1.0) * eta;
    let sigma_down = sigma_to * downstep_ratio;
    let alpha_to = 1.0 - sigma_to;
    let alpha_down = 1.0 - sigma_down;
    if alpha_down == 0.0 {
        return Err(SamplingProfileError::InvalidAncestralStep {
            sigma_from,
            sigma_to,
            eta,
        });
    }
    let sigma_to_squared = sigma_to * sigma_to;
    let sigma_down_squared = sigma_down * sigma_down;
    let alpha_to_squared = alpha_to * alpha_to;
    let alpha_down_squared = alpha_down * alpha_down;
    let radicand = sigma_to_squared - sigma_down_squared * alpha_to_squared / alpha_down_squared;
    let renoise_coefficient = radicand.sqrt();
    if !sigma_down.is_finite()
        || sigma_down < 0.0
        || !renoise_coefficient.is_finite()
        || renoise_coefficient < 0.0
    {
        return Err(SamplingProfileError::InvalidAncestralStep {
            sigma_from,
            sigma_to,
            eta,
        });
    }
    Ok((sigma_down, renoise_coefficient))
}

#[derive(Clone, Debug)]
pub struct DiscreteSamplingProfile {
    identity: SamplingProfileIdentity,
    prediction: PredictionInterpretation,
    sigmas: Arc<[f32]>,
    snr_mode: SamplingSnrMode,
    noise_scale: f32,
    time_mapping: ModelTimeMapping,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ModelTimeMapping {
    DiscreteGrid,
    DiscreteFlow { shift: f32, multiplier: f32 },
    Flux { shift: f32 },
}

impl DiscreteSamplingProfile {
    pub fn new(
        identity: SamplingProfileIdentity,
        prediction: PredictionInterpretation,
        sigmas: Arc<[f32]>,
    ) -> Result<Self, SamplingProfileError> {
        let snr_mode = if prediction == PredictionInterpretation::Flow {
            SamplingSnrMode::ConstantFlow { shift: 1.0 }
        } else {
            SamplingSnrMode::Standard
        };
        Self::new_with_sampling_parameters(identity, prediction, sigmas, snr_mode, 1.0)
    }

    pub fn new_with_sampling_parameters(
        identity: SamplingProfileIdentity,
        prediction: PredictionInterpretation,
        sigmas: Arc<[f32]>,
        snr_mode: SamplingSnrMode,
        noise_scale: f32,
    ) -> Result<Self, SamplingProfileError> {
        Self::new_with_time_mapping(
            identity,
            prediction,
            sigmas,
            snr_mode,
            noise_scale,
            ModelTimeMapping::DiscreteGrid,
        )
    }

    pub fn sd15() -> Result<Self, SamplingProfileError> {
        static SIGMAS: OnceLock<Result<Arc<[f32]>, SamplingProfileError>> = OnceLock::new();
        let sigmas = SIGMAS
            .get_or_init(|| build_sd15_sigmas().map(Arc::from))
            .clone()?;
        Self::new(
            SamplingProfileIdentity::sd15(),
            PredictionInterpretation::Epsilon,
            sigmas,
        )
    }

    pub fn sd15_with_prediction(
        prediction: PredictionInterpretation,
    ) -> Result<Self, SamplingProfileError> {
        let sigmas = Self::sd15()?.sigmas;
        Self::new(
            SamplingProfileIdentity::new(format!(
                "task404-sd15-prediction-{}-v1",
                prediction as u8
            ))?,
            prediction,
            sigmas,
        )
    }

    pub fn discrete_flow_with_shift(
        shift: f32,
        multiplier: f32,
    ) -> Result<Self, SamplingProfileError> {
        let sigmas = build_flow_sigmas(AURAFLOW_TIMESTEPS, |time| time_snr_shift(shift, time))?;
        Self::new_with_time_mapping(
            SamplingProfileIdentity::new(format!(
                "task404-discrete-flow-{:08x}-{:08x}-v1",
                shift.to_bits(),
                multiplier.to_bits()
            ))?,
            PredictionInterpretation::Flow,
            Arc::from(sigmas),
            SamplingSnrMode::ConstantFlow { shift },
            1.0,
            ModelTimeMapping::DiscreteFlow { shift, multiplier },
        )
    }

    pub fn flux_with_shift(shift: f32) -> Result<Self, SamplingProfileError> {
        let sigmas = build_flow_sigmas(QWEN_IMAGE_TIMESTEPS, |time| flux_time_shift(shift, time))?;
        Self::new_with_time_mapping(
            SamplingProfileIdentity::new(format!("task404-flux-{:08x}-v1", shift.to_bits()))?,
            PredictionInterpretation::Flow,
            Arc::from(sigmas),
            SamplingSnrMode::ConstantFlow { shift: shift.exp() },
            1.0,
            ModelTimeMapping::Flux { shift },
        )
    }

    pub fn lotus_sdpose() -> Result<Self, SamplingProfileError> {
        let sigmas = Self::sd15()?.sigmas;
        Self::new(
            SamplingProfileIdentity::new(LOTUS_SDPOSE_SAMPLING_PROFILE_ID)?,
            PredictionInterpretation::Denoised,
            sigmas,
        )
    }

    pub fn auraflow() -> Result<Self, SamplingProfileError> {
        static SIGMAS: OnceLock<Result<Arc<[f32]>, SamplingProfileError>> = OnceLock::new();
        let sigmas = SIGMAS
            .get_or_init(|| {
                build_flow_sigmas(AURAFLOW_TIMESTEPS, |time| {
                    time_snr_shift(AURAFLOW_SHIFT, time)
                })
                .map(Arc::from)
            })
            .clone()?;
        Self::new_with_time_mapping(
            SamplingProfileIdentity::new(AURAFLOW_SAMPLING_PROFILE_ID)?,
            PredictionInterpretation::Flow,
            sigmas,
            SamplingSnrMode::ConstantFlow {
                shift: AURAFLOW_SHIFT,
            },
            1.0,
            ModelTimeMapping::DiscreteFlow {
                shift: AURAFLOW_SHIFT,
                multiplier: AURAFLOW_MULTIPLIER,
            },
        )
    }

    pub fn qwen_image() -> Result<Self, SamplingProfileError> {
        static SIGMAS: OnceLock<Result<Arc<[f32]>, SamplingProfileError>> = OnceLock::new();
        let sigmas = SIGMAS
            .get_or_init(|| {
                build_flow_sigmas(QWEN_IMAGE_TIMESTEPS, |time| {
                    flux_time_shift(QWEN_IMAGE_SHIFT, time)
                })
                .map(Arc::from)
            })
            .clone()?;
        Self::new_with_time_mapping(
            SamplingProfileIdentity::new(QWEN_IMAGE_SAMPLING_PROFILE_ID)?,
            PredictionInterpretation::Flow,
            sigmas,
            SamplingSnrMode::ConstantFlow {
                shift: QWEN_IMAGE_SHIFT.exp(),
            },
            1.0,
            ModelTimeMapping::Flux {
                shift: QWEN_IMAGE_SHIFT,
            },
        )
    }

    fn new_with_time_mapping(
        identity: SamplingProfileIdentity,
        prediction: PredictionInterpretation,
        sigmas: Arc<[f32]>,
        snr_mode: SamplingSnrMode,
        noise_scale: f32,
        time_mapping: ModelTimeMapping,
    ) -> Result<Self, SamplingProfileError> {
        validate_sigma_grid(&sigmas)?;
        validate_sampling_parameters(snr_mode, noise_scale)?;
        match time_mapping {
            ModelTimeMapping::DiscreteGrid => {}
            ModelTimeMapping::DiscreteFlow { shift, multiplier } => {
                if !shift.is_finite()
                    || shift <= 0.0
                    || !multiplier.is_finite()
                    || multiplier <= 0.0
                {
                    return Err(SamplingProfileError::InvalidModelTimeMapping);
                }
            }
            ModelTimeMapping::Flux { shift } if !shift.is_finite() => {
                return Err(SamplingProfileError::InvalidModelTimeMapping);
            }
            ModelTimeMapping::Flux { .. } => {}
        }
        Ok(Self {
            identity,
            prediction,
            sigmas,
            snr_mode,
            noise_scale,
            time_mapping,
        })
    }

    pub fn is_max_denoise(&self, sigma: f32) -> Result<bool, SamplingProfileError> {
        validate_sigma(sigma, true)?;
        let maximum = self.sigma_max();
        let tolerance = 1.0e-5 * maximum.abs().max(sigma.abs());
        Ok(sigma > maximum || (sigma - maximum).abs() <= tolerance)
    }

    pub fn scale_initial_noise_in_place(
        &self,
        noise: &mut [f32],
        latent: &[f32],
        sigma: f32,
        max_denoise: bool,
    ) -> Result<(), SamplingProfileError> {
        validate_equal_lengths(noise, latent, "initial noise and latent")?;
        validate_sigma(sigma, false)?;
        if self.prediction == PredictionInterpretation::Flow {
            for (index, (noise_value, latent_value)) in
                noise.iter_mut().zip(latent.iter()).enumerate()
            {
                let value =
                    (sigma * self.noise_scale).mul_add(*noise_value, (1.0 - sigma) * *latent_value);
                if !value.is_finite() {
                    return Err(SamplingProfileError::NonFiniteOutput {
                        operation: "initial flow-noise scaling",
                        index,
                    });
                }
                *noise_value = value;
            }
            return Ok(());
        }
        let scale = if max_denoise {
            (1.0 + sigma * sigma).sqrt()
        } else {
            sigma
        };
        for (index, (noise_value, latent_value)) in noise.iter_mut().zip(latent.iter()).enumerate()
        {
            let value = noise_value.mul_add(scale, *latent_value);
            if !value.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "initial noise scaling",
                    index,
                });
            }
            *noise_value = value;
        }
        Ok(())
    }

    pub fn scale_model_input_in_place(
        &self,
        values: &mut [f32],
        sigma: f32,
    ) -> Result<(), SamplingProfileError> {
        validate_sigma(sigma, false)?;
        if self.prediction == PredictionInterpretation::Flow {
            return Ok(());
        }
        let divisor = (sigma * sigma + 1.0).sqrt();
        for (index, value) in values.iter_mut().enumerate() {
            *value /= divisor;
            if !value.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "model input scaling",
                    index,
                });
            }
        }
        Ok(())
    }

    pub fn interpret_prediction_in_place(
        &self,
        model_output: &mut [f32],
        model_input: &[f32],
        sigma: f32,
    ) -> Result<(), SamplingProfileError> {
        validate_equal_lengths(model_output, model_input, "prediction and model input")?;
        validate_sigma(sigma, false)?;
        for (index, (output, input)) in model_output.iter_mut().zip(model_input.iter()).enumerate()
        {
            let denoised = match self.prediction {
                PredictionInterpretation::Epsilon => output.mul_add(-sigma, *input),
                PredictionInterpretation::VPrediction => {
                    let denominator = (sigma * sigma + 1.0).sqrt();
                    *input / (sigma * sigma + 1.0) - *output * sigma / denominator
                }
                PredictionInterpretation::Flow => output.mul_add(-sigma, *input),
                PredictionInterpretation::Denoised => *output,
            };
            if !denoised.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "prediction interpretation",
                    index,
                });
            }
            *output = denoised;
        }
        Ok(())
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativePredictionKind {
    Epsilon,
    VPrediction,
    VPredictionDdpm,
    Edm,
    Constant,
    Denoised,
    ImageToImage,
    ImageToImageFlow,
    CosmosRFlow,
    Lcm,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeLatentTransform {
    Identity,
    Scale { scale: f32 },
    ScaleShift { scale: f32, shift: f32 },
    PlaygroundV25,
    Wan21,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeSamplingPolicyKind {
    Discrete {
        prediction: NativePredictionKind,
        zero_terminal_snr: bool,
    },
    ContinuousEdm {
        prediction: NativePredictionKind,
        sigma_min: f32,
        sigma_max: f32,
        sigma_data: f32,
    },
    ContinuousV {
        sigma_min: f32,
        sigma_max: f32,
        sigma_data: f32,
    },
    Flow {
        shift: f32,
        multiplier: f32,
        noise_scale: f32,
    },
    StableCascade {
        shift: f32,
    },
    Sd3 {
        shift: f32,
        multiplier: f32,
        noise_scale: f32,
    },
    Flux {
        shift: f32,
    },
    Ltxv {
        shift: f32,
    },
    CosmosRFlow {
        sigma_min: f32,
        sigma_max: f32,
        sigma_data: f32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "NativeSamplingPolicyWire")]
pub struct NativeSamplingPolicy {
    identity: SamplingProfileIdentity,
    policy: NativeSamplingPolicyKind,
    latent: NativeLatentTransform,
}

#[derive(Deserialize)]
struct NativeSamplingPolicyWire {
    identity: SamplingProfileIdentity,
    policy: NativeSamplingPolicyKind,
    latent: NativeLatentTransform,
}

impl TryFrom<NativeSamplingPolicyWire> for NativeSamplingPolicy {
    type Error = SamplingProfileError;

    fn try_from(wire: NativeSamplingPolicyWire) -> Result<Self, Self::Error> {
        let expected = Self::checked_with_latent(wire.policy, wire.latent)?;
        if expected.identity != wire.identity {
            return Err(SamplingProfileError::PolicyIdentityChanged);
        }
        Ok(expected)
    }
}

impl NativeSamplingPolicy {
    pub fn checked(policy: NativeSamplingPolicyKind) -> Result<Self, SamplingProfileError> {
        Self::checked_with_latent(policy, default_latent_transform(policy))
    }

    pub fn checked_with_latent(
        policy: NativeSamplingPolicyKind,
        latent: NativeLatentTransform,
    ) -> Result<Self, SamplingProfileError> {
        validate_native_policy(policy)?;
        validate_native_latent_transform(latent)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-sampling-policy.v2");
        hash_native_policy(&mut hasher, policy);
        hash_native_latent_transform(&mut hasher, latent);
        let identity =
            SamplingProfileIdentity::new(format!("native-policy-{:x}", hasher.finalize()))?;
        Ok(Self {
            identity,
            policy,
            latent,
        })
    }

    pub fn sd15() -> Result<Self, SamplingProfileError> {
        Self::checked_with_latent(
            NativeSamplingPolicyKind::Discrete {
                prediction: NativePredictionKind::Epsilon,
                zero_terminal_snr: false,
            },
            NativeLatentTransform::Scale { scale: 0.18215 },
        )
    }

    pub fn policy(&self) -> NativeSamplingPolicyKind {
        self.policy
    }

    pub fn source_sigma_for_sampling_percent(
        &self,
        percent: f64,
        return_actual_sigma: bool,
    ) -> Result<f64, SamplingProfileError> {
        if !percent.is_finite() || !(0.0..=1.0).contains(&percent) {
            return Err(SamplingProfileError::InvalidSamplingPercent(percent as f32));
        }
        if return_actual_sigma {
            if percent == 0.0 {
                return Ok(f64::from(self.sigma_max()));
            }
            if percent == 1.0 {
                return Ok(f64::from(self.sigma_min()));
            }
        }
        if percent == 0.0
            && !matches!(
                self.policy,
                NativeSamplingPolicyKind::Flow { .. }
                    | NativeSamplingPolicyKind::Sd3 { .. }
                    | NativeSamplingPolicyKind::Flux { .. }
                    | NativeSamplingPolicyKind::Ltxv { .. }
            )
        {
            // The source returns this Python float before constructing any float32 tensor.
            return Ok(999_999_999.9);
        }
        if percent > 0.0
            && percent < 1.0
            && matches!(self.policy, NativeSamplingPolicyKind::Discrete { .. })
        {
            // Source percentages address the original 1000-step timeline, including distilled LCM grids.
            return self
                .sigma_for_model_time(((1.0 - percent) * 999.0) as f32)
                .map(f64::from);
        }
        self.sigma_for_sampling_percent(percent as f32)
            .map(f64::from)
    }

    pub fn latent_transform(&self) -> NativeLatentTransform {
        self.latent
    }

    pub fn semantic_digest_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-sampling-policy.semantic.v2");
        hash_native_policy(&mut hasher, self.policy);
        hash_native_latent_transform(&mut hasher, self.latent);
        format!("{:x}", hasher.finalize())
    }

    pub fn validate(&self) -> Result<(), SamplingProfileError> {
        let expected = Self::checked_with_latent(self.policy, self.latent)?;
        if expected.identity != self.identity {
            return Err(SamplingProfileError::PolicyIdentityChanged);
        }
        Ok(())
    }

    pub fn resident_bytes(&self) -> u64 {
        std::mem::size_of::<Self>() as u64 + self.identity.as_str().len() as u64
    }

    pub fn interpret_prediction_for_training(
        &self,
        model_output: &[ScalarAutogradValue],
        model_input: &[f32],
        sigma: f32,
        tape: &mut AutogradTape,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<ScalarAutogradValue>, SamplingProfileError> {
        context.check().map_err(|error| {
            SamplingProfileError::Tensor(NativeDiffusionTensorError::Tensor(error))
        })?;
        if model_output.len() != model_input.len() {
            return Err(SamplingProfileError::LengthMismatch {
                operation: "traced prediction and model input",
                left: model_output.len(),
                right: model_input.len(),
            });
        }
        // Admitted prediction policies are affine in model output. Evaluate the unit
        // response through the ordinary owner so training cannot drift from inference.
        let mut derivative = [1.0];
        self.interpret_prediction_in_place(&mut derivative, &[0.0], sigma)?;
        let required_bytes = model_output
            .len()
            .checked_mul(std::mem::size_of::<ScalarAutogradValue>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SamplingProfileError::OutOfMemory("traced prediction"))?;
        if required_bytes > context.scratch.bytes() {
            return Err(SamplingProfileError::OutOfMemory("traced prediction"));
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(model_output.len())
            .map_err(|_| SamplingProfileError::OutOfMemory("traced prediction"))?;
        for (index, (output, input)) in model_output.iter().zip(model_input).enumerate() {
            if index.is_multiple_of(1024) {
                context.check().map_err(|error| {
                    SamplingProfileError::Tensor(NativeDiffusionTensorError::Tensor(error))
                })?;
            }
            let mut interpreted = [output.value()];
            self.interpret_prediction_in_place(&mut interpreted, &[*input], sigma)?;
            let gradient = tape
                .scalar_scale(*output, derivative[0])
                .map_err(SamplingProfileError::Autograd)?;
            result.push(
                tape.scalar_straight_through(gradient, interpreted[0])
                    .map_err(SamplingProfileError::Autograd)?,
            );
        }
        context.check().map_err(|error| {
            SamplingProfileError::Tensor(NativeDiffusionTensorError::Tensor(error))
        })?;
        Ok(result)
    }

    pub fn process_latent_in_place(&self, values: &mut [f32]) -> Result<(), SamplingProfileError> {
        self.process_latent(values, true, None)
    }

    pub fn process_latent_out_place(&self, values: &mut [f32]) -> Result<(), SamplingProfileError> {
        self.process_latent(values, false, None)
    }

    pub fn process_latent_in(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SamplingProfileError> {
        self.process_latent_tensor(backend, latent, true, context)
    }

    pub fn process_latent_out(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SamplingProfileError> {
        self.process_latent_tensor(backend, latent, false, context)
    }

    pub fn inverse_noise_scaling(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        sigma: f32,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SamplingProfileError> {
        let source = tensor_to_f32(backend, latent, context)?;
        let mut values = backend
            .workspace_vec::<f32>(context, source.len())
            .map_err(|error| {
                SamplingProfileError::Tensor(NativeDiffusionTensorError::Tensor(error))
            })?;
        for value in source.iter().copied() {
            values.try_push(value).map_err(|error| {
                SamplingProfileError::Tensor(NativeDiffusionTensorError::Tensor(error))
            })?;
        }
        self.inverse_noise_scaling_in_place(sigma, &mut values)?;
        Ok(tensor_from_f32(
            backend,
            latent.descriptor().shape(),
            &values,
            context,
        )?)
    }

    fn process_latent_tensor(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        input: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SamplingProfileError> {
        let source = tensor_to_f32(backend, latent, context)?;
        let mut values = backend
            .workspace_vec::<f32>(context, source.len())
            .map_err(|error| {
                SamplingProfileError::Tensor(NativeDiffusionTensorError::Tensor(error))
            })?;
        for value in source.iter().copied() {
            values.try_push(value).map_err(|error| {
                SamplingProfileError::Tensor(NativeDiffusionTensorError::Tensor(error))
            })?;
        }
        self.process_latent(&mut values, input, Some(latent.descriptor().shape()))?;
        Ok(tensor_from_f32(
            backend,
            latent.descriptor().shape(),
            &values,
            context,
        )?)
    }

    fn process_latent(
        &self,
        values: &mut [f32],
        input: bool,
        shape: Option<&[u64]>,
    ) -> Result<(), SamplingProfileError> {
        let transform = self.latent;
        const PLAYGROUND_MEAN: [f32; 4] = [-1.6574, 1.886, -1.383, 2.5155];
        const PLAYGROUND_STD: [f32; 4] = [8.4927, 5.9022, 6.5498, 5.2299];
        let channel_count: Option<usize> = match transform {
            NativeLatentTransform::PlaygroundV25 => Some(4),
            NativeLatentTransform::Wan21 => Some(16),
            _ => None,
        };
        let channel_spatial = if let Some(channel_count) = channel_count {
            match shape {
                Some(shape) => {
                    if shape.len() < 2
                        || shape.get(1).copied()
                            != Some(
                                u64::try_from(channel_count)
                                    .map_err(|_| SamplingProfileError::InvalidLatentShape)?,
                            )
                    {
                        return Err(SamplingProfileError::InvalidLatentShape);
                    }
                    Some(
                        shape
                            .get(2..)
                            .ok_or(SamplingProfileError::InvalidLatentShape)?
                            .iter()
                            .try_fold(1_usize, |elements, dimension| {
                                let dimension = usize::try_from(*dimension)
                                    .map_err(|_| SamplingProfileError::InvalidLatentShape)?;
                                elements
                                    .checked_mul(dimension)
                                    .ok_or(SamplingProfileError::InvalidLatentShape)
                            })?,
                    )
                }
                None => None,
            }
        } else {
            None
        };
        const WAN21_MEAN: [f32; 16] = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
            0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ];
        const WAN21_STD: [f32; 16] = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652,
            1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ];
        for (index, value) in values.iter_mut().enumerate() {
            let channel = channel_spatial
                .map(|spatial| (index / spatial) % channel_count.unwrap_or(1))
                .unwrap_or_else(|| index % channel_count.unwrap_or(1));
            let transformed = match transform {
                NativeLatentTransform::Identity => *value,
                NativeLatentTransform::Scale { scale } if input => *value * scale,
                NativeLatentTransform::Scale { scale } => *value / scale,
                NativeLatentTransform::ScaleShift { scale, shift } if input => {
                    (*value - shift) * scale
                }
                NativeLatentTransform::ScaleShift { scale, shift } => *value / scale + shift,
                NativeLatentTransform::PlaygroundV25 if input => {
                    (*value - PLAYGROUND_MEAN[channel]) * 0.5 / PLAYGROUND_STD[channel]
                }
                NativeLatentTransform::PlaygroundV25 => {
                    *value * PLAYGROUND_STD[channel] / 0.5 + PLAYGROUND_MEAN[channel]
                }
                NativeLatentTransform::Wan21 if input => {
                    (*value - WAN21_MEAN[channel]) / WAN21_STD[channel]
                }
                NativeLatentTransform::Wan21 => *value * WAN21_STD[channel] + WAN21_MEAN[channel],
            };
            if !transformed.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "latent transform",
                    index,
                });
            }
            *value = transformed;
        }
        Ok(())
    }

    fn prediction_kind(&self) -> NativePredictionKind {
        match self.policy {
            NativeSamplingPolicyKind::Discrete { prediction, .. }
            | NativeSamplingPolicyKind::ContinuousEdm { prediction, .. } => prediction,
            NativeSamplingPolicyKind::ContinuousV { .. } => NativePredictionKind::VPrediction,
            NativeSamplingPolicyKind::CosmosRFlow { .. } => NativePredictionKind::CosmosRFlow,
            NativeSamplingPolicyKind::StableCascade { .. } => NativePredictionKind::Epsilon,
            NativeSamplingPolicyKind::Flow { .. }
            | NativeSamplingPolicyKind::Sd3 { .. }
            | NativeSamplingPolicyKind::Flux { .. }
            | NativeSamplingPolicyKind::Ltxv { .. } => NativePredictionKind::Constant,
        }
    }

    fn sigma_data(&self) -> f32 {
        match self.policy {
            NativeSamplingPolicyKind::ContinuousEdm { sigma_data, .. }
            | NativeSamplingPolicyKind::ContinuousV { sigma_data, .. }
            | NativeSamplingPolicyKind::CosmosRFlow { sigma_data, .. } => sigma_data,
            _ => 1.0,
        }
    }
}

impl SamplingProfile for NativeSamplingPolicy {
    fn identity(&self) -> &SamplingProfileIdentity {
        &self.identity
    }

    fn prediction(&self) -> PredictionInterpretation {
        match self.prediction_kind() {
            NativePredictionKind::Epsilon => PredictionInterpretation::Epsilon,
            NativePredictionKind::VPrediction | NativePredictionKind::VPredictionDdpm => {
                PredictionInterpretation::VPrediction
            }
            NativePredictionKind::Denoised
            | NativePredictionKind::ImageToImage
            | NativePredictionKind::ImageToImageFlow
            | NativePredictionKind::Lcm => PredictionInterpretation::Denoised,
            NativePredictionKind::Edm => PredictionInterpretation::VPrediction,
            NativePredictionKind::Constant | NativePredictionKind::CosmosRFlow => {
                PredictionInterpretation::Flow
            }
        }
    }

    fn sigma_count(&self) -> usize {
        match self.policy {
            NativeSamplingPolicyKind::StableCascade { .. }
            | NativeSamplingPolicyKind::Flux { .. } => 10_000,
            NativeSamplingPolicyKind::Discrete {
                prediction: NativePredictionKind::Lcm,
                ..
            } => 50,
            _ => 1_000,
        }
    }

    fn sigma_at_index(&self, index: usize) -> Result<f32, SamplingProfileError> {
        if index >= self.sigma_count() {
            return Err(SamplingProfileError::GridIndex(index));
        }
        match self.policy {
            NativeSamplingPolicyKind::Discrete {
                prediction,
                zero_terminal_snr,
            } => {
                let base = DiscreteSamplingProfile::sd15()?;
                let source_index = if prediction == NativePredictionKind::Lcm {
                    let reverse = self.sigma_count() - 1 - index;
                    SD15_TIMESTEPS - 1 - reverse * (SD15_TIMESTEPS / self.sigma_count())
                } else {
                    index
                };
                let sigma = base.sigma_at_index(source_index)?;
                if zero_terminal_snr {
                    let first = base.sigma_at_index(0)?;
                    let last = base.sigma_at_index(SD15_TIMESTEPS - 1)?;
                    let first_alpha = (first * first + 1.0).sqrt().recip();
                    let last_alpha = (last * last + 1.0).sqrt().recip();
                    let mut alpha = ((sigma * sigma + 1.0).sqrt().recip() - last_alpha)
                        * first_alpha
                        / (first_alpha - last_alpha);
                    if source_index == SD15_TIMESTEPS - 1 {
                        alpha = 4.897_345e-8_f32.sqrt();
                    }
                    Ok(((1.0 - alpha * alpha) / (alpha * alpha)).sqrt())
                } else {
                    Ok(sigma)
                }
            }
            NativeSamplingPolicyKind::ContinuousEdm {
                sigma_min,
                sigma_max,
                ..
            }
            | NativeSamplingPolicyKind::ContinuousV {
                sigma_min,
                sigma_max,
                ..
            }
            | NativeSamplingPolicyKind::CosmosRFlow {
                sigma_min,
                sigma_max,
                ..
            } => {
                let fraction = index as f32 / (self.sigma_count() - 1) as f32;
                Ok((sigma_max.ln() - sigma_min.ln())
                    .mul_add(fraction, sigma_min.ln())
                    .exp())
            }
            NativeSamplingPolicyKind::Flow { shift, .. }
            | NativeSamplingPolicyKind::Sd3 { shift, .. }
            | NativeSamplingPolicyKind::Ltxv { shift } => Ok(time_snr_shift(
                shift,
                (index + 1) as f32 / self.sigma_count() as f32,
            )),
            NativeSamplingPolicyKind::StableCascade { .. } => {
                self.sigma_for_model_time((index + 1) as f32 / self.sigma_count() as f32)
            }
            NativeSamplingPolicyKind::Flux { shift } => Ok(flux_time_shift(
                shift,
                (index + 1) as f32 / self.sigma_count() as f32,
            )),
        }
    }

    fn sigma_at_model_time(&self, model_time: f32) -> Result<f32, SamplingProfileError> {
        if !model_time.is_finite() {
            return Err(SamplingProfileError::InvalidModelTime(model_time));
        }
        let maximum = self.sigma_count().saturating_sub(1) as f32;
        let model_time = model_time.clamp(0.0, maximum);
        let low = model_time.floor() as usize;
        let high = model_time.ceil() as usize;
        let weight = model_time.fract();
        let low_sigma = self.sigma_at_index(low)?;
        let high_sigma = self.sigma_at_index(high)?;
        let sigma = (high_sigma.ln() - low_sigma.ln())
            .mul_add(weight, low_sigma.ln())
            .exp();
        validate_sigma(sigma, true)?;
        Ok(sigma)
    }

    fn model_time_for_sigma(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        validate_sigma(sigma, false)?;
        Ok(match self.policy {
            NativeSamplingPolicyKind::Discrete { prediction, .. } => {
                let log_sigma = sigma.ln();
                let (_, logical_index) = (0..self.sigma_count())
                    .map(|index| {
                        self.sigma_at_index(index)
                            .map(|candidate| ((candidate.ln() - log_sigma).abs(), index))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .ok_or(SamplingProfileError::EmptyGrid)?;
                if prediction == NativePredictionKind::Lcm {
                    (logical_index * (SD15_TIMESTEPS / self.sigma_count())
                        + (SD15_TIMESTEPS / self.sigma_count() - 1)) as f32
                } else {
                    logical_index as f32
                }
            }
            NativeSamplingPolicyKind::ContinuousEdm { .. } => 0.25 * sigma.ln(),
            NativeSamplingPolicyKind::ContinuousV { .. } => {
                sigma.atan() / std::f32::consts::PI * 2.0
            }
            NativeSamplingPolicyKind::Flow { multiplier, .. }
            | NativeSamplingPolicyKind::Sd3 { multiplier, .. } => sigma * multiplier,
            NativeSamplingPolicyKind::StableCascade { .. } => stable_cascade_time_for_sigma(sigma)?,
            NativeSamplingPolicyKind::Flux { .. } => sigma,
            NativeSamplingPolicyKind::Ltxv { .. } => sigma,
            NativeSamplingPolicyKind::CosmosRFlow { .. } => sigma / (sigma + 1.0),
        })
    }

    fn sigma_for_model_time(&self, model_time: f32) -> Result<f32, SamplingProfileError> {
        if !model_time.is_finite() {
            return Err(SamplingProfileError::InvalidModelTime(model_time));
        }
        let sigma = match self.policy {
            NativeSamplingPolicyKind::Discrete { prediction, .. } => {
                if model_time < 0.0 {
                    return Err(SamplingProfileError::InvalidModelTime(model_time));
                }
                let logical_time = if prediction == NativePredictionKind::Lcm {
                    let stride = (SD15_TIMESTEPS / self.sigma_count()) as f32;
                    ((model_time - (stride - 1.0)) / stride)
                        .clamp(0.0, self.sigma_count().saturating_sub(1) as f32)
                } else {
                    model_time
                };
                return self.sigma_at_model_time(logical_time);
            }
            NativeSamplingPolicyKind::ContinuousEdm { .. } => (model_time / 0.25).exp(),
            NativeSamplingPolicyKind::ContinuousV { .. } => {
                (model_time * std::f32::consts::PI / 2.0).tan()
            }
            NativeSamplingPolicyKind::Flow {
                shift, multiplier, ..
            }
            | NativeSamplingPolicyKind::Sd3 {
                shift, multiplier, ..
            } => {
                if model_time < 0.0 {
                    return Err(SamplingProfileError::InvalidModelTime(model_time));
                }
                time_snr_shift(shift, model_time / multiplier)
            }
            NativeSamplingPolicyKind::StableCascade { shift } => {
                stable_cascade_sigma(model_time, shift)
            }
            NativeSamplingPolicyKind::Flux { shift } => {
                if model_time <= 0.0 {
                    return Err(SamplingProfileError::InvalidModelTime(model_time));
                }
                flux_time_shift(shift, model_time)
            }
            NativeSamplingPolicyKind::Ltxv { shift } => {
                if model_time < 0.0 {
                    return Err(SamplingProfileError::InvalidModelTime(model_time));
                }
                time_snr_shift(shift, model_time)
            }
            NativeSamplingPolicyKind::CosmosRFlow { sigma_max, .. } => {
                let maximum_time = sigma_max / (sigma_max + 1.0);
                if model_time >= maximum_time {
                    sigma_max
                } else {
                    model_time / (1.0 - model_time)
                }
            }
        };
        validate_sigma(sigma, true)?;
        Ok(sigma)
    }

    fn sampling_percent_for_sigma(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        if sigma == 0.0 {
            return Ok(1.0);
        }
        validate_sigma(sigma, false)?;
        Ok(match self.policy {
            NativeSamplingPolicyKind::Discrete { prediction, .. } => {
                let model_time = self.model_time_for_sigma(sigma)?;
                let logical_time = if prediction == NativePredictionKind::Lcm {
                    let stride = (SD15_TIMESTEPS / self.sigma_count()) as f32;
                    (model_time - (stride - 1.0)) / stride
                } else {
                    model_time
                };
                1.0 - logical_time / self.sigma_count().saturating_sub(1) as f32
            }
            NativeSamplingPolicyKind::ContinuousEdm {
                sigma_min,
                sigma_max,
                ..
            }
            | NativeSamplingPolicyKind::ContinuousV {
                sigma_min,
                sigma_max,
                ..
            }
            | NativeSamplingPolicyKind::CosmosRFlow {
                sigma_min,
                sigma_max,
                ..
            } => 1.0 - (sigma.ln() - sigma_min.ln()) / (sigma_max.ln() - sigma_min.ln()),
            NativeSamplingPolicyKind::Flow { shift, .. }
            | NativeSamplingPolicyKind::Sd3 { shift, .. }
            | NativeSamplingPolicyKind::Ltxv { shift } => shifted_sigma_to_percent(shift, sigma)?,
            NativeSamplingPolicyKind::Flux { shift } => {
                shifted_sigma_to_percent(shift.exp(), sigma)?
            }
            NativeSamplingPolicyKind::StableCascade { shift } => {
                1.0 - stable_cascade_shifted_time_for_sigma(sigma, shift)?
            }
        }
        .clamp(0.0, 1.0))
    }

    fn sigma_for_sampling_percent(&self, percent: f32) -> Result<f32, SamplingProfileError> {
        validate_sampling_percent(percent)?;
        if percent == 0.0 {
            return Ok(match self.policy {
                NativeSamplingPolicyKind::Flow { .. }
                | NativeSamplingPolicyKind::Sd3 { .. }
                | NativeSamplingPolicyKind::Flux { .. }
                | NativeSamplingPolicyKind::Ltxv { .. } => 1.0,
                _ => 999_999_999.9,
            });
        }
        if percent == 1.0 {
            return Ok(0.0);
        }
        Ok(match self.policy {
            NativeSamplingPolicyKind::Discrete { .. } => self.sigma_at_model_time(
                (1.0 - percent) * self.sigma_count().saturating_sub(1) as f32,
            )?,
            NativeSamplingPolicyKind::ContinuousEdm {
                sigma_min,
                sigma_max,
                ..
            }
            | NativeSamplingPolicyKind::ContinuousV {
                sigma_min,
                sigma_max,
                ..
            }
            | NativeSamplingPolicyKind::CosmosRFlow {
                sigma_min,
                sigma_max,
                ..
            } => (sigma_max.ln() - sigma_min.ln())
                .mul_add(1.0 - percent, sigma_min.ln())
                .exp(),
            NativeSamplingPolicyKind::Flow { shift, .. }
            | NativeSamplingPolicyKind::Sd3 { shift, .. }
            | NativeSamplingPolicyKind::Ltxv { shift } => time_snr_shift(shift, 1.0 - percent),
            NativeSamplingPolicyKind::StableCascade { shift } => {
                stable_cascade_sigma(1.0 - percent, shift)
            }
            NativeSamplingPolicyKind::Flux { shift } => flux_time_shift(shift, 1.0 - percent),
        })
    }

    fn sigma_min(&self) -> f32 {
        match self.policy {
            NativeSamplingPolicyKind::ContinuousEdm { sigma_min, .. }
            | NativeSamplingPolicyKind::ContinuousV { sigma_min, .. }
            | NativeSamplingPolicyKind::CosmosRFlow { sigma_min, .. } => sigma_min,
            _ => self.sigma_at_index(0).unwrap_or(f32::MIN_POSITIVE),
        }
    }

    fn sigma_max(&self) -> f32 {
        match self.policy {
            NativeSamplingPolicyKind::ContinuousEdm { sigma_max, .. }
            | NativeSamplingPolicyKind::ContinuousV { sigma_max, .. }
            | NativeSamplingPolicyKind::CosmosRFlow { sigma_max, .. } => sigma_max,
            _ => self
                .sigma_at_index(self.sigma_count().saturating_sub(1))
                .unwrap_or(1.0),
        }
    }

    fn half_log_snr(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        validate_sigma(sigma, false)?;
        match self.prediction() {
            PredictionInterpretation::Flow if sigma < 1.0 => Ok((1.0 - sigma).ln() - sigma.ln()),
            PredictionInterpretation::Flow => Err(SamplingProfileError::InvalidSnrSigma(sigma)),
            _ => Ok(-sigma.ln()),
        }
    }

    fn sigma_from_half_log_snr(&self, half_log_snr: f32) -> Result<f32, SamplingProfileError> {
        if !half_log_snr.is_finite() {
            return Err(SamplingProfileError::InvalidHalfLogSnr(half_log_snr));
        }
        Ok(match self.prediction() {
            PredictionInterpretation::Flow => 1.0 / (1.0 + half_log_snr.exp()),
            _ => (-half_log_snr).exp(),
        })
    }

    fn adjust_first_sigma_for_snr(&self, sigmas: &mut [f32]) -> Result<(), SamplingProfileError> {
        for sigma in sigmas.iter().copied() {
            validate_sigma(sigma, true)?;
        }
        if self.prediction() == PredictionInterpretation::Flow
            && sigmas.len() > 1
            && sigmas[0] >= 1.0
        {
            sigmas[0] = self.sigma_at_index(self.sigma_count().saturating_sub(2))?;
        }
        Ok(())
    }

    fn scale_sampler_noise(&self, sampler_noise_scale: f32) -> Result<f32, SamplingProfileError> {
        if !sampler_noise_scale.is_finite() {
            return Err(SamplingProfileError::InvalidNoiseScale(sampler_noise_scale));
        }
        let scale = match self.policy {
            NativeSamplingPolicyKind::Flow { noise_scale, .. }
            | NativeSamplingPolicyKind::Sd3 { noise_scale, .. } => noise_scale,
            _ => 1.0,
        };
        Ok(sampler_noise_scale * scale)
    }

    fn scale_initial_noise_in_place(
        &self,
        noise: &mut [f32],
        latent: &[f32],
        sigma: f32,
        max_denoise: bool,
    ) -> Result<(), SamplingProfileError> {
        validate_equal_lengths(noise, latent, "initial noise and latent")?;
        // TrainSampler evaluates this same operation at sigma zero for the clean target.
        validate_sigma(sigma, true)?;
        for (index, (noise_value, latent_value)) in noise.iter_mut().zip(latent).enumerate() {
            let value = match self.prediction_kind() {
                NativePredictionKind::ImageToImageFlow => *latent_value,
                NativePredictionKind::Constant => (sigma * self.scale_sampler_noise(1.0)?)
                    .mul_add(*noise_value, (1.0 - sigma) * *latent_value),
                NativePredictionKind::CosmosRFlow => sigma.mul_add(*noise_value, *latent_value),
                _ => {
                    let scale = if max_denoise {
                        (1.0 + sigma * sigma).sqrt()
                    } else {
                        sigma
                    };
                    scale.mul_add(*noise_value, *latent_value)
                }
            };
            if !value.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "initial noise scaling",
                    index,
                });
            }
            *noise_value = value;
        }
        Ok(())
    }

    fn scale_model_input_in_place(
        &self,
        values: &mut [f32],
        sigma: f32,
    ) -> Result<(), SamplingProfileError> {
        validate_sigma(sigma, false)?;
        let prediction = self.prediction_kind();
        let divisor = (sigma * sigma + self.sigma_data() * self.sigma_data()).sqrt();
        for (index, value) in values.iter_mut().enumerate() {
            *value = match prediction {
                NativePredictionKind::Constant
                | NativePredictionKind::ImageToImage
                | NativePredictionKind::ImageToImageFlow
                | NativePredictionKind::VPredictionDdpm => *value,
                NativePredictionKind::CosmosRFlow => *value * (1.0 - sigma / (sigma + 1.0)),
                // The source division rounds differently from multiplication by a reciprocal.
                _ => *value / divisor,
            };
            if !value.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "model input scaling",
                    index,
                });
            }
        }
        Ok(())
    }

    fn inverse_noise_scaling_in_place(
        &self,
        sigma: f32,
        latent: &mut [f32],
    ) -> Result<(), SamplingProfileError> {
        validate_sigma(sigma, true)?;
        for (index, value) in latent.iter_mut().enumerate() {
            *value = match self.prediction_kind() {
                NativePredictionKind::ImageToImageFlow => 1.0 - *value,
                NativePredictionKind::Constant => *value / (1.0 - sigma),
                _ => *value,
            };
            if !value.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "inverse noise scaling",
                    index,
                });
            }
        }
        Ok(())
    }

    fn interpret_prediction_in_place(
        &self,
        model_output: &mut [f32],
        model_input: &[f32],
        sigma: f32,
    ) -> Result<(), SamplingProfileError> {
        validate_equal_lengths(model_output, model_input, "prediction and model input")?;
        validate_sigma(sigma, false)?;
        let sigma_data = self.sigma_data();
        let model_time = self.model_time_for_sigma(sigma)?;
        for (index, (output, input)) in model_output.iter_mut().zip(model_input).enumerate() {
            let value = match self.prediction_kind() {
                NativePredictionKind::Epsilon | NativePredictionKind::Constant => {
                    output.mul_add(-sigma, *input)
                }
                NativePredictionKind::VPrediction => {
                    let denominator = (sigma * sigma + sigma_data * sigma_data).sqrt();
                    *input * sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data)
                        - *output * sigma * sigma_data / denominator
                }
                NativePredictionKind::VPredictionDdpm => {
                    (*input - *output * sigma) / (sigma * sigma + 1.0).sqrt()
                }
                NativePredictionKind::Edm => {
                    let denominator = (sigma * sigma + sigma_data * sigma_data).sqrt();
                    *input * sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data)
                        + *output * sigma * sigma_data / denominator
                }
                NativePredictionKind::Denoised
                | NativePredictionKind::ImageToImage
                | NativePredictionKind::ImageToImageFlow => *output,
                NativePredictionKind::CosmosRFlow => {
                    let normalized = sigma / (sigma + 1.0);
                    *input * (1.0 - normalized) - *output * normalized
                }
                NativePredictionKind::Lcm => {
                    let x0 = output.mul_add(-sigma, *input);
                    let scaled_time = model_time * 10.0;
                    let c_skip = 0.25 / (scaled_time * scaled_time + 0.25);
                    let c_out = scaled_time / (scaled_time * scaled_time + 0.25).sqrt();
                    c_out.mul_add(x0, c_skip * *input)
                }
            };
            if !value.is_finite() {
                return Err(SamplingProfileError::NonFiniteOutput {
                    operation: "prediction interpretation",
                    index,
                });
            }
            *output = value;
        }
        Ok(())
    }
}

fn validate_native_policy(policy: NativeSamplingPolicyKind) -> Result<(), SamplingProfileError> {
    let validate_bounds = |sigma_min: f32, sigma_max: f32, sigma_data: f32| {
        if !sigma_min.is_finite()
            || !sigma_max.is_finite()
            || !sigma_data.is_finite()
            || sigma_min <= 0.0
            || sigma_max <= sigma_min
            || sigma_data <= 0.0
        {
            Err(SamplingProfileError::InvalidPolicyParameters)
        } else {
            Ok(())
        }
    };
    match policy {
        NativeSamplingPolicyKind::Discrete { prediction, .. } => {
            if matches!(
                prediction,
                NativePredictionKind::Epsilon
                    | NativePredictionKind::VPrediction
                    | NativePredictionKind::VPredictionDdpm
                    | NativePredictionKind::Denoised
                    | NativePredictionKind::ImageToImage
                    | NativePredictionKind::ImageToImageFlow
                    | NativePredictionKind::Lcm
            ) {
                Ok(())
            } else {
                Err(SamplingProfileError::InvalidPolicyParameters)
            }
        }
        NativeSamplingPolicyKind::ContinuousEdm {
            prediction,
            sigma_min,
            sigma_max,
            sigma_data,
        } => {
            validate_bounds(sigma_min, sigma_max, sigma_data)?;
            if !matches!(
                prediction,
                NativePredictionKind::Epsilon
                    | NativePredictionKind::VPrediction
                    | NativePredictionKind::Edm
            ) {
                return Err(SamplingProfileError::InvalidPolicyParameters);
            }
            Ok(())
        }
        NativeSamplingPolicyKind::ContinuousV {
            sigma_min,
            sigma_max,
            sigma_data,
        }
        | NativeSamplingPolicyKind::CosmosRFlow {
            sigma_min,
            sigma_max,
            sigma_data,
        } => validate_bounds(sigma_min, sigma_max, sigma_data),
        NativeSamplingPolicyKind::Flow {
            shift,
            multiplier,
            noise_scale,
        }
        | NativeSamplingPolicyKind::Sd3 {
            shift,
            multiplier,
            noise_scale,
        } => {
            if shift.is_finite()
                && shift > 0.0
                && multiplier.is_finite()
                && multiplier > 0.0
                && noise_scale.is_finite()
                && noise_scale >= 0.0
            {
                Ok(())
            } else {
                Err(SamplingProfileError::InvalidPolicyParameters)
            }
        }
        NativeSamplingPolicyKind::StableCascade { shift }
        | NativeSamplingPolicyKind::Flux { shift }
        | NativeSamplingPolicyKind::Ltxv { shift } => {
            if shift.is_finite() && shift > 0.0 {
                Ok(())
            } else {
                Err(SamplingProfileError::InvalidPolicyParameters)
            }
        }
    }
}

fn validate_native_latent_transform(
    transform: NativeLatentTransform,
) -> Result<(), SamplingProfileError> {
    let valid = match transform {
        NativeLatentTransform::Identity
        | NativeLatentTransform::PlaygroundV25
        | NativeLatentTransform::Wan21 => true,
        NativeLatentTransform::Scale { scale } => scale.is_finite() && scale != 0.0,
        NativeLatentTransform::ScaleShift { scale, shift } => {
            scale.is_finite() && scale != 0.0 && shift.is_finite()
        }
    };
    if valid {
        Ok(())
    } else {
        Err(SamplingProfileError::InvalidPolicyParameters)
    }
}

fn default_latent_transform(policy: NativeSamplingPolicyKind) -> NativeLatentTransform {
    match policy {
        NativeSamplingPolicyKind::Sd3 { .. } => NativeLatentTransform::ScaleShift {
            scale: 1.5305,
            shift: 0.0609,
        },
        NativeSamplingPolicyKind::Flux { .. } => NativeLatentTransform::ScaleShift {
            scale: 0.3611,
            shift: 0.1159,
        },
        _ => NativeLatentTransform::Identity,
    }
}

fn hash_native_latent_transform(hasher: &mut Sha256, latent: NativeLatentTransform) {
    match latent {
        NativeLatentTransform::Identity => hasher.update([0]),
        NativeLatentTransform::Scale { scale } => {
            hasher.update([1]);
            hasher.update(scale.to_bits().to_le_bytes());
        }
        NativeLatentTransform::ScaleShift { scale, shift } => {
            hasher.update([2]);
            hasher.update(scale.to_bits().to_le_bytes());
            hasher.update(shift.to_bits().to_le_bytes());
        }
        NativeLatentTransform::PlaygroundV25 => hasher.update([3]),
        NativeLatentTransform::Wan21 => hasher.update([4]),
    }
}

fn hash_native_policy(hasher: &mut Sha256, policy: NativeSamplingPolicyKind) {
    fn hash_float(hasher: &mut Sha256, value: f32) {
        hasher.update(value.to_bits().to_le_bytes());
    }
    match policy {
        NativeSamplingPolicyKind::Discrete {
            prediction,
            zero_terminal_snr,
        } => {
            hasher.update([0, prediction as u8, u8::from(zero_terminal_snr)]);
            if prediction == NativePredictionKind::Denoised {
                hasher.update(b"x0-eps-input-scaling-v1");
            }
        }
        NativeSamplingPolicyKind::ContinuousEdm {
            prediction,
            sigma_min,
            sigma_max,
            sigma_data,
        } => {
            hasher.update([1, prediction as u8]);
            hash_float(hasher, sigma_min);
            hash_float(hasher, sigma_max);
            hash_float(hasher, sigma_data);
        }
        NativeSamplingPolicyKind::ContinuousV {
            sigma_min,
            sigma_max,
            sigma_data,
        } => {
            hasher.update([2]);
            hash_float(hasher, sigma_min);
            hash_float(hasher, sigma_max);
            hash_float(hasher, sigma_data);
        }
        NativeSamplingPolicyKind::Flow {
            shift,
            multiplier,
            noise_scale,
        } => {
            hasher.update([3]);
            hash_float(hasher, shift);
            hash_float(hasher, multiplier);
            hash_float(hasher, noise_scale);
        }
        NativeSamplingPolicyKind::StableCascade { shift } => {
            hasher.update([4]);
            hash_float(hasher, shift);
        }
        NativeSamplingPolicyKind::Sd3 {
            shift,
            multiplier,
            noise_scale,
        } => {
            hasher.update([5]);
            hash_float(hasher, shift);
            hash_float(hasher, multiplier);
            hash_float(hasher, noise_scale);
        }
        NativeSamplingPolicyKind::Flux { shift } => {
            hasher.update([6]);
            hash_float(hasher, shift);
        }
        NativeSamplingPolicyKind::Ltxv { shift } => {
            hasher.update([7]);
            hash_float(hasher, shift);
        }
        NativeSamplingPolicyKind::CosmosRFlow {
            sigma_min,
            sigma_max,
            sigma_data,
        } => {
            hasher.update([8]);
            hash_float(hasher, sigma_min);
            hash_float(hasher, sigma_max);
            hash_float(hasher, sigma_data);
        }
    }
}

fn stable_cascade_sigma(time: f32, shift: f32) -> f32 {
    let cosine_s = 8.0e-3_f32;
    let initial = (cosine_s / (1.0 + cosine_s) * std::f32::consts::FRAC_PI_2)
        .cos()
        .powi(2);
    let mut alpha = ((time + cosine_s) / (1.0 + cosine_s) * std::f32::consts::FRAC_PI_2)
        .cos()
        .powi(2)
        / initial;
    if shift != 1.0 {
        let log_snr = (alpha / (1.0 - alpha)).ln() + 2.0 * (1.0 / shift).ln();
        alpha = 1.0 / (1.0 + (-log_snr).exp());
    }
    alpha = alpha.clamp(0.0001, 0.9999);
    ((1.0 - alpha) / alpha).sqrt()
}

fn stable_cascade_time_for_sigma(sigma: f32) -> Result<f32, SamplingProfileError> {
    validate_sigma(sigma, false)?;
    let cosine_s = 8.0e-3_f32;
    let initial = (cosine_s / (1.0 + cosine_s) * std::f32::consts::FRAC_PI_2)
        .cos()
        .powi(2);
    let variance = (1.0 / (sigma * sigma + 1.0)).clamp(0.0, 1.0);
    let time = (variance * initial).sqrt().acos() / std::f32::consts::FRAC_PI_2 * (1.0 + cosine_s)
        - cosine_s;
    if time.is_finite() {
        Ok(time)
    } else {
        Err(SamplingProfileError::InvalidSigma(sigma))
    }
}

fn stable_cascade_shifted_time_for_sigma(
    sigma: f32,
    shift: f32,
) -> Result<f32, SamplingProfileError> {
    validate_sigma(sigma, false)?;
    let mut low = 0.0_f32;
    let mut high = 1.0_f32;
    for _ in 0..32 {
        let middle = (low + high) * 0.5;
        if stable_cascade_sigma(middle, shift) < sigma {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok((low + high) * 0.5)
}

pub fn profile_for_model(
    model: &NativeModelPayload,
) -> Result<NativeSamplingPolicy, SamplingProfileError> {
    if model.identity().role() != NativeModelResourceRole::Model || model.model().is_none() {
        return Err(SamplingProfileError::UnsupportedModel(
            model.identity().identifier().to_owned(),
        ));
    }
    let Some(resource) = model.native_family_model_resource() else {
        return NativeSamplingPolicy::sd15();
    };
    let family = resource
        .family_identity()
        .map_err(|error| SamplingProfileError::UnsupportedModel(error.to_string()))?;
    let feature_id = family.feature_id();
    let policy = match feature_id {
        sim_model::generated_auraflow_sim_model_0064::MODEL_FAMILY_FEATURE_ID => {
            NativeSamplingPolicyKind::Flow {
                shift: AURAFLOW_SHIFT,
                multiplier: AURAFLOW_MULTIPLIER,
                noise_scale: 1.0,
            }
        }
        sim_model::generated_qwenimage_sim_model_0113::MODEL_FAMILY_FEATURE_ID => {
            NativeSamplingPolicyKind::Flux {
                shift: QWEN_IMAGE_SHIFT,
            }
        }
        _ if feature_id.contains("ltx") => NativeSamplingPolicyKind::Ltxv { shift: 2.37 },
        _ if feature_id.contains("cosmos") => NativeSamplingPolicyKind::CosmosRFlow {
            sigma_min: 0.002,
            sigma_max: 120.0,
            sigma_data: 1.0,
        },
        _ if feature_id.contains("stable_cascade") || feature_id.contains("stable-cascade") => {
            NativeSamplingPolicyKind::StableCascade { shift: 1.0 }
        }
        _ if feature_id.contains("sd3") => NativeSamplingPolicyKind::Sd3 {
            shift: 3.0,
            multiplier: 1_000.0,
            noise_scale: 1.0,
        },
        _ if feature_id.contains("flux") => NativeSamplingPolicyKind::Flux { shift: 1.15 },
        _ => NativeSamplingPolicyKind::Discrete {
            prediction: NativePredictionKind::Epsilon,
            zero_terminal_snr: false,
        },
    };
    let latent = latent_transform_for_identifier(resource.model().profile().latent_identifier);
    NativeSamplingPolicy::checked_with_latent(policy, latent)
}

fn latent_transform_for_identifier(identifier: &str) -> NativeLatentTransform {
    match identifier {
        "SD15" => NativeLatentTransform::Scale { scale: 0.18215 },
        "SDXL" => NativeLatentTransform::Scale { scale: 0.13025 },
        "SDXL_Playground_2_5" => NativeLatentTransform::PlaygroundV25,
        "SD3" => NativeLatentTransform::ScaleShift {
            scale: 1.5305,
            shift: 0.0609,
        },
        "Flux" => NativeLatentTransform::ScaleShift {
            scale: 0.3611,
            shift: 0.1159,
        },
        "Wan21" => NativeLatentTransform::Wan21,
        "SC_B" => NativeLatentTransform::Scale { scale: 1.0 / 0.43 },
        "SC_Prior" | "LTXV" | "Cosmos1CV8x8x8" => NativeLatentTransform::Identity,
        _ => NativeLatentTransform::Identity,
    }
}

impl SamplingProfile for DiscreteSamplingProfile {
    fn identity(&self) -> &SamplingProfileIdentity {
        &self.identity
    }

    fn prediction(&self) -> PredictionInterpretation {
        self.prediction
    }

    fn sigma_count(&self) -> usize {
        self.sigmas.len()
    }

    fn sigma_at_index(&self, index: usize) -> Result<f32, SamplingProfileError> {
        self.sigmas
            .get(index)
            .copied()
            .ok_or(SamplingProfileError::GridIndex(index))
    }

    fn sigma_at_model_time(&self, model_time: f32) -> Result<f32, SamplingProfileError> {
        if !model_time.is_finite() {
            return Err(SamplingProfileError::InvalidModelTime(model_time));
        }
        let maximum = (self.sigmas.len() - 1) as f32;
        let model_time = model_time.clamp(0.0, maximum);
        let low = model_time.floor() as usize;
        let high = model_time.ceil() as usize;
        let weight = model_time.fract();
        let low_sigma = *self
            .sigmas
            .get(low)
            .ok_or(SamplingProfileError::GridIndex(low))?;
        let high_sigma = *self
            .sigmas
            .get(high)
            .ok_or(SamplingProfileError::GridIndex(high))?;
        let log_sigma = (high_sigma.ln() - low_sigma.ln()).mul_add(weight, low_sigma.ln());
        let sigma = log_sigma.exp();
        validate_sigma(sigma, true)?;
        Ok(sigma)
    }

    fn model_time_for_sigma(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        if sigma == 0.0 {
            return Ok(0.0);
        }
        validate_sigma(sigma, false)?;
        match self.time_mapping {
            ModelTimeMapping::DiscreteGrid => {
                let log_sigma = sigma.ln();
                let (_, index) = self
                    .sigmas
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| ((candidate.ln() - log_sigma).abs(), index))
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .ok_or(SamplingProfileError::EmptyGrid)?;
                Ok(index as f32)
            }
            ModelTimeMapping::DiscreteFlow { multiplier, .. } => Ok(sigma * multiplier),
            ModelTimeMapping::Flux { .. } => Ok(sigma),
        }
    }

    fn sigma_for_model_time(&self, model_time: f32) -> Result<f32, SamplingProfileError> {
        if !model_time.is_finite() || model_time < 0.0 {
            return Err(SamplingProfileError::InvalidModelTime(model_time));
        }
        let sigma = match self.time_mapping {
            ModelTimeMapping::DiscreteGrid => return self.sigma_at_model_time(model_time),
            ModelTimeMapping::DiscreteFlow { shift, multiplier } => {
                time_snr_shift(shift, model_time / multiplier)
            }
            ModelTimeMapping::Flux { shift } => flux_time_shift(shift, model_time),
        };
        validate_sigma(sigma, true)?;
        Ok(sigma)
    }

    fn sampling_percent_for_sigma(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        if sigma == 0.0 {
            return Ok(1.0);
        }
        validate_sigma(sigma, false)?;
        match self.time_mapping {
            ModelTimeMapping::DiscreteGrid => {
                let maximum = (self.sigmas.len() - 1) as f32;
                Ok((1.0 - self.model_time_for_sigma(sigma)? / maximum).clamp(0.0, 1.0))
            }
            ModelTimeMapping::DiscreteFlow { shift, .. } => shifted_sigma_to_percent(shift, sigma),
            ModelTimeMapping::Flux { shift } => shifted_sigma_to_percent(shift.exp(), sigma),
        }
    }

    fn sigma_min(&self) -> f32 {
        self.sigmas[0]
    }

    fn sigma_max(&self) -> f32 {
        self.sigmas[self.sigmas.len() - 1]
    }

    fn half_log_snr(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        validate_sigma(sigma, false)?;
        let half_log_snr = match self.snr_mode {
            SamplingSnrMode::Standard => -sigma.ln(),
            SamplingSnrMode::ConstantFlow { .. } => {
                if sigma >= 1.0 {
                    return Err(SamplingProfileError::InvalidSnrSigma(sigma));
                }
                (1.0 - sigma).ln() - sigma.ln()
            }
        };
        if !half_log_snr.is_finite() {
            return Err(SamplingProfileError::InvalidHalfLogSnr(half_log_snr));
        }
        Ok(half_log_snr)
    }

    fn sigma_from_half_log_snr(&self, half_log_snr: f32) -> Result<f32, SamplingProfileError> {
        if !half_log_snr.is_finite() {
            return Err(SamplingProfileError::InvalidHalfLogSnr(half_log_snr));
        }
        let sigma = match self.snr_mode {
            SamplingSnrMode::Standard => (-half_log_snr).exp(),
            SamplingSnrMode::ConstantFlow { .. } if half_log_snr >= 0.0 => {
                let exponential = (-half_log_snr).exp();
                exponential / (1.0 + exponential)
            }
            SamplingSnrMode::ConstantFlow { .. } => 1.0 / (1.0 + half_log_snr.exp()),
        };
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(SamplingProfileError::InvalidSnrSigma(sigma));
        }
        Ok(sigma)
    }

    fn adjust_first_sigma_for_snr(&self, sigmas: &mut [f32]) -> Result<(), SamplingProfileError> {
        for sigma in sigmas.iter().copied() {
            validate_sigma(sigma, true)?;
        }
        if sigmas.len() <= 1 {
            return Ok(());
        }
        if let SamplingSnrMode::ConstantFlow { shift } = self.snr_mode {
            if sigmas[0] >= 1.0 {
                sigmas[0] = constant_flow_percent_to_sigma(shift, 1.0e-4)?;
            }
        }
        Ok(())
    }

    fn scale_sampler_noise(&self, sampler_noise_scale: f32) -> Result<f32, SamplingProfileError> {
        if !sampler_noise_scale.is_finite() {
            return Err(SamplingProfileError::InvalidNoiseScale(sampler_noise_scale));
        }
        let scaled = sampler_noise_scale * self.noise_scale;
        if !scaled.is_finite() {
            return Err(SamplingProfileError::InvalidNoiseScale(scaled));
        }
        Ok(scaled)
    }
}

fn validate_sampling_parameters(
    snr_mode: SamplingSnrMode,
    noise_scale: f32,
) -> Result<(), SamplingProfileError> {
    if let SamplingSnrMode::ConstantFlow { shift } = snr_mode {
        if !shift.is_finite() || shift <= 0.0 {
            return Err(SamplingProfileError::InvalidSnrShift(shift));
        }
    }
    if !noise_scale.is_finite() || noise_scale < 0.0 {
        return Err(SamplingProfileError::InvalidNoiseScale(noise_scale));
    }
    Ok(())
}

fn constant_flow_percent_to_sigma(shift: f32, percent: f32) -> Result<f32, SamplingProfileError> {
    validate_sampling_percent(percent)?;
    if percent <= 0.0 {
        return Ok(1.0);
    }
    if percent >= 1.0 {
        return Ok(0.0);
    }
    let time = 1.0 - percent;
    let sigma = shift * time / (1.0 + (shift - 1.0) * time);
    if !sigma.is_finite() || !(0.0..1.0).contains(&sigma) {
        return Err(SamplingProfileError::InvalidSnrSigma(sigma));
    }
    Ok(sigma)
}

fn validate_sampling_percent(percent: f32) -> Result<(), SamplingProfileError> {
    if percent.is_finite() && (0.0..=1.0).contains(&percent) {
        Ok(())
    } else {
        Err(SamplingProfileError::InvalidSamplingPercent(percent))
    }
}

fn shifted_sigma_to_percent(shift: f32, sigma: f32) -> Result<f32, SamplingProfileError> {
    if sigma >= 1.0 {
        return Ok(0.0);
    }
    let denominator = shift - sigma * (shift - 1.0);
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(SamplingProfileError::InvalidSnrSigma(sigma));
    }
    let percent = 1.0 - sigma / denominator;
    if !percent.is_finite() || !(0.0..=1.0).contains(&percent) {
        return Err(SamplingProfileError::InvalidSamplingPercent(percent));
    }
    Ok(percent)
}

fn time_snr_shift(shift: f32, time: f32) -> f32 {
    shift * time / (1.0 + (shift - 1.0) * time)
}

fn flux_time_shift(shift: f32, time: f32) -> f32 {
    let exponential = shift.exp();
    exponential / (exponential + (time.recip() - 1.0))
}

fn build_flow_sigmas(
    timesteps: usize,
    sigma: impl Fn(f32) -> f32,
) -> Result<Vec<f32>, SamplingProfileError> {
    let mut sigmas = Vec::new();
    sigmas
        .try_reserve_exact(timesteps)
        .map_err(|_| SamplingProfileError::OutOfMemory("flow sigma grid"))?;
    for index in 1..=timesteps {
        sigmas.push(sigma(index as f32 / timesteps as f32));
    }
    validate_sigma_grid(&sigmas)?;
    Ok(sigmas)
}

fn build_sd15_sigmas() -> Result<Vec<f32>, SamplingProfileError> {
    let start = SD15_LINEAR_START.sqrt();
    let end = SD15_LINEAR_END.sqrt();
    let mut cumulative_alpha = 1.0_f64;
    let mut sigmas = Vec::new();
    sigmas
        .try_reserve_exact(SD15_TIMESTEPS)
        .map_err(|_| SamplingProfileError::OutOfMemory("SD15 sigma grid"))?;
    for index in 0..SD15_TIMESTEPS {
        let fraction = index as f64 / (SD15_TIMESTEPS - 1) as f64;
        let root_beta = (end - start).mul_add(fraction, start);
        cumulative_alpha *= 1.0 - root_beta * root_beta;
        sigmas.push((((1.0 - cumulative_alpha) / cumulative_alpha).sqrt()) as f32);
    }
    Ok(sigmas)
}

fn validate_sigma_grid(sigmas: &[f32]) -> Result<(), SamplingProfileError> {
    if sigmas.len() < 2 {
        return Err(SamplingProfileError::EmptyGrid);
    }
    for (index, sigma) in sigmas.iter().copied().enumerate() {
        validate_sigma(sigma, false)?;
        if index > 0 && sigma <= sigmas[index - 1] {
            return Err(SamplingProfileError::UnorderedGrid { index });
        }
    }
    Ok(())
}

fn validate_sigma(sigma: f32, allow_zero: bool) -> Result<(), SamplingProfileError> {
    if !sigma.is_finite() || sigma < 0.0 || (!allow_zero && sigma == 0.0) {
        return Err(SamplingProfileError::InvalidSigma(sigma));
    }
    Ok(())
}

fn validate_equal_lengths(
    left: &[f32],
    right: &[f32],
    operation: &'static str,
) -> Result<(), SamplingProfileError> {
    if left.len() != right.len() {
        return Err(SamplingProfileError::LengthMismatch {
            operation,
            left: left.len(),
            right: right.len(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SamplingProfileError {
    #[error("invalid sampling-profile identity {0:?}")]
    InvalidIdentity(String),
    #[error("sampling profile sigma grid must contain at least two entries")]
    EmptyGrid,
    #[error("sampling profile sigma grid is not strictly increasing at index {index}")]
    UnorderedGrid { index: usize },
    #[error("sampling profile sigma {0} is invalid")]
    InvalidSigma(f32),
    #[error(
        "ancestral step is invalid for sigma_from {sigma_from}, sigma_to {sigma_to}, eta {eta}"
    )]
    InvalidAncestralStep {
        sigma_from: f32,
        sigma_to: f32,
        eta: f32,
    },
    #[error("sampling profile SNR sigma {0} is invalid")]
    InvalidSnrSigma(f32),
    #[error("sampling profile half-log-SNR {0} is invalid")]
    InvalidHalfLogSnr(f32),
    #[error("sampling profile SNR shift {0} is invalid")]
    InvalidSnrShift(f32),
    #[error("sampling profile noise scale {0} is invalid")]
    InvalidNoiseScale(f32),
    #[error("sampling profile percentage {0} is invalid")]
    InvalidSamplingPercent(f32),
    #[error("sampling profile model time {0} is invalid")]
    InvalidModelTime(f32),
    #[error("sampling profile model-time mapping is invalid")]
    InvalidModelTimeMapping,
    #[error("native sampling policy parameters are invalid")]
    InvalidPolicyParameters,
    #[error("native sampling policy latent shape is invalid")]
    InvalidLatentShape,
    #[error("native sampling policy identity changed")]
    PolicyIdentityChanged,
    #[error(transparent)]
    Autograd(#[from] sim_tensor::AutogradError),
    #[error(transparent)]
    Tensor(#[from] NativeDiffusionTensorError),
    #[error("sampling profile is unavailable for MODEL {0}")]
    UnsupportedModel(String),
    #[error("sampling profile grid index {0} is unavailable")]
    GridIndex(usize),
    #[error("sampling profile allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("{operation} length mismatch: {left} versus {right}")]
    LengthMismatch {
        operation: &'static str,
        left: usize,
        right: usize,
    },
    #[error("{operation} produced a non-finite value at index {index}")]
    NonFiniteOutput {
        operation: &'static str,
        index: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CancellationToken, CpuWorkspaceAuthority, StreamId};
    use std::error::Error;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "{actual} differs from {expected}",
        );
    }

    #[test]
    fn traced_prediction_matches_pinned_train_sampler_loss_and_vjp() -> Result<(), Box<dyn Error>> {
        use sha2::{Digest, Sha256};
        use sim_tensor::{AutogradError, GradientMode, GradientReducer, GradientStore, LeafId};

        struct SinglePublication;
        impl GradientReducer for SinglePublication {
            fn add(
                &self,
                _left: Tensor,
                _right: Tensor,
                _cancellation: &CancellationToken,
            ) -> Result<Tensor, AutogradError> {
                Err(AutogradError::InvalidGraph {
                    reason: "source fixture must publish each leaf once".to_owned(),
                })
            }
        }
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../sim_test_support/fixtures/models/auraflow-comfy-model-0064/training-policy-source-oracle.json"
        ))?;
        assert_eq!(
            oracle["generator_sha256"],
            format!(
                "{:x}",
                Sha256::digest(include_bytes!(
                    "../../sim_test_support/fixtures/models/auraflow-comfy-model-0064/generate_training_policy_source_oracle.py"
                ))
            )
        );
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let read_values =
            |case: &serde_json::Value, key: &str| -> Result<Vec<f32>, Box<dyn Error>> {
                case[key]
                    .as_array()
                    .ok_or("source values absent")?
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .map(|value| value as f32)
                            .ok_or_else(|| "source value absent".into())
                    })
                    .collect()
            };
        let compare = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 2e-6 + 2e-5 * expected.abs(),
                    "source prediction differs: {actual} vs {expected}"
                );
            }
        };
        let cases = oracle["cases"].as_array().ok_or("source cases absent")?;
        assert_eq!(cases.len(), 27);
        for case in cases {
            let prediction: NativePredictionKind =
                serde_json::from_value(case["prediction"].clone())?;
            let kind = match prediction {
                NativePredictionKind::Constant => NativeSamplingPolicyKind::Flow {
                    shift: 1.73,
                    multiplier: 1.0,
                    noise_scale: 1.0,
                },
                NativePredictionKind::Edm => NativeSamplingPolicyKind::ContinuousEdm {
                    prediction,
                    sigma_min: 0.002,
                    sigma_max: 120.0,
                    sigma_data: 0.5,
                },
                NativePredictionKind::CosmosRFlow => NativeSamplingPolicyKind::CosmosRFlow {
                    sigma_min: 0.002,
                    sigma_max: 120.0,
                    sigma_data: 1.0,
                },
                _ => NativeSamplingPolicyKind::Discrete {
                    prediction,
                    zero_terminal_snr: false,
                },
            };
            let policy = NativeSamplingPolicy::checked(kind)?;
            let sigma = case["sigma"].as_f64().ok_or("source sigma absent")? as f32;
            let latent = read_values(case, "latent")?;
            let mut noisy = read_values(case, "noise")?;
            policy.scale_initial_noise_in_place(&mut noisy, &latent, sigma, false)?;
            compare(&noisy, &read_values(case, "noisy")?);
            let mut target = vec![0.0; latent.len()];
            policy.scale_initial_noise_in_place(&mut target, &latent, 0.0, false)?;
            compare(&target, &read_values(case, "target")?);
            let mut scaled_input = noisy.clone();
            policy.scale_model_input_in_place(&mut scaled_input, sigma)?;
            compare(&scaled_input, &read_values(case, "scaled_input")?);
            let raw = read_values(case, "raw")?;
            let tensor = tensor_from_f32(&backend, &[raw.len() as u64], &raw, &context)?;
            let leaf = LeafId::new("source_raw_prediction")?;
            let mut tape = AutogradTape::new(GradientMode::Enabled);
            let scalars = tape.record_scalar_parameter(&tensor, leaf.clone(), &context)?;
            let interpreted = policy
                .interpret_prediction_for_training(&scalars, &noisy, sigma, &mut tape, &context)?;
            compare(
                &interpreted
                    .iter()
                    .map(|value| value.value())
                    .collect::<Vec<_>>(),
                &read_values(case, "denoised")?,
            );
            let mut loss = ScalarAutogradValue::constant(0.0);
            for (value, target) in interpreted.iter().zip(&target) {
                let difference = tape.scalar_add(*value, ScalarAutogradValue::constant(-target))?;
                let square = tape.scalar_multiply(difference, difference)?;
                loss = tape.scalar_add(loss, square)?;
            }
            loss = tape.scalar_scale(loss, 1.0 / interpreted.len() as f32)?;
            compare(
                &[loss.value()],
                &[case["loss"].as_f64().ok_or("source loss absent")? as f32],
            );
            let loss = tape.scalar_scale(loss, 0.5)?;
            let mut store = GradientStore::default();
            tape.reverse_scalar_and_publish(
                loss,
                &SinglePublication,
                &mut store,
                &backend,
                &context,
            )?;
            compare(
                &tensor_to_f32(
                    &backend,
                    store.gradient(&leaf).ok_or("gradient absent")?,
                    &context,
                )?,
                &read_values(case, "gradient")?,
            );
        }
        let policy = NativeSamplingPolicy::sd15()?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        assert!(matches!(
            policy.interpret_prediction_for_training(&[], &[1.0], 0.5, &mut tape, &context),
            Err(SamplingProfileError::LengthMismatch { .. })
        ));
        assert!(matches!(
            policy.interpret_prediction_for_training(&[], &[], f32::NAN, &mut tape, &context),
            Err(SamplingProfileError::InvalidSigma(_))
        ));
        let small_context = ExecutionContext {
            scratch: authority.authorize_workspace(1)?,
            ..context
        };
        assert!(matches!(
            policy.interpret_prediction_for_training(
                &[ScalarAutogradValue::constant(1.0)],
                &[1.0],
                0.5,
                &mut tape,
                &small_context
            ),
            Err(SamplingProfileError::OutOfMemory(_))
        ));
        assert!(matches!(
            policy.interpret_prediction_for_training(
                &[ScalarAutogradValue::constant(f32::INFINITY)],
                &[1.0],
                0.5,
                &mut tape,
                &context
            ),
            Err(SamplingProfileError::NonFiniteOutput { .. })
        ));
        assert!(cancellation.cancel());
        assert!(
            policy
                .interpret_prediction_for_training(&[], &[], 0.5, &mut tape, &context)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn family_model_profiles_match_source_flow_equations_and_sd15_compatibility()
    -> Result<(), SamplingProfileError> {
        let sd15 = DiscreteSamplingProfile::sd15()?;
        assert_close(sd15.sampling_percent_for_sigma(sd15.sigma_max())?, 0.0);
        assert_close(sd15.sampling_percent_for_sigma(sd15.sigma_min())?, 1.0);

        let aura = DiscreteSamplingProfile::auraflow()?;
        assert_eq!(aura.identity().as_str(), AURAFLOW_SAMPLING_PROFILE_ID);
        assert_eq!(aura.sigma_count(), AURAFLOW_TIMESTEPS);
        assert_eq!(aura.prediction(), PredictionInterpretation::Flow);
        assert_close(aura.sigma_at_index(AURAFLOW_TIMESTEPS - 1)?, 1.0);
        assert_close(aura.model_time_for_sigma(0.5)?, 0.5);
        assert_close(
            aura.sigma_for_model_time(0.5)?,
            time_snr_shift(AURAFLOW_SHIFT, 0.5),
        );
        assert_close(aura.sigma_for_model_time(0.0)?, 0.0);
        assert_close(aura.sampling_percent_for_sigma(1.0)?, 0.0);
        assert_close(aura.sampling_percent_for_sigma(0.0)?, 1.0);
        assert_close(
            aura.sampling_percent_for_sigma(time_snr_shift(AURAFLOW_SHIFT, 0.25))?,
            0.75,
        );

        let qwen = DiscreteSamplingProfile::qwen_image()?;
        assert_eq!(qwen.identity().as_str(), QWEN_IMAGE_SAMPLING_PROFILE_ID);
        assert_eq!(qwen.sigma_count(), QWEN_IMAGE_TIMESTEPS);
        assert_eq!(qwen.prediction(), PredictionInterpretation::Flow);
        assert_close(qwen.sigma_at_index(QWEN_IMAGE_TIMESTEPS - 1)?, 1.0);
        assert_close(qwen.model_time_for_sigma(0.5)?, 0.5);
        assert_close(
            qwen.sigma_for_model_time(0.5)?,
            flux_time_shift(QWEN_IMAGE_SHIFT, 0.5),
        );
        assert_close(qwen.sigma_for_model_time(0.0)?, 0.0);
        assert_close(qwen.sampling_percent_for_sigma(1.0)?, 0.0);
        assert_close(qwen.sampling_percent_for_sigma(0.0)?, 1.0);
        assert_close(
            qwen.sampling_percent_for_sigma(flux_time_shift(QWEN_IMAGE_SHIFT, 0.25))?,
            0.75,
        );

        let mut noise = [2.0, -1.0];
        let latent = [0.25, 0.5];
        aura.scale_initial_noise_in_place(&mut noise, &latent, 0.75, true)?;
        assert_eq!(noise, [1.5625, -0.625]);
        let mut input = [3.0, -2.0];
        qwen.scale_model_input_in_place(&mut input, 0.5)?;
        assert_eq!(input, [3.0, -2.0]);
        let mut output = [0.5, -1.0];
        aura.interpret_prediction_in_place(&mut output, &[2.0, 3.0], 0.25)?;
        assert_eq!(output, [1.875, 3.25]);
        Ok(())
    }

    #[test]
    fn native_sampling_policy_invalidates_prior_rounding_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let policy = NativeSamplingPolicy::sd15()?;
        let current = serde_json::to_value(&policy)?;
        let restored: NativeSamplingPolicy = serde_json::from_value(current.clone())?;
        assert_eq!(policy, restored);
        let mut prior = current;
        prior["identity"] = serde_json::json!(
            "native-policy-78d3d3ea1790bfcfefbd0bf153baf95b608d091f0a36e294dd2ee5bb89033014"
        );
        assert!(serde_json::from_value::<NativeSamplingPolicy>(prior).is_err());
        Ok(())
    }

    #[test]
    fn x0_source_input_scaling_invalidates_prior_policy_identity() -> Result<(), Box<dyn Error>> {
        let policy = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
            prediction: NativePredictionKind::Denoised,
            zero_terminal_snr: false,
        })?;
        let mut prior_hash = Sha256::new();
        prior_hash.update(b"zed.sim.native-sampling-policy.v2");
        prior_hash.update([0, NativePredictionKind::Denoised as u8, 0]);
        hash_native_latent_transform(&mut prior_hash, policy.latent_transform());
        let prior_identity = format!("native-policy-{:x}", prior_hash.finalize());
        assert_ne!(policy.identity().as_str(), prior_identity);
        let mut prior = serde_json::to_value(&policy)?;
        prior["identity"] = serde_json::Value::String(prior_identity);
        assert!(serde_json::from_value::<NativeSamplingPolicy>(prior).is_err());
        Ok(())
    }

    #[test]
    fn native_discrete_input_scaling_preserves_source_f32_division()
    -> Result<(), SamplingProfileError> {
        let profile = NativeSamplingPolicy::sd15()?;
        for (sigma, input, expected_bits) in [
            (1.5, 0.3, 0x3e2a_6767),
            (3.0, 0.1, 0x3d01_86e2),
            (14.6146, 0.1, 0x3bdf_b0ec),
        ] {
            let mut values = [input, -input];
            profile.scale_model_input_in_place(&mut values, sigma)?;
            assert_eq!(
                values.map(f32::to_bits),
                [expected_bits, expected_bits | 0x8000_0000]
            );
        }
        Ok(())
    }

    #[test]
    fn task407_native_sampling_policies_match_source_equations_and_bounds()
    -> Result<(), SamplingProfileError> {
        let policies = [
            NativeSamplingPolicyKind::Discrete {
                prediction: NativePredictionKind::Epsilon,
                zero_terminal_snr: false,
            },
            NativeSamplingPolicyKind::ContinuousEdm {
                prediction: NativePredictionKind::Edm,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 0.5,
            },
            NativeSamplingPolicyKind::ContinuousV {
                sigma_min: 0.03,
                sigma_max: 500.0,
                sigma_data: 1.0,
            },
            NativeSamplingPolicyKind::Flow {
                shift: 1.73,
                multiplier: 1.0,
                noise_scale: 1.0,
            },
            NativeSamplingPolicyKind::StableCascade { shift: 2.0 },
            NativeSamplingPolicyKind::Sd3 {
                shift: 3.0,
                multiplier: 1_000.0,
                noise_scale: 1.0,
            },
            NativeSamplingPolicyKind::Flux { shift: 1.15 },
            NativeSamplingPolicyKind::Ltxv { shift: 2.37 },
            NativeSamplingPolicyKind::CosmosRFlow {
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 1.0,
            },
        ];
        let mut identities = std::collections::BTreeSet::new();
        for kind in policies {
            let policy = NativeSamplingPolicy::checked(kind)?;
            policy.validate()?;
            assert!(identities.insert(policy.semantic_digest_sha256()));
            assert!(policy.sigma_min() > 0.0);
            assert!(policy.sigma_max() > policy.sigma_min());
            let sigma = policy.sigma_at_index(policy.sigma_count() / 2)?;
            let time = policy.model_time_for_sigma(sigma)?;
            let round_trip = policy.sigma_for_model_time(time)?;
            match kind {
                NativeSamplingPolicyKind::Flow {
                    shift, multiplier, ..
                }
                | NativeSamplingPolicyKind::Sd3 {
                    shift, multiplier, ..
                } => {
                    assert_close(time, sigma * multiplier);
                    assert_close(round_trip, time_snr_shift(shift, sigma));
                }
                NativeSamplingPolicyKind::Flux { shift } => {
                    assert_close(time, sigma);
                    assert_close(round_trip, flux_time_shift(shift, sigma));
                }
                NativeSamplingPolicyKind::Ltxv { shift } => {
                    assert_close(time, sigma);
                    assert_close(round_trip, time_snr_shift(shift, sigma));
                }
                NativeSamplingPolicyKind::StableCascade { .. } => {
                    let cosine_s = 8.0e-3_f32;
                    let initial = (cosine_s / (1.0 + cosine_s) * std::f32::consts::FRAC_PI_2)
                        .cos()
                        .powi(2);
                    let variance = 1.0 / (sigma * sigma + 1.0);
                    let expected_time = (variance * initial).sqrt().acos()
                        / std::f32::consts::FRAC_PI_2
                        * (1.0 + cosine_s)
                        - cosine_s;
                    assert_close(time, expected_time);
                    assert_ne!(round_trip, sigma);
                }
                _ => assert!(
                    (round_trip - sigma).abs() <= sigma.abs().max(1.0) * 2.0e-4,
                    "{kind:?}: sigma={sigma}, time={time}, round_trip={round_trip}",
                ),
            }
            assert!((0.0..=1.0).contains(&policy.sampling_percent_for_sigma(sigma)?));
            let percent_sigma = policy.sigma_for_sampling_percent(0.37)?;
            assert!(percent_sigma.is_finite() && percent_sigma > 0.0);
            assert!(
                (policy.sampling_percent_for_sigma(percent_sigma)? - 0.37).abs() <= 0.002,
                "{kind:?}: percent round trip changed",
            );
        }

        let zsnr = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
            prediction: NativePredictionKind::VPrediction,
            zero_terminal_snr: true,
        })?;
        let ordinary = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
            prediction: NativePredictionKind::VPrediction,
            zero_terminal_snr: false,
        })?;
        assert!((zsnr.sigma_at_index(0)? - ordinary.sigma_at_index(0)?).abs() < 2.0e-6);
        assert!((zsnr.sigma_max() - 4_518.76).abs() < 0.01);
        assert_ne!(
            zsnr.semantic_digest_sha256(),
            ordinary.semantic_digest_sha256()
        );

        let edm = NativeSamplingPolicy::checked_with_latent(
            NativeSamplingPolicyKind::ContinuousEdm {
                prediction: NativePredictionKind::Edm,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 0.5,
            },
            NativeLatentTransform::PlaygroundV25,
        )?;
        let mut prediction = [0.25];
        edm.interpret_prediction_in_place(&mut prediction, &[2.0], 1.0)?;
        assert_close(prediction[0], 0.5118034);
        let mut playground = [-1.6574, 1.886, -1.383, 2.5155];
        edm.process_latent_in_place(&mut playground)?;
        assert_eq!(playground, [0.0; 4]);
        edm.process_latent_out_place(&mut playground)?;
        assert_eq!(playground, [-1.6574, 1.886, -1.383, 2.5155]);

        let cosmos = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::CosmosRFlow {
            sigma_min: 0.002,
            sigma_max: 120.0,
            sigma_data: 1.0,
        })?;
        let mut input = [4.0];
        cosmos.scale_model_input_in_place(&mut input, 3.0)?;
        assert_close(input[0], 1.0);
        let mut output = [2.0];
        cosmos.interpret_prediction_in_place(&mut output, &[4.0], 3.0)?;
        assert_close(output[0], -0.5);

        let sd3 = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Sd3 {
            shift: 3.0,
            multiplier: 1_000.0,
            noise_scale: 0.5,
        })?;
        let mut sd3_latent = [1.0];
        sd3.process_latent_in_place(&mut sd3_latent)?;
        assert_close(sd3_latent[0], (1.0 - 0.0609) * 1.5305);
        sd3.process_latent_out_place(&mut sd3_latent)?;
        assert_close(sd3_latent[0], 1.0);
        let mut flow_noise = [2.0];
        sd3.scale_initial_noise_in_place(&mut flow_noise, &[0.25], 0.75, false)?;
        assert_close(flow_noise[0], 0.8125);
        let flux = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Flux { shift: 1.15 })?;
        let mut flux_latent = [1.0];
        flux.process_latent_in_place(&mut flux_latent)?;
        assert_close(flux_latent[0], (1.0 - 0.1159) * 0.3611);
        flux.process_latent_out_place(&mut flux_latent)?;
        assert_close(flux_latent[0], 1.0);

        let image_flow = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
            prediction: NativePredictionKind::ImageToImageFlow,
            zero_terminal_snr: false,
        })?;
        let mut image_flow_noise = [5.0];
        image_flow.scale_initial_noise_in_place(&mut image_flow_noise, &[0.25], 0.75, false)?;
        assert_eq!(image_flow_noise, [0.25]);
        image_flow.inverse_noise_scaling_in_place(0.25, &mut image_flow_noise)?;
        assert_eq!(image_flow_noise, [0.75]);
        let constant = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Flow {
            shift: 1.0,
            multiplier: 1.0,
            noise_scale: 1.0,
        })?;
        let mut constant_latent = [0.75];
        constant.inverse_noise_scaling_in_place(0.25, &mut constant_latent)?;
        assert_eq!(constant_latent, [1.0]);

        for (prediction, expected) in [
            (NativePredictionKind::Epsilon, 1.0),
            (NativePredictionKind::VPrediction, 0.29289323),
            (NativePredictionKind::Edm, 1.7071068),
        ] {
            let policy = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::ContinuousEdm {
                prediction,
                sigma_min: 0.03,
                sigma_max: 14.0,
                sigma_data: 1.0,
            })?;
            let mut value = [1.0];
            policy.interpret_prediction_in_place(&mut value, &[2.0], 1.0)?;
            assert_close(value[0], expected);
        }
        for (prediction, expected) in [
            (NativePredictionKind::Epsilon, 1.0),
            (NativePredictionKind::VPrediction, 0.29289323),
            (NativePredictionKind::VPredictionDdpm, 0.70710677),
            (NativePredictionKind::Denoised, 1.0),
            (NativePredictionKind::ImageToImage, 1.0),
            (NativePredictionKind::ImageToImageFlow, 1.0),
        ] {
            let policy = NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
                prediction,
                zero_terminal_snr: false,
            })?;
            let mut value = [1.0];
            policy.interpret_prediction_in_place(&mut value, &[2.0], 1.0)?;
            assert_close(value[0], expected);
        }
        assert!(matches!(
            NativeSamplingPolicy::checked(NativeSamplingPolicyKind::ContinuousEdm {
                prediction: NativePredictionKind::Lcm,
                sigma_min: 0.03,
                sigma_max: 14.0,
                sigma_data: 1.0,
            }),
            Err(SamplingProfileError::InvalidPolicyParameters)
        ));
        assert!(matches!(
            edm.sigma_for_sampling_percent(f32::NAN),
            Err(SamplingProfileError::InvalidSamplingPercent(_))
        ));
        let mut encoded = serde_json::to_value(&edm)
            .map_err(|_| SamplingProfileError::InvalidPolicyParameters)?;
        encoded["identity"] = serde_json::Value::String("changed-policy".to_owned());
        assert!(serde_json::from_value::<NativeSamplingPolicy>(encoded).is_err());

        assert!(matches!(
            NativeSamplingPolicy::checked(NativeSamplingPolicyKind::ContinuousV {
                sigma_min: 1.0,
                sigma_max: 0.5,
                sigma_data: 1.0,
            }),
            Err(SamplingProfileError::InvalidPolicyParameters)
        ));
        Ok(())
    }

    #[test]
    fn task407_playground_latent_transform_uses_nchw_channel_axis() -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1 << 18)?,
            &cancellation,
        );
        let policy = NativeSamplingPolicy::checked_with_latent(
            NativeSamplingPolicyKind::ContinuousEdm {
                prediction: NativePredictionKind::Edm,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 0.5,
            },
            NativeLatentTransform::PlaygroundV25,
        )?;
        let source = [
            -1.6574, -1.6574, 1.886, 1.886, -1.383, -1.383, 2.5155, 2.5155,
        ];
        let latent = tensor_from_f32(&backend, &[1, 4, 1, 2], &source, &context)?;
        let transformed = policy.process_latent_in(&backend, &latent, &context)?;
        assert_eq!(
            &*tensor_to_f32(&backend, &transformed, &context)?,
            &[0.0; 8]
        );
        let restored = policy.process_latent_out(&backend, &transformed, &context)?;
        assert_eq!(&*tensor_to_f32(&backend, &restored, &context)?, &source);
        let invalid = tensor_from_f32(&backend, &[1, 2, 1, 2], &[0.0; 4], &context)?;
        assert!(matches!(
            policy.process_latent_in(&backend, &invalid, &context),
            Err(SamplingProfileError::InvalidLatentShape)
        ));

        let discrete = NativeSamplingPolicyKind::Discrete {
            prediction: NativePredictionKind::Epsilon,
            zero_terminal_snr: false,
        };
        for (identifier, source, expected) in [
            ("SD15", 2.0, 2.0 * 0.18215),
            ("SDXL", 2.0, 2.0 * 0.13025),
            ("SC_B", 0.43, 1.0),
            ("SC_Prior", 2.0, 2.0),
            ("LTXV", 2.0, 2.0),
            ("Cosmos1CV8x8x8", 2.0, 2.0),
        ] {
            let policy = NativeSamplingPolicy::checked_with_latent(
                discrete,
                latent_transform_for_identifier(identifier),
            )?;
            let mut value = [source];
            policy.process_latent_in_place(&mut value)?;
            assert_close(value[0], expected);
            policy.process_latent_out_place(&mut value)?;
            assert_close(value[0], source);
        }
        let qwen = NativeSamplingPolicy::checked_with_latent(
            NativeSamplingPolicyKind::Flux { shift: 1.15 },
            latent_transform_for_identifier("Wan21"),
        )?;
        let qwen_source = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
            0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ];
        let qwen_tensor = tensor_from_f32(&backend, &[1, 16, 1, 1, 1], &qwen_source, &context)?;
        let qwen_normalized = qwen.process_latent_in(&backend, &qwen_tensor, &context)?;
        assert_eq!(
            &*tensor_to_f32(&backend, &qwen_normalized, &context)?,
            &[0.0; 16]
        );
        let qwen_restored = qwen.process_latent_out(&backend, &qwen_normalized, &context)?;
        assert_eq!(
            &*tensor_to_f32(&backend, &qwen_restored, &context)?,
            &qwen_source
        );
        Ok(())
    }
}
