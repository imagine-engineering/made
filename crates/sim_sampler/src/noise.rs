use crate::validate_identifier;
use serde::{Deserialize, Deserializer, Serialize};
use sim_tensor::{
    CompatibilityRngTransaction, CpuBackend, DeviceId, ExecutionContext, RetryRngPolicy,
    RngAlgorithm, RngCheckpoint, RngCompatibilityError, RngCompatibilityRequest, RngError,
    RngExecutionScope, RngGenerationPlacement, RngProfileVersion, RngSeedTransform, RngStream,
    RngStreamAddress, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32},
};
use thiserror::Error;

pub const NATIVE_DIFFUSION_NOISE_DOMAIN_ID: &str = "native-diffusion-noise-v1";
pub const INITIAL_NOISE_PHASE_ID: &str = "initial-noise-v1";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct NoisePhaseIdentity(String);

impl NoisePhaseIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, NoiseError> {
        let value = value.into();
        validate_identifier("noise phase", &value)
            .map_err(|_| NoiseError::InvalidPhase(value.clone()))?;
        Ok(Self(value))
    }

    pub fn initial() -> Self {
        Self(INITIAL_NOISE_PHASE_ID.to_owned())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for NoisePhaseIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseRequest {
    pub workflow_id: String,
    pub prompt_id: String,
    pub node_id: String,
    pub node_ordinal: u32,
    pub phase: NoisePhaseIdentity,
    pub phase_ordinal: u32,
    pub attempt: u32,
    pub retry_policy: RetryRngPolicy,
}

impl NoiseRequest {
    pub fn new(
        workflow_id: impl Into<String>,
        prompt_id: impl Into<String>,
        node_id: impl Into<String>,
        node_ordinal: u32,
        phase: NoisePhaseIdentity,
        phase_ordinal: u32,
        attempt: u32,
        retry_policy: RetryRngPolicy,
    ) -> Result<Self, NoiseError> {
        let request = Self {
            workflow_id: workflow_id.into(),
            prompt_id: prompt_id.into(),
            node_id: node_id.into(),
            node_ordinal,
            phase,
            phase_ordinal,
            attempt,
            retry_policy,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn native_diffusion(
        prompt_id: impl Into<String>,
        node_id: impl Into<String>,
    ) -> Result<Self, NoiseError> {
        Self::new(
            NATIVE_DIFFUSION_NOISE_DOMAIN_ID,
            prompt_id,
            node_id,
            0,
            NoisePhaseIdentity::initial(),
            0,
            0,
            RetryRngPolicy::Replay,
        )
    }

    pub fn stream(&self, seed: u64, device: DeviceId) -> Result<RngStream, NoiseError> {
        self.validate()?;
        let address = RngStreamAddress::for_device(
            &self.workflow_id,
            &self.prompt_id,
            &self.node_id,
            self.node_ordinal,
            self.phase.as_str(),
            u64::from(self.phase_ordinal),
            self.attempt,
            self.retry_policy,
            device,
        )?;
        Ok(RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            seed,
            address,
        )?)
    }

    fn validate(&self) -> Result<(), NoiseError> {
        for (kind, value) in [
            ("workflow", self.workflow_id.as_str()),
            ("prompt", self.prompt_id.as_str()),
            ("node", self.node_id.as_str()),
        ] {
            if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
                return Err(NoiseError::InvalidAddress {
                    kind,
                    value: value.to_owned(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityNoiseRequest {
    workflow: String,
    attempt: String,
    node: String,
    output: u32,
    execution_ordinal: u64,
    batch: u64,
    retry: u32,
    retry_policy: RetryRngPolicy,
}

impl CompatibilityNoiseRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        workflow: impl Into<String>,
        attempt: impl Into<String>,
        node: impl Into<String>,
        output: u32,
        execution_ordinal: u64,
        batch: u64,
        retry: u32,
        retry_policy: RetryRngPolicy,
    ) -> Self {
        Self {
            workflow: workflow.into(),
            attempt: attempt.into(),
            node: node.into(),
            output,
            execution_ordinal,
            batch,
            retry,
            retry_policy,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_transaction(
        self,
        contract_id: &'static str,
        base_seed: i128,
        seed_transform: RngSeedTransform,
        generation_placement: RngGenerationPlacement,
        checkpoint: Option<RngCheckpoint>,
        cancellation: &sim_tensor::CancellationToken,
    ) -> Result<CompatibilityRngTransaction, RngCompatibilityError> {
        CompatibilityRngTransaction::open(
            contract_id,
            RngCompatibilityRequest::new(
                self.workflow,
                self.attempt,
                self.node,
                self.output,
                self.execution_ordinal,
                self.batch,
                self.retry,
                self.retry_policy,
                base_seed,
                seed_transform,
                generation_placement,
                RngExecutionScope::Production,
            ),
            checkpoint,
            cancellation,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BrownianNoiseIntervalAddress {
    pub lower_sigma_bits: u32,
    pub upper_sigma_bits: u32,
    pub reverse: bool,
    pub step: u32,
}

impl BrownianNoiseIntervalAddress {
    pub fn new(start_sigma: f32, end_sigma: f32, step: u32) -> Result<Self, NoiseError> {
        validate_positive_sigma(start_sigma)?;
        validate_positive_sigma(end_sigma)?;
        if start_sigma == end_sigma {
            return Err(NoiseError::EmptyBrownianInterval(start_sigma));
        }
        let (lower, upper, reverse) = if start_sigma < end_sigma {
            (start_sigma, end_sigma, false)
        } else {
            (end_sigma, start_sigma, true)
        };
        Ok(Self {
            lower_sigma_bits: lower.to_bits(),
            upper_sigma_bits: upper.to_bits(),
            reverse,
            step,
        })
    }

    pub fn canonical_interval(&self) -> (f32, f32) {
        (
            f32::from_bits(self.lower_sigma_bits),
            f32::from_bits(self.upper_sigma_bits),
        )
    }
}

#[derive(Clone, Debug)]
pub struct NoiseTrace {
    pub noise: Tensor,
    pub before: RngCheckpoint,
    pub after: RngCheckpoint,
}

pub fn normal_noise(
    backend: &CpuBackend,
    shape: &[u64],
    stream: &RngStream,
    context: &ExecutionContext<'_>,
) -> Result<NoiseTrace, NoiseError> {
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(NoiseError::ShapeOverflow)
    })?;
    let count = usize::try_from(count).map_err(|_| NoiseError::ShapeOverflow)?;
    context
        .cancellation
        .check()
        .map_err(|_| NoiseError::Cancelled)?;
    let before = stream.begin(None)?.commit();
    let mut transaction = stream.begin(Some(before.clone()))?;
    transaction.require_device(DeviceId::CPU)?;
    let mut values = backend.workspace_vec::<f32>(context, count)?;
    while values.len() < count {
        let pair = transaction.next_standard_normal_pair(context.cancellation)?;
        values.try_push(pair[0] as f32)?;
        if values.len() < count {
            values.try_push(pair[1] as f32)?;
        }
    }
    let noise = tensor_from_f32(backend, shape, &values, context)?;
    let after = transaction.commit();
    Ok(NoiseTrace {
        noise,
        before,
        after,
    })
}

fn validate_positive_sigma(sigma: f32) -> Result<(), NoiseError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(NoiseError::InvalidSigma(sigma));
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NoiseError {
    #[error("invalid noise phase {0:?}")]
    InvalidPhase(String),
    #[error("invalid RNG {kind} address {value:?}")]
    InvalidAddress { kind: &'static str, value: String },
    #[error("noise shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("invalid Brownian sigma {0}")]
    InvalidSigma(f32),
    #[error("Brownian interval endpoints are identical at {0}")]
    EmptyBrownianInterval(f32),
    #[error("noise generation was cancelled")]
    Cancelled,
    #[error(transparent)]
    Rng(#[from] RngError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
}
