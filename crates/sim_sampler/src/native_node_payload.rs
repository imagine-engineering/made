use crate::{
    CfgPpDenoiserOutput, CompatibilityNoiseRequest, GuidanceDenoiser, GuidanceError, GuidanceHook,
    GuidanceOptions, GuidanceResult, NativeDiffusionPayload, NoiseError, NoiseRequest,
    SamplerIdentity, SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileError,
    SamplingProgress, SamplingTrace, execute_dual_cfg_guidance,
    execute_dual_cfg_guidance_with_modifier_session, execute_dual_model_guidance,
    execute_dual_model_guidance_with_modifier_session, execute_guidance,
    execute_guidance_with_modifier_session,
    generated_native_diffusion::NativeDiffusionSamplerError, profile_for_model,
};
use sha2::{Digest, Sha256};
use sim_model::conditioning::{ConditioningError, ConditioningIdentity, ConditioningSet};
use sim_model::{
    NativeModelPayload, NativeModelPayloadError, NativeModelResourceRole,
    generated_native_diffusion::{sd15_latent_format_identity, sd15_model_family_identity},
};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, RngCheckpoint, RngError, StorageId, Tensor,
    TensorBackend, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    sync::Arc,
};
use thiserror::Error;

const MAX_BATCH_NOISE_INDEX: u64 = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeNoiseSourceKind {
    Zero,
    Random,
    FixedLatent,
    LatentSource,
}

#[derive(Clone)]
enum NativeNoiseResource {
    Zero,
    Random { seed: u64 },
    FixedLatent { seed: u64, samples: Tensor },
    LatentSource { samples: Tensor },
}

#[derive(Clone)]
pub struct NativeNoisePayload {
    resource: NativeNoiseResource,
    semantic_digest_sha256: String,
    resident_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct NativeNoiseGeneration {
    pub noise: Tensor,
    pub before: Option<RngCheckpoint>,
    pub after: Option<RngCheckpoint>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeNoiseTensorResidentAllocation {
    storage_id: StorageId,
    resident_bytes: u64,
}

impl NativeNoiseTensorResidentAllocation {
    pub const fn storage_id(&self) -> StorageId {
        self.storage_id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeNoiseResidentParts {
    owned_bytes: u64,
    tensor_allocation: Option<NativeNoiseTensorResidentAllocation>,
}

impl NativeNoiseResidentParts {
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn tensor_allocation(&self) -> Option<&NativeNoiseTensorResidentAllocation> {
        self.tensor_allocation.as_ref()
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeSamplerPayloadError> {
        self.tensor_allocation
            .map_or(Ok(self.owned_bytes), |allocation| {
                checked_add(self.owned_bytes, allocation.resident_bytes)
            })
    }
}

impl NativeNoisePayload {
    pub fn zero() -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(NativeNoiseResource::Zero)
    }

    pub fn random(seed: u64) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(NativeNoiseResource::Random { seed })
    }

    pub fn fixed_latent(seed: u64, samples: Tensor) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(NativeNoiseResource::FixedLatent { seed, samples })
    }

    pub fn latent_source(samples: Tensor) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(NativeNoiseResource::LatentSource { samples })
    }

    pub const fn kind(&self) -> NativeNoiseSourceKind {
        match &self.resource {
            NativeNoiseResource::Zero => NativeNoiseSourceKind::Zero,
            NativeNoiseResource::Random { .. } => NativeNoiseSourceKind::Random,
            NativeNoiseResource::FixedLatent { .. } => NativeNoiseSourceKind::FixedLatent,
            NativeNoiseResource::LatentSource { .. } => NativeNoiseSourceKind::LatentSource,
        }
    }

    pub const fn seed(&self) -> u64 {
        match &self.resource {
            NativeNoiseResource::Zero | NativeNoiseResource::LatentSource { .. } => 0,
            NativeNoiseResource::Random { seed }
            | NativeNoiseResource::FixedLatent { seed, .. } => *seed,
        }
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeNoiseResidentParts, NativeSamplerPayloadError> {
        let owned_bytes = checked_add(
            usize_to_u64(mem::size_of::<Self>())?,
            usize_to_u64(self.semantic_digest_sha256.capacity())?,
        )?;
        let tensor_allocation = match &self.resource {
            NativeNoiseResource::FixedLatent { samples, .. }
            | NativeNoiseResource::LatentSource { samples } => {
                Some(NativeNoiseTensorResidentAllocation {
                    storage_id: samples.storage_id(),
                    resident_bytes: samples.storage_byte_len(),
                })
            }
            NativeNoiseResource::Zero | NativeNoiseResource::Random { .. } => None,
        };
        let parts = NativeNoiseResidentParts {
            owned_bytes,
            tensor_allocation,
        };
        if parts.resident_bytes()? != self.resident_bytes {
            return Err(NativeSamplerPayloadError::ProjectionChanged(
                "noise residency",
            ));
        }
        Ok(parts)
    }

    pub fn validate(&self) -> Result<(), NativeSamplerPayloadError> {
        let expected = Self::checked(self.resource.clone())?;
        if self.semantic_digest_sha256 != expected.semantic_digest_sha256
            || self.resident_bytes != expected.resident_bytes
        {
            return Err(NativeSamplerPayloadError::ProjectionChanged("noise"));
        }
        Ok(())
    }

    pub fn generate(
        &self,
        backend: &CpuBackend,
        input_latent: &Tensor,
        batch_indices: Option<&[u64]>,
        request: &NoiseRequest,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeNoiseGeneration, NativeSamplerPayloadError> {
        context.check()?;
        self.validate()?;
        if let NativeNoiseResource::LatentSource { samples } = &self.resource {
            let (noise, _) = backend.copy(samples, samples.descriptor().clone(), context)?;
            context.check()?;
            return Ok(NativeNoiseGeneration {
                noise,
                before: None,
                after: None,
            });
        }
        validate_noise_input(input_latent, batch_indices)?;
        match &self.resource {
            NativeNoiseResource::Zero => {
                let count = tensor_element_count(input_latent)?;
                let values = vec![0.0_f32; count];
                Ok(NativeNoiseGeneration {
                    noise: tensor_from_f32(
                        backend,
                        input_latent.descriptor().shape(),
                        &values,
                        context,
                    )?,
                    before: None,
                    after: None,
                })
            }
            NativeNoiseResource::Random { seed } => generate_random_noise(
                backend,
                input_latent,
                batch_indices,
                request,
                *seed,
                context,
            ),
            NativeNoiseResource::LatentSource { .. } => Err(
                NativeSamplerPayloadError::ProjectionChanged("noise source dispatch"),
            ),
            NativeNoiseResource::FixedLatent { samples, .. } => {
                if samples.descriptor() != input_latent.descriptor() {
                    return Err(NativeSamplerPayloadError::NoiseShapeMismatch);
                }
                context.check()?;
                Ok(NativeNoiseGeneration {
                    noise: samples.clone(),
                    before: None,
                    after: None,
                })
            }
        }
    }

    fn checked(resource: NativeNoiseResource) -> Result<Self, NativeSamplerPayloadError> {
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-noise-payload.v1");
        hasher.update([0]);
        let mut resident_bytes = usize_to_u64(mem::size_of::<Self>())?;
        match &resource {
            NativeNoiseResource::Zero => hasher.update(b"zero"),
            NativeNoiseResource::Random { seed } => {
                hasher.update(b"random");
                hasher.update(seed.to_le_bytes());
            }
            NativeNoiseResource::LatentSource { samples } => {
                validate_noise_tensor(samples)?;
                hasher.update(b"latent-source");
                hash_tensor(&mut hasher, samples)?;
                resident_bytes = checked_add(resident_bytes, samples.storage_byte_len())?;
            }
            NativeNoiseResource::FixedLatent { seed, samples } => {
                validate_noise_tensor(samples)?;
                hasher.update(b"fixed-latent");
                hasher.update(seed.to_le_bytes());
                hash_tensor(&mut hasher, samples)?;
                resident_bytes = checked_add(resident_bytes, samples.storage_byte_len())?;
            }
        }
        let semantic_digest_sha256 = format!("{:x}", hasher.finalize());
        resident_bytes = checked_add(
            resident_bytes,
            usize_to_u64(semantic_digest_sha256.capacity())?,
        )?;
        Ok(Self {
            resource,
            semantic_digest_sha256,
            resident_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeGuiderKind {
    Basic,
    Cfg,
    DualCfg,
    DualModel,
}

#[derive(Clone, Copy)]
pub enum NativeGuiderConditioningSets<'a> {
    Basic {
        conditioning: &'a Arc<ConditioningSet>,
    },
    Cfg {
        positive: &'a Arc<ConditioningSet>,
        negative: &'a Arc<ConditioningSet>,
    },
    DualCfg {
        positive: &'a Arc<ConditioningSet>,
        middle: &'a Arc<ConditioningSet>,
        negative: &'a Arc<ConditioningSet>,
    },
    DualModel {
        positive: &'a Arc<ConditioningSet>,
        negative: &'a Arc<ConditioningSet>,
        negative_model: &'a Arc<NativeModelPayload>,
    },
}

#[derive(Clone)]
enum NativeGuiderStrategy {
    Basic {
        conditioning: Arc<ConditioningSet>,
    },
    Cfg {
        positive: Arc<ConditioningSet>,
        negative: Arc<ConditioningSet>,
        guidance: f32,
    },
    DualCfg {
        positive: Arc<ConditioningSet>,
        middle: Arc<ConditioningSet>,
        negative: Arc<ConditioningSet>,
        first_guidance: f32,
        second_guidance: f32,
        nested: bool,
    },
    DualModel {
        positive: Arc<ConditioningSet>,
        negative: Arc<ConditioningSet>,
        negative_model: Arc<NativeModelPayload>,
        guidance: f32,
    },
}

#[derive(Clone)]
pub struct NativeGuiderPayload {
    model: Arc<NativeModelPayload>,
    strategy: NativeGuiderStrategy,
    execution_payload: Option<Arc<NativeDiffusionPayload>>,
    negative_execution_payload: Option<Arc<NativeDiffusionPayload>>,
    semantic_digest_sha256: String,
    resident_bytes: u64,
}

impl NativeGuiderPayload {
    pub fn basic(
        model: Arc<NativeModelPayload>,
        conditioning: Arc<ConditioningSet>,
    ) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(model, NativeGuiderStrategy::Basic { conditioning })
    }

    pub fn cfg(
        model: Arc<NativeModelPayload>,
        positive: Arc<ConditioningSet>,
        negative: Arc<ConditioningSet>,
        guidance: f32,
    ) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(
            model,
            NativeGuiderStrategy::Cfg {
                positive,
                negative,
                guidance,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dual_cfg(
        model: Arc<NativeModelPayload>,
        positive: Arc<ConditioningSet>,
        middle: Arc<ConditioningSet>,
        negative: Arc<ConditioningSet>,
        first_guidance: f32,
        second_guidance: f32,
        nested: bool,
    ) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(
            model,
            NativeGuiderStrategy::DualCfg {
                positive,
                middle,
                negative,
                first_guidance,
                second_guidance,
                nested,
            },
        )
    }

    pub fn dual_model(
        model: Arc<NativeModelPayload>,
        negative_model: Arc<NativeModelPayload>,
        positive: Arc<ConditioningSet>,
        negative: Arc<ConditioningSet>,
        guidance: f32,
    ) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(
            model,
            NativeGuiderStrategy::DualModel {
                positive,
                negative,
                negative_model,
                guidance,
            },
        )
    }

    pub const fn kind(&self) -> NativeGuiderKind {
        match &self.strategy {
            NativeGuiderStrategy::Basic { .. } => NativeGuiderKind::Basic,
            NativeGuiderStrategy::Cfg { .. } => NativeGuiderKind::Cfg,
            NativeGuiderStrategy::DualCfg { .. } => NativeGuiderKind::DualCfg,
            NativeGuiderStrategy::DualModel { .. } => NativeGuiderKind::DualModel,
        }
    }

    pub fn model(&self) -> &Arc<NativeModelPayload> {
        &self.model
    }

    pub fn execution_payload(&self) -> Option<&Arc<NativeDiffusionPayload>> {
        self.execution_payload.as_ref()
    }

    pub fn negative_execution_payload(&self) -> Option<&Arc<NativeDiffusionPayload>> {
        self.negative_execution_payload.as_ref()
    }

    pub fn with_execution_payload(
        mut self,
        payload: Arc<NativeDiffusionPayload>,
    ) -> Result<Self, NativeSamplerPayloadError> {
        payload.validate().map_err(|_| {
            NativeSamplerPayloadError::ProjectionChanged("guider execution payload")
        })?;
        let source_model = payload
            .model_resources()
            .map(|(model, _)| model)
            .ok_or(NativeSamplerPayloadError::GuiderModelRoleMismatch)?;
        if !Arc::ptr_eq(source_model, &self.model)
            || source_model.identity().execution_sha256()
                != self.model.identity().execution_sha256()
        {
            return Err(NativeSamplerPayloadError::GuiderModelRoleMismatch);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-guider-execution-payload.v1");
        hash_field(&mut hasher, self.semantic_digest_sha256.as_bytes())?;
        hash_field(&mut hasher, payload.digest_sha256().as_bytes())?;
        self.semantic_digest_sha256 = format!("{:x}", hasher.finalize());
        self.execution_payload = Some(payload);
        self.resident_bytes = self
            .resident_parts()
            .and_then(|parts| parts.resident_bytes())
            .map_err(Box::new)?;
        Ok(self)
    }

    pub fn with_negative_execution_payload(
        mut self,
        payload: Arc<NativeDiffusionPayload>,
    ) -> Result<Self, NativeSamplerPayloadError> {
        payload.validate().map_err(|_| {
            NativeSamplerPayloadError::ProjectionChanged("negative guider execution payload")
        })?;
        let NativeGuiderStrategy::DualModel { negative_model, .. } = &self.strategy else {
            return Err(NativeSamplerPayloadError::DualModelDenoiserRequired);
        };
        let source_model = payload
            .model_resources()
            .map(|(model, _)| model)
            .ok_or(NativeSamplerPayloadError::GuiderModelRoleMismatch)?;
        if !Arc::ptr_eq(source_model, negative_model)
            || source_model.identity().execution_sha256()
                != negative_model.identity().execution_sha256()
        {
            return Err(NativeSamplerPayloadError::GuiderModelRoleMismatch);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-guider-negative-execution-payload.v1");
        hash_field(&mut hasher, self.semantic_digest_sha256.as_bytes())?;
        hash_field(&mut hasher, payload.digest_sha256().as_bytes())?;
        self.semantic_digest_sha256 = format!("{:x}", hasher.finalize());
        self.negative_execution_payload = Some(payload);
        self.resident_bytes = self
            .resident_parts()
            .and_then(|parts| parts.resident_bytes())
            .map_err(Box::new)?;
        Ok(self)
    }

    pub fn conditioning_sets(&self) -> NativeGuiderConditioningSets<'_> {
        self.strategy.conditioning_sets()
    }

    pub fn owned_resident_bytes(&self) -> Result<u64, NativeSamplerPayloadError> {
        guider_owned_resident_bytes(self.semantic_digest_sha256.capacity())
    }

    pub fn resident_parts(
        &self,
    ) -> Result<crate::NativeDiffusionResidentParts, crate::NativeDiffusionPayloadError> {
        crate::NativeDiffusionResidentParts::for_guider(self)
    }

    pub fn model_execution_digest_sha256(&self) -> &str {
        self.model.identity().execution_sha256()
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn guidance(&self) -> f32 {
        match &self.strategy {
            NativeGuiderStrategy::Basic { .. } => 1.0,
            NativeGuiderStrategy::Cfg { guidance, .. } => *guidance,
            NativeGuiderStrategy::DualCfg {
                second_guidance, ..
            } => *second_guidance,
            NativeGuiderStrategy::DualModel { guidance, .. } => *guidance,
        }
    }

    pub fn validate(&self) -> Result<(), NativeSamplerPayloadError> {
        let expected = Self::checked(self.model.clone(), self.strategy.clone())?;
        let expected = match &self.execution_payload {
            Some(payload) => expected.with_execution_payload(payload.clone())?,
            None => expected,
        };
        let expected = match &self.negative_execution_payload {
            Some(payload) => expected.with_negative_execution_payload(payload.clone())?,
            None => expected,
        };
        if expected.semantic_digest_sha256 != self.semantic_digest_sha256
            || expected.resident_bytes != self.resident_bytes
        {
            return Err(NativeSamplerPayloadError::ProjectionChanged("guider"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        sigma: f32,
        profile: &dyn SamplingProfile,
        plan: &SamplingPlan,
        options: GuidanceOptions,
        denoiser: &mut dyn GuidanceDenoiser,
        hooks: &mut [&mut dyn GuidanceHook],
        context: &ExecutionContext<'_>,
    ) -> Result<GuidanceResult, NativeSamplerPayloadError> {
        context.check()?;
        self.validate()?;
        if plan.guidance().to_bits() != self.guidance().to_bits() {
            return Err(NativeSamplerPayloadError::GuiderPlanMismatch);
        }
        Self::execute_strategy(
            &self.strategy,
            backend,
            latent,
            sigma,
            profile,
            plan,
            options,
            denoiser,
            hooks,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_dual_model(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        sigma: f32,
        profile: &dyn SamplingProfile,
        plan: &SamplingPlan,
        options: GuidanceOptions,
        positive_denoiser: &mut dyn GuidanceDenoiser,
        negative_denoiser: &mut dyn GuidanceDenoiser,
        context: &ExecutionContext<'_>,
    ) -> Result<GuidanceResult, NativeSamplerPayloadError> {
        context.check()?;
        self.validate()?;
        let NativeGuiderStrategy::DualModel {
            positive,
            negative,
            guidance,
            ..
        } = &self.strategy
        else {
            return Err(NativeSamplerPayloadError::DualModelDenoiserRequired);
        };
        if plan.guidance().to_bits() != guidance.to_bits() {
            return Err(NativeSamplerPayloadError::GuiderPlanMismatch);
        }
        Ok(execute_dual_model_guidance(
            backend,
            latent,
            sigma,
            profile,
            plan,
            positive,
            negative,
            options,
            positive_denoiser,
            negative_denoiser,
            context,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_modifier_session(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        sigma: f32,
        profile: &dyn SamplingProfile,
        plan: &SamplingPlan,
        options: GuidanceOptions,
        denoiser: &mut dyn GuidanceDenoiser,
        modifier_session: &mut crate::NativeGuidanceModifierSession<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<GuidanceResult, NativeSamplerPayloadError> {
        context.check()?;
        self.validate()?;
        if plan.guidance().to_bits() != self.guidance().to_bits() {
            return Err(NativeSamplerPayloadError::GuiderPlanMismatch);
        }
        let hook_graph = self
            .execution_payload
            .as_ref()
            .ok_or(NativeSamplerPayloadError::GuiderExecutionPayloadRequired)?
            .hook_graph();
        match &self.strategy {
            NativeGuiderStrategy::Basic { conditioning } => {
                Ok(execute_guidance_with_modifier_session(
                    backend,
                    latent,
                    sigma,
                    profile,
                    plan,
                    conditioning,
                    conditioning,
                    options,
                    denoiser,
                    hook_graph,
                    modifier_session,
                    context,
                )?)
            }
            NativeGuiderStrategy::Cfg {
                positive, negative, ..
            } => Ok(execute_guidance_with_modifier_session(
                backend,
                latent,
                sigma,
                profile,
                plan,
                positive,
                negative,
                options,
                denoiser,
                hook_graph,
                modifier_session,
                context,
            )?),
            NativeGuiderStrategy::DualCfg {
                positive,
                middle,
                negative,
                first_guidance,
                second_guidance,
                nested,
            } => Ok(execute_dual_cfg_guidance_with_modifier_session(
                backend,
                latent,
                sigma,
                profile,
                plan,
                positive,
                middle,
                negative,
                *first_guidance,
                *second_guidance,
                *nested,
                options,
                denoiser,
                hook_graph,
                modifier_session,
                context,
            )?),
            NativeGuiderStrategy::DualModel { .. } => {
                Err(NativeSamplerPayloadError::DualModelDenoiserRequired)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_dual_model_with_modifier_session(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        sigma: f32,
        profile: &dyn SamplingProfile,
        plan: &SamplingPlan,
        options: GuidanceOptions,
        positive_denoiser: &mut dyn GuidanceDenoiser,
        negative_denoiser: &mut dyn GuidanceDenoiser,
        modifier_session: &mut crate::NativeGuidanceModifierSession<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<GuidanceResult, NativeSamplerPayloadError> {
        context.check()?;
        self.validate()?;
        let NativeGuiderStrategy::DualModel {
            positive,
            negative,
            guidance,
            ..
        } = &self.strategy
        else {
            return Err(NativeSamplerPayloadError::DualModelDenoiserRequired);
        };
        if plan.guidance().to_bits() != guidance.to_bits() {
            return Err(NativeSamplerPayloadError::GuiderPlanMismatch);
        }
        let positive_hook_graph = self
            .execution_payload
            .as_ref()
            .ok_or(NativeSamplerPayloadError::GuiderExecutionPayloadRequired)?
            .hook_graph();
        let negative_hook_graph = self
            .negative_execution_payload
            .as_ref()
            .map_or(positive_hook_graph, |payload| payload.hook_graph());
        Ok(execute_dual_model_guidance_with_modifier_session(
            backend,
            latent,
            sigma,
            profile,
            plan,
            positive,
            negative,
            options,
            positive_denoiser,
            negative_denoiser,
            positive_hook_graph,
            negative_hook_graph,
            modifier_session,
            context,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_strategy(
        strategy: &NativeGuiderStrategy,
        backend: &CpuBackend,
        latent: &Tensor,
        sigma: f32,
        profile: &dyn SamplingProfile,
        plan: &SamplingPlan,
        options: GuidanceOptions,
        denoiser: &mut dyn GuidanceDenoiser,
        hooks: &mut [&mut dyn GuidanceHook],
        context: &ExecutionContext<'_>,
    ) -> Result<GuidanceResult, NativeSamplerPayloadError> {
        let (positive, negative) = match strategy {
            NativeGuiderStrategy::Basic { conditioning } => {
                (conditioning.as_ref(), conditioning.as_ref())
            }
            NativeGuiderStrategy::Cfg {
                positive, negative, ..
            } => (positive.as_ref(), negative.as_ref()),
            NativeGuiderStrategy::DualCfg {
                positive,
                middle,
                negative,
                first_guidance,
                second_guidance,
                nested,
            } => {
                return Ok(execute_dual_cfg_guidance(
                    backend,
                    latent,
                    sigma,
                    profile,
                    plan,
                    positive,
                    middle,
                    negative,
                    *first_guidance,
                    *second_guidance,
                    *nested,
                    options,
                    denoiser,
                    hooks,
                    context,
                )?);
            }
            NativeGuiderStrategy::DualModel { .. } => {
                return Err(NativeSamplerPayloadError::DualModelDenoiserRequired);
            }
        };
        Ok(execute_guidance(
            backend, latent, sigma, profile, plan, positive, negative, options, denoiser, hooks,
            context,
        )?)
    }

    fn checked(
        model: Arc<NativeModelPayload>,
        strategy: NativeGuiderStrategy,
    ) -> Result<Self, NativeSamplerPayloadError> {
        if model.identity().role() != NativeModelResourceRole::Model || model.model().is_none() {
            return Err(NativeSamplerPayloadError::GuiderModelRoleMismatch);
        }
        model.validate()?;
        let executable = model
            .model()
            .ok_or(NativeSamplerPayloadError::GuiderModelRoleMismatch)?;
        let profile = profile_for_model(&model)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-guider-payload.v1");
        hash_field(&mut hasher, model.identity().digest_sha256().as_bytes())?;
        hash_field(&mut hasher, executable.execution_digest().as_bytes())?;
        hash_field(&mut hasher, profile.identity().as_str().as_bytes())?;

        match &strategy {
            NativeGuiderStrategy::Basic { conditioning } => {
                hasher.update(b"basic");
                validate_conditioning_for_model(&model, conditioning.identity())?;
                validate_conditioning_digest(conditioning)?;
                hash_field(&mut hasher, conditioning.digest().as_bytes())?;
            }
            NativeGuiderStrategy::Cfg {
                positive,
                negative,
                guidance,
            } => {
                if !guidance.is_finite() || *guidance < 0.0 {
                    return Err(NativeSamplerPayloadError::InvalidGuidance(*guidance));
                }
                validate_conditioning_for_model(&model, positive.identity())?;
                validate_conditioning_for_model(&model, negative.identity())?;
                validate_conditioning_digest(positive)?;
                validate_conditioning_digest(negative)?;
                hasher.update(b"cfg");
                hasher.update(guidance.to_bits().to_le_bytes());
                hash_field(&mut hasher, positive.digest().as_bytes())?;
                hash_field(&mut hasher, negative.digest().as_bytes())?;
            }
            NativeGuiderStrategy::DualCfg {
                positive,
                middle,
                negative,
                first_guidance,
                second_guidance,
                nested,
            } => {
                if !first_guidance.is_finite()
                    || *first_guidance < 0.0
                    || !second_guidance.is_finite()
                    || *second_guidance < 0.0
                {
                    return Err(NativeSamplerPayloadError::InvalidGuidance(
                        first_guidance.min(*second_guidance),
                    ));
                }
                for conditioning in [positive, middle, negative] {
                    validate_conditioning_for_model(&model, conditioning.identity())?;
                    validate_conditioning_digest(conditioning)?;
                }
                hasher.update(b"dual-cfg");
                hasher.update(first_guidance.to_bits().to_le_bytes());
                hasher.update(second_guidance.to_bits().to_le_bytes());
                hasher.update([u8::from(*nested)]);
                for conditioning in [positive, middle, negative] {
                    hash_field(&mut hasher, conditioning.digest().as_bytes())?;
                }
            }
            NativeGuiderStrategy::DualModel {
                positive,
                negative,
                negative_model,
                guidance,
            } => {
                if !guidance.is_finite() || *guidance < 0.0 {
                    return Err(NativeSamplerPayloadError::InvalidGuidance(*guidance));
                }
                negative_model.validate()?;
                validate_conditioning_for_model(&model, positive.identity())?;
                validate_conditioning_for_model(negative_model, negative.identity())?;
                validate_conditioning_digest(positive)?;
                validate_conditioning_digest(negative)?;
                let positive_profile = profile_for_model(&model)?;
                let negative_profile = profile_for_model(negative_model)?;
                if positive_profile.policy() != negative_profile.policy() {
                    return Err(NativeSamplerPayloadError::IncompatibleDualModel);
                }
                hasher.update(b"dual-model");
                hasher.update(guidance.to_bits().to_le_bytes());
                hash_field(
                    &mut hasher,
                    negative_model.identity().digest_sha256().as_bytes(),
                )?;
                hash_field(&mut hasher, positive.digest().as_bytes())?;
                hash_field(&mut hasher, negative.digest().as_bytes())?;
            }
        }
        let semantic_digest_sha256 = format!("{:x}", hasher.finalize());

        let mut payload = Self {
            model,
            strategy,
            execution_payload: None,
            negative_execution_payload: None,
            semantic_digest_sha256,
            resident_bytes: 0,
        };
        payload.resident_bytes = payload
            .resident_parts()
            .and_then(|parts| parts.resident_bytes())
            .map_err(Box::new)?;
        Ok(payload)
    }
}

fn validate_conditioning_for_model(
    model: &NativeModelPayload,
    conditioning: &ConditioningIdentity,
) -> Result<(), NativeSamplerPayloadError> {
    let (expected_family, expected_latent) = match model.native_family_model_resource() {
        Some(resource) => (
            resource
                .family_identity()
                .map_err(|error| NativeSamplerPayloadError::FamilyModel(error.to_string()))?,
            resource.conditioning_identity().latent_format().clone(),
        ),
        None => (
            sd15_model_family_identity()
                .map_err(|error| NativeSamplerPayloadError::FamilyModel(error.to_string()))?,
            sd15_latent_format_identity()
                .map_err(|error| NativeSamplerPayloadError::FamilyModel(error.to_string()))?,
        ),
    };
    if conditioning.model_family() != &expected_family
        || conditioning.latent_format() != &expected_latent
    {
        return Err(NativeSamplerPayloadError::ConditioningIdentityMismatch);
    }
    Ok(())
}

impl NativeGuiderStrategy {
    fn conditioning_sets(&self) -> NativeGuiderConditioningSets<'_> {
        match self {
            Self::Basic { conditioning } => NativeGuiderConditioningSets::Basic { conditioning },
            Self::Cfg {
                positive, negative, ..
            } => NativeGuiderConditioningSets::Cfg { positive, negative },
            Self::DualCfg {
                positive,
                middle,
                negative,
                ..
            } => NativeGuiderConditioningSets::DualCfg {
                positive,
                middle,
                negative,
            },
            Self::DualModel {
                positive,
                negative,
                negative_model,
                ..
            } => NativeGuiderConditioningSets::DualModel {
                positive,
                negative,
                negative_model,
            },
        }
    }
}

fn guider_owned_resident_bytes(
    semantic_digest_capacity: usize,
) -> Result<u64, NativeSamplerPayloadError> {
    checked_add(
        usize_to_u64(mem::size_of::<NativeGuiderPayload>())?,
        usize_to_u64(semantic_digest_capacity)?,
    )
}

pub trait NativeSamplerDenoiser {
    fn denoise(&mut self, input: &Tensor, sigma: f32, step: usize) -> Result<Tensor, String>;

    fn denoise_cfg_pp(
        &mut self,
        _input: &Tensor,
        _sigma: f32,
        _step: usize,
    ) -> Result<CfgPpDenoiserOutput, String> {
        Err("the denoiser does not expose unconditional CFG++ output".to_owned())
    }

    fn denoise_with_replacement_noise(
        &mut self,
        input: &Tensor,
        _replacement_noise: &Tensor,
        sigma: f32,
        step: usize,
    ) -> Result<Tensor, String> {
        self.denoise(input, sigma, step)
    }
}

impl<Function> NativeSamplerDenoiser for Function
where
    Function: FnMut(&Tensor, f32, usize) -> Result<Tensor, String>,
{
    fn denoise(&mut self, input: &Tensor, sigma: f32, step: usize) -> Result<Tensor, String> {
        self(input, sigma, step)
    }
}

#[derive(Clone, Debug)]
pub enum NativeSamplerExecution {
    Fixed(SamplingTrace),
    Adaptive(crate::generated_dpm_adaptive_sim_model_0164::DpmAdaptiveResult),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NativeSamplerProgress {
    Fixed(SamplingProgress),
    Adaptive(crate::AdaptiveSamplingProgress),
}

impl NativeSamplerProgress {
    pub const fn step(self) -> u32 {
        match self {
            Self::Fixed(progress) => progress.step,
            Self::Adaptive(progress) => progress.attempt,
        }
    }
}

impl NativeSamplerExecution {
    pub fn final_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Fixed(trace) => trace.latents.last(),
            Self::Adaptive(result) => Some(&result.output),
        }
    }
}

macro_rules! native_sampler_algorithms {
    ($($variant:ident => $identity:literal),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum NativeSamplerAlgorithm {
            $($variant),+
        }

        impl NativeSamplerAlgorithm {
            pub const fn identity(self) -> &'static str {
                match self {
                    $(Self::$variant => $identity),+
                }
            }

            pub fn from_identity(identity: &str) -> Result<Self, NativeSamplerPayloadError> {
                match identity {
                    $($identity => Ok(Self::$variant),)+
                    value => Err(NativeSamplerPayloadError::UnknownAlgorithm(value.to_owned())),
                }
            }

            pub const fn all() -> &'static [Self] {
                &[$(Self::$variant),+]
            }
        }
    };
}

native_sampler_algorithms! {
    Ddim => "ddim",
    Ddpm => "ddpm",
    Deis => "deis",
    Dpm2 => "dpm_2",
    Dpm2Ancestral => "dpm_2_ancestral",
    DpmAdaptive => "dpm_adaptive",
    DpmFast => "dpm_fast",
    Dpmpp2m => "dpmpp_2m",
    Dpmpp2mCfgPp => "dpmpp_2m_cfg_pp",
    Dpmpp2mSde => "dpmpp_2m_sde",
    Dpmpp2mSdeGpu => "dpmpp_2m_sde_gpu",
    Dpmpp2mSdeHeun => "dpmpp_2m_sde_heun",
    Dpmpp2mSdeHeunGpu => "dpmpp_2m_sde_heun_gpu",
    Dpmpp2sAncestral => "dpmpp_2s_ancestral",
    Dpmpp2sAncestralCfgPp => "dpmpp_2s_ancestral_cfg_pp",
    Dpmpp3mSde => "dpmpp_3m_sde",
    Dpmpp3mSdeGpu => "dpmpp_3m_sde_gpu",
    DpmppSde => "dpmpp_sde",
    DpmppSdeGpu => "dpmpp_sde_gpu",
    ErSde => "er_sde",
    Euler => "euler",
    EulerAncestral => "euler_ancestral",
    EulerAncestralCfgPp => "euler_ancestral_cfg_pp",
    EulerCfgPp => "euler_cfg_pp",
    ExpHeun2X0 => "exp_heun_2_x0",
    ExpHeun2X0Sde => "exp_heun_2_x0_sde",
    GradientEstimation => "gradient_estimation",
    GradientEstimationCfgPp => "gradient_estimation_cfg_pp",
    Heun => "heun",
    HeunPp2 => "heunpp2",
    Ipndm => "ipndm",
    IpndmV => "ipndm_v",
    Lcm => "lcm",
    Lms => "lms",
    ResMultistep => "res_multistep",
    ResMultistepAncestral => "res_multistep_ancestral",
    ResMultistepAncestralCfgPp => "res_multistep_ancestral_cfg_pp",
    ResMultistepCfgPp => "res_multistep_cfg_pp",
    SaSolver => "sa_solver",
    SaSolverPece => "sa_solver_pece",
    Seeds2 => "seeds_2",
    Seeds3 => "seeds_3",
    UniPc => "uni_pc",
    UniPcBh2 => "uni_pc_bh2",
    VoidDdim => "void_ddim",
    LcmUpscale => "lcm_upscale",
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NativeSamplerParameters {
    SourceDefaults,
    Euler(crate::generated_native_diffusion::EulerOptions),
    EulerCfgPp { version: NativeEulerCfgPpVersion },
    Dpm2(crate::generated_dpm_2_sim_model_0162::Dpm2Options),
    Dpm2Ancestral(crate::generated_dpm_2_ancestral_sim_model_0163::Dpm2AncestralOptions),
    DpmAdaptive(crate::generated_dpm_adaptive_sim_model_0164::DpmAdaptiveOptions),
    DpmFast(crate::generated_dpm_fast_sim_model_0165::DpmFastOptions),
    Dpmpp2mSde(crate::generated_dpmpp_2m_sde_sim_model_0168::Dpmpp2mSdeOptions),
    Dpmpp2mSdeGpu(
        crate::generated_dpmpp_2m_sde_gpu_sim_model_0169::Dpmpp2mSdeGpuOptions,
    ),
    EtaNoise { eta: f32, noise_scale: f32 },
    Dpmpp2sAncestral(
        crate::generated_dpmpp_2s_ancestral_sim_model_0172::Dpmpp2sAncestralOptions,
    ),
    Dpmpp2sAncestralCfgPp(
        crate::generated_dpmpp_2s_ancestral_cfg_pp_sim_model_0173::Dpmpp2sAncestralCfgPpOptions,
    ),
    Dpmpp3mSde(crate::generated_dpmpp_3m_sde_sim_model_0174::Dpmpp3mSdeOptions),
    DpmppSde(crate::generated_dpmpp_sde_sim_model_0176::DpmppSdeOptions),
    ErSde {
        solver: NativeErSdeSolver,
        eta: f32,
        noise_scale: f32,
        max_stage: u8,
    },
    EulerAncestral(
        crate::generated_euler_ancestral_sim_model_0180::EulerAncestralOptions,
    ),
    EulerAncestralCfgPp(
        crate::generated_euler_ancestral_cfg_pp_sim_model_0181::EulerAncestralCfgPpOptions,
    ),
    ExpHeun2X0(crate::generated_exp_heun_2_x0_sim_model_0183::ExpHeun2X0SolverType),
    ExpHeun2X0Sde(
        crate::generated_exp_heun_2_x0_sde_sim_model_0184::ExpHeun2X0SdeOptions,
    ),
    GradientEstimation(
        crate::generated_gradient_estimation_sim_model_0185::GradientEstimationOptions,
    ),
    Lcm(crate::generated_lcm_sim_model_0191::LcmOptions),
    Lms { order: u8 },
    ResMultistep(crate::generated_res_multistep_sim_model_0193::ResMultistepOptions),
    ResMultistepAncestral(
        crate::generated_res_multistep_ancestral_sim_model_0194::ResMultistepAncestralOptions,
    ),
    ResMultistepAncestralCfgPp(
        crate::generated_res_multistep_ancestral_cfg_pp_sim_model_0195::ResMultistepAncestralCfgPpOptions,
    ),
    SaSolver(NativeSaSolverParameters),
    SaSolverInterval {
        parameters: NativeSaSolverParameters,
        start_sigma: f32,
        end_sigma: f32,
    },
    Seeds2(crate::generated_seeds_2_sim_model_0199::Seeds2Options),
    Seeds3(crate::generated_seeds_3_sim_model_0200::Seeds3Options),
    LcmUpscale(crate::generated_lcm_upscale::LcmUpscaleOptions),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeErSdeSolver {
    ErSde,
    ReverseTimeSde,
    Ode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeEulerCfgPpVersion {
    Regular,
    Alternative,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeSaSolverParameters {
    pub eta: f32,
    pub sde_start_percent: f32,
    pub sde_end_percent: f32,
    pub noise_scale: f32,
    pub predictor_order: u8,
    pub corrector_order: u8,
    pub use_pece: bool,
    pub simple_order_2: bool,
}

impl NativeSamplerParameters {
    fn validate_for(
        &self,
        algorithm: NativeSamplerAlgorithm,
    ) -> Result<(), NativeSamplerPayloadError> {
        match (algorithm, self) {
            (NativeSamplerAlgorithm::Euler, Self::Euler(_))
            | (NativeSamplerAlgorithm::Heun, Self::Euler(_))
            | (NativeSamplerAlgorithm::HeunPp2, Self::Euler(_))
            | (NativeSamplerAlgorithm::Dpm2, Self::Dpm2(_))
            | (NativeSamplerAlgorithm::Dpm2Ancestral, Self::Dpm2Ancestral(_))
            | (NativeSamplerAlgorithm::DpmFast, Self::DpmFast(_))
            | (NativeSamplerAlgorithm::Dpmpp2mSde, Self::Dpmpp2mSde(_))
            | (NativeSamplerAlgorithm::Dpmpp2mSdeGpu, Self::Dpmpp2mSdeGpu(_))
            | (NativeSamplerAlgorithm::Dpmpp2sAncestral, Self::Dpmpp2sAncestral(_))
            | (NativeSamplerAlgorithm::Dpmpp2sAncestralCfgPp, Self::Dpmpp2sAncestralCfgPp(_))
            | (NativeSamplerAlgorithm::Dpmpp3mSde, Self::Dpmpp3mSde(_))
            | (
                NativeSamplerAlgorithm::DpmppSde | NativeSamplerAlgorithm::DpmppSdeGpu,
                Self::DpmppSde(_),
            )
            | (NativeSamplerAlgorithm::EulerAncestral, Self::EulerAncestral(_))
            | (NativeSamplerAlgorithm::EulerAncestralCfgPp, Self::EulerAncestralCfgPp(_))
            | (NativeSamplerAlgorithm::ExpHeun2X0, Self::ExpHeun2X0(_))
            | (NativeSamplerAlgorithm::ExpHeun2X0Sde, Self::ExpHeun2X0Sde(_))
            | (NativeSamplerAlgorithm::GradientEstimation, Self::GradientEstimation(_))
            | (NativeSamplerAlgorithm::GradientEstimationCfgPp, Self::GradientEstimation(_))
            | (NativeSamplerAlgorithm::Lcm, Self::Lcm(_))
            | (NativeSamplerAlgorithm::ResMultistep, Self::ResMultistep(_))
            | (NativeSamplerAlgorithm::ResMultistepCfgPp, Self::ResMultistep(_))
            | (NativeSamplerAlgorithm::ResMultistepAncestral, Self::ResMultistepAncestral(_))
            | (
                NativeSamplerAlgorithm::ResMultistepAncestralCfgPp,
                Self::ResMultistepAncestralCfgPp(_),
            )
            | (NativeSamplerAlgorithm::Seeds3, Self::Seeds3(_)) => {
                validate_parameter_projection(self)?;
                Ok(())
            }
            (NativeSamplerAlgorithm::DpmAdaptive, Self::DpmAdaptive(options)) => {
                let valid = matches!(options.order, 2 | 3)
                    && options.relative_tolerance.is_finite()
                    && options.relative_tolerance >= 0.0
                    && options.absolute_tolerance.is_finite()
                    && options.absolute_tolerance >= 0.0
                    && options.initial_step_size.is_finite()
                    && options.proportional_coefficient.is_finite()
                    && options.integral_coefficient.is_finite()
                    && options.derivative_coefficient.is_finite()
                    && options.acceptance_safety.is_finite()
                    && options.acceptance_safety >= 0.0
                    && options.eta.is_finite()
                    && options.noise_scale.is_finite()
                    && options.attempt_limit > 0;
                validate_parameter_projection(self)?;
                if valid {
                    Ok(())
                } else {
                    Err(NativeSamplerPayloadError::InvalidParameters(format!(
                        "{self:?}"
                    )))
                }
            }
            (NativeSamplerAlgorithm::Seeds2, Self::Seeds2(options)) => {
                let valid = options.eta.is_finite()
                    && options.noise_scale.is_finite()
                    && options.intermediate_step_ratio.is_finite()
                    && options.intermediate_step_ratio != 0.0;
                validate_parameter_projection(self)?;
                if valid {
                    Ok(())
                } else {
                    Err(NativeSamplerPayloadError::InvalidParameters(format!(
                        "{self:?}"
                    )))
                }
            }
            (NativeSamplerAlgorithm::EulerCfgPp, Self::EulerCfgPp { .. }) => Ok(()),
            (
                NativeSamplerAlgorithm::Dpmpp2mSdeHeun
                | NativeSamplerAlgorithm::Dpmpp2mSdeHeunGpu
                | NativeSamplerAlgorithm::Dpmpp3mSdeGpu,
                Self::EtaNoise { .. },
            ) => {
                validate_parameter_projection(self)?;
                Ok(())
            }
            (NativeSamplerAlgorithm::ErSde, Self::ErSde { .. }) => {
                validate_parameter_projection(self)?;
                Ok(())
            }
            (
                NativeSamplerAlgorithm::SaSolver | NativeSamplerAlgorithm::SaSolverPece,
                Self::SaSolver(_),
            ) => {
                validate_parameter_projection(self)?;
                Ok(())
            }
            (
                NativeSamplerAlgorithm::SaSolver | NativeSamplerAlgorithm::SaSolverPece,
                Self::SaSolverInterval {
                    parameters,
                    start_sigma,
                    end_sigma,
                },
            ) => {
                Self::SaSolver(*parameters).validate_for(algorithm)?;
                if start_sigma.is_finite()
                    && *start_sigma >= 0.0
                    && end_sigma.is_finite()
                    && *end_sigma >= 0.0
                {
                    validate_parameter_projection(self)
                } else {
                    Err(NativeSamplerPayloadError::InvalidParameters(format!(
                        "{self:?}"
                    )))
                }
            }
            (NativeSamplerAlgorithm::Lms, Self::Lms { order }) if (1..=100).contains(order) => {
                Ok(())
            }
            (NativeSamplerAlgorithm::LcmUpscale, Self::LcmUpscale(options)) => {
                crate::generated_lcm_upscale::LcmUpscaleOptions::new(
                    options.total_upscale(),
                    options.upscale_steps(),
                    options.method(),
                )?;
                Ok(())
            }
            (NativeSamplerAlgorithm::Lms, Self::SourceDefaults) => Ok(()),
            (NativeSamplerAlgorithm::LcmUpscale, Self::SourceDefaults) => Ok(()),
            (_, Self::SourceDefaults) => Ok(()),
            _ => Err(NativeSamplerPayloadError::ParameterMismatch {
                algorithm: algorithm.identity(),
                parameters: format!("{self:?}"),
            }),
        }
    }
}

fn validate_parameter_projection(
    parameters: &NativeSamplerParameters,
) -> Result<(), NativeSamplerPayloadError> {
    let projection = format!("{parameters:?}");
    if projection.len() > 2_048 {
        return Err(NativeSamplerPayloadError::ParameterProjectionTooLarge(
            projection.len(),
        ));
    }
    let finite = match parameters {
        NativeSamplerParameters::EtaNoise { eta, noise_scale } => {
            eta.is_finite() && *eta >= 0.0 && noise_scale.is_finite()
        }
        NativeSamplerParameters::ErSde {
            eta,
            noise_scale,
            max_stage,
            ..
        } => {
            eta.is_finite() && *eta >= 0.0 && noise_scale.is_finite() && (1..=3).contains(max_stage)
        }
        NativeSamplerParameters::SaSolver(options) => {
            options.eta.is_finite()
                && options.eta >= 0.0
                && options.noise_scale.is_finite()
                && (0.0..=1.0).contains(&options.sde_start_percent)
                && (0.0..=1.0).contains(&options.sde_end_percent)
                && options.predictor_order >= 1
                && options.predictor_order <= 6
                && options.corrector_order <= 6
        }
        _ => !projection.contains("NaN") && !projection.contains("inf"),
    };
    if finite {
        Ok(())
    } else {
        Err(NativeSamplerPayloadError::InvalidParameters(projection))
    }
}

#[derive(Clone)]
pub struct NativeSamplerPayload {
    identity: SamplerIdentity,
    implementation_digest_sha256: String,
    semantic_digest_sha256: String,
    resident_bytes: u64,
    algorithm: NativeSamplerAlgorithm,
    parameters: NativeSamplerParameters,
}

impl NativeSamplerPayload {
    pub fn checked(
        algorithm: NativeSamplerAlgorithm,
        parameters: NativeSamplerParameters,
    ) -> Result<Self, NativeSamplerPayloadError> {
        parameters.validate_for(algorithm)?;
        let sampler_identity = algorithm.identity();
        let registry = SamplerRegistry::foundational()?;
        let identity = SamplerIdentity::new(sampler_identity)?;
        let definition = registry.resolve(&identity)?;
        let mut implementation_hasher = Sha256::new();
        implementation_hasher.update(b"zed.sim.native-sampler-implementation.v2");
        hash_field(&mut implementation_hasher, definition.feature_id.as_bytes())?;
        hash_field(
            &mut implementation_hasher,
            definition.implementation_module.as_bytes(),
        )?;
        let implementation_digest_sha256 = format!("{:x}", implementation_hasher.finalize());
        let parameter_projection = format!("{parameters:?}");
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-sampler-payload.v2");
        hash_field(&mut hasher, identity.as_str().as_bytes())?;
        hash_field(&mut hasher, implementation_digest_sha256.as_bytes())?;
        hash_field(&mut hasher, parameter_projection.as_bytes())?;
        let semantic_digest_sha256 = format!("{:x}", hasher.finalize());
        let resident_bytes = [
            usize_to_u64(mem::size_of::<Self>())?,
            usize_to_u64(identity.as_str().len())?,
            usize_to_u64(implementation_digest_sha256.capacity())?,
            usize_to_u64(semantic_digest_sha256.capacity())?,
            usize_to_u64(mem::size_of::<NativeSamplerAlgorithm>())?,
            usize_to_u64(mem::size_of::<NativeSamplerParameters>())?,
            usize_to_u64(parameter_projection.len())?,
        ]
        .into_iter()
        .try_fold(0_u64, checked_add)?;
        Ok(Self {
            identity,
            implementation_digest_sha256,
            semantic_digest_sha256,
            resident_bytes,
            algorithm,
            parameters,
        })
    }

    pub fn euler() -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(
            NativeSamplerAlgorithm::Euler,
            NativeSamplerParameters::SourceDefaults,
        )
    }

    pub fn selected(identity: &str) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(
            NativeSamplerAlgorithm::from_identity(identity)?,
            NativeSamplerParameters::SourceDefaults,
        )
    }

    pub fn lms(order: u8) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(
            NativeSamplerAlgorithm::Lms,
            NativeSamplerParameters::Lms { order },
        )
    }

    pub fn void_ddim() -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(
            NativeSamplerAlgorithm::VoidDdim,
            NativeSamplerParameters::SourceDefaults,
        )
    }

    pub fn lcm_upscale(
        options: crate::generated_lcm_upscale::LcmUpscaleOptions,
    ) -> Result<Self, NativeSamplerPayloadError> {
        Self::checked(
            NativeSamplerAlgorithm::LcmUpscale,
            NativeSamplerParameters::LcmUpscale(options),
        )
    }

    pub fn identity(&self) -> &SamplerIdentity {
        &self.identity
    }

    pub fn implementation_digest_sha256(&self) -> &str {
        &self.implementation_digest_sha256
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub const fn algorithm(&self) -> NativeSamplerAlgorithm {
        self.algorithm
    }

    pub fn parameters(&self) -> &NativeSamplerParameters {
        &self.parameters
    }

    pub fn validate(&self) -> Result<(), NativeSamplerPayloadError> {
        let expected = Self::checked(self.algorithm, self.parameters)?;
        if expected.identity != self.identity
            || expected.implementation_digest_sha256 != self.implementation_digest_sha256
            || expected.semantic_digest_sha256 != self.semantic_digest_sha256
            || expected.resident_bytes != self.resident_bytes
            || expected.parameters != self.parameters
        {
            return Err(NativeSamplerPayloadError::ProjectionChanged("sampler"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_checked(
        &self,
        backend: &CpuBackend,
        plan: SamplingPlan,
        profile: &impl SamplingProfile,
        initial: Tensor,
        sigmas: &[f32],
        noise_request: CompatibilityNoiseRequest,
        context: &ExecutionContext<'_>,
        denoiser: &mut dyn NativeSamplerDenoiser,
        callback: &mut dyn FnMut(&NativeSamplerProgress, &Tensor, &Tensor) -> Result<(), String>,
    ) -> Result<NativeSamplerExecution, NativeSamplerPayloadError> {
        context.check()?;
        self.validate()?;
        if plan.sampler() != &self.identity {
            return Err(NativeSamplerPayloadError::PlanAlgorithmMismatch {
                payload: self.identity.as_str().to_owned(),
                plan: plan.sampler().as_str().to_owned(),
            });
        }
        plan.validate(
            &SamplerRegistry::foundational()?,
            &crate::SchedulerRegistry::foundational()?,
            profile.identity(),
        )?;

        macro_rules! fixed {
            ($expression:expr) => {{
                let trace = $expression.map_err(|error| NativeSamplerPayloadError::Algorithm {
                    algorithm: self.algorithm.identity(),
                    reason: error.to_string(),
                })?;
                NativeSamplerExecution::Fixed(trace)
            }};
        }
        macro_rules! first {
            ($expression:expr) => {{
                let (trace, ..) =
                    $expression.map_err(|error| NativeSamplerPayloadError::Algorithm {
                        algorithm: self.algorithm.identity(),
                        reason: error.to_string(),
                    })?;
                NativeSamplerExecution::Fixed(trace)
            }};
        }
        let mut basic =
            |input: &Tensor, sigma: f32, step: usize| denoiser.denoise(input, sigma, step);
        let execution = match self.algorithm {
            NativeSamplerAlgorithm::Ddim => {
                let replacement = crate::generated_ddim_sim_model_0159::ddim_inpaint_replacement_noise(
                    backend,
                    initial.descriptor().shape(),
                    noise_request,
                    i128::from(plan.seed()),
                    context,
                )
                .map_err(|error| NativeSamplerPayloadError::Algorithm {
                    algorithm: self.algorithm.identity(),
                    reason: error.to_string(),
                })?;
                first!(crate::generated_ddim_sim_model_0159::sample_ddim(
                    backend,
                    plan,
                    initial,
                    sigmas,
                    replacement,
                    context,
                    |input, replacement_noise, sigma, step| denoiser
                        .denoise_with_replacement_noise(input, replacement_noise, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                ))
            }
            NativeSamplerAlgorithm::Ddpm => first!(
                crate::generated_ddpm_sim_model_0160::sample_ddpm(
                    backend,
                    plan,
                    initial,
                    sigmas,
                    noise_request,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Deis => fixed!(
                crate::generated_deis_sim_model_0161::sample_deis(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpm2 => first!(
                crate::generated_dpm_2_sim_model_0162::sample_dpm_2(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::Dpm2(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpm2Ancestral => first!(
                crate::generated_dpm_2_ancestral_sim_model_0163::sample_dpm_2_ancestral(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::Dpm2Ancestral(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::DpmAdaptive => {
                let result = crate::generated_dpm_adaptive_sim_model_0164::sample_dpm_adaptive(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::DpmAdaptive(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    |input, sigma, evaluation| {
                        denoiser.denoise(input, sigma, evaluation.attempt as usize)
                    },
                    |progress, latent, output| {
                        callback(
                            &NativeSamplerProgress::Adaptive(*progress),
                            latent,
                            output,
                        )
                    },
                )
                .map_err(|error| NativeSamplerPayloadError::Algorithm {
                    algorithm: self.algorithm.identity(),
                    reason: error.to_string(),
                })?;
                NativeSamplerExecution::Adaptive(result)
            }
            NativeSamplerAlgorithm::DpmFast => first!(
                crate::generated_dpm_fast_sim_model_0165::sample_dpm_fast(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::DpmFast(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    |input, sigma, interval, _| denoiser.denoise(input, sigma, interval as usize),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp2m => fixed!(
                crate::generated_dpmpp_2m_sim_model_0166::sample_dpmpp_2m(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp2mCfgPp => fixed!(
                crate::generated_dpmpp_2m_cfg_pp_sim_model_0167::sample_dpmpp_2m_cfg_pp(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    context,
                    |input, sigma, step| denoiser.denoise_cfg_pp(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp2mSde => first!(
                crate::generated_dpmpp_2m_sde_sim_model_0168::sample_dpmpp_2m_sde(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::Dpmpp2mSde(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp2mSdeGpu => first!(
                crate::generated_dpmpp_2m_sde_gpu_sim_model_0169::sample_dpmpp_2m_sde_gpu(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::Dpmpp2mSdeGpu(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp2mSdeHeun => first!(
                crate::generated_dpmpp_2m_sde_heun_sim_model_0170::sample_dpmpp_2m_sde_heun(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::EtaNoise { eta, .. } => eta,
                        _ => 1.0,
                    },
                    match self.parameters {
                        NativeSamplerParameters::EtaNoise { noise_scale, .. } => noise_scale,
                        _ => 1.0,
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp2mSdeHeunGpu => first!(
                crate::generated_dpmpp_2m_sde_heun_gpu_sim_model_0171::sample_dpmpp_2m_sde_heun_gpu(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::EtaNoise { eta, .. } => eta,
                        _ => 1.0,
                    },
                    match self.parameters {
                        NativeSamplerParameters::EtaNoise { noise_scale, .. } => noise_scale,
                        _ => 1.0,
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp2sAncestral => first!(
                crate::generated_dpmpp_2s_ancestral_sim_model_0172::sample_dpmpp_2s_ancestral(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::Dpmpp2sAncestral(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp2sAncestralCfgPp => first!(
                crate::generated_dpmpp_2s_ancestral_cfg_pp_sim_model_0173::sample_dpmpp_2s_ancestral_cfg_pp(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::Dpmpp2sAncestralCfgPp(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step, _| denoiser.denoise_cfg_pp(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp3mSde => first!(
                crate::generated_dpmpp_3m_sde_sim_model_0174::sample_dpmpp_3m_sde(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::Dpmpp3mSde(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Dpmpp3mSdeGpu => first!(
                crate::generated_dpmpp_3m_sde_gpu_sim_model_0175::sample_dpmpp_3m_sde_gpu(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::EtaNoise { eta, .. } => eta,
                        _ => 1.0,
                    },
                    match self.parameters {
                        NativeSamplerParameters::EtaNoise { noise_scale, .. } => noise_scale,
                        _ => 1.0,
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::DpmppSde => first!(
                crate::generated_dpmpp_sde_sim_model_0176::sample_dpmpp_sde(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::DpmppSde(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::DpmppSdeGpu => first!(
                crate::generated_dpmpp_sde_gpu_sim_model_0177::sample_dpmpp_sde_gpu(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::DpmppSde(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::ErSde => {
                let (solver, eta, options) = match self.parameters {
                    NativeSamplerParameters::ErSde {
                        solver,
                        eta,
                        noise_scale,
                        max_stage,
                    } => (
                        solver,
                        eta,
                        crate::generated_er_sde_sim_model_0178::ErSdeOptions {
                            noise_scale: if solver == NativeErSdeSolver::Ode {
                                0.0
                            } else {
                                noise_scale
                            },
                            max_stage: usize::from(max_stage),
                        },
                    ),
                    _ => (
                        NativeErSdeSolver::ErSde,
                        1.0,
                        Default::default(),
                    ),
                };
                if solver == NativeErSdeSolver::ErSde {
                    first!(crate::generated_er_sde_sim_model_0178::sample_er_sde(
                        backend,
                        plan,
                        profile,
                        initial,
                        sigmas,
                        noise_request,
                        options,
                        context,
                        &mut basic,
                        |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                    ))
                } else {
                    let exponent = if solver == NativeErSdeSolver::Ode {
                        1.0
                    } else {
                        eta + 1.0
                    };
                    first!(crate::generated_er_sde_sim_model_0178::sample_er_sde_with_noise_scaler(
                        backend,
                        plan,
                        profile,
                        initial,
                        sigmas,
                        noise_request,
                        options,
                        context,
                        &mut basic,
                        |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                        |sigma| Ok::<f32, String>(sigma.powf(exponent)),
                    ))
                }
            }
            NativeSamplerAlgorithm::Euler => first!(
                crate::generated_euler_sim_model_0179::sample_euler_sim_model_0179(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::Euler(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::EulerAncestral => first!(
                crate::generated_euler_ancestral_sim_model_0180::sample_euler_ancestral(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::EulerAncestral(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::EulerAncestralCfgPp => first!(
                crate::generated_euler_ancestral_cfg_pp_sim_model_0181::sample_euler_ancestral_cfg_pp(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::EulerAncestralCfgPp(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step| denoiser.denoise_cfg_pp(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::EulerCfgPp => {
                let trace = match self.parameters {
                    NativeSamplerParameters::EulerCfgPp {
                        version: NativeEulerCfgPpVersion::Alternative,
                    } => crate::generated_euler_cfg_pp_alternative::sample_euler_cfg_pp_alternative(
                        backend,
                        plan,
                        initial,
                        sigmas,
                        context,
                        |input, sigma, step| denoiser.denoise_cfg_pp(input, sigma, step),
                        |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                    )
                    .map_err(|error| error.to_string()),
                    _ => crate::generated_euler_cfg_pp_sim_model_0182::sample_euler_cfg_pp(
                        backend,
                        plan,
                        profile,
                        initial,
                        sigmas,
                        noise_request,
                        context,
                        |input, sigma, step| denoiser.denoise_cfg_pp(input, sigma, step),
                        |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                    ).map(|(trace, _, _)| trace).map_err(|error| error.to_string()),
                }
                .map_err(|reason| NativeSamplerPayloadError::Algorithm {
                    algorithm: self.algorithm.identity(),
                    reason,
                })?;
                NativeSamplerExecution::Fixed(trace)
            }
            NativeSamplerAlgorithm::ExpHeun2X0 => fixed!(
                crate::generated_exp_heun_2_x0_sim_model_0183::sample_exp_heun_2_x0(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::ExpHeun2X0(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::ExpHeun2X0Sde => first!(
                crate::generated_exp_heun_2_x0_sde_sim_model_0184::sample_exp_heun_2_x0_sde(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::ExpHeun2X0Sde(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::GradientEstimation => fixed!(
                crate::generated_gradient_estimation_sim_model_0185::sample_gradient_estimation(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::GradientEstimation(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::GradientEstimationCfgPp => fixed!(
                crate::generated_gradient_estimation_cfg_pp_sim_model_0186::sample_gradient_estimation_cfg_pp(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::GradientEstimation(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step| denoiser.denoise_cfg_pp(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Heun => first!(
                crate::generated_heun_sim_model_0187::sample_heun(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::Euler(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::HeunPp2 => first!(
                crate::generated_heunpp2_sim_model_0188::sample_heunpp2(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    match self.parameters {
                        NativeSamplerParameters::Euler(options) => options,
                        _ => Default::default(),
                    },
                    noise_request,
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Ipndm => fixed!(
                crate::generated_ipndm_sim_model_0189::sample_ipndm(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::IpndmV => fixed!(
                crate::generated_ipndm_v_sim_model_0190::sample_ipndm_v(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Lcm => first!(
                crate::generated_lcm_sim_model_0191::sample_lcm(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::Lcm(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Lms => {
                let order = match self.parameters {
                    NativeSamplerParameters::Lms { order } => usize::from(order),
                    NativeSamplerParameters::SourceDefaults => 4,
                    _ => return Err(NativeSamplerPayloadError::ProjectionChanged("sampler")),
                };
                fixed!(crate::generated_lms_sim_model_0192::sample_lms_with_order(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    order,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                ))
            }
            NativeSamplerAlgorithm::ResMultistep => first!(
                crate::generated_res_multistep_sim_model_0193::sample_res_multistep(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::ResMultistep(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::ResMultistepAncestral => first!(
                crate::generated_res_multistep_ancestral_sim_model_0194::sample_res_multistep_ancestral(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::ResMultistepAncestral(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::ResMultistepAncestralCfgPp => first!(
                crate::generated_res_multistep_ancestral_cfg_pp_sim_model_0195::sample_res_multistep_ancestral_cfg_pp(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::ResMultistepAncestralCfgPp(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step| denoiser.denoise_cfg_pp(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::ResMultistepCfgPp => first!(
                crate::generated_res_multistep_cfg_pp_sim_model_0196::sample_res_multistep_cfg_pp(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::ResMultistep(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step| denoiser.denoise_cfg_pp(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::SaSolver | NativeSamplerAlgorithm::SaSolverPece => {
                let parameters = match self.parameters {
                    NativeSamplerParameters::SaSolver(parameters)
                    | NativeSamplerParameters::SaSolverInterval { parameters, .. } => parameters,
                    _ => NativeSaSolverParameters {
                        eta: 1.0,
                        sde_start_percent: 0.2,
                        sde_end_percent: 0.8,
                        noise_scale: 1.0,
                        predictor_order: 3,
                        corrector_order: 4,
                        use_pece: self.algorithm == NativeSamplerAlgorithm::SaSolverPece,
                        simple_order_2: false,
                    },
                };
                let options = crate::generated_sa_solver_sim_model_0197::SaSolverOptions::new(
                    parameters.noise_scale,
                    usize::from(parameters.predictor_order),
                    usize::from(parameters.corrector_order),
                    parameters.simple_order_2,
                )
                .map_err(|error| NativeSamplerPayloadError::Algorithm {
                    algorithm: self.algorithm.identity(),
                    reason: error.to_string(),
                })?;
                let (start_sigma, end_sigma) = match self.parameters {
                    NativeSamplerParameters::SaSolverInterval { start_sigma, end_sigma, .. } =>
                        (start_sigma, end_sigma),
                    _ => {
                        let maximum_time = profile.sigma_count().saturating_sub(1) as f32;
                        (
                            profile.sigma_at_model_time((1.0 - parameters.sde_start_percent) * maximum_time)?,
                            profile.sigma_at_model_time((1.0 - parameters.sde_end_percent) * maximum_time)?,
                        )
                    }
                };
                let family = crate::generated_sa_solver_sim_model_0197::SaSolverFamilyOptions::new(
                    options,
                    parameters.use_pece
                        || self.algorithm == NativeSamplerAlgorithm::SaSolverPece,
                );
                first!(crate::generated_sa_solver_sim_model_0197::sample_sa_solver_family(
                    backend,
                    plan,
                    profile,
                    self.algorithm.identity(),
                    initial,
                    sigmas,
                    noise_request,
                    family,
                    context,
                    |sigma, _| {
                        if sigma <= start_sigma && sigma >= end_sigma {
                            Ok(parameters.eta)
                        } else {
                            Ok(0.0)
                        }
                    },
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                ))
            }
            NativeSamplerAlgorithm::Seeds2 => first!(
                crate::generated_seeds_2_sim_model_0199::sample_seeds_2(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::Seeds2(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::Seeds3 => first!(
                crate::generated_seeds_3_sim_model_0200::sample_seeds_3(
                    backend,
                    plan,
                    profile,
                    initial,
                    sigmas,
                    noise_request,
                    match self.parameters {
                        NativeSamplerParameters::Seeds3(options) => options,
                        _ => Default::default(),
                    },
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::UniPc => fixed!(
                crate::generated_uni_pc_sim_model_0201::sample_uni_pc(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::UniPcBh2 => fixed!(
                crate::generated_uni_pc_bh2_sim_model_0202::sample_uni_pc_bh2(
                    backend,
                    plan,
                    profile.identity(),
                    initial,
                    sigmas,
                    context,
                    |input, sigma, step, _| denoiser.denoise(input, sigma, step),
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::VoidDdim => fixed!(
                crate::generated_void_ddim::sample_void_ddim(
                    backend,
                    plan,
                    initial,
                    sigmas,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                )
            ),
            NativeSamplerAlgorithm::LcmUpscale => {
                let options = match self.parameters {
                    NativeSamplerParameters::LcmUpscale(options) => options,
                    NativeSamplerParameters::SourceDefaults => Default::default(),
                    _ => return Err(NativeSamplerPayloadError::ProjectionChanged("sampler")),
                };
                first!(crate::generated_lcm_upscale::sample_lcm_upscale(
                    backend,
                    plan,
                    initial,
                    sigmas,
                    noise_request,
                    options,
                    context,
                    &mut basic,
                    |progress, latent, output| callback(&NativeSamplerProgress::Fixed(*progress), latent, output),
                ))
            }
        };
        if execution.final_tensor().is_none() {
            return Err(NativeSamplerPayloadError::SamplerOutputMismatch);
        }
        context.check()?;
        Ok(execution)
    }
}

#[derive(Debug, Error)]
pub enum NativeSamplerPayloadError {
    #[error(transparent)]
    Residency(#[from] Box<crate::NativeDiffusionPayloadError>),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Noise(#[from] NoiseError),
    #[error(transparent)]
    Rng(#[from] RngError),
    #[error(transparent)]
    Guidance(#[from] GuidanceError),
    #[error(transparent)]
    Sampling(#[from] crate::SamplingError),
    #[error(transparent)]
    Scheduler(#[from] crate::SchedulerError),
    #[error(transparent)]
    NativeSampler(#[from] NativeDiffusionSamplerError),
    #[error(transparent)]
    LcmUpscale(#[from] crate::generated_lcm_upscale::LcmUpscaleError),
    #[error(transparent)]
    Model(#[from] NativeModelPayloadError),
    #[error(transparent)]
    Conditioning(#[from] ConditioningError),
    #[error(transparent)]
    Profile(#[from] SamplingProfileError),
    #[error("native family MODEL contract is invalid: {0}")]
    FamilyModel(String),
    #[error("native noise input must be a contiguous CPU f32 tensor with nonzero dimensions")]
    InvalidNoiseInput,
    #[error("native noise batch indices do not match the latent batch")]
    NoiseBatchMismatch,
    #[error("native noise batch index {0} exceeds the bounded skip limit")]
    NoiseBatchIndexLimit(u64),
    #[error("fixed native noise shape or target does not match the input latent")]
    NoiseShapeMismatch,
    #[error("native guider guidance must be finite and nonnegative, got {0}")]
    InvalidGuidance(f32),
    #[error("native guider positive and negative conditioning identities differ")]
    ConditioningIdentityMismatch,
    #[error("native guider conditioning digest is invalid")]
    InvalidConditioningDigest,
    #[error("native guider requires a concrete MODEL payload")]
    GuiderModelRoleMismatch,
    #[error("native guider guidance does not match the sampling plan")]
    GuiderPlanMismatch,
    #[error("dual-model guidance requires a separate negative-model denoiser")]
    DualModelDenoiserRequired,
    #[error("native guider has no retained MODEL execution payload")]
    GuiderExecutionPayloadRequired,
    #[error("dual-model guidance MODEL profiles are incompatible")]
    IncompatibleDualModel,
    #[error("native sampler output changed its requested sigma or tensor contract")]
    SamplerOutputMismatch,
    #[error("unknown native sampler algorithm {0:?}")]
    UnknownAlgorithm(String),
    #[error("native sampler {algorithm} does not accept parameter projection {parameters}")]
    ParameterMismatch {
        algorithm: &'static str,
        parameters: String,
    },
    #[error("native sampler parameter projection is too large: {0} bytes")]
    ParameterProjectionTooLarge(usize),
    #[error("native sampler parameters are invalid: {0}")]
    InvalidParameters(String),
    #[error("native sampler payload {payload:?} does not match sampling plan {plan:?}")]
    PlanAlgorithmMismatch { payload: String, plan: String },
    #[error("native sampler {algorithm} failed: {reason}")]
    Algorithm {
        algorithm: &'static str,
        reason: String,
    },
    #[error("native {0} payload projection changed")]
    ProjectionChanged(&'static str),
    #[error("native sampler payload resident byte accounting overflowed")]
    ResidentBytesOverflow,
}

fn generate_random_noise(
    backend: &CpuBackend,
    input_latent: &Tensor,
    batch_indices: Option<&[u64]>,
    request: &NoiseRequest,
    seed: u64,
    context: &ExecutionContext<'_>,
) -> Result<NativeNoiseGeneration, NativeSamplerPayloadError> {
    let stream = request.stream(seed, DeviceId::CPU)?;
    if batch_indices.is_none() {
        let trace = crate::noise::normal_noise(
            backend,
            input_latent.descriptor().shape(),
            &stream,
            context,
        )?;
        return Ok(NativeNoiseGeneration {
            noise: trace.noise,
            before: Some(trace.before),
            after: Some(trace.after),
        });
    }
    let batch_indices = batch_indices.ok_or(NativeSamplerPayloadError::NoiseBatchMismatch)?;
    let shape = input_latent.descriptor().shape();
    let batch = shape
        .first()
        .copied()
        .ok_or(NativeSamplerPayloadError::InvalidNoiseInput)?;
    if usize_to_u64(batch_indices.len())? != batch {
        return Err(NativeSamplerPayloadError::NoiseBatchMismatch);
    }
    let maximum = batch_indices
        .iter()
        .copied()
        .max()
        .ok_or(NativeSamplerPayloadError::NoiseBatchMismatch)?;
    if maximum > MAX_BATCH_NOISE_INDEX {
        return Err(NativeSamplerPayloadError::NoiseBatchIndexLimit(maximum));
    }
    let per_batch_count = shape
        .get(1..)
        .ok_or(NativeSamplerPayloadError::InvalidNoiseInput)?
        .iter()
        .try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(NativeSamplerPayloadError::ResidentBytesOverflow)
        })?;
    let per_batch_count = usize::try_from(per_batch_count)
        .map_err(|_| NativeSamplerPayloadError::ResidentBytesOverflow)?;
    let wanted = batch_indices.iter().copied().collect::<BTreeSet<_>>();
    let before = stream.begin(None)?.commit();
    let mut transaction = stream.begin(Some(before.clone()))?;
    transaction.require_device(DeviceId::CPU)?;
    let mut rows = BTreeMap::new();
    for index in 0..=maximum {
        context.check()?;
        let mut row = Vec::new();
        row.try_reserve_exact(per_batch_count)
            .map_err(|_| NativeSamplerPayloadError::ResidentBytesOverflow)?;
        while row.len() < per_batch_count {
            let pair = transaction.next_standard_normal_pair(context.cancellation)?;
            row.push(pair[0] as f32);
            if row.len() < per_batch_count {
                row.push(pair[1] as f32);
            }
        }
        if wanted.contains(&index) {
            rows.insert(index, row);
        }
    }
    let output_count = tensor_element_count(input_latent)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_count)
        .map_err(|_| NativeSamplerPayloadError::ResidentBytesOverflow)?;
    for index in batch_indices {
        values.extend_from_slice(
            rows.get(index)
                .ok_or(NativeSamplerPayloadError::NoiseBatchMismatch)?,
        );
    }
    let after = transaction.commit();
    Ok(NativeNoiseGeneration {
        noise: tensor_from_f32(backend, shape, &values, context)?,
        before: Some(before),
        after: Some(after),
    })
}

fn validate_noise_input(
    input: &Tensor,
    batch_indices: Option<&[u64]>,
) -> Result<(), NativeSamplerPayloadError> {
    validate_noise_tensor(input)?;
    if let Some(indices) = batch_indices {
        let batch = input
            .descriptor()
            .shape()
            .first()
            .copied()
            .ok_or(NativeSamplerPayloadError::InvalidNoiseInput)?;
        if indices.is_empty() || usize_to_u64(indices.len())? != batch {
            return Err(NativeSamplerPayloadError::NoiseBatchMismatch);
        }
    }
    Ok(())
}

fn validate_noise_tensor(tensor: &Tensor) -> Result<(), NativeSamplerPayloadError> {
    let descriptor = tensor.descriptor();
    if descriptor.dtype() != DType::F32
        || descriptor.device() != DeviceId::CPU
        || !descriptor.is_contiguous()?
        || descriptor.shape().is_empty()
        || descriptor.shape().contains(&0)
    {
        return Err(NativeSamplerPayloadError::InvalidNoiseInput);
    }
    Ok(())
}

fn tensor_element_count(tensor: &Tensor) -> Result<usize, NativeSamplerPayloadError> {
    usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| NativeSamplerPayloadError::ResidentBytesOverflow)
}

fn hash_tensor(hasher: &mut Sha256, tensor: &Tensor) -> Result<(), NativeSamplerPayloadError> {
    let descriptor = tensor.descriptor();
    hasher.update(b"zed.sim.native-noise-tensor.v1");
    hash_field(hasher, descriptor.dtype().catalog_name().as_bytes())?;
    hasher.update(usize_to_u64(descriptor.rank())?.to_le_bytes());
    for dimension in descriptor.shape() {
        hasher.update(dimension.to_le_bytes());
    }
    hash_field(hasher, tensor.contiguous_bytes()?)
}

fn validate_conditioning_digest(
    conditioning: &ConditioningSet,
) -> Result<(), NativeSamplerPayloadError> {
    conditioning.validate()?;
    if valid_sha256(conditioning.digest()) {
        Ok(())
    } else {
        Err(NativeSamplerPayloadError::InvalidConditioningDigest)
    }
}

fn hash_field(hasher: &mut Sha256, value: &[u8]) -> Result<(), NativeSamplerPayloadError> {
    hasher.update(usize_to_u64(value.len())?.to_le_bytes());
    hasher.update(value);
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn usize_to_u64(value: usize) -> Result<u64, NativeSamplerPayloadError> {
    u64::try_from(value).map_err(|_| NativeSamplerPayloadError::ResidentBytesOverflow)
}

fn checked_add(left: u64, right: u64) -> Result<u64, NativeSamplerPayloadError> {
    left.checked_add(right)
        .ok_or(NativeSamplerPayloadError::ResidentBytesOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiscreteSamplingProfile;
    use sim_model::{
        LatentFormatIdentity, ModelFamilyIdentity,
        conditioning::{
            ConditioningEntry, ConditioningEntryOptions, ConditioningIdentity, ConditioningValue,
        },
    };
    use sim_tensor::{
        CancellationToken, CpuWorkspaceAuthority, RetryRngPolicy, StreamId, TensorBackend,
        generated_native_diffusion::tensor_to_f32,
    };
    use std::{collections::BTreeSet, error::Error};

    fn context<'a>(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, TensorError> {
        Ok(backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(4 * 1024 * 1024)?,
            cancellation,
        ))
    }

    fn conditioning(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        cancellation: &CancellationToken,
        namespace: &str,
        value: f32,
    ) -> Result<Arc<ConditioningSet>, Box<dyn Error>> {
        let identity = ConditioningIdentity::new(
            namespace,
            ModelFamilyIdentity::new("COMFY-MODEL-0001", "payload-test", "v1")?,
            LatentFormatIdentity::new("COMFY-MODEL-0002", "payload_latent")?,
        )?;
        let tensor = tensor_from_f32(backend, &[1, 1, 2], &[value, value], context)?;
        Ok(Arc::new(ConditioningSet::checked(
            identity,
            vec![ConditioningEntry::checked(
                "full",
                ConditioningValue::regular(tensor)?,
                ConditioningEntryOptions::default(),
            )?],
            cancellation,
        )?))
    }

    struct ConstantDenoiser<'a> {
        backend: &'a CpuBackend,
    }

    struct SamplerTestDenoiser<'backend, 'context> {
        backend: &'backend CpuBackend,
        context: &'context ExecutionContext<'context>,
        guided: f32,
        unconditional: f32,
        calls: usize,
    }

    impl NativeSamplerDenoiser for SamplerTestDenoiser<'_, '_> {
        fn denoise(&mut self, input: &Tensor, _sigma: f32, _step: usize) -> Result<Tensor, String> {
            self.calls += 1;
            let count = usize::try_from(
                input
                    .descriptor()
                    .element_count()
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            tensor_from_f32(
                self.backend,
                input.descriptor().shape(),
                &vec![self.guided; count],
                self.context,
            )
            .map_err(|error| error.to_string())
        }

        fn denoise_cfg_pp(
            &mut self,
            input: &Tensor,
            _sigma: f32,
            _step: usize,
        ) -> Result<CfgPpDenoiserOutput, String> {
            self.calls += 1;
            let count = usize::try_from(
                input
                    .descriptor()
                    .element_count()
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            Ok(CfgPpDenoiserOutput {
                denoised: tensor_from_f32(
                    self.backend,
                    input.descriptor().shape(),
                    &vec![self.guided; count],
                    self.context,
                )
                .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    self.backend,
                    input.descriptor().shape(),
                    &vec![self.unconditional; count],
                    self.context,
                )
                .map_err(|error| error.to_string())?,
            })
        }

        fn denoise_with_replacement_noise(
            &mut self,
            input: &Tensor,
            _replacement_noise: &Tensor,
            sigma: f32,
            step: usize,
        ) -> Result<Tensor, String> {
            self.denoise(input, sigma, step)
        }
    }

    impl GuidanceDenoiser for ConstantDenoiser<'_> {
        fn evaluate_batch(
            &mut self,
            evaluations: &[crate::GuidanceEvaluation],
            context: &ExecutionContext<'_>,
        ) -> Result<Vec<Tensor>, GuidanceError> {
            evaluations
                .iter()
                .map(|evaluation| {
                    let value = match evaluation.branch() {
                        crate::GuidanceBranch::Conditional => 3.0,
                        crate::GuidanceBranch::Unconditional => 1.0,
                    };
                    tensor_from_f32(
                        self.backend,
                        evaluation.latent().descriptor().shape(),
                        &[value, value],
                        context,
                    )
                    .map_err(GuidanceError::from)
                })
                .collect()
        }
    }

    #[test]
    fn latent_source_noise_ignores_unrelated_input_and_clones_owned_storage()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let samples = tensor_from_f32(&backend, &[1, 1, 2, 2], &[0.25, -0.5, 1.0, 0.0], &context)?;
        let source_storage = samples.storage_id();
        let fixed = NativeNoisePayload::fixed_latent(0, samples.clone())?;
        let source = NativeNoisePayload::latent_source(samples.clone())?;
        assert_eq!(source.kind(), NativeNoiseSourceKind::LatentSource);
        assert_eq!(source.seed(), 0);
        assert_ne!(
            source.semantic_digest_sha256(),
            fixed.semantic_digest_sha256()
        );
        assert_eq!(
            source
                .resident_parts()?
                .tensor_allocation()
                .ok_or("missing source allocation")?
                .storage_id(),
            source_storage
        );
        let input = tensor_from_f32(&backend, &[], &[99.0], &context)?;
        let request = NoiseRequest::native_diffusion("prompt", "noise")?;
        let generated = source.generate(&backend, &input, Some(&[u64::MAX]), &request, &context)?;
        let repeated = source.generate(&backend, &input, Some(&[]), &request, &context)?;
        assert_eq!(generated.noise.descriptor().shape(), &[1, 1, 2, 2]);
        assert_ne!(generated.noise.storage_id(), source_storage);
        assert_ne!(generated.noise.storage_id(), repeated.noise.storage_id());
        assert!(generated.before.is_none() && generated.after.is_none());
        assert_eq!(
            &*tensor_to_f32(&backend, &generated.noise, &context)?,
            &[0.25, -0.5, 1.0, 0.0]
        );
        assert_eq!(
            &*tensor_to_f32(&backend, &samples, &context)?,
            &[0.25, -0.5, 1.0, 0.0]
        );
        assert!(
            fixed
                .generate(&backend, &input, Some(&[u64::MAX]), &request, &context)
                .is_err()
        );
        drop(generated);
        drop(repeated);
        let before = backend.memory_snapshot().current_bytes;
        cancellation.cancel();
        assert!(matches!(
            source.generate(&backend, &input, None, &request, &context),
            Err(NativeSamplerPayloadError::Tensor(TensorError::Cancelled))
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        Ok(())
    }

    #[test]
    fn latent_source_noise_requires_a_fresh_allocation_before_publication()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(32)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(32)?,
            &cancellation,
        );
        let samples = tensor_from_f32(&backend, &[1, 1, 2, 2], &[0.25; 4], &context)?;
        let (blocker, _) = backend.allocate(
            sim_tensor::TensorDescriptor::contiguous(
                vec![1],
                DType::F32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            &context,
        )?;
        let source = NativeNoisePayload::latent_source(samples.clone())?;
        let request = NoiseRequest::native_diffusion("prompt", "noise")?;
        let before = backend.memory_snapshot().current_bytes;
        assert_eq!(before, 32);
        assert!(matches!(
            source.generate(&backend, &blocker, None, &request, &context),
            Err(NativeSamplerPayloadError::Tensor(
                TensorError::AllocationFailed { requested: 16, .. }
            ))
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        drop(blocker);
        assert_eq!(&*tensor_to_f32(&backend, &samples, &context)?, &[0.25; 4]);
        Ok(())
    }

    #[test]
    fn noise_payloads_execute_deterministically_and_validate_batch_indices()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let latent = tensor_from_f32(&backend, &[3, 2], &[1.0; 6], &context)?;
        let request = NoiseRequest::native_diffusion("prompt", "noise")?;

        let zero = NativeNoisePayload::zero()?;
        let generated = zero.generate(&backend, &latent, None, &request, &context)?;
        assert_eq!(
            &*tensor_to_f32(&backend, &generated.noise, &context)?,
            &[0.0; 6]
        );
        assert!(generated.before.is_none() && generated.after.is_none());

        let random = NativeNoisePayload::random(42)?;
        let first = random.generate(&backend, &latent, Some(&[2, 0, 2]), &request, &context)?;
        let second = random.generate(&backend, &latent, Some(&[2, 0, 2]), &request, &context)?;
        let first_values = tensor_to_f32(&backend, &first.noise, &context)?;
        assert_eq!(
            &*first_values,
            &*tensor_to_f32(&backend, &second.noise, &context)?
        );
        assert_eq!(&first_values[0..2], &first_values[4..6]);
        assert_ne!(&first_values[0..2], &first_values[2..4]);
        assert!(first.before.is_some() && first.after.is_some());
        assert!(matches!(
            random.generate(&backend, &latent, Some(&[0]), &request, &context),
            Err(NativeSamplerPayloadError::NoiseBatchMismatch)
        ));
        assert_ne!(
            zero.semantic_digest_sha256(),
            random.semantic_digest_sha256()
        );
        assert!(random.resident_bytes() > 0);
        Ok(())
    }

    #[test]
    fn fixed_noise_semantic_digest_ignores_stream_and_storage_placement()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let default_context = context(&backend, &authority, &cancellation)?;
        let other_context = backend.execution_context(
            StreamId::new(17),
            authority.authorize_workspace(4 * 1024 * 1024)?,
            &cancellation,
        );
        let first_tensor = tensor_from_f32(&backend, &[1, 1, 2], &[0.25, -0.5], &default_context)?;
        let second_tensor = tensor_from_f32(&backend, &[1, 1, 2], &[0.25, -0.5], &other_context)?;
        let shared_tensor = first_tensor.clone();
        assert_ne!(first_tensor.storage_id(), second_tensor.storage_id());
        assert_ne!(
            first_tensor.descriptor().stream(),
            second_tensor.descriptor().stream()
        );

        let first = NativeNoisePayload::fixed_latent(19, first_tensor)?;
        let second = NativeNoisePayload::fixed_latent(19, second_tensor)?;
        let shared = NativeNoisePayload::fixed_latent(19, shared_tensor)?;
        let first_parts = first.resident_parts()?;
        let second_parts = second.resident_parts()?;
        let shared_parts = shared.resident_parts()?;
        assert_eq!(first_parts.owned_bytes(), shared_parts.owned_bytes());
        assert_eq!(
            first_parts.tensor_allocation(),
            shared_parts.tensor_allocation()
        );
        assert_ne!(
            first_parts
                .tensor_allocation()
                .ok_or("fixed noise lost its tensor allocation")?
                .storage_id(),
            second_parts
                .tensor_allocation()
                .ok_or("distinct fixed noise lost its tensor allocation")?
                .storage_id()
        );
        assert_eq!(first_parts.resident_bytes()?, first.resident_bytes());
        assert_eq!(
            first.semantic_digest_sha256(),
            second.semantic_digest_sha256()
        );
        assert_ne!(
            first.semantic_digest_sha256(),
            NativeNoisePayload::fixed_latent(
                19,
                tensor_from_f32(&backend, &[1, 1, 2], &[0.25, -0.25], &default_context)?,
            )?
            .semantic_digest_sha256()
        );
        assert_ne!(
            first.semantic_digest_sha256(),
            NativeNoisePayload::fixed_latent(
                19,
                tensor_from_f32(&backend, &[1, 2, 1], &[0.25, -0.5], &default_context)?,
            )?
            .semantic_digest_sha256()
        );
        Ok(())
    }

    #[test]
    fn cfg_guider_strategy_executes_checked_conditioning() -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let latent = tensor_from_f32(&backend, &[1, 1, 2], &[0.0, 0.0], &context)?;
        let positive = conditioning(&backend, &context, &cancellation, "shared", 1.0)?;
        let negative = conditioning(&backend, &context, &cancellation, "shared", 0.0)?;
        let positive_resident_bytes = positive.resident_bytes()?;
        assert_eq!(
            ConditioningSet::combined_resident_bytes(&[positive.as_ref(), positive.as_ref()])?,
            positive_resident_bytes
        );
        let cfg_resident_bytes =
            ConditioningSet::combined_resident_bytes(&[positive.as_ref(), negative.as_ref()])?;
        let positive_address = Arc::as_ptr(&positive);
        let negative_address = Arc::as_ptr(&negative);
        let strategy = NativeGuiderStrategy::Cfg {
            positive,
            negative,
            guidance: 2.0,
        };
        let NativeGuiderConditioningSets::Cfg { positive, negative } = strategy.conditioning_sets()
        else {
            return Err("CFG guider exposed basic conditioning children".into());
        };
        assert_eq!(Arc::as_ptr(positive), positive_address);
        assert_eq!(Arc::as_ptr(negative), negative_address);
        assert_eq!(
            ConditioningSet::combined_resident_bytes(&[positive.as_ref(), negative.as_ref(),])?,
            cfg_resident_bytes
        );
        let basic_strategy = NativeGuiderStrategy::Basic {
            conditioning: positive.clone(),
        };
        let NativeGuiderConditioningSets::Basic { conditioning } =
            basic_strategy.conditioning_sets()
        else {
            return Err("basic guider exposed CFG conditioning children".into());
        };
        assert_eq!(Arc::as_ptr(conditioning), positive_address);
        assert_eq!(conditioning.resident_bytes()?, positive_resident_bytes);
        assert_eq!(
            guider_owned_resident_bytes(128)?,
            usize_to_u64(mem::size_of::<NativeGuiderPayload>())?
                .checked_add(128)
                .ok_or("guider owned test byte count overflowed")?
        );
        let profile = DiscreteSamplingProfile::sd15()?;
        let plan = SamplingPlan::new(
            "euler",
            "normal",
            profile.identity().clone(),
            7,
            1,
            2.0,
            1.0,
        )?;
        let mut denoiser = ConstantDenoiser { backend: &backend };
        let guided = NativeGuiderPayload::execute_strategy(
            &strategy,
            &backend,
            &latent,
            2.0,
            &profile,
            &plan,
            GuidanceOptions::default(),
            &mut denoiser,
            &mut [],
            &context,
        )?;
        assert_eq!(
            &*tensor_to_f32(&backend, guided.guided(), &context)?,
            &[5.0, 5.0]
        );
        assert!(positive_resident_bytes > 0);
        Ok(())
    }

    #[test]
    fn fixed_noise_and_euler_are_executable_and_fail_closed() -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let latent = tensor_from_f32(&backend, &[1, 1, 2], &[0.0, 0.0], &context)?;
        let fixed_tensor = tensor_from_f32(&backend, &[1, 1, 2], &[0.5, -0.5], &context)?;
        let fixed = NativeNoisePayload::fixed_latent(0, fixed_tensor)?;
        let request = NoiseRequest::native_diffusion("prompt", "fixed")?;
        let generated = fixed.generate(&backend, &latent, None, &request, &context)?;
        assert_eq!(
            &*tensor_to_f32(&backend, &generated.noise, &context)?,
            &[0.5, -0.5]
        );

        let sampler = NativeSamplerPayload::euler()?;
        let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;
        let cancelled_initial = initial.clone();
        let mut denoiser = |input: &Tensor, _sigma: f32, _step: usize| Ok(input.clone());
        let profile = DiscreteSamplingProfile::sd15()?;
        let trace = sampler.execute_checked(
            &backend,
            sampler_plan(NativeSamplerAlgorithm::Euler, 2, 0)?,
            &profile,
            initial,
            &[2.0, 1.0, 0.0],
            compatibility_noise("euler"),
            &context,
            &mut denoiser,
            &mut |_, _, _| Ok(()),
        )?;
        let NativeSamplerExecution::Fixed(trace) = trace else {
            return Err("Euler produced an adaptive trace".into());
        };
        assert_eq!(trace.sigmas, vec![2.0, 1.0, 0.0]);
        assert!(trace.latents.len() >= 2);
        assert!(sampler.resident_bytes() > 0);

        cancellation.cancel();
        assert!(matches!(
            sampler.execute_checked(
                &backend,
                sampler_plan(NativeSamplerAlgorithm::Euler, 1, 0)?,
                &profile,
                cancelled_initial,
                &[1.0, 0.0],
                compatibility_noise("cancelled-euler"),
                &context,
                &mut denoiser,
                &mut |_, _, _| Ok(()),
            ),
            Err(NativeSamplerPayloadError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }

    fn compatibility_noise(node: &str) -> CompatibilityNoiseRequest {
        CompatibilityNoiseRequest::new(
            "task406-workflow",
            "task406-attempt",
            node,
            0,
            0,
            0,
            0,
            RetryRngPolicy::Replay,
        )
    }

    fn sampler_plan(
        algorithm: NativeSamplerAlgorithm,
        steps: u32,
        seed: u64,
    ) -> Result<SamplingPlan, crate::SamplingError> {
        SamplingPlan::new(
            algorithm.identity(),
            crate::NORMAL_SCHEDULER_ID,
            crate::SamplingProfileIdentity::sd15(),
            seed,
            steps,
            1.0,
            1.0,
        )
    }

    #[test]
    fn sampler_payload_identity_is_sealed_exhaustive_and_parameterized()
    -> Result<(), Box<dyn Error>> {
        let registry = SamplerRegistry::foundational()?;
        assert_eq!(NativeSamplerAlgorithm::all().len(), 46);
        assert_eq!(registry.definitions().len(), 46);
        let mut digests = BTreeSet::new();
        for algorithm in NativeSamplerAlgorithm::all().iter().copied() {
            let payload = NativeSamplerPayload::selected(algorithm.identity())?;
            assert_eq!(payload.algorithm(), algorithm);
            assert_eq!(payload.identity().as_str(), algorithm.identity());
            assert!(digests.insert(payload.semantic_digest_sha256().to_owned()));
            payload.validate()?;
        }
        let lms_one = NativeSamplerPayload::lms(1)?;
        let lms_hundred = NativeSamplerPayload::lms(100)?;
        assert_ne!(
            lms_one.semantic_digest_sha256(),
            lms_hundred.semantic_digest_sha256()
        );
        assert!(matches!(
            NativeSamplerPayload::lms(0),
            Err(NativeSamplerPayloadError::ParameterMismatch { .. })
        ));
        assert!(matches!(
            NativeSamplerPayload::checked(
                NativeSamplerAlgorithm::Euler,
                NativeSamplerParameters::Lms { order: 4 },
            ),
            Err(NativeSamplerPayloadError::ParameterMismatch { .. })
        ));
        let mut adaptive =
            crate::generated_dpm_adaptive_sim_model_0164::DpmAdaptiveOptions::default();
        adaptive.order = 1;
        assert!(matches!(
            NativeSamplerPayload::checked(
                NativeSamplerAlgorithm::DpmAdaptive,
                NativeSamplerParameters::DpmAdaptive(adaptive),
            ),
            Err(NativeSamplerPayloadError::InvalidParameters(_))
        ));
        let mut seeds = crate::generated_seeds_2_sim_model_0199::Seeds2Options::default();
        seeds.intermediate_step_ratio = 0.0;
        assert!(matches!(
            NativeSamplerPayload::checked(
                NativeSamplerAlgorithm::Seeds2,
                NativeSamplerParameters::Seeds2(seeds),
            ),
            Err(NativeSamplerPayloadError::InvalidParameters(_))
        ));
        NativeSamplerPayload::checked(
            NativeSamplerAlgorithm::SaSolver,
            NativeSamplerParameters::SaSolver(NativeSaSolverParameters {
                eta: 1.0,
                sde_start_percent: 0.2,
                sde_end_percent: 0.8,
                noise_scale: 1.0,
                predictor_order: 3,
                corrector_order: 0,
                use_pece: false,
                simple_order_2: false,
            }),
        )?;
        assert!(matches!(
            NativeSamplerPayload::selected("not_a_sampler"),
            Err(NativeSamplerPayloadError::UnknownAlgorithm(_))
        ));
        Ok(())
    }

    #[test]
    fn void_ddim_uses_alpha_space_without_standard_noise_scaling() -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let profile = DiscreteSamplingProfile::sd15()?;
        let initial = tensor_from_f32(&backend, &[1], &[2.0], &context)?;
        let denoised = tensor_from_f32(&backend, &[1], &[0.5], &context)?;
        let mut denoiser = move |_input: &Tensor, _sigma: f32, _step: usize| Ok(denoised.clone());
        let mut callbacks = Vec::new();
        let execution = NativeSamplerPayload::void_ddim()?.execute_checked(
            &backend,
            sampler_plan(NativeSamplerAlgorithm::VoidDdim, 1, 7)?,
            &profile,
            initial,
            &[2.0, 1.0],
            compatibility_noise("void"),
            &context,
            &mut denoiser,
            &mut |progress, latent, output| {
                let latent_value = tensor_to_f32(&backend, latent, &context)
                    .map_err(|error| error.to_string())?
                    .first()
                    .copied()
                    .ok_or_else(|| "VOID callback latent is empty".to_owned())?;
                let output_value = tensor_to_f32(&backend, output, &context)
                    .map_err(|error| error.to_string())?
                    .first()
                    .copied()
                    .ok_or_else(|| "VOID callback output is empty".to_owned())?;
                callbacks.push((progress.step(), latent_value, output_value));
                Ok(())
            },
        )?;
        let actual_values = tensor_to_f32(
            &backend,
            execution.final_tensor().ok_or("VOID produced no tensor")?,
            &context,
        )?;
        let actual = actual_values
            .first()
            .copied()
            .ok_or("VOID produced an empty tensor")?;
        let alpha = 1.0_f32 / 5.0;
        let next_alpha = 0.5_f32;
        let epsilon = (2.0 - alpha.sqrt() * 0.5) / (1.0 - alpha).sqrt();
        let expected = next_alpha.sqrt() * 0.5 + (1.0 - next_alpha).sqrt() * epsilon;
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        assert!((actual - 1.25).abs() > 0.1, "VOID aliased Euler");
        assert_eq!(callbacks, vec![(0, 2.0, 0.5)]);

        let half_initial = backend
            .upload_f32_payload(&[1], &[2.0], DType::F16, &context)?
            .0;
        let half_denoised = tensor_from_f32(&backend, &[1], &[0.5], &context)?;
        let half_execution = NativeSamplerPayload::void_ddim()?.execute_checked(
            &backend,
            sampler_plan(NativeSamplerAlgorithm::VoidDdim, 1, 7)?,
            &profile,
            half_initial,
            &[2.0, 1.0],
            compatibility_noise("void-half"),
            &context,
            &mut move |_input: &Tensor, _sigma: f32, _step: usize| Ok(half_denoised.clone()),
            &mut |_, _, _| Ok(()),
        )?;
        assert_eq!(
            half_execution
                .final_tensor()
                .ok_or("VOID half input produced no tensor")?
                .descriptor()
                .dtype(),
            DType::F32
        );
        Ok(())
    }

    #[test]
    fn euler_cfg_pp_alternative_is_a_distinct_source_option() -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let profile = DiscreteSamplingProfile::sd15()?;
        let alternative = NativeSamplerPayload::checked(
            NativeSamplerAlgorithm::EulerCfgPp,
            NativeSamplerParameters::EulerCfgPp {
                version: NativeEulerCfgPpVersion::Alternative,
            },
        )?;
        let regular = NativeSamplerPayload::selected("euler_cfg_pp")?;
        assert_ne!(
            alternative.semantic_digest_sha256(),
            regular.semantic_digest_sha256()
        );
        let mut denoiser = SamplerTestDenoiser {
            backend: &backend,
            context: &context,
            guided: 0.5,
            unconditional: 0.25,
            calls: 0,
        };
        let mut callbacks = Vec::new();
        let execution = alternative.execute_checked(
            &backend,
            sampler_plan(NativeSamplerAlgorithm::EulerCfgPp, 1, 17)?,
            &profile,
            tensor_from_f32(&backend, &[1], &[2.0], &context)?,
            &[2.0, 1.0],
            compatibility_noise("euler-cfg-pp-alternative"),
            &context,
            &mut denoiser,
            &mut |progress, latent, output| {
                callbacks.push((
                    progress.step(),
                    tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?
                        [0],
                    tensor_to_f32(&backend, output, &context).map_err(|error| error.to_string())?
                        [0],
                ));
                Ok(())
            },
        )?;
        let actual = tensor_to_f32(
            &backend,
            execution
                .final_tensor()
                .ok_or("missing alternative output")?,
            &context,
        )?[0];
        assert!((actual - 1.375).abs() < 1.0e-6, "{actual}");
        assert_eq!(callbacks, vec![(0, 2.0, 0.5)]);
        assert_eq!(denoiser.calls, 1);
        Ok(())
    }

    #[test]
    fn checked_sampler_invocation_rejects_before_publication_and_replays_rng()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let profile = DiscreteSamplingProfile::sd15()?;
        let sampler = NativeSamplerPayload::selected("ddpm")?;
        let initial = tensor_from_f32(&backend, &[1], &[1.0], &context)?;
        let mut denoiser = SamplerTestDenoiser {
            backend: &backend,
            context: &context,
            guided: 0.25,
            unconditional: 0.0,
            calls: 0,
        };
        let mut callbacks = 0_usize;
        assert!(
            sampler
                .execute_checked(
                    &backend,
                    sampler_plan(NativeSamplerAlgorithm::Ddpm, 2, 23)?,
                    &profile,
                    initial.clone(),
                    &[2.0, 2.0, 0.0],
                    compatibility_noise("invalid-schedule"),
                    &context,
                    &mut denoiser,
                    &mut |_, _, _| {
                        callbacks += 1;
                        Ok(())
                    },
                )
                .is_err()
        );
        assert_eq!(denoiser.calls, 0);
        assert_eq!(callbacks, 0);

        let incompatible_profile = DiscreteSamplingProfile::lotus_sdpose()?;
        assert!(matches!(
            sampler.execute_checked(
                &backend,
                sampler_plan(NativeSamplerAlgorithm::Ddpm, 1, 23)?,
                &incompatible_profile,
                initial.clone(),
                &[1.0, 0.0],
                compatibility_noise("profile-mismatch"),
                &context,
                &mut denoiser,
                &mut |_, _, _| Ok(()),
            ),
            Err(NativeSamplerPayloadError::Sampling(
                crate::SamplingError::ProfileMismatch { .. }
            ))
        ));
        assert_eq!(denoiser.calls, 0);

        let mut failed_callback = true;
        let failed = sampler.execute_checked(
            &backend,
            sampler_plan(NativeSamplerAlgorithm::Ddpm, 1, 23)?,
            &profile,
            initial.clone(),
            &[1.0, 0.0],
            compatibility_noise("rng-replay"),
            &context,
            &mut denoiser,
            &mut |_, _, _| {
                if failed_callback {
                    failed_callback = false;
                    Err("injected callback failure".to_owned())
                } else {
                    Ok(())
                }
            },
        );
        assert!(failed.is_err());
        let replay = sampler.execute_checked(
            &backend,
            sampler_plan(NativeSamplerAlgorithm::Ddpm, 1, 23)?,
            &profile,
            initial.clone(),
            &[1.0, 0.0],
            compatibility_noise("rng-replay"),
            &context,
            &mut denoiser,
            &mut |_, _, _| Ok(()),
        )?;
        let cancelled_initial = initial.clone();
        let baseline = sampler.execute_checked(
            &backend,
            sampler_plan(NativeSamplerAlgorithm::Ddpm, 1, 23)?,
            &profile,
            initial,
            &[1.0, 0.0],
            compatibility_noise("rng-replay"),
            &context,
            &mut denoiser,
            &mut |_, _, _| Ok(()),
        )?;
        assert_eq!(
            replay
                .final_tensor()
                .ok_or("missing replay output")?
                .contiguous_bytes()?,
            baseline
                .final_tensor()
                .ok_or("missing baseline output")?
                .contiguous_bytes()?
        );

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024)?,
            &cancelled,
        );
        let calls_before = denoiser.calls;
        assert!(
            sampler
                .execute_checked(
                    &backend,
                    sampler_plan(NativeSamplerAlgorithm::Ddpm, 1, 23)?,
                    &profile,
                    cancelled_initial,
                    &[1.0, 0.0],
                    compatibility_noise("cancelled"),
                    &cancelled_context,
                    &mut denoiser,
                    &mut |_, _, _| Ok(()),
                )
                .is_err()
        );
        assert_eq!(denoiser.calls, calls_before);

        let oom_cancellation = CancellationToken::default();
        let tiny_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &oom_cancellation,
        );
        let out_of_memory = NativeSamplerPayload::void_ddim()?.execute_checked(
            &backend,
            sampler_plan(NativeSamplerAlgorithm::VoidDdim, 2, 29)?,
            &profile,
            tensor_from_f32(&backend, &[1], &[1.0], &context)?,
            &[2.0, 1.0, 0.0],
            compatibility_noise("oom"),
            &tiny_context,
            &mut denoiser,
            &mut |_, _, _| Ok(()),
        );
        assert!(matches!(
            out_of_memory,
            Err(NativeSamplerPayloadError::Algorithm {
                algorithm: "void_ddim",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn every_sealed_sampler_identity_reaches_its_canonical_implementation()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let profile = DiscreteSamplingProfile::sd15()?;
        for algorithm in NativeSamplerAlgorithm::all().iter().copied() {
            let payload = NativeSamplerPayload::selected(algorithm.identity())?;
            let mut denoiser = SamplerTestDenoiser {
                backend: &backend,
                context: &context,
                guided: 0.25,
                unconditional: 0.125,
                calls: 0,
            };
            let mut callbacks = 0_usize;
            let mut callback_shape_matches = true;
            let initial_shape = if algorithm == NativeSamplerAlgorithm::LcmUpscale {
                &[1, 1, 1, 1][..]
            } else {
                &[1][..]
            };
            let execution = payload
                .execute_checked(
                    &backend,
                    sampler_plan(algorithm, 3, 31)?,
                    &profile,
                    tensor_from_f32(&backend, initial_shape, &[1.0], &context)?,
                    &[3.0, 2.0, 1.0, 0.0],
                    compatibility_noise(algorithm.identity()),
                    &context,
                    &mut denoiser,
                    &mut |progress, _, _| {
                        callbacks += 1;
                        callback_shape_matches &= match (algorithm, progress) {
                            (
                                NativeSamplerAlgorithm::DpmAdaptive,
                                NativeSamplerProgress::Adaptive(_),
                            ) => true,
                            (NativeSamplerAlgorithm::DpmAdaptive, _) => false,
                            (_, NativeSamplerProgress::Fixed(_)) => true,
                            (_, NativeSamplerProgress::Adaptive(_)) => false,
                        };
                        Ok(())
                    },
                )
                .map_err(|error| format!("{}: {error}", algorithm.identity()))?;
            let values = tensor_to_f32(
                &backend,
                execution
                    .final_tensor()
                    .ok_or_else(|| format!("{} produced no output", algorithm.identity()))?,
                &context,
            )?;
            assert!(
                !values.is_empty() && values.iter().all(|value| value.is_finite()),
                "{} produced invalid output {values:?}",
                algorithm.identity()
            );
            assert!(
                denoiser.calls > 0,
                "{} was not invoked",
                algorithm.identity()
            );
            assert!(
                callbacks > 0,
                "{} callback was not invoked",
                algorithm.identity()
            );
            assert!(
                callback_shape_matches,
                "{} used the wrong callback shape",
                algorithm.identity()
            );
        }
        Ok(())
    }

    #[test]
    fn lms_orders_and_lcm_upscale_methods_execute_their_checked_contracts()
    -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = context(&backend, &authority, &cancellation)?;
        let profile = DiscreteSamplingProfile::sd15()?;

        for (order, expected) in [
            (1_u8, 1.6833624_f32),
            (4, 1.6030869),
            (5, 1.5990347),
            (100, 1.5952116),
        ] {
            let step_count = 8;
            let sigmas = (0..=step_count)
                .map(|index| (step_count - index) as f32)
                .collect::<Vec<_>>();
            let mut sigmas = sigmas;
            if let Some(last) = sigmas.last_mut() {
                *last = 0.0;
            }
            let initial = tensor_from_f32(&backend, &[1], &[2.0], &context)?;
            let mut denoiser = |input: &Tensor, sigma: f32, _step: usize| {
                let values =
                    tensor_to_f32(&backend, input, &context).map_err(|error| error.to_string())?;
                let current = values
                    .first()
                    .copied()
                    .ok_or_else(|| "LMS test tensor is empty".to_owned())?;
                let value = 0.8 * current + 0.1 * sigma;
                tensor_from_f32(&backend, &[1], &[value], &context)
                    .map_err(|error| error.to_string())
            };
            let execution = NativeSamplerPayload::lms(order)?.execute_checked(
                &backend,
                sampler_plan(NativeSamplerAlgorithm::Lms, u32::try_from(step_count)?, 11)?,
                &profile,
                initial,
                &sigmas,
                compatibility_noise("lms"),
                &context,
                &mut denoiser,
                &mut |_, _, _| Ok(()),
            )?;
            let actual = tensor_to_f32(
                &backend,
                execution.final_tensor().ok_or("LMS produced no output")?,
                &context,
            )?[0];
            assert!(
                (actual - expected).abs() < 2.0e-5,
                "LMS order {order}: {actual} != {expected}"
            );
        }

        let input_values = [1.0, 0.0, 0.0, 1.0, 0.5, 0.25, 0.25, 0.5];
        for method in [
            crate::generated_lcm_upscale::LcmUpscaleMethod::Bislerp,
            crate::generated_lcm_upscale::LcmUpscaleMethod::NearestExact,
            crate::generated_lcm_upscale::LcmUpscaleMethod::Bilinear,
            crate::generated_lcm_upscale::LcmUpscaleMethod::Area,
            crate::generated_lcm_upscale::LcmUpscaleMethod::Bicubic,
        ] {
            let options =
                crate::generated_lcm_upscale::LcmUpscaleOptions::new(2.0, Some(1), method)?;
            let payload = NativeSamplerPayload::lcm_upscale(options)?;
            let initial = tensor_from_f32(&backend, &[1, 2, 2, 2], &input_values, &context)?;
            let mut denoiser = |input: &Tensor, _sigma: f32, _step: usize| Ok(input.clone());
            let output = payload.execute_checked(
                &backend,
                sampler_plan(NativeSamplerAlgorithm::LcmUpscale, 1, 13)?,
                &profile,
                initial,
                &[1.0, 0.0],
                compatibility_noise("lcm-upscale"),
                &context,
                &mut denoiser,
                &mut |_, _, _| Ok(()),
            )?;
            let output = output
                .final_tensor()
                .ok_or("LCM-upscale produced no tensor")?;
            assert_eq!(output.descriptor().shape(), &[1, 2, 4, 4]);
            if method == crate::generated_lcm_upscale::LcmUpscaleMethod::Bislerp {
                let actual = tensor_to_f32(&backend, output, &context)?;
                let expected = [
                    1.0, 0.66511476, 0.12761779, 0.0, 0.66511476, 0.50567114, 0.23213615,
                    0.12761779, 0.12761779, 0.23213615, 0.50567114, 0.66511476, 0.0, 0.12761779,
                    0.66511476, 1.0, 0.5, 0.6078398, 0.44923338, 0.25, 0.6078398, 0.6102349,
                    0.52661914, 0.44923338, 0.44923338, 0.52661914, 0.6102349, 0.6078398, 0.25,
                    0.44923338, 0.6078398, 0.5,
                ];
                for (element, (actual, expected)) in
                    actual.iter().copied().zip(expected).enumerate()
                {
                    assert!(
                        (actual - expected).abs() < 2.0e-6,
                        "bislerp element {element}: {actual} != {expected}"
                    );
                }
            }
        }
        Ok(())
    }
}
