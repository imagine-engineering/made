use crate::{
    NativeCompiledModelInvocation, NativeHandleKind, NativeHandleType, NativeNodeContractError,
    native_source_type_projection,
};
use sha2::{Digest, Sha256};
use sim_media::{
    NativeArtifactPayload, NativeAudioPayload, NativeBoundingBoxPayload, NativeCameraPayload,
    NativeFaceLandmarksPayload, NativeFile3DPayload, NativeMediaResidentParts, NativeMeshPayload,
    NativePoseKeypointPayload, NativeSam3TrackDataPayload, NativeSplatPayload, NativeTracksPayload,
    NativeVideoPayload, NativeVoxelPayload,
};
use sim_model::{
    AudioEncoderOutput, IcLoraParameters, LossMap, NativeModelBackingKind, NativeModelPayload,
    NativeModelResourceRole, NativeStructuredResidentParts,
    clip_vision::ClipVisionOutput,
    conditioning::{ConditioningError, ConditioningSet},
};
use sim_sampler::{
    NativeControlPayload, NativeDiffusionPayload, NativeDiffusionResidentAllocation,
    NativeDiffusionResidentAllocationId, NativeGuiderPayload, NativeNoisePayload,
    NativeSamplerPayload,
};
use sim_tensor::{
    NativeLatentBundle, NativeLatentBundleError, NativeTensorPayload, NativeTensorRole,
};
use std::{collections::BTreeMap, mem, sync::Arc};
use thiserror::Error;

const MAX_PROVIDER_NAMESPACE_BYTES: usize = 4_096;
const MAX_PROVIDER_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub struct NativeProviderPayload {
    handle_type: NativeHandleType,
    signed_namespace: String,
    semantic_digest_sha256: String,
    abi_bytes: Vec<u8>,
    abi_digest_sha256: String,
}

impl NativeProviderPayload {
    pub fn from_abi(
        handle_type: NativeHandleType,
        signed_namespace: impl Into<String>,
        abi_bytes: Vec<u8>,
    ) -> Result<Self, NativeStoredPayloadError> {
        let signed_namespace = signed_namespace.into();
        let semantic_digest_sha256 =
            provider_semantic_digest_sha256(&handle_type, &signed_namespace, &abi_bytes);
        Self::checked(
            handle_type,
            signed_namespace,
            semantic_digest_sha256,
            abi_bytes,
        )
    }

    pub fn checked(
        handle_type: NativeHandleType,
        signed_namespace: impl Into<String>,
        semantic_digest_sha256: impl Into<String>,
        abi_bytes: Vec<u8>,
    ) -> Result<Self, NativeStoredPayloadError> {
        let payload = Self {
            handle_type,
            signed_namespace: signed_namespace.into(),
            semantic_digest_sha256: semantic_digest_sha256.into(),
            abi_digest_sha256: format!("{:x}", Sha256::digest(&abi_bytes)),
            abi_bytes,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn handle_type(&self) -> &NativeHandleType {
        &self.handle_type
    }

    pub fn signed_namespace(&self) -> &str {
        &self.signed_namespace
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub fn identity_digest_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.provider-payload-identity.v1");
        hasher.update([0]);
        hasher.update([native_handle_kind_tag(self.handle_type.kind)]);
        hasher.update([0]);
        hasher.update(self.handle_type.type_id.as_bytes());
        hasher.update([0]);
        hasher.update(self.signed_namespace.as_bytes());
        hasher.update([0]);
        hasher.update(self.semantic_digest_sha256.as_bytes());
        hasher.update([0]);
        hasher.update(self.abi_digest_sha256.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn abi_bytes(&self) -> &[u8] {
        &self.abi_bytes
    }

    pub fn resident_bytes(&self) -> Result<usize, NativeStoredPayloadError> {
        mem::size_of::<Self>()
            .checked_add(self.handle_type.type_id.capacity())
            .and_then(|bytes| bytes.checked_add(self.signed_namespace.capacity()))
            .and_then(|bytes| bytes.checked_add(self.semantic_digest_sha256.capacity()))
            .and_then(|bytes| bytes.checked_add(self.abi_bytes.capacity()))
            .and_then(|bytes| bytes.checked_add(self.abi_digest_sha256.capacity()))
            .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)
    }

    pub fn validate(&self) -> Result<(), NativeStoredPayloadError> {
        self.handle_type.validate()?;
        if self.handle_type.kind != NativeHandleKind::ProviderTask
            || self.signed_namespace.is_empty()
            || self.signed_namespace.len() > MAX_PROVIDER_NAMESPACE_BYTES
            || self.signed_namespace.chars().any(char::is_control)
            || self.abi_bytes.len() > MAX_PROVIDER_PAYLOAD_BYTES
            || !valid_sha256(&self.semantic_digest_sha256)
            || !valid_sha256(&self.abi_digest_sha256)
            || self.abi_digest_sha256 != format!("{:x}", Sha256::digest(&self.abi_bytes))
        {
            return Err(NativeStoredPayloadError::InvalidProviderPayload);
        }
        self.resident_bytes()?;
        Ok(())
    }
}

fn provider_semantic_digest_sha256(
    handle_type: &NativeHandleType,
    signed_namespace: &str,
    abi_bytes: &[u8],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.provider-payload-semantic.v1");
    hasher.update([0]);
    hasher.update([native_handle_kind_tag(handle_type.kind)]);
    hasher.update([0]);
    hasher.update(handle_type.type_id.as_bytes());
    hasher.update([0]);
    hasher.update(signed_namespace.as_bytes());
    hasher.update([0]);
    hasher.update(Sha256::digest(abi_bytes));
    format!("{:x}", hasher.finalize())
}

#[derive(Clone)]
pub struct NativeStoredModelPayload {
    resource: NativeStoredModelResource,
    compiled_model: Option<Arc<dyn NativeCompiledModelInvocation>>,
    digest_sha256: String,
}

#[derive(Clone)]
enum NativeStoredModelResource {
    Diffusion(Arc<NativeDiffusionPayload>),
    ModelResource(Arc<NativeModelPayload>),
}

impl NativeStoredModelPayload {
    pub fn native_diffusion(
        diffusion: Arc<NativeDiffusionPayload>,
    ) -> Result<Self, NativeStoredPayloadError> {
        validate_native_diffusion_backing(&diffusion)?;
        let digest_sha256 = diffusion.digest_sha256().to_owned();
        let payload = Self {
            resource: NativeStoredModelResource::Diffusion(diffusion),
            compiled_model: None,
            digest_sha256,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn model_resource(
        resource: Arc<NativeModelPayload>,
    ) -> Result<Self, NativeStoredPayloadError> {
        require_model_resource_role(resource.identity().role())?;
        if !model_resource_is_concrete(&resource) {
            return Err(NativeStoredPayloadError::NonCanonicalModelResourceRole {
                role: resource.identity().role(),
            });
        }
        let payload = Self {
            digest_sha256: resource.identity().digest_sha256().to_owned(),
            resource: NativeStoredModelResource::ModelResource(resource),
            compiled_model: None,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn native_compiled_diffusion(
        diffusion: Arc<NativeDiffusionPayload>,
        compiled_model: Arc<dyn NativeCompiledModelInvocation>,
    ) -> Result<Self, NativeStoredPayloadError> {
        validate_native_diffusion_backing(&diffusion)?;
        compiled_model
            .validate()
            .map_err(|_| NativeStoredPayloadError::InvalidCompiledModel)?;
        let model_execution_sha256 = diffusion
            .model_resources()
            .map(|(model, _)| model.identity().execution_sha256())
            .ok_or(NativeStoredPayloadError::InvalidCompiledModel)?;
        if diffusion.role() != NativeModelResourceRole::Model
            || compiled_model.source_model_sha256() != model_execution_sha256
        {
            return Err(NativeStoredPayloadError::InvalidCompiledModel);
        }
        let digest_sha256 = compiled_model_digest_sha256(
            diffusion.digest_sha256(),
            compiled_model.policy_sha256(),
        )?;
        let payload = Self {
            resource: NativeStoredModelResource::Diffusion(diffusion),
            compiled_model: Some(compiled_model),
            digest_sha256,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn diffusion(&self) -> Option<&Arc<NativeDiffusionPayload>> {
        match &self.resource {
            NativeStoredModelResource::Diffusion(diffusion) => Some(diffusion),
            NativeStoredModelResource::ModelResource(_) => None,
        }
    }

    pub fn model_payload(&self) -> &Arc<NativeModelPayload> {
        match &self.resource {
            NativeStoredModelResource::Diffusion(diffusion) => diffusion.model_payload(),
            NativeStoredModelResource::ModelResource(resource) => resource,
        }
    }

    pub fn compiled_model(&self) -> Option<&Arc<dyn NativeCompiledModelInvocation>> {
        self.compiled_model.as_ref()
    }

    pub fn validate(&self) -> Result<(), NativeStoredPayloadError> {
        match &self.resource {
            NativeStoredModelResource::Diffusion(diffusion) => {
                diffusion.validate()?;
                validate_native_diffusion_backing(diffusion)?;
                let expected_digest = if let Some(compiled_model) = &self.compiled_model {
                    compiled_model
                        .validate()
                        .map_err(|_| NativeStoredPayloadError::InvalidCompiledModel)?;
                    let model_execution_sha256 = diffusion
                        .model_resources()
                        .map(|(model, _)| model.identity().execution_sha256())
                        .ok_or(NativeStoredPayloadError::InvalidCompiledModel)?;
                    if diffusion.role() != NativeModelResourceRole::Model
                        || compiled_model.source_model_sha256() != model_execution_sha256
                    {
                        return Err(NativeStoredPayloadError::InvalidCompiledModel);
                    }
                    compiled_model_digest_sha256(
                        diffusion.digest_sha256(),
                        compiled_model.policy_sha256(),
                    )?
                } else {
                    diffusion.digest_sha256().to_owned()
                };
                if self.digest_sha256 != expected_digest {
                    return Err(NativeStoredPayloadError::InvalidCompiledModel);
                }
            }
            NativeStoredModelResource::ModelResource(resource) => {
                if self.compiled_model.is_some()
                    || self.digest_sha256 != resource.identity().digest_sha256()
                {
                    return Err(NativeStoredPayloadError::InvalidCompiledModel);
                }
                resource.validate()?;
                require_model_resource_role(resource.identity().role())?;
                if !model_resource_is_concrete(resource) {
                    return Err(NativeStoredPayloadError::NonCanonicalModelResourceRole {
                        role: resource.identity().role(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn handle_type(&self) -> Result<NativeHandleType, NativeStoredPayloadError> {
        let source_type = match &self.resource {
            NativeStoredModelResource::Diffusion(diffusion) => diffusion.role().source_type_id(),
            NativeStoredModelResource::ModelResource(resource) => {
                resource.identity().role().source_type_id()
            }
        };
        native_source_type_projection(source_type)?
            .handle_type()?
            .ok_or_else(|| NativeStoredPayloadError::MissingHandleType(source_type.to_owned()))
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn resident_bytes(&self) -> Result<usize, NativeStoredPayloadError> {
        let resource_bytes = match &self.resource {
            NativeStoredModelResource::Diffusion(diffusion) => diffusion.resident_bytes()?,
            NativeStoredModelResource::ModelResource(resource) => {
                usize::try_from(resource.resident_bytes())
                    .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?
            }
        };
        let compiled_bytes = self
            .compiled_model
            .as_ref()
            .map_or(0, |compiled_model| compiled_model.resident_bytes());
        mem::size_of::<Self>()
            .checked_add(resource_bytes)
            .and_then(|bytes| bytes.checked_add(self.digest_sha256.capacity()))
            .and_then(|bytes| bytes.checked_add(compiled_bytes))
            .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)
    }
}

fn compiled_model_digest_sha256(
    source_model_sha256: &str,
    policy_sha256: &str,
) -> Result<String, NativeStoredPayloadError> {
    if !valid_sha256(source_model_sha256) || !valid_sha256(policy_sha256) {
        return Err(NativeStoredPayloadError::InvalidCompiledModel);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.native-compiled-model.v1");
    hasher.update(source_model_sha256.as_bytes());
    hasher.update(policy_sha256.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_native_diffusion_backing(
    diffusion: &NativeDiffusionPayload,
) -> Result<(), NativeStoredPayloadError> {
    let model = diffusion.model_payload();
    let is_canonical = match diffusion.role() {
        NativeModelResourceRole::Model => model.model().is_some(),
        NativeModelResourceRole::Clip => model.clip().is_some(),
        NativeModelResourceRole::Vae => model.vae().is_some() || model.structured_vae().is_some(),
        _ => false,
    };
    if is_canonical {
        Ok(())
    } else {
        Err(NativeStoredPayloadError::NonCanonicalDiffusionResource)
    }
}

fn require_model_resource_role(
    role: NativeModelResourceRole,
) -> Result<(), NativeStoredPayloadError> {
    if matches!(
        role,
        NativeModelResourceRole::AudioEncoder
            | NativeModelResourceRole::BackgroundRemoval
            | NativeModelResourceRole::Model
            | NativeModelResourceRole::OpticalFlow
            | NativeModelResourceRole::ClipVision
            | NativeModelResourceRole::Clip
            | NativeModelResourceRole::FrameInterpolation
            | NativeModelResourceRole::Gligen
            | NativeModelResourceRole::LatentUpscaleModel
            | NativeModelResourceRole::LatentOperation
            | NativeModelResourceRole::LoraModel
            | NativeModelResourceRole::Da3Model
            | NativeModelResourceRole::MogeModel
            | NativeModelResourceRole::Photomaker
            | NativeModelResourceRole::StyleModel
    ) {
        Ok(())
    } else {
        Err(NativeStoredPayloadError::NonCanonicalModelResourceRole { role })
    }
}

fn model_resource_is_concrete(resource: &NativeModelPayload) -> bool {
    match resource.identity().role() {
        NativeModelResourceRole::AudioEncoder => resource.audio_encoder_resource().is_some(),
        NativeModelResourceRole::BackgroundRemoval => {
            resource.background_removal_resource().is_some()
        }
        NativeModelResourceRole::Model => resource.sdpose_model_resource().is_some(),
        NativeModelResourceRole::OpticalFlow => resource.optical_flow_resource().is_some(),
        NativeModelResourceRole::ClipVision => resource.clip_vision_resource().is_some(),
        NativeModelResourceRole::Clip => {
            resource.decoder_clip_resource().is_some()
                || resource.qwen_multimodal_resource().is_some()
                || resource.gemma_multimodal_resource().is_some()
                || resource.native_clip_resource().is_some()
        }
        NativeModelResourceRole::FrameInterpolation => {
            resource.frame_interpolation_resource().is_some()
        }
        NativeModelResourceRole::Gligen => resource.gligen_resource().is_some(),
        NativeModelResourceRole::LatentUpscaleModel => {
            resource.latent_upscale_model_resource().is_some()
        }
        NativeModelResourceRole::LatentOperation => resource.latent_operation_resource().is_some(),
        NativeModelResourceRole::LoraModel => resource.lora_model_resource().is_some(),
        NativeModelResourceRole::Da3Model => resource.depth_anything_3_resource().is_some(),
        NativeModelResourceRole::MogeModel => resource.moge_resource().is_some(),
        NativeModelResourceRole::Photomaker => resource.photomaker_resource().is_some(),
        NativeModelResourceRole::StyleModel => resource.style_model_resource().is_some(),
        _ => false,
    }
}

#[derive(Clone)]
pub struct NativeStoredGuiderPayload {
    guider: Arc<NativeGuiderPayload>,
    compiled_model: Option<Arc<dyn NativeCompiledModelInvocation>>,
    digest_sha256: String,
}

impl NativeStoredGuiderPayload {
    pub fn checked(
        guider: Arc<NativeGuiderPayload>,
        compiled_model: Option<Arc<dyn NativeCompiledModelInvocation>>,
    ) -> Result<Self, NativeStoredPayloadError> {
        guider.validate()?;
        let digest_sha256 = match &compiled_model {
            Some(compiled_model) => {
                compiled_model
                    .validate()
                    .map_err(|_| NativeStoredPayloadError::InvalidCompiledModel)?;
                if compiled_model.source_model_sha256()
                    != guider.model().identity().execution_sha256()
                {
                    return Err(NativeStoredPayloadError::InvalidCompiledModel);
                }
                compiled_model_digest_sha256(
                    guider.semantic_digest_sha256(),
                    compiled_model.policy_sha256(),
                )?
            }
            None => guider.semantic_digest_sha256().to_owned(),
        };
        let payload = Self {
            guider,
            compiled_model,
            digest_sha256,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn guider(&self) -> &Arc<NativeGuiderPayload> {
        &self.guider
    }

    pub fn compiled_model(&self) -> Option<&Arc<dyn NativeCompiledModelInvocation>> {
        self.compiled_model.as_ref()
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn validate(&self) -> Result<(), NativeStoredPayloadError> {
        let expected = Self::checked_digest(&self.guider, self.compiled_model.as_ref())?;
        if expected != self.digest_sha256 {
            return Err(NativeStoredPayloadError::InvalidCompiledModel);
        }
        Ok(())
    }

    fn checked_digest(
        guider: &NativeGuiderPayload,
        compiled_model: Option<&Arc<dyn NativeCompiledModelInvocation>>,
    ) -> Result<String, NativeStoredPayloadError> {
        guider.validate()?;
        match compiled_model {
            Some(compiled_model) => {
                compiled_model
                    .validate()
                    .map_err(|_| NativeStoredPayloadError::InvalidCompiledModel)?;
                if compiled_model.source_model_sha256()
                    != guider.model().identity().execution_sha256()
                {
                    return Err(NativeStoredPayloadError::InvalidCompiledModel);
                }
                compiled_model_digest_sha256(
                    guider.semantic_digest_sha256(),
                    compiled_model.policy_sha256(),
                )
            }
            None => Ok(guider.semantic_digest_sha256().to_owned()),
        }
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeStoredPayloadError> {
        let wrapper_bytes = u64::try_from(
            mem::size_of::<Self>()
                .checked_add(self.digest_sha256.capacity())
                .and_then(|bytes| {
                    bytes.checked_add(
                        self.compiled_model
                            .as_ref()
                            .map_or(0, |compiled_model| compiled_model.resident_bytes()),
                    )
                })
                .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)?,
        )
        .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?;
        self.guider
            .resident_bytes()
            .checked_add(wrapper_bytes)
            .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)
    }

    fn owned_resident_bytes(&self) -> Result<u64, NativeStoredPayloadError> {
        let wrapper_bytes = u64::try_from(
            mem::size_of::<Self>()
                .checked_add(self.digest_sha256.capacity())
                .and_then(|bytes| {
                    bytes.checked_add(
                        self.compiled_model
                            .as_ref()
                            .map_or(0, |compiled_model| compiled_model.resident_bytes()),
                    )
                })
                .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)?,
        )
        .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?;
        self.guider
            .owned_resident_bytes()?
            .checked_add(wrapper_bytes)
            .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)
    }
}

impl std::ops::Deref for NativeStoredGuiderPayload {
    type Target = NativeGuiderPayload;

    fn deref(&self) -> &Self::Target {
        &self.guider
    }
}

#[derive(Clone)]
pub enum NativeStoredPayload {
    Tensor(Arc<NativeTensorPayload>),
    Latent(Arc<NativeLatentBundle>),
    Model(Arc<NativeStoredModelPayload>),
    Control(Arc<NativeControlPayload>),
    Conditioning(Arc<ConditioningSet>),
    Noise(Arc<NativeNoisePayload>),
    Guider(Arc<NativeStoredGuiderPayload>),
    Sampler(Arc<NativeSamplerPayload>),
    BoundingBox(Arc<NativeBoundingBoxPayload>),
    FaceLandmarks(Arc<NativeFaceLandmarksPayload>),
    PoseKeypoint(Arc<NativePoseKeypointPayload>),
    Sam3TrackData(Arc<NativeSam3TrackDataPayload>),
    Tracks(Arc<NativeTracksPayload>),
    AudioEncoderOutput(Arc<AudioEncoderOutput>),
    ClipVisionOutput(Arc<ClipVisionOutput>),
    IcLoraParameters(Arc<IcLoraParameters>),
    LossMap(Arc<LossMap>),
    Audio(Arc<NativeAudioPayload>),
    Video(Arc<NativeVideoPayload>),
    Artifact(Arc<NativeArtifactPayload>),
    File3D(Arc<NativeFile3DPayload>),
    Camera(Arc<NativeCameraPayload>),
    Splat(Arc<NativeSplatPayload>),
    Mesh(Arc<NativeMeshPayload>),
    Voxel(Arc<NativeVoxelPayload>),
    Provider(Arc<NativeProviderPayload>),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeResidentPayloadKind {
    Latent,
    Model,
    Control,
    Noise,
    Guider,
    Sampler,
    BoundingBox,
    FaceLandmarks,
    PoseKeypoint,
    Sam3TrackData,
    Tracks,
    AudioEncoderOutput,
    ClipVisionOutput,
    IcLoraParameters,
    LossMap,
    Audio,
    Video,
    Artifact,
    File3D,
    Camera,
    Splat,
    Mesh,
    Voxel,
    Provider,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeResidentAllocationId {
    PayloadArc {
        kind: NativeResidentPayloadKind,
        address: usize,
    },
    ModelPayloadArc {
        address: usize,
    },
    ModelBacking {
        kind: NativeModelBackingKind,
        address: usize,
    },
    ConditioningSetArc {
        address: usize,
    },
    DiffusionPayloadArc {
        address: usize,
    },
    ConditioningPayloadArc {
        address: usize,
    },
    PatchGraphArc {
        address: usize,
    },
    ExecutionHookGraphArc {
        address: usize,
    },
    SamplingPolicyArc {
        address: usize,
    },
    GuidanceModifierChainArc {
        address: usize,
    },
    ControlExecutionArc {
        address: usize,
    },
    ControlChainArc {
        address: usize,
    },
    ControlExecutorArc {
        address: usize,
    },
    TensorStorage {
        storage_id: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeResidentAllocation {
    id: NativeResidentAllocationId,
    resident_bytes: usize,
}

impl NativeResidentAllocation {
    pub fn id(&self) -> &NativeResidentAllocationId {
        &self.id
    }

    pub const fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePayloadResidency {
    exclusive_bytes: usize,
    shared_allocations: Vec<NativeResidentAllocation>,
}

impl NativePayloadResidency {
    fn checked(
        exclusive_bytes: usize,
        allocations: impl IntoIterator<Item = NativeResidentAllocation>,
    ) -> Result<Self, NativeStoredPayloadError> {
        let mut unique = BTreeMap::new();
        for allocation in allocations {
            if let Some(existing) = unique.insert(allocation.id.clone(), allocation.resident_bytes)
                && existing != allocation.resident_bytes
            {
                return Err(NativeStoredPayloadError::ResidentAllocationChanged);
            }
        }
        let shared_allocations = unique
            .into_iter()
            .map(|(id, resident_bytes)| NativeResidentAllocation { id, resident_bytes })
            .collect::<Vec<_>>();
        shared_allocations
            .iter()
            .try_fold(exclusive_bytes, |total, allocation| {
                total.checked_add(allocation.resident_bytes)
            })
            .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)?;
        Ok(Self {
            exclusive_bytes,
            shared_allocations,
        })
    }

    pub const fn exclusive_bytes(&self) -> usize {
        self.exclusive_bytes
    }

    pub fn shared_allocations(&self) -> &[NativeResidentAllocation] {
        &self.shared_allocations
    }

    pub fn resident_bytes(&self) -> Result<usize, NativeStoredPayloadError> {
        self.shared_allocations
            .iter()
            .try_fold(self.exclusive_bytes, |total, allocation| {
                total.checked_add(allocation.resident_bytes)
            })
            .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)
    }
}

impl NativeStoredPayload {
    pub fn validate(&self) -> Result<(), NativeStoredPayloadError> {
        match self {
            Self::Tensor(payload) => {
                payload.validate()?;
                if matches!(
                    payload.role(),
                    NativeTensorRole::Conditioning | NativeTensorRole::Latent
                ) {
                    return Err(NativeStoredPayloadError::NonCanonicalTensorRole);
                }
            }
            Self::Latent(payload) => payload.validate_retained()?,
            Self::Model(payload) => payload.validate()?,
            Self::Control(payload) => payload.validate()?,
            Self::Conditioning(payload) => payload.validate()?,
            Self::Noise(payload) => payload.validate()?,
            Self::Guider(payload) => payload.validate()?,
            Self::Sampler(payload) => payload.validate()?,
            Self::BoundingBox(payload) => payload.validate()?,
            Self::FaceLandmarks(payload) => payload.validate()?,
            Self::PoseKeypoint(payload) => payload.validate()?,
            Self::Sam3TrackData(payload) => payload.validate()?,
            Self::Tracks(payload) => payload.validate()?,
            Self::AudioEncoderOutput(payload) => payload.validate()?,
            Self::ClipVisionOutput(payload) => payload.validate()?,
            Self::IcLoraParameters(payload) => payload.validate()?,
            Self::LossMap(payload) => payload.validate()?,
            Self::Audio(payload) => payload.validate()?,
            Self::Video(payload) => payload.validate()?,
            Self::Artifact(payload) => payload.validate()?,
            Self::File3D(payload) => payload.validate()?,
            Self::Camera(payload) => payload.validate()?,
            Self::Splat(payload) => payload.validate()?,
            Self::Mesh(payload) => payload.validate()?,
            Self::Voxel(payload) => payload.validate()?,
            Self::Provider(payload) => payload.validate()?,
        }
        let handle_type = self.handle_type()?;
        handle_type.validate()?;
        if !valid_sha256(&self.digest_sha256()) {
            return Err(NativeStoredPayloadError::ProjectionChanged);
        }
        Ok(())
    }

    pub fn handle_type(&self) -> Result<NativeHandleType, NativeStoredPayloadError> {
        let source_type = match self {
            Self::Tensor(payload) => payload.handle_type_id(),
            Self::Latent(_) => "LATENT",
            Self::Model(payload) => return payload.handle_type(),
            Self::Control(payload) => payload.role().source_type_id(),
            Self::Conditioning(_) => "CONDITIONING",
            Self::Noise(_) => "NOISE",
            Self::Guider(_) => "GUIDER",
            Self::Sampler(_) => "SAMPLER",
            Self::BoundingBox(_) => NativeBoundingBoxPayload::SOURCE_TYPE_ID,
            Self::FaceLandmarks(_) => NativeFaceLandmarksPayload::SOURCE_TYPE_ID,
            Self::PoseKeypoint(_) => NativePoseKeypointPayload::SOURCE_TYPE_ID,
            Self::Sam3TrackData(_) => NativeSam3TrackDataPayload::SOURCE_TYPE_ID,
            Self::Tracks(_) => NativeTracksPayload::SOURCE_TYPE_ID,
            Self::AudioEncoderOutput(_) => AudioEncoderOutput::SOURCE_TYPE_ID,
            Self::ClipVisionOutput(_) => ClipVisionOutput::SOURCE_TYPE_ID,
            Self::IcLoraParameters(_) => IcLoraParameters::SOURCE_TYPE_ID,
            Self::LossMap(_) => LossMap::SOURCE_TYPE_ID,
            Self::Audio(_) => NativeAudioPayload::SOURCE_TYPE_ID,
            Self::Video(_) => NativeVideoPayload::SOURCE_TYPE_ID,
            Self::Artifact(payload) => payload.source_type_id(),
            Self::File3D(payload) => payload.source_type_id(),
            Self::Camera(payload) => payload.source_type_id(),
            Self::Splat(_) => NativeSplatPayload::SOURCE_TYPE_ID,
            Self::Mesh(_) => NativeMeshPayload::SOURCE_TYPE_ID,
            Self::Voxel(_) => NativeVoxelPayload::SOURCE_TYPE_ID,
            Self::Provider(payload) => return Ok(payload.handle_type().clone()),
        };
        native_source_type_projection(source_type)?
            .handle_type()?
            .ok_or_else(|| NativeStoredPayloadError::MissingHandleType(source_type.to_owned()))
    }

    pub fn digest_sha256(&self) -> String {
        match self {
            Self::Tensor(payload) => payload.projection().content_digest().to_owned(),
            Self::Latent(payload) => payload.semantic_digest_sha256().to_owned(),
            Self::Model(payload) => payload.digest_sha256().to_owned(),
            Self::Control(payload) => payload.digest_sha256().to_owned(),
            Self::Conditioning(payload) => payload.digest().to_owned(),
            Self::Noise(payload) => payload.semantic_digest_sha256().to_owned(),
            Self::Guider(payload) => payload.semantic_digest_sha256().to_owned(),
            Self::Sampler(payload) => payload.semantic_digest_sha256().to_owned(),
            Self::BoundingBox(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::FaceLandmarks(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::PoseKeypoint(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Sam3TrackData(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Tracks(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::AudioEncoderOutput(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::ClipVisionOutput(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::IcLoraParameters(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::LossMap(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Audio(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Video(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Artifact(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::File3D(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Camera(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Splat(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Mesh(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Voxel(payload) => hex_sha256(payload.semantic_digest_sha256()),
            Self::Provider(payload) => payload.identity_digest_sha256(),
        }
    }

    pub fn resident_bytes(&self) -> Result<usize, NativeStoredPayloadError> {
        match self {
            Self::Tensor(payload) => usize::try_from(payload.projection().resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Latent(payload) => usize::try_from(payload.projection().resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Model(payload) => payload.resident_bytes(),
            Self::Control(payload) => Ok(payload.resident_bytes()?),
            Self::Conditioning(payload) => usize::try_from(payload.resident_bytes()?)
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Noise(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Guider(payload) => usize::try_from(payload.resident_bytes()?)
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Sampler(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::BoundingBox(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::FaceLandmarks(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::PoseKeypoint(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Sam3TrackData(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Tracks(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::AudioEncoderOutput(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::ClipVisionOutput(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::IcLoraParameters(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::LossMap(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Audio(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Video(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Artifact(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::File3D(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Camera(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Splat(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Mesh(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Voxel(payload) => usize::try_from(payload.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow),
            Self::Provider(payload) => payload.resident_bytes(),
        }
    }

    pub fn residency(&self) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
        let residency = match self {
            Self::Tensor(payload) => {
                NativePayloadResidency::checked(0, [tensor_storage_allocation(payload)?])?
            }
            Self::Latent(payload) => latent_residency(payload)?,
            Self::Model(payload) => match &payload.resource {
                NativeStoredModelResource::Diffusion(diffusion) => {
                    stored_diffusion_residency(payload, diffusion)?
                }
                NativeStoredModelResource::ModelResource(model_payload) => {
                    let model_allocations = model_allocations(model_payload)?;
                    let model_bytes = model_payload.resident_parts()?.resident_bytes()?;
                    let wrapper_bytes = payload
                        .resident_bytes()?
                        .checked_sub(
                            usize::try_from(model_bytes)
                                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
                        )
                        .ok_or(NativeStoredPayloadError::ResidentAllocationChanged)?;
                    let mut allocations = vec![arc_allocation(
                        NativeResidentPayloadKind::Model,
                        payload,
                        wrapper_bytes,
                    )];
                    allocations.extend(model_allocations);
                    NativePayloadResidency::checked(0, allocations)?
                }
            },
            Self::Conditioning(payload) => conditioning_residency(payload)?,
            Self::Guider(payload) => guider_residency(payload)?,
            Self::Control(payload) => control_residency(payload)?,
            Self::Noise(payload) => noise_residency(payload)?,
            Self::Sampler(payload) => NativePayloadResidency::checked(
                0,
                [arc_allocation(
                    NativeResidentPayloadKind::Sampler,
                    payload,
                    usize::try_from(payload.resident_bytes())
                        .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
                )],
            )?,
            Self::BoundingBox(payload) => single_arc_residency(
                NativeResidentPayloadKind::BoundingBox,
                payload,
                payload.resident_bytes(),
            )?,
            Self::FaceLandmarks(payload) => single_arc_residency(
                NativeResidentPayloadKind::FaceLandmarks,
                payload,
                payload.resident_bytes(),
            )?,
            Self::PoseKeypoint(payload) => single_arc_residency(
                NativeResidentPayloadKind::PoseKeypoint,
                payload,
                payload.resident_bytes(),
            )?,
            Self::Sam3TrackData(payload) => media_residency(
                NativeResidentPayloadKind::Sam3TrackData,
                payload,
                payload.resident_parts()?,
            )?,
            Self::Tracks(payload) => media_residency(
                NativeResidentPayloadKind::Tracks,
                payload,
                payload.resident_parts()?,
            )?,
            Self::AudioEncoderOutput(payload) => structured_model_residency(
                NativeResidentPayloadKind::AudioEncoderOutput,
                payload,
                payload.resident_parts()?,
            )?,
            Self::ClipVisionOutput(payload) => clip_vision_output_residency(payload)?,
            Self::IcLoraParameters(payload) => single_arc_residency(
                NativeResidentPayloadKind::IcLoraParameters,
                payload,
                payload.resident_bytes(),
            )?,
            Self::LossMap(payload) => structured_model_residency(
                NativeResidentPayloadKind::LossMap,
                payload,
                payload.resident_parts()?,
            )?,
            Self::Audio(payload) => media_residency(
                NativeResidentPayloadKind::Audio,
                payload,
                payload.resident_parts()?,
            )?,
            Self::Video(payload) => media_residency(
                NativeResidentPayloadKind::Video,
                payload,
                payload.resident_parts()?,
            )?,
            Self::Artifact(payload) => single_arc_residency(
                NativeResidentPayloadKind::Artifact,
                payload,
                payload.resident_bytes(),
            )?,
            Self::File3D(payload) => single_arc_residency(
                NativeResidentPayloadKind::File3D,
                payload,
                payload.resident_bytes(),
            )?,
            Self::Camera(payload) => single_arc_residency(
                NativeResidentPayloadKind::Camera,
                payload,
                payload.resident_bytes(),
            )?,
            Self::Splat(payload) => media_residency(
                NativeResidentPayloadKind::Splat,
                payload,
                payload.resident_parts()?,
            )?,
            Self::Mesh(payload) => media_residency(
                NativeResidentPayloadKind::Mesh,
                payload,
                payload.resident_parts()?,
            )?,
            Self::Voxel(payload) => media_residency(
                NativeResidentPayloadKind::Voxel,
                payload,
                payload.resident_parts()?,
            )?,
            Self::Provider(payload) => NativePayloadResidency::checked(
                0,
                [arc_allocation(
                    NativeResidentPayloadKind::Provider,
                    payload,
                    payload.resident_bytes()?,
                )],
            )?,
        };
        if residency.resident_bytes()? != self.resident_bytes()? {
            return Err(NativeStoredPayloadError::ResidentAllocationChanged);
        }
        Ok(residency)
    }
}

fn arc_address<T>(payload: &Arc<T>) -> usize {
    Arc::as_ptr(payload) as usize
}

fn arc_allocation<T>(
    kind: NativeResidentPayloadKind,
    payload: &Arc<T>,
    resident_bytes: usize,
) -> NativeResidentAllocation {
    NativeResidentAllocation {
        id: NativeResidentAllocationId::PayloadArc {
            kind,
            address: arc_address(payload),
        },
        resident_bytes,
    }
}

fn single_arc_residency<T>(
    kind: NativeResidentPayloadKind,
    payload: &Arc<T>,
    resident_bytes: u64,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    NativePayloadResidency::checked(
        0,
        [arc_allocation(
            kind,
            payload,
            usize::try_from(resident_bytes)
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
        )],
    )
}

fn tensor_storage_allocation(
    payload: &Arc<NativeTensorPayload>,
) -> Result<NativeResidentAllocation, NativeStoredPayloadError> {
    Ok(NativeResidentAllocation {
        id: NativeResidentAllocationId::TensorStorage {
            storage_id: payload.tensor().storage_id().get(),
        },
        resident_bytes: usize::try_from(payload.projection().resident_bytes())
            .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
    })
}

fn latent_residency(
    payload: &Arc<NativeLatentBundle>,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    let parts = payload.resident_parts()?;
    tensor_backed_arc_residency(
        NativeResidentPayloadKind::Latent,
        payload,
        parts.owned_bytes(),
        parts
            .tensor_allocations()
            .iter()
            .map(|allocation| (allocation.storage_id().get(), allocation.resident_bytes())),
    )
}

fn conditioning_allocations(
    payload: &Arc<ConditioningSet>,
) -> Result<Vec<NativeResidentAllocation>, NativeStoredPayloadError> {
    let parts = payload.resident_parts()?;
    let mut allocations = Vec::with_capacity(parts.tensor_allocations().len() + 1);
    allocations.push(NativeResidentAllocation {
        id: NativeResidentAllocationId::ConditioningSetArc {
            address: arc_address(payload),
        },
        resident_bytes: usize::try_from(parts.owned_bytes())
            .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
    });
    for allocation in parts.tensor_allocations() {
        allocations.push(NativeResidentAllocation {
            id: NativeResidentAllocationId::TensorStorage {
                storage_id: allocation.storage_id().get(),
            },
            resident_bytes: usize::try_from(allocation.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
        });
    }
    Ok(allocations)
}

fn conditioning_residency(
    payload: &Arc<ConditioningSet>,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    NativePayloadResidency::checked(0, conditioning_allocations(payload)?)
}

fn guider_residency(
    payload: &Arc<NativeStoredGuiderPayload>,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    let parts = payload.guider().resident_parts()?;
    let mut allocations = vec![arc_allocation(
        NativeResidentPayloadKind::Guider,
        payload,
        usize::try_from(payload.owned_resident_bytes()?)
            .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
    )];
    allocations.extend(diffusion_allocations(&parts)?);
    NativePayloadResidency::checked(0, allocations)
}

fn clip_vision_output_residency(
    payload: &Arc<ClipVisionOutput>,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    let parts = payload.resident_parts()?;
    let mut allocations = Vec::with_capacity(parts.tensor_allocations().len() + 1);
    allocations.push(arc_allocation(
        NativeResidentPayloadKind::ClipVisionOutput,
        payload,
        usize::try_from(parts.owned_bytes())
            .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
    ));
    for allocation in parts.tensor_allocations() {
        allocations.push(NativeResidentAllocation {
            id: NativeResidentAllocationId::TensorStorage {
                storage_id: allocation.storage_id().get(),
            },
            resident_bytes: usize::try_from(allocation.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
        });
    }
    NativePayloadResidency::checked(0, allocations)
}

fn tensor_backed_arc_residency<T>(
    kind: NativeResidentPayloadKind,
    payload: &Arc<T>,
    owned_bytes: u64,
    tensor_allocations: impl IntoIterator<Item = (u64, u64)>,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    let mut allocations = vec![arc_allocation(
        kind,
        payload,
        usize::try_from(owned_bytes)
            .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
    )];
    for (storage_id, resident_bytes) in tensor_allocations {
        allocations.push(NativeResidentAllocation {
            id: NativeResidentAllocationId::TensorStorage { storage_id },
            resident_bytes: usize::try_from(resident_bytes)
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
        });
    }
    NativePayloadResidency::checked(0, allocations)
}

fn media_residency<T>(
    kind: NativeResidentPayloadKind,
    payload: &Arc<T>,
    parts: NativeMediaResidentParts,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    tensor_backed_arc_residency(
        kind,
        payload,
        parts.owned_bytes(),
        parts
            .tensor_allocations()
            .iter()
            .map(|allocation| (allocation.storage_id().get(), allocation.resident_bytes())),
    )
}

fn structured_model_residency<T>(
    kind: NativeResidentPayloadKind,
    payload: &Arc<T>,
    parts: NativeStructuredResidentParts,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    tensor_backed_arc_residency(
        kind,
        payload,
        parts.owned_bytes(),
        parts
            .tensor_allocations()
            .iter()
            .map(|allocation| (allocation.storage_id().get(), allocation.resident_bytes())),
    )
}

fn noise_residency(
    payload: &Arc<NativeNoisePayload>,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    let parts = payload.resident_parts()?;
    tensor_backed_arc_residency(
        NativeResidentPayloadKind::Noise,
        payload,
        parts.owned_bytes(),
        parts
            .tensor_allocation()
            .into_iter()
            .map(|allocation| (allocation.storage_id().get(), allocation.resident_bytes())),
    )
}

fn translate_diffusion_allocation(
    allocation: &NativeDiffusionResidentAllocation,
) -> Result<NativeResidentAllocation, NativeStoredPayloadError> {
    let id = match allocation.id() {
        NativeDiffusionResidentAllocationId::DiffusionPayloadArc { address } => {
            NativeResidentAllocationId::DiffusionPayloadArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::ConditioningSetArc { address } => {
            NativeResidentAllocationId::ConditioningSetArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::ModelPayloadArc { address } => {
            NativeResidentAllocationId::ModelPayloadArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::ModelBacking { kind, address } => {
            NativeResidentAllocationId::ModelBacking {
                kind: *kind,
                address: *address,
            }
        }
        NativeDiffusionResidentAllocationId::ConditioningPayloadArc { address } => {
            NativeResidentAllocationId::ConditioningPayloadArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::PatchGraphArc { address } => {
            NativeResidentAllocationId::PatchGraphArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::ExecutionHookGraphArc { address } => {
            NativeResidentAllocationId::ExecutionHookGraphArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::SamplingPolicyArc { address } => {
            NativeResidentAllocationId::SamplingPolicyArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::GuidanceModifierChainArc { address } => {
            NativeResidentAllocationId::GuidanceModifierChainArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::ControlExecutionArc { address } => {
            NativeResidentAllocationId::ControlExecutionArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::ControlChainArc { address } => {
            NativeResidentAllocationId::ControlChainArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::ControlExecutorArc { address } => {
            NativeResidentAllocationId::ControlExecutorArc { address: *address }
        }
        NativeDiffusionResidentAllocationId::TensorStorage { storage_id } => {
            NativeResidentAllocationId::TensorStorage {
                storage_id: *storage_id,
            }
        }
    };
    Ok(NativeResidentAllocation {
        id,
        resident_bytes: usize::try_from(allocation.resident_bytes())
            .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
    })
}

fn diffusion_allocations(
    parts: &sim_sampler::NativeDiffusionResidentParts,
) -> Result<Vec<NativeResidentAllocation>, NativeStoredPayloadError> {
    parts
        .shared_allocations()
        .iter()
        .map(translate_diffusion_allocation)
        .collect()
}

fn control_residency(
    payload: &Arc<NativeControlPayload>,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    let parts = payload.resident_parts()?;
    let mut allocations = vec![arc_allocation(
        NativeResidentPayloadKind::Control,
        payload,
        usize::try_from(parts.owned_bytes())
            .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
    )];
    allocations.extend(diffusion_allocations(&parts)?);
    NativePayloadResidency::checked(0, allocations)
}

fn stored_diffusion_residency(
    payload: &Arc<NativeStoredModelPayload>,
    diffusion: &Arc<NativeDiffusionPayload>,
) -> Result<NativePayloadResidency, NativeStoredPayloadError> {
    let parts = diffusion.resident_parts()?;
    let wrapper_bytes = payload
        .resident_bytes()?
        .checked_sub(
            usize::try_from(parts.resident_bytes()?)
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
        )
        .ok_or(NativeStoredPayloadError::ResidentAllocationChanged)?;
    let mut allocations = vec![
        arc_allocation(NativeResidentPayloadKind::Model, payload, wrapper_bytes),
        NativeResidentAllocation {
            id: NativeResidentAllocationId::DiffusionPayloadArc {
                address: arc_address(diffusion),
            },
            resident_bytes: usize::try_from(parts.owned_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
        },
    ];
    allocations.extend(diffusion_allocations(&parts)?);
    NativePayloadResidency::checked(0, allocations)
}

fn model_allocations(
    payload: &Arc<NativeModelPayload>,
) -> Result<Vec<NativeResidentAllocation>, NativeStoredPayloadError> {
    let parts = payload.resident_parts()?;
    let mut allocations = Vec::with_capacity(parts.backing_allocations().len() + 1);
    allocations.push(NativeResidentAllocation {
        id: NativeResidentAllocationId::ModelPayloadArc {
            address: arc_address(payload),
        },
        resident_bytes: usize::try_from(parts.owned_bytes())
            .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
    });
    for allocation in parts.backing_allocations() {
        allocations.push(NativeResidentAllocation {
            id: NativeResidentAllocationId::ModelBacking {
                kind: allocation.kind(),
                address: allocation.address(),
            },
            resident_bytes: usize::try_from(allocation.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
        });
    }
    for allocation in parts.tensor_allocations() {
        allocations.push(NativeResidentAllocation {
            id: NativeResidentAllocationId::TensorStorage {
                storage_id: allocation.storage_id().get(),
            },
            resident_bytes: usize::try_from(allocation.resident_bytes())
                .map_err(|_| NativeStoredPayloadError::ResidentBytesOverflow)?,
        });
    }
    Ok(allocations)
}

impl std::fmt::Debug for NativeStoredPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeStoredPayload")
            .field("handle_type", &self.handle_type().ok())
            .field("digest_sha256", &self.digest_sha256())
            .field("resident_bytes", &self.resident_bytes().ok())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum NativeStoredPayloadError {
    #[error(transparent)]
    Contract(#[from] NativeNodeContractError),
    #[error(transparent)]
    Tensor(#[from] sim_tensor::NativeTensorPayloadError),
    #[error(transparent)]
    Model(#[from] sim_model::NativeModelPayloadError),
    #[error(transparent)]
    ClipVision(#[from] sim_model::clip_vision::ClipVisionError),
    #[error(transparent)]
    Diffusion(#[from] sim_sampler::NativeDiffusionPayloadError),
    #[error(transparent)]
    Sampler(#[from] Box<sim_sampler::NativeSamplerPayloadError>),
    #[error(transparent)]
    Latent(#[from] NativeLatentBundleError),
    #[error(transparent)]
    Media(#[from] sim_media::NativeMediaPayloadError),
    #[error(transparent)]
    Conditioning(#[from] ConditioningError),
    #[error(transparent)]
    SourceType(#[from] crate::NativeSourceTypeError),
    #[error("native stored payload resident byte count overflowed")]
    ResidentBytesOverflow,
    #[error("native stored payload resident allocation projection changed")]
    ResidentAllocationChanged,
    #[error("native stored payload projection changed")]
    ProjectionChanged,
    #[error("native stored payload source type `{0}` is not handle-backed")]
    MissingHandleType(String),
    #[error("native provider payload is invalid")]
    InvalidProviderPayload,
    #[error("native tensor role must use its canonical stored payload owner")]
    NonCanonicalTensorRole,
    #[error("native model role {role:?} has no canonical non-diffusion stored payload owner")]
    NonCanonicalModelResourceRole { role: NativeModelResourceRole },
    #[error("native diffusion payload has a non-canonical model resource backing")]
    NonCanonicalDiffusionResource,
    #[error("native compiled model wrapper is invalid")]
    InvalidCompiledModel,
}

impl From<sim_sampler::NativeSamplerPayloadError> for NativeStoredPayloadError {
    fn from(error: sim_sampler::NativeSamplerPayloadError) -> Self {
        Self::Sampler(Box::new(error))
    }
}

const fn native_handle_kind_tag(kind: NativeHandleKind) -> u8 {
    match kind {
        NativeHandleKind::Tensor => 1,
        NativeHandleKind::Model => 2,
        NativeHandleKind::Clip => 3,
        NativeHandleKind::Vae => 4,
        NativeHandleKind::ControlNet => 5,
        NativeHandleKind::Conditioning => 6,
        NativeHandleKind::Latent => 7,
        NativeHandleKind::Image => 8,
        NativeHandleKind::Mask => 9,
        NativeHandleKind::Audio => 10,
        NativeHandleKind::Video => 11,
        NativeHandleKind::ThreeD => 12,
        NativeHandleKind::Artifact => 13,
        NativeHandleKind::ProviderTask => 14,
        NativeHandleKind::StructuredCompute => 15,
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex_sha256(value: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in value {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_media::{
        NativeArtifactKind, NativeCameraProjection, NativeCameraRole, NativeFile3DFormat,
        NativeFile3DRole,
    };
    use sim_model::{
        ClipVisionActivation, ClipVisionConfiguration, ClipVisionLayerWeights, ClipVisionModelType,
        ClipVisionWeights, NativeClipVision, NativeRaftLarge, raft_large_exact_native,
    };
    use sim_sampler::NativeDiffusionPayloadError;
    use sim_tensor::{
        CancellationToken, CpuWorkspaceAuthority, DType, DeviceId, NativeLatentBundle,
        NativeLatentMetadata, NativeTensorPayload, NativeTensorRole, StreamId, Tensor,
        TensorDescriptor,
    };
    use std::{collections::BTreeSet, error::Error};

    fn tensor_storage_ids(residency: &NativePayloadResidency) -> BTreeSet<u64> {
        residency
            .shared_allocations()
            .iter()
            .filter_map(|allocation| match allocation.id() {
                NativeResidentAllocationId::TensorStorage { storage_id } => Some(*storage_id),
                _ => None,
            })
            .collect()
    }

    fn loaded_zero_raft() -> Result<Arc<NativeRaftLarge>, Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let mut raft = raft_large_exact_native(false, false, &cancellation)?;
        let mut shared_tensors: Vec<(Vec<u64>, DType, Tensor)> = Vec::new();
        let mut state = BTreeMap::new();
        for spec in raft.state_schema() {
            let tensor = match shared_tensors
                .iter()
                .find(|(shape, dtype, _)| shape == &spec.shape && *dtype == spec.dtype)
            {
                Some((_, _, tensor)) => tensor.clone(),
                None => {
                    let encoded_bytes = spec
                        .shape
                        .iter()
                        .try_fold(spec.dtype.byte_width(), |bytes, dimension| {
                            bytes.checked_mul(*dimension)
                        })
                        .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)?;
                    let descriptor = TensorDescriptor::contiguous(
                        spec.shape.clone(),
                        spec.dtype,
                        DeviceId::CPU,
                        StreamId::DEFAULT,
                    )?;
                    let context = backend.execution_context(
                        StreamId::DEFAULT,
                        authority.authorize_workspace(encoded_bytes.max(1))?,
                        &cancellation,
                    );
                    let bytes = vec![
                        0_u8;
                        usize::try_from(encoded_bytes).map_err(|_| {
                            NativeStoredPayloadError::ResidentBytesOverflow
                        })?
                    ];
                    let tensor = backend.upload_bytes(descriptor, &bytes, &context)?.0;
                    shared_tensors.push((spec.shape.clone(), spec.dtype, tensor.clone()));
                    tensor
                }
            };
            state.insert(spec.name.clone(), tensor);
        }
        raft.load_state_dict(state, &cancellation)?;
        raft.eval();
        Ok(Arc::new(raft))
    }

    fn clip_tensor(
        backend: &sim_tensor::CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
        shape: Vec<u64>,
        value: f32,
    ) -> Result<Tensor, Box<dyn Error>> {
        let elements = shape
            .iter()
            .try_fold(1_u64, |total, dimension| total.checked_mul(*dimension))
            .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)?;
        let descriptor =
            TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(
                elements
                    .checked_mul(4)
                    .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)?,
            )?,
            cancellation,
        );
        Ok(backend
            .upload_f32(
                descriptor,
                &vec![value; usize::try_from(elements)?],
                &context,
            )?
            .0)
    }

    #[test]
    fn latent_bundle_is_the_only_stored_latent_owner_and_deduplicates_aliases()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let samples = clip_tensor(&backend, &authority, &cancellation, vec![1, 1, 2, 2], 1.0)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let bundle = Arc::new(NativeLatentBundle::single(
            samples.clone(),
            Some(samples.clone()),
            Some(vec![4]),
            NativeLatentMetadata::checked(None, None, Some(8), None)?,
            &context,
        )?);
        let payload = NativeStoredPayload::Latent(bundle.clone());
        payload.validate()?;
        assert_eq!(payload.handle_type()?.kind, NativeHandleKind::Latent);
        assert_eq!(payload.handle_type()?.type_id, "LATENT");
        assert_eq!(payload.digest_sha256(), bundle.semantic_digest_sha256());
        let residency = payload.residency()?;
        assert_eq!(tensor_storage_ids(&residency).len(), 1);
        assert_eq!(residency.resident_bytes()?, payload.resident_bytes()?);

        let legacy = NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_tensor(
            NativeTensorRole::Latent,
            samples,
        )?));
        assert!(matches!(
            legacy.validate(),
            Err(NativeStoredPayloadError::NonCanonicalTensorRole)
        ));
        Ok(())
    }

    fn tiny_clip_vision() -> Result<Arc<NativeClipVision>, Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let zero_2 = clip_tensor(&backend, &authority, &cancellation, vec![2], 0.0)?;
        let zero_2x2 = clip_tensor(&backend, &authority, &cancellation, vec![2, 2], 0.0)?;
        let layer = ClipVisionLayerWeights {
            layer_norm_1_weight: clip_tensor(&backend, &authority, &cancellation, vec![2], 1.0)?,
            layer_norm_1_bias: zero_2.clone(),
            query_weight: zero_2x2.clone(),
            query_bias: zero_2.clone(),
            key_weight: zero_2x2.clone(),
            key_bias: zero_2.clone(),
            value_weight: zero_2x2.clone(),
            value_bias: zero_2.clone(),
            output_weight: zero_2x2.clone(),
            output_bias: zero_2.clone(),
            layer_norm_2_weight: clip_tensor(&backend, &authority, &cancellation, vec![2], 1.0)?,
            layer_norm_2_bias: zero_2.clone(),
            feed_forward_1_weight: zero_2x2.clone(),
            feed_forward_1_bias: zero_2.clone(),
            feed_forward_2_weight: zero_2x2,
            feed_forward_2_bias: zero_2.clone(),
        };
        Ok(Arc::new(NativeClipVision::new(
            ClipVisionConfiguration {
                model_type: ClipVisionModelType::Clip,
                dtype: DType::F32,
                device: DeviceId::CPU,
                hidden_size: 2,
                intermediate_size: 2,
                attention_heads: 1,
                layer_count: 1,
                image_size: 2,
                patch_size: 1,
                num_channels: 3,
                max_num_patches: 4,
                activation: ClipVisionActivation::QuickGelu,
                projection_dimension: None,
                llava_projection_dimension: None,
            },
            ClipVisionWeights {
                patch_embedding_weight: clip_tensor(
                    &backend,
                    &authority,
                    &cancellation,
                    vec![2, 3, 1, 1],
                    0.0,
                )?,
                patch_embedding_bias: None,
                class_embedding: Some(zero_2.clone()),
                position_embedding: clip_tensor(
                    &backend,
                    &authority,
                    &cancellation,
                    vec![5, 2],
                    0.0,
                )?,
                pre_layer_norm_weight: Some(clip_tensor(
                    &backend,
                    &authority,
                    &cancellation,
                    vec![2],
                    1.0,
                )?),
                pre_layer_norm_bias: Some(zero_2.clone()),
                layers: vec![layer],
                post_layer_norm_weight: clip_tensor(
                    &backend,
                    &authority,
                    &cancellation,
                    vec![2],
                    1.0,
                )?,
                post_layer_norm_bias: zero_2,
                visual_projection_weight: None,
                llava_linear_1_weight: None,
                llava_linear_1_bias: None,
                llava_linear_2_weight: None,
                llava_linear_2_bias: None,
            },
        )?))
    }

    #[test]
    fn clip_vision_model_resource_keeps_input_model_distinct_from_structured_output()
    -> Result<(), Box<dyn Error>> {
        let clip_vision = tiny_clip_vision()?;
        let model = Arc::new(NativeModelPayload::clip_vision(clip_vision.clone())?);
        let stored = Arc::new(NativeStoredModelPayload::model_resource(model.clone())?);
        let payload = NativeStoredPayload::Model(stored.clone());
        payload.validate()?;
        let handle_type = payload.handle_type()?;
        assert_eq!(handle_type.kind, NativeHandleKind::Clip);
        assert_eq!(handle_type.type_id, "CLIP_VISION");
        assert_ne!(handle_type.type_id, ClipVisionOutput::SOURCE_TYPE_ID);
        assert_eq!(
            payload.digest_sha256(),
            clip_vision.semantic_digest_sha256()
        );
        assert!(Arc::ptr_eq(stored.model_payload(), &model));
        assert!(stored.diffusion().is_none());
        assert!(
            payload
                .residency()?
                .shared_allocations()
                .iter()
                .any(|allocation| {
                    matches!(
                        allocation.id(),
                        NativeResidentAllocationId::ModelBacking {
                            kind: NativeModelBackingKind::ClipVision,
                            address,
                        } if *address == Arc::as_ptr(&clip_vision) as usize
                    )
                })
        );
        let residency = payload.residency()?;
        let storage_ids = tensor_storage_ids(&residency);
        assert!(!storage_ids.is_empty());

        let shared_storage_resource = Arc::new(clip_vision.as_ref().clone());
        assert!(!Arc::ptr_eq(&clip_vision, &shared_storage_resource));
        let shared_storage_payload =
            NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::model_resource(
                Arc::new(NativeModelPayload::clip_vision(shared_storage_resource)?),
            )?));
        assert_eq!(
            payload.digest_sha256(),
            shared_storage_payload.digest_sha256()
        );
        assert_eq!(
            storage_ids,
            tensor_storage_ids(&shared_storage_payload.residency()?)
        );

        let distinct_storage_resource = tiny_clip_vision()?;
        let distinct_storage_payload =
            NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::model_resource(
                Arc::new(NativeModelPayload::clip_vision(distinct_storage_resource)?),
            )?));
        assert_eq!(
            payload.digest_sha256(),
            distinct_storage_payload.digest_sha256()
        );
        assert!(
            storage_ids.is_disjoint(&tensor_storage_ids(&distinct_storage_payload.residency()?))
        );
        assert!(matches!(
            NativeDiffusionPayload::clip(model.clone()),
            Err(NativeDiffusionPayloadError::RoleMismatch)
        ));
        assert!(matches!(
            NativeDiffusionPayload::vae(model),
            Err(NativeDiffusionPayloadError::RoleMismatch)
        ));
        Ok(())
    }

    #[test]
    fn clip_vision_output_residency_splits_owned_state_from_tensor_storage()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let hidden = clip_tensor(&backend, &authority, &cancellation, vec![1, 2, 2], 1.0)?;
        let embeds = clip_tensor(&backend, &authority, &cancellation, vec![1, 2], 2.0)?;
        let output = Arc::new(ClipVisionOutput::checked(
            hidden.clone(),
            Some(hidden),
            embeds,
            None,
            vec![[3, 32, 32]],
        )?);
        let parts = output.resident_parts()?;
        assert_eq!(parts.tensor_allocations().len(), 2);

        let payload = NativeStoredPayload::ClipVisionOutput(output.clone());
        payload.validate()?;
        let residency = payload.residency()?;
        assert_eq!(residency.resident_bytes()?, payload.resident_bytes()?);
        assert_eq!(tensor_storage_ids(&residency).len(), 2);
        let owned_bytes = usize::try_from(parts.owned_bytes())?;
        assert!(residency.shared_allocations().iter().any(|allocation| {
            matches!(
                allocation.id(),
                NativeResidentAllocationId::PayloadArc {
                    kind: NativeResidentPayloadKind::ClipVisionOutput,
                    address,
                } if *address == Arc::as_ptr(&output) as usize
            ) && allocation.resident_bytes() == owned_bytes
        }));
        Ok(())
    }

    #[test]
    fn optical_flow_model_resource_derives_exact_handle_identity_and_residency()
    -> Result<(), Box<dyn Error>> {
        let raft = loaded_zero_raft()?;
        let model = Arc::new(NativeModelPayload::optical_flow(raft.clone())?);
        let stored = Arc::new(NativeStoredModelPayload::model_resource(model.clone())?);
        let payload = NativeStoredPayload::Model(stored.clone());
        payload.validate()?;

        let handle_type = payload.handle_type()?;
        assert_eq!(handle_type.kind, NativeHandleKind::Model);
        assert_eq!(handle_type.type_id, "OPTICAL_FLOW");
        assert_eq!(payload.digest_sha256(), model.identity().digest_sha256());
        assert!(stored.diffusion().is_none());
        assert!(Arc::ptr_eq(stored.model_payload(), &model));
        assert_eq!(
            stored.resident_bytes()?,
            mem::size_of::<NativeStoredModelPayload>()
                .checked_add(usize::try_from(model.resident_bytes())?)
                .and_then(|bytes| bytes.checked_add(stored.digest_sha256.capacity()))
                .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)?
        );

        let residency = payload.residency()?;
        assert_eq!(residency.resident_bytes()?, payload.resident_bytes()?);
        assert_eq!(
            residency.shared_allocations().len(),
            model
                .resident_parts()?
                .tensor_allocations()
                .len()
                .checked_add(3)
                .ok_or(NativeStoredPayloadError::ResidentBytesOverflow)?
        );
        assert!(residency.shared_allocations().iter().any(|allocation| {
            matches!(
                allocation.id(),
                NativeResidentAllocationId::PayloadArc {
                    kind: NativeResidentPayloadKind::Model,
                    address,
                } if *address == Arc::as_ptr(&stored) as usize
            )
        }));
        assert!(residency.shared_allocations().iter().any(|allocation| {
            matches!(
                allocation.id(),
                NativeResidentAllocationId::ModelPayloadArc { address }
                    if *address == Arc::as_ptr(&model) as usize
            )
        }));
        assert!(residency.shared_allocations().iter().any(|allocation| {
            matches!(
                allocation.id(),
                NativeResidentAllocationId::ModelBacking {
                    kind: NativeModelBackingKind::OpticalFlow,
                    address,
                } if *address == Arc::as_ptr(&raft) as usize
            )
        }));
        Ok(())
    }

    #[test]
    fn native_model_resource_execution_closes_direct_role_admission() -> Result<(), Box<dyn Error>>
    {
        for role in [
            NativeModelResourceRole::AudioEncoder,
            NativeModelResourceRole::BackgroundRemoval,
            NativeModelResourceRole::Model,
            NativeModelResourceRole::Clip,
            NativeModelResourceRole::ClipVision,
            NativeModelResourceRole::Da3Model,
            NativeModelResourceRole::FrameInterpolation,
            NativeModelResourceRole::Gligen,
            NativeModelResourceRole::LatentUpscaleModel,
            NativeModelResourceRole::LatentOperation,
            NativeModelResourceRole::LoraModel,
            NativeModelResourceRole::MogeModel,
            NativeModelResourceRole::OpticalFlow,
            NativeModelResourceRole::Photomaker,
            NativeModelResourceRole::StyleModel,
        ] {
            require_model_resource_role(role)?;
        }
        for role in [
            NativeModelResourceRole::Vae,
            NativeModelResourceRole::ControlNet,
            NativeModelResourceRole::FaceDetection,
            NativeModelResourceRole::Hooks,
            NativeModelResourceRole::HookKeyframes,
            NativeModelResourceRole::ModelPatch,
            NativeModelResourceRole::UpscaleModel,
        ] {
            assert!(matches!(
                require_model_resource_role(role),
                Err(NativeStoredPayloadError::NonCanonicalModelResourceRole {
                    role: rejected,
                }) if rejected == role
            ));
        }

        let raft = loaded_zero_raft()?;
        let optical_flow = Arc::new(NativeModelPayload::optical_flow(raft)?);
        assert!(matches!(
            NativeDiffusionPayload::clip(optical_flow.clone()),
            Err(NativeDiffusionPayloadError::RoleMismatch)
        ));
        assert!(matches!(
            NativeDiffusionPayload::vae(optical_flow),
            Err(NativeDiffusionPayloadError::RoleMismatch)
        ));
        Ok(())
    }

    #[test]
    fn byte_backed_media_payloads_derive_exact_handles_and_residency() -> Result<(), Box<dyn Error>>
    {
        let payloads = [
            NativeStoredPayload::Artifact(Arc::new(NativeArtifactPayload::checked(
                NativeArtifactKind::Svg,
                "image/svg+xml".to_owned(),
                b"<svg xmlns=\"http://www.w3.org/2000/svg\"/>".to_vec(),
            )?)),
            NativeStoredPayload::File3D(Arc::new(NativeFile3DPayload::checked(
                NativeFile3DRole::Ply,
                NativeFile3DFormat::Ply,
                b"ply\nformat ascii 1.0\nend_header\n".to_vec(),
            )?)),
            NativeStoredPayload::Camera(Arc::new(NativeCameraPayload::checked(
                NativeCameraRole::Load3D,
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                1.0,
                None,
                NativeCameraProjection::Perspective {
                    fov_degrees: 45.0,
                    aspect_ratio: 4.0 / 3.0,
                    near: 0.01,
                    far: 100.0,
                },
                1_024,
                768,
            )?)),
        ];
        let expected = [
            (NativeHandleKind::Artifact, "SVG"),
            (NativeHandleKind::ThreeD, "FILE_3D_PLY"),
            (NativeHandleKind::ThreeD, "LOAD3D_CAMERA"),
        ];

        for (payload, (kind, type_id)) in payloads.into_iter().zip(expected) {
            payload.validate()?;
            assert_eq!(
                payload.handle_type()?,
                NativeHandleType::new(kind, type_id)?
            );
            assert!(valid_sha256(&payload.digest_sha256()));
            assert_eq!(
                payload.residency()?.resident_bytes()?,
                payload.resident_bytes()?
            );
        }
        Ok(())
    }

    #[test]
    fn provider_resident_bytes_charge_owned_capacity() -> Result<(), Box<dyn std::error::Error>> {
        let mut type_id = String::with_capacity(8_192);
        type_id.push_str("PROVIDER_TASK");
        let mut namespace = String::with_capacity(16_384);
        namespace.push_str("signed-provider");
        let semantic_digest = "a".repeat(64);
        let mut abi_bytes = Vec::with_capacity(1_048_576);
        abi_bytes.extend_from_slice(b"payload");
        let payload = NativeProviderPayload::checked(
            NativeHandleType::new(NativeHandleKind::ProviderTask, type_id)?,
            namespace,
            semantic_digest,
            abi_bytes,
        )?;

        assert!(payload.resident_bytes()? >= 8_192 + 16_384 + 1_048_576);
        Ok(())
    }
}
