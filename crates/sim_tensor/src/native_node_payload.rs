#[cfg(feature = "cpu")]
use crate::ImageTensor;
use crate::{
    CancellationToken, DType, DeviceId, Tensor, TensorDescriptor, TensorError, ViewAccess,
};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, mem};
use thiserror::Error;

pub const NATIVE_TENSOR_PROJECTION_SCHEMA_VERSION: u16 = 1;
pub const NATIVE_LATENT_BUNDLE_PROJECTION_SCHEMA_VERSION: u16 = 1;
const MAX_LATENT_SAMPLE_RATE: u32 = 768_000;
const MAX_LATENT_DOWNSCALE_RATIO: u32 = 65_536;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorRole {
    Image,
    Mask,
    Conditioning,
    Latent,
    Sigmas,
    WanCameraEmbedding,
}

impl NativeTensorRole {
    pub const fn handle_type_id(self) -> &'static str {
        match self {
            Self::Image => "IMAGE",
            Self::Mask => "MASK",
            Self::Conditioning => "CONDITIONING",
            Self::Latent => "LATENT",
            Self::Sigmas => "SIGMAS",
            Self::WanCameraEmbedding => "WAN_CAMERA_EMBEDDING",
        }
    }

    const fn digest_tag(self) -> u8 {
        match self {
            Self::Image => 1,
            Self::Mask => 2,
            Self::Conditioning => 3,
            Self::Latent => 4,
            Self::Sigmas => 5,
            Self::WanCameraEmbedding => 6,
        }
    }

    const fn accepts_image_tensor(self) -> bool {
        matches!(self, Self::Image)
    }

    const fn accepts_raw_tensor(self) -> bool {
        matches!(
            self,
            Self::Mask
                | Self::Conditioning
                | Self::Latent
                | Self::Sigmas
                | Self::WanCameraEmbedding
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeTensorProjection {
    schema_version: u16,
    role: NativeTensorRole,
    descriptor: TensorDescriptor,
    content_digest: String,
    resident_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeTensorProjectionWire {
    schema_version: u16,
    role: NativeTensorRole,
    descriptor: TensorDescriptor,
    content_digest: String,
    resident_bytes: u64,
}

impl NativeTensorProjection {
    fn new(
        role: NativeTensorRole,
        descriptor: TensorDescriptor,
        content_digest: String,
        resident_bytes: u64,
    ) -> Result<Self, NativeTensorPayloadError> {
        let projection = Self {
            schema_version: NATIVE_TENSOR_PROJECTION_SCHEMA_VERSION,
            role,
            descriptor,
            content_digest,
            resident_bytes,
        };
        projection.validate()?;
        Ok(projection)
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn role(&self) -> NativeTensorRole {
        self.role
    }

    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    pub fn content_digest(&self) -> &str {
        &self.content_digest
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn validate(&self) -> Result<(), NativeTensorPayloadError> {
        if self.schema_version != NATIVE_TENSOR_PROJECTION_SCHEMA_VERSION {
            return Err(NativeTensorPayloadError::UnsupportedSchemaVersion {
                actual: self.schema_version,
            });
        }
        if self.content_digest.len() != 64
            || !self
                .content_digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(NativeTensorPayloadError::InvalidContentDigest);
        }
        if !self.descriptor.is_contiguous()? {
            return Err(NativeTensorPayloadError::NonContiguousDescriptor);
        }
        validate_role_descriptor(self.role, &self.descriptor)?;
        let minimum = self.descriptor.minimum_backing_byte_length()?;
        if self.resident_bytes < minimum {
            return Err(NativeTensorPayloadError::ResidentBytesTooSmall {
                minimum,
                actual: self.resident_bytes,
            });
        }
        Ok(())
    }
}

impl TryFrom<NativeTensorProjectionWire> for NativeTensorProjection {
    type Error = NativeTensorPayloadError;

    fn try_from(wire: NativeTensorProjectionWire) -> Result<Self, Self::Error> {
        let projection = Self {
            schema_version: wire.schema_version,
            role: wire.role,
            descriptor: wire.descriptor,
            content_digest: wire.content_digest,
            resident_bytes: wire.resident_bytes,
        };
        projection.validate()?;
        Ok(projection)
    }
}

impl<'de> Deserialize<'de> for NativeTensorProjection {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let wire = NativeTensorProjectionWire::deserialize(deserializer)?;
        Self::try_from(wire).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug)]
enum NativeTensorStorage {
    #[cfg(feature = "cpu")]
    Image(ImageTensor),
    Tensor(Tensor),
}

#[derive(Clone, Debug)]
pub struct NativeTensorPayload {
    projection: NativeTensorProjection,
    storage: NativeTensorStorage,
}

impl NativeTensorPayload {
    #[cfg(feature = "cpu")]
    pub fn from_image(
        role: NativeTensorRole,
        image: ImageTensor,
    ) -> Result<Self, NativeTensorPayloadError> {
        if role == NativeTensorRole::Mask {
            let (batch, height, width, channels) = image.dimensions()?;
            if channels != 1 {
                return Err(NativeTensorPayloadError::RoleDescriptorMismatch { role });
            }
            let descriptor = image
                .tensor()
                .descriptor()
                .reshaped_view(vec![batch, height, width])?;
            let tensor = image.tensor().view(descriptor, ViewAccess::ReadOnly)?;
            let projection = project_tensor(role, &tensor)?;
            return Ok(Self {
                projection,
                storage: NativeTensorStorage::Tensor(tensor),
            });
        }
        if !role.accepts_image_tensor() {
            return Err(NativeTensorPayloadError::RoleStorageMismatch {
                role,
                storage: "ImageTensor",
            });
        }
        let projection = project_tensor(role, image.tensor())?;
        Ok(Self {
            projection,
            storage: NativeTensorStorage::Image(image),
        })
    }

    pub fn from_tensor(
        role: NativeTensorRole,
        tensor: Tensor,
    ) -> Result<Self, NativeTensorPayloadError> {
        if !role.accepts_raw_tensor() {
            return Err(NativeTensorPayloadError::RoleStorageMismatch {
                role,
                storage: "Tensor",
            });
        }
        let projection = project_tensor(role, &tensor)?;
        Ok(Self {
            projection,
            storage: NativeTensorStorage::Tensor(tensor),
        })
    }

    pub fn projection(&self) -> &NativeTensorProjection {
        &self.projection
    }

    pub const fn role(&self) -> NativeTensorRole {
        self.projection.role
    }

    pub fn handle_type_id(&self) -> &'static str {
        self.role().handle_type_id()
    }

    pub fn tensor(&self) -> &Tensor {
        match &self.storage {
            #[cfg(feature = "cpu")]
            NativeTensorStorage::Image(image) => image.tensor(),
            NativeTensorStorage::Tensor(tensor) => tensor,
        }
    }

    #[cfg(feature = "cpu")]
    pub fn image(&self) -> Option<&ImageTensor> {
        match &self.storage {
            NativeTensorStorage::Image(image) => Some(image),
            NativeTensorStorage::Tensor(_) => None,
        }
    }

    pub fn validate(&self) -> Result<(), NativeTensorPayloadError> {
        self.projection.validate()?;
        match &self.storage {
            #[cfg(feature = "cpu")]
            NativeTensorStorage::Image(_) if !self.role().accepts_image_tensor() => {
                return Err(NativeTensorPayloadError::RoleStorageMismatch {
                    role: self.role(),
                    storage: "ImageTensor",
                });
            }
            NativeTensorStorage::Tensor(_) if !self.role().accepts_raw_tensor() => {
                return Err(NativeTensorPayloadError::RoleStorageMismatch {
                    role: self.role(),
                    storage: "Tensor",
                });
            }
            _ => {}
        }
        let actual = project_tensor(self.role(), self.tensor())?;
        if actual != self.projection {
            return Err(NativeTensorPayloadError::ProjectionChanged);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeLatentType {
    Audio,
    #[serde(rename = "hunyuan3dv2")]
    Hunyuan3dV2,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLatentMetadata {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    latent_type: Option<NativeLatentType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(
        rename = "downscale_ratio_spacial",
        skip_serializing_if = "Option::is_none"
    )]
    spatial_downscale_ratio: Option<u32>,
    #[serde(
        rename = "downscale_ratio_temporal",
        skip_serializing_if = "Option::is_none"
    )]
    temporal_downscale_ratio: Option<u32>,
}

impl NativeLatentMetadata {
    pub fn checked(
        latent_type: Option<NativeLatentType>,
        sample_rate: Option<u32>,
        spatial_downscale_ratio: Option<u32>,
        temporal_downscale_ratio: Option<u32>,
    ) -> Result<Self, NativeLatentBundleError> {
        let metadata = Self {
            latent_type,
            sample_rate,
            spatial_downscale_ratio,
            temporal_downscale_ratio,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    pub const fn latent_type(&self) -> Option<NativeLatentType> {
        self.latent_type
    }

    pub const fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    pub const fn spatial_downscale_ratio(&self) -> Option<u32> {
        self.spatial_downscale_ratio
    }

    pub const fn temporal_downscale_ratio(&self) -> Option<u32> {
        self.temporal_downscale_ratio
    }

    fn without_downscale_ratios(&self) -> Self {
        Self {
            latent_type: self.latent_type,
            sample_rate: self.sample_rate,
            spatial_downscale_ratio: None,
            temporal_downscale_ratio: None,
        }
    }

    fn validate(&self) -> Result<(), NativeLatentBundleError> {
        if self
            .sample_rate
            .is_some_and(|value| value == 0 || value > MAX_LATENT_SAMPLE_RATE)
        {
            return Err(NativeLatentBundleError::InvalidMetadata("sample_rate"));
        }
        for (field, value) in [
            ("downscale_ratio_spacial", self.spatial_downscale_ratio),
            ("downscale_ratio_temporal", self.temporal_downscale_ratio),
        ] {
            if value.is_some_and(|value| value == 0 || value > MAX_LATENT_DOWNSCALE_RATIO) {
                return Err(NativeLatentBundleError::InvalidMetadata(field));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum NativeLatentSamples {
    Tensor(Tensor),
    AudioVideo { video: Tensor, audio: Tensor },
}

impl NativeLatentSamples {
    pub fn tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            Self::AudioVideo { .. } => None,
        }
    }

    pub fn audio_video(&self) -> Option<(&Tensor, &Tensor)> {
        match self {
            Self::Tensor(_) => None,
            Self::AudioVideo { video, audio } => Some((video, audio)),
        }
    }

    fn tensors(&self) -> [&Tensor; 2] {
        match self {
            Self::Tensor(tensor) => [tensor, tensor],
            Self::AudioVideo { video, audio } => [video, audio],
        }
    }

    fn is_nested(&self) -> bool {
        matches!(self, Self::AudioVideo { .. })
    }

    fn batch_size(&self) -> Result<u64, NativeLatentBundleError> {
        let [first, second] = self.tensors();
        let first_batch = first
            .descriptor()
            .shape()
            .first()
            .copied()
            .ok_or(NativeLatentBundleError::InvalidSamplesShape)?;
        if self.is_nested() && second.descriptor().shape().first().copied() != Some(first_batch) {
            return Err(NativeLatentBundleError::NestedBatchMismatch);
        }
        Ok(first_batch)
    }
}

#[derive(Clone, Debug)]
pub enum NativeLatentNoiseMask {
    Tensor(Tensor),
    AudioVideo { video: Tensor, audio: Tensor },
}

impl NativeLatentNoiseMask {
    pub fn tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            Self::AudioVideo { .. } => None,
        }
    }

    pub fn audio_video(&self) -> Option<(&Tensor, &Tensor)> {
        match self {
            Self::Tensor(_) => None,
            Self::AudioVideo { video, audio } => Some((video, audio)),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLatentTensorProjection {
    descriptor: TensorDescriptor,
    content_digest: String,
    resident_bytes: u64,
}

impl NativeLatentTensorProjection {
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    pub fn content_digest(&self) -> &str {
        &self.content_digest
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn validate(&self, ranks: &[usize]) -> Result<(), NativeLatentBundleError> {
        validate_latent_tensor_descriptor(&self.descriptor, ranks)?;
        if !valid_lower_sha256(&self.content_digest) {
            return Err(NativeLatentBundleError::InvalidContentDigest);
        }
        if self.resident_bytes < self.descriptor.minimum_backing_byte_length()? {
            return Err(NativeLatentBundleError::ResidentBytesTooSmall);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeLatentSamplesProjection {
    Tensor {
        tensor: NativeLatentTensorProjection,
    },
    AudioVideo {
        video: NativeLatentTensorProjection,
        audio: NativeLatentTensorProjection,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeLatentNoiseMaskProjection {
    Tensor {
        tensor: NativeLatentTensorProjection,
    },
    AudioVideo {
        video: NativeLatentTensorProjection,
        audio: NativeLatentTensorProjection,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeLatentBundleProjection {
    schema_version: u16,
    samples: NativeLatentSamplesProjection,
    #[serde(skip_serializing_if = "Option::is_none")]
    noise_mask: Option<NativeLatentNoiseMaskProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    batch_indices: Option<Vec<u64>>,
    metadata: NativeLatentMetadata,
    semantic_digest_sha256: String,
    resident_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLatentBundleProjectionWire {
    schema_version: u16,
    samples: NativeLatentSamplesProjection,
    noise_mask: Option<NativeLatentNoiseMaskProjection>,
    batch_indices: Option<Vec<u64>>,
    metadata: NativeLatentMetadata,
    semantic_digest_sha256: String,
    resident_bytes: u64,
}

impl NativeLatentBundleProjection {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn samples(&self) -> &NativeLatentSamplesProjection {
        &self.samples
    }

    pub fn noise_mask(&self) -> Option<&NativeLatentNoiseMaskProjection> {
        self.noise_mask.as_ref()
    }

    pub fn batch_indices(&self) -> Option<&[u64]> {
        self.batch_indices.as_deref()
    }

    pub fn metadata(&self) -> &NativeLatentMetadata {
        &self.metadata
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn validate(&self) -> Result<(), NativeLatentBundleError> {
        if self.schema_version != NATIVE_LATENT_BUNDLE_PROJECTION_SCHEMA_VERSION {
            return Err(NativeLatentBundleError::UnsupportedSchemaVersion {
                actual: self.schema_version,
            });
        }
        validate_samples_projection(&self.samples)?;
        if let Some(mask) = &self.noise_mask {
            validate_mask_projection(mask)?;
            validate_projection_structure(&self.samples, mask)?;
        }
        self.metadata.validate()?;
        if !valid_lower_sha256(&self.semantic_digest_sha256) {
            return Err(NativeLatentBundleError::InvalidContentDigest);
        }
        Ok(())
    }
}

impl TryFrom<NativeLatentBundleProjectionWire> for NativeLatentBundleProjection {
    type Error = NativeLatentBundleError;

    fn try_from(wire: NativeLatentBundleProjectionWire) -> Result<Self, Self::Error> {
        let projection = Self {
            schema_version: wire.schema_version,
            samples: wire.samples,
            noise_mask: wire.noise_mask,
            batch_indices: wire.batch_indices,
            metadata: wire.metadata,
            semantic_digest_sha256: wire.semantic_digest_sha256,
            resident_bytes: wire.resident_bytes,
        };
        projection.validate()?;
        Ok(projection)
    }
}

impl<'de> Deserialize<'de> for NativeLatentBundleProjection {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let wire = NativeLatentBundleProjectionWire::deserialize(deserializer)?;
        Self::try_from(wire).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeLatentTensorResidentAllocation {
    storage_id: crate::StorageId,
    resident_bytes: u64,
}

impl NativeLatentTensorResidentAllocation {
    pub const fn storage_id(&self) -> crate::StorageId {
        self.storage_id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeLatentResidentParts {
    owned_bytes: u64,
    tensor_allocations: Vec<NativeLatentTensorResidentAllocation>,
}

impl NativeLatentResidentParts {
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn tensor_allocations(&self) -> &[NativeLatentTensorResidentAllocation] {
        &self.tensor_allocations
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeLatentBundleError> {
        self.tensor_allocations
            .iter()
            .try_fold(self.owned_bytes, |total, allocation| {
                total
                    .checked_add(allocation.resident_bytes)
                    .ok_or(NativeLatentBundleError::ResidentBytesOverflow)
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeLatentMetadataRetention {
    Preserve,
    DropDownscaleRatios,
}

#[derive(Clone, Debug)]
pub struct NativeLatentBundle {
    samples: NativeLatentSamples,
    noise_mask: Option<NativeLatentNoiseMask>,
    batch_indices: Option<Vec<u64>>,
    metadata: NativeLatentMetadata,
    source_index_passthrough: bool,
    projection: NativeLatentBundleProjection,
}

impl NativeLatentBundle {
    pub fn checked(
        samples: NativeLatentSamples,
        noise_mask: Option<NativeLatentNoiseMask>,
        batch_indices: Option<Vec<u64>>,
        metadata: NativeLatentMetadata,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentBundleError> {
        Self::checked_with_index_policy(
            samples,
            noise_mask,
            batch_indices,
            metadata,
            false,
            context,
        )
    }

    pub fn replaced_samples_source_dictionary(
        &self,
        samples: NativeLatentSamples,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentBundleError> {
        Self::checked_with_index_policy(
            samples,
            self.noise_mask.clone(),
            self.batch_indices.clone(),
            self.metadata.clone(),
            true,
            context,
        )
    }

    pub fn replaced_samples_and_indices_source_dictionary(
        &self,
        samples: NativeLatentSamples,
        batch_indices: Option<Vec<u64>>,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentBundleError> {
        Self::checked_with_index_policy(
            samples,
            self.noise_mask.clone(),
            batch_indices,
            self.metadata.clone(),
            true,
            context,
        )
    }

    pub fn replaced_samples_mask_and_indices_source_dictionary(
        &self,
        samples: NativeLatentSamples,
        noise_mask: Option<NativeLatentNoiseMask>,
        batch_indices: Option<Vec<u64>>,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentBundleError> {
        Self::checked_with_index_policy(
            samples,
            noise_mask,
            batch_indices,
            self.metadata.clone(),
            true,
            context,
        )
    }

    fn checked_with_index_policy(
        samples: NativeLatentSamples,
        noise_mask: Option<NativeLatentNoiseMask>,
        batch_indices: Option<Vec<u64>>,
        metadata: NativeLatentMetadata,
        source_index_passthrough: bool,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentBundleError> {
        context.check()?;
        validate_samples(&samples, context)?;
        if let Some(mask) = &noise_mask {
            validate_mask(&samples, mask, context)?;
        }
        metadata.validate()?;
        let batch_size = samples.batch_size()?;
        if let Some(indices) = &batch_indices
            && !source_index_passthrough
        {
            if u64::try_from(indices.len()).map_err(|_| TensorError::ShapeOverflow)? != batch_size {
                return Err(NativeLatentBundleError::BatchIndexLengthMismatch);
            }
        }
        context.check()?;
        let mut bundle = Self {
            samples,
            noise_mask,
            batch_indices,
            metadata,
            source_index_passthrough,
            projection: NativeLatentBundleProjection {
                schema_version: NATIVE_LATENT_BUNDLE_PROJECTION_SCHEMA_VERSION,
                samples: NativeLatentSamplesProjection::Tensor {
                    tensor: empty_latent_tensor_projection()?,
                },
                noise_mask: None,
                batch_indices: None,
                metadata: NativeLatentMetadata::default(),
                semantic_digest_sha256: String::new(),
                resident_bytes: 0,
            },
        };
        bundle.projection = bundle.project(context)?;
        context.check()?;
        Ok(bundle)
    }

    pub fn single(
        samples: Tensor,
        noise_mask: Option<Tensor>,
        batch_indices: Option<Vec<u64>>,
        metadata: NativeLatentMetadata,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentBundleError> {
        Self::checked(
            NativeLatentSamples::Tensor(samples),
            noise_mask.map(NativeLatentNoiseMask::Tensor),
            batch_indices,
            metadata,
            context,
        )
    }

    pub fn audio_video(
        video: Tensor,
        audio: Tensor,
        video_mask: Option<Tensor>,
        audio_mask: Option<Tensor>,
        batch_indices: Option<Vec<u64>>,
        metadata: NativeLatentMetadata,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentBundleError> {
        let noise_mask = match (video_mask, audio_mask) {
            (None, None) => None,
            (Some(video), Some(audio)) => Some(NativeLatentNoiseMask::AudioVideo { video, audio }),
            _ => return Err(NativeLatentBundleError::IncompleteNestedNoiseMask),
        };
        Self::checked(
            NativeLatentSamples::AudioVideo { video, audio },
            noise_mask,
            batch_indices,
            metadata,
            context,
        )
    }

    pub fn samples(&self) -> &NativeLatentSamples {
        &self.samples
    }

    pub fn noise_mask(&self) -> Option<&NativeLatentNoiseMask> {
        self.noise_mask.as_ref()
    }

    pub fn batch_indices(&self) -> Option<&[u64]> {
        self.batch_indices.as_deref()
    }

    pub fn metadata(&self) -> &NativeLatentMetadata {
        &self.metadata
    }

    pub fn projection(&self) -> &NativeLatentBundleProjection {
        &self.projection
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        self.projection.semantic_digest_sha256()
    }

    pub fn resident_parts(&self) -> Result<NativeLatentResidentParts, NativeLatentBundleError> {
        let mut storages = BTreeMap::new();
        for tensor in latent_bundle_tensors(&self.samples, self.noise_mask.as_ref()) {
            let storage_id = tensor.storage_id();
            let resident_bytes = tensor.storage_byte_len();
            if let Some(existing) = storages.insert(storage_id.get(), (storage_id, resident_bytes))
                && existing.1 != resident_bytes
            {
                return Err(NativeLatentBundleError::ResidentAllocationChanged);
            }
        }
        let owned_bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| NativeLatentBundleError::ResidentBytesOverflow)?
            .checked_add(
                u64::try_from(self.batch_indices.as_ref().map_or(0, Vec::len))
                    .map_err(|_| NativeLatentBundleError::ResidentBytesOverflow)?
                    .checked_mul(
                        u64::try_from(mem::size_of::<u64>())
                            .map_err(|_| NativeLatentBundleError::ResidentBytesOverflow)?,
                    )
                    .ok_or(NativeLatentBundleError::ResidentBytesOverflow)?,
            )
            .ok_or(NativeLatentBundleError::ResidentBytesOverflow)?;
        let parts = NativeLatentResidentParts {
            owned_bytes,
            tensor_allocations: storages
                .into_values()
                .map(
                    |(storage_id, resident_bytes)| NativeLatentTensorResidentAllocation {
                        storage_id,
                        resident_bytes,
                    },
                )
                .collect(),
        };
        parts.resident_bytes()?;
        Ok(parts)
    }

    pub fn validate(
        &self,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<(), NativeLatentBundleError> {
        context.check()?;
        validate_samples(&self.samples, context)?;
        if let Some(mask) = &self.noise_mask {
            validate_mask(&self.samples, mask, context)?;
        }
        self.metadata.validate()?;
        if let Some(indices) = &self.batch_indices
            && !self.source_index_passthrough
            && u64::try_from(indices.len()).map_err(|_| TensorError::ShapeOverflow)?
                != self.samples.batch_size()?
        {
            return Err(NativeLatentBundleError::BatchIndexLengthMismatch);
        }
        if self.project(context)? != self.projection {
            return Err(NativeLatentBundleError::ProjectionChanged);
        }
        Ok(())
    }

    pub fn validate_retained(&self) -> Result<(), NativeLatentBundleError> {
        validate_samples_retained(&self.samples)?;
        if let Some(mask) = &self.noise_mask {
            validate_mask_retained(&self.samples, mask)?;
        }
        self.metadata.validate()?;
        if let Some(indices) = &self.batch_indices
            && !self.source_index_passthrough
            && u64::try_from(indices.len()).map_err(|_| TensorError::ShapeOverflow)?
                != self.samples.batch_size()?
        {
            return Err(NativeLatentBundleError::BatchIndexLengthMismatch);
        }
        let samples = project_samples_retained(&self.samples)?;
        let noise_mask = self
            .noise_mask
            .as_ref()
            .map(project_mask_retained)
            .transpose()?;
        let semantic_digest_sha256 = latent_bundle_digest_retained(
            &self.samples,
            self.noise_mask.as_ref(),
            self.batch_indices.as_deref(),
            &self.metadata,
        )?;
        let resident_bytes = self.resident_parts()?.resident_bytes()?;
        let projection = NativeLatentBundleProjection {
            schema_version: NATIVE_LATENT_BUNDLE_PROJECTION_SCHEMA_VERSION,
            samples,
            noise_mask,
            batch_indices: self.batch_indices.clone(),
            metadata: self.metadata.clone(),
            semantic_digest_sha256,
            resident_bytes,
        };
        projection.validate()?;
        if projection != self.projection {
            return Err(NativeLatentBundleError::ProjectionChanged);
        }
        Ok(())
    }

    pub fn replaced_samples(
        &self,
        samples: NativeLatentSamples,
        retention: NativeLatentMetadataRetention,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<Self, NativeLatentBundleError> {
        let metadata = match retention {
            NativeLatentMetadataRetention::Preserve => self.metadata.clone(),
            NativeLatentMetadataRetention::DropDownscaleRatios => {
                self.metadata.without_downscale_ratios()
            }
        };
        Self::checked_with_index_policy(
            samples,
            self.noise_mask.clone(),
            self.batch_indices.clone(),
            metadata,
            self.source_index_passthrough,
            context,
        )
    }

    fn project(
        &self,
        context: &crate::ExecutionContext<'_>,
    ) -> Result<NativeLatentBundleProjection, NativeLatentBundleError> {
        let samples = project_samples(&self.samples, context)?;
        let noise_mask = self
            .noise_mask
            .as_ref()
            .map(|mask| project_mask(mask, context))
            .transpose()?;
        let resident_bytes = self.resident_parts()?.resident_bytes()?;
        let semantic_digest_sha256 = latent_bundle_digest(
            &self.samples,
            self.noise_mask.as_ref(),
            self.batch_indices.as_deref(),
            &self.metadata,
            context,
        )?;
        let projection = NativeLatentBundleProjection {
            schema_version: NATIVE_LATENT_BUNDLE_PROJECTION_SCHEMA_VERSION,
            samples,
            noise_mask,
            batch_indices: self.batch_indices.clone(),
            metadata: self.metadata.clone(),
            semantic_digest_sha256,
            resident_bytes,
        };
        projection.validate()?;
        Ok(projection)
    }
}

fn empty_latent_tensor_projection() -> Result<NativeLatentTensorProjection, NativeLatentBundleError>
{
    Ok(NativeLatentTensorProjection {
        descriptor: TensorDescriptor::contiguous(
            vec![1, 1, 1],
            DType::F32,
            DeviceId::CPU,
            crate::StreamId::DEFAULT,
        )?,
        content_digest: "0".repeat(64),
        resident_bytes: 0,
    })
}

fn valid_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_latent_tensor_descriptor(
    descriptor: &TensorDescriptor,
    ranks: &[usize],
) -> Result<(), NativeLatentBundleError> {
    if descriptor.device() != DeviceId::CPU || descriptor.dtype() != DType::F32 {
        return Err(NativeLatentBundleError::UnsupportedTensorPlacement);
    }
    if !descriptor.is_contiguous()? {
        return Err(NativeLatentBundleError::NonContiguousTensor);
    }
    if !ranks.contains(&descriptor.rank()) || descriptor.shape().contains(&0) {
        return Err(NativeLatentBundleError::InvalidSamplesShape);
    }
    Ok(())
}

fn validate_samples(
    samples: &NativeLatentSamples,
    context: &crate::ExecutionContext<'_>,
) -> Result<(), NativeLatentBundleError> {
    match samples {
        NativeLatentSamples::Tensor(tensor) => {
            context.check()?;
            validate_latent_tensor_descriptor(tensor.descriptor(), &[3, 4, 5])?;
        }
        NativeLatentSamples::AudioVideo { video, audio } => {
            context.check()?;
            validate_latent_tensor_descriptor(video.descriptor(), &[5])?;
            context.check()?;
            // Source latent arithmetic can broadcast leading axes into the audio component.
            validate_latent_tensor_descriptor(audio.descriptor(), &[3, 4, 5])?;
            if video.descriptor().shape().first() != audio.descriptor().shape().first() {
                return Err(NativeLatentBundleError::NestedBatchMismatch);
            }
        }
    }
    Ok(())
}

fn validate_samples_retained(samples: &NativeLatentSamples) -> Result<(), NativeLatentBundleError> {
    match samples {
        NativeLatentSamples::Tensor(tensor) => {
            validate_latent_tensor_descriptor(tensor.descriptor(), &[3, 4, 5])?;
        }
        NativeLatentSamples::AudioVideo { video, audio } => {
            validate_latent_tensor_descriptor(video.descriptor(), &[5])?;
            validate_latent_tensor_descriptor(audio.descriptor(), &[3, 4, 5])?;
            if video.descriptor().shape().first() != audio.descriptor().shape().first() {
                return Err(NativeLatentBundleError::NestedBatchMismatch);
            }
        }
    }
    Ok(())
}

fn validate_mask(
    samples: &NativeLatentSamples,
    mask: &NativeLatentNoiseMask,
    context: &crate::ExecutionContext<'_>,
) -> Result<(), NativeLatentBundleError> {
    match (samples, mask) {
        (NativeLatentSamples::Tensor(samples), NativeLatentNoiseMask::Tensor(mask)) => {
            context.check()?;
            validate_latent_tensor_descriptor(mask.descriptor(), &[3, 4, 5])?;
            validate_mask_geometry(samples.descriptor(), mask.descriptor())?;
        }
        (
            NativeLatentSamples::AudioVideo { video, audio },
            NativeLatentNoiseMask::AudioVideo {
                video: video_mask,
                audio: audio_mask,
            },
        ) => {
            context.check()?;
            validate_latent_tensor_descriptor(video_mask.descriptor(), &[5])?;
            validate_mask_geometry(video.descriptor(), video_mask.descriptor())?;
            context.check()?;
            validate_latent_tensor_descriptor(audio_mask.descriptor(), &[3])?;
            validate_mask_geometry(audio.descriptor(), audio_mask.descriptor())?;
        }
        _ => return Err(NativeLatentBundleError::NoiseMaskStructureMismatch),
    }
    Ok(())
}

fn validate_mask_retained(
    samples: &NativeLatentSamples,
    mask: &NativeLatentNoiseMask,
) -> Result<(), NativeLatentBundleError> {
    match (samples, mask) {
        (NativeLatentSamples::Tensor(samples), NativeLatentNoiseMask::Tensor(mask)) => {
            validate_latent_tensor_descriptor(mask.descriptor(), &[3, 4, 5])?;
            validate_mask_geometry(samples.descriptor(), mask.descriptor())?;
        }
        (
            NativeLatentSamples::AudioVideo { video, audio },
            NativeLatentNoiseMask::AudioVideo {
                video: video_mask,
                audio: audio_mask,
            },
        ) => {
            validate_latent_tensor_descriptor(video_mask.descriptor(), &[5])?;
            validate_mask_geometry(video.descriptor(), video_mask.descriptor())?;
            validate_latent_tensor_descriptor(audio_mask.descriptor(), &[3])?;
            validate_mask_geometry(audio.descriptor(), audio_mask.descriptor())?;
        }
        _ => return Err(NativeLatentBundleError::NoiseMaskStructureMismatch),
    }
    Ok(())
}

fn validate_mask_geometry(
    samples: &TensorDescriptor,
    mask: &TensorDescriptor,
) -> Result<(), NativeLatentBundleError> {
    // Source LATENT dictionaries retain the original mask; sampling resizes it later.
    if samples.rank() == 3 && mask.rank() != 3 {
        return Err(NativeLatentBundleError::NoiseMaskGeometryMismatch);
    }
    Ok(())
}

fn project_latent_tensor(
    tensor: &Tensor,
    context: &crate::ExecutionContext<'_>,
) -> Result<NativeLatentTensorProjection, NativeLatentBundleError> {
    context.check()?;
    let descriptor = tensor.descriptor().clone();
    let content_digest = latent_tensor_digest(&descriptor, tensor.contiguous_bytes()?)?;
    Ok(NativeLatentTensorProjection {
        descriptor,
        content_digest,
        resident_bytes: tensor.storage_byte_len(),
    })
}

fn project_samples(
    samples: &NativeLatentSamples,
    context: &crate::ExecutionContext<'_>,
) -> Result<NativeLatentSamplesProjection, NativeLatentBundleError> {
    Ok(match samples {
        NativeLatentSamples::Tensor(tensor) => NativeLatentSamplesProjection::Tensor {
            tensor: project_latent_tensor(tensor, context)?,
        },
        NativeLatentSamples::AudioVideo { video, audio } => {
            NativeLatentSamplesProjection::AudioVideo {
                video: project_latent_tensor(video, context)?,
                audio: project_latent_tensor(audio, context)?,
            }
        }
    })
}

fn project_samples_retained(
    samples: &NativeLatentSamples,
) -> Result<NativeLatentSamplesProjection, NativeLatentBundleError> {
    Ok(match samples {
        NativeLatentSamples::Tensor(tensor) => NativeLatentSamplesProjection::Tensor {
            tensor: project_latent_tensor_retained(tensor)?,
        },
        NativeLatentSamples::AudioVideo { video, audio } => {
            NativeLatentSamplesProjection::AudioVideo {
                video: project_latent_tensor_retained(video)?,
                audio: project_latent_tensor_retained(audio)?,
            }
        }
    })
}

fn project_mask(
    mask: &NativeLatentNoiseMask,
    context: &crate::ExecutionContext<'_>,
) -> Result<NativeLatentNoiseMaskProjection, NativeLatentBundleError> {
    Ok(match mask {
        NativeLatentNoiseMask::Tensor(tensor) => NativeLatentNoiseMaskProjection::Tensor {
            tensor: project_latent_tensor(tensor, context)?,
        },
        NativeLatentNoiseMask::AudioVideo { video, audio } => {
            NativeLatentNoiseMaskProjection::AudioVideo {
                video: project_latent_tensor(video, context)?,
                audio: project_latent_tensor(audio, context)?,
            }
        }
    })
}

fn project_mask_retained(
    mask: &NativeLatentNoiseMask,
) -> Result<NativeLatentNoiseMaskProjection, NativeLatentBundleError> {
    Ok(match mask {
        NativeLatentNoiseMask::Tensor(tensor) => NativeLatentNoiseMaskProjection::Tensor {
            tensor: project_latent_tensor_retained(tensor)?,
        },
        NativeLatentNoiseMask::AudioVideo { video, audio } => {
            NativeLatentNoiseMaskProjection::AudioVideo {
                video: project_latent_tensor_retained(video)?,
                audio: project_latent_tensor_retained(audio)?,
            }
        }
    })
}

fn project_latent_tensor_retained(
    tensor: &Tensor,
) -> Result<NativeLatentTensorProjection, NativeLatentBundleError> {
    let descriptor = tensor.descriptor().clone();
    let content_digest = latent_tensor_digest(&descriptor, tensor.contiguous_bytes()?)?;
    Ok(NativeLatentTensorProjection {
        descriptor,
        content_digest,
        resident_bytes: tensor.storage_byte_len(),
    })
}

fn validate_samples_projection(
    samples: &NativeLatentSamplesProjection,
) -> Result<(), NativeLatentBundleError> {
    match samples {
        NativeLatentSamplesProjection::Tensor { tensor } => tensor.validate(&[3, 4, 5]),
        NativeLatentSamplesProjection::AudioVideo { video, audio } => {
            video.validate(&[5])?;
            audio.validate(&[3, 4, 5])?;
            if video.descriptor.shape().first() != audio.descriptor.shape().first() {
                return Err(NativeLatentBundleError::NestedBatchMismatch);
            }
            Ok(())
        }
    }
}

fn validate_mask_projection(
    mask: &NativeLatentNoiseMaskProjection,
) -> Result<(), NativeLatentBundleError> {
    match mask {
        NativeLatentNoiseMaskProjection::Tensor { tensor } => tensor.validate(&[3, 4, 5]),
        NativeLatentNoiseMaskProjection::AudioVideo { video, audio } => {
            video.validate(&[5])?;
            audio.validate(&[3])
        }
    }
}

fn validate_projection_structure(
    samples: &NativeLatentSamplesProjection,
    mask: &NativeLatentNoiseMaskProjection,
) -> Result<(), NativeLatentBundleError> {
    match (samples, mask) {
        (
            NativeLatentSamplesProjection::Tensor { tensor: samples },
            NativeLatentNoiseMaskProjection::Tensor { tensor: mask },
        ) => validate_mask_geometry(&samples.descriptor, &mask.descriptor),
        (
            NativeLatentSamplesProjection::AudioVideo { video, audio },
            NativeLatentNoiseMaskProjection::AudioVideo {
                video: video_mask,
                audio: audio_mask,
            },
        ) => {
            validate_mask_geometry(&video.descriptor, &video_mask.descriptor)?;
            validate_mask_geometry(&audio.descriptor, &audio_mask.descriptor)
        }
        _ => Err(NativeLatentBundleError::NoiseMaskStructureMismatch),
    }
}

fn latent_bundle_tensors<'a>(
    samples: &'a NativeLatentSamples,
    mask: Option<&'a NativeLatentNoiseMask>,
) -> Vec<&'a Tensor> {
    let mut tensors = match samples {
        NativeLatentSamples::Tensor(tensor) => vec![tensor],
        NativeLatentSamples::AudioVideo { video, audio } => vec![video, audio],
    };
    match mask {
        Some(NativeLatentNoiseMask::Tensor(tensor)) => tensors.push(tensor),
        Some(NativeLatentNoiseMask::AudioVideo { video, audio }) => {
            tensors.push(video);
            tensors.push(audio);
        }
        None => {}
    }
    tensors
}

fn latent_tensor_digest(
    descriptor: &TensorDescriptor,
    bytes: &[u8],
) -> Result<String, NativeLatentBundleError> {
    latent_tensor_digest_with_cancellation(descriptor, bytes, None)
}

fn latent_tensor_digest_with_cancellation(
    descriptor: &TensorDescriptor,
    bytes: &[u8],
    cancellation: Option<&CancellationToken>,
) -> Result<String, NativeLatentBundleError> {
    latent_tensor_digest_with_check(descriptor, bytes, || {
        if let Some(cancellation) = cancellation {
            cancellation.check()?;
        }
        Ok(())
    })
}

fn latent_tensor_digest_with_check(
    descriptor: &TensorDescriptor,
    bytes: &[u8],
    mut check: impl FnMut() -> Result<(), sim_types::CancellationError>,
) -> Result<String, NativeLatentBundleError> {
    let expected = descriptor.byte_len()?;
    let actual = u64::try_from(bytes.len()).map_err(|_| TensorError::ShapeOverflow)?;
    if expected != actual {
        return Err(NativeLatentBundleError::ContentByteLength { expected, actual });
    }
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.native-latent-tensor.semantic.v1");
    hasher.update(serde_json::to_vec(&descriptor.dtype())?);
    hasher.update(
        u64::try_from(descriptor.rank())
            .map_err(|_| TensorError::ShapeOverflow)?
            .to_le_bytes(),
    );
    for dimension in descriptor.shape() {
        hasher.update(dimension.to_le_bytes());
    }
    for chunk in bytes.chunks(64 * 1024) {
        check()?;
        hasher.update(chunk);
    }
    check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn latent_bundle_digest(
    samples: &NativeLatentSamples,
    mask: Option<&NativeLatentNoiseMask>,
    batch_indices: Option<&[u64]>,
    metadata: &NativeLatentMetadata,
    context: &crate::ExecutionContext<'_>,
) -> Result<String, NativeLatentBundleError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.native-latent-bundle.semantic.v1");
    match samples {
        NativeLatentSamples::Tensor(tensor) => {
            hasher.update([1]);
            context.check()?;
            hasher.update(latent_tensor_digest_with_cancellation(
                tensor.descriptor(),
                tensor.contiguous_bytes()?,
                Some(context.cancellation),
            )?);
        }
        NativeLatentSamples::AudioVideo { video, audio } => {
            hasher.update([2]);
            for tensor in [video, audio] {
                context.check()?;
                hasher.update(latent_tensor_digest_with_cancellation(
                    tensor.descriptor(),
                    tensor.contiguous_bytes()?,
                    Some(context.cancellation),
                )?);
            }
        }
    }
    match mask {
        None => hasher.update([0]),
        Some(NativeLatentNoiseMask::Tensor(tensor)) => {
            hasher.update([1]);
            context.check()?;
            hasher.update(latent_tensor_digest_with_cancellation(
                tensor.descriptor(),
                tensor.contiguous_bytes()?,
                Some(context.cancellation),
            )?);
        }
        Some(NativeLatentNoiseMask::AudioVideo { video, audio }) => {
            hasher.update([2]);
            for tensor in [video, audio] {
                context.check()?;
                hasher.update(latent_tensor_digest_with_cancellation(
                    tensor.descriptor(),
                    tensor.contiguous_bytes()?,
                    Some(context.cancellation),
                )?);
            }
        }
    }
    match batch_indices {
        None => hasher.update([0]),
        Some(indices) => {
            hasher.update([1]);
            hasher.update(
                u64::try_from(indices.len())
                    .map_err(|_| TensorError::ShapeOverflow)?
                    .to_le_bytes(),
            );
            for index in indices {
                hasher.update(index.to_le_bytes());
            }
        }
    }
    hasher.update(serde_json::to_vec(metadata)?);
    context.check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn latent_bundle_digest_retained(
    samples: &NativeLatentSamples,
    mask: Option<&NativeLatentNoiseMask>,
    batch_indices: Option<&[u64]>,
    metadata: &NativeLatentMetadata,
) -> Result<String, NativeLatentBundleError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.native-latent-bundle.semantic.v1");
    match samples {
        NativeLatentSamples::Tensor(tensor) => {
            hasher.update([1]);
            hasher.update(latent_tensor_digest(
                tensor.descriptor(),
                tensor.contiguous_bytes()?,
            )?);
        }
        NativeLatentSamples::AudioVideo { video, audio } => {
            hasher.update([2]);
            for tensor in [video, audio] {
                hasher.update(latent_tensor_digest(
                    tensor.descriptor(),
                    tensor.contiguous_bytes()?,
                )?);
            }
        }
    }
    match mask {
        None => hasher.update([0]),
        Some(NativeLatentNoiseMask::Tensor(tensor)) => {
            hasher.update([1]);
            hasher.update(latent_tensor_digest(
                tensor.descriptor(),
                tensor.contiguous_bytes()?,
            )?);
        }
        Some(NativeLatentNoiseMask::AudioVideo { video, audio }) => {
            hasher.update([2]);
            for tensor in [video, audio] {
                hasher.update(latent_tensor_digest(
                    tensor.descriptor(),
                    tensor.contiguous_bytes()?,
                )?);
            }
        }
    }
    match batch_indices {
        None => hasher.update([0]),
        Some(indices) => {
            hasher.update([1]);
            hasher.update(
                u64::try_from(indices.len())
                    .map_err(|_| TensorError::ShapeOverflow)?
                    .to_le_bytes(),
            );
            for index in indices {
                hasher.update(index.to_le_bytes());
            }
        }
    }
    hasher.update(serde_json::to_vec(metadata)?);
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, Error)]
pub enum NativeLatentBundleError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("native latent projection encoding failed: {0}")]
    Encoding(#[from] serde_json::Error),
    #[error("native latent projection schema version {actual} is unsupported")]
    UnsupportedSchemaVersion { actual: u16 },
    #[error("native latent tensor placement must be contiguous CPU F32")]
    UnsupportedTensorPlacement,
    #[error("native latent tensor must be contiguous")]
    NonContiguousTensor,
    #[error("native latent samples must be a nonempty rank-three, rank-four, or rank-five tensor")]
    InvalidSamplesShape,
    #[error("native nested audio/video latent components must share a batch dimension")]
    NestedBatchMismatch,
    #[error("native latent noise-mask structure does not match its samples")]
    NoiseMaskStructureMismatch,
    #[error("native nested latent noise masks must contain both video and audio components")]
    IncompleteNestedNoiseMask,
    #[error("native latent noise-mask rank cannot be prepared for its samples")]
    NoiseMaskGeometryMismatch,
    #[error("native latent batch-index length does not match its sample batch")]
    BatchIndexLengthMismatch,
    #[error("native latent metadata field `{0}` is out of bounds")]
    InvalidMetadata(&'static str),
    #[error("native latent digest must be 64 lowercase hexadecimal characters")]
    InvalidContentDigest,
    #[error("native latent resident byte projection is smaller than its tensor descriptor")]
    ResidentBytesTooSmall,
    #[error("native latent content requires {expected} bytes, got {actual}")]
    ContentByteLength { expected: u64, actual: u64 },
    #[error("native latent resident byte count overflowed")]
    ResidentBytesOverflow,
    #[error("native latent resident allocation projection changed")]
    ResidentAllocationChanged,
    #[error("native latent projection no longer matches its retained payload")]
    ProjectionChanged,
}

impl From<sim_types::CancellationError> for NativeLatentBundleError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Tensor(TensorError::Cancelled)
    }
}

pub fn native_tensor_digest(
    role: NativeTensorRole,
    descriptor: &TensorDescriptor,
    contiguous_bytes: &[u8],
) -> Result<String, NativeTensorPayloadError> {
    if !descriptor.is_contiguous()? {
        return Err(NativeTensorPayloadError::NonContiguousDescriptor);
    }
    let expected = descriptor.byte_len()?;
    let actual = u64::try_from(contiguous_bytes.len()).map_err(|_| TensorError::ShapeOverflow)?;
    if actual != expected {
        return Err(NativeTensorPayloadError::ContentByteLength { expected, actual });
    }
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.native-tensor.semantic.v1");
    hasher.update([role.digest_tag()]);
    hasher.update(serde_json::to_vec(&descriptor.dtype())?);
    hasher.update(
        u64::try_from(descriptor.rank())
            .map_err(|_| TensorError::ShapeOverflow)?
            .to_le_bytes(),
    );
    for dimension in descriptor.shape() {
        hasher.update(dimension.to_le_bytes());
    }
    hasher.update(contiguous_bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn project_tensor(
    role: NativeTensorRole,
    tensor: &Tensor,
) -> Result<NativeTensorProjection, NativeTensorPayloadError> {
    let descriptor = tensor.descriptor().clone();
    let content_digest = native_tensor_digest(role, &descriptor, tensor.contiguous_bytes()?)?;
    NativeTensorProjection::new(role, descriptor, content_digest, tensor.storage_byte_len())
}

fn validate_role_descriptor(
    role: NativeTensorRole,
    descriptor: &TensorDescriptor,
) -> Result<(), NativeTensorPayloadError> {
    if descriptor.device() != DeviceId::CPU || descriptor.dtype() != DType::F32 {
        return Err(NativeTensorPayloadError::RoleDescriptorMismatch { role });
    }
    let shape = descriptor.shape();
    let valid = match role {
        NativeTensorRole::Image => {
            shape.len() == 4
                && shape
                    .get(3)
                    .is_some_and(|channels| matches!(*channels, 1 | 3 | 4))
        }
        NativeTensorRole::Mask => shape.len() == 3,
        NativeTensorRole::Conditioning => shape.len() == 3,
        NativeTensorRole::Latent => shape.len() == 4,
        NativeTensorRole::Sigmas => shape.len() == 1,
        NativeTensorRole::WanCameraEmbedding => !shape.is_empty(),
    };
    if valid {
        Ok(())
    } else {
        Err(NativeTensorPayloadError::RoleDescriptorMismatch { role })
    }
}

#[derive(Debug, Error)]
pub enum NativeTensorPayloadError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("native tensor descriptor encoding failed: {0}")]
    Encoding(#[from] serde_json::Error),
    #[error("native tensor projection schema version {actual} is unsupported")]
    UnsupportedSchemaVersion { actual: u16 },
    #[error("native tensor projection content digest must be 64 lowercase hexadecimal characters")]
    InvalidContentDigest,
    #[error("native tensor projection descriptor must be contiguous")]
    NonContiguousDescriptor,
    #[error("native tensor projection requires at least {minimum} resident bytes, got {actual}")]
    ResidentBytesTooSmall { minimum: u64, actual: u64 },
    #[error("native tensor content requires {expected} bytes, got {actual}")]
    ContentByteLength { expected: u64, actual: u64 },
    #[error("native tensor role {role:?} cannot contain {storage} storage")]
    RoleStorageMismatch {
        role: NativeTensorRole,
        storage: &'static str,
    },
    #[error("native tensor descriptor does not match role {role:?}")]
    RoleDescriptorMismatch { role: NativeTensorRole },
    #[error("native tensor projection no longer matches its compute payload")]
    ProjectionChanged,
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::{
        CancellationToken, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext, ImageTensor,
        StreamId, TensorDescriptor,
    };
    use serde_json::json;
    use std::error::Error;

    const TEST_MEMORY_LIMIT_BYTES: u64 = 1024 * 1024;

    fn with_context<ResultType>(
        run: impl FnOnce(
            &crate::CpuBackend,
            &ExecutionContext<'_>,
        ) -> Result<ResultType, Box<dyn Error>>,
    ) -> Result<ResultType, Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        run(&backend, &context)
    }

    fn tensor_payload(
        role: NativeTensorRole,
        values: &[f32],
    ) -> Result<NativeTensorPayload, Box<dyn Error>> {
        with_context(|backend, context| {
            let value_count = u64::try_from(values.len())?;
            let shape = match role {
                NativeTensorRole::Conditioning => vec![1, 1, value_count],
                NativeTensorRole::Latent => vec![1, 1, 1, value_count],
                NativeTensorRole::Sigmas => vec![value_count],
                NativeTensorRole::WanCameraEmbedding => vec![1, value_count],
                NativeTensorRole::Image | NativeTensorRole::Mask => vec![1, value_count],
            };
            let descriptor =
                TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
            let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
            Ok(NativeTensorPayload::from_tensor(role, tensor)?)
        })
    }

    fn upload_tensor(
        backend: &crate::CpuBackend,
        context: &ExecutionContext<'_>,
        shape: Vec<u64>,
        values: &[f32],
    ) -> Result<Tensor, Box<dyn Error>> {
        let descriptor =
            TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
        Ok(tensor)
    }

    #[test]
    fn nested_latent_projection_preserves_broadcast_audio_axes() -> Result<(), Box<dyn Error>> {
        with_context(|backend, context| {
            let video = upload_tensor(backend, context, vec![1, 2, 1, 1, 4], &[7.0; 8])?;
            for shape in [vec![1, 3, 4], vec![1, 1, 3, 4], vec![1, 1, 1, 3, 4]] {
                let audio = upload_tensor(backend, context, shape, &[8.0; 12])?;
                let bundle = NativeLatentBundle::audio_video(
                    video.clone(),
                    audio,
                    None,
                    None,
                    None,
                    Default::default(),
                    context,
                )?;
                bundle.validate(context)?;
                bundle.validate_retained()?;
                let restored: NativeLatentBundleProjection =
                    serde_json::from_slice(&serde_json::to_vec(bundle.projection())?)?;
                restored.validate()?;
                assert_eq!(&restored, bundle.projection());
            }
            Ok(())
        })
    }

    #[test]
    fn hunyuan_3d_latent_metadata_survives_projection_and_sample_replacement()
    -> Result<(), Box<dyn Error>> {
        with_context(|backend, context| {
            let samples = upload_tensor(backend, context, vec![1, 64, 1], &[0.0; 64])?;
            let metadata = NativeLatentMetadata::checked(
                Some(NativeLatentType::Hunyuan3dV2),
                None,
                None,
                None,
            )?;
            let bundle =
                NativeLatentBundle::single(samples.clone(), None, None, metadata, context)?;
            let encoded = serde_json::to_value(bundle.projection())?;
            assert_eq!(encoded["metadata"], json!({"type": "hunyuan3dv2"}));
            let decoded: NativeLatentBundleProjection = serde_json::from_value(encoded)?;
            assert_eq!(&decoded, bundle.projection());
            for retention in [
                NativeLatentMetadataRetention::Preserve,
                NativeLatentMetadataRetention::DropDownscaleRatios,
            ] {
                let replaced = bundle.replaced_samples(
                    NativeLatentSamples::Tensor(samples.clone()),
                    retention,
                    context,
                )?;
                assert_eq!(
                    replaced.metadata().latent_type(),
                    Some(NativeLatentType::Hunyuan3dV2)
                );
                assert_eq!(replaced.projection(), bundle.projection());
            }
            Ok(())
        })
    }

    #[test]
    fn latent_bundle_preserves_ranks_masks_batch_indices_and_exact_metadata_wire_names()
    -> Result<(), Box<dyn Error>> {
        with_context(|backend, context| {
            for shape in [vec![2, 1, 3], vec![2, 1, 2, 3], vec![2, 1, 2, 2, 3]] {
                let count = shape.iter().copied().product::<u64>();
                let values = (0..usize::try_from(count)?)
                    .map(|index| index as f32)
                    .collect::<Vec<_>>();
                let samples = upload_tensor(backend, context, shape.clone(), &values)?;
                let mask_shape = shape
                    .iter()
                    .enumerate()
                    .map(|(index, dimension)| if index == 0 { 1 } else { *dimension })
                    .collect::<Vec<_>>();
                let mask_count = mask_shape.iter().copied().product::<u64>();
                let mask = upload_tensor(
                    backend,
                    context,
                    mask_shape,
                    &vec![1.0; usize::try_from(mask_count)?],
                )?;
                let metadata = NativeLatentMetadata::checked(
                    Some(NativeLatentType::Audio),
                    Some(48_000),
                    Some(8),
                    Some(4),
                )?;
                let bundle = NativeLatentBundle::single(
                    samples,
                    Some(mask),
                    Some(vec![7, 7]),
                    metadata,
                    context,
                )?;
                bundle.validate(context)?;
                bundle.validate_retained()?;
                assert_eq!(bundle.batch_indices(), Some([7, 7].as_slice()));
                let encoded = serde_json::to_value(bundle.projection())?;
                assert_eq!(encoded["metadata"]["type"], json!("audio"));
                assert_eq!(encoded["metadata"]["sample_rate"], json!(48_000));
                assert_eq!(encoded["metadata"]["downscale_ratio_spacial"], json!(8));
                assert_eq!(encoded["metadata"]["downscale_ratio_temporal"], json!(4));
                assert!(encoded["metadata"].get("spatial_downscale_ratio").is_none());
                let decoded: NativeLatentBundleProjection = serde_json::from_value(encoded)?;
                assert_eq!(&decoded, bundle.projection());
            }
            Ok(())
        })
    }

    #[test]
    fn nested_audio_video_latents_are_ordered_and_alias_residency_is_counted_once()
    -> Result<(), Box<dyn Error>> {
        with_context(|backend, context| {
            let video = upload_tensor(backend, context, vec![1, 1, 2, 1, 2], &[1.0; 4])?;
            let audio = upload_tensor(backend, context, vec![1, 1, 3], &[2.0; 3])?;
            let bundle = NativeLatentBundle::audio_video(
                video.clone(),
                audio.clone(),
                Some(video),
                Some(audio),
                Some(vec![3]),
                NativeLatentMetadata::default(),
                context,
            )?;
            let (retained_video, retained_audio) = bundle
                .samples()
                .audio_video()
                .ok_or("nested samples are missing")?;
            assert_eq!(retained_video.descriptor().shape(), &[1, 1, 2, 1, 2]);
            assert_eq!(retained_audio.descriptor().shape(), &[1, 1, 3]);
            assert_eq!(bundle.resident_parts()?.tensor_allocations().len(), 2);
            bundle.validate_retained()?;
            Ok(())
        })
    }

    #[test]
    fn source_dictionary_replacement_validates_atomic_mask_and_index_changes()
    -> Result<(), Box<dyn Error>> {
        with_context(|backend, context| {
            let samples = upload_tensor(backend, context, vec![5, 2, 3, 5], &[0.0; 150])?;
            let mask = upload_tensor(backend, context, vec![2, 1, 3, 5], &[1.0; 30])?;
            let metadata = NativeLatentMetadata::checked(None, None, Some(8), Some(4))?;
            let bundle = NativeLatentBundle::single(
                samples,
                Some(mask),
                Some(vec![10, 11, 12, 13, 14]),
                metadata,
                context,
            )?;
            let original_projection = bundle.projection().clone();
            let replacement = upload_tensor(backend, context, vec![3, 2, 3, 5], &[2.0; 90])?;
            let replacement_mask = upload_tensor(backend, context, vec![3, 1, 3, 5], &[0.5; 45])?;
            for indices in [vec![], vec![99], vec![12, 13, 14], vec![10, 11, 12, 13]] {
                let strict = NativeLatentBundle::checked(
                    NativeLatentSamples::Tensor(replacement.clone()),
                    Some(NativeLatentNoiseMask::Tensor(replacement_mask.clone())),
                    Some(indices.clone()),
                    bundle.metadata().clone(),
                    context,
                );
                if indices.len() == 3 {
                    strict?.validate(context)?;
                } else {
                    assert!(matches!(
                        strict,
                        Err(NativeLatentBundleError::BatchIndexLengthMismatch)
                    ));
                }
                let replaced = bundle.replaced_samples_mask_and_indices_source_dictionary(
                    NativeLatentSamples::Tensor(replacement.clone()),
                    Some(NativeLatentNoiseMask::Tensor(replacement_mask.clone())),
                    Some(indices.clone()),
                    context,
                )?;
                replaced.validate(context)?;
                replaced.validate_retained()?;
                assert_eq!(replaced.batch_indices(), Some(indices.as_slice()));
                assert_eq!(replaced.metadata(), bundle.metadata());
                assert_ne!(
                    replaced.semantic_digest_sha256(),
                    bundle.semantic_digest_sha256()
                );
                let restored: NativeLatentBundleProjection =
                    serde_json::from_slice(&serde_json::to_vec(replaced.projection())?)?;
                restored.validate()?;
                assert_eq!(&restored, replaced.projection());
                assert_eq!(bundle.projection(), &original_projection);
                bundle.validate(context)?;
            }
            let cleared = bundle.replaced_samples_mask_and_indices_source_dictionary(
                NativeLatentSamples::Tensor(replacement.clone()),
                None,
                None,
                context,
            )?;
            cleared.validate(context)?;
            cleared.validate_retained()?;
            assert!(cleared.noise_mask().is_none());
            assert!(cleared.batch_indices().is_none());
            assert_eq!(cleared.metadata(), bundle.metadata());
            let audio_samples = upload_tensor(backend, context, vec![3, 2, 5], &[0.0; 30])?;
            assert!(matches!(
                bundle.replaced_samples_mask_and_indices_source_dictionary(
                    NativeLatentSamples::Tensor(audio_samples),
                    Some(NativeLatentNoiseMask::Tensor(replacement_mask.clone())),
                    Some(vec![]),
                    context,
                ),
                Err(NativeLatentBundleError::NoiseMaskGeometryMismatch)
            ));
            context.cancellation.cancel();
            assert!(matches!(
                bundle.replaced_samples_mask_and_indices_source_dictionary(
                    NativeLatentSamples::Tensor(replacement),
                    Some(NativeLatentNoiseMask::Tensor(replacement_mask)),
                    Some(vec![]),
                    context,
                ),
                Err(NativeLatentBundleError::Tensor(TensorError::Cancelled))
            ));
            assert_eq!(bundle.projection(), &original_projection);
            Ok(())
        })
    }

    #[test]
    fn latent_bundle_rejects_invalid_geometry_metadata_and_cancellation()
    -> Result<(), Box<dyn Error>> {
        with_context(|backend, context| {
            let samples = upload_tensor(backend, context, vec![2, 1, 2, 2], &[0.0; 8])?;
            let audio = upload_tensor(backend, context, vec![2, 1, 2], &[0.0; 4])?;
            let bad_mask = upload_tensor(backend, context, vec![2, 1, 3, 2], &[1.0; 12])?;
            assert!(matches!(
                NativeLatentBundle::single(
                    audio,
                    Some(bad_mask),
                    None,
                    NativeLatentMetadata::default(),
                    context,
                ),
                Err(NativeLatentBundleError::NoiseMaskGeometryMismatch)
            ));
            assert!(matches!(
                NativeLatentBundle::single(
                    samples.clone(),
                    None,
                    Some(vec![0]),
                    NativeLatentMetadata::default(),
                    context,
                ),
                Err(NativeLatentBundleError::BatchIndexLengthMismatch)
            ));
            assert!(matches!(
                NativeLatentMetadata::checked(None, Some(0), None, None),
                Err(NativeLatentBundleError::InvalidMetadata("sample_rate"))
            ));

            context.cancellation.cancel();
            assert!(matches!(
                NativeLatentBundle::single(
                    samples,
                    None,
                    None,
                    NativeLatentMetadata::default(),
                    context,
                ),
                Err(NativeLatentBundleError::Tensor(TensorError::Cancelled))
            ));
            Ok(())
        })
    }

    #[test]
    fn latent_tensor_digest_checks_cancellation_between_bounded_chunks()
    -> Result<(), Box<dyn Error>> {
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 1, 1, 32_769],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let byte_length = usize::try_from(descriptor.byte_len()?)?;
        let bytes = vec![0_u8; byte_length];
        let mut checks = 0_usize;
        let error = latent_tensor_digest_with_check(&descriptor, &bytes, || {
            checks = checks.saturating_add(1);
            if checks == 2 {
                Err(sim_types::CancellationError)
            } else {
                Ok(())
            }
        })
        .expect_err("the second digest chunk must observe cancellation");
        assert!(matches!(
            error,
            NativeLatentBundleError::Tensor(TensorError::Cancelled)
        ));
        assert_eq!(checks, 2);
        Ok(())
    }

    #[test]
    fn sampler_sample_replacement_drops_only_downscale_metadata() -> Result<(), Box<dyn Error>> {
        with_context(|backend, context| {
            let samples = upload_tensor(backend, context, vec![1, 1, 1, 2], &[1.0, 2.0])?;
            let mask = upload_tensor(backend, context, vec![1, 1, 1, 2], &[0.0, 1.0])?;
            let bundle = NativeLatentBundle::single(
                samples,
                Some(mask),
                Some(vec![9]),
                NativeLatentMetadata::checked(
                    Some(NativeLatentType::Audio),
                    Some(44_100),
                    Some(8),
                    Some(4),
                )?,
                context,
            )?;
            let replacement = upload_tensor(backend, context, vec![1, 1, 1, 2], &[3.0, 4.0])?;
            let replaced = bundle.replaced_samples(
                NativeLatentSamples::Tensor(replacement),
                NativeLatentMetadataRetention::DropDownscaleRatios,
                context,
            )?;
            assert_eq!(
                replaced.metadata().latent_type(),
                Some(NativeLatentType::Audio)
            );
            assert_eq!(replaced.metadata().sample_rate(), Some(44_100));
            assert_eq!(replaced.metadata().spatial_downscale_ratio(), None);
            assert_eq!(replaced.metadata().temporal_downscale_ratio(), None);
            assert_eq!(replaced.batch_indices(), Some([9].as_slice()));
            assert!(replaced.noise_mask().is_some());
            Ok(())
        })
    }

    #[test]
    fn roles_preserve_source_socket_type_and_digest_identity() -> Result<(), Box<dyn Error>> {
        let expected = [
            (NativeTensorRole::Image, "IMAGE"),
            (NativeTensorRole::Mask, "MASK"),
            (NativeTensorRole::Conditioning, "CONDITIONING"),
            (NativeTensorRole::Latent, "LATENT"),
        ];
        for (role, type_id) in expected {
            assert_eq!(role.handle_type_id(), type_id);
        }

        let conditioning = tensor_payload(NativeTensorRole::Conditioning, &[1.0, -2.0])?;
        let latent = tensor_payload(NativeTensorRole::Latent, &[1.0, -2.0])?;
        assert_ne!(
            conditioning.projection().content_digest(),
            latent.projection().content_digest()
        );
        assert_eq!(
            serde_json::to_string(&NativeTensorRole::Conditioning)?,
            "\"conditioning\""
        );
        Ok(())
    }

    #[test]
    fn semantic_digest_ignores_stream_placement() -> Result<(), Box<dyn Error>> {
        let default_stream =
            TensorDescriptor::contiguous(vec![1, 2], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        let other_stream =
            TensorDescriptor::contiguous(vec![1, 2], DType::F32, DeviceId::CPU, StreamId::new(7))?;
        let bytes = [1.0_f32, -2.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        assert_eq!(
            native_tensor_digest(NativeTensorRole::Latent, &default_stream, &bytes)?,
            native_tensor_digest(NativeTensorRole::Latent, &other_stream, &bytes)?
        );
        assert_ne!(
            native_tensor_digest(NativeTensorRole::Latent, &default_stream, &bytes)?,
            native_tensor_digest(NativeTensorRole::Sigmas, &default_stream, &bytes)?
        );
        Ok(())
    }

    #[test]
    fn tensor_payload_projects_real_storage_and_revalidates() -> Result<(), Box<dyn Error>> {
        let payload = tensor_payload(NativeTensorRole::Latent, &[0.25, 0.5, 0.75])?;
        assert_eq!(payload.projection().schema_version(), 1);
        assert_eq!(payload.projection().role(), NativeTensorRole::Latent);
        assert_eq!(payload.handle_type_id(), "LATENT");
        assert_eq!(
            payload.projection().descriptor(),
            payload.tensor().descriptor()
        );
        assert_eq!(
            payload.projection().resident_bytes(),
            payload.tensor().storage_byte_len()
        );
        assert_eq!(
            payload.projection().content_digest(),
            native_tensor_digest(
                NativeTensorRole::Latent,
                payload.tensor().descriptor(),
                payload.tensor().contiguous_bytes()?,
            )?
        );
        payload.validate()?;
        assert!(payload.image().is_none());
        Ok(())
    }

    #[test]
    fn image_and_mask_payloads_preserve_source_exact_bhwc_and_bhw_shapes()
    -> Result<(), Box<dyn Error>> {
        with_context(|backend, context| {
            let image = ImageTensor::from_f32(
                backend,
                context,
                1,
                1,
                2,
                3,
                &[0.0, 0.25, 0.5, 0.75, 1.0, 0.125],
            )?;
            let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)?;
            assert_eq!(
                payload
                    .image()
                    .ok_or("image storage is missing")?
                    .dimensions()?,
                (1, 1, 2, 3)
            );
            payload.validate()?;

            let mask = ImageTensor::from_f32(backend, context, 1, 1, 2, 1, &[0.0, 1.0])?;
            let payload = NativeTensorPayload::from_image(NativeTensorRole::Mask, mask)?;
            assert_eq!(payload.handle_type_id(), "MASK");
            assert_eq!(payload.tensor().descriptor().shape(), &[1, 1, 2]);
            assert!(payload.image().is_none());
            payload.validate()?;
            Ok(())
        })
    }

    #[test]
    fn role_and_storage_mismatches_fail_closed() -> Result<(), Box<dyn Error>> {
        let latent_as_image = with_context(|backend, context| {
            let image = ImageTensor::from_f32(backend, context, 1, 1, 1, 1, &[0.0])?;
            Ok(NativeTensorPayload::from_image(
                NativeTensorRole::Latent,
                image,
            ))
        })?;
        assert!(matches!(
            latent_as_image,
            Err(NativeTensorPayloadError::RoleStorageMismatch {
                role: NativeTensorRole::Latent,
                storage: "ImageTensor",
            })
        ));

        let image_as_tensor = tensor_payload(NativeTensorRole::Image, &[0.0]);
        assert!(matches!(
            image_as_tensor,
            Err(error)
                if error.downcast_ref::<NativeTensorPayloadError>().is_some_and(|error| matches!(
                    error,
                    NativeTensorPayloadError::RoleStorageMismatch {
                        role: NativeTensorRole::Image,
                        storage: "Tensor",
                    }
                ))
        ));
        Ok(())
    }

    #[test]
    fn digest_rejects_wrong_byte_length() -> Result<(), Box<dyn Error>> {
        let descriptor =
            TensorDescriptor::contiguous(vec![2], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        assert_eq!(
            native_tensor_digest(NativeTensorRole::Latent, &descriptor, &[0; 8])?,
            "21469c6bc170dac513a10652e55babec9b58f4f834ce468ecafafe791efccdea"
        );
        assert!(matches!(
            native_tensor_digest(NativeTensorRole::Latent, &descriptor, &[0; 4]),
            Err(NativeTensorPayloadError::ContentByteLength {
                expected: 8,
                actual: 4,
            })
        ));
        Ok(())
    }

    #[test]
    fn projection_deserialization_is_bounded_and_validated() -> Result<(), Box<dyn Error>> {
        let payload = tensor_payload(NativeTensorRole::Conditioning, &[1.0])?;
        let encoded = serde_json::to_value(payload.projection())?;
        let decoded: NativeTensorProjection = serde_json::from_value(encoded.clone())?;
        assert_eq!(&decoded, payload.projection());

        let mut invalid_version = encoded.clone();
        invalid_version["schema_version"] = json!(2);
        assert!(serde_json::from_value::<NativeTensorProjection>(invalid_version).is_err());

        let mut invalid_digest = encoded.clone();
        invalid_digest["content_digest"] = json!("ABCDEF");
        assert!(serde_json::from_value::<NativeTensorProjection>(invalid_digest).is_err());

        let mut invalid_resident_bytes = encoded;
        invalid_resident_bytes["resident_bytes"] = json!(0);
        assert!(serde_json::from_value::<NativeTensorProjection>(invalid_resident_bytes).is_err());
        Ok(())
    }

    #[test]
    fn zero_byte_tensor_has_a_valid_zero_resident_projection() -> Result<(), Box<dyn Error>> {
        let payload = tensor_payload(NativeTensorRole::Latent, &[])?;
        assert_eq!(payload.projection().resident_bytes(), 0);
        assert!(payload.tensor().contiguous_bytes()?.is_empty());
        payload.validate()?;
        Ok(())
    }

    #[test]
    fn validation_detects_projection_drift() -> Result<(), Box<dyn Error>> {
        let mut payload = tensor_payload(NativeTensorRole::Latent, &[1.0])?;
        let replacement = if payload.projection.content_digest.starts_with('0') {
            "1"
        } else {
            "0"
        };
        payload
            .projection
            .content_digest
            .replace_range(0..1, replacement);
        assert!(matches!(
            payload.validate(),
            Err(NativeTensorPayloadError::ProjectionChanged)
        ));
        Ok(())
    }
}
