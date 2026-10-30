pub use crate::vae_tiling::VaeTileAxisFormula;
use crate::{
    ArtifactAvailability, ArtifactKey, ArtifactRecord, AttentionError, LatentFormatDefinition,
    LatentFormatDescriptor, LatentFormatError, LatentFormatIdentity, LatentFormatRegistry,
    LatentTensorLayout, LatentTransform, LoadedModel, ModelFamilyIdentity, ModelFamilyRegistry,
    ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe, ModelStore,
    ModelStoreError, NativeModule, NativeOpsError, PatchGraph, PatchGraphError, PatchGraphIdentity,
    PatchGraphIdentityError,
    native_ops::{
        NativeExecutionRequirements, NativeModelExecutionAdmission, NativeModelExecutionRole,
        admit_native_model_resource_execution,
    },
    process_latent_in, process_latent_out,
    vae_architecture::{
        ExplicitAutoencoderKlTopology, VaeArchitectureError, VaeArchitectureRegistry,
        VaeArchitectureSelection, VaeBoundaryDomain, VaeExecutionTarget, VaeLoaderConfiguration,
        checked_recipe_configuration_is_explicit, is_registered_vae_architecture,
        validate_architecture_profile_pair, validate_vae_identity_target,
    },
    vae_tiling::{TileExecutionPlan, TileTensorLayout, execute_tiled_scale},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_tensor::{
    BinaryOperation, CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId,
    ExecutionContext, Layout, OperationSupport, ReductionOperation, ResizeMode, Scalar, ScalarSide,
    StreamId, Tensor, TensorBackend, TensorError, ViewAccess,
    generated_native_diffusion::NativeDiffusionTensorError,
};
#[cfg(test)]
use sim_tensor::{
    TensorDescriptor,
    generated_native_diffusion::{
        conv2d, group_norm, nearest_upsample_2x, silu, tensor_from_f32, tensor_to_f32,
    },
};
#[cfg(test)]
use sim_types::DeviceKind;
#[cfg(any(test, feature = "test-support"))]
use std::collections::BTreeSet;
use std::{any::Any, collections::BTreeMap, sync::Arc};
use thiserror::Error;

pub const VAE_SCHEMA_VERSION: u16 = 3;
const MAX_IDENTITY_FIELD_BYTES: usize = 1_024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VaeKernelProfile {
    BlockAverageNearestV1,
    Sd15AutoencoderKlReducedV1,
    TemporalAutoencodingEngineV1,
    TaesdV1,
    StableCascadeStageAV1,
    StableCascadeStageCEncoderV1,
    StableCascadeStageCPreviewerV1,
    StableCascadeStageCCombinedV1,
    HunyuanImageV1,
    HunyuanImageRefinerV1,
    AutoencoderKlV1,
    AutoencoderKlX4V1,
    AutoencoderKlBatchNormV1,
    ExplicitAutoencoderKlV1,
    AutoencodingEngineV1,
    AutoencodingEngineX4V1,
    AutoencodingEngineBatchNormV1,
    AudioOobleck44KhzV1,
    AudioOobleck48KhzV1,
    MochiV1,
    LtxVideoV0 {
        configuration_sha256: Option<String>,
    },
    LtxVideoV1 {
        configuration_sha256: Option<String>,
    },
    LtxVideoV2 {
        configuration_sha256: Option<String>,
    },
    HunyuanVideoRefinerV1,
    CogVideoXV1,
    Causal3dV1,
    CosmosV1,
    Wan21V1,
    Wan22V1,
    HunyuanShapeV1,
    MusicDcaeV1,
    PixelSpaceV1,
    MmAudio16KhzV1,
    TaeHvWan22V1,
    TaeHvLtx2V1,
    LightTaeHv15V1,
    TaeHvHunyuanV1,
    LightTaeWan21V1,
    LtxAudioV1,
    StableAudio3DeepV1,
    StableAudio3ShallowV1,
    TripoSplatV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VaeCanonicalCompatibility {
    Exact(&'static [&'static str]),
    Unavailable(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct VaeProfileContract {
    pub canonical_compatibility: VaeCanonicalCompatibility,
    pub supported_dtypes: &'static [DType],
    pub boundary: VaeBoundaryKind,
    pub latent_dimensions: u8,
    pub target_latent_channels: Option<u64>,
}

const PROFILE_DEFAULT_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
const PROFILE_WIDE_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const PROFILE_F16_F32_DTYPES: &[DType] = &[DType::F16, DType::F32];
const PROFILE_F32_ONLY: &[DType] = &[DType::F32];

impl VaeKernelProfile {
    fn owned_resident_bytes(&self) -> Option<u64> {
        match self {
            Self::LtxVideoV0 {
                configuration_sha256,
            }
            | Self::LtxVideoV1 {
                configuration_sha256,
            }
            | Self::LtxVideoV2 {
                configuration_sha256,
            } => u64::try_from(configuration_sha256.as_ref().map_or(0, String::capacity)).ok(),
            _ => Some(0),
        }
    }

    pub const fn is_conformance_only(&self) -> bool {
        match self {
            Self::BlockAverageNearestV1 => true,
            _ => false,
        }
    }

    pub(crate) const fn uses_explicit_source_admission(&self) -> bool {
        matches!(self, Self::Sd15AutoencoderKlReducedV1)
    }

    fn identity_bytes(&self) -> Result<Vec<u8>, VaeError> {
        match self {
            Self::BlockAverageNearestV1 => Ok(b"block-average-nearest-v1".to_vec()),
            Self::Sd15AutoencoderKlReducedV1 => Ok(b"sd15-autoencoder-kl-reduced-v1".to_vec()),
            _ => serde_json::to_vec(self)
                .map_err(|error| VaeError::IdentitySerialization(error.to_string())),
        }
    }

    pub(crate) fn contract(&self) -> VaeProfileContract {
        use VaeBoundaryKind::{Audio, Image, StructuredOutput, Video};
        use VaeCanonicalCompatibility::{Exact, Unavailable};
        let unavailable = Unavailable(
            "the generated canonical registry has no semantically equivalent latent identity",
        );
        let (canonical_compatibility, supported_dtypes, boundary, latent_dimensions, channels) =
            match self {
                Self::TemporalAutoencodingEngineV1 => {
                    (Exact(&["SD15"]), PROFILE_F32_ONLY, Image, 2, Some(4))
                }
                Self::TaesdV1 => (
                    Exact(&["SD15", "SDXL", "SD_X4", "Flux2"]),
                    PROFILE_F32_ONLY,
                    Image,
                    2,
                    None,
                ),
                Self::AutoencoderKlV1 | Self::AutoencodingEngineV1 => (
                    Exact(&["SD15", "SDXL", "SD_X4"]),
                    PROFILE_F32_ONLY,
                    Image,
                    2,
                    None,
                ),
                Self::AutoencoderKlX4V1 | Self::AutoencodingEngineX4V1 => {
                    (Exact(&["SD_X4"]), PROFILE_F32_ONLY, Image, 2, None)
                }
                Self::AutoencoderKlBatchNormV1 | Self::AutoencodingEngineBatchNormV1 => {
                    (Exact(&["Flux", "SD3"]), PROFILE_F32_ONLY, Image, 2, None)
                }
                Self::ExplicitAutoencoderKlV1 => (
                    Exact(&[
                        "ChromaRadiance",
                        "Flux2",
                        "Flux",
                        "HiDreamO1Pixel",
                        "HunyuanImage21",
                        "PixelDiTPixel",
                        "SC_B",
                        "SC_Prior",
                        "SD15",
                        "SD3",
                        "SD_X4",
                        "SDXL",
                        "SDXL_Playground_2_5",
                        "ZImagePixelSpace",
                    ]),
                    PROFILE_F32_ONLY,
                    Image,
                    2,
                    None,
                ),
                Self::StableCascadeStageAV1 => {
                    (Exact(&["SC_B"]), PROFILE_F32_ONLY, Image, 2, Some(4))
                }
                Self::StableCascadeStageCEncoderV1
                | Self::StableCascadeStageCPreviewerV1
                | Self::StableCascadeStageCCombinedV1 => {
                    (Exact(&["SC_Prior"]), PROFILE_F32_ONLY, Image, 2, Some(16))
                }
                Self::HunyuanImageV1 => (
                    Exact(&["HunyuanImage21"]),
                    PROFILE_F32_ONLY,
                    Image,
                    2,
                    Some(64),
                ),
                Self::HunyuanImageRefinerV1 => (
                    Exact(&["HunyuanImage21Refiner"]),
                    PROFILE_WIDE_DTYPES,
                    Image,
                    3,
                    Some(64),
                ),
                Self::AudioOobleck44KhzV1 | Self::AudioOobleck48KhzV1 => (
                    Exact(&["StableAudio1"]),
                    PROFILE_WIDE_DTYPES,
                    Audio,
                    1,
                    Some(64),
                ),
                Self::MochiV1 => (
                    Exact(&["Mochi"]),
                    PROFILE_F16_F32_DTYPES,
                    Video,
                    3,
                    Some(12),
                ),
                Self::LtxVideoV0 { .. } | Self::LtxVideoV1 { .. } | Self::LtxVideoV2 { .. } => (
                    Exact(&["LTXV"]),
                    PROFILE_DEFAULT_DTYPES,
                    Video,
                    3,
                    Some(128),
                ),
                Self::HunyuanVideoRefinerV1 => (
                    Exact(&["HunyuanVideo15"]),
                    PROFILE_WIDE_DTYPES,
                    Video,
                    3,
                    Some(32),
                ),
                Self::LightTaeHv15V1 => (
                    Exact(&["HunyuanVideo15"]),
                    PROFILE_DEFAULT_DTYPES,
                    Video,
                    3,
                    Some(32),
                ),
                Self::CogVideoXV1 => (
                    Exact(&["CogVideoX", "CogVideoX1_5"]),
                    PROFILE_WIDE_DTYPES,
                    Video,
                    3,
                    Some(16),
                ),
                Self::Causal3dV1 => (unavailable, PROFILE_WIDE_DTYPES, Video, 3, None),
                Self::CosmosV1 => (
                    Exact(&["Cosmos1CV8x8x8"]),
                    PROFILE_DEFAULT_DTYPES,
                    Video,
                    3,
                    Some(16),
                ),
                Self::Wan21V1 => (Exact(&["Wan21"]), PROFILE_WIDE_DTYPES, Video, 3, Some(16)),
                Self::LightTaeWan21V1 => (
                    Exact(&["Wan21"]),
                    PROFILE_DEFAULT_DTYPES,
                    Video,
                    3,
                    Some(16),
                ),
                Self::Wan22V1 => (Exact(&["Wan22"]), PROFILE_WIDE_DTYPES, Video, 3, Some(48)),
                Self::TaeHvWan22V1 => (
                    Exact(&["Wan22"]),
                    PROFILE_DEFAULT_DTYPES,
                    Video,
                    3,
                    Some(48),
                ),
                Self::HunyuanShapeV1 => (
                    Exact(&["Hunyuan3Dv2", "Hunyuan3Dv2_1", "Hunyuan3Dv2mini"]),
                    PROFILE_WIDE_DTYPES,
                    StructuredOutput,
                    1,
                    None,
                ),
                Self::MusicDcaeV1 => (Exact(&["ACEAudio"]), PROFILE_WIDE_DTYPES, Audio, 2, Some(8)),
                Self::PixelSpaceV1 => (
                    Exact(&["PixelDiTPixel", "HiDreamO1Pixel", "ZImagePixelSpace"]),
                    PROFILE_F32_ONLY,
                    Image,
                    2,
                    Some(3),
                ),
                Self::MmAudio16KhzV1 => (unavailable, PROFILE_F32_ONLY, Audio, 1, Some(20)),
                Self::TaeHvLtx2V1 => (
                    Exact(&["LTXV"]),
                    PROFILE_DEFAULT_DTYPES,
                    Video,
                    3,
                    Some(128),
                ),
                Self::TaeHvHunyuanV1 => (
                    Exact(&["HunyuanVideo"]),
                    PROFILE_DEFAULT_DTYPES,
                    Video,
                    3,
                    None,
                ),
                Self::LtxAudioV1 => (Exact(&["ACEAudio"]), PROFILE_F32_ONLY, Audio, 2, Some(8)),
                Self::StableAudio3DeepV1 | Self::StableAudio3ShallowV1 => (
                    Exact(&["StableAudio3"]),
                    PROFILE_WIDE_DTYPES,
                    Audio,
                    1,
                    Some(256),
                ),
                Self::TripoSplatV1 => (
                    Exact(&["TripoSplat"]),
                    PROFILE_WIDE_DTYPES,
                    StructuredOutput,
                    1,
                    Some(16),
                ),
                Self::BlockAverageNearestV1 => (
                    Unavailable("conformance profiles use test-owned identities"),
                    PROFILE_F32_ONLY,
                    Image,
                    2,
                    None,
                ),
                Self::Sd15AutoencoderKlReducedV1 => {
                    (Exact(&["SD15"]), PROFILE_F32_ONLY, Image, 2, Some(4))
                }
            };
        VaeProfileContract {
            canonical_compatibility,
            supported_dtypes,
            boundary,
            latent_dimensions,
            target_latent_channels: channels,
        }
    }

    pub fn canonical_compatibility(&self) -> VaeCanonicalCompatibility {
        self.contract().canonical_compatibility
    }

    pub fn supported_dtypes(&self) -> &'static [DType] {
        self.contract().supported_dtypes
    }

    pub fn expected_boundary_kind(&self) -> VaeBoundaryKind {
        self.contract().boundary
    }

    pub fn latent_dimensions(&self) -> u8 {
        self.contract().latent_dimensions
    }

    pub fn target_latent_channels(&self) -> Option<u64> {
        self.contract().target_latent_channels
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VaeArchitectureIdentity(String);

impl VaeArchitectureIdentity {
    pub(crate) fn checked(value: impl Into<String>) -> Result<Self, VaeError> {
        let value = value.into();
        validate_identifier("architecture", &value)?;
        if !is_registered_vae_architecture(&value) {
            return Err(VaeError::UnregisteredArchitecture(value));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn owned_resident_bytes(&self) -> Option<u64> {
        u64::try_from(self.0.capacity()).ok()
    }
}

impl Serialize for VaeArchitectureIdentity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for VaeArchitectureIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::checked(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VaeBoundaryKind {
    Image,
    Video,
    Audio,
    StructuredOutput,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VaeStructuredOutputKind {
    Shape,
    GaussianSplats,
}

#[derive(Clone, Debug, PartialEq)]
pub enum VaeStructuredDecodeRequest {
    Shape {
        bounds: [f32; 6],
        octree_resolution: u16,
        chunk_size: u32,
    },
    GaussianSplats {
        num_gaussians: u32,
        octree_level: u8,
    },
}

impl VaeStructuredDecodeRequest {
    pub fn shape(
        bounds: [f32; 6],
        octree_resolution: u16,
        chunk_size: u32,
    ) -> Result<Self, VaeError> {
        if bounds.iter().any(|value| !value.is_finite())
            || bounds[0] >= bounds[3]
            || bounds[1] >= bounds[4]
            || bounds[2] >= bounds[5]
            || octree_resolution == 0
            || chunk_size == 0
        {
            return Err(VaeError::InvalidStructuredRequest(
                "shape bounds, resolution, and chunk size must be finite and positive".to_owned(),
            ));
        }
        Ok(Self::Shape {
            bounds,
            octree_resolution,
            chunk_size,
        })
    }

    pub fn gaussian_splats(num_gaussians: u32, octree_level: u8) -> Result<Self, VaeError> {
        if num_gaussians == 0 || !(1..=8).contains(&octree_level) {
            return Err(VaeError::InvalidStructuredRequest(
                "Gaussian count must be positive and octree level must be in 1..=8".to_owned(),
            ));
        }
        Ok(Self::GaussianSplats {
            num_gaussians,
            octree_level,
        })
    }

    pub const fn output_kind(&self) -> VaeStructuredOutputKind {
        match self {
            Self::Shape { .. } => VaeStructuredOutputKind::Shape,
            Self::GaussianSplats { .. } => VaeStructuredOutputKind::GaussianSplats,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VaeShapeField {
    logits: Tensor,
    bounds: [f32; 6],
    resolution: u16,
}

impl VaeShapeField {
    pub(crate) fn checked(
        logits: Tensor,
        bounds: [f32; 6],
        resolution: u16,
    ) -> Result<Self, VaeError> {
        let extent = u64::from(resolution)
            .checked_add(1)
            .ok_or(VaeError::ShapeOverflow)?;
        let shape = logits.descriptor().shape();
        if shape.len() != 4
            || shape[1..] != [extent, extent, extent]
            || bounds.iter().any(|value| !value.is_finite())
            || bounds[0] >= bounds[3]
            || bounds[1] >= bounds[4]
            || bounds[2] >= bounds[5]
        {
            return Err(VaeError::InvalidStructuredResult(
                "shape field must be Bx(R+1)x(R+1)x(R+1) with finite bounds".to_owned(),
            ));
        }
        Ok(Self {
            logits,
            bounds,
            resolution,
        })
    }

    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    pub const fn bounds(&self) -> [f32; 6] {
        self.bounds
    }

    pub const fn resolution(&self) -> u16 {
        self.resolution
    }
}

#[derive(Clone, Debug)]
pub struct VaeGaussianSplatBatch {
    positions: Tensor,
    features_dc: Tensor,
    scales: Tensor,
    rotations: Tensor,
    opacities: Tensor,
}

impl VaeGaussianSplatBatch {
    pub(crate) fn checked(
        positions: Tensor,
        features_dc: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacities: Tensor,
    ) -> Result<Self, VaeError> {
        let count = positions.descriptor().shape().first().copied().unwrap_or(0);
        let compatible = count > 0
            && positions.descriptor().shape() == [count, 3]
            && features_dc.descriptor().shape() == [count, 1, 3]
            && scales.descriptor().shape() == [count, 3]
            && rotations.descriptor().shape() == [count, 4]
            && opacities.descriptor().shape() == [count, 1]
            && [&features_dc, &scales, &rotations, &opacities]
                .into_iter()
                .all(|tensor| {
                    tensor.descriptor().dtype() == positions.descriptor().dtype()
                        && tensor.descriptor().device() == positions.descriptor().device()
                        && tensor.descriptor().stream() == positions.descriptor().stream()
                });
        if !compatible {
            return Err(VaeError::InvalidStructuredResult(
                "Gaussian splat tensors have incompatible field shapes or execution bindings"
                    .to_owned(),
            ));
        }
        Ok(Self {
            positions,
            features_dc,
            scales,
            rotations,
            opacities,
        })
    }

    pub fn positions(&self) -> &Tensor {
        &self.positions
    }

    pub fn features_dc(&self) -> &Tensor {
        &self.features_dc
    }

    pub fn scales(&self) -> &Tensor {
        &self.scales
    }

    pub fn rotations(&self) -> &Tensor {
        &self.rotations
    }

    pub fn opacities(&self) -> &Tensor {
        &self.opacities
    }
}

#[derive(Clone, Debug)]
pub enum VaeStructuredResult {
    Shape(VaeShapeField),
    GaussianSplats(Vec<VaeGaussianSplatBatch>),
}

impl VaeStructuredResult {
    pub const fn kind(&self) -> VaeStructuredOutputKind {
        match self {
            Self::Shape(_) => VaeStructuredOutputKind::Shape,
            Self::GaussianSplats(_) => VaeStructuredOutputKind::GaussianSplats,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaeBoundary {
    kind: VaeBoundaryKind,
    channels: u64,
    sample_rate: Option<u32>,
    structured_output: Option<VaeStructuredOutputKind>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VaeBoundaryWire {
    kind: VaeBoundaryKind,
    channels: u64,
    sample_rate: Option<u32>,
    structured_output: Option<VaeStructuredOutputKind>,
}

impl VaeBoundary {
    pub fn image(channels: u64) -> Result<Self, VaeError> {
        Self::checked(VaeBoundaryKind::Image, channels, None, None)
    }

    pub fn video(channels: u64) -> Result<Self, VaeError> {
        Self::checked(VaeBoundaryKind::Video, channels, None, None)
    }

    pub fn audio(channels: u64, sample_rate: u32) -> Result<Self, VaeError> {
        Self::checked(VaeBoundaryKind::Audio, channels, Some(sample_rate), None)
    }

    pub fn structured_output(
        channels: u64,
        output: VaeStructuredOutputKind,
    ) -> Result<Self, VaeError> {
        Self::checked(
            VaeBoundaryKind::StructuredOutput,
            channels,
            None,
            Some(output),
        )
    }

    fn checked(
        kind: VaeBoundaryKind,
        channels: u64,
        sample_rate: Option<u32>,
        structured_output: Option<VaeStructuredOutputKind>,
    ) -> Result<Self, VaeError> {
        if channels == 0 {
            return Err(VaeError::InvalidPixelChannels(channels));
        }
        let fields_match = match kind {
            VaeBoundaryKind::Image | VaeBoundaryKind::Video => {
                sample_rate.is_none() && structured_output.is_none()
            }
            VaeBoundaryKind::Audio => {
                sample_rate.is_some_and(|sample_rate| sample_rate > 0)
                    && structured_output.is_none()
            }
            VaeBoundaryKind::StructuredOutput => {
                sample_rate.is_none() && structured_output.is_some()
            }
        };
        if !fields_match {
            return Err(VaeError::InvalidBoundary(kind));
        }
        Ok(Self {
            kind,
            channels,
            sample_rate,
            structured_output,
        })
    }

    fn validate_latent_dimensions(&self, dimensions: u8) -> Result<(), VaeError> {
        let matches = match self.kind {
            VaeBoundaryKind::Image => dimensions == 2,
            VaeBoundaryKind::Video => dimensions == 3,
            VaeBoundaryKind::Audio => matches!(dimensions, 1 | 2),
            VaeBoundaryKind::StructuredOutput => matches!(dimensions, 1..=3),
        };
        if matches {
            Ok(())
        } else {
            Err(VaeError::BoundaryLatentDimensionMismatch {
                kind: self.kind,
                dimensions,
            })
        }
    }

    pub const fn kind(&self) -> VaeBoundaryKind {
        self.kind
    }

    pub const fn channels(&self) -> u64 {
        self.channels
    }

    pub const fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    pub const fn structured_kind(&self) -> Option<VaeStructuredOutputKind> {
        self.structured_output
    }
}

impl Serialize for VaeBoundary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        VaeBoundaryWire {
            kind: self.kind,
            channels: self.channels,
            sample_rate: self.sample_rate,
            structured_output: self.structured_output,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for VaeBoundary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = VaeBoundaryWire::deserialize(deserializer)?;
        Self::checked(
            wire.kind,
            wire.channels,
            wire.sample_rate,
            wire.structured_output,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaeIdentity {
    artifact_root_id: String,
    artifact_relative_path: String,
    artifact_sha256: String,
    family: ModelFamilyIdentity,
    latent_format: LatentFormatIdentity,
    architecture: VaeArchitectureIdentity,
    patch: PatchGraphIdentity,
    dtype: DType,
    device: DeviceId,
    boundary: VaeBoundary,
    profile: VaeKernelProfile,
    loader_configuration: VaeLoaderConfiguration,
    digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VaeIdentityWire {
    schema_version: u16,
    artifact_root_id: String,
    artifact_relative_path: String,
    artifact_sha256: String,
    family: ModelFamilyIdentity,
    latent_format: LatentFormatIdentity,
    architecture: VaeArchitectureIdentity,
    patch: PatchGraphIdentity,
    dtype: DType,
    device: DeviceId,
    boundary: VaeBoundary,
    profile: VaeKernelProfile,
    loader_configuration: VaeLoaderConfiguration,
    digest: String,
}

impl VaeIdentity {
    fn checked(
        artifact: &ArtifactRecord,
        family: ModelFamilyIdentity,
        latent_format: LatentFormatIdentity,
        architecture: VaeArchitectureIdentity,
        patch: PatchGraphIdentity,
        dtype: DType,
        device: DeviceId,
        boundary: VaeBoundary,
        profile: VaeKernelProfile,
    ) -> Result<Self, VaeError> {
        if artifact.availability != ArtifactAvailability::Present {
            return Err(VaeError::ArtifactUnavailable);
        }
        validate_identity_field("artifact root id", &artifact.key.root_id)?;
        let artifact_relative_path = artifact
            .key
            .relative_path
            .to_str()
            .ok_or(VaeError::NonUtf8ArtifactPath)?
            .to_owned();
        validate_identity_field("artifact relative path", &artifact_relative_path)?;
        validate_sha256(&artifact.sha256)?;
        validate_patch_identity(artifact, &patch)?;
        validate_execution_dtype(dtype)?;
        validate_architecture_profile_pair(&architecture, &profile)?;
        validate_vae_identity_target(
            &architecture,
            &profile,
            &family,
            &latent_format,
            dtype,
            device,
            boundary.kind(),
        )?;
        let mut identity = Self {
            artifact_root_id: artifact.key.root_id.clone(),
            artifact_relative_path,
            artifact_sha256: artifact.sha256.clone(),
            family,
            latent_format,
            architecture,
            patch,
            dtype,
            device,
            boundary,
            profile,
            loader_configuration: VaeLoaderConfiguration::Automatic,
            digest: String::new(),
        };
        identity.digest = identity.compute_digest()?;
        Ok(identity)
    }

    #[allow(clippy::too_many_arguments)]
    fn checked_recipe(
        artifact_sha256: &str,
        family: ModelFamilyIdentity,
        latent_format: LatentFormatIdentity,
        architecture: VaeArchitectureIdentity,
        patch: PatchGraphIdentity,
        dtype: DType,
        device: DeviceId,
        boundary: VaeBoundary,
        profile: VaeKernelProfile,
    ) -> Result<Self, VaeError> {
        validate_sha256(artifact_sha256)?;
        patch
            .validate_for_base(artifact_sha256)
            .map_err(|_| VaeError::PatchArtifactMismatch)?;
        validate_execution_dtype(dtype)?;
        validate_architecture_profile_pair(&architecture, &profile)?;
        validate_vae_identity_target(
            &architecture,
            &profile,
            &family,
            &latent_format,
            dtype,
            device,
            boundary.kind(),
        )?;
        let mut identity = Self {
            artifact_root_id: "native-model-source".to_owned(),
            artifact_relative_path: artifact_sha256.to_owned(),
            artifact_sha256: artifact_sha256.to_owned(),
            family,
            latent_format,
            architecture,
            patch,
            dtype,
            device,
            boundary,
            profile,
            loader_configuration: VaeLoaderConfiguration::Automatic,
            digest: String::new(),
        };
        identity.digest = identity.compute_digest()?;
        Ok(identity)
    }

    pub fn artifact_root_id(&self) -> &str {
        &self.artifact_root_id
    }

    pub fn artifact_relative_path(&self) -> &str {
        &self.artifact_relative_path
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn family(&self) -> &ModelFamilyIdentity {
        &self.family
    }

    pub fn latent_format(&self) -> &LatentFormatIdentity {
        &self.latent_format
    }

    pub fn architecture(&self) -> &VaeArchitectureIdentity {
        &self.architecture
    }

    pub fn patch(&self) -> &PatchGraphIdentity {
        &self.patch
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn device(&self) -> DeviceId {
        self.device
    }

    pub fn boundary(&self) -> &VaeBoundary {
        &self.boundary
    }

    pub fn profile(&self) -> &VaeKernelProfile {
        &self.profile
    }

    pub fn loader_configuration(&self) -> &VaeLoaderConfiguration {
        &self.loader_configuration
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    fn owned_resident_bytes(&self) -> Option<u64> {
        let strings = [
            &self.artifact_root_id,
            &self.artifact_relative_path,
            &self.artifact_sha256,
            &self.digest,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            total.checked_add(u64::try_from(value.capacity()).ok()?)
        })?;
        strings
            .checked_add(self.family.owned_resident_bytes()?)?
            .checked_add(self.latent_format.owned_resident_bytes()?)?
            .checked_add(self.architecture.owned_resident_bytes()?)?
            .checked_add(self.patch.owned_resident_bytes()?)?
            .checked_add(self.profile.owned_resident_bytes()?)?
            .checked_add(self.loader_configuration.owned_resident_bytes()?)
    }

    fn compute_digest(&self) -> Result<String, VaeError> {
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, &VAE_SCHEMA_VERSION.to_le_bytes());
        hash_field(&mut hasher, self.artifact_root_id.as_bytes());
        hash_field(&mut hasher, self.artifact_relative_path.as_bytes());
        hash_field(&mut hasher, self.artifact_sha256.as_bytes());
        hash_field(&mut hasher, self.family.feature_id().as_bytes());
        hash_field(&mut hasher, self.family.identifier().as_bytes());
        hash_field(&mut hasher, self.family.architecture_version().as_bytes());
        hash_field(&mut hasher, self.latent_format.feature_id().as_bytes());
        hash_field(&mut hasher, self.latent_format.identifier().as_bytes());
        hash_field(&mut hasher, self.architecture.as_str().as_bytes());
        hash_field(&mut hasher, &self.patch.schema_version.to_le_bytes());
        hash_field(&mut hasher, self.patch.base_artifact_digest.as_bytes());
        hash_field(&mut hasher, self.patch.ordered_digest.as_bytes());
        let dtype = serde_json::to_vec(&self.dtype)
            .map_err(|error| VaeError::IdentitySerialization(error.to_string()))?;
        let device = serde_json::to_vec(&self.device)
            .map_err(|error| VaeError::IdentitySerialization(error.to_string()))?;
        let boundary = serde_json::to_vec(&self.boundary)
            .map_err(|error| VaeError::IdentitySerialization(error.to_string()))?;
        hash_field(&mut hasher, &dtype);
        hash_field(&mut hasher, &device);
        hash_field(&mut hasher, &boundary);
        let profile = self.profile.identity_bytes()?;
        hash_field(&mut hasher, &profile);
        let loader_configuration = serde_json::to_vec(&self.loader_configuration)
            .map_err(|error| VaeError::IdentitySerialization(error.to_string()))?;
        hash_field(&mut hasher, &loader_configuration);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

impl Serialize for VaeIdentity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        VaeIdentityWire {
            schema_version: VAE_SCHEMA_VERSION,
            artifact_root_id: self.artifact_root_id.clone(),
            artifact_relative_path: self.artifact_relative_path.clone(),
            artifact_sha256: self.artifact_sha256.clone(),
            family: self.family.clone(),
            latent_format: self.latent_format.clone(),
            architecture: self.architecture.clone(),
            patch: self.patch.clone(),
            dtype: self.dtype,
            device: self.device,
            boundary: self.boundary.clone(),
            profile: self.profile.clone(),
            loader_configuration: self.loader_configuration.clone(),
            digest: self.digest.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for VaeIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = VaeIdentityWire::deserialize(deserializer)?;
        if wire.schema_version != VAE_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(VaeError::SchemaVersion(
                wire.schema_version,
            )));
        }
        validate_identity_field("artifact root id", &wire.artifact_root_id)
            .map_err(serde::de::Error::custom)?;
        validate_identity_field("artifact relative path", &wire.artifact_relative_path)
            .map_err(serde::de::Error::custom)?;
        validate_sha256(&wire.artifact_sha256).map_err(serde::de::Error::custom)?;
        validate_patch_identity_fields(&wire.artifact_sha256, &wire.patch)
            .map_err(serde::de::Error::custom)?;
        validate_execution_dtype(wire.dtype).map_err(serde::de::Error::custom)?;
        validate_architecture_profile_pair(&wire.architecture, &wire.profile)
            .map_err(serde::de::Error::custom)?;
        wire.loader_configuration
            .validate_for_profile(&wire.profile)
            .map_err(serde::de::Error::custom)?;
        validate_vae_identity_target(
            &wire.architecture,
            &wire.profile,
            &wire.family,
            &wire.latent_format,
            wire.dtype,
            wire.device,
            wire.boundary.kind(),
        )
        .map_err(serde::de::Error::custom)?;
        let identity = Self {
            artifact_root_id: wire.artifact_root_id,
            artifact_relative_path: wire.artifact_relative_path,
            artifact_sha256: wire.artifact_sha256,
            family: wire.family,
            latent_format: wire.latent_format,
            architecture: wire.architecture,
            patch: wire.patch,
            dtype: wire.dtype,
            device: wire.device,
            boundary: wire.boundary,
            profile: wire.profile,
            loader_configuration: wire.loader_configuration,
            digest: wire.digest,
        };
        validate_sha256(&identity.digest).map_err(serde::de::Error::custom)?;
        if identity
            .compute_digest()
            .map_err(serde::de::Error::custom)?
            != identity.digest
        {
            return Err(serde::de::Error::custom(VaeError::IdentityDigestMismatch));
        }
        Ok(identity)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VaeDescriptor {
    identity: VaeIdentity,
    latent_format: LatentFormatDescriptor,
    decode_clamp: [f32; 2],
}

pub enum NativeVaeRecipeResource {
    Native(NativeVae),
    Structured(NativeStructuredVae),
}

#[cfg(feature = "test-support")]
pub(crate) fn reduced_recipe_state_module(
    owner: &'static str,
    ordered_state: Vec<(String, Tensor)>,
    cancellation: &CancellationToken,
) -> Result<NativeModule, VaeError> {
    cancellation.check().map_err(TensorError::from)?;
    let mut names = BTreeSet::new();
    let mut children = Vec::new();
    children
        .try_reserve_exact(ordered_state.len())
        .map_err(|_| VaeError::ShapeOverflow)?;
    for (name, tensor) in ordered_state {
        cancellation.check().map_err(TensorError::from)?;
        if !names.insert(name.clone()) {
            return Err(VaeError::InvalidIdentity {
                field: "reduced recipe state",
                value: format!("{owner} contains duplicate state {name}"),
            });
        }
        let elements = tensor.descriptor().element_count()?;
        if elements == 0 {
            return Err(VaeError::InvalidIdentity {
                field: "reduced recipe state",
                value: format!("{owner} contains empty state {name}"),
            });
        }
        let output_features = usize::try_from(elements).map_err(|_| VaeError::ShapeOverflow)?;
        let descriptor = tensor.descriptor().reshaped_view(vec![elements, 1])?;
        let weight = tensor.view(descriptor, ViewAccess::ReadOnly)?;
        let mut child = NativeModule::linear(name, 1, output_features, false, false)?;
        child.load_dense_parameters(weight, None)?;
        children.push(child);
    }
    if children.is_empty() {
        return Err(VaeError::NativeModuleHasNoState);
    }
    Ok(NativeModule::module_dict(owner, children)?)
}

#[derive(Debug, Error)]
pub enum NativeVaeRecipeError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    ModelFamily(#[from] crate::ModelFamilyError),
    #[error(transparent)]
    Architecture(#[from] VaeArchitectureError),
    #[error(transparent)]
    Patch(#[from] PatchGraphError),
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    Image(#[from] Box<crate::vae_image::ImageVaeError>),
    #[error(transparent)]
    Video(#[from] Box<crate::vae_video::VideoVaeError>),
    #[error(transparent)]
    Audio(#[from] Box<crate::vae_audio::AudioVaeError>),
    #[error(transparent)]
    Structured(#[from] Box<crate::vae_structured::StructuredVaeError>),
}

impl From<crate::vae_image::ImageVaeError> for NativeVaeRecipeError {
    fn from(error: crate::vae_image::ImageVaeError) -> Self {
        Self::Image(Box::new(error))
    }
}

impl From<crate::vae_video::VideoVaeError> for NativeVaeRecipeError {
    fn from(error: crate::vae_video::VideoVaeError) -> Self {
        Self::Video(Box::new(error))
    }
}

impl From<crate::vae_audio::AudioVaeError> for NativeVaeRecipeError {
    fn from(error: crate::vae_audio::AudioVaeError) -> Self {
        Self::Audio(Box::new(error))
    }
}

impl From<crate::vae_structured::StructuredVaeError> for NativeVaeRecipeError {
    fn from(error: crate::vae_structured::StructuredVaeError) -> Self {
        Self::Structured(Box::new(error))
    }
}

pub fn load_native_vae_recipe(
    source_form: crate::model_family::NativeModelRecipeSourceForm,
    source_identity_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    explicit_configuration: Option<&str>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<NativeVaeRecipeResource, NativeVaeRecipeError> {
    load_native_vae_recipe_with_construction(
        source_form,
        source_identity_sha256,
        ordered_state,
        explicit_configuration,
        backend,
        context,
        VaeRecipeConstruction::Exact,
    )
}

#[cfg(feature = "test-support")]
pub fn load_native_vae_reduced_recipe(
    source_form: crate::model_family::NativeModelRecipeSourceForm,
    source_identity_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    explicit_configuration: Option<&str>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<NativeVaeRecipeResource, NativeVaeRecipeError> {
    load_native_vae_recipe_with_construction(
        source_form,
        source_identity_sha256,
        ordered_state,
        explicit_configuration,
        backend,
        context,
        VaeRecipeConstruction::ReducedFixture,
    )
}

#[derive(Clone, Copy)]
enum VaeRecipeConstruction {
    Exact,
    #[cfg(feature = "test-support")]
    ReducedFixture,
}

#[allow(clippy::too_many_arguments)]
fn load_native_vae_recipe_with_construction(
    source_form: crate::model_family::NativeModelRecipeSourceForm,
    source_identity_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    explicit_configuration: Option<&str>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    construction: VaeRecipeConstruction,
) -> Result<NativeVaeRecipeResource, NativeVaeRecipeError> {
    context.check()?;
    let configuration_is_explicit = explicit_configuration
        .map(checked_recipe_configuration_is_explicit)
        .transpose()?;
    let mut state = BTreeMap::new();
    let mut facts = BTreeMap::new();
    let has_checkpoint_vae = ordered_state
        .iter()
        .any(|(name, _)| name.starts_with("first_stage_model."));
    if source_form == crate::model_family::NativeModelRecipeSourceForm::Checkpoint
        && !has_checkpoint_vae
    {
        return Err(crate::ModelFamilyError::InvalidStateKey(
            "checkpoint VAE recipe omitted first_stage_model state".to_owned(),
        )
        .into());
    }
    for (name, tensor) in ordered_state {
        let canonical_name = match source_form {
            crate::model_family::NativeModelRecipeSourceForm::Standalone => {
                if name.starts_with("first_stage_model.") {
                    return Err(crate::ModelFamilyError::InvalidStateKey(
                        "standalone VAE recipe contains a checkpoint wrapper".to_owned(),
                    )
                    .into());
                }
                name
            }
            crate::model_family::NativeModelRecipeSourceForm::Checkpoint => {
                if let Some(canonical) = name.strip_prefix("first_stage_model.") {
                    if canonical.is_empty() {
                        return Err(crate::ModelFamilyError::InvalidStateKey(
                            "checkpoint VAE recipe contains an empty canonical key".to_owned(),
                        )
                        .into());
                    }
                    canonical.to_owned()
                } else if name.starts_with("model.")
                    || name.starts_with("cond_stage_model.")
                    || name.starts_with("conditioner.")
                    || name.starts_with("model_ema.")
                    || name.starts_with("embedding_manager.")
                    || name == "global_step"
                {
                    continue;
                } else {
                    return Err(crate::ModelFamilyError::InvalidStateKey(format!(
                        "checkpoint VAE recipe contains disallowed sibling {name}"
                    ))
                    .into());
                }
            }
        };
        if state
            .insert(canonical_name.clone(), tensor.clone())
            .is_some()
        {
            return Err(crate::ModelFamilyError::InvalidStateKey(format!(
                "duplicate canonical VAE recipe tensor {canonical_name}"
            ))
            .into());
        }
        facts.insert(
            canonical_name,
            ModelParsedTensorFact {
                shape: tensor.descriptor().shape().to_vec(),
                storage_dtype: tensor.descriptor().dtype().catalog_name().to_owned(),
            },
        );
    }
    let mut formats = Vec::new();
    let explicit_configuration = explicit_configuration.and_then(|configuration| {
        if configuration_is_explicit == Some(true) {
            Some(configuration)
        } else {
            formats.push(ModelParsedFormatFact {
                identity: "json_config".to_owned(),
                metadata: BTreeMap::from([("config".to_owned(), configuration.to_owned())]),
            });
            None
        }
    });
    let probe = ModelProbe::from_parsed_facts_cancellable(
        ModelParsedFacts {
            tensors: facts,
            formats,
        },
        context.cancellation,
    )?;
    let registry = VaeArchitectureRegistry::checked()?;
    let selection = match explicit_configuration {
        Some(configuration) => {
            registry.select_explicit(&probe, configuration, context.cancellation)?
        }
        None => registry.select(&probe, context.cancellation)?,
    };
    selection.ensure_native_builder_available()?;
    let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
    let target = registry.intended_target(
        &selection,
        &family_registry,
        &latent_registry,
        context.cancellation,
    )?;
    let latent_definition = latent_registry.get(target.latent_format()).ok_or_else(|| {
        VaeArchitectureError::UnknownLatentFormat(target.latent_format().identifier().to_owned())
    })?;
    let boundary = match selection.boundary() {
        VaeBoundaryDomain::Image => VaeBoundary::image(3)?,
        VaeBoundaryDomain::Video => VaeBoundary::video(3)?,
        VaeBoundaryDomain::Audio => {
            let sample_rate = if selection.profile() == &VaeKernelProfile::AudioOobleck44KhzV1 {
                44_100
            } else {
                48_000
            };
            VaeBoundary::audio(2, sample_rate)?
        }
        VaeBoundaryDomain::Structured => {
            let kind = if selection.profile() == &VaeKernelProfile::TripoSplatV1 {
                VaeStructuredOutputKind::GaussianSplats
            } else {
                VaeStructuredOutputKind::Shape
            };
            VaeBoundary::structured_output(1, kind)?
        }
    };
    let patch = PatchGraph::checked_semantic(source_identity_sha256, Vec::new())?.identity();
    let descriptor = VaeDescriptor::checked_recipe_selection(
        source_identity_sha256,
        &selection,
        &target,
        &family_registry,
        &latent_registry,
        patch,
        boundary,
        [-1.0, 1.0],
        context.cancellation,
    )?;
    let ordered_state = state.into_iter().collect();
    Ok(match (selection.boundary(), construction) {
        (VaeBoundaryDomain::Image, VaeRecipeConstruction::Exact) => {
            NativeVaeRecipeResource::Native(crate::vae_image::load_image_vae_from_recipe_state(
                backend,
                source_identity_sha256,
                ordered_state,
                descriptor,
                latent_definition,
                context,
            )?)
        }
        #[cfg(feature = "test-support")]
        (VaeBoundaryDomain::Image, VaeRecipeConstruction::ReducedFixture) => {
            NativeVaeRecipeResource::Native(
                crate::vae_image::load_image_vae_from_reduced_recipe_state(
                    backend,
                    source_identity_sha256,
                    ordered_state,
                    descriptor,
                    latent_definition,
                    context,
                )?,
            )
        }
        (VaeBoundaryDomain::Video, VaeRecipeConstruction::Exact) => {
            NativeVaeRecipeResource::Native(crate::vae_video::load_video_vae_from_recipe_state(
                backend,
                source_identity_sha256,
                ordered_state,
                descriptor,
                latent_definition,
                context,
            )?)
        }
        #[cfg(feature = "test-support")]
        (VaeBoundaryDomain::Video, VaeRecipeConstruction::ReducedFixture) => {
            NativeVaeRecipeResource::Native(
                crate::vae_video::load_video_vae_from_reduced_recipe_state(
                    backend,
                    source_identity_sha256,
                    ordered_state,
                    descriptor,
                    latent_definition,
                    context,
                )?,
            )
        }
        (VaeBoundaryDomain::Audio, VaeRecipeConstruction::Exact) => {
            NativeVaeRecipeResource::Native(crate::vae_audio::load_audio_vae_from_recipe_state(
                backend,
                source_identity_sha256,
                ordered_state,
                descriptor,
                latent_definition,
                context,
            )?)
        }
        #[cfg(feature = "test-support")]
        (VaeBoundaryDomain::Audio, VaeRecipeConstruction::ReducedFixture) => {
            NativeVaeRecipeResource::Native(
                crate::vae_audio::load_audio_vae_from_reduced_recipe_state(
                    backend,
                    source_identity_sha256,
                    ordered_state,
                    descriptor,
                    latent_definition,
                    context,
                )?,
            )
        }
        (VaeBoundaryDomain::Structured, VaeRecipeConstruction::Exact) => {
            NativeVaeRecipeResource::Structured(
                crate::vae_structured::load_structured_vae_from_recipe_state(
                    backend,
                    source_identity_sha256,
                    ordered_state,
                    descriptor,
                    latent_definition,
                    context,
                )?,
            )
        }
        #[cfg(feature = "test-support")]
        (VaeBoundaryDomain::Structured, VaeRecipeConstruction::ReducedFixture) => {
            NativeVaeRecipeResource::Structured(
                crate::vae_structured::load_structured_vae_from_reduced_recipe_state(
                    backend,
                    source_identity_sha256,
                    ordered_state,
                    descriptor,
                    latent_definition,
                    context,
                )?,
            )
        }
    })
}

impl VaeDescriptor {
    fn owned_resident_bytes(&self) -> Option<u64> {
        self.identity
            .owned_resident_bytes()?
            .checked_add(self.latent_format.owned_resident_bytes()?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn checked_selection(
        artifact: &ArtifactRecord,
        selection: &VaeArchitectureSelection,
        target: &VaeExecutionTarget,
        family_registry: &ModelFamilyRegistry,
        latent_registry: &LatentFormatRegistry,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        cancellation: &CancellationToken,
    ) -> Result<Self, VaeError> {
        let architecture_registry = VaeArchitectureRegistry::checked()?;
        architecture_registry.validate_target(
            selection,
            target,
            family_registry,
            latent_registry,
            cancellation,
        )?;
        let expected_boundary = match selection.boundary() {
            VaeBoundaryDomain::Image => VaeBoundaryKind::Image,
            VaeBoundaryDomain::Video => VaeBoundaryKind::Video,
            VaeBoundaryDomain::Audio => VaeBoundaryKind::Audio,
            VaeBoundaryDomain::Structured => VaeBoundaryKind::StructuredOutput,
        };
        if boundary.kind() != expected_boundary {
            return Err(VaeError::SelectionBoundaryMismatch {
                expected: expected_boundary,
                actual: boundary.kind(),
            });
        }
        let latent_definition = latent_registry.get(target.latent_format()).ok_or_else(|| {
            VaeArchitectureError::UnknownLatentFormat(
                target.latent_format().identifier().to_owned(),
            )
        })?;
        let mut descriptor = Self::checked(
            artifact,
            target.family().clone(),
            latent_definition,
            selection.architecture().clone(),
            patch,
            target.dtype(),
            target.device(),
            boundary,
            selection.profile().clone(),
            decode_clamp,
        )?;
        descriptor.identity.loader_configuration = selection.loader_configuration().clone();
        descriptor
            .identity
            .loader_configuration
            .validate_for_profile(descriptor.identity.profile())?;
        descriptor.identity.digest = descriptor.identity.compute_digest()?;
        Ok(descriptor)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn checked_recipe_selection(
        artifact_sha256: &str,
        selection: &VaeArchitectureSelection,
        target: &VaeExecutionTarget,
        family_registry: &ModelFamilyRegistry,
        latent_registry: &LatentFormatRegistry,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        cancellation: &CancellationToken,
    ) -> Result<Self, VaeError> {
        if !decode_clamp[0].is_finite()
            || !decode_clamp[1].is_finite()
            || decode_clamp[0] >= decode_clamp[1]
        {
            return Err(VaeError::InvalidClamp(decode_clamp));
        }
        let architecture_registry = VaeArchitectureRegistry::checked()?;
        architecture_registry.validate_target(
            selection,
            target,
            family_registry,
            latent_registry,
            cancellation,
        )?;
        let latent_definition = latent_registry.get(target.latent_format()).ok_or_else(|| {
            VaeArchitectureError::UnknownLatentFormat(
                target.latent_format().identifier().to_owned(),
            )
        })?;
        let latent_format = LatentFormatDescriptor::checked(latent_definition)?;
        let expected_boundary = match selection.boundary() {
            VaeBoundaryDomain::Image => VaeBoundaryKind::Image,
            VaeBoundaryDomain::Video => VaeBoundaryKind::Video,
            VaeBoundaryDomain::Audio => VaeBoundaryKind::Audio,
            VaeBoundaryDomain::Structured => VaeBoundaryKind::StructuredOutput,
        };
        if boundary.kind() != expected_boundary {
            return Err(VaeError::SelectionBoundaryMismatch {
                expected: expected_boundary,
                actual: boundary.kind(),
            });
        }
        let mut identity = VaeIdentity::checked_recipe(
            artifact_sha256,
            target.family().clone(),
            latent_format.identity.clone(),
            selection.architecture().clone(),
            patch,
            target.dtype(),
            target.device(),
            boundary,
            selection.profile().clone(),
        )?;
        identity.loader_configuration = selection.loader_configuration().clone();
        identity
            .loader_configuration
            .validate_for_profile(identity.profile())?;
        identity.digest = identity.compute_digest()?;
        Ok(Self {
            identity,
            latent_format,
            decode_clamp,
        })
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn checked(
        artifact: &ArtifactRecord,
        family: ModelFamilyIdentity,
        latent_definition: &'static LatentFormatDefinition,
        architecture: VaeArchitectureIdentity,
        patch: PatchGraphIdentity,
        dtype: DType,
        device: DeviceId,
        boundary: VaeBoundary,
        profile: VaeKernelProfile,
        decode_clamp: [f32; 2],
    ) -> Result<Self, VaeError> {
        if !decode_clamp[0].is_finite()
            || !decode_clamp[1].is_finite()
            || decode_clamp[0] >= decode_clamp[1]
        {
            return Err(VaeError::InvalidClamp(decode_clamp));
        }
        let latent_format = LatentFormatDescriptor::checked(latent_definition)?;
        validate_execution_dtype(dtype)?;
        if latent_definition.transform != LatentTransform::Identity && dtype != DType::F32 {
            return Err(VaeError::LatentTransformDTypeMismatch {
                transform: latent_definition.transform,
                dtype,
            });
        }
        if profile.is_conformance_only() {
            boundary.validate_latent_dimensions(latent_definition.dimensions)?;
        } else {
            let contract = profile.contract();
            if boundary.kind() != contract.boundary {
                return Err(VaeError::SelectionBoundaryMismatch {
                    expected: contract.boundary,
                    actual: boundary.kind(),
                });
            }
            if latent_definition.dimensions != contract.latent_dimensions {
                return Err(VaeError::BoundaryLatentDimensionMismatch {
                    kind: boundary.kind(),
                    dimensions: latent_definition.dimensions,
                });
            }
        }
        let identity = VaeIdentity::checked(
            artifact,
            family,
            latent_format.identity.clone(),
            architecture,
            patch,
            dtype,
            device,
            boundary,
            profile,
        )?;
        Ok(Self {
            identity,
            latent_format,
            decode_clamp,
        })
    }

    pub fn identity(&self) -> &VaeIdentity {
        &self.identity
    }

    pub fn latent_format(&self) -> &LatentFormatDescriptor {
        &self.latent_format
    }

    pub const fn pixel_channels(&self) -> u64 {
        self.identity.boundary.channels()
    }

    pub fn boundary(&self) -> &VaeBoundary {
        self.identity.boundary()
    }

    pub fn source_image_encode_downscale_ratio(&self) -> Result<u64, VaeError> {
        source_image_encode_downscale_ratio(
            self.identity.profile(),
            self.identity.loader_configuration(),
            self.latent_format.spatial_downscale_ratio,
        )
    }

    pub fn source_image_decode_upscale_ratio(&self) -> Result<u64, VaeError> {
        source_image_decode_upscale_ratio(
            self.identity.profile(),
            self.identity.loader_configuration(),
            self.latent_format.spatial_downscale_ratio,
        )
    }

    pub fn is_conformance_only(&self) -> bool {
        self.identity.profile().is_conformance_only()
    }

    pub const fn decode_clamp(&self) -> [f32; 2] {
        self.decode_clamp
    }
}

#[derive(Clone, Debug)]
pub(crate) struct VaeModelBinding {
    identity: VaeIdentity,
    module: NativeModule,
    digest: String,
}

#[allow(dead_code)]
impl VaeModelBinding {
    pub(crate) fn checked(
        descriptor: &VaeDescriptor,
        store: &ModelStore,
        loaded_model: Arc<LoadedModel>,
        module: NativeModule,
        cancellation: &CancellationToken,
    ) -> Result<Self, VaeError> {
        cancellation.check().map_err(TensorError::from)?;
        if !descriptor.identity().profile().is_conformance_only()
            && !descriptor
                .identity()
                .profile()
                .uses_explicit_source_admission()
        {
            let probe = store.family_probe(&loaded_model, cancellation)?;
            let registry = VaeArchitectureRegistry::checked()?;
            let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
            let target = VaeExecutionTarget::new(
                descriptor.identity().family().clone(),
                descriptor.identity().latent_format().clone(),
                descriptor.identity().dtype(),
                descriptor.identity().device(),
            );
            let selected = registry.select_for_target(
                &probe,
                &target,
                &family_registry,
                &latent_registry,
                cancellation,
            )?;
            if selected.architecture() != descriptor.identity().architecture()
                || selected.profile() != descriptor.identity().profile()
            {
                return Err(VaeError::ModelArchitectureMismatch {
                    expected: descriptor.identity().architecture().as_str().to_owned(),
                    actual: selected.architecture().as_str().to_owned(),
                });
            }
        }
        let artifact_key = ArtifactKey::new(
            descriptor.identity().artifact_root_id(),
            descriptor.identity().artifact_relative_path(),
        )
        .map_err(|_| VaeError::ModelArtifactMismatch)?;
        store.validate_loaded_artifact_identity(
            &loaded_model,
            &artifact_key,
            descriptor.identity().artifact_sha256(),
            cancellation,
        )?;
        if !module.has_execution_state() {
            return Err(VaeError::NativeModuleHasNoState);
        }
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, descriptor.identity().digest().as_bytes());
        hash_field(&mut hasher, loaded_model.identity().as_bytes());
        hash_field(
            &mut hasher,
            module.semantic_state_digest(cancellation)?.as_bytes(),
        );
        let digest = format!("{:x}", hasher.finalize());
        Ok(Self {
            identity: descriptor.identity().clone(),
            module,
            digest,
        })
    }

    pub(crate) fn checked_recipe(
        descriptor: &VaeDescriptor,
        source_identity_sha256: &str,
        module: NativeModule,
        cancellation: &CancellationToken,
    ) -> Result<Self, VaeError> {
        cancellation.check().map_err(TensorError::from)?;
        validate_sha256(source_identity_sha256)?;
        if source_identity_sha256 != descriptor.identity().artifact_sha256() {
            return Err(VaeError::ModelArtifactMismatch);
        }
        if !module.has_execution_state() {
            return Err(VaeError::NativeModuleHasNoState);
        }
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, b"zed.sim.vae.recipe-binding.v1");
        hash_field(&mut hasher, descriptor.identity().digest().as_bytes());
        hash_field(&mut hasher, source_identity_sha256.as_bytes());
        hash_field(
            &mut hasher,
            module.semantic_state_digest(cancellation)?.as_bytes(),
        );
        Ok(Self {
            identity: descriptor.identity().clone(),
            module,
            digest: format!("{:x}", hasher.finalize()),
        })
    }

    #[cfg(test)]
    pub(crate) fn checked_transport_fixture(
        descriptor: &VaeDescriptor,
        module: NativeModule,
        cancellation: &CancellationToken,
    ) -> Result<Self, VaeError> {
        cancellation.check().map_err(TensorError::from)?;
        if !module.has_execution_state() {
            return Err(VaeError::NativeModuleHasNoState);
        }
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, descriptor.identity().digest().as_bytes());
        hash_field(&mut hasher, b"structured-vae-transport-fixture-v1");
        hash_field(
            &mut hasher,
            module.semantic_state_digest(cancellation)?.as_bytes(),
        );
        Ok(Self {
            identity: descriptor.identity().clone(),
            module,
            digest: format!("{:x}", hasher.finalize()),
        })
    }

    pub(crate) fn identity(&self) -> &VaeIdentity {
        &self.identity
    }

    pub(crate) fn module(&self) -> &NativeModule {
        &self.module
    }

    pub(crate) fn digest(&self) -> &str {
        &self.digest
    }

    fn owned_resident_bytes(&self) -> Option<u64> {
        self.identity
            .owned_resident_bytes()?
            .checked_add(u64::try_from(self.digest.capacity()).ok()?)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VaeOperation {
    Encode,
    Decode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VaeSpatialGeometry {
    encode: Vec<VaeTileAxisFormula>,
    decode: Vec<VaeTileAxisFormula>,
}

impl VaeSpatialGeometry {
    fn owned_resident_bytes(&self) -> Option<u64> {
        self.encode
            .capacity()
            .checked_add(self.decode.capacity())?
            .checked_mul(std::mem::size_of::<VaeTileAxisFormula>())
            .and_then(|bytes| u64::try_from(bytes).ok())
    }

    fn checked(
        profile: &VaeKernelProfile,
        loader_configuration: &VaeLoaderConfiguration,
        definition: &LatentFormatDefinition,
    ) -> Result<Self, VaeError> {
        let dimensions = usize::from(definition.dimensions);
        let default = tile_axis_formulas(definition)?;
        let Some((encode_ratio, decode_ratio)) = image_spatial_ratios(
            profile,
            loader_configuration,
            definition.spatial_downscale_ratio,
        )?
        else {
            return Ok(Self {
                encode: default.clone(),
                decode: default,
            });
        };
        if dimensions != 2 {
            return Err(VaeError::InvalidOperationGeometry {
                profile: format!("{profile:?}"),
                dimensions: definition.dimensions,
            });
        }
        Ok(Self {
            encode: vec![VaeTileAxisFormula::checked_linear(encode_ratio)?; dimensions],
            decode: vec![VaeTileAxisFormula::checked_linear(decode_ratio)?; dimensions],
        })
    }

    fn formulas(&self, operation: VaeOperation) -> &[VaeTileAxisFormula] {
        match operation {
            VaeOperation::Encode => &self.encode,
            VaeOperation::Decode => &self.decode,
        }
    }
}

fn image_spatial_ratios(
    profile: &VaeKernelProfile,
    loader_configuration: &VaeLoaderConfiguration,
    latent_format_ratio: u64,
) -> Result<Option<(u64, u64)>, VaeError> {
    let loader_configuration = innermost_loader_configuration(loader_configuration);
    let ratios = match profile {
        VaeKernelProfile::TemporalAutoencodingEngineV1
        | VaeKernelProfile::AutoencoderKlV1
        | VaeKernelProfile::AutoencodingEngineV1 => (8, 8),
        VaeKernelProfile::AutoencoderKlX4V1 | VaeKernelProfile::AutoencodingEngineX4V1 => (4, 4),
        VaeKernelProfile::AutoencoderKlBatchNormV1
        | VaeKernelProfile::AutoencodingEngineBatchNormV1 => {
            let base_ratio = match loader_configuration {
                VaeLoaderConfiguration::DefaultKl { x4: true, .. } => 4_u64,
                _ => 8_u64,
            };
            let ratio = base_ratio.checked_mul(2).ok_or(VaeError::ShapeOverflow)?;
            (ratio, ratio)
        }
        VaeKernelProfile::ExplicitAutoencoderKlV1 => match loader_configuration {
            VaeLoaderConfiguration::ExplicitAutoencoderKl { params_json, .. } => {
                let topology = ExplicitAutoencoderKlTopology::parse(params_json)?;
                (topology.encode_ratio()?, topology.decode_ratio()?)
            }
            _ => (latent_format_ratio, latent_format_ratio),
        },
        VaeKernelProfile::TaesdV1 => {
            let ratio = match loader_configuration {
                VaeLoaderConfiguration::Taesd {
                    latent_channels: 128,
                    ..
                } => 16,
                _ => 8,
            };
            (ratio, ratio)
        }
        VaeKernelProfile::StableCascadeStageAV1 => (4, 4),
        VaeKernelProfile::StableCascadeStageCEncoderV1
        | VaeKernelProfile::StableCascadeStageCPreviewerV1
        | VaeKernelProfile::StableCascadeStageCCombinedV1 => (32, 8),
        VaeKernelProfile::HunyuanImageV1 => (32, 32),
        VaeKernelProfile::PixelSpaceV1 => (1, 1),
        _ => return Ok(None),
    };
    Ok(Some(ratios))
}

fn source_image_encode_downscale_ratio(
    profile: &VaeKernelProfile,
    loader_configuration: &VaeLoaderConfiguration,
    latent_format_ratio: u64,
) -> Result<u64, VaeError> {
    if profile.expected_boundary_kind() != VaeBoundaryKind::Image {
        return Err(VaeError::InvalidOperationGeometry {
            profile: format!("{profile:?}"),
            dimensions: profile.latent_dimensions(),
        });
    }
    // Explicit KL configurations and metadata-overridden TAESD checkpoints
    // preserve the source wrapper's default even if native encode geometry differs.
    if matches!(
        profile,
        VaeKernelProfile::ExplicitAutoencoderKlV1 | VaeKernelProfile::TaesdV1
    ) {
        return Ok(8);
    }
    // The source previewer leaves VAE.downscale_ratio at its default eight,
    // whereas encoder and combined checkpoints explicitly set it to 32.
    if *profile == VaeKernelProfile::StableCascadeStageCPreviewerV1 {
        return Ok(8);
    }
    Ok(
        image_spatial_ratios(profile, loader_configuration, latent_format_ratio)?
            .map_or(latent_format_ratio, |(encode_ratio, _)| encode_ratio),
    )
}

fn source_image_decode_upscale_ratio(
    profile: &VaeKernelProfile,
    loader_configuration: &VaeLoaderConfiguration,
    latent_format_ratio: u64,
) -> Result<u64, VaeError> {
    if profile.expected_boundary_kind() != VaeBoundaryKind::Image {
        return Err(VaeError::InvalidOperationGeometry {
            profile: format!("{profile:?}"),
            dimensions: profile.latent_dimensions(),
        });
    }
    // These source wrappers retain the default ratio even when the checkpoint's
    // native kernel geometry differs.
    if matches!(
        profile,
        VaeKernelProfile::ExplicitAutoencoderKlV1 | VaeKernelProfile::TaesdV1
    ) {
        return Ok(8);
    }
    Ok(
        image_spatial_ratios(profile, loader_configuration, latent_format_ratio)?
            .map_or(latent_format_ratio, |(_, decode_ratio)| decode_ratio),
    )
}

fn innermost_loader_configuration(
    loader_configuration: &VaeLoaderConfiguration,
) -> &VaeLoaderConfiguration {
    match loader_configuration {
        VaeLoaderConfiguration::DiffusersPreconverted { inner, .. } => {
            innermost_loader_configuration(inner)
        }
        loader_configuration => loader_configuration,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaeTilePlan {
    vae_digest: String,
    operation: VaeOperation,
    input_shape: Vec<u64>,
    output_shape: Vec<u64>,
    execution_output_shape: Vec<u64>,
    extra_1d_channels: Option<u64>,
    tiling: TileExecutionPlan,
}

impl VaeTilePlan {
    pub fn operation(&self) -> VaeOperation {
        self.operation
    }

    pub fn input_shape(&self) -> &[u64] {
        &self.input_shape
    }

    pub fn effective_input_spatial_shape(&self) -> &[u64] {
        self.tiling.input_spatial_shape()
    }

    pub fn input_spatial_offsets(&self) -> &[u64] {
        self.tiling.input_spatial_offsets()
    }

    pub fn output_shape(&self) -> &[u64] {
        &self.output_shape
    }

    pub fn tile_extent(&self) -> &[u64] {
        self.tiling.tile_extent()
    }

    pub fn overlap(&self) -> &[u64] {
        self.tiling.overlap()
    }

    pub fn pass_count(&self) -> usize {
        self.tiling.pass_count()
    }

    pub fn tile_count(&self) -> u64 {
        self.tiling.tile_count()
    }

    fn checked(
        vae_digest: &str,
        operation: VaeOperation,
        input_shape: Vec<u64>,
        output_shape: Vec<u64>,
        execution_output_shape: Vec<u64>,
        extra_1d_channels: Option<u64>,
        tiling: TileExecutionPlan,
    ) -> Result<Self, VaeError> {
        Ok(Self {
            vae_digest: vae_digest.to_owned(),
            operation,
            input_shape,
            output_shape,
            execution_output_shape,
            extra_1d_channels,
            tiling,
        })
    }
}

#[cfg(test)]
const SD15_REDUCED_WIDTH: u64 = 4;
#[cfg(test)]
const SD15_REDUCED_LATENT_CHANNELS: u64 = 4;

#[cfg(test)]
#[derive(Clone, Debug)]
struct Sd15LearnedVaeKernel {
    identity: VaeIdentity,
    state_digest: String,
    weights: BTreeMap<String, Tensor>,
}

#[cfg(test)]
impl Sd15LearnedVaeKernel {
    fn checked(identity: VaeIdentity, weights: BTreeMap<String, Tensor>) -> Result<Self, VaeError> {
        if identity.profile() != &VaeKernelProfile::Sd15AutoencoderKlReducedV1
            || !identity.profile().uses_explicit_source_admission()
        {
            return Err(VaeError::KernelProfileMismatch);
        }
        if identity.dtype() != DType::F32 || identity.device() != DeviceId::CPU {
            return Err(VaeError::KernelExecutionBindingMismatch {
                expected_dtype: DType::F32,
                actual_dtype: identity.dtype(),
                expected_device: DeviceId::CPU,
                actual_device: identity.device(),
            });
        }
        let manifest = sd15_reduced_weight_manifest();
        let expected = manifest.keys().collect::<BTreeSet<_>>();
        let actual = weights.keys().collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(VaeError::LearnedWeightKeys {
                missing: expected
                    .difference(&actual)
                    .map(|value| (*value).clone())
                    .collect(),
                unexpected: actual
                    .difference(&expected)
                    .map(|value| (*value).clone())
                    .collect(),
            });
        }
        let mut hasher = Sha256::new();
        for (name, expected_shape) in &manifest {
            let tensor = weights
                .get(name)
                .ok_or_else(|| VaeError::MissingLearnedWeight(name.clone()))?;
            let descriptor = tensor.descriptor();
            if descriptor.dtype() != DType::F32
                || descriptor.device().kind() != DeviceKind::Cpu
                || descriptor.shape() != expected_shape
            {
                return Err(VaeError::LearnedWeightShape {
                    key: name.clone(),
                    expected: expected_shape.clone(),
                    actual: descriptor.shape().to_vec(),
                    dtype: descriptor.dtype(),
                    device: descriptor.device().kind(),
                });
            }
            hash_field(&mut hasher, name.as_bytes());
            for dimension in expected_shape {
                hasher.update(dimension.to_le_bytes());
            }
            hash_field(&mut hasher, tensor.contiguous_bytes()?);
        }
        let state_digest = format!("{:x}", hasher.finalize());
        Ok(Self {
            identity,
            state_digest,
            weights,
        })
    }

    pub fn identity(&self) -> &VaeIdentity {
        &self.identity
    }

    pub fn state_digest(&self) -> &str {
        &self.state_digest
    }

    fn resident_storage_bytes(&self) -> Result<u64, VaeError> {
        let mut storages = BTreeSet::new();
        self.weights.values().try_fold(0_u64, |total, tensor| {
            if !storages.insert(tensor.storage_id().get()) {
                return Ok(total);
            }
            total
                .checked_add(tensor.storage_byte_len())
                .ok_or_else(|| VaeError::Allocation("resident storage byte overflow".to_owned()))
        })
    }

    fn owned_resident_bytes(&self) -> Option<u64> {
        let entries = self
            .weights
            .len()
            .checked_mul(std::mem::size_of::<(String, Tensor)>())?;
        let bytes = u64::try_from(std::mem::size_of::<Self>())
            .ok()?
            .checked_add(self.identity.owned_resident_bytes()?)?
            .checked_add(u64::try_from(self.state_digest.capacity()).ok()?)?
            .checked_add(u64::try_from(entries).ok()?)?;
        self.weights.keys().try_fold(bytes, |total, name| {
            total.checked_add(u64::try_from(name.capacity()).ok()?)
        })
    }

    fn weight(&self, key: &str) -> Result<&Tensor, VaeError> {
        self.weights
            .get(key)
            .ok_or_else(|| VaeError::MissingLearnedWeight(key.to_owned()))
    }

    fn conv(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        prefix: &str,
        stride: usize,
        padding: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        Ok(conv2d(
            backend,
            input,
            self.weight(&format!("{prefix}.weight"))?,
            Some(self.weight(&format!("{prefix}.bias"))?),
            stride,
            padding,
            context,
        )?)
    }

    fn normalize(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        Ok(group_norm(
            backend,
            input,
            self.weight(&format!("{prefix}.weight"))?,
            self.weight(&format!("{prefix}.bias"))?,
            1,
            1e-6,
            context,
        )?)
    }

    fn block(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let hidden = self.normalize(backend, input, &format!("{prefix}.norm1"), context)?;
        let hidden = silu(backend, &hidden, context)?;
        let hidden = self.conv(backend, &hidden, &format!("{prefix}.conv1"), 1, 1, context)?;
        let hidden = self.normalize(backend, &hidden, &format!("{prefix}.norm2"), context)?;
        let hidden = silu(backend, &hidden, context)?;
        let hidden = self.conv(backend, &hidden, &format!("{prefix}.conv2"), 1, 1, context)?;
        Ok(sim_tensor::generated_native_diffusion::add(
            backend, input, &hidden, context,
        )?)
    }

    fn encode(
        &self,
        backend: &CpuBackend,
        pixels: &Tensor,
        latent_definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        context.check()?;
        let shape = pixels.descriptor().shape();
        if shape.len() != 4 || shape[1] != 3 || shape[0] == 0 || shape[2] < 8 || shape[3] < 8 {
            return Err(VaeError::InvalidShape {
                expected: vec![0, 3, 8, 8],
                actual: shape.to_vec(),
            });
        }
        let cropped = center_crop_and_scale_sd15(backend, pixels, context)?;
        let mut hidden = self.conv(backend, &cropped, "encoder.conv_in", 1, 1, context)?;
        hidden = self.block(backend, &hidden, "encoder.block", context)?;
        for level in 0..3 {
            hidden = self.conv(
                backend,
                &hidden,
                &format!("encoder.down.{level}"),
                2,
                1,
                context,
            )?;
            hidden = self.block(backend, &hidden, "encoder.block", context)?;
        }
        hidden = self.normalize(backend, &hidden, "encoder.norm_out", context)?;
        hidden = silu(backend, &hidden, context)?;
        hidden = self.conv(backend, &hidden, "encoder.conv_out", 1, 1, context)?;
        hidden = self.conv(backend, &hidden, "quant_conv", 1, 0, context)?;
        let moments = tensor_to_f32(backend, &hidden, context)?;
        let moment_shape = hidden.descriptor().shape();
        let spatial = usize::try_from(
            moment_shape[2]
                .checked_mul(moment_shape[3])
                .ok_or(VaeError::ShapeOverflow)?,
        )
        .map_err(|_| VaeError::ShapeOverflow)?;
        let batch = usize::try_from(moment_shape[0]).map_err(|_| VaeError::ShapeOverflow)?;
        let count = batch
            .checked_mul(
                usize::try_from(SD15_REDUCED_LATENT_CHANNELS)
                    .map_err(|_| VaeError::ShapeOverflow)?,
            )
            .and_then(|value| value.checked_mul(spatial))
            .ok_or(VaeError::ShapeOverflow)?;
        let mut mode = backend.workspace_vec(context, count)?;
        for batch_index in 0..batch {
            for channel in 0..usize::try_from(SD15_REDUCED_LATENT_CHANNELS)
                .map_err(|_| VaeError::ShapeOverflow)?
            {
                for position in 0..spatial {
                    check_periodically(
                        u64::try_from(position).map_err(|_| VaeError::ShapeOverflow)?,
                        context,
                    )?;
                    mode.try_push(moments[(batch_index * 8 + channel) * spatial + position])?;
                }
            }
        }
        let mode = tensor_from_f32(
            backend,
            &[moment_shape[0], 4, moment_shape[2], moment_shape[3]],
            &mode,
            context,
        )?;
        Ok(process_latent_in(
            latent_definition,
            backend,
            &mode,
            context,
        )?)
    }

    fn decode(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        latent_definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        context.check()?;
        let shape = latent.descriptor().shape();
        if shape.len() != 4 || shape[1] != 4 || shape.contains(&0) {
            return Err(VaeError::InvalidShape {
                expected: vec![0, 4, 0, 0],
                actual: shape.to_vec(),
            });
        }
        let mut hidden = process_latent_out(latent_definition, backend, latent, context)?;
        hidden = self.conv(backend, &hidden, "post_quant_conv", 1, 0, context)?;
        hidden = self.conv(backend, &hidden, "decoder.conv_in", 1, 1, context)?;
        hidden = self.block(backend, &hidden, "decoder.block", context)?;
        for level in 0..3 {
            hidden = nearest_upsample_2x(backend, &hidden, context)?;
            hidden = self.conv(
                backend,
                &hidden,
                &format!("decoder.up.{level}"),
                1,
                1,
                context,
            )?;
            hidden = self.block(backend, &hidden, "decoder.block", context)?;
        }
        hidden = self.normalize(backend, &hidden, "decoder.norm_out", context)?;
        hidden = silu(backend, &hidden, context)?;
        hidden = self.conv(backend, &hidden, "decoder.conv_out", 1, 1, context)?;
        let mut values = tensor_to_f32(backend, &hidden, context)?;
        for (index, value) in values.iter_mut().enumerate() {
            check_periodically(
                u64::try_from(index).map_err(|_| VaeError::ShapeOverflow)?,
                context,
            )?;
            *value = ((*value + 1.0) * 0.5).clamp(0.0, 1.0);
        }
        Ok(tensor_from_f32(
            backend,
            hidden.descriptor().shape(),
            &values,
            context,
        )?)
    }
}

type VaeKernelFunction = fn(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    cpu_backend: Option<&CpuBackend>,
    input: &Tensor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError>;

pub(crate) type StructuredVaeKernelFunction = fn(
    module: &NativeModule,
    backend: &CpuBackend,
    latent: &Tensor,
    request: &VaeStructuredDecodeRequest,
    context: &ExecutionContext<'_>,
) -> Result<VaeStructuredResult, VaeError>;

#[derive(Clone)]
pub(crate) struct VaeKernelFunctions {
    architecture: VaeArchitectureIdentity,
    encode_raw: VaeKernelFunction,
    decode_raw: VaeKernelFunction,
}

impl std::fmt::Debug for VaeKernelFunctions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VaeKernelFunctions")
            .field("architecture", &self.architecture)
            .finish_non_exhaustive()
    }
}

impl VaeKernelFunctions {
    #[allow(dead_code)]
    pub(crate) fn checked(
        architecture: VaeArchitectureIdentity,
        encode_raw: VaeKernelFunction,
        decode_raw: VaeKernelFunction,
    ) -> Self {
        Self {
            architecture,
            encode_raw,
            decode_raw,
        }
    }

    fn owned_resident_bytes(&self) -> Option<u64> {
        self.architecture.owned_resident_bytes()
    }

    fn encode(
        &self,
        module: &NativeModule,
        backend: &dyn TensorBackend,
        cpu_backend: Option<&CpuBackend>,
        pixels: &Tensor,
        latent_definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        (self.encode_raw)(
            module,
            backend,
            cpu_backend,
            pixels,
            latent_definition,
            context,
        )
    }

    fn decode(
        &self,
        module: &NativeModule,
        backend: &dyn TensorBackend,
        cpu_backend: Option<&CpuBackend>,
        latent: &Tensor,
        latent_definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        (self.decode_raw)(
            module,
            backend,
            cpu_backend,
            latent,
            latent_definition,
            context,
        )
    }
}

#[cfg(test)]
fn sd15_reduced_weight_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut manifest = BTreeMap::new();
    let mut conv = |prefix: &str, output: u64, input: u64, kernel: u64| {
        manifest.insert(
            format!("{prefix}.weight"),
            vec![output, input, kernel, kernel],
        );
        manifest.insert(format!("{prefix}.bias"), vec![output]);
    };
    conv("encoder.conv_in", 4, 3, 3);
    conv("encoder.block.conv1", 4, 4, 3);
    conv("encoder.block.conv2", 4, 4, 3);
    for level in 0..3 {
        conv(&format!("encoder.down.{level}"), 4, 4, 3);
    }
    conv("encoder.conv_out", 8, 4, 3);
    conv("quant_conv", 8, 8, 1);
    conv("post_quant_conv", 4, 4, 1);
    conv("decoder.conv_in", 4, 4, 3);
    conv("decoder.block.conv1", 4, 4, 3);
    conv("decoder.block.conv2", 4, 4, 3);
    for level in 0..3 {
        conv(&format!("decoder.up.{level}"), 4, 4, 3);
    }
    conv("decoder.conv_out", 3, 4, 3);
    for prefix in [
        "encoder.block.norm1",
        "encoder.block.norm2",
        "encoder.norm_out",
        "decoder.block.norm1",
        "decoder.block.norm2",
        "decoder.norm_out",
    ] {
        manifest.insert(format!("{prefix}.weight"), vec![SD15_REDUCED_WIDTH]);
        manifest.insert(format!("{prefix}.bias"), vec![SD15_REDUCED_WIDTH]);
    }
    manifest
}

#[cfg(test)]
fn center_crop_and_scale_sd15(
    backend: &CpuBackend,
    pixels: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    let shape = pixels.descriptor().shape();
    let cropped_height = shape[2] / 8 * 8;
    let cropped_width = shape[3] / 8 * 8;
    let offset_y = (shape[2] - cropped_height) / 2;
    let offset_x = (shape[3] - cropped_width) / 2;
    let source = tensor_to_f32(backend, pixels, context)?;
    let output_count = shape[0]
        .checked_mul(3)
        .and_then(|value| value.checked_mul(cropped_height))
        .and_then(|value| value.checked_mul(cropped_width))
        .ok_or(VaeError::ShapeOverflow)?;
    let mut output = backend.workspace_vec(
        context,
        usize::try_from(output_count).map_err(|_| VaeError::ShapeOverflow)?,
    )?;
    for batch in 0..shape[0] {
        for channel in 0..3 {
            for y in 0..cropped_height {
                check_periodically(y, context)?;
                for x in 0..cropped_width {
                    let source_index = usize::try_from(
                        ((batch * 3 + channel) * shape[2] + y + offset_y) * shape[3] + x + offset_x,
                    )
                    .map_err(|_| VaeError::ShapeOverflow)?;
                    output.try_push(source[source_index].mul_add(2.0, -1.0))?;
                }
            }
        }
    }
    Ok(tensor_from_f32(
        backend,
        &[shape[0], 3, cropped_height, cropped_width],
        &output,
        context,
    )?)
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
enum NativeVaeKernel {
    #[cfg(test)]
    BlockAverageNearest,
    #[cfg(test)]
    Sd15Reduced(Arc<Sd15LearnedVaeKernel>),
    Native {
        binding: VaeModelBinding,
        functions: VaeKernelFunctions,
    },
}

#[derive(Clone, Debug)]
pub struct NativeVae {
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    spatial_geometry: VaeSpatialGeometry,
    kernel: NativeVaeKernel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LtxLatentStatisticsDirection {
    Normalize,
    Unnormalize,
}

#[derive(Clone, Debug)]
pub struct NativeStructuredVae {
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    binding: VaeModelBinding,
    decode_raw: StructuredVaeKernelFunction,
}

fn native_vae_boundary_execution_requirements(
    dtype: DType,
    profile: &VaeKernelProfile,
) -> NativeExecutionRequirements {
    let mut requirements = NativeExecutionRequirements::new();
    requirements.append_tensor_io(dtype);
    for operation in [
        BinaryOperation::Add,
        BinaryOperation::Subtract,
        BinaryOperation::Multiply,
        BinaryOperation::Divide,
    ] {
        requirements.append_binary(operation, dtype);
    }
    if matches!(profile, VaeKernelProfile::BlockAverageNearestV1) {
        requirements.append_reduction(ReductionOperation::Mean, dtype);
        requirements.extend([
            OperationSupport::resize_input(ResizeMode::NearestExact, dtype, Layout::Contiguous),
            OperationSupport::resize_output(ResizeMode::NearestExact, dtype, Layout::Contiguous),
        ]);
    }
    requirements
}

fn native_vae_module_execution_requirements(
    binding: &VaeModelBinding,
    profile: &VaeKernelProfile,
    dtype: DType,
) -> NativeExecutionRequirements {
    let mut requirements = native_vae_boundary_execution_requirements(dtype, profile);
    requirements.extend(binding.module().execution_requirements(dtype).iter());
    requirements
}

impl NativeStructuredVae {
    pub fn admit_execution_target(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeModelExecutionAdmission, VaeError> {
        let identity = self.descriptor.identity();
        let requirements = native_vae_module_execution_requirements(
            &self.binding,
            identity.profile(),
            identity.dtype(),
        );
        Ok(admit_native_model_resource_execution(
            NativeModelExecutionRole::Vae,
            identity.architecture().as_str(),
            &self.execution_digest(),
            &requirements,
            || {
                let resident_bytes = self
                    .resident_bytes()
                    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
                let storage_bytes = self
                    .resident_storage_bytes()
                    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
                let structural_resident_bytes = resident_bytes
                    .checked_sub(storage_bytes)
                    .ok_or(NativeOpsError::ResidentBytesOverflow)?;
                let tensors = self
                    .binding
                    .module()
                    .keyed_state(context.cancellation)?
                    .into_values()
                    .collect();
                Ok((tensors, structural_resident_bytes))
            },
            backend,
            identity.device(),
            identity.dtype(),
            Layout::Contiguous,
            context.stream,
            context,
        )?)
    }

    pub(crate) fn checked_kernel(
        descriptor: VaeDescriptor,
        latent_definition: &'static LatentFormatDefinition,
        binding: VaeModelBinding,
        decode_raw: StructuredVaeKernelFunction,
    ) -> Result<Self, VaeError> {
        if descriptor.boundary().kind() != VaeBoundaryKind::StructuredOutput {
            return Err(VaeError::SelectionBoundaryMismatch {
                expected: VaeBoundaryKind::StructuredOutput,
                actual: descriptor.boundary().kind(),
            });
        }
        if descriptor.identity() != binding.identity()
            || descriptor.latent_format != LatentFormatDescriptor::checked(latent_definition)?
        {
            return Err(VaeError::KernelIdentityBindingMismatch);
        }
        validate_sha256(binding.digest())?;
        Ok(Self {
            descriptor,
            latent_definition,
            binding,
            decode_raw,
        })
    }

    pub fn descriptor(&self) -> &VaeDescriptor {
        &self.descriptor
    }

    pub fn keyed_state(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<BTreeMap<String, Tensor>, VaeError> {
        self.binding
            .module()
            .keyed_state(cancellation)
            .map_err(Into::into)
    }

    pub fn execution_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, self.descriptor.identity.digest.as_bytes());
        hash_field(&mut hasher, self.binding.digest().as_bytes());
        hash_field(&mut hasher, b"typed-structured-vae-v1");
        format!("{:x}", hasher.finalize())
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, VaeError> {
        self.binding
            .module()
            .resident_storage_bytes()
            .map_err(Into::into)
    }

    pub fn resident_bytes(&self) -> Result<u64, VaeError> {
        let object = u64::try_from(std::mem::size_of::<Self>()).map_err(|_| {
            VaeError::Allocation("structured VAE resident object byte overflow".to_owned())
        })?;
        let descriptor = self.descriptor.owned_resident_bytes().ok_or_else(|| {
            VaeError::Allocation("structured VAE descriptor byte overflow".to_owned())
        })?;
        let binding = self.binding.owned_resident_bytes().ok_or_else(|| {
            VaeError::Allocation("structured VAE binding byte overflow".to_owned())
        })?;
        let storage = self.resident_storage_bytes()?;
        object
            .checked_add(descriptor)
            .and_then(|bytes| bytes.checked_add(binding))
            .and_then(|bytes| bytes.checked_add(storage))
            .ok_or_else(|| {
                VaeError::Allocation("structured VAE resident total byte overflow".to_owned())
            })
    }

    pub fn validate(&self) -> Result<(), VaeError> {
        if self.descriptor.boundary().kind() != VaeBoundaryKind::StructuredOutput {
            return Err(VaeError::SelectionBoundaryMismatch {
                expected: VaeBoundaryKind::StructuredOutput,
                actual: self.descriptor.boundary().kind(),
            });
        }
        if self.descriptor.identity() != self.binding.identity()
            || self.descriptor.latent_format
                != LatentFormatDescriptor::checked(self.latent_definition)?
        {
            return Err(VaeError::KernelIdentityBindingMismatch);
        }
        if !self.binding.module().has_execution_state() {
            return Err(VaeError::NativeModuleHasNoState);
        }
        validate_sha256(self.binding.digest())?;
        validate_sha256(&self.execution_digest())?;
        self.resident_bytes()?;
        Ok(())
    }

    pub fn decode(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        request: &VaeStructuredDecodeRequest,
        context: &ExecutionContext<'_>,
    ) -> Result<VaeStructuredResult, VaeError> {
        self.admit_execution_target(backend, context)?;
        context.check()?;
        validate_native_vae_backend_binding(
            backend,
            self.descriptor.identity().dtype(),
            self.descriptor.identity().device(),
        )?;
        if latent.descriptor().dtype() != self.descriptor.identity().dtype() {
            return Err(VaeError::ExecutionDTypeMismatch {
                expected: self.descriptor.identity().dtype(),
                actual: latent.descriptor().dtype(),
            });
        }
        if latent.descriptor().device() != self.descriptor.identity().device() {
            return Err(VaeError::ExecutionDeviceMismatch {
                expected: self.descriptor.identity().device(),
                actual: latent.descriptor().device(),
            });
        }
        if latent.descriptor().stream() != context.stream {
            return Err(VaeError::KernelOutputContractMismatch {
                operation: VaeOperation::Decode,
                expected_shape: latent.descriptor().shape().to_vec(),
                actual_shape: latent.descriptor().shape().to_vec(),
                expected_dtype: latent.descriptor().dtype(),
                actual_dtype: latent.descriptor().dtype(),
                expected_device: latent.descriptor().device(),
                actual_device: latent.descriptor().device(),
                expected_stream: context.stream,
                actual_stream: latent.descriptor().stream(),
            });
        }
        let expected_kind = self
            .descriptor
            .boundary()
            .structured_kind()
            .ok_or(VaeError::StructuredDecodeRequired)?;
        let actual_kind = request.output_kind();
        if expected_kind != actual_kind {
            return Err(VaeError::StructuredKindMismatch {
                expected: expected_kind,
                actual: actual_kind,
            });
        }
        let expected_rank = usize::from(self.latent_definition.dimensions) + 2;
        let shape = latent.descriptor().shape();
        let channel_matches = match self.latent_definition.layout {
            LatentTensorLayout::ChannelsFirst => {
                shape.get(1) == Some(&self.latent_definition.channels)
            }
            LatentTensorLayout::SequenceChannelsLast => {
                shape.last() == Some(&self.latent_definition.channels)
            }
        };
        if shape.len() != expected_rank || shape.contains(&0) || !channel_matches {
            let mut expected = vec![0; expected_rank];
            match self.latent_definition.layout {
                LatentTensorLayout::ChannelsFirst => expected[1] = self.latent_definition.channels,
                LatentTensorLayout::SequenceChannelsLast => {
                    expected[expected_rank - 1] = self.latent_definition.channels
                }
            }
            return Err(VaeError::InvalidShape {
                expected,
                actual: shape.to_vec(),
            });
        }
        let raw = process_latent_out(self.latent_definition, backend, latent, context)?;
        let result = (self.decode_raw)(self.binding.module(), backend, &raw, request, context)?;
        if result.kind() != expected_kind {
            return Err(VaeError::StructuredKindMismatch {
                expected: expected_kind,
                actual: result.kind(),
            });
        }
        context.check()?;
        Ok(result)
    }
}

impl NativeVae {
    pub fn admit_execution_target(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeModelExecutionAdmission, VaeError> {
        let identity = self.descriptor.identity();
        let requirements = match &self.kernel {
            #[cfg(test)]
            NativeVaeKernel::BlockAverageNearest | NativeVaeKernel::Sd15Reduced(_) => {
                native_vae_boundary_execution_requirements(identity.dtype(), identity.profile())
            }
            NativeVaeKernel::Native { binding, .. } => native_vae_module_execution_requirements(
                binding,
                identity.profile(),
                identity.dtype(),
            ),
        };
        Ok(admit_native_model_resource_execution(
            NativeModelExecutionRole::Vae,
            identity.architecture().as_str(),
            &self.execution_digest(),
            &requirements,
            || {
                let resident_bytes = self
                    .resident_bytes()
                    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
                let storage_bytes = self
                    .resident_storage_bytes()
                    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?;
                let structural_resident_bytes = resident_bytes
                    .checked_sub(storage_bytes)
                    .ok_or(NativeOpsError::ResidentBytesOverflow)?;
                let tensors = match &self.kernel {
                    #[cfg(test)]
                    NativeVaeKernel::BlockAverageNearest | NativeVaeKernel::Sd15Reduced(_) => {
                        Vec::new()
                    }
                    NativeVaeKernel::Native { binding, .. } => binding
                        .module()
                        .keyed_state(context.cancellation)?
                        .into_values()
                        .collect(),
                };
                Ok((tensors, structural_resident_bytes))
            },
            backend,
            identity.device(),
            identity.dtype(),
            Layout::Contiguous,
            context.stream,
            context,
        )?)
    }

    #[cfg(test)]
    fn checked(
        descriptor: VaeDescriptor,
        latent_definition: &'static LatentFormatDefinition,
    ) -> Result<Self, VaeError> {
        if descriptor.identity.profile() != &VaeKernelProfile::BlockAverageNearestV1 {
            return Err(VaeError::KernelProfileMismatch);
        }
        if descriptor.latent_format != LatentFormatDescriptor::checked(latent_definition)? {
            return Err(VaeError::LatentFormatBindingMismatch);
        }
        validate_conformance_kernel_execution(&descriptor)?;
        let spatial_geometry = VaeSpatialGeometry::checked(
            descriptor.identity().profile(),
            descriptor.identity().loader_configuration(),
            latent_definition,
        )?;
        Ok(Self {
            descriptor,
            latent_definition,
            spatial_geometry,
            kernel: NativeVaeKernel::BlockAverageNearest,
        })
    }

    #[cfg(test)]
    fn checked_sd15(
        descriptor: VaeDescriptor,
        latent_definition: &'static LatentFormatDefinition,
        kernel: Sd15LearnedVaeKernel,
    ) -> Result<Self, VaeError> {
        if descriptor.identity.profile() != &VaeKernelProfile::Sd15AutoencoderKlReducedV1 {
            return Err(VaeError::KernelProfileMismatch);
        }
        if latent_definition.dimensions != 2
            || latent_definition.channels != SD15_REDUCED_LATENT_CHANNELS
            || latent_definition.layout != LatentTensorLayout::ChannelsFirst
            || latent_definition.spatial_downscale_ratio != 8
            || descriptor.pixel_channels() != 3
            || descriptor.decode_clamp != [0.0, 1.0]
        {
            return Err(VaeError::LearnedArchitectureLatentMismatch);
        }
        if descriptor.identity() != kernel.identity() {
            return Err(VaeError::KernelIdentityBindingMismatch);
        }
        validate_sha256(kernel.state_digest())?;
        if descriptor.latent_format != LatentFormatDescriptor::checked(latent_definition)? {
            return Err(VaeError::LatentFormatBindingMismatch);
        }
        let spatial_geometry = VaeSpatialGeometry::checked(
            descriptor.identity().profile(),
            descriptor.identity().loader_configuration(),
            latent_definition,
        )?;
        Ok(Self {
            descriptor,
            latent_definition,
            spatial_geometry,
            kernel: NativeVaeKernel::Sd15Reduced(Arc::new(kernel)),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn checked_kernel(
        descriptor: VaeDescriptor,
        latent_definition: &'static LatentFormatDefinition,
        binding: VaeModelBinding,
        functions: VaeKernelFunctions,
    ) -> Result<Self, VaeError> {
        validate_generic_vae_boundary(descriptor.boundary())?;
        if descriptor.identity() != binding.identity()
            || descriptor.identity().architecture() != &functions.architecture
        {
            return Err(VaeError::KernelIdentityBindingMismatch);
        }
        validate_sha256(binding.digest())?;
        if descriptor.latent_format != LatentFormatDescriptor::checked(latent_definition)? {
            return Err(VaeError::LatentFormatBindingMismatch);
        }
        let spatial_geometry = VaeSpatialGeometry::checked(
            descriptor.identity().profile(),
            descriptor.identity().loader_configuration(),
            latent_definition,
        )?;
        Ok(Self {
            descriptor,
            latent_definition,
            spatial_geometry,
            kernel: NativeVaeKernel::Native { binding, functions },
        })
    }

    pub fn descriptor(&self) -> &VaeDescriptor {
        &self.descriptor
    }

    pub fn keyed_state(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<BTreeMap<String, Tensor>, VaeError> {
        match &self.kernel {
            NativeVaeKernel::Native { binding, .. } => binding
                .module()
                .keyed_state(cancellation)
                .map_err(Into::into),
            #[cfg(test)]
            NativeVaeKernel::Sd15Reduced(kernel) => Ok(kernel.weights.clone()),
            #[cfg(test)]
            NativeVaeKernel::BlockAverageNearest => Err(VaeError::NativeModuleHasNoState),
        }
    }

    pub fn apply_ltx_latent_statistics(
        &self,
        backend: &dyn TensorBackend,
        input: &Tensor,
        direction: LtxLatentStatisticsDirection,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        context.check()?;
        if !matches!(
            self.descriptor.identity().profile(),
            VaeKernelProfile::LtxVideoV0 { .. }
                | VaeKernelProfile::LtxVideoV1 { .. }
                | VaeKernelProfile::LtxVideoV2 { .. }
        ) {
            return Err(VaeError::KernelProfileMismatch);
        }
        if !matches!(input.descriptor().shape(), [batch, 128, frames, height, width]
            if *batch > 0 && *frames > 0 && *height > 0 && *width > 0)
        {
            return Err(VaeError::InvalidShape {
                expected: vec![1, 128, 1, 1, 1],
                actual: input.descriptor().shape().to_vec(),
            });
        }
        if input.descriptor().dtype() != DType::F32 {
            return Err(VaeError::UnsupportedDType(input.descriptor().dtype()));
        }
        if input.descriptor().device() != DeviceId::CPU {
            return Err(VaeError::ExecutionDeviceMismatch {
                expected: DeviceId::CPU,
                actual: input.descriptor().device(),
            });
        }
        if input.descriptor().stream() != context.stream || !input.descriptor().is_contiguous()? {
            return Err(VaeError::Tensor(TensorError::NonContiguousAccess));
        }
        let binding = match &self.kernel {
            NativeVaeKernel::Native { binding, .. } => binding,
            #[cfg(test)]
            NativeVaeKernel::BlockAverageNearest | NativeVaeKernel::Sd15Reduced(_) => {
                return Err(VaeError::KernelProfileMismatch);
            }
        };
        crate::vae_video::ltx_latent_statistics(
            binding.module(),
            backend,
            input,
            matches!(direction, LtxLatentStatisticsDirection::Normalize),
            context,
        )
    }

    pub fn execution_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, self.descriptor.identity.digest.as_bytes());
        match &self.kernel {
            #[cfg(test)]
            NativeVaeKernel::BlockAverageNearest => {
                hash_field(&mut hasher, b"block-average-nearest-v1")
            }
            #[cfg(test)]
            NativeVaeKernel::Sd15Reduced(kernel) => {
                hash_field(&mut hasher, kernel.state_digest().as_bytes())
            }
            NativeVaeKernel::Native { binding, .. } => {
                hash_field(&mut hasher, binding.digest().as_bytes())
            }
        }
        format!("{:x}", hasher.finalize())
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, VaeError> {
        match &self.kernel {
            #[cfg(test)]
            NativeVaeKernel::BlockAverageNearest => Ok(0),
            #[cfg(test)]
            NativeVaeKernel::Sd15Reduced(kernel) => kernel.resident_storage_bytes(),
            NativeVaeKernel::Native { binding, .. } => binding
                .module()
                .resident_storage_bytes()
                .map_err(Into::into),
        }
    }

    pub fn resident_bytes(&self) -> Result<u64, VaeError> {
        let object = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| VaeError::Allocation("VAE resident object byte overflow".to_owned()))?;
        let descriptor = self.descriptor.owned_resident_bytes().ok_or_else(|| {
            VaeError::Allocation("VAE descriptor resident byte overflow".to_owned())
        })?;
        let geometry = self
            .spatial_geometry
            .owned_resident_bytes()
            .ok_or_else(|| {
                VaeError::Allocation("VAE geometry resident byte overflow".to_owned())
            })?;
        let kernel = match &self.kernel {
            #[cfg(test)]
            NativeVaeKernel::BlockAverageNearest => 0,
            #[cfg(test)]
            NativeVaeKernel::Sd15Reduced(kernel) => {
                kernel.owned_resident_bytes().ok_or_else(|| {
                    VaeError::Allocation("VAE reduced kernel resident byte overflow".to_owned())
                })?
            }
            NativeVaeKernel::Native { binding, functions } => binding
                .owned_resident_bytes()
                .and_then(|bytes| bytes.checked_add(functions.owned_resident_bytes()?))
                .ok_or_else(|| {
                    VaeError::Allocation("VAE native kernel resident byte overflow".to_owned())
                })?,
        };
        let storage = self.resident_storage_bytes()?;
        object
            .checked_add(descriptor)
            .and_then(|bytes| bytes.checked_add(geometry))
            .and_then(|bytes| bytes.checked_add(kernel))
            .and_then(|bytes| bytes.checked_add(storage))
            .ok_or_else(|| VaeError::Allocation("VAE resident total byte overflow".to_owned()))
    }

    pub fn encode<B>(
        &self,
        backend: &B,
        pixels: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError>
    where
        B: TensorBackend + Any,
    {
        let plan = self.plan_encode(pixels)?;
        self.encode_tiled(backend, pixels, &plan, context)
    }

    pub fn plan_encode(&self, pixels: &Tensor) -> Result<VaeTilePlan, VaeError> {
        let input_spatial = if extra_1d_channels(self.descriptor.identity().profile()).is_some() {
            vec![
                *pixels
                    .descriptor()
                    .shape()
                    .get(2)
                    .ok_or(VaeError::ShapeOverflow)?,
            ]
        } else {
            encode_spatial_geometry(
                self.spatial_geometry.formulas(VaeOperation::Encode),
                pixels.descriptor().shape(),
            )?
            .0
        };
        self.make_encode_plan(
            pixels,
            input_spatial.clone(),
            vec![0; input_spatial.len()],
            false,
        )
    }

    pub fn decode<B>(
        &self,
        backend: &B,
        latent: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError>
    where
        B: TensorBackend + Any,
    {
        let raw_shape =
            processed_latent_shape(self.latent_definition, latent.descriptor().shape())?;
        let spatial =
            if let Some(extra_channels) = extra_1d_channels(self.descriptor.identity().profile()) {
                vec![extra_1d_time_extent(
                    self.descriptor.identity().profile(),
                    &raw_shape,
                    self.latent_definition.channels,
                    extra_channels,
                )?]
            } else {
                latent_spatial_shape(self.latent_definition.layout, &raw_shape)?.to_vec()
            };
        let plan = self.make_decode_plan(latent, spatial.clone(), vec![0; spatial.len()], false)?;
        self.decode_tiled(backend, latent, &plan, context)
    }

    #[cfg(test)]
    fn encode_sd15_conformance(
        &self,
        backend: &CpuBackend,
        pixels: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        validate_execution(self, backend, pixels, context)?;
        validate_pixel_input(self, pixels)?;
        match &self.kernel {
            NativeVaeKernel::Sd15Reduced(kernel) => {
                kernel.encode(backend, pixels, self.latent_definition, context)
            }
            _ => Err(VaeError::KernelProfileMismatch),
        }
    }

    #[cfg(test)]
    fn decode_sd15_conformance(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        validate_execution(self, backend, latent, context)?;
        validate_latent_input(self, latent)?;
        match &self.kernel {
            NativeVaeKernel::Sd15Reduced(kernel) => {
                kernel.decode(backend, latent, self.latent_definition, context)
            }
            _ => Err(VaeError::KernelProfileMismatch),
        }
    }

    pub fn plan_encode_tiles(
        &self,
        pixels: &Tensor,
        tile_extent: Vec<u64>,
        overlap: Vec<u64>,
    ) -> Result<VaeTilePlan, VaeError> {
        self.make_encode_plan(pixels, tile_extent, overlap, true)
    }

    pub fn plan_decode_tiles(
        &self,
        latent: &Tensor,
        tile_extent: Vec<u64>,
        overlap: Vec<u64>,
    ) -> Result<VaeTilePlan, VaeError> {
        self.make_decode_plan(latent, tile_extent, overlap, true)
    }

    fn make_encode_plan(
        &self,
        pixels: &Tensor,
        tile_extent: Vec<u64>,
        overlap: Vec<u64>,
        source_tiled: bool,
    ) -> Result<VaeTilePlan, VaeError> {
        validate_pixel_input(self, pixels)?;
        let extra_channels = extra_1d_channels(self.descriptor.identity().profile());
        let (input_spatial, input_offsets, output_spatial, formulas) = if extra_channels.is_some() {
            let formula = extra_1d_formula(self.descriptor.identity().profile())
                .ok_or(VaeError::ShapeOverflow)??;
            let original = pixels.descriptor().shape()[2];
            let effective = match formula {
                VaeTileAxisFormula::Linear { ratio } => original / ratio * ratio,
                VaeTileAxisFormula::Causal { .. } | VaeTileAxisFormula::ResampledCausal { .. } => {
                    original
                }
            };
            if effective == 0 {
                return Err(VaeError::InvalidShape {
                    expected: vec![0, self.descriptor.pixel_channels(), 1],
                    actual: pixels.descriptor().shape().to_vec(),
                });
            }
            (
                vec![effective],
                vec![(original - effective) / 2],
                vec![formula.output_extent(VaeOperation::Encode, effective)?],
                vec![formula],
            )
        } else {
            let formulas = self.spatial_geometry.formulas(VaeOperation::Encode);
            let (input, offsets, output) =
                encode_spatial_geometry(formulas, pixels.descriptor().shape())?;
            (input, offsets, output, formulas.to_vec())
        };
        let raw_shape = if let Some(extra_channels) = extra_channels {
            extra_1d_raw_shape(
                self.descriptor.identity().profile(),
                pixels.descriptor().shape()[0],
                self.latent_definition.channels,
                extra_channels,
                output_spatial[0],
            )
        } else {
            shape_for_latent_layout(
                pixels.descriptor().shape()[0],
                self.latent_definition.channels,
                &output_spatial,
                self.latent_definition.layout,
            )
        };
        let execution_output_shape = if let Some(extra_channels) = extra_channels {
            vec![
                pixels.descriptor().shape()[0],
                self.latent_definition
                    .channels
                    .checked_mul(extra_channels)
                    .ok_or(VaeError::ShapeOverflow)?,
                output_spatial[0],
            ]
        } else {
            raw_shape.clone()
        };
        let tiling = TileExecutionPlan::checked(
            VaeOperation::Encode,
            input_spatial,
            input_offsets,
            output_spatial,
            formulas,
            TileTensorLayout::ChannelsFirst,
            if extra_channels.is_some() {
                TileTensorLayout::ChannelsFirst
            } else {
                tile_tensor_layout(self.latent_definition.layout)
            },
            tile_extent,
            overlap,
            source_tiled && self.latent_definition.dimensions == 2 && extra_channels.is_none(),
            false,
        )?;
        VaeTilePlan::checked(
            &self.execution_digest(),
            VaeOperation::Encode,
            pixels.descriptor().shape().to_vec(),
            raw_shape,
            execution_output_shape,
            extra_channels,
            tiling,
        )
    }

    fn make_decode_plan(
        &self,
        latent: &Tensor,
        tile_extent: Vec<u64>,
        overlap: Vec<u64>,
        source_tiled: bool,
    ) -> Result<VaeTilePlan, VaeError> {
        validate_latent_input(self, latent)?;
        let raw_shape =
            processed_latent_shape(self.latent_definition, latent.descriptor().shape())?;
        let extra_channels = extra_1d_channels(self.descriptor.identity().profile());
        let (input_spatial, formulas) = if let Some(extra_channels) = extra_channels {
            let time = extra_1d_time_extent(
                self.descriptor.identity().profile(),
                &raw_shape,
                self.latent_definition.channels,
                extra_channels,
            )?;
            (
                vec![time],
                vec![
                    extra_1d_formula(self.descriptor.identity().profile())
                        .ok_or(VaeError::ShapeOverflow)??,
                ],
            )
        } else {
            (
                latent_spatial_shape(self.latent_definition.layout, &raw_shape)?.to_vec(),
                self.spatial_geometry
                    .formulas(VaeOperation::Decode)
                    .to_vec(),
            )
        };
        let output_spatial = formulas
            .iter()
            .zip(&input_spatial)
            .map(|(formula, extent)| formula.output_extent(VaeOperation::Decode, *extent))
            .collect::<Result<Vec<_>, _>>()?;
        let output_shape = shape_for_pixel_layout(
            raw_shape[0],
            self.descriptor.pixel_channels(),
            &output_spatial,
        )?;
        let tiling = TileExecutionPlan::checked(
            VaeOperation::Decode,
            input_spatial,
            vec![0; output_spatial.len()],
            output_spatial,
            formulas,
            if extra_channels.is_some() {
                TileTensorLayout::ChannelsFirst
            } else {
                tile_tensor_layout(self.latent_definition.layout)
            },
            TileTensorLayout::ChannelsFirst,
            tile_extent,
            overlap,
            source_tiled && self.latent_definition.dimensions == 2 && extra_channels.is_none(),
            preserve_tiled_batch_group(self.descriptor.identity().profile(), VaeOperation::Decode),
        )?;
        VaeTilePlan::checked(
            &self.execution_digest(),
            VaeOperation::Decode,
            latent.descriptor().shape().to_vec(),
            output_shape.clone(),
            output_shape,
            extra_channels,
            tiling,
        )
    }

    pub fn encode_tiled<B>(
        &self,
        backend: &B,
        pixels: &Tensor,
        plan: &VaeTilePlan,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError>
    where
        B: TensorBackend + Any,
    {
        validate_execution(self, backend, pixels, context)?;
        self.validate_plan(plan, VaeOperation::Encode, pixels)?;
        let cpu_backend = (backend as &dyn Any).downcast_ref::<CpuBackend>();
        let execution_channels = match plan.extra_1d_channels {
            Some(extra_channels) => self
                .latent_definition
                .channels
                .checked_mul(extra_channels)
                .ok_or(VaeError::ShapeOverflow)?,
            None => self.latent_definition.channels,
        };
        let raw_execution = match &self.kernel {
            NativeVaeKernel::Native { binding, functions } => execute_tiled_scale(
                backend,
                pixels,
                &plan.execution_output_shape,
                execution_channels,
                &plan.tiling,
                |tile, expected_spatial, context| {
                    let tile = prepare_pixel_channels(
                        backend,
                        tile,
                        self.descriptor.pixel_channels(),
                        self.descriptor.identity().profile(),
                        context,
                    )?;
                    let encoded = functions.encode(
                        binding.module(),
                        backend,
                        cpu_backend,
                        &tile,
                        self.latent_definition,
                        context,
                    )?;
                    let expected = if let Some(extra_channels) = plan.extra_1d_channels {
                        extra_1d_raw_shape(
                            self.descriptor.identity().profile(),
                            1,
                            self.latent_definition.channels,
                            extra_channels,
                            expected_spatial[0],
                        )
                    } else {
                        shape_for_latent_layout(
                            1,
                            self.latent_definition.channels,
                            expected_spatial,
                            self.latent_definition.layout,
                        )
                    };
                    validate_kernel_output(
                        VaeOperation::Encode,
                        &encoded,
                        &expected,
                        self.descriptor.identity(),
                        context.stream,
                    )?;
                    if plan.extra_1d_channels.is_some() {
                        flatten_extra_1d_latent(
                            backend,
                            &encoded,
                            self.descriptor.identity().profile(),
                            context,
                        )
                    } else {
                        Ok(encoded)
                    }
                },
                context,
            )?,
            #[cfg(test)]
            NativeVaeKernel::Sd15Reduced(_) => {
                return Err(VaeError::ConformanceHarnessRequiresCpuEntryPoint);
            }
            #[cfg(test)]
            NativeVaeKernel::BlockAverageNearest => execute_tiled_scale(
                backend,
                pixels,
                &plan.execution_output_shape,
                execution_channels,
                &plan.tiling,
                |tile, expected_spatial, context| {
                    let tile = prepare_pixel_channels(
                        backend,
                        tile,
                        self.descriptor.pixel_channels(),
                        self.descriptor.identity().profile(),
                        context,
                    )?;
                    self.encode_block_average_tile(backend, &tile, expected_spatial, context)
                },
                context,
            )?,
        };
        let raw = if plan.extra_1d_channels.is_some() {
            unflatten_extra_1d_latent(
                backend,
                &raw_execution,
                self.descriptor.identity().profile(),
                self.latent_definition.channels,
                context,
            )?
        } else {
            raw_execution
        };
        validate_kernel_output(
            VaeOperation::Encode,
            &raw,
            &plan.output_shape,
            self.descriptor.identity(),
            context.stream,
        )?;
        let processed = process_latent_in(self.latent_definition, backend, &raw, context)?;
        let expected_processed_shape =
            processed_encode_shape(self.latent_definition, &plan.output_shape)?;
        validate_kernel_output(
            VaeOperation::Encode,
            &processed,
            &expected_processed_shape,
            self.descriptor.identity(),
            context.stream,
        )?;
        validate_latent_input(self, &processed)?;
        Ok(processed)
    }

    pub fn decode_tiled<B>(
        &self,
        backend: &B,
        latent: &Tensor,
        plan: &VaeTilePlan,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError>
    where
        B: TensorBackend + Any,
    {
        validate_execution(self, backend, latent, context)?;
        self.validate_plan(plan, VaeOperation::Decode, latent)?;
        let cpu_backend = (backend as &dyn Any).downcast_ref::<CpuBackend>();
        let raw = process_latent_out(self.latent_definition, backend, latent, context)?;
        let expected_raw_shape =
            processed_latent_shape(self.latent_definition, latent.descriptor().shape())?;
        validate_kernel_output(
            VaeOperation::Decode,
            &raw,
            &expected_raw_shape,
            self.descriptor.identity(),
            context.stream,
        )?;
        let execution_input = if plan.extra_1d_channels.is_some() {
            flatten_extra_1d_latent(backend, &raw, self.descriptor.identity().profile(), context)?
        } else {
            raw
        };
        let output = match &self.kernel {
            NativeVaeKernel::Native { binding, functions } => execute_tiled_scale(
                backend,
                &execution_input,
                &plan.execution_output_shape,
                self.descriptor.pixel_channels(),
                &plan.tiling,
                |tile, expected_spatial, context| {
                    let tile_batch = tile.descriptor().shape()[0];
                    let tile = if plan.extra_1d_channels.is_some() {
                        unflatten_extra_1d_latent(
                            backend,
                            tile,
                            self.descriptor.identity().profile(),
                            self.latent_definition.channels,
                            context,
                        )?
                    } else {
                        tile.clone()
                    };
                    let decoded = functions.decode(
                        binding.module(),
                        backend,
                        cpu_backend,
                        &tile,
                        self.latent_definition,
                        context,
                    )?;
                    let expected = shape_for_pixel_layout(
                        tile_batch,
                        self.descriptor.pixel_channels(),
                        expected_spatial,
                    )?;
                    validate_kernel_output(
                        VaeOperation::Decode,
                        &decoded,
                        &expected,
                        self.descriptor.identity(),
                        context.stream,
                    )?;
                    Ok(decoded)
                },
                context,
            )?,
            #[cfg(test)]
            NativeVaeKernel::Sd15Reduced(_) => {
                return Err(VaeError::ConformanceHarnessRequiresCpuEntryPoint);
            }
            #[cfg(test)]
            NativeVaeKernel::BlockAverageNearest => execute_tiled_scale(
                backend,
                &execution_input,
                &plan.execution_output_shape,
                self.descriptor.pixel_channels(),
                &plan.tiling,
                |tile, expected_spatial, context| {
                    self.decode_block_average_tile(backend, tile, expected_spatial, context)
                },
                context,
            )?,
        };
        validate_kernel_output(
            VaeOperation::Decode,
            &output,
            &plan.output_shape,
            self.descriptor.identity(),
            context.stream,
        )?;
        if self.descriptor.identity().profile() == &VaeKernelProfile::StableCascadeStageAV1 {
            return Ok(output);
        }
        let clamped =
            clamp_decode_output(backend, &output, self.descriptor.decode_clamp(), context)?;
        validate_kernel_output(
            VaeOperation::Decode,
            &clamped,
            &plan.output_shape,
            self.descriptor.identity(),
            context.stream,
        )?;
        Ok(clamped)
    }

    #[cfg(test)]
    fn encode_block_average_tile(
        &self,
        backend: &dyn TensorBackend,
        pixels: &Tensor,
        output_spatial: &[u64],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let output_shape = shape_for_latent_layout(
            1,
            self.latent_definition.channels,
            output_spatial,
            self.latent_definition.layout,
        );
        let descriptor = TensorDescriptor::contiguous(
            output_shape,
            pixels.descriptor().dtype(),
            pixels.descriptor().device(),
            pixels.descriptor().stream(),
        )?;
        let (mut output, event) = backend.allocate(descriptor, context)?;
        backend.wait_event(event, context)?;
        let spatial_count = checked_product(output_spatial.iter().copied())?;
        {
            let mut write = output.write()?;
            for channel in 0..self.latent_definition.channels {
                for linear in 0..spatial_count {
                    check_periodically(linear, context)?;
                    let spatial = coordinates_for_shape(linear, output_spatial)?;
                    let value = encode_value(self, pixels, 0, channel, &spatial, context)?;
                    let indices =
                        latent_indices(self.latent_definition.layout, 0, channel, &spatial);
                    write_f32(&mut write, &indices, value)?;
                }
            }
        }
        Ok(output)
    }

    #[cfg(test)]
    fn decode_block_average_tile(
        &self,
        backend: &dyn TensorBackend,
        latent: &Tensor,
        output_spatial: &[u64],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let output_shape =
            shape_for_pixel_layout(1, self.descriptor.pixel_channels(), output_spatial)?;
        let descriptor = TensorDescriptor::contiguous(
            output_shape,
            latent.descriptor().dtype(),
            latent.descriptor().device(),
            latent.descriptor().stream(),
        )?;
        let (mut output, event) = backend.allocate(descriptor, context)?;
        backend.wait_event(event, context)?;
        let spatial_count = checked_product(output_spatial.iter().copied())?;
        {
            let mut write = output.write()?;
            for pixel_channel in 0..self.descriptor.pixel_channels() {
                for linear in 0..spatial_count {
                    check_periodically(linear, context)?;
                    let pixel_spatial = coordinates_for_shape(linear, output_spatial)?;
                    let latent_spatial = pixel_spatial
                        .iter()
                        .enumerate()
                        .map(|(dimension, coordinate)| {
                            coordinate / compression_ratio(self.latent_definition, dimension)
                        })
                        .collect::<Vec<_>>();
                    let source = latent_indices(
                        self.latent_definition.layout,
                        0,
                        pixel_channel % self.latent_definition.channels,
                        &latent_spatial,
                    );
                    let value = read_f32(latent, &source)?;
                    let mut destination = Vec::with_capacity(pixel_spatial.len() + 2);
                    destination.push(0);
                    destination.push(pixel_channel);
                    destination.extend(pixel_spatial);
                    write_f32(&mut write, &destination, value)?;
                }
            }
        }
        Ok(output)
    }

    fn validate_plan(
        &self,
        plan: &VaeTilePlan,
        operation: VaeOperation,
        input: &Tensor,
    ) -> Result<(), VaeError> {
        if plan.vae_digest != self.execution_digest() {
            return Err(VaeError::TilePlanIdentityMismatch);
        }
        if plan.operation != operation {
            return Err(VaeError::TilePlanOperationMismatch {
                expected: operation,
                actual: plan.operation,
            });
        }
        if plan.input_shape != input.descriptor().shape() {
            return Err(VaeError::TilePlanShapeMismatch {
                expected: plan.input_shape.clone(),
                actual: input.descriptor().shape().to_vec(),
            });
        }
        Ok(())
    }
}

fn validate_generic_vae_boundary(boundary: &VaeBoundary) -> Result<(), VaeError> {
    if boundary.kind() == VaeBoundaryKind::StructuredOutput {
        Err(VaeError::StructuredDecodeRequired)
    } else {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum VaeError {
    #[error("unsupported VAE schema version {0}")]
    SchemaVersion(u16),
    #[error("VAE artifact is not present")]
    ArtifactUnavailable,
    #[error("VAE artifact relative path is not UTF-8")]
    NonUtf8ArtifactPath,
    #[error("invalid VAE {field}: {value}")]
    InvalidIdentity { field: &'static str, value: String },
    #[error("unregistered VAE architecture {0}")]
    UnregisteredArchitecture(String),
    #[error("invalid VAE SHA-256 digest")]
    InvalidSha256,
    #[error("unsupported VAE patch-graph schema version {0}")]
    PatchSchemaVersion(u16),
    #[error("VAE patch graph belongs to a different base artifact")]
    PatchArtifactMismatch,
    #[error("invalid fields for VAE {0:?} boundary")]
    InvalidBoundary(VaeBoundaryKind),
    #[error("VAE selector requires {expected:?} boundary, got {actual:?}")]
    SelectionBoundaryMismatch {
        expected: VaeBoundaryKind,
        actual: VaeBoundaryKind,
    },
    #[error("VAE {kind:?} boundary does not accept a {dimensions}D latent format")]
    BoundaryLatentDimensionMismatch {
        kind: VaeBoundaryKind,
        dimensions: u8,
    },
    #[error("VAE identity digest does not match its bound fields")]
    IdentityDigestMismatch,
    #[error("VAE identity serialization failed: {0}")]
    IdentitySerialization(String),
    #[error("VAE pixel channel count must be positive, got {0}")]
    InvalidPixelChannels(u64),
    #[error("VAE decode clamp must contain finite ascending bounds, got {0:?}")]
    InvalidClamp([f32; 2]),
    #[error("VAE descriptor is bound to a different latent-format definition")]
    LatentFormatBindingMismatch,
    #[error("VAE descriptor selected a kernel profile incompatible with its constructor")]
    KernelProfileMismatch,
    #[error("VAE kernel identity does not exactly match the checked descriptor identity")]
    KernelIdentityBindingMismatch,
    #[error("the loaded model does not contain the VAE descriptor artifact")]
    ModelArtifactMismatch,
    #[error("loaded VAE architecture mismatch: expected {expected}, got {actual}")]
    ModelArchitectureMismatch { expected: String, actual: String },
    #[error("the canonical native VAE module contains no loaded state")]
    NativeModuleHasNoState,
    #[cfg(test)]
    #[error("the reduced conformance harness requires its test-only CPU entry point")]
    ConformanceHarnessRequiresCpuEntryPoint,
    #[cfg(test)]
    #[error(
        "VAE kernel expected {expected_device:?} {expected_dtype:?}, got {actual_device:?} {actual_dtype:?}"
    )]
    KernelExecutionBindingMismatch {
        expected_dtype: DType,
        actual_dtype: DType,
        expected_device: DeviceId,
        actual_device: DeviceId,
    },
    #[cfg(test)]
    #[error("the learned SD15 VAE requires a two-dimensional four-channel 8x latent format")]
    LearnedArchitectureLatentMismatch,
    #[cfg(test)]
    #[error("learned VAE weight set differs; missing={missing:?}, unexpected={unexpected:?}")]
    LearnedWeightKeys {
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    #[cfg(test)]
    #[error("learned VAE is missing weight {0}")]
    MissingLearnedWeight(String),
    #[cfg(test)]
    #[error(
        "learned VAE weight {key} expected CPU F32 {expected:?}, got {device:?} {dtype:?} {actual:?}"
    )]
    LearnedWeightShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        dtype: DType,
        device: DeviceKind,
    },
    #[error("VAE shape overflow")]
    ShapeOverflow,
    #[error(
        "VAE profile {profile} has image-specific operation geometry but latent rank is {dimensions}"
    )]
    InvalidOperationGeometry { profile: String, dimensions: u8 },
    #[error("VAE expected input shape {expected:?}, got {actual:?}")]
    InvalidShape {
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("VAE supports F32 tensors, got {0:?}")]
    UnsupportedDType(DType),
    #[error("VAE latent transform {transform:?} is unavailable for execution dtype {dtype:?}")]
    LatentTransformDTypeMismatch {
        transform: LatentTransform,
        dtype: DType,
    },
    #[error("VAE execution expected dtype {expected:?}, got {actual:?}")]
    ExecutionDTypeMismatch { expected: DType, actual: DType },
    #[error("VAE execution expected device {expected:?}, got {actual:?}")]
    ExecutionDeviceMismatch {
        expected: DeviceId,
        actual: DeviceId,
    },
    #[error("Stable Cascade Stage C VAE feature execution requires the canonical CpuBackend")]
    StageCRequiresCpuBackend,
    #[error("native image VAE execution requires the canonical CpuBackend")]
    ImageVaeRequiresCpuBackend,
    #[error("native audio preprocessing or vocoder execution requires the canonical CpuBackend")]
    AudioVaeRequiresCpuBackend,
    #[error("structured VAE execution requires a typed structured decode request")]
    StructuredDecodeRequired,
    #[error("invalid structured VAE request: {0}")]
    InvalidStructuredRequest(String),
    #[error("structured VAE request/output kind mismatch: expected {expected:?}, got {actual:?}")]
    StructuredKindMismatch {
        expected: VaeStructuredOutputKind,
        actual: VaeStructuredOutputKind,
    },
    #[error("invalid structured VAE result: {0}")]
    InvalidStructuredResult(String),
    #[error("VAE profile {profile} does not implement {operation:?}")]
    OperationUnavailable {
        profile: String,
        operation: VaeOperation,
    },
    #[error(
        "VAE {operation:?} kernel output contract mismatch: expected shape {expected_shape:?}, {expected_dtype:?}, {expected_device:?}, stream {expected_stream:?}; got shape {actual_shape:?}, {actual_dtype:?}, {actual_device:?}, stream {actual_stream:?}"
    )]
    KernelOutputContractMismatch {
        operation: VaeOperation,
        expected_shape: Vec<u64>,
        actual_shape: Vec<u64>,
        expected_dtype: DType,
        actual_dtype: DType,
        expected_device: DeviceId,
        actual_device: DeviceId,
        expected_stream: StreamId,
        actual_stream: StreamId,
    },
    #[error(
        "VAE tile rank mismatch: expected {expected}, tile extent {tile_extent}, overlap {overlap}"
    )]
    TileRank {
        expected: usize,
        tile_extent: usize,
        overlap: usize,
    },
    #[error(
        "invalid VAE tile dimension {dimension}: size={size}, extent={extent}, overlap={overlap}"
    )]
    InvalidTileDimension {
        dimension: usize,
        size: u64,
        extent: u64,
        overlap: u64,
    },
    #[error("VAE tile plan contains too many tiles: {0}")]
    TooManyTiles(u64),
    #[error("VAE tile scale ratio must be positive, got {ratio}")]
    InvalidTileScale { ratio: u64 },
    #[error("VAE tile scaling produced an empty output")]
    ZeroTileOutput,
    #[error("source-exact three-pass VAE tiling requires two spatial dimensions, got {0}")]
    ThreePassTileRank(usize),
    #[error("phase-sensitive resampling and centered analysis require one whole-input VAE tile")]
    PhaseSensitiveTileRequiresWholeInput,
    #[error("VAE tile plan has no execution passes")]
    NoTilePasses,
    #[error("VAE tile output geometry mismatch: expected {expected:?}, got {actual:?}")]
    TileOutputGeometryMismatch {
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("VAE tile output starts outside dimension {dimension}: start={start}, size={size}")]
    TileOutputOutOfBounds {
        dimension: usize,
        start: u64,
        size: u64,
    },
    #[error("VAE tile kernel output mismatch: expected {expected_shape:?}, got {actual_shape:?}")]
    TileKernelOutputMismatch {
        expected_shape: Vec<u64>,
        actual_shape: Vec<u64>,
    },
    #[error("VAE tile plan belongs to a different VAE")]
    TilePlanIdentityMismatch,
    #[error("VAE tile plan operation mismatch: expected {expected:?}, got {actual:?}")]
    TilePlanOperationMismatch {
        expected: VaeOperation,
        actual: VaeOperation,
    },
    #[error("VAE tile plan input shape mismatch: expected {expected:?}, got {actual:?}")]
    TilePlanShapeMismatch {
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("VAE temporary allocation failed: {0}")]
    Allocation(String),
    #[error(transparent)]
    LatentFormat(#[from] LatentFormatError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    NativeTensor(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    ModelStore(#[from] ModelStoreError),
    #[error(transparent)]
    NativeOps(#[from] NativeOpsError),
    #[error(transparent)]
    Architecture(#[from] VaeArchitectureError),
}

pub fn validate_native_vae_backend_target(
    backend: &CpuBackend,
    target: &VaeExecutionTarget,
) -> Result<(), VaeError> {
    validate_native_vae_backend_binding(backend, target.dtype(), target.device())
}

pub(crate) fn validate_native_vae_backend_binding(
    backend: &CpuBackend,
    dtype: DType,
    device: DeviceId,
) -> Result<(), VaeError> {
    if backend.device() != device {
        return Err(VaeError::ExecutionDeviceMismatch {
            expected: device,
            actual: backend.device(),
        });
    }
    if dtype != DType::F32 {
        return Err(VaeError::UnsupportedDType(dtype));
    }
    Ok(())
}

fn validate_execution(
    vae: &NativeVae,
    backend: &dyn TensorBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(), VaeError> {
    vae.admit_execution_target(backend, context)?;
    context.check()?;
    if input.descriptor().dtype() != vae.descriptor.identity().dtype() {
        return Err(VaeError::ExecutionDTypeMismatch {
            expected: vae.descriptor.identity().dtype(),
            actual: input.descriptor().dtype(),
        });
    }
    if input.descriptor().device() != vae.descriptor.identity().device() {
        return Err(VaeError::ExecutionDeviceMismatch {
            expected: vae.descriptor.identity().device(),
            actual: input.descriptor().device(),
        });
    }
    if backend.device() != input.descriptor().device() {
        return Err(TensorError::DeviceMismatch {
            expected: backend.device(),
            actual: input.descriptor().device(),
        }
        .into());
    }
    if context.stream != input.descriptor().stream() {
        return Err(TensorError::StreamMismatch {
            expected: context.stream,
            actual: input.descriptor().stream(),
        }
        .into());
    }
    Ok(())
}

fn validate_kernel_output(
    operation: VaeOperation,
    output: &Tensor,
    expected_shape: &[u64],
    identity: &VaeIdentity,
    expected_stream: StreamId,
) -> Result<(), VaeError> {
    validate_kernel_output_binding(operation, output, identity, expected_stream)?;
    if output.descriptor().shape() != expected_shape {
        return Err(kernel_output_contract_error(
            operation,
            output,
            expected_shape,
            identity,
            expected_stream,
        ));
    }
    Ok(())
}

fn validate_kernel_output_binding(
    operation: VaeOperation,
    output: &Tensor,
    identity: &VaeIdentity,
    expected_stream: StreamId,
) -> Result<(), VaeError> {
    if output.descriptor().dtype() != identity.dtype()
        || output.descriptor().device() != identity.device()
        || output.descriptor().stream() != expected_stream
    {
        return Err(kernel_output_contract_error(
            operation,
            output,
            output.descriptor().shape(),
            identity,
            expected_stream,
        ));
    }
    Ok(())
}

fn clamp_decode_output(
    backend: &dyn TensorBackend,
    output: &Tensor,
    clamp: [f32; 2],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    let (lower_bounded, _) = backend.binary_scalar(
        BinaryOperation::Maximum,
        output,
        Scalar::Float(f64::from(clamp[0])),
        ScalarSide::Right,
        output.descriptor().clone(),
        context,
    )?;
    let (clamped, _) = backend.binary_scalar(
        BinaryOperation::Minimum,
        &lower_bounded,
        Scalar::Float(f64::from(clamp[1])),
        ScalarSide::Right,
        output.descriptor().clone(),
        context,
    )?;
    Ok(clamped)
}

fn kernel_output_contract_error(
    operation: VaeOperation,
    output: &Tensor,
    expected_shape: &[u64],
    identity: &VaeIdentity,
    expected_stream: StreamId,
) -> VaeError {
    VaeError::KernelOutputContractMismatch {
        operation,
        expected_shape: expected_shape.to_vec(),
        actual_shape: output.descriptor().shape().to_vec(),
        expected_dtype: identity.dtype(),
        actual_dtype: output.descriptor().dtype(),
        expected_device: identity.device(),
        actual_device: output.descriptor().device(),
        expected_stream,
        actual_stream: output.descriptor().stream(),
    }
}

#[cfg(test)]
fn validate_conformance_kernel_execution(descriptor: &VaeDescriptor) -> Result<(), VaeError> {
    if descriptor.identity().dtype() == DType::F32
        && descriptor.identity().device() == DeviceId::CPU
    {
        Ok(())
    } else {
        Err(VaeError::KernelExecutionBindingMismatch {
            expected_dtype: DType::F32,
            actual_dtype: descriptor.identity().dtype(),
            expected_device: DeviceId::CPU,
            actual_device: descriptor.identity().device(),
        })
    }
}

fn validate_pixel_input(vae: &NativeVae, input: &Tensor) -> Result<(), VaeError> {
    let expected_rank = if extra_1d_channels(vae.descriptor.identity().profile()).is_some() {
        3
    } else {
        usize::from(vae.latent_definition.dimensions) + 2
    };
    let shape = input.descriptor().shape();
    if shape.len() != expected_rank
        || shape.first() == Some(&0)
        || shape.get(1) == Some(&0)
        || shape.contains(&0)
    {
        let mut expected = vec![0; expected_rank];
        expected[1] = vae.descriptor.pixel_channels();
        return Err(VaeError::InvalidShape {
            expected,
            actual: shape.to_vec(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum PixelChannelPadding {
    Replicate,
    ConstantOne,
}

fn pixel_channel_padding(profile: &VaeKernelProfile) -> Option<PixelChannelPadding> {
    match profile {
        VaeKernelProfile::AudioOobleck44KhzV1
        | VaeKernelProfile::AudioOobleck48KhzV1
        | VaeKernelProfile::MusicDcaeV1
        | VaeKernelProfile::MmAudio16KhzV1
        | VaeKernelProfile::LtxAudioV1 => Some(PixelChannelPadding::Replicate),
        VaeKernelProfile::Wan21V1 => Some(PixelChannelPadding::ConstantOne),
        _ => None,
    }
}

fn extra_1d_channels(profile: &VaeKernelProfile) -> Option<u64> {
    match profile {
        VaeKernelProfile::MusicDcaeV1 | VaeKernelProfile::LtxAudioV1 => Some(16),
        _ => None,
    }
}

fn extra_1d_raw_shape(
    profile: &VaeKernelProfile,
    batch: u64,
    channels: u64,
    extra_channels: u64,
    time: u64,
) -> Vec<u64> {
    if profile == &VaeKernelProfile::LtxAudioV1 {
        vec![batch, channels, time, extra_channels]
    } else {
        vec![batch, channels, extra_channels, time]
    }
}

fn extra_1d_time_extent(
    profile: &VaeKernelProfile,
    shape: &[u64],
    channels: u64,
    extra_channels: u64,
) -> Result<u64, VaeError> {
    let (time_axis, frequency_axis) = if profile == &VaeKernelProfile::LtxAudioV1 {
        (2, 3)
    } else {
        (3, 2)
    };
    if shape.len() != 4
        || shape.get(1) != Some(&channels)
        || shape.get(frequency_axis) != Some(&extra_channels)
    {
        return Err(VaeError::InvalidShape {
            expected: extra_1d_raw_shape(profile, 0, channels, extra_channels, 0),
            actual: shape.to_vec(),
        });
    }
    shape.get(time_axis).copied().ok_or(VaeError::ShapeOverflow)
}

fn flatten_extra_1d_latent(
    backend: &dyn TensorBackend,
    latent: &Tensor,
    profile: &VaeKernelProfile,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    let shape = latent.descriptor().shape();
    let extra_channels = extra_1d_channels(profile).ok_or(VaeError::ShapeOverflow)?;
    let channels = *shape.get(1).ok_or(VaeError::ShapeOverflow)?;
    let time = extra_1d_time_extent(profile, shape, channels, extra_channels)?;
    let frequency_first = if profile == &VaeKernelProfile::LtxAudioV1 {
        let permuted = crate::vae_video::permute_read_only(latent, &[0, 1, 3, 2])?;
        crate::vae_video::contiguous_copy(backend, &permuted, context)?
    } else {
        latent.clone()
    };
    reshape_read_only(
        &frequency_first,
        vec![
            shape[0],
            channels
                .checked_mul(extra_channels)
                .ok_or(VaeError::ShapeOverflow)?,
            time,
        ],
    )
}

fn unflatten_extra_1d_latent(
    backend: &dyn TensorBackend,
    latent: &Tensor,
    profile: &VaeKernelProfile,
    channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    let shape = latent.descriptor().shape();
    let extra_channels = extra_1d_channels(profile).ok_or(VaeError::ShapeOverflow)?;
    let execution_channels = channels
        .checked_mul(extra_channels)
        .ok_or(VaeError::ShapeOverflow)?;
    if shape.len() != 3 || shape.get(1) != Some(&execution_channels) {
        return Err(VaeError::InvalidShape {
            expected: vec![0, execution_channels, 0],
            actual: shape.to_vec(),
        });
    }
    let frequency_first =
        reshape_read_only(latent, vec![shape[0], channels, extra_channels, shape[2]])?;
    if profile == &VaeKernelProfile::LtxAudioV1 {
        let permuted = crate::vae_video::permute_read_only(&frequency_first, &[0, 1, 3, 2])?;
        crate::vae_video::contiguous_copy(backend, &permuted, context)
    } else {
        Ok(frequency_first)
    }
}

fn extra_1d_ratio(profile: &VaeKernelProfile) -> Option<u64> {
    match profile {
        VaeKernelProfile::LtxAudioV1 => Some(1_764),
        _ => extra_1d_channels(profile).map(|_| 4_096),
    }
}

fn extra_1d_formula(profile: &VaeKernelProfile) -> Option<Result<VaeTileAxisFormula, VaeError>> {
    match profile {
        VaeKernelProfile::LtxAudioV1 => Some(VaeTileAxisFormula::checked_resampled_causal(
            44_100, 16_000, 160, 4, 640, 480,
        )),
        _ => extra_1d_ratio(profile).map(VaeTileAxisFormula::checked_linear),
    }
}

fn preserve_tiled_batch_group(profile: &VaeKernelProfile, operation: VaeOperation) -> bool {
    operation == VaeOperation::Decode
        && matches!(profile, VaeKernelProfile::TemporalAutoencodingEngineV1)
}

pub(crate) fn prepare_pixel_channels(
    backend: &dyn TensorBackend,
    input: &Tensor,
    expected_channels: u64,
    profile: &VaeKernelProfile,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, VaeError> {
    context.check()?;
    let actual_channels = input
        .descriptor()
        .shape()
        .get(1)
        .copied()
        .ok_or(VaeError::ShapeOverflow)?;
    if actual_channels == expected_channels {
        return Ok(input.clone());
    }
    if actual_channels > expected_channels {
        return Ok(input.narrow_read_only(1, 0, expected_channels)?);
    }
    let Some(padding_policy) = pixel_channel_padding(profile) else {
        let mut expected = input.descriptor().shape().to_vec();
        expected[1] = expected_channels;
        return Err(VaeError::InvalidShape {
            expected,
            actual: input.descriptor().shape().to_vec(),
        });
    };
    let missing = expected_channels - actual_channels;
    let mut padding = Vec::with_capacity(input.descriptor().rank() * 2);
    for axis in (0..input.descriptor().rank()).rev() {
        padding.push(0);
        padding.push(if axis == 1 {
            i64::try_from(missing).map_err(|_| VaeError::ShapeOverflow)?
        } else {
            0
        });
    }
    let value = match padding_policy {
        PixelChannelPadding::Replicate => None,
        PixelChannelPadding::ConstantOne => Some(DecodedScalar::Real(1.0)),
    };
    let (mut padded, event) = backend.constant_pad(input, &padding, value, context)?;
    backend.wait_event(event, context)?;
    if matches!(padding_policy, PixelChannelPadding::Replicate) {
        let last_channel = input.narrow_read_only(1, i64::try_from(actual_channels - 1)?, 1)?;
        for channel in actual_channels..expected_channels {
            context.check()?;
            let mut offsets = vec![0; input.descriptor().rank()];
            offsets[1] = channel;
            let (updated, event) =
                backend.replace_rectangular_slice(&padded, &last_channel, &offsets, context)?;
            backend.wait_event(event, context)?;
            padded = updated;
        }
    }
    Ok(padded)
}

fn reshape_read_only(input: &Tensor, shape: Vec<u64>) -> Result<Tensor, VaeError> {
    let descriptor = input.descriptor().reshaped_view(shape)?;
    Ok(input.view(descriptor, ViewAccess::ReadOnly)?)
}

fn validate_latent_input(vae: &NativeVae, input: &Tensor) -> Result<(), VaeError> {
    let definition = vae.latent_definition;
    let expected_rank = usize::from(definition.dimensions) + 2;
    let shape = input.descriptor().shape();
    let expected_channels = if definition.transform == LatentTransform::HunyuanImage21Refiner {
        definition
            .channels
            .checked_mul(2)
            .ok_or(VaeError::ShapeOverflow)?
    } else {
        definition.channels
    };
    let channel_matches = match definition.layout {
        LatentTensorLayout::ChannelsFirst => shape.get(1) == Some(&expected_channels),
        LatentTensorLayout::SequenceChannelsLast => shape.last() == Some(&expected_channels),
    };
    if shape.len() != expected_rank || shape.contains(&0) || !channel_matches {
        let mut expected = vec![0; expected_rank];
        match definition.layout {
            LatentTensorLayout::ChannelsFirst => expected[1] = expected_channels,
            LatentTensorLayout::SequenceChannelsLast => {
                expected[expected_rank - 1] = expected_channels
            }
        }
        return Err(VaeError::InvalidShape {
            expected,
            actual: shape.to_vec(),
        });
    }
    Ok(())
}

fn processed_encode_shape(
    definition: &LatentFormatDefinition,
    raw_shape: &[u64],
) -> Result<Vec<u64>, VaeError> {
    if definition.transform != LatentTransform::HunyuanImage21Refiner {
        return Ok(raw_shape.to_vec());
    }
    if definition.layout != LatentTensorLayout::ChannelsFirst
        || raw_shape.len() != 5
        || raw_shape.get(1) != Some(&definition.channels)
        || raw_shape
            .get(2)
            .is_none_or(|frames| *frames == 0 || frames.is_multiple_of(2))
        || raw_shape.contains(&0)
    {
        return Err(VaeError::InvalidShape {
            expected: vec![0, definition.channels, 1, 0, 0],
            actual: raw_shape.to_vec(),
        });
    }
    Ok(vec![
        raw_shape[0],
        definition
            .channels
            .checked_mul(2)
            .ok_or(VaeError::ShapeOverflow)?,
        raw_shape[2].checked_add(1).ok_or(VaeError::ShapeOverflow)? / 2,
        raw_shape[3],
        raw_shape[4],
    ])
}

#[cfg(test)]
fn encode_raw_spatial_shape(
    definition: &LatentFormatDefinition,
    pixel_shape: &[u64],
) -> Result<Vec<u64>, VaeError> {
    let formulas = tile_axis_formulas(definition)?;
    Ok(encode_spatial_geometry(&formulas, pixel_shape)?.2)
}

fn processed_latent_shape(
    definition: &LatentFormatDefinition,
    processed_shape: &[u64],
) -> Result<Vec<u64>, VaeError> {
    if definition.transform != LatentTransform::HunyuanImage21Refiner {
        return Ok(processed_shape.to_vec());
    }
    if definition.layout != LatentTensorLayout::ChannelsFirst
        || processed_shape.len() != 5
        || processed_shape[1]
            != definition
                .channels
                .checked_mul(2)
                .ok_or(VaeError::ShapeOverflow)?
        || processed_shape[2] == 0
    {
        return Err(VaeError::InvalidShape {
            expected: vec![0, definition.channels * 2, 0, 0, 0],
            actual: processed_shape.to_vec(),
        });
    }
    Ok(vec![
        processed_shape[0],
        definition.channels,
        processed_shape[2]
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(VaeError::ShapeOverflow)?,
        processed_shape[3],
        processed_shape[4],
    ])
}

#[cfg(test)]
fn decode_output_shape(
    definition: &LatentFormatDefinition,
    pixel_channels: u64,
    raw_shape: &[u64],
) -> Result<Vec<u64>, VaeError> {
    let spatial = latent_spatial_shape(definition.layout, raw_shape)?;
    let formulas = tile_axis_formulas(definition)?;
    let output_spatial = formulas
        .iter()
        .zip(spatial)
        .map(|(formula, extent)| formula.output_extent(VaeOperation::Decode, *extent))
        .collect::<Result<Vec<_>, _>>()?;
    shape_for_pixel_layout(raw_shape[0], pixel_channels, &output_spatial)
}

fn encode_spatial_geometry(
    formulas: &[VaeTileAxisFormula],
    pixel_shape: &[u64],
) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), VaeError> {
    let dimensions = formulas.len();
    if pixel_shape.len() != dimensions + 2 {
        return Err(VaeError::InvalidShape {
            expected: vec![0; dimensions + 2],
            actual: pixel_shape.to_vec(),
        });
    }
    let mut input_spatial = Vec::with_capacity(dimensions);
    let mut input_offsets = Vec::with_capacity(dimensions);
    let mut output_spatial = Vec::with_capacity(dimensions);
    for (dimension, formula) in formulas.iter().copied().enumerate() {
        let original = pixel_shape[dimension + 2];
        let effective = match formula {
            VaeTileAxisFormula::Linear { ratio } => original / ratio * ratio,
            VaeTileAxisFormula::Causal { .. } | VaeTileAxisFormula::ResampledCausal { .. } => {
                original
            }
        };
        if effective == 0 {
            return Err(VaeError::InvalidShape {
                expected: formulas
                    .iter()
                    .map(|formula| match formula {
                        VaeTileAxisFormula::Linear { ratio }
                        | VaeTileAxisFormula::Causal { ratio } => *ratio,
                        VaeTileAxisFormula::ResampledCausal {
                            original_frequency, ..
                        } => *original_frequency,
                    })
                    .collect(),
                actual: pixel_shape[2..].to_vec(),
            });
        }
        input_spatial.push(effective);
        input_offsets.push((original - effective) / 2);
        output_spatial.push(formula.output_extent(VaeOperation::Encode, effective)?);
    }
    Ok((input_spatial, input_offsets, output_spatial))
}

fn tile_axis_formulas(
    definition: &LatentFormatDefinition,
) -> Result<Vec<VaeTileAxisFormula>, VaeError> {
    let dimensions = usize::from(definition.dimensions);
    let mut formulas = Vec::with_capacity(dimensions);
    for dimension in 0..dimensions {
        let ratio = compression_ratio(definition, dimension);
        if dimensions == 3 && dimension == 0 && ratio > 1 {
            formulas.push(VaeTileAxisFormula::checked_causal(ratio)?);
        } else {
            formulas.push(VaeTileAxisFormula::checked_linear(ratio)?);
        }
    }
    Ok(formulas)
}

fn tile_tensor_layout(layout: LatentTensorLayout) -> TileTensorLayout {
    match layout {
        LatentTensorLayout::ChannelsFirst => TileTensorLayout::ChannelsFirst,
        LatentTensorLayout::SequenceChannelsLast => TileTensorLayout::SequenceChannelsLast,
    }
}

fn shape_for_latent_layout(
    batch: u64,
    channels: u64,
    spatial: &[u64],
    layout: LatentTensorLayout,
) -> Vec<u64> {
    let mut shape = Vec::with_capacity(spatial.len() + 2);
    shape.push(batch);
    match layout {
        LatentTensorLayout::ChannelsFirst => {
            shape.push(channels);
            shape.extend_from_slice(spatial);
        }
        LatentTensorLayout::SequenceChannelsLast => {
            shape.extend_from_slice(spatial);
            shape.push(channels);
        }
    }
    shape
}

fn shape_for_pixel_layout(
    batch: u64,
    channels: u64,
    spatial: &[u64],
) -> Result<Vec<u64>, VaeError> {
    if batch == 0 || channels == 0 || spatial.is_empty() || spatial.contains(&0) {
        return Err(VaeError::ShapeOverflow);
    }
    let mut shape = Vec::with_capacity(spatial.len() + 2);
    shape.push(batch);
    shape.push(channels);
    shape.extend_from_slice(spatial);
    Ok(shape)
}

fn latent_spatial_shape(layout: LatentTensorLayout, shape: &[u64]) -> Result<&[u64], VaeError> {
    if shape.len() < 3 {
        return Err(VaeError::ShapeOverflow);
    }
    Ok(match layout {
        LatentTensorLayout::ChannelsFirst => &shape[2..],
        LatentTensorLayout::SequenceChannelsLast => &shape[1..shape.len() - 1],
    })
}

fn compression_ratio(definition: &LatentFormatDefinition, dimension: usize) -> u64 {
    if definition.dimensions == 3 && dimension == 0 {
        definition.temporal_downscale_ratio
    } else {
        definition.spatial_downscale_ratio
    }
}

#[cfg(test)]
fn encode_value(
    vae: &NativeVae,
    pixels: &Tensor,
    batch: u64,
    latent_channel: u64,
    latent_spatial: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<f32, VaeError> {
    let source_channel = latent_channel % vae.descriptor.pixel_channels();
    let block_count = checked_product(
        (0..latent_spatial.len())
            .map(|dimension| compression_ratio(vae.latent_definition, dimension)),
    )?;
    let mut sum = 0.0_f64;
    let mut included = 0_u64;
    for block_linear in 0..block_count {
        check_periodically(block_linear, context)?;
        let offsets = coordinates_for_shape(
            block_linear,
            &(0..latent_spatial.len())
                .map(|dimension| compression_ratio(vae.latent_definition, dimension))
                .collect::<Vec<_>>(),
        )?;
        let mut indices = Vec::with_capacity(latent_spatial.len() + 2);
        indices.push(batch);
        indices.push(source_channel);
        let mut inside = true;
        for (dimension, (&coordinate, &offset)) in latent_spatial.iter().zip(&offsets).enumerate() {
            let source = coordinate
                .checked_mul(compression_ratio(vae.latent_definition, dimension))
                .and_then(|value| value.checked_add(offset))
                .ok_or(VaeError::ShapeOverflow)?;
            if source >= pixels.descriptor().shape()[dimension + 2] {
                inside = false;
                break;
            }
            indices.push(source);
        }
        if inside {
            sum += f64::from(read_f32(pixels, &indices)?);
            included = included.checked_add(1).ok_or(VaeError::ShapeOverflow)?;
        }
    }
    if included == 0 {
        return Err(VaeError::ZeroTileOutput);
    }
    Ok((sum / included as f64) as f32)
}

#[cfg(test)]
fn latent_indices(
    layout: LatentTensorLayout,
    batch: u64,
    channel: u64,
    spatial: &[u64],
) -> Vec<u64> {
    let mut indices = Vec::with_capacity(spatial.len() + 2);
    indices.push(batch);
    match layout {
        LatentTensorLayout::ChannelsFirst => {
            indices.push(channel);
            indices.extend_from_slice(spatial);
        }
        LatentTensorLayout::SequenceChannelsLast => {
            indices.extend_from_slice(spatial);
            indices.push(channel);
        }
    }
    indices
}

#[cfg(test)]
fn coordinates_for_shape(mut linear: u64, shape: &[u64]) -> Result<Vec<u64>, VaeError> {
    let mut coordinates = vec![0; shape.len()];
    for dimension in (0..shape.len()).rev() {
        if shape[dimension] == 0 {
            return Err(VaeError::ShapeOverflow);
        }
        coordinates[dimension] = linear % shape[dimension];
        linear /= shape[dimension];
    }
    Ok(coordinates)
}

#[cfg(test)]
fn checked_product(values: impl IntoIterator<Item = u64>) -> Result<u64, VaeError> {
    values.into_iter().try_fold(1_u64, |product, value| {
        product.checked_mul(value).ok_or(VaeError::ShapeOverflow)
    })
}

#[cfg(test)]
fn read_f32(tensor: &Tensor, indices: &[u64]) -> Result<f32, VaeError> {
    let bytes: [u8; 4] = tensor
        .element_bytes(indices)?
        .try_into()
        .map_err(|_| VaeError::UnsupportedDType(tensor.descriptor().dtype()))?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
fn write_f32(
    output: &mut sim_tensor::TensorWrite<'_>,
    indices: &[u64],
    value: f32,
) -> Result<(), VaeError> {
    output
        .element_bytes_mut(indices)?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
fn check_periodically(index: u64, context: &ExecutionContext<'_>) -> Result<(), VaeError> {
    if index & 0x3ff == 0 {
        context.check()?;
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), VaeError> {
    validate_identity_field(field, value)?;
    if value.chars().any(|character| {
        !(character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_' | ':'))
    }) {
        return Err(VaeError::InvalidIdentity {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_identity_field(field: &'static str, value: &str) -> Result<(), VaeError> {
    if value.is_empty() || value.len() > MAX_IDENTITY_FIELD_BYTES || value.trim() != value {
        return Err(VaeError::InvalidIdentity {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), VaeError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(VaeError::InvalidSha256);
    }
    Ok(())
}

fn validate_execution_dtype(dtype: DType) -> Result<(), VaeError> {
    if matches!(dtype, DType::F16 | DType::Bf16 | DType::F32) {
        Ok(())
    } else {
        Err(VaeError::UnsupportedDType(dtype))
    }
}

fn validate_patch_identity(
    artifact: &ArtifactRecord,
    patch: &PatchGraphIdentity,
) -> Result<(), VaeError> {
    validate_patch_identity_fields(&artifact.sha256, patch)
}

fn validate_patch_identity_fields(
    artifact_sha256: &str,
    patch: &PatchGraphIdentity,
) -> Result<(), VaeError> {
    match patch.validate_for_base(artifact_sha256) {
        Ok(()) => Ok(()),
        Err(PatchGraphIdentityError::SchemaVersion { actual, .. }) => {
            Err(VaeError::PatchSchemaVersion(actual))
        }
        Err(PatchGraphIdentityError::InvalidDigest { .. }) => Err(VaeError::InvalidSha256),
        Err(PatchGraphIdentityError::BaseDigestMismatch { .. }) => {
            Err(VaeError::PatchArtifactMismatch)
        }
    }
}

fn hash_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, GENERATED_LATENT_FORMATS,
        GENERATED_MODEL_FAMILY_REGISTRATIONS, ParserLimits, PatchGraph, PreviewReshape,
    };
    use sim_tensor::{CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, StreamId};
    use std::{
        error::Error,
        fs::File,
        io::Write,
        path::{Path, PathBuf},
    };

    static IDENTITY_1D: LatentFormatDefinition = LatentFormatDefinition {
        feature_id: "COMFY-MODEL-0900",
        identifier: "vae_test_1d",
        channels: 2,
        dimensions: 1,
        spatial_downscale_ratio: 2,
        temporal_downscale_ratio: 1,
        scale_factor: 1.0,
        shift_factor: 0.0,
        channel_means: &[],
        channel_stds: &[],
        preview_factors: &[],
        preview_bias: None,
        preview_reshape: PreviewReshape::None,
        decoder_name: None,
        layout: LatentTensorLayout::SequenceChannelsLast,
        transform: LatentTransform::Identity,
    };

    static AFFINE_2D: LatentFormatDefinition = LatentFormatDefinition {
        feature_id: "COMFY-MODEL-0901",
        identifier: "vae_test_2d",
        channels: 2,
        dimensions: 2,
        spatial_downscale_ratio: 2,
        temporal_downscale_ratio: 1,
        scale_factor: 2.0,
        shift_factor: 1.0,
        channel_means: &[],
        channel_stds: &[],
        preview_factors: &[],
        preview_bias: None,
        preview_reshape: PreviewReshape::None,
        decoder_name: None,
        layout: LatentTensorLayout::ChannelsFirst,
        transform: LatentTransform::Affine,
    };

    static IDENTITY_3D: LatentFormatDefinition = LatentFormatDefinition {
        feature_id: "COMFY-MODEL-0902",
        identifier: "vae_test_3d",
        channels: 1,
        dimensions: 3,
        spatial_downscale_ratio: 2,
        temporal_downscale_ratio: 2,
        scale_factor: 1.0,
        shift_factor: 0.0,
        channel_means: &[],
        channel_stds: &[],
        preview_factors: &[],
        preview_bias: None,
        preview_reshape: PreviewReshape::None,
        decoder_name: None,
        layout: LatentTensorLayout::ChannelsFirst,
        transform: LatentTransform::Identity,
    };

    static HUNYUAN_REFINER_3D: LatentFormatDefinition = LatentFormatDefinition {
        feature_id: "COMFY-MODEL-0904",
        identifier: "vae_test_hunyuan_refiner",
        channels: 1,
        dimensions: 3,
        spatial_downscale_ratio: 2,
        temporal_downscale_ratio: 2,
        scale_factor: 1.0,
        shift_factor: 0.0,
        channel_means: &[],
        channel_stds: &[],
        preview_factors: &[],
        preview_bias: None,
        preview_reshape: PreviewReshape::None,
        decoder_name: None,
        layout: LatentTensorLayout::ChannelsFirst,
        transform: LatentTransform::HunyuanImage21Refiner,
    };

    static SD15_REDUCED: LatentFormatDefinition =
        crate::generated_sd15_sim_model_0045::LATENT_FORMAT;

    fn artifact(digest_byte: char) -> Result<ArtifactRecord, Box<dyn Error>> {
        Ok(ArtifactRecord {
            key: ArtifactKey::new("models", PathBuf::from("vae/test.safetensors"))?,
            namespace: "vae".to_owned(),
            canonical_path: PathBuf::from("/verified/models/vae/test.safetensors"),
            byte_size: 4096,
            modified_nanoseconds: 1,
            sha256: std::iter::repeat_n(digest_byte, 64).collect(),
            availability: ArtifactAvailability::Present,
        })
    }

    fn write_safetensors(path: &Path) -> Result<(), Box<dyn Error>> {
        let header = r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&1.0_f32.to_le_bytes())?;
        Ok(())
    }

    fn write_music_safetensors(path: &Path) -> Result<(), Box<dyn Error>> {
        let header = r#"{"vocoder.backbone.channel_layers.0.0.bias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&1.0_f32.to_le_bytes())?;
        Ok(())
    }

    fn write_cog_safetensors(path: &Path) -> Result<(), Box<dyn Error>> {
        let header = r#"{"decoder.conv_in.conv.weight":{"dtype":"F32","shape":[1,16],"data_offsets":[0,64]},"decoder.mid_block.resnets.0.norm1.norm_layer.weight":{"dtype":"F32","shape":[1],"data_offsets":[64,68]},"encoder.conv_out.conv.weight":{"dtype":"F32","shape":[32],"data_offsets":[68,196]}}"#;
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&[0_u8; 196])?;
        Ok(())
    }

    fn write_taesd_safetensors(path: &Path) -> Result<(), Box<dyn Error>> {
        let header =
            r#"{"taesd_decoder.1.weight":{"dtype":"F32","shape":[1,4],"data_offsets":[0,16]}}"#;
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&[0_u8; 16])?;
        Ok(())
    }

    fn patch_identity(artifact: &ArtifactRecord) -> Result<PatchGraphIdentity, Box<dyn Error>> {
        Ok(PatchGraph::checked_semantic(artifact.sha256.clone(), Vec::new())?.identity())
    }

    fn boundary(
        definition: &'static LatentFormatDefinition,
        channels: u64,
    ) -> Result<VaeBoundary, VaeError> {
        match definition.dimensions {
            1 => VaeBoundary::audio(channels, 48_000),
            2 => VaeBoundary::image(channels),
            3 => VaeBoundary::video(channels),
            _ => Err(VaeError::BoundaryLatentDimensionMismatch {
                kind: VaeBoundaryKind::Image,
                dimensions: definition.dimensions,
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn checked_descriptor(
        artifact: &ArtifactRecord,
        family: ModelFamilyIdentity,
        definition: &'static LatentFormatDefinition,
        architecture: &str,
        profile: VaeKernelProfile,
        channels: u64,
        clamp: [f32; 2],
    ) -> Result<VaeDescriptor, VaeError> {
        VaeDescriptor::checked(
            artifact,
            family,
            definition,
            VaeArchitectureIdentity::checked(architecture)?,
            patch_identity(artifact).map_err(|error| VaeError::Allocation(error.to_string()))?,
            DType::F32,
            DeviceId::CPU,
            boundary(definition, channels)?,
            profile,
            clamp,
        )
    }

    fn vae(
        definition: &'static LatentFormatDefinition,
        pixel_channels: u64,
        clamp: [f32; 2],
    ) -> Result<NativeVae, Box<dyn Error>> {
        let artifact = artifact('a')?;
        let descriptor = checked_descriptor(
            &artifact,
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
            definition,
            "zed.vae.block_average_nearest.v1",
            VaeKernelProfile::BlockAverageNearestV1,
            pixel_channels,
            clamp,
        )?;
        Ok(NativeVae::checked(descriptor, definition)?)
    }

    #[test]
    fn native_vae_recipe_rejects_oversized_configuration_before_state_work()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let oversized = format!("{{{}", " ".repeat(256 * 1024));
        let result = load_native_vae_recipe(
            crate::model_family::NativeModelRecipeSourceForm::Standalone,
            &"a".repeat(64),
            Vec::new(),
            Some(&oversized),
            &backend,
            &context,
        );
        match result {
            Err(NativeVaeRecipeError::Architecture(VaeArchitectureError::ConfigurationLimit {
                actual,
                maximum,
            })) => {
                assert_eq!(actual, oversized.len());
                assert_eq!(maximum, 256 * 1024);
            }
            Err(error) => return Err(format!("unexpected oversized VAE error: {error}").into()),
            Ok(_) => return Err("oversized VAE configuration was accepted".into()),
        }
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn source_image_ratio_preserves_stage_c_loader_and_non_image_boundaries()
    -> Result<(), Box<dyn Error>> {
        for (profile, expected) in [
            (VaeKernelProfile::StableCascadeStageCEncoderV1, 32),
            (VaeKernelProfile::StableCascadeStageCCombinedV1, 32),
            (VaeKernelProfile::StableCascadeStageCPreviewerV1, 8),
            (VaeKernelProfile::AutoencoderKlV1, 8),
            (VaeKernelProfile::ExplicitAutoencoderKlV1, 8),
            (VaeKernelProfile::TaesdV1, 8),
            (VaeKernelProfile::PixelSpaceV1, 1),
        ] {
            assert_eq!(
                source_image_encode_downscale_ratio(
                    &profile,
                    &VaeLoaderConfiguration::Automatic,
                    42
                )?,
                expected,
            );
        }
        let taesd = VaeLoaderConfiguration::Taesd {
            latent_channels: 128,
            metadata_override: true,
        };
        assert_eq!(
            source_image_encode_downscale_ratio(&VaeKernelProfile::TaesdV1, &taesd, 42)?,
            8
        );
        assert_eq!(
            image_spatial_ratios(&VaeKernelProfile::TaesdV1, &taesd, 42)?,
            Some((16, 16))
        );
        assert!(matches!(
            source_image_encode_downscale_ratio(
                &VaeKernelProfile::LtxAudioV1,
                &VaeLoaderConfiguration::Automatic,
                42
            ),
            Err(VaeError::InvalidOperationGeometry { .. })
        ));
        Ok(())
    }

    #[test]
    fn source_image_decode_ratio_preserves_wrapper_defaults_and_loader_geometry()
    -> Result<(), Box<dyn Error>> {
        for (profile, expected) in [
            (VaeKernelProfile::StableCascadeStageCEncoderV1, 8),
            (VaeKernelProfile::StableCascadeStageCCombinedV1, 8),
            (VaeKernelProfile::StableCascadeStageCPreviewerV1, 8),
            (VaeKernelProfile::AutoencoderKlV1, 8),
            (VaeKernelProfile::AutoencoderKlX4V1, 4),
            (VaeKernelProfile::ExplicitAutoencoderKlV1, 8),
            (VaeKernelProfile::TaesdV1, 8),
            (VaeKernelProfile::StableCascadeStageAV1, 4),
            (VaeKernelProfile::HunyuanImageV1, 32),
            (VaeKernelProfile::PixelSpaceV1, 1),
        ] {
            assert_eq!(
                source_image_decode_upscale_ratio(
                    &profile,
                    &VaeLoaderConfiguration::Automatic,
                    42
                )?,
                expected
            );
        }
        let taesd = VaeLoaderConfiguration::Taesd {
            latent_channels: 128,
            metadata_override: true,
        };
        assert_eq!(
            source_image_decode_upscale_ratio(&VaeKernelProfile::TaesdV1, &taesd, 42)?,
            8
        );
        assert_eq!(
            image_spatial_ratios(&VaeKernelProfile::TaesdV1, &taesd, 42)?,
            Some((16, 16))
        );
        assert!(matches!(
            source_image_decode_upscale_ratio(
                &VaeKernelProfile::LtxAudioV1,
                &VaeLoaderConfiguration::Automatic,
                42
            ),
            Err(VaeError::InvalidOperationGeometry { .. })
        ));
        Ok(())
    }

    #[test]
    fn complete_ltx_audio_resource_plans_use_original_normal_kernel_time_axis()
    -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../sim_test_support/fixtures/nodes/model-latent-comfy-node-0344/source-contract.json"
        ))?;
        let model = fixture
            .get("empty_audio_model")
            .ok_or("Original audio model missing")?;
        assert_eq!(model["state_count"], 296);
        assert_eq!(model["contiguous_f32_bytes"], 435_889_104_u64);
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(4 * 1024 * 1024)?,
            &cancellation,
        );
        let mut state = Vec::new();
        for entry in model["state"]
            .as_array()
            .ok_or("Original state schema missing")?
        {
            let name = entry["name"].as_str().ok_or("State name missing")?;
            let shape = serde_json::from_value::<Vec<i64>>(entry["shape"].clone())?;
            assert_eq!(entry["dtype"], "torch.float32");
            let value = if name.ends_with("std-of-means") {
                1.0
            } else {
                0.0
            };
            let scalar = tensor_from_f32(&backend, &[], &[value], &context)?;
            let tensor =
                sim_tensor::generated_shape_layout_transform_01::tensor_expand_exact_native(
                    &scalar,
                    &shape,
                    &cancellation,
                )?;
            state.push((name.to_owned(), tensor));
        }
        let configuration = serde_json::to_string(&model["configuration"])?;
        let source_digest = format!("{:x}", Sha256::digest(serde_json::to_vec(model)?));
        let NativeVaeRecipeResource::Native(vae) = load_native_vae_recipe(
            crate::model_family::NativeModelRecipeSourceForm::Standalone,
            &source_digest,
            state,
            Some(&configuration),
            &backend,
            &context,
        )?
        else {
            return Err("Expected complete LTX audio resource".into());
        };
        assert_eq!(
            vae.descriptor().identity().profile(),
            &VaeKernelProfile::LtxAudioV1
        );
        assert!(backend.memory_snapshot().current_bytes < 1024 * 1024);
        for (time, output_samples) in [(1, 160), (3, 1440), (97, 61_600)] {
            let latent = tensor_from_f32(
                &backend,
                &[2, 8, time, 16],
                &vec![0.0; usize::try_from(2 * 8 * time * 16)?],
                &context,
            )?;
            let storage = latent.storage_id();
            let plan = vae.plan_decode_tiles(&latent, vec![time], vec![0])?;
            assert_eq!(plan.input_shape(), &[2, 8, time, 16]);
            assert_eq!(plan.output_shape(), &[2, 2, output_samples]);
            assert_eq!(plan.execution_output_shape, [2, 2, output_samples]);
            assert_eq!(plan.tile_count(), 1);
            assert_eq!(latent.storage_id(), storage);
            assert_eq!(latent.descriptor().shape(), &[2, 8, time, 16]);
        }
        let waveform = tensor_from_f32(&backend, &[2, 2, 44_100], &vec![0.0; 176_400], &context)?;
        let encode = vae.plan_encode_tiles(&waveform, vec![44_100], vec![0])?;
        assert_eq!(encode.input_shape(), &[2, 2, 44_100]);
        assert_eq!(encode.output_shape(), &[2, 8, 26, 16]);
        assert_eq!(encode.execution_output_shape, [2, 128, 26]);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        cancellation.cancel();
        assert!(matches!(
            vae.admit_execution_target(&backend, &context),
            Err(VaeError::NativeOps(NativeOpsError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn ltx_audio_time_axis_survives_tiled_storage_projection() -> Result<(), Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let profile = VaeKernelProfile::LtxAudioV1;
        for time in [1, 3, 97] {
            let shape = [2, 8, time, 16];
            assert_eq!(extra_1d_time_extent(&profile, &shape, 8, 16)?, time);
            let length = usize::try_from(shape.iter().product::<u64>())?;
            let values = (0..length)
                .map(|index| index as f32 * 0.125)
                .collect::<Vec<_>>();
            let latent = tensor_from_f32(&backend, &shape, &values, &context)?;
            let flattened = flatten_extra_1d_latent(&backend, &latent, &profile, &context)?;
            assert_eq!(flattened.descriptor().shape(), &[2, 128, time]);
            let flat_values = tensor_to_f32(&backend, &flattened, &context)?;
            if time == 3 {
                assert_eq!(
                    flat_values
                        .get(..6)
                        .ok_or("LTX first batch projection too short")?,
                    &[0.0, 2.0, 4.0, 0.125, 2.125, 4.125]
                );
                assert_eq!(
                    flat_values
                        .get(384..390)
                        .ok_or("LTX second batch projection too short")?,
                    &[48.0, 50.0, 52.0, 48.125, 50.125, 52.125]
                );
            }
            let restored = unflatten_extra_1d_latent(&backend, &flattened, &profile, 8, &context)?;
            assert_eq!(restored.descriptor().shape(), &shape);
            assert_eq!(&*tensor_to_f32(&backend, &restored, &context)?, &values);
            assert_eq!(&*tensor_to_f32(&backend, &latent, &context)?, &values);
        }
        assert!(extra_1d_time_extent(&profile, &[1, 8, 16, 3], 8, 16).is_err());
        let music_profile = VaeKernelProfile::MusicDcaeV1;
        let music = tensor_from_f32(&backend, &[1, 2, 16, 3], &[0.5; 96], &context)?;
        let music_flat = flatten_extra_1d_latent(&backend, &music, &music_profile, &context)?;
        assert_eq!(music_flat.storage_id(), music.storage_id());
        assert_eq!(music_flat.descriptor().shape(), &[1, 32, 3]);
        assert_eq!(
            extra_1d_time_extent(&music_profile, &[1, 2, 16, 3], 2, 16)?,
            3
        );
        cancellation.cancel();
        assert!(matches!(
            flatten_extra_1d_latent(&backend, &music, &music_profile, &context),
            Err(VaeError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn image_profiles_own_operation_specific_geometry() -> Result<(), Box<dyn Error>> {
        let cases = [
            (
                VaeKernelProfile::AutoencoderKlX4V1,
                VaeLoaderConfiguration::DefaultKl {
                    x4: true,
                    legacy_prefix_rewrite: false,
                    batch_norm_latent: false,
                    asymmetric_decoder_channels: None,
                    embed_dim: Some(4),
                },
                4,
                4,
            ),
            (
                VaeKernelProfile::AutoencoderKlBatchNormV1,
                VaeLoaderConfiguration::DefaultKl {
                    x4: false,
                    legacy_prefix_rewrite: false,
                    batch_norm_latent: true,
                    asymmetric_decoder_channels: None,
                    embed_dim: Some(4),
                },
                16,
                16,
            ),
            (
                VaeKernelProfile::StableCascadeStageCCombinedV1,
                VaeLoaderConfiguration::Automatic,
                32,
                8,
            ),
            (
                VaeKernelProfile::TaesdV1,
                VaeLoaderConfiguration::Taesd {
                    latent_channels: 128,
                    metadata_override: true,
                },
                16,
                16,
            ),
            (
                VaeKernelProfile::PixelSpaceV1,
                VaeLoaderConfiguration::Automatic,
                1,
                1,
            ),
        ];
        for (profile, configuration, encode_ratio, decode_ratio) in cases {
            let geometry = VaeSpatialGeometry::checked(&profile, &configuration, &SD15_REDUCED)?;
            assert_eq!(
                geometry.formulas(VaeOperation::Encode),
                &[VaeTileAxisFormula::Linear {
                    ratio: encode_ratio
                }; 2]
            );
            assert_eq!(
                geometry.formulas(VaeOperation::Decode),
                &[VaeTileAxisFormula::Linear {
                    ratio: decode_ratio
                }; 2]
            );
        }

        let explicit_params = r#"{"ddconfig":{"attn_resolutions":[],"ch":32,"ch_mult":[1,2,4],"double_z":true,"in_channels":3,"num_res_blocks":1,"out_ch":3,"resolution":16,"z_channels":4},"decoder_ddconfig":{"attn_resolutions":[],"ch":32,"ch_mult":[1,2],"in_channels":3,"num_res_blocks":1,"out_ch":3,"resolution":8,"z_channels":4},"embed_dim":4}"#;
        let explicit = VaeLoaderConfiguration::ExplicitAutoencoderKl {
            params_sha256: format!("{:x}", Sha256::digest(explicit_params.as_bytes())),
            params_json: explicit_params.to_owned(),
        };
        let geometry = VaeSpatialGeometry::checked(
            &VaeKernelProfile::ExplicitAutoencoderKlV1,
            &explicit,
            &SD15_REDUCED,
        )?;
        assert_eq!(
            geometry.formulas(VaeOperation::Encode),
            &[VaeTileAxisFormula::Linear { ratio: 4 }; 2]
        );
        assert_eq!(
            geometry.formulas(VaeOperation::Decode),
            &[VaeTileAxisFormula::Linear { ratio: 2 }; 2]
        );
        assert!(preserve_tiled_batch_group(
            &VaeKernelProfile::TemporalAutoencodingEngineV1,
            VaeOperation::Decode,
        ));
        assert!(!preserve_tiled_batch_group(
            &VaeKernelProfile::TemporalAutoencodingEngineV1,
            VaeOperation::Encode,
        ));
        assert!(!preserve_tiled_batch_group(
            &VaeKernelProfile::AutoencoderKlV1,
            VaeOperation::Decode,
        ));
        Ok(())
    }

    fn learned_weights(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        seed: u32,
    ) -> Result<BTreeMap<String, Tensor>, Box<dyn Error>> {
        let mut weights = BTreeMap::new();
        for (name, shape) in sd15_reduced_weight_manifest() {
            let count = usize::try_from(
                shape
                    .iter()
                    .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
                    .ok_or("weight shape overflow")?,
            )?;
            let tensor_values = if name.ends_with("norm1.weight")
                || name.ends_with("norm2.weight")
                || name.ends_with("norm_out.weight")
            {
                vec![1.0; count]
            } else if name.ends_with(".bias") {
                vec![0.001 * seed as f32; count]
            } else {
                (0..count)
                    .map(|index| {
                        let value = ((index as u32).wrapping_add(seed * 7) % 17) as f32 - 8.0;
                        value * 0.0075
                    })
                    .collect()
            };
            weights.insert(name, upload(backend, shape, &tensor_values, context)?);
        }
        Ok(weights)
    }

    fn learned_vae(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        seed: u32,
        artifact_digest: &str,
    ) -> Result<NativeVae, Box<dyn Error>> {
        let mut record = artifact('a')?;
        record.sha256 = artifact_digest.to_owned();
        let descriptor = checked_descriptor(
            &record,
            ModelFamilyIdentity::new("COMFY-MODEL-0117", "SD15", "sd15-v1")?,
            &SD15_REDUCED,
            "sim.ldm.models.autoencoder.AutoencoderKL.reduced.v1",
            VaeKernelProfile::Sd15AutoencoderKlReducedV1,
            3,
            [0.0, 1.0],
        )?;
        let kernel = Sd15LearnedVaeKernel::checked(
            descriptor.identity().clone(),
            learned_weights(backend, context, seed)?,
        )?;
        Ok(NativeVae::checked_sd15(descriptor, &SD15_REDUCED, kernel)?)
    }

    fn backend_and_context<'a>(
        cancellation: &'a CancellationToken,
        memory: u64,
    ) -> Result<
        (
            CpuBackend,
            sim_tensor::BackendWorkspaceAuthority,
            ExecutionContext<'a>,
        ),
        TensorError,
    > {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(memory)?,
            rng_phase: None,
            cancellation,
        };
        Ok((backend, authority, context))
    }

    fn upload(
        backend: &CpuBackend,
        shape: Vec<u64>,
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, TensorError> {
        let descriptor =
            TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        Ok(backend.upload_f32(descriptor, values, context)?.0)
    }

    fn ltx_statistics_vae(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        means: &[f32; 128],
        standard_deviations: &[f32; 128],
    ) -> Result<NativeVae, Box<dyn Error>> {
        let artifact = artifact('9')?;
        let descriptor = checked_descriptor(
            &artifact,
            ModelFamilyIdentity::new("COMFY-MODEL-0103", "LTXV", "ltxv-transformer-v1")?,
            crate::LTXV_LATENT_FORMAT,
            "sim.ldm.lightricks.vae.VideoVAE.v1",
            VaeKernelProfile::LtxVideoV1 {
                configuration_sha256: None,
            },
            3,
            [-1.0, 1.0],
        )?;
        let mean = upload(backend, vec![128], means, context)?;
        let standard_deviation = upload(backend, vec![128], standard_deviations, context)?;
        let module = NativeModule::module_dict(
            "ltx-statistics-test",
            vec![
                NativeModule::buffer("per_channel_statistics.mean-of-means", mean)?,
                NativeModule::buffer("per_channel_statistics.std-of-means", standard_deviation)?,
            ],
        )?;
        let binding =
            VaeModelBinding::checked_transport_fixture(&descriptor, module, context.cancellation)?;
        let functions = VaeKernelFunctions::checked(
            descriptor.identity().architecture().clone(),
            native_test_encode,
            native_test_decode,
        );
        Ok(NativeVae::checked_kernel(
            descriptor,
            crate::LTXV_LATENT_FORMAT,
            binding,
            functions,
        )?)
    }

    fn values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn Error>> {
        let bytes = tensor.contiguous_bytes()?;
        let mut result = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            result.push(f32::from_le_bytes(chunk.try_into()?));
        }
        Ok(result)
    }

    fn native_test_encode(
        module: &NativeModule,
        backend: &dyn TensorBackend,
        _cpu_backend: Option<&CpuBackend>,
        pixels: &Tensor,
        definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let spatial = encode_raw_spatial_shape(definition, pixels.descriptor().shape())?;
        let mut shape = Vec::with_capacity(spatial.len() + 2);
        shape.push(pixels.descriptor().shape()[0]);
        match definition.layout {
            LatentTensorLayout::ChannelsFirst => {
                shape.push(definition.channels);
                shape.extend(spatial);
            }
            LatentTensorLayout::SequenceChannelsLast => {
                shape.extend(spatial);
                shape.push(definition.channels);
            }
        }
        let descriptor = TensorDescriptor::contiguous(
            shape,
            pixels.descriptor().dtype(),
            pixels.descriptor().device(),
            pixels.descriptor().stream(),
        )?;
        Ok(backend
            .fill(
                Scalar::Float(f64::from(native_test_module_value(module)?)),
                descriptor,
                context,
            )?
            .0)
    }

    fn native_test_decode(
        module: &NativeModule,
        backend: &dyn TensorBackend,
        _cpu_backend: Option<&CpuBackend>,
        latent: &Tensor,
        definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let shape =
            decode_output_shape(definition, definition.channels, latent.descriptor().shape())?;
        let descriptor = TensorDescriptor::contiguous(
            shape,
            latent.descriptor().dtype(),
            latent.descriptor().device(),
            latent.descriptor().stream(),
        )?;
        Ok(backend
            .fill(
                Scalar::Float(f64::from(native_test_module_value(module)?)),
                descriptor,
                context,
            )?
            .0)
    }

    fn native_cpu_projection_encode(
        module: &NativeModule,
        backend: &dyn TensorBackend,
        cpu_backend: Option<&CpuBackend>,
        input: &Tensor,
        definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let cpu_backend = cpu_backend.ok_or(VaeError::StageCRequiresCpuBackend)?;
        let backend_address = std::ptr::from_ref(backend).cast::<()>();
        let cpu_address = std::ptr::from_ref(cpu_backend).cast::<()>();
        if backend_address != cpu_address {
            return Err(VaeError::StageCRequiresCpuBackend);
        }
        native_test_encode(
            module,
            backend,
            Some(cpu_backend),
            input,
            definition,
            context,
        )
    }

    fn native_cpu_projection_decode(
        module: &NativeModule,
        backend: &dyn TensorBackend,
        cpu_backend: Option<&CpuBackend>,
        input: &Tensor,
        definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let cpu_backend = cpu_backend.ok_or(VaeError::StageCRequiresCpuBackend)?;
        let backend_address = std::ptr::from_ref(backend).cast::<()>();
        let cpu_address = std::ptr::from_ref(cpu_backend).cast::<()>();
        if backend_address != cpu_address {
            return Err(VaeError::StageCRequiresCpuBackend);
        }
        let shape = decode_output_shape(definition, 3, input.descriptor().shape())?;
        let descriptor = TensorDescriptor::contiguous(
            shape,
            input.descriptor().dtype(),
            input.descriptor().device(),
            input.descriptor().stream(),
        )?;
        Ok(backend
            .fill(
                Scalar::Float(f64::from(native_test_module_value(module)?)),
                descriptor,
                context,
            )?
            .0)
    }

    struct DelegatingCpuBackend {
        inner: CpuBackend,
    }

    impl sim_tensor::CachedAllocationOwner for DelegatingCpuBackend {
        fn cache_device(&self) -> DeviceId {
            self.inner.cache_device()
        }

        fn release_cached_allocations(
            &self,
            cancellation: &CancellationToken,
        ) -> Result<u64, TensorError> {
            self.inner.release_cached_allocations(cancellation)
        }
    }

    impl TensorBackend for DelegatingCpuBackend {
        fn device(&self) -> DeviceId {
            self.inner.device()
        }

        fn capabilities(&self) -> &sim_tensor::BackendCapabilityMatrix {
            self.inner.capabilities()
        }

        fn reserve_workspace(
            &self,
            context: &ExecutionContext<'_>,
            requested: u64,
        ) -> Result<sim_tensor::BackendWorkspaceLease, TensorError> {
            self.inner.reserve_workspace(context, requested)
        }

        fn allocate(
            &self,
            descriptor: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.allocate(descriptor, context)
        }

        fn copy(
            &self,
            source: &Tensor,
            destination: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.copy(source, destination, context)
        }

        fn record_event(
            &self,
            context: &ExecutionContext<'_>,
        ) -> Result<sim_tensor::EventFence, TensorError> {
            self.inner.record_event(context)
        }

        fn wait_event(
            &self,
            event: sim_tensor::EventFence,
            context: &ExecutionContext<'_>,
        ) -> Result<(), TensorError> {
            self.inner.wait_event(event, context)
        }

        fn fill(
            &self,
            value: Scalar,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.fill(value, output, context)
        }

        fn unary(
            &self,
            operation: sim_tensor::UnaryOperation,
            input: &Tensor,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.unary(operation, input, output, context)
        }

        fn binary(
            &self,
            operation: BinaryOperation,
            left: &Tensor,
            right: &Tensor,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.binary(operation, left, right, output, context)
        }

        fn binary_scalar(
            &self,
            operation: BinaryOperation,
            input: &Tensor,
            scalar: Scalar,
            scalar_side: ScalarSide,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner
                .binary_scalar(operation, input, scalar, scalar_side, output, context)
        }

        fn reduction(
            &self,
            operation: &sim_tensor::ReductionSpec,
            input: &Tensor,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.reduction(operation, input, output, context)
        }

        fn indexing(
            &self,
            operation: &sim_tensor::IndexSpec,
            inputs: &[Tensor],
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.indexing(operation, inputs, output, context)
        }

        fn resize(
            &self,
            operation: sim_tensor::ResizeSpec,
            input: &Tensor,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.resize(operation, input, output, context)
        }

        fn convolution(
            &self,
            operation: &sim_tensor::ConvolutionSpec,
            inputs: &[Tensor],
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner.convolution(operation, inputs, output, context)
        }

        fn linear_algebra(
            &self,
            operation: sim_tensor::LinearAlgebraOperation,
            inputs: &[Tensor],
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, sim_tensor::EventFence), TensorError> {
            self.inner
                .linear_algebra(operation, inputs, output, context)
        }

        fn custom_kernel(
            &self,
            kernel: &sim_tensor::CustomKernelId,
            inputs: &[Tensor],
            outputs: &[TensorDescriptor],
            context: &ExecutionContext<'_>,
        ) -> Result<(Vec<Tensor>, sim_tensor::EventFence), TensorError> {
            self.inner.custom_kernel(kernel, inputs, outputs, context)
        }
    }

    fn native_test_module_value(module: &NativeModule) -> Result<f32, VaeError> {
        let buffer = module
            .registered_buffer()
            .ok_or(VaeError::NativeModuleHasNoState)?;
        let bytes: [u8; 4] = buffer
            .linear_element_bytes(0)?
            .try_into()
            .map_err(|_| VaeError::UnsupportedDType(buffer.descriptor().dtype()))?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn native_wrong_shape(
        _module: &NativeModule,
        _backend: &dyn TensorBackend,
        _cpu_backend: Option<&CpuBackend>,
        input: &Tensor,
        _definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        context.check()?;
        Ok(input.clone())
    }

    fn native_wrong_dtype(
        module: &NativeModule,
        backend: &dyn TensorBackend,
        cpu_backend: Option<&CpuBackend>,
        input: &Tensor,
        definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let correctly_shaped =
            native_test_encode(module, backend, cpu_backend, input, definition, context)?;
        let descriptor = TensorDescriptor::contiguous(
            correctly_shaped.descriptor().shape().to_vec(),
            DType::U8,
            correctly_shaped.descriptor().device(),
            correctly_shaped.descriptor().stream(),
        )?;
        Ok(backend.fill(Scalar::Unsigned(0), descriptor, context)?.0)
    }

    fn native_wrong_stream(
        module: &NativeModule,
        backend: &dyn TensorBackend,
        cpu_backend: Option<&CpuBackend>,
        input: &Tensor,
        definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        let correctly_shaped =
            native_test_encode(module, backend, cpu_backend, input, definition, context)?;
        let stream = StreamId::new(
            context
                .stream
                .get()
                .checked_add(1)
                .ok_or(VaeError::ShapeOverflow)?,
        );
        let descriptor = TensorDescriptor::contiguous(
            correctly_shaped.descriptor().shape().to_vec(),
            correctly_shaped.descriptor().dtype(),
            correctly_shaped.descriptor().device(),
            stream,
        )?;
        let other_context = ExecutionContext {
            stream,
            scratch: context.scratch.clone(),
            rng_phase: context.rng_phase,
            cancellation: context.cancellation,
        };
        Ok(backend
            .fill(Scalar::Float(0.0), descriptor, &other_context)?
            .0)
    }

    fn native_extra_audio_encode(
        _module: &NativeModule,
        backend: &dyn TensorBackend,
        _cpu_backend: Option<&CpuBackend>,
        pixels: &Tensor,
        definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        context.check()?;
        let shape = pixels.descriptor().shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != 2 || !shape[2].is_multiple_of(4_096) {
            return Err(VaeError::InvalidShape {
                expected: vec![1, 2, 4_096],
                actual: shape.to_vec(),
            });
        }
        for sample in [0, shape[2] - 1] {
            if read_f32(pixels, &[0, 0, sample])? != read_f32(pixels, &[0, 1, sample])? {
                return Err(VaeError::KernelProfileMismatch);
            }
        }
        let descriptor = TensorDescriptor::contiguous(
            vec![1, definition.channels, 16, shape[2] / 4_096],
            pixels.descriptor().dtype(),
            pixels.descriptor().device(),
            pixels.descriptor().stream(),
        )?;
        Ok(backend.fill(Scalar::Float(0.25), descriptor, context)?.0)
    }

    fn native_extra_audio_decode(
        _module: &NativeModule,
        backend: &dyn TensorBackend,
        _cpu_backend: Option<&CpuBackend>,
        latent: &Tensor,
        definition: &'static LatentFormatDefinition,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, VaeError> {
        context.check()?;
        let shape = latent.descriptor().shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != definition.channels || shape[2] != 16 {
            return Err(VaeError::InvalidShape {
                expected: vec![1, definition.channels, 16, 1],
                actual: shape.to_vec(),
            });
        }
        let output_length = shape[3].checked_mul(4_096).ok_or(VaeError::ShapeOverflow)?;
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 2, output_length],
            latent.descriptor().dtype(),
            latent.descriptor().device(),
            latent.descriptor().stream(),
        )?;
        Ok(backend.fill(Scalar::Float(0.5), descriptor, context)?.0)
    }

    #[test]
    fn identity_is_bound_to_artifact_family_format_architecture_and_profile()
    -> Result<(), Box<dyn Error>> {
        let first = vae(&AFFINE_2D, 3, [-1.0, 1.0])?;
        let second_artifact = artifact('b')?;
        let second_descriptor = checked_descriptor(
            &second_artifact,
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
            &AFFINE_2D,
            "zed.vae.block_average_nearest.v1",
            VaeKernelProfile::BlockAverageNearestV1,
            3,
            [-1.0, 1.0],
        )?;
        let second = NativeVae::checked(second_descriptor, &AFFINE_2D)?;
        assert_ne!(
            first.descriptor().identity().digest(),
            second.descriptor().identity().digest()
        );
        assert_eq!(
            first.descriptor().identity().artifact_sha256(),
            "a".repeat(64)
        );

        let encoded = serde_json::to_vec(first.descriptor().identity())?;
        let decoded: VaeIdentity = serde_json::from_slice(&encoded)?;
        assert_eq!(&decoded, first.descriptor().identity());
        for (field, replacement) in [
            (
                "architecture",
                serde_json::Value::String("zed.vae.changed.v1".to_owned()),
            ),
            ("dtype", serde_json::to_value(DType::Bf16)?),
            (
                "device",
                serde_json::to_value(DeviceId::new(DeviceKind::Metal, 0))?,
            ),
            ("boundary", serde_json::to_value(VaeBoundary::video(3)?)?),
        ] {
            let mut tampered: serde_json::Value = serde_json::from_slice(&encoded)?;
            tampered[field] = replacement;
            assert!(serde_json::from_value::<VaeIdentity>(tampered).is_err());
        }
        let mut tampered_patch: serde_json::Value = serde_json::from_slice(&encoded)?;
        tampered_patch["patch"]["base_artifact_digest"] = serde_json::Value::String("f".repeat(64));
        assert!(serde_json::from_value::<VaeIdentity>(tampered_patch).is_err());
        assert!(VaeArchitectureIdentity::checked("free form architecture").is_err());
        assert!(VaeArchitectureIdentity::checked("zed.vae.unregistered.v1").is_err());
        assert!(first.descriptor().is_conformance_only());
        Ok(())
    }

    #[test]
    fn media_and_structured_boundaries_are_checked_and_round_trip() -> Result<(), Box<dyn Error>> {
        let boundaries = [
            VaeBoundary::image(3)?,
            VaeBoundary::video(3)?,
            VaeBoundary::audio(2, 48_000)?,
            VaeBoundary::structured_output(16, VaeStructuredOutputKind::GaussianSplats)?,
        ];
        for boundary in boundaries {
            let encoded = serde_json::to_vec(&boundary)?;
            assert_eq!(serde_json::from_slice::<VaeBoundary>(&encoded)?, boundary);
        }

        assert!(matches!(
            serde_json::from_value::<VaeBoundary>(serde_json::json!({
                "kind": "image",
                "channels": 3,
                "sample_rate": 48_000,
                "structured_output": null
            })),
            Err(_)
        ));
        assert!(matches!(
            VaeBoundary::audio(2, 0),
            Err(VaeError::InvalidBoundary(VaeBoundaryKind::Audio))
        ));

        let image = VaeBoundary::image(3)?;
        assert!(matches!(
            image.validate_latent_dimensions(3),
            Err(VaeError::BoundaryLatentDimensionMismatch {
                kind: VaeBoundaryKind::Image,
                dimensions: 3
            })
        ));
        assert_eq!(VaeBoundary::audio(2, 44_100)?.sample_rate(), Some(44_100));
        assert_eq!(
            VaeBoundary::structured_output(8, VaeStructuredOutputKind::Shape)?.structured_kind(),
            Some(VaeStructuredOutputKind::Shape)
        );
        Ok(())
    }

    #[test]
    fn descriptor_and_kernel_reject_patch_dtype_device_and_boundary_mismatches()
    -> Result<(), Box<dyn Error>> {
        let artifact = artifact('a')?;
        let family = ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?;
        let architecture = VaeArchitectureIdentity::checked("zed.vae.boundary.v1")?;
        let mut wrong_patch = patch_identity(&artifact)?;
        wrong_patch.base_artifact_digest = "b".repeat(64);
        assert!(matches!(
            VaeDescriptor::checked(
                &artifact,
                family.clone(),
                &AFFINE_2D,
                architecture.clone(),
                wrong_patch,
                DType::F32,
                DeviceId::CPU,
                VaeBoundary::image(3)?,
                VaeKernelProfile::BlockAverageNearestV1,
                [0.0, 1.0],
            ),
            Err(VaeError::PatchArtifactMismatch)
        ));
        assert!(matches!(
            VaeDescriptor::checked(
                &artifact,
                family.clone(),
                &AFFINE_2D,
                architecture.clone(),
                patch_identity(&artifact)?,
                DType::I64,
                DeviceId::CPU,
                VaeBoundary::image(3)?,
                VaeKernelProfile::BlockAverageNearestV1,
                [0.0, 1.0],
            ),
            Err(VaeError::UnsupportedDType(DType::I64))
        ));
        let metal_descriptor = VaeDescriptor::checked(
            &artifact,
            family,
            &AFFINE_2D,
            architecture,
            patch_identity(&artifact)?,
            DType::F32,
            DeviceId::new(DeviceKind::Metal, 0),
            VaeBoundary::image(3)?,
            VaeKernelProfile::BlockAverageNearestV1,
            [0.0, 1.0],
        )?;
        assert!(matches!(
            NativeVae::checked(metal_descriptor, &AFFINE_2D),
            Err(VaeError::KernelExecutionBindingMismatch {
                expected_dtype: DType::F32,
                actual_dtype: DType::F32,
                expected_device: DeviceId::CPU,
                actual_device,
            }) if actual_device == DeviceId::new(DeviceKind::Metal, 0)
        ));
        assert!(matches!(
            checked_descriptor(
                &artifact,
                ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
                &AFFINE_2D,
                "zed.vae.boundary.v1",
                VaeKernelProfile::BlockAverageNearestV1,
                3,
                [0.0, 1.0],
            )?
            .identity()
            .boundary()
            .kind(),
            VaeBoundaryKind::Image
        ));
        assert!(matches!(
            VaeDescriptor::checked(
                &artifact,
                ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
                &AFFINE_2D,
                VaeArchitectureIdentity::checked("zed.vae.boundary.v1")?,
                patch_identity(&artifact)?,
                DType::F32,
                DeviceId::CPU,
                VaeBoundary::video(3)?,
                VaeKernelProfile::BlockAverageNearestV1,
                [0.0, 1.0],
            ),
            Err(VaeError::BoundaryLatentDimensionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn checked_tile_plan_exposes_effective_centered_crop_without_changing_input()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let vae = vae(&SD15_REDUCED, 1, [-1.0, 1.0])?;
        let pixels = upload(&backend, vec![1, 1, 11, 18], &vec![0.25; 198], &context)?;
        let before = pixels.contiguous_bytes()?.to_vec();
        let plan = vae.plan_encode(&pixels)?;
        assert_eq!(plan.pass_count(), 1);
        assert_eq!(plan.input_shape(), &[1, 1, 11, 18]);
        assert_eq!(plan.effective_input_spatial_shape(), &[8, 16]);
        assert_eq!(plan.input_spatial_offsets(), &[1, 1]);
        assert_eq!(plan.output_shape(), &[1, 4, 1, 2]);
        assert_eq!(pixels.contiguous_bytes()?, before);
        Ok(())
    }

    #[test]
    fn val_vae_001_one_dimensional_sequence_layout_crops_averages_and_decodes()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let vae = vae(&IDENTITY_1D, 1, [-100.0, 100.0])?;
        let pixels = upload(
            &backend,
            vec![1, 1, 5],
            &[0.0, 2.0, 4.0, 6.0, 100.0],
            &context,
        )?;
        let latent = vae.encode(&backend, &pixels, &context)?;
        assert_eq!(latent.descriptor().shape(), &[1, 2, 2]);
        assert_eq!(latent.descriptor().dtype(), DType::F32);
        assert_eq!(latent.descriptor().device(), DeviceId::CPU);
        assert_eq!(values(&latent)?, vec![1.0, 1.0, 5.0, 5.0]);

        let decoded = vae.decode(&backend, &latent, &context)?;
        assert_eq!(decoded.descriptor().shape(), &[1, 1, 4]);
        assert_eq!(values(&decoded)?, vec![1.0, 1.0, 5.0, 5.0]);
        Ok(())
    }

    #[test]
    fn val_vae_001_two_dimensional_affine_scaling_clamp_and_three_pass_tiling_are_exact()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let vae = vae(&AFFINE_2D, 1, [-1.0, 3.0])?;
        let pixels = upload(
            &backend,
            vec![1, 1, 4, 6],
            &(0..24).map(|value| value as f32).collect::<Vec<_>>(),
            &context,
        )?;
        let untiled = vae.encode(&backend, &pixels, &context)?;
        let plan = vae.plan_encode_tiles(&pixels, vec![8, 8], vec![2, 2])?;
        assert_eq!(plan.pass_count(), 3);
        assert_eq!(plan.tile_count(), 4);
        let tiled = vae.encode_tiled(&backend, &pixels, &plan, &context)?;
        assert_eq!(values(&tiled)?, values(&untiled)?);
        assert_eq!(untiled.descriptor().shape(), &[1, 2, 2, 3]);
        assert_eq!(
            values(&untiled)?,
            vec![
                5.0, 9.0, 13.0, 29.0, 33.0, 37.0, 5.0, 9.0, 13.0, 29.0, 33.0, 37.0
            ]
        );

        let full_decode = vae.decode(&backend, &untiled, &context)?;
        let decode_plan = vae.plan_decode_tiles(&untiled, vec![4, 4], vec![1, 1])?;
        assert_eq!(decode_plan.pass_count(), 3);
        let tiled_decode = vae.decode_tiled(&backend, &untiled, &decode_plan, &context)?;
        assert_eq!(values(&tiled_decode)?, values(&full_decode)?);
        assert!(
            values(&full_decode)?
                .iter()
                .all(|value| (-1.0..=3.0).contains(value))
        );
        Ok(())
    }

    #[test]
    fn three_dimensional_temporal_and_spatial_compression_is_exact() -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let vae = vae(&IDENTITY_3D, 1, [-100.0, 100.0])?;
        let input_values = (0..64).map(|value| value as f32).collect::<Vec<_>>();
        let pixels = upload(&backend, vec![1, 1, 4, 4, 4], &input_values, &context)?;
        let latent = vae.encode(&backend, &pixels, &context)?;
        assert_eq!(latent.descriptor().shape(), &[1, 1, 2, 2, 2]);
        assert_eq!(
            values(&latent)?,
            vec![10.5, 12.5, 18.5, 20.5, 42.5, 44.5, 50.5, 52.5]
        );
        let decoded = vae.decode(&backend, &latent, &context)?;
        assert_eq!(decoded.descriptor().shape(), &[1, 1, 3, 4, 4]);
        assert_eq!(values(&decoded)?[0], 10.5);
        assert_eq!(values(&decoded)?[47], 52.5);
        Ok(())
    }

    #[test]
    fn val_vae_001_causal_three_dimensional_tiles_match_full_execution()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let vae = vae(&IDENTITY_3D, 1, [-100.0, 100.0])?;
        let input_values = (0..96).map(|value| value as f32).collect::<Vec<_>>();
        let pixels = upload(&backend, vec![1, 1, 6, 4, 4], &input_values, &context)?;
        let full_latent = vae.encode(&backend, &pixels, &context)?;
        let encode_plan = vae.plan_encode_tiles(&pixels, vec![4, 4, 4], vec![2, 0, 0])?;
        assert_eq!(encode_plan.pass_count(), 1);
        assert_eq!(encode_plan.tile_count(), 2);
        let tiled_latent = vae.encode_tiled(&backend, &pixels, &encode_plan, &context)?;
        assert_eq!(tiled_latent.descriptor().shape(), &[1, 1, 3, 2, 2]);
        assert_eq!(values(&tiled_latent)?, values(&full_latent)?);

        let full_pixels = vae.decode(&backend, &full_latent, &context)?;
        let decode_plan = vae.plan_decode_tiles(&full_latent, vec![2, 2, 2], vec![1, 0, 0])?;
        assert_eq!(decode_plan.tile_count(), 2);
        let tiled_pixels = vae.decode_tiled(&backend, &full_latent, &decode_plan, &context)?;
        assert_eq!(tiled_pixels.descriptor().shape(), &[1, 1, 5, 4, 4]);
        assert_eq!(values(&tiled_pixels)?, values(&full_pixels)?);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn val_vae_001_source_channel_crop_replicate_and_constant_padding_are_exact()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;

        let excess = upload(
            &backend,
            vec![1, 3, 2],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            &context,
        )?;
        let cropped = prepare_pixel_channels(
            &backend,
            &excess,
            2,
            &VaeKernelProfile::BlockAverageNearestV1,
            &context,
        )?;
        assert_eq!(cropped.descriptor().shape(), &[1, 2, 2]);
        assert_eq!(read_f32(&cropped, &[0, 0, 0])?, 0.0);
        assert_eq!(read_f32(&cropped, &[0, 1, 1])?, 3.0);

        let missing = upload(&backend, vec![1, 1, 2], &[2.0, 3.0], &context)?;
        let replicated = prepare_pixel_channels(
            &backend,
            &missing,
            3,
            &VaeKernelProfile::AudioOobleck44KhzV1,
            &context,
        )?;
        assert_eq!(values(&replicated)?, vec![2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);

        let padded =
            prepare_pixel_channels(&backend, &missing, 3, &VaeKernelProfile::Wan21V1, &context)?;
        assert_eq!(values(&padded)?, vec![2.0, 3.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn val_vae_001_extra_channel_audio_tiles_reshape_at_the_kernel_boundary()
    -> Result<(), Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        write_music_safetensors(&directory.path().join("music.safetensors"))?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "vae",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let key = ArtifactKey::new("models", "music.safetensors")?;
        let record = index.record(&key).cloned().ok_or("missing music VAE")?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(&index, &key, &cancellation)?;
        let definition = GENERATED_LATENT_FORMATS
            .iter()
            .find(|definition| definition.identifier == "ACEAudio")
            .ok_or("ACEAudio latent format is missing")?;
        let descriptor = VaeDescriptor::checked(
            &record,
            ModelFamilyIdentity::new("COMFY-MODEL-0061", "ACEStep", "ace-step-transformer-2d-v1")?,
            definition,
            VaeArchitectureIdentity::checked("sim.ldm.ace.vae.MusicDCAE.v1")?,
            patch_identity(&record)?,
            DType::F32,
            DeviceId::CPU,
            VaeBoundary::audio(2, 44_100)?,
            VaeKernelProfile::MusicDcaeV1,
            [-1.0, 1.0],
        )?;
        let (backend, _authority, context) = backend_and_context(&cancellation, 2 << 20)?;
        let module =
            NativeModule::buffer("music_vae", upload(&backend, vec![1], &[1.0], &context)?)?;
        let binding = VaeModelBinding::checked(&descriptor, &store, loaded, module, &cancellation)?;
        let architecture = descriptor.identity().architecture().clone();
        let vae = NativeVae::checked_kernel(
            descriptor,
            definition,
            binding,
            VaeKernelFunctions::checked(
                architecture,
                native_extra_audio_encode,
                native_extra_audio_decode,
            ),
        )?;

        let samples = upload(&backend, vec![1, 1, 8_192], &vec![0.25; 8_192], &context)?;
        let full_latent = vae.encode(&backend, &samples, &context)?;
        let encode_plan = vae.plan_encode_tiles(&samples, vec![4_096], vec![0])?;
        assert_eq!(encode_plan.tile_count(), 2);
        let tiled_latent = vae.encode_tiled(&backend, &samples, &encode_plan, &context)?;
        assert_eq!(tiled_latent.descriptor().shape(), &[1, 8, 16, 2]);
        assert_eq!(values(&tiled_latent)?, values(&full_latent)?);

        let full_samples = vae.decode(&backend, &full_latent, &context)?;
        let decode_plan = vae.plan_decode_tiles(&full_latent, vec![1], vec![0])?;
        assert_eq!(decode_plan.tile_count(), 2);
        let tiled_samples = vae.decode_tiled(&backend, &full_latent, &decode_plan, &context)?;
        assert_eq!(tiled_samples.descriptor().shape(), &[1, 2, 8_192]);
        assert_eq!(values(&tiled_samples)?, values(&full_samples)?);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn val_vae_001_invalid_descriptors_shapes_and_tiles_fail_closed() -> Result<(), Box<dyn Error>>
    {
        assert!(matches!(
            VaeBoundary::image(0),
            Err(VaeError::InvalidPixelChannels(0))
        ));
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let vae = vae(&AFFINE_2D, 1, [-1.0, 1.0])?;
        let pixels = upload(&backend, vec![1, 1, 4, 4], &[0.0; 16], &context)?;
        assert!(matches!(
            vae.plan_encode_tiles(&pixels, vec![2], vec![0]),
            Err(VaeError::TileRank { .. })
        ));
        assert!(matches!(
            vae.plan_encode_tiles(&pixels, vec![2, 2], vec![2, 0]),
            Err(VaeError::InvalidTileDimension { dimension: 0, .. })
        ));
        let too_small = upload(&backend, vec![1, 1, 1, 4], &[0.0; 4], &context)?;
        assert!(matches!(
            vae.encode(&backend, &too_small, &context),
            Err(VaeError::InvalidShape { .. })
        ));
        Ok(())
    }

    #[test]
    fn val_vae_001_plans_are_bound_to_operation_identity_and_input_shape()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let first = vae(&AFFINE_2D, 1, [-1.0, 1.0])?;
        let second_artifact = artifact('b')?;
        let second_descriptor = checked_descriptor(
            &second_artifact,
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
            &AFFINE_2D,
            "zed.vae.block_average_nearest.v1",
            VaeKernelProfile::BlockAverageNearestV1,
            1,
            [-1.0, 1.0],
        )?;
        let second = NativeVae::checked(second_descriptor, &AFFINE_2D)?;
        let pixels = upload(&backend, vec![1, 1, 4, 4], &[0.0; 16], &context)?;
        let different_shape = upload(&backend, vec![1, 1, 4, 6], &[0.0; 24], &context)?;
        let plan = first.plan_encode_tiles(&pixels, vec![4, 4], vec![0, 0])?;
        assert!(matches!(
            second.encode_tiled(&backend, &pixels, &plan, &context),
            Err(VaeError::TilePlanIdentityMismatch)
        ));
        assert!(matches!(
            first.encode_tiled(&backend, &different_shape, &plan, &context),
            Err(VaeError::TilePlanShapeMismatch { .. })
        ));
        let latent = first.encode(&backend, &pixels, &context)?;
        assert!(matches!(
            first.decode_tiled(&backend, &latent, &plan, &context),
            Err(VaeError::TilePlanOperationMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn canonical_model_and_native_module_binding_owns_execution_state() -> Result<(), Box<dyn Error>>
    {
        let directory = tempfile::tempdir()?;
        write_safetensors(&directory.path().join("model.safetensors"))?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "vae",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let key = ArtifactKey::new("models", "model.safetensors")?;
        let record = index.record(&key).cloned().ok_or("missing indexed model")?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(&index, &key, &cancellation)?;
        let descriptor = checked_descriptor(
            &record,
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
            &AFFINE_2D,
            "zed.vae.boundary.v1",
            VaeKernelProfile::BlockAverageNearestV1,
            3,
            [0.0, 1.0],
        )?;
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let first_module =
            NativeModule::buffer("vae", upload(&backend, vec![1], &[1.0], &context)?)?;
        let second_module =
            NativeModule::buffer("vae", upload(&backend, vec![1], &[2.0], &context)?)?;
        let first = VaeModelBinding::checked(
            &descriptor,
            &store,
            loaded.clone(),
            first_module,
            &cancellation,
        )?;
        let second = VaeModelBinding::checked(
            &descriptor,
            &store,
            loaded.clone(),
            second_module,
            &cancellation,
        )?;
        assert_eq!(first.identity(), descriptor.identity());
        assert_ne!(first.digest(), second.digest());

        let functions = VaeKernelFunctions::checked(
            descriptor.identity().architecture().clone(),
            native_cpu_projection_encode,
            native_cpu_projection_decode,
        );
        let native = NativeVae::checked_kernel(descriptor.clone(), &AFFINE_2D, first, functions)?;
        let pixels = upload(&backend, vec![1, 3, 4, 4], &[0.25; 48], &context)?;
        let latent = native.encode(&backend, &pixels, &context)?;
        assert_eq!(latent.descriptor().shape(), &[1, 2, 2, 2]);
        let decoded = native.decode(&backend, &latent, &context)?;
        assert_eq!(decoded.descriptor().shape(), &[1, 3, 4, 4]);

        let (delegated_inner, delegated_authority) =
            CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let delegated = DelegatingCpuBackend {
            inner: delegated_inner,
        };
        let delegated_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: delegated_authority.authorize_workspace(1 << 20)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let delegated_pixels = upload(
            &delegated.inner,
            vec![1, 3, 4, 4],
            &[0.25; 48],
            &delegated_context,
        )?;
        let before = delegated.inner.memory_snapshot().current_bytes;
        assert!(matches!(
            native.encode(&delegated, &delegated_pixels, &delegated_context),
            Err(VaeError::StageCRequiresCpuBackend)
        ));
        assert_eq!(delegated.inner.memory_snapshot().current_bytes, before);

        let foreign_store = ModelStore::new(ParserLimits::default())?;
        assert!(matches!(
            VaeModelBinding::checked(
                &descriptor,
                &foreign_store,
                loaded.clone(),
                NativeModule::buffer("vae", upload(&backend, vec![1], &[1.0], &context)?,)?,
                &cancellation,
            ),
            Err(VaeError::ModelStore(ModelStoreError::ForeignModelHandle))
        ));

        let mut wrong_path_artifact = record.clone();
        wrong_path_artifact.key = ArtifactKey::new("models", "other.safetensors")?;
        let wrong_path_descriptor = checked_descriptor(
            &wrong_path_artifact,
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
            &AFFINE_2D,
            "zed.vae.boundary.v1",
            VaeKernelProfile::BlockAverageNearestV1,
            3,
            [0.0, 1.0],
        )?;
        assert!(matches!(
            VaeModelBinding::checked(
                &wrong_path_descriptor,
                &store,
                loaded.clone(),
                NativeModule::buffer("vae", upload(&backend, vec![1], &[1.0], &context)?)?,
                &cancellation,
            ),
            Err(VaeError::ModelStore(
                ModelStoreError::LoadedArtifactIdentityMismatch { .. }
            ))
        ));

        let mut wrong_sha_artifact = record;
        wrong_sha_artifact.sha256 = "f".repeat(64);
        let wrong_sha_descriptor = checked_descriptor(
            &wrong_sha_artifact,
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
            &AFFINE_2D,
            "zed.vae.boundary.v1",
            VaeKernelProfile::BlockAverageNearestV1,
            3,
            [0.0, 1.0],
        )?;
        assert!(matches!(
            VaeModelBinding::checked(
                &wrong_sha_descriptor,
                &store,
                loaded,
                NativeModule::buffer("vae", upload(&backend, vec![1], &[1.0], &context)?)?,
                &cancellation,
            ),
            Err(VaeError::ModelStore(
                ModelStoreError::LoadedArtifactIdentityMismatch { .. }
            ))
        ));
        Ok(())
    }

    #[test]
    fn source_selected_model_binds_native_module_and_rejects_wrong_source_and_family()
    -> Result<(), Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        write_cog_safetensors(&directory.path().join("cog.safetensors"))?;
        write_taesd_safetensors(&directory.path().join("taesd.safetensors"))?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "vae",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let cog_key = ArtifactKey::new("models", "cog.safetensors")?;
        let taesd_key = ArtifactKey::new("models", "taesd.safetensors")?;
        let cog_record = index
            .record(&cog_key)
            .cloned()
            .ok_or("missing Cog artifact")?;
        let taesd_record = index
            .record(&taesd_key)
            .cloned()
            .ok_or("missing TAESD artifact")?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let cog_loaded = store.load(&index, &cog_key, &cancellation)?;
        let taesd_loaded = store.load(&index, &taesd_key, &cancellation)?;
        let architecture_registry = VaeArchitectureRegistry::checked()?;
        let family_registry =
            ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
        let latent_registry = LatentFormatRegistry::checked(GENERATED_LATENT_FORMATS)?;
        let family = family_registry
            .definitions_in_source_order()
            .into_iter()
            .find(|definition| definition.identifier == "CogVideoX_T2V")
            .ok_or("CogVideoX family missing")?;
        let target = VaeExecutionTarget::new(
            ModelFamilyIdentity::new(
                family.feature_id,
                family.identifier,
                family.architecture_version,
            )?,
            LatentFormatIdentity::new(family.latent_feature_id, family.latent_identifier)?,
            DType::F32,
            DeviceId::CPU,
        );
        let selection = architecture_registry.select_loaded(
            &store,
            &cog_loaded,
            &target,
            &family_registry,
            &latent_registry,
            &cancellation,
        )?;
        assert_eq!(selection.profile(), &VaeKernelProfile::CogVideoXV1);
        let descriptor = VaeDescriptor::checked_selection(
            &cog_record,
            &selection,
            &target,
            &family_registry,
            &latent_registry,
            patch_identity(&cog_record)?,
            VaeBoundary::video(3)?,
            [0.0, 1.0],
            &cancellation,
        )?;
        let mut wrong_boundary_identity = descriptor.identity().clone();
        wrong_boundary_identity.boundary = VaeBoundary::image(3)?;
        wrong_boundary_identity.digest = wrong_boundary_identity.compute_digest()?;
        let encoded = serde_json::to_vec(&wrong_boundary_identity)?;
        let error = serde_json::from_slice::<VaeIdentity>(&encoded)
            .expect_err("persisted production identity must reject a recomputed wrong boundary");
        assert!(error.to_string().contains("requires boundary Video"));
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let binding = VaeModelBinding::checked(
            &descriptor,
            &store,
            cog_loaded.clone(),
            NativeModule::buffer("cog_vae", upload(&backend, vec![1], &[1.0], &context)?)?,
            &cancellation,
        )?;
        assert_eq!(binding.identity(), descriptor.identity());
        assert!(matches!(
            VaeModelBinding::checked(
                &descriptor,
                &store,
                cog_loaded,
                NativeModule::container("empty_cog_vae")?,
                &cancellation,
            ),
            Err(VaeError::NativeModuleHasNoState)
        ));

        let wrong_source_descriptor = VaeDescriptor::checked_selection(
            &taesd_record,
            &selection,
            &target,
            &family_registry,
            &latent_registry,
            patch_identity(&taesd_record)?,
            VaeBoundary::video(3)?,
            [0.0, 1.0],
            &cancellation,
        )?;
        assert!(matches!(
            VaeModelBinding::checked(
                &wrong_source_descriptor,
                &store,
                taesd_loaded,
                NativeModule::buffer("wrong_source", upload(&backend, vec![1], &[1.0], &context)?,)?,
                &cancellation,
            ),
            Err(VaeError::Architecture(
                VaeArchitectureError::ProfileLatentMismatch { .. }
            ))
        ));

        let wrong_family = family_registry
            .definitions_in_source_order()
            .into_iter()
            .find(|definition| definition.latent_identifier != family.latent_identifier)
            .ok_or("wrong-family fixture missing")?;
        let wrong_target = VaeExecutionTarget::new(
            ModelFamilyIdentity::new(
                wrong_family.feature_id,
                wrong_family.identifier,
                wrong_family.architecture_version,
            )?,
            LatentFormatIdentity::new(
                wrong_family.latent_feature_id,
                wrong_family.latent_identifier,
            )?,
            DType::F32,
            DeviceId::CPU,
        );
        assert!(matches!(
            architecture_registry.validate_target(
                &selection,
                &wrong_target,
                &family_registry,
                &latent_registry,
                &cancellation,
            ),
            Err(VaeArchitectureError::ProfileLatentMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn native_kernel_uses_only_bound_module_state_and_checks_every_output_contract()
    -> Result<(), Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        write_safetensors(&directory.path().join("model.safetensors"))?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "vae",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let key = ArtifactKey::new("models", "model.safetensors")?;
        let record = index.record(&key).cloned().ok_or("missing indexed model")?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(&index, &key, &cancellation)?;
        let descriptor = checked_descriptor(
            &record,
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
            &AFFINE_2D,
            "zed.vae.boundary.v1",
            VaeKernelProfile::BlockAverageNearestV1,
            2,
            [0.0, 1.0],
        )?;
        let (backend, _authority, context) = backend_and_context(&cancellation, 1 << 20)?;
        let module = NativeModule::buffer("vae", upload(&backend, vec![1], &[2.0], &context)?)?;
        let binding = VaeModelBinding::checked(&descriptor, &store, loaded, module, &cancellation)?;
        let architecture = descriptor.identity().architecture().clone();
        let native = NativeVae::checked_kernel(
            descriptor.clone(),
            &AFFINE_2D,
            binding.clone(),
            VaeKernelFunctions::checked(
                architecture.clone(),
                native_test_encode,
                native_test_decode,
            ),
        )?;
        let pixels = upload(&backend, vec![1, 2, 4, 4], &[0.25; 32], &context)?;
        let latent = native.encode(&backend, &pixels, &context)?;
        assert_eq!(latent.descriptor().shape(), &[1, 2, 2, 2]);
        assert!(values(&latent)?.iter().all(|value| *value == 2.0));
        let decoded = native.decode(&backend, &latent, &context)?;
        assert_eq!(decoded.descriptor().shape(), &[1, 2, 4, 4]);
        assert!(values(&decoded)?.iter().all(|value| *value == 1.0));

        let wrong_encode = NativeVae::checked_kernel(
            descriptor,
            &AFFINE_2D,
            binding.clone(),
            VaeKernelFunctions::checked(architecture, native_wrong_shape, native_test_decode),
        )?;
        assert!(matches!(
            wrong_encode.encode(&backend, &pixels, &context),
            Err(VaeError::KernelOutputContractMismatch {
                operation: VaeOperation::Encode,
                ..
            })
        ));

        for invalid_encode in [native_wrong_dtype as VaeKernelFunction, native_wrong_stream] {
            let invalid = NativeVae::checked_kernel(
                native.descriptor().clone(),
                &AFFINE_2D,
                binding.clone(),
                VaeKernelFunctions::checked(
                    native.descriptor().identity().architecture().clone(),
                    invalid_encode,
                    native_test_decode,
                ),
            )?;
            assert!(matches!(
                invalid.encode(&backend, &pixels, &context),
                Err(VaeError::KernelOutputContractMismatch {
                    operation: VaeOperation::Encode,
                    ..
                })
            ));
        }

        let wrong_decode = NativeVae::checked_kernel(
            native.descriptor().clone(),
            &AFFINE_2D,
            binding,
            VaeKernelFunctions::checked(
                native.descriptor().identity().architecture().clone(),
                native_test_encode,
                native_wrong_shape,
            ),
        )?;
        assert!(matches!(
            wrong_decode.decode(&backend, &latent, &context),
            Err(VaeError::KernelOutputContractMismatch {
                operation: VaeOperation::Decode,
                ..
            })
        ));

        let expected_refiner_shape = processed_encode_shape(&HUNYUAN_REFINER_3D, &[1, 1, 3, 2, 2])?;
        assert_eq!(expected_refiner_shape, [1, 2, 2, 2, 2]);
        let wrong_refiner_shape = upload(&backend, vec![1, 2, 3, 2, 2], &[0.0; 24], &context)?;
        assert!(matches!(
            validate_kernel_output(
                VaeOperation::Encode,
                &wrong_refiner_shape,
                &expected_refiner_shape,
                native.descriptor().identity(),
                context.stream,
            ),
            Err(VaeError::KernelOutputContractMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn non_identity_latent_transforms_reject_non_f32_execution() -> Result<(), Box<dyn Error>> {
        let record = artifact('a')?;
        let descriptor = VaeDescriptor::checked(
            &record,
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "vae_test_family", "v1")?,
            &AFFINE_2D,
            VaeArchitectureIdentity::checked("zed.vae.boundary.v1")?,
            patch_identity(&record)?,
            DType::F16,
            DeviceId::CPU,
            VaeBoundary::image(2)?,
            VaeKernelProfile::BlockAverageNearestV1,
            [0.0, 1.0],
        );
        assert!(matches!(
            descriptor,
            Err(VaeError::LatentTransformDTypeMismatch {
                transform: LatentTransform::Affine,
                dtype: DType::F16,
            })
        ));
        Ok(())
    }

    #[test]
    fn latent_upscale_model_ltx_statistics_are_typed_source_exact_and_atomic()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority, context) = backend_and_context(&cancellation, 8 << 20)?;
        let mut means = [0.0_f32; 128];
        means[..4].copy_from_slice(&[10.0, -2.0, 1.0, 0.0]);
        let mut standard_deviations = [1.0_f32; 128];
        standard_deviations[..4].copy_from_slice(&[2.0, 4.0, 0.5, 8.0]);
        let vae = ltx_statistics_vae(&backend, &context, &means, &standard_deviations)?;
        let mut input_values = [0.0_f32; 128];
        input_values[..4].copy_from_slice(&[-1.0, 0.5, 2.0, 4.0]);
        let input = upload(&backend, vec![1, 128, 1, 1, 1], &input_values, &context)?;
        let input_version = input.mutation_version();
        let unnormalized = vae.apply_ltx_latent_statistics(
            &backend,
            &input,
            LtxLatentStatisticsDirection::Unnormalize,
            &context,
        )?;
        assert_eq!(&values(&unnormalized)?[..4], &[8.0, 0.0, 2.0, 32.0]);

        let mut model_raw = [0.0_f32; 128];
        model_raw[..4].copy_from_slice(&[12.0, 2.0, 1.5, 16.0]);
        let model_raw = upload(&backend, vec![1, 128, 1, 1, 1], &model_raw, &context)?;
        let normalized = vae.apply_ltx_latent_statistics(
            &backend,
            &model_raw,
            LtxLatentStatisticsDirection::Normalize,
            &context,
        )?;
        assert_eq!(&values(&normalized)?[..4], &[1.0, 1.0, 1.0, 2.0]);
        assert_eq!(input.mutation_version(), input_version);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: context.stream,
            scratch: authority.authorize_workspace(1 << 20)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            vae.apply_ltx_latent_statistics(
                &backend,
                &input,
                LtxLatentStatisticsDirection::Unnormalize,
                &cancelled_context,
            ),
            Err(VaeError::Tensor(TensorError::Cancelled))
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

        let (small_backend, small_authority) = CpuWorkspaceAuthority::create_backend(2_048)?;
        let small_context = small_backend.execution_context(
            StreamId::DEFAULT,
            small_authority.authorize_workspace(2_048)?,
            &cancellation,
        );
        let small_vae =
            ltx_statistics_vae(&small_backend, &small_context, &means, &standard_deviations)?;
        let small_input = upload(
            &small_backend,
            vec![1, 128, 1, 1, 1],
            &input_values,
            &small_context,
        )?;
        let tiny_error = small_vae
            .apply_ltx_latent_statistics(
                &small_backend,
                &small_input,
                LtxLatentStatisticsDirection::Unnormalize,
                &small_context,
            )
            .expect_err("tiny LTX statistics workspace must fail");
        assert!(
            matches!(
                tiny_error,
                VaeError::Tensor(TensorError::AllocationFailed { .. }) | VaeError::NativeOps(_)
            ),
            "unexpected tiny-workspace error: {tiny_error:?}"
        );
        assert_eq!(small_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn latent_upscale_model_ltx_bundle_executes_statistics_graph_and_field_projection()
    -> Result<(), Box<dyn Error>> {
        use sim_tensor::{NativeLatentBundle, NativeLatentMetadata};

        let memory = 64 << 20;
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, memory)?;
        let means = [0.0_f32; 128];
        let standard_deviations = [1.0_f32; 128];
        let vae = ltx_statistics_vae(&backend, &context, &means, &standard_deviations)?;
        let mut ordered_state = Vec::new();
        let mut add = |name: &str, shape: Vec<u64>, fill: f32| -> Result<(), Box<dyn Error>> {
            let count = shape.iter().try_fold(1_usize, |count, dimension| {
                count.checked_mul(usize::try_from(*dimension).ok()?)
            });
            let count = count.ok_or("LTX bundle fixture shape overflowed")?;
            ordered_state.push((
                name.to_owned(),
                upload(&backend, shape, &vec![fill; count], &context)?,
            ));
            Ok(())
        };
        add("initial_conv.weight", vec![32, 128, 3, 3], 0.0)?;
        add("initial_conv.bias", vec![32], 0.0)?;
        add("initial_norm.weight", vec![32], 1.0)?;
        add("initial_norm.bias", vec![32], 0.0)?;
        for family in ["res_blocks", "post_upsample_res_blocks"] {
            for convolution in ["conv1", "conv2"] {
                add(
                    &format!("{family}.0.{convolution}.weight"),
                    vec![32, 32, 3, 3],
                    0.0,
                )?;
                add(&format!("{family}.0.{convolution}.bias"), vec![32], 0.0)?;
            }
            for normalization in ["norm1", "norm2"] {
                add(&format!("{family}.0.{normalization}.weight"), vec![32], 1.0)?;
                add(&format!("{family}.0.{normalization}.bias"), vec![32], 0.0)?;
            }
        }
        add("upsampler.0.weight", vec![128, 32, 3, 3], 0.0)?;
        add("upsampler.0.bias", vec![128], 0.0)?;
        add("final_conv.weight", vec![128, 32, 3, 3], 0.0)?;
        add("final_conv.bias", vec![128], 0.0)?;
        let configuration = serde_json::json!({
            "in_channels": 128,
            "mid_channels": 32,
            "num_blocks_per_stage": 1,
            "dims": 2,
            "spatial_upsample": true,
            "temporal_upsample": false,
            "spatial_scale": 7.25,
            "rational_resampler": false
        });
        let resource = crate::NativeLatentUpscaleModelResource::from_checkpoint(
            crate::NativeLatentUpscaleCheckpoint {
                artifact_sha256: "7".repeat(64),
                metadata: BTreeMap::from([(
                    "config".to_owned(),
                    serde_json::to_string(&configuration)?,
                )]),
                ordered_state,
                memory_budget_bytes: memory,
            },
            &context,
        )?;
        let samples = upload(
            &backend,
            vec![1, 128, 1, 2, 2],
            &vec![0.25; 128 * 4],
            &context,
        )?;
        let noise_mask = upload(&backend, vec![1, 1, 1, 2, 2], &[1.0; 4], &context)?;
        let metadata = NativeLatentMetadata::checked(None, Some(48_000), Some(8), Some(4))?;
        let original_version = samples.mutation_version();
        let original = NativeLatentBundle::single(
            samples.clone(),
            Some(noise_mask.clone()),
            Some(vec![17]),
            metadata.clone(),
            &context,
        )?;
        for batch_indices in [vec![17], vec![], vec![17, 18, 19]] {
            let input = original.replaced_samples_mask_and_indices_source_dictionary(
                sim_tensor::NativeLatentSamples::Tensor(samples.clone()),
                Some(sim_tensor::NativeLatentNoiseMask::Tensor(
                    noise_mask.clone(),
                )),
                Some(batch_indices.clone()),
                &context,
            )?;
            let output = resource.invoke_ltx_bundle(&backend, &input, &vae, &context)?;
            let sim_tensor::NativeLatentSamples::Tensor(output_samples) = output.samples() else {
                return Err("LTX bundle produced nested samples".into());
            };
            assert_eq!(output_samples.descriptor().shape(), &[1, 128, 1, 4, 4]);
            assert_eq!(output.batch_indices(), Some(batch_indices.as_slice()));
            assert_eq!(output.metadata(), &metadata);
            assert!(output.noise_mask().is_none());
            assert!(input.noise_mask().is_some());
            assert_eq!(input.batch_indices(), Some(batch_indices.as_slice()));
            assert_eq!(samples.mutation_version(), original_version);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        Ok(())
    }

    #[test]
    fn val_vae_001_cancellation_and_oom_do_not_mutate_input_or_leave_workspace_leased()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, upload_context) = backend_and_context(&cancellation, 256)?;
        let vae = vae(&AFFINE_2D, 1, [-1.0, 1.0])?;
        let pixels = upload(&backend, vec![1, 1, 4, 4], &[0.0; 16], &upload_context)?;
        let input_version = pixels.mutation_version();

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: upload_context.scratch,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            vae.encode(&backend, &pixels, &cancelled_context),
            Err(VaeError::Tensor(TensorError::Cancelled))
                | Err(VaeError::NativeOps(NativeOpsError::Cancelled))
        ));
        assert_eq!(pixels.mutation_version(), input_version);
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

        let oom_cancellation = CancellationToken::default();
        let (small_backend, _small_authority, small_context) =
            backend_and_context(&oom_cancellation, 80)?;
        let small_pixels = upload(&small_backend, vec![1, 1, 4, 4], &[0.0; 16], &small_context)?;
        let small_version = small_pixels.mutation_version();
        assert!(matches!(
            vae.encode(&small_backend, &small_pixels, &small_context),
            Err(VaeError::Tensor(TensorError::AllocationFailed { .. }))
        ));
        assert_eq!(small_pixels.mutation_version(), small_version);
        assert_eq!(small_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn val_vae_001_learned_sd15_encoder_decoder_are_weighted_deterministic_and_not_block_averaging()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 32 << 20)?;
        let artifact_digest = "c".repeat(64);
        let learned = learned_vae(&backend, &context, 3, &artifact_digest)?;
        let block_artifact = artifact('a')?;
        let block_descriptor = checked_descriptor(
            &block_artifact,
            ModelFamilyIdentity::new("COMFY-MODEL-0117", "SD15", "sd15-v1")?,
            &SD15_REDUCED,
            "zed.vae.block_average_nearest.v1",
            VaeKernelProfile::BlockAverageNearestV1,
            3,
            [0.0, 1.0],
        )?;
        let block = NativeVae::checked(block_descriptor, &SD15_REDUCED)?;
        let input_values = (0..(3 * 16 * 16))
            .map(|index| ((index * 37 % 251) as f32) / 250.0)
            .collect::<Vec<_>>();
        let pixels = upload(&backend, vec![1, 3, 16, 16], &input_values, &context)?;

        let first = learned.encode_sd15_conformance(&backend, &pixels, &context)?;
        let second = learned.encode_sd15_conformance(&backend, &pixels, &context)?;
        let averaged = block.encode(&backend, &pixels, &context)?;
        assert_eq!(first.descriptor().shape(), &[1, 4, 2, 2]);
        assert_eq!(first.contiguous_bytes()?, second.contiguous_bytes()?);
        assert_ne!(first.contiguous_bytes()?, averaged.contiguous_bytes()?);
        assert!(values(&first)?.iter().all(|value| value.is_finite()));

        let decoded = learned.decode_sd15_conformance(&backend, &first, &context)?;
        assert_eq!(decoded.descriptor().shape(), &[1, 3, 16, 16]);
        let repeated = learned.decode_sd15_conformance(&backend, &first, &context)?;
        assert_eq!(decoded.contiguous_bytes()?, repeated.contiguous_bytes()?);
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../sim_test_support/fixtures/models/vae-image/learned-sd15-reduced.json"
        ))?;
        let absolute = oracle["tolerance"]["absolute"]
            .as_f64()
            .ok_or("absolute tolerance")?;
        let relative = oracle["tolerance"]["relative"]
            .as_f64()
            .ok_or("relative tolerance")?;
        for (name, tensor) in [("encoded", &first), ("decoded", &decoded)] {
            let expected = oracle[name]["values"].as_array().ok_or("oracle values")?;
            let actual = values(tensor)?;
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let expected = expected.as_f64().ok_or("oracle scalar")?;
                let error = (f64::from(*actual) - expected).abs();
                assert!(
                    error <= absolute + relative * expected.abs(),
                    "{name}[{index}]: native {actual}, source {expected}, error {error}"
                );
            }
        }
        assert!(
            values(&decoded)?
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );

        let encode_plan = learned.plan_encode_tiles(&pixels, vec![8, 8], vec![0, 0])?;
        assert!(matches!(
            learned.encode_tiled(&backend, &pixels, &encode_plan, &context),
            Err(VaeError::ConformanceHarnessRequiresCpuEntryPoint)
        ));
        let decode_plan = learned.plan_decode_tiles(&first, vec![2, 2], vec![0, 0])?;
        assert!(matches!(
            learned.decode_tiled(&backend, &first, &decode_plan, &context),
            Err(VaeError::ConformanceHarnessRequiresCpuEntryPoint)
        ));
        Ok(())
    }

    #[test]
    fn val_vae_001_learned_sd15_execution_identity_binds_weights_architecture_and_artifact()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, 32 << 20)?;
        let digest = "d".repeat(64);
        let first = learned_vae(&backend, &context, 1, &digest)?;
        let second = learned_vae(&backend, &context, 2, &digest)?;
        assert_ne!(first.execution_digest(), second.execution_digest());
        assert_eq!(first.resident_bytes()?, second.resident_bytes()?);
        assert!(first.resident_storage_bytes()? > 0);
        assert!(first.resident_bytes()? > first.resident_storage_bytes()?);
        let pixels = upload(&backend, vec![1, 3, 16, 16], &[0.25; 3 * 16 * 16], &context)?;
        let first_plan = first.plan_encode_tiles(&pixels, vec![2, 2], vec![0, 0])?;
        assert!(matches!(
            second.encode_tiled(&backend, &pixels, &first_plan, &context),
            Err(VaeError::TilePlanIdentityMismatch)
        ));
        assert_eq!(first.descriptor().identity().artifact_sha256(), digest);
        assert_eq!(
            first.descriptor().identity().architecture().as_str(),
            "sim.ldm.models.autoencoder.AutoencoderKL.reduced.v1"
        );

        let mut wrong_artifact = artifact('e')?;
        wrong_artifact.sha256 = "e".repeat(64);
        let wrong_descriptor = checked_descriptor(
            &wrong_artifact,
            ModelFamilyIdentity::new("COMFY-MODEL-0117", "SD15", "sd15-v1")?,
            &SD15_REDUCED,
            "sim.ldm.models.autoencoder.AutoencoderKL.reduced.v1",
            VaeKernelProfile::Sd15AutoencoderKlReducedV1,
            3,
            [0.0, 1.0],
        )?;
        let wrong_kernel = Sd15LearnedVaeKernel::checked(
            wrong_descriptor.identity().clone(),
            learned_weights(&backend, &context, 4)?,
        )?;
        assert!(matches!(
            NativeVae::checked_sd15(first.descriptor().clone(), &SD15_REDUCED, wrong_kernel),
            Err(VaeError::KernelIdentityBindingMismatch)
        ));
        Ok(())
    }

    #[test]
    fn val_vae_001_learned_sd15_cancellation_oom_and_failure_are_atomic()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority, context) = backend_and_context(&cancellation, 32 << 20)?;
        let learned = learned_vae(&backend, &context, 5, &"f".repeat(64))?;
        let pixels = upload(&backend, vec![1, 3, 16, 16], &[0.25; 3 * 16 * 16], &context)?;
        let input_version = pixels.mutation_version();
        let execution_digest = learned.execution_digest();
        let baseline = learned.encode_sd15_conformance(&backend, &pixels, &context)?;
        let baseline_bytes = baseline.contiguous_bytes()?.to_vec();
        let converged_peak = context.scratch.peak_bytes();
        assert!(converged_peak > 0);
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1 << 20)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            learned.encode_sd15_conformance(&backend, &pixels, &cancelled_context),
            Err(VaeError::Tensor(TensorError::Cancelled))
                | Err(VaeError::NativeOps(NativeOpsError::Cancelled))
                | Err(VaeError::NativeTensor(NativeDiffusionTensorError::Tensor(
                    TensorError::Cancelled
                )))
        ));
        assert_eq!(pixels.mutation_version(), input_version);
        assert_eq!(learned.execution_digest(), execution_digest);
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

        let oom_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(8)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        assert!(
            learned
                .encode_sd15_conformance(&backend, &pixels, &oom_context)
                .is_err()
        );
        assert_eq!(pixels.mutation_version(), input_version);
        assert_eq!(learned.execution_digest(), execution_digest);
        assert_eq!(oom_context.scratch.in_use_bytes(), 0);

        let caller_retry = learned.encode_sd15_conformance(&backend, &pixels, &context)?;
        assert_eq!(caller_retry.contiguous_bytes()?, baseline_bytes);
        assert_eq!(pixels.mutation_version(), input_version);
        assert_eq!(learned.execution_digest(), execution_digest);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(context.scratch.peak_bytes(), converged_peak);
        Ok(())
    }

    #[test]
    fn val_vae_001_kernel_boundary_has_no_retry_loop_or_private_workspace_authority() {
        let production = include_str!("vae.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .unwrap_or_default();
        assert!(!production.contains("CpuWorkspaceAuthority"));
        assert!(!production.contains("authorize_workspace"));
        assert!(!production.contains("retry"));
        assert!(!production.contains("attempt"));
        assert!(production.contains("ExecutionContext<'_>"));
    }

    #[test]
    fn generic_native_vae_rejects_structured_decode_boundaries() -> Result<(), VaeError> {
        let structured =
            VaeBoundary::structured_output(1, VaeStructuredOutputKind::GaussianSplats)?;
        assert!(matches!(
            validate_generic_vae_boundary(&structured),
            Err(VaeError::StructuredDecodeRequired)
        ));
        validate_generic_vae_boundary(&VaeBoundary::image(3)?)?;
        validate_generic_vae_boundary(&VaeBoundary::audio(2, 48_000)?)?;
        Ok(())
    }
}
