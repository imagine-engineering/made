use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_media::{
    NativeArtifactKind, NativeArtifactPayload, NativeAudioPayload, NativeCameraPayload,
    NativeCameraProjection, NativeCameraRole, NativeFile3DFormat, NativeFile3DPayload,
    NativeFile3DRole, NativeVideoBitDepth, NativeVideoPayload,
};
use sim_nodes::{NativeHandleKind, NativeHandleType, NativeNodeContext, NativeStoredPayload};
use sim_plugin_sdk::{
    CanonicalTypeId, ProviderInvocationResultV2, ProviderResultReceiptSet, TypeRegistry,
    ValueFamily,
};
use sim_tensor::{
    CancellationToken, DType, DeviceId, ImageTensor, NativeTensorPayload, NativeTensorRole, Tensor,
    TensorDescriptor,
};
use thiserror::Error;

use crate::{
    MAX_PROVIDER_RESULT_RECEIPT_LIFETIME, ProviderInvocationIdentity, ProviderResultNonce,
    ProviderResultReceipt, ProviderResultReceiptIssuer, ProviderResultReceiptVerifier,
};

pub const NATIVE_PROVIDER_TRANSPORT_SCHEMA: &str = "zed:sim-provider-transport@1";
pub const NATIVE_PROVIDER_MATERIALIZER_SCHEMA: &str = "zed:sim-provider-materializer@1";
pub const NATIVE_AUDIO_PERSISTENCE_TRANSPORT_CLASS: &str = "zed:sim-native-audio-persistence@1";
pub const MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_PROVIDER_TRANSPORT_REQUEST_BYTES: usize = 16 * 1024 * 1024;
const PROVIDER_TRANSPORT_REQUEST_DOMAIN: &[u8] = b"zed.sim.provider-transport-request\0";
const PROVIDER_TRANSPORT_RESPONSE_DOMAIN: &[u8] = b"zed.sim.provider-transport-response\0";
const PROVIDER_TRANSPORT_VERSION: u16 = 1;
const MAX_PROVIDER_TRANSPORT_PORTS: usize = 1_024;
const MAX_PROVIDER_TRANSPORT_VALUES_PER_PORT: usize = 4_096;
const MAX_PROVIDER_TRANSPORT_IDENTITY_BYTES: usize = 1_024;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ProviderMaterializationError {
    #[error("provider materialization was cancelled")]
    Cancelled,
    #[error("provider transport schema is not supported by the native materializer")]
    UnsupportedTransportSchema,
    #[error("provider materializer schema is not supported by the native materializer")]
    UnsupportedMaterializerSchema,
    #[error("provider result response exceeds the session bound")]
    ResponseTooLarge,
    #[error("provider result receipt session is already terminal")]
    ReceiptSessionFinished,
    #[error("provider result request ordinal is not the next host-owned ordinal")]
    RequestOrdinalOutOfOrder,
    #[error("provider result receipt is malformed, forged, expired, or belongs to another request")]
    ReceiptRejected,
    #[error("provider result receipt was not issued by this live session")]
    UnknownReceipt,
    #[error("provider result receipts must be resolved in host issuance order")]
    ReceiptOutOfOrder,
    #[error("provider result receipt session still has unresolved responses")]
    UnresolvedReceipts,
    #[error("provider result receipt authority is invalid")]
    InvalidReceiptAuthority,
    #[error("provider transport projection is invalid or exceeds its bound")]
    InvalidTransportProjection,
    #[error("provider payload cannot be materialized by the canonical lower owner")]
    InvalidNativePayload,
    #[error("provider downloaded image decoding failed: {0}")]
    ImageDecode(String),
    #[error("provider downloaded image tensor allocation failed: {0}")]
    ImageAllocation(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderTransportEncoding {
    PluginValue,
    NativePayload,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProviderTensorData {
    shape: Vec<u64>,
    dtype: DType,
    logical_bytes: Vec<u8>,
}

impl ProviderTensorData {
    fn from_tensor(tensor: &Tensor) -> Result<Self, ProviderMaterializationError> {
        let data = Self::project_tensor(tensor)?;
        data.validate()?;
        Ok(data)
    }

    fn from_audio_tensor(tensor: &Tensor) -> Result<Self, ProviderMaterializationError> {
        let data = Self::project_tensor(tensor)?;
        data.validate_audio_waveform()?;
        Ok(data)
    }

    fn project_tensor(tensor: &Tensor) -> Result<Self, ProviderMaterializationError> {
        let descriptor = tensor.descriptor();
        if !matches!(descriptor.dtype(), DType::F32 | DType::U8)
            || !descriptor
                .is_contiguous()
                .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?
        {
            return Err(ProviderMaterializationError::InvalidNativePayload);
        }
        let bytes = tensor
            .contiguous_bytes()
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
        let logical_bytes = if descriptor.dtype() == DType::F32 {
            bytes
                .chunks_exact(4)
                .flat_map(|chunk| {
                    f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).to_le_bytes()
                })
                .collect()
        } else {
            bytes.to_vec()
        };
        Ok(Self {
            shape: descriptor.shape().to_vec(),
            dtype: descriptor.dtype(),
            logical_bytes,
        })
    }

    fn validate(&self) -> Result<(), ProviderMaterializationError> {
        if self.shape.is_empty()
            || self.shape.len() > 8
            || self.shape.contains(&0)
            || !matches!(self.dtype, DType::F32 | DType::U8)
        {
            return Err(ProviderMaterializationError::InvalidNativePayload);
        }
        self.validate_storage_bytes()
    }

    fn validate_audio_waveform(&self) -> Result<(), ProviderMaterializationError> {
        let [batch, channels, _samples] = self.shape.as_slice() else {
            return Err(ProviderMaterializationError::InvalidNativePayload);
        };
        if *batch == 0 || *channels == 0 || self.dtype != DType::F32 {
            return Err(ProviderMaterializationError::InvalidNativePayload);
        }
        self.validate_storage_bytes()
    }

    fn validate_storage_bytes(&self) -> Result<(), ProviderMaterializationError> {
        let element_bytes = match self.dtype {
            DType::F32 => 4_u64,
            DType::U8 => 1_u64,
            _ => return Err(ProviderMaterializationError::InvalidNativePayload),
        };
        let expected = self
            .shape
            .iter()
            .try_fold(element_bytes, |bytes, dimension| {
                bytes.checked_mul(*dimension)
            })
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(ProviderMaterializationError::InvalidNativePayload)?;
        if expected != self.logical_bytes.len()
            || self.logical_bytes.len() > MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES
        {
            return Err(ProviderMaterializationError::InvalidNativePayload);
        }
        Ok(())
    }

    fn materialize(
        &self,
        context: &NativeNodeContext,
    ) -> Result<Tensor, ProviderMaterializationError> {
        self.validate()?;
        self.materialize_validated(context)
    }

    fn materialize_audio_waveform(
        &self,
        context: &NativeNodeContext,
    ) -> Result<Tensor, ProviderMaterializationError> {
        self.validate_audio_waveform()?;
        self.materialize_validated(context)
    }

    fn materialize_validated(
        &self,
        context: &NativeNodeContext,
    ) -> Result<Tensor, ProviderMaterializationError> {
        let compute = context
            .compute_session()
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
        let execution_context = compute
            .execution_context(context)
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
        let descriptor = TensorDescriptor::contiguous(
            self.shape.clone(),
            self.dtype,
            DeviceId::CPU,
            compute.stream(),
        )
        .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
        let bytes = if self.dtype == DType::F32 {
            self.logical_bytes
                .chunks_exact(4)
                .flat_map(|chunk| {
                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).to_ne_bytes()
                })
                .collect::<Vec<_>>()
        } else {
            self.logical_bytes.clone()
        };
        compute
            .backend()
            .upload_bytes(descriptor, &bytes, &execution_context)
            .map(|(tensor, _)| tensor)
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProviderAudioData {
    waveform: ProviderTensorData,
    sample_rate: u32,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ProviderNativePayload {
    Image {
        tensor: ProviderTensorData,
    },
    Mask {
        tensor: ProviderTensorData,
    },
    Audio {
        audio: ProviderAudioData,
    },
    Video {
        frames: ProviderTensorData,
        frame_rate_numerator: u64,
        frame_rate_denominator: u64,
        #[serde(default = "default_video_bit_depth")]
        bit_depth: u8,
        audio: Option<ProviderAudioData>,
        alpha: Option<ProviderTensorData>,
        metadata: BTreeMap<String, String>,
    },
    Artifact {
        source_type_id: String,
        media_type: String,
        bytes: Vec<u8>,
    },
    File3d {
        source_type_id: String,
        format: String,
        bytes: Vec<u8>,
    },
    Camera {
        source_type_id: String,
        position: [f32; 3],
        target: [f32; 3],
        zoom: f32,
        orientation_wxyz: Option<[f32; 4]>,
        projection: ProviderCameraProjection,
        width: u32,
        height: u32,
    },
    ProviderTask {
        semantic_digest_sha256: String,
        abi_bytes: Vec<u8>,
    },
}

const fn default_video_bit_depth() -> u8 {
    NativeVideoBitDepth::Eight.bits()
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ProviderCameraProjection {
    Perspective {
        fov_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    },
}

impl ProviderNativePayload {
    fn from_stored(payload: &NativeStoredPayload) -> Result<Self, ProviderMaterializationError> {
        let payload = match payload {
            NativeStoredPayload::Tensor(payload) => match payload.role() {
                NativeTensorRole::Image => Self::Image {
                    tensor: ProviderTensorData::from_tensor(payload.tensor())?,
                },
                NativeTensorRole::Mask => Self::Mask {
                    tensor: ProviderTensorData::from_tensor(payload.tensor())?,
                },
                _ => return Err(ProviderMaterializationError::InvalidNativePayload),
            },
            NativeStoredPayload::Audio(payload) => Self::Audio {
                audio: ProviderAudioData {
                    waveform: ProviderTensorData::from_audio_tensor(payload.waveform())?,
                    sample_rate: payload.sample_rate(),
                },
            },
            NativeStoredPayload::Video(payload) => {
                let components = payload
                    .components()
                    .ok_or(ProviderMaterializationError::InvalidNativePayload)?;
                Self::Video {
                    frames: ProviderTensorData::from_tensor(components.frames())?,
                    frame_rate_numerator: payload.frame_rate().0,
                    frame_rate_denominator: payload.frame_rate().1,
                    bit_depth: payload.bit_depth().bits(),
                    audio: components
                        .audio()
                        .map(|audio| {
                            Ok(ProviderAudioData {
                                waveform: ProviderTensorData::from_audio_tensor(audio.waveform())?,
                                sample_rate: audio.sample_rate(),
                            })
                        })
                        .transpose()?,
                    alpha: components
                        .alpha()
                        .map(ProviderTensorData::from_tensor)
                        .transpose()?,
                    metadata: components.metadata().clone(),
                }
            }
            NativeStoredPayload::Artifact(payload) => Self::Artifact {
                source_type_id: payload.source_type_id().to_owned(),
                media_type: payload.media_type().to_owned(),
                bytes: payload.bytes().to_vec(),
            },
            NativeStoredPayload::File3D(payload) => Self::File3d {
                source_type_id: payload.source_type_id().to_owned(),
                format: payload.format().extension().to_owned(),
                bytes: payload.bytes().to_vec(),
            },
            NativeStoredPayload::Camera(payload) => Self::Camera {
                source_type_id: payload.source_type_id().to_owned(),
                position: *payload.position(),
                target: *payload.target(),
                zoom: payload.zoom(),
                orientation_wxyz: payload.orientation_wxyz().copied(),
                projection: match payload.projection() {
                    NativeCameraProjection::Perspective {
                        fov_degrees,
                        aspect_ratio,
                        near,
                        far,
                    } => ProviderCameraProjection::Perspective {
                        fov_degrees,
                        aspect_ratio,
                        near,
                        far,
                    },
                    NativeCameraProjection::Orthographic {
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    } => ProviderCameraProjection::Orthographic {
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    },
                },
                width: payload.dimensions().0,
                height: payload.dimensions().1,
            },
            NativeStoredPayload::Provider(payload) => Self::ProviderTask {
                semantic_digest_sha256: payload.semantic_digest_sha256().to_owned(),
                abi_bytes: payload.abi_bytes().to_vec(),
            },
            _ => return Err(ProviderMaterializationError::InvalidNativePayload),
        };
        payload.validate()?;
        Ok(payload)
    }

    fn validate(&self) -> Result<(), ProviderMaterializationError> {
        match self {
            Self::Image { tensor } => {
                tensor.validate()?;
                if tensor.dtype != DType::F32
                    || tensor.shape.len() != 4
                    || !matches!(tensor.shape.get(3), Some(1 | 3 | 4))
                {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                }
            }
            Self::Mask { tensor } => {
                tensor.validate()?;
                if tensor.dtype != DType::F32 || tensor.shape.len() != 3 {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                }
            }
            Self::Audio { audio } => validate_provider_audio(audio)?,
            Self::Video {
                frames,
                frame_rate_numerator,
                frame_rate_denominator,
                bit_depth,
                audio,
                alpha,
                metadata,
            } => {
                frames.validate()?;
                if frames.shape.len() != 4
                    || !matches!(frames.shape.get(3), Some(1 | 3 | 4))
                    || *frame_rate_numerator == 0
                    || *frame_rate_denominator == 0
                    || NativeVideoBitDepth::try_from(*bit_depth).is_err()
                    || metadata.len() > 128
                {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                }
                if let Some(audio) = audio {
                    validate_provider_audio(audio)?;
                }
                if let Some(alpha) = alpha {
                    alpha.validate()?;
                    let expected = [frames.shape[0], frames.shape[1], frames.shape[2], 1];
                    if alpha.dtype != DType::F32 || alpha.shape.as_slice() != expected {
                        return Err(ProviderMaterializationError::InvalidNativePayload);
                    }
                }
            }
            Self::Artifact {
                source_type_id,
                media_type,
                bytes,
            } => {
                if !matches!(source_type_id.as_str(), "SVG" | "AUDIO_RECORD" | "WEBCAM")
                    || media_type.is_empty()
                    || bytes.len() > MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES
                {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                }
            }
            Self::File3d {
                source_type_id,
                format,
                bytes,
            } => {
                file_3d_role(source_type_id)?;
                file_3d_format(format)?;
                if bytes.is_empty() || bytes.len() > MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                }
            }
            Self::Camera { source_type_id, .. } => {
                camera_role(source_type_id)?;
            }
            Self::ProviderTask {
                semantic_digest_sha256,
                abi_bytes,
            } => {
                if !is_sha256(semantic_digest_sha256)
                    || abi_bytes.is_empty()
                    || abi_bytes.len() > MAX_PROVIDER_TRANSPORT_REQUEST_BYTES
                {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                }
            }
        }
        Ok(())
    }

    fn materialize(
        &self,
        expected_type: &NativeHandleType,
        signed_namespace: &str,
        context: &NativeNodeContext,
    ) -> Result<NativeStoredPayload, ProviderMaterializationError> {
        self.validate()?;
        let payload = match self {
            Self::Image { tensor } => {
                require_expected_type(expected_type, NativeHandleKind::Image, "IMAGE")?;
                let tensor = tensor.materialize(context)?;
                let image = ImageTensor::from_tensor(tensor)
                    .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
                NativeStoredPayload::Tensor(Arc::new(
                    NativeTensorPayload::from_image(NativeTensorRole::Image, image)
                        .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
                ))
            }
            Self::Mask { tensor } => {
                require_expected_type(expected_type, NativeHandleKind::Mask, "MASK")?;
                let shape = tensor.shape.as_slice();
                let [batch, height, width] = shape else {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                };
                let values = provider_f32_values(tensor)?;
                let compute = context
                    .compute_session()
                    .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
                let execution_context = compute
                    .execution_context(context)
                    .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
                let image = ImageTensor::from_f32(
                    compute.backend(),
                    &execution_context,
                    *batch,
                    *height,
                    *width,
                    1,
                    &values,
                )
                .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
                NativeStoredPayload::Tensor(Arc::new(
                    NativeTensorPayload::from_image(NativeTensorRole::Mask, image)
                        .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
                ))
            }
            Self::Audio { audio } => {
                require_expected_type(expected_type, NativeHandleKind::Audio, "AUDIO")?;
                NativeStoredPayload::Audio(Arc::new(materialize_audio(audio, context)?))
            }
            Self::Video {
                frames,
                frame_rate_numerator,
                frame_rate_denominator,
                bit_depth,
                audio,
                alpha,
                metadata,
            } => {
                require_expected_type(expected_type, NativeHandleKind::Video, "VIDEO")?;
                NativeStoredPayload::Video(Arc::new(
                    NativeVideoPayload::checked(
                        frames.materialize(context)?,
                        *frame_rate_numerator,
                        *frame_rate_denominator,
                        NativeVideoBitDepth::try_from(*bit_depth)
                            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
                        audio
                            .as_ref()
                            .map(|audio| materialize_audio(audio, context))
                            .transpose()?,
                        alpha
                            .as_ref()
                            .map(|alpha| alpha.materialize(context))
                            .transpose()?,
                        metadata.clone(),
                    )
                    .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
                ))
            }
            Self::Artifact {
                source_type_id,
                media_type,
                bytes,
            } => {
                require_expected_type(expected_type, NativeHandleKind::Artifact, source_type_id)?;
                let kind = match source_type_id.as_str() {
                    "SVG" => NativeArtifactKind::Svg,
                    "AUDIO_RECORD" => NativeArtifactKind::AudioRecord,
                    "WEBCAM" => NativeArtifactKind::Webcam,
                    _ => return Err(ProviderMaterializationError::InvalidNativePayload),
                };
                NativeStoredPayload::Artifact(Arc::new(
                    NativeArtifactPayload::checked(kind, media_type.clone(), bytes.clone())
                        .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
                ))
            }
            Self::File3d {
                source_type_id,
                format,
                bytes,
            } => {
                require_expected_type(expected_type, NativeHandleKind::ThreeD, source_type_id)?;
                NativeStoredPayload::File3D(Arc::new(
                    NativeFile3DPayload::checked(
                        file_3d_role(source_type_id)?,
                        file_3d_format(format)?,
                        bytes.clone(),
                    )
                    .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
                ))
            }
            Self::Camera {
                source_type_id,
                position,
                target,
                zoom,
                orientation_wxyz,
                projection,
                width,
                height,
            } => {
                require_expected_type(expected_type, NativeHandleKind::ThreeD, source_type_id)?;
                let projection = match projection {
                    ProviderCameraProjection::Perspective {
                        fov_degrees,
                        aspect_ratio,
                        near,
                        far,
                    } => NativeCameraProjection::Perspective {
                        fov_degrees: *fov_degrees,
                        aspect_ratio: *aspect_ratio,
                        near: *near,
                        far: *far,
                    },
                    ProviderCameraProjection::Orthographic {
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    } => NativeCameraProjection::Orthographic {
                        left: *left,
                        right: *right,
                        bottom: *bottom,
                        top: *top,
                        near: *near,
                        far: *far,
                    },
                };
                NativeStoredPayload::Camera(Arc::new(
                    NativeCameraPayload::checked(
                        camera_role(source_type_id)?,
                        *position,
                        *target,
                        *zoom,
                        *orientation_wxyz,
                        projection,
                        *width,
                        *height,
                    )
                    .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
                ))
            }
            Self::ProviderTask {
                semantic_digest_sha256,
                abi_bytes,
            } => {
                if expected_type.kind != NativeHandleKind::ProviderTask {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                }
                let payload = sim_nodes::NativeProviderPayload::from_abi(
                    expected_type.clone(),
                    signed_namespace,
                    abi_bytes.clone(),
                )
                .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
                if payload.semantic_digest_sha256() != semantic_digest_sha256 {
                    return Err(ProviderMaterializationError::InvalidNativePayload);
                }
                NativeStoredPayload::Provider(Arc::new(payload))
            }
        };
        payload
            .validate()
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
        if payload
            .handle_type()
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?
            != *expected_type
        {
            return Err(ProviderMaterializationError::InvalidNativePayload);
        }
        Ok(payload)
    }
}

fn validate_provider_audio(audio: &ProviderAudioData) -> Result<(), ProviderMaterializationError> {
    audio.waveform.validate_audio_waveform()?;
    if audio.waveform.dtype != DType::F32
        || audio.waveform.shape.len() != 3
        || !(1..=384_000).contains(&audio.sample_rate)
    {
        return Err(ProviderMaterializationError::InvalidNativePayload);
    }
    Ok(())
}

fn materialize_audio(
    audio: &ProviderAudioData,
    context: &NativeNodeContext,
) -> Result<NativeAudioPayload, ProviderMaterializationError> {
    validate_provider_audio(audio)?;
    NativeAudioPayload::checked(
        audio.waveform.materialize_audio_waveform(context)?,
        audio.sample_rate,
    )
    .map_err(|_| ProviderMaterializationError::InvalidNativePayload)
}

fn provider_f32_values(
    tensor: &ProviderTensorData,
) -> Result<Vec<f32>, ProviderMaterializationError> {
    if tensor.dtype != DType::F32 {
        return Err(ProviderMaterializationError::InvalidNativePayload);
    }
    Ok(tensor
        .logical_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn require_expected_type(
    expected_type: &NativeHandleType,
    kind: NativeHandleKind,
    source_type_id: &str,
) -> Result<(), ProviderMaterializationError> {
    if expected_type.kind != kind || expected_type.type_id != source_type_id {
        return Err(ProviderMaterializationError::InvalidNativePayload);
    }
    Ok(())
}

fn file_3d_format(format: &str) -> Result<NativeFile3DFormat, ProviderMaterializationError> {
    match format {
        "fbx" => Ok(NativeFile3DFormat::Fbx),
        "gltf" => Ok(NativeFile3DFormat::Gltf),
        "glb" => Ok(NativeFile3DFormat::Glb),
        "ksplat" => Ok(NativeFile3DFormat::Ksplat),
        "obj" => Ok(NativeFile3DFormat::Obj),
        "ply" => Ok(NativeFile3DFormat::Ply),
        "splat" => Ok(NativeFile3DFormat::Splat),
        "spz" => Ok(NativeFile3DFormat::Spz),
        "stl" => Ok(NativeFile3DFormat::Stl),
        "usdz" => Ok(NativeFile3DFormat::Usdz),
        _ => Err(ProviderMaterializationError::InvalidNativePayload),
    }
}

fn file_3d_role(source_type_id: &str) -> Result<NativeFile3DRole, ProviderMaterializationError> {
    match source_type_id {
        "FILE_3D" => Ok(NativeFile3DRole::Any),
        "FILE_3D_FBX" => Ok(NativeFile3DRole::Fbx),
        "FILE_3D_GLTF" => Ok(NativeFile3DRole::Gltf),
        "FILE_3D_GLB" => Ok(NativeFile3DRole::Glb),
        "FILE_3D_KSPLAT" => Ok(NativeFile3DRole::Ksplat),
        "FILE_3D_OBJ" => Ok(NativeFile3DRole::Obj),
        "FILE_3D_PLY" => Ok(NativeFile3DRole::Ply),
        "FILE_3D_POINT_CLOUD_ANY" => Ok(NativeFile3DRole::PointCloudAny),
        "FILE_3D_SPLAT_ANY" => Ok(NativeFile3DRole::SplatAny),
        "FILE_3D_SPLAT" => Ok(NativeFile3DRole::Splat),
        "FILE_3D_SPZ" => Ok(NativeFile3DRole::Spz),
        "FILE_3D_STL" => Ok(NativeFile3DRole::Stl),
        "FILE_3D_USDZ" => Ok(NativeFile3DRole::Usdz),
        _ => Err(ProviderMaterializationError::InvalidNativePayload),
    }
}

fn camera_role(source_type_id: &str) -> Result<NativeCameraRole, ProviderMaterializationError> {
    match source_type_id {
        "CAMERA_CONTROL" => Ok(NativeCameraRole::CameraControl),
        "LOAD3D_CAMERA" => Ok(NativeCameraRole::Load3D),
        _ => Err(ProviderMaterializationError::InvalidNativePayload),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTransportValue {
    type_id: String,
    family: ValueFamily,
    encoding: ProviderTransportEncoding,
    abi_bytes: Vec<u8>,
}

impl ProviderTransportValue {
    pub fn checked(
        type_id: impl Into<String>,
        family: ValueFamily,
        abi_bytes: Vec<u8>,
    ) -> Result<Self, ProviderMaterializationError> {
        let value = Self {
            type_id: type_id.into(),
            family,
            encoding: ProviderTransportEncoding::PluginValue,
            abi_bytes,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn type_id(&self) -> &str {
        &self.type_id
    }

    pub const fn family(&self) -> ValueFamily {
        self.family
    }

    pub const fn encoding(&self) -> ProviderTransportEncoding {
        self.encoding
    }

    pub fn abi_bytes(&self) -> &[u8] {
        &self.abi_bytes
    }

    fn validate(&self) -> Result<(), ProviderMaterializationError> {
        if !valid_transport_identity(&self.type_id)
            || self.abi_bytes.is_empty()
            || self.abi_bytes.len() > MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES
        {
            return Err(ProviderMaterializationError::InvalidTransportProjection);
        }
        if self.encoding == ProviderTransportEncoding::NativePayload {
            let payload: ProviderNativePayload = postcard::from_bytes(&self.abi_bytes)
                .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
            payload.validate()?;
        }
        Ok(())
    }

    pub fn from_native_payload(
        type_id: impl Into<String>,
        family: ValueFamily,
        payload: &NativeStoredPayload,
    ) -> Result<Self, ProviderMaterializationError> {
        let native_payload = ProviderNativePayload::from_stored(payload)?;
        let value = Self {
            type_id: type_id.into(),
            family,
            encoding: ProviderTransportEncoding::NativePayload,
            abi_bytes: postcard::to_stdvec(&native_payload)
                .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn materialize_native_payload(
        &self,
        expected_type: &NativeHandleType,
        signed_namespace: &str,
        context: &NativeNodeContext,
    ) -> Result<NativeStoredPayload, ProviderMaterializationError> {
        if self.encoding != ProviderTransportEncoding::NativePayload {
            return Err(ProviderMaterializationError::InvalidNativePayload);
        }
        let payload: ProviderNativePayload = postcard::from_bytes(&self.abi_bytes)
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
        payload.materialize(expected_type, signed_namespace, context)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTransportPort {
    port_id: String,
    present: bool,
    values: Vec<ProviderTransportValue>,
}

impl ProviderTransportPort {
    pub fn checked(
        port_id: impl Into<String>,
        present: bool,
        values: Vec<ProviderTransportValue>,
    ) -> Result<Self, ProviderMaterializationError> {
        let port = Self {
            port_id: port_id.into(),
            present,
            values,
        };
        port.validate()?;
        Ok(port)
    }

    pub fn port_id(&self) -> &str {
        &self.port_id
    }

    pub const fn present(&self) -> bool {
        self.present
    }

    pub fn values(&self) -> &[ProviderTransportValue] {
        &self.values
    }

    fn validate(&self) -> Result<(), ProviderMaterializationError> {
        if !valid_transport_identity(&self.port_id)
            || self.values.len() > MAX_PROVIDER_TRANSPORT_VALUES_PER_PORT
            || (!self.present && !self.values.is_empty())
        {
            return Err(ProviderMaterializationError::InvalidTransportProjection);
        }
        self.values
            .iter()
            .try_for_each(ProviderTransportValue::validate)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProviderTransportProjection {
    class_type: String,
    ports: Vec<ProviderTransportPort>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderTransportRequest(ProviderTransportProjection);

impl ProviderTransportRequest {
    pub fn checked(
        class_type: impl Into<String>,
        ports: Vec<ProviderTransportPort>,
    ) -> Result<Self, ProviderMaterializationError> {
        let projection = ProviderTransportProjection {
            class_type: class_type.into(),
            ports,
        };
        validate_transport_projection(&projection)?;
        Ok(Self(projection))
    }

    pub fn class_type(&self) -> &str {
        &self.0.class_type
    }

    pub fn ports(&self) -> &[ProviderTransportPort] {
        &self.0.ports
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ProviderMaterializationError> {
        encode_transport_projection(
            PROVIDER_TRANSPORT_REQUEST_DOMAIN,
            &self.0,
            MAX_PROVIDER_TRANSPORT_REQUEST_BYTES,
        )
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProviderMaterializationError> {
        decode_transport_projection(
            PROVIDER_TRANSPORT_REQUEST_DOMAIN,
            bytes,
            MAX_PROVIDER_TRANSPORT_REQUEST_BYTES,
        )
        .map(Self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderTransportResponse(ProviderTransportProjection);

impl ProviderTransportResponse {
    pub fn checked(
        class_type: impl Into<String>,
        ports: Vec<ProviderTransportPort>,
    ) -> Result<Self, ProviderMaterializationError> {
        let projection = ProviderTransportProjection {
            class_type: class_type.into(),
            ports,
        };
        validate_transport_projection(&projection)?;
        Ok(Self(projection))
    }

    pub fn class_type(&self) -> &str {
        &self.0.class_type
    }

    pub fn ports(&self) -> &[ProviderTransportPort] {
        &self.0.ports
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ProviderMaterializationError> {
        encode_transport_projection(
            PROVIDER_TRANSPORT_RESPONSE_DOMAIN,
            &self.0,
            MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES,
        )
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProviderMaterializationError> {
        decode_transport_projection(
            PROVIDER_TRANSPORT_RESPONSE_DOMAIN,
            bytes,
            MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES,
        )
        .map(Self)
    }
}

pub fn native_audio_persistence_transport(
    payload: &NativeAudioPayload,
) -> Result<ProviderTransportResponse, ProviderMaterializationError> {
    payload
        .validate()
        .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
    let stored = NativeStoredPayload::Audio(Arc::new(payload.clone()));
    let value = ProviderTransportValue::from_native_payload("AUDIO", ValueFamily::Tensor, &stored)?;
    ProviderTransportResponse::checked(
        NATIVE_AUDIO_PERSISTENCE_TRANSPORT_CLASS,
        vec![
            ProviderTransportPort::checked("audio", true, vec![value.clone()])?,
            ProviderTransportPort::checked("audio_list", true, vec![value])?,
        ],
    )
}

pub fn materialize_provider_invocation_result_v2(
    result: &ProviderInvocationResultV2,
    verified_receipt: &crate::VerifiedProviderRuntimeReceiptV2,
    registry: &TypeRegistry,
    cancellation: &CancellationToken,
) -> Result<ProviderTransportResponse, ProviderMaterializationError> {
    validate_provider_invocation_result_v2_for_materialization(result, registry, cancellation)?;
    cancellation
        .check()
        .map_err(|_| ProviderMaterializationError::Cancelled)?;
    if crate::provider_terminal_completed_receipt_sha256(&result.receipt)
        != verified_receipt.identity().terminal_receipt_sha256
    {
        return Err(ProviderMaterializationError::InvalidTransportProjection);
    }
    materialize_validated_provider_invocation_result_v2(result, cancellation)
}

pub fn provider_invocation_result_v2_materialization_identity(
    result: &ProviderInvocationResultV2,
    registry: &TypeRegistry,
    cancellation: &CancellationToken,
) -> Result<[u8; 32], ProviderMaterializationError> {
    validate_provider_invocation_result_v2_for_materialization(result, registry, cancellation)?;
    let materialization =
        materialize_validated_provider_invocation_result_v2(result, cancellation)?;
    let bytes = materialization.to_bytes()?;
    cancellation
        .check()
        .map_err(|_| ProviderMaterializationError::Cancelled)?;
    Ok(Sha256::digest(bytes).into())
}

fn validate_provider_invocation_result_v2_for_materialization(
    result: &ProviderInvocationResultV2,
    registry: &TypeRegistry,
    cancellation: &CancellationToken,
) -> Result<(), ProviderMaterializationError> {
    cancellation
        .check()
        .map_err(|_| ProviderMaterializationError::Cancelled)?;
    let aggregate_bytes = result.outputs.iter().try_fold(0usize, |total, output| {
        total
            .checked_add(output.port_id.len())
            .and_then(|total| total.checked_add(output.value.type_id.to_string().len()))
            .and_then(|total| total.checked_add(output.value.abi_bytes.len()))
            .and_then(|total| total.checked_add(64))
            .ok_or(ProviderMaterializationError::InvalidTransportProjection)
    })?;
    if aggregate_bytes > MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES {
        return Err(ProviderMaterializationError::InvalidTransportProjection);
    }
    result
        .validate(registry)
        .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)
}

fn materialize_validated_provider_invocation_result_v2(
    result: &ProviderInvocationResultV2,
    cancellation: &CancellationToken,
) -> Result<ProviderTransportResponse, ProviderMaterializationError> {
    let mut ports = Vec::new();
    ports
        .try_reserve_exact(result.outputs.len())
        .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)?;
    for output in &result.outputs {
        cancellation
            .check()
            .map_err(|_| ProviderMaterializationError::Cancelled)?;
        let type_id_source = output.value.type_id.to_string();
        let mut type_id = String::new();
        type_id
            .try_reserve_exact(type_id_source.len())
            .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)?;
        type_id.push_str(&type_id_source);
        let mut abi_bytes = Vec::new();
        abi_bytes
            .try_reserve_exact(output.value.abi_bytes.len())
            .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)?;
        cancellation
            .check()
            .map_err(|_| ProviderMaterializationError::Cancelled)?;
        abi_bytes.extend_from_slice(&output.value.abi_bytes);
        let value = ProviderTransportValue::checked(type_id, output.value.family, abi_bytes)?;
        let mut port_id = String::new();
        port_id
            .try_reserve_exact(output.port_id.len())
            .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)?;
        port_id.push_str(&output.port_id);
        let mut values = Vec::new();
        values
            .try_reserve_exact(1)
            .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)?;
        values.push(value);
        cancellation
            .check()
            .map_err(|_| ProviderMaterializationError::Cancelled)?;
        ports.push(ProviderTransportPort::checked(port_id, true, values)?);
    }
    cancellation
        .check()
        .map_err(|_| ProviderMaterializationError::Cancelled)?;
    ProviderTransportResponse::checked("zed:sim-provider-invocation-result@2", ports)
}

fn validate_transport_projection(
    projection: &ProviderTransportProjection,
) -> Result<(), ProviderMaterializationError> {
    if !valid_transport_identity(&projection.class_type)
        || projection.ports.len() > MAX_PROVIDER_TRANSPORT_PORTS
    {
        return Err(ProviderMaterializationError::InvalidTransportProjection);
    }
    let mut previous_port = None;
    for port in &projection.ports {
        port.validate()?;
        if previous_port.is_some_and(|previous| previous >= port.port_id()) {
            return Err(ProviderMaterializationError::InvalidTransportProjection);
        }
        previous_port = Some(port.port_id());
    }
    Ok(())
}

fn encode_transport_projection(
    domain: &[u8],
    projection: &ProviderTransportProjection,
    maximum_bytes: usize,
) -> Result<Vec<u8>, ProviderMaterializationError> {
    validate_transport_projection(projection)?;
    let payload = postcard::to_stdvec(projection)
        .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)?;
    let total = domain
        .len()
        .checked_add(2)
        .and_then(|length| length.checked_add(payload.len()))
        .ok_or(ProviderMaterializationError::InvalidTransportProjection)?;
    if total > maximum_bytes {
        return Err(ProviderMaterializationError::InvalidTransportProjection);
    }
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&PROVIDER_TRANSPORT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

fn decode_transport_projection(
    domain: &[u8],
    bytes: &[u8],
    maximum_bytes: usize,
) -> Result<ProviderTransportProjection, ProviderMaterializationError> {
    if bytes.len() > maximum_bytes || !bytes.starts_with(domain) {
        return Err(ProviderMaterializationError::InvalidTransportProjection);
    }
    let version_start = domain.len();
    let version_end = version_start
        .checked_add(2)
        .ok_or(ProviderMaterializationError::InvalidTransportProjection)?;
    let version_bytes = bytes
        .get(version_start..version_end)
        .ok_or(ProviderMaterializationError::InvalidTransportProjection)?;
    let version = u16::from_le_bytes(
        version_bytes
            .try_into()
            .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)?,
    );
    if version != PROVIDER_TRANSPORT_VERSION {
        return Err(ProviderMaterializationError::InvalidTransportProjection);
    }
    let projection: ProviderTransportProjection = postcard::from_bytes(
        bytes
            .get(version_end..)
            .ok_or(ProviderMaterializationError::InvalidTransportProjection)?,
    )
    .map_err(|_| ProviderMaterializationError::InvalidTransportProjection)?;
    validate_transport_projection(&projection)?;
    Ok(projection)
}

fn valid_transport_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_PROVIDER_TRANSPORT_IDENTITY_BYTES
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

#[derive(Clone)]
pub struct ProviderResultReceiptAuthority {
    principal_id: String,
    prompt_sha256: String,
    provider_binding_sha256: String,
    issuer: Arc<ProviderResultReceiptIssuer>,
    receipt_lifetime: Duration,
}

impl ProviderResultReceiptAuthority {
    pub fn new(
        principal_id: impl Into<String>,
        prompt_sha256: impl Into<String>,
        provider_binding_sha256: impl Into<String>,
        issuer: Arc<ProviderResultReceiptIssuer>,
        receipt_lifetime: Duration,
    ) -> Result<Self, ProviderMaterializationError> {
        let principal_id = principal_id.into();
        let prompt_sha256 = prompt_sha256.into();
        let provider_binding_sha256 = provider_binding_sha256.into();
        if principal_id.is_empty()
            || principal_id.len() > 1_024
            || principal_id != principal_id.trim()
            || principal_id.chars().any(char::is_control)
            || !is_sha256(&prompt_sha256)
            || !is_sha256(&provider_binding_sha256)
            || receipt_lifetime.is_zero()
            || receipt_lifetime > MAX_PROVIDER_RESULT_RECEIPT_LIFETIME
        {
            return Err(ProviderMaterializationError::InvalidReceiptAuthority);
        }
        Ok(Self {
            principal_id,
            prompt_sha256,
            provider_binding_sha256,
            issuer,
            receipt_lifetime,
        })
    }

    pub fn principal_id(&self) -> &str {
        &self.principal_id
    }

    pub fn prompt_sha256(&self) -> &str {
        &self.prompt_sha256
    }

    pub fn provider_binding_sha256(&self) -> &str {
        &self.provider_binding_sha256
    }

    pub fn receipt_lifetime(&self) -> Duration {
        self.receipt_lifetime
    }

    pub fn begin_session(
        &self,
        maximum_response_bytes: usize,
    ) -> Result<ProviderResultReceiptSession, ProviderMaterializationError> {
        ProviderResultReceiptSession::new(self.issuer.clone(), maximum_response_bytes, 0)
    }
}

impl std::fmt::Debug for ProviderResultReceiptAuthority {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ProviderResultReceiptAuthority([REDACTED])")
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

pub fn validate_native_provider_schemas(
    transport_schema: &CanonicalTypeId,
    materializer_schema: &CanonicalTypeId,
) -> Result<(), ProviderMaterializationError> {
    if transport_schema.to_string() != NATIVE_PROVIDER_TRANSPORT_SCHEMA {
        return Err(ProviderMaterializationError::UnsupportedTransportSchema);
    }
    if materializer_schema.to_string() != NATIVE_PROVIDER_MATERIALIZER_SCHEMA {
        return Err(ProviderMaterializationError::UnsupportedMaterializerSchema);
    }
    Ok(())
}

struct IssuedProviderResult {
    identity: ProviderInvocationIdentity,
    result_sha256: String,
    response: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedProviderResult {
    identity: ProviderInvocationIdentity,
    response: Vec<u8>,
}

impl ResolvedProviderResult {
    pub fn identity(&self) -> &ProviderInvocationIdentity {
        &self.identity
    }

    pub fn response(&self) -> &[u8] {
        &self.response
    }

    pub fn into_response(self) -> Vec<u8> {
        self.response
    }
}

pub struct ProviderResultReceiptSession {
    issuer: Arc<ProviderResultReceiptIssuer>,
    verifier: ProviderResultReceiptVerifier,
    maximum_response_bytes: usize,
    next_request_ordinal: u32,
    issued_order: VecDeque<ProviderResultNonce>,
    issued: BTreeMap<ProviderResultNonce, IssuedProviderResult>,
    terminal: bool,
}

impl ProviderResultReceiptSession {
    pub fn new(
        issuer: Arc<ProviderResultReceiptIssuer>,
        maximum_response_bytes: usize,
        first_request_ordinal: u32,
    ) -> Result<Self, ProviderMaterializationError> {
        if maximum_response_bytes == 0
            || maximum_response_bytes > MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES
        {
            return Err(ProviderMaterializationError::ResponseTooLarge);
        }
        let verifier = issuer
            .verifier()
            .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
        Ok(Self {
            issuer,
            verifier,
            maximum_response_bytes,
            next_request_ordinal: first_request_ordinal,
            issued_order: VecDeque::new(),
            issued: BTreeMap::new(),
            terminal: false,
        })
    }

    pub fn issue(
        &mut self,
        identity: ProviderInvocationIdentity,
        response: Vec<u8>,
        issued_at: Instant,
        expires_at: Instant,
    ) -> Result<Vec<u8>, ProviderMaterializationError> {
        self.check_active()?;
        if identity.request_ordinal() != self.next_request_ordinal {
            return Err(ProviderMaterializationError::RequestOrdinalOutOfOrder);
        }
        if response.len() > self.maximum_response_bytes {
            return Err(ProviderMaterializationError::ResponseTooLarge);
        }
        let next_request_ordinal = self
            .next_request_ordinal
            .checked_add(1)
            .ok_or(ProviderMaterializationError::RequestOrdinalOutOfOrder)?;
        let result_sha256 = format!("{:x}", Sha256::digest(&response));
        let nonce = ProviderResultNonce::generate()
            .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
        let receipt = self
            .issuer
            .issue(
                identity.clone(),
                result_sha256.clone(),
                issued_at,
                expires_at,
                nonce,
            )
            .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
        let receipt_bytes = receipt
            .to_bytes()
            .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
        self.issued.insert(
            nonce,
            IssuedProviderResult {
                identity,
                result_sha256,
                response,
            },
        );
        self.issued_order.push_back(nonce);
        self.next_request_ordinal = next_request_ordinal;
        Ok(receipt_bytes)
    }

    pub fn next_request_ordinal(&self) -> Result<u32, ProviderMaterializationError> {
        self.check_active()?;
        Ok(self.next_request_ordinal)
    }

    pub fn resolve(
        &mut self,
        receipt_bytes: &[u8],
        expected_identity: &ProviderInvocationIdentity,
        now: Instant,
    ) -> Result<Vec<u8>, ProviderMaterializationError> {
        self.check_active()?;
        let receipt = ProviderResultReceipt::from_bytes(receipt_bytes)
            .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
        let nonce = receipt.nonce();
        if self.issued_order.front().copied() != Some(nonce) {
            return if self.issued.contains_key(&nonce) {
                Err(ProviderMaterializationError::ReceiptOutOfOrder)
            } else {
                Err(ProviderMaterializationError::UnknownReceipt)
            };
        }
        let issued = self
            .issued
            .get(&nonce)
            .ok_or(ProviderMaterializationError::UnknownReceipt)?;
        if &issued.identity != expected_identity {
            return Err(ProviderMaterializationError::ReceiptRejected);
        }
        self.verifier
            .verify(&receipt, expected_identity, &issued.result_sha256, now)
            .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
        let issued = self
            .issued
            .remove(&nonce)
            .ok_or(ProviderMaterializationError::UnknownReceipt)?;
        self.issued_order.pop_front();
        Ok(issued.response)
    }

    fn authenticated_response<'a>(
        &'a self,
        receipt_bytes: &[u8],
        expected_identity: &ProviderInvocationIdentity,
        now: Instant,
        context: &NativeNodeContext,
    ) -> Result<&'a [u8], ProviderMaterializationError> {
        context
            .cancellation
            .check()
            .map_err(|_| ProviderMaterializationError::Cancelled)?;
        self.check_active()?;
        if context.prompt_id.0.to_string() != expected_identity.prompt_id()
            || context.attempt_id.0.to_string() != expected_identity.attempt_id()
            || context.node_id.0 != expected_identity.node_id()
            || context
                .provider_execution()
                .map_err(|_| ProviderMaterializationError::ReceiptRejected)?
                .compiled_plan_sha256()
                != expected_identity.prompt_sha256()
        {
            return Err(ProviderMaterializationError::ReceiptRejected);
        }
        let receipt = ProviderResultReceipt::from_bytes(receipt_bytes)
            .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
        let issued = self
            .issued
            .get(&receipt.nonce())
            .ok_or(ProviderMaterializationError::UnknownReceipt)?;
        if &issued.identity != expected_identity {
            return Err(ProviderMaterializationError::ReceiptRejected);
        }
        self.verifier
            .verify(&receipt, expected_identity, &issued.result_sha256, now)
            .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
        if issued.response.len() > self.maximum_response_bytes {
            return Err(ProviderMaterializationError::ResponseTooLarge);
        }
        Ok(&issued.response)
    }

    pub fn is_structured_transport_response(
        &self,
        receipt_bytes: &[u8],
        expected_identity: &ProviderInvocationIdentity,
        now: Instant,
        context: &NativeNodeContext,
    ) -> Result<bool, ProviderMaterializationError> {
        let response =
            self.authenticated_response(receipt_bytes, expected_identity, now, context)?;
        let structured = Self::classify_structured_transport_response(
            ProviderTransportResponse::from_bytes(response),
        )?;
        context
            .cancellation
            .check()
            .map_err(|_| ProviderMaterializationError::Cancelled)?;
        Ok(structured)
    }

    fn classify_structured_transport_response(
        response: Result<ProviderTransportResponse, ProviderMaterializationError>,
    ) -> Result<bool, ProviderMaterializationError> {
        match response {
            Ok(_) => Ok(true),
            Err(
                ProviderMaterializationError::InvalidTransportProjection
                | ProviderMaterializationError::InvalidNativePayload,
            ) => Ok(false),
            Err(error) => Err(error),
        }
    }

    pub fn materialize_downloaded_image(
        &self,
        receipt_bytes: &[u8],
        expected_identity: &ProviderInvocationIdentity,
        now: Instant,
        context: &NativeNodeContext,
    ) -> Result<NativeStoredPayload, ProviderMaterializationError> {
        let response =
            self.authenticated_response(receipt_bytes, expected_identity, now, context)?;
        let compute = context
            .compute_session()
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
        let execution_context = compute
            .execution_context(context)
            .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?;
        let limits = sim_media::PngLimits {
            max_input_bytes: self.maximum_response_bytes,
            max_dimension: sim_media::DEFAULT_MAX_PNG_DIMENSION,
            max_pixels: (self.maximum_response_bytes / 16) as u64,
            max_decoder_allocation_bytes: self.maximum_response_bytes as u64,
        };
        let decoded =
            sim_media::decode_downloaded_image_rgba(response, limits, &context.cancellation)
                .map_err(|error| {
                    if context.cancellation.is_cancelled() {
                        ProviderMaterializationError::Cancelled
                    } else {
                        ProviderMaterializationError::ImageDecode(error.to_string())
                    }
                })?;
        context
            .cancellation
            .check()
            .map_err(|_| ProviderMaterializationError::Cancelled)?;
        let image = ImageTensor::from_f32(
            compute.backend(),
            &execution_context,
            1,
            u64::from(decoded.height),
            u64::from(decoded.width),
            4,
            &decoded.pixels_bhwc,
        )
        .map_err(|error| {
            if context.cancellation.is_cancelled() {
                ProviderMaterializationError::Cancelled
            } else {
                ProviderMaterializationError::ImageAllocation(error.to_string())
            }
        })?;
        let payload = NativeStoredPayload::Tensor(Arc::new(
            NativeTensorPayload::from_image(NativeTensorRole::Image, image)
                .map_err(|_| ProviderMaterializationError::InvalidNativePayload)?,
        ));
        context
            .cancellation
            .check()
            .map_err(|_| ProviderMaterializationError::Cancelled)?;
        Ok(payload)
    }

    pub fn resolve_receipt_set(
        &mut self,
        receipt_set: &ProviderResultReceiptSet,
        now: Instant,
    ) -> Result<Vec<ResolvedProviderResult>, ProviderMaterializationError> {
        self.check_active()?;
        if receipt_set.receipts().len() != self.issued_order.len() {
            return Err(ProviderMaterializationError::UnresolvedReceipts);
        }
        let mut validated_nonces = Vec::with_capacity(receipt_set.receipts().len());
        for (receipt_bytes, expected_nonce) in receipt_set
            .receipts()
            .iter()
            .zip(self.issued_order.iter().copied())
        {
            let receipt = ProviderResultReceipt::from_bytes(receipt_bytes)
                .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
            if receipt.nonce() != expected_nonce {
                return if self.issued.contains_key(&receipt.nonce()) {
                    Err(ProviderMaterializationError::ReceiptOutOfOrder)
                } else {
                    Err(ProviderMaterializationError::UnknownReceipt)
                };
            }
            let issued = self
                .issued
                .get(&expected_nonce)
                .ok_or(ProviderMaterializationError::UnknownReceipt)?;
            self.verifier
                .verify(&receipt, &issued.identity, &issued.result_sha256, now)
                .map_err(|_| ProviderMaterializationError::ReceiptRejected)?;
            validated_nonces.push(expected_nonce);
        }

        let mut resolved = Vec::with_capacity(validated_nonces.len());
        for nonce in validated_nonces {
            if self.issued_order.pop_front() != Some(nonce) {
                return Err(ProviderMaterializationError::ReceiptOutOfOrder);
            }
            let issued = self
                .issued
                .remove(&nonce)
                .ok_or(ProviderMaterializationError::UnknownReceipt)?;
            resolved.push(ResolvedProviderResult {
                identity: issued.identity,
                response: issued.response,
            });
        }
        Ok(resolved)
    }

    pub fn finish(mut self) -> Result<(), ProviderMaterializationError> {
        self.check_active()?;
        if !self.issued.is_empty() || !self.issued_order.is_empty() {
            return Err(ProviderMaterializationError::UnresolvedReceipts);
        }
        self.terminal = true;
        Ok(())
    }

    pub fn abort(mut self) {
        self.issued.clear();
        self.issued_order.clear();
        self.terminal = true;
    }

    fn check_active(&self) -> Result<(), ProviderMaterializationError> {
        if self.terminal {
            Err(ProviderMaterializationError::ReceiptSessionFinished)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_nodes::{NativeNodeComputeSession, NativeNodeServiceIdentity, NativeNodeServices};
    use sim_plugin_sdk::{
        PluginValue, ProviderEncodedValueV2, ProviderHttpMethodV2, ProviderMaterializedOutputV2,
        ScalarValue, TypeRegistry,
    };
    use sim_tensor::{
        CpuWorkspaceAuthority, StreamId, TensorBackend,
        generated_sim_operator_indirection_01::tensor_from_f32_with_context_exact_native,
    };
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use std::time::Duration;
    use uuid::Uuid;

    fn invocation_identity(
        node_id: &str,
        request_ordinal: u32,
        request_byte: char,
    ) -> Result<ProviderInvocationIdentity, Box<dyn std::error::Error>> {
        Ok(ProviderInvocationIdentity::new(
            "principal-a",
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "a".repeat(64),
            "00000000-0000-0000-0000-000000000003",
            node_id,
            request_ordinal,
            request_byte.to_string().repeat(64),
            "plugin.fixture",
            "c".repeat(64),
            "d".repeat(64),
            "fixture",
            "https://fixture.invalid/v1/generate",
        )?)
    }

    #[test]
    fn provider_transport_is_domain_separated_bounded_and_canonical()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = TypeRegistry::built_in()?;
        let type_id: CanonicalTypeId = "sim:string@1".parse()?;
        let value = PluginValue::scalar(
            type_id.clone(),
            ScalarValue::String("result".to_owned()),
            &registry,
        )?;
        let port = ProviderTransportPort::checked(
            "result",
            true,
            vec![ProviderTransportValue::checked(
                type_id.to_string(),
                value.family(),
                value.abi_bytes()?,
            )?],
        )?;
        let request = ProviderTransportRequest::checked("provider.echo", vec![port.clone()])?;
        let response = ProviderTransportResponse::checked("provider.echo", vec![port])?;
        let request_bytes = request.to_bytes()?;
        let response_bytes = response.to_bytes()?;
        assert_ne!(request_bytes, response_bytes);
        assert_eq!(
            ProviderTransportRequest::from_bytes(&request_bytes)?,
            request
        );
        assert_eq!(
            ProviderTransportResponse::from_bytes(&response_bytes)?,
            response
        );
        assert!(ProviderTransportRequest::from_bytes(&response_bytes).is_err());
        assert!(ProviderTransportResponse::from_bytes(&request_bytes).is_err());

        assert!(
            ProviderTransportRequest::checked(
                "provider.echo",
                vec![
                    ProviderTransportPort::checked("z", false, Vec::new())?,
                    ProviderTransportPort::checked("a", false, Vec::new())?,
                ],
            )
            .is_err()
        );
        assert!(
            ProviderTransportPort::checked(
                "result",
                false,
                vec![ProviderTransportValue::checked(
                    type_id.to_string(),
                    ValueFamily::Scalar,
                    value.abi_bytes()?,
                )?,]
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn provider_v2_materialization_requires_the_exact_verified_terminal_receipt()
    -> Result<(), Box<dyn std::error::Error>> {
        let terminal_receipt = b"terminal-receipt".to_vec();
        let identity = crate::ProviderRuntimeReceiptIdentityV2 {
            provider: "plugin.fixture".to_owned(),
            method: ProviderHttpMethodV2::Post,
            endpoint: "https://provider.invalid/v2".to_owned(),
            ordered_headers_sha256: "1".repeat(64),
            secret_id: None,
            request_head_sha256: "2".repeat(64),
            request_body_sha256: "3".repeat(64),
            provider_manifest_sha256: "4".repeat(64),
            component_generation: 1,
            component_digest_sha256: "5".repeat(64),
            binding_generation: 1,
            binding_set_sha256: "6".repeat(64),
            accepted_cost_microunits: 0,
            request_ordinal: 1,
            response_status: 200,
            response_headers_sha256: "7".repeat(64),
            ordered_uploads_sha256: "8".repeat(64),
            ordered_chunks_sha256: "9".repeat(64),
            terminal_receipt_sha256: crate::provider_terminal_completed_receipt_sha256(
                &terminal_receipt,
            ),
            idempotency_identity_sha256: "a".repeat(64),
        };
        let origin = Instant::now();
        let issuer = crate::ProviderRuntimeReceiptIssuerV2::from_seed([0x81; 32], origin)?;
        let receipt = issuer.issue(
            identity.clone(),
            origin,
            origin + Duration::from_secs(30),
            [0x82; 32],
        )?;
        let verified =
            issuer
                .verifier()?
                .verify(&receipt, &identity, origin + Duration::from_secs(1))?;
        let registry = TypeRegistry::built_in()?;
        let type_id: CanonicalTypeId = "sim:string@1".parse()?;
        let value = PluginValue::scalar(
            type_id.clone(),
            ScalarValue::String("result".to_owned()),
            &registry,
        )?;
        let result = ProviderInvocationResultV2 {
            outputs: vec![ProviderMaterializedOutputV2 {
                port_id: "result".to_owned(),
                value: ProviderEncodedValueV2 {
                    type_id,
                    family: value.family(),
                    abi_bytes: value.abi_bytes()?,
                },
            }],
            receipt: terminal_receipt,
        };
        let response = materialize_provider_invocation_result_v2(
            &result,
            &verified,
            &registry,
            &CancellationToken::default(),
        )?;
        assert_eq!(response.ports().len(), 1);

        let mut mutated = result.clone();
        mutated.receipt[0] ^= 1;
        assert_eq!(
            materialize_provider_invocation_result_v2(
                &mutated,
                &verified,
                &registry,
                &CancellationToken::default(),
            ),
            Err(ProviderMaterializationError::InvalidTransportProjection)
        );
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert_eq!(
            materialize_provider_invocation_result_v2(&result, &verified, &registry, &cancellation,),
            Err(ProviderMaterializationError::Cancelled)
        );
        Ok(())
    }

    const RGB_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0c\x49\x44\x41\x54\x78\x9c\x63\x68\x70\x50\x00\x00\x02\x24\x00\xe1\xab\x59\x62\x27\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const RGBA_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0d\x49\x44\x41\x54\x78\x9c\x63\x68\x70\x50\x50\x00\x00\x03\x25\x01\x01\x04\xac\x69\xa5\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const LARGE_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x20\x00\x00\x00\x20\x08\x06\x00\x00\x00\x73\x7a\x7a\xf4\x00\x00\x00\x2b\x49\x44\x41\x54\x78\x9c\xed\xce\x21\x01\x00\x00\x0c\x04\xa1\xeb\x5f\x7a\x8b\xf1\x06\x81\xa7\xea\xc6\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\xb6\x81\x07\xb7\xb5\xfc\x2e\x65\xa9\xc9\xee\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";

    fn downloaded_image_context(
        memory_limit_bytes: u64,
        cancellation: CancellationToken,
    ) -> Result<
        (
            NativeNodeContext,
            Arc<sim_tensor::CpuBackend>,
            crate::NativeHandleStoreGeneration,
        ),
        Box<dyn std::error::Error>,
    > {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit_bytes)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(memory_limit_bytes)?;
        let attempt_id = AttemptId(Uuid::from_u128(3));
        let node_id = NodeId::from("node.fixture");
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x733),
            attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let generation = crate::NativeHandleStoreGeneration::with_capacities(4, 1024 * 1024)?;
        let store = generation.handle_store_for_attempt(attempt_id);
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(2)),
            attempt_id,
            node_id,
            cancellation,
            scratch,
            store,
            NativeNodeServices::checked(None, None, Some(compute))?.with_provider_execution(
                sim_nodes::NativeProviderExecutionIdentity::checked("a".repeat(64))?,
            ),
        )?;
        Ok((context, backend, generation))
    }

    #[test]
    fn downloaded_image_receipts_decode_exact_rgba_and_preserve_terminal_resolution()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = Arc::new(ProviderResultReceiptIssuer::from_seed([37; 32], origin)?);
        let mut session = ProviderResultReceiptSession::new(issuer, 1024, 0)?;
        let (context, backend, generation) =
            downloaded_image_context(1024 * 1024, CancellationToken::default())?;
        let first_identity = invocation_identity("node.fixture", 0, 'e')?;
        let first_receipt = session.issue(
            first_identity,
            b"polling-json".to_vec(),
            origin,
            origin + Duration::from_secs(30),
        )?;
        let mut receipts = vec![first_receipt.clone()];
        let registry = TypeRegistry::built_in()?;
        for (ordinal, bytes, alpha) in [
            (1, RGB_DOWNLOADED_PNG, 1.0),
            (2, RGBA_DOWNLOADED_PNG, 32.0 / 255.0),
        ] {
            let identity = invocation_identity("node.fixture", ordinal, 'f')?;
            let receipt = session.issue(
                identity.clone(),
                bytes.to_vec(),
                origin,
                origin + Duration::from_secs(30),
            )?;
            receipts.push(receipt.clone());
            let payload = session.materialize_downloaded_image(
                &receipt,
                &identity,
                origin + Duration::from_secs(1),
                &context,
            )?;
            assert!(generation.is_empty());
            assert_eq!(
                payload.handle_type()?,
                NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?
            );
            let NativeStoredPayload::Tensor(image) = &payload else {
                return Err("downloaded image was not a canonical tensor".into());
            };
            assert_eq!(image.tensor().descriptor().shape(), [1, 1, 1, 4]);
            assert_eq!(image.tensor().descriptor().dtype(), DType::F32);
            let values = image
                .tensor()
                .contiguous_bytes()?
                .chunks_exact(4)
                .map(|chunk| chunk.try_into().map(f32::from_ne_bytes))
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(values, [128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, alpha]);
            let value = ProviderTransportValue::from_native_payload(
                registry.resolve("IMAGE")?.to_string(),
                ValueFamily::Tensor,
                &payload,
            )?;
            assert_eq!(value.encoding(), ProviderTransportEncoding::NativePayload);
            let restored = value.materialize_native_payload(
                &NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?,
                identity.plugin_id(),
                &context,
            )?;
            assert_eq!(restored.digest_sha256(), payload.digest_sha256());
            drop(restored);
            drop(payload);
            assert_eq!(backend.memory_snapshot().current_bytes, 0);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        assert_eq!(session.next_request_ordinal()?, 3);
        assert_eq!(session.issued_order.len(), 3);
        let receipt_set = ProviderResultReceiptSet::new(receipts)?;
        assert_eq!(receipt_set.receipts().first(), Some(&first_receipt));
        let resolved =
            session.resolve_receipt_set(&receipt_set, origin + Duration::from_secs(1))?;
        assert_eq!(resolved.len(), 3);
        assert_eq!(
            resolved.first().ok_or("missing polling result")?.response(),
            b"polling-json"
        );
        session.finish()?;
        Ok(())
    }

    #[test]
    fn downloaded_image_materialization_rejects_invalid_bounds_identity_and_cancellation_without_residue()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = Arc::new(ProviderResultReceiptIssuer::from_seed([41; 32], origin)?);
        let (context, backend, generation) =
            downloaded_image_context(1024 * 1024, CancellationToken::default())?;
        for bytes in [b"malformed-image".as_slice(), LARGE_DOWNLOADED_PNG] {
            let mut session = ProviderResultReceiptSession::new(issuer.clone(), 1024, 0)?;
            let identity = invocation_identity("node.fixture", 0, 'e')?;
            let receipt = session.issue(
                identity.clone(),
                bytes.to_vec(),
                origin,
                origin + Duration::from_secs(30),
            )?;
            assert!(matches!(
                session.materialize_downloaded_image(
                    &receipt,
                    &identity,
                    origin + Duration::from_secs(1),
                    &context
                ),
                Err(ProviderMaterializationError::ImageDecode(_))
            ));
            assert_eq!(backend.memory_snapshot().current_bytes, 0);
            assert_eq!(context.scratch.in_use_bytes(), 0);
            assert!(generation.is_empty());
            assert_eq!(
                session.resolve(&receipt, &identity, origin + Duration::from_secs(1))?,
                bytes
            );
            session.finish()?;
        }
        let identity = invocation_identity("node.fixture", 0, 'e')?;
        let mut session = ProviderResultReceiptSession::new(issuer, 1024, 0)?;
        assert_eq!(
            session.issue(
                identity.clone(),
                vec![0; 1025],
                origin,
                origin + Duration::from_secs(30)
            ),
            Err(ProviderMaterializationError::ResponseTooLarge)
        );
        assert_eq!(session.next_request_ordinal()?, 0);
        let receipt = session.issue(
            identity.clone(),
            RGB_DOWNLOADED_PNG.to_vec(),
            origin,
            origin + Duration::from_secs(30),
        )?;
        let mut stale_context = context.clone();
        stale_context.node_id = NodeId::from("another.node");
        assert_eq!(
            session
                .materialize_downloaded_image(
                    &receipt,
                    &identity,
                    origin + Duration::from_secs(1),
                    &stale_context
                )
                .err(),
            Some(ProviderMaterializationError::ReceiptRejected)
        );
        let mut forged = receipt.clone();
        *forged.last_mut().ok_or("receipt was empty")? ^= 1;
        assert_eq!(
            session
                .materialize_downloaded_image(
                    &forged,
                    &identity,
                    origin + Duration::from_secs(1),
                    &context
                )
                .err(),
            Some(ProviderMaterializationError::ReceiptRejected)
        );
        let cancelled = context;
        cancelled.cancellation.cancel();
        assert_eq!(
            session
                .materialize_downloaded_image(
                    &receipt,
                    &identity,
                    origin + Duration::from_secs(1),
                    &cancelled
                )
                .err(),
            Some(ProviderMaterializationError::Cancelled)
        );
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        assert!(generation.is_empty());
        let (constrained, constrained_backend, constrained_generation) =
            downloaded_image_context(4, CancellationToken::default())?;
        assert!(matches!(
            session.materialize_downloaded_image(
                &receipt,
                &identity,
                origin + Duration::from_secs(1),
                &constrained
            ),
            Err(ProviderMaterializationError::ImageAllocation(_))
        ));
        assert_eq!(constrained_backend.memory_snapshot().current_bytes, 0);
        assert_eq!(constrained.scratch.in_use_bytes(), 0);
        assert!(constrained_generation.is_empty());
        session.abort();
        Ok(())
    }

    #[test]
    fn provider_native_image_and_video_round_trip_through_the_canonical_lower_owners()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x710));
        let node_id = NodeId::from("provider-image-materializer");
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x711),
            attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let store_generation = crate::NativeHandleStoreGeneration::with_capacities(4, 1024 * 1024)?;
        let store = store_generation.handle_store_for_attempt(attempt_id);
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x712)),
            attempt_id,
            node_id,
            CancellationToken::default(),
            scratch,
            store,
            NativeNodeServices::checked(None, None, Some(compute))?,
        )?;
        let execution_context = context.compute_session()?.execution_context(&context)?;
        let image = ImageTensor::from_f32(
            &backend,
            &execution_context,
            1,
            1,
            2,
            3,
            &[0.0, 0.25, 0.5, 0.75, 1.0, 0.125],
        )?;
        let source = NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
            NativeTensorRole::Image,
            image,
        )?));
        let native_payload = ProviderNativePayload::from_stored(&source)
            .map_err(|error| format!("provider native image projection failed: {error}"))?;
        native_payload
            .validate()
            .map_err(|error| format!("provider native image validation failed: {error}"))?;
        let value = ProviderTransportValue::from_native_payload(
            "sim:image@1",
            ValueFamily::Tensor,
            &source,
        )
        .map_err(|error| format!("provider image transport encoding failed: {error}"))?;
        assert_eq!(value.encoding(), ProviderTransportEncoding::NativePayload);
        let materialized = value
            .materialize_native_payload(
                &NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?,
                "plugin.fixture",
                &context,
            )
            .map_err(|error| format!("provider image materialization failed: {error}"))?;
        assert_eq!(source.handle_type()?, materialized.handle_type()?);
        assert_eq!(source.digest_sha256(), materialized.digest_sha256());
        let NativeStoredPayload::Tensor(materialized) = materialized else {
            return Err("provider image changed stored payload variant".into());
        };
        assert_eq!(materialized.tensor().descriptor().shape(), [1, 1, 2, 3]);
        let empty_audio = NativeAudioPayload::checked(
            tensor_from_f32_with_context_exact_native(
                backend.as_ref(),
                &[1, 1, 0],
                &[],
                DType::F32,
                backend.as_ref().device(),
                &execution_context,
            )?,
            1,
        )?;
        let transport = native_audio_persistence_transport(&empty_audio)?;
        let persisted = ProviderTransportResponse::from_bytes(&transport.to_bytes()?)?;
        let materialized_audio = persisted.ports()[0].values()[0].materialize_native_payload(
            &NativeHandleType::new(NativeHandleKind::Audio, "AUDIO")?,
            "plugin.fixture",
            &context,
        )?;
        let NativeStoredPayload::Audio(materialized_audio) = materialized_audio else {
            return Err("provider AUDIO persistence changed payload variant".into());
        };
        assert_eq!(materialized_audio.sample_rate(), 1);
        assert_eq!(
            materialized_audio.waveform().descriptor().shape(),
            [1, 1, 0]
        );
        assert_eq!(
            materialized_audio.semantic_digest_sha256(),
            empty_audio.semantic_digest_sha256()
        );
        let video = NativeStoredPayload::Video(Arc::new(NativeVideoPayload::checked(
            materialized.tensor().clone(),
            1_054_475_631_502_295,
            35_184_372_088_832,
            NativeVideoBitDepth::Ten,
            None,
            None,
            BTreeMap::from([("source".to_owned(), "provider".to_owned())]),
        )?));
        let value = ProviderTransportValue::from_native_payload(
            "sim:video@1",
            ValueFamily::Tensor,
            &video,
        )?;
        let round_trip = value.materialize_native_payload(
            &NativeHandleType::new(NativeHandleKind::Video, "VIDEO")?,
            "plugin.fixture",
            &context,
        )?;
        assert_eq!(video.digest_sha256(), round_trip.digest_sha256());
        let NativeStoredPayload::Video(round_trip) = round_trip else {
            return Err("provider video changed stored payload variant".into());
        };
        assert_eq!(round_trip.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            round_trip.frame_rate(),
            (1_054_475_631_502_295, 35_184_372_088_832)
        );

        let component = NativeVideoPayload::checked(
            materialized.tensor().clone(),
            1_054_475_631_502_295,
            35_184_372_088_832,
            NativeVideoBitDepth::Eight,
            None,
            None,
            BTreeMap::new(),
        )?;
        let encoded_bytes = b"HMP4";
        let descriptor = TensorDescriptor::contiguous(
            vec![encoded_bytes.len() as u64],
            DType::U8,
            DeviceId::CPU,
            execution_context.stream,
        )?;
        let (encoded_bytes, _) =
            backend.upload_bytes(descriptor, encoded_bytes, &execution_context)?;
        let encoded = NativeVideoPayload::checked_h264_mp4_from_component(
            &component,
            encoded_bytes,
            Sha256::digest(b"HMP4").into(),
            (2, 1),
            (2_997, 100),
            1,
        )?;
        assert!(matches!(
            ProviderNativePayload::from_stored(&NativeStoredPayload::Video(Arc::new(encoded))),
            Err(ProviderMaterializationError::InvalidNativePayload)
        ));
        Ok(())
    }

    #[test]
    fn provider_schemas_are_exact_and_owned_by_the_native_materializer()
    -> Result<(), Box<dyn std::error::Error>> {
        let transport: CanonicalTypeId = NATIVE_PROVIDER_TRANSPORT_SCHEMA.parse()?;
        let materializer: CanonicalTypeId = NATIVE_PROVIDER_MATERIALIZER_SCHEMA.parse()?;
        validate_native_provider_schemas(&transport, &materializer)?;

        let wrong_transport: CanonicalTypeId = "zed:other-provider-transport@1".parse()?;
        assert_eq!(
            validate_native_provider_schemas(&wrong_transport, &materializer),
            Err(ProviderMaterializationError::UnsupportedTransportSchema)
        );
        let wrong_materializer: CanonicalTypeId = "zed:other-provider-materializer@1".parse()?;
        assert_eq!(
            validate_native_provider_schemas(&transport, &wrong_materializer),
            Err(ProviderMaterializationError::UnsupportedMaterializerSchema)
        );
        Ok(())
    }

    #[test]
    fn provider_result_session_resolves_exact_ordered_one_time_receipts()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = Arc::new(ProviderResultReceiptIssuer::from_seed([23; 32], origin)?);
        let mut session = ProviderResultReceiptSession::new(issuer.clone(), 1_024, 4)?;
        let first_identity = invocation_identity("node.fixture", 4, 'e')?;
        let second_identity = invocation_identity("node.fixture", 5, 'f')?;
        let first_response = b"first-provider-response".to_vec();
        let second_response = b"second-provider-response".to_vec();
        let first_receipt = session.issue(
            first_identity.clone(),
            first_response.clone(),
            origin + Duration::from_secs(1),
            origin + Duration::from_secs(31),
        )?;
        let second_receipt = session.issue(
            second_identity.clone(),
            second_response.clone(),
            origin + Duration::from_secs(2),
            origin + Duration::from_secs(32),
        )?;
        assert!(
            !first_receipt
                .windows(first_response.len())
                .any(|window| window == first_response)
        );
        assert_eq!(
            session.resolve(
                &second_receipt,
                &second_identity,
                origin + Duration::from_secs(3),
            ),
            Err(ProviderMaterializationError::ReceiptOutOfOrder)
        );
        assert_eq!(
            session.resolve(
                &first_receipt,
                &invocation_identity("node.other", 4, 'e')?,
                origin + Duration::from_secs(3),
            ),
            Err(ProviderMaterializationError::ReceiptRejected)
        );
        assert_eq!(
            session.resolve(
                &first_receipt,
                &first_identity,
                origin + Duration::from_secs(3),
            )?,
            first_response
        );
        assert_eq!(
            session.resolve(
                &first_receipt,
                &first_identity,
                origin + Duration::from_secs(3),
            ),
            Err(ProviderMaterializationError::UnknownReceipt)
        );
        assert_eq!(
            session.resolve(
                &second_receipt,
                &second_identity,
                origin + Duration::from_secs(3),
            )?,
            second_response
        );
        session.finish()?;

        let mut unresolved = ProviderResultReceiptSession::new(issuer, 8, 0)?;
        assert_eq!(
            unresolved.issue(
                invocation_identity("node.fixture", 1, '1')?,
                vec![1],
                origin,
                origin + Duration::from_secs(1),
            ),
            Err(ProviderMaterializationError::RequestOrdinalOutOfOrder)
        );
        let receipt = unresolved.issue(
            invocation_identity("node.fixture", 0, '0')?,
            vec![1],
            origin,
            origin + Duration::from_secs(1),
        )?;
        assert!(!receipt.is_empty());
        assert_eq!(
            unresolved.finish(),
            Err(ProviderMaterializationError::UnresolvedReceipts)
        );
        Ok(())
    }

    #[test]
    fn provider_result_session_resolves_a_complete_receipt_set_atomically()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = Arc::new(ProviderResultReceiptIssuer::from_seed([29; 32], origin)?);
        let mut session = ProviderResultReceiptSession::new(issuer, 1_024, 7)?;
        let first_identity = invocation_identity("node.fixture", 7, 'a')?;
        let second_identity = invocation_identity("node.fixture", 8, 'b')?;
        let first_receipt = session.issue(
            first_identity.clone(),
            b"first".to_vec(),
            origin + Duration::from_secs(1),
            origin + Duration::from_secs(31),
        )?;
        let second_receipt = session.issue(
            second_identity.clone(),
            b"second".to_vec(),
            origin + Duration::from_secs(2),
            origin + Duration::from_secs(32),
        )?;
        let reversed =
            ProviderResultReceiptSet::new(vec![second_receipt.clone(), first_receipt.clone()])?;
        assert_eq!(
            session.resolve_receipt_set(&reversed, origin + Duration::from_secs(3)),
            Err(ProviderMaterializationError::ReceiptOutOfOrder)
        );

        let receipt_set = ProviderResultReceiptSet::new(vec![first_receipt, second_receipt])?;
        let resolved =
            session.resolve_receipt_set(&receipt_set, origin + Duration::from_secs(3))?;
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].identity(), &first_identity);
        assert_eq!(resolved[0].response(), b"first");
        assert_eq!(resolved[1].identity(), &second_identity);
        assert_eq!(resolved[1].response(), b"second");
        session.finish()?;
        Ok(())
    }
    #[test]
    fn authenticated_response_discriminator_preserves_structured_and_png_settlement()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = Arc::new(ProviderResultReceiptIssuer::from_seed([137; 32], origin)?);
        let mut session = ProviderResultReceiptSession::new(issuer, 4096, 0)?;
        let (context, backend, generation) =
            downloaded_image_context(1024 * 1024, CancellationToken::default())?;
        let structured = ProviderTransportResponse::checked(
            "node.fixture",
            vec![ProviderTransportPort::checked(
                "output_0",
                false,
                Vec::new(),
            )?],
        )?
        .to_bytes()?;
        let mut receipts = Vec::new();
        for (ordinal, original, expected) in [
            (0, structured.as_slice(), true),
            (1, RGB_DOWNLOADED_PNG, false),
            (2, b"{not valid JSON".as_slice(), false),
        ] {
            let identity = invocation_identity("node.fixture", ordinal, 'f')?;
            let receipt = session.issue(
                identity.clone(),
                original.to_vec(),
                origin,
                origin + Duration::from_secs(30),
            )?;
            let ordinal_after_issue = session.next_request_ordinal()?;
            assert_eq!(
                session.is_structured_transport_response(
                    &receipt,
                    &identity,
                    origin + Duration::from_secs(1),
                    &context
                )?,
                expected
            );
            assert_eq!(session.next_request_ordinal()?, ordinal_after_issue);
            assert_eq!(session.issued_order.len(), receipts.len() + 1);
            assert!(generation.is_empty());
            assert_eq!(backend.memory_snapshot().current_bytes, 0);
            assert_eq!(context.scratch.in_use_bytes(), 0);
            receipts.push(receipt);
        }
        let set = ProviderResultReceiptSet::new(receipts)?;
        let resolved = session.resolve_receipt_set(&set, origin + Duration::from_secs(1))?;
        assert_eq!(
            resolved
                .first()
                .ok_or("missing structured settlement")?
                .response(),
            structured
        );
        assert_eq!(
            resolved.get(1).ok_or("missing PNG settlement")?.response(),
            RGB_DOWNLOADED_PNG
        );
        session.finish()?;
        Ok(())
    }

    #[test]
    fn authenticated_response_discriminator_rejects_foreign_forged_expired_cancelled_and_oversized_results()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = Arc::new(ProviderResultReceiptIssuer::from_seed([138; 32], origin)?);
        let mut session = ProviderResultReceiptSession::new(issuer.clone(), 4096, 0)?;
        let cancellation = CancellationToken::default();
        let (context, backend, generation) =
            downloaded_image_context(1024 * 1024, cancellation.clone())?;
        let identity = invocation_identity("node.fixture", 0, 'f')?;
        let receipt = session.issue(
            identity.clone(),
            RGB_DOWNLOADED_PNG.to_vec(),
            origin,
            origin + Duration::from_secs(30),
        )?;
        let foreign = ProviderResultReceiptSession::new(issuer, 4096, 0)?;
        assert!(matches!(
            foreign.is_structured_transport_response(&receipt, &identity, origin, &context),
            Err(ProviderMaterializationError::UnknownReceipt)
        ));
        let changed_identity = invocation_identity("different.node", 0, 'f')?;
        assert!(matches!(
            session.is_structured_transport_response(&receipt, &changed_identity, origin, &context),
            Err(ProviderMaterializationError::ReceiptRejected)
        ));
        let mut forged: serde_json::Value = serde_json::from_slice(&receipt)?;
        forged
            .as_object_mut()
            .ok_or("receipt is not an object")?
            .insert(
                "result_sha256".to_owned(),
                serde_json::json!("0".repeat(64)),
            );
        assert!(matches!(
            session.is_structured_transport_response(
                &serde_json::to_vec(&forged)?,
                &identity,
                origin,
                &context
            ),
            Err(ProviderMaterializationError::ReceiptRejected)
        ));
        assert!(matches!(
            session.is_structured_transport_response(
                &receipt,
                &identity,
                origin + Duration::from_secs(31),
                &context
            ),
            Err(ProviderMaterializationError::ReceiptRejected)
        ));
        session.maximum_response_bytes = 1;
        assert!(matches!(
            session.is_structured_transport_response(&receipt, &identity, origin, &context),
            Err(ProviderMaterializationError::ResponseTooLarge)
        ));
        session.maximum_response_bytes = 4096;
        cancellation.cancel();
        assert!(matches!(
            session.is_structured_transport_response(&receipt, &identity, origin, &context),
            Err(ProviderMaterializationError::Cancelled)
        ));
        assert!(generation.is_empty());
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        session.abort();
        Ok(())
    }
    #[test]
    fn structured_response_classifier_propagates_operational_errors()
    -> Result<(), Box<dyn std::error::Error>> {
        let response = ProviderTransportResponse::checked("node.fixture", Vec::new())?;
        assert_eq!(
            ProviderResultReceiptSession::classify_structured_transport_response(Ok(response)),
            Ok(true)
        );
        for error in [
            ProviderMaterializationError::InvalidTransportProjection,
            ProviderMaterializationError::InvalidNativePayload,
        ] {
            assert_eq!(
                ProviderResultReceiptSession::classify_structured_transport_response(Err(error)),
                Ok(false)
            );
        }
        for error in [
            ProviderMaterializationError::Cancelled,
            ProviderMaterializationError::UnsupportedTransportSchema,
            ProviderMaterializationError::UnsupportedMaterializerSchema,
            ProviderMaterializationError::ResponseTooLarge,
            ProviderMaterializationError::ReceiptSessionFinished,
            ProviderMaterializationError::RequestOrdinalOutOfOrder,
            ProviderMaterializationError::ReceiptRejected,
            ProviderMaterializationError::UnknownReceipt,
            ProviderMaterializationError::ReceiptOutOfOrder,
            ProviderMaterializationError::UnresolvedReceipts,
            ProviderMaterializationError::InvalidReceiptAuthority,
            ProviderMaterializationError::ImageDecode("retained decoder failure".to_owned()),
            ProviderMaterializationError::ImageAllocation("retained allocator failure".to_owned()),
        ] {
            assert_eq!(
                ProviderResultReceiptSession::classify_structured_transport_response(Err(
                    error.clone()
                )),
                Err(error)
            );
        }
        Ok(())
    }

    #[test]
    fn structured_response_parser_format_rejection_and_canonical_bounds_remain_explicit()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = Arc::new(ProviderResultReceiptIssuer::from_seed([142; 32], origin)?);
        assert!(matches!(
            ProviderResultReceiptSession::new(
                issuer,
                MAX_PROVIDER_MATERIALIZATION_RESPONSE_BYTES + 1,
                0
            ),
            Err(ProviderMaterializationError::ResponseTooLarge)
        ));
        let bytes = ProviderTransportResponse::checked(
            "different.class",
            vec![ProviderTransportPort::checked(
                "wrong_output",
                false,
                Vec::new(),
            )?],
        )?
        .to_bytes()?;
        assert!(
            ProviderResultReceiptSession::classify_structured_transport_response(
                ProviderTransportResponse::from_bytes(&bytes)
            )?
        );
        let short_bound = bytes.len().checked_sub(1).ok_or("response is empty")?;
        assert_eq!(
            decode_transport_projection(PROVIDER_TRANSPORT_RESPONSE_DOMAIN, &bytes, short_bound),
            Err(ProviderMaterializationError::InvalidTransportProjection)
        );
        let mut wrong_version = bytes;
        let version = wrong_version
            .get_mut(
                PROVIDER_TRANSPORT_RESPONSE_DOMAIN.len()
                    ..PROVIDER_TRANSPORT_RESPONSE_DOMAIN.len() + 2,
            )
            .ok_or("missing version")?;
        version.copy_from_slice(&(PROVIDER_TRANSPORT_VERSION + 1).to_le_bytes());
        for malformed in [
            RGB_DOWNLOADED_PNG,
            b"{not valid JSON".as_slice(),
            PROVIDER_TRANSPORT_RESPONSE_DOMAIN,
            wrong_version.as_slice(),
        ] {
            assert!(
                !ProviderResultReceiptSession::classify_structured_transport_response(
                    ProviderTransportResponse::from_bytes(malformed)
                )?
            );
        }
        Ok(())
    }
}
