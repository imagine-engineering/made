use crate::{
    CertifiedGeneralVideoCodecDependencyClosure, CertifiedVideoCodecDependencyClosure,
    NativeGeneralVideoCodecRuntimeVersions, NativeH264Mp4SequenceLimits,
    NativeLtxvH264DecodeLimits, NativeLtxvH264DemuxLimits, NativeLtxvH264EncodeLimits,
    NativeLtxvH264PreprocessLimits, NativeVideoCodecAv1EncodeError, NativeVideoCodecBindingError,
    NativeVideoCodecDecodeError, NativeVideoCodecH264EncodeError, NativeVideoCodecLoadError,
    NativeVideoCodecLtxvAdmissionError, NativeVideoCodecLtxvPreprocessError,
    NativeVideoCodecRuntimeVersions, NativeVideoCodecSuite, NativeVideoCodecSuiteAdmissionError,
    NativeVideoCodecVp9EncodeError, NativeVideoContainerMetadata, NativeVp9WebmBatchLimits,
    NativeVp9WebmEncodeLimits, bind_certified_general_video_codec_abi,
    bind_certified_video_codec_abi, load_certified_general_video_codec_closure,
    load_certified_video_codec_closure,
};
use futures::channel::oneshot;
use futures::future::{BoxFuture, FutureExt};
use sha2::{Digest, Sha256};
use sim_media::{
    NativeDecodedVideo, NativeVideoAlphaPolicy, NativeVideoBitDepth, NativeVideoCodec,
    NativeVideoCodecLimits, NativeVideoCodecPlanError, NativeVideoContainer, NativeVideoCrf,
    NativeVideoDecodeLimits, NativeVideoDecodeRequest, NativeVideoEncodeOptions,
    NativeVideoMetadataPolicy, NativeVideoPayload, NativeVideoPixelFormat,
    plan_native_video_encode,
};
use sim_nodes::{
    NativeComponentH264Mp4BackingRequest, NativeComponentH264Mp4BackingService,
    NativeComponentH264Mp4BackingServiceError, NativeComponentH264Mp4BackingServiceIdentity,
    NativeEncodedWebm, NativeLtxvPreprocessService, NativeLtxvPreprocessServiceError,
    NativeLtxvPreprocessServiceIdentity, NativeWebmEncodeRequest, NativeWebmEncodeService,
    NativeWebmEncodeServiceError, NativeWebmEncodeServiceIdentity,
};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, ImageTensor, ScratchReservation, StreamId,
    Tensor, TensorDescriptor, TensorError,
};
use sim_types::CancellationToken;
use std::{
    fmt, io,
    sync::{Arc, Mutex, TryLockError, mpsc},
    thread::{self, JoinHandle},
};
use thiserror::Error;

const VIDEO_CODEC_THREAD_NAME: &str = "sim-video-codec";
const VIDEO_CODEC_THREAD_IDENTITY_VERSION: &str = "zed.sim.video-codec-thread.v9";
#[allow(dead_code, reason = "consumed by the following SaveWEBM node adapter")]
const WEBM_NODE_SERVICE_IDENTITY_VERSION: &str = "zed.sim.webm-node-service.v1";
#[allow(
    dead_code,
    reason = "constructed by the following encoded VIDEO backing adapter"
)]
const COMPONENT_H264_MP4_BACKING_SERVICE_IDENTITY_VERSION: &str =
    "zed.sim.component-h264-mp4-backing-service.v2";
const GENERAL_VIDEO_DECODE_SERVICE_IDENTITY_VERSION: &str =
    "zed.sim.general-video-decode-service.v1";
const REVIEWED_GENERAL_VIDEO_CODEC_SCRATCH_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const REVIEWED_GENERAL_VIDEO_CODEC_OUTPUT_BYTES: usize = 512 * 1024 * 1024;
const REVIEWED_GENERAL_VIDEO_CODEC_AVIO_BYTES: usize = 64 * 1024;
const REVIEWED_GENERAL_VIDEO_CODEC_NATIVE_SESSION_BYTES: u64 = 1024 * 1024 * 1024;
const REVIEWED_GENERAL_VIDEO_CODEC_PACKET_ITERATIONS: usize = 1_000_000;
const REVIEWED_GENERAL_VIDEO_CODEC_FRAMES: usize = 64;
const REVIEWED_COMPONENT_H264_MP4_FRAMES: usize = 240;
const REVIEWED_GENERAL_VIDEO_CODEC_PIXELS: u64 = 268_435_456;

#[allow(
    dead_code,
    reason = "constructed by the native LTXVPreprocess adapter and VP9 owned-byte bridge"
)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeLtxvCodecThreadIdentity {
    target: String,
    primary_catalog_sha256: String,
    configuration_sha256: String,
    runtime_versions: NativeVideoCodecRuntimeVersions,
}

#[allow(
    dead_code,
    reason = "consumed by the native LTXVPreprocess adapter and VP9 owned-byte bridge"
)]
impl NativeLtxvCodecThreadIdentity {
    pub(crate) fn target(&self) -> &str {
        &self.target
    }

    pub(crate) fn primary_catalog_sha256(&self) -> &str {
        &self.primary_catalog_sha256
    }

    pub(crate) fn configuration_sha256(&self) -> &str {
        &self.configuration_sha256
    }

    pub(crate) fn runtime_versions(&self) -> NativeVideoCodecRuntimeVersions {
        self.runtime_versions
    }
}

#[allow(
    dead_code,
    reason = "returned through the native LTXVPreprocess adapter and VP9 owned-byte bridge"
)]
#[derive(Debug, Error)]
pub(crate) enum NativeLtxvCodecThreadError {
    #[error("native video codec thread startup or request was cancelled")]
    Cancelled,
    #[error("native video codec thread could not be spawned: {0}")]
    ThreadSpawn(#[source] io::Error),
    #[error("native video codec thread stopped before completing the operation")]
    ThreadStopped,
    #[error("native video codec thread panicked")]
    ThreadPanicked,
    #[error("native video codec thread state was poisoned")]
    StatePoisoned,
    #[error("native video codec request queue is full")]
    Busy,
    #[error("native video codec request carried scratch from the wrong backend: {0}")]
    InvalidScratch(#[source] Box<TensorError>),
    #[error("native video codec request exhausted its reviewed resources")]
    ResourceExhausted,
    #[error("native video codec loading failed: {0}")]
    Load(#[source] Box<NativeVideoCodecLoadError>),
    #[error("native video codec ABI binding failed: {0}")]
    Binding(#[source] Box<NativeVideoCodecBindingError>),
    #[error("native LTXV codec admission failed: {0}")]
    Admission(#[source] Box<NativeVideoCodecLtxvAdmissionError>),
    #[error("native video codec-suite admission failed: {0}")]
    SuiteAdmission(#[source] Box<NativeVideoCodecSuiteAdmissionError>),
    #[error("native LTXV preprocessing failed: {0}")]
    Preprocess(#[source] Box<NativeVideoCodecLtxvPreprocessError>),
    #[error("native VP9 WebM encoding failed: {0}")]
    Vp9Encode(#[source] Box<NativeVideoCodecVp9EncodeError>),
    #[error("native AV1 WebM encoding failed: {0}")]
    Av1Encode(#[source] Box<NativeVideoCodecAv1EncodeError>),
    #[error("native H.264 MP4 encoding failed: {0}")]
    H264Encode(#[source] Box<NativeVideoCodecH264EncodeError>),
    #[error("native encoded-byte materialization failed: {0}")]
    EncodedOutput(#[source] Box<TensorError>),
    #[error("native general-video decoding failed: {0}")]
    Decode(#[source] Box<NativeVideoCodecDecodeError>),
}

struct NativeLtxvCodecThreadRequest {
    invocation: NativeLtxvCodecThreadInvocation,
    response: oneshot::Sender<Result<NativeVideoCodecThreadOutput, NativeLtxvCodecThreadError>>,
}

struct NativeLtxvCodecThreadInvocation {
    operation: NativeVideoCodecThreadOperation,
    stream: StreamId,
    scratch: ScratchReservation,
    cancellation: CancellationToken,
}

enum NativeVideoCodecThreadOperation {
    Preprocess {
        image: ImageTensor,
        compression: u8,
    },
    EncodeVp9Webm {
        images: ImageTensor,
        frame_rate: (u64, u64),
        crf: NativeVideoCrf,
        limits: NativeVp9WebmBatchLimits,
        metadata: NativeVideoContainerMetadata,
    },
    EncodeAv1Webm {
        images: ImageTensor,
        frame_rate: (u64, u64),
        crf: NativeVideoCrf,
        limits: NativeVp9WebmBatchLimits,
        metadata: NativeVideoContainerMetadata,
    },
    EncodeH264Mp4 {
        images: ImageTensor,
        frame_rate: (u64, u64),
        limits: NativeH264Mp4SequenceLimits,
        bit_depth: NativeVideoBitDepth,
    },
    DecodeGeneralVideo {
        request: NativeVideoDecodeRequest,
    },
}

enum NativeVideoCodecThreadOutput {
    Image(ImageTensor),
    Vp9Webm(NativeOwnedVp9Webm),
    Av1Webm(NativeOwnedAv1Webm),
    H264Mp4(NativeOwnedH264Mp4),
    DecodedVideo(NativeDecodedVideo),
}

#[allow(
    dead_code,
    reason = "consumed by the following encoded-backing video service"
)]
#[derive(Debug)]
pub(crate) struct NativeOwnedH264Mp4 {
    bytes: Tensor,
    content_sha256: [u8; 32],
    width: i32,
    height: i32,
    frame_rate: (i32, i32),
    frame_count: usize,
    bit_depth: NativeVideoBitDepth,
}

#[allow(
    dead_code,
    reason = "consumed by the following encoded-backing video service"
)]
impl NativeOwnedH264Mp4 {
    pub(crate) fn encoded_bytes(&self) -> Result<&[u8], TensorError> {
        self.bytes.contiguous_bytes()
    }

    pub(crate) const fn dimensions(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    pub(crate) const fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }

    pub(crate) const fn frame_rate(&self) -> (i32, i32) {
        self.frame_rate
    }

    pub(crate) const fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub(crate) const fn bit_depth(&self) -> NativeVideoBitDepth {
        self.bit_depth
    }

    pub(crate) const fn pixel_format(&self) -> NativeVideoPixelFormat {
        match self.bit_depth {
            NativeVideoBitDepth::Eight => NativeVideoPixelFormat::Yuv420p,
            NativeVideoBitDepth::Ten => NativeVideoPixelFormat::Yuv420p10le,
        }
    }

    pub(crate) const fn has_alpha(&self) -> bool {
        false
    }

    pub(crate) const fn codec(&self) -> NativeVideoCodec {
        NativeVideoCodec::H264
    }

    pub(crate) const fn container(&self) -> NativeVideoContainer {
        NativeVideoContainer::Mp4
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(self) -> (Tensor, [u8; 32], i32, i32, (i32, i32), usize) {
        (
            self.bytes,
            self.content_sha256,
            self.width,
            self.height,
            self.frame_rate,
            self.frame_count,
        )
    }
}

#[allow(
    dead_code,
    reason = "consumed by the following SaveWEBM prepared-effect adapter"
)]
#[derive(Debug)]
pub(crate) struct NativeOwnedVp9Webm {
    bytes: Tensor,
    content_sha256: [u8; 32],
    width: i32,
    height: i32,
    frame_rate: (i32, i32),
    frame_count: usize,
    has_alpha: bool,
}

#[allow(
    dead_code,
    reason = "consumed by the following SaveWEBM prepared-effect adapter"
)]
impl NativeOwnedVp9Webm {
    pub(crate) fn encoded_bytes(&self) -> Result<&[u8], TensorError> {
        self.bytes.contiguous_bytes()
    }

    pub(crate) fn dimensions(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    pub(crate) fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }

    pub(crate) fn frame_rate(&self) -> (i32, i32) {
        self.frame_rate
    }

    pub(crate) fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub(crate) fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    #[allow(clippy::type_complexity)]
    fn into_parts(self) -> (Tensor, [u8; 32], i32, i32, (i32, i32), usize, bool) {
        (
            self.bytes,
            self.content_sha256,
            self.width,
            self.height,
            self.frame_rate,
            self.frame_count,
            self.has_alpha,
        )
    }
}

#[allow(
    dead_code,
    reason = "consumed by the following SaveWEBM prepared-effect adapter"
)]
#[derive(Debug)]
pub(crate) struct NativeOwnedAv1Webm {
    bytes: Tensor,
    content_sha256: [u8; 32],
    width: i32,
    height: i32,
    frame_rate: (i32, i32),
    frame_count: usize,
}

#[allow(
    dead_code,
    reason = "consumed by the following SaveWEBM prepared-effect adapter"
)]
impl NativeOwnedAv1Webm {
    pub(crate) fn encoded_bytes(&self) -> Result<&[u8], TensorError> {
        self.bytes.contiguous_bytes()
    }

    pub(crate) fn dimensions(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    pub(crate) fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }

    pub(crate) fn frame_rate(&self) -> (i32, i32) {
        self.frame_rate
    }

    pub(crate) fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub(crate) const fn has_alpha(&self) -> bool {
        false
    }

    pub(crate) const fn bit_depth(&self) -> NativeVideoBitDepth {
        NativeVideoBitDepth::Ten
    }

    pub(crate) const fn pixel_format(&self) -> NativeVideoPixelFormat {
        NativeVideoPixelFormat::Yuv420p10le
    }

    #[allow(clippy::type_complexity)]
    fn into_parts(self) -> (Tensor, [u8; 32], i32, i32, (i32, i32), usize) {
        (
            self.bytes,
            self.content_sha256,
            self.width,
            self.height,
            self.frame_rate,
            self.frame_count,
        )
    }
}

struct NativeLtxvCodecThreadInner {
    identity: NativeLtxvCodecThreadIdentity,
    #[allow(
        dead_code,
        reason = "read through the node-service proxy that is installed by the native video adapter"
    )]
    node_service_identity: NativeLtxvPreprocessServiceIdentity,
    sender: Mutex<Option<mpsc::SyncSender<NativeLtxvCodecThreadRequest>>>,
    runner: Mutex<Option<JoinHandle<()>>>,
}

#[allow(
    dead_code,
    reason = "consumed by the native LTXVPreprocess adapter and VP9 owned-byte bridge"
)]
pub(crate) struct NativeLtxvCodecThreadService {
    inner: Arc<NativeLtxvCodecThreadInner>,
}

#[derive(Debug, Error)]
pub enum NativeVideoCodecWorkerServicesError {
    #[error("general video codec worker service startup was cancelled")]
    Cancelled,
    #[error("general video codec package declared an unreviewed codec scratch limit")]
    InvalidScratchLimit,
    #[error("general video codec worker services are unsupported for this target")]
    UnsupportedTarget,
    #[error("general video codec worker service limits are invalid")]
    InvalidReviewedLimits,
    #[error("general video codec worker actor is unavailable")]
    ActorUnavailable,
}

#[derive(Debug, Error)]
#[allow(
    dead_code,
    reason = "retained actor-internal decode port is injected by the following Task564 adapter"
)]
pub(crate) enum NativeGeneralVideoDecodeServiceError {
    #[error("native general-video decode request was cancelled")]
    Cancelled,
    #[error("native general-video decode actor is busy")]
    Busy,
    #[error("native general-video decode exhausted its reviewed resources")]
    ResourceExhausted,
    #[error("native general-video decode request is invalid")]
    InvalidRequest,
    #[error("native general-video decode actor is unavailable")]
    Unavailable,
}

#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained actor-internal decode port is injected by the following Task564 adapter"
)]
pub(crate) struct NativeGeneralVideoDecodeService {
    proxy: NativeLtxvCodecRequestProxy,
    limits: NativeVideoDecodeLimits,
    configuration_sha256: String,
}

#[allow(
    dead_code,
    reason = "retained actor-internal decode port is injected by the following Task564 adapter"
)]
impl NativeGeneralVideoDecodeService {
    fn checked(proxy: NativeLtxvCodecRequestProxy, limits: NativeVideoDecodeLimits) -> Self {
        let mut digest = Sha256::new();
        hash_identity_field(
            &mut digest,
            GENERAL_VIDEO_DECODE_SERVICE_IDENTITY_VERSION.as_bytes(),
        );
        hash_identity_field(
            &mut digest,
            proxy.identity().configuration_sha256().as_bytes(),
        );
        for value in limits.configuration_values() {
            hash_identity_field(&mut digest, &value.to_le_bytes());
        }
        Self {
            proxy,
            limits,
            configuration_sha256: format!("{:x}", digest.finalize()),
        }
    }

    pub(crate) fn configuration_sha256(&self) -> &str {
        &self.configuration_sha256
    }

    pub(crate) fn decode(
        &self,
        request: NativeVideoDecodeRequest,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeDecodedVideo, NativeGeneralVideoDecodeServiceError>> {
        if let Err(error) = admit_general_video_decode_request(
            request.encoded_bytes().len(),
            self.limits,
            context.cancellation,
        ) {
            return async move { Err(error) }.boxed();
        }
        self.proxy
            .decode_general_video(request, context)
            .map(|result| result.map_err(map_general_video_decode_service_error))
            .boxed()
    }
}

fn admit_general_video_decode_request(
    encoded_byte_length: usize,
    limits: NativeVideoDecodeLimits,
    cancellation: &CancellationToken,
) -> Result<(), NativeGeneralVideoDecodeServiceError> {
    cancellation
        .check()
        .map_err(|_| NativeGeneralVideoDecodeServiceError::Cancelled)?;
    if encoded_byte_length > limits.maximum_input_bytes() {
        return Err(NativeGeneralVideoDecodeServiceError::ResourceExhausted);
    }
    Ok(())
}

pub struct NativeVideoCodecWorkerServices {
    ltxv_preprocess_service: Arc<dyn NativeLtxvPreprocessService>,
    webm_encode_service: Arc<dyn NativeWebmEncodeService>,
    component_h264_mp4_backing_service: Arc<dyn NativeComponentH264Mp4BackingService>,
    #[allow(
        dead_code,
        reason = "retained actor-internal decode port is injected by the following Task564 adapter"
    )]
    general_video_decode_service: NativeGeneralVideoDecodeService,
    cache_configuration_sha256: String,
    startup_resident_bytes: u64,
    codec_scratch_bytes: u64,
    codec_residency_bytes: u64,
    actor: NativeLtxvCodecThreadService,
}

impl NativeVideoCodecWorkerServices {
    pub fn start(
        closure: CertifiedGeneralVideoCodecDependencyClosure,
        backend: Arc<CpuBackend>,
        startup_cancellation: &CancellationToken,
    ) -> Result<Self, NativeVideoCodecWorkerServicesError> {
        startup_cancellation
            .check()
            .map_err(|_| NativeVideoCodecWorkerServicesError::Cancelled)?;
        if closure.codec_scratch_bytes() != REVIEWED_GENERAL_VIDEO_CODEC_SCRATCH_BYTES {
            return Err(NativeVideoCodecWorkerServicesError::InvalidScratchLimit);
        }
        let startup_resident_bytes = closure.startup_resident_bytes();
        let codec_scratch_bytes = closure.codec_scratch_bytes();
        let codec_residency_bytes = startup_resident_bytes
            .checked_add(codec_scratch_bytes)
            .ok_or(NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
        let limits = reviewed_general_video_codec_service_limits()?;
        let actor = NativeLtxvCodecThreadService::start_general(
            closure,
            backend,
            limits.preprocess,
            limits.decode,
            startup_cancellation,
        )
        .map_err(map_worker_actor_startup_error)?;
        let proxy = actor.proxy();
        let ltxv_preprocess_service: Arc<dyn NativeLtxvPreprocessService> = Arc::new(proxy.clone());
        let webm_encode_service: Arc<dyn NativeWebmEncodeService> = Arc::new(
            NativeWebmCodecRequestService::checked(
                proxy.clone(),
                limits.webm_batch,
                limits.metadata,
            )
            .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?,
        );
        let component_h264_mp4_backing_service: Arc<dyn NativeComponentH264Mp4BackingService> =
            Arc::new(
                NativeComponentH264Mp4CodecRequestService::checked(
                    proxy.clone(),
                    limits.planning,
                    limits.h264_sequence,
                )
                .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?,
            );
        let general_video_decode_service =
            NativeGeneralVideoDecodeService::checked(proxy, limits.decode);
        let cache_configuration_sha256 = worker_service_cache_configuration_identity(
            ltxv_preprocess_service.as_ref(),
            webm_encode_service.as_ref(),
            component_h264_mp4_backing_service.as_ref(),
            &general_video_decode_service,
        );
        startup_cancellation
            .check()
            .map_err(|_| NativeVideoCodecWorkerServicesError::Cancelled)?;
        Ok(Self {
            ltxv_preprocess_service,
            webm_encode_service,
            component_h264_mp4_backing_service,
            general_video_decode_service,
            cache_configuration_sha256,
            startup_resident_bytes,
            codec_scratch_bytes,
            codec_residency_bytes,
            actor,
        })
    }

    pub fn ltxv_preprocess_service(&self) -> Arc<dyn NativeLtxvPreprocessService> {
        self.ltxv_preprocess_service.clone()
    }

    pub fn webm_encode_service(&self) -> Arc<dyn NativeWebmEncodeService> {
        self.webm_encode_service.clone()
    }

    pub fn component_h264_mp4_backing_service(
        &self,
    ) -> Arc<dyn NativeComponentH264Mp4BackingService> {
        self.component_h264_mp4_backing_service.clone()
    }

    #[allow(
        dead_code,
        reason = "retained actor-internal decode port is injected by the following Task564 adapter"
    )]
    pub(crate) fn general_video_decode_service(&self) -> NativeGeneralVideoDecodeService {
        self.general_video_decode_service.clone()
    }

    pub fn startup_resident_bytes(&self) -> u64 {
        self.startup_resident_bytes
    }

    pub fn cache_configuration_sha256(&self) -> &str {
        &self.cache_configuration_sha256
    }

    pub fn codec_scratch_bytes(&self) -> u64 {
        self.codec_scratch_bytes
    }

    pub fn codec_residency_bytes(&self) -> u64 {
        self.codec_residency_bytes
    }

    pub fn shutdown(self) -> Result<(), NativeVideoCodecWorkerServicesError> {
        self.actor
            .shutdown()
            .map_err(|_| NativeVideoCodecWorkerServicesError::ActorUnavailable)
    }
}

struct ReviewedGeneralVideoCodecServiceLimits {
    preprocess: NativeLtxvH264PreprocessLimits,
    webm_batch: NativeVp9WebmBatchLimits,
    metadata: crate::NativeVideoContainerMetadataLimits,
    planning: NativeVideoCodecLimits,
    h264_sequence: NativeH264Mp4SequenceLimits,
    decode: NativeVideoDecodeLimits,
}

fn reviewed_general_video_codec_service_limits()
-> Result<ReviewedGeneralVideoCodecServiceLimits, NativeVideoCodecWorkerServicesError> {
    let encode = NativeLtxvH264EncodeLimits::checked(
        REVIEWED_GENERAL_VIDEO_CODEC_OUTPUT_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_AVIO_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_NATIVE_SESSION_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_PACKET_ITERATIONS,
    )
    .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let demux = NativeLtxvH264DemuxLimits::checked(
        REVIEWED_GENERAL_VIDEO_CODEC_OUTPUT_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_AVIO_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_NATIVE_SESSION_BYTES,
        32,
    )
    .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let decode = NativeLtxvH264DecodeLimits::checked(
        REVIEWED_GENERAL_VIDEO_CODEC_PACKET_ITERATIONS,
        REVIEWED_GENERAL_VIDEO_CODEC_PACKET_ITERATIONS,
        16_384,
        16_384,
        REVIEWED_GENERAL_VIDEO_CODEC_PIXELS,
        1024 * 1024 * 1024,
        REVIEWED_GENERAL_VIDEO_CODEC_NATIVE_SESSION_BYTES,
    )
    .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let preprocess = NativeLtxvH264PreprocessLimits::checked(
        u64::try_from(REVIEWED_GENERAL_VIDEO_CODEC_FRAMES)
            .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?,
        usize::try_from(REVIEWED_GENERAL_VIDEO_CODEC_PIXELS)
            .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?,
        encode,
        demux,
        decode,
    )
    .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let webm_session = NativeVp9WebmEncodeLimits::checked(
        REVIEWED_GENERAL_VIDEO_CODEC_OUTPUT_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_AVIO_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_NATIVE_SESSION_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_PACKET_ITERATIONS,
    )
    .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let webm_batch = NativeVp9WebmBatchLimits::checked(
        webm_session,
        REVIEWED_GENERAL_VIDEO_CODEC_FRAMES,
        REVIEWED_GENERAL_VIDEO_CODEC_PIXELS,
    )
    .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let metadata =
        crate::NativeVideoContainerMetadataLimits::checked(256, 1024, 16 * 1024, 1024 * 1024)
            .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let planning = NativeVideoCodecLimits::checked(
        u64::try_from(REVIEWED_COMPONENT_H264_MP4_FRAMES)
            .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?,
        REVIEWED_GENERAL_VIDEO_CODEC_PIXELS,
        10_000_000,
        u64::try_from(REVIEWED_GENERAL_VIDEO_CODEC_OUTPUT_BYTES)
            .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?,
    )
    .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let h264_sequence = NativeH264Mp4SequenceLimits::checked(
        REVIEWED_GENERAL_VIDEO_CODEC_OUTPUT_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_AVIO_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_NATIVE_SESSION_BYTES,
        REVIEWED_GENERAL_VIDEO_CODEC_PACKET_ITERATIONS,
        REVIEWED_COMPONENT_H264_MP4_FRAMES,
        REVIEWED_GENERAL_VIDEO_CODEC_PIXELS,
    )
    .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    let decode = NativeVideoDecodeLimits::reviewed();
    decode
        .checked_workspace_peak_bytes(REVIEWED_GENERAL_VIDEO_CODEC_SCRATCH_BYTES)
        .map_err(|_| NativeVideoCodecWorkerServicesError::InvalidReviewedLimits)?;
    Ok(ReviewedGeneralVideoCodecServiceLimits {
        preprocess,
        webm_batch,
        metadata,
        planning,
        h264_sequence,
        decode,
    })
}

fn map_worker_actor_startup_error(
    error: NativeLtxvCodecThreadError,
) -> NativeVideoCodecWorkerServicesError {
    match error {
        NativeLtxvCodecThreadError::Cancelled => NativeVideoCodecWorkerServicesError::Cancelled,
        NativeLtxvCodecThreadError::Load(error)
            if matches!(*error, NativeVideoCodecLoadError::UnsupportedTarget) =>
        {
            NativeVideoCodecWorkerServicesError::UnsupportedTarget
        }
        NativeLtxvCodecThreadError::Binding(error)
            if matches!(*error, NativeVideoCodecBindingError::UnsupportedTarget) =>
        {
            NativeVideoCodecWorkerServicesError::UnsupportedTarget
        }
        NativeLtxvCodecThreadError::Admission(error)
            if matches!(
                *error,
                NativeVideoCodecLtxvAdmissionError::UnsupportedTarget
            ) =>
        {
            NativeVideoCodecWorkerServicesError::UnsupportedTarget
        }
        NativeLtxvCodecThreadError::SuiteAdmission(error)
            if matches!(
                *error,
                NativeVideoCodecSuiteAdmissionError::UnsupportedTarget
            ) =>
        {
            NativeVideoCodecWorkerServicesError::UnsupportedTarget
        }
        _ => NativeVideoCodecWorkerServicesError::ActorUnavailable,
    }
}

#[allow(
    dead_code,
    reason = "retained actor-internal decode port is injected by the following Task564 adapter"
)]
fn map_general_video_decode_service_error(
    error: NativeLtxvCodecThreadError,
) -> NativeGeneralVideoDecodeServiceError {
    match error {
        NativeLtxvCodecThreadError::Cancelled => NativeGeneralVideoDecodeServiceError::Cancelled,
        NativeLtxvCodecThreadError::Busy => NativeGeneralVideoDecodeServiceError::Busy,
        NativeLtxvCodecThreadError::ResourceExhausted => {
            NativeGeneralVideoDecodeServiceError::ResourceExhausted
        }
        NativeLtxvCodecThreadError::Decode(error)
            if matches!(
                error.as_ref(),
                NativeVideoCodecDecodeError::ResourceExhausted { .. }
                    | NativeVideoCodecDecodeError::NativeAllocation { .. }
                    | NativeVideoCodecDecodeError::PacketIterationLimit
                    | NativeVideoCodecDecodeError::ReceiveIterationLimit
            ) =>
        {
            NativeGeneralVideoDecodeServiceError::ResourceExhausted
        }
        NativeLtxvCodecThreadError::Decode(error)
            if matches!(error.as_ref(), NativeVideoCodecDecodeError::Cancelled) =>
        {
            NativeGeneralVideoDecodeServiceError::Cancelled
        }
        NativeLtxvCodecThreadError::Decode(_) => {
            NativeGeneralVideoDecodeServiceError::InvalidRequest
        }
        NativeLtxvCodecThreadError::ThreadStopped
        | NativeLtxvCodecThreadError::ThreadPanicked
        | NativeLtxvCodecThreadError::StatePoisoned => {
            NativeGeneralVideoDecodeServiceError::Unavailable
        }
        _ => NativeGeneralVideoDecodeServiceError::InvalidRequest,
    }
}

fn worker_service_cache_configuration_identity(
    ltxv: &dyn NativeLtxvPreprocessService,
    webm: &dyn NativeWebmEncodeService,
    h264: &dyn NativeComponentH264Mp4BackingService,
    decode: &NativeGeneralVideoDecodeService,
) -> String {
    worker_service_cache_configuration_identity_from_parts(
        ltxv.identity().configuration_sha256(),
        webm.identity().configuration_sha256(),
        h264.identity().configuration_sha256(),
        decode.configuration_sha256(),
    )
}

fn worker_service_cache_configuration_identity_from_parts(
    ltxv_configuration_sha256: &str,
    webm_configuration_sha256: &str,
    h264_configuration_sha256: &str,
    decode_configuration_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    hash_identity_field(
        &mut digest,
        b"zed.sim.general-video-codec-worker-services.v1",
    );
    hash_identity_field(&mut digest, ltxv_configuration_sha256.as_bytes());
    hash_identity_field(&mut digest, webm_configuration_sha256.as_bytes());
    hash_identity_field(&mut digest, h264_configuration_sha256.as_bytes());
    hash_identity_field(&mut digest, decode_configuration_sha256.as_bytes());
    format!("{:x}", digest.finalize())
}

#[allow(
    dead_code,
    reason = "consumed by the native LTXVPreprocess adapter and VP9 owned-byte bridge"
)]
#[derive(Clone)]
pub(crate) struct NativeLtxvCodecRequestProxy {
    inner: Arc<NativeLtxvCodecThreadInner>,
}

#[derive(Clone, Debug)]
#[allow(
    dead_code,
    reason = "constructed by the following SaveWEBM node adapter"
)]
pub(crate) struct NativeWebmCodecRequestService {
    proxy: NativeLtxvCodecRequestProxy,
    batch_limits: NativeVp9WebmBatchLimits,
    metadata_limits: crate::NativeVideoContainerMetadataLimits,
    identity: NativeWebmEncodeServiceIdentity,
}

#[allow(
    dead_code,
    reason = "constructed by the following SaveWEBM node adapter"
)]
impl NativeWebmCodecRequestService {
    pub(crate) fn checked(
        proxy: NativeLtxvCodecRequestProxy,
        batch_limits: NativeVp9WebmBatchLimits,
        metadata_limits: crate::NativeVideoContainerMetadataLimits,
    ) -> Result<Self, NativeWebmEncodeServiceError> {
        let mut digest = Sha256::new();
        digest.update(WEBM_NODE_SERVICE_IDENTITY_VERSION.as_bytes());
        digest.update([0]);
        digest.update(proxy.identity().configuration_sha256().as_bytes());
        for value in batch_limits_configuration_u64(batch_limits)? {
            digest.update(value.to_be_bytes());
        }
        for value in metadata_limits_configuration_u64(metadata_limits)? {
            digest.update(value.to_be_bytes());
        }
        let configuration_sha256 = format!("{:x}", digest.finalize());
        let identity = NativeWebmEncodeServiceIdentity::checked(configuration_sha256)
            .map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?;
        Ok(Self {
            proxy,
            batch_limits,
            metadata_limits,
            identity,
        })
    }
}

#[allow(
    dead_code,
    reason = "constructed by the following encoded VIDEO backing adapter"
)]
#[derive(Clone, Debug)]
pub(crate) struct NativeComponentH264Mp4CodecRequestService {
    proxy: NativeLtxvCodecRequestProxy,
    planning_limits: NativeVideoCodecLimits,
    sequence_limits: NativeH264Mp4SequenceLimits,
    identity: NativeComponentH264Mp4BackingServiceIdentity,
}

#[allow(
    dead_code,
    reason = "constructed by the following encoded VIDEO backing adapter"
)]
impl NativeComponentH264Mp4CodecRequestService {
    pub(crate) fn checked(
        proxy: NativeLtxvCodecRequestProxy,
        planning_limits: NativeVideoCodecLimits,
        sequence_limits: NativeH264Mp4SequenceLimits,
    ) -> Result<Self, NativeComponentH264Mp4BackingServiceError> {
        let sequence_values = h264_mp4_sequence_limits_configuration_u64(sequence_limits)?;
        if planning_limits.max_encoded_bytes() > sequence_values[0]
            || planning_limits.max_frames() > sequence_values[4]
            || planning_limits.max_pixels_per_frame() > sequence_values[5]
        {
            return Err(NativeComponentH264Mp4BackingServiceError::InvalidRequest);
        }
        let mut digest = Sha256::new();
        digest.update(COMPONENT_H264_MP4_BACKING_SERVICE_IDENTITY_VERSION.as_bytes());
        digest.update([0]);
        digest.update(proxy.identity().configuration_sha256().as_bytes());
        for value in [
            planning_limits.max_frames(),
            planning_limits.max_pixels_per_frame(),
            planning_limits.max_audio_samples(),
            planning_limits.max_encoded_bytes(),
        ] {
            digest.update(value.to_be_bytes());
        }
        for value in sequence_values {
            digest.update(value.to_be_bytes());
        }
        let configuration_sha256 = format!("{:x}", digest.finalize());
        let identity = NativeComponentH264Mp4BackingServiceIdentity::checked(configuration_sha256)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidRequest)?;
        Ok(Self {
            proxy,
            planning_limits,
            sequence_limits,
            identity,
        })
    }
}

impl fmt::Debug for NativeLtxvCodecRequestProxy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeLtxvCodecRequestProxy")
            .field("identity", &self.inner.node_service_identity)
            .finish_non_exhaustive()
    }
}

#[allow(
    dead_code,
    reason = "consumed by the native LTXVPreprocess adapter and VP9 owned-byte bridge"
)]
impl NativeLtxvCodecThreadService {
    pub(crate) fn start(
        closure: CertifiedVideoCodecDependencyClosure,
        backend: Arc<CpuBackend>,
        limits: NativeLtxvH264PreprocessLimits,
        startup_cancellation: &CancellationToken,
    ) -> Result<Self, NativeLtxvCodecThreadError> {
        startup_cancellation
            .check()
            .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
        let base_configuration_sha256 = video_codec_thread_base_identity(&closure, limits);
        let startup_cancellation = startup_cancellation.clone();
        start_ltxv_codec_thread(move || {
            startup_cancellation
                .check()
                .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
            let load = load_certified_video_codec_closure(closure, &startup_cancellation)
                .map_err(|error| NativeLtxvCodecThreadError::Load(Box::new(error)))?;
            let binding = bind_certified_video_codec_abi(load, &startup_cancellation)
                .map_err(|error| NativeLtxvCodecThreadError::Binding(Box::new(error)))?;
            let codec = binding
                .admit_ltxv_h264(&startup_cancellation)
                .map_err(|error| NativeLtxvCodecThreadError::Admission(Box::new(error)))?;
            let codec = codec
                .admit_video_suite(&startup_cancellation)
                .map_err(|error| NativeLtxvCodecThreadError::SuiteAdmission(Box::new(error)))?;
            startup_cancellation
                .check()
                .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
            let identity = NativeLtxvCodecThreadIdentity {
                target: codec.target().to_owned(),
                primary_catalog_sha256: codec.primary_catalog_sha256().to_owned(),
                configuration_sha256: finalize_video_codec_thread_identity(
                    &base_configuration_sha256,
                    codec.runtime_versions(),
                ),
                runtime_versions: codec.runtime_versions(),
            };
            let processor = move |request: NativeLtxvCodecThreadInvocation| {
                process_video_codec_request(&codec, &backend, limits, None, request)
            };
            Ok((identity, processor))
        })
    }

    pub(crate) fn start_general(
        closure: CertifiedGeneralVideoCodecDependencyClosure,
        backend: Arc<CpuBackend>,
        limits: NativeLtxvH264PreprocessLimits,
        decode_limits: NativeVideoDecodeLimits,
        startup_cancellation: &CancellationToken,
    ) -> Result<Self, NativeLtxvCodecThreadError> {
        startup_cancellation
            .check()
            .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
        let base_configuration_sha256 =
            general_video_codec_thread_base_identity(&closure, limits, decode_limits);
        let startup_cancellation = startup_cancellation.clone();
        start_ltxv_codec_thread(move || {
            startup_cancellation
                .check()
                .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
            let load = load_certified_general_video_codec_closure(closure, &startup_cancellation)
                .map_err(|error| NativeLtxvCodecThreadError::Load(Box::new(error)))?;
            let binding = bind_certified_general_video_codec_abi(load, &startup_cancellation)
                .map_err(|error| NativeLtxvCodecThreadError::Binding(Box::new(error)))?;
            let general_versions = binding.runtime_versions();
            let codec = binding
                .admit_ltxv_h264(&startup_cancellation)
                .map_err(|error| NativeLtxvCodecThreadError::Admission(Box::new(error)))?;
            let codec = codec
                .admit_video_suite(&startup_cancellation)
                .map_err(|error| NativeLtxvCodecThreadError::SuiteAdmission(Box::new(error)))?;
            startup_cancellation
                .check()
                .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
            let identity = NativeLtxvCodecThreadIdentity {
                target: codec.target().to_owned(),
                primary_catalog_sha256: codec.primary_catalog_sha256().to_owned(),
                configuration_sha256: finalize_general_video_codec_thread_identity(
                    &base_configuration_sha256,
                    general_versions,
                ),
                runtime_versions: general_versions.historical(),
            };
            let processor = move |request: NativeLtxvCodecThreadInvocation| {
                process_video_codec_request(&codec, &backend, limits, Some(decode_limits), request)
            };
            Ok((identity, processor))
        })
    }

    pub(crate) fn proxy(&self) -> NativeLtxvCodecRequestProxy {
        NativeLtxvCodecRequestProxy {
            inner: self.inner.clone(),
        }
    }

    pub(crate) fn shutdown(self) -> Result<(), NativeLtxvCodecThreadError> {
        self.inner.close()
    }
}

impl Drop for NativeLtxvCodecThreadService {
    fn drop(&mut self) {
        if let Err(error) = self.inner.close() {
            eprintln!("native video codec service cleanup failed: {error}");
        }
    }
}

#[allow(
    dead_code,
    reason = "consumed by the native LTXVPreprocess adapter and VP9 owned-byte bridge"
)]
impl NativeLtxvCodecRequestProxy {
    pub(crate) fn identity(&self) -> &NativeLtxvCodecThreadIdentity {
        &self.inner.identity
    }

    pub(crate) fn preprocess_image(
        &self,
        image: &ImageTensor,
        compression: u8,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<ImageTensor, NativeLtxvCodecThreadError>> {
        let result = self.submit(
            NativeVideoCodecThreadOperation::Preprocess {
                image: image.clone(),
                compression,
            },
            context,
        );
        async move {
            match result.await? {
                NativeVideoCodecThreadOutput::Image(image) => Ok(image),
                NativeVideoCodecThreadOutput::Vp9Webm(_)
                | NativeVideoCodecThreadOutput::Av1Webm(_)
                | NativeVideoCodecThreadOutput::H264Mp4(_)
                | NativeVideoCodecThreadOutput::DecodedVideo(_) => {
                    Err(NativeLtxvCodecThreadError::StatePoisoned)
                }
            }
        }
        .boxed()
    }

    pub(crate) fn decode_general_video(
        &self,
        request: NativeVideoDecodeRequest,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeDecodedVideo, NativeLtxvCodecThreadError>> {
        let result = self.submit(
            NativeVideoCodecThreadOperation::DecodeGeneralVideo { request },
            context,
        );
        async move {
            match result.await? {
                NativeVideoCodecThreadOutput::DecodedVideo(video) => Ok(video),
                NativeVideoCodecThreadOutput::Image(_)
                | NativeVideoCodecThreadOutput::Vp9Webm(_)
                | NativeVideoCodecThreadOutput::Av1Webm(_)
                | NativeVideoCodecThreadOutput::H264Mp4(_) => {
                    Err(NativeLtxvCodecThreadError::StatePoisoned)
                }
            }
        }
        .boxed()
    }

    pub(crate) fn encode_vp9_webm_batch(
        &self,
        images: &ImageTensor,
        frame_rate: (u64, u64),
        crf: NativeVideoCrf,
        limits: NativeVp9WebmBatchLimits,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeOwnedVp9Webm, NativeLtxvCodecThreadError>> {
        self.encode_vp9_webm_batch_with_metadata(
            images,
            frame_rate,
            crf,
            limits,
            NativeVideoContainerMetadata::empty(),
            context,
        )
    }

    pub(crate) fn encode_vp9_webm_batch_with_metadata(
        &self,
        images: &ImageTensor,
        frame_rate: (u64, u64),
        crf: NativeVideoCrf,
        limits: NativeVp9WebmBatchLimits,
        metadata: NativeVideoContainerMetadata,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeOwnedVp9Webm, NativeLtxvCodecThreadError>> {
        let result = self.submit(
            NativeVideoCodecThreadOperation::EncodeVp9Webm {
                images: images.clone(),
                frame_rate,
                crf,
                limits,
                metadata,
            },
            context,
        );
        async move {
            match result.await? {
                NativeVideoCodecThreadOutput::Vp9Webm(encoded) => Ok(encoded),
                NativeVideoCodecThreadOutput::Image(_)
                | NativeVideoCodecThreadOutput::Av1Webm(_)
                | NativeVideoCodecThreadOutput::H264Mp4(_)
                | NativeVideoCodecThreadOutput::DecodedVideo(_) => {
                    Err(NativeLtxvCodecThreadError::StatePoisoned)
                }
            }
        }
        .boxed()
    }

    pub(crate) fn encode_av1_webm_batch_with_metadata(
        &self,
        images: &ImageTensor,
        frame_rate: (u64, u64),
        crf: NativeVideoCrf,
        limits: NativeVp9WebmBatchLimits,
        metadata: NativeVideoContainerMetadata,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeOwnedAv1Webm, NativeLtxvCodecThreadError>> {
        let result = self.submit(
            NativeVideoCodecThreadOperation::EncodeAv1Webm {
                images: images.clone(),
                frame_rate,
                crf,
                limits,
                metadata,
            },
            context,
        );
        async move {
            match result.await? {
                NativeVideoCodecThreadOutput::Av1Webm(encoded) => Ok(encoded),
                NativeVideoCodecThreadOutput::Image(_)
                | NativeVideoCodecThreadOutput::Vp9Webm(_)
                | NativeVideoCodecThreadOutput::H264Mp4(_)
                | NativeVideoCodecThreadOutput::DecodedVideo(_) => {
                    Err(NativeLtxvCodecThreadError::StatePoisoned)
                }
            }
        }
        .boxed()
    }

    #[allow(
        dead_code,
        reason = "consumed by the following encoded-backing video service"
    )]
    pub(crate) fn encode_h264_mp4_batch(
        &self,
        images: &ImageTensor,
        frame_rate: (u64, u64),
        limits: NativeH264Mp4SequenceLimits,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeOwnedH264Mp4, NativeLtxvCodecThreadError>> {
        self.encode_h264_mp4_batch_with_bit_depth(
            images,
            frame_rate,
            limits,
            NativeVideoBitDepth::Eight,
            context,
        )
    }

    #[allow(
        dead_code,
        reason = "consumed by the following ten-bit encoded-backing video service"
    )]
    pub(crate) fn encode_h264_mp4_10bit_batch(
        &self,
        images: &ImageTensor,
        frame_rate: (u64, u64),
        limits: NativeH264Mp4SequenceLimits,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeOwnedH264Mp4, NativeLtxvCodecThreadError>> {
        self.encode_h264_mp4_batch_with_bit_depth(
            images,
            frame_rate,
            limits,
            NativeVideoBitDepth::Ten,
            context,
        )
    }

    fn encode_h264_mp4_batch_with_bit_depth(
        &self,
        images: &ImageTensor,
        frame_rate: (u64, u64),
        limits: NativeH264Mp4SequenceLimits,
        bit_depth: NativeVideoBitDepth,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeOwnedH264Mp4, NativeLtxvCodecThreadError>> {
        let result = self.submit(
            NativeVideoCodecThreadOperation::EncodeH264Mp4 {
                images: images.clone(),
                frame_rate,
                limits,
                bit_depth,
            },
            context,
        );
        async move {
            match result.await? {
                NativeVideoCodecThreadOutput::H264Mp4(encoded) => Ok(encoded),
                NativeVideoCodecThreadOutput::Image(_)
                | NativeVideoCodecThreadOutput::Vp9Webm(_)
                | NativeVideoCodecThreadOutput::Av1Webm(_)
                | NativeVideoCodecThreadOutput::DecodedVideo(_) => {
                    Err(NativeLtxvCodecThreadError::StatePoisoned)
                }
            }
        }
        .boxed()
    }

    fn submit(
        &self,
        operation: NativeVideoCodecThreadOperation,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeVideoCodecThreadOutput, NativeLtxvCodecThreadError>> {
        if context.cancellation.check().is_err() {
            return async { Err(NativeLtxvCodecThreadError::Cancelled) }.boxed();
        }
        let sender = match self.inner.sender.try_lock() {
            Ok(sender) => match sender.as_ref() {
                Some(sender) => sender.clone(),
                None => {
                    return async { Err(NativeLtxvCodecThreadError::ThreadStopped) }.boxed();
                }
            },
            Err(TryLockError::WouldBlock) => {
                return async { Err(NativeLtxvCodecThreadError::Busy) }.boxed();
            }
            Err(TryLockError::Poisoned(_)) => {
                return async { Err(NativeLtxvCodecThreadError::StatePoisoned) }.boxed();
            }
        };
        let cancellation = context.cancellation.clone();
        let (response, receiver) = oneshot::channel();
        let request = NativeLtxvCodecThreadRequest {
            invocation: NativeLtxvCodecThreadInvocation {
                operation,
                stream: context.stream,
                scratch: context.scratch.clone(),
                cancellation: cancellation.clone(),
            },
            response,
        };
        match sender.try_send(request) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => {
                return async { Err(NativeLtxvCodecThreadError::Busy) }.boxed();
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                return async { Err(NativeLtxvCodecThreadError::ThreadStopped) }.boxed();
            }
        }
        async move {
            let result = receiver
                .await
                .map_err(|_| NativeLtxvCodecThreadError::ThreadStopped)?;
            cancellation
                .check()
                .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
            result
        }
        .boxed()
    }
}

impl NativeLtxvPreprocessService for NativeLtxvCodecRequestProxy {
    fn identity(&self) -> &NativeLtxvPreprocessServiceIdentity {
        &self.inner.node_service_identity
    }

    fn preprocess_image(
        &self,
        image: &ImageTensor,
        compression: u8,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<ImageTensor, NativeLtxvPreprocessServiceError>> {
        NativeLtxvCodecRequestProxy::preprocess_image(self, image, compression, context)
            .map(|result| result.map_err(map_ltxv_node_service_error))
            .boxed()
    }
}

impl NativeWebmEncodeService for NativeWebmCodecRequestService {
    fn identity(&self) -> &NativeWebmEncodeServiceIdentity {
        &self.identity
    }

    fn encode_webm(
        &self,
        request: NativeWebmEncodeRequest,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeEncodedWebm, NativeWebmEncodeServiceError>> {
        let (images, codec, frame_rate, crf, metadata) = request.into_parts();
        let dimensions = match images.dimensions() {
            Ok((frame_count, height, width, _)) => (frame_count, width, height),
            Err(_) => {
                return async { Err(NativeWebmEncodeServiceError::InvalidRequest) }.boxed();
            }
        };
        let metadata = match NativeVideoContainerMetadata::checked(metadata, self.metadata_limits) {
            Ok(metadata) => metadata,
            Err(error) => {
                return async move { Err(map_webm_metadata_service_error(error)) }.boxed();
            }
        };
        let stream = context.stream;
        let result = match codec {
            NativeVideoCodec::Vp9 => self
                .proxy
                .encode_vp9_webm_batch_with_metadata(
                    &images,
                    frame_rate,
                    crf,
                    self.batch_limits,
                    metadata,
                    context,
                )
                .map(|result| result.map(WebmActorOutput::Vp9))
                .boxed(),
            NativeVideoCodec::Av1 => self
                .proxy
                .encode_av1_webm_batch_with_metadata(
                    &images,
                    frame_rate,
                    crf,
                    self.batch_limits,
                    metadata,
                    context,
                )
                .map(|result| result.map(WebmActorOutput::Av1))
                .boxed(),
            NativeVideoCodec::H264 => {
                return async { Err(NativeWebmEncodeServiceError::InvalidRequest) }.boxed();
            }
        };
        async move {
            let output = result.await.map_err(map_webm_node_service_error)?;
            let encoded = match output {
                WebmActorOutput::Vp9(output) => {
                    let (bytes, digest, width, height, output_rate, frame_count, has_alpha) =
                        output.into_parts();
                    checked_webm_service_result(
                        bytes,
                        digest,
                        NativeVideoCodec::Vp9,
                        width,
                        height,
                        output_rate,
                        frame_count,
                        if has_alpha {
                            NativeVideoPixelFormat::Yuva420p
                        } else {
                            NativeVideoPixelFormat::Yuv420p
                        },
                        NativeVideoBitDepth::Eight,
                        has_alpha,
                        dimensions,
                        frame_rate,
                        stream,
                    )?
                }
                WebmActorOutput::Av1(output) => {
                    let (bytes, digest, width, height, output_rate, frame_count) =
                        output.into_parts();
                    checked_webm_service_result(
                        bytes,
                        digest,
                        NativeVideoCodec::Av1,
                        width,
                        height,
                        output_rate,
                        frame_count,
                        NativeVideoPixelFormat::Yuv420p10le,
                        NativeVideoBitDepth::Ten,
                        false,
                        dimensions,
                        frame_rate,
                        stream,
                    )?
                }
            };
            Ok(encoded)
        }
        .boxed()
    }
}

impl NativeComponentH264Mp4BackingService for NativeComponentH264Mp4CodecRequestService {
    fn identity(&self) -> &NativeComponentH264Mp4BackingServiceIdentity {
        &self.identity
    }

    fn encode_backing(
        &self,
        request: NativeComponentH264Mp4BackingRequest,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeVideoPayload, NativeComponentH264Mp4BackingServiceError>>
    {
        if context.cancellation.check().is_err() {
            return async { Err(NativeComponentH264Mp4BackingServiceError::Cancelled) }.boxed();
        }
        let video = request.into_video();
        let components = match video.components() {
            Some(components) => components,
            None => {
                return async { Err(NativeComponentH264Mp4BackingServiceError::InvalidRequest) }
                    .boxed();
            }
        };
        if components.frames().descriptor().stream() != context.stream {
            return async { Err(NativeComponentH264Mp4BackingServiceError::InvalidRequest) }
                .boxed();
        }
        let plan = match plan_native_video_encode(
            video.as_ref(),
            NativeVideoEncodeOptions::ComponentMp4 {
                metadata: NativeVideoMetadataPolicy::Exclude,
            },
            self.planning_limits,
            &context.cancellation,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                return async move { Err(map_h264_mp4_backing_plan_error(error)) }.boxed();
            }
        };
        let supported_depth = matches!(
            (
                plan.source_bit_depth(),
                plan.output_bit_depth(),
                plan.pixel_format(),
            ),
            (
                NativeVideoBitDepth::Eight,
                NativeVideoBitDepth::Eight,
                NativeVideoPixelFormat::Yuv420p,
            ) | (
                NativeVideoBitDepth::Ten,
                NativeVideoBitDepth::Ten,
                NativeVideoPixelFormat::Yuv420p10le,
            )
        );
        let supported_plan = plan.container() == NativeVideoContainer::Mp4
            && plan.codec() == NativeVideoCodec::H264
            && supported_depth
            && plan.alpha() == NativeVideoAlphaPolicy::Discard
            && plan.audio().is_none()
            && plan.metadata() == NativeVideoMetadataPolicy::Exclude
            && plan.crf().is_none()
            && plan.preset().is_none();
        if !supported_plan {
            return async { Err(NativeComponentH264Mp4BackingServiceError::InvalidRequest) }
                .boxed();
        }
        let images = match ImageTensor::from_tensor(components.frames().clone()) {
            Ok(images) => images,
            Err(_) => {
                return async { Err(NativeComponentH264Mp4BackingServiceError::InvalidRequest) }
                    .boxed();
            }
        };
        let expected_dimensions = plan.dimensions();
        let expected_frame_rate = plan.encode_frame_rate();
        let expected_frame_count = plan.frame_count();
        let expected_bit_depth = plan.output_bit_depth();
        let source_video = video;
        let stream = context.stream;
        let cancellation = context.cancellation.clone();
        let result = match expected_bit_depth {
            NativeVideoBitDepth::Eight => self.proxy.encode_h264_mp4_batch(
                &images,
                expected_frame_rate,
                self.sequence_limits,
                context,
            ),
            NativeVideoBitDepth::Ten => self.proxy.encode_h264_mp4_10bit_batch(
                &images,
                expected_frame_rate,
                self.sequence_limits,
                context,
            ),
        };
        async move {
            let output = result.await.map_err(map_h264_mp4_backing_service_error)?;
            cancellation
                .check()
                .map_err(|_| NativeComponentH264Mp4BackingServiceError::Cancelled)?;
            let backing = checked_component_h264_mp4_backing_result(
                output,
                source_video.as_ref(),
                expected_dimensions,
                expected_frame_rate,
                expected_frame_count,
                expected_bit_depth,
                stream,
            )?;
            cancellation
                .check()
                .map_err(|_| NativeComponentH264Mp4BackingServiceError::Cancelled)?;
            Ok(backing)
        }
        .boxed()
    }
}

#[allow(
    dead_code,
    reason = "called by the encoded VIDEO backing adapter and its conformance tests"
)]
fn checked_component_h264_mp4_backing_result(
    output: NativeOwnedH264Mp4,
    source_video: &NativeVideoPayload,
    expected_dimensions: (u64, u64),
    expected_frame_rate: (u64, u64),
    expected_frame_count: u64,
    expected_bit_depth: NativeVideoBitDepth,
    expected_stream: StreamId,
) -> Result<NativeVideoPayload, NativeComponentH264Mp4BackingServiceError> {
    let expected_pixel_format = match expected_bit_depth {
        NativeVideoBitDepth::Eight => NativeVideoPixelFormat::Yuv420p,
        NativeVideoBitDepth::Ten => NativeVideoPixelFormat::Yuv420p10le,
    };
    if output.bit_depth() != expected_bit_depth || output.pixel_format() != expected_pixel_format {
        return Err(NativeComponentH264Mp4BackingServiceError::InvalidProjection);
    }
    let (bytes, content_sha256, width, height, frame_rate, frame_count) = output.into_parts();
    let dimensions = (
        u64::try_from(width)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidProjection)?,
        u64::try_from(height)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidProjection)?,
    );
    let frame_rate = (
        u64::try_from(frame_rate.0)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidProjection)?,
        u64::try_from(frame_rate.1)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidProjection)?,
    );
    let frame_count = u64::try_from(frame_count)
        .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidProjection)?;
    if dimensions != expected_dimensions
        || frame_rate != expected_frame_rate
        || frame_count != expected_frame_count
        || bytes.descriptor().stream() != expected_stream
    {
        return Err(NativeComponentH264Mp4BackingServiceError::InvalidProjection);
    }
    NativeVideoPayload::checked_h264_mp4_from_component(
        source_video,
        bytes,
        content_sha256,
        dimensions,
        frame_rate,
        frame_count,
    )
    .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidProjection)
}

#[allow(dead_code, reason = "consumed by the following SaveWEBM node adapter")]
enum WebmActorOutput {
    Vp9(NativeOwnedVp9Webm),
    Av1(NativeOwnedAv1Webm),
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code, reason = "consumed by the following SaveWEBM node adapter")]
fn checked_webm_service_result(
    bytes: Tensor,
    content_sha256: [u8; 32],
    codec: NativeVideoCodec,
    width: i32,
    height: i32,
    frame_rate: (i32, i32),
    frame_count: usize,
    pixel_format: NativeVideoPixelFormat,
    bit_depth: NativeVideoBitDepth,
    has_alpha: bool,
    expected: (u64, u64, u64),
    expected_frame_rate: (u64, u64),
    expected_stream: StreamId,
) -> Result<NativeEncodedWebm, NativeWebmEncodeServiceError> {
    let width =
        u64::try_from(width).map_err(|_| NativeWebmEncodeServiceError::InvalidProjection)?;
    let height =
        u64::try_from(height).map_err(|_| NativeWebmEncodeServiceError::InvalidProjection)?;
    let frame_rate = (
        u64::try_from(frame_rate.0).map_err(|_| NativeWebmEncodeServiceError::InvalidProjection)?,
        u64::try_from(frame_rate.1).map_err(|_| NativeWebmEncodeServiceError::InvalidProjection)?,
    );
    let frame_count =
        u64::try_from(frame_count).map_err(|_| NativeWebmEncodeServiceError::InvalidProjection)?;
    if (frame_count, width, height) != expected
        || frame_rate != expected_frame_rate
        || bytes.descriptor().stream() != expected_stream
    {
        return Err(NativeWebmEncodeServiceError::InvalidProjection);
    }
    NativeEncodedWebm::checked(
        bytes,
        content_sha256,
        codec,
        (width, height),
        frame_rate,
        frame_count,
        pixel_format,
        bit_depth,
        has_alpha,
    )
}

#[allow(dead_code, reason = "consumed by the following SaveWEBM node adapter")]
fn batch_limits_configuration_u64(
    limits: NativeVp9WebmBatchLimits,
) -> Result<[u64; 6], NativeWebmEncodeServiceError> {
    let values = limits.configuration_values();
    Ok([
        u64::try_from(values.0).map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?,
        u64::try_from(values.1).map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?,
        values.2,
        u64::try_from(values.3).map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?,
        u64::try_from(values.4).map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?,
        values.5,
    ])
}

#[allow(dead_code, reason = "consumed by the following SaveWEBM node adapter")]
fn metadata_limits_configuration_u64(
    limits: crate::NativeVideoContainerMetadataLimits,
) -> Result<[u64; 4], NativeWebmEncodeServiceError> {
    let values = limits.configuration_values();
    Ok([
        u64::try_from(values.0).map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?,
        u64::try_from(values.1).map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?,
        u64::try_from(values.2).map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?,
        u64::try_from(values.3).map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?,
    ])
}

#[allow(
    dead_code,
    reason = "consumed by the following encoded VIDEO backing adapter"
)]
fn h264_mp4_sequence_limits_configuration_u64(
    limits: NativeH264Mp4SequenceLimits,
) -> Result<[u64; 6], NativeComponentH264Mp4BackingServiceError> {
    let values = limits.configuration_values();
    Ok([
        u64::try_from(values.0)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidRequest)?,
        u64::try_from(values.1)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidRequest)?,
        values.2,
        u64::try_from(values.3)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidRequest)?,
        u64::try_from(values.4)
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidRequest)?,
        values.5,
    ])
}

impl NativeLtxvCodecThreadInner {
    fn close(&self) -> Result<(), NativeLtxvCodecThreadError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?
            .take();
        drop(sender);
        let runner = self
            .runner
            .lock()
            .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?
            .take();
        if let Some(runner) = runner {
            runner
                .join()
                .map_err(|_| NativeLtxvCodecThreadError::ThreadPanicked)?;
        }
        Ok(())
    }
}

impl Drop for NativeLtxvCodecThreadInner {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("native video codec thread cleanup failed: {error}");
        }
    }
}

fn start_ltxv_codec_thread<Factory, Processor>(
    factory: Factory,
) -> Result<NativeLtxvCodecThreadService, NativeLtxvCodecThreadError>
where
    Factory: FnOnce() -> Result<(NativeLtxvCodecThreadIdentity, Processor), NativeLtxvCodecThreadError>
        + Send
        + 'static,
    Processor: FnMut(
            NativeLtxvCodecThreadInvocation,
        ) -> Result<NativeVideoCodecThreadOutput, NativeLtxvCodecThreadError>
        + 'static,
{
    let (sender, receiver) = mpsc::sync_channel(1);
    let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
    let runner = thread::Builder::new()
        .name(VIDEO_CODEC_THREAD_NAME.to_owned())
        .spawn(move || match factory() {
            Ok((identity, mut processor)) => {
                if ready_sender.send(Ok(identity)).is_err() {
                    eprintln!("native video codec thread readiness receiver was dropped");
                    return;
                }
                while let Ok(request) = receiver.recv() {
                    let NativeLtxvCodecThreadRequest {
                        invocation,
                        response,
                    } = request;
                    let result = processor(invocation);
                    if response.send(result).is_err() {
                        eprintln!("native video codec request receiver was dropped");
                    }
                }
            }
            Err(error) => {
                if ready_sender.send(Err(error)).is_err() {
                    eprintln!("native video codec thread startup error receiver was dropped");
                }
            }
        })
        .map_err(NativeLtxvCodecThreadError::ThreadSpawn)?;
    let identity = match ready_receiver.recv() {
        Ok(Ok(identity)) => identity,
        Ok(Err(error)) => {
            if runner.join().is_err() {
                eprintln!("native video codec thread panicked after reporting startup failure");
            }
            return Err(error);
        }
        Err(_) => {
            return match runner.join() {
                Ok(()) => Err(NativeLtxvCodecThreadError::ThreadStopped),
                Err(_) => Err(NativeLtxvCodecThreadError::ThreadPanicked),
            };
        }
    };
    Ok(NativeLtxvCodecThreadService {
        inner: Arc::new(NativeLtxvCodecThreadInner {
            node_service_identity: NativeLtxvPreprocessServiceIdentity::checked(
                identity.configuration_sha256().to_owned(),
            )
            .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?,
            identity,
            sender: Mutex::new(Some(sender)),
            runner: Mutex::new(Some(runner)),
        }),
    })
}

#[allow(
    dead_code,
    reason = "called by the LTXV node-service adapter and its conformance tests"
)]
fn map_ltxv_node_service_error(
    error: NativeLtxvCodecThreadError,
) -> NativeLtxvPreprocessServiceError {
    match error {
        NativeLtxvCodecThreadError::Cancelled => NativeLtxvPreprocessServiceError::Cancelled,
        NativeLtxvCodecThreadError::Busy => NativeLtxvPreprocessServiceError::Busy,
        NativeLtxvCodecThreadError::InvalidScratch(_) => {
            NativeLtxvPreprocessServiceError::InvalidRequest
        }
        NativeLtxvCodecThreadError::ResourceExhausted => {
            NativeLtxvPreprocessServiceError::ResourceExhausted
        }
        NativeLtxvCodecThreadError::ThreadStopped
        | NativeLtxvCodecThreadError::ThreadPanicked
        | NativeLtxvCodecThreadError::StatePoisoned => {
            NativeLtxvPreprocessServiceError::Unavailable
        }
        error @ (NativeLtxvCodecThreadError::ThreadSpawn(_)
        | NativeLtxvCodecThreadError::Load(_)
        | NativeLtxvCodecThreadError::Binding(_)
        | NativeLtxvCodecThreadError::Admission(_)
        | NativeLtxvCodecThreadError::SuiteAdmission(_)
        | NativeLtxvCodecThreadError::Preprocess(_)
        | NativeLtxvCodecThreadError::Vp9Encode(_)
        | NativeLtxvCodecThreadError::Av1Encode(_)
        | NativeLtxvCodecThreadError::H264Encode(_)
        | NativeLtxvCodecThreadError::Decode(_)
        | NativeLtxvCodecThreadError::EncodedOutput(_)) => {
            NativeLtxvPreprocessServiceError::Execution(error.to_string())
        }
    }
}

#[allow(dead_code, reason = "consumed by the following SaveWEBM node adapter")]
fn map_webm_metadata_service_error(
    error: crate::NativeVideoContainerMetadataError,
) -> NativeWebmEncodeServiceError {
    match error {
        crate::NativeVideoContainerMetadataError::LimitExceeded
        | crate::NativeVideoContainerMetadataError::AllocationFailed => {
            NativeWebmEncodeServiceError::ResourceExhausted
        }
        crate::NativeVideoContainerMetadataError::InvalidLimits
        | crate::NativeVideoContainerMetadataError::EmptyKey
        | crate::NativeVideoContainerMetadataError::EmbeddedNul => {
            NativeWebmEncodeServiceError::InvalidRequest
        }
    }
}

#[allow(dead_code, reason = "consumed by the following SaveWEBM node adapter")]
fn map_webm_node_service_error(error: NativeLtxvCodecThreadError) -> NativeWebmEncodeServiceError {
    match error {
        NativeLtxvCodecThreadError::Cancelled => NativeWebmEncodeServiceError::Cancelled,
        NativeLtxvCodecThreadError::Busy => NativeWebmEncodeServiceError::Busy,
        NativeLtxvCodecThreadError::InvalidScratch(_) => {
            NativeWebmEncodeServiceError::InvalidRequest
        }
        NativeLtxvCodecThreadError::ResourceExhausted => {
            NativeWebmEncodeServiceError::ResourceExhausted
        }
        NativeLtxvCodecThreadError::ThreadStopped
        | NativeLtxvCodecThreadError::ThreadPanicked
        | NativeLtxvCodecThreadError::StatePoisoned => NativeWebmEncodeServiceError::Unavailable,
        error @ (NativeLtxvCodecThreadError::ThreadSpawn(_)
        | NativeLtxvCodecThreadError::Load(_)
        | NativeLtxvCodecThreadError::Binding(_)
        | NativeLtxvCodecThreadError::Admission(_)
        | NativeLtxvCodecThreadError::SuiteAdmission(_)
        | NativeLtxvCodecThreadError::Preprocess(_)
        | NativeLtxvCodecThreadError::Vp9Encode(_)
        | NativeLtxvCodecThreadError::Av1Encode(_)
        | NativeLtxvCodecThreadError::H264Encode(_)
        | NativeLtxvCodecThreadError::Decode(_)
        | NativeLtxvCodecThreadError::EncodedOutput(_)) => {
            NativeWebmEncodeServiceError::Execution(error.to_string())
        }
    }
}

#[allow(
    dead_code,
    reason = "consumed by the following encoded VIDEO backing adapter"
)]
fn map_h264_mp4_backing_plan_error(
    error: NativeVideoCodecPlanError,
) -> NativeComponentH264Mp4BackingServiceError {
    match error {
        NativeVideoCodecPlanError::Cancelled => {
            NativeComponentH264Mp4BackingServiceError::Cancelled
        }
        NativeVideoCodecPlanError::LimitExceeded | NativeVideoCodecPlanError::Overflow => {
            NativeComponentH264Mp4BackingServiceError::ResourceExhausted
        }
        NativeVideoCodecPlanError::InvalidVideo
        | NativeVideoCodecPlanError::InvalidOptions
        | NativeVideoCodecPlanError::InvalidLimits => {
            NativeComponentH264Mp4BackingServiceError::InvalidRequest
        }
    }
}

#[allow(
    dead_code,
    reason = "consumed by the following encoded VIDEO backing adapter"
)]
fn map_h264_mp4_backing_service_error(
    error: NativeLtxvCodecThreadError,
) -> NativeComponentH264Mp4BackingServiceError {
    match error {
        NativeLtxvCodecThreadError::Cancelled => {
            NativeComponentH264Mp4BackingServiceError::Cancelled
        }
        NativeLtxvCodecThreadError::Busy => NativeComponentH264Mp4BackingServiceError::Busy,
        NativeLtxvCodecThreadError::InvalidScratch(_) => {
            NativeComponentH264Mp4BackingServiceError::InvalidRequest
        }
        NativeLtxvCodecThreadError::ResourceExhausted => {
            NativeComponentH264Mp4BackingServiceError::ResourceExhausted
        }
        NativeLtxvCodecThreadError::ThreadStopped
        | NativeLtxvCodecThreadError::ThreadPanicked
        | NativeLtxvCodecThreadError::StatePoisoned => {
            NativeComponentH264Mp4BackingServiceError::Unavailable
        }
        error @ (NativeLtxvCodecThreadError::ThreadSpawn(_)
        | NativeLtxvCodecThreadError::Load(_)
        | NativeLtxvCodecThreadError::Binding(_)
        | NativeLtxvCodecThreadError::Admission(_)
        | NativeLtxvCodecThreadError::SuiteAdmission(_)
        | NativeLtxvCodecThreadError::Preprocess(_)
        | NativeLtxvCodecThreadError::Vp9Encode(_)
        | NativeLtxvCodecThreadError::Av1Encode(_)
        | NativeLtxvCodecThreadError::H264Encode(_)
        | NativeLtxvCodecThreadError::Decode(_)
        | NativeLtxvCodecThreadError::EncodedOutput(_)) => {
            NativeComponentH264Mp4BackingServiceError::Execution(error.to_string())
        }
    }
}

fn process_video_codec_request(
    codec: &NativeVideoCodecSuite,
    backend: &CpuBackend,
    limits: NativeLtxvH264PreprocessLimits,
    decode_limits: Option<NativeVideoDecodeLimits>,
    request: NativeLtxvCodecThreadInvocation,
) -> Result<NativeVideoCodecThreadOutput, NativeLtxvCodecThreadError> {
    request
        .cancellation
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    backend
        .validate_scratch_reservation(&request.scratch)
        .map_err(|error| NativeLtxvCodecThreadError::InvalidScratch(Box::new(error)))?;
    let context = ExecutionContext {
        stream: request.stream,
        scratch: request.scratch,
        rng_phase: None,
        cancellation: &request.cancellation,
    };
    let output = match request.operation {
        NativeVideoCodecThreadOperation::Preprocess { image, compression } => {
            NativeVideoCodecThreadOutput::Image(
                codec
                    .preprocess_image(&image, compression, limits, backend, &context)
                    .map_err(map_ltxv_thread_preprocess_error)?,
            )
        }
        NativeVideoCodecThreadOperation::EncodeVp9Webm {
            images,
            frame_rate,
            crf,
            limits,
            metadata,
        } => NativeVideoCodecThreadOutput::Vp9Webm(process_vp9_webm_request(
            codec, backend, &context, &images, frame_rate, crf, limits, &metadata,
        )?),
        NativeVideoCodecThreadOperation::EncodeAv1Webm {
            images,
            frame_rate,
            crf,
            limits,
            metadata,
        } => NativeVideoCodecThreadOutput::Av1Webm(process_av1_webm_request(
            codec, backend, &context, &images, frame_rate, crf, limits, &metadata,
        )?),
        NativeVideoCodecThreadOperation::EncodeH264Mp4 {
            images,
            frame_rate,
            limits,
            bit_depth,
        } => NativeVideoCodecThreadOutput::H264Mp4(process_h264_mp4_request(
            codec, backend, &context, &images, frame_rate, limits, bit_depth,
        )?),
        NativeVideoCodecThreadOperation::DecodeGeneralVideo { request } => {
            let limits = decode_limits.ok_or(NativeLtxvCodecThreadError::StatePoisoned)?;
            NativeVideoCodecThreadOutput::DecodedVideo(
                codec
                    .decode_general_video(
                        request.encoded_bytes(),
                        request.window(),
                        limits,
                        backend,
                        &context,
                    )
                    .map_err(|error| NativeLtxvCodecThreadError::Decode(Box::new(error)))?,
            )
        }
    };
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn process_vp9_webm_request(
    codec: &NativeVideoCodecSuite,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    images: &ImageTensor,
    frame_rate: (u64, u64),
    crf: NativeVideoCrf,
    limits: NativeVp9WebmBatchLimits,
    metadata: &NativeVideoContainerMetadata,
) -> Result<NativeOwnedVp9Webm, NativeLtxvCodecThreadError> {
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    let encoded = codec
        .encode_vp9_webm_batch_with_metadata(
            images, frame_rate, crf, limits, metadata, backend, context,
        )
        .map_err(map_vp9_thread_encode_error)?;
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    let (width, height) = encoded.dimensions();
    let frame_rate = encoded.frame_rate();
    let frame_count = encoded.frame_count();
    let has_alpha = encoded.has_alpha();
    let encoded_bytes = encoded
        .encoded_bytes()
        .map_err(map_vp9_thread_encode_error)?;
    let output = materialize_owned_vp9_webm(
        backend,
        context,
        encoded_bytes,
        width,
        height,
        frame_rate,
        frame_count,
        has_alpha,
    )?;
    drop(encoded);
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn process_av1_webm_request(
    codec: &NativeVideoCodecSuite,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    images: &ImageTensor,
    frame_rate: (u64, u64),
    crf: NativeVideoCrf,
    limits: NativeVp9WebmBatchLimits,
    metadata: &NativeVideoContainerMetadata,
) -> Result<NativeOwnedAv1Webm, NativeLtxvCodecThreadError> {
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    let encoded = codec
        .encode_av1_webm_batch_with_metadata(
            images, frame_rate, crf, limits, metadata, backend, context,
        )
        .map_err(map_av1_thread_encode_error)?;
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    let (width, height) = encoded.dimensions();
    let frame_rate = encoded.frame_rate();
    let frame_count = encoded.frame_count();
    let encoded_bytes = encoded
        .encoded_bytes()
        .map_err(map_av1_thread_encode_error)?;
    let output = materialize_owned_av1_webm(
        backend,
        context,
        encoded_bytes,
        width,
        height,
        frame_rate,
        frame_count,
    )?;
    drop(encoded);
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    Ok(output)
}

fn process_h264_mp4_request(
    codec: &NativeVideoCodecSuite,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    images: &ImageTensor,
    frame_rate: (u64, u64),
    limits: NativeH264Mp4SequenceLimits,
    bit_depth: NativeVideoBitDepth,
) -> Result<NativeOwnedH264Mp4, NativeLtxvCodecThreadError> {
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    let encoded = match bit_depth {
        NativeVideoBitDepth::Eight => {
            codec.encode_h264_mp4_batch(images, frame_rate, limits, backend, context)
        }
        NativeVideoBitDepth::Ten => {
            codec.encode_h264_mp4_10bit_batch(images, frame_rate, limits, backend, context)
        }
    }
    .map_err(map_h264_thread_encode_error)?;
    materialize_h264_mp4_result(backend, context, encoded)
}

fn materialize_h264_mp4_result(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    encoded: crate::NativeH264Mp4<'_>,
) -> Result<NativeOwnedH264Mp4, NativeLtxvCodecThreadError> {
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    let (width, height) = encoded.dimensions();
    let frame_rate = encoded.frame_rate();
    let frame_count = encoded.frame_count();
    let bit_depth = encoded.bit_depth();
    let encoded_bytes = encoded
        .encoded_bytes()
        .map_err(map_h264_thread_encode_error)?;
    let output = materialize_owned_h264_mp4(
        backend,
        context,
        encoded_bytes,
        width,
        height,
        frame_rate,
        frame_count,
        bit_depth,
    )?;
    drop(encoded);
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn materialize_owned_vp9_webm(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    encoded_bytes: &[u8],
    width: i32,
    height: i32,
    frame_rate: (i32, i32),
    frame_count: usize,
    has_alpha: bool,
) -> Result<NativeOwnedVp9Webm, NativeLtxvCodecThreadError> {
    let (bytes, content_sha256) = materialize_owned_encoded_bytes(backend, context, encoded_bytes)?;
    Ok(NativeOwnedVp9Webm {
        bytes,
        content_sha256,
        width,
        height,
        frame_rate,
        frame_count,
        has_alpha,
    })
}

#[allow(clippy::too_many_arguments)]
fn materialize_owned_av1_webm(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    encoded_bytes: &[u8],
    width: i32,
    height: i32,
    frame_rate: (i32, i32),
    frame_count: usize,
) -> Result<NativeOwnedAv1Webm, NativeLtxvCodecThreadError> {
    let (bytes, content_sha256) = materialize_owned_encoded_bytes(backend, context, encoded_bytes)?;
    Ok(NativeOwnedAv1Webm {
        bytes,
        content_sha256,
        width,
        height,
        frame_rate,
        frame_count,
    })
}

#[allow(clippy::too_many_arguments)]
fn materialize_owned_h264_mp4(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    encoded_bytes: &[u8],
    width: i32,
    height: i32,
    frame_rate: (i32, i32),
    frame_count: usize,
    bit_depth: NativeVideoBitDepth,
) -> Result<NativeOwnedH264Mp4, NativeLtxvCodecThreadError> {
    let (bytes, content_sha256) = materialize_owned_encoded_bytes(backend, context, encoded_bytes)?;
    Ok(NativeOwnedH264Mp4 {
        bytes,
        content_sha256,
        width,
        height,
        frame_rate,
        frame_count,
        bit_depth,
    })
}

fn materialize_owned_encoded_bytes(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    encoded_bytes: &[u8],
) -> Result<(Tensor, [u8; 32]), NativeLtxvCodecThreadError> {
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    let byte_length = u64::try_from(encoded_bytes.len())
        .map_err(|_| NativeLtxvCodecThreadError::ResourceExhausted)?;
    let descriptor =
        TensorDescriptor::contiguous(vec![byte_length], DType::U8, DeviceId::CPU, context.stream)
            .map_err(map_encoded_output_tensor_error)?;
    let (bytes, _) = backend
        .upload_bytes(descriptor, encoded_bytes, context)
        .map_err(map_encoded_output_tensor_error)?;
    context
        .check()
        .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
    Ok((bytes, Sha256::digest(encoded_bytes).into()))
}

fn map_vp9_thread_encode_error(
    error: NativeVideoCodecVp9EncodeError,
) -> NativeLtxvCodecThreadError {
    match error {
        NativeVideoCodecVp9EncodeError::Cancelled => NativeLtxvCodecThreadError::Cancelled,
        NativeVideoCodecVp9EncodeError::ResourceExhausted { .. } => {
            NativeLtxvCodecThreadError::ResourceExhausted
        }
        error => NativeLtxvCodecThreadError::Vp9Encode(Box::new(error)),
    }
}

fn map_av1_thread_encode_error(
    error: NativeVideoCodecAv1EncodeError,
) -> NativeLtxvCodecThreadError {
    match error {
        NativeVideoCodecAv1EncodeError::Cancelled => NativeLtxvCodecThreadError::Cancelled,
        NativeVideoCodecAv1EncodeError::ResourceExhausted { .. } => {
            NativeLtxvCodecThreadError::ResourceExhausted
        }
        error => NativeLtxvCodecThreadError::Av1Encode(Box::new(error)),
    }
}

fn map_h264_thread_encode_error(
    error: NativeVideoCodecH264EncodeError,
) -> NativeLtxvCodecThreadError {
    match error {
        NativeVideoCodecH264EncodeError::Cancelled => NativeLtxvCodecThreadError::Cancelled,
        NativeVideoCodecH264EncodeError::NativeAllocation { .. }
        | NativeVideoCodecH264EncodeError::ResourceExhausted { .. } => {
            NativeLtxvCodecThreadError::ResourceExhausted
        }
        error => NativeLtxvCodecThreadError::H264Encode(Box::new(error)),
    }
}

fn map_encoded_output_tensor_error(error: TensorError) -> NativeLtxvCodecThreadError {
    match error {
        TensorError::Cancelled => NativeLtxvCodecThreadError::Cancelled,
        TensorError::AllocationFailed { .. }
        | TensorError::ResourceLimitExceeded { .. }
        | TensorError::WorkspaceAuthorizationExceeded { .. } => {
            NativeLtxvCodecThreadError::ResourceExhausted
        }
        error => NativeLtxvCodecThreadError::EncodedOutput(Box::new(error)),
    }
}

fn map_ltxv_thread_preprocess_error(
    error: NativeVideoCodecLtxvPreprocessError,
) -> NativeLtxvCodecThreadError {
    match error {
        NativeVideoCodecLtxvPreprocessError::Cancelled => NativeLtxvCodecThreadError::Cancelled,
        NativeVideoCodecLtxvPreprocessError::ResourceExhausted => {
            NativeLtxvCodecThreadError::ResourceExhausted
        }
        error => NativeLtxvCodecThreadError::Preprocess(Box::new(error)),
    }
}

fn video_codec_thread_base_identity(
    closure: &CertifiedVideoCodecDependencyClosure,
    limits: NativeLtxvH264PreprocessLimits,
) -> String {
    let mut digest = Sha256::new();
    hash_identity_field(&mut digest, VIDEO_CODEC_THREAD_IDENTITY_VERSION.as_bytes());
    hash_identity_field(&mut digest, closure.target().as_bytes());
    hash_identity_field(&mut digest, closure.primary_catalog_sha256().as_bytes());
    hash_identity_field(&mut digest, closure.source_archive_sha256().as_bytes());
    for (identity, library) in closure.primary_libraries() {
        hash_identity_field(&mut digest, identity.as_bytes());
        hash_identity_field(&mut digest, library.filename().as_bytes());
        hash_identity_field(&mut digest, library.digest_sha256().as_bytes());
        hash_identity_field(&mut digest, &library.abi_major().to_le_bytes());
    }
    for (identity, dependency) in closure.dependencies() {
        hash_identity_field(&mut digest, identity.as_bytes());
        hash_identity_field(&mut digest, dependency.filename().as_bytes());
        hash_identity_field(&mut digest, dependency.digest_sha256().as_bytes());
        hash_identity_field(&mut digest, dependency.abi_version().as_bytes());
        hash_identity_field(&mut digest, dependency.certificate_sponsor().as_bytes());
    }
    for edge in closure.edges() {
        hash_identity_field(&mut digest, edge.consumer().as_bytes());
        hash_identity_field(&mut digest, edge.dependency().as_bytes());
    }
    for (encoder, provider) in closure.encoder_providers() {
        hash_identity_field(&mut digest, encoder.as_bytes());
        hash_identity_field(&mut digest, provider.as_bytes());
    }
    for identity in closure.dependency_first_order() {
        hash_identity_field(&mut digest, identity.as_bytes());
    }
    hash_identity_field(
        &mut digest,
        &closure.retained_dependency_bytes().to_le_bytes(),
    );
    for value in limits.configuration_values() {
        hash_identity_field(&mut digest, &value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn general_video_codec_thread_base_identity(
    closure: &CertifiedGeneralVideoCodecDependencyClosure,
    limits: NativeLtxvH264PreprocessLimits,
    decode_limits: NativeVideoDecodeLimits,
) -> String {
    let mut digest = Sha256::new();
    hash_identity_field(&mut digest, VIDEO_CODEC_THREAD_IDENTITY_VERSION.as_bytes());
    hash_identity_field(&mut digest, b"general-video-codec-package-v1");
    hash_identity_field(&mut digest, closure.semantic_identity().as_bytes());
    for (identity, library) in closure.libraries() {
        hash_identity_field(&mut digest, identity.as_bytes());
        hash_identity_field(&mut digest, library.filename().as_bytes());
        hash_identity_field(&mut digest, library.digest_sha256().as_bytes());
        hash_identity_field(&mut digest, &library.abi_major().to_le_bytes());
        hash_identity_field(&mut digest, library.soname().as_bytes());
    }
    if let Ok(dependency_first_order) = closure.dependency_first_order() {
        for identity in dependency_first_order {
            hash_identity_field(&mut digest, identity.as_bytes());
        }
    } else {
        hash_identity_field(&mut digest, b"invalid-dependency-order");
    }
    hash_identity_field(&mut digest, &closure.retained_image_bytes().to_le_bytes());
    hash_identity_field(&mut digest, &closure.startup_resident_bytes().to_le_bytes());
    hash_identity_field(&mut digest, &closure.codec_scratch_bytes().to_le_bytes());
    for value in limits.configuration_values() {
        hash_identity_field(&mut digest, &value.to_le_bytes());
    }
    for value in decode_limits.configuration_values() {
        hash_identity_field(&mut digest, &value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn finalize_video_codec_thread_identity(
    base_configuration_sha256: &str,
    versions: NativeVideoCodecRuntimeVersions,
) -> String {
    let mut digest = Sha256::new();
    hash_identity_field(&mut digest, base_configuration_sha256.as_bytes());
    for version in [
        versions.avcodec(),
        versions.avformat(),
        versions.avutil(),
        versions.swresample(),
        versions.swscale(),
    ] {
        hash_identity_field(&mut digest, &version.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn finalize_general_video_codec_thread_identity(
    base_configuration_sha256: &str,
    versions: NativeGeneralVideoCodecRuntimeVersions,
) -> String {
    let mut digest = Sha256::new();
    hash_identity_field(&mut digest, base_configuration_sha256.as_bytes());
    let historical = versions.historical();
    for version in [
        historical.avcodec(),
        historical.avformat(),
        historical.avutil(),
        historical.swresample(),
        historical.swscale(),
        versions.avfilter(),
    ] {
        hash_identity_field(&mut digest, &version.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn hash_identity_field(digest: &mut Sha256, value: &[u8]) {
    digest.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
    digest.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NativeVp9WebmEncodeLimits;
    use futures::executor::block_on;
    use sim_media::NativeVideoPayload;
    use sim_tensor::{BackendWorkspaceAuthority, DType, DeviceId, Layout, TensorDescriptor};
    use std::{
        collections::BTreeMap,
        rc::Rc,
        sync::{
            Condvar,
            atomic::{AtomicUsize, Ordering},
        },
    };

    fn test_identity(seed: &str) -> NativeLtxvCodecThreadIdentity {
        let versions = NativeVideoCodecRuntimeVersions::from_components(
            0x3d1364, 0x3d0764, 0x3b2764, 0x050364, 0x080364,
        );
        NativeLtxvCodecThreadIdentity {
            target: "x86_64-unknown-linux-gnu".to_owned(),
            primary_catalog_sha256: "11".repeat(32),
            configuration_sha256: finalize_video_codec_thread_identity(seed, versions),
            runtime_versions: versions,
        }
    }

    fn test_image_and_context(
        cancellation: &CancellationToken,
    ) -> Result<(Arc<CpuBackend>, ImageTensor, ScratchReservation), TensorError> {
        let (backend, authority) = BackendWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: scratch.clone(),
            rng_phase: None,
            cancellation,
        };
        let image = ImageTensor::from_f32(&backend, &context, 1, 1, 1, 3, &[0.0, 0.5, 1.0])?;
        Ok((backend, image, scratch))
    }

    fn request_context<'a>(
        scratch: ScratchReservation,
        cancellation: &'a CancellationToken,
    ) -> ExecutionContext<'a> {
        ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch,
            rng_phase: None,
            cancellation,
        }
    }

    #[test]
    fn component_h264_worker_admits_beeble_frame_limit_and_rejects_overflow_before_dispatch()
    -> Result<(), Box<dyn std::error::Error>> {
        let reviewed = reviewed_general_video_codec_service_limits()?;
        assert_eq!(reviewed.preprocess.configuration_values()[0], 64);
        assert_eq!(reviewed.webm_batch.configuration_values().4, 64);
        assert_eq!(
            reviewed.h264_sequence.configuration_values(),
            (
                512 * 1024 * 1024,
                64 * 1024,
                1024 * 1024 * 1024,
                1_000_000,
                240,
                268_435_456
            )
        );
        let cancellation = CancellationToken::default();
        let (backend, _, scratch) = test_image_and_context(&cancellation)?;
        let context = request_context(scratch.clone(), &cancellation);
        let dispatched = Arc::new(AtomicUsize::new(0));
        let actor = start_ltxv_codec_thread({
            let backend = backend.clone();
            let dispatched = dispatched.clone();
            move || {
                Ok((
                    test_identity("beeble-frame-boundary"),
                    move |request: NativeLtxvCodecThreadInvocation| {
                        let NativeVideoCodecThreadOperation::EncodeH264Mp4 {
                            images,
                            frame_rate,
                            limits,
                            bit_depth,
                        } = request.operation
                        else {
                            return Err(NativeLtxvCodecThreadError::StatePoisoned);
                        };
                        let (frame_count, height, width, _) = images
                            .dimensions()
                            .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?;
                        assert_eq!((width, height), (2, 2));
                        assert_eq!(frame_rate, (2997, 100));
                        assert_eq!(limits, reviewed.h264_sequence);
                        assert!(matches!(frame_count, 64 | 65 | 240));
                        dispatched.fetch_add(1, Ordering::SeqCst);
                        let context = ExecutionContext {
                            stream: request.stream,
                            scratch: request.scratch,
                            rng_phase: None,
                            cancellation: &request.cancellation,
                        };
                        materialize_owned_h264_mp4(
                            &backend,
                            &context,
                            b"recorded H264 dispatch",
                            2,
                            2,
                            (2997, 100),
                            usize::try_from(frame_count)
                                .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?,
                            bit_depth,
                        )
                        .map(NativeVideoCodecThreadOutput::H264Mp4)
                    },
                ))
            }
        })?;
        let service = NativeComponentH264Mp4CodecRequestService::checked(
            actor.proxy(),
            reviewed.planning,
            reviewed.h264_sequence,
        )?;
        for frame_count in [64_usize, 65, 240, 241, 240] {
            for bit_depth in [NativeVideoBitDepth::Eight, NativeVideoBitDepth::Ten] {
                let images = ImageTensor::from_f32(
                    &backend,
                    &context,
                    u64::try_from(frame_count)?,
                    2,
                    2,
                    3,
                    &vec![0.5; frame_count * 12],
                )?;
                let source = Arc::new(NativeVideoPayload::checked(
                    images.tensor().clone(),
                    30_000,
                    1001,
                    bit_depth,
                    None,
                    None,
                    BTreeMap::new(),
                )?);
                let before = dispatched.load(Ordering::SeqCst);
                let result = block_on(service.encode_backing(
                    NativeComponentH264Mp4BackingRequest::checked(source.clone())?,
                    &context,
                ));
                if frame_count == 241 {
                    assert!(matches!(
                        result,
                        Err(NativeComponentH264Mp4BackingServiceError::ResourceExhausted)
                    ));
                    assert_eq!(dispatched.load(Ordering::SeqCst), before);
                } else {
                    let backing = result?;
                    let encoded = backing.encoded().ok_or("missing encoded backing")?;
                    assert_eq!(encoded.frame_count(), u64::try_from(frame_count)?);
                    assert_eq!(encoded.frame_rate(), (2997, 100));
                    assert_eq!(
                        encoded.source_video_sha256(),
                        source.semantic_digest_sha256()
                    );
                    assert_eq!(encoded.bit_depth(), bit_depth);
                    assert_eq!(dispatched.load(Ordering::SeqCst), before + 1);
                }
                assert_eq!(scratch.in_use_bytes(), 0);
            }
        }
        actor.shutdown()?;
        Ok(())
    }

    #[test]
    fn general_video_codec_package_bootstrap_cache_identity_binds_all_four_port_limits() {
        let base = worker_service_cache_configuration_identity_from_parts(
            &"11".repeat(32),
            &"22".repeat(32),
            &"33".repeat(32),
            &"55".repeat(32),
        );
        assert_eq!(base.len(), 64);
        for changed in [
            worker_service_cache_configuration_identity_from_parts(
                &"44".repeat(32),
                &"22".repeat(32),
                &"33".repeat(32),
                &"55".repeat(32),
            ),
            worker_service_cache_configuration_identity_from_parts(
                &"11".repeat(32),
                &"44".repeat(32),
                &"33".repeat(32),
                &"55".repeat(32),
            ),
            worker_service_cache_configuration_identity_from_parts(
                &"11".repeat(32),
                &"22".repeat(32),
                &"44".repeat(32),
                &"55".repeat(32),
            ),
            worker_service_cache_configuration_identity_from_parts(
                &"11".repeat(32),
                &"22".repeat(32),
                &"33".repeat(32),
                &"44".repeat(32),
            ),
        ] {
            assert_ne!(base, changed);
        }
    }

    #[test]
    fn general_video_demux_decode_admission_preserves_pre_cancellation_and_input_ceiling() {
        let limits = NativeVideoDecodeLimits::reviewed();
        let cancellation = CancellationToken::default();
        assert!(admit_general_video_decode_request(
            limits.maximum_input_bytes(),
            limits,
            &cancellation,
        )
        .is_ok());
        assert!(matches!(
            admit_general_video_decode_request(
                limits.maximum_input_bytes() + 1,
                limits,
                &cancellation,
            ),
            Err(NativeGeneralVideoDecodeServiceError::ResourceExhausted)
        ));
        cancellation.cancel();
        assert!(matches!(
            admit_general_video_decode_request(
                limits.maximum_input_bytes() + 1,
                limits,
                &cancellation,
            ),
            Err(NativeGeneralVideoDecodeServiceError::Cancelled)
        ));
    }

    #[test]
    fn retained_ltxv_codec_thread_is_send_sync_serial_and_thread_affine()
    -> Result<(), Box<dyn std::error::Error>> {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NativeLtxvCodecRequestProxy>();

        let cancellation = CancellationToken::default();
        let (_backend, image, scratch) = test_image_and_context(&cancellation)?;
        let events = Arc::new(Mutex::new(Vec::new()));
        let drop_events = events.clone();
        let service = start_ltxv_codec_thread({
            let events = events.clone();
            move || {
                let thread_id = thread::current().id();
                events
                    .lock()
                    .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?
                    .push(("start", thread_id));
                let thread_bound = Rc::new(ThreadBoundDrop {
                    events: drop_events,
                });
                Ok((
                    test_identity("serial"),
                    move |request: NativeLtxvCodecThreadInvocation| {
                        let _thread_bound = &thread_bound;
                        events
                            .lock()
                            .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?
                            .push(("request", thread::current().id()));
                        match request.operation {
                            NativeVideoCodecThreadOperation::Preprocess { image, .. } => {
                                Ok(NativeVideoCodecThreadOutput::Image(image))
                            }
                            NativeVideoCodecThreadOperation::EncodeVp9Webm { crf, .. } => {
                                if crf.bits() != 31.5_f64.to_bits() {
                                    return Err(NativeLtxvCodecThreadError::StatePoisoned);
                                }
                                Err(NativeLtxvCodecThreadError::StatePoisoned)
                            }
                            NativeVideoCodecThreadOperation::EncodeAv1Webm { crf, .. } => {
                                if crf.bits() != 31.5_f64.to_bits() {
                                    return Err(NativeLtxvCodecThreadError::StatePoisoned);
                                }
                                Err(NativeLtxvCodecThreadError::StatePoisoned)
                            }
                            NativeVideoCodecThreadOperation::EncodeH264Mp4 { .. } => {
                                Err(NativeLtxvCodecThreadError::StatePoisoned)
                            }
                            NativeVideoCodecThreadOperation::DecodeGeneralVideo { .. } => {
                                Err(NativeLtxvCodecThreadError::StatePoisoned)
                            }
                        }
                    },
                ))
            }
        })?;
        let proxy = service.proxy();
        assert_eq!(proxy.identity().target(), "x86_64-unknown-linux-gnu");
        assert_eq!(
            NativeLtxvPreprocessService::identity(&proxy).configuration_sha256(),
            proxy.identity().configuration_sha256()
        );
        let context = request_context(scratch, &cancellation);
        let first = block_on(proxy.preprocess_image(&image, 0, &context))?;
        let second = block_on(proxy.preprocess_image(&image, 0, &context))?;
        assert_eq!(first.dimensions()?, (1, 1, 1, 3));
        assert_eq!(second.dimensions()?, (1, 1, 1, 3));
        service.shutdown()?;

        let events = events
            .lock()
            .map_err(|_| "thread event mutex was poisoned")?;
        let actor_thread = events.first().ok_or("missing actor start event")?.1;
        assert_ne!(actor_thread, thread::current().id());
        assert_eq!(
            events.iter().filter(|event| event.0 == "request").count(),
            2
        );
        assert!(events.iter().all(|event| event.1 == actor_thread));
        assert_eq!(events.last().map(|event| event.0), Some("drop"));
        Ok(())
    }

    #[test]
    fn retained_ltxv_codec_thread_bounds_queue_cancellation_failure_and_retry()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (_backend, image, scratch) = test_image_and_context(&cancellation)?;
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let started = Arc::new((Mutex::new(false), Condvar::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let service = start_ltxv_codec_thread({
            let gate = gate.clone();
            let started = started.clone();
            let calls = calls.clone();
            move || {
                Ok((
                    test_identity("bounded"),
                    move |request: NativeLtxvCodecThreadInvocation| {
                        calls.fetch_add(1, Ordering::AcqRel);
                        let NativeLtxvCodecThreadInvocation {
                            operation,
                            cancellation,
                            ..
                        } = request;
                        let NativeVideoCodecThreadOperation::Preprocess { image, compression } =
                            operation
                        else {
                            return Err(NativeLtxvCodecThreadError::StatePoisoned);
                        };
                        if compression == 99 {
                            return Err(NativeLtxvCodecThreadError::ResourceExhausted);
                        }
                        if compression == 98 {
                            cancellation.cancel();
                            return Ok(NativeVideoCodecThreadOutput::Image(image));
                        }
                        let (started_lock, started_condition) = &*started;
                        let mut is_started = started_lock
                            .lock()
                            .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?;
                        *is_started = true;
                        started_condition.notify_all();
                        drop(is_started);
                        let (gate_lock, gate_condition) = &*gate;
                        let mut released = gate_lock
                            .lock()
                            .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?;
                        while !*released {
                            released = gate_condition
                                .wait(released)
                                .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?;
                        }
                        cancellation
                            .check()
                            .map_err(|_| NativeLtxvCodecThreadError::Cancelled)?;
                        Ok(NativeVideoCodecThreadOutput::Image(image))
                    },
                ))
            }
        })?;
        let proxy = service.proxy();
        let context = request_context(scratch.clone(), &cancellation);
        let first = proxy.preprocess_image(&image, 0, &context);
        {
            let (started_lock, started_condition) = &*started;
            let mut is_started = started_lock
                .lock()
                .map_err(|_| "started mutex was poisoned")?;
            while !*is_started {
                is_started = started_condition
                    .wait(is_started)
                    .map_err(|_| "started mutex was poisoned")?;
            }
        }
        let second = proxy.preprocess_image(&image, 0, &context);
        assert!(matches!(
            block_on(proxy.preprocess_image(&image, 0, &context)),
            Err(NativeLtxvCodecThreadError::Busy)
        ));
        {
            let (gate_lock, gate_condition) = &*gate;
            *gate_lock.lock().map_err(|_| "gate mutex was poisoned")? = true;
            gate_condition.notify_all();
        }
        block_on(first)?;
        block_on(second)?;

        assert!(matches!(
            block_on(proxy.preprocess_image(&image, 99, &context)),
            Err(NativeLtxvCodecThreadError::ResourceExhausted)
        ));
        block_on(proxy.preprocess_image(&image, 0, &context))?;
        let late_cancellation = CancellationToken::default();
        let late_context = request_context(scratch.clone(), &late_cancellation);
        assert!(matches!(
            block_on(proxy.preprocess_image(&image, 98, &late_context)),
            Err(NativeLtxvCodecThreadError::Cancelled)
        ));
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = request_context(scratch, &cancelled);
        assert!(matches!(
            block_on(proxy.preprocess_image(&image, 0, &cancelled_context)),
            Err(NativeLtxvCodecThreadError::Cancelled)
        ));
        assert_eq!(calls.load(Ordering::Acquire), 5);
        service.shutdown()?;
        assert!(matches!(
            block_on(proxy.preprocess_image(&image, 0, &context)),
            Err(NativeLtxvCodecThreadError::ThreadStopped)
        ));
        Ok(())
    }

    #[test]
    fn retained_ltxv_codec_thread_startup_and_identity_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(matches!(
            start_ltxv_codec_thread::<_, fn(NativeLtxvCodecThreadInvocation) -> _>(|| {
                Err(NativeLtxvCodecThreadError::Cancelled)
            }),
            Err(NativeLtxvCodecThreadError::Cancelled)
        ));
        let versions = test_identity("first").runtime_versions();
        assert_ne!(
            finalize_video_codec_thread_identity("first", versions),
            finalize_video_codec_thread_identity("second", versions)
        );
        Ok(())
    }

    #[test]
    fn retained_ltxv_codec_thread_node_service_maps_typed_failures() {
        for (error, expected) in [
            (
                NativeLtxvCodecThreadError::Cancelled,
                NativeLtxvPreprocessServiceError::Cancelled,
            ),
            (
                NativeLtxvCodecThreadError::Busy,
                NativeLtxvPreprocessServiceError::Busy,
            ),
            (
                NativeLtxvCodecThreadError::ResourceExhausted,
                NativeLtxvPreprocessServiceError::ResourceExhausted,
            ),
            (
                NativeLtxvCodecThreadError::ThreadStopped,
                NativeLtxvPreprocessServiceError::Unavailable,
            ),
            (
                NativeLtxvCodecThreadError::ThreadPanicked,
                NativeLtxvPreprocessServiceError::Unavailable,
            ),
            (
                NativeLtxvCodecThreadError::StatePoisoned,
                NativeLtxvPreprocessServiceError::Unavailable,
            ),
        ] {
            assert_eq!(
                std::mem::discriminant(&map_ltxv_node_service_error(error)),
                std::mem::discriminant(&expected)
            );
        }
        let execution = map_ltxv_node_service_error(NativeLtxvCodecThreadError::ThreadSpawn(
            io::Error::other("synthetic spawn failure"),
        ));
        assert!(matches!(
            execution,
            NativeLtxvPreprocessServiceError::Execution(message)
                if message.contains("synthetic spawn failure")
        ));
    }

    #[test]
    fn retained_video_codec_thread_returns_owned_codec_bytes_and_preserves_ltxv()
    -> Result<(), Box<dyn std::error::Error>> {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NativeLtxvCodecRequestProxy>();
        assert_send_sync::<NativeOwnedVp9Webm>();
        assert_send_sync::<NativeOwnedAv1Webm>();
        assert_send_sync::<NativeOwnedH264Mp4>();

        let cancellation = CancellationToken::default();
        let (backend, _image, scratch) = test_image_and_context(&cancellation)?;
        let image_context = request_context(scratch.clone(), &cancellation);
        let image = ImageTensor::from_f32(&backend, &image_context, 3, 2, 2, 4, &[0.5; 48])?;
        let input_storage_id = image.tensor().storage_id();
        let h264_limits = NativeH264Mp4SequenceLimits::checked(1024, 256, 1024, 32, 4, 1024)?;
        let events = Arc::new(Mutex::new(Vec::new()));
        let h264_output_storage_ids = Arc::new(Mutex::new(Vec::new()));
        let actor_backend = backend;
        let service = start_ltxv_codec_thread({
            let events = events.clone();
            let h264_output_storage_ids = h264_output_storage_ids.clone();
            move || {
                events
                    .lock()
                    .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?
                    .push(("start", thread::current().id()));
                Ok((
                    test_identity("vp9-owned-bridge"),
                    move |request: NativeLtxvCodecThreadInvocation| {
                        events
                            .lock()
                            .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?
                            .push(("request", thread::current().id()));
                        let NativeLtxvCodecThreadInvocation {
                            operation,
                            stream,
                            scratch,
                            cancellation,
                        } = request;
                        let context = ExecutionContext {
                            stream,
                            scratch,
                            rng_phase: None,
                            cancellation: &cancellation,
                        };
                        match operation {
                            NativeVideoCodecThreadOperation::Preprocess { image, .. } => {
                                Ok(NativeVideoCodecThreadOutput::Image(image))
                            }
                            NativeVideoCodecThreadOperation::EncodeVp9Webm {
                                images,
                                metadata,
                                ..
                            } => {
                                if !matches!(images.dimensions(), Ok((_, _, _, 4))) {
                                    return Err(NativeLtxvCodecThreadError::StatePoisoned);
                                }
                                let entries = metadata.entries();
                                if entries.len() != 3
                                    || entries[0].0.as_bytes() != b"prompt"
                                    || entries[0].1.as_bytes() != b"first"
                                    || entries[1].0.as_bytes() != b"workflow"
                                    || entries[2].0.as_bytes() != b"prompt"
                                    || entries[2].1.as_bytes() != b"last"
                                {
                                    return Err(NativeLtxvCodecThreadError::StatePoisoned);
                                }
                                let encoded = materialize_owned_vp9_webm(
                                    &actor_backend,
                                    &context,
                                    b"HPPPT",
                                    2,
                                    2,
                                    (2997, 125),
                                    3,
                                    true,
                                )?;
                                Ok(NativeVideoCodecThreadOutput::Vp9Webm(encoded))
                            }
                            NativeVideoCodecThreadOperation::EncodeAv1Webm {
                                images,
                                metadata,
                                ..
                            } => {
                                if !matches!(images.dimensions(), Ok((_, _, _, 4))) {
                                    return Err(NativeLtxvCodecThreadError::StatePoisoned);
                                }
                                let entries = metadata.entries();
                                if entries.len() != 2
                                    || entries[0].0.as_bytes() != b"prompt"
                                    || entries[0].1.as_bytes() != b"av1"
                                    || entries[1].0.as_bytes() != b"workflow"
                                {
                                    return Err(NativeLtxvCodecThreadError::StatePoisoned);
                                }
                                let encoded = materialize_owned_av1_webm(
                                    &actor_backend,
                                    &context,
                                    b"HA1T",
                                    2,
                                    2,
                                    (2997, 125),
                                    3,
                                )?;
                                Ok(NativeVideoCodecThreadOutput::Av1Webm(encoded))
                            }
                            NativeVideoCodecThreadOperation::EncodeH264Mp4 {
                                images,
                                frame_rate,
                                limits,
                                bit_depth,
                            } => {
                                if !matches!(images.dimensions(), Ok((3, 2, 2, 4)))
                                    || frame_rate != (2997, 100)
                                    || limits != h264_limits
                                    || images.tensor().storage_id() != input_storage_id
                                {
                                    return Err(NativeLtxvCodecThreadError::StatePoisoned);
                                }
                                let encoded_bytes = match bit_depth {
                                    NativeVideoBitDepth::Eight => b"HMP4".as_slice(),
                                    NativeVideoBitDepth::Ten => b"H10P4".as_slice(),
                                };
                                let encoded = materialize_owned_h264_mp4(
                                    &actor_backend,
                                    &context,
                                    encoded_bytes,
                                    2,
                                    2,
                                    (2997, 100),
                                    3,
                                    bit_depth,
                                )?;
                                h264_output_storage_ids
                                    .lock()
                                    .map_err(|_| NativeLtxvCodecThreadError::StatePoisoned)?
                                    .push(encoded.bytes.storage_id());
                                Ok(NativeVideoCodecThreadOutput::H264Mp4(encoded))
                            }
                            NativeVideoCodecThreadOperation::DecodeGeneralVideo { .. } => {
                                Err(NativeLtxvCodecThreadError::StatePoisoned)
                            }
                        }
                    },
                ))
            }
        })?;
        let proxy = service.proxy();
        let context = request_context(scratch.clone(), &cancellation);
        let preprocessed = block_on(proxy.preprocess_image(&image, 0, &context))?;
        assert_eq!(
            preprocessed.tensor().storage_id(),
            image.tensor().storage_id()
        );

        let session_limits = NativeVp9WebmEncodeLimits::checked(1024, 256, 1024, 32)?;
        let batch_limits = NativeVp9WebmBatchLimits::checked(session_limits, 4, 1024)?;
        let metadata = NativeVideoContainerMetadata::checked(
            vec![
                ("prompt".to_owned(), "first".to_owned()),
                ("workflow".to_owned(), "{}".to_owned()),
                ("prompt".to_owned(), "last".to_owned()),
            ],
            crate::NativeVideoContainerMetadataLimits::checked(3, 16, 16, 64)?,
        )?;
        let encoded = block_on(proxy.encode_vp9_webm_batch_with_metadata(
            &image,
            (2997, 125),
            NativeVideoCrf::checked(31.5)?,
            batch_limits,
            metadata,
            &context,
        ))?;
        assert_eq!(encoded.encoded_bytes()?, b"HPPPT");
        assert_eq!(encoded.dimensions(), (2, 2));
        assert_eq!(encoded.frame_rate(), (2997, 125));
        assert_eq!(encoded.frame_count(), 3);
        assert!(encoded.has_alpha());
        assert_eq!(
            encoded.content_sha256(),
            <[u8; 32]>::from(Sha256::digest(b"HPPPT"))
        );
        assert_eq!(scratch.in_use_bytes(), 0);
        drop(encoded);

        let av1_metadata = NativeVideoContainerMetadata::checked(
            vec![
                ("prompt".to_owned(), "av1".to_owned()),
                ("workflow".to_owned(), "{}".to_owned()),
            ],
            crate::NativeVideoContainerMetadataLimits::checked(2, 16, 16, 64)?,
        )?;
        let av1 = block_on(proxy.encode_av1_webm_batch_with_metadata(
            &image,
            (2997, 125),
            NativeVideoCrf::checked(31.5)?,
            batch_limits,
            av1_metadata,
            &context,
        ))?;
        assert_eq!(av1.encoded_bytes()?, b"HA1T");
        assert_eq!(av1.dimensions(), (2, 2));
        assert_eq!(av1.frame_rate(), (2997, 125));
        assert_eq!(av1.frame_count(), 3);
        assert!(!av1.has_alpha());
        assert_eq!(av1.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(av1.pixel_format(), NativeVideoPixelFormat::Yuv420p10le);
        assert_eq!(
            av1.content_sha256(),
            <[u8; 32]>::from(Sha256::digest(b"HA1T"))
        );
        assert_eq!(scratch.in_use_bytes(), 0);
        drop(av1);

        let h264 =
            block_on(proxy.encode_h264_mp4_batch(&image, (2997, 100), h264_limits, &context))?;
        assert_eq!(h264.encoded_bytes()?, b"HMP4");
        assert_eq!(h264.dimensions(), (2, 2));
        assert_eq!(h264.frame_rate(), (2997, 100));
        assert_eq!(h264.frame_count(), 3);
        assert_eq!(h264.codec(), NativeVideoCodec::H264);
        assert_eq!(h264.container(), NativeVideoContainer::Mp4);
        assert_eq!(h264.bit_depth(), NativeVideoBitDepth::Eight);
        assert_eq!(h264.pixel_format(), NativeVideoPixelFormat::Yuv420p);
        assert!(!h264.has_alpha());
        assert_eq!(
            h264.content_sha256(),
            <[u8; 32]>::from(Sha256::digest(b"HMP4"))
        );
        assert_eq!(scratch.in_use_bytes(), 0);
        drop(h264);

        let h264_10bit = block_on(proxy.encode_h264_mp4_10bit_batch(
            &image,
            (2997, 100),
            h264_limits,
            &context,
        ))?;
        assert_eq!(h264_10bit.encoded_bytes()?, b"H10P4");
        assert_eq!(h264_10bit.dimensions(), (2, 2));
        assert_eq!(h264_10bit.frame_rate(), (2997, 100));
        assert_eq!(h264_10bit.frame_count(), 3);
        assert_eq!(h264_10bit.codec(), NativeVideoCodec::H264);
        assert_eq!(h264_10bit.container(), NativeVideoContainer::Mp4);
        assert_eq!(h264_10bit.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            h264_10bit.pixel_format(),
            NativeVideoPixelFormat::Yuv420p10le
        );
        assert!(!h264_10bit.has_alpha());
        assert_eq!(
            h264_10bit.content_sha256(),
            <[u8; 32]>::from(Sha256::digest(b"H10P4"))
        );
        assert_eq!(scratch.in_use_bytes(), 0);
        drop(h264_10bit);

        let planning_limits = NativeVideoCodecLimits::checked(4, 1024, 1, 1024)?;
        let backing_service = NativeComponentH264Mp4CodecRequestService::checked(
            proxy.clone(),
            planning_limits,
            h264_limits,
        )?;
        let same_backing_service = NativeComponentH264Mp4CodecRequestService::checked(
            proxy.clone(),
            planning_limits,
            h264_limits,
        )?;
        let changed_backing_service = NativeComponentH264Mp4CodecRequestService::checked(
            proxy.clone(),
            NativeVideoCodecLimits::checked(4, 1024, 2, 1024)?,
            h264_limits,
        )?;
        assert_eq!(
            NativeComponentH264Mp4BackingService::identity(&backing_service),
            NativeComponentH264Mp4BackingService::identity(&same_backing_service)
        );
        assert_ne!(
            NativeComponentH264Mp4BackingService::identity(&backing_service),
            NativeComponentH264Mp4BackingService::identity(&changed_backing_service)
        );
        let base_backing_identity =
            NativeComponentH264Mp4BackingService::identity(&backing_service).configuration_sha256();
        for (changed_planning, changed_sequence) in [
            (
                NativeVideoCodecLimits::checked(3, 1024, 1, 1024)?,
                h264_limits,
            ),
            (
                NativeVideoCodecLimits::checked(4, 512, 1, 1024)?,
                h264_limits,
            ),
            (
                NativeVideoCodecLimits::checked(4, 1024, 2, 1024)?,
                h264_limits,
            ),
            (
                NativeVideoCodecLimits::checked(4, 1024, 1, 512)?,
                h264_limits,
            ),
            (
                planning_limits,
                NativeH264Mp4SequenceLimits::checked(2048, 256, 1024, 32, 4, 1024)?,
            ),
            (
                planning_limits,
                NativeH264Mp4SequenceLimits::checked(1024, 512, 1024, 32, 4, 1024)?,
            ),
            (
                planning_limits,
                NativeH264Mp4SequenceLimits::checked(1024, 256, 2048, 32, 4, 1024)?,
            ),
            (
                planning_limits,
                NativeH264Mp4SequenceLimits::checked(1024, 256, 1024, 64, 4, 1024)?,
            ),
            (
                planning_limits,
                NativeH264Mp4SequenceLimits::checked(1024, 256, 1024, 32, 5, 1024)?,
            ),
            (
                planning_limits,
                NativeH264Mp4SequenceLimits::checked(1024, 256, 1024, 32, 4, 2048)?,
            ),
        ] {
            let changed_service = NativeComponentH264Mp4CodecRequestService::checked(
                proxy.clone(),
                changed_planning,
                changed_sequence,
            )?;
            assert_ne!(
                NativeComponentH264Mp4BackingService::identity(&changed_service)
                    .configuration_sha256(),
                base_backing_identity
            );
        }
        let source_video = Arc::new(NativeVideoPayload::checked(
            image.tensor().clone(),
            2_997,
            100,
            NativeVideoBitDepth::Eight,
            None,
            None,
            BTreeMap::from([("prompt".to_owned(), "source-only".to_owned())]),
        )?);
        let source_digest = *source_video.semantic_digest_sha256();
        let backing_request = NativeComponentH264Mp4BackingRequest::checked(source_video.clone())?;
        let backing = block_on(backing_service.encode_backing(backing_request, &context))?;
        let encoded = backing
            .encoded()
            .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidProjection)?;
        assert_eq!(encoded.bytes().contiguous_bytes()?, b"HMP4");
        assert_eq!(
            encoded.content_sha256(),
            &<[u8; 32]>::from(Sha256::digest(b"HMP4"))
        );
        assert_eq!(encoded.source_video_sha256(), &source_digest);
        assert_eq!(encoded.dimensions(), (2, 2));
        assert_eq!(encoded.frame_rate(), (2997, 100));
        assert_eq!(encoded.frame_count(), 3);
        assert_eq!(encoded.codec(), NativeVideoCodec::H264);
        assert_eq!(encoded.container(), NativeVideoContainer::Mp4);
        assert_eq!(encoded.bit_depth(), NativeVideoBitDepth::Eight);
        assert_eq!(encoded.pixel_format(), NativeVideoPixelFormat::Yuv420p);
        assert!(!encoded.has_alpha());
        let output_storage_ids = h264_output_storage_ids
            .lock()
            .map_err(|_| "H.264 output storage mutex was poisoned")?;
        assert_eq!(
            output_storage_ids.last().copied(),
            Some(encoded.bytes().storage_id())
        );
        drop(output_storage_ids);
        assert_eq!(scratch.in_use_bytes(), 0);
        drop(backing);

        let ten_bit_source_video = Arc::new(NativeVideoPayload::checked(
            image.tensor().clone(),
            2_997,
            100,
            NativeVideoBitDepth::Ten,
            None,
            None,
            BTreeMap::from([("prompt".to_owned(), "ten-bit".to_owned())]),
        )?);
        let ten_bit_source_digest = *ten_bit_source_video.semantic_digest_sha256();
        let ten_bit_request = NativeComponentH264Mp4BackingRequest::checked(ten_bit_source_video)?;
        let ten_bit_backing = block_on(backing_service.encode_backing(ten_bit_request, &context))?;
        let ten_bit_encoded = ten_bit_backing
            .encoded()
            .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidProjection)?;
        assert_eq!(ten_bit_encoded.bytes().contiguous_bytes()?, b"H10P4");
        assert_eq!(
            ten_bit_encoded.content_sha256(),
            &<[u8; 32]>::from(Sha256::digest(b"H10P4"))
        );
        assert_eq!(
            ten_bit_encoded.source_video_sha256(),
            &ten_bit_source_digest
        );
        assert_eq!(ten_bit_encoded.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            ten_bit_encoded.pixel_format(),
            NativeVideoPixelFormat::Yuv420p10le
        );
        let output_storage_ids = h264_output_storage_ids
            .lock()
            .map_err(|_| "H.264 output storage mutex was poisoned")?;
        assert_eq!(
            output_storage_ids.last().copied(),
            Some(ten_bit_encoded.bytes().storage_id())
        );
        drop(output_storage_ids);
        assert_eq!(scratch.in_use_bytes(), 0);
        drop(ten_bit_backing);

        assert!(matches!(
            NativeComponentH264Mp4CodecRequestService::checked(
                proxy.clone(),
                NativeVideoCodecLimits::checked(4, 1024, 1, 2048)?,
                h264_limits,
            ),
            Err(NativeComponentH264Mp4BackingServiceError::InvalidRequest)
        ));
        let pre_cancelled = CancellationToken::default();
        assert!(pre_cancelled.cancel());
        let pre_cancelled_context = request_context(scratch.clone(), &pre_cancelled);
        let pre_cancelled_request = NativeComponentH264Mp4BackingRequest::checked(source_video)?;
        assert!(matches!(
            block_on(
                backing_service.encode_backing(pre_cancelled_request, &pre_cancelled_context,)
            ),
            Err(NativeComponentH264Mp4BackingServiceError::Cancelled)
        ));
        assert_eq!(scratch.in_use_bytes(), 0);

        let metadata_limits = crate::NativeVideoContainerMetadataLimits::checked(3, 16, 16, 64)?;
        let webm_service =
            NativeWebmCodecRequestService::checked(proxy.clone(), batch_limits, metadata_limits)?;
        let same_webm_service =
            NativeWebmCodecRequestService::checked(proxy.clone(), batch_limits, metadata_limits)?;
        let changed_webm_service = NativeWebmCodecRequestService::checked(
            proxy,
            batch_limits,
            crate::NativeVideoContainerMetadataLimits::checked(4, 16, 16, 64)?,
        )?;
        assert_eq!(
            NativeWebmEncodeService::identity(&webm_service),
            NativeWebmEncodeService::identity(&same_webm_service)
        );
        assert_ne!(
            NativeWebmEncodeService::identity(&webm_service),
            NativeWebmEncodeService::identity(&changed_webm_service)
        );
        let vp9_request = NativeWebmEncodeRequest::checked(
            image.clone(),
            NativeVideoCodec::Vp9,
            (2997, 125),
            NativeVideoCrf::checked(31.5)?,
            vec![
                ("prompt".to_owned(), "first".to_owned()),
                ("workflow".to_owned(), "{}".to_owned()),
                ("prompt".to_owned(), "last".to_owned()),
            ],
        )?;
        let vp9_result = block_on(webm_service.encode_webm(vp9_request, &context))?;
        assert_eq!(vp9_result.bytes().contiguous_bytes()?, b"HPPPT");
        assert_eq!(vp9_result.codec(), NativeVideoCodec::Vp9);
        assert_eq!(vp9_result.dimensions(), (2, 2));
        assert_eq!(vp9_result.frame_rate(), (2997, 125));
        assert_eq!(vp9_result.frame_count(), 3);
        assert_eq!(vp9_result.pixel_format(), NativeVideoPixelFormat::Yuva420p);
        assert!(vp9_result.has_alpha());

        let av1_request = NativeWebmEncodeRequest::checked(
            image,
            NativeVideoCodec::Av1,
            (2997, 125),
            NativeVideoCrf::checked(31.5)?,
            vec![
                ("prompt".to_owned(), "av1".to_owned()),
                ("workflow".to_owned(), "{}".to_owned()),
            ],
        )?;
        let av1_result = block_on(webm_service.encode_webm(av1_request, &context))?;
        assert_eq!(av1_result.bytes().contiguous_bytes()?, b"HA1T");
        assert_eq!(av1_result.codec(), NativeVideoCodec::Av1);
        assert_eq!(av1_result.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            av1_result.pixel_format(),
            NativeVideoPixelFormat::Yuv420p10le
        );
        assert!(!av1_result.has_alpha());
        assert_eq!(scratch.in_use_bytes(), 0);
        service.shutdown()?;

        let events = events
            .lock()
            .map_err(|_| "thread event mutex was poisoned")?;
        let actor_thread = events.first().ok_or("missing actor start event")?.1;
        assert_ne!(actor_thread, thread::current().id());
        assert_eq!(
            events.iter().filter(|event| event.0 == "request").count(),
            9
        );
        assert!(events.iter().all(|event| event.1 == actor_thread));
        Ok(())
    }

    #[test]
    fn owned_encoded_output_materialization_is_accounted_atomic_and_retryable()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = BackendWorkspaceAuthority::create_backend(1024)?;
        let scratch = authority.authorize_workspace(1024)?;
        let context = request_context(scratch.clone(), &cancellation);
        let source_image = ImageTensor::from_f32(&backend, &context, 3, 2, 2, 3, &[0.0; 36])?;
        let source_video = NativeVideoPayload::checked(
            source_image.tensor().clone(),
            125,
            2_997,
            NativeVideoBitDepth::Eight,
            None,
            None,
            BTreeMap::new(),
        )?;
        let baseline = backend.memory_snapshot().current_bytes;

        let encoded =
            materialize_owned_vp9_webm(&backend, &context, b"HPPPT", 2, 2, (125, 2997), 3, false)?;
        assert_eq!(scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, baseline + 16);
        drop(encoded);
        assert_eq!(backend.memory_snapshot().current_bytes, baseline);

        let h264 = materialize_owned_h264_mp4(
            &backend,
            &context,
            b"HMP4",
            2,
            2,
            (125, 2997),
            3,
            NativeVideoBitDepth::Eight,
        )?;
        assert_eq!(h264.encoded_bytes()?, b"HMP4");
        assert_eq!(scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, baseline + 16);
        drop(h264);
        assert_eq!(backend.memory_snapshot().current_bytes, baseline);

        let ten_bit_source_video = NativeVideoPayload::checked(
            source_image.tensor().clone(),
            125,
            2_997,
            NativeVideoBitDepth::Ten,
            None,
            None,
            BTreeMap::new(),
        )?;
        let ten_bit = materialize_owned_h264_mp4(
            &backend,
            &context,
            b"H10P4",
            2,
            2,
            (125, 2997),
            3,
            NativeVideoBitDepth::Ten,
        )?;
        assert_eq!(ten_bit.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(ten_bit.pixel_format(), NativeVideoPixelFormat::Yuv420p10le);
        let ten_bit_backing = checked_component_h264_mp4_backing_result(
            ten_bit,
            &ten_bit_source_video,
            (2, 2),
            (125, 2997),
            3,
            NativeVideoBitDepth::Ten,
            StreamId::DEFAULT,
        )?;
        let ten_bit_encoded = ten_bit_backing
            .encoded()
            .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidProjection)?;
        assert_eq!(ten_bit_encoded.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            ten_bit_encoded.pixel_format(),
            NativeVideoPixelFormat::Yuv420p10le
        );
        assert_eq!(backend.memory_snapshot().current_bytes, baseline + 16);
        drop(ten_bit_backing);
        assert_eq!(backend.memory_snapshot().current_bytes, baseline);

        let mismatched_depth = materialize_owned_h264_mp4(
            &backend,
            &context,
            b"H10P4",
            2,
            2,
            (125, 2997),
            3,
            NativeVideoBitDepth::Ten,
        )?;
        assert!(matches!(
            checked_component_h264_mp4_backing_result(
                mismatched_depth,
                &source_video,
                (2, 2),
                (125, 2997),
                3,
                NativeVideoBitDepth::Eight,
                StreamId::DEFAULT,
            ),
            Err(NativeComponentH264Mp4BackingServiceError::InvalidProjection)
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, baseline);

        let mismatched = materialize_owned_h264_mp4(
            &backend,
            &context,
            b"HMP4",
            3,
            2,
            (125, 2997),
            3,
            NativeVideoBitDepth::Eight,
        )?;
        assert!(matches!(
            checked_component_h264_mp4_backing_result(
                mismatched,
                &source_video,
                (2, 2),
                (125, 2997),
                3,
                NativeVideoBitDepth::Eight,
                StreamId::DEFAULT,
            ),
            Err(NativeComponentH264Mp4BackingServiceError::InvalidProjection)
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, baseline);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = request_context(scratch.clone(), &cancelled);
        assert!(matches!(
            materialize_owned_vp9_webm(
                &backend,
                &cancelled_context,
                b"HPPPT",
                2,
                2,
                (125, 2997),
                3,
                false,
            ),
            Err(NativeLtxvCodecThreadError::Cancelled)
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, baseline);
        assert_eq!(scratch.in_use_bytes(), 0);

        let (constrained_backend, constrained_authority) =
            BackendWorkspaceAuthority::create_backend(8)?;
        let constrained_scratch = constrained_authority.authorize_workspace(8)?;
        let constrained_context = request_context(constrained_scratch.clone(), &cancellation);
        assert!(matches!(
            materialize_owned_vp9_webm(
                &constrained_backend,
                &constrained_context,
                b"HPPPT",
                2,
                2,
                (125, 2997),
                3,
                false,
            ),
            Err(NativeLtxvCodecThreadError::ResourceExhausted)
        ));
        assert_eq!(constrained_backend.memory_snapshot().current_bytes, 0);
        assert_eq!(constrained_scratch.in_use_bytes(), 0);

        let retry =
            materialize_owned_vp9_webm(&backend, &context, b"HPPPT", 2, 2, (125, 2997), 3, false)?;
        assert_eq!(retry.encoded_bytes()?, b"HPPPT");
        assert_eq!(scratch.in_use_bytes(), 0);
        Ok(())
    }

    struct ThreadBoundDrop {
        events: Arc<Mutex<Vec<(&'static str, thread::ThreadId)>>>,
    }

    impl Drop for ThreadBoundDrop {
        fn drop(&mut self) {
            if let Ok(mut events) = self.events.lock() {
                events.push(("drop", thread::current().id()));
            } else {
                eprintln!("thread-bound codec test drop event mutex was poisoned");
            }
        }
    }

    #[test]
    fn retained_ltxv_codec_thread_test_tensor_contract_is_cpu_f32() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (_backend, image, _scratch) = test_image_and_context(&cancellation)?;
        let descriptor: &TensorDescriptor = image.tensor().descriptor();
        assert_eq!(descriptor.dtype(), DType::F32);
        assert_eq!(descriptor.device(), DeviceId::CPU);
        assert_eq!(descriptor.layout(), Layout::Contiguous);
        Ok(())
    }
}
