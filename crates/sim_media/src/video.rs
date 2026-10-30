use crate::{
    NativeCancellableMediaPayloadError, NativeMediaPayloadError, NativeVideoBitDepth,
    NativeVideoComponentsPayload, NativeVideoPayload,
};
use sim_tensor::{DType, DeviceId};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, mem::size_of, sync::Arc};
use thiserror::Error;

const SOURCE_RATE_DENOMINATOR: u64 = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVideoDecodeWindow {
    start_seconds_bits: u64,
    duration_seconds_bits: u64,
}

impl NativeVideoDecodeWindow {
    pub fn checked(
        start_seconds: f64,
        duration_seconds: f64,
    ) -> Result<Self, NativeVideoDecodePlanError> {
        if !start_seconds.is_finite()
            || start_seconds < 0.0
            || !duration_seconds.is_finite()
            || duration_seconds < 0.0
            || !matches!(start_seconds + duration_seconds, value if value.is_finite())
        {
            return Err(NativeVideoDecodePlanError::InvalidWindow);
        }
        let start_seconds = if start_seconds == 0.0 {
            0.0
        } else {
            start_seconds
        };
        let duration_seconds = if duration_seconds == 0.0 {
            0.0
        } else {
            duration_seconds
        };
        Ok(Self {
            start_seconds_bits: start_seconds.to_bits(),
            duration_seconds_bits: duration_seconds.to_bits(),
        })
    }

    pub const fn start_seconds(self) -> f64 {
        f64::from_bits(self.start_seconds_bits)
    }

    pub const fn duration_seconds(self) -> f64 {
        f64::from_bits(self.duration_seconds_bits)
    }

    pub const fn identity_bits(self) -> (u64, u64) {
        (self.start_seconds_bits, self.duration_seconds_bits)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVideoDecodeLimits {
    maximum_input_bytes: usize,
    maximum_streams: usize,
    maximum_packet_iterations: usize,
    maximum_receive_iterations: usize,
    maximum_frames: usize,
    maximum_pixels_per_frame: u64,
    maximum_audio_samples: usize,
    maximum_metadata_entries: usize,
    maximum_metadata_key_bytes: usize,
    maximum_metadata_value_bytes: usize,
    maximum_metadata_bytes: usize,
    maximum_output_bytes: usize,
    maximum_native_session_bytes: u64,
    avio_buffer_bytes: usize,
}

impl NativeVideoDecodeLimits {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn checked(
        maximum_input_bytes: usize,
        maximum_streams: usize,
        maximum_packet_iterations: usize,
        maximum_receive_iterations: usize,
        maximum_frames: usize,
        maximum_pixels_per_frame: u64,
        maximum_audio_samples: usize,
        maximum_metadata_entries: usize,
        maximum_metadata_key_bytes: usize,
        maximum_metadata_value_bytes: usize,
        maximum_metadata_bytes: usize,
        maximum_output_bytes: usize,
        maximum_native_session_bytes: u64,
        avio_buffer_bytes: usize,
    ) -> Result<Self, NativeVideoDecodePlanError> {
        if maximum_input_bytes == 0
            || maximum_streams == 0
            || maximum_packet_iterations == 0
            || maximum_receive_iterations == 0
            || maximum_frames == 0
            || maximum_pixels_per_frame == 0
            || maximum_audio_samples == 0
            || !(1..=128).contains(&maximum_metadata_entries)
            || !(1..=4_096).contains(&maximum_metadata_key_bytes)
            || !(1..=4_096).contains(&maximum_metadata_value_bytes)
            || maximum_metadata_bytes == 0
            || maximum_output_bytes == 0
            || maximum_native_session_bytes == 0
            || avio_buffer_bytes == 0
            || avio_buffer_bytes > maximum_input_bytes
        {
            return Err(NativeVideoDecodePlanError::InvalidLimits);
        }
        Ok(Self {
            maximum_input_bytes,
            maximum_streams,
            maximum_packet_iterations,
            maximum_receive_iterations,
            maximum_frames,
            maximum_pixels_per_frame,
            maximum_audio_samples,
            maximum_metadata_entries,
            maximum_metadata_key_bytes,
            maximum_metadata_value_bytes,
            maximum_metadata_bytes,
            maximum_output_bytes,
            maximum_native_session_bytes,
            avio_buffer_bytes,
        })
    }

    pub const fn reviewed() -> Self {
        Self {
            maximum_input_bytes: 512 * 1024 * 1024,
            maximum_streams: 32,
            maximum_packet_iterations: 2_000_000,
            maximum_receive_iterations: 4_000_000,
            maximum_frames: 56,
            maximum_pixels_per_frame: 1_280 * 736,
            maximum_audio_samples: 11_520_000,
            maximum_metadata_entries: 128,
            maximum_metadata_key_bytes: 4_096,
            maximum_metadata_value_bytes: 4_096,
            maximum_metadata_bytes: 1024 * 1024,
            maximum_output_bytes: 900_000_000,
            maximum_native_session_bytes: 128 * 1024 * 1024,
            avio_buffer_bytes: 64 * 1024,
        }
    }

    pub const fn maximum_input_bytes(self) -> usize {
        self.maximum_input_bytes
    }

    pub const fn maximum_streams(self) -> usize {
        self.maximum_streams
    }

    pub const fn maximum_packet_iterations(self) -> usize {
        self.maximum_packet_iterations
    }

    pub const fn maximum_receive_iterations(self) -> usize {
        self.maximum_receive_iterations
    }

    pub const fn maximum_frames(self) -> usize {
        self.maximum_frames
    }

    pub const fn maximum_pixels_per_frame(self) -> u64 {
        self.maximum_pixels_per_frame
    }

    pub const fn maximum_audio_samples(self) -> usize {
        self.maximum_audio_samples
    }

    pub const fn maximum_metadata_entries(self) -> usize {
        self.maximum_metadata_entries
    }

    pub const fn maximum_metadata_key_bytes(self) -> usize {
        self.maximum_metadata_key_bytes
    }

    pub const fn maximum_metadata_value_bytes(self) -> usize {
        self.maximum_metadata_value_bytes
    }

    pub const fn maximum_metadata_bytes(self) -> usize {
        self.maximum_metadata_bytes
    }

    pub const fn maximum_output_bytes(self) -> usize {
        self.maximum_output_bytes
    }

    pub const fn maximum_native_session_bytes(self) -> u64 {
        self.maximum_native_session_bytes
    }

    pub const fn avio_buffer_bytes(self) -> usize {
        self.avio_buffer_bytes
    }

    pub const fn configuration_values(self) -> [u64; 14] {
        [
            self.maximum_input_bytes as u64,
            self.maximum_streams as u64,
            self.maximum_packet_iterations as u64,
            self.maximum_receive_iterations as u64,
            self.maximum_frames as u64,
            self.maximum_pixels_per_frame,
            self.maximum_audio_samples as u64,
            self.maximum_metadata_entries as u64,
            self.maximum_metadata_key_bytes as u64,
            self.maximum_metadata_value_bytes as u64,
            self.maximum_metadata_bytes as u64,
            self.maximum_output_bytes as u64,
            self.maximum_native_session_bytes,
            self.avio_buffer_bytes as u64,
        ]
    }

    pub fn maximum_workspace_peak_bytes(self) -> Result<u64, NativeVideoDecodePlanError> {
        let pixels = self
            .maximum_pixels_per_frame
            .checked_mul(
                u64::try_from(self.maximum_frames)
                    .map_err(|_| NativeVideoDecodePlanError::InvalidLimits)?,
            )
            .ok_or(NativeVideoDecodePlanError::InvalidLimits)?;
        let frame_bytes = pixels
            .checked_mul(3)
            .and_then(|value| value.checked_mul(size_of::<f32>() as u64))
            .ok_or(NativeVideoDecodePlanError::InvalidLimits)?;
        let alpha_bytes = pixels
            .checked_mul(size_of::<f32>() as u64)
            .ok_or(NativeVideoDecodePlanError::InvalidLimits)?;
        let audio_bytes = u64::try_from(self.maximum_audio_samples)
            .map_err(|_| NativeVideoDecodePlanError::InvalidLimits)?
            .checked_mul(size_of::<f32>() as u64)
            .ok_or(NativeVideoDecodePlanError::InvalidLimits)?;
        let staging_bytes = frame_bytes
            .checked_add(alpha_bytes)
            .and_then(|value| value.checked_add(audio_bytes))
            .filter(|value| {
                usize::try_from(*value).is_ok_and(|value| value <= self.maximum_output_bytes)
            })
            .ok_or(NativeVideoDecodePlanError::InvalidLimits)?;
        let maximum_video_conversion_bytes = self
            .maximum_pixels_per_frame
            .checked_mul(4)
            .and_then(|value| value.checked_mul(size_of::<f32>() as u64))
            .ok_or(NativeVideoDecodePlanError::InvalidLimits)?;
        let conversion_bytes = maximum_video_conversion_bytes.max(audio_bytes);
        staging_bytes
            .checked_mul(2)
            .and_then(|value| value.checked_add(conversion_bytes))
            .and_then(|value| value.checked_add(self.maximum_native_session_bytes))
            .and_then(|value| value.checked_add(self.maximum_metadata_bytes as u64))
            .and_then(|value| value.checked_add(self.avio_buffer_bytes as u64))
            .ok_or(NativeVideoDecodePlanError::InvalidLimits)
    }

    pub fn checked_workspace_peak_bytes(
        self,
        maximum_workspace_bytes: u64,
    ) -> Result<u64, NativeVideoDecodePlanError> {
        let peak = self.maximum_workspace_peak_bytes()?;
        if peak > maximum_workspace_bytes {
            return Err(NativeVideoDecodePlanError::LimitExceeded);
        }
        Ok(peak)
    }
}

#[derive(Clone)]
pub struct NativeVideoDecodeRequest {
    encoded_bytes: Arc<[u8]>,
    window: NativeVideoDecodeWindow,
}

impl std::fmt::Debug for NativeVideoDecodeRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeVideoDecodeRequest")
            .field("encoded_byte_length", &self.encoded_bytes.len())
            .field("window", &self.window)
            .finish()
    }
}

impl NativeVideoDecodeRequest {
    pub fn checked(
        encoded_bytes: Arc<[u8]>,
        window: NativeVideoDecodeWindow,
    ) -> Result<Self, NativeVideoDecodePlanError> {
        if encoded_bytes.is_empty() {
            return Err(NativeVideoDecodePlanError::EmptyInput);
        }
        Ok(Self {
            encoded_bytes,
            window,
        })
    }

    pub fn encoded_bytes(&self) -> &[u8] {
        &self.encoded_bytes
    }

    pub const fn window(&self) -> NativeVideoDecodeWindow {
        self.window
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoSkippedStreamKind {
    Audio,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoSkippedStreamReason {
    MissingDecoder,
    UnsupportedCodec,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeVideoSkippedStreamDiagnostic {
    stream_index: u32,
    kind: NativeVideoSkippedStreamKind,
    codec_id: i32,
    reason: NativeVideoSkippedStreamReason,
}

impl NativeVideoSkippedStreamDiagnostic {
    pub const fn checked(
        stream_index: u32,
        kind: NativeVideoSkippedStreamKind,
        codec_id: i32,
        reason: NativeVideoSkippedStreamReason,
    ) -> Self {
        Self {
            stream_index,
            kind,
            codec_id,
            reason,
        }
    }

    pub const fn stream_index(&self) -> u32 {
        self.stream_index
    }

    pub const fn kind(&self) -> NativeVideoSkippedStreamKind {
        self.kind
    }

    pub const fn codec_id(&self) -> i32 {
        self.codec_id
    }

    pub const fn reason(&self) -> NativeVideoSkippedStreamReason {
        self.reason
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NativeVideoDecodeDiagnostics {
    skipped_streams: Vec<NativeVideoSkippedStreamDiagnostic>,
}

impl NativeVideoDecodeDiagnostics {
    pub fn checked(
        skipped_streams: Vec<NativeVideoSkippedStreamDiagnostic>,
        maximum_streams: usize,
    ) -> Result<Self, NativeVideoDecodePlanError> {
        if skipped_streams.len() > maximum_streams
            || skipped_streams.iter().any(|diagnostic| {
                diagnostic.codec_id < 0
                    || usize::try_from(diagnostic.stream_index)
                        .map_or(true, |index| index >= maximum_streams)
            })
            || skipped_streams
                .windows(2)
                .any(|pair| pair[0].stream_index >= pair[1].stream_index)
        {
            return Err(NativeVideoDecodePlanError::InvalidOutput);
        }
        Ok(Self { skipped_streams })
    }

    pub fn skipped_streams(&self) -> &[NativeVideoSkippedStreamDiagnostic] {
        &self.skipped_streams
    }
}

#[derive(Clone, Debug)]
pub struct NativeDecodedVideo {
    video: NativeVideoPayload,
    diagnostics: NativeVideoDecodeDiagnostics,
}

impl NativeDecodedVideo {
    pub fn checked(
        video: NativeVideoPayload,
        diagnostics: NativeVideoDecodeDiagnostics,
    ) -> Result<Self, NativeVideoDecodePlanError> {
        validate_decoded_video_shape(&video)?;
        video
            .validate()
            .map_err(|_| NativeVideoDecodePlanError::InvalidOutput)?;
        Ok(Self { video, diagnostics })
    }

    pub fn checked_cancellable(
        video: NativeVideoPayload,
        diagnostics: NativeVideoDecodeDiagnostics,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeCancellableMediaPayloadError> {
        validate_decoded_video_shape(&video).map_err(|_| NativeMediaPayloadError::InvalidVideo)?;
        video.validate_components_cancellable(cancellation)?;
        Ok(Self { video, diagnostics })
    }

    pub const fn video(&self) -> &NativeVideoPayload {
        &self.video
    }

    pub const fn diagnostics(&self) -> &NativeVideoDecodeDiagnostics {
        &self.diagnostics
    }

    pub fn into_video(self) -> NativeVideoPayload {
        self.video
    }
}

fn validate_decoded_video_shape(
    video: &NativeVideoPayload,
) -> Result<(), NativeVideoDecodePlanError> {
    let components = video
        .components()
        .ok_or(NativeVideoDecodePlanError::InvalidOutput)?;
    let frames = components.frames().descriptor();
    let frame_shape = frames.shape();
    let valid_alpha = match components.alpha() {
        Some(alpha) => {
            let descriptor = alpha.descriptor();
            descriptor.dtype() == DType::F32
                && descriptor.device() == DeviceId::CPU
                && descriptor
                    .is_contiguous()
                    .map_err(|_| NativeVideoDecodePlanError::InvalidOutput)?
        }
        None => true,
    };
    let valid_audio = match components.audio() {
        Some(audio) => {
            let descriptor = audio.waveform().descriptor();
            descriptor.dtype() == DType::F32
                && descriptor.device() == DeviceId::CPU
                && descriptor.shape().first() == Some(&1)
                && descriptor
                    .is_contiguous()
                    .map_err(|_| NativeVideoDecodePlanError::InvalidOutput)?
        }
        None => true,
    };
    if frames.dtype() != DType::F32
        || frames.device() != DeviceId::CPU
        || frame_shape.len() != 4
        || frame_shape[3] != 3
        || !frames
            .is_contiguous()
            .map_err(|_| NativeVideoDecodePlanError::InvalidOutput)?
        || !valid_alpha
        || !valid_audio
    {
        return Err(NativeVideoDecodePlanError::InvalidOutput);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum NativeVideoDecodePlanError {
    #[error("native video decode input is empty")]
    EmptyInput,
    #[error("native video decode window is invalid")]
    InvalidWindow,
    #[error("native video decode limits are invalid")]
    InvalidLimits,
    #[error("native video decode input or output exceeds its checked limits")]
    LimitExceeded,
    #[error("native video decode output is invalid")]
    InvalidOutput,
}

pub fn checked_decode_metadata(
    entries: impl IntoIterator<Item = (String, String)>,
    limits: NativeVideoDecodeLimits,
) -> Result<BTreeMap<String, String>, NativeVideoDecodePlanError> {
    let mut metadata = BTreeMap::new();
    let mut aggregate_bytes = 0_usize;
    for (key, value) in entries {
        if key.is_empty()
            || key.chars().any(char::is_control)
            || value.chars().any(char::is_control)
            || metadata.contains_key(&key)
            || metadata.len() >= limits.maximum_metadata_entries
            || key.len() > limits.maximum_metadata_key_bytes
            || value.len() > limits.maximum_metadata_value_bytes
        {
            return Err(NativeVideoDecodePlanError::InvalidOutput);
        }
        aggregate_bytes = aggregate_bytes
            .checked_add(key.len())
            .and_then(|bytes| bytes.checked_add(value.len()))
            .filter(|bytes| *bytes <= limits.maximum_metadata_bytes)
            .ok_or(NativeVideoDecodePlanError::LimitExceeded)?;
        metadata.insert(key, value);
    }
    Ok(metadata)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoContainer {
    Mp4,
    Webm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoCodec {
    H264,
    Vp9,
    Av1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoPixelFormat {
    Yuv420p,
    Yuv420p10le,
    Yuva420p,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoAlphaPolicy {
    Discard,
    Preserve,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoMetadataPolicy {
    Include,
    Exclude,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoAudioLayout {
    Mono,
    Stereo,
    Surround51,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoEncodeOptions {
    ComponentMp4 { metadata: NativeVideoMetadataPolicy },
    WebmVp9 { crf: NativeVideoCrf },
    WebmAv1 { crf: NativeVideoCrf },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NativeVideoCrf {
    bits: u64,
}

impl NativeVideoCrf {
    pub fn checked(value: f64) -> Result<Self, NativeVideoCodecPlanError> {
        if !value.is_finite() || !(0.0..=63.0).contains(&value) {
            return Err(NativeVideoCodecPlanError::InvalidOptions);
        }
        Ok(Self {
            bits: value.to_bits(),
        })
    }

    pub const fn value(self) -> f64 {
        f64::from_bits(self.bits)
    }

    pub const fn bits(self) -> u64 {
        self.bits
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVideoCodecLimits {
    max_frames: u64,
    max_pixels_per_frame: u64,
    max_audio_samples: u64,
    max_encoded_bytes: u64,
}

impl NativeVideoCodecLimits {
    pub fn checked(
        max_frames: u64,
        max_pixels_per_frame: u64,
        max_audio_samples: u64,
        max_encoded_bytes: u64,
    ) -> Result<Self, NativeVideoCodecPlanError> {
        if max_frames == 0
            || max_pixels_per_frame == 0
            || max_audio_samples == 0
            || max_encoded_bytes == 0
        {
            return Err(NativeVideoCodecPlanError::InvalidLimits);
        }
        Ok(Self {
            max_frames,
            max_pixels_per_frame,
            max_audio_samples,
            max_encoded_bytes,
        })
    }

    pub const fn max_frames(self) -> u64 {
        self.max_frames
    }

    pub const fn max_pixels_per_frame(self) -> u64 {
        self.max_pixels_per_frame
    }

    pub const fn max_audio_samples(self) -> u64 {
        self.max_audio_samples
    }

    pub const fn max_encoded_bytes(self) -> u64 {
        self.max_encoded_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVideoAudioPlan {
    layout: NativeVideoAudioLayout,
    input_channels: u64,
    sample_rate: u32,
    sample_count: u64,
}

impl NativeVideoAudioPlan {
    pub const fn layout(self) -> NativeVideoAudioLayout {
        self.layout
    }

    pub const fn input_channels(self) -> u64 {
        self.input_channels
    }

    pub const fn sample_rate(self) -> u32 {
        self.sample_rate
    }

    pub const fn sample_count(self) -> u64 {
        self.sample_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeVideoEncodePlan {
    container: NativeVideoContainer,
    codec: NativeVideoCodec,
    pixel_format: NativeVideoPixelFormat,
    source_bit_depth: NativeVideoBitDepth,
    output_bit_depth: NativeVideoBitDepth,
    source_frame_rate: (u64, u64),
    encode_frame_rate: (u64, u64),
    frame_count: u64,
    dimensions: (u64, u64),
    alpha: NativeVideoAlphaPolicy,
    audio: Option<NativeVideoAudioPlan>,
    metadata: NativeVideoMetadataPolicy,
    crf: Option<NativeVideoCrf>,
    preset: Option<u8>,
    max_encoded_bytes: u64,
}

impl NativeVideoEncodePlan {
    pub const fn container(&self) -> NativeVideoContainer {
        self.container
    }

    pub const fn codec(&self) -> NativeVideoCodec {
        self.codec
    }

    pub const fn pixel_format(&self) -> NativeVideoPixelFormat {
        self.pixel_format
    }

    pub const fn source_bit_depth(&self) -> NativeVideoBitDepth {
        self.source_bit_depth
    }

    pub const fn output_bit_depth(&self) -> NativeVideoBitDepth {
        self.output_bit_depth
    }

    pub const fn source_frame_rate(&self) -> (u64, u64) {
        self.source_frame_rate
    }

    pub const fn encode_frame_rate(&self) -> (u64, u64) {
        self.encode_frame_rate
    }

    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub const fn dimensions(&self) -> (u64, u64) {
        self.dimensions
    }

    pub const fn alpha(&self) -> NativeVideoAlphaPolicy {
        self.alpha
    }

    pub const fn audio(&self) -> Option<NativeVideoAudioPlan> {
        self.audio
    }

    pub const fn metadata(&self) -> NativeVideoMetadataPolicy {
        self.metadata
    }

    pub const fn crf(&self) -> Option<NativeVideoCrf> {
        self.crf
    }

    pub const fn preset(&self) -> Option<u8> {
        self.preset
    }

    pub const fn max_encoded_bytes(&self) -> u64 {
        self.max_encoded_bytes
    }
}

#[derive(Debug, Error)]
pub enum NativeVideoCodecPlanError {
    #[error("video codec planning was cancelled")]
    Cancelled,
    #[error("video component is invalid for the requested codec profile")]
    InvalidVideo,
    #[error("video codec options are invalid")]
    InvalidOptions,
    #[error("video codec limits must be nonzero")]
    InvalidLimits,
    #[error("video component exceeds the caller's codec limits")]
    LimitExceeded,
    #[error("video codec plan arithmetic overflowed")]
    Overflow,
}

pub fn plan_native_video_encode(
    video: &NativeVideoPayload,
    options: NativeVideoEncodeOptions,
    limits: NativeVideoCodecLimits,
    cancellation: &CancellationToken,
) -> Result<NativeVideoEncodePlan, NativeVideoCodecPlanError> {
    check_cancelled(cancellation)?;

    let components = video
        .components()
        .ok_or(NativeVideoCodecPlanError::InvalidVideo)?;
    let frame_shape = components.frames().descriptor().shape();
    let [frame_count, height, width, channels] = frame_shape else {
        return Err(NativeVideoCodecPlanError::InvalidVideo);
    };
    if !matches!(channels, 3 | 4) {
        return Err(NativeVideoCodecPlanError::InvalidVideo);
    }
    let pixels_per_frame = height
        .checked_mul(*width)
        .ok_or(NativeVideoCodecPlanError::Overflow)?;
    if *frame_count > limits.max_frames || pixels_per_frame > limits.max_pixels_per_frame {
        return Err(NativeVideoCodecPlanError::LimitExceeded);
    }

    let source_frame_rate = video.frame_rate();
    let encode_frame_rate = source_rounded_rational_frame_rate(source_frame_rate)?;
    let has_alpha = *channels == 4 || components.alpha().is_some();
    let source_bit_depth = video.bit_depth();

    let (container, codec, pixel_format, output_bit_depth, alpha, audio, metadata, crf, preset) =
        match options {
            NativeVideoEncodeOptions::ComponentMp4 { metadata } => {
                let (pixel_format, output_bit_depth) = match source_bit_depth {
                    NativeVideoBitDepth::Eight => {
                        (NativeVideoPixelFormat::Yuv420p, NativeVideoBitDepth::Eight)
                    }
                    NativeVideoBitDepth::Ten => (
                        NativeVideoPixelFormat::Yuv420p10le,
                        NativeVideoBitDepth::Ten,
                    ),
                };
                let audio = plan_audio(components, *frame_count, encode_frame_rate, limits)?;
                (
                    NativeVideoContainer::Mp4,
                    NativeVideoCodec::H264,
                    pixel_format,
                    output_bit_depth,
                    NativeVideoAlphaPolicy::Discard,
                    audio,
                    metadata,
                    None,
                    None,
                )
            }
            NativeVideoEncodeOptions::WebmVp9 { crf } => {
                require_webm_options(components, crf)?;
                (
                    NativeVideoContainer::Webm,
                    NativeVideoCodec::Vp9,
                    if has_alpha {
                        NativeVideoPixelFormat::Yuva420p
                    } else {
                        NativeVideoPixelFormat::Yuv420p
                    },
                    NativeVideoBitDepth::Eight,
                    if has_alpha {
                        NativeVideoAlphaPolicy::Preserve
                    } else {
                        NativeVideoAlphaPolicy::Discard
                    },
                    None,
                    NativeVideoMetadataPolicy::Include,
                    Some(crf),
                    None,
                )
            }
            NativeVideoEncodeOptions::WebmAv1 { crf } => {
                require_webm_options(components, crf)?;
                (
                    NativeVideoContainer::Webm,
                    NativeVideoCodec::Av1,
                    NativeVideoPixelFormat::Yuv420p10le,
                    NativeVideoBitDepth::Ten,
                    NativeVideoAlphaPolicy::Discard,
                    None,
                    NativeVideoMetadataPolicy::Include,
                    Some(crf),
                    Some(6),
                )
            }
        };

    check_cancelled(cancellation)?;
    Ok(NativeVideoEncodePlan {
        container,
        codec,
        pixel_format,
        source_bit_depth,
        output_bit_depth,
        source_frame_rate,
        encode_frame_rate,
        frame_count: *frame_count,
        dimensions: (*width, *height),
        alpha,
        audio,
        metadata,
        crf,
        preset,
        max_encoded_bytes: limits.max_encoded_bytes,
    })
}

fn plan_audio(
    video: &NativeVideoComponentsPayload,
    frame_count: u64,
    encode_frame_rate: (u64, u64),
    limits: NativeVideoCodecLimits,
) -> Result<Option<NativeVideoAudioPlan>, NativeVideoCodecPlanError> {
    let Some(audio) = video.audio() else {
        return Ok(None);
    };
    let shape = audio.waveform().descriptor().shape();
    let [batch, channels, available_samples] = shape else {
        return Err(NativeVideoCodecPlanError::InvalidVideo);
    };
    if *batch != 1 {
        return Err(NativeVideoCodecPlanError::InvalidVideo);
    }
    let requested_numerator = u128::from(audio.sample_rate())
        .checked_mul(u128::from(frame_count))
        .and_then(|value| value.checked_mul(u128::from(encode_frame_rate.1)))
        .ok_or(NativeVideoCodecPlanError::Overflow)?;
    let requested_samples = ceiling_divide(requested_numerator, u128::from(encode_frame_rate.0))?;
    let requested_samples =
        u64::try_from(requested_samples).map_err(|_| NativeVideoCodecPlanError::Overflow)?;
    let sample_count = (*available_samples).min(requested_samples);
    if sample_count > limits.max_audio_samples {
        return Err(NativeVideoCodecPlanError::LimitExceeded);
    }
    let layout = match channels {
        1 => NativeVideoAudioLayout::Mono,
        2 => NativeVideoAudioLayout::Stereo,
        6 => NativeVideoAudioLayout::Surround51,
        _ => NativeVideoAudioLayout::Stereo,
    };
    Ok(Some(NativeVideoAudioPlan {
        layout,
        input_channels: *channels,
        sample_rate: audio.sample_rate(),
        sample_count,
    }))
}

fn require_webm_options(
    video: &NativeVideoComponentsPayload,
    crf: NativeVideoCrf,
) -> Result<(), NativeVideoCodecPlanError> {
    if !(0.0..=63.0).contains(&crf.value()) || video.audio().is_some() {
        return Err(NativeVideoCodecPlanError::InvalidOptions);
    }
    Ok(())
}

pub fn source_rounded_rational_frame_rate(
    source_frame_rate: (u64, u64),
) -> Result<(u64, u64), NativeVideoCodecPlanError> {
    let scaled_numerator = u128::from(source_frame_rate.0)
        .checked_mul(u128::from(SOURCE_RATE_DENOMINATOR))
        .ok_or(NativeVideoCodecPlanError::Overflow)?;
    let rounded = round_ratio_ties_even(scaled_numerator, u128::from(source_frame_rate.1))?;
    let rounded = u64::try_from(rounded).map_err(|_| NativeVideoCodecPlanError::Overflow)?;
    if rounded == 0 {
        return Err(NativeVideoCodecPlanError::InvalidVideo);
    }
    let divisor = greatest_common_divisor(rounded, SOURCE_RATE_DENOMINATOR);
    Ok((rounded / divisor, SOURCE_RATE_DENOMINATOR / divisor))
}

pub fn source_rounded_millisecond_frame_rate(
    source_frame_rate: f64,
) -> Result<(u64, u64), NativeVideoCodecPlanError> {
    if !source_frame_rate.is_finite() || !(0.01..=1_000.0).contains(&source_frame_rate) {
        return Err(NativeVideoCodecPlanError::InvalidOptions);
    }
    let rounded = (source_frame_rate * SOURCE_RATE_DENOMINATOR as f64).round_ties_even();
    if rounded < 1.0 || rounded > u64::MAX as f64 {
        return Err(NativeVideoCodecPlanError::Overflow);
    }
    let rounded = rounded as u64;
    let divisor = greatest_common_divisor(rounded, SOURCE_RATE_DENOMINATOR);
    Ok((rounded / divisor, SOURCE_RATE_DENOMINATOR / divisor))
}

fn round_ratio_ties_even(
    numerator: u128,
    denominator: u128,
) -> Result<u128, NativeVideoCodecPlanError> {
    if denominator == 0 {
        return Err(NativeVideoCodecPlanError::InvalidVideo);
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let doubled_remainder = remainder
        .checked_mul(2)
        .ok_or(NativeVideoCodecPlanError::Overflow)?;
    if doubled_remainder > denominator || (doubled_remainder == denominator && quotient % 2 == 1) {
        quotient
            .checked_add(1)
            .ok_or(NativeVideoCodecPlanError::Overflow)
    } else {
        Ok(quotient)
    }
}

fn ceiling_divide(numerator: u128, denominator: u128) -> Result<u128, NativeVideoCodecPlanError> {
    if denominator == 0 {
        return Err(NativeVideoCodecPlanError::InvalidVideo);
    }
    let quotient = numerator / denominator;
    if numerator.is_multiple_of(denominator) {
        Ok(quotient)
    } else {
        quotient
            .checked_add(1)
            .ok_or(NativeVideoCodecPlanError::Overflow)
    }
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), NativeVideoCodecPlanError> {
    cancellation
        .check()
        .map_err(|_| NativeVideoCodecPlanError::Cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NativeAudioPayload;
    use sim_tensor::{
        CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext, StreamId, Tensor,
        TensorDescriptor,
    };
    use std::{collections::BTreeMap, error::Error};

    fn tensor(shape: Vec<u64>, dtype: DType, bytes: Vec<u8>) -> Result<Tensor, Box<dyn Error>> {
        let descriptor =
            TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, StreamId::DEFAULT)?;
        let byte_count = u64::try_from(bytes.len())?;
        let (backend, authority) =
            CpuWorkspaceAuthority::create_backend(byte_count.saturating_add(64))?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let (tensor, _) = backend.upload_bytes(descriptor, &bytes, &context)?;
        Ok(tensor)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn video(
        channels: u64,
        bit_depth: NativeVideoBitDepth,
        audio: Option<NativeAudioPayload>,
        alpha: bool,
    ) -> Result<NativeVideoPayload, Box<dyn Error>> {
        let alpha = alpha
            .then(|| tensor(vec![3, 2, 2, 1], DType::F32, f32_bytes(&[0.5; 12])))
            .transpose()?;
        Ok(NativeVideoPayload::checked(
            tensor(
                vec![3, 2, 2, channels],
                DType::U8,
                vec![127; usize::try_from(12_u64.checked_mul(channels).ok_or("overflow")?)?],
            )?,
            1_054_475_631_502_295,
            35_184_372_088_832,
            bit_depth,
            audio,
            alpha,
            BTreeMap::from([("prompt".to_owned(), "fixture".to_owned())]),
        )?)
    }

    fn limits() -> Result<NativeVideoCodecLimits, NativeVideoCodecPlanError> {
        NativeVideoCodecLimits::checked(16, 64, 1_000_000, 8_000_000)
    }

    #[test]
    fn general_video_decode_window_and_request_identity_are_canonical_and_redacted()
    -> Result<(), Box<dyn Error>> {
        let positive = NativeVideoDecodeWindow::checked(0.0, 0.0)?;
        let negative = NativeVideoDecodeWindow::checked(-0.0, -0.0)?;
        assert_eq!(positive, negative);
        assert_eq!(positive.identity_bits(), (0, 0));
        for (start, duration) in [
            (-1.0, 0.0),
            (0.0, -1.0),
            (f64::INFINITY, 0.0),
            (0.0, f64::NAN),
            (f64::MAX, f64::MAX),
        ] {
            assert!(NativeVideoDecodeWindow::checked(start, duration).is_err());
        }

        let distinctive_bytes: Arc<[u8]> = Arc::from(&b"do-not-log-this-video"[..]);
        let request = NativeVideoDecodeRequest::checked(distinctive_bytes, positive)?;
        let debug = format!("{request:?}");
        assert!(debug.contains("encoded_byte_length: 21"));
        assert!(!debug.contains("do-not-log-this-video"));
        Ok(())
    }

    #[test]
    fn general_video_decode_reviewed_peak_fits_exact_frozen_workspace() -> Result<(), Box<dyn Error>>
    {
        const FROZEN_CODEC_WORKSPACE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
        let reviewed = NativeVideoDecodeLimits::reviewed();
        let peak = reviewed.maximum_workspace_peak_bytes()?;
        assert_eq!(peak, 1_961_779_200);
        assert_eq!(
            reviewed.checked_workspace_peak_bytes(FROZEN_CODEC_WORKSPACE_BYTES)?,
            peak
        );
        assert!(
            reviewed
                .checked_workspace_peak_bytes(peak.saturating_sub(1))
                .is_err()
        );

        let invalid_output_limit = NativeVideoDecodeLimits::checked(
            reviewed.maximum_input_bytes,
            reviewed.maximum_streams,
            reviewed.maximum_packet_iterations,
            reviewed.maximum_receive_iterations,
            reviewed.maximum_frames,
            reviewed.maximum_pixels_per_frame,
            reviewed.maximum_audio_samples,
            reviewed.maximum_metadata_entries,
            reviewed.maximum_metadata_key_bytes,
            reviewed.maximum_metadata_value_bytes,
            reviewed.maximum_metadata_bytes,
            1,
            reviewed.maximum_native_session_bytes,
            reviewed.avio_buffer_bytes,
        )?;
        assert!(invalid_output_limit.maximum_workspace_peak_bytes().is_err());
        Ok(())
    }

    #[test]
    fn general_video_decode_diagnostics_reject_out_of_range_streams() {
        let diagnostic = NativeVideoSkippedStreamDiagnostic::checked(
            2,
            NativeVideoSkippedStreamKind::Audio,
            86018,
            NativeVideoSkippedStreamReason::UnsupportedCodec,
        );
        assert!(NativeVideoDecodeDiagnostics::checked(vec![diagnostic], 2).is_err());
        let diagnostic = NativeVideoSkippedStreamDiagnostic::checked(
            u32::MAX,
            NativeVideoSkippedStreamKind::Audio,
            86018,
            NativeVideoSkippedStreamReason::UnsupportedCodec,
        );
        assert!(NativeVideoDecodeDiagnostics::checked(vec![diagnostic], 2).is_err());
    }

    #[test]
    fn codec_crf_preserves_checked_source_float_bits() -> Result<(), Box<dyn Error>> {
        for value in [0.0, -0.0, 0.125, 31.5, 63.0] {
            let crf = NativeVideoCrf::checked(value)?;
            assert_eq!(crf.bits(), value.to_bits());
            assert_eq!(crf.value().to_bits(), value.to_bits());
        }
        for value in [f64::NEG_INFINITY, -0.01, 63.01, f64::INFINITY, f64::NAN] {
            assert!(matches!(
                NativeVideoCrf::checked(value),
                Err(NativeVideoCodecPlanError::InvalidOptions)
            ));
        }
        Ok(())
    }

    #[test]
    fn component_audio_plan_matches_source_encoder_rate_and_sample_admission()
    -> Result<(), Box<dyn Error>> {
        #[derive(serde::Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            frame_count: u64,
            frame_rate: (u64, u64),
            sample_rate: u32,
            channels: u64,
            available_samples: u64,
            output: Output,
        }
        #[derive(serde::Deserialize)]
        struct Output {
            frame_rate: (u64, u64),
            sample_rate: u32,
            layout: String,
            sample_count: u64,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../sim_test_support/fixtures/video/codec-component-audio-plan/source_audio_plan.json"
        ))?;
        assert_eq!(fixture.cases.len(), 18);
        for case in fixture.cases {
            let audio = NativeAudioPayload::checked(
                tensor(
                    vec![1, case.channels, case.available_samples],
                    DType::F32,
                    vec![0; usize::try_from(case.channels * case.available_samples * 4)?],
                )?,
                case.sample_rate,
            )?;
            let source = NativeVideoPayload::checked(
                tensor(
                    vec![case.frame_count, 2, 2, 3],
                    DType::F32,
                    vec![0; usize::try_from(case.frame_count * 12 * 4)?],
                )?,
                case.frame_rate.0,
                case.frame_rate.1,
                NativeVideoBitDepth::Eight,
                Some(audio),
                None,
                BTreeMap::new(),
            )?;
            let source_digest = *source.semantic_digest_sha256();
            let options = NativeVideoEncodeOptions::ComponentMp4 {
                metadata: NativeVideoMetadataPolicy::Exclude,
            };
            let cancellation = CancellationToken::default();
            assert!(matches!(
                plan_native_video_encode(
                    &source,
                    options,
                    NativeVideoCodecLimits::checked(240, 4, case.output.sample_count - 1, 1024)?,
                    &cancellation,
                ),
                Err(NativeVideoCodecPlanError::LimitExceeded)
            ));
            let plan = plan_native_video_encode(
                &source,
                options,
                NativeVideoCodecLimits::checked(240, 4, case.output.sample_count, 1024)?,
                &cancellation,
            )?;
            assert_eq!(plan.source_frame_rate(), case.frame_rate);
            assert_eq!(plan.encode_frame_rate(), case.output.frame_rate);
            let audio = plan.audio().ok_or("missing audio plan")?;
            let expected_layout = match case.output.layout.as_str() {
                "mono" => NativeVideoAudioLayout::Mono,
                "stereo" => NativeVideoAudioLayout::Stereo,
                "5.1" => NativeVideoAudioLayout::Surround51,
                _ => return Err("unexpected source layout".into()),
            };
            assert_eq!(audio.layout(), expected_layout);
            assert_eq!(audio.input_channels(), case.channels);
            assert_eq!(audio.sample_rate(), case.output.sample_rate);
            assert_eq!(audio.sample_count(), case.output.sample_count);
            assert_eq!(source.semantic_digest_sha256(), &source_digest);
        }
        Ok(())
    }

    #[test]
    fn codec_plan_preserves_source_profiles_without_executing_a_codec() -> Result<(), Box<dyn Error>>
    {
        let audio = NativeAudioPayload::checked(
            tensor(vec![1, 6, 20_000], DType::F32, vec![0; 480_000])?,
            48_000,
        )?;
        let component = video(4, NativeVideoBitDepth::Ten, Some(audio), true)?;
        let digest_before = *component.semantic_digest_sha256();
        let plan = plan_native_video_encode(
            &component,
            NativeVideoEncodeOptions::ComponentMp4 {
                metadata: NativeVideoMetadataPolicy::Exclude,
            },
            limits()?,
            &CancellationToken::default(),
        )?;
        assert_eq!(plan.container(), NativeVideoContainer::Mp4);
        assert_eq!(plan.codec(), NativeVideoCodec::H264);
        assert_eq!(plan.pixel_format(), NativeVideoPixelFormat::Yuv420p10le);
        assert_eq!(plan.output_bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            plan.source_frame_rate(),
            (1_054_475_631_502_295, 35_184_372_088_832)
        );
        assert_eq!(plan.encode_frame_rate(), (2_997, 100));
        assert_eq!(plan.alpha(), NativeVideoAlphaPolicy::Discard);
        let audio = plan.audio().ok_or("missing audio plan")?;
        assert_eq!(audio.layout(), NativeVideoAudioLayout::Surround51);
        assert_eq!(audio.input_channels(), 6);
        assert_eq!(audio.sample_count(), 4_805);
        assert_eq!(plan.metadata(), NativeVideoMetadataPolicy::Exclude);
        assert_eq!(component.semantic_digest_sha256(), &digest_before);

        let vp9 = plan_native_video_encode(
            &video(3, NativeVideoBitDepth::Ten, None, true)?,
            NativeVideoEncodeOptions::WebmVp9 {
                crf: NativeVideoCrf::checked(32.0)?,
            },
            limits()?,
            &CancellationToken::default(),
        )?;
        assert_eq!(vp9.codec(), NativeVideoCodec::Vp9);
        assert_eq!(vp9.pixel_format(), NativeVideoPixelFormat::Yuva420p);
        assert_eq!(vp9.output_bit_depth(), NativeVideoBitDepth::Eight);
        assert_eq!(vp9.alpha(), NativeVideoAlphaPolicy::Preserve);
        assert_eq!(vp9.crf(), Some(NativeVideoCrf::checked(32.0)?));

        let av1 = plan_native_video_encode(
            &video(4, NativeVideoBitDepth::Eight, None, false)?,
            NativeVideoEncodeOptions::WebmAv1 {
                crf: NativeVideoCrf::checked(63.0)?,
            },
            limits()?,
            &CancellationToken::default(),
        )?;
        assert_eq!(av1.codec(), NativeVideoCodec::Av1);
        assert_eq!(av1.pixel_format(), NativeVideoPixelFormat::Yuv420p10le);
        assert_eq!(av1.output_bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(av1.alpha(), NativeVideoAlphaPolicy::Discard);
        assert_eq!(av1.preset(), Some(6));
        Ok(())
    }

    #[test]
    fn codec_plan_rejects_unsupported_profiles_limits_and_cancellation()
    -> Result<(), Box<dyn Error>> {
        assert_eq!(round_ratio_ties_even(5, 2)?, 2);
        assert_eq!(round_ratio_ties_even(7, 2)?, 4);
        assert_eq!(
            source_rounded_rational_frame_rate((210_895_126_300_459, 8_796_093_022_208))?,
            (2_997, 125)
        );

        let grayscale = video(1, NativeVideoBitDepth::Eight, None, false)?;
        assert!(matches!(
            plan_native_video_encode(
                &grayscale,
                NativeVideoEncodeOptions::WebmVp9 {
                    crf: NativeVideoCrf::checked(0.0)?,
                },
                limits()?,
                &CancellationToken::default(),
            ),
            Err(NativeVideoCodecPlanError::InvalidVideo)
        ));
        let audio =
            NativeAudioPayload::checked(tensor(vec![1, 1, 4], DType::F32, vec![0; 16])?, 48_000)?;
        let with_audio = video(3, NativeVideoBitDepth::Eight, Some(audio), false)?;
        assert!(matches!(
            plan_native_video_encode(
                &with_audio,
                NativeVideoEncodeOptions::WebmVp9 {
                    crf: NativeVideoCrf {
                        bits: 64.0_f64.to_bits()
                    },
                },
                limits()?,
                &CancellationToken::default(),
            ),
            Err(NativeVideoCodecPlanError::InvalidOptions)
        ));
        assert!(matches!(
            plan_native_video_encode(
                &video(3, NativeVideoBitDepth::Eight, None, false)?,
                NativeVideoEncodeOptions::WebmAv1 {
                    crf: NativeVideoCrf::checked(1.0)?,
                },
                NativeVideoCodecLimits::checked(2, 64, 10, 128)?,
                &CancellationToken::default(),
            ),
            Err(NativeVideoCodecPlanError::LimitExceeded)
        ));
        assert!(NativeVideoCodecLimits::checked(0, 1, 1, 1).is_err());

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            plan_native_video_encode(
                &video(3, NativeVideoBitDepth::Eight, None, false)?,
                NativeVideoEncodeOptions::ComponentMp4 {
                    metadata: NativeVideoMetadataPolicy::Include,
                },
                limits()?,
                &cancellation,
            ),
            Err(NativeVideoCodecPlanError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn source_webm_frame_rate_uses_python_ties_even_millisecond_rounding()
    -> Result<(), Box<dyn Error>> {
        assert_eq!(source_rounded_millisecond_frame_rate(24.0)?, (24, 1));
        assert_eq!(source_rounded_millisecond_frame_rate(29.97)?, (2_997, 100));
        assert_eq!(source_rounded_millisecond_frame_rate(0.0105)?, (1, 100));
        assert_eq!(source_rounded_millisecond_frame_rate(0.0115)?, (3, 250));
        assert_eq!(source_rounded_millisecond_frame_rate(1_000.0)?, (1_000, 1));
        assert!(source_rounded_millisecond_frame_rate(0.009).is_err());
        assert!(source_rounded_millisecond_frame_rate(f64::NAN).is_err());
        Ok(())
    }
}
