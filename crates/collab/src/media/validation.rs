use std::{
    error::Error,
    fmt,
    fs::File,
    io::{BufReader, Cursor, Read, Seek, SeekFrom, Write},
};

use collaboration_domain::{
    MediaByteSize, MediaContentHash, MediaContentType, PrincipalId, TenantContext,
};
use image::{AnimationDecoder, ImageFormat, ImageReader};
use sha2::{Digest, Sha256};

use super::upload_admission::{MediaUploadAdmission, MediaUploadAdmissionError};

pub const MAX_IMAGE_UPLOAD_BYTES: u64 = 25 * 1024 * 1024;
pub const MAX_VIDEO_UPLOAD_BYTES: u64 = 500 * 1024 * 1024;
pub const MAX_DECODED_IMAGE_PIXELS: u64 = 25_000_000;
pub const MAX_VIDEO_DURATION_MILLIS: u64 = 600_000;
pub const MAX_VIDEO_WIDTH: u32 = 3_840;
pub const MAX_VIDEO_HEIGHT: u32 = 2_160;
const MAX_GIF_FRAMES: usize = 1_000;
const MAX_MP4_TOP_LEVEL_BOXES: usize = 1_024;
const MAX_MP4_BOXES: usize = 100_000;
const MAX_MP4_BOX_DEPTH: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaValidationLimits {
    max_image_bytes: MediaByteSize,
    max_video_bytes: MediaByteSize,
    max_decoded_image_pixels: u64,
}

impl MediaValidationLimits {
    pub fn new(
        max_image_bytes: u64,
        max_video_bytes: u64,
        max_decoded_image_pixels: u64,
    ) -> Result<Self, MediaValidationError> {
        if max_image_bytes > MAX_IMAGE_UPLOAD_BYTES
            || max_video_bytes > MAX_VIDEO_UPLOAD_BYTES
            || max_decoded_image_pixels == 0
            || max_decoded_image_pixels > MAX_DECODED_IMAGE_PIXELS
        {
            return Err(MediaValidationError::InvalidConfiguration);
        }
        Ok(Self {
            max_image_bytes: MediaByteSize::new(max_image_bytes)
                .map_err(|_| MediaValidationError::InvalidConfiguration)?,
            max_video_bytes: MediaByteSize::new(max_video_bytes)
                .map_err(|_| MediaValidationError::InvalidConfiguration)?,
            max_decoded_image_pixels,
        })
    }

    fn max_bytes_for(self, kind: SupportedMediaKind) -> MediaByteSize {
        match kind {
            SupportedMediaKind::Mp4 => self.max_video_bytes,
            SupportedMediaKind::Jpeg
            | SupportedMediaKind::Png
            | SupportedMediaKind::Gif
            | SupportedMediaKind::Webp => self.max_image_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SupportedMediaKind {
    Jpeg,
    Png,
    Gif,
    Webp,
    Mp4,
}

impl SupportedMediaKind {
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
            Self::Mp4 => "video/mp4",
        }
    }

    fn from_declared(content_type: &MediaContentType) -> Result<Self, MediaValidationError> {
        match content_type.as_str() {
            "image/jpeg" => Ok(Self::Jpeg),
            "image/png" => Ok(Self::Png),
            "image/gif" => Ok(Self::Gif),
            "image/webp" => Ok(Self::Webp),
            "video/mp4" => Ok(Self::Mp4),
            _ => Err(MediaValidationError::UnsupportedContentType),
        }
    }

    fn image_format(self) -> Option<ImageFormat> {
        match self {
            Self::Jpeg => Some(ImageFormat::Jpeg),
            Self::Png => Some(ImageFormat::Png),
            Self::Gif => Some(ImageFormat::Gif),
            Self::Webp => Some(ImageFormat::WebP),
            Self::Mp4 => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedMediaProperties {
    pub width: u32,
    pub height: u32,
    pub duration_millis: Option<u64>,
    pub has_audio: bool,
}

pub struct ValidatedMedia {
    admission: MediaUploadAdmission,
    content_type: MediaContentType,
    kind: SupportedMediaKind,
    properties: ValidatedMediaProperties,
    file: File,
}

impl ValidatedMedia {
    pub const fn admission(&self) -> MediaUploadAdmission {
        self.admission
    }

    pub fn content_type(&self) -> &MediaContentType {
        &self.content_type
    }

    pub const fn kind(&self) -> SupportedMediaKind {
        self.kind
    }

    pub const fn properties(&self) -> ValidatedMediaProperties {
        self.properties
    }

    pub fn into_reader(mut self) -> Result<File, MediaValidationError> {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|_| MediaValidationError::Io)?;
        Ok(self.file)
    }
}

impl fmt::Debug for ValidatedMedia {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedMedia")
            .field("community_id", &self.admission.community_id())
            .field("principal_id", &self.admission.principal_id())
            .field("operation_id", &self.admission.operation_id())
            .field("content_type", &self.content_type)
            .field("byte_size", &self.admission.byte_size())
            .field("kind", &self.kind)
            .field("properties", &self.properties)
            .finish_non_exhaustive()
    }
}

pub struct MediaValidationSession {
    admission: MediaUploadAdmission,
    declared_content_type: MediaContentType,
    declared_kind: SupportedMediaKind,
    limits: MediaValidationLimits,
    file: Option<File>,
    hasher: Sha256,
    bytes_received: u64,
    failed: bool,
}

impl MediaValidationSession {
    pub fn begin(
        admission: MediaUploadAdmission,
        tenant: &TenantContext,
        principal_id: PrincipalId,
        declared_content_type: MediaContentType,
        now_millis: u64,
        limits: MediaValidationLimits,
    ) -> Result<Self, MediaValidationError> {
        admission.validate_for_processing(tenant, principal_id, now_millis)?;
        let declared_kind = SupportedMediaKind::from_declared(&declared_content_type)?;
        if admission.byte_size() > limits.max_bytes_for(declared_kind) {
            return Err(MediaValidationError::TooLarge);
        }
        Ok(Self {
            admission,
            declared_content_type,
            declared_kind,
            limits,
            file: Some(tempfile::tempfile().map_err(|_| MediaValidationError::Io)?),
            hasher: Sha256::new(),
            bytes_received: 0,
            failed: false,
        })
    }

    pub fn write_chunk(&mut self, chunk: &[u8]) -> Result<(), MediaValidationError> {
        if self.failed {
            return Err(MediaValidationError::InvalidState);
        }
        let chunk_size = u64::try_from(chunk.len()).map_err(|_| MediaValidationError::TooLarge)?;
        let Some(next_size) = self.bytes_received.checked_add(chunk_size) else {
            self.failed = true;
            return Err(MediaValidationError::TooLarge);
        };
        if next_size > self.admission.byte_size().get()
            || next_size > self.limits.max_bytes_for(self.declared_kind).get()
        {
            self.failed = true;
            return Err(MediaValidationError::TooLarge);
        }
        let Some(file) = self.file.as_mut() else {
            self.failed = true;
            return Err(MediaValidationError::InvalidState);
        };
        if file.write_all(chunk).is_err() {
            self.failed = true;
            return Err(MediaValidationError::Io);
        }
        self.hasher.update(chunk);
        self.bytes_received = next_size;
        Ok(())
    }

    pub fn finish(
        mut self,
        tenant: &TenantContext,
        principal_id: PrincipalId,
        now_millis: u64,
    ) -> Result<ValidatedMedia, MediaValidationError> {
        self.admission
            .validate_for_processing(tenant, principal_id, now_millis)?;
        if self.failed {
            return Err(MediaValidationError::InvalidState);
        }
        if self.bytes_received != self.admission.byte_size().get() {
            return Err(MediaValidationError::Truncated);
        }
        let observed_hash = MediaContentHash::from_digest(self.hasher.finalize().into());
        if observed_hash != self.admission.content_hash() {
            return Err(MediaValidationError::HashMismatch);
        }
        let mut file = self.file.take().ok_or(MediaValidationError::InvalidState)?;
        file.flush().map_err(|_| MediaValidationError::Io)?;
        file.seek(SeekFrom::Start(0))
            .map_err(|_| MediaValidationError::Io)?;
        let byte_count =
            usize::try_from(self.bytes_received).map_err(|_| MediaValidationError::TooLarge)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_count)
            .map_err(|_| MediaValidationError::ResourceExhausted)?;
        file.read_to_end(&mut bytes)
            .map_err(|_| MediaValidationError::Io)?;
        if bytes.len() != byte_count {
            return Err(MediaValidationError::Io);
        }
        let observed_kind = detect_kind(&bytes)?;
        if observed_kind != self.declared_kind {
            return Err(MediaValidationError::ContentTypeMismatch);
        }
        let properties = match observed_kind {
            SupportedMediaKind::Mp4 => validate_mp4(&file, self.bytes_received)?,
            kind => validate_image(&bytes, kind, self.limits.max_decoded_image_pixels)?,
        };
        Ok(ValidatedMedia {
            admission: self.admission,
            content_type: self.declared_content_type,
            kind: observed_kind,
            properties,
            file,
        })
    }
}

impl fmt::Debug for MediaValidationSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaValidationSession")
            .field("community_id", &self.admission.community_id())
            .field("principal_id", &self.admission.principal_id())
            .field("operation_id", &self.admission.operation_id())
            .field("declared_content_type", &self.declared_content_type)
            .field("bytes_received", &self.bytes_received)
            .field("failed", &self.failed)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaValidationError {
    InvalidConfiguration,
    InvalidState,
    UnsupportedContentType,
    ContentTypeMismatch,
    TooLarge,
    Truncated,
    HashMismatch,
    InvalidStructure,
    MetadataForbidden,
    DecodeFailed,
    WrongCodec,
    DurationTooLong,
    ResolutionTooHigh,
    ResourceExhausted,
    Admission(MediaUploadAdmissionError),
    Io,
}

impl fmt::Display for MediaValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "media validation configuration is invalid",
            Self::InvalidState => "media validation session is not usable",
            Self::UnsupportedContentType => "media content type is not supported",
            Self::ContentTypeMismatch => "media content type does not match its bytes",
            Self::TooLarge => "media stream exceeds its admitted limit",
            Self::Truncated => "media stream ended before its admitted size",
            Self::HashMismatch => "media content hash does not match its admission",
            Self::InvalidStructure => "media structure is invalid",
            Self::MetadataForbidden => "media contains a forbidden metadata channel",
            Self::DecodeFailed => "media cannot be decoded safely",
            Self::WrongCodec => "media codec is not supported",
            Self::DurationTooLong => "media duration exceeds its limit",
            Self::ResolutionTooHigh => "media dimensions exceed their limit",
            Self::ResourceExhausted => "media validation resources are unavailable",
            Self::Admission(error) => return error.fmt(formatter),
            Self::Io => "media validation I/O failed",
        })
    }
}

impl Error for MediaValidationError {}

impl From<MediaUploadAdmissionError> for MediaValidationError {
    fn from(error: MediaUploadAdmissionError) -> Self {
        Self::Admission(error)
    }
}

fn detect_kind(bytes: &[u8]) -> Result<SupportedMediaKind, MediaValidationError> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Ok(SupportedMediaKind::Jpeg)
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Ok(SupportedMediaKind::Png)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Ok(SupportedMediaKind::Gif)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Ok(SupportedMediaKind::Webp)
    } else if is_mp4_ftyp(bytes) {
        Ok(SupportedMediaKind::Mp4)
    } else {
        Err(MediaValidationError::UnsupportedContentType)
    }
}

fn is_mp4_ftyp(bytes: &[u8]) -> bool {
    const BRANDS: &[[u8; 4]] = &[
        *b"isom", *b"iso2", *b"iso3", *b"iso4", *b"iso5", *b"iso6", *b"iso7", *b"iso8", *b"iso9",
        *b"mp41", *b"mp42", *b"avc1", *b"dash", *b"M4V ",
    ];
    if bytes.len() < 20 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    let Ok(compact_size) = <[u8; 4]>::try_from(&bytes[..4]) else {
        return false;
    };
    let compact_size = u32::from_be_bytes(compact_size) as usize;
    if compact_size < 20 || compact_size > bytes.len() || !(compact_size - 16).is_multiple_of(4) {
        return false;
    }
    bytes[8..12]
        .try_into()
        .ok()
        .is_some_and(|brand| BRANDS.contains(&brand))
        || bytes[16..compact_size]
            .chunks_exact(4)
            .any(|brand| BRANDS.iter().any(|candidate| brand == candidate))
}

fn validate_image(
    bytes: &[u8],
    kind: SupportedMediaKind,
    max_pixels: u64,
) -> Result<ValidatedMediaProperties, MediaValidationError> {
    validate_image_structure(bytes, kind)?;
    let format = kind
        .image_format()
        .ok_or(MediaValidationError::InvalidStructure)?;
    let (width, height) = ImageReader::with_format(Cursor::new(bytes), format)
        .into_dimensions()
        .map_err(|_| MediaValidationError::DecodeFailed)?;
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(MediaValidationError::ResolutionTooHigh)?;
    if width == 0 || height == 0 || pixels > max_pixels {
        return Err(MediaValidationError::ResolutionTooHigh);
    }
    if kind == SupportedMediaKind::Gif {
        let decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes))
            .map_err(|_| MediaValidationError::DecodeFailed)?;
        let mut frame_count = 0usize;
        for frame in decoder.into_frames() {
            frame.map_err(|_| MediaValidationError::DecodeFailed)?;
            frame_count = frame_count
                .checked_add(1)
                .ok_or(MediaValidationError::ResourceExhausted)?;
            if frame_count > MAX_GIF_FRAMES {
                return Err(MediaValidationError::ResourceExhausted);
            }
        }
        if frame_count == 0 {
            return Err(MediaValidationError::DecodeFailed);
        }
    } else {
        image::load_from_memory_with_format(bytes, format)
            .map_err(|_| MediaValidationError::DecodeFailed)?;
    }
    Ok(ValidatedMediaProperties {
        width,
        height,
        duration_millis: None,
        has_audio: false,
    })
}

fn validate_image_structure(
    bytes: &[u8],
    kind: SupportedMediaKind,
) -> Result<(), MediaValidationError> {
    match kind {
        SupportedMediaKind::Jpeg => validate_jpeg(bytes),
        SupportedMediaKind::Png => validate_png(bytes),
        SupportedMediaKind::Gif => validate_gif(bytes),
        SupportedMediaKind::Webp => validate_webp(bytes),
        SupportedMediaKind::Mp4 => Err(MediaValidationError::InvalidStructure),
    }
}

fn validate_jpeg(bytes: &[u8]) -> Result<(), MediaValidationError> {
    if !bytes.starts_with(&[0xff, 0xd8]) {
        return Err(MediaValidationError::InvalidStructure);
    }
    let mut offset = 2usize;
    let mut in_scan = false;
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            if in_scan {
                offset += 1;
                continue;
            }
            return Err(MediaValidationError::InvalidStructure);
        }
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *bytes
            .get(offset)
            .ok_or(MediaValidationError::InvalidStructure)?;
        offset += 1;
        if in_scan && marker == 0x00 {
            continue;
        }
        if (0xd0..=0xd7).contains(&marker) || marker == 0x01 {
            continue;
        }
        if marker == 0xd9 {
            return (offset == bytes.len())
                .then_some(())
                .ok_or(MediaValidationError::MetadataForbidden);
        }
        if marker == 0xd8 {
            return Err(MediaValidationError::InvalidStructure);
        }
        let length_bytes: [u8; 2] = bytes
            .get(offset..offset + 2)
            .ok_or(MediaValidationError::InvalidStructure)?
            .try_into()
            .map_err(|_| MediaValidationError::InvalidStructure)?;
        let length = u16::from_be_bytes(length_bytes) as usize;
        if length < 2 {
            return Err(MediaValidationError::InvalidStructure);
        }
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
            .ok_or(MediaValidationError::InvalidStructure)?;
        if marker == 0xe0 {
            let payload = &bytes[offset + 2..end];
            let canonical_jfif = payload.len() >= 14
                && &payload[..5] == b"JFIF\0"
                && payload.len() == 14 + 3 * payload[12] as usize * payload[13] as usize;
            if !canonical_jfif {
                return Err(MediaValidationError::MetadataForbidden);
            }
        } else if marker == 0xee {
            let payload = &bytes[offset + 2..end];
            if payload.len() != 12 || &payload[..5] != b"Adobe" {
                return Err(MediaValidationError::MetadataForbidden);
            }
        } else if (0xe1..=0xed).contains(&marker) || marker == 0xef || marker == 0xfe {
            return Err(MediaValidationError::MetadataForbidden);
        }
        offset = end;
        in_scan = marker == 0xda;
    }
    Err(MediaValidationError::InvalidStructure)
}

fn validate_png(bytes: &[u8]) -> Result<(), MediaValidationError> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    if !bytes.starts_with(SIGNATURE) {
        return Err(MediaValidationError::InvalidStructure);
    }
    let mut offset = SIGNATURE.len();
    let mut saw_header = false;
    while offset < bytes.len() {
        let length_bytes: [u8; 4] = bytes
            .get(offset..offset + 4)
            .ok_or(MediaValidationError::InvalidStructure)?
            .try_into()
            .map_err(|_| MediaValidationError::InvalidStructure)?;
        let length = u32::from_be_bytes(length_bytes) as usize;
        let kind: [u8; 4] = bytes
            .get(offset + 4..offset + 8)
            .ok_or(MediaValidationError::InvalidStructure)?
            .try_into()
            .map_err(|_| MediaValidationError::InvalidStructure)?;
        let end = offset
            .checked_add(12)
            .and_then(|value| value.checked_add(length))
            .filter(|end| *end <= bytes.len())
            .ok_or(MediaValidationError::InvalidStructure)?;
        if !saw_header {
            if &kind != b"IHDR" || length != 13 {
                return Err(MediaValidationError::InvalidStructure);
            }
            saw_header = true;
        }
        if matches!(&kind, b"eXIf" | b"zTXt" | b"iTXt" | b"iCCP" | b"tEXt") {
            return Err(MediaValidationError::MetadataForbidden);
        }
        let ancillary = kind[0] & 0x20 != 0;
        let known_rendering = matches!(
            &kind,
            b"cHRM"
                | b"gAMA"
                | b"sBIT"
                | b"sRGB"
                | b"bKGD"
                | b"hIST"
                | b"tRNS"
                | b"sPLT"
                | b"acTL"
                | b"fcTL"
                | b"fdAT"
        );
        if ancillary && !known_rendering {
            return Err(MediaValidationError::MetadataForbidden);
        }
        offset = end;
        if &kind == b"IEND" {
            return (length == 0 && offset == bytes.len())
                .then_some(())
                .ok_or(MediaValidationError::MetadataForbidden);
        }
    }
    Err(MediaValidationError::InvalidStructure)
}

fn validate_webp(bytes: &[u8]) -> Result<(), MediaValidationError> {
    fn validate_frame_payload(payload: &[u8]) -> Result<(), MediaValidationError> {
        const FRAME_HEADER_BYTES: usize = 16;
        if payload.len() < FRAME_HEADER_BYTES {
            return Err(MediaValidationError::InvalidStructure);
        }
        let mut offset = FRAME_HEADER_BYTES;
        let mut saw_alpha = false;
        let mut saw_image = false;
        while offset < payload.len() {
            let kind: [u8; 4] = payload
                .get(offset..offset + 4)
                .ok_or(MediaValidationError::InvalidStructure)?
                .try_into()
                .map_err(|_| MediaValidationError::InvalidStructure)?;
            let length_bytes: [u8; 4] = payload
                .get(offset + 4..offset + 8)
                .ok_or(MediaValidationError::InvalidStructure)?
                .try_into()
                .map_err(|_| MediaValidationError::InvalidStructure)?;
            let length = u32::from_le_bytes(length_bytes) as usize;
            let padded = length
                .checked_add(length & 1)
                .ok_or(MediaValidationError::InvalidStructure)?;
            offset = offset
                .checked_add(8)
                .and_then(|start| start.checked_add(padded))
                .filter(|end| *end <= payload.len())
                .ok_or(MediaValidationError::InvalidStructure)?;
            match &kind {
                b"ALPH" if !saw_alpha && !saw_image => saw_alpha = true,
                b"VP8 " if !saw_image => saw_image = true,
                b"VP8L" if !saw_alpha && !saw_image => saw_image = true,
                b"ALPH" | b"VP8 " | b"VP8L" => {
                    return Err(MediaValidationError::InvalidStructure);
                }
                _ => return Err(MediaValidationError::MetadataForbidden),
            }
        }
        saw_image
            .then_some(())
            .ok_or(MediaValidationError::InvalidStructure)
    }

    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return Err(MediaValidationError::InvalidStructure);
    }
    let declared_bytes: [u8; 4] = bytes[4..8]
        .try_into()
        .map_err(|_| MediaValidationError::InvalidStructure)?;
    let declared = u32::from_le_bytes(declared_bytes) as usize;
    if declared.checked_add(8) != Some(bytes.len()) {
        return Err(MediaValidationError::MetadataForbidden);
    }
    let mut offset = 12usize;
    let mut image_chunks = 0usize;
    while offset < bytes.len() {
        let kind: [u8; 4] = bytes
            .get(offset..offset + 4)
            .ok_or(MediaValidationError::InvalidStructure)?
            .try_into()
            .map_err(|_| MediaValidationError::InvalidStructure)?;
        let length_bytes: [u8; 4] = bytes
            .get(offset + 4..offset + 8)
            .ok_or(MediaValidationError::InvalidStructure)?
            .try_into()
            .map_err(|_| MediaValidationError::InvalidStructure)?;
        let length = u32::from_le_bytes(length_bytes) as usize;
        let payload_start = offset + 8;
        let padded = length
            .checked_add(length & 1)
            .ok_or(MediaValidationError::InvalidStructure)?;
        offset = payload_start
            .checked_add(padded)
            .filter(|end| *end <= bytes.len())
            .ok_or(MediaValidationError::InvalidStructure)?;
        if !matches!(
            &kind,
            b"VP8 " | b"VP8L" | b"VP8X" | b"ALPH" | b"ANIM" | b"ANMF"
        ) {
            return Err(MediaValidationError::MetadataForbidden);
        }
        if matches!(&kind, b"VP8 " | b"VP8L" | b"ANMF") {
            image_chunks = image_chunks
                .checked_add(1)
                .ok_or(MediaValidationError::ResourceExhausted)?;
        }
        if &kind == b"VP8X" {
            let flags = *bytes
                .get(payload_start)
                .ok_or(MediaValidationError::InvalidStructure)?;
            if flags & (0x20 | 0x08 | 0x04) != 0 {
                return Err(MediaValidationError::MetadataForbidden);
            }
        } else if &kind == b"ANMF" {
            validate_frame_payload(&bytes[payload_start..payload_start + length])?;
        }
    }
    if image_chunks == 0 {
        return Err(MediaValidationError::InvalidStructure);
    }
    Ok(())
}

fn validate_gif(bytes: &[u8]) -> Result<(), MediaValidationError> {
    if !(bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) || bytes.len() < 13 {
        return Err(MediaValidationError::InvalidStructure);
    }
    fn skip_sub_blocks(bytes: &[u8], offset: &mut usize) -> Result<(), MediaValidationError> {
        loop {
            let length = *bytes
                .get(*offset)
                .ok_or(MediaValidationError::InvalidStructure)? as usize;
            *offset += 1;
            if length == 0 {
                return Ok(());
            }
            *offset = offset
                .checked_add(length)
                .filter(|end| *end <= bytes.len())
                .ok_or(MediaValidationError::InvalidStructure)?;
        }
    }
    let packed = bytes[10];
    let mut offset = 13usize;
    if packed & 0x80 != 0 {
        let table_length = 3usize << ((packed & 0x07) as usize + 1);
        offset = offset
            .checked_add(table_length)
            .filter(|end| *end <= bytes.len())
            .ok_or(MediaValidationError::InvalidStructure)?;
    }
    loop {
        match *bytes
            .get(offset)
            .ok_or(MediaValidationError::InvalidStructure)?
        {
            0x2c => {
                let image_packed = *bytes
                    .get(offset + 9)
                    .ok_or(MediaValidationError::InvalidStructure)?;
                offset += 10;
                if image_packed & 0x80 != 0 {
                    let table_length = 3usize << ((image_packed & 0x07) as usize + 1);
                    offset = offset
                        .checked_add(table_length)
                        .filter(|end| *end <= bytes.len())
                        .ok_or(MediaValidationError::InvalidStructure)?;
                }
                offset = offset
                    .checked_add(1)
                    .filter(|end| *end <= bytes.len())
                    .ok_or(MediaValidationError::InvalidStructure)?;
                skip_sub_blocks(bytes, &mut offset)?;
            }
            0x21 => {
                let label = *bytes
                    .get(offset + 1)
                    .ok_or(MediaValidationError::InvalidStructure)?;
                offset += 2;
                match label {
                    0xf9 => {
                        if bytes.get(offset) != Some(&4) || bytes.get(offset + 5) != Some(&0) {
                            return Err(MediaValidationError::InvalidStructure);
                        }
                        offset += 6;
                    }
                    0xff => {
                        if bytes.get(offset) != Some(&11) {
                            return Err(MediaValidationError::InvalidStructure);
                        }
                        let application = bytes
                            .get(offset + 1..offset + 12)
                            .ok_or(MediaValidationError::InvalidStructure)?;
                        if application != b"NETSCAPE2.0" && application != b"ANIMEXTS1.0" {
                            return Err(MediaValidationError::MetadataForbidden);
                        }
                        offset += 12;
                        if bytes.get(offset) != Some(&3)
                            || bytes.get(offset + 1) != Some(&1)
                            || bytes.get(offset + 4) != Some(&0)
                        {
                            return Err(MediaValidationError::MetadataForbidden);
                        }
                        offset += 5;
                    }
                    _ => return Err(MediaValidationError::MetadataForbidden),
                }
            }
            0x3b => {
                return (offset + 1 == bytes.len())
                    .then_some(())
                    .ok_or(MediaValidationError::MetadataForbidden);
            }
            _ => return Err(MediaValidationError::InvalidStructure),
        }
    }
}

fn validate_mp4(file: &File, size: u64) -> Result<ValidatedMediaProperties, MediaValidationError> {
    validate_mp4_structure(file, size)?;
    let mut reader = file.try_clone().map_err(|_| MediaValidationError::Io)?;
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|_| MediaValidationError::Io)?;
    let mp4 = mp4::Mp4Reader::read_header(BufReader::new(reader), size)
        .map_err(|_| MediaValidationError::DecodeFailed)?;
    if *mp4.major_brand() == mp4::FourCC::from(*b"qt  ") {
        return Err(MediaValidationError::UnsupportedContentType);
    }
    let mut video = None;
    let mut has_audio = false;
    for track in mp4.tracks().values() {
        match track
            .track_type()
            .map_err(|_| MediaValidationError::DecodeFailed)?
        {
            mp4::TrackType::Video => {
                if video.is_some() {
                    return Err(MediaValidationError::MetadataForbidden);
                }
                if track
                    .media_type()
                    .map_err(|_| MediaValidationError::WrongCodec)?
                    != mp4::MediaType::H264
                {
                    return Err(MediaValidationError::WrongCodec);
                }
                if track.timescale() == 0 {
                    return Err(MediaValidationError::InvalidStructure);
                }
                let duration_millis = u64::try_from(track.duration().as_millis())
                    .map_err(|_| MediaValidationError::DurationTooLong)?;
                if duration_millis == 0 || duration_millis > MAX_VIDEO_DURATION_MILLIS {
                    return Err(MediaValidationError::DurationTooLong);
                }
                let width = track.width() as u32;
                let height = track.height() as u32;
                if width == 0 || height == 0 || width > MAX_VIDEO_WIDTH || height > MAX_VIDEO_HEIGHT
                {
                    return Err(MediaValidationError::ResolutionTooHigh);
                }
                video = Some((width, height, duration_millis));
            }
            mp4::TrackType::Audio => {
                if has_audio {
                    return Err(MediaValidationError::MetadataForbidden);
                }
                if track
                    .media_type()
                    .map_err(|_| MediaValidationError::WrongCodec)?
                    != mp4::MediaType::AAC
                {
                    return Err(MediaValidationError::WrongCodec);
                }
                has_audio = true;
            }
            _ => return Err(MediaValidationError::MetadataForbidden),
        }
    }
    let (width, height, duration_millis) = video.ok_or(MediaValidationError::WrongCodec)?;
    Ok(ValidatedMediaProperties {
        width,
        height,
        duration_millis: Some(duration_millis),
        has_audio,
    })
}

fn validate_mp4_structure(file: &File, file_size: u64) -> Result<(), MediaValidationError> {
    const EMPTY_FFMPEG_USER_DATA: &[u8] = &[
        0, 0, 0, 0x35, b'm', b'e', b't', b'a', 0, 0, 0, 0, 0, 0, 0, 0x21, b'h', b'd', b'l', b'r',
        0, 0, 0, 0, 0, 0, 0, 0, b'm', b'd', b'i', b'r', b'a', b'p', b'p', b'l', 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 8, b'i', b'l', b's', b't',
    ];
    const FORBIDDEN: &[[u8; 4]] = &[
        *b"meta",
        *b"ilst",
        *b"keys",
        *b"data",
        *b"uuid",
        *b"xml ",
        *b"bxml",
        *b"loci",
        *b"\xa9xyz",
        *b"name",
        *b"chap",
    ];
    const CONTAINERS: &[[u8; 4]] = &[
        *b"moov", *b"trak", *b"mdia", *b"minf", *b"stbl", *b"edts", *b"dinf", *b"sinf", *b"schi",
    ];
    const ALLOWED: &[[u8; 4]] = &[
        *b"ftyp", *b"moov", *b"mdat", *b"free", *b"skip", *b"wide", *b"trak", *b"mdia", *b"minf",
        *b"stbl", *b"edts", *b"dinf", *b"sinf", *b"schi", *b"udta", *b"mvhd", *b"tkhd", *b"mdhd",
        *b"hdlr", *b"vmhd", *b"smhd", *b"dref", *b"url ", *b"urn ", *b"stsd", *b"stts", *b"stss",
        *b"ctts", *b"stsc", *b"stsz", *b"stco", *b"co64", *b"sgpd", *b"sbgp", *b"sdtp", *b"elst",
    ];

    fn walk(
        file: &mut File,
        start: u64,
        end: u64,
        count: &mut usize,
        depth: usize,
    ) -> Result<(), MediaValidationError> {
        if depth > MAX_MP4_BOX_DEPTH {
            return Err(MediaValidationError::InvalidStructure);
        }
        let mut offset = start;
        while offset < end {
            *count = count
                .checked_add(1)
                .ok_or(MediaValidationError::InvalidStructure)?;
            if *count > MAX_MP4_BOXES || end - offset < 8 {
                return Err(MediaValidationError::InvalidStructure);
            }
            file.seek(SeekFrom::Start(offset))
                .map_err(|_| MediaValidationError::Io)?;
            let mut header = [0u8; 8];
            file.read_exact(&mut header)
                .map_err(|_| MediaValidationError::InvalidStructure)?;
            let compact = u32::from_be_bytes(
                header[..4]
                    .try_into()
                    .map_err(|_| MediaValidationError::InvalidStructure)?,
            ) as u64;
            let kind: [u8; 4] = header[4..]
                .try_into()
                .map_err(|_| MediaValidationError::InvalidStructure)?;
            let (size, header_size) = if compact == 1 {
                let mut extended = [0u8; 8];
                file.read_exact(&mut extended)
                    .map_err(|_| MediaValidationError::InvalidStructure)?;
                (u64::from_be_bytes(extended), 16)
            } else if compact == 0 {
                (end - offset, 8)
            } else {
                (compact, 8)
            };
            let box_end = offset
                .checked_add(size)
                .filter(|box_end| size >= header_size && *box_end <= end)
                .ok_or(MediaValidationError::InvalidStructure)?;
            if FORBIDDEN.contains(&kind) || !ALLOWED.contains(&kind) {
                return Err(MediaValidationError::MetadataForbidden);
            }
            if kind == *b"udta" {
                if size != header_size + EMPTY_FFMPEG_USER_DATA.len() as u64 {
                    return Err(MediaValidationError::MetadataForbidden);
                }
                let mut payload = vec![0; EMPTY_FFMPEG_USER_DATA.len()];
                file.read_exact(&mut payload)
                    .map_err(|_| MediaValidationError::InvalidStructure)?;
                if payload != EMPTY_FFMPEG_USER_DATA {
                    return Err(MediaValidationError::MetadataForbidden);
                }
            } else if CONTAINERS.contains(&kind) {
                let child_start = if kind == *b"dref" {
                    (offset + header_size)
                        .checked_add(8)
                        .filter(|child_start| *child_start <= box_end)
                        .ok_or(MediaValidationError::InvalidStructure)?
                } else {
                    offset + header_size
                };
                walk(file, child_start, box_end, count, depth + 1)?;
            }
            offset = box_end;
        }
        Ok(())
    }

    let mut reader = file.try_clone().map_err(|_| MediaValidationError::Io)?;
    let mut offset = 0u64;
    let mut top_level_boxes = 0usize;
    let mut saw_ftyp = false;
    let mut saw_moov = false;
    let mut saw_mdat = false;
    while offset < file_size {
        top_level_boxes += 1;
        if top_level_boxes > MAX_MP4_TOP_LEVEL_BOXES || file_size - offset < 8 {
            return Err(MediaValidationError::InvalidStructure);
        }
        reader
            .seek(SeekFrom::Start(offset))
            .map_err(|_| MediaValidationError::Io)?;
        let mut header = [0u8; 8];
        reader
            .read_exact(&mut header)
            .map_err(|_| MediaValidationError::InvalidStructure)?;
        let compact = u32::from_be_bytes(
            header[..4]
                .try_into()
                .map_err(|_| MediaValidationError::InvalidStructure)?,
        ) as u64;
        let kind = &header[4..];
        let (size, header_size) = if compact == 1 {
            let mut extended = [0u8; 8];
            reader
                .read_exact(&mut extended)
                .map_err(|_| MediaValidationError::InvalidStructure)?;
            (u64::from_be_bytes(extended), 16)
        } else if compact == 0 {
            (file_size - offset, 8)
        } else {
            (compact, 8)
        };
        let box_end = offset
            .checked_add(size)
            .filter(|box_end| size >= header_size && *box_end <= file_size)
            .ok_or(MediaValidationError::InvalidStructure)?;
        match kind {
            b"ftyp" if offset == 0 && !saw_ftyp => saw_ftyp = true,
            b"moov" if saw_ftyp && !saw_moov && !saw_mdat => saw_moov = true,
            b"mdat" if saw_moov && !saw_mdat => saw_mdat = true,
            b"ftyp" | b"moov" | b"mdat" => return Err(MediaValidationError::InvalidStructure),
            _ => {}
        }
        offset = box_end;
    }
    if !saw_ftyp || !saw_moov || !saw_mdat {
        return Err(MediaValidationError::InvalidStructure);
    }
    let mut count = 0usize;
    walk(&mut reader, 0, file_size, &mut count, 0)
}
