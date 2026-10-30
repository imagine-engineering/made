use collaboration_domain::{
    MediaByteSize, MediaContentHash, MediaContentType, MediaDescriptor, MediaMetadata,
    MediaMetadataError, MediaObjectSelection, MediaTenantPath, MediaVariant, MediaVariantKind,
    TenantContext,
};
use image::{
    DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder, ImageFormat, ImageReader, Limits,
    codecs::jpeg::JpegEncoder,
};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt, io, io::Cursor, io::Write};
use url::Url;

pub const DEFAULT_THUMBNAIL_DIMENSION: u32 = 320;
pub const DEFAULT_MAX_THUMBNAIL_INPUT_BYTES: usize = 25 * 1024 * 1024;
pub const DEFAULT_MAX_THUMBNAIL_PIXELS: u64 = 25_000_000;
pub const DEFAULT_MAX_THUMBNAIL_OUTPUT_BYTES: usize = 1024 * 1024;
pub const MAX_ACCESSIBILITY_LABEL_BYTES: usize = 512;
pub const MAX_LINK_BYTES: usize = 2_048;
pub const MAX_LINK_TITLE_BYTES: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborationThumbnailLimits {
    pub max_input_bytes: usize,
    pub max_input_pixels: u64,
    pub max_output_dimension: u32,
    pub max_output_bytes: usize,
    pub jpeg_quality: u8,
}

impl Default for CollaborationThumbnailLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_THUMBNAIL_INPUT_BYTES,
            max_input_pixels: DEFAULT_MAX_THUMBNAIL_PIXELS,
            max_output_dimension: DEFAULT_THUMBNAIL_DIMENSION,
            max_output_bytes: DEFAULT_MAX_THUMBNAIL_OUTPUT_BYTES,
            jpeg_quality: 82,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedCollaborationThumbnail {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    variant: MediaVariant,
}

impl GeneratedCollaborationThumbnail {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn variant(&self) -> &MediaVariant {
        &self.variant
    }

    pub fn into_parts(self) -> (Vec<u8>, MediaVariant) {
        (self.bytes, self.variant)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeImageFormat {
    Png,
    Jpeg,
    Webp,
    Gif,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AccessibilityLabel(String);

impl AccessibilityLabel {
    pub fn new(value: impl Into<String>) -> Result<Self, CollaborationMediaError> {
        let value = value.into();
        let value = value.trim();
        if value.is_empty()
            || value.len() > MAX_ACCESSIBILITY_LABEL_BYTES
            || contains_unsafe_text(value)
        {
            return Err(CollaborationMediaError::InvalidAccessibilityLabel);
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaRenderSource {
    path: MediaTenantPath,
    descriptor: MediaDescriptor,
}

impl MediaRenderSource {
    pub const fn path(&self) -> MediaTenantPath {
        self.path
    }

    pub const fn descriptor(&self) -> &MediaDescriptor {
        &self.descriptor
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariantFallback {
    NotNeeded,
    MissingOrUnsupported(MediaVariantKind),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeAttachmentPresentation {
    Image {
        source: MediaRenderSource,
        format: NativeImageFormat,
        accessibility_label: AccessibilityLabel,
        variant_fallback: VariantFallback,
        animated_playback_requires_user_action: bool,
    },
    Video {
        source: MediaRenderSource,
        poster: Option<MediaRenderSource>,
        accessibility_label: AccessibilityLabel,
        poster_fallback: VariantFallback,
        autoplay: bool,
    },
    Audio {
        source: MediaRenderSource,
        accessibility_label: AccessibilityLabel,
        autoplay: bool,
    },
    InertDownload {
        source: MediaRenderSource,
        accessibility_label: AccessibilityLabel,
    },
    Link(SafeLinkPresentation),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafeLinkPresentation {
    destination: Url,
    visible_origin: String,
    title: Option<String>,
    accessibility_label: AccessibilityLabel,
    automatic_navigation: bool,
}

impl SafeLinkPresentation {
    pub fn destination(&self) -> &Url {
        &self.destination
    }

    pub fn visible_origin(&self) -> &str {
        &self.visible_origin
    }

    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    pub const fn accessibility_label(&self) -> &AccessibilityLabel {
        &self.accessibility_label
    }

    pub const fn automatic_navigation(&self) -> bool {
        self.automatic_navigation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborationMediaError {
    InvalidLimits,
    UnsupportedThumbnailType,
    InputTooLarge,
    InputSizeMismatch,
    ContentHashMismatch,
    DimensionsExceeded,
    OutputTooLarge,
    Codec(String),
    InvalidAccessibilityLabel,
    InvalidLink,
    Metadata(MediaMetadataError),
}

impl fmt::Display for CollaborationMediaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("collaboration media limits are invalid"),
            Self::UnsupportedThumbnailType => {
                formatter.write_str("attachment type does not support thumbnails")
            }
            Self::InputTooLarge => formatter.write_str("thumbnail input exceeds its byte limit"),
            Self::InputSizeMismatch => {
                formatter.write_str("thumbnail input size differs from canonical metadata")
            }
            Self::ContentHashMismatch => {
                formatter.write_str("thumbnail input hash differs from canonical metadata")
            }
            Self::DimensionsExceeded => {
                formatter.write_str("thumbnail input dimensions exceed configured limits")
            }
            Self::OutputTooLarge => formatter.write_str("thumbnail output exceeds its byte limit"),
            Self::Codec(error) => write!(formatter, "thumbnail codec failed: {error}"),
            Self::InvalidAccessibilityLabel => {
                formatter.write_str("attachment accessibility label is invalid")
            }
            Self::InvalidLink => formatter.write_str("attachment link is invalid"),
            Self::Metadata(error) => write!(formatter, "attachment metadata is invalid: {error}"),
        }
    }
}

impl Error for CollaborationMediaError {}

impl From<MediaMetadataError> for CollaborationMediaError {
    fn from(error: MediaMetadataError) -> Self {
        Self::Metadata(error)
    }
}

pub fn generate_collaboration_thumbnail(
    metadata: &MediaMetadata,
    tenant: &TenantContext,
    original_bytes: &[u8],
    limits: CollaborationThumbnailLimits,
) -> Result<GeneratedCollaborationThumbnail, CollaborationMediaError> {
    metadata.tenant_path(tenant, MediaObjectSelection::Original)?;
    validate_thumbnail_limits(limits)?;

    if original_bytes.len() > limits.max_input_bytes {
        return Err(CollaborationMediaError::InputTooLarge);
    }
    let expected_size = usize::try_from(metadata.fields().byte_size.get())
        .map_err(|_| CollaborationMediaError::InputSizeMismatch)?;
    if original_bytes.len() != expected_size {
        return Err(CollaborationMediaError::InputSizeMismatch);
    }
    let observed_hash: [u8; 32] = Sha256::digest(original_bytes).into();
    if MediaContentHash::from_digest(observed_hash) != metadata.fields().identity.content_hash() {
        return Err(CollaborationMediaError::ContentHashMismatch);
    }

    let image_format = thumbnail_image_format(metadata.fields().content_type.as_str())
        .ok_or(CollaborationMediaError::UnsupportedThumbnailType)?;
    let (width, height) = ImageReader::with_format(Cursor::new(original_bytes), image_format)
        .into_dimensions()
        .map_err(codec_error)?;
    validate_input_dimensions(width, height, limits)?;

    let mut reader = ImageReader::with_format(Cursor::new(original_bytes), image_format);
    let mut decoder_limits = Limits::default();
    decoder_limits.max_image_width = Some(width);
    decoder_limits.max_image_height = Some(height);
    decoder_limits.max_alloc = Some(
        limits
            .max_input_pixels
            .checked_mul(4)
            .ok_or(CollaborationMediaError::InvalidLimits)?,
    );
    reader.limits(decoder_limits);
    let mut decoder = reader.into_decoder().map_err(codec_error)?;
    let orientation = decoder.orientation().map_err(codec_error)?;
    let mut decoded = DynamicImage::from_decoder(decoder).map_err(codec_error)?;
    decoded.apply_orientation(orientation);
    let thumbnail = decoded.thumbnail(limits.max_output_dimension, limits.max_output_dimension);
    let thumbnail = thumbnail.into_rgb8();
    let output_width = thumbnail.width();
    let output_height = thumbnail.height();

    let mut writer = BoundedWriter::new(limits.max_output_bytes);
    let encode_result = JpegEncoder::new_with_quality(&mut writer, limits.jpeg_quality)
        .write_image(
            thumbnail.as_raw(),
            output_width,
            output_height,
            ExtendedColorType::Rgb8,
        );
    if let Err(error) = encode_result {
        if writer.exceeded() {
            return Err(CollaborationMediaError::OutputTooLarge);
        }
        return Err(codec_error(error));
    }
    let bytes = writer.into_inner();
    let byte_size =
        u64::try_from(bytes.len()).map_err(|_| CollaborationMediaError::OutputTooLarge)?;
    let descriptor = MediaDescriptor::new(
        MediaContentHash::from_digest(Sha256::digest(&bytes).into()),
        MediaContentType::new("image/jpeg")?,
        MediaByteSize::new(byte_size)?,
    );

    Ok(GeneratedCollaborationThumbnail {
        bytes,
        width: output_width,
        height: output_height,
        variant: MediaVariant::new(MediaVariantKind::Thumbnail, descriptor),
    })
}

pub fn plan_native_media_attachment(
    metadata: &MediaMetadata,
    tenant: &TenantContext,
    accessibility_label: Option<&str>,
) -> Result<NativeAttachmentPresentation, CollaborationMediaError> {
    let original = original_source(metadata, tenant)?;
    let content_type = metadata.fields().content_type.as_str();

    if native_image_format(content_type).is_some() {
        let (source, variant_fallback) =
            preferred_image_source(metadata, tenant, MediaVariantKind::Thumbnail, original)?;
        let format = native_image_format(source.descriptor.content_type().as_str())
            .ok_or(CollaborationMediaError::UnsupportedThumbnailType)?;
        return Ok(NativeAttachmentPresentation::Image {
            source,
            format,
            accessibility_label: resolved_label(accessibility_label, "Image attachment")?,
            variant_fallback,
            animated_playback_requires_user_action: format == NativeImageFormat::Gif,
        });
    }

    if content_type == "video/mp4" {
        let (poster, poster_fallback) =
            optional_image_variant(metadata, tenant, MediaVariantKind::Poster)?;
        return Ok(NativeAttachmentPresentation::Video {
            source: original,
            poster,
            accessibility_label: resolved_label(accessibility_label, "Video attachment")?,
            poster_fallback,
            autoplay: false,
        });
    }

    if matches!(
        content_type,
        "audio/aac" | "audio/flac" | "audio/mp4" | "audio/mpeg" | "audio/ogg" | "audio/wav"
    ) {
        return Ok(NativeAttachmentPresentation::Audio {
            source: original,
            accessibility_label: resolved_label(accessibility_label, "Audio attachment")?,
            autoplay: false,
        });
    }

    Ok(NativeAttachmentPresentation::InertDownload {
        source: original,
        accessibility_label: resolved_label(accessibility_label, "File attachment")?,
    })
}

pub fn plan_safe_link_attachment(
    destination: &str,
    title: Option<&str>,
    accessibility_label: Option<&str>,
) -> Result<NativeAttachmentPresentation, CollaborationMediaError> {
    if destination.is_empty()
        || destination.len() > MAX_LINK_BYTES
        || contains_unsafe_text(destination)
    {
        return Err(CollaborationMediaError::InvalidLink);
    }
    let destination = Url::parse(destination).map_err(|_| CollaborationMediaError::InvalidLink)?;
    if destination.scheme() != "https"
        || destination.host_str().is_none()
        || !destination.username().is_empty()
        || destination.password().is_some()
    {
        return Err(CollaborationMediaError::InvalidLink);
    }
    let title = title
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(|title| {
            if title.len() > MAX_LINK_TITLE_BYTES || contains_unsafe_text(title) {
                Err(CollaborationMediaError::InvalidLink)
            } else {
                Ok(title.to_owned())
            }
        })
        .transpose()?;
    let visible_origin = destination.origin().ascii_serialization();

    Ok(NativeAttachmentPresentation::Link(SafeLinkPresentation {
        destination,
        visible_origin,
        title,
        accessibility_label: resolved_label(accessibility_label, "External link")?,
        automatic_navigation: false,
    }))
}

fn original_source(
    metadata: &MediaMetadata,
    tenant: &TenantContext,
) -> Result<MediaRenderSource, CollaborationMediaError> {
    let fields = metadata.fields();
    Ok(MediaRenderSource {
        path: metadata.tenant_path(tenant, MediaObjectSelection::Original)?,
        descriptor: MediaDescriptor::new(
            fields.identity.content_hash(),
            fields.content_type.clone(),
            fields.byte_size,
        ),
    })
}

fn preferred_image_source(
    metadata: &MediaMetadata,
    tenant: &TenantContext,
    kind: MediaVariantKind,
    original: MediaRenderSource,
) -> Result<(MediaRenderSource, VariantFallback), CollaborationMediaError> {
    let (variant, fallback) = optional_image_variant(metadata, tenant, kind)?;
    Ok(match variant {
        Some(variant) => (variant, fallback),
        None => (original, fallback),
    })
}

fn optional_image_variant(
    metadata: &MediaMetadata,
    tenant: &TenantContext,
    kind: MediaVariantKind,
) -> Result<(Option<MediaRenderSource>, VariantFallback), CollaborationMediaError> {
    let Some(variant) = metadata
        .fields()
        .variants
        .iter()
        .find(|variant| variant.kind() == kind)
    else {
        return Ok((None, VariantFallback::MissingOrUnsupported(kind)));
    };
    if native_image_format(variant.descriptor().content_type().as_str()).is_none() {
        return Ok((None, VariantFallback::MissingOrUnsupported(kind)));
    }
    Ok((
        Some(MediaRenderSource {
            path: metadata.tenant_path(tenant, MediaObjectSelection::Variant(kind))?,
            descriptor: variant.descriptor().clone(),
        }),
        VariantFallback::NotNeeded,
    ))
}

fn resolved_label(
    value: Option<&str>,
    default: &'static str,
) -> Result<AccessibilityLabel, CollaborationMediaError> {
    AccessibilityLabel::new(value.unwrap_or(default))
}

fn native_image_format(content_type: &str) -> Option<NativeImageFormat> {
    match content_type {
        "image/png" => Some(NativeImageFormat::Png),
        "image/jpeg" => Some(NativeImageFormat::Jpeg),
        "image/webp" => Some(NativeImageFormat::Webp),
        "image/gif" => Some(NativeImageFormat::Gif),
        _ => None,
    }
}

fn thumbnail_image_format(content_type: &str) -> Option<ImageFormat> {
    match content_type {
        "image/png" => Some(ImageFormat::Png),
        "image/jpeg" => Some(ImageFormat::Jpeg),
        "image/webp" => Some(ImageFormat::WebP),
        "image/gif" => Some(ImageFormat::Gif),
        _ => None,
    }
}

fn validate_thumbnail_limits(
    limits: CollaborationThumbnailLimits,
) -> Result<(), CollaborationMediaError> {
    if limits.max_input_bytes == 0
        || limits.max_input_pixels == 0
        || limits.max_output_dimension == 0
        || limits.max_output_bytes == 0
        || !(1..=100).contains(&limits.jpeg_quality)
    {
        return Err(CollaborationMediaError::InvalidLimits);
    }
    Ok(())
}

fn validate_input_dimensions(
    width: u32,
    height: u32,
    limits: CollaborationThumbnailLimits,
) -> Result<(), CollaborationMediaError> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(CollaborationMediaError::DimensionsExceeded)?;
    if width == 0 || height == 0 || pixels > limits.max_input_pixels {
        return Err(CollaborationMediaError::DimensionsExceeded);
    }
    Ok(())
}

fn contains_unsafe_text(value: &str) -> bool {
    value.chars().any(|character| {
        character.is_control()
            || matches!(
                character,
                '\u{202a}'
                    | '\u{202b}'
                    | '\u{202c}'
                    | '\u{202d}'
                    | '\u{202e}'
                    | '\u{2066}'
                    | '\u{2067}'
                    | '\u{2068}'
                    | '\u{2069}'
            )
    })
}

fn codec_error(error: impl fmt::Display) -> CollaborationMediaError {
    CollaborationMediaError::Codec(error.to_string())
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl BoundedWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
        }
    }

    const fn exceeded(&self) -> bool {
        self.exceeded
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for BoundedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let remaining = self.limit.saturating_sub(self.bytes.len());
        if buffer.len() > remaining {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "thumbnail output limit exceeded",
            ));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use collaboration_domain::{CommunityId, MediaIdentity, PrincipalId, TrustedTenantRoute};

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_direct_host(community_id, "media.example")
                    .expect("valid route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn metadata(content_type: &str, bytes: &[u8]) -> MediaMetadata {
        let identity = MediaIdentity::new(
            CommunityId::new(),
            MediaContentHash::from_digest(Sha256::digest(bytes).into()),
        )
        .expect("media identity");
        MediaMetadata::new(
            identity,
            PrincipalId::new(),
            MediaContentType::new(content_type).expect("content type"),
            MediaByteSize::new(u64::try_from(bytes.len()).expect("size conversion"))
                .expect("positive size"),
            1,
        )
        .expect("metadata")
    }

    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let pixel_count = usize::try_from(width)
            .expect("width")
            .checked_mul(usize::try_from(height).expect("height"))
            .expect("pixel count");
        let pixels = vec![127; pixel_count.checked_mul(3).expect("rgb bytes")];
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(&pixels, width, height, ExtendedColorType::Rgb8)
            .expect("encode png");
        bytes
    }

    #[test]
    fn thumbnail_generation_is_bounded_and_failure_publishes_nothing() {
        let bytes = png_bytes(640, 480);
        let metadata = metadata("image/png", &bytes);
        let tenant = tenant(metadata.fields().identity.community_id());

        let generated = generate_collaboration_thumbnail(
            &metadata,
            &tenant,
            &bytes,
            CollaborationThumbnailLimits::default(),
        )
        .expect("thumbnail");
        assert!(generated.width() <= DEFAULT_THUMBNAIL_DIMENSION);
        assert!(generated.height() <= DEFAULT_THUMBNAIL_DIMENSION);
        assert!(generated.bytes().len() <= DEFAULT_MAX_THUMBNAIL_OUTPUT_BYTES);
        assert_eq!(generated.variant().kind(), MediaVariantKind::Thumbnail);
        assert_eq!(
            generated.variant().descriptor().content_type().as_str(),
            "image/jpeg"
        );
        assert!(metadata.fields().variants.is_empty());
        let mut metadata_with_thumbnail = metadata.clone();
        metadata_with_thumbnail
            .add_variant(generated.variant().clone())
            .expect("attach stored thumbnail descriptor");
        let presentation = plan_native_media_attachment(
            &metadata_with_thumbnail,
            &tenant,
            Some("Generated thumbnail"),
        )
        .expect("thumbnail presentation");
        let NativeAttachmentPresentation::Image {
            source,
            variant_fallback,
            ..
        } = presentation
        else {
            panic!("expected image presentation");
        };
        assert_eq!(
            source.path().selection(),
            MediaObjectSelection::Variant(MediaVariantKind::Thumbnail)
        );
        assert_eq!(variant_fallback, VariantFallback::NotNeeded);

        let error = generate_collaboration_thumbnail(
            &metadata,
            &tenant,
            &bytes,
            CollaborationThumbnailLimits {
                max_output_bytes: 8,
                ..CollaborationThumbnailLimits::default()
            },
        )
        .expect_err("bounded output must fail");
        assert_eq!(error, CollaborationMediaError::OutputTooLarge);
        assert!(metadata.fields().variants.is_empty());
    }

    #[test]
    fn unsupported_active_media_is_an_inert_download() {
        let bytes = b"<svg onload='alert(1)'/>";
        let metadata = metadata("image/svg+xml", bytes);
        let tenant = tenant(metadata.fields().identity.community_id());

        let presentation =
            plan_native_media_attachment(&metadata, &tenant, None).expect("safe presentation");
        assert!(matches!(
            presentation,
            NativeAttachmentPresentation::InertDownload { .. }
        ));
    }

    #[test]
    fn renderer_preserves_accessibility_label_and_rejects_deceptive_text() {
        let bytes = png_bytes(2, 2);
        let metadata = metadata("image/png", &bytes);
        let tenant = tenant(metadata.fields().identity.community_id());

        let presentation = plan_native_media_attachment(
            &metadata,
            &tenant,
            Some("Diagram of the deployment topology"),
        )
        .expect("presentation");
        let NativeAttachmentPresentation::Image {
            accessibility_label,
            ..
        } = presentation
        else {
            panic!("expected image presentation");
        };
        assert_eq!(
            accessibility_label.as_str(),
            "Diagram of the deployment topology"
        );
        assert_eq!(
            plan_native_media_attachment(&metadata, &tenant, Some("safe\u{202e}gpj.exe")),
            Err(CollaborationMediaError::InvalidAccessibilityLabel)
        );
    }

    #[test]
    fn missing_thumbnail_falls_back_to_the_original() {
        let bytes = png_bytes(2, 2);
        let metadata = metadata("image/png", &bytes);
        let tenant = tenant(metadata.fields().identity.community_id());

        let presentation =
            plan_native_media_attachment(&metadata, &tenant, None).expect("presentation");
        let NativeAttachmentPresentation::Image {
            source,
            variant_fallback,
            ..
        } = presentation
        else {
            panic!("expected image presentation");
        };
        assert_eq!(source.path().selection(), MediaObjectSelection::Original);
        assert_eq!(
            variant_fallback,
            VariantFallback::MissingOrUnsupported(MediaVariantKind::Thumbnail)
        );
    }

    #[test]
    fn native_audio_and_video_are_user_controlled() {
        let video_bytes = b"validated mp4";
        let video = metadata("video/mp4", video_bytes);
        let video_tenant = tenant(video.fields().identity.community_id());
        let video_presentation =
            plan_native_media_attachment(&video, &video_tenant, None).expect("video plan");
        let NativeAttachmentPresentation::Video {
            poster,
            poster_fallback,
            autoplay,
            ..
        } = video_presentation
        else {
            panic!("expected video presentation");
        };
        assert!(poster.is_none());
        assert_eq!(
            poster_fallback,
            VariantFallback::MissingOrUnsupported(MediaVariantKind::Poster)
        );
        assert!(!autoplay);

        let audio_bytes = b"validated audio";
        let audio = metadata("audio/mpeg", audio_bytes);
        let audio_tenant = tenant(audio.fields().identity.community_id());
        let audio_presentation =
            plan_native_media_attachment(&audio, &audio_tenant, None).expect("audio plan");
        let NativeAttachmentPresentation::Audio { autoplay, .. } = audio_presentation else {
            panic!("expected audio presentation");
        };
        assert!(!autoplay);
    }

    #[test]
    fn link_cards_are_inert_and_show_the_final_origin() {
        let presentation = plan_safe_link_attachment(
            "https://example.com/path?item=1",
            Some("Example"),
            Some("Open Example link"),
        )
        .expect("link presentation");
        let NativeAttachmentPresentation::Link(link) = presentation else {
            panic!("expected link presentation");
        };
        assert_eq!(link.visible_origin(), "https://example.com");
        assert!(!link.automatic_navigation());
        assert_eq!(link.accessibility_label().as_str(), "Open Example link");

        assert_eq!(
            plan_safe_link_attachment("file:///tmp/payload.html", None, None),
            Err(CollaborationMediaError::InvalidLink)
        );
        assert_eq!(
            plan_safe_link_attachment("https://user:secret@example.com", None, None),
            Err(CollaborationMediaError::InvalidLink)
        );
    }
}
