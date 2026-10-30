use std::{
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use graphite_core_types::{
    Color,
    color::SRGBA8,
    list::{ATTR_TRANSFORM, Item, ItemAttributeValues},
    misc::Clampable as _,
};
use graphite_glam::DAffine2;
use graphite_raster_types::{CPU, Image, Raster};
use sha2::{Digest, Sha256};

pub const GRAPHITE_IMAGE_SOURCE_REVISION: &str = "28ed22824fae8828756f6f4163db62f3585f21e0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageResourceIdentity {
    pub asset_id: Arc<str>,
    pub version_id: Arc<str>,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuSampleFormat {
    Rgba8,
    Rgba32Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageColorSpace {
    Srgb,
    LinearSrgb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaAssociation {
    Straight,
    Premultiplied,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuImageDescriptor {
    pub width: u32,
    pub height: u32,
    pub row_stride: usize,
    pub sample_format: CpuSampleFormat,
    pub color_space: ImageColorSpace,
    pub alpha: AlphaAssociation,
    pub transform: [f64; 6],
}

#[derive(Clone, Copy, Debug)]
pub struct CpuImageBudget {
    pub max_input_bytes: usize,
    pub max_decoded_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuImageOrigin {
    Rgba8Rows {
        bytes: usize,
    },
    EncodedPng {
        bytes: usize,
        bit_depth: u8,
        channels: u8,
    },
}

impl CpuImageOrigin {
    pub fn digest_representation(self) -> &'static str {
        match self {
            Self::Rgba8Rows { .. } => "cpu-rgba8-rows-sha256",
            Self::EncodedPng { .. } => "encoded-png-bytes-sha256",
        }
    }

    pub fn byte_length(self) -> usize {
        match self {
            Self::Rgba8Rows { bytes } | Self::EncodedPng { bytes, .. } => bytes,
        }
    }

    pub fn decode_profile(self) -> Option<&'static str> {
        match self {
            Self::Rgba8Rows { .. } => None,
            Self::EncodedPng { .. } => Some("graphite-static-png-rgb-rgba8-rgb-rgba16-srgb-v1"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdmittedCpuImage {
    identity: ImageResourceIdentity,
    origin: CpuImageOrigin,
    descriptor: CpuImageDescriptor,
    image: Arc<Image<Color>>,
}

impl AdmittedCpuImage {
    pub fn identity(&self) -> &ImageResourceIdentity {
        &self.identity
    }

    pub fn origin(&self) -> CpuImageOrigin {
        self.origin
    }

    pub fn descriptor(&self) -> &CpuImageDescriptor {
        &self.descriptor
    }

    pub fn image(&self) -> &Image<Color> {
        &self.image
    }

    pub fn srgba8_pixels(&self) -> impl Iterator<Item = [u8; 4]> + '_ {
        self.image.data.iter().copied().map(|color| {
            let pixel = SRGBA8::from(color);
            pixel.into()
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageAdmissionError {
    MissingIdentity,
    UnsupportedSamples,
    UnsupportedColorSpace,
    UnsupportedAlpha,
    InvalidTransform,
    EmptyDimensions,
    SizeOverflow,
    InvalidStride,
    InvalidLength { expected: usize, actual: usize },
    InputBudgetExceeded,
    DecodedBudgetExceeded,
    DigestMismatch,
    AllocationFailed,
}

impl fmt::Display for ImageAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingIdentity => {
                formatter.write_str("The asset and immutable version identities are required")
            }
            Self::UnsupportedSamples => {
                formatter.write_str("This image adapter requires CPU RGBA8 samples")
            }
            Self::UnsupportedColorSpace => {
                formatter.write_str("This image adapter requires declared sRGB color")
            }
            Self::UnsupportedAlpha => {
                formatter.write_str("This image adapter requires straight alpha")
            }
            Self::InvalidTransform => {
                formatter.write_str("The image transform must contain finite components")
            }
            Self::EmptyDimensions => formatter.write_str("The image dimensions must be nonzero"),
            Self::SizeOverflow => {
                formatter.write_str("The image layout exceeds the addressable size")
            }
            Self::InvalidStride => {
                formatter.write_str("The image row stride cannot contain its declared pixels")
            }
            Self::InvalidLength { expected, actual } => write!(
                formatter,
                "The image has {actual} bytes; its declared layout requires {expected}"
            ),
            Self::InputBudgetExceeded => {
                formatter.write_str("The image exceeds the granted input byte budget")
            }
            Self::DecodedBudgetExceeded => {
                formatter.write_str("The image exceeds the granted decoded work budget")
            }
            Self::DigestMismatch => {
                formatter.write_str("The image bytes do not match the immutable content digest")
            }
            Self::AllocationFailed => {
                formatter.write_str("Memory could not be reserved for the image")
            }
        }
    }
}

impl std::error::Error for ImageAdmissionError {}

pub fn admit_cpu_image(
    identity: ImageResourceIdentity,
    descriptor: CpuImageDescriptor,
    bytes: Arc<[u8]>,
    budget: CpuImageBudget,
) -> Result<AdmittedCpuImage, ImageAdmissionError> {
    if identity.asset_id.trim().is_empty() || identity.version_id.trim().is_empty() {
        return Err(ImageAdmissionError::MissingIdentity);
    }
    if descriptor.sample_format != CpuSampleFormat::Rgba8 {
        return Err(ImageAdmissionError::UnsupportedSamples);
    }
    if descriptor.color_space != ImageColorSpace::Srgb {
        return Err(ImageAdmissionError::UnsupportedColorSpace);
    }
    if descriptor.alpha != AlphaAssociation::Straight {
        return Err(ImageAdmissionError::UnsupportedAlpha);
    }
    if descriptor
        .transform
        .iter()
        .any(|component| !component.is_finite())
    {
        return Err(ImageAdmissionError::InvalidTransform);
    }
    if descriptor.width == 0 || descriptor.height == 0 {
        return Err(ImageAdmissionError::EmptyDimensions);
    }
    let width = usize::try_from(descriptor.width).map_err(|_| ImageAdmissionError::SizeOverflow)?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| ImageAdmissionError::SizeOverflow)?;
    let row_bytes = width
        .checked_mul(4)
        .ok_or(ImageAdmissionError::SizeOverflow)?;
    if descriptor.row_stride < row_bytes {
        return Err(ImageAdmissionError::InvalidStride);
    }
    let input_bytes = descriptor
        .row_stride
        .checked_mul(height)
        .ok_or(ImageAdmissionError::SizeOverflow)?;
    if input_bytes != bytes.len() {
        return Err(ImageAdmissionError::InvalidLength {
            expected: input_bytes,
            actual: bytes.len(),
        });
    }
    if input_bytes > budget.max_input_bytes {
        return Err(ImageAdmissionError::InputBudgetExceeded);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ImageAdmissionError::SizeOverflow)?;
    // Original conversion allocates linear Color pixels while the packed wire rows
    // remain live. Budget both allocations before calling the source constructor.
    let decoded_bytes = pixels
        .checked_mul(size_of::<Color>() + 4)
        .ok_or(ImageAdmissionError::SizeOverflow)?;
    if decoded_bytes > budget.max_decoded_bytes {
        return Err(ImageAdmissionError::DecodedBudgetExceeded);
    }
    let digest: [u8; 32] = Sha256::digest(bytes.as_ref()).into();
    if digest != identity.content_sha256 {
        return Err(ImageAdmissionError::DigestMismatch);
    }
    let packed_bytes = pixels
        .checked_mul(4)
        .ok_or(ImageAdmissionError::SizeOverflow)?;
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(packed_bytes)
        .map_err(|_| ImageAdmissionError::AllocationFailed)?;
    for row in bytes.chunks_exact(descriptor.row_stride) {
        let pixels = row
            .get(..row_bytes)
            .ok_or(ImageAdmissionError::InvalidStride)?;
        packed.extend_from_slice(pixels);
    }
    let image = Image::<Color>::from_image_data(&packed, descriptor.width, descriptor.height);
    Ok(AdmittedCpuImage {
        identity,
        origin: CpuImageOrigin::Rgba8Rows { bytes: input_bytes },
        descriptor,
        image: Arc::new(image),
    })
}

#[derive(Clone, Copy, Debug)]
pub struct EncodedPngBudget {
    pub max_encoded_bytes: usize,
    pub max_pixel_bytes: usize,
    pub max_metadata_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PngAdmissionError {
    Image(ImageAdmissionError),
    InvalidPng,
    UnsupportedPixelFormat,
    InterlacedPng,
    AnimatedPng,
    UnsupportedChunk([u8; 4]),
    MetadataBudgetExceeded,
    SourceDecodeFailed,
    DecodedShapeMismatch,
}

impl fmt::Display for PngAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(error) => fmt::Display::fmt(error, formatter),
            Self::InvalidPng => formatter.write_str("The PNG framing or metadata is invalid"),
            Self::UnsupportedPixelFormat => {
                formatter.write_str("This PNG profile requires RGB or RGBA 8-bit or 16-bit samples")
            }
            Self::InterlacedPng => {
                formatter.write_str("Interlaced PNG requires a separately admitted profile")
            }
            Self::AnimatedPng => {
                formatter.write_str("Animated PNG requires a separately admitted profile")
            }
            Self::UnsupportedChunk(chunk) => write!(
                formatter,
                "This PNG profile cannot interpret chunk {}",
                String::from_utf8_lossy(chunk)
            ),
            Self::MetadataBudgetExceeded => {
                formatter.write_str("The PNG metadata exceeds its byte budget")
            }
            Self::SourceDecodeFailed => {
                formatter.write_str("The original image decoder could not decode this PNG")
            }
            Self::DecodedShapeMismatch => formatter
                .write_str("The original decoded image does not match the verified PNG header"),
        }
    }
}

impl std::error::Error for PngAdmissionError {}

impl From<ImageAdmissionError> for PngAdmissionError {
    fn from(error: ImageAdmissionError) -> Self {
        Self::Image(error)
    }
}

struct PngPixelLayout {
    width: u32,
    height: u32,
    bit_depth: u8,
    channels: u8,
    pixels: usize,
}

fn png_pixel_layout(
    info: &png::Info<'_>,
    budget: EncodedPngBudget,
) -> Result<PngPixelLayout, PngAdmissionError> {
    if info.interlaced {
        return Err(PngAdmissionError::InterlacedPng);
    }
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => return Err(PngAdmissionError::UnsupportedPixelFormat),
    };
    let bit_depth = match info.bit_depth {
        png::BitDepth::Eight => 8,
        png::BitDepth::Sixteen => 16,
        _ => return Err(PngAdmissionError::UnsupportedPixelFormat),
    };
    if info.width == 0 || info.height == 0 {
        return Err(ImageAdmissionError::EmptyDimensions.into());
    }
    let pixels = usize::try_from(info.width)
        .ok()
        .and_then(|width| {
            usize::try_from(info.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(ImageAdmissionError::SizeOverflow)?;
    // Original decode retains integer samples, converts to RGBA32F and collects
    // Color pixels. This admits those pixel buffers, not codec or total heap use.
    let bytes_per_pixel =
        usize::from(channels) * usize::from(bit_depth / 8) + 2 * size_of::<Color>();
    let pixel_bytes = pixels
        .checked_mul(bytes_per_pixel)
        .ok_or(ImageAdmissionError::SizeOverflow)?;
    if pixel_bytes > budget.max_pixel_bytes {
        return Err(ImageAdmissionError::DecodedBudgetExceeded.into());
    }
    Ok(PngPixelLayout {
        width: info.width,
        height: info.height,
        bit_depth,
        channels,
        pixels,
    })
}

fn preflight_png(
    bytes: &[u8],
    budget: EncodedPngBudget,
) -> Result<PngPixelLayout, PngAdmissionError> {
    let mut decoder = png::StreamingDecoder::new();
    decoder.set_ignore_crc(false);
    decoder.set_skip_ancillary_crc_failures(false);
    decoder.set_ignore_text_chunk(true);
    let mut offset = 0;
    let mut metadata_bytes = 0_usize;
    let mut image_bytes = 0_usize;
    let mut layout = None;
    let mut completed = false;
    let mut empty_transitions = 0;
    while offset < bytes.len() {
        let (consumed, event) = decoder
            .update(
                bytes.get(offset..).ok_or(PngAdmissionError::InvalidPng)?,
                None,
            )
            .map_err(|_| PngAdmissionError::InvalidPng)?;
        offset = offset
            .checked_add(consumed)
            .ok_or(ImageAdmissionError::SizeOverflow)?;
        if consumed == 0 {
            // ImageDataFlushed is a legitimate transition without input. Bound
            // repeated empty transitions so a malformed input cannot spin here.
            empty_transitions += 1;
            if empty_transitions > 8 {
                return Err(PngAdmissionError::InvalidPng);
            }
        } else {
            empty_transitions = 0;
        }
        match event {
            png::Decoded::ChunkBegin(length, chunk) => {
                let length =
                    usize::try_from(length).map_err(|_| ImageAdmissionError::SizeOverflow)?;
                let remaining = bytes
                    .len()
                    .checked_sub(offset)
                    .ok_or(PngAdmissionError::InvalidPng)?;
                if length
                    .checked_add(4)
                    .is_none_or(|required| required > remaining)
                {
                    return Err(PngAdmissionError::InvalidPng);
                }
                match chunk {
                    png::chunk::IHDR if length == 13 => {}
                    png::chunk::IEND if length == 0 => {}
                    png::chunk::IDAT => {
                        image_bytes = image_bytes
                            .checked_add(length)
                            .ok_or(ImageAdmissionError::SizeOverflow)?;
                    }
                    png::chunk::sRGB if length == 1 => {}
                    png::chunk::pHYs if length == 9 => {}
                    png::chunk::tIME if length == 7 => {}
                    png::chunk::tEXt | png::chunk::zTXt | png::chunk::iTXt => {}
                    png::chunk::IHDR
                    | png::chunk::IEND
                    | png::chunk::sRGB
                    | png::chunk::pHYs
                    | png::chunk::tIME => return Err(PngAdmissionError::InvalidPng),
                    png::chunk::acTL | png::chunk::fcTL | png::chunk::fdAT => {
                        return Err(PngAdmissionError::AnimatedPng);
                    }
                    _ => return Err(PngAdmissionError::UnsupportedChunk(chunk.0)),
                }
                if chunk != png::chunk::IHDR
                    && chunk != png::chunk::IDAT
                    && chunk != png::chunk::IEND
                {
                    // Even ignored text is buffered by the library. Check its
                    // announced size before allowing the next decoder update.
                    metadata_bytes = metadata_bytes
                        .checked_add(length)
                        .ok_or(ImageAdmissionError::SizeOverflow)?;
                    if metadata_bytes > budget.max_metadata_bytes {
                        return Err(PngAdmissionError::MetadataBudgetExceeded);
                    }
                }
            }
            png::Decoded::ChunkComplete(png::chunk::IHDR) => {
                layout = Some(png_pixel_layout(
                    decoder.info().ok_or(PngAdmissionError::InvalidPng)?,
                    budget,
                )?);
            }
            png::Decoded::ChunkComplete(png::chunk::IEND) => {
                completed = true;
                break;
            }
            png::Decoded::BadAncillaryChunk(_) => return Err(PngAdmissionError::InvalidPng),
            png::Decoded::SkippedAncillaryChunk(chunk)
                if !matches!(
                    chunk,
                    png::chunk::tEXt | png::chunk::zTXt | png::chunk::iTXt | png::chunk::tIME
                ) =>
            {
                return Err(PngAdmissionError::InvalidPng);
            }
            _ => {}
        }
    }
    if !completed || offset != bytes.len() || image_bytes == 0 {
        return Err(PngAdmissionError::InvalidPng);
    }
    layout.ok_or(PngAdmissionError::InvalidPng)
}

/// An immutable PNG envelope validated before original pixel decoding.
/// This does not authenticate a host grant or durable asset version.
pub struct PreparedEncodedPngSource {
    identity: ImageResourceIdentity,
    bytes: Arc<[u8]>,
    layout: PngPixelLayout,
}

impl PreparedEncodedPngSource {
    pub fn identity(&self) -> &ImageResourceIdentity {
        &self.identity
    }

    pub fn width(&self) -> u32 {
        self.layout.width
    }

    pub fn height(&self) -> u32 {
        self.layout.height
    }

    pub(crate) fn encoded_bytes(&self) -> &Arc<[u8]> {
        &self.bytes
    }

    fn with_validated_identity(
        identity: ImageResourceIdentity,
        bytes: Arc<[u8]>,
        budget: EncodedPngBudget,
    ) -> Result<Self, PngAdmissionError> {
        if bytes.len() > budget.max_encoded_bytes {
            return Err(ImageAdmissionError::InputBudgetExceeded.into());
        }
        let digest: [u8; 32] = Sha256::digest(bytes.as_ref()).into();
        if digest != identity.content_sha256 {
            return Err(ImageAdmissionError::DigestMismatch.into());
        }
        let layout = preflight_png(&bytes, budget)?;
        Ok(Self {
            identity,
            bytes,
            layout,
        })
    }
}

fn validate_encoded_png_identity(
    identity: &ImageResourceIdentity,
) -> Result<(), PngAdmissionError> {
    if identity.asset_id.trim().is_empty() || identity.version_id.trim().is_empty() {
        return Err(ImageAdmissionError::MissingIdentity.into());
    }
    Ok(())
}

pub fn prepare_encoded_png_source(
    identity: ImageResourceIdentity,
    bytes: Arc<[u8]>,
    budget: EncodedPngBudget,
) -> Result<PreparedEncodedPngSource, PngAdmissionError> {
    validate_encoded_png_identity(&identity)?;
    PreparedEncodedPngSource::with_validated_identity(identity, bytes, budget)
}

struct PreparedEncodedPng {
    transform: [f64; 6],
    source: PreparedEncodedPngSource,
}

impl PreparedEncodedPng {
    fn new(
        identity: ImageResourceIdentity,
        transform: [f64; 6],
        bytes: Arc<[u8]>,
        budget: EncodedPngBudget,
    ) -> Result<Self, PngAdmissionError> {
        // Decoded admission reports identity and transform errors before byte errors.
        validate_encoded_png_identity(&identity)?;
        if transform.iter().any(|component| !component.is_finite()) {
            return Err(ImageAdmissionError::InvalidTransform.into());
        }
        Ok(Self {
            transform,
            source: PreparedEncodedPngSource::with_validated_identity(identity, bytes, budget)?,
        })
    }
    fn admit(self, image: Image<Color>) -> Result<AdmittedCpuImage, PngAdmissionError> {
        if image.width != self.source.layout.width
            || image.height != self.source.layout.height
            || image.data.len() != self.source.layout.pixels
        {
            return Err(PngAdmissionError::DecodedShapeMismatch);
        }
        let row_stride = usize::try_from(self.source.layout.width)
            .ok()
            .and_then(|width| width.checked_mul(size_of::<Color>()))
            .ok_or(ImageAdmissionError::SizeOverflow)?;
        Ok(AdmittedCpuImage {
            identity: self.source.identity,
            origin: CpuImageOrigin::EncodedPng {
                bytes: self.source.bytes.len(),
                bit_depth: self.source.layout.bit_depth,
                channels: self.source.layout.channels,
            },
            descriptor: CpuImageDescriptor {
                width: self.source.layout.width,
                height: self.source.layout.height,
                row_stride,
                sample_format: CpuSampleFormat::Rgba32Float,
                color_space: ImageColorSpace::LinearSrgb,
                alpha: AlphaAssociation::Straight,
                transform: self.transform,
            },
            image: Arc::new(image),
        })
    }
}

pub fn admit_encoded_png(
    identity: ImageResourceIdentity,
    transform: [f64; 6],
    bytes: Arc<[u8]>,
    budget: EncodedPngBudget,
) -> Result<AdmittedCpuImage, PngAdmissionError> {
    let prepared = PreparedEncodedPng::new(identity, transform, bytes, budget)?;
    let image = catch_unwind(AssertUnwindSafe(|| {
        Image::<Color>::from_encoded(prepared.source.encoded_bytes())
    }))
    .map_err(|_| PngAdmissionError::SourceDecodeFailed)?
    .ok_or(PngAdmissionError::SourceDecodeFailed)?;
    prepared.admit(image)
}

#[derive(Debug)]
pub enum PngNetworkAdmissionError {
    Admission(PngAdmissionError),
    Compilation(String),
    Construction(graphite_graph_craft::proto::GraphErrors),
    UnexpectedContract {
        input: Option<Box<graphite_core_types::Type>>,
        output: Option<Box<graphite_core_types::Type>>,
    },
    MissingTypedOutput,
    UnexpectedAttributes,
    SourcePanic,
}

impl fmt::Display for PngNetworkAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => error.fmt(formatter),
            Self::Compilation(error) => {
                write!(formatter, "Original PNG graph compilation failed: {error}")
            }
            Self::Construction(errors) => write!(
                formatter,
                "Original PNG graph construction failed: {errors:?}"
            ),
            Self::UnexpectedContract { input, output } => write!(
                formatter,
                "Original PNG graph has an unexpected typed contract: {input:?} to {output:?}"
            ),
            Self::MissingTypedOutput => {
                formatter.write_str("Original PNG graph did not produce its typed output")
            }
            Self::UnexpectedAttributes => {
                formatter.write_str("Original PNG decode returned unexpected item attributes")
            }
            Self::SourcePanic => formatter.write_str("Original PNG graph execution panicked"),
        }
    }
}

impl std::error::Error for PngNetworkAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PngAdmissionError> for PngNetworkAdmissionError {
    fn from(error: PngAdmissionError) -> Self {
        Self::Admission(error)
    }
}

pub async fn admit_encoded_png_via_original_network(
    identity: ImageResourceIdentity,
    transform: [f64; 6],
    bytes: Arc<[u8]>,
    budget: EncodedPngBudget,
) -> Result<AdmittedCpuImage, PngNetworkAdmissionError> {
    use futures::FutureExt as _;
    use graphite_core_types::{Context, concrete, future, item};
    use graphite_graph_craft::{
        application_io::{
            PlatformApplicationIo, PlatformEditorApi,
            resource::{HashMapResourceStorage, ResourceStorage},
        },
        document::{
            DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
            value::TaggedValue,
        },
        graphene_compiler::Compiler,
    };
    use graphite_interpreted_executor::dynamic_executor::DynamicExecutor;
    use graphite_std::{
        ops::passthrough,
        platform_application_io::{decode_image, editor_api, resource},
    };

    AssertUnwindSafe(async move {
        let prepared = PreparedEncodedPng::new(identity, transform, bytes, budget)?;
        let storage = HashMapResourceStorage::new();
        let hash = storage.store(prepared.source.encoded_bytes().as_ref());
        let mut application_io = PlatformApplicationIo::default();
        application_io.inject_resource_proxy(Box::new(storage));
        let api = Arc::new(PlatformEditorApi {
            application_io: Some(Arc::new(application_io)),
            ..PlatformEditorApi::default()
        });
        let api_node = NodeId(0);
        let resource_node = NodeId(1);
        let decode_node = NodeId(2);
        let scope = editor_api::IDENTIFIER.as_static_str();
        let network = NodeNetwork {
            nodes: [
                (
                    api_node,
                    DocumentNode {
                        inputs: vec![NodeInput::value(TaggedValue::EditorApi(api), false)],
                        implementation: DocumentNodeImplementation::ProtoNode(
                            passthrough::IDENTIFIER,
                        ),
                        ..DocumentNode::default()
                    },
                ),
                (
                    resource_node,
                    DocumentNode {
                        inputs: vec![
                            NodeInput::scope(scope),
                            NodeInput::value(TaggedValue::ResourceHash(hash), false),
                        ],
                        implementation: DocumentNodeImplementation::ProtoNode(resource::IDENTIFIER),
                        ..DocumentNode::default()
                    },
                ),
                (
                    decode_node,
                    DocumentNode {
                        inputs: vec![NodeInput::node(resource_node, 0)],
                        implementation: DocumentNodeImplementation::ProtoNode(
                            decode_image::IDENTIFIER,
                        ),
                        ..DocumentNode::default()
                    },
                ),
            ]
            .into_iter()
            .collect(),
            scope_injections: [(scope.to_owned(), (api_node, item!(&PlatformEditorApi)))]
                .into_iter()
                .collect(),
            exports: vec![NodeInput::node(decode_node, 0)],
            ..NodeNetwork::default()
        };
        let proto = Compiler {}
            .compile_single(network)
            .map_err(PngNetworkAdmissionError::Compilation)?;
        let executor = DynamicExecutor::new(proto)
            .await
            .map_err(PngNetworkAdmissionError::Construction)?;
        let input = executor.input_type();
        let output = executor.output_type();
        if input != Some(concrete!(Context)) || output != Some(future!(Item<Raster<CPU>>)) {
            return Err(PngNetworkAdmissionError::UnexpectedContract {
                input: input.map(Box::new),
                output: output.map(Box::new),
            });
        }
        // The original tagged output has no Raster variant; evaluate its exact typed tree.
        let output = executor
            .tree()
            .eval::<Context<'static>, Item<Raster<CPU>>>(executor.output(), None)
            .await
            .ok_or(PngNetworkAdmissionError::MissingTypedOutput)?;
        let (raster, attributes) = output.into_parts();
        if attributes.keys().next().is_some() {
            return Err(PngNetworkAdmissionError::UnexpectedAttributes);
        }
        let image = raster.into_data();
        if image.width == 0 && image.height == 0 && image.data.is_empty() {
            return Err(PngAdmissionError::SourceDecodeFailed.into());
        }
        prepared.admit(image).map_err(Into::into)
    })
    .catch_unwind()
    .await
    .map_err(|_| PngNetworkAdmissionError::SourcePanic)?
}

#[derive(Clone, Copy, Debug)]
pub struct CpuBlackAndWhiteParameters {
    pub use_tint: bool,
    pub tint_gamma_srgb: [f32; 4],
    pub reds: f64,
    pub yellows: f64,
    pub greens: f64,
    pub cyans: f64,
    pub blues: f64,
    pub magentas: f64,
}

impl PartialEq for CpuBlackAndWhiteParameters {
    fn eq(&self, other: &Self) -> bool {
        // The exact operation values participate in stale-result detection,
        // including signed zero and any nonfinite source parameter encoding.
        self.use_tint == other.use_tint
            && self.tint_gamma_srgb.map(f32::to_bits) == other.tint_gamma_srgb.map(f32::to_bits)
            && [
                self.reds,
                self.yellows,
                self.greens,
                self.cyans,
                self.blues,
                self.magentas,
            ]
            .map(f64::to_bits)
                == [
                    other.reds,
                    other.yellows,
                    other.greens,
                    other.cyans,
                    other.blues,
                    other.magentas,
                ]
                .map(f64::to_bits)
    }
}

impl Eq for CpuBlackAndWhiteParameters {}

#[derive(Clone, Copy, Debug)]
pub struct CpuBrightnessContrastParameters {
    pub brightness: f64,
    pub contrast: f64,
    pub use_classic: bool,
    pub classic_pivot: f64,
}

impl PartialEq for CpuBrightnessContrastParameters {
    fn eq(&self, other: &Self) -> bool {
        self.use_classic == other.use_classic
            && [self.brightness, self.contrast, self.classic_pivot].map(f64::to_bits)
                == [other.brightness, other.contrast, other.classic_pivot].map(f64::to_bits)
    }
}

impl Eq for CpuBrightnessContrastParameters {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuImageEdit {
    Invert,
    MakeOpaque,
    BlackAndWhite(CpuBlackAndWhiteParameters),
    BrightnessContrast(CpuBrightnessContrastParameters),
}

impl CpuImageEdit {
    pub fn source_identifier(self) -> &'static str {
        match self {
            Self::Invert => "raster_nodes::adjustments::InvertNode",
            Self::MakeOpaque => "raster_nodes::adjustments::MakeOpaqueNode",
            Self::BlackAndWhite(_) => "raster_nodes::adjustments::BlackAndWhiteNode",
            Self::BrightnessContrast(_) => "raster_nodes::adjustments::BrightnessContrastNode",
        }
    }
}

pub use graphite_core_types::blending::BlendMode as CpuBlendMode;

#[derive(Clone, Debug)]
pub struct CpuImagePixels {
    pixels_sha256: [u8; 32],
    descriptor: CpuImageDescriptor,
    output: Arc<Item<Raster<CPU>>>,
}

impl CpuImagePixels {
    pub fn pixels_sha256(&self) -> [u8; 32] {
        self.pixels_sha256
    }
    pub fn descriptor(&self) -> &CpuImageDescriptor {
        &self.descriptor
    }
    pub fn image(&self) -> &Image<Color> {
        self.output.element().data()
    }
    pub fn attributes(&self) -> &ItemAttributeValues {
        self.output.attributes()
    }
    pub fn requires_scene_render(&self) -> bool {
        self.descriptor.transform != [1., 0., 0., 1., 0., 0.]
            || self.attributes().keys().any(|key| key != ATTR_TRANSFORM)
    }
}

#[derive(Clone, Debug)]
pub struct CpuImageResult {
    input_identity: ImageResourceIdentity,
    input_origin: CpuImageOrigin,
    operation: CpuImageEdit,
    pixels: CpuImagePixels,
}

impl CpuImageResult {
    pub(crate) fn original_item(&self) -> &Item<Raster<CPU>> {
        &self.pixels.output
    }

    pub fn input_identity(&self) -> &ImageResourceIdentity {
        &self.input_identity
    }
    pub fn input_origin(&self) -> CpuImageOrigin {
        self.input_origin
    }
    pub fn operation(&self) -> CpuImageEdit {
        self.operation
    }
    pub fn pixels(&self) -> &CpuImagePixels {
        &self.pixels
    }
    pub fn pixels_sha256(&self) -> [u8; 32] {
        self.pixels.pixels_sha256()
    }
    pub fn descriptor(&self) -> &CpuImageDescriptor {
        self.pixels.descriptor()
    }
    pub fn image(&self) -> &Image<Color> {
        self.pixels.image()
    }
    pub fn attributes(&self) -> &ItemAttributeValues {
        self.pixels.attributes()
    }
    pub fn requires_scene_render(&self) -> bool {
        self.pixels.requires_scene_render()
    }
}

#[derive(Clone, Debug)]
pub struct CpuImageParent {
    identity: ImageResourceIdentity,
    origin: CpuImageOrigin,
    descriptor: CpuImageDescriptor,
    pixels_sha256: [u8; 32],
}

impl CpuImageParent {
    pub fn identity(&self) -> &ImageResourceIdentity {
        &self.identity
    }
    pub fn origin(&self) -> CpuImageOrigin {
        self.origin
    }
    pub fn descriptor(&self) -> &CpuImageDescriptor {
        &self.descriptor
    }
    pub fn pixels_sha256(&self) -> [u8; 32] {
        self.pixels_sha256
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CpuMixParameters {
    pub blend_mode: CpuBlendMode,
    pub opacity: f64,
}

impl PartialEq for CpuMixParameters {
    fn eq(&self, other: &Self) -> bool {
        self.blend_mode == other.blend_mode && self.opacity.to_bits() == other.opacity.to_bits()
    }
}

impl Eq for CpuMixParameters {}

impl CpuMixParameters {
    pub fn effective_opacity_percent(self) -> f64 {
        // Generated source hard bounds use max/min, including NaN -> zero.
        self.opacity.clamp_hard_min(0.).clamp_hard_max(100.)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CpuMixBudget {
    pub max_parent_bytes: usize,
    pub max_result_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct CpuMixResult {
    parents: [CpuImageParent; 2],
    parameters: CpuMixParameters,
    pixels: CpuImagePixels,
}

impl CpuMixResult {
    pub fn parents(&self) -> &[CpuImageParent; 2] {
        &self.parents
    }
    pub fn parameters(&self) -> CpuMixParameters {
        self.parameters
    }
    pub fn pixels(&self) -> &CpuImagePixels {
        &self.pixels
    }
    pub fn pixels_sha256(&self) -> [u8; 32] {
        self.pixels.pixels_sha256()
    }
    pub fn descriptor(&self) -> &CpuImageDescriptor {
        self.pixels.descriptor()
    }
    pub fn image(&self) -> &Image<Color> {
        self.pixels.image()
    }
    pub fn attributes(&self) -> &ItemAttributeValues {
        self.pixels.attributes()
    }
    pub fn requires_scene_render(&self) -> bool {
        self.pixels.requires_scene_render()
    }
    pub fn source_operation(&self) -> &'static str {
        "raster_nodes::blending_nodes::MixNode"
    }
    pub fn source_revision(&self) -> &'static str {
        GRAPHITE_IMAGE_SOURCE_REVISION
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuMixError {
    Image(CpuImageResultError),
    ParentShapeMismatch,
    ParentBudgetExceeded,
    SceneRenderRequired,
}

impl fmt::Display for CpuMixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(error) => fmt::Display::fmt(error, formatter),
            Self::ParentShapeMismatch => {
                formatter.write_str("The Mix parents require equal image dimensions")
            }
            Self::ParentBudgetExceeded => {
                formatter.write_str("The Mix parents exceed their combined retained pixel budget")
            }
            Self::SceneRenderRequired => formatter
                .write_str("The Mix parents require scene rendering before pixel composition"),
        }
    }
}

impl std::error::Error for CpuMixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Image(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CpuImageResultError> for CpuMixError {
    fn from(error: CpuImageResultError) -> Self {
        Self::Image(error)
    }
}

pub fn evaluate_cpu_image_mix(
    over: &AdmittedCpuImage,
    under: &AdmittedCpuImage,
    parameters: CpuMixParameters,
    budget: CpuMixBudget,
) -> Result<CpuMixResult, CpuMixError> {
    let over_bytes = result_layout(over.image(), usize::MAX)?;
    let under_bytes = result_layout(under.image(), usize::MAX)?;
    if over.image().width != under.image().width || over.image().height != under.image().height {
        return Err(CpuMixError::ParentShapeMismatch);
    }
    let parent_bytes = over_bytes
        .checked_add(under_bytes)
        .ok_or(CpuImageResultError::SizeOverflow)?;
    if parent_bytes > budget.max_parent_bytes {
        return Err(CpuMixError::ParentBudgetExceeded);
    }
    result_layout(over.image(), budget.max_result_bytes)?;
    // Original CPU Mix zips pixels and retains only Over attributes. Applying
    // either parent's scene transform would invent semantics at this boundary.
    let identity = [1., 0., 0., 1., 0., 0.];
    if over.descriptor().transform != identity || under.descriptor().transform != identity {
        return Err(CpuMixError::SceneRenderRequired);
    }
    let parents = [image_parent(over)?, image_parent(under)?];
    // This accounts for retained parent/result pixels, not the original nodes'
    // bridge/ClonedNode/evaluation Vec clones or codec/preview peak allocation.
    let output = crate::raster::evaluate_mix_cpu(
        Item::new_from_element(Raster::new_cpu(over.image().clone())).with_attribute(
            ATTR_TRANSFORM,
            DAffine2::from_cols_array(&over.descriptor().transform),
        ),
        Item::new_from_element(Raster::new_cpu(under.image().clone())).with_attribute(
            ATTR_TRANSFORM,
            DAffine2::from_cols_array(&under.descriptor().transform),
        ),
        Item::new_from_element(parameters.blend_mode),
        Item::new_from_element(parameters.opacity),
    )
    .map_err(|_| CpuImageResultError::EvaluationFailed)?;
    Ok(CpuMixResult {
        parents,
        parameters,
        pixels: admit_cpu_pixels(output, budget.max_result_bytes)?,
    })
}

fn image_parent(input: &AdmittedCpuImage) -> Result<CpuImageParent, CpuImageResultError> {
    Ok(CpuImageParent {
        identity: input.identity().clone(),
        origin: input.origin(),
        descriptor: *input.descriptor(),
        pixels_sha256: image_pixels_sha256(input.image())?,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuImageResultError {
    EmptyDimensions,
    SizeOverflow,
    InvalidPixelCount { expected: usize, actual: usize },
    ResultBudgetExceeded,
    NonFiniteSample,
    InvalidTransform,
    EvaluationFailed,
}

impl fmt::Display for CpuImageResultError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions => {
                formatter.write_str("The result image dimensions must be nonzero")
            }
            Self::SizeOverflow => {
                formatter.write_str("The result image layout exceeds the addressable size")
            }
            Self::InvalidPixelCount { expected, actual } => write!(
                formatter,
                "The result image has {actual} pixels; its dimensions require {expected}"
            ),
            Self::ResultBudgetExceeded => {
                formatter.write_str("The image result exceeds its retained pixel budget")
            }
            Self::NonFiniteSample => {
                formatter.write_str("The image result contains a nonfinite sample")
            }
            Self::InvalidTransform => {
                formatter.write_str("The image result requires a finite typed transform")
            }
            Self::EvaluationFailed => formatter.write_str("The original image operation failed"),
        }
    }
}

impl std::error::Error for CpuImageResultError {}

fn result_layout(
    image: &Image<Color>,
    max_result_bytes: usize,
) -> Result<usize, CpuImageResultError> {
    if image.width == 0 || image.height == 0 {
        return Err(CpuImageResultError::EmptyDimensions);
    }
    let width = usize::try_from(image.width).map_err(|_| CpuImageResultError::SizeOverflow)?;
    let height = usize::try_from(image.height).map_err(|_| CpuImageResultError::SizeOverflow)?;
    let pixels = width
        .checked_mul(height)
        .ok_or(CpuImageResultError::SizeOverflow)?;
    if image.data.len() != pixels {
        return Err(CpuImageResultError::InvalidPixelCount {
            expected: pixels,
            actual: image.data.len(),
        });
    }
    let bytes = pixels
        .checked_mul(size_of::<Color>())
        .ok_or(CpuImageResultError::SizeOverflow)?;
    if bytes > max_result_bytes {
        return Err(CpuImageResultError::ResultBudgetExceeded);
    }
    Ok(bytes)
}

pub fn evaluate_cpu_image_edit(
    input: &AdmittedCpuImage,
    operation: CpuImageEdit,
    max_result_bytes: usize,
) -> Result<CpuImageResult, CpuImageResultError> {
    result_layout(input.image(), max_result_bytes)?;
    // Original nodes clone their input internally. This limit admits retained
    // result pixels; it is not a bound on the source evaluator's peak allocation.
    let item = Item::new_from_element(Raster::new_cpu(input.image().clone())).with_attribute(
        ATTR_TRANSFORM,
        DAffine2::from_cols_array(&input.descriptor().transform),
    );
    let output = match operation {
        CpuImageEdit::Invert => crate::raster::evaluate_invert_cpu(item),
        CpuImageEdit::MakeOpaque => crate::raster::evaluate_make_opaque_cpu(item),
        CpuImageEdit::BrightnessContrast(parameters) => {
            crate::raster::evaluate_brightness_contrast_cpu(
                item,
                crate::raster::BrightnessContrastParameters {
                    brightness: Item::new_from_element(parameters.brightness),
                    contrast: Item::new_from_element(parameters.contrast),
                    use_classic: Item::new_from_element(parameters.use_classic),
                    classic_pivot: Item::new_from_element(parameters.classic_pivot),
                },
            )
        }
        CpuImageEdit::BlackAndWhite(parameters) => {
            let [red, green, blue, alpha] = parameters.tint_gamma_srgb;
            crate::raster::evaluate_black_and_white_cpu(
                item,
                crate::raster::BlackAndWhiteParameters {
                    use_tint: Item::new_from_element(parameters.use_tint),
                    tint: Item::new_from_element(Color::from_gamma_srgb_channels(
                        red, green, blue, alpha,
                    )),
                    reds: Item::new_from_element(parameters.reds),
                    yellows: Item::new_from_element(parameters.yellows),
                    greens: Item::new_from_element(parameters.greens),
                    cyans: Item::new_from_element(parameters.cyans),
                    blues: Item::new_from_element(parameters.blues),
                    magentas: Item::new_from_element(parameters.magentas),
                },
            )
        }
    }
    .map_err(|_| CpuImageResultError::EvaluationFailed)?;
    admit_cpu_result(
        input.identity().clone(),
        input.origin(),
        operation,
        output,
        max_result_bytes,
    )
}

pub(crate) fn admit_cpu_result(
    input_identity: ImageResourceIdentity,
    input_origin: CpuImageOrigin,
    operation: CpuImageEdit,
    output: graphite_core_types::list::Item<
        graphite_raster_types::Raster<graphite_raster_types::CPU>,
    >,
    max_result_bytes: usize,
) -> Result<CpuImageResult, CpuImageResultError> {
    Ok(CpuImageResult {
        input_identity,
        input_origin,
        operation,
        pixels: admit_cpu_pixels(output, max_result_bytes)?,
    })
}

fn admit_cpu_pixels(
    output: Item<Raster<CPU>>,
    max_result_bytes: usize,
) -> Result<CpuImagePixels, CpuImageResultError> {
    let image = output.element().data();
    result_layout(image, max_result_bytes)?;
    let transform = match output.attributes().get_any(ATTR_TRANSFORM) {
        Some(value) => value
            .downcast_ref::<DAffine2>()
            .ok_or(CpuImageResultError::InvalidTransform)?
            .to_cols_array(),
        None => [1., 0., 0., 1., 0., 0.],
    };
    if transform.iter().any(|component| !component.is_finite()) {
        return Err(CpuImageResultError::InvalidTransform);
    }
    let pixels_sha256 = image_pixels_sha256(image)?;
    let descriptor = CpuImageDescriptor {
        width: image.width,
        height: image.height,
        row_stride: usize::try_from(image.width)
            .map_err(|_| CpuImageResultError::SizeOverflow)?
            .checked_mul(size_of::<Color>())
            .ok_or(CpuImageResultError::SizeOverflow)?,
        sample_format: CpuSampleFormat::Rgba32Float,
        color_space: ImageColorSpace::LinearSrgb,
        alpha: AlphaAssociation::Straight,
        transform,
    };
    Ok(CpuImagePixels {
        pixels_sha256,
        descriptor,
        output: Arc::new(output),
    })
}

fn image_pixels_sha256(image: &Image<Color>) -> Result<[u8; 32], CpuImageResultError> {
    let mut digest = Sha256::new();
    digest.update(b"sim.graphite.cpu-pixels.v1\0");
    digest.update(image.width.to_le_bytes());
    digest.update(image.height.to_le_bytes());
    digest.update(
        u64::try_from(image.data.len())
            .map_err(|_| CpuImageResultError::SizeOverflow)?
            .to_le_bytes(),
    );
    for pixel in &image.data {
        for component in [pixel.r(), pixel.g(), pixel.b(), pixel.a()] {
            if !component.is_finite() {
                return Err(CpuImageResultError::NonFiniteSample);
            }
            digest.update(component.to_bits().to_le_bytes());
        }
    }
    Ok(digest.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(bytes: &[u8]) -> ImageResourceIdentity {
        ImageResourceIdentity {
            asset_id: "project-image".into(),
            version_id: "version-1".into(),
            content_sha256: Sha256::digest(bytes).into(),
        }
    }

    fn descriptor() -> CpuImageDescriptor {
        CpuImageDescriptor {
            width: 2,
            height: 2,
            row_stride: 8,
            sample_format: CpuSampleFormat::Rgba8,
            color_space: ImageColorSpace::Srgb,
            alpha: AlphaAssociation::Straight,
            transform: [1., 0., 0., 1., 0., 0.],
        }
    }

    fn budget() -> CpuImageBudget {
        CpuImageBudget {
            max_input_bytes: 24,
            max_decoded_bytes: 4 * (size_of::<Color>() + 4),
        }
    }

    const PIXELS: [u8; 16] = [
        255, 0, 0, 255, 128, 64, 32, 128, 20, 30, 40, 0, 0, 255, 255, 255,
    ];

    #[test]
    fn at_r08_1_original_color_conversion_and_immutable_ownership()
    -> Result<(), ImageAdmissionError> {
        let bytes: Arc<[u8]> = PIXELS.into();
        let weak = Arc::downgrade(&bytes);
        let image = admit_cpu_image(identity(&bytes), descriptor(), bytes, budget())?;
        let original = Image::<Color>::from_image_data(&PIXELS, 2, 2);
        assert_eq!(image.image(), &original);
        assert_eq!(image.srgba8_pixels().flatten().collect::<Vec<_>>(), PIXELS);
        assert_eq!(image.identity(), &identity(&PIXELS));
        assert_eq!(image.descriptor(), &descriptor());
        assert!(weak.upgrade().is_none());
        let clone = image.clone();
        assert!(std::ptr::eq(image.image(), clone.image()));
        Ok(())
    }

    #[test]
    fn at_r08_1_padded_rows_preserve_exact_pixels_and_wire_digest()
    -> Result<(), ImageAdmissionError> {
        let mut padded = PIXELS.get(..8).expect("first row").to_vec();
        padded.extend_from_slice(&[9, 8, 7, 6]);
        padded.extend_from_slice(PIXELS.get(8..).expect("second row"));
        padded.extend_from_slice(&[5, 4, 3, 2]);
        let descriptor = CpuImageDescriptor {
            row_stride: 12,
            transform: [1., 2., 3., 4., 5., 6.],
            ..descriptor()
        };
        let image = admit_cpu_image(
            identity(&padded),
            descriptor,
            padded.clone().into(),
            budget(),
        )?;
        assert_eq!(
            image.image(),
            &Image::<Color>::from_image_data(&PIXELS, 2, 2)
        );
        let digest: [u8; 32] = Sha256::digest(&padded).into();
        assert_eq!(image.identity().content_sha256, digest);
        assert_eq!(image.descriptor().transform, [1., 2., 3., 4., 5., 6.]);
        assert_ne!(
            image.identity().content_sha256,
            identity(&PIXELS).content_sha256
        );
        Ok(())
    }

    #[test]
    fn at_r28_1_reject_malformed_layout_instead_of_truncating() {
        let cases = [
            (
                CpuImageDescriptor {
                    width: 0,
                    ..descriptor()
                },
                ImageAdmissionError::EmptyDimensions,
            ),
            (
                CpuImageDescriptor {
                    height: 0,
                    ..descriptor()
                },
                ImageAdmissionError::EmptyDimensions,
            ),
            (
                CpuImageDescriptor {
                    row_stride: 7,
                    ..descriptor()
                },
                ImageAdmissionError::InvalidStride,
            ),
            (
                CpuImageDescriptor {
                    row_stride: usize::MAX,
                    ..descriptor()
                },
                ImageAdmissionError::SizeOverflow,
            ),
            (
                CpuImageDescriptor {
                    row_stride: 9,
                    ..descriptor()
                },
                ImageAdmissionError::InvalidLength {
                    expected: 18,
                    actual: 16,
                },
            ),
        ];
        for (descriptor, error) in cases {
            assert_eq!(
                admit_cpu_image(identity(&PIXELS), descriptor, PIXELS.into(), budget()).err(),
                Some(error)
            );
        }
        for length in [15, 17] {
            let bytes = vec![0; length];
            assert_eq!(
                admit_cpu_image(identity(&bytes), descriptor(), bytes.into(), budget()).err(),
                Some(ImageAdmissionError::InvalidLength {
                    expected: 16,
                    actual: length
                })
            );
        }
    }

    #[test]
    fn at_r28_1_explicit_sample_color_alpha_and_transform_admission() {
        let cases = [
            (
                CpuImageDescriptor {
                    sample_format: CpuSampleFormat::Rgba32Float,
                    ..descriptor()
                },
                ImageAdmissionError::UnsupportedSamples,
            ),
            (
                CpuImageDescriptor {
                    color_space: ImageColorSpace::LinearSrgb,
                    ..descriptor()
                },
                ImageAdmissionError::UnsupportedColorSpace,
            ),
            (
                CpuImageDescriptor {
                    alpha: AlphaAssociation::Premultiplied,
                    ..descriptor()
                },
                ImageAdmissionError::UnsupportedAlpha,
            ),
            (
                CpuImageDescriptor {
                    transform: [f64::NAN, 0., 0., 1., 0., 0.],
                    ..descriptor()
                },
                ImageAdmissionError::InvalidTransform,
            ),
            (
                CpuImageDescriptor {
                    transform: [1., 0., 0., 1., f64::INFINITY, 0.],
                    ..descriptor()
                },
                ImageAdmissionError::InvalidTransform,
            ),
        ];
        for (descriptor, error) in cases {
            assert_eq!(
                admit_cpu_image(identity(&PIXELS), descriptor, PIXELS.into(), budget()).err(),
                Some(error)
            );
        }
    }

    #[test]
    fn at_r08_1_content_and_identity_are_checked_without_rewriting() {
        let mut changed = PIXELS;
        changed[0] = 254;
        assert_eq!(
            admit_cpu_image(identity(&PIXELS), descriptor(), changed.into(), budget()).err(),
            Some(ImageAdmissionError::DigestMismatch)
        );
        for identity in [
            ImageResourceIdentity {
                asset_id: " ".into(),
                ..identity(&PIXELS)
            },
            ImageResourceIdentity {
                version_id: "".into(),
                ..identity(&PIXELS)
            },
        ] {
            assert_eq!(
                admit_cpu_image(identity, descriptor(), PIXELS.into(), budget()).err(),
                Some(ImageAdmissionError::MissingIdentity)
            );
        }
    }

    #[test]
    fn at_r08_1_budget_accounts_for_wire_and_original_conversion() -> Result<(), ImageAdmissionError>
    {
        for (budget, error) in [
            (
                CpuImageBudget {
                    max_input_bytes: 15,
                    ..budget()
                },
                ImageAdmissionError::InputBudgetExceeded,
            ),
            (
                CpuImageBudget {
                    max_decoded_bytes: budget().max_decoded_bytes - 1,
                    ..budget()
                },
                ImageAdmissionError::DecodedBudgetExceeded,
            ),
        ] {
            assert_eq!(
                admit_cpu_image(identity(&PIXELS), descriptor(), PIXELS.into(), budget).err(),
                Some(error)
            );
        }
        admit_cpu_image(
            identity(&PIXELS),
            descriptor(),
            PIXELS.into(),
            CpuImageBudget {
                max_input_bytes: 16,
                ..budget()
            },
        )?;
        Ok(())
    }

    fn result_item(width: u32, height: u32, pixels: Vec<Color>) -> Item<Raster<CPU>> {
        Item::new_from_element(Raster::new_cpu(Image {
            width,
            height,
            data: pixels,
            base64_string: None,
        }))
    }

    #[test]
    fn at_r08_cpu_result_make_opaque_preserves_linear_precision_and_parent() {
        let source = admit_cpu_image(identity(&PIXELS), descriptor(), PIXELS.into(), budget())
            .expect("admitted source");
        let before = source.image().clone();
        let result = evaluate_cpu_image_edit(&source, CpuImageEdit::MakeOpaque, 64)
            .expect("original opaque result");
        assert_eq!(result.input_identity(), source.identity());
        assert_eq!(
            result.operation().source_identifier(),
            "raster_nodes::adjustments::MakeOpaqueNode"
        );
        assert_eq!(
            result.descriptor().sample_format,
            CpuSampleFormat::Rgba32Float
        );
        assert_eq!(result.descriptor().color_space, ImageColorSpace::LinearSrgb);
        assert_eq!(result.descriptor().row_stride, 2 * size_of::<Color>());
        assert_eq!((result.image().width, result.image().height), (2, 2));
        for (output, original) in result.image().data.iter().zip(&before.data) {
            assert_eq!(
                [
                    output.r().to_bits(),
                    output.g().to_bits(),
                    output.b().to_bits()
                ],
                [
                    original.r().to_bits(),
                    original.g().to_bits(),
                    original.b().to_bits()
                ]
            );
            assert_eq!(output.a().to_bits(), 1_f32.to_bits());
        }
        assert_eq!(source.image(), &before);
        assert!(!result.requires_scene_render());
        assert_ne!(result.pixels_sha256(), source.identity().content_sha256);
        let repeat = evaluate_cpu_image_edit(&source, CpuImageEdit::MakeOpaque, 64)
            .expect("repeat source result");
        assert_eq!(result.pixels_sha256(), repeat.pixels_sha256());
        assert_eq!(result.image(), repeat.image());
    }

    #[test]
    fn at_r08_cpu_result_invert_uses_source_evaluation_and_retains_parent() {
        let source = admit_cpu_image(identity(&PIXELS), descriptor(), PIXELS.into(), budget())
            .expect("admitted source");
        let before = source.image().clone();
        let result = evaluate_cpu_image_edit(&source, CpuImageEdit::Invert, 64)
            .expect("original invert result");
        let expected = crate::raster::evaluate_invert_cpu(result_item(2, 2, before.data.clone()))
            .expect("original oracle");
        assert_eq!(result.image(), expected.element().data());
        assert_eq!(result.input_identity(), source.identity());
        assert_eq!(
            result.operation().source_identifier(),
            "raster_nodes::adjustments::InvertNode"
        );
        assert_eq!(source.image(), &before);
        let repeat = evaluate_cpu_image_edit(&source, CpuImageEdit::Invert, 64)
            .expect("repeat source result");
        assert_eq!(result.pixels_sha256(), repeat.pixels_sha256());
    }

    #[test]
    fn at_r08_cpu_result_admission_rejects_shape_and_result_budget() {
        for (item, maximum, error) in [
            (
                result_item(0, 1, vec![]),
                64,
                CpuImageResultError::EmptyDimensions,
            ),
            (
                result_item(2, 2, vec![Color::BLACK; 3]),
                64,
                CpuImageResultError::InvalidPixelCount {
                    expected: 4,
                    actual: 3,
                },
            ),
            (
                result_item(2, 2, vec![Color::BLACK; 5]),
                64,
                CpuImageResultError::InvalidPixelCount {
                    expected: 4,
                    actual: 5,
                },
            ),
            (
                result_item(2, 2, vec![Color::BLACK; 4]),
                63,
                CpuImageResultError::ResultBudgetExceeded,
            ),
        ] {
            assert_eq!(
                admit_cpu_result(
                    identity(&PIXELS),
                    CpuImageOrigin::Rgba8Rows {
                        bytes: PIXELS.len()
                    },
                    CpuImageEdit::MakeOpaque,
                    item,
                    maximum
                )
                .err(),
                Some(error)
            );
        }
        let source = admit_cpu_image(identity(&PIXELS), descriptor(), PIXELS.into(), budget())
            .expect("source");
        assert_eq!(
            evaluate_cpu_image_edit(&source, CpuImageEdit::Invert, 63).err(),
            Some(CpuImageResultError::ResultBudgetExceeded)
        );
        assert_eq!(
            source.image(),
            &Image::<Color>::from_image_data(&PIXELS, 2, 2)
        );
    }

    #[test]
    fn at_r08_cpu_result_retains_all_attributes_and_requires_scene_for_unknown_effects() {
        let item = result_item(1, 1, vec![Color::BLACK])
            .with_attribute(ATTR_TRANSFORM, DAffine2::IDENTITY)
            .with_attribute("asset_annotation", String::from("keep editable work"));
        let attributes = item.attributes().clone();
        let result = admit_cpu_result(
            identity(&PIXELS),
            CpuImageOrigin::Rgba8Rows {
                bytes: PIXELS.len(),
            },
            CpuImageEdit::MakeOpaque,
            item,
            16,
        )
        .expect("retained attributes");
        assert_eq!(result.attributes(), &attributes);
        assert!(result.requires_scene_render());
        let source = admit_cpu_image(
            identity(&PIXELS),
            CpuImageDescriptor {
                transform: [1., 0., 0., 1., 3., 4.],
                ..descriptor()
            },
            PIXELS.into(),
            budget(),
        )
        .expect("transformed source");
        let result = evaluate_cpu_image_edit(&source, CpuImageEdit::MakeOpaque, 64)
            .expect("original transformed result");
        assert_eq!(result.descriptor().transform, source.descriptor().transform);
        assert!(result.requires_scene_render());
    }

    #[test]
    fn at_r08_cpu_result_rejects_nonfinite_samples_and_wrong_typed_transform() {
        for pixel in [
            Color::from_rgbaf32_unchecked(f32::NAN, 0., 0., 1.),
            Color::from_rgbaf32_unchecked(0., f32::INFINITY, 0., 1.),
            Color::from_rgbaf32_unchecked(0., 0., 0., f32::NEG_INFINITY),
        ] {
            assert_eq!(
                admit_cpu_result(
                    identity(&PIXELS),
                    CpuImageOrigin::Rgba8Rows {
                        bytes: PIXELS.len()
                    },
                    CpuImageEdit::Invert,
                    result_item(1, 1, vec![pixel]),
                    16
                )
                .err(),
                Some(CpuImageResultError::NonFiniteSample)
            );
        }
        for item in [
            result_item(1, 1, vec![Color::BLACK])
                .with_attribute(ATTR_TRANSFORM, String::from("identity")),
            result_item(1, 1, vec![Color::BLACK]).with_attribute(
                ATTR_TRANSFORM,
                DAffine2::from_cols_array(&[1., 0., 0., 1., f64::NAN, 0.]),
            ),
        ] {
            assert_eq!(
                admit_cpu_result(
                    identity(&PIXELS),
                    CpuImageOrigin::Rgba8Rows {
                        bytes: PIXELS.len()
                    },
                    CpuImageEdit::Invert,
                    item,
                    16
                )
                .err(),
                Some(CpuImageResultError::InvalidTransform)
            );
        }
    }

    #[test]
    fn at_r08_cpu_result_digest_is_bits_shape_and_order_sensitive_without_version_minting() {
        let pixel = Color::from_rgbaf32_unchecked(0.25, -0., 0.75, 0.5);
        let result = admit_cpu_result(
            identity(&PIXELS),
            CpuImageOrigin::Rgba8Rows {
                bytes: PIXELS.len(),
            },
            CpuImageEdit::Invert,
            result_item(1, 1, vec![pixel]),
            16,
        )
        .expect("float result");
        let positive_zero = admit_cpu_result(
            identity(&PIXELS),
            CpuImageOrigin::Rgba8Rows {
                bytes: PIXELS.len(),
            },
            CpuImageEdit::Invert,
            result_item(
                1,
                1,
                vec![Color::from_rgbaf32_unchecked(0.25, 0., 0.75, 0.5)],
            ),
            16,
        )
        .expect("positive zero");
        assert_eq!(
            result.pixels_sha256(),
            [
                98, 114, 162, 130, 147, 148, 150, 134, 52, 215, 139, 249, 226, 74, 12, 255, 97, 55,
                204, 141, 234, 176, 209, 251, 208, 162, 176, 164, 192, 10, 96, 230
            ]
        );
        assert_ne!(result.pixels_sha256(), positive_zero.pixels_sha256());
        assert_eq!(
            result.image().data.first().expect("pixel").g().to_bits(),
            (-0_f32).to_bits()
        );
        let pixels = vec![Color::BLACK, Color::WHITE, pixel, Color::TRANSPARENT];
        let square = admit_cpu_result(
            identity(&PIXELS),
            CpuImageOrigin::Rgba8Rows {
                bytes: PIXELS.len(),
            },
            CpuImageEdit::Invert,
            result_item(2, 2, pixels.clone()),
            64,
        )
        .expect("square");
        let row = admit_cpu_result(
            identity(&PIXELS),
            CpuImageOrigin::Rgba8Rows {
                bytes: PIXELS.len(),
            },
            CpuImageEdit::Invert,
            result_item(4, 1, pixels.clone()),
            64,
        )
        .expect("row");
        let mut reordered = pixels.clone();
        reordered.reverse();
        let reverse = admit_cpu_result(
            identity(&PIXELS),
            CpuImageOrigin::Rgba8Rows {
                bytes: PIXELS.len(),
            },
            CpuImageEdit::Invert,
            result_item(2, 2, reordered),
            64,
        )
        .expect("reverse");
        assert_ne!(square.pixels_sha256(), row.pixels_sha256());
        assert_ne!(square.pixels_sha256(), reverse.pixels_sha256());
        let other_parent = ImageResourceIdentity {
            version_id: "host-version-2".into(),
            ..identity(&PIXELS)
        };
        let second_version = admit_cpu_result(
            other_parent.clone(),
            CpuImageOrigin::Rgba8Rows {
                bytes: PIXELS.len(),
            },
            CpuImageEdit::Invert,
            result_item(2, 2, pixels),
            64,
        )
        .expect("different parent");
        assert_eq!(square.pixels_sha256(), second_version.pixels_sha256());
        assert_eq!(second_version.input_identity(), &other_parent);
        assert_eq!(square.input_identity(), &identity(&PIXELS));
    }

    fn encoded_fixture(
        color: png::ColorType,
        depth: png::BitDepth,
        samples: &[u8],
        chunks: &[(png::chunk::ChunkType, &[u8])],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(color);
            encoder.set_depth(depth);
            let mut writer = encoder.write_header().expect("test PNG header");
            for &(kind, data) in chunks {
                writer
                    .write_chunk(kind, data)
                    .expect("test metadata framing");
            }
            writer.write_image_data(samples).expect("test PNG samples");
            writer.finish().expect("test PNG terminator");
        }
        bytes
    }

    fn encoded_budget() -> EncodedPngBudget {
        EncodedPngBudget {
            max_encoded_bytes: 4096,
            max_pixel_bytes: 4096,
            max_metadata_bytes: 256,
        }
    }

    fn encoded_identity(bytes: &[u8]) -> ImageResourceIdentity {
        ImageResourceIdentity {
            asset_id: "project/encoded-source".into(),
            version_id: "supplied-linked-version-1".into(),
            content_sha256: Sha256::digest(bytes).into(),
        }
    }

    fn admit_fixture(
        bytes: &[u8],
        budget: EncodedPngBudget,
    ) -> Result<AdmittedCpuImage, PngAdmissionError> {
        admit_encoded_png(
            encoded_identity(bytes),
            [1., 0., 0., 1., 0., 0.],
            bytes.into(),
            budget,
        )
    }

    #[test]
    fn at_r08_encoded_png_original_rgb_rgba_8_16_preserves_sample_precision_and_parent_bytes() {
        for (color, channels) in [(png::ColorType::Rgb, 3), (png::ColorType::Rgba, 4)] {
            for (depth, bit_depth) in [(png::BitDepth::Eight, 8), (png::BitDepth::Sixteen, 16)] {
                let values: Vec<u16> = if channels == 3 {
                    vec![1, 256, 0x1234, 65535, 32768, 65534]
                } else {
                    vec![1, 256, 0x1234, 32768, 65535, 32768, 65534, 0]
                };
                let samples: Vec<u8> = if bit_depth == 16 {
                    values
                        .iter()
                        .flat_map(|value| value.to_be_bytes())
                        .collect()
                } else {
                    values.iter().map(|value| (value >> 8) as u8).collect()
                };
                let bytes = encoded_fixture(color, depth, &samples, &[]);
                let before = bytes.clone();
                let identity = encoded_identity(&bytes);
                let image = admit_fixture(&bytes, encoded_budget())
                    .expect("real encoded integer PNG admitted");
                assert_eq!(image.identity(), &identity);
                assert_eq!(
                    image.origin(),
                    CpuImageOrigin::EncodedPng {
                        bytes: bytes.len(),
                        bit_depth,
                        channels
                    }
                );
                assert_eq!(
                    image.origin().digest_representation(),
                    "encoded-png-bytes-sha256"
                );
                assert_eq!(image.origin().byte_length(), bytes.len());
                assert!(image.origin().decode_profile().is_some());
                assert_eq!(
                    image.descriptor().sample_format,
                    CpuSampleFormat::Rgba32Float
                );
                assert_eq!(image.descriptor().color_space, ImageColorSpace::LinearSrgb);
                assert_eq!(image.descriptor().alpha, AlphaAssociation::Straight);
                assert_eq!(image.descriptor().row_stride, 2 * size_of::<Color>());
                assert_eq!(
                    image.image(),
                    &Image::<Color>::from_encoded(&bytes)
                        .expect("unchanged original source oracle")
                );
                for (pixel, input) in image
                    .image()
                    .data
                    .iter()
                    .zip(values.chunks_exact(usize::from(channels)))
                {
                    let channel = |index: usize| {
                        if bit_depth == 16 {
                            f32::from(input[index]) / 65535.
                        } else {
                            f32::from(input[index] >> 8) / 255.
                        }
                    };
                    let alpha = if channels == 4 { channel(3) } else { 1. };
                    let expected =
                        Color::from_gamma_srgb_channels(channel(0), channel(1), channel(2), alpha);
                    assert_eq!(
                        [
                            pixel.r().to_bits(),
                            pixel.g().to_bits(),
                            pixel.b().to_bits(),
                            pixel.a().to_bits()
                        ],
                        [
                            expected.r().to_bits(),
                            expected.g().to_bits(),
                            expected.b().to_bits(),
                            expected.a().to_bits()
                        ]
                    );
                    if channels == 3 {
                        assert_eq!(pixel.a().to_bits(), 1_f32.to_bits());
                    }
                }
                if bit_depth == 16 {
                    assert!(image.image().data.first().expect("first pixel").r() > 0.);
                }
                for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
                    let result =
                        evaluate_cpu_image_edit(&image, operation, 32).expect("original CPU edit");
                    assert_eq!(result.input_identity(), &identity);
                    assert_eq!(result.input_origin(), image.origin());
                    assert_ne!(result.pixels_sha256(), identity.content_sha256);
                }
                assert_eq!(bytes, before);
            }
        }
    }

    #[test]
    fn at_r08_encoded_png_checks_digest_identity_transform_and_pixel_budgets() {
        let bytes = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[],
        );
        let identity = encoded_identity(&bytes);
        assert_eq!(
            admit_fixture(
                &bytes,
                EncodedPngBudget {
                    max_encoded_bytes: bytes.len() - 1,
                    ..encoded_budget()
                }
            )
            .err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::InputBudgetExceeded
            ))
        );
        let exact_pixels = 2 * (3 + 2 * size_of::<Color>());
        assert!(
            admit_fixture(
                &bytes,
                EncodedPngBudget {
                    max_pixel_bytes: exact_pixels,
                    ..encoded_budget()
                }
            )
            .is_ok()
        );
        assert_eq!(
            admit_fixture(
                &bytes,
                EncodedPngBudget {
                    max_pixel_bytes: exact_pixels - 1,
                    ..encoded_budget()
                }
            )
            .err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::DecodedBudgetExceeded
            ))
        );
        let mut wrong = identity.clone();
        wrong.content_sha256[0] ^= 1;
        assert_eq!(
            admit_encoded_png(
                wrong,
                [1., 0., 0., 1., 0., 0.],
                bytes.clone().into(),
                encoded_budget()
            )
            .err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::DigestMismatch
            ))
        );
        let mut missing = identity.clone();
        missing.version_id = " ".into();
        assert_eq!(
            admit_encoded_png(
                missing,
                [1., 0., 0., 1., 0., 0.],
                bytes.clone().into(),
                encoded_budget()
            )
            .err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::MissingIdentity
            ))
        );
        assert_eq!(
            admit_encoded_png(
                identity.clone(),
                [1., 0., 0., 1., f64::NAN, 0.],
                bytes.clone().into(),
                encoded_budget()
            )
            .err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::InvalidTransform
            ))
        );
        let transform = [1., 0., 0., 1., 3., 4.];
        let image = admit_encoded_png(identity, transform, bytes.into(), encoded_budget())
            .expect("explicit transform preserved");
        assert_eq!(image.descriptor().transform, transform);
        assert!(
            evaluate_cpu_image_edit(&image, CpuImageEdit::Invert, 32)
                .expect("source edit")
                .requires_scene_render()
        );
    }

    #[test]
    fn at_r08_encoded_png_retains_workflow_text_bytes_without_parsing_and_bounds_metadata() {
        let text = b"workflow\0{\"nodes\":[1,2]}";
        let bytes = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[
                (png::chunk::tEXt, text),
                (png::chunk::tEXt, b"prompt\0editable source"),
            ],
        );
        let before = bytes.clone();
        let total = text.len() + b"prompt\0editable source".len();
        let image = admit_fixture(
            &bytes,
            EncodedPngBudget {
                max_metadata_bytes: total,
                ..encoded_budget()
            },
        )
        .expect("bounded source workflow text");
        assert_eq!(
            image.identity().content_sha256,
            Sha256::digest(&bytes).as_slice()
        );
        assert_eq!(image.origin().byte_length(), bytes.len());
        assert_eq!(
            admit_fixture(
                &bytes,
                EncodedPngBudget {
                    max_metadata_bytes: total - 1,
                    ..encoded_budget()
                }
            )
            .err(),
            Some(PngAdmissionError::MetadataBudgetExceeded)
        );
        assert_eq!(bytes, before);
    }

    #[test]
    fn at_r08_encoded_png_rejects_unsupported_chunks_before_hidden_payload_parsing() {
        for (kind, payload) in [
            (png::chunk::iCCP, &b"broken ICC"[..]),
            (png::chunk::gAMA, &[0, 0, 177, 143][..]),
            (png::chunk::cHRM, &[0; 32][..]),
            (png::chunk::cICP, &[1, 13, 0, 1][..]),
            (png::chunk::sBIT, &[8, 8, 8][..]),
            (png::chunk::eXIf, &b"retained orientation"[..]),
            (png::chunk::ChunkType(*b"zzZZ"), &b"unknown semantic"[..]),
        ] {
            let bytes = encoded_fixture(
                png::ColorType::Rgb,
                png::BitDepth::Eight,
                &[1, 2, 3, 4, 5, 6],
                &[(kind, payload)],
            );
            assert_eq!(
                admit_fixture(&bytes, encoded_budget()).err(),
                Some(PngAdmissionError::UnsupportedChunk(kind.0))
            );
        }
        let animated = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[(png::chunk::acTL, &[0, 0, 0, 1, 0, 0, 0, 0])],
        );
        assert_eq!(
            admit_fixture(&animated, encoded_budget()).err(),
            Some(PngAdmissionError::AnimatedPng)
        );
        let gray = encoded_fixture(
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            &[1, 2],
            &[],
        );
        assert_eq!(
            admit_fixture(&gray, encoded_budget()).err(),
            Some(PngAdmissionError::UnsupportedPixelFormat)
        );
        let mut header = png::Info::with_size(2, 1);
        header.color_type = png::ColorType::Rgb;
        header.interlaced = true;
        assert_eq!(
            png_pixel_layout(&header, encoded_budget()).err(),
            Some(PngAdmissionError::InterlacedPng)
        );
    }

    #[test]
    fn at_r08_encoded_png_strict_framing_crc_srgb_and_truncation_reject_without_mutation() {
        let bytes = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[],
        );
        for length in [0, 8, 16, bytes.len() - 1] {
            assert_eq!(
                admit_fixture(&bytes[..length], encoded_budget()).err(),
                Some(PngAdmissionError::InvalidPng)
            );
        }
        let mut trailing = bytes.clone();
        trailing.extend_from_slice(b"unexpected trailing data");
        assert_eq!(
            admit_fixture(&trailing, encoded_budget()).err(),
            Some(PngAdmissionError::InvalidPng)
        );
        let mut bad_crc = bytes.clone();
        let last = bad_crc.last_mut().expect("IEND CRC byte");
        *last ^= 1;
        assert_eq!(
            admit_fixture(&bad_crc, encoded_budget()).err(),
            Some(PngAdmissionError::InvalidPng)
        );
        let invalid_srgb = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[(png::chunk::sRGB, &[255])],
        );
        assert_eq!(
            admit_fixture(&invalid_srgb, encoded_budget()).err(),
            Some(PngAdmissionError::InvalidPng)
        );
        let srgb = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[(png::chunk::sRGB, &[0])],
        );
        assert!(admit_fixture(&srgb, encoded_budget()).is_ok());
        let nonempty_iend = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[(png::chunk::IEND, &[0])],
        );
        assert_eq!(
            admit_fixture(&nonempty_iend, encoded_budget()).err(),
            Some(PngAdmissionError::InvalidPng)
        );
        assert_eq!(
            bytes,
            encoded_fixture(
                png::ColorType::Rgb,
                png::BitDepth::Eight,
                &[1, 2, 3, 4, 5, 6],
                &[]
            )
        );
    }

    #[test]
    fn at_r08_encoded_png_preflight_is_not_original_pixel_decode_acceptance() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("header");
            writer
                .write_chunk(png::chunk::IDAT, b"not zlib pixels")
                .expect("valid CRC framing");
            writer.finish().expect("terminator");
        }
        assert!(preflight_png(&bytes, encoded_budget()).is_ok());
        assert_eq!(
            admit_fixture(&bytes, encoded_budget()).err(),
            Some(PngAdmissionError::SourceDecodeFailed)
        );
    }
    #[test]
    fn at_graphite_mix_resource_private_malformed_parents_cannot_zip_truncated_results() {
        let valid = admit_cpu_image(identity(&PIXELS), descriptor(), PIXELS.into(), budget())
            .expect("valid original input");
        let parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Normal,
            opacity: 100.,
        };
        let cap = CpuMixBudget {
            max_parent_bytes: 128,
            max_result_bytes: 64,
        };
        for actual in [3, 5] {
            let mut malformed = valid.clone();
            Arc::make_mut(&mut malformed.image)
                .data
                .resize(actual, Color::BLACK);
            assert!(
                matches!(evaluate_cpu_image_mix(&valid, &malformed, parameters, cap),
                Err(CpuMixError::Image(CpuImageResultError::InvalidPixelCount { expected: 4, actual: count })) if count == actual)
            );
            assert!(
                matches!(evaluate_cpu_image_mix(&malformed, &valid, parameters, cap),
                Err(CpuMixError::Image(CpuImageResultError::InvalidPixelCount { expected: 4, actual: count })) if count == actual)
            );
        }
        for (width, height) in [(0, 2), (2, 0)] {
            let mut malformed = valid.clone();
            let image = Arc::make_mut(&mut malformed.image);
            image.width = width;
            image.height = height;
            assert_eq!(
                evaluate_cpu_image_mix(&valid, &malformed, parameters, cap).err(),
                Some(CpuMixError::Image(CpuImageResultError::EmptyDimensions))
            );
        }
        assert_eq!(
            valid.image(),
            &Image::<Color>::from_image_data(&PIXELS, 2, 2)
        );
    }

    #[test]
    fn at_graphite_mix_resource_nonfinite_parent_and_result_are_rejected_before_publication() {
        let valid = admit_cpu_image(identity(&PIXELS), descriptor(), PIXELS.into(), budget())
            .expect("valid original input");
        let mut malformed = valid.clone();
        let data = &mut Arc::make_mut(&mut malformed.image).data;
        *data.first_mut().expect("nonempty test input") =
            Color::from_rgbaf32_unchecked(f32::NAN, 0., 0., 1.);
        let parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Normal,
            opacity: 100.,
        };
        let cap = CpuMixBudget {
            max_parent_bytes: 128,
            max_result_bytes: 64,
        };
        assert_eq!(
            evaluate_cpu_image_mix(&valid, &malformed, parameters, cap).err(),
            Some(CpuMixError::Image(CpuImageResultError::NonFiniteSample))
        );
        assert_eq!(
            evaluate_cpu_image_mix(&malformed, &valid, parameters, cap).err(),
            Some(CpuMixError::Image(CpuImageResultError::NonFiniteSample))
        );
        let output = result_item(
            1,
            1,
            vec![Color::from_rgbaf32_unchecked(0., f32::INFINITY, 0., 1.)],
        );
        assert_eq!(
            admit_cpu_pixels(output, 16).err(),
            Some(CpuImageResultError::NonFiniteSample)
        );
        assert_eq!(
            valid.image(),
            &Image::<Color>::from_image_data(&PIXELS, 2, 2)
        );
    }

    #[test]
    fn at_graphite_shared_cpu_pixels_preserve_unary_result_digest_and_full_attributes() {
        let original = result_item(2, 2, Image::<Color>::from_image_data(&PIXELS, 2, 2).data)
            .with_attribute(ATTR_TRANSFORM, DAffine2::IDENTITY)
            .with_attribute("document-tag", "editable-source".to_owned());
        let shared = admit_cpu_pixels(original.clone(), 64).expect("shared output");
        let unary = admit_cpu_result(
            identity(&PIXELS),
            CpuImageOrigin::Rgba8Rows {
                bytes: PIXELS.len(),
            },
            CpuImageEdit::Invert,
            original.clone(),
            64,
        )
        .expect("preserved unary API");
        assert_eq!(shared.image(), unary.image());
        assert_eq!(shared.descriptor(), unary.descriptor());
        assert_eq!(shared.pixels_sha256(), unary.pixels_sha256());
        assert_eq!(shared.attributes(), original.attributes());
        assert_eq!(shared.attributes(), unary.attributes());
        assert!(shared.requires_scene_render());
        assert!(unary.requires_scene_render());
        assert!(Arc::ptr_eq(
            &unary.pixels.output,
            &unary.clone().pixels.output
        ));
    }
    #[test]
    fn at_r18_original_png_network_preserves_all_admitted_integer_profiles_and_exact_transform() {
        for (color, channels) in [(png::ColorType::Rgb, 3), (png::ColorType::Rgba, 4)] {
            for (depth, bit_depth) in [(png::BitDepth::Eight, 8), (png::BitDepth::Sixteen, 16)] {
                let values = if channels == 3 {
                    vec![1_u16, 256, 0x1234, 65535, 32768, 65534]
                } else {
                    vec![1_u16, 256, 0x1234, 32768, 65535, 32768, 65534, 0]
                };
                let samples: Vec<u8> = if bit_depth == 16 {
                    values
                        .iter()
                        .flat_map(|value| value.to_be_bytes())
                        .collect()
                } else {
                    values.iter().map(|value| (value >> 8) as u8).collect()
                };
                let bytes = encoded_fixture(color, depth, &samples, &[]);
                let before = bytes.clone();
                let identity = encoded_identity(&bytes);
                let transform = [1., -0., 0., 1., 7., -4.];
                let direct = admit_encoded_png(
                    identity.clone(),
                    transform,
                    bytes.clone().into(),
                    encoded_budget(),
                )
                .expect("actual original direct decoder oracle");
                let result = futures::executor::block_on(admit_encoded_png_via_original_network(
                    identity.clone(),
                    transform,
                    bytes.clone().into(),
                    encoded_budget(),
                ))
                .expect("actual compiled original Resource and DecodeImage nodes");
                assert_eq!(result.identity(), &identity);
                assert_eq!(result.origin(), direct.origin());
                assert_eq!(result.image(), direct.image());
                assert_eq!(result.descriptor(), direct.descriptor());
                assert_eq!(
                    result.descriptor().transform.map(f64::to_bits),
                    transform.map(f64::to_bits)
                );
                assert_eq!(result.descriptor().color_space, ImageColorSpace::LinearSrgb);
                assert_eq!(result.descriptor().alpha, AlphaAssociation::Straight);
                assert_eq!(result.image().width, 2);
                assert_eq!(result.image().height, 1);
                assert_eq!(result.image().data.len(), 2);
                assert_eq!(bytes, before);
            }
        }
    }

    #[test]
    fn at_r18_original_png_network_keeps_preflight_digest_and_budget_rejection_then_recovers() {
        let bytes = encoded_fixture(
            png::ColorType::Rgba,
            png::BitDepth::Eight,
            &[0, 127, 255, 0, 255, 31, 1, 255],
            &[],
        );
        let before = bytes.clone();
        let transform = [1., 0., 0., 1., 0., 0.];
        let run = |identity, transform, budget| {
            futures::executor::block_on(admit_encoded_png_via_original_network(
                identity,
                transform,
                bytes.clone().into(),
                budget,
            ))
        };
        let mut wrong = encoded_identity(&bytes);
        wrong.content_sha256 = [0; 32];
        assert!(matches!(
            run(wrong, transform, encoded_budget()),
            Err(PngNetworkAdmissionError::Admission(
                PngAdmissionError::Image(ImageAdmissionError::DigestMismatch)
            ))
        ));
        let mut invalid = transform;
        invalid[4] = f64::NAN;
        assert!(matches!(
            run(encoded_identity(&bytes), invalid, encoded_budget()),
            Err(PngNetworkAdmissionError::Admission(
                PngAdmissionError::Image(ImageAdmissionError::InvalidTransform)
            ))
        ));
        let mut budget = encoded_budget();
        budget.max_encoded_bytes = bytes.len() - 1;
        assert!(matches!(
            run(encoded_identity(&bytes), transform, budget),
            Err(PngNetworkAdmissionError::Admission(
                PngAdmissionError::Image(ImageAdmissionError::InputBudgetExceeded)
            ))
        ));
        budget = encoded_budget();
        budget.max_pixel_bytes = 2 * (4 + 2 * size_of::<Color>()) - 1;
        assert!(matches!(
            run(encoded_identity(&bytes), transform, budget),
            Err(PngNetworkAdmissionError::Admission(
                PngAdmissionError::Image(ImageAdmissionError::DecodedBudgetExceeded)
            ))
        ));
        budget.max_pixel_bytes += 1;
        let result = run(encoded_identity(&bytes), transform, budget)
            .expect("exact admitted pixel-buffer cap recovers");
        assert_eq!(result.image().data.len(), 2);
        assert_eq!(bytes, before);
    }

    #[test]
    fn at_r18_original_png_network_rejects_real_empty_decoder_fallback_after_valid_framing() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("header");
            writer
                .write_chunk(png::chunk::IDAT, b"not zlib pixels")
                .expect("valid CRC framing");
            writer.finish().expect("terminator");
        }
        let before = bytes.clone();
        assert!(preflight_png(&bytes, encoded_budget()).is_ok());
        let result = futures::executor::block_on(admit_encoded_png_via_original_network(
            encoded_identity(&bytes),
            [1., 0., 0., 1., 0., 0.],
            bytes.clone().into(),
            encoded_budget(),
        ));
        assert!(matches!(
            result,
            Err(PngNetworkAdmissionError::Admission(
                PngAdmissionError::SourceDecodeFailed
            ))
        ));
        assert_eq!(bytes, before);
    }

    #[test]
    fn at_r18_original_png_network_fresh_owners_isolate_results_and_release_supplied_bytes() {
        let first = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 255, 0, 127],
            &[],
        );
        let second = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[255, 0, 0, 0, 255, 0],
            &[],
        );
        let source: Arc<[u8]> = first.clone().into();
        let supplied = Arc::downgrade(&source);
        let old = futures::executor::block_on(admit_encoded_png_via_original_network(
            encoded_identity(&first),
            [1., 0., 0., 1., 0., 0.],
            source,
            encoded_budget(),
        ))
        .expect("first real network output");
        assert!(supplied.upgrade().is_none());
        let old_pixels = old.image().data.clone();
        for bytes in [&second, &first, &second] {
            let current = futures::executor::block_on(admit_encoded_png_via_original_network(
                encoded_identity(bytes),
                [1., 0., 0., 1., 0., 0.],
                bytes.clone().into(),
                encoded_budget(),
            ))
            .expect("fresh original memory store and executor");
            let expected = admit_fixture(bytes, encoded_budget()).expect("original direct oracle");
            assert_eq!(current.image(), expected.image());
            assert_eq!(current.identity(), expected.identity());
            assert_eq!(old.image().data, old_pixels);
            assert!(!Arc::ptr_eq(&old.image, &current.image));
        }
    }

    fn explicit_black_and_white_parameters() -> CpuBlackAndWhiteParameters {
        CpuBlackAndWhiteParameters {
            use_tint: false,
            tint_gamma_srgb: [0.8, 0.3, 0.1, 0.2],
            reds: 41.,
            yellows: 62.,
            greens: 43.,
            cyans: 64.,
            blues: 25.,
            magentas: 86.,
        }
    }

    #[test]
    fn at_gra_node_144_2_cpu_edit_retains_source_provenance_and_exact_connectors() {
        use graphite_core_types::Node as _;
        use graphite_core_types::registry::FutureWrapperNode;
        use graphite_core_types::value::ClonedNode;
        let source = admit_cpu_image(identity(&PIXELS), descriptor(), PIXELS.into(), budget())
            .expect("admitted source");
        let before = source.image().clone();
        for use_tint in [false, true] {
            let mut parameters = explicit_black_and_white_parameters();
            parameters.use_tint = use_tint;
            let operation = CpuImageEdit::BlackAndWhite(parameters);
            let result = evaluate_cpu_image_edit(&source, operation, 64).expect("source edit");
            for (actual, original) in result.image().data.iter().zip(&before.data) {
                let [red, green, blue, alpha] = parameters.tint_gamma_srgb;
                let node = graphite_raster_nodes::adjustments::BlackAndWhiteNode::new(
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(*original))),
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(use_tint))),
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(
                        Color::from_gamma_srgb_channels(red, green, blue, alpha),
                    ))),
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(
                        parameters.reds,
                    ))),
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(
                        parameters.yellows,
                    ))),
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(
                        parameters.greens,
                    ))),
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(
                        parameters.cyans,
                    ))),
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(
                        parameters.blues,
                    ))),
                    FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(
                        parameters.magentas,
                    ))),
                );
                let expected = futures::executor::block_on(node.eval(()));
                assert_eq!(*actual, *expected.element());
                assert_eq!(actual.a().to_bits(), original.a().to_bits());
            }
            assert_eq!(result.operation(), operation);
            assert_eq!(
                result.operation().source_identifier(),
                "raster_nodes::adjustments::BlackAndWhiteNode"
            );
            assert_eq!(result.input_identity(), source.identity());
            assert_eq!(result.input_origin(), source.origin());
            assert_eq!(result.descriptor().transform, source.descriptor().transform);
            assert_eq!(source.image(), &before);
            let repeated = evaluate_cpu_image_edit(&source, operation, 64).expect("repeat");
            assert_eq!(result.pixels_sha256(), repeated.pixels_sha256());
            assert_eq!(
                evaluate_cpu_image_edit(&source, operation, 63).err(),
                Some(CpuImageResultError::ResultBudgetExceeded)
            );
        }
    }

    #[test]
    fn at_gra_node_144_2_operation_equality_preserves_parameter_encoding() {
        let parameters = explicit_black_and_white_parameters();
        for connector in 0..11 {
            let mut changed = parameters;
            match connector {
                0 => changed.use_tint = !changed.use_tint,
                1..=4 => changed.tint_gamma_srgb[connector - 1] += 0.1,
                5 => changed.reds += 1.,
                6 => changed.yellows += 1.,
                7 => changed.greens += 1.,
                8 => changed.cyans += 1.,
                9 => changed.blues += 1.,
                10 => changed.magentas += 1.,
                _ => unreachable!(),
            }
            assert_ne!(
                CpuImageEdit::BlackAndWhite(parameters),
                CpuImageEdit::BlackAndWhite(changed)
            );
        }
        let mut positive_zero = parameters;
        positive_zero.reds = 0.;
        let mut negative_zero = positive_zero;
        negative_zero.reds = -0.;
        assert_ne!(positive_zero, negative_zero);
        let mut nan = parameters;
        nan.reds = f64::from_bits(0x7ff8_0000_0000_0001);
        assert_eq!(nan, nan);
        let mut different_nan = nan;
        different_nan.reds = f64::from_bits(0x7ff8_0000_0000_0002);
        assert_ne!(nan, different_nan);
    }

    #[test]
    fn at_gra_node_145_2_cpu_edit_retains_original_parameters_and_parent_provenance() {
        let source = admit_cpu_image(identity(&PIXELS), descriptor(), PIXELS.into(), budget())
            .expect("admitted source");
        let before = source.image().clone();
        for use_classic in [false, true] {
            let parameters = CpuBrightnessContrastParameters {
                brightness: 63.,
                contrast: -37.,
                use_classic,
                classic_pivot: 81.,
            };
            let operation = CpuImageEdit::BrightnessContrast(parameters);
            let result = evaluate_cpu_image_edit(&source, operation, 64).expect("source edit");
            for (actual, original) in result.image().data.iter().zip(&before.data) {
                let expected = crate::raster::evaluate_brightness_contrast_color(
                    Item::new_from_element(*original),
                    crate::raster::BrightnessContrastParameters {
                        brightness: Item::new_from_element(parameters.brightness),
                        contrast: Item::new_from_element(parameters.contrast),
                        use_classic: Item::new_from_element(parameters.use_classic),
                        classic_pivot: Item::new_from_element(parameters.classic_pivot),
                    },
                )
                .expect("original color evaluator");
                assert_eq!(actual, expected.element());
                assert_eq!(actual.a().to_bits(), original.a().to_bits());
            }
            assert_eq!(result.operation(), operation);
            assert_eq!(result.input_identity(), source.identity());
            assert_eq!(result.input_origin(), source.origin());
            assert_eq!(result.descriptor().transform, source.descriptor().transform);
            assert_eq!(
                operation.source_identifier(),
                "raster_nodes::adjustments::BrightnessContrastNode"
            );
            assert_eq!(source.image(), &before);
            assert_eq!(
                evaluate_cpu_image_edit(&source, operation, 63).err(),
                Some(CpuImageResultError::ResultBudgetExceeded)
            );
            let recovered = evaluate_cpu_image_edit(&source, operation, 64).expect("cap recovery");
            assert_eq!(recovered.pixels_sha256(), result.pixels_sha256());
        }
    }

    #[test]
    fn at_gra_node_145_2_operation_equality_preserves_exact_parameter_bits() {
        let parameters = CpuBrightnessContrastParameters {
            brightness: 0.,
            contrast: 0.,
            use_classic: false,
            classic_pivot: 127.,
        };
        for connector in 0..4 {
            let mut changed = parameters;
            match connector {
                0 => changed.brightness = 1.,
                1 => changed.contrast = 1.,
                2 => changed.use_classic = true,
                3 => changed.classic_pivot = 128.,
                _ => unreachable!(),
            }
            assert_ne!(
                CpuImageEdit::BrightnessContrast(parameters),
                CpuImageEdit::BrightnessContrast(changed)
            );
        }
        let mut negative_zero = parameters;
        negative_zero.brightness = -0.;
        assert_ne!(parameters, negative_zero);
        let mut nan = parameters;
        nan.contrast = f64::from_bits(0x7ff8_0000_0000_0001);
        assert_eq!(nan, nan);
        let mut different_nan = nan;
        different_nan.contrast = f64::from_bits(0x7ff8_0000_0000_0002);
        assert_ne!(nan, different_nan);
    }

    #[test]
    fn at_r08_encoded_source_seal_retains_exact_arc_identity_header_and_input_bytes() {
        let workflow = b"workflow\0original indexed producer text";
        let before = encoded_fixture(
            png::ColorType::Rgba,
            png::BitDepth::Eight,
            &[231, 17, 83, 0, 19, 223, 71, 127],
            &[(png::chunk::tEXt, workflow)],
        );
        let bytes: Arc<[u8]> = before.clone().into();
        let weak = Arc::downgrade(&bytes);
        let identity = encoded_identity(&bytes);
        let source = prepare_encoded_png_source(identity.clone(), bytes.clone(), encoded_budget())
            .expect("encoded envelope");
        assert!(Arc::ptr_eq(source.encoded_bytes(), &bytes));
        assert_eq!(source.identity(), &identity);
        assert_eq!((source.width(), source.height()), (2, 1));
        assert_eq!(source.encoded_bytes().as_ref(), before.as_slice());
        assert_eq!(bytes.as_ref(), before.as_slice());
        drop(bytes);
        assert!(weak.upgrade().is_some());
        drop(source);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn at_r18_encoded_source_seal_does_not_claim_original_decode_acceptance() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("header");
            writer
                .write_chunk(png::chunk::IDAT, b"not zlib pixels")
                .expect("valid CRC framing");
            writer.finish().expect("terminator");
        }
        let before = bytes.clone();
        let bytes: Arc<[u8]> = bytes.into();
        let source =
            prepare_encoded_png_source(encoded_identity(&bytes), bytes.clone(), encoded_budget())
                .expect("valid envelope without pixel decode");
        assert!(Arc::ptr_eq(source.encoded_bytes(), &bytes));
        assert_eq!((source.width(), source.height()), (2, 1));
        assert_eq!(
            admit_encoded_png(
                source.identity().clone(),
                [1., 0., 0., 1., 0., 0.],
                source.encoded_bytes().clone(),
                encoded_budget(),
            )
            .err(),
            Some(PngAdmissionError::SourceDecodeFailed)
        );
        assert_eq!(source.encoded_bytes().as_ref(), before.as_slice());
        let valid = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[],
        );
        let recovered = prepare_encoded_png_source(
            encoded_identity(&valid),
            valid.clone().into(),
            encoded_budget(),
        )
        .expect("fresh valid envelope recovers");
        assert!(admit_fixture(recovered.encoded_bytes(), encoded_budget()).is_ok());
        assert_eq!(valid.as_slice(), recovered.encoded_bytes().as_ref());
    }

    #[test]
    fn at_r08_encoded_source_seal_checks_sha_caps_profile_and_recovers_exact_limits() {
        let workflow = b"workflow\0unaltered";
        let bytes: Arc<[u8]> = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[(png::chunk::tEXt, workflow)],
        )
        .into();
        let before = bytes.to_vec();
        let identity = encoded_identity(&bytes);
        let exact = EncodedPngBudget {
            max_encoded_bytes: bytes.len(),
            max_pixel_bytes: 2 * (3 + 2 * size_of::<Color>()),
            max_metadata_bytes: workflow.len(),
        };
        let mut wrong = identity.clone();
        wrong.content_sha256[0] ^= 1;
        assert_eq!(
            prepare_encoded_png_source(wrong, bytes.clone(), exact).err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::DigestMismatch
            ))
        );
        for maximum in [0, bytes.len() - 1] {
            assert_eq!(
                prepare_encoded_png_source(
                    identity.clone(),
                    bytes.clone(),
                    EncodedPngBudget {
                        max_encoded_bytes: maximum,
                        ..exact
                    },
                )
                .err(),
                Some(PngAdmissionError::Image(
                    ImageAdmissionError::InputBudgetExceeded
                ))
            );
        }
        assert_eq!(
            prepare_encoded_png_source(
                identity.clone(),
                bytes.clone(),
                EncodedPngBudget {
                    max_pixel_bytes: exact.max_pixel_bytes - 1,
                    ..exact
                },
            )
            .err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::DecodedBudgetExceeded
            ))
        );
        assert_eq!(
            prepare_encoded_png_source(
                identity.clone(),
                bytes.clone(),
                EncodedPngBudget {
                    max_metadata_bytes: workflow.len() - 1,
                    ..exact
                },
            )
            .err(),
            Some(PngAdmissionError::MetadataBudgetExceeded)
        );
        let grayscale = encoded_fixture(
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            &[23, 245],
            &[],
        );
        assert_eq!(
            prepare_encoded_png_source(
                encoded_identity(&grayscale),
                grayscale.into(),
                encoded_budget(),
            )
            .err(),
            Some(PngAdmissionError::UnsupportedPixelFormat)
        );
        let source = prepare_encoded_png_source(identity.clone(), bytes.clone(), exact)
            .expect("exact encoded, pixel and metadata budgets recover");
        assert!(Arc::ptr_eq(source.encoded_bytes(), &bytes));
        assert_eq!(source.identity(), &identity);
        assert_eq!((source.width(), source.height()), (2, 1));
        assert_eq!(source.encoded_bytes().as_ref(), before.as_slice());
        assert_eq!(bytes.as_ref(), before.as_slice());
    }

    #[test]
    fn at_r08_encoded_source_refactor_preserves_decoded_validation_precedence() {
        let bytes: Arc<[u8]> = encoded_fixture(
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3, 4, 5, 6],
            &[],
        )
        .into();
        let mut wrong = encoded_identity(&bytes);
        wrong.content_sha256[0] ^= 1;
        let zero = EncodedPngBudget {
            max_encoded_bytes: 0,
            ..encoded_budget()
        };
        let invalid_transform = [1., 0., 0., 1., f64::NAN, 0.];
        let mut missing = wrong.clone();
        missing.asset_id = " ".into();
        assert_eq!(
            prepare_encoded_png_source(missing.clone(), bytes.clone(), zero).err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::MissingIdentity
            ))
        );
        assert_eq!(
            admit_encoded_png(missing, invalid_transform, bytes.clone(), zero).err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::MissingIdentity
            ))
        );
        assert_eq!(
            admit_encoded_png(wrong.clone(), invalid_transform, bytes.clone(), zero).err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::InvalidTransform
            ))
        );
        assert_eq!(
            prepare_encoded_png_source(wrong.clone(), bytes.clone(), zero).err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::InputBudgetExceeded
            ))
        );
        assert_eq!(
            admit_encoded_png(wrong, [1., 0., 0., 1., 0., 0.], bytes, zero).err(),
            Some(PngAdmissionError::Image(
                ImageAdmissionError::InputBudgetExceeded
            ))
        );
    }
}
