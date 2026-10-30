use crate::{
    MetadataDocument, MetadataError, MetadataLimits, MetadataWritePolicy,
    metadata::{
        PngSourcePixelError, PngSourcePixelFormat, PngSourceTransparency, png_source_pixel_format,
    },
};
use image::{
    DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder, ImageError, ImageFormat,
    ImageReader, Limits, codecs::png::PngEncoder,
};
use sim_tensor::{
    CancellationToken, CpuBackend, ExecutionContext, ImageTensor, Layout, MemoryFormatReference,
    ResizeCrop, ResizeMode, Tensor, TensorError,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_shape_layout_transform_01::tensor_unsqueeze_exact_native,
    generated_shape_layout_transform_02::torch_cat_with_context_exact_native,
    generated_shape_layout_transform_03::tensor_permute_exact_native,
    generated_storage_dtype_device_01::contiguous_with_context_exact_native,
};
use std::{collections::BTreeMap, io::Cursor};
use thiserror::Error;

pub const DEFAULT_MAX_PNG_BYTES: usize = 256 * 1024 * 1024;
pub const DEFAULT_MAX_PNG_DIMENSION: u32 = 32_768;
pub const DEFAULT_MAX_PNG_PIXELS: u64 = 268_435_456;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PngLimits {
    pub max_input_bytes: usize,
    pub max_dimension: u32,
    pub max_pixels: u64,
    pub max_decoder_allocation_bytes: u64,
}

impl Default for PngLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_PNG_BYTES,
            max_dimension: DEFAULT_MAX_PNG_DIMENSION,
            max_pixels: DEFAULT_MAX_PNG_PIXELS,
            max_decoder_allocation_bytes: DEFAULT_MAX_PNG_BYTES as u64,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedPng {
    pub width: u32,
    pub height: u32,
    pub pixels_bhwc: Vec<f32>,
    pub mask_bhw: Vec<f32>,
    pub mask_width: u32,
    pub mask_height: u32,
    pub has_alpha: bool,
    pub metadata: MetadataDocument,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PngError {
    #[error("PNG input is {actual} bytes, exceeding the {limit}-byte limit")]
    InputTooLarge { actual: usize, limit: usize },
    #[error("PNG dimensions {width}x{height} exceed configured limits")]
    Dimensions { width: u32, height: u32 },
    #[error("PNG pixel or byte size overflowed")]
    SizeOverflow,
    #[error("PNG IMAGE data has {actual} values, expected {expected}")]
    PixelCount { expected: usize, actual: usize },
    #[error("PNG batch index {index} is outside a batch of {batch}")]
    BatchIndex { index: u64, batch: u64 },
    #[error("downloaded image color type {color:?} has no verified source conversion")]
    UnsupportedDownloadedColor { color: image::ColorType },
    #[error("PNG codec failed: {0}")]
    Codec(String),
    #[error("PNG metadata failed: {0}")]
    Metadata(String),
    #[error("PNG allocation failed: {0}")]
    Allocation(String),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

impl From<ImageError> for PngError {
    fn from(error: ImageError) -> Self {
        Self::Codec(error.to_string())
    }
}

impl From<MetadataError> for PngError {
    fn from(error: MetadataError) -> Self {
        Self::Metadata(error.to_string())
    }
}

pub fn decode_png(bytes: &[u8], limits: PngLimits) -> Result<DecodedPng, PngError> {
    if bytes.len() > limits.max_input_bytes {
        return Err(PngError::InputTooLarge {
            actual: bytes.len(),
            limit: limits.max_input_bytes,
        });
    }
    let mut reader = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Png);
    let mut decoder_limits = Limits::default();
    decoder_limits.max_image_width = Some(limits.max_dimension);
    decoder_limits.max_image_height = Some(limits.max_dimension);
    decoder_limits.max_alloc = Some(limits.max_decoder_allocation_bytes);
    reader.limits(decoder_limits);
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    let width = image.width();
    let height = image.height();
    validate_dimensions(width, height, limits)?;
    let has_alpha = image.color().has_alpha();
    let rgba = image.into_rgba8();
    let pixel_count = checked_pixel_count(width, height)?;
    let rgb_value_count = pixel_count.checked_mul(3).ok_or(PngError::SizeOverflow)?;
    let mut pixels_bhwc = Vec::new();
    pixels_bhwc
        .try_reserve_exact(rgb_value_count)
        .map_err(|error| PngError::Allocation(error.to_string()))?;
    let mask_value_count = if has_alpha { pixel_count } else { 64 * 64 };
    let mut mask_bhw = Vec::new();
    mask_bhw
        .try_reserve_exact(mask_value_count)
        .map_err(|error| PngError::Allocation(error.to_string()))?;
    for pixel in rgba.pixels() {
        pixels_bhwc.extend(pixel.0[..3].iter().map(|value| f32::from(*value) / 255.0));
        if has_alpha {
            mask_bhw.push(1.0 - f32::from(pixel.0[3]) / 255.0);
        }
    }
    if !has_alpha {
        mask_bhw.resize(mask_value_count, 0.0);
    }
    let metadata = MetadataDocument::parse(
        bytes,
        Some("image.png"),
        Some("image/png"),
        MetadataLimits {
            max_input_bytes: limits.max_input_bytes,
            ..MetadataLimits::default()
        },
    )?;
    Ok(DecodedPng {
        width,
        height,
        pixels_bhwc,
        mask_bhw,
        mask_width: if has_alpha { width } else { 64 },
        mask_height: if has_alpha { height } else { 64 },
        has_alpha,
        metadata,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedDownloadedImage {
    pub width: u32,
    pub height: u32,
    pub pixels_bhwc: Vec<f32>,
}

pub fn decode_downloaded_image_rgba(
    bytes: &[u8],
    limits: PngLimits,
    cancellation: &CancellationToken,
) -> Result<DecodedDownloadedImage, PngError> {
    cancellation.check().map_err(TensorError::from)?;
    if bytes.len() > limits.max_input_bytes {
        return Err(PngError::InputTooLarge {
            actual: bytes.len(),
            limit: limits.max_input_bytes,
        });
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| PngError::Codec(error.to_string()))?;
    let format = reader.format();
    let mut decoder_limits = Limits::default();
    decoder_limits.max_image_width = Some(limits.max_dimension);
    decoder_limits.max_image_height = Some(limits.max_dimension);
    decoder_limits.max_alloc = Some(limits.max_decoder_allocation_bytes);
    reader.limits(decoder_limits);
    let decoder: Box<dyn ImageDecoder + '_> = if format == Some(ImageFormat::Jpeg) {
        Box::new(DownloadedSourceJpeg::new(bytes, limits)?)
    } else {
        Box::new(reader.into_decoder()?)
    };
    let color = decoder.color_type();
    let high_depth = color.bits_per_pixel() / u16::from(color.channel_count()) > 8;
    let source_pixel_format = if high_depth {
        if format != Some(ImageFormat::Png) {
            return Err(PngError::UnsupportedDownloadedColor { color });
        }
        Some(
            png_source_pixel_format(bytes, cancellation).map_err(|error| match error {
                PngSourcePixelError::Malformed(message) => PngError::Codec(message),
                PngSourcePixelError::Cancelled => PngError::Tensor(TensorError::Cancelled),
            })?,
        )
    } else {
        None
    };
    let (width, height) = decoder.dimensions();
    validate_dimensions(width, height, limits)?;
    let value_count = checked_pixel_count(width, height)?
        .checked_mul(4)
        .ok_or(PngError::SizeOverflow)?;
    let output_bytes = value_count.checked_mul(4).ok_or(PngError::SizeOverflow)?;
    if output_bytes as u64 > limits.max_decoder_allocation_bytes {
        return Err(PngError::Allocation(
            "decoded RGBA F32 bytes exceed the image allocation bound".to_owned(),
        ));
    }
    cancellation.check().map_err(TensorError::from)?;
    // The provider source converts directly to RGBA without EXIF transposition.
    // Preserve alpha bytes here rather than round-tripping the LoadImage inverse mask.
    let image = DynamicImage::from_decoder(decoder)?;
    let mut pixels_bhwc = Vec::new();
    pixels_bhwc
        .try_reserve_exact(value_count)
        .map_err(|error| PngError::Allocation(error.to_string()))?;
    if let Some(source_pixel_format) = source_pixel_format {
        if source_pixel_format.bit_depth != 16 {
            return Err(PngError::UnsupportedDownloadedColor { color });
        }
        let samples = image
            .as_flat_samples_u16()
            .ok_or(PngError::UnsupportedDownloadedColor { color })?;
        let channels = usize::from(color.channel_count());
        let row_values = usize::try_from(width)
            .map_err(|_| PngError::SizeOverflow)?
            .checked_mul(channels)
            .ok_or(PngError::SizeOverflow)?;
        for row in samples.samples.chunks_exact(row_values) {
            cancellation.check().map_err(TensorError::from)?;
            for pixel in row.chunks_exact(channels) {
                let rgba = downloaded_png16_rgba(pixel, source_pixel_format)?;
                pixels_bhwc.extend(rgba.map(|value| f32::from(value) / 255.0));
            }
        }
    } else {
        let rgba = image.into_rgba8();
        for row in rgba.rows() {
            cancellation.check().map_err(TensorError::from)?;
            for pixel in row {
                pixels_bhwc.extend(pixel.0.iter().map(|value| f32::from(*value) / 255.0));
            }
        }
    }
    if pixels_bhwc.len() != value_count {
        return Err(PngError::PixelCount {
            expected: value_count,
            actual: pixels_bhwc.len(),
        });
    }
    cancellation.check().map_err(TensorError::from)?;
    Ok(DecodedDownloadedImage {
        width,
        height,
        pixels_bhwc,
    })
}

struct DownloadedSourceJpeg<'a> {
    decoder: zune_jpeg::JpegDecoder<zune_jpeg::zune_core::bytestream::ZCursor<&'a [u8]>>,
    dimensions: (u32, u32),
    color: image::ColorType,
}

impl<'a> DownloadedSourceJpeg<'a> {
    fn new(bytes: &'a [u8], limits: PngLimits) -> Result<Self, PngError> {
        use zune_jpeg::zune_core::{
            bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions,
        };
        let options = DecoderOptions::default()
            .set_strict_mode(true)
            .set_max_width(limits.max_dimension as usize)
            .set_max_height(limits.max_dimension as usize);
        let mut decoder = zune_jpeg::JpegDecoder::new_with_options_and_pixel_mode(
            ZCursor::new(bytes),
            options,
            zune_jpeg::JpegPixelMode::PillowSource,
        );
        decoder
            .decode_headers()
            .map_err(|error| PngError::Codec(error.to_string()))?;
        let information = decoder
            .info()
            .ok_or_else(|| PngError::Codec("JPEG headers unavailable".to_owned()))?;
        let dimensions = (u32::from(information.width), u32::from(information.height));
        validate_dimensions(dimensions.0, dimensions.1, limits)?;
        // Progressive coefficient planes and MCU padding are allocated inside the decoder.
        // Bound their conservative worst-case envelope before any entropy decoding starts.
        let padded_width = u64::from(dimensions.0).div_ceil(32) * 32;
        let padded_height = u64::from(dimensions.1).div_ceil(32) * 32;
        let working_bytes = padded_width
            .checked_mul(padded_height)
            .and_then(|value| value.checked_mul(16))
            .and_then(|value| value.checked_add(padded_width.checked_mul(2048)?))
            .and_then(|value| value.checked_add(bytes.len() as u64))
            .ok_or(PngError::SizeOverflow)?;
        if working_bytes > limits.max_decoder_allocation_bytes {
            return Err(PngError::Allocation(
                "JPEG working buffers exceed the image allocation bound".to_owned(),
            ));
        }
        let (output, color) = match decoder.input_colorspace() {
            Some(ColorSpace::Luma) => (ColorSpace::Luma, image::ColorType::L8),
            Some(ColorSpace::RGB | ColorSpace::YCbCr | ColorSpace::CMYK | ColorSpace::YCCK) => {
                (ColorSpace::RGB, image::ColorType::Rgb8)
            }
            _ => {
                return Err(PngError::Codec(
                    "Unsupported source JPEG colorspace".to_owned(),
                ));
            }
        };
        decoder.set_options(decoder.options().jpeg_set_out_colorspace(output));
        Ok(Self {
            decoder,
            dimensions,
            color,
        })
    }
}

impl ImageDecoder for DownloadedSourceJpeg<'_> {
    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }
    fn color_type(&self) -> image::ColorType {
        self.color
    }
    fn read_image(mut self, output: &mut [u8]) -> Result<(), ImageError> {
        if output.len() as u64 != self.total_bytes() {
            return Err(ImageError::Decoding(image::error::DecodingError::new(
                ImageFormat::Jpeg.into(),
                "JPEG output length disagrees with dimensions",
            )));
        }
        self.decoder.decode_into(output).map_err(|error| {
            ImageError::Decoding(image::error::DecodingError::new(
                ImageFormat::Jpeg.into(),
                error,
            ))
        })
    }
    fn read_image_boxed(self: Box<Self>, output: &mut [u8]) -> Result<(), ImageError> {
        (*self).read_image(output)
    }
}

fn downloaded_png16_rgba(pixel: &[u16], source: PngSourcePixelFormat) -> Result<[u8; 4], PngError> {
    // Pillow clips I;16 grayscale, but PNG16 color and alpha lose their low bytes.
    let mut rgba = match (source.color_type, pixel) {
        (0, [gray] | [gray, _]) => {
            let gray = (*gray).min(255).to_le_bytes()[0];
            [gray, gray, gray, 255]
        }
        (2, [red, green, blue] | [red, green, blue, _]) => [
            red.to_be_bytes()[0],
            green.to_be_bytes()[0],
            blue.to_be_bytes()[0],
            255,
        ],
        (4, [gray, alpha]) => {
            let gray = gray.to_be_bytes()[0];
            [gray, gray, gray, alpha.to_be_bytes()[0]]
        }
        (6, [red, green, blue, alpha]) => [
            red.to_be_bytes()[0],
            green.to_be_bytes()[0],
            blue.to_be_bytes()[0],
            alpha.to_be_bytes()[0],
        ],
        _ => {
            return Err(PngError::Codec(
                "PNG source and decoded color types disagree".to_owned(),
            ));
        }
    };
    // Pillow applies tRNS after conversion, comparing low-byte keys to RGBA8.
    let transparent = match source.transparency {
        Some(PngSourceTransparency::Grayscale(gray)) => rgba[0] == gray.to_le_bytes()[0],
        Some(PngSourceTransparency::Rgb([red, green, blue])) => {
            rgba[..3]
                == [
                    red.to_le_bytes()[0],
                    green.to_le_bytes()[0],
                    blue.to_le_bytes()[0],
                ]
        }
        None => false,
    };
    if transparent {
        rgba[3] = 0;
    }
    Ok(rgba)
}

#[allow(clippy::too_many_arguments)]
pub fn encode_png_frame(
    pixels_bhwc: &[f32],
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    batch_index: u64,
    metadata: &BTreeMap<String, String>,
    limits: PngLimits,
) -> Result<Vec<u8>, PngError> {
    encode_png_frame_with_policy(
        pixels_bhwc,
        batch,
        height,
        width,
        channels,
        batch_index,
        metadata,
        MetadataWritePolicy::default(),
        limits,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_png_frame_with_policy(
    pixels_bhwc: &[f32],
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    batch_index: u64,
    metadata: &BTreeMap<String, String>,
    metadata_policy: MetadataWritePolicy,
    limits: PngLimits,
) -> Result<Vec<u8>, PngError> {
    let (frame, width, height, values_per_frame) = checked_png_frame(
        pixels_bhwc,
        batch,
        height,
        width,
        channels,
        batch_index,
        limits,
    )?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(values_per_frame)
        .map_err(|error| PngError::Allocation(error.to_string()))?;
    pixels.extend(frame.iter().map(|value| quantize_channel(*value)));
    encode_pixels_with_metadata(
        &pixels,
        width,
        height,
        channels,
        metadata,
        metadata_policy,
        limits,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_png_frame_with_policy_and_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    pixels_bhwc: &[f32],
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    batch_index: u64,
    metadata: &BTreeMap<String, String>,
    metadata_policy: MetadataWritePolicy,
    limits: PngLimits,
) -> Result<Vec<u8>, PngError> {
    context.check()?;
    let (frame, width, height, values_per_frame) = checked_png_frame(
        pixels_bhwc,
        batch,
        height,
        width,
        channels,
        batch_index,
        limits,
    )?;
    let mut pixels = backend.workspace_vec::<u8>(context, values_per_frame)?;
    for (index, value) in frame.iter().copied().enumerate() {
        if index.is_multiple_of(4_096) {
            context.check()?;
        }
        pixels.try_push(quantize_channel(value))?;
    }
    encode_pixels_with_metadata(
        &pixels,
        width,
        height,
        channels,
        metadata,
        metadata_policy,
        limits,
    )
}

pub fn encode_api_png_frame_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    image: &ImageTensor,
    batch_index: u64,
    maximum_pixels: Option<u64>,
    limits: PngLimits,
) -> Result<Vec<u8>, PngError> {
    context.check()?;
    let (batch, height, width, channels) = image.dimensions()?;
    if batch_index >= batch {
        return Err(PngError::BatchIndex {
            index: batch_index,
            batch,
        });
    }
    let input_pixels = width.checked_mul(height).ok_or(PngError::SizeOverflow)?;
    if input_pixels == 0 {
        return Err(PngError::Dimensions {
            width: 0,
            height: 0,
        });
    }
    let frame = narrow_method_exact_native(
        image.tensor(),
        0,
        i64::try_from(batch_index).map_err(|_| PngError::SizeOverflow)?,
        1,
        context.cancellation,
    );
    context.check()?;
    let mut frame =
        ImageTensor::from_tensor(frame.map_err(|error| PngError::Codec(error.to_string()))?)?;
    if let Some((target_width, target_height)) = maximum_pixels
        .map(|maximum| api_downscale_dimensions(width, height, maximum))
        .transpose()?
        .flatten()
    {
        validate_dimensions(
            u32::try_from(target_width).map_err(|_| PngError::SizeOverflow)?,
            u32::try_from(target_height).map_err(|_| PngError::SizeOverflow)?,
            limits,
        )?;
        frame = frame.resize(
            target_width,
            target_height,
            ResizeMode::Lanczos,
            ResizeCrop::Disabled,
            backend,
            context,
        )?;
        if channels == 1 {
            // Source Lanczos drops its grayscale channel axis before movedim,
            // so API conversion transposes this resized grayscale array.
            frame = api_transpose_grayscale(backend, context, &frame)?;
        }
    }
    let (_, height, width, channels) = frame.dimensions()?;
    let dimensions = [height, width, channels]
        .into_iter()
        .filter(|dimension| maximum_pixels.is_none() || *dimension != 1)
        .collect::<Vec<_>>();
    let (height, width, channels) = match dimensions.as_slice() {
        [height] => (*height, 1, 1),
        [height, width] => (*height, *width, 1),
        [height, width, channels @ (3 | 4)] => (*height, *width, *channels),
        _ => {
            return Err(PngError::Codec(
                "source API image shape cannot be encoded by Pillow".to_owned(),
            ));
        }
    };
    let width = u32::try_from(width).map_err(|_| PngError::SizeOverflow)?;
    let height = u32::try_from(height).map_err(|_| PngError::SizeOverflow)?;
    validate_dimensions(width, height, limits)?;
    let pixels = frame.numpy_image_bytes(backend, context)?;
    let encoded = encode_pixels_with_metadata(
        &pixels,
        width,
        height,
        channels,
        &BTreeMap::new(),
        MetadataWritePolicy {
            metadata_enabled: false,
        },
        limits,
    )?;
    context.check()?;
    Ok(encoded)
}

pub fn encode_api_png_mask_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    mask: &Tensor,
    maximum_pixels: u64,
    limits: PngLimits,
) -> Result<Vec<u8>, PngError> {
    context.check()?;
    let mask = api_normalize_mask(mask, context)?;
    let first = narrow_method_exact_native(&mask, 0, 0, 1, context.cancellation)
        .map_err(|error| PngError::Codec(error.to_string()))?;
    let image = api_mask_rgb(backend, context, &first)?;
    encode_api_png_frame_with_context(backend, context, &image, 0, Some(maximum_pixels), limits)
}

pub fn prepare_api_mask_batch_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    mask: &Tensor,
    maximum_pixels: u64,
    limits: PngLimits,
) -> Result<ImageTensor, PngError> {
    context.check()?;
    let mask = api_normalize_mask(mask, context)?;
    let image = api_mask_rgb(backend, context, &mask)?;
    let (_, height, width, _) = image.dimensions()?;
    if let Some((width, height)) = api_downscale_dimensions(width, height, maximum_pixels)? {
        validate_dimensions(
            u32::try_from(width).map_err(|_| PngError::SizeOverflow)?,
            u32::try_from(height).map_err(|_| PngError::SizeOverflow)?,
            limits,
        )?;
        Ok(image.resize(
            width,
            height,
            ResizeMode::Lanczos,
            ResizeCrop::Disabled,
            backend,
            context,
        )?)
    } else {
        validate_dimensions(
            u32::try_from(width).map_err(|_| PngError::SizeOverflow)?,
            u32::try_from(height).map_err(|_| PngError::SizeOverflow)?,
            limits,
        )?;
        Ok(image)
    }
}

fn api_normalize_mask(mask: &Tensor, context: &ExecutionContext<'_>) -> Result<Tensor, PngError> {
    context.check()?;
    match mask.descriptor().rank() {
        2 => tensor_unsqueeze_exact_native(mask, 0, context.cancellation)
            .map_err(|error| PngError::Codec(error.to_string())),
        3 => Ok(mask.clone()),
        _ => Err(PngError::Codec(
            "source API mask must have rank two or three".to_owned(),
        )),
    }
}

fn api_mask_rgb(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    mask: &Tensor,
) -> Result<ImageTensor, PngError> {
    let channel = tensor_unsqueeze_exact_native(mask, -1, context.cancellation)
        .map_err(|error| PngError::Codec(error.to_string()))?;
    let rgb = torch_cat_with_context_exact_native(
        backend,
        &[channel.clone(), channel.clone(), channel],
        -1,
        context,
    )
    .map_err(|error| PngError::Codec(error.to_string()))?;
    Ok(ImageTensor::from_tensor(rgb)?)
}

pub fn encode_api_png_source_batch_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    image: &ImageTensor,
    maximum_pixels: u64,
    limits: PngLimits,
) -> Result<Vec<u8>, PngError> {
    context.check()?;
    let (batch, height, width, channels) = image.dimensions()?;
    let Some((target_width, target_height)) =
        api_downscale_dimensions(width, height, maximum_pixels)?
    else {
        return encode_api_png_frame_with_context(backend, context, image, 0, None, limits);
    };
    validate_dimensions(
        u32::try_from(target_width).map_err(|_| PngError::SizeOverflow)?,
        u32::try_from(target_height).map_err(|_| PngError::SizeOverflow)?,
        limits,
    )?;
    let image = image.resize(
        target_width,
        target_height,
        ResizeMode::Lanczos,
        ResizeCrop::Disabled,
        backend,
        context,
    )?;
    if channels != 1 {
        return encode_api_png_frame_with_context(backend, context, &image, 0, None, limits);
    }
    // Source grayscale Lanczos collapses BHWC to BWH. Upload sees rank three
    // and passes the whole array to Pillow, interpreting H as the channel count.
    if !matches!(target_height, 2..=4) {
        return Err(PngError::Codec(
            "source API image shape cannot be encoded by Pillow".to_owned(),
        ));
    }
    let image = api_transpose_grayscale(backend, context, &image)?;
    let width = u32::try_from(target_width).map_err(|_| PngError::SizeOverflow)?;
    let height = u32::try_from(batch).map_err(|_| PngError::SizeOverflow)?;
    validate_dimensions(width, height, limits)?;
    let pixels = image.numpy_image_bytes(backend, context)?;
    let encoded = encode_pixels_with_metadata(
        &pixels,
        width,
        height,
        target_height,
        &BTreeMap::new(),
        MetadataWritePolicy {
            metadata_enabled: false,
        },
        limits,
    )?;
    context.check()?;
    Ok(encoded)
}

fn api_downscale_dimensions(
    width: u64,
    height: u64,
    maximum_pixels: u64,
) -> Result<Option<(u64, u64)>, PngError> {
    let pixels = width.checked_mul(height).ok_or(PngError::SizeOverflow)?;
    if pixels == 0 {
        return Err(PngError::Dimensions {
            width: 0,
            height: 0,
        });
    }
    if pixels <= maximum_pixels {
        return Ok(None);
    }
    let scale = (maximum_pixels as f64 / pixels as f64).sqrt();
    let width = ((width as f64 * scale) as u64).max(2);
    let height = ((height as f64 * scale) as u64).max(2);
    Ok(Some((width - width % 2, height - height % 2)))
}

fn api_transpose_grayscale(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    image: &ImageTensor,
) -> Result<ImageTensor, PngError> {
    let permuted = tensor_permute_exact_native(image.tensor(), &[0, 2, 1, 3], context.cancellation);
    context.check()?;
    let permuted = permuted.map_err(|error| PngError::Codec(error.to_string()))?;
    let contiguous = contiguous_with_context_exact_native(
        backend,
        &permuted,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    );
    context.check()?;
    Ok(ImageTensor::from_tensor(
        contiguous.map_err(|error| PngError::Codec(error.to_string()))?,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn checked_png_frame(
    pixels_bhwc: &[f32],
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    batch_index: u64,
    limits: PngLimits,
) -> Result<(&[f32], u32, u32, usize), PngError> {
    if !matches!(channels, 3 | 4) {
        return Err(PngError::Dimensions {
            width: u32::try_from(width).unwrap_or(u32::MAX),
            height: u32::try_from(height).unwrap_or(u32::MAX),
        });
    }
    if batch_index >= batch {
        return Err(PngError::BatchIndex {
            index: batch_index,
            batch,
        });
    }
    let width = u32::try_from(width).map_err(|_| PngError::SizeOverflow)?;
    let height = u32::try_from(height).map_err(|_| PngError::SizeOverflow)?;
    validate_dimensions(width, height, limits)?;
    let values_per_frame = checked_pixel_count(width, height)?
        .checked_mul(usize::try_from(channels).map_err(|_| PngError::SizeOverflow)?)
        .ok_or(PngError::SizeOverflow)?;
    let expected = values_per_frame
        .checked_mul(usize::try_from(batch).map_err(|_| PngError::SizeOverflow)?)
        .ok_or(PngError::SizeOverflow)?;
    if pixels_bhwc.len() != expected {
        return Err(PngError::PixelCount {
            expected,
            actual: pixels_bhwc.len(),
        });
    }
    let frame_start = values_per_frame
        .checked_mul(usize::try_from(batch_index).map_err(|_| PngError::SizeOverflow)?)
        .ok_or(PngError::SizeOverflow)?;
    let frame_end = frame_start
        .checked_add(values_per_frame)
        .ok_or(PngError::SizeOverflow)?;
    let frame = pixels_bhwc
        .get(frame_start..frame_end)
        .ok_or(PngError::SizeOverflow)?;
    Ok((frame, width, height, values_per_frame))
}

fn quantize_channel(value: f32) -> u8 {
    let scaled = (value * 255.0).clamp(0.0, 255.0);
    scaled as u8
}

fn encode_pixels_with_metadata(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u64,
    metadata: &BTreeMap<String, String>,
    metadata_policy: MetadataWritePolicy,
    limits: PngLimits,
) -> Result<Vec<u8>, PngError> {
    let mut encoded = Vec::new();
    let color_type = match channels {
        1 => ExtendedColorType::L8,
        2 => ExtendedColorType::La8,
        3 => ExtendedColorType::Rgb8,
        4 => ExtendedColorType::Rgba8,
        _ => return Err(PngError::Dimensions { width, height }),
    };
    PngEncoder::new(&mut encoded).write_image(pixels, width, height, color_type)?;
    if encoded.len() > limits.max_input_bytes {
        return Err(PngError::InputTooLarge {
            actual: encoded.len(),
            limit: limits.max_input_bytes,
        });
    }
    if metadata.is_empty() {
        return Ok(encoded);
    }
    let document = MetadataDocument::parse(
        &encoded,
        Some("image.png"),
        Some("image/png"),
        MetadataLimits {
            max_input_bytes: limits.max_input_bytes,
            ..MetadataLimits::default()
        },
    )?;
    Ok(document.embed_sim_metadata(
        metadata,
        metadata_policy,
        &MetadataLimits {
            max_input_bytes: limits.max_input_bytes,
            ..MetadataLimits::default()
        },
    )?)
}

fn validate_dimensions(width: u32, height: u32, limits: PngLimits) -> Result<(), PngError> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(PngError::SizeOverflow)?;
    if width == 0
        || height == 0
        || width > limits.max_dimension
        || height > limits.max_dimension
        || pixels > limits.max_pixels
    {
        return Err(PngError::Dimensions { width, height });
    }
    Ok(())
}

fn checked_pixel_count(width: u32, height: u32) -> Result<usize, PngError> {
    usize::try_from(
        u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or(PngError::SizeOverflow)?,
    )
    .map_err(|_| PngError::SizeOverflow)
}

#[cfg(test)]
mod tests {

    #[test]
    fn downloaded_images_preserve_direct_alpha_and_source_orientation()
    -> Result<(), Box<dyn std::error::Error>> {
        let pixels = [128, 64, 32, 32, 0, 255, 64, 255];
        let mut bytes = Vec::new();
        let mut encoder = PngEncoder::new(&mut bytes);
        let orientation = b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        encoder.set_exif_metadata(orientation.to_vec())?;
        encoder.write_image(&pixels, 2, 1, ExtendedColorType::Rgba8)?;
        let cancellation = CancellationToken::default();
        let downloaded = decode_downloaded_image_rgba(&bytes, PngLimits::default(), &cancellation)?;
        assert_eq!((downloaded.width, downloaded.height), (2, 1));
        assert_eq!(
            downloaded.pixels_bhwc,
            pixels.map(|value| f32::from(value) / 255.0)
        );
        let load_image = decode_png(&bytes, PngLimits::default())?;
        assert_eq!((load_image.width, load_image.height), (1, 2));
        let lossy_alpha = 1.0_f32 - (1.0_f32 - 32.0_f32 / 255.0);
        assert_ne!(lossy_alpha, 32.0 / 255.0);
        assert_eq!(downloaded.pixels_bhwc.get(3).copied(), Some(32.0 / 255.0));
        Ok(())
    }

    const L16_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x00\x00\x00\x00\x63\x05\xe7\x6c\x00\x00\x00\x13\x49\x44\x41\x54\x78\x9c\x63\x60\x60\x60\x68\x60\xf8\xcf\xc8\xf0\xff\x3f\x00\x0b\x87\x03\x7f\x35\xb0\x77\xa5\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const LA16_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x04\x00\x00\x00\xec\x67\x70\x3b\x00\x00\x00\x18\x49\x44\x41\x54\x78\x9c\x63\x60\x00\x81\x06\x20\xfc\xcf\xf0\x9f\x91\x81\x91\xe1\x3f\x10\x00\x00\x2c\x05\x06\xfd\xbd\x43\xda\x3c\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const RGB16_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x02\x00\x00\x00\xc9\x0c\x2f\xe7\x00\x00\x00\x16\x49\x44\x41\x54\x78\x9c\x63\x60\x80\x80\x06\x30\xfc\x0f\x82\x8c\x0c\x20\xf8\x1f\x0c\x00\x61\x7b\x0a\x7b\x67\xc9\xe7\xca\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const RGBA16_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x06\x00\x00\x00\x46\x6e\xb8\xb0\x00\x00\x00\x16\x49\x44\x41\x54\x78\x9c\x63\x60\x80\x81\x06\x28\xfc\x0f\x81\x8c\x0c\x10\xf8\x1f\x0a\x00\xab\xe9\x0d\xf9\x4f\xe2\x98\x49\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const GRAY16_TRNS128_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x00\x00\x00\x00\x63\x05\xe7\x6c\x00\x00\x00\x02\x74\x52\x4e\x53\x00\x80\x9b\x2b\x4e\x18\x00\x00\x00\x13\x49\x44\x41\x54\x78\x9c\x63\x60\x60\x60\x68\x60\xf8\xcf\xc8\xf0\xff\x3f\x00\x0b\x87\x03\x7f\x35\xb0\x77\xa5\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const GRAY16_TRNS256_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x00\x00\x00\x00\x63\x05\xe7\x6c\x00\x00\x00\x02\x74\x52\x4e\x53\x01\x00\x6f\x88\xfc\x79\x00\x00\x00\x13\x49\x44\x41\x54\x78\x9c\x63\x60\x60\x60\x68\x60\xf8\xcf\xc8\xf0\xff\x3f\x00\x0b\x87\x03\x7f\x35\xb0\x77\xa5\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const GRAY16_TRNS65535_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x00\x00\x00\x00\x63\x05\xe7\x6c\x00\x00\x00\x02\x74\x52\x4e\x53\xff\xff\xc8\xb5\xdf\xc7\x00\x00\x00\x13\x49\x44\x41\x54\x78\x9c\x63\x60\x60\x60\x68\x60\xf8\xcf\xc8\xf0\xff\x3f\x00\x0b\x87\x03\x7f\x35\xb0\x77\xa5\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const RGB16_TRNS128_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x02\x00\x00\x00\xc9\x0c\x2f\xe7\x00\x00\x00\x06\x74\x52\x4e\x53\x00\x80\x00\x80\x00\x80\xd3\xcd\x2f\xa3\x00\x00\x00\x16\x49\x44\x41\x54\x78\x9c\x63\x60\x80\x80\x06\x30\xfc\x0f\x82\x8c\x0c\x20\xf8\x1f\x0c\x00\x61\x7b\x0a\x7b\x67\xc9\xe7\xca\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const RGB16_TRNS256_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x02\x00\x00\x00\xc9\x0c\x2f\xe7\x00\x00\x00\x06\x74\x52\x4e\x53\x01\x00\x01\x00\x01\x00\x04\x5d\x82\x10\x00\x00\x00\x16\x49\x44\x41\x54\x78\x9c\x63\x60\x80\x80\x06\x30\xfc\x0f\x82\x8c\x0c\x20\xf8\x1f\x0c\x00\x61\x7b\x0a\x7b\x67\xc9\xe7\xca\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const RGB16_TRNS65535_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x02\x00\x00\x00\xc9\x0c\x2f\xe7\x00\x00\x00\x06\x74\x52\x4e\x53\xff\xff\xff\xff\xff\xff\x9e\xbd\x4b\x32\x00\x00\x00\x16\x49\x44\x41\x54\x78\x9c\x63\x60\x80\x80\x06\x30\xfc\x0f\x82\x8c\x0c\x20\xf8\x1f\x0c\x00\x61\x7b\x0a\x7b\x67\xc9\xe7\xca\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";

    const RGBA16_MIXED_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x06\x00\x00\x00\x46\x6e\xb8\xb0\x00\x00\x00\x21\x49\x44\x41\x54\x78\x9c\x2d\x89\xb1\x0d\x00\x00\x0c\x82\xe8\xe7\x7e\xae\x0c\x65\x30\x44\x20\xf4\xa8\x04\x89\x0e\xef\x26\x8b\xe3\x3b\x33\x88\x0e\x79\x7f\x20\x28\xef\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const RGB16_TRNS_MIXED_DOWNLOADED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x05\x00\x00\x00\x01\x10\x02\x00\x00\x00\xc9\x0c\x2f\xe7\x00\x00\x00\x06\x74\x52\x4e\x53\x00\x80\x01\x00\xff\xff\xd9\xda\xc2\x99\x00\x00\x00\x1e\x49\x44\x41\x54\x78\x9c\x63\x68\x60\x60\x60\xf8\xff\x9f\xa1\x01\x44\x02\x89\x06\x30\xc9\xf0\xff\x1f\x23\x48\xa8\x01\x00\xb7\x1a\x0b\x79\x3f\x5d\xd7\x55\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";

    #[test]
    fn downloaded_images_match_pillow_png16_color_and_transparency()
    -> Result<(), Box<dyn std::error::Error>> {
        for (bytes, expected) in [
            (
                L16_DOWNLOADED_PNG,
                [
                    0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255,
                ],
            ),
            (
                LA16_DOWNLOADED_PNG,
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 255, 255, 255, 255,
                ],
            ),
            (
                RGB16_DOWNLOADED_PNG,
                [
                    0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 1, 1, 1, 255, 255, 255, 255, 255,
                ],
            ),
            (
                RGBA16_DOWNLOADED_PNG,
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 255, 255, 255, 255,
                ],
            ),
            (
                GRAY16_TRNS128_DOWNLOADED_PNG,
                [
                    0, 0, 0, 255, 128, 128, 128, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255,
                ],
            ),
            (
                GRAY16_TRNS256_DOWNLOADED_PNG,
                [
                    0, 0, 0, 0, 128, 128, 128, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255,
                ],
            ),
            (
                GRAY16_TRNS65535_DOWNLOADED_PNG,
                [
                    0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255,
                    255, 0,
                ],
            ),
            (
                RGB16_TRNS128_DOWNLOADED_PNG,
                [
                    0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 1, 1, 1, 255, 255, 255, 255, 255,
                ],
            ),
            (
                RGB16_TRNS256_DOWNLOADED_PNG,
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 255, 255, 255, 255, 255,
                ],
            ),
            (
                RGB16_TRNS65535_DOWNLOADED_PNG,
                [
                    0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 1, 1, 1, 255, 255, 255, 255, 0,
                ],
            ),
            (
                RGBA16_MIXED_DOWNLOADED_PNG,
                [
                    0, 0, 1, 255, 255, 128, 0, 0, 1, 255, 0, 128, 0, 0, 128, 255, 255, 0, 0, 1,
                ],
            ),
            (
                RGB16_TRNS_MIXED_DOWNLOADED_PNG,
                [
                    128, 0, 255, 0, 0, 0, 255, 255, 128, 0, 255, 0, 128, 0, 255, 0, 1, 128, 0, 255,
                ],
            ),
        ] {
            let decoded = decode_downloaded_image_rgba(
                bytes,
                PngLimits::default(),
                &CancellationToken::default(),
            )?;
            assert_eq!((decoded.width, decoded.height), (5, 1));
            assert_eq!(
                decoded.pixels_bhwc,
                expected.map(|value: u8| f32::from(value) / 255.0)
            );
            assert!(decode_png(bytes, PngLimits::default()).is_ok());
        }
        Ok(())
    }

    #[test]
    fn downloaded_images_reject_non_png_high_depth_conversion() {
        let bytes = b"P5\n5 1\n65535\n\x00\x00\x00\x80\x00\xff\x01\x00\xff\xff";
        assert_eq!(
            decode_downloaded_image_rgba(
                bytes,
                PngLimits::default(),
                &CancellationToken::default(),
            ),
            Err(PngError::UnsupportedDownloadedColor {
                color: image::ColorType::L16
            }),
        );
    }

    #[test]
    fn downloaded_images_decode_jpeg_and_webp_without_vendor_codecs()
    -> Result<(), Box<dyn std::error::Error>> {
        for format in [ImageFormat::Jpeg, ImageFormat::WebP] {
            let image =
                DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1, 1, image::Rgb([0, 0, 0])));
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, format)?;
            let decoded = decode_downloaded_image_rgba(
                bytes.get_ref(),
                PngLimits::default(),
                &CancellationToken::default(),
            )?;
            assert_eq!((decoded.width, decoded.height), (1, 1));
            assert_eq!(decoded.pixels_bhwc, [0.0, 0.0, 0.0, 1.0]);
        }
        Ok(())
    }

    #[derive(serde::Deserialize)]
    struct DownloadedCodecFixture {
        source_sha256: String,
        cases: Vec<DownloadedCodecCase>,
    }

    #[derive(serde::Deserialize)]
    struct DownloadedCodecCase {
        name: String,
        encoded: String,
        encoded_sha256: String,
        f32le_sha256: String,
        shape: [u32; 4],
    }

    #[test]
    fn downloaded_images_match_pinned_webp_source() -> Result<(), Box<dyn std::error::Error>> {
        let differences = downloaded_codec_source_differences("webp-")?;
        assert!(differences.is_empty(), "{}", differences.join("; "));
        Ok(())
    }

    #[test]
    fn downloaded_images_match_pinned_jpeg_source() -> Result<(), Box<dyn std::error::Error>> {
        let differences = downloaded_codec_source_differences("jpeg-")?;
        assert!(differences.is_empty(), "{}", differences.join("; "));
        Ok(())
    }

    fn downloaded_codec_source_differences(
        prefix: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/media/downloaded-image-codecs");
        let fixture: DownloadedCodecFixture =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        assert_eq!(
            fixture.source_sha256,
            "632476df7c448efcc80acc045b45a3f1c6cf44540c7883bb380615040cc9d7a2"
        );
        assert_eq!(fixture.cases.len(), 12);
        let mut differences = Vec::new();
        let cases: Vec<_> = fixture
            .cases
            .into_iter()
            .filter(|case| case.name.starts_with(prefix))
            .collect();
        assert_eq!(cases.len(), 6);
        for case in cases {
            let encoded = std::fs::read(root.join(&case.encoded))?;
            let expected = std::fs::read(root.join(format!("{}.f32le", case.name)))?;
            assert_eq!(
                format!("{:x}", Sha256::digest(&encoded)),
                case.encoded_sha256
            );
            assert_eq!(
                format!("{:x}", Sha256::digest(&expected)),
                case.f32le_sha256
            );
            let decoded = decode_downloaded_image_rgba(
                &encoded,
                PngLimits::default(),
                &CancellationToken::default(),
            )?;
            assert_eq!([1, decoded.height, decoded.width, 4], case.shape);
            let actual: Vec<u8> = decoded
                .pixels_bhwc
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert_eq!(actual.len(), expected.len());
            let differing_values = actual
                .chunks_exact(4)
                .zip(expected.chunks_exact(4))
                .filter(|(actual, expected)| actual != expected)
                .count();
            if differing_values != 0 {
                differences.push(format!(
                    "{}: {differing_values} differing F32 values",
                    case.name
                ));
            }
        }
        Ok(differences)
    }

    #[test]
    fn downloaded_jpeg_boundary_pixels_match_actual_pinned_source()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/media/downloaded-image-codecs/jpeg-boundaries");
        let fixture: DownloadedCodecFixture =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        assert_eq!(
            fixture.source_sha256,
            "632476df7c448efcc80acc045b45a3f1c6cf44540c7883bb380615040cc9d7a2"
        );
        assert_eq!(fixture.cases.len(), 222);
        let mut differences = Vec::new();
        for case in fixture.cases {
            let encoded = std::fs::read(root.join(&case.encoded))?;
            let expected = std::fs::read(root.join(format!("{}.f32le", case.name)))?;
            assert_eq!(
                format!("{:x}", Sha256::digest(&encoded)),
                case.encoded_sha256
            );
            assert_eq!(
                format!("{:x}", Sha256::digest(&expected)),
                case.f32le_sha256
            );
            let decoded = decode_downloaded_image_rgba(
                &encoded,
                PngLimits::default(),
                &CancellationToken::default(),
            )?;
            assert_eq!([1, decoded.height, decoded.width, 4], case.shape);
            let actual: Vec<u8> = decoded
                .pixels_bhwc
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert_eq!(actual.len(), expected.len());
            let different = actual
                .chunks_exact(4)
                .zip(expected.chunks_exact(4))
                .filter(|(actual, expected)| actual != expected)
                .count();
            if different != 0 {
                differences.push(format!("{}: {different} differing F32 values", case.name));
            }
        }
        assert!(differences.is_empty(), "{}", differences.join("; "));
        Ok(())
    }

    #[test]
    fn jpeg_source_mode_preserves_pristine_legacy_constructor_pixels()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        use zune_jpeg::zune_core::{bytestream::ZCursor, options::DecoderOptions};
        #[derive(serde::Deserialize)]
        struct LegacyFixture {
            cases: Vec<LegacyCase>,
        }
        #[derive(serde::Deserialize)]
        struct LegacyCase {
            encoded: String,
            encoded_sha256: String,
            native_rgb_sha256: String,
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/media/downloaded-image-codecs");
        let fixture: LegacyFixture =
            serde_json::from_slice(&std::fs::read(root.join("legacy-zune-0.5.15.json"))?)?;
        assert_eq!(fixture.cases.len(), 228);
        for case in fixture.cases {
            let bytes = std::fs::read(root.join(&case.encoded))?;
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), case.encoded_sha256);
            let options = DecoderOptions::default().set_strict_mode(false);
            let mut decoder =
                zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes.as_slice()), options);
            let expected = decoder.decode()?;
            assert_eq!(
                format!("{:x}", Sha256::digest(&expected)),
                case.native_rgb_sha256,
                "{}",
                case.encoded
            );
            let mut decoder = zune_jpeg::JpegDecoder::new(ZCursor::new(bytes.as_slice()));
            assert_eq!(decoder.decode()?, expected, "{}", case.encoded);
            let mut decoder = zune_jpeg::JpegDecoder::new_with_options_and_pixel_mode(
                ZCursor::new(bytes.as_slice()),
                options,
                zune_jpeg::JpegPixelMode::Native,
            );
            assert_eq!(decoder.decode()?, expected, "{}", case.encoded);
        }
        Ok(())
    }

    #[test]
    fn downloaded_jpeg_failure_bounds_and_cancellation_publish_no_image()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/media/downloaded-image-codecs");
        let encoded = std::fs::read(root.join("jpeg-progressive.jpg"))?;
        let cancellation = CancellationToken::default();
        assert!(matches!(
            decode_downloaded_image_rgba(
                &encoded,
                PngLimits {
                    max_input_bytes: encoded.len() - 1,
                    ..PngLimits::default()
                },
                &cancellation
            ),
            Err(PngError::InputTooLarge { .. })
        ));
        assert!(matches!(
            decode_downloaded_image_rgba(
                &encoded,
                PngLimits {
                    max_dimension: 4,
                    ..PngLimits::default()
                },
                &cancellation
            ),
            Err(PngError::Dimensions { .. } | PngError::Codec(_))
        ));
        assert!(matches!(
            decode_downloaded_image_rgba(
                &encoded,
                PngLimits {
                    max_decoder_allocation_bytes: 1024,
                    ..PngLimits::default()
                },
                &cancellation
            ),
            Err(PngError::Allocation(_))
        ));
        for corrupt in [
            b"\xff\xd8".as_slice(),
            b"\xff\xd8\xff\xdb\x00\x01".as_slice(),
            b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x00\x00\x00\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
                .as_slice(),
        ] {
            assert!(
                decode_downloaded_image_rgba(corrupt, PngLimits::default(), &cancellation).is_err()
            );
        }
        cancellation.cancel();
        assert!(matches!(
            decode_downloaded_image_rgba(&encoded, PngLimits::default(), &cancellation),
            Err(PngError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn downloaded_image_decode_bounds_and_cancellation_precede_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let image =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(32, 32, image::Rgb([0, 0, 0])));
        let bytes = crate::encode_png(&image)?;
        let cancellation = CancellationToken::default();
        assert!(matches!(
            decode_downloaded_image_rgba(
                &bytes,
                PngLimits {
                    max_decoder_allocation_bytes: 1024,
                    ..PngLimits::default()
                },
                &cancellation
            ),
            Err(PngError::Allocation(_))
        ));
        assert!(matches!(
            decode_downloaded_image_rgba(
                &bytes,
                PngLimits {
                    max_pixels: 1,
                    ..PngLimits::default()
                },
                &cancellation
            ),
            Err(PngError::Dimensions { .. })
        ));
        assert!(matches!(
            decode_downloaded_image_rgba(
                &bytes,
                PngLimits {
                    max_input_bytes: 1,
                    ..PngLimits::default()
                },
                &cancellation
            ),
            Err(PngError::InputTooLarge { .. })
        ));
        assert!(matches!(
            decode_downloaded_image_rgba(b"malformed-image", PngLimits::default(), &cancellation),
            Err(PngError::Codec(_))
        ));
        cancellation.cancel();
        assert!(matches!(
            decode_downloaded_image_rgba(b"malformed-image", PngLimits::default(), &cancellation),
            Err(PngError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }
    use super::*;
    use image::RgbaImage;
    use sim_tensor::{CancellationToken, CpuWorkspaceAuthority, StreamId};

    #[derive(serde::Deserialize)]
    struct ApiImageFixture {
        schema_version: u16,
        sources: BTreeMap<String, String>,
        cases: Vec<ApiImageCase>,
    }

    #[derive(serde::Deserialize)]
    struct ApiImageCase {
        shape: [u64; 4],
        pixels: Vec<f32>,
        maximum_pixels: Option<u64>,
        batch_index: u64,
        output: Option<ApiImageOutput>,
        error: Option<String>,
    }

    #[derive(serde::Deserialize)]
    struct ApiImageOutput {
        width: u32,
        height: u32,
        mode: String,
        pixels: Vec<u8>,
    }

    #[derive(serde::Deserialize)]
    struct BeebleMediaFixture {
        schema_version: u16,
        sources: BTreeMap<String, String>,
        cases: Vec<BeebleMediaCase>,
    }

    #[derive(serde::Deserialize)]
    struct BeebleMediaCase {
        role: String,
        shape: Vec<u64>,
        pixels: Vec<f32>,
        maximum_pixels: u64,
        output: Option<ApiImageOutput>,
        error: Option<String>,
    }

    #[test]
    fn beeble_png_preparation_preserves_source_reference_and_mask_paths()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let fixture: BeebleMediaFixture = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/providers/beeble/source_media.json"
        )))?;
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.cases.len(), 66);
        let source =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../projects/comfy/ComfyUI");
        for (path, expected) in fixture.sources {
            assert_eq!(
                format!("{:x}", Sha256::digest(std::fs::read(source.join(path))?)),
                expected
            );
        }
        for (index, case) in fixture.cases.into_iter().enumerate() {
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
            let cancellation = CancellationToken::default();
            let context = backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(4 * 1024 * 1024)?,
                &cancellation,
            );
            let (tensor, _) = backend.upload_f32(
                sim_tensor::TensorDescriptor::contiguous(
                    case.shape,
                    sim_tensor::DType::F32,
                    sim_tensor::DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                &case.pixels,
                &context,
            )?;
            let result = match case.role.as_str() {
                "mask" => encode_api_png_mask_with_context(
                    &backend,
                    &context,
                    &tensor,
                    case.maximum_pixels,
                    PngLimits::default(),
                ),
                "source" => encode_api_png_source_batch_with_context(
                    &backend,
                    &context,
                    &ImageTensor::from_tensor(tensor)?,
                    case.maximum_pixels,
                    PngLimits::default(),
                ),
                "reference" => encode_api_png_frame_with_context(
                    &backend,
                    &context,
                    &ImageTensor::from_tensor(tensor)?,
                    0,
                    Some(case.maximum_pixels),
                    PngLimits::default(),
                ),
                role => return Err(format!("unexpected media role {role}").into()),
            };
            let Some(expected) = case.output else {
                assert!(case.error.is_some());
                assert!(
                    result.is_err(),
                    "source-error case {index} encoded unexpectedly"
                );
                continue;
            };
            let decoded = image::load_from_memory(
                &result.map_err(|error| format!("case {index}: {error}"))?,
            )?;
            assert_eq!(
                (decoded.width(), decoded.height()),
                (expected.width, expected.height),
                "case {index}"
            );
            let color = match expected.mode.as_str() {
                "L" => image::ColorType::L8,
                "LA" => image::ColorType::La8,
                "RGB" => image::ColorType::Rgb8,
                "RGBA" => image::ColorType::Rgba8,
                mode => return Err(format!("unexpected source color mode {mode}").into()),
            };
            assert_eq!(decoded.color(), color, "case {index}");
            assert_eq!(decoded.into_bytes(), expected.pixels, "case {index}");
        }
        Ok(())
    }

    #[test]
    fn api_image_preparation_matches_pinned_source_pixels_and_shapes()
    -> Result<(), Box<dyn std::error::Error>> {
        use sha2::{Digest, Sha256};
        let fixture: ApiImageFixture = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/providers/common/source_image_upload_linux_x86_64.json"
        )))?;
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.cases.len(), 56);
        let source =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../projects/comfy/ComfyUI");
        for (path, expected) in &fixture.sources {
            assert_eq!(
                format!("{:x}", Sha256::digest(std::fs::read(source.join(path))?)),
                *expected
            );
        }
        for (index, case) in fixture.cases.iter().enumerate() {
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
            let cancellation = CancellationToken::default();
            let context = backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(4 * 1024 * 1024)?,
                &cancellation,
            );
            let [batch, height, width, channels] = case.shape;
            let image = ImageTensor::from_f32(
                &backend,
                &context,
                batch,
                height,
                width,
                channels,
                &case.pixels,
            )?;
            let result = encode_api_png_frame_with_context(
                &backend,
                &context,
                &image,
                case.batch_index,
                case.maximum_pixels,
                PngLimits::default(),
            );
            let Some(expected) = &case.output else {
                assert!(case.error.is_some());
                assert!(
                    result.is_err(),
                    "source-error case {index} unexpectedly encoded"
                );
                continue;
            };
            assert!(case.error.is_none());
            let decoded = image::load_from_memory(&result?)?;
            assert_eq!(
                (decoded.width(), decoded.height()),
                (expected.width, expected.height),
                "case {index}"
            );
            let expected_color = match expected.mode.as_str() {
                "L" => image::ColorType::L8,
                "RGB" => image::ColorType::Rgb8,
                "RGBA" => image::ColorType::Rgba8,
                mode => return Err(format!("unexpected source image mode {mode}").into()),
            };
            assert_eq!(decoded.color(), expected_color, "case {index}");
            assert_eq!(
                decoded.into_bytes(),
                expected.pixels,
                "case {index}, shape {:?}, maximum {:?}",
                case.shape,
                case.maximum_pixels
            );
            assert_eq!(image.as_f32_slice()?, case.pixels);
        }
        Ok(())
    }

    #[test]
    fn beeble_png_preparation_releases_admission_after_failure_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8192)?;
        let cancellation = CancellationToken::default();
        let workspace = authority.authorize_workspace(4096)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);
        let image = ImageTensor::from_f32(&backend, &context, 1, 4, 4, 3, &[0.5; 48])?;
        let mask = backend
            .upload_f32(
                sim_tensor::TensorDescriptor::contiguous(
                    vec![1, 4, 4],
                    sim_tensor::DType::F32,
                    sim_tensor::DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                &[0.5; 16],
                &context,
            )?
            .0;
        let resident = backend.memory_snapshot().current_bytes;
        let insufficient = authority.authorize_workspace(0)?;
        let denied =
            backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
        assert!(
            encode_api_png_source_batch_with_context(
                &backend,
                &denied,
                &image,
                4,
                PngLimits::default()
            )
            .is_err()
        );
        assert!(
            encode_api_png_mask_with_context(&backend, &denied, &mask, 4, PngLimits::default())
                .is_err()
        );
        assert_eq!(insufficient.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, resident);
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        let stopped = backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancelled);
        assert!(
            encode_api_png_source_batch_with_context(
                &backend,
                &stopped,
                &image,
                4,
                PngLimits::default()
            )
            .is_err()
        );
        assert!(
            encode_api_png_mask_with_context(&backend, &stopped, &mask, 4, PngLimits::default())
                .is_err()
        );
        assert_eq!(workspace.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, resident);
        assert!(
            !encode_api_png_source_batch_with_context(
                &backend,
                &context,
                &image,
                4,
                PngLimits::default()
            )?
            .is_empty()
        );
        assert!(
            !encode_api_png_mask_with_context(&backend, &context, &mask, 4, PngLimits::default())?
                .is_empty()
        );
        assert_eq!(workspace.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, resident);
        Ok(())
    }

    #[test]
    fn api_image_preparation_releases_workspace_after_failure_and_recovers()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4096)?;
        let cancellation = CancellationToken::default();
        let workspace = authority.authorize_workspace(1024)?;
        let context =
            backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);
        let image = ImageTensor::from_f32(&backend, &context, 1, 2, 2, 3, &[0.5; 12])?;
        let resident_bytes = backend.memory_snapshot().current_bytes;
        let insufficient = authority.authorize_workspace(0)?;
        let insufficient_context =
            backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
        assert!(matches!(
            encode_api_png_frame_with_context(
                &backend,
                &insufficient_context,
                &image,
                0,
                None,
                PngLimits::default()
            ),
            Err(PngError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(insufficient.in_use_bytes(), 0);
        assert!(matches!(
            encode_api_png_frame_with_context(
                &backend,
                &context,
                &image,
                1,
                None,
                PngLimits::default()
            ),
            Err(PngError::BatchIndex { .. })
        ));
        assert!(matches!(
            encode_api_png_frame_with_context(
                &backend,
                &context,
                &image,
                0,
                None,
                PngLimits {
                    max_pixels: 3,
                    ..PngLimits::default()
                }
            ),
            Err(PngError::Dimensions { .. })
        ));
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        let cancelled_context =
            backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancelled);
        assert!(matches!(
            encode_api_png_frame_with_context(
                &backend,
                &cancelled_context,
                &image,
                0,
                Some(1),
                PngLimits::default()
            ),
            Err(PngError::Tensor(TensorError::Cancelled))
        ));
        let encoded = encode_api_png_frame_with_context(
            &backend,
            &context,
            &image,
            0,
            None,
            PngLimits::default(),
        )?;
        assert_eq!(image::load_from_memory(&encoded)?.into_bytes(), [127; 12]);
        assert_eq!(workspace.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, resident_bytes);
        assert_eq!(image.as_f32_slice()?, [0.5; 12]);
        Ok(())
    }

    #[test]
    fn native_png_encoding_preserves_rgba_preview_alpha() -> Result<(), Box<dyn std::error::Error>>
    {
        let encoded = encode_png_frame(
            &[1.0, 0.0, 0.5, 0.25],
            1,
            1,
            1,
            4,
            0,
            &BTreeMap::new(),
            PngLimits::default(),
        )?;
        let decoded = decode_png(&encoded, PngLimits::default())?;
        assert!(decoded.has_alpha);
        assert_eq!(decoded.pixels_bhwc, vec![1.0, 0.0, 127.0 / 255.0]);
        assert_eq!(decoded.mask_bhw, vec![1.0 - 63.0 / 255.0]);
        Ok(())
    }

    #[test]
    fn native_png_encoding_uses_exact_caller_workspace() -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let exact = authority.authorize_workspace(6)?;
        let context = backend.execution_context(StreamId::DEFAULT, exact.clone(), &cancellation);
        let encoded = encode_png_frame_with_policy_and_context(
            &backend,
            &context,
            &[1.0, 0.0, 0.5, 0.0, 0.25, 1.0],
            1,
            1,
            2,
            3,
            0,
            &BTreeMap::new(),
            MetadataWritePolicy::default(),
            PngLimits::default(),
        )?;
        assert_eq!(decode_png(&encoded, PngLimits::default())?.width, 2);
        assert_eq!(exact.peak_bytes(), 6);
        assert_eq!(exact.in_use_bytes(), 0);

        let insufficient = authority.authorize_workspace(5)?;
        let insufficient_context =
            backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
        assert!(matches!(
            encode_png_frame_with_policy_and_context(
                &backend,
                &insufficient_context,
                &[1.0, 0.0, 0.5, 0.0, 0.25, 1.0],
                1,
                1,
                2,
                3,
                0,
                &BTreeMap::new(),
                MetadataWritePolicy::default(),
                PngLimits::default(),
            ),
            Err(PngError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(insufficient.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        let cancelled_authorization = authority.authorize_workspace(6)?;
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            cancelled_authorization.clone(),
            &cancelled,
        );
        assert!(matches!(
            encode_png_frame_with_policy_and_context(
                &backend,
                &cancelled_context,
                &[1.0, 0.0, 0.5, 0.0, 0.25, 1.0],
                1,
                1,
                2,
                3,
                0,
                &BTreeMap::new(),
                MetadataWritePolicy::default(),
                PngLimits::default(),
            ),
            Err(PngError::Tensor(TensorError::Cancelled))
        ));
        assert_eq!(cancelled_authorization.in_use_bytes(), 0);

        let (constrained_backend, constrained_authority) =
            CpuWorkspaceAuthority::create_backend(8)?;
        let oom_authorization = constrained_authority.authorize_workspace(6)?;
        let oom_context = constrained_backend.execution_context(
            StreamId::DEFAULT,
            oom_authorization.clone(),
            &cancellation,
        );
        assert!(matches!(
            encode_png_frame_with_policy_and_context(
                &constrained_backend,
                &oom_context,
                &[1.0, 0.0, 0.5, 0.0, 0.25, 1.0],
                1,
                1,
                2,
                3,
                0,
                &BTreeMap::new(),
                MetadataWritePolicy::default(),
                PngLimits::default(),
            ),
            Err(PngError::Tensor(TensorError::AllocationFailed { .. }))
        ));
        assert_eq!(oom_authorization.in_use_bytes(), 0);
        assert_eq!(constrained_backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn rgba_decode_matches_sim_rgb_and_inverted_alpha_mask() -> Result<(), PngError> {
        let source = RgbaImage::from_raw(2, 1, vec![255, 0, 128, 255, 0, 64, 255, 0])
            .ok_or(PngError::SizeOverflow)?;
        let mut bytes = Vec::new();
        DynamicImage::ImageRgba8(source)
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)?;
        let decoded = decode_png(&bytes, PngLimits::default())?;
        assert_eq!((decoded.width, decoded.height), (2, 1));
        assert_eq!(
            decoded.pixels_bhwc,
            vec![1.0, 0.0, 128.0 / 255.0, 0.0, 64.0 / 255.0, 1.0]
        );
        assert_eq!(decoded.mask_bhw, vec![0.0, 1.0]);
        assert!(decoded.has_alpha);
        Ok(())
    }

    #[test]
    fn rgb_round_trip_embeds_sim_metadata() -> Result<(), PngError> {
        let metadata = BTreeMap::from([
            ("prompt".to_owned(), "{\"1\":{}}".to_owned()),
            ("workflow".to_owned(), "{\"version\":0.4}".to_owned()),
        ]);
        let encoded = encode_png_frame(
            &[0.0, 0.5, 1.0, 1.0, 0.25, 0.0],
            1,
            1,
            2,
            3,
            0,
            &metadata,
            PngLimits::default(),
        )?;
        let decoded = decode_png(&encoded, PngLimits::default())?;
        assert_eq!(
            decoded.metadata.sim_metadata().prompt,
            metadata.get("prompt").cloned()
        );
        assert_eq!(
            decoded.metadata.sim_metadata().workflow,
            metadata.get("workflow").cloned()
        );
        assert_eq!(
            decoded.pixels_bhwc,
            vec![0.0, 127.0 / 255.0, 1.0, 1.0, 63.0 / 255.0, 0.0]
        );
        assert_eq!((decoded.mask_width, decoded.mask_height), (64, 64));
        assert!(decoded.mask_bhw.iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn disabled_metadata_policy_omits_sim_fields_without_changing_pixels() -> Result<(), PngError> {
        let pixels = [0.0, 0.5, 1.0];
        let metadata = BTreeMap::from([
            ("prompt".to_owned(), "{\"1\":{}}".to_owned()),
            ("workflow".to_owned(), "{\"version\":0.4}".to_owned()),
        ]);
        let encoded = encode_png_frame_with_policy(
            &pixels,
            1,
            1,
            1,
            3,
            0,
            &metadata,
            MetadataWritePolicy {
                metadata_enabled: false,
            },
            PngLimits::default(),
        )?;
        let decoded = decode_png(&encoded, PngLimits::default())?;
        assert_eq!(decoded.pixels_bhwc, [0.0, 127.0 / 255.0, 1.0]);
        assert_eq!(decoded.metadata.sim_metadata(), Default::default());
        Ok(())
    }

    #[test]
    fn decode_rejects_limits_before_unbounded_allocation() {
        let error = decode_png(
            &[0; 9],
            PngLimits {
                max_input_bytes: 8,
                ..PngLimits::default()
            },
        );
        assert_eq!(
            error,
            Err(PngError::InputTooLarge {
                actual: 9,
                limit: 8,
            })
        );
    }
}
