use crate::resources::{
    AdmittedCpuImage, AlphaAssociation, CpuImageEdit, CpuImageOrigin, CpuImageParent,
    CpuImageResult, CpuMixParameters, CpuMixResult, CpuSampleFormat,
    GRAPHITE_IMAGE_SOURCE_REVISION, ImageColorSpace, ImageResourceIdentity,
};
use graphite_core_types::Color;
use graphite_raster_types::Image;
use sha2::{Digest, Sha256};
use std::{
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub struct PreparedPngAsset {
    input_identity: ImageResourceIdentity,
    input_origin: CpuImageOrigin,
    source_operation: Option<CpuImageEdit>,
    source_pixels_sha256: Option<[u8; 32]>,
    png: PreparedPngImage,
}

#[derive(Clone, Debug)]
pub struct PreparedPngImage {
    width: u32,
    height: u32,
    encoded_bytes: Arc<[u8]>,
    content_sha256: [u8; 32],
}

impl PreparedPngImage {
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn encoded_bytes(&self) -> &[u8] {
        &self.encoded_bytes
    }
    pub fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }
    pub fn sample_format(&self) -> CpuSampleFormat {
        CpuSampleFormat::Rgba8
    }
    pub fn color_space(&self) -> ImageColorSpace {
        ImageColorSpace::Srgb
    }
    pub fn alpha(&self) -> AlphaAssociation {
        AlphaAssociation::Straight
    }
    pub fn content_type(&self) -> &'static str {
        "image/png"
    }
}

#[derive(Clone, Debug)]
pub struct PreparedRgba8Png {
    png: PreparedPngImage,
    pixels_sha256: [u8; 32],
}

impl PreparedRgba8Png {
    pub fn png(&self) -> &PreparedPngImage {
        &self.png
    }

    pub fn pixels_sha256(&self) -> [u8; 32] {
        self.pixels_sha256
    }
}

#[derive(Clone, Debug)]
pub struct PreparedMixPngAsset {
    parents: [CpuImageParent; 2],
    parameters: CpuMixParameters,
    pixels_sha256: [u8; 32],
    png: PreparedPngImage,
}

impl PreparedMixPngAsset {
    pub fn parents(&self) -> &[CpuImageParent; 2] {
        &self.parents
    }
    pub fn parameters(&self) -> CpuMixParameters {
        self.parameters
    }
    pub fn pixels_sha256(&self) -> [u8; 32] {
        self.pixels_sha256
    }
    pub fn png(&self) -> &PreparedPngImage {
        &self.png
    }
    pub fn source_operation(&self) -> &'static str {
        "raster_nodes::blending_nodes::MixNode"
    }
    pub fn source_revision(&self) -> &'static str {
        GRAPHITE_IMAGE_SOURCE_REVISION
    }
}

impl PreparedPngAsset {
    pub fn input_identity(&self) -> &ImageResourceIdentity {
        &self.input_identity
    }

    pub fn input_digest_representation(&self) -> &'static str {
        self.input_origin.digest_representation()
    }

    pub fn input_origin(&self) -> CpuImageOrigin {
        self.input_origin
    }

    pub fn source_operation(&self) -> Option<CpuImageEdit> {
        self.source_operation
    }

    pub fn source_pixels_sha256(&self) -> Option<[u8; 32]> {
        self.source_pixels_sha256
    }

    pub fn source_revision(&self) -> &'static str {
        GRAPHITE_IMAGE_SOURCE_REVISION
    }

    pub fn png(&self) -> &PreparedPngImage {
        &self.png
    }
    pub fn width(&self) -> u32 {
        self.png.width()
    }
    pub fn height(&self) -> u32 {
        self.png.height()
    }
    pub fn encoded_bytes(&self) -> &[u8] {
        self.png.encoded_bytes()
    }
    pub fn content_sha256(&self) -> [u8; 32] {
        self.png.content_sha256()
    }
    pub fn sample_format(&self) -> CpuSampleFormat {
        self.png.sample_format()
    }
    pub fn color_space(&self) -> ImageColorSpace {
        self.png.color_space()
    }
    pub fn alpha(&self) -> AlphaAssociation {
        self.png.alpha()
    }
    pub fn content_type(&self) -> &'static str {
        self.png.content_type()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PngExportError {
    InvalidPixelLayout,
    SceneRenderRequired,
    SourceEncodingFailed,
    EncodedSizeExceeded { actual: usize, limit: usize },
}

impl fmt::Display for PngExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPixelLayout => formatter
                .write_str("The PNG requires positive dimensions and exact straight RGBA8 rows."),
            Self::SceneRenderRequired => formatter.write_str(
                "This image requires scene rendering before its pixels can be exported.",
            ),
            Self::SourceEncodingFailed => {
                formatter.write_str("The original image encoder could not prepare the PNG.")
            }
            Self::EncodedSizeExceeded { actual, limit } => write!(
                formatter,
                "The PNG contains {actual} bytes, exceeding the {limit} byte output limit."
            ),
        }
    }
}

impl std::error::Error for PngExportError {}

/// Encodes owned straight sRGB RGBA8 rows without a color or alpha conversion.
/// Color space and alpha describe the API profile, not embedded PNG metadata.
/// The encoded cap is checked after the shared codec runs, not its peak heap.
pub fn prepare_rgba8_png(
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    max_encoded_bytes: usize,
) -> Result<PreparedRgba8Png, PngExportError> {
    let required = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(PngExportError::InvalidPixelLayout)?;
    if width == 0 || height == 0 || pixels.len() != required {
        return Err(PngExportError::InvalidPixelLayout);
    }
    let pixels_sha256 = Sha256::digest(&pixels).into();
    let image = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or(PngExportError::InvalidPixelLayout)?;
    let encoded = sim_media::encode_png(&image::DynamicImage::ImageRgba8(image))
        .map_err(|_| PngExportError::SourceEncodingFailed)?;
    if encoded.len() > max_encoded_bytes {
        return Err(PngExportError::EncodedSizeExceeded {
            actual: encoded.len(),
            limit: max_encoded_bytes,
        });
    }
    Ok(PreparedRgba8Png {
        png: PreparedPngImage {
            width,
            height,
            content_sha256: Sha256::digest(&encoded).into(),
            encoded_bytes: encoded.into(),
        },
        pixels_sha256,
    })
}

pub fn prepare_png(
    image: &AdmittedCpuImage,
    max_encoded_bytes: usize,
) -> Result<PreparedPngAsset, PngExportError> {
    if image.descriptor().transform != [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
        return Err(PngExportError::SceneRenderRequired);
    }
    prepare_source_png(
        image.image(),
        image.identity(),
        image.origin(),
        None,
        None,
        max_encoded_bytes,
    )
}

pub fn prepare_cpu_result_png(
    result: &CpuImageResult,
    max_encoded_bytes: usize,
) -> Result<PreparedPngAsset, PngExportError> {
    if result.requires_scene_render() {
        return Err(PngExportError::SceneRenderRequired);
    }
    prepare_source_png(
        result.image(),
        result.input_identity(),
        result.input_origin(),
        Some(result.operation()),
        Some(result.pixels_sha256()),
        max_encoded_bytes,
    )
}

pub fn prepare_cpu_mix_png(
    result: &CpuMixResult,
    max_encoded_bytes: usize,
) -> Result<PreparedMixPngAsset, PngExportError> {
    if result.requires_scene_render() {
        return Err(PngExportError::SceneRenderRequired);
    }
    Ok(PreparedMixPngAsset {
        parents: result.parents().clone(),
        parameters: result.parameters(),
        pixels_sha256: result.pixels_sha256(),
        png: prepare_encoded_png(result.image(), max_encoded_bytes)?,
    })
}

fn prepare_source_png(
    image: &Image<Color>,
    input_identity: &ImageResourceIdentity,
    input_origin: CpuImageOrigin,
    source_operation: Option<CpuImageEdit>,
    source_pixels_sha256: Option<[u8; 32]>,
    max_encoded_bytes: usize,
) -> Result<PreparedPngAsset, PngExportError> {
    Ok(PreparedPngAsset {
        input_identity: input_identity.clone(),
        input_origin,
        source_operation,
        source_pixels_sha256,
        png: prepare_encoded_png(image, max_encoded_bytes)?,
    })
}

fn prepare_encoded_png(
    image: &Image<Color>,
    max_encoded_bytes: usize,
) -> Result<PreparedPngImage, PngExportError> {
    let encoded = encode_source_image(image)?;
    // The original encoder allocates before this result cap can be checked; it is not a codec working-memory budget.
    if encoded.len() > max_encoded_bytes {
        return Err(PngExportError::EncodedSizeExceeded {
            actual: encoded.len(),
            limit: max_encoded_bytes,
        });
    }
    let content_sha256 = Sha256::digest(&encoded).into();
    Ok(PreparedPngImage {
        width: image.width,
        height: image.height,
        encoded_bytes: encoded.into(),
        content_sha256,
    })
}

fn encode_source_image(image: &Image<Color>) -> Result<Vec<u8>, PngExportError> {
    catch_unwind(AssertUnwindSafe(|| image.to_png()))
        .map_err(|_| PngExportError::SourceEncodingFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{
        CpuImageBudget, CpuImageDescriptor, admit_cpu_image, evaluate_cpu_image_edit,
    };

    const PIXELS: [u8; 16] = [
        255, 0, 0, 255, 0, 255, 0, 128, 17, 83, 201, 0, 0, 0, 255, 64,
    ];

    fn descriptor() -> CpuImageDescriptor {
        CpuImageDescriptor {
            width: 2,
            height: 2,
            row_stride: 8,
            sample_format: CpuSampleFormat::Rgba8,
            color_space: ImageColorSpace::Srgb,
            alpha: AlphaAssociation::Straight,
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    fn image(descriptor: CpuImageDescriptor) -> AdmittedCpuImage {
        admit_cpu_image(
            ImageResourceIdentity {
                asset_id: "project/immutable-image".into(),
                version_id: "input-version-1".into(),
                content_sha256: Sha256::digest(PIXELS).into(),
            },
            descriptor,
            Arc::<[u8]>::from(PIXELS),
            CpuImageBudget {
                max_input_bytes: PIXELS.len(),
                max_decoded_bytes: 256,
            },
        )
        .expect("the original pixel conversion admits this exact hashed fixture")
    }

    #[test]
    fn at_graphite_png_preparation_is_exact_original_encoder_output() {
        let image = image(descriptor());
        let original_bytes = image.image().to_png();
        let prepared =
            prepare_png(&image, original_bytes.len()).expect("exact result cap admits PNG");
        assert_eq!(prepared.encoded_bytes(), original_bytes);
        let repeated =
            prepare_png(&image, original_bytes.len()).expect("same immutable input repeats");
        assert_eq!(repeated.encoded_bytes(), prepared.encoded_bytes());
        assert_eq!(repeated.content_sha256(), prepared.content_sha256());
    }

    #[test]
    fn at_graphite_png_preparation_preserves_original_dimensions_color_and_alpha() {
        let image = image(descriptor());
        let prepared = prepare_png(&image, 4096).expect("bounded original PNG is prepared");
        let decoded = Image::<Color>::from_encoded(prepared.encoded_bytes())
            .expect("the original decoder admits its actual PNG output");
        let (actual, width, height) = decoded.to_flat_u8();
        assert_eq!((width, height), (2, 2));
        assert_eq!(
            actual,
            [255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 0, 0, 0, 0, 255, 64]
        );
        assert_eq!(decoded.to_flat_u8(), image.image().to_flat_u8());
    }

    #[test]
    fn at_graphite_png_preparation_keeps_input_immutable_and_provenance_explicit() {
        let image = image(descriptor());
        let before_identity = image.identity().clone();
        let before_descriptor = *image.descriptor();
        let before_source = image.image().clone();
        let before_pixels = image.srgba8_pixels().collect::<Vec<_>>();
        let prepared = prepare_png(&image, 4096).expect("PNG is prepared without input mutation");
        assert_eq!(image.identity(), &before_identity);
        assert_eq!(image.descriptor(), &before_descriptor);
        assert_eq!(image.image(), &before_source);
        assert_eq!(image.srgba8_pixels().collect::<Vec<_>>(), before_pixels);
        assert_eq!(prepared.input_identity(), &before_identity);
        assert_eq!(
            prepared.source_revision(),
            "28ed22824fae8828756f6f4163db62f3585f21e0"
        );
        assert_eq!((prepared.width(), prepared.height()), (2, 2));
        assert_eq!(prepared.sample_format(), CpuSampleFormat::Rgba8);
        assert_eq!(prepared.color_space(), ImageColorSpace::Srgb);
        assert_eq!(prepared.alpha(), AlphaAssociation::Straight);
        assert_eq!(prepared.content_type(), "image/png");
        let actual_digest: [u8; 32] = Sha256::digest(prepared.encoded_bytes()).into();
        assert_eq!(prepared.content_sha256(), actual_digest);
        assert_ne!(prepared.content_sha256(), image.identity().content_sha256);
    }

    #[test]
    fn at_graphite_png_preparation_size_failure_retains_input_and_recovers() {
        let image = image(descriptor());
        let input_identity = image.identity().clone();
        let source = image.image().clone();
        let original_bytes = source.to_png();
        for limit in [0, original_bytes.len() - 1] {
            assert_eq!(
                prepare_png(&image, limit).expect_err("result-size cap rejects this PNG"),
                PngExportError::EncodedSizeExceeded {
                    actual: original_bytes.len(),
                    limit
                }
            );
            assert_eq!(image.identity(), &input_identity);
            assert_eq!(image.image(), &source);
        }
        let recovered =
            prepare_png(&image, original_bytes.len()).expect("larger result cap succeeds");
        assert_eq!(recovered.encoded_bytes(), original_bytes);
    }

    #[test]
    fn at_graphite_png_preparation_rejects_scene_transform_without_encoding_or_mutation() {
        for transform in [
            [1.0, 0.0, 0.0, 1.0, 10.0, 20.0],
            [2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
        ] {
            let descriptor = CpuImageDescriptor {
                transform,
                ..descriptor()
            };
            let image = image(descriptor);
            let source = image.image().clone();
            let input_identity = image.identity().clone();
            assert_eq!(
                prepare_png(&image, 0).expect_err("rendering must precede transformed export"),
                PngExportError::SceneRenderRequired
            );
            assert_eq!(image.descriptor(), &descriptor);
            assert_eq!(image.identity(), &input_identity);
            assert_eq!(image.image(), &source);
        }
    }

    #[test]
    fn at_graphite_png_preparation_contains_original_malformed_image_assertion() {
        let malformed = Image::<Color> {
            width: 1,
            height: 1,
            data: Vec::new(),
            base64_string: None,
        };
        assert_eq!(
            encode_source_image(&malformed),
            Err(PngExportError::SourceEncodingFailed)
        );
        assert_eq!(malformed.width, 1);
        assert_eq!(malformed.height, 1);
        assert!(malformed.data.is_empty());
        assert!(malformed.base64_string.is_none());
    }
    #[test]
    fn at_graphite_cpu_result_png_preserves_original_encoding_and_result_provenance() {
        for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
            let input = image(descriptor());
            let input_identity = input.identity().clone();
            let input_descriptor = *input.descriptor();
            let input_image = input.image().clone();
            let result = evaluate_cpu_image_edit(&input, operation, 4096)
                .expect("the original node prepares an immutable CPU result");
            let source_image = result.image().clone();
            let source_attributes = result.attributes().clone();
            let source_descriptor = *result.descriptor();
            let source_pixel_bits = result
                .image()
                .data
                .iter()
                .map(|color| {
                    [
                        color.r().to_bits(),
                        color.g().to_bits(),
                        color.b().to_bits(),
                        color.a().to_bits(),
                    ]
                })
                .collect::<Vec<_>>();
            let source_digest = result.pixels_sha256();
            let original_bytes = result.image().to_png();
            let prepared = prepare_cpu_result_png(&result, original_bytes.len())
                .expect("exact output cap admits original result PNG");
            assert_eq!(prepared.encoded_bytes(), original_bytes);
            assert_eq!(
                prepared.encoded_bytes().get(..8),
                Some(&[137, 80, 78, 71, 13, 10, 26, 10][..])
            );
            assert_eq!(prepared.encoded_bytes().get(12..16), Some(&b"IHDR"[..]));
            assert_eq!(
                prepared.encoded_bytes().get(16..24),
                Some(&[0, 0, 0, 2, 0, 0, 0, 2][..])
            );
            assert_eq!(prepared.encoded_bytes().get(24..26), Some(&[8, 6][..]));
            let decoded = Image::<Color>::from_encoded(prepared.encoded_bytes())
                .expect("original decoder reads actual source encoder output");
            assert_eq!(decoded.to_flat_u8(), result.image().to_flat_u8());
            assert_eq!(prepared.input_identity(), &input_identity);
            assert_eq!(
                prepared.input_digest_representation(),
                "cpu-rgba8-rows-sha256"
            );
            assert_eq!(prepared.source_operation(), Some(operation));
            assert_eq!(prepared.source_pixels_sha256(), Some(source_digest));
            assert_eq!(
                prepared.source_revision(),
                "28ed22824fae8828756f6f4163db62f3585f21e0"
            );
            assert_eq!((prepared.width(), prepared.height()), (2, 2));
            assert_eq!(prepared.sample_format(), CpuSampleFormat::Rgba8);
            assert_eq!(prepared.color_space(), ImageColorSpace::Srgb);
            assert_eq!(prepared.alpha(), AlphaAssociation::Straight);
            assert_eq!(prepared.content_type(), "image/png");
            let encoded_digest: [u8; 32] = Sha256::digest(prepared.encoded_bytes()).into();
            assert_eq!(prepared.content_sha256(), encoded_digest);
            assert_ne!(prepared.content_sha256(), source_digest);
            assert_ne!(prepared.content_sha256(), input_identity.content_sha256);
            let repeated = prepare_cpu_result_png(&result, original_bytes.len())
                .expect("same immutable source result repeats");
            assert_eq!(repeated.encoded_bytes(), prepared.encoded_bytes());
            assert_eq!(repeated.content_sha256(), prepared.content_sha256());
            assert_eq!(repeated.source_operation(), prepared.source_operation());
            assert_eq!(
                repeated.source_pixels_sha256(),
                prepared.source_pixels_sha256()
            );
            assert_eq!(result.image(), &source_image);
            assert_eq!(
                result
                    .image()
                    .data
                    .iter()
                    .map(|color| {
                        [
                            color.r().to_bits(),
                            color.g().to_bits(),
                            color.b().to_bits(),
                            color.a().to_bits(),
                        ]
                    })
                    .collect::<Vec<_>>(),
                source_pixel_bits
            );
            assert_eq!(result.attributes(), &source_attributes);
            assert_eq!(result.descriptor(), &source_descriptor);
            assert_eq!(result.pixels_sha256(), source_digest);
            assert_eq!(result.input_identity(), &input_identity);
            assert_eq!(result.operation(), operation);
            assert_eq!(input.image(), &input_image);
            assert_eq!(input.identity(), &input_identity);
            assert_eq!(input.descriptor(), &input_descriptor);
            let admitted =
                prepare_png(&input, 4096).expect("admitted image path remains available");
            assert_eq!(admitted.source_operation(), None);
            assert_eq!(admitted.source_pixels_sha256(), None);
        }
    }

    #[test]
    fn at_graphite_cpu_result_png_size_failure_preserves_result_and_recovers() {
        for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
            let input = image(descriptor());
            let input_image = input.image().clone();
            let result = evaluate_cpu_image_edit(&input, operation, 4096)
                .expect("original operation prepares CPU result");
            let source_image = result.image().clone();
            let source_attributes = result.attributes().clone();
            let source_descriptor = *result.descriptor();
            let source_identity = result.input_identity().clone();
            let source_digest = result.pixels_sha256();
            let original_bytes = result.image().to_png();
            for limit in [0, original_bytes.len() - 1] {
                assert_eq!(
                    prepare_cpu_result_png(&result, limit).expect_err("encoded output exceeds cap"),
                    PngExportError::EncodedSizeExceeded {
                        actual: original_bytes.len(),
                        limit
                    }
                );
                assert_eq!(result.image(), &source_image);
                assert_eq!(result.attributes(), &source_attributes);
                assert_eq!(result.descriptor(), &source_descriptor);
                assert_eq!(result.input_identity(), &source_identity);
                assert_eq!(result.pixels_sha256(), source_digest);
                assert_eq!(result.operation(), operation);
                assert_eq!(input.image(), &input_image);
            }
            let recovered = prepare_cpu_result_png(&result, original_bytes.len())
                .expect("exact result-size cap recovers without changing the result");
            assert_eq!(recovered.encoded_bytes(), original_bytes);
            assert_eq!(recovered.source_operation(), Some(operation));
            assert_eq!(recovered.source_pixels_sha256(), Some(source_digest));
        }
    }

    #[test]
    fn at_graphite_cpu_result_png_rejects_scene_results_before_output_cap() {
        for transform in [
            [1.0, 0.0, 0.0, 1.0, 10.0, 20.0],
            [2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
        ] {
            for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
                let input = image(CpuImageDescriptor {
                    transform,
                    ..descriptor()
                });
                let input_image = input.image().clone();
                let result = evaluate_cpu_image_edit(&input, operation, 4096)
                    .expect("original node retains the transformed source result");
                let source_image = result.image().clone();
                let source_attributes = result.attributes().clone();
                let source_identity = result.input_identity().clone();
                let source_descriptor = *result.descriptor();
                let source_digest = result.pixels_sha256();
                assert!(result.requires_scene_render());
                assert_eq!(result.descriptor().transform, transform);
                assert_eq!(
                    prepare_cpu_result_png(&result, 0)
                        .expect_err("scene rendering precedes encoding"),
                    PngExportError::SceneRenderRequired
                );
                assert_eq!(result.image(), &source_image);
                assert_eq!(result.attributes(), &source_attributes);
                assert_eq!(result.input_identity(), &source_identity);
                assert_eq!(result.descriptor(), &source_descriptor);
                assert_eq!(result.pixels_sha256(), source_digest);
                assert_eq!(result.operation(), operation);
                assert_eq!(input.image(), &input_image);
            }
        }
    }

    #[test]
    fn at_graphite_encoded_result_png_preserves_encoded_parent_provenance() {
        use crate::resources::{EncodedPngBudget, admit_encoded_png};
        let original = image(descriptor());
        let encoded: Arc<[u8]> = original.image().to_png().into();
        let parent = ImageResourceIdentity {
            asset_id: "project/linked-encoded-image".into(),
            version_id: "supplied-immutable-png-version".into(),
            content_sha256: Sha256::digest(encoded.as_ref()).into(),
        };
        let input = admit_encoded_png(
            parent.clone(),
            [1., 0., 0., 1., 0., 0.],
            encoded.clone(),
            EncodedPngBudget {
                max_encoded_bytes: encoded.len(),
                max_pixel_bytes: 4096,
                max_metadata_bytes: 256,
            },
        )
        .expect("original encoded parent admitted");
        assert_ne!(parent.content_sha256, original.identity().content_sha256);
        for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
            let result = evaluate_cpu_image_edit(&input, operation, 4096).expect("original edit");
            let before = result.image().clone();
            let prepared = prepare_cpu_result_png(&result, 4096).expect("original target PNG");
            assert_eq!(prepared.input_identity(), &parent);
            assert_eq!(prepared.input_origin(), input.origin());
            assert_eq!(
                prepared.input_digest_representation(),
                "encoded-png-bytes-sha256"
            );
            assert_eq!(prepared.input_origin().byte_length(), encoded.len());
            assert!(prepared.input_origin().decode_profile().is_some());
            assert_eq!(prepared.source_operation(), Some(operation));
            assert_eq!(
                prepared.source_pixels_sha256(),
                Some(result.pixels_sha256())
            );
            assert_ne!(prepared.content_sha256(), parent.content_sha256);
            assert_ne!(prepared.content_sha256(), result.pixels_sha256());
            assert_eq!(prepared.encoded_bytes(), result.image().to_png());
            assert_eq!(result.image(), &before);
            assert_eq!(
                parent.content_sha256,
                <[u8; 32]>::from(Sha256::digest(encoded.as_ref()))
            );
        }
        let prepared = prepare_png(&input, 4096).expect("unedited PNG preparation shares encoder");
        assert_eq!(prepared.input_origin(), input.origin());
        assert_eq!(prepared.input_identity(), &parent);
        assert_eq!(
            prepared.input_digest_representation(),
            "encoded-png-bytes-sha256"
        );
        assert_eq!(prepared.source_operation(), None);
    }

    #[test]
    fn at_graphite_black_and_white_png_preserves_parent_parameters_and_baked_profile() {
        use crate::resources::{CpuBlackAndWhiteParameters, evaluate_cpu_image_edit};
        let source = image(descriptor());
        let before = source.image().clone();
        for use_tint in [false, true] {
            let parameters = CpuBlackAndWhiteParameters {
                use_tint,
                tint_gamma_srgb: [0.8, 0.4, 0.2, 0.1],
                reds: 41.,
                yellows: 62.,
                greens: 43.,
                cyans: 64.,
                blues: 25.,
                magentas: 86.,
            };
            let operation = CpuImageEdit::BlackAndWhite(parameters);
            let result = evaluate_cpu_image_edit(&source, operation, 64).expect("original edit");
            let expected = result.image().to_png();
            let prepared = prepare_cpu_result_png(&result, expected.len()).expect("baked result");
            assert_eq!(prepared.input_identity(), source.identity());
            assert_eq!(prepared.input_origin(), source.origin());
            assert_eq!(prepared.source_operation(), Some(operation));
            assert_eq!(
                prepared.source_pixels_sha256(),
                Some(result.pixels_sha256())
            );
            assert_eq!(prepared.source_revision(), GRAPHITE_IMAGE_SOURCE_REVISION);
            assert_eq!(prepared.sample_format(), CpuSampleFormat::Rgba8);
            assert_eq!(prepared.color_space(), ImageColorSpace::Srgb);
            assert_eq!(prepared.alpha(), AlphaAssociation::Straight);
            assert_eq!(prepared.encoded_bytes(), expected);
            let decoded =
                Image::<Color>::from_encoded(prepared.encoded_bytes()).expect("source PNG decode");
            assert_eq!(decoded.to_flat_u8(), result.image().to_flat_u8());
            assert_eq!(
                prepared.content_sha256(),
                <[u8; 32]>::from(Sha256::digest(prepared.encoded_bytes()))
            );
            assert!(matches!(
                prepare_cpu_result_png(&result, expected.len() - 1),
                Err(PngExportError::EncodedSizeExceeded { .. })
            ));
            let recovered = prepare_cpu_result_png(&result, expected.len()).expect("cap recovery");
            assert_eq!(recovered.content_sha256(), prepared.content_sha256());
            assert_eq!(recovered.source_operation(), Some(operation));
            assert_eq!(source.image(), &before);
        }
    }

    #[test]
    fn at_graphite_brightness_contrast_png_preserves_parent_and_original_operation() {
        use crate::resources::{CpuBrightnessContrastParameters, evaluate_cpu_image_edit};
        let source = image(descriptor());
        let before = source.image().clone();
        for use_classic in [false, true] {
            let operation = CpuImageEdit::BrightnessContrast(CpuBrightnessContrastParameters {
                brightness: 63.,
                contrast: -37.,
                use_classic,
                classic_pivot: 81.,
            });
            let result = evaluate_cpu_image_edit(&source, operation, 64).expect("original edit");
            let expected = result.image().to_png();
            let prepared = prepare_cpu_result_png(&result, expected.len()).expect("baked result");
            assert_eq!(prepared.input_identity(), source.identity());
            assert_eq!(prepared.input_origin(), source.origin());
            assert_eq!(prepared.source_operation(), Some(operation));
            assert_eq!(
                prepared.source_pixels_sha256(),
                Some(result.pixels_sha256())
            );
            assert_eq!(prepared.source_revision(), GRAPHITE_IMAGE_SOURCE_REVISION);
            assert_eq!(prepared.sample_format(), CpuSampleFormat::Rgba8);
            assert_eq!(prepared.color_space(), ImageColorSpace::Srgb);
            assert_eq!(prepared.alpha(), AlphaAssociation::Straight);
            let decoded =
                Image::<Color>::from_encoded(prepared.encoded_bytes()).expect("source PNG decode");
            assert_eq!(decoded.to_flat_u8(), result.image().to_flat_u8());
            assert_eq!(prepared.encoded_bytes(), expected);
            assert_eq!(
                prepared.content_sha256(),
                <[u8; 32]>::from(Sha256::digest(prepared.encoded_bytes()))
            );
            assert!(matches!(
                prepare_cpu_result_png(&result, expected.len() - 1),
                Err(PngExportError::EncodedSizeExceeded { .. })
            ));
            let recovered = prepare_cpu_result_png(&result, expected.len()).expect("cap recovery");
            assert_eq!(recovered.content_sha256(), prepared.content_sha256());
            assert_eq!(recovered.source_operation(), Some(operation));
            assert_eq!(source.image(), &before);
        }
    }

    #[test]
    fn at_r22_1_straight_rgba_png_preserves_channels_and_partial_alpha_without_color_roundtrip() {
        let pixels = vec![
            231, 17, 83, 0, 19, 223, 71, 1, 11, 53, 241, 127, 201, 31, 97, 255,
        ];
        let prepared = prepare_rgba8_png(2, 2, pixels.clone(), 4096).expect("byte encoder");
        assert_eq!(
            prepared.png().encoded_bytes(),
            include_bytes!("../test_data/original-frontend-png-rgba8-2x2.png").as_slice()
        );
        let decoded = image::load_from_memory(prepared.png().encoded_bytes())
            .expect("PNG decode")
            .into_rgba8();
        assert_eq!(decoded.dimensions(), (2, 2));
        assert_eq!(decoded.as_raw(), &pixels);
        assert_eq!(
            prepared.pixels_sha256(),
            <[u8; 32]>::from(Sha256::digest(&pixels))
        );
        assert_eq!(
            prepared.png().content_sha256(),
            <[u8; 32]>::from(Sha256::digest(prepared.png().encoded_bytes()))
        );
        assert_ne!(prepared.pixels_sha256(), prepared.png().content_sha256());
        assert_eq!(prepared.png().sample_format(), CpuSampleFormat::Rgba8);
        assert_eq!(prepared.png().color_space(), ImageColorSpace::Srgb);
        assert_eq!(prepared.png().alpha(), AlphaAssociation::Straight);
        let size = prepared.png().encoded_bytes().len();
        for cap in [0, size - 1] {
            assert!(
                matches!(prepare_rgba8_png(2, 2, pixels.clone(), cap), Err(PngExportError::EncodedSizeExceeded { actual, limit }) if actual == size && limit == cap)
            );
        }
        let recovered = prepare_rgba8_png(2, 2, pixels, size).expect("exact encoded cap recovery");
        assert_eq!(
            recovered.png().encoded_bytes(),
            prepared.png().encoded_bytes()
        );
    }

    #[test]
    fn at_r22_1_straight_rgba_png_rejects_invalid_layout_and_recovers() {
        for (width, height, pixels) in [
            (0, 1, vec![]),
            (1, 0, vec![]),
            (1, 1, vec![0; 3]),
            (1, 1, vec![0; 5]),
            (u32::MAX, u32::MAX, vec![]),
        ] {
            assert!(matches!(
                prepare_rgba8_png(width, height, pixels, 4096),
                Err(PngExportError::InvalidPixelLayout)
            ));
        }
        let prepared =
            prepare_rgba8_png(1, 1, vec![9, 23, 57, 127], 4096).expect("layout recovery");
        assert_eq!(
            image::load_from_memory(prepared.png().encoded_bytes())
                .expect("decode")
                .into_rgba8()
                .as_raw(),
            &[9, 23, 57, 127]
        );
    }

    fn assert_mix_parent_provenance(actual: &[CpuImageParent; 2], expected: &[CpuImageParent; 2]) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.identity(), expected.identity());
            assert_eq!(actual.origin(), expected.origin());
            assert_eq!(actual.descriptor(), expected.descriptor());
            assert_eq!(
                actual.descriptor().transform.map(f64::to_bits),
                expected.descriptor().transform.map(f64::to_bits)
            );
            assert_eq!(actual.pixels_sha256(), expected.pixels_sha256());
        }
    }

    fn mix_export_fixture() -> [AdmittedCpuImage; 2] {
        let over = image(CpuImageDescriptor {
            transform: [1., -0., 0., 1., -0., 0.],
            ..descriptor()
        });
        let under_pixels: [u8; 16] = [
            31, 79, 211, 255, 193, 47, 109, 191, 241, 191, 33, 255, 73, 157, 29, 127,
        ];
        let under = admit_cpu_image(
            ImageResourceIdentity {
                asset_id: "project/under-image".into(),
                version_id: "supplied-under-version".into(),
                content_sha256: Sha256::digest(under_pixels).into(),
            },
            CpuImageDescriptor {
                transform: [1., 0., -0., 1., 0., -0.],
                ..descriptor()
            },
            Arc::<[u8]>::from(under_pixels),
            CpuImageBudget {
                max_input_bytes: under_pixels.len(),
                max_decoded_bytes: 256,
            },
        )
        .expect("distinct actual hashed Under rows");
        [over, under]
    }

    #[test]
    fn at_r22_1_mix_png_matches_original_encoder_and_retains_ordered_source_provenance() {
        use crate::resources::{CpuBlendMode, CpuMixBudget, evaluate_cpu_image_mix};
        use graphite_core_types::list::{ATTR_TRANSFORM, Item};
        use graphite_glam::DAffine2;
        use graphite_raster_types::Raster;

        let parents = mix_export_fixture();
        let before_images = parents.each_ref().map(|parent| parent.image().clone());
        let before_identities = parents.each_ref().map(|parent| parent.identity().clone());
        let before_descriptors = parents.each_ref().map(|parent| *parent.descriptor());
        for parameters in [
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 100.,
            },
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 37.5,
            },
            CpuMixParameters {
                blend_mode: CpuBlendMode::Multiply,
                opacity: 73.25,
            },
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: -0.,
            },
        ] {
            let result = evaluate_cpu_image_mix(
                &parents[0],
                &parents[1],
                parameters,
                CpuMixBudget {
                    max_parent_bytes: 128,
                    max_result_bytes: 64,
                },
            )
            .expect("genuine original Mix result");
            let original = crate::raster::evaluate_mix_cpu(
                Item::new_from_element(Raster::new_cpu(parents[0].image().clone())).with_attribute(
                    ATTR_TRANSFORM,
                    DAffine2::from_cols_array(&before_descriptors[0].transform),
                ),
                Item::new_from_element(Raster::new_cpu(parents[1].image().clone())).with_attribute(
                    ATTR_TRANSFORM,
                    DAffine2::from_cols_array(&before_descriptors[1].transform),
                ),
                Item::new_from_element(parameters.blend_mode),
                Item::new_from_element(parameters.opacity),
            )
            .expect("independent original generated Mix constructor");
            let original_bytes = original.element().data().to_png();
            let prepared = prepare_cpu_mix_png(&result, original_bytes.len()).expect("exact cap");
            assert_eq!(prepared.png().encoded_bytes(), original_bytes);
            assert_mix_parent_provenance(prepared.parents(), result.parents());
            assert_eq!(prepared.parameters().blend_mode, parameters.blend_mode);
            assert_eq!(
                prepared.parameters().opacity.to_bits(),
                parameters.opacity.to_bits()
            );
            assert_eq!(prepared.pixels_sha256(), result.pixels_sha256());
            assert_eq!(prepared.source_operation(), result.source_operation());
            assert_eq!(prepared.source_revision(), GRAPHITE_IMAGE_SOURCE_REVISION);
            assert_eq!(prepared.png().sample_format(), CpuSampleFormat::Rgba8);
            assert_eq!(prepared.png().color_space(), ImageColorSpace::Srgb);
            assert_eq!(prepared.png().alpha(), AlphaAssociation::Straight);
            assert_eq!(prepared.png().content_type(), "image/png");
            assert_eq!(
                prepared.png().content_sha256(),
                <[u8; 32]>::from(Sha256::digest(&original_bytes))
            );
            assert_ne!(prepared.png().content_sha256(), prepared.pixels_sha256());
            let decoded = Image::<Color>::from_encoded(prepared.png().encoded_bytes())
                .expect("original decode");
            assert_eq!(decoded.to_flat_u8(), original.element().data().to_flat_u8());
            assert_eq!((prepared.png().width(), prepared.png().height()), (2, 2));
            if parameters
                == (CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 100.,
                })
            {
                let pixels = decoded.to_flat_u8().0;
                assert_eq!(pixels.get(..4), Some(&[255, 0, 0, 255][..]));
                assert_eq!(pixels.get(8..12), Some(&[241, 191, 33, 255][..]));
            }
            for index in 0..2 {
                assert_eq!(
                    prepared.parents()[index].identity(),
                    &before_identities[index]
                );
                assert_eq!(prepared.parents()[index].origin(), parents[index].origin());
                assert_eq!(
                    prepared.parents()[index]
                        .descriptor()
                        .transform
                        .map(f64::to_bits),
                    before_descriptors[index].transform.map(f64::to_bits)
                );
                assert_eq!(parents[index].identity(), &before_identities[index]);
                assert_eq!(parents[index].image(), &before_images[index]);
                assert_eq!(
                    parents[index].descriptor().transform.map(f64::to_bits),
                    before_descriptors[index].transform.map(f64::to_bits)
                );
            }
            assert_eq!(
                result.descriptor().transform.map(f64::to_bits),
                before_descriptors[0].transform.map(f64::to_bits)
            );
            let repeated = prepare_cpu_mix_png(&result, original_bytes.len()).expect("repeat");
            assert_eq!(
                repeated.png().encoded_bytes(),
                prepared.png().encoded_bytes()
            );
            assert_mix_parent_provenance(repeated.parents(), prepared.parents());
            assert_eq!(repeated.parameters(), parameters);
        }
        let parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Normal,
            opacity: 37.5,
        };
        let budget = CpuMixBudget {
            max_parent_bytes: 128,
            max_result_bytes: 64,
        };
        let forward = evaluate_cpu_image_mix(&parents[0], &parents[1], parameters, budget)
            .expect("ordered Mix");
        let reverse = evaluate_cpu_image_mix(&parents[1], &parents[0], parameters, budget)
            .expect("reverse Mix");
        assert_ne!(forward.image().to_flat_u8(), reverse.image().to_flat_u8());
        assert_ne!(
            prepare_cpu_mix_png(&forward, 4096)
                .expect("forward PNG")
                .png()
                .encoded_bytes(),
            prepare_cpu_mix_png(&reverse, 4096)
                .expect("reverse PNG")
                .png()
                .encoded_bytes()
        );
    }

    #[test]
    fn at_r22_1_mix_png_cap_failure_preserves_result_and_both_parents_then_recovers() {
        use crate::resources::{CpuBlendMode, CpuMixBudget, evaluate_cpu_image_mix};
        let parents = mix_export_fixture();
        let before_images = parents.each_ref().map(|parent| parent.image().clone());
        let before_identities = parents.each_ref().map(|parent| parent.identity().clone());
        let parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Multiply,
            opacity: 62.5,
        };
        let result = evaluate_cpu_image_mix(
            &parents[0],
            &parents[1],
            parameters,
            CpuMixBudget {
                max_parent_bytes: 128,
                max_result_bytes: 64,
            },
        )
        .expect("original Mix");
        let result_image = result.image().clone();
        let result_attributes = result.attributes().clone();
        let result_descriptor = *result.descriptor();
        let result_parents = result.parents().clone();
        let result_digest = result.pixels_sha256();
        let original_bytes = result.image().to_png();
        for limit in [0, original_bytes.len() - 1] {
            assert_eq!(
                prepare_cpu_mix_png(&result, limit).expect_err("bounded output rejection"),
                PngExportError::EncodedSizeExceeded {
                    actual: original_bytes.len(),
                    limit
                }
            );
            assert_eq!(result.image(), &result_image);
            assert_eq!(result.attributes(), &result_attributes);
            assert_eq!(
                result.descriptor().transform.map(f64::to_bits),
                result_descriptor.transform.map(f64::to_bits)
            );
            assert_mix_parent_provenance(result.parents(), &result_parents);
            assert_eq!(result.parameters(), parameters);
            assert_eq!(result.pixels_sha256(), result_digest);
            for index in 0..2 {
                assert_eq!(parents[index].identity(), &before_identities[index]);
                assert_eq!(parents[index].image(), &before_images[index]);
                assert_eq!(
                    parents[index].descriptor().transform.map(f64::to_bits),
                    result_parents[index]
                        .descriptor()
                        .transform
                        .map(f64::to_bits)
                );
            }
        }
        let recovered =
            prepare_cpu_mix_png(&result, original_bytes.len()).expect("exact cap recovery");
        assert_eq!(recovered.png().encoded_bytes(), original_bytes);
        assert_mix_parent_provenance(recovered.parents(), &result_parents);
        assert_eq!(recovered.parameters(), parameters);
        assert_eq!(recovered.pixels_sha256(), result_digest);
        assert_eq!(
            recovered.png().content_sha256(),
            <[u8; 32]>::from(Sha256::digest(&original_bytes))
        );
        assert_eq!(result.image(), &result_image);
        assert_eq!(result.attributes(), &result_attributes);
    }
}
