use std::{
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use gpui::{RenderImage, SMOOTH_SVG_SCALE_FACTOR, SvgRenderer};
use graphics_core::render::PreparedCpuSvgComposition;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuSvgPreviewError {
    InvalidScale,
    InvalidDimensions,
    RendererSizeLimit,
    SizeOverflow,
    PreviewBudgetExceeded { required: usize, maximum: usize },
    SourceConversion(String),
    SourceConversionPanicked,
    UnexpectedPixelLayout,
}

impl fmt::Display for CpuSvgPreviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScale => {
                formatter.write_str("The scene preview scale must be positive and finite")
            }
            Self::InvalidDimensions => {
                formatter.write_str("The scene preview dimensions cannot be represented")
            }
            Self::RendererSizeLimit => formatter
                .write_str("The scene preview exceeds the native SVG renderer's size limit"),
            Self::SizeOverflow => formatter.write_str("The scene preview pixel size overflowed"),
            Self::PreviewBudgetExceeded { required, maximum } => write!(
                formatter,
                "The scene preview requires {required} bytes, exceeding its {maximum} byte limit"
            ),
            Self::SourceConversion(error) => {
                write!(formatter, "The native SVG conversion failed: {error}")
            }
            Self::SourceConversionPanicked => {
                formatter.write_str("The native SVG conversion panicked")
            }
            Self::UnexpectedPixelLayout => {
                formatter.write_str("The native SVG conversion returned an unexpected pixel layout")
            }
        }
    }
}

impl std::error::Error for CpuSvgPreviewError {}

fn preview_layout(
    scene: &PreparedCpuSvgComposition,
    render_scale: f32,
) -> Result<(u32, u32, usize), CpuSvgPreviewError> {
    if !render_scale.is_finite() || render_scale <= 0. {
        return Err(CpuSvgPreviewError::InvalidScale);
    }
    let configuration = scene.configuration();
    // The original wrapper emits unitless f64 root lengths; usvg converts those
    // to f32 before GPUI multiplies them by its supersampled render scale.
    let width = (f64::from(configuration.viewport.resolution.x) / configuration.scale) as f32;
    let height = (f64::from(configuration.viewport.resolution.y) / configuration.scale) as f32;
    let scale = render_scale * SMOOTH_SVG_SCALE_FACTOR;
    if !width.is_finite()
        || !height.is_finite()
        || width <= 0.
        || height <= 0.
        || !scale.is_finite()
    {
        return Err(CpuSvgPreviewError::InvalidDimensions);
    }
    let width = width * scale;
    let height = height * scale;
    if !width.is_finite() || !height.is_finite() || width < 1. || height < 1. {
        return Err(CpuSvgPreviewError::InvalidDimensions);
    }
    // GPUI silently reduces either axis above 8192. Reject that case instead
    // of presenting a resized scene as the requested output.
    if width > 8192. || height > 8192. {
        return Err(CpuSvgPreviewError::RendererSizeLimit);
    }
    let width = width as u32;
    let height = height as u32;
    let bytes = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(CpuSvgPreviewError::SizeOverflow)?;
    Ok((width, height, bytes))
}

/// Converts an already prepared original CPU scene using GPUI's existing SVG
/// renderer. The returned pixels retain GPUI's straight BGRA conversion.
/// Call from background work; the cap bounds the returned pixel buffer, not
/// SVG parsing, intermediate allocations, atlas storage, or total heap use.
pub fn prepare_cpu_svg_preview(
    scene: &PreparedCpuSvgComposition,
    renderer: &SvgRenderer,
    render_scale: f32,
    max_preview_bytes: usize,
) -> Result<Arc<RenderImage>, CpuSvgPreviewError> {
    let (width, height, bytes) = preview_layout(scene, render_scale)?;
    if bytes > max_preview_bytes {
        return Err(CpuSvgPreviewError::PreviewBudgetExceeded {
            required: bytes,
            maximum: max_preview_bytes,
        });
    }
    let image = catch_unwind(AssertUnwindSafe(|| {
        let parsed = renderer
            .parse_svg(scene.svg().svg.as_bytes())
            .map_err(|error| CpuSvgPreviewError::SourceConversion(error.to_string()))?;
        renderer
            .render_parsed(&parsed, render_scale)
            .map_err(|error| CpuSvgPreviewError::SourceConversion(error.to_string()))
    }))
    .map_err(|_| CpuSvgPreviewError::SourceConversionPanicked)??;
    if image.frame_count() != 1
        || image.size(0).width.0
            != i32::try_from(width).map_err(|_| CpuSvgPreviewError::SizeOverflow)?
        || image.size(0).height.0
            != i32::try_from(height).map_err(|_| CpuSvgPreviewError::SizeOverflow)?
        || image.as_bytes(0).map(<[u8]>::len) != Some(bytes)
    {
        return Err(CpuSvgPreviewError::UnexpectedPixelLayout);
    }
    Ok(image)
}

/// Converts a prepared scene into its explicit physical viewport canvas.
/// The scene scale is a uniform render scale, without legacy supersampling.
/// The cap bounds the output buffer, not parsing or renderer intermediates.
pub fn prepare_cpu_svg_preview_at_target(
    scene: &PreparedCpuSvgComposition,
    renderer: &SvgRenderer,
    max_preview_bytes: usize,
) -> Result<Arc<RenderImage>, CpuSvgPreviewError> {
    let configuration = scene.configuration();
    prepare_svg_bytes_at_target(
        scene.svg().svg.as_bytes(),
        configuration.viewport.resolution.x,
        configuration.viewport.resolution.y,
        configuration.scale,
        renderer,
        max_preview_bytes,
    )
}

/// Converts a genuine accepted original output without certifying that its
/// document still owns it. Callers must recheck ownership before publication.
/// The cap bounds the frame, not parsing, renderer intermediates or atlas use.
pub fn prepare_accepted_cpu_svg_preview_at_target(
    accepted: &graphics_core::evaluate::AcceptedCpuSvgOutput,
    renderer: &SvgRenderer,
    max_preview_bytes: usize,
) -> Result<Arc<RenderImage>, CpuSvgPreviewError> {
    let graphite_graph_craft::document::value::RenderOutputType::Svg { svg, image_data } =
        &accepted.output().data
    else {
        return Err(CpuSvgPreviewError::SourceConversion(
            "The accepted output is not CPU SVG".into(),
        ));
    };
    if !image_data.is_empty() {
        return Err(CpuSvgPreviewError::SourceConversion(
            "The accepted SVG requires external image resources".into(),
        ));
    }
    let configuration = accepted.configuration();
    prepare_svg_bytes_at_target(
        svg.as_bytes(),
        configuration.viewport.resolution.x,
        configuration.viewport.resolution.y,
        configuration.scale,
        renderer,
        max_preview_bytes,
    )
}

fn prepare_svg_bytes_at_target(
    svg_bytes: &[u8],
    width: u32,
    height: u32,
    scale_factor: f64,
    renderer: &SvgRenderer,
    max_preview_bytes: usize,
) -> Result<Arc<RenderImage>, CpuSvgPreviewError> {
    let scale = scale_factor as f32;
    if !scale_factor.is_finite() || !scale.is_finite() || scale <= 0. {
        return Err(CpuSvgPreviewError::InvalidScale);
    }
    if width == 0 || height == 0 {
        return Err(CpuSvgPreviewError::InvalidDimensions);
    }
    if width > 8192 || height > 8192 {
        return Err(CpuSvgPreviewError::RendererSizeLimit);
    }
    let root_width = (f64::from(width) / scale_factor) as f32;
    let root_height = (f64::from(height) / scale_factor) as f32;
    if ![root_width, root_height]
        .into_iter()
        .all(|dimension| dimension.is_finite() && dimension > 0.)
    {
        return Err(CpuSvgPreviewError::InvalidDimensions);
    }
    if ![root_width, root_height].into_iter().all(|dimension| {
        let scaled = dimension * scale;
        scaled.is_finite() && scaled > 0.
    }) {
        return Err(CpuSvgPreviewError::InvalidScale);
    }
    let bytes = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(CpuSvgPreviewError::SizeOverflow)?;
    if bytes > max_preview_bytes {
        return Err(CpuSvgPreviewError::PreviewBudgetExceeded {
            required: bytes,
            maximum: max_preview_bytes,
        });
    }
    let width = i32::try_from(width).map_err(|_| CpuSvgPreviewError::SizeOverflow)?;
    let height = i32::try_from(height).map_err(|_| CpuSvgPreviewError::SizeOverflow)?;
    let image = catch_unwind(AssertUnwindSafe(|| {
        let parsed = renderer
            .parse_svg(svg_bytes)
            .map_err(|error| CpuSvgPreviewError::SourceConversion(error.to_string()))?;
        renderer
            .render_parsed_in_canvas(
                &parsed,
                gpui::size(gpui::DevicePixels(width), gpui::DevicePixels(height)),
                scale,
                max_preview_bytes,
            )
            .map_err(|error| CpuSvgPreviewError::SourceConversion(error.to_string()))
    }))
    .map_err(|_| CpuSvgPreviewError::SourceConversionPanicked)??;
    if image.frame_count() != 1
        || image.size(0).width.0 != width
        || image.size(0).height.0 != height
        || image.as_bytes(0).map(<[u8]>::len) != Some(bytes)
    {
        return Err(CpuSvgPreviewError::UnexpectedPixelLayout);
    }
    Ok(image)
}

pub(crate) fn prepare_cpu_svg_scene_png(
    scene: &PreparedCpuSvgComposition,
    image: &RenderImage,
    max_pixel_bytes: usize,
    max_encoded_bytes: usize,
) -> Result<graphics_core::export::PreparedRgba8Png, String> {
    let resolution = scene.configuration().viewport.resolution;
    prepare_native_frame_png(
        image,
        resolution.x,
        resolution.y,
        max_pixel_bytes,
        max_encoded_bytes,
    )
}

pub(crate) fn prepare_artboard_png(
    preview: &crate::session::PreparedArtboardPreview,
    max_pixel_bytes: usize,
    max_encoded_bytes: usize,
) -> Result<graphics_core::export::PreparedRgba8Png, String> {
    let resolution = preview.output().configuration().viewport.resolution;
    prepare_native_frame_png(
        preview.image(),
        resolution.x,
        resolution.y,
        max_pixel_bytes,
        max_encoded_bytes,
    )
}

fn prepare_native_frame_png(
    image: &RenderImage,
    width: u32,
    height: u32,
    max_pixel_bytes: usize,
    max_encoded_bytes: usize,
) -> Result<graphics_core::export::PreparedRgba8Png, String> {
    let required = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "PNG frame size overflowed".to_owned())?;
    if width == 0
        || height == 0
        || image.frame_count() != 1
        || image.size(0).width.0 != i32::try_from(width).map_err(|error| error.to_string())?
        || image.size(0).height.0 != i32::try_from(height).map_err(|error| error.to_string())?
    {
        return Err("PNG requires one frame at the declared physical target dimensions".into());
    }
    if required > max_pixel_bytes {
        return Err(format!(
            "PNG rows require {required} bytes, exceeding the {max_pixel_bytes} byte limit"
        ));
    }
    let bytes = image
        .as_bytes(0)
        .ok_or_else(|| "PNG frame pixels are unavailable".to_owned())?;
    if bytes.len() != required {
        return Err("PNG frame has an unexpected pixel layout".into());
    }
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(required)
        .map_err(|error| error.to_string())?;
    // GPUI already unpremultiplied this frame. Swap channels once without a second alpha or linear-color conversion.
    for pixel in bytes.as_chunks::<4>().0.iter() {
        pixels.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
    }
    graphics_core::export::prepare_rgba8_png(width, height, pixels, max_encoded_bytes)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::AppContext;
    use graphics_core::{
        render::{CpuSvgSceneConfiguration, CpuSvgSceneLayer, prepare_cpu_svg_composition},
        resources::{
            AlphaAssociation, CpuImageBudget, CpuImageDescriptor, CpuImageEdit, CpuImageResult,
            CpuSampleFormat, ImageColorSpace, ImageResourceIdentity, admit_cpu_image,
            evaluate_cpu_image_edit,
        },
    };
    use graphite_core_types::{
        Color,
        blending::BlendMode,
        color::SRGBA8,
        list::{
            ATTR_APPEARANCE, ATTR_BLEND_MODE, ATTR_CLIPPING_MASK, ATTR_DASH_PATTERN, ATTR_FONT,
            ATTR_FONT_SIZE, ATTR_GRADIENT_FORM, ATTR_GRADIENT_INTERPOLATION, ATTR_GRADIENT_SPACE,
            ATTR_GRADIENT_SPREAD, ATTR_MAX_HEIGHT, ATTR_OPACITY, ATTR_OPACITY_FILL, ATTR_PAINT,
            ATTR_TRANSFORM, Item, List,
        },
        transform::Footprint,
    };
    use graphite_glam::{DAffine2, UVec2};
    use graphite_graphic_types::{Appearance, Coverage, Graphic};
    use graphite_resource::Resource;
    use graphite_std::text_nodes::{
        FALLBACK_FONT_RESOURCE, TextContext, TypesettingConfig, shape_text_item,
    };
    use graphite_vector_types::{
        Gradient, GradientForm, GradientInterpolation, GradientSpace, GradientSpread, GradientStop,
        Vector,
        kurbo::{Affine, BezPath, Point, Shape},
        vector::style::{DashPattern, Stroke, StrokeAlign},
    };
    use sha2::{Digest, Sha256};

    fn configuration(width: u32, height: u32, scale: f64) -> CpuSvgSceneConfiguration {
        CpuSvgSceneConfiguration {
            viewport: Footprint {
                transform: DAffine2::IDENTITY,
                resolution: UVec2::new(width, height),
                quality: Footprint::DEFAULT.quality,
            },
            scale,
            max_svg_bytes: 32 * 1024,
        }
    }

    fn rectangle(x: f64, y: f64, width: f64, height: f64, color: [f32; 4]) -> Item<Vector> {
        let mut path = BezPath::new();
        path.move_to((x, y));
        path.line_to((x + width, y));
        path.line_to((x + width, y + height));
        path.line_to((x, y + height));
        path.close_path();
        Item::new_from_element(Vector::from_bezpath(path)).with_attribute(
            ATTR_APPEARANCE,
            Appearance::new_single(
                Coverage::new_fill(),
                Graphic::from(Color::from_gamma_srgb_channels(
                    color[0], color[1], color[2], color[3],
                )),
            ),
        )
    }

    fn scene(layers: &[CpuSvgSceneLayer], width: u32, height: u32) -> PreparedCpuSvgComposition {
        prepare_cpu_svg_composition(layers, configuration(width, height, 1.))
            .expect("original finite composition")
    }

    fn renderer() -> SvgRenderer {
        SvgRenderer::new(Arc::new(()))
    }

    fn pixel(image: &RenderImage, x: usize, y: usize) -> [u8; 4] {
        let width = usize::try_from(image.size(0).width.0).expect("positive frame width");
        let offset = (y * width + x) * 4;
        image
            .as_bytes(0)
            .expect("native frame")
            .get(offset..offset + 4)
            .expect("in-bounds sample")
            .try_into()
            .expect("BGRA sample")
    }

    fn assert_pixel_close(actual: [u8; 4], expected: [u8; 4], tolerance: u8) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                actual.abs_diff(expected) <= tolerance,
                "{actual} differs from {expected}"
            );
        }
    }

    fn image_result() -> CpuImageResult {
        // Four constant pixels per region contain the native bicubic footprint;
        // the original two-pixel row necessarily mixed color and alpha at these samples.
        let bytes: Arc<[u8]> = Arc::from([
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 128, 0, 0,
            255, 128, 0, 0, 255, 128, 0, 0, 255, 128,
        ]);
        let image = admit_cpu_image(
            ImageResourceIdentity {
                asset_id: "scene-test-input".into(),
                version_id: "explicit-process-local-test-version".into(),
                content_sha256: Sha256::digest(&bytes).into(),
            },
            CpuImageDescriptor {
                width: 8,
                height: 1,
                row_stride: 32,
                sample_format: CpuSampleFormat::Rgba8,
                color_space: ImageColorSpace::Srgb,
                alpha: AlphaAssociation::Straight,
                transform: [16., 0., 0., 8., 0., 0.],
            },
            bytes,
            CpuImageBudget {
                max_input_bytes: 32,
                max_decoded_bytes: 8 * (std::mem::size_of::<Color>() + 4),
            },
        )
        .expect("actual admitted original image");
        evaluate_cpu_image_edit(&image, CpuImageEdit::Invert, 256).expect("original image edit")
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_known_asymmetric_colors_and_straight_alpha() {
        let red = rectangle(0., 0., 8., 8., [1., 0., 0., 1.]);
        let blue = rectangle(8., 0., 8., 8., [0., 0., 1., 0.5]);
        let red_before = red.clone();
        let blue_before = blue.clone();
        let prepared = scene(
            &[
                CpuSvgSceneLayer::Vector(red.clone()),
                CpuSvgSceneLayer::Vector(blue.clone()),
            ],
            16,
            12,
        );
        let svg_before = prepared.svg().svg.clone();
        let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 32 * 24 * 4)
            .expect("actual color scene conversion");
        assert_eq!(pixel(&converted, 8, 8), [0, 0, 255, 255]);
        assert_pixel_close(pixel(&converted, 24, 8), [255, 0, 0, 128], 1);
        assert_eq!(pixel(&converted, 8, 20), [0, 0, 0, 0]);
        assert_eq!(prepared.svg().svg, svg_before);
        match prepared.layers() {
            [
                CpuSvgSceneLayer::Vector(actual_red),
                CpuSvgSceneLayer::Vector(actual_blue),
            ] => {
                assert_eq!(actual_red, &red_before);
                assert_eq!(actual_blue, &blue_before);
            }
            _ => panic!("the prepared original layers changed"),
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_embedded_original_png_preserves_source_and_provenance() {
        let source = image_result();
        let source_before = source.pixels().image().clone();
        let attributes_before = source.pixels().attributes().clone();
        let metadata_before = (
            source.input_identity().clone(),
            source.input_origin(),
            source.operation(),
            *source.descriptor(),
            source.descriptor().transform.map(f64::to_bits),
            source.pixels_sha256(),
        );
        let prepared = scene(&[CpuSvgSceneLayer::Image(source.clone())], 16, 8);
        let before = prepared.svg().svg.clone();
        assert!(before.contains("data:image/png;base64,"));
        let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 32 * 16 * 4)
            .expect("original embedded PNG scene conversion");
        assert_pixel_close(pixel(&converted, 7, 7), [255, 255, 0, 255], 1);
        assert_pixel_close(pixel(&converted, 25, 7), [0, 255, 255, 128], 2);
        let repeated = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 32 * 16 * 4)
            .expect("repeated conversion");
        assert_eq!(converted.as_bytes(0), repeated.as_bytes(0));
        assert_eq!(prepared.svg().svg, before);
        assert_eq!(source.pixels().image(), &source_before);
        assert_eq!(source.pixels().attributes(), &attributes_before);
        assert_eq!(
            (
                source.input_identity().clone(),
                source.input_origin(),
                source.operation(),
                *source.descriptor(),
                source.descriptor().transform.map(f64::to_bits),
                source.pixels_sha256()
            ),
            metadata_before
        );
        match prepared.layers() {
            [CpuSvgSceneLayer::Image(retained)] => {
                assert_eq!(retained.pixels().image(), &source_before);
                assert_eq!(retained.pixels().attributes(), &attributes_before);
                assert_eq!(
                    (
                        retained.input_identity().clone(),
                        retained.input_origin(),
                        retained.operation(),
                        *retained.descriptor(),
                        retained.descriptor().transform.map(f64::to_bits),
                        retained.pixels_sha256()
                    ),
                    metadata_before
                );
                assert_eq!(retained.input_identity(), source.input_identity());
                assert_eq!(retained.input_origin(), source.input_origin());
                assert_eq!(retained.operation(), source.operation());
                assert_eq!(retained.pixels_sha256(), source.pixels_sha256());
            }
            _ => panic!("the original image result was not retained"),
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_mixed_original_scene_and_blend_renderer_differential() {
        let source = image_result();
        let before_image = source.pixels().image().clone();
        let before_attributes = source.pixels().attributes().clone();
        let before_metadata = (
            source.input_identity().clone(),
            source.input_origin(),
            source.operation(),
            *source.descriptor(),
            source.descriptor().transform.map(f64::to_bits),
            source.pixels_sha256(),
        );
        let over = rectangle(0., 0., 8., 8., [1., 0., 0., 1.])
            .with_attribute(ATTR_BLEND_MODE, BlendMode::Multiply);
        let before_over = over.clone();
        let prepared = scene(
            &[
                CpuSvgSceneLayer::Image(source.clone()),
                CpuSvgSceneLayer::Vector(over),
            ],
            16,
            8,
        );
        assert!(prepared.svg().svg.contains("mix-blend-mode: multiply;"));
        let renderer = renderer();
        let actual = prepare_cpu_svg_preview(&prepared, &renderer, 1., 32 * 16 * 4)
            .expect("actual mixed scene preview");
        let original = renderer
            .render_single_frame(prepared.svg().svg.as_bytes(), 1.)
            .expect("existing renderer binding oracle");
        assert_eq!(actual.as_bytes(0), original.as_bytes(0));
        assert_eq!(pixel(&actual, 7, 7), [0, 0, 0, 255]);
        assert_pixel_close(pixel(&actual, 25, 7), [0, 255, 255, 128], 2);
        assert_eq!(source.pixels().image(), &before_image);
        assert_eq!(source.pixels().attributes(), &before_attributes);
        assert_eq!(
            (
                source.input_identity().clone(),
                source.input_origin(),
                source.operation(),
                *source.descriptor(),
                source.descriptor().transform.map(f64::to_bits),
                source.pixels_sha256()
            ),
            before_metadata
        );
        match prepared.layers() {
            [
                CpuSvgSceneLayer::Image(image),
                CpuSvgSceneLayer::Vector(vector),
            ] => {
                assert_eq!(image.pixels().image(), &before_image);
                assert_eq!(image.pixels().attributes(), &before_attributes);
                assert_eq!(
                    (
                        image.input_identity().clone(),
                        image.input_origin(),
                        image.operation(),
                        *image.descriptor(),
                        image.descriptor().transform.map(f64::to_bits),
                        image.pixels_sha256()
                    ),
                    before_metadata
                );
                assert_eq!(image.input_identity(), source.input_identity());
                assert_eq!(image.pixels_sha256(), source.pixels_sha256());
                assert_eq!(
                    image.descriptor().transform.map(f64::to_bits),
                    source.descriptor().transform.map(f64::to_bits)
                );
                assert_eq!(vector, &before_over);
            }
            _ => panic!("mixed scene layer ownership changed"),
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_original_clip_path_and_alpha_mask_pixels() {
        for (alpha, kind, expected) in [
            (1., "<clipPath", [255, 0, 0, 255]),
            (0.5, "<mask", [255, 85, 85, 192]),
        ] {
            let masker = rectangle(0., 0., 8., 8., [1., 1., 1., alpha]);
            let clipped = rectangle(0., 0., 16., 8., [0., 0., 1., 1.])
                .with_attribute(ATTR_CLIPPING_MASK, true);
            let before_masker = masker.clone();
            let before_clipped = clipped.clone();
            let prepared = scene(
                &[
                    CpuSvgSceneLayer::Vector(masker),
                    CpuSvgSceneLayer::Vector(clipped),
                ],
                16,
                8,
            );
            assert!(prepared.svg().svg.contains(kind));
            let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 32 * 16 * 4)
                .expect("original sibling clip conversion");
            assert_pixel_close(pixel(&converted, 8, 8), expected, 2);
            assert_eq!(pixel(&converted, 24, 8), [0, 0, 0, 0]);
            match prepared.layers() {
                [
                    CpuSvgSceneLayer::Vector(masker),
                    CpuSvgSceneLayer::Vector(clipped),
                ] => {
                    assert_eq!(masker, &before_masker);
                    assert_eq!(clipped, &before_clipped);
                }
                _ => panic!("the original clipping layers changed"),
            }
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_original_solid_stroke_has_visible_pixels() {
        let mut vector = rectangle(4., 4., 8., 8., [1., 0., 0., 1.]);
        let mut appearance = Appearance::new_single(
            Coverage::new_fill(),
            Graphic::from(Color::from_gamma_srgb_channels(1., 0., 0., 1.)),
        );
        appearance.0.push(
            Item::new_from_element(Coverage::new_stroke(&Stroke::new(2.))).with_attribute(
                ATTR_PAINT,
                Graphic::from(Color::from_gamma_srgb_channels(0., 0., 1., 1.)),
            ),
        );
        vector.set_attribute(ATTR_APPEARANCE, appearance);
        let before = vector.clone();
        let prepared = scene(&[CpuSvgSceneLayer::Vector(vector)], 16, 16);
        assert!(prepared.svg().svg.contains("stroke-width=\"2\""));
        let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 32 * 32 * 4)
            .expect("original stroke conversion");
        assert_eq!(pixel(&converted, 16, 16), [0, 0, 255, 255]);
        assert_eq!(pixel(&converted, 7, 16), [255, 0, 0, 255]);
        assert_eq!(pixel(&converted, 2, 16), [0, 0, 0, 0]);
        match prepared.layers() {
            [CpuSvgSceneLayer::Vector(actual)] => assert_eq!(actual, &before),
            _ => panic!("the stroked original vector changed"),
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_fractional_root_rounding_scale_and_exact_cap_recovery() {
        let prepared = prepare_cpu_svg_composition(&[], configuration(7, 5, 3.))
            .expect("original fractional root lengths");
        let before = prepared.svg().svg.clone();
        assert_eq!(preview_layout(&prepared, 1.), Ok((4, 3, 48)));
        let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 48)
            .expect("exact truncated pixel buffer cap");
        assert_eq!(
            (converted.size(0).width.0, converted.size(0).height.0),
            (4, 3)
        );
        assert_eq!(converted.as_bytes(0).expect("pixels").len(), 48);
        for maximum in [0, 47] {
            assert_eq!(
                prepare_cpu_svg_preview(&prepared, &renderer(), 1., maximum).err(),
                Some(CpuSvgPreviewError::PreviewBudgetExceeded {
                    required: 48,
                    maximum
                })
            );
            assert_eq!(prepared.svg().svg, before);
        }
        let doubled = prepare_cpu_svg_preview(&prepared, &renderer(), 2., 9 * 6 * 4)
            .expect("explicit doubled render scale");
        assert_eq!((doubled.size(0).width.0, doubled.size(0).height.0), (9, 6));
        assert!(prepare_cpu_svg_preview(&prepared, &renderer(), 1., 48).is_ok());
        let rounded = prepare_cpu_svg_composition(&[], configuration(3, 3, 1.99999995))
            .expect("root length rounds to f32");
        assert_eq!(preview_layout(&rounded, 1.), Ok((3, 3, 36)));
        let image =
            prepare_cpu_svg_preview(&rounded, &renderer(), 1., 36).expect("native root rounding");
        assert_eq!((image.size(0).width.0, image.size(0).height.0), (3, 3));
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_rejects_silent_native_resizing_and_unrepresentable_dimensions() {
        let ordinary = scene(&[], 16, 8);
        for scale in [0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                prepare_cpu_svg_preview(&ordinary, &renderer(), scale, usize::MAX).err(),
                Some(CpuSvgPreviewError::InvalidScale)
            );
        }
        for scale in [f32::from_bits(1), f32::MAX] {
            assert_eq!(
                prepare_cpu_svg_preview(&ordinary, &renderer(), scale, usize::MAX).err(),
                Some(CpuSvgPreviewError::InvalidDimensions)
            );
        }
        let maximum = scene(&[], 4096, 1);
        assert_eq!(preview_layout(&maximum, 1.), Ok((8192, 2, 8192 * 2 * 4)));
        assert_eq!(
            prepare_cpu_svg_preview(&maximum, &renderer(), 1., 0).err(),
            Some(CpuSvgPreviewError::PreviewBudgetExceeded {
                required: 8192 * 2 * 4,
                maximum: 0
            })
        );
        for (width, height) in [(4097, 1), (1, 4097)] {
            let oversized = scene(&[], width, height);
            assert_eq!(
                prepare_cpu_svg_preview(&oversized, &renderer(), 1., usize::MAX).err(),
                Some(CpuSvgPreviewError::RendererSizeLimit)
            );
        }
        for source_scale in [1e-40, 1e40] {
            let prepared = prepare_cpu_svg_composition(&[], configuration(1, 1, source_scale))
                .expect("finite source root size");
            assert_eq!(
                prepare_cpu_svg_preview(&prepared, &renderer(), 1., usize::MAX).err(),
                Some(CpuSvgPreviewError::InvalidDimensions)
            );
        }
        assert!(prepare_cpu_svg_preview(&ordinary, &renderer(), 1., 32 * 16 * 4).is_ok());
    }

    fn gradient_paint(
        form: GradientForm,
        transform: DAffine2,
        stops: &[(f64, [f32; 4])],
    ) -> Item<Gradient> {
        Item::new_from_element(Gradient::new(stops.iter().map(|(position, channels)| {
            GradientStop {
                position: *position,
                midpoint: 0.5,
                color: Color::from_gamma_srgb_channels(
                    channels[0],
                    channels[1],
                    channels[2],
                    channels[3],
                ),
            }
        })))
        .with_attribute(ATTR_GRADIENT_FORM, form)
        .with_attribute(ATTR_GRADIENT_SPACE, GradientSpace::RgbGamma)
        .with_attribute(ATTR_GRADIENT_INTERPOLATION, GradientInterpolation::Linear)
        .with_attribute(ATTR_GRADIENT_SPREAD, GradientSpread::Pad)
        .with_attribute(ATTR_TRANSFORM, transform)
    }

    fn gradient_rectangle(y: f64, height: f64, paint: &Item<Gradient>) -> Item<Vector> {
        rectangle(0., y, 16., height, [0., 0., 0., 1.]).with_attribute(
            ATTR_APPEARANCE,
            Appearance::new_single(Coverage::new_fill(), Graphic::from(paint.clone())),
        )
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_linear_and_radial_gradient_plateaus() {
        let stops = [
            (0., [1., 0., 0., 1.]),
            (0.35, [1., 0., 0., 1.]),
            (0.65, [0., 0., 1., 1.]),
            (1., [0., 0., 1., 1.]),
        ];
        for (form, transform, samples, definition) in [
            (
                GradientForm::Linear,
                DAffine2::from_cols_array(&[16., 0., 0., 16., 0., 0.]),
                [(8, 16), (24, 16)],
                "<linearGradient",
            ),
            (
                GradientForm::Radial,
                DAffine2::from_cols_array(&[8., 0., 0., 8., 8., 8.]),
                [(16, 16), (29, 16)],
                "<radialGradient",
            ),
        ] {
            let paint = gradient_paint(form, transform, &stops);
            let paint_before = paint.clone();
            let vector = gradient_rectangle(0., 16., &paint);
            let vector_before = vector.clone();
            let prepared = prepare_cpu_svg_composition(
                &[CpuSvgSceneLayer::Vector(vector.clone())],
                configuration(32, 32, 2.),
            )
            .expect("original typed gradient composition");
            let svg_before = prepared.svg().svg.clone();
            assert!(svg_before.contains(definition));
            assert!(svg_before.contains("gradientUnits=\"userSpaceOnUse\""));
            assert!(svg_before.contains("stop-color=\"#ff0000\""));
            assert!(svg_before.contains("stop-color=\"#0000ff\""));
            let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 32 * 32 * 4)
                .expect("native full-color gradient");
            assert_eq!(
                (converted.size(0).width.0, converted.size(0).height.0),
                (32, 32)
            );
            // The source unit line/circle is explicitly transformed above. These
            // pixel centers lie inside the constant stops, away from geometry edges.
            assert_eq!(
                pixel(&converted, samples[0].0, samples[0].1),
                [0, 0, 255, 255]
            );
            assert_eq!(
                pixel(&converted, samples[1].0, samples[1].1),
                [255, 0, 0, 255]
            );
            let repeated = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 32 * 32 * 4)
                .expect("same prepared gradient conversion");
            assert_eq!(
                converted.as_bytes(0).expect("gradient frame"),
                repeated.as_bytes(0).expect("repeated gradient frame")
            );
            assert_eq!(prepared.svg().svg, svg_before);
            assert_eq!(paint, paint_before);
            assert_eq!(vector, vector_before);
            assert_eq!(
                prepared.configuration().viewport.resolution,
                UVec2::new(32, 32)
            );
            assert_eq!(
                prepared.configuration().viewport.transform,
                DAffine2::IDENTITY
            );
            assert_eq!(prepared.configuration().scale.to_bits(), 2_f64.to_bits());
            match prepared.layers() {
                [CpuSvgSceneLayer::Vector(retained)] => assert_eq!(retained, &vector_before),
                _ => panic!("the typed gradient layer changed"),
            }
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_gradient_alpha_retains_original_image_provenance() {
        let source = image_result();
        let image_before = source.pixels().image().clone();
        let attributes_before = source.pixels().attributes().clone();
        let metadata_before = (
            source.input_identity().clone(),
            source.input_origin(),
            source.operation(),
            *source.descriptor(),
            source.descriptor().transform.map(f64::to_bits),
            source.pixels_sha256(),
        );
        let paint = gradient_paint(
            GradientForm::Linear,
            DAffine2::from_cols_array(&[16., 0., 0., 8., 0., 8.]),
            &[
                (0., [1., 0., 0., 0.]),
                (0.2, [1., 0., 0., 0.]),
                (0.4, [0., 1., 0., 0.25]),
                (0.6, [0., 1., 0., 0.25]),
                (0.8, [0., 0., 1., 0.5]),
                (1., [0., 0., 1., 0.5]),
            ],
        );
        let paint_before = paint.clone();
        let vector = gradient_rectangle(8., 8., &paint);
        let vector_before = vector.clone();
        let prepared = scene(
            &[
                CpuSvgSceneLayer::Image(source.clone()),
                CpuSvgSceneLayer::Vector(vector.clone()),
            ],
            16,
            16,
        );
        let svg_before = prepared.svg().svg.clone();
        assert!(svg_before.contains("<linearGradient"));
        assert!(svg_before.contains("stop-opacity=\"0.25\""));
        assert!(svg_before.contains("stop-opacity=\"0.5\""));
        assert!(svg_before.contains("data:image/png;base64,"));
        let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 32 * 32 * 4)
            .expect("original image and alpha gradient");
        assert_eq!(
            (converted.size(0).width.0, converted.size(0).height.0),
            (32, 32)
        );
        assert_eq!(pixel(&converted, 2, 24), [0, 0, 0, 0]);
        assert_pixel_close(pixel(&converted, 16, 24), [0, 255, 0, 64], 1);
        assert_pixel_close(pixel(&converted, 28, 24), [255, 0, 0, 128], 1);
        assert_pixel_close(pixel(&converted, 7, 7), [255, 255, 0, 255], 1);
        assert_pixel_close(pixel(&converted, 25, 7), [0, 255, 255, 128], 2);
        for retained in [
            &source,
            match prepared.layers() {
                [
                    CpuSvgSceneLayer::Image(retained),
                    CpuSvgSceneLayer::Vector(retained_vector),
                ] => {
                    assert_eq!(retained_vector, &vector_before);
                    retained
                }
                _ => panic!("the ordered image and gradient source layers changed"),
            },
        ] {
            assert_eq!(retained.pixels().image(), &image_before);
            assert_eq!(retained.pixels().attributes(), &attributes_before);
            assert_eq!(
                (
                    retained.input_identity().clone(),
                    retained.input_origin(),
                    retained.operation(),
                    *retained.descriptor(),
                    retained.descriptor().transform.map(f64::to_bits),
                    retained.pixels_sha256(),
                ),
                metadata_before
            );
        }
        assert_eq!(paint, paint_before);
        assert_eq!(vector, vector_before);
        assert_eq!(prepared.svg().svg, svg_before);
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_gradient_exact_cap_scale_and_recovery() {
        let paint = gradient_paint(
            GradientForm::Radial,
            DAffine2::from_cols_array(&[8., 0., 0., 8., 8., 8.]),
            &[
                (0., [1., 0., 0., 1.]),
                (0.35, [1., 0., 0., 1.]),
                (0.65, [0., 0., 1., 0.5]),
                (1., [0., 0., 1., 0.5]),
            ],
        );
        let paint_before = paint.clone();
        let vector = gradient_rectangle(0., 16., &paint);
        let vector_before = vector.clone();
        let prepared = scene(&[CpuSvgSceneLayer::Vector(vector.clone())], 16, 16);
        let svg_before = prepared.svg().svg.clone();
        for (scale, dimensions, samples) in [
            (0.5, (16, 16), [(8, 8), (14, 8)]),
            (1., (32, 32), [(16, 16), (29, 16)]),
        ] {
            let required = dimensions.0 as usize * dimensions.1 as usize * 4;
            let at_cap = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                .expect("exact native gradient buffer cap");
            assert_eq!(
                (at_cap.size(0).width.0, at_cap.size(0).height.0),
                dimensions
            );
            assert_eq!(at_cap.as_bytes(0).expect("gradient frame").len(), required);
            assert_eq!(pixel(&at_cap, samples[0].0, samples[0].1), [0, 0, 255, 255]);
            assert_pixel_close(
                pixel(&at_cap, samples[1].0, samples[1].1),
                [255, 0, 0, 128],
                1,
            );
            for maximum in [0, required - 1] {
                assert_eq!(
                    prepare_cpu_svg_preview(&prepared, &renderer(), scale, maximum).err(),
                    Some(CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum })
                );
            }
            let recovered = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                .expect("gradient recovery after rejected cap");
            assert_eq!(
                recovered.as_bytes(0).expect("recovered gradient frame"),
                at_cap.as_bytes(0).expect("exact cap frame")
            );
        }
        assert_eq!(prepared.svg().svg, svg_before);
        assert_eq!(paint, paint_before);
        assert_eq!(vector, vector_before);
        match prepared.layers() {
            [CpuSvgSceneLayer::Vector(retained)] => assert_eq!(retained, &vector_before),
            _ => panic!("the original gradient layer changed after cap rejection"),
        }
    }

    fn dashed_line(lengths: &[f64], alpha: f32) -> Item<Vector> {
        let mut path = BezPath::new();
        path.move_to((4., 8.));
        path.line_to((28., 8.));
        let stroke = Stroke {
            dash_lengths: lengths.to_vec(),
            ..Stroke::new(2.)
        };
        let mut appearance =
            Appearance::new_single(Coverage::new_fill(), Graphic::from(Color::TRANSPARENT));
        appearance.0.push(
            Item::new_from_element(Coverage::new_stroke(&stroke)).with_attribute(
                ATTR_PAINT,
                Graphic::from(Color::from_gamma_srgb_channels(0., 0., 1., alpha)),
            ),
        );
        Item::new_from_element(Vector::from_bezpath(path))
            .with_attribute(ATTR_APPEARANCE, appearance)
    }

    fn authored_dash_bits(vector: &Item<Vector>) -> Vec<u64> {
        let appearance = vector
            .attribute::<Appearance>(ATTR_APPEARANCE)
            .expect("declared fill and stroke appearance");
        appearance
            .cover_at(1)
            .expect("original stroke row")
            .0
            .attribute::<DashPattern>(ATTR_DASH_PATTERN)
            .expect("authored dash pattern")
            .0
            .iter_element_values()
            .map(|length| length.to_bits())
            .collect()
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_original_dashed_stroke_phase_alpha_and_zero_patterns() {
        for (lengths, expected_stroke) in [
            (vec![4., -2.], [true, true, true, true, true]),
            (vec![4., 2., 2.], [true, false, true, false, true]),
            (vec![0., 0.], [true, true, true, true, true]),
        ] {
            for alpha in [1., 0.5] {
                let vector = dashed_line(&lengths, alpha);
                let before = vector.clone();
                let bits_before = lengths
                    .iter()
                    .map(|length| length.to_bits())
                    .collect::<Vec<_>>();
                assert_eq!(authored_dash_bits(&vector), bits_before);
                let prepared = scene(&[CpuSvgSceneLayer::Vector(vector.clone())], 32, 16);
                let svg_before = prepared.svg().svg.clone();
                let original_lengths = lengths
                    .iter()
                    .map(|length| length.max(0.).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                assert!(svg_before.contains(&format!("stroke-dasharray=\"{original_lengths}\"")));
                assert!(svg_before.contains("stroke-width=\"2\""));
                let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 64 * 32 * 4)
                    .expect("actual original moderate dash preview");
                assert_eq!(
                    (converted.size(0).width.0, converted.size(0).height.0),
                    (64, 32)
                );
                // The straight path begins at x=4. Odd [4,2,2] repeats with swapped
                // roles; the samples are interior to successive dash/gap intervals.
                for (x, painted) in [14, 18, 22, 28, 34].into_iter().zip(expected_stroke) {
                    if painted {
                        assert_pixel_close(
                            pixel(&converted, x, 16),
                            [255, 0, 0, if alpha == 1. { 255 } else { 128 }],
                            1,
                        );
                    } else {
                        assert_eq!(pixel(&converted, x, 16), [0, 0, 0, 0]);
                    }
                }
                assert_eq!(pixel(&converted, 60, 16), [0, 0, 0, 0]);
                assert_eq!(pixel(&converted, 14, 24), [0, 0, 0, 0]);
                assert_eq!(vector, before);
                assert_eq!(authored_dash_bits(&vector), bits_before);
                assert_eq!(prepared.svg().svg, svg_before);
                match prepared.layers() {
                    [CpuSvgSceneLayer::Vector(retained)] => {
                        assert_eq!(retained, &before);
                        assert_eq!(authored_dash_bits(retained), bits_before);
                    }
                    _ => panic!("the authored dash layer changed"),
                }
            }
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_dashed_stroke_exact_cap_and_recovery_preserve_source() {
        let vector = dashed_line(&[4., 2., 2.], 0.5);
        let before = vector.clone();
        let bits_before = authored_dash_bits(&vector);
        let prepared = scene(&[CpuSvgSceneLayer::Vector(vector.clone())], 32, 16);
        let svg_before = prepared.svg().svg.clone();
        for (scale, dimensions, samples) in [
            (0.5, (32, 16), [(6, 8), (9, 8)]),
            (1., (64, 32), [(12, 16), (18, 16)]),
        ] {
            let required = dimensions.0 as usize * dimensions.1 as usize * 4;
            let converted = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                .expect("exact moderate dashed frame cap");
            assert_eq!(
                (converted.size(0).width.0, converted.size(0).height.0),
                dimensions
            );
            assert_eq!(converted.as_bytes(0).expect("dash frame").len(), required);
            assert_pixel_close(
                pixel(&converted, samples[0].0, samples[0].1),
                [255, 0, 0, 128],
                1,
            );
            assert_eq!(pixel(&converted, samples[1].0, samples[1].1), [0, 0, 0, 0]);
            for maximum in [0, required - 1] {
                assert_eq!(
                    prepare_cpu_svg_preview(&prepared, &renderer(), scale, maximum).err(),
                    Some(CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum })
                );
            }
            let recovered = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                .expect("moderate dash recovery after cap failure");
            assert_eq!(
                recovered.as_bytes(0).expect("recovered dash frame"),
                converted.as_bytes(0).expect("exact cap dash frame")
            );
        }
        assert_eq!(vector, before);
        assert_eq!(authored_dash_bits(&vector), bits_before);
        assert_eq!(prepared.svg().svg, svg_before);
        match prepared.layers() {
            [CpuSvgSceneLayer::Vector(retained)] => {
                assert_eq!(retained, &before);
                assert_eq!(authored_dash_bits(retained), bits_before);
            }
            _ => panic!("the dash layer changed after recovery"),
        }
    }

    fn aligned_rectangle(
        alignment: StrokeAlign,
        fill_alpha: f32,
        stroke_alpha: f32,
    ) -> Item<Vector> {
        let mut vector = rectangle(8., 8., 16., 16., [1., 0., 0., fill_alpha]);
        let mut appearance = Appearance::new_single(
            Coverage::new_fill(),
            Graphic::from(Color::from_gamma_srgb_channels(1., 0., 0., fill_alpha)),
        );
        appearance.0.push(
            Item::new_from_element(Coverage::new_stroke(&Stroke {
                align: alignment,
                ..Stroke::new(4.)
            }))
            .with_attribute(
                ATTR_PAINT,
                Graphic::from(Color::from_gamma_srgb_channels(0., 0., 1., stroke_alpha)),
            ),
        );
        vector.set_attribute(ATTR_APPEARANCE, appearance);
        vector
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_original_inside_clip_and_opaque_outside_stroke() {
        for alignment in [StrokeAlign::Inside, StrokeAlign::Outside] {
            let vector = aligned_rectangle(alignment, 1., 1.);
            let before = vector.clone();
            let prepared = scene(&[CpuSvgSceneLayer::Vector(vector.clone())], 32, 32);
            let svg_before = prepared.svg().svg.clone();
            assert!(svg_before.contains("stroke-width=\"8\""));
            assert_eq!(
                svg_before.contains("<clipPath"),
                alignment == StrokeAlign::Inside
            );
            assert!(!svg_before.contains("<mask"));
            assert_eq!(
                svg_before.contains("paint-order: stroke;"),
                alignment == StrokeAlign::Outside
            );
            let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 64 * 64 * 4)
                .expect("original aligned opaque stroke conversion");
            assert_eq!(
                (converted.size(0).width.0, converted.size(0).height.0),
                (64, 64)
            );
            // A doubled width-eight stroke is clipped to the rectangle for Inside,
            // or covered by its opaque fill for Outside. These samples avoid both boundaries.
            let (outside, inside) = match alignment {
                StrokeAlign::Inside => ([0, 0, 0, 0], [255, 0, 0, 255]),
                StrokeAlign::Outside => ([255, 0, 0, 255], [0, 0, 255, 255]),
                StrokeAlign::Center => panic!("the fixture explicitly selects alignment"),
            };
            assert_eq!(pixel(&converted, 12, 32), outside);
            assert_eq!(pixel(&converted, 20, 32), inside);
            assert_eq!(pixel(&converted, 32, 32), [0, 0, 255, 255]);
            assert_eq!(pixel(&converted, 4, 32), [0, 0, 0, 0]);
            assert_eq!(vector, before);
            assert_eq!(prepared.svg().svg, svg_before);
            match prepared.layers() {
                [CpuSvgSceneLayer::Vector(retained)] => assert_eq!(retained, &before),
                _ => panic!("the original aligned layer changed"),
            }
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_original_alignment_alpha_mask_and_inside_compositing() {
        for alignment in [StrokeAlign::Inside, StrokeAlign::Outside] {
            let vector = aligned_rectangle(alignment, 0.5, 0.5);
            let before = vector.clone();
            let prepared = scene(&[CpuSvgSceneLayer::Vector(vector.clone())], 32, 32);
            let svg_before = prepared.svg().svg.clone();
            assert!(svg_before.contains("id=\"alignment-"));
            assert_eq!(
                svg_before.contains("<clipPath"),
                alignment == StrokeAlign::Inside
            );
            assert_eq!(
                svg_before.contains("<mask"),
                alignment == StrokeAlign::Outside
            );
            assert!(svg_before.contains("stroke-width=\"8\""));
            let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 64 * 64 * 4)
                .expect("original aligned alpha compositing");
            if alignment == StrokeAlign::Inside {
                assert_eq!(pixel(&converted, 12, 32), [0, 0, 0, 0]);
                // Half-alpha blue over half-alpha red gives three-quarter alpha;
                // straight channels are one-third red and two-thirds blue.
                assert_pixel_close(pixel(&converted, 20, 32), [170, 0, 85, 192], 2);
            } else {
                assert_pixel_close(pixel(&converted, 12, 32), [255, 0, 0, 128], 1);
                assert_pixel_close(pixel(&converted, 20, 32), [0, 0, 255, 128], 1);
            }
            assert_pixel_close(pixel(&converted, 32, 32), [0, 0, 255, 128], 1);
            assert_eq!(pixel(&converted, 4, 32), [0, 0, 0, 0]);
            assert_eq!(vector, before);
            assert_eq!(prepared.svg().svg, svg_before);
            match prepared.layers() {
                [CpuSvgSceneLayer::Vector(retained)] => assert_eq!(retained, &before),
                _ => panic!("the original alpha-aligned layer changed"),
            }
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_aligned_mask_exact_cap_and_recovery_preserve_source() {
        let vector = aligned_rectangle(StrokeAlign::Outside, 0.5, 0.5);
        let before = vector.clone();
        let prepared = scene(&[CpuSvgSceneLayer::Vector(vector.clone())], 32, 32);
        let svg_before = prepared.svg().svg.clone();
        assert!(svg_before.contains("<mask"));
        for (scale, dimensions, samples) in [
            (0.5, (32, 32), [(6, 16), (10, 16)]),
            (1., (64, 64), [(12, 32), (20, 32)]),
        ] {
            let required = dimensions.0 as usize * dimensions.1 as usize * 4;
            let converted = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                .expect("exact aligned preview cap");
            assert_eq!(
                (converted.size(0).width.0, converted.size(0).height.0),
                dimensions
            );
            assert_eq!(
                converted.as_bytes(0).expect("aligned frame").len(),
                required
            );
            assert_pixel_close(
                pixel(&converted, samples[0].0, samples[0].1),
                [255, 0, 0, 128],
                1,
            );
            assert_pixel_close(
                pixel(&converted, samples[1].0, samples[1].1),
                [0, 0, 255, 128],
                1,
            );
            for maximum in [0, required - 1] {
                assert_eq!(
                    prepare_cpu_svg_preview(&prepared, &renderer(), scale, maximum).err(),
                    Some(CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum })
                );
            }
            let recovered = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                .expect("alignment recovery after cap rejection");
            assert_eq!(
                recovered.as_bytes(0).expect("recovered aligned frame"),
                converted.as_bytes(0).expect("original aligned frame")
            );
        }
        assert_eq!(vector, before);
        assert_eq!(prepared.svg().svg, svg_before);
        match prepared.layers() {
            [CpuSvgSceneLayer::Vector(retained)] => assert_eq!(retained, &before),
            _ => panic!("the aligned layer changed after recovery"),
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_open_and_mixed_contours_keep_centered_alignment_fallback() {
        for alignment in [StrokeAlign::Inside, StrokeAlign::Outside] {
            for include_closed_contour in [false, true] {
                let mut path = BezPath::new();
                path.move_to((4., 8.));
                path.line_to((20., 8.));
                if include_closed_contour {
                    path.move_to((4., 16.));
                    path.line_to((20., 16.));
                    path.line_to((20., 24.));
                    path.line_to((4., 24.));
                    path.close_path();
                }
                let mut appearance =
                    Appearance::new_single(Coverage::new_fill(), Graphic::from(Color::TRANSPARENT));
                appearance.0.push(
                    Item::new_from_element(Coverage::new_stroke(&Stroke {
                        align: alignment,
                        ..Stroke::new(4.)
                    }))
                    .with_attribute(
                        ATTR_PAINT,
                        Graphic::from(Color::from_gamma_srgb_channels(0., 0., 1., 1.)),
                    ),
                );
                let vector = Item::new_from_element(Vector::from_bezpath(path))
                    .with_attribute(ATTR_APPEARANCE, appearance);
                let before = vector.clone();
                let prepared = scene(&[CpuSvgSceneLayer::Vector(vector.clone())], 24, 32);
                let svg_before = prepared.svg().svg.clone();
                assert!(svg_before.contains("stroke-width=\"4\""));
                assert!(!svg_before.contains("stroke-width=\"8\""));
                assert!(!svg_before.contains("alignment-"));
                assert!(!svg_before.contains("<clipPath"));
                assert!(!svg_before.contains("<mask"));
                let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 48 * 64 * 4)
                    .expect("original whole-vector centered fallback");
                assert_eq!(
                    (converted.size(0).width.0, converted.size(0).height.0),
                    (48, 64)
                );
                // One open contour disables alignment for every contour. The
                // authored width-four stroke extends two units on either side.
                assert_eq!(pixel(&converted, 16, 14), [255, 0, 0, 255]);
                assert_eq!(pixel(&converted, 16, 18), [255, 0, 0, 255]);
                assert_eq!(pixel(&converted, 16, 10), [0, 0, 0, 0]);
                assert_eq!(pixel(&converted, 16, 20), [0, 0, 0, 0]);
                if include_closed_contour {
                    assert_eq!(pixel(&converted, 16, 30), [255, 0, 0, 255]);
                    assert_eq!(pixel(&converted, 16, 34), [255, 0, 0, 255]);
                    assert_eq!(pixel(&converted, 16, 26), [0, 0, 0, 0]);
                    assert_eq!(pixel(&converted, 16, 36), [0, 0, 0, 0]);
                } else {
                    assert_eq!(pixel(&converted, 16, 30), [0, 0, 0, 0]);
                    assert_eq!(pixel(&converted, 16, 34), [0, 0, 0, 0]);
                }
                assert_eq!(vector, before);
                assert_eq!(prepared.svg().svg, svg_before);
                for retained in [
                    &vector,
                    match prepared.layers() {
                        [CpuSvgSceneLayer::Vector(retained)] => retained,
                        _ => panic!("the fallback vector layer changed"),
                    },
                ] {
                    assert_eq!(retained, &before);
                    let stroke = retained
                        .attribute::<Appearance>(ATTR_APPEARANCE)
                        .expect("original appearance")
                        .cover_at(1)
                        .expect("original authored stroke")
                        .stroke_params();
                    assert_eq!(stroke.align, alignment);
                    assert_eq!(stroke.weight.to_bits(), 4_f64.to_bits());
                    assert_eq!(stroke.transform, DAffine2::IDENTITY);
                }
            }
        }
    }

    fn owned_text(text: &str) -> Item<String> {
        let font = Resource::new(FALLBACK_FONT_RESOURCE.as_ref().to_vec());
        Item::new_from_element(text.to_owned())
            .with_attribute(ATTR_FONT, font)
            .with_attribute(ATTR_FONT_SIZE, 48_f64)
            .with_attribute(
                ATTR_TRANSFORM,
                DAffine2::from_cols_array(&[1., 0., 0., 1., 8., 8.]),
            )
    }

    fn original_glyph_interior_samples(text: &Item<String>) -> Vec<(usize, usize)> {
        let shaped = shape_text_item(text, false);
        let mut paths = Vec::new();
        for vector in shaped.into_iter() {
            let transform = vector.attribute_cloned_or_default::<DAffine2>(ATTR_TRANSFORM);
            for mut path in vector.element().stroke_bezpath_iter() {
                path.apply_affine(Affine::new(transform.to_cols_array()));
                paths.push(path);
            }
        }
        let mut samples = Vec::new();
        for y in 0..192 {
            for x in 0..256 {
                let center = Point::new((x as f64 + 0.5) / 2., (y as f64 + 0.5) / 2.);
                // Original outlined glyphs contain this whole pixel footprint
                // plus a margin; avoid choosing antialiased boundary samples.
                let interior = [-0.75, 0., 0.75].into_iter().all(|vertical| {
                    [-0.75, 0., 0.75].into_iter().all(|horizontal| {
                        paths
                            .iter()
                            .map(|path| {
                                path.winding(Point::new(center.x + horizontal, center.y + vertical))
                            })
                            .sum::<i32>()
                            != 0
                    })
                });
                if interior {
                    samples.push((x, y));
                    if samples.len() == 8 {
                        return samples;
                    }
                }
            }
        }
        panic!("the explicit original text must have stable glyph interiors")
    }

    fn assert_owned_text_retained(
        actual: &Item<String>,
        before: &Item<String>,
        is_caller_backing: bool,
    ) {
        assert_eq!(actual, before);
        let font = actual.attribute::<Resource>(ATTR_FONT).expect("owned font");
        let original_font = before
            .attribute::<Resource>(ATTR_FONT)
            .expect("original owned font");
        assert_eq!(font.hash(), original_font.hash());
        assert_eq!(font.as_ref(), original_font.as_ref());
        assert_eq!(font.hash(), Resource::new(font.as_ref().to_vec()).hash());
        let actual_owner: Arc<dyn AsRef<[u8]> + Send + Sync> = font.into();
        let before_owner: Arc<dyn AsRef<[u8]> + Send + Sync> = original_font.into();
        if is_caller_backing {
            assert!(Arc::ptr_eq(&actual_owner, &before_owner));
        } else {
            assert!(!Arc::ptr_eq(&actual_owner, &before_owner));
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_owned_font_black_glyph_interiors_and_retention() {
        let text = owned_text("H");
        let before = text.clone();
        let font = text
            .attribute::<Resource>(ATTR_FONT)
            .expect("explicit font");
        assert_eq!(font.len(), 119_080);
        assert_eq!(font.as_ref(), FALLBACK_FONT_RESOURCE.as_ref());
        let samples = original_glyph_interior_samples(&text);
        let prepared = scene(&[CpuSvgSceneLayer::Text(text.clone())], 128, 96);
        let svg_before = prepared.svg().svg.clone();
        assert!(svg_before.contains("fill=\"black\""));
        assert!(svg_before.contains("fill-rule=\"nonzero\""));
        assert!(!svg_before.contains("<text"));
        let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., 256 * 192 * 4)
            .expect("actual original outlined font preview");
        assert_eq!(
            (converted.size(0).width.0, converted.size(0).height.0),
            (256, 192)
        );
        for (x, y) in samples {
            assert_eq!(pixel(&converted, x, y), [0, 0, 0, 255]);
        }
        assert_eq!(pixel(&converted, 250, 190), [0, 0, 0, 0]);
        for pixel in converted
            .as_bytes(0)
            .expect("text frame")
            .as_chunks::<4>()
            .0
            .iter()
        {
            assert_eq!(&pixel[..3], &[0, 0, 0]);
        }
        assert_owned_text_retained(&text, &before, true);
        assert_eq!(prepared.svg().svg, svg_before);
        match prepared.layers() {
            [CpuSvgSceneLayer::Text(retained)] => {
                assert_owned_text_retained(retained, &before, false)
            }
            _ => panic!("the original text item was not retained"),
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_original_text_alpha_exact_cap_and_recovery() {
        let text = owned_text("H")
            .with_attribute(ATTR_OPACITY, 0.5_f64)
            .with_attribute(ATTR_OPACITY_FILL, 0.5_f64);
        let before = text.clone();
        let samples = original_glyph_interior_samples(&text);
        let prepared = scene(&[CpuSvgSceneLayer::Text(text.clone())], 128, 96);
        let svg_before = prepared.svg().svg.clone();
        assert!(svg_before.contains("opacity=\"0.25\""));
        let required = 256 * 192 * 4;
        let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., required)
            .expect("source combined text opacity");
        assert_eq!(
            (converted.size(0).width.0, converted.size(0).height.0),
            (256, 192)
        );
        assert_eq!(converted.as_bytes(0).expect("text frame").len(), required);
        for (x, y) in samples {
            assert_pixel_close(pixel(&converted, x, y), [0, 0, 0, 64], 1);
        }
        for maximum in [0, required - 1] {
            assert_eq!(
                prepare_cpu_svg_preview(&prepared, &renderer(), 1., maximum).err(),
                Some(CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum })
            );
            assert_owned_text_retained(&text, &before, true);
            assert_eq!(prepared.svg().svg, svg_before);
        }
        let recovered = prepare_cpu_svg_preview(&prepared, &renderer(), 1., required)
            .expect("text preview recovery after cap failure");
        assert_eq!(
            recovered.as_bytes(0).expect("recovered text frame"),
            converted.as_bytes(0).expect("original text frame")
        );
        match prepared.layers() {
            [CpuSvgSceneLayer::Text(retained)] => {
                assert_owned_text_retained(retained, &before, false)
            }
            _ => panic!("the original alpha text item changed"),
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_blank_text_and_visible_control_glyphs_follow_source() {
        for (text, expected_paths) in [
            (owned_text(""), false),
            (owned_text(" \t\n "), true),
            (owned_text("   "), false),
            (
                owned_text("H").with_attribute(ATTR_MAX_HEIGHT, 1_f64),
                false,
            ),
        ] {
            if text.element() == "H" {
                TextContext::with_thread_local(|context| {
                    let font = text
                        .attribute::<Resource>(ATTR_FONT)
                        .expect("explicit font");
                    let layout = context
                        .layout_text(
                            text.element(),
                            font,
                            TypesettingConfig {
                                font_size: *text
                                    .attribute::<f64>(ATTR_FONT_SIZE)
                                    .expect("explicit size"),
                                ..TypesettingConfig::default()
                            },
                        )
                        .expect("original height-clipping baseline");
                    assert!(layout.lines().all(|line| line.metrics().baseline > 1.));
                });
            }
            let before = text.clone();
            let shaped = shape_text_item(&text, false);
            assert_eq!(
                shaped.iter_element_values().any(|vector| {
                    vector
                        .stroke_bezpath_iter()
                        .any(|path| !path.elements().is_empty())
                }),
                expected_paths,
                "original shape helper for {:?}",
                text.element()
            );
            let prepared = scene(&[CpuSvgSceneLayer::Text(text.clone())], 128, 96);
            let svg_before = prepared.svg().svg.clone();
            assert_eq!(svg_before.contains("fill=\"black\""), expected_paths);
            let required = 256 * 192 * 4;
            let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., required)
                .expect("original blank or control-glyph output");
            assert_eq!(
                (converted.size(0).width.0, converted.size(0).height.0),
                (256, 192)
            );
            if expected_paths {
                for (x, y) in original_glyph_interior_samples(&text) {
                    assert_eq!(pixel(&converted, x, y), [0, 0, 0, 255]);
                }
                assert_eq!(pixel(&converted, 250, 190), [0, 0, 0, 0]);
                for pixel in converted
                    .as_bytes(0)
                    .expect("control-glyph frame")
                    .as_chunks::<4>()
                    .0
                    .iter()
                {
                    assert_eq!(&pixel[..3], &[0, 0, 0]);
                }
            } else {
                assert!(
                    converted
                        .as_bytes(0)
                        .expect("empty text frame")
                        .iter()
                        .all(|byte| *byte == 0)
                );
            }
            assert_eq!(
                prepare_cpu_svg_preview(&prepared, &renderer(), 1., required - 1).err(),
                Some(CpuSvgPreviewError::PreviewBudgetExceeded {
                    required,
                    maximum: required - 1
                })
            );
            let recovered = prepare_cpu_svg_preview(&prepared, &renderer(), 1., required)
                .expect("empty text cap recovery");
            assert_eq!(
                recovered.as_bytes(0).expect("recovered empty frame"),
                converted.as_bytes(0).expect("original empty frame")
            );
            assert_owned_text_retained(&text, &before, true);
            assert_eq!(prepared.svg().svg, svg_before);
            match prepared.layers() {
                [CpuSvgSceneLayer::Text(retained)] => {
                    assert_owned_text_retained(retained, &before, false)
                }
                _ => panic!("the legitimate blank text item changed"),
            }
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_ordered_text_list_styles_overlap_cap_and_retention() {
        let first = owned_text("H").with_attribute(ATTR_OPACITY, 0.5_f64);
        let second = owned_text("H")
            .with_attribute(ATTR_FONT_SIZE, 64_f64)
            .with_attribute(
                ATTR_TRANSFORM,
                DAffine2::from_translation(graphite_glam::DVec2::new(6., 8.)),
            )
            .with_attribute(ATTR_OPACITY, 0.25_f64);
        let before = [first.clone(), second.clone()];
        let mut texts = List::new_from_item(first.clone());
        texts.push(second.clone());
        let list_before = texts.clone();
        let prepared = scene(&[CpuSvgSceneLayer::TextList(texts.clone())], 128, 96);
        let svg_before = prepared.svg().svg.clone();
        assert!(
            svg_before
                .find("matrix(1,0,0,1,8,8)")
                .expect("first row transform")
                < svg_before
                    .find("matrix(1,0,0,1,6,8)")
                    .expect("second row transform")
        );
        assert!(
            svg_before
                .find("opacity=\"0.5\"")
                .expect("first row opacity")
                < svg_before
                    .find("opacity=\"0.25\"")
                    .expect("second row opacity")
        );
        let required = 256 * 192 * 4;
        let converted = prepare_cpu_svg_preview(&prepared, &renderer(), 1., required)
            .expect("original ordered text list preview");
        assert_eq!(
            (converted.size(0).width.0, converted.size(0).height.0),
            (256, 192)
        );
        assert_eq!(converted.as_bytes(0).expect("list frame").len(), required);

        let paths = before.each_ref().map(|text| {
            shape_text_item(text, false)
                .into_iter()
                .flat_map(|vector| {
                    let transform = vector.attribute_cloned_or_default::<DAffine2>(ATTR_TRANSFORM);
                    vector
                        .element()
                        .stroke_bezpath_iter()
                        .map(move |mut path| {
                            path.apply_affine(Affine::new(transform.to_cols_array()));
                            path
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        let mut samples = [None, None, None];
        for y in 0..192 {
            for x in 0..256 {
                let center = Point::new((x as f64 + 0.5) / 2., (y as f64 + 0.5) / 2.);
                let coverage = paths.each_ref().map(|paths| {
                    let winding = |horizontal: f64, vertical: f64| {
                        paths
                            .iter()
                            .map(|path| {
                                path.winding(Point::new(center.x + horizontal, center.y + vertical))
                            })
                            .sum::<i32>()
                            != 0
                    };
                    let interior = [-0.75, 0., 0.75].into_iter().all(|vertical| {
                        [-0.75, 0., 0.75]
                            .into_iter()
                            .all(|horizontal| winding(horizontal, vertical))
                    });
                    let exterior = [-0.75, 0., 0.75].into_iter().all(|vertical| {
                        [-0.75, 0., 0.75]
                            .into_iter()
                            .all(|horizontal| !winding(horizontal, vertical))
                    });
                    (interior, exterior)
                });
                let category = match coverage {
                    [(true, _), (_, true)] => Some(0),
                    [(_, true), (true, _)] => Some(1),
                    [(true, _), (true, _)] => Some(2),
                    _ => None,
                };
                if let Some(category) = category
                    && samples[category].is_none()
                {
                    samples[category] = Some((x, y));
                }
            }
        }
        // Both original glyphs are black; their overlap retains source-over alpha.
        for (sample, alpha) in samples.into_iter().zip([128, 64, 159]) {
            let (x, y) =
                sample.expect("source outlines contain exclusive and overlapping interiors");
            assert_pixel_close(pixel(&converted, x, y), [0, 0, 0, alpha], 1);
        }
        assert_eq!(pixel(&converted, 250, 190), [0, 0, 0, 0]);
        assert!(
            converted
                .as_bytes(0)
                .expect("list frame")
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| pixel[..3] == [0, 0, 0])
        );

        let ordered_items = scene(
            &[
                CpuSvgSceneLayer::Text(first.clone()),
                CpuSvgSceneLayer::Text(second.clone()),
            ],
            128,
            96,
        );
        let item_frame = prepare_cpu_svg_preview(&ordered_items, &renderer(), 1., required)
            .expect("same ordered original item renderers");
        assert_eq!(converted.as_bytes(0), item_frame.as_bytes(0));
        let mut reverse = List::new_from_item(second);
        reverse.push(first);
        let reversed = scene(&[CpuSvgSceneLayer::TextList(reverse)], 128, 96);
        let reversed_svg = &reversed.svg().svg;
        assert!(
            reversed_svg
                .find("matrix(1,0,0,1,6,8)")
                .expect("reversed first transform")
                < reversed_svg
                    .find("matrix(1,0,0,1,8,8)")
                    .expect("reversed second transform")
        );
        assert!(
            reversed_svg
                .find("opacity=\"0.25\"")
                .expect("reversed first opacity")
                < reversed_svg
                    .find("opacity=\"0.5\"")
                    .expect("reversed second opacity")
        );
        for maximum in [0, required - 1] {
            assert_eq!(
                prepare_cpu_svg_preview(&prepared, &renderer(), 1., maximum).err(),
                Some(CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum })
            );
        }
        let recovered = prepare_cpu_svg_preview(&prepared, &renderer(), 1., required)
            .expect("text list cap recovery");
        assert_eq!(recovered.as_bytes(0), converted.as_bytes(0));
        assert_eq!(texts, list_before);
        assert_eq!(prepared.svg().svg, svg_before);
        let retained = match prepared.layers() {
            [CpuSvgSceneLayer::TextList(retained)] => retained,
            _ => panic!("the original text list layer changed"),
        };
        assert_eq!(retained, &list_before);
        assert_eq!(retained.len(), 2);
        for (index, original) in before.iter().enumerate() {
            assert_owned_text_retained(
                &texts.clone_item(index).expect("input row"),
                original,
                true,
            );
            assert_owned_text_retained(
                &retained.clone_item(index).expect("retained row"),
                original,
                false,
            );
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_color_pure_rgb_source_alpha_and_ignored_inner_transform() {
        let mut cases = vec![(
            Color::from(SRGBA8::new_opaque(64, 128, 191)),
            [191, 128, 64, 255],
            None,
        )];
        cases.extend(
            [
                (0., 0, Some("0")),
                (0.6251, 159, Some("0.625")),
                (0.6254, 159, Some("0.625")),
                (1., 255, None),
                (2., 255, None),
                (-0.25, 0, Some("-0.25")),
            ]
            .map(|(alpha, expected_alpha, serialized)| {
                (
                    Color::from_rgbaf32_unchecked(1., 0., 0., alpha),
                    if expected_alpha == 0 {
                        [0, 0, 0, 0]
                    } else {
                        [0, 0, 255, expected_alpha]
                    },
                    serialized,
                )
            }),
        );
        for (color, expected, serialized_alpha) in cases {
            let item = Item::new_from_element(color);
            let moved = item.clone().with_attribute(
                ATTR_TRANSFORM,
                DAffine2::from_cols_array(&[-2., 3., 4., -5., 123456., -987654.]),
            );
            let before = moved.clone();
            let prepared = scene(&[CpuSvgSceneLayer::Color(moved.clone())], 16, 8);
            let unmoved = scene(&[CpuSvgSceneLayer::Color(item)], 16, 8);
            let svg_before = prepared.svg().svg.clone();
            assert_eq!(svg_before, unmoved.svg().svg);
            assert!(svg_before.contains("points=\"10000000,10000000 -10000000,10000000 -10000000,-10000000 10000000,-10000000\""));
            match serialized_alpha {
                Some(alpha) => assert!(svg_before.contains(&format!("fill-opacity=\"{alpha}\""))),
                None => assert!(!svg_before.contains("fill-opacity=")),
            }
            assert!(svg_before.contains(if expected[0] == 191 {
                "fill=\"#4080bf\""
            } else {
                "fill=\"#ff0000\""
            }));
            for (scale, dimensions) in [(0.5, (16, 8)), (1., (32, 16))] {
                let required = dimensions.0 as usize * dimensions.1 as usize * 4;
                let converted = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                    .expect("original infinite Color background");
                assert_eq!(
                    (converted.size(0).width.0, converted.size(0).height.0),
                    dimensions
                );
                assert_eq!(converted.as_bytes(0).expect("color frame").len(), required);
                for bytes in converted
                    .as_bytes(0)
                    .expect("color frame")
                    .as_chunks::<4>()
                    .0
                    .iter()
                {
                    let actual: [u8; 4] = *bytes;
                    assert_pixel_close(actual, expected, 1);
                }
                if expected[3] == 255 || expected[3] == 0 {
                    assert!(
                        converted
                            .as_bytes(0)
                            .expect("exact endpoint frame")
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|pixel| *pixel == expected)
                    );
                }
                for maximum in [0, required - 1] {
                    assert_eq!(
                        prepare_cpu_svg_preview(&prepared, &renderer(), scale, maximum).err(),
                        Some(CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum })
                    );
                }
                let recovered = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                    .expect("Color cap recovery");
                assert_eq!(recovered.as_bytes(0), converted.as_bytes(0));
            }
            assert_eq!(moved, before);
            assert_eq!(prepared.svg().svg, svg_before);
            match prepared.layers() {
                [CpuSvgSceneLayer::Color(retained)] => assert_eq!(retained, &before),
                _ => panic!("the original Color layer changed"),
            }
        }
    }

    #[test]
    fn at_r20_2_cpu_svg_preview_ordered_color_list_opacity_pixels_cap_and_retention() {
        let first = Item::new_from_element(Color::RED)
            .with_attribute(ATTR_OPACITY, 0.5_f64)
            .with_attribute(
                ATTR_TRANSFORM,
                DAffine2::from_translation(graphite_glam::DVec2::new(100., 200.)),
            );
        let second = Item::new_from_element(Color::BLUE)
            .with_attribute(ATTR_OPACITY, 0.5_f64)
            .with_attribute(ATTR_OPACITY_FILL, 0.5_f64)
            .with_attribute(
                ATTR_TRANSFORM,
                DAffine2::from_scale(graphite_glam::DVec2::new(-3., 4.)),
            );
        let before = [first.clone(), second.clone()];
        let mut ordered = List::new_from_item(first.clone());
        ordered.push(second.clone());
        let ordered_before = ordered.clone();
        let mut reversed = List::new_from_item(second.clone());
        reversed.push(first.clone());
        let reversed_before = reversed.clone();
        let individual = scene(
            &[
                CpuSvgSceneLayer::Color(first.clone()),
                CpuSvgSceneLayer::Color(second.clone()),
            ],
            16,
            8,
        );
        for (colors, expected, first_fill, second_fill) in [
            (ordered.clone(), [102, 0, 153, 159], "#ff0000", "#0000ff"),
            (reversed.clone(), [51, 0, 204, 159], "#0000ff", "#ff0000"),
        ] {
            let prepared = scene(&[CpuSvgSceneLayer::ColorList(colors.clone())], 16, 8);
            let svg_before = prepared.svg().svg.clone();
            assert!(
                svg_before.find(first_fill).expect("first Color row")
                    < svg_before.find(second_fill).expect("second Color row")
            );
            assert!(svg_before.contains("opacity=\"0.5\""));
            assert!(svg_before.contains("opacity=\"0.25\""));
            for (scale, dimensions) in [(0.5, (16, 8)), (1., (32, 16))] {
                let required = dimensions.0 as usize * dimensions.1 as usize * 4;
                let converted = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                    .expect("ordered original Color backgrounds");
                assert_eq!(
                    (converted.size(0).width.0, converted.size(0).height.0),
                    dimensions
                );
                assert_eq!(
                    converted.as_bytes(0).expect("ColorList frame").len(),
                    required
                );
                // Red alpha .5 then blue .25 yields .625 alpha with RGB .6/0/.4;
                // reversed rows yield .8/0/.2. Native premultiplication rounds bytes.
                for bytes in converted
                    .as_bytes(0)
                    .expect("ColorList frame")
                    .as_chunks::<4>()
                    .0
                    .iter()
                {
                    assert_pixel_close(*bytes, expected, 2);
                }
                if first_fill == "#ff0000" {
                    let item_frame =
                        prepare_cpu_svg_preview(&individual, &renderer(), scale, required)
                            .expect("same original ordered Color Items");
                    assert_eq!(converted.as_bytes(0), item_frame.as_bytes(0));
                }
                for maximum in [0, required - 1] {
                    assert_eq!(
                        prepare_cpu_svg_preview(&prepared, &renderer(), scale, maximum).err(),
                        Some(CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum })
                    );
                }
                let recovered = prepare_cpu_svg_preview(&prepared, &renderer(), scale, required)
                    .expect("ColorList cap recovery");
                assert_eq!(recovered.as_bytes(0), converted.as_bytes(0));
            }
            assert_eq!(prepared.svg().svg, svg_before);
            match prepared.layers() {
                [CpuSvgSceneLayer::ColorList(retained)] => assert_eq!(retained, &colors),
                _ => panic!("the ordered ColorList changed"),
            }
        }
        assert_eq!(ordered, ordered_before);
        assert_eq!(reversed, reversed_before);
        for index in 0..before.len() {
            assert_eq!(
                ordered.clone_item(index).expect("source ordered row"),
                ordered_before.clone_item(index).expect("original List row")
            );
            assert_eq!(
                reversed.clone_item(index).expect("source reversed row"),
                reversed_before
                    .clone_item(index)
                    .expect("original reversed List row")
            );
        }
        assert_eq!(
            ordered
                .clone_item(0)
                .expect("first List row")
                .attribute::<f64>(ATTR_OPACITY_FILL),
            Some(&1_f64)
        );
        assert_eq!(first.attribute::<f64>(ATTR_OPACITY_FILL), None);
        assert_eq!(&first, &before[0]);
        assert_eq!(&second, &before[1]);
        let empty = scene(&[CpuSvgSceneLayer::ColorList(List::new())], 16, 8);
        for (scale, required) in [(0.5, 16 * 8 * 4), (1., 32 * 16 * 4)] {
            let converted = prepare_cpu_svg_preview(&empty, &renderer(), scale, required)
                .expect("original empty ColorList");
            assert!(
                converted
                    .as_bytes(0)
                    .expect("empty ColorList frame")
                    .iter()
                    .all(|byte| *byte == 0)
            );
        }
    }

    #[test]
    fn at_r20_2_exact_canvas_fractional_roots_keep_physical_size_and_uniform_scale() {
        let red = rectangle(0., 0., 6., 5., [1., 0., 0., 1.]);
        let blue = rectangle(6., 0., 6., 5., [0., 0., 1., 0.5]);
        let red_before = red.clone();
        let blue_before = blue.clone();
        let renderer = renderer();
        for scale in [1., 1.5, 2.] {
            let configuration = configuration(17, 11, scale);
            let prepared = prepare_cpu_svg_composition(
                &[
                    CpuSvgSceneLayer::Vector(red.clone()),
                    CpuSvgSceneLayer::Vector(blue.clone()),
                ],
                configuration,
            )
            .expect("explicit non-square physical target");
            let svg_before = prepared.svg().svg.clone();
            let actual = prepare_cpu_svg_preview_at_target(&prepared, &renderer, 17 * 11 * 4)
                .expect("exact target conversion");
            assert_eq!(
                actual.size(0),
                gpui::size(gpui::DevicePixels(17), gpui::DevicePixels(11))
            );
            assert_eq!(actual.as_bytes(0).expect("single frame").len(), 17 * 11 * 4);
            assert_eq!(
                pixel(&actual, (2. * scale) as usize, (2. * scale) as usize),
                [0, 0, 255, 255]
            );
            assert_pixel_close(
                pixel(&actual, (8. * scale) as usize, (2. * scale) as usize),
                [255, 0, 0, 128],
                1,
            );
            assert_eq!(pixel(&actual, 2, 10), [0, 0, 0, 0]);
            assert_eq!(prepared.svg().svg, svg_before);
            assert_eq!(
                prepared.configuration().viewport.resolution,
                configuration.viewport.resolution
            );
            assert_eq!(prepared.configuration().scale.to_bits(), scale.to_bits());
            match prepared.layers() {
                [
                    CpuSvgSceneLayer::Vector(retained_red),
                    CpuSvgSceneLayer::Vector(retained_blue),
                ] => {
                    assert_eq!(retained_red, &red_before);
                    assert_eq!(retained_blue, &blue_before);
                }
                _ => panic!("original vector layers were not retained"),
            }
            let legacy = prepare_cpu_svg_preview(&prepared, &renderer, scale as f32, 34 * 22 * 4)
                .expect("unchanged legacy supersampling");
            let (width, height, bytes) =
                preview_layout(&prepared, scale as f32).expect("legacy layout");
            assert_eq!(legacy.size(0).width.0, width as i32);
            assert_eq!(legacy.size(0).height.0, height as i32);
            assert_eq!(legacy.as_bytes(0).expect("legacy frame").len(), bytes);
        }
    }

    #[test]
    fn at_r20_2_exact_canvas_embedded_png_cap_recovery_retains_original_result() {
        let source = image_result();
        let image_before = source.pixels().image().clone();
        let attributes_before = source.pixels().attributes().clone();
        let metadata_before = (
            source.input_identity().clone(),
            source.input_origin(),
            source.operation(),
            *source.descriptor(),
            source.pixels_sha256(),
        );
        let renderer = renderer();
        for scale in [1., 1.5, 2.] {
            let prepared = prepare_cpu_svg_composition(
                &[CpuSvgSceneLayer::Image(source.clone())],
                configuration(35, 21, scale),
            )
            .expect("original asymmetric embedded PNG");
            let svg_before = prepared.svg().svg.clone();
            let required = 35 * 21 * 4;
            for maximum in [0, required - 1] {
                assert_eq!(
                    prepare_cpu_svg_preview_at_target(&prepared, &renderer, maximum)
                        .expect_err("cap rejection"),
                    CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum },
                );
            }
            let actual = prepare_cpu_svg_preview_at_target(&prepared, &renderer, required)
                .expect("exact cap recovery");
            assert_eq!(
                actual.size(0),
                gpui::size(gpui::DevicePixels(35), gpui::DevicePixels(21))
            );
            assert_eq!(actual.as_bytes(0).expect("single frame").len(), required);
            assert_pixel_close(
                pixel(&actual, (3.5 * scale) as usize, (3.5 * scale) as usize),
                [255, 255, 0, 255],
                1,
            );
            assert_pixel_close(
                pixel(&actual, (12.5 * scale) as usize, (3.5 * scale) as usize),
                [0, 255, 255, 128],
                2,
            );
            assert_eq!(pixel(&actual, 34, 20), [0, 0, 0, 0]);
            let repeated = prepare_cpu_svg_preview_at_target(&prepared, &renderer, required)
                .expect("repeated exact target");
            assert_eq!(actual.as_bytes(0), repeated.as_bytes(0));
            assert_eq!(prepared.svg().svg, svg_before);
            match prepared.layers() {
                [CpuSvgSceneLayer::Image(retained)] => {
                    assert_eq!(retained.pixels().image(), &image_before);
                    assert_eq!(retained.pixels().attributes(), &attributes_before);
                    assert_eq!(
                        (
                            retained.input_identity().clone(),
                            retained.input_origin(),
                            retained.operation(),
                            *retained.descriptor(),
                            retained.pixels_sha256()
                        ),
                        metadata_before
                    );
                }
                _ => panic!("original result layer was not retained"),
            }
        }
        assert_eq!(source.pixels().image(), &image_before);
        assert_eq!(source.pixels().attributes(), &attributes_before);
        assert_eq!(
            (
                source.input_identity().clone(),
                source.input_origin(),
                source.operation(),
                *source.descriptor(),
                source.pixels_sha256()
            ),
            metadata_before
        );
    }

    #[test]
    fn at_r20_2_exact_canvas_rejects_native_axis_limit_without_resizing() {
        let red = rectangle(0., 0., 6., 5., [1., 0., 0., 1.]);
        let before = red.clone();
        for (width, height) in [(8193, 11), (17, 8193)] {
            let prepared = prepare_cpu_svg_composition(
                &[CpuSvgSceneLayer::Vector(red.clone())],
                configuration(width, height, 1.),
            )
            .expect("finite original scene above native canvas limit");
            assert_eq!(
                prepare_cpu_svg_preview_at_target(&prepared, &renderer(), usize::MAX)
                    .expect_err("no silent resize"),
                CpuSvgPreviewError::RendererSizeLimit
            );
            match prepared.layers() {
                [CpuSvgSceneLayer::Vector(retained)] => assert_eq!(retained, &before),
                _ => panic!("original vector was not retained"),
            }
        }
        let prepared = scene(&[CpuSvgSceneLayer::Vector(red)], 17, 11);
        let recovered = prepare_cpu_svg_preview_at_target(&prepared, &renderer(), 17 * 11 * 4)
            .expect("valid canvas after rejected oversized targets");
        assert_eq!(
            recovered.size(0),
            gpui::size(gpui::DevicePixels(17), gpui::DevicePixels(11))
        );
        assert_eq!(pixel(&recovered, 2, 2), [0, 0, 255, 255]);
    }

    #[test]
    fn at_r20_2_exact_canvas_rejects_unrepresentable_root_scale_before_parse_and_recovers() {
        let renderer = renderer();
        let tiny = prepare_cpu_svg_composition(&[], configuration(17, 11, 1e-39))
            .expect("finite original f64 scene with oversized f32 root");
        let svg_before = tiny.svg().svg.clone();
        assert!(tiny.configuration().scale as f32 > 0.);
        assert_eq!(
            prepare_cpu_svg_preview_at_target(&tiny, &renderer, 17 * 11 * 4)
                .expect_err("root narrowing fails before parsing"),
            CpuSvgPreviewError::InvalidDimensions,
        );
        assert_eq!(tiny.svg().svg, svg_before);
        assert!(tiny.layers().is_empty());
        for scale in [1e-50, 1e39] {
            let prepared = prepare_cpu_svg_composition(&[], configuration(17, 11, scale))
                .expect("finite original f64 scale");
            let before = prepared.svg().svg.clone();
            assert_eq!(
                prepare_cpu_svg_preview_at_target(&prepared, &renderer, 17 * 11 * 4)
                    .expect_err("scale cannot narrow to positive finite f32"),
                CpuSvgPreviewError::InvalidScale,
            );
            assert_eq!(prepared.svg().svg, before);
            assert_eq!(prepared.configuration().scale.to_bits(), scale.to_bits());
        }
        let prepared = prepare_cpu_svg_composition(&[], configuration(17, 11, 1.5))
            .expect("representable original scene");
        let recovered = prepare_cpu_svg_preview_at_target(&prepared, &renderer, 17 * 11 * 4)
            .expect("valid fractional root after rejected scales");
        assert_eq!(
            recovered.size(0),
            gpui::size(gpui::DevicePixels(17), gpui::DevicePixels(11))
        );
        assert_eq!(
            recovered.as_bytes(0).expect("single empty frame").len(),
            17 * 11 * 4
        );
        assert!(
            recovered
                .as_bytes(0)
                .expect("empty original scene")
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    fn accepted_artboard_document(
        width: u32,
        height: u32,
        scale: f64,
        parameters: graphics_core::evaluate::CpuSvgArtboardParameters,
    ) -> (
        graphics_core::evaluate::CpuSvgArtboardDocument,
        graphics_core::evaluate::AcceptedCpuSvgOutput,
    ) {
        let configuration = graphite_std::application_io::RenderConfig {
            viewport: Footprint {
                transform: DAffine2::from_translation(graphite_glam::DVec2::ONE),
                resolution: UVec2::new(width, height),
                quality: Footprint::DEFAULT.quality,
            },
            scale,
            time: graphite_std::application_io::TimingInformation {
                time: 0.,
                animation_time: std::time::Duration::ZERO,
            },
            pointer: graphite_glam::DVec2::ZERO,
            render_mode: graphite_vector_types::vector::style::RenderMode::Normal,
            export_format: graphite_std::application_io::ExportFormat::Svg,
            for_export: true,
            for_eyedropper: false,
        };
        let mut document = graphics_core::evaluate::CpuSvgArtboardDocument::new(
            91,
            graphite_core_types::uuid::NodeId(7001),
            parameters,
            configuration,
            graphics_core::evaluate::CpuSvgEvaluationBudget {
                max_viewport_pixels: width.checked_mul(height).expect("finite fixture viewport"),
                max_output_bytes: 64 * 1024,
            },
        )
        .expect("genuine original Artboard document");
        let evaluation = document.submit();
        let completion = futures::executor::block_on(evaluation.evaluate());
        let accepted = document
            .accept(completion)
            .expect("genuine original compiled output")
            .clone();
        (document, accepted)
    }

    fn accepted_artboard_parameters() -> graphics_core::evaluate::CpuSvgArtboardParameters {
        graphics_core::evaluate::CpuSvgArtboardParameters {
            location: graphite_glam::DVec2::new(2., 2.),
            dimensions: graphite_glam::DVec2::new(6., 4.),
            background: Color::from_gamma_srgb_channels(1., 0., 0., 0.5),
            clip: true,
        }
    }

    #[test]
    fn at_r20_2_accepted_original_svg_exact_target_pixels_and_metadata() {
        let renderer = renderer();
        let parameters = accepted_artboard_parameters();
        for scale in [1., 1.5, 2.] {
            let (document, accepted) = accepted_artboard_document(31, 17, scale, parameters);
            let source_before = document.network().clone();
            let output_before = accepted.output().clone();
            let output_lease = accepted.output_lease();
            let configuration_before = accepted.configuration();
            let image =
                prepare_accepted_cpu_svg_preview_at_target(&accepted, &renderer, 31 * 17 * 4)
                    .expect("genuine original output converted at exact target");
            assert_eq!(image.frame_count(), 1);
            assert_eq!(
                image.size(0),
                gpui::size(gpui::DevicePixels(31), gpui::DevicePixels(17))
            );
            assert_eq!(
                image.as_bytes(0).expect("exact single frame").len(),
                31 * 17 * 4
            );
            assert_pixel_close(
                pixel(&image, (5. * scale) as usize, (5. * scale) as usize),
                [0, 0, 255, 128],
                1,
            );
            assert_eq!(pixel(&image, 0, 0), [0, 0, 0, 0]);
            assert_eq!(pixel(&image, 30, 16), [0, 0, 0, 0]);
            let graphite_graph_craft::document::value::RenderOutputType::Svg { svg, image_data } =
                &accepted.output().data
            else {
                panic!("genuine accepted resource-free SVG");
            };
            assert!(image_data.is_empty());
            assert!(svg.contains("fill-opacity=\"0.5\""));
            assert!(svg.contains("fill=\"#ff0000\""));
            let parsed = renderer
                .parse_svg(svg.as_bytes())
                .expect("original SVG tree");
            let original_frame = renderer
                .render_parsed_in_canvas(
                    &parsed,
                    gpui::size(gpui::DevicePixels(31), gpui::DevicePixels(17)),
                    scale as f32,
                    31 * 17 * 4,
                )
                .expect("existing full-color native renderer");
            assert_eq!(image.as_bytes(0), original_frame.as_bytes(0));
            assert_eq!(accepted.output(), &output_before);
            assert_eq!(accepted.output().metadata, output_before.metadata);
            assert_eq!(accepted.configuration(), configuration_before);
            assert_eq!(
                accepted
                    .configuration()
                    .viewport
                    .transform
                    .to_cols_array()
                    .map(f64::to_bits),
                configuration_before
                    .viewport
                    .transform
                    .to_cols_array()
                    .map(f64::to_bits)
            );
            assert_eq!(accepted.configuration().scale.to_bits(), scale.to_bits());
            assert_eq!(accepted.document_id_encoding(), 91);
            assert_eq!(accepted.node_id(), graphite_core_types::uuid::NodeId(7001));
            assert_eq!(accepted.revision(), 0);
            assert_eq!(accepted.output().metadata.backgrounds.len(), 1);
            let background = accepted
                .output()
                .metadata
                .backgrounds
                .first()
                .expect("original background metadata");
            assert_eq!(background.location, parameters.location);
            assert_eq!(background.dimensions, parameters.dimensions);
            assert!(
                accepted
                    .output()
                    .metadata
                    .click_targets
                    .contains_key(&accepted.node_id())
            );
            assert!(
                accepted
                    .output()
                    .metadata
                    .clip_targets
                    .contains(&accepted.node_id())
            );
            assert!(Arc::ptr_eq(&output_lease, &accepted.output_lease()));
            assert!(document.owns_current_output(&accepted));
            assert_eq!(document.network(), &source_before);
        }
    }

    #[test]
    fn at_r20_2_accepted_original_svg_cap_precision_parser_recovery() {
        let renderer = renderer();
        let (document, accepted) =
            accepted_artboard_document(19, 13, 1.5, accepted_artboard_parameters());
        let before = accepted.output().clone();
        let network_before = document.network().clone();
        let required = 19 * 13 * 4;
        for maximum in [0, required - 1] {
            assert_eq!(
                prepare_accepted_cpu_svg_preview_at_target(&accepted, &renderer, maximum)
                    .expect_err("output cap before parsing"),
                CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum },
            );
        }
        let exact = prepare_accepted_cpu_svg_preview_at_target(&accepted, &renderer, required)
            .expect("exact cap accepted");
        assert_eq!(exact.as_bytes(0).expect("one exact frame").len(), required);
        assert_pixel_close(pixel(&exact, 7, 7), [0, 0, 255, 128], 1);
        for (width, height) in [(8193, 11), (17, 8193)] {
            let (large_document, large) =
                accepted_artboard_document(width, height, 1., accepted_artboard_parameters());
            let source_before = large_document.network().clone();
            let output_before = large.output().clone();
            assert_eq!(
                prepare_accepted_cpu_svg_preview_at_target(&large, &renderer, usize::MAX)
                    .expect_err("no silent target resize"),
                CpuSvgPreviewError::RendererSizeLimit
            );
            assert_eq!(large.output(), &output_before);
            assert_eq!(large_document.network(), &source_before);
            assert!(large_document.owns_current_output(&large));
        }
        for (scale, expected) in [
            (1e-39, CpuSvgPreviewError::InvalidDimensions),
            (1e-50, CpuSvgPreviewError::InvalidScale),
            (1e39, CpuSvgPreviewError::InvalidScale),
        ] {
            let (narrowed_document, narrowed) =
                accepted_artboard_document(19, 13, scale, accepted_artboard_parameters());
            let output_before = narrowed.output().clone();
            assert_eq!(
                prepare_accepted_cpu_svg_preview_at_target(&narrowed, &renderer, required)
                    .expect_err("native root and scale narrowing"),
                expected
            );
            assert_eq!(narrowed.output(), &output_before);
            assert_eq!(narrowed.configuration().scale.to_bits(), scale.to_bits());
            assert!(narrowed_document.owns_current_output(&narrowed));
        }
        assert!(matches!(
            prepare_svg_bytes_at_target(b"<svg", 19, 13, 1.5, &renderer, required),
            Err(CpuSvgPreviewError::SourceConversion(_))
        ));
        for scale in [0., -0., f64::NAN, f64::INFINITY] {
            assert_eq!(
                prepare_svg_bytes_at_target(b"<svg", 19, 13, scale, &renderer, required)
                    .expect_err("bad scale rejected before malformed SVG parsing"),
                CpuSvgPreviewError::InvalidScale
            );
        }
        assert_eq!(
            prepare_svg_bytes_at_target(b"<svg", 19, 13, 1.5, &renderer, 0)
                .expect_err("cap rejected before malformed SVG parsing"),
            CpuSvgPreviewError::PreviewBudgetExceeded {
                required,
                maximum: 0
            }
        );
        let recovered = prepare_accepted_cpu_svg_preview_at_target(&accepted, &renderer, required)
            .expect("genuine accepted output after failed conversions");
        assert_eq!(recovered.as_bytes(0), exact.as_bytes(0));
        assert_eq!(accepted.output(), &before);
        assert_eq!(accepted.output().metadata, before.metadata);
        assert_eq!(document.network(), &network_before);
        assert!(document.owns_current_output(&accepted));
    }

    #[test]
    fn at_r20_2_accepted_original_svg_old_snapshot_immutable_owner_rejects_successor() {
        let renderer = renderer();
        let (mut document, old) =
            accepted_artboard_document(19, 13, 1., accepted_artboard_parameters());
        let old_output = old.output().clone();
        let old_lease = old.output_lease();
        let old_network = document.network().clone();
        let old_frame = prepare_accepted_cpu_svg_preview_at_target(&old, &renderer, 19 * 13 * 4)
            .expect("current original frame");
        assert_pixel_close(pixel(&old_frame, 6, 5), [0, 0, 255, 128], 1);
        assert!(document.owns_current_output(&old));
        let parameters = graphics_core::evaluate::CpuSvgArtboardParameters {
            location: graphite_glam::DVec2::new(4., 2.),
            dimensions: graphite_glam::DVec2::new(4., 4.),
            background: Color::from_gamma_srgb_channels(0., 0., 1., 0.75),
            clip: false,
        };
        assert!(
            document
                .apply(0, parameters, old.configuration())
                .expect("genuine source successor")
        );
        assert!(!document.owns_current_output(&old));
        let stale_conversion =
            prepare_accepted_cpu_svg_preview_at_target(&old, &renderer, 19 * 13 * 4)
                .expect("immutable old snapshot can still be converted");
        assert_eq!(stale_conversion.as_bytes(0), old_frame.as_bytes(0));
        assert!(!document.owns_current_output(&old));
        let successor_network = document.network().clone();
        assert_ne!(successor_network, old_network);
        let evaluation = document.submit();
        let completion = futures::executor::block_on(evaluation.evaluate());
        let current_output = completion
            .result()
            .expect("genuine source evaluation")
            .clone();
        let current = document
            .accept(completion)
            .expect("genuine new accepted output")
            .clone();
        let current_frame =
            prepare_accepted_cpu_svg_preview_at_target(&current, &renderer, 19 * 13 * 4)
                .expect("current original source frame");
        assert_pixel_close(pixel(&current_frame, 6, 5), [255, 0, 0, 191], 1);
        assert_eq!(pixel(&current_frame, 3, 5), [0, 0, 0, 0]);
        assert_ne!(current_frame.as_bytes(0), old_frame.as_bytes(0));
        assert!(document.owns_current_output(&current));
        assert!(!document.owns_current_output(&old));
        assert_eq!(current.revision(), 1);
        assert_eq!(current.configuration(), old.configuration());
        assert_eq!(current.output(), &current_output);
        assert_eq!(current.output().metadata, current_output.metadata);
        assert!(!Arc::ptr_eq(&old_lease, &current.output_lease()));
        assert!(Arc::ptr_eq(&old_lease, &old.output_lease()));
        assert_eq!(old.output(), &old_output);
        assert_eq!(old.output().metadata, old_output.metadata);
        assert_eq!(document.network(), &successor_network);
        let repeated = prepare_accepted_cpu_svg_preview_at_target(&current, &renderer, 19 * 13 * 4)
            .expect("repeat conversion of exact new accepted snapshot");
        assert_eq!(repeated.as_bytes(0), current_frame.as_bytes(0));
        assert!(document.owns_current_output(&current));
    }
    fn artboard_rgba_rows(preview: &crate::session::PreparedArtboardPreview) -> Vec<u8> {
        preview
            .image()
            .as_bytes(0)
            .expect("genuine single frame")
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[2], pixel[1], pixel[0], pixel[3]])
            .collect()
    }

    #[gpui::test]
    async fn at_r22_1_artboard_png_genuine_frame_pixels_and_profile(cx: &mut gpui::TestAppContext) {
        for scale in [1., 1.5, 2.] {
            let parameters = accepted_artboard_parameters();
            let (document, _) = accepted_artboard_document(31, 17, scale, parameters);
            let network = document.network().clone();
            let configuration = document.configuration();
            let session = cx.new(|_| crate::session::GraphiteArtboardSession::new(document));
            let preview = session
                .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
                .expect("genuine session request")
                .await
                .expect("genuine accepted and converted session preview");
            let output = preview.output().output().clone();
            let output_lease = preview.output().output_lease();
            let image = preview.image().clone();
            let rows = artboard_rgba_rows(&preview);
            assert_pixel_close(
                pixel(&image, (5. * scale) as usize, (5. * scale) as usize),
                [0, 0, 255, 128],
                1,
            );
            assert_eq!(pixel(&image, 0, 0), [0, 0, 0, 0]);
            assert_eq!(pixel(&image, 30, 16), [0, 0, 0, 0]);
            let prepared = prepare_artboard_png(&preview, rows.len(), 64 * 1024)
                .expect("one native frame PNG");
            let png = prepared.png();
            assert_eq!((png.width(), png.height()), (31, 17));
            let decoded =
                image::load_from_memory_with_format(png.encoded_bytes(), image::ImageFormat::Png)
                    .expect("actual encoded PNG")
                    .into_rgba8();
            assert_eq!(decoded.dimensions(), (31, 17));
            assert_eq!(decoded.as_raw(), &rows);
            let row_digest: [u8; 32] = Sha256::digest(&rows).into();
            let png_digest: [u8; 32] = Sha256::digest(png.encoded_bytes()).into();
            assert_eq!(prepared.pixels_sha256(), row_digest);
            assert_eq!(png.content_sha256(), png_digest);
            assert_ne!(row_digest, png_digest);
            assert_eq!(png.sample_format(), CpuSampleFormat::Rgba8);
            assert_eq!(png.color_space(), ImageColorSpace::Srgb);
            assert_eq!(png.alpha(), AlphaAssociation::Straight);
            assert_eq!(png.content_type(), "image/png");
            let direct = graphics_core::export::prepare_rgba8_png(31, 17, rows.clone(), 64 * 1024)
                .expect("shared original frontend encoder");
            assert_eq!(png.encoded_bytes(), direct.png().encoded_bytes());
            let repeated = prepare_artboard_png(&preview, rows.len(), 64 * 1024)
                .expect("same genuine immutable frame");
            assert_eq!(repeated.png().encoded_bytes(), png.encoded_bytes());
            assert!(Arc::ptr_eq(preview.image(), &image));
            assert!(Arc::ptr_eq(&preview.output().output_lease(), &output_lease));
            assert_eq!(preview.output().output(), &output);
            assert_eq!(preview.output().output().metadata, output.metadata);
            assert_eq!(output.metadata.backgrounds.len(), 1);
            assert_eq!(output.metadata.backgrounds[0].location, parameters.location);
            assert_eq!(
                output.metadata.backgrounds[0].dimensions,
                parameters.dimensions
            );
            assert!(
                output
                    .metadata
                    .clip_targets
                    .contains(&preview.output().node_id())
            );
            assert_eq!(preview.output().configuration(), configuration);
            assert_eq!(preview.output().document_id_encoding(), 91);
            assert_eq!(
                preview.output().node_id(),
                graphite_core_types::uuid::NodeId(7001)
            );
            assert_eq!(preview.output().revision(), 0);
            session.read_with(cx, |session, _| {
                assert_eq!(session.document().network(), &network);
                assert_eq!(session.parameters(), parameters);
                assert_eq!(session.configuration(), configuration);
                assert!(session.preview_is_current(&preview));
            });
        }
    }

    #[gpui::test]
    async fn at_r22_1_artboard_png_caps_recover_without_changing_preview(
        cx: &mut gpui::TestAppContext,
    ) {
        let (document, _) = accepted_artboard_document(31, 17, 1.5, accepted_artboard_parameters());
        let network = document.network().clone();
        let session = cx.new(|_| crate::session::GraphiteArtboardSession::new(document));
        let preview = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("tracked genuine request")
            .await
            .expect("genuine native preview");
        let output = preview.output().output().clone();
        let rows = artboard_rgba_rows(&preview);
        let prepared =
            prepare_artboard_png(&preview, rows.len(), 64 * 1024).expect("baseline immutable PNG");
        for maximum in [0, rows.len() - 1] {
            let error = prepare_artboard_png(&preview, maximum, 64 * 1024)
                .expect_err("pixel cap before RGBA copy");
            assert_eq!(
                error,
                format!(
                    "PNG rows require {} bytes, exceeding the {maximum} byte limit",
                    rows.len()
                )
            );
        }
        let encoded_size = prepared.png().encoded_bytes().len();
        for maximum in [0, encoded_size - 1] {
            let error = prepare_artboard_png(&preview, rows.len(), maximum)
                .expect_err("encoded result cap");
            assert_eq!(
                error,
                graphics_core::export::PngExportError::EncodedSizeExceeded {
                    actual: encoded_size,
                    limit: maximum,
                }
                .to_string()
            );
        }
        let recovered = prepare_artboard_png(&preview, rows.len(), encoded_size)
            .expect("exact pixel and PNG byte caps");
        assert_eq!(
            recovered.png().encoded_bytes(),
            prepared.png().encoded_bytes()
        );
        assert_eq!(recovered.pixels_sha256(), prepared.pixels_sha256());
        assert_eq!(artboard_rgba_rows(&preview), rows);
        assert_eq!(preview.output().output(), &output);
        assert_eq!(preview.output().output().metadata, output.metadata);
        session.read_with(cx, |session, _| {
            assert!(session.preview_is_current(&preview));
            assert_eq!(session.document().network(), &network);
            assert_eq!(session.revision(), 0);
            assert!(!session.is_pending());
            assert!(session.diagnostic().is_none());
        });
    }

    #[gpui::test]
    async fn at_r22_1_artboard_png_old_snapshot_after_genuine_edit_is_immutable(
        cx: &mut gpui::TestAppContext,
    ) {
        let original_parameters = accepted_artboard_parameters();
        let (document, _) = accepted_artboard_document(31, 17, 1., original_parameters);
        let original_network = document.network().clone();
        let configuration = document.configuration();
        let session = cx.new(|_| crate::session::GraphiteArtboardSession::new(document));
        let original = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("original request")
            .await
            .expect("original genuine receipt");
        let original_output = original.output().output().clone();
        let original_lease = original.output().output_lease();
        let original_rows = artboard_rgba_rows(&original);
        let original_png =
            prepare_artboard_png(&original, original_rows.len(), 64 * 1024).expect("original PNG");
        let changed_parameters = graphics_core::evaluate::CpuSvgArtboardParameters {
            background: Color::from_gamma_srgb_channels(0., 0., 1., 0.75),
            ..original_parameters
        };
        assert!(
            session
                .update(cx, |session, cx| {
                    session.apply(0, changed_parameters, configuration, cx)
                })
                .expect("genuine transactional document edit")
        );
        let edited_network =
            session.read_with(cx, |session, _| session.document().network().clone());
        assert_ne!(edited_network, original_network);
        let edited = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("edited request")
            .await
            .expect("edited genuine receipt");
        let edited_rows = artboard_rgba_rows(&edited);
        let edited_output = edited.output().output().clone();
        let edited_png = prepare_artboard_png(&edited, edited_rows.len(), 64 * 1024)
            .expect("edited native frame PNG");
        assert_pixel_close(pixel(edited.image(), 5, 5), [255, 0, 0, 191], 1);
        assert_eq!(pixel(edited.image(), 0, 0), [0, 0, 0, 0]);
        assert_ne!(edited_rows, original_rows);
        assert_ne!(edited_png.pixels_sha256(), original_png.pixels_sha256());
        assert_ne!(
            edited_png.png().content_sha256(),
            original_png.png().content_sha256()
        );
        assert_eq!(
            image::load_from_memory(edited_png.png().encoded_bytes())
                .expect("edited native PNG")
                .into_rgba8()
                .as_raw(),
            &edited_rows
        );
        session.read_with(cx, |session, _| {
            assert!(!session.preview_is_current(&original));
            assert!(session.preview_is_current(&edited));
            assert_eq!(session.document().network(), &edited_network);
            assert_eq!(session.parameters(), changed_parameters);
            assert_eq!(session.configuration(), configuration);
        });
        let old_again = prepare_artboard_png(&original, original_rows.len(), 64 * 1024)
            .expect("pure bake does not itself grant freshness");
        assert_eq!(
            old_again.png().encoded_bytes(),
            original_png.png().encoded_bytes()
        );
        assert_eq!(original.output().output(), &original_output);
        assert_eq!(
            original.output().output().metadata,
            original_output.metadata
        );
        assert!(Arc::ptr_eq(
            &original.output().output_lease(),
            &original_lease
        ));
        assert_eq!(artboard_rgba_rows(&original), original_rows);
        assert_eq!(
            edited.output().configuration(),
            original.output().configuration()
        );
        assert_eq!(edited.output().revision(), 1);
        assert!(
            session
                .update(cx, |session, cx| {
                    session.apply(1, original_parameters, configuration, cx)
                })
                .expect("restore authentic source parameters")
        );
        let restored = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("restored request")
            .await
            .expect("restored genuine receipt");
        let restored_png = prepare_artboard_png(&restored, original_rows.len(), 64 * 1024)
            .expect("restored frame PNG");
        assert_eq!(artboard_rgba_rows(&restored), original_rows);
        assert_eq!(
            restored_png.png().encoded_bytes(),
            original_png.png().encoded_bytes()
        );
        assert_eq!(restored_png.pixels_sha256(), original_png.pixels_sha256());
        assert_eq!(edited.output().output(), &edited_output);
        assert_eq!(edited.output().output().metadata, edited_output.metadata);
        session.read_with(cx, |session, _| {
            assert!(!session.preview_is_current(&edited));
            assert!(session.preview_is_current(&restored));
            assert_eq!(session.document().network(), &original_network);
            assert_eq!(session.parameters(), original_parameters);
            assert_eq!(session.revision(), 2);
        });
    }
}
