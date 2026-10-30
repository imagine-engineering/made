use std::{collections::HashMap, fmt, panic::AssertUnwindSafe, sync::Arc};

use graphite_core_types::list::{
    ATTR_ALIGN, ATTR_APPEARANCE, ATTR_BACKGROUND, ATTR_CAP, ATTR_CLIP, ATTR_CLIPPING_MASK,
    ATTR_DASH_OFFSET, ATTR_DASH_PATTERN, ATTR_DIMENSIONS, ATTR_GRADIENT_CYCLIC, ATTR_GRADIENT_FORM,
    ATTR_GRADIENT_HUE_DIRECTION, ATTR_GRADIENT_INTERPOLATION, ATTR_GRADIENT_SPACE,
    ATTR_GRADIENT_SPREAD, ATTR_JOIN, ATTR_JOIN_MITER_LIMIT, ATTR_LOCATION, ATTR_MIDPOINT,
    ATTR_PAINT, ATTR_POSITION, ATTR_WEIGHT, ItemAttributeValues,
};
use graphite_core_types::{
    ATTR_FONT, ATTR_FONT_SIZE, ATTR_LETTER_SPACING, ATTR_LETTER_TILT, ATTR_LINE_HEIGHT,
    ATTR_MAX_HEIGHT, ATTR_MAX_WIDTH, ATTR_TEXT_ALIGN, Color, FallibleVec2Operations,
    blending::BlendMode,
    list::{ATTR_BLEND_MODE, ATTR_OPACITY, ATTR_OPACITY_FILL, ATTR_TRANSFORM, Item, List},
    math::quad::Quad,
    transform::{Footprint, Transform as _},
};
use graphite_glam::{DAffine2, DMat2, DVec2};
use graphite_graphic_types::{Appearance, Artboard, Cover, Graphic};
use graphite_raster_types::{CPU, Raster};
use graphite_rendering::{Render as _, RenderOutputType, RenderParams, SvgRender, SvgRenderOutput};
use graphite_vector_types::kurbo::{Affine, ParamCurveArclen, PathEl};
use graphite_vector_types::vector::style::{
    DashPattern, RenderMode, Stroke, StrokeAlign, StrokeCap, StrokeJoin,
};
use graphite_vector_types::{
    Gradient, GradientForm, GradientHueDirection, GradientInterpolation, GradientSettings,
    GradientSpace, GradientSpread, Vector,
};

use graphite_skrifa::{
    GlyphId, MetadataProvider,
    instance::{LocationRef, NormalizedCoord, Size},
    outline::{DrawSettings, OutlinePen},
    raw::FontRef,
};
use graphite_std::{
    application_io::resource::{Resource, ResourceHash},
    text_nodes::{TextAlign, TextContext, TypesettingConfig, for_each_styled_glyph_run},
};

use crate::resources::{AdmittedCpuImage, CpuImageResult};

#[derive(Clone, Copy, Debug)]
pub struct CpuSvgSceneConfiguration {
    pub viewport: Footprint,
    pub scale: f64,
    pub max_svg_bytes: usize,
}

pub struct PreparedCpuSvgScene {
    source: CpuImageResult,
    configuration: CpuSvgSceneConfiguration,
    svg: SvgRenderOutput,
}

impl PreparedCpuSvgScene {
    pub fn svg(&self) -> &SvgRenderOutput {
        &self.svg
    }

    pub fn source(&self) -> &CpuImageResult {
        &self.source
    }

    pub fn configuration(&self) -> &CpuSvgSceneConfiguration {
        &self.configuration
    }
}

impl fmt::Debug for PreparedCpuSvgScene {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedCpuSvgScene")
            .field("source", &self.source)
            .field("configuration", &self.configuration)
            .field("svg_bytes", &self.svg.svg.len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuSvgSceneError {
    InvalidScale,
    EmptyViewport,
    InvalidViewportTransform,
    NonFiniteGeometry,
    UnsupportedAttribute(String),
    WrongAttributeType(String),
    InvalidImageTransform,
    NonFiniteOpacity(String),
    NonFiniteCombinedOpacity,
    NonFiniteMaskOpacity,
    UnsupportedBlendMode(BlendMode),
    UnexpectedImageCache,
    UnsupportedVectorAppearance,
    UnsupportedVectorPaint,
    UnsupportedStrokeStyle,
    NonFiniteStrokeParameter(String),
    NonFinitePaint,
    InvalidGradient,
    InvalidVectorGeometry,
    InvalidArtboardGeometry,
    InvalidTextFont,
    NonFiniteTextParameter(String),
    InvalidTextLayout,
    UnsupportedBranchingVector,
    SourceRenderingFailed,
    SvgBudgetExceeded { maximum: usize, actual: usize },
}

impl fmt::Display for CpuSvgSceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTextFont => formatter.write_str("Text requires valid nonempty font bytes with an authentic resource hash"),
            Self::NonFiniteTextParameter(parameter) => write!(formatter, "The {parameter} text parameter exceeds the source's finite representation"),
            Self::InvalidTextLayout => formatter.write_str("The original text layout or glyph geometry cannot be represented finitely"),
            Self::InvalidScale => {
                formatter.write_str("The SVG scene scale must be positive and finite")
            }
            Self::EmptyViewport => {
                formatter.write_str("The SVG viewport resolution must be nonzero")
            }
            Self::InvalidViewportTransform => {
                formatter.write_str("The SVG viewport transform must be finite")
            }
            Self::NonFiniteGeometry => {
                formatter.write_str("The SVG scene geometry exceeds finite coordinates")
            }
            Self::UnsupportedAttribute(attribute) => write!(
                formatter,
                "The CPU SVG profile does not support the {attribute} attribute"
            ),
            Self::WrongAttributeType(attribute) => write!(
                formatter,
                "The {attribute} attribute has the wrong source type"
            ),
            Self::InvalidImageTransform => {
                formatter.write_str("The image transform must contain finite components")
            }
            Self::NonFiniteOpacity(attribute) => {
                write!(formatter, "The {attribute} attribute must be finite")
            }
            Self::NonFiniteCombinedOpacity => formatter
                .write_str("The combined opacity cannot be represented by the original renderer"),
            Self::NonFiniteMaskOpacity => formatter
                .write_str("The mask opacity cannot be represented by the original renderer"),
            Self::UnsupportedBlendMode(mode) => write!(
                formatter,
                "The original SVG renderer does not support {mode}"
            ),
            Self::UnexpectedImageCache => formatter
                .write_str("A cached image payload cannot bypass the verified scene pixels"),
            Self::UnsupportedVectorAppearance => formatter
                .write_str("The CPU SVG vector profile requires one fill and at most one stroke coverage"),
            Self::UnsupportedVectorPaint => formatter
                .write_str("The CPU SVG vector profile requires an unadorned solid color or one typed gradient paint"),
            Self::UnsupportedStrokeStyle => formatter
                .write_str("The CPU SVG profile requires finite invertible stroke space and representable dash progress"),
            Self::NonFiniteStrokeParameter(attribute) => write!(
                formatter,
                "The {attribute} stroke parameter must be finite"
            ),
            Self::NonFinitePaint => formatter.write_str("The solid paint channels must be finite"),
            Self::InvalidGradient => formatter.write_str("The gradient has unsupported or nonfinite source samples or derived placement"),
            Self::InvalidVectorGeometry => {
                formatter.write_str("The vector geometry has invalid source columns or endpoints")
            }
            Self::InvalidArtboardGeometry => formatter.write_str(
                "The prepared artboard requires finite normalized location and dimensions of at least one",
            ),
            Self::UnsupportedBranchingVector => formatter
                .write_str("Branching vector meshes require a separately admitted scene profile"),
            Self::SourceRenderingFailed => {
                formatter.write_str("The original SVG renderer panicked")
            }
            Self::SvgBudgetExceeded { maximum, actual } => write!(
                formatter,
                "The prepared SVG has {actual} bytes, exceeding the {maximum}-byte output limit"
            ),
        }
    }
}

impl std::error::Error for CpuSvgSceneError {}

fn validate_item_attributes(item: &Item<Raster<CPU>>) -> Result<DAffine2, CpuSvgSceneError> {
    let mut transform = DAffine2::IDENTITY;
    let mut opacity = 1.;
    let mut fill_opacity = 1.;
    for key in item.attributes().keys() {
        match key {
            ATTR_TRANSFORM => {
                transform = *item
                    .attribute::<DAffine2>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if transform
                    .to_cols_array()
                    .iter()
                    .any(|component| !component.is_finite())
                {
                    return Err(CpuSvgSceneError::InvalidImageTransform);
                }
            }
            ATTR_BLEND_MODE => {
                let mode = *item
                    .attribute::<BlendMode>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if mode.to_svg_style_name().is_none() {
                    return Err(CpuSvgSceneError::UnsupportedBlendMode(mode));
                }
            }
            ATTR_OPACITY | ATTR_OPACITY_FILL => {
                let value = *item
                    .attribute::<f64>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if !value.is_finite() {
                    return Err(CpuSvgSceneError::NonFiniteOpacity(key.to_owned()));
                }
                if key == ATTR_OPACITY {
                    opacity = value;
                } else {
                    fill_opacity = value;
                }
            }
            _ => return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned())),
        }
    }
    let combined = opacity * fill_opacity;
    if !combined.is_finite() || !(combined as f32).is_finite() {
        return Err(CpuSvgSceneError::NonFiniteCombinedOpacity);
    }
    if item.element().data().base64_string.is_some() {
        return Err(CpuSvgSceneError::UnexpectedImageCache);
    }
    Ok(transform)
}

pub fn prepare_cpu_svg_scene(
    source: &CpuImageResult,
    configuration: CpuSvgSceneConfiguration,
) -> Result<PreparedCpuSvgScene, CpuSvgSceneError> {
    if !configuration.scale.is_finite() || configuration.scale <= 0. {
        return Err(CpuSvgSceneError::InvalidScale);
    }
    if configuration.viewport.resolution.x == 0 || configuration.viewport.resolution.y == 0 {
        return Err(CpuSvgSceneError::EmptyViewport);
    }
    if configuration
        .viewport
        .transform
        .to_cols_array()
        .iter()
        .any(|component| !component.is_finite())
    {
        return Err(CpuSvgSceneError::InvalidViewportTransform);
    }
    let transform = validate_item_attributes(source.original_item())?;
    let effective =
        DAffine2::from_scale(DVec2::splat(configuration.scale)) * configuration.viewport.transform;
    let logical = DAffine2::from_scale(DVec2::splat(1. / configuration.scale)) * effective;
    let resolution = configuration.viewport.resolution.as_dvec2() / configuration.scale;
    let viewport_zoom = configuration.viewport.scale_magnitudes().x;
    let combined = logical * transform;
    if effective
        .to_cols_array()
        .iter()
        .any(|component| !component.is_finite())
        || logical
            .to_cols_array()
            .iter()
            .any(|component| !component.is_finite())
        || combined
            .to_cols_array()
            .iter()
            .any(|component| !component.is_finite())
        || !resolution.is_finite()
        || resolution.x <= 0.
        || resolution.y <= 0.
        || !viewport_zoom.is_finite()
        || [DVec2::ZERO, DVec2::X, DVec2::Y, DVec2::ONE]
            .into_iter()
            .any(|point| !combined.transform_point2(point).is_finite())
    {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    let params = RenderParams {
        render_mode: RenderMode::Normal,
        footprint: Footprint {
            transform: effective,
            ..configuration.viewport
        },
        scale: configuration.scale,
        render_output_type: RenderOutputType::Svg,
        thumbnail: false,
        for_export: true,
        for_mask: false,
        alignment_parent_transform: None,
        aligned_strokes: false,
        stroke_below: false,
        inside_pattern: false,
        artboard_background: Some(Color::TRANSPARENT),
        viewport_zoom,
        inherited_appearance: None,
    };
    let svg = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut render = SvgRender::new();
        List::<Raster<CPU>>::from(source.original_item().clone()).render_svg(&mut render, &params);
        let intermediate = SvgRenderOutput::from(render);
        let mut render = SvgRender::from(&intermediate);
        render.wrap_with_transform(logical, Some(resolution));
        SvgRenderOutput::from(render)
    }))
    .map_err(|_| CpuSvgSceneError::SourceRenderingFailed)?;
    if svg.svg.len() > configuration.max_svg_bytes {
        return Err(CpuSvgSceneError::SvgBudgetExceeded {
            maximum: configuration.max_svg_bytes,
            actual: svg.svg.len(),
        });
    }
    Ok(PreparedCpuSvgScene {
        source: source.clone(),
        configuration,
        svg,
    })
}

#[derive(Clone, Debug)]
pub enum CpuSvgSceneLayer {
    Image(CpuImageResult),
    SourceImage(Arc<AdmittedCpuImage>),
    Vector(Item<Vector>),
    Text(Item<String>),
    TextList(List<String>),
    Color(Item<Color>),
    ColorList(List<Color>),
    Gradient(Item<Gradient>),
    GradientList(List<Gradient>),
}

pub struct PreparedCpuSvgComposition {
    layers: Vec<CpuSvgSceneLayer>,
    configuration: CpuSvgSceneConfiguration,
    svg: SvgRenderOutput,
}

impl PreparedCpuSvgComposition {
    pub fn layers(&self) -> &[CpuSvgSceneLayer] {
        &self.layers
    }
    pub fn configuration(&self) -> &CpuSvgSceneConfiguration {
        &self.configuration
    }
    pub fn svg(&self) -> &SvgRenderOutput {
        &self.svg
    }
    pub fn source_revision(&self) -> &'static str {
        crate::resources::GRAPHITE_IMAGE_SOURCE_REVISION
    }
}

impl fmt::Debug for PreparedCpuSvgComposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedCpuSvgComposition")
            .field("layers", &self.layers)
            .field("configuration", &self.configuration)
            .field("svg_bytes", &self.svg.svg.len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompositionAttributeProfile {
    Image,
    Vector,
    Text,
    Color,
    Gradient,
}

fn composition_attributes(
    attributes: &ItemAttributeValues,
    profile: CompositionAttributeProfile,
) -> Result<DAffine2, CpuSvgSceneError> {
    let mut transform = DAffine2::IDENTITY;
    let mut opacity = 1.;
    let mut fill_opacity = 1.;
    for key in attributes.keys() {
        match key {
            ATTR_TRANSFORM => {
                transform = *attributes
                    .get::<DAffine2>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if !transform.is_finite() {
                    return Err(CpuSvgSceneError::NonFiniteGeometry);
                }
            }
            ATTR_BLEND_MODE => {
                let mode = *attributes
                    .get::<BlendMode>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if mode.to_svg_style_name().is_none() {
                    return Err(CpuSvgSceneError::UnsupportedBlendMode(mode));
                }
            }
            ATTR_OPACITY | ATTR_OPACITY_FILL => {
                let value = *attributes
                    .get::<f64>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if !value.is_finite() {
                    return Err(CpuSvgSceneError::NonFiniteOpacity(key.to_owned()));
                }
                if key == ATTR_OPACITY {
                    opacity = value;
                } else {
                    fill_opacity = value;
                }
            }
            ATTR_CLIPPING_MASK => {
                attributes
                    .get::<bool>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
            }
            ATTR_APPEARANCE if profile == CompositionAttributeProfile::Vector => {
                let appearance = attributes
                    .get::<Appearance>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                validate_cpu_appearance(appearance)?;
            }
            ATTR_GRADIENT_FORM
            | ATTR_GRADIENT_SPREAD
            | ATTR_GRADIENT_CYCLIC
            | ATTR_GRADIENT_SPACE
            | ATTR_GRADIENT_HUE_DIRECTION
            | ATTR_GRADIENT_INTERPOLATION
                if profile == CompositionAttributeProfile::Gradient =>
            {
                let valid = match key {
                    ATTR_GRADIENT_FORM => attributes.get::<GradientForm>(key).is_some(),
                    ATTR_GRADIENT_SPREAD => attributes.get::<GradientSpread>(key).is_some(),
                    ATTR_GRADIENT_CYCLIC => attributes.get::<bool>(key).is_some(),
                    ATTR_GRADIENT_SPACE => attributes.get::<GradientSpace>(key).is_some(),
                    ATTR_GRADIENT_HUE_DIRECTION => {
                        attributes.get::<GradientHueDirection>(key).is_some()
                    }
                    ATTR_GRADIENT_INTERPOLATION => {
                        attributes.get::<GradientInterpolation>(key).is_some()
                    }
                    _ => false,
                };
                if !valid {
                    return Err(CpuSvgSceneError::WrongAttributeType(key.to_owned()));
                }
            }
            ATTR_FONT | ATTR_FONT_SIZE | ATTR_LINE_HEIGHT | ATTR_LETTER_SPACING
            | ATTR_LETTER_TILT | ATTR_MAX_WIDTH | ATTR_MAX_HEIGHT | ATTR_TEXT_ALIGN
                if profile == CompositionAttributeProfile::Text => {}
            _ => return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned())),
        }
    }
    if !(opacity * fill_opacity).is_finite() || !((opacity * fill_opacity) as f32).is_finite() {
        return Err(CpuSvgSceneError::NonFiniteCombinedOpacity);
    }
    if profile == CompositionAttributeProfile::Vector
        && attributes.get::<Appearance>(ATTR_APPEARANCE).is_none()
    {
        return Err(CpuSvgSceneError::UnsupportedVectorAppearance);
    }
    Ok(transform)
}

fn seal_font_resource(
    font: &Resource,
    fonts: &mut HashMap<ResourceHash, Resource>,
) -> Result<Resource, CpuSvgSceneError> {
    // Resource accepts arbitrary AsRef backing, so only one owned snapshot can bind validation, rendering and retained work to the same bytes.
    let bytes: Arc<[u8]> = Arc::from(font.as_ref());
    let sealed = Resource::new(bytes);
    if sealed.is_empty() || sealed.hash() != font.hash() {
        return Err(CpuSvgSceneError::InvalidTextFont);
    }
    Ok(fonts.entry(sealed.hash()).or_insert(sealed).clone())
}

fn seal_composition_layers(
    layers: &[CpuSvgSceneLayer],
) -> Result<Vec<CpuSvgSceneLayer>, CpuSvgSceneError> {
    let mut fonts = HashMap::new();
    let mut sealed = Vec::with_capacity(layers.len());
    for layer in layers {
        sealed.push(match layer {
            CpuSvgSceneLayer::Text(item) => {
                let font = item.attribute::<Resource>(ATTR_FONT).ok_or_else(|| {
                    if item.attributes().keys().any(|key| key == ATTR_FONT) {
                        CpuSvgSceneError::WrongAttributeType(ATTR_FONT.to_owned())
                    } else {
                        CpuSvgSceneError::InvalidTextFont
                    }
                })?;
                let font = seal_font_resource(font, &mut fonts)?;
                CpuSvgSceneLayer::Text(item.clone().with_attribute(ATTR_FONT, font))
            }
            CpuSvgSceneLayer::TextList(items) => {
                let mut items = items.clone();
                for index in 0..items.len() {
                    let font = items
                        .attribute::<Resource>(ATTR_FONT, index)
                        .ok_or_else(|| {
                            if items.attribute_keys().any(|key| key == ATTR_FONT) {
                                CpuSvgSceneError::WrongAttributeType(ATTR_FONT.to_owned())
                            } else {
                                CpuSvgSceneError::InvalidTextFont
                            }
                        })?;
                    let font = seal_font_resource(font, &mut fonts)?;
                    items.set_attribute(ATTR_FONT, index, font);
                }
                CpuSvgSceneLayer::TextList(items)
            }
            other => other.clone(),
        });
    }
    Ok(sealed)
}

fn validate_text_item(item: &Item<String>, logical: DAffine2) -> Result<(), CpuSvgSceneError> {
    let transform = composition_attributes(item.attributes(), CompositionAttributeProfile::Text)?;
    let final_transform = logical * transform;
    if !final_transform.is_finite() {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    let font = item.attribute::<Resource>(ATTR_FONT).ok_or_else(|| {
        if item
            .attributes()
            .keys()
            .any(|attribute| attribute == ATTR_FONT)
        {
            CpuSvgSceneError::WrongAttributeType(ATTR_FONT.to_owned())
        } else {
            CpuSvgSceneError::InvalidTextFont
        }
    })?;
    if font.is_empty() {
        return Err(CpuSvgSceneError::InvalidTextFont);
    }
    for key in [
        ATTR_FONT_SIZE,
        ATTR_LINE_HEIGHT,
        ATTR_LETTER_SPACING,
        ATTR_LETTER_TILT,
        ATTR_MAX_WIDTH,
        ATTR_MAX_HEIGHT,
    ] {
        if let Some(value) = item.attributes().get::<f64>(key) {
            let narrows = matches!(key, ATTR_FONT_SIZE | ATTR_LINE_HEIGHT | ATTR_LETTER_SPACING)
                || (matches!(key, ATTR_MAX_WIDTH | ATTR_MAX_HEIGHT) && *value > 0.);
            if !value.is_finite() || (narrows && !(*value as f32).is_finite()) {
                return Err(CpuSvgSceneError::NonFiniteTextParameter(key.to_owned()));
            }
        } else if item.attributes().keys().any(|attribute| attribute == key) {
            return Err(CpuSvgSceneError::WrongAttributeType(key.to_owned()));
        }
    }
    if item
        .attributes()
        .keys()
        .any(|attribute| attribute == ATTR_TEXT_ALIGN)
        && item.attribute::<TextAlign>(ATTR_TEXT_ALIGN).is_none()
    {
        return Err(CpuSvgSceneError::WrongAttributeType(
            ATTR_TEXT_ALIGN.to_owned(),
        ));
    }
    let typesetting = TypesettingConfig {
        font_size: item.attribute_cloned_or(ATTR_FONT_SIZE, 24.),
        line_height_ratio: item.attribute_cloned_or(ATTR_LINE_HEIGHT, 1.2),
        letter_spacing: item.attribute_cloned_or(ATTR_LETTER_SPACING, 0.),
        letter_tilt: item.attribute_cloned_or(ATTR_LETTER_TILT, 0.),
        max_width: item
            .attribute::<f64>(ATTR_MAX_WIDTH)
            .copied()
            .filter(|width| *width > 0.),
        max_height: item
            .attribute::<f64>(ATTR_MAX_HEIGHT)
            .copied()
            .filter(|height| *height > 0.),
        align: item.attribute_cloned_or_default(ATTR_TEXT_ALIGN),
    };
    if !((typesetting.font_size as f32) * (typesetting.line_height_ratio as f32)).is_finite() {
        return Err(CpuSvgSceneError::InvalidTextLayout);
    }
    let tilt = typesetting.letter_tilt.to_radians().tan();
    if !tilt.is_finite() {
        return Err(CpuSvgSceneError::InvalidTextLayout);
    }
    TextContext::with_thread_local(|context| {
        let layout = context
            .layout_text(item.element(), font, typesetting)
            .ok_or(CpuSvgSceneError::InvalidTextFont)?;
        if [
            layout.width(),
            layout.full_width(),
            layout.height(),
            layout.scale(),
        ]
        .into_iter()
        .any(|value| !value.is_finite())
        {
            return Err(CpuSvgSceneError::InvalidTextLayout);
        }
        for line in layout.lines() {
            let metrics = line.metrics();
            if [
                metrics.ascent,
                metrics.descent,
                metrics.leading,
                metrics.line_height,
                metrics.baseline,
                metrics.offset,
                metrics.advance,
                metrics.trailing_whitespace,
                metrics.inline_min_coord,
                metrics.inline_max_coord,
                metrics.block_min_coord,
                metrics.block_max_coord,
            ]
            .into_iter()
            .any(|value| !value.is_finite())
            {
                return Err(CpuSvgSceneError::InvalidTextLayout);
            }
        }
        let mut failure = None;
        for_each_styled_glyph_run(
            &layout,
            item.element(),
            typesetting,
            |glyph_run, offset, spacing| {
                if failure.is_some() {
                    return;
                }
                let result = (|| {
                    let run = glyph_run.run();
                    let mut horizontal = glyph_run.offset() + offset;
                    let baseline = glyph_run.baseline();
                    let size = run.font_size();
                    if [horizontal, baseline, size, offset, spacing]
                        .into_iter()
                        .any(|value| !value.is_finite())
                    {
                        return Err(CpuSvgSceneError::InvalidTextLayout);
                    }
                    let font = run.font();
                    let font = FontRef::from_index(font.data.as_ref(), font.index)
                        .map_err(|_| CpuSvgSceneError::InvalidTextFont)?;
                    let outlines = font.outline_glyphs();
                    let coordinates: Vec<NormalizedCoord> = run
                        .normalized_coords()
                        .iter()
                        .map(|coordinate| NormalizedCoord::from_bits(*coordinate))
                        .collect();
                    for glyph in glyph_run.glyphs() {
                        let horizontal_origin = horizontal + glyph.x;
                        let vertical_origin = baseline - glyph.y;
                        horizontal += glyph.advance;
                        if [
                            horizontal_origin,
                            vertical_origin,
                            horizontal,
                            glyph.advance,
                        ]
                        .into_iter()
                        .any(|value| !value.is_finite())
                        {
                            return Err(CpuSvgSceneError::InvalidTextLayout);
                        }
                        let Some(outline) = outlines.get(GlyphId::from(glyph.id)) else {
                            continue;
                        };
                        let mut pen = TextGeometryAdmission {
                            origin: DVec2::new(horizontal_origin as f64, vertical_origin as f64),
                            tilt,
                            transform,
                            final_transform,
                            valid: true,
                            bounds: None,
                            has_geometry: false,
                        };
                        // The source SVG pen is private, so this only checks its exact coordinate equations; the original renderer still emits every glyph.
                        outline
                            .draw(
                                DrawSettings::unhinted(
                                    Size::new(size),
                                    LocationRef::new(&coordinates),
                                ),
                                &mut pen,
                            )
                            .map_err(|_| CpuSvgSceneError::InvalidTextFont)?;
                        if !pen.valid {
                            return Err(CpuSvgSceneError::InvalidTextLayout);
                        }
                        if !pen.has_geometry && spacing != 0. && glyph.advance > 0. {
                            horizontal += spacing;
                            if !horizontal.is_finite() {
                                return Err(CpuSvgSceneError::InvalidTextLayout);
                            }
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    failure = Some(error);
                }
            },
        );
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })
}

struct TextGeometryAdmission {
    origin: DVec2,
    tilt: f64,
    transform: DAffine2,
    final_transform: DAffine2,
    valid: bool,
    bounds: Option<[DVec2; 2]>,
    has_geometry: bool,
}

impl TextGeometryAdmission {
    fn point(&mut self, horizontal: f32, vertical: f32) {
        self.has_geometry = true;
        let point = DVec2::new(
            self.origin.x + horizontal as f64 + vertical as f64 * self.tilt,
            self.origin.y - vertical as f64,
        );
        self.valid &= point.is_finite()
            && self.transform.transform_point2(point).is_finite()
            && self.final_transform.transform_point2(point).is_finite();
        let bounds = self.bounds.get_or_insert([point, point]);
        bounds[0] = bounds[0].min(point);
        bounds[1] = bounds[1].max(point);
        self.valid &= (bounds[1] - bounds[0]).is_finite();
    }
}

impl OutlinePen for TextGeometryAdmission {
    fn move_to(&mut self, horizontal: f32, vertical: f32) {
        self.point(horizontal, vertical);
    }
    fn line_to(&mut self, horizontal: f32, vertical: f32) {
        self.point(horizontal, vertical);
    }
    fn quad_to(
        &mut self,
        control_horizontal: f32,
        control_vertical: f32,
        horizontal: f32,
        vertical: f32,
    ) {
        self.point(control_horizontal, control_vertical);
        self.point(horizontal, vertical);
    }
    fn curve_to(
        &mut self,
        first_horizontal: f32,
        first_vertical: f32,
        second_horizontal: f32,
        second_vertical: f32,
        horizontal: f32,
        vertical: f32,
    ) {
        self.point(first_horizontal, first_vertical);
        self.point(second_horizontal, second_vertical);
        self.point(horizontal, vertical);
    }
    fn close(&mut self) {
        self.has_geometry = true;
    }
}

fn validate_color_item(item: &Item<Color>, logical: DAffine2) -> Result<(), CpuSvgSceneError> {
    composition_attributes(item.attributes(), CompositionAttributeProfile::Color)?;
    validate_paint_color(*item.element())?;
    if item.element().a() < 1. && !(item.element().a() * 1000.).is_finite() {
        return Err(CpuSvgSceneError::NonFinitePaint);
    }
    // Color is nonspatial in the source: only the outer viewport transforms its fixed infinite-background stand-in.
    for point in [
        DVec2::new(1e7, 1e7),
        DVec2::new(-1e7, 1e7),
        DVec2::new(-1e7, -1e7),
        DVec2::new(1e7, -1e7),
    ] {
        if !logical.transform_point2(point).is_finite() {
            return Err(CpuSvgSceneError::NonFiniteGeometry);
        }
    }
    Ok(())
}

fn validate_cpu_appearance(appearance: &Appearance) -> Result<(), CpuSvgSceneError> {
    if appearance.is_empty() || appearance.len() > 2 {
        return Err(CpuSvgSceneError::UnsupportedVectorAppearance);
    }
    for key in appearance.0.attribute_keys() {
        if key != ATTR_PAINT {
            return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned()));
        }
    }
    let mut fill_seen = false;
    let mut stroke_seen = false;
    for index in 0..appearance.len() {
        let coverage = appearance
            .cover_at(index)
            .ok_or(CpuSvgSceneError::UnsupportedVectorAppearance)?;
        match coverage.cover() {
            Cover::Fill if !fill_seen => {
                if coverage.0.attributes().keys().next().is_some() {
                    return Err(CpuSvgSceneError::UnsupportedVectorAppearance);
                }
                fill_seen = true;
            }
            Cover::Stroke if !stroke_seen => {
                validate_solid_stroke(coverage)?;
                stroke_seen = true;
            }
            _ => return Err(CpuSvgSceneError::UnsupportedVectorAppearance),
        }
        let paint = appearance
            .paint_at(index)
            .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(ATTR_PAINT.to_owned()))?;
        validate_cpu_paint(paint)?;
    }
    if !fill_seen {
        return Err(CpuSvgSceneError::UnsupportedVectorAppearance);
    }
    Ok(())
}

fn validate_cpu_paint(paint: &Graphic) -> Result<(), CpuSvgSceneError> {
    let color = match paint {
        Graphic::Gradient(item) => return validate_gradient_item(item, false),
        Graphic::GradientList(list) if list.len() <= 1 => {
            if let Some(item) = list.clone().into_iter().next() {
                return validate_gradient_item(&item, false);
            }
            return if list.attribute_keys().next().is_none() {
                Ok(())
            } else {
                Err(CpuSvgSceneError::UnsupportedVectorPaint)
            };
        }
        Graphic::Color(color) if color.attributes().keys().next().is_none() => color.element(),
        Graphic::ColorList(colors)
            if colors.len() == 1 && colors.attribute_keys().next().is_none() =>
        {
            colors
                .element(0)
                .ok_or(CpuSvgSceneError::UnsupportedVectorPaint)?
        }
        _ => return Err(CpuSvgSceneError::UnsupportedVectorPaint),
    };
    validate_paint_color(*color)
}

fn validate_paint_color(color: Color) -> Result<(), CpuSvgSceneError> {
    if [color.r(), color.g(), color.b(), color.a()]
        .iter()
        .any(|value| !value.is_finite())
        || color
            .to_gamma_srgb_channels()
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(CpuSvgSceneError::NonFinitePaint);
    }
    Ok(())
}

fn validate_gradient_item(item: &Item<Gradient>, for_mask: bool) -> Result<(), CpuSvgSceneError> {
    for key in item.attributes().keys() {
        let correct_type = match key {
            ATTR_TRANSFORM => item.attribute::<DAffine2>(key).is_some(),
            ATTR_OPACITY | ATTR_OPACITY_FILL => item.attribute::<f64>(key).is_some(),
            ATTR_GRADIENT_FORM => item.attribute::<GradientForm>(key).is_some(),
            ATTR_GRADIENT_SPREAD => item.attribute::<GradientSpread>(key).is_some(),
            ATTR_GRADIENT_CYCLIC => item.attribute::<bool>(key).is_some(),
            ATTR_GRADIENT_SPACE => item.attribute::<GradientSpace>(key).is_some(),
            ATTR_GRADIENT_HUE_DIRECTION => item.attribute::<GradientHueDirection>(key).is_some(),
            ATTR_GRADIENT_INTERPOLATION => item.attribute::<GradientInterpolation>(key).is_some(),
            _ => return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned())),
        };
        if !correct_type {
            return Err(CpuSvgSceneError::WrongAttributeType(key.to_owned()));
        }
    }
    let local: DAffine2 = item.attribute_cloned_or_default(ATTR_TRANSFORM);
    let opacity = item.attribute_cloned_or::<f64>(ATTR_OPACITY, 1.);
    let fill_opacity = item.attribute_cloned_or::<f64>(ATTR_OPACITY_FILL, 1.);
    let paint_opacity = opacity * if for_mask { 1. } else { fill_opacity };
    if !local.is_finite()
        || !opacity.is_finite()
        || !fill_opacity.is_finite()
        || !paint_opacity.is_finite()
        || !(paint_opacity as f32).is_finite()
    {
        return Err(CpuSvgSceneError::InvalidGradient);
    }
    let samples = validate_gradient_samples(item)?;
    let multiplier = if paint_opacity < 1. {
        paint_opacity as f32
    } else {
        1.
    };
    for (position, color, midpoint) in &samples {
        validate_paint_color(*color)?;
        let alpha = color.a() * multiplier;
        if !position.is_finite()
            || !(position * 1_000_000.).is_finite()
            || midpoint.is_some_and(|value| !value.is_finite() || !(value * 1000.).is_finite())
            || !alpha.is_finite()
            || (alpha < 1. && !(alpha * 1000.).is_finite())
        {
            return Err(CpuSvgSceneError::InvalidGradient);
        }
    }
    if samples.is_empty() && multiplier < 1. && !(multiplier * 1000.).is_finite() {
        return Err(CpuSvgSceneError::InvalidGradient);
    }
    Ok(())
}

type GradientSamples = Vec<(f64, Color, Option<f64>)>;

fn validate_gradient_samples(item: &Item<Gradient>) -> Result<GradientSamples, CpuSvgSceneError> {
    let gradient = item.element();
    for key in gradient.0.attribute_keys() {
        if !matches!(key, ATTR_POSITION | ATTR_MIDPOINT) {
            return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned()));
        }
        let values = gradient
            .0
            .iter_attribute_values::<f64>(key)
            .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
        if values.len() != gradient.0.len() || values.clone().any(|value| !value.is_finite()) {
            return Err(CpuSvgSceneError::InvalidGradient);
        }
    }
    for color in gradient.0.iter_element_values() {
        validate_paint_color(*color)?;
    }
    let samples = gradient.interpolated_samples(GradientSettings::from_item_attributes(item));
    for (position, color, midpoint) in &samples {
        validate_paint_color(*color)?;
        if !position.is_finite() || midpoint.is_some_and(|value| !value.is_finite()) {
            return Err(CpuSvgSceneError::InvalidGradient);
        }
    }
    Ok(samples)
}

fn validate_standalone_gradient(
    item: &Item<Gradient>,
    physical: DAffine2,
    logical: DAffine2,
) -> Result<(), CpuSvgSceneError> {
    let local = composition_attributes(item.attributes(), CompositionAttributeProfile::Gradient)?;
    let gradient_transform = physical * local;
    let final_transform = logical * gradient_transform;
    if !gradient_transform.is_finite()
        || !final_transform.is_finite()
        || [DVec2::ZERO, DVec2::X, DVec2::Y]
            .into_iter()
            .any(|point| !final_transform.transform_point2(point).is_finite())
        || [
            DVec2::splat(1e7),
            DVec2::splat(-1e7),
            DVec2::new(1e7, -1e7),
            DVec2::new(-1e7, 1e7),
        ]
        .into_iter()
        .any(|point| !logical.transform_point2(point).is_finite())
    {
        return Err(CpuSvgSceneError::InvalidGradient);
    }
    if item.attribute_cloned_or_default::<GradientForm>(ATTR_GRADIENT_FORM) == GradientForm::Radial
    {
        // The original radial unit circle includes negative and sheared extrema. Its bounding square is a conservative finite subset, not an alternate ellipse renderer.
        let corners = [
            DVec2::ONE,
            -DVec2::ONE,
            DVec2::new(1., -1.),
            DVec2::new(-1., 1.),
        ];
        if corners.into_iter().any(|point| {
            !gradient_transform.transform_point2(point).is_finite()
                || !final_transform.transform_point2(point).is_finite()
        }) {
            return Err(CpuSvgSceneError::InvalidGradient);
        }
    }
    // Standalone SVG writes sampled offsets and alpha directly, separate from element opacity.
    for (_, _, midpoint) in validate_gradient_samples(item)? {
        if midpoint.is_some_and(|value| !(value * 1000.).is_finite()) {
            return Err(CpuSvgSceneError::InvalidGradient);
        }
    }
    Ok(())
}

fn validate_gradient_list_attributes(items: &List<Gradient>) -> Result<(), CpuSvgSceneError> {
    for key in items.attribute_keys() {
        let valid = match key {
            ATTR_TRANSFORM => items.iter_attribute_values::<DAffine2>(key).is_some(),
            ATTR_OPACITY | ATTR_OPACITY_FILL => items.iter_attribute_values::<f64>(key).is_some(),
            ATTR_BLEND_MODE => items.iter_attribute_values::<BlendMode>(key).is_some(),
            ATTR_CLIPPING_MASK | ATTR_GRADIENT_CYCLIC => {
                items.iter_attribute_values::<bool>(key).is_some()
            }
            ATTR_GRADIENT_FORM => items.iter_attribute_values::<GradientForm>(key).is_some(),
            ATTR_GRADIENT_SPREAD => items.iter_attribute_values::<GradientSpread>(key).is_some(),
            ATTR_GRADIENT_SPACE => items.iter_attribute_values::<GradientSpace>(key).is_some(),
            ATTR_GRADIENT_HUE_DIRECTION => items
                .iter_attribute_values::<GradientHueDirection>(key)
                .is_some(),
            ATTR_GRADIENT_INTERPOLATION => items
                .iter_attribute_values::<GradientInterpolation>(key)
                .is_some(),
            _ => return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned())),
        };
        if !valid {
            return Err(CpuSvgSceneError::WrongAttributeType(key.to_owned()));
        }
    }
    Ok(())
}

fn gradient_paint_item(paint: &Graphic) -> Option<Item<Gradient>> {
    match paint {
        Graphic::Gradient(item) => Some(item.clone()),
        Graphic::GradientList(list) if list.len() == 1 => list.clone().into_iter().next(),
        _ => None,
    }
}

fn validate_gradient_placements(
    appearance: &Appearance,
    item_transform: DAffine2,
    logical: DAffine2,
) -> Result<(), CpuSvgSceneError> {
    if !(0..appearance.len()).any(|index| matches!(appearance.paint_at(index), Some(Graphic::Gradient(_))) || matches!(appearance.paint_at(index), Some(Graphic::GradientList(list)) if list.len() == 1)) {
        return Ok(());
    }
    let stroke_transform = appearance
        .first_coverage_of(Cover::Stroke)
        .map(|coverage| coverage.stroke_params())
        .filter(|stroke| stroke.weight() > 0.)
        .map(|stroke| stroke.transform);
    let element_transform = stroke_transform.map_or(DAffine2::IDENTITY, |stroke| {
        item_transform * stroke.inverse()
    });
    let element_inverse = if element_transform.matrix2.determinant().recip().is_finite() {
        element_transform.inverse()
    } else {
        DAffine2::IDENTITY
    };
    if !element_transform.is_finite() || !element_inverse.is_finite() {
        return Err(CpuSvgSceneError::InvalidGradient);
    }
    for index in 0..appearance.len() {
        let paint = appearance
            .paint_at(index)
            .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(ATTR_PAINT.into()))?;
        let Some(gradient) = gradient_paint_item(paint) else {
            continue;
        };
        let local: DAffine2 = gradient.attribute_cloned_or_default(ATTR_TRANSFORM);
        let document_transform = item_transform * local;
        let placement =
            match gradient.attribute_cloned_or_default::<GradientForm>(ATTR_GRADIENT_FORM) {
                GradientForm::Radial => document_transform,
                GradientForm::Linear => {
                    let axis = document_transform.matrix2.x_axis;
                    let normal = document_transform.matrix2.y_axis.perp();
                    let line = axis.try_project_onto(normal).unwrap_or(axis);
                    DAffine2 {
                        matrix2: DMat2::from_cols(line, line.perp()),
                        translation: document_transform.translation,
                    }
                }
            };
        let paint_transform = element_inverse * placement;
        let effective = logical * element_transform * paint_transform;
        // These mirror the original private placement equations only for finite admission; the source renderer still emits every paint.
        if !document_transform.is_finite()
            || !placement.is_finite()
            || !paint_transform.is_finite()
            || !effective.is_finite()
            || !(logical * document_transform).is_finite()
            || [
                DVec2::NEG_ONE,
                DVec2::new(-1., 1.),
                DVec2::new(1., -1.),
                DVec2::ONE,
                DVec2::ZERO,
            ]
            .into_iter()
            .any(|point| !effective.transform_point2(point).is_finite())
        {
            return Err(CpuSvgSceneError::InvalidGradient);
        }
    }
    Ok(())
}

fn validate_solid_stroke(
    coverage: &graphite_graphic_types::Coverage,
) -> Result<(), CpuSvgSceneError> {
    let attributes = coverage.0.attributes();
    for key in attributes.keys() {
        match key {
            ATTR_WEIGHT | ATTR_DASH_OFFSET | ATTR_JOIN_MITER_LIMIT => {
                let value = attributes
                    .get::<f64>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if !value.is_finite() {
                    return Err(CpuSvgSceneError::NonFiniteStrokeParameter(key.to_owned()));
                }
            }
            ATTR_CAP => {
                attributes
                    .get::<StrokeCap>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
            }
            ATTR_JOIN => {
                attributes
                    .get::<StrokeJoin>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
            }
            ATTR_ALIGN => {
                attributes
                    .get::<StrokeAlign>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
            }
            ATTR_DASH_PATTERN => {
                let pattern = attributes
                    .get::<DashPattern>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if pattern.0.attribute_keys().next().is_some() {
                    return Err(CpuSvgSceneError::UnsupportedStrokeStyle);
                }
                if pattern
                    .0
                    .iter_element_values()
                    .any(|length| !length.is_finite())
                {
                    return Err(CpuSvgSceneError::NonFiniteStrokeParameter(key.to_owned()));
                }
                if coverage.stroke_params().has_renderable_stroke() {
                    let mut period: f64 = pattern
                        .0
                        .iter_element_values()
                        .map(|length| length.max(0.))
                        .sum();
                    if pattern.0.len() % 2 == 1 {
                        period *= 2.;
                    }
                    // An infinite period can make kurbo's negative-offset initialization loop forever, even with finite authored lengths.
                    if !period.is_finite() {
                        return Err(CpuSvgSceneError::NonFiniteStrokeParameter(key.to_owned()));
                    }
                }
            }
            ATTR_TRANSFORM => {
                let transform = attributes
                    .get::<DAffine2>(key)
                    .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(key.to_owned()))?;
                if !transform.is_finite() {
                    return Err(CpuSvgSceneError::NonFiniteGeometry);
                }
            }
            _ => return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned())),
        }
    }
    Ok(())
}

fn composition_configuration(
    configuration: CpuSvgSceneConfiguration,
) -> Result<(RenderParams, DAffine2, DVec2), CpuSvgSceneError> {
    if !configuration.scale.is_finite() || configuration.scale <= 0. {
        return Err(CpuSvgSceneError::InvalidScale);
    }
    if configuration.viewport.resolution.x == 0 || configuration.viewport.resolution.y == 0 {
        return Err(CpuSvgSceneError::EmptyViewport);
    }
    if !configuration.viewport.transform.is_finite() {
        return Err(CpuSvgSceneError::InvalidViewportTransform);
    }
    let effective =
        DAffine2::from_scale(DVec2::splat(configuration.scale)) * configuration.viewport.transform;
    let logical = DAffine2::from_scale(DVec2::splat(1. / configuration.scale)) * effective;
    let resolution = configuration.viewport.resolution.as_dvec2() / configuration.scale;
    let viewport_zoom = configuration.viewport.scale_magnitudes().x;
    if !effective.is_finite()
        || !logical.is_finite()
        || !resolution.is_finite()
        || resolution.x <= 0.
        || resolution.y <= 0.
        || !viewport_zoom.is_finite()
    {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    Ok((
        RenderParams {
            render_mode: RenderMode::Normal,
            footprint: Footprint {
                transform: effective,
                ..configuration.viewport
            },
            scale: configuration.scale,
            render_output_type: RenderOutputType::Svg,
            thumbnail: false,
            for_export: true,
            for_mask: false,
            alignment_parent_transform: None,
            aligned_strokes: false,
            stroke_below: false,
            inside_pattern: false,
            artboard_background: Some(Color::TRANSPARENT),
            viewport_zoom,
            inherited_appearance: None,
        },
        logical,
        resolution,
    ))
}

fn validate_vector_geometry(
    vector: &Vector,
    transform: DAffine2,
    logical: DAffine2,
) -> Result<(), CpuSvgSceneError> {
    let points = vector.point_domain.positions();
    let segments = &vector.segment_domain;
    let count = segments.ids().len();
    if vector.point_domain.ids().len() != points.len()
        || segments.start_point().len() != count
        || segments.end_point().len() != count
        || segments.handles().len() != count
        || segments
            .start_point()
            .iter()
            .chain(segments.end_point())
            .any(|&index| index >= points.len())
    {
        return Err(CpuSvgSceneError::InvalidVectorGeometry);
    }
    if points.iter().any(|point| !point.is_finite())
        || segments
            .handles()
            .iter()
            .any(|handles| !handles.is_finite())
    {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    if vector.is_branching() {
        return Err(CpuSvgSceneError::UnsupportedBranchingVector);
    }
    let combined = logical * transform;
    if !combined.is_finite() {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    for affine in [DAffine2::IDENTITY, transform, combined] {
        if points
            .iter()
            .copied()
            .chain(
                segments
                    .handles()
                    .iter()
                    .flat_map(|handles| handles.start().into_iter().chain(handles.end())),
            )
            .any(|point| !affine.transform_point2(point).is_finite())
        {
            return Err(CpuSvgSceneError::NonFiniteGeometry);
        }
        if vector
            .bounding_box_with_transform(affine)
            .is_some_and(|bounds| {
                !bounds.iter().all(|point| point.is_finite())
                    || !(bounds[1] - bounds[0]).is_finite()
            })
        {
            return Err(CpuSvgSceneError::NonFiniteGeometry);
        }
    }
    Ok(())
}

fn validate_vector_stroke_geometry(
    vector: &Vector,
    item_transform: DAffine2,
    logical: DAffine2,
    stroke: &Stroke,
    appearance: &Appearance,
) -> Result<(), CpuSvgSceneError> {
    if !stroke.has_renderable_stroke() {
        return Ok(());
    }
    let determinant = stroke.transform.matrix2.determinant();
    if !determinant.is_finite() {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    if !determinant.recip().is_finite() {
        return Err(CpuSvgSceneError::UnsupportedStrokeStyle);
    }
    let inverse = stroke.transform.inverse();
    if !inverse.is_finite() {
        return Err(CpuSvgSceneError::UnsupportedStrokeStyle);
    }
    let element_transform = item_transform * inverse;
    let final_transform = logical * element_transform;
    if !element_transform.is_finite() || !final_transform.is_finite() {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    validate_vector_geometry(vector, stroke.transform, final_transform)?;
    let largest_dash = stroke.dash_lengths.iter().copied().fold(0., f64::max);
    if largest_dash > 0. {
        for mut path in vector.stroke_bezpath_iter() {
            path.apply_affine(Affine::new(stroke.transform.to_cols_array()));
            for segment in path.segments() {
                let length = segment.arclen(1e-6);
                if !length.is_finite() {
                    return Err(CpuSvgSceneError::NonFiniteGeometry);
                }
                // A cycle must contain a dash that can reduce the source's remaining arc length; smaller entries may legitimately act like zero-length dashes.
                if length > 0. && length - largest_dash == length {
                    return Err(CpuSvgSceneError::UnsupportedStrokeStyle);
                }
            }
        }
    }
    let path_is_closed = vector
        .stroke_bezpath_iter()
        .all(|path| matches!(path.elements().last(), Some(PathEl::ClosePath)));
    let aligned = path_is_closed && stroke.align != StrokeAlign::Center;
    if (aligned && !(stroke.weight * 2.).is_finite())
        || !stroke.max_aabb_inflation(path_is_closed).is_finite()
    {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    let paints = appearance.fill_and_stroke();
    let visible = paints
        .stroke_paint
        .is_some_and(|paint| !paint.is_guaranteed_fully_transparent());
    let separate_alignment = aligned
        && visible
        && (stroke.align == StrokeAlign::Inside
            || !paints
                .fill_paint
                .is_some_and(Graphic::is_guaranteed_to_cover_opaquely));
    if separate_alignment {
        validate_alignment_mask_geometry(vector, stroke, element_transform, logical)?;
    }
    if let Some(bounds) =
        vector.stroke_inclusive_bounding_box_with_transform(DAffine2::IDENTITY, Some(stroke))
    {
        if !bounds.iter().all(|point| point.is_finite()) || !(bounds[1] - bounds[0]).is_finite() {
            return Err(CpuSvgSceneError::NonFiniteGeometry);
        }
        for point in [
            bounds[0],
            DVec2::new(bounds[0].x, bounds[1].y),
            DVec2::new(bounds[1].x, bounds[0].y),
            bounds[1],
        ] {
            if !item_transform.transform_point2(point).is_finite()
                || !(logical * item_transform)
                    .transform_point2(point)
                    .is_finite()
            {
                return Err(CpuSvgSceneError::NonFiniteGeometry);
            }
        }
    }
    Ok(())
}

fn validate_alignment_mask_geometry(
    vector: &Vector,
    stroke: &Stroke,
    element_transform: DAffine2,
    logical: DAffine2,
) -> Result<(), CpuSvgSceneError> {
    let matrix = stroke.transform.matrix2;
    let [first, second, third, fourth] = [
        matrix.x_axis.x,
        matrix.x_axis.y,
        matrix.y_axis.x,
        matrix.y_axis.y,
    ];
    // The source's private singular_values calculation also runs for Inside clipping; admission checks its intermediates without replacing rendering.
    let trace = first * first + second * second + third * third + fourth * fourth;
    let determinant = first * fourth - second * third;
    let discriminant_squared = trace * trace - 4. * determinant * determinant;
    let discriminant = discriminant_squared.max(0.).sqrt();
    let largest = ((trace + discriminant) * 0.5).sqrt();
    let inflation = stroke.max_aabb_inflation(true) * largest;
    if !trace.is_finite()
        || !discriminant_squared.is_finite()
        || !largest.is_finite()
        || !inflation.is_finite()
    {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    let bounds = vector
        .bounding_box_with_transform(stroke.transform)
        .unwrap_or_default();
    let quad = Quad::from_box(bounds).inflate(inflation);
    let extent = quad.bottom_right() - quad.top_left();
    let final_transform = logical * element_transform;
    if !extent.is_finite()
        || quad.0.into_iter().any(|point| {
            !point.is_finite()
                || !element_transform.transform_point2(point).is_finite()
                || !final_transform.transform_point2(point).is_finite()
        })
    {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    Ok(())
}

fn validate_composition_mask_opacities(
    layers: &[CpuSvgSceneLayer],
    graphics: &List<Graphic>,
) -> Result<(), CpuSvgSceneError> {
    let mut siblings = layers.iter().zip(graphics.iter_element_values()).peekable();
    let mut mask_active = false;
    while let Some((layer, _)) = siblings.next() {
        let next_clips = siblings
            .peek()
            .is_some_and(|(_, graphic)| graphic.had_clip_enabled());
        if next_clips && !mask_active {
            let opacity = match layer {
                CpuSvgSceneLayer::Image(image) => {
                    image.original_item().attribute::<f64>(ATTR_OPACITY)
                }
                CpuSvgSceneLayer::SourceImage(_) => None,
                CpuSvgSceneLayer::Vector(vector) => vector.attribute::<f64>(ATTR_OPACITY),
                CpuSvgSceneLayer::Text(text) => text.attribute::<f64>(ATTR_OPACITY),
                CpuSvgSceneLayer::Color(color) => color.attribute::<f64>(ATTR_OPACITY),
                CpuSvgSceneLayer::Gradient(gradient) => gradient.attribute::<f64>(ATTR_OPACITY),
                CpuSvgSceneLayer::TextList(texts) => {
                    for index in 0..texts.len() {
                        if !(texts.attribute_cloned_or::<f64>(ATTR_OPACITY, index, 1.) as f32)
                            .is_finite()
                        {
                            return Err(CpuSvgSceneError::NonFiniteMaskOpacity);
                        }
                    }
                    None
                }
                CpuSvgSceneLayer::GradientList(gradients) => {
                    for index in 0..gradients.len() {
                        if !(gradients.attribute_cloned_or::<f64>(ATTR_OPACITY, index, 1.) as f32)
                            .is_finite()
                        {
                            return Err(CpuSvgSceneError::NonFiniteMaskOpacity);
                        }
                    }
                    None
                }
                CpuSvgSceneLayer::ColorList(colors) => {
                    for index in 0..colors.len() {
                        if !(colors.attribute_cloned_or::<f64>(ATTR_OPACITY, index, 1.) as f32)
                            .is_finite()
                        {
                            return Err(CpuSvgSceneError::NonFiniteMaskOpacity);
                        }
                    }
                    None
                }
            }
            .copied()
            .unwrap_or(1.);
            // Original mask rendering substitutes 1 for fill opacity before the f32 conversion.
            if !(opacity as f32).is_finite() {
                return Err(CpuSvgSceneError::NonFiniteMaskOpacity);
            }
            if let CpuSvgSceneLayer::Vector(vector) = layer {
                let appearance = vector
                    .attribute::<Appearance>(ATTR_APPEARANCE)
                    .ok_or(CpuSvgSceneError::UnsupportedVectorAppearance)?;
                for index in 0..appearance.len() {
                    let paint = appearance
                        .paint_at(index)
                        .ok_or_else(|| CpuSvgSceneError::WrongAttributeType(ATTR_PAINT.into()))?;
                    if let Some(item) = gradient_paint_item(paint) {
                        validate_gradient_item(&item, true)?;
                    }
                }
            }
            mask_active = true;
        } else if mask_active && !next_clips {
            mask_active = false;
        }
    }
    Ok(())
}

fn source_image_item(source: &AdmittedCpuImage) -> Item<Raster<CPU>> {
    Item::new_from_element(Raster::new_cpu(source.image().clone())).with_attribute(
        ATTR_TRANSFORM,
        DAffine2::from_cols_array(&source.descriptor().transform),
    )
}

fn validate_image_item(
    item: &Item<Raster<CPU>>,
    logical: DAffine2,
) -> Result<(), CpuSvgSceneError> {
    let transform = composition_attributes(item.attributes(), CompositionAttributeProfile::Image)?;
    let image = item.element().data();
    if image.base64_string.is_some() {
        return Err(CpuSvgSceneError::UnexpectedImageCache);
    }
    let combined = logical * transform;
    if !combined.is_finite()
        || [DVec2::ZERO, DVec2::X, DVec2::Y, DVec2::ONE]
            .into_iter()
            .any(|point| !combined.transform_point2(point).is_finite())
    {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    Ok(())
}

fn composition_graphics(
    layers: &[CpuSvgSceneLayer],
    physical: DAffine2,
    logical: DAffine2,
) -> Result<List<Graphic>, CpuSvgSceneError> {
    let mut graphics = List::<Graphic>::new();
    for layer in layers {
        let graphic = match layer {
            CpuSvgSceneLayer::Image(source) => {
                validate_image_item(source.original_item(), logical)?;
                Graphic::from(source.original_item().clone())
            }
            CpuSvgSceneLayer::SourceImage(source) => {
                let item = source_image_item(source);
                validate_image_item(&item, logical)?;
                Graphic::from(item)
            }
            CpuSvgSceneLayer::Vector(item) => {
                let transform =
                    composition_attributes(item.attributes(), CompositionAttributeProfile::Vector)?;
                validate_vector_geometry(item.element(), transform, logical)?;
                let appearance = item
                    .attribute::<Appearance>(ATTR_APPEARANCE)
                    .ok_or(CpuSvgSceneError::UnsupportedVectorAppearance)?;
                validate_gradient_placements(appearance, transform, logical)?;
                if let Some(coverage) = appearance.first_coverage_of(Cover::Stroke) {
                    validate_vector_stroke_geometry(
                        item.element(),
                        transform,
                        logical,
                        &coverage.stroke_params(),
                        appearance,
                    )?;
                }
                Graphic::from(item.clone())
            }
            CpuSvgSceneLayer::Text(item) => {
                validate_text_item(item, logical)?;
                Graphic::from(item.clone())
            }
            CpuSvgSceneLayer::TextList(items) => {
                for key in items.attribute_keys() {
                    if !matches!(
                        key,
                        ATTR_FONT
                            | ATTR_FONT_SIZE
                            | ATTR_LINE_HEIGHT
                            | ATTR_LETTER_SPACING
                            | ATTR_LETTER_TILT
                            | ATTR_MAX_WIDTH
                            | ATTR_MAX_HEIGHT
                            | ATTR_TEXT_ALIGN
                            | ATTR_TRANSFORM
                            | ATTR_OPACITY
                            | ATTR_OPACITY_FILL
                            | ATTR_BLEND_MODE
                            | ATTR_CLIPPING_MASK
                    ) {
                        return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned()));
                    }
                }
                for index in 0..items.len() {
                    let item = items
                        .clone_item(index)
                        .ok_or(CpuSvgSceneError::InvalidTextLayout)?;
                    validate_text_item(&item, logical)?;
                }
                Graphic::from(items.clone())
            }
            CpuSvgSceneLayer::Gradient(item) => {
                validate_standalone_gradient(item, physical, logical)?;
                Graphic::from(item.clone())
            }
            CpuSvgSceneLayer::GradientList(items) => {
                validate_gradient_list_attributes(items)?;
                for index in 0..items.len() {
                    let item = items
                        .clone_item(index)
                        .ok_or(CpuSvgSceneError::InvalidGradient)?;
                    validate_standalone_gradient(&item, physical, logical)?;
                }
                Graphic::from(items.clone())
            }
            CpuSvgSceneLayer::Color(item) => {
                validate_color_item(item, logical)?;
                Graphic::from(item.clone())
            }
            CpuSvgSceneLayer::ColorList(items) => {
                for key in items.attribute_keys() {
                    if !matches!(
                        key,
                        ATTR_TRANSFORM
                            | ATTR_OPACITY
                            | ATTR_OPACITY_FILL
                            | ATTR_BLEND_MODE
                            | ATTR_CLIPPING_MASK
                    ) {
                        return Err(CpuSvgSceneError::UnsupportedAttribute(key.to_owned()));
                    }
                }
                for index in 0..items.len() {
                    let item = items
                        .clone_item(index)
                        .ok_or(CpuSvgSceneError::SourceRenderingFailed)?;
                    validate_color_item(&item, logical)?;
                }
                Graphic::from(items.clone())
            }
        };
        graphics.push(Item::new_from_element(graphic));
    }
    validate_composition_mask_opacities(layers, &graphics)?;
    Ok(graphics)
}

pub fn prepare_cpu_svg_composition(
    layers: &[CpuSvgSceneLayer],
    configuration: CpuSvgSceneConfiguration,
) -> Result<PreparedCpuSvgComposition, CpuSvgSceneError> {
    let (params, logical, resolution) = composition_configuration(configuration)?;
    let (svg, layers) = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let layers = seal_composition_layers(layers)?;
        let graphics = composition_graphics(&layers, params.footprint.transform, logical)?;
        let mut render = SvgRender::new();
        graphics.render_svg(&mut render, &params);
        let intermediate = SvgRenderOutput::from(render);
        let mut render = SvgRender::from(&intermediate);
        render.wrap_with_transform(logical, Some(resolution));
        Ok((SvgRenderOutput::from(render), layers))
    }))
    .map_err(|_| CpuSvgSceneError::SourceRenderingFailed)??;
    if svg.svg.len() > configuration.max_svg_bytes {
        return Err(CpuSvgSceneError::SvgBudgetExceeded {
            maximum: configuration.max_svg_bytes,
            actual: svg.svg.len(),
        });
    }
    Ok(PreparedCpuSvgComposition {
        layers,
        configuration,
        svg,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct CpuSvgArtboardConfiguration {
    pub location: DVec2,
    pub dimensions: DVec2,
    pub background: Color,
    pub clip: bool,
}

pub struct PreparedCpuSvgArtboard {
    layers: Vec<CpuSvgSceneLayer>,
    configuration: CpuSvgSceneConfiguration,
    artboard: CpuSvgArtboardConfiguration,
    svg: SvgRenderOutput,
}

impl PreparedCpuSvgArtboard {
    pub fn layers(&self) -> &[CpuSvgSceneLayer] {
        &self.layers
    }

    pub fn configuration(&self) -> &CpuSvgSceneConfiguration {
        &self.configuration
    }

    pub fn artboard(&self) -> &CpuSvgArtboardConfiguration {
        &self.artboard
    }

    pub fn svg(&self) -> &SvgRenderOutput {
        &self.svg
    }

    pub fn source_revision(&self) -> &'static str {
        crate::resources::GRAPHITE_IMAGE_SOURCE_REVISION
    }
}

pub fn prepare_cpu_svg_artboard(
    layers: &[CpuSvgSceneLayer],
    artboard: CpuSvgArtboardConfiguration,
    configuration: CpuSvgSceneConfiguration,
) -> Result<PreparedCpuSvgArtboard, CpuSvgSceneError> {
    let (params, logical, resolution) = composition_configuration(configuration)?;
    if !artboard.location.is_finite()
        || !artboard.dimensions.is_finite()
        || artboard.dimensions.x < 1.
        || artboard.dimensions.y < 1.
        || !(artboard.location + artboard.dimensions).is_finite()
    {
        return Err(CpuSvgSceneError::InvalidArtboardGeometry);
    }
    if [
        artboard.background.r(),
        artboard.background.g(),
        artboard.background.b(),
        artboard.background.a(),
    ]
    .iter()
    .any(|value| !value.is_finite())
        || artboard
            .background
            .to_gamma_srgb_channels()
            .iter()
            .any(|value| !value.is_finite())
        || (artboard.background.a() < 1. && !(artboard.background.a() * 1000.).is_finite())
    {
        return Err(CpuSvgSceneError::NonFinitePaint);
    }
    let translated = logical * DAffine2::from_translation(artboard.location);
    if !translated.is_finite()
        || [
            DVec2::ZERO,
            DVec2::new(artboard.dimensions.x, 0.),
            artboard.dimensions,
            DVec2::new(0., artboard.dimensions.y),
        ]
        .into_iter()
        .any(|point| !translated.transform_point2(point).is_finite())
    {
        return Err(CpuSvgSceneError::NonFiniteGeometry);
    }
    let (svg, layers) = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let layers = seal_composition_layers(layers)?;
        let graphics = composition_graphics(&layers, params.footprint.transform, translated)?;
        let item = Item::new_from_element(Artboard::new(graphics))
            .with_attribute(ATTR_LOCATION, artboard.location)
            .with_attribute(ATTR_DIMENSIONS, artboard.dimensions)
            .with_attribute(ATTR_BACKGROUND, artboard.background)
            .with_attribute(ATTR_CLIP, artboard.clip);
        let mut render = SvgRender::new();
        List::<Artboard>::from(item).render_svg(&mut render, &params);
        let intermediate = SvgRenderOutput::from(render);
        let mut render = SvgRender::from(&intermediate);
        render.wrap_with_transform(logical, Some(resolution));
        Ok((SvgRenderOutput::from(render), layers))
    }))
    .map_err(|_| CpuSvgSceneError::SourceRenderingFailed)??;
    if svg.svg.len() > configuration.max_svg_bytes {
        return Err(CpuSvgSceneError::SvgBudgetExceeded {
            maximum: configuration.max_svg_bytes,
            actual: svg.svg.len(),
        });
    }
    Ok(PreparedCpuSvgArtboard {
        layers,
        configuration,
        artboard,
        svg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{
        AlphaAssociation, CpuImageBudget, CpuImageDescriptor, CpuImageEdit, CpuSampleFormat,
        ImageColorSpace, ImageResourceIdentity, admit_cpu_image, admit_cpu_result,
        evaluate_cpu_image_edit,
    };
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use graphite_glam::UVec2;
    use graphite_raster_types::Image;
    use sha2::{Digest, Sha256};
    use std::sync::Arc;

    const PIXELS: [u8; 16] = [
        255, 0, 0, 255, 64, 128, 192, 128, 255, 0, 0, 0, 0, 0, 255, 64,
    ];

    fn result(transform: [f64; 6]) -> CpuImageResult {
        let source = admit_cpu_image(
            ImageResourceIdentity {
                asset_id: "test-cpu-svg-source".into(),
                version_id: "explicit-test-version".into(),
                content_sha256: Sha256::digest(PIXELS).into(),
            },
            CpuImageDescriptor {
                width: 2,
                height: 2,
                row_stride: 8,
                sample_format: CpuSampleFormat::Rgba8,
                color_space: ImageColorSpace::Srgb,
                alpha: AlphaAssociation::Straight,
                transform,
            },
            Arc::from(PIXELS),
            CpuImageBudget {
                max_input_bytes: 16,
                max_decoded_bytes: 4 * (std::mem::size_of::<Color>() + 4),
            },
        )
        .expect("actual CPU image admission");
        evaluate_cpu_image_edit(&source, CpuImageEdit::Invert, 64)
            .expect("original Invert constructor")
    }

    fn with_item(source: &CpuImageResult, item: Item<Raster<CPU>>) -> CpuImageResult {
        admit_cpu_result(
            source.input_identity().clone(),
            source.input_origin(),
            source.operation(),
            item,
            64,
        )
        .expect("source item and attributes admitted without replacement")
    }

    fn configuration(scale: f64) -> CpuSvgSceneConfiguration {
        CpuSvgSceneConfiguration {
            viewport: Footprint {
                transform: DAffine2::from_cols_array(&[1.25, 0.5, -0.25, 2., 11., -7.]),
                resolution: UVec2::new(640, 480),
                quality: Footprint::DEFAULT.quality,
            },
            scale,
            max_svg_bytes: 16 * 1024,
        }
    }

    fn original_scene(
        source: &CpuImageResult,
        configuration: CpuSvgSceneConfiguration,
        background: Option<Color>,
    ) -> SvgRenderOutput {
        let footprint = Footprint {
            transform: DAffine2::from_scale(DVec2::splat(configuration.scale))
                * configuration.viewport.transform,
            ..configuration.viewport
        };
        let params = RenderParams {
            footprint,
            scale: configuration.scale,
            render_mode: RenderMode::Normal,
            for_export: true,
            for_mask: false,
            thumbnail: false,
            inside_pattern: false,
            render_output_type: RenderOutputType::Svg,
            artboard_background: background,
            viewport_zoom: configuration.viewport.scale_magnitudes().x,
            ..RenderParams::default()
        };
        let mut render = SvgRender::new();
        List::<Raster<CPU>>::from(source.original_item().clone()).render_svg(&mut render, &params);
        let original_intermediate = SvgRenderOutput::from(render);
        let mut render = SvgRender::from(&original_intermediate);
        render.wrap_with_transform(
            DAffine2::from_scale(DVec2::splat(1. / params.scale)) * footprint.transform,
            Some(footprint.resolution.as_dvec2() / params.scale),
        );
        SvgRenderOutput::from(render)
    }

    fn embedded_png(svg: &SvgRenderOutput) -> Vec<u8> {
        let (_, payload) = svg
            .svg
            .split_once("href=\"data:image/png;base64,")
            .expect("original exported image payload");
        let (payload, _) = payload.split_once('"').expect("original href terminator");
        STANDARD.decode(payload).expect("original base64 PNG")
    }

    #[test]
    fn at_r20_1_cpu_svg_matches_original_affine_opacity_blend_and_pixel_encoder() {
        let source = result([3., 0.75, -1., 2., 9., 13.]);
        let item = source
            .original_item()
            .clone()
            .with_attribute(ATTR_BLEND_MODE, BlendMode::Multiply)
            .with_attribute(ATTR_OPACITY, 0.5_f64)
            .with_attribute(ATTR_OPACITY_FILL, 0.25_f64);
        let source = with_item(&source, item);
        let before = source.original_item().clone();
        let config = configuration(2.);
        let prepared = prepare_cpu_svg_scene(&source, config).expect("bounded original CPU SVG");
        let original = original_scene(&source, config, Some(Color::TRANSPARENT));
        assert_eq!(prepared.svg().svg, original.svg);
        assert_eq!(prepared.svg().svg_defs, original.svg_defs);
        assert!(prepared.svg().svg_defs.is_empty());
        assert!(prepared.svg().image_data.is_empty());
        assert!(prepared.svg().svg.contains("width=\"1\" height=\"1\""));
        assert!(prepared.svg().svg.contains("matrix(3,0.75,-1,2,9,13)"));
        assert!(prepared.svg().svg.contains("opacity=\"0.125\""));
        assert!(prepared.svg().svg.contains("mix-blend-mode: multiply;"));
        assert!(prepared.svg().svg.contains("viewBox=\"0 0 320 240\""));
        assert_eq!(embedded_png(prepared.svg()), source.image().to_png());
        let decoded = Image::<Color>::from_encoded(&embedded_png(prepared.svg()))
            .expect("original PNG decoder");
        assert_eq!(decoded.to_flat_u8(), source.image().to_flat_u8());
        assert_eq!(source.original_item(), &before);
        assert_eq!(prepared.source().original_item(), &before);
        assert_eq!(prepared.source().input_identity(), source.input_identity());
        assert_eq!(prepared.source().input_origin(), source.input_origin());
        assert_eq!(prepared.source().operation(), source.operation());
        assert_eq!(prepared.source().pixels_sha256(), source.pixels_sha256());
        assert_eq!(
            prepared.source().descriptor().transform.map(f64::to_bits),
            source.descriptor().transform.map(f64::to_bits)
        );
        assert_eq!(
            prepared
                .configuration()
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits),
            config.viewport.transform.to_cols_array().map(f64::to_bits)
        );
        assert_eq!(
            prepared.configuration().scale.to_bits(),
            config.scale.to_bits()
        );
        assert_eq!(prepared.configuration().max_svg_bytes, config.max_svg_bytes);
        assert_eq!(
            prepared.svg().svg,
            original_scene(&source, config, Some(Color::WHITE)).svg
        );
    }

    #[test]
    fn at_r20_1_cpu_svg_uses_logical_scale_unit_square_and_preserves_signed_zero_source() {
        let source = result([1., -0., 0., 1., -0., 0.]);
        let original_bits = source.descriptor().transform.map(f64::to_bits);
        for (scale, view_box) in [
            (0.5, "0 0 1280 960"),
            (1., "0 0 640 480"),
            (2., "0 0 320 240"),
        ] {
            let config = configuration(scale);
            let prepared =
                prepare_cpu_svg_scene(&source, config).expect("original scale equations");
            assert!(
                prepared
                    .svg()
                    .svg
                    .contains(&format!("viewBox=\"{view_box}\""))
            );
            assert!(
                prepared
                    .svg()
                    .svg
                    .contains("matrix(1.25,0.5,-0.25,2,11,-7)")
            );
            assert!(prepared.svg().svg.contains("width=\"1\" height=\"1\""));
            assert_eq!(
                prepared.svg().svg,
                original_scene(&source, config, None).svg
            );
            assert_eq!(
                prepared.source().descriptor().transform.map(f64::to_bits),
                original_bits
            );
            assert_eq!(
                source.descriptor().transform.map(f64::to_bits),
                original_bits
            );
            assert_eq!((source.image().width, source.image().height), (2, 2));
        }
    }

    #[test]
    fn at_r20_1_cpu_svg_preserves_finite_source_opacity_and_only_original_svg_modes() {
        let source = result([1., 0., 0., 1., 0., 0.]);
        for (opacity, fill, emitted) in [
            (1., 1., None),
            (2., 1., None),
            (-2., 0.5, Some("opacity=\"-1\"")),
            (0., -2., Some("opacity=\"-0\"")),
            (0.5, 0.5, Some("opacity=\"0.25\"")),
        ] {
            let current = with_item(
                &source,
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_OPACITY, opacity)
                    .with_attribute(ATTR_OPACITY_FILL, fill),
            );
            let prepared = prepare_cpu_svg_scene(&current, configuration(1.))
                .expect("finite original opacity");
            assert_eq!(
                prepared.svg().svg,
                original_scene(&current, configuration(1.), None).svg
            );
            if let Some(value) = emitted {
                assert!(prepared.svg().svg.contains(value));
            } else {
                assert!(!prepared.svg().svg.contains(" opacity="));
            }
        }
        for group in BlendMode::list_svg_subset() {
            for &mode in group {
                let current = with_item(
                    &source,
                    source
                        .original_item()
                        .clone()
                        .with_attribute(ATTR_BLEND_MODE, mode),
                );
                let prepared =
                    prepare_cpu_svg_scene(&current, configuration(1.)).expect("original SVG mode");
                assert_eq!(
                    prepared.svg().svg,
                    original_scene(&current, configuration(1.), None).svg
                );
                if mode != BlendMode::Normal {
                    let style = format!(
                        "mix-blend-mode: {};",
                        mode.to_svg_style_name()
                            .expect("verified original SVG name")
                    );
                    assert!(prepared.svg().svg.contains(&style));
                }
            }
        }
        for mode in [
            BlendMode::LinearBurn,
            BlendMode::DarkerColor,
            BlendMode::LinearDodge,
            BlendMode::Erase,
            BlendMode::Restore,
            BlendMode::MultiplyAlpha,
        ] {
            let current = with_item(
                &source,
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_BLEND_MODE, mode),
            );
            assert_eq!(
                prepare_cpu_svg_scene(&current, configuration(1.)).err(),
                Some(CpuSvgSceneError::UnsupportedBlendMode(mode))
            );
            assert_eq!(
                current
                    .original_item()
                    .attribute::<BlendMode>(ATTR_BLEND_MODE),
                Some(&mode)
            );
        }
    }

    #[test]
    fn at_r20_1_cpu_svg_rejects_unsupported_wrong_type_nonfinite_and_cached_attributes() {
        let source = result([1., 0., 0., 1., 0., 0.]);
        for (item, expected) in [
            (
                source
                    .original_item()
                    .clone()
                    .with_attribute("unhandled", true),
                CpuSvgSceneError::UnsupportedAttribute("unhandled".into()),
            ),
            (
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_BLEND_MODE, String::from("multiply")),
                CpuSvgSceneError::WrongAttributeType(ATTR_BLEND_MODE.into()),
            ),
            (
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_OPACITY, 0.5_f32),
                CpuSvgSceneError::WrongAttributeType(ATTR_OPACITY.into()),
            ),
            (
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_OPACITY_FILL, true),
                CpuSvgSceneError::WrongAttributeType(ATTR_OPACITY_FILL.into()),
            ),
            (
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_OPACITY, f64::NAN),
                CpuSvgSceneError::NonFiniteOpacity(ATTR_OPACITY.into()),
            ),
            (
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_OPACITY_FILL, f64::INFINITY),
                CpuSvgSceneError::NonFiniteOpacity(ATTR_OPACITY_FILL.into()),
            ),
            (
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_OPACITY, f64::MAX)
                    .with_attribute(ATTR_OPACITY_FILL, 2_f64),
                CpuSvgSceneError::NonFiniteCombinedOpacity,
            ),
            (
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_OPACITY, 1e100_f64),
                CpuSvgSceneError::NonFiniteCombinedOpacity,
            ),
        ] {
            let current = with_item(&source, item);
            let before = current.original_item().clone();
            assert_eq!(
                prepare_cpu_svg_scene(&current, configuration(1.)).err(),
                Some(expected)
            );
            let mut actual_attributes = current.attributes().clone();
            let mut before_attributes = before.attributes().clone();
            for key in [ATTR_OPACITY, ATTR_OPACITY_FILL] {
                if before_attributes.get::<f64>(key).is_some() {
                    assert_eq!(
                        actual_attributes.remove::<f64>(key).map(f64::to_bits),
                        before_attributes.remove::<f64>(key).map(f64::to_bits)
                    );
                }
            }
            assert_eq!(actual_attributes, before_attributes);
            assert_eq!(current.image(), before.element().data());
            assert_eq!(current.input_identity(), source.input_identity());
        }
        let wrong_transform = source
            .original_item()
            .clone()
            .with_attribute(ATTR_TRANSFORM, [1_f64, 0., 0., 1., 0., 0.]);
        assert_eq!(
            validate_item_attributes(&wrong_transform),
            Err(CpuSvgSceneError::WrongAttributeType(ATTR_TRANSFORM.into()))
        );
        let nonfinite_transform = source.original_item().clone().with_attribute(
            ATTR_TRANSFORM,
            DAffine2::from_cols_array(&[1., 0., 0., 1., f64::NAN, 0.]),
        );
        assert_eq!(
            validate_item_attributes(&nonfinite_transform),
            Err(CpuSvgSceneError::InvalidImageTransform)
        );
        let mut cached = source.original_item().clone();
        cached.element_mut().data_mut().base64_string =
            Some("data:image/png;base64,foreign".into());
        let cached = with_item(&source, cached);
        assert_eq!(
            prepare_cpu_svg_scene(&cached, configuration(1.)).err(),
            Some(CpuSvgSceneError::UnexpectedImageCache)
        );
        assert_eq!(source.image().base64_string, None);
    }

    #[test]
    fn at_r20_1_cpu_svg_rejects_configuration_and_derived_geometry_overflow_then_recovers() {
        let source = result([1., 0., 0., 1., 0., 0.]);
        for scale in [0., -1., f64::NAN, f64::INFINITY] {
            assert_eq!(
                prepare_cpu_svg_scene(&source, configuration(scale)).err(),
                Some(CpuSvgSceneError::InvalidScale)
            );
        }
        for resolution in [UVec2::new(0, 480), UVec2::new(640, 0)] {
            let mut config = configuration(1.);
            config.viewport.resolution = resolution;
            assert_eq!(
                prepare_cpu_svg_scene(&source, config).err(),
                Some(CpuSvgSceneError::EmptyViewport)
            );
        }
        let mut config = configuration(1.);
        config.viewport.transform = DAffine2::from_cols_array(&[1., 0., 0., 1., f64::INFINITY, 0.]);
        assert_eq!(
            prepare_cpu_svg_scene(&source, config).err(),
            Some(CpuSvgSceneError::InvalidViewportTransform)
        );
        for scale in [f64::MAX, f64::from_bits(1)] {
            assert_eq!(
                prepare_cpu_svg_scene(&source, configuration(scale)).err(),
                Some(CpuSvgSceneError::NonFiniteGeometry)
            );
        }
        let huge = result([1., 0., 0., 1., f64::MAX, 0.]);
        let mut config = configuration(1.);
        config.viewport.transform = DAffine2::from_cols_array(&[2., 0., 0., 1., 0., 0.]);
        assert_eq!(
            prepare_cpu_svg_scene(&huge, config).err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
        assert!(prepare_cpu_svg_scene(&source, configuration(1.)).is_ok());
    }

    #[test]
    fn at_r20_1_cpu_svg_output_cap_is_post_encoding_atomic_and_recoverable() {
        let source = result([2., 0., 0., 3., 5., 7.]);
        let before = source.original_item().clone();
        let first = prepare_cpu_svg_scene(&source, configuration(1.)).expect("original scene");
        let length = first.svg().svg.len();
        let mut config = configuration(1.);
        config.max_svg_bytes = length;
        assert_eq!(
            prepare_cpu_svg_scene(&source, config)
                .expect("exact output cap")
                .svg()
                .svg,
            first.svg().svg
        );
        for maximum in [0, length - 1] {
            config.max_svg_bytes = maximum;
            assert_eq!(
                prepare_cpu_svg_scene(&source, config).err(),
                Some(CpuSvgSceneError::SvgBudgetExceeded {
                    maximum,
                    actual: length
                })
            );
            assert_eq!(source.original_item(), &before);
            assert_eq!(source.input_identity(), first.source().input_identity());
            assert_eq!(source.pixels_sha256(), first.source().pixels_sha256());
        }
        config.max_svg_bytes = length;
        assert_eq!(
            prepare_cpu_svg_scene(&source, config)
                .expect("recovery")
                .svg()
                .svg,
            first.svg().svg
        );
    }

    #[test]
    fn at_r20_1_cpu_svg_embedded_png_keeps_original_tiny_alpha_quantization() {
        let source = result([1., 0., 0., 1., 0., 0.]);
        let mut item = source.original_item().clone();
        item.element_mut().data_mut().data = vec![
            Color::from_gamma_srgb_channels(0.25, 0.5, 0.75, 0.001),
            Color::from_gamma_srgb_channels(0.25, 0.5, 0.75, 0.5),
            Color::from_gamma_srgb_channels(1., 0., 0., 1.),
            Color::from_gamma_srgb_channels(0.2, 0.4, 0.6, 0.),
        ];
        let source = with_item(&source, item);
        let before = source.original_item().clone();
        let prepared = prepare_cpu_svg_scene(&source, configuration(1.))
            .expect("original embedded PNG quantization");
        let bytes = embedded_png(prepared.svg());
        assert_eq!(bytes, source.image().to_png());
        assert_eq!(
            bytes.get(..8),
            Some([137, 80, 78, 71, 13, 10, 26, 10].as_slice())
        );
        let decoded = Image::<Color>::from_encoded(&bytes).expect("original PNG decode");
        let (pixels, width, height) = decoded.to_flat_u8();
        assert_eq!((width, height), (2, 2));
        assert_eq!((pixels.clone(), width, height), source.image().to_flat_u8());
        assert_eq!(pixels.get(..4), Some([0, 0, 0, 0].as_slice()));
        assert_eq!(pixels.get(7), Some(&128));
        assert_eq!(pixels.get(8..12), Some([255, 0, 0, 255].as_slice()));
        assert_eq!(pixels.get(12..), Some([0, 0, 0, 0].as_slice()));
        assert_eq!(source.original_item(), &before);
        assert_eq!(source.image().data[0].a().to_bits(), 0.001_f32.to_bits());
        assert_ne!(
            decoded.data.first().expect("decoded tiny-alpha pixel"),
            source.image().data.first().expect("immutable float source")
        );
    }

    fn solid_vector(alpha: f32) -> Item<Vector> {
        use graphite_vector_types::kurbo::BezPath;
        let mut path = BezPath::new();
        path.move_to((0., 0.));
        path.line_to((3., 0.));
        path.line_to((3., 2.));
        path.line_to((0., 2.));
        path.close_path();
        Item::new_from_element(Vector::from_bezpath(path)).with_attribute(
            ATTR_APPEARANCE,
            Appearance::new_single(
                graphite_graphic_types::Coverage::new_fill(),
                Graphic::from(Color::from_gamma_srgb_channels(0.8, 0.2, 0.1, alpha)),
            ),
        )
    }

    fn original_vector_path(item: &Item<Vector>) -> String {
        let transform = item.attribute_cloned_or_default::<DAffine2>(ATTR_TRANSFORM);
        let mut path = String::new();
        for mut bezpath in item.element().stroke_bezpath_iter() {
            bezpath.apply_affine(graphite_vector_types::kurbo::Affine::new(
                transform.to_cols_array(),
            ));
            path.push_str(&bezpath.to_svg());
        }
        format!("d=\"{path}\"")
    }

    fn original_composition(
        layers: &[CpuSvgSceneLayer],
        config: CpuSvgSceneConfiguration,
    ) -> SvgRenderOutput {
        let footprint = Footprint {
            transform: DAffine2::from_scale(DVec2::splat(config.scale)) * config.viewport.transform,
            ..config.viewport
        };
        let params = RenderParams {
            footprint,
            scale: config.scale,
            render_mode: RenderMode::Normal,
            for_export: true,
            for_mask: false,
            thumbnail: false,
            inside_pattern: false,
            render_output_type: RenderOutputType::Svg,
            artboard_background: Some(Color::TRANSPARENT),
            viewport_zoom: config.viewport.scale_magnitudes().x,
            ..RenderParams::default()
        };
        let mut graphics = List::<Graphic>::new();
        for layer in layers {
            let graphic = match layer {
                CpuSvgSceneLayer::Image(image) => Graphic::from(image.original_item().clone()),
                CpuSvgSceneLayer::SourceImage(source) => Graphic::from(
                    Item::new_from_element(Raster::new_cpu(source.image().clone())).with_attribute(
                        ATTR_TRANSFORM,
                        DAffine2::from_cols_array(&source.descriptor().transform),
                    ),
                ),
                CpuSvgSceneLayer::Vector(vector) => Graphic::from(vector.clone()),
                CpuSvgSceneLayer::Text(text) => Graphic::from(text.clone()),
                CpuSvgSceneLayer::TextList(texts) => Graphic::from(texts.clone()),
                CpuSvgSceneLayer::Color(color) => Graphic::from(color.clone()),
                CpuSvgSceneLayer::ColorList(colors) => Graphic::from(colors.clone()),
                CpuSvgSceneLayer::Gradient(gradient) => Graphic::from(gradient.clone()),
                CpuSvgSceneLayer::GradientList(gradients) => Graphic::from(gradients.clone()),
            };
            graphics.push(Item::new_from_element(graphic));
        }
        let mut render = SvgRender::new();
        graphics.render_svg(&mut render, &params);
        let intermediate = SvgRenderOutput::from(render);
        let mut render = SvgRender::from(&intermediate);
        render.wrap_with_transform(
            DAffine2::from_scale(DVec2::splat(1. / config.scale)) * footprint.transform,
            Some(footprint.resolution.as_dvec2() / config.scale),
        );
        SvgRenderOutput::from(render)
    }

    fn assert_layers_unchanged(actual: &[CpuSvgSceneLayer], before: &[CpuSvgSceneLayer]) {
        assert_eq!(actual.len(), before.len());
        for (actual, before) in actual.iter().zip(before) {
            match (actual, before) {
                (CpuSvgSceneLayer::Image(actual), CpuSvgSceneLayer::Image(before)) => {
                    assert_eq!(actual.original_item(), before.original_item());
                    assert_eq!(actual.input_identity(), before.input_identity());
                    assert_eq!(actual.input_origin(), before.input_origin());
                    assert_eq!(actual.operation(), before.operation());
                    assert_eq!(actual.pixels_sha256(), before.pixels_sha256());
                    assert_eq!(
                        actual.descriptor().transform.map(f64::to_bits),
                        before.descriptor().transform.map(f64::to_bits)
                    );
                }
                (CpuSvgSceneLayer::Vector(actual), CpuSvgSceneLayer::Vector(before)) => {
                    assert_eq!(actual, before)
                }
                (CpuSvgSceneLayer::Text(actual), CpuSvgSceneLayer::Text(before)) => {
                    assert_eq!(actual, before)
                }
                (CpuSvgSceneLayer::SourceImage(actual), CpuSvgSceneLayer::SourceImage(before)) => {
                    assert!(Arc::ptr_eq(actual, before));
                    assert_eq!(actual.identity(), before.identity());
                    assert_eq!(actual.origin(), before.origin());
                    assert_eq!(
                        actual.descriptor().transform.map(f64::to_bits),
                        before.descriptor().transform.map(f64::to_bits)
                    );
                }
                (CpuSvgSceneLayer::TextList(actual), CpuSvgSceneLayer::TextList(before)) => {
                    assert_eq!(actual, before)
                }
                (CpuSvgSceneLayer::Color(actual), CpuSvgSceneLayer::Color(before)) => {
                    assert_eq!(actual, before)
                }
                (CpuSvgSceneLayer::ColorList(actual), CpuSvgSceneLayer::ColorList(before)) => {
                    assert_eq!(actual, before)
                }
                (CpuSvgSceneLayer::Gradient(actual), CpuSvgSceneLayer::Gradient(before)) => {
                    assert_eq!(actual, before)
                }
                (
                    CpuSvgSceneLayer::GradientList(actual),
                    CpuSvgSceneLayer::GradientList(before),
                ) => assert_eq!(actual, before),
                _ => panic!("the source layer variant changed"),
            }
        }
    }

    fn assert_mask_coherence(svg: &str, expected: &[(&str, usize)]) {
        let mut all_ids = std::collections::HashSet::new();
        for &(kind, references) in expected {
            let marker = format!("<{kind} id=\"");
            let ids = svg
                .match_indices(&marker)
                .map(|(position, _)| {
                    let remaining = &svg[position + marker.len()..];
                    remaining
                        .split_once('"')
                        .expect("source def ID terminator")
                        .0
                })
                .collect::<Vec<_>>();
            assert_eq!(ids.len(), 1, "one source {kind} def");
            let id = ids.first().expect("source def");
            assert!(all_ids.insert(*id), "each def has its own original UUID");
            assert!(id.starts_with("mask-"));
            let attribute = if kind == "clipPath" {
                "clip-path"
            } else {
                "mask"
            };
            assert_eq!(
                svg.matches(&format!("{attribute}=\"url(#{id})\"")).count(),
                references
            );
        }
        assert_eq!(
            svg.matches("url(#mask-").count(),
            expected.iter().map(|(_, count)| count).sum::<usize>()
        );
        assert_eq!(
            svg.matches("<clipPath id=").count() + svg.matches("<mask id=").count(),
            expected.len()
        );
    }

    #[test]
    fn at_r20_1_composition_retains_order_source_items_and_original_solid_rendering() {
        let image = result([2., 0., 0., 1., 5., 7.]);
        let vector = solid_vector(0.625)
            .with_attribute(
                ATTR_TRANSFORM,
                DAffine2::from_cols_array(&[1., 0.5, -0.25, 2., 13., 3.]),
            )
            .with_attribute(ATTR_OPACITY, -0.5_f64)
            .with_attribute(ATTR_OPACITY_FILL, 0.75_f64)
            .with_attribute(ATTR_BLEND_MODE, BlendMode::Screen);
        let layers = vec![
            CpuSvgSceneLayer::Image(image.clone()),
            CpuSvgSceneLayer::Vector(vector.clone()),
        ];
        let before = layers.clone();
        let config = configuration(2.);
        let prepared =
            prepare_cpu_svg_composition(&layers, config).expect("original asymmetric composition");
        let original = original_composition(&layers, config);
        assert_eq!(prepared.svg().svg, original.svg);
        assert!(prepared.svg().svg_defs.is_empty());
        assert!(prepared.svg().image_data.is_empty());
        assert!(
            prepared.svg().svg.find("<image").expect("source raster")
                < prepared.svg().svg.find("<path").expect("source solid path")
        );
        assert!(prepared.svg().svg.contains("mix-blend-mode: screen;"));
        assert!(prepared.svg().svg.contains("opacity=\"-0.375\""));
        assert_eq!(embedded_png(prepared.svg()), image.image().to_png());
        assert_layers_unchanged(&layers, &before);
        assert_layers_unchanged(prepared.layers(), &before);
        assert_eq!(
            prepared.source_revision(),
            crate::resources::GRAPHITE_IMAGE_SOURCE_REVISION
        );
        assert_eq!(
            prepared.configuration().viewport.resolution,
            config.viewport.resolution
        );
        assert_eq!(
            prepared
                .configuration()
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits),
            config.viewport.transform.to_cols_array().map(f64::to_bits)
        );
        let reverse = vec![
            CpuSvgSceneLayer::Vector(vector),
            CpuSvgSceneLayer::Image(image),
        ];
        let reversed =
            prepare_cpu_svg_composition(&reverse, config).expect("reversed source order");
        assert_eq!(
            reversed.svg().svg,
            original_composition(&reverse, config).svg
        );
        assert!(
            reversed.svg().svg.find("<path").expect("source path")
                < reversed.svg().svg.find("<image").expect("source raster")
        );
        assert_ne!(prepared.svg().svg, reversed.svg().svg);
        let mut color_list_vector = solid_vector(1.);
        color_list_vector.set_attribute(
            ATTR_APPEARANCE,
            Appearance::new_single(
                graphite_graphic_types::Coverage::new_fill(),
                Graphic::ColorList(List::new_from_element(Color::WHITE)),
            ),
        );
        let layers = [CpuSvgSceneLayer::Vector(color_list_vector)];
        assert_eq!(
            prepare_cpu_svg_composition(&layers, config)
                .expect("singleton original solid paint")
                .svg()
                .svg,
            original_composition(&layers, config).svg
        );
    }

    #[test]
    fn at_r20_1_composition_uses_original_clip_path_and_alpha_mask_selection() {
        let image = result([1., 0., 0., 1., 2., 3.]);
        for (alpha, opacity, kind) in [(1., 1., "clipPath"), (0.5, 1., "mask"), (1., 0.5, "mask")] {
            let masker = solid_vector(alpha)
                .with_attribute(
                    ATTR_TRANSFORM,
                    DAffine2::from_translation(DVec2::new(9., 7.)),
                )
                .with_attribute(ATTR_OPACITY, opacity)
                .with_attribute(ATTR_OPACITY_FILL, 0.25_f64);
            let graphic = Graphic::from(masker.clone());
            let masker_path = original_vector_path(&masker);
            assert_eq!(graphic.can_reduce_to_clip_path(), kind == "clipPath");
            let clipped_image = with_item(
                &image,
                image
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_CLIPPING_MASK, true),
            );
            let clipped_vector = solid_vector(1.).with_attribute(ATTR_CLIPPING_MASK, true);
            assert!(Graphic::from(clipped_vector.clone()).had_clip_enabled());
            let layers = [
                CpuSvgSceneLayer::Vector(masker),
                CpuSvgSceneLayer::Image(clipped_image),
                CpuSvgSceneLayer::Vector(clipped_vector),
            ];
            let before = layers.clone();
            let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                .expect("original sibling clipping");
            assert_mask_coherence(&prepared.svg().svg, &[(kind, 2)]);
            let original = original_composition(&layers, configuration(1.));
            assert_mask_coherence(&original.svg, &[(kind, 2)]);
            assert!(prepared.svg().svg.contains(&masker_path));
            if kind == "mask" {
                assert!(prepared.svg().svg.contains("mask-type=\"alpha\""));
            }
            assert_eq!(embedded_png(prepared.svg()), image.image().to_png());
            assert_layers_unchanged(&layers, &before);
            assert_layers_unchanged(prepared.layers(), &before);
        }
    }

    #[test]
    fn at_r20_1_composition_interrupted_clipping_keeps_terminal_mask_and_restarts() {
        let image = result([1., 0., 0., 1., 0., 0.]);
        let clipped_image = with_item(
            &image,
            image
                .original_item()
                .clone()
                .with_attribute(ATTR_CLIPPING_MASK, true),
        );
        let next_masker = solid_vector(1.).with_attribute(
            ATTR_TRANSFORM,
            DAffine2::from_translation(DVec2::new(20., 30.)),
        );
        let final_sibling = solid_vector(1.).with_attribute(
            ATTR_TRANSFORM,
            DAffine2::from_translation(DVec2::new(40., 50.)),
        );
        let next_masker_path = original_vector_path(&next_masker);
        let final_sibling_path = original_vector_path(&final_sibling);
        let layers = [
            CpuSvgSceneLayer::Image(image.clone()),
            CpuSvgSceneLayer::Vector(solid_vector(1.).with_attribute(ATTR_CLIPPING_MASK, true)),
            CpuSvgSceneLayer::Vector(next_masker),
            CpuSvgSceneLayer::Image(clipped_image),
            CpuSvgSceneLayer::Vector(final_sibling),
        ];
        let before = layers.clone();
        let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("two interrupted clipping runs");
        assert_mask_coherence(&prepared.svg().svg, &[("mask", 1), ("clipPath", 1)]);
        assert_mask_coherence(
            &original_composition(&layers, configuration(1.)).svg,
            &[("mask", 1), ("clipPath", 1)],
        );
        let (_, visible) = prepared
            .svg()
            .svg
            .split_once("</defs>")
            .expect("original def boundary");
        assert!(
            visible.find("mask=\"url(").expect("first clipped terminal")
                < visible
                    .find(&next_masker_path)
                    .expect("unclipped new masker")
        );
        assert!(
            visible
                .find("clip-path=\"url(")
                .expect("second clipped terminal")
                < visible
                    .find(&final_sibling_path)
                    .expect("unclipped final sibling")
        );
        assert_layers_unchanged(&layers, &before);
        let first_clipped = [CpuSvgSceneLayer::Vector(
            solid_vector(1.).with_attribute(ATTR_CLIPPING_MASK, true),
        )];
        let single = prepare_cpu_svg_composition(&first_clipped, configuration(1.))
            .expect("no predecessor invented");
        assert_mask_coherence(&single.svg().svg, &[]);
        assert_eq!(
            single.svg().svg,
            original_composition(&first_clipped, configuration(1.)).svg
        );
    }

    #[test]
    fn at_r20_1_composition_rejects_typed_attribute_and_nested_appearance_drift() {
        let vector = solid_vector(1.);
        let mut extra = Appearance::new_single(
            graphite_graphic_types::Coverage::new_fill(),
            Graphic::from(Color::WHITE),
        );
        extra.0.push(
            Item::new_from_element(graphite_graphic_types::Coverage::new_fill())
                .with_attribute(ATTR_PAINT, Graphic::from(Color::BLACK)),
        );
        let mut wrong_paint = Appearance::new_single(
            graphite_graphic_types::Coverage::new_fill(),
            Graphic::from(Color::WHITE),
        );
        wrong_paint
            .0
            .set_attribute(ATTR_PAINT, 0, "white".to_owned());
        let mut inner = graphite_graphic_types::Coverage::new_fill();
        inner.0.set_attribute("ignored-fill-metadata", true);
        let nested = Appearance::new_single(inner, Graphic::from(Color::WHITE));
        let stroke = Appearance::new_single(
            graphite_graphic_types::Coverage(Item::new_from_element(Cover::Stroke)),
            Graphic::from(Color::WHITE),
        );
        let mut extra_outer = Appearance::new_single(
            graphite_graphic_types::Coverage::new_fill(),
            Graphic::from(Color::WHITE),
        );
        extra_outer.0.set_attribute("unsupported-coverage", 0, true);
        assert!(wrong_paint.0.attribute::<String>(ATTR_PAINT, 0).is_some());
        for (item, expected) in [
            (
                vector.clone().with_attribute("unsupported", true),
                CpuSvgSceneError::UnsupportedAttribute("unsupported".into()),
            ),
            (
                vector
                    .clone()
                    .with_attribute(ATTR_TRANSFORM, [1_f64, 0., 0., 1., 0., 0.]),
                CpuSvgSceneError::WrongAttributeType(ATTR_TRANSFORM.into()),
            ),
            (
                vector
                    .clone()
                    .with_attribute(ATTR_CLIPPING_MASK, "true".to_owned()),
                CpuSvgSceneError::WrongAttributeType(ATTR_CLIPPING_MASK.into()),
            ),
            (
                vector
                    .clone()
                    .with_attribute(ATTR_APPEARANCE, "solid".to_owned()),
                CpuSvgSceneError::WrongAttributeType(ATTR_APPEARANCE.into()),
            ),
            (
                vector.clone().with_attribute(ATTR_OPACITY, 1_f32),
                CpuSvgSceneError::WrongAttributeType(ATTR_OPACITY.into()),
            ),
            (
                vector
                    .clone()
                    .with_attribute(ATTR_BLEND_MODE, BlendMode::Erase),
                CpuSvgSceneError::UnsupportedBlendMode(BlendMode::Erase),
            ),
            (
                vector
                    .clone()
                    .with_attribute(ATTR_APPEARANCE, Appearance::default()),
                CpuSvgSceneError::UnsupportedVectorAppearance,
            ),
            (
                Item::new_from_element(vector.element().clone()),
                CpuSvgSceneError::UnsupportedVectorAppearance,
            ),
            (
                vector.clone().with_attribute(ATTR_APPEARANCE, extra),
                CpuSvgSceneError::UnsupportedVectorAppearance,
            ),
            (
                vector.clone().with_attribute(ATTR_APPEARANCE, wrong_paint),
                CpuSvgSceneError::WrongAttributeType(ATTR_PAINT.into()),
            ),
            (
                vector.clone().with_attribute(ATTR_APPEARANCE, nested),
                CpuSvgSceneError::UnsupportedVectorAppearance,
            ),
            (
                vector.clone().with_attribute(ATTR_APPEARANCE, stroke),
                CpuSvgSceneError::UnsupportedVectorAppearance,
            ),
            (
                vector.clone().with_attribute(ATTR_APPEARANCE, extra_outer),
                CpuSvgSceneError::UnsupportedAttribute("unsupported-coverage".into()),
            ),
            (
                vector.clone().with_attribute(ATTR_OPACITY, f64::INFINITY),
                CpuSvgSceneError::NonFiniteOpacity(ATTR_OPACITY.into()),
            ),
            (
                vector
                    .clone()
                    .with_attribute(ATTR_OPACITY, f64::MAX)
                    .with_attribute(ATTR_OPACITY_FILL, 2_f64),
                CpuSvgSceneError::NonFiniteCombinedOpacity,
            ),
            (
                vector.clone().with_attribute(
                    ATTR_APPEARANCE,
                    Appearance::new_single(
                        graphite_graphic_types::Coverage::new_fill(),
                        Graphic::from(String::from("unsupported text paint")),
                    ),
                ),
                CpuSvgSceneError::UnsupportedVectorPaint,
            ),
            (
                vector.clone().with_attribute(
                    ATTR_APPEARANCE,
                    Appearance::new_single(
                        graphite_graphic_types::Coverage::new_fill(),
                        Graphic::from(Color::from_gamma_srgb_channels(f32::INFINITY, 0., 0., 1.)),
                    ),
                ),
                CpuSvgSceneError::NonFinitePaint,
            ),
            (
                vector.clone().with_attribute(
                    ATTR_APPEARANCE,
                    Appearance::new_single(
                        graphite_graphic_types::Coverage::new_fill(),
                        Graphic::from(
                            Item::new_from_element(Color::WHITE)
                                .with_attribute(ATTR_OPACITY, 0.5_f64),
                        ),
                    ),
                ),
                CpuSvgSceneError::UnsupportedVectorPaint,
            ),
        ] {
            let before = item.clone();
            let layers = [CpuSvgSceneLayer::Vector(item)];
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(expected)
            );
            let CpuSvgSceneLayer::Vector(after) = layers.first().expect("same source layer") else {
                panic!("variant changed")
            };
            assert_eq!(after, &before);
        }
        let image = result([1., 0., 0., 1., 0., 0.]);
        let wrong_clip = with_item(
            &image,
            image
                .original_item()
                .clone()
                .with_attribute(ATTR_CLIPPING_MASK, 1_u32),
        );
        assert_eq!(
            prepare_cpu_svg_composition(&[CpuSvgSceneLayer::Image(wrong_clip)], configuration(1.))
                .err(),
            Some(CpuSvgSceneError::WrongAttributeType(
                ATTR_CLIPPING_MASK.into()
            ))
        );
    }

    #[test]
    fn at_r20_1_composition_checks_actual_points_handles_topology_and_overflow() {
        use graphite_vector_types::kurbo::BezPath;
        use graphite_vector_types::vector::misc::BezierHandles;
        let mut path = BezPath::new();
        path.move_to((0., 0.));
        path.curve_to((1., 3.), (2., -1.), (3., 2.));
        path.line_to((0., 0.));
        path.close_path();
        let curve = Item::new_from_element(Vector::from_bezpath(path)).with_attribute(
            ATTR_APPEARANCE,
            solid_vector(1.)
                .attribute::<Appearance>(ATTR_APPEARANCE)
                .expect("solid source paint")
                .clone(),
        );
        let layers = [CpuSvgSceneLayer::Vector(curve)];
        let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("original finite cubic controls");
        assert_eq!(
            prepared.svg().svg,
            original_composition(&layers, configuration(1.)).svg
        );
        let collapsed = [CpuSvgSceneLayer::Vector(
            solid_vector(1.).with_attribute(ATTR_TRANSFORM, DAffine2::from_cols_array(&[0.; 6])),
        )];
        assert_eq!(
            prepare_cpu_svg_composition(&collapsed, configuration(1.))
                .expect("finite singular solid fill is not categorically rejected")
                .svg()
                .svg,
            original_composition(&collapsed, configuration(1.)).svg
        );
        let mut dangling = solid_vector(1.);
        dangling
            .element_mut()
            .segment_domain
            .set_start_point(0, usize::MAX);
        let mut branching = solid_vector(1.);
        branching.element_mut().segment_domain.set_start_point(0, 1);
        let mut invalid_handle = solid_vector(1.);
        invalid_handle.element_mut().segment_domain.set_handles(
            0,
            BezierHandles::Quadratic {
                handle: DVec2::new(f64::INFINITY, 0.),
            },
        );
        let mut overflow_handle =
            solid_vector(1.).with_attribute(ATTR_TRANSFORM, DAffine2::from_scale(DVec2::splat(2.)));
        overflow_handle.element_mut().segment_domain.set_handles(
            0,
            BezierHandles::Cubic {
                handle_start: DVec2::new(f64::MAX, 0.),
                handle_end: DVec2::ONE,
            },
        );
        for (item, expected) in [
            (dangling, CpuSvgSceneError::InvalidVectorGeometry),
            (branching, CpuSvgSceneError::UnsupportedBranchingVector),
            (invalid_handle, CpuSvgSceneError::NonFiniteGeometry),
            (overflow_handle, CpuSvgSceneError::NonFiniteGeometry),
        ] {
            let before = item.clone();
            let layers = [CpuSvgSceneLayer::Vector(item)];
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(expected)
            );
            let CpuSvgSceneLayer::Vector(after) = layers.first().expect("unchanged vector") else {
                panic!("variant changed")
            };
            assert_eq!(after, &before);
        }
        let mut nonfinite = solid_vector(1.);
        nonfinite
            .element_mut()
            .point_domain
            .positions_mut()
            .next()
            .expect("source point")
            .1
            .x = f64::NAN;
        let layers = [CpuSvgSceneLayer::Vector(nonfinite)];
        assert_eq!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
        let CpuSvgSceneLayer::Vector(after) = layers.first().expect("same vector") else {
            panic!("variant changed")
        };
        assert_eq!(
            after
                .element()
                .point_domain
                .positions()
                .first()
                .expect("same original point")
                .x
                .to_bits(),
            f64::NAN.to_bits()
        );
        let mut huge = solid_vector(1.);
        for (_, point) in huge.element_mut().point_domain.positions_mut() {
            point.x = f64::MAX;
        }
        let mut config = configuration(1.);
        config.viewport.transform = DAffine2::from_scale(DVec2::splat(2.));
        assert_eq!(
            prepare_cpu_svg_composition(&[CpuSvgSceneLayer::Vector(huge)], config).err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
    }

    #[test]
    fn at_r20_1_composition_output_cap_failure_is_atomic_and_empty_scene_recovers() {
        let layers = [
            CpuSvgSceneLayer::Image(result([1., -0., 0., 1., 0., 0.])),
            CpuSvgSceneLayer::Vector(solid_vector(1.)),
        ];
        let before = layers.clone();
        let mut config = configuration(1.);
        let first = prepare_cpu_svg_composition(&layers, config)
            .expect("original deterministic nonclipped scene");
        let length = first.svg().svg.len();
        config.max_svg_bytes = length;
        assert_eq!(
            prepare_cpu_svg_composition(&layers, config)
                .expect("exact cap")
                .svg()
                .svg,
            first.svg().svg
        );
        for maximum in [0, length - 1] {
            config.max_svg_bytes = maximum;
            assert_eq!(
                prepare_cpu_svg_composition(&layers, config).err(),
                Some(CpuSvgSceneError::SvgBudgetExceeded {
                    maximum,
                    actual: length
                })
            );
            assert_layers_unchanged(&layers, &before);
        }
        config.max_svg_bytes = length;
        assert_eq!(
            prepare_cpu_svg_composition(&layers, config)
                .expect("cap recovery")
                .svg()
                .svg,
            first.svg().svg
        );
        for (config, expected) in [
            (configuration(0.), CpuSvgSceneError::InvalidScale),
            (configuration(f64::NAN), CpuSvgSceneError::InvalidScale),
            (
                configuration(f64::from_bits(1)),
                CpuSvgSceneError::NonFiniteGeometry,
            ),
            (
                CpuSvgSceneConfiguration {
                    viewport: Footprint {
                        resolution: UVec2::ZERO,
                        ..configuration(1.).viewport
                    },
                    ..configuration(1.)
                },
                CpuSvgSceneError::EmptyViewport,
            ),
            (
                CpuSvgSceneConfiguration {
                    viewport: Footprint {
                        transform: DAffine2::from_cols_array(&[1., 0., 0., 1., f64::INFINITY, 0.]),
                        ..configuration(1.).viewport
                    },
                    ..configuration(1.)
                },
                CpuSvgSceneError::InvalidViewportTransform,
            ),
        ] {
            assert_eq!(
                prepare_cpu_svg_composition(&layers, config).err(),
                Some(expected)
            );
            assert_layers_unchanged(&layers, &before);
        }
        let empty = prepare_cpu_svg_composition(&[], configuration(1.))
            .expect("original empty Graphic list");
        assert!(empty.layers().is_empty());
        assert_eq!(
            empty.svg().svg,
            original_composition(&[], configuration(1.)).svg
        );
        assert!(empty.svg().image_data.is_empty());
        assert_mask_coherence(&empty.svg().svg, &[]);
    }

    #[test]
    fn at_r20_1_mask_opacity_checks_only_original_clipping_run_starts() {
        let image = result([2., 0.5, -0.25, 3., 7., 11.]);
        let overflowing_vector = solid_vector(0.5)
            .with_attribute(ATTR_OPACITY, -f64::MAX)
            .with_attribute(ATTR_OPACITY_FILL, 0_f64);
        let overflowing_image = with_item(
            &image,
            image
                .original_item()
                .clone()
                .with_attribute(ATTR_OPACITY, -f64::MAX)
                .with_attribute(ATTR_OPACITY_FILL, 0_f64),
        );
        let clipped =
            CpuSvgSceneLayer::Vector(solid_vector(1.).with_attribute(ATTR_CLIPPING_MASK, true));
        for masker in [
            CpuSvgSceneLayer::Vector(overflowing_vector.clone()),
            CpuSvgSceneLayer::Image(overflowing_image.clone()),
        ] {
            let unmasked = [masker.clone()];
            let original = original_composition(&unmasked, configuration(1.));
            assert!(original.svg.contains("opacity=\"-0\""));
            assert!(!original.svg.contains("opacity=\"-inf\""));
            assert_eq!(
                prepare_cpu_svg_composition(&unmasked, configuration(1.))
                    .expect("valid unmasked fill product remains admitted")
                    .svg()
                    .svg,
                original.svg,
            );
            let masked = [masker, clipped.clone()];
            let before = masked.clone();
            let original = original_composition(&masked, configuration(1.));
            let (defs, _) = original.svg.split_once("</defs>").expect("source defs");
            assert!(defs.contains("opacity=\"-inf\""));
            assert_mask_coherence(&original.svg, &[("mask", 1)]);
            assert_eq!(
                prepare_cpu_svg_composition(&masked, configuration(1.)).err(),
                Some(CpuSvgSceneError::NonFiniteMaskOpacity),
            );
            assert_layers_unchanged(&masked, &before);
        }
        let continued_run = [
            CpuSvgSceneLayer::Vector(solid_vector(0.5)),
            CpuSvgSceneLayer::Vector(
                overflowing_vector
                    .clone()
                    .with_attribute(ATTR_CLIPPING_MASK, true),
            ),
            clipped.clone(),
        ];
        let before = continued_run.clone();
        let prepared = prepare_cpu_svg_composition(&continued_run, configuration(1.))
            .expect("continued run does not make another original mask");
        assert_mask_coherence(&prepared.svg().svg, &[("mask", 2)]);
        assert!(!prepared.svg().svg.contains("opacity=\"-inf\""));
        assert!(prepared.svg().svg.contains("opacity=\"-0\""));
        assert_layers_unchanged(&continued_run, &before);
        let restarted_run = [
            continued_run[0].clone(),
            clipped.clone(),
            CpuSvgSceneLayer::Vector(overflowing_vector),
            clipped,
        ];
        let before = restarted_run.clone();
        let original = original_composition(&restarted_run, configuration(1.));
        assert!(original.svg.contains("opacity=\"-inf\""));
        assert_eq!(
            prepare_cpu_svg_composition(&restarted_run, configuration(1.)).err(),
            Some(CpuSvgSceneError::NonFiniteMaskOpacity),
        );
        assert_layers_unchanged(&restarted_run, &before);
        let recovered = [
            CpuSvgSceneLayer::Image(image),
            CpuSvgSceneLayer::Vector(solid_vector(1.).with_attribute(ATTR_CLIPPING_MASK, true)),
        ];
        assert_mask_coherence(
            &prepare_cpu_svg_composition(&recovered, configuration(1.))
                .expect("ordinary alpha mask recovery")
                .svg()
                .svg,
            &[("mask", 1)],
        );
    }

    #[test]
    fn at_r20_1_original_masks_retain_finite_singular_empty_and_extreme_geometry() {
        let image = result([2., 0.5, -0.25, 3., 7., 11.]);
        let clipped_image = with_item(
            &image,
            image
                .original_item()
                .clone()
                .with_attribute(ATTR_CLIPPING_MASK, true),
        );
        for scale in [1., 1e150, 1e-150] {
            for transform in [
                DAffine2::from_cols_array(&[0.; 6]),
                DAffine2::from_scale(DVec2::splat(1e150)),
                DAffine2::from_scale(DVec2::splat(1e-150)),
                DAffine2::from_cols_array(&[2., 0.5, -0.25, 3., 7., 11.]),
            ] {
                for alpha in [1., 0.5] {
                    let masker = solid_vector(alpha).with_attribute(ATTR_TRANSFORM, transform);
                    let path = original_vector_path(&masker);
                    let kind = if alpha == 1. { "clipPath" } else { "mask" };
                    let layers = [
                        CpuSvgSceneLayer::Vector(masker),
                        CpuSvgSceneLayer::Image(clipped_image.clone()),
                    ];
                    let before = layers.clone();
                    let config = configuration(scale);
                    let prepared = prepare_cpu_svg_composition(&layers, config)
                        .expect("finite singular and extreme original mask geometry");
                    assert_mask_coherence(&prepared.svg().svg, &[(kind, 1)]);
                    assert_mask_coherence(&original_composition(&layers, config).svg, &[(kind, 1)]);
                    assert!(prepared.svg().svg.contains(&path));
                    assert!(!prepared.svg().svg.contains("NaN"));
                    assert!(!prepared.svg().svg.contains("inf"));
                    assert_eq!(embedded_png(prepared.svg()), image.image().to_png());
                    assert_layers_unchanged(&layers, &before);
                }
            }
        }
        for alpha in [1., 0.5] {
            let mut empty = solid_vector(alpha);
            *empty.element_mut() = Vector::default();
            let kind = if alpha == 1. { "clipPath" } else { "mask" };
            let layers = [
                CpuSvgSceneLayer::Vector(empty),
                CpuSvgSceneLayer::Image(clipped_image.clone()),
            ];
            let before = layers.clone();
            let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                .expect("empty original vector mask remains explicit");
            assert_mask_coherence(&prepared.svg().svg, &[(kind, 1)]);
            let (defs, _) = prepared
                .svg()
                .svg
                .split_once("</defs>")
                .expect("source defs");
            assert!(defs.contains("d=\"\""));
            assert!(!prepared.svg().svg.contains("NaN"));
            assert!(!prepared.svg().svg.contains("inf"));
            assert_layers_unchanged(&layers, &before);
        }
        let raster_mask = [
            CpuSvgSceneLayer::Image(image.clone()),
            CpuSvgSceneLayer::Vector(solid_vector(1.).with_attribute(ATTR_CLIPPING_MASK, true)),
        ];
        let prepared = prepare_cpu_svg_composition(&raster_mask, configuration(1.))
            .expect("asymmetric original raster alpha mask");
        assert_mask_coherence(&prepared.svg().svg, &[("mask", 1)]);
        assert_eq!(embedded_png(prepared.svg()), image.image().to_png());
        assert!(prepared.svg().svg.contains("width=\"1\" height=\"1\""));
        assert!(prepared.svg().svg.contains("matrix(2,0.5,-0.25,3,7,11)"));
    }

    fn fill_and_stroke_vector(
        stroke: &Stroke,
        stroke_alpha: f32,
        stroke_below: bool,
    ) -> Item<Vector> {
        let mut vector = solid_vector(1.);
        let fill = graphite_graphic_types::Coverage::new_fill();
        let stroke_coverage = graphite_graphic_types::Coverage::new_stroke(stroke);
        let fill_paint = Graphic::from(Color::from_gamma_srgb_channels(0.8, 0.2, 0.1, 1.));
        let stroke_paint =
            Graphic::from(Color::from_gamma_srgb_channels(0.1, 0.3, 0.9, stroke_alpha));
        let (first, first_paint, second, second_paint) = if stroke_below {
            (stroke_coverage, stroke_paint, fill, fill_paint)
        } else {
            (fill, fill_paint, stroke_coverage, stroke_paint)
        };
        let mut appearance = Appearance::new_single(first, first_paint);
        appearance
            .0
            .push(Item::new_from_element(second).with_attribute(ATTR_PAINT, second_paint));
        vector.set_attribute(ATTR_APPEARANCE, appearance);
        vector
    }

    #[test]
    fn at_r20_1_solid_stroke_preserves_original_caps_joins_order_and_authoring_space() {
        let image = result([2., 0., 0., 3., 11., 13.]);
        for cap in [StrokeCap::Butt, StrokeCap::Square, StrokeCap::Round] {
            for join in [StrokeJoin::Miter, StrokeJoin::Bevel, StrokeJoin::Round] {
                for below in [false, true] {
                    let stroke = Stroke {
                        weight: 2.75,
                        cap,
                        join,
                        join_miter_limit: 6.,
                        dash_offset: -1.25,
                        transform: DAffine2::from_cols_array(&[2., 0.5, 0., 3., 4., 5.]),
                        ..Stroke::new(2.75)
                    };
                    let vector = fill_and_stroke_vector(&stroke, 0.625, below).with_attribute(
                        ATTR_TRANSFORM,
                        DAffine2::from_cols_array(&[1., 0.25, -0.5, 2., 7., 9.]),
                    );
                    let appearance = vector
                        .attribute::<Appearance>(ATTR_APPEARANCE)
                        .expect("explicit original appearance");
                    let extracted = appearance
                        .first_coverage_of(Cover::Stroke)
                        .expect("one original stroke")
                        .stroke_params();
                    assert_eq!(extracted, stroke);
                    assert_eq!(appearance.fill_and_stroke().stroke_below, below);
                    let layers = [
                        CpuSvgSceneLayer::Image(image.clone()),
                        CpuSvgSceneLayer::Vector(vector.clone()),
                    ];
                    let before = layers.clone();
                    let prepared = prepare_cpu_svg_composition(&layers, configuration(2.))
                        .expect("finite original solid fill and centered stroke");
                    assert_eq!(
                        prepared.svg().svg,
                        original_composition(&layers, configuration(2.)).svg
                    );
                    assert!(prepared.svg().svg.contains("stroke-width=\"2.75\""));
                    assert!(prepared.svg().svg.contains("stroke-miterlimit=\"6\""));
                    assert!(prepared.svg().svg.contains("stroke-dashoffset=\"-1.25\""));
                    assert_eq!(prepared.svg().svg.contains("paint-order: stroke;"), below);
                    if cap != StrokeCap::Butt {
                        assert!(
                            prepared
                                .svg()
                                .svg
                                .contains(&format!("stroke-linecap=\"{}\"", cap.svg_name()))
                        );
                    }
                    if join != StrokeJoin::Miter {
                        assert!(
                            prepared
                                .svg()
                                .svg
                                .contains(&format!("stroke-linejoin=\"{}\"", join.svg_name()))
                        );
                    }
                    let element_transform = vector
                        .attribute_cloned_or_default::<DAffine2>(ATTR_TRANSFORM)
                        * stroke.transform.inverse();
                    assert!(prepared.svg().svg.contains(&format!(
                        "transform=\"{}\"",
                        graphite_rendering::format_transform_matrix(element_transform)
                    )));
                    assert!(!prepared.svg().svg.contains("stroke-dasharray"));
                    assert_eq!(embedded_png(prepared.svg()), image.image().to_png());
                    assert_layers_unchanged(&layers, &before);
                    assert_layers_unchanged(prepared.layers(), &before);
                }
            }
        }
    }

    #[test]
    fn at_r20_1_solid_stroke_retains_original_sibling_mask_eligibility_and_empty_geometry() {
        let image = result([1., 0., 0., 1., 3., 5.]);
        let clipped = with_item(
            &image,
            image
                .original_item()
                .clone()
                .with_attribute(ATTR_CLIPPING_MASK, true),
        );
        for (weight, alpha, kind) in [
            (2., 1., "mask"),
            (2., 0.5, "mask"),
            (2., 0., "clipPath"),
            (0., 1., "clipPath"),
            (-2., 1., "clipPath"),
        ] {
            let stroke = Stroke::new(weight);
            let vector = fill_and_stroke_vector(&stroke, alpha, false);
            assert_eq!(
                Graphic::from(vector.clone()).can_reduce_to_clip_path(),
                kind == "clipPath"
            );
            let layers = [
                CpuSvgSceneLayer::Vector(vector),
                CpuSvgSceneLayer::Image(clipped.clone()),
            ];
            let before = layers.clone();
            let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                .expect("source solid stroke mask eligibility");
            assert_mask_coherence(&prepared.svg().svg, &[(kind, 1)]);
            assert_mask_coherence(
                &original_composition(&layers, configuration(1.)).svg,
                &[(kind, 1)],
            );
            assert_eq!(embedded_png(prepared.svg()), image.image().to_png());
            if weight <= 0. {
                assert!(!prepared.svg().svg.contains("stroke-width"));
            }
            assert_layers_unchanged(&layers, &before);
        }
        for weight in [0., -2.] {
            let stroke = Stroke {
                transform: DAffine2::from_cols_array(&[0.; 6]),
                ..Stroke::new(weight)
            };
            let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
                &stroke, 1., false,
            ))];
            let before = layers.clone();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.))
                    .expect(
                        "invisible original stroke does not select its singular authoring space"
                    )
                    .svg()
                    .svg,
                original_composition(&layers, configuration(1.)).svg,
            );
            assert_layers_unchanged(&layers, &before);
        }
        let mut empty = fill_and_stroke_vector(&Stroke::new(2.), 1., true);
        *empty.element_mut() = Vector::default();
        let layers = [CpuSvgSceneLayer::Vector(empty)];
        let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("explicit empty solid fill plus stroke");
        assert_eq!(
            prepared.svg().svg,
            original_composition(&layers, configuration(1.)).svg
        );
        assert!(prepared.svg().svg.contains("d=\"\""));
    }

    #[test]
    fn at_r20_1_solid_stroke_rejects_wrong_typed_nested_styles_and_unsupported_profiles() {
        let valid = fill_and_stroke_vector(&Stroke::new(2.), 1., false);
        for (key, value, expected) in [
            (
                ATTR_WEIGHT,
                "2",
                CpuSvgSceneError::WrongAttributeType(ATTR_WEIGHT.into()),
            ),
            (
                ATTR_CAP,
                "round",
                CpuSvgSceneError::WrongAttributeType(ATTR_CAP.into()),
            ),
            (
                ATTR_JOIN,
                "bevel",
                CpuSvgSceneError::WrongAttributeType(ATTR_JOIN.into()),
            ),
            (
                ATTR_JOIN_MITER_LIMIT,
                "4",
                CpuSvgSceneError::WrongAttributeType(ATTR_JOIN_MITER_LIMIT.into()),
            ),
            (
                ATTR_ALIGN,
                "center",
                CpuSvgSceneError::WrongAttributeType(ATTR_ALIGN.into()),
            ),
            (
                ATTR_DASH_PATTERN,
                "empty",
                CpuSvgSceneError::WrongAttributeType(ATTR_DASH_PATTERN.into()),
            ),
            (
                ATTR_DASH_OFFSET,
                "0",
                CpuSvgSceneError::WrongAttributeType(ATTR_DASH_OFFSET.into()),
            ),
            (
                ATTR_TRANSFORM,
                "identity",
                CpuSvgSceneError::WrongAttributeType(ATTR_TRANSFORM.into()),
            ),
            (
                "unsupported-stroke",
                "true",
                CpuSvgSceneError::UnsupportedAttribute("unsupported-stroke".into()),
            ),
        ] {
            let mut coverage = graphite_graphic_types::Coverage::new_stroke(&Stroke::new(2.));
            coverage.0.set_attribute(key, value.to_owned());
            let mut appearance = Appearance::new_single(
                graphite_graphic_types::Coverage::new_fill(),
                Graphic::from(Color::WHITE),
            );
            appearance.0.push(
                Item::new_from_element(coverage)
                    .with_attribute(ATTR_PAINT, Graphic::from(Color::BLACK)),
            );
            let vector = valid.clone().with_attribute(ATTR_APPEARANCE, appearance);
            let layers = [CpuSvgSceneLayer::Vector(vector)];
            let before = layers.clone();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(expected)
            );
            assert_layers_unchanged(&layers, &before);
        }
        for stroke in [
            Stroke {
                align: StrokeAlign::Inside,
                ..Stroke::new(2.)
            },
            Stroke {
                align: StrokeAlign::Outside,
                ..Stroke::new(2.)
            },
            Stroke {
                dash_lengths: vec![4., 2.],
                ..Stroke::new(2.)
            },
            Stroke {
                transform: DAffine2::from_cols_array(&[0.; 6]),
                ..Stroke::new(2.)
            },
            Stroke {
                transform: DAffine2::from_scale(DVec2::splat(1e-200)),
                ..Stroke::new(2.)
            },
        ] {
            let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
                &stroke, 1., false,
            ))];
            let before = layers.clone();
            if stroke.dash_lengths.is_empty() && stroke.align == StrokeAlign::Center {
                assert_eq!(
                    prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                    Some(CpuSvgSceneError::UnsupportedStrokeStyle)
                );
            } else {
                assert_eq!(
                    normalized_paint_ids(
                        &prepare_cpu_svg_composition(&layers, configuration(1.))
                            .expect("newly admitted original stroke style")
                            .svg()
                            .svg,
                    ),
                    normalized_paint_ids(&original_composition(&layers, configuration(1.)).svg)
                );
            }
            assert_layers_unchanged(&layers, &before);
        }
        let mut appearance = valid
            .attribute::<Appearance>(ATTR_APPEARANCE)
            .expect("explicit two rows")
            .clone();
        appearance.0.push(
            Item::new_from_element(graphite_graphic_types::Coverage::new_stroke(&Stroke::new(
                3.,
            )))
            .with_attribute(ATTR_PAINT, Graphic::from(Color::BLACK)),
        );
        assert_eq!(
            prepare_cpu_svg_composition(
                &[CpuSvgSceneLayer::Vector(
                    valid.clone().with_attribute(ATTR_APPEARANCE, appearance)
                )],
                configuration(1.)
            )
            .err(),
            Some(CpuSvgSceneError::UnsupportedVectorAppearance)
        );
        let mut duplicate = Appearance::new_single(
            graphite_graphic_types::Coverage::new_stroke(&Stroke::new(2.)),
            Graphic::from(Color::WHITE),
        );
        duplicate.0.push(
            Item::new_from_element(graphite_graphic_types::Coverage::new_stroke(&Stroke::new(
                3.,
            )))
            .with_attribute(ATTR_PAINT, Graphic::from(Color::BLACK)),
        );
        assert_eq!(
            prepare_cpu_svg_composition(
                &[CpuSvgSceneLayer::Vector(
                    valid.with_attribute(ATTR_APPEARANCE, duplicate)
                )],
                configuration(1.)
            )
            .err(),
            Some(CpuSvgSceneError::UnsupportedVectorAppearance)
        );
    }

    #[test]
    fn at_r20_1_solid_stroke_rejects_nonfinite_and_derived_overflow_then_recovers() {
        for (stroke, expected) in [
            (
                Stroke {
                    weight: f64::INFINITY,
                    ..Stroke::new(2.)
                },
                CpuSvgSceneError::NonFiniteStrokeParameter(ATTR_WEIGHT.into()),
            ),
            (
                Stroke {
                    dash_offset: f64::NEG_INFINITY,
                    ..Stroke::new(2.)
                },
                CpuSvgSceneError::NonFiniteStrokeParameter(ATTR_DASH_OFFSET.into()),
            ),
            (
                Stroke {
                    join_miter_limit: f64::INFINITY,
                    ..Stroke::new(2.)
                },
                CpuSvgSceneError::NonFiniteStrokeParameter(ATTR_JOIN_MITER_LIMIT.into()),
            ),
            (
                Stroke {
                    transform: DAffine2::from_cols_array(&[1., 0., 0., 1., f64::INFINITY, 0.]),
                    ..Stroke::new(2.)
                },
                CpuSvgSceneError::NonFiniteGeometry,
            ),
            (
                Stroke {
                    weight: 1e200,
                    join_miter_limit: 1e200,
                    ..Stroke::new(2.)
                },
                CpuSvgSceneError::NonFiniteGeometry,
            ),
            (
                Stroke {
                    transform: DAffine2::from_scale(DVec2::splat(1e200)),
                    ..Stroke::new(2.)
                },
                CpuSvgSceneError::NonFiniteGeometry,
            ),
        ] {
            let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
                &stroke, 1., false,
            ))];
            let before = layers.clone();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(expected)
            );
            assert_layers_unchanged(&layers, &before);
        }
        let mut huge = fill_and_stroke_vector(
            &Stroke {
                transform: DAffine2::from_scale(DVec2::splat(2.)),
                ..Stroke::new(2.)
            },
            1.,
            false,
        );
        for (_, point) in huge.element_mut().point_domain.positions_mut() {
            point.x = f64::MAX;
        }
        let layers = [CpuSvgSceneLayer::Vector(huge)];
        let before = layers.clone();
        assert_eq!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
        assert_layers_unchanged(&layers, &before);
        let recovered = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
            &Stroke::new(2.),
            1.,
            false,
        ))];
        let initial = prepare_cpu_svg_composition(&recovered, configuration(1.))
            .expect("finite original stroke recovery");
        let mut cap = configuration(1.);
        cap.max_svg_bytes = initial.svg().svg.len() - 1;
        assert_eq!(
            prepare_cpu_svg_composition(&recovered, cap).err(),
            Some(CpuSvgSceneError::SvgBudgetExceeded {
                maximum: cap.max_svg_bytes,
                actual: initial.svg().svg.len()
            })
        );
        cap.max_svg_bytes += 1;
        assert_eq!(
            prepare_cpu_svg_composition(&recovered, cap)
                .expect("exact stroke cap recovery")
                .svg()
                .svg,
            initial.svg().svg
        );
    }

    fn original_artboard(
        layers: &[CpuSvgSceneLayer],
        artboard: CpuSvgArtboardConfiguration,
        config: CpuSvgSceneConfiguration,
    ) -> SvgRenderOutput {
        let footprint = Footprint {
            transform: DAffine2::from_scale(DVec2::splat(config.scale)) * config.viewport.transform,
            ..config.viewport
        };
        let params = RenderParams {
            footprint,
            scale: config.scale,
            render_mode: RenderMode::Normal,
            for_export: true,
            for_mask: false,
            thumbnail: false,
            inside_pattern: false,
            render_output_type: RenderOutputType::Svg,
            artboard_background: Some(Color::TRANSPARENT),
            viewport_zoom: config.viewport.scale_magnitudes().x,
            ..RenderParams::default()
        };
        let mut graphics = List::<Graphic>::new();
        for layer in layers {
            let graphic = match layer {
                CpuSvgSceneLayer::Image(image) => Graphic::from(image.original_item().clone()),
                CpuSvgSceneLayer::SourceImage(source) => Graphic::from(
                    Item::new_from_element(Raster::new_cpu(source.image().clone())).with_attribute(
                        ATTR_TRANSFORM,
                        DAffine2::from_cols_array(&source.descriptor().transform),
                    ),
                ),
                CpuSvgSceneLayer::Vector(vector) => Graphic::from(vector.clone()),
                CpuSvgSceneLayer::Text(text) => Graphic::from(text.clone()),
                CpuSvgSceneLayer::TextList(texts) => Graphic::from(texts.clone()),
                CpuSvgSceneLayer::Color(color) => Graphic::from(color.clone()),
                CpuSvgSceneLayer::ColorList(colors) => Graphic::from(colors.clone()),
                CpuSvgSceneLayer::Gradient(gradient) => Graphic::from(gradient.clone()),
                CpuSvgSceneLayer::GradientList(gradients) => Graphic::from(gradients.clone()),
            };
            graphics.push(Item::new_from_element(graphic));
        }
        let item = Item::new_from_element(Artboard::new(graphics))
            .with_attribute(ATTR_LOCATION, artboard.location)
            .with_attribute(ATTR_DIMENSIONS, artboard.dimensions)
            .with_attribute(ATTR_BACKGROUND, artboard.background)
            .with_attribute(ATTR_CLIP, artboard.clip);
        let mut render = SvgRender::new();
        List::<Artboard>::from(item).render_svg(&mut render, &params);
        let intermediate = SvgRenderOutput::from(render);
        let mut render = SvgRender::from(&intermediate);
        render.wrap_with_transform(
            DAffine2::from_scale(DVec2::splat(1. / config.scale)) * footprint.transform,
            Some(footprint.resolution.as_dvec2() / config.scale),
        );
        SvgRenderOutput::from(render)
    }

    fn normalize_artboard_id(svg: &str) -> String {
        let (_, rest) = svg
            .split_once("<clipPath id=\"")
            .expect("original artboard clip definition");
        let (identity, _) = rest.split_once('"').expect("clip ID terminator");
        assert!(identity.starts_with("artboard-"));
        assert_eq!(svg.matches("<clipPath id=").count(), 1);
        assert_eq!(
            svg.matches(&format!("clip-path=\"url(#{identity})\""))
                .count(),
            1
        );
        svg.replace(identity, "artboard-current-test-id")
    }

    #[test]
    fn at_r20_1_artboard_retains_original_typed_background_translation_clip_and_children() {
        let source = result([2., 0.5, -0.25, 1., 30., -4.]);
        let layers = [
            CpuSvgSceneLayer::Image(source.clone()),
            CpuSvgSceneLayer::Vector(solid_vector(0.5)),
        ];
        let before = layers.clone();
        for background in [
            Color::TRANSPARENT,
            Color::WHITE,
            Color::from_gamma_srgb_channels(0.2, 0.4, 0.8, 0.6251),
        ] {
            for clip in [false, true] {
                let artboard = CpuSvgArtboardConfiguration {
                    location: DVec2::new(-3.5, 4.75),
                    dimensions: DVec2::new(16.5, 9.25),
                    background,
                    clip,
                };
                let config = configuration(2.);
                let prepared = prepare_cpu_svg_artboard(&layers, artboard, config)
                    .expect("normalized render-only artboard");
                let original = original_artboard(&layers, artboard, config);
                if clip {
                    assert_eq!(
                        normalize_artboard_id(&prepared.svg().svg),
                        normalize_artboard_id(&original.svg)
                    );
                    assert!(
                        prepared
                            .svg()
                            .svg
                            .contains("<rect x=\"0\" y=\"0\" width=\"16.5\" height=\"9.25\"")
                    );
                } else {
                    assert_eq!(prepared.svg().svg, original.svg);
                    assert!(!prepared.svg().svg.contains("clip-path="));
                }
                assert!(prepared.svg().svg.contains("matrix(1,0,0,1,-3.5,4.75)"));
                assert!(
                    prepared
                        .svg()
                        .svg
                        .contains("width=\"16.5\" height=\"9.25\"")
                );
                if background.a() == 0.6251 {
                    assert!(prepared.svg().svg.contains("fill-opacity=\"0.625\""));
                }
                assert_eq!(embedded_png(prepared.svg()), source.image().to_png());
                assert_layers_unchanged(prepared.layers(), &before);
                assert_layers_unchanged(&layers, &before);
                assert_eq!(prepared.artboard().location, artboard.location);
                assert_eq!(prepared.artboard().dimensions, artboard.dimensions);
                assert_eq!(prepared.artboard().background, background);
                assert_eq!(prepared.artboard().clip, clip);
                assert_eq!(
                    prepared.configuration().viewport.resolution,
                    config.viewport.resolution
                );
                assert_eq!(
                    prepared.configuration().scale.to_bits(),
                    config.scale.to_bits()
                );
                assert_eq!(
                    prepared.source_revision(),
                    crate::resources::GRAPHITE_IMAGE_SOURCE_REVISION
                );
            }
        }
    }

    #[test]
    fn at_r20_1_artboard_rejects_non_normalized_nonfinite_and_derived_overflow_without_mutation() {
        let layers = [CpuSvgSceneLayer::Image(result([1., 0., 0., 1., 0., 0.]))];
        let before = layers.clone();
        let valid = CpuSvgArtboardConfiguration {
            location: DVec2::ZERO,
            dimensions: DVec2::new(16., 8.),
            background: Color::WHITE,
            clip: false,
        };
        for dimensions in [
            DVec2::ZERO,
            DVec2::new(-2., 8.),
            DVec2::new(0.5, 8.),
            DVec2::new(f64::NAN, 8.),
            DVec2::new(8., f64::INFINITY),
        ] {
            assert_eq!(
                prepare_cpu_svg_artboard(
                    &layers,
                    CpuSvgArtboardConfiguration {
                        dimensions,
                        ..valid
                    },
                    configuration(1.)
                )
                .err(),
                Some(CpuSvgSceneError::InvalidArtboardGeometry)
            );
        }
        for location in [DVec2::new(f64::NAN, 0.), DVec2::new(0., f64::INFINITY)] {
            assert_eq!(
                prepare_cpu_svg_artboard(
                    &layers,
                    CpuSvgArtboardConfiguration { location, ..valid },
                    configuration(1.)
                )
                .err(),
                Some(CpuSvgSceneError::InvalidArtboardGeometry)
            );
        }
        assert_eq!(
            prepare_cpu_svg_artboard(
                &layers,
                CpuSvgArtboardConfiguration {
                    location: DVec2::splat(f64::MAX),
                    dimensions: DVec2::splat(f64::MAX),
                    ..valid
                },
                configuration(1.)
            )
            .err(),
            Some(CpuSvgSceneError::InvalidArtboardGeometry)
        );
        for background in [
            Color::from_rgbaf32_unchecked(f32::NAN, 0., 0., 1.),
            Color::from_rgbaf32_unchecked(0., 0., 0., -f32::MAX),
        ] {
            assert_eq!(
                prepare_cpu_svg_artboard(
                    &layers,
                    CpuSvgArtboardConfiguration {
                        background,
                        ..valid
                    },
                    configuration(1.)
                )
                .err(),
                Some(CpuSvgSceneError::NonFinitePaint)
            );
        }
        let mut config = configuration(1.);
        config.viewport.transform = DAffine2::from_scale(DVec2::splat(f64::MAX / 4.));
        assert_eq!(
            prepare_cpu_svg_artboard(&layers, valid, config).err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
        let bad_item = match &layers[0] {
            CpuSvgSceneLayer::Image(source) => with_item(
                source,
                source
                    .original_item()
                    .clone()
                    .with_attribute(ATTR_OPACITY, true),
            ),
            _ => panic!("explicit image fixture"),
        };
        assert_eq!(
            prepare_cpu_svg_artboard(
                &[CpuSvgSceneLayer::Image(bad_item)],
                valid,
                configuration(1.)
            )
            .err(),
            Some(CpuSvgSceneError::WrongAttributeType(ATTR_OPACITY.into()))
        );
        assert!(prepare_cpu_svg_artboard(&layers, valid, configuration(1.)).is_ok());
        assert_layers_unchanged(&layers, &before);
    }

    #[test]
    fn at_r20_1_artboard_empty_content_and_exact_output_cap_recover_without_replacing_owners() {
        let artboard = CpuSvgArtboardConfiguration {
            location: DVec2::new(3., -2.),
            dimensions: DVec2::new(12.5, 8.),
            background: Color::TRANSPARENT,
            clip: false,
        };
        let layers = [CpuSvgSceneLayer::Image(result([1., 0., 0., 1., 0., 0.]))];
        let before = layers.clone();
        let initial = prepare_cpu_svg_artboard(&layers, artboard, configuration(1.))
            .expect("original artboard");
        let mut config = configuration(1.);
        config.max_svg_bytes = initial.svg().svg.len() - 1;
        assert_eq!(
            prepare_cpu_svg_artboard(&layers, artboard, config).err(),
            Some(CpuSvgSceneError::SvgBudgetExceeded {
                maximum: config.max_svg_bytes,
                actual: initial.svg().svg.len()
            })
        );
        config.max_svg_bytes += 1;
        assert_eq!(
            prepare_cpu_svg_artboard(&layers, artboard, config)
                .expect("exact cap recovery")
                .svg()
                .svg,
            initial.svg().svg
        );
        assert_layers_unchanged(&layers, &before);
        assert_layers_unchanged(initial.layers(), &before);
        let empty = prepare_cpu_svg_artboard(&[], artboard, configuration(1.))
            .expect("original empty content retains background");
        assert!(empty.layers().is_empty());
        assert_eq!(
            empty.svg().svg,
            original_artboard(&[], artboard, configuration(1.)).svg
        );
        assert!(empty.svg().svg.contains("fill-opacity=\"0\""));
    }

    fn gradient_vector(paint: Graphic) -> Item<Vector> {
        solid_vector(1.).with_attribute(
            ATTR_APPEARANCE,
            Appearance::new_single(graphite_graphic_types::Coverage::new_fill(), paint),
        )
    }

    fn gradient_fixture() -> Gradient {
        use graphite_vector_types::GradientStop;
        Gradient::new([
            GradientStop {
                position: 0.8,
                midpoint: 0.3,
                color: Color::from_gamma_srgb_channels(0.1, 0.2, 0.9, 0.5),
            },
            GradientStop {
                position: -0.2,
                midpoint: 0.75,
                color: Color::from_gamma_srgb_channels(0.9, 0.2, 0.1, 1.),
            },
            GradientStop {
                position: 0.8,
                midpoint: 0.5,
                color: Color::from_gamma_srgb_channels(0.2, 0.8, 0.3, 0.25),
            },
        ])
    }

    fn normalized_paint_ids(svg: &str) -> String {
        let identities = svg
            .match_indices(" id=\"")
            .map(|(position, _)| {
                svg.get(position + 5..)
                    .expect("original def marker")
                    .split_once('"')
                    .expect("original def terminator")
                    .0
                    .to_owned()
            })
            .collect::<Vec<_>>();
        let mut unique = std::collections::HashSet::new();
        let mut output = svg.to_owned();
        for (index, identity) in identities.iter().enumerate() {
            assert!(
                unique.insert(identity),
                "original generated definitions must be distinct"
            );
            assert!(
                svg.contains(&format!("url(#{identity})"))
                    || svg.contains(&format!("url('#{identity}')")),
                "each paint/mask definition is referenced"
            );
            output = output
                .replace(
                    &format!("id=\"{identity}\""),
                    &format!("id=\"current-def-{index}\""),
                )
                .replace(
                    &format!("url(#{identity})"),
                    &format!("url(#current-def-{index})"),
                )
                .replace(
                    &format!("url('#{identity}')"),
                    &format!("url('#current-def-{index}')"),
                );
        }
        output
    }

    #[test]
    fn at_r20_1_gradient_item_list_settings_and_source_sampling_remain_exact() {
        for form in [GradientForm::Linear, GradientForm::Radial] {
            for spread in [
                GradientSpread::Pad,
                GradientSpread::Reflect,
                GradientSpread::Repeat,
                GradientSpread::Clear,
            ] {
                for interpolation in [
                    GradientInterpolation::Linear,
                    GradientInterpolation::Stepped,
                    GradientInterpolation::Smooth,
                ] {
                    for cyclic in [false, true] {
                        let item = Item::new_from_element(gradient_fixture())
                            .with_attribute(
                                ATTR_TRANSFORM,
                                DAffine2::from_cols_array(&[3., 0.5, 1., 2., -2., 4.]),
                            )
                            .with_attribute(ATTR_GRADIENT_FORM, form)
                            .with_attribute(ATTR_GRADIENT_SPREAD, spread)
                            .with_attribute(ATTR_GRADIENT_INTERPOLATION, interpolation)
                            .with_attribute(ATTR_GRADIENT_CYCLIC, cyclic)
                            .with_attribute(ATTR_GRADIENT_SPACE, GradientSpace::OkLab)
                            .with_attribute(
                                ATTR_GRADIENT_HUE_DIRECTION,
                                GradientHueDirection::Shorter,
                            )
                            .with_attribute(ATTR_OPACITY, 0.5_f64)
                            .with_attribute(ATTR_OPACITY_FILL, 0.75_f64);
                        let mut item_svg = None;
                        for paint in [
                            Graphic::from(item.clone()),
                            Graphic::from(List::<Gradient>::from(item.clone())),
                        ] {
                            let layers = [CpuSvgSceneLayer::Vector(
                                gradient_vector(paint).with_attribute(
                                    ATTR_TRANSFORM,
                                    DAffine2::from_cols_array(&[2., 1., -0.5, 3., 7., -3.]),
                                ),
                            )];
                            let before = layers.clone();
                            let config = configuration(2.);
                            let prepared = prepare_cpu_svg_composition(&layers, config)
                                .expect("original typed gradient profile");
                            let normalized = normalized_paint_ids(&prepared.svg().svg);
                            assert_eq!(
                                normalized,
                                normalized_paint_ids(&original_composition(&layers, config).svg)
                            );
                            if let Some(expected) = &item_svg {
                                assert_eq!(&normalized, expected);
                            } else {
                                item_svg = Some(normalized);
                            }
                            assert_layers_unchanged(prepared.layers(), &before);
                            assert_layers_unchanged(&layers, &before);
                            assert!(!prepared.svg().svg.contains("NaN"));
                            assert!(!prepared.svg().svg.contains("inf"));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn at_r20_1_gradient_defaults_empty_semantics_and_finite_singular_placement_match_source() {
        for transform in [
            DAffine2::IDENTITY,
            DAffine2::from_cols_array(&[1., 1., 1., 1., 4., 2.]),
            DAffine2::from_cols_array(&[0.; 6]),
        ] {
            for paint in [
                Graphic::from(Gradient::black_to_white()),
                Graphic::from(Gradient::default()),
                Graphic::from(List::<Gradient>::new()),
                Graphic::from(
                    Item::new_from_element(Gradient::default())
                        .with_attribute(ATTR_GRADIENT_SPREAD, GradientSpread::Clear),
                ),
            ] {
                let empty_list = matches!(&paint,Graphic::GradientList(list) if list.is_empty());
                let empty_gradient =
                    matches!(&paint,Graphic::Gradient(item) if item.element().0.is_empty());
                let layers = [CpuSvgSceneLayer::Vector(
                    gradient_vector(paint).with_attribute(ATTR_TRANSFORM, transform),
                )];
                let before = layers.clone();
                let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                    .expect("source default or finite singular gradient");
                assert_eq!(
                    normalized_paint_ids(&prepared.svg().svg),
                    normalized_paint_ids(&original_composition(&layers, configuration(1.)).svg)
                );
                if empty_list {
                    assert!(prepared.svg().svg.contains("fill=\"none\""));
                    assert!(!prepared.svg().svg.contains("<linearGradient"));
                }
                if empty_gradient {
                    assert!(prepared.svg().svg.contains("stop-color=\"#000000\""));
                }
                assert_layers_unchanged(&layers, &before);
            }
        }
        let raw = Item::new_from_element(Gradient::black_to_white());
        let explicit = raw
            .clone()
            .with_attribute(ATTR_GRADIENT_FORM, GradientForm::Linear)
            .with_attribute(ATTR_GRADIENT_SPREAD, GradientSpread::Pad)
            .with_attribute(ATTR_GRADIENT_CYCLIC, false)
            .with_attribute(ATTR_GRADIENT_SPACE, GradientSpace::OkLab)
            .with_attribute(ATTR_GRADIENT_HUE_DIRECTION, GradientHueDirection::Shorter)
            .with_attribute(ATTR_GRADIENT_INTERPOLATION, GradientInterpolation::Linear)
            .with_attribute(ATTR_TRANSFORM, DAffine2::IDENTITY)
            .with_attribute(ATTR_OPACITY, 1_f64)
            .with_attribute(ATTR_OPACITY_FILL, 1_f64);
        let render = |item| {
            prepare_cpu_svg_composition(
                &[CpuSvgSceneLayer::Vector(gradient_vector(Graphic::from(
                    item,
                )))],
                configuration(1.),
            )
            .expect("source defaults")
        };
        assert_eq!(
            normalized_paint_ids(&render(raw).svg().svg),
            normalized_paint_ids(&render(explicit).svg().svg)
        );
    }

    #[test]
    fn at_r20_1_gradient_fill_and_centered_stroke_share_source_transform_semantics() {
        for form in [GradientForm::Linear, GradientForm::Radial] {
            for stroke_transform in [
                DAffine2::IDENTITY,
                DAffine2::from_cols_array(&[1., 0.25, -0.5, 2., 3., 5.]),
            ] {
                let paint = Graphic::from(
                    Item::new_from_element(gradient_fixture())
                        .with_attribute(ATTR_GRADIENT_FORM, form)
                        .with_attribute(
                            ATTR_TRANSFORM,
                            DAffine2::from_cols_array(&[2., 1., 0.5, 3., -1., 2.]),
                        ),
                );
                let stroke = Stroke {
                    weight: 2.,
                    transform: stroke_transform,
                    ..Stroke::default()
                };
                let mut appearance = Appearance::new_single(
                    graphite_graphic_types::Coverage::new_fill(),
                    paint.clone(),
                );
                appearance.replace_or_insert(
                    graphite_graphic_types::Coverage::new_stroke(&stroke),
                    paint,
                    graphite_graphic_types::CoverPlacement::Above,
                );
                let layers = [CpuSvgSceneLayer::Vector(
                    solid_vector(1.)
                        .with_attribute(ATTR_APPEARANCE, appearance)
                        .with_attribute(
                            ATTR_TRANSFORM,
                            DAffine2::from_cols_array(&[2., 0.5, -0.25, 3., 7., 9.]),
                        ),
                )];
                let before = layers.clone();
                let prepared = prepare_cpu_svg_composition(&layers, configuration(2.))
                    .expect("original gradient fill and stroke");
                assert_eq!(
                    normalized_paint_ids(&prepared.svg().svg),
                    normalized_paint_ids(&original_composition(&layers, configuration(2.)).svg)
                );
                assert!(prepared.svg().svg.contains("stroke=\"url(#"));
                assert!(prepared.svg().svg.contains("fill=\"url(#"));
                assert_layers_unchanged(&layers, &before);
                assert_layers_unchanged(prepared.layers(), &before);
            }
        }
    }

    fn gradient_preservation_snapshot(item: &Item<Vector>) -> (Item<Vector>, Vec<u64>) {
        let mut snapshot = item.clone();
        let mut bits = Vec::new();
        let mut appearance = item
            .attribute::<Appearance>(ATTR_APPEARANCE)
            .expect("explicit appearance")
            .clone();
        if let Some(Graphic::Gradient(gradient)) = appearance.paint_at(0) {
            let mut gradient = gradient.clone();
            if let Some(transform) = gradient.attribute::<DAffine2>(ATTR_TRANSFORM) {
                let mut values = transform.to_cols_array();
                bits.extend(values.map(f64::to_bits));
                for value in &mut values {
                    if value.is_nan() {
                        *value = 0.;
                    }
                }
                gradient.set_attribute(ATTR_TRANSFORM, DAffine2::from_cols_array(&values));
            }
            for key in gradient.attributes().keys() {
                if let Some(value) = gradient.attribute::<f64>(key) {
                    bits.push(value.to_bits());
                }
            }
            for key in [ATTR_POSITION, ATTR_MIDPOINT] {
                if let Some(values) = gradient
                    .element_mut()
                    .0
                    .iter_attribute_values_mut::<f64>(key)
                {
                    for value in values {
                        bits.push(value.to_bits());
                        if value.is_nan() {
                            *value = 0.;
                        }
                    }
                }
            }
            for color in gradient.element_mut().0.iter_element_values_mut() {
                let mut values = [color.r(), color.g(), color.b(), color.a()];
                bits.extend(values.map(|value| u64::from(value.to_bits())));
                for value in &mut values {
                    if value.is_nan() {
                        *value = 0.;
                    }
                }
                *color = Color::from_rgbaf32_unchecked(values[0], values[1], values[2], values[3]);
            }
            assert!(appearance.set_paint_of(Cover::Fill, Graphic::Gradient(gradient)));
            snapshot.set_attribute(ATTR_APPEARANCE, appearance);
        }
        (snapshot, bits)
    }

    #[test]
    fn at_r20_1_gradient_rejects_wrong_types_unknown_fields_patterns_and_nonfinite_work() {
        let valid = Item::new_from_element(gradient_fixture());
        let mut wrong_position = gradient_fixture();
        wrong_position.0.remove_attribute(ATTR_POSITION);
        wrong_position.0.set_attribute(ATTR_POSITION, 0, true);
        let mut unknown_stop = gradient_fixture();
        unknown_stop.0.set_attribute("unknown-stop", 0, 1_f64);
        let mut infinite_stop = gradient_fixture();
        infinite_stop
            .0
            .set_attribute(ATTR_MIDPOINT, 0, f64::INFINITY);
        let mut malformed_color = Gradient::black_to_white();
        *malformed_color
            .0
            .element_mut(0)
            .expect("actual source color") = Color::from_rgbaf32_unchecked(f32::NAN, 0., 0., 1.);
        for (paint, expected) in [
            (
                Graphic::from(valid.clone().with_attribute(ATTR_GRADIENT_FORM, false)),
                CpuSvgSceneError::WrongAttributeType(ATTR_GRADIENT_FORM.into()),
            ),
            (
                Graphic::from(valid.clone().with_attribute("unknown-ramp", true)),
                CpuSvgSceneError::UnsupportedAttribute("unknown-ramp".into()),
            ),
            (
                Graphic::from(wrong_position),
                CpuSvgSceneError::WrongAttributeType(ATTR_POSITION.into()),
            ),
            (
                Graphic::from(unknown_stop),
                CpuSvgSceneError::UnsupportedAttribute("unknown-stop".into()),
            ),
            (
                Graphic::from(infinite_stop),
                CpuSvgSceneError::InvalidGradient,
            ),
            (
                Graphic::from(malformed_color),
                CpuSvgSceneError::NonFinitePaint,
            ),
            (
                Graphic::from(valid.clone().with_attribute(ATTR_OPACITY, f64::MAX)),
                CpuSvgSceneError::InvalidGradient,
            ),
            (
                Graphic::from(
                    valid
                        .clone()
                        .with_attribute(ATTR_TRANSFORM, DAffine2::from_cols_array(&[f64::NAN; 6])),
                ),
                CpuSvgSceneError::InvalidGradient,
            ),
            (
                Graphic::from(List::from_iter([valid.clone(), valid.clone()])),
                CpuSvgSceneError::UnsupportedVectorPaint,
            ),
        ] {
            let layers = [CpuSvgSceneLayer::Vector(gradient_vector(paint))];
            let before = layers.clone();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(expected)
            );
            assert_eq!(layers.len(), before.len());
            for (actual, before) in layers.iter().zip(&before) {
                match (actual, before) {
                    (CpuSvgSceneLayer::Vector(actual), CpuSvgSceneLayer::Vector(before)) => {
                        assert_eq!(
                            gradient_preservation_snapshot(actual),
                            gradient_preservation_snapshot(before)
                        );
                    }
                    _ => panic!("explicit vector failure fixtures preserve their variant"),
                }
            }
        }
        let overflow =
            valid.with_attribute(ATTR_TRANSFORM, DAffine2::from_scale(DVec2::splat(f64::MAX)));
        let layers = [CpuSvgSceneLayer::Vector(
            gradient_vector(Graphic::from(overflow))
                .with_attribute(ATTR_TRANSFORM, DAffine2::from_scale(DVec2::splat(2.))),
        )];
        assert_eq!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::InvalidGradient)
        );
        let fallback = Item::new_from_element(Gradient::black_to_white()).with_attribute(
            ATTR_TRANSFORM,
            DAffine2::from_scale(DVec2::new(1e160, 1e-160)),
        );
        let tiny_normal = DVec2::new(0., 1e-160).perp();
        assert!(
            DVec2::new(1e160, 0.)
                .try_project_onto(tiny_normal)
                .is_none(),
            "actual original fallible projection takes the axis fallback"
        );
        let layers = [CpuSvgSceneLayer::Vector(gradient_vector(Graphic::from(
            fallback,
        )))];
        let mut config = configuration(1.);
        config.viewport.transform = DAffine2::from_scale(DVec2::new(1., 1e160));
        assert_eq!(
            prepare_cpu_svg_composition(&layers, config).err(),
            Some(CpuSvgSceneError::InvalidGradient)
        );
    }

    #[test]
    fn at_r20_1_gradient_mask_opacity_and_output_cap_preserve_source_and_recover() {
        let paint = Graphic::from(
            Item::new_from_element(Gradient::black_to_white())
                .with_attribute(ATTR_OPACITY, f64::MAX)
                .with_attribute(ATTR_OPACITY_FILL, 0_f64),
        );
        let vector = gradient_vector(paint);
        let unmasked = [CpuSvgSceneLayer::Vector(vector.clone())];
        assert_eq!(
            normalized_paint_ids(
                &prepare_cpu_svg_composition(&unmasked, configuration(1.))
                    .expect("ordinary zero product remains finite")
                    .svg()
                    .svg
            ),
            normalized_paint_ids(&original_composition(&unmasked, configuration(1.)).svg)
        );
        let source = result([1., 0., 0., 1., 0., 0.]);
        let clipped = with_item(
            &source,
            source
                .original_item()
                .clone()
                .with_attribute(ATTR_CLIPPING_MASK, true),
        );
        assert_eq!(
            prepare_cpu_svg_composition(
                &[
                    CpuSvgSceneLayer::Vector(vector),
                    CpuSvgSceneLayer::Image(clipped.clone())
                ],
                configuration(1.)
            )
            .err(),
            Some(CpuSvgSceneError::InvalidGradient)
        );
        let paint = Graphic::from(
            Item::new_from_element(gradient_fixture())
                .with_attribute(ATTR_OPACITY, 0.5_f64)
                .with_attribute(ATTR_OPACITY_FILL, 0.25_f64),
        );
        let layers = [
            CpuSvgSceneLayer::Vector(gradient_vector(paint)),
            CpuSvgSceneLayer::Image(clipped),
        ];
        let before = layers.clone();
        let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("actual source mask paint opacity");
        assert_eq!(
            normalized_paint_ids(&prepared.svg().svg),
            normalized_paint_ids(&original_composition(&layers, configuration(1.)).svg)
        );
        let mut config = configuration(1.);
        config.max_svg_bytes = 0;
        assert!(
            matches!(prepare_cpu_svg_composition(&layers,config).err(),Some(CpuSvgSceneError::SvgBudgetExceeded {maximum:0,actual}) if actual>0)
        );
        config.max_svg_bytes = 32 * 1024;
        let recovered =
            prepare_cpu_svg_composition(&layers, config).expect("gradient cap recovery");
        assert_eq!(
            normalized_paint_ids(&prepared.svg().svg),
            normalized_paint_ids(&recovered.svg().svg)
        );
        assert_layers_unchanged(&layers, &before);
        assert_layers_unchanged(recovered.layers(), &before);
        assert_eq!(embedded_png(recovered.svg()), source.image().to_png());
    }

    fn dash_preservation_snapshot(vector: &Item<Vector>) -> (Item<Vector>, Vec<Vec<u64>>) {
        let mut normalized = vector.clone();
        let mut bits = Vec::new();
        if let Some(appearance) = normalized.attribute_mut::<Appearance>(ATTR_APPEARANCE) {
            for index in 0..appearance.len() {
                let coverage = appearance.0.element_mut(index).expect("owned coverage");
                if let Some(pattern) = coverage.0.attribute_mut::<DashPattern>(ATTR_DASH_PATTERN) {
                    bits.push(
                        pattern
                            .0
                            .iter_element_values()
                            .map(|length| length.to_bits())
                            .collect(),
                    );
                    for index in 0..pattern.0.len() {
                        let length = pattern.0.element_mut(index).expect("owned dash length");
                        if length.is_nan() {
                            *length = 0.;
                        }
                    }
                }
            }
        }
        (normalized, bits)
    }

    #[test]
    fn at_r20_1_dashes_preserve_original_signed_odd_zero_and_transformed_stroke_space() {
        for lengths in [
            vec![],
            vec![4., 2.],
            vec![3., 1., 2.],
            vec![0., 2., 0.],
            vec![-0., 0., 2.],
            vec![-4., 2.],
            vec![0., 0.],
            vec![-4., -2.],
        ] {
            for offset in [-60., -1.25, -0., 0., 7., 60.] {
                for stroke_below in [false, true] {
                    let stroke = Stroke {
                        dash_lengths: lengths.clone(),
                        dash_offset: offset,
                        transform: DAffine2::from_cols_array(&[2., 0.5, -0.25, 3., 0., 0.]),
                        cap: StrokeCap::Round,
                        join: StrokeJoin::Bevel,
                        ..Stroke::new(2.)
                    };
                    let layers = [CpuSvgSceneLayer::Vector(
                        fill_and_stroke_vector(&stroke, 0.75, stroke_below).with_attribute(
                            ATTR_TRANSFORM,
                            DAffine2::from_cols_array(&[1., 0.25, -0.5, 2., 7., -3.]),
                        ),
                    )];
                    let before = layers.clone();
                    let mut config = configuration(2.);
                    config.viewport.transform =
                        DAffine2::from_cols_array(&[2., -0.25, 0.5, 1., -5., 3.]);
                    let prepared =
                        prepare_cpu_svg_composition(&layers, config).expect("source dashed stroke");
                    assert_eq!(
                        prepared.svg().svg,
                        original_composition(&layers, config).svg
                    );
                    if lengths.is_empty() {
                        assert!(!prepared.svg().svg.contains("stroke-dasharray"));
                    } else {
                        let expected = lengths
                            .iter()
                            .map(|length| length.max(0.).to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        assert!(
                            prepared
                                .svg()
                                .svg
                                .contains(&format!("stroke-dasharray=\"{expected}\""))
                        );
                    }
                    if offset == 0. {
                        assert!(!prepared.svg().svg.contains("stroke-dashoffset"));
                    } else {
                        assert!(
                            prepared
                                .svg()
                                .svg
                                .contains(&format!("stroke-dashoffset=\"{offset}\""))
                        );
                    }
                    assert!(!prepared.svg().svg.contains("NaN"));
                    assert!(!prepared.svg().svg.contains("inf"));
                    assert_layers_unchanged(prepared.layers(), &before);
                    assert_layers_unchanged(&layers, &before);
                }
            }
        }
    }

    #[test]
    fn at_r20_1_dash_nonfinite_and_effective_period_fail_before_source_bounds_and_recover() {
        for lengths in [
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![f64::NEG_INFINITY],
            vec![f64::MAX],
            vec![f64::MAX, f64::MAX],
        ] {
            let stroke = Stroke {
                dash_lengths: lengths,
                dash_offset: -1.,
                ..Stroke::new(2.)
            };
            let vector = fill_and_stroke_vector(&stroke, 1., false);
            let before = dash_preservation_snapshot(&vector);
            let layers = [CpuSvgSceneLayer::Vector(vector)];
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(CpuSvgSceneError::NonFiniteStrokeParameter(
                    ATTR_DASH_PATTERN.into()
                ))
            );
            let CpuSvgSceneLayer::Vector(vector) = &layers[0] else {
                panic!("owned vector")
            };
            assert_eq!(dash_preservation_snapshot(vector), before);
        }
        for weight in [0., -1.] {
            let stroke = Stroke {
                dash_lengths: vec![f64::MAX],
                dash_offset: -1.,
                ..Stroke::new(weight)
            };
            let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
                &stroke, 1., false,
            ))];
            let before = layers.clone();
            let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                .expect("invisible stroke does not evaluate dash bounds");
            assert_eq!(
                prepared.svg().svg,
                original_composition(&layers, configuration(1.)).svg
            );
            assert!(!prepared.svg().svg.contains("stroke-dasharray"));
            assert_layers_unchanged(&layers, &before);
        }
        let stroke = Stroke {
            dash_lengths: vec![f64::MAX / 4., f64::MAX / 4.],
            dash_offset: -1.,
            ..Stroke::new(2.)
        };
        let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
            &stroke, 1., false,
        ))];
        let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("large finite period remains admitted");
        assert_eq!(
            prepared.svg().svg,
            original_composition(&layers, configuration(1.)).svg
        );
        let ordinary = Stroke {
            dash_lengths: vec![4., -2.],
            dash_offset: -1.25,
            ..Stroke::new(2.)
        };
        let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
            &ordinary, 1., false,
        ))];
        let before = layers.clone();
        let mut limited = configuration(1.);
        limited.max_svg_bytes = 0;
        assert!(matches!(
            prepare_cpu_svg_composition(&layers, limited),
            Err(CpuSvgSceneError::SvgBudgetExceeded { .. })
        ));
        let recovered = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("recover after output-cap failure");
        assert_eq!(
            recovered.svg().svg,
            original_composition(&layers, configuration(1.)).svg
        );
        assert_layers_unchanged(&layers, &before);
    }

    #[test]
    fn at_r20_1_subnormal_dash_nonprogress_is_rejected_before_original_bounds() {
        let stroke = Stroke {
            dash_lengths: vec![f64::from_bits(1), f64::from_bits(1)],
            transform: DAffine2::from_scale(DVec2::splat(10.)),
            ..Stroke::new(2.)
        };
        let vector = fill_and_stroke_vector(&stroke, 1., false);
        let before = dash_preservation_snapshot(&vector);
        let layers = [CpuSvgSceneLayer::Vector(vector)];
        assert_eq!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::UnsupportedStrokeStyle)
        );
        let CpuSvgSceneLayer::Vector(vector) = &layers[0] else {
            panic!("owned vector")
        };
        assert_eq!(dash_preservation_snapshot(vector), before);
        for lengths in [
            vec![f64::from_bits(1), 1.],
            vec![f64::from_bits(1), 0., 1.],
            vec![-0., 0., 1.],
        ] {
            let mixed = Stroke {
                dash_lengths: lengths,
                ..stroke.clone()
            };
            let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
                &mixed, 1., false,
            ))];
            let before = layers.clone();
            let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                .expect("a positive advancing dash preserves original small entries");
            assert_eq!(
                prepared.svg().svg,
                original_composition(&layers, configuration(1.)).svg
            );
            assert_layers_unchanged(&layers, &before);
        }
        for weight in [0., -1.] {
            let invisible = Stroke {
                weight,
                ..stroke.clone()
            };
            let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
                &invisible, 1., false,
            ))];
            let before = layers.clone();
            let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                .expect("invisible subnormal dashes skip outline work");
            assert_eq!(
                prepared.svg().svg,
                original_composition(&layers, configuration(1.)).svg
            );
            assert_layers_unchanged(&layers, &before);
        }
    }

    #[test]
    fn at_r20_1_aligned_strokes_preserve_original_clips_masks_width_and_order() {
        for alignment in [
            StrokeAlign::Center,
            StrokeAlign::Inside,
            StrokeAlign::Outside,
        ] {
            for weight in [0.5, 2.] {
                for alpha in [0., 0.5, 1.] {
                    for below in [false, true] {
                        let stroke = Stroke {
                            align: alignment,
                            transform: DAffine2::from_cols_array(&[2., 0.25, -0.5, 3., 4., -2.]),
                            ..Stroke::new(weight)
                        };
                        let mut vector = fill_and_stroke_vector(&stroke, 1., below);
                        let mut appearance = vector
                            .attribute::<Appearance>(ATTR_APPEARANCE)
                            .expect("source appearance")
                            .clone();
                        let fill_index = usize::from(below);
                        appearance.0.set_attribute(
                            ATTR_PAINT,
                            fill_index,
                            Graphic::from(Color::from_gamma_srgb_channels(0.8, 0.2, 0.1, alpha)),
                        );
                        vector.set_attribute(ATTR_APPEARANCE, appearance);
                        vector.set_attribute(
                            ATTR_TRANSFORM,
                            DAffine2::from_cols_array(&[1.5, -0.25, 0.5, 2., 7., 11.]),
                        );
                        let layers = [CpuSvgSceneLayer::Vector(vector)];
                        let before = layers.clone();
                        let config = configuration(2.);
                        let prepared = prepare_cpu_svg_composition(&layers, config)
                            .expect("finite original aligned stroke");
                        let svg = &prepared.svg().svg;
                        assert_eq!(
                            normalized_paint_ids(svg),
                            normalized_paint_ids(&original_composition(&layers, config).svg)
                        );
                        let separate = alignment == StrokeAlign::Inside
                            || (alignment == StrokeAlign::Outside && alpha < 1.);
                        assert_eq!(svg.contains("id=\"alignment-"), separate);
                        assert_eq!(svg.contains("<clipPath"), alignment == StrokeAlign::Inside);
                        assert_eq!(
                            svg.contains("<mask"),
                            alignment == StrokeAlign::Outside && alpha < 1.
                        );
                        assert_eq!(
                            svg.contains("paint-order: stroke;"),
                            below || (alignment == StrokeAlign::Outside && alpha == 1.)
                        );
                        let rendered_width = if alignment == StrokeAlign::Center {
                            weight
                        } else {
                            weight * 2.
                        };
                        if rendered_width == 1. {
                            assert!(!svg.contains("stroke-width="));
                        } else {
                            assert!(svg.contains(&format!("stroke-width=\"{rendered_width}\"")));
                        }
                        assert_layers_unchanged(&layers, &before);
                        assert_layers_unchanged(prepared.layers(), &before);
                    }
                }
            }
        }
    }

    #[test]
    fn at_r20_1_open_mixed_and_invisible_alignment_retains_source_fallback_and_recovery() {
        use graphite_vector_types::kurbo::BezPath;
        let mut open = BezPath::new();
        open.move_to((0., 0.));
        open.line_to((3., 2.));
        let mut mixed = open.clone();
        mixed.move_to((7., 4.));
        mixed.line_to((10., 4.));
        mixed.line_to((10., 7.));
        mixed.close_path();
        for alignment in [StrokeAlign::Inside, StrokeAlign::Outside] {
            for path in [&open, &mixed] {
                let mut vector = fill_and_stroke_vector(
                    &Stroke {
                        align: alignment,
                        ..Stroke::new(2.)
                    },
                    1.,
                    false,
                );
                *vector.element_mut() = Vector::from_bezpath(path.clone());
                let layers = [CpuSvgSceneLayer::Vector(vector)];
                let before = layers.clone();
                let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                    .expect("any open contour falls back to centered stroke");
                assert_eq!(
                    prepared.svg().svg,
                    original_composition(&layers, configuration(1.)).svg
                );
                assert!(!prepared.svg().svg.contains("alignment-"));
                assert!(prepared.svg().svg.contains("stroke-width=\"2\""));
                assert_layers_unchanged(&layers, &before);
            }
            for (weight, alpha) in [(0., 1.), (-2., 1.), (2., 0.)] {
                let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
                    &Stroke {
                        align: alignment,
                        ..Stroke::new(weight)
                    },
                    alpha,
                    true,
                ))];
                let before = layers.clone();
                let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                    .expect("original invisible aligned stroke");
                assert_eq!(
                    prepared.svg().svg,
                    original_composition(&layers, configuration(1.)).svg
                );
                assert!(!prepared.svg().svg.contains("alignment-"));
                let mut config = configuration(1.);
                config.max_svg_bytes = 0;
                assert!(matches!(
                    prepare_cpu_svg_composition(&layers, config).err(),
                    Some(CpuSvgSceneError::SvgBudgetExceeded { maximum: 0, .. })
                ));
                assert_eq!(
                    prepared.svg().svg,
                    prepare_cpu_svg_composition(&layers, configuration(1.))
                        .expect("cap failure is recoverable")
                        .svg()
                        .svg
                );
                assert_layers_unchanged(&layers, &before);
            }
        }
    }

    #[test]
    fn at_r20_1_aligned_derived_overflow_rejects_before_source_and_recovers() {
        use graphite_vector_types::kurbo::BezPath;
        for alignment in [StrokeAlign::Inside, StrokeAlign::Outside] {
            let layers = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
                &Stroke {
                    align: alignment,
                    ..Stroke::new(f64::MAX)
                },
                1.,
                false,
            ))];
            let before = layers.clone();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(CpuSvgSceneError::NonFiniteGeometry)
            );
            assert_layers_unchanged(&layers, &before);
        }
        let mut path = BezPath::new();
        path.move_to((1e-100, 1.));
        path.line_to((2e-100, 1.));
        path.line_to((2e-100, 2.));
        path.line_to((1e-100, 2.));
        path.close_path();
        let stroke = Stroke {
            align: StrokeAlign::Outside,
            transform: DAffine2::from_scale(DVec2::new(1e100, 1.)),
            ..Stroke::new(1e-100)
        };
        let mut vector = fill_and_stroke_vector(&stroke, 1., false);
        *vector.element_mut() = Vector::from_bezpath(path);
        let mut appearance = vector
            .attribute::<Appearance>(ATTR_APPEARANCE)
            .expect("source appearance")
            .clone();
        appearance
            .0
            .set_attribute(ATTR_PAINT, 0, Graphic::from(Color::TRANSPARENT));
        vector.set_attribute(ATTR_APPEARANCE, appearance);
        let layers = [CpuSvgSceneLayer::Vector(vector)];
        let before = layers.clone();
        assert_eq!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
        assert_layers_unchanged(&layers, &before);

        let mut vector = fill_and_stroke_vector(
            &Stroke {
                align: StrokeAlign::Outside,
                join_miter_limit: 3.,
                ..Stroke::new(5e307)
            },
            1.,
            false,
        );
        let mut appearance = vector
            .attribute::<Appearance>(ATTR_APPEARANCE)
            .expect("source appearance")
            .clone();
        appearance
            .0
            .set_attribute(ATTR_PAINT, 0, Graphic::from(Color::TRANSPARENT));
        vector.set_attribute(ATTR_APPEARANCE, appearance);
        let layers = [CpuSvgSceneLayer::Vector(vector)];
        let before = layers.clone();
        assert_eq!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
        assert_layers_unchanged(&layers, &before);
        let recovered = [CpuSvgSceneLayer::Vector(fill_and_stroke_vector(
            &Stroke {
                align: StrokeAlign::Inside,
                ..Stroke::new(2.)
            },
            1.,
            false,
        ))];
        assert_eq!(
            normalized_paint_ids(
                &prepare_cpu_svg_composition(&recovered, configuration(1.))
                    .expect("aligned recovery after overflow")
                    .svg()
                    .svg
            ),
            normalized_paint_ids(&original_composition(&recovered, configuration(1.)).svg)
        );
    }

    fn text_fixture(text: &str) -> Item<String> {
        let font = Resource::new(
            graphite_std::text_nodes::FALLBACK_FONT_RESOURCE
                .as_ref()
                .to_vec(),
        );
        Item::new_from_element(text.to_owned()).with_attribute(ATTR_FONT, font)
    }

    #[test]
    fn at_r20_1_text_item_and_list_preserve_original_layout_paths_and_attributes() {
        for alignment in [
            TextAlign::AlignLeft,
            TextAlign::AlignCenter,
            TextAlign::AlignRight,
            TextAlign::JustifyLeft,
            TextAlign::JustifyCenter,
            TextAlign::JustifyRight,
            TextAlign::JustifyAll,
        ] {
            for width in [None, Some(58.)] {
                let mut text = text_fixture("A B\nHi!")
                    .with_attribute(ATTR_FONT_SIZE, 20.5_f64)
                    .with_attribute(ATTR_LINE_HEIGHT, 1.25_f64)
                    .with_attribute(ATTR_LETTER_SPACING, 0.75_f64)
                    .with_attribute(ATTR_LETTER_TILT, -12.5_f64)
                    .with_attribute(ATTR_TEXT_ALIGN, alignment)
                    .with_attribute(
                        ATTR_TRANSFORM,
                        DAffine2::from_cols_array(&[1.5, 0.25, -0.5, 2., 11., 7.]),
                    )
                    .with_attribute(ATTR_OPACITY, 0.5_f64)
                    .with_attribute(ATTR_OPACITY_FILL, 0.5_f64)
                    .with_attribute(ATTR_BLEND_MODE, BlendMode::Multiply);
                if let Some(width) = width {
                    text.set_attribute(ATTR_MAX_WIDTH, width);
                }
                let font = text
                    .attribute::<Resource>(ATTR_FONT)
                    .expect("owned explicit font");
                let font_bytes = font.as_ref().to_vec();
                let font_hash = font.hash();
                let mut texts = List::new_from_item(text.clone());
                texts.push(
                    text_fixture("second row")
                        .with_attribute(ATTR_FONT_SIZE, 16_f64)
                        .with_attribute(
                            ATTR_TRANSFORM,
                            DAffine2::from_translation(DVec2::new(5., 80.)),
                        ),
                );
                for layer in [
                    CpuSvgSceneLayer::Text(text.clone()),
                    CpuSvgSceneLayer::TextList(texts),
                ] {
                    let layers = [layer];
                    let before = layers.clone();
                    let mut config = configuration(2.);
                    config.max_svg_bytes = 64 * 1024;
                    let prepared =
                        prepare_cpu_svg_composition(&layers, config).expect("original text scene");
                    assert_eq!(
                        prepared.svg().svg,
                        original_composition(&layers, config).svg
                    );
                    assert!(
                        prepared
                            .svg()
                            .svg
                            .contains("fill=\"black\" fill-rule=\"nonzero\"")
                    );
                    assert!(prepared.svg().svg.contains("opacity=\"0.25\""));
                    assert!(prepared.svg().svg.contains("mix-blend-mode: multiply;"));
                    assert_layers_unchanged(&layers, &before);
                    assert_layers_unchanged(prepared.layers(), &before);
                }
                assert_eq!(
                    text.attribute::<Resource>(ATTR_FONT)
                        .expect("font retained")
                        .as_ref(),
                    font_bytes
                );
                assert_eq!(
                    text.attribute::<Resource>(ATTR_FONT)
                        .expect("font retained")
                        .hash(),
                    font_hash
                );
            }
        }
    }

    #[test]
    fn at_r20_1_valid_empty_whitespace_clipped_and_ignored_dimensions_are_not_malformed_fonts() {
        for (text, expected_paths) in [
            (text_fixture(""), false),
            (text_fixture(" \t\n "), true),
            (text_fixture("   "), false),
            (
                text_fixture("H").with_attribute(ATTR_MAX_HEIGHT, 1_f64),
                false,
            ),
        ] {
            if text.element() == "H" {
                TextContext::with_thread_local(|context| {
                    let font = text
                        .attribute::<Resource>(ATTR_FONT)
                        .expect("explicit valid font");
                    let layout = context
                        .layout_text("H", font, TypesettingConfig::default())
                        .expect("original baseline witness");
                    assert!(layout.lines().all(|line| line.metrics().baseline > 1.));
                });
            }
            let layers = [CpuSvgSceneLayer::Text(text)];
            let before = layers.clone();
            let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                .expect("valid source blank or control-glyph scene");
            assert_eq!(
                prepared.svg().svg,
                original_composition(&layers, configuration(1.)).svg
            );
            assert_eq!(
                prepared.svg().svg.contains("fill-rule=\"nonzero\""),
                expected_paths,
                "source fixture {before:?}"
            );
            assert_layers_unchanged(&layers, &before);
        }
        let empty = [CpuSvgSceneLayer::TextList(List::new())];
        assert_eq!(
            prepare_cpu_svg_composition(&empty, configuration(1.))
                .expect("empty original list")
                .svg()
                .svg,
            original_composition(&empty, configuration(1.)).svg
        );
        let original = [CpuSvgSceneLayer::Text(text_fixture("Hi"))];
        let ignored = [CpuSvgSceneLayer::Text(
            text_fixture("Hi")
                .with_attribute(ATTR_MAX_WIDTH, -f64::MAX)
                .with_attribute(ATTR_MAX_HEIGHT, 0_f64),
        )];
        assert_eq!(
            prepare_cpu_svg_composition(&original, configuration(1.))
                .expect("unbounded source text")
                .svg()
                .svg,
            prepare_cpu_svg_composition(&ignored, configuration(1.))
                .expect("nonpositive dimensions ignored by source")
                .svg()
                .svg
        );
        for text in [
            text_fixture("Hi").with_attribute(ATTR_FONT, Resource::empty()),
            text_fixture("").with_attribute(ATTR_FONT, Resource::new(vec![1_u8, 2, 3])),
            Item::new_from_element("Hi".to_owned()),
            text_fixture("Hi").with_attribute(
                ATTR_FONT,
                Resource::new_unchecked(
                    graphite_std::text_nodes::FALLBACK_FONT_RESOURCE
                        .as_ref()
                        .to_vec(),
                    Resource::empty().hash(),
                ),
            ),
        ] {
            let layers = [CpuSvgSceneLayer::Text(text)];
            let before = layers.clone();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(CpuSvgSceneError::InvalidTextFont)
            );
            assert_layers_unchanged(&layers, &before);
        }
    }

    #[test]
    fn at_r20_1_text_rejects_typed_nonfinite_narrowing_and_derived_layout_failures() {
        for key in [
            ATTR_FONT,
            ATTR_FONT_SIZE,
            ATTR_LINE_HEIGHT,
            ATTR_LETTER_SPACING,
            ATTR_LETTER_TILT,
            ATTR_MAX_WIDTH,
            ATTR_MAX_HEIGHT,
            ATTR_TEXT_ALIGN,
        ] {
            let layers = [CpuSvgSceneLayer::Text(
                text_fixture("Hi").with_attribute(key, "wrong-type".to_owned()),
            )];
            let before = layers.clone();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(CpuSvgSceneError::WrongAttributeType(key.to_owned()))
            );
            assert_layers_unchanged(&layers, &before);
        }
        for key in [
            ATTR_FONT_SIZE,
            ATTR_LINE_HEIGHT,
            ATTR_LETTER_SPACING,
            ATTR_MAX_WIDTH,
            ATTR_MAX_HEIGHT,
        ] {
            let layers = [CpuSvgSceneLayer::Text(
                text_fixture("Hi").with_attribute(key, f64::MAX),
            )];
            let before = layers.clone();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(CpuSvgSceneError::NonFiniteTextParameter(key.to_owned()))
            );
            assert_layers_unchanged(&layers, &before);
        }
        for key in [
            ATTR_FONT_SIZE,
            ATTR_LINE_HEIGHT,
            ATTR_LETTER_SPACING,
            ATTR_LETTER_TILT,
            ATTR_MAX_WIDTH,
            ATTR_MAX_HEIGHT,
        ] {
            let layers = [CpuSvgSceneLayer::Text(
                text_fixture("Hi").with_attribute(key, f64::NAN),
            )];
            let before_font = layers.first().expect("source layer");
            let CpuSvgSceneLayer::Text(text) = before_font else {
                panic!("text variant")
            };
            let bytes = text
                .attribute::<Resource>(ATTR_FONT)
                .expect("font")
                .as_ref()
                .to_vec();
            assert_eq!(
                prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                Some(CpuSvgSceneError::NonFiniteTextParameter(key.to_owned()))
            );
            assert!(text.attribute::<f64>(key).expect("unchanged NaN").is_nan());
            assert_eq!(
                text.attribute::<Resource>(ATTR_FONT)
                    .expect("retained font")
                    .as_ref(),
                bytes
            );
        }
        let layers = [CpuSvgSceneLayer::Text(
            text_fixture("Hi")
                .with_attribute(ATTR_FONT_SIZE, 32_f64)
                .with_attribute(ATTR_LINE_HEIGHT, f32::MAX as f64),
        )];
        let before = layers.clone();
        assert!(matches!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::InvalidTextLayout)
        ));
        assert_layers_unchanged(&layers, &before);
        let layers = [CpuSvgSceneLayer::Text(text_fixture("Hi").with_attribute(
            ATTR_TRANSFORM,
            DAffine2::from_scale(DVec2::splat(1e308)),
        ))];
        let before = layers.clone();
        assert!(matches!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::InvalidTextLayout | CpuSvgSceneError::NonFiniteGeometry)
        ));
        assert_layers_unchanged(&layers, &before);
        let layers = [CpuSvgSceneLayer::Text(text_fixture("Hi").with_attribute(
            ATTR_APPEARANCE,
            Appearance::new_single(
                graphite_graphic_types::Coverage::new_fill(),
                Graphic::from(Color::WHITE),
            ),
        ))];
        assert_eq!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::UnsupportedAttribute(
                ATTR_APPEARANCE.to_owned()
            ))
        );
    }

    #[test]
    fn at_r20_1_text_masks_and_cap_failure_preserve_original_work_and_recover() {
        let source = result([1., 0., 0., 1., 11., 13.]);
        let clipped = with_item(
            &source,
            source
                .original_item()
                .clone()
                .with_attribute(ATTR_CLIPPING_MASK, true),
        );
        let text = text_fixture("Hi").with_attribute(ATTR_OPACITY, 0.5_f64);
        let layers = [
            CpuSvgSceneLayer::Text(text.clone()),
            CpuSvgSceneLayer::Image(clipped.clone()),
        ];
        let before = layers.clone();
        let config = configuration(1.);
        let prepared =
            prepare_cpu_svg_composition(&layers, config).expect("actual original text mask");
        assert_eq!(
            normalized_paint_ids(&prepared.svg().svg),
            normalized_paint_ids(&original_composition(&layers, config).svg)
        );
        assert_layers_unchanged(&layers, &before);
        let overflowing = text
            .with_attribute(ATTR_OPACITY, -f64::MAX)
            .with_attribute(ATTR_OPACITY_FILL, 1e-308_f64);
        let unmasked = [CpuSvgSceneLayer::Text(overflowing.clone())];
        assert_eq!(
            prepare_cpu_svg_composition(&unmasked, config)
                .expect("finite ordinary opacity product")
                .svg()
                .svg,
            original_composition(&unmasked, config).svg
        );
        for layer in [
            CpuSvgSceneLayer::Text(overflowing.clone()),
            CpuSvgSceneLayer::TextList(List::new_from_item(overflowing)),
        ] {
            let layers = [layer, CpuSvgSceneLayer::Image(clipped.clone())];
            let before = layers.clone();
            assert!(original_composition(&layers, config).svg.contains("-inf"));
            assert_eq!(
                prepare_cpu_svg_composition(&layers, config).err(),
                Some(CpuSvgSceneError::NonFiniteMaskOpacity)
            );
            assert_layers_unchanged(&layers, &before);
        }
        let cap_layers = [CpuSvgSceneLayer::Text(
            text_fixture("Hi").with_attribute(ATTR_OPACITY, 0.5_f64),
        )];
        let cap_before = cap_layers.clone();
        let cap_prepared =
            prepare_cpu_svg_composition(&cap_layers, config).expect("stable original text SVG");
        let expected = normalized_paint_ids(&cap_prepared.svg().svg);
        let exact = cap_prepared.svg().svg.len();
        for maximum in [exact, exact - 1, 0] {
            let mut limited = config;
            limited.max_svg_bytes = maximum;
            let result = prepare_cpu_svg_composition(&cap_layers, limited);
            if maximum == exact {
                assert_eq!(
                    normalized_paint_ids(&result.expect("exact cap").svg().svg),
                    expected
                );
            } else {
                assert!(
                    matches!(result.err(),Some(CpuSvgSceneError::SvgBudgetExceeded {maximum: observed,actual})
                if observed==maximum && actual>maximum)
                );
            }
            assert_layers_unchanged(&cap_layers, &cap_before);
        }
        assert_eq!(
            normalized_paint_ids(
                &prepare_cpu_svg_composition(&cap_layers, config)
                    .expect("text cap recovery")
                    .svg()
                    .svg
            ),
            expected
        );
    }

    #[test]
    fn at_r20_1_text_seals_changing_resource_backing_and_deduplicates_exact_font_versions() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct SwitchingFont {
            changed: Arc<AtomicBool>,
            original: &'static [u8],
        }
        impl AsRef<[u8]> for SwitchingFont {
            fn as_ref(&self) -> &[u8] {
                if self.changed.load(Ordering::SeqCst) {
                    b"changed invalid font"
                } else {
                    self.original
                }
            }
        }
        let original: &'static [u8] = graphite_std::text_nodes::FALLBACK_FONT_RESOURCE.as_ref();
        let changed = Arc::new(AtomicBool::new(false));
        let font = Resource::new(SwitchingFont {
            changed: changed.clone(),
            original,
        });
        let input_backing: Arc<dyn AsRef<[u8]> + Send + Sync> = (&font).into();
        let text = Item::new_from_element("H".to_owned()).with_attribute(ATTR_FONT, font.clone());
        let mut texts = List::new_from_item(text.clone());
        texts.push(Item::new_from_element("P".to_owned()).with_attribute(ATTR_FONT, font.clone()));
        let layers = [
            CpuSvgSceneLayer::Text(text),
            CpuSvgSceneLayer::TextList(texts),
        ];
        let before = layers.clone();
        let composition = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("authenticated initial snapshot");
        let artboard = prepare_cpu_svg_artboard(
            &layers,
            CpuSvgArtboardConfiguration {
                location: DVec2::ZERO,
                dimensions: DVec2::new(64., 64.),
                background: Color::TRANSPARENT,
                clip: true,
            },
            configuration(1.),
        )
        .expect("same snapshot owner for artboard");
        changed.store(true, Ordering::SeqCst);
        for prepared in [composition.layers(), artboard.layers()] {
            let CpuSvgSceneLayer::Text(text) = &prepared[0] else {
                panic!("text layer retained")
            };
            let CpuSvgSceneLayer::TextList(texts) = &prepared[1] else {
                panic!("text list retained")
            };
            let sealed = text.attribute::<Resource>(ATTR_FONT).expect("sealed font");
            assert_eq!(sealed.as_ref(), original);
            assert_eq!(sealed.hash(), font.hash());
            let first: Arc<dyn AsRef<[u8]> + Send + Sync> = sealed.into();
            assert!(!Arc::ptr_eq(&first, &input_backing));
            for index in 0..texts.len() {
                let sealed = texts
                    .attribute::<Resource>(ATTR_FONT, index)
                    .expect("sealed row font");
                assert_eq!(sealed.as_ref(), original);
                assert_eq!(sealed.hash(), font.hash());
                let row: Arc<dyn AsRef<[u8]> + Send + Sync> = sealed.into();
                assert!(
                    Arc::ptr_eq(&first, &row),
                    "one exact immutable font snapshot per version"
                );
            }
        }
        assert_eq!(font.as_ref(), b"changed invalid font");
        let after: Arc<dyn AsRef<[u8]> + Send + Sync> = (&font).into();
        assert!(Arc::ptr_eq(&after, &input_backing));
        assert_layers_unchanged(&layers, &before);
        assert_eq!(
            prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
            Some(CpuSvgSceneError::InvalidTextFont)
        );
        changed.store(false, Ordering::SeqCst);
        let recovered = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("authentic version recovery");
        assert_eq!(recovered.svg().svg, composition.svg().svg);
    }
    #[test]
    fn at_r20_1_color_items_and_ordered_lists_preserve_original_nonspatial_rendering() {
        for alpha in [0., 0.6251, 1., 2., -0.25, -0.0004] {
            for opacity in [1., 0.25] {
                let item = Item::new_from_element(Color::from_rgbaf32_unchecked(1., 0., 0., alpha))
                    .with_attribute(ATTR_OPACITY, opacity)
                    .with_attribute(ATTR_OPACITY_FILL, 0.5_f64)
                    .with_attribute(ATTR_BLEND_MODE, BlendMode::Multiply);
                let transformed = item.clone().with_attribute(
                    ATTR_TRANSFORM,
                    DAffine2::from_cols_array(&[f64::MAX, -f64::MAX, 0., 0., f64::MAX, -f64::MAX]),
                );
                let mut previous = None;
                for layer in [
                    CpuSvgSceneLayer::Color(item.clone()),
                    CpuSvgSceneLayer::Color(transformed.clone()),
                    CpuSvgSceneLayer::ColorList(List::from(transformed.clone())),
                ] {
                    let layers = [layer];
                    let before = layers.clone();
                    let prepared = prepare_cpu_svg_composition(&layers, configuration(2.))
                        .expect("finite original nonspatial Color");
                    assert_eq!(
                        prepared.svg().svg,
                        original_composition(&layers, configuration(2.)).svg
                    );
                    let svg = &prepared.svg().svg;
                    assert!(svg.contains("points=\"10000000,10000000 -10000000,10000000 -10000000,-10000000 10000000,-10000000\""));
                    assert!(svg.contains("fill=\"#ff0000\""));
                    if alpha < 1. {
                        let source_alpha = (alpha * 1000.).round() / 1000.;
                        assert!(svg.contains(&format!("fill-opacity=\"{source_alpha}\"")));
                    } else {
                        assert!(!svg.contains("fill-opacity="));
                    }
                    assert!(svg.contains("mix-blend-mode: multiply;"));
                    if let Some(expected) = &previous {
                        assert_eq!(svg, expected);
                    }
                    previous = Some(svg.clone());
                    assert_layers_unchanged(&layers, &before);
                    assert_layers_unchanged(prepared.layers(), &before);
                }
            }
        }
        let first = Item::new_from_element(Color::RED).with_attribute(ATTR_OPACITY, 0.5_f64);
        let second = Item::new_from_element(Color::BLUE).with_attribute(ATTR_OPACITY, 0.25_f64);
        let mut colors = List::from(first.clone());
        colors.push(second.clone());
        let layers = [CpuSvgSceneLayer::ColorList(colors)];
        let before = layers.clone();
        let prepared =
            prepare_cpu_svg_composition(&layers, configuration(1.)).expect("ordered colors");
        assert_eq!(
            prepared.svg().svg,
            original_composition(&layers, configuration(1.)).svg
        );
        assert!(
            prepared.svg().svg.find("#ff0000").expect("red row")
                < prepared.svg().svg.find("#0000ff").expect("blue row")
        );
        let mut reverse = List::from(second);
        reverse.push(first);
        let reversed =
            prepare_cpu_svg_composition(&[CpuSvgSceneLayer::ColorList(reverse)], configuration(1.))
                .expect("reverse colors");
        assert!(
            reversed.svg().svg.find("#0000ff").expect("blue row")
                < reversed.svg().svg.find("#ff0000").expect("red row")
        );
        let mut moved = configuration(1.);
        moved.viewport.transform = DAffine2::from_translation(DVec2::new(19., 23.));
        let moved_scene =
            prepare_cpu_svg_composition(&layers, moved).expect("outer viewport color placement");
        assert_eq!(
            moved_scene.svg().svg,
            original_composition(&layers, moved).svg
        );
        assert_ne!(moved_scene.svg().svg, prepared.svg().svg);
        let artboard = CpuSvgArtboardConfiguration {
            location: DVec2::new(5., -3.),
            dimensions: DVec2::new(32., 24.),
            background: Color::TRANSPARENT,
            clip: true,
        };
        let board = prepare_cpu_svg_artboard(&layers, artboard, configuration(1.))
            .expect("source clipped color artboard");
        assert_eq!(
            normalized_paint_ids(&board.svg().svg),
            normalized_paint_ids(&original_artboard(&layers, artboard, configuration(1.)).svg)
        );
        assert_layers_unchanged(&layers, &before);
        assert_layers_unchanged(board.layers(), &before);
        let empty = [CpuSvgSceneLayer::ColorList(List::new())];
        assert_eq!(
            prepare_cpu_svg_composition(&empty, configuration(1.))
                .expect("empty colors")
                .svg()
                .svg,
            original_composition(&empty, configuration(1.)).svg
        );
    }

    #[test]
    fn at_r20_1_color_list_clipping_and_actual_mask_opacity_follow_source_runs() {
        let base = Item::new_from_element(Color::RED);
        for (first_clips, second_clips) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let mut colors = List::from(
                Item::new_from_element(Color::BLUE).with_attribute(ATTR_CLIPPING_MASK, first_clips),
            );
            colors.push(
                Item::new_from_element(Color::WHITE)
                    .with_attribute(ATTR_CLIPPING_MASK, second_clips),
            );
            let layers = [
                CpuSvgSceneLayer::Color(base.clone()),
                CpuSvgSceneLayer::ColorList(colors),
            ];
            let before = layers.clone();
            let prepared = prepare_cpu_svg_composition(&layers, configuration(1.))
                .expect("all-row color clipping");
            assert_eq!(
                normalized_paint_ids(&prepared.svg().svg),
                normalized_paint_ids(&original_composition(&layers, configuration(1.)).svg)
            );
            assert_eq!(
                prepared.svg().svg.contains("<mask"),
                first_clips && second_clips
            );
            assert!(!prepared.svg().svg.contains("<clipPath"));
            assert_layers_unchanged(&layers, &before);
            assert_layers_unchanged(prepared.layers(), &before);
        }
        let empty = [
            CpuSvgSceneLayer::Color(base.clone()),
            CpuSvgSceneLayer::ColorList(List::new()),
        ];
        assert!(
            !prepare_cpu_svg_composition(&empty, configuration(1.))
                .expect("empty list does not clip")
                .svg()
                .svg
                .contains("<mask")
        );
        let ordinary = base
            .with_attribute(ATTR_OPACITY, -f64::MAX)
            .with_attribute(ATTR_OPACITY_FILL, 0_f64);
        let clip = Item::new_from_element(Color::BLUE).with_attribute(ATTR_CLIPPING_MASK, true);
        for layer in [
            CpuSvgSceneLayer::Color(ordinary.clone()),
            CpuSvgSceneLayer::ColorList(List::from(ordinary.clone())),
        ] {
            let unmasked = [layer.clone()];
            assert_eq!(
                prepare_cpu_svg_composition(&unmasked, configuration(1.))
                    .expect("finite ordinary zero product")
                    .svg()
                    .svg,
                original_composition(&unmasked, configuration(1.)).svg
            );
            let masked = [layer.clone(), CpuSvgSceneLayer::Color(clip.clone())];
            let before = masked.clone();
            assert!(
                original_composition(&masked, configuration(1.))
                    .svg
                    .contains("opacity=\"-inf\"")
            );
            assert_eq!(
                prepare_cpu_svg_composition(&masked, configuration(1.)).err(),
                Some(CpuSvgSceneError::NonFiniteMaskOpacity)
            );
            assert_layers_unchanged(&masked, &before);
            let continuing = [
                CpuSvgSceneLayer::Color(Item::new_from_element(Color::WHITE)),
                layer.clone(),
                CpuSvgSceneLayer::Color(clip.clone()),
            ];
            let continuing = match &continuing[1] {
                CpuSvgSceneLayer::Color(item) => [
                    continuing[0].clone(),
                    CpuSvgSceneLayer::Color(item.clone().with_attribute(ATTR_CLIPPING_MASK, true)),
                    continuing[2].clone(),
                ],
                CpuSvgSceneLayer::ColorList(items) => {
                    let item = items
                        .clone_item(0)
                        .expect("source row")
                        .with_attribute(ATTR_CLIPPING_MASK, true);
                    [
                        continuing[0].clone(),
                        CpuSvgSceneLayer::ColorList(List::from(item)),
                        continuing[2].clone(),
                    ]
                }
                _ => panic!("explicit Color variant"),
            };
            let prepared = prepare_cpu_svg_composition(&continuing, configuration(1.))
                .expect("continuing layer uses ordinary product");
            assert_eq!(
                normalized_paint_ids(&prepared.svg().svg),
                normalized_paint_ids(&original_composition(&continuing, configuration(1.)).svg)
            );
            let restarted = [
                CpuSvgSceneLayer::Color(Item::new_from_element(Color::WHITE)),
                CpuSvgSceneLayer::Color(clip.clone()),
                CpuSvgSceneLayer::Color(Item::new_from_element(Color::BLACK)),
                layer,
                CpuSvgSceneLayer::Color(clip.clone()),
            ];
            assert_eq!(
                prepare_cpu_svg_composition(&restarted, configuration(1.)).err(),
                Some(CpuSvgSceneError::NonFiniteMaskOpacity)
            );
        }
        let mut later_overflow = List::from(Item::new_from_element(Color::WHITE));
        later_overflow.push(ordinary);
        let unmasked = [CpuSvgSceneLayer::ColorList(later_overflow.clone())];
        assert_eq!(
            prepare_cpu_svg_composition(&unmasked, configuration(1.))
                .expect("later Color row finite ordinary product")
                .svg()
                .svg,
            original_composition(&unmasked, configuration(1.)).svg
        );
        let masked = [
            CpuSvgSceneLayer::ColorList(later_overflow),
            CpuSvgSceneLayer::Color(clip),
        ];
        let before = masked.clone();
        assert!(
            original_composition(&masked, configuration(1.))
                .svg
                .contains("opacity=\"-inf\"")
        );
        assert_eq!(
            prepare_cpu_svg_composition(&masked, configuration(1.)).err(),
            Some(CpuSvgSceneError::NonFiniteMaskOpacity)
        );
        assert_layers_unchanged(&masked, &before);
        let mut empty_colors = List::<Color>::new();
        assert_eq!(
            empty_colors
                .iter_attribute_values_mut_or_default::<f32>(ATTR_OPACITY)
                .count(),
            0
        );
        let empty = [CpuSvgSceneLayer::ColorList(empty_colors)];
        let before = empty.clone();
        assert_eq!(
            prepare_cpu_svg_composition(&empty, configuration(1.))
                .expect("unused recognized empty column")
                .svg()
                .svg,
            original_composition(&empty, configuration(1.)).svg
        );
        assert_layers_unchanged(&empty, &before);
    }

    #[test]
    fn at_r20_1_color_invalid_types_formatting_geometry_and_cap_preserve_work_and_recover() {
        let valid = Item::new_from_element(Color::RED);
        for (item, expected) in [
            (
                valid.clone().with_attribute(ATTR_OPACITY, 0.5_f32),
                CpuSvgSceneError::WrongAttributeType(ATTR_OPACITY.into()),
            ),
            (
                valid.clone().with_attribute(ATTR_TRANSFORM, true),
                CpuSvgSceneError::WrongAttributeType(ATTR_TRANSFORM.into()),
            ),
            (
                valid.clone().with_attribute(ATTR_BLEND_MODE, true),
                CpuSvgSceneError::WrongAttributeType(ATTR_BLEND_MODE.into()),
            ),
            (
                valid.clone().with_attribute(ATTR_CLIPPING_MASK, 1_f64),
                CpuSvgSceneError::WrongAttributeType(ATTR_CLIPPING_MASK.into()),
            ),
            (
                valid
                    .clone()
                    .with_attribute(ATTR_APPEARANCE, Appearance::default()),
                CpuSvgSceneError::UnsupportedAttribute(ATTR_APPEARANCE.into()),
            ),
        ] {
            for layer in [
                CpuSvgSceneLayer::Color(item.clone()),
                CpuSvgSceneLayer::ColorList(List::from(item.clone())),
            ] {
                let layers = [layer];
                let before = layers.clone();
                assert_eq!(
                    prepare_cpu_svg_composition(&layers, configuration(1.)).err(),
                    Some(expected.clone())
                );
                assert_layers_unchanged(&layers, &before);
            }
        }
        for color in [
            Color::from_rgbaf32_unchecked(f32::NAN, 0., 0., 1.),
            Color::from_rgbaf32_unchecked(1., 0., 0., -f32::MAX),
        ] {
            let item = Item::new_from_element(color);
            let before = [color.r(), color.g(), color.b(), color.a()].map(f32::to_bits);
            for layer in [
                CpuSvgSceneLayer::Color(item.clone()),
                CpuSvgSceneLayer::ColorList(List::from(item.clone())),
            ] {
                assert_eq!(
                    prepare_cpu_svg_composition(&[layer], configuration(1.)).err(),
                    Some(CpuSvgSceneError::NonFinitePaint)
                );
                assert_eq!(
                    [
                        item.element().r(),
                        item.element().g(),
                        item.element().b(),
                        item.element().a()
                    ]
                    .map(f32::to_bits),
                    before
                );
            }
        }
        let layers = [CpuSvgSceneLayer::Color(valid.clone())];
        let before = layers.clone();
        let mut overflowing = configuration(1.);
        overflowing.viewport.transform = DAffine2::from_scale(DVec2::splat(f64::MAX / 16.));
        assert_eq!(
            prepare_cpu_svg_composition(&layers, overflowing).err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
        let baseline = prepare_cpu_svg_composition(&layers, configuration(1.))
            .expect("valid Color before failure");
        for maximum in [0, baseline.svg().svg.len() - 1] {
            let mut config = configuration(1.);
            config.max_svg_bytes = maximum;
            assert_eq!(
                prepare_cpu_svg_composition(&layers, config).err(),
                Some(CpuSvgSceneError::SvgBudgetExceeded {
                    maximum,
                    actual: baseline.svg().svg.len()
                })
            );
            assert_layers_unchanged(&layers, &before);
        }
        let mut exact = configuration(1.);
        exact.max_svg_bytes = baseline.svg().svg.len();
        let recovered =
            prepare_cpu_svg_composition(&layers, exact).expect("exact Color cap recovery");
        assert_eq!(recovered.svg().svg, baseline.svg().svg);
        assert_layers_unchanged(recovered.layers(), &before);
        assert_layers_unchanged(&layers, &before);
    }
    #[test]
    fn at_r20_1_original_source_image_retains_precision_identity_transform_and_owner() {
        use crate::resources::{EncodedPngBudget, admit_encoded_png};
        let values = [1_u16, 256, 0x1234, 32768, 65535, 32768, 65534, 0];
        let samples: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect();
        let mut encoded = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut encoded, 2, 1);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Sixteen);
            let mut writer = encoder.write_header().expect("original input PNG header");
            writer
                .write_image_data(&samples)
                .expect("asymmetric original samples");
            writer.finish().expect("input PNG terminator");
        }
        let identity = ImageResourceIdentity {
            asset_id: "project/original-source".into(),
            version_id: "caller-supplied-input-version".into(),
            content_sha256: Sha256::digest(&encoded).into(),
        };
        let transform = [3., -0., 0., 2., -0., 5.];
        let source = Arc::new(
            admit_encoded_png(
                identity.clone(),
                transform,
                encoded.clone().into(),
                EncodedPngBudget {
                    max_encoded_bytes: 4096,
                    max_pixel_bytes: 4096,
                    max_metadata_bytes: 256,
                },
            )
            .expect("original RGBA16 decode"),
        );
        let before = source.image().clone();
        let original_decoded =
            Image::<Color>::from_encoded(&encoded).expect("pinned original decoder");
        let bits = |image: &Image<Color>| {
            image
                .data
                .iter()
                .map(|color| [color.r(), color.g(), color.b(), color.a()].map(f32::to_bits))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&before), bits(&original_decoded));
        let source_probe = Arc::downgrade(&source);
        let layers = [CpuSvgSceneLayer::SourceImage(source.clone())];
        let mut final_scene = None;
        for scale in [0.5, 1., 2.] {
            let config = configuration(scale);
            let prepared =
                prepare_cpu_svg_composition(&layers, config).expect("unedited original scene");
            assert_eq!(
                prepared.svg().svg,
                original_composition(&layers, config).svg
            );
            assert!(prepared.svg().svg.contains("width=\"1\" height=\"1\""));
            assert!(prepared.svg().svg.contains("matrix(3,0,0,2,0,5)"));
            assert_eq!(embedded_png(prepared.svg()), before.to_png());
            let output = Image::<Color>::from_encoded(&embedded_png(prepared.svg()))
                .expect("source SVG's RGBA8 PNG");
            assert_eq!(output.to_flat_u8(), before.to_flat_u8());
            assert_eq!(bits(source.image()), bits(&before));
            assert_eq!(source.identity(), &identity);
            assert_eq!(
                source.descriptor().transform.map(f64::to_bits),
                transform.map(f64::to_bits)
            );
            match prepared.layers() {
                [CpuSvgSceneLayer::SourceImage(retained)] => {
                    assert!(Arc::ptr_eq(retained, &source));
                    assert_eq!(retained.origin(), source.origin());
                }
                _ => panic!("original owner must remain a source layer"),
            }
            final_scene = Some(prepared);
        }
        drop(source);
        drop(layers);
        assert!(source_probe.upgrade().is_some());
        drop(final_scene);
        assert!(source_probe.upgrade().is_none());
    }

    #[test]
    fn at_r20_1_original_source_image_caps_geometry_masks_and_artboards_preserve_work() {
        let make_source = |transform| {
            Arc::new(
                admit_cpu_image(
                    ImageResourceIdentity {
                        asset_id: "project/unchanged-rows".into(),
                        version_id: "explicit-parent-label".into(),
                        content_sha256: Sha256::digest(PIXELS).into(),
                    },
                    CpuImageDescriptor {
                        width: 2,
                        height: 2,
                        row_stride: 8,
                        sample_format: CpuSampleFormat::Rgba8,
                        color_space: ImageColorSpace::Srgb,
                        alpha: AlphaAssociation::Straight,
                        transform,
                    },
                    Arc::from(PIXELS),
                    CpuImageBudget {
                        max_input_bytes: 16,
                        max_decoded_bytes: 4 * (std::mem::size_of::<Color>() + 4),
                    },
                )
                .expect("authenticated unchanged rows"),
            )
        };
        let source = make_source([16., 0., 0., 8., 0., 0.]);
        let before = source.image().clone();
        let layers = [CpuSvgSceneLayer::SourceImage(source.clone())];
        let baseline =
            prepare_cpu_svg_composition(&layers, configuration(1.)).expect("original before cap");
        for cap in [0, baseline.svg().svg.len() - 1] {
            let mut config = configuration(1.);
            config.max_svg_bytes = cap;
            assert_eq!(
                prepare_cpu_svg_composition(&layers, config).err(),
                Some(CpuSvgSceneError::SvgBudgetExceeded {
                    maximum: cap,
                    actual: baseline.svg().svg.len(),
                })
            );
            assert_eq!(source.image(), &before);
        }
        let mut exact = configuration(1.);
        exact.max_svg_bytes = baseline.svg().svg.len();
        let recovered =
            prepare_cpu_svg_composition(&layers, exact).expect("exact original cap recovery");
        assert_eq!(recovered.svg().svg, baseline.svg().svg);
        let mut overflow = configuration(1.);
        overflow.viewport.transform = DAffine2::from_scale(DVec2::splat(2.));
        let extreme = make_source([1., 0., 0., 1., f64::MAX, 0.]);
        assert_eq!(
            prepare_cpu_svg_composition(
                &[CpuSvgSceneLayer::SourceImage(extreme.clone())],
                overflow
            )
            .err(),
            Some(CpuSvgSceneError::NonFiniteGeometry)
        );
        assert_eq!(
            extreme.descriptor().transform[4].to_bits(),
            f64::MAX.to_bits()
        );
        let masked = [
            layers[0].clone(),
            CpuSvgSceneLayer::Color(
                Item::new_from_element(Color::WHITE).with_attribute(ATTR_CLIPPING_MASK, true),
            ),
        ];
        let prepared =
            prepare_cpu_svg_composition(&masked, configuration(1.)).expect("original source mask");
        assert_eq!(
            normalized_paint_ids(&prepared.svg().svg),
            normalized_paint_ids(&original_composition(&masked, configuration(1.)).svg)
        );
        let artboard = CpuSvgArtboardConfiguration {
            location: DVec2::new(5., 7.),
            dimensions: DVec2::new(32., 24.),
            background: Color::TRANSPARENT,
            clip: true,
        };
        let prepared = prepare_cpu_svg_artboard(&layers, artboard, configuration(1.))
            .expect("clipped original source");
        assert_eq!(
            normalized_paint_ids(&prepared.svg().svg),
            normalized_paint_ids(&original_artboard(&layers, artboard, configuration(1.)).svg)
        );
        assert_eq!(source.image(), &before);
        assert_layers_unchanged(prepared.layers(), &layers);
    }

    fn standalone_gradient_config() -> CpuSvgSceneConfiguration {
        CpuSvgSceneConfiguration {
            max_svg_bytes: 128 * 1024,
            ..configuration(1.5)
        }
    }

    fn assert_standalone_gradient_original(
        layers: &[CpuSvgSceneLayer],
    ) -> PreparedCpuSvgComposition {
        let before = layers.to_vec();
        let config = standalone_gradient_config();
        let prepared = prepare_cpu_svg_composition(layers, config)
            .expect("original standalone gradient scene");
        assert_eq!(
            normalized_paint_ids(&prepared.svg().svg),
            normalized_paint_ids(&original_composition(layers, config).svg)
        );
        assert_layers_unchanged(layers, &before);
        assert_layers_unchanged(prepared.layers(), &before);
        assert_eq!(
            prepared
                .configuration()
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits),
            config.viewport.transform.to_cols_array().map(f64::to_bits)
        );
        assert_eq!(
            prepared.configuration().viewport.resolution,
            config.viewport.resolution
        );
        assert_eq!(
            prepared.configuration().scale.to_bits(),
            config.scale.to_bits()
        );
        assert_eq!(prepared.configuration().max_svg_bytes, config.max_svg_bytes);
        assert_eq!(
            prepared.source_revision(),
            crate::resources::GRAPHITE_IMAGE_SOURCE_REVISION
        );
        prepared
    }

    #[test]
    fn at_r20_1_standalone_gradient_item_list_forms_spreads_and_settings_match_original() {
        for form in [GradientForm::Linear, GradientForm::Radial] {
            for spread in [
                GradientSpread::Pad,
                GradientSpread::Reflect,
                GradientSpread::Repeat,
                GradientSpread::Clear,
            ] {
                for interpolation in [
                    GradientInterpolation::Linear,
                    GradientInterpolation::Stepped,
                    GradientInterpolation::Smooth,
                ] {
                    for cyclic in [false, true] {
                        let item = Item::new_from_element(gradient_fixture())
                            .with_attribute(ATTR_GRADIENT_FORM, form)
                            .with_attribute(ATTR_GRADIENT_SPREAD, spread)
                            .with_attribute(ATTR_GRADIENT_INTERPOLATION, interpolation)
                            .with_attribute(ATTR_GRADIENT_CYCLIC, cyclic)
                            .with_attribute(
                                ATTR_TRANSFORM,
                                DAffine2::from_cols_array(&[3., 0.5, 1., 2., -2., 4.]),
                            )
                            .with_attribute(ATTR_BLEND_MODE, BlendMode::Multiply);
                        let standalone =
                            assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(
                                item.clone(),
                            )]);
                        let list =
                            assert_standalone_gradient_original(&[CpuSvgSceneLayer::GradientList(
                                List::from(item),
                            )]);
                        assert_eq!(
                            normalized_paint_ids(&standalone.svg().svg),
                            normalized_paint_ids(&list.svg().svg)
                        );
                        assert!(standalone.svg().svg.contains(match form {
                            GradientForm::Linear => "<linearGradient",
                            GradientForm::Radial => "<radialGradient",
                        }));
                        assert!(standalone.svg().svg.contains("points=\"10000000,10000000 -10000000,10000000 -10000000,-10000000 10000000,-10000000\""));
                        assert!(standalone.svg().svg.contains("gradientTransform="));
                        assert!(standalone.svg().svg.contains("mix-blend-mode: multiply;"));
                        match spread {
                            GradientSpread::Reflect | GradientSpread::Repeat => assert!(
                                standalone
                                    .svg()
                                    .svg
                                    .contains(&format!("spreadMethod=\"{}\"", spread.svg_name()))
                            ),
                            GradientSpread::Pad | GradientSpread::Clear => {
                                assert!(!standalone.svg().svg.contains("spreadMethod="))
                            }
                        }
                    }
                }
            }
        }
        for space in [
            GradientSpace::OkLab,
            GradientSpace::Lab,
            GradientSpace::OkLCh,
            GradientSpace::LCh,
            GradientSpace::RgbLinear,
            GradientSpace::RgbGamma,
            GradientSpace::Hsv,
            GradientSpace::Hsl,
        ] {
            for direction in [
                GradientHueDirection::Shorter,
                GradientHueDirection::Longer,
                GradientHueDirection::Increasing,
                GradientHueDirection::Decreasing,
            ] {
                let item = Item::new_from_element(gradient_fixture())
                    .with_attribute(ATTR_GRADIENT_SPACE, space)
                    .with_attribute(ATTR_GRADIENT_HUE_DIRECTION, direction);
                assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(item)]);
            }
        }
        let raw = Item::new_from_element(Gradient::black_to_white());
        let explicit = raw
            .clone()
            .with_attribute(ATTR_GRADIENT_FORM, GradientForm::Linear)
            .with_attribute(ATTR_GRADIENT_SPREAD, GradientSpread::Pad)
            .with_attribute(ATTR_GRADIENT_CYCLIC, false)
            .with_attribute(ATTR_GRADIENT_SPACE, GradientSpace::OkLab)
            .with_attribute(ATTR_GRADIENT_HUE_DIRECTION, GradientHueDirection::Shorter)
            .with_attribute(ATTR_GRADIENT_INTERPOLATION, GradientInterpolation::Linear);
        assert_eq!(
            normalized_paint_ids(
                &assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(raw.clone())])
                    .svg()
                    .svg
            ),
            normalized_paint_ids(
                &assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(explicit)])
                    .svg()
                    .svg
            )
        );
        let implicit_stops = Item::new_from_element(Gradient(List::from_iter([
            Item::new_from_element(Color::BLACK),
            Item::new_from_element(Color::WHITE),
        ])));
        assert_eq!(
            normalized_paint_ids(
                &assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(raw.clone())])
                    .svg()
                    .svg
            ),
            normalized_paint_ids(
                &assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(implicit_stops)])
                    .svg()
                    .svg
            )
        );
        assert!(raw.attributes().keys().next().is_none());
    }

    #[test]
    fn at_r20_1_standalone_gradient_raw_alpha_and_empty_samples_use_source_emission() {
        use graphite_vector_types::GradientStop;
        let alpha = 0.12345679_f32;
        let gradient = Gradient::new([
            GradientStop {
                position: 0.,
                midpoint: 0.5,
                color: Color::from_gamma_srgb_channels(1., 0., 0., alpha),
            },
            GradientStop {
                position: 1.,
                midpoint: 0.5,
                color: Color::from_gamma_srgb_channels(0., 0., 1., alpha),
            },
        ]);
        let item = Item::new_from_element(gradient)
            .with_attribute(ATTR_GRADIENT_SPACE, GradientSpace::RgbGamma)
            .with_attribute(ATTR_OPACITY, 0.5_f64)
            .with_attribute(ATTR_OPACITY_FILL, 0.25_f64);
        let prepared =
            assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(item.clone())]);
        assert!(
            prepared
                .svg()
                .svg
                .contains(&format!("stop-opacity=\"{alpha}\""))
        );
        assert!(prepared.svg().svg.contains("opacity=\"0.125\""));
        let samples = item
            .element()
            .interpolated_samples(GradientSettings::from_item_attributes(&item));
        assert!(!samples.is_empty());
        for (position, _, midpoint) in samples {
            assert!(
                prepared
                    .svg()
                    .svg
                    .contains(&format!("offset=\"{position}\""))
            );
            assert!(midpoint.is_none());
        }
        let masked = [
            CpuSvgSceneLayer::Gradient(item),
            CpuSvgSceneLayer::Color(
                Item::new_from_element(Color::WHITE).with_attribute(ATTR_CLIPPING_MASK, true),
            ),
        ];
        let prepared = assert_standalone_gradient_original(&masked);
        assert!(prepared.svg().svg.contains("<mask "));
        assert!(prepared.svg().svg.contains("opacity=\"0.5\""));
        assert!(
            prepared
                .svg()
                .svg
                .contains(&format!("stop-opacity=\"{alpha}\""))
        );
        for form in [GradientForm::Linear, GradientForm::Radial] {
            let empty = Item::new_from_element(Gradient::default())
                .with_attribute(ATTR_GRADIENT_FORM, form);
            let ordinary =
                assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(empty.clone())]);
            assert!(ordinary.svg().svg.contains("Gradient"));
            assert!(ordinary.svg().svg.contains("<polyline"));
            assert!(!ordinary.svg().svg.contains("<stop "));
            let clear = assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(
                empty.with_attribute(ATTR_GRADIENT_SPREAD, GradientSpread::Clear),
            )]);
            assert_eq!(
                clear.svg().svg.matches("<stop ").count(),
                if form == GradientForm::Linear { 4 } else { 3 }
            );
            assert!(clear.svg().svg.contains("stop-color=\"#000000\""));
            assert!(clear.svg().svg.contains("stop-opacity=\"0\""));
        }
        let zero_rows =
            assert_standalone_gradient_original(&[CpuSvgSceneLayer::GradientList(List::new())]);
        assert!(!zero_rows.svg().svg.contains("Gradient"));
        assert!(!zero_rows.svg().svg.contains("<polyline"));
    }

    #[test]
    fn at_r20_1_standalone_gradient_rows_order_and_all_row_clipping_preserve_mask_runs() {
        let first = Item::new_from_element(Gradient::black_to_white()).with_attribute(
            ATTR_TRANSFORM,
            DAffine2::from_translation(DVec2::new(2., 3.)),
        );
        let second = Item::new_from_element(gradient_fixture())
            .with_attribute(ATTR_GRADIENT_FORM, GradientForm::Radial)
            .with_attribute(ATTR_OPACITY_FILL, 0.5_f64)
            .with_attribute(ATTR_CLIPPING_MASK, true);
        let first_before = first.clone();
        let mut rows = List::from(first.clone());
        rows.push(second.clone());
        let rows_before = rows.clone();
        assert!(!Graphic::from(rows.clone()).had_clip_enabled());
        let prepared =
            assert_standalone_gradient_original(&[CpuSvgSceneLayer::GradientList(rows.clone())]);
        let first_tag = prepared
            .svg()
            .svg
            .find("<linearGradient")
            .expect("first original row");
        let second_tag = prepared
            .svg()
            .svg
            .find("<radialGradient")
            .expect("second original row");
        assert!(first_tag < second_tag);
        assert_eq!(first, first_before);
        assert_eq!(rows, rows_before);
        let mut reverse = List::from(second);
        reverse.push(first.clone());
        let reversed =
            assert_standalone_gradient_original(&[CpuSvgSceneLayer::GradientList(reverse)]);
        assert!(
            reversed
                .svg()
                .svg
                .find("<radialGradient")
                .expect("reversed first")
                < reversed
                    .svg()
                    .svg
                    .find("<linearGradient")
                    .expect("reversed second")
        );
        rows.set_attribute(ATTR_CLIPPING_MASK, 0, true);
        assert!(Graphic::from(rows.clone()).had_clip_enabled());
        let clip = Item::new_from_element(Color::WHITE).with_attribute(ATTR_CLIPPING_MASK, true);
        let layers = [
            CpuSvgSceneLayer::Gradient(first.clone()),
            CpuSvgSceneLayer::GradientList(rows.clone()),
            CpuSvgSceneLayer::Gradient(first),
            CpuSvgSceneLayer::Color(clip),
        ];
        let prepared = assert_standalone_gradient_original(&layers);
        assert_eq!(prepared.svg().svg.matches("<mask ").count(), 2);
        assert_eq!(prepared.svg().svg.matches("<clipPath ").count(), 0);
        assert_eq!(prepared.svg().svg.matches("mask=\"url(#mask-").count(), 2);
        let mut partially_clipped = rows;
        partially_clipped.set_attribute(ATTR_CLIPPING_MASK, 1, false);
        let interrupted = [
            layers[0].clone(),
            CpuSvgSceneLayer::GradientList(partially_clipped),
            layers[2].clone(),
            layers[3].clone(),
        ];
        assert_eq!(
            assert_standalone_gradient_original(&interrupted)
                .svg()
                .svg
                .matches("<mask ")
                .count(),
            1
        );
    }

    #[test]
    fn at_r20_1_standalone_gradient_rejects_typed_later_row_and_mask_overflow_atomically() {
        let valid = Item::new_from_element(Gradient::black_to_white());
        let mut wrong_stop = Gradient::black_to_white();
        wrong_stop.0.set_attribute(ATTR_POSITION, 0, true);
        let mut infinite_stop = Gradient::black_to_white();
        infinite_stop
            .0
            .set_attribute(ATTR_MIDPOINT, 0, f64::INFINITY);
        for (item, expected) in [
            (
                valid.clone().with_attribute(ATTR_GRADIENT_FORM, false),
                CpuSvgSceneError::WrongAttributeType(ATTR_GRADIENT_FORM.into()),
            ),
            (
                valid.clone().with_attribute(ATTR_CLIPPING_MASK, 1_f64),
                CpuSvgSceneError::WrongAttributeType(ATTR_CLIPPING_MASK.into()),
            ),
            (
                valid.clone().with_attribute("unknown-gradient", true),
                CpuSvgSceneError::UnsupportedAttribute("unknown-gradient".into()),
            ),
            (
                Item::new_from_element(wrong_stop),
                CpuSvgSceneError::WrongAttributeType(ATTR_POSITION.into()),
            ),
            (
                Item::new_from_element(infinite_stop),
                CpuSvgSceneError::InvalidGradient,
            ),
            (
                valid.clone().with_attribute(ATTR_OPACITY, f64::INFINITY),
                CpuSvgSceneError::NonFiniteOpacity(ATTR_OPACITY.into()),
            ),
            (
                valid
                    .clone()
                    .with_attribute(ATTR_BLEND_MODE, BlendMode::Erase),
                CpuSvgSceneError::UnsupportedBlendMode(BlendMode::Erase),
            ),
            (
                valid
                    .clone()
                    .with_attribute(ATTR_TRANSFORM, DAffine2::from_cols_array(&[f64::MAX; 6])),
                CpuSvgSceneError::InvalidGradient,
            ),
        ] {
            let mut rows = List::from(valid.clone());
            rows.push(item.clone());
            for layer in [
                CpuSvgSceneLayer::Gradient(item),
                CpuSvgSceneLayer::GradientList(rows),
            ] {
                let layers = [layer];
                let before = layers.clone();
                assert_eq!(
                    prepare_cpu_svg_composition(&layers, standalone_gradient_config()).err(),
                    Some(expected.clone())
                );
                assert_layers_unchanged(&layers, &before);
            }
        }
        let finite = valid
            .clone()
            .with_attribute(ATTR_OPACITY, -1e100_f64)
            .with_attribute(ATTR_OPACITY_FILL, 1e-100_f64);
        let rows = List::from_iter([valid.clone(), finite]);
        assert_standalone_gradient_original(&[CpuSvgSceneLayer::GradientList(rows.clone())]);
        let masked = [
            CpuSvgSceneLayer::GradientList(rows),
            CpuSvgSceneLayer::Color(
                Item::new_from_element(Color::WHITE).with_attribute(ATTR_CLIPPING_MASK, true),
            ),
        ];
        let before = masked.clone();
        assert!(
            original_composition(&masked, standalone_gradient_config())
                .svg
                .contains("opacity=\"-inf\"")
        );
        assert_eq!(
            prepare_cpu_svg_composition(&masked, standalone_gradient_config()).err(),
            Some(CpuSvgSceneError::NonFiniteMaskOpacity)
        );
        assert_layers_unchanged(&masked, &before);
        let mut empty = List::<Gradient>::new();
        assert_eq!(
            empty
                .iter_attribute_values_mut_or_default::<f64>(ATTR_OPACITY)
                .count(),
            0
        );
        assert_standalone_gradient_original(&[CpuSvgSceneLayer::GradientList(empty)]);
        let mut wrong_empty = List::<Gradient>::new();
        assert_eq!(
            wrong_empty
                .iter_attribute_values_mut_or_default::<bool>(ATTR_OPACITY)
                .count(),
            0
        );
        assert_eq!(
            prepare_cpu_svg_composition(
                &[CpuSvgSceneLayer::GradientList(wrong_empty)],
                standalone_gradient_config()
            )
            .err(),
            Some(CpuSvgSceneError::WrongAttributeType(ATTR_OPACITY.into()))
        );
        assert_standalone_gradient_original(&[CpuSvgSceneLayer::Gradient(valid)]);
    }

    #[test]
    fn at_r20_1_standalone_gradient_finite_singular_placement_cap_and_artboard_recover() {
        for transform in [
            DAffine2::ZERO,
            DAffine2::from_cols_array(&[2., 0.5, 1., 3., -4., 7.]),
        ] {
            let layers = [CpuSvgSceneLayer::Gradient(
                Item::new_from_element(gradient_fixture())
                    .with_attribute(ATTR_TRANSFORM, transform),
            )];
            let before = layers.clone();
            let prepared = assert_standalone_gradient_original(&layers);
            let mut cap = standalone_gradient_config();
            cap.max_svg_bytes = 0;
            assert!(matches!(
                prepare_cpu_svg_composition(&layers, cap),
                Err(CpuSvgSceneError::SvgBudgetExceeded { maximum: 0, .. })
            ));
            assert_layers_unchanged(&layers, &before);
            assert_eq!(
                normalized_paint_ids(&prepared.svg().svg),
                normalized_paint_ids(&assert_standalone_gradient_original(&layers).svg().svg)
            );
            let artboard = CpuSvgArtboardConfiguration {
                location: DVec2::new(3., 5.),
                dimensions: DVec2::new(20., 10.),
                background: Color::TRANSPARENT,
                clip: true,
            };
            let board = prepare_cpu_svg_artboard(&layers, artboard, standalone_gradient_config())
                .expect("original bounded Artboard and gradient");
            assert_eq!(
                normalized_paint_ids(&board.svg().svg),
                normalized_paint_ids(
                    &original_artboard(&layers, artboard, standalone_gradient_config()).svg
                )
            );
            assert_layers_unchanged(board.layers(), &before);
        }
    }

    #[test]
    fn at_r20_1_standalone_radial_negative_and_sheared_extrema_reject_before_rendering() {
        let maximum = f64::MAX * 0.75;
        let mut config = standalone_gradient_config();
        config.scale = 1.;
        config.viewport.transform = DAffine2::IDENTITY;
        for transform in [
            DAffine2::from_cols_array(&[-maximum, 0., 0., 1., maximum, 0.]),
            DAffine2::from_cols_array(&[maximum, 0., maximum, 1., 0., 0.]),
        ] {
            assert!(transform.is_finite());
            assert!(
                [DVec2::ZERO, DVec2::X, DVec2::Y]
                    .into_iter()
                    .all(|point| transform.transform_point2(point).is_finite())
            );
            assert!(
                [
                    DVec2::ONE,
                    -DVec2::ONE,
                    DVec2::new(1., -1.),
                    DVec2::new(-1., 1.)
                ]
                .into_iter()
                .any(|point| !transform.transform_point2(point).is_finite())
            );
            let item = Item::new_from_element(Gradient::black_to_white())
                .with_attribute(ATTR_GRADIENT_FORM, GradientForm::Radial)
                .with_attribute(ATTR_TRANSFORM, transform);
            let mut rows = List::from(Item::new_from_element(Gradient::black_to_white()));
            rows.push(item.clone());
            for layer in [
                CpuSvgSceneLayer::Gradient(item.clone()),
                CpuSvgSceneLayer::GradientList(rows),
            ] {
                let layers = [layer];
                let before = layers.clone();
                assert_eq!(
                    prepare_cpu_svg_composition(&layers, config).err(),
                    Some(CpuSvgSceneError::InvalidGradient)
                );
                assert_layers_unchanged(&layers, &before);
            }
            let linear = [CpuSvgSceneLayer::Gradient(
                item.with_attribute(ATTR_GRADIENT_FORM, GradientForm::Linear),
            )];
            let before = linear.clone();
            let prepared = prepare_cpu_svg_composition(&linear, config)
                .expect("original finite positive linear domain remains admitted");
            assert_eq!(
                normalized_paint_ids(&prepared.svg().svg),
                normalized_paint_ids(&original_composition(&linear, config).svg)
            );
            assert_layers_unchanged(prepared.layers(), &before);
        }
        let recovery = [CpuSvgSceneLayer::Gradient(
            Item::new_from_element(Gradient::black_to_white())
                .with_attribute(ATTR_GRADIENT_FORM, GradientForm::Radial)
                .with_attribute(
                    ATTR_TRANSFORM,
                    DAffine2::from_cols_array(&[3., 0.5, -1., 2., 4., -3.]),
                ),
        )];
        let prepared =
            prepare_cpu_svg_composition(&recovery, config).expect("finite sheared radial recovery");
        assert_eq!(
            normalized_paint_ids(&prepared.svg().svg),
            normalized_paint_ids(&original_composition(&recovery, config).svg)
        );
        assert_layers_unchanged(prepared.layers(), &recovery);
    }

    #[test]
    fn at_r20_1_standalone_single_stop_retains_original_default_midpoint_early_return() {
        use graphite_vector_types::GradientStop;
        let gradient = Gradient::new([GradientStop {
            position: 0.3,
            midpoint: 0.5,
            color: Color::from_gamma_srgb_channels(0.2, 0.7, 0.4, 0.375),
        }]);
        let item = Item::new_from_element(gradient);
        let samples = item
            .element()
            .interpolated_samples(GradientSettings::from_item_attributes(&item));
        assert_eq!(samples.len(), 1);
        assert_eq!(samples.first().expect("original single-stop sample").0, 0.3);
        assert_eq!(
            samples.first().expect("original single-stop midpoint").2,
            Some(0.5)
        );
        for form in [GradientForm::Linear, GradientForm::Radial] {
            let item = item.clone().with_attribute(ATTR_GRADIENT_FORM, form);
            for layer in [
                CpuSvgSceneLayer::Gradient(item.clone()),
                CpuSvgSceneLayer::GradientList(List::from(item.clone())),
            ] {
                let prepared = assert_standalone_gradient_original(&[layer]);
                assert_eq!(prepared.svg().svg.matches("<stop ").count(), 1);
                assert!(prepared.svg().svg.contains("offset=\"0.3\""));
                assert!(prepared.svg().svg.contains("graphite:midpoint=\"0.5\""));
                assert!(prepared.svg().svg.contains("stop-opacity=\"0.375\""));
            }
        }
    }
}
