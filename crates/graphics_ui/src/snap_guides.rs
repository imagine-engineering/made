use std::collections::VecDeque;

use anyhow::{Context as _, Result, bail, ensure};
use gpui::{FillRule, Pixels, Point, Rgba, Window};
use graphite_editor::messages::{
    portfolio::document::{
        DocumentMessageHandler,
        overlays::{OverlaysMessage, OverlaysMessageContext, OverlaysMessageHandler},
    },
    prelude::{
        FrontendMessage, InputPreprocessorMessageHandler, Message, MessageHandler, ViewportMessage,
        ViewportMessageHandler,
    },
    tool::common_functionality::snapping::{SnapData, SnapManager},
};
use graphite_graph_craft::document::value::DVec2;
use graphite_std::vector_types::kurbo::{Affine, BezPath, Cap, Join, Stroke, StrokeOpts, stroke};
use vello_encoding::{DrawTag, Encoding, PathSegmentType, PathTag, Style, math::f16_to_f32};

use crate::canvas::filled_path_builder;

const MAX_GUIDE_PATHS: usize = 2048;
const MAX_GUIDE_SEGMENTS: usize = 65_536;

#[derive(PartialEq)]
struct SnapGuidePath {
    path: BezPath,
    color: Rgba,
    fill_rule: FillRule,
}

#[derive(PartialEq)]
pub(crate) struct SnapGuides(Vec<SnapGuidePath>);

impl SnapGuides {
    pub fn for_gesture(
        manager: &mut SnapManager,
        document: &DocumentMessageHandler,
        viewport_size: DVec2,
        scale: f64,
    ) -> Result<Option<Self>> {
        if manager.indicator_pos().is_none() || !document.overlays_visibility_settings.all() {
            return Ok(None);
        }
        ensure!(
            scale.is_finite()
                && scale > 0.
                && viewport_size.is_finite()
                && viewport_size.min_element() > 0.
        );
        let mut viewport = ViewportMessageHandler::default();
        let mut responses = VecDeque::new();
        viewport.process_message(
            ViewportMessage::Update {
                x: 0.,
                y: 0.,
                width: viewport_size.x,
                height: viewport_size.y,
                scale,
            },
            &mut responses,
            (),
        );
        responses.clear();
        OverlaysMessageHandler::default().process_message(
            OverlaysMessage::Draw,
            &mut responses,
            OverlaysMessageContext {
                viewport: &viewport,
                visibility_settings: document.overlays_visibility_settings,
            },
        );
        let mut context = responses
            .into_iter()
            .find_map(|message| match message {
                Message::Frontend(FrontendMessage::RenderOverlays { context }) => Some(context),
                _ => None,
            })
            .context("Missing Graphite overlay context")?;
        let input = InputPreprocessorMessageHandler::default();
        manager.draw_overlays(SnapData::new(document, &input, &viewport), &mut context);
        let scene = context.take_scene();
        let guides = Self::decode(scene.encoding(), scale)?;
        Ok((!guides.0.is_empty()).then_some(guides))
    }

    fn decode(encoding: &Encoding, scale: f64) -> Result<Self> {
        ensure!(
            encoding.path_tags.len() <= MAX_GUIDE_SEGMENTS + 3 * MAX_GUIDE_PATHS
                && encoding.path_data.len() <= MAX_GUIDE_SEGMENTS * 8
                && encoding.draw_tags.len() <= MAX_GUIDE_PATHS,
            "Graphite snap guide geometry exceeds its viewport budget",
        );
        let mut coordinates = encoding.path_data.iter().copied();
        let mut colors = encoding.draw_data.iter().zip(&encoding.draw_tags);
        let mut transforms = encoding.transforms.iter();
        let mut styles = encoding.styles.iter();
        let mut transform = None;
        let mut style = None;
        let mut subpath_open = false;
        let mut path = BezPath::new();
        let mut guides = Vec::new();
        for tag in encoding.path_tags.iter().copied() {
            match tag {
                PathTag::TRANSFORM => {
                    ensure!(path.is_empty(), "Transform inside a Graphite snap path");
                    transform = Some(transforms.next().context("Missing snap path transform")?);
                }
                PathTag::STYLE => {
                    ensure!(path.is_empty(), "Style inside a Graphite snap path");
                    style = Some(*styles.next().context("Missing snap path style")?);
                }
                PathTag::PATH => {
                    ensure!(
                        !subpath_open && !path.is_empty(),
                        "Incomplete Graphite snap path"
                    );
                    let (&color, &draw) = colors.next().context("Missing snap path paint")?;
                    ensure!(
                        draw == DrawTag::COLOR,
                        "Unsupported Graphite snap path paint"
                    );
                    let transform = transform.context("Missing snap path transform")?.to_kurbo();
                    let style = style.context("Missing snap path style")?;
                    let (outline, fill_rule) = paint_shape(std::mem::take(&mut path), style)?;
                    let path = (Affine::scale(scale.recip()) * transform) * outline;
                    ensure!(path.is_finite(), "Nonfinite Graphite snap path");
                    guides.push(SnapGuidePath {
                        path,
                        color: solid_color(color),
                        fill_rule,
                    });
                }
                tag if tag.is_path_segment() && tag.is_f32() => {
                    if !subpath_open {
                        path.move_to(read_point(&mut coordinates)?);
                        subpath_open = true;
                    }
                    let segment = tag.path_segment_type();
                    let style = style.context("Missing snap segment style")?;
                    let stroke_marker = is_stroke(style) && tag.is_subpath_end();
                    let first = read_point(&mut coordinates)?;
                    // Vello ends stroked subpaths with a synthetic tangent segment for its GPU stroker.
                    match (segment, stroke_marker) {
                        (PathSegmentType::LINE_TO, true) => path.close_path(),
                        (PathSegmentType::QUAD_TO, true) => {
                            read_point(&mut coordinates)?;
                        }
                        (PathSegmentType::LINE_TO, false) => path.line_to(first),
                        (PathSegmentType::QUAD_TO, false) => {
                            path.quad_to(first, read_point(&mut coordinates)?);
                        }
                        (PathSegmentType::CUBIC_TO, false) => {
                            path.curve_to(
                                first,
                                read_point(&mut coordinates)?,
                                read_point(&mut coordinates)?,
                            );
                        }
                        _ => bail!("Unsupported Graphite snap path segment"),
                    }
                    if tag.is_subpath_end() {
                        if !stroke_marker {
                            path.close_path();
                        }
                        subpath_open = false;
                    }
                }
                _ => bail!("Unsupported Graphite snap path tag"),
            }
        }
        ensure!(
            path.is_empty()
                && coordinates.next().is_none()
                && colors.next().is_none()
                && encoding.draw_tags.len() == encoding.draw_data.len()
                && transforms.next().is_none()
                && styles.next().is_none(),
            "Trailing Graphite snap geometry"
        );
        Ok(Self(guides))
    }

    pub fn paint(&self, origin: Point<Pixels>, window: &mut Window) {
        for guide in &self.0 {
            if let Some(path) = filled_path_builder(&guide.path, origin, guide.fill_rule) {
                match path.build() {
                    Ok(path) => window.paint_path(path, guide.color),
                    Err(error) => log::debug!("Unable to paint Graphite snap guide: {error:#}"),
                }
            }
        }
    }
}

fn read_point(coordinates: &mut impl Iterator<Item = u32>) -> Result<(f64, f64)> {
    let mut next = || {
        coordinates
            .next()
            .map(f32::from_bits)
            .context("Missing snap path coordinate")
    };
    let (x, y) = (next()?, next()?);
    ensure!(
        x.is_finite() && y.is_finite(),
        "Nonfinite snap path coordinate"
    );
    Ok((f64::from(x), f64::from(y)))
}

fn is_stroke(style: Style) -> bool {
    style.flags_and_miter_limit & Style::FLAGS_STYLE_BIT != 0
}

fn paint_shape(path: BezPath, style: Style) -> Result<(BezPath, FillRule)> {
    let flags = style.flags_and_miter_limit;
    if !is_stroke(style) {
        let rule = if flags & Style::FLAGS_FILL_BIT == 0 {
            FillRule::NonZero
        } else {
            FillRule::EvenOdd
        };
        return Ok((path, rule));
    }
    let join = match flags & Style::FLAGS_JOIN_MASK {
        Style::FLAGS_JOIN_BITS_BEVEL => Join::Bevel,
        Style::FLAGS_JOIN_BITS_MITER => Join::Miter,
        Style::FLAGS_JOIN_BITS_ROUND => Join::Round,
        _ => bail!("Unsupported Graphite snap path join"),
    };
    let cap = |bits| match bits {
        Style::FLAGS_CAP_BITS_BUTT => Ok(Cap::Butt),
        Style::FLAGS_CAP_BITS_SQUARE => Ok(Cap::Square),
        Style::FLAGS_CAP_BITS_ROUND => Ok(Cap::Round),
        _ => bail!("Unsupported Graphite snap path cap"),
    };
    let width = f64::from(style.line_width);
    let miter_limit = f64::from(f16_to_f32(flags as u16));
    ensure!(
        width.is_finite() && width > 0. && miter_limit.is_finite(),
        "Invalid Graphite snap path stroke"
    );
    let style = Stroke {
        width,
        join,
        miter_limit,
        start_cap: cap((flags & Style::FLAGS_START_CAP_MASK) >> 2)?,
        end_cap: cap(flags & Style::FLAGS_END_CAP_MASK)?,
        ..Stroke::default()
    };
    // Vello strokes in path space before applying the display transform.
    Ok((
        stroke(path, &style, &StrokeOpts::default(), 0.05),
        FillRule::NonZero,
    ))
}

fn solid_color(encoded: u32) -> Rgba {
    let [red, green, blue, alpha] = encoded.to_le_bytes();
    let component = |value: u8| {
        if alpha == 0 {
            0.
        } else {
            f32::from(value) / f32::from(alpha)
        }
    };
    Rgba {
        r: component(red),
        g: component(green),
        b: component(blue),
        a: f32::from(alpha) / 255.,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphite_editor::{
        consts::{COLOR_OVERLAY_BLUE, COLOR_OVERLAY_WHITE},
        messages::{
            portfolio::document::utility_types::misc::{
                ArtboardSnapSource, ArtboardSnapTarget, SnapSource, SnapTarget,
            },
            tool::common_functionality::snapping::SnappedPoint,
        },
    };
    use graphite_std::{
        renderer::{Quad, Rect},
        vector_types::kurbo::Shape,
    };

    #[test]
    fn at_gra_view_008_1_source_alignment_guides_use_physical_pixel_centers() {
        let viewport = DVec2::new(240., 160.);
        let document = DocumentMessageHandler::default();
        let mut manager = SnapManager::default();
        let first = DVec2::new(20.25, 40.25);
        let last = DVec2::new(150.25, 40.25);
        manager.update_indicator(SnappedPoint {
            snapped_point_document: first,
            alignment_target_horizontal: Some(last),
            ..Default::default()
        });
        let blue = Rgba::try_from(COLOR_OVERLAY_BLUE).expect("source overlay blue");
        for scale in [1., 1.5, 2.] {
            let guides = SnapGuides::for_gesture(&mut manager, &document, viewport, scale)
                .expect("admitted source paths")
                .expect("active alignment guides");
            assert_eq!(guides.0.len(), 5);
            let line = &guides.0[0];
            assert_eq!(line.color, blue);
            let bounds = line.path.bounding_box();
            let expected_y = ((first.y * scale).round() - 0.5) / scale;
            assert!(
                (bounds.center().y - expected_y).abs() < 0.001,
                "scale {scale}: {bounds:?}"
            );
            assert!((bounds.height() - 1.).abs() < 0.001);
            assert!(bounds.width() >= 129.5);
        }
    }

    #[test]
    fn at_gra_view_008_1_source_labels_outlines_and_distribution_are_painted() {
        let viewport = DVec2::new(480., 260.);
        let mut document = DocumentMessageHandler::default();
        let mut manager = SnapManager::default();
        let point = DVec2::new(60., 110.);
        let bounds = [DVec2::new(20., 80.), DVec2::new(60., 110.)];
        manager.update_indicator(SnappedPoint {
            snapped_point_document: point,
            source: SnapSource::Artboard(ArtboardSnapSource::CornerPoint),
            target: SnapTarget::Artboard(ArtboardSnapTarget::CornerPoint),
            target_bounds: Some(Quad::from_box(bounds)),
            ..Default::default()
        });
        let blue = Rgba::try_from(COLOR_OVERLAY_BLUE).expect("source overlay blue");
        let white = Rgba::try_from(COLOR_OVERLAY_WHITE).expect("source overlay white");
        let guides = SnapGuides::for_gesture(&mut manager, &document, viewport, 2.)
            .expect("admitted source paths")
            .expect("active labeled guide");
        assert!(
            guides
                .0
                .iter()
                .any(|guide| guide.color.a > 0.74 && guide.color.a < 0.76)
        );
        assert!(guides.0.iter().any(|guide| guide.color == white));
        assert!(guides.0.iter().any(|guide| {
            let geometry = guide.path.bounding_box();
            guide.color == blue && geometry.width() > 40. && geometry.height() > 30.
        }));
        for guide in &guides.0 {
            filled_path_builder(
                &guide.path,
                gpui::point(gpui::px(10.), gpui::px(15.)),
                guide.fill_rule,
            )
            .expect("finite native path")
            .build()
            .expect("tessellated source guide");
        }
        manager.update_indicator(SnappedPoint {
            snapped_point_document: point,
            distribution_boxes_horizontal: [
                Rect::from_box(bounds),
                Rect::from_box(bounds.map(|point| point + DVec2::new(65., 0.))),
            ]
            .into(),
            distribution_equal_distance_horizontal: Some(25.),
            ..Default::default()
        });
        let guides = SnapGuides::for_gesture(&mut manager, &document, viewport, 1.5)
            .expect("admitted source paths")
            .expect("active distribution guides");
        assert!(guides.0.len() >= 3);
        assert!(guides.0.iter().all(|guide| guide.color == blue));
        document.overlays_visibility_settings.all = false;
        assert!(
            SnapGuides::for_gesture(&mut manager, &document, viewport, 1.5)
                .expect("hidden overlays")
                .is_none()
        );
        document.overlays_visibility_settings.all = true;
        manager.clear_indicator();
        assert!(
            SnapGuides::for_gesture(&mut manager, &document, viewport, 1.5)
                .expect("cleared snap")
                .is_none()
        );
    }

    #[test]
    fn at_gra_view_008_1_rejects_over_budget_or_incomplete_snap_paths() {
        let mut encoding = Encoding::default();
        encoding.path_tags.push(PathTag::TRANSFORM);
        assert!(SnapGuides::decode(&encoding, 1.).is_err());
        encoding
            .path_tags
            .resize(MAX_GUIDE_SEGMENTS + MAX_GUIDE_PATHS * 3 + 1, PathTag::STYLE);
        assert!(SnapGuides::decode(&encoding, 1.).is_err());
    }
}
