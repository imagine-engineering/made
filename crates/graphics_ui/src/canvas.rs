use gpui::{
    Bounds, PathBuilder, PathStyle, Pixels, Point, Rgba, ScrollWheelEvent, Window, point, px,
};
use graphics_core::formats::gdd::GddArtboardCanvasOverlay;
use graphite_editor::{
    consts::{
        COLOR_OVERLAY_BLUE, COLOR_OVERLAY_GRAY, VIEWPORT_SCROLL_RATE,
        VIEWPORT_ZOOM_MIN_FRACTION_COVER, VIEWPORT_ZOOM_WHEEL_RATE,
    },
    messages::{
        input_mapper::utility_types::pointer::ScrollDelta,
        portfolio::document::utility_types::misc::PTZ, prelude::NavigationMessageHandler,
    },
};
use graphite_graph_craft::document::value::{DAffine2, DVec2};
use graphite_std::{
    ATTR_TRANSFORM,
    text::{FALLBACK_FONT_RESOURCE, TextAlign, TextContext, TypesettingConfig},
    vector_types::kurbo::{Affine, BezPath, ParamCurve, PathEl},
};

pub(crate) struct ArtboardOverlay {
    name_paths: Vec<BezPath>,
    outline: [DVec2; 4],
}

impl ArtboardOverlay {
    pub fn prepare(overlay: GddArtboardCanvasOverlay) -> Self {
        let typesetting = TypesettingConfig {
            font_size: 12.,
            line_height_ratio: 1.2,
            align: TextAlign::AlignLeft,
            ..Default::default()
        };
        let (size, text) = TextContext::with_thread_local(|context| {
            (
                context.bounding_box(&overlay.name, &FALLBACK_FONT_RESOURCE, typesetting, false),
                context.to_path(&overlay.name, &FALLBACK_FONT_RESOURCE, typesetting, false),
            )
        });
        let label_transform =
            overlay.name_transform * DAffine2::from_translation(-DVec2::Y * size.y);
        let mut name_paths = Vec::new();
        for index in 0..text.len() {
            let Some(element) = text.element(index) else {
                continue;
            };
            let glyph_transform: DAffine2 = text.attribute_cloned_or_default(ATTR_TRANSFORM, index);
            let transform = Affine::new((label_transform * glyph_transform).to_cols_array());
            let mut path = BezPath::new();
            let mut previous_end = None;
            for (_, segment, start, end) in element.segment_iter() {
                let segment = transform * segment;
                if previous_end != Some(start) {
                    path.move_to(segment.start());
                }
                path.push(segment.as_path_el());
                previous_end = Some(end);
            }
            if !path.is_empty() {
                name_paths.push(path);
            }
        }
        Self {
            name_paths,
            outline: overlay.outline,
        }
    }

    pub fn paint(&self, origin: Point<Pixels>, outline_visible: bool, window: &mut Window) {
        if let Ok(color) = Rgba::try_from(COLOR_OVERLAY_GRAY) {
            for path in &self.name_paths {
                if let Some(builder) = name_path_builder(path, origin) {
                    paint_overlay_path(builder, color, window);
                }
            }
        }
        if outline_visible
            && let Ok(color) = Rgba::try_from(COLOR_OVERLAY_BLUE)
            && let [Some(first), Some(second), Some(third), Some(fourth)] =
                self.outline.map(|corner| pane_point(corner, origin))
        {
            let mut outline = PathBuilder::stroke(px(1.));
            outline.move_to(first);
            outline.line_to(second);
            outline.line_to(third);
            outline.line_to(fourth);
            outline.close();
            paint_overlay_path(outline, color, window);
        }
    }
}

fn name_path_builder(path: &BezPath, origin: Point<Pixels>) -> Option<PathBuilder> {
    let mut builder = PathBuilder::fill().with_style(PathStyle::Fill(
        gpui::FillOptions::default().with_fill_rule(gpui::FillRule::NonZero),
    ));
    let convert = |position: graphite_std::vector_types::kurbo::Point| {
        pane_point(DVec2::new(position.x, position.y), origin)
    };
    for element in path.elements() {
        match *element {
            PathEl::MoveTo(position) => builder.move_to(convert(position)?),
            PathEl::LineTo(position) => builder.line_to(convert(position)?),
            PathEl::QuadTo(control, position) => {
                builder.curve_to(convert(position)?, convert(control)?)
            }
            PathEl::CurveTo(first, second, position) => {
                builder.cubic_bezier_to(convert(position)?, convert(first)?, convert(second)?)
            }
            PathEl::ClosePath => builder.close(),
        }
    }
    Some(builder)
}

fn pane_point(position: DVec2, origin: Point<Pixels>) -> Option<Point<Pixels>> {
    let [x, y] = position.as_vec2().to_array();
    (x.is_finite() && y.is_finite()).then(|| origin + point(px(x), px(y)))
}

fn paint_overlay_path(builder: PathBuilder, color: Rgba, window: &mut Window) {
    match builder.build() {
        Ok(path) => window.paint_path(path, color),
        Err(error) => log::debug!("Unable to paint Graphite Artboard overlay: {error:#}"),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ArtboardNavigation {
    pub camera: PTZ,
    pub artboard_size: DVec2,
}

impl ArtboardNavigation {
    pub fn document_to_pane(&self, viewport_size: DVec2) -> DAffine2 {
        NavigationMessageHandler::default()
            .calculate_offset_transform(viewport_size / 2., &self.camera)
    }

    pub fn pan(&self, viewport_size: DVec2, delta: Point<Pixels>) -> Option<PTZ> {
        let mut camera = self.camera;
        camera.pan += self.document_delta(viewport_size, logical_vector(delta));
        camera.pan.is_finite().then_some(camera)
    }

    fn document_delta(&self, viewport_size: DVec2, pane_delta: DVec2) -> DVec2 {
        self.document_to_pane(viewport_size)
            .inverse()
            .transform_vector2(pane_delta)
    }

    pub fn scroll(
        &self,
        viewport_size: DVec2,
        pane: Bounds<Pixels>,
        event: &ScrollWheelEvent,
        line_height: Pixels,
    ) -> Option<PTZ> {
        if !pane.contains(&event.position) {
            return None;
        }
        let delta = logical_vector(event.delta.pixel_delta(line_height));
        if !delta.is_finite() || delta == DVec2::ZERO {
            return None;
        }
        let mut camera = self.camera;
        let pane_delta =
            if event.modifiers.control || (cfg!(target_os = "macos") && event.modifiers.platform) {
                // The original wheel input uses DOM delta signs, opposite to GPUI scroll offsets.
                let original = ScrollDelta::new(-delta.x, -delta.y, 0.);
                let scroll = original.scroll_delta();
                if !scroll.is_finite() || scroll == 0. {
                    return None;
                }
                let factor = 1. + scroll.abs() * VIEWPORT_ZOOM_WHEEL_RATE;
                let original_zoom = camera.zoom();
                let zoom = if scroll < 0. {
                    original_zoom * factor
                } else {
                    original_zoom / factor
                };
                let coverage_per_zoom = (self.artboard_size / viewport_size).max_element();
                let minimum_zoom = VIEWPORT_ZOOM_MIN_FRACTION_COVER / coverage_per_zoom;
                camera.set_zoom(zoom.max(minimum_zoom));
                let anchor = logical_vector(event.position - pane.origin);
                // Anchor using the zoom actually applied by PTZ, including at a zoom limit.
                (viewport_size / 2. - anchor) * (1. - original_zoom / camera.zoom())
            } else {
                let delta = if event.modifiers.shift && !cfg!(target_os = "macos") {
                    DVec2::new(delta.y, 0.)
                } else {
                    delta
                };
                delta * VIEWPORT_SCROLL_RATE
            };
        camera.pan += self.document_delta(viewport_size, pane_delta);
        (camera != self.camera && camera.pan.is_finite()).then_some(camera)
    }
}

fn logical_vector(point: Point<Pixels>) -> DVec2 {
    DVec2::new(f64::from(f32::from(point.x)), f64::from(f32::from(point.y)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphite_std::vector_types::kurbo::Shape;

    #[test]
    fn at_gra_view_014_1_prepares_the_source_artboard_label_path() {
        let outline = [
            DVec2::new(25., 90.),
            DVec2::new(125., 90.),
            DVec2::new(125., 140.),
            DVec2::new(25., 140.),
        ];
        let overlay = ArtboardOverlay::prepare(GddArtboardCanvasOverlay {
            name: "Artboard Å\nNorth".to_owned(),
            outline,
            name_transform: DAffine2::from_translation(DVec2::new(25., 86.)),
        });
        assert_eq!(overlay.outline, outline);
        assert!(!overlay.name_paths.is_empty());
        for path in &overlay.name_paths {
            let path_bounds = path.bounding_box();
            assert!(path_bounds.min_x() >= 24.);
            assert!(path_bounds.max_y() <= 87.);
            assert!(path_bounds.height() > 0.);
            assert!(
                name_path_builder(path, point(px(12.), px(18.)))
                    .expect("finite source glyph coordinates")
                    .build()
                    .is_ok()
            );
        }
    }
}
